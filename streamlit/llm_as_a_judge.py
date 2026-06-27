import json
import re
import argparse
import requests
import os

CANDIDATES = [
    os.getenv("OLLAMA_BASE_URL", ""),
    "http://localhost:11434",
    "http://ollama:11434",
    "http://host.docker.internal:11434",
]

def resolve_ollama_url():
    for url in CANDIDATES:
        if not url:
            continue
        try:
            requests.get(f"{url}/api/tags", timeout=2)
            print(f"Ollama gefunden: {url}")
            return url
        except:
            continue
    print("Ollama nicht erreichbar, fallback localhost")
    return "http://localhost:11434"

OLLAMA_URL = resolve_ollama_url() + "/api/generate"
DEFAULT_JUDGE_MODEL = "llama3.1:latest"

CRITERIA = {
    "correctness": {
        "desc": "Beantwortet die Antwort die Frage sinnvoll? Falls die Antwort sagt, dass die Info nicht in den Dokumenten steht (egal welche Formulierung), ist das schlecht.",
        "rubric": "Score 1: Die Antwort beantwortet die Frage nicht oder sagt nur 'nicht gefunden'.\nScore 2: Die Antwort enthält kaum relevante Information.\nScore 3: Die Antwort beantwortet die Frage teilweise.\nScore 4: Die Antwort beantwortet die Frage größtenteils.\nScore 5: Die Antwort beantwortet die Frage vollständig und präzise."
    },
    "completeness": {
        "desc": "Wie vollständig ist die Antwort? Wenn die Antwort sagt 'nicht gefunden': 1.",
        "rubric": "Score 1: Keine relevanten Informationen geliefert ('nicht gefunden').\nScore 2: Nur sehr wenige Aspekte abgedeckt.\nScore 3: Einige Aspekte abgedeckt.\nScore 4: Die meisten Aspekte abgedeckt.\nScore 5: Alle relevanten Aspekte vollständig abgedeckt."
    },
    "faithfulness": {
        "desc": "Bleibt die Antwort bei den Quellen (keine Halluzination)? Antwortet der Chatbot mit 'nicht gefunden': 5.",
        "rubric": "Score 1: Die Antwort halluziniert Information, die nicht in den Quellen steht.\nScore 2: Große Teile der Antwort sind nicht durch Quellen gedeckt.\nScore 3: Teile der Antwort sind nicht durch Quellen gedeckt.\nScore 4: Die Antwort ist größtenteils quellenbasiert.\nScore 5: Die Antwort ist vollständig durch die Quellen gedeckt oder sagt korrekt 'nicht gefunden'."
    },
    "clarity": {
        "desc": "Ist die Antwort klar und verständlich?",
        "rubric": "Score 1: Unverständlich oder chaotisch.\nScore 2: Schwer verständlich.\nScore 3: Verständlich mit kleinen Mängeln.\nScore 4: Klar und gut strukturiert.\nScore 5: Sehr klar, präzise und gut formuliert."
    }
}

def call_prometheus(query, answer, context, criterion_key, model):
    c = CRITERIA[criterion_key]
    prompt = f"""##Description:
Du bewertest einen RAG-Chatbot für Vergaberecht. Kriterium: {c["desc"]}

##Task:
Frage: {query}

Quellen: {context}

Antwort des Chatbots: {answer}

Bewerte die Antwort NUR nach dem Kriterium "{criterion_key}".

##Score Rubric:
{c["rubric"]}

##Reference Answer:
Eine gute Antwort basiert auf den Quellen und beantwortet die Frage präzise.

##Response to Evaluate:
{answer}
"""
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model, "prompt": prompt, "stream": False,
            "num_predict": 256, "temperature": 0.0,
        }, timeout=120)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
    except Exception as e:
        print(f"  [FEHLER] {criterion_key}: {e}")
        return 1

    m = re.search(r"Score\s*(\d)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    m = re.search(r"(\d)\s*=?\s*(?:Score|von|Punkt)", text)
    if m:
        return int(m.group(1))

    m = re.search(r"\b([1-5])\b", text)
    if m:
        return int(m.group(1))

    print(f"  [WARN] Kein Score in: {text[:150]}")
    return 1

def build_context_text(quellen):
    parts = []
    for i, q in enumerate(quellen, 1):
        parts.append(f"[{i}] {q.get('datei','?')} (S. {q.get('seite','?')}):\n{q.get('abschnitt','')[:500]}")
    return "\n\n".join(parts) if parts else "(keine Quellen)"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--save", default="judge_results.json")
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entries = raw.get("entries", raw if isinstance(raw, list) else [])
    n = len(entries)

    results = []
    for i, e in enumerate(entries, 1):
        query = e.get("query", "")
        answer = e.get("answer", "")
        context = build_context_text(e.get("quellen", []))
        print(f"[{i}/{n}] {query[:60]}...")
        scores = {}
        for key in ["correctness", "completeness", "faithfulness", "clarity"]:
            score = call_prometheus(query, answer, context, key, args.judge_model)
            scores[key] = score
            print(f"  {key}={score}", end="")
        print()
        results.append(scores)

    if results:
        print("\nDurchschnitt (alle {} Fragen):".format(n))
        for k in ["correctness", "completeness", "faithfulness", "clarity"]:
            avg = sum(r[k] for r in results) / len(results)
            print(f"  {k}: {avg:.2f}")

    with open(args.save, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nGespeichert: {args.save}")

if __name__ == "__main__":
    main()
