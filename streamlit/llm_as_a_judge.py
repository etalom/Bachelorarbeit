import json
import re
import argparse
import requests

OLLAMA_URL = "http://ollama:11434/api/generate"
DEFAULT_JUDGE_MODEL = "llama3.1:8b"

JUDGE_PROMPT_TEMPLATE = """Du bist ein Experte zur Bewertung von Antworten eines Vergaberechts-Chatbots.
Bewerte die folgende Antwort anhand dieser Kriterien auf einer Skala von 1 (schlecht) bis 5 (sehr gut):

- **correctness**: Ist die Antwort fachlich korrekt basierend auf den Quellen?
- **completeness**: Beantwortet die Antwort die Frage vollständig?
- **faithfulness**: Bleibt die Antwort treu zu den gegebenen Quellen (keine Halluzinationen)?
- **clarity**: Ist die Antwort klar, strukturiert und verständlich?

Antworte AUSSCHLIESSLICH mit einem gültigen JSON-Objekt in diesem Format:
{{"correctness": <1-5>, "completeness": <1-5>, "faithfulness": <1-5>, "clarity": <1-5>}}

--- FRAGE ---
{query}

--- QUELLEN (Auszüge) ---
{context}

--- ANTWORT DES CHATBOTS ---
{answer}
"""


def call_judge(query: str, answer: str, context: str, model: str) -> dict | None:
    prompt = JUDGE_PROMPT_TEMPLATE.format(query=query, answer=answer, context=context)

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "num_predict": 256,
                "temperature": 0.0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
    except Exception as e:
        print(f"  [FEHLER] Ollama-Call fehlgeschlagen: {e}")
        return None

    json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if not json_match:
        print(f"  [WARN] Kein JSON in Antwort gefunden:\n{text[:200]}")
        return None

    try:
        scores = json.loads(json_match.group())
        required = {"correctness", "completeness", "faithfulness", "clarity"}
        if not required.issubset(scores.keys()):
            print(f"  [WARN] JSON hat nicht alle erwarteten Keys: {scores.keys()}")
            return None
        for k in required:
            scores[k] = int(scores[k])
        return scores
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"  [WARN] JSON-Parse-Fehler: {e}")
        return None


def build_context_text(quellen: list[dict]) -> str:
    parts = []
    for i, q in enumerate(quellen, 1):
        file_name = q.get("datei", "unbekannt")
        seite = q.get("seite", "?")
        excerpt = q.get("abschnitt", "")
        parts.append(f"[{i}] {file_name} (S. {seite}):\n{excerpt[:500]}")
    return "\n\n".join(parts) if parts else "(keine Quellen angegeben)"


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge für Antwortqualität")
    parser.add_argument("--json", type=str, default="val_results_runB.json")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--save", type=str, default=None, help="Pfad für Ergebnisse-JSON")
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    print(f"📄 {len(entries)} Einträge geladen aus {args.json}")
    print(f"⚖️  Judge-Modell: {args.judge_model}\n")

    all_scores = []
    per_entry = []

    for idx, entry in enumerate(entries, 1):
        query = entry.get("query", "")
        answer = entry.get("answer", "")
        quellen = entry.get("quellen", [])
        context = build_context_text(quellen)

        print(f"[{idx}/{len(entries)}] Bewerte: {query[:60]}...")
        scores = call_judge(query, answer, context, args.judge_model)

        if scores:
            all_scores.append(scores)
            per_entry.append({"query": query, "scores": scores})
            line = "  "
            for k, v in scores.items():
                line += f"{k}={v} "
            print(line)
        else:
            per_entry.append({"query": query, "scores": None})
            print("  ❌ nicht bewertbar")
        print()

    if not all_scores:
        print("❌ Keine gültigen Bewertungen.")
        return

    n = len(all_scores)
    print("=" * 50)
    print("📊 ERGEBNISSE")
    print("=" * 50)
    for criterion in ["correctness", "completeness", "faithfulness", "clarity"]:
        vals = [s[criterion] for s in all_scores]
        avg = sum(vals) / n
        print(f"  {criterion:15s}: {avg:.2f} / 5.0  (min={min(vals)}, max={max(vals)})")

    overall = sum(sum(s.values()) for s in all_scores) / (n * 4)
    print(f"\n  {'OVERALL':15s}: {overall:.2f} / 5.0")

    if args.save:
        result = {
            "source_file": args.json,
            "judge_model": args.judge_model,
            "entries": per_entry,
            "aggregates": {
                c: round(sum(s[c] for s in all_scores) / n, 2)
                for c in ["correctness", "completeness", "faithfulness", "clarity"]
            },
            "overall": round(overall, 2),
        }
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Ergebnisse gespeichert: {args.save}")


if __name__ == "__main__":
    main()
