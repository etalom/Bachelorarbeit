import json
import requests
import time
import subprocess
import sys
import os

API_URL = "http://localhost:8000/api/ask"
DEFAULT_MODEL = "gpt-oss:20b"

QUESTIONS_IN_CONTEXT = [
    "Ab welchem Betrag ist eine Vergabe durchzuführen (UVgO, VV ÖA, LHO)?",
    "Ab welchem Betrag ist eine Beschaffung durchzuführen (UVgO, VV ÖA, LHO)?",
    "Wie lange dauert ein Vergabeverfahren (UVgO, VgV)?",
    "Ist der Höchstbetrag in Brutto oder Netto anzugeben (UVgO, VgV, VV ÖA)?",
    "Warum ist ein Vergabeverfahren durchzuführen (LHO, UVgO, VgV, VV ÖA)?",
    "Welcher Preis ist zu beachten (VV ÖA)?",
    "Als Mitarbeitender oder Professor*in möchte ich wissen, ob Beträge, Preise und Kostenschätzungen in Brutto oder Netto anzugeben sind, um das korrekte Verfahren bestimmen zu können.",
    "Als Institutssekretariat möchte ich wissen, welche Dokumente der Beschaffungsstelle vorgelegt werden müssen, um eine Bestellung rechtssicher einzuleiten.",
    "Als wissenschaftlich Mitarbeitender möchte ich wissen, wie lange eine Ausschreibung offen sein muss und welche Fristen für Angebotsabgabe und Zuschlag gelten.",
    "Als wissenschaftlich Mitarbeitende möchte ich erfahren, ob Angebote selbst eingeholt werden dürfen oder ob dies durch die Beschaffungsstelle erfolgt.",
    "Als Projektleiter*in oder Mitarbeitender möchte ich verstehen, was ein Vergabeverfahren bzw. eine Beschaffung im Kontext der Vergaberegelungen bedeutet.",
    "Als Projektleiter*in einer rheinland-pfälzischen Hochschule möchte ich wissen, warum Vergabeverfahren durchgeführt werden müssen und warum eine direkte Vergabe an einen Wunschlieferanten nicht zulässig ist.",
]

QUESTIONS_OUT_OF_CONTEXT = [
    "Wie mache ich eine Mondlandung?",
    "Wie koche ich Pasta?",
    "Wer hat die Relativitätstheorie erfunden?",
    "Wie repariere ich einen kaputten Laptop?",
    "Ich brauche einen Laptop - wie beschaffe ich diesen?",
]

ALL_QUESTIONS = QUESTIONS_IN_CONTEXT + QUESTIONS_OUT_OF_CONTEXT

def ask_api(query: str, model: str = DEFAULT_MODEL, timeout: int = 300) -> dict | None:
    try:
        resp = requests.get(API_URL, params={"query": query, "model": model}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  FEHLER: {e}")
        return None

def run():
    output_file = "eval_results_17.json"

    print("=" * 60)
    print(f"  RAG-Evaluation mit {len(ALL_QUESTIONS)} Fragen")
    print(f"  Modell: {DEFAULT_MODEL}")
    print("=" * 60)

    entries = []

    for i, query in enumerate(ALL_QUESTIONS, 1):
        context_flag = "KONTEXT" if i <= len(QUESTIONS_IN_CONTEXT) else "OUT-OF-CONTEXT"
        print(f"\n[{i}/{len(ALL_QUESTIONS)}] ({context_flag}) {query[:70]}...")
        sys.stdout.flush()

        start = time.time()
        result = ask_api(query)
        elapsed = time.time() - start

        if result:
            entries.append({
                "query": query,
                "answer": result.get("antwort", ""),
                "model": result.get("model", DEFAULT_MODEL),
                "intent": result.get("intent", ""),
                "retrieval_mode": result.get("retrieval_mode", ""),
                "meta": result.get("meta", {}),
                "quellen": result.get("quellen", []),
                "out_of_context": i > len(QUESTIONS_IN_CONTEXT),
            })
            hits = result.get("meta", {}).get("hits", 0)
            score = result.get("meta", {}).get("score", 0)
            print(f"  ✅ {elapsed:.1f}s | Hits: {hits} | Score: {score}")
            print(f"  Antwort: {result.get('antwort', '')[:120]}...")
        else:
            entries.append({
                "query": query,
                "answer": None,
                "error": "API request failed"
            })
            print(f"  ❌ Fehlgeschlagen")

    data = {
        "export_zeit": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "app": "RAG-Evaluation-17-Fragen",
        "run_name": "eval_17_fragen",
        "model": DEFAULT_MODEL,
        "entries": entries,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  Ergebnisse gespeichert: {output_file}")
    print(f"  Erfolgreich: {sum(1 for e in entries if e.get('answer'))}/{len(entries)}")

    if entries:
        avg_hits = sum(e.get("meta", {}).get("hits", 0) for e in entries if e.get("meta")) / max(len(entries), 1)
        avg_latency = sum(e.get("meta", {}).get("latency_ms", 0) for e in entries if e.get("meta")) / max(len(entries), 1)
        print(f"  Durchschnitt Hits: {avg_hits:.1f}")
        print(f"  Durchschnitt Latenz: {avg_latency:.0f}ms")
    print(f"\nNächste Schritte:")
    print(f"  python evaluation.py --json {output_file} --mode A --k 3")
    print(f"  python evaluation.py --json {output_file} --mode B --k 3")
    print(f"  python llm_as_a_judge.py --json {output_file} --judge-model mistral:latest --save judge_results_17.json")

if __name__ == "__main__":
    run()
