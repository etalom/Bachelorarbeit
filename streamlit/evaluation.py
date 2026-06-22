import json
import argparse

def rel_to_bool(rel, mode):
    if mode == "A":  # relevant = hoch
        return rel == "hoch"
    if mode == "B":  # relevant = hoch + mittel
        return rel in ("hoch", "mittel")
    raise ValueError("mode muss A oder B sein")

def precision_recall_at_k(answers_entries, k, mode):
    """
    Modellannahme:
    - Precision@k = (relevante Treffer in Top-k) / k
    - Recall@k = (relevante Treffer in Top-k) / (alle relevanten Treffer insgesamt)
    """
    precs = []
    recalls = []

    for e in answers_entries:
        sources = e.get("quellen", [])
        if not sources:
            continue

        # sortieren nach score (kleiner score scheint bei dir eher besser/aktueller zu sein – wir nehmen Score aufsteigend)
        sources_sorted = sorted(sources, key=lambda x: x.get("score", 0))
        topk = sources_sorted[:k]

        relevant_total = sum(rel_to_bool(s.get("relevanz"), mode) for s in sources_sorted)
        relevant_in_topk = sum(rel_to_bool(s.get("relevanz"), mode) for s in topk)

        precision = relevant_in_topk / k if k > 0 else 0.0
        recall = (relevant_in_topk / relevant_total) if relevant_total > 0 else 0.0

        precs.append(precision)
        recalls.append(recall)

    avg_prec = sum(precs) / len(precs) if precs else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    return avg_prec, avg_recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="best_chunk_and_antwort.json")
    parser.add_argument("--mode", type=str, default="A", help="A oder B")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    avg_p, avg_r = precision_recall_at_k(entries, args.k, args.mode)

    print(f"Precision@{args.k}: {avg_p:.4f}")
    print(f"Recall@{args.k}:    {avg_r:.4f}")

