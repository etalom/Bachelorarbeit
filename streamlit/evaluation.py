import json
import argparse
import requests
import time
import sys
import os
import numpy as np

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
            return url
        except:
            continue
    return "http://localhost:11434"

OLLAMA_URL = resolve_ollama_url() + "/api/embeddings"
EMBEDDING_MODEL = "bge-m3:latest"

def get_embedding(text: str) -> list[float]:
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": EMBEDDING_MODEL,
            "prompt": text,
        }, timeout=30)
        resp.raise_for_status()
        return resp.json().get("embedding", [])
    except Exception as e:
        print(f"  [FEHLER] Embedding: {e}", file=sys.stderr)
        return []

def cosine_sim(a: list[float], b: list[float]) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_relevance(query_emb: list[float], chunk_text: str) -> tuple[str, float]:
    if not query_emb:
        return "mittel", 0.0
    chunk_emb = get_embedding(chunk_text[:500])
    if not chunk_emb:
        return "mittel", 0.0
    sim = cosine_sim(query_emb, chunk_emb)
    if sim >= 0.40:
        return "hoch", sim
    elif sim >= 0.28:
        return "mittel", sim
    else:
        return "niedrig", sim

def precision_recall_at_k(entries, k, mode):
    precs = []
    recalls = []

    for idx, e in enumerate(entries):
        sources = e.get("quellen", [])
        query = e.get("query", "")
        if not sources or not query:
            continue

        print(f"  [{idx+1}] Embedding für: {query[:50]}...", end=" ")
        sys.stdout.flush()
        query_emb = get_embedding(query)
        if not query_emb:
            print("❌")
            continue

        scored = []
        for s in sources:
            chunk_text = s.get("abschnitt", "")
            rel, sim = compute_relevance(query_emb, chunk_text)
            scored.append({**s, "_relevanz_sim": rel, "_sim": round(sim, 4)})

        scored_sorted = sorted(scored, key=lambda x: x.get("score", 0), reverse=True)
        topk = scored_sorted[:k]

        relevant_total = sum(1 for s in scored if s["_relevanz_sim"] == "hoch" or
                            (mode == "B" and s["_relevanz_sim"] in ("hoch", "mittel")))
        relevant_in_topk = sum(1 for s in topk if s["_relevanz_sim"] == "hoch" or
                              (mode == "B" and s["_relevanz_sim"] in ("hoch", "mittel")))

        precision = relevant_in_topk / k if k > 0 else 0.0
        recall = (relevant_in_topk / relevant_total) if relevant_total > 0 else 0.0

        top_str = ", ".join([f"{s['_relevanz_sim']}({s['_sim']})" for s in topk])
        print(f"P@{k}={precision:.2f} R@{k}={recall:.2f} | {top_str}")

        precs.append(precision)
        recalls.append(recall)

    avg_prec = sum(precs) / len(precs) if precs else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    return avg_prec, avg_recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="eval_results_17.json")
    parser.add_argument("--mode", type=str, default="A", help="A=hoch, B=hoch+mittel")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    print(f"Berechne Embedding-Similarity für {len(entries)} Entries ...")

    avg_p, avg_r = precision_recall_at_k(entries, args.k, args.mode)

    print(f"\nPrecision@{args.k}: {avg_p:.4f}")
    print(f"Recall@{args.k}:    {avg_r:.4f}")
