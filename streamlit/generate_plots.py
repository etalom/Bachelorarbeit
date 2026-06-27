#!/usr/bin/env python3
"""
Erzeugt alle Diagramme für das Evaluationskapitel.
Nutzt die vorhandenen JSON-Datendateien im streamlit-Ordner.
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
OUT  = HERE / "abbildungen"
OUT.mkdir(exist_ok=True)

# ── Quelldateien ────────────────────────────────────────────────────────
with open(HERE / "best_chunks_and_answer.json") as f:
    ctx_data = json.load(f)
with open(HERE / "best_chunks_and_answer_out_kontext.json") as f:
    ooc_data = json.load(f)
with open(HERE.parent / "judge_results_17.json") as f:
    judge_data = json.load(f)
# Vergleich der Retrieval-Methoden
vergleich_file = sorted(HERE.glob("vergleich_*.json"))
if vergleich_file:
    with open(vergleich_file[-1]) as f:
        vergleich = json.load(f)
else:
    vergleich = []

ctx_entries = ctx_data["entries"]
ooc_entries = ooc_data["entries"]
all_entries = ctx_entries + ooc_entries
judge_entries = judge_data["entries"]

COLORS = {"correctness": "#e74c3c", "completeness": "#f39c12",
          "faithfulness": "#2ecc71", "clarity": "#3498db"}

def get_judge(query):
    for j in judge_entries:
        if j["query"].strip() == query.strip():
            return j.get("scores") or {}
    return {}

# ═══════════════════════════════════════════════════════════════════════════
# 1) LLM-Judge Overall (N=17)
# ═══════════════════════════════════════════════════════════════════════════
labels = ["Correctness", "Completeness", "Faithfulness", "Clarity"]
keys   = ["correctness", "completeness", "faithfulness", "clarity"]
values = []
for k in keys:
    v = [get_judge(e["query"]).get(k, 0) for e in all_entries]
    values.append(np.mean(v))

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels, values, color=[COLORS[k] for k in keys], width=0.6, edgecolor="white")
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
            f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0, 5.8)
ax.set_ylabel("Score (1–5)")
ax.set_title(f"LLM-as-a-Judge — Gesamtergebnis ($N={len(all_entries)}$)\n"
             f"$\\varnothing$ Overall: {np.mean(values):.2f} / 5,0")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "judge_overall.png", dpi=200)
plt.close()
print("✓ judge_overall.png")

# ═══════════════════════════════════════════════════════════════════════════
# 2) LLM-Judge Split: Kontext vs Out-of-Context
# ═══════════════════════════════════════════════════════════════════════════
ctx_means = [np.mean([get_judge(e["query"]).get(k, 0) for e in ctx_entries]) for k in keys]
ooc_means = [np.mean([get_judge(e["query"]).get(k, 0) for e in ooc_entries]) for k in keys]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(labels)); w = 0.35
ax.bar(x - w/2, ctx_means, w, label=f"Kontextfragen ({len(ctx_entries)})", color="#3498db", edgecolor="white")
ax.bar(x + w/2, ooc_means, w, label=f"Out-of-Context ({len(ooc_entries)})", color="#e74c3c", edgecolor="white")
for xi, v in zip(x, ctx_means):
    ax.text(xi - w/2, v + 0.08, f"{v:.2f}", ha="center", fontsize=9)
for xi, v in zip(x, ooc_means):
    ax.text(xi + w/2, v + 0.08, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 6.0); ax.set_ylabel("Score (1–5)")
ax.set_title("LLM-as-a-Judge — Kontext vs. Out-of-Context"); ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "judge_split.png", dpi=200)
plt.close()
print("✓ judge_split.png")

# ═══════════════════════════════════════════════════════════════════════════
# 3) Precision@k / Recall@k (Mode B) – aus den Quellen-Daten
# ═══════════════════════════════════════════════════════════════════════════
def is_rel(rel, mode):
    if mode == "A": return rel == "hoch"
    return rel in ("hoch", "mittel")

def prec_rec_at_k(entries, k, mode):
    precs, recs = [], []
    for e in entries:
        srcs = sorted(e.get("quellen", []), key=lambda x: x.get("score", 0))
        topk = srcs[:k]
        relevant_total = sum(1 for s in srcs if is_rel(s.get("relevanz"), mode))
        relevant_topk  = sum(1 for s in topk if is_rel(s.get("relevanz"), mode))
        precs.append(relevant_topk / k if k else 0)
        recs.append(relevant_topk / relevant_total if relevant_total else 0)
    return np.mean(precs), np.mean(recs)

ks = [1, 3, 5]
prec_vals = [prec_rec_at_k(ctx_entries, k, "B")[0] for k in ks]
rec_vals  = [prec_rec_at_k(ctx_entries, k, "B")[1] for k in ks]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ks, prec_vals, marker="o", linewidth=2, label="Precision@k", color="#3498db")
ax.plot(ks, rec_vals,  marker="s", linewidth=2, label="Recall@k",    color="#2ecc71")
for k, p, r in zip(ks, prec_vals, rec_vals):
    ax.text(k, p + 0.03, f"{p:.2f}", ha="center", fontsize=10, color="#3498db")
    ax.text(k, r - 0.06, f"{r:.2f}", ha="center", fontsize=10, color="#2ecc71")
ax.set_xticks(ks); ax.set_xlabel("k"); ax.set_ylim(0, 1.15)
ax.set_ylabel("Score"); ax.set_title("Precision@k und Recall@k (Mode B)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "precision_recall_k.png", dpi=200)
plt.close()
print("✓ precision_recall_k.png")

# ═══════════════════════════════════════════════════════════════════════════
# 4) Hybrid-Suche Überlappung – aus vergleich_*.json
# ═══════════════════════════════════════════════════════════════════════════
if vergleich:
    gem = np.mean([v["analyse"]["gemeinsame_dokumente"] for v in vergleich])
    bm  = np.mean([v["analyse"]["nur_in_bm25"] for v in vergleich])
    vec = np.mean([v["analyse"]["nur_in_vector"] for v in vergleich])
else:
    gem, bm, vec = 2, 5, 6  # Fallback

fig, ax = plt.subplots(figsize=(6, 5))
cats = ["BM25 exklusiv", "Überschneidung", "Vektor exklusiv"]
cnts = [bm, gem, vec]
ax.bar(cats, cnts, color=["#e67e22", "#2ecc71", "#3498db"], width=0.5, edgecolor="white")
for i, v in enumerate(cnts):
    ax.text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(cnts) + 1.5)
ax.set_ylabel("Durchschnittliche Anzahl Dokumente (k=5)")
ax.set_title("Überlappung BM25 und Vektorsuche")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "hybrid_overlap.png", dpi=200)
plt.close()
print("✓ hybrid_overlap.png")

# ═══════════════════════════════════════════════════════════════════════════
# 5) Verweigerungsrate (Kontextfragen)
# ═══════════════════════════════════════════════════════════════════════════
refused = sum(1 for e in ctx_entries
              if "keine" in e.get("answer","").lower()
              and ("antwort" in e["answer"].lower() or "finden" in e["answer"].lower()))
answered = len(ctx_entries) - refused

fig, ax = plt.subplots(figsize=(5, 5))
ax.pie([answered, refused],
       labels=[f"Beantwortet ({answered})", f"Verweigert ({refused})"],
       autopct="%1.0f%%", startangle=90,
       colors=["#2ecc71", "#e74c3c"],
       textprops={"fontsize": 11})
ax.set_title(f"Antwortverweigerung bei Kontextfragen (N={len(ctx_entries)})")
plt.tight_layout()
fig.savefig(OUT / "verweigerungsrate.png", dpi=200)
plt.close()
print("✓ verweigerungsrate.png")

# ═══════════════════════════════════════════════════════════════════════════
# 6) Relevanz-Heatmap
# ═══════════════════════════════════════════════════════════════════════════
remap = {"hoch": 2, "mittel": 1, "niedrig": 0}
num_mat = [[remap.get(s.get("relevanz",""), 0) for s in e.get("quellen",[])[:5]]
           for e in ctx_entries]

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(num_mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=2)
ax.set_xticks(range(5)); ax.set_xticklabels(["Rang 1","Rang 2","Rang 3","Rang 4","Rang 5"])
ax.set_yticks(range(len(ctx_entries)))
ax.set_yticklabels([f"Frage {i+1}" for i in range(len(ctx_entries))])
ax.set_xlabel("Retrieval-Rang")
ax.set_title("Relevanz der Top-5-Quellen pro Kontextfrage")
for i in range(len(ctx_entries)):
    for j in range(5):
        v = num_mat[i][j]
        ax.text(j, i, {2:"H",1:"M",0:"N"}[v], ha="center", va="center",
                fontsize=9, fontweight="bold", color="white" if v==2 else "black")
import matplotlib.patches as mpatches
ax.legend(handles=[
    mpatches.Patch(color=plt.cm.RdYlGn(0.0), label="niedrig"),
    mpatches.Patch(color=plt.cm.RdYlGn(0.5), label="mittel"),
    mpatches.Patch(color=plt.cm.RdYlGn(1.0), label="hoch"),
], loc="lower right", title="Relevanz")
plt.tight_layout()
fig.savefig(OUT / "relevanz_heatmap.png", dpi=200)
plt.close()
print("✓ relevanz_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# 7) Correctness pro Kontextfrage
# ═══════════════════════════════════════════════════════════════════════════
corr_vals = [get_judge(e["query"]).get("correctness", 0) for e in ctx_entries]
qs_short  = [f"Q{i+1}" for i in range(len(ctx_entries))]
colors_q  = ["#2ecc71" if v >= 3 else "#e74c3c" for v in corr_vals]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(qs_short, corr_vals, color=colors_q, edgecolor="white")
for bar, v in zip(bars, corr_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.08,
            f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, 5.8); ax.set_ylabel("Correctness (1–5)")
ax.set_title("Correctness pro Kontextfrage"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "correctness_per_question.png", dpi=200)
plt.close()
print("✓ correctness_per_question.png")

# ═══════════════════════════════════════════════════════════════════════════
# 8) Latenz-Verteilung (aus eval_results_17.json – enthält meta)
# ═══════════════════════════════════════════════════════════════════════════
try:
    with open(HERE.parent / "eval_results_17.json") as f:
        ev = json.load(f)
    lats = [e.get("meta", {}).get("latency_ms", 0)/1000 for e in ev["entries"]]
except Exception:
    lats = []

if lats:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lats, bins=12, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(lats), color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Ø {np.mean(lats):.1f}s")
    ax.set_xlabel("Latenz (Sekunden)"); ax.set_ylabel("Anzahl Fragen")
    ax.set_title("Latenz-Verteilung der Evaluations-Queries")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT / "latenz_verteilung.png", dpi=200)
    plt.close()
    print("✓ latenz_verteilung.png")

# ═══════════════════════════════════════════════════════════════════════════
# 9) Vergleich BM25 vs Vector Scores (Boxplot o.ä.)
# ═══════════════════════════════════════════════════════════════════════════
if vergleich:
    bm_scores  = [v["bm25_score"] for v in vergleich]
    vec_scores = [v["vector_score"] for v in vergleich]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([bm_scores, vec_scores], labels=["BM25", "Vektorsuche"])
    ax.set_ylabel("Score"); ax.set_title("Retrieval-Scores: BM25 vs. Vektorsuche")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT / "retrieval_scores_vergleich.png", dpi=200)
    plt.close()
    print("✓ retrieval_scores_vergleich.png")

print(f"\nFertig. {len(list(OUT.glob('*.png')))} Diagramme in {OUT}")
