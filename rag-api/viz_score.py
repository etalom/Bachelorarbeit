import matplotlib.pyplot as plt
import numpy as np

# ── Daten ──────────────────────────────────────────────
scenarios = ["In-Kontext\n(Schwellenwert)", "Out-of-Context\n(Wetter)"]

# Scores: RRF roh = 0.4/(60+1) + 0.6/(60+1) = 0.01639
rrf_raw = 0.01639                       # alter Score (rein rankbasiert)
in_context_vec = 0.85                   # hohe Cosine-Ähnlichkeit
out_context_vec = 0.48                  # niedrige Cosine-Ähnlichkeit

old_scores   = [rrf_raw, rrf_raw]       # früher identisch
new_scores   = [rrf_raw * in_context_vec,
                rrf_raw * out_context_vec]

# Relevanz-Schwellen
hoch_th  = 0.012
mittel_th = 0.008

# ── Plot ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

x = np.arange(len(scenarios))
width = 0.5

# Farben und Labels
colors_old = ['#d32f2f' if s < hoch_th else '#f57c00' if s < mittel_th else '#388e3c' for s in old_scores]
colors_new = ['#d32f2f' if s < mittel_th else '#f57c00' if s < hoch_th else '#388e3c' for s in new_scores]

bars1 = ax1.bar(x, old_scores, width, color=['#e0e0e0', '#e0e0e0'], edgecolor='#999', linewidth=1.5)
bars2 = ax2.bar(x, new_scores, width, color=['#43a047', '#e53935'], edgecolor='#999', linewidth=1.5)

# Schwellen-Linien
for ax in [ax1, ax2]:
    ax.axhline(hoch_th,   xmin=0, xmax=1, color='#2e7d32', ls='--', lw=1.2, label=f'hoch ≥ {hoch_th}')
    ax.axhline(mittel_th, xmin=0, xmax=1, color='#ef6c00', ls='--', lw=1.2, label=f'mittel ≥ {mittel_th}')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0, 0.020)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

# Werte über Balken
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.0003,
             f'{h:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.0003,
             f'{h:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Relevanz-Label in Balken
for bar, s in zip(bars1, old_scores):
    lbl = 'hoch' if s >= hoch_th else 'mittel' if s >= mittel_th else 'niedrig'
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
             lbl, ha='center', va='center', fontsize=11, fontweight='bold', color='#666')

for bar, s in zip(bars2, new_scores):
    lbl = 'hoch' if s >= hoch_th else 'mittel' if s >= mittel_th else 'niedrig'
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
             lbl, ha='center', va='center', fontsize=11, fontweight='bold', color='white')

ax1.set_title('FRÜHER (nur Rank)', fontsize=13, fontweight='bold', pad=10)
ax2.set_title('JETZT (Rank × Cosine)', fontsize=13, fontweight='bold', pad=10)

# Hinweis unten
fig.text(0.5, -0.02,
         'RRF roh = 0.4/(60+1) + 0.6/(60+1) = 0.01639  |  vector_score (In) = 0.85  |  vector_score (Out) = 0.48',
         ha='center', fontsize=9, style='italic', color='#666')

plt.tight_layout()
plt.savefig('/home/ruki-labor/Schreibtisch/BA-Talom/rag-api/score_vergleich.png', dpi=200, bbox_inches='tight')
print("✅ Bild erstellt: score_vergleich.png")
