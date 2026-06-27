import matplotlib.pyplot as plt

# Richtige Werte
labels = ["Correctness", "Completeness", "Faithfulness", "Clarity"]
scores = [4.75, 4.33, 4.33, 4.67]

# Farben (wie dein Plot: blau + grün)
colors = ["#3b8ec2", "#2ecc71", "#2ecc71", "#2ecc71"]

plt.figure(figsize=(8,5))
bars = plt.bar(labels, scores, color=colors)

# Werte über Balken anzeigen
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.05,
             f"{height:.2f}", ha='center', fontsize=10)

plt.ylim(0, 5.5)
plt.ylabel("Score (1–5)")
plt.title("LLM-as-a-Judge Evaluation\n(Overall: 4.52/5)")

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig("judge_scores_correct.png", dpi=300)
plt.show()

