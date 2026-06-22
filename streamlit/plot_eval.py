import matplotlib.pyplot as plt

ks = [1, 3, 5]

results = {
    "runB": {
        "Precision": {1: 0.0, 3: 0.3333, 5: 0.600},
        "Recall":    {1: 0.0, 3: 0.3333, 5: 1.0}
    }
}

plt.figure(figsize=(12, 8))

# --- Precision@k ---
plt.subplot(1, 2, 1)
for run_name, m in results.items():
    y_vals = [m["Precision"][k] for k in ks]
    plt.plot(ks, y_vals, marker="o", linewidth=2, label=run_name)
    for x, y in zip(ks, y_vals):
        plt.text(x, y, f"{y:.2f}", fontsize=9, ha="center", va="bottom")
plt.title("Precision@k")
plt.xlabel("k")
plt.ylabel("Precision")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()

# --- Recall@k ---
plt.subplot(1, 2, 2)
for run_name, m in results.items():
    y_vals = [m["Recall"][k] for k in ks]
    plt.plot(ks, y_vals, marker="o", linewidth=2, label=run_name)
    for x, y in zip(ks, y_vals):
        plt.text(x, y, f"{y:.2f}", fontsize=9, ha="center", va="bottom")
plt.title("Recall@k")
plt.xlabel("k")
plt.ylabel("Recall")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

