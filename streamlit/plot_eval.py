import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

ks = [1, 3, 5, 7, 10, 15]

results = {
    "In-Context (Run B)": {
        "Precision": {
            1: 1.0000,
            3: 1.0000,
            5: 1.0000,
            7: 1.0000,
            10: 1.0000,
            15: 1.0000
        },
        "Recall": {
            1: 0.0667,
            3: 0.2000,
            5: 0.3333,
            7: 0.4667,
            10: 0.6667,
            15: 1.0000
        }
    },
    "Out-of-Context": {
        "Precision": {
            1: 0.2000,
            3: 0.2000,
            5: 0.2400,
            7: 0.2857,
            10: 0.3200,
            15: 0.3200
        },
        "Recall": {
            1: 0.0154,
            3: 0.0462,
            5: 0.1103,
            7: 0.4090,
            10: 0.6064,
            15: 0.8000
        }
    }
}

plt.figure(figsize=(12, 6))

# --- Precision ---
plt.subplot(1, 2, 1)
for run_name, m in results.items():
    y_vals = [m["Precision"][k] for k in ks]
    plt.plot(ks, y_vals, marker="o", linewidth=2, label=run_name)

    for x, y in zip(ks, y_vals):
        plt.text(x, y, f"{y:.2f}", fontsize=8, ha="center", va="bottom")

plt.title("Precision@k Vergleich")
plt.xlabel("k")
plt.ylabel("Precision")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()

# --- Recall ---
plt.subplot(1, 2, 2)
for run_name, m in results.items():
    y_vals = [m["Recall"][k] for k in ks]
    plt.plot(ks, y_vals, marker="o", linewidth=2, label=run_name)

    for x, y in zip(ks, y_vals):
        plt.text(x, y, f"{y:.2f}", fontsize=8, ha="center", va="bottom")

plt.title("Recall@k Vergleich")
plt.xlabel("k")
plt.ylabel("Recall")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("ic_vs_ooc.png", dpi=300)
plt.show()

