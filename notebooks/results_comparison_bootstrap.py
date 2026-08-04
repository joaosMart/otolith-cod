import json, os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

base = os.path.join(os.path.dirname(__file__), "../outputs/results")
dirs = sorted(glob.glob(f"{base}/*/bootstrap/results.json"))

# Collect data
rows = []
for p in dirs:
    # Name from grandparent dir (e.g. "siglip2" from outputs/results/siglip2/bootstrap/)
    name = os.path.basename(os.path.dirname(os.path.dirname(p)))
    data = json.load(open(p))
    agg = data["aggregated_results"]

    rows.append({
        "name": name,
        "acc_mean": agg["accuracy"]["mean"],
        "acc_lo": agg["accuracy"]["ci_lower"],
        "acc_hi": agg["accuracy"]["ci_upper"],
        "acc1_mean": agg["accuracy_pm1"]["mean"],
        "acc1_lo": agg["accuracy_pm1"]["ci_lower"],
        "acc1_hi": agg["accuracy_pm1"]["ci_upper"],
        "f1_mean": agg["f1"]["mean"],
        "f1_lo": agg["f1"]["ci_lower"],
        "f1_hi": agg["f1"]["ci_upper"],
    })

if not rows:
    print("No bootstrap results found. Run train_classifier.py --eval-mode bootstrap first.")
    exit(1)

# Sort by accuracy
rows.sort(key=lambda r: r["acc_mean"])

# Print table
print(f"{'Configuration':<40} {'Accuracy':>24} {'Acc ±1':>24} {'F1 Score':>24}")
print("-" * 116)
for r in rows:
    print(
        f"{r['name']:<40} "
        f"{r['acc_mean']:.4f} [{r['acc_lo']:.4f}, {r['acc_hi']:.4f}]  "
        f"{r['acc1_mean']:.4f} [{r['acc1_lo']:.4f}, {r['acc1_hi']:.4f}]  "
        f"{r['f1_mean']:.4f} [{r['f1_lo']:.4f}, {r['f1_hi']:.4f}]"
    )

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, max(4, len(rows) * 0.8 + 2)))
names = [r["name"] for r in rows]
x = np.arange(len(names))

metrics = [
    ("acc_mean", "acc_lo", "acc_hi", "Accuracy"),
    ("acc1_mean", "acc1_lo", "acc1_hi", "Accuracy ±1"),
    ("f1_mean", "f1_lo", "f1_hi", "F1 Score"),
]

for ax, (key_mean, key_lo, key_hi, title) in zip(axes, metrics):
    means = np.array([r[key_mean] for r in rows])
    lows = np.array([r[key_lo] for r in rows])
    highs = np.array([r[key_hi] for r in rows])
    # Asymmetric error bars: [lower_err, upper_err]
    err = np.array([means - lows, highs - means])

    bars = ax.barh(x, means, xerr=err, capsize=4,
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, len(names))))
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(min(lows) - 0.03, max(highs) + 0.03)
    for i, (m, hi) in enumerate(zip(means, highs)):
        ax.text(hi + 0.002, i, f"{m:.3f}", va='center', fontsize=9)

plt.suptitle("Bootstrap Evaluation (95% CI)", fontweight='bold', y=1.02)
plt.tight_layout()
out_path = os.path.join(base, "results_comparison_bootstrap.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPlot saved to {out_path}")
