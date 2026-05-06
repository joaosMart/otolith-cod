import json, os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

base = os.path.join(os.path.dirname(__file__), "../outputs/results")
dirs = sorted(glob.glob(f"{base}/*/results.json"))

# Collect data
rows = []
for p in dirs:
    name = os.path.basename(os.path.dirname(p)).replace("_shallow_siglip2", "")
    data = json.load(open(p))
    exps = data["experiment_results"]
    accs = [e["test_metrics"]["accuracy"] for e in exps]
    f1s = [e["test_metrics"]["f1"] for e in exps]
    acc_pm1 = [e["test_metrics"]["accuracy_pm1"] for e in exps]
    rows.append({
        "name": name, 
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "acc1_mean": np.mean(acc_pm1), "acc1_std": np.std(acc_pm1),
        "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
    })

# Sort by accuracy
rows.sort(key=lambda r: r["acc_mean"])

# Print table
print(f"{'Configuration':<40} {'Accuracy':>16} {'Acc ±1':>16} {'F1 Score':>16}")
print("-" * 92)
for r in rows:
    print(f"{r['name']:<40} {r['acc_mean']:.4f} ± {r['acc_std']:.4f}  {r['acc1_mean']:.4f} ± {r['acc1_std']:.4f}  {r['f1_mean']:.4f} ± {r['f1_std']:.4f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
names = [r["name"] for r in rows]
x = np.arange(len(names))

for ax, key, title in zip(axes, 
    [("acc_mean","acc_std"), ("acc1_mean","acc1_std"), ("f1_mean","f1_std")],
    ["Accuracy", "Accuracy ±1", "F1 Score"]):
    means = [r[key[0]] for r in rows]
    stds = [r[key[1]] for r in rows]
    bars = ax.barh(x, means, xerr=stds, capsize=4, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(names))))
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(min(means) - 0.03, max(means) + 0.03)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.002, i, f"{m:.3f}", va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "../outputs/results/results_comparison.png"), dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to outputs/results/results_comparison.png")
