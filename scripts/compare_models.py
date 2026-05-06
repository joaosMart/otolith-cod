#!/usr/bin/env python3
"""
Compare frozen vs LoRA-adapted DINOv2 embeddings.

Loads results from both pipelines and generates comparison tables,
statistical significance tests, and comparison plots.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --frozen-results outputs/results/dinov2_frozen --lora-results outputs/results/dinov2_lora
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(description="Compare frozen vs LoRA-adapted DINOv2")
    parser.add_argument("--frozen-results", type=str, required=True,
                        help="Path to frozen model results.json")
    parser.add_argument("--lora-results", type=str, required=True,
                        help="Path to LoRA model results.json")
    parser.add_argument("--output-dir", type=str, default="outputs/figures/comparison")
    return parser.parse_args()


def load_results(path):
    with open(path) as f:
        return json.load(f)


def extract_per_experiment_metrics(results):
    """Extract per-experiment metric values for paired testing."""
    metrics = {}
    for exp in results["experiment_results"]:
        for key, value in exp["test_metrics"].items():
            metrics.setdefault(key, []).append(value)
    return {k: np.array(v) for k, v in metrics.items()}


def print_comparison_table(frozen, lora, p_values):
    """Print formatted comparison table."""
    metric_names = {
        "accuracy": ("Accuracy", True),
        "accuracy_pm1": ("Acc +/-1", True),
        "f1": ("F1 (macro)", True),
        "precision": ("Precision", True),
        "recall": ("Recall", True),
        "rmse_raw": ("RMSE", False),
    }

    header = f"{'Metric':<15} {'Frozen':>20} {'LoRA':>20} {'p-value':>10} {'Sig?':>5}"
    print(header)
    print("-" * len(header))

    for key, (name, is_pct) in metric_names.items():
        if key not in frozen or key not in lora:
            continue
        f_mean, f_std = frozen[key]
        l_mean, l_std = lora[key]
        p = p_values.get(key, float("nan"))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        if is_pct:
            print(f"{name:<15} {f_mean*100:6.2f} +/- {f_std*100:.2f}%  {l_mean*100:6.2f} +/- {l_std*100:.2f}%  {p:10.4f} {sig:>5}")
        else:
            print(f"{name:<15} {f_mean:6.3f} +/- {f_std:.3f}    {l_mean:6.3f} +/- {l_std:.3f}    {p:10.4f} {sig:>5}")


def plot_comparison_bars(frozen_agg, lora_agg, output_path):
    """Bar chart comparing key metrics."""
    metrics = ["accuracy", "accuracy_pm1", "f1", "rmse_raw"]
    names = ["Accuracy", "Acc +/-1", "F1", "RMSE"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))

    for i, (key, name) in enumerate(zip(metrics, names)):
        if key not in frozen_agg or key not in lora_agg:
            continue
        f_mean, f_std = frozen_agg[key]
        l_mean, l_std = lora_agg[key]

        is_pct = key != "rmse_raw"
        scale = 100 if is_pct else 1

        x = np.array([0, 1])
        vals = [f_mean * scale, l_mean * scale]
        errs = [f_std * scale, l_std * scale]

        bars = axes[i].bar(x, vals, yerr=errs, capsize=5, color=["#4C72B0", "#DD8452"], width=0.5)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(["Frozen", "LoRA"])
        axes[i].set_title(name)
        axes[i].set_ylabel("%" if is_pct else "")
        axes[i].grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + errs[vals.index(val)] + 0.3,
                         f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Frozen vs LoRA-adapted DINOv2", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to: {output_path}")


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("MODEL COMPARISON: Frozen vs LoRA-adapted DINOv2")
    print(f"{'='*60}")

    # Load results
    frozen_data = load_results(args.frozen_results)
    lora_data = load_results(args.lora_results)

    # Extract aggregated results
    frozen_agg = {k: (v["mean"], v["std"]) for k, v in frozen_data["aggregated_results"].items()}
    lora_agg = {k: (v["mean"], v["std"]) for k, v in lora_data["aggregated_results"].items()}

    # Extract per-experiment metrics for paired t-test
    frozen_per_exp = extract_per_experiment_metrics(frozen_data)
    lora_per_exp = extract_per_experiment_metrics(lora_data)

    # Paired t-tests
    p_values = {}
    print("\nPaired t-tests (per experiment):")
    for key in frozen_per_exp:
        if key in lora_per_exp and len(frozen_per_exp[key]) == len(lora_per_exp[key]):
            t_stat, p_val = stats.ttest_rel(frozen_per_exp[key], lora_per_exp[key])
            p_values[key] = p_val

    # Print comparison
    print()
    print_comparison_table(frozen_agg, lora_agg, p_values)

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart
    plot_comparison_bars(frozen_agg, lora_agg, str(output_dir / "comparison_bars.png"))

    # Save comparison summary
    summary = {
        "frozen_results_path": args.frozen_results,
        "lora_results_path": args.lora_results,
        "frozen_aggregated": {k: {"mean": v[0], "std": v[1]} for k, v in frozen_agg.items()},
        "lora_aggregated": {k: {"mean": v[0], "std": v[1]} for k, v in lora_agg.items()},
        "paired_t_test_p_values": {k: float(v) for k, v in p_values.items()},
    }
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
