#!/usr/bin/env python3
"""
Turn the raw run outputs into the tables and figures the manuscript quotes.

Replaces generate_paper_figures.py, which addressed result directories by
hardcoded names from an earlier naming scheme and could not see the current
runs at all.

Everything downstream of the classifier lives here: the per-age breakdown, the
paired McNemar test between frozen and adapted, the encoder comparison, the
preprocessing comparison, the seed spread, and the attribution figures. All of
it is derived from the result JSONs plus a refit of the same Ridge model, so a
number in the paper can be traced to a file rather than to a run that has since
been overwritten.

    python scripts/summarize_results.py --frozen siglip2-frozen \\
        --lora siglip2_lora_r16a32_s42_clahe
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.data.splits import load_split_by_ids
from src.features.metadata import augment_embeddings

RESULTS = Path("outputs/results")
FIGURES = Path("outputs/paper_figures")

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
})
FROZEN_COLOR, LORA_COLOR = "#4C72B0", "#C44E52"


def load_result(name):
    path = RESULTS / name / "bootstrap" / "results.json"
    return json.loads(path.read_text()) if path.exists() else None


def metric(result, key):
    """Point estimate and 95% interval for one metric."""
    agg = result["aggregated_results"][key]
    return agg["mean"], agg.get("ci_lower"), agg.get("ci_upper")


def predictions(name, split_file):
    """Refit the classifier for a run and return its test predictions.

    The stored results carry aggregate metrics only. Anything per-class, and the
    paired test, needs the predictions themselves, so the same Ridge is refit at
    the alpha the run selected. Deterministic, so this reproduces the stored
    accuracy exactly; the caller checks that it does.
    """
    result = load_result(name)
    config = result["config"]
    data = np.load(config["embeddings_path"])

    features, labels = data["features"], data["labels"]
    measurement_ids = data["measurement_ids"]
    if config.get("add_tabular", True):
        features = augment_embeddings(features, measurement_ids,
                                      config["metadata_csv"] if "metadata_csv" in config
                                      else "cod_otolith_age_final_with_scale.csv",
                                      columns=config["tabular_columns"])

    split = load_split_by_ids(split_file, measurement_ids)
    train, test = split.train_indices, split.test_indices

    scaler = StandardScaler().fit(features[train])
    alpha = result["experiment_results"][0]["best_alpha"]
    model = Ridge(alpha=alpha).fit(scaler.transform(features[train]), labels[train])
    predicted = np.clip(np.round(model.predict(scaler.transform(features[test]))), 1, 10)

    return predicted.astype(int), labels[test].astype(int)


def mcnemar(frozen_pred, lora_pred, truth):
    """Paired test on the same test images, with continuity correction."""
    frozen_ok, lora_ok = frozen_pred == truth, lora_pred == truth
    lora_only = int(np.sum(~frozen_ok & lora_ok))
    frozen_only = int(np.sum(frozen_ok & ~lora_ok))
    n = lora_only + frozen_only
    if n == 0:
        return {"lora_only": 0, "frozen_only": 0, "chi2": None, "p_value": None}
    chi2 = (abs(lora_only - frozen_only) - 1) ** 2 / n
    from scipy.stats import chi2 as chi2_dist
    return {"lora_only": lora_only, "frozen_only": frozen_only,
            "chi2": float(chi2), "p_value": float(chi2_dist.sf(chi2, 1))}


def per_age(frozen_pred, lora_pred, truth):
    rows = []
    ages = sorted(np.unique(truth))
    f1_frozen = f1_score(truth, frozen_pred, average=None, labels=ages, zero_division=0)
    f1_lora = f1_score(truth, lora_pred, average=None, labels=ages, zero_division=0)
    for i, age in enumerate(ages):
        mask = truth == age
        rows.append({
            "age": int(age), "n": int(mask.sum()),
            "f1_frozen": 100 * f1_frozen[i], "f1_lora": 100 * f1_lora[i],
            "acc_frozen": 100 * float((frozen_pred[mask] == age).mean()),
            "acc_lora": 100 * float((lora_pred[mask] == age).mean()),
        })
    return rows


def figure_per_age(rows, path):
    ages = [r["age"] for r in rows]
    x = np.arange(len(ages))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in [(axes[0], "f1", "Macro F1 by age class"),
                           (axes[1], "acc", "Accuracy by age class")]:
        ax.bar(x - 0.2, [r[f"{key}_frozen"] for r in rows], 0.4,
               label="Frozen", color=FROZEN_COLOR)
        ax.bar(x + 0.2, [r[f"{key}_lora"] for r in rows], 0.4,
               label="LoRA", color=LORA_COLOR)
        ax.set_xticks(x); ax.set_xticklabels(ages)
        ax.set_xlabel("Age (years)"); ax.set_ylabel(f"{key.upper()} (%)")
        ax.set_title(title); ax.legend()
    top = axes[1].twiny()
    top.set_xlim(axes[1].get_xlim()); top.set_xticks(x)
    top.set_xticklabels([r["n"] for r in rows], fontsize=7)
    top.set_xlabel("test images per class", fontsize=8); top.grid(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def figure_encoders(entries, path):
    """Frozen against adapted for every encoder, with intervals."""
    labels = [e["label"] for e in entries]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "f1", "Macro F1")]:
        for offset, cond, label, color in [(-0.2, "frozen", "Frozen", FROZEN_COLOR),
                                           (0.2, "lora", "LoRA", LORA_COLOR)]:
            values = [e[cond][key][0] if e.get(cond) else np.nan for e in entries]
            lower = [v - (e[cond][key][1] if e.get(cond) else v)
                     for v, e in zip(values, entries)]
            upper = [(e[cond][key][2] if e.get(cond) else v) - v
                     for v, e in zip(values, entries)]
            ax.bar(x + offset, values, 0.4, yerr=[lower, upper], capsize=3,
                   label=label, color=color)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel(title); ax.set_title(title); ax.legend()
        ax.set_ylim(0.45, 0.78)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def figure_seeds(values, path):
    """Where the effect sits relative to run-to-run variation."""
    fig, ax = plt.subplots(figsize=(6, 4))
    seeds = sorted(values)
    accuracies = [values[s] for s in seeds]
    ax.bar([str(s) for s in seeds], accuracies, color=LORA_COLOR, width=0.6)
    mean = float(np.mean(accuracies))
    sd = float(np.std(accuracies, ddof=1))
    ax.axhline(mean, color="black", lw=1.2, label=f"mean {mean:.3f}")
    ax.axhspan(mean - sd, mean + sd, color="black", alpha=0.10,
               label=f"$\\pm$1 sd ({sd:.3f})")
    ax.set_xlabel("Random seed"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0.60, 0.70); ax.legend(fontsize=8)
    ax.set_title("SigLIP2 + LoRA across seeds")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def figure_rank_sweep(points, seed_sd, path):
    """Accuracy against LoRA rank, with the seed band for scale.

    The band is what makes the figure readable: without it a reader sees five
    slightly different bars and infers a trend, when in fact the whole range sits
    inside the variation between two runs of the same configuration.
    """
    ranks = sorted(points)
    acc = [points[r]["accuracy"][0] for r in ranks]
    lo = [points[r]["accuracy"][0] - points[r]["accuracy"][1] for r in ranks]
    hi = [points[r]["accuracy"][2] - points[r]["accuracy"][0] for r in ranks]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    mean = float(np.mean(acc))
    ax.axhspan(mean - seed_sd, mean + seed_sd, color="black", alpha=0.10,
               label=f"$\\pm$1 sd between seeds ({seed_sd:.3f})")
    ax.errorbar(range(len(ranks)), acc, yerr=[lo, hi], marker="o", capsize=4,
                color=LORA_COLOR, lw=1.5)
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([f"{r}\n{points[r]['params']/1e6:.1f}M" for r in ranks])
    ax.set_xlabel("LoRA rank, and trainable parameters")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.60, 0.70)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def figure_learning_curves(curves, frozen, path):
    """Accuracy against the number of labelled images used for adaptation."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    colors = {"siglip2": LORA_COLOR, "clip": FROZEN_COLOR}
    names = {"siglip2": "SigLIP2", "clip": "CLIP"}
    ticks = set()
    for encoder, points in curves.items():
        n = sorted(points)
        ticks.update(n)
        label = names.get(encoder, encoder)
        ax.plot(n, [points[k] for k in n], marker="o", lw=1.6,
                color=colors.get(encoder), label=f"{label} + LoRA")
        if encoder in frozen:
            ax.axhline(frozen[encoder], ls="--", lw=1.2, color=colors.get(encoder),
                       alpha=0.75,
                       label=f"{label} frozen, all {max(n):,} images")
    ax.set_xscale("log")
    ax.set_xticks(sorted(ticks))
    ax.set_xticklabels([f"{t:,}" for t in sorted(ticks)])
    ax.minorticks_off()
    ax.set_xlabel("Labelled images used for adaptation (log scale)")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def figure_attribution(frozen_shap, lora_shap, path_abs, path_share):
    """Attribution per feature group, absolute and as a share of the total."""
    embed_key = next(k for k in frozen_shap if "mbed" in k)
    groups = [k for k in frozen_shap if k != embed_key]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={"width_ratios": [1, 3]})
    axes[0].bar([-0.2], [frozen_shap[embed_key]], 0.4, color=FROZEN_COLOR, label="Frozen")
    axes[0].bar([0.2], [lora_shap[embed_key]], 0.4, color=LORA_COLOR, label="LoRA")
    axes[0].set_xticks([0]); axes[0].set_xticklabels(["Image embeddings"])
    axes[0].set_ylabel("Mean |SHAP|"); axes[0].legend()
    x = np.arange(len(groups))
    axes[1].bar(x - 0.2, [frozen_shap[g] for g in groups], 0.4, color=FROZEN_COLOR)
    axes[1].bar(x + 0.2, [lora_shap[g] for g in groups], 0.4, color=LORA_COLOR)
    axes[1].set_xticks(x); axes[1].set_xticklabels(groups, rotation=30, ha="right")
    axes[1].set_ylabel("Mean |SHAP|")
    fig.tight_layout(); fig.savefig(path_abs); plt.close(fig)

    ftot, ltot = sum(frozen_shap.values()), sum(lora_shap.values())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={"width_ratios": [1, 3]})
    fmeta = 100 * (ftot - frozen_shap[embed_key]) / ftot
    lmeta = 100 * (ltot - lora_shap[embed_key]) / ltot
    axes[0].bar(["Frozen", "LoRA"], [100 - fmeta, 100 - lmeta],
                color=[FROZEN_COLOR, LORA_COLOR], label="Image embeddings")
    axes[0].bar(["Frozen", "LoRA"], [fmeta, lmeta], bottom=[100 - fmeta, 100 - lmeta],
                color=["#8FB0D8", "#E0928F"], label="Metadata")
    for i, v in enumerate([fmeta, lmeta]):
        axes[0].text(i, 100 - v / 2, f"{v:.1f}%", ha="center", fontsize=9)
    axes[0].set_ylabel("Share of total attribution (%)"); axes[0].legend(fontsize=8)
    axes[1].bar(x - 0.2, [100 * frozen_shap[g] / ftot for g in groups], 0.4,
                color=FROZEN_COLOR, label="Frozen")
    axes[1].bar(x + 0.2, [100 * lora_shap[g] / ltot for g in groups], 0.4,
                color=LORA_COLOR, label="LoRA")
    axes[1].set_xticks(x); axes[1].set_xticklabels(groups, rotation=30, ha="right")
    axes[1].set_ylabel("Share of total attribution (%)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(path_share); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frozen", default="siglip2-frozen")
    p.add_argument("--lora", default="siglip2_lora_r16a32_s42_clahe")
    p.add_argument("--split-file", default="outputs/splits/main_split.json")
    p.add_argument("--figures", default=str(FIGURES))
    p.add_argument("--seeds", default="42,7,13,101,2024")
    args = p.parse_args()

    figures = Path(args.figures); figures.mkdir(parents=True, exist_ok=True)
    summary = {}

    # Headline comparison and everything derived from predictions.
    frozen_pred, truth = predictions(args.frozen, args.split_file)
    lora_pred, truth2 = predictions(args.lora, args.split_file)
    assert np.array_equal(truth, truth2), "the two runs were scored on different test sets"

    for name, pred in [(args.frozen, frozen_pred), (args.lora, lora_pred)]:
        stored = metric(load_result(name), "accuracy")[0]
        achieved = float((pred == truth).mean())
        flag = "" if abs(stored - achieved) < 0.005 else "   MISMATCH"
        print(f"refit check {name:<40} stored {stored:.4f} achieved {achieved:.4f}{flag}")

    summary["mcnemar"] = mcnemar(frozen_pred, lora_pred, truth)
    summary["per_age"] = per_age(frozen_pred, lora_pred, truth)
    figure_per_age(summary["per_age"], figures / "per_age_comparison.png")

    # Encoder comparison, frozen against adapted.
    entries = []
    for encoder, label in [("clip", "CLIP"), ("siglip2", "SigLIP2"),
                           ("dinov2", "DINOv2"), ("dinov3", "DINOv3")]:
        entry = {"label": label}
        for cond, name in [("frozen", f"{encoder}-frozen"),
                           ("lora", f"{encoder}_lora_r16a32_s42_clahe")]:
            result = load_result(name)
            if result:
                entry[cond] = {k: metric(result, k)
                               for k in ("accuracy", "f1", "accuracy_pm1", "rmse")}
        entries.append(entry)
    summary["encoders"] = entries
    figure_encoders(entries, figures / "encoder_comparison.png")

    # Seed spread, which is what tells us how much of a difference is real.
    seed_values = {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        result = load_result(f"siglip2_lora_r16a32_s{seed}_clahe")
        if result:
            seed_values[seed] = metric(result, "accuracy")[0]
    if len(seed_values) > 1:
        values = list(seed_values.values())
        summary["seeds"] = {
            "per_seed": seed_values,
            "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
            "min": min(values), "max": max(values), "n": len(values),
        }
        figure_seeds(seed_values, figures / "seed_variation.png")

    # Ablations that need no predictions, only the stored aggregates.
    for key, names in [
        ("preprocessing", {"pad": "siglip2-frozen", "squash": "siglip2-frozen-squash",
                           "crop": "siglip2-frozen-crop"}),
        ("pooling_head", {"lora": args.lora,
                          "lora_pool": "siglip2_lora-pool_r16a32_s42_clahe"}),
    ]:
        block = {}
        for label, name in names.items():
            result = load_result(name)
            if result:
                block[label] = {m: metric(result, m)
                                for m in ("accuracy", "f1", "accuracy_pm1", "rmse")}
        summary[key] = block

    # Attribution, when the analysis has been run.
    shap = {}
    for cond, name in [("frozen", args.frozen), ("lora", args.lora)]:
        path = RESULTS / name / "bootstrap" / "feature_importance" / "feature_importance.json"
        if path.exists():
            data = json.loads(path.read_text())
            shap[cond] = data["shap_importances"][0]
            summary.setdefault("forward_selection", {})[cond] = data["forward_selection"]
            summary.setdefault("permutation", {})[cond] = data["permutation_importances"][0]
    if len(shap) == 2:
        summary["shap"] = shap
        figure_attribution(shap["frozen"], shap["lora"],
                           figures / "shap_comparison.png",
                           figures / "shap_comparison_proportional.png")
    else:
        print("attribution not available yet; skipping those figures")

    out = Path("outputs/paper_summary.json")
    out.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote {out} and figures to {figures}")

    m = summary["mcnemar"]
    print(f"\nMcNemar: LoRA-only correct {m['lora_only']}, frozen-only {m['frozen_only']}, "
          f"chi2={m['chi2']:.2f}, p={m['p_value']:.2e}")
    if "seeds" in summary:
        s = summary["seeds"]
        print(f"Seeds: mean {s['mean']:.4f} sd {s['sd']:.4f} over {s['n']} runs")


if __name__ == "__main__":
    raise SystemExit(main())
