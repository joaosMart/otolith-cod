#!/usr/bin/env python3
"""Generate paper figures from bootstrap results.

Uses the same prediction logic as analyze_performance.py:
load embeddings + split + best_alpha, retrain Ridge, predict.
"""

import sys
sys.path.insert(0, ".")

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.features.metadata import augment_embeddings
from src.data.splits import load_split_by_ids

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

RESULTS = Path("outputs/results")
FIGURES = Path("paper/figures")
METADATA_CSV = "cod_otolith_age_final_with_scale.csv"


def style_barplot(ax, orientation='vertical'):
    """Style a barplot: keep only the bar-origin axis, add grid lines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if orientation == 'vertical':
        ax.spines['left'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.tick_params(left=False)
    else:  # horizontal
        ax.spines['bottom'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, zorder=0)
        ax.tick_params(bottom=False)


def parse_ci(ci_str):
    parts = ci_str.strip().split("[")
    mean = float(parts[0].strip())
    bounds = parts[1].rstrip("]").split(",")
    return mean, float(bounds[0].strip()), float(bounds[1].strip())


def load_summary(model):
    path = RESULTS / model / "bootstrap" / "summary.csv"
    with open(path) as f:
        reader = list(csv.DictReader(f))
    row0, row1 = reader[0], reader[1]
    metrics = {}
    for key in ['test_accuracy', 'test_accuracy_pm1', 'test_f1', 'test_rmse']:
        metrics[key] = {
            'mean': float(row0[key]),
            'ci_lower': parse_ci(row1[key])[1],
            'ci_upper': parse_ci(row1[key])[2],
        }
    return metrics


def get_predictions(model_name):
    """Reproduce predictions using the same logic as analyze_performance.py.

    Loads embeddings, augments with tabular features, loads the fixed split,
    trains Ridge with saved best_alpha, and returns (y_true, y_pred).
    """
    with open(RESULTS / model_name / "bootstrap" / "results.json") as f:
        results = json.load(f)

    config = results['config']
    best_alpha = results['experiment_results'][0]['best_alpha']

    # Load embeddings
    emb_path = config['embeddings_path']
    data = np.load(emb_path, allow_pickle=True)
    feature_key = config.get('feature_key', 'features')
    features = data[feature_key] if feature_key in data else data['features']
    labels = data['labels']
    measurement_ids = data['measurement_ids']

    # Augment with tabular features (same as train_classifier.py)
    if config.get('add_tabular', False):
        features = augment_embeddings(
            features, measurement_ids, METADATA_CSV, config['tabular_columns']
        )

    # Load fixed split (same as analyze_performance.py bootstrap path)
    split_path = RESULTS / model_name / "bootstrap" / "split.json"
    split = load_split_by_ids(str(split_path), measurement_ids)

    X_train = features[split.train_indices]
    y_train = labels[split.train_indices]
    X_test = features[split.test_indices]
    y_test = labels[split.test_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Ridge(alpha=best_alpha)
    model.fit(X_train, y_train)
    y_pred = np.clip(np.round(model.predict(X_test)).astype(int), 1, 10)

    acc = (y_pred == y_test).mean()
    expected = results['experiment_results'][0]['test_metrics']['accuracy']
    print(f"  {model_name}: acc = {acc:.4f} (expected {expected:.4f})")
    return y_test, y_pred


def compute_per_class(y_true, y_pred, classes=range(1, 11)):
    f1_per = f1_score(y_true, y_pred, labels=list(classes), average=None, zero_division=0)
    acc_per, counts = [], []
    for c in classes:
        mask = y_true == c
        n = mask.sum()
        counts.append(n)
        acc_per.append((y_pred[mask] == c).mean() if n > 0 else 0)
    return np.array(list(classes)), f1_per, np.array(acc_per), np.array(counts)


# ============================================================
# FIGURE 1: Simplified encoder comparison
# ============================================================
def figure_encoder_comparison():
    models_4 = ['clip-clahe', 'pe-core-clahe', 'siglip2-clahe', 'siglip2-lora']
    labels_4 = ['CLIP', 'PE-Core', 'SigLIP2\n(frozen)', 'SigLIP2\n(LoRA)']
    models_3 = ['clip-clahe', 'siglip2-clahe', 'siglip2-lora']
    labels_3 = ['CLIP', 'SigLIP2 (frozen)', 'SigLIP2 (LoRA)']

    for suffix, models, labels in [('4model', models_4, labels_4), ('3model', models_3, labels_3)]:
        summaries = {m: load_summary(m) for m in models}
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52'][:len(models)]

        for ax, mkey, mlabel in zip(axes, ['test_accuracy', 'test_f1'], ['Accuracy', 'F1 Score']):
            means = [summaries[m][mkey]['mean'] for m in models]
            ci_lo = [summaries[m][mkey]['ci_lower'] for m in models]
            ci_hi = [summaries[m][mkey]['ci_upper'] for m in models]
            errs = [[m - lo for m, lo in zip(means, ci_lo)],
                    [hi - m for m, hi in zip(means, ci_hi)]]
            bars = ax.bar(range(len(models)), means, yerr=errs, capsize=5,
                         color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(labels)
            ax.set_ylabel(mlabel)
            ax.set_ylim(0.45, 0.75)
            style_barplot(ax, 'vertical')
            for i, (bar, val) in enumerate(zip(bars, means)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errs[1][i] + 0.005,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        fig.savefig(FIGURES / f'encoder_comparison_{suffix}.pdf')
        fig.savefig(FIGURES / f'encoder_comparison_{suffix}.png')
        plt.close()
        print(f"Saved encoder_comparison_{suffix}")


# ============================================================
# FIGURE 2: SHAP comparison
# ============================================================
def figure_shap_comparison():
    with open(RESULTS / "siglip2-clahe" / "bootstrap" / "feature_importance" / "feature_importance.json") as f:
        frozen_shap = json.load(f)['shap_importances'][0]
    with open(RESULTS / "siglip2-lora" / "bootstrap" / "feature_importance" / "feature_importance.json") as f:
        lora_shap = json.load(f)['shap_importances'][0]

    meta_features = ['Length', 'Month', 'Seg Width', 'Seg Height', 'Seg Aspect Ratio',
                     'Longitude', 'Latitude', 'Is Survey']

    print("\n=== SHAP VALUES ===")
    print(f"{'Feature':<20} {'Frozen':>10} {'LoRA':>10} {'Change':>10}")
    print("-" * 55)
    for feat in ['SigLIP Embeddings'] + meta_features:
        fv, lv = frozen_shap[feat], lora_shap[feat]
        print(f"{feat:<20} {fv:>10.4f} {lv:>10.4f} {(lv-fv)/fv*100:>+9.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1, 2.5]})

    vals = [frozen_shap['SigLIP Embeddings'], lora_shap['SigLIP Embeddings']]
    bars = ax1.bar([0, 1], vals, color=['#4C72B0', '#C44E52'], edgecolor='black', linewidth=0.5, width=0.6)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(['Frozen', 'LoRA'])
    ax1.set_ylabel('Mean |SHAP value|')
    style_barplot(ax1, 'vertical')
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    x = np.arange(len(meta_features)); width = 0.35
    ax2.bar(x - width/2, [frozen_shap[f] for f in meta_features], width,
            label='Frozen', color='#4C72B0', edgecolor='black', linewidth=0.5, alpha=0.85)
    ax2.bar(x + width/2, [lora_shap[f] for f in meta_features], width,
            label='LoRA', color='#C44E52', edgecolor='black', linewidth=0.5, alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(meta_features, rotation=35, ha='right')
    ax2.set_ylabel('Mean |SHAP value|')
    ax2.legend()
    style_barplot(ax2, 'vertical')
    plt.tight_layout()
    fig.savefig(FIGURES / 'shap_comparison.pdf')
    fig.savefig(FIGURES / 'shap_comparison.png')
    plt.close()
    print("Saved shap_comparison")

    # --- Proportional SHAP figure ---
    all_features = ['SigLIP Embeddings'] + meta_features
    frozen_total = sum(frozen_shap[f] for f in all_features)
    lora_total = sum(lora_shap[f] for f in all_features)
    frozen_pct = {f: frozen_shap[f] / frozen_total * 100 for f in all_features}
    lora_pct = {f: lora_shap[f] / lora_total * 100 for f in all_features}

    # Sum metadata share
    frozen_meta_pct = sum(frozen_pct[f] for f in meta_features)
    lora_meta_pct = sum(lora_pct[f] for f in meta_features)

    print(f"\n=== PROPORTIONAL SHAP (% of total) ===")
    print(f"{'Feature':<20} {'Frozen %':>10} {'LoRA %':>10}")
    print("-" * 45)
    for feat in all_features:
        print(f"{feat:<20} {frozen_pct[feat]:>9.1f}% {lora_pct[feat]:>9.1f}%")
    print(f"{'--- Total metadata':<20} {frozen_meta_pct:>9.1f}% {lora_meta_pct:>9.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1, 2.5]})

    # Panel A: Embeddings vs total metadata (stacked)
    frozen_emb_pct = frozen_pct['SigLIP Embeddings']
    lora_emb_pct = lora_pct['SigLIP Embeddings']
    x_pos = [0, 1]
    ax1.bar(x_pos, [frozen_emb_pct, lora_emb_pct], color=['#4C72B0', '#C44E52'],
            edgecolor='black', linewidth=0.5, width=0.6, label='Image embeddings')
    ax1.bar(x_pos, [frozen_meta_pct, lora_meta_pct],
            bottom=[frozen_emb_pct, lora_emb_pct], color=['#7EB0D5', '#F4A582'],
            edgecolor='black', linewidth=0.5, width=0.6, label='Metadata')
    ax1.set_xticks(x_pos); ax1.set_xticklabels(['Frozen', 'LoRA'])
    ax1.set_ylabel('Share of total SHAP (%)')
    ax1.legend(fontsize=9); ax1.set_ylim(0, 105)
    style_barplot(ax1, 'vertical')
    # Annotate percentages
    ax1.text(0, frozen_emb_pct / 2, f'{frozen_emb_pct:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold')
    ax1.text(1, lora_emb_pct / 2, f'{lora_emb_pct:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold')
    ax1.text(0, frozen_emb_pct + frozen_meta_pct / 2, f'{frozen_meta_pct:.1f}%', ha='center', va='center', fontsize=9)
    ax1.text(1, lora_emb_pct + lora_meta_pct / 2, f'{lora_meta_pct:.1f}%', ha='center', va='center', fontsize=9)

    # Panel B: Per-metadata feature proportional
    x = np.arange(len(meta_features)); width = 0.35
    ax2.bar(x - width/2, [frozen_pct[f] for f in meta_features], width,
            label='Frozen', color='#4C72B0', edgecolor='black', linewidth=0.5, alpha=0.85)
    ax2.bar(x + width/2, [lora_pct[f] for f in meta_features], width,
            label='LoRA', color='#C44E52', edgecolor='black', linewidth=0.5, alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(meta_features, rotation=35, ha='right')
    ax2.set_ylabel('Share of total SHAP (%)')
    ax2.legend()
    style_barplot(ax2, 'vertical')

    plt.tight_layout()
    fig.savefig(FIGURES / 'shap_comparison_proportional.pdf')
    fig.savefig(FIGURES / 'shap_comparison_proportional.png')
    plt.close()
    print("Saved shap_comparison_proportional")


# ============================================================
# FIGURE 3: Per-age comparison
# ============================================================
def figure_per_age_comparison():
    print("\n=== GENERATING PER-CLASS PREDICTIONS ===")
    y_true_f, y_pred_f = get_predictions('siglip2-clahe')
    y_true_l, y_pred_l = get_predictions('siglip2-lora')

    classes, f1_frozen, acc_frozen, counts = compute_per_class(y_true_f, y_pred_f)
    _, f1_lora, acc_lora, _ = compute_per_class(y_true_l, y_pred_l)

    print(f"\n{'Age':<6} {'Frozen F1':>10} {'LoRA F1':>10} {'Diff':>8} {'N_test':>8}")
    print("-" * 45)
    for i, c in enumerate(classes):
        print(f"{c:<6} {f1_frozen[i]:>10.3f} {f1_lora[i]:>10.3f} {f1_lora[i]-f1_frozen[i]:>+7.3f} {counts[i]:>8d}")

    print(f"\n{'Age':<6} {'Frozen Acc':>10} {'LoRA Acc':>10} {'Diff':>8}")
    print("-" * 40)
    for i, c in enumerate(classes):
        print(f"{c:<6} {acc_frozen[i]:>10.3f} {acc_lora[i]:>10.3f} {acc_lora[i]-acc_frozen[i]:>+7.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(classes)); width = 0.35

    for ax, frozen_vals, lora_vals, ylabel, title in [
        (ax1, f1_frozen, f1_lora, 'F1 Score', 'Per-Age F1 Score'),
        (ax2, acc_frozen, acc_lora, 'Accuracy', 'Per-Age Accuracy'),
    ]:
        ax.bar(x - width/2, frozen_vals, width, label='SigLIP2 (frozen)', color='#4C72B0',
               edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.bar(x + width/2, lora_vals, width, label='SigLIP2 (LoRA)', color='#C44E52',
               edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(classes)
        ax.set_xlabel('Age (years)'); ax.set_ylabel(ylabel)
        ax.legend(loc='lower left')
        ax.set_ylim(0, 1.0)
        style_barplot(ax, 'vertical')

    for ax in (ax1, ax2):
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim()); ax_top.set_xticks(x)
        ax_top.set_xticklabels([f'n={c}' for c in counts], fontsize=8, color='gray')
        ax_top.tick_params(length=0)
        for spine in ax_top.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGURES / 'per_age_comparison.pdf')
    fig.savefig(FIGURES / 'per_age_comparison.png')
    plt.close()
    print("Saved per_age_comparison")


def print_summary():
    print("=" * 60)
    print("KEY NUMBERS FOR PAPER")
    print("=" * 60)
    models = ['clip-clahe', 'pe-core-clahe', 'siglip2-clahe', 'siglip2-lora']
    labels = ['CLIP-CLAHE', 'PE-Core-CLAHE', 'SigLIP2-CLAHE (frozen)', 'SigLIP2-LoRA']
    print(f"\n{'Model':<25} {'Accuracy':>20} {'F1':>20} {'Acc±1':>20}")
    print("-" * 90)
    for m, l in zip(models, labels):
        s = load_summary(m)
        acc, f1, a1 = s['test_accuracy'], s['test_f1'], s['test_accuracy_pm1']
        print(f"{l:<25} {acc['mean']:.4f} [{acc['ci_lower']:.4f}, {acc['ci_upper']:.4f}]"
              f"  {f1['mean']:.4f} [{f1['ci_lower']:.4f}, {f1['ci_upper']:.4f}]"
              f"  {a1['mean']:.4f} [{a1['ci_lower']:.4f}, {a1['ci_upper']:.4f}]")

    frozen, lora = load_summary('siglip2-clahe'), load_summary('siglip2-lora')
    print(f"\nLoRA improvement:")
    for key, label in [('test_accuracy', 'Accuracy'), ('test_f1', 'F1'), ('test_accuracy_pm1', 'Acc±1')]:
        diff = lora[key]['mean'] - frozen[key]['mean']
        print(f"  {label}: +{diff:.4f} ({diff*100:.1f}pp)")


if __name__ == '__main__':
    print_summary()
    figure_encoder_comparison()
    figure_shap_comparison()
    figure_per_age_comparison()
    print("\nAll figures saved to paper/figures/")
