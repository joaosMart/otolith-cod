"""
Feature importance analysis: SHAP, permutation importance, and forward selection ablation.

Provides functions to understand how each input feature (SigLIP embeddings vs tabular metadata)
contributes to the Ridge classifier's age predictions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer

from src.evaluation.metrics import compute_classification_metrics

matplotlib.use("Agg")

# Individual tabular column names (must match order in augment_embeddings)
TABULAR_COLUMN_NAMES = [
    "length",
    "month_sin",
    "month_cos",
    "shot_latitude",
    "shot_longitude",
    "is_survey",
    "seg_width_px",
    "seg_height_px",
    "seg_aspect_ratio",
]

# Feature groups: month_sin + month_cos are grouped as "Month"
TABULAR_GROUP_SPEC = [
    ("Length", ["length"]),
    ("Month", ["month_sin", "month_cos"]),
    ("Latitude", ["shot_latitude"]),
    ("Longitude", ["shot_longitude"]),
    ("Is Survey", ["is_survey"]),
    ("Seg Width", ["seg_width_px"]),
    ("Seg Height", ["seg_height_px"]),
    ("Seg Aspect Ratio", ["seg_aspect_ratio"]),
]

EMBEDDING_DIM = 1152


def build_feature_groups(embedding_dim: int = EMBEDDING_DIM) -> Dict[str, np.ndarray]:
    """Build mapping from group name to feature indices."""
    groups = {"SigLIP Embeddings": np.arange(embedding_dim)}
    for group_name, col_names in TABULAR_GROUP_SPEC:
        indices = [embedding_dim + TABULAR_COLUMN_NAMES.index(c) for c in col_names]
        groups[group_name] = np.array(indices)
    return groups


def _macro_f1_scorer(model, X, y):
    """Macro F1 scorer for permutation importance."""
    y_pred = np.clip(np.round(model.predict(X)).astype(int), 1, 10)
    return f1_score(y, y_pred, average="macro", zero_division=0)


# --- SHAP Analysis ---


def compute_shap_values(model, X_train, X_test, feature_groups):
    """
    Compute SHAP values using LinearExplainer and aggregate by feature group.

    Args:
        model: Fitted Ridge model (on scaled data)
        X_train: Scaled training data
        X_test: Scaled test data
        feature_groups: Dict mapping group name -> array of feature indices

    Returns:
        Dict mapping group name -> mean |SHAP| value
    """
    import shap

    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)  # (n_samples, n_features)

    group_importance = {}
    for name, indices in feature_groups.items():
        # Sum absolute SHAP values across all dims in the group, then mean across samples
        group_importance[name] = float(np.mean(np.sum(np.abs(shap_values[:, indices]), axis=1)))

    return group_importance


def plot_shap_summary(group_importances_list: List[Dict[str, float]], output_path: str):
    """
    Bar plot of mean |SHAP| per feature group, aggregated across splits.

    Args:
        group_importances_list: List of dicts (one per split) mapping group name -> importance
        output_path: Path to save the plot
    """
    # Aggregate across splits
    all_names = list(group_importances_list[0].keys())
    means = {}
    stds = {}
    for name in all_names:
        values = [gi[name] for gi in group_importances_list]
        means[name] = np.mean(values)
        stds[name] = np.std(values)

    # Sort by importance
    sorted_names = sorted(all_names, key=lambda n: means[n], reverse=True)
    sorted_means = [means[n] for n in sorted_names]
    sorted_stds = [stds[n] for n in sorted_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_means, xerr=sorted_stds, align="center", color="#1f77b4", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance (aggregated across splits)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_shap_dependence(
    shap_values_list: List[np.ndarray],
    X_test_list: List[np.ndarray],
    feature_idx: int,
    feature_name: str,
    output_path: str,
):
    """
    Dependence plot for a specific tabular feature across all splits.

    Args:
        shap_values_list: List of SHAP value arrays (one per split)
        X_test_list: List of UNSCALED X_test arrays (for interpretable x-axis)
        feature_idx: Index of the feature in the full feature matrix
        feature_name: Display name
        output_path: Path to save the plot
    """
    all_feature_vals = np.concatenate([X[:, feature_idx] for X in X_test_list])
    all_shap_vals = np.concatenate([sv[:, feature_idx] for sv in shap_values_list])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(all_feature_vals, all_shap_vals, alpha=0.3, s=10, color="#1f77b4")
    ax.set_xlabel(feature_name)
    ax.set_ylabel(f"SHAP value for {feature_name}")
    ax.set_title(f"SHAP Dependence: {feature_name}")
    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_shap_dependence_month(
    shap_values_list: List[np.ndarray],
    X_test_unscaled_list: List[np.ndarray],
    sin_idx: int,
    cos_idx: int,
    output_path: str,
):
    """
    SHAP dependence plot for Month, reconstructing the original month on the x-axis.

    Recovers month from arctan2(sin, cos) using UNSCALED data and sums |SHAP|
    across both components.

    Args:
        shap_values_list: List of SHAP value arrays (one per split)
        X_test_unscaled_list: List of UNSCALED X_test arrays (one per split)
        sin_idx: Index of month_sin in the feature matrix
        cos_idx: Index of month_cos in the feature matrix
        output_path: Path to save the plot
    """
    all_sin = np.concatenate([X[:, sin_idx] for X in X_test_unscaled_list])
    all_cos = np.concatenate([X[:, cos_idx] for X in X_test_unscaled_list])
    all_shap_sin = np.concatenate([sv[:, sin_idx] for sv in shap_values_list])
    all_shap_cos = np.concatenate([sv[:, cos_idx] for sv in shap_values_list])

    # Recover original month: angle = atan2(sin, cos), month = angle * 12 / (2*pi)
    angles = np.arctan2(all_sin, all_cos)
    months = np.round(angles * 12 / (2 * np.pi)) % 12 + 1  # 1-12

    month_labels = np.arange(1, 13)
    rng = np.random.RandomState(42)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot signed SHAP for sin and cos separately, grouped by month
    for component, shap_vals, color, label in [
        ("sin", all_shap_sin, "#1f77b4", "month_sin"),
        ("cos", all_shap_cos, "#ff7f0e", "month_cos"),
    ]:
        offset = -0.15 if component == "sin" else 0.15
        for m in month_labels:
            mask = months == m
            if mask.sum() == 0:
                continue
            vals = shap_vals[mask]
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.full(len(vals), m) + offset + jitter,
                vals,
                alpha=0.25, s=8, color=color,
                label=label if m == month_labels[0] else None,
            )

    # Add median lines per month for each component
    for component, shap_vals, color in [
        ("sin", all_shap_sin, "#1f77b4"),
        ("cos", all_shap_cos, "#ff7f0e"),
    ]:
        offset = -0.15 if component == "sin" else 0.15
        medians_x, medians_y = [], []
        for m in month_labels:
            mask = months == m
            if mask.sum() == 0:
                continue
            medians_x.append(m + offset)
            medians_y.append(np.median(shap_vals[mask]))
        ax.plot(medians_x, medians_y, marker="D", markersize=6, color=color,
                linewidth=1.5, linestyle="--", alpha=0.8)

    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_xticks(month_labels)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("SHAP value (signed)")
    ax.set_title("SHAP Dependence: Month (sin & cos components)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_shap_dependence_month_net(
    shap_values_list: List[np.ndarray],
    X_test_unscaled_list: List[np.ndarray],
    sin_idx: int,
    cos_idx: int,
    output_path: str,
):
    """
    SHAP dependence plot for Month showing net contribution (SHAP_sin + SHAP_cos).

    Positive = month pushes predicted age up, negative = pushes age down.

    Args:
        shap_values_list: List of SHAP value arrays (one per split)
        X_test_unscaled_list: List of UNSCALED X_test arrays (one per split)
        sin_idx: Index of month_sin in the feature matrix
        cos_idx: Index of month_cos in the feature matrix
        output_path: Path to save the plot
    """
    all_sin = np.concatenate([X[:, sin_idx] for X in X_test_unscaled_list])
    all_cos = np.concatenate([X[:, cos_idx] for X in X_test_unscaled_list])
    all_shap_sin = np.concatenate([sv[:, sin_idx] for sv in shap_values_list])
    all_shap_cos = np.concatenate([sv[:, cos_idx] for sv in shap_values_list])

    angles = np.arctan2(all_sin, all_cos)
    months = np.round(angles * 12 / (2 * np.pi)) % 12 + 1

    net_shap = all_shap_sin + all_shap_cos

    fig, ax = plt.subplots(figsize=(10, 5))
    month_labels = np.arange(1, 13)
    rng = np.random.RandomState(42)

    # Scatter points
    for m in month_labels:
        mask = months == m
        if mask.sum() == 0:
            continue
        vals = net_shap[mask]
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), m) + jitter, vals, alpha=0.25, s=8, color="#2ca02c")

    # Median line
    medians_x, medians_y = [], []
    for m in month_labels:
        mask = months == m
        if mask.sum() == 0:
            continue
        medians_x.append(m)
        medians_y.append(np.median(net_shap[mask]))
    ax.plot(medians_x, medians_y, marker="D", markersize=7, color="#d62728",
            linewidth=2, linestyle="-", label="Median", zorder=5)

    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_xticks(month_labels)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Net SHAP value (sin + cos)")
    ax.set_title("SHAP Dependence: Month (net effect on predicted age)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_shap_dependence_month_impact(
    shap_values_list: List[np.ndarray],
    X_test_unscaled_list: List[np.ndarray],
    sin_idx: int,
    cos_idx: int,
    output_path: str,
):
    """
    SHAP dependence plot for Month showing overall impact (|SHAP| combined).

    Args:
        shap_values_list: List of SHAP value arrays (one per split)
        X_test_unscaled_list: List of UNSCALED X_test arrays (one per split)
        sin_idx: Index of month_sin in the feature matrix
        cos_idx: Index of month_cos in the feature matrix
        output_path: Path to save the plot
    """
    all_sin = np.concatenate([X[:, sin_idx] for X in X_test_unscaled_list])
    all_cos = np.concatenate([X[:, cos_idx] for X in X_test_unscaled_list])
    all_shap_sin = np.concatenate([sv[:, sin_idx] for sv in shap_values_list])
    all_shap_cos = np.concatenate([sv[:, cos_idx] for sv in shap_values_list])

    angles = np.arctan2(all_sin, all_cos)
    months = np.round(angles * 12 / (2 * np.pi)) % 12 + 1

    combined_shap = np.abs(all_shap_sin) + np.abs(all_shap_cos)

    fig, ax = plt.subplots(figsize=(10, 5))
    month_labels = np.arange(1, 13)
    box_data = [combined_shap[months == m] for m in month_labels]
    non_empty = [(m, d) for m, d in zip(month_labels, box_data) if len(d) > 0]
    if non_empty:
        labels, data = zip(*non_empty)
        # matplotlib 3.9 renamed this keyword from `labels` to `tick_labels`.
        # Accept either, so the analysis is not tied to one matplotlib release.
        try:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(data, labels=labels, showfliers=False)
        rng = np.random.RandomState(42)
        for i, (m, d) in enumerate(non_empty):
            jitter = rng.uniform(-0.15, 0.15, size=len(d))
            ax.scatter(np.full(len(d), i + 1) + jitter, d, alpha=0.2, s=8, color="#1f77b4")

    ax.set_xlabel("Month")
    ax.set_ylabel("|SHAP value| (sin + cos combined)")
    ax.set_title("SHAP Dependence: Month (overall impact)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# --- Permutation Importance ---


def compute_grouped_permutation_importance(
    model, X_test, y_test, feature_groups, n_repeats=10, random_state=42
):
    """
    Compute permutation importance by feature group.

    For each group, permute all indices in the group together and measure
    the drop in macro F1.

    Args:
        model: Fitted model
        X_test: Scaled test features
        y_test: True labels
        feature_groups: Dict mapping group name -> array of feature indices
        n_repeats: Number of permutation repeats
        random_state: Random seed

    Returns:
        Dict mapping group name -> (mean_importance, std_importance)
    """
    rng = np.random.RandomState(random_state)
    baseline_score = _macro_f1_scorer(model, X_test, y_test)

    results = {}
    for name, indices in feature_groups.items():
        scores = []
        for _ in range(n_repeats):
            X_permuted = X_test.copy()
            # Permute all columns in this group with the same permutation
            perm = rng.permutation(X_permuted.shape[0])
            X_permuted[:, indices] = X_permuted[perm][:, indices]
            score = _macro_f1_scorer(model, X_permuted, y_test)
            scores.append(baseline_score - score)
        results[name] = (float(np.mean(scores)), float(np.std(scores)))

    return results


def plot_permutation_importance(
    perm_importances_list: List[Dict[str, Tuple[float, float]]], output_path: str
):
    """
    Bar plot of permutation importance with error bars, aggregated across splits.

    Args:
        perm_importances_list: List of dicts (one per split) mapping name -> (mean, std)
        output_path: Path to save the plot
    """
    all_names = list(perm_importances_list[0].keys())

    # Aggregate means across splits
    agg_means = {}
    agg_stds = {}
    for name in all_names:
        split_means = [pi[name][0] for pi in perm_importances_list]
        agg_means[name] = np.mean(split_means)
        agg_stds[name] = np.std(split_means)

    sorted_names = sorted(all_names, key=lambda n: agg_means[n], reverse=True)
    sorted_means = [agg_means[n] for n in sorted_names]
    sorted_stds = [agg_stds[n] for n in sorted_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_means, xerr=sorted_stds, align="center", color="#ff7f0e", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel("Drop in Macro F1")
    ax.set_title("Permutation Feature Importance (aggregated across splits)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# --- Forward Selection Ablation ---


def run_forward_selection(
    features: np.ndarray,
    labels: np.ndarray,
    splits,
    alpha_range: np.ndarray,
    cv_folds: int,
    feature_groups: Dict[str, np.ndarray],
    group_order: List[str],
) -> List[Dict]:
    """
    Forward selection ablation: start with SigLIP-only, add features one by one.

    Args:
        features: Full feature matrix (N, D)
        labels: Labels (N,)
        splits: List of DataSplit objects
        alpha_range: Alpha values for RidgeCV
        cv_folds: Inner CV folds
        feature_groups: Dict mapping group name -> feature indices
        group_order: Order to add features (first should be "SigLIP Embeddings")

    Returns:
        List of dicts with keys: step, features_included, f1_mean, f1_std
    """
    from src.evaluation.metrics import aggregate_fold_results

    results = []
    included_indices = np.array([], dtype=int)

    for step, group_name in enumerate(group_order):
        included_indices = np.concatenate([included_indices, feature_groups[group_name]])
        included_indices = included_indices.astype(int)

        X_subset = features[:, included_indices]
        fold_metrics = []

        for split in splits:
            X_train = X_subset[split.train_indices]
            y_train = labels[split.train_indices]
            X_test = X_subset[split.test_indices]
            y_test = labels[split.test_indices]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = RidgeCV(alphas=alpha_range, cv=cv_folds)
            model.fit(X_train, y_train)

            y_pred = np.clip(np.round(model.predict(X_test)).astype(int), 1, 10)
            metrics = compute_classification_metrics(y_test, y_pred)
            fold_metrics.append(metrics)

        agg = aggregate_fold_results(fold_metrics)
        features_so_far = group_order[: step + 1]

        results.append(
            {
                "step": step,
                "feature_added": group_name,
                "features_included": features_so_far.copy(),
                "n_features": len(included_indices),
                "f1_mean": agg["f1"][0],
                "f1_std": agg["f1"][1],
                "accuracy_mean": agg["accuracy"][0],
                "accuracy_std": agg["accuracy"][1],
            }
        )

        print(
            f"  Step {step}: +{group_name:20s} | "
            f"F1={agg['f1'][0]*100:.2f}±{agg['f1'][1]*100:.2f}% | "
            f"Acc={agg['accuracy'][0]*100:.2f}±{agg['accuracy'][1]*100:.2f}% | "
            f"dims={len(included_indices)}"
        )

    return results


def plot_forward_selection(results: List[Dict], output_path: str):
    """
    Line chart showing cumulative F1 as features are added.

    Args:
        results: Output from run_forward_selection
        output_path: Path to save the plot
    """
    steps = [r["step"] for r in results]
    f1_means = [r["f1_mean"] * 100 for r in results]
    f1_stds = [r["f1_std"] * 100 for r in results]
    labels = [r["feature_added"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(steps, f1_means, yerr=f1_stds, marker="o", linewidth=2, capsize=4, color="#2ca02c")

    ax.set_xticks(steps)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Feature Added")
    ax.set_ylabel("Macro F1 (%)")
    ax.set_title("Forward Selection Ablation: Cumulative Feature Addition")
    ax.grid(True, alpha=0.3)

    # Annotate each point
    for i, (x, y) in enumerate(zip(steps, f1_means)):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def save_results_json(
    shap_importances: List[Dict],
    perm_importances: List[Dict],
    forward_results: List[Dict],
    output_path: str,
):
    """Save all numerical results to JSON."""
    data = {
        "shap_importances": shap_importances,
        "permutation_importances": [
            {name: {"mean": v[0], "std": v[1]} for name, v in pi.items()}
            for pi in perm_importances
        ],
        "forward_selection": forward_results,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {output_path}")
