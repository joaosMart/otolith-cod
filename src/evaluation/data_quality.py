"""Data quality analysis using cleanlab for label error detection.

Functions for identifying potential label issues and data quality problems
using cross-validated predictions.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Cleanlab is optional - check if available
try:
    from cleanlab.filter import find_label_issues
    from cleanlab import Datalab

    CLEANLAB_AVAILABLE = True
except ImportError:
    CLEANLAB_AVAILABLE = False


def compute_out_of_fold_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.1,
    n_folds: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute out-of-fold predictions using cross-validation.

    For Ridge regression, converts continuous predictions to pseudo-probabilities
    using softmax over class distances.

    Args:
        features: Feature matrix (N, D)
        labels: Label vector (N,)
        alpha: Ridge regularization parameter
        n_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility

    Returns:
        Tuple of:
            - pred_probs: Predicted probabilities (N, n_classes)
            - oof_predictions: Out-of-fold class predictions (N,)
    """
    n_samples = len(features)
    classes = np.unique(labels)
    n_classes = len(classes)

    # Initialize arrays
    pred_probs = np.zeros((n_samples, n_classes))
    oof_predictions = np.zeros(n_samples, dtype=int)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, val_idx) in enumerate(
        tqdm(skf.split(features, labels), total=n_folds, desc="CV folds")
    ):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Train Ridge model
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        # Get continuous predictions
        y_val_pred_continuous = model.predict(X_val)

        # Convert to class predictions
        y_val_pred = np.round(y_val_pred_continuous).astype(int)
        # Clip to valid class range
        y_val_pred = np.clip(y_val_pred, classes.min(), classes.max())
        oof_predictions[val_idx] = y_val_pred

        # Convert to pseudo-probabilities using distance to each class
        # Score for each class = exp(-distance^2 / temperature)
        temperature = 1.0
        distances = np.abs(y_val_pred_continuous[:, np.newaxis] - classes[np.newaxis, :])
        scores = np.exp(-distances**2 / temperature)

        # Normalize to probabilities
        probs = scores / scores.sum(axis=1, keepdims=True)
        pred_probs[val_idx] = probs

    return pred_probs, oof_predictions



def find_label_issues_cleanlab(
    features: np.ndarray,
    labels: np.ndarray,
    pred_probs: Optional[np.ndarray] = None,
    alpha: float = 0.1,
    n_folds: int = 10,
    filter_by: str = "both",
    min_examples_per_class: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Use cleanlab to identify potential label issues.

    Args:
        features: Feature matrix (N, D)
        labels: Label vector (N,)
        pred_probs: Pre-computed predicted probabilities (N, n_classes).
            If None, will compute using cross-validation.
        alpha: Ridge regularization parameter
        n_folds: Number of CV folds (used if pred_probs is None)
        filter_by: Cleanlab filtering method ("prune_by_noise_rate", "prune_by_class",
            "both", "confident_learning", "predicted_neq_given")
        min_examples_per_class: Minimum examples per class for filtering
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with label issues, sorted by label quality score

    Raises:
        ImportError: If cleanlab is not installed
    """
    if not CLEANLAB_AVAILABLE:
        raise ImportError(
            "cleanlab is not installed. Install with: pip install cleanlab>=2.6.0"
        )

    # Compute out-of-fold predictions if not provided
    if pred_probs is None:
        print("Computing out-of-fold predictions...")
        pred_probs, _ = compute_out_of_fold_predictions(
            features=features,
            labels=labels,
            alpha=alpha,
            n_folds=n_folds,
            random_state=random_state,
        )

    # Convert 1-indexed labels (1-10) to 0-indexed (0-9) for cleanlab
    # cleanlab expects labels to start at 0 and pred_probs columns to match label indices
    labels_0indexed = labels - 1
    min_label = labels.min()

    # Find label issues using cleanlab
    print(f"Finding label issues using cleanlab (filter_by={filter_by})...")
    label_issues_mask = find_label_issues(
        labels=labels_0indexed,
        pred_probs=pred_probs,
        filter_by=filter_by,
        return_indices_ranked_by="self_confidence",
        min_examples_per_class=min_examples_per_class,
    )

    # Get indices of issues
    issue_indices = np.where(label_issues_mask)[0]

    # Build results dataframe with 1-indexed labels for display
    results = []
    for idx in issue_indices:
        given_label = labels[idx]  # Original 1-indexed label
        pred_label_0indexed = np.argmax(pred_probs[idx])
        pred_label = pred_label_0indexed + min_label  # Convert back to 1-indexed
        confidence_in_given = pred_probs[idx, labels_0indexed[idx]]

        results.append(
            {
                "sample_index": idx,
                "given_label": given_label,
                "suggested_label": pred_label,
                "confidence_in_given": confidence_in_given,
                "confidence_in_suggested": np.max(pred_probs[idx]),
                "suggests_relabel": pred_label != given_label,
            }
        )

    if len(results) == 0:
        print("No label issues found!")
        return pd.DataFrame(
            columns=[
                "sample_index",
                "given_label",
                "suggested_label",
                "confidence_in_given",
                "confidence_in_suggested",
                "suggests_relabel",
            ]
        )

    df = pd.DataFrame(results)
    df = df.sort_values("confidence_in_given", ascending=True)

    print(f"Found {len(df)} potential label issues")

    return df


def plot_label_issues_per_class(
    label_issues_df: pd.DataFrame,
    labels: np.ndarray,
    title: str = "Label Issues per Class",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> "plt.Figure":
    """Bar chart of flagged label issues per age class with issue rate overlay."""
    import matplotlib.pyplot as plt

    classes = np.sort(np.unique(labels))
    class_counts = {cls: np.sum(labels == cls) for cls in classes}

    # Only count issues where the suggested label differs from the given label
    relabel_df = label_issues_df[label_issues_df["suggests_relabel"]]
    issue_counts = relabel_df["given_label"].value_counts().reindex(classes, fill_value=0)
    issue_rates = [issue_counts[cls] / class_counts[cls] if class_counts[cls] > 0 else 0 for cls in classes]

    fig, ax1 = plt.subplots(figsize=figsize)
    x = np.arange(len(classes))
    bars = ax1.bar(x, issue_counts.values, color="steelblue", edgecolor="black", linewidth=0.5, label="Issue Count")
    ax1.set_xlabel("Age Class (Given Label)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Issues", fontsize=12, fontweight="bold", color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(int(c)) for c in classes])
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(x, issue_rates, "o-", color="orangered", linewidth=2, markersize=6, label="Issue Rate")
    ax2.set_ylabel("Issue Rate", fontsize=12, fontweight="bold", color="orangered")
    ax2.tick_params(axis="y", labelcolor="orangered")
    ax2.set_ylim(0, max(issue_rates) * 1.3 if max(issue_rates) > 0 else 1)

    ax1.set_title(title, fontsize=14, fontweight="bold", pad=20)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        from pathlib import Path as _Path
        save_path = _Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved label issues per class to {save_path}")

    return fig


def plot_suggested_label_transitions(
    label_issues_df: pd.DataFrame,
    title: str = "Label Transition Heatmap (Given → Suggested)",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> "plt.Figure":
    """Heatmap of given_label → suggested_label counts for flagged samples."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Only show samples where cleanlab suggests a different label
    relabel_df = label_issues_df[label_issues_df["suggests_relabel"]].copy()

    if len(relabel_df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No relabel suggestions found", ha="center", va="center", transform=ax.transAxes)
        return fig

    all_labels = np.sort(np.unique(np.concatenate([
        relabel_df["given_label"].values,
        relabel_df["suggested_label"].values,
    ])))

    transition_matrix = np.zeros((len(all_labels), len(all_labels)), dtype=int)
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    for _, row in relabel_df.iterrows():
        i = label_to_idx[row["given_label"]]
        j = label_to_idx[row["suggested_label"]]
        transition_matrix[i, j] += 1

    fig, ax = plt.subplots(figsize=figsize)
    tick_labels = [str(int(l)) for l in all_labels]
    sns.heatmap(
        transition_matrix, annot=True, fmt="d", cmap="YlOrRd", square=True,
        linewidths=0.5, cbar_kws={"label": "Count"},
        xticklabels=tick_labels, yticklabels=tick_labels, ax=ax,
    )
    ax.set_xlabel("Suggested Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("Given Label", fontsize=12, fontweight="bold")
    n_total = len(label_issues_df)
    n_relabel = len(relabel_df)
    subtitle = f"{n_relabel} relabel suggestions out of {n_total} flagged issues"
    ax.set_title(f"{title}\n{subtitle}", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved label transitions heatmap to {save_path}")

    return fig


def plot_confidence_distribution(
    label_issues_df: pd.DataFrame,
    pred_probs: np.ndarray,
    labels: np.ndarray,
    title: str = "Confidence Distribution: Flagged vs Non-Flagged",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> "plt.Figure":
    """Histogram comparing confidence_in_given for flagged vs non-flagged samples."""
    import matplotlib.pyplot as plt

    # Compute confidence in given label for all samples
    min_label = labels.min()
    labels_0indexed = labels - min_label
    all_confidence = np.array([pred_probs[i, labels_0indexed[i]] for i in range(len(labels))])

    flagged_indices = set(label_issues_df["sample_index"].values)
    flagged_mask = np.array([i in flagged_indices for i in range(len(labels))])

    fig, ax = plt.subplots(figsize=figsize)
    # Use data range for bins since pseudo-probabilities may be compressed
    lo = max(0, all_confidence.min() - 0.02)
    hi = min(1, all_confidence.max() + 0.02)
    bins = np.linspace(lo, hi, 30)
    ax.hist(all_confidence[~flagged_mask], bins=bins, alpha=0.6, label="Non-flagged",
            color="steelblue", edgecolor="black", linewidth=0.5, density=True)
    ax.hist(all_confidence[flagged_mask], bins=bins, alpha=0.6, label="Flagged",
            color="orangered", edgecolor="black", linewidth=0.5, density=True)

    ax.set_xlabel("Confidence in Given Label (pseudo-probability)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved confidence distribution to {save_path}")

    return fig


def generate_data_quality_report(
    features: np.ndarray,
    labels: np.ndarray,
    measurement_ids: Optional[np.ndarray] = None,
    alpha: float = 0.1,
    n_folds: int = 10,
    output_dir: Optional[Path] = None,
    random_state: int = 42,
) -> Dict:
    """
    Generate comprehensive data quality report using cleanlab.

    Args:
        features: Feature matrix (N, D)
        labels: Label vector (N,)
        measurement_ids: Optional measurement IDs for traceability (N,)
        alpha: Ridge regularization parameter
        n_folds: Number of CV folds
        output_dir: Optional directory to save reports
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with:
            - pred_probs: Predicted probabilities (N, n_classes)
            - oof_predictions: Out-of-fold predictions (N,)
            - label_issues: DataFrame with potential label issues
            - oof_accuracy: Out-of-fold accuracy

    Raises:
        ImportError: If cleanlab is not installed
    """
    if not CLEANLAB_AVAILABLE:
        raise ImportError(
            "cleanlab is not installed. Install with: pip install cleanlab>=2.6.0"
        )

    print("=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)

    # Compute out-of-fold predictions
    pred_probs, oof_predictions = compute_out_of_fold_predictions(
        features=features,
        labels=labels,
        alpha=alpha,
        n_folds=n_folds,
        random_state=random_state,
    )

    # Compute out-of-fold accuracy
    oof_accuracy = np.mean(oof_predictions == labels)
    print(f"\nOut-of-fold accuracy: {oof_accuracy:.4f}")

    # Find label issues
    label_issues_df = find_label_issues_cleanlab(
        features=features,
        labels=labels,
        pred_probs=pred_probs,
        alpha=alpha,
        n_folds=n_folds,
        filter_by="both",
        random_state=random_state,
    )


    # Add measurement IDs if provided
    if measurement_ids is not None and len(label_issues_df) > 0:
        label_issues_df["measurement_id"] = measurement_ids[
            label_issues_df["sample_index"].values
        ]

        
        # Reorder columns
        cols = [
            "sample_index",
            "measurement_id",
            "given_label",
            "suggested_label",
            "confidence_in_given",
            "confidence_in_suggested",
            "suggests_relabel",
        ]
        label_issues_df = label_issues_df[cols]

    # Save to files if output directory provided
    if output_dir and len(label_issues_df) > 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save label issues
        label_issues_df.to_csv(output_dir / "label_issues.csv", index=False)
        print(f"\nSaved label issues to {output_dir / 'label_issues.csv'}")
        
        # Save out-of-fold predictions
        oof_df = pd.DataFrame(
            {
                "sample_index": np.arange(len(labels)),
                "true_label": labels,
                "oof_prediction": oof_predictions,
            }
        )
        if measurement_ids is not None:
            oof_df["measurement_id"] = measurement_ids

        oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
        print(f"Saved out-of-fold predictions to {output_dir / 'oof_predictions.csv'}")

        # Save summary statistics
        summary = {
            "total_samples": len(labels),
            "n_folds": n_folds,
            "alpha": alpha,
            "oof_accuracy": oof_accuracy,
            "n_flagged_samples": len(label_issues_df),
            "n_label_issues": len(label_issues_df[label_issues_df["suggests_relabel"]]),
            "label_issue_rate": len(label_issues_df[label_issues_df["suggests_relabel"]]) / len(labels)
        }
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(output_dir / "data_quality_summary.csv", index=False)
        print(f"Saved summary to {output_dir / 'data_quality_summary.csv'}")

    print("\n" + "=" * 60)

    return {
        "pred_probs": pred_probs,
        "oof_predictions": oof_predictions,
        "label_issues": label_issues_df,
        "oof_accuracy": oof_accuracy,
    }


def check_cleanlab_available() -> bool:
    """
    Check if cleanlab is installed.

    Returns:
        True if cleanlab is available, False otherwise
    """
    return CLEANLAB_AVAILABLE
