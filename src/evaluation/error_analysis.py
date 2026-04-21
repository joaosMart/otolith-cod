"""Error analysis utilities for classification models.

Functions for identifying and analyzing misclassifications:
- Identify all misclassified samples with error details
- Filter large errors (> ±1 class difference)
- Visualize error distributions
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def identify_misclassified_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    measurement_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Identify all misclassified samples with detailed error information.

    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        measurement_ids: Optional measurement IDs for traceability (N,)

    Returns:
        DataFrame with columns:
            - measurement_id (if provided)
            - true_label
            - predicted_label
            - error (true - predicted)
            - abs_error
            - is_large_error (abs_error > 1)
    """
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    misclassified_mask = errors != 0

    results = {
        "true_label": y_true[misclassified_mask],
        "predicted_label": y_pred[misclassified_mask],
        "error": errors[misclassified_mask],
        "abs_error": abs_errors[misclassified_mask],
        "is_large_error": abs_errors[misclassified_mask] > 1,
    }

    if measurement_ids is not None:
        results["measurement_id"] = measurement_ids[misclassified_mask]
        # Reorder columns to put measurement_id first
        column_order = [
            "measurement_id",
            "true_label",
            "predicted_label",
            "error",
            "abs_error",
            "is_large_error",
        ]
        df = pd.DataFrame(results)[column_order]
    else:
        df = pd.DataFrame(results)

    # Sort by absolute error (descending)
    df = df.sort_values("abs_error", ascending=False).reset_index(drop=True)

    return df


def get_large_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    measurement_ids: Optional[np.ndarray] = None,
    threshold: int = 1,
) -> pd.DataFrame:
    """
    Get samples with large prediction errors (> threshold).

    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        measurement_ids: Optional measurement IDs (N,)
        threshold: Error threshold (default: 1, meaning abs_error > 1)

    Returns:
        DataFrame with large error samples, sorted by absolute error
    """
    df = identify_misclassified_samples(y_true, y_pred, measurement_ids)
    large_errors = df[df["abs_error"] > threshold].copy()

    return large_errors


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Error Distribution",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot error distribution across predicted vs actual classes.

    Creates a 2D histogram showing:
    - X-axis: Predicted class
    - Y-axis: True class
    - Color intensity: Error magnitude

    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    errors = y_true - y_pred

    # Create error histogram
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    # Build error matrix
    error_matrix = np.zeros((n_classes, n_classes))
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            mask = (y_true == true_cls) & (y_pred == pred_cls)
            error_matrix[i, j] = np.sum(mask)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        error_matrix,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Number of Samples"},
        xticklabels=[str(int(cls)) for cls in classes],
        yticklabels=[str(int(cls)) for cls in classes],
        ax=ax,
    )

    # Highlight diagonal (correct predictions)
    for i in range(n_classes):
        ax.add_patch(
            plt.Rectangle(
                (i, i), 1, 1, fill=False, edgecolor="blue", lw=2, linestyle="--"
            )
        )

    ax.set_xlabel("Predicted Age", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Age", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved error distribution to {save_path}")

    return fig


def plot_error_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Prediction Error Histogram",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot histogram of prediction errors.

    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    errors = y_true - y_pred

    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    bins = np.arange(errors.min() - 0.5, errors.max() + 1.5, 1)
    counts, _, patches = ax.hist(errors, bins=bins, edgecolor="black", alpha=0.7)

    # Color bars based on error magnitude
    for i, patch in enumerate(patches):
        error_val = bins[i] + 0.5
        if abs(error_val) <= 1:
            patch.set_facecolor("green")
        elif abs(error_val) <= 2:
            patch.set_facecolor("orange")
        else:
            patch.set_facecolor("red")

    # Add statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    mae = np.mean(np.abs(errors))

    stats_text = f"Mean: {mean_error:.2f}\nStd: {std_error:.2f}\nMAE: {mae:.2f}"
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Prediction Error (True - Predicted)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.grid(axis="y", alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", label="±1 error"),
        Patch(facecolor="orange", label="±2 error"),
        Patch(facecolor="red", label=">±2 error"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved error histogram to {save_path}")

    return fig


def plot_per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Per-Class F1 Score",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot F1 score per age class as a bar chart, sorted and color-coded."""
    from sklearn.metrics import f1_score

    classes = np.unique(y_true)
    # Per-class F1 via sklearn (returns one value per class)
    f1_per_class = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

    # Sort by F1
    order = np.argsort(f1_per_class)
    sorted_classes = classes[order]
    sorted_f1 = f1_per_class[order]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.RdYlGn(sorted_f1)
    ax.barh(range(len(sorted_classes)), sorted_f1, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(sorted_classes)))
    ax.set_yticklabels([str(int(c)) for c in sorted_classes])
    ax.set_xlabel("F1 Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Age Class", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, v in enumerate(sorted_f1):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved per-class F1 to {save_path}")

    return fig


def plot_per_class_error_magnitude(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Error Distribution per Class (When Wrong)",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot error distribution per class for misclassified samples only."""
    classes = np.unique(y_true)
    error_data = []
    counts = []
    for cls in classes:
        mask = (y_true == cls) & (y_pred != cls)
        if mask.sum() > 0:
            error_data.append(np.abs(y_true[mask] - y_pred[mask]))
            counts.append(mask.sum())
        else:
            error_data.append(np.array([]))
            counts.append(0)

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(
        [d for d in error_data if len(d) > 0],
        positions=[int(c) for c, d in zip(classes, error_data) if len(d) > 0],
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="#e74c3c", alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    ax.set_xlabel("Age Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("Absolute Error", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks([int(c) for c in classes])
    ax.set_xticklabels([str(int(c)) for c in classes])
    ax.grid(axis="y", alpha=0.3)

    # Add sample count labels above each box
    for cls, count, d in zip(classes, counts, error_data):
        if len(d) > 0:
            ax.text(int(cls), ax.get_ylim()[1] * 0.97, f"n={count}",
                    ha="center", va="top", fontsize=8)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved per-class error magnitude to {save_path}")

    return fig


def plot_most_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_n: int = 15,
    title: str = "Most Confused Class Pairs",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot top-N most frequent (true -> predicted) confusion pairs."""
    # Count confusion pairs (excluding correct predictions)
    mask = y_true != y_pred
    pairs = list(zip(y_true[mask], y_pred[mask]))
    pair_counts = {}
    for pair in pairs:
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Sort and take top N
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    if not sorted_pairs:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No misclassifications", ha="center", va="center", transform=ax.transAxes)
        return fig

    labels = [f"{int(t)} → {int(p)}" for (t, p), _ in sorted_pairs]
    values = [c for _, c in sorted_pairs]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(values)))[::-1]
    ax.barh(range(len(labels)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Count", fontsize=12, fontweight="bold")
    ax.set_ylabel("True → Predicted", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    for i, v in enumerate(values):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=9)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved most confused pairs to {save_path}")

    return fig


def plot_f1_vs_frequency(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "F1 Score vs Class Frequency",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Scatter plot of F1 score vs sample count per class."""
    from sklearn.metrics import f1_score

    classes = np.unique(y_true)
    f1_per_class = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    counts = [np.sum(y_true == cls) for cls in classes]

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(counts, f1_per_class, c=f1_per_class, cmap="RdYlGn",
                         s=120, edgecolors="black", linewidth=0.5, vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax, label="F1 Score")

    for cls, cnt, f1 in zip(classes, counts, f1_per_class):
        ax.annotate(str(int(cls)), (cnt, f1), textcoords="offset points",
                    xytext=(8, 4), fontsize=10, fontweight="bold")

    ax.set_xlabel("Sample Count", fontsize=12, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved F1 vs frequency to {save_path}")

    return fig


def generate_error_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    measurement_ids: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Generate comprehensive error analysis report.

    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        measurement_ids: Optional measurement IDs (N,)
        output_dir: Optional directory to save CSV files

    Returns:
        Dictionary with error statistics and dataframes
    """
    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    # Compute statistics
    stats = {
        "total_samples": len(y_true),
        "correct_predictions": np.sum(errors == 0),
        "misclassified": np.sum(errors != 0),
        "accuracy": np.mean(errors == 0),
        "accuracy_pm1": np.mean(abs_errors <= 1),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "mae": float(np.mean(abs_errors)),
        "large_errors": np.sum(abs_errors > 1),
        "large_error_rate": np.mean(abs_errors > 1),
    }

    # Get misclassified samples
    misclassified_df = identify_misclassified_samples(y_true, y_pred, measurement_ids)
    large_errors_df = get_large_errors(y_true, y_pred, measurement_ids)

    # Save to CSV if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        misclassified_df.to_csv(
            output_dir / "misclassified_samples.csv", index=False
        )
        large_errors_df.to_csv(output_dir / "large_errors.csv", index=False)

        # Save statistics
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(output_dir / "error_statistics.csv", index=False)

        print(f"Saved error reports to {output_dir}")

    return {
        "statistics": stats,
        "misclassified_samples": misclassified_df,
        "large_errors": large_errors_df,
    }
