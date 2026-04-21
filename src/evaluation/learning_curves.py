"""Learning curve analysis for assessing data efficiency.

Functions for computing and visualizing learning curves by training models
on varying fractions of the training data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.data import DataSplit
from src.evaluation import compute_classification_metrics


def compute_learning_curve(
    features: np.ndarray,
    labels: np.ndarray,
    split: DataSplit,
    train_fractions: Optional[List[float]] = None,
    alpha: float = 0.1,
    random_state: int = 42,
) -> Dict[str, List[float]]:
    """
    Compute learning curve for a single data split.

    Trains Ridge models on varying fractions of training data and evaluates
    on a fixed test set. Uses stratified sampling to maintain class balance.

    Args:
        features: Feature matrix (N, D)
        labels: Label vector (N,)
        split: DataSplit object with train/val/test indices
        train_fractions: List of training data fractions (default: [0.1, 0.2, ..., 1.0])
        alpha: Ridge regularization parameter
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with keys:
            - train_sizes: List of training set sizes used
            - train_accuracy: Accuracy on training subset
            - test_accuracy: Accuracy on fixed test set
            - test_accuracy_pm1: ±1 accuracy on test set
            - test_f1: Macro F1 score on test set
            - test_rmse: RMSE on test set
    """
    if train_fractions is None:
        train_fractions = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    X_train_raw = features[split.train_indices]
    y_train = labels[split.train_indices]
    X_test_raw = features[split.test_indices]
    y_test = labels[split.test_indices]

    # Scale features (fit on full train set)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    results = {
        "train_sizes": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "test_accuracy_pm1": [],
        "test_f1": [],
        "test_rmse": [],
    }

    rng = np.random.RandomState(random_state)

    for fraction in train_fractions:
        # Stratified subsampling
        n_samples = int(len(X_train) * fraction)

        # Get stratified indices
        unique_classes = np.unique(y_train)
        subset_indices = []

        for cls in unique_classes:
            cls_indices = np.where(y_train == cls)[0]
            n_cls_samples = max(1, int(len(cls_indices) * fraction))
            sampled = rng.choice(cls_indices, size=n_cls_samples, replace=False)
            subset_indices.extend(sampled)

        subset_indices = np.array(subset_indices)
        rng.shuffle(subset_indices)

        X_subset = X_train[subset_indices]
        y_subset = y_train[subset_indices]

        # Train model
        model = Ridge(alpha=alpha)
        model.fit(X_subset, y_subset)

        # Evaluate on training subset
        y_train_pred = np.clip(np.round(model.predict(X_subset)).astype(int), 1, 10)
        train_accuracy = np.mean(y_train_pred == y_subset)

        # Evaluate on test set
        y_test_pred = np.clip(np.round(model.predict(X_test)).astype(int), 1, 10)
        test_metrics = compute_classification_metrics(y_test, y_test_pred)

        results["train_sizes"].append(len(X_subset))
        results["train_accuracy"].append(train_accuracy)
        results["test_accuracy"].append(test_metrics["accuracy"])
        results["test_accuracy_pm1"].append(test_metrics["accuracy_pm1"])
        results["test_f1"].append(test_metrics["f1"])
        results["test_rmse"].append(test_metrics["rmse"])

    return results


def run_learning_curve_experiment(
    features: np.ndarray,
    labels: np.ndarray,
    splits: List[DataSplit],
    train_fractions: Optional[List[float]] = None,
    alpha: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    Run learning curve analysis across multiple data splits.

    Args:
        features: Feature matrix (N, D)
        labels: Label vector (N,)
        splits: List of DataSplit objects
        train_fractions: List of training data fractions
        alpha: Ridge regularization parameter

    Returns:
        Dictionary with aggregated results across splits:
            - train_sizes: Array of training sizes (n_fractions,)
            - test_accuracy_mean: Mean test accuracy (n_fractions,)
            - test_accuracy_std: Std test accuracy (n_fractions,)
            - test_accuracy_pm1_mean: Mean ±1 accuracy (n_fractions,)
            - test_accuracy_pm1_std: Std ±1 accuracy (n_fractions,)
            - test_f1_mean: Mean F1 score (n_fractions,)
            - test_f1_std: Std F1 score (n_fractions,)
    """
    all_results = []

    for split in tqdm(splits, desc="Computing learning curves"):
        result = compute_learning_curve(
            features=features,
            labels=labels,
            split=split,
            train_fractions=train_fractions,
            alpha=alpha,
            random_state=split.fold,
        )
        all_results.append(result)

    # Aggregate results
    train_sizes = all_results[0]["train_sizes"]

    test_accuracy_all = np.array([r["test_accuracy"] for r in all_results])
    test_accuracy_pm1_all = np.array([r["test_accuracy_pm1"] for r in all_results])
    test_f1_all = np.array([r["test_f1"] for r in all_results])

    return {
        "train_sizes": np.array(train_sizes),
        "test_accuracy_mean": np.mean(test_accuracy_all, axis=0),
        "test_accuracy_std": np.std(test_accuracy_all, axis=0),
        "test_accuracy_pm1_mean": np.mean(test_accuracy_pm1_all, axis=0),
        "test_accuracy_pm1_std": np.std(test_accuracy_pm1_all, axis=0),
        "test_f1_mean": np.mean(test_f1_all, axis=0),
        "test_f1_std": np.std(test_f1_all, axis=0),
    }


def plot_learning_curve(
    results: Dict[str, np.ndarray],
    metric: str = "accuracy",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot learning curve with confidence bands.

    Args:
        results: Dictionary from run_learning_curve_experiment()
        metric: Metric to plot ("accuracy", "accuracy_pm1", or "f1")
        title: Plot title (default: auto-generated)
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    metric_names = {
        "accuracy": "Accuracy",
        "accuracy_pm1": "±1 Accuracy",
        "f1": "F1 Score (Macro)",
    }

    if title is None:
        title = f"Learning Curve - {metric_names[metric]}"

    train_sizes = results["train_sizes"]
    mean_key = f"test_{metric}_mean"
    std_key = f"test_{metric}_std"

    mean = results[mean_key]
    std = results[std_key]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot mean with confidence band
    ax.plot(train_sizes, mean, "o-", linewidth=2, markersize=8, label="Mean")
    ax.fill_between(
        train_sizes,
        mean - std,
        mean + std,
        alpha=0.2,
        label="±1 Std Dev",
    )

    ax.set_xlabel("Training Set Size", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_names[metric], fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # Set y-axis limits based on metric
    if metric in ["accuracy", "accuracy_pm1"]:
        ax.set_ylim([0, 1.05])
    elif metric == "f1":
        ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved learning curve to {save_path}")

    return fig


def plot_multiple_learning_curves(
    results: Dict[str, np.ndarray],
    metrics: List[str] = None,
    title: str = "Learning Curves",
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot multiple learning curves side by side.

    Args:
        results: Dictionary from run_learning_curve_experiment()
        metrics: List of metrics to plot (default: ["accuracy", "accuracy_pm1", "f1"])
        title: Overall plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    if metrics is None:
        metrics = ["accuracy", "accuracy_pm1", "f1"]

    metric_names = {
        "accuracy": "Accuracy",
        "accuracy_pm1": "±1 Accuracy",
        "f1": "F1 Score (Macro)",
    }

    train_sizes = results["train_sizes"]

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        mean_key = f"test_{metric}_mean"
        std_key = f"test_{metric}_std"

        mean = results[mean_key]
        std = results[std_key]

        ax.plot(train_sizes, mean, "o-", linewidth=2, markersize=8, label="Mean")
        ax.fill_between(
            train_sizes,
            mean - std,
            mean + std,
            alpha=0.2,
            label="±1 Std Dev",
        )

        ax.set_xlabel("Training Set Size", fontsize=11, fontweight="bold")
        ax.set_ylabel(metric_names[metric], fontsize=11, fontweight="bold")
        ax.set_title(metric_names[metric], fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)

        if metric in ["accuracy", "accuracy_pm1", "f1"]:
            ax.set_ylim([0, 1.05])

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved learning curves to {save_path}")

    return fig
