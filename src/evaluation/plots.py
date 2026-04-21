"""Classification visualization utilities.

Functions for generating publication-quality plots for multi-class classification:
- Confusion matrices with normalization options
- ROC curves using One-vs-Rest strategy
- Precision-Recall curves for each class
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Optional[List[str]] = None,
    normalize: str = "true",
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot confusion matrix with optional normalization.

    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        class_labels: List of class names (default: ["1", "2", ..., "10"])
        normalize: Normalization mode:
            - "true": Normalize over true labels (rows sum to 1)
            - "pred": Normalize over predictions (columns sum to 1)
            - "all": Normalize over all samples
            - None: No normalization (raw counts)
        title: Plot title
        cmap: Colormap name
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    if class_labels is None:
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        class_labels = [str(int(label)) for label in unique_labels]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Apply normalization
    if normalize == "true":
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        cbar_label = "Recall (True Positive Rate)"
    elif normalize == "pred":
        cm = cm.astype(float) / cm.sum(axis=0, keepdims=True)
        fmt = ".2f"
        cbar_label = "Precision"
    elif normalize == "all":
        cm = cm.astype(float) / cm.sum()
        fmt = ".2f"
        cbar_label = "Proportion of Total"
    else:
        fmt = "d"
        cbar_label = "Count"

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": cbar_label},
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
    )

    ax.set_xlabel("Predicted Age", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Age", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")

    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_labels: Optional[List[str]] = None,
    title: str = "ROC Curves (One-vs-Rest)",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification using One-vs-Rest strategy.

    Args:
        y_true: True labels (N,)
        y_scores: Continuous prediction scores (N, n_classes) or (N,) for regression
            If 1D, will be converted to scores by treating as ordinal predictions
        class_labels: List of class names (default: ["1", "2", ..., "10"])
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object

    Note:
        For Ridge regression, y_scores should be the continuous predictions.
        We compute class scores using distance from each class center.
    """
    # Handle 1D scores from regression
    if y_scores.ndim == 1:
        # Convert continuous predictions to class scores
        # For each class, score = -|prediction - class|
        classes = np.unique(y_true)
        n_classes = len(classes)
        scores = np.zeros((len(y_scores), n_classes))
        for i, cls in enumerate(classes):
            scores[:, i] = -np.abs(y_scores - cls)
        y_scores = scores
    else:
        classes = np.arange(y_scores.shape[1])

    if class_labels is None:
        class_labels = [str(int(cls)) for cls in classes]

    # Binarize labels for One-vs-Rest
    y_true_bin = label_binarize(y_true, classes=classes)

    # Compute ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i, cls in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot micro-average
    ax.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"Micro-average (AUC = {roc_auc['micro']:.3f})",
        color="deeppink",
        linestyle="--",
        linewidth=2,
    )

    # Plot per-class curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    for i, (cls, color) in enumerate(zip(classes, colors)):
        ax.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=1.5,
            label=f"Age {class_labels[i]} (AUC = {roc_auc[i]:.3f})",
        )

    # Plot diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved ROC curves to {save_path}")

    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_labels: Optional[List[str]] = None,
    title: str = "Precision-Recall Curves",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multi-class classification.

    Args:
        y_true: True labels (N,)
        y_scores: Continuous prediction scores (N, n_classes) or (N,) for regression
        class_labels: List of class names (default: ["1", "2", ..., "10"])
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    # Handle 1D scores from regression
    if y_scores.ndim == 1:
        classes = np.unique(y_true)
        n_classes = len(classes)
        scores = np.zeros((len(y_scores), n_classes))
        for i, cls in enumerate(classes):
            scores[:, i] = -np.abs(y_scores - cls)
        y_scores = scores
    else:
        classes = np.arange(y_scores.shape[1])

    if class_labels is None:
        class_labels = [str(int(cls)) for cls in classes]

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=classes)

    # Compute PR curve for each class
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i, cls in enumerate(classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_scores[:, i]
        )
        average_precision[i] = auc(recall[i], precision[i])

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    for i, (cls, color) in enumerate(zip(classes, colors)):
        ax.plot(
            recall[i],
            precision[i],
            color=color,
            lw=1.5,
            label=f"Age {class_labels[i]} (AP = {average_precision[i]:.3f})",
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved precision-recall curves to {save_path}")

    return fig


def plot_aggregated_confusion_matrix(
    all_y_true: List[np.ndarray],
    all_y_pred: List[np.ndarray],
    class_labels: Optional[List[str]] = None,
    normalize: str = "true",
    title: str = "Aggregated Confusion Matrix (10 Experiments)",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot aggregated confusion matrix across multiple experiments.

    Args:
        all_y_true: List of true label arrays from each experiment
        all_y_pred: List of predicted label arrays from each experiment
        class_labels: List of class names
        normalize: Normalization mode
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    # Concatenate all predictions
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    return plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_labels=class_labels,
        normalize=normalize,
        title=title,
        figsize=figsize,
        save_path=save_path,
        dpi=dpi,
    )
