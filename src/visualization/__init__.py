"""
Visualization module for otolith age prediction.

Provides plotting utilities for results visualization.
"""

from .classification_plots import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_aggregated_confusion_matrix,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "plot_aggregated_confusion_matrix",
]
