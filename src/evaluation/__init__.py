"""
Evaluation module for otolith age prediction.

Provides metrics computation and result formatting.
"""

from .metrics import (
    compute_accuracy,
    compute_accuracy_pm1,
    compute_rmse,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_classification_metrics,
    compute_all_metrics,
    aggregate_fold_results,
    format_results_table,
    compare_models_significance,
)

__all__ = [
    "compute_accuracy",
    "compute_accuracy_pm1",
    "compute_rmse",
    "compute_precision",
    "compute_recall",
    "compute_f1",
    "compute_classification_metrics",
    "compute_all_metrics",
    "aggregate_fold_results",
    "format_results_table",
    "compare_models_significance",
]
