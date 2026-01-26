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

from .error_analysis import (
    identify_misclassified_samples,
    get_large_errors,
    plot_error_distribution,
    plot_error_histogram,
    generate_error_report,
)

from .learning_curves import (
    compute_learning_curve,
    run_learning_curve_experiment,
    plot_learning_curve,
    plot_multiple_learning_curves,
)

from .data_quality import (
    compute_out_of_fold_predictions,
    find_label_issues_cleanlab,
    generate_data_quality_report,
    check_cleanlab_available,
)

__all__ = [
    # Metrics
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
    # Error analysis
    "identify_misclassified_samples",
    "get_large_errors",
    "plot_error_distribution",
    "plot_error_histogram",
    "generate_error_report",
    # Learning curves
    "compute_learning_curve",
    "run_learning_curve_experiment",
    "plot_learning_curve",
    "plot_multiple_learning_curves",
    # Data quality
    "compute_out_of_fold_predictions",
    "find_label_issues_cleanlab",
    "generate_data_quality_report",
    "check_cleanlab_available",
]
