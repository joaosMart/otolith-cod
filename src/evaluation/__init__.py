"""
Evaluation module for otolith age prediction.

Provides metrics computation, error analysis, learning curves,
data quality analysis, and visualization plots.
"""

from .metrics import (
    compute_classification_metrics,
    aggregate_fold_results,
    format_results_table,
    compare_models_significance,
    bootstrap_metrics,
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

from .feature_importance import (
    build_feature_groups,
    compute_shap_values,
    compute_grouped_permutation_importance,
    run_forward_selection,
    plot_shap_summary,
    plot_shap_dependence,
    plot_shap_dependence_month,
    plot_permutation_importance,
    plot_forward_selection,
)

from .plots import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_aggregated_confusion_matrix,
)

__all__ = [
    # Metrics
    "compute_classification_metrics",
    "aggregate_fold_results",
    "format_results_table",
    "compare_models_significance",
    "bootstrap_metrics",
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
    # Feature importance
    "build_feature_groups",
    "compute_shap_values",
    "compute_grouped_permutation_importance",
    "run_forward_selection",
    "plot_shap_summary",
    "plot_shap_dependence",
    "plot_shap_dependence_month",
    "plot_permutation_importance",
    "plot_forward_selection",
    # Plots
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "plot_aggregated_confusion_matrix",
]
