"""
Evaluation Metrics Module.

Computes accuracy, ±1 accuracy, and RMSE for age prediction.
Matches the paper's evaluation methodology.

Based on: Sigurðardóttir et al. (2023) - Ecological Informatics
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for precision/recall/f1

    Returns:
        Dictionary with accuracy, accuracy_pm1, precision, recall, f1, rmse
    """
    within_one = np.abs(y_true - y_pred) <= 1

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "accuracy_pm1": float(np.mean(within_one)),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def aggregate_fold_results(fold_metrics: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate metrics across CV folds.

    Args:
        fold_metrics: List of metric dictionaries from each fold

    Returns:
        Dictionary with (mean, std) for each metric
    """
    metrics_array = {
        key: [fm[key] for fm in fold_metrics] for key in fold_metrics[0].keys()
    }

    return {
        key: (np.mean(values), np.std(values))
        for key, values in metrics_array.items()
    }


def format_results_table(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    paper_reference: Dict[str, float] = None,
) -> str:
    """
    Format results as a markdown table.

    Args:
        results: Dictionary mapping model_name -> metrics
        paper_reference: Optional reference values from the paper

    Returns:
        Formatted markdown table string
    """
    lines = [
        "| Model | F1 (Macro) | Accuracy | ±1 Accuracy | RMSE |",
        "|-------|------------|----------|-------------|------|",
    ]

    for model_name, metrics in results.items():
        f1_mean, f1_std = metrics["f1"]
        acc_mean, acc_std = metrics["accuracy"]
        pm1_mean, pm1_std = metrics["accuracy_pm1"]
        rmse_mean, rmse_std = metrics["rmse"]

        lines.append(
            f"| {model_name} | {f1_mean*100:.2f}±{f1_std*100:.2f}% | {acc_mean*100:.2f}±{acc_std*100:.2f}% | "
            f"{pm1_mean*100:.2f}±{pm1_std*100:.2f}% | {rmse_mean:.3f}±{rmse_std:.3f} |"
        )

    if paper_reference:
        lines.append(
            f"| Paper (cod) | {paper_reference['accuracy']:.2f}±{paper_reference['accuracy_std']:.2f}% | "
            f"{paper_reference['accuracy_pm1']:.2f}±{paper_reference['accuracy_pm1_std']:.2f}% | "
            f"{paper_reference['rmse']:.2f}±{paper_reference['rmse_std']:.2f} |"
        )

    return "\n".join(lines)


def compare_models_significance(
    model1_scores: List[float],
    model2_scores: List[float],
    metric_name: str = "accuracy",
) -> Dict[str, float]:
    """
    Perform paired t-test to compare two models across CV folds.

    Args:
        model1_scores: Metric scores for model 1 across folds
        model2_scores: Metric scores for model 2 across folds
        metric_name: Name of the metric being compared

    Returns:
        Dictionary with t-statistic, p-value, and mean difference
    """
    from scipy import stats

    scores1 = np.array(model1_scores)
    scores2 = np.array(model2_scores)

    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    mean_diff = np.mean(scores2 - scores1)

    return {
        "metric": metric_name,
        "t_statistic": t_stat,
        "p_value": p_value,
        "mean_difference": mean_diff,
        "model2_better": mean_diff > 0,
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for classification metrics.

    Resamples test predictions with replacement and computes metrics on each
    resample. Returns point estimate (original) and CI bounds.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_bootstrap: Number of bootstrap resamples
        ci: Confidence interval level (e.g. 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Dict with same keys as compute_classification_metrics. Each value is
        {"mean": float, "ci_lower": float, "ci_upper": float}.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    alpha = (1 - ci) / 2

    # Point estimate on full test set
    point_metrics = compute_classification_metrics(y_true, y_pred)

    # Bootstrap resamples
    bootstrap_values = {key: [] for key in point_metrics}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample_metrics = compute_classification_metrics(y_true[idx], y_pred[idx])
        for key, val in sample_metrics.items():
            bootstrap_values[key].append(val)

    result = {}
    for key in point_metrics:
        values = np.array(bootstrap_values[key])
        result[key] = {
            "mean": point_metrics[key],
            "ci_lower": float(np.percentile(values, alpha * 100)),
            "ci_upper": float(np.percentile(values, (1 - alpha) * 100)),
        }

    return result
