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


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute exact accuracy."""
    return accuracy_score(y_true, y_pred)


def compute_accuracy_pm1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute ±1 accuracy (predictions within 1 year of true age).

    This is a key metric for otolith age prediction since being off
    by one year is often acceptable in fisheries research.
    """
    within_one = np.abs(y_true - y_pred) <= 1
    return np.mean(within_one)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_precision(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> float:
    """
    Compute precision with specified averaging.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('macro', 'weighted', 'micro')

    Returns:
        Precision score
    """
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def compute_recall(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> float:
    """
    Compute recall with specified averaging.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('macro', 'weighted', 'micro')

    Returns:
        Recall score
    """
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def compute_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> float:
    """
    Compute F1-score with specified averaging.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('macro', 'weighted', 'micro')

    Returns:
        F1 score
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


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
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "accuracy_pm1": compute_accuracy_pm1(y_true, y_pred),
        "precision": compute_precision(y_true, y_pred, average),
        "recall": compute_recall(y_true, y_pred, average),
        "f1": compute_f1(y_true, y_pred, average),
        "rmse": compute_rmse(y_true, y_pred),
    }


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: True age labels
        y_pred: Predicted age labels

    Returns:
        Dictionary with accuracy, accuracy_pm1, and rmse
    """
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "accuracy_pm1": compute_accuracy_pm1(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
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
        "| Model | Accuracy | ±1 Accuracy | RMSE |",
        "|-------|----------|-------------|------|",
    ]

    for model_name, metrics in results.items():
        acc_mean, acc_std = metrics["accuracy"]
        pm1_mean, pm1_std = metrics["accuracy_pm1"]
        rmse_mean, rmse_std = metrics["rmse"]

        lines.append(
            f"| {model_name} | {acc_mean*100:.2f}±{acc_std*100:.2f}% | "
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
