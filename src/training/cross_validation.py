"""
Cross-validation orchestration for training experiments.

Provides functions to run k-fold or independent-split experiments
using cached embeddings and Ridge classifiers.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from src.data.splits import DataSplit, create_kfold_splits, create_train_test_splits
from src.training.classifiers import train_ridge, predict_ridge
from src.evaluation.metrics import compute_classification_metrics, aggregate_fold_results


def run_kfold_cv(
    features: np.ndarray,
    labels: np.ndarray,
    alpha: float = 6.0,
    n_splits: int = 10,
    random_state: int = 42,
) -> List[Dict]:
    """
    Run k-fold cross-validation and return per-fold metrics.

    Args:
        features: Feature matrix (N, D)
        labels: Label vector (N,)
        alpha: Ridge regularization parameter
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        List of metric dictionaries, one per fold
    """
    splits = create_kfold_splits(labels, n_splits=n_splits, random_state=random_state)
    fold_metrics = []

    for split in tqdm(splits, desc=f"Running {n_splits}-fold CV"):
        X_train = features[split.train_indices]
        y_train = labels[split.train_indices]
        X_test = features[split.test_indices]
        y_test = labels[split.test_indices]

        model = train_ridge(X_train, y_train, alpha=alpha, random_state=random_state)
        y_pred = predict_ridge(model, X_test)
        metrics = compute_classification_metrics(y_test, y_pred)
        fold_metrics.append(metrics)

    return fold_metrics


def run_independent_splits(
    features: np.ndarray,
    labels: np.ndarray,
    splits: List[DataSplit],
    alpha: float = 6.0,
) -> List[Dict]:
    """
    Run experiments on pre-defined splits.

    Args:
        features: Feature matrix (N, D)
        labels: Label vector (N,)
        splits: List of DataSplit objects
        alpha: Ridge regularization parameter

    Returns:
        List of result dicts with metrics, predictions, and model info
    """
    all_results = []

    for split in tqdm(splits, desc="Experiments"):
        X_train = features[split.train_indices]
        y_train = labels[split.train_indices]
        X_test = features[split.test_indices]
        y_test = labels[split.test_indices]

        model = train_ridge(X_train, y_train, alpha=alpha, random_state=split.fold)
        y_pred = predict_ridge(model, X_test)
        y_scores = model.predict(X_test)
        test_metrics = compute_classification_metrics(y_test, y_pred)

        all_results.append({
            "experiment": split.fold,
            "alpha": alpha,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "test_metrics": test_metrics,
            "predictions": y_pred,
            "scores": y_scores,
            "true_labels": y_test,
        })

    return all_results


def save_splits(splits: List[DataSplit], output_path: str, random_state: int = 42) -> None:
    """Save split indices to JSON for reproducibility."""
    splits_data = {
        "n_experiments": len(splits),
        "random_state": random_state,
        "splits": [
            {
                "experiment": split.fold,
                "train_indices": split.train_indices.tolist(),
                "test_indices": split.test_indices.tolist(),
            }
            for split in splits
        ],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits_data, f, indent=2)
    print(f"Saved splits to {path}")


def load_splits(splits_path: str) -> List[DataSplit]:
    """Load split indices from JSON."""
    with open(splits_path, "r") as f:
        data = json.load(f)

    splits = []
    for s in data["splits"]:
        splits.append(DataSplit(
            train_indices=np.array(s["train_indices"]),
            test_indices=np.array(s["test_indices"]),
            fold=s["experiment"],
        ))
    return splits
