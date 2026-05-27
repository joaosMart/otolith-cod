"""
Data Splitting Utilities.

This module provides functions for creating stratified k-fold cross-validation
splits, following the paper's methodology.

Paper methodology:
- 10-fold cross-validation for robust evaluation
- Stratified by age class

Based on: Sigurðardóttir et al. (2023) - Ecological Informatics
"""

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class DataSplit:
    """Container for a single data split."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    fold: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"DataSplit(train={len(self.train_indices)}, "
            f"test={len(self.test_indices)}, "
            f"fold={self.fold})"
        )


def create_kfold_splits(
    labels: np.ndarray,
    n_splits: int = 10,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> List[DataSplit]:
    """
    Create k-fold stratified cross-validation splits.

    For each fold:
    - Test set is the held-out fold (~10% for 10-fold)
    - Remaining 90% is split into train/val (maintaining 65/15 ratio of total)

    The paper uses 10-fold CV and reports mean ± std across folds.

    Args:
        labels: Array of age labels for all samples
        n_splits: Number of CV folds (default: 10)
        val_ratio: Validation ratio from non-test data (default: 0.15)
        random_state: Random seed for reproducibility

    Returns:
        List of DataSplit objects, one per fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = np.arange(len(labels))

    splits = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(indices, labels)):
        # Test is approximately 1/n_splits of data
        # From remaining (train + val), we want val to be ~15% of total
        # So val_ratio within train_val = 0.15 / (1 - 1/n_splits)
        # For 10-fold: val_ratio = 0.15 / 0.9 ≈ 0.167
        val_ratio_adjusted = val_ratio / (1 - 1 / n_splits)

        # Split train_val into train and validation
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio_adjusted,
            stratify=labels[train_val_idx],
            random_state=random_state + fold_idx,  # Different seed per fold
        )

        splits.append(
            DataSplit(
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_idx,
                fold=fold_idx,
            )
        )

    return splits


def create_train_test_splits(
    labels: np.ndarray,
    n_experiments: int = 10,
    train_ratio: float = 0.85,
    test_ratio: float = 0.15,
    random_state: int = 42,
    seeds: Optional[List[int]] = None,
) -> List[DataSplit]:
    """
    Create n independent stratified train/test splits.

    Each split is independent (samples can appear in test sets of multiple splits).

    Args:
        labels: Array of class labels for all samples
        n_experiments: Number of independent splits to create (default: 10).
            Ignored when seeds is provided (inferred from len(seeds)).
        train_ratio: Proportion for training (default: 0.85)
        test_ratio: Proportion for testing (default: 0.15)
        random_state: Base random seed for reproducibility
        seeds: Optional list of explicit seeds, one per experiment.
            When provided, n_experiments is inferred from len(seeds).

    Returns:
        List of DataSplit objects, one per experiment
    """
    assert abs(train_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio + test_ratio}"
    )

    if seeds is not None:
        n_experiments = len(seeds)

    splits = []
    indices = np.arange(len(labels))

    for exp_idx in range(n_experiments):
        seed = seeds[exp_idx] if seeds is not None else random_state + exp_idx

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=seed
        )
        train_idx, test_idx = next(sss.split(indices, labels))

        splits.append(
            DataSplit(
                train_indices=train_idx,
                test_indices=test_idx,
                fold=exp_idx,
            )
        )

    return splits


def create_fixed_split(
    labels: np.ndarray,
    measurement_ids: np.ndarray,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Create a single stratified train/test split, returned as measurement IDs."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_idx, test_idx = next(sss.split(np.arange(len(labels)), labels))
    return {
        "seed": seed,
        "test_ratio": test_ratio,
        "train_measurement_ids": measurement_ids[train_idx].tolist(),
        "test_measurement_ids": measurement_ids[test_idx].tolist(),
    }


def save_split_by_ids(path: str, split_dict: dict) -> None:
    """Save a fixed split (by measurement IDs) to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(split_dict, f, indent=2)
    print(f"Saved fixed split to {p}")


def load_split_by_ids(path: str, measurement_ids: np.ndarray) -> DataSplit:
    """Load a fixed split from JSON and map measurement IDs to current array indices."""
    with open(path, "r") as f:
        data = json.load(f)

    id_to_idx = {int(mid): i for i, mid in enumerate(measurement_ids)}

    train_ids = data["train_measurement_ids"]
    test_ids = data["test_measurement_ids"]

    missing_train = [mid for mid in train_ids if mid not in id_to_idx]
    missing_test = [mid for mid in test_ids if mid not in id_to_idx]
    if missing_train or missing_test:
        raise ValueError(
            f"Split file references {len(missing_train)} train and "
            f"{len(missing_test)} test measurement IDs not found in embeddings."
        )

    train_indices = np.array([id_to_idx[mid] for mid in train_ids])
    test_indices = np.array([id_to_idx[mid] for mid in test_ids])

    return DataSplit(train_indices=train_indices, test_indices=test_indices, fold=0)
