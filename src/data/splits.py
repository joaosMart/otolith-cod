"""
Data Splitting Utilities.

This module provides functions for creating stratified k-fold cross-validation
splits, following the paper's methodology.

Paper methodology:
- 10-fold cross-validation for robust evaluation
- Stratified by age class

Based on: Sigurðardóttir et al. (2023) - Ecological Informatics
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class DataSplit:
    """Container for a single data split."""

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    fold: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"DataSplit(train={len(self.train_indices)}, "
            f"val={len(self.val_indices)}, "
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


def create_train_val_test_splits(
    labels: np.ndarray,
    n_experiments: int = 10,
    train_ratio: float = 0.65,
    val_ratio: float = 0.15,
    test_ratio: float = 0.20,
    random_state: int = 42,
) -> List[DataSplit]:
    """
    Create n independent stratified train/val/test splits.

    Unlike k-fold CV, each split is independent (samples can appear
    in test sets of multiple splits). Follows paper's methodology
    with fixed ratios rather than fold-based partitioning.

    Args:
        labels: Array of class labels for all samples
        n_experiments: Number of independent splits to create (default: 10)
        train_ratio: Proportion for training (default: 0.65)
        val_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for testing (default: 0.20)
        random_state: Base random seed for reproducibility

    Returns:
        List of DataSplit objects, one per experiment

    Raises:
        AssertionError: If ratios don't sum to 1.0
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    splits = []
    indices = np.arange(len(labels))

    for exp_idx in range(n_experiments):
        seed = random_state + exp_idx

        # First split: separate test set
        sss_test = StratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=seed
        )
        train_val_idx, test_idx = next(sss_test.split(indices, labels))

        # Second split: separate train and val from remaining data
        # Adjust val_size for the remaining (train + val) portion
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        sss_val = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size_adjusted, random_state=seed
        )
        train_idx_local, val_idx_local = next(
            sss_val.split(train_val_idx, labels[train_val_idx])
        )

        # Map back to original indices
        train_idx = train_val_idx[train_idx_local]
        val_idx = train_val_idx[val_idx_local]

        splits.append(
            DataSplit(
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_idx,
                fold=exp_idx,
            )
        )

    return splits
