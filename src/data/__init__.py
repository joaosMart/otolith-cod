"""
Data processing module for otolith age prediction.

This module provides:
- OtolithDataset: PyTorch dataset for loading otolith images
- Stratified k-fold cross-validation splitting
"""

from .dataset import OtolithDataset
from .splits import DataSplit, create_kfold_splits, create_train_val_test_splits

__all__ = [
    "OtolithDataset",
    "DataSplit",
    "create_kfold_splits",
    "create_train_val_test_splits",
]
