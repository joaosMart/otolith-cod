"""
Data processing module for otolith age prediction.

Provides:
- OtolithDataset: PyTorch dataset for loading otolith images
- Stratified k-fold cross-validation splitting
"""

from .dataset import OtolithDataset, SUPPORTED_DATA
from .splits import (
    DataSplit,
    create_kfold_splits,
    create_train_test_splits,
    create_fixed_split,
    save_split_by_ids,
    load_split_by_ids,
)

__all__ = [
    "OtolithDataset",
    "SUPPORTED_DATA",
    "DataSplit",
    "create_kfold_splits",
    "create_train_test_splits",
    "create_fixed_split",
    "save_split_by_ids",
    "load_split_by_ids",
]
