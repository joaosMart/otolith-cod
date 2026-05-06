"""
Training module for otolith age prediction.

Provides classifier creation, cross-validation, and hyperparameter search.
"""

from .classifiers import (
    create_ridge,
    create_ridge_classifier,
    create_svc,
    train_ridge,
    predict_ridge,
)
from .cross_validation import (
    run_kfold_cv,
    run_independent_splits,
    save_splits,
    load_splits,
)
from .hyperparameter_search import (
    grid_search_alpha,
    run_experiment_with_search,
)

__all__ = [
    "create_ridge",
    "create_ridge_classifier",
    "create_svc",
    "train_ridge",
    "predict_ridge",
    "run_kfold_cv",
    "run_independent_splits",
    "save_splits",
    "load_splits",
    "grid_search_alpha",
    "run_experiment_with_search",
]
