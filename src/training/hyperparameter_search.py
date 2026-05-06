"""
Hyperparameter search utilities.

RidgeCV for Ridge alpha optimization on embeddings.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluation.metrics import compute_classification_metrics


def grid_search_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha_range: np.ndarray,
    cv_folds: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Dict:
    """
    Run GridSearchCV to find optimal Ridge alpha.

    Args:
        X_train: Training features (N, D)
        y_train: Training labels (N,)
        alpha_range: Array of alpha values to search
        cv_folds: Number of inner CV folds
        n_jobs: Number of parallel jobs (-1 for all cores)
        random_state: Random seed

    Returns:
        Dictionary with best_alpha, best_score, and best_model
    """
    sample_weights = compute_sample_weight("balanced", y_train)

    clf = Ridge(random_state=random_state)
    param_grid = {"alpha": alpha_range}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        clf,
        param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
        refit=True,
    )

    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    return {
        "best_alpha": grid_search.best_params_["alpha"],
        "best_score": grid_search.best_score_,
        "best_model": grid_search.best_estimator_,
    }


def run_experiment_with_search(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    alpha_range: np.ndarray,
    cv_folds: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Dict:
    """
    Run a single experiment with RidgeCV alpha selection.

    Args:
        features: Full feature matrix (N, D)
        labels: Full label vector (N,)
        train_indices: Training sample indices
        test_indices: Test sample indices
        alpha_range: Alpha values to search
        cv_folds: Inner CV folds for RidgeCV
        n_jobs: Parallel jobs (unused, kept for interface compatibility)
        random_state: Random seed (unused, kept for interface compatibility)

    Returns:
        Dictionary with search results and test metrics
    """
    X_train = features[train_indices]
    y_train = labels[train_indices]
    X_test = features[test_indices]
    y_test = labels[test_indices]

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # RidgeCV selects best alpha internally via cross-validation
    model = RidgeCV(alphas=alpha_range, cv=cv_folds)
    model.fit(X_train, y_train)

    best_alpha = float(model.alpha_)
    best_cv_score = float(model.best_score_)

    # Evaluate on test set
    y_test_pred = np.clip(np.round(model.predict(X_test)).astype(int), 1, 10)
    test_metrics = compute_classification_metrics(y_test, y_test_pred)

    # Add raw RMSE (before rounding)
    from sklearn.metrics import mean_squared_error
    test_metrics["rmse_raw"] = float(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

    return {
        "best_alpha": best_alpha,
        "best_cv_score": best_cv_score,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_metrics": test_metrics,
    }
