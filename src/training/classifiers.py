"""
Classifier creation and training utilities.

Ridge Regression and SVC classifiers for few-shot learning on embeddings.

Based on: Sigurðardóttir et al. (2023) - Ecological Informatics
"""

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from typing import Optional


def create_ridge(alpha: float = 6.0, random_state: int = 42) -> Ridge:
    """Create a Ridge regression model for ordinal age prediction."""
    return Ridge(alpha=alpha, random_state=random_state)


def create_ridge_classifier(alpha: float = 6.0) -> RidgeClassifier:
    """Create a Ridge classification model with balanced class weights."""
    return RidgeClassifier(alpha=alpha, class_weight="balanced")


def create_svc(
    C: float = 0.1,
    kernel: str = "linear",
    class_weight: str = "balanced",
) -> SVC:
    """Create a Support Vector Classifier with one-vs-one decision function."""
    return SVC(
        C=C,
        kernel=kernel,
        decision_function_shape="ovo",
        class_weight=class_weight,
    )


def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 6.0,
    random_state: int = 42,
    balanced: bool = True,
) -> Ridge:
    """
    Train a Ridge model on embeddings.

    Args:
        X_train: Training features (N, D)
        y_train: Training labels (N,)
        alpha: Regularization strength
        random_state: Random seed
        balanced: Whether to use balanced sample weights

    Returns:
        Trained Ridge model
    """
    model = Ridge(alpha=alpha, random_state=random_state)
    sample_weights = compute_sample_weight("balanced", y_train) if balanced else None
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def predict_ridge(model: Ridge, X: np.ndarray, clip_range: Optional[tuple] = None) -> np.ndarray:
    """
    Predict with a Ridge model, rounding to integer classes.

    Args:
        model: Trained Ridge model
        X: Features (N, D)
        clip_range: Optional (min, max) to clip predictions

    Returns:
        Integer predictions (N,)
    """
    y_pred = np.round(model.predict(X)).astype(int)
    if clip_range is not None:
        y_pred = np.clip(y_pred, clip_range[0], clip_range[1])
    return y_pred
