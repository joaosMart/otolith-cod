"""
Shallow Classification Models.

Ridge Regression and SVC classifiers for few-shot learning on embeddings.
Follows the paper's methodology: train lightweight classifier on frozen encoder features.

Based on: Sigurðardóttir et al. (2023) - Ecological Informatics
"""

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, Tuple


def create_ridge_classifier(alpha: float = 6.0) -> RidgeClassifier:
    """
    Create a Ridge Regression classifier.

    Paper found stable performance with alpha in [5.0, 15.0].
    Default alpha=6.0 matches the paper's optimal setting.

    Args:
        alpha: L2 regularization strength

    Returns:
        Configured RidgeClassifier
    """
    return RidgeClassifier(alpha=alpha, class_weight="balanced")


def create_svc_classifier(
    C: float = 0.1,
    kernel: str = "linear",
    class_weight: str = "balanced",
) -> SVC:
    """
    Create a Support Vector Classifier.

    Paper explored C in [0.001, 1.0] and chose C=0.1.
    Uses one-vs-one decision function with linear kernel.

    Args:
        C: Inverse regularization strength
        kernel: Kernel type ("linear" recommended)
        class_weight: Class weighting strategy

    Returns:
        Configured SVC
    """
    return SVC(
        C=C,
        kernel=kernel,
        decision_function_shape="ovo",
        class_weight=class_weight,
    )


def train_classifier(
    classifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_features: bool = False,
) -> Tuple[Any, Optional[StandardScaler]]:
    """
    Train a classifier on embedding features.

    Args:
        classifier: sklearn classifier instance
        X_train: Training features (N, D)
        y_train: Training labels (N,)
        scale_features: Whether to standardize features

    Returns:
        Tuple of (trained_classifier, scaler_or_None)
    """
    scaler = None

    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    classifier.fit(X_train, y_train)

    return classifier, scaler


def predict(
    classifier,
    X: np.ndarray,
    scaler: Optional[StandardScaler] = None,
) -> np.ndarray:
    """
    Make predictions using a trained classifier.

    Args:
        classifier: Trained classifier
        X: Features to predict (N, D)
        scaler: Optional scaler to apply

    Returns:
        Predictions (N,)
    """
    if scaler is not None:
        X = scaler.transform(X)

    return classifier.predict(X)


def run_single_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier_type: str = "ridge",
    classifier_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a single fold of cross-validation.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        classifier_type: "ridge" or "svc"
        classifier_params: Optional classifier parameters

    Returns:
        Dictionary with predictions, true labels, and classifier
    """
    params = classifier_params or {}

    if classifier_type == "ridge":
        classifier = create_ridge_classifier(**params)
    elif classifier_type == "svc":
        classifier = create_svc_classifier(**params)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    classifier, scaler = train_classifier(classifier, X_train, y_train)
    predictions = predict(classifier, X_test, scaler)

    return {
        "predictions": predictions,
        "true_labels": y_test,
        "classifier": classifier,
        "scaler": scaler,
    }
