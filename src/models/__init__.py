"""
Models module for otolith age prediction.

Provides:
- Feature extraction using CLIP and SigLIP2
- Shallow classifiers (Ridge, SVC) for few-shot learning
"""

from .feature_extractor import (
    SUPPORTED_MODELS,
    load_model,
    extract_features,
    extract_and_cache_features,
    get_embedding_dim,
    get_device,
)
from .shallow_models import (
    create_ridge_classifier,
    create_svc_classifier,
    train_classifier,
    predict,
    run_single_fold,
)

__all__ = [
    # Feature extraction
    "SUPPORTED_MODELS",
    "load_model",
    "extract_features",
    "extract_and_cache_features",
    "get_embedding_dim",
    "get_device",
    # Shallow models
    "create_ridge_classifier",
    "create_svc_classifier",
    "train_classifier",
    "predict",
    "run_single_fold",
]
