"""
Feature extraction module for otolith age prediction.

Provides model loading, feature extraction, caching, and metadata augmentation.
"""

from .extractor import (
    SUPPORTED_MODELS,
    load_model,
    extract_features,
    get_embedding_dim,
    get_model_family,
)
from .cache import (
    load_cached_embeddings,
    save_cached_embeddings,
    get_cache_path,
)
from .metadata import (
    load_metadata,
    get_tabular_features,
    augment_embeddings,
)

__all__ = [
    "SUPPORTED_MODELS",
    "load_model",
    "extract_features",
    "get_embedding_dim",
    "get_model_family",
    "load_cached_embeddings",
    "save_cached_embeddings",
    "get_cache_path",
    "load_metadata",
    "get_tabular_features",
    "augment_embeddings",
]
