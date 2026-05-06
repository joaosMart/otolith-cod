"""
Embedding caching utilities.

Provides functions to save and load cached embeddings (.npz files),
which serve as the interface between feature extraction and training.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


# Keys that are not feature arrays
_META_KEYS = {"labels", "measurement_ids"}


def load_cached_embeddings(
    cache_path: str,
) -> Tuple[dict, np.ndarray, Optional[np.ndarray]]:
    """
    Load cached embeddings from a .npz file.

    Args:
        cache_path: Path to the .npz file

    Returns:
        Tuple of (features_dict, labels, measurement_ids)
        features_dict maps feature names to arrays, e.g.:
          - CLIP/SigLIP/PE: {"features": array}
          - DINOv2: {"features_cls": array, "features_patch": array, "features_patch_mean_pool": array}
        measurement_ids may be None if not present in file.
    """
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    data = np.load(path, allow_pickle=False)

    # Extract labels
    labels = data["labels"]
    measurement_ids = data["measurement_ids"] if "measurement_ids" in data.files else None

    # Everything else is a feature array
    features_dict = {}
    for key in data.files:
        if key not in _META_KEYS:
            features_dict[key] = data[key]

    # Backward compat: rename legacy "embeddings" key to "features"
    if "embeddings" in features_dict and "features" not in features_dict:
        features_dict["features"] = features_dict.pop("embeddings")

    data.close()
    return features_dict, labels, measurement_ids


def save_cached_embeddings(
    cache_path: str,
    features_dict: dict,
    labels: np.ndarray,
    measurement_ids: Optional[np.ndarray] = None,
) -> None:
    """
    Save embeddings to a .npz file.

    Args:
        cache_path: Path to save the .npz file
        features_dict: Dict mapping feature names to arrays
        labels: Label vector (N,)
        measurement_ids: Optional measurement IDs (N,)
    """
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {**features_dict, "labels": labels}
    if measurement_ids is not None:
        arrays["measurement_ids"] = measurement_ids

    np.savez(path, **arrays)
    print(f"Saved embeddings to {path}")


def get_cache_path(model_name: str,
                   cache_dir: str = "outputs/embeddings",
                   apply_clahe: bool = False,
                   repeat_clahe: bool = False) -> Path:
    """Get the standard cache file path for a model."""
    if apply_clahe:
        model_name += '_clahe'
    if repeat_clahe:
        model_name += '_repeat'
    return Path(cache_dir) / f"{model_name}_embeddings.npz"
