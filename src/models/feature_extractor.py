"""
Vision Feature Extraction Module (HuggingFace Transformers).

Provides functions to load vision models (CLIP, SigLIP2) and extract embeddings
from otolith images. Designed for comparing frozen encoder performance.

Based on: Sigurðardóttir et al. (2023) - Ecological Informatics
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, Callable
from src.utils.device import get_device
from tqdm import tqdm

from transformers import AutoModel, AutoProcessor


# Model configurations: (hf_model_id, embedding_dim)
SUPPORTED_MODELS = {
    "clip-vit-l-14-336": ("openai/clip-vit-large-patch14-336", 768),
    "siglip2-so400m-14-384": ("google/siglip2-so400m-patch14-384", 1152),
}


def load_model(
    model_name: str = "clip-vit-l-14-336",
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, Callable]:
    """
    Load a vision model for feature extraction.

    Args:
        model_name: Model identifier from SUPPORTED_MODELS
        device: Target device (auto-detected if None)

    Returns:
        Tuple of (model, preprocess_fn) - preprocess_fn wraps the HuggingFace processor

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in SUPPORTED_MODELS:
        available = ", ".join(SUPPORTED_MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    model_id, _ = SUPPORTED_MODELS[model_name]
    device = device or get_device()

    print(f"Loading {model_id} from HuggingFace...")
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # float32 for MPS stability
        attn_implementation="sdpa",  # MPS compatible
    )
    processor = AutoProcessor.from_pretrained(model_id)

    model = model.to(device)
    model.eval()

    # Create a preprocess function that wraps the processor
    # This makes it compatible with the existing dataset interface
    def preprocess(image):
        """Preprocess image using HuggingFace processor."""
        inputs = processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    return model, preprocess


def extract_features(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    normalize: bool = True,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract image features using a frozen vision encoder.

    Args:
        model: Vision model (CLIP or SigLIP2)
        dataloader: DataLoader yielding (images, labels) batches
        device: Target device (auto-detected if None)
        normalize: Whether to L2-normalize features
        show_progress: Whether to show progress bar

    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    device = device or get_device()
    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    iterator = tqdm(dataloader, desc="Extracting features") if show_progress else dataloader

    with torch.inference_mode():
        for images, labels in iterator:
            images = images.to(device)

            # Extract image features using HuggingFace API
            features = model.get_image_features(pixel_values=images)

            # L2 normalize if requested (F.normalize handles zero-norm vectors)
            if normalize:
                features = F.normalize(features, p=2, dim=-1, eps=1e-8)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    return features, labels


def extract_and_cache_features(
    model_name: str,
    dataloader: torch.utils.data.DataLoader,
    cache_dir: Union[str, Path],
    normalize: bool = True,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features and cache to disk, or load from cache if available.

    Args:
        model_name: Model identifier
        dataloader: DataLoader yielding (images, labels)
        cache_dir: Directory for caching embeddings
        normalize: Whether to L2-normalize features
        force_recompute: Whether to ignore existing cache

    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"{model_name}_embeddings.npz"

    if cache_file.exists() and not force_recompute:
        print(f"Loading cached embeddings from {cache_file}")
        data = np.load(cache_file, allow_pickle=False)
        return data["features"], data["labels"]

    print(f"Extracting features using {model_name}...")
    model, _ = load_model(model_name)
    features, labels = extract_features(model, dataloader, normalize=normalize)

    print(f"Caching embeddings to {cache_file}")
    np.savez(cache_file, features=features, labels=labels)

    return features, labels


def get_embedding_dim(model_name: str) -> int:
    """Get the embedding dimension for a model."""
    if model_name not in SUPPORTED_MODELS:
        return 768  # default
    _, dim = SUPPORTED_MODELS[model_name]
    return dim
