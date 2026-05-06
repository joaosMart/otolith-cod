"""
Vision Feature Extraction Module (HuggingFace Transformers).

Provides functions to load vision models (CLIP, SigLIP2, DINOv2, PE) and extract embeddings
from otolith images. Designed for comparing frozen encoder performance.

Based on: Sigurðardóttir et al. (2023) - Ecological Informatics
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable
from src.utils.device import get_device
from tqdm import tqdm
import cv2
import PIL.Image as Image

from transformers import AutoModel, AutoProcessor, AutoImageProcessor
try:
    import core.vision_encoder.pe as pe_encoder
    import core.vision_encoder.transforms as pe_transforms
except ImportError:
    pe_encoder = None
    pe_transforms = None

# Model configurations: (hf_model_id, embedding_dim, family)
# family determines extraction logic: "clip" uses get_image_features(),
# "dinov2" uses last_hidden_state, "pe" uses get_image_features()
SUPPORTED_MODELS = {
    "clip-vit-l-14-336": ("openai/clip-vit-large-patch14-336", 768, "clip"),
    "siglip2-so400m-14-384": ("google/siglip2-so400m-patch14-384", 1152, "clip"),
    "dinov2-vitl14-reg": ("facebook/dinov2-with-registers-large", 1024, "dinov2"),
    "pe-core-l14-336": ("PE-Core-L14-336", 1024, "pe"),
}



def clahe_enhancement(image, 
                      clip_limit: float = 3.0,
                      tile_size: int = 25, 
                      repeat_clahe: bool = True):
    """Apply CLAHE to a single image with adaptive tile grid size based on image dimensions."""
    # Convert to grayscale treating it as a Light channel
    gray_image = np.array(image.convert("L")) 
    # Extrapolate grid size from image dimensions and tile size
    h, w = gray_image.shape[:2]
    grid_y = max(1, h // tile_size)
    grid_x = max(1, w // tile_size)
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_x, grid_y))
    enhanced_image = clahe.apply(gray_image)
    if repeat_clahe:
        enhanced_image = clahe.apply(enhanced_image)
    return Image.fromarray(np.stack([enhanced_image] * 3, axis=-1)) # Replicate to 3 channels

def load_model(
    model_name: str = "clip-vit-l-14-336",
    device: Optional[torch.device] = None,
    apply_clahe: bool = True,
    repeat_clahe: bool = True,
) -> Tuple[torch.nn.Module, Callable]:
    """
    Load a vision model for feature extraction.

    Args:
        model_name: Model identifier from SUPPORTED_MODELS
        device: Target device (auto-detected if None)
        apply_clahe: Whether to apply CLAHE enhancement
        repeat_clahe: Whether to apply CLAHE twice

    Returns:
        Tuple of (model, preprocess_fn)

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in SUPPORTED_MODELS:
        available = ", ".join(SUPPORTED_MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    model_id, _, family = SUPPORTED_MODELS[model_name]
    device = device or get_device()

    if family == "pe":
        print(f"Loading {model_id} via PE native API (pretrained=True)...")
        pe_model = pe_encoder.CLIP.from_config(model_id, pretrained=True)
        pe_model = pe_model.to(device)
        pe_model.eval()

        pe_preprocess = pe_transforms.get_image_transform(pe_model.image_size)

        def preprocess(image):
            if apply_clahe:
                image = clahe_enhancement(image, repeat_clahe=repeat_clahe)
            return pe_preprocess(image)

        return pe_model, preprocess

    print(f"Loading {model_id} from HuggingFace...")
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )

    # DINOv2 uses AutoImageProcessor; CLIP/SigLIP use AutoProcessor
    if family == "dinov2":
        processor = AutoImageProcessor.from_pretrained(model_id)
    else:
        processor = AutoProcessor.from_pretrained(model_id)

    model = model.to(device)
    model.eval()

    def preprocess(image):
        """Preprocess image using HuggingFace processor."""
        if apply_clahe:
            image = clahe_enhancement(image, repeat_clahe=repeat_clahe)
        inputs = processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    return model, preprocess


def extract_features(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    model_name: str = "clip-vit-l-14-336",
    device: Optional[torch.device] = None,
    normalize: bool = True,
    show_progress: bool = True,
) -> Tuple[dict, np.ndarray]:
    """
    Extract image features using a frozen vision encoder.

    Args:
        model: Vision model
        dataloader: DataLoader yielding (images, labels) batches
        model_name: Model name from SUPPORTED_MODELS (determines extraction logic)
        device: Target device (auto-detected if None)
        normalize: Whether to L2-normalize features
        show_progress: Whether to show progress bar

    Returns:
        Tuple of (features_dict, labels) where features_dict keys depend on model family:
        - clip/pe: {"features": array}
        - dinov2: {"features_cls": array, "features_patch": array, "features_patch_mean_pool": array}
    """
    device = device or get_device()
    family = get_model_family(model_name)
    model = model.to(device)
    model.eval()

    all_labels = []
    collectors = {}  # key -> list of batch arrays

    iterator = tqdm(dataloader, desc="Extracting features") if show_progress else dataloader

    with torch.inference_mode():
        for images, labels in iterator:
            images = images.to(device)

            if family == "dinov2":
                outputs = model(pixel_values=images)
                hidden = outputs.last_hidden_state
                cls_features = hidden[:, 0, :]
                patch_features = hidden[:, 1:, :]
                mean_pool_features = patch_features.mean(dim=1)

                if normalize:
                    cls_features = F.normalize(cls_features, p=2, dim=-1, eps=1e-8)
                    mean_pool_features = F.normalize(mean_pool_features, p=2, dim=-1, eps=1e-8)
                    # Normalize each patch vector independently
                    patch_features = F.normalize(patch_features, p=2, dim=-1, eps=1e-8)

                collectors.setdefault("features_cls", []).append(cls_features.cpu().numpy())
                collectors.setdefault("features_patch", []).append(patch_features.cpu().numpy())
                collectors.setdefault("features_patch_mean_pool", []).append(mean_pool_features.cpu().numpy())
            elif family == "pe":
                features = model.encode_image(images)
                if normalize:
                    features = F.normalize(features, p=2, dim=-1, eps=1e-8)
                collectors.setdefault("features", []).append(features.cpu().numpy())
            else:
                # clip/siglip families use get_image_features
                features = model.get_image_features(pixel_values=images)
                if normalize:
                    features = F.normalize(features, p=2, dim=-1, eps=1e-8)
                collectors.setdefault("features", []).append(features.cpu().numpy())

            all_labels.append(labels.numpy())

    features_dict = {}
    for key, batches in collectors.items():
        features_dict[key] = np.concatenate(batches, axis=0)
    labels = np.concatenate(all_labels)

    return features_dict, labels



def get_embedding_dim(model_name: str) -> int:
    """Get the embedding dimension for a model."""
    if model_name not in SUPPORTED_MODELS:
        return 768  # default
    _, dim, _ = SUPPORTED_MODELS[model_name]
    return dim


def get_model_family(model_name: str) -> str:
    """Get the model family for dispatch logic."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    _, _, family = SUPPORTED_MODELS[model_name]
    return family
