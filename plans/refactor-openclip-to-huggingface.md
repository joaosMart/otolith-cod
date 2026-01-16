# refactor: Migrate from OpenCLIP to HuggingFace Transformers

## Overview

Replace the OpenCLIP library with HuggingFace Transformers for loading both CLIP and SigLIP2 models. This change resolves MPS compatibility issues with SigLIP2 and provides a unified, well-supported API for both model families.

## Problem Statement / Motivation

The current implementation uses `open_clip` to load vision models:
- CLIP ViT-L/14@336px works correctly on MPS
- SigLIP2 SO400M-14-378 fails with MPS buffer size error:
  ```
  [MPSNDArray, initWithBufferImpl:offset:descriptor:isForNDArrayAlias:isUserBuffer:]
  Error: buffer is not large enough. Must be 147456 bytes
  ```

OpenCLIP's SigLIP2 support via `hf-hub:timm/` pathway has MPS compatibility issues. HuggingFace Transformers has native, well-tested support for both model families with better MPS compatibility.

## Proposed Solution

Migrate from OpenCLIP to HuggingFace Transformers:

| Aspect | Current (OpenCLIP) | New (HuggingFace) |
|--------|-------------------|-------------------|
| Import | `import open_clip` | `from transformers import AutoModel, AutoProcessor` |
| Load model | `open_clip.create_model_and_transforms()` | `AutoModel.from_pretrained()` |
| Load processor | Returned with model | `AutoProcessor.from_pretrained()` |
| Extract features | `model.encode_image(images)` | `model.get_image_features(pixel_values=...)` |
| CLIP model ID | `ViT-L-14-336` | `openai/clip-vit-large-patch14-336` |
| SigLIP2 model ID | `hf-hub:timm/ViT-SO400M-14-SigLIP2-378` | `google/siglip2-so400m-patch14-384` |

**Note:** The SigLIP2 model is actually 384px resolution, not 378px. The HuggingFace model ID is `google/siglip2-so400m-patch14-384`.

## Technical Considerations

### Architecture Impact
- Minimal - same frozen encoder → Ridge classifier pipeline
- Feature dimensions remain unchanged (768 for CLIP, 1152 for SigLIP2)
- L2 normalization still applied before classification

### Preprocessing Changes
- OpenCLIP returns a `preprocess` callable with the model
- HuggingFace uses `AutoProcessor` which handles preprocessing differently
- The processor must be called before creating tensors

### MPS Compatibility
- Use `attn_implementation="sdpa"` for MPS-compatible attention
- Use `torch.float32` dtype on MPS for stability
- HuggingFace Transformers has better MPS testing coverage

## Acceptance Criteria

- [ ] CLIP ViT-L/14@336px loads and extracts features on MPS
- [ ] SigLIP2 SO400M-14-384 loads and extracts features on MPS
- [ ] Embedding dimensions match expected values (768, 1152)
- [ ] L2 normalization produces unit vectors
- [ ] Cached embeddings work correctly
- [ ] Full experiment runs without MPS errors

## MVP Implementation

### src/models/feature_extractor.py

```python
"""
Vision Feature Extraction Module (HuggingFace Transformers).
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
from src.utils.device import get_device
from tqdm import tqdm
from PIL import Image

from transformers import AutoModel, AutoProcessor


# Model configurations: (hf_model_id, embedding_dim)
SUPPORTED_MODELS = {
    "clip-vit-l-14-336": ("openai/clip-vit-large-patch14-336", 768),
    "siglip2-so400m-14-384": ("google/siglip2-so400m-patch14-384", 1152),
}


def load_model(
    model_name: str = "clip-vit-l-14-336",
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, AutoProcessor]:
    """
    Load a vision model for feature extraction.

    Args:
        model_name: Model identifier from SUPPORTED_MODELS
        device: Target device (auto-detected if None)

    Returns:
        Tuple of (model, processor)
    """
    if model_name not in SUPPORTED_MODELS:
        available = ", ".join(SUPPORTED_MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    model_id, _ = SUPPORTED_MODELS[model_name]
    device = device or get_device()

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # float32 for MPS stability
        attn_implementation="sdpa",  # MPS compatible
    )
    processor = AutoProcessor.from_pretrained(model_id)

    model = model.to(device)
    model.eval()

    return model, processor


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

            # L2 normalize if requested
            if normalize:
                features = features / features.norm(dim=-1, keepdim=True)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    return features, labels


def get_embedding_dim(model_name: str) -> int:
    """Get the embedding dimension for a model."""
    if model_name not in SUPPORTED_MODELS:
        return 768  # default
    _, dim = SUPPORTED_MODELS[model_name]
    return dim
```

### configs/config.yaml (model section)

```yaml
# Vision Models (HuggingFace Transformers)
models:
  clip-vit-l-14-336:
    model_id: "openai/clip-vit-large-patch14-336"
    embedding_dim: 768
    image_size: 336

  siglip2-so400m-14-384:
    model_id: "google/siglip2-so400m-patch14-384"
    embedding_dim: 1152
    image_size: 384

# Default model for experiments
feature_extraction:
  default_model: "clip-vit-l-14-336"

# Models to compare in experiments
experiment:
  models_to_run:
    - "clip-vit-l-14-336"
    - "siglip2-so400m-14-384"
```

### scripts/run_experiment.py (extract_features_for_model function)

```python
def extract_features_for_model(
    model_name: str,
    data_dir: Path,
    cache_dir: Path,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features for a specific model."""
    cache_file = cache_dir / f"{model_name}_embeddings.npz"

    if cache_file.exists() and not force_recompute:
        print(f"Loading cached embeddings from {cache_file}")
        data = np.load(cache_file)
        return data["features"], data["labels"]

    print(f"\nExtracting features using {model_name}...")

    # Load model and processor
    model, processor = load_model(model_name)
    device = get_device()

    # Create dataset with HuggingFace processor
    dataset = OtolithDataset(
        root_dir=data_dir,
        processor=processor,  # Pass processor instead of transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,  # MPS works better with num_workers=0
    )

    # Extract features
    features, labels = extract_features(model, dataloader, device)

    # Cache to disk
    print(f"Caching embeddings to {cache_file}")
    np.savez(cache_file, features=features, labels=labels)

    return features, labels
```

### src/data/dataset.py (OtolithDataset with processor)

```python
class OtolithDataset(Dataset):
    """Dataset that uses HuggingFace processor for preprocessing."""

    def __init__(self, root_dir: Path, processor, label_encoder=None):
        self.root_dir = Path(root_dir)
        self.processor = processor
        self.label_encoder = label_encoder

        # Find all images and extract labels from directory names
        self.samples = []
        for age_dir in sorted(self.root_dir.iterdir()):
            if age_dir.is_dir():
                age = int(age_dir.name)
                for img_path in age_dir.glob("*.jpg"):
                    self.samples.append((img_path, age))

        # Fit label encoder if needed
        if self.label_encoder is None:
            ages = sorted(set(age for _, age in self.samples))
            self.label_encoder = {age: i for i, age in enumerate(ages)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        # Use processor for preprocessing
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        label = self.label_encoder[age]
        return pixel_values, label
```

### pyproject.toml (dependencies)

```toml
dependencies = [
    # Core ML - PyTorch with MPS support
    "torch>=2.0.0",
    "torchvision>=0.15.0",

    # HuggingFace Transformers for CLIP and SigLIP2
    "transformers>=4.49.0",  # Required for SigLIP2 support

    # ML utilities
    "scikit-learn>=1.3.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",

    # Visualization
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",

    # Image processing
    "pillow>=9.5.0",

    # Utilities
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
]
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/models/feature_extractor.py:17` | Change import from `open_clip` to `transformers` |
| `src/models/feature_extractor.py:21-24` | Update `SUPPORTED_MODELS` with HuggingFace model IDs |
| `src/models/feature_extractor.py:51-58` | Rewrite `load_model()` to use `AutoModel.from_pretrained()` |
| `src/models/feature_extractor.py:100` | Change `model.encode_image()` to `model.get_image_features()` |
| `src/models/feature_extractor.py:155-161` | Update `get_embedding_dim()` |
| `configs/config.yaml:18-34` | Update model configurations with HuggingFace IDs |
| `scripts/run_experiment.py:88-92` | Update dataset creation to use processor |
| `pyproject.toml:17-19` | Remove `open-clip-torch` and `timm`, update `transformers` version |
| `src/data/dataset.py` | Update `OtolithDataset` to use processor |

## Dependencies & Risks

### Dependencies
- `transformers>=4.49.0` - Required for SigLIP2 support
- PyTorch MPS backend

### Risks
- **Low:** API differences might require minor adjustments
- **Low:** Preprocessing differences between OpenCLIP and HuggingFace (handled by processor)
- **Mitigated:** MPS compatibility tested with both models

## Success Metrics

1. Both models load without errors on MPS
2. Feature extraction completes for all 8,637 otolith images
3. Embedding shapes match expected dimensions
4. 10-fold cross-validation runs successfully
5. Results are comparable to expected baseline performance

## References

### Internal References
- Current implementation: `src/models/feature_extractor.py:51-58`
- Dataset loading: `src/data/dataset.py`
- Experiment script: `scripts/run_experiment.py`

### External References
- [HuggingFace CLIP Documentation](https://huggingface.co/docs/transformers/model_doc/clip)
- [HuggingFace SigLIP2 Documentation](https://huggingface.co/docs/transformers/model_doc/siglip2)
- [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)
- [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384)
