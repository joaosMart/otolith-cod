# Multi-Model Embedding Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DINOv2 and Perception Encoder embedding extraction alongside existing CLIP/SigLIP2, with full pipeline integration.

**Architecture:** Extend SUPPORTED_MODELS with a model family field. Dispatch extraction logic in `extract_features()` based on family. DINOv2 returns CLS + patch + mean-pooled features; CLIP/SigLIP/PE return a single feature vector. Cache and downstream code updated for dict-based feature returns.

**Tech Stack:** HuggingFace transformers (AutoModel, AutoProcessor, AutoImageProcessor), PyTorch, NumPy

---

### Task 1: Update SUPPORTED_MODELS registry

**Files:**
- Modify: `src/features/extractor.py:20-27`

- [ ] **Step 1: Update imports to include AutoImageProcessor**

In `src/features/extractor.py`, change line 20 from:
```python
from transformers import AutoModel, AutoProcessor
```
to:
```python
from transformers import AutoModel, AutoProcessor, AutoImageProcessor
```

- [ ] **Step 2: Update SUPPORTED_MODELS to include family field and new models**

Replace lines 23-27:
```python
# Model configurations: (hf_model_id, embedding_dim)
SUPPORTED_MODELS = {
    "clip-vit-l-14-336": ("openai/clip-vit-large-patch14-336", 768),
    "siglip2-so400m-14-384": ("google/siglip2-so400m-patch14-384", 1152),
}
```
with:
```python
# Model configurations: (hf_model_id, embedding_dim, family)
# family determines extraction logic: "clip" uses get_image_features(),
# "dinov2" uses last_hidden_state, "pe" uses get_image_features()
SUPPORTED_MODELS = {
    "clip-vit-l-14-336": ("openai/clip-vit-large-patch14-336", 768, "clip"),
    "siglip2-so400m-14-384": ("google/siglip2-so400m-patch14-384", 1152, "clip"),
    "dinov2-vitl14-reg": ("facebook/dinov2-with-registers-large", 1024, "dinov2"),
    "pe-core-l14-336": ("facebook/PE-Core-L14-336-hf", 768, "pe"),
}
```

- [ ] **Step 3: Update get_embedding_dim to handle 3-tuple**

Replace `get_embedding_dim` (lines 155-160):
```python
def get_embedding_dim(model_name: str) -> int:
    """Get the embedding dimension for a model."""
    if model_name not in SUPPORTED_MODELS:
        return 768  # default
    _, dim = SUPPORTED_MODELS[model_name]
    return dim
```
with:
```python
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
```

- [ ] **Step 4: Commit**

```bash
git add src/features/extractor.py
git commit -m "feat: expand SUPPORTED_MODELS with DINOv2, PE, and family field"
```

---

### Task 2: Update load_model for all model families

**Files:**
- Modify: `src/features/extractor.py:49-95`

- [ ] **Step 1: Update load_model to handle DINOv2 processor**

Replace the `load_model` function (lines 49-95) with:
```python
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

    print(f"Loading {model_id} from HuggingFace...")
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )

    # DINOv2 uses AutoImageProcessor; CLIP/SigLIP/PE use AutoProcessor
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
```

- [ ] **Step 2: Verify existing models still load (quick smoke test)**

Run:
```bash
python -c "from src.features.extractor import load_model, SUPPORTED_MODELS; print(list(SUPPORTED_MODELS.keys()))"
```
Expected: prints list with all 4 model names.

- [ ] **Step 3: Commit**

```bash
git add src/features/extractor.py
git commit -m "feat: update load_model to handle DINOv2 and PE processors"
```

---

### Task 3: Update extract_features for multi-family dispatch

**Files:**
- Modify: `src/features/extractor.py:98-151`

- [ ] **Step 1: Replace extract_features with family-aware version**

Replace the `extract_features` function (lines 98-151) with:
```python
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
            else:
                # clip and pe families both use get_image_features
                features = model.get_image_features(pixel_values=images)
                if normalize:
                    features = F.normalize(features, p=2, dim=-1, eps=1e-8)
                collectors.setdefault("features", []).append(features.cpu().numpy())

            all_labels.append(labels.numpy())

    features_dict = {}
    for key, batches in collectors.items():
        features_dict[key] = np.concatenate(batches) if batches[0].ndim == 1 else np.vstack(batches) if batches[0].ndim == 2 else np.concatenate(batches, axis=0)
    labels = np.concatenate(all_labels)

    return features_dict, labels
```

- [ ] **Step 2: Commit**

```bash
git add src/features/extractor.py
git commit -m "feat: add multi-family dispatch to extract_features"
```

---

### Task 4: Update cache.py for dict-based features

**Files:**
- Modify: `src/features/cache.py:1-86`

- [ ] **Step 1: Replace cache.py with dict-aware version**

Replace the full content of `src/features/cache.py` with:
```python
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
```

- [ ] **Step 2: Commit**

```bash
git add src/features/cache.py
git commit -m "feat: update cache to support dict-based features"
```

---

### Task 5: Update extract_features.py script

**Files:**
- Modify: `scripts/extract_features.py`

- [ ] **Step 1: Update the script to handle dict features and new models**

Replace `scripts/extract_features.py` with:
```python
#!/usr/bin/env python3
"""
Extract and cache vision embeddings from otolith images.

Usage:
    python scripts/extract_features.py --model siglip2-so400m-14-384
    python scripts/extract_features.py --model dinov2-vitl14-reg --force
    python scripts/extract_features.py --model pe-core-l14-336 --clahe
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from torch.utils.data import DataLoader

from src.data import OtolithDataset, SUPPORTED_DATA
from src.features import (
    SUPPORTED_MODELS,
    load_model,
    extract_features,
    load_cached_embeddings,
    save_cached_embeddings,
    get_cache_path,
)
from src.utils import load_config, get_output_paths, print_device_info


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and cache vision embeddings")
    parser.add_argument(
        "--model",
        type=str,
        default="siglip2-so400m-14-384",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Vision model to use for feature extraction",
    )
    parser.add_argument(
        "--images-path",
        type=str,
        default="segmented_images",
        choices=list(SUPPORTED_DATA.keys()),
        help="Image dataset to use for feature extraction",
    )
    parser.add_argument(
        "--clahe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply CLAHE image enhancement before feature extraction",
    )
    parser.add_argument(
        "--repeat-clahe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply CLAHE enhancement twice for stronger effect",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for embeddings (default: from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for feature extraction (default: 64)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if cache exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)
    print_device_info()

    # Load config
    config = load_config(args.config)
    output_paths = get_output_paths(config)
    if args.output_dir:
        cache_dir = args.output_dir
    elif args.images_path == "segmented_images":
        cache_dir = "outputs/segmented_embeddings"
    else:
        cache_dir = str(output_paths["embeddings"])
    cache_path = get_cache_path(args.model, cache_dir, apply_clahe=args.clahe, repeat_clahe=args.repeat_clahe)

    # Check cache
    if cache_path.exists() and not args.force:
        print(f"\nCached embeddings found: {cache_path}")
        features_dict, labels, measurement_ids = load_cached_embeddings(str(cache_path))
        for key, arr in features_dict.items():
            print(f"  {key}: {arr.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Measurement IDs: {'present' if measurement_ids is not None else 'missing'}")
        if measurement_ids is not None:
            print("\nUse --force to re-extract.")
            return
        else:
            print("  Missing measurement_ids, re-extracting...")

    # Load dataset and model
    data_config = config["data"]
    print(f"\nLoading model: {args.model}")
    model, preprocess = load_model(args.model,
                                   apply_clahe=args.clahe,
                                   repeat_clahe=args.repeat_clahe)

    metadata_csv = data_config.get("metadata_csv")
    root_dir = SUPPORTED_DATA[args.images_path]
    dataset = OtolithDataset(
        root_dir=root_dir,
        transform=preprocess,
        age_range=tuple(data_config["age_range"]),
        metadata_csv=metadata_csv,
    )
    print(f"Dataset: {len(dataset)} images")
    print(f"Class distribution: {dataset.get_class_counts()}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Extract features
    features_dict, labels = extract_features(
        model, dataloader, model_name=args.model, normalize=True
    )

    # Extract measurement_ids from image filenames
    paths = dataset.get_paths()
    measurement_ids = np.array([int(p.stem) for p in paths])

    # Save
    save_cached_embeddings(str(cache_path), features_dict, labels, measurement_ids)
    print(f"\nFeature shapes:")
    for key, arr in features_dict.items():
        print(f"  {key}: {arr.shape}")
    print(f"Saved to: {cache_path}")
    print("\nFeature extraction complete!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/extract_features.py
git commit -m "feat: update extract script for multi-model support"
```

---

### Task 6: Update train_classifier.py for dict-based embeddings

**Files:**
- Modify: `scripts/train_classifier.py:116`

- [ ] **Step 1: Update embedding loading in train_classifier.py**

In `scripts/train_classifier.py`, replace line 116:
```python
    features, labels, measurement_ids = load_cached_embeddings(args.embeddings)
```
with:
```python
    features_dict, labels, measurement_ids = load_cached_embeddings(args.embeddings)
    # Use CLS features for DINOv2, or standard features for CLIP/SigLIP/PE
    if "features" in features_dict:
        features = features_dict["features"]
    elif "features_cls" in features_dict:
        features = features_dict["features_cls"]
    else:
        raise KeyError(f"No recognized feature key in embeddings. Found: {list(features_dict.keys())}")
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_classifier.py
git commit -m "fix: update train_classifier for dict-based embeddings"
```

---

### Task 7: Update __init__.py exports and config.yaml

**Files:**
- Modify: `src/features/__init__.py`
- Modify: `configs/config.yaml`

- [ ] **Step 1: Add get_model_family to __init__.py exports**

In `src/features/__init__.py`, add `get_model_family` to the import from `.extractor`:
```python
from .extractor import (
    SUPPORTED_MODELS,
    load_model,
    extract_features,
    get_embedding_dim,
    get_model_family,
)
```

And add `"get_model_family"` to the `__all__` list.

- [ ] **Step 2: Add new models to config.yaml**

In `configs/config.yaml`, after the `siglip2-so400m-14-384` entry (line 30), add:
```yaml

  # DINOv2 with registers (Meta, 2024)
  dinov2-vitl14-reg:
    model_id: "facebook/dinov2-with-registers-large"
    embedding_dim: 1024
    image_size: 518

  # Perception Encoder (Meta, 2025)
  pe-core-l14-336:
    model_id: "facebook/PE-Core-L14-336-hf"
    embedding_dim: 768
    image_size: 336
```

And update `models_to_run` (lines 125-127) to include the new models:
```yaml
  models_to_run:
    - "clip-vit-l-14-336"
    - "siglip2-so400m-14-384"
    - "dinov2-vitl14-reg"
    - "pe-core-l14-336"
```

- [ ] **Step 3: Commit**

```bash
git add src/features/__init__.py configs/config.yaml
git commit -m "feat: add DINOv2 and PE to config and exports"
```

---

### Task 8: Smoke test all models

- [ ] **Step 1: Verify imports and model registry**

Run:
```bash
python -c "
from src.features import SUPPORTED_MODELS, get_model_family, get_embedding_dim
for name in SUPPORTED_MODELS:
    family = get_model_family(name)
    dim = get_embedding_dim(name)
    print(f'{name}: family={family}, dim={dim}')
"
```
Expected output:
```
clip-vit-l-14-336: family=clip, dim=768
siglip2-so400m-14-384: family=clip, dim=1152
dinov2-vitl14-reg: family=dinov2, dim=1024
pe-core-l14-336: family=pe, dim=768
```

- [ ] **Step 2: Test feature extraction with one model (siglip2, quickest to verify)**

Run:
```bash
python scripts/extract_features.py --model siglip2-so400m-14-384 --batch-size 4 --force
```
Expected: completes without error, prints feature shapes with `features: (N, 1152)`.

- [ ] **Step 3: Test DINOv2 extraction**

Run:
```bash
python scripts/extract_features.py --model dinov2-vitl14-reg --batch-size 4 --force
```
Expected: completes without error, prints shapes for `features_cls`, `features_patch`, and `features_patch_mean_pool`.

- [ ] **Step 4: Test PE extraction**

Run:
```bash
python scripts/extract_features.py --model pe-core-l14-336 --batch-size 4 --force
```
Expected: completes without error, prints feature shapes with `features: (N, 768)`.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: multi-model embedding extraction (DINOv2 + Perception Encoder)"
```
