# Multi-Model Embedding Extraction: DINOv2 + Perception Encoder

**Date:** 2026-04-21
**Status:** Approved

## Goal

Add DINOv2 (dinov2-vitl14-reg) and Perception Encoder (PE-Core-L14-336) support to the existing embedding extraction pipeline, alongside CLIP and SigLIP2.

## Models

| Key | HuggingFace ID | Embedding Dim | Image Size | Family | Tokens |
|-----|----------------|---------------|------------|--------|--------|
| `clip-vit-l-14-336` | `openai/clip-vit-large-patch14-336` | 768 | 336 | clip | CLS |
| `siglip2-so400m-14-384` | `google/siglip2-so400m-patch14-384` | 1152 | 384 | clip | CLS |
| `dinov2-vitl14-reg` | `facebook/dinov2-with-registers-large` | 1024 | 518 | dinov2 | CLS + patches + mean-pool |
| `pe-core-l14-336` | `facebook/PE-Core-L14-336-hf` | 768 | 336 | pe | CLS |

## Approach: Model-specific extraction functions

Extend `SUPPORTED_MODELS` with a model family field. `load_model()` handles all families via HuggingFace transformers. `extract_features()` dispatches extraction logic based on model family.

## Changes

### `src/features/extractor.py`

1. **SUPPORTED_MODELS** — add third tuple element for model family:
   - `"clip"` for CLIP and SigLIP2 (use `get_image_features()`)
   - `"dinov2"` for DINOv2 (use `last_hidden_state` from forward pass)
   - `"pe"` for Perception Encoder (use `get_image_features()`)

2. **`load_model()`** — handle different processor types:
   - `clip`/`pe`: `AutoModel` + `AutoProcessor` (unchanged for clip)
   - `dinov2`: `AutoModel` + `AutoImageProcessor`

3. **`extract_features()`** — accept `model_name` parameter to determine family:
   - `clip`/`pe` family: `model.get_image_features(pixel_values=images)` returns single feature vector per image
   - `dinov2` family: `model(pixel_values=images).last_hidden_state` then split into:
     - `features_cls`: index 0 — shape (N, 1024)
     - `features_patch`: index 1: — shape (N, num_patches, 1024)
     - `features_patch_mean_pool`: mean over patch dimension — shape (N, 1024)

4. **Return type** — changes from `(features, labels)` to `(features_dict, labels)`:
   - CLIP/SigLIP/PE: `{"features": array}`
   - DINOv2: `{"features_cls": array, "features_patch": array, "features_patch_mean_pool": array}`

### `src/features/cache.py`

- `save_cached_embeddings()` — accept `features_dict: dict` instead of single array. Save all keys.
- `load_cached_embeddings()` — return features as a dict. Backward compatible: if file has `features` key, return `{"features": array}`.
- No change to `get_cache_path()`.

### `scripts/extract_features.py`

- Add new model names to `--model` choices.
- Handle dict return from `extract_features()`.
- Print shapes for each feature key.

### `configs/config.yaml`

- Add `dinov2-vitl14-reg` and `pe-core-l14-336` model entries.
- Add them to `models_to_run` list.

### `src/features/__init__.py`

- No new exports needed (same functions, changed signatures).

## Dependencies

No new pip dependencies. All models load via `transformers` (already installed).

## Cache Format

Single `.npz` file per model. Keys:

- CLIP/SigLIP/PE: `features`, `labels`, `measurement_ids`
- DINOv2: `features_cls`, `features_patch`, `features_patch_mean_pool`, `labels`, `measurement_ids`

## Downstream Impact

Downstream consumers (training scripts, notebooks) that load embeddings will need to handle the dict return from `load_cached_embeddings()`. For DINOv2, they choose which feature type to use. For CLIP/SigLIP/PE, they use `features_dict["features"]` as before.
