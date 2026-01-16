# feat: Compare SigLIP SOTA Model Against Original CLIP

**Created:** 2026-01-12
**Type:** Enhancement
**Priority:** P2

---

## Overview

Replace the extensive fine-tuning baselines (ViT, ResNet-50, Inception-V3) with a single, more meaningful comparison: the original **CLIP ViT-L/14@336px** (2021) vs a modern **SigLIP** variant (2023-2024). This provides a cleaner experiment design that answers: *"How much has vision-language model quality improved since the original paper?"*

---

## Problem Statement

The current plan includes 4 fine-tuning baselines from the paper:
- ViT Fine-tuning
- ResNet-50 Fine-tuning
- Vanilla Inception-V3
- DeepOtolith Inception-V3

This adds significant complexity without scientific value for your use case. A more meaningful comparison is **original CLIP vs modern SOTA** using the same few-shot methodology.

---

## Research Findings

### Model Comparison Table

| Model | Year | ImageNet Acc | Resolution | Embedding Dim | OpenCLIP Support |
|-------|------|--------------|------------|---------------|------------------|
| **CLIP ViT-L/14** (paper) | 2021 | ~75.5% | 336×336 | 768 | ✅ Native |
| **SigLIP SO400M-14-384** | 2023 | ~83.1% | 384×384 | 1152 | ✅ via timm |
| **SigLIP2 SO400M-14-378** | 2025 | ~84% | 378×378 | 1152 | ✅ via timm |
| **SigLIP2 SO400M-16-384** | 2025 | 84.1% | 384×384 | 1152 | ✅ via timm |
| **SigLIP2 SO400M-16-512** | 2025 | ~85% | 512×512 | 1152 | ✅ via timm |


### SigLIP2 Variants Deep Dive (Feb 2025)

SigLIP2 represents a significant evolution with improved semantic understanding, localization, and dense features.

#### Available SigLIP2 SO400M Variants

| Variant | Patch Size | Resolution | HF Hub Path |
|---------|------------|------------|-------------|
| SO400M-14-SigLIP2 | 14 | 224×224 | `hf-hub:timm/ViT-SO400M-14-SigLIP2` |
| SO400M-14-SigLIP2-378 | 14 | 378×378 | `hf-hub:timm/ViT-SO400M-14-SigLIP2-378` |
| SO400M-16-SigLIP2-256 | 16 | 256×256 | `hf-hub:timm/ViT-SO400M-16-SigLIP2-256` |
| SO400M-16-SigLIP2-384 | 16 | 384×384 | `hf-hub:timm/ViT-SO400M-16-SigLIP2-384` |
| SO400M-16-SigLIP2-512 | 16 | 512×512 | `hf-hub:timm/ViT-SO400M-16-SigLIP2-512` |

#### SigLIP2 Key Improvements Over SigLIP1

1. **Text Decoder**: Added for bounding box prediction and region-specific captions
2. **Global-Local Loss**: Self-distillation where student matches teacher on partial views
3. **Masked Prediction**: 50% patch masking improves spatial awareness
4. **NaFlex Variants**: Dynamic resolution support (preserves native aspect ratios)
5. **Better Localization**: Surpasses SigLIP and even larger OpenCLIP G/14 on segmentation

### Recommended Model: **SigLIP2 SO400M-14-378**

**Primary Recommendation: SigLIP2** (Feb 2025 release)

**Why SigLIP2 over SigLIP1:**

1. **Latest SOTA** - Released Feb 2025, most recent vision-language encoder
2. **Better localization** - Improved spatial awareness via masked prediction
3. **Same API** - Drop-in replacement via OpenCLIP/timm
4. **~9% higher ImageNet accuracy** than original CLIP (~84% vs ~75.5%)
5. **Well documented** - Google Research, Apache 2.0 license

**Fallback: SigLIP SO400M-14-384** (2023) if SigLIP2 dependency issues arise.

### SigLIP/SigLIP2 Technical Details

**What makes SigLIP better than CLIP:**

```
CLIP Loss:  softmax over batch (requires large batches, ~32K)
SigLIP Loss: sigmoid per image-text pair (works with smaller batches)
```

**SigLIP2 additional improvements:**
- Global-Local self-distillation loss
- 50% patch masking for spatial awareness
- Text decoder for region understanding

### Loading Models with OpenCLIP

```python
import open_clip

# Original CLIP (from paper - baseline)
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14-336',
    pretrained='openai'
)

# SigLIP2 SOTA (RECOMMENDED - Feb 2025)
siglip2_model, siglip2_preprocess = open_clip.create_model_from_pretrained(
    'hf-hub:timm/ViT-SO400M-14-SigLIP2-378'
)
# Note: Requires open-clip-torch>=2.31.0 and timm>=1.0.15

# Alternative: SigLIP1 (2023) - more stable dependencies
siglip_model, _, siglip_preprocess = open_clip.create_model_and_transforms(
    'ViT-SO400M-14-SigLIP-384',
    pretrained='webli'
)
# Note: Requires open-clip-torch>=2.23.0 and timm>=0.9.8
```

---

## Proposed Solution

### Simplified Model Comparison

Instead of 6 models, compare **2-3 models** using identical methodology:

| Model | Role | Methodology |
|-------|------|-------------|
| CLIP ViT-L/14@336px | Baseline (paper) | Frozen encoder → Ridge/SVC |
| SigLIP2 SO400M-14-378 | SOTA comparison (primary) | Frozen encoder → Ridge/SVC |
| SigLIP SO400M-14-384 | SOTA comparison (fallback) | Frozen encoder → Ridge/SVC |

Both use the **exact same pipeline**:
1. Extract embeddings with frozen encoder
2. Train Ridge Regression (α=6.0) on embeddings
3. 10-fold stratified cross-validation
4. Report accuracy, ±1 accuracy, RMSE

### Architecture Changes

```
# BEFORE (complex)
src/models/
├── clip_extractor.py      # CLIP only
├── shallow_models.py      # Ridge, SVC
└── finetuning.py          # ViT, ResNet, Inception (REMOVE)

# AFTER (simplified)
src/models/
├── feature_extractor.py   # Generic: CLIP, SigLIP, any OpenCLIP model
└── shallow_models.py      # Ridge, SVC (unchanged)
```

---

## Technical Considerations

### Embedding Dimension Difference

| Model | Embedding Dim |
|-------|---------------|
| CLIP ViT-L/14 | 768 |
| SigLIP SO400M-14 | 1152 |

**Impact:** Ridge regression handles different dimensions automatically. No code changes needed.

### Resolution Difference

| Model | Input Resolution |
|-------|-----------------|
| CLIP ViT-L/14@336px | 336×336 |
| SigLIP SO400M-14-384 | 384×384 |

**Impact:** Each model has its own preprocessing. OpenCLIP handles this automatically via `create_model_and_transforms()`.

### Memory Considerations (MPS)

| Model | Params | Est. Memory |
|-------|--------|-------------|
| CLIP ViT-L/14 | 428M | ~2GB |
| SigLIP SO400M-14 | 400M | ~2GB |

Both fit comfortably on Apple Silicon with 8GB+ unified memory.

---

## Acceptance Criteria

### Functional Requirements
- [ ] Support loading any OpenCLIP-compatible model by name
- [ ] Extract embeddings using same API for CLIP and SigLIP
- [ ] Cache embeddings separately per model (different dimensions)
- [ ] Run identical 10-fold CV on both model embeddings
- [ ] Generate comparison table and plots

### Code Changes Required
- [ ] Rename `clip_extractor.py` → `feature_extractor.py`
- [ ] Add model configuration in `config.yaml`
- [ ] Update `run_experiment.py` to iterate over models
- [ ] Remove `finetuning.py` (not needed)

### Output Requirements
- [ ] Results table comparing CLIP vs SigLIP
- [ ] Bar chart showing accuracy comparison
- [ ] Statistical significance test (paired t-test across folds)

---

## Implementation Plan

### Phase 1: Generalize Feature Extractor

**File:** `src/models/feature_extractor.py`

```python
import open_clip

class VisionFeatureExtractor:
    """Generic feature extractor for any OpenCLIP model."""

    SUPPORTED_MODELS = {
        # (model_id, pretrained, loader_type)
        'clip-vit-l-14-336': ('ViT-L-14-336', 'openai', 'openclip'),
        'siglip2-so400m-14-378': ('hf-hub:timm/ViT-SO400M-14-SigLIP2-378', None, 'hf-hub'),
        'siglip2-so400m-16-384': ('hf-hub:timm/ViT-SO400M-16-SigLIP2-384', None, 'hf-hub'),
        'siglip-so400m-14-384': ('ViT-SO400M-14-SigLIP-384', 'webli', 'openclip'),
    }

    def __init__(self, model_name: str = 'clip-vit-l-14-336'):
        model_id, pretrained, loader = self.SUPPORTED_MODELS[model_name]

        if loader == 'hf-hub':
            # SigLIP2 models via HuggingFace Hub
            self.model, self.preprocess = open_clip.create_model_from_pretrained(model_id)
        else:
            # Native OpenCLIP models (CLIP, SigLIP1)
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_id, pretrained=pretrained
            )
        # ... rest of implementation
```

### Phase 2: Update Configuration

**File:** `configs/config.yaml`

```yaml
models:
  - name: clip-vit-l-14-336
    description: "Original CLIP from paper (OpenAI, 2021)"
    model_id: "ViT-L-14-336"
    pretrained: "openai"
    loader: "openclip"  # native OpenCLIP

  - name: siglip2-so400m-14-378
    description: "SigLIP2 SOTA (Google, Feb 2025)"
    model_id: "hf-hub:timm/ViT-SO400M-14-SigLIP2-378"
    pretrained: null  # included in hf-hub path
    loader: "hf-hub"  # use create_model_from_pretrained

  - name: siglip-so400m-14-384
    description: "SigLIP v1 (Google, 2023) - fallback"
    model_id: "ViT-SO400M-14-SigLIP-384"
    pretrained: "webli"
    loader: "openclip"
```

### Phase 3: Run Experiments

```bash
# Extract features for both models
python scripts/extract_features.py --model clip-vit-l-14-336
python scripts/extract_features.py --model siglip-so400m-14-384

# Run 10-fold CV on both
python scripts/run_experiment.py --models clip-vit-l-14-336,siglip-so400m-14-384
```

---

## Expected Results

### Hypothesis

SigLIP should outperform CLIP due to:
1. Better pre-training methodology (sigmoid loss)
2. More training data (WebLI vs WIT-400M)
3. Higher base ImageNet accuracy (+8%)

### Predicted Outcomes

| Model | Accuracy | ±1 Accuracy | RMSE |
|-------|----------|-------------|------|
| CLIP ViT-L/14 (baseline) | ~50% | ~94% | ~0.85 |
| SigLIP SO400M-14-384 (2023) | ~53-56% | ~95-96% | ~0.75-0.80 |
| SigLIP2 SO400M-14-378 (2025) | ~54-58% | ~95-97% | ~0.70-0.78 |

*Note: Improvements may be modest since otolith age classification is domain-specific and may not benefit as much from general vision improvements. However, SigLIP2's improved spatial awareness (via masked prediction) could provide better discrimination of otolith ring patterns.*

---

## Dependencies

### Python Packages

```toml
# pyproject.toml additions
dependencies = [
    "open-clip-torch>=2.31.0",  # Required for SigLIP2 (use >=2.23.0 for SigLIP1 only)
    "timm>=1.0.15",             # Required for SigLIP2 models (use >=0.9.8 for SigLIP1)
]
```

**Version requirements by model:**
| Model | open-clip-torch | timm |
|-------|-----------------|------|
| CLIP ViT-L/14 | ≥2.0.0 | N/A |
| SigLIP (2023) | ≥2.23.0 | ≥0.9.8 |
| SigLIP2 (2025) | ≥2.31.0 | ≥1.0.15 |

### Model Downloads

Models are downloaded automatically on first use:
- CLIP: ~1.7GB
- SigLIP: ~1.6GB

---

## Alternative Approaches Considered

### 1. EVA-CLIP Instead of SigLIP

**Pros:** Same embedding dimension (768) as CLIP
**Cons:** Less improvement over CLIP (~80% vs ~83% ImageNet)
**Decision:** SigLIP has better performance gains

### 2. DINOv2 Instead of SigLIP

**Pros:** State-of-the-art self-supervised features
**Cons:** Different API (not OpenCLIP), no text encoder
**Decision:** SigLIP maintains apples-to-apples comparison

### 3. Keep Fine-tuning Baselines

**Pros:** Matches paper methodology exactly
**Cons:** Adds 3-4x implementation complexity, slower experiments
**Decision:** Not needed for your scientific question

---

## References

### Papers
- [SigLIP 2: Multilingual Vision-Language Encoders](https://arxiv.org/abs/2502.14786) (Google, Feb 2025)
- [SigLIP: Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) (Zhai et al., 2023)
- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)
- [Original Otolith Paper](https://doi.org/10.1016/j.ecoinf.2023.102046) (Sigurðardóttir et al., 2023)

### Code & Models
- [OpenCLIP GitHub](https://github.com/mlfoundations/open_clip)
- [SigLIP2 HuggingFace Blog](https://huggingface.co/blog/siglip2)
- [SigLIP2 SO400M-14-378 on HuggingFace](https://huggingface.co/timm/ViT-SO400M-14-SigLIP2-378)
- [SigLIP2 SO400M-16-384 on HuggingFace](https://huggingface.co/timm/ViT-SO400M-16-SigLIP2-384)
- [SigLIP (v1) on HuggingFace](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384)

### Benchmarks
- [OpenCLIP Model Comparison](https://github.com/mlfoundations/open_clip#pretrained-model-interface)
