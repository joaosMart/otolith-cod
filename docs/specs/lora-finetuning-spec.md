# LoRA Fine-Tuning with CORN Ordinal Regression

## Overview

We fine-tune frozen Vision Transformers (DINOv2, SigLIP2) using Low-Rank Adaptation (LoRA) adapters paired with a CORN ordinal regression head for otolith age classification. Only the LoRA adapter weights and classification head are trained; the base model weights remain frozen.

## Architecture

### Backbone Models

| Model | HuggingFace ID | Embedding Dim | Pooling Strategy |
|-------|---------------|---------------|------------------|
| DINOv2-L (registers) | `facebook/dinov2-with-registers-large` | 1024 | CLS token (`last_hidden_state[:, 0, :]`) |
| SigLIP2-So400m | `google/siglip2-so400m-patch14-384` | 1152 | MAP head (`vision_model.pooler_output`) — learned attention pooling over patch tokens, no CLS token |

### LoRA Adapters

Low-rank matrices are injected into the attention and MLP layers of the vision backbone:

- **DINOv2 targets:** `query`, `key`, `value`, `fc1`, `fc2`
- **SigLIP2 targets:** `q_proj`, `k_proj`, `v_proj`, `fc1`, `fc2`
- **SigLIP2 note:** LoRA is applied only to `vision_model` to avoid touching the text tower

Default hyperparameters: rank=16, alpha=32, dropout=0.1.

### CORN Head

A single linear layer (`embedding_dim → K-1`) producing K-1 logits for K ordinal age classes (ages 1–10, so 9 logits). CORN (Conditional Ordinal Regression with Neural networks) ensures rank-consistent predictions — the model learns conditional probabilities P(age > k | age > k-1) rather than independent class probabilities.

Reference: Shi et al. (2021), "Deep Neural Networks for Rank-Consistent Ordinal Regression Based on Conditional Probabilities."

## Training Pipeline

### Data Flow

1. **CLAHE preprocessing** — adaptive histogram equalization on grayscale, replicated to 3 channels
2. **Augmentation** (train only) — random horizontal flip, rotation (±15°), Gaussian blur
3. **Resize + normalize** — to model's native resolution with model-specific mean/std

### Data Split

The dataset is split using stratified sampling:
- 85% train pool / 15% test (held out, matching the frozen extraction pipeline)
- Train pool further split 80/20 into FT-train / FT-val

### Optimization

- **Optimizer:** AdamW with weight decay 0.01
- **Learning rates:** base LR for backbone LoRA params, 10x LR for the CORN head
- **Scheduler:** cosine annealing with linear warmup (3 epochs default)
- **Gradient clipping:** max norm 1.0
- **Early stopping:** patience on validation MAE (default 5 epochs)
- **Loss:** CORN loss (applied to 0-indexed age labels)

### What Gets Updated During Training

| Component | Trainable | Saved |
|-----------|-----------|-------|
| Base ViT weights | Frozen | No |
| LoRA adapter weights | Yes | `lora_adapter/` |
| CORN linear head | Yes | `corn_head.pt` |

## Inference / Feature Extraction

### Frozen (no LoRA)

Features are extracted via `extractor.py` using the pretrained weights only:

- **DINOv2:** `model(pixel_values) → last_hidden_state` — extracts CLS token, patch features, and mean-pooled patches separately
- **SigLIP2:** `model.get_image_features(pixel_values)` — returns MAP-pooled + projected embedding

### With LoRA Adapters

The LoRA adapter weights are loaded back onto the base model. Feature extraction then follows the same forward pass but with the adapted weights active, producing modified embeddings that reflect the fine-tuned representations.

The CORN head is only used during fine-tuning for the ordinal loss signal — at inference time, only the backbone embeddings are extracted for downstream classification (e.g., SVM, kNN).

## Metrics

Tracked during training and reported at completion:

- **Exact-match accuracy** — predicted age == true age
- **MAE** — mean absolute error in age classes
- **Within-1 accuracy** — predicted age within ±1 of true age
