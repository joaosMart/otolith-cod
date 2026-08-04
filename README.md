# Otolith Age Prediction

LoRA fine-tuning of vision foundation models for Atlantic cod (*Gadus morhua*) otolith age classification. Compares frozen encoders with LoRA-adapted SigLIP2, using Ridge regression on concatenated image embeddings and biological metadata.

## Overview

This project extends the approach from [Sigurðardóttir et al. (2023)](https://doi.org/10.1016/j.ecoinf.2023.102046), which uses CLIP embeddings with Ridge regression to predict fish age from otolith images. We compare multiple vision foundation models and show that LoRA fine-tuning of SigLIP2 significantly improves performance.

| Model | Source | Embedding Dim | Image Size |
|-------|--------|---------------|------------|
| CLIP ViT-L/14 | OpenAI (2021) | 768 | 336 |
| SigLIP2 SO400M | Google (2025) | 1152 | 384 |
| DINOv2 ViT-L/14 | Meta (2024) | 1024 | 518 |
| Perception Encoder L/14 | Meta (2025) | 1024 | 336 |

The pipeline supports two modes:
- **Frozen encoder**: images pass through a frozen vision encoder, embeddings are cached, and a Ridge classifier is trained on `[embedding | metadata]`.
- **LoRA fine-tuned**: LoRA adapters are added to the encoder, trained with CORN ordinal loss, then embeddings are re-extracted and classified with Ridge regression.

## Project Structure

```
├── configs/config.yaml              # Experiment parameters
├── src/
│   ├── data/                        # Dataset loading and splitting
│   │   ├── dataset.py               # OtolithDataset class
│   │   └── splits.py                # Stratified splits, ID-based persistence
│   ├── features/                    # Feature extraction
│   │   ├── extractor.py             # Multi-model encoder + CLAHE
│   │   ├── cache.py                 # Embedding caching
│   │   └── metadata.py              # Tabular feature augmentation
│   ├── training/                    # Classifier training
│   │   ├── classifiers.py           # Ridge, SVC classifiers
│   │   ├── cross_validation.py      # CV routines
│   │   └── hyperparameter_search.py # Alpha search
│   ├── evaluation/                  # Metrics and visualization
│   │   ├── metrics.py               # Accuracy, F1, RMSE, bootstrap CIs
│   │   ├── plots.py                 # Confusion matrices, ROC, PR curves
│   │   ├── error_analysis.py        # Per-class error breakdown
│   │   ├── learning_curves.py       # Learning curve generation
│   │   ├── feature_importance.py    # SHAP and permutation importance
│   │   └── data_quality.py          # Cleanlab integration
│   └── utils/                       # Device selection, logging, config
├── scripts/                         # CLI entry points (see Usage below)
├── paper/                           # LaTeX manuscript and figures
│   ├── sections/                    # intro.tex, methods.tex, results.tex, discussion.tex
│   └── figures/                     # Generated figures for the paper
├── notebooks/                       # Exploration and analysis
├── outputs/                         # Embeddings, models, results (not tracked)
└── otolith_images/                  # Image data (not tracked)
```

## Setup

Requires Python >= 3.11.

```bash
pip install -e .

# Optional extras
pip install -e ".[dev]"       # Testing and linting
pip install -e ".[notebook]"  # Jupyter support
pip install -e ".[analysis]"  # Cleanlab, SHAP
pip install -e ".[pe]"        # Perception Encoder (Meta)
```

## Usage

### 1. Preprocessing (optional)

Apply CLAHE contrast enhancement to images before extraction.

```bash
python scripts/clahe_pre_process.py \
    --input-dir otolith_images/segmented_images \
    --output-dir otolith_images/segmented_images_clahe

# Double CLAHE for stronger enhancement
python scripts/clahe_pre_process.py \
    --input-dir otolith_images/segmented_images \
    --output-dir otolith_images/segmented_images_clahe_repeat \
    --repeat-clahe
```

### 2. Extract embeddings (frozen encoder)

Pass otolith images through a frozen vision encoder to produce cached `.npz` embeddings.

```bash
# Default model (SigLIP2)
python scripts/extract_features.py

# Choose a specific model
python scripts/extract_features.py --model clip-vit-l-14-336
python scripts/extract_features.py --model dinov2-vitl14-reg
python scripts/extract_features.py --model pe-core-l14-336

# With CLAHE enhancement (applied on-the-fly)
python scripts/extract_features.py --model siglip2-so400m-14-384 --clahe

# Force re-extraction (ignores cache)
python scripts/extract_features.py --model siglip2-so400m-14-384 --force
```

Available models: `clip-vit-l-14-336`, `siglip2-so400m-14-384`, `dinov2-vitl14-reg`, `pe-core-l14-336`.

Embeddings are saved to `outputs/embeddings/` or `outputs/segmented_embeddings/`.

### 3. LoRA fine-tuning

Fine-tune a vision encoder with LoRA adapters and a CORN ordinal regression head.

```bash
# Fine-tune SigLIP2 (recommended)
python scripts/finetune_dinov2_lora.py --model siglip2

# Fine-tune DINOv2
python scripts/finetune_dinov2_lora.py --model dinov2

# Custom hyperparameters
python scripts/finetune_dinov2_lora.py --model siglip2 --epochs 50 --lr 5e-5
```

LoRA adapters and the CORN head are saved to `outputs/lora/<model>/`. A `split.json` recording the train/test split is also saved for reproducibility.

### 4. Extract embeddings (LoRA-adapted encoder)

Extract embeddings using the LoRA-adapted model.

```bash
# SigLIP2 with LoRA
python scripts/extract_features_lora.py --model siglip2

# Custom adapter path
python scripts/extract_features_lora.py --model siglip2 \
    --adapter-path outputs/lora/siglip2/lora_adapter

# Merge LoRA weights into base model before extraction
python scripts/extract_features_lora.py --model siglip2 --merge
```

### 5. Train classifier

Train a Ridge classifier on cached embeddings with RidgeCV alpha optimization.

Two evaluation modes:
- **`splits`** (default): 10 independent stratified train/test splits, reports mean +/- std.
- **`bootstrap`**: single fixed train/test split, bootstrap resampling of the test set for 95% CIs. Use this when comparing with LoRA-adapted encoders to avoid data leakage.

```bash
# Default: 10 independent splits
python scripts/train_classifier.py \
    --embeddings outputs/segmented_embeddings/siglip2-so400m-14-384_embeddings.npz

# Bootstrap mode with LoRA split (ensures same test set)
python scripts/train_classifier.py \
    --embeddings outputs/segmented_embeddings/siglip2-so400m-14-384_clahe_embeddings.npz \
    --eval-mode bootstrap \
    --split-file outputs/lora/siglip2/split.json

# Without tabular feature augmentation
python scripts/train_classifier.py --no-add-tabular
```

Results are saved to `outputs/results/<name>/<eval-mode>/`.

### 6. Evaluate CORN head directly

Evaluate the LoRA-adapted model using its CORN ordinal regression head (without Ridge).

```bash
python scripts/evaluate_corn_head.py --model siglip2

python scripts/evaluate_corn_head.py --model dinov2 \
    --adapter-path outputs/lora/lora_adapter \
    --corn-head outputs/lora/corn_head.pt
```

### 7. Analyze performance

Generate confusion matrices, ROC curves, precision-recall curves, and error analysis.

```bash
# Basic analysis
python scripts/analyze_performance.py \
    --results outputs/results/shallow_siglip2/bootstrap/results.json

# Full analysis with visualizations
python scripts/analyze_performance.py \
    --results outputs/results/shallow_siglip2/bootstrap/results.json \
    --embeddings outputs/segmented_embeddings/siglip2-so400m-14-384_embeddings.npz

# Include learning curves and data quality analysis
python scripts/analyze_performance.py \
    --results outputs/results/shallow_siglip2/splits/results.json \
    --embeddings outputs/segmented_embeddings/siglip2-so400m-14-384_embeddings.npz \
    --learning-curves --cleanlab
```

### 8. Feature importance analysis

SHAP values, permutation importance, and forward selection to understand which features drive predictions.

```bash
python scripts/analyze_feature_importance.py \
    --embeddings outputs/segmented_embeddings/siglip2-so400m-14-384_clahe_embeddings.npz \
    --results outputs/results/siglip2-clahe/bootstrap/results.json \
    --output-dir outputs/results/siglip2-clahe/bootstrap/feature_importance
```

Requires `pip install -e ".[analysis]"` for SHAP.

### 9. Compare models

Compare frozen vs LoRA-adapted model results side by side.

```bash
python scripts/compare_models.py \
    --frozen-results outputs/results/siglip2-clahe/bootstrap/results.json \
    --lora-results outputs/results/siglip2-lora/bootstrap/results.json
```

### 10. Visualize attention maps

Generate attention heatmaps comparing frozen vs LoRA-adapted models.

```bash
python scripts/visualize_attention.py
python scripts/visualize_attention.py --n-samples 10 --layer -1
```

### 11. Generate paper figures

Generate publication-ready figures from bootstrap results.

```bash
python scripts/generate_paper_figures.py
```

Figures are saved to `paper/figures/`.

## Reference Results

From Sigurðardóttir et al. (2023) on cod otoliths (n=1170):

| Metric | Value |
|--------|-------|
| Accuracy | 50.47 +/- 2.37% |
| Accuracy +/-1 | 94.10 +/- 1.24% |
| RMSE | 0.84 +/- 0.04 |

## License

MIT
