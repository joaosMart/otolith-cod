# Otolith Age Prediction

Few-shot learning for Atlantic cod (*Gadus morhua*) otolith age determination using frozen vision encoders and shallow classifiers.

## Overview

This project replicates and extends the approach from [Sigurðardóttir et al. (2023)](https://doi.org/10.1016/j.ecoinf.2023.102046), which uses CLIP embeddings with Ridge regression to predict fish age from otolith images. We compare multiple vision foundation models:

| Model | Source | Embedding Dim | Image Size |
|-------|--------|---------------|------------|
| CLIP ViT-L/14 | OpenAI (2021) | 768 | 336 |
| SigLIP2 SO400M | Google (2025) | 1152 | 384 |
| DINOv2 ViT-L/14 | Meta (2024) | 1024 | 518 |
| Perception Encoder L/14 | Meta (2025) | 1024 | 336 |

Images are passed through a frozen encoder, and the resulting embeddings are fed to a Ridge regressor (or SVC) for age classification across ages 1–10.

## Project Structure

```
├── configs/config.yaml       # All experiment parameters
├── src/
│   ├── data/                 # Dataset loading and splitting
│   ├── features/             # Feature extraction (multi-model)
│   ├── training/             # Classifier training (Ridge, SVC)
│   ├── evaluation/           # Metrics and visualization
│   └── utils/                # Device selection, logging
├── scripts/
│   ├── extract_features.py   # Extract embeddings from images
│   ├── train_classifier.py   # Train and evaluate classifiers
│   ├── analyze_performance.py
│   └── analyze_feature_importance.py
├── notebooks/                # Exploration and analysis
├── outputs/                  # Embeddings, models, figures
└── otolith_images/           # Image data (not tracked)
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

The pipeline has four steps. All configuration lives in `configs/config.yaml`.

### 1. Extract embeddings

Pass otolith images through a frozen vision encoder to produce cached `.npz` embeddings.

```bash
# Default model (SigLIP2)
python scripts/extract_features.py

# Choose a specific model
python scripts/extract_features.py --model clip-vit-l-14-336
python scripts/extract_features.py --model dinov2-vitl14-reg
python scripts/extract_features.py --model pe-core-l14-336

# Use raw images instead of segmented (default: segmented_images)
python scripts/extract_features.py --images-path raw_images

# With CLAHE image enhancement
python scripts/extract_features.py --model siglip2-so400m-14-384 --clahe

# Force re-extraction (ignores cache)
python scripts/extract_features.py --model siglip2-so400m-14-384 --force
```

Available models: `clip-vit-l-14-336`, `siglip2-so400m-14-384`, `dinov2-vitl14-reg`, `pe-core-l14-336`.
Image sources: `segmented_images` (default), `raw_images`.

Embeddings are saved to `outputs/embeddings/` (raw) or `outputs/segmented_embeddings/` (segmented).

### 2. Train classifier

Train a Ridge regressor on the cached embeddings with RidgeCV alpha optimization. Runs multiple independent experiments with stratified splits.

```bash
# Default: SigLIP2 embeddings, 10 experiments, tabular features included
python scripts/train_classifier.py

# Specify embeddings and output directory
python scripts/train_classifier.py \
    --embeddings outputs/segmented_embeddings/siglip2-so400m-14-384_clahe_embeddings.npz \
    --output-dir outputs/results/my_experiment

# Custom experiment seeds and alpha range
python scripts/train_classifier.py --seeds 82,15,4 --alpha-log-min 1.0 --alpha-log-max 3.5

# Without tabular feature augmentation
python scripts/train_classifier.py --no-add-tabular
```

Outputs `results.json`, `summary.csv`, and `splits.json` to the output directory.

### 3. Analyze performance

Generate confusion matrices, ROC curves, precision-recall curves, and error analysis.

```bash
# Basic analysis (just prints aggregated metrics)
python scripts/analyze_performance.py --results outputs/results/shallow_siglip2/results.json

# Full analysis with visualizations
python scripts/analyze_performance.py \
    --results outputs/results/shallow_siglip2/results.json \
    --embeddings outputs/segmented_embeddings/siglip2-so400m-14-384_clahe_embeddings.npz

# Include learning curves and data quality analysis
python scripts/analyze_performance.py \
    --results outputs/results/shallow_siglip2/results.json \
    --embeddings outputs/segmented_embeddings/siglip2-so400m-14-384_clahe_embeddings.npz \
    --learning-curves --cleanlab
```

Figures are saved to `outputs/analysis/figures/`.

### 4. Feature importance analysis

SHAP values, permutation importance, and forward selection ablation to understand which features drive predictions.

```bash
python scripts/analyze_feature_importance.py \
    --embeddings outputs/segmented_embeddings/siglip2-so400m-14-384_clahe_embeddings.npz \
    --results outputs/results/shallow_siglip2/results.json \
    --output-dir outputs/feature_importance
```

Requires `pip install -e ".[analysis]"` for SHAP.

## Reference Results

From Sigurðardóttir et al. (2023) on cod otoliths (n=1170):

| Metric | Value |
|--------|-------|
| Accuracy | 50.47 ± 2.37% |
| Accuracy ±1 | 94.10 ± 1.24% |
| RMSE | 0.84 ± 0.04 |

## License

MIT
