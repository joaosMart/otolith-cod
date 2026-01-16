# Otolith Age Prediction - Work Log

## Project Goal

Reproduce the results from the paper "Otolith age determination with a simple computer vision based few-shot learning method" (Sigurðardóttir et al., 2023) using cod otolith data.

---

## Paper Summary

The paper proposes a few-shot learning approach using:
1. **CLIP vision encoder** (ViT-L/14@336px) as a frozen feature extractor
2. **Ridge Regression** or **SVC** trained on CLIP embeddings
3. Compared against fine-tuning approaches (ViT, ResNet-50, Inception-V3)

### Paper Results for Atlantic Cod
| Metric | Value |
|--------|-------|
| Accuracy | 50.47% ± 2.37% |
| ±1 Accuracy | 94.10% ± 1.24% |
| RMSE | 0.84 ± 0.04 |

---

## Dataset Overview

**Location:** `otolith_images/`

**Total Images:** 8,637 cod otolith images

### Age Distribution
| Age | Count | Age | Count |
|-----|-------|-----|-------|
| 1 | 110 | 10 | 220 |
| 2 | 223 | 11 | 79 |
| 3 | 348 | 12 | 50 |
| 4 | 768 | 13 | 26 |
| 5 | 1,695 | 14 | 17 |
| 6 | 1,941 | 15 | 11 |
| 7 | 1,477 | 16 | 7 |
| 8 | 927 | 17 | 2 |
| 9 | 736 | | |

**Note:** Significant class imbalance - ages 5-7 dominate (~60% of data), ages 14-17 have very few samples (<40 total).

**Recommendation:** Following the paper, clip ages to [1, 10] range (older ages merged into age 10).

---

## Project Architecture

```
otolith-cod/
├── pyproject.toml                 # Project configuration with dependencies
├── WORKLOG.md                     # This file - implementation plan and progress
├── configs/
│   └── config.yaml                # Hyperparameters and paths configuration
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # OtolithDataset class for PyTorch
│   │   ├── preprocessing.py       # Image preprocessing pipelines
│   │   └── splits.py              # Stratified train/val/test splitting
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clip_extractor.py      # CLIP feature extraction
│   │   ├── shallow_models.py      # Ridge Regression and SVC wrappers
│   │   └── finetuning.py          # ViT, ResNet-50, Inception-V3 fine-tuning
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training loop for fine-tuning models
│   │   └── cross_validation.py    # 10-fold stratified CV implementation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Accuracy, RMSE, ±1 accuracy, F1, etc.
│   │   └── analysis.py            # Per-class analysis and error analysis
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plots.py               # Confusion matrices, learning curves
│   │   └── distributions.py       # Age distribution, prediction error plots
│   └── utils/
│       ├── __init__.py
│       └── device.py              # MPS/CUDA/CPU device selection
├── scripts/
│   ├── extract_features.py        # Extract CLIP embeddings for all images
│   ├── train_shallow.py           # Train Ridge/SVC on CLIP embeddings
│   ├── train_finetune.py          # Fine-tune ViT/ResNet/Inception
│   ├── evaluate.py                # Run evaluation on test set
│   └── run_experiment.py          # Full 10-fold CV experiment
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA and class distribution analysis
│   ├── 02_feature_analysis.ipynb  # CLIP embedding visualization (t-SNE/UMAP)
│   └── 03_results_analysis.ipynb  # Final results and comparisons
├── tests/
│   ├── test_dataset.py
│   ├── test_metrics.py
│   └── test_models.py
└── outputs/
    ├── embeddings/                # Cached CLIP embeddings
    ├── models/                    # Saved model checkpoints
    ├── results/                   # Evaluation results (JSON/CSV)
    └── figures/                   # Generated plots
```

---

## Key Hyperparameters (from paper)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Age range | [1, 10] | Clip older ages to 10 |
| Train/Val/Test split | 65% / 15% / 20% | Stratified by age |
| Cross-validation | 10-fold stratified | Report mean ± std |
| CLIP model | ViT-L/14@336px | 768-dim embeddings |
| Ridge α | 6.0 | L2 regularization |
| SVC C | 0.1 | Inverse regularization |
| SVC kernel | linear | One-vs-one scheme |
| Fine-tune LR (ViT) | 1e-5 | AdamW optimizer |
| Fine-tune LR (ResNet) | 1e-4 | AdamW optimizer |
| Fine-tune LR (Inception) | 1e-5 | Adam optimizer |
| Batch size | 16 | For fine-tuning |
| Early stopping patience | 10 epochs | Based on validation loss |

---

## Implementation Phases

### Phase 1: Setup and Data Exploration
- [x] Create project structure
- [x] Create pyproject.toml with dependencies
- [ ] Create config.yaml with hyperparameters
- [ ] Verify image formats and quality
- [ ] Create data exploration notebook
- [ ] Plot age distribution

### Phase 2: Data Processing Module
- [ ] Implement `OtolithDataset` class
- [ ] Implement CLIP preprocessing transforms
- [ ] Implement stratified splitting (65/15/20)
- [ ] Implement 10-fold CV split generator
- [ ] Add class weight computation for imbalanced data

### Phase 3: CLIP Feature Extraction
- [ ] Implement `CLIPFeatureExtractor` class
- [ ] Extract embeddings for all 8,637 images
- [ ] Cache embeddings to disk for reuse
- [ ] Verify embedding dimensions (768-D)

### Phase 4: Shallow Models (Few-Shot Approach)
- [ ] Implement `CLIPRidgeRegressor` (α=6.0)
- [ ] Implement `CLIPSVClassifier` (C=0.1)
- [ ] Run 10-fold cross-validation
- [ ] Compare results with paper benchmarks

### Phase 5: Fine-Tuning Baselines (Optional)
- [ ] Implement ViT fine-tuning
- [ ] Implement ResNet-50 fine-tuning
- [ ] Implement Inception-V3 fine-tuning (Vanilla + DeepOtolith)
- [ ] Train with class-weighted cross-entropy loss
- [ ] Run cross-validation for comparison

### Phase 6: Evaluation and Visualization
- [ ] Implement all metrics (accuracy, ±1 accuracy, RMSE, F1)
- [ ] Generate confusion matrices
- [ ] Plot prediction error distributions
- [ ] Create model comparison bar charts
- [ ] Generate learning curves
- [ ] Document final results

---

## Core Components Design

### 1. Data Processing Module

**`src/data/dataset.py`** - OtolithDataset class
- Loads images from `{age}/image.jpg` structure
- Clips ages to specified range [1, 10]
- Applies transforms (CLIP or ImageNet preprocessing)
- Computes class weights for imbalanced data

**`src/data/splits.py`** - Data splitting
- Stratified train/val/test split (65/15/20)
- 10-fold stratified cross-validation
- Ensures each fold has representative age distribution

**`src/data/preprocessing.py`** - Image transforms
- CLIP preprocessing: Resize 336x336, center crop, CLIP normalization
- Fine-tuning preprocessing: Resize 224x224 (or 299 for Inception), ImageNet normalization
- Optional augmentation for fine-tuning (random crop, flip, rotation)

### 2. Models Module

**`src/models/clip_extractor.py`** - CLIP feature extraction
- Uses OpenCLIP's ViT-L/14@336px model
- Extracts 768-dimensional embeddings
- L2 normalization of features
- Batch processing with progress bar
- Save/load embeddings to/from disk

**`src/models/shallow_models.py`** - Ridge and SVC
- `CLIPRidgeRegressor`: Ridge regression with α=6.0
  - Outputs continuous age predictions
  - Round to integer for classification metrics
  - Use raw predictions for RMSE
- `CLIPSVClassifier`: Linear SVC with C=0.1
  - One-vs-one multiclass scheme
  - Balanced class weights

**`src/models/finetuning.py`** - Deep learning baselines
- `ViTFineTuner`: HuggingFace ViT with new classifier head
- `ResNet50FineTuner`: HuggingFace ResNet-50
- `InceptionV3FineTuner`: Keras Inception-V3 (Vanilla and DeepOtolith variants)

### 3. Training Module

**`src/training/trainer.py`** - Fine-tuning trainer
- Class-weighted cross-entropy loss
- AdamW optimizer (Adam for Inception)
- Cosine annealing learning rate schedule
- Early stopping based on validation loss
- MPS (Apple Silicon) support

**`src/training/cross_validation.py`** - Cross-validation
- 10-fold stratified CV for shallow models
- Aggregates results across folds
- Reports mean ± std for all metrics

### 4. Evaluation Module

**`src/evaluation/metrics.py`** - Metrics computation
- `compute_accuracy()`: Standard classification accuracy
- `compute_accuracy_pm1()`: Accuracy with ±1 margin of error
- `compute_rmse()`: Root mean squared error (use raw predictions)
- `compute_per_class_metrics()`: Precision, recall, F1 per age class
- `compute_confusion_matrix()`: Normalized confusion matrix

**`src/evaluation/analysis.py`** - Error analysis
- Error distribution analysis
- Over/under estimation bias
- Per-age error statistics
- Classification report generation

### 5. Visualization Module

**`src/visualization/plots.py`** - Main plots
- `plot_confusion_matrix()`: Heatmap (replicates Fig. 8)
- `plot_prediction_error_distribution()`: Histogram (replicates Fig. 7, 9)
- `plot_learning_curve()`: Accuracy vs training size (replicates Fig. 11)
- `plot_model_comparison()`: Bar chart with error bars (replicates Fig. 10)

**`src/visualization/distributions.py`** - Data visualization
- `plot_age_distribution()`: Age class histogram
- `plot_training_history()`: Loss and accuracy curves

### 6. Utilities Module

**`src/utils/device.py`** - Device management
- Auto-detect best device (MPS > CUDA > CPU)
- MPS environment configuration
- Fallback handling for unsupported operations

---

## Metal GPU (Apple Silicon) Considerations

1. **Device Selection**: Use `torch.device('mps')` for GPU acceleration
2. **DataLoader**: Use `num_workers=0` for best MPS performance
3. **pin_memory**: Not applicable for MPS (set to False)
4. **Memory Management**: Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
5. **Fallback**: Some operations may need CPU fallback

```python
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
```

---

## Expected Results

With 8,637 images (vs paper's 1,170), results should be comparable or better:

| Metric | Paper (Cod) | Expected Range |
|--------|-------------|----------------|
| Accuracy | 50.47% | 48-55% |
| ±1 Accuracy | 94.10% | 92-96% |
| RMSE | 0.84 | 0.75-0.90 |

**Factors that may improve results:**
- Larger dataset (7x more images)
- More samples in middle age ranges

**Factors that may limit results:**
- Severe class imbalance at extreme ages
- Single annotator (paper had 1 annotator for cod too)
- No additional features (length, sex, quarter)

---

## Progress Log

### 2024-01-12
- Created project directory structure
- Created pyproject.toml with all dependencies
- Created WORKLOG.md with implementation plan

---

## References

1. Sigurðardóttir, A.R., et al. (2023). "Otolith age determination with a simple computer vision based few-shot learning method." Ecological Informatics, 76, 102046.
2. Radford, A., et al. (2021). "Learning Transferable Visual Models from Natural Language Supervision." (CLIP paper)
3. OpenCLIP: https://github.com/mlfoundations/open_clip
