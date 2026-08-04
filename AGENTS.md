# AGENTS.md

## Project Overview

This project classifies the age of Atlantic cod (*Gadus morhua*) from otolith (ear bone) cross-section images. It uses frozen vision foundation models (SigLIP2, CLIP, DINOv2, Perception Encoder) as feature extractors, concatenates their embeddings with biological metadata (fish length, otolith dimensions, catch info), and trains a Ridge regression classifier. The main contribution is showing that LoRA fine-tuning of SigLIP2 improves classification accuracy by +6.3pp while shifting feature importance away from metadata toward learned visual features, demonstrated via SHAP analysis.

## Environment and Setup

- **Python**: >= 3.11 (developed on 3.11, Apple Silicon arm64 with MPS acceleration)
- **Package manager**: pip with pyproject.toml, or uv
- **Install**: `pip install -e .` from project root. Use `pip install -e ".[analysis]"` for SHAP/cleanlab support.
- **GPU**: Supports MPS (Apple Silicon), CUDA, or CPU. Auto-detected; set `device.preferred` in `configs/config.yaml` to override.
- **Data**: Otolith images go in `otolith_images/segmented_images/`. Metadata CSV (`cod_otolith_age_final_with_scale.csv`) must be in the project root. Neither is tracked in git.
- **Outputs**: All generated files (embeddings, models, results, figures) go under `outputs/`, which is gitignored.

## Reproducing the Paper Results

Run these steps in order. Each step depends on outputs from the previous one.

### Step 1: Extract frozen SigLIP2 embeddings with CLAHE

```bash
python scripts/extract_features.py --model siglip2-so400m-14-384 --clahe
```

Output: `outputs/segmented_embeddings/siglip2-so400m-14-384_clahe_embeddings.npz`

### Step 2: LoRA fine-tune SigLIP2

```bash
python scripts/finetune_dinov2_lora.py --model siglip2
```

Output: `outputs/lora/siglip2/` (adapter weights, CORN head, `split.json` with train/test split)

### Step 3: Extract LoRA-adapted embeddings

```bash
python scripts/extract_features_lora.py --model siglip2
```

Output: `outputs/segmented_embeddings/siglip2-lora_embeddings.npz`

### Step 4: Train classifiers (both conditions use the same test set)

```bash
# Frozen baseline (reuses LoRA split for fair comparison)
python scripts/train_classifier.py \
    --embeddings outputs/segmented_embeddings/siglip2-so400m-14-384_clahe_embeddings.npz \
    --eval-mode bootstrap \
    --split-file outputs/lora/siglip2/split.json

# LoRA-adapted
python scripts/train_classifier.py \
    --embeddings outputs/segmented_embeddings/siglip2-lora_embeddings.npz \
    --eval-mode bootstrap \
    --split-file outputs/lora/siglip2/split.json
```

Output: `outputs/results/siglip2-clahe/bootstrap/` and `outputs/results/siglip2-lora/bootstrap/`

### Step 5: Feature importance (SHAP) for both conditions

```bash
python scripts/analyze_feature_importance.py \
    --embeddings outputs/segmented_embeddings/siglip2-so400m-14-384_clahe_embeddings.npz \
    --results outputs/results/siglip2-clahe/bootstrap/results.json \
    --output-dir outputs/results/siglip2-clahe/bootstrap/feature_importance

python scripts/analyze_feature_importance.py \
    --embeddings outputs/segmented_embeddings/siglip2-lora_embeddings.npz \
    --results outputs/results/siglip2-lora/bootstrap/results.json \
    --output-dir outputs/results/siglip2-lora/bootstrap/feature_importance
```

### Step 6: Performance analysis and figures

```bash
# Per-condition analysis
python scripts/analyze_performance.py \
    --results outputs/results/siglip2-clahe/bootstrap/results.json \
    --embeddings outputs/segmented_embeddings/siglip2-so400m-14-384_clahe_embeddings.npz

python scripts/analyze_performance.py \
    --results outputs/results/siglip2-lora/bootstrap/results.json \
    --embeddings outputs/segmented_embeddings/siglip2-lora_embeddings.npz

# Paper figures
python scripts/generate_paper_figures.py
```

## Conventions

- **Evaluation modes**: Use `bootstrap` (not `splits`) when comparing frozen vs LoRA. The `splits` mode creates independent random splits each run and cannot guarantee the same test set across conditions. Bootstrap mode uses a single fixed split with bootstrap resampling for confidence intervals.
- **Split sharing**: Always pass `--split-file outputs/lora/siglip2/split.json` to `train_classifier.py` for the frozen baseline. This ensures both conditions are evaluated on the exact same test images. Splits are keyed by `measurement_id`, not array index, so they remain valid even if the embedding files have different row orders.
- **CLAHE**: Applied to all frozen encoder experiments as a fixed preprocessing step. The LoRA pipeline applies CLAHE internally during training, so LoRA embeddings already include it.
- **Result naming**: Directory names under `outputs/results/` follow the pattern `<model>[-clahe][-lora]/<eval-mode>/`. The name is auto-derived from the embeddings filename.
- **Age range**: Clipped to ages 1-10. The raw data contains ages up to 17 but older classes have too few samples.

## Non-Obvious Constraints

- **MPS memory**: Apple Silicon MPS can run out of memory during LoRA fine-tuning with large batch sizes. The default batch size in `finetune_dinov2_lora.py` is tuned for 32GB unified memory. Reduce `--batch-size` if you hit OOM.
- **Perception Encoder**: Requires a separate install (`pip install -e ".[pe]"`) and does not have macOS ARM64 wheels for `decord`. May need to be run on Linux/CUDA.
- **Embedding determinism**: Frozen encoder extraction is deterministic. LoRA fine-tuning is not perfectly deterministic on MPS due to non-deterministic operations. Results will vary slightly across runs.
- **SHAP runtime**: SHAP computation on the full embedding+metadata feature set (~1150+ features) is slow. Expect 10-30 minutes per condition.
- **No validation set**: The pipeline uses 85/15 train/test split with no separate validation set. Hyperparameter tuning (RidgeCV alpha) is done via cross-validation within the training set.
