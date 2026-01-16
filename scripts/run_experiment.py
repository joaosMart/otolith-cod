#!/usr/bin/env python3
"""
Main Experiment Script: Compare CLIP vs SigLIP2 for Otolith Age Prediction.

Runs 10-fold stratified cross-validation comparing:
- CLIP ViT-L/14@336px (baseline from paper)
- SigLIP2 SO400M-14-378 (SOTA comparison)

Based on: Sigurðardóttir et al. (2023) - Ecological Informatics

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --models clip-vit-l-14-336
    python scripts/run_experiment.py --models clip-vit-l-14-336,siglip2-so400m-14-384
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import OtolithDataset, create_kfold_splits
from src.models import (
    SUPPORTED_MODELS,
    load_model,
    extract_features,
    extract_and_cache_features,
    run_single_fold,
)
from src.evaluation import (
    compute_all_metrics,
    aggregate_fold_results,
    format_results_table,
    compare_models_significance,
)
from src.utils import load_config, get_output_paths, print_device_info


def parse_args():
    parser = argparse.ArgumentParser(description="Run CLIP vs SigLIP2 comparison experiment")
    parser.add_argument(
        "--models",
        type=str,
        default="clip-vit-l-14-336,siglip2-so400m-14-384",
        help="Comma-separated list of models to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force re-extraction of features (ignore cache)",
    )
    return parser.parse_args()


def extract_features_for_model(
    model_name: str,
    dataset: OtolithDataset,
    cache_dir: Path,
    batch_size: int = 32,
    force_recompute: bool = False,
) -> tuple:
    """Extract features for a model and cache them."""
    cache_file = cache_dir / f"{model_name}_embeddings.npz"

    if cache_file.exists() and not force_recompute:
        print(f"Loading cached embeddings from {cache_file}")
        data = np.load(cache_file, allow_pickle=False)
        return data["features"], data["labels"]

    print(f"\nExtracting features using {model_name}...")
    model, preprocess = load_model(model_name)

    # Create dataset with model's preprocessing
    preprocessed_dataset = OtolithDataset(
        root_dir=dataset.root_dir,
        transform=preprocess,
        age_range=dataset.age_range,
    )

    dataloader = DataLoader(
        preprocessed_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # MPS works best with 0 workers
    )

    features, labels = extract_features(model, dataloader, normalize=True)

    # Cache to disk
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Caching embeddings to {cache_file}")
    np.savez(cache_file, features=features, labels=labels)

    return features, labels


def run_cv_experiment(
    features: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 10,
    classifier_type: str = "ridge",
    classifier_params: dict = None,
    random_state: int = 42,
) -> list:
    """Run k-fold cross-validation and return metrics per fold."""
    splits = create_kfold_splits(labels, n_splits=n_splits, random_state=random_state)
    fold_metrics = []

    for split in tqdm(splits, desc=f"Running {n_splits}-fold CV"):
        X_train = features[split.train_indices]
        y_train = labels[split.train_indices]
        X_test = features[split.test_indices]
        y_test = labels[split.test_indices]

        result = run_single_fold(
            X_train, y_train, X_test, y_test,
            classifier_type=classifier_type,
            classifier_params=classifier_params,
        )

        metrics = compute_all_metrics(result["true_labels"], result["predictions"])
        fold_metrics.append(metrics)

    return fold_metrics


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    output_paths = get_output_paths(config)

    # Parse models
    model_names = [m.strip() for m in args.models.split(",")]
    for model in model_names:
        if model not in SUPPORTED_MODELS:
            print(f"Error: Unknown model '{model}'")
            print(f"Available: {', '.join(SUPPORTED_MODELS.keys())}")
            sys.exit(1)

    # Print device info
    print("\n" + "=" * 60)
    print("CLIP vs SigLIP2 Comparison Experiment")
    print("=" * 60)
    print_device_info()

    # Load dataset (without transforms - we'll apply them per model)
    data_config = config["data"]
    dataset = OtolithDataset(
        root_dir=data_config["root_dir"],
        transform=None,
        age_range=tuple(data_config["age_range"]),
    )
    print(f"\nDataset: {len(dataset)} images")
    print(f"Class distribution: {dataset.get_class_counts()}")

    # Extract features for each model
    all_features = {}
    labels = None

    for model_name in model_names:
        features, labels = extract_features_for_model(
            model_name=model_name,
            dataset=dataset,
            cache_dir=output_paths["embeddings"],
            batch_size=config.get("batch_size", 32),
            force_recompute=args.force_extract,
        )
        all_features[model_name] = features
        print(f"  {model_name}: {features.shape} embeddings")

    # Run CV for each model
    print("\n" + "-" * 60)
    print("Running 10-fold Cross-Validation")
    print("-" * 60)

    cv_config = config.get("cv", {})
    ridge_config = config.get("ridge", {})

    all_results = {}
    all_fold_scores = {}

    for model_name in model_names:
        print(f"\n{model_name}:")
        fold_metrics = run_cv_experiment(
            features=all_features[model_name],
            labels=labels,
            n_splits=cv_config.get("n_splits", 10),
            classifier_type="ridge",
            classifier_params={"alpha": ridge_config.get("alpha", 6.0)},
            random_state=cv_config.get("random_state", 42),
        )

        aggregated = aggregate_fold_results(fold_metrics)
        all_results[model_name] = aggregated
        all_fold_scores[model_name] = [fm["accuracy"] for fm in fold_metrics]

        acc_mean, acc_std = aggregated["accuracy"]
        pm1_mean, pm1_std = aggregated["accuracy_pm1"]
        rmse_mean, rmse_std = aggregated["rmse"]

        print(f"  Accuracy: {acc_mean*100:.2f} ± {acc_std*100:.2f}%")
        print(f"  ±1 Accuracy: {pm1_mean*100:.2f} ± {pm1_std*100:.2f}%")
        print(f"  RMSE: {rmse_mean:.3f} ± {rmse_std:.3f}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    paper_ref = config.get("paper_reference", {}).get("cod", None)
    table = format_results_table(all_results, paper_ref)
    print(table)

    # Statistical significance test (if we have 2+ models)
    if len(model_names) >= 2:
        print("\n" + "-" * 60)
        print("Statistical Significance (Paired t-test)")
        print("-" * 60)

        baseline = model_names[0]
        for comparison in model_names[1:]:
            result = compare_models_significance(
                all_fold_scores[baseline],
                all_fold_scores[comparison],
                metric_name="accuracy",
            )
            better_or_worse = "better" if result["model2_better"] else "worse"
            significance = "significant" if result["p_value"] < 0.05 else "not significant"

            print(f"\n{comparison} vs {baseline}:")
            print(f"  Mean difference: {result['mean_difference']*100:+.2f}% ({better_or_worse})")
            print(f"  t-statistic: {result['t_statistic']:.3f}")
            print(f"  p-value: {result['p_value']:.4f} ({significance})")

    # Save results
    results_dir = output_paths["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "comparison_results.txt"
    with open(results_file, "w") as f:
        f.write("CLIP vs SigLIP2 Comparison Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(table + "\n")

    print(f"\nResults saved to {results_file}")
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
