#!/usr/bin/env python3
"""
Train Ridge Classifier on cached embeddings.

Runs independent experiments with stratified splits, RidgeCV
for alpha optimization, and comprehensive metrics.

Usage:
    python scripts/train_classifier.py --embeddings outputs/embeddings/siglip2-so400m-14-384_embeddings.npz
    python scripts/train_classifier.py --embeddings embeddings.npz --seeds 82,15,4 --alpha-steps 5
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features import load_cached_embeddings, augment_embeddings
from src.data import create_train_test_splits
from src.training import run_experiment_with_search, save_splits
from src.evaluation import compute_classification_metrics, aggregate_fold_results
from src.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Ridge classifier on cached embeddings with GridSearchCV"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="outputs/embeddings/siglip2-so400m-14-384_embeddings.npz",
        help="Path to embeddings .npz file",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--n-experiments", type=int, default=10, help="Number of independent experiments (ignored when --seeds is set)")
    parser.add_argument(
        "--seeds",
        type=str,
        default="82,15,4,95,36,32,29,18,14,87",
        help="Comma-separated seeds for each experiment (default: 82,15,4,95,36,32,29,18,14,87)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.85, help="Training set ratio (default: 0.85)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio (default: 0.15)")
    parser.add_argument("--alpha-log-min", type=float, default=1.5, help="Log10 minimum alpha for RidgeCV (default: 1.5, i.e. ~31.6)")
    parser.add_argument("--alpha-log-max", type=float, default=3, help="Log10 maximum alpha for RidgeCV (default: 3, i.e. 1000)")
    parser.add_argument("--alpha-steps", type=int, default=30, help="Number of alpha values in RidgeCV (default: 30)")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds for RidgeCV (default: 5)")
    parser.add_argument("--random-state", type=int, default=42, help="Base random state (default: 42, used when --seeds not set)")
    parser.add_argument("--output-dir", type=str, default="outputs/results/shallow_siglip2", help="Directory for output files")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV (-1 for all cores)")
    parser.add_argument("--metadata-csv", type=str, default="cod_otolith_age_final_with_scale.csv", help="Path to metadata CSV file")
    parser.add_argument(
        "--feature-key",
        type=str,
        default=None,
        help="Which feature key to use from embeddings (e.g. features_cls, features_patch_mean_pool). Auto-detected if not set.",
    )
    parser.add_argument("--add-tabular", action=argparse.BooleanOptionalAction, default=True, help="Add tabular features to embeddings")
    parser.add_argument(
        "--tabular-columns",
        type=str,
        default="length,month_sin,month_cos,shot_latitude,shot_longitude,is_survey,seg_width_px,seg_height_px,seg_aspect_ratio",
        help="Comma-separated list of tabular columns to add",
    )
    return parser.parse_args()


def format_results_summary(aggregated: dict, paper_reference: dict = None) -> str:
    lines = [
        "=" * 60,
        "RESULTS SUMMARY",
        "=" * 60,
        "",
        "Test Set Metrics (mean +/- std across all experiments):",
        "-" * 50,
    ]
    metric_names = {
        "f1": "F1-Score (macro)",
        "accuracy": "Accuracy",
        "accuracy_pm1": "+/-1 Accuracy",
        "precision": "Precision (macro)",
        "recall": "Recall (macro)",
        "rmse_raw": "RMSE",
    }
    for key, display_name in metric_names.items():
        if key in aggregated:
            mean, std = aggregated[key]
            if key in ["accuracy", "accuracy_pm1", "precision", "recall", "f1"]:
                lines.append(f"  {display_name:20s}: {mean*100:6.2f} +/- {std*100:.2f}%")
            else:
                lines.append(f"  {display_name:20s}: {mean:6.3f} +/- {std:.3f}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("RIDGE CLASSIFIER TRAINING")
    print("=" * 60)

    # Load config
    config = {}
    paper_reference = None
    try:
        config = load_config(args.config)
        paper_reference = config.get("paper_reference", {}).get("cod", None)
    except Exception as e:
        print(f"Warning: Could not load config: {e}")

    # Load embeddings
    features_dict, labels, measurement_ids = load_cached_embeddings(args.embeddings)
    if args.feature_key:
        if args.feature_key not in features_dict:
            raise KeyError(f"Key '{args.feature_key}' not found. Available: {list(features_dict.keys())}")
        used_feature_key = args.feature_key
    elif "features" in features_dict:
        used_feature_key = "features"
    elif "features_cls" in features_dict:
        used_feature_key = "features_cls"
    else:
        raise KeyError(f"No recognized feature key in embeddings. Found: {list(features_dict.keys())}")
    features = features_dict[used_feature_key]
    print(f"  Feature key: {used_feature_key}")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")

    # Augment with tabular features
    if args.add_tabular:
        columns = args.tabular_columns.split(",")
        print(f"\nAugmenting embeddings with tabular features: {columns}")
        if measurement_ids is None:
            print("Error: Embeddings missing measurement_ids. Re-extract with updated script.")
            sys.exit(1)
        features = augment_embeddings(features, measurement_ids, args.metadata_csv, columns)
        print(f"  Augmented features shape: {features.shape}")

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} samples ({cnt/len(labels)*100:.1f}%)")

    # Alpha range
    alpha_range = np.logspace(args.alpha_log_min, args.alpha_log_max, args.alpha_steps)
    print(f"\nAlpha grid: {args.alpha_steps} values from {alpha_range[0]:.1f} to {alpha_range[-1]:.1f} (logspace)")

    # Parse seeds
    seeds = [int(s) for s in args.seeds.split(",")]
    n_experiments = len(seeds)

    # Create splits
    print(f"\nCreating {n_experiments} independent splits (seeds: {seeds})...")
    print(f"  Train/Test ratios: {args.train_ratio}/{args.test_ratio}")

    splits = create_train_test_splits(
        labels=labels,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        seeds=seeds,
    )

    # Run experiments
    print(f"\nRunning {n_experiments} experiments with {args.cv_folds}-fold RidgeCV...")
    print("-" * 60)

    all_results = []
    test_metrics_list = []

    for split in tqdm(splits, desc="Experiments"):
        result = run_experiment_with_search(
            features=features,
            labels=labels,
            train_indices=split.train_indices,
            test_indices=split.test_indices,
            alpha_range=alpha_range,
            cv_folds=args.cv_folds,
            n_jobs=args.n_jobs,
            random_state=split.fold,
        )
        result["experiment"] = split.fold
        all_results.append(result)
        test_metrics_list.append(result["test_metrics"])

        print(
            f"  Exp {result['experiment']:2d}: "
            f"alpha={result['best_alpha']:.2f}, "
            f"CV_MSE={result['best_cv_score']:.4f}, "
            f"Test_Acc={result['test_metrics']['accuracy']*100:.2f}%, "
            f"Test_F1={result['test_metrics']['f1']*100:.2f}%"
        )

    # Aggregate
    aggregated = aggregate_fold_results(test_metrics_list)
    print("\n" + format_results_summary(aggregated, paper_reference))

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Results JSON
    full_results = {
        "timestamp": timestamp,
        "config": {
            "embeddings_path": args.embeddings,
            "feature_key": used_feature_key,
            "add_tabular": args.add_tabular,
            "tabular_columns": args.tabular_columns.split(",") if args.add_tabular else [],
            "seeds": seeds,
            "n_experiments": n_experiments,
            "train_ratio": args.train_ratio,
            "test_ratio": args.test_ratio,
            "alpha_log_min": args.alpha_log_min,
            "alpha_log_max": args.alpha_log_max,
            "alpha_steps": args.alpha_steps,
            "cv_folds": args.cv_folds,
        },
        "experiment_results": all_results,
        "aggregated_results": {
            key: {"mean": float(mean), "std": float(std)}
            for key, (mean, std) in aggregated.items()
        },
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Summary CSV
    summary_rows = []
    for result in all_results:
        row = {
            "experiment": result["experiment"],
            "best_alpha": result["best_alpha"],
            "best_cv_score": result["best_cv_score"],
            "train_size": result["train_size"],
            "test_size": result["test_size"],
        }
        for key, value in result["test_metrics"].items():
            row[f"test_{key}"] = value
        summary_rows.append(row)

    agg_row = {
        "experiment": "MEAN+/-STD",
        "best_alpha": np.mean([r["best_alpha"] for r in all_results]),
    }
    for key, (mean, std) in aggregated.items():
        agg_row[f"test_{key}"] = f"{mean:.4f}+/-{std:.4f}"
    summary_rows.append(agg_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / "summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    # Save splits
    save_splits(splits, str(output_dir / "splits.json"), args.random_state)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
