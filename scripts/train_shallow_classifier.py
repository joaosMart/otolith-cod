#!/usr/bin/env python3
"""
Train Ridge Classifier on SigLIP2 Embeddings.

Runs 10 independent experiments with stratified 65/15/20% splits,
GridSearchCV for alpha optimization, and comprehensive metrics.

Based on: Sigurðardóttir et al. (2023) - Ecological Informatics

Usage:
    python scripts/train_shallow_classifier.py
    python scripts/train_shallow_classifier.py --embeddings outputs/embeddings/siglip2-so400m-14-384_embeddings.npz
    python scripts/train_shallow_classifier.py --n-experiments 2 --alpha-steps 5  # Quick test
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

from src.data import DataSplit, create_train_val_test_splits
from src.evaluation import compute_classification_metrics, aggregate_fold_results
from src.utils import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Ridge Classifier on embeddings with GridSearchCV"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default= "outputs/embeddings/siglip2-so400m-14-384_embeddings.npz",
        help="Path to embeddings .npz file",
    )
    parser.add_argument(
        "--config",
        type=str, 
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--n-experiments",
        type=int,
        default=10,
        help="Number of independent experiments (default: 10)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.65)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.20)",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.1,
        help="Minimum alpha for grid search (default: 0.1)",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=19.6,
        help="Maximum alpha for grid search (default: 19.6)",
    )
    parser.add_argument(
        "--alpha-steps",
        type=int,
        default=100,
        help="Number of alpha values in grid search (default: 20)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=10,
        help="Number of CV folds for GridSearchCV (default: 5)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Base random state for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/results/shallow_siglip2",
        help="Directory for output files",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for GridSearchCV (-1 for all cores)",
    )
    return parser.parse_args()


def load_embeddings(embeddings_path: str) -> tuple:
    """
    Load embeddings from .npz file.

    Args:
        embeddings_path: Path to embeddings file

    Returns:
        Tuple of (features, labels)

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If expected keys are missing
    """
    path = Path(embeddings_path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    print(f"Loading embeddings from {path}")
    data = np.load(path)

    # Support both 'features'/'labels' and 'embeddings'/'labels' key names
    if "features" in data.files:
        features = data["features"]
    elif "embeddings" in data.files:
        features = data["embeddings"]
    else:
        raise KeyError(f"Expected 'features' or 'embeddings' key, got: {data.files}")

    labels = data["labels"]
    data.close()

    ##########
    ## TESTING
    ##########

    target_ages = [1,2,3,4,5,6,7,8,9,10]
    mask = np.isin(labels, target_ages)
    filter_features = features[mask]
    filter_labels = labels[mask]

    print(f"  Features shape: {filter_features.shape}")
    print(f"  Labels shape: {filter_labels.shape}")
    print(f"  Unique classes: {len(np.unique(filter_labels))}")

    return filter_features, filter_labels


def run_single_experiment(
    features: np.ndarray,
    labels: np.ndarray,
    split: DataSplit,
    alpha_range: np.ndarray,
    cv_folds: int = 5,
    n_jobs: int = -1,
) -> dict:
    """
    Run single experiment with GridSearchCV.

    Args:
        features: Feature matrix (N, D)
        labels: Label vector (N,)
        split: DataSplit object with train/val/test indices
        alpha_range: Array of alpha values to search
        cv_folds: Number of CV folds for grid search
        n_jobs: Number of parallel jobs

    Returns:
        Dictionary with experiment results
    """
    X_train = features[split.train_indices]
    y_train = labels[split.train_indices]
    X_val = features[split.val_indices]
    y_val = labels[split.val_indices]
    X_test = features[split.test_indices]
    y_test = labels[split.test_indices]

    # Compute sample weights for class balancing
    sample_weights = compute_sample_weight("balanced", y_train)

    # GridSearchCV on training data
    clf = Ridge(random_state=split.fold)
    param_grid = {"alpha": alpha_range}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=split.fold)

    # Create custom scorer that rounds predictions for classification metrics
    from sklearn.metrics import make_scorer, f1_score
    
    def f1_rounded(y_true, y_pred):
        return f1_score(y_true, np.round(y_pred).astype(int), average='macro')

    scorer = make_scorer(f1_rounded)

    grid_search = GridSearchCV(
        clf,
        param_grid,
        cv=cv,
        scoring= "neg_mean_squared_error",
        n_jobs=n_jobs,
        refit=True,
    )


    # Fit with sample weights
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    best_alpha = grid_search.best_params_["alpha"]
    best_cv_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    # Evaluate on validation set (monitoring only)
    y_val_pred = np.round(best_model.predict(X_val)).astype(int)
    val_metrics = compute_classification_metrics(y_val, y_val_pred)

    # Evaluate on test set (final evaluation)
    y_test_pred = np.round(best_model.predict(X_test)).astype(int)
    test_metrics = compute_classification_metrics(y_test, y_test_pred)

    from sklearn.metrics import mean_squared_error
    test_metrics['rmse_raw'] = np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))

    return {
        "experiment": split.fold,
        "best_alpha": float(best_alpha),
        "best_cv_score": float(best_cv_score),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def format_results_summary(aggregated: dict, paper_reference: dict = None) -> str:
    """
    Format aggregated results as a readable summary.

    Args:
        aggregated: Dictionary with (mean, std) tuples for each metric
        paper_reference: Optional paper reference values

    Returns:
        Formatted string summary
    """
    lines = [
        "=" * 60,
        "RESULTS SUMMARY",
        "=" * 60,
        "",
        "Test Set Metrics (mean ± std across all experiments):",
        "-" * 50,
    ]

    # Display metrics
    metric_names = {
        "accuracy": "Accuracy",
        "accuracy_pm1": "±1 Accuracy",
        "precision": "Precision (macro)",
        "recall": "Recall (macro)",
        "f1": "F1-Score (macro)",
        "rmse_raw": "RMSE",
    }

    for key, display_name in metric_names.items():
        if key in aggregated:
            mean, std = aggregated[key]
            if key in ["accuracy", "accuracy_pm1", "precision", "recall", "f1"]:
                lines.append(f"  {display_name:20s}: {mean*100:6.2f} ± {std*100:.2f}%")
            else:
                lines.append(f"  {display_name:20s}: {mean:6.3f} ± {std:.3f}")

    # Add paper reference if provided
    if paper_reference:
        lines.extend([
            "",
            "Paper Reference (Cod dataset, N=1170):",
            "-" * 50,
            f"  Accuracy:            {paper_reference.get('accuracy', 'N/A'):.2f} ± {paper_reference.get('accuracy_std', 0):.2f}%",
            f"  ±1 Accuracy:         {paper_reference.get('accuracy_pm1', 'N/A'):.2f} ± {paper_reference.get('accuracy_pm1_std', 0):.2f}%",
            f"  RMSE:                {paper_reference.get('rmse', 'N/A'):.2f} ± {paper_reference.get('rmse_std', 0):.2f}",
        ])

    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    """Main entry point."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("Ridge Classifier Training on SigLIP2 Embeddings")
    print("=" * 60)

    # Load config if available
    config = {}
    paper_reference = None
    try:
        config = load_config(args.config)
        paper_reference = config.get("paper_reference", {}).get("cod", None)
        print(paper_reference)
    except Exception as e:
        print(f"Warning: Could not load config: {e}")

    # Load embeddings
    try:
        features, labels = load_embeddings(args.embeddings)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Print class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} samples ({cnt/len(labels)*100:.1f}%)")

    # Create alpha range for grid search
    alpha_range = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    print(f"\nAlpha grid: {args.alpha_steps} values from {args.alpha_min} to {args.alpha_max}")

    # Create data splits
    print(f"\nCreating {args.n_experiments} independent splits...")
    print(f"  Train/Val/Test ratios: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")

    splits = create_train_val_test_splits(
        labels=labels,
        n_experiments=args.n_experiments,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )

    # Run experiments
    print(f"\nRunning {args.n_experiments} experiments with {args.cv_folds}-fold GridSearchCV...")
    print("-" * 60)

    all_results = []
    test_metrics_list = []

    for split in tqdm(splits, desc="Experiments"):
        result = run_single_experiment(
            features=features,
            labels=labels,
            split=split,
            alpha_range=alpha_range,
            cv_folds=args.cv_folds,
            n_jobs=args.n_jobs,
        )
        all_results.append(result)
        test_metrics_list.append(result["test_metrics"])

        # Print per-experiment summary
        print(
            f"  Exp {result['experiment']:2d}: "
            f"alpha={result['best_alpha']:.2f}, "
            f"CV_MSE={result['best_cv_score']:.4f}, "
            f"Test_Acc={result['test_metrics']['accuracy']*100:.2f}%, "
            f"Test_F1={result['test_metrics']['f1']*100:.2f}%"
        )

    # Aggregate results
    aggregated = aggregate_fold_results(test_metrics_list)

    # Print summary
    print("\n" + format_results_summary(aggregated, paper_reference))

    # Prepare output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full results to JSON
    full_results = {
        "timestamp": timestamp,
        "config": {
            "embeddings_path": args.embeddings,
            "n_experiments": args.n_experiments,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "alpha_steps": args.alpha_steps,
            "cv_folds": args.cv_folds,
            "random_state": args.random_state,
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
    print(f"\nFull results saved to: {results_file}")

    # Save summary CSV
    summary_rows = []
    for result in all_results:
        row = {
            "experiment": result["experiment"],
            "best_alpha": result["best_alpha"],
            "best_cv_score": result["best_cv_score"],
            "train_size": result["train_size"],
            "val_size": result["val_size"],
            "test_size": result["test_size"],
        }
        # Add test metrics
        for key, value in result["test_metrics"].items():
            row[f"test_{key}"] = value
        summary_rows.append(row)

    # Add aggregated row
    agg_row = {
        "experiment": "MEAN±STD",
        "best_alpha": np.mean([r["best_alpha"] for r in all_results]),
    }
    for key, (mean, std) in aggregated.items():
        agg_row[f"test_{key}"] = f"{mean:.4f}±{std:.4f}"
    summary_rows.append(agg_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / "summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary CSV saved to: {summary_file}")

    # Save split indices for reproducibility
    splits_data = {
        "n_experiments": args.n_experiments,
        "random_state": args.random_state,
        "splits": [
            {
                "experiment": split.fold,
                "train_indices": split.train_indices.tolist(),
                "val_indices": split.val_indices.tolist(),
                "test_indices": split.test_indices.tolist(),
            }
            for split in splits
        ],
    }
    splits_file = output_dir / "splits.json"
    with open(splits_file, "w") as f:
        json.dump(splits_data, f, indent=2)
    print(f"Split indices saved to: {splits_file}")

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
