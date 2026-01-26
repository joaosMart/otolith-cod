#!/usr/bin/env python3
"""
Analyze Ridge Classifier Performance.

Comprehensive performance analysis including:
- Confusion matrices and classification metrics visualization
- Error analysis with misclassification reports
- Optional learning curves for data efficiency assessment
- Optional data quality analysis using cleanlab

Usage:
    python scripts/analyze_performance.py --results outputs/results/shallow_siglip2/results.json
    python scripts/analyze_performance.py --results results.json --learning-curves
    python scripts/analyze_performance.py --results results.json --cleanlab
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.data import create_train_val_test_splits
from src.evaluation import (
    compute_classification_metrics,
)
from src.evaluation.error_analysis import (
    generate_error_report,
    plot_error_distribution,
    plot_error_histogram,
)
from src.evaluation.learning_curves import (
    run_learning_curve_experiment,
    plot_multiple_learning_curves,
)
from src.evaluation.data_quality import (
    check_cleanlab_available,
    generate_data_quality_report,
)
from src.visualization.classification_plots import (
    plot_aggregated_confusion_matrix,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze classifier performance with comprehensive visualizations"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results.json from training",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        help="Path to embeddings .npz file (required for learning curves and cleanlab)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/analysis",
        help="Output directory for figures and reports",
    )
    parser.add_argument(
        "--learning-curves",
        action="store_true",
        help="Compute learning curves (requires --embeddings)",
    )
    parser.add_argument(
        "--cleanlab",
        action="store_true",
        help="Run data quality analysis with cleanlab (requires --embeddings)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures (default: 300)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    return parser.parse_args()


def load_results(results_path: str) -> dict:
    """Load results JSON file."""
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path, "r") as f:
        results = json.load(f)

    print(f"Loaded results from {path}")
    print(f"  Experiments: {len(results['experiment_results'])}")
    return results


def load_embeddings(embeddings_path: str) -> tuple:
    """Load embeddings from .npz file."""
    path = Path(embeddings_path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    data = np.load(path, allow_pickle=False)

    if "features" in data.files:
        features = data["features"]
    elif "embeddings" in data.files:
        features = data["embeddings"]
    else:
        raise KeyError(f"Expected 'features' or 'embeddings' key, got: {data.files}")

    labels = data["labels"]
    measurement_ids = data.get("measurement_ids", None)

    data.close()

    print(f"Loaded embeddings from {path}")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")

    return features, labels, measurement_ids


def extract_predictions_from_results(results: dict, splits_file: Path) -> tuple:
    """
    Extract predictions from results JSON.

    Args:
        results: Results dictionary from training
        splits_file: Path to splits.json file

    Returns:
        Tuple of (all_y_true, all_y_pred, all_y_scores)
    """
    # This is a simplified version - in practice, you'd need to reload
    # the model and make predictions, or save predictions during training
    print("\nNote: This function requires predictions to be saved during training.")
    print("For now, we'll reconstruct from results if possible.")

    return None, None, None


def main():
    """Main entry point."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("RIDGE CLASSIFIER PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(args.results)
    config = results["config"]

    # Extract test metrics from all experiments
    print("\n" + "-" * 60)
    print("AGGREGATED RESULTS")
    print("-" * 60)

    agg_results = results["aggregated_results"]
    print(f"Test Accuracy:    {agg_results['accuracy']['mean']*100:.2f} ± {agg_results['accuracy']['std']*100:.2f}%")
    print(f"Test ±1 Accuracy: {agg_results['accuracy_pm1']['mean']*100:.2f} ± {agg_results['accuracy_pm1']['std']*100:.2f}%")
    print(f"Test F1 Score:    {agg_results['f1']['mean']*100:.2f} ± {agg_results['f1']['std']*100:.2f}%")
    print(f"Test RMSE:        {agg_results['rmse']['mean']:.3f} ± {agg_results['rmse']['std']:.3f}")

    # Check if we have embeddings for advanced analysis
    if args.embeddings:
        features, labels, measurement_ids = load_embeddings(args.embeddings)

        # Recreate splits using same configuration
        print("\n" + "-" * 60)
        print("RECREATING DATA SPLITS")
        print("-" * 60)

        splits = create_train_val_test_splits(
            labels=labels,
            n_experiments=config["n_experiments"],
            train_ratio=config["train_ratio"],
            val_ratio=config["val_ratio"],
            test_ratio=config["test_ratio"],
            random_state=config["random_state"],
        )

        # Collect predictions from all experiments
        # Note: We'll need to retrain or load saved models to get predictions
        # For now, we'll work with what we have

        print("\nNote: For full analysis with confusion matrices and ROC curves,")
        print("predictions need to be saved during training or models need to be reloaded.")

    else:
        print("\n" + "-" * 60)
        print("LIMITED ANALYSIS MODE")
        print("-" * 60)
        print("No embeddings file provided. Skipping confusion matrices, ROC curves,")
        print("error analysis, learning curves, and cleanlab analysis.")
        print("\nTo enable full analysis, provide --embeddings path.")

    # Learning curves
    if args.learning_curves:
        if not args.embeddings:
            print("\nError: --learning-curves requires --embeddings")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("LEARNING CURVE ANALYSIS")
        print("=" * 60)

        # Use best alpha from results
        best_alpha = np.mean([r["best_alpha"] for r in results["experiment_results"]])
        print(f"Using mean alpha: {best_alpha:.3f}")

        lc_results = run_learning_curve_experiment(
            features=features,
            labels=labels,
            splits=splits,
            alpha=best_alpha,
        )

        # Plot learning curves
        fig = plot_multiple_learning_curves(
            results=lc_results,
            metrics=["accuracy", "accuracy_pm1", "f1"],
            save_path=figures_dir / "learning_curves.png",
            dpi=args.dpi,
        )

        print(f"\nSaved learning curves to {figures_dir / 'learning_curves.png'}")

    # Data quality analysis with cleanlab
    if args.cleanlab:
        if not args.embeddings:
            print("\nError: --cleanlab requires --embeddings")
            sys.exit(1)

        if not check_cleanlab_available():
            print("\nError: cleanlab is not installed")
            print("Install with: pip install cleanlab>=2.6.0")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("DATA QUALITY ANALYSIS (CLEANLAB)")
        print("=" * 60)

        # Use best alpha from results
        best_alpha = np.mean([r["best_alpha"] for r in results["experiment_results"]])

        quality_report = generate_data_quality_report(
            features=features,
            labels=labels,
            measurement_ids=measurement_ids,
            alpha=best_alpha,
            n_folds=5,
            output_dir=output_dir / "data_quality",
            random_state=args.random_state,
        )

        print(f"\nData quality reports saved to {output_dir / 'data_quality'}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
