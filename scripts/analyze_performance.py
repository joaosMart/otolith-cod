#!/usr/bin/env python3
"""
Analyze Ridge Classifier Performance.

Comprehensive performance analysis including confusion matrices, error analysis,
optional learning curves, and optional data quality analysis using cleanlab.

Usage:
    python scripts/analyze_performance.py --results outputs/results/shallow_siglip2/results.json
    python scripts/analyze_performance.py --results results.json --embeddings embeddings.npz --learning-curves
    python scripts/analyze_performance.py --results results.json --embeddings embeddings.npz --cleanlab
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.features import load_cached_embeddings, augment_embeddings
from src.training import load_splits
from src.data import create_train_test_splits
from src.evaluation import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
)
from src.evaluation.error_analysis import (
    generate_error_report,
    plot_error_distribution,
    plot_error_histogram,
    plot_per_class_f1,
    plot_per_class_error_magnitude,
    plot_most_confused_pairs,
    plot_f1_vs_frequency,
)
from src.evaluation.learning_curves import (
    run_learning_curve_experiment,
    plot_multiple_learning_curves,
)
from src.evaluation.data_quality import (
    check_cleanlab_available,
    generate_data_quality_report,
    plot_label_issues_per_class,
    plot_suggested_label_transitions,
    plot_confidence_distribution,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze classifier performance with comprehensive visualizations"
    )
    parser.add_argument("--results", type=str, required=True, help="Path to results.json from training")
    parser.add_argument("--embeddings", type=str, help="Path to embeddings .npz file (required for visualizations)")
    parser.add_argument("--splits", type=str, help="Path to splits.json (default: same dir as results)")
    parser.add_argument("--output-dir", type=str, default="outputs/analysis", help="Output directory for figures and reports")
    parser.add_argument("--learning-curves", action="store_true", help="Compute learning curves (requires --embeddings)")
    parser.add_argument("--cleanlab", action="store_true", help="Run data quality analysis with cleanlab (requires --embeddings)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures (default: 300)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--metadata-csv", type=str, default="cod_otolith_age_final_with_scale.csv", help="Path to metadata CSV")
    parser.add_argument(
        "--tabular-columns",
        type=str,
        default="length,month_sin,month_cos,shot_latitude,shot_longitude,is_survey,seg_width_px,seg_height_px,seg_aspect_ratio",
        help="Comma-separated list of tabular columns to add",
    )
    return parser.parse_args()


def load_results(results_path: str) -> dict:
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "r") as f:
        results = json.load(f)
    print(f"Loaded results from {path}")
    print(f"  Experiments: {len(results['experiment_results'])}")
    return results


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("RIDGE CLASSIFIER PERFORMANCE ANALYSIS")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(args.results)
    eval_mode = results.get("eval_mode", "splits")
    print(f"  Eval mode: {eval_mode}")
    config = results["config"]

    figures_dir = output_dir / "figures" / eval_mode
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Print aggregated results
    print("\n" + "-" * 60)
    print(f"AGGREGATED RESULTS (eval_mode={eval_mode})")
    print("-" * 60)
    agg = results["aggregated_results"]

    if eval_mode == "bootstrap":
        print(f"Test Macro F1:      {agg['f1']['mean']*100:.2f}% [{agg['f1']['ci_lower']*100:.2f}, {agg['f1']['ci_upper']*100:.2f}]")
        print(f"Test Accuracy:      {agg['accuracy']['mean']*100:.2f}% [{agg['accuracy']['ci_lower']*100:.2f}, {agg['accuracy']['ci_upper']*100:.2f}]")
        print(f"Test +/-1 Accuracy: {agg['accuracy_pm1']['mean']*100:.2f}% [{agg['accuracy_pm1']['ci_lower']*100:.2f}, {agg['accuracy_pm1']['ci_upper']*100:.2f}]")
        print(f"Test RMSE:          {agg['rmse']['mean']:.3f} [{agg['rmse']['ci_lower']:.3f}, {agg['rmse']['ci_upper']:.3f}]")
    else:
        print(f"Test Macro F1:      {agg['f1']['mean']*100:.2f} +/- {agg['f1']['std']*100:.2f}%")
        print(f"Test Accuracy:      {agg['accuracy']['mean']*100:.2f} +/- {agg['accuracy']['std']*100:.2f}%")
        print(f"Test +/-1 Accuracy: {agg['accuracy_pm1']['mean']*100:.2f} +/- {agg['accuracy_pm1']['std']*100:.2f}%")
        print(f"Test RMSE:          {agg['rmse']['mean']:.3f} +/- {agg['rmse']['std']:.3f}")

    if not args.embeddings:
        print("\nNo embeddings provided. Skipping visualizations.")
        print("Provide --embeddings for full analysis.")
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        return

    # Load embeddings using feature config from training
    feature_key = config.get("feature_key", None)
    add_tabular = config.get("add_tabular", True)
    tabular_columns = config.get("tabular_columns", args.tabular_columns.split(","))

    features_dict, labels, measurement_ids = load_cached_embeddings(args.embeddings)

    # Select the same feature key used during training
    if feature_key and feature_key in features_dict:
        features = features_dict[feature_key]
        print(f"\n  Using feature key from training: {feature_key}")
    elif "features" in features_dict:
        features = features_dict["features"]
    elif "features_cls" in features_dict:
        features = features_dict["features_cls"]
    else:
        raise KeyError(f"No recognized feature key. Found: {list(features_dict.keys())}")
    print(f"  Features shape: {features.shape}")

    # Augment if training used tabular features
    if add_tabular and args.metadata_csv and Path(args.metadata_csv).exists():
        print(f"\nAugmenting embeddings with tabular features from {args.metadata_csv}...")
        columns = tabular_columns
        try:
            features = augment_embeddings(features, measurement_ids, args.metadata_csv, columns)
            print(f"  Augmented features shape: {features.shape}")
        except Exception as e:
            print(f"  Warning: Could not augment embeddings: {e}")
            print("  Proceeding with original embeddings.")

    # Load or recreate splits based on eval mode
    if eval_mode == "bootstrap":
        # Single fixed split — load from split.json
        from src.data import load_split_by_ids

        split_path = args.splits
        if split_path is None:
            candidate = Path(args.results).parent / "split.json"
            if candidate.exists():
                split_path = str(candidate)

        if split_path is None or not Path(split_path).exists():
            print("Error: No split.json found for bootstrap mode. Expected next to results.json.")
            sys.exit(1)

        print(f"\nLoading fixed split from {split_path}")
        split = load_split_by_ids(split_path, measurement_ids)
        experiment_alphas = [results["experiment_results"][0]["best_alpha"]]
        splits = [split]

    else:
        # Existing splits mode
        splits_path = args.splits
        if splits_path is None:
            candidate = Path(args.results).parent / "splits.json"
            if candidate.exists():
                splits_path = str(candidate)

        if splits_path and Path(splits_path).exists():
            print(f"\nLoading splits from {splits_path}")
            splits = load_splits(splits_path)
        else:
            print("\nRecreating splits from config parameters...")
            splits = create_train_test_splits(
                labels=labels,
                n_experiments=config["n_experiments"],
                train_ratio=config["train_ratio"],
                test_ratio=config["test_ratio"],
                random_state=config.get("random_state", 42),
            )
        experiment_alphas = [r["best_alpha"] for r in results["experiment_results"]]

    # Generate predictions
    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS FOR VISUALIZATION")
    print("=" * 60)

    print(f"Using per-experiment alphas: {[f'{a:.1f}' for a in experiment_alphas]}")

    all_y_true = []
    all_y_pred = []
    all_y_scores = []
    all_measurement_ids = []

    for split, alpha in tqdm(zip(splits, experiment_alphas), total=len(splits), desc="Generating predictions"):
        X_train = features[split.train_indices]
        y_train = labels[split.train_indices]
        X_test = features[split.test_indices]
        y_test = labels[split.test_indices]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        y_scores = model.predict(X_test)
        y_pred = np.clip(np.round(y_scores).astype(int), 1, 10)

        all_y_true.append(y_test)
        all_y_pred.append(y_pred)
        all_y_scores.append(y_scores)
        if measurement_ids is not None:
            all_measurement_ids.append(measurement_ids[split.test_indices])

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)
    y_scores_all = np.concatenate(all_y_scores)
    mids_all = np.concatenate(all_measurement_ids) if all_measurement_ids else None

    # Confusion matrices
    print("\nGenerating confusion matrices...")
    plot_confusion_matrix(
        y_true=y_true_all, y_pred=y_pred_all, normalize="true",
        title="Confusion Matrix (Normalized by True Label)",
        save_path=figures_dir / "confusion_matrix_normalized.png", dpi=args.dpi,
    )
    plot_confusion_matrix(
        y_true=y_true_all, y_pred=y_pred_all, normalize=None,
        title="Confusion Matrix (Raw Counts)",
        save_path=figures_dir / "confusion_matrix_counts.png", dpi=args.dpi,
    )

    # ROC curves
    print("Generating ROC curves...")
    plot_roc_curves(
        y_true=y_true_all, y_scores=y_scores_all,
        save_path=figures_dir / "roc_curves.png", dpi=args.dpi,
    )

    # Precision-Recall curves
    print("Generating Precision-Recall curves...")
    plot_precision_recall_curves(
        y_true=y_true_all, y_scores=y_scores_all,
        save_path=figures_dir / "precision_recall_curves.png", dpi=args.dpi,
    )

    # Error analysis
    print("Generating error analysis...")
    plot_error_distribution(
        y_true=y_true_all, y_pred=y_pred_all,
        save_path=figures_dir / "error_distribution.png", dpi=args.dpi,
    )
    plot_error_histogram(
        y_true=y_true_all, y_pred=y_pred_all,
        save_path=figures_dir / "error_histogram.png", dpi=args.dpi,
    )
    plot_per_class_f1(
        y_true=y_true_all, y_pred=y_pred_all,
        save_path=figures_dir / "per_class_f1.png", dpi=args.dpi,
    )
    plot_per_class_error_magnitude(
        y_true=y_true_all, y_pred=y_pred_all,
        save_path=figures_dir / "per_class_error_magnitude.png", dpi=args.dpi,
    )
    plot_most_confused_pairs(
        y_true=y_true_all, y_pred=y_pred_all,
        save_path=figures_dir / "most_confused_pairs.png", dpi=args.dpi,
    )
    plot_f1_vs_frequency(
        y_true=y_true_all, y_pred=y_pred_all,
        save_path=figures_dir / "f1_vs_frequency.png", dpi=args.dpi,
    )
    error_report = generate_error_report(
        y_true=y_true_all, y_pred=y_pred_all,
        measurement_ids=mids_all, output_dir=output_dir / "error_analysis",
    )
    stats = error_report["statistics"]
    print(f"\nError Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Correct: {stats['correct_predictions']} ({stats['accuracy']*100:.2f}%)")
    print(f"  Misclassified: {stats['misclassified']}")
    print(f"  Large errors (>+/-1): {stats['large_errors']} ({stats['large_error_rate']*100:.2f}%)")

    # Learning curves
    if args.learning_curves:
        print("\n" + "=" * 60)
        print("LEARNING CURVE ANALYSIS")
        print("=" * 60)
        lc_results = run_learning_curve_experiment(
            features=features, labels=labels, splits=splits, alpha=np.mean(experiment_alphas),
        )
        plot_multiple_learning_curves(
            results=lc_results, metrics=["f1", "accuracy", "accuracy_pm1"],
            save_path=figures_dir / "learning_curves.png", dpi=args.dpi,
        )
        print(f"Saved learning curves to {figures_dir / 'learning_curves.png'}")

    # Cleanlab analysis
    if args.cleanlab:
        if not check_cleanlab_available():
            print("\nError: cleanlab is not installed. Install with: pip install cleanlab>=2.6.0")
            sys.exit(1)
        print("\n" + "=" * 60)
        print("DATA QUALITY ANALYSIS (CLEANLAB)")
        print("=" * 60)
        dq_report = generate_data_quality_report(
            features=features, labels=labels,
            measurement_ids=measurement_ids, alpha=np.mean(experiment_alphas),
            n_folds=10, output_dir=output_dir / "data_quality",
            random_state=args.random_state,
        )
        label_issues_df = dq_report["label_issues"]
        if len(label_issues_df) > 0:
            plot_label_issues_per_class(
                label_issues_df=label_issues_df, labels=labels,
                save_path=figures_dir / "label_issues_per_class.png", dpi=args.dpi,
            )
            plot_suggested_label_transitions(
                label_issues_df=label_issues_df,
                save_path=figures_dir / "label_transitions_heatmap.png", dpi=args.dpi,
            )
            plot_confidence_distribution(
                label_issues_df=label_issues_df, pred_probs=dq_report["pred_probs"],
                labels=labels,
                save_path=figures_dir / "confidence_distribution.png", dpi=args.dpi,
            )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
