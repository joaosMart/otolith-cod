#!/usr/bin/env python3
"""
Feature Importance Analysis: SHAP + Permutation Importance + Forward Selection.

Analyzes how each input feature (SigLIP embeddings vs tabular metadata)
contributes to the Ridge classifier's age predictions.

Usage:
    python scripts/analyze_feature_importance.py \
        --embeddings outputs/segmented_embeddings/siglip2-so400m-14-384_embeddings.npz \
        --results outputs/results/shallow_siglip2/results.json \
        --output-dir outputs/feature_importance
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from src.features import load_cached_embeddings, augment_embeddings
from src.data import create_train_test_splits
from src.evaluation.feature_importance import (
    build_feature_groups,
    compute_shap_values,
    compute_grouped_permutation_importance,
    plot_shap_summary,
    plot_shap_dependence,
    plot_permutation_importance,
    run_forward_selection,
    plot_forward_selection,
    save_results_json,
    plot_shap_dependence_month,
    plot_shap_dependence_month_impact,
    plot_shap_dependence_month_net,
    TABULAR_GROUP_SPEC,
    TABULAR_COLUMN_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Feature importance analysis")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="outputs/segmented_embeddings/siglip2-so400m-14-384_clahe_embeddings.npz",
        help="Path to embeddings .npz file",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="outputs/results/shallow_siglip2/results.json",
        help="Path to results.json from training (used for config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/feature_importance",
        help="Directory for output plots and data",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="cod_otolith_age_final_with_scale.csv",
        help="Path to metadata CSV",
    )
    parser.add_argument(
        "--tabular-columns",
        type=str,
        default="length,month_sin,month_cos,shot_latitude,shot_longitude,is_survey,seg_width_px,seg_height_px,seg_aspect_ratio",
        help="Comma-separated tabular columns",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="82,15,4,95,36,32,29,18,14,87",
        help="Comma-separated seeds for splits",
    )
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--alpha-log-min", type=float, default=1.5)
    parser.add_argument("--alpha-log-max", type=float, default=3)
    parser.add_argument("--alpha-steps", type=int, default=30)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--perm-repeats", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load config from results.json
    if Path(args.results).exists():
        with open(args.results) as f:
            saved_results = json.load(f)
        config = saved_results.get("config", {})
        # Override defaults with saved config
        if "seeds" in config:
            seeds_from_config = config["seeds"]
        else:
            seeds_from_config = None
    else:
        seeds_from_config = None

    # Parse seeds (CLI takes priority)
    seeds = [int(s) for s in args.seeds.split(",")]

    # Load embeddings
    print("=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    features, labels, measurement_ids = load_cached_embeddings(args.embeddings)
    embedding_dim = features.shape[1]
    print(f"  Embeddings: {features.shape}")

    # Augment with tabular features
    columns = args.tabular_columns.split(",")
    print(f"  Adding tabular features: {columns}")
    features = augment_embeddings(features, measurement_ids, args.metadata_csv, columns)
    print(f"  Augmented shape: {features.shape}")

    # Alpha range
    alpha_range = np.logspace(args.alpha_log_min, args.alpha_log_max, args.alpha_steps)

    # Create splits
    splits = create_train_test_splits(
        labels=labels,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        seeds=seeds,
    )
    print(f"  Splits: {len(splits)} independent experiments")

    # Feature groups
    feature_groups = build_feature_groups(embedding_dim)

    # --- Phase 1: SHAP + Permutation Importance per split ---
    print("\n" + "-" * 60)
    print("Phase 1: SHAP & Permutation Importance")
    print("-" * 60)

    all_shap_importances = []
    all_perm_importances = []
    all_shap_values_raw = []
    all_X_test_scaled = []
    all_X_test_unscaled = []

    for i, split in enumerate(splits):
        print(f"\n  Split {i + 1}/{len(splits)}...")

        X_train = features[split.train_indices]
        y_train = labels[split.train_indices]
        X_test = features[split.test_indices]
        y_test = labels[split.test_indices]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train model
        model = RidgeCV(alphas=alpha_range, cv=args.cv_folds)
        model.fit(X_train_s, y_train)

        # SHAP
        shap_imp = compute_shap_values(model, X_train_s, X_test_s, feature_groups)
        all_shap_importances.append(shap_imp)

        # Store raw SHAP values for dependence plots
        import shap
        explainer = shap.LinearExplainer(model, X_train_s)
        raw_shap = explainer.shap_values(X_test_s)
        all_shap_values_raw.append(raw_shap)
        all_X_test_scaled.append(X_test_s)
        all_X_test_unscaled.append(X_test)

        # Permutation importance
        perm_imp = compute_grouped_permutation_importance(
            model, X_test_s, y_test, feature_groups,
            n_repeats=args.perm_repeats, random_state=seeds[i],
        )
        all_perm_importances.append(perm_imp)

        print(f"    SHAP top: {sorted(shap_imp.items(), key=lambda x: -x[1])[:3]}")

    # --- Phase 2: Plots ---
    print("\n" + "-" * 60)
    print("Phase 2: Generating Plots")
    print("-" * 60)

    # SHAP summary
    plot_shap_summary(all_shap_importances, str(output_dir / "shap_summary.png"))

    # SHAP dependence plots for ALL tabular features
    tabular_group_names = [name for name, _ in TABULAR_GROUP_SPEC]
    agg_shap = {}
    for name in tabular_group_names:
        agg_shap[name] = np.mean([si[name] for si in all_shap_importances])

    for feat_name, cols in TABULAR_GROUP_SPEC:
        if feat_name == "Month":
            sin_idx = embedding_dim + TABULAR_COLUMN_NAMES.index("month_sin")
            cos_idx = embedding_dim + TABULAR_COLUMN_NAMES.index("month_cos")
            # Signed sin/cos plot (direction of influence)
            plot_shap_dependence_month(
                all_shap_values_raw,
                all_X_test_unscaled,
                sin_idx,
                cos_idx,
                str(output_dir / "shap_dependence_month.png"),
            )
            # Overall impact plot (|SHAP| combined)
            plot_shap_dependence_month_impact(
                all_shap_values_raw,
                all_X_test_unscaled,
                sin_idx,
                cos_idx,
                str(output_dir / "shap_dependence_month_impact.png"),
            )

            plot_shap_dependence_month_net(
                all_shap_values_raw,
                all_X_test_unscaled,
                sin_idx,
                cos_idx,
                str(output_dir / "shap_dependence_month_net.png"),
            )
        else:
            feat_idx = embedding_dim + TABULAR_COLUMN_NAMES.index(cols[0])
            safe_name = feat_name.lower().replace(" ", "_")
            plot_shap_dependence(
                all_shap_values_raw,
                all_X_test_unscaled,
                feat_idx,
                feat_name,
                str(output_dir / f"shap_dependence_{safe_name}.png"),
            )

    # Permutation importance
    plot_permutation_importance(all_perm_importances, str(output_dir / "permutation_importance.png"))

    # --- Phase 3: Forward Selection Ablation ---
    print("\n" + "-" * 60)
    print("Phase 3: Forward Selection Ablation")
    print("-" * 60)

    # Determine SHAP-ranked order for tabular features
    tabular_order = [name for name, _ in sorted(agg_shap.items(), key=lambda x: -x[1])]
    group_order = ["SigLIP Embeddings"] + tabular_order
    print(f"  Feature addition order: {group_order}")

    forward_results = run_forward_selection(
        features=features,
        labels=labels,
        splits=splits,
        alpha_range=alpha_range,
        cv_folds=args.cv_folds,
        feature_groups=feature_groups,
        group_order=group_order,
    )

    plot_forward_selection(forward_results, str(output_dir / "forward_selection.png"))

    # --- Save all results ---
    print("\n" + "-" * 60)
    print("Saving Results")
    print("-" * 60)

    save_results_json(
        all_shap_importances,
        all_perm_importances,
        forward_results,
        str(output_dir / "feature_importance.json"),
    )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
