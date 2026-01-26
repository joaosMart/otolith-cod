"""Data quality analysis using cleanlab for label error detection.

Functions for identifying potential label issues and data quality problems
using cross-validated predictions.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

# Cleanlab is optional - check if available
try:
    from cleanlab.filter import find_label_issues

    CLEANLAB_AVAILABLE = True
except ImportError:
    CLEANLAB_AVAILABLE = False


def compute_out_of_fold_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.1,
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute out-of-fold predictions using cross-validation.

    For Ridge regression, converts continuous predictions to pseudo-probabilities
    using softmax over class distances.

    Args:
        features: Feature matrix (N, D)
        labels: Label vector (N,)
        alpha: Ridge regularization parameter
        n_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility

    Returns:
        Tuple of:
            - pred_probs: Predicted probabilities (N, n_classes)
            - oof_predictions: Out-of-fold class predictions (N,)
    """
    n_samples = len(features)
    classes = np.unique(labels)
    n_classes = len(classes)

    # Initialize arrays
    pred_probs = np.zeros((n_samples, n_classes))
    oof_predictions = np.zeros(n_samples, dtype=int)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, val_idx) in enumerate(
        tqdm(skf.split(features, labels), total=n_folds, desc="CV folds")
    ):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Train Ridge model
        sample_weights = compute_sample_weight("balanced", y_train)
        model = Ridge(alpha=alpha, random_state=random_state)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        # Get continuous predictions
        y_val_pred_continuous = model.predict(X_val)

        # Convert to class predictions
        y_val_pred = np.round(y_val_pred_continuous).astype(int)
        # Clip to valid class range
        y_val_pred = np.clip(y_val_pred, classes.min(), classes.max())
        oof_predictions[val_idx] = y_val_pred

        # Convert to pseudo-probabilities using distance to each class
        # Score for each class = exp(-distance^2 / temperature)
        temperature = 1.0
        distances = np.abs(y_val_pred_continuous[:, np.newaxis] - classes[np.newaxis, :])
        scores = np.exp(-distances**2 / temperature)

        # Normalize to probabilities
        probs = scores / scores.sum(axis=1, keepdims=True)
        pred_probs[val_idx] = probs

    return pred_probs, oof_predictions


def find_label_issues_cleanlab(
    features: np.ndarray,
    labels: np.ndarray,
    pred_probs: Optional[np.ndarray] = None,
    alpha: float = 0.1,
    n_folds: int = 5,
    filter_by: str = "both",
    min_examples_per_class: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Use cleanlab to identify potential label issues.

    Args:
        features: Feature matrix (N, D)
        labels: Label vector (N,)
        pred_probs: Pre-computed predicted probabilities (N, n_classes).
            If None, will compute using cross-validation.
        alpha: Ridge regularization parameter
        n_folds: Number of CV folds (used if pred_probs is None)
        filter_by: Cleanlab filtering method ("prune_by_noise_rate", "prune_by_class",
            "both", "confident_learning", "predicted_neq_given")
        min_examples_per_class: Minimum examples per class for filtering
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with label issues, sorted by label quality score

    Raises:
        ImportError: If cleanlab is not installed
    """
    if not CLEANLAB_AVAILABLE:
        raise ImportError(
            "cleanlab is not installed. Install with: pip install cleanlab>=2.6.0"
        )

    # Compute out-of-fold predictions if not provided
    if pred_probs is None:
        print("Computing out-of-fold predictions...")
        pred_probs, _ = compute_out_of_fold_predictions(
            features=features,
            labels=labels,
            alpha=alpha,
            n_folds=n_folds,
            random_state=random_state,
        )

    # Convert 1-indexed labels (1-10) to 0-indexed (0-9) for cleanlab
    # cleanlab expects labels to start at 0 and pred_probs columns to match label indices
    labels_0indexed = labels - 1
    min_label = labels.min()

    # Find label issues using cleanlab
    print(f"Finding label issues using cleanlab (filter_by={filter_by})...")
    label_issues_mask = find_label_issues(
        labels=labels_0indexed,
        pred_probs=pred_probs,
        filter_by=filter_by,
        return_indices_ranked_by="self_confidence",
        min_examples_per_class=min_examples_per_class,
    )

    # Get indices of issues
    issue_indices = np.where(label_issues_mask)[0]

    # Build results dataframe with 1-indexed labels for display
    results = []
    for idx in issue_indices:
        given_label = labels[idx]  # Original 1-indexed label
        pred_label_0indexed = np.argmax(pred_probs[idx])
        pred_label = pred_label_0indexed + min_label  # Convert back to 1-indexed
        confidence_in_given = pred_probs[idx, labels_0indexed[idx]]

        results.append(
            {
                "sample_index": idx,
                "given_label": given_label,
                "suggested_label": pred_label,
                "confidence_in_given": confidence_in_given,
                "confidence_in_suggested": np.max(pred_probs[idx]),
            }
        )

    if len(results) == 0:
        print("No label issues found!")
        return pd.DataFrame(
            columns=[
                "sample_index",
                "given_label",
                "suggested_label",
                "confidence_in_given",
                "confidence_in_suggested",
            ]
        )

    df = pd.DataFrame(results)
    df = df.sort_values("confidence_in_given", ascending=True)

    print(f"Found {len(df)} potential label issues")

    return df


def generate_data_quality_report(
    features: np.ndarray,
    labels: np.ndarray,
    measurement_ids: Optional[np.ndarray] = None,
    alpha: float = 0.1,
    n_folds: int = 5,
    output_dir: Optional[Path] = None,
    random_state: int = 42,
) -> Dict:
    """
    Generate comprehensive data quality report using cleanlab.

    Args:
        features: Feature matrix (N, D)
        labels: Label vector (N,)
        measurement_ids: Optional measurement IDs for traceability (N,)
        alpha: Ridge regularization parameter
        n_folds: Number of CV folds
        output_dir: Optional directory to save reports
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with:
            - pred_probs: Predicted probabilities (N, n_classes)
            - oof_predictions: Out-of-fold predictions (N,)
            - label_issues: DataFrame with potential label issues
            - oof_accuracy: Out-of-fold accuracy

    Raises:
        ImportError: If cleanlab is not installed
    """
    if not CLEANLAB_AVAILABLE:
        raise ImportError(
            "cleanlab is not installed. Install with: pip install cleanlab>=2.6.0"
        )

    print("=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)

    # Compute out-of-fold predictions
    pred_probs, oof_predictions = compute_out_of_fold_predictions(
        features=features,
        labels=labels,
        alpha=alpha,
        n_folds=n_folds,
        random_state=random_state,
    )

    # Compute out-of-fold accuracy
    oof_accuracy = np.mean(oof_predictions == labels)
    print(f"\nOut-of-fold accuracy: {oof_accuracy:.4f}")

    # Find label issues
    label_issues_df = find_label_issues_cleanlab(
        features=features,
        labels=labels,
        pred_probs=pred_probs,
        alpha=alpha,
        n_folds=n_folds,
        filter_by="both",
        random_state=random_state,
    )

    # Add measurement IDs if provided
    if measurement_ids is not None and len(label_issues_df) > 0:
        label_issues_df["measurement_id"] = measurement_ids[
            label_issues_df["sample_index"].values
        ]
        # Reorder columns
        cols = [
            "sample_index",
            "measurement_id",
            "given_label",
            "suggested_label",
            "confidence_in_given",
            "confidence_in_suggested",
        ]
        label_issues_df = label_issues_df[cols]

    # Save to files if output directory provided
    if output_dir and len(label_issues_df) > 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save label issues
        label_issues_df.to_csv(output_dir / "label_issues.csv", index=False)
        print(f"\nSaved label issues to {output_dir / 'label_issues.csv'}")

        # Save out-of-fold predictions
        oof_df = pd.DataFrame(
            {
                "sample_index": np.arange(len(labels)),
                "true_label": labels,
                "oof_prediction": oof_predictions,
            }
        )
        if measurement_ids is not None:
            oof_df["measurement_id"] = measurement_ids

        oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
        print(f"Saved out-of-fold predictions to {output_dir / 'oof_predictions.csv'}")

        # Save summary statistics
        summary = {
            "total_samples": len(labels),
            "n_folds": n_folds,
            "alpha": alpha,
            "oof_accuracy": oof_accuracy,
            "n_label_issues": len(label_issues_df),
            "label_issue_rate": len(label_issues_df) / len(labels),
        }
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(output_dir / "data_quality_summary.csv", index=False)
        print(f"Saved summary to {output_dir / 'data_quality_summary.csv'}")

    print("\n" + "=" * 60)

    return {
        "pred_probs": pred_probs,
        "oof_predictions": oof_predictions,
        "label_issues": label_issues_df,
        "oof_accuracy": oof_accuracy,
    }


def check_cleanlab_available() -> bool:
    """
    Check if cleanlab is installed.

    Returns:
        True if cleanlab is available, False otherwise
    """
    return CLEANLAB_AVAILABLE
