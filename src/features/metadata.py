"""
Metadata loading utilities for tabular feature augmentation.

Provides functions to load metadata from CSV and augment vision embeddings
with tabular features (fish length, capture quarter).

Based on: Sigurðardóttir et al. (2023) - Ecological Informatics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# Metadata columns configuration with scaling factors
METADATA_COLUMNS = {
    'length': {'scale': 0.01},  # Scale to match embedding magnitude
    'month_sin': {'scale': 1.0},
    'month_cos': {'scale': 1.0}
}


def load_metadata(
    csv_path: str = "cod_otolith_age_final_with_scale.csv",
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load metadata CSV and return relevant columns.

    Args:
        csv_path: Path to CSV file
        columns: List of columns to load (default: length + quarters)

    Returns:
        DataFrame with measurement_id as index

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        KeyError: If required columns are missing
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate required columns
    if 'measurement_id' not in df.columns:
        raise KeyError("CSV missing required column: 'measurement_id'")

    if columns is None:
        columns = list(METADATA_COLUMNS.keys())

    # Validate requested columns exist
    missing_cols = set(columns) - set(df.columns)
    if missing_cols:
        raise KeyError(f"CSV missing requested columns: {missing_cols}")

    # Keep measurement_id and requested columns
    keep_cols = ['measurement_id'] + columns
    df = df[keep_cols].copy()

    # Apply scaling
    for col, config in METADATA_COLUMNS.items():
        if col in df.columns:
            df[col] = df[col].astype(float) * config['scale']

    return df.set_index('measurement_id')


def get_tabular_features(
    measurement_ids: np.ndarray,
    metadata_df: pd.DataFrame,
) -> np.ndarray:
    """
    Get tabular features for given measurement_ids, preserving order.

    Args:
        measurement_ids: Array of measurement IDs (N,)
        metadata_df: DataFrame with measurement_id as index

    Returns:
        Tabular features array (N, n_features)

    Raises:
        ValueError: If any measurement_id is missing from metadata
    """
    # Check for missing IDs
    missing = set(measurement_ids) - set(metadata_df.index)
    if missing:
        raise ValueError(
            f"Missing metadata for {len(missing)} measurement_ids: "
            f"{sorted(list(missing))[:5]}..."
        )

    # Retrieve in order
    tabular = metadata_df.loc[measurement_ids].values

    return tabular.astype(np.float32)


def augment_embeddings(
    features: np.ndarray,
    measurement_ids: np.ndarray,
    metadata_csv: str = "cod_otolith_age_final_with_quarters.csv",
    columns: Optional[list] = None,
) -> np.ndarray:
    """
    Concatenate vision embeddings with tabular features.

    Args:
        features: Vision embeddings (N, embedding_dim)
        measurement_ids: Array of measurement IDs (N,)
        metadata_csv: Path to metadata CSV
        columns: Columns to add (default: length + quarters)

    Returns:
        Augmented features (N, embedding_dim + n_tabular)

    Raises:
        ValueError: If features and measurement_ids have different lengths
        FileNotFoundError: If metadata CSV doesn't exist
        KeyError: If required columns are missing
    """
    if len(features) != len(measurement_ids):
        raise ValueError(
            f"Features and measurement_ids length mismatch: "
            f"{len(features)} != {len(measurement_ids)}"
        )

    metadata_df = load_metadata(metadata_csv, columns)
    tabular = get_tabular_features(measurement_ids, metadata_df)

    return np.hstack([features, tabular])
