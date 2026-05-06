#!/usr/bin/env python3
"""
Quick test script to verify metadata loading and augmentation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.data import load_metadata, augment_embeddings

# Test 1: Load metadata
print("Test 1: Loading metadata...")
try:
    metadata_df = load_metadata("cod_otolith_age_final_with_quarters.csv")
    print(f"  ✓ Loaded metadata with {len(metadata_df)} rows")
    print(f"  ✓ Columns: {list(metadata_df.columns)}")
    print(f"  ✓ Sample row:\n{metadata_df.head(1)}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 2: Create dummy embeddings
print("\nTest 2: Creating dummy embeddings...")
n_samples = 100
embedding_dim = 1152
dummy_features = np.random.randn(n_samples, embedding_dim).astype(np.float32)

# Get first 100 measurement_ids from metadata
dummy_measurement_ids = metadata_df.index[:n_samples].values
print(f"  ✓ Created {n_samples} dummy embeddings of dimension {embedding_dim}")
print(f"  ✓ Using measurement_ids: {dummy_measurement_ids[:5]}...")

# Test 3: Augment embeddings
print("\nTest 3: Augmenting embeddings with tabular features...")
try:
    augmented = augment_embeddings(
        dummy_features,
        dummy_measurement_ids,
        "cod_otolith_age_final_with_quarters.csv",
    )
    print(f"  ✓ Original shape: {dummy_features.shape}")
    print(f"  ✓ Augmented shape: {augmented.shape}")
    print(f"  ✓ Added {augmented.shape[1] - dummy_features.shape[1]} tabular features")

    # Verify the augmented features
    print(f"\n  Sample augmented row (last 4 features are tabular):")
    print(f"    {augmented[0, -4:]}")  # Should be [length*0.01, q2, q3, q4]

except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

print("\n✓ All tests passed!")
