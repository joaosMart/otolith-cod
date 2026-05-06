#!/usr/bin/env python3
"""
Extract and cache vision embeddings from otolith images.

Usage:
    python scripts/extract_features.py --model siglip2-so400m-14-384
    python scripts/extract_features.py --model dinov2-vitl14-reg --force
    python scripts/extract_features.py --model pe-core-l14-336 --clahe
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from torch.utils.data import DataLoader

from src.data import OtolithDataset, SUPPORTED_DATA
from src.features import (
    SUPPORTED_MODELS,
    load_model,
    extract_features,
    load_cached_embeddings,
    save_cached_embeddings,
    get_cache_path,
)
from src.utils import load_config, get_output_paths, print_device_info


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and cache vision embeddings")
    parser.add_argument(
        "--model",
        type=str,
        default="siglip2-so400m-14-384",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Vision model to use for feature extraction",
    )
    parser.add_argument(
        "--images-path",
        type=str,
        default="segmented_images",
        choices=list(SUPPORTED_DATA.keys()),
        help="Image dataset to use for feature extraction",
    )
    parser.add_argument(
        "--clahe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply CLAHE image enhancement before feature extraction",
    )
    parser.add_argument(
        "--repeat-clahe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply CLAHE enhancement twice for stronger effect",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for embeddings (default: from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for feature extraction (default: 64)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if cache exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)
    print_device_info()

    # Load config
    config = load_config(args.config)
    output_paths = get_output_paths(config)
    if args.output_dir:
        cache_dir = args.output_dir
    elif args.images_path == "segmented_images":
        cache_dir = "outputs/segmented_embeddings"
    else:
        cache_dir = str(output_paths["embeddings"])
    cache_path = get_cache_path(args.model, cache_dir, apply_clahe=args.clahe, repeat_clahe=args.repeat_clahe)

    # Check cache
    if cache_path.exists() and not args.force:
        print(f"\nCached embeddings found: {cache_path}")
        features_dict, labels, measurement_ids = load_cached_embeddings(str(cache_path))
        for key, arr in features_dict.items():
            print(f"  {key}: {arr.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Measurement IDs: {'present' if measurement_ids is not None else 'missing'}")
        if measurement_ids is not None:
            print("\nUse --force to re-extract.")
            return
        else:
            print("  Missing measurement_ids, re-extracting...")

    # Load dataset and model
    data_config = config["data"]
    print(f"\nLoading model: {args.model}")
    model, preprocess = load_model(args.model,
                                   apply_clahe=args.clahe,
                                   repeat_clahe=args.repeat_clahe)

    metadata_csv = data_config.get("metadata_csv")
    root_dir = SUPPORTED_DATA[args.images_path]
    dataset = OtolithDataset(
        root_dir=root_dir,
        transform=preprocess,
        age_range=tuple(data_config["age_range"]),
        metadata_csv=metadata_csv,
    )
    print(f"Dataset: {len(dataset)} images")
    print(f"Class distribution: {dataset.get_class_counts()}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Extract features
    features_dict, labels = extract_features(
        model, dataloader, model_name=args.model, normalize=True
    )

    # Extract measurement_ids from image filenames
    paths = dataset.get_paths()
    measurement_ids = np.array([int(p.stem) for p in paths])

    # Save
    save_cached_embeddings(str(cache_path), features_dict, labels, measurement_ids)
    print(f"\nFeature shapes:")
    for key, arr in features_dict.items():
        print(f"  {key}: {arr.shape}")
    print(f"Saved to: {cache_path}")
    print("\nFeature extraction complete!")


if __name__ == "__main__":
    main()
