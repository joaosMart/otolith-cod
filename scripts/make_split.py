#!/usr/bin/env python3
"""
Create the canonical train/test split, once, for every experiment to share.

Every condition in the paper has to be scored on the same test images, otherwise
differences between them are partly differences between test sets. Previously the
split was a by-product of whichever run happened to go first, which made the
ordering of the pipeline load-bearing. Here it is an explicit artifact.

The split is keyed by measurement ID rather than row index, so it stays valid
across encoders whose embedding files list images in a different order.

Deterministic: same dataset and same seed always give the same split, so
re-running this is a no-op and every job can safely start with it.

    python scripts/make_split.py
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from src.data import OtolithDataset, save_split_by_ids
from src.utils import load_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--output", default="outputs/splits/main_split.json")
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="Rewrite even if the file exists. Only do this if you "
                        "intend to invalidate every result computed so far.")
    return p.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)

    if output.exists() and not args.force:
        print(f"Split already exists at {output}. Leaving it alone.")
        return 0

    config = load_config(args.config)
    data_config = config["data"]

    dataset = OtolithDataset(
        Path(data_config["root_dir"]),
        transform=None,
        age_range=tuple(data_config["age_range"]),
        metadata_csv=data_config["metadata_csv"],
    )
    labels = np.array([age for _, age in dataset.samples])
    measurement_ids = np.array([int(p.stem) for p, _ in dataset.samples])
    print(f"{len(labels)} images across ages "
          f"{labels.min()} to {labels.max()}")

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.test_ratio, random_state=args.seed
    )
    train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))

    output.parent.mkdir(parents=True, exist_ok=True)
    save_split_by_ids(output, {
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "train_measurement_ids": measurement_ids[train_idx].tolist(),
        "test_measurement_ids": measurement_ids[test_idx].tolist(),
    })

    print(f"train {len(train_idx)}   test {len(test_idx)}")
    for age in np.unique(labels):
        print(f"  age {age:>2}: train {int((labels[train_idx] == age).sum()):>5}   "
              f"test {int((labels[test_idx] == age).sum()):>4}")
    print(f"\nWritten to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
