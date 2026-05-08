"""
Otolith Dataset for PyTorch.

This module provides the OtolithDataset class for loading cod otolith images
organized in a directory structure where each subdirectory represents an age class.

Structure expected:
    otolith_images/
    ├── 1/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── 2/
    │   └── ...
    └── ...

Based on: Sigurðardóttir et al. (2023) - Ecological Informatics
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Union
import numpy as np


SUPPORTED_DATA = {
    "raw_images": "otolith_images/raw_images",
    "segmented_images": "otolith_images/segmented_images",
}


class OtolithDataset(Dataset):
    """
    PyTorch Dataset for cod otolith images organized by age.

    Attributes:
        root_dir: Path to otolith_images directory
        transform: Image transforms (CLIP preprocessing or custom)
        age_range: Tuple (min_age, max_age) to clip ages
        samples: List of (image_path, age) tuples

    Example:
        >>> dataset = OtolithDataset(
        ...     root_dir="otolith_images",
        ...     transform=get_clip_preprocess(),
        ...     age_range=(1, 10)
        ... )
        >>> image, age = dataset[0]
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        age_range: Tuple[int, int] = (1, 10),
        indices: Optional[List[int]] = None,
        image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        metadata_csv: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the OtolithDataset.

        Args:
            root_dir: Path to the root directory containing age subdirectories
            transform: Optional transform to apply to images
            age_range: (min_age, max_age) - ages outside this range are clipped
            indices: Optional list of indices to use (for train/val/test splits)
            image_extensions: Tuple of valid image file extensions

        Raises:
            FileNotFoundError: If root_dir does not exist
            NotADirectoryError: If root_dir is not a directory
            ValueError: If no valid samples are found
        """
        self.root_dir = Path(root_dir)

        # Validate root directory exists
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {self.root_dir}")

        self.transform = transform
        self.age_range = age_range
        self.image_extensions = image_extensions

        # Load valid measurement IDs from metadata CSV
        self._valid_ids: Optional[set] = None
        if metadata_csv is not None:
            df = pd.read_csv(metadata_csv, usecols=["measurement_id"])
            self._valid_ids = set(df["measurement_id"].astype(int).values)

        # Collect all samples
        self._all_samples = self._collect_samples()

        # Filter by indices if provided (for train/val/test splits)
        if indices is not None:
            self.samples = [self._all_samples[i] for i in indices]
        else:
            self.samples = self._all_samples

        # Validate we have samples
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {self.root_dir}")

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        """
        Collect all image paths and their ages, clipping to age_range.

        Ages outside the specified range are clipped to the boundaries,
        following the paper's methodology.

        Returns:
            List of (image_path, clipped_age) tuples
        """
        samples = []
        min_age, max_age = self.age_range

        # Iterate through subdirectories
        for age_dir in sorted(self.root_dir.iterdir()):
            # Skip non-directories and hidden files
            if not age_dir.is_dir() or age_dir.name.startswith("."):
                continue

            # Parse age from directory name
            try:
                age = int(age_dir.name)
            except ValueError:
                # Skip directories that aren't numeric ages
                continue

            # Clip ages outside range to boundaries (as per paper)
            if age < min_age:
                clipped_age = min_age
            elif age > max_age:
                clipped_age = max_age
            else:
                clipped_age = age

            # Collect all images in this age directory
            for ext in self.image_extensions:
                for img_path in age_dir.glob(f"*{ext}"):
                    # Filter by metadata CSV if provided
                    if self._valid_ids is not None:
                        try:
                            mid = int(img_path.stem)
                        except ValueError:
                            continue
                        if mid not in self._valid_ids:
                            continue
                    samples.append((img_path, clipped_age))

        samples.sort(key=lambda s: s[0].name)
        return samples

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (transformed_image, age)
        """
        img_path, age = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, age

    def get_labels(self) -> np.ndarray:
        """
        Return all labels for stratification.

        Returns:
            Array of age labels for all samples
        """
        return np.array([s[1] for s in self.samples])

    def get_paths(self) -> List[Path]:
        """
        Return all image paths.

        Returns:
            List of Path objects for all images
        """
        return [s[0] for s in self.samples]

    def get_class_counts(self) -> dict:
        """
        Get the count of samples per age class.

        Returns:
            Dictionary mapping age to count
        """
        labels = self.get_labels()
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def __repr__(self) -> str:
        """String representation of the dataset."""
        class_counts = self.get_class_counts()
        return (
            f"OtolithDataset(\n"
            f"  root_dir={self.root_dir},\n"
            f"  n_samples={len(self)},\n"
            f"  age_range={self.age_range},\n"
            f"  class_counts={class_counts}\n"
            f")"
        )
