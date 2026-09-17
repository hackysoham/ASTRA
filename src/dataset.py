# Team Astra -- Mars HiRISE Unsupervised Anomaly Detection Pipeline
# NSSC 2026, IIT Kharagpur | 
"""
dataset.py -- PyTorch Dataset and DataLoader for Mars HiRISE imagery.

Loads ~10,000 grayscale 227x227 crops indexed by crop_metadata_index.csv.
Applies min-max normalization to [0, 1].
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class MarsHiRISEDataset(Dataset):
    """PyTorch Dataset for Mars HiRISE imagery driven by a metadata CSV.

    Reads ``crop_metadata_index.csv`` to obtain the mapping from crop
    filenames to their parent ``source_image_id``.  Each image is loaded
    as single-channel grayscale, optionally resized to 227x227, and
    min-max normalised to the [0, 1] range.

    Attributes:
        image_dir (Path):   Directory containing the image files.
        metadata  (DataFrame): Contents of crop_metadata_index.csv.
        filenames (list[str]): Ordered list of crop filenames.
        source_ids (list[str]): Matching source_image_id per crop.
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    TARGET_SIZE = (227, 227)

    def __init__(
        self,
        image_dir: str,
        metadata_csv: str,
    ) -> None:
        """
        Args:
            image_dir:    Path to the directory that holds the crop images.
            metadata_csv: Path to ``crop_metadata_index.csv`` with at least
                          the columns ``filename`` and ``source_image_id``.

        Raises:
            FileNotFoundError: If *image_dir* or *metadata_csv* is missing.
            KeyError:          If required columns are absent from the CSV.
        """
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        csv_path = Path(metadata_csv)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found: {csv_path}"
            )

        # ---- Load and validate CSV ----------------------------------------
        self.metadata = pd.read_csv(csv_path)
        for col in ("filename", "source_image_id"):
            if col not in self.metadata.columns:
                raise KeyError(
                    f"Required column '{col}' missing from {csv_path}. "
                    f"Available columns: {list(self.metadata.columns)}"
                )

        # ---- Filter to files that actually exist on disk -------------------
        existing_mask = self.metadata["filename"].apply(
            lambda fn: (self.image_dir / fn).exists()
        )
        n_missing = (~existing_mask).sum()
        if n_missing > 0:
            print(
                f"[Dataset] WARNING: {n_missing} filenames in the CSV "
                f"have no matching file on disk -- they will be skipped."
            )
        self.metadata = self.metadata[existing_mask].reset_index(drop=True)

        if len(self.metadata) == 0:
            raise ValueError(
                f"No valid images found.  Checked directory: {self.image_dir}"
            )

        self.filenames: List[str] = self.metadata["filename"].tolist()
        self.source_ids: List[str] = (
            self.metadata["source_image_id"].astype(str).tolist()
        )

        print(
            f"[Dataset] Loaded {len(self)} crops from {self.image_dir} "
            f"(CSV: {csv_path.name})"
        )

    # ------------------------------------------------------------------ #
    #  Dunder helpers                                                      #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        """Return a single crop.

        Returns:
            ``(image_tensor, filename, source_image_id)``
            where *image_tensor* has shape ``(1, 227, 227)`` in [0, 1].
        """
        fname = self.filenames[idx]
        src_id = self.source_ids[idx]
        img_path = self.image_dir / fname

        # Load as grayscale (mode "L")
        img = Image.open(img_path).convert("L")

        # Safety resize (dataset should already be 227x227)
        if img.size != self.TARGET_SIZE:
            img = img.resize(self.TARGET_SIZE, Image.LANCZOS)

        # NumPy -> float32 -> min-max normalise to [0, 1]
        arr = np.array(img, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        if hi - lo > 0:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = np.zeros_like(arr)  # constant image edge-case

        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 227, 227)
        return tensor, fname, src_id


# ====================================================================== #
#  DataLoader factory                                                      #
# ====================================================================== #

def create_dataloaders(
    image_dir: str,
    metadata_csv: str,
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / validation / full-inference DataLoaders.

    Args:
        image_dir:    Directory with crop images.
        metadata_csv: Path to crop_metadata_index.csv.
        batch_size:   Mini-batch size.
        val_split:    Fraction held out for validation (0-1).
        num_workers:  Parallel loading workers.
        seed:         RNG seed for the train/val split.

    Returns:
        ``(train_loader, val_loader, full_loader)``
    """
    full_dataset = MarsHiRISEDataset(image_dir, metadata_csv)

    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=gen)

    print(f"[Split] Train: {n_train} | Val: {n_val}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    full_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, full_loader
