"""
dataloader.py
--------------
Constructs train and validation DataLoaders for ISIC 2019 using
ISICSkinDataset and optionally the EquilibrationSampler.

Transforms are handled entirely within ISICSkinDataset — this file
does not apply any additional image processing.
"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from isic2019_dataset import (
    ISICSkinDataset,
    get_train_dataset,
    DEFAULT_TEST_TRANSFORM,
)
from equilibration_sampler import EquilibrationSampler


# ---------------------------------------------------------------------------
# Subset wrapper
# ---------------------------------------------------------------------------

class SubsetWithTransform(Dataset):
    """
    A lightweight subset wrapper that selects a list of indices from a base
    ISICSkinDataset and optionally overrides the transform. This is needed
    to apply different transforms to the train and val splits, which are
    derived from the same base dataset object.

    Args:
        base (ISICSkinDataset):
            The full dataset to index into.
        indices (list[int]):
            The dataset indices belonging to this subset (e.g. train or val
            indices produced by random_split).
        transform (callable | None):
            If provided, overrides the transform from the base dataset.
            Used to apply DEFAULT_TEST_TRANSFORM to the val subset, since
            the base dataset is constructed with the train transform.
    """
    def __init__(self, base: ISICSkinDataset, indices, transform=None):
        self.base = base
        self.indices = list(indices)
        self.transform = transform  # None means use base.transform as-is

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        # Map the subset index back to the base dataset index, then delegate
        # to the base dataset's __getitem__ to load the image and metadata.
        # We then override the transform if one was provided for this subset.
        base_idx = self.indices[i]
        image, label, meta_tensor, image_id = self.base[base_idx]

        # If this subset has its own transform (e.g. val uses test transform),
        # undo the base transform by reloading the raw image and re-applying.
        # This is necessary because base.__getitem__ already applies its own
        # transform before we can intercept it.
        if self.transform is not None:
            from PIL import Image as PILImage
            img_path = self.base.image_dir / f"{image_id}.jpg"
            raw_image = PILImage.open(img_path).convert("RGB")
            image = self.transform(raw_image)

        return image, label, meta_tensor, image_id

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds_to(seed: int) -> None:
    """Sets random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# ---------------------------------------------------------------------------
# Setup: device, dataset, and split indices
# ---------------------------------------------------------------------------

def set_up(seed: int):
    """
    Initializes the device, loads the full training dataset, and produces
    an 80/20 train/val index split.

    The split is performed on indices only — no data is loaded at this stage.
    The actual train and val DataLoaders are constructed separately in
    make_loaders(), which applies the correct transform to each split.

    Note: the split is random and does not account for patient ID — samples
    from the same patient may appear in both train and val.

    Returns:
        device    : torch.device
        base_ds   : ISICSkinDataset — the full training dataset
        train_idx : list[int] — dataset indices for the train split
        val_idx   : list[int] — dataset indices for the val split
    """
    set_seeds_to(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Accelerator: {device}")

    # Ensure deterministic behaviour in cuDNN for reproducibility.
    # benchmark=False disables the auto-tuner which can introduce
    # non-determinism; deterministic=True forces deterministic algorithms.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load the full training dataset. Images are not loaded here — only the
    # sample list (image_id, label, meta_vec) is built at this stage.
    base_ds = get_train_dataset()

    n = len(base_ds)
    train_size = int(0.8 * n)
    val_size = n - train_size

    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        base_ds, [train_size, val_size], generator=g
    )
    train_idx = train_subset.indices
    val_idx = val_subset.indices
    return device, base_ds, train_idx, val_idx

# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------

def make_loaders(
    base_ds: ISICSkinDataset,
    train_idx: list,
    val_idx: list,
    seed: int,
    batch_size: int,
    num_workers: int,
    use_equilibration: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Constructs train and validation DataLoaders from pre-computed index splits.

    The train split uses the base dataset's DEFAULT_TRAIN_TRANSFORM.
    The val split overrides this with DEFAULT_TEST_TRANSFORM, since
    validation should never use augmentation.

    Args:
        base_ds           : the full ISICSkinDataset returned by set_up()
        train_idx         : list of dataset indices for the train split
        val_idx           : list of dataset indices for the val split
        seed              : random seed for the DataLoader generator
        batch_size        : number of samples per batch
        num_workers       : number of parallel data loading workers
        use_equilibration : if True, uses EquilibrationSampler to produce
                            class-balanced batches; if False, uses standard
                            random shuffling

    Returns:
        train_loader, val_loader
    """
    # Train subset: uses the train transform already set on base_ds
    # Val subset: overrides to DEFAULT_TEST_TRANSFORM (no augmentation)
    train_ds = SubsetWithTransform(base_ds, train_idx, transform=None)
    val_ds = SubsetWithTransform(base_ds, val_idx, transform=DEFAULT_TEST_TRANSFORM)

    pin = torch.cuda.is_available()

    # --- train loader ---
    if use_equilibration:
        sampler = EquilibrationSampler(
            labels = [base_ds.samples[i][1] for i in train_idx],
            num_classes = 8,
            batch_size = batch_size,
            seed = seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size = batch_size,
            sampler = sampler,
            drop_last = True,   # ensures every batch is exactly batch_size
            num_workers = num_workers,
            pin_memory = pin,
        )
    else:
        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_ds,
            batch_size = batch_size,
            shuffle = True,
            drop_last = True,
            num_workers = num_workers,
            generator = g,
            pin_memory = pin,
        )

    # --- val loader ---
    # shuffle=False and no drop_last so every val sample is evaluated exactly once
    val_loader = DataLoader(
        val_ds,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = pin,
    )

    return train_loader, val_loader


