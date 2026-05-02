"""
equilibration_sampler.py
-------------------------
Implements Equilibration Mini-batch Sampling for training on class-imbalanced
datasets, based on the algorithm described in:

    "An Improved Learning Algorithm of Neural Networks With Imbalanced Data"
    IEEE Transactions on Neural Networks and Learning Systems, 2020.
    DOI: 10.1109/TNNLS.2020.2978386

Background: the class imbalance problem
-----------------------------------------------------
In datasets like ISIC 2019, some classes (e.g. NV - melanocytic nevi) contain
orders of magnitude more samples than others (e.g. DF - dermatofibroma). When
training naively on such data, the model is exposed to majority-class examples
far more often than minority-class examples, which biases it toward predicting
the majority class and degrades performance on rare but clinically important
conditions.

How equilibration mini-batch sampling addresses this
-----------------------------------------------------
Rather than altering loss weights after the fact, this approach fixes the
problem at the data level by ensuring every mini-batch is perfectly balanced
across all classes before training even sees the data.

Given K classes and a desired mini-batch size m, a fixed per-class quota is
computed once at initialization:
    Q = m // K   (samples per class per batch)

When assembling each mini-batch, every class is treated independently:

    Major class  (c_i > Q):  The class has more samples than we need for one
                              batch, so we randomly draw exactly Q of them
                              WITHOUT replacement. This is under-sampling.
                              Because a fresh random subset is drawn each batch,
                              most major-class samples will still be seen across
                              a full training epoch even though only a fraction
                              appear in any individual batch.

    Minor class  (c_i < Q):  The class has fewer samples than the quota, so we
                              first include all c_i available samples, then
                              randomly draw (Q - c_i) additional samples WITH
                              replacement to reach the quota. This is
                              over-sampling. Taking all real samples first
                              minimises unnecessary duplication compared to
                              pure replacement sampling from scratch.

    Exact match  (c_i == Q): The class count exactly matches the quota, so all
                              samples are included as-is with no modification.

The result is that every mini-batch contains exactly Q * K = m samples split
evenly across all classes, giving the model equal exposure to each class
regardless of how imbalanced the underlying dataset is.

Epoch length
---------------------------------------------
Rather than defining an epoch as N / m batches (which would under-represent
major classes since we only use Q of their samples per batch), the number of
batches per epoch is instead anchored on the largest class:
    num_batches = ceil(max_class_count / Q)
This means the largest class drives epoch length, and the random under-sampling
ensures most of its samples appear across the epoch in aggregate.

Usage with ISICSkinDataset
---------------------------------------------
    from torch.utils.data import DataLoader
    from equilibration_sampler import EquilibrationSampler

    train_dataset = get_train_dataset(...)

    sampler = EquilibrationSampler(
        labels      = [sample[1] for sample in train_dataset.samples],
        num_classes = 8,
        batch_size  = 32,   # Q = 32 // 8 = 4 samples per class per batch
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = 32,
        sampler     = sampler,
        drop_last   = True,   # strongly recommended — see Notes below
        num_workers = 4,
    )

    # Standard training loop — no changes needed here
    for images, labels, meta_data, image_ids in train_loader:
        logits = model(images, meta_data)
        loss   = criterion(logits, labels)
        ...

Notes
---------------------------------------------
- This is a map-style sampler: __iter__ pre-generates the complete list of
  dataset indices for the entire epoch upfront, then yields them one by one.
  DataLoader uses these indices to look up individual samples from the dataset.
  This means all sampling decisions happen at the start of each epoch, before
  any batches are processed.

- drop_last=True is strongly recommended. The sampler constructs batches of
  exactly Q * num_classes indices. If drop_last=False, DataLoader may try to
  form a partial batch from leftover indices at the end of the epoch, which
  would break the per-batch class balance guarantee.

- Because __iter__ is called freshly each epoch, the random under- and
  over-sampling draws are different every epoch. Major classes therefore see
  different random subsets across epochs, improving generalisation over time.

- batch_size should ideally be divisible by num_classes to keep Q a whole
  number. If it is not, Q is truncated (floor division) and the effective batch
  size will be slightly smaller than requested; a warning is raised.
"""

import math
import random
import warnings
import torch
from collections import defaultdict
from typing import Iterator, List

from torch.utils.data import Sampler


class EquilibrationSampler(Sampler):
    """
    Equilibration Mini-batch Sampler.

    Assembles balanced mini-batches from an imbalanced dataset by
    under-sampling classes that exceed the per-class quota and
    over-sampling classes that fall below it. Every batch produced
    contains exactly Q * num_classes samples with Q samples per class,
    where Q = batch_size // num_classes.

    Args:
        labels (List[int]):
            Integer class label for every sample in the dataset, in the
            same order as the underlying dataset indices (i.e. labels[i]
            is the class of dataset[i]). For ISICSkinDataset, pass:
                [sample[1] for sample in dataset.samples]
        num_classes (int):
            Total number of classes K. For ISIC 2019 this is 8.
        batch_size (int):
            Desired total mini-batch size m. Must be >= num_classes.
            Ideally divisible by num_classes so Q is exact.
        shuffle (bool):
            If True (default), each class's sample pool is shuffled at
            the start of every epoch before batches are built, and the
            indices within each batch are also shuffled. This ensures the
            model does not see classes in a fixed block order, which could
            otherwise introduce gradient bias. Set to False only for
            deterministic debugging or reproducibility testing.
        seed (int | None):
            Optional integer seed for the internal random number generator.
            When set, sampling is fully reproducible across runs.
    """

    def __init__(
        self,
        labels: List[int],
        num_classes: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        super().__init__()

        if batch_size < num_classes:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= num_classes ({num_classes}) "
                "so that every class can receive at least 1 sample per batch."
            )
        if batch_size % num_classes != 0:
            warnings.warn(
                f"batch_size ({batch_size}) is not divisible by num_classes ({num_classes}). "
                f"The per-class quota Q will be floored to {batch_size // num_classes}, "
                f"giving an effective batch size of {(batch_size // num_classes) * num_classes} "
                f"rather than {batch_size}."
            )

        self.labels      = labels
        self.num_classes = num_classes
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.seed        = seed

        # Q = m // K — the fixed per-class quota, computed once and reused
        # for every batch throughout training (Algorithm 1, line 1).
        self.quota = batch_size // num_classes

        # Build a lookup from class index to the list of dataset indices that
        # belong to that class. This is constructed once at initialization and
        # reused every epoch. Each entry is a list of integers where each
        # integer is a valid index into the dataset (i.e. dataset[index] gives
        # a sample of that class).
        self._class_to_indices: dict[int, List[int]] = defaultdict(list)
        for dataset_idx, label in enumerate(labels):
            self._class_to_indices[label].append(dataset_idx)

        # Verify every expected class has at least one sample. If a class is
        # completely absent from the dataset, over-sampling it is impossible.
        for cls in range(num_classes):
            if cls not in self._class_to_indices or len(self._class_to_indices[cls]) == 0:
                raise ValueError(
                    f"Class {cls} has no samples in the provided labels. "
                    "EquilibrationSampler requires every class to have at least "
                    "one sample so that over-sampling is always well-defined."
                )

        # Determine epoch length by anchoring on the largest class. We want
        # enough batches that the most frequent class gets approximately full
        # coverage across the epoch (since only Q of its samples appear per
        # batch, we need ceil(max_count / Q) batches to expose all of them).
        max_class_count   = max(len(v) for v in self._class_to_indices.values())
        self._num_batches = math.ceil(max_class_count / self.quota)

        # Initialize RNG once so its state evolves across epochs, giving
        # genuine epoch-to-epoch variation in batch composition. The seed
        # controls the starting state for reproducibility but is not reset
        # at the start of each epoch.
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------
    # Internal: build the full sequence of indices for one epoch
    # ------------------------------------------------------------------

    def _build_epoch_indices(self) -> List[int]:
        """
        Generates the complete ordered list of dataset indices to be yielded
        during one epoch. Concretely, this means constructing self._num_batches
        balanced batches and concatenating their indices into a flat list.

        Called once per epoch at the start of __iter__. Because self._rng
        is initialized once at construction and its state carries forward
        across epochs, the random draws differ every epoch as intended.
        """
        rng = self._rng

        # Make a fresh, optionally shuffled copy of each class's index pool
        # for this epoch. We copy so that the original self._class_to_indices
        # mapping is never mutated and can be reused next epoch.
        class_pools: dict[int, List[int]] = {}
        for cls, indices in self._class_to_indices.items():
            pool = list(indices)
            if self.shuffle:
                rng.shuffle(pool)
            class_pools[cls] = pool

        all_indices: List[int] = []

        for _ in range(self._num_batches):
            batch_indices: List[int] = []

            for cls in range(self.num_classes):
                pool = class_pools[cls]
                c_i  = len(pool)   # how many samples this class currently has
                Q_i  = self.quota  # how many we want from it in this batch

                if c_i > Q_i:
                    # Major class: more samples available than the quota.
                    # Randomly select exactly Q_i of them without replacement,
                    # so the same sample cannot appear twice within one batch.
                    selected = rng.sample(pool, Q_i)

                elif c_i < Q_i:
                    # Minor class: fewer samples than the quota.
                    # Include every available sample once (guaranteeing full
                    # coverage of the real data), then fill the remaining
                    # (Q_i - c_i) slots by sampling with replacement.
                    # This hybrid approach limits duplication compared to
                    # drawing all Q_i samples purely with replacement.
                    remainder = Q_i - c_i
                    selected  = list(pool) + rng.choices(pool, k=remainder)

                else:
                    # Exact match: class size perfectly equals the quota.
                    # No modification needed — include all samples as-is.
                    selected = list(pool)

                batch_indices.extend(selected)

            # Shuffle the assembled batch indices before appending, so that
            # samples from different classes are interleaved rather than
            # appearing in K consecutive class-blocks. This ensures the model
            # sees a varied mix of classes throughout each batch rather than
            # processing all samples of one class before moving to the next.
            if self.shuffle:
                rng.shuffle(batch_indices)

            all_indices.extend(batch_indices)

        return all_indices

    # ------------------------------------------------------------------
    # Sampler interface (called by DataLoader)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[int]:
        """
        Entry point called by DataLoader at the start of each epoch.
        Rebuilds the full index sequence from scratch so that every epoch
        uses fresh random samples (new under/over-sampling draws).
        """
        epoch_indices = self._build_epoch_indices()
        return iter(epoch_indices)

    def __len__(self) -> int:
        """
        Total number of dataset indices emitted per epoch.
        Equals num_batches * (quota * num_classes), i.e. the total count
        of samples across all balanced batches in the epoch.
        """
        return self._num_batches * (self.quota * self.num_classes)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def class_summary(self) -> None:
        """
        Prints a human-readable summary of the sampler configuration:
        the per-class quota, whether each class will be under- or
        over-sampled, and the total number of batches per epoch.

        Call this after construction and before training to verify that
        the quota and class modes look as expected for your dataset.
        """
        print(f"\nEquilibration Sampler Summary")
        print(f"  batch_size = {self.batch_size}")
        print(f"  num_classes = {self.num_classes}")
        print(f"  quota per class (Q) = {self.quota} samples/class/batch")
        print(f"  batches per epoch = {self._num_batches}")
        print(f"  total indices / epoch = {len(self)}\n")
        print(f"  {'Class':>6}  {'Count (c_i)':>12}  {'Quota (Q_i)':>12}  {'Mode':>14}")
        print(f"  {'-'*52}")
        for cls in range(self.num_classes):
            c_i = len(self._class_to_indices[cls])
            if c_i > self.quota:
                mode = "under-sample"
            elif c_i < self.quota:
                mode = "over-sample"
            else:
                mode = "exact"
            print(f"  {cls:>6}  {c_i:>12}  {self.quota:>12}  {mode:>14}")
        print()


    # ------------------------------------------------------------------
    # Logging utility for batches
    # ------------------------------------------------------------------
    @staticmethod
    def log_batch_class_distribution(
            labels: torch.Tensor,
            num_classes: int,
            batch_idx: int,
            log_every_n: int = 50,
    ) -> None:
        """
        Logs the class count distribution for a single batch. Intended to be
        called inside the training loop to verify that the EquilibrationSampler
        is producing balanced batches as expected.

        Every batch should show exactly Q = batch_size // num_classes samples
        per class. Any deviation indicates a problem with sampler configuration
        or DataLoader settings (e.g. drop_last=False producing a partial batch).

        Args:
            labels (torch.Tensor):
                The integer label tensor for the current batch, shape [B].
                This is the second element returned by ISICSkinDataset.__getitem__,
                i.e. the 'labels' variable in a typical training loop.
            num_classes (int):
                Total number of classes. Should match the value passed to
                EquilibrationSampler at construction (8 for ISIC 2019).
            batch_idx (int):
                The current batch index within the epoch, used for log output
                and to control how often logging fires.
            log_every_n (int):
                Log every N batches to avoid flooding output during training.
                Set to 1 to log every single batch (useful for initial debugging),
                or a larger value for periodic spot-checks during normal training.
        """
        if batch_idx % log_every_n != 0:
            return

        counts = torch.bincount(labels, minlength=num_classes)
        expected_q = len(labels) // num_classes
        is_balanced = all(c == expected_q for c in counts.tolist())

        lines = [f"\n  Batch {batch_idx} class distribution (balanced={is_balanced}):"]
        for cls_idx, count in enumerate(counts.tolist()):
            deviation = count - expected_q
            deviation_str = f"  <-- WARNING: expected {expected_q}" if deviation != 0 else ""
            lines.append(f"    class {cls_idx:>2}: {count:>4} samples{deviation_str}")
        lines.append(f"  Total: {len(labels)} samples, expected Q={expected_q} per class")
        print("\n".join(lines))
