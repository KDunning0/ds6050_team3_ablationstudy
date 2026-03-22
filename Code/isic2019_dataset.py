"""
isic2019_dataset_v3.py
--------------------
Data loading and preprocessing for the ISIC 2019 skin lesion dataset.

Includes ability to: 
  - Download images and CSVs via the isic-cli tool
  - Create an ISICSkinDataset: a PyTorch Dataset that loads images, integer class labels,
    and one-hot encoded metadata (age, sex, anatomical site) for use with
    SkinEffnetB4 (with or without MetaBlock).

---------------------------------------------------------------------------
DOWNLOAD INSTRUCTIONS FOR DATA
---------------------------------------------------------------------------
Install and use the isic-cli tool:
    https://github.com/ImageMarkup/isic-cli/releases/latest

Download training images (collection id = 65):
    isic image download --collections 65 data/train/images/

Download test images (collection id = 72):
    isic image download --collections 72 data/test/images/

Download ground truth CSVs manually from:
    Training: https://challenge.isic-archive.com/data/#2019
        -> "ISIC_2019_Training_GroundTruth.csv"  (save to data/train/)
    Test:     https://challenge.isic-archive.com/data/#2019
        -> "ISIC_2019_Test_GroundTruth.csv"      (save to data/test/)

Download 2019 challenge metadata CSVs manually from: 
    Training: https://api.isic-archive.com/collections/65/ (select "Actions" -> "Download Metadata")
        -> "challenge-2019-training_metadata.csv" (save to data/train/)

    Test: https://api.isic-archive.com/collections/72/ (select "Actions" -> "Download Metadata")
        -> "challenge-2019-test_metadata.csv" (save to data/test/)

---------------------------------------------------------------------------
METADATA ENCODING
---------------------------------------------------------------------------
Metadata is one-hot encoded into a fixed-length float32 vector of length
META_DIM = 15, composed of three groups, each with an explicit unknown flag:

    Age (6 dims):
        [0]  age < 30
        [1]  30 <= age < 45
        [2]  45 <= age < 60
        [3]  60 <= age < 75
        [4]  age >= 75
        [5]  age_unknown  <- 1 if age_approx is missing, 0 otherwise

    Sex (3 dims):
        [6]  female
        [7]  male
        [8]  sex_unknown  <- 1 if sex is missing, 0 otherwise

    Anatomical site (6 dims):
        [9]   torso      (anterior torso, lateral torso, posterior torso)
        [10]  head/neck
        [11]  lower extremity
        [12]  upper extremity
        [13]  other      (oral/genital, palms/soles)
        [14]  site_unknown  <- 1 if anatom_site_general is missing, 0 otherwise

    Total vector length: 6 + 3 + 6 = 15  ->  set meta_num=15 in SkinEffnetB4.

    The unknown flags make missingness an explicit, learnable signal rather
    than an implicit all-zeros pattern. This is important because ~11.5% of
    training samples have missing metadata, and missingness correlates with
    data source (attribution), meaning all-zeros would conflate "unknown"
    with source-level confounds.
---------------------------------------------------------------------------
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Class label mapping
# ---------------------------------------------------------------------------

# The 8 valid diagnostic classes in ISIC 2019. UNK is excluded as it is
# always empty in the training ground truth CSV and has no test ground truth.
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
# CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)} - available if needed

# ---------------------------------------------------------------------------
# Metadata encoding layout
# ---------------------------------------------------------------------------

# Age bracket boundaries used to bin age_approx into 5 one-hot slots,
# plus 1 explicit unknown slot. Each known age maps to exactly one bracket
# dimension; missing age sets the age_unknown dimension instead.
AGE_BRACKETS = [
    (0,  30),   # dim 0: age < 30
    (30, 45),   # dim 1: 30 <= age < 45
    (45, 60),   # dim 2: 45 <= age < 60
    (60, 75),   # dim 3: 60 <= age < 75
    (75, 200),  # dim 4: age >= 75
]
N_AGE_DIMS = len(AGE_BRACKETS) + 1  # 5 brackets + 1 unknown = 6
AGE_UNKNOWN_DIM = N_AGE_DIMS - 1    # index 5 within the age block

# Sex encoding: [female, male, sex_unknown].
# Known sex sets one of the first two dims; missing sex sets the third.
SEX_CATEGORIES = ["female", "male"]
N_SEX_DIMS = len(SEX_CATEGORIES) + 1  # 2 categories + 1 unknown = 3
SEX_UNKNOWN_DIM = N_SEX_DIMS - 1      # index 2 within the sex block

# Anatomical site groupings: 5 broader groups plus 1 unknown slot.
# Known site sets one of the first five dims; missing site sets the sixth.
SITE_GROUPS = {
    "torso":           ["anterior torso", "lateral torso", "posterior torso"],
    "head/neck":       ["head/neck"],
    "lower extremity": ["lower extremity"],
    "upper extremity": ["upper extremity"],
    "other":           ["oral/genital", "palms/soles"],
}
SITE_GROUP_NAMES = list(SITE_GROUPS.keys())     # ordered list for index lookup

# Flat map from raw ISIC site string -> group name, for fast lookup
SITE_RAW_TO_GROUP = {
    raw: group
    for group, raws in SITE_GROUPS.items()
    for raw in raws
}
N_SITE_DIMS = len(SITE_GROUP_NAMES) + 1    # 5 groups + 1 unknown = 6
SITE_UNKNOWN_DIM = N_SITE_DIMS - 1         # index 5 within the site block

# ---------------------------------------------------------------------------
# Derived offsets: where each block starts in the flat metadata vector.
# These are used in _encode_metadata to address the correct dimensions.
# ---------------------------------------------------------------------------
AGE_OFFSET  = 0                          # dims  0-5
SEX_OFFSET  = AGE_OFFSET  + N_AGE_DIMS  # dims  6-8
SITE_OFFSET = SEX_OFFSET  + N_SEX_DIMS  # dims  9-14

# Total metadata vector length: must match meta_num in SkinEffnetB4.
META_DIM = N_AGE_DIMS + N_SEX_DIMS + N_SITE_DIMS  # 6 + 3 + 6 = 15

# ---------------------------------------------------------------------------
# Default image transforms
# ---------------------------------------------------------------------------

# EfficientNet-B4 expects 380x380 inputs. Resizing and normalization occurs using
# ImageNet statistics. No augmentation is applied here; augmentation 
# can be added via the transform argument when constructing ISICSkinDataset if needed.
EFFNET_B4_INPUT_SIZE = 380

DEFAULT_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((EFFNET_B4_INPUT_SIZE, EFFNET_B4_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std=[0.229, 0.224, 0.225]),
])

DEFAULT_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((EFFNET_B4_INPUT_SIZE, EFFNET_B4_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Helper: one-hot metadata encoding with explicit unknown flags
# ---------------------------------------------------------------------------

def _encode_metadata(row: pd.Series) -> np.ndarray:
    """
    Encode a single metadata row into a one-hot float32 vector of length
    META_DIM (15).

    Each of the three metadata fields (age, sex, anatomical site) has its
    own explicit 'unknown' indicator dimension as the final slot in its block.
    When a field is present and valid, exactly one of its content dimensions
    is set to 1.0 and its unknown dimension remains 0. When a field is missing
    or unrecognized, all content dimensions remain 0 and the unknown dimension
    is set to 1.0 instead.

    Vector layout:
        dims  0-5: age   (5 brackets + age_unknown)
        dims  6-8: sex   (female, male, sex_unknown)
        dims  9-14: site  (5 groups + site_unknown)

    Args:
        row: a single row from the metadata DataFrame, with fields
             age_approx, sex, and anatom_site_general.

    Returns:
        np.ndarray of shape (META_DIM,) with dtype float32.
    """
    vec = np.zeros(META_DIM, dtype=np.float32)

    # --- age (dims 0-5) ---
    # Try to match the age value to one of the 5 brackets. If the value is
    # present and valid, set that bracket's dimension. If it is missing or
    # cannot be parsed, set the age_unknown dimension (dim 5) instead.
    age_raw = row.get("age_approx", np.nan)
    if not pd.isna(age_raw):
        age = float(age_raw)
        matched = False
        for i, (low, high) in enumerate(AGE_BRACKETS):
            if low <= age < high:
                vec[AGE_OFFSET + i] = 1.0
                matched = True
                break
        if not matched:
            # Age value present but fell outside all defined brackets
            vec[AGE_OFFSET + AGE_UNKNOWN_DIM] = 1.0
    else:
        # age_approx is NaN — explicitly flag as unknown
        vec[AGE_OFFSET + AGE_UNKNOWN_DIM] = 1.0

    # --- sex (dims 6-8) ---
    # Try to match the sex string to [female, male]. If present and
    # recognized, set that dimension. Otherwise set sex_unknown (dim 8).
    sex_raw = str(row.get("sex", "")).strip().lower()
    if sex_raw in SEX_CATEGORIES:
        vec[SEX_OFFSET + SEX_CATEGORIES.index(sex_raw)] = 1.0
    else:
        # Missing, empty, or unrecognised sex value — explicitly flag as unknown
        vec[SEX_OFFSET + SEX_UNKNOWN_DIM] = 1.0

    # --- anatomical site (dims 9-14) ---
    # Map the raw ISIC site string to one of 5 broader site groups. If
    # recognized, set that group's dimension. Otherwise set site_unknown (dim 14).
    site_raw = str(row.get("anatom_site_general", "")).strip().lower()
    group = SITE_RAW_TO_GROUP.get(site_raw, None)
    if group is not None:
        vec[SITE_OFFSET + SITE_GROUP_NAMES.index(group)] = 1.0
    else:
        # Missing, empty, or unrecognised site value — explicitly flag as unknown
        vec[SITE_OFFSET + SITE_UNKNOWN_DIM] = 1.0

    return vec


# ---------------------------------------------------------------------------
# ISICSkinDataset
# ---------------------------------------------------------------------------

class ISICSkinDataset(Dataset):
    """
    PyTorch Dataset for ISIC 2019 skin lesion images.

    Loads images, integer class labels from the ground truth CSV,
    and one-hot encoded metadata (age, sex, anatomical site) from the
    metadata CSV.

    Args:
        image_dir (str | Path):
            Directory containing downloaded .jpg images
            (e.g. "data/train/images/").
        ground_truth_csv (str | Path):
            Path to the ground truth CSV with one-hot class columns
            (e.g. "data/train/ISIC_2019_Training_GroundTruth.csv").
            The UNK column, if present, is automatically ignored.
        metadata_csv (str | Path | None):
            Path to the metadata CSV (uses 'isic_id' column as the image
            key). If None, meta_data will be returned as None from
            __getitem__, which is incompatible with use_metablock=True
            in SkinEffnetB4.
        transform (callable | None):
            Torchvision transform applied to each PIL image. Defaults to
            DEFAULT_TRAIN_TRANSFORM when train=True, DEFAULT_TEST_TRANSFORM
            otherwise. Pass a custom transform to override.
        train (bool):
            Controls which default transform is used when transform=None.
            Has no effect if transform is provided explicitly.

    Returns per __getitem__:
        image: FloatTensor [3, 456, 456]
        label: int in [0, 7] — integer class index
        meta_data: FloatTensor [META_DIM=15] or None
        image_id: str — e.g. "ISIC_0024306"
    """

    def __init__(
        self,
        image_dir: str | Path,
        ground_truth_csv: str | Path,
        metadata_csv: str | Path | None = None,
        transform=None,
        train: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.train = train

        # --- transform ---
        if transform is not None:
            self.transform = transform
        else:
            self.transform = DEFAULT_TRAIN_TRANSFORM if train else DEFAULT_TEST_TRANSFORM

        # --- ground truth labels ---
        gt_df = pd.read_csv(ground_truth_csv)
        gt_df.columns = gt_df.columns.str.strip()

        # Validate that all 8 class columns are present. UNK is intentionally
        # excluded from CLASS_NAMES and will be ignored if present.
        missing_cols = [c for c in CLASS_NAMES if c not in gt_df.columns]
        if missing_cols:
            raise ValueError(
                f"Ground truth CSV is missing expected class columns: {missing_cols}. "
                f"Found columns: {list(gt_df.columns)}"
            )
        if "image" not in gt_df.columns:
            raise ValueError("Ground truth CSV must have an 'image' column.")

        # Drop UNK rows before label assignment
        valid_mask = gt_df[CLASS_NAMES].sum(axis=1) == 1
        n_unk = (~valid_mask).sum()
        if n_unk > 0:
            warnings.warn(f"Dropping {n_unk} rows with no valid class label (UNK).")
        gt_df = gt_df[valid_mask]

        # Strip '_downsampled' suffix from image IDs in the ground truth CSV if
        # present, so they match the actual image filenames and the
        # isic_id values in the metadata CSV.
        gt_df["image"] = gt_df["image"].str.replace("_downsampled", "", regex=False)

        # Collapse one-hot class columns to a single integer index (0-7).
        # argmax over CLASS_NAMES only — UNK column is never included here.
        gt_df["label"] = gt_df[CLASS_NAMES].values.argmax(axis=1)
        gt_df = gt_df[["image", "label"]].set_index("image")

        # --- metadata ---
        self.use_metadata = metadata_csv is not None
        if self.use_metadata:
            meta_df = pd.read_csv(metadata_csv)
            meta_df.columns = meta_df.columns.str.strip()
            # The metadata CSV uses 'isic_id' as its image identifier column,
            # which corresponds to the 'image' column in the ground truth CSV.
            if "isic_id" not in meta_df.columns:
                raise ValueError(
                    "Metadata CSV must have an 'isic_id' column. "
                    f"Found columns: {list(meta_df.columns)}"
                )
            meta_df = meta_df.set_index("isic_id")
        else:
            warnings.warn(
                "metadata_csv is None. meta_data will be returned as None from "
                "__getitem__. This is incompatible with use_metablock=True in SkinEffnetB4."
            )
            meta_df = None

        # --- build sample list, validating each image exists ---
        # self.samples is the single source of truth used by __len__,
        # __getitem__, and EquilibrationSampler (via sample[1] for labels).
        self.samples = []  # list of (image_id: str, label: int, meta_vec: np.ndarray | None)

        images_on_disk = {p.stem for p in self.image_dir.glob("*.jpg")}
        if not images_on_disk:
            raise FileNotFoundError(
                f"No .jpg files found in image_dir: {self.image_dir}. "
                "Have you run the isic-cli download commands?"
            )

        n_missing_images   = 0
        n_missing_metadata = 0

        for image_id, row in gt_df.iterrows():
            label = int(row["label"])

            # Skip samples whose image file was not downloaded
            if image_id not in images_on_disk:
                n_missing_images += 1
                continue

            # Encode metadata. If this image has no row in the metadata CSV,
            # construct a fully-unknown vector by calling _encode_metadata with
            # an empty Series — this ensures all three unknown flags are set,
            # which is the correct representation for a completely missing sample.
            if self.use_metadata:
                if image_id in meta_df.index:
                    meta_vec = _encode_metadata(meta_df.loc[image_id])
                else:
                    n_missing_metadata += 1
                    meta_vec = _encode_metadata(pd.Series(dtype=object))
            else:
                meta_vec = None

            self.samples.append((image_id, label, meta_vec))

        if n_missing_images > 0:
            warnings.warn(
                f"{n_missing_images} image(s) listed in the ground truth CSV were "
                f"not found in {self.image_dir} and have been skipped."
            )
        if n_missing_metadata > 0:
            warnings.warn(
                f"{n_missing_metadata} image(s) had no matching row in the metadata "
                f"CSV. Fully-unknown metadata vectors were used for these samples "
                f"(all three unknown flags set to 1)."
            )

        print(
            f"ISICSkinDataset: loaded {len(self.samples)} samples "
            f"({'train' if train else 'test'}) | "
            f"metadata={'yes (dim=' + str(META_DIM) + ')' if self.use_metadata else 'no'}"
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_id, label, meta_vec = self.samples[idx]

        # Load and transform image
        img_path = self.image_dir / f"{image_id}.jpg"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Convert pre-computed metadata array to tensor
        if meta_vec is not None:
            meta_tensor = torch.tensor(meta_vec, dtype=torch.float32)
        else:
            meta_tensor = None

        return image, label, meta_tensor, image_id


# ---------------------------------------------------------------------------
# Convenience functions to get datasets
# ---------------------------------------------------------------------------

def get_train_dataset(
    image_dir: str | Path = "data/train/images",
    ground_truth_csv: str | Path = "data/train/ISIC_2019_Training_GroundTruth.csv",
    metadata_csv: str | Path | None = "data/train/challenge-2019-training_metadata.csv",
    transform=None,
) -> ISICSkinDataset:
    """Returns an ISICSkinDataset configured for training (train=True)."""
    return ISICSkinDataset(
        image_dir=image_dir,
        ground_truth_csv=ground_truth_csv,
        metadata_csv=metadata_csv,
        transform=transform,
        train=True,
    )


def get_test_dataset(
    image_dir: str | Path = "data/test/images",
    ground_truth_csv: str | Path = "data/test/ISIC_2019_Test_GroundTruth.csv",
    metadata_csv: str | Path | None = "data/test/challenge-2019-test_metadata.csv",
    transform=None,
) -> ISICSkinDataset:
    """Returns an ISICSkinDataset configured for testing (train=False)."""
    return ISICSkinDataset(
        image_dir=image_dir,
        ground_truth_csv=ground_truth_csv,
        metadata_csv=metadata_csv,
        transform=transform,
        train=False,
    )
