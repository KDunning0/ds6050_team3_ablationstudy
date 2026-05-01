# Ablation Study of Equilibration Mini-Batch Sampling, Transfer Learning, and Metadata Fusion with EfficientNet-B4 for Skin Lesion Classification

**DS 6050 - Deep Learning | University of Virginia, School of Data Science | Spring 2026**

Claire Dozier, Katie Dunning, Michael Ieraci, Emma Polson

---

## Overview

This repository contains the code for a structured ablation study on the [ISIC 2019 Challenge](https://challenge.isic-archive.com/data/#2019) skin lesion classification dataset. The project evaluates how three techniques interact when training an EfficientNet-B4 classifier under severe class imbalance:

| Technique | Level | Purpose |
|---|---|---|
| Equilibration mini-batch sampling (EM) | Data loading | Constructs class-balanced mini-batches so rare classes contribute consistently to gradient updates |
| ImageNet transfer learning (TL) | Model initialization | Starts EfficientNet-B4 from pretrained ImageNet weights instead of random initialization |
| MetaBlock metadata fusion | Model architecture | Uses patient metadata to scale and shift CNN feature groups before classification |

The primary metric is **macro-averaged recall (MAR)**, which weights each lesion class equally and is better aligned with rare-class performance than overall accuracy.

## Study Design

The final study uses a full 2 x 2 x 2 ablation grid:

| Condition | Transfer learning | EM sampling | MetaBlock |
|---|---:|---:|---:|
| `SCRATCH` | No | No | No |
| `SCRATCH_EM` | No | Yes | No |
| `SCRATCH_META` | No | No | Yes |
| `SCRATCH_EM_META` | No | Yes | Yes |
| `TL` | Yes | No | No |
| `TL_EM` | Yes | Yes | No |
| `TL_META` | Yes | No | Yes |
| `TL_EM_META` | Yes | Yes | Yes |

The team also ran a supplementary `TL_INV` condition using inverse-frequency-weighted cross entropy.

### Architecture Note

The original project proposal considered EfficientNet-B5. During implementation, B5 exceeded available GPU memory for the planned batch sizes on Rivanna A40 GPUs, so the final experiments use **EfficientNet-B4** with 380 x 380 images. EfficientNet-B4 provides a smaller backbone while preserving strong performance for skin lesion classification.

## Final Project Notes

- The model wrapper is `SkinEffnetB4`.
- The EfficientNet-B4 backbone outputs 1,792 convolutional features.
- MetaBlock reshapes those features into 32 metadata-controlled feature groups.
- Metadata is represented as a 15-dimensional vector covering age, sex, and anatomical site, including explicit unknown categories.
- Transfer-learning runs use dropout `0.0`; scratch runs use dropout `0.5`.
- Experiments train for up to 30 epochs with early stopping on validation MAR.
- Best checkpoints are saved as `{condition}_best_weights.pth`.
- Metrics and confusion matrices are logged to Weights & Biases under the `ds6050-g03-ISIC2019-Experiments` project.

The final report contains the complete results table, interaction analysis, and MetaBlock coefficient visualization. The MetaBlock visualization inspects the learned scale (`f_b`) and shift (`g_b`) coefficients across feature-map groups for the four MetaBlock conditions.

## Dataset Setup

The ISIC 2019 image files and CSV metadata are not stored in this repository. Download them before running the training code.

### 1. Install the ISIC CLI

Download the ISIC command-line tool from:

https://github.com/ImageMarkup/isic-cli/releases/latest

### 2. Download Images

```bash
# Training images: collection 65, about 25K images
isic image download --collections 65 data/train/images/

# Test images: collection 72
isic image download --collections 72 data/test/images/
```

### 3. Download CSV Files

Download these files from the ISIC 2019 challenge page and ISIC Archive API:

| File | Save to | Source |
|---|---|---|
| `ISIC_2019_Training_GroundTruth.csv` | `data/train/` | ISIC 2019 challenge data page |
| `ISIC_2019_Test_GroundTruth.csv` | `data/test/` | ISIC 2019 challenge data page |
| `challenge-2019-training_metadata.csv` | `data/train/` | Collection 65 metadata |
| `challenge-2019-test_metadata.csv` | `data/test/` | Collection 72 metadata |

Expected layout:

```text
data/
├── train/
│   ├── images/
│   ├── ISIC_2019_Training_GroundTruth.csv
│   └── challenge-2019-training_metadata.csv
└── test/
    ├── images/
    ├── ISIC_2019_Test_GroundTruth.csv
    └── challenge-2019-test_metadata.csv
```

The test set includes `UNK` images. These are excluded during loading so evaluation is performed on the eight known lesion classes.

## Repository Structure

```text
ds6050_team3_ablationstudy/
├── Code/
│   ├── isic2019_dataset.py
│   ├── equilibration_sampler.py
│   ├── metablock.py
│   ├── model.py
│   ├── dataloader.py
│   ├── runner.py
│   ├── tune_params-4workers-dropout.py
│   ├── check_model_size.py
│   ├── data_investigation_v2.ipynb
│   ├── EM Batch Exploration/
│   ├── Model Size Info/
│   └── Optuna Database Files/
├── LICENSE
└── README.md
```

## Code Modules

### `isic2019_dataset.py`

Defines the PyTorch dataset for ISIC 2019. It loads image paths, maps ground-truth labels to integer classes, encodes metadata, applies train/test transforms, and excludes unknown-class test images.

### `equilibration_sampler.py`

Implements equilibration mini-batch sampling. For an 8-class batch, each mini-batch receives equal class representation where possible, with undersampling for majority classes and replacement sampling for minority classes.

### `metablock.py`

Implements the Metadata Processing Block. The block contains two metadata-conditioned branches:

- `scale_branch`: learned scale coefficients `f_b`
- `shift_branch`: learned shift coefficients `g_b`

Each branch maps the 15 metadata features to 32 feature groups using `Linear(15 -> 32)` followed by `BatchNorm1d(32)`.

### `model.py`

Defines `SkinEffnetB4`, an EfficientNet-B4 model with optional transfer learning, optional feature extraction, optional MetaBlock fusion, and configurable dropout. When MetaBlock is active, the 1,792-channel EfficientNet feature tensor is reshaped into 32 groups before metadata-conditioned modulation.

### `dataloader.py`

Builds train and validation data loaders using either standard shuffled sampling or EM sampling. It uses a reproducible train/validation split and applies the appropriate image transforms.

### `runner.py`

Main experiment runner. It parses the condition name, loads the matching Optuna study, builds the model and data loaders, trains with early stopping, evaluates on the test set, saves best weights, and logs metrics to Weights & Biases.

### `tune_params-4workers-dropout.py`

Runs Optuna hyperparameter tuning for the scratch and transfer-learning tracks. The search includes learning rate, weight decay, batch size, and scheduler choice.

## Running Experiments

Run commands from inside the `Code/` directory or adjust paths accordingly.

```bash
cd Code
```

Scratch track:

```bash
python runner.py -c SCRATCH
python runner.py -c SCRATCH_EM
python runner.py -c SCRATCH_META
python runner.py -c SCRATCH_EM_META
```

Transfer-learning track:

```bash
python runner.py -c TL
python runner.py -c TL_EM
python runner.py -c TL_META
python runner.py -c TL_EM_META
```

The runner determines active features by checking the condition string:

- `TL` enables ImageNet transfer learning.
- `EM` enables equilibration mini-batch sampling.
- `META` enables MetaBlock metadata fusion.
- `FEAT` enables frozen-backbone feature extraction if included.

## Optuna Databases

The runner expects Optuna SQLite databases to be available in the working directory used for execution:

| Track | Study name | Database |
|---|---|---|
| Transfer learning | `TL` | `optuna_TL.db` |
| Scratch | `SCRATCH_DO_v2` | `optuna_SCRATCH_dropout_v2.db` |

The repository includes the tuning databases under `Code/Optuna Database Files/`. Copy or symlink the needed `.db` files into `Code/` before running `runner.py`, or update the database paths in the runner.

## Running on Rivanna

The experiments were designed for UVA's Rivanna HPC cluster. GPU runs should request a CUDA-capable GPU and load a PyTorch environment with `torchvision`, `optuna`, `wandb`, `scikit-learn`, `pandas`, `numpy`, and `Pillow`.

Example Slurm GPU directives:

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
```

Because the final model uses EfficientNet-B4, it is lighter than the originally proposed B5 model, but full training is still intended for GPU execution. Weight-only MetaBlock coefficient visualization can be run locally because it only reads checkpoint state dictionaries.

## Evaluation

The primary evaluation metric is macro-averaged recall:

```text
MAR = mean(recall_1, recall_2, ..., recall_8)
```

This gives each lesion class equal weight, which is important because ISIC 2019 is highly imbalanced. Per-class AUC and confusion matrices are tracked as secondary diagnostics.

## References

- Ya-Guan, Q. et al. (2020). "EMSGD: An Improved Learning Algorithm of Neural Networks With Imbalanced Data." IEEE Access.
- Pacheco, A. G. and Krohling, R. A. (2021). "An Attention-Based Mechanism to Combine Images and Metadata in Deep Learning Models Applied to Skin Cancer Classification." IEEE Journal of Biomedical and Health Informatics.
- Tan, M. and Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML.
- Pham, T. C. et al. (2020). "Improving Skin-Disease Classification Based on Customized Loss Function Combined With Balanced Mini-Batch Logic and Real-Time Image Augmentation." IEEE Access.
- Kassem, M. A. et al. (2020). "Skin Lesions Classification Into Eight Classes for ISIC 2019 Using Deep Convolutional Neural Network and Transfer Learning." IEEE Access.

## License

This project is released under the [MIT License](LICENSE).
