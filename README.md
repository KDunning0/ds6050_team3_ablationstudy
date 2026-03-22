# Ablation Study of Equilibration Mini-Batch Sampling, Transfer Learning, and Metadata Fusion with EfficientNet-B4 for Skin Lesion Classification

**DS 6050 — Deep Learning | University of Virginia, School of Data Science | Spring 2026**

Claire Dozier, Katie Dunning, Michael Ieraci, Emma Polson

---

## Overview

This repository contains the code for a structured ablation study investigating how three techniques — **equilibration mini-batch (EM) sampling**, **ImageNet transfer learning**, and **MetaBlock metadata fusion** — interact to improve classification of rare skin lesions on the [ISIC 2019 Challenge](https://challenge.isic-archive.com/data/#2019) dataset.

The ISIC 2019 dataset presents two core challenges: severe class imbalance (the largest class comprises 50.8% of samples while the smallest is under 1%) and limited overall dataset size (~25K images). Each of the three techniques addresses a different aspect of this problem at a different level of the training pipeline:

| Technique | Level | Purpose |
|---|---|---|
| EM Sampling | Data | Balances each mini-batch so every class receives equal representation, counteracting gradient dominance by majority classes |
| Transfer Learning | Architecture (weights) | Initializes EfficientNet-B4 with ImageNet-pretrained weights to compensate for limited training data |
| MetaBlock | Architecture (features) | Uses patient metadata (age, sex, anatomical site) to modulate CNN feature maps via a learned attention mechanism |

The ablation study uses a **2 × 4 design** across 8 experiments:

| Experiment | Track 1 (Baseline) | Track 2 (Transfer Learning) |
|---|---|---|
| Foundation | EfficientNet-B4 from scratch | EfficientNet-B4 + ImageNet weights |
| + EM Sampling | EM | TL + EM |
| + MetaBlock | MetaBlock | TL + MetaBlock |
| + Both | EM + MetaBlock | TL + EM + MetaBlock |

> **Note on architecture choice:** The original proposal specified EfficientNet-B5 (30M parameters, 456×456 input). During development, higher batch sizes caused out-of-memory errors on Rivanna A40 GPUs, so the architecture was changed to EfficientNet-B4 (19M parameters, 380×380 input). B4 has strong literature support for skin lesion classification (Pham et al., 2020) and the reduced parameter count is better suited to the 25K-image dataset.

## Baseline Results (Milestone 2)

Two baseline models have been trained and evaluated on the ISIC 2019 test set (UNK class excluded):

| Model | Val MAR | Test MAR | Early Stopping |
|---|---|---|---|
| B4 Scratch (dropout=0.5) | 0.515 | 0.377 | Not triggered (30 epochs) |
| B4 Transfer Learning | 0.784 | 0.549 | Epoch 22 (best weights from epoch 12) |

Transfer learning improves test macro recall by 45% over the scratch baseline. The scratch model exhibits severe overfitting (train loss → 0.1, val loss → 1.4), while the TL model shows moderate overfitting mitigated by pretrained weight stability. Per-class recall improvements from TL are most dramatic for rare classes: DF (0.15 → 0.40), BKL (0.24 → 0.51), SCC (0.24 → 0.43), VASC (0.23 → 0.39).

Detailed training curves, confusion matrices, and per-class AUC tables are logged to Weights & Biases.

## Dataset Setup

### 1. Install the ISIC CLI tool

Download from: https://github.com/ImageMarkup/isic-cli/releases/latest

### 2. Download images

```bash
# Training images (collection 65) — ~9.1 GB, 25,331 images
isic image download --collections 65 data/train/images/

# Test images (collection 72)
isic image download --collections 72 data/test/images/
```

### 3. Download CSV files

Download the following files manually from https://challenge.isic-archive.com/data/#2019 and the ISIC Archive API:

| File | Save to | Source |
|---|---|---|
| `ISIC_2019_Training_GroundTruth.csv` | `data/train/` | Challenge page |
| `ISIC_2019_Test_GroundTruth.csv` | `data/test/` | Challenge page |
| `challenge-2019-training_metadata.csv` | `data/train/` | [Collection 65 API](https://api.isic-archive.com/collections/65/) → Actions → Download Metadata |
| `challenge-2019-test_metadata.csv` | `data/test/` | [Collection 72 API](https://api.isic-archive.com/collections/72/) → Actions → Download Metadata |

### Expected data directory structure

```
data/
├── train/
│   ├── images/                                    # 25,331 .jpg files
│   ├── ISIC_2019_Training_GroundTruth.csv
│   └── challenge-2019-training_metadata.csv
└── test/
    ├── images/                                    # 8,238 .jpg files
    ├── ISIC_2019_Test_GroundTruth.csv
    └── challenge-2019-test_metadata.csv
```

> **Note:** The test set contains 2,047 UNK (unknown class) images that are automatically excluded during loading, leaving 6,191 test images across the 8 training classes.

## Repository Structure

```
ds6050_team3_ablationstudy/
├── Code/
│   ├── isic2019_dataset.py              # Dataset class: image loading, metadata encoding, train/test splits
│   ├── equilibration_sampler.py         # EM sampler: class-balanced mini-batch construction
│   ├── metablock.py                     # MetaBlock: metadata-driven feature map modulation
│   ├── model.py                         # SkinEffnetB4: EfficientNet-B4 with optional MetaBlock and transfer learning
│   ├── dataloader.py                    # DataLoader construction with optional EM sampling
│   ├── runner.py                        # Main training/evaluation script (CLI-driven experiment configs)
│   ├── tune_params-4workers-dropout.py  # Optuna hyperparameter optimization (with AMP mixed precision)
│   ├── check_model_size.py              # Parameter count summary for all model variants
│   ├── data_investigation_v2.ipynb      # Exploratory data analysis and metadata completeness
│   ├── EM Batch Exploration/            # EM sampler verification tests and analysis
│   ├── Model Size Info/                 # Model parameter summaries
│   └── Optuna Database Files/           # Optuna SQLite databases from hyperparameter tuning
│       ├── optuna_TL.db
│       ├── optuna_SCRATCH.db
│       └── optuna_SCRATCH_dropout_v2.db
├── .gitignore
├── LICENSE
└── README.md
```

## Module Descriptions

### `isic2019_dataset.py`
PyTorch `Dataset` that loads ISIC 2019 images with integer class labels and one-hot encoded metadata. Metadata is encoded as a 15-dimensional vector: 6 dims for age (5 brackets + unknown), 3 dims for sex (female, male, unknown), and 6 dims for anatomical site (5 grouped regions + unknown). Each metadata field has an explicit unknown flag so that missingness is a learnable signal rather than an implicit all-zeros pattern. Images are resized to 380×380 for EfficientNet-B4 and normalized using ImageNet statistics.

### `equilibration_sampler.py`
Custom PyTorch `Sampler` implementing the equilibration mini-batch strategy from [Ya-Guan et al. (2020)](https://ieeexplore.ieee.org/document/9055020). Given batch size `m` and `K=8` classes, each mini-batch contains exactly `Q = m/K` samples per class. Classes with more than `Q` available samples are undersampled (without replacement); classes with fewer are oversampled (all real samples included first, then remainder filled with replacement). Epoch length is anchored on the largest class to ensure coverage. Includes a `class_summary()` diagnostic and batch-level logging utility for verification.

### `metablock.py`
Implementation of the Metadata Processing Block from [Pacheco and Krohling (2021)](https://ieeexplore.ieee.org/document/9364366). Applies learned scale (`f_b`) and shift (`g_b`) transformations — conditioned on patient metadata — to CNN feature map groups using the gating equation: `x̃ = σ[tanh(f_b(x_meta) ⊙ x_img) + g_b(x_meta)]`. Each branch is a single linear layer with batch normalization.

### `model.py`
Wraps EfficientNet-B4 with configurable modes:
- `pretrained=True/False`: ImageNet weights vs. random initialization
- `feature_extract=True/False`: frozen backbone vs. full fine-tuning
- `use_metablock=True/False`: with or without MetaBlock metadata fusion
- `dropout_p`: dropout rate on the final FC layer (0.0 for TL, 0.5 for scratch)

The backbone outputs 1,792 channels which are reshaped into 32 groups of 56 channels each when MetaBlock is active. The original classifier is replaced with a linear layer for 8-class output.

### `dataloader.py`
Constructs train/validation `DataLoader` pairs from an 80/20 index split. Handles the toggle between standard random sampling and EM sampling via the `use_equilibration` flag. The validation split uses the test transform (resize + normalize only). Uses `drop_last=True` to ensure consistent batch sizes.

### `runner.py`
Main entry point for running experiments. Uses CLI flags to configure experiment conditions:

```bash
# Track 1 (scratch) experiments:
python runner.py -c SCRATCH           # Baseline from scratch
python runner.py -c EM                # Scratch + EM sampling
python runner.py -c META              # Scratch + MetaBlock
python runner.py -c EM_META           # Scratch + EM + MetaBlock

# Track 2 (transfer learning) experiments:
python runner.py -c TL                # Transfer learning baseline
python runner.py -c TL_EM             # TL + EM sampling
python runner.py -c TL_META           # TL + MetaBlock
python runner.py -c TL_EM_META        # TL + EM + MetaBlock
```

Loads best hyperparameters from pre-computed Optuna SQLite databases (TL track from `optuna_TL.db`, scratch track from `optuna_SCRATCH_dropout_v2.db`). Trains with early stopping (patience=10 on validation macro recall), saves best model weights to disk, and logs all metrics including a test-set confusion matrix to Weights & Biases. Scratch models use dropout=0.5 on the final FC layer; TL models use no dropout.

### `tune_params-4workers-dropout.py`
Optuna hyperparameter search over learning rate (1e-5 to 1e-2, log scale), weight decay (1e-5 to 1e-2, log scale), batch size ({16, 24, 32, 48, 64}), and LR scheduler (cosine vs. step with step_size=10). Runs 40 trials with median pruning after 5 warmup epochs. Uses mixed precision training (AMP) with `GradScaler` to reduce VRAM usage. Includes memory management (explicit `gc.collect()` and `torch.cuda.empty_cache()` between trials) and catches OOM errors gracefully. Scratch models are tuned with dropout=0.5; TL models with dropout=0.0.

### `check_model_size.py`
Generates a detailed parameter summary for all four model variants (TL, scratch, TL+MetaBlock, scratch+MetaBlock), including per-layer output shapes, parameter counts, and trainable parameter counts. Output is saved to `model_parameter_summary.txt`.

## Running on Rivanna

This project runs on UVA's Rivanna HPC cluster. Request A100 GPUs for reliable execution:

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
```

### Execution order

1. **Hyperparameter tuning** (once per track):
   ```bash
   python tune_params-4workers-dropout.py -c TL
   python tune_params-4workers-dropout.py -c SCRATCH
   ```

2. **Baseline training** (Milestone 2):
   ```bash
   python runner.py -c TL
   python runner.py -c SCRATCH
   ```

3. **Ablation experiments** (Milestone 3):
   ```bash
   python runner.py -c TL_EM
   python runner.py -c TL_META
   python runner.py -c TL_EM_META
   python runner.py -c EM
   python runner.py -c META
   python runner.py -c EM_META
   ```

## Dependencies

- Python 3.10+
- PyTorch 2.x (with CUDA)
- torchvision
- optuna
- wandb (Weights & Biases)
- scikit-learn
- pandas, numpy, Pillow

## Evaluation Metrics

The primary evaluation metric is **macro-averaged recall (MAR)**, equivalent to balanced accuracy, which weights all 8 classes equally regardless of sample count. This follows the ISIC 2019 challenge convention. Per-class AUC (one-vs-rest) is tracked as a secondary metric to monitor performance on individual classes, particularly the rare ones (VASC, DF, SCC).

All training and evaluation metrics are logged to [Weights & Biases](https://wandb.ai/) under the `ds6050_team3` entity.

## References

- Ya-Guan, Q. et al. (2020). "EMSGD: An Improved Learning Algorithm of Neural Networks With Imbalanced Data." *IEEE Access*, 8, 64086–64098.
- Pacheco, A. G. & Krohling, R. A. (2021). "An Attention-Based Mechanism to Combine Images and Metadata in Deep Learning Models Applied to Skin Cancer Classification." *IEEE JBHI*, 25(9), 3554–3563.
- Hasan, S. M. et al. (2023). "Enhancing Multi-Class Skin Lesion Classification with Modified EfficientNets." *ICICT4SD*, 94–98.
- Pan, S. J. & Yang, Q. (2010). "A Survey on Transfer Learning." *IEEE TKDE*, 22(10), 1345–1359.
- Tan, M. & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML*.
- Pham, T. C. et al. (2020). "Improving Skin-Disease Classification Based on Customized Loss Function Combined With Balanced Mini-Batch Logic and Real-Time Image Augmentation." *IEEE Access*, 8, 150725–150737.
- Kassem, M. A. et al. (2020). "Skin Lesions Classification Into Eight Classes for ISIC 2019 Using Deep Convolutional Neural Network and Transfer Learning." *IEEE Access*, 8, 114822–114832.
- Liu, Y. et al. (2017). "Detecting Cancer Metastases on Gigapixel Pathology Images." *arXiv:1703.02442*.
- Unnisa, Z. et al. (2025). "Impact of Fine-Tuning Parameters of Convolutional Neural Network for Skin Cancer Detection." *Scientific Reports*, 15(1), 14779.
- Guo, C. et al. (2017). "On Calibration of Modern Neural Networks." *ICML*.

## License

This project is released under the [MIT License](LICENSE).
