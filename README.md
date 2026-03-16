# Ablation Study of Equilibration Mini-Batch Sampling, Transfer Learning, and Metadata Fusion with EfficientNet-B5 for Skin Lesion Classification

**DS 6050 — Deep Learning | University of Virginia, School of Data Science | Spring 2026**

Claire Dozier, Katie Dunning, Michael Ieraci, Emma Polson

---

## Overview

This repository contains the code for a structured ablation study investigating how three techniques — **equilibration mini-batch (EM) sampling**, **ImageNet transfer learning**, and **MetaBlock metadata fusion** — interact to improve classification of rare skin lesions on the [ISIC 2019 Challenge](https://challenge.isic-archive.com/data/#2019) dataset.

The ISIC 2019 dataset presents two core challenges: severe class imbalance (the largest class comprises 50.8% of samples while the smallest is under 1%) and limited overall dataset size (~25K images). Each of the three techniques addresses a different aspect of this problem at a different level of the training pipeline:

| Technique | Level | Purpose |
|---|---|---|
| EM Sampling | Data | Balances each mini-batch so every class receives equal representation, counteracting gradient dominance by majority classes |
| Transfer Learning | Architecture (weights) | Initializes EfficientNet-B5 with ImageNet-pretrained weights to compensate for limited training data |
| MetaBlock | Architecture (features) | Uses patient metadata (age, sex, anatomical site) to modulate CNN feature maps via a learned attention mechanism |

The ablation study uses a **2 × 4 design** across 8 experiments:

| Experiment | Track 1 (Baseline) | Track 2 (Transfer Learning) |
|---|---|---|
| Foundation | EfficientNet-B5 from scratch | EfficientNet-B5 + ImageNet weights |
| + EM Sampling | EM | TL + EM |
| + MetaBlock | MetaBlock | TL + MetaBlock |
| + Both | EM + MetaBlock | TL + EM + MetaBlock |

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

### Expected directory structure

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
├── current code files/
│   ├── isic2019_dataset.py        # Dataset class: image loading, metadata encoding, train/test splits
│   ├── equilibration_sampler.py   # EM sampler: class-balanced mini-batch construction
│   ├── metablock.py               # MetaBlock: metadata-driven feature map modulation
│   ├── model.py                   # SkinEffnetB5: EfficientNet-B5 with optional MetaBlock and transfer learning
│   ├── dataloader.py              # DataLoader construction with optional EM sampling
│   ├── runner.py                  # Main training/evaluation script (CLI-driven experiment configs)
│   ├── tune_params.py             # Optuna hyperparameter optimization
│   └── data_investigation.ipynb   # Exploratory data analysis
├── .gitignore
├── LICENSE
└── README.md
```

## Module Descriptions

### `isic2019_dataset.py`
PyTorch `Dataset` that loads ISIC 2019 images with integer class labels and one-hot encoded metadata. Metadata is encoded as a 15-dimensional vector: 6 dims for age (5 brackets + unknown), 3 dims for sex (female, male, unknown), and 6 dims for anatomical site (5 grouped regions + unknown). Each metadata field has an explicit unknown flag so that missingness is a learnable signal rather than an implicit all-zeros pattern.

### `equilibration_sampler.py`
Custom PyTorch `Sampler` implementing the equilibration mini-batch strategy from [Ya-Guan et al. (2020)](https://ieeexplore.ieee.org/document/9055020). Given batch size `m` and `K=8` classes, each mini-batch contains exactly `Q = m/K` samples per class. Classes with more than `Q` available samples are undersampled (without replacement); classes with fewer are oversampled (all real samples included first, then remainder filled with replacement). Epoch length is anchored on the largest class to ensure coverage.

### `metablock.py`
Implementation of the Metadata Processing Block from [Pacheco and Krohling (2021)](https://ieeexplore.ieee.org/document/9364366). Applies learned scale (`f_b`) and shift (`g_b`) transformations — conditioned on patient metadata — to CNN feature map groups using the gating equation: `x̃ = σ[tanh(f_b(x_meta) ⊙ x_img) + g_b(x_meta)]`.

### `model.py`
Wraps EfficientNet-B5 with configurable transfer learning (ImageNet pretrained vs. random init), optional MetaBlock integration, and a replaceable classification head. The original classifier is removed and replaced with a linear layer for 8-class output. When MetaBlock is active, the 2,048-channel feature maps are reshaped into 32 groups before metadata modulation.

### `dataloader.py`
Constructs train/validation `DataLoader` pairs from an 80/20 index split. Handles the toggle between standard random sampling and EM sampling. The validation split always uses the test transform (resize + normalize, no augmentation).

### `runner.py`
Main entry point for running experiments. Uses CLI flags to configure experiment conditions:

```bash
# Examples:
python runner.py -c TL            # Transfer learning only
python runner.py -c TL_EM         # Transfer learning + EM sampling
python runner.py -c TL_META       # Transfer learning + MetaBlock
python runner.py -c TL_EM_META    # Transfer learning + EM + MetaBlock
python runner.py -c EM            # Baseline + EM sampling
python runner.py -c META          # Baseline + MetaBlock
python runner.py -c EM_META       # Baseline + EM + MetaBlock
python runner.py -c SCRATCH       # Baseline only (no flags)
```

Loads best hyperparameters from a pre-computed Optuna SQLite database, trains with early stopping (patience=5 on validation macro recall), and logs all metrics to Weights & Biases.

### `tune_params.py`
Optuna hyperparameter search over learning rate, weight decay, batch size (multiples of 8), and LR scheduler (cosine vs. step). Runs 50 trials with median pruning after 5 warmup epochs. Saves results to SQLite for retrieval by `runner.py`. Two separate studies are maintained: one for transfer learning conditions, one for scratch conditions.

## Environment and Dependencies

This project runs on UVA's Rivanna HPC cluster using GPU nodes. Key dependencies:

- Python 3.10+
- PyTorch 2.x (with CUDA)
- torchvision
- optuna
- wandb (Weights & Biases)
- scikit-learn
- pandas, numpy, Pillow

## Evaluation Metrics

The primary evaluation metric is **macro recall** (equivalent to balanced accuracy), which weights all 8 classes equally regardless of sample count. This follows the ISIC 2019 challenge convention. Per-class AUC (one-vs-rest) is tracked as a secondary metric to monitor performance on individual classes, particularly the rare ones (VASC, DF, SCC).

All training and evaluation metrics are logged to [Weights & Biases](https://wandb.ai/) under the project `ds6050-g03-ISIC2019`.

## References

- Ya-Guan, Q. et al. (2020). "EMSGD: An Improved Learning Algorithm of Neural Networks With Imbalanced Data." *IEEE Access*, 8, 64086–64098.
- Pacheco, A. G. & Krohling, R. A. (2021). "An Attention-Based Mechanism to Combine Images and Metadata in Deep Learning Models Applied to Skin Cancer Classification." *IEEE JBHI*, 25(9), 3554–3563.
- Hasan, S. M. et al. (2023). "Enhancing Multi-Class Skin Lesion Classification with Modified EfficientNets." *ICICT4SD*, 94–98.
- Pan, S. J. & Yang, Q. (2010). "A Survey on Transfer Learning." *IEEE TKDE*, 22(10), 1345–1359.
- Tan, M. & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML*.
- Kassem, M. A. et al. (2020). "Skin Lesions Classification Into Eight Classes for ISIC 2019 Using Deep Convolutional Neural Network and Transfer Learning." *IEEE Access*, 8, 114822–114832.
- Pham, T. C. et al. (2020). "Improving Skin-Disease Classification Based on Customized Loss Function Combined With Balanced Mini-Batch Logic and Real-Time Image Augmentation." *IEEE Access*, 8, 150725–150737.
- Liu, Y. et al. (2017). "Detecting Cancer Metastases on Gigapixel Pathology Images." *arXiv:1703.02442*.
- Unnisa, Z. et al. (2025). "Impact of Fine-Tuning Parameters of Convolutional Neural Network for Skin Cancer Detection." *Scientific Reports*, 15(1), 14779.

## License

This project is released under the [MIT License](LICENSE).
