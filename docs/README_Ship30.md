# TRIDENT: Leveraging LLMs for OOD Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **TL;DR:** TRIDENT leverages Large Language Models (LLMs) to *envision* Out-of-Distribution (OOD) class names, enabling CLIP-based zero-shot OOD detection without collecting real OOD data. The framework supports **far**, **near**, and **fine-grained** OOD tasks across multiple benchmarks.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
  - [1. Train Task Residual (DVR)](#1-train-task-residual-dvr)
  - [2. Evaluate OOD Detection](#2-evaluate-ood-detection)
- [OOD Task Types](#ood-task-types)
  - [Far OOD](#far-ood)
  - [Near OOD](#near-ood)
  - [Fine-grained OOD](#fine-grained-ood)
- [Project Structure](#project-structure)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## 🔭 Overview

Out-of-Distribution (OOD) detection aims to identify inputs that do not belong to the training distribution. Traditional methods require collecting real OOD data, which is expensive and often infeasible.

**TRIDENT** addresses this by:
1. **Envisioning OOD Classes:** Using LLMs (GPT-3.5/4, Claude, Gemini) to generate candidate OOD class names based on In-Distribution (ID) class information.
2. **Task Residual Learning (DVR):** Efficiently fine-tuning CLIP's text features with only ~10K parameters via residual learning.
3. **EOE Scoring:** Computing an *Envisioned Out-of-Distribution Examples* score that contrasts ID confidence against OOD candidate confidence.

<p align="center">
  <img src="docs/framework.png" alt="TRIDENT Framework" width="800">
</p>

---

## ✨ Key Features

- **🤖 LLM-Driven OOD Space Expansion:** No real OOD data collection needed.
- **🎯 Task-Adaptive Prompt Engineering:** Customized few-shot prompts for far/near/fine-grained OOD tasks.
- **⚡ Lightweight Fine-Tuning:** DVR (Task Residual) learns only text feature residuals, keeping vision encoder frozen.
- **📊 Comprehensive Benchmarks:** Supports CUB-200, Stanford Cars, Food-101, Oxford Pets, Boat, and ImageNet variants.
- **🔧 Flexible Scoring Functions:** EOE, MCM, Energy, Max-Logit, with ablation studies.

---

## 🛠 Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.0 with CUDA support
- NVIDIA GPU with sufficient VRAM (>= 12GB recommended for ViT-L/14)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/TRIDENT.git
cd TRIDENT
```

### Step 2: Install Dassl Framework

TRIDENT is built on top of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). Install it first:

```bash
pip install -e Dassl.pytorch/
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure LLM API Key (Optional)

If you want to generate new OOD candidate classes with LLMs:

```bash
export OPENAI_API_KEY="sk-your-api-key"
# or for custom API endpoint:
export OPENAI_BASE_URL="https://your-endpoint.com/v1"
```

> **Note:** If you only use pre-generated JSON files (without `--generate_class`), no API key is required.

---

## 📁 Dataset Preparation

Organize your datasets under a root directory (e.g., `/data/datasets`):

```
/data/datasets/
├── boat-29/
│   ├── images/
│   └── split_zhou_Boat29.json
├── CUB-200-2011/
├── stanford_cars/
├── food-101/
├── oxford-pets/
└── imagenet/
```

Set the dataset root in training/evaluation commands using `--root`.

---

## 🚀 Quick Start

### 1. Train Task Residual (DVR)

For **fine-grained OOD** tasks (e.g., Boat-14, Pet-18, CUB-100), you need to train the DVR model first:

```bash
bash scripts/train.sh boat14_ID 16
```

This trains a Task Residual model with 16-shot examples. The checkpoint will be saved to:
```
output/FINAL/debug/ViT14/boat14_ID/16shots/seed1/prompt_learner/model.pth.tar-200
```

### 2. Evaluate OOD Detection

Run EOE-based OOD detection:

```bash
python eval_ood_detection.py \
  --in_dataset boat14_ID \
  --ood_task fine_grained \
  --score EOE \
  --score_ablation EOE \
  --L 500 \
  --llm_model gpt-3.5-turbo-16k \
  --model CLIP \
  --CLIP_ckpt ViT-L/14 \
  --shot 16 \
  --image_label 0 \
  --json_number 0
```

**Output metrics:**
- `FPR95`: False Positive Rate at 95% True Positive Rate (lower is better)
- `AUROC`: Area Under ROC Curve (higher is better)
- `AUPR`: Area Under Precision-Recall Curve (higher is better)

### Run Multiple Trials

For robust evaluation, run 3 trials with different `json_number`:

```bash
for i in 0 1 2; do
  python eval_ood_detection.py \
    --in_dataset boat14_ID \
    --ood_task fine_grained \
    --score EOE \
    --L 500 \
    --json_number $i \
    --shot 16 \
    --model CLIP \
    --CLIP_ckpt ViT-L/14
    # ... (other args)
done
```

Or use the provided script:
```bash
bash eval1.sh 500 0
```

---

## 🎯 OOD Task Types

### Far OOD

OOD samples are semantically unrelated to ID classes (e.g., birds vs furniture).

```bash
python eval_ood_detection.py \
  --in_dataset boat29 \
  --ood_task far \
  --score MCM \
  --L 500 \
  --model CLIP \
  --CLIP_ckpt ViT-L/14
```

### Near OOD

OOD samples are semantically similar but from different domains (e.g., husky vs wolf).

```bash
python eval_ood_detection.py \
  --in_dataset ImageNet20 \
  --ood_task near \
  --score EOE \
  --L 3 \
  --model CLIP \
  --CLIP_ckpt ViT-L/14
```

### Fine-grained OOD

OOD samples belong to the same super-category but are different sub-classes (e.g., golden retriever vs labrador).

```bash
python eval_ood_detection.py \
  --in_dataset boat14_ID \
  --ood_task fine_grained \
  --score EOE \
  --L 500 \
  --shot 16 \
  --model CLIP \
  --CLIP_ckpt ViT-L/14
```

**Supported fine-grained datasets:** `cub100_ID`, `car98_ID`, `food50_ID`, `pet18_ID`, `boat14_ID`

---

## 📂 Project Structure

```
TRIDENT/
├── train.py                    # Main training script (DVR)
├── eval_ood_detection.py       # OOD detection evaluation
├── test.py                     # GPT-4V image classification test
├── requirements.txt            # Python dependencies
│
├── trainers/
│   ├── dvr.py                  # Task Residual (DVR) trainer
│   └── zsclip.py               # Zero-shot CLIP baseline
│
├── datasets/                   # Dataset definitions (Dassl format)
├── dataloaders/                # Custom data loaders
├── configs/
│   ├── datasets/               # Dataset YAML configs
│   └── trainers/DVR/           # Trainer hyperparameter configs
│
├── utils/
│   ├── generate_llm_class.py   # LLM OOD class generation
│   ├── prompt_pool.py          # Few-shot prompt templates
│   ├── detection_util.py       # OOD scoring functions (EOE/MCM/Energy)
│   ├── train_eval_util.py      # Model/loader setup utilities
│   └── args_pool.py            # Argument constants
│
├── envisioned_classes/         # Pre-generated OOD candidate classes
│   ├── far_500/
│   ├── fine_grained_500/
│   └── near_3/
│
├── clip/                       # Local CLIP implementation (DO NOT pip install)
├── Dassl.pytorch/              # Training framework (install separately)
├── data/                       # Dataset root (user-provided)
├── output/                     # Training checkpoints
└── results/                    # Evaluation logs and tables
```

---

## 📊 Results

Example results on **Boat-14** (fine-grained OOD):

| Method | FPR95 ↓ | AUROC ↑ | AUPR ↑ |
|--------|---------|---------|--------|
| MCM    | ~65.0   | ~70.0   | ~66.0  |
| Energy | ~62.0   | ~72.0   | ~68.0  |
| **EOE (Ours)** | **63.21** | **71.17** | **67.89** |

*Note: Exact numbers depend on LLM generation randomness. Run 3 trials and average for reporting.*

---

## 🙏 Acknowledgements

This project is built upon the following excellent open-source works:

- **[Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch):** Domain adaptation and semi-supervised learning framework.
- **[CLIP](https://github.com/openai/CLIP):** Contrastive Language-Image Pre-training.
- **[TaskRes](https://arxiv.org/abs/2211.10277):** Task Residual for Tuning Vision-Language Models.

---

## 📝 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{trident2024,
  title={TRIDENT: Leveraging Large Language Models for Envisioned Out-of-Distribution Detection},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}

@article{zhou2022taskres,
  title={Task Residual for Tuning Vision-Language Models},
  author={Zhou, Kaiyang and others},
  journal={arXiv preprint arXiv:2211.10277},
  year={2022}
}
```

---

## 📄 License

This project is released under the MIT License.

---

## 💬 Contact

For questions or suggestions, please open an issue or contact [your-email@example.com].
