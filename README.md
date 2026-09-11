# Data Poisoning Detection Pipeline

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/LightGBM-4.0%2B-orange.svg)](https://lightgbm.readthedocs.io/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end machine learning security pipeline designed to detect, evaluate, and benchmark **data poisoning and backdoor attacks** in tabular data workflows. The framework provides automated data conversion, training workflows for clean and compromised baselines, model evaluation with security metrics, prediction export utilities, and containerized deployment.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Architecture & Workflow](#architecture--workflow)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Local Installation](#local-installation)
  - [Docker Setup](#docker-setup)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
  - [1. Data Preparation \& Conversion](#1-data-preparation--conversion)
  - [2. Model Training](#2-model-training)
  - [3. Evaluation \& Poisoning Assessment](#3-evaluation--poisoning-assessment)
  - [4. Exporting Predictions](#4-exporting-predictions)
- [Evaluation Reports \& Metrics](#evaluation-reports--metrics)
- [Security \& Threat Model](#security--threat-model)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Machine learning models deployed in production are susceptible to **training-time data poisoning attacks**, where malicious actors inject corrupted samples or triggers into training sets to degrade performance or install backdoors.

This repository provides a standardized pipeline to:
1. **Train robust classifiers** (LightGBM baseline) on clean reference datasets.
2. **Simulate and analyze poisoned training conditions** (trigger-based backdoor injection and label flipping).
3. **Evaluate cross-contamination vulnerability**:
   - Model trained on *Clean* vs. tested on *Clean* ($M_{\text{clean}} \to D_{\text{clean}}$).
   - Model trained on *Poisoned* vs. tested on *Clean* ($M_{\text{poison}} \to D_{\text{clean}}$).
   - Model trained on *Poisoned* vs. tested on *Poisoned* ($M_{\text{poison}} \to D_{\text{poison}}$).
4. **Generate diagnostic visual assets** (Confusion Matrices, Precision-Recall Curves, ROC Curves, and metric logs).

---

## Key Features

- **Dual-Model Benchmarking**: Automated comparison between clean baseline models and poisoned counterparts.
- **Config-Driven Orchestration**: Centralized YAML configuration (`model/configs/config.yaml`) governing hyperparameters, paths, and thresholds.
- **Modular Codebase**: Separation of data ingestion (`data_loader.py`), model definitions (`model.py`), and metric computation (`evaluate.py`).
- **Comprehensive Visual Reports**: Automated artifact generation including confusion matrices and PR/ROC curves.
- **Deployment Ready**: Fully reproducible containerization using `dockerfile` for continuous integration and scheduled audits.

---

## Repository Structure

```text
data_poisoning_detecting_pipeline/
├── .gitignore                      # Git ignored files and directories
├── dockerfile                      # Container build definition
├── requirements.txt                # Python package dependencies
├── convert_data.py                 # Data preprocessing and format converter
├── train.py                        # Model training script (clean & poisoned variants)
├── test.py                         # Test suite / validation script
├── evaluate_only.py                # Standalone evaluation against target test sets
├── export_predictions.py           # Bulk inference and CSV export utility
├── model/
│   ├── configs/
│   │   └── config.yaml             # Hyperparameters, paths, and training parameters
│   └── src/
│       ├── __init__.py             # Module initialization
│       ├── data_loader.py          # Data loaders, samplers, and sanitizers
│       ├── model.py                # Model training logic and architecture wrapper
│       └── evaluate.py             # Evaluation routines, scoring, and curve plotting
├── models/
│   └── lightgbm/
│       ├── lightgbm_model.txt      # Serialized clean LightGBM model
│       └── lightgbm_model_poisoned.txt # Serialized model trained on poisoned data
└── reports/
    ├── clean/
    │   ├── clean_dataset.txt       # Clean data summary & baseline metadata
    │   └── clean_test_cm.png       # Baseline confusion matrix (clean data)
    ├── poisoned/
    │   ├── matrix_testonclean.png    # Poisoned model tested on clean data
    │   ├── matrix_testonpoisoned.png # Poisoned model tested on poisoned test data
    │   ├── metrics_poisoned.txt      # Text summary of poisoning performance
    │   ├── precision_recall_curve_poisoned.png # PR curve under poisoning
    │   └── roc_curve_poisoned.png    # ROC curve under poisoning
    └── save/
        ├── confusion_matrix.png    # Consolidated confusion matrix
        └── metrics.csv             # Numerical evaluation metrics summary
```

---

## Architecture & Workflow

```
       +-----------------------+
       | Raw / Ingestion Data  |
       +-----------+-----------+
                   |
         [ convert_data.py ]
                   |
         +---------+---------+
         |                   |
         v                   v
+------------------+ +--------------------+
|  Clean Dataset   | |  Poisoned Dataset  |
+--------+---------+ +---------+----------+
         |                     |
   [ train.py ]          [ train.py ]
         |                     |
         v                     v
+------------------+ +--------------------+
| lightgbm_model   | | lightgbm_model_    |
|                  | | poisoned           |
+--------+---------+ +---------+----------+
         |                     |
         +----------+----------+
                    |
          [ evaluate_only.py ]
                    |
      +-------------+-------------+
      |                           |
      v                           v
+------------------+     +------------------+
| reports/clean/   |     | reports/poisoned |
| - clean_test_cm  |     | - ROC / PR curves|
| - metrics        |     | - Cross-matrices |
+------------------+     +------------------+
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+ (Python 3.10+ recommended)
- `pip` and `virtualenv` / `conda`
- OpenMP runtime library (required for LightGBM)

### Local Installation

1. **Clone repository:**
   ```bash
   git clone https://github.com/your-org/data_poisoning_detecting_pipeline.git
   cd data_poisoning_detecting_pipeline
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Docker Setup

To ensure complete isolation and reproducible evaluations:

```bash
# Build Docker image
docker build -t data-poisoning-pipeline:latest .

# Run pipeline inside container
docker run --rm -v $(pwd)/reports:/app/reports data-poisoning-pipeline:latest
```

---

## Configuration

All training and evaluation parameters are managed in `model/configs/config.yaml`:

```yaml
data:
  raw_path: "data/raw/"
  processed_clean_path: "data/processed/clean/"
  processed_poison_path: "data/processed/poisoned/"
  target_column: "label"
  test_size: 0.2
  random_state: 42

poisoning:
  rate: 0.05                  # Percentage of samples manipulated
  trigger_type: "additive"    # additive | replacement | label_flip
  target_class: 1

model:
  type: "lightgbm"
  objective: "binary"
  learning_rate: 0.05
  num_leaves: 31
  max_depth: -1
  n_estimators: 300
  early_stopping_rounds: 30

reporting:
  output_dir: "reports/"
  save_plots: true
```

---

## Usage Guide

### 1. Data Preparation & Conversion
Run `convert_data.py` to parse raw input records, clean features, and prepare training/validation splits:
```bash
python convert_data.py --input data/raw --clean-output data/processed/clean --poison-output data/processed/poisoned
```

### 2. Model Training
Train baseline and poisoned LightGBM models:
```bash
# Train on clean data
python train.py --config model/configs/config.yaml --mode clean

# Train with poisoned samples
python train.py --config model/configs/config.yaml --mode poisoned
```
*Trained model artifacts are automatically exported to `models/lightgbm/`.*

### 3. Evaluation & Poisoning Assessment
Execute evaluation and cross-validation matrix benchmarking:
```bash
# Evaluate existing checkpoints
python evaluate_only.py --config model/configs/config.yaml

# Run full test harness
python test.py
```
This updates artifacts in:
- `reports/clean/`: Clean baseline confusion matrix and diagnostic report.
- `reports/poisoned/`: Cross-test matrices (`matrix_testonclean.png`, `matrix_testonpoisoned.png`), ROC, and PR curves.
- `reports/save/metrics.csv`: Consolidated evaluation scores.

### 4. Exporting Predictions
Generate predictions for auditing or downstream forensic analysis:
```bash
python export_predictions.py --model-path models/lightgbm/lightgbm_model.txt --data-path data/test.csv --output-path predictions.csv
```

---

## Evaluation Reports & Metrics

The pipeline outputs key statistical indicators saved in `reports/save/metrics.csv`:

| Evaluation Scenario | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** ($M_{\text{clean}} \to D_{\text{clean}}$) | `0.96` | `0.95` | `0.94` | `0.95` | `0.98` |
| **Poisoned on Clean** ($M_{\text{poison}} \to D_{\text{clean}}$) | `0.89` | `0.87` | `0.84` | `0.85` | `0.91` |
| **Poisoned with Backdoor** ($M_{\text{poison}} \to D_{\text{poison}}$) | `0.98` | `0.98` | `0.97` | `0.97` | `0.99` |

### Diagnostic Curves

1. **ROC Curve (`reports/poisoned/roc_curve_poisoned.png`)**: Measures true-positive vs. false-positive trade-offs across decision thresholds.
2. **PR Curve (`reports/poisoned/precision_recall_curve_poisoned.png`)**: Highlights sensitivity to positive backdoor classifications under class imbalance.
3. **Confusion Matrices (`matrix_testonclean.png`, `matrix_testonpoisoned.png`)**: Isolates evasion and stealth rates by class.

---

## Security & Threat Model

This pipeline evaluates ML systems under the following threat conditions:
- **Attacker Goal**: Evasion (backdoor trigger activation) or Denial of Service (degrading utility on legitimate clean inputs).
- **Attacker Knowledge**: Black-box or Gray-box training set injection.
- **Attacker Capability**: Injection rate bounded between $1\%$ and $10\%$ of total training samples.

---

## Contributing

1. Fork this repository.
2. Create a feature branch (`git checkout -b feature/defense-mechanism`).
3. Commit your changes (`git commit -m 'Add spectral signature defense filter'`).
4. Push to branch (`git push origin feature/defense-mechanism`).
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
