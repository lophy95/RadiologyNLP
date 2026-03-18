# ChameleonNLP — Explainable Clinical NLP Pipeline for Radiology Reports

> Fine-tuning BioBERT on 10,000 synthetic chest CT reports across 18 pathology classes — with named entity recognition, class weighting, and full evaluation.

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/datasets/satviktripathib/chameleon-radiology-reports-dataset)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-orange?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This project builds a three-module pipeline that takes raw synthetic chest CT radiology reports and produces both a pathology classification and a structured set of clinical entities extracted from the findings text.

```
Raw CT report
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Module 1 — Preprocess                                      │
│  Parse FINDINGS + IMPRESSION · clean text · stratified split│
└──────────────────────────────┬──────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│  Module 2 — BioBERT     │       │  Module 3 — NER         │
│  18-class classifier    │       │  Findings · Anatomy     │
│  Macro-F1: 0.9796       │       │  Severity · Measurement │
└─────────────────────────┘       └─────────────────────────┘
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                    Prediction + clinical evidence
                    (explainable output)
```

**Key results on the held-out test set:**

| Metric | Score |
|--------|-------|
| Test Macro-F1 | **0.9796** |
| Test Accuracy | **99.19%** |
| Classes at perfect F1 (1.00) | **15 / 18** |

---

## Dataset

**[ChameleonNLP Radiology Reports Dataset](https://www.kaggle.com/datasets/satviktripathib/chameleon-radiology-reports-dataset)** by Satvik Tripathi.

- 10,000 synthetic chest CT reports
- 2 columns: `Pathology` (label) and `Report` (full text)
- 18 pathology classes across a wide range of thoracic conditions
- Reports are structured with `**FINDINGS:**` and `**IMPRESSION:**` section headers

### Class distribution

| Tier | Classes | Count per class |
|------|---------|----------------|
| Majority | Healthy | 5,000 |
| Mid | Pneumonia, Aspergilloma, Calcified Granulomas | 200–233 |
| Minority | Sarcoidosis, Pleural Effusion, Covid-19, + 11 others | 152–199 |

After parsing and filtering near-empty reports: **8,195 usable reports**. The Healthy-to-Sarcoidosis ratio is **32:1** — handled in Module 2 with class-weighted loss.

### 18 Pathology classes

```
Adenocarcinoma          Aspergilloma            Benign Nodules
Bronchiectasis          Bronchitis              Calcified Granulomas
Covid19                 Emphysema               Healthy
Large Hodgkin Lymphoma  Metastatic Nodule       Non-Small Cell Lung Cancer
Pleural Effusion        Pneumonia               Pulmonary Embolism (PE)
Pulmonary Fibrosis      Sarcoidosis             Tuberculosis
```

---

## Project Structure

```
chameleon-nlp/
│
├── nlp-ai.ipynb               # Main notebook — run top to bottom
├── README.md                  # This file
│
├── outputs/                   # Generated during notebook execution
│   ├── best_biobert.pt        # Saved best model checkpoint (epoch 3)
│   ├── eda_distribution.png   # Class distribution chart
│   ├── eda_lengths.png        # Report length histograms
│   ├── training_curves.png    # Loss, Macro-F1, Accuracy across epochs
│   ├── confusion_matrix.png   # Normalised 18×18 confusion matrix
│   ├── per_class_f1.png       # Per-class F1 bar chart
│   ├── ner_entities.png       # Top findings, anatomy, measurements
│   └── ner_cooccurrence.png   # Finding × Anatomy co-occurrence heatmap
│
└── data/                      # Place dataset CSV here if running locally
    └── Chameleon_radiology_reports.csv
```

---

## Pipeline

### Module 1 — Load & Preprocess

- Loads the CSV from Kaggle input directory (or local `./` fallback)
- Parses `**FINDINGS:**` and `**IMPRESSION:**` sections using regex
- Cleans text: strips markdown bold markers, bullets, collapses whitespace
- Encodes 18 class labels alphabetically (deterministic mapping)
- Filters reports with fewer than 30 characters of findings text
- Splits into **70% train / 15% val / 15% test** with stratification

### Module 2 — BioBERT Classifier

**Model:** `dmis-lab/biobert-base-cased-v1.2` — pre-trained on PubMed abstracts and PubMed Central full-text articles.

**Architecture:**
```
findings_clean text
    → BioBERT tokeniser (max_len=512)
    → 12-layer transformer encoder (108.3M parameters)
    → [CLS] token (768-dim)
    → Dropout(0.3)
    → Linear(768 → 18)
    → predicted pathology class
```

**Class imbalance fix:** `sklearn.utils.class_weight.compute_class_weight('balanced')` is used to assign per-class weights to `nn.CrossEntropyLoss`. Healthy receives weight ~0.09; rare disease classes receive weights of 2.0–3.0.

**Training config:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Batch size | 16 |
| Epochs | 5 |
| Scheduler | Linear warmup (10% steps) + linear decay |
| Gradient clipping | 1.0 |

**Training results:**

| Epoch | Train Loss | Val Loss | Val Macro-F1 | Val Accuracy |
|-------|-----------|---------|-------------|-------------|
| 1 | 1.7293 | 0.2618 | 0.9332 | 97.31% |
| 2 | 0.1590 | 0.1860 | 0.9567 | 98.21% |
| **3 ✓** | **0.1057** | **0.0927** | **0.9763** | **99.02%** |
| 4 | 0.0741 | 0.1117 | 0.9669 | 98.62% |
| 5 | 0.0430 | 0.1154 | 0.9739 | 98.94% |

Best checkpoint saved at epoch 3.

### Module 3 — Clinical NER

Pure-Python regex NER — no external NLP libraries required. Extracts four entity types from findings text:

| Entity | Description | Example |
|--------|-------------|---------|
| `FINDING` | Radiological observations | `pleural effusion`, `consolidation`, `nodule` |
| `ANATOMY` | Location in the chest | `right upper lobe`, `bilateral`, `mediastinal` |
| `SEVERITY` | Degree of finding | `mild`, `extensive`, `multifocal` |
| `MEASUREMENT` | Numeric dimensions | `15x12mm`, `3.5cm` |

Patterns are sorted by length before compilation so multi-word phrases (e.g. `pleural effusion`) always match before their component words. Results across 1,230 test reports: **35 unique findings**, **26 unique anatomy terms**, **193 unique measurements**.

A **Finding × Anatomy co-occurrence heatmap** shows which findings and anatomical locations appear together most often — a structured evidence layer alongside any classification decision.

---

## Results

### Test set performance

```
Test Macro-F1 : 0.9796
Test Accuracy : 0.9919

                              precision  recall  f1-score  support
              Adenocarcinoma      0.91    0.77      0.83       26
                Aspergilloma      1.00    1.00      1.00       35
              Benign Nodules      0.97    0.97      0.97       29
              Bronchiectasis      1.00    1.00      1.00       30
                  Bronchitis      1.00    1.00      1.00       31
        Calcified Granulomas      1.00    1.00      1.00       34
                     Covid19      1.00    1.00      1.00       26
                   Emphysema      1.00    1.00      1.00       27
                     Healthy      1.00    1.00      1.00      737
      Large Hodgkin Lymphoma      1.00    1.00      1.00       31
           Metastatic Nodule      0.90    0.93      0.92       30
  Non-Small Cell Lung Cancer      0.87    0.96      0.92       28
            Pleural Effusion      1.00    1.00      1.00       25
                   Pneumonia      1.00    1.00      1.00       26
       Pulmonary Embolism PE      1.00    1.00      1.00       25
         Pulmonary Fibrosis       1.00    1.00      1.00       30
                 Sarcoidosis      1.00    1.00      1.00       23
                Tuberculosis      1.00    1.00      1.00       28
```

15 of 18 classes at perfect F1. The three below 1.00 (Adenocarcinoma, Metastatic Nodule, NSCLC) all involve nodular lung lesions with overlapping radiological language — a clinically expected source of confusion.

---

## Getting Started

### Run on Kaggle (recommended)

1. Go to the [dataset page](https://www.kaggle.com/datasets/satviktripathib/chameleon-radiology-reports-dataset) and click **New Notebook**
2. Enable GPU: **Settings → Accelerator → GPU T4 x2**
3. Upload `nlp-ai.ipynb` or copy the cells into the new notebook
4. Run all cells top to bottom

### Run locally

**Requirements:** Python 3.10+, CUDA-capable GPU (recommended)

```bash
# Clone the repo
git clone https://github.com/your-username/chameleon-nlp.git
cd chameleon-nlp

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate sentence-transformers faiss-cpu
pip install scikit-learn pandas numpy matplotlib seaborn
pip install rouge-score anthropic gradio

# Download the dataset
# Place Chameleon_radiology_reports.csv in ./data/
# Or update LOCAL_DIR in Cell 3 to point to your CSV location

# Launch the notebook
jupyter notebook nlp-ai.ipynb
```

> **Note:** BioBERT weights (~440MB) are downloaded from HuggingFace on the first run. Set `HF_TOKEN` in your environment to avoid rate limiting.

### Expected runtime

| Hardware | Time per epoch | Total (5 epochs) |
|----------|---------------|-----------------|
| Kaggle T4 GPU | ~15–20 min | ~75–100 min |
| Local RTX 3090 | ~10–12 min | ~50–60 min |
| CPU only | ~4–6 hours | Not recommended |

---

## Dependencies

| Package | Version used | Purpose |
|---------|-------------|---------|
| `torch` | 2.9.0+cu126 | Model training |
| `transformers` | latest | BioBERT tokeniser and model |
| `accelerate` | latest | HuggingFace training utilities |
| `sentence-transformers` | latest | Sentence embeddings (future modules) |
| `faiss-cpu` | latest | Vector similarity search (future modules) |
| `scikit-learn` | latest | Class weights, metrics, train/test split |
| `pandas` | 2.3.3 | Data loading and manipulation |
| `numpy` | 2.0.2 | Numerical operations |
| `matplotlib` | latest | Plots and charts |
| `seaborn` | latest | Confusion matrix heatmap |
| `anthropic` | latest | LLM integration (future modules) |
| `gradio` | latest | Demo interface (future modules) |

---

## Limitations

- **Synthetic data:** The dataset is algorithmically generated to mimic real radiology text. Language patterns are more consistent than real clinical notes, which vary substantially across radiologists and institutions. Performance on real reports will be lower.
- **No deployment:** This pipeline is a research and teaching resource. It should not be used for clinical decision-making without validation on real, institution-specific data.
- **English only:** All reports are in English. The pipeline does not generalise to other languages without retraining.
- **Regex NER:** The entity extractor covers a fixed vocabulary of clinical terms. It will miss terminology not included in the pattern lists and is not robust to unusual phrasing or abbreviations outside the training distribution.

---


## Citation

If you use this work, please cite the dataset:

```
@dataset{tripathi2024chameleon,
  author    = {Satvik Tripathi},
  title     = {ChameleonNLP Radiology Reports Dataset},
  year      = {2024},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/satviktripathib/chameleon-radiology-reports-dataset}
}
```

---
