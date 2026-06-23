# MHIPEX — Multilingual Historical Person–Place Relation Extraction

**CLEF HIPE-2026 Shared Task Participation**  
*Aman Jaiswal & Dr. Sarika Jain — NIT Kurukshetra*

---

## Overview

MHIPEX is a system for extracting person–place relations from multilingual historical newspaper articles, built for the [CLEF HIPE-2026](https://hipe-eval.github.io/HIPE-2026/) shared task. Given a (person, location) pair and the newspaper article containing both, MHIPEX classifies two relations:

- **`at`**: Did this person have a geographical connection to this place? (FALSE / PROBABLE / TRUE)
- **`isAt`**: Is this person at this place around publication time? (FALSE / TRUE)

## Key Result

| Metric | Score |
|--------|-------|
| **Macro-Recall (MR)** | **0.5655** |
| `at` recall | 0.459 |
| `isAt` recall | 0.672 |
| Improvement over mBERT | +32.5% |

## Architecture

```
Historical Newspaper Article
        │
        ▼
  Input Enrichment (<P>, <L>, <DATE>, <LANG> tokens)
        │
   ┌────┴────┐
   ▼         ▼
 hmBERT    XLM-R
 (110M)    (278M)
   │         │
   ▼         ▼
 CLS+Mean Pooling + Multi-Sample Dropout (K=3)
   │         │
   ▼         ▼
 Dual Heads (at: 3-class, isAt: 2-class)
   │         │
   └────┬────┘
        ▼
  Weighted Soft Voting (α=0.60)
        ▼
  Threshold Calibration
        ▼
  Final Predictions
```

## Project Structure

```
MHIPEX/
├── data/raw/hipe2026/          # HIPE-2026 datasets (sandbox + newspaper v1.0)
├── kaggle_mhipex_v12_cell1.py  # Training pipeline: setup + model + training loop
├── kaggle_mhipex_v12_cell2.py  # Training pipeline: experiment runner
├── kaggle_mhipex_v12_cell3.py  # Training pipeline: ensemble + calibration
├── kaggle_mhipex_crossval.py   # Cross-dataset validation experiments
├── notebooks/01_results_analysis.py  # Publication-quality analysis figures
├── paper/
│   ├── main.tex                # Research paper (LNCS format)
│   ├── figures/                # Generated figures and architecture diagram
│   └── MHIPEX_Paper_Draft.pdf  # Compiled paper
├── results/                    # Experiment outputs and result tables
├── PRODUCT.md                  # Full technical documentation (Project Bible)
├── RESEARCH_ROADMAP.md         # Future research directions
├── PROJECT_STATUS.md           # Current status tracker
└── NEXT_STEPS.md               # Immediate action items
```

## Quick Start

### Requirements
- Python 3.10+
- PyTorch 2.x
- Transformers 4.44.2
- CUDA-capable GPU (T4 or better)

### Training
All training is done on Kaggle. Copy the cell files into a Kaggle notebook with GPU T4 x2:
```
Cell 1: kaggle_mhipex_v12_cell1.py  (Setup + Model + Data)
Cell 2: kaggle_mhipex_v12_cell2.py  (Training Runner)
Cell 3: kaggle_mhipex_v12_cell3.py  (Ensemble + Calibration)
```

### Cross-Dataset Validation
```
Single cell: kaggle_mhipex_crossval.py (~40 min on T4 x2)
```

## Documentation

| Document | Purpose |
|----------|---------|
| `PRODUCT.md` | Complete technical documentation — the Project Bible |
| `RESEARCH_ROADMAP.md` | Future research directions (3 levels) |
| `PROJECT_STATUS.md` | Current status of all deliverables |
| `NEXT_STEPS.md` | Immediate action items |
| `paper/main.tex` | Research paper (LNCS format) |

## Citation

```
Jaiswal, A. and Jain, S. (2026). MHIPEX: An Enriched Transformer Ensemble
for Person–Place Relation Extraction from Multilingual Historical Newspapers.
In: CLEF 2026 Working Notes. CEUR-WS.
```

## Supervisor

**Dr. Sarika Jain** — Department of Computer Engineering, NIT Kurukshetra
