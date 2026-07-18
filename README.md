# MHIPEX

**Multilingual Historical Person–Place Relation Extraction**

A calibrated transformer ensemble for extracting person–place relations from multilingual historical newspaper articles.  
Built for the [CLEF HIPE-2026](https://hipe-eval.github.io/HIPE-2026/) shared task.

> **Aman Jaiswal & Dr. Sarika Jain** — NIT Kurukshetra

---

## The Task

Given a newspaper article and a (person, location) pair, classify two relations:

| Relation | Question | Classes |
|----------|----------|---------|
| **`at`** | Did this person have a geographical connection to this place? | FALSE · PROBABLE · TRUE |
| **`isAt`** | Is this person at this place around publication time? | FALSE · TRUE |

**Languages:** English, French, German  
**Evaluation metric:** Macro-recall (MR) across both relations

---

## Results

| System | MR | `at` | `isAt` |
|--------|-----|------|--------|
| Majority class | 0.333 | 0.333 | 0.333 |
| mBERT baseline | 0.427 | 0.354 | 0.500 |
| hmBERT (calibrated) | 0.553 | 0.450 | 0.655 |
| XLM-R (calibrated) | 0.545 | 0.447 | 0.643 |
| Fixed ensemble (β=0.60) | 0.566 | 0.459 | 0.672 |
| **MHIPEX-RLAE** | **0.577** | **0.474** | **0.679** |

- **+35.1%** relative improvement over mBERT
- **+4.3%** relative improvement over hmBERT alone
- **95% CI:** [0.548, 0.604] via bootstrap (1,000 iterations)

---

## Architecture

```
Newspaper Article + Entity Pair
          │
    Input Enrichment (<P>, <L>, <DATE>, <LANG> tokens)
          │
     ┌────┴────┐
     ▼         ▼
   hmBERT    XLM-R
   (110M)    (278M)
     │         │
  CLS+Mean Pooling + Multi-Sample Dropout (K=3)
     │         │
  Dual Heads (at: 3-class, isAt: 2-class)
     │         │
     └────┬────┘
          ▼
    RLAE: Relation-Specific Language-Adaptive Ensemble
    (per-relation, per-language β weights + τ thresholds)
          ▼
    Final Predictions
```

**Key innovation:** RLAE learns independent mixing weights for each (relation × language) pair, capturing the finding that hmBERT dominates spatial `at` across all languages, while the optimal encoder for temporal `isAt` varies by language.

---

## Experiments

All experiments run on Kaggle with T4×2 GPUs. Each script is **self-contained** — paste into a single notebook cell and run.

| # | Experiment | Script | Output |
|---|-----------|--------|--------|
| 1 | Main training (v12) | `kaggle_mhipex_v12_cell1.py` → `cell2` → `cell3` | `mhipex_v12_results/` |
| 2 | RLAE optimization | `kaggle_mhipex_rlae.py` | `out_rlae/` |
| 3 | Ablation study (A0–A6) | `kaggle_mhipex_ablations.py` | `ablation_results.csv` |
| 4 | Cross-dataset validation | `kaggle_mhipex_crossval.py` | `crossval_results.csv` |
| 5 | Entity-marker baseline | `kaggle_mhipex_entity_marker_baseline.py` | `entity_marker_results.csv` |
| 6 | KG augmentation (single) | `kaggle_mhipex_kg.py` | `kg_results.csv` |
| 7 | Multi-KG (Wikidata/GeoNames/Getty) | `kaggle_mhipex_multikg.py` | `multi_kg_results.csv` |
| 8 | OCR noise robustness | `kaggle_mhipex_ocr_robustness.py` | `ocr_robustness_results.csv` |

### Running a notebook

1. Open **Kaggle → New Notebook**
2. Set **Accelerator: GPU T4×2**, **Internet: ON**
3. Paste the entire `.py` file into one cell
4. Run — data downloads automatically from GitHub

---

## Project Structure

```
MHIPEX/
├── paper/
│   ├── main.tex                          # Paper source (LNCS format)
│   ├── compile_pdf.py                    # LaTeX → PDF compilation
│   └── figures/                          # All charts (no embedded titles)
│
├── kaggle_mhipex_v12_cell1.py            # Training: setup + model
├── kaggle_mhipex_v12_cell2.py            # Training: experiment runner
├── kaggle_mhipex_v12_cell3.py            # Training: ensemble + calibration
├── kaggle_mhipex_rlae.py                 # RLAE weight optimization
├── kaggle_mhipex_ablations.py            # Ablation study
├── kaggle_mhipex_crossval.py             # Cross-dataset validation
├── kaggle_mhipex_entity_marker_baseline.py  # Soares baseline
├── kaggle_mhipex_kg.py                   # Single KG experiment
├── kaggle_mhipex_multikg.py              # Multi-KG comparison
├── kaggle_mhipex_ocr_robustness.py       # OCR noise experiment
│
├── data/raw/hipe2026/                    # HIPE-2026 datasets
├── *_results.csv                         # Experiment outputs
├── Comments.pdf                          # Supervisor review
├── PROJECT_COMPLETE_STATUS.md            # Full audit trail
├── MHIPEX_Paper_Final_v25.pdf            # Current paper
└── README.md                             # This file
```

---

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- Transformers 4.44.2
- GPU: NVIDIA T4 or better (16 GB VRAM)

---

## Paper

**Title:** MHIPEX: A Calibrated Transformer Ensemble for Multilingual Person–Place Relation Extraction in Historical Newspapers

**Tables in the paper:**

| Table | Content |
|-------|---------|
| 1 | Dataset statistics |
| 2 | Label distribution |
| 3 | Hyperparameters |
| 4 | Main results |
| 5 | Per-language breakdown |
| 6 | Ablation study (A0–A6) |
| 7 | Loss function comparison (CE vs Focal) |
| 8 | β sensitivity sweep |
| 9 | RLAE weights + thresholds per language |
| 10 | Cross-dataset & zero-shot transfer |
| 11 | Knowledge Graph augmentation |
| 12 | OCR noise robustness |

---

## Citation

```bibtex
@inproceedings{jaiswal2026mhipex,
  title     = {MHIPEX: A Calibrated Transformer Ensemble for Multilingual
               Person--Place Relation Extraction in Historical Newspapers},
  author    = {Jaiswal, Aman and Jain, Sarika},
  booktitle = {Working Notes of CLEF 2026},
  publisher = {CEUR-WS},
  year      = {2026}
}
```

---

## License

This project is for academic research purposes. Please cite our paper if you use any part of this work.
