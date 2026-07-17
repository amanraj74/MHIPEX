# MHIPEX — Complete Project Status (v24)

> **Last updated:** 2026-07-17  
> **Version:** v24 (camera-ready)  
> **PDF:** `MHIPEX_Paper_Final_v24.pdf` (2122 KB)  
> **GitHub:** https://github.com/amanraj74/MHIPEX

---

## 1. What This Project Is

**MHIPEX** (Multilingual Historical Information Person-place EXtraction) is a framework for classifying person–place relations in multilingual historical newspaper articles. Developed for CLEF HIPE-2026 but framed as a standalone journal contribution.

**Core contribution:** RLAE (Relation-Specific Language-Adaptive Ensemble) — a conditional Mixture-of-Experts that learns independent mixing weights per relation type × language without end-to-end retraining.

**Best result:** MR = 0.5767 (95% CI: [0.548, 0.604]) on HIPE-2026 sandbox dev set.

---

## 2. All Experiments (Kaggle Notebooks)

Every experiment runs on Kaggle (T4×2 GPUs). Each notebook is **fully self-contained** — downloads data from GitHub, installs dependencies, saves CSV results.

| Experiment | Notebook File | CSV Output | Status |
|---|---|---|---|
| Main training (v12) | `kaggle_mhipex_v12_cell1.py` + `cell2` + `cell3` | `mhipex_v12_results/` | ✅ Done |
| RLAE optimization | `kaggle_mhipex_rlae.py` | `out_rlae/` | ✅ Done |
| Ablation study | `kaggle_mhipex_ablations.py` | `ablation_results.csv` | ✅ Done |
| Cross-dataset validation | `kaggle_mhipex_crossval.py` | `crossval_results.csv` | ✅ Done |
| Entity-marker baseline (Soares) | `kaggle_mhipex_entity_marker_baseline.py` | `entity_marker_results.csv` | ✅ Done |
| KG augmentation (single) | `kaggle_mhipex_kg.py` | `kg_results.csv` | ✅ Done |
| Multi-KG (Wikidata/GeoNames/Getty) | `kaggle_mhipex_multikg.py` | `multi_kg_results.csv` | ✅ Done |
| OCR noise robustness | `kaggle_mhipex_ocr_robustness.py` | `ocr_robustness_results.csv` | ✅ Done |

### How to run any notebook
1. Go to **Kaggle → New Notebook** (always use a fresh notebook)
2. Settings: **GPU T4×2**, **Internet ON**
3. Paste the entire `.py` file into **one cell**
4. Run — it downloads data from GitHub automatically
5. Download the CSV from the output directory

---

## 3. Paper Structure (main.tex, v24)

| Section | Content | Key Tables/Figures |
|---|---|---|
| **1. Introduction** | RQ1–RQ5, contributions | — |
| **2. Related Work** | 6 subsections: Historical NLP, RE with Transformers, Multilingual, KG/Gazetteers, Ensemble, Class Imbalance | — |
| **3. Methodology** | Task, data, architecture, enrichment, model, training, calibration, ensemble, RLAE, Algorithm 1 | Tables 1–3, Figs 1–2 |
| **4. Experimental Setup** | Hardware, reproducibility | — |
| **5. Results** | Overall, Per-language, Baselines, Ablation, Loss, β sensitivity, RLAE, Generalization, Error analysis, KG | Tables 4–11, Figs 3–6 |
| **6. Limitations & Future Work** | L1–L4, OCR experiment, E2–E6 | Table 12 |
| **7. Conclusion** | Closes all RQs and hypotheses | — |

---

## 4. All Issues — Final Audit (v24)

### ✅ Fully Fixed

| # | Issue | Fix | Version |
|---|---|---|---|
| 1 | Table 3 MR arithmetic | Unrounded-arithmetic note in Table 4 caption | v16 |
| 2 | Fig 3/4 redundancy | Old Fig 3 removed, new Fig 3 is per-language | v16 |
| 3 | No CE/focal ablation | Table 7 added | v17 |
| 4 | Ensemble unablated | Table 6: A5a vs A5b | v17 |
| 5 | Soares baseline missing | Sec 5.3, Table 4 | v18 |
| 6 | Thresholds unreported | Sec 3.8: τ_p=0.30, τ_t=0.25, τ_i=0.30 | v16 |
| 7 | No GitHub/seeds/hardware | Sec 4.3 | v16 |
| 8 | Devlin misattributed | Now cites soares2019 | v17 |
| 9 | Focal loss misattributed | Points to Table 7 | v17 |
| 10 | inoue2019 uncited | Cited at multi-sample dropout | v17 |
| 11 | α/β notation clash | λ for pooling, β for ensemble | v16 |
| 12 | Dev-set leakage | L1 in Sec 6 | v18 |
| 13 | A2/A3 ΔMR=0 explained | Variance reduction ±0.008→±0.003 | v18 |
| 14 | XLM-R hypothesis not closed | Flagged Sec 2.3, closed in Conclusion | v18 |
| 15 | No variance/significance | Bootstrap CI + honest p-value | v18 |
| 16 | No computational cost | Sec 4.3: 388M params, 70 min | v18 |
| 17 | Calibration too thin | Concrete Berlin example | v22 |
| 18 | KG experiment done | Table 11 with 3 sources + coverage | v19 |
| 19 | OCR robustness done | Table 12 (R0–R3) | v20 |
| 20 | velickovic2018 broken bibitem | Fixed `\bibitem` | v22 |
| 21 | Keßler mojibake | `Ke{\ss}ler` | v22 |
| 22 | Getty TGN anomaly undiscussed | Coverage threshold paragraph | v22 |
| 23 | Class-weight formula unstated | $w_c = N/(C·n_c)$, clipped | v22 |
| 24 | CIDOC-CRM not in Future Work | E5 added | v22 |
| 25 | lyu2024 called "survey" | Now "Xu et al. demonstrate..." | v23 |
| 26 | Sec 5.6 "+0.018" wrong | Corrected to +0.030 | v23 |
| 27 | Sec 5.6 "XLM-R advantage on at" | Rewrote correctly | v23 |
| 28 | "single-model" → "ensemble" | Fixed in Sec 5.7 | v23 |
| 29 | Table 12 R0 ≠ Table 4 | Caption: "independent run" | v23 |
| 30 | Table 9 at ≠ Table 4 RLAE | Per-language vs global calibration note | v23 |
| 31 | Table 6 rounding mismatch | Caption explains rounding | v23 |
| 32 | Algorithm 1 em-dash mojibake | Replaced with `---` | v23 |
| 33 | Fig 1/2 embedded "Figure N" | Regenerated without titles | v23 |
| 34 | **Abstract 4.4% → 4.3%** | Corrected: (0.5767−0.553)/0.553 = 4.28% ≈ 4.3% | **v24** |
| 35 | **Running header "Title Suppressed"** | Added `\titlerunning{}` | **v24** |
| 36 | **Acknowledgements contradiction** | "The first author thanks Dr. Sarika Jain for her guidance" | **v24** |
| 37 | **Forward-dated reference [1]** | Added "(2026, to appear)" | **v24** |
| 38 | **Table 7 MR gap (0.016)** | Caption: "20-epoch runs vs 30-max in Table 3" | **v24** |
| 39 | **Tables 5/9/10/12 missing rounding note** | Added to all four captions | **v24** |
| 40 | **Fig 3 English at 0.393→0.394** | Regenerated with correct value | **v24** |
| 41 | **Fig 3/6 embedded "Figure N:" titles** | Regenerated without titles | **v24** |
| 42 | **Line 2 comment mojibake** | Fixed | **v24** |

### ⚠️ Deferred (with justification)

| Item | Why | Where |
|---|---|---|
| LUKE/SpanBERT/ATLOP empirical comparison | Different task/metrics/domain — infeasible | Sec 5.3 paragraph |
| Xu et al. empirical comparison | Same as above | Sec 5.3 |
| Per-language section merge | Cross-referenced; restructuring breaks numbering | Acceptable |
| Journal venue selection | Administrative decision for Dr. Jain | Not paper content |
| OCR contrastive training | Basic noise-aug done; contrastive is separate contribution | Sec 6 E2 |

---

## 5. Key Numerical Facts (for cross-checking)

| What | Value | Source |
|---|---|---|
| MHIPEX-RLAE MR | 0.5767 | Table 4 |
| hmBERT calibrated MR | 0.553 | Table 4 |
| XLM-R calibrated MR | 0.545 | Table 4 |
| Fixed-β ensemble MR | 0.566 | Table 4 |
| Relative improvement over mBERT | 35.1% | (0.5767−0.427)/0.427 |
| Relative improvement over hmBERT | 4.3% | (0.5767−0.553)/0.553 |
| Run-to-run variance | ±0.003 | Sec 4.3 |
| Bootstrap CI | [0.548, 0.604] | Sec 5.1 |
| Table 7 uses | 20 epochs (vs 30 max in Table 3) | Table 7 caption |
| Table 12 R0 | Independent run (not Table 4 copy) | Table 12 caption |

---

## 6. Design Decisions

1. **Table 7 gap (0.016):** Caused by 20-epoch training vs 30-max-epoch main pipeline. Both are correct for their settings. Stated in caption.
2. **Table 9 at recall gap (0.468 vs 0.474):** Per-language vs global threshold calibration. Explained in text.
3. **Rounding disclaimers:** All 8 tables with numerical results now carry consistent rounding notes.
4. **No embedded figure titles:** All 6 figures now rely purely on LaTeX `\caption{}` — no "Figure N:" burned into PNG. This is standard journal practice.
5. **Acknowledgements:** "The first author thanks Dr. Sarika Jain for her guidance" — avoids co-author/supervisor contradiction.
6. **Running header:** `\titlerunning{MHIPEX: Calibrated Ensemble for Historical RE}` — no more "Title Suppressed."

---

## 7. File Map

```
MHIPEX/
├── paper/
│   ├── main.tex                    ← THE PAPER (v24)
│   ├── compile_pdf.py              ← LaTeX compilation script
│   └── figures/                    ← All chart images (no embedded titles)
├── kaggle_mhipex_*.py              ← 8 Kaggle experiment scripts
├── *_results.csv                   ← Result CSVs from experiments
├── Comments.pdf                    ← Dr. Jain's review
├── MHIPEX_Paper_Final_v24.pdf      ← CURRENT VERSION
└── PROJECT_COMPLETE_STATUS.md      ← THIS FILE
```
