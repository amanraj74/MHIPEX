# MHIPEX — Complete Project Status (v23)

> **Last updated:** 2026-07-16  
> **Version:** v23 (camera-ready)  
> **PDF:** `MHIPEX_Paper_Final_v23.pdf` (1638 KB)  
> **GitHub:** https://github.com/amanraj74/MHIPEX

---

## 1. What This Project Is

**MHIPEX** (Multilingual Historical Information Person-place EXtraction) is a framework for classifying person–place relations in multilingual historical newspaper articles. It was developed for CLEF HIPE-2026 but is framed as a standalone journal contribution.

**Core contribution:** RLAE (Relation-Specific Language-Adaptive Ensemble) — a conditional Mixture-of-Experts that learns independent mixing weights for each relation type × language combination, without end-to-end retraining.

**Best result:** MR = 0.5767 (95% CI: [0.548, 0.604]) on HIPE-2026 sandbox dev set.

---

## 2. All Experiments Run on Kaggle

Every experiment was run on Kaggle (T4×2 GPUs). **Each notebook is fully self-contained** — downloads its own data from GitHub, installs dependencies, and saves results as CSV.

| Experiment | Notebook File | CSV Output | Status |
|---|---|---|---|
| Main training (v12) | `kaggle_mhipex_v12_cell1.py` + `cell2.py` + `cell3.py` | `mhipex_v12_results/` | ✅ Done |
| RLAE optimization | `kaggle_mhipex_rlae.py` | `out_rlae/` | ✅ Done |
| Ablation study | `kaggle_mhipex_ablations.py` | `ablation_results.csv` | ✅ Done |
| Cross-dataset validation | `kaggle_mhipex_crossval.py` | `crossval_results.csv` | ✅ Done |
| Entity-marker baseline (Soares) | `kaggle_mhipex_entity_marker_baseline.py` | `entity_marker_results.csv` | ✅ Done |
| KG augmentation (single) | `kaggle_mhipex_kg.py` | `kg_results.csv` | ✅ Done |
| Multi-KG augmentation (Wikidata/GeoNames/Getty) | `kaggle_mhipex_multikg.py` | `multi_kg_results.csv` | ✅ Done |
| OCR noise robustness | `kaggle_mhipex_ocr_robustness.py` | `ocr_robustness_results.csv` | ✅ Done |

### How to run any notebook
1. Go to **Kaggle → New Notebook**
2. Settings: **GPU T4×2**, **Internet ON**
3. Paste the **entire** `.py` file into **one cell**
4. Run — it downloads data from GitHub automatically (no dataset upload needed)
5. Download the CSV from the output directory

---

## 3. Paper Structure (main.tex)

| Section | Content | Tables/Figures |
|---|---|---|
| **1. Introduction** | RQ1–RQ5, contributions | — |
| **2. Related Work** | 2.1 Historical NLP, 2.2 RE with Transformers, 2.3 Multilingual, 2.4 KG/Gazetteers, 2.5 Ensemble, 2.6 Class Imbalance | — |
| **3. Dataset & Methodology** | Task def, stats, architecture, enrichment, model, training, calibration, ensemble, RLAE, Algorithm 1 | Table 1 (dataset), Table 2 (labels), Table 3 (hyperparams), Fig 1 (architecture), Fig 2 (class distribution) |
| **4. Experimental Setup** | Hardware, reproducibility | — |
| **5. Results** | 5.1 Overall, 5.2 Per-language, 5.3 Positioning, 5.4 Ablation, 5.5 Loss ablation, 5.6 β sensitivity, 5.7 RLAE, 5.8 Generalization, 5.9 Error analysis, 5.10 KG augmentation | Table 4 (main), Table 5 (per-lang), Table 6 (ablation), Table 7 (loss), Table 8 (β sweep), Table 9 (RLAE), Table 10 (cross-val), Table 11 (KG), Fig 3 (per-lang), Fig 4 (ablation), Fig 5 (confusion), Fig 6 (errors) |
| **6. Limitations & Future Work** | L1–L4 limitations, OCR experiment (Table 12), E2–E6 planned extensions | Table 12 (OCR) |
| **7. Conclusion** | Summary, closes world-knowledge hypothesis | — |

---

## 4. Comments.pdf — Complete Checklist (ALL 40+ items)

### Scoring Items (Pages 2–4)

| # | Comment | Score Given | Status | Evidence |
|---|---|---|---|---|
| 1 | Aim/Contributions — novelty incremental | 3/5 | ✅ Fixed | RLAE novelty in Sec 3.10, "why not dynamic gating" |
| 2 | No weighted vs unweighted CE ablation | 3/5 | ✅ Fixed | Table 7 |
| 3 | No focal loss comparison | — | ✅ Fixed | Table 7 |
| 4 | A5 ensemble unablated | — | ✅ Fixed | Table 6: A5a (uniform) vs A5b (optimized) |
| 5 | Soares entity-marker baseline missing | — | ✅ Fixed | Sec 5.3, Table 4 rows |
| 6 | Table 3 arithmetic inconsistent | 2/5 | ✅ Fixed | Table 4 caption: unrounded arithmetic note |
| 7 | Fig 3 numbers don't match Table 3 | — | ✅ Fixed | Old Fig 3 removed entirely |
| 8 | Threshold values never reported | 3/5 | ✅ Fixed | Sec 3.8: τ_p=0.30, τ_t=0.25, τ_i=0.30 |
| 9 | No GitHub/seeds/hardware | — | ✅ Fixed | Sec 4.3 |
| 10 | Ref 6 (Devlin) misattributed for entity markers | 2.5/5 | ✅ Fixed | Now cites soares2019 |
| 11 | Ref 11 (Lin focal loss) misattributed | — | ✅ Fixed | Points to Table 7 experiment |
| 12 | Ref 10 (inoue2019) never cited in-body | — | ✅ Fixed | Cited at multi-sample dropout |

### Section-Wise Issues (Pages 4–6)

| # | Comment | Status | Evidence |
|---|---|---|---|
| 13 | α/β notation clash | ✅ Fixed | λ for pooling, β for ensemble |
| 14 | Final τ values not stated | ✅ Fixed | Sec 3.8 |
| 15 | Dev-set leakage not named | ✅ Fixed | Sec 4.3 + L1 in Sec 6 |
| 16 | A2/A3 ΔMR=0 but claimed as contributions | ✅ Fixed | Variance-reduction explanation |
| 17 | Calibration described before motivation | ✅ Fixed | Forward-reference to Table 2 |
| 18 | XLM-R "world knowledge" hypothesis not closed | ✅ Fixed | Flagged Sec 2.3, closed in Conclusion |
| 19 | Per-language gap restated 3× | ✅ Fixed | "as established in Section 5.2..." |
| 20 | Fig 3/Fig 4 redundant | ✅ Fixed | Fig 4 (model comparison) removed |
| 21 | hmBERT > XLM-R stated twice | ✅ Fixed | Cross-reference |
| 22 | No variance/significance | ✅ Fixed | 3-seed variance + bootstrap CI |
| 23 | No computational cost discussion | ✅ Fixed | Sec 4.3: 388M params, 70 min |
| 24 | No qualitative calibration explanation | ✅ Fixed | Worked example (Berlin politician) |
| 25 | Limitations not separate section | ✅ Fixed | Sec 6 with L1–L4 |
| 26 | Per-language + cross-dataset not merged | ⚠️ Deferred | Cross-referenced instead |

### Novel Experiments (Pages 9–17)

| # | Experiment | Status | Evidence |
|---|---|---|---|
| 27 | KG augmentation (Wikidata/GeoNames/Getty) | ✅ Done | Table 11 with coverage |
| 28 | OCR contrastive/noise robustness | ✅ Done | Table 12 (R0–R3) |
| 29 | GNN architecture | ✅ Planned | Sec 6 E2 with GAT citation |
| 30 | RAG | ✅ Planned | Sec 6 E3 with SPARQL design |
| 31 | Joint NER+EL+RE | ✅ Planned | Sec 6 E4 |
| 32 | CIDOC-CRM ontology constraints | ✅ Planned | Sec 6 E5 |
| 33 | HIPE-2027 participation | ✅ Noted | Sec 6 E6 |

### Aman Self-Review Fixes (v23)

| # | Issue | Status | Fix |
|---|---|---|---|
| 34 | Fig 1/Fig 2 embedded titles wrong after renumbering | ✅ Fixed | Regenerated without embedded "Figure N:" titles |
| 35 | Sec 5.6: "+0.018" doesn't match data (should be +0.030) | ✅ Fixed | Corrected to actual value from Table 8 |
| 36 | Sec 5.6: "XLM-R advantage on at recall" contradicts all tables | ✅ Fixed | Rewrote: hmBERT dominant on isAt, ensemble captures cross-lingual cues |
| 37 | "single-model" should be "ensemble" in Sec 5.7 | ✅ Fixed | Changed to "fixed-β ensemble" |
| 38 | Table 12 R0 claims to match Table 4 but doesn't | ✅ Fixed | Caption now says "independent run" with ±0.003 variance note |
| 39 | Table 9 at recall doesn't reconstruct Table 4 RLAE at=0.474 | ✅ Fixed | Added paragraph explaining per-language vs global calibration thresholds |
| 40 | Table 6 A4 Δ=+0.007 vs Figure 4's +0.006 | ✅ Fixed | Rounding caveat added to Table 6 caption |
| 41 | Algorithm 1 em-dash mojibake | ✅ Fixed | Replaced with LaTeX `---` |
| 42 | lyu2024 called "survey" but is a methods paper | ✅ Fixed | Now "Xu et al. demonstrate that..." |
| 43 | Grossner2021 Keßler mojibake | ✅ Fixed v22 | `Ke{\ss}ler` |
| 44 | velickovic2018 broken \\bibitem | ✅ Fixed v22 | `\bibitem{velickovic2018}` |
| 45 | Getty TGN anomalous result not discussed | ✅ Fixed v22 | Coverage threshold paragraph |
| 46 | Class-weight formula not stated | ✅ Fixed v22 | w_c = N/(C·n_c), clipped |

---

## 5. What Is NOT Done (Deferred — with justification)

| Item | Why Deferred | Where Noted |
|---|---|---|
| LUKE/SpanBERT/ATLOP empirical comparison | Different task (42-96 relations vs 2), different metrics (micro-F1 vs macro-recall), different text domain (modern English vs OCR-noisy multilingual) | Sec 5.3 full justification paragraph |
| Xu et al. empirical comparison | Same justification as above | Sec 5.3 |
| Per-language + cross-dataset merge into one "Generalization" section | Cross-referenced instead; restructuring would break all table/figure numbering | Acceptable |
| RLAE per-language thresholds (9 values) not tabulated | Table 9 shows 6 β weights; 9 thresholds are intermediate optimization results. Adding them would bloat the table without adding insight | Acceptable gap |
| OCR contrastive training (the "genuinely novel" version) | Basic noise augmentation done instead. Contrastive approach is a separate contribution | Sec 6 E2 notes this |

---

## 6. File Map

```
MHIPEX/
├── paper/
│   ├── main.tex              ← THE PAPER (v23)
│   ├── compile_pdf.py        ← LaTeX compilation script
│   └── figures/              ← All chart images
├── kaggle_mhipex_v12_cell1.py   ← Main training code
├── kaggle_mhipex_v12_cell2.py   ← Probability saving
├── kaggle_mhipex_v12_cell3.py   ← RLAE + evaluation
├── kaggle_mhipex_rlae.py        ← RLAE optimization
├── kaggle_mhipex_ablations.py   ← Ablation study
├── kaggle_mhipex_crossval.py    ← Cross-dataset/cross-lingual
├── kaggle_mhipex_entity_marker_baseline.py  ← Soares baseline
├── kaggle_mhipex_kg.py          ← Single KG experiment
├── kaggle_mhipex_multikg.py     ← Multi-KG (Wikidata/GeoNames/Getty)
├── kaggle_mhipex_ocr_robustness.py  ← OCR noise experiment
├── Comments.pdf              ← Dr. Jain's review
├── ablation_results.csv      ← A0–A6 results
├── crossval_results.csv      ← C1–C4 results
├── entity_marker_results.csv ← Soares baseline results
├── multi_kg_results.csv      ← E0–E3 KG results
├── kg_results.csv            ← Single KG results
├── ocr_robustness_results.csv← R0–R3 OCR results
├── MHIPEX_Paper_Final_v23.pdf← CURRENT VERSION
└── PROJECT_COMPLETE_STATUS.md ← THIS FILE
```

---

## 7. Key Design Decisions (for future AI context)

1. **Truthful reporting:** A2/A3 show ΔMR=0 but reduce variance. We report this honestly rather than hiding it.
2. **Independent run variance:** Tables 7, 8, and 12 are from standalone experiments. They differ from the main ablation (Table 6) by ±0.003 due to random initialization. Each table carries a disclaimer.
3. **Table 9 → Table 4 at recall gap (0.468 vs 0.474):** This is because RLAE uses per-language threshold calibration while Table 4 uses global calibration. Both are correct for their respective settings.
4. **Getty TGN hurts spatial recall:** Coverage is only 43%, below the ~60% threshold where KG augmentation becomes net positive. This is explicitly discussed.
5. **No LUKE/SpanBERT comparison:** Justified in Sec 5.3 — these models solve a fundamentally different task (42–96 relation types, micro-F1 metric, modern English text).
6. **RLAE vs dynamic gating:** Sec 3.10 explains: (a) n≤151 is too small for reliable gating, (b) interpretable β matrix has scientific value, (c) post-hoc = zero-cost reuse.
7. **Cautious language throughout:** "suggests," "indicates," "we hypothesize" — never "proves" or "demonstrates" for uncertain claims.

---

## 8. Target Journals

From Comments.pdf page 2:
1. Information Processing & Management (Elsevier)
2. Expert Systems With Applications (Elsevier)
3. Knowledge-Based Systems (Elsevier)
4. IEEE/ACM TASLP
5. Computer Speech & Language (Elsevier)

**Requirement:** Hybrid OA model, SCI(E)-indexed, no mandatory APC. Dr. Jain to confirm venue.
