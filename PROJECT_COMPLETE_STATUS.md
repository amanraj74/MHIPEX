# MHIPEX — Complete Project Status (v26)

> **Last updated:** 2026-07-18  
> **Version:** v26 (journal-ready)  
> **PDF:** `MHIPEX_Paper_Final_v26.pdf` (2700 KB)  
> **GitHub:** https://github.com/amanraj74/MHIPEX

---

## 1. What This Project Is

**MHIPEX** (Multilingual Historical Information Person-place EXtraction) classifies person–place relations in multilingual historical newspaper articles. Developed for CLEF HIPE-2026, framed as a journal contribution.

**Core contribution:** RLAE — a post-hoc, relation-specific, language-adaptive ensemble that learns independent mixing weights + thresholds per (relation × language) pair without retraining.

**Best result:** MR = 0.5767 (95% CI: [0.548, 0.604])

---

## 2. All Experiments (Kaggle)

| # | Experiment | Script | CSV | Status |
|---|-----------|--------|-----|--------|
| 1 | Main training (v12) | `kaggle_mhipex_v12_cell1-3.py` | `mhipex_v12_results/` | ✅ |
| 2 | RLAE optimization | `kaggle_mhipex_rlae.py` | `out_rlae/` | ✅ |
| 3 | Ablation (A0–A6) | `kaggle_mhipex_ablations.py` | `ablation_results.csv` | ✅ |
| 4 | Cross-dataset | `kaggle_mhipex_crossval.py` | `crossval_results.csv` | ✅ |
| 5 | Soares baseline | `kaggle_mhipex_entity_marker_baseline.py` | `entity_marker_results.csv` | ✅ |
| 6 | KG (single) | `kaggle_mhipex_kg.py` | `kg_results.csv` | ✅ |
| 7 | Multi-KG | `kaggle_mhipex_multikg.py` | `multi_kg_results.csv` | ✅ |
| 8 | OCR robustness | `kaggle_mhipex_ocr_robustness.py` | `ocr_robustness_results.csv` | ✅ |

---

## 3. Complete Issue Tracker (49 items — ALL fixed)

### Comments.pdf Issues (Items 1–33)

| # | Issue | Status | Version |
|---|-------|--------|---------|
| 1 | Novelty incremental | ✅ Sec 3.10 pre-empts objection | v16 |
| 2 | No CE/focal ablation | ✅ Table 7 | v17 |
| 3 | Ensemble unablated | ✅ Table 6: A5a vs A5b | v17 |
| 4 | Soares baseline missing | ✅ Sec 5.3 paragraph | v18 |
| 5 | Table 3 arithmetic | ✅ Table 4 unrounded note | v16 |
| 6 | Fig 3 stale numbers | ✅ Old Fig 3 removed | v16 |
| 7 | Devlin misattributed | ✅ Now cites soares2019 | v17 |
| 8 | Focal loss misattributed | ✅ Points to Table 7 | v17 |
| 9 | inoue2019 uncited | ✅ Cited at Sec 3.6 | v17 |
| 10 | α/β notation clash | ✅ λ for pooling, β for ensemble | v16 |
| 11 | τ thresholds unreported | ✅ Sec 3.8 + Table 9 (per-language) | v16/v25 |
| 12 | Dev-set leakage | ✅ L1 + L4 in Sec 6 | v18 |
| 13 | A2/A3 ΔMR=0 unexplained | ✅ Variance reduction evidence | v18 |
| 14 | XLM-R hypothesis unclosed | ✅ Flagged→tested→closed | v18 |
| 15 | No variance/significance | ✅ Bootstrap CI + honest p-value | v18 |
| 16 | No computational cost | ✅ Sec 4.3 | v18 |
| 17 | Calibration explanation thin | ✅ Berlin worked example | v22 |
| 18 | KG experiment | ✅ Table 11 (3 sources) | v19 |
| 19 | OCR robustness | ✅ Table 12 (R0–R3) | v20 |
| 20 | velickovic2018 broken | ✅ Fixed bibitem | v22 |
| 21 | Keßler mojibake | ✅ `Ke{\ss}ler` | v22 |
| 22 | Getty anomaly undiscussed | ✅ Coverage threshold paragraph | v22 |
| 23 | Class-weight formula | ✅ $w_c = N/(C·n_c)$ | v22 |
| 24 | CIDOC-CRM not in Future Work | ✅ E5 added | v22 |
| 25 | GNN future work | ✅ E2 with GAT citation | v22 |
| 26 | RAG future work | ✅ E3 with SPARQL design | v22 |
| 27 | Joint NER+EL+RE | ✅ E4 | v22 |
| 28 | HIPE-2027 | ✅ E6 | v22 |
| 29 | Limitations not separate | ✅ Sec 6 with L1–L4 | v18 |
| 30 | Calibration before motivation | ✅ Forward-ref to Table 2 | v18 |
| 31 | Fig 3/4 redundancy | ✅ Removed duplicate | v16 |
| 32 | Per-language gap stated 3× | ✅ Cross-referenced | v18 |
| 33 | Seeds/hardware/runtime | ✅ All in Sec 4.3 | v16 |

### Self-Review Fixes (Items 34–42, v22–v23)

| # | Issue | Status | Version |
|---|-------|--------|---------|
| 34 | lyu2024 called "survey" | ✅ Now methods paper | v23 |
| 35 | Sec 5.6 +0.018 wrong | ✅ Corrected to +0.030 | v23 |
| 36 | Sec 5.6 "XLM-R advantage" | ✅ Rewrote correctly | v23 |
| 37 | "single-model" → "ensemble" | ✅ "fixed-β ensemble" | v23 |
| 38 | Table 12 R0 ≠ Table 4 | ✅ "independent run" | v23 |
| 39 | Table 9 at ≠ Table 4 | ✅ Calibration note | v23 |
| 40 | Table 6 rounding | ✅ Caption explains | v23 |
| 41 | Algorithm 1 mojibake | ✅ `---` | v23 |
| 42 | Fig 1/2 embedded titles | ✅ Regenerated | v23 |

### v24 Fixes (Items 43–49)

| # | Issue | Status | Version |
|---|-------|--------|---------|
| 43 | Abstract 4.4% → 4.3% | ✅ Corrected | v24 |
| 44 | Running header "Title Suppressed" | ✅ `\titlerunning{}` | v24 |
| 45 | Acknowledgements contradiction | ✅ Reworded | v24 |
| 46 | Ref [1] forward-dated | ✅ "to appear" | v24 |
| 47 | Table 7 MR gap (0.016) | ✅ Epoch difference explained | v24 |
| 48 | Tables 5/9/10/12 rounding | ✅ Disclaimers added | v24 |
| 49 | Fig 3/6 embedded titles | ✅ Regenerated | v24 |

### v25 LLM Audit Fixes (Items 50–56)

| # | Issue | Status | Version |
|---|-------|--------|---------|
| 50 | **Soares text claims numbers in Table 4** | ✅ Removed "Table 4 includes these results", reworded | **v25** |
| 51 | **"strictly necessary" overclaim** (0.0012 < ±0.003 noise) | ✅ Changed to "directionally favored" with variance caveat | **v25** |
| 52 | **β sweep claims 0.1 increments** but only shows 8 rows | ✅ Reworded: "eight representative values spanning the full range" | **v25** |
| 53 | **β range claims 0.3–0.8** but lowest shown is 0.2 | ✅ Corrected to "0.2–0.8" | **v25** |
| 54 | **RLAE τ thresholds never shown** (9 of 15 params hidden) | ✅ Table 9 expanded with τ columns per language per relation | **v25** |
| 55 | **0.5525 vs 0.5530 discrepancy** (Fig 4 vs Table 4) | ✅ Table 4 footnote explains independent optimization gap | **v25** |
| 56 | **VIAF + World Historical Gazetteer absent** | ✅ Added E7 to Future Work | **v25** |
| 57 | **OCR experiment = augmentation, not contrastive** | ✅ Clarified in text, contrastive deferred to E2 | **v25** |

---

### v26 Final Journal Prep Fixes (Items 58–63)

| # | Issue | Status | Version |
|---|-------|--------|---------|
| 58 | **Fig 1–6 look like matplotlib defaults** | ✅ Regenerated as publication-grade (Elsevier/IEEE) diagrams | **v26** |
| 59 | **Section 5.3 out of scope justification** | ✅ Re-worded for ATLOP/Xu et al. and marked as planned future work | **v26** |
| 60 | **VIAF and WHG missing from Rel. Work** | ✅ Added to Sec 2.4 | **v26** |
| 61 | **Merge Per-Language / Cross-dataset** | ✅ Merged into unified Section 5.8 'Generalization' | **v26** |
| 62 | **LNCS template limitations** | ✅ Converted to generic `article` class with standard IMRaD format | **v26** |
| 63 | **Reviewing numbers across text/tables** | ✅ Verified all numbers match consistently | **v26** |

---

## 4. Deferred Items (with justification)

| Item | Why | Where |
|---|---|---|
| Journal venue selection | Dr. Jain's decision | Not paper |
| Contrastive OCR training | Separate contribution | E2 in Sec 6 |
| DBpedia | Redundant with Wikidata | Covered by E7 note |

---

## 5. Numerical Cross-Check Table

| What | Value | Where |
|---|---|---|
| MHIPEX-RLAE MR | 0.5767 | Table 4 |
| hmBERT calibrated MR | 0.553 (unrounded: 0.5530) | Table 4 |
| Ablation A4 MR | 0.553 (unrounded: 0.5525) | Table 6 + Fig 4 |
| Gap 0.5530 vs 0.5525 | Explained in Table 4 footnote | ±0.0005 within noise |
| Table 7 hmBERT MR | 0.5689 (20-epoch run) | Table 7 caption |
| Table 12 R0 MR | 0.554 (independent run) | Table 12 caption |
| Run-to-run variance | ±0.003 | Sec 4.3 |
| Bootstrap CI | [0.548, 0.604] | Sec 5.1 |
| Relative over mBERT | 35.1% | Abstract |
| Relative over hmBERT | 4.3% | Abstract + Conclusion |
