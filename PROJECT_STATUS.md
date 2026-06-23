# MHIPEX — Project Status

**Last Updated:** June 20, 2026

---

## Deliverable Status

| Deliverable | Status | Notes |
|------------|--------|-------|
| Research proposal | ✅ Complete | Approved by Dr. Jain |
| Dataset pipeline | ✅ Complete | v12 enriched format |
| mBERT baseline | ✅ Complete | MR = 0.427 |
| hmBERT experiments | ✅ Complete | MR = 0.553 (calibrated) |
| XLM-R experiments | ✅ Complete | MR = 0.545 (calibrated) |
| Ensemble (MHIPEX) | ✅ Complete | **MR = 0.5655** |
| Ablation study | ✅ Complete | 6-row cumulative ablation |
| Cross-dataset validation | ✅ Complete | 4 experiments (C1–C4) |
| Error analysis | ✅ Complete | 4 error categories identified |
| Analysis figures | ✅ Complete | 6 publication-quality figures |
| Paper draft (v1) | ✅ Complete | 11 sections, submitted to Dr. Jain |
| Paper revision (v2) | ✅ Complete | 7 sections (merged), limitations added |
| Architecture diagram | ✅ Complete | figures/architecture.png |
| Project documentation | ✅ Complete | README, PRODUCT.md, etc. |

## Pending Items

| Item | Priority | Estimated Effort |
|------|----------|-----------------|
| Upload to Overleaf for final PDF | 🔴 High | 30 min |
| Official test set evaluation | 🔴 High | When released by organizers |
| XLM-R Large experiments | 🟡 Medium | 2 hours (Kaggle) |
| HIPE-2020/2022 transfer learning | 🟡 Medium | 4 hours |
| Camera-ready paper | 🟡 Medium | After Dr. Jain's final feedback |

## Supervisor Feedback Log

| Date | Feedback | Action Taken |
|------|----------|-------------|
| May 2026 | "Proceed with paper draft" | Started paper writing |
| Jun 13 | "Should have baselines, SOTA, ablation, cross-dataset" | Added all four |
| Jun 16 | "Merge sections, add diagram, add limitations" | Restructured to 7 sections, added Figure 1, added L1–L7 |
| Jun 20 | *(awaiting next feedback)* | — |

## Key Metrics

| Metric | Value | Context |
|--------|-------|---------|
| Best MR (ensemble) | 0.5655 | On sandbox dev set |
| Best MR (cross-domain) | 0.588 | Sandbox → Newspaper v1.0 |
| Improvement over mBERT | +32.5% | Relative improvement |
| Training time | ~35 min/model | Kaggle T4 x2 |
| Paper length | 15 pages | LNCS format |
| References | 12 | Peer-reviewed venues |
