# MHIPEX — Next Steps

**Priority-ordered action items as of June 20, 2026**

---

## Immediate (This Week)

### 1. Upload Paper to Overleaf
- Upload `paper/main.tex` + `paper/figures/architecture.png`
- Compile → verify Figure 1 renders correctly
- Download final PDF with embedded diagram
- **Send to Dr. Jain for review**

### 2. Await Dr. Jain's Feedback on v2
- She may request additional experiments or writing changes
- Be ready to run experiments on Kaggle within 24 hours

---

## Short-Term (Next 2 Weeks)

### 3. XLM-R Large Experiment
- **Why:** Dr. Jain asked for SOTA comparison; XLM-R Large (560M params) is the obvious next baseline
- **How:** Modify `kaggle_mhipex_v12_cell2.py` to use `xlm-roberta-large`
- **Expected:** 2–3% MR improvement; strengthen the paper's comparison
- **Risk:** May require gradient accumulation due to memory (16GB T4)

### 4. Official Test Set Submission
- When HIPE-2026 releases the test set, run inference immediately
- Use the saved ensemble model from v12 training
- Submit predictions in the required format

### 5. Camera-Ready Paper
- Incorporate Dr. Jain's final feedback
- Proofread carefully — check all table values match actual results
- Ensure all references have complete bibliographic info

---

## Medium-Term (Next Month)

### 6. Transfer Learning Experiments
- Download HIPE-2020/2022 NER data
- Pre-fine-tune hmBERT on NER before relation extraction
- Expected benefit: better entity boundary understanding

### 7. Knowledge Graph Integration (Proof of Concept)
- Link person/location entities to Wikidata QIDs (already in the data)
- Fetch biographical facts → append to input
- Test whether world knowledge reduces `at` errors

### 8. Additional Publication Venues
- CLEF 2026 Working Notes (primary)
- Consider expanded version for journal submission (IJDL, LRE)
- Workshop papers at ACL/EMNLP on historical NLP

---

## Decision Points

| Decision | Owner | Deadline |
|----------|-------|----------|
| Paper structure finalized? | Dr. Jain | After v2 review |
| Run XLM-R Large? | Aman | After Dr. Jain confirms |
| Submit to additional venues? | Dr. Jain | After CLEF deadline |
| Expand to journal paper? | Both | Q3 2026 |
