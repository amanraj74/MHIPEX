# MHIPEX — Research Roadmap

**Three-level plan for extending the MHIPEX research program**

---

## Level 1: Immediate Extensions (1–2 weeks, no new architecture)

### 1.1 XLM-R Large Baseline
- **What:** Replace `xlm-roberta-base` (278M) with `xlm-roberta-large` (560M)
- **Expected benefit:** +2–4% MR based on NER literature
- **Difficulty:** Low — same code, adjust batch size for memory
- **Implementation:** Modify model name in `kaggle_mhipex_v12_cell2.py`, reduce batch size to 16, increase gradient accumulation to 4

### 1.2 HIPE-2020/2022 Pre-Fine-Tuning
- **What:** Fine-tune hmBERT on HIPE-2020/2022 NER data before relation extraction
- **Expected benefit:** Better entity boundary understanding, +1–3% MR
- **Difficulty:** Medium — need to download and process NER data, create intermediate training step
- **Implementation:** Two-stage training: Stage 1 (NER on HIPE-2022), Stage 2 (RE on HIPE-2026)
- **Note:** HIPE-2020/2022 use IOB NER format, not relation extraction. The transfer is indirect — improving the encoder's understanding of historical entity mentions.

### 1.3 mDeBERTa-v3 Baseline
- **What:** Add `microsoft/mdeberta-v3-base` as a third encoder
- **Expected benefit:** DeBERTa's disentangled attention may better capture entity-relation interactions
- **Difficulty:** Low — drop-in replacement
- **Implementation:** Add as third model in training loop, test as standalone and in 3-way ensemble

### 1.4 Dynamic Ensemble Weighting
- **What:** Replace fixed α=0.60 with language-specific or confidence-based weights
- **Expected benefit:** +0.5–1% MR (German and English have different optimal weights)
- **Difficulty:** Low — grid search over per-language α values
- **Implementation:** Compute ensemble on per-language subsets, optimize α_en, α_fr, α_de separately

---

## Level 2: Advanced Extensions (1–2 months, moderate engineering)

### 2.1 Knowledge-Augmented Input
- **What:** Use Wikidata QIDs (already in the data!) to fetch biographical facts
- **Expected benefit:** Directly addresses L3 (26% of errors from missing world knowledge)
- **Difficulty:** Medium — need API calls, prompt engineering for fact selection
- **Implementation:**
  1. Extract `pers_wikidata_QID` and `loc_wikidata_QID` from JSONL
  2. Query Wikidata for birth place, death place, occupation, coordinates
  3. Append facts to input: `<WIKI> born: Paris, occupation: politician </WIKI>`
  4. Fine-tune with augmented input

### 2.2 Ordinal Regression for `at`
- **What:** Replace 3-class cross-entropy with ordinal regression (FALSE < PROBABLE < TRUE)
- **Expected benefit:** Directly addresses L4 (PROBABLE class confusion)
- **Difficulty:** Medium — custom loss function, threshold adjustment
- **Implementation:** Use CORAL loss (Cao et al., 2020) or simple ordinal encoding with cumulative probabilities

### 2.3 OCR-Aware Training
- **What:** Augment training data with synthetic OCR noise
- **Expected benefit:** Addresses L6 (28% of errors from OCR corruption)
- **Difficulty:** Medium — need OCR error model (character substitution, deletion, merging)
- **Implementation:** Apply random character-level noise (p=0.05) to training inputs, creating 2x augmented dataset

### 2.4 Retrieval-Augmented Context
- **What:** For each (person, location) pair, retrieve relevant Wikipedia/newspaper passages
- **Expected benefit:** Provides external context for knowledge-dependent relations
- **Difficulty:** High — need retrieval index, passage ranking, context fusion
- **Implementation:** Build BM25 index over Wikipedia person articles, retrieve top-3 passages, concatenate with input

---

## Level 3: Research-Grade Extensions (3–6 months, potential publications)

### 3.1 LLM-Assisted Relation Extraction
- **What:** Use GPT-4 / Llama-3 for in-context learning on the HIPE-2026 task
- **Expected benefit:** LLMs have implicit world knowledge that could solve `at` classification
- **Difficulty:** High — prompt engineering, cost management, evaluation at scale
- **Research question:** Can few-shot LLM prompting match fine-tuned BERT for historical RE?
- **Publication potential:** Strong — comparing fine-tuned vs. prompted approaches on historical text

### 3.2 Historical Entity Linking Pipeline
- **What:** Build end-to-end system: NER → Entity Linking → Relation Extraction
- **Expected benefit:** Currently entities are pre-identified; full pipeline enables real-world deployment
- **Difficulty:** Very High — requires NER model, entity disambiguation, coreference resolution
- **Publication potential:** Very Strong — full system paper for CLEF or ACL

### 3.3 Temporal Reasoning Module
- **What:** Dedicated temporal reasoning component for `isAt`
- **Expected benefit:** German `isAt` recall is only 0.557 — temporal reasoning is the bottleneck
- **Difficulty:** High — need temporal expression extraction, event ordering
- **Implementation:** Extract temporal expressions (HeidelTime), create temporal feature vector, add as auxiliary input

### 3.4 Cross-Century Transfer Study
- **What:** Study how model performance degrades across time periods (1800s vs 1900s vs modern)
- **Expected benefit:** Fundamental research contribution on temporal domain shift in NLP
- **Difficulty:** Medium — need time-stratified evaluation
- **Publication potential:** Strong — addresses a fundamental question in historical NLP

### 3.5 Multilingual Knowledge Graph Construction
- **What:** Use MHIPEX predictions to automatically construct person-place knowledge graphs from newspaper archives
- **Expected benefit:** Downstream application that demonstrates real-world value
- **Difficulty:** High — need entity resolution, graph construction, quality assessment
- **Publication potential:** Very Strong — bridges NLP and Digital Humanities

---

## Publication Opportunities

| Venue | Type | Deadline | Fit |
|-------|------|----------|-----|
| CLEF 2026 Working Notes | Workshop paper | Jul 2026 | ★★★★★ Primary target |
| LaTeCH-CLfL (ACL workshop) | Workshop paper | ~Apr 2027 | ★★★★ Historical NLP |
| IJDL (Int'l Journal on Digital Libraries) | Journal | Rolling | ★★★★ Expanded version |
| Language Resources and Evaluation | Journal | Rolling | ★★★ Multilingual focus |
| EMNLP Findings | Conference | Jun 2027 | ★★★ If strong new results |
| Digital Scholarship in the Humanities | Journal | Rolling | ★★★ DH audience |
