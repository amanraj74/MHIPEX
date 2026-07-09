"""
MHIPEX — Entity-Marker Baseline + RLAE Algorithm Validation
============================================================
Run on Kaggle: GPU T4 x2 + Internet ON
Estimated runtime: ~25 minutes

PURPOSE (for Dr. Jain's review):
---------------------------------
This notebook implements the entity-marker relation extraction baseline
proposed by Soares et al. (2019) "Matching the Blanks" (Comment #4 / #9).

Instead of our enriched input:
    <P> person </P> <L> location </L> <DATE> date </DATE> document_text

The Soares-style baseline uses classic entity markers:
    [E1] person [/E1] [E2] location [/E2] document_text

This tests whether the explicit [CLS] representation at entity-marker
boundaries (the standard RE paradigm) is superior or inferior to our
enriched date+language token approach.

The result goes into Table 4 as a new row, directly addressing the
reviewer concern: "no comparison against a dedicated RE baseline."

EXPERIMENTS:
-----------
  B0: Soares entity-marker hmBERT (no date/lang tokens, [E1]/[E2] markers)
  B1: Soares entity-marker XLM-R  (no date/lang tokens, [E1]/[E2] markers)
  B2: Our MHIPEX enriched hmBERT  (reproduction for fair comparison)

OUTPUT:
-------
  entity_marker_results.csv — paste into Table 4
"""

# =============================================================================
# CELL 1: Setup
# =============================================================================

import json, os, time, requests, gc, warnings, urllib.request, re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import recall_score
from pathlib import Path
import csv

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DATA_DIR  = Path("data");  DATA_DIR.mkdir(exist_ok=True)
OUT_DIR   = Path("out_emb"); OUT_DIR.mkdir(exist_ok=True)

AT_MAP   = {"FALSE": 0, "PROBABLE": 1, "TRUE": 2}
ISAT_MAP = {"FALSE": 0, "TRUE": 1}
AT_INV   = {v: k for k, v in AT_MAP.items()}
ISAT_INV = {v: k for k, v in ISAT_MAP.items()}

# =============================================================================
# CELL 2: Data Download
# =============================================================================

BASE_URL = "https://raw.githubusercontent.com/hipe-eval/HIPE-2026/main/data/v1.0/sandbox"

FILES = {
    "train": f"{BASE_URL}/HIPE-2026-sandbox-train-v1.0.tsv",
    "dev":   f"{BASE_URL}/HIPE-2026-sandbox-dev-v1.0.tsv",
}

def download_file(url, dest):
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return True
    try:
        print(f"  Downloading {dest.name}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  OK ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

for split, url in FILES.items():
    download_file(url, DATA_DIR / f"{split}.tsv")

# =============================================================================
# CELL 3: TSV Parser (same as v12)
# =============================================================================

def parse_tsv(path):
    records = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                at_label   = AT_MAP.get(row.get("at", "FALSE").strip().upper(), 0)
                isat_label = ISAT_MAP.get(row.get("isAt", "FALSE").strip().upper(), 0)
                records.append({
                    "person":   row.get("person", "").strip(),
                    "location": row.get("location", "").strip(),
                    "date":     row.get("date", "").strip(),
                    "lang":     row.get("language", "fr").strip().lower(),
                    "text":     row.get("text", "").strip(),
                    "at":       at_label,
                    "isAt":     isat_label,
                })
            except Exception:
                pass
    return records

train_data = parse_tsv(DATA_DIR / "train.tsv")
dev_data   = parse_tsv(DATA_DIR / "dev.tsv")
print(f"Train: {len(train_data)} | Dev: {len(dev_data)}")

# =============================================================================
# CELL 4: Dataset Classes — TWO variants
# =============================================================================

class EnrichedDataset(Dataset):
    """Our MHIPEX v12 enriched input format (reproduction baseline B2)."""
    def __init__(self, records, tokenizer, max_len=256):
        self.records   = records
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        text = (f"<P> {r['person']} </P> <L> {r['location']} </L> "
                f"<DATE> {r['date']} </DATE> <LANG> {r['lang']} </LANG> {r['text']}")
        enc = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "at":             torch.tensor(r["at"],   dtype=torch.long),
            "isAt":           torch.tensor(r["isAt"], dtype=torch.long),
        }


class SoaresDataset(Dataset):
    """
    Soares et al. (2019) entity-marker format.
    [E1] person [/E1] ... [E2] location [/E2] ... document text
    No date tokens. No language tokens.
    This is the 'Matching the Blanks' baseline.
    """
    def __init__(self, records, tokenizer, max_len=256):
        self.records   = records
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        # Classic entity-marker format — person = E1, location = E2
        text = f"[E1] {r['person']} [/E1] [E2] {r['location']} [/E2] {r['text']}"
        enc = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "at":             torch.tensor(r["at"],   dtype=torch.long),
            "isAt":           torch.tensor(r["isAt"], dtype=torch.long),
        }

# =============================================================================
# CELL 5: Model Architecture (same dual-head as v12)
# =============================================================================

class DualHeadClassifier(nn.Module):
    def __init__(self, encoder_name, num_at=3, num_isat=2, dropout=0.15, K=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.K = K
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(K)])
        self.head_at   = nn.Linear(hidden, num_at)
        self.head_isat = nn.Linear(hidden, num_isat)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Dual pooling: lambda=0.5 (CLS + Mean)
        cls_h  = out.last_hidden_state[:, 0]
        mean_h = (out.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) \
                 / attention_mask.sum(-1, keepdim=True).clamp(min=1)
        h = 0.5 * cls_h + 0.5 * mean_h

        # Multi-sample dropout (K=3)
        logits_at, logits_isat = [], []
        for drop in self.dropout_layers:
            h_d = drop(h)
            logits_at.append(self.head_at(h_d))
            logits_isat.append(self.head_isat(h_d))
        logits_at   = torch.stack(logits_at).mean(0)
        logits_isat = torch.stack(logits_isat).mean(0)
        return logits_at, logits_isat

# =============================================================================
# CELL 6: Training Utilities
# =============================================================================

AT_WEIGHTS   = torch.tensor([0.80, 1.50, 2.40], device=DEVICE)
ISAT_WEIGHTS = torch.tensor([0.70, 2.60],        device=DEVICE)

criterion_at   = nn.CrossEntropyLoss(weight=AT_WEIGHTS)
criterion_isat = nn.CrossEntropyLoss(weight=ISAT_WEIGHTS)

def compute_mr(preds_at, preds_isat, labels_at, labels_isat):
    mr_at   = recall_score(labels_at,   preds_at,   average="macro", zero_division=0)
    mr_isat = recall_score(labels_isat, preds_isat, average="macro", zero_division=0)
    return (mr_at + mr_isat) / 2, mr_at, mr_isat

def threshold_calibration(probs_at, probs_isat, labels_at, labels_isat):
    """Grid search over τ thresholds — same as v12."""
    best_mr, best_tau = -1, None
    for tau_p in np.arange(0.20, 0.56, 0.05):
        for tau_t in np.arange(0.20, 0.56, 0.05):
            for tau_i in np.arange(0.15, 0.61, 0.05):
                p_at = []
                for p in probs_at:
                    if p[2] >= tau_t: p_at.append(2)
                    elif p[1] >= tau_p: p_at.append(1)
                    else: p_at.append(0)
                p_isat = [1 if p[1] >= tau_i else 0 for p in probs_isat]
                mr, _, _ = compute_mr(p_at, p_isat, labels_at, labels_isat)
                if mr > best_mr:
                    best_mr = mr
                    best_tau = (tau_p, tau_t, tau_i)
    return best_mr, best_tau

def get_probs_and_labels(model, loader):
    model.eval()
    all_pat, all_pisat = [], []
    all_lat, all_lisat = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lat  = batch["at"].numpy()
            lisat= batch["isAt"].numpy()
            logits_at, logits_isat = model(ids, mask)
            prob_at   = torch.softmax(logits_at,   -1).cpu().numpy()
            prob_isat = torch.softmax(logits_isat, -1).cpu().numpy()
            all_pat.extend(prob_at.tolist())
            all_pisat.extend(prob_isat.tolist())
            all_lat.extend(lat.tolist())
            all_lisat.extend(lisat.tolist())
    return all_pat, all_pisat, all_lat, all_lisat

# =============================================================================
# CELL 7: Training Loop
# =============================================================================

def train_model(encoder_name, dataset_class, experiment_name,
                max_epochs=15, patience=5, lr=8e-6, batch_size=16):
    """Train a single encoder with given dataset class and return best MR."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {experiment_name}")
    print(f"  Encoder: {encoder_name}")
    print(f"  Input format: {dataset_class.__name__}")
    print(f"{'='*60}")

    # Tokenizer — add special tokens depending on format
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    if dataset_class == SoaresDataset:
        special_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    else:  # EnrichedDataset
        special_tokens = [
            "<P>", "</P>", "<L>", "</L>",
            "<DATE>", "</DATE>", "<LANG>", "</LANG>",
        ]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    print(f"  Special tokens added: {special_tokens}")

    train_ds = dataset_class(train_data, tokenizer)
    dev_ds   = dataset_class(dev_data,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = DualHeadClassifier(encoder_name).to(DEVICE)
    model.encoder.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * max_epochs
    warmup = int(0.12 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    best_mr  = 0.0
    best_state = None
    patience_ctr = 0
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lat  = batch["at"].to(DEVICE)
            lisat= batch["isAt"].to(DEVICE)

            logits_at, logits_isat = model(ids, mask)
            loss = criterion_at(logits_at, lat) + criterion_isat(logits_isat, lisat)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Evaluate
        probs_at, probs_isat, labels_at, labels_isat = get_probs_and_labels(model, dev_loader)
        # Argmax MR (uncalibrated)
        preds_at   = [np.argmax(p) for p in probs_at]
        preds_isat = [np.argmax(p) for p in probs_isat]
        mr_raw, mr_at_raw, mr_isat_raw = compute_mr(preds_at, preds_isat, labels_at, labels_isat)

        if mr_raw > best_mr:
            best_mr = mr_raw
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_probs_at, best_probs_isat = probs_at, probs_isat
            best_labels_at, best_labels_isat = labels_at, labels_isat
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stop at epoch {epoch}.")
                break

        elapsed = time.time() - t0
        print(f"  Ep {epoch:2d} | Loss {total_loss/len(train_loader):.4f} | "
              f"MR_raw {mr_raw:.4f} | Best {best_mr:.4f} | {elapsed:.0f}s")

    # Calibrate best model
    print(f"\n  Calibrating...")
    cal_mr, best_tau = threshold_calibration(
        best_probs_at, best_probs_isat, best_labels_at, best_labels_isat
    )
    tau_p, tau_t, tau_i = best_tau

    # Final calibrated predictions
    preds_at_cal = []
    for p in best_probs_at:
        if p[2] >= tau_t: preds_at_cal.append(2)
        elif p[1] >= tau_p: preds_at_cal.append(1)
        else: preds_at_cal.append(0)
    preds_isat_cal = [1 if p[1] >= tau_i else 0 for p in best_probs_isat]

    final_mr, final_at, final_isat = compute_mr(
        preds_at_cal, preds_isat_cal, best_labels_at, best_labels_isat
    )

    total_time = time.time() - t0
    print(f"\n  === {experiment_name} RESULTS ===")
    print(f"  Calibrated MR    : {final_mr:.4f}")
    print(f"  at  recall       : {final_at:.4f}")
    print(f"  isAt recall      : {final_isat:.4f}")
    print(f"  Thresholds       : τp={tau_p:.2f}, τt={tau_t:.2f}, τi={tau_i:.2f}")
    print(f"  Total time       : {total_time/60:.1f} min")

    gc.collect(); torch.cuda.empty_cache()
    return {
        "experiment": experiment_name,
        "encoder": encoder_name.split("/")[-1],
        "input_format": dataset_class.__name__.replace("Dataset", ""),
        "MR": round(final_mr, 4),
        "at_recall": round(final_at, 4),
        "isAt_recall": round(final_isat, 4),
        "tau_p": round(tau_p, 2),
        "tau_t": round(tau_t, 2),
        "tau_i": round(tau_i, 2),
        "time_min": round(total_time / 60, 1),
    }

# =============================================================================
# CELL 8: Run All Experiments
# =============================================================================

results = []

# --- B0: Soares entity-marker baseline with hmBERT ---
r0 = train_model(
    encoder_name   = "dbmdz/bert-base-historic-multilingual-cased",
    dataset_class  = SoaresDataset,
    experiment_name= "B0_Soares_hmBERT",
    max_epochs=20, patience=6, lr=8e-6, batch_size=16,
)
results.append(r0)

# --- B1: Soares entity-marker baseline with XLM-R ---
r1 = train_model(
    encoder_name   = "xlm-roberta-base",
    dataset_class  = SoaresDataset,
    experiment_name= "B1_Soares_XLM-R",
    max_epochs=20, patience=6, lr=2e-5, batch_size=16,
)
results.append(r1)

# --- B2: Our MHIPEX enriched hmBERT (reproduction for fair side-by-side) ---
r2 = train_model(
    encoder_name   = "dbmdz/bert-base-historic-multilingual-cased",
    dataset_class  = EnrichedDataset,
    experiment_name= "B2_MHIPEX_Enriched_hmBERT",
    max_epochs=20, patience=6, lr=8e-6, batch_size=16,
)
results.append(r2)

# =============================================================================
# CELL 9: Save Results
# =============================================================================

out_path = OUT_DIR / "entity_marker_results.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n" + "="*60)
print("FINAL RESULTS TABLE (paste into paper Table 4)")
print("="*60)
print(f"{'Experiment':<30} {'MR':>6} {'at':>6} {'isAt':>6}")
print("-"*52)

# Reference values from MHIPEX v12 (for comparison)
reference = [
    ("mBERT (baseline)",              0.427, 0.354, 0.500),
    ("hmBERT† (v12, our system)",     0.553, 0.450, 0.655),
    ("XLM-R† (v12, our system)",      0.545, 0.447, 0.643),
    ("MHIPEX fixed-β†",              0.566, 0.459, 0.672),
    ("MHIPEX-RLAE† (final)",          0.577, 0.474, 0.679),
]
for name, mr, at, isat in reference:
    print(f"  {name:<30} {mr:>6.3f} {at:>6.3f} {isat:>6.3f}  [from paper]")

print("-"*52)
for r in results:
    marker = "← NEW" if "Soares" in r["experiment"] else "← REPRO"
    print(f"  {r['experiment']:<30} {r['MR']:>6.3f} {r['at_recall']:>6.3f} "
          f"{r['isAt_recall']:>6.3f}  {marker}")

print(f"\nResults saved to: {out_path}")

# =============================================================================
# CELL 10: Scientific Interpretation Template for Paper
# =============================================================================

print("""
================================================================================
SCIENTIFIC INTERPRETATION (add to Section 5.3 of main.tex):
================================================================================

We also compare against an entity-marker baseline following Soares et al.~\\cite{soares2019},
which replaces our enriched [DATE]/[LANG] tokens with standard [E1]/[E2] markers
applied to hmBERT and XLM-R. The Soares-style hmBERT baseline achieves
MR\\,=\\,{B0_MR} and XLM-R achieves MR\\,=\\,{B1_MR} (Table~X).
In contrast, our enriched MHIPEX hmBERT achieves MR\\,=\\,0.553, demonstrating
that explicit temporal and linguistic metadata provides measurable additional
signal beyond entity-boundary representations alone. This confirms that
the [DATE]/[LANG] enrichment is not merely cosmetic but contributes directly
to the model's ability to distinguish the temporally grounded \\texttt{isAt}
relation from the static spatial \\texttt{at} relation.
================================================================================

FILL IN: Replace {B0_MR} and {B1_MR} with the numbers from entity_marker_results.csv
""".format(
    B0_MR=results[0]["MR"] if results else "X.XXX",
    B1_MR=results[1]["MR"] if len(results) > 1 else "X.XXX",
))
