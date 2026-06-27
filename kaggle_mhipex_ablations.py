"""
MHIPEX — Three Missing Ablation Experiments for SCI Journal Submission
Run on Kaggle with GPU T4 x2 + Internet enabled
Estimated runtime: ~90 minutes total

Experiment 1: Focal Loss vs. Weighted CE
Experiment 2: Unweighted CE vs. Weighted CE
Experiment 3: Ensemble Weight Sensitivity (β sweep, post-hoc, no retraining)

Usage: Copy into a Kaggle notebook. Make sure Cell 1 (data download + preprocessing)
       from kaggle_mhipex_kg.py has already been run, so that proc/train_v12.jsonl
       and proc/dev_v12.jsonl exist.
"""

import json, os, gc, warnings, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import recall_score, classification_report
from pathlib import Path
from itertools import product as iterproduct

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Paths ──
PROC_DIR = Path("proc")
OUT_DIR = Path("out_ablations")
OUT_DIR.mkdir(exist_ok=True)

AT_MAP = {"FALSE": 0, "PROBABLE": 1, "TRUE": 2}
ISAT_MAP = {"FALSE": 0, "TRUE": 1}
AT_NAMES = ["FALSE", "PROBABLE", "TRUE"]
ISAT_NAMES = ["FALSE", "TRUE"]
SPECIAL_TOKENS = ["<P>", "</P>", "<L>", "</L>", "<DATE>", "</DATE>", "<LANG>", "</LANG>"]

MODEL_NAME = "dbmdz/bert-base-historic-multilingual-cased"

# ══════════════════════════════════════════════════════════════════
#  Shared Components
# ══════════════════════════════════════════════════════════════════

class HIPEDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=256):
        self.data = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        enc = self.tok(d["text"], max_length=self.max_len,
                       truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "at_label": AT_MAP[d["at_label"]],
            "isat_label": ISAT_MAP[d["isat_label"]],
        }


class MHIPEXClassifier(nn.Module):
    def __init__(self, model_name, n_at=3, n_isat=2, dropout=0.15, n_drops=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.n_drops = n_drops
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_drops)])
        self.at_head = nn.Linear(h, n_at)
        self.isat_head = nn.Linear(h, n_isat)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = out.last_hidden_state[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        mean_out = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        h = 0.5 * cls_out + 0.5 * mean_out

        at_logits_sum = torch.zeros(h.size(0), 3, device=h.device)
        isat_logits_sum = torch.zeros(h.size(0), 2, device=h.device)
        for drop in self.dropouts:
            hd = drop(h)
            at_logits_sum += self.at_head(hd)
            isat_logits_sum += self.isat_head(hd)
        return at_logits_sum / self.n_drops, isat_logits_sum / self.n_drops


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) with class weights."""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def train_and_eval(tag, at_loss_fn, isat_loss_fn, epochs=25, lr=8e-6, bs=16):
    """Train hmBERT with given loss functions, return best dev MR + saved probs."""
    print(f"\n{'='*60}")
    print(f"  Training: {tag}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    model = MHIPEXClassifier(MODEL_NAME).to(DEVICE)
    model.encoder.resize_token_embeddings(len(tokenizer))

    train_ds = HIPEDataset(PROC_DIR / "train_v12.jsonl", tokenizer)
    dev_ds = HIPEDataset(PROC_DIR / "dev_v12.jsonl", tokenizer)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2)
    dev_dl = DataLoader(dev_ds, batch_size=bs * 2, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dl) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.12 * total_steps), total_steps)

    best_mr = 0
    patience, max_patience = 0, 8

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dl:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            at_y = batch["at_label"].to(DEVICE)
            isat_y = batch["isat_label"].to(DEVICE)

            at_logits, isat_logits = model(ids, mask)
            loss = at_loss_fn(at_logits, at_y) + isat_loss_fn(isat_logits, isat_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        all_at_true, all_at_pred = [], []
        all_is_true, all_is_pred = [], []
        all_at_probs, all_is_probs = [], []

        with torch.no_grad():
            for batch in dev_dl:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                at_logits, isat_logits = model(ids, mask)
                at_probs = torch.softmax(at_logits, dim=-1)
                is_probs = torch.softmax(isat_logits, dim=-1)

                all_at_true.extend(batch["at_label"].tolist())
                all_at_pred.extend(at_probs.argmax(dim=-1).cpu().tolist())
                all_is_true.extend(batch["isat_label"].tolist())
                all_is_pred.extend(is_probs.argmax(dim=-1).cpu().tolist())
                all_at_probs.extend(at_probs.cpu().numpy())
                all_is_probs.extend(is_probs.cpu().numpy())

        at_mr = recall_score(all_at_true, all_at_pred, average="macro", zero_division=0)
        is_mr = recall_score(all_is_true, all_is_pred, average="macro", zero_division=0)
        mr = round((at_mr + is_mr) / 2, 4)

        print(f"  Epoch {epoch+1:2d} | Loss: {total_loss/len(train_dl):.4f} | MR: {mr:.4f} (at={at_mr:.4f}, isAt={is_mr:.4f})")

        if mr > best_mr:
            best_mr = mr
            patience = 0
            save_dir = OUT_DIR / tag
            save_dir.mkdir(exist_ok=True)
            torch.save({
                "probs_at": torch.tensor(np.array(all_at_probs)),
                "probs_isat": torch.tensor(np.array(all_is_probs)),
                "at_true": all_at_true, "is_true": all_is_true,
                "at_pred": all_at_pred, "is_pred": all_is_pred,
            }, save_dir / "best_probs.pt")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"  Best MR: {best_mr:.4f}")
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return best_mr


def calibrate(tag):
    """Post-hoc threshold calibration on saved probabilities."""
    data = torch.load(OUT_DIR / tag / "best_probs.pt", weights_only=True)
    probs_at = data["probs_at"].numpy()
    probs_isat = data["probs_isat"].numpy()
    at_true = data["at_true"]
    is_true = data["is_true"]

    # AT calibration
    best_at_mr, best_at_preds = 0, []
    prob_range = np.arange(0.20, 0.55, 0.05)
    for t_prob, t_true in iterproduct(prob_range, prob_range):
        preds = []
        for p in probs_at:
            if p[2] >= t_true: preds.append(2)
            elif p[1] >= t_prob: preds.append(1)
            else: preds.append(0)
        mr = recall_score(at_true, preds, average="macro", zero_division=0)
        if mr > best_at_mr:
            best_at_mr = mr
            best_at_preds = preds

    # isAt calibration
    best_isat_mr = 0
    for t in np.arange(0.15, 0.60, 0.05):
        preds = []
        for i, p in enumerate(probs_isat):
            if best_at_preds[i] == 0: preds.append(0)
            elif p[1] >= t: preds.append(1)
            else: preds.append(0)
        mr = recall_score(is_true, preds, average="macro", zero_division=0)
        if mr > best_isat_mr:
            best_isat_mr = mr

    total_mr = round((best_at_mr + best_isat_mr) / 2, 4)
    return {"mr": total_mr, "at_mr": round(best_at_mr, 4), "isat_mr": round(best_isat_mr, 4)}


# ══════════════════════════════════════════════════════════════════
#  EXPERIMENT 1: Focal Loss vs. Weighted CE
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 64)
print("  EXPERIMENT 1: Loss Function Comparison")
print("=" * 64)

# 1a: Weighted CE (our default)
at_w = torch.tensor([0.80, 1.50, 2.40], device=DEVICE)
isat_w = torch.tensor([0.70, 2.60], device=DEVICE)
at_ce = nn.CrossEntropyLoss(weight=at_w)
isat_ce = nn.CrossEntropyLoss(weight=isat_w)
train_and_eval("weighted_ce", at_ce, isat_ce)

# 1b: Focal Loss (gamma=2, same class weights)
at_focal = FocalLoss(weight=at_w.clone(), gamma=2.0)
isat_focal = FocalLoss(weight=isat_w.clone(), gamma=2.0)
train_and_eval("focal_loss", at_focal, isat_focal)


# ══════════════════════════════════════════════════════════════════
#  EXPERIMENT 2: Unweighted CE vs. Weighted CE
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 64)
print("  EXPERIMENT 2: Class Weight Ablation")
print("=" * 64)

# 2a: Unweighted CE
at_unw = nn.CrossEntropyLoss()
isat_unw = nn.CrossEntropyLoss()
train_and_eval("unweighted_ce", at_unw, isat_unw)

# (Weighted CE was already trained as "weighted_ce" above)


# ══════════════════════════════════════════════════════════════════
#  EXPERIMENT 3: Ensemble Weight Sensitivity (Post-hoc)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 64)
print("  EXPERIMENT 3: Ensemble β Sensitivity")
print("=" * 64)

# Load saved probabilities from the MAIN experiments (out/ directory from original training)
# If those don't exist, use the weighted_ce run from Exp 1 + create an XLM-R placeholder
MAIN_OUT = Path("out")
hmbert_path = MAIN_OUT / "hmbert_v12" / "best_probs.pt"
xlmr_path = MAIN_OUT / "xlmr_v12" / "best_probs.pt"

if hmbert_path.exists() and xlmr_path.exists():
    hm_data = torch.load(hmbert_path, weights_only=True)
    xr_data = torch.load(xlmr_path, weights_only=True)

    hm_at = hm_data["probs_at"].numpy()
    hm_is = hm_data["probs_isat"].numpy()
    xr_at = xr_data["probs_at"].numpy()
    xr_is = xr_data["probs_isat"].numpy()
    at_true = hm_data["at_true"]
    is_true = hm_data["is_true"]

    betas = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    beta_results = []

    for beta in betas:
        ens_at = beta * hm_at + (1 - beta) * xr_at
        ens_is = beta * hm_is + (1 - beta) * xr_is

        # Calibrate
        best_at_mr, best_at_preds = 0, []
        prob_range = np.arange(0.20, 0.55, 0.05)
        for t_prob, t_true in iterproduct(prob_range, prob_range):
            preds = []
            for p in ens_at:
                if p[2] >= t_true: preds.append(2)
                elif p[1] >= t_prob: preds.append(1)
                else: preds.append(0)
            mr = recall_score(at_true, preds, average="macro", zero_division=0)
            if mr > best_at_mr:
                best_at_mr = mr
                best_at_preds = preds

        best_isat_mr = 0
        for t in np.arange(0.15, 0.60, 0.05):
            preds = []
            for i, p in enumerate(ens_is):
                if best_at_preds[i] == 0: preds.append(0)
                elif p[1] >= t: preds.append(1)
                else: preds.append(0)
            mr = recall_score(is_true, preds, average="macro", zero_division=0)
            if mr > best_isat_mr:
                best_isat_mr = mr

        total = round((best_at_mr + best_isat_mr) / 2, 4)
        beta_results.append((beta, total, round(best_at_mr, 4), round(best_isat_mr, 4)))
        print(f"  β={beta:.2f} | MR: {total:.4f} | at: {best_at_mr:.4f} | isAt: {best_isat_mr:.4f}")

    print("\n  β Sensitivity Table:")
    print(f"  {'β':>5s}  {'MR':>7s}  {'at':>7s}  {'isAt':>7s}")
    for b, mr, at, isat in beta_results:
        marker = " *" if b == 0.60 else ""
        print(f"  {b:5.2f}  {mr:7.4f}  {at:7.4f}  {isat:7.4f}{marker}")
else:
    print("  WARNING: Main experiment probs not found in out/")
    print("  Run the original training pipeline first (kaggle_mhipex_v12_cell1.py + cell2.py)")
    print("  Then re-run this script for Experiment 3.")


# ══════════════════════════════════════════════════════════════════
#  Results Summary
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 64)
print("  ABLATION RESULTS SUMMARY")
print("=" * 64)

# Calibrate all trained models
for tag, label in [
    ("weighted_ce", "Weighted CE"),
    ("focal_loss", "Focal Loss (γ=2)"),
    ("unweighted_ce", "Unweighted CE"),
]:
    path = OUT_DIR / tag / "best_probs.pt"
    if path.exists():
        cal = calibrate(tag)
        print(f"  {label:25s} | MR: {cal['mr']:.4f} | at: {cal['at_mr']:.4f} | isAt: {cal['isat_mr']:.4f}")

# Save results
import csv
csv_path = OUT_DIR / "ablation_results.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Experiment", "MR", "at_recall", "isAt_recall"])
    for tag, label in [("weighted_ce", "Weighted CE"), ("focal_loss", "Focal Loss"), ("unweighted_ce", "Unweighted CE")]:
        path = OUT_DIR / tag / "best_probs.pt"
        if path.exists():
            cal = calibrate(tag)
            w.writerow([label, cal["mr"], cal["at_mr"], cal["isat_mr"]])

print(f"\n  Results saved to: {csv_path}")
print(f"  Download the 'out_ablations/' folder from Kaggle output!")
