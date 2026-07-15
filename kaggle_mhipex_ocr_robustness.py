"""
OCR Noise Robustness Experiment for MHIPEX
==========================================
Experiment design (per Dr. Jain Comments.pdf, page 12):
  - R0: Standard hmBERT (no OCR augmentation) — text-only baseline
  - R1: hmBERT evaluated on synthetically corrupted dev set
  - R2: hmBERT trained WITH OCR noise augmentation, evaluated on clean dev
  - R3: hmBERT trained WITH OCR noise augmentation, evaluated on corrupted dev

Results go into Table 12 of MHIPEX paper.
Common OCR substitutions used: rn→m, l→1, c→e, f→f(long-s), ii→n, etc.
"""

import os, json, re, gc, random, csv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"]     = "expandable_segments:True"

import subprocess
subprocess.run(["pip","install","-q","transformers==4.44.2","scikit-learn","tqdm"], check=True)

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import recall_score
from pathlib import Path
from tqdm import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} | GPUs: {torch.cuda.device_count()}")

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
OUT_DIR  = Path("out_ocr"); OUT_DIR.mkdir(exist_ok=True)

HM_MODEL = "dbmdz/bert-base-historic-multilingual-cased"
MAX_LEN   = 256
BATCH     = 16
EPOCHS    = 20
LR        = 8e-6
PATIENCE  = 6

# ─── OCR Noise Functions ───────────────────────────────────────────────────────
# Common historical OCR substitutions (verified against literature)
OCR_SUBS = [
    ("rn", "m"),   # rn→m (most common)
    ("m",  "rn"),  # reverse
    ("l",  "1"),   # l→1
    ("1",  "l"),
    ("c",  "e"),   # c→e
    ("e",  "c"),
    ("ii", "n"),   # ii→n
    ("u",  "n"),   # u→n
    ("n",  "u"),
    ("a",  "o"),   # a→o
    ("o",  "a"),
    ("cl", "d"),   # cl→d
    ("vv", "w"),   # vv→w
    ("f",  "s"),   # long-s confusion
]

def apply_ocr_noise(text: str, prob: float = 0.05) -> str:
    """Apply synthetic OCR character-level corruption with probability `prob` per word."""
    words = text.split()
    noisy = []
    for word in words:
        if random.random() < prob and len(word) > 2:
            # pick a random substitution
            sub_from, sub_to = random.choice(OCR_SUBS)
            if sub_from in word.lower():
                idx = word.lower().find(sub_from)
                word = word[:idx] + sub_to + word[idx+len(sub_from):]
        noisy.append(word)
    return " ".join(noisy)

# ─── Data Loading ──────────────────────────────────────────────────────────────
LANGS = ["en", "fr", "de"]
AT_LABELS   = {"false": 0, "probable": 1, "true": 2}
ISAT_LABELS = {"false": 0, "true": 1}

def load_hipe_jsonl(split: str, noise_prob: float = 0.0):
    records = []
    for lang in LANGS:
        # Try Kaggle dataset paths first
        candidates = [
            Path(f"/kaggle/input/hipe2026/{lang}_{split}.jsonl"),
            Path(f"/kaggle/input/hipe-2026-sandbox/{lang}_{split}.jsonl"),
            DATA_DIR / f"{lang}_{split}.jsonl",
        ]
        fpath = next((p for p in candidates if p.exists()), None)
        if fpath is None:
            print(f"  WARNING: {lang}_{split}.jsonl not found, skipping")
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", obj.get("sentence", ""))
                if noise_prob > 0:
                    text = apply_ocr_noise(text, prob=noise_prob)
                person   = obj.get("person",   obj.get("entity1", ""))
                location = obj.get("location", obj.get("entity2", ""))
                date     = obj.get("date",     obj.get("pub_date", ""))
                at_raw   = str(obj.get("at",   obj.get("at_label",   "false"))).lower()
                isat_raw = str(obj.get("isAt", obj.get("isat_label", "false"))).lower()
                records.append({
                    "text": text, "person": person, "location": location,
                    "date": date, "lang": lang,
                    "at":   AT_LABELS.get(at_raw, 0),
                    "isAt": ISAT_LABELS.get(isat_raw, 0),
                })
    return records

# ─── Dataset ──────────────────────────────────────────────────────────────────
class MHIPEXDataset(Dataset):
    def __init__(self, records, tokenizer, max_len, augment_noise=0.0):
        self.records = records
        self.tok = tokenizer
        self.max_len = max_len
        self.aug = augment_noise

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        text = apply_ocr_noise(r["text"], self.aug) if self.aug > 0 else r["text"]
        seq = (f"<P> {r['person']} </P> <L> {r['location']} </L> "
               f"<DATE> {r['date']} </DATE> <LANG> {r['lang']} </LANG> {text}")
        enc = self.tok(seq, max_length=self.max_len, padding="max_length",
                       truncation=True, return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "at":   torch.tensor(r["at"],   dtype=torch.long),
            "isAt": torch.tensor(r["isAt"], dtype=torch.long),
        }

# ─── Model ────────────────────────────────────────────────────────────────────
class MHIPEXModel(nn.Module):
    def __init__(self, model_name, dropout=0.15):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.at_head   = nn.Linear(h, 3)
        self.isat_head = nn.Linear(h, 2)

    def forward(self, input_ids, attention_mask):
        out  = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls  = out.last_hidden_state[:, 0]
        mean = (out.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) \
               / attention_mask.sum(1, keepdim=True)
        h = self.drop(0.5 * cls + 0.5 * mean)
        return self.at_head(h), self.isat_head(h)

# ─── Training & Evaluation ────────────────────────────────────────────────────
AT_W   = torch.tensor([0.80, 1.50, 2.40], device=DEVICE)
ISAT_W = torch.tensor([0.70, 2.60],       device=DEVICE)

def macro_recall(y_true, y_pred, n_classes):
    return recall_score(y_true, y_pred, average="macro",
                        labels=list(range(n_classes)), zero_division=0)

def train_one_epoch(model, loader, opt, sched, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, leave=False):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        at_y = batch["at"].to(DEVICE)
        is_y = batch["isAt"].to(DEVICE)
        opt.zero_grad()
        with autocast("cuda"):
            at_l, is_l = model(ids, mask)
            loss = (nn.CrossEntropyLoss(weight=AT_W)(at_l, at_y) +
                    nn.CrossEntropyLoss(weight=ISAT_W)(is_l, is_y))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(opt); scaler.update(); sched.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    at_true, at_pred, is_true, is_pred = [], [], [], []
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        at_l, is_l = model(ids, mask)
        at_true.extend(batch["at"].tolist())
        is_true.extend(batch["isAt"].tolist())
        at_pred.extend(at_l.argmax(-1).cpu().tolist())
        is_pred.extend(is_l.argmax(-1).cpu().tolist())
    mr_at   = macro_recall(at_true,   at_pred,   3)
    mr_isat = macro_recall(is_true,   is_pred,   2)
    return (mr_at + mr_isat) / 2, mr_at, mr_isat

def run_experiment(name, train_noise=0.0, eval_noise=0.0):
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"  Train noise={train_noise:.0%} | Eval noise={eval_noise:.0%}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(HM_MODEL)
    special = ["<P>","</P>","<L>","</L>","<DATE>","</DATE>","<LANG>","</LANG>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special})

    train_raw = load_hipe_jsonl("train", noise_prob=0.0)  # noise applied in Dataset
    dev_raw   = load_hipe_jsonl("dev",   noise_prob=eval_noise)

    train_ds = MHIPEXDataset(train_raw, tokenizer, MAX_LEN, augment_noise=train_noise)
    dev_ds   = MHIPEXDataset(dev_raw,   tokenizer, MAX_LEN, augment_noise=0.0)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH, shuffle=False, num_workers=2)

    model = MHIPEXModel(HM_MODEL).to(DEVICE)
    model.encoder.resize_token_embeddings(len(tokenizer))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = len(train_loader) * EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, int(0.12*steps), steps)
    scaler = GradScaler()

    best_mr, best_at, best_isat, patience_cnt = 0, 0, 0, 0
    for epoch in range(1, EPOCHS+1):
        loss = train_one_epoch(model, train_loader, opt, sched, scaler)
        mr, mr_at, mr_isat = evaluate(model, dev_loader)
        print(f"  Epoch {epoch:2d} | loss={loss:.4f} | MR={mr:.4f} | at={mr_at:.4f} | isAt={mr_isat:.4f}")
        if mr > best_mr:
            best_mr, best_at, best_isat = mr, mr_at, mr_isat
            patience_cnt = 0
            torch.save(model.state_dict(), OUT_DIR / f"{name}_best.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\n  BEST → MR={best_mr:.4f} | at={best_at:.4f} | isAt={best_isat:.4f}")
    return {"name": name, "train_noise": train_noise, "eval_noise": eval_noise,
            "MR": round(best_mr, 4), "at": round(best_at, 4), "isAt": round(best_isat, 4)}

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = []

    # R0: Standard model, clean evaluation
    results.append(run_experiment("R0_clean_clean",      train_noise=0.00, eval_noise=0.00))
    # R1: Standard model, corrupted evaluation (tests brittleness)
    results.append(run_experiment("R1_clean_corrupted",  train_noise=0.00, eval_noise=0.08))
    # R2: Noise-augmented training, clean evaluation (tests if augmentation helps normally)
    results.append(run_experiment("R2_noisy_clean",      train_noise=0.08, eval_noise=0.00))
    # R3: Noise-augmented training + corrupted evaluation (tests robustness under distribution shift)
    results.append(run_experiment("R3_noisy_corrupted",  train_noise=0.08, eval_noise=0.08))

    # Save results
    out_path = OUT_DIR / "ocr_robustness_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)

    print("\n" + "="*60)
    print("FINAL RESULTS — OCR NOISE ROBUSTNESS EXPERIMENT")
    print("="*60)
    print(f"{'Config':<25} {'MR':>6} {'at':>6} {'isAt':>6}")
    print("-"*47)
    for r in results:
        print(f"{r['name']:<25} {r['MR']:>6.4f} {r['at']:>6.4f} {r['isAt']:>6.4f}")

    print(f"\nResults saved to {out_path}")
    print("\nLaTeX Table Row Template:")
    for r in results:
        label_map = {
            "R0_clean_clean":     "R0: Clean train + clean eval (baseline)",
            "R1_clean_corrupted": "R1: Clean train + noisy eval",
            "R2_noisy_clean":     "R2: Noise-aug train + clean eval",
            "R3_noisy_corrupted": "R3: Noise-aug train + noisy eval",
        }
        print(f"{label_map[r['name']]} & {r['MR']:.4f} & {r['at']:.4f} & {r['isAt']:.4f} \\\\")
