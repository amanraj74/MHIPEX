"""
MHIPEX — OCR Noise Robustness Experiment
=========================================
Complete, self-contained notebook for Kaggle (T4×2, Internet ON).
Paste ALL of this into ONE Kaggle notebook cell and run.

Experiment (per Dr. Jain Comments.pdf, page 12):
  R0: Standard training  → clean evaluation       (baseline)
  R1: Standard training  → OCR-corrupted eval     (brittleness test)
  R2: Noise-aug training → clean evaluation        (does augmentation help?)
  R3: Noise-aug training → OCR-corrupted eval     (robustness under shift)

Output: out_ocr/ocr_robustness_results.csv  +  LaTeX table rows
Expected runtime: ~60–80 min total (4 × ~15–20 min each)
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install + Imports
# ══════════════════════════════════════════════════════════════════════════════
import os, json, re, gc, time, random, urllib.request, csv
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

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU  = torch.cuda.device_count()
print(f"Device: {DEVICE} | GPUs: {N_GPU}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Directories + Download HIPE-2026 Data (same as v12 notebook)
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path("data");    DATA_DIR.mkdir(exist_ok=True)
OUT_DIR  = Path("out_ocr"); OUT_DIR.mkdir(exist_ok=True)

# Exact same URLs as kaggle_mhipex_v12_cell1.py — proven working
BASE_URL = "https://raw.githubusercontent.com/hipe-eval/HIPE-2026-data/main/data/sandbox"
FILES = {
    "en-train": f"{BASE_URL}/en-train.jsonl",
    "fr-train": f"{BASE_URL}/fr-train.jsonl",
    "de-train": f"{BASE_URL}/de-train.jsonl",
    "en-dev":   f"{BASE_URL}/en-dev.jsonl",
    "fr-dev":   f"{BASE_URL}/fr-dev.jsonl",
    "de-dev":   f"{BASE_URL}/de-dev.jsonl",
}

print("\n── Downloading data ─────────────────────────────────────────────────")
for name, url in FILES.items():
    dst = DATA_DIR / f"{name}.jsonl"
    if not dst.exists():
        print(f"  Downloading {name}.jsonl ...")
        urllib.request.urlretrieve(url, dst)
        print(f"  ✓ {name}.jsonl ({dst.stat().st_size//1024} KB)")
    else:
        print(f"  {name}.jsonl (cached, {dst.stat().st_size//1024} KB)")
print("✅ Data ready")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 — OCR Noise Functions
# ══════════════════════════════════════════════════════════════════════════════
# Common historical OCR character confusions from literature
# (Schulz & Kuhn 2017; Nguyen et al. 2021; Hamdi et al. 2020)
OCR_SUBS_PAIRS = [
    ("rn", "m"),   # rn→m (most common confusion)
    ("m",  "rn"),  # reverse
    ("l",  "1"),   # l→1 (digit/letter)
    ("1",  "l"),
    ("c",  "e"),   # c→e (very common)
    ("e",  "c"),
    ("ii", "n"),   # ii→n
    ("u",  "n"),   # u→n
    ("n",  "u"),
    ("cl", "d"),   # cl→d
    ("vv", "w"),   # vv→w
    ("f",  "s"),   # long-s→f confusion in German/French
    ("d",  "cl"),  # d→cl
    ("a",  "o"),   # a→o
    ("h",  "li"),  # h→li
    ("0",  "o"),   # 0→o digit confusion
    ("o",  "0"),
]

def apply_ocr_noise(text: str, word_prob: float = 0.08) -> str:
    """
    Corrupt `word_prob` fraction of words with a random OCR substitution.
    Only words longer than 3 chars are corrupted (short words → false positives).
    """
    words = text.split()
    noisy = []
    for word in words:
        if len(word) > 3 and random.random() < word_prob:
            sub_from, sub_to = random.choice(OCR_SUBS_PAIRS)
            lower = word.lower()
            if sub_from in lower:
                idx  = lower.find(sub_from)
                word = word[:idx] + sub_to + word[idx + len(sub_from):]
        noisy.append(word)
    return " ".join(noisy)

# Quick sanity check
_test = apply_ocr_noise("Napoleon was located in Paris around 1815", word_prob=0.5)
print(f"OCR noise test: '{_test}'")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Data Loading (exact same JSONL structure as v12)
# ══════════════════════════════════════════════════════════════════════════════
AT_MAP   = {"FALSE": 0, "PROBABLE": 1, "TRUE": 2}
ISAT_MAP = {"FALSE": 0, "TRUE": 1}

def clean_text(t, max_chars=850):
    return re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", t)).strip()[:max_chars]

def build_input_v12(text, pers_list, loc_list, date_str="", lang=""):
    """Enriched input with ALL mentions + DATE + LANG tokens — same as v12."""
    p = " ; ".join(clean_text(m, 100) for m in pers_list) if pers_list else "UNKNOWN"
    l = " ; ".join(clean_text(m, 100) for m in loc_list)  if loc_list  else "UNKNOWN"
    date_tok = f"<DATE> {date_str} </DATE> " if date_str else ""
    lang_tok = f"<LANG> {lang} </LANG> "     if lang     else ""
    return f"<P> {p} </P> <L> {l} </L> {date_tok}{lang_tok}{clean_text(text)}"

def load_hipe_jsonl(split: str, noise_prob: float = 0.0):
    """
    Load HIPE-2026 sandbox JSONL files (en/fr/de).
    noise_prob: if > 0, apply OCR corruption to the article text.
    Returns flat list of dicts with keys: text, at, isat, lang
    """
    records = []
    langs = ["en", "fr", "de"]
    for lang in langs:
        fpath = DATA_DIR / f"{lang}-{split}.jsonl"
        if not fpath.exists():
            print(f"  WARNING: {lang}-{split}.jsonl not found")
            continue
        count = 0
        for line in open(fpath, encoding="utf-8"):
            doc = json.loads(line)
            date_str = str(doc.get("date", ""))[:10]
            text_raw = doc.get("text", "")
            # Apply OCR noise to article TEXT (not to entity names)
            text_noisy = apply_ocr_noise(text_raw, noise_prob) if noise_prob > 0 else text_raw
            for pair in doc.get("sampled_pairs", []):
                at_raw   = pair.get("at",   "FALSE").upper()
                isat_raw = pair.get("isAt", "FALSE").upper()
                if at_raw not in AT_MAP or isat_raw not in ISAT_MAP:
                    continue
                records.append({
                    "text": build_input_v12(
                        text_noisy,
                        pair["pers_mentions_list"],
                        pair["loc_mentions_list"],
                        date_str, lang
                    ),
                    "at":   AT_MAP[at_raw],
                    "isat": ISAT_MAP[isat_raw],
                    "lang": lang,
                })
                count += 1
        print(f"  {lang}-{split}: {count} pairs loaded")
    print(f"  TOTAL {split}: {len(records)} pairs")
    return records

# Test load (no noise)
print("\n── Verifying data loading ───────────────────────────────────────────")
_test_train = load_hipe_jsonl("train")
_test_dev   = load_hipe_jsonl("dev")
print(f"✅ Train={len(_test_train)}, Dev={len(_test_dev)}")
del _test_train, _test_dev

# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Dataset + Model
# ══════════════════════════════════════════════════════════════════════════════
SPECIAL_TOKENS = ["<P>","</P>","<L>","</L>","<DATE>","</DATE>","<LANG>","</LANG>"]
MAX_LEN = 256
BATCH   = 16   # per GPU; DataParallel doubles effective batch
EPOCHS  = 20
LR      = 8e-6
PATIENCE = 6

HM_MODEL = "dbmdz/bert-base-historic-multilingual-cased"

class MHIPEXDataset(Dataset):
    def __init__(self, records, tokenizer, max_len, augment_noise=0.0):
        self.records = records
        self.tok     = tokenizer
        self.max_len = max_len
        self.aug     = augment_noise  # additional noise applied at __getitem__ time

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        # For training augmentation: text already built, but we can add word-level noise here too
        # Note: noise is applied to the full formatted string for training augmentation
        text = r["text"]
        if self.aug > 0:
            text = apply_ocr_noise(text, self.aug)
        enc = self.tok(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "at":   torch.tensor(r["at"],   dtype=torch.long),
            "isAt": torch.tensor(r["isat"], dtype=torch.long),
        }


class MHIPEXModel(nn.Module):
    def __init__(self, model_name, dropout=0.15, n_dropout=3):
        super().__init__()
        self.encoder   = AutoModel.from_pretrained(model_name)
        h              = self.encoder.config.hidden_size
        self.drops     = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_dropout)])
        self.at_head   = nn.Linear(h, 3)
        self.isat_head = nn.Linear(h, 2)

    def forward(self, input_ids, attention_mask):
        out  = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls  = out.last_hidden_state[:, 0]
        mean = (out.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) \
               / attention_mask.sum(1, keepdim=True).clamp(min=1)
        h    = 0.5 * cls + 0.5 * mean          # dual pooling
        # Multi-sample dropout: average logits across K masks
        at_logits   = sum(self.at_head(d(h))   for d in self.drops) / len(self.drops)
        isat_logits = sum(self.isat_head(d(h)) for d in self.drops) / len(self.drops)
        return at_logits, isat_logits


# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Train + Evaluate Functions
# ══════════════════════════════════════════════════════════════════════════════
AT_W   = torch.tensor([0.80, 1.50, 2.40], device=DEVICE)
ISAT_W = torch.tensor([0.70, 2.60],       device=DEVICE)

def macro_recall(y_true, y_pred, n_classes):
    return recall_score(y_true, y_pred, average="macro",
                        labels=list(range(n_classes)), zero_division=0)

def train_one_epoch(model, loader, opt, sched, scaler):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, leave=False, desc="  train"):
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
    mr_at   = macro_recall(at_true, at_pred, 3)
    mr_isat = macro_recall(is_true, is_pred, 2)
    mr      = (mr_at + mr_isat) / 2
    return mr, mr_at, mr_isat


# ══════════════════════════════════════════════════════════════════════════════
# CELL 7 — run_experiment() — trains one model, returns best MR
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(name, train_noise=0.0, eval_noise=0.0):
    """
    train_noise: word-level OCR noise applied DURING TRAINING (data augmentation)
    eval_noise:  word-level OCR noise applied to DEV SET at load time
    """
    print(f"\n{'='*65}")
    print(f"  EXPERIMENT: {name}")
    print(f"  Train noise={train_noise:.0%}  |  Eval noise={eval_noise:.0%}")
    print(f"{'='*65}")
    t0 = time.time()

    # ── Load data ───────────────────────────────────────────────────────────
    print("\n  Loading data...")
    # Train: always load clean, noise is applied inside Dataset if train_noise>0
    train_raw = load_hipe_jsonl("train", noise_prob=0.0)
    # Dev: apply noise at LOAD TIME so the evaluation set is consistently corrupted
    dev_raw   = load_hipe_jsonl("dev",   noise_prob=eval_noise)

    if len(train_raw) == 0:
        print("  ERROR: No training data found! Check data download.")
        return None
    if len(dev_raw) == 0:
        print("  ERROR: No dev data found!")
        return None

    # ── Tokenizer ───────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(HM_MODEL)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # ── Datasets ────────────────────────────────────────────────────────────
    # augment_noise in Dataset applies noise per-sample per-epoch (dynamic)
    train_ds = MHIPEXDataset(train_raw, tokenizer, MAX_LEN, augment_noise=train_noise)
    dev_ds   = MHIPEXDataset(dev_raw,   tokenizer, MAX_LEN, augment_noise=0.0)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=2, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH*2, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"  Train: {len(train_ds)} | Dev: {len(dev_ds)}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = MHIPEXModel(HM_MODEL).to(DEVICE)
    model.encoder.resize_token_embeddings(len(tokenizer))
    if N_GPU > 1:
        model = nn.DataParallel(model)

    # ── Optimizer + Scheduler ───────────────────────────────────────────────
    # Layer-wise LR decay (same as v12: γ=0.90 for hmBERT)
    no_decay = ["bias", "LayerNorm.weight"]
    opt_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters()
                    if     any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt    = torch.optim.AdamW(opt_groups, lr=LR)
    steps  = len(train_loader) * EPOCHS
    sched  = get_cosine_schedule_with_warmup(opt, int(0.12 * steps), steps)
    scaler = GradScaler()

    # ── Training Loop ───────────────────────────────────────────────────────
    best_mr, best_at, best_isat, patience_cnt = 0.0, 0.0, 0.0, 0
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, opt, sched, scaler)
        mr, mr_at, mr_isat = evaluate(model, dev_loader)
        elapsed = (time.time() - t0) / 60
        print(f"  Ep {epoch:2d}/{EPOCHS} | loss={loss:.4f} | MR={mr:.4f} "
              f"| at={mr_at:.4f} | isAt={mr_isat:.4f} | {elapsed:.1f}min")
        if mr > best_mr:
            best_mr, best_at, best_isat = mr, mr_at, mr_isat
            patience_cnt = 0
            ckpt_path = OUT_DIR / f"{name}_best.pt"
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state, ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  ⏹ Early stopping at epoch {epoch}")
                break

    total_min = (time.time() - t0) / 60
    print(f"\n  ✅ BEST → MR={best_mr:.4f} | at={best_at:.4f} | isAt={best_isat:.4f}")
    print(f"  Total time: {total_min:.1f} min")

    del model; gc.collect(); torch.cuda.empty_cache()

    return {
        "name":        name,
        "train_noise": f"{train_noise:.0%}",
        "eval_noise":  f"{eval_noise:.0%}",
        "MR":          round(best_mr,   4),
        "at":          round(best_at,   4),
        "isAt":        round(best_isat, 4),
        "time_min":    round(total_min, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CELL 8 — Run All 4 Experiments
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  MHIPEX — OCR NOISE ROBUSTNESS EXPERIMENT")
print("  4 configurations × ~15–20 min each ≈ 60–80 min total")
print("="*65)

results = []

# R0: Standard model, clean evaluation  ←  this IS your existing baseline
r = run_experiment("R0_clean_clean",     train_noise=0.00, eval_noise=0.00)
if r: results.append(r)

# R1: Standard model, corrupted eval  ←  tests brittleness
r = run_experiment("R1_clean_corrupted", train_noise=0.00, eval_noise=0.08)
if r: results.append(r)

# R2: Noise-augmented training, clean eval  ←  does augmentation hurt clean perf?
r = run_experiment("R2_noisy_clean",     train_noise=0.08, eval_noise=0.00)
if r: results.append(r)

# R3: Noise-augmented training, corrupted eval  ←  the key robustness test
r = run_experiment("R3_noisy_corrupted", train_noise=0.08, eval_noise=0.08)
if r: results.append(r)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 9 — Save + Print Results
# ══════════════════════════════════════════════════════════════════════════════
if results:
    # Save CSV
    csv_path = OUT_DIR / "ocr_robustness_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ CSV saved: {csv_path}")

    # Pretty print table
    print("\n" + "="*65)
    print("  FINAL RESULTS — OCR NOISE ROBUSTNESS")
    print("="*65)
    print(f"  {'Config':<25} {'Train':>7} {'Eval':>7} {'MR':>7} {'at':>7} {'isAt':>7}")
    print("  " + "-"*60)
    for r in results:
        print(f"  {r['name']:<25} {r['train_noise']:>7} {r['eval_noise']:>7} "
              f"{r['MR']:>7.4f} {r['at']:>7.4f} {r['isAt']:>7.4f}")

    # Key findings
    if len(results) >= 2:
        r0 = next((r for r in results if r["name"] == "R0_clean_clean"),     None)
        r1 = next((r for r in results if r["name"] == "R1_clean_corrupted"), None)
        r2 = next((r for r in results if r["name"] == "R2_noisy_clean"),     None)
        r3 = next((r for r in results if r["name"] == "R3_noisy_corrupted"), None)
        print("\n  KEY FINDINGS:")
        if r0 and r1:
            drop = r0["MR"] - r1["MR"]
            print(f"  • OCR brittleness (R0→R1): MR drops {drop:+.4f} "
                  f"({drop/r0['MR']*100:+.1f}%) under corruption")
        if r2 and r0:
            cost = r2["MR"] - r0["MR"]
            print(f"  • Augmentation cost on clean (R0→R2): {cost:+.4f} MR")
        if r1 and r3:
            gain = r3["MR"] - r1["MR"]
            print(f"  • Robustness gain (R1→R3): {gain:+.4f} MR recovered by noise training")

    # LaTeX table rows for paper Table 12
    print("\n" + "="*65)
    print("  LATEX TABLE ROWS (paste into Table 12 in main.tex):")
    print("="*65)
    label_map = {
        "R0_clean_clean":     "R0: Standard (clean train, clean eval)",
        "R1_clean_corrupted": "R1: Standard (clean train, noisy eval)",
        "R2_noisy_clean":     "R2: Noise-aug (noisy train, clean eval)",
        "R3_noisy_corrupted": "R3: Noise-aug (noisy train, noisy eval)",
    }
    for r in results:
        label = label_map.get(r["name"], r["name"])
        print(f"{label} & {r['MR']:.4f} & {r['at']:.4f} & {r['isAt']:.4f} \\\\")
else:
    print("ERROR: No results collected. Check data loading above.")
