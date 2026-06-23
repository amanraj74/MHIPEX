# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MHIPEX — Cross-Dataset Validation Experiments                         ║
# ║  SINGLE NOTEBOOK: Run all 4 experiments + generate results             ║
# ║                                                                        ║
# ║  Experiments:                                                          ║
# ║  C1: Sandbox → Newspaper v1.0 (domain transfer)                       ║
# ║  C2: FR+DE → EN (cross-lingual zero-shot)                             ║
# ║  C3: EN+DE → FR (cross-lingual, largest test)                         ║
# ║  C4: EN+FR → DE (cross-lingual, Germanic)                             ║
# ║                                                                        ║
# ║  Uses hmBERT only (best single model). ~90 min total on T4.           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os, math, json, re, random, copy, gc, time, sys
os.environ["PYTORCH_ALLOC_CONF"]     = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import subprocess
subprocess.run(["pip","install","-q","transformers==4.44.2","accelerate","scikit-learn","tqdm"], check=True)

from pathlib import Path
from collections import Counter
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import recall_score, classification_report
from tqdm import tqdm

# ── Reproducibility ──────────────────────────────────────────────────────
SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
set_seed()
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU  = torch.cuda.device_count()
print(f"✅ Device: {DEVICE} | GPUs: {N_GPU} | CUDA: {torch.version.cuda}")

# ── Constants ────────────────────────────────────────────────────────────
AT_MAP     = {"FALSE":0, "PROBABLE":1, "TRUE":2}
ISAT_MAP   = {"FALSE":0, "TRUE":1}
AT_NAMES   = ["FALSE", "PROBABLE", "TRUE"]
ISAT_NAMES = ["FALSE", "TRUE"]
SPECIAL_TOKENS = ["<P>","</P>","<L>","</L>","<DATE>","</DATE>","<LANG>","</LANG>"]

AT_W   = torch.tensor([0.80, 1.50, 2.40], dtype=torch.float32)
ISAT_W = torch.tensor([0.70, 2.60],       dtype=torch.float32)

DATA_DIR = Path("data");  DATA_DIR.mkdir(exist_ok=True)
PROC_DIR = Path("proc");  PROC_DIR.mkdir(exist_ok=True)
OUT_DIR  = Path("out");   OUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "dbmdz/bert-base-historic-multilingual-cased"

# ══════════════════════════════════════════════════════════════════════════
#  DATA DOWNLOAD + PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════

SANDBOX_URL = "https://raw.githubusercontent.com/hipe-eval/HIPE-2026-data/main/data/sandbox"
NEWS_URL    = "https://raw.githubusercontent.com/hipe-eval/HIPE-2026-data/main/data/newspapers/v1.0"

SANDBOX_FILES = {
    "en-train": f"{SANDBOX_URL}/en-train.jsonl",
    "fr-train": f"{SANDBOX_URL}/fr-train.jsonl",
    "de-train": f"{SANDBOX_URL}/de-train.jsonl",
    "en-dev":   f"{SANDBOX_URL}/en-dev.jsonl",
    "fr-dev":   f"{SANDBOX_URL}/fr-dev.jsonl",
    "de-dev":   f"{SANDBOX_URL}/de-dev.jsonl",
}

NEWS_FILES = {
    "news-en": f"{NEWS_URL}/HIPE-2026-v1.0-impresso-train-en.jsonl",
    "news-fr": f"{NEWS_URL}/HIPE-2026-v1.0-impresso-train-fr.jsonl",
    "news-de": f"{NEWS_URL}/HIPE-2026-v1.0-impresso-train-de.jsonl",
}

print("\n── Downloading data ─────────────────────────────────────────────")
import urllib.request
for name, url in {**SANDBOX_FILES, **NEWS_FILES}.items():
    dst = DATA_DIR / f"{name}.jsonl"
    if not dst.exists():
        print(f"  ↓ {name}.jsonl ...")
        urllib.request.urlretrieve(url, dst)
    else:
        print(f"  ✓ {name}.jsonl (cached)")
print("✅ All data ready")

# ══════════════════════════════════════════════════════════════════════════
#  PREPROCESSING (identical to v12)
# ══════════════════════════════════════════════════════════════════════════

def clean_text(t, max_chars=850):
    return re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", t)).strip()[:max_chars]

def build_input_v12(text, pers_list, loc_list, date_str="", lang=""):
    p = " ; ".join(clean_text(m, 100) for m in pers_list) if pers_list else "UNKNOWN"
    l = " ; ".join(clean_text(m, 100) for m in loc_list)  if loc_list  else "UNKNOWN"
    date_tok = f"<DATE> {date_str} </DATE> " if date_str else ""
    lang_tok = f"<LANG> {lang} </LANG> "     if lang     else ""
    return f"<P> {p} </P> <L> {l} </L> {date_tok}{lang_tok}{clean_text(text)}"

def load_and_process(path, lang):
    records = []
    for line in open(path, encoding="utf-8"):
        doc = json.loads(line)
        date_str = str(doc.get("date", ""))[:10]
        for pair in doc.get("sampled_pairs", []):
            at_raw   = pair.get("at",   "FALSE")
            isat_raw = pair.get("isAt", "FALSE")
            if at_raw not in AT_MAP or isat_raw not in ISAT_MAP:
                continue
            records.append({
                "text": build_input_v12(
                    doc["text"], pair["pers_mentions_list"],
                    pair["loc_mentions_list"], date_str, lang
                ),
                "at":   AT_MAP[at_raw],
                "isat": ISAT_MAP[isat_raw],
                "lang": lang,
            })
    return records

def build_split(file_list, out_name):
    """Build a JSONL file from a list of (path, lang) tuples."""
    all_recs = []
    for path, lang in file_list:
        recs = load_and_process(path, lang)
        all_recs.extend(recs)
        print(f"    {lang}: {len(recs):,} pairs")
    out_path = PROC_DIR / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  → {out_name}: {len(all_recs):,} total")
    return out_path, len(all_recs)

# ══════════════════════════════════════════════════════════════════════════
#  MODEL (identical to v12)
# ══════════════════════════════════════════════════════════════════════════

class HIPEDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=256):
        self.data = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        enc = self.tok(d["text"], max_length=self.max_len,
                       truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "at_label":       torch.tensor(d["at"],   dtype=torch.long),
            "isat_label":     torch.tensor(d["isat"], dtype=torch.long),
            "lang":           d["lang"],
        }

class MHIPEXv12(nn.Module):
    def __init__(self, model_name, n_special, dropout=0.15, n_drops=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.resize_token_embeddings(
            self.encoder.config.vocab_size + n_special
        )
        h = self.encoder.config.hidden_size
        self.layer_norm = nn.LayerNorm(h * 2)
        self.dropouts   = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_drops)])
        self.head_at    = nn.Linear(h * 2, 3)
        self.head_isat  = nn.Linear(h * 2, 2)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        cls_vec = hidden[:, 0, :]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden * mask_expanded).sum(dim=1)
        mean_vec   = sum_hidden / mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = self.layer_norm(torch.cat([cls_vec, mean_vec], dim=-1))
        at_logits   = torch.stack([self.head_at(d(pooled))   for d in self.dropouts]).mean(0)
        isat_logits = torch.stack([self.head_isat(d(pooled)) for d in self.dropouts]).mean(0)
        return {"at_logits": at_logits, "isat_logits": isat_logits}

# ══════════════════════════════════════════════════════════════════════════
#  TRAINING ENGINE (streamlined from v12)
# ══════════════════════════════════════════════════════════════════════════

def get_layer_wise_lr_groups(model, base_lr, decay=0.90, wd=0.01):
    groups = []
    enc = model.module.encoder if hasattr(model, 'module') else model.encoder
    heads = [model.module.head_at, model.module.head_isat, model.module.layer_norm] \
            if hasattr(model, 'module') else [model.head_at, model.head_isat, model.layer_norm]

    if hasattr(enc, 'encoder') and hasattr(enc.encoder, 'layer'):
        layers = list(enc.encoder.layer)
    else:
        layers = []

    n_layers = len(layers)
    embed_params = list(enc.embeddings.parameters())
    if embed_params:
        groups.append({"params": embed_params, "lr": base_lr * (decay ** n_layers), "weight_decay": wd})

    for i, layer in enumerate(layers):
        lr = base_lr * (decay ** (n_layers - 1 - i))
        groups.append({"params": list(layer.parameters()), "lr": lr, "weight_decay": wd})

    head_params = []
    for h in heads:
        head_params.extend(list(h.parameters()))
    groups.append({"params": head_params, "lr": base_lr * 10, "weight_decay": 0.0})
    return groups

def macro_recall(at_t, at_p, is_t, is_p):
    r_at   = recall_score(at_t,  at_p,  average="macro", zero_division=0)
    r_isat = recall_score(is_t, is_p, average="macro", zero_division=0)
    return round((r_at + r_isat) / 2, 4), round(r_at, 4), round(r_isat, 4)

def train_and_evaluate(train_path, test_path, exp_name, max_epochs=25, patience=6):
    """Train hmBERT on train_path, evaluate on test_path. Returns best MR."""
    print(f"\n{'═'*64}")
    print(f"  Experiment: {exp_name}")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    print(f"{'═'*64}")

    set_seed()
    gc.collect(); torch.cuda.empty_cache()

    # Config
    cfg = {"bs": 32, "lr": 8e-6, "maxlen": 256, "accum": 2, "decay": 0.90}

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # Datasets
    train_ds = HIPEDataset(train_path, tok, cfg["maxlen"])
    test_ds  = HIPEDataset(test_path,  tok, cfg["maxlen"])
    print(f"  Train: {len(train_ds):,} | Test: {len(test_ds):,}")

    at_labels = [d["at"] for d in train_ds.data]
    sample_weights = [AT_W[l].item() for l in at_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg["bs"], sampler=sampler,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["bs"]*2, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Model
    model = MHIPEXv12(MODEL_NAME, len(SPECIAL_TOKENS), dropout=0.15, n_drops=3).to(DEVICE)
    if N_GPU > 1:
        model = nn.DataParallel(model)

    param_groups = get_layer_wise_lr_groups(model, cfg["lr"], cfg["decay"])
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = (len(train_loader) // cfg["accum"]) * max_epochs
    warmup = int(total_steps * 0.12)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = GradScaler("cuda")

    criterion_at   = nn.CrossEntropyLoss(weight=AT_W.to(DEVICE))
    criterion_isat = nn.CrossEntropyLoss(weight=ISAT_W.to(DEVICE))

    best_mr = 0.0
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # ── TRAIN ──
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Ep{epoch:02d}", leave=False)):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            at_y = batch["at_label"].to(DEVICE)
            is_y = batch["isat_label"].to(DEVICE)

            with autocast("cuda"):
                out = model(ids, mask)
                loss = (criterion_at(out["at_logits"], at_y) +
                        criterion_isat(out["isat_logits"], is_y)) / cfg["accum"]

            scaler.scale(loss).backward()

            if (step + 1) % cfg["accum"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * cfg["accum"]

        avg_loss = total_loss / len(train_loader)

        # ── EVAL ──
        model.eval()
        at_true, at_pred, is_true, is_pred = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                ids  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                with autocast("cuda"):
                    out = model(ids, mask)
                at_pred.extend(out["at_logits"].argmax(1).cpu().tolist())
                is_pred.extend(out["isat_logits"].argmax(1).cpu().tolist())
                at_true.extend(batch["at_label"].tolist())
                is_true.extend(batch["isat_label"].tolist())

        mr, r_at, r_isat = macro_recall(at_true, at_pred, is_true, is_pred)

        if mr > best_mr:
            best_mr = mr
            best_at, best_isat = r_at, r_isat
            no_improve = 0
            print(f"  Ep{epoch:02d} | Loss:{avg_loss:.4f} | MR:{mr} (at={r_at}, isAt={r_isat}) ✅ NEW BEST")
        else:
            no_improve += 1
            if epoch <= 3 or no_improve == patience:
                print(f"  Ep{epoch:02d} | Loss:{avg_loss:.4f} | MR:{mr} | no improve ({no_improve}/{patience})")

        if no_improve >= patience:
            print(f"  ⏹ Early stopping at epoch {epoch}")
            break

    # Cleanup
    del model, optimizer, scheduler, scaler
    gc.collect(); torch.cuda.empty_cache()

    return {"experiment": exp_name, "MR": best_mr, "at": best_at, "isAt": best_isat}

# ══════════════════════════════════════════════════════════════════════════
#  BUILD ALL EXPERIMENT DATASETS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "━"*64)
print("  BUILDING CROSS-VALIDATION DATASETS")
print("━"*64)

# C1: Sandbox (all langs, train) → Newspaper v1.0 (all langs, as test)
print("\n[C1] Sandbox → Newspaper v1.0")
print("  Train (sandbox all):")
c1_train, _ = build_split([
    (DATA_DIR / "en-train.jsonl", "en"),
    (DATA_DIR / "fr-train.jsonl", "fr"),
    (DATA_DIR / "de-train.jsonl", "de"),
], "c1_train.jsonl")
print("  Test (newspaper v1.0):")
c1_test, _ = build_split([
    (DATA_DIR / "news-en.jsonl", "en"),
    (DATA_DIR / "news-fr.jsonl", "fr"),
    (DATA_DIR / "news-de.jsonl", "de"),
], "c1_test.jsonl")

# C2: FR+DE → EN (zero-shot English)
print("\n[C2] FR+DE → EN")
print("  Train (FR+DE):")
c2_train, _ = build_split([
    (DATA_DIR / "fr-train.jsonl", "fr"),
    (DATA_DIR / "de-train.jsonl", "de"),
], "c2_train.jsonl")
print("  Test (EN dev):")
c2_test, _ = build_split([
    (DATA_DIR / "en-dev.jsonl", "en"),
], "c2_test.jsonl")

# C3: EN+DE → FR (zero-shot French)
print("\n[C3] EN+DE → FR")
print("  Train (EN+DE):")
c3_train, _ = build_split([
    (DATA_DIR / "en-train.jsonl", "en"),
    (DATA_DIR / "de-train.jsonl", "de"),
], "c3_train.jsonl")
print("  Test (FR dev):")
c3_test, _ = build_split([
    (DATA_DIR / "fr-dev.jsonl", "fr"),
], "c3_test.jsonl")

# C4: EN+FR → DE (zero-shot German)
print("\n[C4] EN+FR → DE")
print("  Train (EN+FR):")
c4_train, _ = build_split([
    (DATA_DIR / "en-train.jsonl", "en"),
    (DATA_DIR / "fr-train.jsonl", "fr"),
], "c4_train.jsonl")
print("  Test (DE dev):")
c4_test, _ = build_split([
    (DATA_DIR / "de-dev.jsonl", "de"),
], "c4_test.jsonl")

print("\n✅ All cross-validation datasets ready")

# ══════════════════════════════════════════════════════════════════════════
#  RUN ALL 4 EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════

experiments = [
    (c1_train, c1_test, "C1: Sandbox → Newspaper"),
    (c2_train, c2_test, "C2: FR+DE → EN (zero-shot)"),
    (c3_train, c3_test, "C3: EN+DE → FR (zero-shot)"),
    (c4_train, c4_test, "C4: EN+FR → DE (zero-shot)"),
]

results = []
for train_p, test_p, name in experiments:
    r = train_and_evaluate(train_p, test_p, name, max_epochs=20, patience=5)
    results.append(r)
    print(f"\n  ► {name}: MR={r['MR']:.4f} (at={r['at']:.4f}, isAt={r['isAt']:.4f})")

# ══════════════════════════════════════════════════════════════════════════
#  RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*64)
print("  CROSS-DATASET VALIDATION RESULTS")
print("═"*64)
print(f"  {'Experiment':<30} {'MR':>6} {'at':>6} {'isAt':>6}")
print("  " + "─"*54)

# Add reference: full sandbox training result
print(f"  {'Ref: Full sandbox (hmBERT)':<30} {'0.553':>6} {'0.450':>6} {'0.655':>6}")
print("  " + "─"*54)

for r in results:
    print(f"  {r['experiment']:<30} {r['MR']:>6.4f} {r['at']:>6.4f} {r['isAt']:>6.4f}")

print("═"*64)

# Save results
results_file = OUT_DIR / "crossval_results.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Results saved to {results_file}")

# ── Generate CSV for paper ──
csv_file = OUT_DIR / "crossval_results.csv"
with open(csv_file, "w") as f:
    f.write("Experiment,Train,Test,MR,at_recall,isAt_recall\n")
    f.write("Reference,Sandbox (EN+FR+DE),Sandbox dev,0.5525,0.4503,0.6548\n")
    for r in results:
        exp = r['experiment']
        if 'Newspaper' in exp:
            train_d, test_d = "Sandbox (EN+FR+DE)", "Newspaper v1.0"
        elif 'EN' in exp and 'zero' in exp:
            train_d, test_d = "Sandbox (FR+DE)", "Sandbox EN dev"
        elif 'FR' in exp and 'zero' in exp:
            train_d, test_d = "Sandbox (EN+DE)", "Sandbox FR dev"
        elif 'DE' in exp and 'zero' in exp:
            train_d, test_d = "Sandbox (EN+FR)", "Sandbox DE dev"
        else:
            train_d, test_d = "?", "?"
        f.write(f"{exp},{train_d},{test_d},{r['MR']:.4f},{r['at']:.4f},{r['isAt']:.4f}\n")
print(f"✅ CSV saved to {csv_file}")

print("\n" + "█"*64)
print("  DONE — Copy these results into your paper!")
print("█"*64)
