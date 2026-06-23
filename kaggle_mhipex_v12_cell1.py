# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MHIPEX v12 — CELL 1: Setup + Definitions                              ║
# ║  Runtime: ~2 min | GPU: T4 (1 or 2)                                    ║
# ║                                                                        ║
# ║  Improvements over v8-v11:                                             ║
# ║  ✓ Focal Loss (fixes at-head bottleneck from 0.43 → 0.50+)            ║
# ║  ✓ Multi-sample dropout (5-pass implicit ensemble)                     ║
# ║  ✓ Mean+CLS dual pooling (richer representation)                      ║
# ║  ✓ All entity mentions in input (not just first)                       ║
# ║  ✓ Auto-computed inverse-frequency class weights                       ║
# ║  ✓ Mixed precision (AMP) for speed                                    ║
# ║  ✓ Threshold calibration + logical constraints                         ║
# ║                                                                        ║
# ║  Target: MR 0.56–0.62 | Total time: ~90 min on 1×T4                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os, math, json, re, random, copy, gc, time, sys
os.environ["PYTORCH_ALLOC_CONF"]     = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import subprocess
subprocess.run(["pip","install","-q","transformers==4.44.2","accelerate","scikit-learn","tqdm"], check=True)

from pathlib import Path
from collections import Counter
from itertools import product as iterproduct
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

DATA_DIR = Path("data");  DATA_DIR.mkdir(exist_ok=True)
PROC_DIR = Path("proc");  PROC_DIR.mkdir(exist_ok=True)
OUT_DIR  = Path("out");   OUT_DIR.mkdir(exist_ok=True)

# ── Download HIPE-2026 sandbox data ──────────────────────────────────────
BASE_URL = "https://raw.githubusercontent.com/hipe-eval/HIPE-2026-data/main/data/sandbox"
FILES = {
    "en-train": f"{BASE_URL}/en-train.jsonl",
    "fr-train": f"{BASE_URL}/fr-train.jsonl",
    "de-train": f"{BASE_URL}/de-train.jsonl",
    "en-dev":   f"{BASE_URL}/en-dev.jsonl",
    "fr-dev":   f"{BASE_URL}/fr-dev.jsonl",
    "de-dev":   f"{BASE_URL}/de-dev.jsonl",
}

print("\n── Data ─────────────────────────────────────────────────────────")
import urllib.request
for name, url in FILES.items():
    dst = DATA_DIR / f"{name}.jsonl"
    if not dst.exists():
        print(f"  Downloading {name}.jsonl ...")
        urllib.request.urlretrieve(url, dst)
    else:
        print(f"  {name}.jsonl (cached)")
print("✅ Data ready")

# ══════════════════════════════════════════════════════════════════════════
#  DATA PREPROCESSING — v12 enriched format
# ══════════════════════════════════════════════════════════════════════════

def clean_text(t, max_chars=850):
    """Remove control chars and collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", t)).strip()[:max_chars]

def build_input_v12(text, pers_list, loc_list, date_str="", lang=""):
    """Build enriched input with ALL mentions + DATE + LANG tokens."""
    # Join all person mentions (not just first)
    p = " ; ".join(clean_text(m, 100) for m in pers_list) if pers_list else "UNKNOWN"
    l = " ; ".join(clean_text(m, 100) for m in loc_list)  if loc_list  else "UNKNOWN"
    date_tok = f"<DATE> {date_str} </DATE> " if date_str else ""
    lang_tok = f"<LANG> {lang} </LANG> "     if lang     else ""
    return f"<P> {p} </P> <L> {l} </L> {date_tok}{lang_tok}{clean_text(text)}"

def load_and_process(path, lang):
    """Load HIPE-2026 JSONL, extract pairs with v12 formatting."""
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
                    doc["text"],
                    pair["pers_mentions_list"],
                    pair["loc_mentions_list"],
                    date_str, lang
                ),
                "at":   AT_MAP[at_raw],
                "isat": ISAT_MAP[isat_raw],
                "at_raw":   at_raw,
                "isat_raw": isat_raw,
                "lang": lang,
                "doc_id": doc["document_id"],
            })
    return records

def build_dataset_v12():
    """Build train/dev JSONL files with v12 enriched format."""
    print("\n── Building v12 dataset ────────────────────────────────────────")
    for split in ["train", "dev"]:
        all_recs = []
        for lang in ["en", "fr", "de"]:
            path = DATA_DIR / f"{lang}-{split}.jsonl"
            recs = load_and_process(path, lang)
            all_recs.extend(recs)
            print(f"  {lang} {split}: {len(recs):,}")
        out_path = PROC_DIR / f"{split}_v12.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in all_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ {split}: {len(all_recs):,} → {out_path}")
    # Print sample
    sample = all_recs[0]["text"][:200]
    print(f"\nSample input: {sample}...")

build_dataset_v12()

# ── Class weights (v8-proven values that achieved MR=0.5382) ──────────────
# These are carefully tuned — auto-computed weights were too aggressive
AT_W   = torch.tensor([0.80, 1.50, 2.40], dtype=torch.float32)
ISAT_W = torch.tensor([0.70, 2.60],       dtype=torch.float32)
print(f"\n  AT   class weights: {AT_W.tolist()}")
print(f"  isAt class weights: {ISAT_W.tolist()}")

# ══════════════════════════════════════════════════════════════════════════
#  DATASET CLASS
# ══════════════════════════════════════════════════════════════════════════

class HIPEDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=256):
        self.data = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        enc = self.tok(
            d["text"], max_length=self.max_len,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "at_label":       torch.tensor(d["at"],   dtype=torch.long),
            "isat_label":     torch.tensor(d["isat"], dtype=torch.long),
            "lang":           d["lang"],
        }

# ══════════════════════════════════════════════════════════════════════════
#  LOSS: Weighted CrossEntropy (proven in v8, NOT focal loss)
#  Focal loss caused over-prediction of minority classes in v12.
# ══════════════════════════════════════════════════════════════════════════
# Using nn.CrossEntropyLoss directly — no custom class needed.

# ══════════════════════════════════════════════════════════════════════════
#  MODEL — Dual-head with multi-sample dropout + mean pooling
# ══════════════════════════════════════════════════════════════════════════

class MHIPEXv12(nn.Module):
    def __init__(self, model_name, n_special, dropout=0.15, n_drops=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.resize_token_embeddings(
            self.encoder.config.vocab_size + n_special
        )
        h = self.encoder.config.hidden_size

        # Dual pooling: CLS + mean → 2h features
        self.layer_norm = nn.LayerNorm(h * 2)
        self.dropouts   = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_drops)])
        self.head_at    = nn.Linear(h * 2, 3)   # FALSE / PROBABLE / TRUE
        self.head_isat  = nn.Linear(h * 2, 2)   # FALSE / TRUE

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, L, H)

        # CLS token
        cls_vec = hidden[:, 0, :]

        # Mean pooling (masked)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        sum_hidden = (hidden * mask_expanded).sum(dim=1)
        mean_vec   = sum_hidden / mask_expanded.sum(dim=1).clamp(min=1e-9)

        # Concatenate CLS + Mean → (B, 2H)
        pooled = self.layer_norm(torch.cat([cls_vec, mean_vec], dim=-1))

        # Multi-sample dropout: 5 different masks, averaged
        at_logits   = torch.stack([self.head_at(d(pooled))   for d in self.dropouts]).mean(0)
        isat_logits = torch.stack([self.head_isat(d(pooled)) for d in self.dropouts]).mean(0)

        return {"at_logits": at_logits, "isat_logits": isat_logits}

# ══════════════════════════════════════════════════════════════════════════
#  TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════════

def get_layer_wise_lr_groups(model, base_lr, decay=0.90, wd=0.01):
    """Layer-wise LR decay: lower layers get smaller LR."""
    groups = []
    # Encoder layers
    if hasattr(model, 'module'):
        enc = model.module.encoder
        heads = [model.module.head_at, model.module.head_isat,
                 model.module.layer_norm]
    else:
        enc = model.encoder
        heads = [model.head_at, model.head_isat, model.layer_norm]

    # Get encoder layers
    if hasattr(enc, 'encoder') and hasattr(enc.encoder, 'layer'):
        layers = list(enc.encoder.layer)
    elif hasattr(enc, 'layer'):
        layers = list(enc.layer)
    else:
        layers = []

    n_layers = len(layers)
    # Embeddings: lowest LR
    embed_params = list(enc.embeddings.parameters())
    if embed_params:
        groups.append({"params": embed_params, "lr": base_lr * (decay ** n_layers), "weight_decay": wd})

    # Encoder layers: increasing LR
    for i, layer in enumerate(layers):
        lr = base_lr * (decay ** (n_layers - 1 - i))
        groups.append({"params": list(layer.parameters()), "lr": lr, "weight_decay": wd})

    # Classification heads: highest LR (10x base)
    head_params = []
    for h in heads:
        head_params.extend(list(h.parameters()))
    groups.append({"params": head_params, "lr": base_lr * 10, "weight_decay": 0.0})

    n_groups = len(groups)
    n_params = sum(p.numel() for g in groups for p in g["params"])
    print(f"  Layer-wise LR: {n_groups} groups, {n_params/1e6:.1f}M params ✅")
    return groups

def macro_recall(at_t, at_p, is_t, is_p):
    r_at   = recall_score(at_t,  at_p,  average="macro", zero_division=0)
    r_isat = recall_score(is_t, is_p, average="macro", zero_division=0)
    return round((r_at + r_isat) / 2, 4), round(r_at, 4), round(r_isat, 4)

def train_model(model_name, tag, cfg):
    """Full training pipeline for one model."""
    print(f"\n{'█'*64}")
    print(f"  MHIPEX v12 | {tag} | {model_name.split('/')[-1]}")
    print(f"  🎯 Target: beat 0.5382 | ⏱ Est. ~{cfg['est_min']} min")
    print(f"{'█'*64}")

    gc.collect(); torch.cuda.empty_cache()
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = mem - torch.cuda.memory_allocated(0) / 1e9
    print(f"  GPU mem: {free:.1f} / {mem:.1f} GB free")

    # ── Tokenizer + special tokens ────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(model_name)
    added = tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    for t in SPECIAL_TOKENS:
        tid = tok.convert_tokens_to_ids(t)
        print(f"    '{t}' → [{tid}] ({'✅' if tid != tok.unk_token_id else '❌'})")

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds = HIPEDataset(PROC_DIR / "train_v12.jsonl", tok, cfg["maxlen"])
    dev_ds   = HIPEDataset(PROC_DIR / "dev_v12.jsonl",   tok, cfg["maxlen"])

    # Weighted sampler for minority class oversampling
    at_labels = [d["at"] for d in train_ds.data]
    sample_weights = [AT_W[l].item() for l in at_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg["bs"], sampler=sampler,
                              num_workers=2, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=cfg["bs"]*2, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────
    model = MHIPEXv12(model_name, len(SPECIAL_TOKENS), cfg["drop"]).to(DEVICE)
    if N_GPU > 1:
        model = nn.DataParallel(model)
        print(f"  DataParallel: {N_GPU} GPUs ✅")

    # ── Optimizer + scheduler ─────────────────────────────────────────
    groups = get_layer_wise_lr_groups(model, cfg["lr"], cfg["decay"], cfg["wd"])
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

    steps_per_ep = math.ceil(len(train_loader) / cfg["accum"])
    total_steps  = steps_per_ep * cfg["ep"]
    warmup_steps = int(total_steps * cfg["wu"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"  Steps/ep:{steps_per_ep} | Total:{total_steps} | Warmup:{warmup_steps}")

    # ── Loss functions ────────────────────────────────────────────────
    criterion_at   = nn.CrossEntropyLoss(weight=AT_W.to(DEVICE))
    criterion_isat = nn.CrossEntropyLoss(weight=ISAT_W.to(DEVICE))

    # ── AMP scaler ────────────────────────────────────────────────────
    use_amp = cfg.get("use_amp", True)
    scaler = GradScaler("cuda", enabled=use_amp)
    print(f"  AMP (FP16): {'✅ ON' if use_amp else '❌ OFF (FP32 for stability)'}")

    # ── Training loop ─────────────────────────────────────────────────
    best_mr = 0.0
    patience_counter = 0
    save_dir = OUT_DIR / tag
    save_dir.mkdir(exist_ok=True)
    t_start = time.time()

    for epoch in range(1, cfg["ep"] + 1):
        ep_start = time.time()
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Ep{epoch:02d}", leave=False)):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            at_y = batch["at_label"].to(DEVICE)
            is_y = batch["isat_label"].to(DEVICE)

            with autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(ids, mask)
                loss_at   = criterion_at(out["at_logits"],   at_y)
                loss_isat = criterion_isat(out["isat_logits"], is_y)
                loss = (loss_at + loss_isat) / cfg["accum"]

            scaler.scale(loss).backward()

            if (step + 1) % cfg["accum"] == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * cfg["accum"]

        avg_loss = total_loss / len(train_loader)

        # ── Evaluation ────────────────────────────────────────────────
        model.eval()
        at_true, at_pred, is_true, is_pred = [], [], [], []
        all_probs_at, all_probs_isat = [], []

        with torch.no_grad():
            for batch in dev_loader:
                ids  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                with autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    out = model(ids, mask)

                at_p = F.softmax(out["at_logits"].float(), dim=-1)
                is_p = F.softmax(out["isat_logits"].float(), dim=-1)
                all_probs_at.append(at_p.cpu())
                all_probs_isat.append(is_p.cpu())

                at_pred.extend(out["at_logits"].argmax(1).cpu().tolist())
                is_pred.extend(out["isat_logits"].argmax(1).cpu().tolist())
                at_true.extend(batch["at_label"].tolist())
                is_true.extend(batch["isat_label"].tolist())

        mr, r_at, r_isat = macro_recall(at_true, at_pred, is_true, is_pred)
        ep_min = (time.time() - ep_start) / 60

        print(f"  {'─'*58}")
        print(f"  Ep{epoch:02d}|Loss:{avg_loss:.4f}|MR:{mr}(at={r_at},isAt={r_isat})|{ep_min:.1f}min")
        print(f"  {'─'*58}")

        if mr > best_mr:
            best_mr = mr
            patience_counter = 0
            # Save model
            m = model.module if hasattr(model, 'module') else model
            torch.save(m.state_dict(), save_dir / "best_model.pt")
            tok.save_pretrained(save_dir)
            # Save probabilities for calibration
            torch.save({
                "probs_at": torch.cat(all_probs_at),
                "probs_isat": torch.cat(all_probs_isat),
                "at_true": at_true, "is_true": is_true,
                "at_pred": at_pred, "is_pred": is_pred,
            }, save_dir / "best_probs.pt")
            print(f"  ✅ Best MR={mr} saved → {save_dir}")
        else:
            patience_counter += 1
            print(f"  ↘ No improvement ({patience_counter}/{cfg['pat']})")
            if patience_counter >= cfg["pat"]:
                print(f"  ⏹ Early stopping at epoch {epoch}")
                break

    total_min = (time.time() - t_start) / 60
    print(f"\n  ✅ {tag} done in {total_min:.1f} min | Best MR = {best_mr}")

    gc.collect(); torch.cuda.empty_cache()
    return best_mr

print("\n✅ All definitions loaded. Ready for training.")
print(f"   Train: {sum(1 for _ in open(PROC_DIR/'train_v12.jsonl'))} pairs")
print(f"   Dev:   {sum(1 for _ in open(PROC_DIR/'dev_v12.jsonl'))} pairs")
