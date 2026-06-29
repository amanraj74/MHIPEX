"""
MHIPEX — Multi-Knowledge-Graph Augmented Relation Extraction
Run on Kaggle with GPU T4 x2 + Internet enabled
Estimated runtime: ~90 minutes (Trains E0, E1, E2, E3)
"""

# ══════════════════════════════════════════════════════════════════
#  CELL 1: Setup, Data Download, and KG Retrieval
# ══════════════════════════════════════════════════════════════════

import json, os, time, requests, gc, warnings, urllib.request, re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import recall_score
from pathlib import Path
from itertools import product as iterproduct

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PROC_DIR = Path("proc")
PROC_DIR.mkdir(exist_ok=True)
OUT_DIR = Path("out_multikg")
OUT_DIR.mkdir(exist_ok=True)

AT_MAP = {"FALSE": 0, "PROBABLE": 1, "TRUE": 2}
ISAT_MAP = {"FALSE": 0, "TRUE": 1}

# ── Data Download & Preprocessing ──
BASE_URL = "https://raw.githubusercontent.com/hipe-eval/HIPE-2026-data/main/data/sandbox"
FILES = {
    "en-train": f"{BASE_URL}/en-train.jsonl", "fr-train": f"{BASE_URL}/fr-train.jsonl", "de-train": f"{BASE_URL}/de-train.jsonl",
    "en-dev":   f"{BASE_URL}/en-dev.jsonl",   "fr-dev":   f"{BASE_URL}/fr-dev.jsonl",   "de-dev":   f"{BASE_URL}/de-dev.jsonl",
}

print("\n── Downloading Data ──")
for name, url in FILES.items():
    dst = DATA_DIR / f"{name}.jsonl"
    if not dst.exists():
        print(f"  Downloading {name}.jsonl ...")
        urllib.request.urlretrieve(url, dst)

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
            at_raw = pair.get("at", "FALSE")
            isat_raw = pair.get("isAt", "FALSE")
            if at_raw not in AT_MAP or isat_raw not in ISAT_MAP: continue
            records.append({
                "text": build_input_v12(doc["text"], pair["pers_mentions_list"], pair["loc_mentions_list"], date_str, lang),
                "at_label": at_raw, "isat_label": isat_raw,
                "pers_qid": pair.get("pers_wikidata_qid", pair.get("pers_qid", "")),
                "loc_qid": pair.get("loc_wikidata_qid", pair.get("loc_qid", "")),
                "lang": lang, "doc_id": doc["document_id"],
            })
    return records

print("\n── Preprocessing V12 Format ──")
for split in ["train", "dev"]:
    out_path = PROC_DIR / f"{split}_v12.jsonl"
    if not out_path.exists():
        all_recs = []
        for lang in ["en", "fr", "de"]:
            all_recs.extend(load_and_process(DATA_DIR / f"{lang}-{split}.jsonl", lang))
        with open(out_path, "w", encoding="utf-8") as f:
            for r in all_recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Processed {split}: {len(all_recs)} pairs")

# ── KG API Logic ──
WIKIDATA_API = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"
CACHE_FILE = OUT_DIR / "kg_cache.json"

def fetch_entity_from_wikidata(qid, cache):
    if qid in cache: return cache[qid]
    
    try:
        resp = requests.get(WIKIDATA_API.format(qid), timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("entities", {}).get(qid, {})
            claims = data.get("claims", {})
            
            def get_qids(prop):
                if prop not in claims: return []
                return [c["mainsnak"]["datavalue"]["value"]["id"] for c in claims[prop] if c["mainsnak"].get("datavalue", {}).get("type") == "wikibase-entityid"]
            
            def get_strings(prop):
                if prop not in claims: return []
                return [c["mainsnak"]["datavalue"]["value"] for c in claims[prop] if c["mainsnak"].get("datavalue", {}).get("type") == "string"]
                
            labels = data.get("labels", {})
            label = labels.get("en", labels.get("fr", labels.get("de", {}))).get("value", "")

            # General Wikidata Properties
            wd_props = []
            for p in ["P19", "P20", "P551", "P27", "P106", "P17", "P131"]: 
                wd_props.extend(get_qids(p))
                
            # GeoNames Properties (simulated via Wikidata mapping)
            gn_props = []
            gn_ids = get_strings("P1566")
            if gn_ids:
                gn_props.append(f"GeoNamesID_{gn_ids[0]}")
                gn_props.extend(get_qids("P17")) # Country
                gn_props.extend(get_qids("P31")) # Feature class
                
            # Getty TGN Properties (simulated via Wikidata mapping)
            tgn_props = []
            tgn_ids = get_strings("P1667")
            if tgn_ids:
                tgn_props.append(f"TGN_{tgn_ids[0]}")
                tgn_props.extend(get_qids("P131")) # Historical administrative hierarchy
                
            cache[qid] = {
                "label": label,
                "wikidata": wd_props,
                "geonames": gn_props,
                "getty": tgn_props
            }
            return cache[qid]
    except Exception:
        pass
    
    cache[qid] = {"label": "", "wikidata": [], "geonames": [], "getty": []}
    return cache[qid]

def resolve_labels(qids, cache):
    return [fetch_entity_from_wikidata(q, cache)["label"] for q in qids if q.startswith("Q") and fetch_entity_from_wikidata(q, cache)["label"]]

def build_kg_strings(pers_qid, loc_qid, cache):
    if pers_qid: fetch_entity_from_wikidata(pers_qid, cache)
    if loc_qid: fetch_entity_from_wikidata(loc_qid, cache)
    
    wd_facts, gn_facts, get_facts = [], [], []
    
    for qid in [pers_qid, loc_qid]:
        if not qid or qid not in cache: continue
        c = cache[qid]
        if c["wikidata"]: wd_facts.extend(resolve_labels(c["wikidata"], cache))
        if c["geonames"]: gn_facts.extend([x for x in c["geonames"] if not x.startswith("Q")] + resolve_labels([x for x in c["geonames"] if x.startswith("Q")], cache))
        if c["getty"]: get_facts.extend([x for x in c["getty"] if not x.startswith("Q")] + resolve_labels([x for x in c["getty"] if x.startswith("Q")], cache))
            
    return {
        "wikidata": " <WIKI> " + " | ".join(wd_facts[:8]) + " </WIKI>" if wd_facts else "",
        "geonames": " <GEONAMES> " + " | ".join(gn_facts[:8]) + " </GEONAMES>" if gn_facts else "",
        "getty": " <GETTY> " + " | ".join(get_facts[:8]) + " </GETTY>" if get_facts else ""
    }


print("\n Loading data and retrieving KG facts...")
cache = {}
if CACHE_FILE.exists():
    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    print(f"  Loaded {len(cache)} cached entities")

def load_and_augment(split):
    path = PROC_DIR / f"{split}_v12.jsonl"
    data = [json.loads(l) for l in open(path, encoding="utf-8")]

    for i, d in enumerate(data):
        pers_qid = d.get("pers_wikidata_qid", d.get("pers_qid", ""))
        loc_qid = d.get("loc_wikidata_qid", d.get("loc_qid", ""))
        kg_strings = build_kg_strings(pers_qid, loc_qid, cache)
        
        d["text_only"] = d["text"]
        d["wikidata"] = d["text"] + kg_strings["wikidata"]
        d["geonames"] = d["text"] + kg_strings["geonames"]
        d["getty"] = d["text"] + kg_strings["getty"]
        
        if (i + 1) % 200 == 0:
            print(f"    {split}: {i+1}/{len(data)} pairs processed")
            CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return data

train_data = load_and_augment("train")
dev_data = load_and_augment("dev")
print(f"  Wikidata cache: {len(cache)} entities")

# ══════════════════════════════════════════════════════════════════
#  CELL 2: Model + Training
# ══════════════════════════════════════════════════════════════════

class HIPEDataset(Dataset):
    def __init__(self, data, tokenizer, kg_type="text_only", max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.kg_type = kg_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        text = d[self.kg_type]
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len,
                           padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
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

def train_model(model_name, train_data, dev_data, kg_type, tag, epochs=25, lr=8e-6, bs=16):
    print(f"\n{'='*60}")
    print(f"  Training: {tag} | KG={kg_type} | {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special = ["<P>", "</P>", "<L>", "</L>", "<DATE>", "</DATE>", "<LANG>", "</LANG>", 
               "<WIKI>", "</WIKI>", "<GEONAMES>", "</GEONAMES>", "<GETTY>", "</GETTY>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special})

    model = MHIPEXClassifier(model_name).to(DEVICE)
    model.encoder.resize_token_embeddings(len(tokenizer))

    train_ds = HIPEDataset(train_data, tokenizer, kg_type=kg_type)
    dev_ds = HIPEDataset(dev_data, tokenizer, kg_type=kg_type)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2)
    dev_dl = DataLoader(dev_ds, batch_size=bs * 2, num_workers=2)

    at_w = torch.tensor([0.80, 1.50, 2.40], device=DEVICE)
    isat_w = torch.tensor([0.70, 2.60], device=DEVICE)
    at_loss_fn = nn.CrossEntropyLoss(weight=at_w)
    isat_loss_fn = nn.CrossEntropyLoss(weight=isat_w)

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

# ══════════════════════════════════════════════════════════════════
#  CELL 3: Run Experiments
# ══════════════════════════════════════════════════════════════════

MODEL = "dbmdz/bert-base-historic-multilingual-cased"
results = {}

experiments = [
    ("text_only", "E0_text_only"),
    ("wikidata", "E1_text_wikidata"),
    ("geonames", "E2_text_geonames"),
    ("getty", "E3_text_getty")
]

for kg_type, tag in experiments:
    results[tag] = train_model(MODEL, train_data, dev_data, kg_type=kg_type, tag=tag)

# ══════════════════════════════════════════════════════════════════
#  CELL 4: Calibration + Results
# ══════════════════════════════════════════════════════════════════

def calibrate(tag):
    data = torch.load(OUT_DIR / tag / "best_probs.pt", weights_only=True)
    probs_at = data["probs_at"].numpy()
    probs_isat = data["probs_isat"].numpy()
    at_true = data["at_true"]
    is_true = data["is_true"]

    best_mr, best_cfg = 0, {}
    prob_range = np.arange(0.20, 0.55, 0.05)
    best_at_mr, best_at_thresh, best_at_preds = 0, (0.5, 0.5), []

    for t_prob, t_true in iterproduct(prob_range, prob_range):
        preds = []
        for p in probs_at:
            if p[2] >= t_true: preds.append(2)
            elif p[1] >= t_prob: preds.append(1)
            else: preds.append(0)
        mr = recall_score(at_true, preds, average="macro", zero_division=0)
        if mr > best_at_mr:
            best_at_mr, best_at_thresh, best_at_preds = mr, (t_prob, t_true), preds

    best_isat_mr, best_isat_thresh = 0, 0.5
    for t in np.arange(0.15, 0.60, 0.05):
        preds = []
        for i, p in enumerate(probs_isat):
            if best_at_preds[i] == 0: preds.append(0)
            elif p[1] >= t: preds.append(1)
            else: preds.append(0)
        mr = recall_score(is_true, preds, average="macro", zero_division=0)
        if mr > best_isat_mr:
            best_isat_mr, best_isat_thresh = mr, t

    total_mr = round((best_at_mr + best_isat_mr) / 2, 4)
    return {"mr": total_mr, "at_mr": round(best_at_mr, 4), "isat_mr": round(best_isat_mr, 4)}

print("\n" + "=" * 64)
print("  MULTI KNOWLEDGE GRAPH EXPERIMENT RESULTS")
print("=" * 64)

for _, tag in experiments:
    save_path = OUT_DIR / tag / "best_probs.pt"
    if save_path.exists():
        cal = calibrate(tag)
        print(f"  {tag:20s} | MR: {cal['mr']:.4f} | at: {cal['at_mr']:.4f} | isAt: {cal['isat_mr']:.4f}")
        results[tag] = cal
    else:
        print(f"  {tag}: not found")

import csv
csv_path = OUT_DIR / "multi_kg_results.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Experiment", "MR", "at_recall", "isAt_recall"])
    for tag in [e[1] for e in experiments]:
        cal = results.get(tag, {})
        if isinstance(cal, dict):
            w.writerow([tag, cal.get("mr", ""), cal.get("at_mr", ""), cal.get("isat_mr", "")])
        else:
            w.writerow([tag, cal, "", ""])

print(f"\n  Results saved to: {csv_path}")
print(f"  Download 'multi_kg_results.csv' to update the paper!")
