import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import recall_score, f1_score, classification_report
from tqdm import tqdm
import json, os, random
import numpy as np

from src.utils.config import *
from src.utils.dataset import HIPEDataset
from src.models.classifier import MHIPEXClassifier

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def macro_recall(at_true, at_pred, isat_true, isat_pred):
    r_at   = recall_score(at_true,   at_pred,   average="macro", zero_division=0)
    r_isat = recall_score(isat_true, isat_pred, average="macro", zero_division=0)
    return round((r_at + r_isat) / 2, 4)

def train():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading datasets...")
    train_ds = HIPEDataset("data/processed/train.jsonl", tokenizer)
    dev_ds   = HIPEDataset("data/processed/dev.jsonl",   tokenizer)
    print(f"  Train: {len(train_ds)} | Dev: {len(dev_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print("Loading model...")
    model = MHIPEXClassifier(MODEL_NAME, DROPOUT).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion_at   = nn.CrossEntropyLoss()
    criterion_isat = nn.CrossEntropyLoss()

    best_recall = 0.0
    os.makedirs("experiments/runs", exist_ok=True)

    print("\nStarting training...\n")
    for epoch in range(1, EPOCHS + 1):
        # ── TRAIN ──
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            at_labels      = batch["at_label"].to(device)
            isat_labels    = batch["isat_label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion_at(outputs["at_logits"], at_labels) + \
                   criterion_isat(outputs["isat_logits"], isat_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = round(total_loss / len(train_loader), 4)

        # ── EVAL ──
        model.eval()
        at_true, at_pred, isat_true, isat_pred = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch}/{EPOCHS} [Eval] "):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask)
                at_pred_b   = outputs["at_logits"].argmax(dim=1).cpu().tolist()
                isat_pred_b = outputs["isat_logits"].argmax(dim=1).cpu().tolist()
                at_true.extend(batch["at_label"].tolist())
                at_pred.extend(at_pred_b)
                isat_true.extend(batch["isat_label"].tolist())
                isat_pred.extend(isat_pred_b)

        mr = macro_recall(at_true, at_pred, isat_true, isat_pred)
        print(f"\nEpoch {epoch} | Loss: {avg_loss} | Macro-Recall: {mr}")

        if mr > best_recall:
            best_recall = mr
            torch.save(model.state_dict(), "experiments/runs/best_model.pt")
            print(f"  *** New best saved: {mr} ***")

        # Detailed report every 2 epochs
        if epoch % 2 == 0:
            print("\n[at] Classification Report:")
            print(classification_report(at_true, at_pred,
                  target_names=AT_LABEL_NAMES, zero_division=0))
            print("[isAt] Classification Report:")
            print(classification_report(isat_true, isat_pred,
                  target_names=ISAT_LABEL_NAMES, zero_division=0))

    print(f"\nTraining complete. Best Macro-Recall: {best_recall}")

if __name__ == "__main__":
    train()
