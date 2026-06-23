import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.utils.config import MAX_LENGTH, MODEL_NAME

class HIPEDataset(Dataset):
    def __init__(self, filepath, tokenizer=None):
        self.records = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["input_text"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "at_label":       torch.tensor(rec["at_label"],   dtype=torch.long),
            "isat_label":     torch.tensor(rec["isat_label"], dtype=torch.long),
            "language":       rec["language"],
        }
