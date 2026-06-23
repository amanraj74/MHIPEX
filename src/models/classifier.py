import torch
import torch.nn as nn
from transformers import AutoModel
from src.utils.config import MODEL_NAME

class MHIPEXClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.head_at    = nn.Linear(hidden, 3)  # FALSE / PROBABLE / TRUE
        self.head_isat  = nn.Linear(hidden, 2)  # FALSE / TRUE

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return {
            "at_logits":   self.head_at(cls),
            "isat_logits": self.head_isat(cls)
        }
