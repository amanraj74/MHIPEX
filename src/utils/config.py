# All hyperparameters in one place - change here, affects everything

AT_LABEL_MAP   = {"FALSE": 0, "PROBABLE": 1, "TRUE": 2}   # 3-class
ISAT_LABEL_MAP = {"FALSE": 0, "TRUE": 1}                   # 2-class

AT_LABEL_NAMES   = ["FALSE", "PROBABLE", "TRUE"]
ISAT_LABEL_NAMES = ["FALSE", "TRUE"]

MODEL_NAME   = "xlm-roberta-base"
MAX_LENGTH   = 256
BATCH_SIZE   = 16
LR           = 2e-5
EPOCHS       = 10
WARMUP_RATIO = 0.1
DROPOUT      = 0.1
WEIGHT_DECAY = 0.01
SEED         = 42

TRAIN_FILES = {
    "en": "data/raw/hipe2026/data/sandbox/en-train.jsonl",
    "fr": "data/raw/hipe2026/data/sandbox/fr-train.jsonl",
    "de": "data/raw/hipe2026/data/sandbox/de-train.jsonl",
}
DEV_FILES = {
    "en": "data/raw/hipe2026/data/sandbox/en-dev.jsonl",
    "fr": "data/raw/hipe2026/data/sandbox/fr-dev.jsonl",
    "de": "data/raw/hipe2026/data/sandbox/de-dev.jsonl",
}
