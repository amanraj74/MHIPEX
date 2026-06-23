# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MHIPEX v12 — CELL 2: Train Both Models                                ║
# ║  Runtime: ~80 min total on 1×T4 (hmBERT ~35min + XLM-R ~45min)        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── MODEL 1: hmBERT (historical multilingual BERT) ────────────────────
# Best for historical text. Your v8 scored 0.5382. Let's beat it.
HMBERT_CFG = {
    "bs": 32, "accum": 2, "lr": 8e-6, "decay": 0.90, "wd": 0.01,
    "pat": 8, "ep": 30, "wu": 0.12, "maxlen": 256,
    "drop": 0.15, "clip": 0.5, "est_min": 35, "use_amp": True,
}
mr_hmbert = train_model(
    "dbmdz/bert-base-historic-multilingual-cased",
    "hmbert_v12", HMBERT_CFG
)

# ── MODEL 2: XLM-RoBERTa-base (multilingual transformer) ─────────────
# use_amp=False because XLM-R + FP16 = NaN explosion (known issue)
XLMR_CFG = {
    "bs": 16, "accum": 4, "lr": 2e-5, "decay": 0.92, "wd": 0.01,
    "pat": 6, "ep": 25, "wu": 0.10, "maxlen": 256,
    "drop": 0.15, "clip": 1.0, "est_min": 60, "use_amp": False,
}
mr_xlmr = train_model(
    "xlm-roberta-base",
    "xlmr_v12", XLMR_CFG
)

print(f"\n{'═'*64}")
print(f"  TRAINING COMPLETE")
print(f"  hmBERT best MR: {mr_hmbert}")
print(f"  XLM-R  best MR: {mr_xlmr}")
print(f"{'═'*64}")
