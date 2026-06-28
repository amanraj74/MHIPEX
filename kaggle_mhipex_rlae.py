"""
MHIPEX — Phase 1: Novel Algorithm & Statistical Significance
Run on Kaggle. Requires CPU/GPU (CPU is fine, it's just inference/metrics).
Estimated runtime: ~5 minutes

Outputs:
1. Ensemble β Sensitivity Sweep
2. Relation-Specific Language-Adaptive Ensemble (RLAE)
3. Bootstrap Confidence Intervals (Statistical Significance)
"""

import json, os, urllib.request, re
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import recall_score
from itertools import product as iterproduct

# ══════════════════════════════════════════════════════════════════
#  PATHS & CONFIG
# ══════════════════════════════════════════════════════════════════
# Ensure these point to the directory containing hmbert_v12/ and xlmr_v12/
# If you uploaded them to Kaggle as a dataset, it might be:
# BASE_DIR = Path("/kaggle/input/YOUR-DATASET-NAME/mhipex_v12_results")
# Alternatively, if you uploaded the folders directly into the working dir:
BASE_DIR = Path(".")

HMBERT_PATH = BASE_DIR / "hmbert_v12" / "best_probs.pt"
XLMR_PATH = BASE_DIR / "xlmr_v12" / "best_probs.pt"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PROC_DIR = Path("proc")
PROC_DIR.mkdir(exist_ok=True)
OUT_DIR = Path("out_rlae")
OUT_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  1. DATA DOWNLOAD & PREPROCESSING (We need languages!)
# ══════════════════════════════════════════════════════════════════
BASE_URL = "https://raw.githubusercontent.com/hipe-eval/HIPE-2026-data/main/data/sandbox"
FILES = {
    "en-dev": f"{BASE_URL}/en-dev.jsonl",
    "fr-dev": f"{BASE_URL}/fr-dev.jsonl",
    "de-dev": f"{BASE_URL}/de-dev.jsonl",
}

print("\n── Downloading Dev Data ──")
for name, url in FILES.items():
    dst = DATA_DIR / f"{name}.jsonl"
    if not dst.exists():
        print(f"  Downloading {name}.jsonl ...")
        urllib.request.urlretrieve(url, dst)

def load_and_process(path, lang):
    records = []
    AT_MAP = {"FALSE": 0, "PROBABLE": 1, "TRUE": 2}
    ISAT_MAP = {"FALSE": 0, "TRUE": 1}
    for line in open(path, encoding="utf-8"):
        doc = json.loads(line)
        for pair in doc.get("sampled_pairs", []):
            at_raw = pair.get("at", "FALSE")
            isat_raw = pair.get("isAt", "FALSE")
            if at_raw not in AT_MAP or isat_raw not in ISAT_MAP: continue
            records.append({
                "at_label": AT_MAP[at_raw], "isat_label": ISAT_MAP[isat_raw],
                "lang": lang
            })
    return records

print("\n── Preprocessing Languages ──")
dev_path = PROC_DIR / "dev_v12.jsonl"
if not dev_path.exists():
    all_recs = []
    for lang in ["en", "fr", "de"]:
        all_recs.extend(load_and_process(DATA_DIR / f"{lang}-dev.jsonl", lang))
    with open(dev_path, "w", encoding="utf-8") as f:
        for r in all_recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
else:
    all_recs = [json.loads(l) for l in open(dev_path, encoding="utf-8")]
    
langs = np.array([r["lang"] for r in all_recs])
print(f"  Loaded {len(langs)} dev examples.")


# ══════════════════════════════════════════════════════════════════
#  2. LOAD PROBABILITIES
# ══════════════════════════════════════════════════════════════════
print("\n── Loading Probabilities ──")
if not HMBERT_PATH.exists() or not XLMR_PATH.exists():
    print(f"❌ ERROR: Could not find probability files.")
    print(f"Expected:\n  {HMBERT_PATH}\n  {XLMR_PATH}")
    print("Please check your BASE_DIR path!")
    exit(1)

hm_data = torch.load(HMBERT_PATH, weights_only=True)
xr_data = torch.load(XLMR_PATH, weights_only=True)

hm_at, hm_is = hm_data["probs_at"].numpy(), hm_data["probs_isat"].numpy()
xr_at, xr_is = xr_data["probs_at"].numpy(), xr_data["probs_isat"].numpy()
at_true, is_true = np.array(hm_data["at_true"]), np.array(hm_data["is_true"])


def get_mr(at_p, is_p, at_t, is_t):
    at_mr = recall_score(at_t, at_p, average="macro", zero_division=0)
    is_mr = recall_score(is_t, is_p, average="macro", zero_division=0)
    return round((at_mr + is_mr) / 2, 4), round(at_mr, 4), round(is_mr, 4)

def calibrate(probs_at, probs_isat, at_t, is_t):
    best_at_mr, best_at_preds = 0, []
    best_at_thresh = None
    prob_range = np.arange(0.20, 0.60, 0.05)
    
    # AT
    for t_prob, t_true in iterproduct(prob_range, prob_range):
        preds = np.zeros(len(probs_at), dtype=int)
        preds[probs_at[:, 1] >= t_prob] = 1
        preds[probs_at[:, 2] >= t_true] = 2
        
        mr = recall_score(at_t, preds, average="macro", zero_division=0)
        if mr > best_at_mr:
            best_at_mr = mr
            best_at_preds = preds.copy()
            best_at_thresh = (t_prob, t_true)
            
    # isAt
    best_isat_mr, best_isat_preds = 0, []
    best_isat_thresh = None
    for t in np.arange(0.15, 0.65, 0.05):
        preds = np.zeros(len(probs_isat), dtype=int)
        preds[probs_isat[:, 1] >= t] = 1
        preds[best_at_preds == 0] = 0 # Logical constraint
        
        mr = recall_score(is_t, preds, average="macro", zero_division=0)
        if mr > best_isat_mr:
            best_isat_mr = mr
            best_isat_preds = preds.copy()
            best_isat_thresh = t
            
    return (round((best_at_mr + best_isat_mr)/2, 4), round(best_at_mr, 4), round(best_isat_mr, 4), 
            best_at_preds, best_isat_preds, best_at_thresh, best_isat_thresh)


# ══════════════════════════════════════════════════════════════════
#  3. ENSEMBLE BETA SENSITIVITY (Global)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PRIORITY 2: ENSEMBLE β SENSITIVITY SWEEP")
print("=" * 60)

beta_results = []
best_global_mr = 0
best_global_preds_at = None
best_global_preds_isat = None

print(f"  {'β (hmBERT)':<12} | {'MR':<8} | {'at':<8} | {'isAt':<8}")
print(f"  {'-'*45}")
for beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    ens_at = beta * hm_at + (1 - beta) * xr_at
    ens_is = beta * hm_is + (1 - beta) * xr_is
    mr, atr, isr, atp, isp, th_a, th_i = calibrate(ens_at, ens_is, at_true, is_true)
    beta_results.append({"beta": beta, "MR": mr, "at_recall": atr, "isAt_recall": isr})
    
    marker = " ⭐" if mr > best_global_mr else ""
    print(f"  {beta:<12.2f} | {mr:<8.4f} | {atr:<8.4f} | {isr:<8.4f}{marker}")
    
    if mr > best_global_mr:
        best_global_mr = mr
        best_global_preds_at = atp
        best_global_preds_isat = isp

pd.DataFrame(beta_results).to_csv(OUT_DIR / "beta_sensitivity.csv", index=False)


# ══════════════════════════════════════════════════════════════════
#  4. PRIORITY 1: RLAE (Relation-Specific Language-Adaptive Ensemble)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PRIORITY 1: NOVEL RLAE ALGORITHM")
print("=" * 60)

rlae_preds_at = np.zeros_like(at_true)
rlae_preds_isat = np.zeros_like(is_true)

rlae_params = []
betas_to_search = np.arange(0.2, 0.85, 0.05)

for lang in ["en", "fr", "de"]:
    idx = (langs == lang)
    if not idx.any(): continue
    
    # ── Optimize 'at' for this language ──
    best_l_at_mr, best_l_at_preds, best_l_at_beta = 0, None, 0
    for b in betas_to_search:
        ens_at = b * hm_at[idx] + (1 - b) * xr_at[idx]
        mr, atr, isr, atp, isp, tha, thi = calibrate(ens_at, hm_is[idx], at_true[idx], is_true[idx])
        if atr > best_l_at_mr:
            best_l_at_mr = atr
            best_l_at_preds = atp
            best_l_at_beta = b
            
    # ── Optimize 'isAt' for this language ──
    best_l_is_mr, best_l_is_preds, best_l_is_beta = 0, None, 0
    for b in betas_to_search:
        ens_is = b * hm_is[idx] + (1 - b) * xr_is[idx]
        
        # Need to re-calibrate isAt using the BEST AT predictions for this language
        # to apply the logical constraint correctly
        best_is_mr_inner, best_isp = 0, None
        for t in np.arange(0.15, 0.65, 0.05):
            preds = np.zeros(len(ens_is), dtype=int)
            preds[ens_is[:, 1] >= t] = 1
            preds[best_l_at_preds == 0] = 0
            
            imr = recall_score(is_true[idx], preds, average="macro", zero_division=0)
            if imr > best_is_mr_inner:
                best_is_mr_inner = imr
                best_isp = preds
                
        if best_is_mr_inner > best_l_is_mr:
            best_l_is_mr = best_is_mr_inner
            best_l_is_preds = best_isp
            best_l_is_beta = b
            
    rlae_preds_at[idx] = best_l_at_preds
    rlae_preds_isat[idx] = best_l_is_preds
    
    rlae_params.append({
        "Language": lang,
        "Optimal_at_beta": best_l_at_beta,
        "Optimal_isAt_beta": best_l_is_beta,
        "at_recall": best_l_at_mr,
        "isAt_recall": best_l_is_mr
    })
    
    print(f"  {lang.upper()}: β_at={best_l_at_beta:.2f} (at={best_l_at_mr:.4f}) | β_isAt={best_l_is_beta:.2f} (isAt={best_l_is_mr:.4f})")

rlae_mr, rlae_at, rlae_is = get_mr(rlae_preds_at, rlae_preds_isat, at_true, is_true)

print(f"\n  Final RLAE Overall: MR={rlae_mr:.4f} | at={rlae_at:.4f} | isAt={rlae_is:.4f}")
print(f"  Gain over Fixed β={best_global_mr:.4f}: +{rlae_mr - best_global_mr:.4f} MR")
pd.DataFrame(rlae_params).to_csv(OUT_DIR / "rlae_parameters.csv", index=False)


# ══════════════════════════════════════════════════════════════════
#  5. PRIORITY 6: STATISTICAL SIGNIFICANCE (Bootstrap CI)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PRIORITY 6: STATISTICAL SIGNIFICANCE (Bootstrap CI)")
print("=" * 60)

n_bootstraps = 1000
np.random.seed(42)

def bootstrap_mr(preds_at, preds_isat, true_at, true_isat, n=1000):
    scores = []
    idx = np.arange(len(preds_at))
    for _ in range(n):
        sample = np.random.choice(idx, size=len(idx), replace=True)
        scores.append(get_mr(preds_at[sample], preds_isat[sample], true_at[sample], true_isat[sample])[0])
    return np.array(scores)

print("  Running 1000 bootstrap resamples (takes a few seconds)...")

# Get Baseline hmBERT predictions (beta=1.0)
_, _, _, base_atp, base_isp, _, _ = calibrate(hm_at, hm_is, at_true, is_true)
base_scores = bootstrap_mr(base_atp, base_isp, at_true, is_true)

# Get RLAE predictions
rlae_scores = bootstrap_mr(rlae_preds_at, rlae_preds_isat, at_true, is_true)

def report_stats(name, scores):
    mean = np.mean(scores)
    std = np.std(scores)
    ci_lower, ci_upper = np.percentile(scores, 2.5), np.percentile(scores, 97.5)
    print(f"  {name:<15}: {mean:.4f} ± {std:.4f} | 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    return mean, std, ci_lower, ci_upper

m1, s1, l1, u1 = report_stats("hmBERT Base", base_scores)
m2, s2, l2, u2 = report_stats("MHIPEX RLAE", rlae_scores)

# Paired test: % of times RLAE > Baseline
p_better = np.mean(rlae_scores > base_scores) * 100
print(f"\n  RLAE is better than Baseline in {p_better:.1f}% of bootstrap samples.")
if p_better > 95:
    print("  ✅ Difference is STATISTICALLY SIGNIFICANT (p < 0.05)")
else:
    print("  ⚠️ Difference is trending positive, but not strictly significant at p=0.05")

# Save summary
with open(OUT_DIR / "stats_summary.txt", "w") as f:
    f.write(f"hmBERT Base: {m1:.4f} ± {s1:.4f} | 95% CI: [{l1:.4f}, {u1:.4f}]\n")
    f.write(f"MHIPEX RLAE: {m2:.4f} ± {s2:.4f} | 95% CI: [{l2:.4f}, {u2:.4f}]\n")
    f.write(f"RLAE > Base: {p_better:.1f}%\n")

print("\n✅ All done! Download the 'out_rlae' folder.")
