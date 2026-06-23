# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MHIPEX v12 — CELL 3: Calibration + Ensemble + Final Results           ║
# ║  Runtime: ~5 min | No training needed, just post-processing            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import json, gc
import numpy as np
import torch, torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import recall_score, classification_report
from itertools import product as iterproduct

AT_NAMES   = ["FALSE", "PROBABLE", "TRUE"]
ISAT_NAMES = ["FALSE", "TRUE"]
OUT_DIR    = Path("out")
PROC_DIR   = Path("proc")

def load_probs(tag):
    """Load saved probabilities from training."""
    data = torch.load(OUT_DIR / tag / "best_probs.pt", weights_only=True)
    return data

def calibrate_thresholds(probs_at, probs_isat, at_true, is_true):
    """Grid-search optimal decision thresholds on dev set."""
    best_mr = 0
    best_cfg = {}

    # ── AT thresholds ─────────────────────────────────────────────────
    # For 3-class: threshold for PROBABLE and TRUE
    prob_range = np.arange(0.20, 0.60, 0.05)
    best_at_mr = 0
    best_at_thresh = (0.5, 0.5)

    for t_prob, t_true in iterproduct(prob_range, prob_range):
        preds = []
        for p in probs_at:
            if p[2] >= t_true:
                preds.append(2)
            elif p[1] >= t_prob:
                preds.append(1)
            else:
                preds.append(0)
        mr = recall_score(at_true, preds, average="macro", zero_division=0)
        if mr > best_at_mr:
            best_at_mr = mr
            best_at_thresh = (t_prob, t_true)
            best_at_preds = preds

    # ── isAt thresholds ───────────────────────────────────────────────
    best_isat_mr = 0
    best_isat_thresh = 0.5

    for t in np.arange(0.15, 0.65, 0.05):
        preds = []
        for i, p in enumerate(probs_isat):
            # Logical constraint: if at=FALSE → isAt must be FALSE
            if best_at_preds[i] == 0:
                preds.append(0)
            elif p[1] >= t:
                preds.append(1)
            else:
                preds.append(0)
        mr = recall_score(is_true, preds, average="macro", zero_division=0)
        if mr > best_isat_mr:
            best_isat_mr = mr
            best_isat_thresh = t
            best_isat_preds = preds

    total_mr = round((best_at_mr + best_isat_mr) / 2, 4)
    return {
        "at_thresh": best_at_thresh,
        "isat_thresh": best_isat_thresh,
        "at_preds": best_at_preds,
        "isat_preds": best_isat_preds,
        "mr": total_mr,
        "at_mr": round(best_at_mr, 4),
        "isat_mr": round(best_isat_mr, 4),
    }

# ══════════════════════════════════════════════════════════════════════════
#  LOAD AND CALIBRATE EACH MODEL
# ══════════════════════════════════════════════════════════════════════════

results = {}
for tag in ["hmbert_v12", "xlmr_v12"]:
    prob_path = OUT_DIR / tag / "best_probs.pt"
    if not prob_path.exists():
        print(f"  ⚠ {tag} not found, skipping")
        continue

    data = load_probs(tag)
    probs_at   = data["probs_at"].numpy()
    probs_isat = data["probs_isat"].numpy()

    # Baseline (argmax)
    base_mr, base_at, base_isat = recall_score(
        data["at_true"], data["at_pred"], average="macro", zero_division=0
    ), 0, 0
    at_r = recall_score(data["at_true"], data["at_pred"], average="macro", zero_division=0)
    is_r = recall_score(data["is_true"], data["is_pred"], average="macro", zero_division=0)
    base_mr = round((at_r + is_r) / 2, 4)

    print(f"\n{'═'*60}")
    print(f"  {tag}")
    print(f"  Baseline (argmax): MR = {base_mr} (at={at_r:.4f}, isAt={is_r:.4f})")

    # Calibrate
    cal = calibrate_thresholds(probs_at, probs_isat, data["at_true"], data["is_true"])
    print(f"  Calibrated:        MR = {cal['mr']} (at={cal['at_mr']}, isAt={cal['isat_mr']})")
    print(f"  Gain:             +{cal['mr'] - base_mr:.4f}")
    print(f"  AT thresh:  PROB≥{cal['at_thresh'][0]:.2f}, TRUE≥{cal['at_thresh'][1]:.2f}")
    print(f"  isAt thresh: TRUE≥{cal['isat_thresh']:.2f}")
    print(f"{'═'*60}")

    results[tag] = {
        "probs_at": probs_at,
        "probs_isat": probs_isat,
        "calibrated": cal,
        "base_mr": base_mr,
    }

# ══════════════════════════════════════════════════════════════════════════
#  WEIGHTED ENSEMBLE (if both models available)
# ══════════════════════════════════════════════════════════════════════════

if len(results) == 2:
    print(f"\n{'█'*60}")
    print(f"  WEIGHTED ENSEMBLE — Grid searching optimal alpha...")
    print(f"{'█'*60}")

    hmbert_data = load_probs("hmbert_v12")
    xlmr_data   = load_probs("xlmr_v12")

    at_true = hmbert_data["at_true"]
    is_true = hmbert_data["is_true"]

    best_ens_mr = 0
    best_alpha  = 0.5

    for alpha in np.arange(0.30, 0.75, 0.05):
        # Blend probabilities
        blended_at   = alpha * results["hmbert_v12"]["probs_at"]   + (1-alpha) * results["xlmr_v12"]["probs_at"]
        blended_isat = alpha * results["hmbert_v12"]["probs_isat"] + (1-alpha) * results["xlmr_v12"]["probs_isat"]

        # Calibrate the blended probs
        cal = calibrate_thresholds(blended_at, blended_isat, at_true, is_true)
        if cal["mr"] > best_ens_mr:
            best_ens_mr = cal["mr"]
            best_alpha  = alpha
            best_ens_cal = cal

    # Final ensemble with best alpha
    blended_at   = best_alpha * results["hmbert_v12"]["probs_at"]   + (1-best_alpha) * results["xlmr_v12"]["probs_at"]
    blended_isat = best_alpha * results["hmbert_v12"]["probs_isat"] + (1-best_alpha) * results["xlmr_v12"]["probs_isat"]
    best_ens_cal = calibrate_thresholds(blended_at, blended_isat, at_true, is_true)

    print(f"\n  Best alpha (hmBERT weight): {best_alpha:.2f}")
    print(f"  Ensemble MR: {best_ens_cal['mr']} (at={best_ens_cal['at_mr']}, isAt={best_ens_cal['isat_mr']})")

    # Use ensemble if it's better, otherwise use best single model
    best_single_tag = max(results, key=lambda k: results[k]["calibrated"]["mr"])
    best_single_mr  = results[best_single_tag]["calibrated"]["mr"]

    if best_ens_cal["mr"] > best_single_mr:
        final_at_preds   = best_ens_cal["at_preds"]
        final_isat_preds = best_ens_cal["isat_preds"]
        final_mr = best_ens_cal["mr"]
        final_method = f"Ensemble (α={best_alpha:.2f})"
    else:
        final_at_preds   = results[best_single_tag]["calibrated"]["at_preds"]
        final_isat_preds = results[best_single_tag]["calibrated"]["isat_preds"]
        final_mr = best_single_mr
        final_method = f"{best_single_tag} (single)"
else:
    # Only one model available
    tag = list(results.keys())[0]
    final_at_preds   = results[tag]["calibrated"]["at_preds"]
    final_isat_preds = results[tag]["calibrated"]["isat_preds"]
    final_mr = results[tag]["calibrated"]["mr"]
    final_method = f"{tag} (calibrated)"
    at_true = load_probs(tag)["at_true"]
    is_true = load_probs(tag)["is_true"]

# ══════════════════════════════════════════════════════════════════════════
#  FINAL RESULTS
# ══════════════════════════════════════════════════════════════════════════

print(f"\n{'█'*64}")
print(f"  ╔═══════════════════════════════════════════════════════════╗")
print(f"  ║              MHIPEX v12 — FINAL RESULTS                  ║")
print(f"  ╚═══════════════════════════════════════════════════════════╝")
print(f"{'█'*64}")

# Per-model results
print(f"\n  ── Individual Models ──")
for tag, r in results.items():
    print(f"  {tag:20s} | Baseline MR: {r['base_mr']:.4f} | Calibrated MR: {r['calibrated']['mr']:.4f}")

# Previous best
print(f"\n  ── Comparison ──")
print(f"  {'Previous best (v8)':20s} | MR: 0.5382")
print(f"  {'MHIPEX v12':20s} | MR: {final_mr:.4f}  ← {final_method}")
if final_mr > 0.5382:
    print(f"  🏆 NEW BEST! +{final_mr - 0.5382:.4f} improvement!")
else:
    print(f"  ⚠ Still below v8 by {0.5382 - final_mr:.4f}")

# Classification reports
print(f"\n  ── [at] Classification Report ──")
print(classification_report(at_true, final_at_preds,
      target_names=AT_NAMES, zero_division=0, digits=4))

print(f"  ── [isAt] Classification Report ──")
print(classification_report(is_true, final_isat_preds,
      target_names=ISAT_NAMES, zero_division=0, digits=4))

# Per-language breakdown
dev_data = [json.loads(l) for l in open(PROC_DIR / "dev_v12.jsonl", encoding="utf-8")]
print(f"\n  ── Per-Language Breakdown ──")
for lang in ["en", "fr", "de"]:
    indices = [i for i, d in enumerate(dev_data) if d["lang"] == lang]
    if not indices:
        continue
    l_at_t = [at_true[i] for i in indices]
    l_at_p = [final_at_preds[i] for i in indices]
    l_is_t = [is_true[i] for i in indices]
    l_is_p = [final_isat_preds[i] for i in indices]
    mr_l, at_l, isat_l = recall_score(l_at_t, l_at_p, average="macro", zero_division=0), 0, 0
    at_l  = recall_score(l_at_t, l_at_p, average="macro", zero_division=0)
    isat_l = recall_score(l_is_t, l_is_p, average="macro", zero_division=0)
    mr_l = round((at_l + isat_l) / 2, 4)
    print(f"  {lang}: MR={mr_l:.4f} | at={at_l:.4f} | isAt={isat_l:.4f} | N={len(indices)}")

# ══════════════════════════════════════════════════════════════════════════
#  SAVE PREDICTIONS (for paper / submission)
# ══════════════════════════════════════════════════════════════════════════

preds_out = []
for i, d in enumerate(dev_data):
    preds_out.append({
        "doc_id":    d["doc_id"],
        "lang":      d["lang"],
        "at_true":   AT_NAMES[at_true[i]],
        "at_pred":   AT_NAMES[final_at_preds[i]],
        "isat_true": ISAT_NAMES[is_true[i]],
        "isat_pred": ISAT_NAMES[final_isat_preds[i]],
    })

pred_file = OUT_DIR / "predictions_v12_final.jsonl"
with open(pred_file, "w", encoding="utf-8") as f:
    for r in preds_out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\n  ✅ Predictions saved → {pred_file}")

# ── Summary table for paper ───────────────────────────────────────────
print(f"\n{'═'*64}")
print(f"  TABLE FOR PAPER (copy this)")
print(f"{'═'*64}")
print(f"  {'Model':<30s} | {'MR':>6s} | {'at':>6s} | {'isAt':>6s}")
print(f"  {'─'*30}-+{'─'*8}+{'─'*8}+{'─'*8}")
for tag, r in results.items():
    cal = r["calibrated"]
    print(f"  {tag:<30s} | {cal['mr']:>6.4f} | {cal['at_mr']:>6.4f} | {cal['isat_mr']:>6.4f}")
if len(results) == 2:
    print(f"  {'Ensemble (calibrated)':<30s} | {best_ens_cal['mr']:>6.4f} | {best_ens_cal['at_mr']:>6.4f} | {best_ens_cal['isat_mr']:>6.4f}")
print(f"  {'─'*30}-+{'─'*8}+{'─'*8}+{'─'*8}")
print(f"  {'Previous best (v8 hmBERT)':<30s} | 0.5382 |  0.44  |  0.64")
print(f"\n✅ All done! Download 'out/' folder from Kaggle output.")
