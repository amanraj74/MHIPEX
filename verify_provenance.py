import torch
import numpy as np
from itertools import product as iterproduct
from sklearn.metrics import recall_score
import json

def calibrate(probs_at, probs_isat, at_t, is_t):
    best_at_mr, best_at_preds = 0, None
    prob_range = np.arange(0.20, 0.60, 0.05)
    for t_prob, t_true in iterproduct(prob_range, prob_range):
        preds = np.zeros(len(probs_at), dtype=int)
        preds[probs_at[:, 1] >= t_prob] = 1
        preds[probs_at[:, 2] >= t_true] = 2
        mr = recall_score(at_t, preds, average='macro', zero_division=0)
        if mr > best_at_mr: best_at_mr = mr; best_at_preds = preds.copy()
            
    best_isat_mr, best_isat_preds = 0, None
    for t in np.arange(0.15, 0.65, 0.05):
        preds = np.zeros(len(probs_isat), dtype=int)
        preds[probs_isat[:, 1] >= t] = 1
        preds[best_at_preds == 0] = 0
        mr = recall_score(is_t, preds, average='macro', zero_division=0)
        if mr > best_isat_mr: best_isat_mr = mr; best_isat_preds = preds.copy()
            
    return (best_at_mr + best_isat_mr)/2, best_at_mr, best_isat_mr, best_at_preds, best_isat_preds

def main():
    print("--- TABLE 4 / TABLE 5 PROVENANCE VERIFICATION ---\n")
    try:
        hm_data = torch.load('experiments/runs/hmbert_v12/best_probs.pt')
        xr_data = torch.load('experiments/runs/xlmr_v12/best_probs.pt')
    except Exception as e:
        print("Could not load best_probs.pt files:", e)
        return

    hm_at, hm_is = hm_data['probs_at'].numpy(), hm_data['probs_isat'].numpy()
    xr_at, xr_is = xr_data['probs_at'].numpy(), xr_data['probs_isat'].numpy()
    at_true, is_true = np.array(hm_data['at_true']), np.array(hm_data['is_true'])

    # Table 4: hmBERT Single+calib (and Table 5: A4)
    mr, atr, isr, _, _ = calibrate(hm_at, hm_is, at_true, is_true)
    print(f"Table 4 row 3 (hmBERT Single+calib) / Table 5 A4:")
    print(f"Computed Unrounded: MR_at={atr:.5f}, MR_isAt={isr:.5f} => MR={mr:.5f}")
    
    # Table 4: XLM-R Single+calib
    mr, atr, isr, _, _ = calibrate(xr_at, xr_is, at_true, is_true)
    print(f"\nTable 4 row 4 (XLM-R Single+calib):")
    print(f"Computed Unrounded: MR_at={atr:.5f}, MR_isAt={isr:.5f} => MR={mr:.5f}")
    
    # Table 4: Fixed-beta Ensemble (and Table 5 A5b)
    ens_at_60 = 0.60 * hm_at + 0.40 * xr_at
    ens_is_60 = 0.60 * hm_is + 0.40 * xr_is
    mr, atr, isr, _, _ = calibrate(ens_at_60, ens_is_60, at_true, is_true)
    print(f"\nTable 4 row 5 (MHIPEX Fixed-beta=0.60) / Table 5 A5b:")
    print(f"Computed Unrounded: MR_at={atr:.5f}, MR_isAt={isr:.5f} => MR={mr:.5f}")
    
    # Table 5 A5a (Uniform Ensemble beta=0.50)
    ens_at_50 = 0.50 * hm_at + 0.50 * xr_at
    ens_is_50 = 0.50 * hm_is + 0.50 * xr_is
    mr, atr, isr, _, _ = calibrate(ens_at_50, ens_is_50, at_true, is_true)
    print(f"\nTable 5 A5a (Uniform Ensemble beta=0.50):")
    print(f"Computed Unrounded: MR_at={atr:.5f}, MR_isAt={isr:.5f} => MR={mr:.5f}")

    # Table 4 RLAE (and Table 5 A6)
    langs = []
    try:
        for line in open('proc/dev_v12.jsonl', encoding='utf-8'):
            obj = json.loads(line)
            l = obj.get('language') or obj.get('lang') or [k for k in obj.keys() if 'lang' in k.lower()]
            if type(l) == list: l = obj[l[0]]
            langs.append(l)
    except Exception as e:
        print("Could not load dev_v12.jsonl for languages:", e)
        return
        
    idx_en = np.array(langs) == 'en'
    idx_fr = np.array(langs) == 'fr'
    idx_de = np.array(langs) == 'de'

    all_at_preds = np.zeros(len(at_true), dtype=int)
    all_is_preds = np.zeros(len(is_true), dtype=int)

    # RLAE optimal weights
    # EN: at=0.65, isAt=0.70
    ens_at_en = 0.65 * hm_at[idx_en] + 0.35 * xr_at[idx_en]
    ens_is_en = 0.70 * hm_is[idx_en] + 0.30 * xr_is[idx_en]
    _, _, _, p_at, p_is = calibrate(ens_at_en, ens_is_en, at_true[idx_en], is_true[idx_en])
    all_at_preds[idx_en] = p_at; all_is_preds[idx_en] = p_is

    # FR: at=0.60, isAt=0.35
    ens_at_fr = 0.60 * hm_at[idx_fr] + 0.40 * xr_at[idx_fr]
    ens_is_fr = 0.35 * hm_is[idx_fr] + 0.65 * xr_is[idx_fr]
    _, _, _, p_at, p_is = calibrate(ens_at_fr, ens_is_fr, at_true[idx_fr], is_true[idx_fr])
    all_at_preds[idx_fr] = p_at; all_is_preds[idx_fr] = p_is

    # DE: at=0.60, isAt=0.30
    ens_at_de = 0.60 * hm_at[idx_de] + 0.40 * xr_at[idx_de]
    ens_is_de = 0.30 * hm_is[idx_de] + 0.70 * xr_is[idx_de]
    _, _, _, p_at, p_is = calibrate(ens_at_de, ens_is_de, at_true[idx_de], is_true[idx_de])
    all_at_preds[idx_de] = p_at; all_is_preds[idx_de] = p_is

    atr = recall_score(at_true, all_at_preds, average='macro', zero_division=0)
    isr = recall_score(is_true, all_is_preds, average='macro', zero_division=0)
    mr = (atr + isr) / 2
    
    print(f"\nTable 4 row 6 (MHIPEX-RLAE) / Table 5 A6:")
    print(f"Computed Unrounded: MR_at={atr:.5f}, MR_isAt={isr:.5f} => MR={mr:.5f}")

    print("\n--- MISSING FILES FOR TABLE 5 ---")
    print("I do not have the raw prediction files (best_probs.pt) for A0, A1, A2, and A3.")
    print("These correspond to older models from the ablation progression:")
    print("A0: hmBERT (v8 baseline)")
    print("A1: + DATE/LANG tokens")
    print("A2: + Multi-sample dropout")
    print("A3: + CLS+Mean dual pooling")
    print("To perfectly verify these, you would need to run this same evaluation script against their respective output tensors.")

if __name__ == "__main__":
    main()
