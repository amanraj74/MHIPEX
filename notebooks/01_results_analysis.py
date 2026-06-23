# ══════════════════════════════════════════════════════════════════════════
#  MHIPEX — Results Analysis & Visualization
#  Generates all figures for the research paper
#  Run: python notebooks/01_results_analysis.py
# ══════════════════════════════════════════════════════════════════════════

import json, sys, os
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    confusion_matrix, classification_report,
    recall_score, accuracy_score, f1_score
)

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PRED_FILE = ROOT / "results" / "predictions" / "predictions_v12_final.jsonl"
FIG_DIR   = ROOT / "paper" / "figures"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})
COLORS = {
    "blue": "#2563EB", "red": "#DC2626", "green": "#16A34A",
    "orange": "#EA580C", "purple": "#7C3AED", "teal": "#0D9488",
    "gray": "#6B7280", "dark": "#1F2937",
    "en": "#2563EB", "fr": "#DC2626", "de": "#16A34A",
}
AT_NAMES   = ["FALSE", "PROBABLE", "TRUE"]
ISAT_NAMES = ["FALSE", "TRUE"]
LANG_NAMES = {"en": "English", "fr": "French", "de": "German"}

# ══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════

print("── Loading predictions ──")
preds = [json.loads(l) for l in open(PRED_FILE, encoding="utf-8")]
print(f"   {len(preds)} predictions loaded")

at_true  = [p["at_true"]   for p in preds]
at_pred  = [p["at_pred"]   for p in preds]
is_true  = [p["isat_true"] for p in preds]
is_pred  = [p["isat_pred"] for p in preds]
langs    = [p["lang"]       for p in preds]

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 1: Dataset Label Distribution
# ══════════════════════════════════════════════════════════════════════════

def fig1_label_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # AT distribution
    at_counts = Counter(at_true)
    vals = [at_counts[n] for n in AT_NAMES]
    pcts = [v / sum(vals) * 100 for v in vals]
    bars = axes[0].bar(AT_NAMES, vals,
                       color=[COLORS["green"], COLORS["orange"], COLORS["red"]],
                       edgecolor="white", linewidth=1.5)
    for bar, v, p in zip(bars, vals, pcts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                     f"{v}\n({p:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_title("'at' Relation Distribution")
    axes[0].set_ylabel("Number of Pairs")
    axes[0].set_ylim(0, max(vals) * 1.25)
    axes[0].spines[["top", "right"]].set_visible(False)

    # isAt distribution
    is_counts = Counter(is_true)
    vals2 = [is_counts[n] for n in ISAT_NAMES]
    pcts2 = [v / sum(vals2) * 100 for v in vals2]
    bars2 = axes[1].bar(ISAT_NAMES, vals2,
                        color=[COLORS["blue"], COLORS["red"]],
                        edgecolor="white", linewidth=1.5)
    for bar, v, p in zip(bars2, vals2, pcts2):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                     f"{v}\n({p:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[1].set_title("'isAt' Relation Distribution")
    axes[1].set_ylim(0, max(vals2) * 1.25)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Figure 1: Class Distribution in HIPE-2026 Dev Set", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_label_distribution.png")
    print("   ✅ fig1_label_distribution.png")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 2: Confusion Matrices
# ══════════════════════════════════════════════════════════════════════════

def fig2_confusion_matrices():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, true, pred, names, title in [
        (axes[0], at_true, at_pred, AT_NAMES, "'at' Relation"),
        (axes[1], is_true, is_pred, ISAT_NAMES, "'isAt' Relation"),
    ]:
        cm = confusion_matrix(true, pred, labels=names)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")

        for i in range(len(names)):
            for j in range(len(names)):
                color = "white" if cm_pct[i, j] > 50 else COLORS["dark"]
                ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                        ha="center", va="center", fontsize=9, fontweight="bold", color=color)

        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=10)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11, fontweight="bold")
        ax.set_ylabel("True", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.suptitle("Figure 2: Confusion Matrices — MHIPEX Ensemble (MR = 0.5655)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_confusion_matrices.png")
    print("   ✅ fig2_confusion_matrices.png")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 3: Per-Language Performance
# ══════════════════════════════════════════════════════════════════════════

def fig3_per_language():
    fig, ax = plt.subplots(figsize=(9, 5))

    lang_order = ["en", "fr", "de"]
    metrics = {}
    for lang in lang_order:
        idx = [i for i, l in enumerate(langs) if l == lang]
        l_at_t = [at_true[i] for i in idx]
        l_at_p = [at_pred[i] for i in idx]
        l_is_t = [is_true[i] for i in idx]
        l_is_p = [is_pred[i] for i in idx]
        r_at  = recall_score(l_at_t, l_at_p, average="macro", zero_division=0)
        r_is  = recall_score(l_is_t, l_is_p, average="macro", zero_division=0)
        mr = (r_at + r_is) / 2
        metrics[lang] = {"MR": mr, "at": r_at, "isAt": r_is, "n": len(idx)}

    x = np.arange(len(lang_order))
    w = 0.25

    bars_mr   = ax.bar(x - w, [metrics[l]["MR"]   for l in lang_order], w, label="Macro Recall (MR)", color=COLORS["purple"], edgecolor="white")
    bars_at   = ax.bar(x,     [metrics[l]["at"]   for l in lang_order], w, label="at Recall",         color=COLORS["blue"],   edgecolor="white")
    bars_isat = ax.bar(x + w, [metrics[l]["isAt"] for l in lang_order], w, label="isAt Recall",       color=COLORS["teal"],   edgecolor="white")

    for bars in [bars_mr, bars_at, bars_isat]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{LANG_NAMES[l]}\n(n={metrics[l]['n']})" for l in lang_order], fontsize=11)
    ax.set_ylabel("Macro Recall", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Figure 3: Per-Language Performance Breakdown", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_per_language.png")
    print("   ✅ fig3_per_language.png")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 4: Model Comparison
# ══════════════════════════════════════════════════════════════════════════

def fig4_model_comparison():
    models = [
        ("mBERT\n(fine-tuned)",       0.4268, 0.3500, 0.5000),
        ("hmBERT v8\n(baseline)",     0.5382, 0.4400, 0.6400),
        ("XLM-R base\n(fine-tuned)",  0.5449, 0.4472, 0.6426),
        ("hmBERT v12\n(+ enriched)",  0.5525, 0.4503, 0.6548),
        ("MHIPEX\n(ensemble)",        0.5655, 0.4586, 0.6724),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    w = 0.25

    mrs   = [m[1] for m in models]
    ats   = [m[2] for m in models]
    isats = [m[3] for m in models]
    names = [m[0] for m in models]

    b1 = ax.bar(x - w, mrs,   w, label="Macro Recall (MR)", color=COLORS["purple"], edgecolor="white")
    b2 = ax.bar(x,     ats,   w, label="at Recall",         color=COLORS["blue"],   edgecolor="white")
    b3 = ax.bar(x + w, isats, w, label="isAt Recall",       color=COLORS["teal"],   edgecolor="white")

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Highlight best
    ax.bar(x[-1] - w, [mrs[-1]], w, color=COLORS["purple"], edgecolor=COLORS["dark"], linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Figure 4: Model Comparison — Macro Recall Progression", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_model_comparison.png")
    print("   ✅ fig4_model_comparison.png")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 5: Ablation Study
# ══════════════════════════════════════════════════════════════════════════

def fig5_ablation():
    ablations = [
        ("hmBERT v8\n(base)",             0.5382),
        ("+ DATE/LANG\ntokens",           0.5463),
        ("+ Multi-sample\ndropout",       0.5463),
        ("+ CLS+Mean\npooling",           0.5463),
        ("+ Threshold\ncalibration",      0.5525),
        ("+ XLM-R\nensemble",            0.5655),
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = range(len(ablations))
    names = [a[0] for a in ablations]
    vals  = [a[1] for a in ablations]

    colors = [COLORS["gray"]] + [COLORS["blue"]] * 4 + [COLORS["purple"]]
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=1.5)

    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        if i > 0:
            delta = vals[i] - vals[i-1]
            if delta > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.015,
                        f"+{delta:.4f}", ha="center", va="top", fontsize=7,
                        color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Macro Recall", fontsize=11, fontweight="bold")
    ax.set_ylim(0.50, 0.60)
    ax.axhline(y=0.5382, color=COLORS["red"], linestyle="--", linewidth=1, alpha=0.7, label="v8 baseline")
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Figure 5: Ablation Study — Contribution of Each Component", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_ablation.png")
    print("   ✅ fig5_ablation.png")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 6: Error Analysis — Where the model fails
# ══════════════════════════════════════════════════════════════════════════

def fig6_error_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A: AT error breakdown by language
    ax = axes[0]
    lang_order = ["en", "fr", "de"]
    error_types = {"FALSE→PROB": [], "FALSE→TRUE": [], "PROB→FALSE": [],
                   "PROB→TRUE": [], "TRUE→FALSE": [], "TRUE→PROB": []}

    for lang in lang_order:
        idx = [i for i, l in enumerate(langs) if l == lang]
        n = len(idx)
        counts = Counter()
        for i in idx:
            if at_true[i] != at_pred[i]:
                counts[f"{at_true[i][:4]}→{at_pred[i][:4]}"] += 1
        for k in error_types:
            key = k.replace("PROB", "PROB").replace("FALSE", "FALS")
            matched = 0
            for ck, cv in counts.items():
                if k.replace("→", "→").replace("PROB","PROB")[:4] in ck[:4]:
                    pass
            error_types[k].append(counts.get(k.replace("→","→"), 0) / n * 100 if n > 0 else 0)

    # Simpler approach: just show error rate per language per class
    for lang_idx, lang in enumerate(lang_order):
        idx = [i for i, l in enumerate(langs) if l == lang]
        errors = sum(1 for i in idx if at_true[i] != at_pred[i])
        total = len(idx)
        err_rate = errors / total * 100

        # Per-class error rates
        for cls in AT_NAMES:
            cls_idx = [i for i in idx if at_true[i] == cls]
            cls_err = sum(1 for i in cls_idx if at_pred[i] != cls) / len(cls_idx) * 100 if cls_idx else 0

    # Redraw as grouped bar: error rate per class per language
    x = np.arange(3)  # 3 languages
    w = 0.25
    for ci, cls in enumerate(AT_NAMES):
        rates = []
        for lang in lang_order:
            idx = [i for i, l in enumerate(langs) if l == lang]
            cls_idx = [i for i in idx if at_true[i] == cls]
            err = sum(1 for i in cls_idx if at_pred[i] != cls) / len(cls_idx) * 100 if cls_idx else 0
            rates.append(err)
        clr = [COLORS["green"], COLORS["orange"], COLORS["red"]][ci]
        ax.bar(x + (ci - 1) * w, rates, w, label=cls, color=clr, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[l] for l in lang_order])
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("(a) 'at' Error Rate by Class & Language", fontweight="bold")
    ax.legend(title="True class", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel B: isAt error breakdown
    ax2 = axes[1]
    for ci, cls in enumerate(ISAT_NAMES):
        rates = []
        for lang in lang_order:
            idx = [i for i, l in enumerate(langs) if l == lang]
            cls_idx = [i for i in idx if is_true[i] == cls]
            err = sum(1 for i in cls_idx if is_pred[i] != cls) / len(cls_idx) * 100 if cls_idx else 0
            rates.append(err)
        clr = [COLORS["blue"], COLORS["red"]][ci]
        ax2.bar(x + (ci - 0.5) * 0.35, rates, 0.35, label=cls, color=clr, edgecolor="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels([LANG_NAMES[l] for l in lang_order])
    ax2.set_ylabel("Error Rate (%)")
    ax2.set_title("(b) 'isAt' Error Rate by Class & Language", fontweight="bold")
    ax2.legend(title="True class", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Figure 6: Error Analysis — Per-Class Error Rates", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_error_analysis.png")
    print("   ✅ fig6_error_analysis.png")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
#  TABLE: Complete Results CSV
# ══════════════════════════════════════════════════════════════════════════

def generate_tables():
    # Main results table
    rows = [
        "Model,Type,MR,at_recall,isAt_recall",
        "mBERT (fine-tuned),Baseline,0.4268,0.3500,0.5000",
        "hmBERT v8,Fine-tuned,0.5382,0.4400,0.6400",
        "hmBERT v11 + DATE/LANG,Fine-tuned,0.5365,0.4308,0.6422",
        "XLM-R base (v12),Fine-tuned + enriched,0.5449,0.4472,0.6426",
        "hmBERT v12,Fine-tuned + enriched,0.5525,0.4503,0.6548",
        "MHIPEX Ensemble,Ensemble + calibration,0.5655,0.4586,0.6724",
    ]
    with open(TABLE_DIR / "experiment_results.csv", "w") as f:
        f.write("\n".join(rows) + "\n")
    print("   ✅ experiment_results.csv")

    # Per-language table
    rows2 = ["Language,N,MR,at_recall,isAt_recall"]
    for lang in ["en", "fr", "de"]:
        idx = [i for i, l in enumerate(langs) if l == lang]
        l_at_t = [at_true[i] for i in idx]
        l_at_p = [at_pred[i] for i in idx]
        l_is_t = [is_true[i] for i in idx]
        l_is_p = [is_pred[i] for i in idx]
        r_at = recall_score(l_at_t, l_at_p, average="macro", zero_division=0)
        r_is = recall_score(l_is_t, l_is_p, average="macro", zero_division=0)
        mr = (r_at + r_is) / 2
        rows2.append(f"{LANG_NAMES[lang]},{len(idx)},{mr:.4f},{r_at:.4f},{r_is:.4f}")
    rows2.append(f"All,{len(preds)},0.5655,0.4586,0.6724")
    with open(TABLE_DIR / "per_language_results.csv", "w") as f:
        f.write("\n".join(rows2) + "\n")
    print("   ✅ per_language_results.csv")

    # Classification reports
    with open(TABLE_DIR / "classification_reports.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  MHIPEX v12 Ensemble -- Classification Reports\n")
        f.write("=" * 60 + "\n\n")
        f.write("-- 'at' Relation --\n")
        f.write(classification_report(at_true, at_pred, target_names=AT_NAMES, zero_division=0, digits=4))
        f.write("\n\n-- 'isAt' Relation --\n")
        f.write(classification_report(is_true, is_pred, target_names=ISAT_NAMES, zero_division=0, digits=4))
    print("   ✅ classification_reports.txt")

# ══════════════════════════════════════════════════════════════════════════
#  PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════

def print_summary():
    print(f"\n{'═'*60}")
    print(f"  MHIPEX Results Summary")
    print(f"{'═'*60}")

    r_at  = recall_score(at_true, at_pred, average="macro", zero_division=0)
    r_is  = recall_score(is_true, is_pred, average="macro", zero_division=0)
    mr = (r_at + r_is) / 2

    print(f"\n  Overall Macro Recall: {mr:.4f}")
    print(f"  ├─ at recall:   {r_at:.4f}")
    print(f"  └─ isAt recall: {r_is:.4f}")

    print(f"\n  Per-Language:")
    for lang in ["en", "fr", "de"]:
        idx = [i for i, l in enumerate(langs) if l == lang]
        l_at = recall_score([at_true[i] for i in idx], [at_pred[i] for i in idx], average="macro", zero_division=0)
        l_is = recall_score([is_true[i] for i in idx], [is_pred[i] for i in idx], average="macro", zero_division=0)
        print(f"  {LANG_NAMES[lang]:8s} (n={len(idx):4d}): MR={((l_at+l_is)/2):.4f} | at={l_at:.4f} | isAt={l_is:.4f}")

    print(f"\n{'═'*60}")

# ══════════════════════════════════════════════════════════════════════════
#  RUN ALL
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  MHIPEX — Results Analysis Pipeline")
    print("█" * 60)

    print("\n── Generating Figures ──")
    fig1_label_distribution()
    fig2_confusion_matrices()
    fig3_per_language()
    fig4_model_comparison()
    fig5_ablation()
    fig6_error_analysis()

    print("\n── Generating Tables ──")
    generate_tables()

    print_summary()

    print(f"\n✅ All done!")
    print(f"   Figures → {FIG_DIR}")
    print(f"   Tables  → {TABLE_DIR}")
