"""06_evaluate.py — figures + printed tables from models/metrics.json and
models/predictions.parquet. Writes figures/*.png (Agg backend)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

# reference palette (dataviz skill, light mode)
C = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100",
     "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e6e5e0", "surface": "#fcfcfb"}
VARIANT_COLORS = {
    "xgb_struct": C["blue"], "xgb_struct_text": C["aqua"], "xgb_text": C["yellow"],
    "xgb_text_raw": C["violet"], "logit_controls": C["green"],
    "xgb_devintage": C["red"], "xgb_struct_year": C["violet"], "prior": C["ink2"],
}
VARIANT_LABELS = {
    "prior": "prior baseline", "logit_controls": "logit (controls)",
    "xgb_struct": "XGB structured (main)", "xgb_struct_year": "XGB + batch_year",
    "xgb_devintage": "XGB de-vintaged", "xgb_struct_text": "XGB struct + text (scrubbed)",
    "xgb_text": "XGB text only (scrubbed)", "xgb_text_raw": "XGB text only (raw)",
}

plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink2"],
    "text.color": C["ink"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight",
})

M = json.load(open(ROOT / "models" / "metrics.json"))
IMP = json.load(open(ROOT / "models" / "importance.json"))
preds = pd.read_parquet(ROOT / "models" / "predictions.parquet")


def save(fig, name):
    fig.savefig(FIGS / name)
    plt.close(fig)
    print("wrote", FIGS / name)


# ---------------------------------------------- 1. temporal fold performance
fold_labels = [f"F{d['fold']}\n{'-'.join(str(y)[2:] for y in d['test_years'])}"
               for d in M["meta"]["fold_defs"]]
show = ["logit_controls", "xgb_struct", "xgb_struct_text", "xgb_text"]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for v in show:
    aucs = [f["roc_auc"] for f in M["strict"]["temporal"][v]["folds"]]
    ax.plot(range(1, 6), aucs, marker="o", ms=5, lw=2,
            color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])
ax.axhline(0.5, color=C["ink2"], lw=1, ls="--")
ax.annotate("chance (0.5)", (1.02, 0.503), fontsize=8, color=C["ink2"])
ax.set_xticks(range(1, 6), fold_labels)
ax.set_xlim(0.8, 5.2)
ax.set_ylabel("ROC-AUC (test block)")
ax.set_xlabel("forward-chaining fold (test batch years)")
ax.set_title("Temporal CV: per-fold ROC-AUC, label_strict", loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncols=2)
save(fig, "temporal_folds_auc.png")

# ---------------------------------------------- 2. random vs temporal gap
gap = M["strict"]["random_vs_temporal_gap"]
order = ["logit_controls", "xgb_struct", "xgb_struct_year", "xgb_devintage",
         "xgb_struct_text", "xgb_text", "xgb_text_raw"]
y = np.arange(len(order))
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.barh(y + 0.19, [gap[v]["random_auc"] for v in order], height=0.34,
        color=C["yellow"], label="random 5-fold CV")
ax.barh(y - 0.19, [gap[v]["temporal_auc"] for v in order], height=0.34,
        color=C["blue"], label="temporal CV")
for i, v in enumerate(order):
    ax.annotate(f"{gap[v]['random_auc']:.3f}", (gap[v]["random_auc"], i + 0.19),
                xytext=(3, 0), textcoords="offset points", va="center", fontsize=8, color=C["ink2"])
    ax.annotate(f"{gap[v]['temporal_auc']:.3f}  (gap +{gap[v]['gap']:.3f})",
                (gap[v]["temporal_auc"], i - 0.19),
                xytext=(3, 0), textcoords="offset points", va="center", fontsize=8, color=C["ink2"])
ax.set_yticks(y, [VARIANT_LABELS[v] for v in order])
ax.invert_yaxis()
ax.axvline(0.5, color=C["ink2"], lw=1, ls="--")
ax.set_xlim(0.45, 0.78)
ax.set_xlabel("pooled ROC-AUC (dev pool, batches <= 2018)")
ax.set_title("Random CV overstates skill vs temporal CV (vintage leakage)",
             loc="left", pad=26)
ax.legend(frameon=False, fontsize=9, loc="lower left",
          bbox_to_anchor=(0.0, 1.0), ncols=2)
save(fig, "random_vs_temporal_gap.png")

# ---------------------------------------------- 3. ROC + PR on holdout
hold = preds[(preds.label == "strict") & (preds.scheme == "holdout")]
base = M["strict"]["holdout_base_rate"]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
for v in ["logit_controls", "xgb_struct", "xgb_struct_text", "xgb_text"]:
    d = hold[hold.variant == v]
    fpr, tpr, _ = roc_curve(d.y, d.p)
    auc = M["strict"]["holdout"][v]["roc_auc"]
    axes[0].plot(fpr, tpr, lw=2, color=VARIANT_COLORS[v],
                 label=f"{VARIANT_LABELS[v]}  AUC={auc:.3f}")
    prec, rec, _ = precision_recall_curve(d.y, d.p)
    ap = M["strict"]["holdout"][v]["pr_auc"]
    axes[1].plot(rec, prec, lw=2, color=VARIANT_COLORS[v],
                 label=f"{VARIANT_LABELS[v]}  PR-AUC={ap:.3f}")
axes[0].plot([0, 1], [0, 1], ls="--", lw=1, color=C["ink2"])
axes[0].set(xlabel="false positive rate", ylabel="true positive rate",
            title="ROC — holdout (2019-21 batches)")
axes[1].axhline(base, ls="--", lw=1, color=C["ink2"])
axes[1].annotate(f"base rate {base:.3f}", (0.72, base - 0.05), fontsize=8, color=C["ink2"])
axes[1].set(xlabel="recall", ylabel="precision",
            title="Precision-recall — holdout", ylim=(0, 1))
for a in axes:
    a.legend(frameon=False, fontsize=7.8, loc="lower right")
    a.title.set_ha("left"); a.title.set_position((0, 1.02))
save(fig, "roc_pr_holdout.png")

# ---------------------------------------------- 4. calibration on holdout
fig, ax = plt.subplots(figsize=(5.4, 4.6))
for v in ["xgb_struct", "xgb_struct_text", "logit_controls"]:
    d = hold[hold.variant == v]
    frac, mean_p = calibration_curve(d.y, d.p, n_bins=8, strategy="quantile")
    br = M["strict"]["holdout"][v]["brier"]
    ax.plot(mean_p, frac, marker="o", ms=5, lw=2, color=VARIANT_COLORS[v],
            label=f"{VARIANT_LABELS[v]}  Brier={br:.3f}")
ax.plot([0, 1], [0, 1], ls="--", lw=1, color=C["ink2"])
ax.annotate("perfect calibration", (0.62, 0.57), rotation=38, fontsize=8, color=C["ink2"])
ax.set(xlabel="mean predicted probability (bin)", ylabel="observed success rate",
       xlim=(0, 1), ylim=(0, 1))
ax.set_title("Calibration on holdout: scores run hot (vintage shift)", loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
save(fig, "calibration_holdout.png")

# ---------------------------------------------- 5. importance top-20
perm = IMP["permutation_auc_drop"]
gain_src = IMP["gain_by_source_feature"]
top_perm = list(perm.items())[:20]
top_gain = list(gain_src.items())[:20]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.6))
names = [k for k, _ in top_perm][::-1]
vals = [v["mean"] for _, v in top_perm][::-1]
errs = [v["std"] for _, v in top_perm][::-1]
axes[0].barh(names, vals, xerr=errs, color=C["blue"], height=0.62,
             error_kw=dict(ecolor=C["ink2"], lw=1))
axes[0].set_title("Permutation importance (holdout AUC drop)", loc="left")
axes[0].set_xlabel("mean AUC drop, 10 permutations")
gnames = [k for k, _ in top_gain][::-1]
gvals = [v for _, v in top_gain][::-1]
axes[1].barh(gnames, gvals, color=C["aqua"], height=0.62)
axes[1].set_title("XGBoost gain (aggregated to source feature)", loc="left")
axes[1].set_xlabel("total gain")
fig.suptitle("Feature importance — final XGB structured model", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "importance_top20.png")

# ---------------------------------------------- 6. top-decile lift (holdout)
fig, ax = plt.subplots(figsize=(6.6, 3.8))
order2 = ["prior", "logit_controls", "xgb_struct", "xgb_struct_text", "xgb_text"]
precs = [M["strict"]["holdout"][v]["precision_at_top_decile"] for v in order2]
ax.barh([VARIANT_LABELS[v] for v in order2][::-1], precs[::-1],
        color=[VARIANT_COLORS[v] for v in order2][::-1], height=0.6)
ax.axvline(base, ls="--", lw=1.2, color=C["ink2"])
ax.set_ylim(-0.6, 5.1)
ax.annotate(f"holdout base rate {base:.3f}", (base + 0.006, 4.55),
            fontsize=8.5, color=C["ink2"])
for i, p in enumerate(precs[::-1]):
    lift = p / base
    ax.annotate(f"{p:.3f}  ({lift:.2f}x)", (p, i), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8.5, color=C["ink2"])
ax.set_xlim(0, 0.78)
ax.set_xlabel("precision among top 10% of holdout by model score")
ax.set_title("Top-decile precision on holdout (lift vs base rate)", loc="left")
save(fig, "top_decile_lift_holdout.png")

# ---------------------------------------------- printed summary tables
def table(scheme):
    rows = []
    for v, r in M["strict"][scheme].items():
        p = r["pooled"] if scheme != "holdout" else r
        rows.append({"variant": v, **{k: p[k] for k in
                     ["roc_auc", "pr_auc", "brier", "precision_at_top_decile",
                      "lift_at_top_decile"]}})
    return pd.DataFrame(rows)

for s in ["temporal", "random", "holdout"]:
    print(f"\n== strict / {s} ==")
    print(table(s).to_string(index=False))
print("\n== per-fold AUC, xgb_struct temporal ==")
print(pd.DataFrame(M["strict"]["temporal"]["xgb_struct"]["folds"])[
    ["fold", "train_n", "n", "base_rate", "roc_auc", "pr_auc", "brier",
     "precision_at_top_decile", "lift_at_top_decile"]].to_string(index=False))
