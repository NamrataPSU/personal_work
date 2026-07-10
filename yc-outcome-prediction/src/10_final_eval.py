"""10 — Final evaluation: founder features + MiniLM embeddings.

Adds two new feature arms to the pipeline and re-runs temporal CV,
holdout evaluation, and the random-vs-model portfolio simulation:

  FOUNDER features (from 08_scrape_founders.py output):
    n_founders       total founders listed (active + departed) — set at founding
    solo_founder     n_founders == 1
    has_technical_founder  any founder title matching CTO/engineer/technical
    has_ceo_title    any founder titled CEO (role formalization signal)
    founders_missing no founder data on page (coverage indicator)
  LEAKY (diagnostic only, never features): n_inactive_founders — is_active
    reflects TODAY's status (departed founders correlate with dead/acquired).

  TEXT arm: all-MiniLM-L6-v2 embeddings (384d) of scrubbed descriptions
  (09_embed_st.py). Pretrained frozen encoder — nothing fit on our data.

Validation identical to 05/07: forward-chaining temporal CV on dev pool
(batches <= 2018), final holdout 2019-21, 20k random portfolios per k.

Outputs: models/final_eval_results.json, figures/final_temporal_cv.png,
figures/final_portfolios.png, reports/06_founders_transformer.md (written
separately from results).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

SEED = 42
rng = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
FIGS = ROOT / "figures"

CAT_COLS = ["batch_season", "industry", "subindustry", "primary_region"]
NUM_COLS = [
    "cohort_size", "n_industries", "n_tags",
    "tag_ai", "tag_saas", "tag_b2b", "tag_fintech", "tag_developer_tools",
    "tag_marketplace", "tag_consumer", "tag_machine_learning", "tag_healthcare",
    "tag_e_commerce", "tag_analytics", "tag_health_tech", "tag_education",
    "tag_productivity", "tag_open_source", "tag_ai_assistant", "tag_payments",
    "tag_climate", "tag_api", "tag_biotech", "tag_hardware", "tag_logistics",
    "tag_sales", "tag_enterprise_software", "tag_digital_health",
    "is_us", "is_bay_area", "is_remote", "is_international", "has_location",
    "one_liner_len", "long_description_len", "has_description",
    "has_former_names", "nonprofit", "app_video_public", "demo_day_video_public",
]
FOUNDER_COLS = ["n_founders", "solo_founder", "has_technical_founder",
                "has_ceo_title", "founders_missing"]
TECH_TITLE = re.compile(r"(?i)\b(cto|chief\s+tech|engineer|technical|"
                        r"vp\s+eng|head\s+of\s+eng|scientist)\b")
CEO_TITLE = re.compile(r"(?i)\bceo|chief\s+exec\b")

# ---------------------------------------------------------------- founders
recs = [json.loads(l) for l in open(RAW / "founders.jsonl") if l.strip()]
frows = []
for r in recs:
    if "error" in r:
        frows.append({"slug": r["slug"], "founders_missing": 1})
        continue
    fs = r["founders"]
    titles = " | ".join(str(f.get("title") or "") for f in fs)
    frows.append({
        "slug": r["slug"],
        "n_founders": len(fs),
        "solo_founder": int(len(fs) == 1),
        "has_technical_founder": int(bool(TECH_TITLE.search(titles))),
        "has_ceo_title": int(bool(CEO_TITLE.search(titles))),
        "founders_missing": int(len(fs) == 0),
        "leaky_n_inactive_founders": sum(
            1 for f in fs if f.get("is_active") is False),
    })
fdf = pd.DataFrame(frows)

slug_map = pd.DataFrame(
    [(c["id"], c["slug"]) for c in json.load(open(RAW / "yc_companies_all.json"))],
    columns=["id", "slug"])

df = pd.read_parquet(PROC / "features.parquet")
df = df.sort_values(["batch_date", "id"]).reset_index(drop=True)
df = df.merge(slug_map, on="id", how="left").merge(fdf, on="slug", how="left")
df["founders_missing"] = df["founders_missing"].fillna(1)
for c in ["n_founders", "solo_founder", "has_technical_founder", "has_ceo_title"]:
    df[c] = df[c].fillna(0)

emb = np.load(PROC / "st_minilm_scrubbed.npy")
assert len(emb) == len(df)

strict_mask = df.label_strict.notna().to_numpy()
strict = df[strict_mask].copy()
E = emb[strict_mask]
strict["y"] = strict.label_strict.astype(int)
yr = strict.batch_date.dt.year.to_numpy()
dev_m, hold_m = yr <= 2018, yr >= 2019
dev, hold = strict[dev_m].copy(), strict[hold_m].copy()
E_dev, E_hold = E[dev_m], E[hold_m]
y_dev, y_hold = dev.y.to_numpy(), hold.y.to_numpy()
base_rate = y_hold.mean()

cov = 1 - strict.founders_missing.mean()
print(f"founder coverage (strict cohort): {cov:.1%}; "
      f"dev n={len(dev)}, holdout n={len(hold)} base={base_rate:.3f}")
print(strict.groupby('n_founders', observed=True).y.agg(['count', 'mean']).head(8))

# ------------------------------------------------------------- matrices
def struct_matrix(train, test, extra_num):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                               min_frequency=10, sparse_output=True), CAT_COLS),
         ("num", "passthrough", NUM_COLS + extra_num)],
        sparse_threshold=1.0)
    tr, te = train.copy(), test.copy()
    for c in CAT_COLS:
        tr[c] = tr[c].fillna("Missing").astype(str)
        te[c] = te[c].fillna("Missing").astype(str)
    return sp.csr_matrix(pre.fit_transform(tr)), sp.csr_matrix(pre.transform(te))


def build(train, test, Etr, Ete, variant):
    use_f = "founders" in variant
    use_e = "minilm" in variant
    use_s = "struct" in variant
    mats_tr, mats_te = [], []
    if use_s or use_f:
        extra = FOUNDER_COLS if use_f else []
        if not use_s:
            # founders only: numeric founder cols, no struct
            Xtr = sp.csr_matrix(train[FOUNDER_COLS].to_numpy(dtype=float))
            Xte = sp.csr_matrix(test[FOUNDER_COLS].to_numpy(dtype=float))
            mats_tr.append(Xtr); mats_te.append(Xte)
        else:
            Xtr, Xte = struct_matrix(train, test, extra)
            mats_tr.append(Xtr); mats_te.append(Xte)
    if use_e:
        mats_tr.append(sp.csr_matrix(Etr)); mats_te.append(sp.csr_matrix(Ete))
    return sp.hstack(mats_tr).tocsr(), sp.hstack(mats_te).tocsr()


def fit_xgb(Xd, yd, Xv, yv):
    best, best_auc = None, -1
    for depth in (3, 4):
        m = xgb.XGBClassifier(
            n_estimators=800, learning_rate=0.05, max_depth=depth,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            tree_method="hist", eval_metric="auc",
            early_stopping_rounds=50, random_state=SEED, n_jobs=4)
        m.fit(Xd, yd, eval_set=[(Xv, yv)], verbose=False)
        if m.best_score > best_auc:
            best, best_auc = m, m.best_score
    return best

VARIANTS = ["struct", "struct_founders", "struct_minilm",
            "struct_founders_minilm", "minilm", "founders"]

FOLD_DEFS = [(2011, [2012, 2013]), (2013, [2014, 2015]), (2015, [2016]),
             (2016, [2017]), (2017, [2018])]
results = {"founder_coverage_strict": round(float(cov), 4),
           "holdout_n": int(len(hold)), "holdout_base_rate": float(base_rate),
           "variants": {}}

dev_idx = np.arange(len(dev))
for variant in VARIANTS:
    # temporal CV
    fold_aucs, pooled_s, pooled_y = [], [], []
    dyr = dev.batch_date.dt.year.to_numpy()
    for tr_end, te_years in FOLD_DEFS:
        tr_m, te_m = dyr <= tr_end, np.isin(dyr, te_years)
        tr_i, te_i = dev_idx[tr_m], dev_idx[te_m]
        cut = int(len(tr_i) * 0.85)
        Xtr, Xte = build(dev.iloc[tr_i], dev.iloc[te_i],
                         E_dev[tr_i], E_dev[te_i], variant)
        m = fit_xgb(Xtr[:cut], y_dev[tr_i][:cut], Xtr[cut:], y_dev[tr_i][cut:])
        s = m.predict_proba(Xte)[:, 1]
        fold_aucs.append(round(float(roc_auc_score(y_dev[te_i], s)), 4))
        pooled_s.append(s); pooled_y.append(y_dev[te_i])
    pooled_s, pooled_y = np.concatenate(pooled_s), np.concatenate(pooled_y)

    # holdout
    cut = int(len(dev) * 0.85)
    Xd, Xh = build(dev, hold, E_dev, E_hold, variant)
    m = fit_xgb(Xd[:cut], y_dev[:cut], Xd[cut:], y_dev[cut:])
    s_hold = m.predict_proba(Xh)[:, 1]

    entry = {
        "temporal_cv_auc_pooled": round(float(roc_auc_score(pooled_y, pooled_s)), 4),
        "temporal_cv_auc_folds": fold_aucs,
        "holdout_auc": round(float(roc_auc_score(y_hold, s_hold)), 4),
        "holdout_pr_auc": round(float(average_precision_score(y_hold, s_hold)), 4),
        "portfolios": {},
    }
    for k in (25, 53, 100):
        top = np.argsort(-s_hold)[:k]
        entry["portfolios"][k] = {"model_success_rate":
                                  round(float(y_hold[top].mean()), 4)}
    results["variants"][variant] = entry
    print(variant, entry["temporal_cv_auc_pooled"], entry["holdout_auc"],
          entry["portfolios"], flush=True)

# random benchmark
for k in (25, 53, 100):
    draws = np.array([y_hold[rng.choice(len(hold), k, replace=False)].mean()
                      for _ in range(20000)])
    for v in results["variants"].values():
        e = v["portfolios"][k]
        r = e["model_success_rate"]
        e["lift_vs_random"] = round(r / draws.mean(), 3)
        e["p_value"] = round(float((draws >= r).mean()), 4)

# leaky diagnostic, for the report only
leak = strict.groupby(strict.leaky_n_inactive_founders.fillna(0) > 0).y.mean()
results["diagnostic_inactive_founders_present_vs_not_success_rate"] = {
    str(k): round(float(v), 4) for k, v in leak.items()}

with open(MODELS / "final_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
names = VARIANTS
x = np.arange(len(names))
t = [results["variants"][n]["temporal_cv_auc_pooled"] for n in names]
h = [results["variants"][n]["holdout_auc"] for n in names]
ax1.bar(x - 0.2, t, 0.4, label="temporal CV (pooled)", color="#2c7fb8")
ax1.bar(x + 0.2, h, 0.4, label="holdout 2019-21", color="#f4a261")
ax1.axhline(0.5, color="k", ls="--", lw=1)
ax1.set_xticks(x, names, rotation=25, ha="right")
ax1.set_ylim(0.45, 0.72); ax1.set_ylabel("ROC-AUC")
ax1.set_title("AUC by variant"); ax1.legend()

k = 53
rates = [results["variants"][n]["portfolios"][k]["model_success_rate"] for n in names]
ax2.bar(x, rates, color="#1b9e77")
ax2.axhline(base_rate, color="k", ls="--", lw=1.2,
            label=f"random selection ({base_rate:.1%})")
ax2.set_xticks(x, names, rotation=25, ha="right")
ax2.set_ylabel("success rate, top decile (k=53)")
ax2.set_title("Portfolio: model top-decile vs random"); ax2.legend()
fig.tight_layout()
fig.savefig(FIGS / "final_eval.png", dpi=130)
print("saved models/final_eval_results.json, figures/final_eval.png")
