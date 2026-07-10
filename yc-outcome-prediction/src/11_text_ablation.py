"""11 — Text-composition ablation (user hypothesis): is a MiniLM embedding of
one_liner + industry/subindustry + tags a better variable than embedding the
long_description?

Three compositions, all scrubbed of outcome-revealing sentences:
  short:  one_liner + "Industry: industry / subindustry" + "Tags: ..."
          (tags_highlighted ignored: empty for 5,909/5,999 companies)
  long:   long_description only
  full:   one_liner + long_description (what 09/10 used)

Each embedded with all-MiniLM-L6-v2; evaluated alone and with structured
features. Same protocol as 10_final_eval (temporal CV, 2019-21 holdout,
portfolio top-k). Output: models/text_ablation_results.json,
figures/text_ablation.png.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

SEED = 42
rng = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parents[1]
PROC, RAW, MODELS, FIGS = (ROOT / "data/processed", ROOT / "data/raw",
                           ROOT / "models", ROOT / "figures")

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
SCRUB_PAT = re.compile(
    r"(?i)\b(acquired?|acquisition|acquires|acqui-?hired?|merged?|merger|"
    r"went\s+public|going\s+public|ipo'?d?|public\s+offering|"
    r"shut(ting)?\s+down|shutdown|closed\s+(down|its|in)|closing\s+down|"
    r"no\s+longer\s+(operating|active|in\s+operation|in\s+business)|"
    r"ceased\s+operations?|wound\s+down|winding\s+down|out\s+of\s+business|"
    r"defunct|sold\s+to|sold\s+the\s+company|exited\s+to|now\s+part\s+of|"
    r"joined\s+forces\s+with|discontinued)\b")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def scrub(t) -> str:
    if not isinstance(t, str) or not t:
        return ""
    return " ".join(s for s in SENT_SPLIT.split(t) if not SCRUB_PAT.search(s))


raw = {c["id"]: c for c in json.load(open(RAW / "yc_companies_all.json"))}
df = pd.read_parquet(PROC / "features.parquet")
df = df.sort_values(["batch_date", "id"]).reset_index(drop=True)

texts = {"short": [], "long": [], "full": []}
for cid in df["id"]:
    c = raw[cid]
    ol = scrub(c.get("one_liner") or "")
    ld = scrub(c.get("long_description") or "")
    ind = c.get("industry") or ""
    sub = c.get("subindustry") or ""
    tags = ", ".join(c.get("tags") or [])
    texts["short"].append(
        f"{ol} Industry: {ind} / {sub}. Tags: {tags}.".strip())
    texts["long"].append(ld)
    texts["full"].append(f"{ol} {ld}".strip())

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EMB = {k: model.encode(v, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=False).astype(np.float32)
       for k, v in texts.items()}
print("embedded:", {k: e.shape for k, e in EMB.items()}, flush=True)

mask = df.label_strict.notna().to_numpy()
strict = df[mask].copy()
strict["y"] = strict.label_strict.astype(int)
yr = strict.batch_date.dt.year.to_numpy()
dev_m, hold_m = yr <= 2018, yr >= 2019
dev, hold = strict[dev_m], strict[hold_m]
y_dev, y_hold = dev.y.to_numpy(), hold.y.to_numpy()
base_rate = float(y_hold.mean())
E = {k: e[mask] for k, e in EMB.items()}
E_dev = {k: e[dev_m] for k, e in E.items()}
E_hold = {k: e[hold_m] for k, e in E.items()}


def struct_matrix(train, test):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                               min_frequency=10, sparse_output=True), CAT_COLS),
         ("num", "passthrough", NUM_COLS)], sparse_threshold=1.0)
    tr, te = train.copy(), test.copy()
    for c in CAT_COLS:
        tr[c] = tr[c].fillna("Missing").astype(str)
        te[c] = te[c].fillna("Missing").astype(str)
    return sp.csr_matrix(pre.fit_transform(tr)), sp.csr_matrix(pre.transform(te))


def fit_xgb(Xd, yd, Xv, yv):
    best, best_auc = None, -1
    for depth in (3, 4):
        m = xgb.XGBClassifier(n_estimators=800, learning_rate=0.05,
                              max_depth=depth, subsample=0.8,
                              colsample_bytree=0.8, min_child_weight=5,
                              tree_method="hist", eval_metric="auc",
                              early_stopping_rounds=50, random_state=SEED,
                              n_jobs=4)
        m.fit(Xd, yd, eval_set=[(Xv, yv)], verbose=False)
        if m.best_score > best_auc:
            best, best_auc = m, m.best_score
    return best


FOLD_DEFS = [(2011, [2012, 2013]), (2013, [2014, 2015]), (2015, [2016]),
             (2016, [2017]), (2017, [2018])]
results = {"holdout_n": int(len(hold)), "holdout_base_rate": base_rate,
           "variants": {}}
dev_idx = np.arange(len(dev))
dyr = dev.batch_date.dt.year.to_numpy()

for comp in ("short", "long", "full"):
    for with_struct in (False, True):
        name = ("struct_" if with_struct else "") + f"minilm_{comp}"
        pooled_s, pooled_y = [], []
        for tr_end, te_years in FOLD_DEFS:
            tr_i = dev_idx[dyr <= tr_end]
            te_i = dev_idx[np.isin(dyr, te_years)]
            mats_tr = [sp.csr_matrix(E_dev[comp][tr_i])]
            mats_te = [sp.csr_matrix(E_dev[comp][te_i])]
            if with_struct:
                Str, Ste = struct_matrix(dev.iloc[tr_i], dev.iloc[te_i])
                mats_tr.insert(0, Str); mats_te.insert(0, Ste)
            Xtr = sp.hstack(mats_tr).tocsr()
            Xte = sp.hstack(mats_te).tocsr()
            cut = int(Xtr.shape[0] * 0.85)
            m = fit_xgb(Xtr[:cut], y_dev[tr_i][:cut], Xtr[cut:], y_dev[tr_i][cut:])
            pooled_s.append(m.predict_proba(Xte)[:, 1])
            pooled_y.append(y_dev[te_i])
        pooled_s, pooled_y = np.concatenate(pooled_s), np.concatenate(pooled_y)

        mats_d = [sp.csr_matrix(E_dev[comp])]
        mats_h = [sp.csr_matrix(E_hold[comp])]
        if with_struct:
            Sd, Sh = struct_matrix(dev, hold)
            mats_d.insert(0, Sd); mats_h.insert(0, Sh)
        Xd, Xh = sp.hstack(mats_d).tocsr(), sp.hstack(mats_h).tocsr()
        cut = int(Xd.shape[0] * 0.85)
        m = fit_xgb(Xd[:cut], y_dev[:cut], Xd[cut:], y_dev[cut:])
        s_hold = m.predict_proba(Xh)[:, 1]

        entry = {
            "temporal_cv_auc_pooled":
                round(float(roc_auc_score(pooled_y, pooled_s)), 4),
            "holdout_auc": round(float(roc_auc_score(y_hold, s_hold)), 4),
            "holdout_pr_auc":
                round(float(average_precision_score(y_hold, s_hold)), 4),
            "portfolios": {}}
        for k in (25, 53, 100):
            top = np.argsort(-s_hold)[:k]
            entry["portfolios"][k] = {
                "model_success_rate": round(float(y_hold[top].mean()), 4)}
        results["variants"][name] = entry
        print(name, entry["temporal_cv_auc_pooled"], entry["holdout_auc"],
              entry["portfolios"], flush=True)

for k in (25, 53, 100):
    draws = np.array([y_hold[rng.choice(len(hold), k, replace=False)].mean()
                      for _ in range(20000)])
    for v in results["variants"].values():
        e = v["portfolios"][k]
        e["lift_vs_random"] = round(e["model_success_rate"] / draws.mean(), 3)
        e["p_value"] = round(float((draws >= e["model_success_rate"]).mean()), 4)

with open(MODELS / "text_ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = list(results["variants"])
x = np.arange(len(names))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.bar(x - 0.2, [results["variants"][n]["temporal_cv_auc_pooled"] for n in names],
        0.4, label="temporal CV", color="#2c7fb8")
ax1.bar(x + 0.2, [results["variants"][n]["holdout_auc"] for n in names],
        0.4, label="holdout", color="#f4a261")
ax1.axhline(0.5, color="k", ls="--", lw=1)
ax1.set_xticks(x, names, rotation=25, ha="right")
ax1.set_ylim(0.45, 0.72); ax1.set_ylabel("ROC-AUC")
ax1.set_title("Text composition ablation — AUC"); ax1.legend()
r53 = [results["variants"][n]["portfolios"][53]["model_success_rate"] for n in names]
ax2.bar(x, r53, color="#1b9e77")
ax2.axhline(base_rate, color="k", ls="--", label=f"random ({base_rate:.1%})")
ax2.set_xticks(x, names, rotation=25, ha="right")
ax2.set_ylabel("top-decile success rate")
ax2.set_title("Portfolio top decile (k=53)"); ax2.legend()
fig.tight_layout()
fig.savefig(FIGS / "text_ablation.png", dpi=130)
print("saved models/text_ablation_results.json, figures/text_ablation.png")
