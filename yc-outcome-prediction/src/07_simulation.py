"""07 — Portfolio simulation: random selection vs model-guided selection.

Question: an investor picks k companies from the 2019-21 holdout (n=523,
base success rate 40.2%, label_strict). What success rate does random
selection achieve vs picking the model's top-k scores?

Discipline (pre-investment features only):
  - No leaky_* columns (current team size / stage / hiring / top_company).
  - Text scrubbed of outcome-revealing sentences (same scrubber as 05_model).
  - All text representations fit on the dev pool (batches <= 2018) ONLY.

Text representations (huggingface.co unreachable from this environment, so
no pretrained sentence transformer — documented fallbacks, swappable later):
  - LSA:  TF-IDF(2000) -> TruncatedSVD(256), dev-fit.
  - W2V:  word2vec (100d) trained on dev-pool scrubbed text, SIF-weighted
          mean per document, first-PC removal (Arora et al. 2017), dev-fit.

Outputs: models/simulation_results.json, figures/simulation_portfolios.png,
figures/simulation_auc_by_variant.png.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from gensim.models import Word2Vec
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

SEED = 42
rng = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
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

SCRUB_PAT = re.compile(
    r"(?i)\b("
    r"acquired?|acquisition|acquires|acqui-?hired?|merged?|merger|"
    r"went\s+public|going\s+public|ipo'?d?|public\s+offering|"
    r"shut(ting)?\s+down|shutdown|closed\s+(down|its|in)|closing\s+down|"
    r"no\s+longer\s+(operating|active|in\s+operation|in\s+business)|"
    r"ceased\s+operations?|wound\s+down|winding\s+down|out\s+of\s+business|"
    r"defunct|sold\s+to|sold\s+the\s+company|exited\s+to|now\s+part\s+of|"
    r"joined\s+forces\s+with|discontinued"
    r")\b"
)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
TOKEN = re.compile(r"[a-z][a-z0-9\-]{1,}")


def scrub_text(t) -> str:
    if not isinstance(t, str) or not t:
        return ""
    kept = [s for s in SENT_SPLIT.split(t) if not SCRUB_PAT.search(s)]
    return " ".join(kept)


def tokenize(t: str) -> list[str]:
    return TOKEN.findall(t.lower())


# ---------------------------------------------------------------- data
df = pd.read_parquet(PROC / "features.parquet")
df = df.sort_values(["batch_date", "id"]).reset_index(drop=True)
df["text_scrubbed"] = df["text_clean"].fillna("").map(scrub_text)

strict = df[df.label_strict.notna()].copy()
strict["y"] = strict.label_strict.astype(int)
yr = strict.batch_date.dt.year
dev = strict[yr <= 2018].copy()
hold = strict[yr >= 2019].copy()
base_rate = hold.y.mean()
print(f"dev n={len(dev)} pos={dev.y.mean():.3f} | holdout n={len(hold)} pos={base_rate:.3f}")

# ------------------------------------------- text representations (dev-fit)
# LSA
tfidf = TfidfVectorizer(max_features=2000, min_df=5, stop_words="english",
                        sublinear_tf=True)
T_dev = tfidf.fit_transform(dev.text_scrubbed)
T_hold = tfidf.transform(hold.text_scrubbed)
svd = TruncatedSVD(n_components=256, random_state=SEED)
L_dev = svd.fit_transform(T_dev)
L_hold = svd.transform(T_hold)
print(f"LSA: 256 dims, explained variance {svd.explained_variance_ratio_.sum():.3f}")

# word2vec + SIF
dev_tokens = [tokenize(t) for t in dev.text_scrubbed]
hold_tokens = [tokenize(t) for t in hold.text_scrubbed]
w2v = Word2Vec(dev_tokens, vector_size=100, window=5, min_count=3,
               workers=4, epochs=30, seed=SEED)
freq = pd.Series([w for doc in dev_tokens for w in doc]).value_counts()
total = freq.sum()
A = 1e-3
sif_w = {w: A / (A + c / total) for w, c in freq.items()}


def sif_embed(docs: list[list[str]]) -> np.ndarray:
    out = np.zeros((len(docs), w2v.vector_size))
    for i, doc in enumerate(docs):
        vecs = [w2v.wv[w] * sif_w.get(w, 1.0) for w in doc if w in w2v.wv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


W_dev = sif_embed(dev_tokens)
W_hold = sif_embed(hold_tokens)
# first-PC removal, PC computed on dev only
u = TruncatedSVD(n_components=1, random_state=SEED).fit(W_dev).components_[0]
W_dev = W_dev - np.outer(W_dev @ u, u)
W_hold = W_hold - np.outer(W_hold @ u, u)
n_oov = int((np.abs(W_hold).sum(axis=1) == 0).sum())
print(f"w2v vocab {len(w2v.wv)}, holdout all-OOV/empty docs: {n_oov}")

# ------------------------------------------------------------ structured X
pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                           sparse_output=True), CAT_COLS),
     ("num", "passthrough", NUM_COLS)],
    sparse_threshold=1.0,
)
dev_p, hold_p = dev.copy(), hold.copy()
for c in CAT_COLS:
    dev_p[c] = dev_p[c].fillna("Missing").astype(str)
    hold_p[c] = hold_p[c].fillna("Missing").astype(str)
S_dev = sp.csr_matrix(pre.fit_transform(dev_p))
S_hold = sp.csr_matrix(pre.transform(hold_p))

VARIANTS = {
    "logit_struct": (S_dev, S_hold, "logit"),
    "xgb_struct": (S_dev, S_hold, "xgb"),
    "xgb_struct_lsa": (sp.hstack([S_dev, sp.csr_matrix(L_dev)]).tocsr(),
                       sp.hstack([S_hold, sp.csr_matrix(L_hold)]).tocsr(), "xgb"),
    "xgb_lsa": (sp.csr_matrix(L_dev), sp.csr_matrix(L_hold), "xgb"),
    "xgb_struct_w2v": (sp.hstack([S_dev, sp.csr_matrix(W_dev)]).tocsr(),
                       sp.hstack([S_hold, sp.csr_matrix(W_hold)]).tocsr(), "xgb"),
    "xgb_w2v": (sp.csr_matrix(W_dev), sp.csr_matrix(W_hold), "xgb"),
}

# early-stopping split: temporal tail (last 15% of dev by batch_date)
cut = int(len(dev) * 0.85)
tr_idx, va_idx = np.arange(cut), np.arange(cut, len(dev))
y_dev, y_hold = dev.y.to_numpy(), hold.y.to_numpy()


def fit_score(Xd, Xh, kind: str) -> np.ndarray:
    if kind == "logit":
        m = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                               C=0.5, max_iter=5000, random_state=SEED)
        m.fit(Xd, y_dev)
        return m.predict_proba(Xh)[:, 1]
    best, best_auc = None, -1.0
    for depth in (3, 4):
        m = xgb.XGBClassifier(
            n_estimators=800, learning_rate=0.05, max_depth=depth,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            tree_method="hist", eval_metric="auc",
            early_stopping_rounds=50, random_state=SEED, n_jobs=4,
        )
        m.fit(Xd[tr_idx], y_dev[tr_idx],
              eval_set=[(Xd[va_idx], y_dev[va_idx])], verbose=False)
        auc = m.best_score
        if auc > best_auc:
            best, best_auc = m, auc
    return best.predict_proba(Xh)[:, 1]


scores = {name: fit_score(Xd, Xh, kind) for name, (Xd, Xh, kind) in VARIANTS.items()}

# --------------------------------------------------------- Monte Carlo sim
N_SIM = 20_000
KS = [10, 25, 53, 100]
results = {"holdout_n": int(len(hold)), "holdout_base_rate": float(base_rate),
           "n_simulations": N_SIM, "variants": {}}

rand_dists = {}
for k in KS:
    draws = np.array([y_hold[rng.choice(len(hold), size=k, replace=False)].mean()
                      for _ in range(N_SIM)])
    rand_dists[k] = draws

for name, s in scores.items():
    auc = roc_auc_score(y_hold, s)
    pr = average_precision_score(y_hold, s)
    entry = {"holdout_auc": round(float(auc), 4),
             "holdout_pr_auc": round(float(pr), 4), "portfolios": {}}
    for k in KS:
        top = np.argsort(-s)[:k]
        model_rate = float(y_hold[top].mean())
        d = rand_dists[k]
        entry["portfolios"][k] = {
            "model_success_rate": round(model_rate, 4),
            "random_mean": round(float(d.mean()), 4),
            "random_p05": round(float(np.quantile(d, 0.05)), 4),
            "random_p95": round(float(np.quantile(d, 0.95)), 4),
            "lift_vs_random": round(model_rate / d.mean(), 3),
            "pct_of_random_draws_beaten": round(float((d < model_rate).mean()) * 100, 2),
            "p_value_random_geq_model": round(float((d >= model_rate).mean()), 4),
        }
    results["variants"][name] = entry

with open(MODELS / "simulation_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ----------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

show = ["xgb_struct", "xgb_struct_lsa", "xgb_struct_w2v"]
colors = {"xgb_struct": "#d95f02", "xgb_struct_lsa": "#7570b3",
          "xgb_struct_w2v": "#1b9e77"}
labels = {"xgb_struct": "XGB structured", "xgb_struct_lsa": "XGB struct+LSA",
          "xgb_struct_w2v": "XGB struct+w2v"}

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, k in zip(axes.ravel(), KS):
    d = rand_dists[k]
    ax.hist(d, bins=40, color="#bbbbbb", edgecolor="white", density=True,
            label=f"random portfolios (mean {d.mean():.1%})")
    for name in show:
        r = results["variants"][name]["portfolios"][k]["model_success_rate"]
        ax.axvline(r, color=colors[name], lw=2.2,
                   label=f"{labels[name]}: {r:.1%}")
    ax.set_title(f"portfolio size k={k}"
                 + ("  (top decile)" if k == 53 else ""))
    ax.set_xlabel("realized success rate")
    ax.legend(fontsize=8)
fig.suptitle("Random selection vs model top-k — 2019-21 holdout (n=523, base 40.2%), "
             f"{N_SIM:,} random draws per k", fontsize=12)
fig.tight_layout()
fig.savefig(FIGS / "simulation_portfolios.png", dpi=130)

fig2, ax = plt.subplots(figsize=(8, 4.5))
names = list(results["variants"])
aucs = [results["variants"][n]["holdout_auc"] for n in names]
ax.barh(names, aucs, color=["#666" if n.startswith("logit") else "#2c7fb8" for n in names])
ax.axvline(0.5, color="k", ls="--", lw=1, label="chance")
ax.set_xlim(0.45, 0.65)
ax.set_xlabel("holdout ROC-AUC")
ax.set_title("Holdout AUC by variant (text = scrubbed, dev-fit representations)")
for i, v in enumerate(aucs):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
ax.legend()
fig2.tight_layout()
fig2.savefig(FIGS / "simulation_auc_by_variant.png", dpi=130)

print(json.dumps(results, indent=2)[:3500])
print("saved: models/simulation_results.json, figures/simulation_portfolios.png,"
      " figures/simulation_auc_by_variant.png")
