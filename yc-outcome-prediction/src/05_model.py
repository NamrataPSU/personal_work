"""05_model.py — model training + validation for YC outcome prediction.

Design
------
Primary target: label_strict (Acquired|Public=1 vs Inactive=0), mature cohort.
Secondary sensitivity: label_survival.

Validation:
  * Final holdout = strict-cohort batches with batch_date >= 2019-01-01
    (2019-2021 vintages, untouched until final evaluation).
  * Primary CV = forward-chaining temporal CV on the dev pool (<= 2018):
      F1 train <=2011 -> test 2012-13   F2 train <=2013 -> test 2014-15
      F3 train <=2015 -> test 2016      F4 train <=2016 -> test 2017
      F5 train <=2017 -> test 2018
  * Secondary CV = stratified random 5-fold on the same dev pool; the
    random-vs-temporal gap quantifies vintage leakage.

Variants (all XGBoost unless noted):
  prior            majority/prior baseline (constant = train pos rate)
  logit_controls   elastic-net logistic on industry+region+season+cohort_size
  xgb_struct       MAIN: structured features (no batch_year)
  xgb_struct_year  de-vintage check: + batch_year as a feature
  xgb_devintage    de-vintage check: minus all batch-derived features
                   (batch_season, cohort_size)
  xgb_struct_text  structured + scrubbed TF-IDF (vectorizer refit per fold)
  xgb_text         scrubbed TF-IDF only
  xgb_text_raw     UNscrubbed TF-IDF only (quantifies text leakage)

Text scrubbing: sentences containing outcome-revealing phrases (acquired,
acquisition, IPO, went public, shut down, closed, no longer operating, ...)
are dropped before TF-IDF fitting/transform.

Outputs: models/metrics.json, models/predictions.parquet,
models/xgb_strict_final.json, models/preprocessor_final.joblib,
models/importance.json.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

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
BATCH_DERIVED = ["batch_season", "cohort_size"]

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


def scrub_text(t) -> str:
    """Drop whole sentences containing outcome-revealing phrases."""
    if not isinstance(t, str) or not t:
        return ""
    kept = [s for s in SENT_SPLIT.split(t) if not SCRUB_PAT.search(s)]
    return " ".join(kept)


# ---------------------------------------------------------------- data
df = pd.read_parquet(PROC / "features.parquet")
df = df.sort_values(["batch_date", "id"]).reset_index(drop=True)
df["text_raw"] = df["text_clean"].fillna("")
df["text_scrubbed"] = df["text_raw"].map(scrub_text)
n_scrubbed = int((df["text_scrubbed"] != df["text_raw"]).sum())

strict = df[df.label_strict.notna()].copy()
strict["y"] = strict.label_strict.astype(int)
surv = df[df.label_survival.notna()].copy()
surv["y"] = surv.label_survival.astype(int)

HOLDOUT_START = pd.Timestamp("2019-01-01")
FOLD_DEFS = [  # (train_end_year_inclusive, test_years)
    (2011, [2012, 2013]),
    (2013, [2014, 2015]),
    (2015, [2016]),
    (2016, [2017]),
    (2017, [2018]),
]


def temporal_folds(frame: pd.DataFrame):
    yr = frame.batch_date.dt.year
    for i, (tr_end, te_years) in enumerate(FOLD_DEFS, 1):
        tr = frame.index[yr <= tr_end]
        te = frame.index[yr.isin(te_years)]
        yield i, tr, te


def random_folds(frame: pd.DataFrame):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    idx = frame.index.to_numpy()
    for i, (tr, te) in enumerate(skf.split(idx, frame["y"]), 1):
        yield i, idx[tr], idx[te]


# ------------------------------------------------------- preprocessing
def make_preprocessor(num_cols, cat_cols):
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=10, sparse_output=True), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        sparse_threshold=1.0,
    )


def prep_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for c in CAT_COLS:
        out[c] = out[c].fillna("Missing").astype(str)
    return out


def build_X(train: pd.DataFrame, test: pd.DataFrame, variant: str):
    """Fit preprocessing on train only; return sparse X_train, X_test, names, fitted objects."""
    num = list(NUM_COLS)
    cat = list(CAT_COLS)
    use_text, text_col, use_struct = False, "text_scrubbed", True
    if variant == "xgb_struct_year":
        num = num + ["batch_year"]
    elif variant == "xgb_devintage":
        num = [c for c in num if c not in BATCH_DERIVED]
        cat = [c for c in cat if c not in BATCH_DERIVED]
    elif variant == "xgb_struct_text":
        use_text = True
    elif variant == "xgb_text":
        use_text, use_struct = True, False
    elif variant == "xgb_text_raw":
        use_text, use_struct, text_col = True, False, "text_raw"

    mats_tr, mats_te, names = [], [], []
    pre = vec = None
    if use_struct:
        pre = make_preprocessor(num, cat)
        tr_p, te_p = prep_frame(train), prep_frame(test)
        mats_tr.append(sp.csr_matrix(pre.fit_transform(tr_p)))
        mats_te.append(sp.csr_matrix(pre.transform(te_p)))
        names += list(pre.get_feature_names_out())
    if use_text:
        vec = TfidfVectorizer(max_features=2000, min_df=5, stop_words="english",
                              sublinear_tf=True)
        mats_tr.append(vec.fit_transform(train[text_col]))
        mats_te.append(vec.transform(test[text_col]))
        names += [f"tfidf__{w}" for w in vec.get_feature_names_out()]
    Xtr = sp.hstack(mats_tr, format="csr") if len(mats_tr) > 1 else mats_tr[0]
    Xte = sp.hstack(mats_te, format="csr") if len(mats_te) > 1 else mats_te[0]
    return Xtr, Xte, names, pre, vec


# ------------------------------------------------------------- models
XGB_GRID = [dict(max_depth=3, learning_rate=0.05), dict(max_depth=4, learning_rate=0.05)]
XGB_FIXED = dict(
    n_estimators=800, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_lambda=1.0, tree_method="hist", eval_metric="auc",
    early_stopping_rounds=50, random_state=SEED, n_jobs=4,
)


def fit_xgb(Xtr, ytr, temporal: bool, dates=None):
    """Light tuning: 2-point grid, early stopping on an inner 15% val split
    (temporal tail for temporal folds, stratified random otherwise)."""
    if temporal:
        order = np.argsort(dates.to_numpy(), kind="stable")
        cut = int(len(order) * 0.85)
        itr, iva = order[:cut], order[cut:]
    else:
        itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.15,
                                    stratify=ytr, random_state=SEED)
    best, best_auc = None, -1.0
    for g in XGB_GRID:
        m = XGBClassifier(**XGB_FIXED, **g)
        m.fit(Xtr[itr], ytr[itr], eval_set=[(Xtr[iva], ytr[iva])], verbose=False)
        auc = roc_auc_score(ytr[iva], m.predict_proba(Xtr[iva])[:, 1])
        if auc > best_auc:
            best, best_auc = m, auc
    return best


def fit_predict(variant: str, train: pd.DataFrame, test: pd.DataFrame,
                temporal: bool):
    ytr = train["y"].to_numpy()
    if variant == "prior":
        return np.full(len(test), ytr.mean())
    if variant == "logit_controls":
        cols_cat = ["industry", "primary_region", "batch_season"]
        pipe = Pipeline([
            ("prep", ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                      min_frequency=10), cols_cat),
                ("num", StandardScaler(), ["cohort_size"]),
            ])),
            ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                       l1_ratio=0.5, C=1.0, max_iter=5000,
                                       random_state=SEED)),
        ])
        pipe.fit(prep_frame(train), ytr)
        return pipe.predict_proba(prep_frame(test))[:, 1]
    Xtr, Xte, _, _, _ = build_X(train, test, variant)
    m = fit_xgb(Xtr, ytr, temporal, dates=train["batch_date"])
    return m.predict_proba(Xte)[:, 1]


# ------------------------------------------------------------- metrics
def top_decile(y, p):
    k = max(1, int(np.ceil(0.10 * len(y))))
    idx = np.argsort(-p, kind="stable")[:k]
    return float(np.mean(np.asarray(y)[idx])), k


def compute_metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    base = float(y.mean())
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "base_rate": round(base, 4),
           "brier": round(float(brier_score_loss(y, p)), 4)}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = round(float(roc_auc_score(y, p)), 4)
        out["pr_auc"] = round(float(average_precision_score(y, p)), 4)
        prec, k = top_decile(y, p)
        out["precision_at_top_decile"] = round(prec, 4)
        out["top_decile_n"] = k
        out["lift_at_top_decile"] = round(prec / base, 3) if base > 0 else None
    else:
        out.update({"roc_auc": None, "pr_auc": None,
                    "precision_at_top_decile": None, "lift_at_top_decile": None})
    return out


# ------------------------------------------------------- run CV suites
VARIANTS = ["prior", "logit_controls", "xgb_struct", "xgb_struct_year",
            "xgb_devintage", "xgb_struct_text", "xgb_text", "xgb_text_raw"]

pred_rows = []
metrics = {"meta": {
    "seed": SEED,
    "holdout_start": str(HOLDOUT_START.date()),
    "fold_defs": [{"fold": i + 1, "train_through": d[0], "test_years": d[1]}
                  for i, d in enumerate(FOLD_DEFS)],
    "n_rows_text_scrubbed_of_5999": n_scrubbed,
    "strict": {"n": int(len(strict)), "pos_rate": round(float(strict.y.mean()), 4)},
    "survival": {"n": int(len(surv)), "pos_rate": round(float(surv.y.mean()), 4)},
}}


def run_suite(frame: pd.DataFrame, label_name: str, variants):
    dev = frame[frame.batch_date < HOLDOUT_START]
    hold = frame[frame.batch_date >= HOLDOUT_START]
    res = {"dev_n": int(len(dev)), "holdout_n": int(len(hold)),
           "dev_base_rate": round(float(dev.y.mean()), 4),
           "holdout_base_rate": round(float(hold.y.mean()), 4),
           "temporal": {}, "random": {}, "holdout": {}}
    for variant in variants:
        for scheme, folds in [("temporal", temporal_folds(dev)),
                              ("random", random_folds(dev))]:
            fold_metrics, ys, ps = [], [], []
            for i, tr_idx, te_idx in folds:
                tr, te = dev.loc[tr_idx], dev.loc[te_idx]
                p = fit_predict(variant, tr, te, temporal=(scheme == "temporal"))
                fm = compute_metrics(te["y"], p)
                fm["fold"] = i
                fm["train_n"] = int(len(tr))
                fold_metrics.append(fm)
                ys.append(te["y"].to_numpy()); ps.append(p)
                pred_rows.extend(
                    {"label": label_name, "variant": variant, "scheme": scheme,
                     "fold": i, "id": int(r), "y": int(yy), "p": float(pp)}
                    for r, yy, pp in zip(te["id"], te["y"], p))
            pooled = compute_metrics(np.concatenate(ys), np.concatenate(ps))
            res[scheme][variant] = {"folds": fold_metrics, "pooled": pooled}
            print(f"[{label_name}] {variant:16s} {scheme:8s} pooled "
                  f"AUC={pooled['roc_auc']} PR={pooled['pr_auc']} "
                  f"Brier={pooled['brier']}")
        # final holdout: train on full dev pool
        p_hold = fit_predict(variant, dev, hold, temporal=True)
        hm = compute_metrics(hold["y"], p_hold)
        res["holdout"][variant] = hm
        pred_rows.extend(
            {"label": label_name, "variant": variant, "scheme": "holdout",
             "fold": 0, "id": int(r), "y": int(yy), "p": float(pp)}
            for r, yy, pp in zip(hold["id"], hold["y"], p_hold))
        print(f"[{label_name}] {variant:16s} HOLDOUT  "
              f"AUC={hm['roc_auc']} PR={hm['pr_auc']} Brier={hm['brier']} "
              f"P@10%={hm['precision_at_top_decile']} lift={hm['lift_at_top_decile']}")
    return res


print("=== label_strict (primary) ===")
metrics["strict"] = run_suite(strict, "strict", VARIANTS)
print("=== label_survival (sensitivity) ===")
metrics["survival"] = run_suite(surv, "survival", ["prior", "xgb_struct"])

# random-vs-temporal gap summary
gap = {}
for v in VARIANTS:
    r = metrics["strict"]["random"][v]["pooled"]["roc_auc"]
    t = metrics["strict"]["temporal"][v]["pooled"]["roc_auc"]
    if r is not None and t is not None:
        gap[v] = {"random_auc": r, "temporal_auc": t, "gap": round(r - t, 4)}
metrics["strict"]["random_vs_temporal_gap"] = gap

# --------------------------------------------- final model + artifacts
print("=== final model (xgb_struct, train on dev pool <=2018) ===")
dev = strict[strict.batch_date < HOLDOUT_START]
hold = strict[strict.batch_date >= HOLDOUT_START]
Xtr, Xho, feat_names, pre_final, _ = build_X(dev, hold, "xgb_struct")
final = fit_xgb(Xtr, dev["y"].to_numpy(), temporal=True, dates=dev["batch_date"])
final.get_booster().feature_names = [re.sub(r"[\[\]<>]", "_", n) for n in feat_names]
final.get_booster().save_model(str(MODELS / "xgb_strict_final.json"))
joblib.dump({"preprocessor": pre_final, "feature_names": feat_names,
             "num_cols": NUM_COLS, "cat_cols": CAT_COLS},
            MODELS / "preprocessor_final.joblib")

# gain importance, aggregated to source features
gain = final.get_booster().get_score(importance_type="gain")
name_map = dict(zip(final.get_booster().feature_names, feat_names))


def source_feature(transformed: str) -> str:
    if transformed.startswith("num__"):
        return transformed[5:]
    if transformed.startswith("cat__"):
        rest = transformed[5:]
        for c in CAT_COLS:
            if rest.startswith(c + "_"):
                return c
    return transformed


gain_src = {}
for k, v in gain.items():
    gain_src[source_feature(name_map[k])] = gain_src.get(source_feature(name_map[k]), 0.0) + v

# grouped permutation importance on holdout (permute raw column, re-transform)
print("permutation importance on holdout ...")
hold_p = prep_frame(hold)
base_auc = roc_auc_score(hold["y"], final.predict_proba(sp.csr_matrix(
    pre_final.transform(hold_p)))[:, 1])
perm = {}
for col in NUM_COLS + CAT_COLS:
    drops = []
    for rep in range(10):
        shuf = hold_p.copy()
        shuf[col] = rng.permutation(shuf[col].to_numpy())
        auc = roc_auc_score(hold["y"], final.predict_proba(
            sp.csr_matrix(pre_final.transform(shuf)))[:, 1])
        drops.append(base_auc - auc)
    perm[col] = {"mean": round(float(np.mean(drops)), 5),
                 "std": round(float(np.std(drops)), 5)}

# direction of effect for top-5 permutation features: observed pos rate and
# mean predicted prob by feature level (binary/cats) or tercile (continuous)
p_hold_final = final.predict_proba(sp.csr_matrix(pre_final.transform(hold_p)))[:, 1]
top5 = sorted(perm, key=lambda c: -perm[c]["mean"])[:5]
direction = {}
for col in top5:
    s = hold_p[col]
    if col in CAT_COLS:
        g = s
    elif s.nunique() <= 3:
        g = s.astype(str)
    else:
        g = pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"])
    tab = (pd.DataFrame({"g": g, "y": hold["y"].to_numpy(), "p": p_hold_final})
           .groupby("g", observed=True).agg(n=("y", "size"), pos_rate=("y", "mean"),
                                            mean_pred=("p", "mean")).round(4))
    tab = tab[tab.n >= 10].sort_values("mean_pred")
    direction[col] = tab.reset_index().astype({"g": str}).to_dict("records")

json.dump({"holdout_base_auc": round(float(base_auc), 4),
           "gain_by_source_feature": {k: round(v, 2) for k, v in
                                      sorted(gain_src.items(), key=lambda x: -x[1])},
           "permutation_auc_drop": dict(sorted(perm.items(),
                                               key=lambda x: -x[1]["mean"])),
           "direction_top5": direction},
          open(MODELS / "importance.json", "w"), indent=2)

json.dump(metrics, open(MODELS / "metrics.json", "w"), indent=2)
pd.DataFrame(pred_rows).to_parquet(MODELS / "predictions.parquet", index=False)
print("done. artifacts in", MODELS)
