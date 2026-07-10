"""12 — Advanced founder features: bio-industry similarity, prior experience,
repeat-YC founders.

Features (strict cohort), per company:
  CLEAN (point-in-time reconstructable):
    any_repeat_yc_founder   a founder also founded an EARLIER-batch YC company
                            (user_id match within dataset; batch-date ordered)
    n_prior_yc_companies    count of such earlier companies over all founders
  BIO-DERIVED (medium leakage — bios are current-day; two-stage scrubbed):
    bio_industry_sim_mean/max  cosine(MiniLM(founder bio), MiniLM(company
                               industry/subindustry/tags text)) — "does the
                               founder's background match the domain?"
    prior_founder_exp       bio mentions previous founding
    big_employer_exp        bio mentions FAANG/top-tier employer or consultancy
    phd_flag / prof_flag / mba_flag
    yoe_mentioned           bio cites "N years" of experience
    founders_bio_coverage   share of founders with a non-empty bio (liveness
                            indicator — itself outcome-adjacent, kept visible)

Bio scrubbing = outcome sentences (as 05/09) PLUS career-continuation
sentences (currently/now at, went on to, later founded/joined, after
selling/leaving, joined <company>) — hindsight narration.

Eval protocol identical to 10: temporal CV (5 folds), 2019-21 holdout,
20k-draw portfolio benchmark. Variants: struct / struct+fadv /
struct+fadv+minilm_full / fadv only.

Outputs: models/founder_advanced_results.json, figures/founder_advanced.png,
data/processed/founder_advanced.csv.
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

OUTCOME_PAT = (
    r"acquired?|acquisition|acquires|acqui-?hired?|merged?|merger|"
    r"went\s+public|going\s+public|ipo'?d?|public\s+offering|"
    r"shut(ting)?\s+down|shutdown|closed\s+(down|its|in)|closing\s+down|"
    r"no\s+longer\s+(operating|active|in\s+operation|in\s+business)|"
    r"ceased\s+operations?|wound\s+down|winding\s+down|out\s+of\s+business|"
    r"defunct|sold\s+to|sold\s+the\s+company|exited\s+to|now\s+part\s+of|"
    r"joined\s+forces\s+with|discontinued")
CAREER_PAT = (
    r"currently\s+(at|the|an?\s|works?|leads?|serv\w+)|now\s+(at|an?\s|the|"
    r"works?|leads?|runs?|heads?)|today\s+(he|she|they)|went\s+on\s+to|"
    r"later\s+(founded|joined|started|became)|after\s+(selling|leaving|"
    r"closing|exiting)|since\s+(leaving|selling)|most\s+recently")
SCRUB_BIO = re.compile(f"(?i)\\b({OUTCOME_PAT}|{CAREER_PAT})\\b")
SCRUB_CO = re.compile(f"(?i)\\b({OUTCOME_PAT})\\b")
SENT_SPLIT = re.compile(r"(?<=[.!?\n])\s+|\n+|(?<=^)\*\s*|\s\*\s")

PRIOR_FOUNDER = re.compile(
    r"(?i)\b(previously\s+(co-?)?founded|prior\s+(co-?)?found\w+|"
    r"(co-?)?founded\s+(two|three|four|several|multiple|\d+)|"
    r"serial\s+entrepreneur|second|2nd|third|3rd)[-\s]?(time\s+founder)?\b")
BIG_EMPLOYER = re.compile(
    r"(?i)\b(google|meta\b|facebook|amazon|aws|microsoft|apple|netflix|"
    r"stripe|uber|airbnb|linkedin|twitter|palantir|tesla|spacex|openai|"
    r"deepmind|nvidia|salesforce|oracle|goldman|mckinsey|bain|bcg|"
    r"morgan\s+stanley|jane\s+street|two\s+sigma|citadel)\b")
PHD = re.compile(r"(?i)\bph\.?d|doctorate\b")
PROF = re.compile(r"(?i)\b(professor|faculty|researcher|research\s+scientist|postdoc)\b")
MBA = re.compile(r"(?i)\bmba\b")
YOE = re.compile(r"(?i)\b\d+\+?\s*years?\b")


def scrub(t: str, pat: re.Pattern) -> str:
    if not isinstance(t, str) or not t.strip():
        return ""
    return " ".join(s for s in SENT_SPLIT.split(t)
                    if s and not pat.search(s)).strip()


# --------------------------------------------------- load + founder table
raw = {c["id"]: c for c in json.load(open(RAW / "yc_companies_all.json"))}
slug2id = {c["slug"]: c["id"] for c in raw.values()}
recs = [json.loads(l) for l in open(RAW / "founders.jsonl") if l.strip()]

df = pd.read_parquet(PROC / "features.parquet")
df = df.sort_values(["batch_date", "id"]).reset_index(drop=True)
bdate = dict(zip(df.id, df.batch_date))

# founder -> companies map for repeat detection
user_cos: dict[int, list] = {}
comp_founders: dict[int, list] = {}
for r in recs:
    if "error" in r or r["slug"] not in slug2id:
        continue
    cid = slug2id[r["slug"]]
    comp_founders[cid] = r["founders"]
    for f in r["founders"]:
        uid = f.get("user_id")
        if uid is not None and cid in bdate and pd.notna(bdate[cid]):
            user_cos.setdefault(uid, []).append((bdate[cid], cid))

# ------------------------------------------------------------- embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

ind_texts, bio_concat, rows = [], [], []
per_founder_bios, per_founder_company_ix = [], []
for i, cid in enumerate(df.id):
    c = raw[cid]
    tags = ", ".join(c.get("tags") or [])
    ind_texts.append(f"{c.get('industry') or ''} / {c.get('subindustry') or ''}."
                     f" Tags: {tags}.")
    fs = comp_founders.get(cid, [])
    bios = [scrub(f.get("founder_bio") or "", SCRUB_BIO) for f in fs]
    bios_ne = [b for b in bios if b]
    for b in bios_ne:
        per_founder_bios.append(b)
        per_founder_company_ix.append(i)

    all_bio = " | ".join(bios_ne)
    n_prior = n_repeat = 0
    for f in fs:
        uid = f.get("user_id")
        if uid is None or cid not in bdate or pd.isna(bdate[cid]):
            continue
        earlier = [x for x in user_cos.get(uid, [])
                   if x[1] != cid and pd.notna(x[0]) and x[0] < bdate[cid]]
        n_prior += len(earlier)
        n_repeat += bool(earlier)
    rows.append({
        "id": cid,
        "any_repeat_yc_founder": int(n_repeat > 0),
        "n_prior_yc_companies": n_prior,
        "prior_founder_exp": int(bool(PRIOR_FOUNDER.search(all_bio))),
        "big_employer_exp": int(bool(BIG_EMPLOYER.search(all_bio))),
        "phd_flag": int(bool(PHD.search(all_bio))),
        "prof_flag": int(bool(PROF.search(all_bio))),
        "mba_flag": int(bool(MBA.search(all_bio))),
        "yoe_mentioned": int(bool(YOE.search(all_bio))),
        "founders_bio_coverage": (len(bios_ne) / len(fs)) if fs else np.nan,
    })

E_ind = model.encode(ind_texts, batch_size=64, normalize_embeddings=True,
                     show_progress_bar=False)
E_bio = model.encode(per_founder_bios, batch_size=64,
                     normalize_embeddings=True, show_progress_bar=False)
sims_mean = np.full(len(df), np.nan)
sims_max = np.full(len(df), np.nan)
ix = np.array(per_founder_company_ix)
sims = np.einsum("ij,ij->i", E_bio, E_ind[ix])
for i in np.unique(ix):
    s = sims[ix == i]
    sims_mean[i], sims_max[i] = s.mean(), s.max()

fadv = pd.DataFrame(rows)
fadv["bio_industry_sim_mean"] = sims_mean
fadv["bio_industry_sim_max"] = sims_max
fadv.to_csv(PROC / "founder_advanced.csv", index=False)
df = df.merge(fadv, on="id", how="left")

FADV_COLS = ["any_repeat_yc_founder", "n_prior_yc_companies",
             "prior_founder_exp", "big_employer_exp", "phd_flag", "prof_flag",
             "mba_flag", "yoe_mentioned", "founders_bio_coverage",
             "bio_industry_sim_mean", "bio_industry_sim_max"]

# univariate diagnostics on strict cohort
strict = df[df.label_strict.notna()].copy()
strict["y"] = strict.label_strict.astype(int)
print("=== univariate success rates (strict cohort) ===")
for c in FADV_COLS[:8]:
    g = strict.groupby(strict[c] > 0).y.agg(["count", "mean"]).round(3)
    print(c, g.to_dict("index"), flush=True)
for c in ["bio_industry_sim_mean", "founders_bio_coverage"]:
    q = pd.qcut(strict[c], 3, duplicates="drop")
    print(c, strict.groupby(q, observed=True).y.agg(["count", "mean"])
          .round(3).to_dict("index"), flush=True)

# ------------------------------------------------------------- evaluation
E_full = np.load(PROC / "st_minilm_scrubbed.npy")
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

mask = df.label_strict.notna().to_numpy()
strict = df[mask].copy()
strict["y"] = strict.label_strict.astype(int)
E_strict = E_full[mask]
yr = strict.batch_date.dt.year.to_numpy()
dev_m, hold_m = yr <= 2018, yr >= 2019
dev, hold = strict[dev_m], strict[hold_m]
E_dev, E_hold = E_strict[dev_m], E_strict[hold_m]
y_dev, y_hold = dev.y.to_numpy(), hold.y.to_numpy()
base_rate = float(y_hold.mean())


def matrices(train, test, variant):
    mats_tr, mats_te = [], []
    if "struct" in variant:
        extra = FADV_COLS if "fadv" in variant else []
        pre = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                   min_frequency=10, sparse_output=True),
              CAT_COLS),
             ("num", "passthrough", NUM_COLS + extra)], sparse_threshold=1.0)
        tr, te = train.copy(), test.copy()
        for c in CAT_COLS:
            tr[c] = tr[c].fillna("Missing").astype(str)
            te[c] = te[c].fillna("Missing").astype(str)
        mats_tr.append(sp.csr_matrix(pre.fit_transform(tr)))
        mats_te.append(sp.csr_matrix(pre.transform(te)))
    elif "fadv" in variant:
        mats_tr.append(sp.csr_matrix(train[FADV_COLS].to_numpy(dtype=float)))
        mats_te.append(sp.csr_matrix(test[FADV_COLS].to_numpy(dtype=float)))
    return mats_tr, mats_te


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
VARIANTS = ["struct", "struct_fadv", "struct_fadv_minilm", "fadv"]
results = {"holdout_n": int(len(hold)), "holdout_base_rate": base_rate,
           "variants": {}}
dev_idx = np.arange(len(dev))
dyr = dev.batch_date.dt.year.to_numpy()

for variant in VARIANTS:
    pooled_s, pooled_y = [], []
    for tr_end, te_years in FOLD_DEFS:
        tr_i = dev_idx[dyr <= tr_end]
        te_i = dev_idx[np.isin(dyr, te_years)]
        mats_tr, mats_te = matrices(dev.iloc[tr_i], dev.iloc[te_i], variant)
        if "minilm" in variant:
            mats_tr.append(sp.csr_matrix(E_dev[tr_i]))
            mats_te.append(sp.csr_matrix(E_dev[te_i]))
        Xtr, Xte = sp.hstack(mats_tr).tocsr(), sp.hstack(mats_te).tocsr()
        cut = int(Xtr.shape[0] * 0.85)
        m = fit_xgb(Xtr[:cut], y_dev[tr_i][:cut], Xtr[cut:], y_dev[tr_i][cut:])
        pooled_s.append(m.predict_proba(Xte)[:, 1])
        pooled_y.append(y_dev[te_i])
    pooled_s, pooled_y = np.concatenate(pooled_s), np.concatenate(pooled_y)

    mats_d, mats_h = matrices(dev, hold, variant)
    if "minilm" in variant:
        mats_d.append(sp.csr_matrix(E_dev))
        mats_h.append(sp.csr_matrix(E_hold))
    Xd, Xh = sp.hstack(mats_d).tocsr(), sp.hstack(mats_h).tocsr()
    cut = int(Xd.shape[0] * 0.85)
    m = fit_xgb(Xd[:cut], y_dev[:cut], Xd[cut:], y_dev[cut:])
    s_hold = m.predict_proba(Xh)[:, 1]

    entry = {"temporal_cv_auc_pooled":
                 round(float(roc_auc_score(pooled_y, pooled_s)), 4),
             "holdout_auc": round(float(roc_auc_score(y_hold, s_hold)), 4),
             "holdout_pr_auc":
                 round(float(average_precision_score(y_hold, s_hold)), 4),
             "portfolios": {}}
    for k in (25, 53, 100):
        top = np.argsort(-s_hold)[:k]
        entry["portfolios"][k] = {"model_success_rate":
                                  round(float(y_hold[top].mean()), 4)}
    results["variants"][variant] = entry
    print(variant, entry["temporal_cv_auc_pooled"], entry["holdout_auc"],
          entry["portfolios"], flush=True)

for k in (25, 53, 100):
    draws = np.array([y_hold[rng.choice(len(hold), k, replace=False)].mean()
                      for _ in range(20000)])
    for v in results["variants"].values():
        e = v["portfolios"][k]
        e["lift_vs_random"] = round(e["model_success_rate"] / draws.mean(), 3)
        e["p_value"] = round(float((draws >= e["model_success_rate"]).mean()), 4)

with open(MODELS / "founder_advanced_results.json", "w") as f:
    json.dump(results, f, indent=2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = VARIANTS
x = np.arange(len(names))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
ax1.bar(x - 0.2, [results["variants"][n]["temporal_cv_auc_pooled"] for n in names],
        0.4, label="temporal CV", color="#2c7fb8")
ax1.bar(x + 0.2, [results["variants"][n]["holdout_auc"] for n in names],
        0.4, label="holdout", color="#f4a261")
ax1.axhline(0.5, color="k", ls="--", lw=1)
ax1.set_xticks(x, names, rotation=20, ha="right")
ax1.set_ylim(0.45, 0.72); ax1.set_ylabel("ROC-AUC")
ax1.set_title("Advanced founder features — AUC"); ax1.legend()
r53 = [results["variants"][n]["portfolios"][53]["model_success_rate"] for n in names]
ax2.bar(x, r53, color="#1b9e77")
ax2.axhline(base_rate, color="k", ls="--", label=f"random ({base_rate:.1%})")
ax2.set_xticks(x, names, rotation=20, ha="right")
ax2.set_ylabel("top-decile success rate")
ax2.set_title("Portfolio top decile (k=53)"); ax2.legend()
fig.tight_layout()
fig.savefig(FIGS / "founder_advanced.png", dpi=130)
print("saved models/founder_advanced_results.json, figures/founder_advanced.png")
