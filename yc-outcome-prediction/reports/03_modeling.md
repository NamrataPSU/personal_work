# 03 — Modeling & validation

Code: `src/05_model.py` (training, metrics, importance), `src/06_evaluate.py`
(figures, tables). Machine-readable results: `models/metrics.json`,
`models/importance.json`; per-row scores in `models/predictions.parquet`.
Seed 42 throughout.

## 1. Setup

**Primary target** — `label_strict`: Acquired/Public = 1 vs Inactive = 0 on the
mature cohort (batch ≤ 2021-07-02, still-Active companies excluded).
N = 1,575, 700 positive (44.4%). **Secondary sensitivity** — `label_survival`
(Acquired/Public/Active = 1 vs Inactive = 0), N = 3,202, 72.7% positive.

**Validation design.** Outcomes take years to resolve, so the honest question
is "trained on old batches, can we score new ones?" — random CV answers a
different, easier question (interpolation within a mixed-vintage pool).

* **Final holdout:** all strict-cohort batches from 2019-01-01 on
  (2019–2021 vintages): n = 523, base rate 40.2%. Untouched until final
  evaluation.
* **Primary CV:** forward-chaining temporal CV on the dev pool (batches
  ≤ 2018, n = 1,052, base rate 46.6%), 5 expanding-window folds:

  | fold | train batches | train n | test batches | test n | test base rate |
  |---|---|---|---|---|---|
  | F1 | ≤ 2011 | 264 | 2012–13 | 184 | 0.451 |
  | F2 | ≤ 2013 | 448 | 2014–15 | 231 | 0.533 |
  | F3 | ≤ 2015 | 679 | 2016 | 123 | 0.390 |
  | F4 | ≤ 2016 | 802 | 2017 | 131 | 0.481 |
  | F5 | ≤ 2017 | 933 | 2018 | 119 | 0.437 |

* **Secondary CV:** stratified random 5-fold on the same dev pool. The
  random-minus-temporal gap quantifies vintage leakage.

**Models.** (1) prior baseline (constant = train pos-rate); (2) elastic-net
logistic on industry + primary_region + batch_season + cohort_size
(one-hot, missing-as-category); (3) XGBoost variants: `hist`, subsample 0.8,
colsample 0.8, min_child_weight 5, up to 800 trees with early stopping
(50 rounds) on an inner 15% validation split (temporal tail of the training
window for temporal folds; stratified random otherwise), light grid over
max_depth ∈ {3, 4} at eta 0.05 selected by inner-val AUC.

**Text handling.** `text_clean` is current-day directory text; 108 companies
literally say "acquired by". Before any text modeling, whole sentences
matching outcome-revealing patterns (acquired/acquisition, merged, went
public/IPO, shut down, closed, no longer operating, ceased operations,
defunct, sold to, ...) were dropped — 225 of 5,999 rows had at least one
sentence removed. Sentence-level (not token-level) removal also strips
acquirer names ("acquired by Stripe" leaks via "stripe"). The TF-IDF
vectorizer (max_features 2000, min_df 5, English stopwords, sublinear TF) is
**refit inside every training fold** on scrubbed text; the shipped
`text_tfidf.npz` was not used for modeling. An unscrubbed text-only variant
is run purely to quantify the leakage.

## 2. Results — label_strict

### Temporal CV (pooled over 5 forward-chaining folds, n = 788 test rows)

| variant | ROC-AUC | PR-AUC | Brier | P@top-10% | lift |
|---|---|---|---|---|---|
| prior baseline | 0.450 | 0.439 | 0.250 | 0.354 | 0.76 |
| logit (controls) | 0.560 | 0.489 | 0.248 | 0.430 | 0.92 |
| **XGB structured (main)** | **0.637** | **0.580** | **0.236** | 0.620 | 1.33 |
| XGB + batch_year | 0.635 | 0.583 | 0.240 | 0.620 | 1.33 |
| XGB de-vintaged (no season/cohort_size) | 0.625 | 0.584 | 0.239 | 0.633 | 1.35 |
| XGB struct + text (scrubbed) | 0.594 | 0.548 | 0.243 | 0.557 | 1.19 |
| XGB text only (scrubbed) | 0.567 | 0.537 | 0.252 | 0.646 | 1.38 |
| XGB text only (raw, unscrubbed) | 0.626 | 0.597 | 0.238 | 0.709 | 1.51 |

(The prior baseline pools below 0.5 because its constant prediction tracks the
training-window base rate, which drifts across vintages — itself a symptom of
the vintage confound. Within any single fold a constant has AUC exactly 0.5.)

Per-fold ROC-AUC, main model: 0.570, 0.643, 0.692, 0.650, 0.720 (F1→F5) —
skill grows with training-window size and never dips to chance.

### Random vs temporal CV — the vintage-leakage gap

| variant | random AUC | temporal AUC | gap |
|---|---|---|---|
| logit (controls) | 0.604 | 0.560 | +0.044 |
| XGB structured (main) | 0.683 | 0.637 | +0.046 |
| XGB + batch_year | 0.700 | 0.635 | +0.064 |
| XGB de-vintaged | 0.691 | 0.625 | +0.066 |
| XGB struct + text (scrubbed) | 0.677 | 0.594 | +0.084 |
| XGB text only (scrubbed) | 0.634 | 0.567 | +0.068 |
| XGB text only (raw) | 0.683 | 0.626 | +0.057 |

Random CV inflates every model by **+0.04–0.08 AUC**; the inflation is worst
exactly where leakage risk is highest (text). Under random CV, giving the
model `batch_year` "helps" (0.683 → 0.700) — under temporal CV it does
nothing (0.637 → 0.635), confirming the year signal is interpolation, not
transferable skill. Top-decile lift inflates too: 1.62× (random) vs 1.33×
(temporal) for the main model.

### Final holdout (2019–21 batches, n = 523, base rate 0.402)

Models trained once on the full dev pool (≤ 2018).

| variant | ROC-AUC | PR-AUC | Brier | P@top-10% (n=53) | lift |
|---|---|---|---|---|---|
| prior baseline | 0.500 | 0.402 | 0.244 | 0.396 | 0.99 |
| logit (controls) | 0.590 | 0.454 | 0.236 | 0.396 | 0.99 |
| **XGB structured (main)** | 0.566 | 0.459 | **0.288** | 0.509 | 1.27 |
| XGB + batch_year | 0.579 | 0.478 | 0.308 | 0.547 | 1.36 |
| XGB de-vintaged | 0.573 | 0.474 | 0.311 | 0.528 | 1.32 |
| XGB struct + text (scrubbed) | 0.571 | 0.495 | 0.287 | 0.623 | 1.55 |
| XGB text only (scrubbed) | 0.540 | 0.457 | 0.241 | 0.642 | 1.60 |
| XGB text only (raw) | 0.553 | 0.475 | 0.240 | 0.585 | 1.46 |

Holdout ranking beats chance but is modest: AUC ~0.54–0.59 across models,
well below both random-CV (~0.68) and temporal-CV (~0.64) estimates. The
2019–21 block is the hardest regime: shortest exit runway (5–7 years) and a
base rate (40.2%) below the dev pool's (46.6%).

### Text ablation

Scrubbing costs the text-only model 0.059 temporal AUC (0.626 → 0.567) —
that is the measured size of the "acquired by ..." leak; unscrubbed text
metrics should not be trusted. After scrubbing, text alone still ranks above
chance (0.567 temporal / 0.540 holdout) and, notably, is the **best
top-decile selector on the holdout** (P@10% = 0.642, 1.60×), while adding
text to structure *hurts* pooled temporal AUC (0.637 → 0.594) — text signal
is concentrated in the extreme tail (rich, specific descriptions) and noisy
in the middle of the ranking. Structure + scrubbed text is the best holdout
PR-AUC (0.495) and a strong decile selector (0.623, 1.55×).

### De-vintage check

Removing all batch-derived features (batch_season, cohort_size) barely moves
temporal AUC (0.637 → 0.625) and adding batch_year outright doesn't help
under temporal evaluation (0.635). Conclusion: the main model's temporal-CV
skill is **not** riding on vintage features — but the random-CV numbers of
*any* variant are contaminated by implicit vintage proxies (e.g. tag and
industry mixes drift by era), which is why the gap persists even for the
de-vintaged model (+0.066).

### Calibration

`figures/calibration_holdout.png`. On the holdout the XGB scores **run hot**:
mean predicted probability ≈ 0.57 vs realized 0.40, Brier 0.288 — *worse than
the constant baseline* (0.244) despite better ranking. This is the vintage
confound in probability space: the model calibrates to the ≤ 2018 exit
environment, and 2019–21 batches have had less time to exit. Scores should be
treated as **ranks, not probabilities**; deployment would require recalibration
against a recent-vintage estimate of the base rate (e.g. Platt/isotonic on a
rolling window, or an explicit age adjustment).

### Sensitivity: label_survival (N = 3,202, 72.7% positive)

XGB structured: temporal-CV pooled AUC 0.673 (PR-AUC 0.814 vs base 0.691),
random-CV 0.765 (gap +0.092); holdout (n = 1,535, base 0.796) AUC 0.615,
PR-AUC 0.856, Brier 0.165, P@top-10% 0.916 (1.15×). Same qualitative story:
moderate transferable signal, large random-CV inflation.

## 3. Interpretation

`figures/importance_top20.png`; details in `models/importance.json`.
Permutation importance = mean holdout AUC drop over 10 permutations of the
raw column (re-transformed each time); gain aggregated to source features.

Top features (permutation | direction from holdout grouped means):

1. **tag_saas** (+0.020) — SaaS-tagged companies score higher and genuinely
   succeed more (holdout pos-rate 0.555 vs 0.337).
2. **long_description_len** (+0.014) — longer description → higher score;
   observed pos-rate rises low→high tercile (0.377 → 0.414). Partly real
   (maintained listings), partly a liveness artifact — flagged medium-leakage.
3. **industry** (+0.012) — B2B (0.490 pos-rate) and Fintech (0.426) score and
   perform above Consumer (0.232) and Healthcare (0.333).
4. **one_liner_len** (+0.009) — mid-length one-liners score highest; weak
   monotone relation with outcomes.
5. **is_remote** (+0.006, high variance) — remote-flagged companies score
   higher (holdout pos-rate 0.418 vs 0.352); unstable, and remote listing is
   era-correlated, so treat with caution.

Gain ranking adds `subindustry` (top by gain — many low-count splits),
`tag_b2b`, `primary_region`, `has_location`. Consistent picture: **what the
company sells (B2B/SaaS/fintech vs consumer) plus listing completeness**
carries most of the transferable signal; geography adds little on holdout.

## 4. What an investor gets

If a scorer could only pick the top 10% of the 2019–21 holdout (53 of 523
companies): structure-only 50.9% success vs 40.2% base (1.27×); structure +
scrubbed text 62.3% (1.55×); scrubbed text alone 64.2% (1.60×). Real but
modest edge — consistent with the prior that public directory metadata
contains little of what actually determines startup outcomes (team, product
execution, market timing).

## 5. Honest limitations

1. **Residual censoring in the holdout.** 2019–21 strict labels compare
   already-exited vs already-dead at 5–7 years; slow winners are excluded
   entirely (Active companies are NaN under label_strict), so holdout metrics
   measure "early exit vs early death", not final outcomes.
2. **Probabilities are not trustworthy.** Holdout Brier (0.288) is worse than
   the prior baseline; use ranks only, recalibrate before any decision use.
3. **Feature timestamps are current-day.** Text, tags, even industry can be
   edited post-hoc; scrubbing removes explicit outcome phrases but cannot
   remove subtler survivorship in how live companies curate their pages.
   The unscrubbed-vs-scrubbed delta (0.059 AUC) is a lower bound on such
   contamination.
4. **Small folds.** Test blocks of 119–231 companies give AUC standard errors
   of roughly ±0.03–0.05; per-fold differences within that band are noise.
5. **Single-source data.** No funding, team, or traction covariates; results
   characterize what *YC directory metadata* predicts, not what is predictable.

## Figures

| file | content |
|---|---|
| `figures/temporal_folds_auc.png` | per-fold temporal-CV ROC-AUC, 4 key variants |
| `figures/random_vs_temporal_gap.png` | pooled AUC, random vs temporal, all variants |
| `figures/roc_pr_holdout.png` | holdout ROC and PR curves (base-rate reference) |
| `figures/calibration_holdout.png` | holdout reliability curves |
| `figures/importance_top20.png` | permutation + gain importance, top 20 |
| `figures/top_decile_lift_holdout.png` | holdout precision@top-decile vs base rate |

## Artifacts

`models/xgb_strict_final.json` (final structured model, trained on ≤ 2018),
`models/preprocessor_final.joblib` (fitted ColumnTransformer + feature names),
`models/metrics.json`, `models/importance.json`, `models/predictions.parquet`.
