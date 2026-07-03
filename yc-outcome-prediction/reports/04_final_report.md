# Predicting Startup Outcomes from YC Portfolio Data — Final Report

**Pipeline:** `src/01_acquire.py` → `02_profile.py` → `03_labels.py` → `04_features.py` → `05_model.py` → `06_evaluate.py`.
**Data:** yc-oss/api mirror of the public YC directory, 5,999 companies, batches 2005–2027, downloaded 2026-07-02.
**Stage reports:** `01_data_profile.md`, `02_labels_features.md`, `03_modeling.md` (this document synthesizes them).

## Executive summary

We ran the full cycle — acquisition, profiling, labeling, feature engineering,
modeling, temporal validation — on the largest freely available YC portfolio
snapshot. The brief's central hypothesis is confirmed empirically, twice over:

1. **Feasibility, not volume or technique, is the binding constraint.** Of the
   six candidate feature categories in the brief, only *market/sector
   classification* and *geographic/timing factors* are cleanly reconstructable
   at historical scale from this class of source. Founder background, funding
   history, early traction, and team-growth trajectory are absent or leaky, and
   each would require external sources whose historical coverage is itself
   correlated with the outcome (see §1).
2. **The reconstructable features carry real but modest signal.** XGBoost on
   leakage-audited structured features reaches **0.637 pooled ROC-AUC under
   forward-chaining temporal CV** (vs 0.5 chance, 0.560 for a logistic
   baseline), and a **top-decile hit rate of 50.9–62.3% on a 2019–21 holdout
   whose base rate is 40.2%** (lift 1.27–1.55×). That is a usable screening
   edge, not an oracle — consistent with the prior that public directory
   metadata omits most of what determines outcomes (team, execution, timing).
3. **Naive validation overstates performance by +0.04–0.08 AUC.** Random CV
   vs temporal CV on identical models quantifies the vintage confound the
   brief worried about. Every historical-scale claim in this space should be
   discounted accordingly unless the validation was explicitly temporal.
4. **Retroactive text is measurably contaminated.** 108 company descriptions
   literally contain "acquired by"; scrubbing outcome-revealing sentences costs
   a text-only model 0.059 AUC — a measured *lower bound* on hindsight
   contamination in any retroactively collected feature. This is the concrete,
   quantified version of the brief's skepticism about retroactive founder
   psychometrics.

---

## 1. Feature feasibility audit

Verdict per category in the brief, based on what we found in the data (fields,
missingness, leakage) and what external reconstruction would require:

| Category | Feasible at historical scale? | Evidence & reasoning |
|---|---|---|
| **Market/sector classification** | **Yes — best available.** | `industry` (9 values), `subindustry` (59), `tags` (332) are 100% populated for all 5,999 companies. Caveat: taxonomy is *current-day* (an "AI" tag on a 2012 company reflects re-curation), so tag features double as era proxies — handled by temporal validation. Empirically the strongest transferable signal: `tag_saas` tops permutation importance (holdout success 55.5% vs 33.7%); B2B (49.0%) and Fintech (42.6%) beat Consumer (23.2%) and Healthcare (33.3%). |
| **Geographic factors** | **Yes.** | `all_locations`/`regions` populated for 96% of companies; location at listing is mostly stable (low retro-edit risk). But it adds little predictive value on holdout — Bay Area membership (51% of the dataset) is near-uninformative *within* YC. |
| **Timing factors (batch/vintage)** | **Yes to reconstruct — but use as controls, not features.** | Batch is perfectly recorded and defines the temporal splits. Adding `batch_year` as a feature raises random-CV AUC (0.683→0.700) but does nothing under temporal CV (0.637→0.635): vintage "signal" is interpolation, not transferable skill. |
| **Founder background / education / prior experience** | **No, from this source; high-risk externally.** | Zero founder fields exist in the dataset (`app_answers` null for 5,920/5,999). Reconstruction via LinkedIn/Crunchbase suffers from the exact failure mode the brief flagged for psychometrics: profiles are *today's* bios, edited with knowledge of the outcome, and dead-company founders disproportionately vanish — differential missingness correlated with the label. Feasible only with point-in-time snapshots (Wayback captures of YC company pages, historical Crunchbase dumps), at high effort and partial coverage. |
| **Funding history** | **Partially — medium risk, external source required.** | Not in this dataset (`stage` is current-day and quarantined as leaky). Crunchbase/PitchBook round dates do allow point-in-time cuts ("funding raised within 18 months of batch"), which is the right construction. Risk: coverage is systematically worse for dead and small companies, so missingness itself leaks the label; must model missingness explicitly and validate coverage by vintage. |
| **Early product/traction signals** | **Mostly no.** | Nothing usable here: `launched_at` floors at 2010-01-17 (it records directory-listing time, not launch) and is quarantined. External options (Wayback, HN Launch posts, Product Hunt) are patchy and biased toward consumer/dev-tools companies. |
| **Team growth trajectory** | **No.** | `team_size` is *current* headcount (median 6, max 8,600; 125 zeros = dead companies) — it encodes the outcome and is excluded as a feature. Historical headcount reconstruction (LinkedIn) is sparse before ~2015. |
| **Founder personality / psychometrics** (Rebel Fund-style) | **No — and now with a measured reason.** | No historical instrument exists; any retroactive scoring is done with outcome knowledge. Our text-scrubbing experiment puts a number on this failure mode: even *mechanical* outcome contamination in text inflates AUC by ≥0.059. Human retroactive scoring of founders would be worse and unmeasurable. |

**Practical rule that emerged:** for every candidate feature, ask *"what date
does this field's value reflect?"* If the answer is "today" rather than "at
batch time," it is either leaky (team_size, stage, isHiring, top_company —
all quarantined as `leaky_*` diagnostics) or contaminated-but-salvageable
(descriptions, tags — usable only with scrubbing + ablation + temporal
validation). The feature manifest (`data/processed/feature_manifest.csv`)
records this judgment per column.

## 2. Outcome label definition — recommendation

**Recommended: a two-label sensitivity pair on a maturity-gated cohort**, as
implemented:

- **Maturity gate:** only batches ≥ 5 years old (≤ Summer 2021 at the
  2026-07-02 reference date; mature N = 3,202 of 5,999). Younger cohorts are
  right-censored to the point of meaninglessness (2025–26 batches are 98–100%
  "Active").
- **`label_strict`** — success = Acquired ∪ Public, failure = Inactive,
  still-Active excluded (N = 1,575, 44.4% positive). Answers "exit vs death."
- **`label_survival`** — success = Acquired ∪ Public ∪ Active, failure =
  Inactive (N = 3,202, 72.7% positive). Answers "survived vs died."

Neither alone is honest: acquisition conflates acquihires with great exits,
and "Active at 5 years" conflates Stripe with zombies — the public status
field cannot distinguish either. Running both and requiring qualitative
agreement (which held: same feature ranking, same temporal-vs-random gap) is
the robustness check. With a valuation source (Crunchbase), the better label
would be "exit or last-round valuation above a threshold"; that is the single
highest-value data upgrade available.

**Horizon:** 5 years minimum for the gate; note the residual censoring even
inside the mature cohort — strict positive rate declines 0.67→0.39 across
2005→2021 vintages because recent "mature" companies have had less time to
exit. A 7–10 year gate is cleaner but halves N and skews to the pre-2015 YC
era. The right long-term formulation is **survival analysis** (time-to-exit /
time-to-death with censoring) rather than binary classification; `batch_date`,
`status`, and `company_age_years` in the processed table support it.

## 3. Survivorship bias and historical-data pitfalls — what we actually hit

1. **Right-censoring masquerading as class balance.** "Active" means
   *unresolved*, not *alive and well*, for young cohorts. Fix: maturity gate +
   explicit exclusion (strict) or explicit inclusion (survival), never mixing.
2. **The vintage confound — measured.** Label base rates drift monotonically
   with batch year in opposite directions for the two labels; any random-split
   model partially learns the calendar. Random CV inflated *every* model
   variant by +0.04–0.08 AUC, worst for text (+0.084). This is the single
   largest pitfall and it is invisible unless you run both validation schemes.
3. **Present-day snapshot bias.** All fields reflect 2026 values. Direct form:
   108 descriptions saying "acquired by" (83 of them Acquired). Subtle form:
   dead companies have shorter/emptier listings (`long_description_len` ranks
   #2 in importance partly for this reason — flagged medium-leakage in the
   manifest). Sentence-level scrubbing of outcome phrases (225 companies
   affected) removes the direct form; the subtle form is irreducible in this
   source and bounds how much text gains should be trusted.
4. **Calibration does not transfer across vintages.** The final model's
   holdout Brier (0.288) is *worse than a constant baseline* (0.244) despite
   better ranking: it calibrates to the ≤2018 exit environment and runs hot
   (mean predicted 0.57 vs realized 0.40) on 2019–21. Treat scores as ranks;
   recalibrate on a rolling recent window before any decision use.
5. **One bias this source largely avoids:** the YC directory *retains* dead
   companies (1,045 Inactive), unlike sources that silently drop failures.
   That is precisely why this dataset is usable at all — external enrichment
   (Crunchbase, LinkedIn) would reintroduce differential coverage of failures,
   and must be audited for it per vintage.

## 4. Modeling approach for heterogeneous-reliability features

What we recommend (and did):

- **Gradient-boosted trees (XGBoost) on structured features** as the main
  model: handles mixed types, missingness-as-information, and nonlinearities
  at N≈1,500–3,200 where deep models have no advantage. Light tuning only
  (depth 3–4, eta 0.05, early stopping on a temporal inner-validation tail);
  elastic-net logistic on sector+region+batch as the honesty baseline.
- **Reliability tiering, enforced in code, not prose.** Every column carries a
  role and leakage rating in the manifest; `leaky_*` columns are physically
  renamed so they cannot be used silently. Medium-reliability features (text,
  listing-completeness) enter only through **ablation pairs** — every result
  is reported with and without them, so their contribution is always visible
  and never load-bearing.
- **Text via fold-refit TF-IDF on scrubbed text only.** Never reuse a
  vectorizer fit on pooled data (that alone leaks vintage vocabulary). Result:
  scrubbed text alone is weak in overall ranking (temporal AUC 0.567) but the
  best top-decile selector on holdout (64.2% precision, 1.60×) — signal lives
  in the extreme tail of rich, specific descriptions.
- **De-vintaging as a standard ablation:** run every model ± batch-derived
  features; if temporal-CV performance moves, the model is riding the
  calendar. (Ours didn't: 0.637 vs 0.625.)
- N is *not* large enough for per-sector models or heavy hyperparameter
  search — both overfit the fold structure. Resist them.

## 5. Validation strategy for slow-resolving outcomes

The design we recommend as the template for any venture-outcome model:

1. **Forward-chaining temporal CV** on batch date (5 expanding-window folds,
   train ≤ t → test next block), pooled *and* per-fold. Per-fold main-model
   AUC: 0.570, 0.643, 0.692, 0.650, 0.720 — skill grows with training window
   and never dips to chance, which is the pattern honest signal shows.
2. **A final untouched vintage holdout** (2019–21, n = 523) evaluated once.
   Expect degradation (0.637 temporal-CV → 0.566 holdout): the newest vintages
   are the hardest regime (shortest runway, shifted base rate) *and* the only
   regime deployment cares about. Report it, don't average it away.
3. **Random CV run in parallel purely as a leakage meter.** The gap (here
   +0.04–0.08 AUC) is a finding, not a nuisance — publish it.
4. **Decision-relevant metrics alongside AUC:** precision@top-decile with
   explicit base-rate reference and lift (an investor picks a shortlist, not a
   threshold); PR-AUC; Brier + reliability curves because probability quality
   fails across vintages even when ranking holds.
5. **Uncertainty honesty:** test blocks of 119–231 give AUC standard errors of
   ±0.03–0.05; differences within that band are noise. With outcomes taking
   5+ years to resolve, the *only* way to enlarge test sets is to wait or
   widen the maturity gate — a structural limit of the domain that no
   modeling choice removes.

## 6. Headline results (label_strict; full tables in `03_modeling.md`)

| Evaluation | Model | ROC-AUC | P@top-10% (lift) |
|---|---|---|---|
| Temporal CV (pooled) | XGB structured | **0.637** | 0.620 (1.33×) |
| Temporal CV (pooled) | logit baseline | 0.560 | 0.430 (0.92×) |
| Random CV (same model) | XGB structured | 0.683 (**+0.046 inflation**) | 1.62× |
| Holdout 2019–21 (base 40.2%) | XGB structured | 0.566 | 0.509 (1.27×) |
| Holdout 2019–21 | XGB + scrubbed text | 0.571 | **0.623 (1.55×)** |

Top transferable features: `tag_saas`, description length (partly liveness
artifact — flagged), `industry` (B2B/Fintech > Consumer/Healthcare),
one-liner length, remote flag. Geography ≈ nothing within YC.

## 7. Recommended next steps, in value order

1. **Valuation-aware labels** via a Crunchbase/PitchBook join with as-of-date
   round data (fixes the acquihire-vs-exit conflation — the largest label
   weakness).
2. **Point-in-time founder features** from Wayback captures of YC company
   pages (founder count, technical mix at batch time) — the only
   founder-feature construction that avoids retroactive-bio bias.
3. **Survival-analysis reformulation** (discrete-time hazard or Cox with
   batch-year strata) to use censored Active companies instead of discarding
   them.
4. **Rolling recalibration** (isotonic on the most recent resolved vintages)
   before any use of scores as probabilities.
5. Refresh the dataset annually: each new year resolves ~200–400 more
   companies' outcomes and extends the temporal CV by one fold.

## Repository map

```
yc-outcome-prediction/
├── data/raw/          yc_companies_all.json (5,999 cos), meta, PROVENANCE.md, field_summary.csv
├── data/processed/    features.parquet (5,999×60), labels.csv, feature_manifest.csv,
│                      text_tfidf.npz + vocab, README.md
├── src/               01_acquire → 06_evaluate (each stage reproducible, seeded)
├── models/            xgb_strict_final.json, preprocessor_final.joblib,
│                      metrics.json, importance.json, predictions.parquet
├── figures/           temporal folds, random-vs-temporal gap, ROC/PR, calibration,
│                      importance, top-decile lift (6 PNGs)
└── reports/           01_data_profile.md, 02_labels_features.md, 03_modeling.md,
                       04_final_report.md (this file)
```
