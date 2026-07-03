# Labels & Features — YC Outcome Prediction

Scripts: `src/03_labels.py`, `src/04_features.py`. Outputs in
`data/processed/` (see its README.md). Reference date: **2026-07-02**.

## 1. Label scheme

`status` is a current-state snapshot with four values (Active 4,136 /
Inactive 1,045 / Acquired 795 / Public 23). Recent batches are heavily
right-censored (2025–26 are ~98–100% Active), so "Active" is uninterpretable
for young companies but meaningful for old ones.

**Scheme implemented**

1. Parse `batch` → (season, year) → `batch_date` (Winter=Jan 1, Spring=Apr 1,
   Summer=Jun 1, Fall=Sep 1 of the batch year).
2. **Mature cohort**: `batch_date <= 2021-07-02` (>= ~5 years to resolve).
   Includes Winter 2021 and Summer 2021; excludes everything later, plus the
   1 "Unspecified" row. Mature N = 3,202.
3. Two label variants on the mature cohort:
   - **label_strict** — positive = Acquired|Public; negative = Inactive;
     mature Active (1,627 companies) **excluded**. N = 1,575.
   - **label_survival** — positive = Acquired|Public|Active; negative =
     Inactive. N = 3,202.

**Why two variants (sensitivity pair).** Acquisition is not always success:
acquihires and asset sales are recorded identically to great exits — a real
limitation of this source. Symmetrically, Active at 5+ years can be a thriving
private company (Stripe-like) or a zombie — also indistinguishable here.
label_strict answers "exit vs death"; label_survival answers "survived vs
died". Agreement between models on both is the robustness check.

**Alternatives considered.** (a) Longer maturity window (7–10y): cleaner
resolution but halves the cohort and skews it to the pre-2015 YC era.
(b) Team-size-gated Active labeling: rejected, team_size is a leaky
current-state field. (c) Survival analysis with censoring: viable and left as
an option for the modeling agent — `batch_date` + `status` +
`company_age_years` are in the table.

### Per-year label balance

**label_strict** (N=1,575; 700 pos / 875 neg; 44.4% positive)

| year | n | pos | neg | pos_rate |
|---|---|---|---|---|
| 2005 | 9 | 6 | 3 | 0.667 |
| 2006 | 17 | 6 | 11 | 0.353 |
| 2007 | 29 | 13 | 16 | 0.448 |
| 2008 | 40 | 9 | 31 | 0.225 |
| 2009 | 34 | 13 | 21 | 0.382 |
| 2010 | 56 | 32 | 24 | 0.571 |
| 2011 | 79 | 42 | 37 | 0.532 |
| 2012 | 114 | 49 | 65 | 0.430 |
| 2013 | 70 | 34 | 36 | 0.486 |
| 2014 | 102 | 57 | 45 | 0.559 |
| 2015 | 129 | 66 | 63 | 0.512 |
| 2016 | 123 | 48 | 75 | 0.390 |
| 2017 | 131 | 63 | 68 | 0.481 |
| 2018 | 119 | 52 | 67 | 0.437 |
| 2019 | 143 | 57 | 86 | 0.399 |
| 2020 | 176 | 74 | 102 | 0.420 |
| 2021 | 204 | 79 | 125 | 0.387 |

**label_survival** (N=3,202; 2,327 pos / 875 neg; 72.7% positive)

| year | n | pos | neg | pos_rate |
|---|---|---|---|---|
| 2005 | 9 | 6 | 3 | 0.667 |
| 2006 | 18 | 7 | 11 | 0.389 |
| 2007 | 32 | 16 | 16 | 0.500 |
| 2008 | 43 | 12 | 31 | 0.279 |
| 2009 | 42 | 21 | 21 | 0.500 |
| 2010 | 63 | 39 | 24 | 0.619 |
| 2011 | 105 | 68 | 37 | 0.648 |
| 2012 | 149 | 84 | 65 | 0.564 |
| 2013 | 98 | 62 | 36 | 0.633 |
| 2014 | 152 | 107 | 45 | 0.704 |
| 2015 | 214 | 151 | 63 | 0.706 |
| 2016 | 224 | 149 | 75 | 0.665 |
| 2017 | 241 | 173 | 68 | 0.718 |
| 2018 | 277 | 210 | 67 | 0.758 |
| 2019 | 371 | 285 | 86 | 0.768 |
| 2020 | 437 | 335 | 102 | 0.767 |
| 2021 | 727 | 602 | 125 | 0.828 |

Note the vintage trends: strict pos_rate drifts *down* in later years (less
time to exit — residual censoring of exits even in "mature" cohorts), while
survival pos_rate drifts *up* (less time to die). **The modeling agent must
control for batch year** (temporal splits and/or year as covariate); any model
predicting these labels partly learns vintage, not company quality.

## 2. Features (full manifest: `data/processed/feature_manifest.csv`)

| feature | type | leakage | note |
|---|---|---|---|
| batch_season | categorical | none | fixed at admission |
| cohort_size | numeric | none | batch size, 1–398; timing/competition signal |
| industry, subindustry | categorical | low | usually stable; pivots possible |
| n_industries, n_tags | numeric | low | |
| tag_* (25 indicators) | numeric | low | AI/Artificial Intelligence/Generative AI merged into `tag_ai`; tags can be re-curated post-hoc (AI tags era-biased) |
| primary_region | categorical | none | first non-Remote region |
| is_us, is_bay_area, is_remote, is_international, has_location | numeric | low | location at listing |
| one_liner_len, long_description_len, has_description | numeric | **medium** | current-day text; empty text correlates with dead listings |
| text_clean | text | **medium** | 108 rows literally contain "acquired by" (83 Acquired) — direct outcome leakage in text; ablate/scrub |
| has_former_names | numeric | low | ~half of companies; mostly legal-name variants, weak pivot signal |
| nonprofit | numeric | none | |
| app_video_public | numeric | low | application video = pre-batch artifact; publish choice may be later |
| demo_day_video_public | numeric | low | demo-day video = end-of-batch (early-stage) artifact — plausibly pre-outcome; publish choice may be later |

TF-IDF (2,000 features, min_df=5, English stopwords, sublinear TF) saved
separately in `text_tfidf.npz` + vocab JSON; **fit on the mature cohort
only**, transformed for all rows; modeling agent decides to use/drop and must
refit inside CV folds if used.

**Excluded as features, kept as `leaky_*` diagnostics**: team_size, stage,
isHiring, top_company (current-state outcome proxies) and launched_at (floors
at 2010-01-17 = directory-listing time, not founding; unusable for pre-2010
batches).

## 3. Features we could NOT build at historical scale

Absent from this source; would dominate any real predictor if available:

- **Founder background** (count, prior exits, education, technical mix). No
  founder fields at all here. External: YC company pages, LinkedIn,
  Crunchbase. Reconstruction risk: *high* — profiles reflect today's bios;
  survivorship in who keeps a LinkedIn/Crunchbase presence.
- **Funding history** (post-batch rounds, amounts, investors). External:
  Crunchbase/PitchBook. Risk: *medium* — round dates allow point-in-time
  cuts, but coverage is worse for dead/small companies (differential
  missingness correlated with the label).
- **Early traction** (users/revenue at demo day, Launch-HN/Product Hunt
  reception, web traffic). External: Wayback Machine, HN archives. Risk:
  *medium-high* — patchy, biased toward consumer/dev-tool companies.
- **Team growth over time** (headcount trajectory in year 1–2). External:
  LinkedIn headcount snapshots, Wayback team pages. Risk: *high* — historical
  snapshots sparse before ~2015.
- `app_answers` exists in the schema but is null for 5,920/5,999 rows —
  effectively unusable.
