# Processed data — YC outcome prediction

Produced by `src/03_labels.py` and `src/04_features.py` from
`data/raw/yc_companies_all.json` (5,999 companies). Reference date for ages
and maturity: **2026-07-02**.

## Files

| file | contents |
| --- | --- |
| `features.parquet` | 5,999 rows x 60 cols: metadata, both labels, features, `leaky_*` diagnostics. All companies included; labeled modeling population = rows where the chosen label is non-null. |
| `feature_manifest.csv` | Column-role manifest: `column, role, description, leakage_risk, justification`. Roles: metadata, label, feature_numeric, feature_categorical, feature_text, leaky_diagnostic. |
| `labels.csv` | Label-only view (id, batch parsing, age, is_mature, both labels). Redundant with features.parquet; kept for convenience. |
| `text_tfidf.npz` | scipy sparse TF-IDF matrix, shape (5999, 2000). Row i corresponds to row i of features.parquet. |
| `text_tfidf_vocab.json` | Vocabulary + fit parameters for the TF-IDF matrix. |

## Labels (mature cohort = batch_date <= 2021-07-02, i.e. >= ~5 years old)

- `label_strict`: 1 = Acquired/Public, 0 = Inactive, NaN = Active or immature.
  N = 1,575 (700 pos / 875 neg, 44.4% positive).
- `label_survival`: 1 = Acquired/Public/Active (mature), 0 = Inactive.
  N = 3,202 (2,327 pos / 875 neg, 72.7% positive).
- 1,627 mature Active companies differ between the two variants.
- The 1 "Unspecified"-batch row has NaN batch_date and no labels.

## Leakage warnings

- **Never use `leaky_*` columns as features** (`leaky_team_size`,
  `leaky_stage`, `leaky_is_hiring`, `leaky_top_company`,
  `leaky_launched_at`). They encode current state / post-hoc outcome.
  `leaky_launched_at` additionally floors at 2010 (directory-listing time,
  not founding) and is unusable for pre-2010 batches.
- **Text is current-day**: `one_liner` / `long_description` describe the
  company as displayed today. 108 rows' text literally contains
  "acquired by" (83 of them status=Acquired). TF-IDF and the length/
  `has_description` features therefore carry medium leakage — ablate them,
  and consider scrubbing outcome phrases before trusting text-driven gains.
  TF-IDF was fit only on the mature cohort; proper per-fold refitting is the
  modeling agent's responsibility (raw `text_clean` column is provided).
- Tags may be re-curated over time (the AI tags especially); the three
  overlapping AI tags ("AI", "Artificial Intelligence", "Generative AI")
  were merged into the single `tag_ai` indicator.
- `app_video_public` / `demo_day_video_public`: the videos themselves are
  pre/end-of-batch artifacts (low risk), but the *publish* choice may be made
  later — treated as low-leakage features.

## Missing-value conventions

Empty strings in the raw data were converted to NaN (`one_liner`,
`long_description`, `all_locations`, `industry`/`subindustry`
"Unspecified"). `text_clean` is NaN for 57 rows with no text at all;
TF-IDF rows for these are all-zero. `primary_region` is NaN when the regions
list contains only Remote/Unspecified entries.
