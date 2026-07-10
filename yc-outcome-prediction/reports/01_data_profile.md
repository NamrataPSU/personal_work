# YC Companies Dataset — Data Profile

Source: `data/raw/yc_companies_all.json` (see `data/raw/PROVENANCE.md`). Profiled with `src/02_profile.py`.

**Companies: 5,999** | **Fields: 29** | Duplicate `id`s: 0 | Duplicate `slug`s: 0 | Duplicate `name`s: 95

## 1. Fields, types, missingness

| field | dtype | n_missing | pct_missing | n_empty_string | n_empty_list | n_unique |
| --- | --- | --- | --- | --- | --- | --- |
| id | int64 (int) | 0 | 0.0 | 0 | 0 | 5999 |
| name | str (str) | 0 | 0.0 | 0 | 0 | 5904 |
| slug | str (str) | 0 | 0.0 | 0 | 0 | 5999 |
| former_names | object (list) | 0 | 0.0 | 0 | 3091 | 2898 |
| small_logo_thumb_url | str (str) | 0 | 0.0 | 0 | 0 | 5371 |
| website | str (str) | 1 | 0.02 | 33 | 0 | 5962 |
| all_locations | str (str) | 0 | 0.0 | 229 | 0 | 732 |
| long_description | str (str) | 30 | 0.5 | 370 | 0 | 5599 |
| one_liner | str (str) | 0 | 0.0 | 161 | 0 | 5838 |
| team_size | float64 (float) | 104 | 1.73 | 0 | 0 | 201 |
| industry | str (str) | 0 | 0.0 | 0 | 0 | 9 |
| subindustry | str (str) | 0 | 0.0 | 0 | 0 | 59 |
| launched_at | int64 (int) | 0 | 0.0 | 0 | 0 | 5709 |
| tags | object (list) | 0 | 0.0 | 0 | 933 | 3682 |
| tags_highlighted | object (list) | 0 | 0.0 | 0 | 5909 | 2 |
| top_company | object (bool) | 1 | 0.02 | 0 | 0 | 2 |
| isHiring | bool (bool) | 0 | 0.0 | 0 | 0 | 2 |
| nonprofit | bool (bool) | 0 | 0.0 | 0 | 0 | 2 |
| batch | str (str) | 0 | 0.0 | 0 | 0 | 50 |
| status | str (str) | 0 | 0.0 | 0 | 0 | 4 |
| industries | object (list) | 0 | 0.0 | 0 | 0 | 59 |
| regions | object (list) | 0 | 0.0 | 0 | 0 | 238 |
| stage | str (str) | 0 | 0.0 | 0 | 0 | 2 |
| app_video_public | bool (bool) | 0 | 0.0 | 0 | 0 | 2 |
| demo_day_video_public | bool (bool) | 0 | 0.0 | 0 | 0 | 2 |
| app_answers | object (bool) | 5920 | 98.68 | 0 | 0 | 2 |
| question_answers | bool (bool) | 0 | 0.0 | 0 | 0 | 2 |
| url | str (str) | 0 | 0.0 | 0 | 0 | 5999 |
| api | str (str) | 0 | 0.0 | 0 | 0 | 5999 |

Machine-readable copy: `data/raw/field_summary.csv`.

## 2. Outcome variable: `status`

| status | count | pct |
| --- | --- | --- |
| Active | 4136 | 68.94 |
| Inactive | 1045 | 17.42 |
| Acquired | 795 | 13.25 |
| Public | 23 | 0.38 |

## 3. Batch coverage

Distinct batch labels: 50. Year range: 2005–2027 (rows with unparseable batch year: 1, labels: ['Unspecified']).

### Companies per batch

| batch | count |
| --- | --- |
| Winter 2022 | 398 |
| Summer 2021 | 391 |
| Winter 2021 | 336 |
| Winter 2023 | 274 |
| Winter 2024 | 249 |
| Summer 2024 | 248 |
| Summer 2022 | 234 |
| Winter 2020 | 229 |
| Summer 2023 | 219 |
| Summer 2020 | 208 |
| Winter 2026 | 198 |
| Spring 2026 | 197 |
| Winter 2019 | 195 |
| Summer 2019 | 176 |
| Winter 2025 | 167 |
| Summer 2025 | 167 |
| Fall 2025 | 149 |
| Winter 2018 | 146 |
| Spring 2025 | 143 |
| Summer 2018 | 131 |
| Summer 2017 | 125 |
| Winter 2016 | 122 |
| Winter 2017 | 116 |
| Winter 2015 | 110 |
| Summer 2015 | 104 |
| Summer 2016 | 102 |
| Fall 2024 | 94 |
| Summer 2012 | 83 |
| Summer 2014 | 78 |
| Winter 2014 | 74 |
| Winter 2012 | 66 |
| Summer 2011 | 60 |
| Summer 2026 | 54 |
| Summer 2013 | 52 |
| Winter 2013 | 46 |
| Winter 2011 | 45 |
| Summer 2010 | 36 |
| Winter 2010 | 27 |
| Summer 2009 | 26 |
| Summer 2008 | 22 |
| Winter 2008 | 21 |
| Summer 2007 | 19 |
| Winter 2009 | 16 |
| Winter 2007 | 13 |
| Summer 2006 | 11 |
| Summer 2005 | 9 |
| Winter 2006 | 7 |
| Fall 2026 | 4 |
| Unspecified | 1 |
| Winter 2027 | 1 |

### Companies per year

| batch_year | companies |
| --- | --- |
| 2005.0 | 9.0 |
| 2006.0 | 18.0 |
| 2007.0 | 32.0 |
| 2008.0 | 43.0 |
| 2009.0 | 42.0 |
| 2010.0 | 63.0 |
| 2011.0 | 105.0 |
| 2012.0 | 149.0 |
| 2013.0 | 98.0 |
| 2014.0 | 152.0 |
| 2015.0 | 214.0 |
| 2016.0 | 224.0 |
| 2017.0 | 241.0 |
| 2018.0 | 277.0 |
| 2019.0 | 371.0 |
| 2020.0 | 437.0 |
| 2021.0 | 727.0 |
| 2022.0 | 632.0 |
| 2023.0 | 493.0 |
| 2024.0 | 591.0 |
| 2025.0 | 626.0 |
| 2026.0 | 453.0 |
| 2027.0 | 1.0 |
|  | 1.0 |

## 4. `status` by batch year

| batch_year | Acquired | Active | Inactive | Public | total | Acquired_pct | Active_pct | Inactive_pct | Public_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2005.0 | 6.0 | 0.0 | 3.0 | 0.0 | 9.0 | 66.7 | 0.0 | 33.3 | 0.0 |
| 2006.0 | 6.0 | 1.0 | 11.0 | 0.0 | 18.0 | 33.3 | 5.6 | 61.1 | 0.0 |
| 2007.0 | 12.0 | 3.0 | 16.0 | 1.0 | 32.0 | 37.5 | 9.4 | 50.0 | 3.1 |
| 2008.0 | 9.0 | 3.0 | 31.0 | 0.0 | 43.0 | 20.9 | 7.0 | 72.1 | 0.0 |
| 2009.0 | 12.0 | 8.0 | 21.0 | 1.0 | 42.0 | 28.6 | 19.0 | 50.0 | 2.4 |
| 2010.0 | 30.0 | 7.0 | 24.0 | 2.0 | 63.0 | 47.6 | 11.1 | 38.1 | 3.2 |
| 2011.0 | 42.0 | 26.0 | 37.0 | 0.0 | 105.0 | 40.0 | 24.8 | 35.2 | 0.0 |
| 2012.0 | 45.0 | 35.0 | 65.0 | 4.0 | 149.0 | 30.2 | 23.5 | 43.6 | 2.7 |
| 2013.0 | 33.0 | 28.0 | 36.0 | 1.0 | 98.0 | 33.7 | 28.6 | 36.7 | 1.0 |
| 2014.0 | 53.0 | 50.0 | 45.0 | 4.0 | 152.0 | 34.9 | 32.9 | 29.6 | 2.6 |
| 2015.0 | 62.0 | 85.0 | 63.0 | 4.0 | 214.0 | 29.0 | 39.7 | 29.4 | 1.9 |
| 2016.0 | 46.0 | 101.0 | 75.0 | 2.0 | 224.0 | 20.5 | 45.1 | 33.5 | 0.9 |
| 2017.0 | 62.0 | 110.0 | 68.0 | 1.0 | 241.0 | 25.7 | 45.6 | 28.2 | 0.4 |
| 2018.0 | 50.0 | 158.0 | 67.0 | 2.0 | 277.0 | 18.1 | 57.0 | 24.2 | 0.7 |
| 2019.0 | 57.0 | 228.0 | 86.0 | 0.0 | 371.0 | 15.4 | 61.5 | 23.2 | 0.0 |
| 2020.0 | 73.0 | 261.0 | 102.0 | 1.0 | 437.0 | 16.7 | 59.7 | 23.3 | 0.2 |
| 2021.0 | 79.0 | 523.0 | 125.0 | 0.0 | 727.0 | 10.9 | 71.9 | 17.2 | 0.0 |
| 2022.0 | 49.0 | 505.0 | 78.0 | 0.0 | 632.0 | 7.8 | 79.9 | 12.3 | 0.0 |
| 2023.0 | 47.0 | 392.0 | 54.0 | 0.0 | 493.0 | 9.5 | 79.5 | 11.0 | 0.0 |
| 2024.0 | 15.0 | 546.0 | 30.0 | 0.0 | 591.0 | 2.5 | 92.4 | 5.1 | 0.0 |
| 2025.0 | 7.0 | 612.0 | 7.0 | 0.0 | 626.0 | 1.1 | 97.8 | 1.1 | 0.0 |
| 2026.0 | 0.0 | 452.0 | 1.0 | 0.0 | 453.0 | 0.0 | 99.8 | 0.2 | 0.0 |
| 2027.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 100.0 | 0.0 | 0.0 |
|  | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 100.0 | 0.0 | 0.0 |

_Note: recent batches are overwhelmingly Active simply because outcomes have not had time to resolve — right-censoring. Labeling agent must handle this._

## 5. Key field distributions

### team_size (WARNING: current headcount — leaky as a feature)

missing: 104 | zero: 125 | mean: 46.9 | median: 6 | p90: 60 | p99: 700 | max: 8600

### industry (top 9 of 9 unique)

| industry | count |
| --- | --- |
| B2B | 3068 |
| Consumer | 867 |
| Healthcare | 684 |
| Fintech | 638 |
| Industrials | 399 |
| Real Estate and Construction | 159 |
| Education | 125 |
| Government | 41 |
| Unspecified | 18 |

### subindustry (top 20 of 59 unique)

| subindustry | count |
| --- | --- |
| B2B -> Engineering, Product and Design | 607 |
| B2B | 599 |
| B2B -> Infrastructure | 307 |
| B2B -> Productivity | 228 |
| B2B -> Marketing | 168 |
| Fintech | 167 |
| Consumer | 163 |
| B2B -> Operations | 145 |
| Healthcare -> Healthcare IT | 144 |
| B2B -> Sales | 135 |
| B2B -> Finance and Accounting | 134 |
| B2B -> Supply Chain and Logistics | 134 |
| B2B -> Retail | 130 |
| B2B -> Analytics | 127 |
| Fintech -> Payments | 125 |
| Education | 125 |
| Consumer -> Home and Personal | 123 |
| B2B -> Security | 114 |
| Healthcare -> Consumer Health and Wellness | 114 |
| Consumer -> Social | 113 |

### tags (flattened) (top 20 of 332 unique)

| tags | count |
| --- | --- |
| SaaS | 1078 |
| B2B | 1074 |
| Artificial Intelligence | 894 |
| AI | 805 |
| Fintech | 688 |
| Developer Tools | 522 |
| Marketplace | 301 |
| Generative AI | 247 |
| Consumer | 238 |
| Machine Learning | 229 |
| Healthcare | 202 |
| E-commerce | 190 |
| Analytics | 183 |
| Health Tech | 169 |
| Education | 164 |
| Productivity | 163 |
| Open Source | 162 |
| AI Assistant | 155 |
| Payments | 146 |
| Climate | 141 |

### regions (flattened) (top 20 of 98 unique)

| regions | count |
| --- | --- |
| America / Canada | 4632 |
| United States of America | 4499 |
| Remote | 3021 |
| Partly Remote | 1874 |
| Fully Remote | 1147 |
| Europe | 444 |
| South Asia | 225 |
| United Kingdom | 222 |
| India | 213 |
| Latin America | 213 |
| Unspecified | 179 |
| Canada | 144 |
| Southeast Asia | 104 |
| Mexico | 87 |
| Africa | 84 |
| Middle East and North Africa | 70 |
| France | 61 |
| Nigeria | 57 |
| Singapore | 52 |
| Brazil | 49 |

### stage (WARNING: current stage — leaky) (top 2 of 2 unique)

| stage | count |
| --- | --- |
| Early | 4928 |
| Growth | 1071 |

### all_locations (top values) (top 15 of 732 unique)

| all_locations | count |
| --- | --- |
| San Francisco, CA, USA | 2155 |
| New York, NY, USA | 392 |
| San Francisco, CA, USA; Remote | 357 |
|  | 229 |
| New York City, NY, USA | 118 |
| Los Angeles, CA, USA | 102 |
| London, England, United Kingdom | 101 |
| New York, NY, USA; Remote | 99 |
| Bengaluru, KA, India | 90 |
| Remote | 62 |
| Palo Alto, CA, USA | 57 |
| Boston, MA, USA | 56 |
| Mountain View, CA, USA | 55 |
| Toronto, ON, Canada | 55 |
| Seattle, WA, USA | 52 |

### Boolean flags

| field | true | false | missing |
| --- | --- | --- | --- |
| top_company | 91 | 5907 | 1 |
| isHiring | 1480 | 4519 | 0 |
| nonprofit | 42 | 5957 | 0 |
| app_video_public | 100 | 5899 | 0 |
| demo_day_video_public | 157 | 5842 | 0 |
| question_answers | 213 | 5786 | 0 |

### launched_at (unix timestamp)

missing: 0 | min: 2010-01-17 09:03:33 | max: 2026-07-01 23:28:48 | pre-2005 (suspicious): 0

## 6. Field roles for modeling (feature vs leaky vs identifier)

| field | role |
| --- | --- |
| id | identifier |
| name | identifier |
| slug | identifier |
| former_names | feature-candidate (weak; pivot signal) |
| small_logo_thumb_url | identifier/URL |
| website | identifier/URL (presence/liveness could be outcome-adjacent) |
| all_locations | feature-candidate (location at listing; mostly stable) |
| long_description | feature-candidate (text; describes CURRENT product — mild leakage risk) |
| one_liner | feature-candidate (text; same caveat) |
| team_size | LEAKY/outcome-adjacent (CURRENT headcount reflects outcome, not starting state) |
| industry | feature-candidate |
| subindustry | feature-candidate |
| launched_at | feature-candidate (launch timestamp; relative to batch = early-traction signal) |
| tags | feature-candidate |
| tags_highlighted | identifier/derived (UI artifact of tags) |
| top_company | LEAKY (YC's post-hoc success designation) |
| isHiring | LEAKY/outcome-adjacent (current hiring status) |
| nonprofit | feature-candidate |
| batch | feature-candidate (cohort/vintage; also defines eval splits) |
| status | TARGET (outcome label source) |
| industries | feature-candidate (list form of industry+subindustry) |
| regions | feature-candidate (includes Remote flags) |
| stage | LEAKY/outcome-adjacent (CURRENT funding stage = outcome proxy) |
| app_video_public | feature-candidate (weak; disclosure choice) |
| demo_day_video_public | feature-candidate (weak; disclosure choice) |
| app_answers | feature-candidate (application answers; but ~all null) |
| question_answers | identifier/flag (whether app_answers exist) |
| url | identifier/URL |
| api | identifier/URL |

**Founder-level data: NOT present.** No founder names, counts, bios, prior companies, or education fields exist in this dataset. `app_answers` is null for 5920/5999 rows. Founder features would require a separate source (e.g., per-company pages or YC's site).

## 7. Data quirks

- Duplicate ids: 0; duplicate slugs: 0; names shared by >1 company: 85.
- Batch label formats present: ['Fall', 'Spring', 'Summer', 'Unspecified', 'Winter'] + year (long form only in this dump); 'Unspecified' batch rows: 1.
- Empty strings vs nulls: `one_liner` empty-string rows: 161, `long_description` empty: 370, `website` empty: 33, `all_locations` empty: 229 — missing text is encoded as '' not null.
- team_size: 104 null and 125 zero values (zero for many dead companies).
- Companies with former_names: 2908 (~half the dataset; mostly legal-name variants, not true pivots — treat with care).
- launched_at min is 2010-01-17 even though batches go back to 2005 — it records when the company was ADDED/LAUNCHED on the YC directory, not founding date; unreliable as a company-age feature for pre-2010 batches.
