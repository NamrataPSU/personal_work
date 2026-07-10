# 06 — Founder features & sentence-transformer embeddings

Code: `src/08_scrape_founders.py` (scrape), `09_embed_st.py` (embeddings),
`10_final_eval.py` (evaluation). Results: `models/final_eval_results.json`,
`figures/final_eval.png`. Data: `data/raw/founders.jsonl`.

## What was added

- **Founder data**, scraped from each company's ycombinator.com page
  (5,998/5,999 pages, 8.3 min, 1 HTTP error). Pages embed a JSON blob with
  founder name, title, bio, and `is_active`. Coverage on the strict-label
  cohort: **97.8%**.
- **Pre-investment founder features:** `n_founders` (all listed founders,
  active + departed), `solo_founder`, `has_technical_founder` (title regex:
  CTO/engineer/technical/scientist), `has_ceo_title`, `founders_missing`.
- **Quarantined:** `is_active`-derived counts. Validation of that choice:
  companies with ≥1 departed founder succeed **28.7% vs 56.1%** — a huge
  gap that is overwhelmingly *reverse causation* (company dies/exits →
  founders leave → flag flips). Textbook outcome proxy; never a feature.
  Founder *bios* also excluded (written post-hoc).
- **Text arm upgraded:** TF-IDF/LSA replaced by
  `sentence-transformers/all-MiniLM-L6-v2` (384-dim) embeddings of
  scrubbed descriptions. Frozen pretrained encoder — nothing fit on our
  data, so no per-fold refitting is needed; scrubbing remains the only
  text leakage control.

## Results (label_strict; holdout 2019–21, n = 523, base 40.2%)

| variant | temporal CV AUC | holdout AUC | holdout PR-AUC | top-25 | top decile (k=53) | k=100 |
|---|---|---|---|---|---|---|
| struct (baseline) | 0.637 | 0.566 | 0.459 | 52% (1.30×) | 51% (1.27×) | 48% (1.20×) |
| struct + founders | **0.644** | 0.573 | 0.466 | 48% | 45% | 55% (1.37×, p=.001) |
| **struct + MiniLM** | 0.577 | 0.564 | **0.501** | **76% (1.90×, p<.001)** | **62% (1.55×, p=.001)** | 47% |
| struct + founders + MiniLM | 0.568 | **0.573** | 0.500 | 68% (1.70×, p=.004) | 60% (1.51×, p=.001) | 52% (1.29×, p=.005) |
| MiniLM only | 0.590 | 0.541 | 0.464 | 56% | 49% | 51% |
| founders only | 0.575 | 0.532 | 0.422 | 44% | 43% | 42% |

## Findings

1. **Founder structure adds almost nothing.** +0.007 temporal AUC and
   +0.007 holdout AUC over the structured baseline; founders-only ranks
   barely above chance on the holdout (0.532) and its portfolios are
   indistinguishable from random (p ≥ 0.25 at k ≤ 53). The raw
   success-by-founder-count table explains why — it is nearly flat:
   solo 44.0%, two founders 43.7%, three 49.8%, four 31.9% (n=47).
   **Within YC-admitted companies, solo founders do not underperform** —
   contrary to standard VC folklore. Plausible reading: YC's selection
   already conditions on founder quality, so the residual variation
   carries little signal (a range-restriction effect). This does NOT mean
   founders don't matter — it means founder *count/role structure*,
   measured post-selection, doesn't.
2. **The sentence transformer is the best portfolio selector yet.** The
   struct+MiniLM top-25 portfolio hits **76% winners vs 40.2% base
   (1.90×, p < 0.001)**, top decile 62.3% (1.55×) — beating LSA
   (56.6%) and TF-IDF, with the best PR-AUC (0.501). Same shape as
   before, sharper: dense text signal concentrates in the extreme tail
   (rich, specific, technical descriptions), while *diluting* mid-ranking
   discrimination — pooled temporal AUC drops to 0.577. Rule of thumb:
   **rank with structure, cherry-pick the tail with text**.
3. **Combined model is the best all-rounder** (holdout 0.573 with
   significant lift at every k) but not the best at anything individually.
4. Caveats unchanged: single holdout realization (top-25 binomial SE
   ≈ ±9pp — 76% is directional, not a promise); scrubbed but current-day
   text; probabilities still require recalibration (ranks only).

## Verdict on the brief's founder-features question

The project's last open feasibility question is now answered empirically:
founder-structure features CAN be reconstructed at historical scale
(97.8% coverage, one 8-minute scrape) — and once reconstructed honestly
(departed-founder signal quarantined as reverse causation), they add
approximately nothing within a YC-selected population. The expensive
founder-data enrichments (LinkedIn histories, psychometrics) would have
to overcome both the reconstruction-bias problem AND this
range-restriction ceiling. Meanwhile the cheap upgrade — a frozen
pretrained sentence encoder on scrubbed text — delivered the largest
portfolio-selection gain in the project.
