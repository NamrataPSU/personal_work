# 07 — Advanced founder features: similarity, prior experience, repeat founders

Code: `src/12_founder_advanced.py`. Results: `models/founder_advanced_results.json`,
`figures/founder_advanced.png`, features in `data/processed/founder_advanced.csv`.

## Features built (per company, strict cohort)

**Clean (point-in-time reconstructable):**
- `any_repeat_yc_founder` / `n_prior_yc_companies` — founder user-ID matched
  to an *earlier-batch* YC company in the dataset (250 multi-company founders
  found; ordering by batch date makes this hindsight-free).

**Bio-derived (medium leakage; two-stage scrubbing = outcome sentences +
career-continuation sentences like "currently COO at…", "went on to…"):**
- `bio_industry_sim_mean/max` — cosine similarity between MiniLM embeddings
  of each founder's scrubbed bio and the company's industry/subindustry/tags
  text ("does the founder's background match the domain?").
- `prior_founder_exp`, `big_employer_exp` (FAANG/top consultancies/banks),
  `phd_flag`, `prof_flag`, `mba_flag`, `yoe_mentioned`.
- `founders_bio_coverage` — share of founders with a non-empty bio
  (kept visible as a liveness indicator, interpreted with suspicion).

## Univariate reality check (success rate vs 44.4% cohort base)

| feature | with | without | n(with) | verdict |
|---|---|---|---|---|
| repeat YC founder | 50.0% | 44.2% | 60 | directionally +, not significant (SE ±6.5pp) |
| PhD founder | 54.3% | 44.1% | 46 | directionally +, not significant |
| bio–industry similarity (top tercile) | 49.3% | ~44.9% | 347 | mild + |
| big-tech/consultancy experience | 45.5% | 44.3% | 209 | nothing |
| prior founding mentioned | 42.9% | 44.5% | 63 | nothing |
| MBA founder | 31.6% | 44.8% | 38 | directionally −, small n |
| professor/researcher | 31.8% | 44.6% | 22 | directionally −, small n |
| bio coverage > 1/3 | 46.7% | 41.2% | 937 | liveness artifact, as predicted |

## Model results (2019–21 holdout, n=523, base 40.2%)

| variant | temporal CV | holdout AUC | k=25 | top decile (k=53) | k=100 |
|---|---|---|---|---|---|
| struct (ref) | 0.637 | 0.566 | 52% (p=.149) | 51% (1.27×, p=.061) | 48% (1.20×) |
| struct + adv founders | 0.641 | 0.565 | 56% (p=.071) | 57% (1.41×, p=.008) | 54% (1.34×, p=.001) |
| struct + adv founders + MiniLM | 0.610 | 0.559 | 72% (1.80×, p=.001) | 62% (1.55×, p=.001) | 54% (1.34×, p=.001) |
| adv founders only | 0.530 | 0.504 | 36% (worse than random) | 34% | 38% |

## Conclusions

1. **Advanced founder features alone have zero holdout ranking power**
   (AUC 0.504 ≈ coin flip; portfolios at or below random). Even the
   conceptually attractive constructions — domain-fit similarity, prior
   founding, elite employers — do not separate winners inside a
   YC-admitted population.
2. **As additions they are decoration, not signal:** +0.004 temporal AUC,
   −0.001 holdout AUC. The apparent top-decile improvement of struct+fadv
   (57% vs 51%) is within tail noise (±7pp SE), and the strongest arm
   (+MiniLM, 62% top decile) matches what struct+MiniLM already achieved
   *without* founder features in report 06.
3. **The interesting weak positives** — repeat YC founder (+5.8pp, n=60),
   PhD (+10.2pp, n=46), domain-fit similarity (+4.4pp in top tercile) —
   have the right signs per VC folklore but sample sizes that keep them
   indistinguishable from noise. A 10× larger multi-accelerator dataset
   could settle them; YC alone cannot.
4. This closes the founder-feature question for this data source twice
   over: neither simple structure (report 06) nor semantic/experience
   constructions (this report) add predictive value within YC's
   pre-selected population. Range restriction is the operative constraint —
   YC partners already priced founder quality into admission.
