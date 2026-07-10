# 05 — Portfolio simulation: random vs model-guided selection

Code: `src/07_simulation.py`. Machine-readable results:
`models/simulation_results.json`. Figures: `simulation_portfolios.png`,
`simulation_auc_by_variant.png`.

## Question

An investor must pick k companies from the 2019–21 strict-label holdout
(n = 523, realized success rate 40.2%). What does random selection achieve,
and how much better is picking the model's top-k scores?

## Setup

- **Training:** batches ≤ 2018 only (n = 1,052). Holdout untouched.
- **Pre-investment discipline:** no `leaky_*` columns; descriptions scrubbed
  of outcome-revealing sentences (sentence-level, same scrubber as
  `05_model.py`); every text representation fit on the dev pool only.
- **Random benchmark:** 20,000 uniform draws of k companies per k ∈
  {10, 25, 53, 100}; k = 53 is the top decile.
- **Text representations.** huggingface.co (and all pretrained-weight hosts)
  are unreachable from this environment, so no pretrained sentence
  transformer was possible. Two dense fallbacks, both dev-fit:
  - **LSA**: TF-IDF(2000) → TruncatedSVD(256) (33.4% variance retained);
  - **w2v-SIF**: word2vec (100d, 30 epochs) trained on dev-pool scrubbed
    text, SIF-weighted document means, first-PC removal.
  These are swappable for a real sentence transformer (e.g.
  all-MiniLM-L6-v2) once network policy allows; the pipeline is unchanged.

## Results (holdout; lift = model rate / random mean)

Random selection converges to the base rate at every k (mean 40.1–40.2%);
what shrinks with k is the spread (90% interval at k=10: 20–70%; at k=53:
30–51%).

| variant | AUC | PR-AUC | k=10 | k=25 | k=53 (top decile) | k=100 |
|---|---|---|---|---|---|---|
| logit structured | 0.546 | 0.437 | 50% (1.25×) | 44% (1.09×) | 43% (1.08×, beats 65%) | 45% (1.12×) |
| XGB structured | 0.566 | 0.459 | 40% (1.00×) | 52% (1.29×) | 51% (1.27×, beats 94%) | 48% (1.20×) |
| **XGB struct + LSA** | 0.558 | **0.473** | 60% (1.50×) | **56% (1.39×)** | **57% (1.41×, beats 99.3%, p = 0.007)** | **50% (1.25×)** |
| XGB LSA only | 0.570 | 0.468 | 80% (2.00×, p = 0.011) | 64% (1.59×) | 51% (1.27×) | 44% (1.09×) |
| XGB struct + w2v | 0.572 | 0.471 | 70% (1.75×) | 52% (1.29×) | 53% (1.32×, beats 97%) | 50% (1.25×) |
| XGB w2v only | 0.566 | 0.452 | 60% (1.50×) | 36% (0.90×) | 32% (0.80×, beats 8%) | 44% (1.09×) |

## Reading

1. **The model edge is real at portfolio scale.** At the top decile, the
   best variant (structured + LSA) returns **56.6% winners vs 40.2% for
   random** — a 1.41× lift that beats 99.33% of the 20,000 random
   portfolios (p = 0.0067). Structured-only achieves 50.9% (1.27×,
   p = 0.063).
2. **Dense text representations help where TF-IDF hurt.** In `03_modeling`,
   adding raw TF-IDF to structured features *lowered* pooled temporal AUC.
   Dense low-rank representations (LSA, w2v-SIF) instead add PR-AUC
   (0.459 → 0.473) and portfolio lift — less overfitting surface than
   2,000 sparse columns.
3. **Tiny portfolios are lottery tickets.** At k = 10 the random 90%
   interval spans 20–70%; the LSA-only model's 80% (8/10) looks
   spectacular but is a single draw from a fat distribution (p = 0.011,
   and unstable — the same model is only average at k = 53 on some
   variants). Conclusions should rest on k ≥ 25.
4. **Text-only remains unreliable** (w2v-only: 32% at top decile, *worse*
   than random). Text is a complement to structure, not a substitute.
5. **Caveats.** Single holdout realization — the model's top-k rate is
   itself one draw (binomial SE at k=53 ≈ ±6.8pp); scrubbing removes
   explicit outcome phrases but subtler curation survivorship in
   current-day text cannot be excluded; and these embeddings are fallbacks,
   not transformers — the text arm's ceiling is untested until
   huggingface.co is allowlisted.

## Bottom line

Random picking of 53 companies from this cohort wins ~40% of the time per
company. Following the model's top-decile ranking wins ~51–57%, an edge that
would occur by chance well under 1-in-100 times for the best variant. That is
economically meaningful for screening — and still far from sufficient as a
sole decision mechanism.
