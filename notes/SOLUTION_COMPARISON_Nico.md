# Comparison of the four hackathon solutions

## Executive recommendation

**Submit Nico's root solution as the forecasting core.** It has the strongest
claim to a fair, operational day-ahead result: it models the original 15-minute
target, applies the real pre-gate information cutoff, uses no realized
target-day weather, selects the model on two chronological validation folds,
and evaluates the selected model once on a later test period.

Its honest headline is:

- global LightGBM selected on validation;
- test portfolio WAPE **12.30%**, versus **14.12%** for the daily/weekly blend;
- median household nMAE **0.590**, versus **0.651** for the blend;
- it beats the blend for **92.3% of test households**.

The numerically lower 6–8% errors reported by the other folders are not evidence
that they forecast better. They use hourly aggregation, different households
and dates, and realized target-day weather. Mona and Zhineng also multiply a
predicted per-active-household mean by the *realized* number of reporting
households at the target hour. Zhineng additionally uses a target-aligned
24-hour load lag that is unavailable for much of the delivery day at its stated
forecast origin.

If the submission may combine work, the best package is:

1. Nico's leakage-safe forecast and evaluation as the quantitative core.
2. Zhineng's structure for the business and uncertainty story, but only after
   recomputing those figures from Nico's forecasts and a shared German price
   assumption.
3. Anita's PV-identification result as an exploratory Level 1 insight, clearly
   separated from the primary forecast claim.
4. Mona's newsvendor explanation as an intuitive decision rule, without
   carrying over the Swiss-price result or the oracle-weather accuracy number.

Do **not** splice headline metrics from different solutions into one leaderboard.

## What each solution actually does

| Dimension | Nico — repository root | Anita Aghazadeh | Mona Alkhayat | Zhineng |
|---|---|---|---|---|
| Primary objective | One global model that works across households and aggregates to a portfolio | Global hourly model plus PV identification and PV/non-PV variants | Full L0–L3 chain: clustering, forecasting, procurement cost, quantile bid | Clustering hypothesis tested with validation, German prices, uncertainty and a slide deck |
| Target resolution | **15 minutes**, 96 outputs/day | Hourly | Hourly | Hourly |
| Forecast unit | Household forecast, then portfolio sum | Household forecast, then portfolio sum | Portfolio/cluster mean per active household, then scaled | Portfolio/cluster mean per active household, then scaled |
| Stated information time | Strictly before **10:00 UTC on D−1** for delivery day D | End of D−2; deliberately conservative | Described as noon D−1, but load features are at least 48 h old | Midnight at start of D−1 |
| Load lags | Cutoff-aware D−1 where available; otherwise D−2, D−3, D−7, D−14 and past-only summaries | 48–192 h lags and past rolling summaries | 48 h, 168 h and D−2 mean | 24 h, 48 h and 168 h |
| Target-day weather | **No**; past observations only | **Yes: realized weather oracle** | **Yes: realized weather oracle** | **Yes: realized weather oracle**, including same-day daily aggregates |
| Main model | Global LightGBM with household scaling and metadata | Global LightGBM; RF PV classifier; optional PV flag/group models | HistGradientBoostingRegressor globally and by k-medoids cluster | HistGradientBoostingRegressor globally and by k-means cluster |
| Model selection | Two chronological folds covering summer and winter | Early stopping within training; three separate reported test windows | k chosen by training-period silhouette; no outer accuracy-validation fold | Dedicated sub-train/validation selects K by cost, then one final test |
| Main test | 2023-05-29 to 2024-02-28 | Calendar 2023; also Apr–Sep 2023 and 2023-09-30 onward | 2023-09-01 to 2024-02-27 | 2023-03-15 to 2024-02-27 |
| Test households | 403 with evaluable rows; 409 in source arrays | 403 in the calendar-2023 run | 390 | 328 |
| Uncertainty / cost | Not in the finalized Level 0 result | No | Rolling 28-day residual quantiles; Swiss prices and assumed imbalance penalties | Rolling 56-day hour-conditioned quantiles; German prices, bootstrap intervals and assumed penalties |
| Submission assets | Executed EDA and Level 0 notebook, artifacts and tests | Concise PDF, scripts and result plots | Concise PDF, four-level plots and scripts | Most complete presentation package, report, figures and PPTX |

## Claimed results, interpreted correctly

The table below compares improvement **within each solution's own protocol**.
It is useful for judging whether an approach adds value over its own baseline,
but not for ranking absolute accuracy across folders.

| Solution and experiment | Baseline | Proposed model | Within-protocol conclusion |
|---|---:|---:|---|
| Nico, test portfolio WAPE | Daily/weekly blend 14.12% | Selected LightGBM 12.30% | 12.9% relative WAPE reduction; strong household breadth |
| Anita, 2023 hourly portfolio WAPE | Seven-day same-hour mean 15.87% | Global LightGBM 8.49% | 46.5% relative reduction, with realized weather |
| Anita, 2023 Level 1 | Global LightGBM 8.49% | Separate PV/non-PV 8.34% | Only 0.15 percentage point gain; correctly described as inconclusive |
| Mona, seed-42 hourly nMAE | Last week 20.10% | Global model 6.12% | Large gain over persistence, with realized weather |
| Mona, seed-42 clustering | Global model 6.12% | Per-cluster 6.01% | 1.7% relative accuracy gain; only 0.2% cost gain |
| Mona, five-seed clustering claim | No clustering 6.24% | k-medoids 5.96% | 4.6% relative gain across the recorded runs |
| Zhineng, final hourly nMAE | K=1 global model 6.275% | Validation-selected K=4 6.245% | 0.48% relative gain; K=5 was better on test but was not selected |
| Zhineng, test imbalance cost | €32.319/home-year | €32.259/home-year | 0.19% saving; bootstrap interval crosses zero |

### Why the absolute values differ so much

1. **Hourly aggregation is easier than 15-minute forecasting.** It smooths
   household spikes and reduces four target intervals to one.
2. **Portfolio aggregation is easier than household forecasting.** Opposite
   household errors cancel. Nico deliberately makes household performance a
   primary requirement rather than optimizing only the aggregate.
3. **Realized weather is an oracle.** Anita, Mona and Zhineng use observations
   from the delivery day as if they were weather forecasts. This is reasonable
   as a labelled upper-bound experiment, but not directly comparable with
   Nico's operational no-future-weather result.
4. **Dates and seasons differ.** Mona tests six months beginning in September;
   Anita's headline spans calendar 2023; Zhineng tests nearly a year beginning
   in March; Nico tests late May through late February.
5. **The set of households differs.** Eligibility rules leave between 328 and
   403 test households, and availability changes within some portfolio targets.
6. **The forecast contract differs.** Nico uses a 10:00 UTC D−1 cutoff, Anita
   and Mona deliberately discard more recent load, and Zhineng uses information
   that extends beyond its claimed issuance time.

## Detailed assessment

### 1. Nico — root solution

#### Approach

The root pipeline constructs one global household model across almost the whole
dataset. Each target and load feature is normalized by the household's trailing
28-day mean, so large households do not dominate learning. It uses
target-aligned daily and weekly lags, recent levels and trends, calendar and
holiday features, static household metadata, and weather observed only before
the bid cutoff. The selected L2 LightGBM is a conditional-mean forecast, which
is appropriate when household forecasts are summed for procurement.

Two expanding validation folds cover summer 2022 and winter 2022/23. The model
is selected by pooled median household nMAE among mean-forecast models, then
reported on the May 2023–February 2024 test. All compared models share the same
household-target rows.

Track A adds holiday, morning-trend and shared portfolio signals. Its best
validation variant improves the median household nMAE from about 0.621 to
0.612, but its test result has intentionally not been generated. Track B tests
LightGBM, XGBoost, CatBoost, MLP, CNN and GRU capacity. Those results are useful
research, but they should not replace the already selected Level 0 result:
many alternatives have now been inspected against the test period, and the CNN
has good household nMAE but worse portfolio bias/WAPE than the selected
LightGBM.

#### Strengths

- Best information-set discipline of all four solutions.
- Only solution retaining the native 15-minute challenge resolution.
- Strong chronological selection with both summer and winter validation.
- Household-first evidence prevents a few large loads or error cancellation
  from hiding poor customer-level forecasts.
- No target-day weather oracle.
- Clear artifacts, tests, feature importance, PV slices and an executed
  notebook.
- Correctly distinguishes the household-optimal median model from the
  portfolio-optimal mean model.

#### Limitations

- The finalized story covers Level 0 deeply but does not yet provide a common
  price-based metric or calibrated uncertainty.
- Portfolio scoring sums the households with evaluable rows at each timestamp;
  it does not yet express a fully fixed contracted portfolio when meters are
  unavailable.
- The headline WAPE looks worse than the hourly/oracle solutions unless the
  resolution and information-set difference is explained prominently.
- Track A and B should be presented as experiments, not additional test-set
  model selection.

#### What to claim

Claim the selected Level 0 LightGBM result from `artifacts/level0`, not the best
number visible in Track B. Emphasize real cutoff compliance, household breadth,
and the 12.9% portfolio-WAPE improvement over the same-protocol blend.

### 2. Anita Aghazadeh

#### Approach

Anita aggregates only complete sets of four quarter-hours to hourly consumption
and creates a complete hourly grid before shifting, so gaps do not turn row
shifts into false time lags. Load features use data no newer than D−2. A global
LightGBM is compared with weekly, D−2 and seven-day same-hour baselines.

For Level 1, a random forest identifies PV-like households from training-only
seasonal load-shape features. Known survey labels are used for cross-validation;
unknown labels are inferred. The forecast comparison tests a global model, a
global model with the PV flag, and separate PV/non-PV models.

#### Strengths

- Concise, understandable and reproducible three-script pipeline.
- Correct timestamp grid and conservative load-feature timing.
- Strong calendar-year test plus two robustness windows.
- Useful Level 1 insight: PV status can be inferred reasonably well, but
  splitting the forecast model by PV offers little robust benefit.
- Reports household-hour, hourly portfolio and daily portfolio WAPE rather
  than one favorable metric.
- The write-up is appropriately cautious about the small PV-model gain.

#### Limitations

- Target-hour realized temperature, humidity, sunshine, wind and rain—and
  whole-target-day weather summaries—are used as forecast inputs. The 8.3–8.5%
  portfolio WAPE is therefore an oracle-weather result.
- There is no outer validation period for choosing among the global/PV model
  variants; all three are compared on the reported test windows.
- Portfolio membership follows available rows rather than a fixed contract,
  and the different test windows have slightly different household sets.
- No business-cost or uncertainty layer.
- The stated forecast time is intentionally much earlier than the real gate,
  leaving useful D−1 morning load unused.

#### What to reuse

The PV classifier and the honest conclusion that PV households are harder but
do not warrant separate models. Do not compare its absolute WAPE directly with
Nico's no-oracle 15-minute WAPE.

### 3. Mona Alkhayat

#### Approach

Mona gives the clearest single-chain narrative across all challenge levels.
K-medoids clusters 390 eligible homes from seasonal profiles, absolute level,
heating sensitivity and a sunshine/PV signature. One boosted-tree model is
trained globally and one per cluster. Forecast errors are translated into cost
with Swiss hourly day-ahead prices and assumed asymmetric penalties. Rolling
28-day residual quantiles support a newsvendor bid at tau = 0.625.

#### Strengths

- Best concise end-to-end explanation of why forecasts matter economically.
- Interpretable PV-like cluster, validated against survey labels without using
  those labels for clustering.
- Load lags are conservatively at least 48 hours old.
- Cost sensitivity acknowledges that assumed penalties can change the model
  ranking.
- Rolling uncertainty uses only old residuals and connects directly to a
  decision rule.
- Reproducible plots and a one-page report are already available.

#### Limitations

- Uses realized delivery-day weather, including same-day aggregates.
- Scales the per-household forecast with the realized number of active meters
  at each target hour. That denominator is not known when the bid is made and
  changes the target away from a fixed contracted portfolio.
- No separate outer validation period for forecast architecture. The test is
  used for the main comparison and five-seed summary.
- The clustering increment is economically negligible in the reported main
  scenario: roughly 0.2% cost improvement.
- Uses Swiss prices and `Europe/Zurich`, while the challenge context and meter
  geography point to Germany. This weakens the literal euro claim.
- The reported tau optimum is evaluated on a grid that includes the test
  outcome; the theoretically chosen tau remains valid, but “cheapest in
  hindsight” is descriptive rather than independent validation.

#### What to reuse

The L0-to-L3 narrative, PV-cluster visualization and intuitive newsvendor
explanation. Recompute any euro result with a common German price series and
operational forecasts before making it a submission headline.

### 4. Zhineng

#### Approach

Zhineng has the most complete formal experiment around the clustering question.
It builds nine size-free load/temperature/PV-signature features, evaluates
K=1–6, chooses K on a six-month validation window using imbalance cost and a
parsimony rule, refits before a nearly one-year test, and applies German DE-LU
prices. It also provides rolling residual intervals, block-bootstrap confidence
intervals, penalty sensitivity, a report and a presentation deck.

#### Strengths

- Strongest outer selection design among the three hourly solutions.
- K=1 is included, so clustering must beat a proper global-model baseline.
- Uses German rather than Swiss prices.
- Reports confidence intervals and honestly shows that the selected clustering
  gain is not statistically persuasive on test.
- Prediction intervals are well calibrated: the nominal 90% interval covers
  about 89.3% for K=4.
- The uncertainty and bidding analysis reports negative as well as positive
  results: its tau-star bid is slightly more expensive than its point forecast
  on test, with an interval crossing zero.
- Most polished set of reusable figures and slides.

#### Limitations

- `lag_24h` is target-aligned. For a target at hour h on day D it reads hour h
  on D−1. At the stated midnight-at-start-of-D−1 origin, and even at a noon
  gate, later D−1 hours are not yet observed. This is load leakage for a large
  part of the output horizon.
- Realized target-hour temperature and sunshine are inputs; `temp_day_mean` and
  `sun_day_sum` use the complete delivery day's weather.
- Like Mona, it multiplies by the realized target-hour active-household count.
- The selected K=4 improves test nMAE by only 0.48% and cost by 0.19%; both the
  cost interval and the tau-bid interval cross zero. The scientifically correct
  conclusion is that clustering is interpretable but not proven to improve
  portfolio procurement.
- The polished deck could make the small effect look more decisive than the
  tables support unless the null result is stated clearly.

#### What to reuse

The validation/test structure, German price source, bootstrap comparison,
uncertainty diagnostics and slide design. Do not submit its forecast accuracy
as operational until the 24-hour lag, target weather and active-count issues
are corrected.

## A fair comparison protocol

If time permits one harmonized rerun, use the following contract for every
approach.

### Forecast task

- Target: `kWh_received_Total` for a fixed portfolio.
- Delivery: all 96 quarter-hours of UTC day D.
- Information cutoff: strictly before 10:00 UTC on D−1.
- Primary operational track: no future weather.
- Optional weather track: the same archived day-ahead weather forecast for all
  models. Realized weather may be reported only as an oracle upper bound.
- Never use realized target-hour meter availability as an input. Either require
  a fixed cohort or forecast contracted members and report availability
  separately.

### Population and rows

- Freeze one cohort using only pre-test availability criteria.
- Use the same households, timestamps and actual values for all models.
- Keep gaps as gaps; do not interpolate consumption targets.
- Score a timestamp only if a predeclared actual-meter coverage threshold is
  met, and expose that coverage.
- For an hourly secondary comparison, aggregate the *same* 15-minute rows after
  forecasts are produced. This lets the hourly folders participate without
  replacing the primary 15-minute task.

### Splits and selection

- Use Nico's two seasonal validation folds and final test window, or another
  single split agreed before rerunning.
- Recompute clusters inside each training fold.
- Tune K, hyperparameters, model class and bid policy on validation only.
- Refit the selected configuration before test and evaluate test once.
- Treat the already inspected common test as descriptive if additional tuning
  is performed now; do not claim it remains untouched.

### Metrics

Primary:

- portfolio WAPE on the common rows;
- median and 90th-percentile household nMAE;
- share of households beating the common daily/weekly baseline.

Supporting:

- portfolio bias and RMSE;
- error by season, hour, horizon and PV status;
- prediction coverage and runtime;
- German price-weighted imbalance cost under one shared formula;
- paired daily or weekly block-bootstrap interval for model differences;
- empirical interval coverage, width and pinball loss for uncertainty models.

### Minimal rerun matrix

To answer the real scientific questions without rerunning every variant:

1. Common daily/weekly persistence blend.
2. Nico global LightGBM, no future weather.
3. Anita-style global LightGBM under the same cutoff and no-weather contract.
4. K=1 global portfolio model and one validation-selected clustered model, with
   Zhineng's lag corrected to only information available at the cutoff.
5. The same two tree models with a shared archived weather forecast, if one can
   be sourced.

This separates three questions cleanly: whether global ML beats persistence,
whether clustering helps beyond global ML, and how much actual forecast weather
is worth.

## Submission decision

### If exactly one existing solution must be submitted unchanged

Choose **Nico/root**. It is the most defensible answer to “what could E.ON have
known when placing the day-ahead bid?” and its improvement is broad across
households, not just an aggregate artifact.

### If there is time for a combined presentation but not a full rerun

Use Nico's result as the only forecast-accuracy headline. Add:

- Anita's PV classifier as an exploratory customer-segmentation result;
- Zhineng's price/cost and uncertainty framework as the proposed next layer;
- Mona's newsvendor diagram to explain why asymmetric penalties change the
  optimal bid.

Label all borrowed realized-weather results as oracle experiments, and avoid a
euro-saving headline unless it is recomputed from the submitted forecast.

### If there is time for one substantive correction

Correct Zhineng's forecast features to the common cutoff, remove realized
weather and realized active counts, then compare K=1 with validation-selected K
on Nico's exact test rows. If clustering still wins with a confidence interval
excluding zero, it becomes a credible Level 1 addition. Otherwise submit the
global model and present clustering as a valuable diagnostic rather than a
forecast improvement.

## Bottom line

- **Best forecasting evidence:** Nico.
- **Best PV-identification evidence:** Anita.
- **Best intuitive L0–L3 narrative:** Mona.
- **Best experimental business/uncertainty framework and presentation assets:**
  Zhineng.
- **Best current submission:** Nico's root solution, enriched with carefully
  labelled insights—not headline metrics—from the other three.
