# Clustering + cluster-based day-ahead forecasting

**Question:** Does clustering households before forecasting improve day-ahead portfolio load forecasts, and does it save procurement cost?

Pipeline: households → clusters → mean load per active household in each cluster → one forecast per cluster
→ multiply by the number of active households → sum to the portfolio forecast.
K = 1 (one model on the whole portfolio) is the "no clustering" baseline.

## Data split (80/20 by time span)

| Part | Period | Used for |
|---|---|---|
| Sub-train | until 2022-09-15 | Selection stage: clustering + model fit (224 eligible households) |
| Validation | 2022-09-15 to 2023-03-15 | **Choosing K only** |
| Train (80%) | until 2023-03-15 | Final clustering + model refit (328 eligible households) |
| Test (20%) | 2023-03-15 to 2024-02-27 | **Final report only**; no decision uses it |

K rule: the smallest K (K=1 included) whose validation imbalance cost is within 0.5% of the lowest cost.

## Run

```powershell
cd cluster_forecast
..\.venv\Scripts\python.exe run_all.py           # ~4 min; prices and preprocessing are cached (--fresh rebuilds)
..\.venv\Scripts\python.exe 05_interpret.py 3    # interpretation figures for a specific K
```

| Step | What it does |
|---|---|
| `00_fetch_prices.py` | Real hourly DE-LU day-ahead prices from SMARD (Bundesnetzagentur) |
| `01_preprocess.py` | 15-min → hourly kWh (hours with ≥3/4 quarter-hours kept); hourly weather |
| `02_cluster.py` | Final clustering on training data (< 2023-03-15): 9 load features → k-means, K = 2–6 |
| `03_select_k.py` | Selection stage: re-cluster on sub-train, forecast the validation window, choose K by cost |
| `04_forecast.py` | Final test: refit on the 80% training data, forecast the 20% test set for K = 1–6 |
| `05_interpret.py` | Load profiles, feature heatmap, PV share and metadata per cluster |
| `06_k_selection_plots.py` | Elbow, silhouette, validation cost, cluster sizes, silhouette diagrams |
| `07_level2_business.py` | Level 2: imbalance cost with real prices, bootstrap CI, sensitivity, German price figure |
| `08_level3_uncertainty.py` | Level 3: rolling residual quantiles, calibration, where it is uncertain, newsvendor bids, penalty sensitivity |
| `09_slide_figures.py` | Single-panel, large-font figures for the slides (`outputs/figures/slides/`) |
| `10_build_pptx.py` | 5-slide, 3-minute deck with speaker notes → `outputs/clustering_story.pptx` (needs `python-pptx`) |

Shared code is in `common.py` and settings are in `config.py`.

## Level 2 cost model

- The day-ahead forecast is bought at the **real** DE-LU day-ahead price p.
- Shortfall is bought intraday at p + 40 + 25%·|p| EUR/MWh; surplus is sold at p − 30 − 25%·|p|. These penalties are **assumed** and set in `config.py`.
  - The proportional part makes errors in expensive hours cost more.
  - Using |p| keeps the rule valid in negative-price hours.
  - A purely fixed surcharge would cancel the day-ahead price out of the comparison entirely.
- Imbalance cost is the extra cost compared with a perfect forecast.
- Metrics: imbalance cost (EUR per household per year, % of the day-ahead bill), shortfall and surplus energy, P95 shortfall, evening-peak MAE, price-weighted MAE, daily energy error, nMAE and bias.
- Significance: weekly block bootstrap (2000 resamples) of the daily cost difference vs the baseline.

## Level 3: uncertainty and bidding

- Uncertainty is estimated at the portfolio level, because cluster quantiles are not additive.
  - Relative residual r = (actual − forecast) / forecast.
  - For day D and local hour h: empirical quantiles (5%–95%) of r over the previous 56 days, pooling hours h ± 1. All of these residuals are known by midnight of D-1.
- Evaluation runs from 2023-05-10, after an 8-week warm-up of the test set.
- Bid rule (newsvendor): bid the τ\* quantile, where τ\* = c_short / (c_short + c_surplus). c_short and c_surplus are computed from **yesterday's** same-hour DA price, because today's price is unknown when bidding. Costs are settled at today's real price.
- Compared bidding strategies: point forecast, median (bias-corrected), τ\* quantile; each for the baseline and the selected K.

## Other assumptions

- Households need ≥ 90 complete days before the respective training cut-off.
- Clustering features: temperature sensitivity (`temp_slope_rel`, `temp_r2`), base-load share (`summer_winter_ratio`), PV signature (`pv_midday_ratio`, `pv_sun_corr`) and energy shares by time of day. Absolute household size is excluded.
- The PV label and metadata are **not** clustering inputs; they are used only to validate and interpret the clusters.
- Forecast features: load lags ≥ 24h (data up to midnight of D-1); local calendar; temperature and sunshine on day D. Observed weather is used as a perfect weather forecast.
- Ignored: German public holidays and the heat-pump optimisation visit (`AffectsTimePoint`).
- Results of the earlier version (split at 2023-03-01, synthetic prices) are kept in `outputs_v1/`.
