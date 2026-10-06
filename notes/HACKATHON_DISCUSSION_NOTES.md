# Bringing the Heat — Discussion Notes

## 1. Challenge in one sentence

Build a leakage-safe day-ahead forecast of household or portfolio grid demand, using historical smart-meter, weather, and household data, and translate forecast quality and uncertainty into a procurement-relevant decision.

## 2. What the data supports

- 410 heat-pump households, mostly with more than one year of observations.
- Smart-meter readings at 15-minute resolution (up to 96 values per day), with unequal date coverage between households.
- Total grid energy received is the broadly available target; heat-pump and other-load components are not available for every household.
- Household metadata is available for 393 households and includes a PV-ownership flag, although this flag is missing for some households.
- Each household maps to one of eight hourly weather-station datasets.
- The intervention fields (`Group` and `AffectsTimePoint`) may represent a structural change after heat-pump optimisation and should not be treated as ordinary static features.

### Important interpretation limit

The smart-meter files contain neither PV generation nor energy exported to the grid. Therefore, the project cannot directly forecast PV production. At most, it can forecast **grid import** and study whether the import profile contains a detectable PV signature. This distinction should be stated clearly in the presentation.

## 3. Recommended hackathon scope

Aim for one complete, defensible pipeline rather than many unfinished models:

1. Forecast the next day's 24 hourly grid-import values, initially at household level or directly at portfolio level.
2. Establish strong seasonal-naive baselines: same hour yesterday and same hour one week ago.
3. Train one practical tabular model using lag, calendar, rolling-demand, weather, and household features.
4. Evaluate with chronological rolling windows and compare against the baselines.
5. Convert errors into a simple procurement-cost simulation and, if time permits, prediction intervals.

Hourly aggregation is a sensible first choice: it matches the weather resolution, reduces meter noise and missing intervals, and produces a result that is easier to finish and explain. Retain 15-minute forecasting only if it is explicitly valuable and the hourly pipeline is already complete.

## 4. Suggested modelling setup

### Target

- Primary: `kWh_received_Total` for each hour of day D+1.
- Business-aligned alternative: aggregate households first and forecast the portfolio load. Aggregation usually smooths household-specific noise and is closer to procurement needs.
- Useful comparison: household forecasts summed to a portfolio versus a model trained directly on the portfolio.

### Features available before the prediction deadline

- Calendar: hour, weekday, weekend, month, season, and public-holiday indicator if external data is permitted.
- Demand lags: 24 hours, 48 hours, and 168 hours; rolling means/dispersion computed only from past values.
- Weather: temperature first, then humidity, sunshine, precipitation, and wind if they add measurable value.
- Household: PV flag, building/living-area attributes, heat-pump type, and group, with explicit missing-value handling.
- Regime: before/after intervention, used carefully because post-visit behaviour may differ from pre-visit behaviour.

Candidate models, in priority order:

1. Seasonal-naive baselines.
2. Regularised linear regression as an interpretable benchmark.
3. Histogram gradient boosting or random forest as the main nonlinear model.
4. Quantile gradient boosting for uncertainty, only after the point-forecast pipeline works.

Deep learning is probably unnecessary for a one-day event unless the team already has a reusable time-series pipeline.

## 5. Evaluation design

Random train/test splitting is invalid. Use one or more chronological forecast origins:

- Train on the past, validate on a later contiguous period, and reserve the latest period for the final test.
- Prefer rolling-origin backtesting to a single split so that different seasons and weather conditions are represented.
- Fit preprocessing, imputation, clustering, and feature selection separately inside each training fold.
- Ensure every lag and rolling statistic is shifted so that no target-day observation enters its own prediction.

Report at least:

- **MAE:** easy to explain in kWh and robust to occasional spikes.
- **RMSE:** highlights costly large errors.
- **WAPE:** portfolio-friendly relative error; avoid MAPE because near-zero intervals make it unstable.
- **Peak-period MAE:** useful if procurement errors at high-load hours matter more.
- **Bias:** systematic under- or over-purchasing can matter even when average absolute error looks good.

Always report improvement over the seasonal-naive baseline, not only the model's standalone score. Show errors by hour of day, season, PV flag, and before/after intervention.

## 6. Procurement and uncertainty story

Forecast accuracy is not the final decision objective. A simple business layer can make the solution much stronger:

$$
\text{cost} = c_{DA}\,q + c_{under}\max(y-q,0) + c_{over}\max(q-y,0)
$$

where `q` is the day-ahead purchase and `y` is realised demand. Since actual market prices are not supplied, use clearly labelled scenarios rather than presenting assumed prices as facts.

If under-purchasing is more expensive than over-purchasing, the optimal bid is generally above the median forecast. Quantile forecasts make this explicit: select the quantile implied by the chosen under/over-cost ratio, then compare procurement cost with a point-forecast bid.

For uncertainty, evaluate interval coverage as well as width. A narrow interval is not useful if it frequently misses the realised demand.

## 7. Main risks and safeguards

- **Target ambiguity:** “energy consumption” in the challenge may actually be measured grid import. Use the exact measurement name in technical claims.
- **Weather leakage:** historical realised weather is not necessarily available at day-ahead bidding time. Ideally use weather forecasts; if unavailable, describe realised weather as an optimistic proxy and run a no-weather comparison.
- **UTC and daylight saving:** timestamps are in UTC. Decide whether calendar features should represent local time, and handle DST explicitly when converting.
- **Incomplete intervals:** distinguish a true zero from a missing 15-minute reading; check daily completeness before aggregation.
- **Unequal histories:** use a common evaluation period where possible and avoid letting long-history households dominate every metric.
- **Intervention shift:** do not casually mix pre- and post-optimisation records. Compare regimes or train on the regime that matches deployment.
- **PV labels:** missing PV status is “unknown,” not “no PV.” Any attempt to infer PV ownership needs a time-based evaluation and should be presented as a secondary analysis.
- **Household leakage:** if the goal includes generalising to unseen households, add a group-by-household holdout. A time split alone only tests future prediction for known households.

## 8. High-value analyses and visuals

- Coverage and missingness by household and date.
- Average daily load curve split by season and PV flag.
- Temperature versus load, especially for heat-pump demand.
- Portfolio actual versus forecast over the final test week.
- Error by hour and forecast horizon.
- Cost versus chosen forecast quantile under two or three transparent price scenarios.
- A compact ablation table: baseline, lags/calendar, plus weather, plus metadata.

## 9. Questions to settle with mentors early

- Is the desired deliverable household-level demand, aggregate portfolio demand, or both?
- Is the operational forecast horizon exactly the next calendar day, and at what bidding cutoff time?
- Should evaluation use 15-minute or hourly values?
- Are external holiday data and forecast weather allowed?
- What do `kWh_received_Total` and the intervention labels mean operationally?
- Is there an official scoring metric or a preferred asymmetric cost assumption?

## 10. Four-person parallel work plan

The work split should follow the team's expertise and the natural chain from data to forecasts to uncertainty to procurement decisions. Start with a short joint alignment session to freeze the target, forecast horizon, time split, market assumptions, and table interfaces.

| Owner | Best-fit responsibility | Concrete output | Main handoff |
|---|---|---|---|
| **You — ML lead** | Own the leakage-safe forecasting pipeline: feature engineering, seasonal-naive baselines, main ML model, temporal backtesting, and standard predictive metrics | Point forecasts, residuals, feature/ablation evidence, and reproducible training code | Forecasts and out-of-sample residuals to the UQ specialist |
| **Optimization and UQ specialist** | Quantify forecast uncertainty and turn it into decision inputs: quantile or conformal intervals, calibration checks, scenario generation, and an uncertainty-aware bidding rule | Calibrated intervals/quantiles, demand scenarios, coverage results, and comparison with deterministic decisions | Scenarios or quantiles to the power-system optimizer; cost sensitivities back to ML |
| **Power-system optimization specialist** | Define the operational problem: day-ahead versus imbalance assumptions, objective and constraints, deterministic/stochastic procurement model, and power-system plausibility checks | Procurement formulation, baseline buying policy, optimized bids, cost and feasibility results, and clearly stated market assumptions | Required forecast/scenario format to ML and UQ; business results to the demo |
| **Fourth teammate — data and integration lead** | Own ingestion, data-quality checks, hourly aggregation, weather/metadata joins, shared tables, experiment tracking, and final notebook/presentation integration | Canonical modelling dataset, EDA plots, one-command or one-notebook demo, consolidated figures, and presentation backup | Clean data to all specialists and final artifacts from all workstreams |

If the fourth teammate has a stronger specialist profile, preserve ownership of the shared data pipeline but pair them with the closest workstream—for example, visualization/presentation, software engineering, or domain research.

### How the specialist work connects

The core flow should be:

`clean data → ML forecasts and residuals → calibrated uncertainty/scenarios → optimized procurement bid → realised-cost evaluation`

The split between the two optimization-oriented specialists should remain explicit:

- The **optimization/UQ specialist** models uncertainty and how it propagates into decisions.
- The **power-system specialist** decides what is optimized, under which operational assumptions and constraints.
- The **ML lead** improves predictive signal but should use downstream cost feedback to choose between models, rather than optimizing only RMSE.
- The **integration lead** protects reproducibility and prevents each workstream from constructing a different version of the data.

### Shared interface to agree first

All workstreams should exchange a small number of stable tables:

- **Model input:** `timestamp`, `Household_ID` or portfolio ID, target, and permitted features.
- **Predictions:** `forecast_origin`, `target_timestamp`, optional entity ID, `actual`, `prediction`, model name, and optional lower/upper quantiles.
- **Scenarios:** scenario ID, probability or weight, target timestamp, and simulated demand.
- **Procurement decisions:** target timestamp, purchased quantity, strategy name, assumed prices, realised imbalance, and total cost.
- **Evaluation:** one row per model and test slice with MAE, RMSE, WAPE, bias, interval coverage, and simulated cost.

This contract lets modelling, evaluation, and presentation proceed before the full data pipeline is finished.

### Suggested checkpoints during the day

1. **Kickoff:** agree on the primary portfolio/hourly scope, one fixed temporal holdout, naming conventions, and the minimum viable demo.
2. **Interface test:** the integration lead supplies a small clean table; the ML lead emits baseline forecasts; the UQ specialist creates provisional scenarios from residuals; the power-system specialist consumes them in a minimal cost model.
3. **Midpoint decision:** retain one point model and one uncertainty method. Verify that the optimization comparison answers a meaningful business question before attempting stretch work.
4. **Results freeze:** rerun the full chain on the agreed test period and export all predictions, decisions, metrics, and plots from one configuration.
5. **Final rehearsal:** each specialist explains their link in the chain; the integration lead controls the narrative and keeps static results as a live-demo backup.

### Integration rules

- Keep one canonical preprocessing path; do not let each model clean the data differently.
- Commit small, scoped changes and avoid editing the same notebook simultaneously. Prefer separate modules or notebooks per workstream and one integration notebook.
- Save predictions and metrics in the agreed schema so experiments remain comparable.
- Record random seeds, split dates, feature availability assumptions, and model parameters.
- Assign every optional idea a cutoff time. If it misses the cutoff, present it as future work rather than destabilising the core demo.

### Minimum viable fallback

If time runs short, converge on: hourly portfolio data, weekly seasonal-naive versus one gradient-boosting model, empirical residual quantiles for uncertainty, and a small procurement problem comparing point-forecast purchasing with one uncertainty-aware policy. Use a single chronological test period, MAE/WAPE/bias, interval coverage, realised cost, and one end-to-end figure. PV classification, household clustering, 15-minute forecasts, and elaborate stochastic optimization can remain stretch goals.

## 11. Definition of a convincing final demo

A strong submission can be modest in scope if it shows:

- a reproducible data pipeline with explicit missing-data rules;
- leakage-safe chronological evaluation;
- a seasonal-naive baseline and one improved model;
- interpretable feature or ablation evidence;
- an honest statement of what can and cannot be inferred about PV;
- a portfolio forecast chart and a simple procurement-cost comparison;
- limitations and the next experiment the team would run with more time.

The central story should be: **better forecasting is useful, but a trustworthy procurement decision requires correct timing assumptions, calibrated uncertainty, and costs that distinguish over- from under-purchasing.**
