# Bringing the Heat: from forecast to cost-optimal day-ahead bid

E.ON challenge, RWTH hackathon. Team: Mona Alkhayat, [add all team member names].
The one-page description is `Description_BringingTheHeat.pdf`.

## How to run

1. This folder sits inside the challenge repository, next to its `data/` folder:
   ```
   rwth_hackathon/
   ├── data/            (from the challenge repository)
   └── Mona_Alkhayat/   (this folder)
   ```
2. Install: `pip install -r requirements.txt` (Python ≥ 3.11).
3. Run in this order from inside this folder:

| Step | Script | Level | What it does | Main outputs in `outputs/` |
|---|---|---|---|---|
| 1 | `cluster_households.py` | L1 | 15-min → hourly; 52 features per home from data before 1 Sep 2023; k-medoids (or k-means), k by silhouette; checks clusters against the survey PV flag | `kmedoids/household_clusters.csv`, `kmedoids/cluster_hourly.csv` |
| 2 | `forecast_by_cluster.py` | L0, L1 | Day-ahead forecasts (noon D−1, load lags ≥ 48 h): naive, one global model, one model per cluster | `kmedoids/forecast/portfolio_metrics.csv`, `test_forecasts.csv` |
| 3 | `procurement_metrics.py` | L2 | Downloads hourly Swiss day-ahead prices; cost of error and other procurement metrics; 9 penalty scenarios | `kmedoids/procurement/procurement_metrics.csv`, `sensitivity.csv` |
| 4 | `level3_bidding.py` | L3 | Rolling 28-day error quantiles; cost against bid quantile τ; newsvendor bid τ* = 0.625 | `kmedoids/level3/newsvendor_result.csv`, `cost_vs_tau.csv`, `full_chain.csv` |
| 5 | `make_final_plots.py` | all | One figure per level | `final_plots/Level0…Level3*.png` |
| opt. | `compare_clustering.py` | L1 | No clustering vs k-means vs k-medoids (run steps 1–3 with `METHOD=kmeans` first) | `comparison/` |

Settings via environment variables: `METHOD` = `kmedoids` (default) or `kmeans`; `SEED` (default 42).
The first run of step 1 reads 1.7 GB of CSVs (about 3 minutes) and caches them in `outputs/hourly_total.parquet`.
The 5-seed nMAE values used in the plots come from running step 2 with `SEED` = 42, 1, 2, 3, 4.

## Design in short

- **Target:** hourly grid imports of the whole portfolio (390 homes), which is what E.ON buys.
- **Timing:** each forecast for day D is made at noon on D−1, so load inputs are at least 48 h old.
- **Split:** time-ordered; train Jan 2021 – Aug 2023, test Sep 2023 – Feb 2024 (last 24% of the data).
- **Accuracy:** nMAE = mean absolute hourly error ÷ mean actual load in the test period.
- **Cost of error:** shortfall × 0.5·|p| + surplus × 0.3·|p|, with p the real hourly Swiss day-ahead price
  (Energy-Charts API, Bundesnetzagentur | SMARD.de, CC BY 4.0); the penalties are assumptions.
- **Methods:** gradient-boosted trees (supervised) for forecasting, k-medoids (unsupervised) for grouping,
  newsvendor quantile for the bid. No neural networks.

## Results (test period)

| Level | Result |
|---|---|
| L0 | nMAE 20.1% (naive) → 6.24% (model, mean of 5 runs) |
| L1 | k = 2; PV-pattern group of 70 homes, 100% precision vs survey (recall 52%); k-means agrees on 96%; model per group nMAE 5.96% (−4.6% relative, 5 of 5 runs) |
| L2 | Cost of error €92.6 → €24.9 per home per year (−73%); clustering changes it by only −0.2%; the better model flips with the penalties |
| L3 | τ* = 0.625 equals the cheapest quantile in hindsight; −2.1% vs bidding the mean (−17% if short costs +100%) |

## Assumptions and limitations

Measured weather stands in for a weather forecast; the data has no PV generation or export; the homes are assumed
to be in Switzerland (Swiss prices, Europe/Zurich time); imbalance penalties are assumed; one portfolio-wide quantile.
