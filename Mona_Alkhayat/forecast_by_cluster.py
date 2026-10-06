# %% [markdown]
# Level 1: does forecasting per cluster beat one global model?
#
# Target  = hourly portfolio load (kWh) for day D, issued at D-1 noon (day-ahead gate closure)
# Global  = one model on the portfolio mean load per household, x number of households
# Cluster = one model per cluster on that cluster's mean load per household, x its households, summed
# Naive   = same hour one week earlier
# Load lags are >= 48 h so they are known at D-1 noon for every hour of D.
# Weather for day D uses MEASURED values as a stand-in for the weather forecast ("oracle" weather):
# real forecast errors would add some error on top, equally for all models.

# %% Config
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"   # the challenge repo's data folder
OUT = HERE / "outputs"
METHOD = os.environ.get("METHOD", "kmedoids")
RES = OUT / METHOD
FC = RES / "forecast"
FC.mkdir(parents=True, exist_ok=True)

TZ = "Europe/Zurich"
TRAIN_START = pd.Timestamp("2021-01-01", tz="UTC")
CUTOFF = pd.Timestamp("2023-09-01", tz="UTC")   # same as clustering: test set starts here
MIN_N = 10                                       # ignore hours where a series has < 10 households
SEED = int(os.environ.get("SEED", 42))


# %% Load cluster series and weather
agg = pd.read_csv(RES / "cluster_hourly.csv")
agg["Timestamp"] = pd.to_datetime(agg["Timestamp"], utc=True)
kwh = agg.pivot(index="Timestamp", columns="cluster", values="kWh_total")
n = agg.pivot(index="Timestamp", columns="cluster", values="n_households")
idx = pd.date_range(kwh.index.min(), kwh.index.max(), freq="h")
kwh, n = kwh.reindex(idx), n.reindex(idx)
clusters = list(kwh.columns)

# portfolio = all clusters together
kwh["all"] = kwh[clusters].sum(axis=1, min_count=len(clusters))
n["all"] = n[clusters].sum(axis=1, min_count=len(clusters))

weather = []
for f in (DATA / "weather_data_hourly").glob("*.csv"):
    w = pd.read_csv(f, sep=";", usecols=["Timestamp", "Temperature_avg_hourly", "Sunshine_duration_hourly"])
    w["Timestamp"] = pd.to_datetime(w["Timestamp"], utc=True)
    weather.append(w)
weather = (pd.concat(weather).groupby("Timestamp").mean()   # mean over stations (all in one region)
             .rename(columns={"Temperature_avg_hourly": "T", "Sunshine_duration_hourly": "sun"})
             .reindex(idx))

print("households reporting at a few dates:")
print(n.loc[n.index.isin(pd.to_datetime(["2021-01-01", "2022-01-01", "2023-01-01", "2023-09-01",
                                          "2024-02-01"], utc=True))])


# %% Features
def make_features(y):
    """y = mean kWh per household for one series (hourly, UTC)."""
    local = idx.tz_convert(TZ)
    X = pd.DataFrame(index=idx)
    X["hour"] = local.hour
    X["dow"] = local.dayofweek
    X["month"] = local.month
    X["lag48"] = y.shift(48)
    X["lag168"] = y.shift(168)
    X["mean_d2"] = y.rolling(24, min_periods=20).mean().shift(48)   # mean of the 24 h ending 48 h ago
    X["T"] = weather["T"]
    X["T_24h"] = weather["T"].rolling(24, min_periods=12).mean()
    X["HDD"] = (15 - X["T_24h"]).clip(lower=0)
    X["sun"] = weather["sun"]
    X["sun_24h"] = weather["sun"].rolling(24, min_periods=12).sum()
    return X


def fit_predict(key):
    y = kwh[key] / n[key]
    X = make_features(y)
    ok = y.notna() & (n[key] >= MIN_N) & X[["lag48", "lag168"]].notna().all(axis=1)
    train = ok & (idx >= TRAIN_START) & (idx < CUTOFF)
    model = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=SEED)
    model.fit(X[train], y[train])
    pred = pd.Series(np.nan, index=idx)
    pred[ok] = model.predict(X[ok])
    naive = X["lag168"]
    # back to kWh using the number of households under contract at that hour
    return pred * n[key], naive * n[key]


preds, naives = {}, {}
for key in ["all"] + clusters:
    preds[key], naives[key] = fit_predict(key)
    print(f"fitted series {key}")


# %% Evaluate on the test period (portfolio total)
test = (idx >= CUTOFF) & kwh["all"].notna() & (n[clusters] >= MIN_N).all(axis=1)
test &= pd.concat([preds[k] for k in ["all"] + clusters] + [naives["all"]], axis=1).notna().all(axis=1)

actual = kwh["all"][test]
fc = pd.DataFrame({
    "Naive (last week)": naives["all"][test],
    "Global model": preds["all"][test],
    "Per-cluster models": sum(preds[c][test] for c in clusters),
})


def metrics(a, f):
    e = f - a
    return pd.Series({"MAE kWh": e.abs().mean(), "RMSE kWh": np.sqrt((e ** 2).mean()),
                      "nMAE %": 100 * e.abs().mean() / a.mean(), "bias %": 100 * e.mean() / a.mean()})


table = fc.apply(lambda f: metrics(actual, f)).T
print(f"\nTest period {actual.index.min():%Y-%m-%d} to {actual.index.max():%Y-%m-%d}, "
      f"{test.sum()} hours, portfolio mean {actual.mean():.1f} kWh/h")
print(table.round(2))
table.to_csv(FC / "portfolio_metrics.csv")

# per-cluster accuracy of each cluster's own model
per_cluster = pd.DataFrame({
    f"cluster {c}": pd.concat([metrics(kwh[c][test], preds[c][test]).rename("model"),
                               metrics(kwh[c][test], naives[c][test]).rename("naive")])
    for c in clusters}).T
print("\nper-cluster errors (model then naive):")
print(per_cluster.round(2))
per_cluster.to_csv(FC / "per_cluster_metrics.csv")

# monthly nMAE: where does the cluster approach help?
month = actual.index.tz_convert(TZ).to_period("M")
monthly = fc.apply(lambda f: (f - actual).abs().groupby(month).mean() / actual.groupby(month).mean() * 100)
print("\nmonthly nMAE %:")
print(monthly.round(2))
monthly.to_csv(FC / "monthly_nmae.csv")


# %% Plots
fig, ax = plt.subplots(figsize=(6, 3.5))
table["nMAE %"].plot.barh(ax=ax, color=["grey", "C0", "C1"])
for i, v in enumerate(table["nMAE %"]):
    ax.text(v, i, f" {v:.1f}%", va="center")
ax.set(xlabel="normalised MAE on portfolio total (%)", title="Day-ahead error, test period")
fig.tight_layout(); fig.savefig(FC / "01_nmae_bars.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 3.5))
monthly[["Global model", "Per-cluster models"]].plot(ax=ax, marker="o", color=["C0", "C1"])
ax.set(ylabel="nMAE (%)", xlabel="", title="Monthly error on portfolio total")
fig.tight_layout(); fig.savefig(FC / "02_monthly_nmae.png", dpi=150)

week = actual.index[(actual.index >= pd.Timestamp("2024-01-08", tz="UTC")) &
                    (actual.index < pd.Timestamp("2024-01-15", tz="UTC"))]
if len(week):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(week.tz_convert(TZ), actual[week], c="k", lw=1.5, label="actual")
    ax.plot(week.tz_convert(TZ), fc.loc[week, "Global model"], c="C0", lw=1, label="global model")
    ax.plot(week.tz_convert(TZ), fc.loc[week, "Per-cluster models"], c="C1", lw=1, label="per-cluster models")
    ax.set(ylabel="portfolio kWh per hour", title="Example test week (January 2024)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FC / "03_example_week.png", dpi=150)

pd.concat([actual.rename("actual"), fc], axis=1).to_csv(FC / "test_forecasts.csv")
print(f"\nsaved to {FC}")
plt.show()
