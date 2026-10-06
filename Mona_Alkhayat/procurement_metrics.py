# %% [markdown]
# Level 2: score the forecasts the way procurement pays for them
#
# E.ON buys the forecast on the day-ahead market at the hourly Swiss day-ahead price p.
# The error is settled afterwards:
#   short (actual > bid): buy the gap at p + SHORT_PREMIUM * |p|
#   long  (actual < bid): sell the surplus at p - LONG_DISCOUNT * |p|
# Imbalance cost = shortfall * SHORT_PREMIUM*|p| + surplus * LONG_DISCOUNT*|p|  (EUR; perfect forecast = 0)
# Prices: Energy-Charts API, bidding zone CH, EUR/MWh (Bundesnetzagentur | SMARD.de, CC BY 4.0).
# The premiums are ASSUMPTIONS (the brief allows this); a sensitivity table shows the ranking holds.

# %% Config
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
METHOD = os.environ.get("METHOD", "kmedoids")
FC = OUT / METHOD / "forecast"
PR = OUT / METHOD / "procurement"
PR.mkdir(parents=True, exist_ok=True)

TZ = "Europe/Zurich"
SHORT_PREMIUM = 0.5    # being short costs 50% above day-ahead
LONG_DISCOUNT = 0.3    # surplus sold 30% below day-ahead
PEAK_HOURS = range(17, 21)  # local evening peak
SCALE_HOUSEHOLDS = 100_000


# %% Load forecasts, household counts and prices
fc = pd.read_csv(FC / "test_forecasts.csv", index_col=0)
fc.index = pd.to_datetime(fc.index, utc=True)
models = [c for c in fc.columns if c != "actual"]

agg = pd.read_csv(OUT / METHOD / "cluster_hourly.csv")
agg["Timestamp"] = pd.to_datetime(agg["Timestamp"], utc=True)
n_hh = agg.groupby("Timestamp")["n_households"].sum().reindex(fc.index)

PRICE_FILE = OUT / "ch_da_prices.json"
if not PRICE_FILE.exists():   # hourly Swiss day-ahead prices, Energy-Charts API (Bundesnetzagentur | SMARD.de, CC BY 4.0)
    import urllib.request
    urllib.request.urlretrieve("https://api.energy-charts.info/price?bzn=CH&start=2023-08-25&end=2024-03-01", PRICE_FILE)
d = json.load(open(PRICE_FILE))
price = pd.Series(d["price"], index=pd.to_datetime(d["unix_seconds"], unit="s", utc=True)).reindex(fc.index)
print(f"{len(fc)} test hours, {price.isna().sum()} without price; mean price {price.mean():.1f} EUR/MWh")
fc, price, n_hh = fc[price.notna()], price.dropna(), n_hh[price.notna()]

local = fc.index.tz_convert(TZ)
actual = fc["actual"]
years = (fc.index.max() - fc.index.min()).total_seconds() / (365.25 * 86400)
household_years = n_hh.mean() * years


# %% Metrics
def imbalance_cost(a, f, p, short=SHORT_PREMIUM, long=LONG_DISCOUNT):
    """EUR per hour; kWh -> MWh."""
    shortfall = (a - f).clip(lower=0) / 1000
    surplus = (f - a).clip(lower=0) / 1000
    return shortfall * short * p.abs() + surplus * long * p.abs()


def score(f):
    e = f - actual
    cost = imbalance_cost(actual, f, price)
    daily_err = e.groupby(local.date).sum().abs()
    peak = local.hour.isin(PEAK_HOURS)
    return pd.Series({
        # technical
        "MAE (kWh/h)": e.abs().mean(),
        "RMSE (kWh/h)": np.sqrt((e ** 2).mean()),
        "nMAE (%)": 100 * e.abs().mean() / actual.mean(),
        # shape of error
        "bias (%)": 100 * e.mean() / actual.mean(),
        "share of error that is short (%)": 100 * (-e).clip(lower=0).sum() / e.abs().sum(),
        "peak-hour nMAE 17-21h (%)": 100 * e[peak].abs().mean() / actual[peak].mean(),
        "P95 hourly error (kWh)": e.abs().quantile(0.95),
        "worst day energy error (kWh)": daily_err.max(),
        # business
        "imbalance cost (EUR)": cost.sum(),
        "EUR per household-year": cost.sum() / household_years,
        "% of energy bill": 100 * cost.sum() / (actual / 1000 * price).sum(),
    })


table = pd.DataFrame({m: score(fc[m]) for m in models}).T
naive_cost = table.loc["Naive (last week)", "imbalance cost (EUR)"]
table["saving vs naive (%)"] = 100 * (1 - table["imbalance cost (EUR)"] / naive_cost)
table[f"EUR per year for {SCALE_HOUSEHOLDS:,} households"] = table["EUR per household-year"] * SCALE_HOUSEHOLDS
pd.set_option("display.width", 200)
print(table.T.round(2))
table.T.to_csv(PR / "procurement_metrics.csv")


# %% Sensitivity: does the ranking survive other penalty assumptions?
rows = []
for s in [0.25, 0.5, 1.0]:
    for l in [0.1, 0.3, 0.5]:
        c = {m: imbalance_cost(actual, fc[m], price, s, l).sum() / household_years for m in models}
        rows.append({"short premium": s, "long discount": l, **c,
                     "best": min(c, key=c.get)})
sens = pd.DataFrame(rows)
print("\nEUR per household-year under different penalty assumptions:")
print(sens.round(2).to_string(index=False))
sens.to_csv(PR / "sensitivity.csv", index=False)


# %% Monthly cost of forecast error
month = local.to_period("M")
monthly = pd.DataFrame({m: imbalance_cost(actual, fc[m], price).groupby(month).sum() for m in models})
monthly.index = monthly.index.astype(str)
print("\nmonthly imbalance cost (EUR):")
print(monthly.round(1))
monthly.to_csv(PR / "monthly_cost.csv")

colors = {"Naive (last week)": "grey", "Global model": "C0", "Per-cluster models": "C1"}
fig, ax = plt.subplots(figsize=(8, 4))
monthly.plot.bar(ax=ax, color=[colors[m] for m in models], width=0.8)
ax.set(ylabel="imbalance cost (EUR)", xlabel="", title="Cost of forecast error per month, 390 households")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout(); fig.savefig(PR / "01_monthly_cost.png", dpi=150)


# %% When is error expensive? cost by hour of day
by_hour = pd.DataFrame({m: imbalance_cost(actual, fc[m], price).groupby(local.hour).sum() for m in models})
fig, ax = plt.subplots(figsize=(8, 4))
for m in models:
    ax.plot(by_hour.index, by_hour[m], marker=".", color=colors[m], label=m)
ax2 = ax.twinx()
ax2.plot(price.groupby(local.hour).mean(), color="k", ls=":", label="mean day-ahead price")
ax2.set_ylabel("mean day-ahead price (EUR/MWh)")
ax.set(xlabel="hour of day (local)", ylabel="imbalance cost (EUR)", title="Where the error money goes",
       xticks=range(0, 24, 3))
ax.legend(loc="upper left"); ax2.legend(loc="upper right")
fig.tight_layout(); fig.savefig(PR / "02_cost_by_hour.png", dpi=150)

print(f"\nsaved to {PR}")
plt.show()
