# %% [markdown]
# Level 3: use forecast uncertainty to choose the bid (newsvendor)
#
# Expected imbalance cost E[C] = pi_s*|p|*E[(y-b)+] + pi_l*|p|*E[(b-y)+] is minimised by bidding the
# tau* = pi_s / (pi_s + pi_l) quantile of the demand distribution, not the mean.
# Uncertainty is estimated from the model's own recent errors, using only information known at noon D-1:
# relative errors e = (actual - forecast) / forecast over the 28 days ending on D-2.
# Bid for each hour of D = forecast * (1 + quantile_tau(e)).

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
L3 = OUT / METHOD / "level3"
L3.mkdir(parents=True, exist_ok=True)

TZ = "Europe/Zurich"
MODEL = "Per-cluster models"
SHORT_PREMIUM, LONG_DISCOUNT = 0.5, 0.3      # same assumptions as Level 2
WINDOW_DAYS = 28
TAUS = np.round(np.arange(0.30, 0.86, 0.025), 3)


# %% Data
fc = pd.read_csv(FC / "test_forecasts.csv", index_col=0)
fc.index = pd.to_datetime(fc.index, utc=True)
d = json.load(open(OUT / "ch_da_prices.json"))
price = pd.Series(d["price"], index=pd.to_datetime(d["unix_seconds"], unit="s", utc=True)).reindex(fc.index)

actual, f = fc["actual"], fc[MODEL]
rel_err = (actual - f) / f
day = pd.Series(fc.index.tz_convert(TZ).date, index=fc.index)
days = sorted(day.unique())


def cost(bid, short=SHORT_PREMIUM, long=LONG_DISCOUNT):
    shortfall = (actual - bid).clip(lower=0) / 1000
    surplus = (bid - actual).clip(lower=0) / 1000
    return shortfall * short * price.abs() + surplus * long * price.abs()


# %% Rolling error quantiles: for day D use errors from days D-29 ... D-2 (known at noon D-1)
eval_days = [dd for dd in days if (pd.Timestamp(dd) - pd.Timestamp(days[0])).days >= WINDOW_DAYS + 1]
q = {}  # (day, tau) -> quantile of relative error
for dd in eval_days:
    lo, hi = pd.Timestamp(dd) - pd.Timedelta(days=WINDOW_DAYS + 1), pd.Timestamp(dd) - pd.Timedelta(days=1)
    past = rel_err[(day >= lo.date()) & (day < hi.date())].dropna()
    for t in TAUS:
        q[(dd, t)] = past.quantile(t)

mask = day.isin(eval_days)
print(f"evaluation: {eval_days[0]} to {eval_days[-1]} ({mask.sum()} hours); first {WINDOW_DAYS + 1} days "
      "used only to estimate uncertainty")


def bid_for(tau):
    shift = day.map({dd: q[(dd, tau)] for dd in eval_days})
    return f * (1 + shift)


hh_years = None  # cost reported for the 390-home portfolio over the evaluation period
mean_cost = cost(f)[mask].sum()
curve = pd.Series({t: cost(bid_for(t))[mask].sum() for t in TAUS})
tau_star = SHORT_PREMIUM / (SHORT_PREMIUM + LONG_DISCOUNT)
tau_star_grid = TAUS[np.argmin(np.abs(TAUS - tau_star))]
best_tau = curve.idxmin()

res = pd.Series({
    "cost, bid = mean forecast (EUR)": mean_cost,
    "cost, bid = median (tau 0.5) (EUR)": curve[0.5],
    f"cost, bid = newsvendor tau* {tau_star:.3f} (EUR)": curve[tau_star_grid],
    f"cost, best tau in hindsight {best_tau} (EUR)": curve[best_tau],
    "saving of newsvendor vs mean (%)": 100 * (1 - curve[tau_star_grid] / mean_cost),
    "share of hours short, mean bid (%)": 100 * (actual > f)[mask].mean(),
    "share of hours short, newsvendor bid (%)": 100 * (actual > bid_for(tau_star_grid))[mask].mean(),
    "extra energy bought vs mean bid (%)": 100 * (bid_for(tau_star_grid)[mask].sum() / f[mask].sum() - 1),
})
print(res.round(2).to_string())
res.to_csv(L3 / "newsvendor_result.csv")
curve.rename("imbalance cost (EUR)").to_csv(L3 / "cost_vs_tau.csv")


# %% Does tau* follow the cost asymmetry? (other penalty assumptions)
rows = []
for s_, l_ in [(0.25, 0.5), (0.3, 0.3), (0.5, 0.3), (1.0, 0.3), (1.0, 0.1)]:
    c = pd.Series({t: cost(bid_for(t), s_, l_)[mask].sum() for t in TAUS})
    ts = s_ / (s_ + l_)
    tg = TAUS[np.argmin(np.abs(TAUS - ts))]
    m = cost(f, s_, l_)[mask].sum()
    rows.append({"short premium": s_, "long discount": l_, "tau*": round(ts, 3), "best tau": c.idxmin(),
                 "saving tau* vs mean (%)": round(100 * (1 - c[tg] / m), 1)})
sens = pd.DataFrame(rows)
print("\n", sens.to_string(index=False))
sens.to_csv(L3 / "sensitivity.csv", index=False)


# %% Plot: cost against bid quantile
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(curve.index, curve.values, marker="o", color="C1", label="bid = quantile tau of forecast")
ax.axhline(mean_cost, color="C0", ls="--", label="bid = mean forecast")
ax.axvline(tau_star, color="grey", ls=":", label=f"newsvendor tau* = {tau_star:.3f}")
ax.set(xlabel="bid quantile tau", ylabel="imbalance cost (EUR)", title="Bidding above the mean pays")
ax.legend()
fig.tight_layout(); fig.savefig(L3 / "01_cost_vs_tau.png", dpi=150)
print(f"\nsaved to {L3}")


# %% Full chain on the same evaluation window: naive bid -> global -> per-cluster -> per-cluster + newsvendor
agg = pd.read_csv(OUT / METHOD / "cluster_hourly.csv")
agg["Timestamp"] = pd.to_datetime(agg["Timestamp"], utc=True)
n_hh = agg.groupby("Timestamp")["n_households"].sum().reindex(fc.index)[mask]
hh_years = n_hh.mean() * mask.sum() / 8766
chain = pd.Series({
    "L0 naive (last week)": cost(fc["Naive (last week)"])[mask].sum(),
    "L0 global model": cost(fc["Global model"])[mask].sum(),
    "L1 per-cluster models": mean_cost,
    "L3 per-cluster + newsvendor bid": curve[tau_star_grid],
}) / hh_years
print("\nEUR per household per year, evaluation window:")
print(chain.round(2).to_string())
chain.rename("EUR per household-year").to_csv(L3 / "full_chain.csv")
