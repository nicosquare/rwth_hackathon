# %% [markdown]
# Without clustering vs k-means vs k-medoids: forecast error and cost of error on the same test period.
# Needs forecast_by_cluster.py and procurement_metrics.py run for METHOD=kmeans and METHOD=kmedoids.

# %% Config
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
CMP = OUT / "comparison"
CMP.mkdir(exist_ok=True)

# portfolio nMAE (%) from 5 runs with different random seeds (42, 1, 2, 3, 4), copied from the run logs
SEEDS = {
    "No clustering\n(global model)": [6.12, 6.20, 6.34, 6.28, 6.27],
    "k-means\n(2 clusters)": [6.00, 5.95, 6.08, 5.96, 5.89],
    "k-medoids\n(2 clusters)": [6.01, 5.97, 5.99, 5.94, 5.87],
}
COLORS = ["C0", "C2", "C1"]

# %% Load seed-42 outputs for both methods
eur, monthly = {}, {}
for method, label in [("kmeans", "k-means\n(2 clusters)"), ("kmedoids", "k-medoids\n(2 clusters)")]:
    m = pd.read_csv(OUT / method / "procurement" / "procurement_metrics.csv", index_col=0)
    eur["Naive\n(last week)"] = m.loc["EUR per household-year", "Naive (last week)"]
    eur["No clustering\n(global model)"] = m.loc["EUR per household-year", "Global model"]
    eur[label] = m.loc["EUR per household-year", "Per-cluster models"]
    mn = pd.read_csv(OUT / method / "forecast" / "monthly_nmae.csv", index_col=0)
    monthly["No clustering (global model)"] = mn["Global model"]
    monthly[label.replace("\n", " ")] = mn["Per-cluster models"]
eur = pd.Series(eur)[["Naive\n(last week)", "No clustering\n(global model)", "k-means\n(2 clusters)",
                      "k-medoids\n(2 clusters)"]]
monthly = pd.DataFrame(monthly)
monthly.index = [pd.Period(p).strftime("%b %y") for p in monthly.index]

means = {k: np.mean(v) for k, v in SEEDS.items()}
base = means["No clustering\n(global model)"]
summary = pd.DataFrame({
    "nMAE mean of 5 runs (%)": means,
    "nMAE min (%)": {k: min(v) for k, v in SEEDS.items()},
    "nMAE max (%)": {k: max(v) for k, v in SEEDS.items()},
    "error reduction vs no clustering (%)": {k: 100 * (1 - v / base) for k, v in means.items()},
    "runs better than no clustering": {k: sum(a < b for a, b in zip(v, SEEDS["No clustering\n(global model)"]))
                                       for k, v in SEEDS.items()},
})
summary["EUR per home per year (seed 42)"] = eur
summary["cost reduction vs no clustering (%)"] = 100 * (1 - eur / eur["No clustering\n(global model)"])
summary.index = [i.replace("\n", " ") for i in summary.index]
print(summary.round(2).to_string())
summary.to_csv(CMP / "clustering_comparison.csv")

# %% Figure: three panels
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1.15, 1.3]})

# (a) forecast error, mean and range over 5 seeds
ax = axes[0]
x = np.arange(len(SEEDS))
vals = np.array(list(means.values()))
lo = vals - np.array([min(v) for v in SEEDS.values()])
hi = np.array([max(v) for v in SEEDS.values()]) - vals
ax.bar(x, vals, color=COLORS, width=0.6, yerr=[lo, hi], capsize=6, ecolor="#555555")
for i, (k, v) in enumerate(means.items()):
    lab = f"{v:.2f}%" if i == 0 else f"{v:.2f}%\n(−{100 * (1 - v / base):.1f}%)"
    ax.text(i, v + hi[i] + 0.03, lab, ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks(x, list(SEEDS))
ax.set_ylim(5.5, 6.65)
ax.set_ylabel("portfolio nMAE (%)  — axis starts at 5.5")
ax.set_title("(a) Forecast error, mean and range of 5 runs", loc="left", fontsize=12)

# (b) cost of error in euros
ax = axes[1]
x = np.arange(len(eur))
ax.bar(x, eur.values, color=["grey"] + COLORS, width=0.6)
for i, v in enumerate(eur.values):
    ax.text(i, v + 1.5, f"€{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks(x, list(eur.index))
ax.set_ylim(0, eur.max() * 1.15)
ax.set_ylabel("imbalance cost, € per home per year")
ax.set_title("(b) Cost of forecast error (Swiss prices, +50% / −30%)", loc="left", fontsize=12)

# (c) where the gain comes from
ax = axes[2]
for col, c in zip(monthly.columns, COLORS):
    ax.plot(monthly.index, monthly[col], marker="o", color=c, lw=2, label=col)
ax.set_ylabel("monthly nMAE (%)")
ax.set_title("(c) Error by month: clustering helps most in Sep and Feb", loc="left", fontsize=12)
ax.legend(frameon=False)

for a in axes:
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
fig.suptitle("Does clustering improve the day-ahead forecast?  Test period Sep 2023 – Feb 2024, 390 homes",
             x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(CMP / "clustering_comparison.png", dpi=170)
print(f"\nsaved to {CMP}")
