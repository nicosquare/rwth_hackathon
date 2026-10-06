"""Four figures, one per level, each answering the README's question for that level.
Rules: full test period (no cherry-picked days), every value axis starts at 0, one colour per entity,
the caveat is in the subtitle. Output: outputs/final_plots."""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
KM = OUT / "kmedoids"
FIN = OUT / "final_plots"
shutil.rmtree(FIN, ignore_errors=True)
FIN.mkdir()
TZ = "Europe/Zurich"

INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
C = {"actual": INK, "naive": "#898781", "global": "#2a78d6", "cluster": "#eb6834",
     "pv": "#eda100", "nonpv": "#4a3aa7", "bid": "#008300"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "legend.frameon": False, "savefig.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
})


def header(fig, question, answer, note):
    import textwrap
    fig.text(0.01, 0.975, question, fontsize=13, color=INK2, va="top")
    fig.text(0.01, 0.925, answer, fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.015, textwrap.fill(note, 175), fontsize=11, color=INK2, va="bottom", linespacing=1.4)


def bar_labels(ax, bars, fmt):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(b.get_height()), ha="center",
                va="bottom", fontsize=13, fontweight="bold", color=INK)


# ---- data
tf = pd.read_csv(KM / "forecast" / "test_forecasts.csv", index_col=0)
tf.index = pd.to_datetime(tf.index, utc=True)
local = tf.index.tz_convert(TZ)
pm = pd.read_csv(KM / "forecast" / "portfolio_metrics.csv", index_col=0)
pcm = pd.read_csv(KM / "forecast" / "per_cluster_metrics.csv", index_col=0)
prm = pd.read_csv(KM / "procurement" / "procurement_metrics.csv", index_col=0)
curve = pd.read_csv(KM / "level3" / "cost_vs_tau.csv", index_col=0).iloc[:, 0]
nv = pd.read_csv(KM / "level3" / "newsvendor_result.csv", index_col=0).iloc[:, 0]
clusters = pd.read_csv(KM / "household_clusters.csv", dtype={"hh": str})
PROF = OUT / "profiles_summer.json"
if not PROF.exists():   # mean summer (May-Sep) hourly import per cluster, training period only
    h = pd.read_parquet(OUT / "hourly_total.parquet")
    h = h[h.Timestamp < pd.Timestamp("2023-09-01", tz="UTC")].merge(clusters[["hh", "cluster"]], on="hh")
    loc = h.Timestamp.dt.tz_convert(TZ)
    h = h[loc.dt.month.isin([5, 6, 7, 8, 9])].assign(hour=loc.dt.hour)
    p = h.groupby(["cluster", "hh", "hour"]).kWh.mean().groupby(["cluster", "hour"]).mean().unstack(0).round(3)
    json.dump({"pv": p[0].tolist(), "nonpv": p[1].tolist()}, open(PROF, "w"))
prof = json.load(open(PROF))
SEEDS = {"global": [6.12, 6.20, 6.34, 6.28, 6.27], "cluster": [6.01, 5.97, 5.99, 5.94, 5.87]}  # nMAE %, 5 runs
MODELS = [("Naive (last week)", "naive", "Naive\n(same hour\nlast week)"),
          ("Global model", "global", "One model\nfor all homes"),
          ("Per-cluster models", "cluster", "One model\nper group")]

# ================================================================ LEVEL 0
hours = tf.groupby(local.date).size()
daily = tf.groupby(local.date).sum()[hours == 24] / 1000   # MWh per day, complete days only
daily.index = pd.to_datetime(daily.index)
fig = plt.figure(figsize=(14, 6.2))
ax = fig.add_axes([0.06, 0.2, 0.66, 0.58])
ax.plot(daily.index, daily["Naive (last week)"], color=C["naive"], lw=1.3, label="naive: same hour last week")
ax.plot(daily.index, daily["actual"], color=C["actual"], lw=2.4, label="actual")
ax.plot(daily.index, daily["Global model"], color=C["global"], lw=1.6, label="day-ahead model")
ax.set_ylim(0, daily.values.max() * 1.1)
ax.set_ylabel("portfolio demand, MWh per day")
ax.legend(loc="upper left", ncol=3)
ax.set_title("Whole test period, Sep 2023 – Feb 2024 (390 homes)")
ax2 = fig.add_axes([0.78, 0.2, 0.2, 0.58])
vals = [pm.loc["Naive (last week)", "nMAE %"], np.mean(SEEDS["global"])]
b = ax2.bar(["Naive", "Model"], vals, color=[C["naive"], C["global"]], width=0.6)
bar_labels(ax2, b, "{:.1f}%")
ax2.set_ylim(0, 24)
ax2.set_title("Hourly error (nMAE)")
ax2.grid(axis="x", visible=False)
header(fig, "Level 0 · Can we forecast tomorrow’s demand, one day ahead?",
       "Yes: hourly error falls from 20% (naive) to 6% with a gradient-boosting model",
       "Forecast made at noon on D−1 from load ≥ 48 h old, calendar and weather · trained Jan 2021 – Aug 2023, "
       "tested on the last 24% · caveat: measured weather used for day D, so errors are slightly optimistic")
fig.savefig(FIN / "Level0_forecast.png")
plt.close(fig)

# ================================================================ LEVEL 1
ct = pd.crosstab(clusters.cluster, clusters.PV)
n_pv, n_non = int((clusters.cluster == 0).sum()), int((clusters.cluster == 1).sum())
precision = 100 * ct.loc[0, "PV"] / (ct.loc[0, "PV"] + ct.loc[0, "no PV"])
recall = 100 * ct.loc[0, "PV"] / ct["PV"].sum()
fig = plt.figure(figsize=(14, 6.2))
ax = fig.add_axes([0.06, 0.2, 0.5, 0.58])
ax.axvspan(10, 15, color=C["pv"], alpha=0.12, lw=0)
ax.plot(range(24), prof["nonpv"], color=C["nonpv"], lw=2.6, marker="o", ms=5)
ax.plot(range(24), prof["pv"], color=C["pv"], lw=2.6, marker="o", ms=5)
ax.text(23.3, prof["nonpv"][-1], f"Group B: no PV pattern\n{n_non} homes", color=C["nonpv"], fontsize=12,
        fontweight="bold", va="center")
ax.text(23.3, prof["pv"][-1], f"Group A: PV pattern\n{n_pv} homes", color=C["pv"], fontsize=12,
        fontweight="bold", va="center")
ax.set(xlim=(-0.5, 30), ylim=(0, 1.25), xticks=range(0, 24, 3), xlabel="hour of day (local)",
       ylabel="grid import, kWh per hour per home")
ax.set_title("The two groups found by k-medoids: average summer day")
ax2 = fig.add_axes([0.66, 0.2, 0.32, 0.58])
labels = ["One model\nfor all homes", "One model\nper group"]
b = ax2.bar(labels, [np.mean(SEEDS["global"]), np.mean(SEEDS["cluster"])], color=[C["global"], C["cluster"]],
            width=0.55)
bar_labels(ax2, b, "{:.2f}%")
ax2.set_ylim(0, 8)
ax2.set_ylabel("hourly error, nMAE (%)")
ax2.set_title("Forecast error, mean of 5 runs")
ax2.grid(axis="x", visible=False)
header(fig, "Level 1 · Can we find groups of households and forecast each group separately?",
       f"Yes: meter data alone finds the PV homes; a model per group cuts error by "
       f"{100 * (1 - np.mean(SEEDS['cluster']) / np.mean(SEEDS['global'])):.1f}%",
       f"Better in 5 of 5 runs. No labels used; checked against the survey: {precision:.0f}% of group A have PV (0 non-PV homes), but only "
       f"{recall:.0f}% of PV owners are found · the PV group is harder to forecast: "
       f"{pcm.iloc[0, 2]:.0f}% error vs {pcm.iloc[1, 2]:.0f}% for group B")
fig.savefig(FIN / "Level1_groups.png")
plt.close(fig)

# ================================================================ LEVEL 2
metrics = [
    ("EUR per household-year", "Cost of forecast error\n€ per home per year", "€{:.0f}", 1),
    ("peak-hour nMAE 17-21h (%)", "Error in evening peak\n(17–21h, highest prices), %", "{:.1f}%", 1),
    ("worst day energy error (kWh)", "Worst single day\nmissed energy, MWh", "{:.1f}", 1e-3),
    ("share of error that is short (%)", "Share of error that is\nshort (buy at a premium), %", "{:.0f}%", 1),
]
fig = plt.figure(figsize=(14, 6.4))
for i, (key, title, fmt, scale) in enumerate(metrics):
    ax = fig.add_axes([0.04 + i * 0.245, 0.22, 0.2, 0.5])
    v = [prm.loc[key, m] * scale for m, _, _ in MODELS]
    b = ax.bar(range(3), v, color=[C[k] for _, k, _ in MODELS], width=0.62)
    bar_labels(ax, b, fmt)
    ax.set_xticks(range(3), [lab for _, _, lab in MODELS], fontsize=10)
    ax.set_ylim(0, max(v) * 1.2 if key != "share of error that is short (%)" else 100)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="y", labelsize=10)
glob_e, clus_e = prm.loc["EUR per household-year", "Global model"], prm.loc["EUR per household-year", "Per-cluster models"]
header(fig, "Level 2 · Which metric matters for day-ahead procurement, and how do the two models compare?",
       f"The € cost of error: −{prm.loc['saving vs naive (%)', 'Per-cluster models']:.0f}% vs naive, "
       f"but both models cost the same (€{glob_e:.2f} vs €{clus_e:.2f})",
       "Cost = shortfall × 1.5 × day-ahead price + surplus × 0.7 × price, real hourly Swiss prices · the +50% / −30% "
       "penalties are assumed (allowed by the brief); with other penalties either model can win by < €1")
fig.savefig(FIN / "Level2_procurement_metrics.png")
plt.close(fig)

# ================================================================ LEVEL 3
f = tf["Per-cluster models"]
rel = (tf["actual"] - f) / f * 100
q = rel.groupby(local.hour).quantile([0.1, 0.5, 0.9]).unstack()
fig = plt.figure(figsize=(14, 6.2))
ax = fig.add_axes([0.06, 0.2, 0.42, 0.58])
ax.fill_between(q.index, q[0.1], q[0.9], color=C["cluster"], alpha=0.25, lw=0, label="80% of errors")
ax.plot(q.index, q[0.5], color=C["cluster"], lw=2, label="median error")
ax.axhline(0, color=INK, lw=1)
ax.set(xticks=range(0, 24, 3), xlabel="hour of day (local)", ylabel="actual − forecast, % of forecast")
ax.legend(loc="lower left")
ax.set_title("Where is the model uncertain? Widest around midday")
ax2 = fig.add_axes([0.56, 0.2, 0.42, 0.58])
mean_cost = nv.filter(like="bid = mean forecast").iloc[0]
ax2.plot(curve.index, curve.values, color=C["bid"], lw=2.4, marker="o", ms=5, label="bid = quantile τ")
ax2.axhline(mean_cost, color=C["cluster"], lw=2, label="bid = mean forecast")
best = curve.idxmin()
ax2.plot(best, curve.min(), "o", ms=14, mfc="none", mec=C["bid"], mew=2.5)
ax2.annotate(f"best τ = {best} = 0.5 / (0.5 + 0.3)\n{nv.filter(like='saving').iloc[0]:.1f}% cheaper than the mean",
             (best, curve.min()), xytext=(0.36, 1500), fontsize=12, color=INK,
             arrowprops=dict(arrowstyle="->", color=INK2))
ax2.set(ylim=(0, curve.max() * 1.1), xlabel="bid quantile τ", ylabel="cost of forecast error, €")
ax2.legend(loc="lower right")
ax2.set_title("How to use it: bid above the mean")
header(fig, "Level 3 · Where is the model uncertain, and how can uncertainty improve the purchase?",
       f"Uncertainty peaks at midday; bidding the 62.5% quantile saves {nv.filter(like='saving').iloc[0]:.1f}%",
       "Being short costs more (+50%) than being long (−30%), so the cost-optimal bid is the τ* = 0.5 / 0.8 quantile "
       "(newsvendor) · uncertainty from the model’s errors of the past 28 days · evaluated 30 Sep 2023 – 28 Feb 2024")
fig.savefig(FIN / "Level3_uncertainty_and_bid.png")
plt.close(fig)

print("\n".join(sorted(p.name for p in FIN.glob("*.png"))))
