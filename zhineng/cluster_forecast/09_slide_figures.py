"""Step 9: single-panel, large-font figures for the presentation -> outputs/figures/slides/."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CACHE, FIG, LOCAL_TZ, OUT, SHORT_FIXED

SLIDES = FIG / "slides"
SLIDES.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                     "legend.fontsize": 13, "axes.spines.top": False, "axes.spines.right": False})
BASE, CLUS, PV, GREY = "#c0392b", "#2471a3", "#e67e22", "#b3b6b7"


def save(fig, name):
    fig.tight_layout(); fig.savefig(SLIDES / name, dpi=200); plt.close(fig)


def main():
    k_sel = json.load(open(OUT / "selected_k.json"))["selected_k"]

    # ---- slide 1: baseline forecast, one winter week ----
    preds = pd.read_parquet(CACHE / "portfolio_predictions.parquet")
    s = pd.Timestamp("2024-01-15", tz="UTC")
    w = preds.loc[s:s + pd.Timedelta(days=7)]
    t = w.index.tz_convert(LOCAL_TZ)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(t, w["actual"], color="k", lw=2, label="Actual load")
    ax.plot(t, w["pred_K1"], color=BASE, lw=1.8, ls="--", label="Day-ahead forecast (baseline)")
    ax.set(ylabel="Portfolio load (kWh/h)", title="One winter week, 328 households (test set)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a", tz=LOCAL_TZ))
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    save(fig, "s1_forecast_week.png")

    # ---- slide 2: summer load shape per cluster + PV share ----
    feats = pd.read_parquet(CACHE / "features.parquet")
    assign = pd.read_csv(OUT / "cluster_assignments.csv", dtype={"Household_ID": str}) \
        .set_index("Household_ID").loc[feats.index]
    lab = assign[f"K{k_sel}"]
    pv = assign.groupby(lab)["HasPV"].mean()
    n = lab.value_counts().sort_index()
    pv_c = int(pv.idxmax())
    cols = [f"prof_summer_{h:02d}" for h in range(24)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in sorted(lab.unique()):
        p = feats.loc[lab == c, cols].mean().to_numpy()
        is_pv = c == pv_c
        ax.plot(range(24), p, lw=4 if is_pv else 2, color=PV if is_pv else GREY,
                label=f"Cluster {c} ({'solar' if is_pv else 'heat pump'}, n={n[c]})", zorder=3 if is_pv else 2)
    ax.axvspan(10, 16, color=PV, alpha=0.08)
    ax.text(13, ax.get_ylim()[1] * 0.95, "solar hours", ha="center", color=PV, fontsize=13)
    ax.set(xlabel="Hour of day", ylabel="Load / daily mean", xticks=range(0, 24, 3),
           title="Summer daily load shape per cluster")
    ax.legend(fontsize=11, loc="upper left", bbox_to_anchor=(0.0, 0.88)); ax.grid(alpha=0.3)
    save(fig, "s2_cluster_profiles.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([f"C{c}" for c in pv.index], 100 * pv,
                  color=[PV if c == pv_c else GREY for c in pv.index])
    for b_, v in zip(bars, pv):
        ax.text(b_.get_x() + b_.get_width() / 2, 100 * v + 2, f"{v:.0%}", ha="center", fontsize=15,
                fontweight="bold")
    overall = assign["HasPV"].mean()
    ax.axhline(100 * overall, ls="--", color="k", lw=1)
    ax.text(len(pv) - 0.5, 100 * overall + 2, f"all households {overall:.0%}", ha="right", fontsize=12)
    ax.set(ylabel="Households with PV (%)", ylim=(0, 112), title="PV ownership per cluster\n(PV label NOT used)")
    save(fig, "s2_pv_share.png")

    # ---- slide 3: validation vs test saving ----
    val = pd.read_csv(OUT / "validation_selection.csv").set_index("K").loc[k_sel]
    test = pd.read_csv(OUT / "level2_metrics.csv").set_index("K").loc[k_sel]
    vals = [-val["cost_vs_baseline_pct"], test["Cost_saving_vs_baseline_pct"]]
    lo = [val["Saving_CI95_low_pct"], test["Saving_CI95_low_pct"]]
    hi = [val["Saving_CI95_high_pct"], test["Saving_CI95_high_pct"]]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(2)
    ax.bar(x, vals, color=[CLUS, GREY], width=0.55)
    ax.errorbar(x, vals, yerr=[np.subtract(vals, lo), np.subtract(hi, vals)], fmt="none", ecolor="k",
                capsize=10, lw=2)
    for xi, v, l_, h_ in zip(x, vals, lo, hi):
        ax.text(xi, h_ + 0.3, f"{v:+.1f}%  [{l_:+.1f}, {h_:+.1f}]", ha="center", va="bottom",
                fontsize=14, fontweight="bold")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x, ["Validation\n(Sep 2022 - Mar 2023)", "Test year\n(Mar 2023 - Feb 2024)"])
    ax.set_ylim(min(lo) - 0.8, max(hi) + 1.5)
    ax.set(ylabel="Saving vs baseline (%)", xlim=(-0.6, 1.6),
           title=f"Clustering (K={k_sel}) vs no clustering\n95% bootstrap CI")
    save(fig, "s3_validation_vs_test.png")

    # ---- slide 4: value of uncertainty vs penalty asymmetry ----
    sens = pd.read_csv(OUT / "level3_sensitivity.csv")
    s4 = sens[sens["K"] == k_sel]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.fill_between(s4["Short_fixed_EUR_MWh"], s4["CI95_low_pct"], s4["CI95_high_pct"], color=CLUS, alpha=0.18,
                    label="95% bootstrap CI")
    ax.plot(s4["Short_fixed_EUR_MWh"], s4["Saving_pct"], "o-", color=CLUS, lw=3, ms=8,
            label="Uncertainty-aware bid vs point forecast")
    for xv, yv, tv in zip(s4["Short_fixed_EUR_MWh"], s4["Saving_pct"], s4["Mean_tau_star"]):
        ax.annotate(f"bid q{tv * 100:.0f}", (xv, yv), xytext=(-6, 12), textcoords="offset points",
                    ha="right" if xv > SHORT_FIXED else "left", fontsize=11)
    ax.axhline(0, color="k", lw=1)
    ax.axvline(SHORT_FIXED, color="gray", ls="--", lw=1.5)
    ax.text(SHORT_FIXED + 3, s4["CI95_high_pct"].max() * 0.9, "our base\nassumption", fontsize=12, color="gray")
    ax.set(xlabel="Penalty for buying a shortfall intraday (EUR/MWh)",
           ylabel="Cost saving vs point forecast (%)", title="Value of uncertainty-aware bidding")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    save(fig, "s4_value_of_uncertainty.png")

    q = pd.read_parquet(CACHE / f"level3_quantiles_K{k_sel}.parquet")
    lq = q.index.tz_convert(LOCAL_TZ)
    season = pd.Series(lq.month, index=q.index).map(
        lambda m: "Winter" if m in (12, 1, 2) else "Summer" if m in (6, 7, 8) else "Spring/Autumn")
    width = 100 * (q["q90"] - q["q10"]) / q["forecast"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for sname, col in [("Winter", CLUS), ("Spring/Autumn", "#27ae60"), ("Summer", PV)]:
        m = (season == sname).to_numpy()
        prof = width[m].groupby(lq.hour[m]).mean()
        ax.plot(prof.index, prof.to_numpy(), lw=3, color=col, label=sname)
    ax.set(xlabel="Hour of day", ylabel="80% interval width (% of forecast)", xticks=range(0, 24, 3),
           title="Where is the forecast uncertain?", ylim=(0, None))
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, "s4_uncertainty_by_hour.png")
    print(f"saved slide figures to {SLIDES}")


if __name__ == "__main__":
    main()
