"""Step 7 (Level 2): procurement-relevant comparison on the TEST set with real German prices.

Cost model:
  Day-ahead purchase of the forecast at the real DE-LU day-ahead price p (SMARD).
  Shortfall bought intraday at  p + SHORT_FIXED   + SHORT_PROP   * |p|   (penalty ASSUMED)
  Surplus  sold   intraday at  p - SURPLUS_FIXED - SURPLUS_PROP * |p|
  Imbalance cost = extra cost vs. perfect foresight.
Compared: baseline (K=1) vs. the K selected on the validation window (03_select_k.py).

Outputs:
  outputs/level2_metrics.csv
  outputs/figures/level2_business.png, level2_german_prices.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import block_bootstrap_diff, imbalance_cost, load_da_price, local_day
from config import (CACHE, FIG, K_LIST, LOCAL_TZ, OUT, SEED, SHORT_FIXED, SHORT_PROP,
                    SURPLUS_FIXED, SURPLUS_PROP)


def business_metrics(A, F, price, day, evening, n_hh, year_scale):
    err = F - A
    c = imbalance_cost(A, F, price)
    return {
        "nMAE_pct": 100 * err.abs().mean() / A.mean(),
        "Bias_pct": 100 * err.mean() / A.mean(),
        "Daily_energy_nMAE_pct": 100 * err.groupby(day).sum().abs().mean() / A.groupby(day).sum().mean(),
        "Evening_peak_MAE_kWh": err[evening].abs().mean(),
        "Price_weighted_MAE_EUR_per_h": (err.abs() / 1000 * price.abs()).mean(),
        "Shortfall_MWh_per_yr": (-err).clip(lower=0).sum() / 1000 * year_scale,
        "Surplus_MWh_per_yr": err.clip(lower=0).sum() / 1000 * year_scale,
        "P95_shortfall_kWh": (-err).clip(lower=0).quantile(0.95),
        "Imbalance_cost_EUR_per_yr": c.sum() * year_scale,
        "Imbalance_cost_EUR_per_HH_yr": c.sum() * year_scale / n_hh,
        "Imbalance_cost_pct_of_DA_bill": 100 * c.sum() / (A / 1000 * price).sum(),
    }


def main():
    preds = pd.read_parquet(CACHE / "portfolio_predictions.parquet")
    k_sel = json.load(open(OUT / "selected_k.json"))["selected_k"]
    n_hh = len(pd.read_csv(OUT / "cluster_assignments.csv"))
    A = preds["actual"]
    price = load_da_price(preds.index)
    loc = preds.index.tz_convert(LOCAL_TZ)
    day = local_day(preds.index)
    year_scale = 365 / day.nunique()
    evening = (loc.hour >= 17) & (loc.hour <= 20)
    rng = np.random.default_rng(SEED)
    ks = [1] + K_LIST
    cost = {k: imbalance_cost(A, preds[f"pred_K{k}"], price) for k in ks}

    rows = []
    for k in ks:
        r = {"K": k, "Role": "baseline" if k == 1 else
             "selected on validation" if k == k_sel else "not used for selection",
             **business_metrics(A, preds[f"pred_K{k}"], price, day, evening, n_hh, year_scale)}
        if k > 1:
            diff = cost[k].groupby(day).sum() - cost[1].groupby(day).sum()
            (lo, hi), p_better = block_bootstrap_diff(diff, rng)
            base = cost[1].sum()
            r.update({"Cost_saving_vs_baseline_pct": -100 * diff.sum() / base,
                      "Saving_CI95_low_pct": -100 * hi / base,
                      "Saving_CI95_high_pct": -100 * lo / base,
                      "P(cheaper than baseline)": p_better,
                      "Share_of_days_cheaper": (diff < 0).mean()})
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("K")
    tab.round(3).to_csv(OUT / "level2_metrics.csv")
    with pd.option_context("display.width", 200):
        print(tab.loc[[1, k_sel]].round(3).T.to_string())

    sel = tab.loc[k_sel]
    c_sel, c_base = "C0", "C3"

    # ======================= business figure =======================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) annual imbalance cost per household for every K
    ax = axes[0, 0]
    for k in ks:
        F = preds[f"pred_K{k}"]
        c_short = imbalance_cost(A, F, price, surplus_fixed=0, surplus_prop=0).sum() * year_scale / n_hh
        c_surp = imbalance_cost(A, F, price, short_fixed=0, short_prop=0).sum() * year_scale / n_hh
        col = c_base if k == 1 else c_sel if k == k_sel else "lightgray"
        ax.bar(k, c_short, color=col, label="Shortfall (bought intraday)" if k == 1 else None)
        ax.bar(k, c_surp, bottom=c_short, color=col, alpha=0.45, hatch="//",
               label="Surplus (sold intraday)" if k == 1 else None)
        txt = f"{c_short + c_surp:.2f}"
        if k == k_sel:
            txt += (f"\n{sel['Cost_saving_vs_baseline_pct']:+.1f}%\n"
                    f"[{sel['Saving_CI95_low_pct']:+.1f}, {sel['Saving_CI95_high_pct']:+.1f}]")
        ax.text(k, c_short + c_surp, txt, ha="center", va="bottom", fontsize=8,
                fontweight="bold" if k == k_sel else None)
    ax.set_xticks(ks, ["Baseline\nK=1"] + [f"K={k}" + ("\nselected" if k == k_sel else "") for k in K_LIST])
    ax.set(ylabel="Imbalance cost (EUR / household / year)",
           title=f"(a) Test-set imbalance cost; selected K={k_sel}: saving [95% CI]\n"
                 "(grey = not used for selection, shown for transparency)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.legend(fontsize=8, loc="lower right")

    # (b) monthly saving of the selected K
    ax = axes[0, 1]
    month = pd.Index(loc.strftime("%Y-%m"), name="month")
    base_m = cost[1].groupby(month).sum()
    d = 100 * (base_m - cost[k_sel].groupby(month).sum()) / base_m
    ax.bar(range(len(d)), d, color=np.where(d >= 0, c_sel, "C1"))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(d)), d.index, rotation=45, ha="right", fontsize=8)
    ax.set(ylabel="Cost saving vs baseline (%)",
           title=f"(b) Monthly imbalance-cost saving, K={k_sel} vs baseline")

    # (c) MAE by hour vs real German price profile
    ax = axes[1, 0]
    for k, col, name in [(1, c_base, "Baseline"), (k_sel, c_sel, f"Clustered K={k_sel}")]:
        mae_h = (preds[f"pred_K{k}"] - A).abs().groupby(loc.hour).mean()
        ax.plot(mae_h.index, mae_h.to_numpy(), "o-", ms=3, color=col, label=name)
    ax.set(xlabel="Hour of day (local)", ylabel="MAE (kWh/h)", xticks=range(0, 24, 3),
           title="(c) Forecast error by hour vs mean German DA price (test period)")
    ax2 = ax.twinx()
    p_h = price.groupby(loc.hour).mean()
    ax2.plot(p_h.index, p_h.to_numpy(), color="gray", ls=":", lw=2, label="Mean DE-LU DA price")
    ax2.set_ylabel("EUR/MWh")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    # (d) sensitivity to the assumed intraday penalty
    ax = axes[1, 1]
    fixed = np.arange(0, 151, 10)
    for prop, ls in [(0.0, ":"), (SHORT_PROP, "-"), (0.5, "--")]:
        sav = [100 * (1 - imbalance_cost(A, preds[f"pred_K{k_sel}"], price, f, prop,
                                         SURPLUS_FIXED, prop).sum()
                      / imbalance_cost(A, preds["pred_K1"], price, f, prop,
                                       SURPLUS_FIXED, prop).sum()) for f in fixed]
        ax.plot(fixed, sav, ls=ls, lw=2, color=c_sel, label=f"proportional part = {prop:.0%} of |DA|")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(SHORT_FIXED, color="gray", ls="--", lw=1, label=f"Assumed ({SHORT_FIXED:.0f} EUR/MWh)")
    ax.set(xlabel="Fixed shortfall surcharge (EUR/MWh)", ylabel="Cost saving vs baseline (%)",
           title=f"(d) Sensitivity to the assumed intraday penalty (K={k_sel})")
    ax.legend(fontsize=8)

    for a in axes.flat:
        a.grid(alpha=0.3)
    fig.suptitle(f"Level 2 - day-ahead procurement on the test set ({n_hh} households, "
                 f"{preds.index.min():%Y-%m-%d} to {preds.index.max():%Y-%m-%d}); "
                 f"penalty: short +{SHORT_FIXED:.0f}+{SHORT_PROP:.0%}|p|, "
                 f"surplus -{SURPLUS_FIXED:.0f}-{SURPLUS_PROP:.0%}|p|", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "level2_business.png", dpi=150); plt.close(fig)

    # ======================= German price pattern =======================
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    season = pd.Series(loc.month, index=preds.index).map(
        lambda m: "Winter (Dec-Feb)" if m in (12, 1, 2) else "Summer (Jun-Aug)" if m in (6, 7, 8)
        else "Spring/Autumn")
    for s, col in [("Winter (Dec-Feb)", "C0"), ("Spring/Autumn", "C2"), ("Summer (Jun-Aug)", "C1")]:
        for wk, ls in [(False, "-"), (True, ":")]:
            m = (season == s).to_numpy() & ((loc.dayofweek >= 5) == wk)
            prof = price[m].groupby(loc.hour[m]).mean()
            ax.plot(prof.index, prof.to_numpy(), ls=ls, color=col,
                    label=f"{s}, {'weekend' if wk else 'weekday'}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="Hour of day (local)", ylabel="Mean DA price (EUR/MWh)", xticks=range(0, 24, 3),
           title="(a) German day-ahead price by hour (test period)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax = axes[1]
    hm = price.groupby([pd.Index(loc.strftime("%Y-%m"), name="m"), pd.Index(loc.hour, name="h")]) \
        .mean().unstack()
    im = ax.imshow(hm.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(hm)), hm.index, fontsize=8)
    ax.set_xticks(range(0, 24, 3), range(0, 24, 3))
    ax.set(xlabel="Hour of day (local)",
           title=f"(b) Mean price per month and hour ({int((price < 0).sum())} negative-price hours)")
    fig.colorbar(im, ax=ax, label="EUR/MWh")
    fig.suptitle("Real DE-LU day-ahead prices (source: SMARD / Bundesnetzagentur)")
    fig.tight_layout(); fig.savefig(FIG / "level2_german_prices.png", dpi=150); plt.close(fig)
    print("saved level2_metrics.csv, level2_business.png, level2_german_prices.png")


if __name__ == "__main__":
    main()
