"""Step 8 (Level 3): forecast uncertainty and cost-optimal day-ahead bids.

Uncertainty (model-agnostic, portfolio level - cluster quantiles are not additive):
  relative residual r = (actual - forecast) / forecast
  for forecast day D and local hour h: empirical quantiles of r over the previous
  UNC_WINDOW_DAYS days, hours h +/- UNC_HOUR_POOL (all known by midnight of D-1)
  interval = forecast * (1 + quantile)
Decision (newsvendor): with shortfall penalty c_s and surplus penalty c_u per MWh,
  the cost-minimising bid is the tau* = c_s / (c_s + c_u) quantile.
  c_s, c_u use YESTERDAY's same-hour DA price (today's price is unknown when bidding);
  costs are settled at today's real price.
Evaluation starts after the first UNC_WINDOW_DAYS test days (warm-up).

Outputs:
  outputs/level3_metrics.csv
  outputs/figures/level3_uncertainty.png, level3_decision.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import block_bootstrap_diff, imbalance_cost, load_inputs, local_day
from config import (CACHE, FIG, LOCAL_TZ, OUT, SEED, SHORT_FIXED, SHORT_PROP, SURPLUS_FIXED,
                    SURPLUS_PROP, TRAIN_END, UNC_HOUR_POOL, UNC_TAUS, UNC_WINDOW_DAYS)


# shortfall surcharge scenarios (EUR/MWh) for the penalty-asymmetry sensitivity
SHORT_FIXED_SCENARIOS = [40, 60, 80, 100, 130, 160, 200]


def tau_star(price, short_fixed=SHORT_FIXED):
    c_s = short_fixed + SHORT_PROP * price.abs()
    c_u = SURPLUS_FIXED + SURPLUS_PROP * price.abs()
    return c_s / (c_s + c_u)


def rolling_quantiles(r, hours, days, taus, tau_dyn, eval_days):
    """r: relative residuals (hourly); tau_dyn: (n_hours, n_scenarios) hour-specific tau*.
    Returns DataFrame of quantiles per hour of eval_days: fixed levels 'qNN' and
    'q_tau_star_{j}' for every scenario j."""
    out = []
    r_np, h_np, d_np = r.to_numpy(), hours, days.to_numpy()
    for d in eval_days:
        lo = d - pd.Timedelta(days=UNC_WINDOW_DAYS)
        win = (d_np >= lo) & (d_np < d) & np.isfinite(r_np)
        today = np.flatnonzero(d_np == d)
        for i in today:
            h = h_np[i]
            near = np.abs(((h_np[win] - h) + 12) % 24 - 12) <= UNC_HOUR_POOL   # circular hours
            sample = r_np[win][near]
            q = np.quantile(sample, taus + list(tau_dyn[i]))
            out.append((i, *q))
    cols = [f"q{int(t * 100):02d}" for t in taus] + \
        [f"q_tau_star_{j}" for j in range(tau_dyn.shape[1])]
    return pd.DataFrame(out, columns=["i"] + cols).set_index("i")


def pinball(actual, q, tau):
    d = actual - q
    return np.maximum(tau * d, (tau - 1) * d).mean()


def main():
    preds = pd.read_parquet(CACHE / "portfolio_predictions.parquet")
    k_sel = json.load(open(OUT / "selected_k.json"))["selected_k"]
    n_hh = len(pd.read_csv(OUT / "cluster_assignments.csv"))
    models = {1: "Baseline (K=1)", k_sel: f"Clustered (K={k_sel})"}

    idx = preds.index
    loc = idx.tz_convert(LOCAL_TZ)
    days = local_day(idx)
    hours = loc.hour.to_numpy()
    prices = pd.read_parquet(CACHE / "da_price_hourly.parquet")["da_price"]
    price = prices.reindex(idx)
    price_bid = pd.Series(prices.reindex(idx - pd.Timedelta(hours=24)).to_numpy(), index=idx)
    tau_dyn_all = np.column_stack([tau_star(price_bid, s).to_numpy() for s in SHORT_FIXED_SCENARIOS])
    j_main = SHORT_FIXED_SCENARIOS.index(SHORT_FIXED)
    tau_dyn = tau_dyn_all[:, j_main]

    first_day = pd.Timestamp(TRAIN_END.tz_convert(LOCAL_TZ).date()) + pd.Timedelta(days=UNC_WINDOW_DAYS)
    eval_days = sorted(d for d in days.unique() if d >= first_day)
    A_all = preds["actual"]

    # ---------- weather for "where is it uncertain" ----------
    _, temp, sun, _ = load_inputs()
    assign = pd.read_csv(OUT / "cluster_assignments.csv", dtype={"Household_ID": str})
    t_port = temp[assign["Weather_ID"].tolist()].mean(axis=1).reindex(idx)
    sun_port = sun.reindex(idx)

    results, rows, sens_rows = {}, [], []
    rng = np.random.default_rng(SEED)
    for k, name in models.items():
        F_all = preds[f"pred_K{k}"]
        r = (A_all - F_all) / F_all.where(F_all > 1)
        qs = rolling_quantiles(r, hours, days, list(UNC_TAUS), tau_dyn_all, eval_days)
        ev = idx[qs.index]
        A, F = A_all.loc[ev], F_all.loc[ev]
        Q = qs.mul(F.to_numpy(), axis=0).add(F.to_numpy(), axis=0)   # absolute quantiles
        Q.index = ev
        Q["q_tau_star"] = Q[f"q_tau_star_{j_main}"]
        bids = {"Point forecast": F, "Median (bias-corrected)": Q["q50"], "Newsvendor tau*": Q["q_tau_star"]}
        p = price.loc[ev]
        cost = {b: imbalance_cost(A, v, p) for b, v in bids.items()}
        d_ev = days[qs.index]
        year_scale = 365 / len(eval_days)
        results[k] = {"Q": Q, "A": A, "F": F, "cost": cost, "bids": bids, "days": d_ev}
        Q.assign(actual=A, forecast=F).to_parquet(CACHE / f"level3_quantiles_K{k}.parquet")

        # penalty-asymmetry sensitivity: tau* bid vs point forecast under each scenario
        for j, s in enumerate(SHORT_FIXED_SCENARIOS):
            c_pt = imbalance_cost(A, F, p, short_fixed=s)
            c_ts = imbalance_cost(A, Q[f"q_tau_star_{j}"], p, short_fixed=s)
            (lo, hi), pb = block_bootstrap_diff(c_ts.groupby(d_ev).sum() - c_pt.groupby(d_ev).sum(), rng)
            sens_rows.append({"Model": name, "K": k, "Short_fixed_EUR_MWh": s,
                              "Mean_tau_star": tau_dyn_all[qs.index, j].mean(),
                              "Cost_point_EUR_per_HH_yr": c_pt.sum() * year_scale / n_hh,
                              "Cost_tau_star_EUR_per_HH_yr": c_ts.sum() * year_scale / n_hh,
                              "Saving_pct": -100 * (c_ts.sum() - c_pt.sum()) / c_pt.sum(),
                              "CI95_low_pct": -100 * hi / c_pt.sum(),
                              "CI95_high_pct": -100 * lo / c_pt.sum(), "P(cheaper)": pb})

        cov80 = ((A >= Q["q10"]) & (A <= Q["q90"])).mean()
        cov90 = ((A >= Q["q05"]) & (A <= Q["q95"])).mean()
        row = {"Model": name,
               "Coverage_80pct_interval": cov80, "Coverage_90pct_interval": cov90,
               "Mean_rel_width_80pct": ((Q["q90"] - Q["q10"]) / F).mean(),
               "Mean_pinball_loss_kWh": np.mean([pinball(A, Q[f"q{int(t * 100):02d}"], t) for t in UNC_TAUS]),
               "Mean_tau_star": tau_dyn[qs.index].mean()}
        for b, c in cost.items():
            row[f"Cost_EUR_per_HH_yr [{b}]"] = c.sum() * year_scale / n_hh
        for b in ["Median (bias-corrected)", "Newsvendor tau*"]:
            diff = cost[b].groupby(d_ev).sum() - cost["Point forecast"].groupby(d_ev).sum()
            (lo, hi), pb = block_bootstrap_diff(diff, rng)
            base = cost["Point forecast"].sum()
            row[f"Saving_vs_point_pct [{b}]"] = -100 * diff.sum() / base
            row[f"Saving_CI95 [{b}]"] = f"[{-100 * hi / base:+.2f}, {-100 * lo / base:+.2f}]"
            row[f"P(cheaper) [{b}]"] = pb
        season = pd.Series(loc[qs.index].month, index=ev).map(
            lambda m: "winter" if m in (12, 1, 2) else "summer" if m in (6, 7, 8) else "spring_autumn")
        inside = (A >= Q["q10"]) & (A <= Q["q90"])
        for s, v in inside.groupby(season):
            row[f"Coverage_80pct_{s}"] = v.mean()
        row["Shortfall_MWh_yr [point]"] = (A - F).clip(lower=0).sum() / 1000 * year_scale
        row["Surplus_MWh_yr [point]"] = (F - A).clip(lower=0).sum() / 1000 * year_scale
        row["Shortfall_MWh_yr [tau*]"] = (A - Q["q_tau_star"]).clip(lower=0).sum() / 1000 * year_scale
        row["Surplus_MWh_yr [tau*]"] = (Q["q_tau_star"] - A).clip(lower=0).sum() / 1000 * year_scale
        rows.append(row)

    # clustered vs baseline, both with tau* bids
    c1 = results[1]["cost"]["Newsvendor tau*"]; ck = results[k_sel]["cost"]["Newsvendor tau*"]
    d_ev = results[1]["days"]
    (lo, hi), pb = block_bootstrap_diff(ck.groupby(d_ev).sum() - c1.groupby(d_ev).sum(), rng)
    rows[1]["Saving_vs_baseline_tau*_pct"] = -100 * (ck.sum() - c1.sum()) / c1.sum()
    rows[1]["Saving_vs_baseline_tau*_CI95"] = f"[{-100 * hi / c1.sum():+.2f}, {-100 * lo / c1.sum():+.2f}]"
    tab = pd.DataFrame(rows).set_index("Model")
    tab.T.to_csv(OUT / "level3_metrics.csv")
    with pd.option_context("display.width", 200, "display.max_colwidth", 30):
        print(tab.T.to_string())

    # ======================= uncertainty figure (selected K) =======================
    R = results[k_sel]
    Q, A, F = R["Q"], R["A"], R["F"]
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1])

    # (a) fan chart: winter week and summer week
    for j, (title, start) in enumerate([("Winter week", "2024-01-15"), ("Summer week", "2023-07-10")]):
        ax = fig.add_subplot(gs[0, j])
        s = pd.Timestamp(start, tz="UTC"); w = slice(s, s + pd.Timedelta(days=7))
        ax.fill_between(Q.loc[w].index, Q.loc[w, "q05"], Q.loc[w, "q95"], color="C0", alpha=0.15, label="90% interval")
        ax.fill_between(Q.loc[w].index, Q.loc[w, "q10"], Q.loc[w, "q90"], color="C0", alpha=0.3, label="80% interval")
        ax.plot(A.loc[w].index, A.loc[w], color="k", lw=1.3, label="Actual")
        ax.plot(F.loc[w].index, F.loc[w], color="C0", lw=1, label="Point forecast")
        ax.plot(Q.loc[w].index, Q.loc[w, "q_tau_star"], color="C3", lw=1, ls="--", label="tau* bid")
        ax.set(title=f"(a) {title}: prediction intervals (K={k_sel})", ylabel="kWh/h")
        ax.tick_params(axis="x", labelsize=7); ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7, ncol=2)

    # (b) calibration (reliability) for both models
    ax = fig.add_subplot(gs[1, 0])
    for k, col in [(1, "C3"), (k_sel, "C0")]:
        Rk = results[k]
        emp = [(Rk["A"] <= Rk["Q"][f"q{int(t * 100):02d}"]).mean() for t in UNC_TAUS]
        ax.plot(UNC_TAUS, emp, "o-", color=col, label=models[k])
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set(xlabel="Nominal quantile level", ylabel="Observed share actual <= quantile",
           title="(b) Calibration of predicted quantiles (test set)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) heatmap of 80% interval width (% of forecast) by month x hour
    ax = fig.add_subplot(gs[1, 1])
    lq = Q.index.tz_convert(LOCAL_TZ)
    width = 100 * (Q["q90"] - Q["q10"]) / F
    hm = width.groupby([pd.Index(lq.strftime("%Y-%m"), name="m"), pd.Index(lq.hour, name="h")]).mean().unstack()
    im = ax.imshow(hm.to_numpy(), aspect="auto", cmap="magma_r")
    ax.set_yticks(range(len(hm)), hm.index, fontsize=8)
    ax.set_xticks(range(0, 24, 3), range(0, 24, 3))
    ax.set(xlabel="Hour of day (local)", title="(c) Where is it uncertain? 80% interval width (% of forecast)")
    fig.colorbar(im, ax=ax, label="% of forecast")

    # (d) realized error and predicted width vs daily temperature and sunshine
    dd = pd.DataFrame({"abs_err": (A - F).abs() / F, "width": width / 100 / 2,
                       "t": t_port.loc[Q.index], "sun": sun_port.loc[Q.index]})
    daily = dd.groupby(local_day(Q.index)).agg(abs_err=("abs_err", "mean"), width=("width", "mean"),
                                                t=("t", "mean"), sun=("sun", "sum"))
    for j, (col, label, bins) in enumerate([("t", "Daily mean temperature (degC)", np.arange(-6, 28, 3)),
                                            ("sun", "Daily sunshine (sum of hourly values)", 8)]):
        ax = fig.add_subplot(gs[2, j])
        b = pd.cut(daily[col], bins)
        g = daily.groupby(b, observed=True).agg(abs_err=("abs_err", "mean"), width=("width", "mean"),
                                                 n=("abs_err", "size"))
        x = [iv.mid for iv in g.index]
        ax.plot(x, 100 * g["abs_err"], "o-", color="k", label="Realized mean |error| (% of forecast)")
        ax.plot(x, 100 * g["width"], "s--", color="C0", label="Predicted 80% half-width (% of forecast)")
        for xi, n in zip(x, g["n"]):
            ax.annotate(f"n={n}", (xi, 0), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6)
        ax.set(xlabel=label, ylabel="% of forecast", ylim=(0, None),
               title=f"(d{j + 1}) Uncertainty vs {'temperature' if col == 't' else 'sunshine'} (daily, K={k_sel})")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle(f"Level 3 - forecast uncertainty (rolling {UNC_WINDOW_DAYS}-day residual quantiles, "
                 f"evaluated {eval_days[0]:%Y-%m-%d} to {eval_days[-1]:%Y-%m-%d})", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG / "level3_uncertainty.png", dpi=150); plt.close(fig)

    # ======================= decision figure =======================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    year_scale = 365 / len(eval_days)

    # (a) cost per bidding strategy and model
    ax = axes[0, 0]
    strategies = list(results[1]["bids"])
    w = 0.38
    for j, (k, col) in enumerate([(1, "C3"), (k_sel, "C0")]):
        vals = [results[k]["cost"][s].sum() * year_scale / n_hh for s in strategies]
        bars = ax.bar(np.arange(3) + (j - 0.5) * w, vals, width=w, color=col, label=models[k])
        for b_, v, s in zip(bars, vals, strategies):
            lab = f"{v:.2f}"
            if s != "Point forecast":
                lab += f"\n{tab.loc[models[k], f'Saving_vs_point_pct [{s}]']:+.1f}%"
            ax.text(b_.get_x() + b_.get_width() / 2, v, lab, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(3), strategies)
    ax.set(ylabel="Imbalance cost (EUR / household / year)",
           title="(a) Cost by bidding strategy (labels: saving vs point forecast)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15); ax.legend(fontsize=8)

    # (b) tau* vs yesterday's price
    ax = axes[0, 1]
    ev_idx = results[k_sel]["Q"].index
    ax.scatter(price_bid.loc[ev_idx], tau_star(price_bid.loc[ev_idx]), s=3, alpha=0.3, color="C0")
    ax.set(xlabel="Yesterday's same-hour DA price (EUR/MWh)", ylabel="tau* (bid quantile)",
           title="(b) Cost-optimal bid quantile tau* = c_short / (c_short + c_surplus)")
    ax.axhline(0.5, color="k", lw=0.8, ls=":")

    # (c) monthly saving of tau* bid vs point forecast (selected K)
    ax = axes[1, 0]
    R = results[k_sel]
    month = pd.Index(R["Q"].index.tz_convert(LOCAL_TZ).strftime("%Y-%m"), name="m")
    base_m = R["cost"]["Point forecast"].groupby(month).sum()
    sav_m = 100 * (base_m - R["cost"]["Newsvendor tau*"].groupby(month).sum()) / base_m
    ax.bar(range(len(sav_m)), sav_m, color=np.where(sav_m >= 0, "C0", "C1"))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(sav_m)), sav_m.index, rotation=45, ha="right", fontsize=8)
    ax.set(ylabel="Saving vs point forecast (%)", title=f"(c) Monthly saving of tau* bidding (K={k_sel})")

    # (d) shortfall / surplus energy before and after
    ax = axes[1, 1]
    r = tab.loc[models[k_sel]]
    labels = ["Point forecast", "tau* bid"]
    sf = [r["Shortfall_MWh_yr [point]"], r["Shortfall_MWh_yr [tau*]"]]
    sp = [r["Surplus_MWh_yr [point]"], r["Surplus_MWh_yr [tau*]"]]
    ax.bar(labels, sf, color="C3", label="Shortfall (bought intraday)")
    ax.bar(labels, sp, bottom=sf, color="C2", alpha=0.6, label="Surplus (sold intraday)")
    for i in range(2):
        ax.text(i, sf[i] / 2, f"{sf[i]:.0f}", ha="center", color="white")
        ax.text(i, sf[i] + sp[i] / 2, f"{sp[i]:.0f}", ha="center")
    ax.set(ylabel="MWh / year", title=f"(d) Imbalance energy: shortfall vs surplus (K={k_sel})")
    ax.legend(fontsize=8)

    for a in axes.flat:
        a.grid(alpha=0.3)
    fig.suptitle("Level 3 - using uncertainty for day-ahead bidding (real DE-LU prices; "
                 f"penalty short +{SHORT_FIXED:.0f}+{SHORT_PROP:.0%}|p|, surplus -{SURPLUS_FIXED:.0f}-{SURPLUS_PROP:.0%}|p|)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "level3_decision.png", dpi=150); plt.close(fig)

    # ======================= penalty-asymmetry sensitivity =======================
    sens = pd.DataFrame(sens_rows)
    sens.round(4).to_csv(OUT / "level3_sensitivity.csv", index=False)
    print(sens.round(3).to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for k, col in [(1, "C3"), (k_sel, "C0")]:
        s = sens[sens["K"] == k]
        ax.plot(s["Short_fixed_EUR_MWh"], s["Saving_pct"], "o-", color=col, lw=2, label=models[k])
        ax.fill_between(s["Short_fixed_EUR_MWh"], s["CI95_low_pct"], s["CI95_high_pct"],
                        color=col, alpha=0.15)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(SHORT_FIXED, color="gray", ls="--", lw=1, label=f"Main assumption ({SHORT_FIXED:.0f})")
    s1 = sens[sens["K"] == k_sel]
    for x, y, t in zip(s1["Short_fixed_EUR_MWh"], s1["Saving_pct"], s1["Mean_tau_star"]):
        ax.annotate(f"tau*={t:.2f}", (x, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=7)
    ax.set(xlabel=f"Fixed shortfall surcharge (EUR/MWh); surplus surcharge fixed at {SURPLUS_FIXED:.0f}",
           ylabel="Saving of tau* bid vs point forecast (%)",
           title="(a) Value of uncertainty grows with penalty asymmetry\n(band = 95% bootstrap CI)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    for k, col in [(1, "C3"), (k_sel, "C0")]:
        s = sens[sens["K"] == k]
        ax.plot(s["Short_fixed_EUR_MWh"], s["Cost_point_EUR_per_HH_yr"], ":", color=col, lw=2,
                label=f"{models[k]}: point forecast")
        ax.plot(s["Short_fixed_EUR_MWh"], s["Cost_tau_star_EUR_per_HH_yr"], "-", color=col, lw=2,
                label=f"{models[k]}: tau* bid")
    ax.set(xlabel="Fixed shortfall surcharge (EUR/MWh)", ylabel="Imbalance cost (EUR / household / year)",
           title="(b) Imbalance cost: point forecast vs uncertainty-aware bid")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Level 3 - when does uncertainty pay off? (test set, real DE-LU prices)", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG / "level3_sensitivity.png", dpi=150); plt.close(fig)
    print("saved level3_metrics.csv, level3_sensitivity.csv, level3_uncertainty.png, "
          "level3_decision.png, level3_sensitivity.png")


if __name__ == "__main__":
    main()
