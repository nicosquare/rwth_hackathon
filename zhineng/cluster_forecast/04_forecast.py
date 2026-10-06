"""Step 4: final test. Models refit on ALL data < TRAIN_END (clusters from 02_cluster.py),
forecast the test window [TRAIN_END, TEST_END) for K = 1 (baseline) and K = 2..6.
K was chosen beforehand on the validation window (03_select_k.py); the other K are
reported for transparency only.

Outputs:
  outputs/metrics_by_k.csv, outputs/metrics_by_cluster.csv
  cache/portfolio_predictions.parquet
  outputs/figures/forecast_error_vs_k.png, forecast_example_weeks.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import accuracy, load_inputs, portfolio_forecasts
from config import CACHE, FIG, K_LIST, OUT, TEST_END, TRAIN_END


def main():
    load, temp, sun, _ = load_inputs()
    assign = pd.read_csv(OUT / "cluster_assignments.csv", dtype={"Household_ID": str}) \
        .set_index("Household_ID")
    k_sel = json.load(open(OUT / "selected_k.json"))["selected_k"]
    print(f"Test stage: train < {TRAIN_END:%Y-%m-%d}, test until {TEST_END:%Y-%m-%d}, "
          f"selected K={k_sel}")
    preds, by_cluster = portfolio_forecasts(load, temp, sun, assign, [1] + K_LIST,
                                            TRAIN_END, TRAIN_END, TEST_END)
    preds.to_parquet(CACHE / "portfolio_predictions.parquet")
    by_cluster.round(4).to_csv(OUT / "metrics_by_cluster.csv", index=False)

    mk = pd.DataFrame([{"K": k, **accuracy(preds["actual"], preds[f"pred_K{k}"])}
                       for k in [1] + K_LIST])
    base = mk.loc[mk["K"] == 1, "nMAE_pct"].iloc[0]
    mk["nMAE_improvement_vs_baseline_pct"] = 100 * (base - mk["nMAE_pct"]) / base
    mk["role"] = mk["K"].map(lambda k: "baseline" if k == 1 else
                             "selected on validation" if k == k_sel else "not used for selection")
    mk = mk.merge(pd.read_csv(OUT / "silhouette.csv")[["K", "silhouette"]], on="K", how="left")
    mk.round(4).to_csv(OUT / "metrics_by_k.csv", index=False)
    print(mk.round(3).to_string(index=False))

    # --- forecast error vs K: selected K highlighted, others greyed ---
    fig, ax = plt.subplots(figsize=(6.5, 4))
    clus = mk[mk["K"] > 1]
    ax.plot(clus["K"], clus["nMAE_pct"], "o-", color="lightgray", lw=2,
            label="Other K (not used for selection)")
    sel = mk[mk["K"] == k_sel]
    ax.plot(sel["K"], sel["nMAE_pct"], "*", ms=18, color="C0",
            label=f"Selected on validation: K={k_sel}")
    ax.axhline(base, ls="--", color="C3", label=f"Baseline, no clustering ({base:.2f}%)")
    for _, r in clus.iterrows():
        ax.annotate(f"{r['nMAE_improvement_vs_baseline_pct']:+.1f}%", (r["K"], r["nMAE_pct"]),
                    xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set(xlabel="Number of clusters K", ylabel="Portfolio nMAE on test set (%)", xticks=K_LIST,
           title="Test-set accuracy (labels: improvement vs baseline)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "forecast_error_vs_k.png", dpi=150); plt.close(fig)

    # --- example winter + summer week ---
    weeks = {"Winter week": "2024-01-15", "Summer week": "2023-07-10"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
    for ax, (name, start) in zip(axes, weeks.items()):
        s = pd.Timestamp(start, tz="UTC")
        w = preds.loc[s:s + pd.Timedelta(days=7)]
        ax.plot(w.index, w["actual"], color="k", lw=1.5, label="Actual")
        ax.plot(w.index, w["pred_K1"], color="C3", lw=1.1, label="Baseline (no clustering)")
        ax.plot(w.index, w[f"pred_K{k_sel}"], color="C0", lw=1.1, label=f"Clustered, K={k_sel}")
        ax.set(title=name, ylabel="Portfolio load (kWh/h)"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "forecast_example_weeks.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
