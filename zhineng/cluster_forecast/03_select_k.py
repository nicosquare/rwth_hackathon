"""Step 3: choose K on the VALIDATION window only (no test data touched).

Selection stage:
  cluster + fit models on sub-train  [.., VAL_START)
  forecast validation window         [VAL_START, TRAIN_END)
  imbalance cost with real German DA prices
Rule: smallest K (incl. K=1 = no clustering) whose validation cost is within
PARSIMONY_TOL of the lowest cost.

Outputs:
  outputs/validation_selection.csv, outputs/selected_k.json
  outputs/figures/validation_k_selection.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (accuracy, block_bootstrap_diff, build_features, cluster_all_k,
                    imbalance_cost, load_da_price, load_inputs, local_day, portfolio_forecasts)
from config import FIG, K_LIST, OUT, PARSIMONY_TOL, SEED, TRAIN_END, VAL_START


def main():
    load, temp, sun, hh = load_inputs()
    print(f"Selection stage: sub-train < {VAL_START:%Y-%m-%d}, "
          f"validation {VAL_START:%Y-%m-%d} .. {TRAIN_END:%Y-%m-%d}")
    feats = build_features(load, temp, sun, hh, VAL_START)
    assign, sil = cluster_all_k(feats)
    assign = assign.join(hh[["Weather_ID"]])
    ks = [1] + K_LIST
    preds, _ = portfolio_forecasts(load, temp, sun, assign, ks, VAL_START, VAL_START, TRAIN_END)

    A = preds["actual"]
    price = load_da_price(preds.index)
    day = local_day(preds.index)
    cost = {k: imbalance_cost(A, preds[f"pred_K{k}"], price) for k in ks}
    base = cost[1].sum()
    rng = np.random.default_rng(SEED)
    rows = []
    for k in ks:
        r = {"K": k, "n_households": len(assign), "imbalance_cost_EUR": cost[k].sum(),
             "cost_vs_baseline_pct": 100 * (cost[k].sum() - base) / base,
             **accuracy(A, preds[f"pred_K{k}"])}
        if k > 1:
            r["silhouette"] = sil.set_index("K").loc[k, "silhouette"]
            diff = cost[k].groupby(day).sum() - cost[1].groupby(day).sum()
            (lo, hi), r["P(cheaper than baseline)"] = block_bootstrap_diff(diff, rng)
            r["Saving_CI95_low_pct"] = -100 * hi / base
            r["Saving_CI95_high_pct"] = -100 * lo / base
        rows.append(r)
    tab = pd.DataFrame(rows)

    best = tab["imbalance_cost_EUR"].min()
    ok = tab[tab["imbalance_cost_EUR"] <= best * (1 + PARSIMONY_TOL)]
    k_sel = int(ok["K"].min())
    k_best = int(tab.loc[tab["imbalance_cost_EUR"].idxmin(), "K"])
    tab["selected"] = tab["K"] == k_sel
    tab.round(4).to_csv(OUT / "validation_selection.csv", index=False)
    json.dump({"selected_k": k_sel, "lowest_cost_k": k_best,
               "rule": f"smallest K with validation cost <= {1 + PARSIMONY_TOL:.3f} x lowest",
               "validation_window": [str(VAL_START), str(TRAIN_END)]},
              open(OUT / "selected_k.json", "w"), indent=2)
    print(tab.round(3).to_string(index=False))
    print(f"\nlowest validation cost at K={k_best}; selected K={k_sel} (parsimony rule)")

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    ax.plot(tab["K"], tab["imbalance_cost_EUR"] / 1000, "o-", lw=2, color="C0")
    ax.axhline(base / 1000, ls="--", color="C3", label="Baseline K=1 (no clustering)")
    ax.axhspan(best / 1000, best * (1 + PARSIMONY_TOL) / 1000, color="C2", alpha=0.2,
               label=f"Within {PARSIMONY_TOL:.1%} of lowest cost")
    ax.plot(k_sel, tab.loc[tab["K"] == k_sel, "imbalance_cost_EUR"].iloc[0] / 1000, "*", ms=20,
            color="C2", label=f"Selected K={k_sel}")
    for _, r in tab.iterrows():
        ax.annotate(f"{-r['cost_vs_baseline_pct']:+.1f}%", (r["K"], r["imbalance_cost_EUR"] / 1000),
                    xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set(xlabel="K (1 = no clustering)", ylabel="Imbalance cost on validation (kEUR)",
           xticks=ks, title="(a) Validation imbalance cost - decides K\n(labels: saving vs baseline)")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(tab["K"], tab["nMAE_pct"], "o-", lw=2, color="C1", label="nMAE")
    ax.set(xlabel="K (1 = no clustering)", ylabel="Validation nMAE (%)", xticks=ks,
           title="(b) Validation accuracy (for reference)")
    ax2 = ax.twinx()
    ax2.plot(sil["K"], sil["silhouette"], "s:", color="gray", label="Silhouette (sub-train)")
    ax2.set_ylabel("Silhouette")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle(f"K selection on validation window {VAL_START:%Y-%m-%d} to {TRAIN_END:%Y-%m-%d} "
                 f"({len(assign)} households, real DE-LU day-ahead prices)")
    fig.tight_layout(); fig.savefig(FIG / "validation_k_selection.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
