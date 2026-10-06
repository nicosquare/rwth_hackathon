"""Step 6: K-selection figures.

Outputs:
  outputs/figures/k_selection.png         elbow | silhouette | validation cost | cluster sizes + PV share
  outputs/figures/silhouette_diagrams.png per-household silhouette values for each K
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples

from common import prepare_matrix
from config import CACHE, FIG, K_LIST, OUT


def main():
    sil = pd.read_csv(OUT / "silhouette.csv")
    val = pd.read_csv(OUT / "validation_selection.csv")
    k_sel = json.load(open(OUT / "selected_k.json"))["selected_k"]
    feats = pd.read_parquet(CACHE / "features.parquet")
    assign = pd.read_csv(OUT / "cluster_assignments.csv", dtype={"Household_ID": str}) \
        .set_index("Household_ID").loc[feats.index]
    best_sil = int(sil.loc[sil["silhouette"].idxmax(), "K"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    # (a) elbow
    ax = axes[0, 0]
    ax.plot(sil["K"], sil["inertia"], "o-", lw=2, color="C0")
    ax.set(title="(a) Elbow: within-cluster sum of squares", xlabel="K",
           ylabel="Inertia (standardized features)", xticks=K_LIST)

    # (b) silhouette
    ax = axes[0, 1]
    ax.plot(sil["K"], sil["silhouette"], "o-", lw=2, color="C2")
    ax.plot(best_sil, sil["silhouette"].max(), "*", ms=18, color="C2")
    ax.annotate(f"best separation: K={best_sil}", (best_sil, sil["silhouette"].max()),
                xytext=(10, -15), textcoords="offset points", fontsize=9)
    ax.set(title="(b) Mean silhouette score (higher = better separated)", xlabel="K",
           ylabel="Silhouette", xticks=K_LIST)

    # (c) validation imbalance cost - the criterion that decides K
    ax = axes[1, 0]
    v = val[val["K"] > 1]
    ax.plot(v["K"], v["cost_vs_baseline_pct"].mul(-1), "o-", lw=2, color="C0",
            label="Clustered forecast")
    ax.axhline(0, ls="--", color="C3", label="Baseline, no clustering")
    ax.plot(k_sel, -val.loc[val["K"] == k_sel, "cost_vs_baseline_pct"].iloc[0], "*", ms=18,
            color="C0", label=f"Selected K={k_sel}")
    ax.set(title="(c) Imbalance-cost saving on VALIDATION window\n(selection criterion; real DE prices)",
           xlabel="K", ylabel="Cost saving vs baseline (%)", xticks=K_LIST)
    ax.legend(fontsize=8)

    # (d) cluster sizes, colored by PV share
    ax = axes[1, 1]
    cmap = plt.get_cmap("viridis")
    for k in K_LIST:
        g = assign.groupby(f"K{k}").agg(n=("HasPV", "size"), pv=("HasPV", "mean"))
        bottom = 0
        for c, r in g.iterrows():
            ax.bar(k, r["n"], bottom=bottom, color=cmap(r["pv"]), edgecolor="white", width=0.7)
            if r["n"] >= 25:
                ax.text(k, bottom + r["n"] / 2, f"C{c}\n{r['pv']:.0%}", ha="center", va="center",
                        fontsize=7, color="white" if r["pv"] < 0.6 else "black")
            bottom += r["n"]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, label="PV share in cluster")
    ax.set(title="(d) Final clusters: sizes and PV share (PV label not used)", xlabel="K",
           ylabel="Households", xticks=K_LIST)

    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Choosing K: best separation at K={best_sil}; "
                 f"K={k_sel} selected by validation imbalance cost", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG / "k_selection.png", dpi=150); plt.close(fig)

    # --- per-household silhouette diagrams ---
    Xz = prepare_matrix(feats)
    fig, axes = plt.subplots(1, len(K_LIST), figsize=(4 * len(K_LIST), 5), sharex=True)
    for ax, k in zip(axes, K_LIST):
        lab = assign[f"K{k}"].to_numpy()
        s = silhouette_samples(Xz, lab)
        y0 = 0
        for i, c in enumerate(sorted(np.unique(lab))):
            v = np.sort(s[lab == c])
            ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=f"C{i}", alpha=0.8)
            ax.text(-0.05, y0 + len(v) / 2, f"C{c}", ha="right", va="center", fontsize=8)
            y0 += len(v) + 8
        ax.axvline(s.mean(), ls="--", color="k", lw=1)
        neg = (s < 0).mean()
        ax.set(title=f"K={k}: mean={s.mean():.2f}, {neg:.0%} negative",
               xlabel="Silhouette value", yticks=[])
        ax.set_xlim(-0.4, 0.8)
        ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
    fig.suptitle("Per-household silhouette (dashed = mean; negative = closer to another cluster)")
    fig.tight_layout(); fig.savefig(FIG / "silhouette_diagrams.png", dpi=150); plt.close(fig)
    print("saved k_selection.png, silhouette_diagrams.png")


if __name__ == "__main__":
    main()
