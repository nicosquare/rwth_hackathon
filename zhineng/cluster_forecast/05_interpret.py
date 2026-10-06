"""Step 5: interpret clusters (load profiles, feature heatmap, PV share, metadata).

Usage:  python 05_interpret.py            -> K selected on validation + best-silhouette K
        python 05_interpret.py 4          -> a specific K
Outputs per K:
  outputs/cluster_summary_K{k}.csv
  outputs/figures/cluster_profiles_K{k}.png, feature_heatmap_K{k}.png, pv_share_K{k}.png
Plus outputs/pv_share_by_k.csv (PV share per cluster for all K).
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CACHE, CLUSTER_FEATURES, FIG, K_LIST, META_DIR, OUT


def to_float_bool(s):
    return s.astype(str).str.lower().map({"true": 1.0, "false": 0.0})


def load_metadata():
    m = pd.read_csv(META_DIR / "meta_data.csv", sep=";", dtype={"Household_ID": str}) \
        .set_index("Household_ID")
    return pd.DataFrame({
        "LivingArea_m2": pd.to_numeric(m["Survey_Building_LivingArea"], errors="coerce"),
        "Residents": pd.to_numeric(m["Survey_Building_Residents"], errors="coerce"),
        "GroundSourceHP_share": m["Survey_HeatPump_Installation_Type"].astype(str)
            .str.contains("ground").where(m["Survey_HeatPump_Installation_Type"].notna()).astype(float),
        "FloorHeating_share": to_float_bool(m["Survey_HeatDistribution_System_FloorHeating"]),
        "DHWbyHP_share": to_float_bool(m["Survey_DHW_Production_ByHeatPump"]),
        "EV_share": to_float_bool(m["Survey_Installation_HasElectricVehicle"]),
    })


def interpret(k, feats, assign, meta):
    lab = assign[f"K{k}"]
    clusters = sorted(lab.unique())
    colors = {c: f"C{i}" for i, c in enumerate(clusters)}

    # --- summary table: load features + PV validation + metadata (all mean per cluster) ---
    df = feats[["mean_daily_kwh"] + CLUSTER_FEATURES].join(assign[["HasPV"]]).join(meta)
    df["Cluster"] = lab
    summary = df.groupby("Cluster").mean()
    summary.insert(0, "N", lab.value_counts().sort_index())
    summary.insert(summary.columns.get_loc("HasPV") + 1, "N_PV_known",
                   df.groupby("Cluster")["HasPV"].count())
    summary = summary.rename(columns={"HasPV": "PV_share"})
    summary.round(3).to_csv(OUT / f"cluster_summary_K{k}.csv")

    # --- 24h normalized profiles: all year / winter / summer ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, tag, title in zip(axes, ["all", "winter", "summer"],
                              ["All year", "Winter (Dec-Feb)", "Summer (Jun-Aug)"]):
        cols = [f"prof_{tag}_{h:02d}" for h in range(24)]
        for c in clusters:
            p = feats.loc[lab.index[lab == c], cols].mean()
            pv = summary.loc[c, "PV_share"]
            ax.plot(range(24), p.to_numpy(), lw=2, color=colors[c],
                    label=f"C{c} (n={summary.loc[c, 'N']}, PV {pv:.0%})")
        ax.set(title=title, xlabel="Hour of day (local time)", xticks=range(0, 24, 3))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Load / daily mean load")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Typical normalized daily load shape per cluster (K={k}, train period)")
    fig.tight_layout(); fig.savefig(FIG / f"cluster_profiles_K{k}.png", dpi=150); plt.close(fig)

    # --- heatmap of standardized feature means ---
    X = feats[CLUSTER_FEATURES]
    Xz = (X - X.mean()) / X.std()
    centers = Xz.groupby(lab).mean()
    fig, ax = plt.subplots(figsize=(10, 0.6 * k + 2))
    im = ax.imshow(centers.to_numpy(), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    for (i, j), v in np.ndenumerate(centers.to_numpy()):
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(CLUSTER_FEATURES)), CLUSTER_FEATURES, rotation=35, ha="right")
    ax.set_yticks(range(k), [f"C{c} (n={summary.loc[c, 'N']})" for c in clusters])
    fig.colorbar(im, ax=ax, label="z-score (cluster mean)")
    ax.set_title(f"Cluster characteristics, K={k} (standardized feature means)")
    fig.tight_layout(); fig.savefig(FIG / f"feature_heatmap_K{k}.png", dpi=150); plt.close(fig)

    # --- PV share per cluster (validation; PV label was NOT used for clustering) ---
    overall = assign["HasPV"].mean()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar([f"C{c}" for c in clusters], summary["PV_share"], color=[colors[c] for c in clusters])
    for i, c in enumerate(clusters):
        ax.text(i, summary.loc[c, "PV_share"] + 0.01,
                f"{summary.loc[c, 'PV_share']:.0%}\n(n={summary.loc[c, 'N_PV_known']})",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(overall, ls="--", color="gray", label=f"All households ({overall:.0%})")
    ax.set(ylabel="Share of households with PV", ylim=(0, 1),
           title=f"PV ownership per cluster (K={k}) - validation only")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / f"pv_share_K{k}.png", dpi=150); plt.close(fig)

    print(f"\n=== K={k} ===")
    print(summary[["N", "mean_daily_kwh", "temp_slope_rel", "summer_winter_ratio",
                   "pv_midday_ratio", "pv_sun_corr", "PV_share", "N_PV_known"]].round(2).to_string())


def main():
    feats = pd.read_parquet(CACHE / "features.parquet")
    assign = pd.read_csv(OUT / "cluster_assignments.csv", dtype={"Household_ID": str}) \
        .set_index("Household_ID").loc[feats.index]
    meta = load_metadata()

    if len(sys.argv) > 1:
        ks = [int(a) for a in sys.argv[1:]]
    else:
        sil = pd.read_csv(OUT / "silhouette.csv")
        k_sel = json.load(open(OUT / "selected_k.json"))["selected_k"]
        ks = sorted({int(sil.loc[sil["silhouette"].idxmax(), "K"])} | ({k_sel} - {1}))
    for k in ks:
        interpret(k, feats, assign, meta)

    pv = pd.DataFrame({f"K{k}": assign.groupby(f"K{k}")["HasPV"].mean() for k in K_LIST})
    pv.index.name = "Cluster"
    pv.round(3).to_csv(OUT / "pv_share_by_k.csv")
    print(f"\nPV share per cluster for every K (overall {assign['HasPV'].mean():.0%}):")
    print(pv.round(2).to_string())


if __name__ == "__main__":
    main()
