"""Step 2: final clustering on ALL training data (< TRAIN_END) for K = 2..6.

Outputs:
  cache/features.parquet           raw features + 24h profiles per eligible household
  outputs/cluster_assignments.csv  Household_ID, K1..K6 labels, HasPV, Weather_ID
  outputs/silhouette.csv, outputs/pca_loadings.csv
  outputs/figures/silhouette_vs_k.png, pca_clusters.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from common import build_features, cluster_all_k, load_inputs, prepare_matrix
from config import CACHE, CLUSTER_FEATURES, FIG, K_LIST, OUT, SEED, TRAIN_END


def main():
    load, temp, sun, hh = load_inputs()
    feats = build_features(load, temp, sun, hh, TRAIN_END)
    feats.to_parquet(CACHE / "features.parquet")
    assign, sil = cluster_all_k(feats)
    sil.to_csv(OUT / "silhouette.csv", index=False)
    assign = assign.join(hh[["HasPV", "Weather_ID"]])
    assign.to_csv(OUT / "cluster_assignments.csv")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(sil["K"], sil["silhouette"], "o-", lw=2)
    ax.set(xlabel="Number of clusters K", ylabel="Mean silhouette score",
           title="Cluster quality vs K", xticks=K_LIST)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "silhouette_vs_k.png", dpi=150); plt.close(fig)

    Xz = prepare_matrix(feats)
    pca = PCA(n_components=2, random_state=SEED).fit(Xz)
    pc = pca.transform(Xz)
    fig, axes = plt.subplots(1, len(K_LIST), figsize=(4 * len(K_LIST), 3.8), sharex=True, sharey=True)
    for ax, k in zip(axes, K_LIST):
        for c in sorted(assign[f"K{k}"].unique()):
            m = assign[f"K{k}"].to_numpy() == c
            ax.scatter(pc[m, 0], pc[m, 1], s=12, alpha=0.7, label=f"C{c} (n={m.sum()})")
        ax.set_title(f"K={k}  (sil={sil.set_index('K').loc[k, 'silhouette']:.2f})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    fig.suptitle("Household clusters in PCA space (training-period load features)")
    fig.tight_layout(); fig.savefig(FIG / "pca_clusters.png", dpi=150); plt.close(fig)

    pd.DataFrame(pca.components_.T, index=CLUSTER_FEATURES, columns=["PC1", "PC2"]).round(3) \
        .to_csv(OUT / "pca_loadings.csv")
    print("saved cluster_assignments.csv, silhouette.csv, figures")


if __name__ == "__main__":
    main()
