# %% [markdown]
# Level 1a: cluster households by consumption pattern
#
# Each household is described by
#   - its daily load SHAPE in the heating season (Oct-Apr) and non-heating season (May-Sep), 24 h each
#   - LEVEL: mean daily consumption (log)
#   - HEATING SENSITIVITY: extra kWh/day per degree below 15 C, relative to its mean
#   - PV SIGNATURE: correlation of midday imports with sunshine in summer (PV owners -> negative)
# Features use only data BEFORE the cutoff, so the forecasting test period is never seen.
# Run cell by cell (VS Code / Spyder) or as a script.

# %% Config
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")       # avoids KMeans/MKL memory-leak warning on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"   # the challenge repo's data folder
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

TZ = "Europe/Zurich"
CUTOFF = pd.Timestamp("2023-09-01", tz="UTC")  # agree this with the team = start of test set
MIN_DAYS_PER_SEASON = 14
METHOD = os.environ.get("METHOD", "kmedoids")  # "kmeans" or "kmedoids"
K = None              # None = choose by silhouette; set e.g. K = 4 to force
K_RANGE = range(2, 9)
HEATING_MONTHS = [10, 11, 12, 1, 2, 3, 4]
MIDDAY_HOURS = range(10, 15)
SEED = 42
RES = OUT / METHOD  # results per method; data cache stays in OUT
RES.mkdir(parents=True, exist_ok=True)


# %% Load smart meter data (15 min -> hourly, cached)
def load_hourly():
    cache = OUT / "hourly_total.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    parts = []
    files = sorted((DATA / "15min").glob("*.csv"))
    for i, f in enumerate(files):
        d = pd.read_csv(f, sep=";", usecols=["Timestamp", "kWh_received_Total"])
        d["Timestamp"] = pd.to_datetime(d["Timestamp"], utc=True)
        s = d.set_index("Timestamp")["kWh_received_Total"].resample("h").sum(min_count=4)
        parts.append(s.rename("kWh").reset_index().assign(hh=f.stem))
        if i % 50 == 0:
            print(f"  loaded {i + 1}/{len(files)}")
    df = pd.concat(parts, ignore_index=True).dropna(subset=["kWh"])
    df.to_parquet(cache)
    return df


def load_weather():
    parts = []
    for f in (DATA / "weather_data_hourly").glob("*.csv"):
        w = pd.read_csv(f, sep=";", usecols=["Weather_ID", "Timestamp",
                                             "Temperature_avg_hourly", "Sunshine_duration_hourly"])
        w["Timestamp"] = pd.to_datetime(w["Timestamp"], utc=True)
        parts.append(w)
    return pd.concat(parts, ignore_index=True).rename(
        columns={"Temperature_avg_hourly": "T", "Sunshine_duration_hourly": "sun"})


hourly = load_hourly()
weather = load_weather()
hh_meta = pd.read_csv(DATA / "smart_meter_meta_data" / "households.csv", sep=";", dtype={"Household_ID": str})
survey = pd.read_csv(DATA / "smart_meter_meta_data" / "meta_data.csv", sep=";", dtype={"Household_ID": str})
hh_meta = hh_meta.rename(columns={"Household_ID": "hh"})
survey = survey.rename(columns={"Household_ID": "hh"})
print(f"{hourly.hh.nunique()} households, {len(hourly):,} hourly rows")


# %% Build features (training period only)
df = hourly[hourly.Timestamp < CUTOFF].merge(hh_meta[["hh", "Weather_ID"]], on="hh", how="left")
df = df.merge(weather, on=["Weather_ID", "Timestamp"], how="left")
local = df.Timestamp.dt.tz_convert(TZ)
df["hour"] = local.dt.hour
df["date"] = local.dt.date
df["heating"] = local.dt.month.isin(HEATING_MONTHS)
df["midday_kWh"] = df.kWh.where(df.hour.isin(MIDDAY_HOURS), 0)

# daily table (complete days only)
daily = (df.groupby(["hh", "date"])
           .agg(kWh=("kWh", "sum"), n=("kWh", "size"), T=("T", "mean"), sun=("sun", "sum"),
                heating=("heating", "first"), midday=("midday_kWh", "sum"))
           .reset_index())
daily = daily[daily.n >= 23]
daily["HDD"] = (15 - daily["T"]).clip(lower=0)

# enough data in both seasons?
season_days = daily.groupby(["hh", "heating"]).size().unstack(fill_value=0)
ok = season_days[(season_days.get(True, 0) >= MIN_DAYS_PER_SEASON) &
                 (season_days.get(False, 0) >= MIN_DAYS_PER_SEASON)].index
print(f"{len(ok)} households with >= {MIN_DAYS_PER_SEASON} days in both seasons before cutoff")
df, daily = df[df.hh.isin(ok)], daily[daily.hh.isin(ok)]

# 1) normalised daily shapes, 24 h x 2 seasons
prof = df.groupby(["hh", "heating", "hour"])["kWh"].mean().unstack(["heating", "hour"])
shape_h = prof[True].div(prof[True].mean(axis=1), axis=0).add_prefix("heat_h")
shape_n = prof[False].div(prof[False].mean(axis=1), axis=0).add_prefix("summer_h")


# 2) scalars
def hh_scalars(g):
    mean_day = g.kWh.mean()
    w = g.dropna(subset=["HDD"])  # skip days with weather gaps
    slope = np.polyfit(w.HDD, w.kWh, 1)[0] if len(w) > 10 and w.HDD.std() > 0 else np.nan
    summer = g[~g.heating & g.sun.notna()]
    share = summer.midday / summer.kWh.replace(0, np.nan)
    pv_corr = share.corr(summer.sun) if len(summer) > 10 else np.nan
    return pd.Series({"mean_daily_kWh": mean_day,
                      "log_level": np.log(mean_day),
                      "heat_sensitivity": slope / mean_day,
                      "pv_corr": pv_corr})


scal = daily.groupby("hh").apply(hh_scalars)
scal["pv_corr"] = scal["pv_corr"].fillna(0)

feat = pd.concat([shape_h, shape_n, scal], axis=1).dropna()
print(f"feature matrix: {feat.shape}")


# %% Scale + weight blocks so each shape block counts like one scalar
SCALARS = ["log_level", "heat_sensitivity", "pv_corr"]
X = pd.DataFrame(StandardScaler().fit_transform(feat), index=feat.index, columns=feat.columns)
shape_cols = [c for c in X.columns if c.startswith(("heat_h", "summer_h"))]
X[shape_cols] /= np.sqrt(24)
X = X[shape_cols + SCALARS]


# %% Clustering methods
from scipy.spatial.distance import cdist


def kmedoids(D, k, n_init=20, max_iter=100, seed=SEED):
    """Alternating k-medoids on a precomputed distance matrix, k-medoids++ init, best of n_init."""
    rng = np.random.default_rng(seed)
    n, best = len(D), None
    for _ in range(n_init):
        med = [rng.integers(n)]
        for _ in range(k - 1):
            d = D[:, med].min(axis=1) ** 2
            med.append(rng.choice(n, p=d / d.sum()))
        med = np.array(med)
        for _ in range(max_iter):
            lab = D[:, med].argmin(axis=1)
            new = np.array([np.flatnonzero(lab == c)[D[np.ix_(lab == c, lab == c)].sum(axis=0).argmin()]
                            for c in range(k)])
            if set(new) == set(med):
                break
            med = new
        lab = D[:, med].argmin(axis=1)
        cost = D[np.arange(n), med[lab]].sum()
        if best is None or cost < best[0]:
            best = (cost, lab, med)
    return best[1], best[2]


D = cdist(X.values, X.values)


def fit(k, n_init):
    if METHOD == "kmedoids":
        return kmedoids(D, k, n_init=n_init)
    km = KMeans(k, n_init=n_init, random_state=SEED).fit(X)
    return km.labels_, None


# %% Choose k
sil = {k: silhouette_score(D, fit(k, 20)[0], metric="precomputed") for k in K_RANGE}
print(f"[{METHOD}] silhouette:", {k: round(v, 3) for k, v in sil.items()})
k_best = K or max(sil, key=sil.get)
print(f"using k = {k_best}")

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(list(sil), list(sil.values()), "o-")
ax.axvline(k_best, ls="--", c="grey")
ax.set(xlabel="number of clusters k", ylabel="silhouette score", title=f"Choosing k ({METHOD})")
fig.tight_layout(); fig.savefig(RES / "01_silhouette.png", dpi=150)


# %% Fit final clustering, order clusters by consumption level
labels, medoids = fit(k_best, 50)
lab = pd.Series(labels, index=X.index)
order = feat.groupby(lab)["mean_daily_kWh"].mean().sort_values().index
lab = lab.map({old: new for new, old in enumerate(order)})
feat["cluster"] = lab
# medoids are real households: the "most typical" member of each cluster
feat["is_medoid"] = feat.index.isin(X.index[medoids]) if medoids is not None else False

res = (feat.reset_index(names="hh")
           .merge(hh_meta[["hh", "Group", "Weather_ID", "Installation_HasPVSystem"]], on="hh", how="left")
           .merge(survey, on="hh", how="left"))
res["PV"] = res.Installation_HasPVSystem.map({True: "PV", False: "no PV", "True": "PV", "False": "no PV"}).fillna("unknown")
if medoids is not None:
    print("medoid households:", res.loc[res.is_medoid, ["hh", "cluster", "PV"]].to_dict("records"))


# %% Describe clusters
summary = res.groupby("cluster").agg(
    n=("hh", "size"),
    mean_daily_kWh=("mean_daily_kWh", "mean"),
    heat_sensitivity=("heat_sensitivity", "mean"),
    pv_corr=("pv_corr", "mean"))
print(summary.round(3))

# how well do clusters line up with known metadata? (validation, not used in clustering)
for col in ["PV", "Survey_HeatPump_Installation_Type", "Survey_Building_Type",
            "Survey_Installation_HasElectricVehicle", "Survey_HeatDistribution_System_FloorHeating"]:
    print(f"\n--- {col} (row %) ---")
    print((pd.crosstab(res.cluster, res[col].fillna("unknown"), normalize="index") * 100).round(0))


# %% Plot: average hourly profile per cluster (kWh, local time)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, season, title in [(axes[0], True, "Heating season (Oct-Apr)"),
                          (axes[1], False, "Non-heating season (May-Sep)")]:
    for c in sorted(res.cluster.unique()):
        members = res.hh[res.cluster == c]
        ax.plot(prof.loc[members, season].mean(), marker=".",
                label=f"cluster {c} (n={len(members)})")
    ax.set(title=title, xlabel="hour of day (local)", xticks=range(0, 24, 3))
axes[0].set_ylabel("mean kWh per hour")
axes[1].legend()
fig.tight_layout(); fig.savefig(RES / "02_cluster_profiles.png", dpi=150)


# %% Plot: heating sensitivity vs PV signature, marker = known PV flag
fig, ax = plt.subplots(figsize=(6.5, 5))
for pv, m in [("PV", "^"), ("no PV", "o"), ("unknown", "x")]:
    sub = res[res.PV == pv]
    sc = ax.scatter(sub.pv_corr, sub.heat_sensitivity, c=sub.cluster, cmap="tab10",
                    vmin=0, vmax=9, marker=m, s=30, label=pv)
ax.axvline(0, c="grey", lw=0.5)
ax.set(xlabel="PV signature: corr(midday share, sunshine)  [negative = PV-like]",
       ylabel="heating sensitivity (1/K)", title="Households coloured by cluster")
ax.legend(title="survey PV flag")
fig.tight_layout(); fig.savefig(RES / "03_sensitivity_vs_pv.png", dpi=150)


# %% Plot: PCA view
pcs = PCA(2, random_state=SEED).fit_transform(X)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(pcs[:, 0], pcs[:, 1], c=lab.loc[X.index], cmap="tab10", vmin=0, vmax=9, s=20)
ax.set(xlabel="PC1", ylabel="PC2", title="Clusters in PCA space")
fig.tight_layout(); fig.savefig(RES / "04_pca.png", dpi=150)


# %% Compare with the other method (if it has been run)
other = OUT / ("kmeans" if METHOD == "kmedoids" else "kmedoids") / "household_clusters.csv"
if other.exists():
    from sklearn.metrics import adjusted_rand_score
    o = pd.read_csv(other, dtype={"hh": str})[["hh", "cluster"]].merge(res[["hh", "cluster"]], on="hh",
                                                                      suffixes=("_other", "_this"))
    print(f"\nagreement with {other.parent.name}: ARI = {adjusted_rand_score(o.cluster_other, o.cluster_this):.3f}")
    print(pd.crosstab(o.cluster_other, o.cluster_this,
                      rownames=[other.parent.name], colnames=[METHOD]))


# %% Save for the forecasting team
res.to_csv(RES / "household_clusters.csv", index=False)
summary.to_csv(RES / "cluster_summary.csv")

# cluster-level hourly series over the FULL period (train + test), per-household mean
agg = (hourly.merge(res[["hh", "cluster"]], on="hh")
             .groupby(["cluster", "Timestamp"])
             .agg(kWh_total=("kWh", "sum"), n_households=("kWh", "size"))
             .reset_index())
agg["kWh_per_household"] = agg.kWh_total / agg.n_households
agg.to_csv(RES / "cluster_hourly.csv", index=False)
print(f"\nsaved to {RES}")
plt.show()
