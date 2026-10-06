"""Shared building blocks: household features, clustering, cluster-level forecasting,
metrics, and the procurement cost model. Every stage (validation / final test) calls the
same functions with different time windows, so selection and evaluation are consistent."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import (CACHE, CLUSTER_FEATURES, K_LIST, LAGS_H, LOCAL_TZ, META_DIR,
                    MIN_ACTIVE_PER_CLUSTER, MIN_TRAIN_DAYS, SEED, SHORT_FIXED, SHORT_PROP,
                    SURPLUS_FIXED, SURPLUS_PROP)


# =============================== inputs ===============================
def load_inputs():
    load = pd.read_parquet(CACHE / "load_hourly.parquet")
    temp = pd.read_parquet(CACHE / "temp_hourly.parquet")
    sun = pd.read_parquet(CACHE / "sun_hourly.parquet")["sun"]
    hh = pd.read_csv(META_DIR / "households.csv", sep=";", dtype={"Household_ID": str})
    hh["HasPV"] = hh["Installation_HasPVSystem"].astype(str).str.lower() \
        .map({"true": 1.0, "false": 0.0})
    hh = hh.set_index("Household_ID")[["Weather_ID", "HasPV", "Group"]]
    return load, temp, sun, hh


# ========================= household features =========================
def safe_ratio(a, b):
    return a / b if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan


def household_features(y, temp, sun):
    """y: training-period hourly load of one household (UTC index). Returns dict or None."""
    df = pd.DataFrame({"y": y, "t": temp, "sun": sun}).dropna(subset=["y"])
    if df.empty:
        return None
    loc = df.index.tz_convert(LOCAL_TZ)
    df["hour"], df["month"] = loc.hour, loc.month
    df["date"] = loc.normalize().tz_localize(None)

    daily = df.groupby("date").agg(kwh=("y", "sum"), n=("y", "size"), t=("t", "mean"))
    daily = daily[daily["n"] >= 22]                      # ~complete days (DST-safe)
    if len(daily) < MIN_TRAIN_DAYS:
        return None
    daily["month"] = daily.index.month
    mean_daily = daily["kwh"].mean()
    f = {"n_train_days": len(daily), "mean_daily_kwh": mean_daily}

    # temperature sensitivity (heating regime)
    cold = daily[daily["t"] < 15]
    if len(cold) >= 20 and cold["t"].std() > 1:
        slope, _ = np.polyfit(cold["t"], cold["kwh"], 1)
        f["temp_slope_rel"] = -slope / mean_daily * 100
        f["temp_r2"] = np.corrcoef(cold["t"], cold["kwh"])[0, 1] ** 2
    else:
        f["temp_slope_rel"] = f["temp_r2"] = np.nan

    summer_d = daily.loc[daily["month"].isin([6, 7, 8]), "kwh"].mean()
    winter_d = daily.loc[daily["month"].isin([12, 1, 2]), "kwh"].mean()
    f["summer_winter_ratio"] = safe_ratio(summer_d, winter_d)

    # PV signature (summer half-year, Apr-Sep)
    s = df[df["month"].between(4, 9)]
    midday = s[s["hour"].between(10, 15)]
    shoulder = s[s["hour"].isin([6, 7, 8, 17, 18, 19, 20])]
    f["pv_midday_ratio"] = safe_ratio(midday["y"].mean(), shoulder["y"].mean())
    md = midday.groupby("date").agg(y=("y", "sum"), sun=("sun", "sum"), n=("y", "size"))
    md = md[md["n"] == 6]
    f["pv_sun_corr"] = md["y"].corr(md["sun"]) if len(md) >= 30 else np.nan

    # time-of-day energy shares
    tot = df["y"].sum()
    for name, (a, b) in {"night": (0, 5), "morning": (6, 9), "midday": (10, 16),
                         "evening": (17, 23)}.items():
        f[f"share_{name}"] = df.loc[df["hour"].between(a, b), "y"].sum() / tot

    # normalized 24h profiles (interpretation plots only)
    for tag, months in {"all": range(1, 13), "summer": [6, 7, 8], "winter": [12, 1, 2]}.items():
        p = df[df["month"].isin(months)].groupby("hour")["y"].mean().reindex(range(24))
        p = p / p.mean() if p.notna().sum() > 12 else p * np.nan
        for h in range(24):
            f[f"prof_{tag}_{h:02d}"] = p.iloc[h]
    return f


def build_features(load, temp, sun, hh, train_end, verbose=True):
    train = load[load.index < train_end]
    rows = {}
    for hid in train.columns:
        wid = hh["Weather_ID"].get(hid)
        t = temp[wid] if wid in temp.columns else temp.mean(axis=1)
        f = household_features(train[hid], t[train.index], sun[train.index])
        if f is not None:
            rows[hid] = f
    feats = pd.DataFrame.from_dict(rows, orient="index")
    feats.index.name = "Household_ID"
    if verbose:
        print(f"eligible households (>= {MIN_TRAIN_DAYS} complete days before "
              f"{train_end:%Y-%m-%d}): {len(feats)}/{load.shape[1]}")
    return feats


# ============================== clustering ============================
def prepare_matrix(feats):
    X = feats[CLUSTER_FEATURES].copy()
    X = X.fillna(X.median())
    X = X.clip(X.quantile(0.01), X.quantile(0.99), axis=1)   # k-means is outlier-sensitive
    return StandardScaler().fit_transform(X)


def relabel_by_size(labels):
    order = pd.Series(labels).value_counts().index           # largest cluster -> 1
    mapping = {old: new for new, old in enumerate(order, 1)}
    return np.array([mapping[l] for l in labels])


def cluster_all_k(feats, verbose=True):
    """k-means for every K in K_LIST. Returns (assign[K1..K6], silhouette table)."""
    Xz = prepare_matrix(feats)
    assign = pd.DataFrame({"K1": 1}, index=feats.index)
    sil = []
    for k in K_LIST:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xz)
        assign[f"K{k}"] = relabel_by_size(km.labels_)
        sil.append({"K": k, "silhouette": silhouette_score(Xz, km.labels_), "inertia": km.inertia_})
        if verbose:
            print(f"  K={k}: silhouette={sil[-1]['silhouette']:.3f}, "
                  f"sizes={np.bincount(assign[f'K{k}'])[1:].tolist()}")
    return assign, pd.DataFrame(sil)


# ============================= forecasting ============================
def calendar_features(idx):
    loc = idx.tz_convert(LOCAL_TZ)
    return pd.DataFrame({"hour": loc.hour, "dow": loc.dayofweek,
                         "is_weekend": (loc.dayofweek >= 5).astype(int)}, index=idx)


def daily_mean_local(s):
    """Mean of s over the local calendar day each hour belongs to (same-day info)."""
    return s.groupby(s.index.tz_convert(LOCAL_TZ).normalize()).transform("mean")


def build_design(mean_load, temp_c, sun, cal):
    X = cal.copy()
    for lag in LAGS_H:                                     # >= 24h: known at midnight of D-1
        X[f"lag_{lag}h"] = mean_load.shift(lag)
    X["lag_prevday_mean"] = mean_load.shift(24).rolling(24, min_periods=18).mean()
    X["temp"] = temp_c                                     # observed weather = perfect forecast
    X["temp_day_mean"] = daily_mean_local(temp_c)
    X["temp_lag24"] = temp_c.shift(24)
    X["sun"] = sun
    X["sun_day_sum"] = daily_mean_local(sun) * 24
    return X


def forecast_cluster(load_c, temp_c, sun, cal, train_end, eval_start, eval_end):
    """Fit on [.., train_end) and forecast [eval_start, eval_end).
    Target = mean load per active household; returned as cluster TOTAL (x n_active)."""
    n = load_c.notna().sum(axis=1)
    mean_load = load_c.mean(axis=1)
    X = build_design(mean_load, temp_c, sun, cal)
    tr = (X.index < train_end) & mean_load.notna() & (n >= MIN_ACTIVE_PER_CLUSTER)
    ev = (X.index >= eval_start) & (X.index < eval_end) & (n > 0)
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=40, early_stopping=False,
                                          random_state=SEED)
    model.fit(X[tr], mean_load[tr], sample_weight=n[tr])  # weight reliable (many-household) hours
    pred_mean = np.clip(model.predict(X[ev]), 0, None)
    return pd.DataFrame({"actual": load_c[ev].sum(axis=1), "pred": pred_mean * n[ev]},
                        index=X.index[ev])


def portfolio_forecasts(load, temp, sun, assign, ks, train_end, eval_start, eval_end,
                        verbose=True):
    """Portfolio forecast for each K = sum of its cluster forecasts. Returns
    (DataFrame[actual, pred_K*], per-cluster metric rows)."""
    load = load[assign.index]                     # same households for every K
    cal = calendar_features(load.index)
    out, rows_c = None, []
    for k in ks:
        parts = []
        for c in sorted(assign[f"K{k}"].unique()):
            members = assign.index[assign[f"K{k}"] == c]
            temp_c = temp[assign.loc[members, "Weather_ID"].tolist()].mean(axis=1)
            res = forecast_cluster(load[members], temp_c, sun, cal, train_end, eval_start, eval_end)
            parts.append(res)
            rows_c.append({"K": k, "Cluster": c, "n_households": len(members),
                           **accuracy(res["actual"], res["pred"])})
        tot = pd.concat(parts).groupby(level=0).sum()
        if out is None:
            out = tot[["actual"]].copy()
        out[f"pred_K{k}"] = tot["pred"]
        if verbose:
            print(f"  K={k}: nMAE={accuracy(tot['actual'], tot['pred'])['nMAE_pct']:.2f}%")
    return out, pd.DataFrame(rows_c)


def accuracy(actual, pred):
    err = pred - actual
    mae = err.abs().mean()
    return {"MAE_kWh": mae, "RMSE_kWh": np.sqrt((err ** 2).mean()),
            "nMAE_pct": 100 * mae / actual.mean(), "Bias_pct": 100 * err.mean() / actual.mean()}


# ============================ cost model ==============================
def load_da_price(idx):
    p = pd.read_parquet(CACHE / "da_price_hourly.parquet")["da_price"].reindex(idx)
    if p.isna().any():
        raise ValueError(f"{p.isna().sum()} hours without DA price - rerun 00_fetch_prices.py")
    return p


def imbalance_cost(actual, pred, price, short_fixed=SHORT_FIXED, short_prop=SHORT_PROP,
                   surplus_fixed=SURPLUS_FIXED, surplus_prop=SURPLUS_PROP):
    """Extra cost (EUR) vs perfect foresight. Load in kWh/h, price in EUR/MWh.
    shortfall bought at DA + fixed + prop*|DA|; surplus sold at DA - fixed - prop*|DA|."""
    short = (actual - pred).clip(lower=0) / 1000
    surplus = (pred - actual).clip(lower=0) / 1000
    return short * (short_fixed + short_prop * price.abs()) + \
        surplus * (surplus_fixed + surplus_prop * price.abs())


def block_bootstrap_diff(daily_diff, rng, n_boot=2000, block=7):
    """95% CI of the summed daily difference, resampling whole weeks; and P(sum < 0)."""
    v = daily_diff.to_numpy()
    sums = np.array([v[i:i + block].sum() for i in range(0, len(v), block)])
    boot = rng.choice(sums, size=(n_boot, len(sums)), replace=True).sum(axis=1)
    return np.percentile(boot, [2.5, 97.5]), (boot < 0).mean()


def local_day(idx):
    return pd.Index(idx.tz_convert(LOCAL_TZ).normalize().tz_localize(None), name="day")
