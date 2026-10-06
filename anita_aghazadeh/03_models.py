"""Step 3: Level 0 (baselines + global LightGBM) and Level 1 (PV identification + group models).

Usage: python3 -I 03_models.py <work_dir> <out_dir> <train_end> <test_start> <test_end> <tag>
  train rows: ts <  train_end          (strictly older than every test row)
  test rows : test_start <= ts < test_end
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

work, out, train_end, test_start, test_end, tag = sys.argv[1:7]
os.makedirs(out, exist_ok=True)
D = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)  # noqa: E731
train_end, test_start, test_end = D(train_end), D(test_start), D(test_end)
T0 = time.time()
log = lambda *a: print(f"[{tag} {time.time() - T0:6.0f}s]", *a, flush=True)  # noqa: E731

FEATS = ["lag_48", "lag_72", "lag_96", "lag_168", "samehour_mean7", "samehour_std7", "lastday_mean",
         "roll7_mean", "roll28_mean", "lastday_ratio", "temp", "hum", "sun", "wind", "precip",
         "temp_dmean", "temp_dmin", "sun_dsum", "hdd", "hour", "dow", "month", "doy", "is_weekend"]
STRICT = ["lag_48", "lag_168", "samehour_mean7", "roll28_mean"]  # rows needed for a fair baseline comparison

f = pl.read_parquet(f"{work}/features.parquet")
train = f.filter(pl.col("ts") < train_end)
test = f.filter((pl.col("ts") >= test_start) & (pl.col("ts") < test_end)).drop_nulls(STRICT)
log("train rows", train.height, "test rows", test.height, "| test households", test["Household_ID"].n_unique())
assert train["ts"].max() < test["ts"].min(), "temporal leakage"

# ------------------------------------------------------------------ Level 1a: PV identification
S, W_ = [6, 7, 8], [12, 1, 2]
MID, EVE, NIGHT = list(range(10, 15)), list(range(18, 22)), [1, 2, 3, 4]
c = pl.col
sm, wm = c("month").is_in(S), c("month").is_in(W_)
mid, eve, nig = c("hour").is_in(MID), c("hour").is_in(EVE), c("hour").is_in(NIGHT)
n_s, n_w = sm.sum(), wm.sum()


def seasonal(m, name, n):
    return [pl.when(n >= 200).then(c("kwh").filter(m).mean()).alias(f"{name}_all"),
            pl.when(n >= 200).then(c("kwh").filter(m & mid).mean()).alias(f"{name}_mid"),
            pl.when(n >= 200).then(c("kwh").filter(m & eve).mean()).alias(f"{name}_eve"),
            pl.when(n >= 200).then(c("kwh").filter(m & nig).mean()).alias(f"{name}_night")]


g = train.group_by("Household_ID").agg(*seasonal(sm, "s", n_s), *seasonal(wm, "w", n_w))
dm = (train.filter(c("month").is_in([4, 5, 6, 7, 8, 9]))
      .with_columns(c("ts").dt.convert_time_zone("Europe/Zurich").dt.date().alias("ld"))
      .group_by("Household_ID", "ld").agg(c("kwh").filter(mid).mean().alias("dmid"), c("kwh").mean().alias("dall"),
                                          c("temp_dmean").first().alias("t"), c("kwh").count().alias("n"))
      .filter(c("n") >= 20).with_columns((c("dmid") / (c("dall") + 1e-6)).alias("share"))
      .group_by("Household_ID").agg(pl.corr("share", "t").alias("corr_mid_temp"), c("share").count().alias("nd"))
      .with_columns(pl.when(c("nd") >= 20).then(c("corr_mid_temp")).otherwise(None)))
g = g.join(dm.drop("nd"), on="Household_ID", how="left").with_columns(
    (c("s_mid") / c("s_all")).alias("s_mid_share"), (c("w_mid") / c("w_all")).alias("w_mid_share"),
    (c("s_mid") / (c("s_night") + 1e-6)).alias("s_mid_to_night"),
    (c("s_mid") / (c("s_eve") + 1e-6)).alias("s_mid_to_eve"),
    (c("s_all") / (c("w_all") + 1e-6)).alias("sw_ratio"),
).with_columns((c("s_mid_share") - c("w_mid_share")).alias("mid_share_diff"))
PVF = ["s_mid_share", "w_mid_share", "mid_share_diff", "s_mid_to_night", "s_mid_to_eve", "sw_ratio", "corr_mid_temp"]
hh = g.join(f.select("Household_ID", "pv_known").unique(), on="Household_ID", how="right").to_pandas().set_index("Household_ID")
hh = hh[~hh.index.duplicated()]
lab = hh[hh.pv_known.notna() & hh[PVF].notna().any(axis=1)]
Xl, yl = lab[PVF].values, lab.pv_known.astype(int).values
clf = make_pipeline(SimpleImputer(strategy="median"),
                    RandomForestClassifier(500, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=2))
oof = cross_val_predict(clf, Xl, yl, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
cls_res = {"n_labelled": int(len(yl)), "pv_share": float(yl.mean()), "cv_auc": float(roc_auc_score(yl, oof)),
           "cv_accuracy": float(accuracy_score(yl, oof > 0.5))}
log("PV classifier (5-fold CV on labelled):", cls_res)
clf.fit(Xl, yl)
hh["pv_prob"] = np.nan
has_feat = hh[PVF].notna().any(axis=1)
hh.loc[has_feat, "pv_prob"] = clf.predict_proba(hh.loc[has_feat, PVF].values)[:, 1]
hh.loc[lab.index, "pv_prob"] = oof  # out-of-fold probability for labelled households
hh["pv_final"] = np.where(hh.pv_known.notna(), hh.pv_known, (hh.pv_prob > 0.5).astype(float))
hh.loc[hh.pv_known.isna() & hh.pv_prob.isna(), "pv_final"] = 0.0  # no training history: default non-PV
hh["pv_source"] = np.where(hh.pv_known.notna(), "known_flag", np.where(hh.pv_prob.notna(), "inferred", "default_no_history"))
hh["confident"] = (hh.pv_prob.sub(0.5).abs() >= 0.25) | (hh.pv_source == "known_flag")
log("group sizes:", hh.groupby(["pv_source", "pv_final"]).size().to_dict(),
    "| inferred & not confident:", int(((hh.pv_source == "inferred") & ~hh.confident).sum()))
hh.reset_index()[["Household_ID", "pv_known", "pv_prob", "pv_final", "pv_source"]].to_csv(f"{out}/pv_groups_{tag}.csv", index=False)
gmap = pl.from_pandas(hh.reset_index()[["Household_ID", "pv_final", "pv_source"]]).with_columns(
    c("Household_ID").cast(pl.Int64), c("pv_final").cast(pl.Int8))
train, test = train.join(gmap, on="Household_ID", how="left"), test.join(gmap, on="Household_ID", how="left")

# ------------------------------------------------------------------ models
tr_all = train.drop_nulls(["lag_48", "roll7_mean"])
cut_v = tr_all["ts"].sort()[int(tr_all.height * 0.9)]
PARAMS = dict(objective="regression", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=2, verbose=-1, seed=0,
              deterministic=True, force_row_wise=True)


def fit(tr, feats, n_train, label):
    a, b = tr.filter(c("ts") < cut_v), tr.filter(c("ts") >= cut_v)
    a, b = a.sample(n=min(n_train, a.height), seed=0), b.sample(n=min(250_000, b.height), seed=0)
    X = lambda d: d.select(feats).cast(pl.Float32).to_numpy()  # noqa: E731
    dtr, dva = lgb.Dataset(X(a), a["kwh"].to_numpy(), free_raw_data=False), None
    dva = lgb.Dataset(X(b), b["kwh"].to_numpy(), reference=dtr)
    m = lgb.train(PARAMS, dtr, 600, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False)])
    log(f"{label}: trained on {a.height:,} rows, best_iter={m.best_iteration}, valid L2={m.best_score['valid_0']['l2']:.4f}")
    return m


def predict(m, d, feats):
    return np.clip(m.predict(d.select(feats).cast(pl.Float32).to_numpy(), num_threads=2), 0, None)


N = 2_000_000
m_global = fit(tr_all, FEATS, N, "L0 global")
imp = sorted(zip(FEATS, m_global.feature_importance("gain")), key=lambda x: -x[1])[:8]
log("top features:", [(k, round(v / sum(i[1] for i in imp), 3)) for k, v in imp])
test = test.with_columns(pl.Series("p_global", predict(m_global, test, FEATS)))

FEATS_PV = FEATS + ["pv_final"]
m_flag = fit(tr_all, FEATS_PV, N, "L1 global+PV-flag")
test = test.with_columns(pl.Series("p_flag", predict(m_flag, test, FEATS_PV)))

p_group = np.full(test.height, np.nan)
for grp in (0, 1):
    mg = fit(tr_all.filter(c("pv_final") == grp), FEATS, N // 2, f"L1 group model pv={grp}")
    idx = np.where(test["pv_final"].to_numpy() == grp)[0]
    p_group[idx] = predict(mg, test[idx], FEATS)
test = test.with_columns(pl.Series("p_group", p_group), c("lag_168").alias("p_naive168"),
                         c("samehour_mean7").alias("p_mean7"), c("lag_48").alias("p_naive48"))

# ------------------------------------------------------------------ evaluation
MODELS = {"naive_168h (same hour last week)": "p_naive168", "naive_48h": "p_naive48",
          "mean_same_hour_7d": "p_mean7", "L0 LightGBM (global)": "p_global",
          "L1 LightGBM + PV flag": "p_flag", "L1 group models (PV / non-PV)": "p_group"}
pdf = test.select("Household_ID", "ts", "kwh", "pv_final", "pv_source", *MODELS.values()).to_pandas()
pdf["ldate"] = pdf.ts.dt.tz_convert("Europe/Zurich").dt.date


def score(d):
    r = {}
    for name, col in MODELS.items():
        e, y = d[col] - d.kwh, d.kwh
        pt = d.groupby("ts")[["kwh", col]].sum()          # portfolio (all households in the subset) per hour
        pd_ = d.groupby("ldate")[["kwh", col]].sum()      # portfolio per day
        r[name] = {"MAE_kWh": e.abs().mean(), "RMSE_kWh": float(np.sqrt((e ** 2).mean())),
                   "WAPE_household_hour": e.abs().sum() / y.sum(), "bias": e.sum() / y.sum(),
                   "WAPE_portfolio_hour": (pt[col] - pt.kwh).abs().sum() / pt.kwh.sum(),
                   "WAPE_portfolio_day": (pd_[col] - pd_.kwh).abs().sum() / pd_.kwh.sum()}
    return pd.DataFrame(r).T


subsets = {"all households": pdf, "PV households": pdf[pdf.pv_final == 1], "non-PV households": pdf[pdf.pv_final == 0],
           "known-flag PV only": pdf[(pdf.pv_final == 1) & (pdf.pv_source == "known_flag")],
           "known-flag non-PV only": pdf[(pdf.pv_final == 0) & (pdf.pv_source == "known_flag")]}
res = {k: score(v) for k, v in subsets.items()}
pd.options.display.float_format = "{:.3f}".format
for k, v in res.items():
    log(f"== {k}: {v.shape[0]} models, {len(subsets[k]):,} rows, {subsets[k].Household_ID.nunique()} households")
    print(v.to_string(), flush=True)
pd.concat(res, names=["subset", "model"]).to_csv(f"{out}/metrics_{tag}.csv")

# month-by-month breakdown (portfolio and household-hour WAPE), overall and per PV group
pdf["month"] = pdf.ts.dt.tz_convert("Europe/Zurich").dt.month
mrows = []
for (mth, grp), d in list(pdf.groupby(["month", "pv_final"])) + [((m, "all"), d) for m, d in pdf.groupby("month")]:
    for name, col in MODELS.items():
        pt = d.groupby("ts")[["kwh", col]].sum()
        mrows.append({"month": mth, "group": grp, "model": name, "n_households": d.Household_ID.nunique(),
                      "WAPE_household_hour": (d[col] - d.kwh).abs().sum() / d.kwh.sum(),
                      "WAPE_portfolio_hour": (pt[col] - pt.kwh).abs().sum() / pt.kwh.sum()})
by_month = pd.DataFrame(mrows)
by_month.to_csv(f"{out}/metrics_by_month_{tag}.csv", index=False)
for g in ["all", 1, 0]:
    t = by_month[by_month.group == g].pivot(index="month", columns="model", values="WAPE_portfolio_hour")[list(MODELS)]
    log(f"== portfolio WAPE (hourly) by month, group={g}")
    print(t.to_string(), flush=True)
json.dump({"classifier": cls_res, "train_end": str(train_end), "test": [str(test_start), str(test_end)],
           "test_rows": int(test.height), "households": int(pdf.Household_ID.nunique())},
          open(f"{out}/summary_{tag}.json", "w"), indent=1)

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
show = ["naive_168h (same hour last week)", "mean_same_hour_7d", "L0 LightGBM (global)", "L1 group models (PV / non-PV)"]
for a, k in zip(ax[:2], ["PV households", "non-PV households"]):
    pt = pdf[pdf.pv_final == (1 if k.startswith("PV") else 0)].groupby("ts")[["kwh"] + [MODELS[s] for s in show]].sum()
    wk = pt.iloc[len(pt) // 2: len(pt) // 2 + 24 * 7]
    a.plot(wk.index, wk.kwh, "k", lw=2, label="actual")
    for s in show[1:]:
        a.plot(wk.index, wk[MODELS[s]], lw=1, label=s)
    a.set_title(f"{k}: portfolio load, one test week ({tag})"); a.set_ylabel("kWh per hour"); a.tick_params(axis="x", rotation=30)
ax[0].legend(fontsize=7)
sub = res["PV households"]["WAPE_portfolio_hour"], res["non-PV households"]["WAPE_portfolio_hour"]
x = np.arange(len(MODELS)); ax[2].bar(x - .2, sub[0].values, .4, label="PV"); ax[2].bar(x + .2, sub[1].values, .4, label="non-PV")
ax[2].set_xticks(x); ax[2].set_xticklabels([m.replace(" (", "\n(") for m in MODELS], rotation=60, ha="right", fontsize=7)
ax[2].set_ylabel("portfolio WAPE (hourly)"); ax[2].legend(); ax[2].set_title("Day-ahead error by group")
plt.tight_layout(); plt.savefig(f"{out}/overview_{tag}.png", dpi=130)

# full test-period view: daily load per household (actual vs forecasts) + monthly error curves
fig, ax = plt.subplots(2, 2, figsize=(17, 9), gridspec_kw={"width_ratios": [2.2, 1]})
for r, (k, g) in enumerate([("PV households", 1), ("non-PV households", 0)]):
    d = pdf[pdf.pv_final == g]
    daily = d.groupby("ldate")[["kwh", "p_mean7", "p_global", "p_group"]].sum()
    nh = d.groupby("ldate").Household_ID.nunique()
    full = (d.groupby("ldate").size() / nh) >= 20          # drop days with < 20 hours of data per household
    daily, nh = daily[full], nh[full]
    daily = daily.div(nh, axis=0)
    daily.index = pd.to_datetime(daily.index)
    a = ax[r, 0]
    a.plot(daily.index, daily.kwh, "k", lw=1.4, label="actual")
    a.plot(daily.index, daily.p_mean7, lw=0.8, alpha=0.8, label="mean_same_hour_7d")
    a.plot(daily.index, daily.p_global, lw=0.9, label="L0 LightGBM (global)")
    a.plot(daily.index, daily.p_group, lw=0.9, label="L1 group models")
    a.set_title(f"{k}: daily load per household over the whole test period ({tag})"); a.set_ylabel("kWh per household per day")
    a.legend(fontsize=8, ncol=4)
    b = ax[r, 1]
    for name in ["naive_168h (same hour last week)", "mean_same_hour_7d", "L0 LightGBM (global)",
                 "L1 LightGBM + PV flag", "L1 group models (PV / non-PV)"]:
        t = by_month[(by_month.group == g) & (by_month.model == name)].sort_values("month")
        b.plot(t.month, t.WAPE_portfolio_hour, marker="o", label=name)
    b.set_xticks(range(1, 13)); b.set_xlabel("month"); b.set_ylabel("portfolio WAPE (hourly)")
    b.set_title(f"{k}: error by month")
ax[0, 1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{out}/fullperiod_{tag}.png", dpi=130)
log("done")
