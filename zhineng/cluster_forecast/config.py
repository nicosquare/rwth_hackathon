"""Shared settings for the clustering + cluster-based forecasting pipeline."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
METER_DIR = DATA / "15min"
META_DIR = DATA / "smart_meter_meta_data"
WEATHER_DIR = DATA / "weather_data_hourly"

CACHE = ROOT / "cache"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
for d in (CACHE, OUT, FIG):
    d.mkdir(parents=True, exist_ok=True)

# ---- Temporal split (UTC), 80/20 by time span ----
#   sub-train  [start, VAL_START)       -> selection stage: clustering + model fit
#   validation [VAL_START, TRAIN_END)   -> choose K only
#   test       [TRAIN_END, TEST_END)    -> final report only (models refit on all data < TRAIN_END)
VAL_START = pd.Timestamp("2022-09-15 00:00", tz="UTC")
TRAIN_END = pd.Timestamp("2023-03-15 00:00", tz="UTC")   # exclusive; = 80% of the time span
TEST_END = pd.Timestamp("2024-02-28 00:00", tz="UTC")    # exclusive
LOCAL_TZ = "Europe/Berlin"                                # calendar features

# ---- Household eligibility ----
MIN_TRAIN_DAYS = 90          # min. number of complete train days to be clustered
MIN_QUARTERS_PER_HOUR = 3    # hourly value kept if >= 3 of 4 quarter-hours exist

# ---- Clustering ----
K_LIST = [2, 3, 4, 5, 6]
SEED = 42

# ---- Forecasting ----
LAGS_H = [24, 48, 168]       # info available up to midnight of D-1
MIN_ACTIVE_PER_CLUSTER = 3   # skip training hours where a cluster has < 3 active households

# ---- Business evaluation (real German DA prices from SMARD; intraday penalty ASSUMED) ----
# shortfall bought intraday at  DA + SHORT_FIXED   + SHORT_PROP   * |DA|
# surplus  sold   intraday at  DA - SURPLUS_FIXED - SURPLUS_PROP * |DA|
SHORT_FIXED, SHORT_PROP = 40.0, 0.25       # EUR/MWh, share of |DA|
SURPLUS_FIXED, SURPLUS_PROP = 30.0, 0.25
PARSIMONY_TOL = 0.005    # choose the smallest K whose validation cost is within 0.5% of the best

# ---- Level 3: uncertainty (rolling empirical quantiles of relative residuals) ----
UNC_WINDOW_DAYS = 56     # residual history per forecast day (8 weeks, known by midnight of D-1)
UNC_HOUR_POOL = 1        # pool residuals of hour h +/- 1 (more samples for tail quantiles)
UNC_TAUS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

CLUSTER_FEATURES = [
    "temp_slope_rel",        # heating sensitivity: % of mean daily energy per -1 degC (T < 15 degC)
    "temp_r2",               # how much of daily energy variance temperature explains
    "summer_winter_ratio",   # base-load share: summer daily energy / winter daily energy
    "pv_midday_ratio",       # summer midday load / summer morning+evening load (PV dip)
    "pv_sun_corr",           # corr(summer midday load, sunshine) -> negative for PV owners
    "share_night",           # 00-05 local
    "share_morning",         # 06-09
    "share_midday",          # 10-16
    "share_evening",         # 17-23
]
