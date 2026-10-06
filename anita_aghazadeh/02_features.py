"""Step 2: leakage-safe day-ahead features.

Forecast origin: end of UTC day D-2 (conservative vs. the real gate closure at ~12:00 on D-1).
For a target hour t on day D every consumption feature is built only from data with ts <= D-2 23:00.
Weather at the target hour is the OBSERVED weather, used as a stand-in for a weather forecast.

Usage: python3 -I 02_features.py <repo_root> <work_dir>
"""
import glob
import sys
from datetime import timedelta

import polars as pl

root, work = sys.argv[1], sys.argv[2]
h = pl.read_parquet(f"{work}/hourly.parquet").select(
    "Household_ID", "ts", "kwh", "Group", "Weather_ID", "pv_known").sort("Household_ID", "ts")

# ---- complete hourly grid per household so that shifts are true time shifts ----
rng = h.group_by("Household_ID").agg(pl.col("ts").min().alias("a"), pl.col("ts").max().alias("b"))
grid = (rng.with_columns(ts=pl.datetime_ranges(pl.col("a"), pl.col("b"), "1h", time_zone="UTC"))
           .explode("ts").select("Household_ID", "ts")
           .join(h.select("Household_ID", "ts", "kwh"), on=["Household_ID", "ts"], how="left")
           .sort("Household_ID", "ts"))
print("grid rows", grid.height, " observed share", round(grid["kwh"].is_not_null().mean(), 3))

LAGS = [48, 72, 96, 120, 144, 168, 192]
grid = grid.with_columns(
    *[pl.col("kwh").shift(k).over("Household_ID").alias(f"lag_{k}") for k in LAGS],
    pl.col("kwh").rolling_mean(24, min_samples=18).over("Household_ID").alias("r1"),
    pl.col("kwh").rolling_mean(168, min_samples=72).over("Household_ID").alias("r7"),
    pl.col("kwh").rolling_mean(672, min_samples=240).over("Household_ID").alias("r28"),
)
grid = grid.with_columns(
    pl.mean_horizontal([f"lag_{k}" for k in LAGS]).alias("samehour_mean7"),
    pl.concat_list([f"lag_{k}" for k in LAGS]).list.std().alias("samehour_std7"),
)

# end-of-day rolling stats of day D-2, joined to every hour of day D
eod = (grid.filter(pl.col("ts").dt.hour() == 23)
           .select("Household_ID", (pl.col("ts").dt.date() + timedelta(days=2)).alias("date"),
                   pl.col("r1").alias("lastday_mean"), pl.col("r7").alias("roll7_mean"),
                   pl.col("r28").alias("roll28_mean")))

# ---- weather (hourly) + daily aggregates per station and local day ----
w = pl.concat([pl.read_csv(p, separator=";", infer_schema=False)
               for p in glob.glob(f"{root}/data/weather_data_hourly/*.csv")])
wc = ["Temperature_avg_hourly", "Humidity_avg_hourly", "Sunshine_duration_hourly",
      "WindSpeed_hourly", "Precipitation_total_hourly"]
w = w.with_columns(
    pl.col("Timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S%z", time_zone="UTC").dt.cast_time_unit("us").alias("ts"),
    *[pl.col(c).cast(pl.Float32, strict=False) for c in wc],
).select(["Weather_ID", "ts"] + wc).rename({
    "Temperature_avg_hourly": "temp", "Humidity_avg_hourly": "hum", "Sunshine_duration_hourly": "sun",
    "WindSpeed_hourly": "wind", "Precipitation_total_hourly": "precip"})
w = w.with_columns(pl.col("ts").dt.convert_time_zone("Europe/Zurich").dt.date().alias("ldate"))
wd = w.group_by("Weather_ID", "ldate").agg(
    pl.col("temp").mean().alias("temp_dmean"), pl.col("temp").min().alias("temp_dmin"),
    pl.when(pl.col("sun").count() > 0).then(pl.col("sun").sum()).otherwise(None).alias("sun_dsum"))
w = w.join(wd, on=["Weather_ID", "ldate"], how="left")

# ---- assemble target rows ----
f = (h.join(grid.drop("kwh", "r1", "r7", "r28"), on=["Household_ID", "ts"], how="left")
       .with_columns(pl.col("ts").dt.date().alias("date")).join(eod, on=["Household_ID", "date"], how="left")
       .join(w.drop("ldate"), on=["Weather_ID", "ts"], how="left").drop("date"))
loc = pl.col("ts").dt.convert_time_zone("Europe/Zurich")
f = f.with_columns(
    loc.dt.hour().cast(pl.Int8).alias("hour"), (loc.dt.weekday() - 1).cast(pl.Int8).alias("dow"),
    loc.dt.month().cast(pl.Int8).alias("month"), loc.dt.ordinal_day().cast(pl.Int16).alias("doy"),
)
f = f.with_columns(
    (pl.col("dow") >= 5).cast(pl.Int8).alias("is_weekend"),
    (18.0 - pl.col("temp_dmean")).clip(lower_bound=0).alias("hdd"),
    (pl.col("lastday_mean") / (pl.col("roll28_mean") + 1e-3)).alias("lastday_ratio"),
)
f = f.with_columns([pl.col(c).cast(pl.Float32) for c in f.columns
                    if f.schema[c] == pl.Float64 and c != "kwh"])
print(f.shape)
print(f.null_count().to_dicts()[0])
f.write_parquet(f"{work}/features.parquet")
