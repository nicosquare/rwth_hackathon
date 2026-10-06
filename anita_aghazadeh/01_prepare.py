"""Step 1: read all 15-min smart-meter files, aggregate to hourly, attach weather + metadata.

Output: <out_dir>/hourly.parquet with one row per (household, hour) in UTC.
Usage: python3 -I 01_prepare.py <repo_root> <out_dir>
"""
import glob
import os
import sys

import polars as pl

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

frames = []
bad_total = 0
for f in sorted(glob.glob(f"{root}/data/15min/*.csv")):
    d = pl.read_csv(f, separator=";", infer_schema=False,
                    columns=["Household_ID", "Timestamp", "kWh_received_Total"])
    d = d.with_columns(
        pl.col("Household_ID").cast(pl.Int64),
        pl.col("Timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S%z", time_zone="UTC")
          .dt.replace_time_zone("UTC").dt.cast_time_unit("us"),
        pl.col("kWh_received_Total").cast(pl.Float64, strict=False).alias("kwh"),
    ).drop("kWh_received_Total")
    bad_total += d["kwh"].null_count()
    d = d.drop_nulls("kwh")
    # hourly: sum of the four quarter-hours, only if all four are present
    h = (d.sort("Timestamp")
          .group_by_dynamic("Timestamp", every="1h", group_by="Household_ID")
          .agg(pl.col("kwh").sum().alias("kwh"), pl.col("kwh").count().alias("n")))
    frames.append(h.filter(pl.col("n") == 4).drop("n"))

hourly = pl.concat(frames).with_columns(pl.col("kwh").cast(pl.Float32))
print("hourly rows:", hourly.shape, "null/invalid quarter-hours dropped:", bad_total)

# weather: hourly, one file per station
w = pl.concat([pl.read_csv(p, separator=";", infer_schema=False) for p in glob.glob(f"{root}/data/weather_data_hourly/*.csv")])
wcols = ["Temperature_avg_hourly", "Humidity_avg_hourly", "Sunshine_duration_hourly",
         "WindSpeed_hourly", "Precipitation_total_hourly"]
w = w.with_columns(
    pl.col("Timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S%z", time_zone="UTC").dt.cast_time_unit("us"),
    *[pl.col(c).cast(pl.Float32, strict=False) for c in wcols],
).select(["Weather_ID", "Timestamp"] + wcols)

hh = pl.read_csv(f"{root}/data/smart_meter_meta_data/households.csv", separator=";", infer_schema=False)
hh = hh.select(
    pl.col("Household_ID").cast(pl.Int64), "Group", "Weather_ID",
    pl.when(pl.col("Installation_HasPVSystem") == "True").then(1)
      .when(pl.col("Installation_HasPVSystem") == "False").then(0)
      .otherwise(None).cast(pl.Int8).alias("pv_known"),
)
hourly = (hourly.join(hh, on="Household_ID", how="left")
                .join(w, on=["Weather_ID", "Timestamp"], how="left")
                .rename({"Timestamp": "ts"}))
print("rows with missing temperature:", hourly["Temperature_avg_hourly"].null_count())
hourly.write_parquet(f"{out}/hourly.parquet")
print("households:", hourly["Household_ID"].n_unique(), "range:", hourly["ts"].min(), hourly["ts"].max())
