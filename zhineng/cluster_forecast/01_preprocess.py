"""Step 1: 15-min household data -> hourly wide table; hourly weather tables.

Outputs (cache/):
  load_hourly.parquet   index = UTC hour, columns = Household_ID, values = kWh/h
  temp_hourly.parquet   index = UTC hour, columns = Weather_ID, values = degC
  sun_hourly.parquet    index = UTC hour, column 'sun' = regional mean sunshine duration
"""
import time

import numpy as np
import pandas as pd

from config import CACHE, METER_DIR, MIN_QUARTERS_PER_HOUR, TEST_END, WEATHER_DIR


def household_hourly(path):
    df = pd.read_csv(path, sep=";", usecols=["Timestamp", "kWh_received_Total"])
    ts = pd.to_datetime(df["Timestamp"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", utc=True)
    y = pd.to_numeric(df["kWh_received_Total"], errors="coerce")
    ok = y.notna() & (y >= 0)
    s = pd.Series(y[ok].to_numpy(), index=ts[ok].dt.floor("h"))
    g = s.groupby(level=0).agg(["sum", "count"])
    g = g[g["count"] >= MIN_QUARTERS_PER_HOUR]
    # scale hours with one missing quarter-hour up to a full hour
    return (g["sum"] * 4.0 / g["count"]).astype("float32")


def main():
    t0 = time.time()
    files = sorted(METER_DIR.glob("*.csv"))
    series = {}
    for i, f in enumerate(files, 1):
        series[f.stem] = household_hourly(f)
        if i % 50 == 0:
            print(f"  {i}/{len(files)} households ({time.time() - t0:.0f}s)")

    start = min(s.index.min() for s in series.values())
    idx = pd.date_range(start, TEST_END, freq="h", inclusive="left", tz="UTC")
    load = pd.DataFrame({k: v.reindex(idx) for k, v in series.items()}, index=idx)
    load.index.name = "time"
    load.to_parquet(CACHE / "load_hourly.parquet")
    print(f"load_hourly: {load.shape}, {np.isfinite(load.to_numpy()).mean():.1%} filled")

    temps, suns = {}, {}
    for f in sorted(WEATHER_DIR.glob("*.csv")):
        w = pd.read_csv(f, sep=";")
        w.index = pd.to_datetime(w["Timestamp"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", utc=True)
        w = w[~w.index.duplicated()].reindex(idx)
        temps[f.stem] = w["Temperature_avg_hourly"].interpolate(limit=6)
        sun = w["Sunshine_duration_hourly"]
        if sun.notna().mean() > 0.5:          # 3 stations have no sunshine signal
            suns[f.stem] = sun
    temp = pd.DataFrame(temps, index=idx)
    temp = temp.apply(lambda c: c.fillna(temp.mean(axis=1)))   # gaps -> regional mean
    sun = pd.DataFrame({"sun": pd.DataFrame(suns).mean(axis=1).interpolate(limit=6).fillna(0.0)})
    temp.index.name = sun.index.name = "time"
    temp.to_parquet(CACHE / "temp_hourly.parquet")
    sun.to_parquet(CACHE / "sun_hourly.parquet")
    print(f"weather: temp stations={list(temp.columns)}, sunshine from {list(suns)}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
