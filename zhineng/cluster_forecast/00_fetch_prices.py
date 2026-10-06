"""Step 0: download real German day-ahead prices (bidding zone DE-LU, hourly, EUR/MWh)
from SMARD (Bundesnetzagentur) -> cache/da_price_hourly.parquet.

SMARD API: index file lists weekly chunk start times; each chunk holds [[ms_utc, price], ...].
"""
import json
import time
import urllib.request

import pandas as pd

from config import CACHE, TEST_END, VAL_START

BASE = "https://www.smard.de/app/chart_data/4169/DE"   # 4169 = day-ahead price DE/LU


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def main():
    start = VAL_START - pd.Timedelta(days=400)          # some history for plots / context
    stamps = get_json(f"{BASE}/index_hour.json")["timestamps"]
    stamps = [t for t in stamps
              if start - pd.Timedelta(days=7) <= pd.Timestamp(t, unit="ms", tz="UTC") < TEST_END]
    rows = []
    for t in stamps:
        rows += get_json(f"{BASE}/4169_DE_hour_{t}.json")["series"]
        time.sleep(0.1)
    s = pd.DataFrame(rows, columns=["ms", "price"]).dropna().drop_duplicates("ms")
    s.index = pd.to_datetime(s["ms"], unit="ms", utc=True)
    s = s["price"].sort_index().loc[start:TEST_END - pd.Timedelta(hours=1)]
    idx = pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC")
    s = s.reindex(idx).interpolate(limit=3)
    s.index.name = "time"
    s.to_frame("da_price").to_parquet(CACHE / "da_price_hourly.parquet")
    print(f"{len(s)} hourly prices {s.index.min()} .. {s.index.max()}, "
          f"missing={s.isna().sum()}, mean={s.mean():.1f}, min={s.min():.1f}, max={s.max():.1f} EUR/MWh, "
          f"negative hours={int((s < 0).sum())}")


if __name__ == "__main__":
    main()
