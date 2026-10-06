"""Run the full pipeline: preprocess -> cluster -> forecast -> interpret."""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
fresh = "--fresh" in sys.argv
cached = {"00_fetch_prices.py": "da_price_hourly.parquet", "01_preprocess.py": "load_hourly.parquet"}

for step in ["00_fetch_prices.py", "01_preprocess.py", "02_cluster.py", "03_select_k.py",
             "04_forecast.py", "05_interpret.py", "06_k_selection_plots.py", "07_level2_business.py", "08_level3_uncertainty.py", "09_slide_figures.py", "10_build_pptx.py"]:
    if step in cached and (HERE / "cache" / cached[step]).exists() and not fresh:
        print(f"== skipping {step} (cache exists; pass --fresh to rebuild)")
        continue
    print(f"\n== {step}")
    sys.argv = [step]
    runpy.run_path(str(HERE / step), run_name="__main__")
