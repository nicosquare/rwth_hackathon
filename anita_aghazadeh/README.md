# Day-ahead load forecasting for heat-pump households (Levels 0 and 1)

Author: Anita Aghazadeh. See description.pdf for the one-page write-up.

Run from the repository root (the folder that contains `data/`):

    python 01_prepare.py . work
    python 02_features.py . work
    python 03_models.py work results 2023-01-01T00:00 2023-01-01T00:00 2024-01-01T00:00 year2023

Requires polars, pandas, scikit-learn, lightgbm, matplotlib, pyarrow. Charts and metric tables are in `results/`.
