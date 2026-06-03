Data processed layout for MuleGuard AI prototype

- `raw.parquet` — ingested raw table from `DataSet.csv`.
- `normalized.parquet` — cleaned and normalized table used for window extraction.
- `windows/` — directory containing per-window `.npy` arrays and `manifest.parquet` describing them.
- `labels.parquet` — account-level split and label mapping (`account_id`, `label`, `split`).
- `cohorts.parquet` — account-level cohort assignment.

Use `ml/data_pipeline.py` CLI to create these artifacts.
