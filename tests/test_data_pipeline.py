import os
import tempfile
import numpy as np
import pandas as pd
from ml.data_pipeline import ingest, normalize, extract_windows, split_accounts, create_cohorts


def _make_sample_csv(path):
    # create simple dataset with two accounts, one positive
    rows = []
    idx = pd.date_range('2024-01-01', periods=60, freq='D')
    for t in idx:
        rows.append({'account_id': 'A', 'timestamp': t, 'F3924': 0, 'F321': 0, 'F3836': 1.0, 'F2082': 0})
    # positive account B with label late in series
    idx2 = pd.date_range('2024-01-01', periods=60, freq='D')
    for i, t in enumerate(idx2):
        label = 1 if i == 50 else 0
        rows.append({'account_id': 'B', 'timestamp': t, 'F3924': label, 'F321': float(i%3), 'F3836': float(i%5), 'F2082': float((i+1)%4)})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_pipeline_end_to_end():
    tmpdir = tempfile.mkdtemp()
    raw = os.path.join(tmpdir, 'data.csv')
    _make_sample_csv(raw)
    proc_dir = os.path.join(tmpdir, 'processed')
    os.makedirs(proc_dir, exist_ok=True)
    raw_parquet = ingest(raw, proc_dir)
    norm = os.path.join(proc_dir, 'normalized.parquet')
    normalize(raw_parquet, norm)
    win_dir = os.path.join(proc_dir, 'windows')
    os.makedirs(win_dir, exist_ok=True)
    manifest = extract_windows(norm, win_dir, window_days=14, stride_days=7)
    assert os.path.exists(manifest)
    manifest_df = pd.read_parquet(manifest)
    positives = manifest_df[manifest_df['label'] == 1].copy()
    assert not positives.empty
    positives['window_end'] = pd.to_datetime(positives['window_end'])
    positives['window_start'] = pd.to_datetime(positives['window_start'])
    positives['activation_timestamp'] = pd.to_datetime(positives['activation_timestamp'])
    latest_per_account = positives.sort_values(['account_id', 'window_end', 'window_start']).groupby('account_id', as_index=False).tail(1)
    assert all(latest_per_account['window_start'] <= latest_per_account['activation_timestamp'])
    assert all(latest_per_account['activation_timestamp'] <= latest_per_account['window_end'])
    splits = os.path.join(proc_dir, 'labels.parquet')
    split_accounts(norm, splits, train_frac=0.6, val_frac=0.2)
    cohorts = os.path.join(proc_dir, 'cohorts.parquet')
    create_cohorts(norm, cohorts, n_clusters=2)
    assert os.path.exists(cohorts)
