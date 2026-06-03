import os
import tempfile
import json
import pandas as pd
import joblib
from ml.data_pipeline import ingest, normalize, extract_windows
from ml.frauddna import build_library


def test_build_frauddna():
    tmpdir = tempfile.mkdtemp()
    # reuse the small synthetic dataset
    raw = os.path.join(tmpdir, 'data.csv')
    # create via data pipeline test helper
    from tests.test_data_pipeline import _make_sample_csv
    _make_sample_csv(raw)
    proc_dir = os.path.join(tmpdir, 'processed')
    ingest(raw, proc_dir)
    norm = os.path.join(proc_dir, 'normalized.parquet')
    normalize(os.path.join(proc_dir, 'raw.parquet'), norm)
    win_dir = os.path.join(proc_dir, 'windows')
    os.makedirs(win_dir, exist_ok=True)
    manifest = extract_windows(norm, win_dir, window_days=14, stride_days=7)
    out_models = os.path.join(tmpdir, 'models')
    os.makedirs(out_models, exist_ok=True)
    frauddna_manifest, canon = build_library(manifest, win_dir, out_models, n_clusters=1)
    assert os.path.exists(frauddna_manifest)
    assert os.path.exists(canon)
    dataset_version_path = os.path.join(out_models, 'frauddna_dataset_version.json')
    assert os.path.exists(dataset_version_path)
    with open(dataset_version_path, 'r') as f:
        dataset_version = json.load(f)
    version_manifest_path = os.path.join(out_models, 'frauddna_library', dataset_version['library_version'], 'frauddna_manifest.parquet')
    assert os.path.exists(version_manifest_path)
    embedding_index_path = os.path.join(out_models, 'frauddna_embedding_index.joblib')
    assert os.path.exists(embedding_index_path)
    embedding_index = joblib.load(embedding_index_path)
    assert embedding_index['hash_tables']
    manifest_df = pd.read_parquet(frauddna_manifest)
    assert manifest_df['library_version'].nunique() == 1
    assert set(manifest_df['prototype_type']) == {'confirmed_mule_precrime'}
    assert all(pd.to_datetime(manifest_df['window_start']) <= pd.to_datetime(manifest_df['activation_timestamp']))
    assert all(pd.to_datetime(manifest_df['activation_timestamp']) <= pd.to_datetime(manifest_df['window_end']))
    assert dataset_version['pattern_count'] == len(manifest_df)
