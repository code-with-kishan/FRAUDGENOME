import os
import tempfile
from ml.data_pipeline import ingest, normalize, extract_windows
from ml.frauddna import build_library
from ml.frauddna_matcher import load_index, match_timeseries_prefilter
from ml.dtw_utils import dtw_distance
import numpy as np


def test_matcher_prefilter():
    tmpdir = tempfile.mkdtemp()
    raw = os.path.join(tmpdir, 'data.csv')
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
    build_library(manifest, win_dir, out_models, n_clusters=1)
    index_path = os.path.join(out_models, 'frauddna_index.npz')
    idx = load_index(index_path)
    assert idx is not None
    assert idx['embedding_index'] is not None
    # craft a timeseries similar to the positive pattern
    # load one pattern and use it as timeseries
    import glob
    patt_files = glob.glob(os.path.join(out_models, 'frauddna_patterns','*.npy'))
    assert len(patt_files) > 0
    patt = np.load(patt_files[0])
    ts = patt.tolist()
    res = match_timeseries_prefilter(ts, index_path, out_models, top_k=3, prefilter_k=5)
    assert isinstance(res, list)


def test_dtw_distance_prunes_when_cutoff_is_too_small():
    a = np.array([0.0, 0.0, 0.0, 0.0])
    b = np.array([10.0, 10.0, 10.0, 10.0])
    distance = dtw_distance(a, b, radius=1, cutoff=1.0)
    assert np.isinf(distance)
