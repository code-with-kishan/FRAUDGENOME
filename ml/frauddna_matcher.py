import os
import numpy as np
import joblib
from ml.dtw_utils import multivariate_dtw
from ml.embedding_index import embedding_candidates


def load_index(index_path: str):
    if not os.path.exists(index_path):
        return None
    data = np.load(index_path, allow_pickle=True)
    index = {
        'pattern_ids': data['pattern_ids'].astype(str).tolist(),
        'means': data['means'],
        'file_paths': data['file_paths'].astype(str).tolist()
    }

    embedding_index_path = os.path.join(os.path.dirname(index_path), 'frauddna_embedding_index.joblib')
    if os.path.exists(embedding_index_path):
        try:
            index['embedding_index'] = joblib.load(embedding_index_path)
        except Exception:
            index['embedding_index'] = None
    else:
        index['embedding_index'] = None
    return index


def _as_2d_float_array(timeseries):
    arr = np.asarray(timeseries, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError('timeseries must be a 2D array-like object')
    return arr


def match_timeseries_prefilter(timeseries, index_path: str, patterns_base_dir: str, top_k: int = 3, prefilter_k: int = 20, dtw_radius: int = 1):
    """Prefilter patterns by Euclidean distance on per-pattern mean, then compute DTW on a shortlist.

    timeseries: numpy array shape (T, F)
    index_path: path to frauddna_index.npz
    patterns_base_dir: base dir where pattern files are located (so file_paths in index are relative to it)
    """
    idx = load_index(index_path)
    if idx is None:
        return []
    means = idx['means']  # shape (P, F)
    if timeseries is None or len(timeseries) == 0:
        return []
    try:
        timeseries_arr = _as_2d_float_array(timeseries)
    except Exception:
        return []

    cand_idxs = []
    if idx.get('embedding_index') is not None:
        try:
            cand_idxs = embedding_candidates(timeseries_arr, idx['embedding_index'], max_candidates=prefilter_k)
        except Exception:
            cand_idxs = []

    if not cand_idxs:
        ts_mean = np.nanmean(timeseries_arr, axis=0)
        # compute euclidean distances to means
        diffs = means - ts_mean[None, :]
        edists = np.linalg.norm(diffs, axis=1)
        prefilter_k = min(max(prefilter_k, top_k), len(edists))
        cand_idxs = np.argsort(edists)[:prefilter_k].tolist()

    if len(cand_idxs) < min(max(prefilter_k, top_k), len(means)):
        ts_mean = np.nanmean(timeseries_arr, axis=0)
        diffs = means - ts_mean[None, :]
        edists = np.linalg.norm(diffs, axis=1)
        fallback_idxs = np.argsort(edists)
        combined = []
        seen = set()
        for candidate_index in list(cand_idxs) + list(fallback_idxs):
            candidate = int(candidate_index)
            if candidate in seen:
                continue
            combined.append(candidate)
            seen.add(candidate)
            if len(combined) >= min(max(prefilter_k, top_k), len(means)):
                break
        cand_idxs = combined

    results = []
    for ci in cand_idxs:
        pid = idx['pattern_ids'][ci]
        ppath = os.path.join(patterns_base_dir, idx['file_paths'][ci])
        if os.path.exists(ppath):
            patt = np.load(ppath)
            try:
                cutoff = results[-1][1] if len(results) >= top_k else None
                d = float(multivariate_dtw(timeseries_arr, patt, radius=dtw_radius, cutoff=cutoff))
                if np.isinf(d):
                    continue
            except Exception:
                try:
                    d = float(np.linalg.norm(timeseries_arr.ravel() - np.asarray(patt, dtype=float).ravel()))
                except Exception:
                    continue
            results.append((pid, d, idx['file_paths'][ci]))
            results = sorted(results, key=lambda x: x[1])[:top_k]
    results = sorted(results, key=lambda x: x[1])[:top_k]
    # map to dicts with metadata if manifest present
    return [{'pattern_id': r[0], 'distance': float(r[1]), 'file_path': r[2]} for r in results]


def match_timeseries_prefilter_with_manifest(timeseries, index_path: str, patterns_base_dir: str, manifest_path: str, top_k: int = 3, prefilter_k: int = 20, dtw_radius: int = 1):
    """Like match_timeseries_prefilter but enriches results with manifest metadata (cluster_id, support_count)."""
    results = match_timeseries_prefilter(timeseries, index_path, patterns_base_dir, top_k=top_k, prefilter_k=prefilter_k, dtw_radius=dtw_radius)
    if not results:
        return []
    if os.path.exists(manifest_path):
        import pandas as pd
        manifest = pd.read_parquet(manifest_path)
    else:
        manifest = None
    enriched = []
    for r in results:
        item = r.copy()
        if manifest is not None:
            row = manifest[manifest['pattern_id'] == r['pattern_id']]
            if not row.empty:
                item['cluster_id'] = int(row.iloc[0].get('cluster_id', -1))
                item['support_count'] = int(row.iloc[0].get('support_count', 0))
        enriched.append(item)
    return enriched
