"""Fraud DNA library builder.

Reads the window manifest produced by `ml.data_pipeline.extract_windows`, gathers
confirmed-mule cases, keeps the activation-inclusive pre-crime window for each
account, and persists both a versioned Fraud DNA library and legacy compatibility
artifacts under `models/`.

Outputs:
- `models/frauddna_library/<version>/` — versioned dataset bundle
- `models/frauddna_patterns/{pattern_id}.npy` — latest compatibility prototype arrays
- `models/frauddna_manifest.parquet` — metadata for the latest library
- `models/frauddna_dataset_version.json` — pointer to the versioned bundle
- `models/canon.npy` — canonical anchor vector (mean of per-mule anchor means)
"""

import json
import os
import uuid
from datetime import datetime, timezone
import shutil
import numpy as np
import pandas as pd
from ml.dtw_utils import multivariate_dtw
from ml.embedding_index import build_embedding_index


def load_manifest(manifest_parquet: str) -> pd.DataFrame:
    return pd.read_parquet(manifest_parquet)


def _load_window(windows_dir: str, file_path: str) -> np.ndarray:
    return np.load(os.path.join(windows_dir, file_path))


def _pairwise_distances(arrs):
    n = len(arrs)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            try:
                d = multivariate_dtw(arrs[i], arrs[j])
            except Exception:
                # fallback to Euclidean on flattened arrays
                d = float(np.linalg.norm(arrs[i].ravel() - arrs[j].ravel()))
            D[i, j] = d
            D[j, i] = d
    return D


def _version_id() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]


def _select_confirmed_mule_windows(manifest: pd.DataFrame) -> pd.DataFrame:
    positives = manifest[manifest['label'] == 1].copy()
    if positives.empty:
        raise ValueError('No positive windows found in manifest')

    if 'activation_timestamp' not in positives.columns:
        positives['activation_timestamp'] = positives['window_end']

    positives['window_end'] = pd.to_datetime(positives['window_end'])
    positives['window_start'] = pd.to_datetime(positives['window_start'])
    positives['activation_timestamp'] = pd.to_datetime(positives['activation_timestamp'])

    confirmed = (
        positives.sort_values(['account_id', 'window_end', 'window_start'])
        .groupby('account_id', as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return confirmed


def _write_versioned_bundle(version_dir: str, manifest_out: pd.DataFrame, arrs, rows):
    os.makedirs(version_dir, exist_ok=True)
    version_patterns_dir = os.path.join(version_dir, 'frauddna_patterns')
    os.makedirs(version_patterns_dir, exist_ok=True)

    for row, arr in zip(rows, arrs):
        np.save(os.path.join(version_patterns_dir, row['version_file_path'].split('/')[-1]), arr)

    manifest_out.to_parquet(os.path.join(version_dir, 'frauddna_manifest.parquet'), index=False)
    return version_patterns_dir


def build_library(manifest_parquet: str, windows_dir: str, out_dir: str, n_clusters: int = None):
    manifest = load_manifest(manifest_parquet)
    confirmed = _select_confirmed_mule_windows(manifest)

    rows = []
    arrs = []
    per_window_means = []

    os.makedirs(out_dir, exist_ok=True)
    legacy_patterns_dir = os.path.join(out_dir, 'frauddna_patterns')
    os.makedirs(legacy_patterns_dir, exist_ok=True)

    library_version = _version_id()
    version_dir = os.path.join(out_dir, 'frauddna_library', library_version)
    version_patterns_dir = os.path.join(version_dir, 'frauddna_patterns')
    os.makedirs(version_patterns_dir, exist_ok=True)

    for _, row in confirmed.iterrows():
        arr = _load_window(windows_dir, row['file_path'])
        pattern_id = str(row['pattern_id'])
        outname = f'{pattern_id}.npy'
        legacy_path = os.path.join(legacy_patterns_dir, outname)
        version_path = os.path.join(version_patterns_dir, outname)
        np.save(legacy_path, arr)
        shutil.copy2(legacy_path, version_path)

        if arr.size == 0:
            mv = np.zeros((arr.shape[1],), dtype=float) if arr.ndim == 2 else np.array([0.0])
        else:
            mv = np.nanmean(arr, axis=0)

        arrs.append(arr)
        per_window_means.append(mv)
        rows.append({
            'account_id': row['account_id'],
            'pattern_id': pattern_id,
            'support_count': 1,
            'cluster_id': 0,
            'window_start': row['window_start'],
            'window_end': row['window_end'],
            'activation_timestamp': row['activation_timestamp'],
            'source_file': row['file_path'],
            'file_path': os.path.join('frauddna_patterns', outname),
            'version_file_path': os.path.join('frauddna_patterns', outname),
            'library_version': library_version,
            'library_root': os.path.join('frauddna_library', library_version),
            'prototype_type': 'confirmed_mule_precrime'
        })

    manifest_out = pd.DataFrame(rows)
    manifest_out_path = os.path.join(out_dir, 'frauddna_manifest.parquet')
    manifest_out.to_parquet(manifest_out_path, index=False)

    canon = np.nanmean(np.vstack(per_window_means), axis=0)
    canon_path = os.path.join(out_dir, 'canon.npy')
    np.save(canon_path, canon)

    means_stack = np.vstack(per_window_means)
    pattern_ids = np.array([row['pattern_id'] for row in rows])
    file_paths = np.array([row['file_path'] for row in rows])
    index_path = os.path.join(out_dir, 'frauddna_index.npz')
    np.savez(index_path, pattern_ids=pattern_ids, means=means_stack, file_paths=file_paths)

    embedding_index = build_embedding_index(
        arrs,
        pattern_ids=pattern_ids.tolist(),
        file_paths=file_paths.tolist(),
        target_steps=16,
        max_components=16,
        n_tables=4,
        n_bits=12,
        random_state=42,
    )
    embedding_index_path = os.path.join(out_dir, 'frauddna_embedding_index.joblib')
    try:
        import joblib
        joblib.dump(embedding_index, embedding_index_path)
    except Exception:
        embedding_index_path = None

    dataset_version = {
        'library_version': library_version,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'manifest_path': os.path.basename(manifest_out_path),
        'canon_path': os.path.basename(canon_path),
        'index_path': os.path.basename(index_path),
        'pattern_count': int(len(rows)),
        'confirmed_mule_count': int(len(rows)),
        'prototype_type': 'confirmed_mule_precrime',
        'embedding_index_path': os.path.basename(embedding_index_path) if embedding_index_path else None,
    }
    dataset_version_path = os.path.join(out_dir, 'frauddna_dataset_version.json')
    with open(dataset_version_path, 'w') as f:
        json.dump(dataset_version, f, indent=2)

    version_manifest_path = os.path.join(version_dir, 'frauddna_manifest.parquet')
    manifest_out.to_parquet(version_manifest_path, index=False)
    with open(os.path.join(version_dir, 'dataset_version.json'), 'w') as f:
        json.dump(dataset_version, f, indent=2)
    np.save(os.path.join(version_dir, 'canon.npy'), canon)
    np.savez(os.path.join(version_dir, 'frauddna_index.npz'), pattern_ids=pattern_ids, means=means_stack, file_paths=file_paths)
    if embedding_index_path and os.path.exists(embedding_index_path):
        shutil.copy2(embedding_index_path, os.path.join(version_dir, 'frauddna_embedding_index.joblib'))

    return manifest_out_path, canon_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--windows', required=True)
    parser.add_argument('--out', default='models')
    parser.add_argument('--n-clusters', type=int, default=None)
    args = parser.parse_args()
    m_out, canon = build_library(args.manifest, args.windows, args.out, n_clusters=args.n_clusters)
    print('FraudDNA manifest:', m_out)
    print('Canon saved:', canon)
