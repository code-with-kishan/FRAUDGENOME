"""Learned time-series embeddings with LSH lookup helpers.

The index is designed for the small FraudDNA pattern library already produced by
`ml.frauddna.build_library`:
- each multivariate window is summarized into a fixed-length feature vector
- a PCA projection turns those summaries into compact learned embeddings
- multiple random-hyperplane LSH tables provide a fast candidate shortlist

The shortlist is intentionally approximate and is meant to prefilter candidates
before the existing DTW rerank step.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.decomposition import PCA

_EPS = 1e-8


def _as_2d_float_array(timeseries: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(timeseries, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError('timeseries must be a 2D array-like object')
    return arr


def _interp_feature(values: np.ndarray, target_steps: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(target_steps, dtype=float)
    if len(values) == 1:
        return np.repeat(float(values[0]), target_steps)

    source_x = np.linspace(0.0, 1.0, num=len(values), dtype=float)
    target_x = np.linspace(0.0, 1.0, num=target_steps, dtype=float)
    return np.interp(target_x, source_x, values.astype(float))


def series_feature_vector(timeseries: Sequence[Sequence[float]] | np.ndarray, target_steps: int = 16) -> np.ndarray:
    """Convert a multivariate window into a fixed-length feature vector."""
    arr = _as_2d_float_array(timeseries)
    if arr.size == 0:
        return np.zeros(target_steps, dtype=float)

    features: List[float] = []
    for column_index in range(arr.shape[1]):
        column = np.nan_to_num(arr[:, column_index].astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        features.extend([
            float(np.mean(column)),
            float(np.std(column)),
            float(np.min(column)),
            float(np.max(column)),
            float(np.median(column)),
            float(np.percentile(column, 25)),
            float(np.percentile(column, 75)),
            float(column[0]),
            float(column[-1]),
        ])

        x = np.arange(len(column), dtype=float)
        slope = 0.0
        if len(column) > 1:
            try:
                slope = float(np.polyfit(x, column, 1)[0])
            except Exception:
                slope = 0.0
        features.append(slope)

        features.extend(_interp_feature(column, target_steps).tolist())

    return np.asarray(features, dtype=float)


def _signature_key(bits: np.ndarray) -> str:
    packed = np.packbits(bits.astype(np.uint8), axis=None)
    return packed.tobytes().hex()


def _fit_embedding_projection(feature_matrix: np.ndarray, max_components: int, random_state: int) -> Dict[str, Any]:
    feature_mean = feature_matrix.mean(axis=0)
    feature_scale = feature_matrix.std(axis=0)
    feature_scale = np.where(feature_scale < _EPS, 1.0, feature_scale)
    normalized = (feature_matrix - feature_mean) / feature_scale

    if normalized.shape[0] > 1 and normalized.shape[1] > 1:
        component_count = min(max_components, normalized.shape[0] - 1, normalized.shape[1])
    else:
        component_count = min(max_components, normalized.shape[1])

    if normalized.shape[0] > 1 and component_count >= 1:
        pca = PCA(n_components=component_count, random_state=random_state)
        embeddings = pca.fit_transform(normalized)
        projection = pca.components_.astype(float)
        projection_mean = pca.mean_.astype(float)
    else:
        projection = np.empty((0, normalized.shape[1]), dtype=float)
        projection_mean = np.zeros(normalized.shape[1], dtype=float)
        embeddings = normalized[:, :component_count] if component_count > 0 else normalized[:, :1]

    embeddings = np.asarray(embeddings, dtype=float)
    embedding_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(embedding_norm, 1.0)

    return {
        'feature_mean': feature_mean,
        'feature_scale': feature_scale,
        'pca_components': projection,
        'pca_mean': projection_mean,
        'embeddings': embeddings,
    }


def build_embedding_index(
    timeseries_list: Sequence[Sequence[Sequence[float]] | np.ndarray],
    pattern_ids: Sequence[str],
    file_paths: Sequence[str],
    *,
    target_steps: int = 16,
    max_components: int = 16,
    n_tables: int = 4,
    n_bits: int = 12,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Build a learned-embedding + LSH index for fast pattern shortlist lookup."""
    if len(timeseries_list) == 0:
        return {
            'target_steps': int(target_steps),
            'max_components': int(max_components),
            'n_tables': int(n_tables),
            'n_bits': int(n_bits),
            'pattern_ids': [],
            'file_paths': [],
            'features': np.empty((0, 0), dtype=float),
            'embeddings': np.empty((0, 0), dtype=float),
            'feature_mean': np.empty((0,), dtype=float),
            'feature_scale': np.empty((0,), dtype=float),
            'pca_components': np.empty((0, 0), dtype=float),
            'pca_mean': np.empty((0,), dtype=float),
            'projection_matrices': np.empty((0, 0, 0), dtype=float),
            'hash_tables': [],
        }

    feature_matrix = np.vstack([series_feature_vector(timeseries, target_steps=target_steps) for timeseries in timeseries_list])
    projection_info = _fit_embedding_projection(feature_matrix, max_components=max_components, random_state=random_state)
    embeddings = projection_info['embeddings']

    embedding_dim = embeddings.shape[1]
    rng = np.random.default_rng(random_state)
    projection_matrices = rng.normal(size=(n_tables, n_bits, max(1, embedding_dim))).astype(float)

    hash_tables: List[Dict[str, List[int]]] = []
    for table_index in range(n_tables):
        buckets: Dict[str, List[int]] = defaultdict(list)
        table_planes = projection_matrices[table_index]
        for row_index, embedding in enumerate(embeddings):
            bits = np.dot(table_planes, embedding) >= 0.0
            buckets[_signature_key(bits)].append(row_index)
        hash_tables.append(dict(buckets))

    return {
        'target_steps': int(target_steps),
        'max_components': int(max_components),
        'n_tables': int(n_tables),
        'n_bits': int(n_bits),
        'pattern_ids': list(pattern_ids),
        'file_paths': list(file_paths),
        'features': feature_matrix,
        'embeddings': embeddings,
        'feature_mean': projection_info['feature_mean'],
        'feature_scale': projection_info['feature_scale'],
        'pca_components': projection_info['pca_components'],
        'pca_mean': projection_info['pca_mean'],
        'projection_matrices': projection_matrices,
        'hash_tables': hash_tables,
    }


def _transform_embedding(timeseries: Sequence[Sequence[float]] | np.ndarray, embedding_index: Dict[str, Any]) -> np.ndarray:
    features = series_feature_vector(timeseries, target_steps=int(embedding_index['target_steps']))
    feature_mean = np.asarray(embedding_index['feature_mean'], dtype=float)
    feature_scale = np.asarray(embedding_index['feature_scale'], dtype=float)
    if feature_mean.size == 0 or feature_scale.size == 0:
        return np.asarray(features, dtype=float)

    normalized = (features - feature_mean) / np.where(feature_scale < _EPS, 1.0, feature_scale)
    pca_components = np.asarray(embedding_index['pca_components'], dtype=float)
    if pca_components.size > 0:
        pca_mean = np.asarray(embedding_index['pca_mean'], dtype=float)
        embedding = np.dot(normalized - pca_mean, pca_components.T)
    else:
        embedding_dim = min(int(embedding_index['max_components']), normalized.shape[0])
        embedding = normalized[:embedding_dim]

    embedding = np.asarray(embedding, dtype=float)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def embedding_candidates(
    timeseries: Sequence[Sequence[float]] | np.ndarray,
    embedding_index: Dict[str, Any],
    max_candidates: int = 20,
) -> List[int]:
    """Return candidate row indices from the learned embedding LSH index."""
    if not embedding_index:
        return []

    embeddings = np.asarray(embedding_index.get('embeddings', np.empty((0, 0))), dtype=float)
    if embeddings.size == 0:
        return []

    query_embedding = _transform_embedding(timeseries, embedding_index)
    projection_matrices = np.asarray(embedding_index['projection_matrices'], dtype=float)
    candidate_counter: Counter[int] = Counter()

    for table_index, hash_table in enumerate(embedding_index.get('hash_tables', [])):
        table_planes = projection_matrices[table_index]
        bits = np.dot(table_planes, query_embedding) >= 0.0
        bucket = hash_table.get(_signature_key(bits), [])
        candidate_counter.update(bucket)

    seen: set[int] = set()
    ordered_candidates: List[int] = []

    if candidate_counter:
        ranked = sorted(candidate_counter.items(), key=lambda item: (-item[1], item[0]))
        for candidate_index, _ in ranked:
            if candidate_index not in seen:
                ordered_candidates.append(candidate_index)
                seen.add(candidate_index)
            if len(ordered_candidates) >= max_candidates:
                break

    if len(ordered_candidates) < max_candidates:
        distances = np.linalg.norm(embeddings - query_embedding[None, :], axis=1)
        for candidate_index in np.argsort(distances):
            candidate = int(candidate_index)
            if candidate in seen:
                continue
            ordered_candidates.append(candidate)
            seen.add(candidate)
            if len(ordered_candidates) >= max_candidates:
                break

    return ordered_candidates
