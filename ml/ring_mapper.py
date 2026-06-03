"""Ring topology mapper using behavioral proximity and Louvain community detection."""

import os
from typing import Dict, Iterable, List, Sequence

import networkx as nx
import numpy as np
import pandas as pd

try:
    import community as community_louvain
except Exception:
    community_louvain = None

try:
    from networkx.algorithms.community import louvain_communities
except Exception:
    louvain_communities = None


DEFAULT_ANCHORS: Sequence[str] = ('F321', 'F3836', 'F2082')
LIFECYCLE_STAGES: Sequence[str] = ('Recruiting', 'Active', 'Dispersing', 'Dormant')


def _resolve_anchor_columns(df: pd.DataFrame, anchors: Sequence[str]) -> List[str]:
    return [anchor for anchor in anchors if anchor in df.columns]


def _detect_spike_dates(account_df: pd.DataFrame, feature: str = 'F3836') -> List[pd.Timestamp]:
    if feature not in account_df.columns or account_df.empty:
        return []

    series = account_df[['date', feature]].dropna().sort_values('date')
    if series.empty:
        return []

    values = series[feature].astype(float)
    median = float(values.median())
    mad = float(np.median(np.abs(values - median)))
    if np.isfinite(mad) and mad > 0.0:
        scores = 0.6745 * (values - median) / mad
        spikes = series.loc[scores >= 3.5, 'date'].tolist()
    else:
        threshold = float(values.mean() + max(values.std(ddof=0), 1.0))
        spikes = series.loc[values >= threshold, 'date'].tolist()

    return list(dict.fromkeys(spikes))


def _feature_vector(account_df: pd.DataFrame, anchors: Sequence[str]) -> np.ndarray:
    vector: List[float] = []
    for anchor in anchors:
        values = account_df[anchor].astype(float) if anchor in account_df.columns else pd.Series(dtype=float)
        if values.empty:
            vector.extend([0.0, 0.0, 0.0])
            continue
        vector.append(float(values.mean()))
        vector.append(float(values.std(ddof=0)) if len(values) > 1 else 0.0)
        vector.append(float(values.iloc[-1] - values.iloc[0]) if len(values) > 1 else 0.0)
    return np.asarray(vector, dtype=float)


def _similarity_score(left_vector: np.ndarray, right_vector: np.ndarray) -> float:
    scale = float(np.maximum(np.abs(left_vector).mean() + np.abs(right_vector).mean(), 1.0))
    distance = float(np.linalg.norm(left_vector - right_vector) / scale)
    if not np.isfinite(distance):
        return 0.0
    return float(np.exp(-distance))


def _temporal_sync_score(left_spikes: List[pd.Timestamp], right_spikes: List[pd.Timestamp], sync_hours: int) -> float:
    if not left_spikes or not right_spikes:
        return 0.0

    matched = 0
    for left_spike in left_spikes:
        if any(abs((left_spike - right_spike).total_seconds()) <= sync_hours * 3600 for right_spike in right_spikes):
            matched += 1
    return matched / max(len(left_spikes), len(right_spikes))


def _pairwise_correlation(left_df: pd.DataFrame, right_df: pd.DataFrame, feature: str, corr_window_days: int) -> float:
    if feature not in left_df.columns or feature not in right_df.columns:
        return 0.0

    joined = left_df[['date', feature]].merge(right_df[['date', feature]], on='date', how='inner', suffixes=('_left', '_right'))
    if len(joined) < max(3, corr_window_days):
        return 0.0

    corr = joined[f'{feature}_left'].rolling(window=corr_window_days).corr(joined[f'{feature}_right']).dropna().mean()
    if pd.isna(corr):
        return 0.0
    return float(corr)


def classify_community_stage(community_stats: Dict[str, float]) -> Dict[str, float]:
    members = int(community_stats.get('members', 0))
    if members == 0:
        return {
            'stage': 'Dormant',
            'stage_score': 0.0,
            'members': 0,
            'label_rate': 0.0,
            'recent_activity_rate': 0.0,
            'spike_rate': 0.0,
            'community_density': 0.0,
            'community_weight': 0.0,
        }

    label_rate = float(community_stats.get('label_rate', 0.0))
    recent_activity_rate = float(community_stats.get('recent_activity_rate', 0.0))
    spike_rate = float(community_stats.get('spike_rate', 0.0))
    community_density = float(community_stats.get('community_density', 0.0))
    community_weight = float(community_stats.get('community_weight', 0.0))
    recency_gap = float(community_stats.get('recency_gap', 999.0))

    if label_rate >= 0.25 and recent_activity_rate >= 0.35:
        stage = 'Active'
    elif recent_activity_rate >= 0.5 and spike_rate >= 0.25 and label_rate < 0.2:
        stage = 'Recruiting'
    elif label_rate > 0.0 and recent_activity_rate < 0.35 and recency_gap > 14.0:
        stage = 'Dispersing'
    else:
        stage = 'Dormant'

    stage_score = float(
        np.clip(
            0.45 * label_rate
            + 0.25 * recent_activity_rate
            + 0.15 * spike_rate
            + 0.1 * community_density
            + 0.05 * max(community_weight - 1.0, 0.0),
            0.0,
            1.0,
        )
    )

    return {
        'stage': stage,
        'stage_score': round(stage_score, 6),
        'members': members,
        'label_rate': round(label_rate, 6),
        'recent_activity_rate': round(recent_activity_rate, 6),
        'spike_rate': round(spike_rate, 6),
        'community_density': round(community_density, 6),
        'community_weight': round(community_weight, 6),
        'recency_gap': round(recency_gap, 6),
    }


def _best_partition(graph: nx.Graph) -> Dict[str, int]:
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_edges() == 0:
        return {node: 0 for node in graph.nodes()}

    if community_louvain is not None:
        return {node: int(cluster_id) for node, cluster_id in community_louvain.best_partition(graph, weight='weight').items()}

    if louvain_communities is not None:
        communities = louvain_communities(graph, weight='weight', seed=42)
        partition: Dict[str, int] = {}
        for cluster_id, members in enumerate(communities):
            for node in members:
                partition[node] = int(cluster_id)
        return partition

    return {node: 0 for node in graph.nodes()}


def build_graph(
    normalized_parquet: str,
    out_dir: str,
    anchors: Sequence[str] = DEFAULT_ANCHORS,
    sync_hours: int = 72,
    corr_window_days: int = 7,
    corr_threshold: float = 0.35,
):
    df = pd.read_parquet(normalized_parquet)
    os.makedirs(out_dir, exist_ok=True)

    if 'timestamp' not in df.columns:
        raise ValueError('normalized parquet must include timestamp')
    if 'account_id' not in df.columns:
        raise ValueError('normalized parquet must include account_id')

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.floor('D')

    anchor_columns = _resolve_anchor_columns(df, anchors)
    if not anchor_columns:
        raise ValueError('normalized parquet does not contain any anchor columns')

    if 'F3924' not in df.columns:
        df['F3924'] = 0

    daily = df.groupby(['account_id', 'date'], as_index=False)[anchor_columns + ['F3924']].mean()
    accounts = list(daily['account_id'].drop_duplicates())
    graph = nx.Graph()
    latest_timestamp = df['timestamp'].max()

    account_frames: Dict[str, pd.DataFrame] = {}
    account_vectors: Dict[str, np.ndarray] = {}
    account_spikes: Dict[str, List[pd.Timestamp]] = {}

    for account_id in accounts:
        account_df = daily[daily['account_id'] == account_id].sort_values('date').reset_index(drop=True)
        account_frames[account_id] = account_df
        account_vectors[account_id] = _feature_vector(account_df, anchor_columns)
        account_spikes[account_id] = _detect_spike_dates(account_df)

        label_rate = float(account_df['F3924'].mean()) if len(account_df) else 0.0
        activation_dates = account_df.loc[account_df['F3924'] > 0, 'date'] if len(account_df) else pd.Series(dtype='datetime64[ns]')
        first_seen = account_df['date'].min() if len(account_df) else pd.NaT
        last_seen = account_df['date'].max() if len(account_df) else pd.NaT
        days_since_last_seen = int((latest_timestamp.floor('D') - last_seen).days) if pd.notna(last_seen) else 0

        graph.add_node(
            account_id,
            activity_days=int(account_df['date'].nunique()),
            days_since_last_seen=days_since_last_seen,
            label_rate=float(round(label_rate, 6)),
            activation_count=int(len(activation_dates)),
            spike_count=int(len(account_spikes[account_id])),
            first_seen=first_seen,
            last_seen=last_seen,
        )

    for index, source in enumerate(accounts):
        for target in accounts[index + 1:]:
            similarity = _similarity_score(account_vectors[source], account_vectors[target])
            sync_score = _temporal_sync_score(account_spikes[source], account_spikes[target], sync_hours)
            corr = max(0.0, _pairwise_correlation(account_frames[source], account_frames[target], 'F3836', corr_window_days))
            if corr < corr_threshold:
                corr = 0.0

            combined = 0.6 * similarity + 0.25 * sync_score + 0.15 * corr
            if combined <= 0.0:
                continue

            if combined >= corr_threshold or sync_score > 0.0 or corr > 0.0:
                graph.add_edge(
                    source,
                    target,
                    weight=round(float(combined), 6),
                    similarity=round(float(similarity), 6),
                    sync_score=round(float(sync_score), 6),
                    corr=round(float(corr), 6),
                )

    partition = _best_partition(graph)
    for node, community_id in partition.items():
        graph.nodes[node]['community'] = int(community_id)

    community_rows: List[Dict[str, object]] = []
    for community_id in sorted(set(partition.values()) if partition else {0}):
        members = [node for node in graph.nodes if graph.nodes[node].get('community', 0) == community_id]
        if not members:
            continue

        subgraph = graph.subgraph(members)
        community_density = float(nx.density(subgraph)) if len(members) > 1 else 0.0
        edge_weights = [edge_data.get('weight', 0.0) for _, _, edge_data in subgraph.edges(data=True)]
        community_weight = float(np.mean(edge_weights)) if edge_weights else 0.0

        member_frame = pd.DataFrame([graph.nodes[node] for node in members])
        label_rate = float(member_frame['label_rate'].mean()) if not member_frame.empty else 0.0
        recent_activity_rate = float((member_frame['days_since_last_seen'] <= 14).mean()) if not member_frame.empty else 0.0
        spike_rate = float((member_frame['spike_count'] > 0).mean()) if not member_frame.empty else 0.0
        recency_gap = float(member_frame['days_since_last_seen'].median()) if not member_frame.empty else 999.0

        summary = classify_community_stage(
            {
                'members': len(members),
                'label_rate': label_rate,
                'recent_activity_rate': recent_activity_rate,
                'spike_rate': spike_rate,
                'community_density': community_density,
                'community_weight': community_weight,
                'recency_gap': recency_gap,
            }
        )

        community_rows.append(
            {
                'community_id': int(community_id),
                **summary,
                'member_count': int(len(members)),
            }
        )

        for node in members:
            graph.nodes[node]['community_density'] = round(community_density, 6)
            graph.nodes[node]['community_weight'] = round(community_weight, 6)
            graph.nodes[node]['community_stage'] = summary['stage']
            graph.nodes[node]['community_stage_score'] = summary['stage_score']

    edges = nx.to_pandas_edgelist(graph)
    nodes = pd.DataFrame([{'account_id': node, **graph.nodes[node]} for node in graph.nodes()])
    communities = pd.DataFrame(community_rows)

    edges.to_parquet(os.path.join(out_dir, 'graph_edges.parquet'), index=False)
    nodes.to_parquet(os.path.join(out_dir, 'graph_nodes.parquet'), index=False)
    communities.to_parquet(os.path.join(out_dir, 'graph_communities.parquet'), index=False)

    return os.path.join(out_dir, 'graph_nodes.parquet'), os.path.join(out_dir, 'graph_edges.parquet')


def load_graph(out_dir: str):
    edges = pd.read_parquet(os.path.join(out_dir, 'graph_edges.parquet'))
    nodes = pd.read_parquet(os.path.join(out_dir, 'graph_nodes.parquet'))
    communities = pd.read_parquet(os.path.join(out_dir, 'graph_communities.parquet'))
    return nodes, edges, communities


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build ring graph from normalized parquet')
    parser.add_argument('--normalized', required=True, help='Path to normalized parquet')
    parser.add_argument('--out', default='models/graph', help='Output directory for graph artifacts')
    parser.add_argument('--n-clusters', type=int, default=8, help='(unused)')
    args = parser.parse_args()
    nodes_path, edges_path = build_graph(args.normalized, args.out)
    print('Nodes:', nodes_path)
    print('Edges:', edges_path)
