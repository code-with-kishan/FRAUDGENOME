import os
import tempfile
from ml.data_pipeline import ingest, normalize, extract_windows
from ml.ring_mapper import build_graph, classify_community_stage, load_graph


def _make_normalized_sample(path):
    import pandas as pd
    idx = pd.date_range('2024-01-01', periods=30, freq='D')
    rows = []
    for a in ['A','B','C','D']:
        for t in idx:
            rows.append({'account_id': a, 'timestamp': t, 'F3924': 0, 'F321': 0.0, 'F3836': float((ord(a)-64) + (t.day%5)), 'F2082': 0.0})
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def test_build_and_load_graph():
    tmp = tempfile.mkdtemp()
    norm = os.path.join(tmp, 'normalized.parquet')
    _make_normalized_sample(norm)
    out = os.path.join(tmp, 'graph')
    os.makedirs(out, exist_ok=True)
    nodes_path, edges_path = build_graph(norm, out)
    assert os.path.exists(nodes_path)
    assert os.path.exists(edges_path)
    nodes, edges, communities = load_graph(out)
    assert 'account_id' in nodes.columns
    assert not edges.empty
    assert 'stage' in communities.columns
    assert set(communities['stage']).issubset({'Recruiting', 'Active', 'Dispersing', 'Dormant'})


def test_classify_community_stage_heuristic():
    recruiting = classify_community_stage({
        'members': 6,
        'label_rate': 0.0,
        'recent_activity_rate': 0.75,
        'spike_rate': 0.5,
        'community_density': 0.4,
        'community_weight': 1.2,
        'recency_gap': 2.0,
    })
    assert recruiting['stage'] == 'Recruiting'

    active = classify_community_stage({
        'members': 8,
        'label_rate': 0.3,
        'recent_activity_rate': 0.5,
        'spike_rate': 0.25,
        'community_density': 0.35,
        'community_weight': 1.3,
        'recency_gap': 1.0,
    })
    assert active['stage'] == 'Active'

    dispersing = classify_community_stage({
        'members': 5,
        'label_rate': 0.4,
        'recent_activity_rate': 0.1,
        'spike_rate': 0.0,
        'community_density': 0.2,
        'community_weight': 1.05,
        'recency_gap': 20.0,
    })
    assert dispersing['stage'] == 'Dispersing'

    dormant = classify_community_stage({
        'members': 4,
        'label_rate': 0.0,
        'recent_activity_rate': 0.0,
        'spike_rate': 0.0,
        'community_density': 0.0,
        'community_weight': 1.0,
        'recency_gap': 999.0,
    })
    assert dormant['stage'] == 'Dormant'
