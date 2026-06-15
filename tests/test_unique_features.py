import pytest
import numpy as np
import pandas as pd
from ml.unique_features import (
    compute_mule_dna_signature_score,
    compute_signature_confidence_index,
    compute_signature_decay_index,
    compute_signature_survival_rate,
    track_signature_evolution,
    generate_fraud_genome_timeline,
    query_fraud_genome,
    build_pattern_family_tree,
    compute_signature_similarity,
    detect_and_merge_signatures,
    detect_signature_splits,
    detect_fraud_mutation,
    detect_emerging_patterns,
    explore_high_risk_clusters,
    generate_behavioral_fingerprint,
    generate_mule_persona,
    discover_fraud_archetypes,
    compute_recruitment_risk_score,
    compute_first_time_mule_probability,
    compute_repeat_mule_probability
)

@pytest.fixture
def sample_features():
    return {"F321": 2.5, "F3836": 1.8, "F2082": 0.5}

@pytest.fixture
def sample_signatures():
    return [
        {
            "signature_id": "SIG-001",
            "description": "High F321",
            "type": "threshold_high",
            "feature": "F321",
            "threshold": 2.0,
            "status": "active",
            "lift": 12.5,
            "coverage": 0.35,
            "stability": 0.85,
            "decay_score": 0.15,
            "added_at": "2026-01-01T00:00:00Z"
        },
        {
            "signature_id": "SIG-002",
            "parent_signature_id": "SIG-001",
            "description": "High F321 & F3836",
            "type": "combination",
            "thresholds": {"F321": 2.0, "F3836": 1.5},
            "status": "active",
            "lift": 22.0,
            "coverage": 0.22,
            "stability": 0.78,
            "decay_score": 0.45,
            "added_at": "2026-02-01T00:00:00Z"
        }
    ]

def test_mule_dna_signature_score(sample_features, sample_signatures):
    score = compute_mule_dna_signature_score(sample_features, sample_signatures)
    assert score > 0.0
    assert score <= 1.0

def test_signature_confidence_index(sample_signatures):
    idx = compute_signature_confidence_index(sample_signatures[0])
    assert 0.0 <= idx <= 100.0

def test_signature_decay_index(sample_signatures):
    idx = compute_signature_decay_index(sample_signatures[0])
    assert idx == 15.0

def test_signature_survival_rate(sample_signatures):
    rate = compute_signature_survival_rate(sample_signatures[0], 30)
    assert 0.0 <= rate <= 1.0

def test_track_signature_evolution(sample_signatures):
    lineage = track_signature_evolution(sample_signatures)
    assert len(lineage) == 2
    assert lineage[1]["evolution_type"] == "modification"

def test_generate_fraud_genome_timeline():
    history = [
        {"timestamp": 1000, "event_type": "alert", "risk_score": 45.0, "description": "Triggered alert", "fingerprint": "xyz"},
        {"timestamp": 500, "event_type": "registration", "risk_score": 10.0, "description": "Account created", "fingerprint": "abc"}
    ]
    timeline = generate_fraud_genome_timeline(history)
    assert len(timeline) == 2
    assert timeline[0]["timestamp"] == 500

def test_query_fraud_genome(sample_signatures):
    results = query_fraud_genome(sample_signatures, {"status": "active", "type": "threshold_high"})
    assert len(results) == 1
    assert results[0]["signature_id"] == "SIG-001"

def test_build_pattern_family_tree(sample_signatures):
    tree = build_pattern_family_tree(sample_signatures)
    assert "roots" in tree
    assert len(tree["roots"]) == 1

def test_compute_signature_similarity(sample_signatures):
    sim = compute_signature_similarity(sample_signatures[0], sample_signatures[0])
    assert sim == 1.0

def test_detect_and_merge_signatures(sample_signatures):
    merged = detect_and_merge_signatures(sample_signatures, threshold=0.95)
    assert len(merged) == 2

def test_detect_signature_splits(sample_signatures):
    df = pd.DataFrame(np.random.randn(20, 3), columns=["F321", "F3836", "F2082"])
    splits = detect_signature_splits(sample_signatures[0], df, ["F321", "F3836", "F2082"])
    # May or may not split depending on random data, but should run without error
    assert isinstance(splits, list)

def test_detect_fraud_mutation(sample_signatures):
    hist = pd.DataFrame(np.random.randn(10, 3), columns=["F321", "F3836", "F2082"])
    recent = pd.DataFrame(np.random.randn(10, 3) + 2.0, columns=["F321", "F3836", "F2082"])
    mutation = detect_fraud_mutation(sample_signatures[0], hist, recent, ["F321", "F3836", "F2082"])
    assert "mutated" in mutation
    assert "mutation_score" in mutation

def test_detect_emerging_patterns():
    df = pd.DataFrame(np.random.randn(15, 3) + 3.0, columns=["F321", "F3836", "F2082"])
    emerging = detect_emerging_patterns(df, ["F321", "F3836", "F2082"], min_size=5)
    assert isinstance(emerging, list)

def test_explore_high_risk_clusters():
    emb = np.random.randn(10, 4)
    ids = [f"ACC-{i}" for i in range(10)]
    clusters = explore_high_risk_clusters(emb, ids, n_clusters=2)
    assert len(clusters) == 2

def test_generate_behavioral_fingerprint(sample_features):
    fp = generate_behavioral_fingerprint(sample_features, ["F321", "F3836", "F2082"])
    assert "F321" in fp
    assert "F3836" in fp

def test_generate_mule_persona(sample_features):
    persona = generate_mule_persona(sample_features)
    assert isinstance(persona, str)

def test_discover_fraud_archetypes(sample_features):
    arch = discover_fraud_archetypes(sample_features)
    assert "best_archetype" in arch

def test_compute_recruitment_risk_score():
    score = compute_recruitment_risk_score(45.0, "Recruiting")
    assert score == 60.0

def test_compute_first_time_mule_probability(sample_features):
    prob = compute_first_time_mule_probability(sample_features, 120)
    assert 0.0 <= prob <= 1.0

def test_compute_repeat_mule_probability(sample_features):
    prob = compute_repeat_mule_probability(sample_features, 3)
    assert 0.0 <= prob <= 1.0
