import numpy as np
import pandas as pd
import hashlib
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from sklearn.cluster import KMeans
from ml.signature_engine import get_library, validate_signature

# 1. Mule DNA Signature Score
def compute_mule_dna_signature_score(account_features: Dict[str, float], signatures: List[Dict[str, Any]]) -> float:
    """Compute how strongly the account matches any active signature in the library."""
    if not signatures:
        return 0.0
    scores = []
    for sig in signatures:
        if sig.get("status") != "active":
            continue
        sig_type = sig.get("type", "")
        if sig_type == "threshold_high":
            feat = sig.get("feature")
            thresh = sig.get("threshold", 0.0)
            val = account_features.get(feat, 0.0)
            if val >= thresh:
                scores.append(1.0)
            else:
                scores.append(val / max(thresh, 1e-5))
        elif sig_type == "combination":
            match_ratios = []
            for feat, thresh in sig.get("thresholds", {}).items():
                val = account_features.get(feat, 0.0)
                if val >= thresh:
                    match_ratios.append(1.0)
                else:
                    match_ratios.append(val / max(thresh, 1e-5))
            if match_ratios:
                scores.append(sum(match_ratios) / len(match_ratios))
    return float(np.max(scores)) if scores else 0.0

# 2. Signature Confidence Index
def compute_signature_confidence_index(sig: Dict[str, Any]) -> float:
    """Confidence index based on lift, coverage, and stability (0 to 100)."""
    lift = float(sig.get("lift", 1.0))
    coverage = float(sig.get("coverage", 0.1))
    stability = float(sig.get("stability", 0.5))
    # Normalize and scale
    norm_lift = min(1.0, lift / 50.0)
    index = (norm_lift * 0.4 + coverage * 0.3 + stability * 0.3) * 100.0
    return round(float(index), 2)

# 3. Signature Decay Index
def compute_signature_decay_index(sig: Dict[str, Any]) -> float:
    """Decay index representing how much the signature has degraded over time (0 to 100)."""
    decay_score = float(sig.get("decay_score", 0.0))
    return round(min(100.0, max(0.0, decay_score * 100.0)), 2)

# 4. Signature Survival Rate
def compute_signature_survival_rate(sig: Dict[str, Any], days_active: int) -> float:
    """Predict the probability that the signature remains valid after N days (0.0 to 1.0)."""
    decay_score = float(sig.get("decay_score", 0.1))
    # Exponential survival model: S(t) = exp(-lambda * t)
    lambda_decay = max(0.001, decay_score / 30.0)
    survival = np.exp(-lambda_decay * days_active)
    return float(round(max(0.0, min(1.0, survival)), 4))

# 5. Signature Evolution Tracker
def track_signature_evolution(signatures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a lineage tree representing the evolution of signatures (e.g. parent/child)."""
    lineage = []
    for sig in signatures:
        parent_id = sig.get("parent_signature_id", None)
        lineage.append({
            "signature_id": sig.get("signature_id"),
            "parent_signature_id": parent_id,
            "version": sig.get("version", 1),
            "evolution_type": "modification" if parent_id else "original",
            "timestamp": sig.get("updated_at") or sig.get("added_at")
        })
    return lineage

# 6. Fraud Genome Timeline
def generate_fraud_genome_timeline(account_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate chronological timeline of security/risk events for an account."""
    timeline = []
    # Sort history by timestamp
    sorted_history = sorted(account_history, key=lambda x: x.get("timestamp", 0))
    for i, event in enumerate(sorted_history):
        timeline.append({
            "step": i + 1,
            "timestamp": event.get("timestamp"),
            "event_type": event.get("event_type", "observation"),
            "risk_score": event.get("risk_score", 0.0),
            "description": event.get("description", ""),
            "fingerprint": event.get("fingerprint", "")
        })
    return timeline

# 7. Fraud Genome Explorer
def query_fraud_genome(signatures: List[Dict[str, Any]], query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Advanced search and metadata exploration of the Fraud Genome."""
    results = []
    for sig in signatures:
        match = True
        for k, v in query_params.items():
            if k in sig:
                if isinstance(sig[k], list):
                    if v not in sig[k]:
                        match = False
                elif str(sig[k]).lower() != str(v).lower():
                    match = False
            else:
                match = False
        if match:
            results.append(sig)
    return results

# 8. Fraud Pattern Family Tree
def build_pattern_family_tree(signatures: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a tree structure of how signatures relate to each other."""
    nodes = {}
    roots = []
    for sig in signatures:
        sid = sig.get("signature_id")
        nodes[sid] = {
            "id": sid,
            "description": sig.get("description"),
            "children": []
        }
    for sig in signatures:
        sid = sig.get("signature_id")
        pid = sig.get("parent_signature_id")
        if pid and pid in nodes:
            nodes[pid]["children"].append(nodes[sid])
        else:
            roots.append(nodes[sid])
    return {"roots": roots}

# 9. Signature Similarity Engine
def compute_signature_similarity(sig_a: Dict[str, Any], sig_b: Dict[str, Any]) -> float:
    """Calculate the similarity between two signature definitions."""
    type_a = sig_a.get("type")
    type_b = sig_b.get("type")
    if type_a != type_b:
        return 0.0
    
    if type_a == "threshold_high":
        feat_a = sig_a.get("feature")
        feat_b = sig_b.get("feature")
        if feat_a != feat_b:
            return 0.0
        val_a = float(sig_a.get("threshold", 0.0))
        val_b = float(sig_b.get("threshold", 0.0))
        denominator = max(val_a, val_b, 1e-5)
        return float(1.0 - abs(val_a - val_b) / denominator)
        
    elif type_a == "combination":
        keys_a = set(sig_a.get("thresholds", {}).keys())
        keys_b = set(sig_b.get("thresholds", {}).keys())
        intersection = keys_a.intersection(keys_b)
        union = keys_a.union(keys_b)
        if not union:
            return 0.0
        jaccard = len(intersection) / len(union)
        
        # Calculate value similarity for overlapping features
        val_sims = []
        for key in intersection:
            va = float(sig_a["thresholds"][key])
            vb = float(sig_b["thresholds"][key])
            val_sims.append(1.0 - abs(va - vb) / max(va, vb, 1e-5))
            
        val_sim = sum(val_sims) / len(val_sims) if val_sims else 0.0
        return float(jaccard * 0.5 + val_sim * 0.5)
        
    return 0.0

# 10. Signature Merge Engine
def detect_and_merge_signatures(signatures: List[Dict[str, Any]], threshold: float = 0.90) -> List[Dict[str, Any]]:
    """Identify highly similar signatures and return a merged set."""
    merged_sigs = []
    skip = set()
    for i in range(len(signatures)):
        if signatures[i].get("signature_id") in skip:
            continue
        sig_i = signatures[i]
        merged_with_any = False
        for j in range(i + 1, len(signatures)):
            sig_j = signatures[j]
            if sig_j.get("signature_id") in skip:
                continue
            sim = compute_signature_similarity(sig_i, sig_j)
            if sim >= threshold:
                # Merge sig_j into sig_i
                skip.add(sig_j.get("signature_id"))
                merged_with_any = True
                # Construct merged signature
                merged_sig = sig_i.copy()
                merged_sig["signature_id"] = f"MERGED-{sig_i.get('signature_id')}-{sig_j.get('signature_id')}"
                merged_sig["description"] = f"Merged: {sig_i.get('description')} / {sig_j.get('description')}"
                if "thresholds" in merged_sig and "thresholds" in sig_j:
                    for k in merged_sig["thresholds"]:
                        if k in sig_j["thresholds"]:
                            # Average thresholds
                            merged_sig["thresholds"][k] = float(np.mean([merged_sig["thresholds"][k], sig_j["thresholds"][k]]))
                merged_sigs.append(merged_sig)
                break
        if not merged_with_any:
            merged_sigs.append(sig_i)
    return merged_sigs

# 11. Signature Split Detection
def detect_signature_splits(sig: Dict[str, Any], matched_accounts_df: pd.DataFrame, features: List[str]) -> List[Dict[str, Any]]:
    """Detect if accounts matching a signature split into multiple distinct sub-behaviors."""
    if len(matched_accounts_df) < 10:
        return [] # not enough samples to detect split
    
    # Cluster the matched accounts into 2 groups
    X = matched_accounts_df[features].fillna(0.0).values
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # If clusters are well-separated, recommend a split
    centers = kmeans.cluster_centers_
    dist = np.linalg.norm(centers[0] - centers[1])
    if dist > 1.5:
        # Split detected! Generate two sub-signatures
        split_sigs = []
        for i, center in enumerate(centers):
            sub_sig = sig.copy()
            sub_sig["signature_id"] = f"{sig.get('signature_id')}-SPLIT-{i+1}"
            sub_sig["parent_signature_id"] = sig.get("signature_id")
            sub_sig["description"] = f"Split sub-behavior {i+1} of {sig.get('signature_id')}"
            split_sigs.append(sub_sig)
        return split_sigs
    return []

# 12. Fraud Mutation Detection
def detect_fraud_mutation(sig: Dict[str, Any], historical_matches: pd.DataFrame, recent_matches: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    """Detect if the fraud pattern has mutated (changed significantly) compared to history."""
    if historical_matches.empty or recent_matches.empty:
        return {"mutated": False, "mutation_score": 0.0}
    
    hist_mean = historical_matches[features].mean().values
    rec_mean = recent_matches[features].mean().values
    
    distance = float(np.linalg.norm(hist_mean - rec_mean))
    mutated = distance > 1.0
    return {
        "mutated": mutated,
        "mutation_score": round(distance, 4),
        "details": f"Feature drift distance between historical and recent matches is {distance:.4f}."
    }

# 13. Emerging Pattern Detection
def detect_emerging_patterns(unexplained_high_risk_df: pd.DataFrame, features: List[str], min_size: int = 5) -> List[Dict[str, Any]]:
    """Detect emerging, recurrent clusters of fraud among unexplained high-risk accounts."""
    if len(unexplained_high_risk_df) < min_size:
        return []
    
    X = unexplained_high_risk_df[features].fillna(0.0).values
    n_clusters = max(2, len(X) // 5)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    emerging = []
    for cluster_id in range(n_clusters):
        size = int((labels == cluster_id).sum())
        if size >= min_size:
            cluster_center = kmeans.cluster_centers_[cluster_id]
            # Find representative features
            rep_features = {}
            for i, feat in enumerate(features):
                if cluster_center[i] > 1.0: # threshold for significant activity
                    rep_features[feat] = float(round(cluster_center[i], 4))
            
            if rep_features:
                emerging.append({
                    "pattern_id": f"EMERGING-{uuid.uuid4().hex[:6].upper()}",
                    "cluster_size": size,
                    "representative_features": rep_features,
                    "description": f"Emerging cluster of {size} accounts characterized by high " + ", ".join(rep_features.keys())
                })
    return emerging

# 14. High-Risk Cluster Explorer
def explore_high_risk_clusters(embeddings: np.ndarray, account_ids: List[str], n_clusters: int = 5) -> Dict[str, List[str]]:
    """Categorize high-risk accounts into dense clusters based on their behavioral embeddings."""
    if len(embeddings) < n_clusters:
        return {"cluster_0": account_ids}
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    clusters = {}
    for i, label in enumerate(labels):
        cid = f"cluster_{label}"
        clusters.setdefault(cid, []).append(account_ids[i])
    return clusters

# 15. Behavioral Fingerprint Generator
def generate_behavioral_fingerprint(features: Dict[str, float], anchor_features: List[str]) -> str:
    """Generate a string/hash representation (fingerprint) of an account's signature behavior."""
    fingerprint_parts = []
    for feat in sorted(anchor_features):
        val = features.get(feat, 0.0)
        if val > 2.0:
            level = "H"
        elif val > 0.5:
            level = "M"
        else:
            level = "L"
        fingerprint_parts.append(f"{feat}:{level}")
    raw_str = "|".join(fingerprint_parts)
    # Generate short hash
    h = hashlib.sha1(raw_str.encode("utf-8")).hexdigest()[:8]
    return f"{raw_str} ({h})"

# 16. Mule Persona Generator
def generate_mule_persona(features: Dict[str, float]) -> str:
    """Map account features to a descriptive human-readable persona."""
    f321 = features.get("F321", 0.0) # Dormancy proxy
    f3836 = features.get("F3836", 0.0) # Transaction velocity
    f2082 = features.get("F2082", 0.0) # Amount / value proxy
    
    if f321 > 1.5 and f3836 > 1.5:
        return "Sleepy Mule (Sudden reactivation with high velocity)"
    if f3836 > 2.0 and f2082 > 2.0:
        return "Rapid Pipeline (Extremely fast high-value transit)"
    if f321 > 2.0:
        return "Zombie Conduit (Long-dormant account suddenly active)"
    if f2082 > 2.5:
        return "Heavy Whaler (Unusually large single transaction flows)"
    return "Standard Anomalous Profile"

# 17. Fraud Archetype Discovery
def discover_fraud_archetypes(features: Dict[str, float]) -> Dict[str, Any]:
    """Determine match confidence against canonical industry fraud archetypes."""
    f321 = features.get("F321", 0.0)
    f3836 = features.get("F3836", 0.0)
    
    archetypes = {
        "Money Mule (First Party)": min(100.0, (f321 * 0.3 + f3836 * 0.7) * 40.0),
        "Account Takeover (ATO)": min(100.0, (f321 * 0.8 + f3836 * 0.2) * 35.0),
        "Smurfing / Structured Deposits": min(100.0, f3836 * 45.0)
    }
    # Find best match
    best = max(archetypes, key=archetypes.get)
    return {
        "best_archetype": best,
        "confidence_scores": {k: round(v, 2) for k, v in archetypes.items()}
    }

# 18. Recruitment Risk Score
def compute_recruitment_risk_score(contagion_score: float, community_stage: Optional[str]) -> float:
    """Score the likelihood that the account is currently being recruited into a mule network."""
    base = contagion_score
    if community_stage == "Recruiting":
        base += 15.0
    elif community_stage == "Active":
        base += 5.0
    return float(round(min(100.0, max(0.0, base)), 2))

# 19. First-Time Mule Probability
def compute_first_time_mule_probability(features: Dict[str, float], account_age_days: int) -> float:
    """Probability that this is the first time the account is participating in fraud."""
    f321 = features.get("F321", 0.0)
    # Younger accounts + high dormancy reactivation suggest high first-time likelihood
    age_factor = max(0.1, 1.0 - (account_age_days / 365.0))
    prob = (f321 * 0.6 + age_factor * 0.4)
    return float(round(min(1.0, max(0.0, prob)), 4))

# 20. Repeat Mule Probability
def compute_repeat_mule_probability(features: Dict[str, float], historical_notes_count: int) -> float:
    """Probability that this is a repeat/serial offender."""
    f3836 = features.get("F3836", 0.0)
    note_factor = min(3, historical_notes_count) / 3.0
    prob = (f3836 * 0.4 + note_factor * 0.6)
    return float(round(min(1.0, max(0.0, prob)), 4))
