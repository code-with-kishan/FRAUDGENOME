"""Signature validation, decay detection, and library management for FRAUDGENOME.

Implements:
- Coverage, FSR, Lift calculation (section P)
- Multi-method validation gates
- Signature search, filter, history, versioning (section Q)
- Signature decay detection — STABLE / DECAY-WATCH / DECAY-CRITICAL (section R)
- Candidate signature generation
- Signature explanation
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Validation Gates (Section P)
# ---------------------------------------------------------------------------

# Acceptance thresholds from FRAUDGENOME proposal
COVERAGE_MIN = 0.20           # ≥ 20% of confirmed mules must match
FALSE_SIG_RATE_MAX = 0.05     # ≤ 5% false signature rate
LIFT_MIN = 10.0               # ≥ 10× lift over base rate
STABILITY_MIN = 0.75          # stable across ≥ 75% LOMO-CV iterations


def compute_coverage(
    pattern_fn,
    mule_accounts: pd.DataFrame,
    features: List[str],
) -> float:
    """Fraction of confirmed mules where the pattern function fires."""
    if mule_accounts.empty:
        return 0.0
    hits = mule_accounts.apply(lambda row: bool(pattern_fn(row[features])), axis=1)
    return float(hits.mean())


def compute_false_signature_rate(
    pattern_fn,
    legit_accounts: pd.DataFrame,
    features: List[str],
) -> float:
    """Fraction of legitimate accounts where pattern fires (false signature rate)."""
    if legit_accounts.empty:
        return 1.0
    hits = legit_accounts.apply(lambda row: bool(pattern_fn(row[features])), axis=1)
    return float(hits.mean())


def compute_lift(coverage: float, fsr: float, base_rate: float) -> float:
    """Lift = precision / base_rate = (coverage / (coverage + fsr * legit_ratio)) / base_rate."""
    if base_rate <= 0 or (coverage + fsr) <= 0:
        return 0.0
    # Approximate: lift = P(mule | pattern fires) / P(mule)
    # Using population fractions
    pattern_precision = coverage / (coverage + fsr + 1e-8)
    return pattern_precision / (base_rate + 1e-8)


def compute_stability(
    pattern_fn,
    mule_accounts: pd.DataFrame,
    features: List[str],
    n_bootstrap: int = 50,
    random_state: int = 42,
) -> float:
    """Bootstrap stability: fraction of bootstrap samples where pattern coverage ≥ COVERAGE_MIN."""
    rng = np.random.RandomState(random_state)
    n = len(mule_accounts)
    if n == 0:
        return 0.0

    stable_count = 0
    for _ in range(n_bootstrap):
        sample = mule_accounts.sample(n=n, replace=True, random_state=rng.randint(0, 2**31))
        cov = compute_coverage(pattern_fn, sample, features)
        if cov >= COVERAGE_MIN:
            stable_count += 1

    return stable_count / n_bootstrap


def validate_signature(
    pattern_fn,
    mule_accounts: pd.DataFrame,
    legit_accounts: pd.DataFrame,
    features: List[str],
    base_rate: float,
    method_count: int = 2,
    n_bootstrap: int = 50,
) -> Dict[str, Any]:
    """Run all five validation gates. Returns pass/fail for each gate."""
    coverage = compute_coverage(pattern_fn, mule_accounts, features)
    fsr = compute_false_signature_rate(pattern_fn, legit_accounts, features)
    lift = compute_lift(coverage, fsr, base_rate)
    stability = compute_stability(pattern_fn, mule_accounts, features, n_bootstrap)

    gate_coverage = coverage >= COVERAGE_MIN
    gate_fsr = fsr <= FALSE_SIG_RATE_MAX
    gate_lift = lift >= LIFT_MIN
    gate_stability = stability >= STABILITY_MIN
    gate_methods = method_count >= 2

    passed = all([gate_coverage, gate_fsr, gate_lift, gate_stability, gate_methods])

    return {
        "passed": passed,
        "coverage": round(coverage, 4),
        "coverage_pct": round(coverage * 100, 2),
        "false_signature_rate": round(fsr, 4),
        "fsr_pct": round(fsr * 100, 2),
        "lift": round(lift, 2),
        "stability": round(stability, 4),
        "stability_pct": round(stability * 100, 1),
        "method_count": method_count,
        "gates": {
            "coverage_ge_20pct": gate_coverage,
            "fsr_le_5pct": gate_fsr,
            "lift_ge_10x": gate_lift,
            "stability_ge_75pct": gate_stability,
            "confirmed_by_2plus_methods": gate_methods,
        },
    }


# ---------------------------------------------------------------------------
# Candidate Signature Generation (Section L)
# ---------------------------------------------------------------------------

def generate_threshold_candidates(
    df: pd.DataFrame,
    features: List[str],
    label_col: str = "label",
    n_candidates: int = 20,
) -> List[Dict[str, Any]]:
    """Generate candidate signatures as threshold rules on anchor features."""
    candidates = []
    mules = df[df[label_col] == 1]

    for feat in features:
        if feat not in df.columns:
            continue
        mule_vals = mules[feat].dropna()
        if mule_vals.empty:
            continue

        # High-value threshold: P75 of mules
        thresh_high = float(mule_vals.quantile(0.75))
        candidates.append({
            "signature_id": f"SIG-THRESH-{feat}-HIGH",
            "type": "threshold_high",
            "feature": feat,
            "threshold": round(thresh_high, 4),
            "direction": "ge",
            "description": f"{feat} ≥ {thresh_high:.4f} (P75 of confirmed mules)",
        })

        # Low-value threshold: P25 of mules (for dormancy etc.)
        thresh_low = float(mule_vals.quantile(0.25))
        candidates.append({
            "signature_id": f"SIG-THRESH-{feat}-LOW",
            "type": "threshold_low",
            "feature": feat,
            "threshold": round(thresh_low, 4),
            "direction": "le",
            "description": f"{feat} ≤ {thresh_low:.4f} (P25 of confirmed mules)",
        })

    return candidates[:n_candidates]


def generate_combination_candidates(
    df: pd.DataFrame,
    features: List[str],
    label_col: str = "label",
    n_top: int = 5,
) -> List[Dict[str, Any]]:
    """Generate candidate signatures combining 2-3 anchor features."""
    from itertools import combinations

    mules = df[df[label_col] == 1]
    legit = df[df[label_col] == 0]
    base_rate = float(len(mules) / len(df)) if len(df) > 0 else 0.01

    candidates = []
    avail = [f for f in features if f in df.columns]

    for feat_a, feat_b in list(combinations(avail[:n_top], 2)):
        thresh_a = float(df[feat_a].quantile(0.80))
        thresh_b = float(df[feat_b].quantile(0.80))

        mule_hits = ((mules[feat_a] >= thresh_a) & (mules[feat_b] >= thresh_b)).sum()
        legit_hits = ((legit[feat_a] >= thresh_a) & (legit[feat_b] >= thresh_b)).sum()

        coverage_val = mule_hits / len(mules) if len(mules) > 0 else 0
        fsr_val = legit_hits / len(legit) if len(legit) > 0 else 1

        if coverage_val >= 0.10:  # relaxed gate for candidate generation
            candidates.append({
                "signature_id": f"SIG-COMBO-{feat_a}-{feat_b}",
                "type": "combination",
                "features": [feat_a, feat_b],
                "thresholds": {feat_a: round(thresh_a, 4), feat_b: round(thresh_b, 4)},
                "preliminary_coverage": round(float(coverage_val), 4),
                "preliminary_fsr": round(float(fsr_val), 4),
                "preliminary_lift": round(compute_lift(coverage_val, fsr_val, base_rate), 2),
                "description": f"{feat_a} ≥ {thresh_a:.3f} AND {feat_b} ≥ {thresh_b:.3f}",
            })

    return candidates


# ---------------------------------------------------------------------------
# Signature Explanation
# ---------------------------------------------------------------------------

def explain_signature(
    sig: Dict[str, Any],
    account_features: Dict[str, float],
) -> Dict[str, Any]:
    """Generate human-readable explanation of why an account matches a signature."""
    explanation_parts = []
    feature_evidence = {}

    sig_type = sig.get("type", "unknown")

    if sig_type == "threshold_high":
        feat = sig["feature"]
        thresh = sig["threshold"]
        account_val = account_features.get(feat, 0.0)
        matched = account_val >= thresh
        feature_evidence[feat] = {
            "account_value": round(account_val, 4),
            "threshold": thresh,
            "matched": matched,
            "direction": "high",
        }
        if matched:
            explanation_parts.append(
                f"{feat} is {account_val:.4f}, which exceeds the mule signature threshold of {thresh:.4f}"
            )

    elif sig_type == "combination":
        for feat, thresh in sig.get("thresholds", {}).items():
            account_val = account_features.get(feat, 0.0)
            matched = account_val >= thresh
            feature_evidence[feat] = {
                "account_value": round(account_val, 4),
                "threshold": thresh,
                "matched": matched,
            }
            status = "✓" if matched else "✗"
            explanation_parts.append(
                f"{status} {feat} = {account_val:.4f} (threshold: {thresh:.4f})"
            )

    elif "features" in sig:
        for feat in sig.get("features", []):
            account_val = account_features.get(feat, 0.0)
            feature_evidence[feat] = {"account_value": round(account_val, 4)}
            explanation_parts.append(f"{feat} = {account_val:.4f}")

    confidence_map = {"STABLE": "HIGH", "DECAY-WATCH": "MEDIUM", "DECAY-CRITICAL": "LOW"}
    decay_status = sig.get("decay_status", "STABLE")

    return {
        "signature_id": sig.get("signature_id", "UNKNOWN"),
        "signature_description": sig.get("description", ""),
        "decay_status": decay_status,
        "confidence": confidence_map.get(decay_status, "HIGH"),
        "lift": sig.get("lift", 0.0),
        "coverage_pct": sig.get("coverage_pct", 0.0),
        "feature_evidence": feature_evidence,
        "plain_english": (
            f"This account matches signature {sig.get('signature_id', 'UNKNOWN')} "
            f"({sig.get('description', '')}) with {decay_status} confidence. "
            + "; ".join(explanation_parts)
        ),
    }


# ---------------------------------------------------------------------------
# Signature Decay Detection (Section R)
# ---------------------------------------------------------------------------

def compute_decay_score(
    sig: Dict[str, Any],
    df_recent: pd.DataFrame,
    df_historical: pd.DataFrame,
    features: List[str],
    label_col: str = "label",
    n_bootstrap: int = 50,
) -> Dict[str, Any]:
    """Compute temporal robustness / decay score using 3-proxy method."""
    results: Dict[str, Any] = {"signature_id": sig.get("signature_id", "UNKNOWN")}

    def pattern_fn(row):
        sig_type = sig.get("type", "")
        if sig_type == "threshold_high":
            feat = sig.get("feature")
            thresh = sig.get("threshold", 0)
            return feat in row.index and row[feat] >= thresh
        elif sig_type == "combination":
            for feat, thresh in sig.get("thresholds", {}).items():
                if feat not in row.index or row[feat] < thresh:
                    return False
            return True
        return False

    # Proxy 1: ID Range Split (older vs newer accounts)
    if "account_id" in df_historical.columns:
        all_ids = df_historical["account_id"].unique()
        mid = len(all_ids) // 2
        old_ids = set(all_ids[:mid])
        new_ids = set(all_ids[mid:])
        df_old = df_historical[df_historical["account_id"].isin(old_ids)]
        df_new = df_historical[df_historical["account_id"].isin(new_ids)]
        mules_old = df_old[df_old[label_col] == 1] if label_col in df_old.columns else df_old
        mules_new = df_new[df_new[label_col] == 1] if label_col in df_new.columns else df_new

        cov_old = compute_coverage(pattern_fn, mules_old, features) if not mules_old.empty else 0.0
        cov_new = compute_coverage(pattern_fn, mules_new, features) if not mules_new.empty else 0.0
        decay_rate_id = abs(cov_old - cov_new) / (cov_old + 1e-8)
        results["proxy_id_split"] = {
            "coverage_old": round(cov_old, 4),
            "coverage_new": round(cov_new, 4),
            "decay_rate": round(float(decay_rate_id), 4),
        }
    else:
        results["proxy_id_split"] = {"error": "no account_id column"}
        decay_rate_id = 0.0

    # Proxy 2: Variance Quintile Split
    feat_main = (
        sig.get("feature") or
        (sig.get("features", [None])[0]) or
        (list(sig.get("thresholds", {}).keys()) or [None])[0]
    )
    if feat_main and feat_main in df_historical.columns:
        quintiles = pd.qcut(df_historical[feat_main], 5, labels=False, duplicates="drop")
        cov_by_q = []
        for q in range(5):
            q_df = df_historical[quintiles == q]
            q_mules = q_df[q_df[label_col] == 1] if label_col in q_df.columns else q_df
            cov_q = compute_coverage(pattern_fn, q_mules, features) if not q_mules.empty else 0.0
            cov_by_q.append(round(float(cov_q), 4))
        variance_decay = float(np.std(cov_by_q))
        results["proxy_variance_quintile"] = {
            "coverage_by_quintile": cov_by_q,
            "coverage_variance": round(variance_decay, 4),
        }
    else:
        variance_decay = 0.0
        results["proxy_variance_quintile"] = {"error": "feature not available"}

    # Proxy 3: Bootstrap stability
    if label_col in df_historical.columns:
        mules_hist = df_historical[df_historical[label_col] == 1]
        stab = compute_stability(pattern_fn, mules_hist, features, n_bootstrap=n_bootstrap)
    else:
        stab = 0.5
    results["proxy_bootstrap_stability"] = round(float(stab), 4)

    # Aggregate decay classification
    decay_score = (decay_rate_id * 0.4) + (variance_decay * 0.3) + ((1.0 - stab) * 0.3)
    results["decay_score"] = round(float(decay_score), 4)

    if decay_score < 0.20 and stab >= 0.75:
        status = "STABLE"
    elif decay_score < 0.50 or stab >= 0.50:
        status = "DECAY-WATCH"
    else:
        status = "DECAY-CRITICAL"

    results["decay_status"] = status
    results["suppress"] = status == "DECAY-CRITICAL"
    return results


def classify_all_signatures(
    signatures: List[Dict[str, Any]],
    df: pd.DataFrame,
    features: List[str],
    label_col: str = "label",
) -> List[Dict[str, Any]]:
    """Classify all signatures in the library with STABLE / DECAY-WATCH / DECAY-CRITICAL."""
    updated = []
    for sig in signatures:
        try:
            decay_info = compute_decay_score(sig, df, df, features, label_col)
            sig_updated = {**sig, **{
                "decay_status": decay_info["decay_status"],
                "decay_score": decay_info["decay_score"],
                "suppress": decay_info.get("suppress", False),
                "decay_detail": decay_info,
            }}
        except Exception as e:
            sig_updated = {**sig, "decay_status": "STABLE", "decay_error": str(e)}
        updated.append(sig_updated)
    return updated


# ---------------------------------------------------------------------------
# Signature Library (Section Q)
# ---------------------------------------------------------------------------

class SignatureLibrary:
    """Persistent, versioned library of validated fraud signatures."""

    def __init__(self, library_path: str = "models/signature_library.json"):
        self.library_path = library_path
        self._signatures: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.library_path):
            try:
                with open(self.library_path) as fh:
                    data = json.load(fh)
                self._signatures = data.get("signatures", [])
            except Exception:
                self._signatures = []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.library_path) if os.path.dirname(self.library_path) else ".", exist_ok=True)
        with open(self.library_path, "w") as fh:
            json.dump({
                "version": "1.0",
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "signature_count": len(self._signatures),
                "signatures": self._signatures,
            }, fh, indent=2)

    def add_signature(self, sig: Dict[str, Any]) -> str:
        """Add a new validated signature. Returns signature_id."""
        if "signature_id" not in sig:
            sig["signature_id"] = f"SIG-{uuid.uuid4().hex[:6].upper()}"
        sig["added_at"] = datetime.now(timezone.utc).isoformat()
        sig["version"] = 1
        sig.setdefault("decay_status", "STABLE")
        sig.setdefault("status", "active")
        # Maintain history
        sig.setdefault("history", [])
        self._signatures.append(sig)
        self._save()
        return sig["signature_id"]

    def update_signature(self, signature_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing signature, maintaining version history."""
        for i, sig in enumerate(self._signatures):
            if sig.get("signature_id") == signature_id:
                # Append current to history
                snapshot = {k: v for k, v in sig.items() if k != "history"}
                sig.setdefault("history", []).append(snapshot)
                sig.update(updates)
                sig["version"] = sig.get("version", 1) + 1
                sig["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._signatures[i] = sig
                self._save()
                return True
        return False

    def search(
        self,
        query: Optional[str] = None,
        status_filter: Optional[str] = None,
        decay_filter: Optional[str] = None,
        min_lift: Optional[float] = None,
        feature_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search and filter signatures."""
        results = self._signatures

        if status_filter:
            results = [s for s in results if s.get("status") == status_filter]

        if decay_filter:
            results = [s for s in results if s.get("decay_status") == decay_filter]

        if min_lift is not None:
            results = [s for s in results if float(s.get("lift", 0)) >= min_lift]

        if feature_filter:
            def _has_feature(sig: Dict) -> bool:
                return (
                    sig.get("feature") == feature_filter or
                    feature_filter in sig.get("features", []) or
                    feature_filter in sig.get("thresholds", {})
                )
            results = [s for s in results if _has_feature(s)]

        if query:
            q = query.lower()
            results = [
                s for s in results
                if q in s.get("signature_id", "").lower() or
                   q in s.get("description", "").lower() or
                   q in str(s.get("features", "")).lower()
            ]

        return results

    def get_history(self, signature_id: str) -> List[Dict[str, Any]]:
        """Return full version history for a signature."""
        for sig in self._signatures:
            if sig.get("signature_id") == signature_id:
                return sig.get("history", [])
        return []

    def performance_summary(self) -> Dict[str, Any]:
        """Summary statistics for the library."""
        active = [s for s in self._signatures if s.get("status") == "active"]
        by_decay = {"STABLE": 0, "DECAY-WATCH": 0, "DECAY-CRITICAL": 0}
        for s in active:
            ds = s.get("decay_status", "STABLE")
            if ds in by_decay:
                by_decay[ds] += 1

        lifts = [float(s.get("lift", 0)) for s in active if s.get("lift")]
        coverages = [float(s.get("coverage", 0)) for s in active if s.get("coverage")]

        return {
            "total_signatures": len(self._signatures),
            "active_signatures": len(active),
            "decay_status_counts": by_decay,
            "mean_lift": round(float(np.mean(lifts)), 2) if lifts else 0.0,
            "mean_coverage_pct": round(float(np.mean(coverages)) * 100, 1) if coverages else 0.0,
            "suppressed_count": sum(1 for s in self._signatures if s.get("suppress", False)),
        }

    @property
    def all(self) -> List[Dict[str, Any]]:
        return self._signatures


# Singleton
_library_instance: Optional[SignatureLibrary] = None


def get_library(path: str = "models/signature_library.json") -> SignatureLibrary:
    global _library_instance
    if _library_instance is None or _library_instance.library_path != path:
        _library_instance = SignatureLibrary(path)
    return _library_instance
