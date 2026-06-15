"""Feature engineering, selection, and analysis for FRAUDGENOME.

Implements:
- Feature selection and ranking (SHAP-based)
- Correlation and multicollinearity detection
- Outlier detection (IQR + Z-score)
- Dimensionality reduction (PCA)
- Feature metadata storage
- Cross-validation utilities
- Anchor feature detection
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ANCHOR_FEATURES = [
    "F321", "F527", "F1692", "F115", "F3894", "F531", "F670", "F2082",
    "F2122", "F2582", "F2678", "F2737", "F2956", "F3043", "F3836",
    "F3887", "F3889", "F3891",
]


# ---------------------------------------------------------------------------
# Outlier Detection
# ---------------------------------------------------------------------------

def detect_outliers_iqr(df: pd.DataFrame, features: List[str], threshold: float = 3.0) -> pd.DataFrame:
    """Detect outliers using IQR method. Returns boolean mask DataFrame."""
    outlier_mask = pd.DataFrame(False, index=df.index, columns=features)
    for feat in features:
        if feat not in df.columns:
            continue
        col = df[feat].dropna()
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outlier_mask[feat] = (df[feat] < lower) | (df[feat] > upper)
    return outlier_mask


def detect_outliers_zscore(df: pd.DataFrame, features: List[str], threshold: float = 3.5) -> pd.DataFrame:
    """Detect outliers using modified Z-score (median-based)."""
    outlier_mask = pd.DataFrame(False, index=df.index, columns=features)
    for feat in features:
        if feat not in df.columns:
            continue
        col = df[feat].fillna(df[feat].median())
        median = col.median()
        mad = np.median(np.abs(col - median))
        if mad == 0:
            continue
        modified_z = 0.6745 * (col - median) / mad
        outlier_mask[feat] = np.abs(modified_z) > threshold
    return outlier_mask


def outlier_detection_report(df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    """Generate comprehensive outlier detection report."""
    iqr_mask = detect_outliers_iqr(df, features)
    z_mask = detect_outliers_zscore(df, features)

    report: Dict[str, Any] = {
        "total_accounts": len(df),
        "features_checked": len(features),
        "method_iqr": {
            "accounts_with_any_outlier": int(iqr_mask.any(axis=1).sum()),
            "total_outlier_values": int(iqr_mask.sum().sum()),
            "top_outlier_features": iqr_mask.sum().nlargest(10).to_dict(),
        },
        "method_zscore": {
            "accounts_with_any_outlier": int(z_mask.any(axis=1).sum()),
            "total_outlier_values": int(z_mask.sum().sum()),
            "top_outlier_features": z_mask.sum().nlargest(10).to_dict(),
        },
        "consensus_outliers": int((iqr_mask & z_mask).any(axis=1).sum()),
    }
    return report


# ---------------------------------------------------------------------------
# Correlation & Multicollinearity
# ---------------------------------------------------------------------------

def correlation_analysis(
    df: pd.DataFrame,
    features: List[str],
    target_col: Optional[str] = None,
    threshold: float = 0.95,
) -> Dict[str, Any]:
    """Compute correlation matrix and detect highly correlated feature pairs."""
    feat_cols = [f for f in features if f in df.columns]
    corr_matrix = df[feat_cols].corr(method="spearman")

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(feat_cols)):
        for j in range(i + 1, len(feat_cols)):
            val = abs(corr_matrix.iloc[i, j])
            if val >= threshold:
                high_corr_pairs.append({
                    "feature_a": feat_cols[i],
                    "feature_b": feat_cols[j],
                    "correlation": round(float(val), 4),
                })

    result: Dict[str, Any] = {
        "n_features": len(feat_cols),
        "high_corr_pairs": sorted(high_corr_pairs, key=lambda x: -x["correlation"]),
        "redundant_features": list({p["feature_b"] for p in high_corr_pairs}),
        "threshold": threshold,
    }

    if target_col and target_col in df.columns:
        target_corr = df[feat_cols + [target_col]].corr(method="spearman")[target_col]
        result["target_correlation"] = (
            target_corr.drop(target_col)
            .abs()
            .nlargest(20)
            .to_dict()
        )

    return result


def detect_multicollinearity(df: pd.DataFrame, features: List[str], vif_threshold: float = 10.0) -> Dict[str, Any]:
    """Detect multicollinearity using VIF approximation."""
    feat_cols = [f for f in features if f in df.columns]
    X = df[feat_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    vif_scores = {}
    try:
        from numpy.linalg import lstsq
        for idx, feat in enumerate(feat_cols):
            y_col = X_scaled[:, idx]
            X_rest = np.delete(X_scaled, idx, axis=1)
            coef, _, _, _ = lstsq(X_rest, y_col, rcond=None)
            y_hat = X_rest @ coef
            ss_res = np.sum((y_col - y_hat) ** 2)
            ss_tot = np.sum((y_col - y_col.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
            vif_scores[feat] = round(float(vif), 2)
    except Exception:
        pass

    high_vif = {k: v for k, v in vif_scores.items() if v > vif_threshold}
    return {
        "vif_scores": vif_scores,
        "high_vif_features": high_vif,
        "n_multicollinear": len(high_vif),
        "threshold": vif_threshold,
    }


# ---------------------------------------------------------------------------
# SHAP Feature Importance & Selection
# ---------------------------------------------------------------------------

def compute_shap_importance(
    model: Any,
    X: pd.DataFrame,
    top_k: int = 80,
    sample_size: int = 300,
) -> Tuple[List[str], Dict[str, float]]:
    """Run SHAP TreeExplainer and return top-k features with mean |SHAP| importance."""
    import shap

    X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        # multi-class or binary with list output
        arr = np.abs(shap_values[-1]).mean(axis=0)
    else:
        arr = np.abs(shap_values).mean(axis=0)

    importance = {col: float(arr[i]) for i, col in enumerate(X.columns)}
    ranked = sorted(importance.items(), key=lambda x: -x[1])
    top_features = [feat for feat, _ in ranked[:top_k]]
    return top_features, importance


def feature_ranking_report(importance: Dict[str, float], anchor_features: List[str]) -> Dict[str, Any]:
    """Create a feature ranking report highlighting anchor features."""
    ranked = sorted(importance.items(), key=lambda x: -x[1])
    anchor_set = set(anchor_features)

    anchor_ranks = {}
    top_50 = [feat for feat, _ in ranked[:50]]
    anchor_in_top50 = [f for f in anchor_features if f in top_50]

    for rank, (feat, val) in enumerate(ranked, start=1):
        if feat in anchor_set:
            anchor_ranks[feat] = {"rank": rank, "importance": round(val, 6)}

    return {
        "total_features": len(ranked),
        "top_10": [{"feature": f, "importance": round(v, 6)} for f, v in ranked[:10]],
        "top_50": top_50,
        "anchor_feature_ranks": anchor_ranks,
        "anchor_features_in_top_50": anchor_in_top50,
        "anchor_coverage_pct": round(100.0 * len(anchor_in_top50) / len(anchor_features), 1) if anchor_features else 0,
    }


# ---------------------------------------------------------------------------
# Dimensionality Reduction
# ---------------------------------------------------------------------------

def apply_pca_compression(
    X: pd.DataFrame,
    n_components: int = 80,
    variance_threshold: float = 0.95,
    out_path: Optional[str] = None,
) -> Tuple[np.ndarray, PCA, int]:
    """Apply PCA compression. Chooses min components to explain variance_threshold."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))

    # Find n_components to explain threshold
    pca_probe = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1))
    pca_probe.fit(X_scaled)

    cumvar = np.cumsum(pca_probe.explained_variance_ratio_)
    n_needed = int(np.searchsorted(cumvar, variance_threshold)) + 1
    n_final = min(n_needed, n_components, pca_probe.n_components_)

    pca_final = PCA(n_components=n_final)
    X_compressed = pca_final.fit_transform(X_scaled)

    if out_path:
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        joblib.dump({"pca": pca_final, "scaler": scaler}, out_path)

    return X_compressed, pca_final, n_final


# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------

def stratified_kfold_cv(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run stratified k-fold cross-validation. model_fn() should return a fitted sklearn estimator."""
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_val)[:, 1]
        preds = (proba >= 0.5).astype(int)

        fold_results.append({
            "fold": fold + 1,
            "roc_auc": float(roc_auc_score(y_val, proba)) if y_val.sum() > 0 else 0.0,
            "pr_auc": float(average_precision_score(y_val, proba)) if y_val.sum() > 0 else 0.0,
            "f1": float(f1_score(y_val, preds, zero_division=0)),
        })

    roc_vals = [r["roc_auc"] for r in fold_results]
    pr_vals = [r["pr_auc"] for r in fold_results]
    f1_vals = [r["f1"] for r in fold_results]

    return {
        "n_splits": n_splits,
        "folds": fold_results,
        "mean_roc_auc": float(np.mean(roc_vals)),
        "std_roc_auc": float(np.std(roc_vals)),
        "mean_pr_auc": float(np.mean(pr_vals)),
        "std_pr_auc": float(np.std(pr_vals)),
        "mean_f1": float(np.mean(f1_vals)),
        "std_f1": float(np.std(f1_vals)),
    }


def leave_one_mule_out_cv(
    model_fn,
    X: pd.DataFrame,
    y: pd.Series,
    max_iters: int = 81,
) -> Dict[str, Any]:
    """Leave-One-Mule-Out cross validation — runs up to max_iters retraining iterations."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    mule_indices = y[y == 1].index.tolist()
    n_iters = min(len(mule_indices), max_iters)
    results = []

    for i, leave_out_idx in enumerate(mule_indices[:n_iters]):
        train_mask = X.index != leave_out_idx
        X_tr = X[train_mask].values
        y_tr = y[train_mask].values
        X_held = X.loc[[leave_out_idx]].values
        y_held = y.loc[[leave_out_idx]].values

        try:
            model = model_fn()
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_held)[:, 1][0]
            results.append({
                "iter": i + 1,
                "left_out_idx": str(leave_out_idx),
                "mule_score": float(proba),
                "detected": bool(proba >= 0.5),
            })
        except Exception as e:
            results.append({"iter": i + 1, "left_out_idx": str(leave_out_idx), "error": str(e)})

    detected = [r for r in results if r.get("detected")]
    return {
        "n_mules": len(mule_indices),
        "n_iterations": n_iters,
        "detection_rate": round(len(detected) / n_iters, 4) if n_iters > 0 else 0.0,
        "mean_mule_score": round(float(np.mean([r["mule_score"] for r in results if "mule_score" in r])), 4),
        "iterations": results,
    }


# ---------------------------------------------------------------------------
# Feature Metadata Storage
# ---------------------------------------------------------------------------

def save_feature_metadata(
    features: List[str],
    importance: Dict[str, float],
    anchor_features: List[str],
    out_path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Save full feature metadata JSON for downstream use."""
    anchor_set = set(anchor_features)
    metadata = {
        "total_features": len(features),
        "selected_features": features,
        "anchor_features": anchor_features,
        "importance_scores": {f: round(importance.get(f, 0.0), 8) for f in features},
        "anchor_feature_detail": {
            f: {
                "rank": next((i + 1 for i, feat in enumerate(
                    sorted(importance, key=lambda x: -importance[x])
                ) if feat == f), None),
                "importance": round(importance.get(f, 0.0), 8),
                "is_anchor": True,
            }
            for f in anchor_features if f in set(features)
        },
        "non_anchor_selected": [f for f in features if f not in anchor_set],
    }
    if extra:
        metadata.update(extra)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    return out_path


# ---------------------------------------------------------------------------
# Duplicate Detection
# ---------------------------------------------------------------------------

def detect_duplicates(df: pd.DataFrame, subset_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Detect duplicate rows (exact or near-duplicate on account_id)."""
    result: Dict[str, Any] = {}

    if "account_id" in df.columns:
        dup_accounts = df[df.duplicated(subset=["account_id"], keep=False)]
        result["duplicate_account_ids"] = int(dup_accounts["account_id"].nunique())
        result["total_duplicate_rows"] = int(len(dup_accounts))
        result["duplicate_account_id_list"] = dup_accounts["account_id"].unique()[:20].tolist()

    if subset_cols:
        cols = [c for c in subset_cols if c in df.columns]
        if cols:
            exact_dups = df[df.duplicated(subset=cols, keep=False)]
            result["exact_duplicate_rows_on_subset"] = int(len(exact_dups))

    feature_cols = [c for c in df.columns if c.startswith("F")]
    if feature_cols:
        sample_feats = feature_cols[:50]
        near_dups = df[df.duplicated(subset=sample_feats, keep=False)]
        result["near_duplicate_rows"] = int(len(near_dups))

    return result


# ---------------------------------------------------------------------------
# Anchor Feature Detection
# ---------------------------------------------------------------------------

def detect_anchor_features(
    df: pd.DataFrame,
    label_col: str = "label",
    candidate_anchors: Optional[List[str]] = None,
    top_n: int = 18,
) -> Dict[str, Any]:
    """Detect which features behave as anchor (high discriminative power for mules)."""
    if candidate_anchors is None:
        candidate_anchors = ANCHOR_FEATURES

    feat_cols = [f for f in candidate_anchors if f in df.columns]
    if label_col not in df.columns or not feat_cols:
        return {"error": "label column or anchor features missing"}

    mules = df[df[label_col] == 1][feat_cols]
    legit = df[df[label_col] == 0][feat_cols]

    discriminative = []
    for feat in feat_cols:
        m_mean = float(mules[feat].mean()) if not mules[feat].empty else 0.0
        l_mean = float(legit[feat].mean()) if not legit[feat].empty else 0.0
        m_std = float(mules[feat].std()) if not mules[feat].empty else 1.0
        separation = abs(m_mean - l_mean) / (m_std + 1e-8)
        discriminative.append({
            "feature": feat,
            "mule_mean": round(m_mean, 4),
            "legit_mean": round(l_mean, 4),
            "separation_score": round(float(separation), 4),
        })

    discriminative.sort(key=lambda x: -x["separation_score"])
    confirmed_anchors = [d["feature"] for d in discriminative[:top_n]]

    return {
        "confirmed_anchor_features": confirmed_anchors,
        "anchor_detail": discriminative,
        "n_anchors": len(confirmed_anchors),
    }
