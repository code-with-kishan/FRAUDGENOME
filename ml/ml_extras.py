"""
FRAUDGENOME ML Extras
Implements missing ML features:
- F: Logistic Regression baseline, Hard voting ensemble, Isotonic calibration
- G: Hyperparameter tuning
- I: Global SHAP analysis, SHAP summary/dependence plots
- K: Embedding visualization
"""

from __future__ import annotations

import json
import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger("fraudgenome.ml_extras")


# ===========================================================================
# F. Logistic Regression Baseline
# ===========================================================================

def train_logistic_regression_baseline(X_train, y_train, X_val, y_val, scale_pos_weight: float = 111.0):
    """Train a Logistic Regression baseline model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    class_weight = {0: 1.0, 1: float(scale_pos_weight)}
    lr = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        solver="lbfgs",
        C=0.1,
        random_state=42,
    )
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.0
    pr_auc = average_precision_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.0
    logger.info(f"LR Baseline — ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    return lr, {"roc_auc": roc_auc, "pr_auc": pr_auc}


# ===========================================================================
# F. Hard Voting Ensemble
# ===========================================================================

def hard_voting_ensemble(models: List, X, threshold: float = 0.5) -> np.ndarray:
    """Hard vote: majority of models must predict positive."""
    votes = []
    for m in models:
        if hasattr(m, "predict_proba"):
            prob = m.predict_proba(X)[:, 1]
        else:
            prob = m.predict(X)
        votes.append((prob >= threshold).astype(int))
    votes_arr = np.stack(votes, axis=1)  # shape (n, num_models)
    majority = (votes_arr.sum(axis=1) > len(models) / 2).astype(int)
    return majority


# ===========================================================================
# F. Isotonic Calibration
# ===========================================================================

def apply_isotonic_calibration(base_model, X_cal, y_cal):
    """Wrap a model with isotonic regression calibration."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.base import BaseEstimator, ClassifierMixin

    class SklearnWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, model):
            self.model = model
            self.classes_ = np.array([0, 1])

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            if hasattr(self.model, "predict_proba"):
                p = self.model.predict_proba(X)[:, 1]
            else:
                p = self.model.predict(X)
            return np.column_stack([1 - p, p])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    wrapper = SklearnWrapper(base_model)
    calibrated = CalibratedClassifierCV(wrapper, method="isotonic", cv="prefit")
    calibrated.fit(X_cal, y_cal)
    logger.info("Isotonic calibration applied.")
    return calibrated


# ===========================================================================
# G. Hyperparameter Tuning
# ===========================================================================

def tune_xgboost(X_train, y_train, scale_pos_weight: float = 111.0, n_trials: int = 20) -> Dict:
    """Hyperparameter tuning for XGBoost using random search."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import average_precision_score
    import xgboost as xgb

    rng = np.random.RandomState(42)
    param_space = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
    }

    best_score = -1
    best_params = {}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for trial in range(n_trials):
        params = {k: rng.choice(v) for k, v in param_space.items()}
        params["scale_pos_weight"] = scale_pos_weight
        params["use_label_encoder"] = False
        params["eval_metric"] = "aucpr"
        params["random_state"] = 42

        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            Xt, Xv = X_train[train_idx], X_train[val_idx]
            yt, yv = y_train[train_idx], y_train[val_idx]
            m = xgb.XGBClassifier(**params, n_jobs=-1, verbosity=0)
            m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            prob = m.predict_proba(Xv)[:, 1]
            score = average_precision_score(yv, prob) if len(np.unique(yv)) > 1 else 0.0
            scores.append(score)

        mean_score = float(np.mean(scores))
        logger.info(f"Trial {trial+1}/{n_trials}: PR-AUC={mean_score:.4f}, params={params}")
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    logger.info(f"Best params: PR-AUC={best_score:.4f}, {best_params}")
    return {"best_params": best_params, "best_pr_auc": best_score}


def tune_lightgbm(X_train, y_train, scale_pos_weight: float = 111.0, n_trials: int = 20) -> Dict:
    """Hyperparameter tuning for LightGBM using random search."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import average_precision_score
    import lightgbm as lgb

    rng = np.random.RandomState(42)
    param_space = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, -1],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_samples": [5, 10, 20, 50],
    }

    best_score = -1
    best_params = {}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for trial in range(n_trials):
        params = {k: rng.choice(v) for k, v in param_space.items()}
        params["scale_pos_weight"] = scale_pos_weight
        params["objective"] = "binary"
        params["metric"] = "average_precision"
        params["verbosity"] = -1
        params["random_state"] = 42

        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            Xt, Xv = X_train[train_idx], X_train[val_idx]
            yt, yv = y_train[train_idx], y_train[val_idx]
            m = lgb.LGBMClassifier(**params, n_jobs=-1)
            m.fit(Xt, yt, eval_set=[(Xv, yv)])
            prob = m.predict_proba(Xv)[:, 1]
            score = average_precision_score(yv, prob) if len(np.unique(yv)) > 1 else 0.0
            scores.append(score)

        mean_score = float(np.mean(scores))
        logger.info(f"LGB Trial {trial+1}/{n_trials}: PR-AUC={mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    logger.info(f"LGB Best params: PR-AUC={best_score:.4f}")
    return {"best_params": best_params, "best_pr_auc": best_score}


# ===========================================================================
# I. Global SHAP Analysis, SHAP Summary/Dependence Plot Data
# ===========================================================================

def compute_global_shap(model, X: np.ndarray, feature_names: List[str], max_samples: int = 500) -> Dict:
    """Compute global SHAP values and return summary statistics."""
    try:
        import shap
        if X.shape[0] > max_samples:
            idx = np.random.RandomState(42).choice(X.shape[0], max_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs = np.abs(shap_values).mean(axis=0)
        global_importance = [
            {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
            for i in np.argsort(mean_abs)[::-1]
        ]
        return {
            "global_importance": global_importance,
            "n_samples": int(X_sample.shape[0]),
            "shap_matrix": shap_values.tolist() if X_sample.shape[0] <= 200 else None,
        }
    except Exception as e:
        logger.warning(f"Global SHAP failed: {e}")
        return {"global_importance": [], "error": str(e)}


def compute_shap_summary_plot_data(model, X: np.ndarray, feature_names: List[str], top_n: int = 20) -> Dict:
    """Produce data for a SHAP beeswarm/summary plot."""
    try:
        import shap
        X_sample = X[:300] if X.shape[0] > 300 else X
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list):
            sv = sv[1]

        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_n]
        result = []
        for i in top_idx:
            result.append({
                "feature": feature_names[i],
                "shap_values": sv[:, i].tolist(),
                "feature_values": X_sample[:, i].tolist(),
                "mean_abs_shap": float(mean_abs[i]),
            })
        return {"summary_data": result, "n_samples": int(X_sample.shape[0])}
    except Exception as e:
        logger.warning(f"SHAP summary plot failed: {e}")
        return {"summary_data": [], "error": str(e)}


def compute_shap_dependence_plot_data(
    model, X: np.ndarray, feature_names: List[str], feature: str, interaction_feature: Optional[str] = None
) -> Dict:
    """Produce data for a SHAP dependence plot."""
    try:
        import shap
        X_sample = X[:500] if X.shape[0] > 500 else X
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list):
            sv = sv[1]

        feat_idx = feature_names.index(feature) if feature in feature_names else 0
        color_idx = (
            feature_names.index(interaction_feature)
            if interaction_feature and interaction_feature in feature_names
            else None
        )
        result = {
            "feature": feature,
            "x_values": X_sample[:, feat_idx].tolist(),
            "shap_values": sv[:, feat_idx].tolist(),
        }
        if color_idx is not None:
            result["color_feature"] = interaction_feature
            result["color_values"] = X_sample[:, color_idx].tolist()
        return result
    except Exception as e:
        logger.warning(f"SHAP dependence plot failed: {e}")
        return {"error": str(e)}


# ===========================================================================
# K. Embedding Visualization (UMAP / t-SNE data)
# ===========================================================================

def compute_embedding_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    account_ids: Optional[List[str]] = None,
    method: str = "umap",
    n_components: int = 2,
) -> Dict:
    """Reduce embeddings to 2D for visualization. Returns plot-ready data."""
    try:
        if method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=15)
                coords = reducer.fit_transform(embeddings)
            except ImportError:
                method = "tsne"

        if method == "tsne":
            from sklearn.manifold import TSNE
            perp = min(30, max(5, embeddings.shape[0] // 5))
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=perp, max_iter=500)
            coords = reducer.fit_transform(embeddings)

        points = []
        for i in range(len(coords)):
            p = {
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "label": int(labels[i]) if labels is not None else 0,
                "is_mule": int(labels[i]) == 1 if labels is not None else False,
            }
            if account_ids:
                p["account_id"] = account_ids[i]
            points.append(p)

        return {"method": method, "points": points, "n_points": len(points)}
    except Exception as e:
        logger.warning(f"Embedding visualization failed: {e}")
        return {"error": str(e), "points": []}


# ===========================================================================
# M. SHAP Interaction Mining
# ===========================================================================

def compute_shap_interaction_values(model, X: np.ndarray, feature_names: List[str], max_samples: int = 200) -> Dict:
    """Compute SHAP interaction values for feature pair discovery."""
    try:
        import shap
        X_s = X[:max_samples] if X.shape[0] > max_samples else X
        explainer = shap.TreeExplainer(model)
        interaction_vals = explainer.shap_interaction_values(X_s)
        if isinstance(interaction_vals, list):
            interaction_vals = interaction_vals[1]

        # Mean absolute interaction for each pair
        mean_interaction = np.abs(interaction_vals).mean(axis=0)
        n = len(feature_names)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append({
                    "feature_a": feature_names[i],
                    "feature_b": feature_names[j],
                    "mean_abs_interaction": float(mean_interaction[i, j]),
                })
        pairs.sort(key=lambda x: x["mean_abs_interaction"], reverse=True)
        return {"feature_pairs": pairs[:50], "n_samples": int(X_s.shape[0])}
    except Exception as e:
        logger.warning(f"SHAP interaction values failed: {e}")
        return {"feature_pairs": [], "error": str(e)}


def rank_interactions(interaction_data: Dict) -> List[Dict]:
    """Rank feature pairs by interaction strength."""
    pairs = interaction_data.get("feature_pairs", [])
    return sorted(pairs, key=lambda x: x["mean_abs_interaction"], reverse=True)


def contrastive_interaction_analysis(
    model, X_mule: np.ndarray, X_legit: np.ndarray, feature_names: List[str]
) -> Dict:
    """Find interactions that differ most between mules and legitimate accounts."""
    try:
        import shap
        max_s = 100
        Xm = X_mule[:max_s]
        Xl = X_legit[:max_s]
        explainer = shap.TreeExplainer(model)

        iv_mule = explainer.shap_interaction_values(Xm)
        iv_legit = explainer.shap_interaction_values(Xl)
        if isinstance(iv_mule, list):
            iv_mule = iv_mule[1]
        if isinstance(iv_legit, list):
            iv_legit = iv_legit[1]

        diff = np.abs(iv_mule).mean(axis=0) - np.abs(iv_legit).mean(axis=0)
        n = len(feature_names)
        contrastive = []
        for i in range(n):
            for j in range(i + 1, n):
                contrastive.append({
                    "feature_a": feature_names[i],
                    "feature_b": feature_names[j],
                    "mule_interaction": float(np.abs(iv_mule[:, i, j]).mean()),
                    "legit_interaction": float(np.abs(iv_legit[:, i, j]).mean()),
                    "contrast_score": float(diff[i, j]),
                })
        contrastive.sort(key=lambda x: abs(x["contrast_score"]), reverse=True)
        return {"contrastive_pairs": contrastive[:30]}
    except Exception as e:
        logger.warning(f"Contrastive interaction analysis failed: {e}")
        return {"contrastive_pairs": [], "error": str(e)}


def extract_candidate_patterns_from_interactions(interaction_data: Dict, top_n: int = 10) -> List[Dict]:
    """Extract candidate fraud patterns from top interaction pairs."""
    pairs = rank_interactions(interaction_data)[:top_n]
    patterns = []
    for p in pairs:
        patterns.append({
            "type": "interaction_pair",
            "features": [p["feature_a"], p["feature_b"]],
            "strength": p["mean_abs_interaction"],
            "candidate_id": f"CAND-INT-{p['feature_a']}-{p['feature_b']}",
        })
    return patterns


# ===========================================================================
# N. XGBoost Path Mining
# ===========================================================================

def extract_tree_paths(model, feature_names: List[str], max_trees: int = 50) -> List[Dict]:
    """Extract decision paths from XGBoost trees."""
    try:
        import xgboost as xgb
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        df_trees = booster.trees_to_dataframe()
        paths = []
        for tree_id in df_trees["Tree"].unique()[:max_trees]:
            tree_df = df_trees[df_trees["Tree"] == tree_id]
            # Find leaf nodes
            leaves = tree_df[tree_df["Feature"] == "Leaf"]
            for _, leaf in leaves.iterrows():
                path = _trace_path(tree_df, leaf["Node"])
                if path:
                    paths.append({
                        "tree_id": int(tree_id),
                        "leaf_id": int(leaf["Node"]),
                        "leaf_value": float(leaf["Gain"]),
                        "path": path,
                        "depth": len(path),
                    })
        return paths
    except Exception as e:
        logger.warning(f"Tree path extraction failed: {e}")
        return []


def _trace_path(tree_df: pd.DataFrame, leaf_node: int) -> List[Dict]:
    """Trace path from root to a leaf node."""
    path = []
    # Build parent map
    parent_map = {}
    for _, row in tree_df.iterrows():
        if row["Yes"] is not None and not pd.isna(row["Yes"]):
            try:
                parent_map[int(row["Yes"])] = (int(row["Node"]), "yes")
            except Exception:
                pass
        if row["No"] is not None and not pd.isna(row["No"]):
            try:
                parent_map[int(row["No"])] = (int(row["Node"]), "no")
            except Exception:
                pass

    node = leaf_node
    while node in parent_map:
        parent_id, direction = parent_map[node]
        parent_row = tree_df[tree_df["Node"] == parent_id].iloc[0]
        path.insert(0, {
            "node": parent_id,
            "feature": parent_row["Feature"],
            "split": float(parent_row["Split"]) if not pd.isna(parent_row["Split"]) else None,
            "direction": direction,
        })
        node = parent_id
    return path


def analyze_leaf_paths(paths: List[Dict]) -> Dict:
    """Analyze leaf paths for common split patterns."""
    from collections import Counter
    feature_freq = Counter()
    for p in paths:
        for step in p.get("path", []):
            if step["feature"] != "Leaf":
                feature_freq[step["feature"]] += 1

    split_pairs = Counter()
    for p in paths:
        path_feats = [s["feature"] for s in p.get("path", []) if s["feature"] != "Leaf"]
        for i in range(len(path_feats) - 1):
            split_pairs[(path_feats[i], path_feats[i + 1])] += 1

    return {
        "feature_frequency": dict(feature_freq.most_common(20)),
        "split_pairs": {f"{a}→{b}": c for (a, b), c in split_pairs.most_common(20)},
        "total_paths": len(paths),
    }


def mine_feature_cooccurrences(paths: List[Dict], min_support: int = 5) -> List[Dict]:
    """Find features that co-occur frequently in the same path."""
    from collections import Counter
    from itertools import combinations
    cooccur = Counter()
    for p in paths:
        feats = list({s["feature"] for s in p.get("path", []) if s["feature"] != "Leaf"})
        for combo in combinations(sorted(feats), 2):
            cooccur[combo] += 1

    result = [
        {"feature_a": a, "feature_b": b, "cooccurrence": c}
        for (a, b), c in cooccur.most_common(30)
        if c >= min_support
    ]
    return result


def discover_triple_patterns(paths: List[Dict], min_support: int = 3) -> List[Dict]:
    """Find triplets of features that frequently appear together in paths."""
    from collections import Counter
    from itertools import combinations
    triple_count = Counter()
    for p in paths:
        feats = list({s["feature"] for s in p.get("path", []) if s["feature"] != "Leaf"})
        for combo in combinations(sorted(feats), 3):
            triple_count[combo] += 1

    result = [
        {"features": list(t), "support": c}
        for t, c in triple_count.most_common(20)
        if c >= min_support
    ]
    return result


# ===========================================================================
# O. Lasso Interaction Engine
# ===========================================================================

def generate_interaction_features(X: np.ndarray, feature_names: List[str], top_n: int = 20) -> Tuple[np.ndarray, List[str]]:
    """Generate pairwise interaction features for top-N features."""
    from itertools import combinations
    n_feat = min(top_n, X.shape[1])
    X_sub = X[:, :n_feat]
    names_sub = feature_names[:n_feat]

    interaction_cols = []
    interaction_names = []
    for i, j in combinations(range(n_feat), 2):
        col = X_sub[:, i] * X_sub[:, j]
        interaction_cols.append(col)
        interaction_names.append(f"{names_sub[i]}*{names_sub[j]}")

    if not interaction_cols:
        return X, feature_names

    X_interactions = np.column_stack(interaction_cols)
    X_combined = np.hstack([X, X_interactions])
    all_names = feature_names + interaction_names
    return X_combined, all_names


def train_lasso_on_interactions(X_int: np.ndarray, y: np.ndarray, feature_names: List[str], alphas=None) -> Dict:
    """Train Lasso on interaction features to find significant patterns."""
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_int)

    lasso = LassoCV(alphas=alphas, cv=5, max_iter=5000, random_state=42)
    lasso.fit(X_scaled, y)

    coefs = lasso.coef_
    nonzero = [(feature_names[i], float(coefs[i])) for i in range(len(coefs)) if abs(coefs[i]) > 1e-6]
    nonzero.sort(key=lambda x: abs(x[1]), reverse=True)

    logger.info(f"Lasso selected {len(nonzero)} features/interactions, alpha={lasso.alpha_:.4f}")
    return {
        "best_alpha": float(lasso.alpha_),
        "selected_features": nonzero[:30],
        "n_nonzero": len(nonzero),
    }


def extract_patterns_from_lasso(lasso_result: Dict) -> List[Dict]:
    """Extract interaction patterns from Lasso coefficients."""
    patterns = []
    for name, coef in lasso_result.get("selected_features", []):
        if "*" in name:
            feats = name.split("*")
            patterns.append({
                "type": "lasso_interaction",
                "features": feats,
                "coefficient": coef,
                "candidate_id": f"CAND-LASSO-{name}",
            })
    return patterns
