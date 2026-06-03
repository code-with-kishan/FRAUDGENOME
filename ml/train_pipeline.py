"""Improved training pipeline: SMOTE-Tomek, SHAP-driven feature selection,
LightGBM + XGBoost training, ensemble calibration, evaluation, and artifact saving.

Usage:
  python -m ml.train_pipeline --normalized data/processed/normalized.parquet --labels data/processed/labels.parquet --out models/
"""
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    lgb = None
    LGB_AVAILABLE = False
import xgboost as xgb

try:
    from imblearn.combine import SMOTETomek
except Exception:
    SMOTETomek = None

from ml.explain import save_shap_sample


def load_data(normalized_parquet: str, labels_parquet: str, feature_candidates: list = None):
    df = pd.read_parquet(normalized_parquet)
    labels = pd.read_parquet(labels_parquet)
    # aggregate per-account features (mean)
    if feature_candidates is None:
        feature_candidates = [c for c in df.columns if c.startswith('F')]
    agg = df.groupby('account_id')[feature_candidates].mean().reset_index()
    data = agg.merge(labels, on='account_id', how='inner')
    return data, feature_candidates


def resample_if_needed(X, y):
    if SMOTETomek is None:
        return X, y
    smt = SMOTETomek(random_state=42)
    try:
        Xr, yr = smt.fit_resample(X, y)
        return Xr, yr
    except Exception:
        return X, y


def select_topk_shap(model, X, k=50):
    import shap
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        arr = np.abs(shap_vals[1]).mean(axis=0)
    else:
        arr = np.abs(shap_vals).mean(axis=0)
    idxs = np.argsort(-arr)[:k]
    return [X.columns[i] for i in idxs]


def train(args):
    os.makedirs(args.out, exist_ok=True)
    data, feature_candidates = load_data(args.normalized, args.labels, feature_candidates=None)
    features = [c for c in feature_candidates if c in data.columns]

    X = data[features]
    y = data['label'].astype(int)
    splits = data['split']
    X_train = X[splits == 'train']
    y_train = y[splits == 'train']
    X_val = X[splits == 'val']
    y_val = y[splits == 'val']

    # Resample training set if required
    Xr, yr = resample_if_needed(X_train.values, y_train.values)

    # Train initial model for feature importance (LightGBM preferred)
    if LGB_AVAILABLE:
        init_clf = lgb.LGBMClassifier(n_estimators=200, random_state=42)
    else:
        init_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, random_state=42)
    init_clf.fit(Xr, yr)

    # SHAP-driven feature selection
    topk = args.topk if args.topk else min(50, X.shape[1])
    try:
        top_features = select_topk_shap(init_clf, X_train, k=topk)
    except Exception:
        # fallback to feature importances
        imp = lgb_clf.feature_importances_
        idxs = np.argsort(-imp)[:topk]
        top_features = [X_train.columns[i] for i in idxs]

    # Retrain final models on selected features
    Xr_sel = pd.DataFrame(Xr, columns=X_train.columns)[top_features].values
    X_val_sel = X_val[top_features].values

    lgb_final = None
    if LGB_AVAILABLE:
        lgb_final = lgb.LGBMClassifier(n_estimators=500, random_state=42)
        lgb_final.fit(Xr_sel, yr)
        joblib.dump(lgb_final, os.path.join(args.out, 'lgb_model.joblib'))

    # XGBoost final (sklearn API)
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=500, random_state=42)
    xgb_clf.fit(Xr_sel, yr)
    xgb_clf.save_model(os.path.join(args.out, 'xgb_model.json'))

    # Ensemble raw probabilities on validation
    p_xgb = xgb_clf.predict_proba(X_val_sel)[:, 1]
    if lgb_final is not None:
        p_lgb = lgb_final.predict_proba(X_val_sel)[:, 1]
        ensemble_raw = 0.5 * p_lgb + 0.5 * p_xgb
    else:
        ensemble_raw = p_xgb

    # Calibrate ensemble with simple logistic regression on validation
    calib = LogisticRegression(solver='lbfgs')
    calib.fit(ensemble_raw.reshape(-1, 1), y_val.values)
    joblib.dump(calib, os.path.join(args.out, 'calibrator.joblib'))

    ensemble_calibrated = calib.predict_proba(ensemble_raw.reshape(-1, 1))[:, 1]

    # Evaluate metrics
    precision, recall, _ = precision_recall_curve(y_val, ensemble_calibrated)
    pr_auc = auc(recall, precision)
    roc = roc_auc_score(y_val, ensemble_calibrated)

    # Save SHAP explainer sample (on a small X_sample)
    try:
        X_sample = X_train[top_features].sample(n=min(200, len(X_train)))
        # prefer saving explainer for the LightGBM final model if available
        model_for_shap = lgb_final if lgb_final is not None else xgb_clf
        save_shap_sample(model_for_shap, X_sample, os.path.join(args.out, 'shap_sample.joblib'))
    except Exception:
        pass

    metrics = {'pr_auc': float(pr_auc), 'roc_auc': float(roc), 'n_features': len(top_features)}
    with open(os.path.join(args.out, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f)

    # save selected feature list
    with open(os.path.join(args.out, 'selected_features.json'), 'w') as f:
        json.dump(top_features, f)

    print('PR-AUC:', pr_auc)
    print('ROC-AUC:', roc)
    return metrics


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalized', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--out', default='models')
    parser.add_argument('--topk', type=int, default=50)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    _cli()
