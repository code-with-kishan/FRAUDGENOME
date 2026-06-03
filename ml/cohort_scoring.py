"""Cohort-relative anomaly scoring.

Computes cohort statistics (mean/std) per feature, produces per-account cohort-relative
z-scores, computes a cohort anomaly score (mean absolute z across selected features),
and evaluates false-positive reduction when applying a cohort-based filter to model flags.

Usage:
  python -m ml.cohort_scoring --normalized data/processed/normalized.parquet --labels data/processed/labels.parquet --cohorts data/processed/cohorts.parquet --models models/ --selected models/selected_features.json

Outputs a short report with baseline FP and adjusted FP and percentage reduction.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb


def compute_cohort_stats(normalized_parquet: str, cohorts_parquet: str, features: list):
    df = pd.read_parquet(normalized_parquet)
    cohorts = pd.read_parquet(cohorts_parquet)
    merged = df.groupby('account_id')[features].mean().reset_index().merge(cohorts, on='account_id', how='left')
    stats = merged.groupby('cohort_id')[features].agg(['mean','std'])
    # flatten
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    return merged, stats


def compute_cohort_zscores(merged_df: pd.DataFrame, stats_df: pd.DataFrame, features: list):
    # for each account, compute z = (value - cohort_mean)/cohort_std per feature
    z_records = []
    for _, row in merged_df.iterrows():
        cid = row.get('cohort_id')
        zvals = []
        for f in features:
            mean_col = f + '_mean'
            std_col = f + '_std'
            if cid in stats_df.index:
                mean = stats_df.loc[cid].get(mean_col, 0.0)
                std = stats_df.loc[cid].get(std_col, 0.0)
            else:
                mean = 0.0
                std = 0.0
            if std and std > 0:
                z = (row[f] - mean) / std
            else:
                z = 0.0
            zvals.append(z)
        z_records.append(np.abs(zvals))
    z_array = np.vstack(z_records)
    # cohort anomaly score: mean absolute z across features
    cohort_scores = z_array.mean(axis=1)
    merged_df = merged_df.copy()
    merged_df['cohort_anomaly_score'] = cohort_scores
    return merged_df


def load_ensemble(models_dir: str, feature_order: list):
    lgb_path = os.path.join(models_dir, 'lgb_model.joblib')
    xgb_path = os.path.join(models_dir, 'xgb_model.json')
    lgb = None
    xgbm = None
    if os.path.exists(lgb_path):
        try:
            lgb = joblib.load(lgb_path)
        except Exception:
            lgb = None
    if os.path.exists(xgb_path):
        try:
            xgbm = xgb.Booster()
            xgbm.load_model(xgb_path)
        except Exception:
            xgbm = None
    return lgb, xgbm


def compute_ensemble_probs(merged_df: pd.DataFrame, models_dir: str, feature_order: list):
    lgb, xgbm = load_ensemble(models_dir, feature_order)
    probs = np.zeros(len(merged_df))
    if lgb is None or xgbm is None:
        return None
    X = merged_df[feature_order].fillna(0.0).values
    try:
        p1 = lgb.predict(X)
    except Exception:
        p1 = np.zeros(len(X))
    try:
        p2 = xgbm.predict(xgb.DMatrix(X))
    except Exception:
        p2 = np.zeros(len(X))
    probs = 0.5 * np.array(p1) + 0.5 * np.array(p2)
    return probs


def evaluate_cohort_filter(merged_df: pd.DataFrame, probs: np.ndarray, labels: pd.Series, cohort_scores: np.ndarray, cohort_threshold: float = None):
    # baseline flags at prob >= 0.5
    if probs is None:
        raise ValueError('Model probabilities not provided')
    baseline_flags = probs >= 0.5
    baseline_fp = np.sum((baseline_flags) & (labels == 0))
    baseline_tp = np.sum((baseline_flags) & (labels == 1))
    baseline_flagged = np.sum(baseline_flags)

    # set cohort_threshold to percentile if None (e.g., 75th percentile)
    if cohort_threshold is None:
        cohort_threshold = np.percentile(cohort_scores, 75)

    # apply cohort filter: demote flags where cohort_score < threshold
    adjusted_flags = baseline_flags.copy()
    demote = (cohort_scores < cohort_threshold) & baseline_flags
    adjusted_flags[demote] = False
    adjusted_fp = np.sum((adjusted_flags) & (labels == 0))
    adjusted_tp = np.sum((adjusted_flags) & (labels == 1))
    adjusted_flagged = np.sum(adjusted_flags)

    results = {
        'baseline_flagged': int(baseline_flagged),
        'baseline_tp': int(baseline_tp),
        'baseline_fp': int(baseline_fp),
        'adjusted_flagged': int(adjusted_flagged),
        'adjusted_tp': int(adjusted_tp),
        'adjusted_fp': int(adjusted_fp),
        'fp_reduction_percent': float(100.0 * (baseline_fp - adjusted_fp) / baseline_fp) if baseline_fp>0 else 0.0,
        'tp_loss_percent': float(100.0 * (baseline_tp - adjusted_tp) / baseline_tp) if baseline_tp>0 else 0.0,
        'cohort_threshold': float(cohort_threshold)
    }
    return results


def main(args):
    # load selected features
    with open(args.selected, 'r') as f:
        sel = json.load(f)
    features = sel
    merged, stats = compute_cohort_stats(args.normalized, args.cohorts, features)
    merged = compute_cohort_zscores(merged, stats, features)
    probs = compute_ensemble_probs(merged, args.models, features)
    if probs is None:
        raise RuntimeError('Models not found in models dir; run training first')
    labels_map = pd.read_parquet(args.labels)
    merged = merged.merge(labels_map[['account_id','label']], on='account_id', how='left')
    labels = merged['label'].fillna(0).astype(int).values
    cohort_scores = merged['cohort_anomaly_score'].values
    results = evaluate_cohort_filter(merged, probs, labels, cohort_scores, cohort_threshold=None)
    # write report
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'cohort_filter_report.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('Cohort filter evaluation:')
    print(json.dumps(results, indent=2))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalized', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--cohorts', required=True)
    parser.add_argument('--models', default='models')
    parser.add_argument('--selected', default='models/selected_features.json')
    parser.add_argument('--out', default='models')
    args = parser.parse_args()
    main(args)
