"""Temporal FraudDNA backtest.

Runs an expanding-window temporal cross-validation over account-level events,
reports PR-AUC and F1 at the fold operating point, and breaks the results down
by cohort.

Usage:
  python -m ml.frauddna_backtest --normalized data/processed/normalized.parquet --labels data/processed/labels.parquet --cohorts data/processed/cohorts.parquet --out models/
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, precision_recall_curve

try:
    from imblearn.combine import SMOTETomek
except Exception:
    SMOTETomek = None


def _feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith('F') and c != 'F3924']


def _resample_if_needed(X, y):
    if SMOTETomek is None:
        return X, y
    smt = SMOTETomek(random_state=42)
    try:
        return smt.fit_resample(X, y)
    except Exception:
        return X, y


def _select_topk_features(model, features: list, topk: int):
    importance = getattr(model, 'feature_importances_', None)
    if importance is None:
        return features[:topk]
    idxs = np.argsort(-np.asarray(importance))[:min(topk, len(features))]
    return [features[i] for i in idxs]


def _load_cohorts(normalized_df: pd.DataFrame, cohorts_parquet: str | None, features: list) -> pd.DataFrame:
    if cohorts_parquet and os.path.exists(cohorts_parquet):
        cohorts = pd.read_parquet(cohorts_parquet)[['account_id', 'cohort_id']].copy()
        return cohorts

    aggregated = normalized_df.groupby('account_id')[features].mean().fillna(0.0)
    n_clusters = min(8, len(aggregated))
    if n_clusters <= 1:
        return pd.DataFrame({'account_id': aggregated.index, 'cohort_id': 0})

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cohort_ids = kmeans.fit_predict(aggregated.values)
    return pd.DataFrame({'account_id': aggregated.index, 'cohort_id': cohort_ids})


def load_account_frame(normalized_parquet: str, labels_parquet: str, cohorts_parquet: str | None = None):
    df = pd.read_parquet(normalized_parquet)
    labels = pd.read_parquet(labels_parquet)
    features = _feature_columns(df)

    if 'label' not in labels.columns:
        if 'F3924' not in df.columns:
            raise ValueError('labels parquet must contain a label column when normalized data has no F3924 column')
        labels = df.groupby('account_id')['F3924'].max().reset_index().rename(columns={'F3924': 'label'})

    account_features = df.groupby('account_id')[features].mean().reset_index()
    timestamps = df.groupby('account_id')['timestamp'].agg(['min', 'max']).reset_index()
    activation = df[df['F3924'] == 1].groupby('account_id')['timestamp'].min().reset_index().rename(columns={'timestamp': 'activation_timestamp'})

    account_frame = account_features.merge(timestamps, on='account_id', how='left')
    account_frame = account_frame.merge(activation, on='account_id', how='left')
    account_frame = account_frame.merge(labels[['account_id', 'label']], on='account_id', how='inner')
    account_frame = account_frame.merge(_load_cohorts(df, cohorts_parquet, features), on='account_id', how='left')
    account_frame['event_time'] = account_frame['activation_timestamp'].fillna(account_frame['max'])
    account_frame['event_time'] = pd.to_datetime(account_frame['event_time'])
    account_frame['cohort_id'] = account_frame['cohort_id'].fillna(-1).astype(int)
    account_frame['label'] = account_frame['label'].astype(int)
    account_frame = account_frame.sort_values(['event_time', 'account_id']).reset_index(drop=True)
    return account_frame, features


def build_temporal_folds(account_frame: pd.DataFrame, n_splits: int):
    unique_times = np.array(sorted(pd.Series(account_frame['event_time'].dropna().unique())))
    if len(unique_times) < 2:
        return []

    n_bins = min(max(2, n_splits), len(unique_times))
    edge_indexes = np.linspace(0, len(unique_times), num=n_bins + 1, dtype=int)
    folds = []

    for fold_idx in range(1, len(edge_indexes) - 1):
        train_end_idx = edge_indexes[fold_idx] - 1
        test_start_idx = edge_indexes[fold_idx]
        test_end_idx = edge_indexes[fold_idx + 1] - 1
        if train_end_idx < 0 or test_start_idx >= len(unique_times) or test_end_idx < test_start_idx:
            continue

        train_end = unique_times[train_end_idx]
        test_start = unique_times[test_start_idx]
        test_end = unique_times[test_end_idx]

        train_mask = account_frame['event_time'] <= train_end
        test_mask = (account_frame['event_time'] >= test_start) & (account_frame['event_time'] <= test_end)
        train_df = account_frame.loc[train_mask].copy()
        test_df = account_frame.loc[test_mask].copy()

        if train_df.empty or test_df.empty:
            continue

        folds.append({
            'fold': fold_idx,
            'train_start': str(pd.Timestamp(train_df['event_time'].min())),
            'train_end': str(pd.Timestamp(train_df['event_time'].max())),
            'test_start': str(pd.Timestamp(test_df['event_time'].min())),
            'test_end': str(pd.Timestamp(test_df['event_time'].max())),
            'train_df': train_df,
            'test_df': test_df,
        })

    return folds


def _safe_pr_auc(y_true: np.ndarray, scores: np.ndarray):
    if len(np.unique(y_true)) < 2:
        return None
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return float(auc(recall, precision))


def _best_f1_threshold(y_true: np.ndarray, scores: np.ndarray):
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0

    f1_scores = 2.0 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    if len(f1_scores) == 0:
        return 0.5, 0.0

    best_idx = int(np.nanargmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def _cohort_metrics(frame: pd.DataFrame, threshold: float):
    rows = []
    for cohort_id, subset in frame.groupby('cohort_id'):
        y_true = subset['label'].astype(int).values
        scores = subset['score'].astype(float).values
        preds = (scores >= threshold).astype(int)
        pr_auc = _safe_pr_auc(y_true, scores)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append({
            'cohort_id': int(cohort_id),
            'support': int(len(subset)),
            'positives': int(np.sum(y_true == 1)),
            'pr_auc': pr_auc,
            'precision': precision,
            'recall': recall,
            'f1_at_operating_point': f1,
        })
    return rows


def _fit_fold_model(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list, topk: int):
    X_train = train_df[features].fillna(0.0)
    y_train = train_df['label'].astype(int)
    X_test = test_df[features].fillna(0.0)

    if y_train.nunique() < 2:
        return None, None

    Xr, yr = _resample_if_needed(X_train.values, y_train.values)
    Xr_df = pd.DataFrame(Xr, columns=features)

    probe = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced_subsample')
    probe.fit(Xr, yr)

    try:
        selected = _select_topk_features(probe, features, topk)
    except Exception:
        selected = features[:min(topk, len(features))]

    Xr_sel = Xr_df[selected].values
    X_test_sel = X_test[selected].values

    model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced_subsample')
    model.fit(Xr_sel, yr)

    scores = model.predict_proba(X_test_sel)[:, 1]
    return scores, selected


def backtest(normalized_parquet: str, labels_parquet: str, cohorts_parquet: str | None = None, out_dir: str = 'models', n_splits: int = 4, topk: int = 50):
    account_frame, features = load_account_frame(normalized_parquet, labels_parquet, cohorts_parquet=cohorts_parquet)
    folds = build_temporal_folds(account_frame, n_splits=n_splits)
    if not folds:
        raise ValueError('Not enough temporal diversity to create backtest folds')

    os.makedirs(out_dir, exist_ok=True)
    oof_rows = []
    fold_reports = []

    for fold in folds:
        scores, selected_features = _fit_fold_model(fold['train_df'], fold['test_df'], features, topk=topk)
        if scores is None:
            continue

        fold_pred_frame = fold['test_df'][['account_id', 'event_time', 'label', 'cohort_id']].copy()
        fold_pred_frame['score'] = scores
        threshold, fold_f1 = _best_f1_threshold(fold_pred_frame['label'].values, fold_pred_frame['score'].values)
        fold_pred_frame['prediction'] = (fold_pred_frame['score'] >= threshold).astype(int)

        tp = int(np.sum((fold_pred_frame['prediction'] == 1) & (fold_pred_frame['label'] == 1)))
        fp = int(np.sum((fold_pred_frame['prediction'] == 1) & (fold_pred_frame['label'] == 0)))
        fn = int(np.sum((fold_pred_frame['prediction'] == 0) & (fold_pred_frame['label'] == 1)))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        fold_report = {
            'fold': fold['fold'],
            'train_start': fold['train_start'],
            'train_end': fold['train_end'],
            'test_start': fold['test_start'],
            'test_end': fold['test_end'],
            'n_train': int(len(fold['train_df'])),
            'n_test': int(len(fold['test_df'])),
            'selected_features': selected_features,
            'pr_auc': _safe_pr_auc(fold_pred_frame['label'].values, fold_pred_frame['score'].values),
            'operating_threshold': threshold,
            'precision_at_operating_point': precision,
            'recall_at_operating_point': recall,
            'f1_at_operating_point': float(fold_f1),
            'cohort_metrics': _cohort_metrics(fold_pred_frame, threshold),
        }
        fold_reports.append(fold_report)
        oof_rows.append(fold_pred_frame.assign(fold=fold['fold']))

    if not oof_rows:
        raise RuntimeError('Temporal folds were generated but no fold produced predictions')

    oof = pd.concat(oof_rows, ignore_index=True)
    global_threshold, global_f1 = _best_f1_threshold(oof['label'].values, oof['score'].values)
    oof['prediction'] = (oof['score'] >= global_threshold).astype(int)

    tp = int(np.sum((oof['prediction'] == 1) & (oof['label'] == 1)))
    fp = int(np.sum((oof['prediction'] == 1) & (oof['label'] == 0)))
    fn = int(np.sum((oof['prediction'] == 0) & (oof['label'] == 1)))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    report = {
        'n_accounts': int(account_frame['account_id'].nunique()),
        'n_folds': int(len(fold_reports)),
        'overall': {
            'pr_auc': _safe_pr_auc(oof['label'].values, oof['score'].values),
            'operating_threshold': global_threshold,
            'precision_at_operating_point': precision,
            'recall_at_operating_point': recall,
            'f1_at_operating_point': float(global_f1),
        },
        'folds': fold_reports,
        'cohort_summary': _cohort_metrics(oof, global_threshold),
    }

    oof_out = os.path.join(out_dir, 'frauddna_backtest_predictions.parquet')
    report_out = os.path.join(out_dir, 'frauddna_backtest_report.json')
    oof.to_parquet(oof_out, index=False)
    with open(report_out, 'w') as f:
        json.dump(report, f, indent=2)

    return report


def main(args):
    report = backtest(
        normalized_parquet=args.normalized,
        labels_parquet=args.labels,
        cohorts_parquet=args.cohorts,
        out_dir=args.out,
        n_splits=args.splits,
        topk=args.topk,
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalized', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--cohorts', default=None)
    parser.add_argument('--out', default='models')
    parser.add_argument('--splits', type=int, default=4)
    parser.add_argument('--topk', type=int, default=50)
    args = parser.parse_args()
    main(args)