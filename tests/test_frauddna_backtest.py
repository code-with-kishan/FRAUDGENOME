import os
import tempfile

import pandas as pd

from ml.frauddna_backtest import backtest


def _make_temporal_backtest_fixture(tmpdir):
    rows = []
    cohorts = []
    labels = []

    base = pd.Timestamp('2024-01-01')
    for account_idx in range(12):
        account_id = f'A{account_idx:02d}'
        cohort_id = account_idx % 3
        is_positive = int(account_idx % 4 == 3)
        labels.append({'account_id': account_id, 'label': is_positive, 'split': 'train'})
        cohorts.append({'account_id': account_id, 'cohort_id': cohort_id})

        start = base + pd.Timedelta(days=account_idx * 3)
        for day in range(6):
            timestamp = start + pd.Timedelta(days=day)
            signal = float(account_idx) / 10.0
            if is_positive and day >= 3:
                signal += 2.5
            rows.append({
                'account_id': account_id,
                'timestamp': timestamp,
                'F1': signal,
                'F2': signal * 0.8 + cohort_id,
                'F3': signal * 0.5,
                'F3924': is_positive if day == 4 else 0,
            })

    normalized = pd.DataFrame(rows)
    labels_df = pd.DataFrame(labels)
    cohorts_df = pd.DataFrame(cohorts)

    normalized_path = os.path.join(tmpdir, 'normalized.parquet')
    labels_path = os.path.join(tmpdir, 'labels.parquet')
    cohorts_path = os.path.join(tmpdir, 'cohorts.parquet')

    normalized.to_parquet(normalized_path, index=False)
    labels_df.to_parquet(labels_path, index=False)
    cohorts_df.to_parquet(cohorts_path, index=False)

    return normalized_path, labels_path, cohorts_path


def test_temporal_backtest_reports_pr_auc_and_f1_by_cohort():
    tmpdir = tempfile.mkdtemp()
    normalized_path, labels_path, cohorts_path = _make_temporal_backtest_fixture(tmpdir)
    out_dir = os.path.join(tmpdir, 'models')

    report = backtest(
        normalized_parquet=normalized_path,
        labels_parquet=labels_path,
        cohorts_parquet=cohorts_path,
        out_dir=out_dir,
        n_splits=4,
        topk=3,
    )

    assert report['n_folds'] >= 1
    assert 'pr_auc' in report['overall']
    assert 'f1_at_operating_point' in report['overall']
    assert report['overall']['f1_at_operating_point'] >= 0.0
    assert len(report['cohort_summary']) == 3
    assert os.path.exists(os.path.join(out_dir, 'frauddna_backtest_report.json'))
    assert os.path.exists(os.path.join(out_dir, 'frauddna_backtest_predictions.parquet'))