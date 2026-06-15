"""Data ingestion, normalization, window extraction, labeling, and sampling utilities.

Usage examples (commands provided as convenience wrappers):
  python -m ml.data_pipeline ingest --input DataSet.csv --out data/processed/
  python -m ml.data_pipeline normalize --in data/processed/raw.parquet --out data/processed/normalized.parquet
  python -m ml.data_pipeline windows --in data/processed/normalized.parquet --anchors F321 F3836 F2082 --window-days 30 --stride 7 --out data/processed/windows/
  python -m ml.data_pipeline split --in data/processed/normalized.parquet --out data/processed/labels.parquet --train 0.7 --val 0.2
"""

import argparse
import os
import uuid
import hashlib
import json
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import joblib

from ml.schema import REQUIRED_COLUMNS, ANCHOR_FEATURES


def _schema_signature(columns: List[str]) -> str:
    return hashlib.sha256("|".join(columns).encode("utf-8")).hexdigest()[:16]


def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _write_metadata(metadata_path: str, payload: Dict[str, Any]) -> None:
    _ensure_dir_for_file(metadata_path)
    # Merge with existing metadata.json (best-effort, to avoid accidental overwrite)
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                existing.update(payload)
                payload_to_write = existing
            else:
                payload_to_write = payload
        else:
            payload_to_write = payload
    except Exception:
        payload_to_write = payload

    with open(metadata_path, "w") as f:
        json.dump(payload_to_write, f, indent=2, default=str)


def _coerce_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _basic_dedup_account_timestamp(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if "account_id" not in df.columns or "timestamp" not in df.columns:
        return df, 0
    before = int(len(df))
    df = df.sort_values(["account_id", "timestamp"]).drop_duplicates(subset=["account_id", "timestamp"], keep="last")
    after = int(len(df))
    return df, max(0, before - after)


def _profile_quality(df: pd.DataFrame) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            continue
        null_count = int(df[c].isna().sum())
        total = int(len(df))
        null_pct = float(null_count / max(total, 1))
        profile[f"{c}__null_count"] = null_count
        profile[f"{c}__null_pct"] = round(null_pct, 6)

    # simple numeric coercion checks for anchors
    for c in ANCHOR_FEATURES + ["F3924"]:
        if c not in df.columns:
            continue
        coerced = pd.to_numeric(df[c], errors="coerce")
        bad_count = int(coerced.isna().sum())
        profile[f"{c}__coercion_nan_count"] = bad_count

    # timestamp monotonicity per account
    if "account_id" in df.columns and "timestamp" in df.columns:
        mono_violations = 0
        for _, g in df.groupby("account_id"):
            ts = g["timestamp"].values
            if len(ts) <= 1:
                continue
            if np.any(pd.to_datetime(ts[1:]) < pd.to_datetime(ts[:-1])):
                mono_violations += 1
        profile["timestamp__monotonic_violations_accounts"] = int(mono_violations)

    # derive quality score
    # Start at 100; penalize missing required values and coercion NaNs
    penalty = 0.0
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            penalty += 0.25
            continue
        total = max(len(df), 1)
        penalty += float(df[c].isna().sum()) / total * 1.2

    # cap and convert
    score = max(0.0, 100.0 - 100.0 * min(0.95, penalty / 5.0))
    profile["data_quality_score"] = round(float(score), 3)
    return profile


def ingest(input_path: str, out_dir: str) -> str:
    """
    Ingest input CSV or Parquet and write out raw.parquet.

    Also writes metadata.json in out_dir with schema signature, duplicates, and quality summary.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "raw.parquet")
    metadata_path = os.path.join(out_dir, "metadata.json")

    if input_path.lower().endswith(".parquet"):
        df = pd.read_parquet(input_path)
    elif input_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    df = _coerce_timestamp(df)

    # standardize column names (minimal)
    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
        df = _coerce_timestamp(df)

    # check required columns existence early (schema validation)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df, dup_count = _basic_dedup_account_timestamp(df)

    # basic type coercion for required columns
    df["account_id"] = df["account_id"].astype(str)
    # keep F3924 as-is; normalize() will coerce numeric anchors
    df.to_parquet(out_path, index=False)

    quality = _profile_quality(df)
    payload = {
        "stage": "ingest",
        "input_path": input_path,
        "output_path": out_path,
        "schema_signature": _schema_signature(list(REQUIRED_COLUMNS)),
        "duplicates__account_id_timestamp_removed": int(dup_count),
        **quality,
    }
    _write_metadata(metadata_path, payload)
    return out_path


def normalize(in_parquet: str, out_parquet: str, imputation: str = 'zero', scaler: str = 'none') -> str:
    """
    Clean and normalize raw.parquet into normalized.parquet.

    Adds:
    - stronger missing handling
    - outlier detection summaries
    - feature consistency checks
    - data_quality_score
    - additional derived features (derived columns; existing required columns preserved)
    Also writes/updates metadata.json next to out_parquet.
    """
    df = pd.read_parquet(in_parquet)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["account_id"] = df["account_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

    # Missing value handling for required numeric columns
    for col in ["F3924"] + ANCHOR_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Data cleaning: null patterns analysis
    null_counts = {c: int(df[c].isna().sum()) for c in (["F3924"] + ANCHOR_FEATURES) if c in df.columns}

    # Outlier detection (simple robust z-score on anchors and F3924 within each column)
    outlier_summary: Dict[str, Any] = {}
    for col in ["F3924"] + ANCHOR_FEATURES:
        series = df[col].astype(float)
        median = float(series.median(skipna=True)) if series.notna().any() else 0.0
        mad = float(np.median(np.abs(series - median))) if series.notna().any() else 0.0
        if np.isfinite(mad) and mad > 0:
            robust_z = 0.6745 * (series - median) / mad
            outlier_rate = float((np.abs(robust_z) > 6.0).mean()) if len(series) else 0.0
            outlier_summary[f"{col}__outlier_rate"] = round(outlier_rate, 6)
        else:
            outlier_summary[f"{col}__outlier_rate"] = 0.0

    # Imputation rules:
    for col in ANCHOR_FEATURES:
        if imputation == 'mean':
            val = df[col].mean(skipna=True)
        elif imputation == 'median':
            val = df[col].median(skipna=True)
        else:
            val = 0.0
        if pd.isna(val):
            val = 0.0
        df[col] = df[col].fillna(val).astype(float)

    df["F3924"] = df["F3924"].fillna(0.0).astype(float)
    df["F3924"] = (df["F3924"] > 0.5).astype(int)

    # Feature Scaling logic
    if scaler != 'none':
        if scaler == 'standard':
            s = StandardScaler()
        elif scaler == 'robust':
            s = RobustScaler()
        elif scaler == 'minmax':
            s = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler: {scaler}")
            
        df[ANCHOR_FEATURES] = s.fit_transform(df[ANCHOR_FEATURES])
        # Save the scaler model
        scaler_path = os.path.join(os.path.dirname(out_parquet), f"scaler_{scaler}.joblib")
        joblib.dump(s, scaler_path)

    # Feature consistency checks
    # - timestamp strictly non-decreasing per account after sorting (should be true; record any violations)
    timestamp_mono_violations = 0
    for _, g in df.groupby("account_id"):
        ts = g["timestamp"].values
        if len(ts) > 1 and np.any(pd.to_datetime(ts[1:]) < pd.to_datetime(ts[:-1])):
            timestamp_mono_violations += 1

    # Engineered columns:
    # base account age
    acct_min = df.groupby("account_id")["timestamp"].transform("min")
    df["account_age_days"] = (df["timestamp"] - acct_min).dt.days.astype(int)

    # rolling/statistical features per account over time
    # Set index for rolling windows per account
    df = df.set_index("timestamp", drop=False)
    g = df.groupby("account_id", group_keys=False)

    # rolling windows on anchors
    for col in ANCHOR_FEATURES:
        df[f"{col}__roll7_mean"] = g[col].rolling("7D", min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{col}__roll14_mean"] = g[col].rolling("14D", min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{col}__roll30_mean"] = g[col].rolling("30D", min_periods=1).mean().reset_index(level=0, drop=True)

    # velocity proxy: slope of F3836 over last 14 days (simple linear regression on daily series)
    if "F3836" in df.columns:
        daily = (
            df.groupby("account_id")
            .resample("1D", on="timestamp")[["F3836", "F321", "F2082", "F3924"]]
            .mean()
            .reset_index()
        )
        daily = daily.sort_values(["account_id", "timestamp"])
        daily["F3836__velocity14_slope"] = 0.0

        def _slope(x: np.ndarray) -> float:
            if x.size < 2:
                return 0.0
            t = np.arange(x.size, dtype=float)
            denom = float(np.var(t))
            if denom <= 0:
                return 0.0
            return float(np.cov(t, x, bias=True)[0, 1] / denom)

        for acct_id, acct_daily in daily.groupby("account_id"):
            idx = acct_daily.index
            vals = acct_daily["F3836"].astype(float).values
            slopes = np.zeros_like(vals, dtype=float)
            for i in range(len(vals)):
                start = max(0, i - 13)
                slopes[i] = _slope(vals[start : i + 1])
            daily.loc[idx, "F3836__velocity14_slope"] = slopes

        # Align back to original df
        df['date'] = df['timestamp'].dt.normalize()
        daily['date'] = daily['timestamp'].dt.normalize()
        df = df.merge(daily[['account_id', 'date', 'F3836__velocity14_slope']], on=['account_id', 'date'], how='left')
        df = df.drop(columns=['date'])
        df['F3836__velocity14_slope'] = df['F3836__velocity14_slope'].fillna(0.0)

    # Save output
    df = df.reset_index(drop=True)
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    metadata_path = os.path.join(os.path.dirname(out_parquet), "metadata.json")
    payload = {
        "stage": "normalize",
        "input_path": in_parquet,
        "output_path": out_parquet,
        "imputation": imputation,
        "scaler": scaler,
        "timestamp_monotonicity_violations": int(timestamp_mono_violations),
        **outlier_summary
    }
    _write_metadata(metadata_path, payload)
    return out_parquet



def _resample_to_daily(account_df: pd.DataFrame, anchors: List[str], window_start, window_end) -> np.ndarray:
    # produce daily summary of anchor features between start and end, both inclusive
    dr = pd.date_range(window_start.normalize(), window_end.normalize(), freq='D')
    if len(dr) == 0:
        return np.zeros((0, len(anchors)), dtype=float)
    tmp = account_df.set_index('timestamp')
    daily = tmp[anchors].loc[window_start:window_end].resample('D').mean().reindex(dr).fillna(0.0)
    return daily.values


def extract_windows(in_parquet: str, out_dir: str, anchors: List[str]=ANCHOR_FEATURES, window_days: int=30, stride_days: int=7) -> str:
    df = pd.read_parquet(in_parquet)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    accounts = df['account_id'].unique()
    for acct in accounts:
        acct_df = df[df['account_id'] == acct].copy()
        # determine label timestamp if positive
        pos_rows = acct_df[acct_df['F3924'] == 1]
        if not pos_rows.empty:
            label_time = pos_rows['timestamp'].iloc[0]
            # extract multiple windows ending on the activation timestamp so the
            # first window captures the confirmed-mule day and prior behaviour
            end = label_time
            # create windows: end at end - k*stride, for k until we go before earliest timestamp
            k = 0
            while True:
                window_end = end - pd.Timedelta(days=k*stride_days)
                window_start = window_end - pd.Timedelta(days=window_days)
                if window_start < acct_df['timestamp'].min():
                    break
                arr = _resample_to_daily(acct_df, anchors, window_start, window_end)
                if arr.shape[0] == 0:
                    break
                pattern_id = str(uuid.uuid4())
                fname = f'{acct}__{pattern_id}.npy'
                np.save(os.path.join(out_dir, fname), arr)
                manifest.append({'account_id': acct, 'pattern_id': pattern_id, 'window_start': window_start, 'window_end': window_end, 'activation_timestamp': label_time, 'file_path': fname, 'label': 1})
                k += 1
        else:
            # negatives: produce a single rolling window ending at last observed timestamp
            end = acct_df['timestamp'].max()
            window_end = end
            window_start = window_end - pd.Timedelta(days=window_days)
            arr = _resample_to_daily(acct_df, anchors, window_start, window_end)
            if arr.shape[0] > 0:
                pattern_id = str(uuid.uuid4())
                fname = f'{acct}__{pattern_id}.npy'
                np.save(os.path.join(out_dir, fname), arr)
                manifest.append({'account_id': acct, 'pattern_id': pattern_id, 'window_start': window_start, 'window_end': window_end, 'activation_timestamp': pd.NaT, 'file_path': fname, 'label': 0})

    manifest_df = pd.DataFrame(manifest)
    manifest_path = os.path.join(out_dir, 'manifest.parquet')
    manifest_df.to_parquet(manifest_path, index=False)
    return manifest_path


def _deterministic_pattern_id(account_id: str, window_start: pd.Timestamp, window_end: pd.Timestamp) -> str:
    key = f"{account_id}|{window_start.isoformat()}|{window_end.isoformat()}"
    return hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]


def extract_windows_repro(in_parquet: str, out_dir: str, anchors: List[str]=ANCHOR_FEATURES, window_days_list: List[int]=[30], stride_days: int=7, anchor_policy: str='first') -> str:
    """Deterministic window extraction.

    - `window_days_list` may contain multiple window sizes (e.g. [14,30]).
    - `anchor_policy` chooses 'first' or 'last' activation timestamp when multiple labels exist.
    Pattern ids and filenames are deterministic (SHA1 of account and window bounds).
    Saves an `anchors.parquet` file in `out_dir` recording per-account activation timestamps.
    """
    df = pd.read_parquet(in_parquet)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    anchors = list(anchors)

    # compute activation anchors per account (reproducible)
    acct_anchors = []
    for acct, acct_df in df.groupby('account_id'):
        pos_rows = acct_df[acct_df['F3924'] == 1]
        if not pos_rows.empty:
            if anchor_policy == 'last':
                activation = pos_rows['timestamp'].max()
            else:
                activation = pos_rows['timestamp'].min()
        else:
            activation = pd.NaT
        acct_anchors.append({'account_id': acct, 'activation_timestamp': activation})
    anchors_df = pd.DataFrame(acct_anchors)
    anchors_df.to_parquet(os.path.join(out_dir, 'anchors.parquet'), index=False)

    accounts = df['account_id'].unique()
    for acct in accounts:
        acct_df = df[df['account_id'] == acct].copy()
        pos_rows = acct_df[acct_df['F3924'] == 1]
        if not pos_rows.empty:
            if anchor_policy == 'last':
                label_time = pos_rows['timestamp'].iloc[-1]
            else:
                label_time = pos_rows['timestamp'].iloc[0]
            # for each requested window size produce windows backward from label_time
            for window_days in window_days_list:
                k = 0
                while True:
                    window_end = label_time - pd.Timedelta(days=k*stride_days)
                    window_start = window_end - pd.Timedelta(days=window_days)
                    if window_start < acct_df['timestamp'].min():
                        break
                    arr = _resample_to_daily(acct_df, anchors, window_start, window_end)
                    if arr.shape[0] == 0:
                        break
                    pattern_id = _deterministic_pattern_id(acct, pd.to_datetime(window_start), pd.to_datetime(window_end))
                    fname = f'{acct}__{pattern_id}.npy'
                    np.save(os.path.join(out_dir, fname), arr)
                    manifest.append({'account_id': acct, 'pattern_id': pattern_id, 'window_start': window_start, 'window_end': window_end, 'activation_timestamp': label_time, 'file_path': fname, 'label': 1, 'window_days': int(window_days)})
                    k += 1
        else:
            # negatives: for each window size produce a single deterministic window ending at last observed timestamp
            end = acct_df['timestamp'].max()
            for window_days in window_days_list:
                window_end = end
                window_start = window_end - pd.Timedelta(days=window_days)
                arr = _resample_to_daily(acct_df, anchors, window_start, window_end)
                if arr.shape[0] > 0:
                    pattern_id = _deterministic_pattern_id(acct, pd.to_datetime(window_start), pd.to_datetime(window_end))
                    fname = f'{acct}__{pattern_id}.npy'
                    np.save(os.path.join(out_dir, fname), arr)
                    manifest.append({'account_id': acct, 'pattern_id': pattern_id, 'window_start': window_start, 'window_end': window_end, 'activation_timestamp': pd.NaT, 'file_path': fname, 'label': 0, 'window_days': int(window_days)})

    manifest_df = pd.DataFrame(manifest)
    manifest_path = os.path.join(out_dir, 'manifest.parquet')
    manifest_df.to_parquet(manifest_path, index=False)
    return manifest_path


def split_accounts(in_parquet: str, out_path: str, train_frac: float=0.7, val_frac: float=0.2, random_state: int=42) -> str:
    df = pd.read_parquet(in_parquet)
    acct_summary = df.groupby('account_id').agg({
        'F3924': 'max'
    }).reset_index()
    acct_summary['label'] = acct_summary['F3924'].astype(int)

    n_samples = len(acct_summary)
    if n_samples < 3:
        splits_list = []
        if n_samples >= 1:
            splits_list.append(acct_summary.iloc[[0]].assign(split='train'))
        if n_samples >= 2:
            splits_list.append(acct_summary.iloc[[1]].assign(split='val'))
        splits = pd.concat(splits_list, axis=0)
    else:
        try:
            train, temp = train_test_split(acct_summary, train_size=train_frac, stratify=acct_summary['label'], random_state=random_state)
        except ValueError:
            train, temp = train_test_split(acct_summary, train_size=train_frac, random_state=random_state)
        
        val_size = val_frac / (1.0 - train_frac)
        try:
            val, test = train_test_split(temp, train_size=val_size, stratify=temp['label'], random_state=random_state)
        except ValueError:
            try:
                val, test = train_test_split(temp, train_size=val_size, random_state=random_state)
            except ValueError:
                val = temp.assign(split='val')
                test = pd.DataFrame(columns=temp.columns).assign(split='test')

        splits = pd.concat([
            train.assign(split='train'),
            val.assign(split='val'),
            test.assign(split='test')
        ], axis=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    splits[['account_id','label','split']].to_parquet(out_path, index=False)
    return out_path


def create_cohorts(in_parquet: str, out_path: str, n_clusters: int=8, anchors: List[str]=ANCHOR_FEATURES) -> str:
    df = pd.read_parquet(in_parquet)
    agg = df.groupby('account_id')[anchors].mean().fillna(0.0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cohort_ids = kmeans.fit_predict(agg.values)
    cohorts = pd.DataFrame({'account_id': agg.index, 'cohort_id': cohort_ids})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cohorts.to_parquet(out_path, index=False)
    return out_path


def get_batch(manifest_parquet: str, split_parquet: str, split: str='train', batch_size: int=32, anchors: List[str]=ANCHOR_FEATURES, out_dir: str=None) -> Tuple[np.ndarray, np.ndarray, list]:
    manifest = pd.read_parquet(manifest_parquet)
    splits = pd.read_parquet(split_parquet)
    acct_ids = splits[splits['split'] == split]['account_id'].unique()
    candidates = manifest[manifest['account_id'].isin(acct_ids)]
    # sample
    sampled = candidates.sample(n=min(batch_size, len(candidates)), random_state=42)
    X = []
    y = []
    meta = []
    for _, row in sampled.iterrows():
        arr = np.load(os.path.join(os.path.dirname(manifest_parquet), row['file_path']))
        X.append(arr)
        y.append(int(row['label']))
        meta.append({'account_id': row['account_id'], 'pattern_id': row['pattern_id']})
    return np.array(X, dtype=object), np.array(y), meta


def _cli():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    p_ing = sub.add_parser('ingest')
    p_ing.add_argument('--input', required=True)
    p_ing.add_argument('--out', required=True)

    p_norm = sub.add_parser('normalize')
    p_norm.add_argument('--in', dest='in_path', required=True)
    p_norm.add_argument('--out', dest='out_path', required=True)

    p_win = sub.add_parser('windows')
    p_win.add_argument('--in', dest='in_path', required=True)
    p_win.add_argument('--out', dest='out_dir', required=True)
    p_win.add_argument('--window-days', type=int, default=30)
    p_win.add_argument('--stride', type=int, default=7)

    p_win_repro = sub.add_parser('windows-repro')
    p_win_repro.add_argument('--in', dest='in_path', required=True)
    p_win_repro.add_argument('--out', dest='out_dir', required=True)
    p_win_repro.add_argument('--window-days', dest='window_days', type=str, default='30',
                            help='Comma-separated list of window sizes in days, e.g. "14,30"')
    p_win_repro.add_argument('--stride', type=int, default=7)
    p_win_repro.add_argument('--anchor-policy', choices=['first','last'], default='first')

    p_split = sub.add_parser('split')
    p_split.add_argument('--in', dest='in_path', required=True)
    p_split.add_argument('--out', dest='out_path', required=True)
    p_split.add_argument('--train', type=float, default=0.7)
    p_split.add_argument('--val', type=float, default=0.2)

    p_cohort = sub.add_parser('cohorts')
    p_cohort.add_argument('--in', dest='in_path', required=True)
    p_cohort.add_argument('--out', dest='out_path', required=True)
    p_cohort.add_argument('--n', type=int, default=8)

    args = parser.parse_args()
    if args.cmd == 'ingest':
        print(ingest(args.input, args.out))
    elif args.cmd == 'normalize':
        print(normalize(args.in_path, args.out_path))
    elif args.cmd == 'windows':
        print(extract_windows(args.in_path, args.out_dir, window_days=args.window_days, stride_days=args.stride))
    elif args.cmd == 'windows-repro':
        # parse comma separated window days into list of ints
        window_days_list = [int(x.strip()) for x in args.window_days.split(',') if x.strip()]
        print(extract_windows_repro(args.in_path, args.out_dir, window_days_list=window_days_list, stride_days=args.stride, anchor_policy=args.anchor_policy))
    elif args.cmd == 'split':
        print(split_accounts(args.in_path, args.out_path, train_frac=args.train, val_frac=args.val))
    elif args.cmd == 'cohorts':
        print(create_cohorts(args.in_path, args.out_path, n_clusters=args.n))
    else:
        parser.print_help()


if __name__ == '__main__':
    _cli()
