import os
import joblib
import pandas as pd
import numpy as np
from river.drift import ADWIN
from typing import Dict, Any
import logging
import subprocess

logger = logging.getLogger('muleguard.drift')


class DriftMonitor:
    def __init__(self, delta: float = 0.002):
        # ADWIN sensitivity parameter; smaller delta -> more sensitive
        self.adwin = ADWIN(delta=delta)

    def feed(self, value: float) -> bool:
        """Feed a numeric value (e.g., absolute error) and return True if drift detected."""
        self.adwin.update(value)
        return self.adwin.change_detected


def detect_drift_on_validation(model_path: str, normalized_parquet: str, labels_parquet: str,
                                feature_order: list = None, error_metric: str = 'abs_error',
                                delta: float = 0.002) -> Dict[str, Any]:
    """
    Stream through validation data and run ADWIN on the chosen error metric.
    Returns {'drift': bool, 'samples_seen': int, 'last_value': float}
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    # load model (assume joblib LightGBM or sklearn-like)
    try:
        model = joblib.load(model_path)
    except Exception:
        model = None

    if model is None:
        raise RuntimeError('Could not load model for drift detection')

    if not os.path.exists(normalized_parquet) or not os.path.exists(labels_parquet):
        raise FileNotFoundError('Validation data not found')

    X = pd.read_parquet(normalized_parquet)
    y = pd.read_parquet(labels_parquet)
    # Align rows
    df = X.join(y.set_index('account_id'), on='account_id', how='inner') if 'account_id' in y.columns else X.join(y)

    monitor = DriftMonitor(delta=delta)
    last_val = None
    seen = 0
    feature_names = feature_order or (getattr(model, 'feature_name', lambda: list(X.columns))())

    for _, row in df.iterrows():
        seen += 1
        x = np.array([row.get(fn, 0.0) for fn in feature_names], dtype=float).reshape(1, -1)
        try:
            if hasattr(model, 'predict_proba'):
                p = float(model.predict_proba(x)[0][1])
            else:
                p = float(model.predict(x))
        except Exception:
            p = 0.0
        true = float(row.get('label', 0.0)) if 'label' in row else float(row.iloc[-1])
        if error_metric == 'abs_error':
            val = abs(p - true)
        else:
            val = abs(p - true)
        last_val = val
        drifted = monitor.feed(val)
        if drifted:
            logger.warning('Drift detected after %d samples (val=%f)', seen, val)
            return {'drift': True, 'samples_seen': seen, 'last_value': float(val)}

    return {'drift': False, 'samples_seen': seen, 'last_value': float(last_val) if last_val is not None else 0.0}


def trigger_retrain_and_shadow_eval(train_cmd: list, shadow_eval_cmd: list = None) -> Dict[str, Any]:
    """Run the training command (as subprocess). Optionally run shadow evaluation command.
    Returns dict with 'trained': bool and shadow eval results if any.
    """
    res = {'trained': False, 'train_returncode': None, 'shadow': None}
    logger.info('Starting retrain: %s', ' '.join(train_cmd))
    p = subprocess.run(train_cmd, capture_output=True, text=True)
    res['train_returncode'] = p.returncode
    res['train_stdout'] = p.stdout
    res['train_stderr'] = p.stderr
    res['trained'] = p.returncode == 0

    if shadow_eval_cmd and res['trained']:
        logger.info('Running shadow evaluation: %s', ' '.join(shadow_eval_cmd))
        q = subprocess.run(shadow_eval_cmd, capture_output=True, text=True)
        res['shadow'] = {'returncode': q.returncode, 'stdout': q.stdout, 'stderr': q.stderr}

    return res
