import os
import joblib
import numpy as np
import pandas as pd
import shap

def save_shap_sample(model, X_sample: pd.DataFrame, out_path: str):
    """Compute and save SHAP explainer and sample values for later API use."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({'explainer': explainer, 'X_sample': X_sample, 'shap_values': shap_values}, out_path)
    return out_path


def compute_shap_for_row(model, X_row: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)
    # return feature names, shap values for class 1 if binary
    if isinstance(shap_values, list):
        vals = shap_values[1][0]
    else:
        vals = shap_values[0]
    return dict(zip(X_row.columns.tolist(), vals.tolist()))


def topk_plain_english(shap_dict: dict, feature_map: dict, k: int = 5):
    """Convert SHAP dict (feat->value) to top-k plain-English statements.

    `feature_map` provides templates for features, e.g. {'F3836': 'High transaction velocity in last 7 days'}
    Returns list of strings.
    """
    items = sorted(shap_dict.items(), key=lambda x: -abs(x[1]))
    out = []
    for fname, val in items[:k]:
        desc = feature_map.get(fname, fname)
        sign = 'increased' if val > 0 else 'decreased'
        out.append(f"{desc}: {sign} impact ({val:.3f})")
    return out


def explain_row_with_artifact(artifact: dict, X_row: pd.DataFrame):
    """Use a saved SHAP artifact (explainer) to compute SHAP for a row.

    artifact is expected to be a dict with keys: 'explainer', optionally 'X_sample'.
    Returns a dict of feature->shap_value.
    """
    explainer = artifact.get('explainer')
    if explainer is None:
        raise ValueError('artifact missing explainer')
    shap_values = explainer.shap_values(X_row)
    if isinstance(shap_values, list):
        vals = shap_values[1][0]
    else:
        vals = shap_values[0]
    return dict(zip(X_row.columns.tolist(), vals.tolist()))
