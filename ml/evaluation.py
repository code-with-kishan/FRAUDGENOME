"""Model evaluation metrics for FRAUDGENOME.

Implements full evaluation suite:
- Accuracy, Precision, Recall, F1
- ROC AUC, PR AUC
- Confusion matrix and classification report
- Threshold optimization (F1, Youden's J, cost-based)
- Calibration curve
- Precision@K / Recall@K
- Cost analysis (false positive / false negative costs)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Cost constants from FRAUDGENOME proposal
FP_COST_INR = 2_125          # ₹2,125 per false positive (investigation cost)
FN_COST_INR = 1_47_700       # ₹1,47,700 per false negative (downstream victim losses)
INVESTIGATOR_HOURS_PER_MANUAL = 0.75  # hours per manual investigation
INVESTIGATOR_HOURS_FRAUDGENOME = 0.2  # hours per FRAUDGENOME-assisted investigation


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b != 0 else default


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, int]:
    """Compute TP, TN, FP, FN."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute full classification metrics suite."""
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_recall_curve,
        roc_curve,
        auc,
    )

    cm = compute_confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]

    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)   # sensitivity
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    f2 = _safe_div(5 * precision * recall, 4 * precision + recall)  # F2 weights recall more

    metrics: Dict[str, Any] = {
        "threshold": threshold,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "f2_score": round(f2, 4),
        "confusion_matrix": cm,
        "support_positives": int(y_true.sum()),
        "support_negatives": int((y_true == 0).sum()),
        "total_samples": len(y_true),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
        except Exception:
            metrics["roc_auc"] = None

        try:
            pr_p, pr_r, _ = precision_recall_curve(y_true, y_proba)
            metrics["pr_auc"] = round(float(auc(pr_r, pr_p)), 4)
        except Exception:
            metrics["pr_auc"] = None

    return metrics


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = "f1",
    fp_cost: float = FP_COST_INR,
    fn_cost: float = FN_COST_INR,
) -> Dict[str, Any]:
    """Find optimal classification threshold.

    Strategies:
    - 'f1': maximize F1 score
    - 'youden': maximize Youden's J (sensitivity + specificity - 1)
    - 'cost': minimize total cost (fp_cost * FP + fn_cost * FN)
    - 'precision_recall': balance precision and recall
    """
    from sklearn.metrics import roc_curve, precision_recall_curve

    thresholds_to_try = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_score = -np.inf

    if strategy == "youden":
        fpr, tpr, roc_thresh = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = float(roc_thresh[best_idx])
        best_score = float(j_scores[best_idx])

    elif strategy == "cost":
        best_cost = np.inf
        for thresh in thresholds_to_try:
            y_pred = (y_proba >= thresh).astype(int)
            cm = compute_confusion_matrix(y_true, y_pred)
            total_cost = fp_cost * cm["FP"] + fn_cost * cm["FN"]
            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = float(thresh)
        best_score = -best_cost

    else:  # default: f1
        for thresh in thresholds_to_try:
            y_pred = (y_proba >= thresh).astype(int)
            cm = compute_confusion_matrix(y_true, y_pred)
            tp, fp_n, fn_n = cm["TP"], cm["FP"], cm["FN"]
            precision = _safe_div(tp, tp + fp_n)
            recall = _safe_div(tp, tp + fn_n)
            f1 = _safe_div(2 * precision * recall, precision + recall)
            if f1 > best_score:
                best_score = f1
                best_threshold = float(thresh)

    # Compute metrics at optimal threshold
    y_pred_opt = (y_proba >= best_threshold).astype(int)
    metrics_opt = compute_classification_metrics(y_true, y_pred_opt, y_proba, best_threshold)

    return {
        "strategy": strategy,
        "optimal_threshold": round(best_threshold, 4),
        "best_score": round(best_score, 4),
        "metrics_at_optimal": metrics_opt,
    }


def compute_precision_at_k(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute Precision@K and Recall@K for various K values."""
    if k_values is None:
        k_values = [10, 20, 50, 100, 200]

    # Sort by predicted probability (descending)
    sorted_idx = np.argsort(-y_proba)
    y_sorted = y_true[sorted_idx]
    total_positives = int(y_true.sum())

    results = {}
    for k in k_values:
        k_actual = min(k, len(y_sorted))
        top_k = y_sorted[:k_actual]
        tp_at_k = int(top_k.sum())
        results[f"precision@{k}"] = round(_safe_div(tp_at_k, k_actual), 4)
        results[f"recall@{k}"] = round(_safe_div(tp_at_k, total_positives), 4)

    return results


def compute_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute probability calibration curve data."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_midpoints = []
    fraction_positives = []
    mean_predicted = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_proba >= low) & (y_proba < high if i < n_bins - 1 else y_proba <= high)
        count = int(mask.sum())
        bin_counts.append(count)
        if count > 0:
            frac_pos = float(y_true[mask].mean())
            mean_pred = float(y_proba[mask].mean())
        else:
            frac_pos = 0.0
            mean_pred = (low + high) / 2
        bin_midpoints.append(round((low + high) / 2, 3))
        fraction_positives.append(round(frac_pos, 4))
        mean_predicted.append(round(mean_pred, 4))

    # Brier score
    brier = float(np.mean((y_proba - y_true) ** 2))

    return {
        "bin_midpoints": bin_midpoints,
        "fraction_positives": fraction_positives,
        "mean_predicted_probability": mean_predicted,
        "bin_counts": bin_counts,
        "brier_score": round(brier, 4),
        "n_bins": n_bins,
    }


def cost_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fp_cost: float = FP_COST_INR,
    fn_cost: float = FN_COST_INR,
    investigator_hourly_rate_inr: float = 1_500,
) -> Dict[str, Any]:
    """Compute full cost analysis in INR terms."""
    cm = compute_confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    n_accounts = len(y_true)

    # Direct costs
    fp_total = fp * fp_cost
    fn_total = fn * fn_cost
    total_cost = fp_total + fn_total

    # Investigator time savings
    manual_hours = n_accounts * INVESTIGATOR_HOURS_PER_MANUAL
    fraudgenome_hours = n_accounts * INVESTIGATOR_HOURS_FRAUDGENOME
    hours_saved = manual_hours - fraudgenome_hours
    hours_saved_inr = hours_saved * investigator_hourly_rate_inr

    # Per-100-account review
    if n_accounts > 0:
        per_100_fp = round(fp / n_accounts * 100, 1)
        per_100_fn = round(fn / n_accounts * 100, 1)
        per_100_tp = round(tp / n_accounts * 100, 1)
        per_100_cost = round(total_cost / n_accounts * 100)
    else:
        per_100_fp = per_100_fn = per_100_tp = per_100_cost = 0

    return {
        "n_accounts_reviewed": n_accounts,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "fp_cost_inr": fp_total,
        "fn_cost_inr": fn_total,
        "total_cost_inr": total_cost,
        "per_100_accounts": {
            "true_positives": per_100_tp,
            "false_positives": per_100_fp,
            "false_negatives": per_100_fn,
            "total_cost_inr": per_100_cost,
        },
        "investigator_hours_manual": round(manual_hours, 1),
        "investigator_hours_fraudgenome": round(fraudgenome_hours, 1),
        "investigator_hours_saved": round(hours_saved, 1),
        "investigator_cost_saved_inr": round(hours_saved_inr),
        "total_value_inr": round(hours_saved_inr - total_cost),
    }


def full_evaluation_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the complete evaluation suite and return a single report dict."""
    y_pred = (y_proba >= threshold).astype(int)

    report: Dict[str, Any] = {
        "classification_metrics": compute_classification_metrics(y_true, y_pred, y_proba, threshold),
        "threshold_optimization": {
            "f1": optimize_threshold(y_true, y_proba, "f1"),
            "youden": optimize_threshold(y_true, y_proba, "youden"),
            "cost_minimization": optimize_threshold(y_true, y_proba, "cost"),
        },
        "precision_recall_at_k": compute_precision_at_k(y_true, y_proba),
        "calibration_curve": compute_calibration_curve(y_true, y_proba),
        "cost_analysis": cost_analysis(y_true, y_pred),
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)

    return report
