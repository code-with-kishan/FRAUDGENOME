"""Prototype training script for MuleGuard AI.

This script loads `DataSet.csv`, trains LightGBM and XGBoost, computes a simple DTW
FraudDNA score for anchor features, and saves models to `models/`.
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
import lightgbm as lgb
import xgboost as xgb
import joblib
from ml.dtw_utils import multivariate_dtw


def load_data(path):
    df = pd.read_csv(path)
    return df


def compute_dtw_scores(df, anchors=['F321','F3836','F2082']):
    # For prototype: treat anchors as scalar features and compute simple distance
    patterns = df[df['F3924']==1][anchors].values
    # take mean pattern as canonical (very rough)
    if len(patterns)==0:
        return np.zeros(len(df))
    canon = np.nanmean(patterns, axis=0)
    scores = []
    for _, row in df.iterrows():
        vec = row[anchors].values.astype(float)
        dist = np.linalg.norm(vec - canon)
        score = 1.0 / (1.0 + dist)
        scores.append(score)
    return np.array(scores)


def train_models(X_train, y_train, X_val, y_val, outdir):
    os.makedirs(outdir, exist_ok=True)
    # LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    params = {'objective':'binary','metric':'auc','verbosity':-1}
    gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_val], early_stopping_rounds=10, verbose_eval=False)
    joblib.dump(gbm, os.path.join(outdir,'lgb_model.joblib'))

    # XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    xparams = {'objective':'binary:logistic','eval_metric':'auc'}
    bst = xgb.train(xparams, dtrain, num_boost_round=100, evals=[(dval,'val')], early_stopping_rounds=10, verbose_eval=False)
    bst.save_model(os.path.join(outdir,'xgb_model.json'))

    return gbm, bst


def main(args):
    df = load_data(args.data)
    # Basic feature selection: use the anchor features + some common features if present
    features = ['F115','F321','F527','F531','F670','F1692','F2082','F2122','F2582','F2678','F2737','F2956','F3043','F3836','F3887','F3889','F3891','F3894']
    available = [f for f in features if f in df.columns]
    X = df[available].fillna(0).astype(float)
    y = df['F3924']
    # add simple DTW-based fraudDNA score as a feature
    dtw_scores = compute_dtw_scores(df, anchors=['F321','F3836','F2082'])
    X['frauddna_score'] = dtw_scores

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    gbm, bst = train_models(X_train, y_train, X_val, y_val, args.out)

    # Evaluate ensemble simple average
    pred_lgb = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    pred_xgb = bst.predict(xgb.DMatrix(X_val))
    ensemble = 0.5 * pred_lgb + 0.5 * pred_xgb
    precision, recall, _ = precision_recall_curve(y_val, ensemble)
    pr_auc = auc(recall, precision)
    print(f'Validation PR AUC: {pr_auc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='models')
    args = parser.parse_args()
    main(args)
