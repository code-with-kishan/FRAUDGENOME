from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
import os
import math
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
import json
import time
from ml.dtw_utils import multivariate_dtw
from ml.frauddna_matcher import match_timeseries_prefilter, match_timeseries_prefilter_with_manifest
from ml.explain import compute_shap_for_row, topk_plain_english
from ml.ring_mapper import build_graph, load_graph
from ml import briefs
from ml import drift as drift_module
from ml import mlflow_utils
from ml import shieldscan
import threading
import time as _time
import logging

from .config import MODEL_DIR, configure_logging

configure_logging()
logger = logging.getLogger('muleguard.api')

app = FastAPI(title='MuleGuard AI - API')

# Serve static assets (web UI + generated briefs)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
static_dir = os.path.join(BASE_DIR, 'web')
models_static = os.path.join(BASE_DIR, 'models')
if not os.path.exists(models_static):
    os.makedirs(models_static, exist_ok=True)
app.mount('/static', StaticFiles(directory=models_static), name='static')
app.mount('/', StaticFiles(directory=static_dir, html=True), name='web')


AUDIT_LOG = os.environ.get('MULEGUARD_AUDIT_LOG', os.path.join(os.path.dirname(__file__), '..', 'models', 'audit.log'))
API_KEYS = [k.strip() for k in os.environ.get('MULEGUARD_API_KEYS', '').split(',') if k.strip()]


@app.middleware('http')
async def audit_and_auth_middleware(request, call_next):
    # Authentication: if API_KEYS set, require X-API-Key header
    key_required = len(API_KEYS) > 0
    api_key = request.headers.get('x-api-key')
    if key_required and (not api_key or api_key not in API_KEYS):
        from starlette.responses import JSONResponse
        resp = JSONResponse({'detail': 'Unauthorized'}, status_code=401)
        # log attempt
        _log_audit(request, resp.status_code, 'unauthorized')
        return resp

    response = await call_next(request)
    # log successful request
    _log_audit(request, response.status_code, 'ok')
    return response


def _log_audit(request, status_code: int, outcome: str):
    try:
        entry = {
            'ts': _time.time(),
            'method': request.method,
            'path': request.url.path,
            'query': str(request.url.query),
            'client': request.client.host if request.client is not None else None,
            'status': int(status_code),
            'outcome': outcome,
        }
        # Write JSON line
        with open(AUDIT_LOG, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception:
        pass


class CTIRequest(BaseModel):
    account_id: Optional[str]
    features: Dict[str, float]
    # Optional multivariate time-series for anchors: list of time steps, each is list of anchor values
    # Example: [[f321_t1, f3836_t1, f2082_t1], [f321_t2, f3836_t2, f2082_t2], ...]
    timeseries: Optional[List[List[float]]]


class CTIResponse(BaseModel):
    account_id: Optional[str]
    cti: float
    components: Dict[str, float]
    level: str
    explain: Optional[Dict[str, Any]]


class ModelStore:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.lgb = None
        self.xgb = None
        self.canon = None
        self.shap_sample = None
        self.load()

    def load(self):
        # Load LightGBM (joblib) if present
        lgb_path = os.path.join(self.models_dir, 'lgb_model.joblib')
        xgb_path = os.path.join(self.models_dir, 'xgb_model.json')
        canon_path = os.path.join(self.models_dir, 'canon.npy')
        shap_path = os.path.join(self.models_dir, 'shap_sample.joblib')
        frauddna_manifest_path = os.path.join(self.models_dir, 'frauddna_manifest.parquet')
        logger.info('Loading models from %s', self.models_dir)
        if os.path.exists(lgb_path):
            try:
                self.lgb = joblib.load(lgb_path)
                logger.info('Loaded LightGBM model')
            except Exception as e:
                logger.exception('Failed to load LightGBM: %s', e)
                self.lgb = None
        else:
            logger.warning('LightGBM model not found at %s', lgb_path)
        if os.path.exists(xgb_path):
            try:
                self.xgb = xgb.Booster()
                self.xgb.load_model(xgb_path)
                logger.info('Loaded XGBoost model')
            except Exception as e:
                logger.exception('Failed to load XGBoost: %s', e)
                self.xgb = None
        else:
            logger.warning('XGBoost model not found at %s', xgb_path)
        if os.path.exists(canon_path):
            try:
                self.canon = np.load(canon_path)
                logger.info('Loaded FraudDNA canon')
            except Exception as e:
                logger.exception('Failed to load canon: %s', e)
                self.canon = None
        else:
            logger.info('No canon found at %s', canon_path)
        if os.path.exists(shap_path):
            try:
                self.shap_sample = joblib.load(shap_path)
                logger.info('Loaded SHAP sample')
            except Exception as e:
                logger.exception('Failed to load shap sample: %s', e)
                self.shap_sample = None
        else:
            logger.info('No shap sample found at %s', shap_path)
        # Load FraudDNA manifest and patterns if present
        if os.path.exists(frauddna_manifest_path):
            try:
                self.frauddna_manifest = pd.read_parquet(frauddna_manifest_path)
                # load pattern arrays into memory
                self.frauddna_patterns = []
                for _, r in self.frauddna_manifest.iterrows():
                    ppath = os.path.join(self.models_dir, r['file_path'])
                    if os.path.exists(ppath):
                        try:
                            self.frauddna_patterns.append((r['pattern_id'], np.load(ppath)))
                        except Exception:
                            continue
                logger.info('Loaded %d FraudDNA patterns', len(self.frauddna_patterns))
            except Exception as e:
                logger.exception('Failed to load frauddna manifest: %s', e)
                self.frauddna_manifest = None
                self.frauddna_patterns = []
        else:
            self.frauddna_manifest = None
            self.frauddna_patterns = []


STORE = ModelStore(models_dir=os.environ.get('MULEGUARD_MODEL_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))))


def compute_frauddna_score(features: Dict[str, float]):
    """Compute prototype FraudDNA score using canonical vector if available.
    Expects `F321`, `F3836`, `F2082` if canon is available.
    """
    # legacy single-vector canon-based score
    if STORE.canon is None:
        return 0.0
    anchors = ['F321','F3836','F2082']
    try:
        vec = np.array([features[a] for a in anchors], dtype=float)
    except Exception:
        return 0.0
    dist = np.linalg.norm(vec - STORE.canon)
    score = 1.0 / (1.0 + float(dist))
    return float(score)


def calibrate_cti_thresholds() -> Dict[str, float]:
    """Return calibrated CTI thresholds on the 0-100 scale."""
    defaults = {
        'medium': 30.0,
        'high': 55.0,
        'critical': 80.0,
    }
    resolved = {
        'medium': float(os.environ.get('MULEGUARD_CTI_MEDIUM_THRESHOLD', defaults['medium'])),
        'high': float(os.environ.get('MULEGUARD_CTI_HIGH_THRESHOLD', defaults['high'])),
        'critical': float(os.environ.get('MULEGUARD_CTI_CRITICAL_THRESHOLD', defaults['critical'])),
    }
    resolved['medium'] = max(0.0, min(resolved['medium'], 100.0))
    resolved['high'] = max(resolved['medium'] + 1.0, min(resolved['high'], 100.0))
    resolved['critical'] = max(resolved['high'] + 1.0, min(resolved['critical'], 100.0))
    return resolved


CTI_THRESHOLDS = calibrate_cti_thresholds()


def compose_cti_score(dtw_score: Optional[float], ml_probability: Optional[float]) -> Dict[str, float]:
    """Compose a 0-100 CTI from DTW and model probability signals."""
    weights = {
        'dtw_weight': 0.6,
        'ml_weight': 0.4,
    }
    signals = []
    if dtw_score is not None and math.isfinite(float(dtw_score)):
        signals.append(('dtw_score', max(0.0, min(1.0, float(dtw_score))), weights['dtw_weight']))
    if ml_probability is not None and math.isfinite(float(ml_probability)):
        signals.append(('ml_probability', max(0.0, min(1.0, float(ml_probability))), weights['ml_weight']))

    if not signals:
        return {
            'cti': 0.0,
            'dtw_score': 0.0,
            'ml_probability': 0.0,
            'dtw_weight': 0.0,
            'ml_weight': 0.0,
        }

    total_weight = sum(weight for _, _, weight in signals)
    weighted_score = sum(score * weight for _, score, weight in signals)
    resolved = {
        'cti': max(0.0, min(100.0, 100.0 * weighted_score / total_weight)),
        'dtw_score': 0.0,
        'ml_probability': 0.0,
        'dtw_weight': 0.0,
        'ml_weight': 0.0,
    }
    for name, score, weight in signals:
        resolved[name] = score
        resolved[f'{name.split("_")[0]}_weight'] = weight / total_weight
    return resolved


def match_frauddna(timeseries: Optional[List[List[float]]], top_k: int = 3):
    """Match provided multivariate timeseries to loaded FraudDNA patterns using DTW.

    Returns list of (pattern_id, distance) sorted by distance ascending.
    """
    if timeseries is None:
        return []
    try:
        arr = np.array(timeseries, dtype=float)
        if arr.ndim != 2:
            return []
    except Exception:
        return []

    # use index if available for fast prefilter
    index_path = os.path.join(MODEL_DIR, 'frauddna_index.npz')
    patterns_base = MODEL_DIR
    if os.path.exists(index_path):
        try:
            manifest_path = os.path.join(MODEL_DIR, 'frauddna_manifest.parquet')
            return match_timeseries_prefilter_with_manifest(arr, index_path, patterns_base, manifest_path, top_k=top_k, prefilter_k=50)
        except Exception:
            pass

    # fallback to brute-force
    if not hasattr(STORE, 'frauddna_patterns') or not STORE.frauddna_patterns:
        return []
    results = []
    for pid, patt in STORE.frauddna_patterns:
        try:
            d = float(multivariate_dtw(arr, patt))
        except Exception:
            d = float(np.linalg.norm(arr.ravel() - patt.ravel()))
        results.append((pid, d))
    results = sorted(results, key=lambda x: x[1])[:top_k]
    return results


def compute_ensemble_prob(features: Dict[str, float]):
    """Compute ensemble probability from loaded models. Returns probability 0..1 or None if models missing."""
    if STORE.lgb is None or STORE.xgb is None:
        return None
    # convert to array in model feature order: attempt to use lgb.feature_name_ if available
    try:
        feature_names = STORE.lgb.feature_name()
    except Exception:
        # fallback: use keys order
        feature_names = list(features.keys())
    X = np.array([features.get(fn, 0.0) for fn in feature_names], dtype=float).reshape(1, -1)
    try:
        p1 = STORE.lgb.predict(X, num_iteration=STORE.lgb.best_iteration if hasattr(STORE.lgb, 'best_iteration') else None)
        p2 = STORE.xgb.predict(xgb.DMatrix(X))
        prob = 0.5 * float(p1[0]) + 0.5 * float(p2[0])
        return prob
    except Exception:
        return None


def map_level(cti: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    thresholds = thresholds or CTI_THRESHOLDS
    if cti >= thresholds['critical']:
        return 'Critical'
    if cti >= thresholds['high']:
        return 'High'
    if cti >= thresholds['medium']:
        return 'Medium'
    return 'Low'


@app.get('/health')
def health():
    loaded = STORE.lgb is not None and STORE.xgb is not None
    return {'status': 'ok', 'models_loaded': loaded}


@app.post('/models/reload')
def reload_models():
    STORE.load()
    loaded = STORE.lgb is not None or STORE.xgb is not None
    return {'loaded': loaded}


@app.get('/models/version')
def models_version():
    return {
        'lgb_loaded': STORE.lgb is not None,
        'xgb_loaded': STORE.xgb is not None,
        'canon_loaded': STORE.canon is not None,
    }


@app.post('/explain/shap')
def explain_shap(req: CTIRequest):
    # require models loaded
    if STORE.lgb is None:
        raise HTTPException(status_code=503, detail='LightGBM model not loaded')
    # build a single-row dataframe in model's feature order
    try:
        feature_names = STORE.lgb.feature_name()
    except Exception:
        feature_names = list(req.features.keys())
    import pandas as pd
    row = pd.DataFrame([[req.features.get(fn, 0.0) for fn in feature_names]], columns=feature_names)
    # Use saved SHAP artifact if available for consistent explainers and faster response
    try:
        if STORE.shap_sample is not None:
            shap_map = None
            try:
                from ml.explain import explain_row_with_artifact
                shap_map = explain_row_with_artifact(STORE.shap_sample, row)
            except Exception:
                shap_map = compute_shap_for_row(STORE.lgb, row)
        else:
            shap_map = compute_shap_for_row(STORE.lgb, row)
    except Exception:
        shap_map = compute_shap_for_row(STORE.lgb, row)

    # plain-English mapping (use selected_features.json if available)
    feature_map = {fn: fn for fn in feature_names}
    sel_path = os.path.join(MODEL_DIR, 'selected_features.json')
    if os.path.exists(sel_path):
        try:
            with open(sel_path, 'r') as f:
                sel = json.load(f)
            # map selected features to human-friendly names if provided (identity mapping by default)
            for fn in sel:
                feature_map.setdefault(fn, fn)
        except Exception:
            pass
    plain = topk_plain_english(shap_map, feature_map, k=5)
    return {'shap': shap_map, 'plain_english': plain}


@app.post('/rings/build')
def api_build_rings(payload: Dict[str, Any]):
    """Build ring graph from normalized parquet; payload must include `normalized_path` and `out_dir`."""
    normalized = payload.get('normalized_path')
    out_dir = payload.get('out_dir', 'models/graph')
    if not normalized or not os.path.exists(normalized):
        raise HTTPException(status_code=400, detail='normalized_path missing or not found')
    nodes_path, edges_path = build_graph(normalized, out_dir)
    return {'nodes': nodes_path, 'edges': edges_path}


@app.get('/rings/account/{account_id}')
def api_get_account_ring(account_id: str, out_dir: str = 'models/graph'):
    nodes, edges, communities = load_graph(out_dir)
    node = nodes[nodes['account_id'] == account_id]
    if node.empty:
        raise HTTPException(status_code=404, detail='account not in graph')
    community_id = int(node.iloc[0]['community'])
    community = communities[communities['community_id'] == community_id]
    members = nodes[nodes['community'] == community_id]['account_id'].tolist()
    community_summary = community.to_dict(orient='records')
    stage = community_summary[0].get('stage') if community_summary else None
    stage_score = community_summary[0].get('stage_score') if community_summary else None
    return {
        'community_id': int(community_id),
        'members': members,
        'stage': stage,
        'stage_score': stage_score,
        'community_summary': community_summary,
    }


@app.post('/accounts/compute_cti', response_model=CTIResponse)
def compute_cti(req: CTIRequest):
    # Compute DTW-based matches if timeseries provided
    frauddna_matches = match_frauddna(req.timeseries)
    ensemble_prob = compute_ensemble_prob(req.features)
    if ensemble_prob is None:
        # Models not available — return informative error
        raise HTTPException(status_code=503, detail='Models not loaded. Run training and place models in models/ then call /models/reload')

    dtw_best_match = None
    dtw_score = None
    if frauddna_matches:
        def _distance_of(match: Any) -> float:
            if isinstance(match, dict):
                return float(match.get('distance', float('inf')))
            return float(match[1])

        dtw_best_match = min(frauddna_matches, key=_distance_of)
        best_distance = _distance_of(dtw_best_match)
        if math.isfinite(best_distance):
            dtw_score = 1.0 / (1.0 + best_distance)

    cti_parts = compose_cti_score(dtw_score, float(ensemble_prob))
    CTI = cti_parts['cti']

    explain = {
        'dtw_score': round(cti_parts['dtw_score'], 6),
        'ml_probability': round(cti_parts['ml_probability'], 6),
        'dtw_weight': round(cti_parts['dtw_weight'], 6),
        'ml_weight': round(cti_parts['ml_weight'], 6),
        'medium_threshold': float(CTI_THRESHOLDS['medium']),
        'high_threshold': float(CTI_THRESHOLDS['high']),
        'critical_threshold': float(CTI_THRESHOLDS['critical']),
    }
    # attach DTW match details if available
    if frauddna_matches:
        # frauddna_matches may be list of dicts (from index) or list of tuples (pid,d)
        formatted = []
        for m in frauddna_matches:
            if isinstance(m, dict):
                pid = m.get('pattern_id')
                d = float(m.get('distance', 0.0))
                score = 1.0/(1.0+d) if d is not None else None
                formatted.append({
                    'pattern_id': pid,
                    'distance': d,
                    'score': score,
                    'file_path': m.get('file_path'),
                    'cluster_id': m.get('cluster_id'),
                    'support_count': m.get('support_count')
                })
            else:
                pid, d = m
                d = float(d)
                formatted.append({'pattern_id': pid, 'distance': d, 'score': 1.0/(1.0+d)})
        explain['frauddna_matches'] = formatted
        if dtw_best_match is not None:
            if isinstance(dtw_best_match, dict):
                best_distance = float(dtw_best_match.get('distance', 0.0))
                explain['dtw_best_match'] = {
                    'pattern_id': dtw_best_match.get('pattern_id'),
                    'distance': best_distance,
                    'score': round(1.0 / (1.0 + best_distance), 6),
                    'file_path': dtw_best_match.get('file_path'),
                    'cluster_id': dtw_best_match.get('cluster_id'),
                    'support_count': dtw_best_match.get('support_count'),
                }
            else:
                pid, d = dtw_best_match
                d = float(d)
                explain['dtw_best_match'] = {'pattern_id': pid, 'distance': d, 'score': round(1.0 / (1.0 + d), 6)}
    # attach shap sample if available
    if STORE.shap_sample is not None:
        explain['shap_sample_available'] = True
    else:
        explain['shap_sample_available'] = False

    components = {
        'cti': round(CTI, 6),
        'dtw_score': round(cti_parts['dtw_score'], 6),
        'ml_probability': round(cti_parts['ml_probability'], 6),
        'dtw_weight': round(cti_parts['dtw_weight'], 6),
        'ml_weight': round(cti_parts['ml_weight'], 6),
        'medium_threshold': float(CTI_THRESHOLDS['medium']),
        'high_threshold': float(CTI_THRESHOLDS['high']),
        'critical_threshold': float(CTI_THRESHOLDS['critical']),
    }

    return CTIResponse(account_id=req.account_id, cti=round(CTI,2), components=components, level=map_level(CTI), explain=explain)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)


class BriefRequest(BaseModel):
    account_id: str
    cti_score: Optional[float] = None
    include_shap: bool = True
    include_dtw: bool = True
    include_ring: bool = True
    notes: Optional[str] = None


@app.post('/briefs/generate')
async def generate_brief(req: BriefRequest):
    # Build payload from available helpers; fall back gracefully if a piece is missing
    shap_map = {}
    try:
        if req.include_shap:
            # explain.compute_shap_for_row expects a single-row DataFrame normally; here call wrapper if exists
            try:
                shap_map = compute_shap_for_row(STORE.lgb, req.account_id)  # best-effort
            except Exception:
                shap_map = {}
    except Exception:
        shap_map = {}

    dtw_matches = []
    try:
        if req.include_dtw:
            try:
                # try the matcher entrypoints
                dtw_matches = match_timeseries_prefilter(req.account_id, top_k=10)
            except Exception:
                # fallback to API helper match_frauddna; expects timeseries in request in real flows
                dtw_matches = []
    except Exception:
        dtw_matches = []

    ring_summary = {}
    try:
        if req.include_ring:
            try:
                nodes, edges, communities = load_graph(os.path.join(MODEL_DIR, 'graph'))
                node = nodes[nodes['account_id'] == req.account_id]
                if not node.empty:
                    cid = int(node.iloc[0]['community'])
                    members = nodes[nodes['community'] == cid]['account_id'].tolist()
                    community_rows = communities[communities['community_id'] == cid].to_dict(orient='records')
                    community_row = community_rows[0] if community_rows else {}
                    ring_summary = {
                        'community_id': cid,
                        'members': members,
                        'stage': community_row.get('stage'),
                        'stage_score': community_row.get('stage_score'),
                        'label_rate': community_row.get('label_rate'),
                        'recent_activity_rate': community_row.get('recent_activity_rate'),
                    }
            except Exception:
                ring_summary = {}
    except Exception:
        ring_summary = {}

    payload = briefs.build_brief_payload(
        account_id=req.account_id,
        cti_score=float(req.cti_score or 0.0),
        shap_explanations=shap_map,
        frauddna_matches=dtw_matches,
        ring_summary=ring_summary,
        notes=req.notes,
    )

    out_dir = os.path.join('models', 'briefs')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"brief_{req.account_id}_{int(time.time())}.pdf")
    briefs.generate_pdf_from_payload(payload, out_file)
    return FileResponse(out_file, media_type='application/pdf', filename=os.path.basename(out_file))



@app.post('/models/drift_check')
def api_drift_check(payload: Dict[str, Any]):
    """Run ADWIN-based drift detection on validation set and optionally trigger retrain.

    Payload keys:
    - model_path: path to current model (joblib)
    - normalized: path to normalized.parquet
    - labels: path to labels.parquet
    - retrain_cmd: optional list command to run retraining (example: ['python','-m','ml.train_pipeline'])
    - shadow_eval_cmd: optional list command to run shadow evaluation after retrain
    """
    model_path = payload.get('model_path', os.path.join(MODEL_DIR, 'lgb_model.joblib'))
    normalized = payload.get('normalized')
    labels = payload.get('labels')
    retrain_cmd = payload.get('retrain_cmd')
    shadow_eval_cmd = payload.get('shadow_eval_cmd')

    if not normalized or not labels:
        raise HTTPException(status_code=400, detail='normalized and labels paths required')

    try:
        res = drift_module.detect_drift_on_validation(model_path=model_path, normalized_parquet=normalized, labels_parquet=labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'drift detection failed: {e}')

    out = {'drift': res.get('drift'), 'samples_seen': res.get('samples_seen'), 'last_value': res.get('last_value')}

    if res.get('drift') and retrain_cmd:
        # run retraining and optional shadow eval
        try:
            train_res = drift_module.trigger_retrain_and_shadow_eval(retrain_cmd, shadow_eval_cmd)
            out['retrain'] = train_res
        except Exception as e:
            out['retrain_error'] = str(e)

    return out



@app.post('/shieldscan/analyze')
def api_shieldscan_analyze(payload: Dict[str, Any]):
    """Analyze APK and correlate with accounts. Payload keys:
    - apk_path: path to .apk file
    - frauddna_manifest: optional path to frauddna_manifest.parquet
    - accounts_events: optional path to accounts events parquet
    - dynamic_trace: optional path to dynamic trace JSON
    """
    apk = payload.get('apk_path')
    if not apk or not os.path.exists(apk):
        raise HTTPException(status_code=400, detail='apk_path missing or not found')
    frauddna_manifest = payload.get('frauddna_manifest', os.path.join(MODEL_DIR, 'frauddna_manifest.parquet'))
    accounts_events = payload.get('accounts_events')
    dynamic_trace = payload.get('dynamic_trace')
    try:
        report_path = shieldscan.generate_apk_correlation_report(apk, out_dir=os.path.join('models', 'shieldscan'), frauddna_manifest_path=frauddna_manifest, accounts_events_path=accounts_events, dynamic_trace=dynamic_trace)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'shieldscan failed: {e}')
    # return the JSON content for convenience
    with open(report_path, 'r') as f:
        data = json.load(f)
    return data



@app.websocket('/ws/cti')
async def websocket_cti(ws: WebSocket):
    await ws.accept()
    subs = set()
    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                data = {'action': 'ping'}
            action = data.get('action')
            if action == 'subscribe':
                acct = data.get('account_id')
                if acct:
                    subs.add(acct)
                    # send an immediate simulated CTI
                    await ws.send_text(json.dumps({'type':'cti_update','account_id':acct,'cti': round(20+80*_time.time()%1,2)}))
            elif action == 'generate_brief':
                acct = data.get('account_id')
                # try to create brief and provide static URL
                try:
                    payload = briefs.build_brief_payload(account_id=acct, cti_score=0.0, shap_explanations={}, frauddna_matches=[], ring_summary={})
                    out_dir = os.path.join('models','briefs')
                    os.makedirs(out_dir, exist_ok=True)
                    out_file = os.path.join(out_dir, f'brief_{acct}_{int(_time.time())}.pdf')
                    briefs.generate_pdf_from_payload(payload, out_file)
                    url = '/static/briefs/' + os.path.basename(out_file)
                    await ws.send_text(json.dumps({'type':'brief_ready','url':url,'filename':os.path.basename(out_file)}))
                except Exception as e:
                    await ws.send_text(json.dumps({'type':'error','error':str(e)}))
            else:
                # unknown action
                await ws.send_text(json.dumps({'type':'pong'}))
    except WebSocketDisconnect:
        return
