import sys
import types

import pytest

fake_shap = types.ModuleType('shap')
fake_shap.TreeExplainer = object
sys.modules.setdefault('shap', fake_shap)

import api.app as appmod


def test_compose_cti_score_renormalizes_missing_dtw():
    parts = appmod.compose_cti_score(None, 0.8)

    assert parts['cti'] == pytest.approx(80.0)
    assert parts['dtw_score'] == pytest.approx(0.0)
    assert parts['ml_probability'] == pytest.approx(0.8)
    assert parts['dtw_weight'] == pytest.approx(0.0)
    assert parts['ml_weight'] == pytest.approx(1.0)
    assert appmod.map_level(parts['cti']) == 'Critical'


def test_map_level_accepts_custom_thresholds():
    thresholds = {'medium': 25.0, 'high': 50.0, 'critical': 75.0}

    assert appmod.map_level(20.0, thresholds) == 'Low'
    assert appmod.map_level(30.0, thresholds) == 'Medium'
    assert appmod.map_level(60.0, thresholds) == 'High'
    assert appmod.map_level(80.0, thresholds) == 'Critical'


def test_compute_cti_combines_dtw_and_probability(monkeypatch):
    monkeypatch.setattr(appmod, 'match_frauddna', lambda timeseries: [('pattern-1', 0.25), ('pattern-2', 1.0)])
    monkeypatch.setattr(appmod, 'compute_ensemble_prob', lambda features: 0.5)
    monkeypatch.setattr(appmod.STORE, 'shap_sample', None, raising=False)

    req = appmod.CTIRequest(account_id='acct-1', features={'f1': 1.0}, timeseries=[[1.0], [2.0]])
    resp = appmod.compute_cti(req)

    assert resp.cti == pytest.approx(68.0)
    assert resp.components['cti'] == pytest.approx(68.0)
    assert resp.components['dtw_score'] == pytest.approx(0.8)
    assert resp.components['ml_probability'] == pytest.approx(0.5)
    assert resp.level == 'High'
    assert resp.explain['dtw_best_match']['pattern_id'] == 'pattern-1'