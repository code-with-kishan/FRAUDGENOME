import json
import pytest
from fastapi.testclient import TestClient
import api.app as appmod

def test_investigate_account_endpoint(monkeypatch):
    # Mock models and matchers
    monkeypatch.setattr(appmod, "match_frauddna", lambda timeseries: [("pattern-1", 0.25)])
    monkeypatch.setattr(appmod, "compute_ensemble_prob", lambda features: 0.85)
    monkeypatch.setattr(appmod, "get_ring_summary_for_account", lambda account_id, out_dir=None: {
        "community_id": 7,
        "members": ["acct-1", "acct-2"],
        "stage": "Active",
        "stage_score": 0.9
    })
    monkeypatch.setattr(appmod, "load_notes_store", lambda: {"acct-1": [{"note": "Previous note", "analyst": "John", "status": "Watchlist", "created_at": 1700000000}]})

    client = TestClient(appmod.app)
    req_body = {
        "account_id": "acct-1",
        "features": {"F321": 1.2, "F3836": 2.1, "F2082": 0.9},
        "timeseries": [[1.2, 2.1, 0.9]]
    }
    r = client.post("/accounts/investigate", json=req_body)
    assert r.status_code == 200
    data = r.json()
    assert data["account_id"] == "acct-1"
    assert "risk_score" in data
    assert data["risk_level"] == "Critical"
    assert len(data["signature_matches"]) == 1
    assert data["signature_matches"][0]["pattern_id"] == "pattern-1"
    assert len(data["notes"]) == 1
    assert data["notes"][0]["note"] == "Previous note"

def test_notes_endpoints(tmp_path, monkeypatch):
    # Set custom notes store path in a temp directory to avoid modifying active workspace notes
    notes_file = tmp_path / "investigator_notes.json"
    monkeypatch.setattr(appmod, "get_notes_store_path", lambda: str(notes_file))
    
    client = TestClient(appmod.app)
    
    # 1. Get notes when store is empty
    r_get = client.get("/accounts/acct-101/notes")
    assert r_get.status_code == 200
    assert r_get.json()["notes"] == []

    # 2. Add a note
    note_req = {
        "note": "Suspicious pattern detected",
        "analyst": "Sarah",
        "status": "Escalated"
    }
    r_post = client.post("/accounts/acct-101/notes", json=note_req)
    assert r_post.status_code == 200
    post_res = r_post.json()
    assert post_res["saved"] is True
    assert len(post_res["notes"]) == 1
    assert post_res["notes"][0]["note"] == "Suspicious pattern detected"
    assert post_res["notes"][0]["analyst"] == "Sarah"
    assert post_res["notes"][0]["status"] == "Escalated"
    assert "created_at" in post_res["notes"][0]

    # 3. Add an empty note (should return validation error 400 or similar)
    r_post_empty = client.post("/accounts/acct-101/notes", json={"note": ""})
    assert r_post_empty.status_code == 400

    # 4. Get notes again to verify persistence
    r_get_again = client.get("/accounts/acct-101/notes")
    assert r_get_again.status_code == 200
    assert len(r_get_again.json()["notes"]) == 1
