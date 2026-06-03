import os
import json
from fastapi.testclient import TestClient

import api.app as appmod


def test_health_and_audit_log(tmp_path, monkeypatch):
    # set audit log to tmp file
    audit_file = tmp_path / 'audit.log'
    monkeypatch.setenv('MULEGUARD_AUDIT_LOG', str(audit_file))
    client = TestClient(appmod.app)
    r = client.get('/health')
    assert r.status_code == 200
    # audit log should exist and contain an entry
    assert audit_file.exists()
    lines = audit_file.read_text().strip().splitlines()
    assert len(lines) >= 1
    e = json.loads(lines[-1])
    assert e['path'] == '/health'


def test_api_key_enforced(tmp_path, monkeypatch):
    # require a key
    monkeypatch.setenv('MULEGUARD_API_KEYS', 'testkey123')
    client = TestClient(appmod.app)
    r = client.get('/health')
    # missing key -> 401
    assert r.status_code == 401
    # with key
    r2 = client.get('/health', headers={'x-api-key': 'testkey123'})
    assert r2.status_code == 200
