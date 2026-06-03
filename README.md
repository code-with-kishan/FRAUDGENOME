# MuleGuard AI + ShieldScan

Enterprise-grade fraud intelligence platform scaffold.

Contents:
- `ml/` — prototype training and DTW utilities
- `api/` — FastAPI inference skeleton
- `notebooks/` — prototype notebooks and usage
- `docs/` — architecture and design notes
- `infra/` — docker-compose for quick local run

Quick start

1. Create a Python venv and install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run prototype training (reads `DataSet.csv` in repo root):

```bash
python ml/train.py --data 'DataSet.csv' --out models/
```

3. Start API server (serves CTI stub):

```bash
uvicorn api.app:app --reload --port 8000
```

This scaffold provides a runnable prototype and place to expand into the full MuleGuard AI + ShieldScan platform.

Additional utilities:
- Cohort scoring and evaluation: `python -m ml.cohort_scoring --normalized data/processed/normalized.parquet --labels data/processed/labels.parquet --cohorts data/processed/cohorts.parquet --models models/ --selected models/selected_features.json --out models/`
- FraudDNA temporal backtest: `python -m ml.frauddna_backtest --normalized data/processed/normalized.parquet --labels data/processed/labels.parquet --cohorts data/processed/cohorts.parquet --out models/`

Real-time demo (local):

- Start API server:

```bash
uvicorn api.app:app --reload
```

- Open http://localhost:8000/ in a browser to see the D3 force demo. Click an account to subscribe to simulated CTI updates and use "Generate Brief" to download a PDF draft.

Security & Compliance:

- Failure Mode & Remediation template: [docs/FMR.md](docs/FMR.md)

Additional notes & developer quicklinks
------------------------------------

- Run the full prototype (recommended order):
	1. Build processed data: run your ingestion scripts to produce `data/processed/normalized.parquet`, `data/processed/labels.parquet`, and optional `data/processed/cohorts.parquet`.
	2. Train ensemble models and artifacts:

```bash
python -m ml.train_pipeline --input data/processed/normalized.parquet --labels data/processed/labels.parquet --out models/
```

	This writes model artifacts (`models/lgb_model.joblib`, `models/xgb_model.json`), `selected_features.json`, `shap_sample.joblib`, and `frauddna_manifest.parquet` used by the API and demo.

- Generate cohort scoring report (validation):

```bash
python -m ml.cohort_scoring --normalized data/processed/normalized.parquet --labels data/processed/labels.parquet --cohorts data/processed/cohorts.parquet --models models/ --selected models/selected_features.json --out models/
```

- Run drift check and optional retrain (ADWIN + retrain):

```bash
curl -X POST http://localhost:8000/models/drift_check -H "Content-Type: application/json" -d '{"model_path":"models/lgb_model.joblib","normalized":"data/processed/normalized.parquet","labels":"data/processed/labels.parquet","retrain_cmd":["python3","-m","ml.train_pipeline"]}'
```

- ShieldScan (APK analysis + correlation):

```bash
curl -X POST http://localhost:8000/shieldscan/analyze -H "Content-Type: application/json" -d '{"apk_path":"samples/apks/suspicious.apk","frauddna_manifest":"models/frauddna_manifest.parquet","accounts_events":"data/processed/accounts_events.parquet","dynamic_trace":"samples/traces/suspicious_trace.json"}'
```

- GenAI brief drafts (server-side PDF): use the demo UI or call `/briefs/generate` with `account_id` and available evidence. The `ml/briefs.py` module prepares a Claude prompt if you want to call an LLM from a secure deployment layer.

- Security & audit:
	- Set `MULEGUARD_API_KEYS` to a comma-separated list of API keys to require `X-API-Key` for API access.
	- Audit log file location: `MULEGUARD_AUDIT_LOG` (defaults to `models/audit.log`).

- Tests and CI:
	- Unit tests are in `tests/`. Run them locally with:

```bash
pip install -r requirements.txt
pytest -q
```

If CI is enabled, tests will run automatically on PRs.

Support & next steps
---------------------
- If you'd like, I can wire secure Claude integration (server-side) for final brief polishing, add an emulator runner for ShieldScan dynamic analysis, or harden the demo for production (JWT, RBAC, signed brief URLs). Open which option you prefer and I'll implement it.
