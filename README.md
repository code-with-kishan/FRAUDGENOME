# FRAUDGENOME

AI-powered fraud intelligence platform for real-time mule account detection.

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

3. Start API server:

```bash
uvicorn api.app:app --reload --port 8000
```

This scaffold provides a runnable prototype for the FRAUDGENOME MVP: ingestion, feature engineering, ensemble scoring, explainability, FraudDNA signatures, risk scoring, and an investigator dashboard.

Additional utilities:
- Cohort scoring and evaluation: `python -m ml.cohort_scoring --normalized data/processed/normalized.parquet --labels data/processed/labels.parquet --cohorts data/processed/cohorts.parquet --models models/ --selected models/selected_features.json --out models/`
- FraudDNA temporal backtest: `python -m ml.frauddna_backtest --normalized data/processed/normalized.parquet --labels data/processed/labels.parquet --cohorts data/processed/cohorts.parquet --out models/`

Real-time demo (local):

- Start API server:

```bash
uvicorn api.app:app --reload
```

- Open http://localhost:8000/ in a browser to explore the investigator dashboard. Click an account to subscribe to simulated risk updates and use "Generate Brief" to download a PDF draft.

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

- Dashboard summary and signature library:

```bash
curl http://localhost:8000/dashboard/summary
curl http://localhost:8000/signatures/library
```

- GenAI brief drafts (server-side PDF): use the demo UI or call `/briefs/generate` with `account_id` and available evidence. The `ml/briefs.py` module prepares a Claude prompt if you want to call an LLM from a secure deployment layer.

- Security & audit:
	- Set `FRAUDGENOME_API_KEYS` or legacy `MULEGUARD_API_KEYS` to a comma-separated list of API keys to require `X-API-Key` for API access.
	- Audit log file location: `FRAUDGENOME_AUDIT_LOG` or legacy `MULEGUARD_AUDIT_LOG` (defaults to `models/audit.log`).

- Tests and CI:
	- Unit tests are in `tests/`. Run them locally with:

```bash
pip install -r requirements.txt
pytest -q
```

If CI is enabled, tests will run automatically on PRs.

Support & next steps
---------------------
- If you'd like, the next strong upgrades would be persistent PostgreSQL storage, Redis-backed real-time scoring, JWT/RBAC, and a richer investigator case workflow.
