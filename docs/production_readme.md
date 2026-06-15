# Production Runbook (Prototype → Production)

This file contains pragmatic steps to deploy the FRAUDGENOME prototype to a production-like environment.

1. Build container

```bash
docker build -t fraudgenome:latest .
```

2. Provide models

Place trained models in `models/` in repo root with filenames expected by the API:
- `lgb_model.joblib`
- `xgb_model.json`
- `canon.npy` (optional)
- `shap_sample.joblib` (optional)

3. Run container

```bash
docker run --rm -p 8000:8000 -e FRAUDGENOME_LOG_LEVEL=INFO fraudgenome:latest
```

4. Health checks

- `GET /health`
- `GET /models/version`

5. CI & tests

Pushes to `main`/`master` run the CI suite which executes unit tests.

6. TODO (production hardening)
- Add secrets management for any GenAI keys
- Integrate logging sink (ELK/Datadog)
- Add Prometheus metrics and liveness/readiness probes
- Harden security (authentication, TLS, RBAC)
