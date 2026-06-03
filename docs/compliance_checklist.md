# MuleGuard Compliance Checklist

This checklist summarizes security and compliance items for the prototype.

- [ ] Secrets: No secrets committed to repository. Use env vars for keys and tokens.
- [ ] API Auth: `X-API-Key` header enforced when `MULEGUARD_API_KEYS` is set.
- [ ] Audit Logging: Requests and outcomes are written to `models/audit.log` (JSONL).
- [ ] Data Access: Models, artifacts, and PII datasets are stored under `models/` and `data/` with restricted permissions.
- [ ] FMR: Failure Mode & Remediation template exists at `docs/FMR.md`.
- [ ] Tests: Unit tests and integration tests executed in CI; include security tests for auth and audit logging.
- [ ] Containerization: Dockerfile provided; ensure runtime secrets injection (env) and non-root user execution.
- [ ] Monitoring: Add alerting for drift detector `ml/drift.py` and failed retrain attempts.
- [ ] MLflow: Use secured tracking server for model registration in production; restrict access to model registry.
- [ ] Dependencies: Regular SCA scanning for known CVEs.

Recommended next steps:
- Rotate API keys and set strict ACLs for MLflow.
- Add signed brief URLs and short-lived tokens for PDF download.
- Add data retention and purge policies for logs and generated artifacts.
