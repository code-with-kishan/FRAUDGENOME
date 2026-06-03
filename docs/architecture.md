# Architecture Overview

This document summarizes the initial architecture for MuleGuard AI + ShieldScan.

- Ingest via Kafka/MSK
- Feature store: Postgres + Redis
- ML Training: MLflow, Spark / Python
- Serving: FastAPI + Redis cache
- Graph engine: NetworkX microservice with Louvain
- APK Sandbox: isolated K8s pods

See `README.md` for quick start.
