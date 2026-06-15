# Architecture Overview

This document summarizes the initial architecture for FRAUDGENOME.

- Ingest via Kafka/MSK
- Feature store: Postgres + Redis
- ML Training: MLflow, Spark / Python
- Serving: FastAPI + Redis cache
- Graph engine: NetworkX microservice with Louvain

See `README.md` for quick start.
