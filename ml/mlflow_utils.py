import mlflow
import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger('muleguard.mlflow')


def start_training_run(experiment_name: str = 'muleguard_training', params: Dict[str, Any] = None):
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run()
    if params:
        mlflow.log_params(params)
    return run.info.run_id


def log_model_artifact(model_path: str, artifact_path: str = 'models'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    mlflow.log_artifact(model_path, artifact_path)


def register_model(run_id: str, model_artifact_path: str, model_name: str = 'muleguard_model') -> str:
    # model_artifact_path is path inside run artifacts (e.g., 'models/lgb_model.joblib')
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    try:
        result = mlflow.register_model(model_uri, model_name)
        logger.info('Registered model %s at version %s', model_name, result.version)
        return result.version
    except Exception as e:
        logger.exception('Failed to register model: %s', e)
        raise


def promote_model(model_name: str, version: str, stage: str = 'Production'):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(name=model_name, version=version, stage=stage)
    logger.info('Promoted model %s version %s to %s', model_name, version, stage)


def list_models(model_name: str):
    client = mlflow.tracking.MlflowClient()
    return client.get_latest_versions(model_name)
