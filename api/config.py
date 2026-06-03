import os
import logging

MODEL_DIR = os.environ.get('MULEGUARD_MODEL_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
LOG_LEVEL = os.environ.get('MULEGUARD_LOG_LEVEL', 'INFO')
HOST = os.environ.get('MULEGUARD_HOST', '0.0.0.0')
PORT = int(os.environ.get('MULEGUARD_PORT', '8000'))

def configure_logging():
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(name)s %(message)s')
