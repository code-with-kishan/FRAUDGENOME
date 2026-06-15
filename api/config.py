import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()
def _env(primary: str, legacy: str, default: str) -> str:
    return os.environ.get(primary, os.environ.get(legacy, default))


MODEL_DIR = _env('FRAUDGENOME_MODEL_DIR', 'MULEGUARD_MODEL_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
LOG_LEVEL = _env('FRAUDGENOME_LOG_LEVEL', 'MULEGUARD_LOG_LEVEL', 'INFO')
HOST = _env('FRAUDGENOME_HOST', 'MULEGUARD_HOST', '0.0.0.0')
PORT = int(_env('FRAUDGENOME_PORT', 'MULEGUARD_PORT', '8000'))

def configure_logging():
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(name)s %(message)s')
