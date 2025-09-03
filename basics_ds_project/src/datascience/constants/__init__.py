from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent

CONFIG_FILE_PATH = BASE_PATH / "config" / "config.yaml"
PARAMS_FILE_PATH = BASE_PATH / "config" / "params.yaml"
SCHEMA_FILE_PATH = BASE_PATH / "config" / "schema.yaml"
