"""Paths defenitions"""
from pathlib import Path

PROJ_NAME = "Shopee"

DATA_ROOT = Path(f"~/Data/{PROJ_NAME}").expanduser()
PROJ_PATH = Path(f"~/Projects/{PROJ_NAME}").expanduser()
ON_DRIVE_PATH = Path(f"/content/drive/My Drive/Colab Notebooks/{PROJ_NAME}")

LOGS_PATH = PROJ_PATH / "logs"
MODELS_PATH = PROJ_PATH / "models"
