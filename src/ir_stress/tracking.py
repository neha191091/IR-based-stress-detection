"""MLflow tracking setup."""

import os
from pathlib import Path


def ensure_mlflow() -> None:
    """Use SQLite tracking backend when none is configured."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        db = Path("mlflow.db").resolve()
        os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{db}"
