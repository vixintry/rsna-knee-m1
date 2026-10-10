
"""Load and validate competition CSV files."""

from pathlib import Path

import pandas as pd

from src.labels import TARGETS


def load_train_csv(path: str | Path) -> pd.DataFrame:
    """Load the training CSV and check its required columns."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {path}")

    df = pd.read_csv(path)

    required_columns = ["StudyInstanceUID", "Report", *TARGETS]
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Training CSV is missing columns: {missing}")

    if df["StudyInstanceUID"].isna().any():
        raise ValueError("Training CSV contains missing StudyInstanceUID values.")

    if df["StudyInstanceUID"].duplicated().any():
        raise ValueError("Training CSV contains duplicate StudyInstanceUID values.")

    return df
