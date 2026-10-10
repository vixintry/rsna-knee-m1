
"""Validation utilities for competition labels."""

import pandas as pd

from src.labels import TARGETS


def validate_label_columns(df: pd.DataFrame) -> None:
    """Raise an error if any expected target column is missing."""
    missing = [target for target in TARGETS if target not in df.columns]

    if missing:
        raise ValueError(f"Missing target columns: {missing}")


def get_label_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise observed positive, negative, and missing labels."""
    validate_label_columns(df)

    summary = pd.DataFrame(
        {
            "positive": (df[TARGETS] == 1).sum(),
            "negative": (df[TARGETS] == 0).sum(),
            "missing": df[TARGETS].isna().sum(),
        }
    )

    return summary
