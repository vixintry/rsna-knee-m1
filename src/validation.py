
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


def validate_label_values(df: pd.DataFrame) -> None:
    """Allow only binary labels (0 or 1) and missing values."""
    validate_label_columns(df)

    invalid = {}

    for target in TARGETS:
        values = df[target].dropna()
        bad_values = values[~values.isin([0, 1])].unique()

        if len(bad_values) > 0:
            invalid[target] = bad_values.tolist()

    if invalid:
        raise ValueError(f"Invalid label values: {invalid}")
