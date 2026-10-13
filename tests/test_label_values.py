import pandas as pd
import pytest

from src.labels import TARGETS
from src.validation import validate_label_values


def test_valid_labels_including_missing_are_accepted():
    df = pd.DataFrame({target: [0, 1, None] for target in TARGETS})
    validate_label_values(df)


def test_non_binary_label_is_rejected():
    df = pd.DataFrame({target: [0, 1, None] for target in TARGETS})
    df.loc[0, "ACL"] = 2
    with pytest.raises(ValueError):
        validate_label_values(df)
