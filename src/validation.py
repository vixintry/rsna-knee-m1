"""Validation utilities for competition labels and study-level evaluation."""

from typing import Sequence

import numpy as np
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
    return pd.DataFrame({
        "positive": (df[TARGETS] == 1).sum(),
        "negative": (df[TARGETS] == 0).sum(),
        "missing": df[TARGETS].isna().sum(),
    })


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


def make_multilabel_folds(
    study_ids: Sequence[str],
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Assign deterministic study-level folds, balancing rare positive labels.

    This is a heuristic iterative-stratification procedure, not an optimality
    guarantee. The caller must inspect support in every fold before training.
    """
    ids = np.asarray(study_ids, dtype=str)
    y = np.asarray(labels)
    if ids.ndim != 1:
        raise ValueError("study_ids must be one-dimensional")
    if y.ndim != 2 or y.shape[0] != len(ids) or y.shape[1] == 0:
        raise ValueError("labels must be a non-empty 2D array with one row per study")
    if len(ids) == 0:
        raise ValueError("at least one study is required")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("study_ids must be unique; duplicate studies risk leakage")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("labels must contain only finite binary values 0 and 1")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2")
    if len(ids) < n_splits:
        raise ValueError("n_splits cannot exceed the number of studies")

    y = y.astype(np.int8, copy=False)
    positives = y.sum(axis=0)
    negatives = len(ids) - positives
    bad = [j for j, (p, n) in enumerate(zip(positives, negatives))
           if p < n_splits or n < n_splits]
    if bad:
        raise ValueError(
            "each label needs at least n_splits positive and negative studies "
            "to support both classes in every validation fold; "
            f"invalid label column indices: {bad}"
        )

    rng = np.random.default_rng(seed)
    desired = np.tile(positives / float(n_splits), (n_splits, 1))
    sizes = np.zeros(n_splits, dtype=np.int64)
    folds = np.full(len(ids), -1, dtype=np.int64)
    remaining = set(range(len(ids)))

    while remaining:
        remaining_idx = np.fromiter(sorted(remaining), dtype=np.int64)
        counts = y[remaining_idx].sum(axis=0)
        active = np.flatnonzero(counts > 0)
        if len(active) == 0:
            for i in rng.permutation(remaining_idx):
                choices = np.flatnonzero(sizes == sizes.min())
                f = int(rng.choice(choices))
                folds[i] = f
                sizes[f] += 1
                remaining.remove(int(i))
            break

        rare = active[counts[active] == counts[active].min()]
        label = int(rng.choice(rare))
        candidates_studies = np.array(
            [i for i in remaining_idx if y[i, label] == 1], dtype=np.int64
        )
        rng.shuffle(candidates_studies)
        for i in candidates_studies:
            if int(i) not in remaining:
                continue
            best = desired[:, label].max()
            choices = np.flatnonzero(np.isclose(desired[:, label], best))
            choices = choices[sizes[choices] == sizes[choices].min()]
            f = int(rng.choice(choices))
            folds[i] = f
            sizes[f] += 1
            desired[f] -= y[i]
            remaining.remove(int(i))

    if (folds < 0).any() or set(folds.tolist()) != set(range(n_splits)):
        raise RuntimeError("internal fold assignment error: one or more folds are missing")
    return folds


def fold_support(
    labels: np.ndarray, folds: Sequence[int]
) -> dict[int, dict[str, int | list[int]]]:
    """Summarise per-fold study count and positive/negative counts by label."""
    y = np.asarray(labels)
    fids = np.asarray(folds)
    if y.ndim != 2 or fids.ndim != 1 or len(fids) != len(y):
        raise ValueError("labels must be 2D and folds must have one entry per row")
    result = {}
    for f in sorted(np.unique(fids).tolist()):
        yf = y[fids == f]
        result[int(f)] = {
            "n_studies": int(len(yf)),
            "positive": yf.sum(axis=0).astype(int).tolist(),
            "negative": (len(yf) - yf.sum(axis=0)).astype(int).tolist(),
        }
    return result


def per_label_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: Sequence[str] | None = None,
) -> dict[str, float | None]:
    """Return per-label AUC; a single-class target is explicitly reported None."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true)
    p = np.asarray(y_score, dtype=float)
    if y.ndim != 2 or p.shape != y.shape:
        raise ValueError("y_true and y_score must be 2D arrays with identical shape")
    if not np.isfinite(p).all():
        raise ValueError("y_score contains non-finite predictions")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only binary values 0 and 1")
    names = list(label_names) if label_names is not None else [
        f"label_{i}" for i in range(y.shape[1])
    ]
    if len(names) != y.shape[1] or len(set(names)) != len(names):
        raise ValueError("label_names must be unique and match the number of columns")
    scores = {}
    for j, name in enumerate(names):
        scores[name] = (
            None if np.unique(y[:, j]).size < 2
            else float(roc_auc_score(y[:, j], p[:, j]))
        )
    return scores


def macro_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: Sequence[str] | None = None,
    *,
    require_all_labels: bool = True,
) -> float:
    """Compute macro AUC; by default fail rather than silently omit undefined labels."""
    scores = per_label_roc_auc(y_true, y_score, label_names)
    undefined = [name for name, score in scores.items() if score is None]
    if undefined and require_all_labels:
        raise ValueError(f"ROC-AUC undefined for single-class labels: {undefined}")
    valid = [score for score in scores.values() if score is not None]
    if not valid:
        raise ValueError("macro ROC-AUC is undefined because no label has both classes")
    return float(np.mean(valid))
