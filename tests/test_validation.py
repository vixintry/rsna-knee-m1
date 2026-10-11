import numpy as np
import pytest

from src.validation import (
    fold_support,
    macro_roc_auc,
    make_multilabel_folds,
    per_label_roc_auc,
)


def sample_data():
    rng = np.random.default_rng(123)
    y = np.column_stack([
        np.array([1] * 12 + [0] * 18),
        np.array([1] * 10 + [0] * 20),
        np.array([1] * 8 + [0] * 22),
        np.array([1] * 15 + [0] * 15),
    ])
    order = rng.permutation(len(y))
    return [f"study-{i:03d}" for i in order], y[order]


def test_folds_are_deterministic_and_cover_each_study_once():
    ids, y = sample_data()
    a = make_multilabel_folds(ids, y, n_splits=5, seed=42)
    b = make_multilabel_folds(ids, y, n_splits=5, seed=42)
    assert np.array_equal(a, b)
    assert len(a) == len(ids)
    assert set(a.tolist()) == {0, 1, 2, 3, 4}


def test_folds_have_balanced_sizes_and_each_fold_has_both_classes():
    ids, y = sample_data()
    folds = make_multilabel_folds(ids, y, n_splits=5, seed=42)
    sizes = np.bincount(folds)
    assert sizes.max() - sizes.min() <= 1
    support = fold_support(y, folds)
    for f in support:
        assert support[f]["n_studies"] > 0
        assert all(p > 0 and n > 0 for p, n in zip(support[f]["positive"], support[f]["negative"]))


def test_duplicate_study_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        make_multilabel_folds(["a", "a", "c"], np.array([[1, 0], [0, 1], [1, 1]]), 2)


def test_missing_or_nonbinary_labels_are_rejected():
    with pytest.raises(ValueError, match="binary"):
        make_multilabel_folds(["a", "b", "c"], np.array([[1, 0], [0, 1], [1, np.nan]]), 2)
    with pytest.raises(ValueError, match="binary"):
        make_multilabel_folds(["a", "b", "c"], np.array([[1, 0], [0, 1], [1, 2]]), 2)


def test_rare_class_fewer_than_folds_is_rejected():
    y = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    with pytest.raises(ValueError, match="n_splits positive and negative"):
        make_multilabel_folds(["a", "b", "c", "d"], y, 3)


def test_auc_functions_return_per_label_and_macro_scores():
    y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    p = np.array([[0.1, 0.2], [0.2, 0.8], [0.7, 0.3], [0.9, 0.7]])
    scores = per_label_roc_auc(y, p, ["ACL", "MCL"])
    assert scores == {"ACL": 1.0, "MCL": 1.0}
    assert macro_roc_auc(y, p, ["ACL", "MCL"]) == 1.0


def test_single_class_auc_is_explicit_not_silently_invented():
    y = np.array([[0, 1], [0, 1], [0, 0]])
    p = np.array([[0.2, 0.8], [0.3, 0.7], [0.1, 0.4]])
    scores = per_label_roc_auc(y, p, ["ACL", "MCL"])
    assert scores["ACL"] is None
    assert isinstance(scores["MCL"], float)
    with pytest.raises(ValueError, match="undefined"):
        macro_roc_auc(y, p, ["ACL", "MCL"])
    assert macro_roc_auc(y, p, ["ACL", "MCL"], require_all_labels=False) == scores["MCL"]
