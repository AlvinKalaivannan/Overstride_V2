"""The evaluation protocol. CLAUDE.md makes subject-level grouping
non-negotiable; these assert it actually holds."""
import numpy as np
import pytest
from phase1_eval import check_no_group_leakage, ci, make_folds, paired_delta


def _cohort(n=200, n_sub=60, seed=0):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, n_sub, n).astype(str)
    y = rng.integers(0, 2, n)
    return y, groups


def test_no_subject_straddles_a_fold():
    y, groups = _cohort()
    folds = make_folds(y, groups)
    check_no_group_leakage(folds, groups)          # must not raise
    assert len(folds) == 25, "5 splits x 5 seeds"


def test_leakage_check_actually_catches_leakage():
    """A guard that never fires is worse than none."""
    y, groups = _cohort()
    folds = make_folds(y, groups)
    folds[0]["train"] = np.concatenate([folds[0]["train"], folds[0]["test"][:1]])
    with pytest.raises(AssertionError):
        check_no_group_leakage(folds, groups)


def test_every_session_is_held_out_once_per_seed():
    y, groups = _cohort()
    folds = make_folds(y, groups)
    seen = np.zeros(len(y), int)
    for f in folds:
        seen[f["test"]] += 1
    assert set(np.unique(seen)) == {5}, "each row must be tested once per seed"


def test_paired_delta_requires_matching_folds():
    a = {"name": "a", "auc": np.zeros(25)}
    b = {"name": "b", "auc": np.zeros(24)}
    with pytest.raises(ValueError):
        paired_delta(a, b)


def test_ci_is_a_percentile_interval():
    v = np.linspace(0.0, 1.0, 1001)
    lo, hi = ci(v)
    assert abs(lo - 0.025) < 1e-6 and abs(hi - 0.975) < 1e-6
