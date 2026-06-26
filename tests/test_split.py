"""Tests for leakage-safe splitting."""
import numpy as np
import pandas as pd

from src.data import split


def test_stratified_split_preserves_prevalence_and_is_disjoint():
    rng = np.random.default_rng(0)
    y = pd.Series((rng.uniform(size=10000) < 0.18).astype(int))
    tr, te = split.stratified_split(y, test_size=0.2, seed=42)
    assert len(set(tr) & set(te)) == 0
    assert len(tr) + len(te) == len(y)
    assert abs(y.iloc[tr].mean() - y.iloc[te].mean()) < 0.01


def test_stratified_split_is_deterministic():
    y = pd.Series((np.arange(1000) % 5 == 0).astype(int))
    a = split.stratified_split(y, seed=7)
    b = split.stratified_split(y, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_temporal_split_respects_cutoff():
    X = pd.DataFrame({"ApprovalFY": [2000, 2005, 2006, 2007, 2010, np.nan]})
    tr, te = split.temporal_split(X, cutoff_year=2006)
    # train years all <= 2006, test all > 2006, NaN excluded from both
    assert set(X["ApprovalFY"].iloc[tr]) <= {2000, 2005, 2006}
    assert set(X["ApprovalFY"].iloc[te]) <= {2007, 2010}
    assert len(set(tr) & set(te)) == 0
    assert 5 not in set(tr) | set(te)   # the NaN row index
