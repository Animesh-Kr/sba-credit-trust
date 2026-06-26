"""Tests for the serving bundle: round-trip, calibrated prediction, conformal decision."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import clean, schema
from src.models.baselines import build_preprocessor
from src.serve.bundle import ServingBundle


def _toy_bundle():
    rng = np.random.default_rng(0)
    n = 400
    raw = pd.DataFrame({
        "MIS_Status": rng.choice(["CHGOFF", "P I F"], n),
        "GrAppv": ["$100,000.00 "] * n, "SBA_Appv": ["$50,000.00 "] * n,
        "Term": rng.integers(12, 300, n), "NoEmp": rng.integers(0, 20, n),
        "CreateJob": 0, "RetainedJob": 0, "ApprovalFY": rng.integers(1998, 2010, n),
        "State": rng.choice(["CA", "TX", "NY"], n), "BankState": "CA",
        "NewExist": rng.choice([1, 2], n), "UrbanRural": 1,
        "RevLineCr": rng.choice(["Y", "N"], n), "LowDoc": "N",
        "NAICS": 451120, "FranchiseCode": 1,
    })
    X = clean.engineer(raw)
    y = clean.make_target(raw).fillna(0).astype(int)
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(X, y)
    return ServingBundle(
        pipeline=pipe, temperature=1.3,
        conformal={0.05: {"t_lo": 0.1, "t_hi": 0.9}, 0.10: {"t_lo": 0.2, "t_hi": 0.8}},
        numeric_cols=schema.NUMERIC_FEATURES + schema.ENGINEERED_NUMERIC,
        categorical_cols=schema.CATEGORICAL_FEATURES,
        choices={"State": ["CA", "TX", "NY"]}, numeric_defaults={"Term": 84.0},
        metadata={"model": "toy"},
    ), X


def test_predict_proba_in_unit_interval():
    b, X = _toy_bundle()
    p = b.predict_proba(X)
    assert p.shape == (len(X),)
    assert (p >= 0).all() and (p <= 1).all()


def test_decide_labels_and_thresholds():
    b, _ = _toy_bundle()
    probs = np.array([0.05, 0.5, 0.95])
    dec = b.decide(probs, alpha=0.05)
    assert dec[0] == "auto-approve"      # below t_lo
    assert dec[1] == "abstain / refer"   # between
    assert dec[2] == "auto-reject"       # above t_hi


def test_bundle_roundtrip(tmp_path):
    b, X = _toy_bundle()
    p = tmp_path / "b.joblib"
    b.save(str(p))
    b2 = ServingBundle.load(str(p))
    assert np.allclose(b.predict_proba(X), b2.predict_proba(X))
    assert b2.temperature == b.temperature
