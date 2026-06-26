"""Tests for leakage-safe encoders."""
import numpy as np
import pandas as pd

from src.data import encoders


def test_categorical_encoder_unknown_and_rare():
    X = pd.DataFrame({"c": ["a"] * 100 + ["b"] * 100 + ["rare"] * 3})
    enc = encoders.CategoricalEncoder(["c"], min_frequency=50).fit(X)
    # 'a' and 'b' kept (idx 1,2); 'rare' folded into UNKNOWN=0; cardinality = 2 kept + 1 unknown
    assert enc.cardinalities_["c"] == 3
    out = enc.transform(X)
    assert set(np.unique(out)).issubset({0, 1, 2})
    assert (out[200:] == 0).all()  # rare -> UNKNOWN


def test_categorical_encoder_unseen_category_maps_to_unknown():
    train = pd.DataFrame({"c": ["a"] * 60 + ["b"] * 60})
    test = pd.DataFrame({"c": ["a", "z", "b"]})  # 'z' unseen
    enc = encoders.CategoricalEncoder(["c"], min_frequency=50).fit(train)
    out = enc.transform(test)
    assert out[1, 0] == 0  # unseen -> UNKNOWN


def test_numeric_standardizer_fit_on_train_only():
    train = pd.DataFrame({"x": [0.0, 10.0, 20.0, np.nan]})
    std = encoders.NumericStandardizer(["x"]).fit(train)
    # NaN imputed with train median (10), then standardised; train mean ~ 10
    out = std.transform(train)
    assert out.shape == (4, 1)
    assert abs(float(out.mean())) < 1e-6  # standardised train ~ zero mean
    # applying to a new frame uses stored stats, not the new frame's
    new = pd.DataFrame({"x": [10.0]})
    assert abs(float(std.transform(new)[0, 0])) < 1e-6  # 10 == train mean -> ~0


def test_tabular_encoder_shapes_and_cardinalities():
    import pandas as pd

    from src.data import clean

    raw = pd.DataFrame(
        {
            "MIS_Status": ["CHGOFF", "P I F"] * 60,
            "GrAppv": ["$100,000.00 "] * 120,
            "SBA_Appv": ["$50,000.00 "] * 120,
            "Term": [120] * 120,
            "NoEmp": [3] * 120,
            "CreateJob": [0] * 120,
            "RetainedJob": [0] * 120,
            "ApprovalFY": [2004] * 120,
            "State": (["IN"] * 60) + (["OH"] * 60),
            "BankState": ["OH"] * 120,
            "NewExist": [1] * 120,
            "UrbanRural": [1] * 120,
            "RevLineCr": ["N"] * 120,
            "LowDoc": ["Y"] * 120,
            "NAICS": [451120] * 120,
            "FranchiseCode": [1] * 120,
        }
    )
    X = clean.engineer(raw)
    enc = encoders.TabularEncoder().fit(X)
    xn, xc = enc.transform(X)
    assert xn.shape == (120, enc.n_numeric)
    assert xc.shape == (120, len(enc.categorical_cols))
    assert len(enc.cardinalities) == len(enc.categorical_cols)
    assert all(c >= 1 for c in enc.cardinalities)
