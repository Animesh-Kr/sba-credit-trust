"""Tests for cleaning + feature engineering + target derivation."""
import numpy as np
import pandas as pd

from src.data import clean, schema


def test_parse_currency():
    s = pd.Series(["$60,000.00 ", "$0.00 ", "1,234.5", "", "abc"])
    out = clean.parse_currency(s)
    assert out.iloc[0] == 60000.0
    assert out.iloc[1] == 0.0
    assert out.iloc[2] == 1234.5
    assert np.isnan(out.iloc[3])
    assert np.isnan(out.iloc[4])


def test_make_target_mapping():
    df = pd.DataFrame({"MIS_Status": ["CHGOFF", "P I F", np.nan, "weird"]})
    y = clean.make_target(df)
    assert y.iloc[0] == 1.0      # default = event
    assert y.iloc[1] == 0.0      # paid
    assert np.isnan(y.iloc[2])
    assert np.isnan(y.iloc[3])


def _toy_raw():
    return pd.DataFrame(
        {
            "MIS_Status": ["CHGOFF", "P I F"],
            "GrAppv": ["$100,000.00 ", "$50,000.00 "],
            "SBA_Appv": ["$50,000.00 ", "$50,000.00 "],
            "Term": [300, 60],
            "NoEmp": [5, 2],
            "CreateJob": [1, 0],
            "RetainedJob": [0, 3],
            "ApprovalFY": [2004, 1998],
            "State": ["IN", "OH"],
            "BankState": ["OH", "OH"],
            "NewExist": [2, 1],
            "UrbanRural": [1, 0],
            "RevLineCr": ["N", "T"],          # 'T' is junk -> NaN
            "LowDoc": ["Y", "N"],
            "NAICS": [451120, 0],             # 0 -> missing sector
            "FranchiseCode": [1, 55555],      # 1 -> not franchise, 55555 -> franchise
            "DisbursementGross": ["$100,000.00 ", "$50,000.00 "],
        }
    )


def test_engineer_shapes_and_features():
    X = clean.engineer(_toy_raw())
    assert list(X.columns) == schema.feature_columns(include_disbursement=False)
    # sba_portion = SBA_Appv / GrAppv
    assert abs(X["sba_portion"].iloc[0] - 0.5) < 1e-9
    assert abs(X["sba_portion"].iloc[1] - 1.0) < 1e-9
    # real_estate: Term>=240
    assert X["real_estate"].iloc[0] == 1
    assert X["real_estate"].iloc[1] == 0
    # naics sector first 2 digits / missing
    assert X["naics_sector"].iloc[0] == "45"
    assert pd.isna(X["naics_sector"].iloc[1])
    # franchise flag
    assert X["is_franchise"].iloc[0] == 0
    assert X["is_franchise"].iloc[1] == 1
    # junk RevLineCr coerced to NaN
    assert pd.isna(X["RevLineCr"].iloc[1])


def test_engineer_excludes_leakage_columns():
    X = clean.engineer(_toy_raw(), include_disbursement=False)
    for col in schema.POST_OUTCOME_LEAKAGE + schema.POST_DECISION:
        assert col not in X.columns


def test_feature_and_leakage_lists_disjoint():
    feats = set(schema.feature_columns(include_disbursement=True))
    leaks = set(schema.leakage_columns(include_disbursement=True))
    assert feats.isdisjoint(leaks)
