import pandas as pd

from validate import validate_training



def test_missing_columns_is_error():
    r = validate_training(pd.DataFrame({"a": [1]}))
    assert r["errors"]


def test_negative_tenure_and_bad_target(trained):
    df = pd.read_csv(trained / "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    assert validate_training(df)["errors"] == []
    bad = df.copy()
    bad.loc[0, "tenure"] = -5
    bad.loc[1, "Churn"] = "Maybe"
    errs = " ".join(validate_training(bad)["errors"])
    assert "negative" in errs and "Churn" in errs
