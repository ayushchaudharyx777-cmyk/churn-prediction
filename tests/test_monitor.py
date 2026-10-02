import numpy as np
import pandas as pd

from features import ENGINEERED
from monitor import (drift_report, monitored_columns, prepare_current, psi_categorical, psi_numeric,
                     simulate_drift, summarize)


def test_psi_identical_is_near_zero():
    a = np.random.default_rng(0).normal(size=3000)
    assert psi_numeric(a, a) < 0.01


def test_psi_detects_numeric_shift():
    a = np.random.default_rng(0).normal(size=3000)
    assert psi_numeric(a, a + 2) > 0.25


def test_psi_detects_categorical_shift():
    assert psi_categorical(["a"] * 80 + ["b"] * 20, ["a"] * 20 + ["b"] * 80) > 0.25


def test_insufficient_data_flag():
    rep = pd.DataFrame({"feature": ["x"], "psi": [0.9], "status": ["alert"]})
    assert summarize(rep, 5)["overall"] == "insufficient_data"


def test_simulated_drift_raises_alert(trained):
    md = trained / "models"
    ref = pd.read_csv(md / "reference.csv")
    base = ref.drop(columns=["churn_probability", *ENGINEERED]).sample(400, random_state=1)
    cur = prepare_current(simulate_drift(base), md)
    rep = drift_report(ref, cur, monitored_columns(md))
    assert summarize(rep, len(cur))["overall"] == "alert"


def test_no_alert_on_same_distribution(trained):
    md = trained / "models"
    ref = pd.read_csv(md / "reference.csv")
    half = len(ref) // 2
    rep = drift_report(ref.iloc[:half], ref.iloc[half:], monitored_columns(md))
    assert summarize(rep, len(ref) - half)["overall"] != "alert"


def test_make_drift_batch_is_raw_and_drifted(trained):
    from monitor import make_drift_batch
    ref = pd.read_csv(trained / "models/reference.csv")
    batch = make_drift_batch(ref, n=200)
    assert "churn_probability" not in batch.columns and "n_services" not in batch.columns
    assert batch["tenure"].mean() < ref["tenure"].mean()
