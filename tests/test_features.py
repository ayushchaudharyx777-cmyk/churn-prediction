import pandas as pd

from features import SVC, add_features, is_actionable, risk_band, suggest_action


def _row(total):
    d = {c: "Yes" for c in SVC}
    d.update({"tenure": 9, "MonthlyCharges": 50.0, "TotalCharges": total})
    return pd.DataFrame([d])


def test_blank_total_charges_becomes_nan():
    assert pd.isna(add_features(_row(" "))["TotalCharges"][0])


def test_engineered_values():
    out = add_features(_row("100"))
    assert out["n_services"][0] == len(SVC)
    assert out["charges_per_month"][0] == 10.0


def test_actionability():
    assert is_actionable("Contract_Month-to-month") and is_actionable("PaymentMethod_Electronic check")
    assert not is_actionable("tenure") and not is_actionable("TotalCharges") and not is_actionable("SeniorCitizen")


def test_action_depends_on_reason_order():
    assert "contract" in suggest_action(["Contract_Month-to-month", "TechSupport_No"])
    assert "tech-support" in suggest_action(["TechSupport_No", "Contract_Month-to-month"])
    assert suggest_action([]) == "Standard retention call"
    assert suggest_action("nothing_matches") == "Standard retention call"


def test_risk_bands():
    assert risk_band(0.9, 0.3, 0.6) == "Critical"
    assert risk_band(0.4, 0.3, 0.6) == "High"
    assert risk_band(0.2, 0.3, 0.6) == "Medium"
    assert risk_band(0.01, 0.3, 0.6) == "Low"
