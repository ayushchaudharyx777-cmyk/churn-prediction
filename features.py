"""Shared feature engineering, actionability, risk bands, retention actions."""
import pandas as pd

SVC = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
       "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
ENGINEERED = ["n_services", "charges_per_month", "monthly_vs_avg"]

# Features the business can actually change. tenure / TotalCharges / demographics explain churn but cannot be "fixed".
ACTIONABLE = ["Contract", "PaymentMethod", "PaperlessBilling", "OnlineSecurity", "OnlineBackup",
              "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "InternetService",
              "MultipleLines", "PhoneService", "MonthlyCharges"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blanks -> NaN
    df["n_services"] = df[SVC].isin(["Yes", "DSL", "Fiber optic"]).sum(axis=1)
    df["charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["monthly_vs_avg"] = df["MonthlyCharges"] - df["charges_per_month"]  # price-change proxy
    return df


def is_actionable(feature_name: str) -> bool:
    return any(feature_name == c or feature_name.startswith(c + "_") for c in ACTIONABLE)


def risk_band(p: float, threshold: float, capacity_cutoff: float) -> str:
    """Critical = in the call-list capacity (top K%), High = worth an offer by cost, Medium/Low = rest."""
    if p >= capacity_cutoff:
        return "Critical"
    if p >= threshold:
        return "High"
    if p >= threshold / 2:
        return "Medium"
    return "Low"


# (substring of one-hot feature name, retention action)
ACTIONS = [
    ("Contract_Month-to-month", "Offer discount for a 1-year contract"),
    ("PaymentMethod_Electronic check", "Nudge to auto-pay with a small credit"),
    ("OnlineSecurity_No", "Offer free online-security add-on for 3 months"),
    ("TechSupport_No", "Offer free tech-support trial"),
    ("OnlineBackup_No", "Bundle online backup free for 3 months"),
    ("DeviceProtection_No", "Offer a device-protection trial"),
    ("InternetService_Fiber optic", "Check fiber quality & price; offer a plan review"),
    ("StreamingTV_Yes", "Offer a streaming bundle discount"),
    ("StreamingMovies_Yes", "Offer a streaming bundle discount"),
    ("PaperlessBilling_Yes", "Proactive billing-clarity check-in call"),
    ("MonthlyCharges", "Review plan pricing; offer a cheaper bundle"),
    ("monthly_vs_avg", "Review recent price change; offer a loyalty credit"),
]


def suggest_action(reasons) -> str:
    """First matching action for the customer's reasons, in order of importance for THAT customer."""
    if isinstance(reasons, str):
        reasons = [r.strip() for r in reasons.split(",") if r.strip()]
    for r in reasons:
        for key, action in ACTIONS:
            if key in r:
                return action
    return "Standard retention call"
