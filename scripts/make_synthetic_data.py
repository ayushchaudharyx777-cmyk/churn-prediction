"""Create a SYNTHETIC Telco-like CSV for CI / smoke tests only. Never report metrics from this data."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
ap.add_argument("--rows", type=int, default=1500)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
n = a.rows


def yn():
    return rng.choice(["Yes", "No"], n)


ten = rng.integers(0, 72, n)
mc = rng.uniform(20, 110, n).round(2)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .2])
logit = -1 + 1.5 * (contract == "Month-to-month") - 0.03 * ten + 0.01 * mc
churn = rng.random(n) < 1 / (1 + np.exp(-logit))
tc = (ten * mc).round(2).astype(str)
tc[rng.choice(n, max(3, n // 200), replace=False)] = " "   # blanks, like the real data
df = pd.DataFrame({
    "customerID": [f"C{i}" for i in range(n)], "gender": rng.choice(["Male", "Female"], n),
    "SeniorCitizen": rng.integers(0, 2, n), "Partner": yn(), "Dependents": yn(), "tenure": ten,
    "PhoneService": yn(), "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
    "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n), "OnlineSecurity": yn(),
    "OnlineBackup": yn(), "DeviceProtection": yn(), "TechSupport": yn(), "StreamingTV": yn(),
    "StreamingMovies": yn(), "Contract": contract, "PaperlessBilling": yn(),
    "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                 "Credit card (automatic)"], n),
    "MonthlyCharges": mc, "TotalCharges": tc, "Churn": np.where(churn, "Yes", "No")})
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(f"wrote {a.out} ({n} rows, SYNTHETIC)")
