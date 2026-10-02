"""Data validation: training-data checks and inference-data checks."""
import argparse
import json
import sys

import pandas as pd

from features import SVC

REQUIRED = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", *SVC,
            "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]


def validate_training(df: pd.DataFrame) -> dict:
    errors, warnings = [], []
    missing = [c for c in REQUIRED + ["Churn"] if c not in df.columns]
    if missing:
        return {"errors": [f"missing columns: {missing}"], "warnings": [], "n_rows": len(df)}
    if not set(df["Churn"].dropna().unique()) <= {"Yes", "No"}:
        errors.append("Churn must only contain Yes/No")
    for c in ["tenure", "MonthlyCharges"]:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any():
            errors.append(f"{c} has non-numeric / missing values")
        if (v < 0).any():
            errors.append(f"{c} has negative values")
    tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if (tc < 0).any():
        errors.append("TotalCharges has negative values")
    n_blank = int(tc.isna().sum())
    if n_blank / len(df) > 0.05:
        warnings.append(f"TotalCharges missing in {n_blank} rows (>5%)")
    n_dup = int(df["customerID"].duplicated().sum())
    if n_dup:
        warnings.append(f"{n_dup} duplicate customerID rows")
    rate = float((df["Churn"] == "Yes").mean())
    if not 0.05 < rate < 0.7:
        warnings.append(f"unusual churn rate {rate:.1%}")
    if len(df) < 500:
        warnings.append("fewer than 500 rows")
    return {"errors": errors, "warnings": warnings, "n_rows": len(df),
            "churn_rate": round(rate, 4), "n_blank_total_charges": n_blank}


def validate_inference(df: pd.DataFrame, meta: dict) -> list:
    """Return a list of issues for new data (missing columns, unseen categories)."""
    issues = []
    need = [c for c in meta["columns"] if c in REQUIRED]
    missing = [c for c in need if c not in df.columns]
    if missing:
        issues.append(f"missing columns: {missing}")
        return issues
    for c, allowed in meta["raw_cat"].items():
        bad = set(df[c].dropna().unique()) - set(allowed)
        if bad:
            issues.append(f"{c}: unseen values {sorted(map(str, bad))}")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    a = ap.parse_args()
    r = validate_training(pd.read_csv(a.data))
    print(json.dumps(r, indent=2))
    sys.exit(1 if r["errors"] else 0)


if __name__ == "__main__":
    main()
