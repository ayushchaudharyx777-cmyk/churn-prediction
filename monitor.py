"""Drift monitoring (PSI + KS) for input features and the predicted churn score.

PSI < 0.10 ok | 0.10-0.25 warning | > 0.25 alert.
CLI:
  python monitor.py --current logs/predictions.jsonl     # compare API traffic with training reference
  python monitor.py --current new_customers.csv
  python monitor.py --simulate                           # demo with artificially drifted data
Exit code 1 on alert (so it can run in cron / CI).
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from features import ENGINEERED, add_features

EPS, WARN, ALERT, MIN_ROWS = 1e-4, 0.10, 0.25, 30


def _psi(r, c):
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref, cur) -> float:
    ref, cur = pd.Series(ref).dropna().astype(str), pd.Series(cur).dropna().astype(str)
    if ref.empty or cur.empty:
        return float("nan")
    cats = sorted(set(ref) | set(cur))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(r, c)


def psi_numeric(ref, cur, bins: int = 10) -> float:
    ref = pd.Series(ref).dropna().to_numpy(float)
    cur = pd.Series(cur).dropna().to_numpy(float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                      # (almost) constant feature
        return psi_categorical(ref, cur)
    inner = edges[1:-1]
    r = np.bincount(np.digitize(ref, inner), minlength=len(edges) - 1) / len(ref)
    c = np.bincount(np.digitize(cur, inner), minlength=len(edges) - 1) / len(cur)
    return _psi(r, c)


def _status(psi: float) -> str:
    if np.isnan(psi):
        return "n/a"
    return "alert" if psi >= ALERT else "warning" if psi >= WARN else "ok"


def _as_cat(s):
    return pd.to_numeric(s, errors="coerce").astype(float) if pd.api.types.is_numeric_dtype(s) else s


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, columns) -> pd.DataFrame:
    rows = []
    for c in columns:
        if c not in ref or c not in cur:
            continue
        r, k = ref[c], cur[c]
        if pd.api.types.is_numeric_dtype(r) and r.nunique() > 10:
            k = pd.to_numeric(k, errors="coerce")
            psi = psi_numeric(r, k)
            ks_p = float(ks_2samp(r.dropna(), k.dropna()).pvalue)
            kind, rm, cm = "numeric", float(r.mean()), float(k.mean())
        else:
            psi = psi_categorical(_as_cat(r), _as_cat(k))
            kind, ks_p, rm, cm = "categorical", np.nan, np.nan, np.nan
        rows.append({"feature": c, "kind": kind, "psi": round(psi, 4), "ks_pvalue": round(ks_p, 4),
                     "ref_mean": round(rm, 3), "cur_mean": round(cm, 3), "status": _status(psi)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def summarize(report: pd.DataFrame, n_current: int) -> dict:
    if n_current < MIN_ROWS:
        overall = "insufficient_data"
    elif (report["status"] == "alert").any():
        overall = "alert"
    elif (report["status"] == "warning").any():
        overall = "warning"
    else:
        overall = "ok"
    top = report.iloc[0] if len(report) else None
    enough = n_current >= MIN_ROWS      # PSI on a handful of rows is noise: do not count alerts
    return {"overall": overall, "n_current": int(n_current),
            "n_alert": int((report["status"] == "alert").sum()) if enough else 0,
            "n_warning": int((report["status"] == "warning").sum()) if enough else 0,
            "worst_feature": None if top is None else top["feature"],
            "max_psi": None if top is None else float(top["psi"])}


def simulate_drift(raw: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Make a drifted copy of RAW customer data: newer, pricier, more month-to-month customers."""
    rng = np.random.default_rng(seed)
    d = raw.copy()
    d["tenure"] = (d["tenure"] * 0.5).round().astype(int)
    d["MonthlyCharges"] = (d["MonthlyCharges"] * 1.25).round(2)
    d["TotalCharges"] = (pd.to_numeric(d["TotalCharges"], errors="coerce") * 0.6).round(2)
    flip = rng.random(len(d)) < 0.7
    d.loc[flip, "Contract"] = "Month-to-month"
    return d


def make_drift_batch(ref: pd.DataFrame, n: int = 800, seed: int = 1) -> pd.DataFrame:
    """RAW drifted batch built from the training reference (for demos and the scheduled-monitor workflow)."""
    base = ref.drop(columns=["churn_probability", *ENGINEERED]).sample(min(n, len(ref)), random_state=seed)
    return simulate_drift(base)


def prepare_current(df: pd.DataFrame, model_dir="models") -> pd.DataFrame:
    """Add engineered features + churn score so new data is comparable with the reference."""
    model_dir = Path(model_dir)
    meta = json.load(open(model_dir / "meta.json"))
    missing = [c for c in meta["columns"] if c not in df.columns and c not in ENGINEERED]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = add_features(df)
    if "churn_probability" not in df.columns:
        model = joblib.load(model_dir / "model.joblib")
        df["churn_probability"] = model.predict_proba(df[meta["columns"]])[:, 1]
    return df


def load_current(path) -> pd.DataFrame:
    path = str(path)
    return pd.read_json(path, lines=True) if path.endswith(".jsonl") else pd.read_csv(path)


def monitored_columns(model_dir="models"):
    meta = json.load(open(Path(model_dir) / "meta.json"))
    return meta["columns"] + ["churn_probability"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", help="CSV (raw customer format) or API log .jsonl")
    ap.add_argument("--simulate", action="store_true", help="demo using artificially drifted data")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out", default="reports/drift_report.csv")
    ap.add_argument("--save-batch", help="with --simulate: also write the drifted RAW batch to this CSV")
    a = ap.parse_args()

    ref = pd.read_csv(Path(a.model_dir) / "reference.csv")
    if a.simulate:
        cur = make_drift_batch(ref)
        if a.save_batch:
            Path(a.save_batch).parent.mkdir(parents=True, exist_ok=True)
            cur.to_csv(a.save_batch, index=False)
    elif a.current:
        cur = load_current(a.current)
    else:
        ap.error("give --current or --simulate")
    cur = prepare_current(cur, a.model_dir)
    rep = drift_report(ref, cur, monitored_columns(a.model_dir))
    s = summarize(rep, len(cur))
    s["mean_score_ref"] = round(float(ref["churn_probability"].mean()), 4)
    s["mean_score_current"] = round(float(cur["churn_probability"].mean()), 4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(a.out, index=False)
    json.dump(s, open(str(Path(a.out).with_suffix(".json")), "w"), indent=2)
    print(rep.head(12).to_string(index=False))
    print(json.dumps(s, indent=2))
    sys.exit(1 if s["overall"] == "alert" else 0)


if __name__ == "__main__":
    main()
