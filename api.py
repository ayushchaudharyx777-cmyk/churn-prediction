"""FastAPI service. Run: uvicorn api:app --reload   -> docs at http://localhost:8000/docs
POST /predict   score a customer (logged to logs/predictions.jsonl for monitoring)
POST /feedback  report the real outcome later (logged to logs/labels.jsonl -> live performance monitoring)"""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from explain import env_mismatch, make_explainer
from features import add_features, is_actionable, risk_band, suggest_action

app = FastAPI(title="Churn Prediction API", version="2.0")
_STATE = {}


class Customer(BaseModel):
    # gender is intentionally NOT a model input (extra fields such as gender are ignored)
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=120)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Optional[float] = Field(default=None, ge=0)


class Prediction(BaseModel):
    request_id: str
    churn_probability: float
    risk_band: str
    in_call_list: bool
    high_risk: bool
    threshold: float
    capacity_cutoff: float
    top_drivers: List[str]
    actionable_reasons: List[str]
    suggested_action: str
    version: str
    model_name: str


class Feedback(BaseModel):
    request_id: str
    churned: bool


def get_state():
    d = os.getenv("MODEL_DIR", "models")
    if d not in _STATE:
        p = Path(d)
        if not (p / "model.joblib").exists():
            raise HTTPException(503, f"model not found in '{d}'. Run: python train.py")
        model = joblib.load(p / "model.joblib")
        meta = json.load(open(p / "meta.json"))
        names, fn = make_explainer(model, pd.read_csv(p / "reference.csv")[meta["columns"]])
        _STATE[d] = {"model": model, "meta": meta, "pre": model.named_steps["pre"], "names": names, "fn": fn}
    return _STATE[d]


def _append(path_env: str, default: str, record: dict):
    try:  # logging must never break serving
        path = Path(os.getenv(path_env, default))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


@app.get("/health")
def health():
    s = get_state()
    m = s["meta"]
    return {"status": "ok", "version": m["model_version"], "model_name": m["model_name"],
            "threshold": m["threshold"], "capacity_cutoff": m["capacity_cutoff"],
            "env_mismatch": env_mismatch(m.get("env"))}


@app.post("/predict", response_model=Prediction)
def predict(c: Customer):
    s = get_state()
    m = s["meta"]
    row = c.model_dump()
    for k, allowed in m["raw_cat"].items():
        if row.get(k) not in allowed:
            raise HTTPException(422, f"{k}: '{row.get(k)}' not in allowed values {allowed}")
    X = add_features(pd.DataFrame([row]))[m["columns"]]
    p = float(s["model"].predict_proba(X)[0, 1])
    contrib = pd.Series(s["fn"](s["pre"].transform(X))[0], index=s["names"]).sort_values(ascending=False)
    drivers = [n for n, v in contrib.items() if v > 0][:3]
    actionable = [n for n, v in contrib.items() if v > 0 and is_actionable(n)][:3]
    rid = str(uuid.uuid4())
    _append("PRED_LOG", "logs/predictions.jsonl",
            {"ts": datetime.now(timezone.utc).isoformat(), "request_id": rid, "version": m["model_version"],
             **row, "churn_probability": round(p, 6)})
    return Prediction(request_id=rid, churn_probability=round(p, 4),
                      risk_band=risk_band(p, m["threshold"], m["capacity_cutoff"]),
                      in_call_list=p >= m["capacity_cutoff"], high_risk=p >= m["threshold"],
                      threshold=m["threshold"], capacity_cutoff=m["capacity_cutoff"], top_drivers=drivers,
                      actionable_reasons=actionable, suggested_action=suggest_action(actionable),
                      version=m["model_version"], model_name=m["model_name"])


@app.post("/feedback")
def feedback(f: Feedback):
    _append("LABEL_LOG", "logs/labels.jsonl",
            {"ts": datetime.now(timezone.utc).isoformat(), "request_id": f.request_id, "churned": f.churned})
    return {"status": "recorded"}
