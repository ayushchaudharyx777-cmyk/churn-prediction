"""Live performance monitoring: join logged predictions with ground-truth labels that arrive later.
Labels come from the API (POST /feedback -> logs/labels.jsonl) or a CSV with columns: request_id, churned.
CLI: python performance.py --labels logs/labels.jsonl      (exit code 1 if AUC dropped too much)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from policy import ece


def _to01(v) -> int:
    return 1 if str(v).strip().lower() in ("1", "yes", "true", "churn") else 0


def load_labels(path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True) if str(path).endswith(".jsonl") else pd.read_csv(path)
    col = "churned" if "churned" in df.columns else "churn"
    df["churn"] = df[col].map(_to01)
    return df[["request_id", "churn"]].drop_duplicates("request_id", keep="last")


def live_performance(log: pd.DataFrame, labels: pd.DataFrame, baseline_auc: float,
                     min_rows: int = 100, max_auc_drop: float = 0.05) -> dict:
    m = log.merge(labels, on="request_id", how="inner")
    out = {"n_labeled": int(len(m)), "baseline_roc_auc": round(float(baseline_auc), 4)}
    if len(m) < min_rows or m["churn"].nunique() < 2:
        out["status"] = "insufficient_data"
        return out
    auc = float(roc_auc_score(m["churn"], m["churn_probability"]))
    out.update({
        "roc_auc": round(auc, 4),
        "pr_auc": round(float(average_precision_score(m["churn"], m["churn_probability"])), 4),
        "brier": round(float(brier_score_loss(m["churn"], m["churn_probability"])), 4),
        "ece": round(ece(m["churn"], m["churn_probability"]), 4),
        "mean_predicted": round(float(m["churn_probability"].mean()), 4),
        "actual_churn_rate": round(float(m["churn"].mean()), 4),
        "auc_drop": round(baseline_auc - auc, 4),
    })
    out["status"] = "degraded" if baseline_auc - auc > max_auc_drop else "ok"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/predictions.jsonl")
    ap.add_argument("--labels", default="logs/labels.jsonl")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out", default="reports/live_performance.json")
    a = ap.parse_args()
    meta = json.load(open(Path(a.model_dir) / "meta.json"))
    res = live_performance(pd.read_json(a.log, lines=True), load_labels(a.labels), meta["baseline_roc_auc"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    sys.exit(1 if res["status"] == "degraded" else 0)


if __name__ == "__main__":
    main()
