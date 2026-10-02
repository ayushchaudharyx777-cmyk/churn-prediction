"""Quality gate: block a new model that is clearly worse. Used by DVC (stage `gate`) and Airflow.
Exit code 1 = fail. Rules: AUC >= MIN_AUC, top-K lift >= MIN_LIFT, and no big drop vs the previous model."""
import json
import os
import sys

MIN_AUC = float(os.getenv("GATE_MIN_AUC", "0.78"))
MIN_LIFT, MAX_DROP = 1.5, 0.02
fm = json.load(open("reports/final_metrics.json"))
reg = json.load(open("models/registry.json"))
problems = []
if fm["roc_auc"] < MIN_AUC:
    problems.append(f"ROC-AUC {fm['roc_auc']} < {MIN_AUC}")
if fm["lift_at_capacity"] < MIN_LIFT:
    problems.append(f"lift@capacity {fm['lift_at_capacity']} < {MIN_LIFT}")
if len(reg) >= 2 and reg[-2]["roc_auc"] - fm["roc_auc"] > MAX_DROP:
    problems.append(f"AUC dropped {reg[-2]['roc_auc']} -> {fm['roc_auc']} (> {MAX_DROP})")
if problems:
    print("QUALITY GATE FAILED:", "; ".join(problems))
    sys.exit(1)
print(f"quality gate passed (AUC {fm['roc_auc']}, lift {fm['lift_at_capacity']})")
