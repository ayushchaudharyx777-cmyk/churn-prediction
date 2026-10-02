"""Customer Churn Prediction - validate, tune, select, benchmark, decide, explain, audit, register.
Run: python train.py            (FAST=1 python train.py for a quick smoke run; SKIP_CHALLENGERS=1 to skip slow models)
Needs: data/WA_Fn-UseC_-Telco-Customer-Churn.csv      Settings: params.yaml
"""
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             average_precision_score, brier_score_loss, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from deep import TorchMLPClassifier
from explain import env_versions, make_explainer
from features import ENGINEERED, add_features, is_actionable, risk_band, suggest_action
from policy import (bootstrap_ci, breakeven_save_rate, ece, expected_random_cost, optimal_threshold,
                    paired_bootstrap_diff, topk_mask, topk_metrics, total_cost)
from validate import validate_training

for d in ("data", "reports", "models", "models/archive"):
    os.makedirs(d, exist_ok=True)

# ---------- settings (params.yaml, overridden by FAST=1) ----------
FAST = os.getenv("FAST") == "1"
P = {"seed": 42, "capacity": 0.20, "loss_per_churner": 500, "offer_cost": 50, "save_rate": 0.30,
     "n_iter": 25, "cv_splits": 5, "n_boot": 1000}
if Path("params.yaml").exists():
    try:
        import yaml
        P.update(yaml.safe_load(open("params.yaml")).get("train", {}))
    except ImportError:
        print("pyyaml not installed - using default settings")
if FAST:
    P.update(n_iter=3, cv_splits=3, n_boot=200)
SEED, CAPACITY, LOSS, OFFER, SAVE = P["seed"], P["capacity"], P["loss_per_churner"], P["offer_cost"], P["save_rate"]
N_ITER, N_SPLITS, N_BOOT = P["n_iter"], P["cv_splits"], P["n_boot"]
K_PCT = int(round(CAPACITY * 100))
DATA = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
EXCLUDE_FROM_MODEL = ["gender"]          # kept in the data ONLY for the fairness audit
VERSION = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "nogit"


def df_to_md(d):
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in d.astype(object).iterrows()]
    return "\n".join(lines)


# ---------- 1. Load + validate ----------
raw = pd.read_csv(DATA)
val = validate_training(raw)
json.dump(val, open("reports/data_validation.json", "w"), indent=2)
if val["errors"]:
    raise SystemExit(f"Data validation failed: {val['errors']}")
for w in val["warnings"]:
    print("WARNING:", w)

df = raw.copy()
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df = add_features(df)
ids = df["customerID"]
X = df.drop(columns=["Churn", "customerID", *EXCLUDE_FROM_MODEL])
y = df["Churn"]
X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
    X, y, ids, test_size=0.2, stratify=y, random_state=SEED)
y_arr, ytr_arr = y_te.values, y_tr.values
n_te, pos_te = len(y_arr), int(y_arr.sum())

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)


def make_pre(scale):
    num = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ], sparse_threshold=0)


def xgb_pipe(**kw):
    return Pipeline([("pre", make_pre(False)),
                     ("clf", XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1, **kw))])


# ---------- 2. Models (natural class distribution => probabilities stay calibrated) ----------
models = {
    "Logistic Regression": Pipeline([("pre", make_pre(True)), ("clf", LogisticRegression(max_iter=1000))]),
    "Random Forest": Pipeline([("pre", make_pre(False)),
        ("clf", RandomForestClassifier(n_estimators=100 if FAST else 400, min_samples_leaf=5,
                                       random_state=SEED, n_jobs=-1))]),
    "XGBoost (default)": xgb_pipe(),
}
challengers = {}                               # benchmark-only: not explainable per customer
if os.getenv("SKIP_CHALLENGERS") != "1":
    n_est = 50 if FAST else 200
    challengers["Stacking (LR+RF+XGB)"] = Pipeline([("pre", make_pre(True)), ("clf", StackingClassifier(
        [("lr", LogisticRegression(max_iter=1000)),
         ("rf", RandomForestClassifier(n_estimators=n_est, min_samples_leaf=5, random_state=SEED, n_jobs=-1)),
         ("xgb", XGBClassifier(n_estimators=n_est, max_depth=4, learning_rate=0.05, eval_metric="logloss",
                               random_state=SEED, n_jobs=1))],
        final_estimator=LogisticRegression(max_iter=1000), cv=3 if FAST else 5,
        stack_method="predict_proba", n_jobs=1))])
    if importlib.util.find_spec("torch") is not None:
        challengers["Neural net (PyTorch MLP)"] = Pipeline([("pre", make_pre(True)),
            ("clf", TorchMLPClassifier(epochs=15 if FAST else 60, random_state=SEED))])
    else:
        print("torch not installed - skipping the neural-net challenger (pip install torch to enable)")

# ---------- 3. Hyper-parameter tuning (CV on train only) ----------
print("Tuning XGBoost ...")
search = RandomizedSearchCV(
    xgb_pipe(),
    {"clf__n_estimators": [200, 300, 500], "clf__max_depth": [3, 4, 5, 6],
     "clf__learning_rate": [0.02, 0.05, 0.1], "clf__subsample": [0.7, 0.8, 1.0],
     "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__min_child_weight": [1, 3, 5],
     "clf__reg_lambda": [1, 5, 10]},
    n_iter=N_ITER, cv=cv, scoring="roc_auc", random_state=SEED, n_jobs=-1)
search.fit(X_tr, y_tr)
best_params = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
               for k, v in search.best_params_.items()}
json.dump(best_params, open("reports/best_params.json", "w"), indent=2)
models["XGBoost (tuned)"] = search.best_estimator_

# ---------- 4. Out-of-fold predictions + test evaluation for every model ----------
figs = {k: plt.subplots(figsize=(6.5, 5)) for k in ("roc", "pr_curve", "calibration")}
rows, fitted, oofs = [], {}, {}


def evaluate(name, pipe, oof):
    fitted[name], oofs[name] = pipe, oof
    p = pipe.predict_proba(X_te)[:, 1]
    t = topk_metrics(y_arr, p, CAPACITY)
    rows.append({"model": name, "cv_auc": roc_auc_score(ytr_arr, oof), "test_roc_auc": roc_auc_score(y_arr, p),
                 "pr_auc": average_precision_score(y_arr, p), "brier": brier_score_loss(y_arr, p),
                 "ece": ece(y_arr, p), "precision_at_k": t["precision"], "recall_at_k": t["recall"],
                 "lift_at_k": t["lift"]})
    RocCurveDisplay.from_estimator(pipe, X_te, y_te, ax=figs["roc"][1], name=name)
    PrecisionRecallDisplay.from_estimator(pipe, X_te, y_te, ax=figs["pr_curve"][1], name=name)
    CalibrationDisplay.from_estimator(pipe, X_te, y_te, ax=figs["calibration"][1], name=name, n_bins=10)


for name, pipe in {**models, **challengers}.items():
    print(f"Evaluating {name} ...")
    oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba",
                            n_jobs=-1 if name in models else 1)[:, 1]
    if name != "XGBoost (tuned)":
        pipe.fit(X_tr, y_tr)
    evaluate(name, pipe, oof)

titles = {"roc": "ROC curves", "pr_curve": "Precision-Recall curves", "calibration": "Calibration curves"}
for k, (fig, ax) in figs.items():
    ax.set_title(titles[k]); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"reports/{k}.png", dpi=150)
plt.close("all")
res = pd.DataFrame(rows).round(4).sort_values("cv_auc", ascending=False)
print(res.to_string(index=False))
res.to_csv("reports/metrics.csv", index=False)

# ---------- 5. Principled model selection (explainable models only, decided on TRAIN out-of-fold) ----------
d_mean, d_lo, d_hi = paired_bootstrap_diff(ytr_arr, oofs["XGBoost (tuned)"], oofs["Logistic Regression"],
                                           roc_auc_score, N_BOOT, SEED)
if d_lo > 0:
    final_name = "XGBoost (tuned)"
    selection = (f"XGBoost (tuned) beats Logistic Regression by {d_mean:+.4f} AUC "
                 f"(95% CI [{d_lo:+.4f}, {d_hi:+.4f}], out-of-fold on train) -> the gain is real, XGBoost deployed.")
else:
    final_name = "Logistic Regression"
    why = ("is significantly worse" if d_hi < 0 else "is not significantly better (CI includes 0)")
    selection = (f"XGBoost (tuned) minus Logistic Regression = {d_mean:+.4f} AUC (95% CI [{d_lo:+.4f}, {d_hi:+.4f}], "
                 f"out-of-fold on train): XGBoost {why} -> the simpler, directly interpretable Logistic Regression "
                 "is deployed.")
print("SELECTION:", selection)
final, oof = fitted[final_name], oofs[final_name]
p_te = final.predict_proba(X_te)[:, 1]

chal_rows = []
for name in list(fitted):
    if name == final_name:
        continue
    m, lo, hi = paired_bootstrap_diff(ytr_arr, oofs[name], oof, roc_auc_score, N_BOOT, SEED)
    verdict = ("significantly better" if lo > 0 else "significantly worse" if hi < 0 else "no significant difference")
    chal_rows.append({"model": name, "auc_diff_vs_deployed": round(m, 4), "ci_low": round(lo, 4),
                      "ci_high": round(hi, 4), "verdict": verdict,
                      "deployable": name in ("Logistic Regression", "XGBoost (tuned)")})
chal = pd.DataFrame(chal_rows)

# ---------- 6. Decision policy: cost-optimal threshold + capacity-constrained call list ----------
ths = np.round(np.arange(0.02, 0.96, 0.01), 2)
costs = [total_cost(ytr_arr, oof >= t, LOSS, OFFER, SAVE) for t in ths]
best_t = float(ths[int(np.argmin(costs))])
theo_t = optimal_threshold(LOSS, OFFER, SAVE)
cap_cut = float(np.quantile(oof, 1 - CAPACITY))
plt.figure(figsize=(6, 4))
plt.plot(ths, costs); plt.axvline(best_t, color="r", ls="--", label=f"empirical best = {best_t:.2f}")
plt.axvline(theo_t, color="g", ls=":", label=f"theory offer/(loss*save) = {theo_t:.2f}")
plt.xlabel("Threshold"); plt.ylabel("Total cost (train, out-of-fold)"); plt.legend()
plt.title("Cost vs decision threshold"); plt.tight_layout(); plt.savefig("reports/threshold_cost.png", dpi=150)
plt.close()

flag = p_te >= best_t
cap_mask = topk_mask(p_te, CAPACITY)
do_nothing = total_cost(y_arr, np.zeros(n_te, bool), LOSS, OFFER, SAVE)
policy_rows = []


def add_policy(name, cost, treated):
    policy_rows.append({"policy": name, "contacted": int(round(treated)), "cost": int(round(cost)),
                        "saving_vs_do_nothing": int(round(do_nothing - cost)),
                        "saving_pct": round(100 * (do_nothing - cost) / do_nothing, 1)})


add_policy("Do nothing", do_nothing, 0)
add_policy("Offer everyone", total_cost(y_arr, np.ones(n_te, bool), LOSS, OFFER, SAVE), n_te)
add_policy(f"Random {K_PCT}% of customers", expected_random_cost(y_arr, CAPACITY, LOSS, OFFER, SAVE), CAPACITY * n_te)
add_policy(f"Model: top {K_PCT}% (call list)", total_cost(y_arr, cap_mask, LOSS, OFFER, SAVE), cap_mask.sum())
add_policy(f"Model: cost-optimal threshold {best_t:.2f}", total_cost(y_arr, flag, LOSS, OFFER, SAVE), flag.sum())
policies = pd.DataFrame(policy_rows)

tk = topk_metrics(y_arr, p_te, CAPACITY)
be_save = breakeven_save_rate(tk["precision"], LOSS, OFFER)
topk_tbl = pd.DataFrame([{**topk_metrics(y_arr, p_te, k), "random_recall": k} for k in (0.05, 0.10, 0.20, 0.30, 0.50)])
topk_tbl = topk_tbl.round(3).drop(columns=["k"]).assign(contact_pct=[5, 10, 20, 30, 50])
topk_tbl["n_contacted"] = topk_tbl["n_contacted"].astype(int)

sens = []
for s in np.round(np.arange(0.05, 0.65, 0.05), 2):
    thr_s = optimal_threshold(LOSS, OFFER, s)
    c0 = total_cost(y_arr, np.zeros(n_te, bool), LOSS, OFFER, s)
    sens.append({"save_rate": s, "threshold": round(thr_s, 3),
                 "saving_top_k": round(c0 - total_cost(y_arr, cap_mask, LOSS, OFFER, s)),
                 "saving_threshold_policy": round(c0 - total_cost(y_arr, p_te >= thr_s, LOSS, OFFER, s))})
sens = pd.DataFrame(sens)
sens.to_csv("reports/sensitivity.csv", index=False)
plt.figure(figsize=(6.5, 4))
plt.plot(sens.save_rate, sens.saving_top_k, label=f"Top {K_PCT}% call list")
plt.plot(sens.save_rate, sens.saving_threshold_policy, label="Cost-optimal threshold")
plt.axhline(0, color="k", lw=0.8); plt.axvline(SAVE, color="grey", ls=":", label=f"assumed save rate {SAVE}")
plt.xlabel("Offer success rate (share of contacted churners retained)"); plt.ylabel("Net saving vs doing nothing")
plt.title("Is the campaign worth it? Sensitivity"); plt.legend(); plt.tight_layout()
plt.savefig("reports/sensitivity.png", dpi=150); plt.close()

# ---------- 7. Test metrics with bootstrap confidence intervals ----------
ci_auc = bootstrap_ci(y_arr, p_te, roc_auc_score, N_BOOT, SEED)
ci_pr = bootstrap_ci(y_arr, p_te, average_precision_score, N_BOOT, SEED)
ci_prec = bootstrap_ci(y_arr, p_te, lambda a, b: topk_metrics(a, b, CAPACITY)["precision"], N_BOOT, SEED)
ci_rec = bootstrap_ci(y_arr, p_te, lambda a, b: topk_metrics(a, b, CAPACITY)["recall"], N_BOOT, SEED)
cap_row = policies.iloc[3]
final_metrics = {
    "version": VERSION, "deployed_model": final_name, "selection_reason": selection,
    "threshold": best_t, "theoretical_threshold": round(theo_t, 4), "capacity": CAPACITY,
    "capacity_cutoff": round(cap_cut, 4),
    "roc_auc": round(roc_auc_score(y_arr, p_te), 4), "roc_auc_ci": [round(ci_auc[0], 4), round(ci_auc[1], 4)],
    "pr_auc": round(average_precision_score(y_arr, p_te), 4), "pr_auc_ci": [round(ci_pr[0], 4), round(ci_pr[1], 4)],
    "brier": round(brier_score_loss(y_arr, p_te), 4), "ece": round(ece(y_arr, p_te), 4),
    "precision_at_capacity": round(tk["precision"], 4), "precision_at_capacity_ci": [round(ci_prec[0], 4), round(ci_prec[1], 4)],
    "recall_at_capacity": round(tk["recall"], 4), "recall_at_capacity_ci": [round(ci_rec[0], 4), round(ci_rec[1], 4)],
    "lift_at_capacity": round(tk["lift"], 3), "base_churn_rate": round(pos_te / n_te, 4),
    "threshold_policy": {"flagged_share": round(float(flag.mean()), 4),
                         "precision": round(float(precision_score(y_arr, flag)), 4),
                         "recall": round(float(recall_score(y_arr, flag)), 4),
                         "f1": round(float(f1_score(y_arr, flag)), 4)},
    "breakeven_save_rate": round(be_save, 3),
    "saving_capacity_policy": int(cap_row["saving_vs_do_nothing"]),
    "saving_threshold_policy": int(policies.iloc[4]["saving_vs_do_nothing"]),
    "policies": policy_rows,
    "challengers": chal_rows,
    "assumptions": {"loss_per_churner": LOSS, "offer_cost": OFFER, "save_rate": SAVE, "capacity": CAPACITY},
}
json.dump(final_metrics, open("reports/final_metrics.json", "w"), indent=2)
print(json.dumps({k: v for k, v in final_metrics.items() if k not in ("policies", "challengers")}, indent=2))

ConfusionMatrixDisplay.from_predictions(y_te, flag.astype(int), display_labels=["Stay", "Churn"])
plt.title(f"Confusion matrix @ cost-optimal threshold {best_t:.2f}"); plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150); plt.close()

order = np.argsort(-p_te, kind="stable")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(np.arange(1, n_te + 1) / n_te, np.cumsum(y_arr[order]) / pos_te, label="Model")
a1.plot([0, 1], [0, 1], "--", label="Random"); a1.axvline(CAPACITY, color="grey", ls=":")
a1.set_xlabel("Share of customers contacted"); a1.set_ylabel("Share of churners reached")
a1.set_title("Cumulative gain"); a1.legend()
a2.bar(range(1, 11), [y_arr[d].mean() / y_arr.mean() for d in np.array_split(order, 10)])
a2.axhline(1, color="k", lw=0.8); a2.set_xlabel("Risk decile (1 = highest)"); a2.set_ylabel("Lift")
a2.set_title("Lift by decile"); fig.tight_layout(); fig.savefig("reports/gain_lift.png", dpi=150); plt.close()

# ---------- 8. Explanations: all drivers + ACTIONABLE reasons -> retention action ----------
names, contrib_fn = make_explainer(final, X_tr)
Xt = final.named_steps["pre"].transform(X_te)
sv = contrib_fn(Xt)
plt.figure()
shap.summary_plot(sv, Xt, feature_names=names, plot_type="bar", show=False, max_display=12)
plt.tight_layout(); plt.savefig("reports/shap_top_features.png", dpi=150); plt.close()
imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
act_flags = np.array([is_actionable(n) for n in names])


def reasons(row, k=3, only_actionable=False):
    return [names[i] for i in np.argsort(row)[::-1] if row[i] > 0 and (act_flags[i] or not only_actionable)][:k]


drv = [reasons(r) for r in sv]
act = [reasons(r, only_actionable=True) for r in sv]
risk = pd.DataFrame({
    "customerID": id_te.values, "churn_probability": p_te.round(4),
    "risk_band": [risk_band(p, best_t, cap_cut) for p in p_te],
    "top_drivers": [", ".join(r) for r in drv], "actionable_reasons": [", ".join(r) for r in act],
    "suggested_action": [suggest_action(r) for r in act], "actual_churn": y_arr,
}).sort_values("churn_probability", ascending=False)
risk.to_csv("reports/high_risk_customers.csv", index=False)

# ---------- 9. Fairness / slice audit (gender is NOT a model input, but we still audit it) ----------
test_df = df.loc[X_te.index]
fair_rows = []
for attr in ("gender", "SeniorCitizen"):
    vals = test_df[attr].to_numpy()
    for v in sorted(pd.unique(vals)):
        m = vals == v
        yt, pt, ft = y_arr[m], p_te[m], flag[m]
        tp = int((ft & (yt == 1)).sum())
        fair_rows.append({"attribute": attr, "group": str(v), "n": int(m.sum()),
                          "churn_rate": round(float(yt.mean()), 4), "flag_rate": round(float(ft.mean()), 4),
                          "recall": round(tp / max(int(yt.sum()), 1), 4),
                          "precision": round(tp / max(int(ft.sum()), 1), 4),
                          "auc": round(float(roc_auc_score(yt, pt)), 4) if len(set(yt)) > 1 else None})
fair = pd.DataFrame(fair_rows)
fair.to_csv("reports/fairness_slices.csv", index=False)
fair_summary = []
for attr, g in fair.groupby("attribute"):
    fair_summary.append({"attribute": attr,
                         "flag_rate_ratio_min_over_max": round(float(g.flag_rate.min() / max(g.flag_rate.max(), 1e-9)), 3),
                         "recall_gap": round(float(g.recall.max() - g.recall.min()), 3)})
json.dump(fair_summary, open("reports/fairness_summary.json", "w"), indent=2)

# ---------- 10. Save model, metadata, reference data, registry ----------
commit = git_commit()
joblib.dump(final, "models/model.joblib")
joblib.dump(final, f"models/archive/{VERSION}.joblib")
raw_x = X.drop(columns=ENGINEERED)
meta = {"model_version": VERSION, "model_name": final_name, "git_commit": commit,
        "trained_at": datetime.now(timezone.utc).isoformat(), "threshold": best_t, "capacity": CAPACITY,
        "capacity_cutoff": round(cap_cut, 6), "baseline_roc_auc": final_metrics["roc_auc"],
        "columns": X.columns.tolist(), "excluded_features": EXCLUDE_FROM_MODEL, "env": env_versions(),
        "raw_numeric": {}, "raw_cat": {}}
for c in raw_x.columns:
    if not pd.api.types.is_numeric_dtype(raw_x[c]) or raw_x[c].nunique() <= 2:
        meta["raw_cat"][c] = sorted(raw_x[c].dropna().unique().tolist())
    else:
        meta["raw_numeric"][c] = [float(raw_x[c].min()), float(raw_x[c].max()), float(raw_x[c].median()),
                                  bool((raw_x[c].dropna() % 1 == 0).all())]
json.dump(meta, open("models/meta.json", "w"), indent=2)
ref = X_tr.copy()
ref["churn_probability"] = oof            # out-of-fold scores = honest reference score distribution
ref.sample(min(2000, len(ref)), random_state=SEED).to_csv("models/reference.csv", index=False)
reg_path = Path("models/registry.json")
registry = json.load(open(reg_path)) if reg_path.exists() else []
registry.append({"version": VERSION, "model": final_name, "git_commit": commit, "roc_auc": final_metrics["roc_auc"],
                 "lift_at_capacity": final_metrics["lift_at_capacity"], "threshold": best_t, "n_train": int(len(X_tr))})
json.dump(registry, open(reg_path, "w"), indent=2)

# ---------- 11. Auto-generated RESULTS.md, ONE_PAGER.md, MODEL_CARD.md ----------
top_drivers = "\n".join(f"- `{n}` ({v:.3f})" for n, v in imp.head(8).items())
top_actionable = "\n".join(f"- `{n}` ({v:.3f})" for n, v in imp[[is_actionable(n) for n in imp.index]].head(5).items())
ps = final_metrics["policies"]
cap_name = f"Model: top {K_PCT}% (call list)"
rand_saving = next(r["saving_vs_do_nothing"] for r in ps if r["policy"].startswith("Random"))
open("reports/RESULTS.md", "w").write(f"""# Results (auto-generated by train.py, version {VERSION})

Deployed model: **{final_name}**. {selection}

## Model comparison (test set; cv_auc = pooled {N_SPLITS}-fold out-of-fold AUC on train)
Ranking metrics @ top {K_PCT}% contacted. No class re-weighting, so probabilities are calibrated (see brier / ece).
{df_to_md(res)}

## Challengers vs deployed model (paired bootstrap on train out-of-fold AUC)
{df_to_md(chal) if len(chal) else '(none)'}
Neural-net / stacking models are benchmarks: they cannot give per-customer reasons, so they are only
deployable if a model-agnostic explainer is added.

## Deployed model on the held-out test set (95% bootstrap CI)
- ROC-AUC {final_metrics['roc_auc']} {final_metrics['roc_auc_ci']}, PR-AUC {final_metrics['pr_auc']} {final_metrics['pr_auc_ci']}
- Calibration: Brier {final_metrics['brier']}, ECE {final_metrics['ece']}
- Contacting the top {K_PCT}% reaches {final_metrics['recall_at_capacity']:.1%} of churners {final_metrics['recall_at_capacity_ci']}
  with precision {final_metrics['precision_at_capacity']:.1%} (base churn rate {final_metrics['base_churn_rate']:.1%}) -> lift {final_metrics['lift_at_capacity']}x

## Ranking quality by contact budget
{df_to_md(topk_tbl)}

## Business impact (assumed: churner loses {LOSS}, offer costs {OFFER}, offer keeps {SAVE:.0%} of contacted churners)
{df_to_md(policies)}

- Contact-worthy threshold in theory = offer/(loss*save) = {theo_t:.2f}; empirical (train out-of-fold) = {best_t:.2f}.
- Break-even: the top-{K_PCT}% campaign pays off only if the offer success rate is above **{be_save:.0%}**.
- Random targeting of {K_PCT}% saves {rand_saving:,}; the model call list saves {final_metrics['saving_capacity_policy']:,}.
- See `sensitivity.png` for savings at other success rates.

## Top churn drivers (mean |contribution|)
{top_drivers}

## Top ACTIONABLE drivers (the business can change these)
{top_actionable}

## Fairness / slice audit (cost-optimal threshold policy; gender is not a model input)
{df_to_md(fair)}

Disparity summary: `{json.dumps(fair_summary)}`

## Best XGBoost hyper-parameters
`{json.dumps(best_params)}`
""")
open("reports/ONE_PAGER.md", "w").write(f"""# Churn prediction - one page

**Problem.** A telecom wants to spend a limited retention budget on customers who are about to leave.

**Approach.** Validated data -> leakage-safe sklearn pipelines -> Logistic Regression, Random Forest, XGBoost
(tuned) plus stacking/neural-net benchmarks -> model chosen by out-of-fold bootstrap, not by guesswork ->
decision policy from campaign economics -> per-customer reasons mapped to retention actions -> API, dashboard,
drift + performance monitoring, CI/CD.

**Result.** Deployed **{final_name}**: ROC-AUC {final_metrics['roc_auc']} {final_metrics['roc_auc_ci']}.
Calling the top {K_PCT}% of customers reaches {final_metrics['recall_at_capacity']:.0%} of churners
({final_metrics['lift_at_capacity']}x better than random).

**Business impact (assumptions, not facts).** With loss {LOSS}, offer {OFFER}, success rate {SAVE:.0%}:
top-{K_PCT}% campaign saves {final_metrics['saving_capacity_policy']:,} ({cap_row['saving_pct']}% of the no-campaign loss);
random targeting of the same size saves only {rand_saving:,}. Break-even success rate: {be_save:.0%}.

**Honest findings.** {selection} Calibration ECE {final_metrics['ece']}.

**Limitations.** Static snapshot (no time split); cost and success-rate values are assumptions; reasons are
correlational; retention actions are rules, not learned uplift; fairness audit covers gender and senior-citizen only.
""")
open("reports/MODEL_CARD.md", "w").write(f"""# Model card - Churn predictor ({VERSION})

- **Model:** {final_name} in a sklearn Pipeline (impute + scale/encode). Selected by paired bootstrap on out-of-fold AUC.
- **Intended use:** rank existing telecom customers to prioritise retention outreach under a contact budget.
- **Not for:** automated adverse decisions, credit/eligibility decisions, other industries without retraining.
- **Data:** Telco Customer Churn (IBM sample), {val['n_rows']:,} rows, churn rate {val['churn_rate']:.1%}.
- **Features:** demographics except gender (excluded from the model, audited only), services, contract, billing, tenure.
- **Metrics (held-out test):** ROC-AUC {final_metrics['roc_auc']} {final_metrics['roc_auc_ci']}, ECE {final_metrics['ece']},
  recall@top{K_PCT}% {final_metrics['recall_at_capacity']}, lift {final_metrics['lift_at_capacity']}x.
- **Decision policy:** call list = top {K_PCT}% by score; cost-optimal threshold {best_t:.2f} for automated offers.
- **Fairness audit:** `reports/fairness_slices.csv` (gender, SeniorCitizen). Group differences in flag rate may reflect real
  churn differences; review before any deployment.
- **Limitations:** static data, assumed economics, correlational explanations, rule-based actions.
- **Monitoring:** PSI/KS drift (`monitor.py`), live AUC once labels arrive (`performance.py`, POST /feedback).
""")

# ---------- 12. Experiment tracking (MLflow) ----------
if os.getenv("DISABLE_MLFLOW") != "1":
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment("churn-prediction")
        with mlflow.start_run(run_name=VERSION):
            mlflow.log_params({**best_params, **P, "deployed_model": final_name, "git_commit": commit})
            mlflow.log_metrics({k: final_metrics[k] for k in (
                "threshold", "roc_auc", "pr_auc", "brier", "ece", "precision_at_capacity", "recall_at_capacity",
                "lift_at_capacity", "breakeven_save_rate", "saving_capacity_policy", "saving_threshold_policy")})
            mlflow.log_artifacts("reports", artifact_path="reports")
            mlflow.log_artifact("models/meta.json", artifact_path="model")
            mlflow.log_artifact(f"models/archive/{VERSION}.joblib", artifact_path="model")
        print("MLflow run logged. View: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    except ImportError:
        print("mlflow not installed - skipping experiment tracking")
print(f"Done (version {VERSION}). Next: streamlit run app.py | uvicorn api:app | python monitor.py --simulate")
