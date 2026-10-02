"""Streamlit dashboard. Run: streamlit run app.py   (after python train.py)"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from explain import env_mismatch, make_explainer
from features import ENGINEERED, add_features, is_actionable, risk_band, suggest_action
from monitor import (drift_report, load_current, make_drift_batch, monitored_columns, prepare_current,
                     summarize)
from performance import live_performance, load_labels
from policy import breakeven_save_rate, topk_mask, topk_metrics, total_cost

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("Customer Churn Prediction")


@st.cache_resource
def load():
    model = joblib.load("models/model.joblib")
    meta = json.load(open("models/meta.json"))
    ref = pd.read_csv("models/reference.csv")
    names, fn = make_explainer(model, ref[meta["columns"]])
    return model, meta, ref, names, fn


model, meta, ref, names, contrib_fn = load()
fm = json.load(open("reports/final_metrics.json"))
A = fm["assumptions"]
st.caption(f"Model: {meta['model_name']} | version {meta['model_version']} | git {meta['git_commit']} | "
           f"cost threshold {meta['threshold']} | call-list cutoff {meta['capacity_cutoff']}")
mism = env_mismatch(meta.get("env"))
if mism:
    st.warning(f"Library versions differ from training (model may behave differently): {mism}. "
               "Run `python scripts/pin_requirements.py` before deploying.")
tabs = st.tabs(["Overview", "Model comparison", "Call list", "Predict a customer", "Monitoring", "Fairness"])

with tabs[0]:
    c = st.columns(5)
    c[0].metric("ROC-AUC", fm["roc_auc"], help=f"95% CI {fm['roc_auc_ci']}")
    c[1].metric(f"Churners reached (top {int(A['capacity'] * 100)}%)", f"{fm['recall_at_capacity']:.0%}")
    c[2].metric("Lift vs random", f"{fm['lift_at_capacity']}x")
    c[3].metric("Net saving (call list)", f"{fm['saving_capacity_policy']:,}")
    c[4].metric("Break-even offer success", f"{fm['breakeven_save_rate']:.0%}")
    st.caption(f"ASSUMPTIONS: churner loses {A['loss_per_churner']}, offer costs {A['offer_cost']}, "
               f"offer keeps {A['save_rate']:.0%} of contacted churners. {fm['selection_reason']}")
    st.subheader("Policies compared on the test set")
    st.dataframe(pd.DataFrame(fm["policies"]), width="stretch")
    a, b = st.columns(2)
    a.image("reports/gain_lift.png", caption="Cumulative gain and lift by decile")
    b.image("reports/sensitivity.png", caption="Net saving vs assumed offer success rate")
    a, b = st.columns(2)
    a.image("reports/shap_top_features.png", caption="Top churn drivers")
    b.image("reports/threshold_cost.png", caption="Cost vs threshold (empirical vs theory)")

with tabs[1]:
    st.dataframe(pd.read_csv("reports/metrics.csv"), width="stretch")
    st.subheader("Challengers vs deployed model")
    st.dataframe(pd.DataFrame(fm["challengers"]), width="stretch")
    a, b, c = st.columns(3)
    a.image("reports/roc.png"); b.image("reports/pr_curve.png"); c.image("reports/calibration.png")
    if Path("models/registry.json").exists():
        st.subheader("Model registry (every training run)")
        st.dataframe(pd.DataFrame(json.load(open("models/registry.json"))), width="stretch")

with tabs[2]:
    risk = pd.read_csv("reports/high_risk_customers.csv")
    c1, c2 = st.columns(2)
    k = c1.slider("Call capacity (% of customers contacted)", 1, 50, int(A["capacity"] * 100)) / 100
    save = c2.slider("Assumed offer success rate", 0.05, 0.80, float(A["save_rate"]), 0.05)
    y, p = risk["actual_churn"].values, risk["churn_probability"].values
    mask, m = topk_mask(p, k), topk_metrics(risk["actual_churn"].values, p, k)
    saving = (total_cost(y, np.zeros(len(y), bool), A["loss_per_churner"], A["offer_cost"], save)
              - total_cost(y, mask, A["loss_per_churner"], A["offer_cost"], save))
    c = st.columns(5)
    c[0].metric("Customers contacted", m["n_contacted"]); c[1].metric("Churners reached", f"{m['recall']:.0%}")
    c[2].metric("Precision", f"{m['precision']:.0%}"); c[3].metric("Lift", f"{m['lift']:.2f}x")
    c[4].metric("Net saving", f"{saving:,.0f}",
                help=f"Break-even success rate {breakeven_save_rate(m['precision'], A['loss_per_churner'], A['offer_cost']):.0%}")
    view = risk[mask].drop(columns=["actual_churn"])
    st.dataframe(view.head(300), width="stretch")
    st.download_button("Download call list (CSV)", view.to_csv(index=False), "call_list.csv")

with tabs[3]:
    st.write("Enter customer details to get a churn score with reasons.")
    row, cols, i = {}, st.columns(3), 0
    for col in meta["columns"]:
        if col in ENGINEERED:
            continue
        w = cols[i % 3]; i += 1
        if col in meta["raw_cat"]:
            row[col] = w.selectbox(col, meta["raw_cat"][col], key=col)
        else:
            lo, hi, med, is_int = meta["raw_numeric"][col]
            cast = int if is_int else float
            row[col] = w.number_input(col, min_value=cast(lo), max_value=cast(hi), value=cast(med), key=col)
    if st.button("Predict churn risk"):
        X = add_features(pd.DataFrame([row]))[meta["columns"]]
        p1 = float(model.predict_proba(X)[0, 1])
        band = risk_band(p1, meta["threshold"], meta["capacity_cutoff"])
        st.metric("Churn probability", f"{p1:.1%}")
        {"Critical": st.error, "High": st.warning}.get(band, st.success)(f"Risk band: {band}")
        s = pd.Series(contrib_fn(model.named_steps["pre"].transform(X))[0], index=names)
        st.bar_chart(s.reindex(s.abs().sort_values(ascending=False).index[:8]))
        act = [n for n, v in s.sort_values(ascending=False).items() if v > 0 and is_actionable(n)][:3]
        st.info(f"Actionable reasons: {', '.join(act) or 'none'} -> {suggest_action(act)}")

with tabs[4]:
    st.subheader("Data & prediction drift (PSI: <0.10 ok, 0.10-0.25 warning, >0.25 alert)")
    src = st.radio("Compare training reference with", ["Simulated drift (demo)", "Upload CSV (Telco format)",
                                                       "API prediction log"], horizontal=True)
    cur = None
    if src.startswith("Simulated"):
        cur = make_drift_batch(ref)
    elif src.startswith("Upload"):
        up = st.file_uploader("Upload CSV with the raw Telco columns", type="csv")
        cur = pd.read_csv(up) if up else None
    elif Path("logs/predictions.jsonl").exists():
        cur = load_current("logs/predictions.jsonl")
    else:
        st.info("No API log yet. Start the API (uvicorn api:app) and call /predict first.")
    if cur is not None:
        try:
            cur = prepare_current(cur)
            rep = drift_report(ref, cur, monitored_columns())
            s = summarize(rep, len(cur))
            if s["overall"] == "insufficient_data":
                st.info(f"Only {s['n_current']} rows so far - need at least 30 before drift numbers mean anything.")
            banner = {"ok": st.success, "warning": st.warning, "alert": st.error}.get(s["overall"], st.info)
            banner(f"Overall: {s['overall'].upper()} | {s['n_alert']} alert, {s['n_warning']} warning "
                   f"| worst: {s['worst_feature']} (PSI {s['max_psi']}) | rows: {s['n_current']}")
            m1, m2 = st.columns(2)
            m1.metric("Mean churn score (reference)", f"{ref['churn_probability'].mean():.3f}")
            m2.metric("Mean churn score (current)", f"{cur['churn_probability'].mean():.3f}")
            st.bar_chart(rep.set_index("feature")["psi"])
            st.dataframe(rep, width="stretch")
        except ValueError as e:
            st.error(str(e))
    st.subheader("Live model performance (needs real outcomes)")
    labels = load_labels("logs/labels.jsonl") if Path("logs/labels.jsonl").exists() else None
    up2 = st.file_uploader("Or upload labels CSV (columns: request_id, churned)", type="csv", key="labels")
    if up2:
        labels = load_labels(up2)
    if Path("logs/predictions.jsonl").exists() and labels is not None:
        res = live_performance(load_current("logs/predictions.jsonl"), labels, meta["baseline_roc_auc"])
        {"ok": st.success, "degraded": st.error}.get(res["status"], st.info)(f"Live status: {res['status']}")
        st.json(res)
    else:
        st.info("Send outcomes with POST /feedback {request_id, churned} (or upload a CSV) to track live AUC.")

with tabs[5]:
    st.write("Gender is NOT a model input; we still audit outcomes by gender and senior-citizen status "
             "(cost-optimal threshold policy). Rate differences can reflect real churn differences - review, don't assume.")
    st.dataframe(pd.read_csv("reports/fairness_slices.csv"), width="stretch")
    st.json(json.load(open("reports/fairness_summary.json")))
