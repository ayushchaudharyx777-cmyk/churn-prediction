# Model card - Churn predictor (20261002-202417)

- **Model:** Logistic Regression in a sklearn Pipeline (impute + scale/encode). Selected by paired bootstrap on out-of-fold AUC.
- **Intended use:** rank existing telecom customers to prioritise retention outreach under a contact budget.
- **Not for:** automated adverse decisions, credit/eligibility decisions, other industries without retraining.
- **Data:** Telco Customer Churn (IBM sample), 7,043 rows, churn rate 26.5%.
- **Features:** demographics except gender (excluded from the model, audited only), services, contract, billing, tenure.
- **Metrics (held-out test):** ROC-AUC 0.8476 [0.8252, 0.8676], ECE 0.0231,
  recall@top20% 0.5187, lift 2.592x.
- **Decision policy:** call list = top 20% by score; cost-optimal threshold 0.38 for automated offers.
- **Fairness audit:** `reports/fairness_slices.csv` (gender, SeniorCitizen). Group differences in flag rate may reflect real
  churn differences; review before any deployment.
- **Limitations:** static data, assumed economics, correlational explanations, rule-based actions.
- **Monitoring:** PSI/KS drift (`monitor.py`), live AUC once labels arrive (`performance.py`, POST /feedback).
