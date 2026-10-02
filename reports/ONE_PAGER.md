# Churn prediction - one page

**Problem.** A telecom wants to spend a limited retention budget on customers who are about to leave.

**Approach.** Validated data -> leakage-safe sklearn pipelines -> Logistic Regression, Random Forest, XGBoost
(tuned) plus stacking/neural-net benchmarks -> model chosen by out-of-fold bootstrap, not by guesswork ->
decision policy from campaign economics -> per-customer reasons mapped to retention actions -> API, dashboard,
drift + performance monitoring, CI/CD.

**Result.** Deployed **Logistic Regression**: ROC-AUC 0.8476 [0.8252, 0.8676].
Calling the top 20% of customers reaches 52% of churners
(2.592x better than random).

**Business impact (assumptions, not facts).** With loss 500, offer 50, success rate 30%:
top-20% campaign saves 15,000 (8.0% of the no-campaign loss);
random targeting of the same size saves only -2,870. Break-even success rate: 15%.

**Honest findings.** XGBoost (tuned) minus Logistic Regression = -0.0002 AUC (95% CI [-0.0032, +0.0028], out-of-fold on train): XGBoost is not significantly better (CI includes 0) -> the simpler, directly interpretable Logistic Regression is deployed. Calibration ECE 0.0231.

**Limitations.** Static snapshot (no time split); cost and success-rate values are assumptions; reasons are
correlational; retention actions are rules, not learned uplift; fairness audit covers gender and senior-citizen only.
