# Interview prep - be able to defend every part

Rule: only claim what the repo really does. These are the honest answers.

## Core
**Why did you deploy Logistic Regression and not XGBoost?**  
Out-of-fold paired bootstrap on train showed XGBoost was not better (see `reports/RESULTS.md`, "Honest findings").
Equal accuracy -> choose the simpler, faster, directly interpretable model. The test set was never used to choose.

**Why not just use accuracy / threshold 0.5?**  
Churn is ~26% of customers and the costs are asymmetric. I compare models by ranking metrics (AUC, lift@top-K),
check calibration (Brier, ECE), and decide who to contact from campaign economics.

**Where do 500 / 50 / 30% come from?**  
They are assumptions (no real business data). That is why the repo shows a sensitivity plot and a break-even
success rate instead of a single "savings" number.

**What is PSI?**  
Population Stability Index: compares the distribution of a feature (or the score) in live data vs. training,
over bins. <0.10 stable, 0.10-0.25 watch, >0.25 significant shift. Input drift is a warning, not proof the
model got worse - that needs labels (`performance.py`).

**How did you avoid leakage?**  
Preprocessing lives inside sklearn Pipelines (fit on train folds only); tuning/selection/threshold use train-only
cross-validation; the test set is used once for reporting.

**Is the fairness audit enough?**  
No. It checks gender and SeniorCitizen on one dataset. Gender is not a feature. It is an audit, not a guarantee.

## Tier 3 (what each one really is in this repo)
**DVC - what does it do here?**  
Versions the raw dataset (pointer file in git, bytes in a DVC remote) and defines the pipeline in `dvc.yaml`
(validate -> train -> gate) with `params.yaml`. `dvc repro` reruns only stages whose inputs changed;
`dvc params diff` / `dvc metrics diff` compare experiments. My remote is a local folder - in a team it would be S3/GCS/DagsHub.

**Airflow?**  
Two DAGs (weekly retrain with quality gate, daily monitoring) that call the same scripts. It shows how the
pipeline would be orchestrated; I did not run it on a production Airflow cluster. Airflow has no native Windows support (WSL2/Docker).

**Kubernetes?**  
Manifests for API (2 replicas, probes, resource limits, HPA), dashboard, and a daily drift CronJob. They pass
schema validation; I did not operate a real cluster. Logs use a single ReadWriteOnce volume - fine for a demo,
production would stream logs to a queue/DB.

**Deep learning / stacking?**  
A PyTorch MLP and a stacking ensemble are benchmarks. They did not beat the simple model by a significant margin on this
tabular data (typical), and they cannot give per-customer reasons, so they are not deployed. Knowing when NOT to use
a bigger model is the point.

**What would you do next?**  
Time-based validation on real data, uplift modelling (who is *persuadable*, not just who will churn), collecting labels via
`/feedback`, and a shadow deployment before switching models.
