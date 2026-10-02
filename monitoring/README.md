# monitoring/

Drop the latest batch of customers here as `current.csv` (same raw columns as the Telco data, no churn label).
The scheduled workflow `.github/workflows/monitor.yml` compares it with the training reference every day.

Try it: `python monitor.py --simulate --save-batch monitoring/current.csv`, commit, then run the workflow manually.
