import numpy as np
import pandas as pd

from performance import live_performance


def _frames(n, informative, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    y = (rng.random(n) < p).astype(int) if informative else rng.integers(0, 2, n)
    ids = [f"r{i}" for i in range(n)]
    return pd.DataFrame({"request_id": ids, "churn_probability": p}), pd.DataFrame({"request_id": ids, "churn": y})


def test_ok_when_scores_are_informative():
    log, lab = _frames(800, True)
    assert live_performance(log, lab, baseline_auc=0.667)["status"] == "ok"


def test_degraded_when_scores_are_noise():
    log, lab = _frames(800, False)
    r = live_performance(log, lab, baseline_auc=0.667)
    assert r["status"] == "degraded" and r["auc_drop"] > 0.05


def test_insufficient_labels():
    log, lab = _frames(20, True)
    assert live_performance(log, lab, baseline_auc=0.7)["status"] == "insufficient_data"
