import numpy as np

from policy import (bootstrap_ci, breakeven_save_rate, ece, optimal_threshold, paired_bootstrap_diff,
                    topk_mask, topk_metrics, total_cost)
from sklearn.metrics import roc_auc_score


def test_topk_metrics_toy():
    m = topk_metrics([1, 0, 1, 0, 0], [0.9, 0.8, 0.7, 0.2, 0.1], 0.4)
    assert m["n_contacted"] == 2 and m["precision"] == 0.5 and m["recall"] == 0.5 and abs(m["lift"] - 1.25) < 1e-9
    assert topk_mask([0.1, 0.9, 0.5], 0.34).tolist() == [False, True, True]


def test_total_cost_economics():
    # contacted churner: offer + (1-save)*loss ; untreated churner: loss ; contacted stayer: offer
    assert abs(total_cost([1], [True], 500, 50, 0.3) - (50 + 0.7 * 500)) < 1e-9
    assert total_cost([1], [False], 500, 50, 0.3) == 500
    assert total_cost([0], [True], 500, 50, 0.3) == 50


def test_threshold_and_breakeven():
    assert abs(optimal_threshold(500, 50, 0.3) - 1 / 3) < 1e-9
    assert abs(breakeven_save_rate(0.5, 500, 50) - 0.2) < 1e-9


def test_ece_perfect_and_bad():
    y = np.array([0, 1] * 50)
    assert ece(y, y.astype(float)) == 0.0
    assert ece(y, np.full(100, 0.99)) > 0.4


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    p = rng.random(800)
    y = (rng.random(800) < p).astype(int)
    lo, hi = bootstrap_ci(y, p, roc_auc_score, 300)
    assert lo < roc_auc_score(y, p) < hi


def test_paired_diff_of_identical_models_includes_zero():
    rng = np.random.default_rng(1)
    p = rng.random(600)
    y = (rng.random(600) < p).astype(int)
    mean, lo, hi = paired_bootstrap_diff(y, p, p, roc_auc_score, 200)
    assert mean == 0 and lo <= 0 <= hi
