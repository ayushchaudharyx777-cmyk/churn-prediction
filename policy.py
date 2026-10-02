"""Business-policy maths: capacity targeting, campaign economics, calibration, bootstrap CIs.

Economics (all values are ASSUMPTIONS, edit in train.py):
  loss      = revenue lost when a churner leaves
  offer     = cost of one retention offer (paid for every contacted customer)
  save      = probability that an offer actually keeps a churner
  contacting a churner costs  offer + (1-save)*loss ;  an uncontacted churner costs loss.
  => contact customer only if  p * save * loss > offer   (threshold = offer / (loss*save))
"""
import numpy as np


def topk_mask(p, k: float):
    p = np.asarray(p)
    m = int(np.ceil(k * len(p)))
    idx = np.argsort(-p, kind="stable")[:m]
    mask = np.zeros(len(p), dtype=bool)
    mask[idx] = True
    return mask


def topk_metrics(y, p, k: float) -> dict:
    y = np.asarray(y)
    mask = topk_mask(p, k)
    tp, n_t, pos = int((y[mask] == 1).sum()), int(mask.sum()), int((y == 1).sum())
    base = pos / len(y) if len(y) else float("nan")
    prec = tp / n_t if n_t else 0.0
    return {"k": k, "n_contacted": n_t, "precision": prec, "recall": tp / pos if pos else 0.0,
            "lift": prec / base if base else float("nan")}


def total_cost(y, treat, loss: float, offer: float, save: float) -> float:
    y, t = np.asarray(y), np.asarray(treat, dtype=bool)
    untreated = int(((y == 1) & ~t).sum())
    treated = int(((y == 1) & t).sum())
    return float(loss * untreated + loss * (1 - save) * treated + offer * t.sum())


def expected_random_cost(y, k: float, loss: float, offer: float, save: float) -> float:
    y = np.asarray(y)
    pos, n = float((y == 1).sum()), len(y)
    return float(loss * pos * (1 - k) + loss * (1 - save) * pos * k + offer * k * n)


def optimal_threshold(loss: float, offer: float, save: float) -> float:
    return float(min(1.0, offer / (loss * save)))


def breakeven_save_rate(precision: float, loss: float, offer: float) -> float:
    """Offer success rate above which contacting a group with this precision is profitable."""
    return float(offer / (loss * precision)) if precision > 0 else float("inf")


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error (equal-width bins)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(err)


def bootstrap_ci(y, p, fn, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(y, p1, p2, fn, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """CI for fn(y,p1) - fn(y,p2) using the SAME resamples for both models."""
    rng = np.random.default_rng(seed)
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(fn(y[i], p1[i]) - fn(y[i], p2[i]))
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(diffs)), float(lo), float(hi)
