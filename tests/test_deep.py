import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_torch_mlp_learns_a_signal():
    from sklearn.metrics import roc_auc_score
    from deep import TorchMLPClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=600) > 0).astype(int)
    m = TorchMLPClassifier(epochs=20, random_state=0).fit(X[:450], y[:450])
    p = m.predict_proba(X[450:])
    assert p.shape == (150, 2) and np.allclose(p.sum(1), 1)
    assert roc_auc_score(y[450:], p[:, 1]) > 0.85
