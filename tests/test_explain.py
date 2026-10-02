import json

import numpy as np
import pandas as pd
import joblib

from explain import env_mismatch, env_versions, make_explainer


def test_explainer_shapes_and_additivity(trained):
    model = joblib.load(trained / "models/model.joblib")
    meta = json.load(open(trained / "models/meta.json"))
    ref = pd.read_csv(trained / "models/reference.csv")[meta["columns"]]
    names, fn = make_explainer(model, ref)
    Xt = model.named_steps["pre"].transform(ref.head(50))
    contrib = fn(Xt)
    assert contrib.shape == (50, len(names))
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):      # linear model: contributions must sum to logit - mean logit
        mean = model.named_steps["pre"].transform(ref).mean(axis=0)
        expected = model.decision_function(ref.head(50)) - (clf.intercept_[0] + mean @ clf.coef_[0])
        assert np.allclose(contrib.sum(axis=1), expected, atol=1e-6)


def test_env_mismatch_detects_change():
    saved = env_versions()
    assert env_mismatch(saved) == {}
    saved["numpy"] = "0.0.1"
    assert "numpy" in env_mismatch(saved)
