import json

FILES = ["models/model.joblib", "models/meta.json", "models/reference.csv", "models/registry.json",
         "reports/final_metrics.json", "reports/metrics.csv", "reports/high_risk_customers.csv",
         "reports/RESULTS.md", "reports/MODEL_CARD.md", "reports/data_validation.json",
         "reports/roc.png", "reports/shap_top_features.png", "reports/ONE_PAGER.md", "reports/fairness_slices.csv",
         "reports/gain_lift.png", "reports/sensitivity.csv", "reports/sensitivity.png"]


def test_artifacts_exist(trained):
    for f in FILES:
        assert (trained / f).exists(), f


def test_metrics_sane(trained):
    fm = json.load(open(trained / "reports/final_metrics.json"))
    assert fm["roc_auc"] > 0.6
    assert 0 < fm["threshold"] < 1
    assert fm["lift_at_capacity"] > 1.0
    assert fm["saving_capacity_policy"] > 0                      # call list beats doing nothing
    rand = next(r for r in fm["policies"] if r["policy"].startswith("Random"))
    assert rand["saving_vs_do_nothing"] < fm["saving_capacity_policy"]   # and beats random targeting


def test_gender_excluded_from_model(trained):
    meta = json.load(open(trained / "models/meta.json"))
    assert "gender" not in meta["columns"] and meta["excluded_features"] == ["gender"]
    assert meta["env"]["scikit-learn"]


def test_registry_has_version(trained):
    reg = json.load(open(trained / "models/registry.json"))
    meta = json.load(open(trained / "models/meta.json"))
    assert reg[-1]["version"] == meta["model_version"]
