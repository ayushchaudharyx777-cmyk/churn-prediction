"""Per-customer explanations for the deployed model (linear or XGBoost) + environment checks."""
import importlib.metadata as md

import shap

PKGS = ["scikit-learn", "xgboost", "shap", "pandas", "numpy"]


def env_versions() -> dict:
    out = {}
    for p in PKGS:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "n/a"
    return out


def env_mismatch(saved: dict) -> dict:
    cur = env_versions()
    return {p: {"trained": (saved or {})[p], "running": cur[p]}
            for p in cur if p in (saved or {}) and saved[p] != cur[p]}


def make_explainer(pipe, background_df):
    """Return (feature_names, fn) where fn(transformed_matrix) -> contribution matrix (n, features).
    Linear model: coef * (x - mean) (exact SHAP for independent features). XGBoost: TreeExplainer."""
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    if hasattr(clf, "coef_"):
        coef = clf.coef_[0]
        mean = pre.transform(background_df).mean(axis=0)
        return names, (lambda Xt: (Xt - mean) * coef)
    expl = shap.TreeExplainer(clf)
    return names, (lambda Xt: expl.shap_values(Xt))
