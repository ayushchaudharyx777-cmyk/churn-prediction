import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Generate synthetic data and run the REAL train.py in FAST mode inside a temp dir."""
    wd = tmp_path_factory.mktemp("run")
    subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic_data.py"), "--out",
                    str(wd / "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"), "--rows", "1500"], check=True)
    env = {**os.environ, "FAST": "1", "DISABLE_MLFLOW": "1"}
    subprocess.run([sys.executable, str(ROOT / "train.py")], cwd=wd, env=env, check=True)
    return wd
