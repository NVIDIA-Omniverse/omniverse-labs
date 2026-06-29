"""Shared test setup for ovx-USD-FMI tests."""

import os
import sys
import tempfile
from pathlib import Path

import pytest


def _load_app_env():
    """Load the setup-generated .env so tests find the isolated USD Python."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return

    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        value = value.replace("${PYTHONPATH:-}", os.environ.get("PYTHONPATH", ""))
        os.environ.setdefault(key, value)


def pytest_configure():
    _load_app_env()

    if not sys.platform.startswith("win"):
        return

    repo_root = Path(__file__).resolve().parents[3]
    tmp_dir = repo_root / "build" / "pytest-tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TEMP"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def bouncing_ball_defaults():
    """Default bouncing ball FMU parameters."""
    return {
        "h0": 1.0,
        "v0": 0.0,
        "g": -9.81,
        "e": 0.7,
        "v_min": 0.1,
    }


@pytest.fixture
def pd_controller_defaults():
    """Default PD controller FMU parameters (Python simulation API)."""
    return {
        "target": [0.0, 2.0, 0.0],
        "kp": 50.0,
        "kd": 10.0,
        "mass": 1.0,
        "gravity": -9.81,
    }
