"""Run local tests with native-runtime initialization before pytest plugins."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Ambient third-party pytest plugins are not part of this project's test suite.
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
import torch  # noqa: E402,F401
import pytest  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if not any(arg == "--basetemp" or arg.startswith("--basetemp=") for arg in args):
        args = ["--basetemp", str(ROOT / ".pytest-runtime"), *args]
    raise SystemExit(pytest.main([str(ROOT / "tests"), "-c", str(ROOT / "pytest.ini"), *args]))
