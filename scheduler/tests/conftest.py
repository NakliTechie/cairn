import pathlib
import sys

import pytest

# Make the package importable whether or not it's been pip-installed into the env.
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cairn_scheduler import load_model_config  # noqa: E402

CONFIGS = pathlib.Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture
def configs_dir():
    return CONFIGS


@pytest.fixture
def model_cfg():
    return load_model_config(CONFIGS / "llama-3.1-8b.yaml")
