"""Cross-impl conformance (S8): the model registry + error shape are single-sourced, not triplicated.

`configs/models.json` is the ONE model registry both gateways read — Python here (`load_model_names`
+ the demo), and the TS Worker imports the same file (control/test/conformance.test.ts asserts its
side). This guards the Python side + the shared artifact:
  - models.json is FRESH: equals the set of top-level `name:` across configs/*.yaml, so adding a config
    without re-running configs/gen_models.py fails here instead of silently drifting (the S8 drift).
  - the demo gateway accepts EXACTLY the registry — no hardcoded per-gateway list (the old S8 source).
  - GatewayError.to_error() emits the agreed OpenAI error shape {error:{message,type,code}}.
"""
import json
import pathlib
import sys

import yaml

from cairn_scheduler import load_model_names
from cairn_scheduler.gateway import GatewayError

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
MANIFEST = CONFIGS / "models.json"


def _yaml_names():
    return sorted(yaml.safe_load(y.read_text())["name"] for y in CONFIGS.glob("*.yaml"))


def test_manifest_is_fresh_from_configs():
    """models.json is derived from the YAMLs — run `python configs/gen_models.py` if this fails."""
    assert json.loads(MANIFEST.read_text())["models"] == _yaml_names()


def test_load_model_names_is_the_registry():
    assert load_model_names(MANIFEST) == set(_yaml_names())


def test_demo_gateway_advertises_the_registry():
    """The demo (the historically-hardcoded gateway) now serves exactly the single-sourced registry."""
    _demo = str(ROOT / "demo")
    if _demo not in sys.path:
        sys.path.insert(0, _demo)
    from server import _engine
    assert _engine().gateway.model_names == load_model_names(MANIFEST)


def test_error_shape_is_openai_compatible():
    err = GatewayError(404, "model 'x' not found", "model_not_found").to_error()
    assert set(err) == {"error"}
    assert set(err["error"]) == {"message", "type", "code"}
    assert err["error"]["type"] == "model_not_found"
    assert err["error"]["code"] == "model_not_found"
