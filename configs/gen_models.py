#!/usr/bin/env python3
"""Generate configs/models.json — the single-source model registry (invariant #3: model is config).

The gateway model set is DERIVED from the per-model config YAMLs, never hand-maintained per gateway:
a model is "known" iff `configs/<model>.yaml` exists. Both gateways read `configs/models.json` (the
Python lib/demo at runtime via `load_model_names`; the TS Worker imports it), so adding a model =
add a YAML + re-run this. The conformance tests (scheduler/tests/test_conformance.py,
control/test/conformance.test.ts) fail if models.json drifts from the YAMLs.

    python configs/gen_models.py        # rewrites configs/models.json from configs/*.yaml
"""
from __future__ import annotations

import json
import pathlib

CONFIGS = pathlib.Path(__file__).resolve().parent


def _name_of(yaml_path: pathlib.Path) -> str | None:
    """Read the top-level `name:` (prefer a real YAML parse; fall back to a line scan so the
    generator runs even without pyyaml installed)."""
    text = yaml_path.read_text()
    try:
        import yaml  # pyyaml is a project dep, but keep the generator runnable without it
        data = yaml.safe_load(text)
        if isinstance(data, dict) and isinstance(data.get("name"), str):
            return data["name"]
    except Exception:
        pass
    for line in text.splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip()
    return None


def model_names() -> list[str]:
    names = [n for y in sorted(CONFIGS.glob("*.yaml")) if (n := _name_of(y))]
    return sorted(names)


def main() -> int:
    names = model_names()
    out = CONFIGS / "models.json"
    out.write_text(json.dumps({"models": names}, indent=2) + "\n")
    print(f"wrote {out.relative_to(CONFIGS.parent)} ({len(names)} models): {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
