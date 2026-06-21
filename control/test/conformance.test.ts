// Cross-impl conformance (S8): the model registry + error shape are single-sourced, not triplicated.
// The Worker imports configs/models.json (the SAME file the Python gateway reads via load_model_names);
// the Python side asserts that file is fresh from the YAMLs (scheduler/tests/test_conformance.py). So:
// worker model set == models.json == configs/*.yaml names == Python registry == demo set.
import { describe, expect, it } from "vitest";
import { MODELS } from "../src/index";
import { GatewayError } from "../src/gateway";
import modelRegistry from "../../configs/models.json";

describe("conformance (S8): single-sourced model registry + error shape", () => {
  it("the Worker model set is exactly configs/models.json (no per-gateway literal)", () => {
    expect([...MODELS].sort()).toEqual([...modelRegistry.models].sort());
  });

  it("the registry is non-empty and carries the v1.0 proof model", () => {
    expect(modelRegistry.models.length).toBeGreaterThan(0);
    expect(MODELS.has("gpt-oss-120b")).toBe(true);
  });

  it("GatewayError.toError() emits the agreed OpenAI error shape {error:{message,type,code}}", () => {
    const e = new GatewayError(404, "model 'x' not found", "model_not_found").toError();
    expect(Object.keys(e)).toEqual(["error"]);
    expect(Object.keys(e.error).sort()).toEqual(["code", "message", "type"]);
    expect(e.error.type).toBe("model_not_found");
    expect(e.error.code).toBe("model_not_found");
  });
});
