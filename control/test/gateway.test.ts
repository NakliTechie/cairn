import { describe, expect, it } from "vitest";
import { authenticate, GatewayError, parseRequest } from "../src/gateway";

const KEYS = new Set(["sk-good"]);
const MODELS = new Set(["llama-3.1-8b"]);

function body(over: Record<string, unknown> = {}) {
  return { model: "llama-3.1-8b", messages: [{ role: "user", content: "hello" }], max_tokens: 8, ...over };
}

describe("authenticate", () => {
  it("accepts a valid bearer key", () => {
    expect(authenticate("Bearer sk-good", KEYS)).toBe("sk-good");
  });
  it("rejects missing/malformed/invalid with 401", () => {
    for (const bad of [null, "", "sk-good", "Bearer ", "Bearer sk-nope"]) {
      expect(() => authenticate(bad, KEYS)).toThrowError(GatewayError);
      try {
        authenticate(bad, KEYS);
      } catch (e) {
        expect((e as GatewayError).status).toBe(401);
      }
    }
  });
});

describe("parseRequest", () => {
  it("parses a valid request", () => {
    const req = parseRequest(body(), MODELS);
    expect(req.model).toBe("llama-3.1-8b");
    expect(req.max_tokens).toBe(8);
    expect(req.stream).toBe(false);
  });

  it.each([
    [{ messages: [{ role: "user", content: "x" }] }, 400], // missing model
    [body({ model: "nope" }), 404], // unknown model
    [body({ messages: [] }), 400], // empty messages
    [body({ messages: [{ role: "user" }] }), 400], // malformed message
    [body({ max_tokens: 0 }), 400], // bad max_tokens
    [body({ max_tokens: 10_000_000 }), 400], // H1: over the ceiling
    [body({ messages: Array(300).fill({ role: "user", content: "x" }) }), 400], // H2: too many messages
    [body({ messages: [{ role: "user", content: "x".repeat(200_000) }] }), 400], // H2: prompt too large
  ])("rejects %o with status %i", (b, status) => {
    try {
      parseRequest(b, MODELS);
      throw new Error("should have thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(GatewayError);
      expect((e as GatewayError).status).toBe(status);
    }
  });
});

describe("temperature (W3)", () => {
  it("keeps a numeric temperature, defaults anything else to 0 — never throws", () => {
    expect(parseRequest(body({ temperature: 0.7 }), MODELS).temperature).toBe(0.7);
    for (const bad of ["hot", null, true, [0.5]]) {
      expect(parseRequest(body({ temperature: bad }), MODELS).temperature).toBe(0);
    }
  });
});
