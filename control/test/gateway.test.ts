import { describe, expect, it } from "vitest";
import { authenticate, formatResponse, GatewayError, parseRequest } from "../src/gateway";

const KEYS = new Set(["sk-good"]);
const MODELS = new Set(["gpt-oss-120b"]);

function body(over: Record<string, unknown> = {}) {
  return { model: "gpt-oss-120b", messages: [{ role: "user", content: "hello" }], max_tokens: 8, ...over };
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
    expect(req.model).toBe("gpt-oss-120b");
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

describe("formatResponse", () => {
  it("produces an OpenAI-compatible completion with usage", () => {
    const req = parseRequest(body({ max_tokens: 3 }), MODELS);
    const resp = formatResponse("chatcmpl-1", req, "101 102 103", 5, 3);
    expect(resp.object).toBe("chat.completion");
    expect(resp.choices[0].message.role).toBe("assistant");
    expect(resp.choices[0].finish_reason).toBe("stop");
    expect(resp.usage.total_tokens).toBe(8);
  });
});
