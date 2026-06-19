import { describe, expect, it } from "vitest";
import worker, { type Env } from "../src/index";

const env = { SERVICE_VERSION: "0.1.0", CAIRN_API_KEYS: "sk-good" } as unknown as Env;

function post(bodyObj: unknown, auth?: string): Request {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (auth) headers["authorization"] = auth;
  return new Request("https://cairn.example/v1/chat/completions", {
    method: "POST",
    headers,
    body: JSON.stringify(bodyObj),
  });
}

describe("worker routes", () => {
  it("GET /version returns the service version", async () => {
    const r = await worker.fetch(new Request("https://cairn.example/version"), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toEqual({ service: "cairn", version: "0.1.0" });
  });

  it("GET /v1/health is ok", async () => {
    const r = await worker.fetch(new Request("https://cairn.example/v1/health"), env);
    expect(r.status).toBe(200);
    expect((await r.json() as { status: string }).status).toBe("ok");
  });

  it("rejects chat completion without auth (401)", async () => {
    const r = await worker.fetch(post({ model: "gpt-oss-120b", messages: [{ role: "user", content: "hi" }] }), env);
    expect(r.status).toBe(401);
  });

  it("rejects an invalid body with 400 (authed)", async () => {
    const r = await worker.fetch(post({ messages: [] }, "Bearer sk-good"), env);
    expect(r.status).toBe(400);
  });

  it("rejects an oversized body (H2)", async () => {
    // 413 when Content-Length is present (real HTTP); 400 via the gateway char-cap when it
    // isn't (the in-process Request doesn't always set it). Either way the body is rejected,
    // never buffered into the fleet — that's the security property.
    const big = { model: "gpt-oss-120b", messages: [{ role: "user", content: "a".repeat(1_100_000) }] };
    const r = await worker.fetch(post(big, "Bearer sk-good"), env);
    expect([400, 413]).toContain(r.status);
  });

  it("returns 503 when the data plane is not attached (valid request)", async () => {
    const r = await worker.fetch(
      post({ model: "gpt-oss-120b", messages: [{ role: "user", content: "hi" }] }, "Bearer sk-good"),
      env,
    );
    expect(r.status).toBe(503);
  });

  it("404s an unknown route", async () => {
    const r = await worker.fetch(new Request("https://cairn.example/nope"), env);
    expect(r.status).toBe(404);
  });
});
