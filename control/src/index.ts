// Cairn control plane — the Workers gateway (spec §7). One ingress (invariant #7):
// auth → validate → admit → route → stream. NEVER in the per-token hot path
// (invariant #4): on a valid request it forwards to the in-VPC entry node, which runs
// the decode loop; the Worker only shapes the request and relays the stream back.

import { authenticate, GatewayError, parseRequest } from "./gateway";
import { FleetDO } from "./fleet-do";

export interface Env {
  SERVICE_VERSION?: string;
  CAIRN_API_KEYS?: string; // comma-separated, set via `wrangler secret put` (spec §8)
  DATA_PLANE_URL?: string; // the in-VPC entry node ingress (set per deployment)
  FLEET: DurableObjectNamespace;
}

const MODELS = new Set(["gpt-oss-120b", "qwen3.5-397b-a17b"]);
const MAX_BODY_BYTES = 1_048_576; // 1 MiB request-body ceiling (H2)

function apiKeys(env: Env): Set<string> {
  return new Set((env.CAIRN_API_KEYS ?? "").split(",").map((s) => s.trim()).filter(Boolean));
}

function json(obj: unknown, status = 200): Response {
  return new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json" } });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/version") {
      return json({ service: "cairn", version: env.SERVICE_VERSION ?? "dev" });
    }
    if (request.method === "GET" && url.pathname === "/v1/health") {
      return json({ status: "ok" });
    }
    if (request.method === "POST" && url.pathname === "/v1/chat/completions") {
      try {
        authenticate(request.headers.get("authorization"), apiKeys(env));
        const declaredLen = Number(request.headers.get("content-length") ?? 0);
        if (declaredLen > MAX_BODY_BYTES) {
          throw new GatewayError(413, "request body too large", "payload_too_large");
        }
        const body = await request.json().catch(() => {
          throw new GatewayError(400, "invalid JSON body");
        });
        const req = parseRequest(body, MODELS);
        if (!env.DATA_PLANE_URL) {
          return json(
            { error: { message: "data plane not attached (control-plane-only build)", type: "service_unavailable", code: "service_unavailable" } },
            503,
          );
        }
        // Forward to the in-VPC entry node — the decode loop runs there, not on CF.
        const upstream = await fetch(env.DATA_PLANE_URL, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(req),
        });
        return new Response(upstream.body, { status: upstream.status, headers: upstream.headers });
      } catch (e) {
        if (e instanceof GatewayError) return json(e.toError(), e.status);
        return json({ error: { message: "internal error", type: "internal_error", code: "internal_error" } }, 500);
      }
    }
    return json({ error: { message: "not found", type: "not_found", code: "not_found" } }, 404);
  },
};

export { FleetDO };
