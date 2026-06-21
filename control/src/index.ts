// Cairn control plane — the Workers gateway (spec §7). One ingress (invariant #7):
// auth → validate → admit → route → stream. NEVER in the per-token hot path
// (invariant #4): on a valid request it forwards to the in-VPC entry node, which runs
// the decode loop; the Worker only shapes the request and relays the stream back.
//
// It also exposes an authed /internal/* control surface (S4) backed by the FleetDO: the
// data plane registers nodes / heartbeats / mirrors token-history; the recovery loop reads
// health + policy. Internal calls are authenticated by CAIRN_INTERNAL_TOKEN, never the
// public API keys.

import { authenticate, constantTimeEqual, GatewayError, parseRequest } from "./gateway";
import { FleetDO } from "./fleet-do";
import modelRegistry from "../../configs/models.json";

export interface Env {
  SERVICE_VERSION?: string;
  CAIRN_API_KEYS?: string; // comma-separated, set via `wrangler secret put` (spec §8)
  CAIRN_INTERNAL_TOKEN?: string; // shared secret for the data-plane → control-plane /internal/* calls
  DATA_PLANE_URL?: string; // the in-VPC entry node ingress (set per deployment)
  FLEET: DurableObjectNamespace<FleetDO>;
}

// Single-sourced from configs/models.json (invariant #3) — generated from the per-model YAMLs by
// configs/gen_models.py, the SAME registry the Python gateway reads (load_model_names). Never a
// per-gateway literal. Conformance: control/test/conformance.test.ts.
export const MODELS = new Set<string>(modelRegistry.models);
const MAX_BODY_BYTES = 1_048_576; // 1 MiB request-body ceiling (H2)
const UPSTREAM_TIMEOUT_MS = 120_000; // abort a hung data-plane forward (H4)
const RATE_LIMIT_PER_MIN = 120; // per-key request ceiling (M7)
const RATE_WINDOW_MS = 60_000;
// Only these upstream response headers reach the client — never Set-Cookie or internal
// banners (H3). text/event-stream (streaming) rides on content-type.
const ALLOWED_RESPONSE_HEADERS = ["content-type", "cache-control", "x-request-id"];

const _keyCache = new WeakMap<Env, Set<string>>(); // memoize per env — avoid re-parsing each request (S9)
function apiKeys(env: Env): Set<string> {
  let keys = _keyCache.get(env);
  if (!keys) {
    keys = new Set((env.CAIRN_API_KEYS ?? "").split(",").map((s) => s.trim()).filter(Boolean));
    _keyCache.set(env, keys);
  }
  return keys;
}

function json(obj: unknown, status = 200): Response {
  return new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json" } });
}

// The single fleet-state Durable Object for the cluster.
function fleet(env: Env): DurableObjectStub<FleetDO> {
  return env.FLEET.get(env.FLEET.idFromName("fleet"));
}

function internalAuthOk(request: Request, env: Env): boolean {
  const tok = request.headers.get("x-cairn-internal");
  // Fail closed: no configured token ⇒ the internal surface is unreachable.
  return !!env.CAIRN_INTERNAL_TOKEN && !!tok && constantTimeEqual(tok, env.CAIRN_INTERNAL_TOKEN);
}

// Authed data-plane / recovery-loop control surface, backed by the FleetDO (S4).
async function handleInternal(request: Request, url: URL, env: Env): Promise<Response> {
  if (!internalAuthOk(request, env)) {
    return json({ error: { message: "internal auth failed", type: "authentication_error", code: "authentication_error" } }, 401);
  }
  const f = fleet(env);
  const p = url.pathname;
  try {
    if (request.method === "POST") {
      const b = (await request.json().catch(() => ({}))) as Record<string, unknown>;
      switch (p) {
        case "/internal/register":
          await f.registerNode(b.nodeId as string, b.stage as number, b.layerStart as number, b.layerEnd as number, b.version as string, (b.t as number) ?? 0);
          return json({ ok: true });
        case "/internal/state":
          await f.setState(b.nodeId as string, b.state as NodeStateLike, b.t as number);
          return json({ ok: true });
        case "/internal/topology":
          await f.setTopology((b.order as string[]) ?? []);
          return json({ ok: true });
        case "/internal/heartbeat":
          await f.heartbeat(b.nodeId as string, (b.t as number) ?? Date.now());
          return json({ ok: true });
        case "/internal/tokens":
          await f.recordTokens(b.streamId as string, (b.tokenIds as number[]) ?? []);
          return json({ ok: true });
        case "/internal/stream/complete":
          await f.dropStream(b.streamId as string);
          return json({ ok: true });
      }
    } else if (request.method === "GET") {
      const q = url.searchParams;
      if (p === "/internal/recovery") {
        return json(await f.selectRecovery(q.get("nodeId") ?? "", Number(q.get("warmSpares") ?? 0), q.get("spareVersion") ?? ""));
      }
      if (p === "/internal/stale") {
        return json({ stale: await f.staleNodes(Number(q.get("now") ?? 0), Number(q.get("timeout") ?? 0)) });
      }
    }
    return json({ error: { message: "not found", type: "not_found", code: "not_found" } }, 404);
  } catch (e) {
    // FleetError (illegal transition, overlap, over-cap, …) → 400 with the message.
    return json({ error: { message: String((e as Error)?.message ?? e), type: "invalid_request_error", code: "invalid_request_error" } }, 400);
  }
}

// The DO's RPC type for setState's enum arg (kept loose to avoid importing the enum here).
type NodeStateLike = Parameters<FleetDO["setState"]>[1];

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname.startsWith("/internal/")) {
      return handleInternal(request, url, env);
    }
    if (request.method === "GET" && url.pathname === "/version") {
      return json({ service: "cairn", version: env.SERVICE_VERSION ?? "dev" });
    }
    if (request.method === "GET" && url.pathname === "/v1/health") {
      return json({ status: "ok" });
    }
    if (request.method === "POST" && url.pathname === "/v1/chat/completions") {
      try {
        const key = authenticate(request.headers.get("authorization"), apiKeys(env));
        // M7: per-key rate limit via the FleetDO counter (skipped if no DO binding, e.g. in unit tests).
        if (env.FLEET) {
          const rl = await fleet(env).checkRate(key, RATE_LIMIT_PER_MIN, RATE_WINDOW_MS);
          if (!rl.allowed) {
            return json({ error: { message: "rate limit exceeded", type: "rate_limit_error", code: "rate_limit_exceeded" } }, 429);
          }
        }
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
        let upstream: Response;
        try {
          upstream = await fetch(env.DATA_PLANE_URL, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(req),
            signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS), // H4: never hang on a dead node
          });
        } catch {
          return json({ error: { message: "upstream timeout or unreachable", type: "upstream_unavailable", code: "upstream_unavailable" } }, 504);
        }
        // H3: forward only an allowlist of response headers — never Set-Cookie / internal banners.
        const safe = new Headers();
        for (const h of ALLOWED_RESPONSE_HEADERS) {
          const v = upstream.headers.get(h);
          if (v) safe.set(h, v);
        }
        return new Response(upstream.body, { status: upstream.status, headers: safe });
      } catch (e) {
        if (e instanceof GatewayError) return json(e.toError(), e.status);
        return json({ error: { message: "internal error", type: "internal_error", code: "internal_error" } }, 500);
      }
    }
    return json({ error: { message: "not found", type: "not_found", code: "not_found" } }, 404);
  },
};

export { FleetDO };
