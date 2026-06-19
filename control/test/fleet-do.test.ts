import { describe, expect, it } from "vitest";

import { NodeState } from "../src/fleet";
import { FleetDO } from "../src/fleet-do";

// In-memory Durable-Object-state harness: get/put/delete/list + blockConcurrencyWhile.
// Exercises FleetDO's persist/hydrate, per-stream storage (M6), and heartbeat-seed-on-hydrate
// (M9) in node — without workerd. (Full workerd integration needs vitest 4; deferred.)
function fakeCtx(store = new Map<string, unknown>()) {
  const ctx: any = {
    storage: {
      get: async (k: string) => store.get(k),
      put: async (k: string, v: unknown) => void store.set(k, v),
      delete: async (k: string) => void store.delete(k),
      list: async (opts?: { prefix?: string }) => {
        const m = new Map<string, unknown>();
        for (const [k, v] of store) if (!opts?.prefix || k.startsWith(opts.prefix)) m.set(k, v);
        return m;
      },
    },
    blockConcurrencyWhile: (fn: () => Promise<void>) => (ctx._ready = fn()),
    _store: store,
    _ready: Promise.resolve(),
  };
  return ctx;
}

describe("FleetDO", () => {
  it("persists registry + per-stream history; a fresh instance hydrates both (M6)", async () => {
    const store = new Map<string, unknown>();

    const ctxA = fakeCtx(store);
    const a = new FleetDO(ctxA as any, {});
    await ctxA._ready;
    await a.registerNode("n0", 0, 0, 35, "v1");
    await a.recordTokens("s", [1, 2, 3]);
    await a.recordTokens("s", [4, 5]);
    expect(store.has("hist:s")).toBe(true); // per-stream key, not one global blob

    const ctx2 = fakeCtx(store);
    const b = new FleetDO(ctx2 as any, {});
    await ctx2._ready;
    expect(b.historyOf("s")).toEqual([1, 2, 3, 4, 5]);
  });

  it("drops a stream's history key on completion (retention)", async () => {
    const store = new Map<string, unknown>();
    const ctx = fakeCtx(store);
    const d = new FleetDO(ctx as any, {});
    await ctx._ready;
    await d.recordTokens("s", [9, 9]);
    await d.dropStream("s");
    expect(d.historyOf("s")).toEqual([]);
    expect(store.has("hist:s")).toBe(false);
  });

  it("seeds an ACTIVE node's heartbeat on hydrate so it isn't read as instantly stale (M9)", async () => {
    const store = new Map<string, unknown>();
    store.set("registry", {
      nodes: [{ nodeId: "n0", stage: 0, layerStart: 0, layerEnd: 11, version: "v1", state: NodeState.Active, lastHeartbeat: 1 }],
      topology: [],
    });
    const ctx = fakeCtx(store);
    const d = new FleetDO(ctx as any, {});
    await ctx._ready;
    // Without the hydrate-seed, lastHeartbeat=1 → instantly stale; with it, ≈Date.now().
    expect(d.staleNodes(Date.now(), 60_000)).toEqual([]);
  });

  it("selects the recovery policy over RPC", async () => {
    const ctx = fakeCtx();
    const d = new FleetDO(ctx as any, {});
    await ctx._ready;
    await d.registerNode("n0", 0, 0, 11, "v1");
    expect(d.selectRecovery("n0", 1, "v1").policy).toBe("reassign");
    expect(d.selectRecovery("n0", 0, "v1").policy).toBe("rebuild");
    expect(d.selectRecovery("n0", 1, "v2").policy).toBe("rebuild");
  });

  it("enforces the node state machine", async () => {
    const ctx = fakeCtx();
    const d = new FleetDO(ctx as any, {});
    await ctx._ready;
    await d.registerNode("n0", 0, 0, 11, "v1");
    await expect(d.setState("n0", NodeState.Active)).rejects.toThrow(); // PROVISIONING→ACTIVE illegal
    await d.setState("n0", NodeState.Staging);
    await d.setState("n0", NodeState.Warm);
    await d.setState("n0", NodeState.Loading);
    await d.setState("n0", NodeState.Active);
  });

  it("rate-limits per key (M7)", async () => {
    const ctx = fakeCtx();
    const d = new FleetDO(ctx as any, {});
    await ctx._ready;
    expect(d.checkRate("k", 2, 60_000).allowed).toBe(true);
    expect(d.checkRate("k", 2, 60_000).allowed).toBe(true);
    expect(d.checkRate("k", 2, 60_000).allowed).toBe(false);
  });
});
