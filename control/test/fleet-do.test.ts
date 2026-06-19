import { describe, expect, it } from "vitest";

import { NodeState } from "../src/fleet";
import { FleetDO } from "../src/fleet-do";

// In-memory Durable-Object-state harness: exercises the FleetDO wrapper's persistence,
// hydration, RPC methods, and state-machine enforcement in node — without workerd. (Full
// workerd-runtime integration via @cloudflare/vitest-pool-workers needs vitest 4; deferred
// to the deploy step. The pure Fleet logic is also covered by fleet.test.ts.)
function fakeCtx(store = new Map<string, unknown>()) {
  const ctx: any = {
    storage: {
      get: async (k: string) => store.get(k),
      put: async (k: string, v: unknown) => {
        store.set(k, v);
      },
    },
    blockConcurrencyWhile: (fn: () => Promise<void>) => (ctx._ready = fn()),
    _store: store,
    _ready: Promise.resolve(),
  };
  return ctx;
}

describe("FleetDO", () => {
  it("persists registry + token-history; a fresh instance hydrates from storage", async () => {
    const store = new Map<string, unknown>();

    const ctx1 = fakeCtx(store);
    const a = new FleetDO(ctx1 as any, {});
    await ctx1._ready;
    await a.registerNode("n0", 0, 0, 35, "v1");
    await a.recordTokens("s", [1, 2, 3]);
    await a.recordTokens("s", [4, 5]);

    // A fresh instance over the SAME storage must reconstruct the persisted snapshot.
    const ctx2 = fakeCtx(store);
    const b = new FleetDO(ctx2 as any, {});
    await ctx2._ready;
    expect(b.historyOf("s")).toEqual([1, 2, 3, 4, 5]);
  });

  it("drops a stream's history (retention — freed on completion)", async () => {
    const ctx = fakeCtx();
    const d = new FleetDO(ctx as any, {});
    await ctx._ready;
    await d.recordTokens("s", [9, 9]);
    await d.dropStream("s");
    expect(d.historyOf("s")).toEqual([]);
  });

  it("selects the recovery policy over RPC", async () => {
    const ctx = fakeCtx();
    const d = new FleetDO(ctx as any, {});
    await ctx._ready;
    await d.registerNode("n0", 0, 0, 11, "v1");
    expect(d.selectRecovery("n0", 1, "v1").policy).toBe("reassign");
    expect(d.selectRecovery("n0", 0, "v1").policy).toBe("rebuild"); // no spare
    expect(d.selectRecovery("n0", 1, "v2").policy).toBe("rebuild"); // version skew
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
    await d.setState("n0", NodeState.Active); // legal now
  });
});
