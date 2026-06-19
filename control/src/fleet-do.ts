// FleetDO — the Durable Object holding fleet state (spec §7), wrapping the pure Fleet
// logic and persisting to DO storage so the durable stream token-history (invariant #2)
// survives. RPC methods are called by the gateway Worker's /internal/* endpoints (S4).
//
// Storage model (M6): the registry (nodes + topology) lives under one "registry" key;
// each stream's token-history lives under its own "hist:<streamId>" key — so recordTokens
// rewrites only that stream (not an ever-growing global blob) and dropStream is an O(1)
// delete. Per-stream length is capped in Fleet.recordTokens; streams are dropped on
// completion via the /internal/stream/complete endpoint.
//
// Runtime-tested at deploy with workerd; here the pure Fleet logic is unit-tested
// (test/fleet.test.ts) and the persist/hydrate path via an in-memory harness (test/fleet-do.test.ts).

import { DurableObject } from "cloudflare:workers";
import { Fleet, NodeInfo, NodeState } from "./fleet";

interface Registry {
  nodes: NodeInfo[];
  topology: string[];
}

export class FleetDO extends DurableObject {
  private fleet = new Fleet();

  constructor(ctx: DurableObjectState, env: unknown) {
    super(ctx, env as never);
    ctx.blockConcurrencyWhile(async () => {
      const reg = await ctx.storage.get<Registry>("registry");
      if (reg) {
        const now = Date.now();
        for (const n of reg.nodes) {
          // M9: a hydrated ACTIVE node would otherwise read as instantly stale (its
          // persisted lastHeartbeat predates the wake) → spurious recovery. Seed it.
          if (n.state === NodeState.Active) n.lastHeartbeat = now;
          this.fleet.nodes.set(n.nodeId, n);
        }
        this.fleet.topology = reg.topology;
      }
      const hist = await ctx.storage.list<number[]>({ prefix: "hist:" });
      for (const [key, toks] of hist) this.fleet.tokenHistory.set(key.slice("hist:".length), toks);
    });
  }

  private async persistRegistry(): Promise<void> {
    await this.ctx.storage.put<Registry>("registry", {
      nodes: [...this.fleet.nodes.values()],
      topology: this.fleet.topology,
    });
  }

  // --- registry / topology / health (registry-key writes) ---
  async registerNode(nodeId: string, stage: number, layerStart: number, layerEnd: number, version: string, t = 0): Promise<void> {
    this.fleet.registerNode(nodeId, stage, layerStart, layerEnd, version, t);
    await this.persistRegistry();
  }

  async setState(nodeId: string, state: NodeState, t = Date.now()): Promise<void> {
    this.fleet.setState(nodeId, state, t);
    await this.persistRegistry();
  }

  async setTopology(orderedNodeIds: string[]): Promise<void> {
    this.fleet.setTopology(orderedNodeIds);
    await this.persistRegistry();
  }

  async heartbeat(nodeId: string, t: number): Promise<void> {
    this.fleet.heartbeat(nodeId, t); // frequent + non-critical → in-memory, not persisted
  }

  staleNodes(now: number, timeout: number): string[] {
    return this.fleet.staleNodes(now, timeout);
  }

  // --- durable token-history (per-stream keys) ---
  async recordTokens(streamId: string, tokenIds: number[]): Promise<void> {
    this.fleet.recordTokens(streamId, tokenIds);
    await this.ctx.storage.put(`hist:${streamId}`, this.fleet.historyOf(streamId));
  }

  historyOf(streamId: string): number[] {
    return this.fleet.historyOf(streamId);
  }

  async dropStream(streamId: string): Promise<void> {
    this.fleet.dropStream(streamId);
    await this.ctx.storage.delete(`hist:${streamId}`); // freed on completion (spec §8 retention)
  }

  // --- recovery policy + rate limit ---
  selectRecovery(nodeId: string, warmSpares: number, spareVersion: string) {
    return this.fleet.selectRecovery(nodeId, warmSpares, spareVersion);
  }

  checkRate(key: string, limit: number, windowMs: number): { allowed: boolean; count: number } {
    return this.fleet.checkRate(key, limit, windowMs, Date.now()); // M7 — ephemeral, not persisted
  }

  isCompletePipeline(): boolean {
    return this.fleet.isCompletePipeline();
  }
}
