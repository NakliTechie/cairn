// FleetDO — the Durable Object that holds fleet state (spec §7), wrapping the pure
// Fleet logic and persisting a snapshot to DO storage so the durable token-history
// (invariant #2) survives. RPC methods are called by the gateway Worker.
//
// Runtime-tested at deploy with workerd; here it is typecheck-verified (the pure Fleet
// logic it delegates to is unit-tested in test/fleet.test.ts).

import { DurableObject } from "cloudflare:workers";
import { Fleet, NodeInfo, NodeState } from "./fleet";

interface Snapshot {
  nodes: NodeInfo[];
  topology: string[];
  history: [string, number[]][];
}

export class FleetDO extends DurableObject {
  private fleet = new Fleet();

  constructor(ctx: DurableObjectState, env: unknown) {
    super(ctx, env as never);
    ctx.blockConcurrencyWhile(async () => {
      const snap = await ctx.storage.get<Snapshot>("snapshot");
      if (snap) {
        for (const n of snap.nodes) this.fleet.nodes.set(n.nodeId, n);
        this.fleet.topology = snap.topology;
        for (const [k, v] of snap.history) this.fleet.tokenHistory.set(k, v);
      }
    });
  }

  private async persist(): Promise<void> {
    const snap: Snapshot = {
      nodes: [...this.fleet.nodes.values()],
      topology: this.fleet.topology,
      history: [...this.fleet.tokenHistory.entries()],
    };
    await this.ctx.storage.put("snapshot", snap);
  }

  async registerNode(nodeId: string, stage: number, layerStart: number, layerEnd: number, version: string, t = 0): Promise<void> {
    this.fleet.registerNode(nodeId, stage, layerStart, layerEnd, version, t);
    await this.persist();
  }

  async setState(nodeId: string, state: NodeState): Promise<void> {
    this.fleet.setState(nodeId, state);
    await this.persist();
  }

  async setTopology(orderedNodeIds: string[]): Promise<void> {
    this.fleet.setTopology(orderedNodeIds);
    await this.persist();
  }

  async heartbeat(nodeId: string, t: number): Promise<void> {
    this.fleet.heartbeat(nodeId, t);
    // heartbeats are frequent + non-critical — not persisted on the hot path
  }

  staleNodes(now: number, timeout: number): string[] {
    return this.fleet.staleNodes(now, timeout);
  }

  async recordTokens(streamId: string, tokenIds: number[]): Promise<void> {
    this.fleet.recordTokens(streamId, tokenIds);
    await this.persist();
  }

  historyOf(streamId: string): number[] {
    return this.fleet.historyOf(streamId);
  }

  async dropStream(streamId: string): Promise<void> {
    this.fleet.dropStream(streamId);
    await this.persist();
  }

  selectRecovery(nodeId: string, warmSpares: number, spareVersion: string) {
    return this.fleet.selectRecovery(nodeId, warmSpares, spareVersion);
  }

  isCompletePipeline(): boolean {
    return this.fleet.isCompletePipeline();
  }
}
