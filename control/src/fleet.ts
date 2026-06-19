// Fleet state logic (spec §7) — registry / topology / health / durable token-history
// + recovery-policy selection. Pure, in-memory, unit-testable; the FleetDO (fleet-do.ts)
// wraps this and persists to Durable Object storage. Mirrors scheduler/.../fleet.py.

export enum NodeState {
  Provisioning = "provisioning",
  Staging = "staging",
  Warm = "warm",
  Loading = "loading",
  Active = "active",
  Draining = "draining",
  Dead = "dead",
}

const TRANSITIONS: Record<NodeState, NodeState[]> = {
  [NodeState.Provisioning]: [NodeState.Staging, NodeState.Dead],
  [NodeState.Staging]: [NodeState.Warm, NodeState.Dead],
  [NodeState.Warm]: [NodeState.Loading, NodeState.Draining, NodeState.Dead],
  [NodeState.Loading]: [NodeState.Active, NodeState.Dead],
  [NodeState.Active]: [NodeState.Draining, NodeState.Dead],
  [NodeState.Draining]: [NodeState.Dead],
  [NodeState.Dead]: [],
};

export function canTransition(from: NodeState, to: NodeState): boolean {
  return TRANSITIONS[from].includes(to);
}

export interface NodeInfo {
  nodeId: string;
  stage: number;
  layerStart: number;
  layerEnd: number;
  version: string;
  state: NodeState;
  lastHeartbeat: number;
}

export class FleetError extends Error {}

export class Fleet {
  nodes = new Map<string, NodeInfo>();
  topology: string[] = [];
  tokenHistory = new Map<string, number[]>();

  registerNode(nodeId: string, stage: number, layerStart: number, layerEnd: number, version: string, t = 0): NodeInfo {
    if (this.nodes.has(nodeId)) throw new FleetError(`node ${nodeId} already registered`);
    const info: NodeInfo = {
      nodeId, stage, layerStart, layerEnd, version, state: NodeState.Provisioning, lastHeartbeat: t,
    };
    this.nodes.set(nodeId, info);
    return info;
  }

  private node(nodeId: string): NodeInfo {
    const n = this.nodes.get(nodeId);
    if (!n) throw new FleetError(`unknown node ${nodeId}`);
    return n;
  }

  setState(nodeId: string, state: NodeState): void {
    const info = this.node(nodeId);
    if (!canTransition(info.state, state)) {
      throw new FleetError(`illegal transition ${info.state} → ${state} for ${nodeId}`);
    }
    info.state = state;
  }

  setTopology(orderedNodeIds: string[]): void {
    for (const id of orderedNodeIds) this.node(id);
    this.topology = [...orderedNodeIds];
  }

  pipeline(): NodeInfo[] {
    return this.topology.map((id) => this.node(id));
  }

  isCompletePipeline(): boolean {
    const nodes = this.pipeline();
    if (nodes.length === 0) return false;
    if (nodes.some((n) => n.state !== NodeState.Active)) return false;
    let expect = nodes[0].layerStart;
    for (const n of nodes) {
      if (n.layerStart !== expect) return false;
      expect = n.layerEnd + 1;
    }
    return true;
  }

  heartbeat(nodeId: string, t: number): void {
    this.node(nodeId).lastHeartbeat = t;
  }

  staleNodes(now: number, timeout: number): string[] {
    const out: string[] = [];
    for (const [id, n] of this.nodes) {
      if (n.state === NodeState.Active && now - n.lastHeartbeat > timeout) out.push(id);
    }
    return out;
  }

  recordTokens(streamId: string, tokenIds: number[]): void {
    const cur = this.tokenHistory.get(streamId) ?? [];
    cur.push(...tokenIds);
    this.tokenHistory.set(streamId, cur);
  }

  historyOf(streamId: string): number[] {
    return [...(this.tokenHistory.get(streamId) ?? [])];
  }

  dropStream(streamId: string): void {
    this.tokenHistory.delete(streamId); // freed on completion (spec §8 retention)
  }

  // The drain/retry/reassign loop's policy selection (spec §5.1).
  selectRecovery(nodeId: string, warmSpares: number, spareVersion: string): { policy: "reassign" | "rebuild"; reason: string } {
    const node = this.node(nodeId);
    if (warmSpares <= 0) return { policy: "rebuild", reason: "no warm spare available" };
    if (node.version !== spareVersion) {
      return { policy: "rebuild", reason: `version skew (${node.version} != ${spareVersion})` };
    }
    return { policy: "reassign", reason: "warm spare available, versions match" };
  }

  noteEvictionWarning(nodeId: string): void {
    this.setState(nodeId, NodeState.Draining);
  }
}
