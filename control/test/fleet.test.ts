import { describe, expect, it } from "vitest";
import { Fleet, FleetError, MAX_HISTORY_PER_STREAM, NodeState } from "../src/fleet";

function activate(f: Fleet, id: string) {
  for (const s of [NodeState.Staging, NodeState.Warm, NodeState.Loading, NodeState.Active]) {
    f.setState(id, s);
  }
}

function threeNode(): Fleet {
  const f = new Fleet();
  f.registerNode("n0", 0, 0, 11, "v1");
  f.registerNode("n1", 1, 12, 23, "v1");
  f.registerNode("n2", 2, 24, 35, "v1");
  for (const id of ["n0", "n1", "n2"]) activate(f, id);
  f.setTopology(["n0", "n1", "n2"]);
  return f;
}

describe("Fleet", () => {
  it("registers nodes and reports a complete pipeline", () => {
    const f = threeNode();
    expect(f.pipeline().map((n) => n.nodeId)).toEqual(["n0", "n1", "n2"]);
    expect(f.isCompletePipeline()).toBe(true);
  });

  it("detects a layer gap in the topology", () => {
    const f = new Fleet();
    f.registerNode("n0", 0, 0, 11, "v1");
    f.registerNode("n1", 1, 13, 23, "v1"); // gap: layer 12 missing
    for (const id of ["n0", "n1"]) activate(f, id);
    f.setTopology(["n0", "n1"]);
    expect(f.isCompletePipeline()).toBe(false);
  });

  it("rejects illegal state transitions", () => {
    const f = new Fleet();
    f.registerNode("n0", 0, 0, 11, "v1");
    expect(() => f.setState("n0", NodeState.Active)).toThrowError(FleetError);
  });

  it("flags stale heartbeats", () => {
    const f = threeNode();
    for (const id of ["n0", "n1", "n2"]) f.heartbeat(id, 100);
    f.heartbeat("n1", 50);
    expect(f.staleNodes(110, 30)).toEqual(["n1"]);
    expect(f.staleNodes(110, 120)).toEqual([]);
  });

  it("records and drops durable token-history", () => {
    const f = new Fleet();
    f.recordTokens("s", [1, 2, 3]);
    f.recordTokens("s", [4, 5]);
    expect(f.historyOf("s")).toEqual([1, 2, 3, 4, 5]);
    f.dropStream("s");
    expect(f.historyOf("s")).toEqual([]);
  });

  it("selects the recovery policy", () => {
    const f = threeNode();
    expect(f.selectRecovery("n1", 1, "v1").policy).toBe("reassign");
    expect(f.selectRecovery("n1", 0, "v1").policy).toBe("rebuild");
    expect(f.selectRecovery("n1", 1, "v2").policy).toBe("rebuild"); // version skew
  });

  it("rejects invalid + overlapping layer ranges (L5)", () => {
    const f = new Fleet();
    expect(() => f.registerNode("bad", 0, 10, 4, "v1")).toThrowError(FleetError); // end < start-1
    f.registerNode("n0", 0, 0, 11, "v1");
    expect(() => f.registerNode("n1", 1, 6, 17, "v1")).toThrowError(FleetError); // overlaps n0
    f.registerNode("tail", 2, 24, 23, "v1"); // 0-layer lm_head tail — allowed (no overlap)
  });

  it("seeds lastHeartbeat when a node first goes ACTIVE (L4)", () => {
    const f = new Fleet();
    f.registerNode("n0", 0, 0, 11, "v1"); // lastHeartbeat defaults to 0
    for (const s of [NodeState.Staging, NodeState.Warm, NodeState.Loading]) f.setState("n0", s);
    f.setState("n0", NodeState.Active, 500);
    expect(f.staleNodes(600, 1000)).toEqual([]); // not stale: 600 - 500 < 1000 (without L4 it'd be 600-0)
  });

  it("caps per-stream history (M6)", () => {
    const f = new Fleet();
    f.recordTokens("s", [1, 2, 3]);
    expect(() => f.recordTokens("s", new Array(MAX_HISTORY_PER_STREAM).fill(0))).toThrowError(FleetError);
    expect(f.historyOf("s")).toEqual([1, 2, 3]); // overflow rejected, history intact for replay
  });

  it("rate-limits per key in a fixed window (M7)", () => {
    const f = new Fleet();
    for (let i = 0; i < 3; i++) expect(f.checkRate("k", 3, 1000, 100).allowed).toBe(true);
    expect(f.checkRate("k", 3, 1000, 100).allowed).toBe(false); // 4th in-window
    expect(f.checkRate("k", 3, 1000, 2000).allowed).toBe(true); // new window
  });
});
