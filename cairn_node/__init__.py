"""cairn_node — the per-node serving layer for a REAL multi-node Cairn pipeline (the binary the fleet
runs, referenced by infra/skypilot/cairn-block.sky.yaml). Built 2026-06-21 atop the proven
SglangNodeRuntime block-forward + the tested sealed wire (fork/shard/{wire,transport}.py).

  - `serve`    — one node's wire loop: recv hidden from the prev node, run this block's forward, send
                 to the next. Separate process per node (each sglang ModelRunner needs its own process —
                 the global tensor-parallel group can't be init'd twice).
  - `pipeline` — the driver/orchestrator: spawn one `serve` per stage, wire them, greedy-decode.

v1.0 minimal: topology is passed on argv (no control-plane registration / recovery / heartbeat yet —
that's Chunk C). The point now is to prove split==single-ref + replay-rebuild ACROSS PROCESSES, over
the real wire, which the in-process harness can't (two ModelRunners can't share a process)."""
