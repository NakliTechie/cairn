# Cairn — rung-1 simulation gate artifacts

**Model:** `llama-3.1-8b`  ·  **Fit:** N=4 stages, layers/stage=[8, 8, 8, 8], K_target=16

> Rung-1 SIMULATION gates (no GPU). Real-fleet gates need the Shard fork + a g6 pool.

**All hard sim-gates pass: ✅ YES**

| Gate | Result | Detail |
|------|--------|--------|
| v1.0 reliability | ✅ pass | 20/20 clean completions |
| v1.0 correctness (split == single reference) | ✅ pass | split N=4 token-for-token == single-node reference |
| v1.0 induced interruption (kill mid-decode → reassign → resume) | ✅ pass | stage 2/4 reassign; output uncorrupted; resumed at t=51.3 |
| v1.1 utilisation (per-stage occupancy under K streams) | ✅ pass | single-stream=0.255 (≈1/N), K=N occupancy=0.982 (floor 0.8) |
| v1.1 cross-stream correctness (HARD gate — the silent bug) | ✅ pass | all 6 concurrent streams token-for-token == their solo runs |
| v1.1 cost/token (ILLUSTRATIVE — real $/hr + throughput from §12) | ℹ️ illustrative | occupancy-driven; numbers illustrative, not a $ claim (spec §6 'pitch it narrow') |
