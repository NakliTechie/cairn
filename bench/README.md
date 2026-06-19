# bench/ — benchmark harness + gate artifacts

The v1.0 "human face" (handoff §12): the benchmark harness + the metrics/logs. Each
gate artifact is committed evidence here (handoff §8, spec §9).

- **v1.0:** reliability log (N×g6 clean completions) · correctness (split == single
  reference, greedy) · induced-interruption recovery timeline · crypto self-test.
- **v1.1:** per-stage occupancy ≥ floor · cost/token vs baseline · **cross-stream
  KV-isolation** (the hard gate).
- **v1.2:** endurance through real spot interruptions · thrash bound · eviction telemetry.

The orchestration gates (split-correctness, KV-isolation, replay-rebuild) are first
proven in sim under `scheduler/tests/` (rung 1); this dir holds their **real-fleet**
counterparts. **Status: not started.**
