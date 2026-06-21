# Cairn — objectives

_What we're building, the end state, and the one honest claim. (2026-06-20.)_

## In one line
Serve a **frontier open LLM that no single affordable GPU can hold** on a fleet of **cheap, individually-reclaimable single-GPU spot instances**, and keep it serving **through spot preemption** via a minimal hot-swap recovery loop — **on-demand-grade reliability at roughly spot cost.**

## Targets
- **Proof:** gpt-oss-120b (Apache-2.0) on g6/L4 (Spain) — the mechanism on a current open model.
- **Headlines (DSA → Blackwell g7e, 4-bit NVFP4, Seoul):** **GLM-5.2** (753B, the big flex, N≈6) and **DeepSeek-V4-Flash** (158B, MLA so tiny KV, the light/cheap one, N≈2) — both MIT.
- **Comparability (head-to-head vs the papers):** GPT-NeoX-20B (SpotServe), Llama-3.1-8B (KevlarFlow).
- Model-as-config (inv #3): each target is a YAML, never a code path.

## The contribution (honest scope)
Cairn adds **one thing**: a **hot-swap failover layer** for pipeline-parallel LLM serving on preemptible spot. It is a **systems/ops layer on top of existing serving** (SGLang, wrapped per-block) — **not** a new model, attention mechanism, or quantization. The specific primitive: **fixed blocks; lose one → swap it onto a pre-staged warm spare in seconds → rebuild only that block's KV by replaying the durable token history.** Blast radius 1/N.

Why it's worth doing:
- **It unlocks the spot discount; it doesn't add cost.** Spot is ~3–4× cheaper than on-demand but gets preempted; the failover is what lets you *use* that cheap capacity reliably. The headline is "≈ on-demand reliability at spot cost" (cf. SpotServe's **54% < on-demand**), not "a bit more than bare spot." The warm spare is the only standing overhead.
- **At frontier scale it's the difference between usable and unusable.** When a model needs many GPUs, one preemption kills the whole pipeline — and more GPUs means more preemptions. Hot-swap is mandatory there, not marginal.

## Positioning (builds on a published lineage)
This is **not new territory — and that's the validation.** Fault-tolerant LLM serving on preemptible instances is an active, peer-reviewed area:
- **SpotServe** (ASPLOS '24) — first LLM serving on preemptible instances; 54% < on-demand; recovers by **reparallelization + KV migration**.
- **Petals** (NeurIPS '23) — Cairn's architecture (blocks across nodes, reroute on loss), at internet/volunteer scale.
- **KevlarFlow** (2026) — recovers by **background KV replication**; 20× MTTR.

Cairn's differentiator is the **combination**, not any single mechanism: a model **big enough that the split is mandatory** × **production multi-stream throughput** × a **minimal recovery primitive** (replay the dead block, no full reparallelization) × **commodity single-VPC spot**. The bet to prove: the minimal primitive stays competitive with reparallelization/replication at frontier scale.

## End state — the deliverables
1. **A working system**, proven up the cost-ladder (rungs 0→5: sim → 1-GPU → 2-GPU split + recovery → proof fleet → frontier headline fleet → resiliency).
2. **Benchmark metrics** — the gate artifacts: cost/token vs on-demand, MTTR, per-stage occupancy, split-correctness, induced-interruption timeline — including **head-to-head vs SpotServe (54%) and KevlarFlow (MTTR) on their exact models** (GPT-NeoX-20B, Llama-3.1-8B).
3. **A paper** — on GitHub (arXiv if accepted). Thesis: *a minimal fixed-block hot-swap recovery primitive that keeps a frontier LLM serving through spot preemption — competitive with reparallelization (SpotServe) and KV-replication (KevlarFlow) at a fraction of the complexity, validated head-to-head on their models.*
4. **An open-source repo** people take away and run their own fault-tolerant spot serving in production.

## License & stance
- **Open source.** Clean licenses throughout — Apache-2.0 (gpt-oss-120b, GPT-NeoX-20B), MIT (GLM-5.2, DeepSeek-V4-Flash); deliberately no OPT/LLaMA-1 (non-commercial/leaked). Model-as-config keeps it model-agnostic.
- **Monetization = consulting, not a product.** The work is a credibility/portfolio piece that lands engagements — not a SaaS to sell. _(Supersedes the earlier "commercial edge-first, not in the public portfolio" framing: this is public and open.)_

## Non-goals
- Not a new model / attention / quantization — a serving-reliability layer.
- Not a closed product.
- Not a claim to have invented distributed inference on spot (that's SpotServe / Petals) — Cairn is a distinct, simpler recovery primitive at frontier scale.
