# Cairn — cost-claim methodology (the smell-test spec)

The headline at the end of the ladder is one sentence:

> Serving **model M** at **throughput T** and **on-demand-grade reliability R**, Cairn on
> interruption-tolerant **spot** costs **\$C_spot per 1M output tokens**, versus **\$C_od per 1M** for the
> same model on the **same instance type on-demand** — a **(1 − C_spot/C_od)** saving, taken from the
> **actual tagged AWS bill**, not an estimate.

This doc fixes how that number is computed so it survives scrutiny. The rule of thumb: **every overhead
that makes spot cheap-but-annoying must land on the spot side of the ledger.** If a skeptic can say "but
you didn't count X," X belongs here.

---

## 1. The metric: cost per 1M output tokens, at equal throughput AND reliability

- **Per token, not per hour.** Hourly price hides throughput differences; cost/token normalizes them.
  `cost_per_1M = total_$ / (output_tokens / 1e6)`, measured over the same workload.
- **Equal throughput.** Both runs serve the SAME workload/trace at the SAME target tokens/s. If the spot
  run is slower (recovery gaps), that shows up as more node-hours per token — which is the honest penalty.
- **Equal reliability is a GATE, not a footnote.** The claim is "on-demand-grade reliability at ~spot
  cost." A cheaper-but-flakier run is **not** a win. The spot run must meet the reliability bar
  (completion rate ≥ on-demand's; P99 latency within an agreed tolerance; zero dropped streams) or the
  cost number is void. Report cost and reliability **together**, always.

## 2. Apples-to-apples (the comparison is rigged without these)

- **Same model**, same quantization, same context/workload trace.
- **Same instance type**: spot g6.xlarge vs **on-demand g6.xlarge** — NOT spot-cheap-instance vs
  on-demand-pricey-instance. Same silicon, same per-node throughput.
- **Same fit**: same N (active block count), same per-stage layer assignment. Only the *pricing* and the
  *recovery machinery* differ.
- **Same region/AZ** where possible (price and interruption rate are regional).

## 3. The spot cost — count ALL of it (from the tagged bill)

Taken from AWS Cost Explorer filtered by tag `cairn=true`, `cairn-purpose=spot-proof`, over the run window:

| Component | Counted? | Notes |
|---|---|---|
| Active spot node-hours × spot price **paid** | ✅ | actual price paid, not a quoted trough |
| **Warm-spare** spot node-hours × spot price | ✅ **(the #1 hidden cost)** | recovery's standing cost; v1.0 keeps 1 spare |
| On-demand floor (if any) × on-demand price | ✅ | v1.0 = 0 floor; count it if a rung adds one |
| Re-work after interruptions | ✅ implicitly | re-run tokens consume node-hours → already in the bill; don't also credit them as "served" |
| Control plane (CF Workers + Durable Objects) | ✅ | small but real; include for rigor |
| Storage (EBS for staged weights) + data transfer | ✅ | in-VPC egress ≈ \$0 (note it); EBS per node is real |

`C_spot = (all of the above) / (output_tokens / 1e6)`. **Output tokens = tokens actually delivered to
clients** — net of any re-runs (a replayed token is not a second delivered token).

## 4. The on-demand baseline — same work, no spare

- **N active nodes** (the same active count as the spot run) × on-demand hourly × wall-clock to serve the
  same workload at the same throughput. **No warm spare** (on-demand instances are not reclaimed, so the
  recovery machinery isn't needed) — this is the baseline's structural advantage and we grant it.
- + the same control-plane + storage (small).
- The on-demand price is **AWS-published (a fact)**; the node count + throughput come from the same fit.
- **Primary:** compute `C_od` from the published on-demand price × the equivalent node-hours. **Optional
  validation:** a short *actual* on-demand run (tag `cairn-purpose=ondemand-baseline`) to confirm the
  computed figure — cheap insurance that the number is real, not modeled.

## 5. Smell-test guardrails (each is a way to cheat; each has a rule)

1. **Hiding the warm spare** → it's a line item in §3. A 4-active+1-spare spot fleet is *five* nodes' spot
   cost, not four. (Even so, 5 × spot can be far below 4 × on-demand — that's the honest win.)
2. **No real interruptions** → a 10-min run in a <5%-interruption region may see **zero** evictions, making
   spot look free. **Run long enough to experience the region's actual interruption rate, OR inject
   interruptions at the measured rate** — and SAY which. Report the interruption count + the resulting
   re-work %. A spot claim with zero interruptions observed is not a recovery claim.
3. **Cherry-picked spot price** → use the **actual price paid** over a representative window; report the
   **spot/on-demand ratio** and the window. Note that the region was chosen for low interruption + good
   discount (eu-south-2/Spain) — disclose that, don't hide it.
4. **Different instances** → §2: same type, both sides.
5. **Double-counting throughput** → output tokens are *delivered* tokens; replayed/re-run tokens are cost,
   not extra output.
6. **Ignoring control-plane/storage** → included in §3 (small, but excluding them is sloppy).
7. **Reliability asterisk** → §1: no cost number without the paired reliability number.

## 6. When the claim is allowed to exist (the ladder)

- **rungs 0–3 (sim → single/2-GPU correctness, where we are now):** mechanism proven (split-correctness,
  KV-replay recovery). **NO cost claim yet** — single dev boxes are *our* spend, tracked in the launch
  ledger (`infra/skypilot/launchlog.py`), and are NOT the product number.
- **rungs 4–5 (Chunk C, the real fleet):** the v1.0/v1.1 gate runs produce the reliability evidence AND the
  tagged bill → only here do `C_spot` / `C_od` / the saving get filled in. The number is born from a real
  fleet run that experienced real (or representatively-injected) interruptions, not before.

## 7. The reconciliation (actual, not estimated)

- **Tags** make the bill self-segmenting: every instance carries `cairn=true` (+ `cairn-purpose`,
  `cairn-model`). Spot-proof runs vs on-demand-baseline runs get distinct `cairn-purpose` values, so
  `cost-report.sh` returns the two actual figures **directly from AWS**. (One-time: activate `cairn` as a
  cost-allocation tag in AWS Billing — ~24h to populate.)
- **Local ledger** (`launchlog.py`) mirrors every up/down with est cost → reconcile total spend against
  the tagged bill. Estimate vs actual should agree within the spot-price drift.

## 8. vs the prior art (same models, ≥ their rigor)

The comparability benchmark (configs in `configs/gpt-neox-20b.yaml`, `configs/llama-3.1-8b.yaml`) reruns
**SpotServe** (its 54% throughput-retention) and **KevlarFlow** (its ~20× MTTR) on **their exact models**,
under this methodology. Our claim must be at least as rigorous as theirs: same models, full cost
accounting above, reliability reported alongside cost. See `plan/prior-art.md` + `docs/cairn-objectives.md`.

---

## Worked example — HYPOTHETICAL, illustrative only (real numbers come from the Chunk-C run)

> ⚠ Placeholder arithmetic to show the *shape* of the honest calc. NOT a claim. Prices are illustrative;
> the actual figures come from the tagged bill of a real fleet run.

- Proof: DeepSeek-V4-Flash FP8, **4 active + 2 warm spares**, g7e.2xlarge, us-east-2.
- Run: 4h wall-clock; served W output tokens; **3 interruptions** (~8% re-work, in the node-hours).
- Spot price paid ≈ \$0.16/hr; on-demand g6.xlarge ≈ \$0.80/hr (CONFIRM both from the bill / pricing).
- **Spot** = (4 active + 1 spare) × 4h × \$0.16 + control-plane/storage ≈ \$3.20 + \$0.30 = **\$3.50**
- **On-demand** = 4 active × 4h × \$0.80 + \$0.30 = **\$13.10** (no spare)
- **Saving = 1 − 3.50/13.10 ≈ 73%** — *and only if* the spot run met the reliability bar (completion ≥
  on-demand, P99 within tolerance). The spare (a full extra node) is counted; same instance both sides;
  interruptions actually happened. That is the number that passes the smell test.
