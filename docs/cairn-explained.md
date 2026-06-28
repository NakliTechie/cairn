# Cairn — explained simply

A plain-English primer on what this project is and every technical term that comes up. No background
assumed. (Tip: paste this into a fresh Claude chat and say *"I'm not technical — explain each of these
to me one at a time, simply, starting from the basics,"* then walk the glossary.)

---

## What the project is, in one breath

Cairn makes big AI models run on **cheap, unreliable rented GPUs** and keep running even when those GPUs
get yanked away — so you get **cheap prices with expensive-grade reliability**. It is an open-source
artifact and a research paper, not a commercial product: the code and the measurements are the deliverable.

## The core idea (an analogy)

- A big AI model is too large to fit on one GPU (the chip that runs AI). So you **cut the model into
  chunks** and run each chunk on a different GPU, passing the work down the line like an **assembly
  line**. (The jargon for this is a *pipeline*.)
- Renting GPUs from Amazon (AWS) is expensive. But Amazon sells its spare capacity — called **"spot"**
  GPUs — for roughly **80% less**. The catch: Amazon can **take a spot GPU back with about 2 minutes'
  warning, at any time**.
- Normally, if one GPU in your assembly line vanishes, the whole line breaks — so people don't dare use
  cheap spot GPUs for this.
- **Cairn's trick:** keep a **spare** GPU warm and ready; when one disappears, instantly swap the spare
  in and **rebuild** the lost work, so the line keeps flowing. Crucially, this needs **no cross-machine
  coordination** during the swap — the spare rebuilds the lost memory by replaying the conversation log,
  so the blast radius of any one machine dying is just **1 out of N**. Result: **spot price,
  on-demand-grade reliability.**
- This is not theoretical: in a live run a rented GPU was reclaimed mid-work, and the system survived it
  and finished the answer unchanged. Surviving that is the entire point of Cairn.

## How an AI model actually runs (so the rest makes sense)

- A model is a tall stack of **layers** (math steps). "Splitting the model" = giving different layers to
  different machines.
- Each machine runs its layers and passes a blob of numbers (the **"hidden state"**) to the next machine.
- The model writes its answer one **token** (≈ a word-piece) at a time.
- While writing, it keeps a short-term memory of the conversation so far, called the **"KV cache."** If a
  machine dies you lose its KV cache — but you can **rebuild it by replaying the conversation**. (We
  proved this works; it's called *replay-rebuild*.)

## What we've proved (the heart of it)

1. **Split-correctness** — a model split across machines produces the *exact same answer*, word for word,
   as the un-split model. (If splitting changed the answer, the whole idea would be worthless.)
2. **Replay-rebuild** — when a machine is lost, a fresh one replays the conversation and resumes
   *identically*. (This is the recovery guarantee that makes cheap spot GPUs safe.)
3. **Live warm-spare recovery — the keystone.** The two pieces above, combined into the real thing: a
   frontier model (**DeepSeek-V4-Flash**, see below) running pipeline-split across **separate rented
   machines**, one of them **killed mid-sentence** (simulating Amazon yanking a spot GPU), and the system
   swaps in the warm spare, rebuilds the lost memory, and finishes — **bit-for-bit identical** to a run
   where nothing died, **with zero dropped tokens.** This is the entire promise of Cairn, demonstrated on
   real hardware, not a simulation. The measured recovery times (how long from "a machine died" to
   "answers flowing again", i.e. MTTR):

   - **0.93 seconds** when a machine is killed abruptly mid-decode,
   - **3.26 seconds** when the machine drains gracefully (the polite ~2-minute eviction warning case),
   - **0.023 seconds (23 milliseconds)** when the spare is **pre-warmed** — the fast path.

4. **Multi-streaming — keeping the cheap GPUs busy.** A pipeline serving one request at a time wastes
   most of the GPUs (only one machine in the line works at a time). Running several requests through the
   line at once fills the gaps. Measured throughput climbed **6.1 → 12.4 → 24.5 → 24.9 tokens/second**
   as we ran **1, 2, 4, then 8 concurrent streams** — proof the idea actually fills the pipeline.

## The pre-warming lesson (why 0.023s is the number that matters)

An early cross-machine recovery once took ~39 seconds, and almost all of that gap was **not** the
recovery itself — it was the spare machine doing its *first-ever* calculation, which forces a one-time
compile of its GPU math routines (see *flashinfer / JIT* in the glossary). The spare had the model
**loaded** but had never actually "turned the engine over." The lesson: a **"warm spare" has to be warm
all the way through** — it should do one throwaway calculation at startup so it's instant when it's
actually needed.

**And it works.** Making every machine do one throwaway calculation at startup moves that compile to
load time, and the pre-warmed recovery lands at **0.023 seconds** — same bit-for-bit-identical answer.
So the real story is: a machine dies mid-sentence, and the system is back to producing the *identical*
answer in **23 milliseconds**.

---

## Glossary — the concepts to explain, one by one

**AI / model basics**
- LLM · "model" · "inference"
- Layers / transformer
- Tokens / tokenizer
- KV cache
- Attention — plus the fancy variants **MLA** and **DSA**
- Quantization — plus the variants **FP8** (what Cairn runs) · MXFP4 · NVFP4
- MoE (mixture-of-experts)
- Greedy decoding

**The software that runs a model on a GPU**
- **SGLang** (the inference engine we use)
- **flashinfer** (the fast GPU math SGLang relies on; "JIT compiling" kernels)
- Pipeline parallelism (splitting layers across GPUs)

**Cairn's own design**
- Block / stage / shard (one chunk of layers on one machine)
- The **"wire"** (how machines send hidden states to each other, encrypted)
- Scheduler
- Warm spare — and **pre-warming** (making the spare do a throwaway calculation at startup so it's
  *truly* warm, not just loaded — the cross-box warm-up lesson)
- Recovery: **"reassign"** vs **"rebuild"**
- **MTTR** (mean time to recovery) — how long from "a machine died" to "answers flowing again." Lower is
  better. Measured live: **0.93s** (abrupt kill), **3.26s** (graceful drain), **0.023s** (pre-warmed spare).
- Control plane (a small always-on coordinator; ours runs on Cloudflare)

**Cloud / hardware**
- AWS · EC2 · "instance"
- **Spot vs on-demand** instances
- GPU models: **L4** (the cheap card used for early proofs), **RTX PRO 6000 Blackwell**
  ("sm_89 / sm_120" = GPU generations; the headline run uses Blackwell / sm_120)
- SkyPilot (the tool that rents, launches, and tears down the GPUs)
- Region (the headline fleet runs in AWS us-east-2 / Ohio) · VPC

**The model we serve**
- **DeepSeek-V4-Flash** — the headline model: 671B parameters (a mixture-of-experts design — 256
  experts, 6 active + 1 shared per token — across 43 layers), run as **FP8** (~294 GB), MIT-licensed.
  This is what Cairn pipeline-splits 4 ways across Blackwell spot GPUs.
- **GLM-5.2** — a possible future config (Cairn treats "the model" as just configuration, so adding one
  is a new config file, not new code).

**The money question**
- Cost per (million) tokens
- Spot-vs-on-demand savings %, done honestly (passing the "smell test")
- SpotServe · KevlarFlow — prior research we benchmark ourselves against

**The unglamorous engineering (lessons worth knowing)**
- Python "dependency hell" / **ResolutionImpossible** (why software versions fight each other)
- The **"box image"** (the exact set of software installed on a GPU machine)
- The cost ledger · tags · the live HTML report
- **Cheap-first** — prove the logic on ONE cheap machine before the expensive multi-machine run. It pays
  off: a single cheap machine surfaces most bugs that would be slow and costly to find on the full setup.
- The **gang scheduler** (the rental tool, SkyPilot, treats several rented machines as one group — if one
  "fails," it assumes the whole job failed and tears the group down). A recovery test *deliberately* kills
  a machine, which looks like a failure, so the tool has to be taught that *this* death is expected.
- An **environment-variable leak** — a kill-switch meant for ONE machine ("die after N steps," a test
  trigger) was accidentally inherited by ALL the machines, so they all died. Fixed by being explicit about
  which machine gets the setting. (A classic "a setting meant for one thing quietly affected everything.")

---

## The honest open risks

- The **headline model** combines newer math (DSA / MLA), an FP8 number format, and newer chips
  (Blackwell sm_120), served through a community patch to the inference engine. Getting all of that to
  cooperate was real work, and the same combination is the part most likely to need care on a fresh setup.
- The big **cost-savings claim is held back** until a real multi-GPU fleet runs and the actual AWS bill is
  read — anything before that is a guess, not a result. The recovery and throughput results above are
  live-measured; the headline dollars-per-token number is not yet claimed.

---

## A one-line "why should anyone care"

If Cairn works, anyone can serve large models at ~spot prices without the usual spot unreliability — a
meaningful cost cut for AI serving, given away as open source.
