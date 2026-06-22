# Cairn — explained simply

A plain-English primer on what this project is and every technical term that comes up. No background
assumed. (Tip: paste this into a fresh Claude chat and say *"I'm not technical — explain each of these
to me one at a time, simply, starting from the basics,"* then walk the glossary.)

---

## What the project is, in one breath

Cairn makes big AI models run on **cheap, unreliable rented GPUs** and keep running even when those GPUs
get yanked away — so you get **cheap prices with expensive-grade reliability**. The goal is to prove it
works, measure how much money it saves, and publish it (a research paper + open-source code).

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
  in and **rebuild** the lost work, so the line keeps flowing. Result: **spot price, on-demand-grade
  reliability.**
- We saw the exact problem live today: one of our rented GPUs got reclaimed mid-work after ~50 minutes.
  Surviving that is the entire point of Cairn.

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
3. **Live warm-spare recovery — the keystone (proven 2026-06-22).** The two pieces above, combined into
   the real thing: a model running across **separate rented machines**, one of them **killed mid-sentence**
   (simulating Amazon yanking a spot GPU), and the system swaps in the warm spare, rebuilds the lost
   memory, and finishes — **word-for-word identical** to a run where nothing died. Proven twice: on **one
   machine** (recovery took **0.58 seconds**) and across **3 separate machines over the network**
   (**38.9 seconds** — see the warm-up note). This is the entire promise of Cairn, demonstrated on real
   hardware, not a simulation.

## The warm-up lesson (from the cross-box recovery)

Recovering on one machine took 0.58 seconds; across separate machines it took 38.9. Almost all of that
gap was **not** the recovery itself — it was the spare machine doing its *first-ever* calculation, which
forces a one-time ~38-second compile of its GPU math routines (see *flashinfer / JIT* in the glossary).
The spare had the model **loaded** but had never actually "turned the engine over." The lesson: a
**"warm spare" has to be warm all the way through** — it should do one throwaway calculation at startup so
it's instant when it's actually needed.

**And it works.** We made every machine do one throwaway calculation at startup (moving that ~38-second
compile to load time), and the cross-box recovery dropped from **38.9 seconds to 0.023 seconds** — a
~1,700× speed-up, same word-for-word-identical answer. So the real story is now: a machine dies
mid-sentence, and the system is back to producing the *identical* answer in **23 milliseconds**.

---

## Glossary — the concepts to explain, one by one

**AI / model basics**
- LLM · "model" · "inference"
- Layers / transformer
- Tokens / tokenizer
- KV cache
- Attention — plus the fancy variants **MLA** and **DSA**
- Quantization — plus the variants **MXFP4 · NVFP4 · FP8**
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
  better. We measured **0.58s** (one machine) and **38.9s** (across machines, cold spare).
- Control plane (a small always-on coordinator; ours runs on Cloudflare)

**Cloud / hardware**
- AWS · EC2 · "instance"
- **Spot vs on-demand** instances
- GPU models: **L4**, **Blackwell** ("sm_89 / sm_120" = GPU generations)
- SkyPilot (the tool that rents, launches, and tears down the GPUs)
- Region (we use Spain and Seoul) · VPC

**The models we're targeting**
- gpt-oss-120b — our "proof" model
- GLM-5.2 · DeepSeek-V4-Flash — the "headline" models (newer, fancier)
- Llama-3.1 · Qwen · GPT-NeoX — used to compare against other researchers' results

**The money question**
- Cost per (million) tokens
- Spot-vs-on-demand savings %, done honestly (passing the "smell test")
- SpotServe · KevlarFlow — prior research we benchmark ourselves against

**The unglamorous engineering that ate today**
- Python "dependency hell" / **ResolutionImpossible** (why software versions fight each other)
- The **"box image"** (the exact set of software installed on a GPU machine)
- The cost ledger · tags · the live HTML report
- **Cheap-first** — prove the logic on ONE cheap machine (~$0.16/hr) before the expensive multi-machine
  run (~$0.48/hr). It paid off literally: the one cheap machine surfaced three separate bugs that would
  have been slow + costly to find on the full setup.
- The **gang scheduler** (the rental tool, SkyPilot, treats several rented machines as one group — if one
  "fails," it assumes the whole job failed and tears the group down). Our test *deliberately* kills a
  machine, which looked like a failure, so we had to teach the tool that *this* death is expected.
- An **environment-variable leak** — a kill-switch meant for ONE machine ("die after 4 steps," our test
  trigger) was accidentally inherited by ALL the machines, so they all died. Fixed by being explicit about
  which machine gets the setting. (A classic "a setting meant for one thing quietly affected everything.")

---

## The honest open risks (worth asking about later)

- The **headline models** (GLM / DeepSeek) combine newer math (DSA / MLA), newer number formats (NVFP4),
  and newer chips (Blackwell). Each is unproven for us and could be real work.
- The big **cost-savings claim is not allowed to be made** until we run the real multi-GPU fleet and read
  the actual AWS bill — anything before that is a guess, not a result.

---

## A one-line "why should anyone care"

If Cairn works, anyone can serve large models at ~spot prices without the usual spot unreliability — a
meaningful cost cut for AI serving, given away as open source.
