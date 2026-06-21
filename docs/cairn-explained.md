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

## The two things we proved today (the heart of it)

1. **Split-correctness** — a model split across machines produces the *exact same answer*, word for word,
   as the un-split model. (If splitting changed the answer, the whole idea would be worthless.)
2. **Replay-rebuild** — when a machine is lost, a fresh one replays the conversation and resumes
   *identically*. (This is the recovery guarantee that makes cheap spot GPUs safe.)

Both proven on a real GPU today.

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
- Warm spare
- Recovery: **"reassign"** vs **"rebuild"**
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
