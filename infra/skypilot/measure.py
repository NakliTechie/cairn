#!/usr/bin/env python3
"""On-box probe — measure the "resolve on real hardware" numbers and emit config-ready values.

The fit (`cairn_scheduler`) reads three hardware numbers from `configs/<model>.yaml` that are
DOCUMENTED PLACEHOLDERS until measured on the real card:

    pool.gpu_vram_bytes                 usable VRAM the driver actually exposes (< the marketing total)
    overheads.framework_overhead_bytes  CUDA context + cuBLAS/cuDNN + torch/SGLang fixed reserve (no model)
    overheads.activation_buffer_bytes   per-stage activation + scratch high-water above the weights

Run ON THE BOX (GPU-only). Prints a YAML block to paste over the placeholders, plus the raw
breakdown so the numbers are auditable.

    python skypilot/measure.py                                   # VRAM + framework floor (no model needed)
    python skypilot/measure.py --model Qwen/Qwen2.5-0.5B-Instruct --seq 2048   # + activation buffer

Caveats (read before trusting the activation number):
  * framework_overhead and gpu_vram are SGLang-independent and solid here.
  * activation_buffer is measured via a transformers forward — a BALLPARK. The authoritative per-stage
    number comes from instrumenting the real SglangNodeRuntime.forward (paged-KV + CUDA graphs differ
    from a dense transformers forward). Use this to replace the placeholder, then refine from a live
    run (e.g. heartbeat VRAM deltas during test_sglang_split.py).
  * Point --model at a model that FITS on the card (a small model, or one stage's weights) — a large
    checkpoint will not load whole on a single card; VRAM + framework floor don't need a model at all.
"""
from __future__ import annotations

import argparse
import sys


def _gib(n: int) -> str:
    return f"{n / 2**30:.2f} GiB"


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure the hardware numbers for the Cairn fit.")
    ap.add_argument("--device", default="cuda:0", help="CUDA device (default cuda:0)")
    ap.add_argument("--model", default=None, help="HF id to measure the activation buffer against (optional)")
    ap.add_argument("--dtype", default="float16",
                    help="torch dtype for the model load (default float16)")
    ap.add_argument("--seq", type=int, default=2048, help="representative prefill length (default 2048)")
    ap.add_argument("--batch", type=int, default=1, help="representative batch / concurrent streams (default 1)")
    args = ap.parse_args()

    try:
        import torch
    except Exception as e:  # pragma: no cover
        print(f"torch not importable: {e}\nThis is a GPU box tool — run it on the box.", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("no CUDA — run this on the GPU box (it measures the real card).", file=sys.stderr)
        return 2

    idx = int(args.device.split(":")[1]) if ":" in args.device else 0
    dev = f"cuda:{idx}"
    props = torch.cuda.get_device_properties(idx)
    print(f"# device: {props.name}  (cc {props.major}.{props.minor})")

    # --- usable VRAM: what the driver actually exposes (already < nominal; the L4 reports ~23 GiB) ---
    free0, total = torch.cuda.mem_get_info(idx)
    print(f"# total usable VRAM: {_gib(total)}   free at start: {_gib(free0)}")

    # --- framework floor: force the CUDA context + cuBLAS/cuDNN, then read the fixed cost (no model) ---
    torch.cuda.reset_peak_memory_stats(idx)
    dtype = getattr(torch, args.dtype)
    warm = torch.randn(512, 512, device=dev, dtype=dtype)
    _ = (warm @ warm)                       # forces cuBLAS handle load into the context
    torch.cuda.synchronize(idx)
    free1, _ = torch.cuda.mem_get_info(idx)
    allocated = torch.cuda.memory_allocated(idx)        # our warmup tensors (negligible)
    framework_overhead = (total - free1) - allocated    # context + cuBLAS/cuDNN + reserve, minus our tensors
    del warm
    print(f"# framework floor: used={_gib(total - free1)}  (minus warmup tensors {_gib(allocated)})"
          f"  reserved={_gib(torch.cuda.memory_reserved(idx))}")

    activation_buffer = None
    weights = None
    if args.model:
        print(f"# loading {args.model} ({args.dtype}) to measure the activation high-water ...")
        import torch as _t
        from transformers import AutoModelForCausalLM
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(idx)
        base = torch.cuda.memory_allocated(idx)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()
        weights = torch.cuda.memory_allocated(idx) - base
        torch.cuda.reset_peak_memory_stats(idx)
        vocab = int(getattr(model.config, "vocab_size", 32000))
        ids = _t.randint(0, vocab, (args.batch, args.seq), device=dev)
        with _t.no_grad():
            model(ids, use_cache=True)
        torch.cuda.synchronize(idx)
        peak = torch.cuda.max_memory_allocated(idx)
        activation_buffer = peak - (base + weights)
        print(f"# weights: {_gib(weights)}   forward peak above weights "
              f"(batch={args.batch}, seq={args.seq}): {_gib(activation_buffer)}")

    # --- emit config-ready YAML ---
    print("\n# ---- paste into configs/<model>.yaml (measured on this card) ----")
    print("pool:")
    print(f"  gpu_vram_bytes: {total}            # {_gib(total)} usable ({props.name})")
    print("overheads:")
    print(f"  framework_overhead_bytes: {framework_overhead}   # {_gib(framework_overhead)} — CUDA ctx + cuBLAS/cuDNN")
    if activation_buffer is not None:
        print(f"  activation_buffer_bytes: {activation_buffer}     # {_gib(activation_buffer)} "
              f"@ batch={args.batch} seq={args.seq} (transformers ballpark — refine from a live SGLang run)")
    else:
        print("  # activation_buffer_bytes: <pass --model to measure> (transformers ballpark; "
              "refine from a live SGLang run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
