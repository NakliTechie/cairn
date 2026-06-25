"""SglangNodeRuntime — the per-node block runtime (Cairn's rung-2/3 implementation).

⚠️ GPU-UNVERIFIED HEAD START. The structure, the architecture decision, and the
concretely-writable parts (heartbeat, KV bookkeeping, config) are here; the SGLang
block-forward integration is marked TODO and MUST be iterated on a real GPU with a
pinned SGLang version. Do not assume it runs as-is.

Architecture decision (load-bearing — record it):

  SGLang HAS native pipeline parallelism (`--pp-size`/`--nnodes`, NCCL between stages).
  **Cairn does NOT use it across nodes.** SGLang's PP is one coordinated launch whose
  stages hand off activations over NCCL collectives — which *hang or abort when any rank
  vanishes*. Our nodes are independent spot instances that get reclaimed mid-decode; the
  whole product is reassign-one-block + replay-KV + keep-serving (blast radius 1/N). A
  NCCL-coupled pipeline has blast radius = total. So each node runs SGLang loaded with
  ONLY its contiguous block of layers and exposes `forward(hidden, kv_meta) → hidden`;
  **Cairn's own wire/transport/scheduler/recovery stitch the stages** (spec §0 inv #1,
  §5; handoff §3). We keep SGLang's kernels/attention/paging/quant — not its transport.

Integration point (the rung-2 work): load a *layer-range subset* of the model and drive
a single block's forward with per-seq paged KV. Candidate paths, to settle on a GPU:
  (A) Drive SGLang's per-stage model runner directly (it already loads a layer subset
      for its own PP) but bypass its NCCL send/recv — feed our `hidden` in, take the
      stage output out. Deepest, most reuse.
  (B) A thin custom runner that loads the layer slice + SGLang's RadixAttention/paged-KV
      for this block's cache. More code, fewer SGLang-internal assumptions.
  (C) Rung-2 stepping stone: a transformers reference block forward (no SGLang) to prove
      split-correctness + KV-replay cheaply, then swap to (A)/(B) for performance.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .node import LayerRange, NodeRuntime

try:  # keep this module importable on CPU (for structural checks) — deps load lazily
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


class SglangNotAvailable(RuntimeError):
    pass


class _CairnFakePPGroup:
    """Stand-in for sglang's pipeline-parallel GroupCoordinator (path-β loader).

    sglang's per-model `make_layers(..., pp_rank, pp_size)` reads from `get_pp_group()` and
    uses `rank_in_group`/`world_size` to compute the layer slice for THIS rank. Layers outside
    the slice become `PPMissingLayer` placeholders (no weights resident in VRAM).

    Cairn's invariant (2026-06-19): no NCCL between nodes. We get the loader-side partition by
    monkey-patching `parallel_state.get_pp_group` to return THIS fake. sglang's distributed init
    stays world_size=1 (no inter-node TCP rendezvous, no NCCL groups). Forward goes through
    Cairn's own wire (per-layer driven), so the unused collective methods below are never called.
    Stubbed regardless so a mistaken call crashes loudly rather than hanging on a real NCCL op.
    """

    def __init__(self, rank: int, size: int) -> None:
        self.rank_in_group = rank
        self.world_size = size
        self.ranks = list(range(size))
        self.device_group = None
        self.cpu_group = None

    @property
    def is_first_rank(self) -> bool:
        return self.rank_in_group == 0

    @property
    def is_last_rank(self) -> bool:
        return self.rank_in_group == self.world_size - 1

    @property
    def next_rank(self) -> int:
        return self.ranks[(self.rank_in_group + 1) % self.world_size]

    @property
    def prev_rank(self) -> int:
        return self.ranks[(self.rank_in_group - 1) % self.world_size]

    def _refuse(self, name: str):
        raise RuntimeError(
            f"_CairnFakePPGroup.{name}() called: Cairn drives layers directly, NCCL inter-stage "
            "collectives must never fire (would hang on a vanished spot node, spec inv §0)."
        )

    def all_reduce(self, *a, **kw):  return self._refuse("all_reduce")
    def all_gather(self, *a, **kw):  return self._refuse("all_gather")
    def broadcast(self, *a, **kw):   return self._refuse("broadcast")
    def send(self, *a, **kw):        return self._refuse("send")
    def recv(self, *a, **kw):        return self._refuse("recv")
    def barrier(self, *a, **kw):     pass     # tolerate barrier no-op; some init paths call it


def _set_pp_identity(group: Any, rank: int, size: int) -> None:
    """Override an sglang pp GroupCoordinator's *partition identity* in place (path-β core).

    sglang decides each box's layer slice from `get_pp_group().rank_in_group` / `.world_size`:
    `DeepseekV*Model.__init__` reads them and passes `pp_rank`/`pp_size` to `make_layers`, which
    calls `get_pp_indices` to compute `[start,end)`. We must make those read Cairn's `(k, N)`.

    Why mutate-in-place from a `load_model` hook (not replace `get_pp_group` before ModelRunner):
    `ModelRunner.__init__` runs `init_torch_distributed()` → `initialize_model_parallel(pp_size=1)`,
    which REBUILDS the real `_PP` at world_size=1 *before* the model is constructed — silently
    discarding any earlier patch (the 2026-06-24 maiden-run OOM: every box loaded the full model).
    So we mutate the REAL group object (keeping all its real methods) from inside a wrapper around
    `ModelRunner.load_model`, which runs AFTER that re-init and right before the model is built. The
    actual torch process group still has ONE rank → no NCCL collective ever crosses a box (Cairn's
    own wire drives the forward; `world_size` here is only READ for the loader's layer partition).
    """
    group.world_size = size
    group.rank_in_group = rank
    try:
        group.ranks = list(range(size))
    except Exception:  # pragma: no cover - defensive; .ranks shape varies across sglang versions
        pass
    # is_first_rank / is_last_rank: in the V4-Blackwell sglang build these are @property values
    # derived from the GLOBAL torch rank (0) + ranks list — NOT from rank_in_group — so setting
    # rank_in_group above does NOT make them per-stage-correct (is_last_rank stays False on every
    # stage; GPU-confirmed 2026-06-25). They don't gate model CONSTRUCTION (embed/norm/lm_head are
    # built unconditionally — deepseek_v4.py L1189/1211/1374), only the load-time weight-skip gates,
    # which we handle separately in _install_load_weights_rank_fix(). This loop still helps on sglang
    # versions where they ARE plain attributes (then setattr makes them correct; on this build it's a
    # no-op because they're properties).
    for attr, val in (("is_first_rank", rank == 0), ("is_last_rank", rank == size - 1)):
        if not isinstance(getattr(type(group), attr, None), property):
            try:
                setattr(group, attr, val)
            except Exception:  # pragma: no cover
                pass


class _LoadTimeAllRanksPPGroup:
    """A pp-group view whose is_first_rank/is_last_rank both read True — installed ONLY for the
    duration of the model's load_weights call (see _install_load_weights_rank_fix).

    Why (GPU-confirmed 2026-06-25): sglang's DeepseekV4 load_weights skips embed weights when
    `not pp_group.is_first_rank` and skips ANY `.norm.` weight when `not pp_group.is_last_rank`
    (deepseek_v4.py ~L1814/1818). That `.norm.` gate assumes the only norm is the final
    `model.norm.weight`, but V4's sparse attention adds per-layer `…compressor.norm.weight` /
    `…indexer.compressor.norm.weight` (they also contain `.norm.`), which live on whatever stage
    owns the layer. Under Cairn's (k,N) layer-split, is_first/is_last_rank are @property values
    derived from the GLOBAL torch rank (0, world_size=1 — our slicing only sets rank_in_group/
    world_size), so is_last_rank is False on every stage → every stage drops its compressor.norm/
    indexer.compressor.norm (+ the final norm) → the strict "weights not initialized from
    checkpoints" RuntimeError.

    Safe because embed_tokens / norm / lm_head are built UNCONDITIONALLY on every stage
    (deepseek_v4.py L1189/1211/1374 — not pp-gated), so forcing both gates open is memory-neutral
    (the params are already allocated) and correct: each stage still loads only its sliced LAYERS'
    weights (that gate uses start/end_layer, untouched) PLUS the unconditional embed/norm/lm_head —
    harmless on the stages that don't use them, since Cairn's wire decides which stage embeds the
    tokens / applies the final norm+lm_head."""

    def __init__(self, real: Any) -> None:
        object.__setattr__(self, "_real", real)

    @property
    def is_first_rank(self) -> bool:
        return True

    @property
    def is_last_rank(self) -> bool:
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)


def _install_load_weights_rank_fix() -> None:
    """Patch DeepseekV4ForCausalLM.load_weights to view its pp_group as all-ranks-True for the
    call, so the per-stage embed/norm weight-skip gates don't drop V4's per-layer norms (see
    _LoadTimeAllRanksPPGroup). Idempotent; no-op if the V4 model class isn't importable (non-V4
    images / CPU-mock tests). The class patch is left installed; it only swaps pp_group for the
    duration of each load_weights call and restores it in a finally, so it never leaks."""
    try:
        import sglang.srt.models.deepseek_v4 as _m
    except Exception:  # pragma: no cover - only present in the V4-Blackwell image
        return
    cls = getattr(_m, "DeepseekV4ForCausalLM", None)
    if cls is None or getattr(getattr(cls, "load_weights", None), "_cairn_norm_gate_fix", False):
        return
    _orig_load_weights = cls.load_weights

    def load_weights(self, *a, **kw):
        real = getattr(self, "pp_group", None)
        if real is None or isinstance(real, _LoadTimeAllRanksPPGroup):
            return _orig_load_weights(self, *a, **kw)
        self.pp_group = _LoadTimeAllRanksPPGroup(real)
        try:
            return _orig_load_weights(self, *a, **kw)
        finally:
            self.pp_group = real

    load_weights._cairn_norm_gate_fix = True
    cls.load_weights = load_weights


def _install_post_load_weights_fix() -> None:
    """Patch DeepseekV4ForCausalLM.post_load_weights to SKIP PPMissingLayer placeholders.

    Why (GPU-confirmed 2026-06-25, the step right after the norm-gate fix): post_load_weights runs
    a V4 sparse-attention "APE hotfix" pass — `for layer in self.model.layers: layer.self_attn…`
    (deepseek_v4.py ~L1506). Under Cairn's (k,N) layer-split the OUT-OF-SLICE layers are weightless
    `PPMissingLayer` stubs (the very thing that proves slicing works) with no `self_attn`, so the
    unguarded `layer.self_attn` raises `AttributeError: 'PPMissingLayer' object has no attribute
    'self_attn'`. Upstream never hits this (single-stage runs have no PPMissingLayer); it's specific
    to V4 + multi-stage PP = Cairn. Faithful re-implementation of the short method with a
    `getattr(layer, 'self_attn', None)` guard (the hotfix only applies to the real layers in-slice).
    Idempotent; no-op if the V4 model class isn't importable."""
    try:
        import sglang.srt.models.deepseek_v4 as _m
    except Exception:  # pragma: no cover - only present in the V4-Blackwell image
        return
    cls = getattr(_m, "DeepseekV4ForCausalLM", None)
    if cls is None or getattr(getattr(cls, "post_load_weights", None), "_cairn_ppmissing_fix", False):
        return

    def post_load_weights(self, is_nextn=False, weight_names=None):
        if getattr(_m, "_FP8_WO_A_GEMM", False):
            self._setup_fp8_wo_a_scales(is_nextn)
        if is_nextn:
            return
        for layer in self.model.layers:
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:      # PPMissingLayer (out-of-slice) — no attn to hotfix
                continue
            if self_attn.compress_ratio != 0 and not self_attn.compressor.ape_converted:
                self_attn.compressor.apply_ape_hotfix()
            if self_attn.compress_ratio == 4 and not self_attn.indexer.compressor.ape_converted:
                self_attn.indexer.compressor.apply_ape_hotfix()

    post_load_weights._cairn_ppmissing_fix = True
    cls.post_load_weights = post_load_weights


class SglangNodeRuntime(NodeRuntime):
    """Serve one contiguous block of layers via SGLang (rung 2/3, GPU).

    PROVEN on a real L4 (2026-06-21): split == unsplit token-for-token (inv #1) over multi-step
    greedy decode, on SGLang's own flashinfer kernels + paged KV — by driving SGLang's per-layer
    forward directly (path A), with its NCCL pipeline-parallel transport bypassed (Cairn's wire
    stitches the stages instead; spec §0 inv #1, handoff §3).

    The seam:
      - `load_shard` builds an in-process SGLang `ModelRunner` (tp=1, pp=1 — no NCCL) that loads the
        model + a paged-KV pool + the flashinfer attention backend, and keeps a view of this block's
        layers `[start, end)`. SGLang's distributed state is GLOBAL per process, so all runtimes in
        ONE process SHARE one ModelRunner (`_SHARED`): a real node is its own process → its own
        runner; the single-GPU split test = N runtimes sharing one. The KV pool is layer-partitioned,
        so blocks never collide.
      - `forward` runs `layers[start:end]` over the incoming hidden (embedding the ids on stage 0),
        keeping this block's paged KV per seq via an SGLang `ScheduleBatch` (first call = prefill /
        extend, later calls = decode). SGLang uses the residual-stream pattern (each layer returns
        `(hidden, residual)`); at a non-tail boundary we FOLD them — `hidden + residual` — so the wire
        carries ONE tensor and the next stage resumes with `residual=None` (algebraically exact). The
        tail applies `norm` + `lm_head` → next-token logits.

    Path-β partial-layer load (for headline-class models that exceed per-box VRAM at full load):
      pass `cairn_pp_rank=k, cairn_pp_size=N` (or set `CAIRN_PP_{RANK,SIZE}` env vars). `load_shard`
      monkey-patches sglang's `get_pp_group()` to return a `_CairnFakePPGroup(k, N)` before
      ModelRunner construction; sglang's `make_layers` then loads ONLY layers
      `[k*L/N, (k+1)*L/N)` (the rest become weightless `PPMissingLayer`). sglang's own dist init
      still runs world_size=1 (no TCP rendezvous between Cairn boxes, no NCCL across nodes — the
      2026-06-19 invariant). `layer_range` MUST match the sglang partition; `load_shard` asserts.
    """

    _SHARED: Dict = {}   # (model, device, pp_rank, pp_size) -> ModelRunner; one per process

    def __init__(self, model: str, layer_range: LayerRange, device: str = "cuda:0",
                 quant=None, *, cairn_pp_rank: Optional[int] = None,
                 cairn_pp_size: Optional[int] = None,
                 cairn_pp_partition: Optional[list] = None) -> None:
        super().__init__(model, layer_range, device)
        self.quant = quant                       # None = the model's native dtype (don't force a quant)
        parts = device.split(":")                # tolerate "cpu"/"cuda"/"cuda:N" without crashing (L1)
        self.device_index = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        import os
        rank = cairn_pp_rank if cairn_pp_rank is not None else os.environ.get("CAIRN_PP_RANK")
        size = cairn_pp_size if cairn_pp_size is not None else os.environ.get("CAIRN_PP_SIZE")
        self.cairn_pp_rank = int(rank) if rank is not None else None
        self.cairn_pp_size = int(size) if size is not None else None
        # Optional EXPLICIT layer partition (per-stage layer counts, e.g. [15,14,14] for 43 layers).
        # Pins sglang's slice boundaries to Cairn's via SGLANG_PP_LAYER_PARTITION, instead of relying
        # on sglang's even-split default — required when Cairn's split differs (it does: front-loaded
        # vs sglang's remainder-last) and the door to memory-balanced non-uniform splits (MoE layers
        # are far heavier than the few dense layers). None ⇒ fall back to sglang's even split.
        part = cairn_pp_partition if cairn_pp_partition is not None else os.environ.get("CAIRN_PP_LAYER_PARTITION")
        if isinstance(part, str):
            part = [int(x) for x in part.split(",") if x.strip()] or None
        self.cairn_pp_partition = list(part) if part else None
        if (self.cairn_pp_rank is None) != (self.cairn_pp_size is None):
            raise ValueError("cairn_pp_rank and cairn_pp_size must be set together (or both None)")
        if self.cairn_pp_size is not None:
            if not (0 <= self.cairn_pp_rank < self.cairn_pp_size):
                raise ValueError(
                    f"cairn_pp_rank ({self.cairn_pp_rank}) must be in [0, cairn_pp_size={self.cairn_pp_size})"
                )
            if self.cairn_pp_size < 1:
                raise ValueError(f"cairn_pp_size must be ≥ 1 (got {self.cairn_pp_size})")
        self._runner = None
        self._inner = None                       # the inner decoder model (.layers/.embed_tokens/.norm)
        self._is_v4 = False                      # set in load_shard: DeepSeek-V4 hyper-connection forward path
        self._batches: Dict[str, Any] = {}       # seq_id -> SGLang ScheduleBatch (KV lifecycle)
        self._steps: Dict[str, int] = {}         # seq_id -> forward count (0 = needs prefill)

    # ---- load: build/attach the in-process ModelRunner + this block's layer view ----
    def load_shard(self) -> None:
        if not _HAS_TORCH:
            raise SglangNotAvailable(
                "torch+sglang required; install infra/requirements-gpu.txt on a GPU box"
            )
        import os
        import socket
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_executor.model_runner import ModelRunner

        key = (self.model, self.device, self.cairn_pp_rank, self.cairn_pp_size)
        runner = SglangNodeRuntime._SHARED.get(key)
        if runner is None:
            mem = float(os.environ.get("CAIRN_SGLANG_MEM_FRACTION", "0.8"))
            sock = socket.socket(); sock.bind(("", 0)); port = sock.getsockname()[1]; sock.close()
            sa = ServerArgs(model_path=self.model, tp_size=1, pp_size=1, mem_fraction_static=mem,
                            disable_cuda_graph=True, trust_remote_code=True)
            # Path-β layer slicing (2026-06-24 root-cause + rewrite — replaces the get_pp_group patch
            # that was silently discarded by ModelRunner's dist re-init; see _set_pp_identity docstring).
            #   (1) Pin the exact slice boundaries via SGLANG_PP_LAYER_PARTITION (so sglang's
            #       get_pp_indices matches Cairn's layer_range — incl. front-loaded / non-uniform splits).
            #   (2) Wrap ModelRunner.load_model — runs AFTER init_torch_distributed, just before the
            #       model is built — to mutate the live _PP to (k, N). The model then reads
            #       get_pp_group() = (k, N) → make_layers slices → only our layers materialize.
            # ⚠ GPU-UNVERIFIED (box torn down before re-test): confirm on next launch (a) `load_model`
            # is the post-dist-init hook point, (b) is_first/last_rank derive from the attrs we set,
            # (c) PPMissingLayer outside the slice + per-box VRAM ~slice-sized, not full.
            saved_load_model = None
            if self.cairn_pp_size is not None and self.cairn_pp_size > 1:
                k, N = self.cairn_pp_rank, self.cairn_pp_size
                if self.cairn_pp_partition is not None:
                    if len(self.cairn_pp_partition) != N:
                        raise ValueError(
                            f"cairn_pp_partition {self.cairn_pp_partition} must have cairn_pp_size={N} "
                            f"entries (one layer-count per stage)"
                        )
                    os.environ["SGLANG_PP_LAYER_PARTITION"] = ",".join(str(x) for x in self.cairn_pp_partition)
                if not hasattr(ModelRunner, "load_model"):
                    raise RuntimeError(
                        "path-β: ModelRunner has no 'load_model' to hook — sglang internals changed; "
                        "update the load_model wrapper hook point in fork/shard/sglang_node.py."
                    )
                saved_load_model = ModelRunner.load_model

                def _cairn_load_model(_mr, *a, _orig=saved_load_model, _k=k, _N=N, **kw):
                    # Runs after ModelRunner.init_torch_distributed() rebuilt the real _PP (world_size=1);
                    # set it to our (k, N) so the model-under-construction slices to our layers.
                    import sglang.srt.distributed.parallel_state as _pstate
                    grp = getattr(_pstate, "_PP", None)
                    if grp is not None:
                        _set_pp_identity(grp, _k, _N)
                    return _orig(_mr, *a, **kw)

                ModelRunner.load_model = _cairn_load_model
                # V4-Flash: sglang's load_weights skips embed/`.norm.` weights by is_first/is_last_rank,
                # but those @properties read the GLOBAL rank (0) not our (k,N) → every stage would drop
                # its per-layer compressor.norm/indexer.compressor.norm (+ final norm). Force the gates
                # open for the load only (GPU-confirmed 2026-06-25; see _LoadTimeAllRanksPPGroup).
                _install_load_weights_rank_fix()
                # And skip PPMissingLayer stubs in V4's post-load APE-hotfix pass (out-of-slice layers
                # have no .self_attn under the layer-split; GPU-confirmed 2026-06-25).
                _install_post_load_weights_fix()
            try:
                runner = ModelRunner(
                    model_config=ModelConfig.from_server_args(sa), mem_fraction_static=mem,
                    gpu_id=self.device_index, tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
                    # MoE expert-parallel rank/size — REQUIRED by the sglang build in the V4-Blackwell
                    # image (added for MoE models like V4). Cairn splits by LAYER, not by expert, so
                    # each box holds ALL experts for its layer slice → no EP sharding (rank 0, size 1).
                    moe_ep_rank=0, moe_ep_size=1,
                    nccl_port=port, server_args=sa,
                )
            finally:
                if saved_load_model is not None:
                    ModelRunner.load_model = saved_load_model     # never leak the wrapper to other loads
            SglangNodeRuntime._SHARED[key] = runner
        self._runner = runner
        self._inner = runner.model.model
        # DeepSeek-V4 uses a different layer contract than the residual-stream models the generic
        # forward was proven on (hyper-connection 3D hidden, internal residual, 5-arg layers,
        # hc_head+norm tail). Detect it by its hallmark attr and drive it via the V4 forward path.
        self._is_v4 = hasattr(self._inner, "hc_mult")
        n = len(self._inner.layers)
        s, e = self.layer_range.start, self.layer_range.end   # end EXCLUSIVE (LayerRange convention)
        if not (0 <= s < e <= n):
            raise ValueError(f"layer_range [{s},{e}) out of [0,{n}] for {self.model}")
        # Path-β invariant: when cairn_pp_size > 1, layer_range MUST match the sglang loader's slice.
        # Otherwise we'd try to forward through PPMissingLayer placeholders (no weights, just stubs).
        if self.cairn_pp_size is not None and self.cairn_pp_size > 1:
            from sglang.srt.distributed import get_pp_indices  # type: ignore[import-not-found]
            ps, pe = get_pp_indices(n, self.cairn_pp_rank, self.cairn_pp_size)
            if (s, e) != (ps, pe):
                raise ValueError(
                    f"layer_range [{s},{e}) does not match sglang pp partition [{ps},{pe}) "
                    f"for cairn_pp_rank={self.cairn_pp_rank} cairn_pp_size={self.cairn_pp_size}, "
                    f"num_layers={n}. Cairn's scheduler must set layer_range == get_pp_indices(...) "
                    "or the forward would hit PPMissingLayer placeholders."
                )
        self._start, self._end, self._n = s, e, n
        self._is_embed = (s == 0)                # stage 0 embeds the token ids
        self._is_tail = (e == n)                 # last stage applies norm + lm_head

    # ---- forward: run this block's layers over `hidden_states`; per-seq paged KV stays here ----
    def forward(self, hidden_states: Any, kv_meta: Dict[str, Any]) -> Any:
        if self._runner is None:
            raise RuntimeError("SglangNodeRuntime.forward: call load_shard() first")
        import torch
        from sglang.srt.managers.schedule_batch import ScheduleBatch, Req
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        seq = kv_meta["seq"]
        rid = f"{seq}::{self._start}-{self._end}"        # unique per (stream, block) on the shared pool
        x = hidden_states                                # [1,S] token ids (embed) or [1,S,H] hidden
        s_len = x.shape[1]
        step = self._steps.get(seq, 0)
        if step == 0:                                    # prefill (extend): ingest the prompt / replay
            ids = x[0].tolist() if self._is_embed else [0] * s_len   # mid blocks don't embed → dummy ids
            req = Req(rid=rid, origin_input_text="", origin_input_ids=ids,
                      sampling_params=SamplingParams(temperature=0.0, max_new_tokens=1))
            req.prefix_indices = []
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(ids)
            req.logprob_start_len = len(ids) - 1
            batch = ScheduleBatch.init_new(
                reqs=[req], req_to_token_pool=self._runner.req_to_token_pool,
                token_to_kv_pool_allocator=self._runner.token_to_kv_pool_allocator, tree_cache=None,
                model_config=self._runner.model_config, enable_overlap=False,
                spec_algorithm=SpeculativeAlgorithm.NONE,   # init_new dropped enable_custom_logit_processor
            )
            batch.prepare_for_extend()
            self._batches[seq] = batch
        else:                                            # decode: one token
            batch = self._batches[seq]
            tok = int(x[0, -1]) if self._is_embed else 0
            batch.output_ids = torch.tensor([tok], dtype=torch.int64, device=x.device)
            batch.prepare_for_decode()
        fb = ForwardBatch.init_new(batch.get_model_worker_batch(), self._runner)
        self._runner.attn_backend.init_forward_metadata(fb)   # flashinfer metadata for this batch

        self._steps[seq] = step + 1
        return self._forward_layers(x, fb, s_len)

    def _forward_layers(self, x: Any, fb: Any, s_len: int) -> Any:
        """Drive this stage's `[start,end)` layers over the incoming hidden and return the per-stage
        output: the next-stage hidden hand-off, or — at the tail — next-token logits. Branches on
        model architecture, since the layer/residual/tail contract differs.

        V4 (DeepSeek-V4-Flash, hyper-connection): hidden is 3D `[S, hc_mult, H]` (embed then
        `unsqueeze(1).repeat(1,hc,1)`); each layer is
        `(positions, hidden_states, input_ids, forward_batch, input_ids_global) -> tensor` with the
        residual folded INTERNALLY (nothing crosses the wire but the one hidden tensor); the tail does
        `hc_head` (collapse the hc dim) -> single-arg `norm` -> `lm_head`. `input_ids_global == input_ids`
        at tp=1/dp=1. NOTE: mid/tail stages pass dummy `input_ids` (the wire carries only hidden); V4's
        MoE routes on the hidden and only touches input_ids in the bypassed DP/EP gather, so this is
        correct for single-GPU stages — if a decode ever comes out incoherent, carry real ids on the wire.

        Residual-stream (PROVEN on L4 2026-06-21; e.g. Llama): flat `[S, H]` hidden, layers return
        `(hidden, residual)`, the boundary folds `hidden+residual`, the tail is `norm(hidden, residual)`."""
        m = self._runner.model
        ids = fb.input_ids                                    # real on entry; dummy [0]* on mid/tail
        if self._is_v4:
            hc = self._inner.hc_mult
            if self._is_embed:
                hidden = self._inner.embed_tokens(ids)                       # [S, H]
                hidden = hidden.unsqueeze(1).repeat(1, hc, 1)               # [S, hc_mult, H]
            else:
                hidden = x.reshape(-1, hc, x.shape[-1])                     # wire [1,S,hc,H] -> [S,hc,H]
            for i in range(self._start, self._end):
                hidden = self._inner.layers[i](
                    positions=fb.positions, hidden_states=hidden,
                    input_ids=ids, forward_batch=fb, input_ids_global=ids,
                )
            if self._is_tail:
                hidden = self._inner.hc_head(                                # collapse the hc_mult dim
                    hidden, self._inner.hc_head_fn, self._inner.hc_head_scale, self._inner.hc_head_base)
                hidden = self._inner.norm(hidden)                           # V4 norm: single-arg
                logits = m.logits_processor(ids, hidden, m.lm_head, fb).next_token_logits
                return logits.reshape(1, -1, logits.shape[-1])              # [1, 1, V]
            return hidden.reshape(1, s_len, hc, -1)                         # [1, S, hc_mult, H] -> next

        if self._is_embed:
            hidden = self._inner.embed_tokens(ids)
            residual = None
        else:
            hidden = x.reshape(-1, x.shape[-1])                            # [1,S,H] -> flat [S,H]
            residual = None
        for i in range(self._start, self._end):
            hidden, residual = self._inner.layers[i](fb.positions, hidden, fb, residual)
        if self._is_tail:
            hidden, _ = self._inner.norm(hidden, residual)
            logits = m.logits_processor(ids, hidden, m.lm_head, fb).next_token_logits
            return logits.reshape(1, -1, logits.shape[-1])                 # [1, 1, V]
        folded = hidden + residual if residual is not None else hidden
        return folded.reshape(1, s_len, -1)                                # [1, S, H] -> next stage

    # ---- warmup: pre-compile the flashinfer JIT kernels so the first REAL forward isn't where it lands ----
    def warmup(self, n_tokens: int = 4) -> None:
        """Run a throwaway prefill + decode to force flashinfer's one-time ~30-45s JIT kernel compile NOW,
        at load time, instead of on the first real forward. Critical for a warm SPARE: without it the
        spare's first forward IS the recovery forward, so the compile dominates MTTR (cross-box recovery
        measured 38.9s — almost all of it this compile). Uses a throwaway seq and frees it; no-op if
        unloaded. The compiled kernels persist in flashinfer's on-disk cache for this box."""
        if self._runner is None:
            return
        import torch
        w = self._inner.embed_tokens.weight                  # [vocab, H] — gives H, dtype, device
        H, dtype, device = w.shape[1], w.dtype, w.device
        seq = "__cairn_warmup__"
        try:
            if self._is_embed:                               # entry: token ids in
                self.forward(torch.arange(n_tokens, device=device).reshape(1, n_tokens), {"seq": seq, "pos": 0})
                self.forward(torch.tensor([[0]], device=device), {"seq": seq, "pos": n_tokens})
            else:                                            # mid/tail: hidden state in
                self.forward(torch.zeros(1, n_tokens, H, dtype=dtype, device=device), {"seq": seq, "pos": 0})
                self.forward(torch.zeros(1, 1, H, dtype=dtype, device=device), {"seq": seq, "pos": n_tokens})
        finally:
            self.free_seq(seq)                               # drop the throwaway KV — leaves real state untouched

    def free_seq(self, seq_id: str) -> None:
        """Drop a seq's KV on this block (on completion, or before a replay-rebuild)."""
        self._steps.pop(seq_id, None)
        batch = self._batches.pop(seq_id, None)
        if batch is not None:
            try:                                          # best-effort release back to the pools
                self._runner.req_to_token_pool.free(batch.req_pool_indices)
                self._runner.token_to_kv_pool_allocator.free(batch.out_cache_loc)
            except Exception:
                pass

    def kv_tokens(self, seq_id: str) -> int:
        batch = self._batches.get(seq_id)
        try:
            return int(batch.seq_lens[0]) if batch is not None else 0
        except Exception:
            return self._steps.get(seq_id, 0)

    # ---- heartbeat: real VRAM/liveness (concretely writable now) ----
    def heartbeat(self) -> dict:
        info: dict = {
            "loaded": self._runner is not None,
            "layers": [self.layer_range.start, self.layer_range.end],
            "device": self.device,
            "kv_seqs": len(self._batches),   # seqs with KV cached on this block (renamed from _kv_seqs in the rung-2 rewrite)
        }
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info(self.device_index)
                info["vram_used_gb"] = round((total - free) / 2**30, 2)
                info["vram_total_gb"] = round(total / 2**30, 2)
                info["alive"] = True
            except Exception as e:  # bad ordinal / driver error → unhealthy, never throw (L2)
                info["alive"] = False
                info["note"] = f"VRAM probe failed: {e}"
        else:
            info["alive"] = False
            info["note"] = "no CUDA — this runtime is GPU-only (rung 2/3)"
        return info
