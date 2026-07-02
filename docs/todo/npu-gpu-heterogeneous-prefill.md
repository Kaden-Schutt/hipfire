# TODO: heterogeneous NPU + GPU prefill (concurrent co-processing)

Status: TODO (research idea, not yet approved for build). Depends on prereqs
that do not exist yet — see Blockers.
Date: 2026-07-02

## Motivation

On a Phoenix-class APU the XDNA1 NPU (`accel0`) sits **completely idle** while
the gfx1103 GPU runs a forward pass. Prefill is the one regime where a second
compute engine can plausibly help:

- **Prefill is batched and compute-bound.** Every weight is read once and reused
  across all `N` prompt positions, so the big projections are GEMMs with high
  arithmetic intensity — the regime where adding FLOP/s helps.
- **Decode is not.** Decode is `B=1`, memory-bound, reads all weights per token,
  and the NPU gets *no* bandwidth advantage on a unified-memory APU (NPU + GPU
  share the same LPDDR5). The int8 NPU array also can't run `N < 128`. Decode is
  the real latency bottleneck and this idea does **nothing** for it. Scope this
  strictly to prefill.

This is the exact "only upside path" the OQ→NPU feasibility spike flagged as
worth a look but not yet actionable (see `docs/plans/2026-06-23-oq8-npu-spike.md`
and the spike's NO-GO writeup). This doc records the design space so the idea
isn't re-derived from scratch when the prerequisites land.

## What is already proven (don't re-litigate)

- **NPU int8 GEMM is bit-exact.** The OQ grouped int8 GEMM (int8→int32,
  `aie::mmul<4,8,8>` = GPU's `v_wmma_i32_16x16x16_iu8`) runs bit-exact vs numpy
  on XDNA1 for OQ8 (W8A8) and OQ+ (W4A8, int4 host-unpacked). Correctness is not
  the risk.
- **Toolchain is validated** (with the two documented workarounds: IRON API
  drift + the user-space `libboost_program_options.1.83` `LD_LIBRARY_PATH`
  shim). The swiglu smoke builds an xclbin and runs on NPU1.
- **The current int8 NPU ceiling is a *tooling* floor, not silicon.** Measured
  ~2.0 TFLOP/s (~12% of the ~16-TOPS int8 peak), per-core occupancy ~21%,
  weight-DMA-bound ~4 GB/s. `bf16` matmul compiles + runs correct via the
  Triton-XDNA aie2 path; `int8` is blocked by an `mlir-air` split-pass SIGABRT
  (`AIRSplitL2MemrefForBufferConstraintPass`). See
  `docs/npu/triton-xdna-aie2-int8.md`.

## Goal

An **opt-in, feature-gated** prefill path that dispatches part of each layer's
GEMM work to the NPU concurrently with the GPU, on APU silicon that has an idle
NPU, and measurably reduces prefill wall-time versus GPU-only — without
regressing the GPU-only baseline and without ever becoming a *required* backend.

Non-goals / guardrails:

- **GPU HIP-direct stays canonical.** The NPU path is opportunistic co-processor
  offload on AMD APU silicon, not a new portable backend. It must be
  `#[cfg]`/runtime-feature gated and absent by default. It does not touch RDNA2
  and is a no-op on any box without an XDNA NPU. This is *not* a cross-vendor
  compute backend (no Vulkan/wgpu); it is a second AMD accelerator reached via
  XRT, used only when present.
- **No Python in the prefill hot path.** The spike's NPU driver was Python
  (`tools/npu/*`); a serving offload needs a Rust/XRT dispatch path. The Python
  tools stay as bring-up/bench harnesses only.
- Decode is explicitly out of scope.

## Design space (how to split the work)

On a UMA APU the weights and activations live in one address space, so an NPU
tile can read the *same* weight buffer the GPU uses — no host copy, no duplicate
resident weights. The question is only *which* work to hand off. Options, roughly
in increasing coupling:

1. **Independent-projection co-schedule (most promising first cut).** Within a
   layer, several GEMMs are mutually independent given the same normed input:
   `gate` ∥ `up`; `q` ∥ `k` ∥ `v`. Run e.g. `up_proj` on the NPU while the GPU
   runs `gate_proj`, then join for `gelu_mul`. No output-tile stitching, minimal
   scheduler complexity. Ceiling is bounded by the *smaller* of the two GEMMs.
2. **Output-row partition of one big GEMM.** Split a single projection's output
   rows `[0..m)` between NPU (`[0..s)`) and GPU (`[s..m)`); both read the shared
   weight+activation, write disjoint output slices, join before the next op.
   Tunable split ratio `s` to match the two engines' throughput. More plumbing
   (a partial-GEMM entry point on each side) but the split ratio is the single
   knob that makes the offload *pay*.
3. **Prompt-chunk split — rejected.** Splitting positions `N` across engines
   fights causal attention (later chunks depend on earlier KV); not worth it.
4. **Whole-layer ping-pong — rejected.** Layer `L+1` depends on `L`; no
   inter-layer parallelism in a single sequence.

The gate/up co-schedule (1) is the cheapest experiment with a real answer; the
output-row partition (2) is where the actual throughput knob lives.

## Blockers / prerequisites (why this is TODO, not a plan)

1. **Async, non-blocking NPU dispatch from Rust.** The spike dispatched the NPU
   *synchronously*, so "NPU time is pure addition" — it measured −2.4% on
   elementwise SwiGLU. Concurrency is the entire premise here: without a
   fire-and-continue XRT submit + a completion fence the GPU can overlap, there
   is no win. This infra does not exist. **This is the gating dependency** — do
   not start the scheduler until non-blocking dispatch lands (for this or any
   other reason).
2. **A usable NPU int8 kernel at real occupancy**, OR commit to the bf16 path.
   At today's ~2 TFLOP/s int8 floor the NPU's contribution is marginal against
   the GPU's prefill throughput and likely lost to shared-LPDDR5 contention.
   Either (a) hand-write the aie2 int8 kernel in raw IRON to bypass the broken
   `mlir-air` split pass, or (b) use the working Triton-XDNA **bf16** matmul and
   accept a bf16 offload arm. Quantify both before committing.
3. **A fused per-group-rescale kernel + on-tile int4 unpack** for the OQ formats
   (so the NPU arm matches the GPU's grouped dequant), if going the int8 route.
4. **A split scheduler + memory-bandwidth budget.** On UMA the NPU and GPU
   contend for the same DRAM. Model the BW split before assuming additive FLOP/s;
   a compute win that the GPU pays for in stalled bandwidth is not a win.

## Reality check / go–no-go criteria

Before any serving integration, a standalone bench must show, at a representative
prefill shape (e.g. a 1.5B/8B FFN `down`/`up` at `N ≥ 256`):

- concurrent NPU+GPU wall-time **< 0.90×** GPU-only for the offloaded GEMM, and
- **no** regression to the *rest* of the layer from BW contention (measure GPU
  prefill TPS with the NPU arm on vs off), and
- correctness within the existing coherence/KLD tolerance (int8 is already
  bit-exact; a bf16 arm needs its own KLD check).

If the offloaded GEMM can't clear 0.90× concurrently, stop — the spike's
"marginal gain, high effort" verdict stands and the effort is better spent on the
GPU decode path.

## Pointers

- `docs/plans/2026-06-23-oq8-npu-spike.md` — OQ→NPU spike design + NO-GO.
- `docs/npu/triton-xdna-aie2-int8.md` — bf16 works / int8 `mlir-air` blocker,
  full repro.
- `tools/npu/{oq_gemm_design,test_oq_gemm_npu,bench_oq_gemm_npu}.py`,
  `tools/npu/triton_xdna/bench_matmul.py` — bring-up + bench harnesses.
- Toolchain workarounds (IRON API drift, boost `LD_LIBRARY_PATH`): see the NPU
  bring-up notes; both are required for every NPU build+run.
