# MiniMax fused MoE decode megakernel (Ship 6 EP perf)

**Goal:** cut MiniMax EP decode dispatch count (~5000 kernels/token, GPU-CP-bound
at ~95% util — see `ship6-substrate-ep.md` perf section). Fuse the MoE FFN block
(currently ~6–8 dispatches/layer) into **one** kernel, eliminating the
`gate_batch`/`up_batch`/`rot_batch`/`down_expanded` global round-trips. Targets
the B70's `VLLM_XPU_USE_LLM_SCALER_MOE` lever. Both **gfx1201** (RDNA4) and
**gfx1151** (RDNA3.5) — both wave32, one source.

## What it fuses (per top-k expert; the rmsnorm + rotate(x) + router + topk stay
   separate — they're cheap glue and produce `x_rot`/`topk_*`):

One block per `(krank ∈ 0..k_top)`. Inputs: `expert_gate_up_ptrs`,
`expert_down_ptrs`, `topk_indices`, `topk_weights`, `x_rot[hidden]`, output.
1. **gate+up GEMV** — Lloyd 4-entry codebook (72 B/256-group) · `x_rot` → `gate[mi]`,
   `up[mi]` held in **LDS** (mirror `gemv_mq2g256_lloyd_moe_gate_up_indexed.hip`).
2. **SwiGLU** — `t[i] = silu(gate[i]) * up[i]` in LDS.
3. **FWHT rotate** `t[mi]` in LDS — log₂(mi) butterfly stages, `__syncthreads`
   between stages. (This is the data barrier that forces gate/up to fully
   materialize before down; mi·4 B fits LDS.) Match
   `fused_silu_mul_rotate_mq_batched_for` semantics exactly.
4. **down GEMV** — MQ4 (or MQ2/MQ3-Lloyd for the uniform tiers) · `t[mi]` →
   `d[hidden]`.
5. **combine** — `atomicAdd(out[h], topk_weight[krank] * d[h])` into the
   residual/`routed_out` partial (residual-scaled, mirrors the current down arm).

mq2-lloyd tier = Lloyd gate/up + **MQ4 down**; mq3 = MQ3-Lloyd uniform; mq3-lloyd =
MQ3-Lloyd gate/up + MQ4 down. The kernel branches the down-dequant on the down
dtype (gate/up always Lloyd here). Start with the mq2-lloyd (Lloyd gate/up + MQ4
down) shape since that's the resident model.

## No-clobber architecture
- New file `kernels/src/minimax_fused_moe_decode.hip`.
- New `Gpu::minimax_fused_moe_decode(...)` in `rdna-compute` (own module fn,
  distinct JIT module name — see `reference_kernel_module_cache_collision`).
- Called ONLY from `minimax_moe_block`, behind `HIPFIRE_MINIMAX_FUSED_MOE`
  (default OFF). When off, the existing indexed-GEMV path runs unchanged.
- qwen35-A3B + DeepSeek MoE paths are NOT touched (separate kernels/dispatch).

## Build + validation plan (incremental — do NOT trust until coherence-gated)
1. Kernel skeleton: gate+up into LDS, write back to a scratch — A/B the gate/up
   output vs the existing kernel (bit/cosine) BEFORE adding silu/FWHT/down.
2. Add SwiGLU + FWHT in LDS — A/B `t[mi]` vs `fused_silu_mul_rotate_mq` output.
3. Add down + combine — A/B the full block output vs the current 3-kernel path.
4. Wire the gated hook; run `ep_minimax` on **gfx1201 (hiptrx)** AND
   **gfx1151 (hipx)**: confirm coherent generation (capitals prompt) + compare
   gen FNV to the unfused path (argmax-identical or coherent-equivalent).
5. Only then measure tok/s (target: meaningfully > 51; the dispatch cut is the win).
6. Occupancy check via `gfx-kernel-metadata` skill (VGPR/LDS/spill) on both archs.

## Risks
- FWHT stride/stage bugs → plausible garbage (coherence-gate catches it).
- LDS pressure (gate+up+t = ~3·mi·4 B) may cap occupancy; may need to stream
  gate/up rather than hold both. Measure.
- Lloyd codebook + MQ4 dequant in one kernel = two unpack paths; keep them
  branch-clean.
- Per-arch (gfx1201 vs gfx1151) wave32 is shared, but LDS size / occupancy
  differ — validate both.
