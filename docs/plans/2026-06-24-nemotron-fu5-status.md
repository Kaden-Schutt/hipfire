# Nemotron follow-ups — status (2026-06-24)

Branch: `chaingun`. Source of truth for scope: `docs/plans/2026-06-24-nemotron-followups.md`.

This file is a point-in-time status snapshot of the FU2–FU6 follow-up work on the
`nemotron_h` arch (NVIDIA Nemotron-3-Nano-4B, arch_id 14: Mamba-2 + GQA-NoPE +
ReLU²-MLP hybrid).

## Summary

| FU | Title | State |
|----|-------|-------|
| FU2 | HF-reference numeric bisect | **DONE** (committed earlier) |
| FU3 | quantizer → nemotron_h mq4 .hfq | **DONE** |
| FU4 | loader compat (load .hfq + quantized gemv) + serving | **DONE** |
| FU5 | N6 batched prefill + q8 SSM state | **mostly done** — q8 state + benchmarks remain |
| FU6 | Nano-30B MoE ('E' block) | not started |
| FU1 | chat-template / coherence | blocked (newline attractor; no valid HF ref) |

## FU4 (DONE, committed)

- `86e86b8b3` — `NemotronModel::from_hfq`: quantized HFQ loader (Q8 + MQ4G256 via
  dispatched gemv; bf16 recurrence/norms → f32; Q8 embedding lookup; runtime
  out_proj rescale). Validated vs f32 forward: argmax match, cosine 0.99.
  Fixed two real `hipfire-quantize` bugs (see memory
  `project_nemotron_quantizer_bugs`): `is_embed` missed
  `backbone.embeddings.weight` (→ 4-bit instead of Q8); HFQ4G128 fallback
  assumed `k%128==0` but hidden=3136 divides neither 256 nor 128 → garbage
  (cos 0.003), now falls back to Q8.
- `89100cc02` — serving wiring: `load_model` arch_id==14 HFQ branch →
  `from_hfq`. Daemon serves `/tmp/nano4b-mq4.hfq` end-to-end (smoke-tested:
  loads 3.56 GB, streams tokens). Output coherence (newline attractor) is the
  pre-existing FU1 blocker, not a loader issue.

## FU5 (N6 batched prefill) — committed pieces

The plan's stated top risk — "the chunked inter-chunk recurrence must be
bit-faithful to the decode recurrence" — was de-risked CPU-first, then each GPU
piece validated gpu-vs-cpu (or gpu-vs-decode) against an oracle.

| Commit | Piece | Validation |
|--------|-------|-----------|
| `3c829a0e5` | Phase A CPU oracle: `ssd::ssd_sequence` + `ssd::ssd_chunked` | `chunked == sequential` within 1e-3 (no-gpu) |
| `994eb30e1` | `mamba2_ssd_seq_f32` kernel (single-launch SSD prefill scan) | gpu==cpu 3.7e-9 |
| `67d9352c5` | `conv1d_bias_silu_seq_f32` kernel (batched short-conv) | gpu==cpu 1.5e-8 |
| `7c04a3765` | `mamba2_gated_norm_seq_f32` kernel (batched gated group-RMSNorm) | gpu==cpu 2.4e-7 |
| `c85f40f52` | `Mamba2BlockGpu::prefill` (composes the 3 kernels + gemm + `strided_copy_2d` split) + `LinearWeight::gemm_seq` (F32) | prefill == decode loop 4.5e-8 |
| `7a478ae08` | `MlpRelu2Gpu::prefill` | prefill == forward loop 3.7e-9 |
| `0dfec9e4e` | `NemotronAttnGpu::prefill` (NoPE GQA, `attention_f32_batched_masked`) | prefill == decode loop 1.3e-8 |
| `211888d7a` | `NemotronModel::prefill_batched` + `SimpleAr::prefill` f32 fast path | model prefill == decode loop 2.98e-7, argmax match |
| `27f994e00` | Quantized `LinearWeight::gemm_seq` (Q8, HFQ4G256, HFQ4G128, MQ4G256 via batched FWHT rotate + HFQ4G256 GEMM) + HFQ batched prefill enable | HFQ model prefill == decode loop 1.29e-5, argmax match |

Key facts established:
- `gemm_f32_train(trans_b)` does `out[seq,m] = x[seq,k]·Wᵀ` from the `[m,k]`-stored
  weight (matches the gemv convention).
- `strided_copy_2d(src, src_off, src_stride, dst, dst_off, dst_stride, rows,
  cols, accumulate)` uses ELEMENT offsets — used for the proj→z/xBC/dt and
  xBC→x/B/C splits.
- KV cache is pos-major (`dst[pos*kv_dim+i]`), so a position-0 prefill KV write
  is a plain contiguous copy.
- `attention_f32_batched_masked` already supports GQA + causal masking via
  per-query `positions` — for a full prefill: positions=[0..seq], block_start=0,
  block_cols=seq, max_ctx_len=seq, batch_size=seq, tree_bias=None. No new kernel.
- `relu2_f32` and `add_inplace_f32` are elementwise (batch for free);
  `rmsnorm_batched(x, w, out, batch, n, eps)` exists.

## FU5 — current prefill state

`SimpleAr::prefill` now resets sequence state and routes through
`NemotronModel::prefill_batched` when `can_batched_prefill()` is true. Both f32
models (`new`) and the supported quantized HFQ models (`from_hfq`) take the
batched path; `forward_gpu`/`decode_step` stay as the per-token decode path.

`prefill_batched` embeds all tokens into `[seq,hidden]`, then per layer runs
`rmsnorm_batched` → `block.prefill` → residual add, followed by final
`rmsnorm_batched` and `lm_head` on the last position. Quantized `gemm_seq`
supports Q8, HFQ4G256, HFQ4G128, and MQ4G256; MQ4G256 rotates the whole
`[seq,k]` activation with `rotate_x_mq_batched`/`rotate_x_mq_awq_batched`, then
uses the HFQ4G256 batched GEMM against the rotated weight layout.

Validation run locally on gfx1151:
- `test_model_prefill_gpu`: synthetic f32 Mamba/MLP/attention model,
  max|Δlogit|=2.98e-7, argmax match.
- `test_model_prefill_hfq_gpu`: real `/tmp/nano4b-mq4.hfq`,
  max|Δlogit|=1.29e-5, argmax match.
- Commit hooks for `211888d7a` and `27f994e00`: rustfmt, clippy, short
  coherence battery (no hard errors), fast agentic gate, and MQ4 speed gate all
  passed. Tiny-fixture golden still drifted on existing Qwen fixtures and
  escalated to the full short coherence battery in both hooks.

## FU5 — REMAINING

1. **q8 SSM state.** Quantize the Mamba `h` state to q8 between steps (mirror the
   GDN q8-state pattern) to cut state memory/bandwidth. Pairs with the kernels.
2. **Prefill tok/s benchmark.** Per `docs/methodology/perf-benchmarking.md` (warm
   cache, fresh-process probe) — quantify the launch-reduction win on f32 and
   quantized HFQ models.
3. **(Optional) chunked masked-flash for very long prompts.** The attention
   prefill uses a single masked-flash block (`block_cols=seq`); shared-mem scales
   with seq. Fine for normal prompts; long-context needs block tiling.

## FU6 (Nano-30B MoE) — not started

Same `nemotron_h` arch but hidden=2688, 52 layers, pattern `MEMEM*EMEM…` →
introduces a new **'E' (MoE) block**: `BlockKind::Moe` + `NemotronMoeGpu` (router
+ experts). Source checkpoint in `/srv/huggingface`. Deferred.

## FU1 (coherence) — standing blocker

Daemon generation produces a newline attractor. Forward NUMERICS are validated
(FU4 cos 0.99; FU5 prefill==decode). The attractor is a generation/chat-template/
sampling issue with no valid HF reference (nemotron_h needs mamba-ssm CUDA
kernels that don't build on ROCm; HF's pure-torch `torch_forward` is itself
broken for generation). Possibly addressable via a repeat penalty / chat-template
fix (cf. the medgemma `*`-attractor → repeat-penalty fix), but not yet
investigated. See memory `project_nemotron_mamba2`.

## Validation entry points (all gpu-tcas-coordinated via `hipfire lock`)

```
cargo run -p hipfire-arch-nemotron --example test_ssd_seq_gpu        # SSD kernel
cargo run -p hipfire-arch-nemotron --example test_conv1d_seq_gpu     # conv1d kernel
cargo run -p hipfire-arch-nemotron --example test_gated_norm_seq_gpu # gated-norm kernel
cargo run -p hipfire-arch-nemotron --example test_block_prefill_gpu  # Mamba block
cargo run -p hipfire-arch-nemotron --example test_mlp_prefill_gpu    # MLP block
cargo run -p hipfire-arch-nemotron --example test_attn_prefill_gpu   # attention block
cargo run -p hipfire-arch-nemotron --example test_model_prefill_gpu  # full model (FU5 gate)
cargo run -p hipfire-arch-nemotron --example test_model_prefill_hfq_gpu # full HFQ model
cargo run -p hipfire-arch-nemotron --example hfq_vs_f32              # FU4 quant loader
```
