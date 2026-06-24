# Nemotron follow-ups — status (2026-06-24; updated 2026-06-25)

Branch: `chaingun`. Source of truth for scope: `docs/plans/2026-06-24-nemotron-followups.md`.

This file is a point-in-time status snapshot of the FU2–FU6 follow-up work on the
`nemotron_h` arch (NVIDIA Nemotron-3-Nano-4B, arch_id 14: Mamba-2 + GQA-NoPE +
ReLU²-MLP hybrid).

## Summary

| FU | Title | State |
|----|-------|-------|
| FU2 | HF-reference numeric bisect | **DONE** (committed earlier; Python native-Mamba reference refreshed) |
| FU3 | quantizer → nemotron_h mq4/q8 .hfq | **DONE** — mq4 protects Nemotron residual writers as q8 |
| FU4 | loader compat (load .hfq + quantized gemv) + serving | **DONE** |
| FU5 | N6 batched prefill + q8 SSM state | **mostly done** — q8 state + benchmarks remain |
| FU6 | Nano-30B MoE ('E' block) | not started |
| FU1 | chat-template / coherence | blocked (EOS/Jinja/CLI controls fixed; q8 matches f32 but generation still lacks a coherent reference) |

## FU1 update (2026-06-25)

The originally planned chat-template fixes are now live:

- arch 14 resolves serving EOS from tokenizer `<|im_end|>` (id 11), not
  `config.eos_token_id` (`</s>` = 2), in both safetensors and HFQ load paths.
- `generate_nemotron` defaults to the embedded Jinja chat template when present
  (opt out with `HIPFIRE_JINJA_CHAT=0`).
- `hipfire chat` now forwards config-derived thinking controls to the daemon:
  default `thinking = "off"` maps to `assistant_prefix = "closed_think"` and
  `max_think_tokens = 1`, matching the HTTP path.

Validation on gfx1151 still reproduces the FU1 blocker:

- `/tmp/nano4b-mq4.hfq` via current `target/debug/hipfire-daemon` emits 11
  newline tokens for `Answer in one short sentence: What is 2+2?`.
- The same result occurs with the Plain ChatFrame fallback
  (`HIPFIRE_JINJA_CHAT=0`) and with a non-empty system prompt.
- The model-card prompt `Write a haiku about GPUs` with `temperature=1.0`
  produced incoherent sampled text, not a usable answer.
- HF tokenizer rendering is byte/ID-identical to Hipfire for reasoning-off
  ChatML (`[10,25708,1010,11,...,12,13]` for the 2+2 prompt).
- Local HF pure-Torch fallback (with `mamba_ssm` stubs and trained `dt_bias`
  restored from safetensors) matches Hipfire f32 at the first-token boundary for
  the same prompt: top-2 are `<|im_end|>` id 11 then newline id 1010.
- `benchmarks/nemotron/dump_hf_reference.py` is now the repeatable Python
  reference harness for this comparison. It can render the same Jinja ChatML
  prompt as Hipfire, restore trained `dt_bias`, patch Nemotron Mamba mixers to
  Transformers' canonical `Mamba2Mixer.torch_forward`, dump Hipfire-aligned
  per-block hidden states, and record first-token top-k/generation metadata.
- The closed-think 2+2 reference dump compared against `bisect_nano4b`
  (`CAP_POS=last`, pos 28) has no per-layer divergence above 5%; final logits
  match top-5 `[11, 1010, 1058, 1050, 1319]`, logit max|delta|=0.33261,
  relative logit delta=0.0221.
- `/tmp/nano4b-mq4.hfq` flips that close f32 boundary to newline:
  f32 argmax=11, mq4 argmax=1010, logit cosine 0.989951, top-5 overlap 4/5.
- `/tmp/nano4b-q8.hfq` tracks f32: argmax=11, identical top-5, logit cosine
  0.999967, mean|delta|=0.01770, max|delta|=0.11181.
- `hipfire-quantize --format mq4` now protects Nemotron-H residual writers
  (`*.mixer.out_proj.weight`, `*.mixer.down_proj.weight`, `*.mixer.o_proj.weight`)
  as Q8. On Nano-4B this makes the fresh protected artifact
  `/tmp/nano4b-mq4-protected.hfq` Q8-sized (4.0G) because the remaining linear
  readers already fall back to Q8 on ragged K=3136. The protected artifact
  matches f32 at the closed-think boundary: argmax=11, identical top-5, logit
  cosine 0.999967, mean|delta|=0.01770, max|delta|=0.11181.
- Q8 serving therefore avoids the newline loop on the closed-think 2+2 prompt,
  but greedy generation emits 0 tokens because the first token is `<|im_end|>`.
  A reasoning-on sampled haiku prompt still produced incoherent numeric text.

So FU1 is no longer "do the chat-template work." The immediate conclusion is
that unprotected mq4 is too lossy for this uncalibrated Nano-4B artifact at a
close generation boundary; protected mq4/q8 is numerically faithful but still
does not prove coherent serving. The refreshed Python/native-Mamba reference
proves the Hipfire f32 forward/generation boundary for the closed-think prompt,
but it still says the correct greedy first token is immediate `<|im_end|>`. What
remains is to get a coherent generation convention or a production reference path
for this checkpoint (CUDA/mamba-ssm, vLLM, or another NVIDIA runtime). The local
Python reference is useful for first-token and per-layer diagnostics, but not by
itself a coherence oracle.

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

Daemon generation with the original unprotected mq4 artifact produces a newline
attractor; q8 and the fresh protected-mq4 artifact follow f32 and stop
immediately on the closed-think 2+2 prompt. Forward numerics are validated
(FU4/FU5 prefill==decode, protected-mq4/q8-vs-f32 cosine 0.999967), and the
concrete EOS/Jinja/thinking-control work is done. The Python native-Mamba
reference now gives a repeatable local first-token/per-layer oracle and matches
Hipfire f32 on the closed-think prompt. The remaining blocker is
coherence/reference quality: the reference boundary itself chooses immediate
`<|im_end|>` for that prompt, and the local ROCm/Python path is not enough to
prove the intended production generation convention.

Next useful FU1 work:

1. Use `benchmarks/nemotron/dump_hf_reference.py` as the first local oracle for
   every prompt under investigation; keep the rendered ChatML IDs and first-token
   top-k with the evidence.
2. Capture a valid CUDA/vLLM/NVIDIA-runtime reference for Nano-4B on 3-4 prompts
   if the local native-Mamba reference still stops or samples incoherently.
3. Compare Hipfire f32/q8 against that reference at the generation boundary. If
   the reference is coherent, bisect the first divergence against the already
   added per-layer/block hooks.
4. Treat real 4-bit Nano-4B as a sensitivity issue until calibrated. The current
   `--format mq4` policy protects projection-back residual writers as q8; an
   imatrix/AWQ/Lloyd pass is required before claiming true mq4 coherence.
5. Only after a coherent reference exists, tune sampling/repeat penalties or
   prompt policy. Current prompt/EOS-only changes do not fix the blocker.

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

# Python native-Mamba reference + Hipfire-side bisect for a rendered prompt:
python3 benchmarks/nemotron/dump_hf_reference.py --mode jinja --thinking off \
  --text 'Answer in one short sentence: What is 2+2?' \
  --max-new-tokens 1 --out /tmp/nemo_hf_ref_closed2p2.npz
NEMO_TOKENS='10,25708,1010,11,1010,10,3263,1010,31106,1294,1925,4958,19286,1058,5675,1395,1032,1050,1043,1050,1063,11,1010,10,1503,19464,1010,12,13' \
  CAP_POS=last cargo run -p hipfire-arch-nemotron --example bisect_nano4b -- /tmp/nemo_hipfire_closed2p2.bin
python3 benchmarks/nemotron/compare_bisect.py /tmp/nemo_hf_ref_closed2p2.npz /tmp/nemo_hipfire_closed2p2.bin 28

# Quantizer policy smoke: protected mq4 should no longer flip EOS to newline.
hipfire-quantize --input <Nano-4B-BF16-snapshot> --output /tmp/nano4b-mq4-protected.hfq --format mq4 --threads 16
NEMO_TOKENS='10,25708,1010,11,1010,10,3263,1010,31106,1294,1925,4958,19286,1058,5675,1395,1032,1050,1043,1050,1063,11,1010,10,1503,19464,1010,12,13' \
  cargo run -p hipfire-arch-nemotron --example hfq_vs_f32 -- /tmp/nano4b-mq4-protected.hfq
```
