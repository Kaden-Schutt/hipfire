# Nemotron follow-ups — status (2026-06-24; updated 2026-06-25)

Branch: `chaingun`. Source of truth for scope: `docs/plans/2026-06-24-nemotron-followups.md`.

This file is a point-in-time status snapshot of the FU2–FU6 follow-up work on the
`nemotron_h` arch (NVIDIA Nemotron-3-Nano-4B, arch_id 14: Mamba-2 + GQA-NoPE +
ReLU²-MLP hybrid).

## Summary

| FU | Title | State |
|----|-------|-------|
| FU2 | HF-reference numeric bisect | **DONE** (committed earlier; Python native-Mamba reference refreshed) |
| FU3 | quantizer → nemotron_h mq4/q8 .hfq | **DONE** — mq4 protects Nemotron residual writers and Mamba `in_proj` as q8 |
| FU4 | loader compat (load .hfq + quantized gemv) + serving | **DONE** |
| FU5 | N6 batched prefill + q8 SSM state | **mostly done** — batched HFQ benchmark captured; q8 state opt-in validated |
| FU6 | Nano-30B MoE ('E' block) | in progress — 30B MQ4/HFQ artifact built; Mamba scale + hybrid MoE prefill restore closed-think 2+2 |
| FU1 | chat-template / coherence | blocked (EOS/Jinja/CLI controls fixed; vLLM reference is coherent; Hipfire/native-HF diverge at first token) |

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
  plus Mamba `*.mixer.in_proj.weight` as Q8. On Nano-4B this makes the fresh
  protected artifact
  `/tmp/nano4b-mq4-protected.hfq` Q8-sized (4.0G) because the remaining linear
  readers already fall back to Q8 on ragged K=3136. The protected artifact
  matches f32 at the closed-think boundary: argmax=11, identical top-5, logit
  cosine 0.999967, mean|delta|=0.01770, max|delta|=0.11181.
- `/home/sadara/vllm0.22.1` provides a coherent external reference when run with
  its ROCm environment repaired for this host (`PYTHONPATH` for bundled AMD SMI,
  executable bits restored on the venv Python/script entrypoints, and
  `VLLM_ROCM_USE_SKINNY_GEMM=0 VLLM_ROCM_USE_AITER_LINEAR=0
  VLLM_ROCM_USE_AITER_TRITON_GEMM=0` to avoid the missing `_rocm_C.wvSplitK`
  path). With byte-identical prompt ids, vLLM generates `2 + 2 equals 4.` and
  first-token top-5 `[1050, 31035, 1052, 2757, 16489]` (`2`, `Four`, `4`, `It`,
  `Two`).
- The Lyra ROCm stack (`/home/sadara/.venv`, with local `mamba_ssm` and
  `causal_conv1d` builds on `PYTHONPATH`) now provides a second local
  Transformers control. `dump_hf_reference.py --mamba-import real
  --mamba-reference remote` loads `/home/sadara/Models/NVIDIA-Nemotron-3-Nano-4B-BF16`
  with the actual installed Mamba kernels and writes
  `/tmp/nemotron_lyra_real_closed2p2.npz`. It still picks immediate
  `<|im_end|>` for the closed-think 2+2 prompt, with top-5
  `[11, 1010, 1058, 1050, 1319]`. That matches the prior native/stubbed
  Transformers reference and Hipfire f32, not vLLM.
- While validating Lyra, the local Nemotron remote-code cache path needed a small
  bug fix: `HybridMambaAttentionDynamicCache` did not store `conv_kernel_size`
  and used `.device` on Python lists for `conv_states`/`ssm_states`. After
  patching the local model copy and Transformers dynamic-module cache copies,
  explicit-cache prefill and one-token decode both produce finite logits. This
  proves the Lyra full-model prefill/decode path is usable for reference work.
- Q8 serving therefore avoids the newline loop on the closed-think 2+2 prompt,
  but greedy Hipfire/native-HF generation emits 0 tokens because the first token
  is `<|im_end|>`. A reasoning-on sampled haiku prompt still produced incoherent
  numeric text.

So FU1 is no longer "do the chat-template work." The immediate conclusion is
that unprotected mq4 is too lossy for this uncalibrated Nano-4B artifact at a
close generation boundary; protected mq4/q8 is numerically faithful to Hipfire
f32. Both stubbed/native and real-Mamba Transformers references follow Hipfire's
immediate-`<|im_end|>` boundary, while vLLM alone follows the coherent
production-style boundary. What remains is a vLLM-vs-Transformers/Hipfire bisect
to find the first generation-convention divergence.

## Local `~/Models` control check (2026-06-25)

The user-requested `~/Model` check resolved to `/home/sadara/Models` on this
host; `/home/sadara/Model` does not exist. The only local snapshot there is
`models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
which is a Qwen3.5-4B VL/text wrapper, not a Nemotron-H checkpoint. It is useful
as a control for the Qwen3.5 text path and for validating the fallback reference
machinery after vLLM proved brittle.

Evidence:

- vLLM 0.22.1 on `/home/sadara/vllm0.22.1` loads the Qwen3.5-4B text-only model
  when run with the same host fixes used for Nemotron plus
  `HIPFIRE_VLLM_HIDE_FLASH_ATTN=1 --language-model-only`. Without that flag,
  the broken local `flash_attn_2_cuda` extension aborts import; without
  text-only mode, vLLM's multimodal dummy profiling tries to allocate 256 GiB.
- Even with `--language-model-only --max-num-seqs 1 --max-num-batched-tokens 256`
  and eager execution, vLLM hangs after loading 7.99 GiB of weights and logging
  GDN/Triton plus Mamba/attention page sizing. The engine spins CPU-bound and
  never reaches generation, so vLLM is not a usable reference for this local
  Qwen snapshot on this host.
- `benchmarks/nemotron/run_transformers_reference.py` is now the generic CPU
  Transformers fallback reference. With GPU visibility disabled, it runs the
  same closed-think 2+2 prompt in float32 on CPU and writes
  `/tmp/qwen35_4b_transformers_cpu_closed2p2.json`: generated text
  `2 + 2 equals 4.<|im_end|>`, tokens
  `[17, 478, 220, 17, 16327, 220, 19, 13, 248046]`, ~1.83 tok/s.
- `hipfire-quantize --format mq4` successfully converted the local snapshot to
  `/home/sadara/.hipfire/models/qwen3.5-4b-local-models-mq4.hfq`, 2.59 GB on
  disk. The quantizer skipped 454M visual/MTP params by default, wrote 426 text
  tensors, and stamped arch `qwen3_5`.
- `hipfire chat` loaded that MQ4 artifact through the normal daemon path
  (2.40 GiB payload, Q8 KV cache, FP32 DeltaNet state) and generated
  `2+2 equals 4.` in 8 tokens at ~61 tok/s.
- `greedy_dump_top5` over the exact CPU-reference prompt token IDs produced
  `2 plus 2 equals 4.<|im_end|>`; this run force-switched KV to FP32 because the
  artifact still contains BF16 side tensors, so it is a diagnostic rather than a
  byte-identical chat replay. The first divergence from the CPU reference is a
  near-tie at generated step 1: CPU FP32 ranks token `478` (`" +"`) first and
  token `5346` (`" plus"`) third, while Hipfire MQ4 ranks `5346` first and
  `478` second with margin 0.142. The answer remains coherent.

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
| 2026-06-25 MoE prefill slice | `MoeRelu2Gpu::prefill` row-wise composition inside the batched residual contract; MoE models no longer hard-disable `can_batched_prefill()` | MoE block prefill max|Δ|=4.47e-8; real 30B HFQ prefill-vs-decode max|Δlogit|=8.106e-6, argmax 1052 |

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
uses the HFQ4G256 batched GEMM against the rotated weight layout. MoE blocks
currently compose the validated decode primitive row-by-row inside the same
`[seq,hidden]` contract; expert-sorted/grouped MoE prefill remains a throughput
optimization, not a correctness prerequisite.

Validation run locally on gfx1151:
- `test_model_prefill_gpu`: synthetic f32 Mamba/MLP/attention model,
  max|Δlogit|=2.98e-7, argmax match.
- `test_model_prefill_hfq_gpu`: real `/tmp/nano4b-mq4-protected.hfq`,
  max|Δlogit|=1.29e-5, argmax match.
- `test_block_q8_state_gpu`: opt-in Q8 Mamba-2 SSM state
  (`HIPFIRE_NEMOTRON_SSM_STATE=q8`) preserves the prefill/decode handoff
  (q8 prefill-vs-q8 decode max|Δ|=2.98e-8) and stays close to the f32-state
  decode reference (max|Δ|=7.861e-5) on the synthetic block.
- `HIPFIRE_NEMOTRON_SSM_STATE=q8 test_model_prefill_hfq_gpu`: real
  `/tmp/nano4b-mq4-protected.hfq`, max|Δlogit|=2.38e-4, argmax match.
- `HIPFIRE_NEMOTRON_SSM_STATE=q8 hfq_vs_f32`: protected HFQ tracks the
  safetensors reference with q8 SSM state enabled in both paths:
  argmax match, logit cosine 0.999910, mean|Δ|=0.03916, max|Δ|=0.30284.
- `bench_prefill_hfq_gpu`: release fresh-process protected-mq4 benchmark on
  gfx1151:
  - seq=128, warmup=2, iters=5: batched mean=2476.31ms (51.7 tok/s),
    decode-loop mean=3356.73ms (38.1 tok/s), speedup=1.36x.
  - seq=256, warmup=1, iters=3: batched mean=5101.45ms (50.2 tok/s),
    decode-loop mean=6713.97ms (38.1 tok/s), speedup=1.32x.
- `test_moe_gpu`: isolated MoE block forward and row-wise prefill both match the
  CPU oracle with max|Δ|=4.47e-8 on gfx1151.
- `test_load_nano30b_hfq` on the real 30B HFQ artifact now validates
  prefill-vs-decode: default 2-token smoke max|Δlogit|=6.676e-6, argmax match;
  full 29-token closed-think 2+2 prompt max|Δlogit|=8.106e-6, final argmax
  `1052` (`4`).
- Commit hooks for `211888d7a` and `27f994e00`: rustfmt, clippy, short
  coherence battery (no hard errors), fast agentic gate, and MQ4 speed gate all
  passed. Tiny-fixture golden still drifted on existing Qwen fixtures and
  escalated to the full short coherence battery in both hooks.

## FU5 — REMAINING

1. **Default q8 SSM-state promotion.** The opt-in path is implemented and
   validated through the block and protected-HFQ model gates. Keep FP32 as the
   default correctness floor until longer-generation quality evidence decides
   whether `HIPFIRE_NEMOTRON_SSM_STATE=q8` should become the default.
2. **(Optional) broaden prefill benchmarks.** Current evidence covers the
   protected mq4 HFQ artifact on gfx1151 plus correctness for the real 30B HFQ
   artifact. A fuller benchmark-grade sweep can add f32/q8, 30B prompt-length
   sweeps, and row-wise-MoE vs future expert-sorted MoE comparisons.
3. **(Optional) chunked masked-flash for very long prompts.** The attention
   prefill uses a single masked-flash block (`block_cols=seq`); shared-mem scales
   with seq. Fine for normal prompts; long-context needs block tiling.

## FU6 (Nano-30B MoE) — in progress

Same `nemotron_h` arch but hidden=2688, 52 layers, pattern `MEMEM*EMEM…` →
introduces a new **'E' (MoE) block**. The first bounded slice is now in place:

- `BlockKind::Moe` parses `E`; `mixer_profile()` still excludes MoE because it
  is FFN-only, not recurrent/KV state.
- `NemotronHConfig` parses the Nano-30B MoE fields:
  `n_routed_experts=128`, `num_experts_per_tok=6`,
  `moe_intermediate_size=1856`, `n_shared_experts=1`,
  `moe_shared_expert_intermediate_size=3712`, `n_group=1`,
  `topk_group=1`, `norm_topk_prob=true`, `routed_scaling_factor=2.5`.
- The live safetensors index was checked for the first MoE block. Tensor names
  are `backbone.layers.1.mixer.gate.weight`,
  `backbone.layers.1.mixer.gate.e_score_correction_bias`,
  `backbone.layers.1.mixer.shared_experts.{up_proj,down_proj}.weight`, and
  split per-expert 2D tensors
  `backbone.layers.1.mixer.experts.{E}.{up_proj,down_proj}.weight`.
- `hipfire-quantize` now classifies those split MoE weights as quantizable,
  keeps the correction-bias sidecar out of lossy quantization, and Q8-protects
  `*.mixer.gate.weight` routers so top-k expert selection is not decided by
  base MQ4 router noise.
- `MoeRelu2Gpu` implements a decode-first correctness path: router GEMV →
  sigmoid → bias-aware top-k/renormalize/scale, shared ReLU² expert, and a
  simple selected-expert loop using existing `MlpRelu2Gpu` blocks. Its prefill
  path applies that validated primitive row-by-row to materialize `[seq,hidden]`
  outputs, which unblocks model-level batched prefill for 30B while keeping
  expert-sorted grouping as a later performance optimization.
- Validation: `cargo run -p hipfire-arch-nemotron --example test_moe_gpu`
  passes on gfx1151 with max|Δ|=4.47e-8 against the CPU oracle for both
  single-row forward and sequence prefill.
- The real 30B checkpoint was quantized with the rebuilt `hipfire-quantize`
  mq4 policy to
  `/home/sadara/.hipfire/models/nemotron-3-nano-30b-a3b-mq4.hfq`.
  Quantization evidence from `/tmp/nemotron30b-quant.log`:
  31,577,940,288 total params, 31,576,989,696 quantized params, mean quant
  error 0.00094749, max quant error 0.05048829, `Done: 25681.0 MB written`.
  Artifact sha256:
  `3660ae47c8f7309110d85ee2c013629cb6437b2fdfbb76d41a1a7cb3a49fe2f6`.
- `test_load_nano30b_hfq` now validates the real HFQ artifact through the
  Hipfire path: load config + HFQ, assert hybrid batched prefill is available,
  compare prefill logits against the decode loop, and fail on non-finite logits.
  Local gfx1151 smoke passed on the default 2-token prompt with max|Δlogit|=
  `6.676e-6`, argmax match, final argmax=1307.
- The rebuilt daemon can load/generate/unload the same 30B HFQ artifact through
  the JSONL serving path. Load response reports
  `arch=nemotron_h`, `dim=2688`, `layers=52`, `vocab=131072`, and
  `model_file_bytes=25680997504`; stderr reports the real block mix
  `(23 M / 6 * / 0 - / 23 E)`. With the current checkout daemon and the Mamba
  scale fix, greedy closed-think 2+2 returns `4` in one token.
- A Lyra real-Mamba Transformers reference for the same 30B checkpoint and the
  same closed-think 2+2 prompt IDs completed on gfx1151:
  `/tmp/nemotron30b_lyra_real_closed2p2.npz` plus
  `/tmp/nemotron30b_lyra_real_closed2p2.meta.json`. The rendered prompt hash is
  `15b33ada01389ee985aa13f846b8e9f9efcc2a5375e9528da8e5bc313789107a`; the
  reference final top-5 is `[1052, 31035, 1784, 1050, 31106]`
  (`4`, `Four`, `The`, `2`, `Answer`) with generated token `1052` (`4`) and
  top-2 margin 3.125. The run used `torch=2.12.0a0+rocm7.13.0a20260411`,
  `mamba_import=real`, `mamba_reference=remote`, and restored 23 trained
  `dt_bias` tensors.
- The 30B HFQ-vs-BF16 bisect found the old comma loop's root cause: Hipfire was
  applying the dense Nano-4B Mamba `out_proj.weight` runtime scale
  (`1/sqrt(num_layers)`) to the MoE 30B checkpoint. CPU layer-0 reconstruction
  from safetensors plus the Lyra dump shows Nano-4B matches BF16 with that
  scale, while Nano-30B-A3B matches BF16 only with no extra runtime scale. The
  fix is `NemotronHConfig::mamba_out_proj_runtime_scale()`: dense 4B keeps the
  scale, MoE variants use `1.0`.
- With the scale fix and the original canonical artifact
  `/home/sadara/.hipfire/models/nemotron-3-nano-30b-a3b-mq4.hfq`,
  `test_load_nano30b_hfq` over the same 29 prompt IDs now reports final argmax
  `1052` (`4`). The HFQ-vs-BF16 bisect moves the first divergence from
  `hidden_1` to `hidden_2`, and final logits are close enough to preserve the
  reference top token: HF top-5 `[1052, 31035, 1784, 1050, 31106]`, Hipfire
  top-5 `[1052, 1784, 1050, 31035, 31106]`, logit rel delta `0.0454`.
- With hybrid MoE prefill enabled, the same 29-token prompt matches the
  per-token decode loop with max|Δlogit|=`8.106e-6`, prefill argmax=1052, and
  decode argmax=1052.
- A rebuilt exploratory artifact with Mamba `in_proj` promoted to Q8,
  `/home/sadara/.hipfire/models/nemotron-3-nano-30b-a3b-inproj-q8-mq4.hfq`
  (sha256
  `49cdf0b534729bebc24ede0b8c243a848cf06b38ac06e61d20c72cdf7e37743f`,
  25,999.5 MB written), is slightly closer at the same boundary
  (logit rel delta `0.0386`) but is not required for the basic coherence fix.
- `HIPFIRE_DAEMON_BIN=target/debug/hipfire-daemon ./target/debug/hipfire chat
  --model /home/sadara/.hipfire/models/nemotron-3-nano-30b-a3b-mq4.hfq
  --temperature 0 --max-tokens 16 "Answer in one short sentence: What is 2+2?"`
  returns `4` in one token. Without `HIPFIRE_DAEMON_BIN`, this checkout can still
  pick the older installed daemon, which lacks the scale fix and may fail before
  arch dispatch.

Remaining FU6 work is throughput and admission-quality evidence at 30B scale:
replace row-wise MoE prefill with a batched/expert-sorted grouped path if 30B
prefill throughput matters, and run broader quality/perf evidence before
promoting any changed quant policy. The current policy is still an ingress
policy, not a quality-promoted calibration policy; router tensors are
Q8-protected, but expert promotion still needs router-hit and quality evidence
before any "better than baseline" claim.

## FU1 (coherence) — standing blocker

Daemon generation with the original unprotected mq4 artifact produces a newline
attractor; q8 and the fresh protected-mq4 artifact follow Hipfire f32 and stop
immediately on the closed-think 2+2 prompt. Forward numerics are validated
(FU4/FU5 prefill==decode, protected-mq4/q8-vs-f32 cosine 0.999967), and the
concrete EOS/Jinja/thinking-control work is done. The Python native-Mamba
reference now gives a repeatable local first-token/per-layer oracle and matches
Hipfire f32 on the closed-think prompt. vLLM 0.22.1 on the same prompt is
coherent, so the remaining blocker is now a concrete first-token divergence:
vLLM chooses `2` while Hipfire/native-HF choose immediate `<|im_end|>`.

Next useful FU1 work:

1. Use `benchmarks/nemotron/dump_hf_reference.py` as the first local oracle for
   every prompt under investigation; keep the rendered ChatML IDs and first-token
   top-k with the evidence.
2. Use `benchmarks/nemotron/run_vllm_reference.py` to capture vLLM 0.22.1
   references for 3-4 prompts, starting with the closed-think 2+2 prompt.
3. Compare Hipfire f32/q8 against vLLM at the generation boundary, then bisect
   the first divergence against the already added per-layer/block hooks.
4. Treat real 4-bit Nano-4B as a sensitivity issue until calibrated. The current
   `--format mq4` policy protects projection-back residual writers and Mamba
   `in_proj` as q8; an imatrix/AWQ/Lloyd pass is required before claiming true
   mq4 coherence.
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
cargo run -p hipfire-arch-nemotron --example test_moe_gpu            # MoE block + row-wise prefill
cargo run -p hipfire-arch-nemotron --example test_load_nano30b_hfq   # real 30B HFQ prefill-vs-decode
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

# Lyra real-Mamba Transformers control for the same prompt:
PYTHONPATH=/home/sadara/Lyra/build/mamba/build/lib.linux-x86_64-cpython-314:/home/sadara/Lyra/build/mamba:/home/sadara/Lyra/build/causal-conv1d/build/lib.linux-x86_64-cpython-314:/home/sadara/Lyra/build/causal-conv1d \
LD_LIBRARY_PATH=/home/sadara/.venv/lib/python3.14/site-packages/_rocm_sdk_core/lib:/home/sadara/.venv/lib/python3.14/site-packages/torch/lib:$LD_LIBRARY_PATH \
/home/sadara/.venv/bin/python -u benchmarks/nemotron/dump_hf_reference.py \
  --model /home/sadara/Models/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --mode jinja --thinking off \
  --text 'Answer in one short sentence: What is 2+2?' \
  --mamba-import real --mamba-reference remote \
  --dtype bfloat16 --device cuda \
  --max-new-tokens 1 --out /tmp/nemotron_lyra_real_closed2p2.npz

# Quantizer policy smoke: protected mq4 should no longer flip EOS to newline.
hipfire-quantize --input <Nano-4B-BF16-snapshot> --output /tmp/nano4b-mq4-protected.hfq --format mq4 --threads 16
NEMO_TOKENS='10,25708,1010,11,1010,10,3263,1010,31106,1294,1925,4958,19286,1058,5675,1395,1032,1050,1043,1050,1063,11,1010,10,1503,19464,1010,12,13' \
  cargo run -p hipfire-arch-nemotron --example hfq_vs_f32 -- /tmp/nano4b-mq4-protected.hfq

# vLLM 0.22.1 coherent reference on this host:
PYTHONPATH=/home/sadara/vllm0.22.1/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi \
VLLM_ROCM_USE_SKINNY_GEMM=0 \
VLLM_ROCM_USE_AITER_LINEAR=0 \
VLLM_ROCM_USE_AITER_TRITON_GEMM=0 \
/home/sadara/vllm0.22.1/bin/python3 benchmarks/nemotron/run_vllm_reference.py \
  --thinking off \
  --text 'Answer in one short sentence: What is 2+2?' \
  --temperature 0 \
  --max-tokens 16 \
  --out /tmp/nemotron_vllm_closed2p2.json

# Generic Transformers CPU fallback reference (used when vLLM is not usable):
CUDA_VISIBLE_DEVICES= \
HIPFIRE_HIDE_FLASH_ATTN=1 \
OMP_NUM_THREADS=16 \
MKL_NUM_THREADS=16 \
/home/sadara/vllm0.22.1/bin/python3 benchmarks/nemotron/run_transformers_reference.py \
  --model /home/sadara/Models/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --thinking off \
  --text 'Answer in one short sentence: What is 2+2?' \
  --dtype float32 \
  --device cpu \
  --max-new-tokens 16 \
  --out /tmp/qwen35_4b_transformers_cpu_closed2p2.json
```
