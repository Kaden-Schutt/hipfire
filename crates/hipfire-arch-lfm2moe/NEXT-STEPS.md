<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->
# LFM2.5-8B-A1B (arch_id 11) — status & next steps

## Status: forward VALIDATED, real model COHERENT, decode 241 tok/s (Q8) / 259 tok/s (proj-MQ4 opt-in) on gfx1201

Shipped on branch `lfm2moe/impl` (off `minimax/m2.7-impl`):

- **crate `hipfire-arch-lfm2moe`** — config / loader / forward (free-fn
  `decode_step`) / examples (`dump_lfm2moe_hidden_states`, `infer_lfm2moe`).
- **kernel `conv1d_gated_decode.hip`** (+ dispatch wrapper) — the LIV double-gated
  short-conv decode op (B·x pre-gate, depthwise causal conv runtime-K, C·conv_out
  post-gate, rolling conv-state advance) fused in one launch.
- **quantizer** — `"lfm2_moe" => 11`, `is_lfm2moe` ingest (bf16 → experts MQ4G256,
  expert_bias F32, all else Q8).
- **daemon** — arch_id 11 registered (LoadedModel slots, load branch,
  `generate_lfm2moe`, Cargo `arch-lfm2moe`).

**Validation:** tiny-oracle per-layer cosine 0.99996 → 0.99931 (≥0.999, 4-bit
experts) vs HF `Lfm2MoeForCausalLM`; real mq4 model coherent through the daemon
(Tokyo / working Fibonacci / 80 km/h), ~247–253 tok/s decode.

## Architecture recap (ground truth: transformers lfm2_moe + repo config.json)
- 24 layers; `layer_types` = 18 `conv` + 6 `full_attention` (attn at L 2,6,10,14,18,21).
- FFN: layers 0–1 dense SwiGLU (inter 7168); 2–23 top-4 MoE (32 experts, moe_inter 1792).
- hidden 2048, 32 q-heads / 8 kv-heads, head_dim 64, RoPE θ 5e6, RMSNorm eps 1e-5
  (standard, no +1), conv K=3 depthwise, tie_word_embeddings, vocab 128000.
- MoE routing: sigmoid(gate) + expert_bias (selection only) → top-4 → gather
  unbiased → norm_topk → scale (1.0). Maps to `deepseek4_moe_topk_bias_aware_f32`.

## Next steps (priority order)

### 0. DONE — MQ6-experts QUALITY lever (opt-in): `HIPFIRE_LFM2_EXPERT_MQ6=1`
rocprofv3 showed decode is BANDWIDTH-bound (gemv_q8_0 = 49.5%); bf16-referenced
KLD showed the 4-bit EXPERTS (not the projections) dominate the quality gap. So
added 6-bit experts via a new HFQ6 indexed MoE GEMV kernel
(`gemv_hfq6g256_moe_gate_up_k8_indexed_batched`; the `_down_*_expanded` sibling
already existed) + forward routing on MQ6G256 expert dtype + quantizer flag.
Model `lfm2.5-8b-a1b.mq6e`: **KL vs bf16 0.424→0.135 (−68%), top-1 72.7→79.5%,
for −16% decode (241→203 tok/s) + 1.8 GB VRAM (4.6→6.4).** Coherent (chat-framed
coherence_probe verdict OK, 0 hard/0 soft). Opt-in; default stays mq4 (max speed).
See design doc "MQ6-experts". NOTE: the bare-prompt `infer_lfm2moe` smoke loops
("France is France is") — that's the documented bare-completion artifact (mq4
does it too), NOT a kernel bug; validate coherence via the daemon/ChatFrame.

### 0. DONE — proj-MQ4 perf lever (+7.2%, opt-in): `HIPFIRE_LFM2_PROJ_MQ4=1`
4-bit-ing the dense projections (conv in/out_proj, attn q/k/v/out, dense MLP)
gives **258.8 vs 241.5 tok/s = +7.2%** on matched full-256-tok runs (6 runs each,
±0.1%), real model coherent (Paris / 80 km/h). OFF by default due to a quality
cost (tiny-oracle cosine 0.94 < 0.99 4-bit gate — quant noise, milder on the real
wide projections). Fast variant model: `~/.hipfire/models/lfm2.5-8b-a1b.mq4p`. To
make default: KLD/PPL vs Q8 first. See design doc "PERF TUNING".
(Measurement note: an early "+18%" was an EOS-truncation artifact and a "WASH"
was a fabricated number — corrected; the +7.2% is from grep-able matched logs.)

### TESTED NEGATIVE — compile-time-K3 conv: no-op (+0.25%, within noise), reverted
Implemented `conv1d_gated_decode_k3_f32` (unrolled 3-tap, no runtime-K loop /
win[] array), dispatched on K==3, verified bit-identical (tiny cosine 0.99910).
Matched A/B (5× 256-tok, fresh process): **242.1 vs 241.5 tok/s = +0.25%, within
noise.** The conv kernel is launch/latency-bound (one tiny single launch, ~3
FMAs), not ALU-bound — unrolling buys nothing. "K3 = launch-count reducer" was a
mis-framing: it's one already-single launch. Reverted (complexity for no gain).
See design doc "Tested NEGATIVE" for detail. **Do not re-attempt.**

### 1. Perf: further tuning (task #9) — baseline 241 tok/s (Q8) / 259 (proj-MQ4 opt-in)
Real decode bound at batch=1 is launch OVERHEAD (~330 launches/tok), not the conv
body. Levers that genuinely cut launch COUNT (all higher-effort, all need cosine
+ coherence re-validation):
- **HIP graph capture** of the per-token kernel sequence — amortize per-launch
  cost across the ~330 launches; likely the biggest single decode lever.
- **rmsnorm→gemv fusion** for the Q8 path (the existing fused-rmsnorm-rotate is
  MQ-only; a Q8 fused variant would remove 1 launch × 24 layers).
- **MoE down+combine fusion** — an hfq4 residual-scaled-down kernel (the
  MQ2-Lloyd path already fuses these; no hfq4 equivalent yet) removes 1 launch ×
  22 layers.
- **Bandwidth axis**: proj-MQ4 (+7.2%, shipped opt-in); MQ6-proj untried
  middle-ground (less quality loss, ~half the saving).
- **Prefill batching** (large win for long prompts; needs batched conv1d scan +
  batched attn/MoE — substantial).
Always: warm first (`HIPFIRE_DPM_WARMUP_SECS=10`, fresh process, byte-identical
prompt, median of ≥3 MATCHED full-length runs — NOT EOS-truncated). Treat ≥5% as
real; re-validate cosine + coherence after every gain.
- **Fuse the conv gates into in_proj/out_proj** if the 3 conv launches/layer show
  up hot — e.g. a fused in_proj+gate or out_proj-residual (minimax's MoE uses
  `weight_gemv_residual` fusion; conv out_proj already does).
- **MoE GEMV variant sweep** — confirm the batched k4 path is optimal vs the
  non-batched indexed path at batch=1; the wave64 vs wave32 split is auto (gfx1201
  is wave32). Check whether `moe_down_combine` + separate down can fuse to the
  residual-scaled down (minimax's MQ2-Lloyd path does this — no hfq4 equivalent yet).
- **Re-validate after every gain**: cosine (tiny) must stay ≥0.999 AND coherence
  gate must stay PASS. A tok/s win that breaks either is a regression (see CLAUDE.md
  synth-win→prod-falsify history).

### 2. Quality: KLD vs llama.cpp / higher-precision sweep
- The current mq4 (4-bit experts) is the VRAM-efficient default. Run a precision
  sweep (experts mq4 → mq6 → q8) and measure KLD/perplexity vs an f16 reference
  (or llama.cpp if it gains lfm2_moe support) to quantify the 4-bit quality cost.
- Per-tensor tiering (e.g. first/last MoE layers mq6, middle mq4) à la the K-map
  path is a lever if KLD is too high.

### 3. Prefill batching (perf, larger win for long prompts)
- Currently prefill is per-token `decode_step` (same as minimax bring-up). A
  batched prefill kernel set (like qwen35's `forward_prefill_batch`) would speed
  long-context prompt ingestion substantially. Requires a batched conv1d scan
  (the `conv1d_silu_split_f32_n` N-token scan is the reference shape) + batched
  attention/MoE. Non-trivial; only worth it once decode is tuned.

### 4. Coherence gate entry (task #8)
- Added 2 lfm2 rows to `scripts/coherence-gate.sh` (lfm2-cap, lfm2-reason). They
  skip automatically when the model file is absent. Keep them in the matrix.

### 5. Spec-decode / DFlash (optional, large effort)
- The conv-state cache already has a tree/snapshot pattern in qwen35
  (`conv1d_silu_split_tree`, speculative.rs conv_state_bufs). If LFM2 ever needs
  spec-decode, that's the reference — but it's out of scope for the base port.

## Known caveats
- **Chat framing required**: raw completion prompts loop on this instruct/thinking
  model. The daemon ChatFrame handles it; `infer_lfm2moe` with a bare prompt will
  degenerate (not an arch bug — cosine ≥0.999 proves the forward).
- **conv_L_cache non-snake-case warning** in config.rs (the serde field mirrors the
  HF config key `conv_L_cache`) — harmless `#[warn(non_snake_case)]`; add
  `#[allow(non_snake_case)]` or `#[serde(rename)]` to silence if desired.
- **MoE GEMV is the `_k8`-named family with runtime k_top=4** — correct (the name
  is historical; k_top is a runtime arg). No `_k4` GEMV kernel was needed.
