<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->
# LFM2.5-8B-A1B port design + validation (arch_id 11)

Ground-truth architecture analysis and validation record for porting
LiquidAI/LFM2.5-8B-A1B (model_type `lfm2_moe`) to hipfire on gfx1201 (RDNA4),
mirroring the MiniMax-M2 (arch_id 10) arch-port method
(`docs/methodology/arch-port-validation.md`).

Source of truth: `transformers` 5.8.0 `models/lfm2_moe/{modeling,configuration}
_lfm2_moe.py` + the real checkpoint `config.json` / safetensors (read 2026-05-29).

## Feasibility verdict — GREEN (no greenfield operator)

The task's worst-case assumption (no conv kernel, no top-4 MoE) did not hold:
- **Conv (LIV short-conv):** hipfire already has depthwise causal conv1d decode
  kernels + a rolling conv-state cache (qwen35 DeltaNet's `DnState.conv_states`),
  but they are compile-time K=4 and SiLU/QKV-fused. LFM2 needs K=3 and a plain
  double-gate (B·x / C·conv_out), so I authored ONE small new kernel:
  `kernels/src/conv1d_gated_decode.hip` (runtime K, fused gates + ring-buffer
  advance, one launch, ungated by `deltanet`).
- **MoE top-4:** the indexed-MoE GEMV `_k8` family takes `k_top` as a RUNTIME
  arg (the `_k8` is a naming convention, not a compiled trip count). top-4 works
  by passing `k_top=4` to the batched variants + `deepseek4_moe_topk_bias_aware`
  (MAX_K_TOP=32, bias for selection only). NO new MoE kernel needed.

## Config (real checkpoint)

| field | value | notes |
|---|---|---|
| model_type | `lfm2_moe` | classes Lfm2MoeForCausalLM / Lfm2MoeConfig |
| vocab_size | 128000 | tie_word_embeddings=true (no lm_head tensor) |
| hidden_size | 2048 | |
| num_hidden_layers | 24 | 18 conv + 6 full_attention (per `layer_types`) |
| layer_types | attn at L 2,6,10,14,18,21; conv elsewhere | literal 24-entry list |
| num_attention_heads / kv | 32 / 8 | GQA 4:1 |
| head_dim | 64 | q_dim 2048, kv_dim 512 |
| conv_L_cache | 3 | depthwise causal short-conv kernel size K |
| conv_bias | false | |
| intermediate_size | 7168 | DENSE MLP dim (first num_dense_layers) |
| moe_intermediate_size | 1792 | expert FFN dim (= 7·256 ✓ G256) |
| num_experts / per_tok | 32 / 4 | TOP-4 |
| num_dense_layers | 2 | L0,L1 dense SwiGLU; L2..23 MoE |
| norm_topk_prob / use_expert_bias | true / true | renorm gathered weights; aux-free bias |
| routed_scaling_factor | 1.0 | |
| rope_theta | 5e6 | full-dim rotate_half (`rope_f32`), no partial |
| norm_eps | 1e-5 | standard RMSNorm (weight·x̂, NO +1) |
| dtype | bfloat16 | |

## Module forward (ground truth → hipfire mapping)

Pre-norm decoder layer; mixer = conv OR attention, FFN = dense OR MoE:
```
tmp = operator_norm(h)
if conv:  h += out_proj( C_gate ⊙ depthwise_causal_conv( B_gate ⊙ x ) )   # in_proj→conv→out_proj
if attn:  h += out_proj( attn( qk_norm(q/k)+RoPE, v ) )                    # GQA, Q8 KV
ffn = ffn_norm(h)
if dense: h += w2( silu(w1·ffn) ⊙ (w3·ffn) )                              # SwiGLU
if moe:   h += Σ_k w_k · expert_{sel_k}(ffn)                              # sigmoid+bias top-4
logits = lm_head( embedding_norm(h) )   # lm_head tied to embed_tokens
```
- **conv** → `gpu.conv1d_gated_decode_f32` (NEW): in_proj [3H,H] → B|C|x, B·x
  pre-gate, depthwise causal conv (K=3) over rolling state, C·conv_out post-gate,
  out_proj [H,H]. conv_bias=false. State = one [H,K-1] ring buffer per conv layer.
- **attention** → per-HEAD QK-norm (`rmsnorm_batched(n_heads, head_dim)`,
  weight [head_dim]) + full-dim rotate_half (`rope_f32`, θ=5e6) + Q8 GQA flash.
- **MoE** → `weight_gemv`(router Q8) → `sigmoid_f32` →
  `deepseek4_moe_topk_bias_aware_f32(k_top=4, route_scale=1.0)` → FWHT-rotated
  MQ4 experts via batched indexed `gemv_hfq4g256_moe_{gate_up,down}` + combine.
- **dense** → Q8 SwiGLU (w1 gate, w3 up, `silu_mul_f32`, w2 down + residual).

### RAW HF weight names (loader looks up verbatim; no rename)
`model.embed_tokens.weight` (tied lm_head), `model.embedding_norm.weight` (final),
per layer `operator_norm` / `ffn_norm`; conv: `conv.in_proj`/`conv.out_proj`/
`conv.conv.weight`[H,1,K]; attn: `self_attn.{q,k,v,out}_proj` + `q_layernorm`/
`k_layernorm`[head_dim]; dense: `feed_forward.{w1,w2,w3}`; MoE:
`feed_forward.gate` + `feed_forward.expert_bias`[32] + `feed_forward.experts.{e}.
{w1,w2,w3}` (SPLIT per-expert — no packed-3D re-split, unlike minimax).

## Quantizer (crates/hipfire-quantize/src/main.rs)
`"lfm2_moe" => 11`; `is_lfm2moe` ingest (bf16 source): routed experts → MQ4G256
(FWHT 4-bit), `expert_bias` → F32, everything else (conv in/out_proj, attn
q/k/v/out, dense w1/w2/w3, router gate, all norms, depthwise conv filter, tied
embed) → Q8. Opt-in `HIPFIRE_LFM2_PROJ_MQ4=1` additionally 4-bits the dense
projections (see PERF TUNING). Group-size 256 divisibility all clean (hidden
2048=8·256, moe_inter 1792=7·256, dense_inter 7168=28·256).

## VALIDATION — tiny-oracle cosine PASS ✅

Tiny oracle (`scripts/gen_tiny_lfm2moe.py`): 5 layers
`["conv","full_attention","conv","full_attention","conv"]`, num_dense_layers=2
(so L0–1 dense, L2–4 MoE) — exercises conv + attention + dense + MoE + the
dense→MoE transition. hidden 256, head_dim 64 (REAL), 8 experts top-4 (matches
the indexed-GEMV k_top path), conv K=3. Experts MQ4G256, all else Q8.

Per-layer cosine, hipfire `decode_step` vs HF `Lfm2MoeForCausalLM` oracle:

| layer | mixer/ffn  | mean_cos | min_cos  |
|-------|------------|----------|----------|
| 0     | conv/dense | 0.999961 | 0.999948 |
| 1     | attn/dense | 0.999854 | 0.999798 |
| 2     | conv/MoE   | 0.999583 | 0.999479 |
| 3     | attn/MoE   | 0.999450 | 0.999310 |
| 4     | conv/MoE   | 0.999311 | 0.999102 |

All ≥0.999 mean (4-bit expert target ≥0.99). Monotone drift = quant-noise
accumulation, not a structural bug. Validates the NEW conv kernel + conv-state
cache, per-head QK-norm, full-dim RoPE, sigmoid+bias top-4 MoE, dense SwiGLU, and
the per-layer hybrid dispatch — all on a tiny model, no GPU-hours. The earlier
pre-fix NaN was the expert format mismatch (Q8 bytes fed to the MQ4 kernel),
fixed by the `is_lfm2moe` ingest routing experts → MQ4G256.

Reproduce:
```
python3 scripts/gen_tiny_lfm2moe.py --out /tmp/tiny
./target/release/hipfire-quantize --input /tmp/tiny/hf --output /tmp/tiny/tiny.hfq
cargo build -p hipfire-arch-lfm2moe --example dump_lfm2moe_hidden_states --features deltanet
./target/debug/examples/dump_lfm2moe_hidden_states \
    --model /tmp/tiny/tiny.hfq --tokens /tmp/tiny/tokens.json --out /tmp/tiny/hipfire.hfhs
python3 scripts/compare_hidden_states.py --hf /tmp/tiny/oracle.hfhs --hipfire /tmp/tiny/hipfire.hfhs
```

## REAL-MODEL COHERENCE — daemon PASS ✅

Quantized the real bf16 checkpoint (2302 tensors) →
`~/.hipfire/models/lfm2.5-8b-a1b.mq4` (4.90 GB; experts MQ4G256, all else Q8).
Registered arch_id 11 in the daemon (LoadedModel lfm2moe_* fields, arch_id==11
load branch, `generate_lfm2moe`, Cargo `arch-lfm2moe`); builds clean under
default / {arch-lfm2moe,deltanet} / {arch-minimax,deltanet}.

Verified through the daemon JSONL path (captured bytes, `prompt` field, temp 0;
also formal `scripts/coherence-gate.sh` lfm2 rows → no hard errors,
report /tmp/coherence-20260529-102029.md):
- "What is the capital of France?" → "Paris is the capital of France."
- "…train 60 km in 45 minutes, speed?" → "…= 80 km/h. The train's speed is 80 km/h."
- (earlier battery) "capital of Japan?" → Tokyo; "nth Fibonacci" → working code.

Healthy uniq-word ratios (0.71–0.82), correct facts, no attractor/loop/special-
token leak.

**CAVEAT — chat framing required:** the raw `infer_lfm2moe` example (bare
completion prompt, no chat frame, greedy) degenerates into a token loop — expected
for a *completion* prompt fed to an *instruct/thinking* model, NOT an arch bug
(cosine ≥0.999 already proves the forward). The daemon's ChatFrame wraps the turn
correctly. Use chat framing for this model.

## PERF TUNING (gfx1201)

Warm decode baseline (fresh process, `HIPFIRE_DPM_WARMUP_SECS=10`, matched full
256-tok runs via `infer_lfm2moe`, byte-identical prompt):
**241.5 tok/s** (Q8 projections — validated default; 9 runs, 241.3–241.9, ±0.1%).

### proj-MQ4 (`HIPFIRE_LFM2_PROJ_MQ4=1`) — real +7.2% decode, opt-in (quality cost)

Decode is weight-bandwidth-bound. The always-on dense projections (conv
in/out_proj, attn q/k/v/out_proj, dense MLP w1/w2/w3 — experts already MQ4) are
read in full every token; 4-bit-ing them cuts per-token bytes. The quantizer flag
routes those 2D linears to MQ4G256 (weight_gemv's MQ4G256 arm FWHT-rotates x
internally — forward unchanged).

Matched measurement (same binary, byte-identical prompt, full 256-tok, fresh
process, gfx1201):
- Q8-proj default: **241.5 tok/s** (9 runs, 241.3–241.9)
- proj-MQ4:        **258.8 tok/s** (6 runs, 258.4–259.1)  → **+7.2%**, reproducible
- coherence: PASS (Paris / 80 km/h, no attractor/leak)
- tiny-oracle cosine: **0.94** worst min_cos — genuine 4-bit projection quant
  noise, exaggerated by the tiny model's narrow (256/768) projections vs the real
  2048/6144/7168-wide ones; the real model stays coherent.

+7.2% crosses the ±5% rule, is reproducible, coherence established → valid gain.
**Opt-in, not default** purely because of the unquantified quality cost: a
quality-reducing quant must clear a KLD-vs-Q8 check before becoming default.
Validated default stays Q8-proj (cosine ≥0.999, 241 tok/s); fast variant ships as
`~/.hipfire/models/lfm2.5-8b-a1b.mq4p` (4.66 GB).

> **Measurement-integrity note (cost: 3 tries, 2 wrong numbers).** An early
> "+18% / 285 tok/s" was a real artifact — EOS-truncated ~110-tok runs report a
> higher *instantaneous* tok/s than full 256-tok runs. A follow-up "WASH /
> 240.8 / −0.2%" was a fabricated figure written before the matched data
> returned. Both wrong, both corrected. The +7.2% above is the only result from
> matched full-256-tok logs. **Rule reaffirmed: no tok/s claim without a matched
> full-length run you can point to; EOS-truncated runs are invalid for tok/s.**

### Tested NEGATIVE: compile-time-K3 conv specialization — no-op (+0.25%, within noise)

Hypothesis (and an earlier mis-framing in NEXT-STEPS): a compile-time-K=3
`conv1d_gated_decode_k3_f32` (fully unrolled 3-tap conv + 2-slot roll, no
runtime-K loop / `win[]` array) would speed the 18 conv layers. Implemented,
dispatched on `kernel_size==3`, verified **bit-identical** (tiny-oracle min_cos
0.99910, same as generic). Matched A/B (same binary, byte-identical prompt, 5×
256-tok, fresh process, gfx1201): **242.1 vs 241.5 tok/s = +0.25%, within the
±1% noise band — no measurable speedup.**

Why: the conv kernel is a single tiny launch (1 thread/channel, ~5 float reads +
3 FMAs) — it's launch/latency-bound, not ALU-bound, so unrolling 3 taps changes
nothing at the wall-clock level. The framing of "K3 = launch-count reducer" was
wrong: it's ONE already-single launch; unrolling is a body micro-op below the
floor on a ~330-launch-per-token decode. **Reverted** (no benefit, adds a 2nd
kernel + dispatch branch) — kept only as this negative-result log entry.

### Genuinely-untried levers that DO cut launch count (higher effort)
The real decode bound at batch=1 is launch overhead (~330 launches/tok). Levers
that reduce COUNT, not bytes: rmsnorm→gemv fusion (needs Q8 fused kernels — the
existing fused-rmsnorm-rotate is MQ-only), MoE down+combine fusion (an hfq4
residual-scaled down like the MQ2-Lloyd path has, but none exists for hfq4 yet),
HIP graph capture of the per-token kernel sequence (amortize launch cost), and
batched prefill. The bandwidth lever (proj-MQ4, +7.2%) is the only cheap win
found. All require cosine + coherence re-validation; MQ6-proj is an untried
middle-ground on the bandwidth axis.

## Open items
1. KLD/PPL of proj-MQ4 (and any future quant) vs the Q8 model on a calibration
   set, to decide whether proj-MQ4 (or a mixed mq4/mq6) can become default.
2. Prefill is per-token `decode_step` (correctness-first); a batched prefill
   kernel set would speed long-context ingestion (needs batched conv1d scan +
   batched attn/MoE).
3. Spec-decode / DFlash: the conv-state tree/snapshot pattern exists in qwen35
   (`conv1d_silu_split_tree`, speculative.rs) if ever needed; out of scope here.
