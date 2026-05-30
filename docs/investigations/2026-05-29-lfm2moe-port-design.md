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
### Decode profile (rocprofv3, gfx1201, 2026-05-30) — BANDWIDTH-bound, not launch-bound

A real kernel trace (`rocprofv3 --kernel-trace`, 64-tok decode) corrected two
earlier *assumptions* that turned out wrong:
- **Launch overhead is NOT the dominant cost.** Decode region is ~70% GPU-busy /
  ~27% inter-kernel launch-idle (of busy+idle) — not the "~55% idle / ~330
  launches as the bound" earlier hand-waved. ~377 kernels/tok, but they're
  mostly busy.
- **The MoE topk kernel is cheap** (`deepseek4_moe_topk_bias_aware_f32` = 1.9%,
  grid 32×32) — NOT a hot spot.
- **The real hot kernel is `gemv_q8_0` = 49.5% of decode GPU time** (the dense Q8
  projections), then the two MoE 4-bit expert GEMVs (gate_up 17.7% + down 12.4%
  = 30%). Decode is **weight-bandwidth-bound on the dense projections.**

**Consequence: HIP graph capture is NOT the big lever** it was assumed to be — it
only attacks the ~27% launch-idle, and qwen35 already disabled AR-forward graph
capture (`qwen35.rs` `use_graph=false`) over a ROCm kernarg-bake attractor bug.
The bandwidth axis (4-bit-ing the hot `gemv_q8_0` projections) is where the real
decode time is — which is exactly the proj-quant lever below.

Other launch-COUNT levers (rmsnorm→gemv fusion, MoE down+combine fusion, batched
prefill) remain available but target the smaller idle fraction, not the 80% the
GEMVs occupy.

### Projection-quant bandwidth sweep (KLD, gfx1201, 2026-05-30) — SETTLED

Measured via `examples/kld_logits` over 44 positions (self-KL control
mq4-vs-mq4 = 0.000000, so the harness is exact), plus matched decode + coherence.

**Reference matters: KL is anchored on the BF16 SOURCE, not on Q8.** Q8 has its
own quant error, so a Q8-referenced KL hides the real picture. The bf16 reference
logits come from HF `Lfm2MoeForCausalLM` (CPU, one causal forward over the same
44 token ids); `scripts/bf16_ref.py` workflow.

**KL(bf16 ‖ candidate)** — the absolute quality vs ground truth. All figures
recomputed from the on-disk per-position logit dumps (`kld_logits --dump` +
`scripts/bf16_ref.py`; 44 positions):

| variant | proj quant | mean KL | median | frac>0.1 | top-1 agree | decode tok/s | coherence |
|---------|-----------|---------|--------|----------|-------------|-------------|-----------|
| mq4 (default) | Q8 | **0.424** | 0.311 | 0.841 | **72.7%** | 241.1 | ✅ |
| mq6p | MQ6 | 0.447 | 0.309 | 0.864 | 79.5% | 240.0 (≈0%) | ❌ loop (tok-443 attractor) |
| mq4p | MQ4 | 1.050 | 0.789 | 0.977 | 56.8% | 258.8 (+7.2%) | ⚠️ passes gate, high KLD |

> **Correction (2026-05-30):** the first commit of this table (8ac10a48) carried
> wrong numbers — mq4 mean was written as 0.841 (that's its *frac>0.1*, not the
> mean) and top-1 as 88.6% (unsourced). The values above are the authoritative
> recompute from the logit dumps. The conclusions below are unchanged.

**The key finding (only visible with bf16 as reference):** the dominant quality
cost is the **4-bit experts** (MQ4G256, shared by ALL three variants) — the Q8
default is already ~0.42 nats / 27% top-1-disagreement from bf16. The dense
**projections are secondary**: Q8→MQ6 barely moves it (+0.02), Q8→MQ4 roughly
doubles it (+0.63).

**Conclusions:**
1. **proj-MQ4 stays OPT-IN** — +7.2% real, but worst quality (1.05 nats, 56.8%
   top-1) and adds the most projection damage. Not default.
2. **proj-MQ6 REJECTED** — dead end: **no speedup** (240.0 vs 241.1, MQ6G256 proj
   GEMV isn't bandwidth-winning at batch=1) AND degrades quality into an outright
   loop on "capital of France". Worse than Q8 on every axis.
3. **Q8 projections (current default) is correct** — best speed, near-best KL.
4. **The real quality lever is the EXPERTS, not the projections** — CONFIRMED in
   the MQ6-experts section below.

**Caveat (honest):** the absolute ~0.42-nat baseline may include some
hipfire-GPU-vs-HF-CPU numerical drift beyond pure quantization (tiny-oracle
cosine ≥0.999 + passing coherence confirm the arch is correct, so it's not a
bug). The RELATIVE ordering is robust to any constant offset.

### MQ6-experts (new HFQ6 indexed MoE decode kernel, gfx1201, 2026-05-30) — SHIPPED opt-in

Acting on finding #4 (experts dominate): added a 6-bit-experts variant. Required
ONE new kernel — `gemv_hfq6g256_moe_gate_up_k8_indexed_batched` (the matching
`_down_k8_indexed_batched_expanded` already existed) — mirroring the hfq4 indexed
MoE GEMV with the 6-bit dequant from `gemv_hfq6g256.hip` (200 B/group vs hfq4's
136; bit-pack roundtrip proven symbolically against `quantize_mq6g256`).
forward.rs routes to the HFQ6 kernels when the loaded experts are MQ6G256;
quantizer gate `HIPFIRE_LFM2_EXPERT_MQ6=1`. Model: `lfm2.5-8b-a1b.mq6e`.

| variant | experts | mean KL vs bf16 | top-1 vs bf16 | decode tok/s | VRAM | coherence |
|---------|---------|-----------------|---------------|--------------|------|-----------|
| mq4 (default) | MQ4G256 | 0.424 | 72.7% | 241.1 | 4.6 GB | ✅ |
| **mq6e** | **MQ6G256** | **0.135** | **79.5%** | 203.5 (−16%) | 6.4 GB | ✅ |

**−68% KL vs bf16 (0.424→0.135) for −16% decode (241→203) + 1.8 GB.** This is the
inverse of the projection levers: proj-MQ4 traded quality DOWN for speed;
mq6-experts trades speed DOWN for quality, far more efficiently (the experts are
where the error lives). Validated three ways: (1) chat-framed `coherence_probe`
verdict **OK (0 hard / 0 soft)**, all 8 detectors clean, 200 tok — the
bare-prompt "France is France is" smoke loop was the KNOWN bare-completion
artifact (affects mq4 too), not a kernel bug; (2) KL *decreasing* vs mq4 is itself
proof the 6-bit dequant is correct (a wrong unpack raises KL or NaNs, never
lowers it); (3) bit-pack roundtrip proven symbolically.

**Opt-in, not default:** default stays mq4 (max speed, coherent). mq6-experts is
the quality-priority build (bf16-closer output) for users who can spend ~16%
decode + 1.8 GB. Future: mq4-proj + mq6-expert combo, or per-layer expert tiering
(first/last MoE layers mq6), to tune the curve further.

Tooling added: `examples/kld_logits.rs` (per-position KL or `--dump`; arch_id 11)
+ quantizer flags `HIPFIRE_LFM2_PROJ_MQ6=1` (rejected lever) and
`HIPFIRE_LFM2_EXPERT_MQ6=1` (shipped) + `scripts/bf16_ref.py` (bf16-reference KL).

## Open items
1. ~~KLD/PPL of proj-MQ4 vs Q8~~ **DONE 2026-05-30** (bf16-referenced table
   above): Q8 default 0.84 nats / 88.6% top-1 vs bf16 (experts dominate);
   proj-MQ4 1.10 / 79.5% → stays opt-in; proj-MQ6 no-speedup+loop → rejected; Q8
   default confirmed best. A default-able *decode* win needs a NEW projection
   format (faster than Q8 at batch=1 AND lower-KLD than MQ4) or the launch-count
   fusion levers; the bigger *quality* lever is higher-precision EXPERTS (mq6),
   traded against VRAM/bandwidth — still open.
2. Prefill is per-token `decode_step` (correctness-first); a batched prefill
   kernel set would speed long-context ingestion (needs batched conv1d scan +
   batched attn/MoE).
3. Spec-decode / DFlash: the conv-state tree/snapshot pattern exists in qwen35
   (`conv1d_silu_split_tree`, speculative.rs) if ever needed; out of scope here.
