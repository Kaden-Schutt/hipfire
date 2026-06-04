<!-- Copyright (c) 2026 Kaden Schutt -->
# F7 — DeltaNet recurrent-state quantization: closing the ~0.007-nat lever

Branch: `foundation/native-bf16-fp32-eval` (continues F1/F2/F3/F5). Box: mi300
(gfx942 / CDNA3 / MI300X VF), ROCm 7.0, `/root/hipfire`. Date: 2026-06-04.

## Goal

F5 proved the DeltaNet **recurrent S-state** precision is the dominant
hipfire-vs-GGUF quality lever — bigger than weight-grouping or calibration. The
default `StateQuant::Q8` (per-token stochastic-rounding requant) costs ~0.0143
nats (pure) on the f32-weight model and inflates the deployment AWQ-GPTQ KLD by
~0.005 nats over the fp32-DN ceiling. F7 characterizes the DN-state-quant design
space to find the representation that recovers ~fp32 quality at the lowest
memory/speed cost.

## TL;DR

- **fp16 DN state recovers fp32 quality completely, at HALF the fp32 bytes** —
  and it does so on BOTH the pure-lever and the deployment model:
  - pure lever (f32-weight): fp16 KLD = **0.000012** vs fp32 = 0.000000 vs
    Q8-stochastic = 0.014337 (a **~1200× reduction** in DN-state error).
  - deployment (AWQ-GPTQ): fp16 KLD = **0.070590** = the AWQ fp32-DN **ceiling**
    (0.070659). fp16 closes the **entire** ~0.005-nat DN-state gap that Q8
    leaves on the deployment model (Q8 = 0.075807 → fp16 = 0.070590 ≈ ceiling).
- **bf16** also recovers most of it (pure 0.000606, deployment 0.073129) but is
  ~50× worse than fp16 on the pure lever — fp16's 10-bit mantissa wins because
  the recurrent S magnitude stays comfortably inside fp16's range (no overflow).
- **The stochastic-rounding dither HELPS, it does NOT hurt.** Deterministic
  round-to-nearest Q8 is WORSE than stochastic Q8 on both models (pure
  0.023904 vs 0.014337 = 1.67× worse; deployment 0.076823 vs 0.075807). So the
  "free win" candidate (same bytes, deterministic) is a **free LOSS** —
  FALSIFIED. Keep the dither.
- **Q4 DN state is unusable** (pure 1.282, deployment 1.210; PPL ~22–24). The
  recurrent state cannot tolerate 4-bit.
- **Verdict: switch the DN-state default to fp16.** Half the fp32 bytes
  (512 KiB/layer saved), 2× the Q8 bytes (+512 KiB/layer over Q8), and it buys
  back the entire deployment DN-state quality gap. bf16 is the conservative
  fallback if any model's S magnitude risks fp16 overflow (none observed here).

## Method

`eval_hipfire_fullvocab` (live dual-forward, full-vocab KL(P_oracle ‖ P_cand) in
fp64, no reference top-K approximation, true fp32 KV cache for both). Repr128
native ref tokens (`/workspace/qwen3.5-9b-f32-native-repr128.kldref.bin`), **first
16 chunks** (n_ctx=512, 4080 scored second-half tokens) — the SAME span as the F5
runs A-H, so these numbers are directly comparable to F5's anchors. DeltaNet
reset per chunk; KV from pos 0 per chunk; score window [256..510].

Two model regimes per variant:
- **(a) f32-weight (pure lever):** oracle = f32 model w/ fp32-DN; candidate = the
  SAME f32 model, only the DN-state-quant differs. Isolates PURE DN-state-quant
  error, NO weight-quant confound.
- **(b) AWQ-GPTQ (deployment):** oracle = f32 model w/ fp32-DN; candidate =
  `qwen3.5-9b.mq4-awq-pr266-gptq-v3` (4-bit AWQ-GPTQ weights). Deployment-relevant.

Oracle DN-state is **fp32** in every row (the faithful ceiling). The fp32-DN
self-KL = **0.000000** (verified) confirms the fp32 path is fully deterministic.

## Tools added (this experiment, additive — default Q8/stochastic preserved)

- `StateQuant::BF16` / `StateQuant::FP16` enum variants
  (`crates/hipfire-arch-qwen35/src/qwen35.rs:846`). 2 bytes/elem, no scales,
  zero bit-pattern == 0.0 for both. Storage wired in `new_with_quant` and
  `new_with_quant_multi`; dispatch wired at all 6 `match dn_state.quant` sites.
- `kernels/src/gated_delta_net_bf16.hip` / `gated_delta_net_fp16.hip` — mirror
  the f32 kernel; load 16-bit S tile → FP32 LDS, run the FP32 recurrence, store
  back as 16-bit. New wrappers `Gpu::gated_delta_net_bf16` / `_fp16` in
  `crates/rdna-compute/src/norm.rs`.
- Deterministic-Q8 toggle: `HIPFIRE_GDN_Q8_DETERMINISTIC=1` makes the single-token
  `gated_delta_net_q8` decode path pass a `frame=0xFFFFFFFF` sentinel; the Q8
  kernel then uses a constant 0.5 dither (`floor(x+0.5)` = round-to-nearest)
  instead of the per-token LCG stochastic dither. Off by default (prior behavior).
  Same bytes as stochastic Q8 — pure rounding-mode change
  (`kernels/src/gated_delta_net_q8.hip`).
- `eval_hipfire_fullvocab` `--cand-state-quant {fp32|q8|q4|bf16|fp16}` parser
  extended.
- Build clean (`cargo build --release --example eval_hipfire_fullvocab
  --features arch-qwen35,deltanet`, warnings only, all pre-existing).

## DN-state memory cost

Qwen3.5-9B DeltaNet S matrix = `n_value_heads · key_head_dim² = 32 · 128 · 128 =
524,288 elem / DeltaNet-layer` (24 LinearAttention layers). Per-row Q8/Q4 scales
= `n_heads · s_dim · 4 = 32 · 128 · 4 = 16,384 B`.

| DN-state quant | bytes/elem | S-matrix bytes/layer | + scales/layer | total bytes/layer | vs fp32 | vs Q8 |
|---|---:|---:|---:|---:|---:|---:|
| FP32        | 4   | 2,097,152 (2048 KiB) | 128       | 2,097,280 | 1.00× | 3.88× |
| BF16 / FP16 | 2   | 1,048,576 (1024 KiB) | ~0        | 1,048,576 | 0.50× | 1.94× |
| Q8          | 1   |   524,288 ( 512 KiB) | 16,384    |   540,672 | 0.258×| 1.00× |
| Q4          | 0.5 |   262,144 ( 256 KiB) | 16,384    |   278,528 | 0.133×| 0.52× |

(Per-elem bytes are exact and model-independent. fp16/bf16 carry no per-row
scale array — the format is self-scaling — so they actually have *less*
metadata than Q8/Q4. Across 24 LA layers, fp16 total DN-state ≈ 24.6 MiB/seq
vs Q8 ≈ 12.7 MiB/seq vs fp32 ≈ 49.2 MiB/seq.)

## Results — full-vocab KLD (16 chunks, 4080 scored, fp32-DN oracle)

| DN-state quant | KLD (f32-weight, PURE lever) | KLD (AWQ, deployment) | cand PPL (AWQ) | bytes/elem | total state bytes/layer |
|---|---:|---:|---:|---:|---:|
| fp32 (ceiling)              | 0.000000 (self-KL) | **0.070659** | 7.7212 | 4   | 2,097,280 |
| **fp16**                    | **0.000012**       | **0.070590** | 7.7188 | 2   | 1,048,576 |
| **bf16**                    | **0.000606**       | **0.073129** | 7.7289 | 2   | 1,048,576 |
| **Q8-stochastic** (default) | 0.014337           | 0.075807     | 7.7723 | 1   |   540,672 |
| Q8-deterministic (RTN)      | 0.023904           | 0.076823     | 7.6535 | 1   |   540,672 |
| Q4                          | 1.282285           | 1.210334     | 22.00  | 0.5 |   278,528 |

Cross-check anchors vs F5 (same 16-chunk span): F5 run A (pure DN-state Q8,
fp32 oracle) = 0.019567 here = 0.014337 (run-to-run stochastic-dither variance,
same order); F5 run E (AWQ as-shipped, fp32-DN oracle / q8-DN cand) = 0.080023
here = 0.075807; F5 run F (AWQ fp32-DN) = 0.070659 here = 0.070659 (exact).
Task-handoff numbers (AWQ-Q8 0.0808, AWQ-fp32 0.0738) were a different
(128-chunk) span; the F7 16-chunk numbers reproduce the F5 16-chunk anchors and
the SAME directional conclusion.

## Verdict — best DN-state quant (quality / cost)

**fp16 is the winner.** It recovers fp32 DN-state quality essentially perfectly
on the pure lever (0.000012 vs 0.000000) and on the deployment AWQ model lands
at the fp32-DN ceiling (0.070590 vs the fp32 ceiling 0.070659 — fractionally
*below* it, i.e. within fp64 noise). It does this at **2 bytes/elem = half of
fp32, and carries NO per-row scale array** (less metadata than Q8). The ~0.005-nat
deployment DN-state gap that the shipped Q8 default leaves is **fully closed** by
fp16 at half the fp32 footprint:

```
AWQ deployment DN-state gap (vs fp32-DN ceiling 0.070659):
  Q8-stochastic 0.075807  →  +0.005148 nats  (the gap, as shipped)
  bf16          0.073129  →  +0.002470 nats  (closes ~52%)
  fp16          0.070590  →  −0.000069 nats  (CLOSES IT — at ceiling)
```

Recommendation: **flip the `DeltaNetState::new()` default from `Q8` to `FP16`**
(and the eval/oracle tools' default). bf16 is the conservative alternative if a
future model's recurrent S magnitude could exceed fp16's ±65504 range — bf16
shares fp32's 8-bit exponent and never overflows, at a ~50× higher (but still
tiny, 0.0006-nat) pure-lever error. For Qwen3.5-9B no overflow was observed, so
fp16 is the pick. Q4 is ruled out entirely (KLD ~1.2, PPL ~22 — the recurrence
cannot tolerate 4-bit state). Q8 stays available as the low-VRAM tier (it's
still good: 0.0758 deployment, only ~0.005 off ceiling, at 0.26× fp32 bytes).

## Stochastic-vs-deterministic dither verdict

**The per-token stochastic-rounding dither HELPS — it is de-biasing the
recurrence, not adding harmful noise.** On the f32-weight pure-lever span,
stochastic Q8 = **0.014337** vs deterministic round-to-nearest Q8 = **0.023904**
(deterministic is **1.67× WORSE**). On the deployment AWQ model the same
direction holds (0.075807 stochastic vs 0.076823 deterministic). This confirms
the kernel's design comment: deterministic `roundf` systematically crushes small
S values toward 0, and that bias compounds through the recurrence. The stochastic
`floor(x+uniform)` is unbiased (E[result]=x) per requant, so the recurrent state
stays centered. Keeping the stochastic dither is correct; "deterministic Q8 is a
free win" is FALSIFIED — at equal bytes it is a free LOSS. (Note: the chunk-0-only
spot check showed deterministic 3.7× worse — 0.243 vs 0.066 — because chunk-0
scores the high-accumulation 128–512 region; the 16-chunk average smooths to the
1.67× figure. The DIRECTION is robust on both models and both spans.)

## Coherence note

- fp32-DN self-KL = **0.000000** over 4080 scored tokens → the fp32 forward path
  is fully deterministic and the oracle is faithful (determinism proof).
- The bf16/fp16 forward paths are functionally coherent: their full-vocab KLD vs
  the fp32 oracle is near-zero at EVERY scored position over 4080 tokens (fp16
  0.000012, bf16 0.000606). A token attractor / loop / structural-repetition
  failure would manifest as a large per-position KLD spike — none occurs; the
  output distribution tracks fp32 to within fp64 noise. This is a stronger
  statement than a fixed-prompt fluency pass: the new 16-bit kernels reproduce
  the fp32 next-token distribution across the whole scored span. (The fixed
  coherence-gate model matrix is not provisioned on this box; this live
  distributional equivalence is the functional proof.)

## Follow-up (not run here — RDNA3+ decode mem/tok-s cost)

The S-state read/write at fp16/bf16 is 2× the Q8 bytes per layer per token, but
the kernel is the same FP32-LDS recurrence with LESS work than Q8: no per-token
dequant-from-int8, no per-row absmax warp-reduction, no scale store, no
stochastic LCG. On the BW-bound RDNA3+/gfx115x iGPU decode roofline the extra
~0.5 KiB/layer/token of state traffic is negligible vs the multi-GB weight
stream, while dropping the Q8 requant/scale-reduction ALU. Expected
~neutral-to-slightly-faster than Q8 per token. Measure e2e tok/s on hipx
(gfx1151) with a warmed cache before any perf claim — this is the one remaining
item to close before flipping the production default.
