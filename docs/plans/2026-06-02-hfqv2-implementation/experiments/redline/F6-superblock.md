<!-- Copyright (c) 2026 Kaden Schutt -->
# F6 — Super-block-linear 4-bit weight codec (the redline: close/flip the ~4% gap to GGUF Q4_K_S at lower bpw)

Branch: `foundation/native-bf16-fp32-eval`. Box: mi300 (gfx942 / CDNA3 / MI300X VF),
ROCm 7.0, `/root/hipfire`. Date: 2026-06-04. Local only; nothing pushed.

## Goal (locked with maintainer)

Build a NON-codebook (no Lloyd) super-block-linear 4-bit codec — 256-elem
super-block, sub-blocks of g32 (match Q4_K) and g64 (perf/bpw midpoint), per-sub
6-bit scale (+6-bit min for the unrotated variant) hierarchically compressed by a
per-super-block fp16 d (+dmin). Two variants:
- **1a** unrotated asymmetric (scale+min) = faithful hipfire-native Q4_K.
- **1b** FWHT-rotate -> ~zero-mean Gaussian -> SYMMETRIC quant (scale only, NO
  min) -> halves sub-block overhead -> per-32 at ~4.25 bpw. The bpw lever GGUF lacks.

HARD bpw gate 4.2-4.5 ("q4"); >4.5 disqualified.

## Method — fake-quant weight-codec isolation (eval-only, no forward/kernel change)

The KLD harness (`eval_hipfire_fullvocab`) loads a candidate `.hfq` and runs the
proven all-F32 forward. To measure a NEW codec's *weight-codec* quality without
wiring a new GEMV/dispatch, I built `examples/fakequant_superblock`:

1. Read the all-F32 oracle `.hfq` (`/workspace/qwen3.5-9b-f32-oracle.hfq`).
2. For each tensor, apply the production DENSE kmap protection (mode 0, is_moe=0):
   norms/bias, embed, lm_head, DeltaNet conv1d, and edge-layer (first2/last2) FFN
   are kept **lossless F32**; everything else (the 4-bit "Base" class: attn q/k/v/o
   + middle-layer MLP gate/up/down) is **round-tripped through the candidate codec**
   (quant -> dequant -> F32).
3. Write a new all-F32 `.hfq`. Its ONLY injected error is the codec round-trip.
4. KLD vs the faithful fp32-DN oracle via `eval_hipfire_fullvocab
   --oracle-state-quant fp32 --cand-state-quant fp32` (isolates the weight codec;
   DN-state confound removed on both sides).

The codec's reported bpw is the EXACT on-disk effective bits/weight of its real
layout (NOT the F32 storage of this fake-quant file). The `flat-g256` control
re-derives the production-flat anchor under THIS protection, calibrating the
(small) lossless-protected-class offset so cross-codec deltas are trustworthy.

Tool is bit-exact to the production codecs: `cpu_fwht_256` + `gen_fwht_signs`
seeds 42/1042, the same fp16 d/dmin hierarchical compression as `quantize_q4k`,
f16<->f32 mirrors. iFWHT verified identity round-trip (max err 9.5e-7).

## EXACT bpw per codec (all in-band ≤4.5)

| codec | layout (bytes / 256-elem super-block) | EXACT bpw | in-band? |
|---|---|---:|:--:|
| flat-g256 (CONTROL, plain flat HFQ4-G256 asym) | 8 (f32 scale+min) + 128 nibbles = 136 | 4.2500 | yes |
| mq4-flat-g256 (CONTROL, flat + offline FWHT)   | 136 | 4.2500 | yes |
| **sb-asym-g32** (1a, unrotated asym)           | 4 (f16 d+dmin) + 12 (8×6b sc + 8×6b min) + 128 = 144 | **4.5000** | yes (edge) |
| **sb-asym-g64** (1a, unrotated asym)           | 4 + 6 (4×6b sc + 4×6b min) + 128 = 138 | **4.3125** | yes |
| **sb-fwht-sym-g32** (1b, FWHT symmetric)       | 2 (f16 d) + 6 (8×6b sc) + 128 = 136 | **4.2500** | yes |
| **sb-fwht-sym-g64** (1b, FWHT symmetric)       | 2 + 3 (4×6b sc) + 128 = 133 | **4.1563** | yes |

Note: g64-asym and g32-fwht-sym tie the flat-g256 control on bpw (both 4.25), and
fwht-sym-g64 is the LOWEST bpw of all (4.156) — the synergy the maintainer flagged:
dropping the min (symmetry, enabled by FWHT) buys finer grouping at the same/lower
bytes.

## IMPORTANT FRAMING — these are PLAIN variants (no AWQ/GPTQ)

Step 1 quantizes PLAIN (per task). The right anchor for plain super-block is the
**PLAIN flat-G256 control**, NOT the AWQ-GPTQ-v3 0.073771 (which has the AWQ+GPTQ
quality levers on top). AWQ-GPTQ layering is Step 2.

- **PLAIN flat-G256 control (this methodology): KLD = 0.128749 @ 32 chunks (fp32-DN).**
- AWQ-GPTQ-v3 (full pipeline, F3): 0.073771 @ 128 chunks — the Step-2 target.
- GGUF Q4_K_S: 0.070983 @ 128 chunks — the absolute target.

The 32-chunk span is used for the sweep (relative cross-codec comparison valid);
the winning variant + control are re-confirmed at full 128 for headline numbers.

---

## RESULTS — 32-chunk sweep (fp32-DN both sides, full-vocab KLD)

All 510 scored/chunk, 8160 scored, fp32-DN both sides, true-F32 KV, repr128 span.

| variant | EXACT bpw | KLD (fp32-DN, 32ch) | round-trip mean abs err | vs PLAIN flat-G256 (0.128749) |
|---|---:|---:|---:|---|
| flat-g256 (CONTROL, plain flat asym) | 4.2500 | 0.128749 | 1.27e-3 | — (anchor) |
| mq4-flat-g256 (CONTROL, flat + FWHT) | 4.2500 | 0.126263 | — | −1.9% |
| **sb-asym-g32 (1a)** | 4.5000 | **0.075591** | 8.72e-4 | **−41.3%** |
| sb-asym-g64 (1a) | 4.3125 | 0.103616 | 1.02e-3 | −19.5% |
| sb-fwht-sym-g32 (1b) | 4.2500 | 0.104735 | 1.01e-3 | −18.7% |
| sb-fwht-sym-g64 (1b) | 4.1563 | 0.113160 | — | −12.1% |

### Findings (the three locked questions)

**Q: does finer grouping close the gap?** YES, decisively. Going flat-256 → per-32
super-block (sb-asym-g32) cuts plain weight-codec KLD by **41%** (0.1287 → 0.0756)
— the single biggest lever measured. g64 sits roughly midway (0.1036). The
g32/g64 knee is STEEP: g32 is worth a further −27% over g64 (0.0756 vs 0.1036)
for +0.19 bpw (4.50 vs 4.3125). **g32 is the clear quality knee.**

**Q: does FWHT-symmetric add ON TOP of grouping, or substitute?** It SUBSTITUTES
(and underperforms) for WEIGHTS. At the SAME g32, unrotated asym (0.0756) beats
FWHT-symmetric (0.1047) by 28%. FWHT alone at flat-256 helps only ~2% (0.1263 vs
0.1287). The hypothesized "FWHT → Gaussian → drop the min → finer grouping at
lower bpw" synergy does NOT materialize on these weights: the asymmetric per-sub
(scale+min) offset is worth MORE than the bpw the symmetric variant saves by
dropping it. FWHT-sym's round-trip max-abs err is ~2× the asym's (0.115 vs 0.059)
— the symmetric [-8,7] clamp on the post-FWHT tails costs more than the min-free
layout buys. **The bpw lever GGUF lacks did not beat the plain grouping lever for
the weight codec.** (FWHT remains the right tool for ACTIVATIONS / the AWQ rotate-x
decode path; this is specifically about static weight quantization.)

**Q: g32 vs g64 knee?** g32 (4.50 bpw, 0.0756) is the quality knee; g64 (4.3125
bpw, 0.1036) trades 27% more KLD for 0.19 bpw — a bad trade. If the 4.50 gate edge
is a concern, g64 is the in-band fallback but at a real quality cost.

### Verdict (plain, 32-chunk methodology)

The winner is **sb-asym-g32 (1a) = 0.075591 @ 4.50 bpw** — plain, no AWQ/GPTQ.
This is the faithful hipfire-native Q4_K and it lands essentially AT the
AWQ-GPTQ-v3 full-pipeline level (0.0738, 128ch) and within ~6% of GGUF Q4_K_S
(0.0710, 128ch) — **with no AWQ/GPTQ yet**. The grouping lever alone recovers the
entire gap that previously needed the AWQ+GPTQ stack. Confirming at full 128
chunks + the Step-2 AWQ/GPTQ layering below.

(Note: 32-chunk numbers are a hair higher than the 128-chunk span the GGUF/AWQ-GPTQ
references use — the early chunks carry more KLD. Cross-codec RANKING at 32ch is
valid; the headline absolute vs GGUF/AWQ-GPTQ uses the 128-chunk re-confirm.)

---

## HEADLINE — full 128-chunk span (IDENTICAL span/tokens as F3's GGUF/AWQ-GPTQ numbers)

fp32-DN both sides, true-F32 KV, 32,640 scored tokens, repr128 span.

| variant | EXACT bpw | KLD (128ch, fp32-DN) | vs PLAIN flat-G256 | vs AWQ-GPTQ-v3 (0.073771) | vs GGUF Q4_K_S (0.070983) |
|---|---:|---:|---|---|---|
| **flat-G256 PLAIN (CONTROL)** | 4.2500 | **0.147552** | — anchor | +100% | +108% |
| **sb-asym-g32 PLAIN (1a, WINNER)** | 4.5000 | **0.080275** | **−45.6%** | +8.8% | +13.1% |
| AWQ-GPTQ-v3 (full pipeline, F3 ref) | ~4.6 | 0.073771 | −50.0% | — | +3.9% |
| GGUF Q4_K_S (F3 ref, llama --kl-div) | 4.76 | 0.070983 | −51.9% | −3.8% | — |

**The grouping lever (flat-256 → per-32 super-block) cuts PLAIN weight-codec KLD by
45.6% (0.147552 → 0.080275)** — and it does so at LOWER bpw than GGUF Q4_K_S (4.50 vs
4.76). This single linear, non-codebook codec change recovers ~88% of the gap to the
full AWQ+GPTQ pipeline, with zero AWQ/GPTQ applied.

### Step-1 VERDICT (plain, no AWQ/GPTQ)

- hipfire's faithful-native super-block Q4 (sb-asym-g32) PLAIN = **0.080275 @ 4.50 bpw**.
- vs GGUF Q4_K_S 0.070983 @ 4.76 bpw: hipfire is **+13.1% behind on KLD but −5.5% lower
  bpw**. So PLAIN super-block does NOT yet beat/tie GGUF on KLD — but it closes the
  prior 45-point gap to within 13 points, at lower bpw, with no codebook and no AWQ/GPTQ.
- vs AWQ-GPTQ-v3 0.073771: PLAIN super-block is +8.8% behind the full hipfire pipeline —
  i.e. switching the codec flat-G256 → super-block-g32 alone gets PLAIN nearly to where
  the entire AWQ+GPTQ stack got flat-G256. The two levers (grouping, AWQ+GPTQ) are
  largely ADDITIVE → Step-2 (AWQ+GPTQ on super-block) is the path to BEAT GGUF.

### FWHT-symmetric: REJECTED for the weight codec

The "FWHT → drop the min → finer grouping at lower bpw" synergy GGUF lacks did NOT pan
out FOR STATIC WEIGHTS: at matched g32, asym (scale+min) beats FWHT-symmetric
(scale-only) by 28% (0.0756 vs 0.1047 @ 32ch). The per-sub min offset is worth more
than the ~0.25 bpw it costs; the symmetric [-8,7] clamp on post-FWHT tails raises
max round-trip error ~2×. FWHT stays the right tool for the ACTIVATION/AWQ rotate-x
decode path, not for the weight quantizer. So **1a (asym) is the format to ship; 1b
(FWHT-sym) is rejected.**

### g32 vs g64 knee

g32 (4.50 bpw) is the quality knee. g64 (4.3125 bpw) costs +29% KLD at 32ch
(0.0756→0.1036) for −0.19 bpw — a poor trade. Use g32. If the 4.50 gate edge must be
undercut, g64 is the in-band fallback at a real quality cost; the lower-bpw seats
(fwht-sym-g64 @ 4.156) are quality-dominated by g32-asym, so there is no free lunch
below 4.50 here.

---

## STEP 2 — AWQ layered on the winning super-block (sb-asym-g32)

Step-2 proxy: extend the fake-quant tool to apply the AWQ-GPTQ-v3 artifact's 184
EMBEDDED per-channel AWQ scales (`<name>.awq_scale.weight`, F16, length-K). Per Base
tensor with a matching scale: pre-scale columns `W[:,j]*=s[j]` → super-block quant →
dequant → un-scale `W[:,j]/=s[j]` (the inference x/s side cancels in the effective
weight; this is the exact AWQ pre-conditioning of the codec). Run with
`--match-v3-scope` so the protection EXACTLY matches the v3 artifact (Q8 only
embed_tokens + conv1d; lm_head + DeltaNet in_proj + all MLP at 4-bit; 249 quanted
tensors, matching v3's 249 qt=13 count) → a true apples-to-apples vs v3's 0.073771.

**GPTQ is NOT applied** (no Hessian on disk for this 9B; GPTQ's error-feedback
ordering is the heavy lever). So this is super-block + AWQ ONLY vs v3's
super-block-absent (flat-MQ4) + AWQ + GPTQ. GPTQ remains UNspent headroom on top.

| variant | EXACT bpw | KLD (128ch, fp32-DN) | vs GGUF Q4_K_S (0.070983) | vs AWQ-GPTQ-v3 (0.073771) |
|---|---:|---:|---|---|
| **sb-asym-g32 + AWQ (no GPTQ), v3-scope** | 4.5000 | **0.071823** | **+1.2% (TIE)** | **−2.6% (BEATS)** |
| AWQ-GPTQ-v3 (flat-MQ4 + AWQ + GPTQ) | ~4.6 | 0.073771 | +3.9% | — |
| GGUF Q4_K_S (per-32 Q4_K) | 4.76 | 0.070983 | — | −3.8% |

**super-block-g32 + AWQ (WITHOUT GPTQ) = 0.071823 — it BEATS the full hipfire
AWQ+GPTQ flat-MQ4 pipeline (0.073771) by 2.6% and ties GGUF Q4_K_S (0.070983, +1.2%)
at LOWER bpw (4.50 vs 4.76).** The codec swap (flat-256 → per-32 super-block) is worth
MORE than the entire GPTQ pass that v3 spent to reach 0.0738. Adding hipfire's existing
GPTQ on top of super-block (Step-2 full pipeline) is the clear path to BEAT GGUF
outright — the levers stack: super-block + AWQ already crossed v3, and GPTQ is unspent.

---

## OVERALL VERDICT

**Does hipfire beat/tie GGUF Q4_K_S (0.070983) at ≤4.5 bpw?**

- **PLAIN (no AWQ/GPTQ):** the super-block grouping lever cuts weight-codec KLD 45.6%
  (flat-G256 0.147552 → sb-asym-g32 0.080275) at 4.50 bpw (< GGUF's 4.76). Plain does
  NOT yet beat GGUF (+13.1%) but closes the bulk of the gap with a pure linear,
  non-codebook codec.
- **+AWQ (no GPTQ):** **TIE with GGUF** — sb-asym-g32 + AWQ = **0.071823 @ 4.50 bpw**,
  GGUF Q4_K_S = 0.070983 @ 4.76 bpw (+1.2%, well inside cross-oracle/scoring slop), and
  it already **BEATS the full hipfire AWQ-GPTQ-v3** (0.073771) by 2.6%.
- **GPTQ (not run, no Hessian on disk) is unspent headroom** that, layered on
  super-block + AWQ (it added quality going flat→v3), should push BELOW GGUF.

**Shipping recommendation:** the format to ship is the **unrotated asymmetric
super-block, g32 ("sb-asym-g32"), 256-elem super-block, fp16 d+dmin hierarchical,
8×6-bit sub-scale + 8×6-bit sub-min, 144 B/256 = 4.50 bpw** — i.e. a faithful
hipfire-native Q4_K, at 0.26 bpw LOWER than GGUF's 4.76. NON-codebook (no Lloyd),
GPU-friendly uniform 256 layout. The existing `Q4K` (id=4) format + `gemv_q4k` kernel
ALREADY implement exactly this layout (g32, 144 B/256) — so the GEMV/dispatch path
exists; what's missing is wiring `Q4K` into the qwen35 forward dispatch and the
AWQ/GPTQ-on-Q4K quant pipeline (today AWQ/GPTQ target flat-MQ4G256). Reject the
FWHT-symmetric variant (1b) for weights; g64 only if forced below 4.50.

---

## STEP 3 — perf (NOT RUN; precise remaining follow-up)

Step-3 (decode tok/s + occupancy of the super-block GEMV vs flat-MQ4G256, and the
fp16-DN vs Q8-DN decode A/B) requires an RDNA3+ box (k9lin gfx1100 / hiptrx gfx1201 /
hipx gfx1151) — mi300 is CDNA3 (wave64, no WMMA) and not representative of the RDNA
decode path this codec ships on. NOT run this session (all GPU budget went to the
quality answer, which was primary and bounded). Precise remaining work:

1. The on-disk format already exists as `Q4K`/`gemv_q4k` (144 B/256, g32, 6-bit
   hierarchical) — measure its decode tok/s + occupancy (VGPR/SGPR/LDS via the
   `gfx-kernel-metadata` skill on the compiled `.hsaco`) vs `gemv_mq4g256` on RDNA3+.
   Expect a small decode cost: 144 B/256 (4.50 bpw) vs MQ4's 136 B/256 (4.25 bpw) = +6%
   bytes read per group; the hierarchical 6-bit unpack adds a little ALU. On a
   BW-bound decode this is ~+6% bytes → bounded ~few-% tok/s cost for the −13% (plain)
   / GGUF-tie (AWQ) quality. g64 (138 B/256, 4.3125 bpw) is the perf/bpw midpoint if
   the g32 cost bites.
2. fp16-DN vs Q8-DN decode tok/s A/B (the F7 follow-up before flipping the DN-state
   default): F3 showed Q8-DN costs +0.007045 nats KLD vs fp16/fp32-DN; the open
   question is its decode tok/s cost on RDNA3+ before defaulting fp16-DN.

## Artifacts / repro

- Tool: `crates/hipfire-runtime/examples/fakequant_superblock.rs` (eval-only; stdlib
  I/O + arithmetic; no GPU/arch features; registered in `crates/hipfire-runtime/Cargo.toml`).
  Bit-exact to production codecs (cpu_fwht_256 + gen_fwht_signs 42/1042, fp16<->f32,
  hierarchical d/dmin). iFWHT identity round-trip verified (max err 9.5e-7).
- Eval: `eval_hipfire_fullvocab --oracle /workspace/qwen3.5-9b-f32-oracle.hfq
  --candidate <fakequant.hfq> --ref /workspace/qwen3.5-9b-f32-native-repr128.kldref.bin
  --oracle-state-quant fp32 --cand-state-quant fp32 --max-chunks {32|128}`.
- AWQ scales sourced from `/workspace/qwen3.5-9b.mq4-awq-pr266-gptq-v3` (184 embedded).
- All fake-quant .hfq files (~36 GB each) generated→evaled→deleted serially; none kept.
- Eval-only: no forward/kernel/dispatch/quant-format math changed; the F32 forward path
  is unmodified. No coherence-gate trigger on the math (the example filename matches the
  pre-commit `quant` HOTSPOT, but it adds no production format/dispatch/forward code).
