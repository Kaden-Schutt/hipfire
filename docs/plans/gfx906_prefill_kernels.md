# gfx906 prefill kernel gap analysis — RDNA2 features worth porting

**Status:** Open. Exploratory plan; nothing scheduled yet.
**Hardware:** AMD Instinct MI50 (gfx906, Vega 20, CDNA1-adjacent wave64 + dp4a).
**Created:** 2026-05-23.
**See also:**
- `docs/plans/gfx906-mmq-prd.md` — closed predecessor PRD that landed
  the 95%-of-llama.cpp gfx906 MMQ work in 2026-05.
- Upstream PR #298 (HFQ3 MMQ on gfx1030), PR #299 (HFQ4 follow-up,
  closed), PR #315 (HFQ4 MMQ on gfx1030 + HFQ3 polish).
- `experiments/hetero-gfx906/` (this branch) — empirical hetero PFlash
  results that prompted this analysis.

## 1. Motivation

The 2026-05-23 hetero PFlash exploration on a gfx906 (target) + gfx1031
(drafter) pair surfaced a structural asymmetry: **the same 0.8B MQ4
drafter prefill runs ~4× faster on gfx1031 than on gfx906** (3.45 s
solo on gfx906 vs 870 ms hetero on gfx1031 for an 11k-token prompt).
The gap is large enough that "put the drafter on the secondary card"
beats "keep everything on the primary" even when the secondary is the
weaker chip on paper (12 GB RDNA2 vs 32 GB CDNA1).

The proximate cause is **PR #315's HFQ4 MMQ family for gfx1030/1031** —
fused-projection wave32 MMQ kernels that gfx906 has not received yet.
gfx906 has the building blocks (`gemm_hfq4g256_mmq_set_gfx906`, the
2026-05 redesign of the residual MMQ-set path) and the lower-batch
fused-dp4a path (`gemm_qkv_hfq4g256_wave64_dp4a` and siblings), but
the **fused-projection MMQ** layer for prefill-batch sizes is missing.

This plan inventories the gap, ranks the kernels by expected impact,
and lists what's worth exploring vs. what to skip.

## 2. Current gfx906 prefill dispatch landscape

For HFQ4G256 (the shipping MQ4 quant), B>1 prefill on gfx906 routes
through `dispatch.rs`. The decision tree is, in order:

1. **Above MMQ cutover** (`should_use_mmq` returns true; gfx906 default
   is batch_size ≥ 8): three separate single-output MMQ-set kernels.
   - QKV → `gemm_hfq4g256_mmq_set_gfx906(a_q, ...); ... (a_k, ...); ... (a_v, ...)`
   - QKVZA → same, 4 calls (qkv + z + β + α)
   - gate_up → same, 2 calls (gate + up)
   - residual `wo` → single `gemm_hfq4g256_residual_mmq_gfx906(...)`
2. **Below MMQ cutover but `gemv_dp4a_enabled`** (typically batch_size
   in [2, 7]): fused-projection dp4a-wave64 kernels.
   - QKV → `gemm_qkv_hfq4g256_wave64_dp4a` (1 launch)
   - QKVZA → `gemm_qkvza_hfq4g256_wave64_dp4a` (1 launch)
   - gate_up → `gemm_gate_up_hfq4g256_wave64_dp4a` (1 launch)
3. **Capture mode or screen-rejected**: fp16 wave64 fallback.

The gap is at level (1): the **prefill regime where MMQ dominates
gfx906's compute**, the path is structurally split into N separate
LDS-tiled kernel launches that each re-stage the X activation tile.

By contrast, gfx1030/1031 after PR #315 has:
- `gemm_qkv_hfq4g256_mmq` — 3-way fused, single launch, single X tile load
- `gemm_qkvza_hfq4g256_mmq` — 4-way fused with the split-routing
  fallback when β/α aren't MMQ_Y-aligned
- `gemm_gate_up_hfq4g256_mmq` — 2-way fused

Each of these reads the Q8_1-quantized X tile **once** and broadcasts
across the fused outputs, instead of re-loading the (memory-bandwidth-
dominant on gfx906) X tile per output.

## 3. What gfx906 already has

Useful to enumerate explicitly to avoid duplicating work:

| Surface | gfx906 status | gfx1030 status |
|---|---|---|
| MMQ residual (wo path) | `gemm_hfq4g256_residual_mmq_gfx906_x{8..64}` (mmq_x sweep, dp4a) | `gemm_hfq4g256_residual_mmq_x16/x32/x32_y64` (PR #315) |
| MMQ-set single output | `gemm_hfq4g256_mmq_set_gfx906` | `gemm_hfq4g256_mmq_set_prequant` |
| Fused QKV (low-batch) | `gemm_qkv_hfq4g256_wave64_dp4a` (B>1, below MMQ cutover) | not needed — MMQ handles it |
| Fused QKVZA (low-batch) | `gemm_qkvza_hfq4g256_wave64_dp4a` | same |
| Fused gate_up (low-batch) | `gemm_gate_up_hfq4g256_wave64_dp4a` | same |
| FP16 fallback (screen-reject) | `gemm_qkv_hfq4g256_fp16_wave64`, siblings | shared with other RDNA |
| MQ3 / HFQ3 prefill | none (gfx906 is HFQ4-only on the MQ path) | PR #298 + #315 cover MQ3 batched |
| FP16 shadow (rocBLAS) | `rocblas_gemm_hfq4_prefill` (CDNA3 only — gfx906 not eligible) | gfx906 not eligible |

So gfx906 has dp4a fused projections at low batch and separate-output
MMQ at high batch. **The missing tier is fused-projection MMQ at high
batch.**

## 4. What's missing — ranked by expected impact

### 4.1 Tier A — fused-projection MMQ for HFQ4 (high-impact)

The direct port of PR #315's value. Three new wave64 dp4a kernels:

- **`gemm_qkv_hfq4g256_mmq_gfx906`** — 3-way fused MMQ. One LDS X
  tile, three weight streams (Q, K, V), three accumulators, three
  outputs. Saves 2× redundant X-tile reads + Q8_1 row-sum reuse vs.
  the current 3× separate `gemm_hfq4g256_mmq_set_gfx906` calls.
- **`gemm_qkvza_hfq4g256_mmq_gfx906`** — 4-way fused. Same idea
  for LinearAttention's qkv + z + β + α. Mirrors PR #315 Phase 4's
  split-routing: when β/α aren't MMQ_Y-aligned, route them through
  2-way `gate_up_mmq` (qkv+z) + 2-way dp4a (β+α). All Qwen3.5 family
  weights are MMQ_Y(128)-aligned on q/k/v/z, so the split path will be
  the common case.
- **`gemm_gate_up_hfq4g256_mmq_gfx906`** — 2-way fused for FFN.

**Why this should win on gfx906:** the existing
`gemm_hfq4g256_residual_mmq_gfx906_body.cuh` (the redesigned residual
body) is the proof point — it ships a 128×mmq_x tile with sdot4 inner
loop and reaches ~95% of llama.cpp parity at the residual call. The
same X-tile-reuse insight applies even more strongly when 3 or 4
outputs share the same X — the LDS X stage amortizes across more work.

**Expected magnitude:** PR #315 reported `+22% prefill on Qwen3.5 LA
layers` from the qkvza split-routing change alone on gfx1031. gfx906's
equivalent absolute win depends on what fraction of current QKVZA
walltime is dominated by X-tile re-staging vs. compute. A rocprof
attribution pass on the post-2026-05 separate-output MMQ-set path
would set the upper bound — but as a back-of-envelope:
- QKVZA today on gfx906 batched prefill = 4× `mmq_set_gfx906` calls
- Each call re-loads the same X tile from HBM (or L2 at best)
- Fusing eliminates 3 of 4 X-loads on the **memory-bandwidth-dominant**
  axis. If X-load is ~25% of mmq_set wall (typical for LDS-tiled MMQ
  on bandwidth-bound chips), fusing 4-way saves ~18% of QKVZA wall.

### 4.2 Tier A — MMQ_Y=64 occupancy probe for gfx906

PR #315 (`MMQ_Y=64 variant` finding) showed +5-14% on RDNA2 by cutting
per-WG LDS budget 26→15 KB and doubling theoretical CU occupancy
(2 WG/CU → 4 WG/CU). **gfx906 has a different occupancy ceiling**
(LDS per CU = 64 KB and wave64 vs. RDNA2's 64 KB + wave32), so the
exact MMQ_Y sweet spot may differ.

Worth probing:
- MMQ_Y = 128 (current redesign default)
- MMQ_Y = 64 (RDNA2 sweet spot)
- MMQ_Y = 96 (gfx906 LDS leaves room for a non-power-of-2)

The gfx906 body at `gemm_hfq4g256_residual_mmq_gfx906_body.cuh:75-76`
hardcodes MMQ_Y=128. Adding a parameterized variant (mirroring PR #315's
`#define MMQ_Y N` pattern in `gemm_hfq3g256_residual_mmq_body.cuh`) is
the minimum-risk first step.

**Important caveat from PR #315 work:** the `tid < 128` → `tid < MMQ_Y`
fix in `gemm_hfq{3,4}g256_residual_mmq_body.cuh` (LDS OOB at MMQ_Y=64)
applies symmetrically to any gfx906 port. Audit the gfx906 body for the
same pattern before adding MMQ_Y variants.

### 4.3 Tier B — HFQ3 batched prefill on gfx906

gfx906 currently has **no HFQ3 batched prefill at all** (the prefill
fast-path for MQ3 dispatch on gfx906 falls through to the FP16 wave64
path because the MMQ family is HFQ4-only on gfx906). MQ3-quantized
9B / 4B models exist (`qwen3.5-4b.mq3-lloyd`, `qwen3.5-9b.mq3`) and
are useful for VRAM-constrained deployments.

PR #298 + #315 cover MQ3 on gfx1030 via:
- `gemm_qkv_hfq3g256_mmq_x{8,16,32}.gfx1030.hip`
- `gemm_qkvza_hfq3g256_mmq_x{8,16,32}.gfx1030.hip`
- `gemm_gate_up_hfq3g256_mmq_x{8,16,32,32_y64,32_y96}.gfx1030.hip`
- `gemm_hfq3g256_residual_mmq_x{8,16,32,32_y32,32_y64}.gfx1030.hip`

A gfx906 port would essentially be:
- Copy the gfx906 HFQ4 redesign topology (128×mmq_x, wave64, dp4a) to
  HFQ3 (the only differences vs. HFQ4 are the per-group stride 104 vs.
  136 B and the 3-bit-trit unpack vs. 4-bit nibble).
- Reuse the existing per-mmq_x kernel sweep convention.

**Lower priority than 4.1** because MQ3 is a niche quant on this hardware
(MQ4 is the shipping default and MQ3's KLD penalty is significant), and
because the fused-MMQ work at 4.1 is the larger structural win that
applies to *every* MQ4 prefill on gfx906.

### 4.4 Tier B — Q8_1 X scratch reuse across MMQ + fused-dp4a paths

The dispatcher already does this — `ensure_q8_1_mmq_x(x, batch_size, k)`
is called once per layer, and the result is passed to all MMQ-set
calls. But the fused-dp4a fallback path (`gemm_qkv_hfq4g256_wave64_dp4a`,
called below the MMQ cutover or in capture mode) **re-quantizes X
internally** from the fp32 activation — it doesn't accept a pre-quantized
input.

Adding a `_prequant` variant for each of:
- `gemm_qkv_hfq4g256_wave64_dp4a_prequant`
- `gemm_qkvza_hfq4g256_wave64_dp4a_prequant` (one already exists at
  dispatch.rs:5712 — confirm coverage)
- `gemm_gate_up_hfq4g256_wave64_dp4a_prequant`

would let the dispatcher reuse the Q8_1 scratch across screen-reject
fallback paths, saving one Q8_1 quant pass per layer. Small win on
its own, but composes with 4.1.

### 4.5 Tier C — investigate `mmq_screen_weight` thresholds on gfx906

The dispatcher's `mmq_screen_weight` rejects MMQ for outlier-heavy
weights (the screen was tuned for fp16-vs-dp4a fallback quality). With
fused-projection MMQ in tier 4.1, the cost of falling through to fp16
goes up (because the screen-reject path can't yet fuse 3-way / 4-way
MMQ, only the legacy fp16 wave64). It's worth re-measuring the screen
rejection rate on Qwen3.5 AWQ-quantized weights post-4.1: AWQ
specifically reduces the outlier surface area that `mmq_screen_weight`
keys on, so the screen may reject 0% of layers on AWQ models and the
fused-fp16 fallback can be deprioritized.

### 4.6 Tier C — rocBLAS HFQ4 FP16 shadow eligibility on gfx906

gfx906 is **not** in `rocblas_arch_eligible()` (CDNA3 / gfx940-942 only,
per dispatch.rs:6433-6437 and 7180-7185). Reason: rocBLAS on gfx906
through ROCm 6.4.3 has known performance regressions for HFQ-shaped
GEMMs, and the fp16-shadow path doubles VRAM for weights. **Probably
not worth revisiting** unless rocBLAS gfx906 perf improves substantially
in a future ROCm release — listed here only so future readers don't
re-investigate from scratch.

### 4.7 Tier D — DeltaNet recurrence kernels

The recurrence (`gated_delta_net_q8_batch_seq`) is **per-token sequential
by construction** — the chunk loop in `forward_prefill_chunk` calls it
N times per chunk regardless of batched-projection optimizations.
Batched-MMQ work at 4.1 will NOT help DeltaNet wall time.

Potential lever: chunk size. The current `PREFILL_MAX_BATCH` (256) is
tuned for the projection kernels' tile geometry, not the recurrence.
A gfx906-specific micro-tuning of chunk size MIGHT amortize recurrence
fixed costs better at very long prompts, but the effect should be
small. **Skip unless rocprof shows recurrence as the new dominant
bucket after 4.1 lands.**

### 4.8 Tier D — splitK / Stream-K for residual `wo`

The `wo` GEMM after attention has shape `(M=dim, K=v_dim, N=batch)`
where dim ≈ 4096 on 9B. Stream-K-style atomic-add reduction has been
known to help on similar shapes on CDNA — but the existing residual
redesign already reaches 95% of llama.cpp on gfx906, so the absolute
upside is small. **Defer until after 4.1 / 4.2.**

## 5. Out of scope

- **MFP4G32**: per `project_mfp4_gfx906_no_accel` memory, gfx906 has no
  accelerated MFP4G32 kernel and falls through to a wave32-oriented
  generic path. Optimizing MFP4G32 on gfx906 is a much larger effort
  (new format, no shipped kernel families) and is orthogonal to closing
  the MQ4-prefill gap. Not addressed here.
- **WMMA**: gfx906 has no WMMA instructions (RDNA3+ feature). The
  WMMA branches in the dispatcher (`has_wmma_f16`, `has_wmma_f16_gfx12`)
  correctly skip gfx906. Not a portable target.
- **CDNA3 / MFMA**: rocBLAS HFQ4 path requires MFMA which is gfx940+
  only. Already covered above as tier 4.6.
- **gfx906 i8 WMMA**: doesn't exist on the chip.

## 6. Recommended ordering

If we pick this work up later:

1. **Probe (low risk, high info)**: rocprof attribution pass on
   gfx906 batched QKVZA prefill, current state. Goal: confirm that
   X-tile-load (or Q8_1 X scratch read) is ≥15% of QKVZA wall. If
   <5%, the fused-projection thesis is wrong and tier 4.1 should be
   deprioritized. Expected outcome (from PR #315's gfx1030 data and
   the gfx906 MMQ redesign body's known LDS pressure): X-load is a
   meaningful fraction. ~half-day.

2. **Tier 4.1 first kernel — qkv 3-way fused** (the lower-risk of
   the three; only 3 weight streams and all Qwen3.5 q_m/k_m/v_m are
   MMQ_Y-aligned, so no tail routing needed). Mirror the
   `gemm_hfq4g256_residual_mmq_gfx906_body.cuh` template structure
   exactly, parameterizing on mmq_x. Validate against
   `gemm_hfq4g256_mmq_set_gfx906` reference + KLD sweep. ~2-3 days.

3. **Tier 4.1 qkvza 4-way fused + split routing**. Once qkv 3-way
   is landed, qkvza is mostly a row-routing prologue change (skip
   the qkv/z branches when those Ms are zero, mirror existing
   `gemm_qkvza_hfq4g256_fp16_wave64.hip:54-61` pattern). The split
   routing for β/α tail is the same shape as PR #315 Phase 4. ~2 days.

4. **Tier 4.1 gate_up 2-way fused**. Smallest of the three; same
   topology as qkv minus one output. ~1 day.

5. **Tier 4.2 MMQ_Y sweep** on the new fused kernels — only after
   correctness is established. Add the parameterized MMQ_Y to the
   body.cuh, compile x32_y64 and x32_y96 variants, microbench. ~1 day.

6. **Tier 4.3 HFQ3 port** as a separate sub-project, if there's
   demand. Could be deferred indefinitely without blocking 4.1's
   gains. Plan-only here.

Each step should be gated behind a `HIPFIRE_HFQ4_MMQ_GFX906_FUSED=1`
env flag during development (mirroring PR #315's `HIPFIRE_HFQ4_MMQ_RDNA2`
opt-in gate) until the KLD / coherence validation passes.

## 7. Validation strategy

For each new fused-projection MMQ kernel:

1. **Correctness vs. reference**: A/B against the existing
   `gemm_hfq4g256_mmq_set_gfx906` × N path on identical inputs.
   Same precision delta as the current MMQ-set vs. fp16 wave64
   comparison documented in `gfx906-mmq-prd.md`. Acceptable bound:
   per-element max-abs error within 2× of the current MMQ-set path.
2. **End-to-end coherence**: `./scripts/coherence-gate.sh` on solo
   gfx906 9B prefill+decode. Must not regress.
3. **KLD vs. fp16 reference**: 512-prompt KLD sweep using
   `mq4_masked_calib.py` against the screen-reject fp16 path.
   Within +0.005 of the current MMQ-set KLD.
4. **Perf**: rocprof per-kernel attribution, fresh process. Must
   show fused kernel wall time ≤ 0.80 × sum-of-separate-calls.

## 8. Why this matters even with hetero PFlash working

The 2026-05-23 hetero PFlash result (TTFT 2.85 s on niah_8k, ✅) is good
**precisely because** the gfx906 prefill gap exists. If gfx906 batched
prefill closes to ~parity with gfx1031 on the 0.8B drafter (i.e., the
4× gap shrinks to ~1.5×, comparable to the raw hardware ratio), two
things change:

- **PFlash solo on gfx906** becomes competitive with PFlash hetero
  for medium prompts (8k tokens). Today: solo 3.13 s vs hetero
  2.85 s. Closing the drafter gap halves the solo-vs-hetero advantage.
- **Single-GPU deployments without a secondary card** become much
  more viable. Many gfx906 users won't have a second GPU sitting
  around; making solo PFlash fast is the broader win.

The hetero path stays valuable for very long prompts (16k+, where the
gap is 2.6 s today) because compress wall time scales with prompt
length and the secondary card is genuinely doing parallel work. But
narrowing the gap raises the floor for everyone.

## 9. Open questions

- ~~Is there a structural reason the gfx906 MMQ-set kernels can't be
  fused (e.g., LDS pressure from 3-4 accumulator registers)?~~
  **Resolved 2026-05-23:** No structural blocker. The accumulator
  footprint at the actually-shipped 2-way and 3-way fusions
  (`(mmq_x/4) × (MMQ_Y/64) × N_OUT` floats per lane) fits comfortably
  in wave64's 256 VGPRs/SIMD. The dispatcher's row-band routing
  prologue keeps only ONE accumulator live per WG (different WGs
  handle different output bands), so 4-way would only need the same
  per-lane regs as 3-way. The fusion is structurally safe; the
  4-way QKVZA wasn't built because the gfx906 dispatcher already
  splits QKVZA into a 2-way head (qkv+z, served by the 2-way kernel)
  plus a 2-way dp4a tail (β+α) — fusing all four into one kernel
  would mean ditching the working split, with no clear win.
- ~~Does AWQ scaling change the screen-reject rate?~~ **Moot:**
  `mmq_screen` is WMMA-only on gfx906 (confirmed during Phase 2
  implementation). `self.mmq_screen` defaults to false on gfx906,
  so `qkv_safe = true` always and the screen-reject branch is dead
  code on this arch. Tier 4.5 not applicable.
- ~~Should we ship gated symmetrically with `HIPFIRE_HFQ4_MMQ_RDNA2`?~~
  **Resolved 2026-05-23:** Shipped as separate flag
  `HIPFIRE_HFQ4_MMQ_GFX906_FUSED`, matching the established arch-
  specific convention. An umbrella flag would have to make
  per-arch routing decisions inside the gate, which is exactly
  what the per-arch flag already does at the dispatcher level.

## 10. Outcomes (2026-05-23)

Phases 2 + 3 + 4 shipped on branch `feat/gfx906-hfq4-mmq-fused`:

| Phase | Commit | Result |
|-------|--------|--------|
| §6.1 probe | 33749bd6 | mmq_set 53.6% of prefill wall; thesis confirmed |
| §6.2 qkv 3-way | 701b8caa | byte-exact, fires 8×/forward (FA layers) |
| §6.3 + §6.4 gate_up 2-way (also serves QKVZA-head) | f9f0fb62 | byte-exact, fires 56×/forward (gate_up + qkvza-head) |
| §6.5 MMQ_Y=64 sweep | 1b52c325 | **NEGATIVE** −5.6%, kept gated |
| §6.6 HFQ3 port | deferred per plan | n/a |
| Default-on (no env-gate) | (this commit) | post-coherence-gate + post-KLD-validation |

**Banked win:** +7.3% async prefill on Qwen3.5 9B MQ4 at B=256
(726.9 → 779.9 tok/s, σ ≈ 0.1%), **default on for gfx906**.
Within the §6.1 ceiling estimate (5-12%). Decode untouched.
Byte-exact A/B verified.

**Validation evidence for default-on:**
- byte-exact greedy_dump A/B (md5 match across all 3 fused paths)
- coherence-gate.sh on gfx906: 9 model/prompt cells passed (4 MQ4,
  5 MQ3 collateral), no hard errors
- KLD eval (n=50 chunks, asym3-KV, prefill scoring): 0.323142,
  inside CI band of canonical §1.1 master baseline 0.3376
  (CI 0.3263–0.3494) on the same `qwen3.5-9b.mq4` file

The MMQ_Y=64 research scaffold stays gated behind
`HIPFIRE_HFQ4_MMQ_GFX906_Y64=1` (default OFF) per PR #315's
kept-gated-negative-result convention.

## 11. Open next steps (post-Phase 4 lever inventory)

Phase 2-3 closed the structural-fusion gap predicted by §6.1. The
post-fusion attribution table (2026-05-23 re-probe, all fused
enabled, B=256 9B MQ4) leaves these as the dominant remaining
wall-time consumers:

| rank | kernel | %wall | per-call | calls/fwd | fusion state |
|------|--------|------:|---------:|----------:|--------------|
| 1 | `gate_up_hfq4g256_mmq_gfx906` | 46.4% | 2681 µs | 56 | fused; only kernel-internal levers left |
| 2 | `hfq4g256_residual_mmq_gfx906` (wo) | 34.2% | 1726 µs | 64 | single-output, no peer to fuse with |
| 3 | `gated_delta_net_q8_batch_seq` | 9.3% | 1257 µs | 24 | sequential by construction |
| 4 | `qkv_hfq4g256_mmq_gfx906` | 3.8% | 1547 µs | 8 | fused |
| 5+ | tail (rmsnorm-rotate, silu-mul-rotate, etc.) | 6.3% | — | — | already aggressively fused multi-op kernels |

The top 4 = 93.7% of wall. Levers below are ranked by expected
payoff × tractability. **None of them is on the critical path** —
the +7.3% from Phases 2-3 is the bulk of what was theoretically
available. These are diminishing-returns territory.

### 11.1 Tier A — Wider mmq_x sweep on fused gate_up (recommended next)

Today the fused gate_up has wrappers for mmq_x ∈ {8, 16, 32, 64}.
The residual sibling has the full 8-value sweep {8, 16, 24, 32,
40, 48, 56, 64}; that sweep exists because PMC validation on the
residual showed per-call wins at the in-between values. The
fused 2-way kernel has the same inner loop and likely the same
sweet spots, but at different absolute values (the LDS layout
differs because the 2-way kernel has 1 accumulator set vs the
residual's set-mode-also-1).

**Expected payoff:** 1-5% prefill-wide (most likely 1-3%).
gate_up dominates 46.4% of wall; even a 5% per-call win there is
2.3% prefill-wide. The most likely outcome is that the existing
x32/x64 choices are near-optimal and one in-between value
(x40 or x48) gives 1-2%.

**Tractability:** Low effort. Generate 4 new .hip wrappers via
sed (same as we did for x16/x32/x64), wire kernels.rs, extend
the dispatcher's batch-size heuristic from 4 buckets to 8.
~1 day work.

**Risks:**
- LDS budget at large mmq_x: at mmq_x=64 the body already uses
  30,720 B/WG (per the body's commentary, 2 WG/CU limit). Adding
  x48, x56 stays under this; no occupancy regression risk for
  these. x40 with stride=40 sits at the same 2 WG/CU.
- The b32 fallback path for mmq_x<32 in the body: in-between
  values straddle the cliff. x24 stays on b32, x40 on b128. The
  sweep needs to confirm the cliff is at the right boundary for
  the 2-way kernel (residual's PMC showed it; 2-way is
  structurally identical so likely the same).
- The byte-exact A/B test still applies — each new wrapper needs
  greedy_dump validation before being added to the dispatcher's
  selection map.

### 11.2 Tier B — Residual `wo` MMQ probe (34.2% of wall, untouched)

The post-attention `wo` GEMM is the SECOND-largest wall consumer
and has had no Phase 2-3 work. It's a single-output kernel using
the original `gemm_hfq4g256_residual_mmq_gfx906` path (the
add-mode variant from the §3 inventory). Per-call it runs slower
than the fused QKV at similar shape (1726 µs vs 1547 µs), and
PR #315 didn't ship a Stream-K variant either, suggesting the
team didn't find an obvious win.

**Possible angles** (ordered by tractability):

1. **rocprof attribution on the residual itself.** Why is it
   15.3 GiB/s vs the fused gate_up's 21.9 GiB/s? Same body,
   different write-back (add vs set). The add-mode adds one
   load + one store per output element vs set's just-store —
   that's a 2× write-traffic delta. If write-traffic is the
   limiter, the residual is bandwidth-bound and there's no
   kernel-internal lever.

2. **Stream-K / split-K reduction.** Plan §4.8 raised this and
   deferred. The wo shape is `(M=4096, K=v_dim=1024, N=256)` —
   short K, square M, modest N. Split-K-2 would split the K=1024
   axis into two K=512 partial GEMMs + an atomic-add reduction.
   On gfx906 with 60 CUs and a 76-WG residual grid, splitting
   K would double WG count → better CU utilization.
   **Expected payoff:** 5-15% on the residual kernel = 1.7-5%
   prefill-wide. **Tractability:** medium effort (new kernel,
   new dispatcher entry, atomic-add semantics carry the same
   "soft output change" risk as any non-deterministic
   accumulation order).

3. **Same wider mmq_x sweep as Tier A.** Residual already has
   the full 8-value sweep, but the auto-selector picks one based
   on a fixed heuristic. May not be optimal post-fusion (the
   post-fusion forward has different L2 residency for Xq vs
   pre-fusion). **Tractability:** trivial (no new code, just
   re-tune the existing sweep selection). **Risk:** very low.
   **Payoff:** probably <2% — the existing heuristic was tuned
   pre-Phase 2-3 but the residual's L2 conditions haven't
   changed much.

**Risks (general):**
- Stream-K's atomic-add reduction introduces non-deterministic
  accumulation order. Byte-exact A/B test will fail. Need to
  switch to the KLD-bound validation strategy from plan §7.3
  for this kernel.
- PR #315 not shipping Stream-K is evidence (but not proof) that
  the team probed it and found no clean win on gfx1030. gfx906
  may differ — but expect to spend half the time understanding
  why the obvious approach is more complicated than it looks.

### 11.3 Tier C — gate_up kernel-internal optimization (already-fused improvements)

Three concrete sub-levers inside the existing fused 2-way kernel:

1. **dp4a prefetch.** Prefetch the next sub-block's x_qs / tile_y
   while the current sub-block computes. **Risk:** memory
   `feedback_dp4a_prefetch_no_op_2026_05_18` reports this was a
   measured no-op on MoE dp4a kernels in 2026-05-18 — the dp4a
   body is too compute-light to benefit from prefetch. The fused
   2-way is heavier per inner iter than the MoE kernels were
   (2× accumulators), so it might respond differently. **Honest
   payoff estimate:** probably 0%, possibly 1-2%. **Tractability:**
   easy A/B.

2. **LDS bank conflict probe at mmq_x≥32.** The body uses
   stride=40, which is `40 % 32 = 8` → 4-way bank conflict
   (documented in body lines 82-100). PMC-validated as the right
   choice for the residual at mmq_x ∈ {32, 40, 48, 56}, but
   the 2-way fused kernel may have a different optimum given
   the different per-WG work distribution. **Tractability:**
   easy (run a stride=33/40/41/48 sweep at fixed mmq_x). **Risk:**
   PMC counter access via rocprofv3 segfaults on gfx906 (per
   the §6.1 probe writeup) — would need to compare via wall-time
   only, which is noisier. **Payoff:** 1-3% if a different
   stride wins.

3. **`__launch_bounds__(256, 2)` tuning.** Current hint is 2 WG/CU.
   At gfx906 LDS budget the actual ceiling is 2 WG/CU (30,720 B
   each fits in 64 KB). Adjusting the hint to 1 (let compiler
   spill less) or 3 (over-promise) could shift register
   allocation. **Tractability:** trivial. **Payoff:** probably 0
   (the actual ceiling is hardware-determined). Listed for
   completeness.

### 11.4 Tier D — Knobs (no new kernels)

1. **`PREFILL_MAX_BATCH` increase.** Currently 256 (per plan §4.7
   reference). Larger chunks → fewer kernel launches per token.
   **Risk:** VRAM cost grows linearly (`PREFILL_MAX_BATCH × K`
   for the Q8_1 X scratch alone), and the kernel auto-selector
   already picks mmq_x=64 at B=256 — at B=512 it'd still pick 64
   so the per-call cost doesn't scale down. Net effect:
   half-as-many launches at twice the per-launch cost. Probably
   ~0%. **Tractability:** trivial (one constant). **Worth
   probing:** yes, because cheap.

2. **`HIPFIRE_HFQ4_MMQ_GFX906_FUSED=1` becomes default on gfx906.**
   After this branch lands and bakes for a release cycle without
   coherence reports, flip the env-gate default to `true`. Saves
   the production user from having to know about the flag.
   **Tractability:** one-line dispatch.rs change. **Risk:** the
   model matrix coverage today is mq4-only (9B verified). MQ3
   models route through different gemm_hfq3 family kernels not
   affected by this branch, so no risk there. MQ6 + AWQ-stacked
   weights should be sanity-checked first.

### 11.5 Tier D / Skip — Already-fused tail kernels

The 6.3% combined tail is fragmented across:
`fused_rmsnorm_mq_rotate_batched`, `fused_silu_mul_mq_rotate_batched`,
`fused_qk_l2_norm_scale_interleave_f32_batched`,
`fused_sigmoid_alpha_gate_f32_batched`, etc. These are already
multi-op fused kernels. No obvious next-level fusion target —
they're at different points in the forward pass and can't be
combined without restructuring the whole layer-level dispatch.
**Skip.**

### 11.6 Tier D / Skip — DeltaNet recurrence

Plan §4.7 covered this: the recurrence is sequential by construction.
9.3% of wall is the intrinsic floor without a batched-DN kernel,
which is a separate research project. **Skip.**

### Honest summary of remaining headroom

| Tier | Lever | Effort | Expected payoff | Risk |
|------|-------|--------|-----------------|------|
| A | Wider mmq_x sweep on fused gate_up | 1 day | 1-5% (likely 1-3%) | low |
| B.1 | Residual rocprof attribution | half day | informational | none |
| B.2 | Residual Stream-K split-K | 2-3 days | 1.7-5% prefill-wide | medium (non-determinism, may match PR 315's deferred conclusion) |
| B.3 | Residual mmq_x re-tune | trivial | <2% | very low |
| C.1 | gate_up dp4a prefetch | 1 day | probably 0%, possibly 1-2% | low |
| C.2 | gate_up LDS bank conflict probe | 1 day | 1-3% | low (PMC access limited) |
| C.3 | gate_up launch_bounds tuning | trivial | probably 0% | very low |
| D.1 | PREFILL_MAX_BATCH increase | trivial | probably 0% | VRAM cost |
| D.2 | Make FUSED=1 the default | trivial | 0 (latent: ergonomics) | model matrix coverage gap |

**Realistic ceiling on this stack of levers without writing
fundamentally new kernels: probably +3-7% additional prefill,
bringing total over baseline to ~10-15%.** That's roughly half
of what we already banked with Phase 2-3. Diminishing returns.

If we want a step-change rather than another marginal squeeze,
the right next project is **batched DeltaNet recurrence** (would
unlock the 9.3% bucket entirely) or **rocBLAS HFQ4 prefill path**
(plan §4.6 — currently CDNA3-only, gfx906 not eligible until
rocBLAS's gfx906 GEMM performance improves in a future ROCm
release). Both are separate plan documents.
