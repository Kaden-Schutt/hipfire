# MQ3 batched-prefill on gfx10 (RDNA1 / RDNA2) — starting-point plan

Branch: `feat/mq3-gfx10-perf` (repurposed from the closed decode-perf
investigation — see `docs/plans/mq3_gfx10_perf.md` for the
predecessor)
Branched off: `master` at `e0381119`
Status: starting-point recommendation; not yet a committed scope.

## Why this exists

Phase 0 of the decode-perf investigation (2026-05-19, see
`findings/mq3-gfx10-perf-2026-05-19/README.md`) measured the actual
gfx1031 numbers and revealed:

| Workload | MQ3 tok/s | MQ4 tok/s | Ratio |
|---|---|---|---|
| AR decode (batch=1) | 55 | 50 | **MQ3 1.11× faster** |
| Prefill (batch>1) | 55 | 225 | **MQ3 4.0× slower** |

The decode side is fine — MQ3 is already winning. The prefill side
is the actual gap, and it's structural: `is_batchable_la(MQ3G256,
gfx10)` returns `false` (`qwen35.rs:4014-4046`), so MQ3 prefill on
gfx10 falls through to per-token `forward_scratch`. MQ3 prefill rate
≈ MQ3 decode rate ≈ 55 tok/s — there's no batched kernel doing any
amortization.

MQ4 on the same arch hits the batched HFQ4 GEMM family
(`gemm_qkv_hfq4g256.hip`, `gemm_gate_up_hfq4g256.hip`,
`gemm_hfq4g256_residual.hip`, plus dp4a / fp16 / dot2 variants) and
gets a 4× speedup from amortizing weight loads across the batch
dimension.

The goal: **build the batched HFQ3 prefill family for gfx10 so MQ3
can join `is_batchable_la`'s allow-list, closing the 4× prefill
gap.** The user-visible impact is the eval becoming tractable (~3h
→ ~45 min for n=256) AND prefill-heavy workloads (long prompts, RAG,
agentic) becoming usable on MQ3-quantized models on RDNA2 hardware.

This is a **much larger project than decode tuning** — order of
several weeks of focused kernel work vs the decode plan's ~2-day
investigation. The doc below is a starting point, not a fully spec'd
design.

## Reference inventory — what exists for HFQ4 that we'd need for HFQ3

Counted from `ls kernels/src/`:

| Family | HFQ4 variants on gfx10 | HFQ3 variants on gfx10 |
|---|---|---|
| `gemm_qkv_*` | scalar, dot2, fp16, fp16_wave64, wave64, wave64_dp4a | none (only gfx11/gfx12 WMMA) |
| `gemm_qkvza_*` (MoE) | scalar, dot2, fp16, fp16_wave64, wave64, wave64_dp4a | none |
| `gemm_gate_up_*` | scalar, dot2, fp16, fp16_wave64, wave64, wave64_dp4a, wmma_ldsx | none |
| `gemm_*_residual_*` | residual, residual_fp16, residual_fp16_wave64, residual_mmq_gfx906_x{8..64} (8 tile sizes) | only gfx11/gfx12 WMMA |
| `fused_qkv_*` | scalar, wave64, wave64_dp4a | none |
| `fused_qkvza_*` | scalar, wave64, wave64_dp4a | none |
| `fused_gate_up_*` | scalar, wave64, wave64_dp4a | none |

That's roughly **20+ HFQ4 variants** with no HFQ3 sibling. The plan
below is **not** to write 20+ kernels — only the minimum needed to
flip `is_batchable_la` for gfx10 MQ3.

## Goals + non-goals

**Goals (in priority order, calibrated against Phase 0 baseline at
`findings/mq3-gfx10-prefill-2026-05-19/`):**
1. **MQ3 prefill tok/s on gfx1031 ≥ 3× current** (~56 → ~165
   tok/s) — the Phase 1 milestone. Closes roughly half the gap to
   MQ4's 290 saturation. Conservative because the first port is
   scalar (no dp4a / dot2), comparable in structure to
   `gemm_qkv_hfq4g256.hip` baseline before its tuned variants.
2. **MQ3 prefill tok/s ≥ 5× current** (~56 → ~280 tok/s) — the
   eventual Phase 2-3 milestone with dp4a or MMQ. Close to MQ4
   parity.
3. **`is_batchable_la(MQ3G256, gfx10)` returns `true`** with the
   new kernel family in place, gated on the gfx1030/gfx1031 arch.
4. **No decode regression** — MQ3 AR decode stays ≥ 1.05× MQ4 (it's
   currently 1.11×; small drop OK if the prefill win is large).
5. **No correctness regression** on the existing coherence-gate
   `mq3-awq-paris` row + the lm_head AWQ gate from PR #292.

**Non-goals:**
1. **MoE / A3B MQ3 prefill.** `qkvza` variants for MoE are out of
   scope. MQ3 in MoE blocks stays refused upstream (`is_mq3_any`
   checks in qwen35.rs:3589/3599/3767). Worth ~25% of the kernel
   work; defer to a follow-up if the dense MQ3 prefill wins justify it.
2. **MQ3-Lloyd prefill on gfx10.** Lloyd kernels are WMMA-only by
   design. Separate, larger project.
3. **gfx906 / gfx94x.** The MMQ pattern is mined here for
   inspiration but a real port to those archs is separate.
4. **RDNA1 (gfx1010).** Same arch class as gfx1030 for this purpose,
   but the cache hierarchy differs (no Infinity Cache). Backfill if
   the gfx1030 v1 kernels port cleanly; otherwise file as follow-up.
5. **Batched lm_head (`gemm_hfq3g256_batched_lmhead`).** Still
   deferred per the decode plan's glm5 C1: unreachable from current
   callers (DFlash MQ3 on gfx10 refused upstream; eval and AR decode
   use `weight_gemv`). Re-open IFF DFlash MQ3 on gfx10 is unblocked.

## Reference points (kernels we mine)

### Reference A: scalar HFQ4 batched-prefill family (gfx10 baseline)

The plain `gemm_*_hfq4g256.hip` family (no arch suffix) is the
non-WMMA, non-dp4a, non-wave64 baseline. These run on gfx10 today as
the fallback when more specialized variants don't apply. Patterns:

- `gemm_qkv_hfq4g256.hip` — fuses Q, K, V projections into one
  kernel. Reads x once from VRAM, writes to three output tiles.
- `gemm_gate_up_hfq4g256.hip` — fuses gate + up projections.
- `gemm_hfq4g256_residual.hip` — y[b] += A @ x[b], for the
  post-attention/post-MLP residual fusion.

These are the smallest-effort port targets. Replicate the kernel
shapes with HFQ3's 104 B group stride and 3-bit unpack.

### Reference B: gfx906 MMQ family (dp4a + Q8_1 pre-quantized x)

`gemm_hfq4g256_residual_mmq_gfx906_x{8..64}.hip` (8 tile sizes,
shared body at `gemm_hfq4g256_residual_mmq_gfx906_body.cuh`):

- Pre-quantizes x to `block_q8_1_mmq` once before the batched
  matmul. Cost amortized over the batch.
- Inner loop uses `v_dot4_i32_i8` (`__builtin_amdgcn_sdot4`) for 4
  INT8 MACs per VALU cycle (4× MAC throughput vs FP32 FMA, per
  `gemv_mq8g256.hip:5`).
- `launch_bounds(256, 2)` — 256-thread workgroups for tile-level
  parallelism over the X dimension.
- 8 tile sizes (x8, x16, x24, x32, x40, x48, x56, x64) selected per
  batch size to balance occupancy and tile overhead.

**The key portable lesson:** the MMQ structure (Q8_1 staging +
dp4a inner loop + tiled X) is **not gfx906-specific** — it works
on any arch with `v_dot4_i32_i8`, which includes RDNA2 (gfx1030+).
The `_wave64` suffix on those files is incidental; RDNA2 wave32 +
dp4a is the equivalent setup.

### Reference C: WMMA HFQ3 prefill family (gfx11/12, correctness ref)

`gemm_qkv_hfq3g256_wmma.hip`,
`gemm_qkvza_hfq3g256_wmma.hip`,
`gemm_gate_up_hfq3g256_wmma.hip`,
`gemm_hfq3g256_residual_wmma.hip` — the existing gfx11+ MQ3
prefill family. **Cannot port** (WMMA-only), but useful as a
golden output: any new gfx10 HFQ3 prefill kernel should produce
results within FP-non-associativity tolerance of these. Test
matrix should include cross-arch parity checks.

## Phases

Cheapest-first. Each phase produces a shippable increment that
either (a) widens `is_batchable_la` for MQ3 on gfx10 or (b)
explicitly defers that until the kernel quality is high enough.

### Phase 0 — measurement baseline ✅ **DONE 2026-05-19**

Full findings: `findings/mq3-gfx10-prefill-2026-05-19/README.md`.
Summary:

1. **Prefill scaling curve confirms the gap is batch-dependent and
   widens to ~5.16× at saturation** (MQ3 flat at ~56 tok/s; MQ4
   saturates near 290 tok/s by batch ~200). 4-prompt sweep across
   prefill sizes 10 / 21 / 191 / 240 tokens on the same hardware.
2. **HFQ4 batched-prefill kernel metadata** establishes the budget:
   - 61 VGPRs for single-token fused (`fused_qkv_hfq4g256`,
     `fused_gate_up_hfq4g256`) — 16 waves/SIMD
   - **98 VGPRs for batched prefill** (`gemm_qkv_hfq4g256_dot2`,
     `gemm_gate_up_hfq4g256_dot2`, `gemm_hfq4g256_residual_fp16`)
     — 9 waves/SIMD, occupancy traded for per-thread work
   - LDS=0 across all batched prefill kernels (no shared memory)
   - workgroup_size=32 (single warp) universal — thread-level
     parallelism comes from grid.x, not workgroup cooperation
3. **Decision:** Phase 1 targets the 98-VGPR tier. New HFQ3
   batched kernels should mirror the launch_bounds and occupancy
   profile of their HFQ4 siblings.

### Phase 1 — minimum viable batched prefill: scalar HFQ3 family

The smallest possible scope that flips `is_batchable_la`. Three
new kernels modeled on the plain HFQ4 family:

- `kernels/src/gemm_qkv_hfq3g256.hip` (mirror of
  `gemm_qkv_hfq4g256.hip`, swap 136 B → 104 B group stride, swap
  4-bit nibble unpack → 3-bit trit unpack from packed uint24)
- `kernels/src/gemm_gate_up_hfq3g256.hip` (same pattern)
- `kernels/src/gemm_hfq3g256_residual.hip` (already exists for the
  WMMA path — verify the non-WMMA arch-fallback variant or write one)

Plus dispatcher wiring:

- New entries in `dispatch.rs` for each kernel
- `qwen35.rs::is_batchable_la` admits `MQ3G256` on
  `gfx1030 | gfx1031` (and probably `gfx1010` if it ports cleanly)
- The four `is_mq3` matchers in `qwen35.rs` (lines 4063, 4360,
  4768, 4919 per the comment at qwen35.rs:4002) route to the new
  kernels on gfx10 instead of the WMMA-only family that currently
  exists

**Expected uplift:** modest. Scalar HFQ4 batched prefill on gfx1031
runs through `gemm_qkv_hfq4g256.hip` etc. The HFQ3 equivalents at
the same arch should land at similar perf — but HFQ4 itself uses
the dp4a / dot2 variants where possible, so this baseline only
catches the cases where MQ4 falls back to scalar too. Realistic
estimate: prefill tok/s 55 → 100-120, closing ~half the gap to MQ4.

**Validation gate:** coherence-gate clean, AR decode no regression,
prefill tok/s ≥ 1.8× current. Bit-exact reference test against the
gfx11 WMMA variants for the same M/K/N at FP-non-associativity
tolerance.

### Phase 2a — `__launch_bounds__(32, 6)` for HFQ3 batched family ✅ **DONE 2026-05-19**

Eliminates spills by capping waves/SIMD at 6 (was unbounded, leading
to register pressure → spills). All 4 batched HFQ3 kernels rebuilt at
158 VGPR + 0 spills. Lands ~+8-16% prefill on top of Phase 1.

Cumulative result (Phase 1 + 2a): 9B MQ3 prefill 56 → 148 tok/s
(2.63×) at gfx1031. Coherence + KLD healthy.

### Phase 2b — `v_dot2_f32_f16` family for HFQ3 batched prefill ✅ **DONE 2026-05-19**

Unscheduled but landed cleanly. Mirrors the HFQ4 dot2 family — 4
new HFQ3 kernels using `amd_mixed_dot` (one `v_dot2_f32_f16` per
half2 dot pair with FP32 accumulation):

- `kernels/src/gemm_qkv_hfq3g256_dot2.hip`
- `kernels/src/gemm_qkvza_hfq3g256_dot2.hip`
- `kernels/src/gemm_gate_up_hfq3g256_dot2.hip`
- `kernels/src/gemm_hfq3g256_residual_dot2.hip`

Each fires from the scalar dispatcher when `has_dot2_f32_f16(arch)`
and `batch_size > 1` — covers gfx1011/1012/1030-1032 and gfx11/12.
Inputs go through `ensure_fp16_x` so X is pre-converted FP16.

All 4 kernels: 98 VGPR + 52 SGPR + 0 spills + 0 LDS (matches the
HFQ4 dot2 sibling budget).

Result on 9B MQ3 prefill at gfx1031: **148 → 224-249 tok/s** across
the eyeball matrix (`paris/sheep/code/awq` from
`/tmp/eyeball_phase2a.jsonl`), **~+60% vs Phase 2a baseline**, drift
<0.3% across two warm-cache runs. Cumulative **56 → 234 tok/s
(4.18×)** vs the pre-batched scalar baseline.

KLD eval at n=30, KV=Q8 on
`/data/hipfire/qwen3.5-9b.mq3-awq-gptq-f2-lmhead-a100.hfq`:
slice-mean KLD = 0.191693, PPL = 9.9466 — healthy (Phase 1 baseline
at n=256 was KLD 0.219, PPL 9.93; n=30 is a smaller sample so not
directly comparable, but the magnitude confirms no regression).
Note: eval_hipfire's scoring path runs per-token forward, not
batched-prefill — so KLD here mostly validates the per-token path is
unchanged. The dot2 quality signal comes from the eyeball
(fluent across 4 prompts on 2 runs).

Decode unchanged at 55 tok/s.

### Phase 2c — `v_pk_fma_f16` family for gfx1010 / gfx1013 ✅ **DONE 2026-05-19**

Mirrors Phase 2b for archs without the dot extension (gfx1010 Navi 10 /
RX 5700 XT — the project's primary target — and gfx1013 Van Gogh /
BC-250 APU). Four new HFQ3 kernels using `v_pk_fma_f16`
(`__hmul2` + 3× `__hfma2` + extract + add, FP32 cross-group
accumulation):

- `kernels/src/gemm_qkv_hfq3g256_fp16.hip`
- `kernels/src/gemm_qkvza_hfq3g256_fp16.hip`
- `kernels/src/gemm_gate_up_hfq3g256_fp16.hip`
- `kernels/src/gemm_hfq3g256_residual_fp16.hip`

Each scalar HFQ3 dispatcher's fan-out is now: `batch_size > 1 &&
!fp16_disabled()` → `has_dot2_f32_f16(arch)` → dot2, else fp16.
Mirrors the HFQ4 fan-out exactly (which goes WMMA → dot2 → fp16).

All 4 kernels: 98 VGPR + 52 SGPR + 0 spills + 0 LDS (identical
budget to the dot2 family — same loop structure, different
inner-op set).

Validation: `verify_hfq3_batched` extended with fp16-direct calls
(bypassing auto-routing since gfx1031 prefers dot2). All passes
at 2e-1 max-abs-err tolerance against the FP32 per-row reference
— ~0.14 max_err which matches the FP16 mantissa precision over a
512-element accumulation. The end-to-end perf signal on gfx1010
can't be measured on the current dev host (gfx1031 doesn't route
through fp16), but the dot2 family at the same dispatch shape +
launch geometry delivered +60% on gfx1031; fp16 should give the
matching uplift on archs without dot2 (≈+30-50% over scalar based
on the HFQ4 fp16 vs scalar parity reference).

### Phase 2 — dp4a inner loop for the qkv / gate_up GEMMs ⚠️ **NEGATIVE RESULT 2026-05-19**

Ported the gfx906 wave64-dp4a inner loop to wave32 + HFQ3 unpack on
RDNA2:

- `kernels/src/gemm_qkv_hfq3g256_dp4a.gfx1030.hip`
- `kernels/src/gemm_gate_up_hfq3g256_dp4a.gfx1030.hip`

Both kernels compile clean (35 VGPR + 21 SGPR + 0 spills — much
lower budget than the 98-VGPR dot2 family, plenty of occupancy
headroom) and emit two `v_dot4_i32_i8` per inner iteration as
expected. Output is coherent (eyeball matrix fluent across all
prompts, within INT8 quant noise of the dot2 baseline).

**Perf result on gfx1031, 9B MQ3:**

Short-prefill (eyeball matrix, 21-36 prompt tokens):

| Prompt | dot2 (Phase 2b, shipping) | dp4a (Phase 2) | Δ |
|---|---|---|---|
| paris  | 223 tok/s | 175 tok/s | **−22%** |
| sheep  | 249 tok/s | 216 tok/s | **−13%** |
| code   | 224 tok/s | 189 tok/s | **−16%** |
| awq    | 231 tok/s | 197 tok/s | **−15%** |

Long-prefill probe (testing whether Q8_1 X conversion amortizes):

| Prefill tokens | dot2 | dp4a | Δ |
|---|---|---|---|
| ~30 (median of eyeball) | 234 tok/s | 194 tok/s | **−17%** |
| 240 (LRU PEP-8) | 292 tok/s | 242 tok/s | **−17%** |
| 1188 (LRU ×5) | 278 tok/s | 244 tok/s | **−12%** |

**Median ~15% regression vs dot2, persists across short and long
prefill.** The gap narrows slightly at very long N (one-shot
conversion better amortized) but dp4a never catches dot2 — the
disadvantage is per-batch-element, not a fixed overhead. The plan's flagged risk (3-bit
unpack overhead eating the sdot4 ALU lift) materialized exactly on
RDNA2. Hypotheses:
1. **gfx906 win was relative to weak fp16:** gfx906's fp16 path is
   wave64-only and saw a real lift from dp4a's 4×-ALU throughput.
   RDNA2 already has a strong dot2 path at native FP16, only 2×
   ALU but no Q8_1-X conversion. The relative win disappears.
2. **3-bit unpack costlier than 4-bit:** HFQ4 nibble unpack is 8
   shifts + 8 masks. HFQ3 trit unpack adds 8 byte subtractions
   (signed mapping) + uint24 byte-combine. More VALU pressure per
   K-element offsets the sdot4 win.
3. **Q8_1 X conversion cost amortized worse at short N:** the
   `ensure_q8_1_mmq_x` adds a one-shot conversion cost. At the
   short-prompt regime tested (21-36 prompt tokens), this hurts
   more than it helps.

**Disposition:** Code is gated behind `HIPFIRE_HFQ3_DP4A=1` (default
off — dot2 ships). Keep the gated implementation for:
- Reproducible negative-result reference (this is why we test, not
  just port)
- A/B comparison on different RDNA2 SKUs / models / prompt regimes
  where the calculus might invert (long prefill, larger Infinity
  Cache, etc.)
- Future MQ3/Q8 hybrid quantization experiments

**Do NOT enable by default.** Phase 3 (MMQ tile sweep) reuses the
same Q8_1 infrastructure — should be evaluated with the
understanding that the dp4a inner-loop did not pan out here.

**Validation:** `verify_hfq3_batched` with `HIPFIRE_HFQ3_DP4A=1`
exercises both new kernels at 5e-1 tolerance (Q8_1 X adds ~3× the
dot2 error band). Coherence eyeball produces fluent text across the
4-prompt matrix.

### Phase 3 — MMQ tiling for batched-lm_head adjacency

Reuse the gfx906 MMQ tile-size sweep pattern for the gfx10 HFQ3
family. The 8 tile sizes (x8..x64) in HFQ4-mmq-gfx906 exist
because different batch sizes favor different occupancy/tile
trade-offs.

Two flavors:
- `gemm_hfq3g256_residual_mmq_gfx1030_x{16,24,32}.hip` (start with
  3 tile sizes, not 8 — let Phase 0 / Phase 2 numbers dictate
  which are worth shipping)
- Shared body at `gemm_hfq3g256_residual_mmq_gfx1030_body.cuh`
  (mirror of the gfx906 body file)

This is the largest engineering investment in the plan. Skip if
Phase 2 lands close enough to MQ4 parity that the marginal MMQ
work isn't worth it.

**Validation gate:** Prefill tok/s ≥ 3× current (the headline goal).

### Phase 4 — wider arch coverage (gfx1010 backfill)

If Phase 1-3 land cleanly on gfx1030/1031, retest on gfx1010 (RDNA1
Navi 10, no Infinity Cache). The kernel might run unchanged but the
cache strategy (Phase 3 MMQ tile size) is likely different.

If any kernel regresses on gfx1010, ship the gfx1030/1031-only path
first; backfill gfx1010 separately. Tracked here for completeness;
not the priority.

## Validation gates (apply to every phase)

1. **Build clean.** Standard `cargo build --release --example daemon
   --features deltanet`.
2. **Coherence-gate clean on gfx1031.** `mq3-awq-paris` row from PR
   #292 catches AWQ correctness; new prefill paths must not break it.
3. **`coherence-gate-dflash.sh` clean on gfx1100** (where WMMA MQ3
   prefill lives). Cross-arch regression catcher.
4. **AR decode no regression** — 4-prompt sweep, MQ3 decode tok/s
   on gfx1031 stays ≥ 1.05× MQ4 (currently 1.11×). The new prefill
   kernels must not perturb decode dispatch via shared scratch
   buffers or fused-norm wiring.
5. **KLD eval parity** — every phase that touches inner-loop
   numerics (Phase 2's dp4a, Phase 3's MMQ x-quantization) runs
   `eval_hipfire --n=256 --kv-mode q8` on the 9B mq3-awq-gptq-f2-lmhead
   artifact. Slice-mean KLD must stay within 5% of the previous
   phase's baseline.
6. **Prefill tok/s headline** per phase — the goal of this plan.

## First steps (Phase 0, before any kernel code)

1. **Build a minimal prefill microbench** that exercises
   `forward_prefill_batch` directly on a synthetic batch (no model
   load) for both MQ3 and MQ4 on gfx1031. Decouples the kernel
   measurement from the eval / inference framework overhead.
2. **Catalog the existing scalar HFQ4 batched kernels** in detail —
   not just the file list above, but the actual launch
   configuration, VGPR usage, and gfx1031 wall-time per launch.
   Establishes "Phase 1 will land here-ish" expectations.
3. **Decide MoE qkvza in/out before writing any kernel.** The
   `gemm_qkvza_hfq3g256.hip` family is ~25% extra effort. If MoE-MQ3
   isn't on a near-term roadmap, drop it from the scope explicitly.

## Risks + open questions

- **Bit-extraction overhead may dominate** any dp4a or wave-density
  win. Gemini's 2.1 from the decode plan applies here too — 3-bit
  unpack is more VALU-heavy than 4-bit. The Phase 2 dp4a port might
  not beat Phase 1 scalar. Need explicit Phase 0 measurement of
  bit-op count.
- **Group stride mismatch corruption** — the four `is_mq3` matchers
  in qwen35.rs are paired with the WMMA-only family. Adding HFQ3
  scalar/dp4a variants without updating ALL matchers risks the
  "HFQ4-stride GEMM reading a different-stride weight block"
  failure mode the comment at `qwen35.rs:4001-4008` calls out.
  Update matchers atomically with the dispatcher widening.
- **MoE-MQ3 might be silently exercised** if a user runs an A3B
  model with MQ3 weights — the gates at qwen35.rs:3589/3599/3767
  should refuse this but Phase 1's `is_batchable_la` widening risks
  a code path that drops into `forward_prefill_chunk` for MoE
  blocks. Audit before flipping the gate.
- **Coherence cliff** — MQ3 at 3-bit is already at the edge.
  Activation quantization for dp4a (Phase 2) compounds the noise.
  KLD gate is the canary, not the coherence-gate substring check.
- **Engineering scope** — this is realistically 2-4 weeks of focused
  kernel work for one engineer. If the user's timeline is
  measured in days not weeks, ship Phase 1 only (scalar HFQ3
  batched + dispatcher wiring + `is_batchable_la` widening) and
  accept ~2× prefill speedup instead of the 4× target. Half the
  win for 1/4 the work.

## References

- Closed predecessor: `docs/plans/mq3_gfx10_perf.md` (decode-perf
  investigation, Rev 3 — Phase 0 findings caused the pivot here)
- Sibling correctness: `docs/plans/mq3_gfx10.md` §12 (MQ3 cross-arch
  correctness, resolved 2026-05-18 via AWQ-loader fix)
- Phase 0 data: `findings/mq3-gfx10-perf-2026-05-19/README.md`
- Dispatcher: `qwen35.rs::is_batchable_la` (line 4014) and the four
  `is_mq3` matchers at qwen35.rs:4063/4360/4768/4919
- Reference kernels: see §"Reference points" above

## tl;dr

- **Prefill is the actual MQ3-on-gfx10 user pain.** 4× slower than
  MQ4 in the same workload class. Decode is already faster than
  MQ4 — no action there.
- **Root cause:** `is_batchable_la(MQ3G256, gfx10)` is `false`.
  Forces per-token forward_scratch path. The fix is building the
  batched HFQ3 prefill family that admits MQ3 to the allow-list.
- **Phase 1** (scalar HFQ3 qkv + gate_up + residual + dispatcher
  wiring) is the minimum viable: ~50% of the gap closed for ~25%
  of the kernel work. ✅ DONE 2026-05-19 — 56 → 137 tok/s (2.45×).
- **Phase 2a** (`__launch_bounds__(32, 6)`, spill elimination) ✅ DONE
  2026-05-19 — 137 → 148 tok/s (+8%).
- **Phase 2b** (HFQ3 dot2 family — qkv + qkvza + gate_up + residual
  using `v_dot2_f32_f16`) ✅ DONE 2026-05-19 — 148 → 234 tok/s
  (+58%). **Cumulative 56 → 234 tok/s (4.18×).**
- **Phase 2c** (HFQ3 fp16-packed family for gfx1010/1013 — same
  4 kernels using `v_pk_fma_f16`, dispatcher fan-out: dot2 → fp16) ✅
  DONE 2026-05-19. No perf signal on gfx1031 (auto-routes to dot2);
  benches deferred to gfx1010 hardware. Same VGPR/spill budget as
  the dot2 family.
- **Phase 2 (dp4a)** ⚠️ NEGATIVE RESULT 2026-05-19 — port of gfx906
  wave64-dp4a to RDNA2 wave32 with HFQ3 unpack is ~15% SLOWER than
  the shipping dot2 path. Code retained behind `HIPFIRE_HFQ3_DP4A=1`
  for reproducibility / future experiments but does not ship.
- **Phase 3 (MMQ tile sweep)** remaining as the last lever for any
  further headroom — should be evaluated against the dot2 baseline,
  not scalar, and with the Phase 2 negative result in mind.
- **Phase 0 measurement required first** — same lesson as the
  decode plan. Don't write kernels before measuring.
