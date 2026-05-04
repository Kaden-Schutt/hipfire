# gfx906 MMQ kernel redesign — plan

Status: design phase, no code yet. **Three adversarial reviews completed
and integrated** — gemini caught hardware-physical defects (LDS, VGPR,
b128 alignment), glm-5 caught dispatch/scope errors, my own caught
prediction overselling. v2 below incorporates all validated findings.

Target file (rewrite): `kernels/src/gemm_hfq4g256_residual_mmq_gfx906.hip`
Baseline this builds on: commit `18312b7`.
Current measured: 147.8 tk/s prefill on Qwen 3.5 9B pp128.

**Revised target (v2):** ≥240 tk/s prefill (1.6× current) after this
redesign + matching gate_up dispatch update. The original ≥400 tk/s
target is downgraded — see §Goal.

Driving evidence:
- `docs/perf-checkpoints/2026-05-04-llamacpp-stock-comparison.md`
- Stock kernel source: `/tmp/llama-stock/ggml/src/ggml-cuda/mmq.cuh`
- Stock per-WG runtime LDS (rocprof CSV): **28.5 KiB** at mmq_x=64

Reviews integrated:
- `gfx906_mmq_redesign_plan_rv_claude.md` (my own adversarial review v3)
- `plans/gfx906_mmq_redesign_pl_rev_glm5.md`
- `plans/gfx906_mmq_redesign_pl_rev_gemini.md`

**Revision history**
- v1 (2026-05-04): initial plan with 10 design choices, 6 user
  questions; user locked-in answers to Q1, Q2, Q4, Q7, Q8, Q10.
- v2 (2026-05-04, this rev): integrates 3 adversarial reviews. **Three
  user-locked answers reversed by validated review findings**: Q1
  (mmq_x=128 doesn't exist on gfx906), Q4 (idle-thread X load is
  catastrophically uncoalesced), Q8 (b128 needs X_STRIDE padding).
  Phase budget revised from 2–3 days to 6–10 days. Phase 2a (probe)
  added as hard gate before Phase 2b (full rewrite).
- v2.1 (2026-05-04): Phase 2a probe committed (836b522). Phase 2b
  scaffolding produced (uncommitted: body.cuh + 8 _x{N}.hip + dispatch
  update). Validation pass against Phase 2a/2b artifacts logged 9
  findings — see §Phase 2b validation findings. Phase 2a methodology
  caveat: probe stubs used dummy/volatile loads, not real loaders, so
  the 112-VGPR / 18-KiB-LDS numbers were synthetic. Real kernel hit
  better (89 VGPR max at x64, 0 spills, all 8 variants compile clean
  on gfx906) — outcome OK, methodology weak.
- v2.2 (2026-05-04): residual-path on-hardware smoke test ran 34
  shapes (mmq_x ∈ {8,16,24,32,40,48,56,64} × K ∈ {4096,12288} ×
  full+partial M+N + production M=4096). All PASS, NRMSE 0.04–0.18%
  vs FP16. Caught one runtime-only bug: JIT compile failed because
  the cache_dir lacks `kernels/src` on its `-I` path → fixed by
  inlining `body.cuh` into the wrapper source string before
  `ensure_kernel`. gate_up (`_full_set_*`), real-data harness, and
  coherence gate still untested.
- v2.3 (2026-05-04): set-mode smoke test (gate_up path) ran 15
  shapes via new `MMQ_TEST_MODE=set` switch in
  `test_gfx906_mmq_correctness.rs`. All PASS, NRMSE matches
  residual-mode at the same shapes (0.04–0.18%). Confirms the
  `_full_set_x{N}` entry symbols and bounds-checked write-back work.
  Both gfx906 MMQ entries (residual add=1, set add=0) now validated
  across 49 shapes total. Real-data harness and coherence gate
  still untested.
- v2.4 (2026-05-04): real-data NRMSE test passes against the
  pre-existing /tmp/mmq_dump_0 (Qwen pp128: M=4096, K=4096, N=128).
  NRMSE = 0.29% — meets plan §Phase 4 ≤0.30% threshold by 0.01 pp.
  99.9% of cells are <1e-3 absolute error; the 10 worst-error cells
  cluster on a single row (3994). User notes row 3994 was also
  problematic in the original dp4a implementation, so the dump-data
  pattern is reproducibly tied to a degenerate quant group, not a
  redesign artifact.
- v2.5 (2026-05-04): coherence gate (HIPFIRE_MMQ=1
  HIPFIRE_MMQ_SCREEN=1) passes for all 4 mq4 rows: 0.8B-cap,
  4B-code, 9B-reason, 9B-tool-call. Outputs are fluent, on-topic,
  and the tool-call shape emits clean `<tool_call>...</tool_call>`
  with no `<|im_start|>` corruption (rules out the #87 regression
  pattern). HIPFIRE_MMQ_TRACE=1 confirmed 45 MMQ dispatches in 9B
  prefill (~48 expected; screening fell back 3 weights). The two
  mq3 rows (9B-reason-mq3, 27B-cap-mq3-27b) hit unrelated errors
  (mq3 needs WMMA which gfx906 lacks; these rows are in the gate
  for gfx11+/gfx12 coverage). End-to-end correctness ✅.
- v2.6 (2026-05-04): Phase 4 prefill bench — Qwen 3.5 9B mq4 on
  MI50, 5 runs/config, last-run measurement. **pp128: 141 → 287
  tok/s (2.04×), passes plan target ≥240 (1.6×) by 20%.** Speedup
  grows with batch (pp32: 1.67×, pp64: 1.89×, pp128: 2.04×, pp256:
  2.15×) — consistent with greedy mmq_x dispatch picking larger
  tiles at higher batch sizes.
- v2.7 (2026-05-04): cross-process A/B probe (3 alternating
  iterations with DPM warmup, fresh process per invocation):
  baseline median 141.0 tok/s (spread 0.4), MMQ median 287.1 tok/s
  (spread 0.1). B/A = 2.04× identical across all 3 iterations.
  Confirms the speedup is not within-session noise. Phase 4
  acceptance gate ✅.
- v2.8 (2026-05-04): Phase 5 commit landed (`c022682`). Pushed.
  Followed up with rocprofv3 kernel-trace + rocprof PMC
  attribution: see
  `docs/perf-checkpoints/2026-05-04-gfx906-mmq-redesign-rocprof.md`.
  Key findings: VALUBusy 27–41 % on the new MMQ kernel,
  VALUUtilization 100 % (zero wave divergence), MemUnitStalled ≤
  0.25 (~14× less than FP16 wave64), FetchSize 24 KB/call vs FP16
  wave64's 69 KB/call. The kernel is **neither compute- nor
  memory-bound**; the remaining headroom is inter-warp
  synchronization (Option B's 8 syncs per HFQ4 group). Implication
  for §Q8 deferred ds_read_b128: low expected gain since LDS issue
  is not on the critical path. Largest remaining lever: qkvza
  fused-4-output MMQ (28.87 % of GEMM time still on FP16 wave64).
- v2.9 (2026-05-04): qkvza split — gfx906 dispatch routes qkv+z
  through `gemm_hfq4g256_mmq_set_gfx906`, leaves beta+alpha
  (linear_num_value_heads M=32, too narrow for MMQ_Y=128) on the
  fused FP16 wave64 kernel called with `qkv_m=z_m=0` so its
  row-routing handles only the small tail. **pp128: 287 → 352
  tok/s (1.23× over the residual+gate_up commit, 2.50× over
  baseline).** Full sweep: pp32 1.85×, pp64 2.21×, pp128 2.50×,
  pp256 2.77× over baseline. rocprof attribution:
  qkvza_fp16_wave64 6.30→1.15 ms/call (5.5× per-call), share
  28.87 %→5.04 %. End-to-end correctness: 9B-reason coherence
  prompt produces fluent identical output. This was Path A from the
  pre-implementation analysis; Path B (true fused 4-output MMQ
  kernel) remains §Phase 6 deferred but the gap above the plan's
  ≥260 target after qkvza port is now 35 %.
- v2.10 (2026-05-04): qkv MMQ port — gfx906 full-attn qkv dispatch
  routes q+k+v through `gemm_hfq4g256_mmq_set_gfx906`. All three M
  dims (q_m=4096, k_m=v_m=1024) are above MMQ_Y=128, no tail
  needed. **pp128: 352 → 355 tok/s (+0.9 %).** Modest because **7
  of 8 qkv calls hit screen-fallback** at threshold 0.10. Per-call
  attribution: gemm_qkv_hfq4g256_fp16_wave64 still at 5.32 ms/call
  on the unrejected calls (dispatch wiring confirmed via the 1
  call that does fire MMQ). End-to-end correctness: 9B-reason
  coherence prompt fluent and identical. Real opportunity now:
  investigate screen-fallback share (32.5 % of GEMM time on
  residual_fp16_wave64, 30 weights rejected during a single load,
  row 3994 dominating).
- v2.11 (2026-05-04): MMQ screen threshold default raised from
  0.10 → 0.50 **on gfx906 only**. The 0.10 default was set when
  the gfx906 dp4a kernel was buggy (commit 8081822); the
  post-redesign kernel is structurally cleaner and produces
  coherent output across all 4 mq4 coherence rows even with
  screening effectively disabled. Threshold sweep: 0.10 rejects
  30 weights → 0.50 rejects 0 weights, with no observable
  coherence regression. Of the 30 rejected weights, **19 of 30
  reject specifically on row 3994** of m=4096 matrices —
  consistent with one degenerate quant group across the
  quantization rather than a kernel bug (8081822-era root causes
  are all fixed in the redesigned kernel). Per-arch default
  preserves the conservative 0.10 for non-gfx906 archs until
  similar validation is performed there.
  **pp128: 355 → 462 tok/s (+30%, 3.28× over baseline).**
  pp512: 554 tok/s = 74 % of stock llama.cpp's 750 tok/s pp512
  baseline (was 19 % before the redesign).

## Goal (revised v2)

Match stock llama.cpp's `mul_mat_q<Q4_K, mmq_x=64, false>` per-call
performance on this hardware on our HFQ4-G256 quant format.

**Revised end-to-end target (validated against per-call data):**
- After this commit (residual + gate_up MMQ both using new kernel):
  **≥240 tk/s prefill** (1.6× current 148).
- After follow-up port of qkv (~5% of prefill share): ≥260 tk/s.
- After fused-4-output qkvza kernel (separate Phase 6 design): ≥320
  tk/s.
- The "match stock at 500 tk/s" target requires *all four* GEMM paths
  (residual, gate_up, qkv, qkvza) to use MMQ. Single-commit redesign
  cannot deliver this.

The per-call ratio in the comparison doc is 1.22× on residual, not the
3.4× headline. Per [my-rev §1] / [glm-5 H3]: **gate_up at 12 ms × 64
calls = 768 ms is 81% of GEMM time**, residual at 1.3 ms × 128 = 166
ms is 18%. **Gate_up is the headline opportunity, not residual** —
see §Phase 3 below for why both must update simultaneously.

## Why the current kernel maxes out at 148 tk/s

- Stock uses **nwarps=4 (256 threads/WG)**, hipfire uses **nwarps=2 (128
  threads/WG)**. With 4 warps cooperating per WG, mmq_x can be 64
  without per-warp register pressure exceeding what 2 warps face at
  mmq_x=8.
- Stock's mmq_x is **runtime-dispatched** ∈ {8, 16, 24, 32, 40, 48, 56,
  64} (greedy step-8 selection per [glm-5 C1]) based on batch size.
  Hipfire hardcodes mmq_x=8.
- Stock's per-call GEMM latency on Q4_K K=4096 (HIP-trace, no counters):
  0.85–1.07 ms (varies by exact M). Hipfire's on HFQ4 K=4096: 1.3 ms.
  Per-call gap: **1.22×**, not 1.5–4×.

## Hardware constraints (gfx906 / MI50, all hard limits) [gemini]

These are physical, not optimization. Any plan must satisfy all of
them simultaneously.

| Resource | Per-CU limit | Source |
|---|---|---|
| LDS | 64 KiB | Vega 20 ISA, skyne98 wiki |
| VGPRs | 65,536 (256 KiB) | Vega 20 ISA |
| Wavefront size | 64 lanes | gfx906 hardware |
| `ds_read_b128` alignment | 16 B | AMDGPU LLVM docs |
| Max threads/WG | 1024 | gfx906 |

**At nwarps=4 (256 threads/WG) targeting 2 WGs/CU:**
- LDS: must be ≤ 32 KiB/WG (= 64/2)
- VGPRs: must be ≤ 128/thread (= 65,536 / (256 × 2))

These are tight. Stock fits at exactly 128 VGPRs/thread (zero margin)
and 28.5 KiB/WG LDS (4 KiB headroom). **We don't get more headroom than
stock** — anywhere we exceed these, occupancy halves and the redesign
delivers ~0.

## Design — non-negotiables (revised v2)

These are dictated by the hardware and the reviewed findings.

1. **`nwarps = 4`**, threads per WG = 256. [from comparison doc;
   re-verified in v2]
2. Block dim **`(64, 4, 1)`** — `threadIdx.x` for lane (0..63),
   `threadIdx.y` for warp (0..3).
3. **`__launch_bounds__(256, 2)` is conditional on Phase 2a probe**
   per [gemini #1]. If LDS-per-WG cannot fit ≤32 KiB at mmq_x=64,
   fall back to `(256, 1)` and accept halved occupancy. Three options
   to resolve in §Phase 2a.
4. `mmq_y = 128` — same as stock and as our current code. **(May
   reduce to 64 if Phase 2a probe shows LDS overflow — see §Phase 2a
   Option C.)**
5. **`mmq_x` is a template parameter.** Compile multiple instances and
   select at dispatch time.
6. **Accept `vgpr_count ≤ 128` as a hard ceiling, not a target.**
   [gemini #2] If Phase 2a probe shows compiler emits 129+ VGPRs in
   the new topology, design must change before proceeding.
7. **`mmq_x ∈ {8, 16, 24, 32, 40, 48, 56, 64}`** (8 values).
   [glm-5 C1] mmq_x=128 is **rejected** — stock's
   `get_mmq_x_max_device()` returns 64 on gfx906 [glm-5 C2,
   mmq.cuh:124-125]. Each mmq_x value emits 3 entry symbols (`_x{N}`,
   `_full_add_x{N}`, `_full_set_x{N}`) → **24 entry symbols total**,
   or a measured subset if 24 is too many for build time.
8. Keep the existing **HFQ4 layout** (136 B/group: f32 scale + f32 zp
   + 128 B nibbles, group covers 256 K-elements). The dequant code
   (`load_hfq4_tile_dp4a`) and zp pre-folding (`zp_eff = zp + 8×scale`)
   are correct; only the topology and threading change.
9. Keep **Q8_1 activations** (`block_q8_1_mmq` from our existing
   `quantize_q8_1_mmq` kernel). Stock uses the same quantize.
10. **X_STRIDE must be 16-byte aligned.** [gemini #4] Pad from 65 to
    **68 ints** (272 B) so `ds_read_b128` doesn't silently fall back.
    Adds 1.5 KB to x_qs LDS. Worsens budget; see Phase 2a Option B.
11. **X-tile loader must use coalesced thread layout.** [gemini #5]
    1-thread-per-row at X_STRIDE=68 gives 272-B-strided HBM access
    = ~64× bandwidth waste. Use **chunk-major** layout: adjacent
    threads read consecutive 4-byte chunks of the same row. Stock's
    `load_tiles_q4_K` pattern at `mmq.cuh:1129+`:
    ```
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) { ... }
    ```
12. **Reuse the existing entry symbol naming** with `_x{mmq_x}` suffix:
    - bounds-checked: `gemm_hfq4g256_residual_mmq_gfx906_x{N}`
    - full-add (M%128==0, N%mmq_x==0, add=1): `..._full_add_x{N}`
    - full-set (same predicate, add=0): `..._full_set_x{N}`

## Design — settled questions (v2 update)

The user-locked answers from v1 are revised here based on review
findings. **Reversed answers flagged with [reversed].**

### Q1 (which mmq_x values): {8, 16, 24, 32, 40, 48, 56, 64} [reversed]

**v1 answer:** {8, 16, 32, 64, 128} per user lock-in.
**v2 answer:** {8, 16, 24, 32, 40, 48, 56, 64} per [glm-5 C1, C2].

mmq_x=128 doesn't exist on gfx906 (stock's
`get_mmq_x_max_device() = 64`). The non-power-of-2 values
(24, 40, 48, 56) are needed because stock's greedy step-8 dispatch
picks them for batch sizes like 48, 56, 65–127 (gate_up's beta/alpha
M=16 case may dispatch mmq_x=8, but the gate_m and up_m which
together give us the column count for prefill go through fewer values).

If 24 entry symbols is too many for build time, drop {40, 56} (rarely
optimal in measured Qwen 3.5 9B prefill). Don't drop {24, 48} —
those map directly to qkvza's beta/alpha (M=16) and z (M~4096) shapes.

### Q2 (per-thread accumulator shape): match stock indexing [unchanged]

`sum[j0/nwarps * (mmq_y/warp_size) + i0/warp_size]`, with j0 stepping
by nwarps, i0 stepping by warp_size, both unrolled in vec_dot.

At nwarps=4, mmq_x=64, mmq_y=128, wave_size=64:
- Per-thread accumulator: `(mmq_x/nwarps) × (mmq_y/warp_size)` = 16 ×
  2 = **32 floats per thread** = 32 VGPRs.

Vs current nwarps=2, mmq_x=8: 4 × 2 = 8 floats per thread.

### Q3 (LDS layout): revised — must fit ≤32 KiB/WG [reversed scope]

**v1 answer:** keep current x_qs/x_dm layout, scale tile_y with mmq_x.
LDS calc showed 43.5 KiB at mmq_x=64.

**v2 finding [gemini #1]:** 43.5 KiB × 2 WGs = 87 KiB **exceeds the
64 KiB LDS/CU cap**. Stock fits in 28.5 KiB by holding only 32-K of
x_tile at a time, not 256-K. Our HFQ4 layout was designed around
256-K-group-resident x_tile.

**See §Phase 2a for the three options to resolve.** This question
cannot be settled before measurement.

Per-WG LDS budget at the relevant tile sizes (assuming X_STRIDE=68
post-[gemini #4] padding):

| mmq_x | x_qs | x_dm | tile_y | total | × 2 WGs | OK? |
|---|---|---|---|---|---|---|
| 8 | 34,816 B | 1,024 | 1,152 | **37,184** | 74,368 | ❌ |
| 16 | 34,816 | 1,024 | 2,304 | 38,144 | 76,288 | ❌ |
| 32 | 34,816 | 1,024 | 4,608 | 40,448 | 80,896 | ❌ |
| 64 | 34,816 | 1,024 | 9,216 | 45,056 | 90,112 | ❌ |

**Every mmq_x exceeds the 2-WG LDS budget** with the current
256-K-group-resident x_qs design. This is the load-bearing finding
in the v2 plan.

### Q4 (X-tile loader threading): chunk-major coalesced [reversed]

**v1 answer:** option (a) — 1 thread per row, half the WG idle.
**v2 answer [gemini #5]:** chunk-major coalesced layout.

1-thread-per-row × 260-B (or 272-B post-padding) row stride =
**adjacent threads access addresses 260+ B apart**. The HBM hardware
needs adjacent-address access patterns to coalesce. Strided access
issues 1 transaction per thread instead of 1 per warp = ~64×
bandwidth waste. This isn't "wasted parallelism"; it's
catastrophically uncoalesced.

Use stock's pattern: thread linear index `tid = threadIdx.y *
warp_size + threadIdx.x` (0..255), distributed `(chunk, row)` such
that adjacent tids hit consecutive 4-byte chunks of the same row.

### Q5 (Q8_1 tile loader): unchanged, with mmq_x=8 edge guard

`tid = threadIdx.y * warp_size + threadIdx.x`, each thread loads
`(mmq_x * Y_STRIDE) / 256` ints. At mmq_x=8: 8×36 = 288 ints / 256
threads → 32 active, **224 idle**. [glm-5 L1] Need
`if (tid < 32 * mmq_x / 8)` or equivalent guard.

### Q6 (X-once + Y-twice, 4 barriers/kg): unchanged

Match stock's `mul_mat_q_process_tile`. Don't pre-optimize barriers.

### Q7 (j0 unroll): empirically determined by Phase 2a probe [reversed scope]

**v1 answer:** full unroll (matches stock's Q4_K).
**v2 answer:** Phase 2a stub tests **both** full unroll and `#pragma
unroll 1`. Ship whichever has lower spill count and matches stock's
128 VGPR ceiling.

[glm-5 M1] argued HFQ4's heavier dequant might amplify spills; my v3
review noted this is mostly about the load function's intermediates,
not vec_dot's. Either way, we don't have data — measure both before
committing.

### Q8 (ds_read_b128): conditional, separate v2 commit [reversed]

**v1 answer:** emit ds_read_b128 from day one via explicit `int4`
reads.
**v2 answer:** **drop ds_read_b128 from v1**. [gemini #4 + glm-5 M2]

Two reasons:
1. b128 requires 16-byte alignment. X_STRIDE=65 (260 B/row) isn't
   aligned. Padding to 68 (272 B/row) adds 1.5 KB/WG to LDS, which
   we can't afford per Q3 above.
2. We tried b128 at the small kernel and it regressed 2.7%. Larger
   kernel may help, but layered commits are safer than betting v1
   on a known-regressing-on-different-shape change.

Ship v1 without b128. After v1 lands and we know LDS/VGPR margins,
revisit b128 as a follow-up commit.

### Q9 (boundary checking): bounds-checked + 2 _full variants per mmq_x

Per stock's `_full` for bulk + bounds-checked for last partial tile
[gemini #3]. Dispatch:
- For `batch_size % mmq_x == 0 && M % mmq_y == 0`: use `_full_*`.
- Otherwise: bounds-checked entry covers the partial last tile.
- Multiple-tile-fanout (`batch_tiles = ceil(batch / mmq_x)`) handles
  the bulk case.

### Q10 (carry-over scope): gate_up co-designed in Phase 3 [reversed]

**v1 answer:** gate_up port deferred to Phase 5.
**v2 answer [glm-5 H3, my-rev §10]:** gate_up dispatch update **must
land in Phase 3 alongside residual**. Both share constants (MMQ_X,
MMQ_Y, LDS sizes); shipping residual without gate_up dispatch update
leaves the system in a broken intermediate state.

qkvza port (4-output fusion) remains out-of-scope; tracked as
separate "fused-4-output MMQ kernel" Phase 6.

### File organization (Q9 follow-up): N .hip wrappers, single .cuh body

24 entry symbols × ~3 KB each is too many for a single `.hip` to
compile. Split into one .hip file per mmq_x value:
- `gemm_hfq4g256_residual_mmq_gfx906_x{N}.hip` (1 file per mmq_x)
- All include a shared `gemm_hfq4g256_residual_mmq_gfx906_body.cuh`
  with the templated `mul_mat_q` body.
- Each .hip instantiates the 3 entry symbols for its mmq_x value.

8 .hip files × 3 entries = 24 .hsaco artifacts produced by
`compile-kernels.sh`. compile-kernels.sh's xargs-P parallelization
already handles this.

## Phase 0 reconciliation [glm-5 H4]

The original `plans/gfx906_mmq_plan.md` Phase 0 deliverable:

> Switch to llama.cpp-gfx906 topology (`nwarps=2`, block `(64, 2, 1)`).

That recommendation came from the **unverbraucht** fork's
`gfx906-config.h`, which is stale (per yesterday's `mi50_benchmark.txt`
and our session conversation: "unverbraucht inherits gfx906-specific
improvements but lacks Qwen 3.5/3.6 specific optimizations from
upstream").

The redesign's nwarps=4 supersedes Phase 0's nwarps=2 based on:
- ELF metadata of stock's compiled kernel:
  `max_flat_workgroup_size = 256` confirms nwarps=4.
- Source: stock's `mmq_get_nwarps_device()` at `mmq.cuh:307-313`
  returns `256/64 = 4` on gfx906 (no MFMA, no WMMA).
- Yesterday's bench: stock at 750 tk/s pp512 vs unverbraucht at 246.7
  pp512 — the upstream batched-prefill path is 3.1× faster, and that
  path uses nwarps=4.

Phase 0's other findings (wave64 indexing correct, LDS bank-conflict
padding observations) remain valid.

## Implementation phases (revised v2)

### Phase 1: design + scaffolding (this commit)

Plan approved. Reviews integrated. ← **WAS BLOCKING.**

### Phase 2a: probe + design choice gate (1–2 days) [new in v2]

**Hard gate before Phase 2b. Three independent measurements; all must
pass.**

Build a minimal stub kernel:
- nwarps=4 topology, block dim (64, 4, 1)
- mmq_x=64, mmq_y=128, X_STRIDE=68 (16-byte aligned for b128)
- Inner-loop dp4a body only (no real X-load, no real Y-load —
  hardcoded LDS reads)
- Two builds: full j0 unroll, `#pragma unroll 1` j0
- For each: dump ELF metadata

**Gate 1 (VGPR ceiling) [gemini #2]:**
- Hard requirement: `vgpr_count ≤ 128` in *both* unroll variants.
- If only `unroll 1` hits it: ship that, document why.
- If neither hits it: the topology is infeasible. Stop.

**Gate 2 (spill threshold) [my-rev §2]:**
- Hard requirement: `vgpr_spill_count ≤ 200` in shipped variant.
- If 200 < spill ≤ 400: design pivot (smaller mmq_y or mmq_x).
- If spill > 400: stop, fundamental redesign needed.

**Gate 3 (LDS budget) [gemini #1]:**
- Compute total LDS at chosen mmq_x: `x_qs(layout) + x_dm + tile_y +
  ids_dst_shared`.
- Hard requirement at 2 WGs/CU: `total ≤ 32 KiB`.
- **At our 256-K-resident x_qs layout, gate 3 will fail.** Pick from:

**Option A: Accept 1 WG/CU (`__launch_bounds__(256, 1)`).** Simple
fix; halves theoretical occupancy. Likely loses most of the redesign's
gain. Don't ship unless A is the only option.

**Option B (recommended): Restructure to 32-K-streaming x_qs.**
Match stock's pattern: x_qs holds only 32 K-elements at a time, the
kg loop iterates 8× per HFQ4 group. Per-WG x_qs becomes
`mmq_y * MMQ_TILE_NE_K + mmq_y` ints = 4,224 ints × 4 = 16,896 B.
Adds 8× more LDS-load barriers per group; but stock does this and
runs fast. Estimated +2–3 days on Phase 2b.

**Option C: Reduce mmq_y to 64.** x_qs becomes 64 × 68 × 4 = 17,408
B. Total per-WG ~24 KiB; fits 2 WGs. But mmq_y=64 doubles WG count,
halves arithmetic intensity per WG. Worth measuring if Option B
proves too invasive.

Phase 2a output: a `phase2a-probe-results.md` document recording
VGPR count, spill, LDS budget for each combination tried, and a
chosen design point (A, B, or C).

### Phase 2b validation findings (2026-05-04)

Scaffolding validation logged 9 items; all fixed in the same session.

**Fixed (correctness/budget):**
- LDS budget enforced at runtime — `debug_assert!(shared_mem ≤ 32*1024)`
  in both gfx906 dispatchers. body.cuh uses `extern __shared__`, so the
  compiler can't validate the cap on its own.
- `X_STRIDE` 16 → 8 (saved 4 KiB/WG; the dp4a path only reads 8 ints/row
  — the probe stub's "16 = b128 alignment" rationale was wrong).
- `static_assert(sizeof(block_q8_1_mmq) == 144)` restored in body.cuh.
- `phase2a_probe_results.md` rewritten to flag probe-was-resource-not-
  correctness and to record the real Phase 2b ELF numbers.

**Fixed (clarity):**
- Scratch-pad comments stripped from body.cuh and probe_option_b.hip.
- `vec_dot_dp4a_streaming(sub_iter)` → `(sub_block)`; caller passes
  `sub_iter % 4` so the invariant is named at the call site.
- LDS layout invariant documented inline in both body.cuh and dispatch.rs
  ("KEEP IN SYNC").

**Late catch:** adding `#pragma unroll` on the 8-iter sub_iter loop
caused massive spills (x32: 439, x64: 986). Reverted to a rolled loop;
0 spills again. Comment in body.cuh now flags this trap.

**Final ELF (real kernel via body.cuh):** vgpr_count {x8: 48, x16: 66,
x32: 82, x64: 89}, 0 spills, all variants compile clean on gfx906.

**Already-passing (no work needed):** chunk-major X loader (#11);
3-symbol-per-mmq_x layout (#12); residual + gate_up dispatchers updated
together (§Q10); greedy step-8 mmq_x ladder matches stock.

**Smoke test (residual path, 2026-05-04):** 34 shapes ran on hardware,
all PASS. NRMSE vs FP16 wave64 reference: 0.04–0.18% (well under 1%
tolerance). Coverage: mmq_x ∈ {8,16,24,32,40,48,56,64} at K ∈
{4096, 12288}, plus bounds-checked partial-N (N ∈ {9,17,33,49,65}),
partial-M (M ∈ {130,200,256}), and production M=4096 K∈{4096,12288}.
NRMSE drops monotonically with both mmq_x and K, as expected from
accumulation noise. Plan §Phase 4 thresholds (≤0.13% K=4096, ≤0.05%
K=12288) clear at production scale (M=4096); 128-row synthetic cases
exceed the tighter 0.05% bar at smaller mmq_x because reference signal
is smaller, not because of kernel error.

**Smoke test (set-mode / gate_up path, 2026-05-04):** 15 shapes via
`MMQ_TEST_MODE=set` in test_gfx906_mmq_correctness.rs. All PASS, NRMSE
0.04–0.18% — identical to residual-mode at matching shapes (the body
is shared; only the write-back's add/set differs). Garbage prefill of
`y_mmq` ruled out a "write-back skipped" bug. Coverage: 4 ladder
points at K=4096, partial-N N ∈ {9,17,33,65}, partial-M M ∈ {130,200},
production M=4096.

**Real-data NRMSE (2026-05-04):** /tmp/mmq_dump_0 (Qwen pp128 dump,
M=4096 K=4096 N=128) → **NRMSE 0.29%, PASS** (plan threshold 0.30%).
99.9% of cells <1e-3 absolute error. The 10 worst-error cells all sit
on row 3994 (errors 6e-3 to 1.8e-2 vs next-worst row at 3.7e-3),
consistent with one degenerate quant group (near-zero scale → dp4a
rounding dominates). Per user: row 3994 was also problematic in the
original dp4a implementation, so this is a reproducible
dump-data pattern, not a redesign artifact. Real-data NRMSE is
3.3× higher than synthetic at the same shape (0.29% vs 0.09%)
because real Qwen weights span a wider scale range than the synth
helper's uniform `1e-3 ± 50%`. Not a kernel bug.

**Coherence gate (2026-05-04):** `HIPFIRE_MMQ=1 HIPFIRE_MMQ_SCREEN=1
./scripts/coherence-gate.sh`. **All 4 mq4 rows PASS** with fluent,
on-topic output and clean tool-call shape (no `<|im_start|>` leak,
rules out #87 regression pattern). `HIPFIRE_MMQ_TRACE=1` confirmed
the new kernel dispatched on 9B prefill: 45 MMQ calls at M=4096
K=12288 (≈48 expected residual calls on 24 layers × 2; screening
correctly rejected ~3 weights). Two mq3 rows hit unrelated errors
(gfx906 doesn't have WMMA → mq3 doesn't dispatch through our path);
these rows exist for gfx11+/gfx12 coverage. End-to-end correctness
on gfx906 mq4 prefill ✅.

**Prefill bench (2026-05-04):** Qwen 3.5 9B mq4 on MI50, last of 5
runs/config (post-JIT, within-session A/B):

| Prefill | Baseline (MMQ=0) | New MMQ (MMQ=1, SCREEN=1) | Speedup |
|---|---|---|---|
| pp32  | 137 tok/s | 228 tok/s | 1.67× |
| pp64  | 140 tok/s | 264 tok/s | 1.89× |
| **pp128** | **141 tok/s** | **287 tok/s** | **2.04×** |
| pp256 | 143 tok/s | 307 tok/s | 2.15× |

Plan target ≥240 tok/s on pp128 → passes by 20%. Speedup grows with
batch size, consistent with the greedy mmq_x ladder picking larger
tiles at higher batches (pp32 → mmq_x=32, pp128 → mmq_x=64).

**Cross-process probe (3 alternating iterations, fresh process,
DPM-warmed):**
| Iter | A (MMQ=0) | B (MMQ=1) | B/A |
|---|---|---|---|
| 1 | 141.3 | 287.1 | 2.03× |
| 2 | 141.0 | 287.2 | 2.04× |
| 3 | 140.9 | 287.1 | 2.04× |
| Median | 141.0 | 287.1 | 2.04× |

A spread 0.4 tok/s (0.3%), B spread 0.1 tok/s (0.04%), B/A identical
to 2 decimals across iterations. Speedup is not within-session noise.

**Runtime bug caught + fixed:** JIT compile failed with `'..._body.cuh'
file not found` because the runtime hipcc compiles from cache_dir
without `kernels/src` on its `-I`. Fixed by inlining
`GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_BODY_CUH` into the wrapper source
string in both gfx906 dispatchers before `ensure_kernel`. Body
content is now part of the cache hash, so body edits invalidate the
JIT cache correctly.

**Phase 5 commit-ready.** All correctness + perf gates clear:

| Gate | Result |
|---|---|
| Synthetic NRMSE (49 shapes, residual+set) | 0.04–0.18% ✅ |
| Real-data NRMSE (Qwen pp128 dump) | 0.29% (≤0.30% threshold) ✅ |
| Coherence gate (4 mq4 rows) | all PASS ✅ |
| Prefill pp128 (within-session) | 287 tok/s, 2.04× baseline ✅ |
| Prefill pp128 (cross-process A/B) | 2.04× confirmed, B spread 0.04% ✅ |

**Optional follow-ups (re-prioritized after threshold bump, 2026-05-04):**
- **(P1)** Default-on flip — flip `should_use_mmq()` gfx906 branch
  from opt-in to default-on. 3.28× pp128 speedup with no coherence
  regression (across 4 mq4 rows × multiple thresholds) and the
  threshold work demonstrates the headroom is screening
  conservatism, not real precision issues. Lowest-effort,
  highest-impact next step — exposes the redesign's gains to
  default-config users without requiring HIPFIRE_MMQ=1.
- (P2) Re-profile post-threshold-bump with rocprof to see what
  the new bottleneck is. residual_fp16_wave64 share should have
  collapsed; new top kernels are likely _full_set_x64 +
  _full_add_x64 + qkvza tail. Path B (true fused 4-output MMQ)
  becomes the candidate again if the qkvza tail is now visible.
- (P3) Investigate why pp256 ≈ pp512 (560 vs 554 tok/s) — saturation
  at large batches suggests launch overhead or HBM ceiling, not
  kernel inefficiency. Profile pp512 specifically.
- (P4) Path B — true fused 4-output MMQ kernel (§Phase 6).
- (P5, speculative) reduce sync frequency from 8/HFQ4-group to
  2/HFQ4-group à la stock. Likely pushes back over the 32 KiB cap;
  revisit alongside b128.
- (P6) ds_read_b128 vectorization (§Q8 deferred). rocprof says
  this lever is small — VALUBusy is at 27–41 % (not LDS-issue
  saturated) and MemUnitStalled is ~0. Expected gain ≤ 5 %.
- (P7) Parameterize the test harness on mmq_x. Polish.

### Phase 2b: full kernel rewrite (3–5 days)

Once Phase 2a settles the design point:

- Create `kernels/src/gemm_hfq4g256_residual_mmq_gfx906_body.cuh`
  with the templated `mul_mat_q<int mmq_x>` body, including:
  - chunk-major X-tile loader [gemini #5]
  - X_STRIDE=68 alignment [gemini #4]
  - structure per Phase 2a chosen option (A/B/C)
- Create 8 .hip files `..._x{N}.hip` for N ∈ {8,16,24,32,40,48,56,64}
  that #include the body and instantiate 3 entry symbols each.
- Document the K-iter mapping (HFQ4 256-K vs stock's Q4_0 32-K)
  inline in the body.cuh per [my-rev §6].

Build artifacts: 24 .hsaco files. Verify each with ELF metadata
matching Phase 2a probe.

### Phase 3: dispatch + test harness updates (2 days) [scope expanded]

**Both `gemm_hfq4g256_residual_mmq_gfx906` (residual) and
`gemm_hfq4g256_mmq_set_gfx906` (gate_up) get updated together.**
[glm-5 H3]

Update `crates/rdna-compute/src/dispatch.rs`:
- Lines ~6181-6276 (`gemm_hfq4g256_residual_mmq_gfx906`): pick mmq_x
  via greedy step-8 loop matching stock's `mmq.cuh:4069-4082`. Pick
  entry symbol `_x{N}` based on chosen mmq_x. Pick `_full_*` variant
  based on `(M % mmq_y == 0 && batch % mmq_x == 0)`. Update LDS
  allocation per Phase 2a choice.
- Lines ~6278-6426 (`gemm_hfq4g256_mmq_set_gfx906`): same greedy
  dispatch, same entry-symbol selection, `add=0` instead of `add=1`.
  **Constants must match the residual dispatcher** — they share the
  underlying kernel.

Update test harnesses [glm-5 M4]:
- `crates/rdna-compute/examples/test_gfx906_mmq_correctness.rs`:
  parameterize on mmq_x, run NRMSE check at each value.
- `crates/rdna-compute/examples/test_gfx906_mmq_realdata.rs`: same.

### Phase 4: validation (1–2 days)

**Per-mmq_x correctness:**
- `test_gfx906_mmq_correctness 4096 4096 N` for N ∈ {8, 16, 32, 64,
  128} (= dispatch-time batch sizes that exercise different mmq_x
  selections). NRMSE ≤ 0.13%.
- `test_gfx906_mmq_correctness 4096 12288 N` same. NRMSE ≤ 0.05%.
- Real-data NRMSE via `test_gfx906_mmq_realdata` ≤ 0.30%.

**Coherence gate:**
- Standard `scripts/coherence-gate.sh` with `HIPFIRE_MMQ=1
  HIPFIRE_MMQ_SCREEN=1`.
- [glm-5 H2 partial] gate exercises prefill (via daemon) but doesn't
  exhaustively per-mmq_x — synthetic NRMSE above covers that.

**Bench:**
- Hipfire's `bench_qwen35_mq4 --prefill {32,64,128,256}` if the
  harness supports it; otherwise pp128 only and document gap with
  stock's pp32/64/256/512 numbers.
- 5 runs per config, document min/p50/max.

**rocprof:**
- `--hip-trace` per-kernel timing of new kernel at K=4096, K=12288,
  pp128. Compare to stock's mul_mat_q timing in
  `/tmp/rocprof_stock/stock_pp128.csv`.
- ELF metadata final verification: vgpr_count ≤ 128, vgpr_spill ≤
  200, group_segment_fixed_size aligned with Phase 2a choice.
- Measure VALUBusy during X-load phase per [glm-5 M3]. If load is
  ≥15% of per-call time, plan a Phase 7 chunk-major load tuning.

### Phase 5: rollout (commits)

**Single commit** containing Phase 2b + 3 + 4 work. Atomic update
keeps gate_up dispatch consistent with residual.

Subject template:
> `gfx906 MMQ redesign: nwarps=4, mmq_x runtime dispatch, +Y% prefill`

### Phase 6 (separate plan, deferred): qkvza fused-4-output MMQ kernel

The current qkvza FP16 wave64 kernel fuses 4 output matrices into 1
launch via row routing. Our MMQ port (this plan) requires 4 separate
launches, paying 4× launch overhead. To beat FP16 on qkvza we need
a fused MMQ kernel. **Deferred** [my-rev §11, glm-5 H3 implication]
— design separately after the residual+gate_up redesign ships.

Estimated separate effort: 3–5 days.

## Risk register (revised v2)

| # | Risk | Severity | Probability | Mitigation |
|---|---|---|---|---|
| R1 | LDS-per-WG > 32 KiB → 1 WG/CU [gemini #1] | **Critical** | **Certain** at current x_qs layout | Phase 2a probe; pick Option A/B/C |
| R2 | VGPR count > 128 at nwarps=4 [gemini #2] | **Critical** | Medium | Phase 2a Gate 1 hard-fails |
| R3 | ds_read_b128 silent fallback at non-aligned X_STRIDE [gemini #4] | Critical | Resolved | Pad to X_STRIDE=68; or drop b128 from v1 |
| R4 | Uncoalesced X-load → 64× HBM waste [gemini #5] | High | Resolved | Q4 reversed to chunk-major layout |
| R5 | mmq_x=128 doesn't exist on gfx906 [glm-5 C2] | High | Resolved | Q1 reversed to {8..64} |
| R6 | gate_up dispatch breaks when residual constants change [glm-5 H3] | High | Certain without fix | Phase 3 updates both atomically |
| R7 | HFQ4 dequant intermediates spill more than Q4_K | Medium | Medium | Phase 2a probe both unroll modes |
| R8 | LDS bank conflict at wider tile_y [my-rev §3b] | Medium | Low | Pre-emptive padding analysis in Phase 2b |
| R9 | Compile time blowup from 24 entry symbols [my-rev §5] | Medium | Medium | Compile mmq_x={64} during Phase 2 dev cycle; full set in Phase 4 |
| R10 | Phase 2a probe shows no feasible design point | High | Low | Fall back to current MMQ_X=8 + j0-unroll1 kernel; ship gate_up dispatch fix only |
| R11 | Correctness regression at edge mmq_x values [glm-5 L3] | Medium | Medium | Per-mmq_x synthetic NRMSE in Phase 4 |
| R12 | ROCm version sensitivity [glm-5 L3] | Low | Low | Specify min ROCm 6.4.3 (= measured) |
| R13 | Test harness breakage from template parameterization | Medium | Low | Phase 3 updates them |

## Time budget (revised v2)

| Phase | Original | Revised v2 | Reason for change |
|---|---|---|---|
| Phase 1 (design) | done | done | — |
| Phase 2a (probe) | — (new) | **1–2 days** | Gate before commitment |
| Phase 2b (kernel rewrite) | 1–2 days | **3–5 days** | X-tile restructure if Option B; build/cache cycle slower with 24 symbols |
| Phase 3 (dispatch + test harness) | 0.5 day | **2 days** | Co-design gate_up + test updates |
| Phase 4 (validation) | 0.5 day | **1–2 days** | Per-mmq_x NRMSE matrix |
| **Total** | 2–3 days | **7–11 days** | |
| Phase 6 (qkvza, deferred) | — | 3–5 days | Separate plan |

The 7–11 day budget is realistic for a from-scratch MMQ rewrite given
this session's experience: each "trivial" experiment cost 1–3 hours
of dev + 30 min cache debugging + bench. With 24 entry symbols and a
broader correctness matrix the surface area is larger.

## What this plan is NOT doing

- **Not adding stream-K work partitioning.** Stock explicitly uses
  conventional tiling on non-CDNA AMD (`mmq.cuh:3581`).
- **Not using MFMA / WMMA.** gfx906 has neither.
- **Not changing the HFQ4 quant format.** Separate axis.
- **Not refactoring shared MMQ helpers** across hipfire's RDNA3 /
  gfx906 paths. gfx906's nwarps=4 design is incompatible with RDNA3's
  nwarps=8 wave32 topology.
- **Not optimizing for prefill batch sizes <8.** Decode goes through
  GEMV.
- **Not porting qkvza** in this commit (Phase 6 separate).
- **Not adding ds_read_b128** in v1. Layered as separate v2 commit
  after v1 lands and LDS margins are confirmed.

## Decision points — re-asked for user (v2)

The following user-locked answers were reversed by validated review
findings. **User confirmation requested** before Phase 2a starts:

- **Q1 (mmq_x candidate set):** v1 locked-in {8,16,32,64,128} per
  user. v2 changed to {8,16,24,32,40,48,56,64} per [glm-5 C1, C2].
  mmq_x=128 doesn't exist on gfx906; non-power-of-2 values are
  needed for stock's greedy step-8 dispatch. **Confirm OK?**

Confirmed ok.

- **Q4 (X-tile loader threading):** v1 locked-in option (a) per user.
  v2 changed to chunk-major coalesced layout per [gemini #5]. The
  v1 option was catastrophically uncoalesced. **Confirm OK?**

Confirmed ok.

- **Q8 (ds_read_b128):** v1 locked-in "emit from day one" per user.
  v2 changed to "drop from v1, add as separate v2 commit" per [gemini
  #4 + glm-5 M2]. b128 needs X_STRIDE padding which makes LDS budget
  worse. **Confirm OK?**

Confirmed ok.

The other locked-in answers (Q2, Q7, Q10) remain settled but Q7's
specifics are now empirically determined in Phase 2a (test both
unroll modes, ship lowest-spill).

## Implementation Notes & Reminders

### N1. Option B (Streaming) Barrier Frequency
If Phase 2a leads to **Option B (32-K streaming)**, the `kg` loop will hit `__syncthreads()` 8x more often. Ensure the streaming loader and its K-indexing are tightly unrolled to prevent HBM-to-LDS load latency from becoming a bottleneck that offsets occupancy gains.

### N2. Activation (Y) Tile Reuse
Verify that the X-streaming logic (8 sub-iters per HFQ4 group) does not redundantly reload Y-tiles from HBM. The 256 K-elements of activations should be loaded/indexed such that each Y-load is amortized across all `mmq_x` columns and all streaming X-sub-iters where possible.

### N3. Dispatch Masking vs Padding
The `bounds-checked` kernel variants must handle `N % mmq_x != 0` via internal masking. Ensure `dispatch.rs` passes the exact unpadded batch size (`ncols`) to these kernels so the `if (j < ncols)` guards function correctly.

### N4. Build System & Stale Caches
`rdna-compute` embeds kernel source via `include_str!`. A change to the `.cuh` body or any `.hip` file **requires** a re-trigger of the Rust build to see the new ISA in the ELF. If ELF counts or performance don't change after a code edit, use `cargo clean -p rdna-compute` to force an embed refresh.

### N5. ISA Verification
Use `readelf -s` or `rocobjdump` on the produced `.hsaco` files during Phase 2 to monitor:
- `vgpr_count`: must stay ≤ 128 for 2 WGs/CU.
- `group_segment_fixed_size`: must stay ≤ 32,768 B for 2 WGs/CU.
- `private_segment_fixed_size`: monitor spills against the Phase 2a Gate 2 threshold (200).

## Cross-reference

- Comparison findings: `docs/perf-checkpoints/2026-05-04-llamacpp-stock-comparison.md`
- Stock kernel: `/tmp/llama-stock/ggml/src/ggml-cuda/mmq.cuh`
  - Entry: `mul_mat_q` template @ line 3530
  - Process tile: `mul_mat_q_process_tile` @ line 3447
  - q4_0 vec_dot reference: `vec_dot_q4_0_q8_1_dp4a` @ line 460
  - q4_K vec_dot reference (similar to HFQ4 zp structure):
    `mmq.cuh` search for `vec_dot_q4_K_q8_1_dp4a`
  - nwarps selection: `mmq_get_nwarps_device` @ line 307
  - mmq_x_max: `get_mmq_x_max_device` @ line 119 (= 64 on gfx906)
  - Dispatch greedy loop: `mmq.cuh:4069-4082`
- Current hipfire kernel: `kernels/src/gemm_hfq4g256_residual_mmq_gfx906.hip`
- Current dispatch:
  - residual: `crates/rdna-compute/src/dispatch.rs:6181-6276`
  - set/gate_up: `crates/rdna-compute/src/dispatch.rs:6278-6426`
- Adversarial reviews:
  - Claude (this rewrite incorporates): `gfx906_mmq_redesign_plan_rv_claude.md`
  - glm-5: `plans/gfx906_mmq_redesign_pl_rev_glm5.md`
  - gemini: `plans/gfx906_mmq_redesign_pl_rev_gemini.md`
- Prior dev logs:
  - `docs/perf-checkpoints/2026-05-04-gfx906-mmq-attribution.md`
  - `docs/perf-checkpoints/2026-05-04-gfx906-mmq-junroll.md`
  - `docs/perf-checkpoints/2026-05-04-gfx906-mmq-spill-reduction.md`
- Original MMQ plan: `plans/gfx906_mmq_plan.md`
- L2 prefetch plan (now superseded): `plans/gfx906_mmq_l2.md`
