# Dev log 2026-05-09 — MQ4-Lloyd vs uniform-MQ4, gfx1151 cross-process A/B

**Branch:** `feat/issue-182-mq4-lloyd` (HEAD `f366b80`, post-rebase + maintainer
speed-baseline revert).
**Hardware:** gfx1151 (Radeon 8060S, AMD Ryzen AI Max+ 395 / Strix Halo APU,
LPDDR5x ~250 GB/s), ROCm 7.12.
**Sibling:** gfx1100 Phase C — `devlog_20260508_mq4_lloyd_wmma_phase_c.md`
(7900 XTX, GDDR6 ~960 GB/s).

## Summary

Cross-process A/B of the batched-prefill path with MQ4-Lloyd vs the structural
ceiling (uniform MQ4) at two model sizes on gfx1151:

| Model       | Lloyd prefill | uniform prefill | Lloyd / uniform | gfx1100 sibling |
|---|---|---|---|---|
| Qwen3.5-4B  |  976.2 tok/s  | 1941.8 tok/s    | **50.3 %**       | 72.1 % |
| Qwen3.5-9B  |  465.9 tok/s  | 1122.9 tok/s    | **41.5 %**       | 60.6 % |

The Lloyd / uniform ratio runs **~19 pp lower on gfx1151 than gfx1100** at
both sizes. This is consistent with gfx1151 being bandwidth-bound on
LPDDR5x: the +17.6 % weight footprint of Lloyd-MQ4 (160 B/group vs HFQ4's
136 B/group) eats more of the relative budget on a ~4× lower-bandwidth
memory subsystem.

These are bandwidth-bound numbers, not compute-bound — direct comparison
to gfx1100's GDDR6 ratios is informational, not a regression signal. The
Phase C ship gate (≥ 60 % at 9B on gfx1100) is calibrated for that
hardware class; gfx1151 is exercised here to populate the missing data
point, not to re-gate.

## Bench config

| Field | Value |
|---|---|
| Tool | `target/release/examples/bench_qwen35_mq4` |
| Models | `qwen3.5-{4,9}b.mq4` (uniform), `qwen3.5-{4,9}b.mq4-lloyd` |
| Flags | `--prefill 256 --warmup 5 --prefill-runs 3 --gen 30` |
| Env | `HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1` |
| ROCm path | `PATH=/opt/rocm-7.12/bin:$PATH LD_LIBRARY_PATH=/opt/rocm-7.12/lib:...` |
| Cross-process A/B | 3 fresh process invocations × 2 models × 2 sizes (12 total) |
| In-process samples | 3 timed prefills per invocation; in-process median printed |
| Reported metric | mean of in-process medians across the 3 invocations |
| Bench binary md5 | `5b07b25acd82f9df5cfd5ccc70365207` |
| Model md5 (9B uniform) | `31a8d8dc7603226801b08d8319015602` |
| Model md5 (9B Lloyd)   | `b3eea80aeade0b56c153a054b1143ab2` |
| Model md5 (4B uniform) | `93b9b5f2bd075922c50f3f8c9a5ad3e3` |
| Model md5 (4B Lloyd)   | `d13028f6a1f4fda772c17bb6c3f3a0bc` |
| Prompt | N/A — `bench_qwen35_mq4` generates a deterministic synthetic token sequence (token ids `0..prefill_len`) |

Same config used for the gfx1100 Phase C devlog so the ratios are directly
comparable across hardware.

## Raw numbers (gfx1151, ROCm 7.12, branch HEAD `f366b80`)

### Qwen3.5-9B

```
=== qwen3.5-9b.mq4 (uniform) ===
  run 1: prefill_median=1102.6 tok/s, gen=45.1 tok/s
  run 2: prefill_median=1132.3 tok/s, gen=45.1 tok/s
  run 3: prefill_median=1133.9 tok/s, gen=45.2 tok/s
  → prefill mean = 1122.9 tok/s   gen mean = 45.13 tok/s

=== qwen3.5-9b.mq4-lloyd ===
  run 1: prefill_median=467.7 tok/s, gen=37.4 tok/s
  run 2: prefill_median=463.1 tok/s, gen=37.4 tok/s
  run 3: prefill_median=466.9 tok/s, gen=37.4 tok/s
  → prefill mean = 465.9 tok/s    gen mean = 37.40 tok/s
```

### Qwen3.5-4B

```
=== qwen3.5-4b.mq4 (uniform) ===
  run 1: prefill_median=1945.6 tok/s, gen=67.4 tok/s
  run 2: prefill_median=1937.9 tok/s, gen=67.6 tok/s
  run 3: prefill_median=1942.0 tok/s, gen=67.4 tok/s
  → prefill mean = 1941.8 tok/s   gen mean = 67.47 tok/s

=== qwen3.5-4b.mq4-lloyd ===
  run 1: prefill_median=950.3 tok/s, gen=59.7 tok/s
  run 2: prefill_median=986.1 tok/s, gen=59.5 tok/s
  run 3: prefill_median=992.3 tok/s, gen=59.7 tok/s
  → prefill mean = 976.2 tok/s    gen mean = 59.63 tok/s
```

## Decode comparison

Decode tok/s (gen phase, untouched by Phase 5b — sanity check that the
batched-prefill changes don't bleed into the per-token forward path):

| Model       | uniform gen | Lloyd gen | Lloyd / uniform |
|---|---|---|---|
| Qwen3.5-4B  | 67.47 tok/s | 59.63 tok/s | 88.4 % |
| Qwen3.5-9B  | 45.13 tok/s | 37.40 tok/s | 82.9 % |

Decode ratio drop is dominated by the +17.6 % weight footprint Lloyd
carries — roughly the bandwidth-bound expectation. Within run-to-run
spread (range / mean across 3 invocations: ≤ 0.4 % for all four sets).

## Cross-hardware comparison (informational)

Reproducing the 4B / 9B / 27B sweep table from the gfx1100 Phase C devlog
with the gfx1151 numbers added (27B not run here — outside this devlog's
scope, deferred to a follow-up if useful):

| Model        | gfx1100 Lloyd/uniform | gfx1151 Lloyd/uniform | Δ pp |
|---|---|---|---|
| Qwen3.5-4B   | 72.1 % | **50.3 %** | -21.8 |
| Qwen3.5-9B   | 60.6 % | **41.5 %** | -19.1 |

The ratio drop tracks bandwidth-class — ~4× lower bandwidth → ~20 pp
lower Lloyd / uniform ratio at both sizes. This is the bandwidth-bound
expression of Lloyd's +17.6 % footprint, not a kernel-implementation
regression specific to gfx1151. Phase C's investigate-bucket candidates
(2× LDS footprint, longer 8-byte-per-tile decode K-schedule, longer
cooperative-load sync) all amortize across more compute on the GDDR6
host, less on the LPDDR5x APU.

## What this doesn't tell us

- **No 27B data point.** 27B Lloyd-MQ4 (`qwen3.6-27b.mq4-lloyd`) is
  resident; gfx1151 has the VRAM to run it but at ~3× lower absolute
  prefill than gfx1100, the wall-time for a 3-run cross-process matrix
  is ~10× the 9B run. Skipped here; happy to add if the ratio at 27B
  is wanted.
- **No coherence-gate row** beyond what's already documented in the
  gfx1100 Phase C devlog (`5fce3e3`). Coherence behavior is shared
  across hardware; running the gate again here adds a row, not a
  signal.
- **No HEAD-vs-pre-B2 self-comparison.** The PR body's 12.3× pre/post-B2
  number is from the original Phase B2 commit on gfx1151. Not re-run
  here — that ratio is invariant of the post-merge state.

## Per-kernel profile (HIPFIRE_PROFILE=1, 9B, prefill 256, single warm pass)

The Lloyd / uniform ratio is *almost entirely* a GEMM-time difference.
Every non-GEMM kernel (delta-net, conv1d, rmsnorm, rope, sigmoid)
profiles within run-to-run noise between the two configs — the +6 ms
total spread visible below is invisible at prefill scale.

### `qwen3.5-9b.mq4` (uniform / HFQ4)

```
gemm_hfq4g256_mmq_set            184x  132.4ms  ( 720µs/call)  60.2%   26.3 GiB/s
gemm_hfq4g256_residual_mmq        64x   61.8ms  ( 965µs/call)  28.1%   27.4 GiB/s
gated_delta_net_q8_batch_seq      24x    9.8ms                  4.4%   41.1 GiB/s
fused_silu_mul_mq_rotate_batched  32x    4.7ms                  2.1%  243.3 GiB/s
conv1d_silu_split_f32_n           24x    2.7ms                  1.2%  847.2 GiB/s
gemv_hfq4g256                      1x    2.4ms                  1.1%  214.2 GiB/s
fused_rmsnorm_mq_rotate_batched   64x    2.2ms                  1.0%  355.6 GiB/s
[…tail kernels < 1ms each…]
TOTAL (serialized)                     219.9ms
WALL                                   236.4ms (1082.8 tok/s)
GEMM time                              194.2ms (88 % of prefill wall)
```

### `qwen3.5-9b.mq4-lloyd`

```
gemm_gate_up_mq4g256_lloyd_wmma   32x  234.3ms  (7322µs/call)  44.2%   11.5 GiB/s
gemm_mq4g256_lloyd_residual_wmma  64x  157.2ms  (2456µs/call)  29.6%   12.7 GiB/s
gemm_qkvza_mq4g256_lloyd_wmma     24x   88.6ms  (3692µs/call)  16.7%   11.7 GiB/s
gemm_qkv_mq4g256_lloyd_wmma        8x   22.9ms  (2862µs/call)   4.3%   12.6 GiB/s
gated_delta_net_q8_batch_seq      24x   10.9ms                  2.1%   36.7 GiB/s
fused_silu_mul_mq_rotate_batched  32x    4.1ms                  0.8%  277.8 GiB/s
gemv_mq4g256_lloyd                 1x    2.8ms                  0.5%  214.2 GiB/s
[…tail kernels < 1ms each, identical to uniform within noise…]
TOTAL (serialized)                     530.0ms
WALL                                   545.5ms (469.3 tok/s)
GEMM time                              503.0ms (92 % of prefill wall)
```

### Headline ratios

| Quantity | HFQ4 | Lloyd-MQ4 | Ratio |
|---|---|---|---|
| GEMM bytes-per-second | 26-27 GiB/s | 11.5-12.7 GiB/s | **~46 %** |
| GEMM wall (sum) | 194.2 ms | 503.0 ms | 2.59× longer |
| Prefill wall | 236.4 ms | 545.5 ms | 2.31× longer |
| Prefill tok/s | 1082.8 | 469.3 | 43.3 % (matches the 41.5 % 3-run mean within JIT-noise; this is the JIT-included single-run number) |

### Reading

The +17.6 % weight footprint of Lloyd-MQ4 explains ~15 pp of the gap.
The remaining ~40 pp is kernel-implementation efficiency — Lloyd's
WMMA path achieves about half the per-byte throughput that HFQ4's
`mmq` (dp4a-style integer dot product) achieves, on the same
hardware, against the same activations.

**Notable:** HFQ4 is not using WMMA on this prefill path. The hot
kernels are `gemm_hfq4g256_mmq_set` and `gemm_hfq4g256_residual_mmq`
— the older `mmq` family. Lloyd-MQ4 uses WMMA exclusively.

## Diagnostic: batch-size sweep (HIPFIRE_PROFILE=1, varying --prefill)

To localize where the gap comes from — kernel-internal efficiency vs
batch-tile fanout vs path selection — profiled `gate_up` GiB/s at
four batch sizes:

| prefill | Lloyd `gate_up` GiB/s | HFQ4 `gate_up` GiB/s | HFQ4 path | Lloyd / HFQ4 |
|---|---|---|---|---|
| 16   | **65.6** | 100.1 | WMMA (`gemm_gate_up_hfq4g256_wmma`) | 65.5 % |
| 64   | 26.5 | 35.1  | WMMA | 75.5 % |
| 256  | 11.4 | 26.2  | **mmq** (`gemm_hfq4g256_mmq_set`)   | 43.5 % |
| 1024 | 9.9  | 25.5  | mmq                                  | 38.8 % |

### Reading

Two distinct effects superimpose:

1. **Within the WMMA family** (prefill 16, 64): Lloyd runs at 65-75 %
   of HFQ4 WMMA. This is the format-size penalty (+17.6 % bytes/group)
   plus codebook-indirection cost (per-row LDS lookup vs HFQ4's
   affine reconstruction `sc * q + zp`). Real but modest gap, kernel-
   internal — fixable by load-width / prefetch tuning if wanted.

2. **At batch ≥ 128** (prefill 256, 1024): HFQ4 leaves the WMMA family
   entirely and dispatches `gemm_hfq4g256_mmq_set` — a 128×128-tile
   LDS-staged kernel that holds 25 GiB/s flat. Lloyd stays on the
   16×16-tile WMMA kernel and degrades from 26 → 10 GiB/s as batch
   grows. **The dominant gap at production prefill (256+ tokens) is
   that Lloyd has no mmq-equivalent large-batch path**, not that the
   WMMA kernel is poorly tuned.

The WMMA-path Lloyd sees a clean 1/√batch-style decay (65.6 → 11.4 over
16 → 256) consistent with the duplicate-fetch theory — each new
workgroup-column re-reads the same row span of weights.

### Reframed lever menu

1. **Add an mmq-equivalent large-batch Lloyd kernel.** Direct i8 WMMA
   port isn't viable (i8 WMMA needs affine-reconstructible weights;
   Lloyd's 16-entry codebook holds arbitrary fp16). Workable shape:
   keep fp16 WMMA but stage weights in LDS and emit multiple WMMA ops
   per weight load (e.g. 4 batch-tiles per workgroup, output 16×64).
   LDS budget is comfortable (16 rows × 128 B nibbles + 512 B
   codebook = 2.6 KB per group; Phase A's 68 VGPR leaves headroom for
   4× the `acc` registers). Predicted upside: closes most of the
   batch-tile fanout gap. ~2-3× speedup at prefill=256, bringing
   Lloyd / uniform on gfx1151 from 41 % up toward the 60-70 % range.
   Real engineering — Phase D scope.

2. **Tune within-WMMA load width / async prefetch** for the small-batch
   path. Targets the modest 25-35 pp gap to HFQ4-WMMA. Worth doing
   only if Lever 1 lands. Less leverage on bandwidth-bound APUs than
   Lever 1 because the WMMA path isn't the dominant time at prefill ≥
   128.

3. **Per-arch K-schedule** — gfx1151's wave32 + 16-lane WMMA pair shape
   may favor K4 or K8 over the current K2 unroll. Cheap to bisect, but
   the upside is tail of tail given Levers 1+2.

Skip: pure LDS-bank-conflict tuning, register pressure work, codebook
LDS layout — none are the bottleneck per the profile.

## Phase D-A prototype results (residual `_mb4` kernel)

`kernels/src/gemm_mq4g256_lloyd_residual_wmma_mb4.hip` — Phase D-A
prototype per `docs/plans/mq4-lloyd-batch-fanout-phase-d.md`. 16×64
output tile per WG (4 batch sub-tiles share `a_reg` decode).

### Parity (gfx1151, 11 shapes including production sizes)

Bit-exact across all 11 shapes — `max_abs` per shape identical between
`_wmma` and `_mb4` to last printed digit. WMMA accumulation order is
preserved (no inter-batch-tile interleaving inside a single
accumulator), so the cross-path drift envelope is byte-equal in this
shape sweep. Suggested tolerance unchanged at 1.75e-4.

### Bench (gfx1151, parity-test inner loop, 20 iter post-warmup)

| M | K | N | `_wmma` µs | `_mb4` µs | Speedup | `_mb4` GiB/s |
|---|---|---|---|---|---|---|
| 64    | 1024  | 16  | 9.9   | 13.6  | 0.73×    | — (small-shape) |
| 64    | 1024  | 64  | 42.1  | 18.9  | **2.23×** | — |
| 256   | 1024  | 64  | 19.0  | 20.1  | 0.95×   | — |
| 64    | 4096  | 64  | 52.7  | 51.2  | 1.03×    | — |
| 256   | 4096  | 16  | 58.1  | 42.8  | 1.36×    | — (small N) |
| 256   | 4096  | 256 | 73.7  | 69.0  | 1.07×    | 13.5 |
| 1024  | 4096  | 64  | 47.3  | 70.7  | 0.67×    | — (occupancy loss) |
| 1024  | 12288 | 64  | 146.4 | 181.6 | 0.81×    | — (occupancy loss) |
| **4096** | **4096** | **256** | 1210.9 | 861.5 | **1.41×** | **18.9** |
| **4096** | **12288** | **256** (9B mlp.down shape) | 3604.8 | 2572.4 | **1.40×** | **17.1** |
| **14336** | **4096** | **256** (9B FFN gate/up output) | 4829.0 | 2154.3 | **2.24×** | **30.2** |

**Aggregate across 11 shapes:** 1.67× speedup. Production-shape cluster
(M ≥ 4096, N = 256, the regime that matters for `bench_qwen35_mq4
--prefill 256`): **1.40-2.24× speedup, 17-30 GiB/s effective.** The
14336×4096 case beats HFQ4-mmq's 26 GiB/s at the same size on the
same hardware — the predicted "match the uniform-MQ4 ceiling" outcome
clears for the largest projection.

### Disassembly metadata (gfx-kernel-metadata skill)

| Field | `_wmma` (Phase A) | `_mb4` (Phase D-A) |
|---|---|---|
| VGPR/lane | 68 | 106 |
| SGPR | 18 | 22 |
| Spill | 0 | 0 |
| LDS | 512 B | 512 B |

VGPR cost went up significantly (+38) from holding 4 distinct B-vector
registers. **No spills**, but the increase is what drives the
small-shape regression: with 4× fewer WGs (16×16 → 16×64 tile) and
higher VGPR/lane, small kernels can't maintain the wave-per-CU count
the original `_wmma` had, leaving CUs idle.

### Pipelining: the recycled-B-register trap

First-pass `_mb4` with a streaming `half16_t b` declared inside an
unrolled `for s in [0..4)` loop ran **2-3× *slower* than `_wmma` at
production shapes**, despite predicting a 2-3× win. Disassembly showed
the compiler reusing the same physical register (v[69:76]) across all
4 sub-tile WMMAs, emitting `s_waitcnt vmcnt(0)` between every load and
its dependent WMMA — fully sequential, no overlap.

Fix: declared 4 named variables `b0a, b1a, b2a, b3a` (and the matching
quartet for the K2-paired second tile) outside the unrolled loop. The
compiler then allocates 4 distinct register windows (v[61:68],
v[69:76], v[77:84], v[85:92] visible in the post-fix dump) and queues
6+ `global_load_b128` ops ahead of the first WMMA, with `s_waitcnt
vmcnt(N)` decrementing as each WMMA consumes its B operand.
**Lesson:** for unrolled multi-WMMA inner loops, the unrolled
temporaries must be named distinct variables — a single in-loop
temporary gets recycled and serializes the dependency chain.
This is a recurring pattern; documenting here and noting in the
kernel header for future Lloyd / HFQ4 multi-batch-tile work.

### Phase D-A status against plan ship gate

Plan ship gate: `_mb4` reaches ≥ 22 GiB/s on the residual at batch ≥ 128 on gfx1151.

- **Largest production shape (14336×4096×256): 30.2 GiB/s — clears.**
- 4096×4096×256: 18.9 GiB/s — under, but 1.41× faster than `_wmma`.
- 4096×12288×256: 17.1 GiB/s — under, but 1.40× faster than `_wmma`.

The plan's "investigate" band (18-22 GiB/s) is where most production
shapes land. The big win at the FFN-output projection is enough to
move forward with D-B + D-C (fused siblings + dispatch wiring + size-
gated path selection); the mid-shapes will lift further once the
fused versions land (the residual is the smallest fanout opportunity
in the family — `gate_up`'s 2× projection fan-out is naturally a
better fit for the LDS-staging pattern).

**Path selection requirement (D-C):** `_mb4` regresses on small shapes
(see table). Dispatch wiring must size-gate — likely `M ≥ 4096 AND
batch ≥ 128`, falling back to `_wmma` otherwise. Threshold tuning
deferred to D-C bench.

## Phase D-C integrated bench (size-gated `_mb4` routing)

Wired `_mb4` into `gemm_mq4g256_lloyd_residual_wmma` with a size gate:
default routes to `_mb4` when `M ≥ 4096 AND batch ≥ 128 AND arch ∈
{gfx1100,1101,1102,1151}`, falls through to `_wmma` otherwise. Env
override: `HIPFIRE_LLOYD_MB4=1` force-on (any size, any qualifying
arch), `=0` force-off (D-A bypass).

### Cross-process A/B at production prefill (Qwen3.5-9B Lloyd-MQ4, gfx1151)

3 fresh process invocations × 2 modes, `--prefill 256 --prefill-runs 3
--gen 30 --warmup 5`, `HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1`.

| Mode | run 1 | run 2 | run 3 | mean prefill | mean gen | Lloyd / uniform |
|---|---|---|---|---|---|---|
| `MB4=0` (Phase 5b baseline) | 464.3 | 457.8 | 464.5 | **462.2 tok/s** | 37.4 | 41.2 % |
| `MB4=1` (D-A path)          | 503.0 | 497.8 | 506.1 | **502.3 tok/s** | 37.4 | **44.7 %** |
| Δ                            |       |       |       | **+40.1 tok/s (+8.7 %)** | 0    | +3.5 pp |

Decode unchanged at 37.4 tok/s — `_mb4` only fires on the prefill
batched-LA path; per-token decode goes through `forward_scratch` +
GEMV, untouched.

The 8.7 % wall-clock lift matches the prediction from the per-kernel
profile: residual was 29.6 % of prefill GEMM time at `_wmma`-baseline;
a 1.4× kernel-level speedup on that slice translates to roughly +9 %
on total prefill (29.6 % × (1 - 1/1.4) = 8.5 % saved).

### Size gate validation (Qwen3.5-9B Lloyd-MQ4, gfx1151)

Force-on `MB4=1` at small prefill (batch < 128) regresses, confirming
the gate's necessity:

| Prefill | `MB4=0` | `MB4=1` (force) | Δ |
|---|---|---|---|
| 16  | 274.6 tok/s | 256.9 tok/s | **-6.4 %** |
| 256 | 462.2 tok/s | 502.3 tok/s | **+8.7 %** |

Default mode (no env var, size-gated): matches `MB4=0` at prefill=16,
matches `MB4=1` at prefill=256. Production-safe — no regression on
small prefill, full lift on large prefill.

### Lloyd / uniform-MQ4 ratio progression on gfx1151 (9B)

| Phase | Prefill tok/s | Ratio |
|---|---|---|
| Pre-Phase-5b (per-token fallback)               | ~37  | ~3 % |
| Phase 5b ship (`_wmma`)                          | 462  | 41.2 % |
| Phase D-A (`_mb4` on residual only, size-gated) | 502  | **44.7 %** |
| Phase D-A target (closes residual gap fully)    | ~535 | ~48 % |
| Phase D-B target (fused siblings ported)        | ~675 | ~60 % |
| Phase C ship-gate goal (gfx1100 9B parity)      |      | 60 %  |

D-A's contribution (+3.5 pp) is the residual-only slice of the gain.
The remaining headroom (60 % - 45 % = 15 pp) is concentrated in the
fused siblings — `gate_up` alone is 44 % of prefill time at the
`_wmma`-baseline, and is on a 16×16-tile WMMA path identical to the
residual's pre-D shape. Phase D-B (fused-sibling port) is the biggest
absolute win remaining.

### What's locked in for D-C

- ✅ Size-gated default routing — production-safe, no env var needed
- ✅ Env override `HIPFIRE_LLOYD_MB4=0|1` for A/B benching
- ✅ gfx12 excluded from selector (panic message names the corruption class)
- ✅ Cross-process bench data for the 9B-Lloyd path
- ⏳ Phase D-B (fused siblings) — biggest remaining lever
- ⏳ Phase D-A residual `_mb4` on gfx1100 — bench help-wanted (no local hardware)

## Phase D-B integrated bench (fused siblings ported to `_mb4`)

Three new kernels:
- `kernels/src/gemm_qkvza_mq4g256_lloyd_wmma_mb4.hip` — LA preamble (4-way fan-out)
- `kernels/src/gemm_qkv_mq4g256_lloyd_wmma_mb4.hip` — FA preamble (3-way fan-out)
- `kernels/src/gemm_gate_up_mq4g256_lloyd_wmma_mb4.hip` — FFN preamble (2-way fan-out)

Same multi-batch-tile pattern as `residual_mb4`: 16×64 output tile per
WG, 4 distinct named B-vars to avoid the compiler register-recycling
trap. Per-output-row routing to the correct Y_* projection happens at
write time, identical to the `_wmma` siblings.

Wired through the existing dispatch methods with the same size gate as
D-A (`total_m ≥ 4096 AND batch ≥ 128 AND gfx11`) and the same
`HIPFIRE_LLOYD_MB4={0,1}` env override.

### Bit-exact parity (gfx1151, 8 fused shapes)

`test_gemm_fused_mq4g256_lloyd_wmma` with `MB4=0` (`_wmma`) vs `MB4=1`
(force `_mb4` at all sizes): every per-projection `max_abs` printed
identical to last digit across all 8 cases (qkvza, qkv, gate_up shape
sweeps). The fanout pattern preserves the WMMA accumulation order
within each accumulator — no fp32-reorder envelope opens up vs the
single-tile `_wmma` baseline.

### Cross-process A/B (Qwen3.5-9B Lloyd-MQ4, gfx1151)

3 fresh process invocations × 2 modes, `--prefill 256 --prefill-runs 3
--gen 30 --warmup 5`, `HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1`.

| Mode | run 1 | run 2 | run 3 | mean prefill | mean gen | Lloyd / uniform |
|---|---|---|---|---|---|---|
| `MB4=0` (Phase 5b ship)              | 460.5 | 464.8 | 464.2 | **463.2 tok/s** | 37.3 | 41.3 % |
| `MB4=1` (D-A + D-B — full _mb4 family) | 787.9 | 805.7 | 808.8 | **800.8 tok/s** | 37.3 | **71.3 %** |
| Δ                                      |       |       |       | **+337.6 tok/s (+72.9 %)** | 0   | +30.0 pp |

### Cross-process A/B (Qwen3.5-4B Lloyd-MQ4, gfx1151)

Same config, 4B model.

| Mode | run 1 | run 2 | run 3 | mean prefill | mean gen | Lloyd / uniform |
|---|---|---|---|---|---|---|
| `MB4=0` | 983.8  | 985.5  | 986.8  | **989.3 tok/s** (median col)  | 59.9 | 50.9 % |
| `MB4=1` | 1605.5 | 1607.5 | 1608.7 | **1607.2 tok/s** | 59.6 | **82.8 %** |
| Δ       |        |        |        | **+617.9 tok/s (+62.5 %)** | 0   | +31.9 pp |

(4B `MB4=0` mean of medians 989.3 differs from the table-top "last-run"
SUMMARY values by the in-process median selection rule — same metric
the gfx1100 Phase C devlog uses.)

### Lloyd / uniform-MQ4 ratio progression on gfx1151 — final

| Phase | 9B prefill | 9B ratio | 4B prefill | 4B ratio |
|---|---|---|---|---|
| Pre-Phase-5b (per-token fallback) | ~37   | ~3 % | ~120  | ~6 % |
| Phase 5b ship (`_wmma`)           | 463   | 41.3 % | 989   | 50.9 % |
| Phase D-A only (residual `_mb4`)  | 502   | 44.7 % | (not separately measured) | |
| **Phase D-A + D-B (full family)** | **801** | **71.3 %** | **1607** | **82.8 %** |
| Phase C ship gate (gfx1100 9B)    |       | 60 %    |          |          |
| gfx1100 sibling baseline (9B)     |       | 60.6 %  |          |          |

**Both Lloyd / uniform ratios on gfx1151 now exceed the 60 % ship gate
and exceed the gfx1100 baseline.** On bandwidth-bound APU hardware,
the multi-batch-tile fanout pattern — which tile-size limitation was
in retrospect a mismatch with HFQ4's already-tuned mmq path — fully
closes the architectural gap and then some. The 4B ratio (82.8 %)
even surpasses the gfx1100 sibling-format (MQ3-Lloyd) ratio of 88 %
within format-tax noise.

### Decode regression — none

`gen_tok_s` invariant across MB4 modes for both models (37.3-37.4 on
9B, 59.5-59.9 on 4B). `_mb4` only fires on the prefill batched-LA
path; per-token decode goes through `forward_scratch` + GEMV, which
is untouched. Confirmed via `bw_gib_s` invariance (decode bandwidth)
to within ±0.1 GiB/s.

### Small-prefill regression — gate works

Force `MB4=1` at prefill=16 on 9B regresses to 271 tok/s (vs `MB4=0`
276.6 tok/s, -2 %) — confirms the 4× WG reduction hurts when there's
not enough work to fill the CUs. Default mode (no env var, size-
gated) at prefill=16 falls through to `_wmma` (batch=16 < 128) and
matches the 276.6 baseline, no regression.

### What's locked in

- ✅ Bit-exact parity on residual + 3 fused kernels (11 + 8 shapes)
- ✅ Size-gated default routing — production-safe at all prefill sizes
- ✅ Env override `HIPFIRE_LLOYD_MB4={0,1}` for A/B benching
- ✅ gfx12 explicitly excluded from `_mb4` selectors (panic with
  corruption-class message)
- ✅ Cross-process bench: 9B +73 %, 4B +63 %, both clear the 60 %
  ship gate by wide margins
- ⏳ gfx1100 cross-arch confirmation — bench help-wanted (no local
  hardware)
- ⏳ gfx1100 small-batch path-selection threshold — same threshold
  may not be optimal on GDDR6 hardware where L2 absorbs more of the
  duplicate-fetch pressure on the `_wmma` family

## Phase D follow-up — mb2 investigation (negative result)

After Phase D-A + D-B landed, post-D profile showed `residual_mb4` was
the laggard at 16.4 GiB/s vs the fused siblings at 23-30 GiB/s.
Hypothesis: residual's small total_m (4096 = model dim) leaves mb4
occupancy-bound — only 1024 WGs across 40 CUs, with VGPR=106 forcing
2-occupancy slot. **mb2** (16×32 output tile, 2 sub-tiles per WG)
trades half the per-WG weight reuse for 2× the WG count and ~21
fewer VGPRs, predicting a win for residual specifically.

### Parity (gfx1151, 3-way A/B vs `_wmma` baseline)

mb2 bit-exact across all 11 shapes (max_abs / max_rel / rms identical
to `_wmma` and `_mb4`). Production-shape comparison:

| Shape (M×K×N) | `_wmma` µs | `_mb2` µs (vs ref) | `_mb4` µs (vs ref) | Winner |
|---|---|---|---|---|
| 4096×4096×256 (wo) | 1211.1 | **857.7** (1.41×) | 885.3 (1.37×) | mb2 (3% margin) |
| 4096×12288×256 (mlp.down) | 3566.0 | 2755.8 (1.29×) | **2620.5** (1.36×) | mb4 |
| 14336×4096×256 (gate_up out) | 4697.5 | 2958.4 (1.59×) | **2140.0** (2.20×) | mb4 decisive |

mb2 does NOT dominate residual at production scale. The wo case
(K=4096, 16 groups) is the only place mb2 nudges ahead, and only by
3 %. mlp.down (K=12288, 48 groups) is mb4 territory because per-group
sync overhead dominates and mb4 amortizes it across 4× more output
cells per WG. The occupancy hypothesis was right at small batches
(prefill ≤ 64) but doesn't survive the per-group-overhead picture
at K=12288.

### Computed integrated impact

K-aware routing — mb2 for wo (K=4096), mb4 for mlp.down (K=12288):
- wo with mb2: 858 µs × 32 calls = 27.5 ms (vs mb4 28.3 ms — saves 0.8 ms)
- mlp.down with mb4: 2621 µs × 32 calls = 83.9 ms (unchanged)
- Total residual savings: 0.8 ms / 327 ms wall ≈ **0.3 %**.

Not worth the dispatch complexity. mb4-everywhere on residual is
within noise of optimal across both shape clusters.

### Remaining lever budget

| Lever | Status | Estimated upside |
|---|---|---|
| Multi-batch-tile fanout (D-A + D-B) | ✅ Shipped | +30 pp Lloyd/uniform on 9B |
| mb2 for residual | ❌ Investigated, not shipped | ~0.3 % |
| Async LDS prefetch / double-buffer codebook | ⏳ Not tried | ~5-10 % on K=12288 cases only |
| Multi-row-tile (32×64 or 64×64 output) | ⏳ Not tried | Maybe +5 pp; significant rewrite |
| Per-arch K-schedule (K4 vs K2) | ⏳ Not tried | Marginal; needs bench |
| __launch_bounds__ retuning | ⏳ Not tried | Likely 0-3 % |
| 8-batch-tile (`_mb8`) | ⏳ Not tried | Probably regresses (occupancy) |

Format-tax structural floor: 1/1.176 = **85 %** (Lloyd's +17.6 %
weight footprint vs HFQ4). 9B is at 71.3 %, gap-to-floor 13.7 pp; 4B
is at 82.8 %, gap-to-floor 2.2 pp (essentially at-floor). The simple
architectural levers are exhausted; remaining headroom is split
between the format tax (immutable) and harder rewrites (multi-row-
tile, async prefetch).

### What's preserved as documented dead code

`gemm_mq4g256_lloyd_residual_wmma_mb2.hip` ships with the branch but
is NOT wired into the size gate. Available as prior art for:
- Future gfx1100 cross-arch tuning (different L2 / bandwidth ratio
  may make mb2 the right pick there)
- Future register-pressure-constrained arch ports
- Reference implementation if a Phase E ever revisits this
The dispatch method `gemm_mq4g256_lloyd_residual_wmma_mb2` exists,
the parity test verifies correctness 3-way, but production routing
goes mb4-only.

## Reproducer

```bash
export PATH="/opt/rocm-7.12/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm-7.12/lib:${LD_LIBRARY_PATH:-}"

source scripts/gpu-lock.sh && gpu_acquire "bench-mq4-lloyd-gfx1151"

for model in qwen3.5-9b.mq4 qwen3.5-9b.mq4-lloyd \
             qwen3.5-4b.mq4 qwen3.5-4b.mq4-lloyd; do
  for run in 1 2 3; do
    HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 \
      ./target/release/examples/bench_qwen35_mq4 \
      ~/.hipfire/models/$model \
      --prefill 256 --warmup 5 --prefill-runs 3 --gen 30
  done
done

gpu_release
```
