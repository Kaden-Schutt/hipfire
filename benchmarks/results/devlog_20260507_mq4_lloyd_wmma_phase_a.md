# Dev log 2026-05-07 — MQ4-Lloyd WMMA prefill, Phase A

**Branch:** `feat/issue-182-mq4-lloyd` (Phase A bundled into the issue #182
PR per maintainer direction — "let's land MQ4 in one go").
**Plan:** `docs/plans/mq4-lloyd-wmma-prefill.md` (commit `4f3f1ec`).
**Hardware:** gfx1151 (Strix Halo APU, RDNA3.5), ROCm 7.12.
**Sibling:** MQ3-Lloyd Phase A — `devlog_20260507_lloyd_wmma_phase_a.md`,
commit `869236d` (gfx1100, 7900 XTX). Multiple resolved decisions
inherited from there.

## Summary

Phase A MVP shipped: the residual WMMA kernel for MQ4-Lloyd parses
correctly, passes parity at fp32-acc-from-fp16 noise tolerance, runs
spill-free, and lands at slightly lower VGPR pressure than MQ3 Phase A
(68 VGPR vs 82) thanks to the simpler nibble-pair decode. fp16-LDS
adopted directly from MQ3 Phase A's empirical conclusion — no fp32
sibling implemented.

All four Phase A acceptance gates per the plan are cleared:

1. Parity test PASS at empirically-set tolerance.
2. No VGPR spills (`.private_segment_fixed_size: 0`).
3. Zero `__syncthreads()` inside the K-tile loop (2 total per group,
   both at group boundaries).
4. fp16-vs-fp32 LDS choice resolved by inheritance — no MQ4 sibling
   bench needed.

## What landed

- `kernels/src/gemm_mq4g256_lloyd_residual_wmma.hip` — the production
  kernel (fp16-LDS, K2 unroll, single `float8_t acc` per lane).
- `crates/rdna-compute/src/kernels.rs` — `GEMM_MQ4G256_LLOYD_RESIDUAL_WMMA_SRC`
  constant.
- `crates/rdna-compute/src/dispatch.rs` — `gemm_mq4g256_lloyd_residual_wmma`
  Rust binding (gfx1100 family + gfx1151; gfx12 sibling deferred to
  Phase B1).
- `crates/rdna-compute/examples/test_gemm_mq4g256_lloyd_residual_wmma.rs`
  — single-variant parity + per-shape timing across 8 canonical shapes.

No `is_batchable_la` change. No matcher updates in `qwen35.rs`. The
kernel is dead code at the dispatch level — invoked only by the
parity-test example. Per plan, this is Phase A scope.

Single variant: fp16-LDS only. No `_lds_f32.hip` sibling and no Variant
enum in the test — MQ3 Phase A's `devlog_20260507_lloyd_wmma_phase_a.md`
established that the f32→f16 narrow at decode produces bit-identical
inputs to WMMA, so the fp32 path pays for the cvt at no numerical
benefit. Same logic applies to MQ4's 16-entry codebook (codebook is
stored as fp16 in global memory; both paths converge on the same
`a_reg`).

## Parity results

CPU reference accumulates in fp64; X is f16-roundtripped to match the
GPU's view after `ensure_fp16_x` conversion. Same 8-shape grid as
MQ3 Phase A.

| M | K | N | max_abs | max_rel | rms | us/call | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
|   64 |  1024 |  16 | 1.371e-6 | 3.651e-4 | 4.751e-7 |  34.1 | PASS |
|   64 |  1024 |  64 | 1.371e-6 | 4.142e-4 | 4.931e-7 |   9.8 | PASS |
|  256 |  1024 |  64 | 1.490e-6 | 4.838e-4 | 4.852e-7 |  41.8 | PASS |
|   64 |  4096 |  64 | 1.210e-5 | 2.606e-3 | 4.241e-6 |  28.3 | PASS |
|  256 |  4096 |  16 | 1.317e-5 | 2.606e-3 | 4.291e-6 |  43.9 | PASS |
|  256 |  4096 | 256 | 1.317e-5 | 3.036e-3 | 4.237e-6 |  48.1 | PASS |
| 1024 |  4096 |  64 | 1.317e-5 | 3.036e-3 | 4.244e-6 |  47.6 | PASS |
| 1024 | 12288 |  64 | **7.463e-5** | 1.940e-2 | 3.144e-5 | 132.2 | PASS |

**Suggested B1 tolerance**: ~3× max-abs observed at K=12288 = **2.24e-4**.
Slightly looser than MQ3 Phase A's suggested 1.75e-4 (3× of MQ3's
5.83e-5), consistent with the larger codebook + LDS — same envelope.

## Disassembly metadata (gfx1151, fp16-LDS)

Extracted via the `gfx-kernel-metadata` skill recipe (commit `9140c9d`)
on the cached kernel at `.hipfire_kernels/gfx1151/gemm_mq4g256_lloyd_residual_wmma.hsaco`:

| Field | MQ4 value | (MQ3 Phase A reference) |
|---|---:|---:|
| `.vgpr_count` | **68** | 82 |
| `.sgpr_count` | 18 | 18 |
| `.private_segment_fixed_size` | **0** (no spills) | 0 |
| `.vgpr_spill_count` | 0 | (canary) |
| `.sgpr_spill_count` | 0 | (canary) |
| `.group_segment_fixed_size` | 512 (16 codebooks × 16 entries × 2 B fp16) | 256 |
| `.wavefront_size` | 32 | 32 |
| `.max_flat_workgroup_size` | 32 | 32 |

**MQ4 ends up at lower VGPR pressure than MQ3** (-14 VGPRs) despite
having 2× the LDS budget. The simpler nibble-pair decode (8 bytes →
2 uint32 reads → 16 nibbles via shifts of 0/4/.../28, mirroring
HFQ4's pattern at `gemm_hfq4g256_residual_wmma.hip:53`) needs fewer
register intermediates than MQ3's 6-byte cross-byte uint24 unpack
(which requires `(b1 << 2)`-style cross-byte reassembly). LDS doubles
to 512 B as expected (16-row × 16-entry codebook × 2 B fp16); 64 KB
LDS budget is irrelevant here.

VGPR occupancy on RDNA3 (1024 VGPRs/SIMD): 1024 / 68 ≈ 15 waves/SIMD
theoretical. `__launch_bounds__(32, 2)` is the binder. LDS-side: 512 B
/ 64 KB = 128 workgroups/CU (still not the binder). Plenty of
headroom; nothing to retune.

Also verified by source + assembly grep:
- 2 `__syncthreads()` in source — match 2 `; wave barrier` markers in
  the .s output.
- 2 `v_wmma_f32_16x16x16_f16` instructions in assembly — match 2 WMMA
  calls per K-tile iteration (K2 unroll → 8 outer iters × 2 = 16 total
  WMMA dispatches per output-tile per group at runtime).
- ZERO `s_barrier`-class instructions inside the K-tile loop body —
  the 2 barriers sit at group iteration boundaries (post-coop-load
  and pre-next-coop-load), bracketing the K-tile loop.

## Bench results — gfx1151 (caveats apply)

20 timed iterations per shape, 3 warmup, wall-clock from `download_f32`
boundaries (forces device sync).

| M | K | N | µs/call |
|---:|---:|---:|---:|
|   64 |  1024 |  16 |  34.1 |
|   64 |  1024 |  64 |   9.8 |
|  256 |  1024 |  64 |  41.8 |
|   64 |  4096 |  64 |  28.3 |
|  256 |  4096 |  16 |  43.9 |
|  256 |  4096 | 256 |  48.1 |
| 1024 |  4096 |  64 |  47.6 |
| 1024 | 12288 |  64 | 132.2 |
| **aggregate** | | | **385.9 µs** |

**These are gfx1151 numbers, not the headline.** The MQ3 Phase A
devlog ran on gfx1100 (7900 XTX, GDDR6 ~960 GB/s). gfx1151 is the
Strix Halo APU on shared LPDDR5x (~250 GB/s). Cross-arch perf
comparisons against MQ3 Phase A's gfx1100 numbers (621.0 µs aggregate
fp16-LDS) are not apples-to-apples. The gfx1151 run is the on-host
sanity that the kernel produces correct outputs at acceptable
latencies; **definitive perf comparisons happen later when this lands
on the gfx1100 maintainer host**.

The plan's Phase C bench-vs-MQ4-non-Lloyd comparison is the load-
bearing perf gate, not these gfx1151 numbers.

## fp16/fp32 LDS — inherited resolution

Skipped the fp32-LDS sibling per MQ3 Phase A's finding (commit
`869236d`, `devlog_20260507_lloyd_wmma_phase_a.md` "fp16 ≈ fp32
numerical equivalence"). Logic is identical for MQ4: codebook is
stored as fp16 in the 32 B group header → both fp16-LDS (raw fp16
copy) and fp32-LDS (`__half2float` at load + `(_Float16)` cast at
decode) produce the same `a_reg` bits because the f32 → f16 narrow is
exact for fp16-stored values. WMMA receives identical inputs → bit-
equal outputs.

The fp32 variant would only pay extra cost (`v_cvt_f16_f32` per element
at decode site, ~16 cvt ops × 16 K-tiles × tile-row count). MQ3 Phase
A measured this as a 7.15% aggregate hit on fp32. MQ4 has 2× more
LDS to read per-K-tile, so the fp32 hit would be at least the same.
No reason to implement.

## What's next (Phase B1)

Per plan, B1 is the kernels-only-dead-code commit:

- Add 3 fused kernels (qkvza / qkv / gate_up) each as gfx1100 + gfx12.
- Add residual gfx12 sibling for the kernel landed here.
- Add 3 new dispatch arms in `dispatch.rs` (one per fused kernel),
  no `is_batchable_la` change, no matcher updates.
- Parity tests for each fused kernel with mocked-distinguishable
  inputs to catch arg-reordering bugs at parity time. Particularly
  important for HFQ4 fused-QKVZA's 13-argument signature.

Then B2: the all-together corruption-prevention commit (matcher
updates + `is_batchable_la` widening for `MQ4G256Lloyd` + the 5-item
reviewer checklist from the plan). Then B3: coherence-gate row + Phase
C perf validation.

## Reproducing

```sh
# Build:
cargo build --release -p rdna-compute --example test_gemm_mq4g256_lloyd_residual_wmma

# Run (under GPU lock):
source scripts/gpu-lock.sh && gpu_acquire "phase-a-mq4-lloyd-wmma" && \
  ./target/release/examples/test_gemm_mq4g256_lloyd_residual_wmma
gpu_release

# Disassembly metadata (requires the gfx-kernel-metadata skill, or use
# the same recipe directly):
ARCH=gfx1151 ROCM=/opt/rocm-7.12/llvm/bin
$ROCM/clang-offload-bundler --type=o --unbundle \
    --input=.hipfire_kernels/$ARCH/gemm_mq4g256_lloyd_residual_wmma.hsaco \
    --output=/tmp/k.elf \
    --targets=hipv4-amdgcn-amd-amdhsa--$ARCH
readelf --notes /tmp/k.elf | grep -E "vgpr|sgpr|segment|wavefront"
```
