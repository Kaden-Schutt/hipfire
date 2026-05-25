# Dev log 2026-05-07 — MQ3-Lloyd WMMA prefill, Phase A

**Branch:** `feat/mq3-lloyd-wmma-prefill`
**Plan:** `docs/plans/mq3-lloyd-wmma-prefill.md` (rev 2, commit a654099)
**Hardware:** gfx1100 (7900 XTX), ROCm 7.2.

## Summary

Phase A MVP shipped: the residual WMMA kernel for MQ3-Lloyd parses
correctly, passes parity at fp32-acc-from-fp16 noise tolerance, runs
spill-free, and the open empirical question (fp16 vs fp32 LDS storage)
is resolved in favour of **fp16-LDS** by a 7.15% wall-time margin.

All four Phase A acceptance gates per the plan are cleared:
1. Parity test PASS at empirically-set tolerance.
2. No VGPR spills (`.private_segment_fixed_size: 0`).
3. Zero `__syncthreads()` inside the K-tile loop (2 total per group, both
   at group boundaries).
4. fp16-vs-fp32 LDS choice resolved with bench numbers below.

## What landed

- `kernels/src/gemm_mq3g256_lloyd_residual_wmma.hip` — the production
  kernel (fp16 LDS).
- `kernels/src/gemm_mq3g256_lloyd_residual_wmma_lds_f32.hip` — sibling
  fp32-LDS variant for the bench A/B; can be retained for future
  contributors to re-validate or removed in Phase B1 cleanup.
- `crates/rdna-compute/src/kernels.rs` — two new constants, one per
  variant.
- `crates/rdna-compute/src/dispatch.rs` — `gemm_mq3g256_lloyd_residual_wmma`
  + `gemm_mq3g256_lloyd_residual_wmma_lds_f32` dispatch arms (gfx1100
  only; gfx12 sibling deferred to Phase B1).
- `crates/rdna-compute/examples/test_gemm_mq3g256_lloyd_residual_wmma.rs`
  — parity test + per-variant timing across 8 canonical shapes.

No `is_batchable_la` change. No matcher updates in `qwen35.rs`. The
kernels are dead code at the dispatch level — invoked only by the
parity-test example. Per plan, this is Phase A scope.

## Parity results

CPU reference accumulates in fp64; X is f16-roundtripped to match the
GPU's view after `ensure_fp16_x` conversion. Max-abs error scales as
expected with K (sqrt(K) × ulp at the partial-sum magnitude).

| M | K | N | max_abs | max_rel | rms | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 64   | 1024  | 16  | 1.49e-6 | 1.93e-4 | 4.13e-7 | PASS |
| 64   | 1024  | 64  | 1.49e-6 | 3.31e-4 | 4.37e-7 | PASS |
| 256  | 1024  | 64  | 1.88e-6 | 3.77e-4 | 4.27e-7 | PASS |
| 64   | 4096  | 64  | 9.42e-6 | 1.18e-3 | 3.66e-6 | PASS |
| 256  | 4096  | 16  | 1.05e-5 | 7.58e-4 | 3.80e-6 | PASS |
| 256  | 4096  | 256 | 1.05e-5 | 1.51e-3 | 3.77e-6 | PASS |
| 1024 | 4096  | 64  | 1.05e-5 | 1.51e-3 | 3.77e-6 | PASS |
| 1024 | 12288 | 64  | 5.83e-5 | 2.65e-2 | 2.57e-5 | PASS |

(fp16-LDS and fp32-LDS variants produce bit-identical outputs — see
"fp16 ≈ fp32 numerical equivalence" below.)

**Suggested B1 tolerance**: ~3× max-abs observed at K=12288 = **1.75e-4**.
Use this for the regression-detection threshold in Phase B1's expanded
parity tests.

## Disassembly metadata (gfx1100, fp16-LDS variant)

Extracted via the `gfx-kernel-metadata` skill (`docs/skills/gfx-kernel-metadata`):

| Field | Value |
|---|---:|
| `.vgpr_count` | 82 |
| `.sgpr_count` | 18 |
| `.private_segment_fixed_size` | **0** (no spills) |
| `.group_segment_fixed_size` | 256 (16 codebooks × 8 entries × 2 B fp16) |
| `.wavefront_size` | 32 |
| `.max_flat_workgroup_size` | 32 |

VGPR occupancy on gfx1100 (1024 VGPRs/SIMD): 1024 / 82 ≈ 12 waves/SIMD
theoretical. Asked `__launch_bounds__(32, 2)` so 2 waves/SIMD is the
binder. LDS-side: 256 B / 64 KB = 256 workgroups/CU (not the binder).
Plenty of headroom; nothing to retune.

## fp16 ≈ fp32 numerical equivalence

Surprising-but-correct observation: max-abs error of the fp32-LDS
variant matches the fp16-LDS variant *bit-equal* across every shape
(the table above is identical for both variants). Rationale:

- Codebook is stored as fp16 in global memory (16 B header at group
  offset 0).
- Both variants `__half2float` the value at load (fp32-LDS variant) or
  copy fp16 directly (fp16-LDS variant) into LDS.
- Both variants assign the LDS lookup result to `_Float16 a_reg[i]`
  for WMMA — fp32-LDS does an explicit `(_Float16)cb_lds[idx]` cast,
  fp16-LDS reads fp16 directly.
- The narrowing `f32 → f16` (fp32 path) and the no-op `f16 → f16`
  (fp16 path) end up producing the same `a_reg` bits, because every
  cb_lds value started as fp16 and the f32 conversion is bit-exactly
  reversed by the f16 narrow.

WMMA receives identical inputs in both variants, so outputs are
bit-equal. The fp32-LDS variant pays for the cvt at no numerical
benefit.

## Bench results (fp16-LDS vs fp32-LDS)

20 timed iterations per variant per shape, 3 warmup, wall-clock from
`download_f32` boundaries (forces device sync). Same shapes as parity.

| M | K | N | fp16-LDS µs/call | fp32-LDS µs/call | Δ |
|---:|---:|---:|---:|---:|---:|
| 64   | 1024  | 16  | 21.8  | 23.4  | +7.3% |
| 64   | 1024  | 64  | 21.7  | 22.7  | +4.6% |
| 256  | 1024  | 64  | 23.0  | 24.2  | +5.2% |
| 64   | 4096  | 64  | 57.4  | 61.3  | +6.8% |
| 256  | 4096  | 16  | 57.6  | 61.6  | +6.9% |
| 256  | 4096  | 256 | 93.8  | 97.1  | +3.5% |
| 1024 | 4096  | 64  | 89.7  | 98.6  | +9.9% |
| 1024 | 12288 | 64  | 256.0 | 276.5 | +8.0% |

**Aggregate**: fp16-LDS = 621.0 µs, fp32-LDS = 665.4 µs. **fp16-LDS
wins by 7.15%**, consistent across every individual shape (fp32 never
wins).

The 7% margin matches the rough budget for `v_cvt_f16_f32` at the
decode site: ~16 cvt ops per K-tile per row × 16 K-tiles × N output
columns. Not a huge win in absolute terms, but consistent enough to
remove the question from Phase B1's scope.

### Bench caveat (within-session)

Wall-clock A/B in a single session has CLAUDE.md-documented ±10–15%
DPM/thermal variance. The 7.15% margin sits within that envelope, but
the per-shape 4–10% spread (with fp32 always slower) is structural,
not noise. A cross-process bench via `probe_commits.sh`-style harness
would tighten this; deferred to Phase C since the decision is already
clear.

## Decision: ship fp16-LDS

- Production kernel: `gemm_mq3g256_lloyd_residual_wmma.hip` (fp16-LDS).
- B1 follow-up: replicate fp16-LDS pattern for the 3 fused kernels
  (qkvza / qkv / gate_up). The fp32-LDS sibling is bench-only;
  either retain it for B1 reviewer reference or drop in B1's cleanup
  commit.

The fp16-LDS pattern is a clean fit for the 16-row-codebook layout:

- 256 B/workgroup LDS — trivial.
- Cooperative load: 32 threads × 4 fp16 each, no conversion cost.
- 16-row × 8-entry layout means each tile-row's codebook sits
  contiguously at `cb_lds[row*8 .. row*8+8]` — vectorizable in B1
  if a future optimization wants to issue `half4_t` LDS reads.

## What's next (Phase B1)

Per plan, B1 is the kernels-only-dead-code commit:

- Add 3 fused kernels (qkvza / qkv / gate_up) each as gfx1100 + gfx12.
- Add residual gfx12 sibling for the kernel landed in Phase A.
- Add 3 new dispatch arms in `dispatch.rs` (one per fused kernel),
  no `is_batchable_la` change, no matcher updates.
- Parity tests for each fused kernel using mocked-distinguishable
  inputs to catch arg-reordering bugs at parity time.

Then B2: the all-together corruption-prevention commit (matcher
updates + `is_batchable_la` widening + the 5-item reviewer checklist
from the plan).

## Reproducing

```sh
# Build:
cargo build --release -p rdna-compute --example test_gemm_mq3g256_lloyd_residual_wmma

# Run (under GPU lock):
source scripts/gpu-lock.sh && gpu_acquire "phase-a-mq3-lloyd-wmma" && \
  ./target/release/examples/test_gemm_mq3g256_lloyd_residual_wmma
gpu_release

# Disassembly metadata (requires the gfx-kernel-metadata skill):
ARCH=gfx1100 ROCM=/opt/rocm/llvm/bin
$ROCM/clang-offload-bundler --type=o --unbundle \
    --input=.hipfire_kernels/$ARCH/gemm_mq3g256_lloyd_residual_wmma.hsaco \
    --output=/tmp/k.elf \
    --targets=hipv4-amdgcn-amd-amdhsa--$ARCH
$ROCM/llvm-readelf --notes /tmp/k.elf | \
  grep -E "\.(vgpr_count|sgpr_count|private_segment_fixed_size|group_segment_fixed_size|wavefront_size):"
```
