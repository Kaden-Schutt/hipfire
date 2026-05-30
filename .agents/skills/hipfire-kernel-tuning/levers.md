# Tuning levers catalog

The actual optimization patterns hipfire uses, with pointers to the
commits where they landed (or were tried-and-reverted). Pick the one
that matches the bottleneck you root-caused in `playbook.md` step 2.

Rule of thumb: pick ONE lever per commit. Bundling makes bisect
useless when the win turns out to be one of three changes and the
other two are wash-or-regression.

## 1. Wave-size port (CDNA3 ⇄ RDNA)

**When**: target arch is wave64 (CDNA1/2/3) and you're running a
wave32 kernel — half the lanes are silently masked out, kernel
correctness still works but throughput is halved or worse.

**Reference commit**: `4105035` — "perf(cdna3): full wave64 port of
all hot HFQ4 kernels — MI300X decode 48.6 → 96 tok/s". Ten HFQ4
kernels ported to 2-rows-per-block wave64 layout. A3B decode jumped
from 48.6 to 96 tok/s, matching 7900 XTX on the same model.

**Pattern**: separate `<name>.wave64.hip` file (or `.gfx942.hip`
chip-specific) using wave-64 lane decomposition (32 M-rows × 2 K
groups). Dispatch via the chip detection in `Gpu::init`.

**Don't try**: porting wave64 → wave32 for RDNA. RDNA is wave32
native; the wave64 mode (when supported) costs occupancy and rarely
helps inference.

## 2. Multi-row GEMV

**When**: decode is launch-overhead-bound (kernel-launch count
dominates per-token cost), not BW-bound. Profile shows low GB/s on
GEMV kernels but high invocation count.

**Pattern**: process R output rows per warp instead of 1, sharing
the `x` register state across rows. R=2 / R=4 / R=8 variants exist
in the tree:

```
kernels/src/gemv_hfq4g256_multirow.hip
kernels/src/gemv_hfq4g256_multirow.gfx1100.hip      # chip-tuned
kernels/src/gemv_hfq4g256_residual_multirow.gfx1100.hip
```

**Trade-off**: more VGPR pressure. Past R=8 you spill on RDNA3 and
the win disappears. Tuning is per-arch — gfx1100 likes R=4–8 for
hot decode kernels; gfx1010 prefers R=2 because of the smaller VGPR
budget.

**Reference**: existing `*_multirow*` kernels. Don't reinvent —
fork the closest existing variant and adjust R.

## 3. K-tile depth (K2 / K4 / K-split)

**When**: WMMA-bound prefill kernel where the inner K-tile loop
dominates wall-clock. Deeper unrolls amortize per-tile overhead but
push register pressure.

**Pattern**: K2 (process 2 K-tiles per loop body, soft-pipelined
via early load + lagged WMMA) is the canonical baseline. K4 has
more software pipelining headroom. K-split is for when K isn't a
multiple of the tile depth.

**Reference (positive)**: `gemm_hfq4g256_residual_wmma_k2.hip` is
the deployed baseline across all dominant prefill GEMMs
(`gemm_gate_up_hfq4g256_wmma`, `gemm_qkv_hfq4g256_wmma`,
`gemm_qkvza_hfq4g256_wmma`, residual). The K2 step is fully shipped.

**K4 step status**: `gemm_hfq4g256_residual_wmma_k4.hip` had a
swapped-output-mapping investigation (case-studies §8). After fix, K4
ties K2 byte-for-byte at m=4096 on 9B residual but loses to ksplit by
~33% per-call at small batch (CU-starved grid: 3.3 vs 13 blocks/CU under
K4's `__launch_bounds__(32, 1)`). Verify the current dispatch policy
before using K4, and do not cite an exact historical hash unless it exists
in the checkout you are reporting from.

**Reference (NEGATIVE — null result, important to know)**: commit
`f670e16` — "experiment(gemm): k2x32 wider-row lm_head — null
result". 32-row block × K2 unroll measured 46% **slower** at
M=248320. Hypothesis: doubled accumulator + 4× dequant live ranges
push past comfortable register budget, forcing spills. Lesson:
register pressure is the gating constraint past a certain point;
more parallel work doesn't help when you can't pipeline it.

## 4. WMMA / MFMA matrix engine

**When**: prefill GEMM (batch_size > 1) on an arch with a matrix
engine. Hipfire dispatches WMMA on gfx11/gfx12 and MFMA on gfx94x;
non-WMMA archs (gfx1010, gfx1030) fall through to the dot2 / scalar
path.

**Pattern**: 16×16×16 fp16→fp32 tile is the workhorse. The actual
builtin name + operand layout differs per arch family — see
`.agents/skills/hipfire-arch-port/wmma-matrix.md` if you're porting to a
new arch.

**Reference (positive)**: PR #56 — gfx12 WMMA port, channel-tested
on R9700, all 6 hot fused-projection kernels covered.

**Reference (silent-corruption cautionary tale)**: commit `b7ac66a`
— gfx11 C-mapping (`acc[j] = C[2*j + (tid>>4)][tid & 15]`) was
silently wrong for ~6 weeks. WMMA passed every functional test that
didn't compare element-by-element against a CPU reference. Lesson:
**channel-test is non-negotiable** for any WMMA / MFMA work.

## 5. Software prefetch (`s_prefetch_data`)

**When**: hot decode kernel that re-reads weights every token; the
profiler shows L2 misses dominating decode latency. Available on
gfx12 (RDNA4) only.

**Reference**: `kernels/src/gemv_hfq4g256.gfx1201.hip` — uses
`s_prefetch_data` for 2-group lookahead (~272 bytes). Comment in
the kernel header explains: "RDNA4 allows 96 VGPRs at max 16-wave
occupancy" — bigger budget there enables more aggressive prefetch.

**Pattern**: chip-specific override. `<name>.gfx1201.hip` covers
9070 XT only; if you want 9070 XT + R9700 (also gfx1201), the family
tag `<name>.gfx12.hip` covers both. See `cross-arch.md`.

**Don't try**: porting `s_prefetch_data` to gfx11. The RDNA3
intrinsic doesn't expose the same prefetch primitive.

## 6. Fused projections (multi-output kernels)

**When**: multiple GEMM/GEMV calls share an input vector (Q, K, V
all read the same x; gate + up share the same x). Each separate
launch eats kernel-launch overhead; fusing into one kernel lets you
load x once and write multiple Y outputs.

**Pattern**: kernels named `gemm_qkv_*` (3-way fused QKV),
`gemm_qkvza_*` (4-way for DeltaNet), `gemm_gate_up_*` (2-way for
SwiGLU FFN). Each takes multiple A weight buffers + multiple Y
output buffers in one launch.

**Reference (positive)**: `9d05c9f` — "perf(fused-projection):
consolidate gfx1100 + baseline kernels into one cross-arch family".
The fused QKV path is one of the main reasons hipfire's decode
beats llama.cpp's by 1.7–2.1× on small models — llama.cpp does 3
separate GEMV launches per layer; hipfire does 1.

**Caveat**: fused kernels are bigger (more VGPRs). On VRAM-tight
arches (gfx1010, gfx1013, gfx1032) they sometimes lose to the
unfused path because of occupancy. Fall back via the dispatch tree
when this matters.

## 7. Per-kernel hipcc flags (magic comments)

**When**: a single kernel benefits from non-default compiler flags
(`-mcumode`, `-fno-unroll-loops`, custom `-ffast-math`-style, etc.)
that you don't want to apply globally.

**Reference**: `5f65005` — "feat(compile): per-kernel hipcc flag
magic-comment plumbing + bisect helpers". The kernel JIT picks up
`// HIPCC_FLAGS: -foo -bar` magic comments at the top of the .hip
file and adds them to that kernel's compile invocation only.

**Pattern**: add the magic comment, rebuild kernel hashes
(`./scripts/write-kernel-hashes.sh`), validate via the three gates.
Per-kernel flags don't propagate to other kernels in the tree, so
this is the safe way to experiment without globally affecting
compilation.

## 8. rocBLAS / cuBLAS-class GEMM fallback

**When**: prefill GEMM on a CDNA3 arch (gfx94x). The MFMA path
through rocBLAS can outpace hipfire's hand-rolled MFMA at very
large M.

**Reference**: `07a2b1c` and friends — "feat(rocblas): wire CDNA3
MFMA path into gemm_hfq4g256 + gate_up". `HIPFIRE_ROCBLAS_OFF=1`
kill-switch (`1316f8e`) for A/B benching;
`HIPFIRE_ROCBLAS_ALL_ARCHS=1` opens the path to RDNA3 (`05d104d`).

**Caveat**: rocBLAS dispatch is per-arch. Don't enable on RDNA
unless you've measured a win on YOUR arch — the default off for
RDNA is an empirical decision.

## 9. Things to NOT try (negative results documented)

These failed in past experiments. Don't burn cycles re-running
unless you have a NEW idea about why this time would be different.

### `nontemporal` weight loads on gfx1100

Commit `34eb024` (revert of `0532579`). Original commit claimed
+2% based on within-session A/B. Bisect against the committed
baseline showed actual −13% on 9B MQ4 decode. Hypothesis: bypassing
cache-line allocation also defeats wave-level coalescing on RDNA3.
Don't re-try without a different mechanism for cache control.

### k2x32 wider-row variant

Commit `f670e16` — null result, kept the kernel for future revisit.
46% slower on 27B M=248320 lm_head due to register-pressure spills.
If you want to retry, lead with an LDS-staged B-share + manual
register budget plan.

### Always-on hipGraph capture

Commits `33b8861` / `5705a59` / `0180b68` / `688b4fd` — series of
fixes to make hipGraph capture safe. Default-on hipGraph caused
silent garbage output (dangling stack-pointer kernargs from raw
`launch_kernel` calls in `forward_scratch_layers`). Now opt-in via
`HIPFIRE_GRAPH=1` only, and even then perf-neutral or slightly
worse on most archs. Don't make it default-on without a thorough
correctness pass.

### LDS-staged X share on gate_up (gfx1100)

Historical branch note: variant kernel `gemm_gate_up_hfq4g256_wmma_ldsx.hip`
opt-in via `HIPFIRE_GATE_UP_VARIANT=ldsx`, default off. ISA-clean (75 VGPRs vs
80 baseline, single-wave block makes `__syncthreads()` a no-op so
the compiler kept weight prefetch above the LDS-write phase) but
**per-call wall regressed +20% / +29% / +37% at pp32 / pp128 /
pp512 on Qwen 3.5 9B**. Replaced the baseline's 2 VMEM stalls per
inner-iteration with 3 VMEM + 2 LGKM stalls — the new stalls
weren't hidden by wave-level ILP the way the original `vmcnt(0)`
was. See case-studies §7 for the full diagnosis. Don't re-try
without (a) a fundamentally different LDS layout that doesn't
serialize through register, or (b) on RDNA4 (gfx12), where
`s_prefetch_data` may change the calculus.

## 10. Barrier-free syncthreads elimination (LDS → direct global)

**When**: a kernel inside a loop does cooperative LDS staging (all
warps load data into LDS, then `__syncthreads()`, then each warp
reads from LDS, then `__syncthreads()` to release). Profiling shows
low BW utilization (<10% of peak) — meaning the redundant global
reads from independent per-warp loads are cheaper than the barrier
serialization.

**Pattern**: remove the LDS buffer and both `__syncthreads()` calls.
Each warp loads its data directly from global memory on demand. For
X/activation staging (each warp needs all 16 slots), each warp reads
all 16 from global — 4× redundant. For weight staging (each warp
needs its own subset), no redundancy.

**Reference commits**:
- `7e0ac9b9` — "perf(deepseek4): barrier-free MoE grouped WMMA
  kernel (+39-43%)". LDS X-staging removed from 4-warp MQ2-Lloyd
  grouped WMMA. +25% occupancy from VGPR optimization produced 0%
  gain — the real win was removing the barriers.
- `7d51a401` — "perf: barrier-free gate_up + MoE grouped WMMA
  kernels". Repeats the pattern on the HFQ4 `gemm_gate_up_hfq4g256
  _wmma_ldscoop` kernel: +53%.

**Prerequisites**:
1. Kernel BW utilization < 15% of peak (check per-kernel timer in
   profile output: `total_MiB / total_us` vs arch peak).
2. The LDS data is either per‑lane‑partitioned (each lane reads one
   element from the LDS share — zero redundancy cost) or shared only
   within a warp (all lanes in a warp read the same element — 1/4 ×
   number_of_warps redundancy).
3. The `__syncthreads()` is inside a loop, not a one-time phase
   barrier.

**Don't try**: on kernels where the LDS share is genuinely
cross-warp (e.g. DFlash attention V page). That requires a
fundamentally different algorithm (per-wave V buffering) to
eliminate barriers.

**Anti-pattern**: "barriers are free on single-wave blocks." A
1-wave kernel's `__syncthreads()` is a compiler fence, not a HW
barrier — it doesn't stall. The optimization only applies to
multi-wave workgroups (128 threads/4 waves in hipfire's canonical
layout). Don't add nosync variants for 1-wave kernels.

**Completed implementations** (all with `_nosync.hip` variant created,
build-verified, correctness-tested):

| Kernel | Status | Production-active? |
|--------|--------|-------------------|
| `gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload` | **+43% measured** (PR #356) | ✅ DeepSeek V4 MoE |
| `gemm_gate_up_hfq4g256_wmma_ldscoop` | **+53% measured** (PR #359) | ✅ HFQ4 gate_up |
| `gemm_gate_up_hfq4g256_wmma_mmq.gfx1151` | nosync created, build OK | ❌ Not wired yet |
| `gemm_gate_up_hfq4g256_wmma_mmq_k4.gfx1151` | nosync created, build OK | ❌ Not wired yet |
| `gemm_gate_up_mq4g256_lloyd_wmma` | nosync created, tested | ❌ Not wired yet |
| `gemm_gate_up_mq4g256_lloyd_wmma_mb4` | nosync created, tested | ❌ Not wired yet |
| `gemm_gate_up_mq4g256_lloyd_wmma.gfx1151` | nosync created, tested | ❌ Not wired yet |
| `gemm_gate_up_mq4g256_lloyd_wmma_mb4.gfx1151` | nosync created, tested | ❌ Not wired yet |
| `gemm_gate_up_mq3g256_lloyd_wmma` | nosync created, tested | ❌ Not wired yet |
| `gemm_gate_up_mq3g256_lloyd_wmma_mb4` | nosync created, tested | ❌ Not wired yet |
| `gemm_q8_0_wmma_4w` | nosync wired, `HIPFIRE_Q8_NOSYNC=1` | ✅ Q8_0 path |

The unwired kernels pass the `test_gemm_fused_mq4g256_lloyd_wmma` 
correctness test but aren't linked from the arch forward pass
(`qwen35.rs`). When they get productionized, the `_nosync` variants
are ready and the `HIPFIRE_GATE_UP_NOSYNC=1` env var gates them.

## When you're done

Commit your win (or your null-result revert) with the commit-message
template from `playbook.md` step 6. The next contributor reading the
git log saves the same hours you spent — that's the durable value
of this kind of writeup.
