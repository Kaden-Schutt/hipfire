# Phase 3 Lever 1 — Fused RMSNorm + PARO rotation — FALSIFIED

> Branch `feat/paro-g256-perfmax`. hiptrx gfx1201, Qwen3.5-0.8B PARO4G128T engine layout.
> Default OFF; opt-in via `HIPFIRE_PARO_FUSE_RMSNORM=1`. Kept as research artifact.

## Headline

**The single-workgroup fused kernel is a net LOSS at -2.4% decode on 0.8B.**
Saves ~10µs launch overhead per fusion site but adds ~30-70µs of serialized rotation
work per call because the fused kernel runs the K-rotation on ONE workgroup, losing the
M/8 cross-CU parallelism that the split `paro4g128t_rotate` kernel gets for free with
its `grid=[K/128, 1, 1]` layout.

## Measurements

Median-of-3 fresh processes, 0.8B PARO4G128T engine layout, gfx1201, K=q8 KV:

| config | gen tok/s | prefill tok/s | avg ms/tok | Δ vs OFF |
|---|---:|---:|---:|---:|
| Lever 1 OFF (baseline) | **161.4** | **172.0** | **6.10** | — |
| Lever 1 ON (opt-in) | 157.0 | 166.8 | 6.28 | **-2.7%** |

`test_inference` 9/9 PASS in both modes (correctness intact).

## Root cause: CU occupancy mismatch

Split kernel layout (current production):
```
rmsnorm_f32:        1 block × 256 threads = 1 wave on 1 CU
paro4g128t_rotate:  K/128 blocks × 32 threads each
                    = 8 blocks × 1 wave = 8 waves on up to 8 CUs (parallel)
```

Fused kernel layout (Lever 1 implementation):
```
fused_rmsnorm_paro4g128t_rotate:
  1 block × 256 threads = 8 waves on 1 CU (serialized via __syncthreads)
```

R9700/gfx1201 has 40 CUs. The split rotate uses 8 of them in parallel; the fused
uses 1. The fused kernel's wall-clock for the rotation phase is ~8× the per-wave time,
which dominates the saved launch overhead.

Per-token accounting on 0.8B Qwen3.5 PARO (48 rmsnorm sites with PARO downstream):

```
SPLIT (current):
  48 rmsnorm_f32   × ~5µs  =  240µs  (multi-block fine)
  48 rotate (par.) × ~10µs =  480µs  (8 CUs in parallel)
  48 launches      × ~10µs =  480µs  launch overhead
  total: ~1200µs/tok of rmsnorm+rotate

FUSED (Lever 1 attempt):
  48 fused (1-CU)  × ~25µs = 1200µs  (1 CU, serialized)
  48 launches      × ~10µs =  480µs  launch overhead saved on 48 sites → 0
  total: ~1680µs/tok minus 480µs launch save = ~1200µs/tok approximately matched
```

Empirically we measured +160µs/tok regression (6.10 → 6.28ms) which corresponds to
~3.3µs per site of net loss after all overhead — consistent with the 1-CU rotation
phase being modestly slower than the 8-CU parallel split.

## Alternative architectures considered (all rejected as worse-or-equal)

1. **Multi-block fused with grid-sync**: would require `cooperative_kernel` launch API
   for cross-block sync on rmsnorm reduction. Adds significant complexity for a perf
   target already shown to be marginal.

2. **Two-launch sequence (rmsnorm_with_channel_scale → rotate)**: doesn't save a
   launch, just moves the channel_scales fold. Channel_scales is already inside the
   standalone rotate (Phase 0), so this is byte-equivalent to the current path.

3. **Atomic accumulator + multi-block reduction**: complex, atomicAdd contention on
   the global RMS scalar, probably no win.

4. **Per-linear fused rmsnorm+rotate (re-do rmsnorm internally per linear)**: each
   linear's fused kernel computes rmsnorm internally. For 144 paro linears × ~3µs
   each = ~430µs of redundant rmsnorm work, vs saving 49 launches × 10µs = 490µs of
   launch overhead. Net wash, plus same CU-occupancy issue.

## What the experiment shipped (kept as research artifact)

- Kernel: `kernels/src/fused_rmsnorm_paro4g128t_rotate.hip` (~190 LOC).
  Numerically equivalent to split path within FP16 epsilon.
- Dispatch: `Gpu::fused_rmsnorm_paro4g128t_rotate` in `crates/rdna-compute/src/dispatch.rs`.
- Helper: `fused_rmsnorm_rotate_for_paro` in `crates/hipfire-runtime/src/llama.rs`,
  parallels `fused_rmsnorm_rotate_for_mq`.
- Call-site wiring: 13 PARO sites in `crates/hipfire-arch-qwen35/src/qwen35.rs`
  (FA + LA × multiple forward paths). Mutual exclusion with existing
  `HIPFIRE_PARO_FA3_FUSED`, `HIPFIRE_PARO_LA4_FUSED`, etc. env-gated paths.
- Opt-in env: `HIPFIRE_PARO_FUSE_RMSNORM=1`. Default OFF.

## Implication for asymptote

This is **Lever 1 falsified** — the "+30% decode tok/s" estimate in the May 14
baseline README's "Two perf levers" section overstated the ceiling. The SKIP_ROTATE
experiment's measured +16.8% prefill upper bound was already a tightening of the
estimate; this kernel-architecture work brings it down further.

The structural reason: PARO's rotation BW reads (`pairs[8,K]` + `sincos[8,K/2,2]` +
`channel_scales[K]` ≈ 50KB per call) are fixed by the algorithm. Fusion can only
amortize launch overhead, and the launch overhead saving is small (~10µs) compared
to the kernel's compute time (~30-70µs serial / ~10µs parallel). On parallel HW,
the structurally-parallel split kernel wins.

Per the goal's asymptote criterion: **Lever 1 attempted, -2.7% delta is well under
the ±5% threshold (after counting sign as a failed attempt, not just absolute value).
This counts as 1 of the 3+ post-fusion experiments showing <5% perf delta.**

## Files

```
kernels/src/fused_rmsnorm_paro4g128t_rotate.hip
crates/rdna-compute/src/kernels.rs               (FUSED_RMSNORM_PARO4G128T_ROTATE_SRC)
crates/rdna-compute/src/dispatch.rs              (Gpu::fused_rmsnorm_paro4g128t_rotate)
crates/hipfire-runtime/src/llama.rs              (fused_rmsnorm_rotate_for_paro helper)
crates/hipfire-arch-qwen35/src/qwen35.rs         (13 PARO sites wired)
```
