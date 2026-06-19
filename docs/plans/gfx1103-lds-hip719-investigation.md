# gfx1103 LDS HIP-719 Investigation

Living notes for narrowing the intermittent, sticky HIP-719 launch failure seen
with LDS-backed `gemm_f32_train` variants on gfx1103.

## Scope

- Kernel under investigation: `kernels/src/gemm_f32_train.hip`.
- Reference branch: `chaingun`.
- Risky experiments: throwaway worktrees only.
- Goal: diagnose the failure mechanism well enough to decide whether this is
  a hipfire kernel bug, compiler/codegen bug, or gfx1103 ROCm/amdgpu LDS runtime
  bug. Do not land a production kernel rewrite from this note alone.

## Local Test Target

Observed local hardware/software for the first repro pass:

- GPU: `gfx1103`, AMD Radeon 780M Graphics / Phoenix APU.
- ROCm driver reported by `rocm-smi`: `6.19.0`.
- ROCm tools present:
  - `/opt/rocm/bin/rocprofv3`
  - `/opt/rocm/bin/rocprof-compute`
  - `/opt/rocm/bin/rocgdb`
  - `/opt/rocm/bin/hipcc`
  - `/opt/rocm/llvm/bin/llvm-objdump`
  - `/opt/rocm/llvm/bin/llvm-readobj`
- Local driver source available:
  - `/usr/src/amdgpu-6.19.0-2307534.24.04/amd/amdgpu`
- Full ROCm runtime/compiler source trees are not currently present locally.

## Commit History Examined

Three relevant file states were compared:

- `c3765ea9`: introduced shared-memory tiled GEMM, `TILE=16`, one output per
  thread, `__shared__` A/B tiles.
- `b41368bb`: kept the LDS tiled kernel but removed the second
  `__launch_bounds__` argument after HIP-719 launch failures.
- `5546fe12`: replaced the LDS tiled kernel with a no-LDS register-tiled
  micro-tile kernel after finding the LDS variant unreliable on gfx1103.

Current `chaingun` uses the `5546fe12` no-LDS register-tiled kernel and a
`ceil(N/64) x ceil(M/64)` host launch grid.

## Repro Harness

Initial repro used a throwaway worktree:

```bash
git worktree add --detach /tmp/hipfire-lds-repro HEAD
git -C /tmp/hipfire-lds-repro checkout b41368bb -- kernels/src/gemm_f32_train.hip
# Patch dispatch grid back to ceil(N/16) x ceil(M/16), block 16x16.
source ./scripts/rocm-env.sh 2>/dev/null || true
cargo run -p hipfire-train --release --example gemm_f32_train_recover
```

The recovery harness repeatedly launches:

```rust
gpu.gemm_f32_train(&x, &w, &c, 512, 3072, 3072, 3072, 3072, false, true)
```

and after a launch failure tries:

1. `clear_last_error()`
2. `device_synchronize()`
3. `clear_last_error()`
4. relaunch, up to 8 retries

Current artifact root for this investigation pass:

```text
/tmp/hipfire-lds-artifacts-v2/
```

The runner preserves variant-local `HIPFIRE_KERNEL_CACHE` output, `dmesg`
snapshots, run logs, and generated source/code-object files when the runtime
compiler writes them. `sudo` is available for root-only amdgpu sysfs evidence.

Standalone HIP probes were added in the throwaway worktree only:

- `/tmp/hipfire-lds-repro/lds_standalone_probe.hip`: LDS-only shared-memory
  store/load/barrier stress kernels, no GEMM global matrix traffic.
- `/tmp/hipfire-lds-repro/lds_gemm_standalone_probe.hip`: direct HIP tiled
  GEMM reproducer using the same A/B/C global-memory shape as the hipfire
  training example. Later extended with reduction modes and a compile-time
  synthetic no-global kernel.
- `/tmp/hipfire-lds-repro/lds_rect_active_probe.hip`: rectangular no-output
  LDS probe with independent active and block dimensions. This keeps the
  GEMM-shaped `As[ACTIVE_Y][K_TILE]` / `Bs[K_TILE][ACTIVE_X]` pattern while
  separating exact one-wave blocks, active lanes, inactive lanes, and
  multi-wave barriers.
- `/tmp/hipfire-lds-repro/lds_direct_active_probe.hip`: rectangular no-output
  LDS probe with independent active and block dimensions, but direct per-lane
  stores into one LDS array. This removes the cooperative A/B staging loops so
  block shape and active masks can be compared without extra producer
  iterations.
- `/tmp/hipfire-lds-repro/lds_direct_ab_probe.hip`: rectangular no-output LDS
  probe with direct per-lane stores into two LDS arrays. This preserves a
  two-slab A/B-like footprint without cooperative producer loops.

Their artifact roots are:

```text
/tmp/hipfire-lds-standalone-artifacts/
/tmp/hipfire-lds-standalone-artifacts-v2/
/tmp/hipfire-lds-gemm-standalone-artifacts/
/tmp/hipfire-lds-rect-active-artifacts/
/tmp/hipfire-lds-direct-active-artifacts/
/tmp/hipfire-lds-direct-ab-artifacts/
```

## Variant Matrix

All rows below use the same gfx1103 Phoenix APU unless noted.

| Variant | Result | Notes |
|---|---:|---|
| b413 LDS `TILE=16`, 16x16 block | FAIL | Unrecoverable at launch 11 after 8 retries. |
| b413 LDS `TILE=16` with `TILE+1` padded LDS rows | FAIL | Unrecoverable at launch 13. Padding does not fix it. |
| A-only LDS, B direct global | FAIL | Unrecoverable at launch 8. |
| B-only LDS, A direct global | FAIL | Unrecoverable at launch 8. |
| LDS `TILE=8`, 8x8 block | FAIL | Unrecoverable at launch 7. |
| LDS `TILE=6`, 6x6 block | FAIL | Unrecoverable at launch 5. |
| LDS `TILE=5`, 5x5 block | PASS | 100 launches, 0 retries. |
| LDS `TILE=4`, 4x4 block | PASS | 100 launches, 0 retries. |
| 4x4 active LDS subset inside 8x8 block | PASS | 100 launches, 0 retries; barriers span 64 threads, only 16 lanes touch LDS. |
| Standalone LDS-only `TILE=6`, 6x6 block | PASS | 100 direct HIP launches, 64x64 grid, no dmesg delta. |
| Standalone LDS-only `TILE=8`, 8x8 block | PASS | 100 direct HIP launches, 64x64 grid, no dmesg delta. |
| Standalone LDS-only `TILE=16`, 16x16 block | PASS | 100 direct HIP launches after fixing an output-allocation bug in the probe; no new dmesg delta. |
| Standalone LDS-only `TILE=6`, 128 iterations, 448x86 grid | PASS | Large grid alone is not enough with the short loop. |
| Standalone HIP GEMM `TILE=5`, 5x5 block | PASS | 100 direct HIP launches at M=512, N=3072, K=3072; no dmesg delta. |
| Standalone HIP GEMM `TILE=6`, 6x6 block | FAIL | Direct HIP, no hipfire Rust/JIT path; sync 20 failed with HIP 719 and MES reset. |
| Standalone GEMM reduction `TILE=6`, no C store | FAIL | C/global output write is not required; failed with HIP 719. |
| Standalone GEMM reduction `TILE=6`, A-only | FAIL | B global load is not required; failed with HIP 719. |
| Standalone GEMM reduction `TILE=6`, B-only | FAIL | A global load is not required; failed with HIP 719. |
| Standalone GEMM reduction `TILE=6`, no global/no store | FAIL | Runtime-mode version failed; global memory access is not required. |
| Standalone GEMM synthetic `TILE=6`, no global/no store, K=1536 | PASS | 100 launches at M=512, N=3072. |
| Standalone GEMM synthetic `TILE=6`, no global/no store, K=2048 | FAIL | Repeated failure, sync ~82-83; same MES/GDS fault state. |
| Standalone GEMM synthetic `TILE=5`, no global/no store, K=3072 | PASS | 100 launches at M=512, N=3072. |
| Standalone LDS-only `TILE=6`, 512 iterations | PASS | 100 launches at 64x64 grid; long LDS loop alone still passes. |
| Standalone GEMM synthetic `TILE=6`, M=512, K=2048, N=2496 | PASS | Grid around 416x86 blocks. |
| Standalone GEMM synthetic `TILE=6`, M=512, K=2048, N=2688 | FAIL | Grid around 448x86 blocks; same MES/GDS fault state. |
| Standalone GEMM synthetic `TILE=6`, M=512, N=2688, K=2048, 90 launches | PASS | Same reduced no-global/no-store kernel; launch-count threshold control. |
| Standalone GEMM synthetic `TILE=6`, M=512, N=2688, K=2048, 95 launches | MIXED | Earlier launch-repeat artifact failed at sync 94; fresh launch-bisect artifact passed. |
| Standalone GEMM synthetic `TILE=6`, M=512, N=2688, K=2048, 96 launches | FAIL | Fresh launch-bisect artifact failed at sync 93 after a reset. |
| Standalone GEMM synthetic `TILE=6`, M=512, N=2688, K=2048, 100 launches | FAIL | Fresh launch-bisect artifact failed at sync 95. |
| Standalone GEMM synthetic `TILE=6`, M=512, N=2880, K=2048 | FAIL | 100-launch shape repeat failed at sync 87. |
| Standalone GEMM synthetic `TILE=6`, M=512, N=3072, K=2048 | FAIL | 100-launch shape repeat failed at sync 81. |
| Standalone GEMM synthetic masked `TILE=6`, M=512, N=2688, K=2048 | FAIL | Exec-mask regions were emitted around active LDS regions; still failed at sync 95. |
| Standalone LDS-only `TILE=6`, 512 iterations, 288x86 grid | PASS | Simple LDS-only threshold control. |
| Standalone LDS-only `TILE=6`, 512 iterations, 297x86 grid | PASS | Tight grid-edge control at grid_y=86. |
| Standalone LDS-only `TILE=6`, 512 iterations, 298x86 grid | FAIL | Tight grid-edge repro; failed at sync 98. |
| Standalone LDS-only `TILE=6`, 512 iterations, 304x86 grid | FAIL | Simple LDS-only repro; failed at sync 97. |
| Standalone LDS-only `TILE=6`, 512 iterations, 320x86 grid | FAIL | Failed at sync 90; same coredump signature. |
| Standalone LDS-only `TILE=6`, 512 iterations, 448x86 grid | FAIL | Grid-matched to synthetic N=2688/M=512; failed at sync 64. |
| Standalone LDS-only `TILE=5`, 512 iterations, 512x86 grid | PASS | One-wave control still passes at a larger grid than the `TILE=6` failing edge. |
| Standalone LDS-only `TILE=6`, 256 iterations, 512x86 grid | PASS | Loop-depth correlate for synthetic `K_LIMIT=1536` pass. |
| Standalone LDS-only `TILE=6`, 320 iterations, 512x86 grid | FAIL | Loop-depth correlate; failed at sync 87 with same coredump signature. |
| Standalone LDS-only `TILE=6`, 336 iterations, 512x86 grid | FAIL | Loop-depth correlate near synthetic `K_LIMIT=2048`; failed at sync 84. |
| Standalone LDS-only `TILE=6`, 320 iterations, 448x86 grid | PASS | Tight iteration-edge control at full grid. |
| Standalone LDS-only `TILE=6`, 336 iterations, 448x86 grid | FAIL | Tight iteration-edge repro; failed at sync 98. |
| Minimal no-output LDS-only `TILE=5`, 512 iterations, 512x86 grid | PASS | Single-instantiation no-output control. |
| Minimal no-output LDS-only `TILE=6`, 256 iterations, 512x86 grid | PASS | No host allocations or global stores; mirrors `K_LIMIT=1536` pass. |
| Minimal no-output LDS-only `TILE=6`, 320 iterations, 512x86 grid | FAIL | No host allocations or global stores; failed at sync 86. |
| Minimal no-output LDS-only `TILE=6`, 336 iterations, 512x86 grid | FAIL | No host allocations or global stores; failed at sync 84. |
| Minimal no-output LDS-only `TILE=6`, 320 iterations, 448x86 grid | PASS | Preserves the 448x86 loop-depth edge without global stores. |
| Minimal no-output LDS-only `TILE=6`, 336 iterations, 448x86 grid | FAIL | Preserves the 448x86 loop-depth edge; failed at sync 96. |
| Standalone LDS-only no-mask `TILE=6`, 512 iterations, 288x86 grid | PASS | Removes exec-mask regions; same pass side as masked control. |
| Standalone LDS-only no-mask `TILE=6`, 512 iterations, 304x86 grid | FAIL | Removes exec-mask regions; failed at sync 98. |
| Rect-active no-output `6x6` active/block, K=6, 320 iterations, 512x86 grid | FAIL | Rectangular probe baseline; failed at sync 87. No global load/store ISA. |
| Rect-active no-output `6x6` active/block, K=6, 256 iterations, 512x86 grid | PASS | Low side of the rectangular K=6 loop-depth edge. |
| Rect-active no-output `6x6` active/block, K=6, 272 iterations, 512x86 grid | PASS | Same code-object resource metadata as the 280-fail case. |
| Rect-active no-output `6x6` active/block, K=6, 280 iterations, 512x86 grid | FAIL | Same code-object resource metadata as the 272-pass case; failed at sync 99. |
| Rect-active no-output `6x6` active/block, K=5, 320/336/384 iterations, 512x86 grid | PASS | K=5 shifts the same all-active 6x6 threshold upward. |
| Rect-active no-output `6x6` active/block, K=5, 416/448/512 iterations, 512x86 grid | FAIL | K=5 edge is between 384 and 416 iterations; 416 failed at sync 91. |
| Rect-active no-output `8x4` active/block, K=6, 336 iterations, 512x86 grid | PASS | Exactly one wave, K=6, code-object LDS segment 288 B. |
| Rect-active no-output `8x4` active/block, K=6, 512 iterations, 512x86 grid | PASS | Exact one-wave K=6 control remains stable at longer loop depth. |
| Rect-active no-output `8x4` active inside `8x5` block, K=6, 336 iterations, 512x86 grid | PASS | Two-wave block and barriers with only 32 active LDS lanes; code-object LDS segment 288 B. |
| Rect-active no-output `8x4` active inside `8x5` block, K=6, 512 iterations, 512x86 grid | FAIL | Same code object as the 336-pass control; failed at sync 94 after longer loop work. |
| Rect-active no-output `7x5` active/block, K=6, 320 iterations, 512x86 grid | FAIL | 35 active lanes, K=6; failed at sync 74. |
| Rect-active no-output `7x5` active/block, K=6, 336 iterations, 512x86 grid | FAIL | Same shape; failed at sync 71. |
| Rect-active no-output `5x5` active inside `6x6` block, K=5, 512 iterations, 512x86 grid | PASS | Two-wave block, 25 active lanes, K=5 control. |
| Rect-active no-output `5x5` active/block, K=6, 512 iterations, 512x86 grid | PASS | One-wave block, 25 active lanes, K=6 control. |
| Rect-active no-output `5x5` active inside `6x6` block, K=6, 320 iterations, 512x86 grid | FAIL | Two-wave block, 25 active lanes, K=6; failed at sync 50. |
| Rect-active no-output `5x5` active inside `6x6` block, K=6, 336 iterations, 512x86 grid | FAIL | Same shape; failed at sync 47. |
| Rect-active no-output `5x5` active inside `6x6` block, K=6, 512 iterations, 512x86 grid | FAIL | Same shape; failed at sync 31. |
| Rect-active no-output `6x4` active inside `6x6` block, K=6, 320 iterations, 512x86 grid | FAIL | 24 active lanes; failed at sync 98. |
| Rect-active no-output `4x6` active inside `6x6` block, K=6, 320 iterations, 512x86 grid | FAIL | Transposed 24-active control; failed at sync 75. |
| Rect-active no-output `4x4` active inside `6x6` block, K=6, 320 iterations, 512x86 grid | FAIL | 16 active lanes; failed at sync 65. |
| Rect-active no-output `4x4` active inside `6x6` block, K=5, 320 iterations, 512x86 grid | FAIL | 16 active lanes; failed at sync 67. |
| Direct-active no-output `6x6` active/block, K=6, 320/384/448/464 iterations, 512x86 grid | PASS | Direct per-lane store shifts the all-active 6x6 threshold upward. |
| Direct-active no-output `6x6` active/block, K=6, 480/512 iterations, 512x86 grid | FAIL | Edge is between 464 and 480 iterations; 512 failed at sync 91. |
| Direct-active no-output `6x6` active/block, K=5, 512 iterations, 512x86 grid | PASS | K=5 control for the direct-store source. |
| Direct-active no-output `8x4` active/block, K=6, 512 iterations, 512x86 grid | PASS | Exact one-wave direct-store control. |
| Direct-active no-output `8x4` active inside `8x5` block, K=6, 512 iterations, 512x86 grid | PASS | Two-wave block with 32 active lanes; cooperative-loader version failed at 512. |
| Direct-active no-output `4x4` active inside `6x6` block, K=6, 512 iterations, 512x86 grid | PASS | Small-active control; cooperative-loader version failed at 320. |
| Direct-active no-output `5x5` active inside `6x6` block, K=6, 512 iterations, 512x86 grid | PASS | Small-active control; cooperative-loader version failed at 320/336/512. |
| Direct-active no-output `6x4`/`4x6` active inside `6x6` block, K=6, 512 iterations, 512x86 grid | PASS | 24-active orientation controls; cooperative-loader versions failed at 320. |
| Direct-AB no-output `6x6` active/block, reads=1/2, 512 iterations, 512x86 grid | PASS | Same 288 B LDS footprint as failing reads=3+ controls; footprint alone is not enough. |
| Direct-AB no-output `6x6` active/block, reads=3, 384 iterations, 512x86 grid | PASS | Low side of reads=3 edge. |
| Direct-AB no-output `6x6` active/block, reads=3, 448/512 iterations, 512x86 grid | FAIL | Reads=3 edge is between 384 and 448; same coredump signature. |
| Direct-AB no-output `6x6` active/block, reads=5, 192/224 iterations, 512x86 grid | PASS | Reads=5 low side. |
| Direct-AB no-output `6x6` active/block, reads=5, 256/320 iterations, 512x86 grid | FAIL | Reads=5 edge is between 224 and 256. |
| Direct-AB no-output `6x6` active/block, reads=6, 176 iterations, 512x86 grid | PASS | Reads=6 low side. |
| Direct-AB no-output `6x6` active/block, reads=6, 192/256/320 iterations, 512x86 grid | FAIL | Reads=6 edge is between 176 and 192. |
| Direct-AB no-output `6x6` active/block, reads=6, 192 iterations, 509x86 grid | PASS | Grid-width low side at fixed read/loop edge. |
| Direct-AB no-output `6x6` active/block, reads=6, 192 iterations, 510x86 grid | MIXED | Earlier grid sweep failed at sync 99; fresh launch-count replay passed through 100 launches. Treat exact 510x86 edge as reset/state-sensitive. |
| Direct-AB no-output `6x6` active/block, reads=6, 192 iterations, 511/512x86 grid | FAIL | Fresh replay failed at sync 99 for 100 requested launches. |
| Direct-AB no-output `6x6` active/block, reads=6, 192 iterations, 511x86 grid, 96-98 launches | PASS | Launch-count low side at the fresh grid edge. |
| Direct-AB no-output `6x6` active/block, reads=6, 192 iterations, 511x86 grid, 99 launches | MIXED | Passed in the first launch-count sweep, then failed at sync 98 after reset pressure with the reused binary. |
| Direct-AB no-output `6x6` active/block, reads=6, 192 iterations, 511x86 grid, 100 launches | FAIL | Launch-count high side; failed at sync 98-99 with the same gfxhub/GDS coredump signature. |
| Direct-AB no-output `6x6` active/block, reads=6, 192 iterations, 511x86 grid, split-process 98+1 launches | PASS | Same reused binary; three trials passed both the 98-launch process and the follow-up 1-launch process. |
| Direct-AB no-output `6x6` active/block, reads=6, 192 iterations, 511x86 grid, one-process 99 launches after split controls | FAIL | Same reused binary and total launch count as split 98+1; failed at sync 98. |
| Direct-AB phase-mode `6x6`, reads=6, 192 iterations, 511x86 grid, same-process 98+1 | PASS | Phase-mode runner, null stream, extra boundary synchronize; total 99 passed. |
| Direct-AB phase-mode `6x6`, reads=6, 192 iterations, 511x86 grid, same-process 99+0 | PASS | Phase-mode runner, total 99 passed before the edge shifted again. |
| Direct-AB phase-mode `6x6`, reads=6, 192 iterations, 511x86 grid, same-process 99+1 | FAIL | Failed on phase2 launch 0 / global launch 99. |
| Direct-AB phase-mode `6x6`, reads=6, 192 iterations, 511x86 grid, same-process 98+2 | MIXED | Earlier preserved repeats failed 2/2 at phase2 launch 1 / global launch 99; later confirmation passed. The edge is state-sensitive. |
| Direct-AB phase-mode `6x6`, reads=6, 192 iterations, 511x86 grid, device-reset 98+2 | FAIL | `hipDeviceReset()` between phases returned success, but phase2 launch 1 / global launch 99 still failed. |
| Direct-AB phase-mode `6x6`, reads=6, 192 iterations, 511x86 grid, stream-recreate 98+2 | FAIL | Destroying phase1 stream and creating phase2 stream did not clear the edge. |
| Direct-AB phase-mode `6x6`, reads=6, 192 iterations, 511x86 grid, same-stream 98+2 | MIXED | Explicit non-default stream was state-sensitive: one preserved pass and three preserved failures. |
| Direct-AB phase-mode `6x6`, reads=6, 192 iterations, 511x86 grid, cross-process 98+2 | PASS | Two trials passed `98+0` in one process followed by `2+0` in a new process. |
| Direct-AB phase-mode `6x6`, reads=6, 192 iterations, 511x86 grid, primary-ctx reset/release 98+2 | FAIL | Deprecated `hipDevicePrimaryCtxReset(0)` and `hipDevicePrimaryCtxRelease(0)` returned success, but phase2 launch 1 / global launch 99 still failed. |
| Direct-AB phase-mode `6x6`, reads=6, 192 iterations, 511x86 grid, HSA shutdown 98+2 | CRASH | Direct `hsa_shut_down()` modes segfaulted or made `hsa_init()` fail; not a clean in-process teardown lever for HIP here. |
| Direct-AB exec-parent `6x6`, reads=6, 192 iterations, 511x86 grid, child-process 98+2 | PASS | Parent process survived across both phases; phase1 and phase2 ran via fork/exec children. Plain parent, HIP-initialized parent, parent reset-before, and parent reset-between modes all passed. |
| Direct-AB phase-mode confirmation `6x6`, reads=6, 192 iterations, 511x86 grid, same-process 100+0/100+1/101+0/101+1 | FAIL | Current calibration failed at phase1 sync 99 / global launch 99, while same-process 99+1 passed later. |
| Direct-AB exec-parent confirmation `6x6`, reads=6, 192 iterations, 511x86 grid, child-process 99+1 | PASS | Parent plain and HIP-initialized modes both passed. |
| Direct-AB no-output `6x6` active/block, reads=3, 448 iterations, 511x86 grid | PASS | Grid-width low side at fixed read/loop edge. |
| Direct-AB no-output `6x6` active/block, reads=3, 448 iterations, 512x86 grid | MIXED | Fails on repeat, but the exact launch edge moves with reset/GPU state. |
| Direct-AB no-output `6x6` active/block, reads=3, 448 iterations, 512x86 grid, 94-99 launches | PASS | Initial launch-count sweep low side. |
| Direct-AB no-output `6x6` active/block, reads=3, 448 iterations, 512x86 grid, 100 launches | MIXED | Initial run passed; deliberate repeat after reset pressure failed at sync 99. |
| Direct-AB no-output `6x6` active/block, reads=3, 448 iterations, 512x86 grid, 110/120/130/150 launches | FAIL | Extended launch-count sweep failed around sync 98-101. |
| Direct-AB phase-mode `6x6`, reads=3, 448 iterations, 512x86 grid, same-process 99+0/99+1/100+0 | PASS | Second-edge phase probe; low side remained stable before reset pressure moved the edge. |
| Direct-AB phase-mode `6x6`, reads=3, 448 iterations, 512x86 grid, same-process 100+1/101+0 | FAIL | Failed during phase1 at sync 98 / 97, showing the edge had shifted before the explicit phase boundary. |
| Direct-AB phase-mode `6x6`, reads=3, 448 iterations, 512x86 grid, same-process 99+2 | PASS | Total 101 passed after nearby failures; exact counters remain state-sensitive. |
| Direct-AB phase-mode `6x6`, reads=3, 448 iterations, 512x86 grid, same-process 98+3 | FAIL | Phase1 completed, boundary sync succeeded, then phase2 launch 1 / global launch 99 failed with HIP 719. |
| Direct-AB exec-parent `6x6`, reads=3, 448 iterations, 512x86 grid, child-process 98+3, plain parent | PASS | First run and repeat both passed with phase1 and phase2 in fork/exec children. |
| Direct-AB exec-parent `6x6`, reads=3, 448 iterations, 512x86 grid, child-process 98+3, HIP-initialized parent | MIXED | First trial failed inside the phase1 child at sync 97; repeat passed. Treat as edge state sensitivity, not deterministic parent-state retention. |
| Direct-AB exec-parent `6x6`, reads=3, 448 iterations, 512x86 grid, child-process 98+3, HIP-initialized parent reset-between | PASS | Parent `hipDeviceReset()` between children returned success and both child phases passed. |
| Direct-AB phase-mode `6x6`, reads=3, 448 iterations, 512x86 grid, same-process 96+5 | FAIL | Lower-risk split: phase1 completed, boundary sync succeeded, then phase2 launch 2 / global launch 98 failed with HIP 719. |
| Direct-AB phase-mode `6x6`, reads=3, 448 iterations, 512x86 grid, same-process 97+4 | PASS | Same total launch count as 96+5, confirming ordering/state sensitivity near the edge rather than a simple total counter. |
| Direct-AB exec-parent `6x6`, reads=3, 448 iterations, 512x86 grid, child-process 96+5 | PASS | Plain, HIP-initialized, and HIP-initialized reset-between parent modes passed; repeat plain and hipinit trials also passed. |
| Direct-AB coredump repeat `6x6`, reads=3, 448 iterations, 512x86 grid, same-process 96+5 / 100+1 | PASS | After clearing generic devcoredump state, both repeat controls passed; another example of edge movement after reset/coredump pressure. |
| Direct-AB coredump repeat `6x6`, reads=3, 448 iterations, 512x86 grid, same-process 110+0 | FAIL | After clearing stale devcoredump state, failed at phase1 sync 99 / global launch 99. A fresh generic `devcd28` node appeared late and captured the same gfxhub/GDS signature. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 512x86 grid, one child `101` | FAIL | Parent ran one fork/exec child with 101 launches; child failed at sync/global launch 100 and late `devcd29` captured the same gfxhub/GDS signature. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 512x86 grid, chunks `96,5` | PASS | Same total launches as one-child `101`, but phase work split across two child processes. Passed for plain and HIP-initialized parent modes. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 512x86 grid, chunks `50,30,21` | PASS | Same total launches as one-child `101`, split across three child processes. Passed for plain and HIP-initialized parent modes. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 511x86 grid, one child `90`/`95`/`96`/`98` | PASS | Lower-grid one-child low side after reset pressure. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 511x86 grid, one child `99`/`100`/`101`/`102` | FAIL | Lower-grid bracket after reset pressure; 99 and 100 failed at sync/global launch 98, 101 and 102 at sync/global launch 99. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 511x86 grid, one child `120` | FAIL | Lower-grid replay; one fork/exec child failed at sync/global launch 101 and late `devcd30` captured the same gfxhub/GDS signature. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 511x86 grid, chunks `96,24` / `60,60` | PASS | Same total launches as one-child `120`, split across child processes. Both split shapes passed for plain and HIP-initialized parent modes. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 510x86 grid, one child `99` | PASS | Next lower grid low side. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 510x86 grid, one child `100`/`120` | FAIL | One-child `120` failed at sync/global launch 99; follow-up `100` failed at sync/global launch 96. Both produced the same coredump signature. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 510x86 grid, chunks `96,24` / `60,60` | PASS | Same total launches as one-child `120`, split across child processes. Both split shapes passed for plain and HIP-initialized parent modes. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 509x86 grid, one child `90`/`95`/`98` | PASS | Next lower grid low side. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 509x86 grid, one child `99`/`100` | FAIL | One-child `100` failed first at sync/global launch 99; low-to-high sweep then found 98 pass / 99 fail, with 99 failing at sync/global launch 97. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 509x86 grid, chunks `96,24` / `60,60` | PASS | Same total launches as one-child `120`-style controls, split across child processes. Both split shapes passed for plain and HIP-initialized parent modes. |
| Direct-AB no-output `8x4` active/block, reads=6, 512 iterations, 512x86 grid | PASS | Exact one-wave, two-array control. |
| Direct-AB no-output `8x4` active inside `8x5` block, reads=6, 512 iterations, 512x86 grid | PASS | Two-wave block with 32 active lanes; still stable. |
| Direct-AB no-output `5x5`/`4x4` active inside `6x6` block, reads=6, 512 iterations, 512x86 grid | PASS | Small active controls remain stable without cooperative producer loops. |

Latest artifact paths:

- `/tmp/hipfire-lds-artifacts-v2/tile5_t5_b5_n100/`: pass case, includes run
  log, saved kernel source, `gemm_f32_train.hsaco`, metadata, and ISA dump.
- `tile6_t6_b6_n100/`: fail case, includes run log, saved kernel source, and
  generated `gemm_f32_train.hsaco` metadata/ISA dumps. Also includes a
  root-copied `devcoredump.data` sample from
  `/sys/class/drm/card0/device/devcoredump/data`.
- `active4_block8_t4_b8_n100/`: pass control, 4x4 active LDS subset inside
  8x8 block. This run did not leave `gemm_f32_train.hsaco`; it wrote only the
  runtime source and hash under the variant-local cache, so exact code-object
  comparison for this control still needs runner cleanup.
- `tile6_dmesg_probe/`: `dmesg.before.txt` and `dmesg.after.txt` around the
  failing `tile6` run.
- `/tmp/hipfire-lds-gemm-standalone-artifacts/tile6_n100_m512_n3072_k3072/`:
  direct HIP standalone GEMM reproducer. Includes generated object/ISA dumps,
  dmesg snapshots, final dmesg tail, and a root-copied `devcoredump.data`.
- `/tmp/hipfire-lds-gemm-klimit-repeat-artifacts/`: repeated no-global/no-store
  K-limit sweep, including pass at `K_LIMIT=1536`, repeated failures at
  `K_LIMIT=2048`, and tile5 pass at full K.
- `/tmp/hipfire-lds-gemm-synth-shape-artifacts2/`: compile-time synthetic
  no-global/no-store shape sweep, including pass at N=2496 and failure at
  N=2688 for M=512, K_LIMIT=2048.
- `/tmp/hipfire-lds-gemm-shape-repeat-artifacts/`: preserved shape repeat for
  the reduced no-global/no-store synthetic kernel. At 100 launches, N=2496
  passes; N=2688, 2880, and 3072 fail at sync 95, 87, and 81 respectively.
- `/tmp/hipfire-lds-gemm-launch-repeat-artifacts/`: preserved launch-count
  repeat at M=512, N=2688, K_LIMIT=2048. 80, 85, and 90 launches pass; 95 and
  100 launches both fail at sync 94.
- `/tmp/hipfire-lds-gemm-launch-bisect-artifacts/`: fresh launch-count edge
  check at the same shape. 91, 92, 93, 94, and 95 launches pass; 96 launches
  fails at sync 93 and 100 launches fails at sync 95. The 96-launch artifact
  includes a manually copied `devcoredump.data` sample via passwordless sudo.
- `/tmp/hipfire-lds-gemm-mask-artifacts/`: throwaway masked synthetic variant
  at M=512, N=2688, K_LIMIT=2048. The compiler emitted exec-mask instructions
  around the active LDS regions, but 100 launches still failed at sync 95 and
  produced the same GDS/GDS-VM coredump signature.
- `/tmp/hipfire-lds-standalone-long-artifacts/`: passing long-loop LDS-only
  control, including `TILE=6`, 512 iterations, 100 launches at 64x64 grid.
- `/tmp/hipfire-lds-standalone-gridmatch-artifacts/`: grid/work sweep for the
  existing masked LDS-only `TILE=6`, 512-iteration control. At 100 launches
  and grid_y=86, grid_x 192, 256, and 288 pass; grid_x 304, 320, 384, 416,
  and 448 fail. Failure moves earlier as the grid grows: sync 97 at 304x86,
  sync 90 at 320x86, sync 75 at 384x86, sync 68 at 416x86, sync 64 at 448x86.
  The short 128-iteration `tile6` control passes at 448x86.
- `/tmp/hipfire-lds-standalone-nomask-artifacts/`: no-mask LDS-only controls.
  `tile6_i512_nomask` matches the masked threshold: 288x86 passes, 304x86
  fails at sync 98. `tile6_nomask` with 128 iterations passes at 448x86.
- `/tmp/hipfire-lds-standalone-grid-bisect-artifacts/`: tight grid-edge
  bisect for masked `tile6_i512` at grid_y=86. At 100 launches, grid_x 296
  and 297 pass; grid_x 298 and 300 fail at sync 98. The 298x86 and 300x86
  artifacts include root-copied coredumps with the same gfxhub/GDS signature.
- `/tmp/hipfire-lds-standalone-iter-artifacts/`: iteration-depth sweep for
  masked `tile6` at grid 448x86. At 100 launches, 256 and 320 iterations pass;
  336, 352, and 384 iterations fail at sync 98, 91, and 84 respectively. The
  failing artifacts include root-copied coredumps with the same signature.
- `/tmp/hipfire-lds-standalone-correlate-artifacts/`: correlation sweep at
  grid 512x86. `tile5_i512` passes, preserving the one-wave control at large
  grid. `tile6_i256` passes, while `tile6_i320` and `tile6_i336` fail at sync
  87 and 84 respectively. The `tile6_i320` rerun has a coredump captured
  immediately after the failure; it matches the same gfxhub/GDS signature.
- `/tmp/hipfire-lds-minimal-artifacts/`: single-instantiation minimal
  no-output kernel. It has no host allocations, no global-memory kernel
  arguments, no final global store, no `s_and_saveexec`, and no global
  load/store instructions in ISA. It preserves the correlation: tile5/512
  passes at 512x86, tile6/256 passes at 512x86, tile6/320 and tile6/336 fail
  at 512x86, and the 448x86 edge remains tile6/320 pass vs tile6/336 fail.
- `/tmp/hipfire-lds-rect-active-artifacts/`: rectangular no-output probe with
  separate active and launched block dimensions. This is the first split that
  separates exact one-wave K=6 from multi-wave K=6: `8x4` active/block passes
  even at 512 iterations, while `8x4` active inside an `8x5` two-wave block
  passes at 336 iterations but fails at 512. It also shows K-depth matters:
  `5x5` active inside a `6x6` block passes with K=5 at 512 iterations but
  fails with K=6 at 320/336/512; `5x5` active/block with K=6 passes at 512.
  The all-active `6x6` source gives a tighter K-depth comparison: K=6 passes
  at 272 iterations and fails at 280 with identical resource metadata, while
  K=5 passes at 384 and fails at 416.
- `/tmp/hipfire-lds-direct-active-artifacts/`: direct per-lane no-output LDS
  probe. It removes the cooperative A/B producer loops and uses one LDS array.
  All small active-in-6x6 controls that failed under cooperative staging pass
  here at 512 iterations. All-active `6x6`, K=6 still fails, but the threshold
  moves upward: 464 iterations passes and 480/512 fail. The 512 failure has the
  same gfxhub/GDS coredump signature.
- `/tmp/hipfire-lds-direct-ab-artifacts/`: two-array direct per-lane no-output
  LDS probe. It keeps a 288 B A/B-like LDS footprint for `6x6` while removing
  cooperative producer loops. Footprint alone is not sufficient: reads=1 and
  reads=2 pass at 512 iterations. Read traffic shifts the threshold: reads=3
  passes at 384 and fails at 448, reads=5 passes at 224 and fails at 256, and
  reads=6 passes at 176 and fails at 192. Failures keep the same gfxhub/GDS
  coredump signature. Grid-width sweeps at fixed edge points show sharp total
  work thresholds: reads=6/192 passes through 509x86 and fails at 511x86 in
  the fresh replay, while the exact 510x86 edge is mixed across runs.
  Reads=3/448 passes through 511x86 and fails on repeat at 512x86. Launch-count
  controls are preserved in the same root: for reads=6/192/511x86, 99 launches
  pass and 100 launches fail at sync 99; for reads=3/448/512x86, 99 launches
  pass, 100 launches is mixed, and longer requested runs fail around sync
  98-101. The deliberate reads=3 100-launch repeat overwrote the earlier
  pass artifact, so use the 99-pass and 100-fail directories for preserved
  low/high artifacts.
- `/tmp/hipfire-lds-direct-ab-split-artifacts/`: reused-binary split-process
  controls for the reads=6/192/511x86 edge. The setup compile artifact is
  `a6x6_b6x6_r6_i192_n99_g511x86/` and uses the same code object for all
  follow-up runs. A one-process 100-launch run failed at sync 98. After reset
  pressure, the 99-launch edge lowered and a one-process 99-launch run failed
  at sync 98 with the same gfxhub/GDS signature. In contrast, three
  split-process `98 + 1` trials passed both halves. This is the strongest
  current evidence that the immediate edge is tied to same-process/HIP queue
  lifetime or same-queue dispatch sequence, not just total LDS work submitted
  across a process boundary.
- `/tmp/hipfire-lds-direct-ab-phase-artifacts/`: phase-mode direct-AB probe
  artifacts. The kernel body matches the direct-AB no-output source; only host
  launch sequencing changes. At reads=6/192/511x86, same-process `99 + 1`
  fails on phase2 launch 0 / global launch 99, and same-process `98 + 2`
  fails on phase2 launch 1 / global launch 99. `hipDeviceReset()` between
  `98 + 2` phases returns success but does not clear the edge; stream
  destroy/recreate also does not clear it. Cross-process `98+0` followed by
  `2+0` passes in two trials using the same phase-probe binary.
- `/tmp/hipfire-lds-direct-ab-phase-repeat-artifacts/`: preserved repeats for
  phase-mode `98 + 2`. Default/null-stream same-process mode failed 2/2 at
  phase2 launch 1 / global launch 99. Explicit same-stream mode was mixed:
  one preserved pass and three preserved failures, so stream mode changes the
  state sensitivity but is not a reliable fix or root-cause discriminator.
- `/tmp/hipfire-lds-direct-ab-exec-artifacts/`: exec-parent wrapper controls
  for phase-mode `98 + 2`. The parent process stays alive across both phases
  and runs phase1/phase2 through fork/exec child processes. Plain parent,
  HIP-initialized parent, HIP-initialized parent with `hipDeviceReset()` before
  children, and HIP-initialized parent with `hipDeviceReset()` between children
  all pass. This means parent process lifetime, even with an initialized HIP
  context, is not enough to retain the bad state; the HIP-launching process
  exiting between phases is the meaningful boundary.
- `/tmp/hipfire-lds-direct-ab-exec-confirm-artifacts/`: current-edge
  confirmation around the exec-parent result. Same-process phase-mode `100+0`,
  `100+1`, `101+0`, and `101+1` all fail at phase1 sync 99 / global launch 99
  with the same gfxhub/GDS signature captured for `100+1` and `101+0`.
  Same-process `99+1` passed later, reinforcing that the exact boundary is
  state-sensitive. Exec-parent `99+1` passed in both plain-parent and
  HIP-initialized-parent modes.
- `/tmp/hipfire-lds-direct-ab-teardown-artifacts/`: in-process teardown API
  checks at the active reads=6/192/511x86 `98 + 2` edge. `same` and
  `hipDeviceReset()` controls both fail on phase2 launch 1 / global launch 99.
  Deprecated `hipDevicePrimaryCtxReset(0)` and
  `hipDevicePrimaryCtxRelease(0)` both return success but still fail on the
  same phase2/global launch with the same gfxhub/GDS signature. Direct
  `hsa_shut_down()` inside the HIP process is not usable as a clean teardown
  lever here: `hsa_shutdown`, `hsa_shutdown_init`, and
  `hsa_shutdown_hip_reset` all terminate with SIGSEGV or leave `hsa_init()`
  returning `HSA_STATUS_ERROR_OUT_OF_RESOURCES`.
- `/tmp/hipfire-lds-direct-ab-second-edge-artifacts/`: second-edge phase-mode
  and exec-parent controls for reads=3/448/512x86. Same-process `100+1` and
  `101+0` failed during phase1 at sync 98/97, `99+2` passed, and `98+3`
  failed after a clean phase boundary at phase2 launch 1 / global launch 99.
  Exec-parent `98+3` passed with a plain parent and with a HIP-initialized
  parent that reset between children. One HIP-initialized-parent trial failed
  inside the first child at sync 97, so preserve it as a state-sensitivity
  artifact rather than treating it as a clean parent-lifetime result.
- `/tmp/hipfire-lds-direct-ab-second-edge-rerun-artifacts/`: repeat
  exec-parent controls for reads=3/448/512x86 `98+3`. Both plain parent and
  HIP-initialized parent passed, confirming that the earlier hipinit-parent
  failure is not deterministic.
- `/tmp/hipfire-lds-direct-ab-lower-split-artifacts/`: lower-risk
  reads=3/448/512x86 split controls. Same-process `96+5` failed after a clean
  phase boundary at phase2 launch 2 / global launch 98, while same-process
  `97+4` passed. Exec-parent `96+5` passed in plain, HIP-initialized, and
  HIP-initialized reset-between parent modes. No fresh coredump was captured
  because the devcoredump sysfs node was absent at copy time, but the dmesg
  delta captured the same `REMOVE_QUEUE` failure and MES reset-begin path.
- `/tmp/hipfire-lds-direct-ab-lower-split-repeat-artifacts/`: repeat
  exec-parent controls for reads=3/448/512x86 `96+5`. Plain parent and
  HIP-initialized parent both passed again.
- `/tmp/hipfire-lds-direct-ab-coredump-artifacts/`: explicit generic
  devcoredump clearing/capture pass for reads=3/448/512x86. The existing
  generic devcoredump node was freed with a write to its `data` file, then
  same-process `96+5` and `100+1` both passed. Same-process `110+0` failed at
  phase1 sync 99 / global launch 99. Its immediate capture missed the
  late-created generic node, but `/sys/class/devcoredump/devcd28` appeared
  shortly afterward and was copied under
  `coredump-capture-p110_0-late-devcd28-*`. The copied 64 KiB text coredump
  has the same gfxhub/GDS signature as the earlier direct-AB failures. No new
  `dmesg` lines appeared after 12:13 UTC, so this artifact is evidence from
  the sysfs coredump node rather than a fresh dmesg delta.
- `/tmp/hipfire-lds-direct-ab-multi-exec-artifacts/`: multi-child exec-parent
  controls for reads=3/448/512x86. The scratch harness runs a persistent
  parent process and a comma-separated list of fork/exec children, each child
  invoking the phase probe for its own local launch count. A one-child `101`
  run failed at child sync/global launch 100 and captured a late generic
  `devcd29` coredump 2 seconds after failure. The same total launch count
  passed when split into `96,5` or `50,30,21` child processes. Both split
  shapes also passed with a HIP-initialized parent.
- `/tmp/hipfire-lds-direct-ab-lower-grid-multi-exec-artifacts/`: lower-grid
  multi-child controls for reads=3/448/511x86. Reducing grid_x by one shifts
  the one-child failure edge upward but does not remove it: one child with
  `120` requested launches failed at sync/global launch 101 and captured a late
  generic `devcd30` coredump 2 seconds after failure. The same total requested
  launch count passed when split as `96,24` or `60,60` child processes. Both
  split shapes also passed with a HIP-initialized parent. The failing run had
  an empty `dmesg.since.txt`, so the devcoredump payload is the authoritative
  low-level artifact for that repeat. A follow-up one-child bracket at the
  same grid showed the edge had shifted lower after reset pressure: `100`,
  `101`, and `102` all failed, then a low-to-high sweep passed `90`, `95`,
  `96`, and `98` before `99` failed at sync/global launch 98. The `99` failure
  captured late generic `devcd34` with the same signature.
- The same artifact root also contains the next grid step, reads=3/448/510x86.
  One child with `99` requested launches passed. One child with `120` requested
  launches failed at sync/global launch 99 and captured late generic `devcd35`;
  after split controls, one child with `100` requested launches failed at
  sync/global launch 96 and captured late generic `devcd36`. The same total
  `120` requested launches passed when split as `96,24` or `60,60` child
  processes, in both plain-parent and HIP-initialized-parent modes.
- At reads=3/448/509x86, one child with `100` requested launches failed at
  sync/global launch 99 and captured late generic `devcd37`. A follow-up
  low-to-high sweep passed `90`, `95`, and `98`, then `99` failed at
  sync/global launch 97 with late generic `devcd38`. The split controls again
  passed for `96,24` and `60,60` in both plain-parent and
  HIP-initialized-parent modes.

## Current Narrowing

Evidence argues against these as sole causes:

- `__launch_bounds__` second argument.
- Simple LDS bank-layout issue: row padding still fails.
- A-side vs B-side address math: A-only and B-only LDS both fail.
- LDS allocation size alone: tiny 4x4 active LDS inside an 8x8 block passes.
- Multi-wave `__syncthreads()` alone: 4x4 active LDS inside 8x8 block passes.
- Multi-wave LDS store/load/barrier alone at small grids: standalone HIP
  LDS-only kernels pass for `TILE=6`, `TILE=8`, and `TILE=16` at 100 launches
  with 64x64 grids.
- hipfire Rust runtime/JIT/dispatch as the root cause: a standalone HIP GEMM
  repro using `hipcc` and direct `hipLaunchKernelGGL` still fails.
- Actual global memory access as the root cause: a compile-time standalone
  synthetic `TILE=6` kernel with no global loads and no C/global store still
  fails once K-loop work and grid size are high enough.
- Exec-mask presence as the root cause: adding LDS-only-style exec-mask regions
  to the synthetic GEMM-shaped kernel did not help, and removing exec-mask
  regions from the LDS-only control did not move the 288x86 pass / 304x86 fail
  threshold.

The `tile6` dmesg delta shows a driver-side device wedge, not a simple HIP
runtime recoverable error. The latest v2 failing run again reset through the
same path:

```text
amdgpu ... MES failed to respond to msg=REMOVE_QUEUE
amdgpu ... failed to remove hardware queue from MES, doorbell=0x1802
amdgpu ... MES might be in unrecoverable state, issue a GPU reset
amdgpu ... Failed to evict queue 1
amdgpu ... Failed to evict process queues
amdgpu ... GPU reset begin!. Source:  3
amdgpu ... remove_all_kfd_queues_mes: Failed to remove queue 0 for dev 42885
amdgpu ... Dumping IP State
amdgpu ... MODE2 reset
amdgpu ... GPU reset succeeded, trying to resume
amdgpu ... AMDGPU device coredump file has been created
amdgpu ... GPU reset(12) succeeded!
amdgpu ... [drm] device wedged, but recovered through reset
```

Driver-source mapping from the local amdgpu tree:

- `GPU reset begin!. Source:  3` maps to `AMDGPU_RESET_SRC_MES` in
  `amd/amdgpu/amdgpu_reset.h`.
- The reset work path chooses `AMDGPU_RESET_SRC_MES` when `adev->enable_mes`
  is true in `amd/amdgpu/amdgpu_amdkfd.c`.
- `Failed to evict queue`, `Failed to evict process queues`, and
  `remove_all_kfd_queues_mes` are KFD queue eviction / MES queue removal paths.

With passwordless sudo, the devcoredump sysfs node can be sampled. The latest
captured coredump is text-formatted, 64 KiB, and starts with:

```text
**** AMDGPU Device Coredump ****
kernel: 6.17.0-35-generic
module: amdgpu
HWIP: GC[1][0]: v11.0.1.0.0
MES_KIQ feature version: 6, fw version: 0x00000109
MES feature version: 1, fw version: 0x00000087
[gfxhub] Page fault observed
Faulty page starting at address: 0x0000000000000000
Protection fault status register: 0x0
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

Decoded against `gc_11_0_3_sh_mask.h`, the two GDS registers both have
`WRITE_DIS`, `FAULT_DETECTED`, and `GRBM` set. Their decoded address field is
`0xfc0`; `GDS_VM_PROTECTION_FAULT` reports `VMID=1`.

This materially changes the lower-level description: the user-visible recovery
path is a MES queue removal/reset wedge, while the devcoredump also records a
gfxhub page-fault snapshot and GDS/GDS-VM protection fault registers. That makes
this look much more like a GPU/kernel-codegen/driver interaction than an
ordinary HIP launch bookkeeping issue.

The standalone HIP GEMM repro independently reproduced the failure outside
hipfire:

```text
sync 20 failed: unspecified launch failure (719)
amdgpu ... MES failed to respond to msg=REMOVE_QUEUE
amdgpu ... failed to remove hardware queue from MES, doorbell=0x1802
amdgpu ... GPU reset begin!. Source:  3
amdgpu ... MODE2 reset
amdgpu ... GPU reset(13) succeeded!
```

Its coredump is also text-formatted, 64 KiB, and reports:

```text
[gfxhub] Page fault observed
Faulty page starting at address: 0x000074669d000000
Protection fault status register: 0x841051
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

The GDS/GDS-VM protection registers match the earlier hipfire-run failure. The
fault address differs: the earlier coredump captured address `0x0`, while the
standalone GEMM captured a concrete process GPUVA-like address. Both paths
still converge on the same MES reset and GDS protection state.

Reduction results after extending the standalone HIP GEMM probe:

- `nostore` failed, so the final C write is not required.
- `aonly` and `bonly` failed, so neither A nor B global load is individually
  required.
- `noglobal_nostore` failed, so no actual global memory access is required.
- A compile-time `tile6_synth` kernel with no global pointers and no C store
  also failed at `K_LIMIT=2048` and `3072`, ruling out dead global branches in
  the runtime-mode kernel as the trigger.
- `tile6_synth` passed at `K_LIMIT=1536` and failed repeatedly at
  `K_LIMIT=2048` for M=512, N=3072.
- `tile5_synth` / no-global/no-store passed at full K=3072 for M=512, N=3072,
  preserving the one-wave vs multi-wave boundary.
- `tile6_synth` at M=512, K_LIMIT=2048 passed up to N=2496 and failed at
  N=2688, 2880, and 3072. This points at total grid/work duration as part of
  the trigger.
- Preserved repeat runs sharpen that into a cumulative launch/work threshold,
  but not an exact deterministic launch counter. For M=512, N=2688,
  K_LIMIT=2048, one artifact family passes at 80, 85, and 90 launches, then
  fails at sync 94 when asked for 95 or 100 launches. A fresh bisect artifact
  family passes at 91, 92, 93, 94, and 95 launches, then fails at sync 93 for
  96 launches and sync 95 for 100 launches. Treat the edge as a narrow,
  reset/state-sensitive band around roughly launches 94-96. Holding launch
  count at 100, larger N still moves failure earlier: N=2688 fails around sync
  81-95 across repeats, N=2880 at sync 87, and N=3072 at sync 81.
- Adding LDS-only-style active-lane exec-mask regions to the synthetic kernel
  does not avoid the fault. The masked synthetic `TILE=6` kernel still fails at
  sync 95 for M=512, N=2688, K_LIMIT=2048. This weakens the hypothesis that the
  passing LDS-only control survives solely because it has `s_and_saveexec_b32`
  / `s_cbranch_execz` around LDS store/load regions.
- Grid-matching the simpler LDS-only control changes the conclusion. The
  `tile6_i512` LDS-only kernel that passes at 64x64 fails once the grid and
  total LDS work are large enough, without GEMM global-memory traffic and
  without the synthetic GEMM-shaped source. At 100 launches with grid_y=86,
  grid_x 297 passes and 298 fails; larger grids fail earlier. The short
  128-iteration `tile6` LDS-only kernel still passes at 448x86, so grid size
  alone is not enough.
- At the grid-matched 448x86 shape, loop depth has its own tight edge. The
  same LDS-only pattern passes at 320 iterations and fails at 336 iterations;
  larger loop depths fail earlier.
- The LDS-only loop-depth edge lines up with the synthetic GEMM K-limit edge.
  For `TILE=6`, `K_LIMIT=1536` is 256 loop trips and the LDS-only `tile6_i256`
  control passes even at grid 512x86. `K_LIMIT=2048` is about 342 loop trips;
  LDS-only `tile6_i320` and `tile6_i336` already fail at grid 512x86. This
  makes the synthetic GEMM threshold look like the same active-LDS
  loop-depth/grid threshold rather than a separate GEMM-shaped source effect.
- The one-wave boundary still holds under the larger grid: `tile5_i512` passes
  at 512x86, while `tile6_i320` and above fail.
- A minimal no-output kernel preserves the same thresholds. It removes host
  device allocations, kernel global-pointer arguments, the final global store,
  exec-mask regions, and object-aggregate template noise. Its ISA has no
  global load/store instructions and no `s_and_saveexec`; it still passes at
  tile6/256 and fails at tile6/320 for grid 512x86, and preserves the 448x86
  320-pass / 336-fail edge.
- Removing exec-mask regions from the LDS-only control does not shift that
  threshold materially. `tile6_i512_nomask` passes at 288x86 and fails at
  304x86, matching the masked control. The no-mask failure's coredump has the
  same gfxhub/GDS/GDS-VM signature.
- Rectangular active/block controls refine the earlier active-lane hypothesis.
  Exact one-wave `8x4` active/block with K=6 and a 288 B LDS segment passes at
  512 iterations. A two-wave `8x5` block with the same `8x4` active LDS region
  passes at 336 iterations but fails at 512. A `7x5` all-active block with K=6
  fails at 320 and 336. This means crossing 32 active lanes is an early failure
  accelerator, not the only Boolean trigger.
- The same rectangular controls show K-depth is part of the trigger. `5x5`
  active lanes inside a two-wave `6x6` block pass at K=5 and 512 iterations,
  but fail at K=6 even at 320 iterations. The corresponding one-wave `5x5`
  active/block, K=6 control passes at 512 iterations. This moves the current
  model from "active lanes > 32" to "multi-wave block plus K=6 LDS
  producer/consumer loops plus cumulative work".
- For the all-active `6x6` rectangular source, K-depth shifts rather than
  creates the loop-depth threshold. K=6 passes at 256/272 iterations and fails
  at 280/288/320. The 272-pass and 280-fail artifacts have identical resource
  metadata: 288 B LDS, 30 VGPR, 5 SGPR, 4 `s_barrier`, 2 `ds_store*`, 10
  `ds_load*`, 5 `s_waitcnt`, 2 `s_and_saveexec`, and no global load/store
  instructions. K=5 on the same `6x6` active/block shape passes at
  320/336/384 and fails at 416/448/512. Its 384-pass and 416-fail artifacts
  also have identical resource metadata: 248 B LDS, 47 VGPR, 5 SGPR, 8
  `s_barrier`, 8 `ds_store*`, 24 `ds_load*`, 10 `s_waitcnt`, 4
  `s_and_saveexec`, and no global load/store instructions.
- The rectangular probe is not a pure active-lane experiment once the active
  rectangle has fewer lanes than the number of staged A/B LDS elements. In
  those cases some lanes execute extra producer-loop iterations before the
  barrier. This appears to be another trigger accelerator: `6x4`, `4x6`, and
  even `4x4` active regions inside a `6x6` block fail at K=6/320, and
  `4x4` inside `6x6` also fails at K=5/320. The earlier direct active4-in-8x8
  pass used a different K=4 direct-store source shape, so it should not be
  treated as equivalent to these rectangular cooperative-loader controls.
- A new direct per-lane LDS probe removes that cooperative-loader variable.
  With one LDS store per active lane per iteration, the previously failing
  small active-in-6x6 controls all pass at K=6/512: `4x4`, `5x5`, `6x4`, and
  `4x6` active rectangles inside a `6x6` launched block. The two-wave
  `8x4` active-in-`8x5` control also passes at K=6/512, while the cooperative
  A/B staging version failed at that point. This strengthens the conclusion
  that cooperative A/B staging and extra producer work accelerate the fault.
- The direct per-lane probe still reproduces HIP 719 once the all-active
  `6x6`, K=6 loop runs long enough. It passes at 320/384/448/464 iterations
  and fails at 480/512 at grid 512x86. K=5 on the same all-active `6x6`
  direct-store source passes at 512. The direct-store failure therefore keeps
  K-depth and repeated multi-wave LDS consumer work in the suspect set even
  after removing cooperative A/B staging.
- A two-array direct-store probe separates LDS footprint from cooperative
  producer-loop structure. With all-active `6x6`, it uses a 288 B LDS segment
  like the original square A/B cases, but each active lane directly stores one
  A and one B element. Reads=1 and reads=2 pass at 512 iterations, so the
  288 B footprint alone is not sufficient. Increasing repeated LDS read work
  moves the threshold sharply: reads=3 passes at 384 and fails at 448, reads=5
  passes at 224 and fails at 256, and reads=6 passes at 176 and fails at 192.
  This makes repeated LDS read pressure, not just LDS allocation size, a
  load-bearing trigger.
- The same two-array direct-store probe preserves stable controls: exact
  one-wave `8x4` and `8x4` active inside an `8x5` two-wave block both pass at
  reads=6/512, and small active rectangles (`5x5`, `4x4`) inside `6x6` pass at
  reads=6/512. This keeps the suspect set centered on all-active multi-wave
  LDS read pressure rather than barrier count alone.
- Grid-width sweeps on the direct-AB edge preserve the cumulative-work model.
  At reads=6 and 192 iterations, grid_x 256, 320, 384, 448, 480, 496, 504,
  508, and 509 all pass at grid_y=86. The exact 510x86 edge is
  reset/state-sensitive: an earlier grid sweep failed at sync 99, while a
  fresh launch-count replay passed through 100 launches. The fresh replay gives
  the cleaner edge at 511x86/512x86, both failing at sync 99 for 100 requested
  launches. At reads=3 and 448 iterations, grid_x 256, 320, 384, 448, 480,
  496, 504, 508, 510, and 511 pass, while grid_x 512 fails on repeat around
  sync 97-99. This is sharper than the earlier LDS-only grid threshold but
  points in the same direction: the fault appears after a narrow cumulative
  LDS-read/work threshold, not at kernel launch or compile time.
- Launch-count controls on the same direct-AB edge strengthen the cumulative
  exposure model, with the exact edge moving after reset pressure. At
  reads=6/192/511x86, launch counts 96, 97, and 98 pass. A 99-launch run
  passed in the first sweep but failed at sync 98 after later failures, while
  100-launch runs fail at sync 98-99. Reusing the exact same binary gives a
  sharper process-boundary control under the shifted edge: three split-process
  trials of `98 + 1` launches all pass, but a one-process 99-launch run fails
  at sync 98. That points at same-process/HIP-queue lifetime or same-queue
  dispatch sequence as part of the immediate trigger, not merely total LDS work
  submitted across process boundaries. At reads=3/448/512x86, launch counts
  94-99 pass in the initial sweep; extending to 110/120/130/150 requested
  launches fails around sync 98-101; a deliberate 100-vs-101 repeat after reset
  pressure failed at sync 99 and sync 98. Treat exact counters as
  state-sensitive, but not the broad fact that slightly shorter same-process
  runs pass and slightly longer same-process runs fail with the same generated
  code.
- Phase-mode controls sharpen the process-boundary result. With the same
  direct-AB kernel body at reads=6/192/511x86, same-process `99 + 1` fails on
  phase2 launch 0 / global launch 99, and same-process `98 + 2` fails on
  phase2 launch 1 / global launch 99 in the preserved failing repeats.
  `hipDeviceReset()` between `98 + 2` phases returns success but still fails on
  the same phase2/global launch, so HIP context reset from inside the process
  is not sufficient. Destroying and recreating a stream between phases is also
  insufficient. Cross-process `98+0` followed by `2+0` passes 2/2, so process
  exit still clears enough state to avoid the immediate edge. Explicit
  same-stream `98 + 2` is mixed: one preserved pass and three preserved
  failures. Later confirmation runs show the exact edge is still
  state-sensitive: same-process `98 + 2` and `99 + 1` both passed in one later
  run, while `100+0`, `100+1`, `101+0`, and `101+1` failed at global launch 99.
  Treat stream choice and exact phase split as state-sensitivity modifiers, not
  reliable explanations.
- Exec-parent controls separate parent lifetime from HIP-launching process
  lifetime. A parent wrapper that stays alive across both phases but runs each
  phase through a fork/exec child passes for `98 + 2` in all tested modes:
  plain parent, HIP-initialized parent, parent `hipDeviceReset()` before
  children, and parent `hipDeviceReset()` between children. The same wrapper
  also passes `99 + 1` in plain and HIP-initialized parent modes. This means an
  unrelated surviving parent process, even one with HIP initialized, does not
  retain the bad state. The meaningful cleanup boundary is exit of the process
  that actually launched the edge workload.
- A second direct-AB edge at reads=3/448/512x86 mostly preserves the same
  process-boundary shape, but shows the edge is not deterministic enough for
  single-trial overclaims. Same-process `98 + 3` completed phase1 and boundary
  sync, then failed on phase2 launch 1 / global launch 99. The same `98 + 3`
  split passed under an exec-parent plain parent, and passed again on repeat
  for both plain and HIP-initialized parents. One HIP-initialized-parent trial
  failed inside the phase1 child at sync 97 before any split-boundary question
  was exercised. Treat that as reset/state sensitivity at the edge, not as
  evidence that parent HIP initialization deterministically retains bad child
  state.
- A lower-risk reads=3/448/512x86 split strengthens the process-boundary
  result. Same-process `96 + 5` completed phase1 and boundary sync, then
  failed on phase2 launch 2 / global launch 98. Same-process `97 + 4` passed
  despite the same total requested launch count, reinforcing that ordering and
  GPU/process state matter near the edge. Exec-parent `96 + 5` passed in all
  first-pass parent modes, including a HIP-initialized parent, and passed again
  in repeat plain/hipinit trials. This makes the previous one-off
  HIP-initialized-parent failure at `98 + 3` look like ordinary edge
  state-sensitivity rather than deterministic parent HIP context retention.
- Explicitly clearing the generic devcoredump node before another reads=3 edge
  repeat did not stabilize the launch edge. Same-process `96 + 5` and
  `100 + 1` both passed after the clear, then a longer same-process `110 + 0`
  failed at phase1 sync 99 / global launch 99. A new generic devcoredump node
  (`devcd28`) appeared after the immediate capture window and contains the same
  gfxhub page fault, `0x841051` protection status, and
  `regGDS_* 0x3f000007/0x0fc00113` state. This gives a post-clear coredump
  match, but the missing fresh dmesg lines mean the sysfs coredump, not dmesg,
  is the authoritative evidence for this particular repeat.
- Multi-child exec-parent controls separate total submitted work from
  child-local launch-sequence length. At reads=3/448/512x86, a persistent
  plain parent running one child with `101` launches fails inside that child at
  sync/global launch 100 and produces the same gfxhub/GDS coredump. The same
  parent running the same total launch count split as `96,5` passes, and a
  three-child split `50,30,21` also passes. The `96,5` and `50,30,21` splits
  also pass when the parent has initialized HIP. This tightens the process
  boundary result: the cleanup that matters is exit of the process issuing the
  long launch sequence, not merely the existence of a surviving parent process
  or total launches across a parent-supervised job.
- The same multi-child result survives a lower grid. At reads=3/448/511x86,
  one child with `120` requested launches fails at sync/global launch 101 and
  produces the same late gfxhub/GDS coredump. Splitting the same total work as
  `96,24` or `60,60` passes in both plain-parent and HIP-initialized-parent
  modes. Reducing grid_x from 512 to 511 therefore shifts the child-local edge
  upward but preserves the process-exit boundary: total parent-supervised
  launches are not enough by themselves, while a long-enough sequence in one
  HIP-launching child process still crosses the failure side. A subsequent
  one-child bracket after reset pressure narrowed the shifted edge to `98`
  passing and `99` failing, with the `99` run failing at sync/global launch 98
  and producing the same late coredump signature.
- Stepping down again to reads=3/448/510x86 preserves the same process-boundary
  shape. One child with `99` requested launches passes, while one child with
  `100` requested launches fails after reset pressure. A one-child `120` run
  also fails, but the same total work split as `96,24` or `60,60` passes in
  both plain-parent and HIP-initialized-parent modes. This makes the grid-width
  effect look like a movement of the child-local launch threshold, not removal
  of the process-local failure mode.
- Reads=3/448/509x86 still preserves that shape. One-child `98` passes and
  one-child `99` fails, while `96,24` and `60,60` split-child controls pass in
  both plain-parent and HIP-initialized-parent modes. At this point, lowering
  grid_x from 511 to 509 has not eliminated the process-local failure edge; it
  has kept the practical bracket near the same 98/99 child-local launch count
  after reset pressure.
- Additional in-process teardown checks did not find a clean middle ground
  between `hipDeviceReset()` and process exit. `hipDevicePrimaryCtxReset(0)`
  and `hipDevicePrimaryCtxRelease(0)` both return success but still fail on the
  same phase2/global launch. Direct `hsa_shut_down()` after HIP work is not a
  practical recovery lever: the process segfaults or `hsa_init()` fails with
  `HSA_STATUS_ERROR_OUT_OF_RESOURCES` before phase2 can run cleanly.

The synthetic failure coredump again reports:

```text
[gfxhub] Page fault observed
Faulty page starting at address: 0x000074669d000000
Protection fault status register: 0x841051
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

The fresh launch-bisect coredump copied from the 96-launch failure reports the
same low-level signature:

```text
[gfxhub] Page fault observed
Faulty page starting at address: 0x000074669d000000
Protection fault status register: 0x841051
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

The direct-AB launch-count failures copied from the reads=6/192/511x86 and
reads=3/448/512x86 controls report the same signature:

```text
[gfxhub] Page fault observed
Faulty page starting at address: 0x000074669d000000
Protection fault status register: 0x841051
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

The reused-binary split-process controls captured the same signature for the
one-process 99-launch failure and the failed 99-launch half of the `99 + 1`
split attempt. The successful `98 + 1` split-process trials did not create a
new coredump.

Phase-mode failures captured the same signature for same-process `98 + 2`,
`hipDeviceReset()`-between-phases `98 + 2`, stream-recreate `98 + 2`, and
failed same-stream `98 + 2` repeats:

```text
[gfxhub] Page fault observed
Faulty page starting at address: 0x000074669d000000
Protection fault status register: 0x841051
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

The exec-parent confirmation failures for same-process `100+1` and `101+0`
again captured the same signature. The exec-parent pass cases did not create a
coredump.

The in-process teardown failures for `hipDeviceReset()`,
`hipDevicePrimaryCtxReset(0)`, and `hipDevicePrimaryCtxRelease(0)` also captured
the same gfxhub/GDS signature. The direct `hsa_shut_down()` modes are excluded
from fault-mechanism interpretation because they crashed the host process rather
than producing a clean phase2 HIP launch result.

The reads=3/448/512x86 second-edge phase-mode failures captured the same
low-level signature for same-process `101+0`, same-process `98+3`, and the
single failed HIP-initialized exec-parent trial:

```text
[gfxhub] Page fault observed
Faulty page starting at address: 0x000074669d000000
Protection fault status register: 0x841051
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

After explicitly freeing the generic devcoredump node, the same
reads=3/448/512x86 edge produced another matching coredump on a longer
same-process `110+0` run that failed at phase1 sync 99 / global launch 99.
The fresh node appeared as `/sys/class/devcoredump/devcd28` shortly after the
immediate capture window, and the copied 64 KiB text payload reported:

```text
[gfxhub] Page fault observed
Faulty page starting at address: 0x000074669d000000
Protection fault status register: 0x841051
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

The multi-child exec-parent one-child `101` failure also produced a late
generic devcoredump (`devcd29`) with the same fields:

```text
[gfxhub] Page fault observed
Faulty page starting at address: 0x000074669d000000
Protection fault status register: 0x841051
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

The lower-grid multi-child one-child `120` failure produced another late
generic devcoredump (`devcd30`) with the same fields:

```text
[gfxhub] Page fault observed
Faulty page starting at address: 0x000074669d000000
Protection fault status register: 0x841051
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

The lower-grid one-child bracket failures (`100`, `101`, `102`, then `99`)
captured the same fields in `devcd31` through `devcd34`; the tightest preserved
bracket is `98` pass / `99` fail at 511x86:

```text
[gfxhub] Page fault observed
Faulty page starting at address: 0x000074669d000000
Protection fault status register: 0x841051
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

The 510x86 one-child failures (`120`, then `100`) captured the same fields in
`devcd35` and `devcd36`; the preserved one-child bracket is `99` pass / `100`
fail:

```text
[gfxhub] Page fault observed
Faulty page starting at address: 0x000074669d000000
Protection fault status register: 0x841051
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

The 509x86 one-child failures (`100`, then `99`) captured the same fields in
`devcd37` and `devcd38`; the preserved one-child bracket is `98` pass / `99`
fail:

```text
[gfxhub] Page fault observed
Faulty page starting at address: 0x000074669d000000
Protection fault status register: 0x841051
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

Code object/resource observations from `llvm-readobj` dumps:

| Variant | Workgroup | LDS group segment | VGPR | SGPR | Spills | Wavefront |
|---|---:|---:|---:|---:|---:|---:|
| `tile5` pass | 25 | 212 B | 18 | 20/21 | 0 | 32 |
| `tile6` fail | 36 | 288 B | 20 | 20/21 | 0 | 32 |
| standalone GEMM `tile5` pass | 25 | 212 B | 23 | 16 | 0 | 32 |
| standalone GEMM `tile6` fail | 36 | 288 B | 25 | 16 | 0 | 32 |
| standalone synthetic `tile6` fail | 36 | 288 B | 26 | 17 | 0 | 32 |

Per-symbol metadata for the newest reduced repro versus the passing long-loop
LDS-only control:

| Variant | Result | Kernel symbol | LDS group segment | VGPR | SGPR | Spills | Wavefront |
|---|---:|---|---:|---:|---:|---:|---:|
| synthetic GEMM-shaped `TILE=6`, no global/no store | FAIL | `_Z20gemm_lds_synth_probeILi6EEviiii` | 288 B | 18 | 5 | 0 | 32 |
| masked synthetic GEMM-shaped `TILE=6`, no global/no store | FAIL | `_Z27gemm_lds_synth_masked_probeILi6EEviiii` | 288 B | 18 | 7 | 0 | 32 |
| LDS-only `TILE=6`, 512 iterations | PASS | `_Z9lds_probeILi6ELi6ELi512EEvPfi` | 288 B | 20 | 8 | 0 | 32 |
| LDS-only no-mask `TILE=6`, 512 iterations | FAIL at 304x86 | `_Z16lds_probe_nomaskILi6ELi512EEvPfi` | 288 B | 56 | 8 | 0 | 32 |
| minimal no-output LDS-only `TILE=6` | FAIL at 320 iterations / 512x86 | `_Z17lds_minimal_probev` | 288 B | 54 | 2 | 0 | 32 |
| rect-active no-output `6x6` block, K=6 | FAIL at 320 iterations / 512x86 | `_Z21lds_rect_active_probev` | 288 B | 30 | 5 | 0 | 32 |
| rect-active no-output `6x6` block, K=6 | PASS at 272, FAIL at 280 iterations / 512x86 | `_Z21lds_rect_active_probev` | 288 B | 30 | 5 | 0 | 32 |
| rect-active no-output `6x6` block, K=5 | PASS at 384, FAIL at 416 iterations / 512x86 | `_Z21lds_rect_active_probev` | 248 B | 47 | 5 | 0 | 32 |
| rect-active no-output `8x4` block, K=6 | PASS at 512 iterations / 512x86 | `_Z21lds_rect_active_probev` | 288 B | 33 | 7 | 0 | 32 |
| rect-active no-output `8x4` active in `8x5` block, K=6 | PASS at 336, FAIL at 512 iterations / 512x86 | `_Z21lds_rect_active_probev` | 288 B | 21 | 8 | 0 | 32 |
| rect-active no-output `5x5` active in `6x6` block, K=5 | PASS at 512 iterations / 512x86 | `_Z21lds_rect_active_probev` | 212 B | 16 | 7 | 0 | 32 |
| rect-active no-output `5x5` active in `6x6` block, K=6 | FAIL at 320 iterations / 512x86 | `_Z21lds_rect_active_probev` | 248 B | 16 | 9 | 0 | 32 |
| rect-active no-output `6x4` active in `6x6` block, K=6 | FAIL at 320 iterations / 512x86 | `_Z21lds_rect_active_probev` | 240 B | 18 | 9 | 0 | 32 |
| rect-active no-output `4x6` active in `6x6` block, K=6 | FAIL at 320 iterations / 512x86 | `_Z21lds_rect_active_probev` | 240 B | 20 | 10 | 0 | 32 |
| rect-active no-output `4x4` active in `6x6` block, K=5 | FAIL at 320 iterations / 512x86 | `_Z21lds_rect_active_probev` | 160 B | 18 | 8 | 0 | 32 |
| direct-active no-output `6x6` block, K=6 | PASS at 464, FAIL at 480/512 iterations / 512x86 | `_Z23lds_direct_active_probev` | 144 B | 33-45 | 2 | 0 | 32 |
| direct-active no-output `6x6` block, K=5 | PASS at 512 iterations / 512x86 | `_Z23lds_direct_active_probev` | 144 B | 28 | 2 | 0 | 32 |
| direct-active no-output `8x4` active in `8x5` block, K=6 | PASS at 512 iterations / 512x86 | `_Z23lds_direct_active_probev` | 128 B | 14 | 5 | 0 | 32 |
| direct-active no-output `4x4` active in `6x6` block, K=6 | PASS at 512 iterations / 512x86 | `_Z23lds_direct_active_probev` | 64 B | 13 | 4 | 0 | 32 |
| direct-AB no-output `6x6` block, reads=1 | PASS at 512 iterations / 512x86 | `_Z19lds_direct_ab_probev` | 288 B | 22 | 2 | 0 | 32 |
| direct-AB no-output `6x6` block, reads=2 | PASS at 512 iterations / 512x86 | `_Z19lds_direct_ab_probev` | 288 B | 24 | 2 | 0 | 32 |
| direct-AB no-output `6x6` block, reads=3 | PASS at 384, FAIL at 448/512 iterations / 512x86 | `_Z19lds_direct_ab_probev` | 288 B | 34 | 2 | 0 | 32 |
| direct-AB no-output `6x6` block, reads=5 | PASS at 224, FAIL at 256 iterations / 512x86 | `_Z19lds_direct_ab_probev` | 288 B | 54 | 2 | 0 | 32 |
| direct-AB no-output `6x6` block, reads=6 | PASS at 176, FAIL at 192 iterations / 512x86 | `_Z19lds_direct_ab_probev` | 288 B | 40-52 | 2 | 0 | 32 |
| direct-AB no-output `6x6` block, reads=6, 192 iters | PASS at 509x86, MIXED at 510x86, FAIL at 511x86 | `_Z19lds_direct_ab_probev` | 288 B | 52 | 2 | 0 | 32 |
| direct-AB no-output `6x6` block, reads=3, 448 iters | PASS at 511x86, FAIL on repeat at 512x86 | `_Z19lds_direct_ab_probev` | 288 B | 34 | 2 | 0 | 32 |
| direct-AB no-output `6x6` block, reads=6, 192 iters, 511x86 | PASS at 98 launches, MIXED at 99 launches, FAIL at 100 launches | `_Z19lds_direct_ab_probev` | 288 B | 52 | 2 | 0 | 32 |
| direct-AB no-output `6x6` block, reads=6, 192 iters, 511x86, split-process | PASS for 98+1 split, FAIL for one-process 99 | `_Z19lds_direct_ab_probev` | 288 B | 52 | 2 | 0 | 32 |
| direct-AB phase-mode `6x6` block, reads=6, 192 iters, 511x86 | PASS for cross-process 98+2, FAIL for same-process 98+2 / device-reset 98+2 / stream-recreate 98+2 | `_Z25lds_direct_ab_phase_probev` | 288 B | 52 | 2 | 0 | 32 |
| direct-AB phase-mode teardown `6x6` block, reads=6, 192 iters, 511x86 | FAIL for primary-ctx reset/release; HSA shutdown crashes host process | `_Z25lds_direct_ab_phase_probev` | 288 B | 52 | 2 | 0 | 32 |
| direct-AB exec-parent `6x6` block, reads=6, 192 iters, 511x86 | PASS for child-process 98+2 and 99+1 even with HIP-initialized parent | `_Z25lds_direct_ab_phase_probev` | 288 B | 52 | 2 | 0 | 32 |
| direct-AB no-output `6x6` block, reads=3, 448 iters, 512x86 | PASS at 99 launches, FAIL on 100+ launch repeats | `_Z19lds_direct_ab_probev` | 288 B | 34 | 2 | 0 | 32 |
| direct-AB multi-exec `6x6` block, reads=3, 448 iters, 511x86 | PASS through one-child 98; FAIL at one-child 99+; PASS for 96,24 and 60,60 child splits | `_Z25lds_direct_ab_phase_probev` | 288 B | 34 | 2 | 0 | 32 |
| direct-AB multi-exec `6x6` block, reads=3, 448 iters, 510x86 | PASS through one-child 99; FAIL at one-child 100+; PASS for 96,24 and 60,60 child splits | `_Z25lds_direct_ab_phase_probev` | 288 B | 34 | 2 | 0 | 32 |
| direct-AB multi-exec `6x6` block, reads=3, 448 iters, 509x86 | PASS through one-child 98; FAIL at one-child 99+; PASS for 96,24 and 60,60 child splits | `_Z25lds_direct_ab_phase_probev` | 288 B | 34 | 2 | 0 | 32 |
| direct-AB no-output `8x4` active in `8x5` block, reads=6 | PASS at 512 iterations / 512x86 | `_Z19lds_direct_ab_probev` | 256 B | 22 | 5 | 0 | 32 |

ISA observations:

- `tile5` is a single-wave workgroup (`25 < 32`). The compiler appears to
  remove explicit `s_barrier` instructions in the runtime-generated code object.
- `tile6` is a two-wave workgroup (`36 > 32`) and retains `s_barrier`
  instructions around LDS traffic.
- Both `tile5` and `tile6` still contain LDS instructions. The current ISA
  counts are: `tile5` = 0 `s_barrier`, 4 `ds_store*`, 12 `ds_load*`; `tile6` =
  4 `s_barrier`, 4 `ds_store*`, 10 `ds_load*`.
- Standalone GEMM object counts across all compiled template variants are:
  6 `s_barrier`, 6 `ds_store*`, 23 `ds_load*`, 6 `global_load*`, and
  3 `global_store*`. The standalone object contains `TILE=5`, `TILE=6`, and
  `TILE=16` template instantiations, so use per-symbol disassembly before
  over-interpreting the aggregate counts.
- Aggregate object counts for the newest saved objects are not directly
  comparable because each object contains several template instantiations. The
  failing synthetic object as a whole has 8 `s_barrier`, 10 `ds_store*`, 28
  `ds_load*`, 29 `s_waitcnt`, and 59 `s_cbranch` instances. The passing
  LDS-only long-loop object as a whole has more LDS/barrier traffic: 26
  `s_barrier`, 13 `ds_store*`, 74 `ds_load*`, 64 `s_waitcnt`, and 40
  `s_cbranch` instances.
- The failing synthetic `tile6` symbol has the compact GEMM-shaped loop:
  `ds_store_2addr_b32`, `s_waitcnt lgkmcnt(0)`, `s_barrier`,
  `buffer_gl0_inv`, a cluster of `ds_load_*`, staged `s_waitcnt`/`v_fmac`,
  another `s_barrier`, and a scalar loop back edge. It has no global load/store
  in this symbol.
- The passing `lds_probe<TILE=6, ACTIVE=6, ITERS=512>` symbol uses the same
  288-byte LDS footprint but carries explicit exec-mask control
  (`s_and_saveexec_b32` / `s_cbranch_execz`) around active-lane store/load
  regions, includes two LDS phases per loop iteration pair, and finishes with a
  global store. It passes despite higher aggregate barrier and DS counts.
- The masked synthetic `tile6` symbol also carries exec-mask control around
  LDS regions and has 288 B LDS, 18 VGPR, 7 SGPR, and zero spills. It still
  fails, so the remaining difference from the passing LDS-only long-loop
  control is not just the presence of exec-mask instructions.
- The no-mask LDS-only `tile6_i512_nomask` symbol has no `s_and_saveexec`
  inside the symbol, 288 B LDS, 56 VGPR, 8 SGPR, and zero spills. Per-symbol
  counts for that symbol are 8 `s_barrier`, 4 `ds_store*`, 20 `ds_load*`, 13
  `s_waitcnt`, 1 `s_cbranch`, and 1 final `global_store`. It shares the same
  288x86 pass / 304x86 fail threshold as the masked LDS-only control.
- The minimal no-output `lds_minimal_probe` symbol has 288 B LDS, 54 VGPR, 2
  SGPR, zero spills, no global load/store instructions, and no `s_and_saveexec`.
  Its instruction counts are 8 `s_barrier`, 4 `ds_store*`, 20 `ds_load*`, 12
  `s_waitcnt`, and 1 `s_cbranch`.
- The active4-in-8x8 control passed even though the launched block spans two
  waves; only 16 lanes actively touch LDS. This keeps the current hypothesis on
  active LDS traffic across waves rather than barrier presence alone.
- Rect-active no-output failures preserve the same coredump signature as the
  earlier minimal and synthetic failures. Sampled `7x5` K=6, `8x4` active in
  `8x5` K=6, and `5x5` active in `6x6` K=6 failures all report the same
  `gfxhub` page fault at `0x000074669d000000`, protection status `0x841051`,
  `regGDS_PROTECTION_FAULT 0x3f000007`, and
  `regGDS_VM_PROTECTION_FAULT 0x0fc00113`.
- Direct-active no-output failures keep the same low-level signature. The
  captured all-active `6x6`, K=6, 512-iteration coredump reports the same
  `gfxhub` page fault at `0x000074669d000000`, protection status `0x841051`,
  `regGDS_PROTECTION_FAULT 0x3f000007`, and
  `regGDS_VM_PROTECTION_FAULT 0x0fc00113`.
- Direct-AB no-output failures keep the same low-level signature. Captured
  reads=3, reads=5, and reads=6 failures report the same `gfxhub` page fault at
  `0x000074669d000000`, protection status `0x841051`,
  `regGDS_PROTECTION_FAULT 0x3f000007`, and
  `regGDS_VM_PROTECTION_FAULT 0x0fc00113`. Captured grid-width-edge failures
  for reads=6/192 at 510x86 and reads=3/448 at 512x86 report the same
  signature.

Best current hypothesis:

> On gfx1103 with this ROCm/amdgpu stack, the failure is a multi-wave
> LDS loop/grid-duration/cumulative-launch fault, not a plain global-memory
> bug and not specific to the original GEMM global-memory traffic. The original
> square-kernel symptom remains `TILE=5`/K=5 one-wave passing versus
> `TILE=6`/K=6 multi-wave failing, but rectangular controls refine that into a
> more precise model: exact one-wave K=6 LDS blocks are stable, while multi-wave
> blocks with repeated LDS producer/consumer work fail after a duration
> threshold whose position depends on active shape, read count, K-depth, grid
> work, launched block shape, and LDS producer-loop shape. Direct per-lane LDS
> stores still reproduce HIP 719, but much later than cooperative A/B staging
> unless the direct source uses two arrays and enough repeated reads. LDS
> footprint alone is not enough: two-array reads=1/2 passes at 512 despite a
> 288 B segment. Crossing 32 active lanes, increasing read/K-depth, and
> requiring extra cooperative producer iterations all accelerate the failure
> rather than solely defining it. A no-global, no-store synthetic GEMM-shaped
> kernel reproduces HIP 719, and simpler no-output LDS probes reproduce the same
> gfxhub/GDS coredump signature once the block/read-depth/grid/launch threshold
> is crossed. The latest direct-AB grid sweeps make that threshold extremely
> narrow near the fail edge: one grid column separates pass from fail in the
> reads=3/448 case. Reused-binary split-process controls narrow the cumulative
> part further: at the reads=6/192/511x86 edge, `98 + 1` launches split across
> two processes pass repeatedly, while 99 launches in one process fail. That
> implicates same-process lifetime in the immediate trigger. Phase-mode controls
> refine that further: `hipDeviceReset()` and stream destroy/recreate inside the
> same process do not clear the edge for `98 + 2`, while process exit between
> `98` and `2` does. Exec-parent controls show that a surviving parent process,
> even one with HIP initialized, does not retain the bad state when the
> HIP-launching children exit between phases. The remaining suspect layer is
> therefore more like state owned by the HIP/HSA/KFD process that launched the
> kernels: code-object/queue bookkeeping, process-scoped GPUVM or queue state,
> or GPU state keyed by that process, not merely a user-visible stream lifetime
> or parent process lifetime. The exposed in-process HIP reset APIs tested so
> far (`hipDeviceReset`, primary-context reset/release) do not clear it, and
> calling raw `hsa_shut_down()` after HIP work is not a clean recovery path on
> this stack. A second direct-AB edge at reads=3/448/512x86 strengthens the
> process-boundary result but also underscores the state sensitivity: a plain
> child-process split passes where same-process `98+3` fails after the phase
> boundary, while one HIP-initialized-parent trial failed before the boundary
> and then passed on repeat. Process exit appears to clear enough state near
> the edge, but it is not a deterministic explanation for every trial once the
> first child itself lands on the shifted failure side. A lower-risk `96+5`
> split makes the parent-state picture cleaner: same-process `96+5` fails after
> the boundary, same-process `97+4` passes, and exec-parent `96+5` passes in
> plain and HIP-initialized parent modes across repeats. That points back to
> state retained by the process actually issuing a long sequence of launches,
> not parent process lifetime by itself. A post-clear generic devcoredump
> capture on same-process `110+0` again matches the gfxhub/GDS signature, but
> the edge still moves enough that post-clear `96+5` and `100+1` can pass.
> Multi-child exec-parent controls tighten that further: one child running
> `101` launches at 512x86 fails, while `96,5` and `50,30,21` child splits
> with the same total launch count pass even when the parent process has
> initialized HIP. The lower-grid 511x86 replay preserves the same shape:
> one child running `120` launches fails, while `96,24` and `60,60` splits
> with the same total pass in both plain-parent and HIP-initialized-parent
> modes. A follow-up one-child bracket at 511x86 shifted lower after reset
> pressure but stayed sharp: `98` passes and `99` fails. Stepping grid_x down
> to 510 keeps the same pattern with `99` pass / `100` fail and split children
> passing at the same total work. Stepping to 509 still gives `98` pass / `99`
> fail and split-child passes.
> Exec-mask structure alone does not appear to be the deciding factor.

## Next Evidence To Capture

Continue improving the small repro matrix so it emits and preserves artifacts
for each variant (`TILE=4`, `5`, `6`, `8`, `16`, and the 4-active-in-8x8
control):

- Exact patch or generated kernel source.
- pass/fail launch count and retry behavior.
- `dmesg` / amdgpu log delta around each run.
- generated code object metadata via ROCm LLVM tools.
- ISA dump via `llvm-objdump`.
- rocprof/rocprof-compute output for passing variants and any failing variants
  that complete far enough to profile.
- root-only follow-up: keep using the late generic `/sys/class/devcoredump`
  wait wrapper for new failing probes; it successfully captured `devcd29` two
  seconds after the one-child multi-exec failure.
- improve the throwaway matrix runner so it always preserves the exact
  runtime-generated `.hsaco`; the active4 control passed but did not leave a
  `.hsaco` under the expected cache name in the latest run.
- reduce the standalone HIP synthetic reproducer further: binary-search the
  launch-count edge around 94-96 at N=2688 with repeated fresh-process trials,
  the N=2496-2688 grid edge at 100 launches, and the K_LIMIT=1536-2048 edge
  independently.
- reduce the LDS-only reproducer further: repeat the tight grid_x 297/298 edge
  and loop-depth 320/336 edge in fresh processes to determine how much state
  sensitivity remains.
- use the minimal no-output repro for the next reduction: repeat the 256-pass /
  320-fail correlate in fresh processes and try smaller active-lane shapes
  around the one-wave/two-wave boundary.
- use the direct-AB phase-mode repro for the next reduction: the 511x86
  lower-grid multi-child replay preserved the child-local launch sequence
  finding (`120` in one child fails; `96,24` and `60,60` split children pass),
  and a follow-up one-child bracket now has `98` pass / `99` fail at the same
  grid after reset pressure. The 510x86 replay has `99` pass / `100` fail, with
  the same split-child passes at total `120`, and 509x86 still has `98` pass /
  `99` fail with the same split-child passes. Next, either step grid_x lower
  again or vary the split child-local count around `98/99` to see whether the
  failing unit is launch count, total work per child, or a narrower queue
  sequence property. Treat the common in-process HIP reset APIs as already
  tested; only revisit teardown if a genuinely different ROCm mechanism is
  identified.
- create a single-instantiation compile unit for the failing synthetic symbol
  and the passing long-loop symbol so instruction counts can be per-symbol
  instead of object-aggregate.
- create single-instantiation LDS-only masked/no-mask compile units so resource
  and instruction counts are not polluted by unused template variants.

Compare pass/fail boundary for:

- active lanes and waves per workgroup,
- LDS instructions and barrier sequence,
- VGPR/SGPR/LDS resource usage,
- occupancy/workgroup metadata,
- any kernel log evidence of GPUVM fault, queue fault, ring timeout, or trap.

## Working Conclusion

`5546fe12`'s no-LDS register-tiled production choice is currently justified.
The 288 GFLOP/s LDS path from `b41368bb` is not safe on gfx1103. The strongest
current lead is not just "LDS is flaky"; LDS-only direct-HIP stress passes, but
a multi-wave GEMM-shaped synthetic LDS loop with no global memory traffic can
still fail, producing a MES reset path plus GDS/GDS-VM protection-fault state in
the amdgpu coredump. A production mitigation should not be attempted until the
standalone synthetic repro is reduced enough to identify which grid/work/ISA
feature crosses from valid LDS use into the gfx1103 fault path.
