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
| Direct-AB no-output `6x6` active/block, reads=6, 192 iterations, 510/511/512x86 grid | FAIL | Grid-width edge is between 509 and 510 at grid_y=86. |
| Direct-AB no-output `6x6` active/block, reads=3, 448 iterations, 511x86 grid | PASS | Grid-width low side at fixed read/loop edge. |
| Direct-AB no-output `6x6` active/block, reads=3, 448 iterations, 512x86 grid | FAIL | Grid-width edge is between 511 and 512 at grid_y=86. |
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
  work thresholds: reads=6/192 passes through 509x86 and fails at 510x86,
  while reads=3/448 passes through 511x86 and fails at 512x86.

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
  508, and 509 all pass at grid_y=86, while grid_x 510, 511, and 512 fail at
  sync 99 with identical resource metadata. At reads=3 and 448 iterations,
  grid_x 256, 320, 384, 448, 480, 496, 504, 508, 510, and 511 pass, while
  grid_x 512 fails on repeat at sync 97-98. This is sharper than the earlier
  LDS-only grid threshold but points in the same direction: the fault appears
  after a narrow cumulative LDS-read/work threshold, not at kernel launch or
  compile time.

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
| direct-AB no-output `6x6` block, reads=6, 192 iters | PASS at 509x86, FAIL at 510x86 | `_Z19lds_direct_ab_probev` | 288 B | 52 | 2 | 0 | 32 |
| direct-AB no-output `6x6` block, reads=3, 448 iters | PASS at 511x86, FAIL at 512x86 | `_Z19lds_direct_ab_probev` | 288 B | 34 | 2 | 0 | 32 |
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
> reads=3/448 case. Exec-mask structure alone does not appear to be the deciding
> factor.

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
- root-only follow-up: capture a fresh full devcoredump after clearing the old
  sysfs node, then compare its GDS/GDS-VM registers against the v2 sample.
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
- use the direct-AB no-output repro for the next reduction: vary launch count
  at the reads=6/192/510x86 and reads=3/448/512x86 edges, then repeat one edge
  in fresh processes to quantify state sensitivity.
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
