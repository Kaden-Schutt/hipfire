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

Their artifact roots are:

```text
/tmp/hipfire-lds-standalone-artifacts/
/tmp/hipfire-lds-standalone-artifacts-v2/
/tmp/hipfire-lds-gemm-standalone-artifacts/
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

## Current Narrowing

Evidence argues against these as sole causes:

- `__launch_bounds__` second argument.
- Simple LDS bank-layout issue: row padding still fails.
- A-side vs B-side address math: A-only and B-only LDS both fail.
- LDS allocation size alone: tiny 4x4 active LDS inside an 8x8 block passes.
- Multi-wave `__syncthreads()` alone: 4x4 active LDS inside 8x8 block passes.
- Multi-wave LDS store/load/barrier alone: standalone HIP LDS-only kernels pass
  for `TILE=6`, `TILE=8`, and `TILE=16` at 100 launches with 64x64 grids.
- hipfire Rust runtime/JIT/dispatch as the root cause: a standalone HIP GEMM
  repro using `hipcc` and direct `hipLaunchKernelGGL` still fails.
- Actual global memory access as the root cause: a compile-time standalone
  synthetic `TILE=6` kernel with no global loads and no C/global store still
  fails once K-loop work and grid size are high enough.
- Long LDS loop alone as the root cause: the simpler LDS-only `TILE=6` probe
  with 512 iterations passes at 100 launches and a 64x64 grid.

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
- The active4-in-8x8 control passed even though the launched block spans two
  waves; only 16 lanes actively touch LDS. This keeps the current hypothesis on
  active LDS traffic across waves rather than barrier presence alone.

Best current hypothesis:

> On gfx1103 with this ROCm/amdgpu stack, the failure is a multi-wave,
> GEMM-shaped LDS loop/grid-duration/cumulative-launch fault, not a plain
> global-memory bug and not LDS/barriers in isolation. The sharp lane boundary
> remains `TILE=5` (25 active lanes, pass) vs `TILE=6` (36 active lanes, fail).
> A no-global, no-store, compile-time synthetic GEMM-shaped kernel still
> reproduces HIP 719 and the same MES/GDS coredump state, but only when K-loop
> work, grid size, and repeated launches cross a narrow, reset-sensitive
> threshold band.

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
- create a single-instantiation compile unit for the failing synthetic symbol
  and the passing long-loop symbol so instruction counts can be per-symbol
  instead of object-aggregate.
- test whether removing the exec-mask regions from the LDS-only control moves
  its failure boundary; adding similar regions to the synthetic kernel did not
  help.

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
