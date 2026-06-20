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

Cross-system 780M repro command for the current split dynamic row-load control:

```bash
VARIANT=tile6_lds_store_then_load_dynamiccols_load5_split4_use5 \
MODE=full \
N_LAUNCH=100 \
M=512 \
N=3072 \
K=3072 \
K_LIMIT=0 \
scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-split4-780m
```

Compare with the adjacent controls:

```bash
VARIANT=tile6_lds_store_then_load_dynamiccols_load4_use4 MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-load4-780m
VARIANT=tile6_lds_store_then_load_dynamiccols_load5_serial_use5 MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-serial5-780m
VARIANT=tile6_lds_store_then_load_dynamiccols_load5_split3_use5 MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-split3-780m
VARIANT=tile6_lds_store_then_load_dynamiccols_load5_split2_2_1_use5 MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-split2-2-1-780m
VARIANT=tile6_lds_store_then_load_dynamiccols_load5_split1_keep5 MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-split1-keep5-780m
VARIANT=tile6_lds_store_then_load_dynamiccols_load4_split1_keep4 MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-split1-keep4-780m
VARIANT=tile6_lds_store_then_load_dynamiccols_load3_split1_keep3 MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-split1-keep3-780m
VARIANT=tile6_lds_store_then_load_dynamiccols_load4_split1_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-consume4-pinned-780m
VARIANT=tile6_lds_store_then_load_dynamiccols_load3_split1_consume3_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-consume3-pinned-780m
VARIANT=tile6_lds_store_then_load_dynamiccols_load4_noextra_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-noextra-consume4-pinned-780m
VARIANT=tile6_lds_store_then_load_dynamiccols_load3_noextra_consume3_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-noextra-consume3-pinned-780m
VARIANT=tile6_lds_single_store_then_load_dynamiccols_load4_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-single-store-consume4-pinned-780m
VARIANT=tile6_lds_load_then_store_dynamiccols_load4_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-load-then-store-consume4-pinned-780m
VARIANT=tile6_lds_preloop_load_then_store_dynamiccols_load4_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-preloop-load-consume4-pinned-780m
VARIANT=tile6_lds_store_then_load_dynamiccols_load4_nextrow_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-nextrow-consume4-pinned-780m
VARIANT=tile6_lds_store_then_rowload_then_extra_load4_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-postrow-extra-consume4-pinned-780m
VARIANT=tile6_lds_store_then_rowload_barrier_noextra_load4_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-postrow-barrier-noextra-consume4-pinned-780m
VARIANT=tile6_lds_prestore_barrier_noextra_load4_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-prestore-barrier-noextra-consume4-pinned-780m
VARIANT=tile6_lds_preloop_barrier_noextra_load4_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-preloop-barrier-noextra-consume4-pinned-780m
VARIANT=tile6_lds_firstiter_barrier_noextra_load4_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-firstiter-barrier-noextra-consume4-pinned-780m
VARIANT=tile6_lds_single_store_prestore_barrier_noextra_load4_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-single-store-prestore-barrier-noextra-consume4-pinned-780m
VARIANT=tile6_lds_prestore_barrier_noextra_load3_consume3_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-prestore-barrier-noextra-consume3-pinned-780m
VARIANT=tile6_lds_betweenstore_barrier_noextra_load4_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-betweenstore-barrier-noextra-consume4-pinned-780m
VARIANT=tile6_lds_store_then_prerow_barrier_noextra_load4_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-prerow-barrier-noextra-consume4-pinned-780m
VARIANT=tile6_lds_store_then_rowload_barrier_gap_noextra_load4_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-postrow-barrier-gap-noextra-consume4-pinned-780m
VARIANT=tile6_lds_single_store_rowload_barrier_noextra_load4_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-single-store-postrow-barrier-noextra-consume4-pinned-780m
VARIANT=tile6_lds_store_then_rowload_barrier_noextra_load3_consume3_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-postrow-barrier-noextra-consume3-pinned-780m
VARIANT=tile6_lds_store_then_load_dynamiccols_load4_separate_tile_consume4_pinned MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-separate-tile-consume4-pinned-780m
```

The currently promoted standalone GEMM jig lives in the repo:

- `scripts/lds_gemm_standalone_probe.hip`: direct HIP tiled GEMM and reduced
  LDS stress variants, including the dynamic row-load split controls.
- `scripts/lds_gemm_standalone_matrix.sh`: builds/runs one selected variant and
  preserves logs, code objects, ISA/readobj output, dmesg snapshots, git
  revision, host/kernel details, HIP/ROCm tool versions, and relevant runtime
  environment variables. Set `BUILD_ONLY=1` to capture build/codegen metadata
  without launching the generated HIP binary.
- `scripts/lds_gemm_isa_compare.sh`: compile-only helper that builds the
  standalone probe, does not launch any GPU kernel, and writes per-symbol
  resource/ISA counters for the promoted scalar-control comparison.
  Set `SINGLE_INSTANTIATION=1` to compile each mapped kernel in its own
  generated source/object. The summary also records scalar-control placement
  around the first LDS store and the final barrier/backedge window
  (`pre_ds_s_nop`, `pre_ds_s_add_i32`, `tail_s_nop`, `tail_s_add_i32`,
  `tail_window`) so pass/fail controls can be compared without manually
  reading the disassembly.
- `scripts/lds_gemm_isa_summary_compare.sh`: read-only comparator for two
  `isa-summary.tsv` files. It classifies rows as `same`, `placement-drift`,
  `lds-control-drift`, `resource-drift`, `size-drift`, or missing on one side,
  with explicit left/right tail-window and pre-DS/tail scalar counts.
- `scripts/lds_gemm_artifact_summary.sh`: read-only artifact summarizer for
  cross-machine repro results. It writes TSV and Markdown summaries from
  `meta.txt`, `run.log`, `exit_code.txt`, dmesg deltas, devcoredump presence,
  saved ISA/readobj files, key devcoredump fault fields, and short SHA-256
  hashes for the captured HIP source, AMDGPU object, raw AMDGPU ISA dump,
  normalized AMDGPU ISA dump, and the selected variant's normalized ISA section
  when it can map the variant name to a generated kernel symbol. The normalized
  whole-ISA hash strips the disassembly file-format line because it embeds the
  artifact path and otherwise produces false codegen-drift reports. Dmesg
  deltas are counted with a multiset difference between `dmesg.before.txt` and
  `dmesg.after.txt`, which handles kernel ring-buffer snapshots while
  preserving repeated new reset messages. GFXHUB/GCVM protection status plus
  GDS/GDS-VM devcoredump registers are decoded with the gfx11 masks from
  `/usr/src/amdgpu-6.19.0-2307534.24.04/amd/include/asic_reg/gc/gc_11_0_3_sh_mask.h`.
- `scripts/lds_gemm_summary_compare.sh`: read-only TSV comparator for two
  artifact summaries. It compares selected-variant ISA hashes first, then whole
  normalized ISA hashes, then raw object/disassembly hashes. It classifies
  differences as source drift, codegen drift, same-codegen runtime difference,
  same-result environment difference, codegen metadata drift, or same. The
  compare output also includes `dmesg_sig`, `devcore_sig`, and `gcvm_sig`
  checks so a second 780M run can show whether it hit the same
  remove-queue/MODE2/GDS dmesg family, raw gfxhub/GDS devcoredump signature,
  and decoded GCVM flags/CID/RW/VMID.
- `scripts/lds_gemm_780m_runbook.sh`: command-only runbook printer for the
  second-780M flow: safe build-only preflight, risky repro, optional K-edge
  repro, summary regeneration, runtime summary comparison, and placement-aware
  ISA summary comparison.
- `scripts/narrow-gemm-lds-shape.sh`: runs the curated variant matrix and
  writes a TSV report plus summary.
- `scripts/lds_standalone_probe.hip`: promoted LDS-only store/load/barrier
  stress probe used for the long-loop masked/no-mask controls; the file has a
  `HIPFIRE_LDS_STANDALONE_NO_MAIN` guard so compile-only helpers can emit
  explicit single-instantiation objects.
- `scripts/lds_standalone_isa_compare.sh`: read-only compile/ISA summarizer
  for the LDS-only long-loop probe. It defaults to `SINGLE_INSTANTIATION=1`
  and writes per-symbol size, instruction, DS, barrier, wait, exec-mask,
  global-store, and code-object resource metadata.
- `scripts/lds_direct_ab_phase_probe.hip`: promoted direct-AB no-output
  reduced repro. It keeps two LDS arrays and repeated active-lane reads while
  removing GEMM global-memory traffic.
- `scripts/lds_direct_ab_multi_exec_parent.cpp`: parent process that runs the
  phase probe through one or more fork/exec children so child-local launch
  sequence length can be separated from total parent-supervised work.
- `scripts/lds_direct_ab_multi_exec_matrix.sh`: build/run wrapper for the
  direct-AB multi-exec repro. It defaults to `BUILD_ONLY=1`, compiles the child
  and parent, captures ISA/readobj artifacts, and does not launch the risky
  repro unless `BUILD_ONLY=0` is explicitly set. `LAYOUT_X` / `LAYOUT_Y` can
  be set independently of `ACTIVE_X` / `ACTIVE_Y` to pad the A/B LDS arrays
  while preserving the active-lane shape.
- `scripts/lds_direct_ab_780m_test_jig.sh`: one-command handoff jig for a
  second gfx1103/780M system. The default `--build-only` mode safely compiles
  the direct-AB active-shape controls and captures codegen artifacts under
  `/tmp/hipfire-lds-direct-ab-780m-buildonly`. `--risky` runs the current
  focused READS=2 sequence: 8x4 one-child pass control, 9x4 one-child
  pass-side control, 9x4 split-child pass control, then the 9x4 one-child
  fail-side repro. It writes `report.tsv`, `summary.txt`, and
  `direct-ab-artifact-summary.tsv/.md`, and `--compare` delegates to the
  direct-AB summary comparator for local-vs-remote result checks.
- `scripts/lds_direct_ab_artifact_summary.sh`: read-only summarizer for
  direct-AB artifact roots. It writes `direct-ab-artifact-summary.tsv/.md` with
  shape/chunk metadata, exit/sync failure, code-object hashes/resources, dmesg
  deltas, decoded gfxhub/GCVM/GDS coredump fields, and compact ISA counters
  (`s_barrier`, DS ops, `s_waitcnt`, scalar branches, unique
  `ds_store_2addr_b32 offset1` values, and non-default layout).
- `scripts/lds_direct_ab_summary_compare.sh`: read-only comparator for two
  direct-AB summary TSVs. It compares source/code-object hashes, normalized ISA,
  resource tuples, build/risk mode, runtime exit/sync result, environment, dmesg
  deltas, and devcore/GCVM/GDS signatures.
- `tests/gfx1103-lds-tail-snop-repro.sh`: focused cross-system pass/fail
  wrapper for a second 780M. The default `PROFILE=repro` checks the no-extra
  baseline against the tail-loop `s_nop` repro and writes a TSV report under
  `/tmp/hipfire-lds-tail-snop-repro`; it also invokes the artifact summarizer
  at the end of the run. With `BUILD_ONLY=1`, runtime pass/fail expectations
  are intentionally bypassed and each case is judged only on whether it built
  and captured artifacts successfully.

Older throwaway HIP probes used during the investigation:

- `/tmp/hipfire-lds-repro/lds_standalone_probe.hip`: LDS-only shared-memory
  store/load/barrier stress kernels, no GEMM global matrix traffic. This source
  is now promoted as `scripts/lds_standalone_probe.hip`.
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
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 480x86 grid, one child `99`/`100`/`101`/`102`/`103`/`104` | PASS | Mid-grid pass side; lowering grid_x moved the one-child edge upward. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 480x86 grid, one child `105`/`120` | FAIL | One-child `105` failed at sync/global launch 103; one-child `120` failed at sync/global launch 104. Both produced the same coredump signature. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 480x86 grid, chunks `104,16` / `60,60` | PASS | Total 120 split across child processes; `104,16` passed with the first child exactly on the pass side. Both split shapes passed for plain and HIP-initialized parent modes. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 448x86 grid, one child `105`/`120` | PASS | Lower grid moved the initial pass side upward. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 448x86 grid, one child `121`/`122`/`130`/`160` | FAIL | After reset pressure, 121 failed at sync/global launch 114; 122/130/160 failed at launch 112-115. Same coredump signature. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 448x86 grid, chunks `120,40` | FAIL | Failed inside the first 120-launch child in both plain and HIP-initialized parent modes, showing the earlier 120 pass was state-sensitive. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 448x86 grid, chunks `80,80` | PASS | Same total 160 as failing one-child `160`; passed in both plain and HIP-initialized parent modes. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 416x86 grid, one child `120`/`122`/`124` | PASS | Lower grid pass side. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 416x86 grid, one child `125`/`126`/`130` | FAIL | One-child `125`/`126` failed at sync/global launch 120; one-child `130` failed at launch 124. Same coredump signature. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 416x86 grid, chunks `124,36` | FAIL | Failed inside the first 124-launch child in both plain and HIP-initialized parent modes, showing the earlier 124 pass became state-sensitive after reset pressure. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 416x86 grid, chunks `80,80` | PASS | Same total 160 as failing one-child runs; passed in both plain and HIP-initialized parent modes. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 384x86 grid, one child `125`/`128`/`130`/`132`/`134` | PASS | Lower grid pass side; 134 remained the highest preserved one-child pass before repeat failure at 135. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 384x86 grid, one child `135` | FAIL | Failed twice: first at sync/global launch 132, repeat at 133. Same coredump signature. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 384x86 grid, one child `134`/`135` (fresh) | PASS at `134`, FAIL at `135` | Fresh same-work rerun reproduced the same edge family and supports the same process-state behavior as 352/320 family. |
| Direct-AB multi-exec `5x5`, reads=3, 448 iterations, 384x86 grid, one child `150` | PASS | Active-lane reduction to 25/24-thread blocks survives same total at the 384x86 edge. |
| Direct-AB multi-exec `4x4`, reads=3, 448 iterations, 384x86 grid, one child `200` | PASS | 16-thread variant survives in same geometry/iterness where 6x6 fails. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 384x86 grid, chunks `134,46` | FAIL | Failed inside the first 134-launch child in both plain and HIP-initialized parent modes, showing the earlier 134 pass became state-sensitive after reset pressure. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 384x86 grid, chunks `90,90` | PASS | Same total 180 as failing near-edge split; passed in both plain and HIP-initialized parent modes. |
| Direct-AB multi-exec `5x5`, reads=3, 448 iterations, 384x86 grid, one child `260`/`280`/`300`/`320` | PASS / FAIL / FAIL / PASS | Fresh 260 passed; 280/300 failed at sync 264; 320 passed, reinforcing state-sensitive non-monotonicity even with 25 lanes. |
| Direct-AB multi-exec `4x4`, reads=3, 448 iterations, 384x86 grid, one child `270`/`278`/`279` | PASS / PASS / FAIL | 4x4 passes through 278 and fails at 279 in fresh one-child runs. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 352x86 grid, one child `148`/`149`/`150`/`151`/`152` | PASS at `150` in the initial run; plain-mode fresh rerun had `148`/`149` fail, `150` pass, `151`/`152` fail; `hipinit_reset_before` mode had all fail | State-sensitive narrow pass window with strong process-state sensitivity. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 352x86 grid, chunks `105,45` | PASS | Split child work passed in both plain and HIP-initialized parent modes. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 320x86 grid, one child `156`/`160`/`162` | PASS | Lower grid moves the one-child pass side higher. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 320x86 grid, one child `163`/`164`/`165`/`166` | PASS at `163`/`164` in the initial run; fresh process rerun shows fresh non-monotonicity (`162` fail in one plain-run, while `163`/`164` passed) and `>=162` fail in `hipinit_reset_before` | Band no longer deterministic, reinforcing process-state coupling. |
| Direct-AB multi-exec `6x6`, reads=3, 448 iterations, 320x86 grid, chunks `98,67` / `80,85` | PASS | Split child work passed in both plain and HIP-initialized parent modes. |
| Direct-AB no-output `8x4` active/block, reads=6, 512 iterations, 512x86 grid | PASS | Exact one-wave, two-array control. |
| Direct-AB no-output `8x4` active inside `8x5` block, reads=6, 512 iterations, 512x86 grid | PASS | Two-wave block with 32 active lanes; still stable. |
| Direct-AB no-output `5x5`/`4x4` active inside `6x6` block, reads=6, 512 iterations, 512x86 grid | PASS | Small active controls remain stable without cooperative producer loops. |

### Dispatch Geometry Control

- `c3765ea9` and `b41368bb` are correct with a `TILE=16` kernel launch of
  `ceil(N/16) × ceil(M/16)`; this is what those commits contain.
- For comparison, I built a throwaway control where that LDS kernel was forced to
  `ceil(N/64) × ceil(M/64)`. It still ran and did not hit a fault in 100 launches,
  but this mode covers only 1/4 of the intended output columns (`3072` would be
  covered as `768`), so the work profile changed.
- Returning to the commit-correct `ceil(N/16)` launch in the same throwaway setup
  reproduced the fault (unrecoverable by retry around launch ~29) with the same
  MES reset / `gfxhub` + GDS protection-fault behavior.
- The throwaway `gemm_f32_train_recover` harness is also state-sensitive on the
  same LDS-backed shape: `HIPFIRE_LAUNCHES=120` failed unrecoverably at launch
  28 after 8 retries, while `HIPFIRE_LAUNCHES=27` failed at launch 13 without
  profiling and at launch 8 under `rocprofv3`. The driver signature remained the
  same `MES failed to respond to msg=REMOVE_QUEUE` / reset / coredump path.
- A standalone LDS-only `lds_minimal_probe` at `TILE=6`, `ITERS=320`,
  `512x86`, and 100 launches completed successfully, so the bare LDS loop by
  itself is not enough.
- The standalone GEMM-shaped `lds_gemm_standalone_probe` at
  `tile6 full 100 512 3072 3072` failed at sync 19 with HIP 719 and the same
  MES `REMOVE_QUEUE` reset path, showing the kernel shape alone is sufficient
  outside hipfire.
- A laddered standalone narrowing run now shows `tile5 full` passes at
  `512x3072x3072` while every `tile6` variant in the ladder fails at the same
  `100`-launch configuration, including `tile6_synth`, `tile6_synth_masked`,
  `nostore`, `noglobal`, `aonly`, `bonly`, and the full GEMM-shaped variant.
  The minimum distinguishing factor is therefore still the `tile6` GEMM shape,
  not the hipfire runtime.
- A subsequent `K_LIMIT` bisect on the standalone GEMM probe found a nominal
  `1651` pass / `1652` fail bracket, but immediate reruns of both values also
  failed, so the exact loop-depth edge is still state-sensitive. Treat that
  cutoff as provisional rather than fixed.
- Direct single-axis bisection at 56 launches now gives provisional brackets of
  `M=507` pass / `508` fail, `N=3041` pass / `3042` fail, and `K=3020` pass /
  `3021` fail for the standalone `tile6_synth` probe. The original
  `512x3072x3072` control still fails at sync 55, but a combined reduced
  control can pass, so these cutoffs describe the current edge family rather
  than an independent axis model.
- In the current standalone GEMM probe state, launch counts `50` and `55`
  pass, while `56` and `60` fail, so the reusable-process edge is now tightly
  bracketed between `55` and `56` launches for the `tile6_synth` shape.
- A throwaway same-shape kernel-body split in
  `/tmp/hipfire-lds-chaingun` sharpened the LDS boundary. At
  `512x3072x3072`, 100 launches, and the same `6x6` block / `ceil(dim/6)`
  grid:
  - `tile6_nolds_synth` passed. This keeps launch geometry, loop count, and
    arithmetic, but removes LDS and barriers.
  - `tile6_barrier_synth` passed. This keeps the two barriers per K tile but
    still removes LDS.
  - `tile6_barrier3_synth` passed. This keeps three barriers per K tile but
    still removes LDS (`group_segment_fixed_size=0`).
  - `tile6_lds_one_synth` passed. This adds one `6x6` shared tile
    (`group_segment_fixed_size=144`) and reads it back.
  - `tile6_lds_padded_one_synth` passed. This uses one padded shared array
    with `group_segment_fixed_size=288`, but still has only one per-K LDS
    store/read stream.
  - `tile6_lds_two_store_once_one_read` passed. This allocates two `6x6`
    shared tiles and writes the second tile once before the K loop, then only
    uses the first tile inside the loop.
  - `tile6_lds_forced_same_second_store` failed with HIP 719 at sync 92. This
    uses one `6x6` tile (`group_segment_fixed_size=144`) but forces a per-K
    `ds_store -> barrier -> ds_load/ds_store -> barrier` sequence on the same
    LDS addresses before the regular inner-product reads.
  - `tile6_lds_forced_same_load_only` passed. It keeps the first store, an
    extra same-address `ds_load`, three barriers per K tile, and the regular
    six-wide row readback, but does not write the loaded value back.
  - `tile6_lds_same_phase_no_wide_read` passed. It keeps the same-address
    `ds_store -> barrier -> ds_load/ds_store -> barrier` phase, but removes the
    regular six-wide row readback and uses the loaded value directly.
  - Repeating `tile6_lds_forced_same_second_store` in that same compile unit
    failed again, this time at sync 91.
  - A read-width split then showed `tile6_lds_forced_same_read1`,
    `tile6_lds_forced_same_read2`, `tile6_lds_forced_same_read4`, and
    `tile6_lds_forced_same_read5` all pass at 100 launches. These keep the
    same-address load/store phase and vary only the number of post-phase row
    reads. The six-wide `tile6_lds_forced_same_second_store` failed twice in
    that sequence, both times at sync 0 after prior reset pressure.
  - `tile6_lds_second_store_only_read6` passed. It has the same resource shape
    as the failing forced-same-address splitter (`group_segment_fixed_size=144`,
    `sgpr_count=5`, `vgpr_count=12`) and emits `ds_store -> barrier ->
    ds_store -> barrier -> six-wide row readback`, but it does not load from LDS
    in the second producer phase.
  - `tile6_lds_load_independent_store_read6` failed with HIP 719 at sync 90.
    It emits `ds_store -> barrier -> ds_load -> ds_store -> barrier ->
    six-wide row readback`, but the second store writes an independent value
    rather than the loaded value. The load-to-store data dependency is not
    required for the reset trigger.
  - Address-split throwaway variants both failed. `tile6_lds_load_next_store_same_read6`
    failed with HIP 719 at sync 0 after prior reset pressure; a recovery pass
    control then succeeded, and `tile6_lds_load_same_store_next_read6` failed
    with HIP 719 at sync 89. The emitted ISA keeps ordinary `ds_store_b32`,
    `ds_load_b32`, `ds_store_b32`, barriers, and six-wide row readback. This
    rules out an exact same-address LDS load/store alias as a requirement.
  - The order split `tile6_lds_store_then_load_read6` also failed with HIP 719.
    It emits ordinary `ds_store_b32 -> barrier -> ds_store_b32 -> barrier ->
    ds_load_b32 -> six-wide row readback`, uses one 144 B LDS tile, and keeps
    `sgpr_count=5` / `vgpr_count=12`. Therefore a load-before-second-store
    ordering is not required either; a second LDS store followed by an ordinary
    LDS load before the full row consumption is enough to reproduce the reset.
  - The store-then-load read-width split showed
    `tile6_lds_store_then_load_read0`, `read1`, `read2`, `read4`, and `read5`
    all pass at 100 launches. The six-wide `tile6_lds_store_then_load_read6`
    failed at sync 91 in the same sequence. This preserves the six-wide row
    consumption threshold for the store-before-load ordering.
  - The dynamic-column row-read split
    `tile6_lds_store_then_load_dynamiccols_read6` failed at sync 85 under the
    100-launch stress envelope after passing a one-launch smoke. It keeps the
    same one-tile store-then-load producer pattern, but hides the six row
    column constants behind empty inline asm so the row readback emits six
    separate `ds_load_b32` instructions instead of the packed
    `ds_load_2addr_b64` / `ds_load_b64` sequence. A recovery
    `tile6_lds_two_store_once_one_read` control passed.
  - The dynamic-column load/use split then found a sharper independent-load
    boundary. `tile6_lds_store_then_load_dynamiccols_load4_use4` passed at 100
    launches, while `tile6_lds_store_then_load_dynamiccols_load5_use5` failed
    at sync 92. Throwaway auxiliary controls also failed:
    `dynamiccols_load6_use5` at sync 82 and `dynamiccols_load5_use6` at sync
    91. This means the old static read-width boundary is instruction-form
    dependent: packed/static read5 still passes, but five independent
    dynamic-address row `ds_load_b32` operations are enough to reproduce the
    reset.
  - Serializing those five dynamic row loads with explicit
    `s_waitcnt lgkmcnt(0)` after each load flips the result back to pass:
    `tile6_lds_store_then_load_dynamiccols_load5_serial_use5` passed at 100
    launches. The non-serialized `dynamiccols_load5_use5` contrast failed at
    sync 91 in the same compile unit. Later reset-pressure testing found the
    same serialized control can fail at sync 87 after several nearby
    reset-producing variants, so treat this pass as a clean-state contrast, not
    an unconditional workaround.
  - The promoted split wait control
    `tile6_lds_store_then_load_dynamiccols_load5_split4_use5` issues four
    dynamic row `ds_load_b32` operations, waits, then issues the fifth. It
    failed at sync 92 under the same 100-launch stress envelope, while the
    recovery control passed afterward. That rules out the fully packed five
    load group as the only bad shape; a four-load outstanding group followed by
    another dynamic row load is still sufficient on this local 780M.
  - Further promoted split wait controls also failed:
    `tile6_lds_store_then_load_dynamiccols_load5_split3_use5` at sync 91,
    `tile6_lds_store_then_load_dynamiccols_load5_split2_3_use5` at sync 92,
    and `tile6_lds_store_then_load_dynamiccols_load5_split2_2_1_use5` at sync
    92. All passed one-launch smoke, kept `144` B LDS,
    `sgpr_count=5`, and `vgpr_count=15`, and produced the same MES reset path.
    This rules out a simple peak outstanding row-load threshold of four, three,
    or two.
  - The max-one-outstanding delayed-consume control
    `tile6_lds_store_then_load_dynamiccols_load5_split1_keep5` also failed at
    sync 87. It waits after every dynamic row load but delays the final
    accumulation until all five loaded values are live. That showed
    wait-after-each-load is not sufficient when the values remain live for
    delayed consumption. Recovery controls passed between reset-producing runs.
  - The live-value threshold split sharpens that model. A known pass-side
    sanity run of `tile6_lds_store_then_load_dynamiccols_load4_use4` passed at
    100 launches immediately before the test. Then
    `tile6_lds_store_then_load_dynamiccols_load4_split1_keep4` failed at sync
    96 even though it waits after each of four dynamic row loads and only
    delays accumulation. After recovery,
    `tile6_lds_store_then_load_dynamiccols_load3_split1_keep3` passed at 100
    launches with the same producer phase and delayed consumption pattern. The
    local delayed-consume boundary is therefore between three and four live
    dynamic row values.
  - Pinned immediate-consume controls kept the same 3/4 boundary while removing
    delayed live-value pressure. `tile6_lds_store_then_load_dynamiccols_load4_split1_consume4_pinned`
    emits an FMAC after each waited dynamic row load, uses only
    `vgpr_count=10`, and still failed at sync 89. The matching three-load
    pinned control passed at 100 launches with `vgpr_count=9`. This moves the
    local model away from delayed live values as the sole cause and toward a
    four dynamic row-load / explicit wait / interleaved consume threshold after
    the second-store/extra-load producer phase.
  - No-extra-load pinned controls passed on both sides of that boundary. The
    four-load no-extra pinned variant passed 100 launches with the same
    `group_segment_fixed_size=144`, `sgpr_count=5`, and `vgpr_count=10` as the
    failing extra-load four-load pinned variant. The three-load no-extra pinned
    variant also passed 100 launches with `vgpr_count=9`. This shows the extra
    ordinary LDS load remains a required trigger component for the four-load
    pinned failure on this system.
  - The single-store pinned extra-load control passed 100 launches. It keeps
    the extra ordinary LDS load and the four waited/interleaved dynamic row
    loads, but removes the second per-K LDS store. It matches the failing
    second-store pinned resource tuple (`group_segment_fixed_size=144`,
    `sgpr_count=5`, `vgpr_count=10`). This shows the second per-K LDS store
    remains a required trigger component in the current minimized shape.
  - The pinned load-then-store order split failed at sync 91. It emits first
    store -> barrier -> extra ordinary LDS load -> second store -> barrier ->
    four waited/interleaved dynamic row loads, with the same `144`,
    `sgpr_count=5`, `vgpr_count=10` resource tuple. This shows the extra load
    does not need to happen after the second store in the pinned minimized
    shape; the second store must still be present before the four dynamic row
    loads.
  - The preloop-only extra-load control passed 100 launches. It performs one
    extra ordinary LDS load before entering the K loop, then each K iteration
    does the two stores and four waited/interleaved dynamic row loads with no
    per-K extra load. This indicates the extra LDS load must be per-K/proximate
    to the row-consumption phase, not merely present somewhere earlier in the
    kernel. Caveat: the preloop setup changes the tuple to `144`,
    `sgpr_count=6`, `vgpr_count=11`.
  - The pinned next-row extra-load control failed at sync 86. It keeps the same
    one-tile second store and four waited/interleaved dynamic row loads, but
    changes only the extra ordinary LDS load from `As[ty][tx]` to
    `As[(ty + 1) % TILE][tx]`. This rules out same-row membership between the
    extra ordinary LDS load and the four dynamic row loads as a required
    trigger component in the pinned form. Caveat: the computed next-row address
    raises the VGPR count to 13.
  - The pinned separate-tile extra-load control failed at sync 88. It
    initializes `Xs` once, keeps the per-K second store and four waited /
    interleaved dynamic row loads on `As`, and changes only the extra ordinary
    LDS load to `Xs[ty][tx]`. This shows the extra ordinary LDS load does not
    need to come from the same LDS tile as the four dynamic row loads in the
    pinned four-load shape. Caveat: this uses `group_segment_fixed_size=288`
    and `sgpr_count=6`.
  - The pinned post-row extra-load control is now tracked as
    `tile6_lds_store_then_rowload_then_extra_load4_consume4_pinned`. It keeps
    the two per-K stores and four waited/interleaved dynamic row loads, but
    moves the extra ordinary LDS load after the four row loads. The 100-launch
    stress run failed at sync 88 with HIP 719 and the same MES `REMOVE_QUEUE`
    reset family; a follow-up one-launch `load4_noextra_consume4_pinned`
    recovery smoke passed. This shows the per-K/proximate extra LDS load does
    not need to occur before the row-consumption phase.
  - The no-extra post-row double-barrier control
    `tile6_lds_store_then_rowload_barrier_noextra_load4_consume4_pinned`
    failed at sync 94 with HIP 719. It removes the extra ordinary LDS load but
    adds a second closing `__syncthreads()` after the four waited/interleaved
    row loads. ISA emits first store -> barrier -> second store -> barrier ->
    four row loads/FMAs -> barrier -> barrier, with no intervening LDS load.
    Metadata is the same tight `144`, `sgpr_count=5`, `vgpr_count=10` tuple.
    The existing one-closing-barrier `load4_noextra_consume4_pinned` recovery
    smoke passed afterward. This shows the extra ordinary LDS load is not the
    only fail-side post-consumption operation; an additional post-row barrier
    epoch can substitute under the stress envelope.
  - The no-extra pre-row extra-barrier control
    `tile6_lds_store_then_prerow_barrier_noextra_load4_consume4_pinned`
    also failed at sync 94 with HIP 719. It keeps the same two stores and four
    waited/interleaved row loads, but moves the fourth barrier to immediately
    before row consumption: first store -> barrier -> second store -> barrier
    -> barrier -> four row loads/FMAs -> barrier. Metadata is still `144`,
    `sgpr_count=5`, `vgpr_count=10`, and no spills. This shows the added
    barrier epoch does not need to be post-consumption.
  - The no-extra prestore extra-barrier control
    `tile6_lds_prestore_barrier_noextra_load4_consume4_pinned` failed at sync
    93 with HIP 719. It moves the fourth barrier to the start of each K
    iteration, before any per-K LDS store: barrier -> first store -> barrier
    -> second store -> barrier -> four row loads/FMAs -> barrier. Metadata
    remains `144`, `sgpr_count=5`, `vgpr_count=10`, and no spills. This shows
    the added barrier epoch does not need to follow an LDS producer phase.
  - The one-time pre-loop barrier control
    `tile6_lds_preloop_barrier_noextra_load4_consume4_pinned` passed 100
    launches. It keeps the two stores and four waited/interleaved row loads,
    but moves the extra barrier outside the K loop so the repeated
    per-iteration sync epoch is removed. Metadata is `144`, `sgpr_count=6`,
    `vgpr_count=10`, and no spills. This shows the prestore barrier failure
    requires more than the presence of one extra barrier somewhere in the
    kernel.
  - The first-iteration-only barrier control
    `tile6_lds_firstiter_barrier_noextra_load4_consume4_pinned` passed 100
    launches. It keeps the extra barrier inside the K loop but guards it with
    `kt == 0`, leaving later K iterations at the normal two-store, four-load,
    one-closing-barrier shape. Metadata is `144`, `sgpr_count=5`,
    `vgpr_count=11`, and no spills. This distinguishes one-time in-loop sync
    from repeated per-K sync: the failing prestore barrier shape needs the
    extra sync epoch to recur across K iterations.
  - The single-store prestore extra-barrier control
    `tile6_lds_single_store_prestore_barrier_noextra_load4_consume4_pinned`
    passed 100 launches with `144`, `sgpr_count=5`, `vgpr_count=10`, and no
    spills. It keeps four barriers in the K iteration but removes the second
    per-K LDS store: barrier -> first store -> barrier -> barrier -> four row
    loads/FMAs -> barrier. This keeps the second-store requirement intact for
    the early-barrier branch and rules out four same-loop barriers alone.
  - The two-store prestore extra-barrier three-load control
    `tile6_lds_prestore_barrier_noextra_load3_consume3_pinned` passed 100
    launches with `144`, `sgpr_count=5`, `vgpr_count=9`, and no spills. This
    keeps the four-row-load width requirement intact for the early-barrier
    branch.
  - The no-extra between-store extra-barrier control
    `tile6_lds_betweenstore_barrier_noextra_load4_consume4_pinned` failed at
    sync 94 with HIP 719. ISA emits first store -> barrier -> barrier ->
    second store -> barrier -> four row loads/FMAs -> barrier, with the same
    `144`, `sgpr_count=5`, `vgpr_count=10` tuple and no spills. This shows the
    extra sync epoch can sit between the two producer phases and still trigger.
  - The gapped post-row double-barrier control
    `tile6_lds_store_then_rowload_barrier_gap_noextra_load4_consume4_pinned`
    failed at sync 94 with HIP 719. ISA emits first store -> barrier -> second
    store -> barrier -> four row loads/FMAs -> barrier -> `s_nop 0` ->
    barrier, with the same `144`, `sgpr_count=5`, `vgpr_count=10` tuple and
    no spills. This rules out immediate adjacent closing barriers as the
    required trigger component.
  - The single-store no-extra double-barrier control
    `tile6_lds_single_store_rowload_barrier_noextra_load4_consume4_pinned`
    passed 100 launches with the same `144`, `sgpr_count=5`,
    `vgpr_count=10` tuple as the failing two-store double-barrier control.
    This keeps the second per-K LDS store as a required component for the
    double-barrier branch, not only for the extra-load branch.
  - The two-store no-extra double-barrier three-load control
    `tile6_lds_store_then_rowload_barrier_noextra_load3_consume3_pinned`
    passed 100 launches with `144`, `sgpr_count=5`, and `vgpr_count=9`.
    This keeps the row-load width boundary at four dynamic row loads for the
    double-barrier branch.
  - The cross-row load split `tile6_lds_store_then_load_nextrow_read6` also
    failed under the 100-launch stress envelope, at sync 88. Clean follow-up
    runs passed at `N_LAUNCH=1` and `N_LAUNCH=2`, so treat the earlier
    reset-adjacent two-launch failure as contaminated. The useful boundary is
    that the extra ordinary LDS load can target a different row from the
    six-wide row readback and still reproduce the reset under stress.
  - The separate-tile load split
    `tile6_lds_store_then_load_separate_tile_read6` failed at sync 91. It keeps
    the second store and six-wide row readback on `As`, but initializes a
    separate `Xs` tile once before the K loop and performs the extra LDS load
    from `Xs`. The recovery `tile6_lds_two_store_once_one_read` control passed,
    so a second tile initialized once remains pass-side without the extra load.
  - The separate read-tile split
    `tile6_lds_store_then_load_separate_readtile` failed at sync 91 under the
    100-launch stress envelope after passing a one-launch smoke. It initializes
    `Xs` once before the K loop, then loops with `As` first store, `As` second
    store, an extra LDS load from `As`, and the six-wide row readback from
    `Xs`. The recovery `tile6_lds_two_store_once_one_read` control passed, so
    the six-wide row readback does not need to consume the same LDS tile that
    receives the second store and extra load.
  - Repeating `tile6_lds_forced_same_second_store` immediately after that pass
    failed at sync 92, preserving the read-modify-write distinction in the same
    compile unit.
  - `tile6_lds_two_store_one_read` failed with HIP 719 at sync 97. This writes
    two `6x6` shared tiles (`group_segment_fixed_size=288`) but only reads one
    tile back.
  - The existing `tile6_synth` failed with HIP 719 at sync 56 in the same
    envelope (`group_segment_fixed_size=288`).
  This points away from launch count, arithmetic, barriers alone, and a single
  LDS producer/consumer tile. The 288 B group segment alone also passes, and a
  second tile written only once passes. The smallest current failing splitter
  is not strictly "second address range"; a forced second same-address LDS
  read/write phase also fails, but only when followed by the normal six-wide
  LDS row readback. Read widths 1/2/4/5 are still pass-side, and a second
  same-address store followed by six reads is still pass-side. The better
  current model is a per-K second LDS store plus an ordinary LDS load before
  six-wide row LDS consumption; value dependency, exact address aliasing, and
  load-before-store ordering have all been ruled out as required trigger
  components. The extra LDS load also does not need to target the same row as
  the six-wide row readback, or even the same LDS tile. The six-wide row
  readback likewise does not need to target the same LDS tile as the second
  store and extra load. The separate-tile controls are less minimal because
  they use `group_segment_fixed_size=288`, but together they show same-tile
  membership is not required on either side of the split. The dynamic-column
  split rules out the packed contiguous row-load instruction form as a
  requirement; independent dynamic-address `ds_load_b32` row loads are still
  enough, and their stress threshold is lower: four pass and five fail when
  the loads are outstanding together. Clean-state immediate-consume
  serialization can pass, but split controls down to max-one outstanding still
  fail with four row-loads even when FMACs are pinned between loads. Removing
  the extra ordinary LDS load makes the four-load pinned variant pass 100
  launches with the same resource tuple, and removing the second per-K LDS
  store while keeping the extra load and four dynamic row loads also passes
  with the same tuple. Moving the extra load before the second store, to the
  next row, to a separate one-time-initialized LDS tile, or after the four
  waited/interleaved row loads still fails, but a one-time preloop extra LDS
  load passes. Separately, adding a second closing barrier after the four
  row-load consumes also fails even with the extra ordinary LDS load removed.
  The leading local suspect is now the conjunction of a second per-K LDS store,
  four dynamic row LDS loads separated by explicit waits with interleaved
  consumes, and either additional same-loop LDS activity or an additional
  same-loop synchronization epoch repeated across K iterations, with repeated
  launch/process state acting as the stress amplifier. Removing the second
  store, reducing the row-load width from four to three, or making the extra
  barrier one-time-only keeps these no-extra barrier shapes pass-side. The
  repeated prestore-barrier shape also has a high K-depth requirement:
  `K_LIMIT=2815` (about 470 loop trips) remains pass-side at 100 launches,
  while confirmation runs from `K_LIMIT=2880` upward (about 480+ loop trips)
  fail with the same reset family. The exact top-end cutoff is stress-history
  sensitive, so treat this as a work-depth envelope rather than a single
  deterministic scalar threshold. A periodic-predicate throwaway complicates
  the "extra sync" interpretation: period-512, period-256, period-128,
  period-64, period-32, period-16, and period-8 prestore-barrier controls all
  failed at full K, but a period-512 no-barrier `s_nop 0` control also failed.
  The direct first-iteration-only barrier still passed after those failures.
  This keeps recurrent same-loop scalar/control-flow perturbation in the
  suspect set alongside recurrent extra sync.
  Same-row and same-tile membership are no longer required by the current
  evidence, and neither extra-load-before-second-store nor extra-load-after-row
  ordering prevents the failure, though the separate-tile and preloop controls
  change resource tuples. Barrier count alone is still ruled out by earlier
  no-LDS and isolated phase controls, but an extra post-consumption barrier
  epoch inside this two-store/four-row-load shape is now fail-side.
- The throwaway split's resource metadata is useful: no-LDS and barrier-only
  variants use `group_segment_fixed_size=0`, one-tile LDS uses `144`, and the
  two-tile failing splitter uses `288` with the same `sgpr_count=5` and
  `vgpr_count=12` as the passing one-tile variant. The padded-one passing
  variant uses `288`, `sgpr_count=6`, and `vgpr_count=24`; the write-once
  passing variant uses `288`, `sgpr_count=6`, and `vgpr_count=12`. The failing
  forced-same-address splitter uses `144`, `sgpr_count=5`, and `vgpr_count=12`.
  The pass-side `tile6_lds_forced_same_load_only` uses the same resource counts,
  while `tile6_lds_same_phase_no_wide_read` uses `144`, `sgpr_count=5`, and
  `vgpr_count=6`. The read-width variants all keep `group_segment_fixed_size=144`
  and `sgpr_count=5`; pass-side VGPR counts rose with width (`read1=7`,
  `read2=8`, `read4=10`, `read5=11`), while the six-wide failing splitter uses
  `vgpr_count=12`. The pass-side `tile6_lds_second_store_only_read6` also uses
  `144`, `sgpr_count=5`, and `vgpr_count=12`, which rules out that resource
  tuple alone. The failing `tile6_lds_load_independent_store_read6` also uses
  `144`, `sgpr_count=5`, and `vgpr_count=12`. The address-split variants use
  `144`, `sgpr_count=5`, and `vgpr_count=13`, and both still fail when the LDS
  load and second LDS store target different columns. The order split
  `tile6_lds_store_then_load_read6` returns to `144`, `sgpr_count=5`, and
  `vgpr_count=12`, and still fails with the second store before the extra LDS
  load. The store-then-load read-width variants keep `group_segment_fixed_size=144`;
  pass-side counts are `read0 sgpr=4/vgpr=4`, then `read1=7`, `read2=8`,
  `read4=10`, and `read5=11` VGPRs with `sgpr_count=5`. The clean boundary is
  now a second LDS store plus an extra ordinary LDS load before six-wide row
  readback, not a load-to-store value dependency, exact same-address alias, or
  load-before-store ordering. The dynamic-column split keeps `144`,
  `sgpr_count=5`, and raises `vgpr_count` to `17` while replacing packed row
  LDS loads with six separate dynamic-address `ds_load_b32` operations. The
  promoted dynamic load/use controls keep `144` and `sgpr_count=5`;
  `load4_use4` passes with `vgpr_count=13`, while `load5_use5` fails with
  `vgpr_count=15`. Throwaway `load6_use5` and `load5_use6` also fail with
  `vgpr_count=16` and `15`, respectively. The serialized
  `load5_serial_use5` clean-state control passed with the same `144`,
  `sgpr_count=5`, and `vgpr_count=15` resource tuple as the failing
  non-serialized `load5_use5`, which ruled out that tuple alone in the clean
  sequence. After repeated reset-producing tests the same serialized control
  failed at sync 87, so state pressure can move this boundary. The split-wait
  `load5_split4_use5`, `split3_use5`, `split2_3_use5`, `split2_2_1_use5`, and
  `split1_keep5` controls all fail with the same `144`, `sgpr_count=5`, and
  `vgpr_count=15` tuple. `load4_split1_keep4` fails with `144`,
  `sgpr_count=5`, and `vgpr_count=13`, while `load3_split1_keep3` passes with
  `144`, `sgpr_count=5`, and `vgpr_count=11`. Pinned immediate-consume
  `load4_split1_consume4_pinned` fails with `144`, `sgpr_count=5`, and
  `vgpr_count=10`, while `load3_split1_consume3_pinned` passes with `144`,
  `sgpr_count=5`, and `vgpr_count=9`. The issue is therefore not removed by
  reducing only the outstanding dynamic row-load group size, even to one, or by
  consuming each waited value immediately. The no-extra pinned controls
  `load4_noextra_consume4_pinned` and `load3_noextra_consume3_pinned` pass with
  `144`, `sgpr_count=5`, and `vgpr_count=10`/`9`, respectively, showing the
  extra ordinary LDS load is required for the four-load pinned failure and that
  the failing `load4_split1_consume4_pinned` resource tuple is pass-side when
  that load is removed. The single-store pinned extra-load control also passes
  with `144`, `sgpr_count=5`, and `vgpr_count=10`, showing the second per-K LDS
  store is required and that the failing tuple remains pass-side when that
  store is removed. The pinned load-then-store order split fails with the same
  `144`, `sgpr_count=5`, and `vgpr_count=10` tuple, showing the extra ordinary
  LDS load does not need to follow the second store as long as the second store
  precedes the four-row consumption phase. The preloop-only extra-load control
  passes with `144`, `sgpr_count=6`, and `vgpr_count=11`, indicating the extra
  LDS load must occur per K / near the row-consumption phase rather than only
  once before the loop, with the resource-tuple caveat. The pinned next-row
  extra-load control fails with
  `144`, `sgpr_count=5`, and `vgpr_count=13`, which rules out same-row
  membership in this minimized pinned shape but has a higher VGPR count due to
  the computed next-row address. The pinned separate-tile extra-load control
  fails with `288`, `sgpr_count=6`, and `vgpr_count=11`, showing same-tile
  membership is also not required in the pinned four-load shape, with the
  caveat that the resource tuple differs. The pinned post-row extra-load
  control fails at sync 88 under the 100-launch stress envelope with `144`,
  `sgpr_count=5`, and `vgpr_count=10`, matching the failing pre-row pinned
  tuple and showing that moving the extra load after row consumption is still
  fail-side. The no-extra double-barrier control fails at sync 94 with the
  same `144`, `sgpr_count=5`, and `vgpr_count=10` tuple, showing the tuple is
  still pass-side in the one-closing-barrier no-extra control but fail-side
  when a second closing barrier is added after row consumption. Moving that
  extra barrier before the first per-K store, between the two store phases, or
  before the row loads still fails with the same tuple, and inserting an
  `s_nop 0` between the two closing barriers still fails, so the double-barrier
  branch is not specifically post-consumption, post-producer, or
  adjacent-barrier sensitive. The matching single-store double-barrier control
  passes with the same `144`, `sgpr_count=5`, and `vgpr_count=10` tuple; the
  single-store four-barrier prestore control also passes with that tuple; and
  the two-store three-load controls pass with `144`, `sgpr_count=5`, and
  `vgpr_count=9`. Together they keep the second-store and four-row-load
  thresholds intact for the double-barrier branch. The cross-row load split
  uses `144`, `sgpr_count=5`, and `vgpr_count=13`, and rules out same-row
  membership as a required trigger component. The separate-tile load split uses
  `288`,
  `sgpr_count=4`, and `vgpr_count=13`, and rules out same-tile membership when
  compared with the pass-side write-once second tile control. The separate
  read-tile split also uses `288`, `sgpr_count=4`, and `vgpr_count=12`, and
  rules out requiring the six-wide row readback to consume the same LDS tile as
  the second store and extra load. The failing `tile6_synth` uses `288`,
  `sgpr_count=5`, and `vgpr_count=18`.
- A naive same-address double-store variant was not useful: the compiler reduced
  it to a single `ds_store`. A volatile padded second-address variant failed at
  sync 57, but compiled the shared accesses as `flat_store_b32` /
  `flat_load_b32` with cache modifiers rather than ordinary `ds_*`, so treat it
  as a side observation rather than the clean LDS-store boundary.
- Caution for those latest throwaway runs: `run.log`/`exit_code.txt` are the
  authoritative pass/fail evidence. The captured dmesg snapshots and live
  `dmesg --ctime` preserve the same MES `REMOVE_QUEUE` reset family, but the
  ctime stamps lag file timestamps on this machine, so do not use ctime alone
  to order those runs.
- Conclusion: launch-grid mismatch can mask the bug by reducing workload, but
  the correct launch geometry still faults.

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
- `/tmp/hipfire-lds-chaingun-runs/`: current throwaway `chaingun` body-split
  controls. Preserved subdirectories include `nolds-control`,
  `barrier-control`, `lds-one-control`, `lds-two-store-one-read`, and
  `lds-synth-control`.
- `/tmp/hipfire-lds-chaingun-next-runs/`: follow-up throwaway splitter controls
  from current `chaingun`, including `lds-padded-one`,
  `lds-two-store-once-one-read`, and the repeat failing
  `lds-two-store-perk-one-read` (`tile6_lds_two_store_one_read` failed at sync
  96).
- `/tmp/hipfire-lds-store-split-runs/`: same-address/second-address splitter
  controls. `tile6_barrier3_synth` passed at 100 launches, while
  `tile6_lds_forced_same_second_store` failed at sync 92 with a real
  `ds_store`, `ds_load`, `ds_store` same-address sequence.
- `/tmp/hipfire-lds-phase-split-runs/`: phase splitter controls. Extra
  same-address load plus wide reads passed, same-address load/store without
  wide row reads passed, and the full forced same-address load/store plus wide
  row reads failed at sync 91.
- `/tmp/hipfire-lds-readwidth-runs/`: row-read-width splitter controls. Forced
  same-address load/store plus read widths 1, 2, 4, and 5 passed; the six-wide
  `tile6_lds_forced_same_second_store` contrast failed in both repeats.
- `/tmp/hipfire-lds-storeonly-runs/`: second-producer-phase splitter controls.
  `tile6_lds_second_store_only_read6` passed with two same-address stores and
  six-wide row readback; `tile6_lds_forced_same_second_store` failed at sync 92
  in the same compile unit.
- `/tmp/hipfire-lds-dependency-runs/`: load-dependency splitter controls.
  `tile6_lds_second_store_only_read6` passed, while
  `tile6_lds_load_independent_store_read6` failed at sync 90. The failing
  variant has the same resource tuple as the pass-side store-only control
  (`group_segment_fixed_size=144`, `sgpr_count=5`, `vgpr_count=12`) and confirms
  that an LDS load before the second same-address store is sufficient; the
  second store does not need to consume the loaded value.
- `/tmp/hipfire-lds-address-split-runs/`: address-dependency splitter controls.
  The pass-side `tile6_lds_second_store_only_read6` control passed at 100
  launches, `tile6_lds_load_next_store_same_read6` failed with HIP 719, a
  20-launch recovery pass control succeeded, and
  `tile6_lds_load_same_store_next_read6` failed with HIP 719. Both failing
  address-split variants use one 144-byte LDS tile with `sgpr_count=5` and
  `vgpr_count=13`, and both emit ordinary `ds_load_b32`/`ds_store_b32`
  sequences. Exact same-address load/store aliasing is not required.
- `/tmp/hipfire-lds-order-split-runs/`: ordering splitter controls. The first
  same-thread store-then-load attempt was not useful because the compiler
  forwarded the stored value; a volatile version forced a `flat_load_b32` rather
  than ordinary LDS. The clean barrier-separated variant
  `tile6_lds_store_then_load_read6` emitted `ds_store_b32 -> barrier ->
  ds_store_b32 -> barrier -> ds_load_b32 -> six-wide row readback` and failed
  with HIP 719, while using `group_segment_fixed_size=144`, `sgpr_count=5`, and
  `vgpr_count=12`.
- `/tmp/hipfire-lds-storeload-readwidth-runs/`: store-then-load row-width
  controls. The pass-side `tile6_lds_second_store_only_read6` control passed,
  and store-then-load read widths 0, 1, 2, 4, and 5 all passed at 100 launches.
  The six-wide `tile6_lds_store_then_load_read6` endpoint failed at sync 91
  with the same MES reset signature. ISA confirms ordinary `ds_store_b32`,
  `ds_load_b32`, and row-read `ds_load*` instructions; width 0 emits the extra
  LDS load but no row readback.
- `/tmp/hipfire-lds-scrambled-read-runs/`: row-read instruction-form controls.
  A source-order scramble still compiled to the same packed row-read sequence
  as the existing six-wide case, so it was not promoted. The useful
  `tile6_lds_store_then_load_dynamiccols_read6` variant passed at one launch
  and failed at sync 85 under 100 launches. It emits the same store-then-load
  producer sequence followed by six independent dynamic-address `ds_load_b32`
  row loads. Metadata is `group_segment_fixed_size=144`, `sgpr_count=5`, and
  `vgpr_count=17`. A recovery `tile6_lds_two_store_once_one_read` control
  passed at 20 launches.
- `/tmp/hipfire-lds-load-use-split-runs/`: dynamic row-load/use-count controls.
  Promoted `tile6_lds_store_then_load_dynamiccols_load4_use4` passed at 100
  launches with four independent row `ds_load_b32` operations (`144`,
  `sgpr_count=5`, `vgpr_count=13`). Promoted
  `tile6_lds_store_then_load_dynamiccols_load5_use5` failed at sync 92 with
  five independent row `ds_load_b32` operations (`144`, `sgpr_count=5`,
  `vgpr_count=15`). Throwaway auxiliaries
  `tile6_lds_store_then_load_dynamiccols_load6_use5` and
  `tile6_lds_store_then_load_dynamiccols_load5_use6` failed at sync 82 and
  sync 91, respectively. Recovery controls passed between reset-producing
  runs.
- `/tmp/hipfire-lds-serial-load-runs/`: serialized dynamic row-load controls.
  `tile6_lds_store_then_load_dynamiccols_load5_serial_use5` passed at 100
  launches. ISA shows five dynamic row `ds_load_b32` operations, each followed
  by explicit `s_waitcnt lgkmcnt(0)`, with metadata
  `group_segment_fixed_size=144`, `sgpr_count=5`, and `vgpr_count=15`. The
  non-serialized
  `tile6_lds_store_then_load_dynamiccols_load5_use5` contrast failed at sync
  91 in the same compile unit, and the recovery control passed afterward. A
  later serialized contrast in `/tmp/hipfire-lds-split3-load-runs/` failed at
  sync 87 after several reset-producing split controls, so this is a
  clean-state contrast rather than a stable recovery workaround.
- `/tmp/hipfire-lds-split-load-runs/`: split dynamic row-load controls.
  Promoted `tile6_lds_store_then_load_dynamiccols_load5_split4_use5` passed a
  one-launch smoke, then failed at sync 92 over 100 launches. ISA shows four
  dynamic row `ds_load_b32` operations, `s_waitcnt lgkmcnt(0)`, then the fifth
  dynamic row `ds_load_b32`; metadata is `group_segment_fixed_size=144`,
  `sgpr_count=5`, and `vgpr_count=15`. A recovery
  `tile6_lds_two_store_once_one_read` control passed afterward.
- `/tmp/hipfire-lds-split3-load-runs/`: follow-up split dynamic row-load
  controls. Promoted `tile6_lds_store_then_load_dynamiccols_load5_split3_use5`
  passed one-launch smoke, emitted three dynamic row `ds_load_b32` operations,
  `s_waitcnt lgkmcnt(0)`, then two more dynamic row `ds_load_b32` operations,
  and failed at sync 91 over 100 launches. Promoted
  `tile6_lds_store_then_load_dynamiccols_load5_split2_3_use5` passed smoke and
  failed at sync 92. Promoted
  `tile6_lds_store_then_load_dynamiccols_load5_split2_2_1_use5` passed smoke
  and failed at sync 92. Promoted
  `tile6_lds_store_then_load_dynamiccols_load5_split1_keep5` passed smoke,
  emitted wait-after-each dynamic row load, kept all five loaded values live
  until final accumulation, and failed at sync 87. All four controls use
  `group_segment_fixed_size=144`, `sgpr_count=5`, and `vgpr_count=15`, and
  each failure produced the same MES remove-queue / MODE2 reset path. Recovery
  `tile6_lds_two_store_once_one_read` controls passed between reset-producing
  runs.
- `/tmp/hipfire-lds-live-threshold-runs/`: delayed-consume live-value
  threshold controls. A sanity run of promoted
  `tile6_lds_store_then_load_dynamiccols_load4_use4` passed at 100 launches
  before the throwaway edits. Throwaway
  `tile6_lds_store_then_load_dynamiccols_load4_split1_keep4` passed smoke,
  emitted wait-after-each dynamic row load, kept four loaded values live until
  final accumulation, and failed at sync 96 over 100 launches. It uses
  `group_segment_fixed_size=144`, `sgpr_count=5`, and `vgpr_count=13`.
  Recovery passed afterward. Throwaway
  `tile6_lds_store_then_load_dynamiccols_load3_split1_keep3` passed smoke and
  passed 100 launches with wait-after-each dynamic row load, three delayed live
  values, `group_segment_fixed_size=144`, `sgpr_count=5`, and
  `vgpr_count=11`. This brackets the delayed-live boundary between three and
  four dynamic row values for the one-tile store-then-load producer shape.
- `/tmp/hipfire-lds-immediate-consume-runs/`: pinned immediate-consume
  controls. A sanity run of promoted
  `tile6_lds_store_then_load_dynamiccols_load4_use4` passed at 100 launches
  before this throwaway worktree. An initial unpinned consume4 source was not
  used as evidence because codegen still delayed all FMACs until after the row
  loads. Throwaway
  `tile6_lds_store_then_load_dynamiccols_load4_split1_consume4_pinned` passed
  smoke, emitted dynamic row load / wait / FMAC groups, used
  `group_segment_fixed_size=144`, `sgpr_count=5`, `vgpr_count=10`, and failed
  at sync 89 over 100 launches. Recovery passed afterward. Throwaway
  `tile6_lds_store_then_load_dynamiccols_load3_split1_consume3_pinned` passed
  smoke and passed 100 launches with the same pinned consume form,
  `group_segment_fixed_size=144`, `sgpr_count=5`, and `vgpr_count=9`. This
  shows the 3/4 boundary survives even when delayed live-value pressure is
  removed and the FMACs are interleaved with waited row loads.
- `/tmp/hipfire-lds-noextra-runs/`: no-extra-load pinned controls. A clean
  promoted `tile6_lds_store_then_load_dynamiccols_load3_split1_consume3_pinned`
  sanity run passed 100 launches before the throwaway edits. Throwaway
  `tile6_lds_store_then_load_dynamiccols_load4_noextra_consume4_pinned` passed
  smoke, ISA inspection confirmed the pre-row `tmp = As[ty][tx]` LDS load was
  absent, metadata was `group_segment_fixed_size=144`, `sgpr_count=5`, and
  `vgpr_count=10`, and it passed 100 launches. Throwaway
  `tile6_lds_store_then_load_dynamiccols_load3_noextra_consume3_pinned` passed
  smoke and passed 100 launches with `group_segment_fixed_size=144`,
  `sgpr_count=5`, and `vgpr_count=9`. These are pass-side controls showing the
  extra ordinary LDS load remains required for the four-load pinned failure.
- `/tmp/hipfire-lds-single-store-pinned-runs/`: single-store pinned extra-load
  control. Throwaway
  `tile6_lds_single_store_then_load_dynamiccols_load4_consume4_pinned` passed
  one-launch smoke. ISA emits exactly one per-K `ds_store_b32`, one extra
  `ds_load_b32`, then four waited/interleaved dynamic row loads. Metadata is
  `group_segment_fixed_size=144`, `sgpr_count=5`, and `vgpr_count=10`, matching
  the failing second-store pinned tuple. The 100-launch run passed. This shows
  the second per-K LDS store remains required in the current minimized
  four-load pinned shape.
- `/tmp/hipfire-lds-order-pinned-runs/`: pinned load-then-store order split.
  Throwaway `tile6_lds_load_then_store_dynamiccols_load4_consume4_pinned`
  passed one-launch smoke. ISA emits first store -> barrier -> extra
  `ds_load_b32` -> second store -> barrier -> four waited/interleaved dynamic
  row loads. Metadata is `group_segment_fixed_size=144`, `sgpr_count=5`, and
  `vgpr_count=10`, matching the failing store-then-load pinned tuple. The
  100-launch run failed at sync 91 with HIP 719. A follow-up one-launch
  single-store pinned recovery smoke passed. This shows the extra ordinary LDS
  load does not need to occur after the second store in the pinned minimized
  shape; the second store only needs to be present before the four dynamic row
  loads.
- `/tmp/hipfire-lds-preloop-pinned-runs/`: preloop-only extra-load control.
  Throwaway `tile6_lds_preloop_load_then_store_dynamiccols_load4_consume4_pinned`
  passed one-launch smoke. ISA emits one preloop `ds_store_b32` / `ds_load_b32`
  setup pair, then the K loop emits first store -> barrier -> second store ->
  barrier -> four waited/interleaved dynamic row loads, with no per-K extra LDS
  load. Metadata is `group_segment_fixed_size=144`, `sgpr_count=6`, and
  `vgpr_count=11`. The 100-launch run passed. This indicates the extra
  ordinary LDS load must happen per K / near the row-consumption phase, not
  merely once before the loop. Resource tuple differs from the failing pinned
  control because of the preloop setup.
- `/tmp/hipfire-lds-extra-row-runs/`: pinned next-row extra-load control.
  Throwaway `tile6_lds_store_then_load_dynamiccols_load4_nextrow_consume4_pinned`
  passed one-launch smoke. ISA emits the second store, then an extra
  `ds_load_b32` from the next-row address, then four waited/interleaved dynamic
  row loads. Metadata is `group_segment_fixed_size=144`, `sgpr_count=5`, and
  `vgpr_count=13`. The 100-launch run failed at sync 86 with HIP 719. A
  follow-up one-launch `load4_noextra_consume4_pinned` recovery smoke passed.
  This shows the extra ordinary LDS load does not need to target the same row
  as the four dynamic row loads in the pinned minimized shape.
- `/tmp/hipfire-lds-separate-tile-pinned-runs/`: pinned separate-tile
  extra-load control. Throwaway
  `tile6_lds_store_then_load_dynamiccols_load4_separate_tile_consume4_pinned`
  passed one-launch smoke. ISA initializes `Xs` once before the loop, then
  emits the per-K second store on `As`, an extra `ds_load_b32` from `Xs`, and
  four waited/interleaved dynamic row loads from `As`. Metadata is
  `group_segment_fixed_size=288`, `sgpr_count=6`, and `vgpr_count=11`. The
  100-launch run failed at sync 88 with HIP 719. A follow-up one-launch
  `load4_noextra_consume4_pinned` recovery smoke passed. This shows the extra
  ordinary LDS load does not need to come from the same LDS tile as the four
  dynamic row loads in the pinned four-load shape.
- `/tmp/hipfire-lds-postrow-pinned-promote-smoke/`: promoted post-row
  extra-load control. `tile6_lds_store_then_rowload_then_extra_load4_consume4_pinned`
  passed one-launch smoke. ISA emits first store -> barrier -> second store ->
  barrier -> four waited/interleaved dynamic row loads and FMACs -> extra
  `ds_load_b32` -> barrier. Metadata is `group_segment_fixed_size=144`,
  `sgpr_count=5`, `vgpr_count=10`, and no spills, matching the failing
  pre-row pinned tuple.
- `/tmp/hipfire-lds-postrow-pinned-runs/`: post-row extra-load stress and
  recovery. `tile6_lds_store_then_rowload_then_extra_load4_consume4_pinned`
  failed at sync 88 with HIP 719 under `N_LAUNCH=100`, producing the same MES
  `REMOVE_QUEUE` / MODE2 reset family. Live dmesg recorded reset 523 at
  2026-06-19 23:06:25-23:06:29 UTC. A follow-up one-launch
  `tile6_lds_store_then_load_dynamiccols_load4_noextra_consume4_pinned`
  recovery smoke passed.
- `/tmp/hipfire-lds-postbarrier-runs/`: post-row double-barrier controls.
  Throwaway `tile6_lds_store_then_rowload_barrier_noextra_load4_consume4_pinned`
  passed one-launch smoke, then failed at sync 94 with HIP 719 under
  `N_LAUNCH=100`, producing the same MES `REMOVE_QUEUE` reset family. ISA
  emits first store -> barrier -> second store -> barrier -> four waited row
  loads/FMAs -> barrier -> barrier; metadata is `group_segment_fixed_size=144`,
  `sgpr_count=5`, `vgpr_count=10`, and no spills. A follow-up one-launch
  `tile6_lds_store_then_load_dynamiccols_load4_noextra_consume4_pinned`
  recovery smoke passed. A post-barrier extra-load variant was built and
  one-launch smoked in the throwaway worktree, but its stress result is not
  useful until the no-extra double-barrier failure is understood.
- `/tmp/hipfire-lds-barrier-width-store-runs/`: double-barrier pass-side
  store-count and width controls. Throwaway
  `tile6_lds_single_store_rowload_barrier_noextra_load4_consume4_pinned`
  passed 100 launches. ISA emits one `ds_store_b32`, four waited row loads,
  then two closing barriers; metadata is `group_segment_fixed_size=144`,
  `sgpr_count=5`, `vgpr_count=10`, and no spills. Throwaway
  `tile6_lds_store_then_rowload_barrier_noextra_load3_consume3_pinned`
  also passed 100 launches. ISA emits two `ds_store_b32` phases, three waited
  row loads, then two closing barriers; metadata is `group_segment_fixed_size=144`,
  `sgpr_count=5`, `vgpr_count=9`, and no spills.
- `/tmp/hipfire-lds-barrier-placement-runs/`: double-barrier placement
  controls. Throwaway
  `tile6_lds_store_then_prerow_barrier_noextra_load4_consume4_pinned`
  passed one-launch smoke and failed at sync 94 under 100 launches. ISA emits
  first store -> barrier -> second store -> barrier -> barrier -> four waited
  row loads -> barrier; metadata is `group_segment_fixed_size=144`,
  `sgpr_count=5`, `vgpr_count=10`, and no spills. A follow-up one-launch
  `tile6_lds_store_then_load_dynamiccols_load4_noextra_consume4_pinned`
  recovery smoke passed. Throwaway
  `tile6_lds_store_then_rowload_barrier_gap_noextra_load4_consume4_pinned`
  passed one-launch smoke and failed at sync 94 under 100 launches. ISA emits
  first store -> barrier -> second store -> barrier -> four waited row loads
  -> barrier -> `s_nop 0` -> barrier; metadata is again `144`,
  `sgpr_count=5`, `vgpr_count=10`, and no spills. Live dmesg recorded the same
  MES `REMOVE_QUEUE` / MODE2 reset family, including reset 525 for the pre-row
  control and reset 526 for the gapped post-row control.
- `/tmp/hipfire-lds-barrier-early-runs/`: early double-barrier placement
  controls. Throwaway `tile6_lds_prestore_barrier_noextra_load4_consume4_pinned`
  passed one-launch smoke and failed at sync 93 under 100 launches. ISA emits
  barrier -> first store -> barrier -> second store -> barrier -> four waited
  row loads -> barrier; metadata is `group_segment_fixed_size=144`,
  `sgpr_count=5`, `vgpr_count=10`, and no spills. A follow-up one-launch
  `tile6_lds_store_then_load_dynamiccols_load4_noextra_consume4_pinned`
  recovery smoke passed. Throwaway
  `tile6_lds_betweenstore_barrier_noextra_load4_consume4_pinned` passed
  one-launch smoke and failed at sync 94 under 100 launches. ISA emits first
  store -> barrier -> barrier -> second store -> barrier -> four waited row
  loads -> barrier; metadata is again `144`, `sgpr_count=5`, `vgpr_count=10`,
  and no spills. Another follow-up one-launch no-extra recovery smoke passed.
  Live dmesg recorded the same MES `REMOVE_QUEUE` / MODE2 reset family,
  including reset 527 for the prestore control and reset 528 for the
  between-store control.
- `/tmp/hipfire-lds-barrier-reqs-runs/`: pass-side requirement controls for
  the early-barrier branch. Throwaway
  `tile6_lds_single_store_prestore_barrier_noextra_load4_consume4_pinned`
  passed one-launch smoke and 100-launch stress. ISA emits barrier -> one
  `ds_store_b32` -> barrier -> barrier -> four waited row loads -> barrier;
  metadata is `group_segment_fixed_size=144`, `sgpr_count=5`,
  `vgpr_count=10`, and no spills. Throwaway
  `tile6_lds_prestore_barrier_noextra_load3_consume3_pinned` also passed
  one-launch smoke and 100-launch stress. ISA emits barrier -> first store ->
  barrier -> second store -> barrier -> three waited row loads -> barrier;
  metadata is `144`, `sgpr_count=5`, `vgpr_count=9`, and no spills. Live dmesg
  showed no new reset during either passing stress run.
- `/tmp/hipfire-lds-barrier-once-runs/`: one-time barrier controls. Throwaway
  `tile6_lds_preloop_barrier_noextra_load4_consume4_pinned` passed a
  one-launch smoke. ISA/resource metadata showed `group_segment_fixed_size=144`,
  `sgpr_count=6`, `vgpr_count=10`, and no spills. Throwaway
  `tile6_lds_firstiter_barrier_noextra_load4_consume4_pinned` also passed a
  one-launch smoke with `group_segment_fixed_size=144`, `sgpr_count=5`,
  `vgpr_count=11`, and no spills. These controls are now promoted for
  cross-system 100-launch stress on another 780M. After promotion in the main
  checkout, both variants passed one-launch compile/run smokes at
  `/tmp/hipfire-lds-promote-preloop-barrier-n1/` and
  `/tmp/hipfire-lds-promote-firstiter-barrier-n1/`.
- `/tmp/hipfire-lds-barrier-once-stress/`: promoted one-time barrier stress
  controls. `tile6_lds_preloop_barrier_noextra_load4_consume4_pinned` passed
  100 launches at
  `/tmp/hipfire-lds-barrier-once-stress/preloop-barrier-load4-n100/`; metadata
  is `group_segment_fixed_size=144`, `sgpr_count=6`, `vgpr_count=10`,
  `wavefront_size=32`, and no private segment.
  `tile6_lds_firstiter_barrier_noextra_load4_consume4_pinned` passed 100
  launches at
  `/tmp/hipfire-lds-barrier-once-stress/firstiter-barrier-load4-n100/`;
  metadata is `group_segment_fixed_size=144`, `sgpr_count=5`, `vgpr_count=11`,
  `wavefront_size=32`, and no private segment. Live dmesg still ended at the
  earlier reset 528 after both passing runs.
- `/tmp/hipfire-lds-prestore-barrier-klimit-bisect/` and
  `/tmp/hipfire-lds-prestore-barrier-klimit-confirm/`: K-depth runs for the
  repeated prestore-barrier no-extra shape. The first bisect reported a nominal
  `K_LIMIT=2943` pass / `2944` fail boundary, but those values have the same
  `TILE=6` loop-trip count, and manual repeats showed the top edge is
  stress-history sensitive. Robust pass-side repeats: `K_LIMIT=2048`, `2559`,
  and `2815`. Fail-side confirmation after reset pressure: `K_LIMIT=2880`,
  `2910`, `2928`, `2934`, `2940`, `2941`, `2946`, and `2947`, all failing
  under 100 launches with HIP 719 around sync 96-99. Live dmesg advanced
  through the same MES `REMOVE_QUEUE` / MODE2 reset family, reaching reset 545.
- `/tmp/hipfire-lds-periodic-barrier-runs/`: periodic prestore-control
  throwaway. Periodic extra-barrier controls with periods `512`, `256`, `128`,
  `64`, `32`, `16`, and `8` all failed under 100 launches at full K, generally
  around sync 95-96, with metadata `group_segment_fixed_size=144`,
  `sgpr_count=5`, `vgpr_count=10`, `wavefront_size=32`, and no private segment.
  The period-512 case should take the extra barrier only once for K=3072, but
  its ISA carries a loop counter plus `s_and_b32` / branch predicate. A matching
  period-512 no-barrier control that executes `s_nop 0` on the same predicate
  also failed at sync 96 with the same resource tuple. The direct
  `tile6_lds_firstiter_barrier_noextra_load4_consume4_pinned` control was
  rerun afterward and passed 100 launches, with no new dmesg reset after the
  pass. Live dmesg advanced through the same MES `REMOVE_QUEUE` / MODE2 reset
  family during the periodic failures, reaching reset 553.
- `/tmp/hipfire-lds-crossrow-runs/`: cross-row extra-load controls. The first
  two-launch run failed adjacent to prior reset pressure, but a recovery
  `tile6_lds_second_store_only_read6` control passed, then clean
  `tile6_lds_store_then_load_nextrow_read6` runs passed at one and two launches.
  The 100-launch cross-row run failed at sync 88. ISA emits `ds_store_b32 ->
  barrier -> ds_store_b32 -> barrier -> ds_load_b32` from `As[(ty+1)%6][tx]`,
  followed by the six-wide row readback from `As[ty][0..5]`; metadata is
  `group_segment_fixed_size=144`, `sgpr_count=5`, and `vgpr_count=13`.
- `/tmp/hipfire-lds-separate-tile-runs/`: separate-tile extra-load controls.
  `tile6_lds_store_then_load_separate_tile_read6` passed at one launch and
  failed at sync 91 under 100 launches. The variant initializes `Xs` once before
  the K loop, then loops with `As` first store, `As` second store, extra
  `ds_load_b32` from `Xs`, and six-wide row readback from `As`. Metadata is
  `group_segment_fixed_size=288`, `sgpr_count=4`, and `vgpr_count=13`. A
  recovery `tile6_lds_two_store_once_one_read` control passed at 20 launches.
- `/tmp/hipfire-lds-separate-rowread-runs/`: separate read-tile controls.
  `tile6_lds_store_then_load_separate_readtile` passed at one launch and failed
  at sync 91 under 100 launches. The variant initializes `Xs` once before the K
  loop, then loops with `As` first store, `As` second store, extra `ds_load_b32`
  from `As`, and six-wide row readback from `Xs`. Metadata is
  `group_segment_fixed_size=288`, `sgpr_count=4`, and `vgpr_count=12`. A
  recovery `tile6_lds_two_store_once_one_read` control passed at 20 launches.
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
- At reads=3/448/480x86, the one-child edge moved upward: one child passed at
  `99`, `100`, `101`, `102`, `103`, and `104` requested launches. One child
  failed at `105` requested launches, failing at sync/global launch 103 and
  capturing late generic `devcd40`; one child with `120` requested launches
  failed at sync/global launch 104 and captured late generic `devcd39`. Split
  controls at the same total `120` requested launches passed as `104,16` and
  `60,60` in both plain-parent and HIP-initialized-parent modes.
- At reads=3/448/448x86, one child initially passed at `105` and `120`
  requested launches, then failed at `160` with late generic `devcd41`. A
  bracket pass after that found `130`, `122`, and `121` all failing with late
  generic `devcd42`, `devcd43`, and `devcd44`; the `121` run failed at
  sync/global launch 114. Split controls at total `160` then separated a lower
  child-local count from the shifted edge: `120,40` failed inside the first
  120-launch child in both plain-parent and HIP-initialized-parent modes
  (`devcd45` / `devcd46`), while `80,80` passed in both modes.
- At reads=3/448/416x86, one child passed at `120`, `122`, and `124`
  requested launches. One child failed at `130` with late generic `devcd47`,
  then the tightened sweep failed at `126` and `125` with `devcd48` and
  `devcd49`; the `125` and `126` runs failed at sync/global launch 120. Split
  controls at total `160` matched the 448x86 shape: `124,36` failed inside the
  first 124-launch child in both plain-parent and HIP-initialized-parent modes
  (`devcd50` / `devcd51`), while `80,80` passed in both modes.
- At reads=3/448/384x86, one child passed at `125`, `128`, `130`, `132`, and
  `134` requested launches. One child with `135` requested launches failed at
  sync/global launch 132 with late generic `devcd52`, then a repeat failed at
  sync/global launch 133 and preserved `devcd53` in the same artifact
  directory. Split controls at total `180` matched the lower-grid
  state-sensitive shape: `134,46` failed inside the first 134-launch child in
  both plain-parent and HIP-initialized-parent modes (`devcd54` / `devcd55`),
  while `90,90` passed in both modes.
- At reads=3/448/352x86, one child with `150` requested launches passed, while
  `151` and `152` failed. A nearby check also observed `148`/`149` failures
  after prior state exposure. Split control `105,45` passed in both parent
  modes.
- At reads=3/448/320x86, one child passed at `156`, `160`, and `162`, while
  `163` through `166` failed. Split controls `80,85` and `98,67` passed at the
  same totals in both plain-parent and HIP-initialized-parent modes.
- All `352x86` and `320x86` follow-up runs are stored under:
  `/tmp/hipfire-lds-direct-ab-lower-grid-352-320-artifacts/`.
- At reads=3/448/352x86 fresh rerun, plain mode gave `148`/`149` fail, `150`
  pass, `151`/`152` fail; `MODE=hipinit_reset_before` failed `148`–`152` all
  together. At reads=3/448/320x86 fresh rerun, plain mode gave
  `160`/`161` pass, `162` fail, `163`/`164` pass, while
  `MODE=hipinit_reset_before` shifted to `160`/`161` pass and `162`+ fail.
  All failures still captured the same signature (`[gfxhub] Page fault
  observed`, faulty page `0x000074669d000000`, `Protection fault status
  register: 0x841051`). Fresh runs are under:
  `/tmp/hipfire-lds-direct-ab-lower-grid-fresh/`.
- In the next fresh pass/fail extension (`/next`), plain mode at `384x86` gave
  `134` pass / `135` fail while `hipinit_reset_before` gave `134` fail /
  `135` pass; at `416x86` both modes failed for `124` and `125`.
- A fresh 384x86 follow-up (`/tmp/hipfire-lds-active-lane-fresh/`) kept this
  active-thread threshold behavior: `6x6` failed at one-child `135`, while `5x5`
  and `4x4` variants were green at `150` and `200`, respectively. Additional
  active-lane sweeps (`/tmp/hipfire-lds-active-lane-fresh/act4_plain_scan_*` and
  `/tmp/hipfire-lds-active-lane-fresh/act5_plain_scan_*`) show fresh one-child
  non-monotonic behavior (`4x4` fails at `279`, `5x5` fails at `280` and
  `300` but passes at `320`).
- A fresh 512x86 asymmetric active-lane pass
  (`/tmp/hipfire-lds-direct-ab-asym-artifacts/`, throwaway worktree
  `/tmp/hipfire-lds-direct-ab-asym-1781922739`) pins the direct-AB one-child
  `reads=3`, `iters=448`, `chunks=150` edge to crossing one wavefront:
  `6x5` and `5x6` (30 active lanes, 248 B group segment) passed, `8x4` and
  `4x8` (32 lanes, 256 B group segment) passed, while `11x3`/`3x11`
  (33 lanes, 276 B), `7x5`/`5x7` (35 lanes, 284 B), and `6x6`
  (36 lanes, 288 B) failed. All failing rows captured the same direct-AB
  signature: HIP-719 at sync/global launch 98-99, `dmesg_remove_queue=3`,
  fault address `0x000074669d000000`, prot status `0x841051`, GCVM flags
  `MORE_FAULTS,PERMISSION_FAULTS,RW`, and GDS/GDS-VM registers
  `0x3f000007` / `0x0fc00113`.
- Regenerating the asymmetric summary with the expanded direct-AB summarizer
  shows the pass/fail boundary does not change barrier/DS/wait instruction
  counts: every `chunks=150` boundary row has `isa_s_barrier=8`,
  `isa_ds=28`, `isa_s_waitcnt=14`, `isa_s_cbranch=1`, `sgpr=2`, and `vgpr=34`.
  The compact visible codegen difference at this boundary is
  `ds_store_2addr_b32 offset1`: passing `30`/`32`-lane rows report `offset1=32`,
  while failing `33`/`35`/`36`-lane rows report `offset1=36`.
- A follow-up padded-layout control
  (`/tmp/hipfire-lds-direct-ab-layout-artifacts/`, throwaway worktree
  `/tmp/hipfire-lds-direct-ab-layout-1781923273`) decoupled active lanes from
  LDS footprint/offset. `8x4` active with `LAYOUT=9x4` and `4x8` active with
  `LAYOUT=4x9` both passed at `reads=3`, `iters=448`, `chunks=150`,
  `grid=512x86`, with `group_segment=288` and `isa_ds_store_offset1=36`.
  The same run series failed the `6x6` anchor at sync/global launch 98 with
  the same `0x841051` GCVM and GDS/GDS-VM signature. Therefore 288 B LDS and
  `offset1=36` are not sufficient to trigger the failure without more than one
  wavefront of active lanes.
- A two-wave-block control
  (`/tmp/hipfire-lds-direct-ab-block-artifacts/`, throwaway worktree
  `/tmp/hipfire-lds-direct-ab-block-1781923566`) then decoupled block size from
  active LDS traffic. `8x4` active lanes inside a `9x4` block with
  `LAYOUT=9x4` passed at `reads=3`, `iters=448`, `chunks=150`,
  `grid=512x86`, while the all-active `9x4` and `6x6` anchors failed at
  sync/global launches 98-99 with the same `0x841051` GCVM and GDS/GDS-VM
  signature. The inactive-lane control compiles differently
  (`sgpr=5`, `vgpr=15`, `isa_s_barrier=4`, `isa_ds=14`, `isa_s_cbranch=5`),
  so it is not an ISA-matched replacement for the all-active shape comparisons.
- A follow-up traffic-mask replay
  (`/tmp/hipfire-lds-direct-ab-traffic-idle-artifacts/`, throwaway worktree
  `/tmp/hipfire-lds-direct-ab-traffic-1781923745`) found that this inactive-wave
  style control is itself state-sensitive. `9x4` active/block/layout with only
  `8x4` LDS traffic and no extra active-lane arithmetic failed at sync/global
  launch 100 with the same GCVM/GDS signature. Its normalized ISA hash and
  resource tuple match the earlier passing `8x4`-active-inside-`9x4`-block
  row (`isa_norm=9e53a96d22d0a718`, `sgpr=5`, `vgpr=15`, `barrier=4`,
  `ds=14`, `wait=8`, `branch=5`, `offset1=36`), so do not use the earlier
  pass as a stable disproof that a two-wave block/barrier can enter the failure
  family after reset pressure.
- A read-pressure sweep on the stable all-active first-two-wave shape
  (`/tmp/hipfire-lds-direct-ab-reads-artifacts/`, throwaway worktree
  `/tmp/hipfire-lds-direct-ab-reads-1781924058`) used `9x4` active/block/layout,
  `iters=448`, `grid=512x86`, and one child. At `chunks=150`, `READS=1`
  passed, while `READS=2` failed at sync/global launch 131 and `READS=3`
  failed at launch 98. A `READS=2` launch-count bracket with the same normalized
  ISA (`277a9cab2146459e`) passed at `120` and `130` launches, then failed at
  `140` and `150` with the same `0x841051` GCVM and GDS/GDS-VM signature.
- A follow-up `READS=1` high-count pass
  (`/tmp/hipfire-lds-direct-ab-reads1-artifacts/`, throwaway worktree
  `/tmp/hipfire-lds-direct-ab-reads1-1781924252`) kept the same `9x4`,
  `iters=448`, `grid=512x86` shape and passed one-child `220`, `260`, `300`,
  and `500` launches with no dmesg or devcoredump deltas. This makes `READS=1`
  a strong pass-side for the first-two-wave shape on this stack, not merely a
  low-launch-count pass.
- A `READS=2` process-split control
  (`/tmp/hipfire-lds-direct-ab-reads2split-artifacts/`, throwaway worktree
  `/tmp/hipfire-lds-direct-ab-reads2split-1781924402`) used the same `9x4`,
  `iters=448`, `grid=512x86` code object as the failing one-child `140`
  run (`isa_norm=277a9cab2146459e`) but split total work across child
  processes. `130,10` and `120,20` both passed with no dmesg/devcoredump
  deltas, while the one-child `140` failed at sync/global launch 133.
- A `READS=2` high-count one-wave control
  (`/tmp/hipfire-lds-direct-ab-reads2-8x4-artifacts/`, throwaway worktree
  `/tmp/hipfire-lds-direct-ab-reads2-8x4-1781924533`) kept the same
  `iters=448`, `grid=512x86`, one-child shape but used all-active `8x4`
  (`32` active lanes). It passed `500` and `1000` launches with no
  dmesg/devcoredump deltas. The code object has the same static barrier/DS/wait
  counts as `9x4 READS=2` (`8` / `20` / `12`) but the one-wave LDS layout
  (`group_segment=256`, `offset1=32`, `isa_norm=5188faa843fa5475`).
- A `READS=2` first-over-one-wave sweep
  (`/tmp/hipfire-lds-direct-ab-r2-threshold-artifacts/`, throwaway worktree
  `/tmp/hipfire-lds-direct-ab-r2-threshold-1781924934`) tested the same
  `iters=448`, `grid=512x86`, one-child edge with 33 active lanes. Both
  `11x3` and transposed `3x11` passed `130` launches and failed at `140`
  launches (`11x3`: sync/global 132; `3x11`: sync/global 133). The `11x3`
  `130,10` split-child run passed. Both failing rows kept the canonical
  HIP-719 signature (`REMOVE_QUEUE=3`, GCVM
  `MORE_FAULTS,PERMISSION_FAULTS,RW`, VMID 8, GDS-VM `0x0fc00113`) and the
  same static counts as the `9x4 READS=2` failure (`s_barrier=8`, DS=20,
  `s_waitcnt=12`, `sgpr=2`, `vgpr=24`, `offset1=36`, `group_segment=276`).
- A follow-up exact-edge sweep
  (`/tmp/hipfire-lds-direct-ab-r1r2-edge-artifacts/`, throwaway worktree
  `/tmp/hipfire-lds-direct-ab-r1r2-edge-1781925201`) kept the 33-lane
  `11x3`/`3x11`, `iters=448`, `grid=512x86` shapes and first ran READS=1
  high-count controls. Both orientations passed `500` launches at READS=1
  (`s_barrier=16`, DS=24, `s_waitcnt=16`, `vgpr=22`, `offset1=36`). The
  READS=2 one-child threshold tightened to `11x3` passing `131`/`132` and
  failing at `133` (sync/global 130), while `3x11` passed `131`/`132`/`133`
  and failed at `134` (sync/global 132). Both failures kept the canonical
  coredump signature. A split-at-edge `11x3` `132,1` run passed, but a later
  `3x11` `133,1` run failed inside child 0 at global 131 after reset pressure,
  so that transposed split is recorded as state-sensitive rather than clean
  process-boundary evidence.
- A split-vs-same-process check on the 33-lane `3x11` READS=2 edge used
  `/tmp/hipfire-lds-direct-ab-state-edge-artifacts/` (throwaway worktree
  `/tmp/hipfire-lds-direct-ab-state-edge-1781925481`) and
  `/tmp/hipfire-lds-direct-ab-same-first-artifacts/` (throwaway worktree
  `/tmp/hipfire-lds-direct-ab-same-first-1781925585`). Running the
  cross-process split `132,1` passed cleanly. Running the same total as one
  process with phase1 `132`, phase2 `1` failed on phase2 launch 0 / global 132
  even when it was the first risky run in a fresh artifact root. The failure
  kept the canonical coredump signature (`REMOVE_QUEUE=3`, GCVM
  `MORE_FAULTS,PERMISSION_FAULTS,RW`, VMID 8, GDS-VM `0x0fc00113`) with the
  same static codegen (`s_barrier=8`, DS=20, `s_waitcnt=12`, `offset1=36`).
- A device-reset boundary check
  (`/tmp/hipfire-lds-direct-ab-device-reset-artifacts/`, throwaway worktree
  `/tmp/hipfire-lds-direct-ab-device-reset-1781925734`) ran the same `3x11`,
  READS=2, `132+1`, `grid=512x86` phase split with `hipDeviceReset()` between
  phases as the first risky row. It passed: phase1 `132` OK, boundary
  `hipDeviceReset` OK, phase2 `1` OK. In that post-reset state,
  `stream_recreate` and ordinary `same` mode for `132+1` also passed, while
  one-child `134` still failed at sync/global 132 with the canonical coredump
  signature. This suggests an in-process device reset can clear enough
  near-edge state for the split case, but it does not eliminate the underlying
  failing one-child shape.
- First-risky cleanup-mode checks for the same `3x11`, READS=2, `132+1`
  shape show that stream/primary-context variants are not equivalent to
  `hipDeviceReset`. `stream_recreate` from a fresh throwaway worktree
  (`/tmp/hipfire-lds-direct-ab-stream-first-artifacts/`, worktree
  `/tmp/hipfire-lds-direct-ab-stream-first-1781925950`) failed during phase1
  at sync/global 69. `primary_ctx_reset` from a separate fresh throwaway
  worktree (`/tmp/hipfire-lds-direct-ab-primary-first-artifacts/`, worktree
  `/tmp/hipfire-lds-direct-ab-primary-first-1781925950`) failed during phase1
  at sync/global 33. Both kept the canonical coredump signature. Because they
  did not reach the phase boundary, these rows are threshold-perturbation
  evidence, not boundary-cleanup evidence.

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
- One extra barrier anywhere in the kernel as the root cause: a pre-loop extra
  barrier passes, and a first-iteration-only extra barrier inside the K loop
  also passes 100 launches. The fail-side barrier controls require the extra
  sync epoch to repeat across K iterations.
- Taken extra barrier count alone as the root cause: a period-512 periodic
  barrier control fails even though it should take only one extra barrier at
  K=3072, and a matching period-512 periodic `s_nop 0` control with no extra
  barrier also fails. The direct first-iteration-only barrier remains pass-side.
- Small K-depth alone for the repeated prestore-barrier branch: the same
  two-store/four-load/four-barrier shape passes at `K_LIMIT=2815` under 100
  launches. The fail-side reappears near the top of K, with confirmed failures
  from `K_LIMIT=2880` upward after reset pressure.

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
- In the installed `/usr/src/amdgpu-6.19.0-2307534.24.04` source,
  `amd/amdgpu/mes_v11_0.c` emits `MES failed to respond to
  msg=REMOVE_QUEUE` after `amdgpu_fence_wait_polling()` fails or the MES status
  word is not written. `amd/amdkfd/kfd_device_queue_manager.c` then reports
  `failed to remove hardware queue from MES`, marks the HWS path hung via
  `kfd_hws_hang()`, and its eviction/remove-all callers emit the observed
  `Failed to evict queue` / `remove_all_kfd_queues_mes` messages. On Phoenix's
  soc21 path, `amd/amdgpu/soc21.c` emits `MODE2 reset`; `amd/amdgpu/amdgpu.h`
  describes MODE2 as a lower-scope ASIC reset that avoids CPU-shared IPs and
  memory-controller reset on APUs.

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

Decoded against `gc_11_0_3_sh_mask.h`, the GCVM protection status is clear
(`0x0`) for this captured hipfire-run sample. The two GDS registers both have
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
The decoded GCVM status for `0x841051` is:
`MORE_FAULTS,PERMISSION_FAULTS,RW`, `CID=8`, `VMID=8`, with no walker error,
mapping error, atomic, VF/VFID, PRT, or FED bit set. This gives future
second-780M comparisons a more precise gfxhub signature than the raw
protection-status hex alone.

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
- Fresh reruns at 352x86 and 320x86 one-child confirm this state sensitivity in
  the same kernel family. Plain mode is non-monotonic in 320x86 (160/161 pass,
  162 fail, 163/164 pass), while `hipinit_reset_before` gives `160`/`161` pass
  and `>=162` fail. At 352x86, plain-mode fresh reruns show `148`/`149` fail,
  `150` pass, and `151`/`152` fail; `hipinit_reset_before` failed all `148`–`152`.
  Signature remains unchanged (`[gfxhub] Page fault observed`, faulting page
  `0x000074669d000000`, status `0x841051`), so this is still the same fault
  family with moving timing thresholds.
- Extending to fresh 384/416 one-child points reinforced this: plain mode gave
  `384x86` `134` pass / `135` fail, while `hipinit_reset_before` inverted to
  `134` fail / `135` pass, and both modes failed fresh `416x86` at `124` and
  `125`.
- Active-lane sweeps at the same `384x86` geometry show the trigger shifts with
  active thread count but stays state-sensitive: `5x5` fails around `280`–`300`
  while `4x4` pushes to `279` (`320`/`260` pass), and `6x6` fails at `135`.
  That pattern supports “work-per-thread + queue lifetime” coupling rather than
  a fixed launch-count constant.
- The fresh 512x86 asymmetric pass sharpens that to a wavefront boundary for
  the promoted direct-AB phase kernel at `reads=3`/`iters=448`/`chunks=150`:
  <=32 active lanes pass and 33+ active lanes fail, independent of orientation.
  Because `6x5`/`5x6` and `8x4`/`4x8` pass but `11x3`/`3x11`,
  `7x5`/`5x7`, and `6x6` fail with the same GCVM/GDS signature, this is not an
  exact-square `6x6` artifact. It points at the first two-wave direct-AB LDS
  use as the load-bearing condition, with the launch-count/grid/work threshold
  still providing the timing edge.
- The appended ISA counters argue against a simple "more barriers" or "more DS
  instructions" explanation across that boundary: pass and fail rows have the
  same `8` barriers, `28` DS ops, `14` waits, and one scalar loop branch.
  What changes with the active-lane boundary is the LDS layout/codegen for the
  paired A/B store (`offset1=32` on the <=32-lane pass side, `offset1=36` on
  the 33+ fail side). Treat this as a correlation to preserve in follow-up
  controls, not yet a standalone cause.
- The padded-layout controls make that correlation non-causal by itself:
  exact-wave `8x4`/`4x8` shapes survive with the same `288 B` group segment and
  `offset1=36` as failing `6x6`. The load-bearing condition is now better
  stated as two-wave active direct-AB LDS traffic plus enough repeated
  launch/grid work, not LDS footprint or A/B base spacing alone.
- The inactive-second-wave controls are mixed after reset pressure. The first
  `8x4`-active-inside-`9x4` run passed while all-active `9x4` failed, but a
  later traffic-mask replay with the same normalized ISA/resource tuple failed
  with the canonical signature. This weakens any claim that two-wave
  workgroup/barrier participation is harmless by itself. The more stable
  all-active evidence remains the 32/33 active-lane boundary: <=32 active lanes
  pass at the tested edge, while 33+ all-active lanes fail.
- The `9x4` all-active read-pressure sweep shows the first-two-wave condition
  also needs enough LDS read work at the tested edge: `READS=1` survives at
  least `500` launches, while `READS=2` crosses between `130` and `140`
  launches and `READS=3` fails earlier. The failure signature is unchanged, so
  this is the same fault family with a shifted launch-count threshold, not a new
  error. `READS=1` has more static barriers/waits than `READS=2`
  (`16`/`16` versus `8`/`12`) but remains pass-side, so static barrier/wait
  count is not the trigger.
- The `READS=2` `140`-launch edge repeats the process-boundary pattern:
  one child fails, but the same total split as `130,10` or `120,20` passes.
  This keeps pointing at per-process/queue lifetime state coupled to LDS read
  pressure rather than total parent-supervised launch count alone.
- The `8x4 READS=2` high-count pass keeps the active-lane boundary stable under
  read pressure: a one-wave `32`-lane shape survives `1000` one-child launches
  with the same barrier/DS/wait counts as failing `9x4 READS=2`, while the
  first-two-wave `36`-lane shape fails near `140`. That makes the current
  trigger model "two-wave active LDS traffic plus enough LDS reads and
  per-process launch lifetime" rather than LDS read count alone.
- The READS=2 first-over-one-wave sweep sharpens that boundary: `11x3` and
  `3x11` are only `33` active lanes but both cross from pass at `130` to fail
  at one-child `140`, while `11x3` split as `130,10` passes. The transposed
  result argues against a single row-pitch special case; the edge follows
  crossing into a second active wave plus per-process launch lifetime.
- The exact 33-lane edge remains read-pressure dependent and mildly
  orientation/state sensitive. READS=1 on both 33-lane orientations survived
  `500` launches, while READS=2 failed at one-child `133` for `11x3` and `134`
  for `3x11`. The same run showed a clean `11x3` process split pass at total
  `133` (`132,1`), but the `3x11` split-at-edge replay shifted downward after
  reset pressure and failed in child 0. Treat the one-child thresholds as tight
  local edges, and treat split behavior near the edge as state-sensitive unless
  reproduced from a freshly reset stack.
- Fresh split-vs-same-process evidence removes the ambiguity for the `3x11`
  edge just below the failing one-child total: cross-process `132,1` passes,
  while same-process `132+1` fails at phase2 launch 0 / global 132 as the
  first risky run. The fault is therefore tied to process-local launch/queue
  lifetime, not just total launches submitted by the parent harness.
- The same `3x11` `132+1` split passed when `hipDeviceReset()` was inserted
  between phases as the first risky row. After that reset-mode pass, both
  `stream_recreate` and plain `same` mode also passed at `132+1`, but the
  one-child `134` control still failed. Treat `hipDeviceReset` as a diagnostic
  state-clearing lever near the edge, not as evidence that the reduced trigger
  is gone.
- Explicit stream and primary-context boundary modes are not interchangeable
  with `hipDeviceReset`: first-risky `stream_recreate` and
  `primary_ctx_reset` runs failed in phase1 before reaching the boundary. That
  keeps the useful cleanup claim narrow: only the observed `hipDeviceReset`
  path cleared enough state for the `132+1` split, while other HIP API shapes
  can shift the launch threshold downward.
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
- A larger grid step to reads=3/448/480x86 finally moves the one-child bracket
  upward: one-child `104` passes and one-child `105` fails. This confirms that
  total per-launch work still contributes to the edge. The process-boundary
  result still holds, though: total `120` work split as `104,16` passes in
  both plain-parent and HIP-initialized-parent modes, even though one child
  with `120` fails. The current discriminator is therefore not plain launch
  count alone and not total parent-supervised work alone; it is child-local
  same-process launch sequence length weighted by per-launch LDS work.
- At reads=3/448/448x86 the total-work term is clearer but the boundary is
  also more state-sensitive. A one-child `120` run initially passes, but after
  nearby failures the first 120-launch child of a `120,40` split fails in both
  parent modes. In contrast, `80,80` passes in both parent modes at the same
  total 160 requested launches that failed in one child. This preserves the
  child-local/process-local interpretation for lower child counts, but shows
  the pass side can move under reset pressure and should be reported as a
  bracket band rather than a deterministic threshold.
- Reads=3/448/416x86 repeats the same state-sensitive pattern at a lower
  grid. One-child `124` passes before the tightened edge, but `125` fails; then
  a `124,36` split fails inside its first child after nearby reset pressure.
  The `80,80` split still passes in both parent modes. The one-child pass/fail
  products for 480x86 (`104/105`), 448x86 (`120/121` mixed), and 416x86
  (`124/125`) cluster in the same rough total-work band, but reset pressure can
  move a previously passing child-local count onto the failing side.
- Reads=3/448/384x86 strengthens the total-work interpretation. One-child
  `134` passes, while `135` fails twice. The rough product
  `384 * 134 = 51456` is close to the prior 416x86 and 480x86 boundaries.
  Process-local state still matters: `134,46` fails inside the first child
  after reset pressure, while `90,90` passes in both parent modes. This keeps
  the model as a weighted work/sequence band rather than a pure total-launch or
  pure total-work threshold.
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

The 480x86 one-child failures (`120`, then `105`) captured the same fields in
`devcd39` and `devcd40`; the preserved one-child bracket is `104` pass / `105`
fail:

```text
[gfxhub] Page fault observed
Faulty page starting at address: 0x000074669d000000
Protection fault status register: 0x841051
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

The 448x86 one-child and split-child failures captured the same fields in
`devcd41` through `devcd46`; the lower split control `80,80` passed:

```text
[gfxhub] Page fault observed
Faulty page starting at address: 0x000074669d000000
Protection fault status register: 0x841051
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

The 416x86 one-child and split-child failures captured the same fields in
`devcd47` through `devcd51`; the lower split control `80,80` passed:

```text
[gfxhub] Page fault observed
Faulty page starting at address: 0x000074669d000000
Protection fault status register: 0x841051
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

The 384x86 one-child and split-child failures captured the same fields in
`devcd52` through `devcd55`; the lower split control `90,90` passed:

```text
[gfxhub] Page fault observed
Faulty page starting at address: 0x000074669d000000
Protection fault status register: 0x841051
regGDS_PROTECTION_FAULT                             0x3f000007
regGDS_VM_PROTECTION_FAULT                          0x0fc00113
```

The 352x86 and 320x86 runs used the same failure signature in `devcd58`
through `devcd65`:

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
| direct-AB multi-exec `6x6` block, reads=3, 448 iters, 480x86 | PASS through one-child 104; FAIL at one-child 105+; PASS for 104,16 and 60,60 child splits | `_Z25lds_direct_ab_phase_probev` | 288 B | 34 | 2 | 0 | 32 |
| direct-AB multi-exec `6x6` block, reads=3, 448 iters, 448x86 | MIXED at one-child 120 after reset pressure; FAIL at one-child 121+; PASS for 80,80 child splits | `_Z25lds_direct_ab_phase_probev` | 288 B | 34 | 2 | 0 | 32 |
| direct-AB multi-exec `6x6` block, reads=3, 448 iters, 416x86 | PASS through one-child 124 before reset pressure; FAIL at one-child 125+; PASS for 80,80 child splits | `_Z25lds_direct_ab_phase_probev` | 288 B | 34 | 2 | 0 | 32 |
| direct-AB multi-exec `6x6` block, reads=3, 448 iters, 384x86 | PASS through one-child 134; FAIL at one-child 135; PASS for 90,90 child splits | `_Z25lds_direct_ab_phase_probev` | 288 B | 34 | 2 | 0 | 32 |
| direct-AB multi-exec `6x6` block, reads=3, 448 iters, 352x86 | PASS at one-child 150; FAIL at one-child 151+; PASS for 105,45 child splits | `_Z25lds_direct_ab_phase_probev` | 288 B | 34 | 2 | 0 | 32 |
| direct-AB multi-exec `6x6` block, reads=3, 448 iters, 320x86 | PASS through one-child 162; FAIL at one-child 163+; PASS for 98,67 and 80,85 child splits | `_Z25lds_direct_ab_phase_probev` | 288 B | 34 | 2 | 0 | 32 |
| direct-AB multi-exec `6x5`/`5x6` block, reads=3, 448 iters, 512x86, one child `150` | PASS | `_Z25lds_direct_ab_phase_probev` | 248 B | 34 | 2 | 0 | 32 |
| direct-AB multi-exec `8x4`/`4x8` block, reads=3, 448 iters, 512x86, one child `150` | PASS | `_Z25lds_direct_ab_phase_probev` | 256 B | 34 | 2 | 0 | 32 |
| direct-AB multi-exec `11x3`/`3x11` block, reads=3, 448 iters, 512x86, one child `150` | FAIL at sync/global launch 98 | `_Z25lds_direct_ab_phase_probev` | 276 B | 34 | 2 | 0 | 32 |
| direct-AB multi-exec `7x5`/`5x7` block, reads=3, 448 iters, 512x86, one child `150` | FAIL at sync/global launch 98-99 | `_Z25lds_direct_ab_phase_probev` | 284 B | 34 | 2 | 0 | 32 |
| direct-AB multi-exec `6x6` block, reads=3, 448 iters, 512x86, one child `150` | FAIL at sync/global launch 99 | `_Z25lds_direct_ab_phase_probev` | 288 B | 34 | 2 | 0 | 32 |
| direct-AB multi-exec `8x4` active / `9x4` layout and `4x8` active / `4x9` layout, reads=3, 448 iters, 512x86, one child `150` | PASS | `_Z25lds_direct_ab_phase_probev` | 288 B | 34 | 2 | 0 | 32 |
| direct-AB multi-exec `8x4` active inside `9x4` block/layout, reads=3, 448 iters, 512x86, one child `150` | MIXED: initial pass; later same-ISA traffic-mask replay failed at sync/global launch 100; all-active `9x4` failed at 98 | `_Z25lds_direct_ab_phase_probev` | 288 B | 15 | 5 | 0 | 32 |
| direct-AB multi-exec `9x4` block/layout, reads=1/2/3, 448 iters, 512x86, one child `150` | reads=1 PASS; reads=2 FAIL at launch 131; reads=3 FAIL at launch 98 | `_Z25lds_direct_ab_phase_probev` | 288 B | 22/24/34 | 2 | 0 | 32 |
| direct-AB multi-exec `9x4` block/layout, reads=2, 448 iters, 512x86 | PASS at one-child `120`/`130`; FAIL at `140`/`150` | `_Z25lds_direct_ab_phase_probev` | 288 B | 24 | 2 | 0 | 32 |
| direct-AB multi-exec `9x4` block/layout, reads=2, 448 iters, 512x86, split child | PASS for `130,10` and `120,20` total `140`; one-child `140` fails | `_Z25lds_direct_ab_phase_probev` | 288 B | 24 | 2 | 0 | 32 |
| direct-AB multi-exec `9x4` block/layout, reads=1, 448 iters, 512x86 | PASS at one-child `220`/`260`/`300`/`500` | `_Z25lds_direct_ab_phase_probev` | 288 B | 22 | 2 | 0 | 32 |
| direct-AB multi-exec `8x4` block/layout, reads=2, 448 iters, 512x86 | PASS at one-child `500`/`1000` | `_Z25lds_direct_ab_phase_probev` | 256 B | 24 | 2 | 0 | 32 |
| direct-AB multi-exec `11x3`/`3x11` block/layout, reads=1, 448 iters, 512x86 | PASS at one-child `500` for both orientations | `_Z25lds_direct_ab_phase_probev` | 276 B | 22 | 2 | 0 | 32 |
| direct-AB multi-exec `11x3`/`3x11` block/layout, reads=2, 448 iters, 512x86 | `11x3`: PASS `131`/`132`, FAIL `133`, split `132,1` PASS; `3x11`: PASS `131`-`133`, FAIL `134`, cross-process `132,1` PASS, same-process `132+1` FAIL at phase2 launch 0, `device_reset` boundary `132+1` PASS | `_Z25lds_direct_ab_phase_probev` | 276 B | 24 | 2 | 0 | 32 |
| direct-AB phase-mode `3x11` block/layout, reads=2, 448 iters, 512x86, first-risky cleanup modes | `stream_recreate 132+1` FAILS in phase1/global 69; `primary_ctx_reset 132+1` FAILS in phase1/global 33 | `_Z25lds_direct_ab_phase_probev` | 276 B | 24 | 2 | 0 | 32 |
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
> fail and split-child passes. A larger step to 480x86 moves the one-child
> edge to `104` pass / `105` fail, but `104,16` split children still pass at
> total 120. The immediate trigger now looks like process-local launch sequence
> state weighted by per-launch LDS work, not launch count alone. At 448x86,
> `80,80` split children still pass while one-child `160` fails, but the
> `120` child-local point becomes mixed after reset pressure. The edge is a
> moving band, not a deterministic scalar threshold. At 416x86, the one-child
> edge moves to `124` pass / `125` fail before `124,36` also fails after reset
> pressure, while `80,80` still passes. The total-work term is real, but the
> process-local state term remains load-bearing. At 384x86, the one-child edge
> lands at `134` pass / `135` fail, closely matching the rough
> `grid_x * launches` work band, while `134,46` still fails after reset
> pressure and `90,90` passes. This is the cleanest current evidence that both
> total per-child LDS work and process-local state are involved.
> Exec-mask structure alone does not appear to be the deciding factor.

## Public Report Refresh

Checked public gfx1103/780M ROCm reports again on 2026-06-20. The closest
external symptom match remains ROCm/TheRock issue
`AMD Radeon 780M (gfx1103) hanging, debugging tips?`, where a Ryzen 7840HS /
Radeon 780M system reports `hipSyncError: 719` and display reset/recovery on a
large rocBLAS `cgemm` workload, with ROCm 6.3.1 passing and 6.3.3 / 6.4.1 /
7-rc failing:
<https://github.com/ROCm/TheRock/issues/1264>. That issue is still not reduced
to an LDS/backedge/kernel-lifetime trigger, but it does support the broader
idea that gfx1103 can hit ROCm-stack 719/reset failures on matrix workloads.

Other recent gfx1103 reports are less directly diagnostic for this bug:
llama.cpp issue `#20839` and ROCm issue `#6049` focus on missing
`TensileLibrary_lazy_gfx1103.dat`, rocBLAS routing, and WMMA/FlashAttention
fallback behavior rather than an isolated custom-HIP LDS loop:
<https://github.com/ggml-org/llama.cpp/issues/20839> and
<https://github.com/ROCm/ROCm/issues/6049>. HIP's public error-code docs map
illegal-memory failures to `hipErrorIllegalAddress` (`700`), not the observed
HIP `719`, so the decisive evidence continues to be the amdgpu/MES dmesg and
devcoredump signature captured locally:
<https://rocm.docs.amd.com/projects/HIP/en/latest/reference/error_codes.html>.

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
  `99` fail with the same split-child passes. At 480x86 the edge moves to
  `104` pass / `105` fail, while total `120` split as `104,16` still passes.
  At 448x86, `80,80` split children pass while one-child `160` fails, but
  child-local `120` is mixed after reset pressure. Next, repeat the 384x86/416x86
  near-edge region after fresh one-child runs (plain/`hipinit_reset_before`) to
  quantify reset-pressure drift before fitting `grid_x * grid_y * child_launches`.
  Treat the common
  in-process HIP reset APIs as already tested; only revisit teardown if a
  genuinely different ROCm mechanism is identified.
- rerun the passing long-loop symbol on the second 780M only if its compile-only
  ISA/resource rows drift from the local reference; the source and summarizer
  are now promoted in `scripts/`.

### Promoted Direct-AB Multi-Exec Jig

The strongest no-output reduced repro is now promoted into the repo:

```bash
# Safe: build child/parent and capture ISA/readobj only.
BUILD_ONLY=1 scripts/lds_direct_ab_multi_exec_matrix.sh /tmp/hipfire-lds-direct-ab-promoted

# Risky: one child at the known reads=3 / 448-iteration / 512x86 edge.
BUILD_ONLY=0 CHUNKS=101 GRID_X=512 GRID_Y=86 \
  scripts/lds_direct_ab_multi_exec_matrix.sh /tmp/hipfire-lds-direct-ab-promoted

# Risky control: same total split across child processes.
BUILD_ONLY=0 CHUNKS=96,5 GRID_X=512 GRID_Y=86 \
  scripts/lds_direct_ab_multi_exec_matrix.sh /tmp/hipfire-lds-direct-ab-promoted
```

Validation on this checkout:

```text
BUILD_ONLY=1 scripts/lds_direct_ab_multi_exec_matrix.sh /tmp/hipfire-lds-direct-ab-promote-buildonly
```

The build-only artifact
`/tmp/hipfire-lds-direct-ab-promote-buildonly/a6x6_b6x6_r3_i448_chunks96_5_multi_plain_g512x86/`
compiled both binaries and captured the expected direct-AB kernel metadata:
`group_segment_fixed_size=288`, `sgpr_count=2`, `vgpr_count=34`,
`wavefront_size=32`. The wrapper now regenerates:

```text
/tmp/hipfire-lds-direct-ab-promote-buildonly/direct-ab-artifact-summary.tsv
/tmp/hipfire-lds-direct-ab-promote-buildonly/direct-ab-artifact-summary.md
```

The earlier reads=6 / 192-iteration / 511x86 edge also compiles from the
promoted source in build-only mode:

```text
READS=6 ITERS=192 CHUNKS=98,2 GRID_X=511 GRID_Y=86 BUILD_ONLY=1 \
  scripts/lds_direct_ab_multi_exec_matrix.sh /tmp/hipfire-lds-direct-ab-promote-r6-buildonly
```

Its build-only artifact captured `group_segment_fixed_size=288`,
`sgpr_count=2`, `vgpr_count=52`, and `wavefront_size=32`.

The direct-AB summarizer also parses older risky multi-exec artifacts. On
`/tmp/hipfire-lds-direct-ab-multi-exec-artifacts/`, the current summary captures
the known failing rows with:
`sync_failure=phase1 sync ... failed: unspecified launch failure (719)`,
`devcore_fault_addr=0x000074669d000000`,
`devcore_prot_status=0x841051`,
`devcore_gcvm_flags=MORE_FAULTS,PERMISSION_FAULTS,RW`,
`devcore_gds_protection_fault=0x3f000007`, and
`devcore_gds_vm_protection_fault=0x0fc00113`.

Compare two direct-AB summary TSVs directly with:

```bash
scripts/lds_direct_ab_summary_compare.sh \
  /tmp/hipfire-lds-direct-ab-promote-buildonly/direct-ab-artifact-summary.tsv \
  /path/to/other-direct-ab-artifact-summary.tsv
```

Comparator validation:

```text
scripts/lds_direct_ab_summary_compare.sh self self
```

returns 24-column `same` rows for build-only and risky saved roots. Artificial
mutations correctly classify an exit/sync change as
`same-codegen-runtime-diff`, a resource tuple change as `resource-drift`, and
non-overlapping reads/iteration shapes as `left-only` / `right-only`.
Legacy direct-AB artifacts that predate `build_only` / `hipcc` metadata treat
those blank fields as unknown rather than drift. A same-exit but different
sync-failure line is classified as `same-codegen-sync-detail-diff`.

Fresh promoted risky run from detached throwaway worktree
`/tmp/hipfire-lds-direct-ab-risky-c9522387/` at commit `c9522387`:

```text
BUILD_ONLY=0 CLEAR_COREDUMP=1 WAIT_DEVCD_MS=12000 CHUNKS=96,5 GRID_X=512 GRID_Y=86 READS=3 ITERS=448 MODE=plain \
  scripts/lds_direct_ab_multi_exec_matrix.sh /tmp/hipfire-lds-direct-ab-risky-c9522387-artifacts

BUILD_ONLY=0 CLEAR_COREDUMP=1 WAIT_DEVCD_MS=12000 CHUNKS=101 GRID_X=512 GRID_Y=86 READS=3 ITERS=448 MODE=plain \
  scripts/lds_direct_ab_multi_exec_matrix.sh /tmp/hipfire-lds-direct-ab-risky-c9522387-artifacts
```

Results:

- `chunks=96,5`: passed, exit `0`, no devcoredump, no dmesg delta.
- `chunks=101`: failed, exit `4`, `phase1 sync 24 global 24 failed:
  unspecified launch failure (719)`, with `dmesg_remove_queue=3` and a late
  generic coredump at `coredumps/late_2000ms.devcd135.data`.
- The failing promoted run reproduced the direct-AB fault signature:
  `devcore_fault_addr=0x000074669d000000`,
  `devcore_prot_status=0x841051`,
  `devcore_gcvm_flags=MORE_FAULTS,PERMISSION_FAULTS,RW`,
  `devcore_gcvm_cid=8`, `devcore_gcvm_rw=1`, `devcore_gcvm_vmid=8`,
  `devcore_gds_protection_fault=0x3f000007`,
  `devcore_gds_vm_protection_fault=0x0fc00113`.
- Comparing this fresh root against
  `/tmp/hipfire-lds-direct-ab-multi-exec-artifacts/direct-ab-artifact-summary-validate.tsv`
  reports the `chunks=101` row as `same-codegen-sync-detail-diff`: normalized
  ISA, resources, exit code, devcore, GCVM, and GDS signatures match, while the
  failure point moved from old `sync 100/global 100` to fresh `sync 24/global
  24`. This reinforces the state-sensitive threshold model while preserving the
  same fault mechanism.

### Additional Sequence Sweep (Fresh)

- Fresh standalone launcher added at `/tmp/hipfire-lds-prewarm-probe.hip` splits
  a run into a warmup phase and a target phase inside one HIP process. The target
  kernel is the same all-active `6x6` direct-AB probe (`READS=6`,
  `ITERS=192`, `grid=511x86`, `ARCH=gfx1103`, `BLOCK=6x6`).
- Artifact directory: `/tmp/hipfire-lds-prewarm-artifacts/`.
- Results:
  - `w0_t100_g511x86` (`0` warmup, `100` target): **fail** at target
    `sync 98`.
  - `w0_t80_g511x86` (`0` warmup, `80` target): **pass**.
  - `w20_t60_g511x86` (`20` warmup, `60` target): **pass**.
  - `w40_t60_g511x86` (`40` warmup, `60` target): **pass**.
  - `w20_t80_g511x86` (`20` warmup, `80` target): **pass**.
  - `w80_t100_g511x86` (`80` warmup, `100` target): **fail** at target
    `sync 98`.
  - `w160_t100_g511x86` (`160` warmup, `100` target): **fail** at target
    `sync 98`.
  - Fresh single-process edge sweep at `g=511` (`0` warmup):
    `t=90`, `91`, `92`, `93`, `94`, `95`, `96`, `97`, `98` **pass**;  
    `t=99`, `100`, `101`, `102` **fail** (HIP `719`, `target sync 98/99`).
- All 100-target failures captured the same coredump signature in
  `devcoredump.data`:
  `0x074669d000000`, protection status `0x841051`,
  `regGDS_PROTECTION_FAULT 0x3f000007`,
  `regGDS_VM_PROTECTION_FAULT 0x0fc00113`.
- Interpretation: with this protocol, large warmup launch mass does not shift
  the edge; the immediate trigger remains attached to the target-kernel launch
  sequence/work, supporting a target-specific accumulation model rather than
  pure process-launch volume.

### Promoted Scalar-Control Jig

- The standalone matrix runner now includes the scalar-control probes needed
  for cross-machine 780M testing:
  - `tile6_lds_snop_noextra_load4_consume4_pinned`
  - `tile6_lds_tail_snop_noextra_load4_consume4_pinned`
  - `tile6_lds_counter_noextra_load4_consume4_pinned`
  - `tile6_lds_counter_mask_noextra_load4_consume4_pinned`
- Baseline cross-check command:

```bash
VARIANT=tile6_lds_snop_noextra_load4_consume4_pinned \
MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 \
scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-scalar-control-runs
```

- Focused cross-system test jig:

```bash
OUT=/tmp/hipfire-lds-tail-snop-780m tests/gfx1103-lds-tail-snop-repro.sh
```

Use `PROFILE=kedge` for the current K-limit edge (`K_LIMIT=3024` pass-side,
`K_LIMIT=3032` fail-side) or `PROFILE=full` for baseline, K-edge, and full-depth
tail-noop checks in one run.

Print the full second-780M command sequence with:

```bash
scripts/lds_gemm_780m_runbook.sh
```

- One-command second-780M jig:

```bash
# Current direct-AB reduced repro. Safe: compiles and captures codegen only.
scripts/lds_direct_ab_780m_test_jig.sh

# Risky: expected pass-side controls followed by the READS=2 9x4 fail-side.
scripts/lds_direct_ab_780m_test_jig.sh --risky
```

The direct-AB jig writes:

```text
/tmp/hipfire-lds-direct-ab-780m-buildonly/direct-ab-artifact-summary.tsv
```

Compare two machines' direct-AB summaries with:

```bash
scripts/lds_direct_ab_780m_test_jig.sh --compare local-direct-ab-summary.tsv other-780m-direct-ab-summary.tsv
```

- Legacy GEMM/tail-snop second-780M jig:

```bash
# Safe: compiles and captures codegen artifacts only.
scripts/lds_gemm_780m_test_jig.sh

# Risky: expected to exercise the HIP-719/reset path on affected stacks.
scripts/lds_gemm_780m_test_jig.sh --risky
scripts/lds_gemm_780m_test_jig.sh --kedge
```

The default safe jig also writes:

```text
/tmp/hipfire-lds-tail-snop-780m-buildonly/isa-placement-single/isa-summary.tsv
```

Compare two machines' placement-aware ISA summaries with:

```bash
scripts/lds_gemm_isa_summary_compare.sh local-isa-summary.tsv other-780m-isa-summary.tsv
# or:
scripts/lds_gemm_780m_test_jig.sh --isa-compare local-isa-summary.tsv other-780m-isa-summary.tsv
```

- Summarize an artifact root for cross-machine comparison:

```bash
scripts/lds_gemm_artifact_summary.sh /tmp/hipfire-lds-tail-snop-780m
```

Compare the `source_sha256`, `amdgpu_obj_sha256`, and `amdgpu_isa_sha256`
columns first when comparing 780M machines. Matching source with different
object/ISA hashes means compiler/toolchain or environment drift is part of the
comparison; matching object/ISA hashes makes runtime/driver behavior the main
remaining difference.

Compare two summary TSVs directly with:

```bash
scripts/lds_gemm_summary_compare.sh local-summary.tsv other-780m-summary.tsv
```

- Current clean local build-only reference from `chaingun` commit
  `29307782b127` on this 780M (`BUILD_ONLY=1`, ROCm HIP
  `7.13.26176-79e85e1468`, driver `6.19.0`):
  - no-extra baseline:
    `source_sha256=4267f867c3901afc`,
    `selected_isa_norm_sha256=07a8198f82d17006`.
  - tail `s_nop`:
    `source_sha256=4267f867c3901afc`,
    `selected_isa_norm_sha256=abcf16851242d139`.
  The current build-only wrapper treats successful build-only capture as a
  match for both rows, so it should exit zero when both variants compile and
  summarize cleanly.
  The expected failing coredump signature for the reset-prone path is
  `devcore_sig=1/0x0000000000000000/0x0/0x3f000007/0x0fc00113`,
  with `devcore_gds_addr=0xfc0`, `devcore_gds_vm_vmid=1`, and
  `devcore_gds_vm_addr=0xfc0`.

- Safe build/codegen metadata capture without launching the repro:

```bash
BUILD_ONLY=1 VARIANT=tile6_lds_tail_snop_noextra_load4_consume4_pinned \
MODE=full N_LAUNCH=100 M=512 N=3072 K=3072 K_LIMIT=0 \
scripts/lds_gemm_standalone_matrix.sh /tmp/hipfire-lds-tail-snop-buildonly
```

- A tighter throwaway control in `/tmp/hipfire-lds-scalar-nop-runs/` shows
  `tile6_lds_snop_noextra_load4_consume4_pinned` passed one-launch smoke and
  failed under the 100-launch full-shape run at `sync 98` with HIP `719`.
  The object metadata stayed at `group_segment_fixed_size=144`,
  `sgpr_count=5`, `vgpr_count=10`, `wavefront_size=32`, and no private
  segment. The ISA delta from the pass-side no-extra shape is one recurrent
  `s_nop 0` before the first LDS store in each K iteration. A follow-up
  `tile6_lds_store_then_load_dynamiccols_load4_noextra_consume4_pinned`
  100-launch recovery run passed, so this split is not explained by device
  wedging alone.
- A placement follow-up in `/tmp/hipfire-lds-snop-placement-runs/` shows
  `tile6_lds_tail_snop_noextra_load4_consume4_pinned` also passed one-launch
  smoke and failed under the 100-launch full-shape run at `sync 98` with HIP
  `719`. The object metadata stayed at `group_segment_fixed_size=144`,
  `sgpr_count=5`, `vgpr_count=10`, `wavefront_size=32`, and no private
  segment. The ISA places the only inserted `s_nop 0` after the final
  `s_barrier`/`buffer_gl0_inv` and immediately before the K-loop branch. A
  follow-up no-extra 100-launch recovery run passed.
- Offline comparison of the saved code object from
  `/tmp/hipfire-lds-snop-placement-runs/tail-n100/` rules out a larger-resource
  explanation for the tail-noop failure. The pass-side no-extra symbol
  (`tile6_lds_store_then_load_dynamiccols_load4_noextra_consume4_pinned`) is
  392 bytes with 79 instructions, five `s_nop 0` instructions, two
  `ds_store_b32`, four `ds_load_b32`, three `s_barrier`, and two scalar
  branches. All five no-extra `s_nop 0` instructions are prologue/padding
  before the loop body. The failing tail-noop symbol is smaller at 368 bytes
  with 73 instructions, one `s_nop 0`, the same two LDS stores, four LDS
  loads, three barriers, and two scalar branches. Both symbols use
  `group_segment_fixed_size=144`, `private_segment_fixed_size=0`,
  `sgpr_count=5`, `vgpr_count=10`, and `wavefront_size=32`. The distinguishing
  delta is placement of the recurrent scalar padding at the loop tail/backedge,
  after the final barrier/gl0 invalidation and before the branch, not total
  instruction count, register pressure, or LDS footprint.
- The same comparison can now be regenerated without launching the repro:

```bash
scripts/lds_gemm_isa_compare.sh /tmp/hipfire-lds-gemm-isa-compare
SINGLE_INSTANTIATION=1 scripts/lds_gemm_isa_compare.sh /tmp/hipfire-lds-gemm-isa-compare-single
```

The helper writes `isa-summary.tsv` plus per-symbol ISA and key-op extracts,
with `SYMBOLS="..."` available for later single-symbol comparisons. The
default symbol set now covers no-extra, counter-only, pre-store `s_nop`, tail
`s_nop`, and counter-mask. The TSV now includes placement columns for the
current scalar-control lead: `pre_ds_s_nop`, `pre_ds_s_add_i32`, `tail_s_nop`,
`tail_s_add_i32`, and `tail_window`. Full-object and single-instantiation modes
emit the same core counts for these symbols, so the saved full-object
comparison is not being skewed by neighboring template variants.
- A sequential K-limit sweep in
  `/tmp/hipfire-lds-tail-snop-klimit-runs/` shows the tail-noop trigger is
  concentrated near the top of the K loop. The initial concurrent
  `K_LIMIT=2048`/`2816` run is invalid protocol and should be ignored: both
  jobs were stressing the GPU at once and failed early. In clean sequential
  runs, `K_LIMIT=512`, `1024`, `1536`, `2048`, `2560`, `2816`, `2880`,
  `2944`, `3008`, `3024`, and `3030` passed 100 launches. Full depth
  (`K_LIMIT=0`) failed at `sync 98`; `K_LIMIT=3040` failed at `sync 99`;
  `K_LIMIT=3032` failed twice at `sync 99`; and `K_LIMIT=3031` first passed
  then failed at `sync 99` after additional reset pressure. Follow-up
  no-extra 100-launch recovery runs passed after the failing tail-noop runs.
  This makes the upper edge stress-history-sensitive, but the robust pass
  side currently extends through `K_LIMIT=3024`.
- A launch-count bracket in `/tmp/hipfire-lds-tail-snop-launch-runs/` shows
  the near-full tail-noop edge is strongly tied to the 100th launch.
  `K_LIMIT=3032` passed with `N_LAUNCH=96`, `98`, and `99`, then failed at
  `sync 99` with `N_LAUNCH=100`. `K_LIMIT=3040` also passed with
  `N_LAUNCH=99` and failed at `sync 99` with `N_LAUNCH=100`. Full-depth
  tail-noop was harsher after reset pressure, failing at `sync 98` even with
  `N_LAUNCH=99`. After that sequence, the previously pass-side
  `tile6_lds_store_then_load_dynamiccols_load4_noextra_consume4_pinned`
  also became fragile at the 100th launch: it failed at `sync 99` with
  `N_LAUNCH=100` but passed with `N_LAUNCH=99`. Treat late-session
  100-launch results as reset-pressure-sensitive unless bracketed by fresh
  process/recovery controls.
- On the first 780M system, the counter-only variant passed one-launch smoke
  and failed under the 100-launch full-shape run at `sync 98` with HIP `719`.
  The object metadata stayed at `group_segment_fixed_size=144`,
  `sgpr_count=5`, `vgpr_count=10`, `wavefront_size=32`, and no private
  segment. The ISA delta from the pass-side no-extra shape is a loop-carried
  scalar increment (`s_add_i32`) and sink, with no extra LDS operation and no
  extra branch beyond the normal K-loop branch.
- The compile-only single-instantiation helper now covers both LDS-only
  synthetic probes plus all promoted scalar-control variants. With
  `SINGLE_INSTANTIATION=1`, the current object counts are:
  - unmasked synthetic GEMM-shaped `tile6_synth`: 300 bytes, 61 instructions,
    zero `s_nop 0`, one LDS store, five LDS loads, two barriers, two scalar
    branches, `group_segment_fixed_size=288`, `sgpr_count=5`, `vgpr_count=18`,
    `wavefront_size=32`.
  - masked synthetic GEMM-shaped `tile6_synth_masked`: 392 bytes,
    84 instructions, twelve `s_nop 0`, one LDS store, five LDS loads,
    two barriers, four scalar branches, `group_segment_fixed_size=288`,
    `sgpr_count=7`, `vgpr_count=18`, `wavefront_size=32`.
  - no-extra pass-side: 392 bytes, 79 instructions, five `s_nop 0` prologue
    padding ops, two LDS stores, four LDS loads, three barriers, two scalar
    branches.
  - counter-only: 372 bytes, 74 instructions, zero `s_nop 0`, same LDS/barrier
    counts, same branch count, with an extra loop-carried `s_add_i32` before
    the first LDS store.
  - pre-store `s_nop`: 368 bytes, 73 instructions, one `s_nop 0` before the
    first LDS store, same LDS/barrier/branch counts.
  - tail `s_nop`: 368 bytes, 73 instructions, one `s_nop 0` after the final
    barrier/gl0 invalidation and before the loop branch, same
    LDS/barrier/branch counts.
  - counter-mask: 400 bytes, 80 instructions, zero `s_nop 0`, same LDS/barrier
    counts, same branch count.
  The five scalar-control probes all use `group_segment_fixed_size=144`,
  `private_segment_fixed_size=0`, `sgpr_count=5`, `vgpr_count=10`, and
  `wavefront_size=32`. A fresh placement-aware compile-only summary at
  `/tmp/hipfire-lds-gemm-isa-placement-single/` adds these scalar placement
  facts:
  - no-extra pass-side:
    `pre_ds_s_nop=5`, `pre_ds_s_add_i32=0`, `tail_s_nop=0`,
    `tail_window=s_barrier;buffer_gl0_inv;s_cbranch_scc0`.
  - counter-only:
    `pre_ds_s_nop=0`, `pre_ds_s_add_i32=1`, `tail_s_nop=0`,
    `tail_window=s_barrier;buffer_gl0_inv;s_cbranch_scc0`.
  - pre-store `s_nop`:
    `pre_ds_s_nop=1`, `pre_ds_s_add_i32=0`, `tail_s_nop=0`,
    `tail_window=s_barrier;buffer_gl0_inv;s_cbranch_scc0`.
  - tail `s_nop`:
    `pre_ds_s_nop=0`, `pre_ds_s_add_i32=0`, `tail_s_nop=1`,
    `tail_window=s_barrier;buffer_gl0_inv;s_nop;s_cbranch_scc0`.
  - counter-mask:
    `pre_ds_s_nop=0`, `pre_ds_s_add_i32=1`, `tail_s_nop=0`,
    `tail_window=s_barrier;buffer_gl0_inv;s_cbranch_scc0`.
  This strengthens the scalar-control/backedge-timing
  lead: the observed failures are not tracking LDS operation count, barrier
  count, register pressure, LDS footprint, or total instruction count in a
  monotonic way. Full-object mode and isolated mode agree for the masked synth
  and scalar-control rows; the unmasked synthetic row differs in VGPR metadata
  (`26` in the full object versus `18` isolated), so use
  `SINGLE_INSTANTIATION=1` for resource-count evidence on that symbol.
- The promoted LDS-only long-loop helper captures the passing masked control and
  the no-mask threshold control without launching either kernel:

```bash
SINGLE_INSTANTIATION=1 scripts/lds_standalone_isa_compare.sh /tmp/hipfire-lds-standalone-isa-single
SINGLE_INSTANTIATION=0 scripts/lds_standalone_isa_compare.sh /tmp/hipfire-lds-standalone-isa-full
```

  On this stack, full-object and isolated mode agree for both rows:
  - masked passing long-loop `_Z9lds_probeILi6ELi6ELi512EEvPfi`: 592 bytes,
    115 instructions, zero `s_nop 0`, two LDS stores, ten LDS loads,
    four barriers, nine waits, six scalar branches, five `s_and_saveexec`,
    one global store, `group_segment_fixed_size=288`, `sgpr_count=8`,
    `vgpr_count=20`, `wavefront_size=32`.
  - no-mask long-loop `_Z16lds_probe_nomaskILi6ELi512EEvPfi`: 780 bytes,
    146 instructions, zero `s_nop 0`, four LDS stores, twenty LDS loads,
    eight barriers, thirteen waits, one scalar branch, no `s_and_saveexec`,
    one global store, `group_segment_fixed_size=288`, `sgpr_count=8`,
    `vgpr_count=56`, `wavefront_size=32`.
  This resolves the per-symbol artifact gap for the passing long-loop control:
  it genuinely has more DS/barrier/wait traffic than the failing compact
  synthetic symbols, so aggregate object counts were not hiding a simpler
  "more LDS ops fails" relation.
- This sharpens the current suspect further: in the
  second-store/four-waited-row-load loop, a single recurrent scalar no-op is
  enough to move the pass-side no-extra shape into the faulting class under
  full-K repeated-launch stress even when the no-op is placed at the loop tail,
  after the LDS work and final barrier. It does not require an extra recurrent
  barrier, loop-carried scalar data dependency, extra LDS operation, register
  pressure change, spill, or different LDS allocation size.

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
