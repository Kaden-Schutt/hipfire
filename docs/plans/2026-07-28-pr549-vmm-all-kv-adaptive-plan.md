# PR #549 VMM all-KV + adaptive implementation plan

**Date:** 2026-07-28  
**Design contract:** `docs/plans/2026-07-28-pr549-vmm-all-kv-adaptive-design.md`  
**Execution scope:** single-GPU Qwen3.5/3.6 masked KV only

## Outcome

PR #549 is mergeable only when the VMM backend owns stable K/V virtual addresses for all seven static KV modes, legal Lloyd-V overrides, and adaptive KV; adaptive transitions and resets remain capacity-safe; unload cannot hide pending VMM arenas; and the CLI/serve proof regressions found in review are closed.

This plan deliberately excludes PP, multi-GPU, CASK/TriAttention compaction, flat-Llama carriers, and adaptive DFlash. Static Q8/FWHT3 DFlash remains required.

## Shared contracts

1. **Backend ownership:** a real VMM K/V tensor is never replaced by `gpu.zeros`, `set_v_mode_realloc`, or `set_adaptive_floor_alloc` after publication.
2. **Stable address:** each real FA layer has one K arena and one V arena for the load lifetime.
3. **Capacity:** K and V reserve/mapping arithmetic is independent; admitted tokens are the minimum of both current-tier capacities.
4. **Adaptive schedule:** start is exactly FWHT4/Q8. Reserve tiers come from the controller’s actual floors, not separately parsed hints.
5. **Failure:** map failure precedes write; partial transcode poisons the model; teardown failure is returned and leaves a registered retry target.
6. **Execution capture:** static mapping growth preserves capture identity; tier changes/reset invalidate HipGraph and retained replay state.
7. **Proof:** load markers are not request-route proof. DFlash requires current-attempt, request-level evidence.

## Phase 1 — impact map and API boundary

Before editing each named symbol, run GitNexus upstream impact and warn on HIGH/CRITICAL results as required by `AGENTS.md`. Run LSP references before changing exported signatures.

Establish the narrow API boundary:

- `crates/hipfire-runtime/src/llama.rs`
  - replace the Q8/Asym3/FWHT3-only layout branching with one validated private K/V layout representation;
  - keep existing public wrappers where they remain useful, but make layout bytes the single source of truth;
  - represent reserve bytes separately from current encoding stride so adaptive can reserve at floors while starting FWHT4/Q8.
- `crates/hipfire-arch-qwen35/src/carrier.rs`
  - resolve K mode, static V override, and adaptive controller before allocating KV;
  - pass one resolved storage request into `KvCache`.
- `crates/hipfire-arch-qwen35/src/carrier.rs` / `crates/hipfire-loader/src/lib.rs`
  - carry `Option<KvAdaptive>` through `Qwen35Bundle` and attach it to `LoadedModel`.

Do not add a second public configuration surface. Existing `kv_mode`, `HIPFIRE_KV_V`, `kv_adaptive`, and `kv_backend` remain authoritative.

## Phase 2 — all static VMM layouts

### Runtime storage changes

In `crates/hipfire-runtime/src/llama.rs`:

1. Generalize packed K bytes/head for Q8 and asym/FWHT 2/3/4-bit formats.
2. Generalize V bytes/head for Q8 and Lloyd2/3/4.
3. Compute checked per-token K/V strides and checked reserve elements independently.
4. Construct VMM caches for:
   - `q8`;
   - `asym2`, `asym3`, `asym4`;
   - `fwht2`, `fwht3`, `fwht4`;
   - legal FWHT-K + Lloyd-V combinations.
5. Reuse existing Givens/FWHT sign-table rules exactly; upgrade to 256-wide signs when the selected Lloyd/FWHT path requires them.
6. Extend backend validation to the seven static modes while preserving single-GPU, masked-layer, positive-capacity, and no-CASK gates.
7. Make `vmm_bytes_per_token`, `fast_mapped_token_capacity`, `mapped_token_capacity`, and `ensure_mapped_capacity` derive from current cache flags/V mode for every legal static encoding.
8. Preserve constructor rollback: every allocated arena is released if a later allocation, initial map, or marker calculation fails.

### Carrier changes

In `crates/hipfire-arch-qwen35/src/carrier.rs`:

- parse `HIPFIRE_KV_V` before cache construction;
- reject illegal Asym/Lloyd combinations explicitly;
- for VMM, construct the final static K/V layout directly;
- for contiguous, preserve existing observable behavior while sharing validation where practical.

### Functional smoke checkpoint

On gfx1201 GPU2, exercise each static mode through load, prefill, AR decode, graph capture/replay, at least one mapping boundary, unload, and reload. Stop and fix source behavior before admitting cleanup work if any mode silently falls back to contiguous or produces incoherent output.

## Phase 3 — adaptive VMM integration

### Resolve one authoritative schedule

In `crates/hipfire-arch-qwen35/src/carrier.rs` and `crates/hipfire-runtime/src/kv_adaptive.rs`:

1. Parse preset/advanced policy once.
2. Build `KvAdaptive` first and take `k_floor`/`v_floor` from that controller.
3. Correct Balanced to FWHT3/Lloyd3 everywhere.
4. Require the adaptive start encoding exactly FWHT4/Q8.
5. Reject adaptive with explicit Lloyd override, CASK, PP/multi-GPU, unsupported head geometry, insufficient start capacity, or DFlash.
6. Treat malformed/unsupported explicit adaptive requests as load errors rather than ignored warnings.

### Floor-reserved VMM construction

In `crates/hipfire-runtime/src/llama.rs`:

- reserve K at `max_seq × n_kv_heads × k_floor_bph`;
- reserve V at `max_seq × n_kv_heads × v_floor_bph`;
- expose current strides as FWHT4/Q8 initially;
- map based on current strides and current required positions;
- after successful downshift, retain mapped pages and recompute token capacity from the new current stride;
- never call the contiguous adaptive resize path on VMM owners.

### Controller attachment and runtime hooks

- Add `kv_adaptive: Option<KvAdaptive>` to `Qwen35Bundle` or an equivalently narrow load result.
- Move it into `LoadedModel.kv_adaptive` in `finish_qwen35_load`.
- Keep prefill-chunk, post-prefill, and linear-decode hooks, but propagate failures.
- A map-growth error aborts before the next write.
- A transcode error after any layer may have changed poisons the loaded model; further generation is rejected until unload/reload.

### Reset and capture invalidation

- Replace controller-only reset with one operation that, after the cache is cleared, restores cache flags and controller state to FWHT4/Q8 step 0.
- Use it for explicit reset, abort cleanup, context rollover, and reload lifecycle.
- Extend `Gpu::invalidate_for_kv_mode_switch` (or add a narrowly named sibling) to destroy HipGraph state and reset/poison the retained replay controller so no old KV-mode tape can run.

### Functional smoke checkpoint

On gfx1201 GPU2, force conservative, balanced, and aggressive thresholds during multi-chunk prefill and AR decode. Observe stable K/V base addresses, successful tier logs, increased mapped-token capacity after stride reduction, coherent output, reset back to FWHT4/Q8, and clean unload/reload.

## Phase 4 — teardown and retry semantics

Affected files:

- `crates/hipfire-runtime/src/llama.rs`
- `crates/rdna-compute/src/dispatch.rs`
- `crates/hipfire-loader/src/lib.rs`
- `crates/hipfire-runtime/examples/daemon.rs`
- low-level VMM smoke examples as needed

Required behavior:

1. `KvCache::free_gpu` attempts every tensor and returns cleanup failure instead of discarding it.
2. `Gpu::free_tensor` continues registering arenas whose release did not complete.
3. Model unload invokes registered-arena retry after model-specific frees and ordinary pool drain.
4. The daemon emits unload success only when no pending VMM arena remains.
5. A new model load retries and then refuses to proceed if prior arenas remain pending.
6. Initial-map/access failures release all earlier owners; mid-growth failure preserves the prior prefix.

Functional checkpoint: injected map/access/unmap/release failures must never produce a false clean unload, and a successful retry must restore the exact idle allocation count before reload.

## Phase 5 — CLI and serve proof corrections

These are independent of the storage implementation and may execute in parallel once their impact analyses are complete.

### Final DFlash draft projection

In `crates/hipfire-cli/src/main.rs`:

- finalize the CLI speculation selector first;
- then project inherited `HIPFIRE_DFLASH_DRAFT` when the final selector enables DFlash;
- cover config-off + inherited draft + `run --spec dflash` without changing config-only behavior.

### Current-attempt, request-level serve proof

In `scripts/serve_harness.py`:

- record the serve-log byte offset before each spawn attempt;
- evaluate startup VMM/draft/graph markers only in that attempt’s slice;
- run DFlash route assertions after requests return;
- require request-level DFlash `tau`/cycle or equivalent exact discriminator;
- fail an all-AR run even if the draft loaded.

Functional checkpoint: a synthetic stale prior-attempt log cannot satisfy current proof, and a loaded-but-forced-AR DFlash run fails closed.

## Phase 6 — integrated hardware smoke

Use the approved GPU2 route and current source/binary identity.

1. Repeat the all-seven static matrix.
2. Repeat adaptive conservative/balanced/aggressive transition/reset matrix.
3. Repeat static Q8 and FWHT3 production serve DFlash with modern speculation TOML and request-level proof.
4. Repeat graph capture, context growth, unload/reload, and allocation-failure paths.
5. Repeat MQ4R Q8 TG128 retained-PM4 matched comparison:
   - valid lifecycle and timed route proof;
   - contiguous and VMM stationary measurements;
   - `max_seq=32768` load VRAM;
   - exact idle footprint after unload.

The existing measured reference is not a guaranteed floor: contiguous 200.975 tok/s, VMM 202.544/201.121 tok/s, 320 MiB lower initial VRAM, idle 59,912,192 B. New evidence must be reported as a fresh matched observation.

## Cleanup admission

Only after the integrated smoke demonstrates the requested behavior may the execution tracker add cleanup tasks for permanent tests, docs reconciliation, rustfmt, clippy, workspace unit checks, GitNexus `detect_changes`, and final adversarial review. This preserves the repository workflow rule that housekeeping cannot steer an unproven implementation.
