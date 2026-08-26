# Design: `HIPFIRE_EMULATE_GPUS` — single-card dual-GPU emulation for multi-GPU paths

**Date:** 2026-07-03
**Branch:** `feature/parallel-expansion`
**Status:** design approved, pending spec review

> **SUPERSEDED — historical (2026-08-26 mainline merge).** This design's
> behavior no longer matches the merged control plane. `HIPFIRE_EMULATE_GPUS`
> remains env-only **hardware-resolution emulation** (`hipfire-hardware`
> `resolve_device_ids` aliasing), but it **no longer defaults `tp`**: the
> `resolve_parallelism` "default mode is TP" precedence was deleted with the
> pre-merge runtime config, and the daemon parallelizes only on explicit
> `params.pp` / `params.tp` / `params.ep` (non-legacy loads are preflighted by
> `hipfire_loader::admit_path`; the legacy non-VL Qwen3.5 HFQ `pp>1` path is
> exempt and classified before admission). `HIPFIRE_PP` and the TypeScript/Bun CLI
> (`cli/index.ts`, `buildLoadMessage`) were deleted; the native Rust CLI
> forwards `serve --tp N` as `params.tp` and has no `--pp`/`--ep` flags.
> Keep as provenance; current semantics live in
> [`docs/multi-gpu.md`](../../multi-gpu.md) and
> [`docs/env-vars.md`](../../env-vars.md).

## Problem

hipfire's multi-GPU code paths — pipeline-parallel (PP, qwen35 arch 5/6) and
expert-parallel (EP / TP, deepseek4 arch 9 + minimax arch 10) — only run when
the engine sees ≥2 physical GPUs. On a single-GPU box (gfx1151 / halo) they are
unreachable, so they cannot be exercised, debugged, or gated locally.

We want a **debug env var that makes hipfire treat the single physical gfx1151
as N logical GPUs, all aliased to physical device 0**, so the multi-GPU paths
run end-to-end on one card — driven from the normal CLI.

## Key findings (from code exploration)

The device model, gates, and the single hook point are already well understood:

1. **`HIPFIRE_DEVICES=0,0` + `pp=2` already works today with zero engine
   changes.** `resolve_device_ids` (multi_gpu.rs:644) does no de-duplication and
   `Gpu::init_with_device` (dispatch.rs:453) only bounds-checks `id < count`, so
   the id list `[0,0]` builds two independent `Gpu` handles both bound to
   physical device 0, and the entire `*_multi` / EP path runs unmodified. The
   feature is about making this **ergonomic, discoverable, and reachable from the
   CLI** — not net-new capability.

2. **The only gate** stopping the default multi-GPU path on one card is
   device-id resolution: with `pp=2`/`tp=2` and `HIPFIRE_DEVICES` unset,
   `resolve_device_ids` returns `(0..n) = [0,1]`, and `Gpu::init_with_device(1)`
   errors `"device id 1 out of range (count=1)"`. No forward-pass code has a
   `device_count >= 2` assertion.

3. **`resolve_device_ids` is the single funnel** for both PP (`init_uniform` /
   `init_layers`) and TP/EP (`init_tp`). Aliasing there serves every multi-GPU
   mode.

4. **qwen35 has no TP path.** The `tp > 1` route (`load_model_ep`, daemon.rs
   ~5124) explicitly errors for anything except deepseek4 (9) and minimax (10):
   `"EP serving (tp=N) supports DeepSeek-V4 (9) and MiniMax-M2 (10); got arch_id
   {arch_id}"`. qwen35's only multi-GPU path is PP (`load_model_pp` →
   `*_multi`). So the default mode and qwen35 are on different code paths.

5. **The Bun/TS CLI does not plumb `pp`.** `buildLoadMessage` (cli/index.ts
   ~492) forwards only `tp` (from `HIPFIRE_TP`), never `pp`. Today PP is only
   reachable by hand-sending a raw JSON load message to the daemon.

## Chosen behavior

- **`HIPFIRE_EMULATE_GPUS=N`** (count-valued; unset = off). When set, the engine
  aliases every requested logical device id into the physical range by modulo,
  so on a 1-GPU box any multi-GPU request lands entirely on physical device 0.
- **Default mode is TP.** When the var is set and neither `pp` nor `tp` is
  explicitly requested, default `tp = N` (routes to `load_model_ep`, i.e.
  deepseek4 / minimax EP). This matches the primary multi-GPU mode in use.
- **PP is opt-in.** To exercise qwen35 PP, set `HIPFIRE_PP=N` (or `params.pp`)
  explicitly. PP and TP remain mutually exclusive (existing daemon guard).
- **Same physical device, N logical entries.** All aliased handles bind physical
  device 0; no logical/physical id split is introduced anywhere (both ids stay
  physically valid, = 0). This is the minimal, lowest-blast-radius approach.

### Rejected alternatives

- **Spoof device count in rdna-compute** (make `Gpu` report N and add a
  physical≠logical id split): much larger blast radius — touches
  `bind_thread`/`set_device`/memory pool/peer/rocBLAS in a lower crate. Overkill
  for debug emulation.
- **Just document `HIPFIRE_DEVICES=0,0`**: zero code, but not the discoverable
  debug var requested, easy to misuse (it's a real logical-id knob), requires
  matching the zero-count to the parallelism degree, and does nothing for the
  CLI-can't-reach-PP gap.

## Components

### 1. Config field (`crates/hipfire-runtime/src/config.rs`)
Add `emulate_gpus: Option<usize>`, read from `HIPFIRE_EMULATE_GPUS`, mirroring
the existing `devices` / `tp` fields. A value `< 2` (or unparseable) is treated
as off.

### 2. Device aliasing (`crates/hipfire-runtime/src/multi_gpu.rs`)
Extract a pure helper:
```rust
fn alias_ids(ids: &[i32], real_count: i32) -> Vec<i32>
    // maps each id to id.rem_euclid(real_count)
```
In `resolve_device_ids` (line 644): when `emulate_gpus` is set, query the real
physical device count once (`HipRuntime::load()?.device_count()`) and run the
resolved id list through `alias_ids`. This applies to the default `(0..n)` list
*and* to an explicit `HIPFIRE_DEVICES` list (so neither can reintroduce an
out-of-range id). On this box: `[0,1] → [0,0]`; on a 2-GPU box `emulate=4` →
`[0,1,0,1]`.

### 3. Default-mode selection (`crates/hipfire-runtime/examples/daemon.rs`, pp/tp parse ~1835)
After parsing `pp` and `tp` from the load message, apply precedence:
1. explicit `pp > 1` → PP (wins)
2. else explicit `tp > 1` → TP
3. else `emulate_gpus = Some(n)` with `n >= 2` → **default `tp = n`**

The existing `pp>1 && tp>1` mutual-exclusion guard (daemon.rs:1848) is preserved
and never triggered by this defaulting (only one branch fires).

### 4. CLI pp plumbing (`cli/index.ts`, `buildLoadMessage` ~510)
Mirror the existing `HIPFIRE_TP → params.tp` plumbing with `HIPFIRE_PP →
params.pp`. Closes the real gap that the CLI cannot request PP at all, and is the
mechanism by which PP debugging works from the CLI. The daemon child inherits the
shell env, so `HIPFIRE_EMULATE_GPUS` reaches both processes.

## Data flow

```
shell env (HIPFIRE_EMULATE_GPUS=2 [+ HIPFIRE_PP=2])
  → CLI spawns daemon (env inherited); buildLoadMessage sets params.tp/params.pp
  → daemon parses pp/tp; if neither set & emulate → tp=2   (or pp=2 if HIPFIRE_PP set)
  → load_model_ep(tp=2)  |  load_model_pp(pp=2)
  → Gpus::init_tp / init_uniform → resolve_device_ids(2) → alias_ids → [0,0]
  → two Gpu{device_id:0} handles, both physical device 0
  → EP: experts sharded e%2==r, dense replicated; all-reduce = same-device d2d
    PP: 64 layers split ~32/32 across bands; boundary_copy = same-device d2d
  → generate_ep / generate_multi drives decode
```

## VRAM

- **PP:** weights split by band, so total weight footprint on device 0 ≈ 1× the
  model (half + half); only per-device scratch/KV/DeltaNet-state duplicated.
- **TP/EP:** experts sharded, dense weights replicated per rank → ≈ 2× dense +
  1× experts. Fine for MoE models where experts dominate.

Both comfortably fit 128 GB UMA for the model sizes in use.

## Error handling / existing guards (unchanged)

- `pp>1` refuses DFlash (unless `HIPFIRE_PP_DFLASH=1`), CASK, PFlash, and
  non-{5,6} arch — all acceptable for a plain qwen35 PP test.
- `tp>1` refuses DFlash drafters; `load_model_ep` refuses arch ∉ {9,10}.
- Preflight arch-match + VRAM-delta pass trivially (identical card, ~0 delta).
- **Keep RCCL off for emulation** (`HIPFIRE_TP_USE_RCCL` unset): RCCL will not
  init two communicators on the same physical device. The default peer-direct
  all-reduce degrades to same-device d2d copies, which is the intended path.

## Testing & success criteria

### No-GPU (CI-runnable)
- Unit test for `alias_ids`:
  - `alias_ids(&[0,1], 1) == [0,0]`
  - `alias_ids(&[0,1,2,3], 2) == [0,1,0,1]`
  - `alias_ids(&[0,0], 1) == [0,0]` (idempotent)
- `cargo build --release --workspace` green.
- `bun` typecheck of the CLI change.

### GPU integration (validates the one make-or-break unknown: two
`Gpu::init_with_device(0)` handles coexisting on gfx1151 with independent
streams over the shared primary context)
1. **TP default:** deepseek4 or minimax `.mq4` + `HIPFIRE_EMULATE_GPUS=2` →
   coherent output; logs confirm EP engaged (2 ranks, expert sharding).
2. **PP explicit:** qwen35 `.mq4` + `HIPFIRE_EMULATE_GPUS=2 HIPFIRE_PP=2` →
   coherent output; logs confirm the 2-band split + boundary copies.

### Gate
- Routes forward-pass dispatch through `*_multi` / EP, so
  `./scripts/coherence-gate.sh` applies before commit.

**Definition of done:** (1) build + `alias_ids` unit test + CLI typecheck green
(no GPU); (2) both integration runs produce coherent output with logs showing the
multi-rank split; (3) coherence gate passes.

## Out of scope

- Real logical/physical id split (rejected Approach 2).
- TP-specific CLI ergonomics beyond forwarding `tp` (already present) and `pp`.
- Exposing `HIPFIRE_EMULATE_GPUS` as a supported (non-debug) product flag.
- RCCL-based emulation.
