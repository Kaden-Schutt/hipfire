# HIPFIRE_EMULATE_GPUS Dual-GPU Emulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a debug env var `HIPFIRE_EMULATE_GPUS=N` that makes hipfire treat the single physical gfx1151 as N logical GPUs (all aliased to device 0), so the multi-GPU paths (TP/EP by default, PP opt-in) run on one card, driven from the CLI.

**Architecture:** Aliasing lives in one funnel — `resolve_device_ids` maps every requested logical device id into the physical range by euclidean remainder. Default parallelism mode is decided by a pure `resolve_parallelism` helper (TP when neither pp nor tp is requested). The CLI gains `HIPFIRE_PP` plumbing so PP is reachable. No logical/physical id split is introduced; both aliased ids stay physically valid (= 0).

**Tech Stack:** Rust (hipfire-runtime crate, hip-bridge FFI), TypeScript/Bun CLI.

## Global Constraints

- **No `cargo fmt`** — format only files you edit, per-file: `rustfmt --edition 2021 --config skip_children=true <file>`. `scripts/fmt-changed.sh` is also unsafe on a branch. (CLAUDE.md)
- **Coherence gate** — any change routing the forward pass through `*_multi`/EP must pass `./scripts/coherence-gate.sh` before commit (Task 5 covers this; earlier tasks are GPU-free).
- **Conventional commits**; commit each task separately; end commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Daemon build command:** `cargo build --release --example daemon --features deltanet -p hipfire-runtime`.
- **GPU lock (Task 5 only):** `source scripts/gpu-lock.sh && gpu_acquire "parallel-expansion" && gpu_release`. Never `rm` the lockfile.
- **Value semantics:** `HIPFIRE_EMULATE_GPUS` < 2 or unparseable = off (`None`).

---

### Task 1: Config field `emulate_gpus`

**Files:**
- Modify: `crates/hipfire-runtime/src/config.rs` (struct ~line 34, `from_env` ~line 102, tests mod ~line 119)

**Interfaces:**
- Produces: `RuntimeConfig.emulate_gpus: Option<usize>` — `Some(n)` with `n >= 2` when the env var is set to a valid count, else `None`. Consumed by Tasks 2 and 3.

- [ ] **Step 1: Write the failing test**

Add inside the existing `#[cfg(test)] mod tests` block in `config.rs` (after the `normalize_prompt_accepts_no_as_false` test):

```rust
    #[test]
    fn emulate_gpus_parses_and_filters() {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("HIPFIRE_EMULATE_GPUS").ok();

        std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
        assert_eq!(RuntimeConfig::from_env().emulate_gpus, Some(2));

        std::env::set_var("HIPFIRE_EMULATE_GPUS", "1"); // < 2 = off
        assert_eq!(RuntimeConfig::from_env().emulate_gpus, None);

        std::env::set_var("HIPFIRE_EMULATE_GPUS", "abc"); // unparseable = off
        assert_eq!(RuntimeConfig::from_env().emulate_gpus, None);

        std::env::remove_var("HIPFIRE_EMULATE_GPUS");
        assert_eq!(RuntimeConfig::from_env().emulate_gpus, None);

        match prev {
            Some(v) => std::env::set_var("HIPFIRE_EMULATE_GPUS", v),
            None => std::env::remove_var("HIPFIRE_EMULATE_GPUS"),
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p hipfire-runtime --lib config::tests::emulate_gpus_parses_and_filters`
Expected: FAIL — compile error `no field 'emulate_gpus' on type 'RuntimeConfig'`.

- [ ] **Step 3: Add the struct field**

In `config.rs`, in `pub struct RuntimeConfig`, immediately after `pub devices: Option<String>,` (line 34):

```rust
    /// Debug: `HIPFIRE_EMULATE_GPUS=N` makes the engine treat the single
    /// physical GPU as N logical devices (all aliased to device 0), so the
    /// multi-GPU (PP / EP) paths run on one card. Values < 2 = off. See
    /// docs/superpowers/specs/2026-07-03-emulate-gpus-dual-gpu-design.md.
    pub emulate_gpus: Option<usize>,
```

- [ ] **Step 4: Parse it in `from_env`**

In `config.rs` `from_env`, immediately after the `devices: std::env::var("HIPFIRE_DEVICES").ok(),` line (line 102):

```rust
            emulate_gpus: std::env::var("HIPFIRE_EMULATE_GPUS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n >= 2),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p hipfire-runtime --lib config::tests::emulate_gpus_parses_and_filters`
Expected: PASS.

- [ ] **Step 6: Format + commit**

```bash
rustfmt --edition 2021 --config skip_children=true crates/hipfire-runtime/src/config.rs
git add crates/hipfire-runtime/src/config.rs
git commit -m "feat(config): add HIPFIRE_EMULATE_GPUS field (< 2 = off)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `alias_ids` helper + `resolve_device_ids` hook

**Files:**
- Modify: `crates/hipfire-runtime/src/multi_gpu.rs` (import block ~line 23, `resolve_device_ids` line 644, add `alias_ids` + a tests module)

**Interfaces:**
- Consumes: `crate::config::get().emulate_gpus` (Task 1), `hip_bridge::HipRuntime::{load, device_count}`.
- Produces: `fn alias_ids(ids: &[i32], real_count: i32) -> Vec<i32>` (module-private) and the aliasing behavior in `resolve_device_ids`.

- [ ] **Step 1: Write the failing test**

Add a tests module at the **end** of `multi_gpu.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::alias_ids;

    #[test]
    fn alias_ids_maps_into_physical_range() {
        assert_eq!(alias_ids(&[0, 1], 1), vec![0, 0]); // 2 logical -> 1 physical
        assert_eq!(alias_ids(&[0, 1, 2, 3], 2), vec![0, 1, 0, 1]); // 4 -> 2
        assert_eq!(alias_ids(&[0, 0], 1), vec![0, 0]); // idempotent
        assert_eq!(alias_ids(&[0, 1], 2), vec![0, 1]); // no-op when in range
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p hipfire-runtime --lib multi_gpu::tests::alias_ids_maps_into_physical_range`
Expected: FAIL — compile error `cannot find function 'alias_ids'`.

- [ ] **Step 3: Add the `alias_ids` helper**

In `multi_gpu.rs`, immediately **before** `fn resolve_device_ids` (line 644):

```rust
/// Map each requested logical device id into the physical range
/// `[0, real_count)` by euclidean remainder. Used by `HIPFIRE_EMULATE_GPUS`
/// to alias N logical devices onto the (fewer) physical devices — e.g.
/// `[0, 1] -> [0, 0]` on a 1-GPU box, `[0, 1, 2, 3] -> [0, 1, 0, 1]` on a
/// 2-GPU box. A non-positive `real_count` is left untouched (no physical
/// devices to alias onto — the caller will surface the real error).
fn alias_ids(ids: &[i32], real_count: i32) -> Vec<i32> {
    if real_count <= 0 {
        return ids.to_vec();
    }
    ids.iter().map(|&id| id.rem_euclid(real_count)).collect()
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p hipfire-runtime --lib multi_gpu::tests::alias_ids_maps_into_physical_range`
Expected: PASS.

- [ ] **Step 5: Import `HipRuntime`**

In `multi_gpu.rs`, extend the `use hip_bridge::{...}` block (line 23) to include `HipRuntime`:

```rust
use hip_bridge::{
    DeviceBuffer, Event, HipError, HipResult, HipRuntime, RcclComms,
    HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED, HIP_ERROR_PEER_ACCESS_UNSUPPORTED,
};
```

- [ ] **Step 6: Wire aliasing into `resolve_device_ids`**

Replace the entire body of `fn resolve_device_ids` (lines 644-665) with:

```rust
fn resolve_device_ids(n_devices: usize) -> HipResult<Vec<i32>> {
    let ids: Vec<i32> = if let Some(ref s) = crate::config::get().devices {
        let parsed: Vec<i32> = s
            .split(',')
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .map(|p| p.parse::<i32>())
            .collect::<Result<_, _>>()
            .map_err(|e| HipError::new(0, &format!("HIPFIRE_DEVICES parse: {e}")))?;
        if parsed.len() < n_devices {
            return Err(HipError::new(
                0,
                &format!(
                    "HIPFIRE_DEVICES has {} ids but n_devices = {n_devices}",
                    parsed.len(),
                ),
            ));
        }
        parsed[..n_devices].to_vec()
    } else {
        (0..n_devices as i32).collect()
    };

    // Debug dual-GPU emulation: alias every logical id into the physical range
    // so a single card can serve an N-way PP/EP load. Applies to both the
    // explicit HIPFIRE_DEVICES list and the default 0..n so neither can leave
    // an out-of-range id. See config::emulate_gpus.
    if crate::config::get().emulate_gpus.is_some() {
        let real = HipRuntime::load()?.device_count()?;
        return Ok(alias_ids(&ids, real));
    }
    Ok(ids)
}
```

- [ ] **Step 7: Build to verify compilation**

Run: `cargo build -p hipfire-runtime`
Expected: builds clean (pre-existing warnings only).

- [ ] **Step 8: Format + commit**

```bash
rustfmt --edition 2021 --config skip_children=true crates/hipfire-runtime/src/multi_gpu.rs
git add crates/hipfire-runtime/src/multi_gpu.rs
git commit -m "feat(multi-gpu): alias logical device ids under HIPFIRE_EMULATE_GPUS

resolve_device_ids maps every requested id into the physical range by
rem_euclid, so [0,1] -> [0,0] on a single card. Serves both PP and EP.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `resolve_parallelism` helper + daemon default-mode wiring

**Files:**
- Modify: `crates/hipfire-runtime/src/config.rs` (add pub fn + test)
- Modify: `crates/hipfire-runtime/examples/daemon.rs` (after tp parse ~line 1847)

**Interfaces:**
- Consumes: `RuntimeConfig.emulate_gpus` (Task 1).
- Produces: `pub fn resolve_parallelism(pp: usize, tp: usize, emulate_gpus: Option<usize>) -> (usize, usize)` — returns `(pp, n)` (default TP) only when both `pp == 1` and `tp == 1` and `emulate_gpus == Some(n>=2)`; otherwise returns `(pp, tp)` unchanged.

- [ ] **Step 1: Write the failing test**

Add inside the `#[cfg(test)] mod tests` block in `config.rs`:

```rust
    #[test]
    fn resolve_parallelism_defaults_tp_only_when_unset() {
        // Neither requested + emulate -> default TP.
        assert_eq!(super::resolve_parallelism(1, 1, Some(2)), (1, 2));
        // Explicit pp wins; no tp default.
        assert_eq!(super::resolve_parallelism(2, 1, Some(2)), (2, 1));
        // Explicit tp wins.
        assert_eq!(super::resolve_parallelism(1, 4, Some(2)), (1, 4));
        // No emulate -> unchanged.
        assert_eq!(super::resolve_parallelism(1, 1, None), (1, 1));
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p hipfire-runtime --lib config::tests::resolve_parallelism_defaults_tp_only_when_unset`
Expected: FAIL — compile error `cannot find function 'resolve_parallelism'`.

- [ ] **Step 3: Add the helper**

In `config.rs`, at module level (e.g. immediately before the `#[cfg(test)]` line, line 119):

```rust
/// Decide the effective `(pp, tp)` parallelism degrees for a load. An
/// explicitly-requested `pp > 1` or `tp > 1` (from the load message) always
/// wins. Only when neither is requested does `HIPFIRE_EMULATE_GPUS` default
/// the mode — to TP (EP), the primary multi-GPU path; PP stays opt-in via an
/// explicit `pp`. The pp/tp mutual-exclusion check remains with the caller.
pub fn resolve_parallelism(pp: usize, tp: usize, emulate_gpus: Option<usize>) -> (usize, usize) {
    if pp == 1 && tp == 1 {
        if let Some(n) = emulate_gpus {
            if n >= 2 {
                return (pp, n);
            }
        }
    }
    (pp, tp)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p hipfire-runtime --lib config::tests::resolve_parallelism_defaults_tp_only_when_unset`
Expected: PASS.

- [ ] **Step 5: Wire it into the daemon**

In `crates/hipfire-runtime/examples/daemon.rs`, immediately **after** the `tp` parse block (the `let tp = ... .unwrap_or(1) as usize;` ending at line 1847) and **before** the `if tp > 1 && pp > 1` mutual-exclusion check (line 1848), insert:

```rust
                // Debug dual-GPU emulation (HIPFIRE_EMULATE_GPUS): when neither
                // pp nor tp is explicitly requested, default the mode to TP so
                // the EP path runs on a single card. PP stays opt-in via pp.
                // Shadows pp/tp; the mutual-exclusion check below still holds
                // because defaulting never sets both > 1.
                let (pp, tp) = hipfire_runtime::config::resolve_parallelism(
                    pp,
                    tp,
                    hipfire_runtime::config::get().emulate_gpus,
                );
```

- [ ] **Step 6: Build the daemon to verify compilation**

Run: `cargo build --release --example daemon --features deltanet -p hipfire-runtime`
Expected: builds clean (pre-existing warnings only).

- [ ] **Step 7: Format + commit**

```bash
rustfmt --edition 2021 --config skip_children=true crates/hipfire-runtime/src/config.rs crates/hipfire-runtime/examples/daemon.rs
git add crates/hipfire-runtime/src/config.rs crates/hipfire-runtime/examples/daemon.rs
git commit -m "feat(daemon): default to TP under HIPFIRE_EMULATE_GPUS when pp/tp unset

resolve_parallelism defaults the mode to TP (EP) when the emulation var is
set and neither pp nor tp is explicitly requested. PP stays opt-in.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: CLI `HIPFIRE_PP` plumbing

**Files:**
- Modify: `cli/index.ts` (`buildLoadMessage`, after the `tp` block ~line 512)

**Interfaces:**
- Produces: `params.pp` forwarded to the daemon when `HIPFIRE_PP > 1`. Mirrors the existing `HIPFIRE_TP → params.tp` plumbing. Makes PP reachable from the CLI (previously impossible).

- [ ] **Step 1: Add the plumbing**

In `cli/index.ts`, in `buildLoadMessage`, immediately **after** the existing `tp` block (the `{ const tp = parseInt(...); if (...) params.tp = tp; }` block ending at line 512), insert:

```typescript
  // Pipeline-parallel degree (PP). `HIPFIRE_PP=N` routes qwen35 (dense/MoE)
  // through the daemon's load_model_pp (layers banded across N ranks).
  // Forwarded only when > 1 so single-GPU loads stay byte-identical; mutually
  // exclusive with tp. Primary use: exercising PP under HIPFIRE_EMULATE_GPUS.
  {
    const pp = parseInt(process.env.HIPFIRE_PP ?? "1", 10);
    if (Number.isInteger(pp) && pp > 1) params.pp = pp;
  }
```

- [ ] **Step 2: Typecheck**

Run: `cd cli && bun run typecheck`
Expected: PASS (`tsc --noEmit`, no errors).

- [ ] **Step 3: Commit**

```bash
git add cli/index.ts
git commit -m "feat(cli): plumb HIPFIRE_PP into load message params.pp

Mirrors HIPFIRE_TP plumbing; makes the daemon's PP path reachable from the
CLI (used to exercise qwen35 PP under HIPFIRE_EMULATE_GPUS).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: GPU integration validation + coherence gate

**Files:** none modified — this task validates the assembled feature and the one make-or-break runtime unknown (two `Gpu::init_with_device(0)` handles coexisting on gfx1151 with independent streams over the shared primary context).

**Interfaces:** consumes the full daemon + CLI built in Tasks 1-4.

> **Note:** This task requires the GPU and a model on disk. Model filenames are fleet-specific — locate a deepseek4 or minimax `.mq4` (TP/EP) and a qwen35 `.mq4` (PP) under the models dir (`~/.hipfire/models/` or `$MODELS_DIR`). If no ds4/minimax model is available, run only the qwen35 PP arm and record that TP was not validated locally.

- [ ] **Step 1: Build daemon + CLI fresh**

Run:
```bash
cargo build --release --example daemon --example coherence_probe --features deltanet -p hipfire-runtime
```
Expected: builds clean.

- [ ] **Step 2: Acquire the GPU lock**

Run: `source scripts/gpu-lock.sh && gpu_acquire "parallel-expansion"`
Expected: lock acquired (or waits for a live holder — do not `rm` the lockfile).

- [ ] **Step 3: Validate TP default (EP path) — the make-or-break check**

With `M` = path to a deepseek4 or minimax `.mq4`, and RCCL left OFF (do not set `HIPFIRE_TP_USE_RCCL`):
```bash
HIPFIRE_EMULATE_GPUS=2 ./target/release/examples/coherence_probe \
    --model "$M" --prompt "What is the capital of France?" \
    --max-tokens 64 --temperature 0.0
```
Expected: coherent answer (names Paris); no panic; daemon log shows the EP path with 2 ranks / expert sharding. If ROCm rejects two device-0 handles, capture the exact error — that is the pivotal finding and blocks the feature; stop and report before proceeding.

- [ ] **Step 4: Validate PP explicit (qwen35 bands)**

With `Q` = path to a qwen35 dense/MoE `.mq4`:
```bash
HIPFIRE_EMULATE_GPUS=2 HIPFIRE_PP=2 ./target/release/examples/coherence_probe \
    --model "$Q" --prompt "What is the capital of France?" \
    --max-tokens 64 --temperature 0.0
```
Expected: coherent answer; daemon log shows `pp=2`, a ~half/half band split across 2 devices, and boundary copies at the seam.

- [ ] **Step 5: Run the coherence gate**

Run: `./scripts/coherence-gate.sh`
Expected: report shows each model fluent, on-topic, not looping; exit 0. (Set the gate's model env to the qwen35 model exercised above if it does not auto-discover it.)

- [ ] **Step 6: Release the GPU lock**

Run: `gpu_release`

- [ ] **Step 7: Record results**

Append the two probe outcomes (coherent? which paths engaged? any ROCm coexistence error?) and the gate status to the spec's validation section, or a short note under `.agent-memory/` via `scripts/mem.sh remember`. No code commit unless a fix was needed.

---

## Self-Review

**Spec coverage:**
- Config field (spec §Components 1) → Task 1. ✓
- Device aliasing / `alias_ids` + `resolve_device_ids` (spec §Components 2) → Task 2. ✓
- Default-mode = TP (spec §Components 3) → Task 3. ✓
- CLI pp plumbing (spec §Components 4) → Task 4. ✓
- No-GPU unit tests: `emulate_gpus` parse (T1), `alias_ids` (T2), `resolve_parallelism` (T3). ✓
- GPU integration TP + PP + coherence gate + make-or-break coexistence check (spec §Testing) → Task 5. ✓
- RCCL-off guidance (spec §Error handling) → Task 5 Step 3. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. Model paths in Task 5 are intentionally fleet-specific (`$M`/`$Q`) with a documented locate step — not a placeholder in the code sense.

**Type consistency:** `emulate_gpus: Option<usize>` defined in T1, consumed verbatim in T2 (`crate::config::get().emulate_gpus`) and T3 (`resolve_parallelism(.., emulate_gpus)`). `alias_ids(&[i32], i32) -> Vec<i32>` defined and tested identically in T2. `resolve_parallelism(usize, usize, Option<usize>) -> (usize, usize)` defined in T3, called with matching arg order/types in the daemon. Consistent. ✓
