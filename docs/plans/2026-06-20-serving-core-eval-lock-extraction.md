# Plan: `hipfire-serving-core` extraction + eval modularization + `hipfire-lock`

Status: **proposed** — 2026-06-20. Follows the daemon `main.rs` 12-module split
(18,290 → 5,006) and the E2 `ServingBackend`/`GenerateCtx` seam in
`hipfire-runtime::arch`.

## Goal

Let `hipfire-eval` and `hipfire-daemon` **share orchestration code** (model
load, generate/prefill/decode, sampling, sessions) instead of eval reaching the
daemon only over a spawned-subprocess JSONL boundary. Achieve it by extracting a
shared **library** — NOT by merging the two ~18k-line monoliths into one crate.
Also: break the 18.7k-line `hipfire-eval/src/lib.rs` into modules, and lift the
duplicated `flock(2)` logic (daemon singleton lock + CLI `gpu-lock`) into a
reusable crate.

Three independent workstreams; A is the keystone, B and C can land in any order.

---

## Current shape (verified)

- `hipfire-daemon` is **bin-only** (no `lib.rs`). The 12 modules from the split
  (`model`, `load`, `generate`, `generate_arch`, `generate_vl`, `qwen35_prefill`,
  `qwen35_decode`, `session`, `memory`, `output_filter`, `events`, `dummy`) are
  `pub(crate)` inside the bin crate → **not linkable by eval**.
- `hipfire-eval` has **no dep on `hipfire-daemon`**; it spawns the daemon binary
  via `hipfire-daemon-adapter` (`DaemonEngine::spawn(bin)` → JSONL over stdio).
  `hipfire-eval/src/lib.rs` is **18,739 lines**; `main.rs` is a 10-line shim.
- The serving seam (`ServingBackend`, `SimpleAr`, `GenerateCtx { sink: &mut dyn
  Write, .. }`, `run_simple_ar`, `decode_loop`) already lives in
  `hipfire-runtime::arch` — a crate eval already depends on.
- Two `flock(2)` implementations: `acquire_daemon_lock()` in daemon `main.rs`
  (singleton `~/.hipfire/daemon.pid`) and `crates/hipfire-cli/src/commands/
  gpu_lock.rs` (323 lines: acquire/poll/detached-holder/release/status). The
  `/tmp/hipfire-resource-locks` leases referenced in AGENTS.md live in
  `hipfire-daemon-adapter` (client-side wait), not the daemon proper.

Dependency fact that forces a *new* crate (not folding into `hipfire-runtime`):
the orchestration layer depends on **every arch crate** (qwen35, llama, qwen2,
deepseek4, minimax, lfm2moe, dots-ocr, qwen35-vl, gemma3, gemma3-vl), and those
arch crates depend on `hipfire-runtime`. So orchestration must sit **above** them
in a new crate to avoid a cycle.

---

## Workstream A — `hipfire-serving-core` (the keystone)

New library crate `crates/hipfire-serving-core/`. Both `hipfire-daemon` (bin) and
`hipfire-eval` depend on it. Dep edges: `hipfire-serving-core` → all arch crates
+ `hipfire-runtime` + `hipfire-generate` + `hipfire-model` + `hipfire-prompt` +
`hipfire-state` + `hipfire-evidence` + `rdna-compute`/`hip-bridge`.

### What moves (daemon bin → serving-core lib)

Promote `pub(crate)` → `pub` as it crosses the crate boundary:

| Module | Contents | Notes |
|--------|----------|-------|
| `model` | `LoadedModel` + fields, `Eviction`, `CaskConfig`, `DflashState`, `DdtreeState`, `effective_raw`, `RAW_OVERRIDE` | `RAW_OVERRIDE` thread-local is a request-loop wart — see "Seam work" |
| `load` | `load_model{,_safetensors,_pp}`, `unload_model`, `load_dflash_state`, config helpers | |
| `session` | `Qwen35RequestSessionState`, arena dispatch, worker lifecycle | |
| `memory` | byte accounting | |
| `output_filter` | prompt/stop filtering, attractor guards, loop-guard ctor | |
| `qwen35_prefill` / `qwen35_decode` | batched prefill/decode + validators | |
| `generate` / `generate_arch` / `generate_vl` | the AR + per-arch + VL generate paths | depend on the sink seam below |
| `dummy` | GPU-free backend | useful to eval's mock executor too |
| `events` | JSONL envelope builders | **split**: envelope construction is shareable; raw `Stdout` writes stay daemon-side (see seam) |

### What stays in `hipfire-daemon` (bin)

`main()`, the request-dispatch loop (the big match on message `type`), stdin
reader / stdout writer, the daemon singleton lock (→ workstream C), and the
`mod`/`use` wiring. The bin becomes a **thin protocol shell** over the core:
parse JSONL → build a `GenerateCtx` with `sink = stdout` → call into the core →
stream events back. Target: daemon bin ≈ 1,500–2,500 lines.

### Seam work (the hard, valuable part)

The generate paths currently couple to the daemon runtime in three ways that
block in-process reuse by eval:

1. **Output:** they write JSONL straight to `&mut Stdout`. → Route all visible
   output through `GenerateCtx::sink: &mut dyn Write` (already exists for the
   qwen2/gemma3/gemma3-vl `ServingBackend` archs; **extend to the qwen35 fast
   paths** `generate`/`generate_mtp`/`generate_dflash`/`generate_multi`). This is
   E2 phases P4–P6 in `2026-06-19-daemon-family-seam.md`.
2. **Thread-locals:** `RAW_OVERRIDE` is set by the request loop and read by
   `effective_raw`. → Fold into `GenerateCtx` (an explicit `raw: Option<bool>`)
   so callers pass it rather than mutate global state.
3. **Event vocabulary:** `events.rs` mixes envelope construction with
   stdout I/O. → Keep envelope builders (`serde_json::json!` → `Value`) in the
   core; the daemon bin owns the `Stdout` flush.

Once the qwen35 paths are sink-abstracted, **eval drives the core in-process**
via `ServingBackend::serve(gpu, tok, &mut ctx)` with its own sink — no subprocess,
no JSONL — for the rows that want it (load + direct generate). The
**daemon-battery rows keep the subprocess+JSONL path on purpose** (they test the
shipped binary + wire protocol end to end).

### Sequencing (A)

1. **A0** — scaffold `crates/hipfire-serving-core`, move the modules verbatim,
   flip the cross-crate-public surface `pub(crate)`→`pub`. Daemon bin gains
   `hipfire-serving-core = { path = .. }` and `use hipfire_serving_core::*`.
   No behavior change; daemon tests + coherence/speed gates green. (Mechanical;
   mirrors the split rhythm — the per-module `pub(crate)` cascades are already
   mapped from the split.)
2. **A1** — sink/thread-local seam: extend `GenerateCtx` (sink already there; add
   `raw`), route qwen35 `generate*` through it. Coherence + DFlash gates.
3. **A2** — eval depends on `hipfire-serving-core`; add an in-process executor
   path (`--executor core`?) that calls `ServingBackend`/`load_model` directly,
   replacing the bits of the `direct`/`examples` executors that shell out where
   in-process is cheaper. Keep `--executor daemon` (subprocess) intact.

A0 is shippable alone and immediately lets eval link `load_model` + the
already-seamed archs.

---

## Workstream B — modularize `hipfire-eval`

Split `hipfire-eval/src/lib.rs` (18,739 lines) the same way we did the daemon:
extract → `pub(crate)`/`pub` → `mod`/`use` → clippy `--all-targets` + tests +
commit, one module per commit. Proposed modules (boundaries from the item map):

| Module | Contents |
|--------|----------|
| `config` | `EvalTier`/`TierBudget`/`BatteryId`/`SuiteId`/`DflashMode`/`ProfileMode`/`EvalExecutorMode`/`EvalCacheMode`/`EvalConfig`, `parse_args_from`, `usage`, `version_report`, arg-parse helpers, `default_batteries/suites/output_dir/*cache` |
| `result` | `EvalResult`, `EvalManifest`, `DatasetManifestEntry`, `Comparison*`, `Metric*`, `Admission*`, `EvalContext`, `eval_status_str` |
| `datasets` | `resolve_datasets`, `builtin_dataset_entry`, `fetch_dataset`, `FetchedDataset`, dataset provenance + barrage prompt artifacts |
| `host_profile` | `collect_host_profile`, `collect_default_host_profile`, `run_host_capability_profile_anchor`, `HostProfileOverrides` |
| `rocprof` | `run_rocprof_speed_anchor`, `resolve_rocprofv3_bin`, `RocprofKernelStats`, CSV parse, profile evidence |
| `evidence` | `write_evidence_artifacts`, `run_provenance{,_value}`, `run_metadata_artifact_value`, dataset-provenance metrics |
| `quality` | `quality_json_rows`, `kld_reference_rows`, `load_quality_json_rows` |
| `performance` | `performance_json_rows`, `load_performance_json_rows` |
| `executor/mock` | `mock_battery_rows`, `mock_metric_family_rows`, `mock_barrage_rows` |
| `executor/daemon` | `daemon_battery_rows`, `run_daemon_*` (async + sync), session helpers, skip/failure rows (uses `hipfire-daemon-adapter`; later optionally `hipfire-serving-core` in-process) |
| `executor/examples` | `examples_battery_rows` + the `run_examples_*` family (coherence/profile/longctx/calibrate/perplexity/qwen35-speed/agentic/runtime/dflash/pflash) |
| `executor/direct` | `direct_battery_rows` (prime candidate to call `hipfire-serving-core` in-process post-A2) |
| `run` | `run_eval` top driver, `run_passive_profile_collectors` |

`main.rs` stays a shim. ~13 modules; lib.rs → a thin `pub mod` + re-export root.
Independent of A (no `hipfire-serving-core` dep needed), so it can land first to
de-risk and shrink the diff that A2 later touches.

---

## Workstream C — `hipfire-lock` (reusable flock)

### Inventory (the impls to unify — verified 2026-06-20)

| # | Impl | Lockfile | Path env | Notes |
|---|------|----------|----------|-------|
| 1 | `hipfire-daemon/main.rs::acquire_daemon_lock` | `~/.hipfire/daemon.pid` | — | singleton, `LOCK_EX\|LOCK_NB`, fatal-on-busy |
| 2 | `hipfire-cli/commands/gpu_lock.rs` (Rust) | `/tmp/hipfire-gpu.lock` | `HIPFIRE_GPU_LOCKFILE` | **the real GPU-mutex flock**: acquire/release/status/hold, detached `setsid` holder |
| 4 | daemon **resource-lock leases** (`hipfire-daemon-adapter::try_acquire_resource_lock`) | `/tmp/hipfire-resource-locks/<res>.lock/` | `HIPFIRE_RESOURCE_LOCK_DIR` (`_WAIT_MS` to wait) | per-resource GPU/NPU/CPU leases before HIP init — **NOT flock**: atomic `mkdir` + `owner.json` pidfile with stale-reclaim |

**Correction (verified):** `scripts/gpu-lock.sh` is **NOT a separate flock impl** —
its `gpu_acquire/release/status` shell out to `hipfire gpu-lock … --watch-pid $$`
(impl #2). It's a bash ergonomics layer (reentrancy guard via
`HIPFIRE_GPU_LOCK_OWNER`, loud contention/stale warnings, trap-friendly funcs)
over the one Rust CLI. So there are exactly **three** crates that hand-roll
`flock`: #1, #2, #4 — and *those* collapse into `hipfire-lock`. `gpu-lock.sh` and
its **41** `source`ing consumers (`tests/*.sh`, `scripts/*.sh`, `benchmarks/*.sh`)
stay **untouched** (they already require the `hipfire` binary, since gpu-lock.sh
calls it). `tests/daemon_mutex.sh` tests #1. **No Python lock exists** today.

**Bug to fix in consolidation (now diagnostic-only):** gpu-lock.sh reads
`HIPFIRE_GPU_LOCK_FILE` only to *print* the holder/stale path in its warnings,
while the actual lock path comes from the Rust CLI's `HIPFIRE_GPU_LOCKFILE` — so
overriding the path makes gpu-lock.sh's warning text point at the wrong file
(the locking itself stays correct, since it delegates). `hipfire-lock` owns ONE
path + ONE env-var name + ONE metadata-line format; #1/#2 route through it, and
gpu-lock.sh's warning reads the same env var.

**Scope:** only #1 and #2 are `flock`-based → they consolidate onto
`FlockGuard`. #4 is a *different* primitive (atomic `mkdir` + pidfile, with
stale-owner reclaim) living in the adapter for the client side; it's cohesive
and correct as-is. Leave it; optionally relocate it into `hipfire-lock` as a
second `dir_lock` module in a follow-up so all lock primitives share one home —
but it is **not** part of the flock consolidation.

New tiny crate `crates/hipfire-lock/` (unix `flock(2)` primitive). Module shape:

- `FlockGuard` — `open(path)` → `try_lock()` (`LOCK_EX|LOCK_NB`) /
  `lock_blocking(poll_interval, timeout, on_busy: impl FnMut(&str))` /
  unlock-on-drop (fd close). Holder-metadata read/write that preserves the lock
  (truncate + write via the held fd).
- `probe(path) -> LockState::{Free, Busy(holder)}` — the non-blocking `status`.

Consumers (the three flock crates):
- **daemon** `acquire_daemon_lock()` → `FlockGuard::open(~/.hipfire/daemon.pid)
  .try_lock()` with the existing fatal-on-busy message. (Moves out of the bin;
  the bin keeps only the call + the "already running (PID N)" error.)
- **CLI `gpu-lock`** → the acquire/poll/release/status logic uses `FlockGuard`.
  The **detached-holder + `setsid` dance stays gpu-lock-specific** (it's a
  process-lifetime design, not a lock primitive) but can live in a
  `hipfire-lock::holder` submodule if we want it shared with future callers.
  `scripts/gpu-lock.sh` keeps calling `hipfire gpu-lock` unchanged.
- (optional) the `/tmp/hipfire-resource-locks` leases (#4) adopt the same
  primitive later (keep the lease/wait protocol on top).

### Python / pytorch access — shell out, don't bind or reimpl

`flock(2)` is a kernel primitive: any process that opens the same path and flocks
the same inode shares the mutex regardless of language. So the contract (path +
flock + metadata line), not the code, is what interoperates.

- **Rejected — pyo3 binding:** adds maturin/cdylib builds + per-platform wheels +
  version coupling, all to call the same syscall. No correctness benefit.
- **Rejected as default — native `fcntl.flock` reimpl:** correct and FFI-free, but
  a 4th source of truth to keep in sync, and it would have to re-derive the
  detached-holder/watch-pid logic to match #2/#3.
- **Chosen — a ~15-line Python context manager that shells out to
  `hipfire gpu-lock` (or `gpu-lock.sh`)** with `--watch-pid <os.getpid()>`:
  ```python
  with gpu_lock("imatrix-calib"):   # subprocess acquire; holder watches this pid
      train()                       # auto-released on process exit
  ```
  One source of truth (the unified Rust/bash impl), full holder-model fidelity,
  subprocess cost irrelevant (acquire once per run). Reimplement natively only if
  a future caller needs zero-subprocess acquisition in a hot loop (GPU acquire
  isn't one).

Independent of A and B. Smallest workstream; good warm-up. Ship order within C:
(1) `hipfire-lock` crate + unify #1/#2 + fix the env-var split; (2) point
`gpu-lock.sh` at the same contract (or have it call `hipfire gpu-lock`); (3) the
Python `gpu_lock` context-manager wrapper; (4) a cross-language interop test
(rust/bash holds → python probes BUSY, and vice-versa) to guard contract drift.

---

## Risks / watch-items

- **`pub(crate)` → `pub` cascade (A0):** mechanical but wide; the compiler drives
  it (same as the split). Watch the `cargo fix` test-module trap: it strips
  imports the daemon/eval `#[cfg(test)]` modules reach via `super::*` — scope
  those into the test module, don't let `cargo fix` delete them.
- **HOTSPOT gates:** `generate_arch.rs` / any `*arch.rs` filename triggers the
  coherence+speed gates on commit; serving-core commits touching those run the
  full battery. Quiet box only (UMA APU perf-gate caveat).
- **Wire-contract drift (orthogonal but worth doing):** the daemon hand-rolls
  `serde_json::Value` while the adapter uses typed `hipfire-daemon-protocol`.
  Independent of this plan, but once the bin is a thin shell, having it consume
  `hipfire-daemon-protocol` types makes the contract compile-checked on both
  sides. Optional follow-up.
- **Don't regress eval's daemon battery:** A2 adds an in-process path; it must
  not replace the subprocess+JSONL daemon-battery rows, which exist to test the
  shipped binary.

## Suggested order

C (warm-up, tiny) → B (de-risk + shrink eval) → A0 (scaffold + move, mechanical)
→ A1 (sink/thread-local seam) → A2 (eval in-process executor). A0 alone already
delivers the "shared load/generate code" win.
