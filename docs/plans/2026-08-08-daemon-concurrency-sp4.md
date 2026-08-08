# SP4 — Daemon Concurrency and Admission Control: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the multi-slot machinery reachable by real clients — several concurrent requests, each streaming its own tokens, on one GPU, with the 32 GB budget enforced rather than documented.

**Architecture:** A concurrent path *alongside* the existing single-session daemon loop, not a rewrite. A `SessionTable` maps clients to slots; an `AdmissionController` decides who gets in and with how much context; the existing sequential path is untouched when only one request is in flight.

**Tech Stack:** Rust (`hipfire-runtime`), JSONL over stdin as today, SP1–SP3 machinery underneath.

**Spec:** `docs/specs/2026-08-08-daemon-concurrency-sp4.md`. Read it first.

## Global Constraints

- **Branch:** `feat/batched-attn-impl`, worktree `~/repos/hipfire-batchattn-impl`.
- **`--features deltanet` (plus `arch-qwen35` for `hipfire-runtime`) on every cargo command**; scope to `-p <crate>`, never the whole workspace.
- **Do NOT rewrite `crates/hipfire-runtime/examples/daemon.rs`.** It is ~14,000 lines, single-session, and it is what the user runs daily. Add a concurrent path alongside it. Breaking the working single-user daemon is a far worse outcome than duplication — the same reasoning that kept SP3 out of `forward_prefill_batch_with_pbs_opts`.
- **Admission control IS the production memory gate.** In the harnesses that job belongs to `preflight_alloc`; in the daemon it belongs here. Getting it wrong does not fail a test, it takes down the desktop.
- **MEMORY — binding and measured.**
  - **The cgroup does NOT contain amdgpu GTT.** A gated run still invoked the *global* OOM killer and killed the user's Slack and three steamwebhelper processes. `MemoryMax` bounds host RSS only.
  - Run every GPU harness through `./scripts/run-bounded.sh`; it refuses unless `MemAvailable >= cap + 10 GiB`, which is the **primary** protection.
  - **Never run a GPU harness while a serve process holds a model.** Check `pgrep -f 'hipfire/bin/daemon'`. One resident model takes MemAvailable from ~58 GiB to ~19 GiB.
  - A live `free` shows nothing after the fact — check `journalctl -k | grep -E 'Out of memory|CONSTRAINT_NONE'`.
- **Never write to `~/.hipfire/`.** Everything needed is an env var. An agent edited the user's config and died before restoring it, leaving `max_seq` at 16384 instead of 131072.
- **Production `src/` may not read `HIPFIRE_*` directly** — `scripts/check-env-docs.py` rejects it. Route through `crates/rdna-compute/src/feature_flags.rs`, or take a parameter and let `examples/` read it.
- **Budget arithmetic, measured — do not re-derive:**

  | model | weights | KV/token | 4 agents × 128K | 4 × 96K |
  |---|---|---|---|---|
  | `qwen3.6:27b` | 15.0 GB | 34 KB | 33.25 GB — **does not fit** | 28.69 GB — fits |
  | `qwen3.6:35b-a3b` | ~20 GB | 10.6 KB | 25.8 GB — fits | 24.4 GB — fits |

  asym3 would have relaxed the 27B's limit but was **rejected on quality** (~30% of top-1 token choices change), so the cap is real, not temporary.
- **Three gates green after every task:** `./scripts/no-gpu-ci.sh` exit 0; `./scripts/kernel_resource_gate.sh` identical to its baseline (compile-only, always runnable); `./scripts/attn_legacy_baseline.sh` bitwise identical (needs GPU).
- `no-gpu-ci.sh` flakes with `ExecutableFileBusy` in the unrelated `hipfire-client` crate under parallel load — if you retry, **say so** rather than presenting the retry as a first-run pass.
- Check exit statuses directly; `cmd | tail && echo OK` prints OK when only `tail` succeeded.
- Licence header on new files. Commit with `git add <specific paths>` — never `git add -A`.

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `crates/hipfire-runtime/src/admission.rs` | `AdmissionController`: budget arithmetic and the admit/reject decision | create |
| `crates/hipfire-runtime/src/session_table.rs` | `SessionTable`: session ↔ slot mapping and per-session state | create |
| `crates/hipfire-runtime/src/lib.rs` | register both modules | modify |
| `crates/hipfire-runtime/examples/demo_concurrent_serve.rs` | several concurrent clients, each streaming its own tokens | create |

**Task order:** Task 1 (`AdmissionController`) is pure arithmetic and CPU-testable — and it is the safety-critical piece, so it comes first. Task 2 (`SessionTable`) is also CPU-only. Task 3 wires them into a concurrent demo, which needs the GPU.

---

### Task 1: `AdmissionController` — the budget gate

**Files:**
- Create: `crates/hipfire-runtime/src/admission.rs`
- Modify: `crates/hipfire-runtime/src/lib.rs`

**Interfaces:**
- Consumes: nothing (deliberately pure — it is testable without a GPU or a model).
- Produces:
  - `pub struct ModelFootprint { pub weights_bytes: u64, pub kv_bytes_per_token: u64 }`
  - `pub struct AdmissionController { footprint: ModelFootprint, budget_bytes: u64, admitted: Vec<usize> }`
  - `AdmissionController::new(footprint: ModelFootprint, budget_bytes: u64) -> Self`
  - `AdmissionController::admit(&mut self, requested_ctx: usize) -> Result<usize, AdmitError>` — returns the granted context, which may be less than requested
  - `AdmissionController::release(&mut self, granted_ctx: usize)`
  - `AdmissionController::used_bytes(&self) -> u64`
  - `pub enum AdmitError { PoolFull, WouldExceedBudget { need: u64, available: u64 } }`

- [ ] **Step 1: Write the failing tests**

Create `crates/hipfire-runtime/src/admission.rs` with the licence header, a stub whose methods `todo!()`, and:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    /// qwen3.6:27b — 15.0 GB of weights, 34 KB of KV per token.
    fn f27b() -> ModelFootprint {
        ModelFootprint { weights_bytes: 15 * GIB, kv_bytes_per_token: 34 * 1024 }
    }

    /// qwen3.6:35b-a3b — ~20 GB of weights, 10.6 KB of KV per token.
    fn f35b() -> ModelFootprint {
        ModelFootprint { weights_bytes: 20 * GIB, kv_bytes_per_token: 10_854 }
    }

    #[test]
    fn weights_are_charged_once_not_per_session() {
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        a.admit(1024).unwrap();
        let after_one = a.used_bytes();
        a.admit(1024).unwrap();
        let after_two = a.used_bytes();
        // The second session adds only its KV, never another copy of the weights.
        assert!(after_two - after_one < GIB, "weights charged twice");
        assert!(after_one >= 15 * GIB, "weights not charged at all");
    }

    #[test]
    fn the_27b_cannot_take_four_agents_at_128k() {
        // 15 GB + 4 x 4.25 GB = 32.25 GB against a 32 GB card.
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        for _ in 0..3 {
            a.admit(128 * 1024).expect("first three must fit");
        }
        let e = a.admit(128 * 1024).unwrap_err();
        assert!(matches!(e, AdmitError::WouldExceedBudget { .. }), "got {e:?}");
    }

    #[test]
    fn the_27b_does_take_four_agents_at_96k() {
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        for i in 0..4 {
            a.admit(96 * 1024).unwrap_or_else(|e| panic!("agent {i} rejected: {e:?}"));
        }
    }

    #[test]
    fn the_35b_does_take_four_agents_at_128k() {
        let mut a = AdmissionController::new(f35b(), 32 * GIB);
        for i in 0..4 {
            a.admit(128 * 1024).unwrap_or_else(|e| panic!("agent {i} rejected: {e:?}"));
        }
    }

    #[test]
    fn release_returns_budget_so_a_later_session_fits() {
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        for _ in 0..3 {
            a.admit(128 * 1024).unwrap();
        }
        assert!(a.admit(128 * 1024).is_err());
        a.release(128 * 1024);
        a.admit(128 * 1024).expect("budget must be reusable after release");
    }

    #[test]
    fn rejection_reports_the_numbers_not_just_a_failure() {
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        for _ in 0..3 {
            a.admit(128 * 1024).unwrap();
        }
        match a.admit(128 * 1024).unwrap_err() {
            AdmitError::WouldExceedBudget { need, available } => {
                assert!(need > available, "need {need} should exceed available {available}");
                assert!(available < 32 * GIB);
            }
            other => panic!("expected a budget rejection, got {other:?}"),
        }
    }

    #[test]
    fn a_single_session_over_budget_is_rejected_not_silently_capped() {
        // One agent asking for more than the whole card can hold.
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        assert!(a.admit(2 * 1024 * 1024).is_err(), "must reject, not silently truncate");
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --release -p hipfire-runtime --features deltanet,arch-qwen35 admission`
Expected: FAIL — the stub panics.

- [ ] **Step 3: Implement**

```rust
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// AdmissionController — decides whether a session can be admitted, and with how
// much context.
//
// In the test harnesses `kv_slots::preflight_alloc` is what stops an oversized
// configuration. In the daemon that job is HERE. The difference matters: on this
// hardware the GPU allocates from system RAM and the cgroup does NOT contain
// amdgpu GTT, so a wrong decision here does not fail a request — it takes down
// the user's desktop with a global OOM.

/// What one loaded model costs, split into the part charged once and the part
/// charged per session.
#[derive(Debug, Clone, Copy)]
pub struct ModelFootprint {
    /// Charged ONCE, however many sessions are admitted.
    pub weights_bytes: u64,
    /// Charged per session, per token of granted context.
    pub kv_bytes_per_token: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmitError {
    PoolFull,
    WouldExceedBudget { need: u64, available: u64 },
}

impl std::fmt::Display for AdmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gib = |b: u64| b as f64 / 1073741824.0;
        match self {
            AdmitError::PoolFull => write!(f, "no free slot"),
            AdmitError::WouldExceedBudget { need, available } => write!(
                f,
                "needs {:.2} GiB but only {:.2} GiB of the budget remains",
                gib(*need),
                gib(*available)
            ),
        }
    }
}

pub struct AdmissionController {
    footprint: ModelFootprint,
    budget_bytes: u64,
    /// Granted context per admitted session, in tokens.
    admitted: Vec<usize>,
}

impl AdmissionController {
    pub fn new(footprint: ModelFootprint, budget_bytes: u64) -> Self {
        Self { footprint, budget_bytes, admitted: Vec::new() }
    }

    /// Bytes currently committed: weights once (if anything is admitted) plus
    /// each session's KV.
    pub fn used_bytes(&self) -> u64 {
        if self.admitted.is_empty() {
            return 0;
        }
        let kv: u64 = self
            .admitted
            .iter()
            .map(|&ctx| ctx as u64 * self.footprint.kv_bytes_per_token)
            .sum();
        self.footprint.weights_bytes + kv
    }

    /// Admit a session at `requested_ctx` tokens, or explain why not.
    ///
    /// Rejects rather than silently capping: a caller that asked for 128K and
    /// silently got 8K would produce baffling truncation far from here.
    pub fn admit(&mut self, requested_ctx: usize) -> Result<usize, AdmitError> {
        let kv_need = requested_ctx as u64 * self.footprint.kv_bytes_per_token;
        // Weights are charged once, on the first admission.
        let weights_need =
            if self.admitted.is_empty() { self.footprint.weights_bytes } else { 0 };
        let need = kv_need + weights_need;
        let available = self.budget_bytes.saturating_sub(self.used_bytes());
        if need > available {
            return Err(AdmitError::WouldExceedBudget { need, available });
        }
        self.admitted.push(requested_ctx);
        Ok(requested_ctx)
    }

    /// Return a session's context allowance to the budget.
    pub fn release(&mut self, granted_ctx: usize) {
        if let Some(i) = self.admitted.iter().position(|&c| c == granted_ctx) {
            self.admitted.remove(i);
        }
    }
}
```

Register in `crates/hipfire-runtime/src/lib.rs` alongside the other `pub mod` declarations:

```rust
pub mod admission;
```

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test --release -p hipfire-runtime --features deltanet,arch-qwen35 admission`
Expected: PASS, 7 tests.

- [ ] **Step 5: Gates and commit**

```bash
./scripts/kernel_resource_gate.sh > /tmp/sp4t1res.txt 2>&1
diff scripts/kernel_resource_gate.beta.txt /tmp/sp4t1res.txt && echo RESOURCE_GATE_OK
./scripts/no-gpu-ci.sh > /tmp/sp4t1ci.txt 2>&1; echo "CI=$?"
```
Expected: `RESOURCE_GATE_OK` (no kernel touched) and `CI=0`.

```bash
git add crates/hipfire-runtime/src/admission.rs crates/hipfire-runtime/src/lib.rs
git commit -m "feat(sp4): AdmissionController — the production memory gate

Weights charged once, KV per session. Rejects with the numbers rather
than silently capping. In the harnesses preflight_alloc does this job; in
the daemon it is here, and a wrong decision takes down the desktop rather
than failing a request."
```

---

### Task 2: `SessionTable` — sessions to slots

**Files:**
- Create: `crates/hipfire-runtime/src/session_table.rs`
- Modify: `crates/hipfire-runtime/src/lib.rs`

**Interfaces:**
- Consumes: `rdna_compute::slot_pool::{SlotPool, SlotId}` (SP2 Task 1); `AdmissionController` (Task 1).
- Produces:
  - `pub struct SessionId(pub u64)`
  - `pub struct Session { pub slot: SlotId, pub granted_ctx: usize, pub tokens: Vec<u32>, pub next_pos: usize }`
  - `pub struct SessionTable { ... }`
  - `SessionTable::open(&mut self, pool: &mut SlotPool, adm: &mut AdmissionController, requested_ctx: usize) -> Result<SessionId, AdmitError>`
  - `SessionTable::close(&mut self, pool: &mut SlotPool, adm: &mut AdmissionController, id: SessionId)`
  - `SessionTable::get(&self, id: SessionId) -> Option<&Session>`
  - `SessionTable::get_mut(&mut self, id: SessionId) -> Option<&mut Session>`
  - `SessionTable::active(&self) -> usize`

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::{AdmissionController, ModelFootprint};
    use rdna_compute::slot_pool::SlotPool;

    const GIB: u64 = 1024 * 1024 * 1024;
    const PPB: usize = 1088;

    fn rig(n_slots: usize) -> (SlotPool, AdmissionController, SessionTable) {
        let pool = SlotPool::new(n_slots, 4096, PPB).unwrap();
        let adm = AdmissionController::new(
            ModelFootprint { weights_bytes: GIB, kv_bytes_per_token: 1024 },
            32 * GIB,
        );
        (pool, adm, SessionTable::default())
    }

    #[test]
    fn open_assigns_a_slot_and_close_returns_it() {
        let (mut pool, mut adm, mut t) = rig(1);
        let id = t.open(&mut pool, &mut adm, 1024).unwrap();
        assert_eq!(t.active(), 1);
        // The single slot is taken.
        assert!(t.open(&mut pool, &mut adm, 1024).is_err());
        t.close(&mut pool, &mut adm, id);
        assert_eq!(t.active(), 0);
        // And is reusable.
        t.open(&mut pool, &mut adm, 1024).expect("slot must be reusable");
    }

    #[test]
    fn a_rejected_admission_does_not_consume_a_slot() {
        let (mut pool, mut adm, mut t) = rig(2);
        // Far beyond the budget.
        assert!(t.open(&mut pool, &mut adm, 100_000_000).is_err());
        assert_eq!(t.active(), 0, "a rejected session must leave the pool untouched");
        t.open(&mut pool, &mut adm, 1024).expect("pool must still have both slots");
    }

    #[test]
    fn sessions_keep_independent_token_history() {
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        let b = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.get_mut(a).unwrap().tokens.extend_from_slice(&[1, 2, 3]);
        t.get_mut(b).unwrap().tokens.push(9);
        assert_eq!(t.get(a).unwrap().tokens, vec![1, 2, 3]);
        assert_eq!(t.get(b).unwrap().tokens, vec![9]);
    }

    #[test]
    fn closing_frees_budget_for_a_later_session() {
        let (mut pool, mut adm, mut t) = rig(2);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        let before = adm.used_bytes();
        t.close(&mut pool, &mut adm, a);
        assert!(adm.used_bytes() < before, "close must return budget");
    }

    #[test]
    fn a_closed_session_id_is_not_reusable_by_accident() {
        let (mut pool, mut adm, mut t) = rig(1);
        let a = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.close(&mut pool, &mut adm, a);
        assert!(t.get(a).is_none(), "a closed session must not resolve");
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --release -p hipfire-runtime --features deltanet,arch-qwen35 session_table`
Expected: FAIL.

- [ ] **Step 3: Implement**

`SessionTable` holds a `HashMap<u64, Session>` and a monotonically increasing id counter — **ids are never reused**, so a stale id from a closed session resolves to `None` rather than silently addressing whoever now holds that slot.

`open` must call `adm.admit(requested_ctx)` **first** and only take a slot from the pool if admission succeeds, so a rejected request leaves the pool untouched. `close` releases the slot and the budget together.

- [ ] **Step 4: Run to verify they pass, then commit**

Run: `cargo test --release -p hipfire-runtime --features deltanet,arch-qwen35 session_table`
Expected: PASS, 5 tests.

```bash
git add crates/hipfire-runtime/src/session_table.rs crates/hipfire-runtime/src/lib.rs
git commit -m "feat(sp4): SessionTable — sessions to slots, ids never reused

Admission runs before the slot is taken, so a rejected request leaves the
pool untouched. Session ids are never reused, so a stale id resolves to
None rather than silently addressing whoever now holds that slot."
```

---

### Task 3: Concurrent serve demo

**Files:**
- Create: `crates/hipfire-runtime/examples/demo_concurrent_serve.rs`

**Interfaces:**
- Consumes: everything above, plus SP3's `Scheduler` and `forward_batch_slots`.
- Produces: a runnable demo.

- [ ] **Step 1: Write the demo**

Accept N prompts (env var, default 3), open a session per prompt through `SessionTable`, and drive `Scheduler` + `forward_batch_slots` in a loop, printing each session's tokens as produced with a session prefix so interleaving is visible. Print each admission decision — including rejections with their numbers — so the budget gate is observable.

- [ ] **Step 2: Run, only if the box is free**

```bash
pgrep -f 'hipfire/bin/daemon'   # must find nothing
awk '/MemAvailable/{printf "%.1f GiB\n", $2/1048576}' /proc/meminfo
HIPFIRE_MEM_CAP=28G ./scripts/run-bounded.sh cargo run --release -p hipfire-runtime --features deltanet,arch-qwen35 --example demo_concurrent_serve
```
Cap context so the run stays well inside budget. If a daemon is resident or memory is short, commit and report as **written but unrun**. A partial safe result is correct; an OOM is not.

- [ ] **Step 3: Commit**

```bash
git add crates/hipfire-runtime/examples/demo_concurrent_serve.rs
git commit -m "feat(sp4): concurrent serve demo

Several sessions admitted against a real budget, each streaming its own
tokens from one GPU."
```

---

## Completion

SP4 is done when the spec's five success criteria hold — most importantly that several concurrent clients each receive their own tokens, and that admission control refuses over-budget requests with a reason rather than OOMing.

**What SP4 does not deliver:** KV swap-on-idle. It is specified (SP1 spec §15) and deliberately conditional on first confirming that 4 × ~96K is genuinely insufficient, since that fits today.
