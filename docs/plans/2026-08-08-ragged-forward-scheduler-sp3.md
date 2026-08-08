# SP3 — Ragged Multi-Slot Forward and Scheduler: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble SP1's slot-aware kernels and SP2's slot state into a forward pass that advances N slots per step, plus a minimal scheduler — producing the programme's first visible result: N sequences generating concurrently.

**Architecture:** A *parallel* entry point, `forward_batch_slots`, driving the same per-layer sequence as `forward_prefill_batch_with_pbs_opts` but with slot-aware calls. The existing function is not modified, so every current caller (chat, spec decode, MTP, TUI) is untouched while the multi-slot path matures.

**Tech Stack:** Rust (`hipfire-arch-qwen35`, `rdna-compute`), HIP kernels already ported in SP1/SP2.

**Spec:** `docs/specs/2026-08-08-ragged-forward-scheduler-sp3.md`. Read it first.

## Global Constraints

- **Branch:** `feat/batched-attn-impl`, worktree `~/repos/hipfire-batchattn-impl`.
- **`--features deltanet` on every cargo command**; scope to `-p <crate>`, never the whole workspace.
- **Do not modify `forward_prefill_batch_with_pbs_opts`.** It carries hipGraph capture eligibility, tree-verify, GDN tape and MTP interactions whose interaction with a slot axis is unknown. Breaking AR decode is a far worse outcome than duplication.
- **`positions[]` is authoritative for the causal bound — never `desc.seq_len`.** They differ whenever a slot has more than one query row. This caused SP1's only Critical defect.
- **The KV write resolves both arenas through `k_base`.** Correct under the Q8 ABI (`v_base == k_base`, enforced by `SlotPool`), but **asym3 cannot use that path** — its K and V strides differ. Do not treat the write kernel as mode-agnostic.
- **DeltaNet: one launch per slot.** The recurrence is sequential within a slot. Mixing slots in one launch would interleave independent recurrences.
- **RoPE is slot-agnostic** — verified in SP2 Task 2, no work needed.
- **`tree_bias` with descriptors is asserted out of scope** in SP1 and stays that way.
- **MEMORY — binding and measured.**
  - **The cgroup does NOT contain amdgpu GTT.** A gated run still invoked the *global* OOM killer and killed the user's Slack and three steamwebhelper processes.
  - Run every GPU harness through `./scripts/run-bounded.sh`; it refuses unless `MemAvailable >= cap + 10 GiB`, which is the **primary** protection.
  - **Never run a GPU harness while a serve process holds a model.** Check `pgrep -f 'hipfire/bin/daemon'`; one resident model takes MemAvailable from ~58 GiB to ~19 GiB. Wait rather than contend.
  - Free per-iteration GPU tensors inside loops.
  - A live `free` shows nothing after the fact — check `journalctl -k | grep -E 'Out of memory|CONSTRAINT_NONE'`.
- **Never write to `~/.hipfire/`.** Everything needed is an env var.
- **No device `assert()`** — `compiler.rs` never passes `-DNDEBUG`, so they ship in release (SP1 measured 64 B/lane of scratch on four kernels).
- **Three gates green after every task:**
  - `./scripts/attn_legacy_baseline.sh` vs `scripts/attn_legacy_baseline.beta.txt` → bitwise identical (needs GPU).
  - `./scripts/kernel_resource_gate.sh` vs `scripts/kernel_resource_gate.beta.txt` → identical (compile-only, always runnable; covers six kernels including DeltaNet).
  - `./scripts/no-gpu-ci.sh` → exit 0. It flakes with `ExecutableFileBusy` in the unrelated `hipfire-client` crate under parallel load; if you retry, **say so** rather than presenting the retry as a first-run pass.
- Check exit statuses directly; `cmd | tail && echo OK` prints OK when only `tail` succeeded.
- Licence header on new files. Commit with `git add <specific paths>` — never `git add -A`.

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `crates/hipfire-arch-qwen35/src/slot_batch.rs` | `SlotBatch`: per-step ragged work description + builder | create |
| `crates/hipfire-arch-qwen35/src/forward_slots.rs` | `forward_batch_slots`: the N-slot forward | create |
| `crates/hipfire-arch-qwen35/src/lib.rs` | register both modules | modify |
| `crates/hipfire-arch-qwen35/src/scheduler.rs` | round-robin scheduler mixing chunked prefill with decode | create |
| `crates/hipfire-runtime/examples/demo_multislot_generate.rs` | the visible result: N sequences generating concurrently | create |

**Task order:** Task 1 is pure data (`SlotBatch`) and CPU-testable. Task 2 is the forward, the largest piece. Task 3 is golden equivalence — the gate. Task 4 is the scheduler. Task 5 is the demo.

---

### Task 1: `SlotBatch` — one step's ragged work

**Files:**
- Create: `crates/hipfire-arch-qwen35/src/slot_batch.rs`
- Modify: `crates/hipfire-arch-qwen35/src/lib.rs`

**Interfaces:**
- Consumes: `rdna_compute::kv_slots::KvSlotDesc`, `rdna_compute::slot_pool::{SlotPool, SlotId}` (SP2).
- Produces:
  - `pub struct SlotBatch { pub m_per_slot: Vec<usize>, pub tokens: Vec<u32>, pub positions: Vec<i32>, pub row_slot: Vec<i32> }`
  - `SlotBatch::build(per_slot: &[(SlotId, &[u32], usize)]) -> SlotBatch` where the tuple is `(slot, tokens, start_pos)`
  - `SlotBatch::total_rows(&self) -> usize`
  - `SlotBatch::is_empty(&self) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `crates/hipfire-arch-qwen35/src/slot_batch.rs` with the licence header, a stub whose methods `unimplemented!()`, and:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::slot_pool::SlotId;

    #[test]
    fn packs_tokens_in_slot_order() {
        let b = SlotBatch::build(&[
            (SlotId(0), &[10u32, 11][..], 100),
            (SlotId(1), &[20u32][..], 5),
        ]);
        assert_eq!(b.tokens, vec![10, 11, 20]);
        assert_eq!(b.m_per_slot, vec![2, 1]);
        assert_eq!(b.total_rows(), 3);
    }

    #[test]
    fn positions_advance_within_a_slot_from_its_own_start() {
        // Slot 0 verifying 3 tokens at start_pos 100 occupies 100,101,102.
        // Slot 1 decoding 1 token at start_pos 5 occupies 5. They are
        // independent -- positions are per-slot absolute, not batch-global.
        let b = SlotBatch::build(&[
            (SlotId(0), &[1u32, 2, 3][..], 100),
            (SlotId(1), &[9u32][..], 5),
        ]);
        assert_eq!(b.positions, vec![100, 101, 102, 5]);
    }

    #[test]
    fn row_slot_maps_every_flat_row_to_its_slot() {
        let b = SlotBatch::build(&[
            (SlotId(0), &[1u32, 2, 3][..], 0),
            (SlotId(2), &[7u32][..], 0),
        ]);
        assert_eq!(b.row_slot, vec![0, 0, 0, 2]);
    }

    #[test]
    fn idle_slots_contribute_no_rows() {
        let b = SlotBatch::build(&[
            (SlotId(0), &[][..], 0),
            (SlotId(1), &[5u32][..], 42),
        ]);
        assert_eq!(b.m_per_slot, vec![0, 1]);
        assert_eq!(b.tokens, vec![5]);
        assert_eq!(b.positions, vec![42]);
        assert_eq!(b.row_slot, vec![1]);
    }

    #[test]
    fn an_all_idle_batch_is_empty() {
        let b = SlotBatch::build(&[(SlotId(0), &[][..], 0)]);
        assert!(b.is_empty());
        assert_eq!(b.total_rows(), 0);
    }

    #[test]
    fn mixed_prefill_and_decode_is_the_shape_this_exists_for() {
        // slot 0 verifies 8 draft tokens, slot 1 chunk-prefills 256,
        // slots 2-3 decode 1 each.
        let p0: Vec<u32> = (0..8).collect();
        let p1: Vec<u32> = (0..256).collect();
        let b = SlotBatch::build(&[
            (SlotId(0), &p0[..], 1000),
            (SlotId(1), &p1[..], 0),
            (SlotId(2), &[1u32][..], 50),
            (SlotId(3), &[2u32][..], 77),
        ]);
        assert_eq!(b.total_rows(), 266);
        assert_eq!(b.row_slot.iter().filter(|&&s| s == 1).count(), 256);
        assert_eq!(b.positions[b.positions.len() - 1], 77);
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --release -p hipfire-arch-qwen35 --features deltanet slot_batch`
Expected: FAIL — the stub panics.

- [ ] **Step 3: Implement**

```rust
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SlotBatch — one forward step's ragged work across N slots.
//
// A step mixes freely: one slot verifying 8 draft tokens, another
// chunk-prefilling 256, others decoding 1 each. That raggedness is what SP1's
// kernels were built for.

use rdna_compute::slot_pool::SlotId;

#[derive(Debug, Clone, Default)]
pub struct SlotBatch {
    /// Per-slot token counts for this step. 0 means the slot is idle.
    pub m_per_slot: Vec<usize>,
    /// Flat token ids, packed across slots in slot order.
    pub tokens: Vec<u32>,
    /// Per-row ABSOLUTE position within that row's own slot.
    ///
    /// Authoritative for the causal bound — never `desc.seq_len`. The two
    /// differ whenever a slot has more than one query row, and conflating them
    /// caused SP1's only Critical defect.
    pub positions: Vec<i32>,
    /// Slot index for each flat row.
    pub row_slot: Vec<i32>,
}

impl SlotBatch {
    /// Build a step from `(slot, tokens, start_pos)` triples. Slots with no
    /// tokens contribute no rows.
    pub fn build(per_slot: &[(SlotId, &[u32], usize)]) -> Self {
        let mut b = SlotBatch::default();
        for (slot, toks, start_pos) in per_slot {
            b.m_per_slot.push(toks.len());
            for (i, t) in toks.iter().enumerate() {
                b.tokens.push(*t);
                b.positions.push((start_pos + i) as i32);
                b.row_slot.push(slot.0 as i32);
            }
        }
        b
    }

    pub fn total_rows(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}
```

Register in `crates/hipfire-arch-qwen35/src/lib.rs` alongside the other `pub mod` declarations:

```rust
pub mod slot_batch;
```

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test --release -p hipfire-arch-qwen35 --features deltanet slot_batch`
Expected: PASS, 6 tests.

- [ ] **Step 5: Gates and commit**

```bash
./scripts/kernel_resource_gate.sh > /tmp/sp3t1res.txt 2>&1
diff scripts/kernel_resource_gate.beta.txt /tmp/sp3t1res.txt && echo RESOURCE_GATE_OK
./scripts/no-gpu-ci.sh > /tmp/sp3t1ci.txt 2>&1; echo "CI=$?"
```
Expected: `RESOURCE_GATE_OK` (this task touches no kernel) and `CI=0`.

```bash
git add crates/hipfire-arch-qwen35/src/slot_batch.rs crates/hipfire-arch-qwen35/src/lib.rs
git commit -m "feat(sp3): SlotBatch — one step's ragged work across N slots

Positions are per-slot absolute and authoritative for the causal bound;
conflating them with desc.seq_len caused SP1's only Critical defect."
```

---

### Task 2: `forward_batch_slots` — the N-slot forward

The largest task in the plan. It mirrors the per-layer sequence of `forward_prefill_batch_with_pbs_opts` but calls the slot-aware entry points SP1 and SP2 built.

**Files:**
- Create: `crates/hipfire-arch-qwen35/src/forward_slots.rs`
- Modify: `crates/hipfire-arch-qwen35/src/lib.rs`

**Interfaces:**
- Consumes: `SlotBatch` (Task 1); `SlotPool` (SP2 Task 1); `Gpu::kv_cache_write_q8_0_batched_slots` (SP2 Task 3); `Gpu::gated_delta_net_q8_batch_seq_slots` (SP2 Task 4); `Gpu::sample_per_slot` (SP2 Task 5); `Gpu::attention_flash_q8_0_batched_masked_slots` and `Gpu::attention_q8_0_kv_batched_masked_slots` (SP1).
- Produces: `pub fn forward_batch_slots(gpu, weights, config, batch: &SlotBatch, pool: &mut SlotPool, dn_states: &mut [DeltaNetState], scratch, logits_out: &GpuTensor) -> HipResult<()>`

- [ ] **Step 1: Read the reference implementation**

Read `crates/hipfire-arch-qwen35/src/qwen35.rs:6554` onward — `forward_prefill_batch_with_pbs_opts`. Write down, in your report, the per-layer sequence it performs, in order, distinguishing `LayerType::FullAttention` from `LayerType::LinearAttention` layers. You are mirroring this; you cannot mirror what you have not enumerated.

**Do not modify that function.**

- [ ] **Step 2: Implement the skeleton with attention only**

Write `forward_batch_slots` handling **only** `FullAttention` layers first, leaving DeltaNet layers as an explicit `unimplemented!("Step 3")`. Upload the descriptor table from `pool.descriptors()` once per step, guarded by `pool.descriptors_dirty()`, calling `pool.mark_uploaded()` after.

Route attention through the `_slots` entry points, passing `Some(&descs_dev)` and `Some(&row_slot_dev)` — **both or neither**, the kernels assert it.

- [ ] **Step 3: Add the DeltaNet layers**

One launch per slot, per layer. For each slot with `m_per_slot[s] > 0`, call `gated_delta_net_q8_batch_seq_slots` with that slot's token rows and its own state, using the slot stride. The recurrence is sequential within a slot; do not attempt to batch slots into one launch.

- [ ] **Step 4: Build and gate**

```bash
cargo build --release -p hipfire-arch-qwen35 --features deltanet
./scripts/kernel_resource_gate.sh > /tmp/sp3t2res.txt 2>&1
diff scripts/kernel_resource_gate.beta.txt /tmp/sp3t2res.txt && echo RESOURCE_GATE_OK
```
Expected: builds clean, `RESOURCE_GATE_OK` (this task adds no kernel).

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-arch-qwen35/src/forward_slots.rs crates/hipfire-arch-qwen35/src/lib.rs
git commit -m "feat(sp3): forward_batch_slots — advance N slots in one step

A parallel entry point rather than a modification of
forward_prefill_batch_with_pbs_opts, which carries graph-capture, MTP and
spec-decode interactions whose interaction with a slot axis is unknown."
```

---

### Task 3: Golden equivalence — the SP3 gate

**Files:**
- Create: `crates/hipfire-runtime/examples/test_forward_slots_golden.rs`

**Interfaces:**
- Consumes: `forward_batch_slots` (Task 2).
- Produces: a pass/fail harness.

**The comparison must be on logits, not tokens.** Sampling makes runs diverge; greedy decoding on a quantised model can also diverge on a near-tie. Comparing per-slot logits at each step against the same sequence run alone through the existing single-sequence path is the sound check.

- [ ] **Step 1: Write the harness**

For `n_slots` in 1..=4, with per-slot prompts of differing length:
1. Run each prompt alone through the existing single-sequence forward, recording per-step logits.
2. Run all slots together through `forward_batch_slots`, recording per-step logits.
3. Compare with a tolerance-based `assert_close` that **rejects an all-zero reference** — SP1 found two all-zero arrays passing at `0.000x`.

- [ ] **Step 2: Add a negative control**

Corrupt the **candidate arm only** — for example point two slots at the same `SlotId` — and confirm a genuine numeric mismatch. SP1's first attempt corrupted both arms and could never fail; the corrected version produced a mismatch at `91.44x tolerance`. Report the control's actual output.

- [ ] **Step 3: Run, only if the box is free**

```bash
pgrep -f 'hipfire/bin/daemon'   # must find nothing
awk '/MemAvailable/{printf "%.1f GiB\n", $2/1048576}' /proc/meminfo
./scripts/run-bounded.sh cargo run --release -p hipfire-runtime --features deltanet,arch-qwen35 --example test_forward_slots_golden
```
If a daemon is resident or memory is short, **commit the harness, report it as written-but-unrun, and stop.** A partial safe result is correct; an OOM is not.

- [ ] **Step 4: Commit**

```bash
git add crates/hipfire-runtime/examples/test_forward_slots_golden.rs
git commit -m "test(sp3): golden equivalence for the multi-slot forward

Compares logits rather than tokens, because sampling and near-ties make
token-level equivalence unsound on a quantised model."
```

---

### Task 4: Scheduler

**Files:**
- Create: `crates/hipfire-arch-qwen35/src/scheduler.rs`
- Modify: `crates/hipfire-arch-qwen35/src/lib.rs`

**Interfaces:**
- Consumes: `SlotBatch` (Task 1), `SlotPool` (SP2).
- Produces:
  - `pub struct Scheduler { chunk_size: usize }`
  - `pub struct PendingWork { pub slot: SlotId, pub remaining_prompt: Vec<u32>, pub next_pos: usize, pub decoding: bool }`
  - `Scheduler::next_batch(&mut self, work: &mut [PendingWork]) -> SlotBatch`

Deliberately minimal: round-robin, chunked prefill mixed with decode, no preemption or priorities. Justified by measurement — batching wins ~1.36× at 8 slots and recovers under a tenth of the bandwidth headroom, so contorting the scheduler for batch width is not where the performance is.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::slot_pool::SlotId;

    fn prompt(n: usize) -> Vec<u32> { (0..n as u32).collect() }

    #[test]
    fn a_long_prompt_is_chunked_not_run_whole() {
        let mut s = Scheduler { chunk_size: 256 };
        let mut work = vec![PendingWork {
            slot: SlotId(0), remaining_prompt: prompt(1000), next_pos: 0, decoding: false,
        }];
        let b = s.next_batch(&mut work);
        assert_eq!(b.total_rows(), 256, "must take one chunk, not the whole prompt");
        assert_eq!(work[0].remaining_prompt.len(), 744);
        assert_eq!(work[0].next_pos, 256);
    }

    #[test]
    fn prefill_and_decode_mix_in_one_batch() {
        let mut s = Scheduler { chunk_size: 256 };
        let mut work = vec![
            PendingWork { slot: SlotId(0), remaining_prompt: prompt(300), next_pos: 0, decoding: false },
            PendingWork { slot: SlotId(1), remaining_prompt: vec![42], next_pos: 10, decoding: true },
        ];
        let b = s.next_batch(&mut work);
        assert_eq!(b.m_per_slot, vec![256, 1], "a prefilling slot must not block a decoding one");
    }

    #[test]
    fn a_prompt_shorter_than_a_chunk_completes_in_one_batch() {
        let mut s = Scheduler { chunk_size: 256 };
        let mut work = vec![PendingWork {
            slot: SlotId(0), remaining_prompt: prompt(10), next_pos: 0, decoding: false,
        }];
        let b = s.next_batch(&mut work);
        assert_eq!(b.total_rows(), 10);
        assert!(work[0].remaining_prompt.is_empty());
    }

    #[test]
    fn an_idle_slot_contributes_nothing() {
        let mut s = Scheduler { chunk_size: 256 };
        let mut work = vec![PendingWork {
            slot: SlotId(0), remaining_prompt: vec![], next_pos: 0, decoding: false,
        }];
        let b = s.next_batch(&mut work);
        assert!(b.is_empty());
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --release -p hipfire-arch-qwen35 --features deltanet scheduler`
Expected: FAIL.

- [ ] **Step 3: Implement**

`next_batch` walks `work`, taking `min(chunk_size, remaining_prompt.len())` tokens from each entry, advancing `next_pos` by what it took, and building a `SlotBatch` from the results. A decoding entry contributes its single pending token.

- [ ] **Step 4: Run to verify they pass, then commit**

Run: `cargo test --release -p hipfire-arch-qwen35 --features deltanet scheduler`
Expected: PASS, 4 tests.

```bash
git add crates/hipfire-arch-qwen35/src/scheduler.rs crates/hipfire-arch-qwen35/src/lib.rs
git commit -m "feat(sp3): round-robin scheduler mixing chunked prefill with decode

Minimal by design: batching wins ~1.36x at 8 slots and recovers under a
tenth of the bandwidth headroom, so contorting the scheduler for batch
width is not where the performance is."
```

---

### Task 5: The demo — N sequences generating concurrently

The programme's first visible result.

**Files:**
- Create: `crates/hipfire-runtime/examples/demo_multislot_generate.rs`

**Interfaces:**
- Consumes: everything above.
- Produces: a runnable demo.

- [ ] **Step 1: Write the demo**

Load one model, create a `SlotPool`, admit N prompts (N from an env var, default 3), and drive `Scheduler` + `forward_batch_slots` in a loop, printing each slot's tokens as they are produced with a slot prefix so interleaving is visible.

Call `kv_slots::preflight_alloc` before allocating, with the TOTAL held live at once — **device and host**.

- [ ] **Step 2: Run, only if the box is free**

```bash
pgrep -f 'hipfire/bin/daemon'   # must find nothing
HIPFIRE_MEM_CAP=28G ./scripts/run-bounded.sh cargo run --release -p hipfire-runtime --features deltanet,arch-qwen35 --example demo_multislot_generate
```
Cap evaluation context so the run stays well inside the budget. If the box is busy, commit and report as written-but-unrun.

- [ ] **Step 3: Commit**

```bash
git add crates/hipfire-runtime/examples/demo_multislot_generate.rs
git commit -m "feat(sp3): demo — N sequences generating concurrently

The programme's first visible result: SP1's kernels, SP2's state and SP3's
forward and scheduler driving several independent sequences on one GPU."
```

---

## Completion

SP3 is done when the spec's six success criteria hold, most importantly that the demo runs N sequences concurrently and that the existing single-sequence paths are untouched.

**What SP3 does not deliver:** anything reachable by a client. That is SP4.
