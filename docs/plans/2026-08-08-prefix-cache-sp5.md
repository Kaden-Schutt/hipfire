# SP5 Session Prefix Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A session's second turn reuses the KV it already built, prefilling only the tokens that diverge from its stored conversation.

**Architecture:** Compute the longest common prefix (LCP) between a session's stored tokens and the incoming prompt, rewind the slot's `seq_len` to that point, and prefill only `prompt[lcp..]`. The rewind moves no data — KV past `lcp` is simply overwritten by the next prefill. This generalises `daemon.rs`'s existing single-session mechanism to N slots.

**Tech Stack:** Rust, `hipfire-runtime` crate, `rdna_compute::slot_pool::SlotPool`, existing `hipfire-arch-qwen35` scheduler and `forward_batch_slots_graphed`.

## Global Constraints

- **The tokens are authoritative; KV is only a cache.** Every failure path must degrade to re-prefilling from `session.tokens`, never to wrong output.
- LCP is computed against **a token slice**, not against "this session's tokens" — a future shared-prefix table must be able to supply the other side without a rewrite.
- Never reuse more KV than the slot actually holds: cap LCP at the slot's current `seq_len`.
- Always leave at least one token to prefill, so the forward produces logits.
- Formatting: run `BASE_REF=origin/beta ./scripts/fmt-changed.sh` before committing.
- GPU harnesses run under `./scripts/run-bounded.sh` with `HIPFIRE_MEM_CAP` set.
- Existing gates must stay green: `test_forward_slots_golden` at 0.000×, `kernel_resource_gate.sh` and `attn_legacy_baseline.sh` bitwise identical to beta.

## File Structure

| File | Responsibility |
|---|---|
| `crates/hipfire-runtime/src/prefix.rs` (create) | `lcp()` and `TurnPlan` — pure logic, no GPU, no session types |
| `crates/hipfire-runtime/src/session_table.rs` (modify) | `SessionTable::begin_turn` — applies a `TurnPlan` to the pool and the session |
| `crates/hipfire-runtime/src/lib.rs` (modify) | register `pub mod prefix;` |
| `crates/hipfire-runtime/examples/test_prefix_cache_equivalence.rs` (create) | the SP5 gate: reused-prefix turn vs cold full prefill, token-for-token |

---

### Task 1: `lcp()` and `TurnPlan` — the pure logic

**Files:**
- Create: `crates/hipfire-runtime/src/prefix.rs`
- Modify: `crates/hipfire-runtime/src/lib.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `pub fn lcp(a: &[u32], b: &[u32]) -> usize`; `pub struct TurnPlan { pub lcp: usize, pub reused: usize, pub to_prefill: usize }`; `pub fn plan_turn(cached: &[u32], seq_len: usize, prompt: &[u32]) -> TurnPlan`.

- [ ] **Step 1: Write the failing tests**

Create `crates/hipfire-runtime/src/prefix.rs` containing only this test module plus `use` lines:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcp_counts_the_shared_leading_run() {
        assert_eq!(lcp(&[1, 2, 3, 4], &[1, 2, 9, 4]), 2);
        assert_eq!(lcp(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(lcp(&[], &[1, 2]), 0);
        assert_eq!(lcp(&[1, 2], &[]), 0);
        assert_eq!(lcp(&[9], &[1]), 0);
    }

    #[test]
    fn a_continuing_turn_reuses_everything_but_the_last_token() {
        // Turn 2 is turn 1's conversation plus new user text.
        let cached = [1, 2, 3, 4];
        let prompt = [1, 2, 3, 4, 5, 6];
        let p = plan_turn(&cached, 4, &prompt);
        assert_eq!(p.lcp, 4);
        assert_eq!(p.reused, 4);
        assert_eq!(p.to_prefill, 2, "only the new tokens are prefilled");
    }

    #[test]
    fn an_identical_prompt_still_leaves_one_token_to_prefill() {
        // Without this the forward gets an empty batch and produces no logits.
        let cached = [1, 2, 3, 4];
        let p = plan_turn(&cached, 4, &[1, 2, 3, 4]);
        assert_eq!(p.lcp, 3, "must stop one short of the whole prompt");
        assert_eq!(p.to_prefill, 1);
    }

    #[test]
    fn reuse_is_capped_by_what_the_slot_actually_holds() {
        // The session remembers 4 tokens but the slot only has 2 of them.
        let cached = [1, 2, 3, 4];
        let p = plan_turn(&cached, 2, &[1, 2, 3, 4, 5]);
        assert_eq!(p.lcp, 2, "must not claim KV the slot does not hold");
        assert_eq!(p.to_prefill, 3);
    }

    #[test]
    fn a_diverging_turn_falls_back_to_near_full_prefill() {
        let cached = [1, 2, 3, 4];
        let p = plan_turn(&cached, 4, &[9, 9, 9]);
        assert_eq!(p.lcp, 0);
        assert_eq!(p.reused, 0);
        assert_eq!(p.to_prefill, 3);
    }

    #[test]
    fn a_cold_session_prefills_the_whole_prompt() {
        let p = plan_turn(&[], 0, &[1, 2, 3]);
        assert_eq!(p.lcp, 0);
        assert_eq!(p.to_prefill, 3);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p hipfire-runtime --lib prefix 2>&1 | tail -20`
Expected: FAIL to compile — `cannot find function 'lcp'`, `cannot find function 'plan_turn'`.

- [ ] **Step 3: Write the implementation**

Prepend to `crates/hipfire-runtime/src/prefix.rs`, above the test module:

```rust
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Prefix reuse: how much of a session's existing KV a new turn can keep.
//
// Pure logic, deliberately knowing nothing about sessions, slots or the GPU.
// `lcp` takes two token slices rather than a session, so a future
// cross-session shared-prefix table can supply the other side without any
// change here (spec §5.1, §10).

/// Length of the longest common prefix of two token sequences.
pub fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// What a turn can reuse and what it must compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnPlan {
    /// Tokens of `prompt` already represented in the slot's KV.
    pub lcp: usize,
    /// Same as `lcp`; named separately because it is the quantity the slot's
    /// `seq_len` is rewound to, and conflating the two is how the SP1
    /// Critical defect happened (`positions[]` vs `desc.seq_len`).
    pub reused: usize,
    /// Tokens that must actually be prefilled: `prompt.len() - lcp`.
    pub to_prefill: usize,
}

/// Decide how much of `prompt` a slot holding `seq_len` valid tokens of
/// `cached` can skip.
///
/// Two caps, both load-bearing:
///
/// 1. **`seq_len`** — the session may remember more tokens than the slot's KV
///    actually holds (it was swapped out and partially restored, or rewound
///    by an earlier turn). Reusing beyond that reads KV that was never
///    written.
/// 2. **`prompt.len() - 1`** — a turn whose prompt exactly equals the cached
///    conversation would otherwise have nothing to prefill, and the forward
///    would produce no logits to sample. Always leave the last token.
pub fn plan_turn(cached: &[u32], seq_len: usize, prompt: &[u32]) -> TurnPlan {
    let shared = lcp(cached, prompt);
    let capped = shared.min(seq_len).min(prompt.len().saturating_sub(1));
    TurnPlan {
        lcp: capped,
        reused: capped,
        to_prefill: prompt.len() - capped,
    }
}
```

Add to `crates/hipfire-runtime/src/lib.rs` alongside the other `pub mod` lines:

```rust
pub mod prefix;
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p hipfire-runtime --lib prefix 2>&1 | grep "test result"`
Expected: `test result: ok. 6 passed; 0 failed`

- [ ] **Step 5: Commit**

```bash
BASE_REF=origin/beta ./scripts/fmt-changed.sh
git add crates/hipfire-runtime/src/prefix.rs crates/hipfire-runtime/src/lib.rs
git commit -m "feat(prefix): LCP and TurnPlan for session prefix reuse"
```

---

### Task 2: `SessionTable::begin_turn` — apply the plan

**Files:**
- Modify: `crates/hipfire-runtime/src/session_table.rs`

**Interfaces:**
- Consumes: `prefix::{plan_turn, TurnPlan}` from Task 1; existing `Session { slot, granted_ctx, tokens, next_pos }`; `SlotPool::{descriptors, set_seq_len}`.
- Produces: `pub fn begin_turn(&mut self, pool: &mut SlotPool, id: SessionId, prompt: &[u32]) -> Result<TurnPlan, String>`.

- [ ] **Step 1: Write the failing tests**

Append inside the existing `mod tests` block in `crates/hipfire-runtime/src/session_table.rs`:

```rust
    #[test]
    fn begin_turn_rewinds_the_slot_and_reports_the_suffix() {
        let (mut pool, mut adm, mut t) = rig(1);
        let id = t.open(&mut pool, &mut adm, 1024).unwrap();
        // Turn 1: four tokens are prefilled and recorded.
        {
            let s = t.get_mut(id).unwrap();
            s.tokens.extend_from_slice(&[1, 2, 3, 4]);
            s.next_pos = 4;
        }
        pool.set_seq_len(t.get(id).unwrap().slot, 4).unwrap();

        // Turn 2 continues the same conversation.
        let plan = t.begin_turn(&mut pool, id, &[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(plan.reused, 4);
        assert_eq!(plan.to_prefill, 2);
        let s = t.get(id).unwrap();
        assert_eq!(s.next_pos, 4, "next_pos must resume at the reuse point");
        assert_eq!(s.tokens, vec![1, 2, 3, 4], "tokens truncated to the reuse point");
        assert_eq!(pool.descriptors()[s.slot.0].seq_len, 4);
    }

    #[test]
    fn begin_turn_on_divergence_rewinds_to_the_common_prefix() {
        let (mut pool, mut adm, mut t) = rig(1);
        let id = t.open(&mut pool, &mut adm, 1024).unwrap();
        {
            let s = t.get_mut(id).unwrap();
            s.tokens.extend_from_slice(&[1, 2, 3, 4]);
            s.next_pos = 4;
        }
        pool.set_seq_len(t.get(id).unwrap().slot, 4).unwrap();

        let plan = t.begin_turn(&mut pool, id, &[1, 2, 7, 8]).unwrap();
        assert_eq!(plan.reused, 2);
        assert_eq!(plan.to_prefill, 2);
        let s = t.get(id).unwrap();
        assert_eq!(s.tokens, vec![1, 2], "diverged tokens are dropped");
        assert_eq!(s.next_pos, 2);
        assert_eq!(
            pool.descriptors()[s.slot.0].seq_len,
            2,
            "the slot must forget the diverged KV"
        );
    }

    #[test]
    fn begin_turn_on_an_unknown_session_is_an_error_not_a_panic() {
        let (mut pool, mut adm, mut t) = rig(1);
        let id = t.open(&mut pool, &mut adm, 1024).unwrap();
        t.close(&mut pool, &mut adm, id);
        assert!(t.begin_turn(&mut pool, id, &[1, 2]).is_err());
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p hipfire-runtime --lib session_table 2>&1 | tail -20`
Expected: FAIL to compile — `no method named 'begin_turn'`.

- [ ] **Step 3: Write the implementation**

Add `use crate::prefix::{plan_turn, TurnPlan};` to the imports at the top of `crates/hipfire-runtime/src/session_table.rs`, then add this method inside `impl SessionTable`:

```rust
    /// Start a turn: decide how much of `prompt` this session's slot already
    /// holds, rewind to that point, and report what remains to prefill.
    ///
    /// The rewind moves no data. Lowering `seq_len` is enough because KV past
    /// that point is overwritten by the next prefill, and `positions[]` — not
    /// `seq_len` — is what bounds attention per row.
    ///
    /// The caller must then prefill `prompt[plan.reused..]` and append those
    /// tokens to `session.tokens`.
    pub fn begin_turn(
        &mut self,
        pool: &mut SlotPool,
        id: SessionId,
        prompt: &[u32],
    ) -> Result<TurnPlan, String> {
        let session = self
            .sessions
            .get_mut(&id.0)
            .ok_or_else(|| format!("begin_turn: unknown session {}", id.0))?;
        let held = pool.descriptors()[session.slot.0].seq_len as usize;
        let plan = plan_turn(&session.tokens, held, prompt);
        pool.set_seq_len(session.slot, plan.reused)
            .map_err(|e| format!("begin_turn: {e}"))?;
        session.tokens.truncate(plan.reused);
        session.next_pos = plan.reused;
        Ok(plan)
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p hipfire-runtime --lib session_table 2>&1 | grep "test result"`
Expected: `test result: ok. 8 passed; 0 failed`

- [ ] **Step 5: Commit**

```bash
BASE_REF=origin/beta ./scripts/fmt-changed.sh
git add crates/hipfire-runtime/src/session_table.rs
git commit -m "feat(sessions): begin_turn rewinds a slot to the reusable prefix"
```

---

### Task 3: The SP5 gate — reuse must equal cold prefill, token for token

**Files:**
- Create: `crates/hipfire-runtime/examples/test_prefix_cache_equivalence.rs`

**Interfaces:**
- Consumes: `SessionTable::begin_turn` (Task 2); `hipfire_arch_qwen35::forward_slots::{forward_batch_slots_graphed, SlotDecodeGraph, SlotDescStaging}`; `hipfire_arch_qwen35::scheduler::{PendingWork, Scheduler}`; `rdna_compute::slot_pool::SlotPool`.
- Produces: the gate binary; no library API.

**Why this gate:** Tasks 1–2 are unit-tested arithmetic. The thing that can actually be wrong is whether reused KV yields the *same logits* as freshly computed KV. Only a real model answers that.

- [ ] **Step 1: Write the gate**

Create `crates/hipfire-runtime/examples/test_prefix_cache_equivalence.rs`. Model the setup (model load, `SlotPool::new`, arenas, `DeltaNetState`, `SlotDescStaging`, `PrefillBatchScratch`) directly on `crates/hipfire-runtime/examples/demo_multislot_generate.rs`, which already builds every one of those; copy that preamble verbatim, including its `preflight_alloc` accounting, then use this body:

```rust
// Two arms, one slot each, greedy (argmax) sampling.
//
//   REFERENCE: prefill the whole turn-2 prompt cold, decode N tokens.
//   CANDIDATE: prefill turn 1, then begin_turn() against the turn-2 prompt so
//              the shared prefix is reused, prefill only the suffix, decode N.
//
// The two must produce identical token ids. Anything else means reused KV is
// not equivalent to recomputed KV, which would make the prefix cache silently
// change model output — the one failure class that matters.

let turn1: Vec<u32> = tokenizer.encode("The capital of France is Paris. It is known for");
let turn2: Vec<u32> = {
    let mut v = turn1.clone();
    v.extend(tokenizer.encode(" its museums and food. The capital of Italy is"));
    v
};
const DECODE_N: usize = 24;

// ---- REFERENCE arm: cold, full prefill of turn2 ----
let reference = run_arm(&mut gpu, /* prompt */ &turn2, /* warm_with */ None, DECODE_N);

// ---- CANDIDATE arm: turn1 first, then reuse ----
let candidate = run_arm(&mut gpu, &turn2, Some(&turn1), DECODE_N);

assert_eq!(
    reference, candidate,
    "prefix-cache reuse changed the output; reused KV is not equivalent to recomputed KV"
);
println!("{DECODE_N}/{DECODE_N} tokens identical — prefix reuse is output-equivalent");

// ---- negative control: the comparison must be able to fail ----
let mut corrupted = candidate.clone();
corrupted[0] = corrupted[0].wrapping_add(1);
assert_ne!(reference, corrupted, "negative control: comparison is not sensitive");
println!("negative control fired");
println!("ALL CHECKS PASS");
```

`run_arm` prefills `warm_with` (if any) through the scheduler, calls `table.begin_turn(&mut pool, id, prompt)`, prefills `prompt[plan.reused..]`, then greedily decodes `DECODE_N` tokens via `forward_batch_slots_graphed` + `gpu.sample_per_slot` with argmax parameters, returning `Vec<u32>`. It must report `plan.reused` so the run visibly proves reuse happened:

```rust
println!("  arm reused {} of {} prompt tokens", plan.reused, prompt.len());
```

- [ ] **Step 2: Build it**

Run: `cargo build --release -p hipfire-runtime --features deltanet,arch-qwen35 --example test_prefix_cache_equivalence 2>&1 | grep -E "^error" -A 5`
Expected: no output (builds clean).

- [ ] **Step 3: Run the gate**

Run:
```bash
HIPFIRE_MEM_CAP=30G ./scripts/run-bounded.sh \
  ./target/release/examples/test_prefix_cache_equivalence \
  ~/.hipfire/models/qwen3.6-35b-a3b.mq4r
```
Expected: the candidate arm reports a non-zero `reused` count, then `24/24 tokens identical`, `negative control fired`, `ALL CHECKS PASS`, exit 0, and `run-bounded` reports no OOM.

If the arms differ, do **not** relax the assertion. Report it: it means reused KV is not equivalent, which is a defect in SP5 or an unstated assumption in the forward.

- [ ] **Step 4: Confirm the existing gates are untouched**

Run:
```bash
HIPFIRE_MEM_CAP=30G ./scripts/run-bounded.sh \
  ./target/release/examples/test_forward_slots_golden ~/.hipfire/models/qwen3.6-35b-a3b.mq4r 2>&1 | tail -2
./scripts/kernel_resource_gate.sh > /tmp/kr.txt 2>&1 && diff -q /tmp/kr.txt scripts/kernel_resource_gate.beta.txt && echo GATE_OK
```
Expected: `ALL CHECKS PASS`, then `GATE_OK`.

- [ ] **Step 5: Commit**

```bash
BASE_REF=origin/beta ./scripts/fmt-changed.sh
git add crates/hipfire-runtime/examples/test_prefix_cache_equivalence.rs
git commit -m "test(prefix): gate that reused KV is output-equivalent to recomputed KV"
```

---

## Self-Review

**Spec coverage.** §5.1 `SessionStore` LCP → Tasks 1–2 (residency fields are Phase 2, per §11). §6 steps 1, 3 (LCP, rewind, truncate) → Task 2; steps 4–5 (prefill suffix, decode) use the existing scheduler unchanged and are exercised by Task 3. §8 gate #4 (prefix-cache equivalence) → Task 3. §3's invariant is enforced structurally: `begin_turn` only ever *lowers* `seq_len` and truncates tokens, so the worst case is a full re-prefill.

**Not covered here, by design:** §5.2 `SlotSnapshot`, §5.3 `SwapManager`, §5.4 admission extension, §7 failure handling, §8 gates #1–#3 — all Phase 2 (SP6), which depends on this residency model.

**Type consistency.** `TurnPlan { lcp, reused, to_prefill }` is defined in Task 1 and used with those exact field names in Tasks 2 and 3. `plan_turn(cached, seq_len, prompt)` and `begin_turn(pool, id, prompt)` signatures match between definition and use. `SlotId.0` is used as the `descriptors()` index, matching `SlotPool`'s existing tests.
