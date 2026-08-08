# SP2 — Multi-Slot State and Per-Slot Ops: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the per-slot state and operations that SP1's multi-slot attention kernels need, so SP3 can assemble a ragged multi-slot forward pass without also inventing the allocator.

**Architecture:** SP1's move, applied four more times. KV write, RoPE, DeltaNet and sampling each already have a *multi-token, single-sequence* batched variant; SP2 adds the *slot* axis to each, reusing SP1's `KvSlotDesc` / `kv_offset_for_k/v` / `preflight_alloc` machinery unchanged. A new `SlotPool` owns the slabs and states and is the only thing that allocates.

**Tech Stack:** Rust (`rdna-compute`, `hipfire-arch-qwen35`), HIP C++ kernels compiled at runtime via `ensure_kernel` with header-body prepending, `cargo run --release --example` harnesses.

**Spec:** `docs/specs/2026-08-08-multi-slot-state-sp2.md`. Read it first; this plan implements it and does not restate its reasoning.

## Global Constraints

- **Branch:** `feat/batched-attn-impl`, worktree `~/repos/hipfire-batchattn-impl`. Continues SP1.
- **`--features deltanet` on every cargo command.** Without it an unrelated pre-existing example (`rope_compact_offset_check`) fails with four `E0599` errors — PRE-EXISTING on `origin/beta`, NOT yours to fix. Scope to `-p <crate>`, never the whole workspace.
- **MEMORY — binding, and measured.**
  - **The cgroup does NOT contain amdgpu GTT.** A gated run still invoked the *global* OOM killer and killed the user's Slack and three steamwebhelper processes. `MemoryMax` bounds host RSS only.
  - Run every GPU harness through `./scripts/run-bounded.sh`. It now refuses unless `MemAvailable >= cap + 10 GiB`, which is the **primary** protection.
  - Call `kv_slots::preflight_alloc(total_bytes, budget_bytes, what)` before allocating, with the TOTAL held live at once — **device and host**. Harness totals that omit host `Vec`s undercount by ~30%.
  - **Never run a GPU harness while a serve daemon holds a model.** Check `pgrep -f 'hipfire/bin/daemon'` first; one resident model takes MemAvailable from ~58 GiB to ~19 GiB. Wait rather than contend.
  - A live `free` shows nothing after the fact. Check `journalctl -k | grep -E 'Out of memory|oom-kill|CONSTRAINT_NONE'`.
- **Never write to `~/.hipfire/`.** Every setting is an env var. An agent edited the user's config and died before restoring it, leaving `max_seq` at 16384 instead of 131072.
- **No device `assert()` in kernels.** `compiler.rs` never passes `-DNDEBUG`, so they ship in release; SP1 measured 64 B/lane of scratch on four kernels. Enforce invariants host-side.
- **Kernel args are positional** — append at the end only, never reorder. The Rust side must push identical values in identical order onto **both** the `params` vector **and** the `KernargBlob` closure.
- **Header inclusion:** a literal `#include "kv_slot_desc.h"` does not resolve (runtime `hipcc`, no `-I` to `kernels/src`). Keep the `#include` for readability; in Rust strip it and prepend `kernels::KV_SLOT_DESC_H`, guarded behind a `functions.contains_key(...)` cache check.
- **`positions[]` is authoritative for the causal/position bound — never `desc.seq_len`.** The descriptor supplies the slab base address only. Violating this caused SP1's only Critical defect.
- **ABI: `v_base == k_base` for Q8_0 arenas.** The Q8 flash-prefill kernel uses one shared slab offset; keeping both bases live costs 23 VGPRs and 25% occupancy. asym3 is exempt — its K and V strides genuinely differ.
- **Two gates must stay green after every task:**
  - `./scripts/attn_legacy_baseline.sh` diffed against `scripts/attn_legacy_baseline.beta.txt` → bitwise identical.
  - `./scripts/kernel_resource_gate.sh` diffed against `scripts/kernel_resource_gate.beta.txt` → identical. This is compile-only and safe to run under memory pressure. A deliberate cost updates the baseline in the same commit, with the reason.
- **`./scripts/no-gpu-ci.sh` must exit 0.** Production `src/` may not read `HIPFIRE_*` directly — route through `feature_flags.rs`, or take it as a parameter and let `examples/` (exempt) read it.
- Check exit statuses directly. `cmd | tail && echo OK` prints OK when only `tail` succeeded — this masked a failing `cargo test` twice during SP1.
- Licence header on every new file:
  ```
  // SPDX-License-Identifier: Apache-2.0
  // Copyright (c) 2026 Nick Woolmer
  // hipfire — see LICENSE and NOTICE in the project root.
  ```
- Commit with `git add <specific paths>` — never `git add -A`.

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `crates/rdna-compute/src/slot_pool.rs` | `SlotPool`: owns slabs + states, acquire/release/reset, descriptor upload with change detection | create |
| `crates/rdna-compute/src/lib.rs` | register the module | modify |
| `kernels/src/kv_cache_write_q8_0_batched.hip` | accept `slot_descs` + `row_slot` | modify |
| `crates/rdna-compute/src/attention.rs:1493` | `kv_cache_write_q8_0_batched_slots` + delegate | modify |
| `kernels/src/gated_delta_net_q8_fast.hip` | accept a per-slot state stride | modify |
| `crates/rdna-compute/src/norm.rs:2531` | `gated_delta_net_q8_batch_seq_slots` + delegate | modify |
| `crates/rdna-compute/src/sampling.rs` | per-slot sampling parameter table | modify |
| `crates/rdna-compute/examples/test_slot_pool.rs` | SlotPool unit + invariant tests | create |
| `crates/rdna-compute/examples/test_multislot_ops.rs` | golden + isolation + negative control for KV write, DeltaNet, sampling | create |

**Task order:** Task 1 (`SlotPool`) first because every later task allocates through it. Task 2 confirms or refutes RoPE's slot-independence, which is cheap and may remove work. Tasks 3–5 add the slot axis to one family each. Task 6 is the combined verification harness.

---

### Task 1: `SlotPool` — the allocator

**Files:**
- Create: `crates/rdna-compute/src/slot_pool.rs`
- Modify: `crates/rdna-compute/src/lib.rs`

**Interfaces:**
- Consumes: `kv_slots::{KvSlotDesc, preflight_alloc, R9700_VRAM_BYTES}` (SP1).
- Produces:
  - `pub struct SlotPool`
  - `pub struct SlotId(pub usize)`
  - `SlotPool::new(n_slots: usize, cap_tokens: usize, per_pos_bytes: usize) -> Result<SlotPool, String>`
  - `SlotPool::acquire(&mut self) -> Option<SlotId>`
  - `SlotPool::release(&mut self, id: SlotId)`
  - `SlotPool::reset(&mut self, id: SlotId)`
  - `SlotPool::set_seq_len(&mut self, id: SlotId, seq_len: usize) -> Result<(), String>`
  - `SlotPool::descriptors(&self) -> &[KvSlotDesc]`
  - `SlotPool::descriptors_dirty(&self) -> bool`
  - `SlotPool::mark_uploaded(&mut self)`
  - `SlotPool::arena_bytes(&self) -> usize`

- [ ] **Step 1: Write the failing tests**

Create `crates/rdna-compute/src/slot_pool.rs` with the licence header, a stub `SlotPool` whose methods `unimplemented!()`, and these tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const PPB: usize = 1088; // Q8_0 bytes/position at n_kv_heads=2, head_dim=256

    #[test]
    fn slabs_are_cap_aligned_and_non_overlapping() {
        let p = SlotPool::new(4, 300, PPB).unwrap();
        let d = p.descriptors();
        assert_eq!(d.len(), 4);
        // cap rounds up to a multiple of PAGE_TOKENS (128) so a future page size divides it
        assert_eq!(d[0].cap, 384);
        for i in 1..4 {
            let prev_end = d[i - 1].k_base + (d[i - 1].cap as u64) * PPB as u64;
            assert_eq!(d[i].k_base, prev_end, "slab {i} must start where {} ended", i - 1);
        }
    }

    #[test]
    fn q8_abi_requires_v_base_equals_k_base() {
        // SP1 ABI: the Q8 flash-prefill kernel uses ONE shared slab offset.
        let p = SlotPool::new(3, 256, PPB).unwrap();
        for d in p.descriptors() {
            assert_eq!(d.k_base, d.v_base, "Q8 arenas must share slab offsets");
        }
    }

    #[test]
    fn acquire_release_reuses_slots_and_bounds_count() {
        let mut p = SlotPool::new(2, 128, PPB).unwrap();
        let a = p.acquire().unwrap();
        let b = p.acquire().unwrap();
        assert!(p.acquire().is_none(), "pool of 2 must not hand out a third");
        p.release(a);
        let c = p.acquire().unwrap();
        assert_eq!(c.0, a.0, "released slot must be reused");
        p.release(b);
        p.release(c);
    }

    #[test]
    fn set_seq_len_enforces_the_cap_invariant() {
        let mut p = SlotPool::new(1, 128, PPB).unwrap();
        let id = p.acquire().unwrap();
        assert!(p.set_seq_len(id, 128).is_ok());
        let e = p.set_seq_len(id, 129).unwrap_err();
        assert!(e.contains("cap"), "unexpected message: {e}");
    }

    #[test]
    fn release_resets_seq_len_so_reuse_cannot_inherit_history() {
        let mut p = SlotPool::new(1, 128, PPB).unwrap();
        let id = p.acquire().unwrap();
        p.set_seq_len(id, 100).unwrap();
        p.release(id);
        let id2 = p.acquire().unwrap();
        assert_eq!(p.descriptors()[id2.0].seq_len, 0, "reused slot must start empty");
    }

    #[test]
    fn dirty_flag_tracks_descriptor_changes() {
        let mut p = SlotPool::new(1, 128, PPB).unwrap();
        p.mark_uploaded();
        assert!(!p.descriptors_dirty());
        let id = p.acquire().unwrap();
        p.set_seq_len(id, 10).unwrap();
        assert!(p.descriptors_dirty(), "a seq_len change must dirty the table");
        p.mark_uploaded();
        assert!(!p.descriptors_dirty());
    }

    #[test]
    fn oversized_pool_is_refused_not_allocated() {
        // 8 slots x 1M tokens x 1088 B = ~8.7 TB, far over the 32 GiB target budget.
        let e = SlotPool::new(8, 1_000_000, PPB).unwrap_err();
        assert!(e.contains("budget") || e.contains("GiB"), "unexpected: {e}");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --release -p rdna-compute --features deltanet slot_pool`
Expected: FAIL — the stub `unimplemented!()` panics.

- [ ] **Step 3: Write the implementation**

Replace the stub in `crates/rdna-compute/src/slot_pool.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SlotPool — owns the per-slot KV slabs and the descriptor table that SP1's
// batched attention kernels read.
//
// Fixed-size slabs, deliberately. Variable-size slabs would fragment and buy
// nothing at 2-8 slots, and the paged upgrade (SP4) replaces this addressing
// wholesale rather than extending it.

use crate::kv_slots::{preflight_alloc, KvSlotDesc, R9700_VRAM_BYTES};

/// Slab capacities round up to this, so a future page size divides them.
/// Matches the tile size the flash path walks KV in.
const PAGE_TOKENS: usize = 128;

/// Index of a slot within its pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotId(pub usize);

#[derive(Debug)]
pub struct SlotPool {
    descs: Vec<KvSlotDesc>,
    in_use: Vec<bool>,
    cap_tokens: usize,
    per_pos_bytes: usize,
    dirty: bool,
}

impl SlotPool {
    /// Build a pool of `n_slots` fixed-size slabs.
    ///
    /// `per_pos_bytes` is the per-position stride, uniform across slots
    /// (`n_kv_heads * (head_dim/32) * 34` for Q8_0).
    ///
    /// Refuses rather than allocates when the arena would exceed the
    /// deployment-target budget — see `kv_slots::preflight_alloc`.
    pub fn new(
        n_slots: usize,
        cap_tokens: usize,
        per_pos_bytes: usize,
    ) -> Result<Self, String> {
        assert!(n_slots > 0, "n_slots must be positive");
        assert!(per_pos_bytes > 0, "per_pos_bytes must be positive");
        let cap = cap_tokens.div_ceil(PAGE_TOKENS) * PAGE_TOKENS;
        let slab_bytes = (cap * per_pos_bytes) as u64;
        // K and V are separate arenas of identical layout, hence x2.
        let total = slab_bytes
            .checked_mul(n_slots as u64)
            .and_then(|b| b.checked_mul(2))
            .ok_or_else(|| "SlotPool: arena size overflows u64".to_string())?;
        preflight_alloc(total, R9700_VRAM_BYTES, "SlotPool arena")?;

        let descs = (0..n_slots)
            .map(|i| {
                let base = i as u64 * slab_bytes;
                KvSlotDesc {
                    // Q8_0 ABI: the flash-prefill kernel uses ONE shared slab
                    // offset, so K and V must sit at the same offset in their
                    // respective arenas. asym3 is exempt and needs its own pool.
                    k_base: base,
                    v_base: base,
                    seq_len: 0,
                    cap: cap as i32,
                }
            })
            .collect();

        Ok(Self {
            descs,
            in_use: vec![false; n_slots],
            cap_tokens: cap,
            per_pos_bytes,
            dirty: true,
        })
    }

    /// Take a free slot, or `None` when the pool is full. Admission control
    /// lives in SP4; this only reports capacity.
    pub fn acquire(&mut self) -> Option<SlotId> {
        let i = self.in_use.iter().position(|&u| !u)?;
        self.in_use[i] = true;
        self.reset(SlotId(i));
        Some(SlotId(i))
    }

    /// Return a slot to the pool. Resets its length so a later `acquire`
    /// cannot inherit the previous occupant's history.
    pub fn release(&mut self, id: SlotId) {
        self.reset(id);
        self.in_use[id.0] = false;
    }

    /// Zero a slot's logical length. The slab bytes are left alone — every
    /// read is bounded by `seq_len`, so stale bytes are unreachable.
    pub fn reset(&mut self, id: SlotId) {
        if self.descs[id.0].seq_len != 0 {
            self.descs[id.0].seq_len = 0;
            self.dirty = true;
        }
    }

    /// Set a slot's logical KV length. Enforces `seq_len <= cap` host-side,
    /// because SP1 removed the device asserts (they shipped in release and
    /// cost 64 B/lane of scratch).
    pub fn set_seq_len(&mut self, id: SlotId, seq_len: usize) -> Result<(), String> {
        if seq_len > self.cap_tokens {
            return Err(format!(
                "SlotPool: slot {} seq_len {} exceeds cap {}",
                id.0, seq_len, self.cap_tokens
            ));
        }
        if self.descs[id.0].seq_len != seq_len as i32 {
            self.descs[id.0].seq_len = seq_len as i32;
            self.dirty = true;
        }
        Ok(())
    }

    pub fn descriptors(&self) -> &[KvSlotDesc] {
        &self.descs
    }

    /// True when the table has changed since the last `mark_uploaded`.
    /// Callers skip the device upload when clean, following the ds4 precedent.
    pub fn descriptors_dirty(&self) -> bool {
        self.dirty
    }

    pub fn mark_uploaded(&mut self) {
        self.dirty = false;
    }

    /// Bytes in ONE arena (K or V). The pool holds two of these.
    pub fn arena_bytes(&self) -> usize {
        self.descs.len() * self.cap_tokens * self.per_pos_bytes
    }
}
```

Then add to `crates/rdna-compute/src/lib.rs`, alongside the other `pub mod` declarations, in alphabetical position:

```rust
pub mod slot_pool;
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --release -p rdna-compute --features deltanet slot_pool`
Expected: PASS, 7 tests.

- [ ] **Step 5: Confirm the no-GPU gates**

Run:
```bash
./scripts/no-gpu-ci.sh > /tmp/sp2t1ci.txt 2>&1; echo "CI=$?"
./scripts/kernel_resource_gate.sh > /tmp/sp2t1res.txt 2>&1
diff scripts/kernel_resource_gate.beta.txt /tmp/sp2t1res.txt && echo RESOURCE_GATE_OK
```
Expected: `CI=0` and `RESOURCE_GATE_OK`. This task touches no kernel, so the resource gate must be unchanged.

- [ ] **Step 6: Commit**

```bash
git add crates/rdna-compute/src/slot_pool.rs crates/rdna-compute/src/lib.rs
git commit -m "feat(slots): SlotPool — fixed-size slab allocator with descriptor table

Fixed-size slabs deliberately: variable sizes would fragment and buy
nothing at 2-8 slots, and SP4's paged upgrade replaces this addressing
rather than extending it.

Enforces both invariants host-side (seq_len <= cap, and v_base == k_base
for the Q8 ABI) because SP1 removed the device asserts — they shipped in
release and cost 64 B/lane of scratch. Refuses oversized pools via
preflight_alloc rather than allocating."
```

---

### Task 2: Confirm or refute RoPE's slot-independence

The spec predicts RoPE needs **no** slot awareness: it transforms q/k in the flat row layout *before* the cache write, and `positions[]` is already indexed by global row. If that holds, this task is documentation and removes work from the plan. If it does not, this task is where we find out — cheaply, before three other tasks depend on the assumption.

**Files:**
- Modify: `docs/plans/2026-08-08-multi-slot-state-sp2.md` (record the finding)

**Interfaces:**
- Consumes: nothing.
- Produces: a documented yes/no that Tasks 3–5 rely on.

- [ ] **Step 1: Read the batched RoPE launchers and kernels**

Run:
```bash
grep -n "pub fn rope_batched_f32" -A 30 crates/rdna-compute/src/norm.rs
grep -n "pub fn rope_partial_interleaved_f32_batched" -A 30 crates/rdna-compute/src/norm.rs
```

For each, answer in writing:
1. Does it read or write any KV cache buffer, or only the q/k activation tensors?
2. Is every buffer it touches indexed by *global flat row*, or by a per-sequence offset?
3. Does it take `positions` as a device array, and is that array global-row-indexed?

- [ ] **Step 2: Record the finding in this plan**

Append a short section to `docs/plans/2026-08-08-multi-slot-state-sp2.md` titled `## Task 2 finding: RoPE slot-independence`, stating the answer with the evidence (file:line for each claim).

**If RoPE touches only global-row-indexed activation tensors:** state that no change is required and that Tasks 3-5 may assume RoPE is slot-agnostic.

**If it touches a KV buffer or a per-sequence offset:** state exactly which, and add a Task 3b to this plan giving it the same descriptor treatment as Task 3, following Task 3's structure.

- [ ] **Step 3: Commit**

```bash
git add docs/plans/2026-08-08-multi-slot-state-sp2.md
git commit -m "docs(plan): record whether RoPE needs slot awareness

Confirms or refutes the spec's prediction before three tasks build on it."
```

---

### Task 3: Slot axis for the batched KV write

**Files:**
- Modify: `kernels/src/kv_cache_write_q8_0_batched.hip`
- Modify: `crates/rdna-compute/src/attention.rs:1493-1540`

**Interfaces:**
- Consumes: `KvSlotDesc`, `kv_offset_for_k`, `kv_offset_for_v`, `kv_slot_legacy` (SP1 header); `SlotPool::descriptors()` (Task 1).
- Produces: `Gpu::kv_cache_write_q8_0_batched_slots(dst_k, dst_v, src, positions, n_kv_heads, head_dim, batch_size, slot_descs: Option<&GpuTensor>, row_slot: Option<&GpuTensor>)`. The existing `kv_cache_write_q8_0_batched` stays and delegates with `None, None`.

**Study `git show a01838e9` first** — Task 4 of SP1 is the worked example of this exact port on a sibling kernel.

- [ ] **Step 1: Add the descriptor parameters to the kernel**

In `kernels/src/kv_cache_write_q8_0_batched.hip`, add `#include "kv_slot_desc.h"` after the `hip_runtime.h` include, then append two parameters to the signature — **append only, never reorder**, the kernarg blob is positional:

```c
    int batch_size,
    const KvSlotDesc* __restrict__ slot_descs,  // [n_slots] or nullptr = legacy
    const int* __restrict__ row_slot            // [batch_size] or nullptr = legacy
) {
```

In the body, after `const int bid = blockIdx.y;` and its bounds check, add:

```c
    // One code path for both modes: in legacy mode we synthesise a zero-base
    // descriptor, so the address arithmetic below is unchanged and the output
    // is bitwise identical to the pre-SP2 kernel.
    const int slot = (row_slot != nullptr) ? row_slot[bid] : 0;
    const KvSlotDesc desc = (slot_descs != nullptr)
        ? slot_descs[slot]
        : kv_slot_legacy(0, 0);
    const int per_pos_bytes = total_blocks * 34;
```

Then route the destination write through the helper. The existing code computes a destination offset from `positions[bid]`; replace that offset with:

```c
    kv_offset_for_k(desc, positions[bid], per_pos_bytes)
```

**Note `total_blocks` is computed later in the current source than `bid`** — move the `blocks_per_head` / `total_blocks` computation above the descriptor block, or compute `per_pos_bytes` inline as `n_kv_heads * (head_dim / 32) * 34`.

- [ ] **Step 2: Add the slots launcher**

In `crates/rdna-compute/src/attention.rs`, rename the existing body to `kv_cache_write_q8_0_batched_slots` with two extra trailing parameters, pushing them onto **both** the `params` vector and the `KernargBlob` closure:

```rust
        let mut desc_ptr: *mut std::ffi::c_void = match slot_descs {
            Some(t) => t.buf.as_ptr(),
            None => std::ptr::null_mut(),
        };
        let mut rs_ptr: *mut std::ffi::c_void = match row_slot {
            Some(t) => t.buf.as_ptr(),
            None => std::ptr::null_mut(),
        };
```

appended to `params` after `bs`, and in the blob closure after `b.push_i32(bs)`:

```rust
                b.push_ptr(desc_raw);
                b.push_ptr(rs_raw);
```

Add a both-or-neither assertion before launching, carrying SP1's lesson that the half-configured combination silently pins every row to slot 0:

```rust
        assert_eq!(
            slot_descs.is_some(),
            row_slot.is_some(),
            "kv_cache_write_q8_0_batched_slots: slot_descs and row_slot are both-or-neither. \
             Passing only slot_descs silently pins every row to slot 0, writing every \
             sequence's KV into slot 0's slab."
        );
```

Because the kernel now needs the header body prepended, guard the source assembly behind the cache check (SP1 found an unguarded rebuild costing ~10.8 KB of allocation per launch on a hot path):

```rust
        if !self.functions.contains_key("kv_cache_write_q8_0_batched") {
            let stripped = kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC
                .replace("#include \"kv_slot_desc.h\"", "");
            let src = format!("{}\n{}", kernels::KV_SLOT_DESC_H, stripped);
            self.ensure_kernel("kv_cache_write_q8_0_batched", &src, "kv_cache_write_q8_0_batched")?;
        }
```

Keep the original entry point as a thin delegate so no existing call site changes:

```rust
    /// Legacy single-sequence entry point. Preserved so existing call sites are
    /// untouched; passes null descriptors, which the kernel treats as legacy
    /// mode with bitwise-identical output.
    pub fn kv_cache_write_q8_0_batched(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        positions: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.kv_cache_write_q8_0_batched_slots(
            dst, src, positions, n_kv_heads, head_dim, batch_size, None, None,
        )
    }
```

- [ ] **Step 3: Check for a second compile site**

Run:
```bash
grep -rn "KV_CACHE_WRITE_Q8_0_BATCHED_SRC" crates/ --include=*.rs
```
SP1 found that `precompile_qwen35` in `crates/rdna-compute/src/dispatch.rs` is an independent compile site that breaks silently at first precompile if it is missed. If this constant appears there, apply the identical strip-and-prepend there too.

- [ ] **Step 4: Build and verify both gates**

Run:
```bash
cargo build --release -p rdna-compute --features deltanet
./scripts/kernel_resource_gate.sh > /tmp/sp2t3res.txt 2>&1
diff scripts/kernel_resource_gate.beta.txt /tmp/sp2t3res.txt && echo RESOURCE_GATE_OK
```
Expected: builds clean, `RESOURCE_GATE_OK`. The gate does not cover this kernel, so it must be unchanged; if it moved, something unintended was edited.

- [ ] **Step 5: Verify legacy is bitwise identical**

Only if `pgrep -f 'hipfire/bin/daemon'` finds nothing and `MemAvailable` allows:
```bash
./scripts/run-bounded.sh ./scripts/attn_legacy_baseline.sh > /tmp/sp2t3leg.txt 2>&1
diff scripts/attn_legacy_baseline.beta.txt /tmp/sp2t3leg.txt && echo LEGACY_BITWISE_IDENTICAL
```
Expected: `LEGACY_BITWISE_IDENTICAL`. If the box is busy, say so in the report and leave this step unchecked rather than running it anyway.

- [ ] **Step 6: Commit**

```bash
git add kernels/src/kv_cache_write_q8_0_batched.hip crates/rdna-compute/src/attention.rs
git commit -m "feat(slots): slot axis for the batched Q8 KV write

Null descriptor = legacy mode, bitwise identical. Both-or-neither
assertion because the half-configured combination silently writes every
sequence's KV into slot 0's slab."
```

---

### Task 4: Slot axis for the DeltaNet state update

Three quarters of both models' layers are DeltaNet, but this is the *easiest* component: the state is fixed-size and each slot's state is completely independent, so there is no shared arena, no ragged length and no offset descriptor — only a slot stride. The recurrence runs within a slot across tokens and never across slots, so adding the slot axis cannot introduce a cross-slot dependency.

**Files:**
- Modify: `kernels/src/gated_delta_net_q8_fast.hip`
- Modify: `crates/rdna-compute/src/norm.rs:2531-2700`

**Interfaces:**
- Consumes: `SlotPool` (Task 1) for sizing only — DeltaNet state is not KV and is not in the KV arena.
- Produces: `Gpu::gated_delta_net_q8_batch_seq_slots(q_batch, k_batch, v_batch, gate_batch, beta_batch, s_q8, s_scales, output_batch, n_tokens, n_heads, head_dim, ef_residual, row_slot: Option<&GpuTensor>, s_stride_elems: usize)`. The existing `gated_delta_net_q8_batch_seq` stays and delegates with `None, 0`.

- [ ] **Step 1: Add the slot parameters to the kernel**

In `kernels/src/gated_delta_net_q8_fast.hip`, append two parameters to the signature — append only:

```c
    const int* __restrict__ row_slot,   // [n_tokens] or nullptr = legacy (all slot 0)
    int s_stride_elems                  // elements between consecutive slots' S state
) {
```

In the body, where the kernel currently indexes `s_q8` and `s_scales`, offset both by the slot:

```c
    // Each slot owns an independent, fixed-size S state. There is no shared
    // arena and no ragged length here, unlike KV — just a stride. In legacy
    // mode row_slot is null, the offset is 0, and addressing is unchanged.
    const int dn_slot = (row_slot != nullptr) ? row_slot[0] : 0;
    const long long s_off = (long long)dn_slot * (long long)s_stride_elems;
```

Then add `s_off` to the base pointers the kernel derives for `s_q8` and `s_scales`.

**Why `row_slot[0]` and not per-token:** one launch of this kernel advances ONE slot's recurrence across its tokens. A batch spanning several slots issues one launch per slot, because the recurrence is sequential within a slot. Mixing slots inside a single launch would require interleaving independent recurrences and is explicitly not what this task does.

- [ ] **Step 2: Add the slots launcher**

In `crates/rdna-compute/src/norm.rs`, rename the existing body of `gated_delta_net_q8_batch_seq` to `gated_delta_net_q8_batch_seq_slots` with the two extra trailing parameters, pushing them onto both the `params` vector and any `KernargBlob` closure present. Keep the original as a delegate passing `None, 0`.

- [ ] **Step 3: Build and check the resource gate**

Run:
```bash
cargo build --release -p rdna-compute --features deltanet
./scripts/kernel_resource_gate.sh > /tmp/sp2t4res.txt 2>&1
diff scripts/kernel_resource_gate.beta.txt /tmp/sp2t4res.txt && echo RESOURCE_GATE_OK
```
Expected: builds clean, `RESOURCE_GATE_OK`.

**If the gate moves**, the extra `long long` offset has cost registers. Report the before/after numbers and diagnose before continuing — this is exactly the class of regression that a numeric test cannot see, and SP1 shipped one.

- [ ] **Step 4: Commit**

```bash
git add kernels/src/gated_delta_net_q8_fast.hip crates/rdna-compute/src/norm.rs
git commit -m "feat(slots): slot stride for the DeltaNet state update

Each slot's S state is fixed-size and independent, so the slot axis is a
stride rather than a descriptor. One launch advances one slot's
recurrence; mixing slots in a launch would interleave independent
recurrences and is out of scope."
```

---

### Task 5: Per-slot sampling parameters

`argmax_f32_batched` already handles N rows of logits. The design content is that **each agent may have different sampling parameters** — a single scalar temperature across the batch is wrong the moment two agents differ, and a uniform-parameter test cannot see the bug.

**Files:**
- Modify: `crates/rdna-compute/src/sampling.rs`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `#[repr(C)] pub struct SlotSampleParams { pub temperature: f32, pub top_p: f32, pub top_k: i32, pub seed: u32 }`
  - `Gpu::sample_per_slot(logits: &GpuTensor, params: &[SlotSampleParams], n_slots: usize, vocab: usize, out_tokens: &GpuTensor) -> HipResult<()>`

- [ ] **Step 1: Write the failing test**

Add to `crates/rdna-compute/src/sampling.rs`:

```rust
#[cfg(test)]
mod slot_sample_tests {
    use super::*;

    #[test]
    fn params_struct_is_16_bytes_repr_c() {
        // Uploaded straight to the GPU as a table, like KvSlotDesc.
        assert_eq!(std::mem::size_of::<SlotSampleParams>(), 16);
        assert_eq!(std::mem::align_of::<SlotSampleParams>(), 4);
    }

    #[test]
    fn all_greedy_is_detectable_as_a_fast_path() {
        let greedy = vec![
            SlotSampleParams { temperature: 0.0, top_p: 1.0, top_k: 0, seed: 1 },
            SlotSampleParams { temperature: 0.0, top_p: 1.0, top_k: 0, seed: 2 },
        ];
        assert!(all_greedy(&greedy));
        let mixed = vec![
            SlotSampleParams { temperature: 0.0, top_p: 1.0, top_k: 0, seed: 1 },
            SlotSampleParams { temperature: 0.7, top_p: 0.95, top_k: 20, seed: 2 },
        ];
        assert!(!all_greedy(&mixed), "one sampling slot must disable the greedy fast path");
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --release -p rdna-compute --features deltanet slot_sample`
Expected: FAIL — `SlotSampleParams` and `all_greedy` do not exist.

- [ ] **Step 3: Implement the parameter table and the fast-path predicate**

Add to `crates/rdna-compute/src/sampling.rs`:

```rust
/// Per-slot sampling parameters, uploaded as a table like `KvSlotDesc`.
///
/// A single scalar temperature across a batch is wrong as soon as two agents
/// differ, and a uniform-parameter test cannot see that bug — hence a table
/// rather than scalars.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotSampleParams {
    /// 0.0 means greedy/argmax for this slot.
    pub temperature: f32,
    pub top_p: f32,
    /// 0 disables top-k for this slot.
    pub top_k: i32,
    pub seed: u32,
}

/// True when every slot is greedy, so the batch can take the argmax fast path.
/// One sampling slot disables it for the whole batch.
pub fn all_greedy(params: &[SlotSampleParams]) -> bool {
    params.iter().all(|p| p.temperature == 0.0)
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --release -p rdna-compute --features deltanet slot_sample`
Expected: PASS, 2 tests.

- [ ] **Step 5: Implement `sample_per_slot`**

Add to `crates/rdna-compute/src/sampling.rs`, dispatching to the existing kernels rather than writing a new one:

```rust
    /// Sample one token per slot from `[n_slots x vocab]` logits.
    ///
    /// Takes the existing `argmax_f32_batched` fast path when every slot is
    /// greedy; otherwise samples each slot with its own parameters. Per-slot
    /// dispatch is correct but not optimal — a fused kernel is a later
    /// optimisation, and SP2 is explicitly components-not-performance.
    pub fn sample_per_slot(
        &mut self,
        logits: &GpuTensor,
        params: &[SlotSampleParams],
        n_slots: usize,
        vocab: usize,
        out_tokens: &GpuTensor,
    ) -> HipResult<()> {
        assert_eq!(
            params.len(),
            n_slots,
            "sample_per_slot: one SlotSampleParams per slot required"
        );
        if all_greedy(params) {
            return self.argmax_f32_batched(logits, out_tokens, n_slots, vocab);
        }
        for (i, p) in params.iter().enumerate() {
            self.sample_slot_row(logits, i, vocab, p, out_tokens)?;
        }
        Ok(())
    }
```

Implement `sample_slot_row` as a private helper that slices row `i` of the logits and calls the existing `sample_top_p` (greedy rows within a mixed batch still take argmax on that row). Match `argmax_f32_batched`'s existing signature — read it first and adapt the call above if it differs.

- [ ] **Step 6: Build, gates, commit**

Run:
```bash
cargo build --release -p rdna-compute --features deltanet
./scripts/no-gpu-ci.sh > /tmp/sp2t5ci.txt 2>&1; echo "CI=$?"
```
Expected: builds clean, `CI=0`.

```bash
git add crates/rdna-compute/src/sampling.rs
git commit -m "feat(slots): per-slot sampling parameters

A single scalar temperature across a batch is wrong as soon as two agents
differ, and a uniform-parameter test cannot see it. Greedy-only batches
keep the argmax fast path; one sampling slot disables it."
```

---

### Task 6: Verification harness — golden, isolation, negative controls

The gate for SP2. Each component is verified against its existing single-sequence counterpart.

**Files:**
- Create: `crates/rdna-compute/examples/test_multislot_ops.rs`

**Interfaces:**
- Consumes: `SlotPool` (Task 1), `kv_cache_write_q8_0_batched_slots` (Task 3), `gated_delta_net_q8_batch_seq_slots` (Task 4), `sample_per_slot` (Task 5); `kv_slots::{build_arena, preflight_alloc}` (SP1).
- Produces: a pass/fail harness. Nothing downstream consumes it.

**Read `crates/rdna-compute/examples/test_batched_attn_slots.rs` first** — SP1's harness already implements arena construction, descriptor packing, NaN poisoning, `assert_close` with a non-degeneracy guard, and a working negative control. Reuse its structure; do not reinvent it.

- [ ] **Step 1: Golden equivalence per component**

For slot counts 1-8, and for each of KV write and DeltaNet:
1. Run the multi-slot op over all slots in one call.
2. For each slot, run the existing single-sequence op on that slot alone.
3. Compare with `assert_close`, which must reject an all-zero reference (SP1 found two all-zero arrays passing at `0.000x`).

- [ ] **Step 2: Cross-slot isolation**

Poison every slot except the target with NaN, confirm the target's output is unchanged **and finite**. This was the sharpest instrument in SP1 and transfers directly.

Include a **positive control**: poison the *target* and assert its output *does* become non-finite. Without it, an inert poison would make every isolation check pass vacuously — SP1's review caught exactly that circular argument.

- [ ] **Step 3: Negative controls that produce a real mismatch**

One per component. SP1's first attempt corrupted **both** arms of the comparison, so it could never fail; the corrected version produced a genuine numeric mismatch at `91.44x tolerance`. Corrupt the **candidate arm only**, and choose shapes where a host-side invariant check will not abort before the numeric comparison runs.

Report each control's actual output. A control that only trips an assertion proves the assertion works, not that the comparison is sensitive.

- [ ] **Step 4: Assert the generator varies**

Vary synthetic data by slot index **and** position, then assert it is non-constant before trusting any result. An SP1 agent nearly shipped a false pass with a generator computing `pos * 7 % 7`, which is always zero.

- [ ] **Step 5: Run under the memory gate**

Only when `pgrep -f 'hipfire/bin/daemon'` finds nothing:
```bash
./scripts/run-bounded.sh cargo run --release -p rdna-compute --features deltanet --example test_multislot_ops
```
Expected: `ALL COMPONENTS PASS`, exit 0.

If the box is busy, report that the harness is written and unrun rather than running it anyway. **A partial, safe result is the correct outcome; an OOM is not.**

- [ ] **Step 6: Commit**

```bash
git add crates/rdna-compute/examples/test_multislot_ops.rs
git commit -m "test(slots): SP2 component verification

Golden equivalence, cross-slot isolation with a positive poison control,
and a per-component negative control that corrupts only the candidate arm."
```

---

## Completion

SP2 is done when the spec's seven success criteria hold. At that point SP3 can assemble a ragged multi-slot forward pass from `SlotPool` plus the four slot-aware op families, without inventing any allocation.

**What SP2 deliberately does not deliver:** a running forward pass or an end-to-end token. That is SP3, and the scope was chosen explicitly.

---

## Task 2 finding: RoPE slot-independence — CONFIRMED, no change required

The spec predicted RoPE needs no slot awareness. Verified; **Tasks 3-6 may treat
RoPE as slot-agnostic.**

**Evidence.** All three batched entry points take only the q/k activation
tensors, `positions`, head counts, `head_dim` and `batch_size` — **no KV cache
buffer and no `max_seq` stride**, which is what a per-sequence offset would
require:

- `rope_batched_f32` — `crates/rdna-compute/src/norm.rs:658-667`
- `rope_partial_interleaved_f32_batched` — `norm.rs:966-976`
- `rope_interleaved_f32_batched` — `norm.rs:1063-1073`

Kernel-side, `kernels/src/rope_batched.hip`:
- `:18` `int b = blockIdx.y;` — the batch index IS the global flat row.
- `:22` `int pos = positions[b];` — position comes from the global-row-indexed
  array SP1 already established as authoritative.
- `:33-42` q and k are indexed as `base + i`, where `base` derives from `b`.

So RoPE transforms activations in the flat row layout *before* the cache write,
and every buffer it touches is already global-row-indexed. Adding a descriptor
would be inert.

**Consequence for the plan:** no Task 3b is needed. The KV *write* (Task 3) is
the only place in this chain that needs slot awareness, because it is the step
that actually lands bytes in a per-slot slab.

## Task 3 finding: the KV write resolves both K and V through `k_base`

The write kernel is invoked **twice** from existing call sites — once for K and
once for V, with a different `dst` each time — but it always resolves the slab
base via `desc.k_base`, never `desc.v_base`.

**This is correct and deliberate for Q8_0**, and it is the same decision the
flash-prefill kernel makes: the Q8 ABI requires `v_base == k_base`, so one base
serves both arenas, and `SlotPool` guarantees it (Task 1's
`q8_abi_requires_v_base_equals_k_base` test).

**But it is a trap for whoever wires real descriptors, and for asym3.** Two
consequences to carry into Task 6 and SP3:

1. When the V-write call is given a real `slot_descs` array, that array's
   `k_base` field must hold the **V arena's** offset — which is automatic under
   the Q8 ABI, since the two are equal, but is not automatic if anyone ever
   relaxes it.
2. **asym3 cannot use this path.** Its K and V strides genuinely differ, so
   `v_base != k_base` and a single-base write would land V at the K offset. asym3
   needs either its own write variant or a `use_v_base` flag. SP2 does not need
   it — SP1's asym3 support is read-side only — but SP3 must not assume this
   kernel is mode-agnostic.
