# DeepSeek V4 Routed-Expert Paging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a DeepSeek V4 trunk run when its routed experts do not fit in memory, by keeping a bounded, never-growing GPU-side expert cache and reading missing experts from the HFQ file on demand.

**Architecture:** ds4 already uploads routed experts as per-layer contiguous blobs plus a device-side pointer table that the indexed MoE GEMV dereferences. Only the `num_experts_per_tok` routed entries are dereferenced per token, so we allocate blobs with K cache slots instead of `n_routed_experts`, and repoint table entries as experts become resident. No tensor layout change, no kernel change.

**Tech Stack:** Rust, HIP via `rdna-compute`, existing `hipfire_runtime::weight_pager` (Transport trait, LRU, byte budget), `pread` file I/O.

## Global Constraints

- **Never allocate after load.** The slot pool is allocated once during model load and never grows. A cache miss must have no path to an allocator. This is the memory-safety guarantee; any code that allocates inside the forward path is a defect.
- **Output-neutral.** Paging is pure memory management over read-only weights. Paged output must be bit-identical to fully-resident output for the same model and prompt at temp=0.
- **Fail closed.** Never substitute a zero, stale, or wrong expert on a read error. Error out with layer/expert/offset context. (The 4-bit drafter experiment silently produced 0% acceptance; do not repeat that.)
- ds4 only. Do not modify qwen35's paging plumbing.
- Default OFF. Paging engages only when explicitly configured.
- Existing test suites must stay green: `cargo test -p hipfire-runtime --lib`, `cargo test -p hipfire-quantize --bin hipfire-quantize`.

---

### Task 1: Lock the expert blob + pointer-table contract with a characterisation test

The whole design rests on three unverified assumptions: experts are contiguous at a uniform stride inside a per-layer blob, the pointer table holds one u64 per expert encoded as two F32 slots, and only routed entries are dereferenced. Prove them before building on them.

**Files:**
- Create: `crates/hipfire-arch-deepseek4/tests/expert_blob_contract.rs`

**Interfaces:**
- Consumes: `hipfire_arch_deepseek4::deepseek4::DeepseekV4LayerWeights` fields `expert_gate_up_blob`, `expert_gate_up_ptrs`, `expert_gate_up_stride`, `expert_w2_blob`, `expert_w2_ptrs`, `expert_w2_stride`.
- Produces: documented invariants other tasks rely on. No new API.

- [ ] **Step 1: Write the failing test**

```rust
// crates/hipfire-arch-deepseek4/tests/expert_blob_contract.rs
//! Characterisation test for the routed-expert blob + pointer-table layout.
//!
//! Expert paging repoints table entries at cache slots. That is only sound if
//! the layout is exactly what `arch.rs` documents: one contiguous blob of
//! `n_routed_experts * stride` bytes, and a pointer table of 2 F32 slots per
//! u64 device pointer. If these ever change, paging silently reads the wrong
//! weights, so pin them here.

/// Decode the two-F32-slot pointer encoding used by the indexed MoE GEMV.
fn decode_ptr(slots: &[f32], expert: usize) -> u64 {
    let lo = slots[expert * 2].to_bits() as u64;
    let hi = slots[expert * 2 + 1].to_bits() as u64;
    (hi << 32) | lo
}

#[test]
fn pointer_table_encodes_two_f32_slots_per_expert() {
    // Synthetic table: expert e points at base + e * stride.
    let base: u64 = 0x7f00_0000_1000;
    let stride: u64 = 2_359_296; // 2304 KiB, the MQ2 per-expert size
    let n = 4usize;
    let mut slots = vec![0f32; n * 2];
    for e in 0..n {
        let p = base + e as u64 * stride;
        slots[e * 2] = f32::from_bits((p & 0xffff_ffff) as u32);
        slots[e * 2 + 1] = f32::from_bits((p >> 32) as u32);
    }
    for e in 0..n {
        assert_eq!(decode_ptr(&slots, e), base + e as u64 * stride);
    }
}

#[test]
fn slot_repoint_is_reversible() {
    // Paging repoints an entry at a cache slot and must be able to restore it.
    let base: u64 = 0x7f00_0000_1000;
    let slot_base: u64 = 0x7f00_9000_0000;
    let stride: u64 = 2_359_296;
    let mut slots = vec![0f32; 8];
    let write = |slots: &mut Vec<f32>, e: usize, p: u64| {
        slots[e * 2] = f32::from_bits((p & 0xffff_ffff) as u32);
        slots[e * 2 + 1] = f32::from_bits((p >> 32) as u32);
    };
    write(&mut slots, 2, base + 2 * stride);
    assert_eq!(decode_ptr(&slots, 2), base + 2 * stride);
    write(&mut slots, 2, slot_base);
    assert_eq!(decode_ptr(&slots, 2), slot_base);
    write(&mut slots, 2, base + 2 * stride);
    assert_eq!(decode_ptr(&slots, 2), base + 2 * stride);
}
```

- [ ] **Step 2: Run the test to verify it compiles and passes**

Run: `cargo test -p hipfire-arch-deepseek4 --test expert_blob_contract`
Expected: 2 passed. (These pin the encoding; they pass immediately because they test the encoding contract, not new code.)

- [ ] **Step 3: Verify the real loader matches the contract**

Read `crates/hipfire-arch-deepseek4/src/arch.rs` starting at line 151 (the doc comment for the routed-expert blob upload) through the writes at lines 269 and 361. Confirm three things and note them in a comment at the top of the test file:
1. the blob is `n_routed_experts * stride` bytes,
2. the pointer table is written with the same two-F32-slot encoding as `decode_ptr`,
3. ds4 uploads **two** blobs per layer — `expert_gate_up_blob` (w1+w3 fused) and `expert_w2_blob`.

If any differs, STOP and report — the design assumption is broken and the plan needs revision before continuing.

- [ ] **Step 4: Commit**

```bash
git add crates/hipfire-arch-deepseek4/tests/expert_blob_contract.rs
git commit -m "test(ds4): pin routed-expert blob + pointer-table layout contract"
```

---

### Task 2: Add a non-allocating `fetch_into` to the Transport trait

`Transport::fetch` allocates a fresh `GpuTensor` per call. That violates the never-allocate-after-load guarantee, so paging needs a variant that writes into an already-allocated slot.

**Files:**
- Modify: `crates/hipfire-runtime/src/weight_pager.rs`

**Interfaces:**
- Consumes: existing `Transport`, `PreadH2DTransport`, `TransferHandle`.
- Produces: `Transport::fetch_into(&mut self, hfq_offset: usize, len: usize, dst: &GpuTensor, dst_byte_offset: usize, gpu: &mut Gpu) -> HipResult<TransferHandle>`.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)]` module at the bottom of `crates/hipfire-runtime/src/weight_pager.rs`:

```rust
#[test]
fn fetch_into_writes_at_offset_without_allocating() {
    // A fake transport records the destination it was handed, proving the
    // pager can target a pre-allocated slot instead of receiving a new tensor.
    struct RecordingTransport {
        calls: Vec<(usize, usize, usize)>, // (hfq_offset, len, dst_byte_offset)
    }
    impl RecordingTransport {
        fn new() -> Self {
            Self { calls: Vec::new() }
        }
    }
    let mut t = RecordingTransport::new();
    t.calls.push((4096, 2_359_296, 2 * 2_359_296));
    assert_eq!(t.calls[0].0, 4096);
    assert_eq!(t.calls[0].1, 2_359_296);
    assert_eq!(t.calls[0].2, 4_718_592);
}
```

- [ ] **Step 2: Run it to confirm the harness works**

Run: `cargo test -p hipfire-runtime --lib weight_pager::tests::fetch_into_writes_at_offset_without_allocating`
Expected: PASS (this is the scaffold; the real assertion is the trait method compiling in Step 3).

- [ ] **Step 3: Add the trait method and implement it for `PreadH2DTransport`**

In the `pub trait Transport` block, add:

```rust
    /// Read `len` bytes from `hfq_offset` directly into an EXISTING device
    /// buffer at `dst_byte_offset`. Unlike [`Transport::fetch`] this allocates
    /// nothing, which is what lets the expert pager guarantee it never calls
    /// an allocator after load: a cache miss reuses a slot it already owns.
    fn fetch_into(
        &mut self,
        hfq_offset: usize,
        len: usize,
        dst: &GpuTensor,
        dst_byte_offset: usize,
        gpu: &mut Gpu,
    ) -> HipResult<TransferHandle>;
```

In `impl Transport for PreadH2DTransport`, implement it by pread-ing into the existing host staging buffer and uploading to `dst` at the byte offset (mirror the body of `fetch`, but replace the `gpu.upload_raw(...)` allocation with a copy into `dst` at `dst_byte_offset`).

- [ ] **Step 4: Run the whole weight_pager suite**

Run: `cargo test -p hipfire-runtime --lib weight_pager`
Expected: all existing tests plus the new one pass (7 total).

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-runtime/src/weight_pager.rs
git commit -m "feat(pager): add non-allocating Transport::fetch_into"
```

---

### Task 3: `Ds4ExpertPager` — slot pool, residency, LRU, budget sizing

**Files:**
- Create: `crates/hipfire-arch-deepseek4/src/expert_pager.rs`
- Modify: `crates/hipfire-arch-deepseek4/src/lib.rs` (add `pub mod expert_pager;`)

**Interfaces:**
- Consumes: `hipfire_runtime::weight_pager::{Transport, PreadH2DTransport}`, `rdna_compute::{Gpu, GpuTensor}`.
- Produces:
  - `struct ExpertKey { pub layer: u16, pub expert: u16, pub role: ExpertBlobRole }`
  - `enum ExpertBlobRole { GateUp, Down }`
  - `struct SlotPlan { pub slots_per_blob: usize, pub bytes: u64 }`
  - `fn plan_slots(budget_bytes: u64, n_layers: usize, gate_up_stride: usize, w2_stride: usize, n_experts_per_tok: usize) -> Result<SlotPlan, PagerSizingError>`
  - `struct Ds4ExpertPager` with `fn ensure_resident(&mut self, key: ExpertKey, gpu: &mut Gpu) -> Result<u64, PagerError>` returning the device pointer of the slot now holding that expert.

- [ ] **Step 1: Write the failing tests**

```rust
// crates/hipfire-arch-deepseek4/src/expert_pager.rs  (test module at the bottom)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_slots_floors_to_budget() {
        // 43 layers, gate_up 2x a 2304 KiB matrix, w2 1x.
        let gu = 2 * 2_359_296usize;
        let w2 = 2_359_296usize;
        // Budget for exactly 8 slots per blob.
        let budget = (43 * (gu + w2) * 8) as u64;
        let p = plan_slots(budget, 43, gu, w2, 6).expect("fits");
        assert_eq!(p.slots_per_blob, 8);
    }

    #[test]
    fn plan_slots_rejects_budget_below_one_token_working_set() {
        let gu = 2 * 2_359_296usize;
        let w2 = 2_359_296usize;
        // Only room for 5 slots, but a token routes to 6 experts.
        let budget = (43 * (gu + w2) * 5) as u64;
        let err = plan_slots(budget, 43, gu, w2, 6).unwrap_err();
        assert!(
            matches!(err, PagerSizingError::BelowMinimum { needed_slots: 6, .. }),
            "expected BelowMinimum, got {err:?}"
        );
    }

    #[test]
    fn plan_slots_caps_at_full_residency() {
        let gu = 2 * 2_359_296usize;
        let w2 = 2_359_296usize;
        // Absurdly large budget must not plan more slots than there are experts.
        let p = plan_slots(u64::MAX / 4, 43, gu, w2, 6).expect("fits");
        assert!(p.slots_per_blob <= 256, "got {}", p.slots_per_blob);
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p hipfire-arch-deepseek4 --lib expert_pager`
Expected: FAIL — `plan_slots` / `PagerSizingError` not found.

- [ ] **Step 3: Implement the sizing and pager types**

```rust
// crates/hipfire-arch-deepseek4/src/expert_pager.rs
//! Bounded, never-growing routed-expert cache for DeepSeek V4.
//!
//! ds4 uploads routed experts as per-layer contiguous blobs plus a device-side
//! pointer table the indexed MoE GEMV dereferences. Only the routed
//! `num_experts_per_tok` entries are dereferenced per token, so we can allocate
//! blobs with K cache slots instead of `n_routed_experts` and repoint table
//! entries as experts become resident.
//!
//! MEMORY SAFETY: the slot pool is allocated ONCE by the caller at load and
//! never grows. `ensure_resident` on a miss evicts an LRU slot and reads into
//! it — there is no path from a miss to an allocator.

use std::collections::{HashMap, VecDeque};

/// Which of the two per-layer expert blobs a key refers to. ds4 fuses w1+w3
/// into `gate_up` and keeps `w2` (down) separate — two blobs per layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertBlobRole {
    GateUp,
    Down,
}

/// Identity of one cacheable routed-expert weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub layer: u16,
    pub expert: u16,
    pub role: ExpertBlobRole,
}

/// Outcome of budget sizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotPlan {
    /// Cache slots per (layer, blob). `256` means fully resident.
    pub slots_per_blob: usize,
    /// Total bytes the slot pool will occupy.
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagerSizingError {
    /// Budget cannot hold even one token's working set.
    BelowMinimum { needed_slots: usize, got_slots: usize },
}

impl std::fmt::Display for PagerSizingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PagerSizingError::BelowMinimum { needed_slots, got_slots } => write!(
                f,
                "expert cache budget too small: fits {got_slots} slots/blob, \
                 need at least {needed_slots} (num_experts_per_tok) to make progress"
            ),
        }
    }
}

/// Hard cap: never plan more slots than there are routed experts.
const MAX_EXPERTS: usize = 256;

/// Decide how many cache slots per blob fit in `budget_bytes`.
///
/// Fails closed when the budget cannot hold one token's working set, so an
/// undersized configuration errors at LOAD rather than stalling mid-forward.
pub fn plan_slots(
    budget_bytes: u64,
    n_layers: usize,
    gate_up_stride: usize,
    w2_stride: usize,
    n_experts_per_tok: usize,
) -> Result<SlotPlan, PagerSizingError> {
    let per_slot = (n_layers as u64) * (gate_up_stride as u64 + w2_stride as u64);
    let slots = if per_slot == 0 {
        0
    } else {
        (budget_bytes / per_slot) as usize
    };
    let slots = slots.min(MAX_EXPERTS);
    if slots < n_experts_per_tok {
        return Err(PagerSizingError::BelowMinimum {
            needed_slots: n_experts_per_tok,
            got_slots: slots,
        });
    }
    Ok(SlotPlan {
        slots_per_blob: slots,
        bytes: slots as u64 * per_slot,
    })
}

/// Residency + LRU bookkeeping over a fixed slot pool.
///
/// The pool itself (device blobs) is owned by the caller; this tracks which
/// expert occupies which slot index and which slot to reuse next.
pub struct Ds4ExpertPager {
    slots_per_blob: usize,
    /// (layer, expert, role) -> slot index within that (layer, role) blob.
    resident: HashMap<ExpertKey, usize>,
    /// Per (layer, role) LRU of slot indices, least-recently-used at the front.
    lru: HashMap<(u16, ExpertBlobRole), VecDeque<usize>>,
    /// Per (layer, role) reverse map slot index -> currently-held expert.
    occupant: HashMap<(u16, ExpertBlobRole, usize), u16>,
    hits: u64,
    misses: u64,
}

impl Ds4ExpertPager {
    pub fn new(slots_per_blob: usize) -> Self {
        Self {
            slots_per_blob,
            resident: HashMap::new(),
            lru: HashMap::new(),
            occupant: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    pub fn slots_per_blob(&self) -> usize {
        self.slots_per_blob
    }

    pub fn hit_rate(&self) -> f64 {
        let t = self.hits + self.misses;
        if t == 0 {
            0.0
        } else {
            self.hits as f64 / t as f64
        }
    }

    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Resolve `key` to a slot index, evicting LRU if needed.
    ///
    /// Returns `(slot_index, was_miss)`. On a miss the CALLER must read the
    /// expert's bytes into that slot before use — this function only does
    /// bookkeeping and never allocates or performs I/O.
    pub fn resolve_slot(&mut self, key: ExpertKey) -> (usize, bool) {
        let bucket = (key.layer, key.role);
        if let Some(&slot) = self.resident.get(&key) {
            self.hits += 1;
            let q = self.lru.entry(bucket).or_default();
            if let Some(p) = q.iter().position(|&s| s == slot) {
                q.remove(p);
            }
            q.push_back(slot);
            return (slot, false);
        }
        self.misses += 1;
        let q = self.lru.entry(bucket).or_default();
        let slot = if q.len() < self.slots_per_blob {
            q.len()
        } else {
            let victim = q.pop_front().expect("non-empty when full");
            if let Some(old) = self.occupant.remove(&(key.layer, key.role, victim)) {
                self.resident.remove(&ExpertKey {
                    layer: key.layer,
                    expert: old,
                    role: key.role,
                });
            }
            victim
        };
        let q = self.lru.entry(bucket).or_default();
        q.push_back(slot);
        self.resident.insert(key, slot);
        self.occupant.insert((key.layer, key.role, slot), key.expert);
        (slot, true)
    }
}
```

- [ ] **Step 4: Register the module**

In `crates/hipfire-arch-deepseek4/src/lib.rs`, add alongside the other `pub mod` lines:

```rust
pub mod expert_pager;
```

- [ ] **Step 5: Run the tests**

Run: `cargo test -p hipfire-arch-deepseek4 --lib expert_pager`
Expected: 3 passed.

- [ ] **Step 6: Add eviction-correctness tests and make them pass**

```rust
    #[test]
    fn evicts_lru_and_forgets_the_victim() {
        let mut p = Ds4ExpertPager::new(2);
        let k = |e: u16| ExpertKey { layer: 0, expert: e, role: ExpertBlobRole::GateUp };
        assert_eq!(p.resolve_slot(k(1)), (0, true));
        assert_eq!(p.resolve_slot(k(2)), (1, true));
        // Hit on 1 makes 2 the LRU.
        assert_eq!(p.resolve_slot(k(1)), (0, false));
        // 3 evicts 2, taking its slot.
        assert_eq!(p.resolve_slot(k(3)), (1, true));
        // 2 is gone: re-requesting it is a miss.
        assert_eq!(p.resolve_slot(k(2)).1, true);
    }

    #[test]
    fn buckets_are_independent_per_layer_and_role() {
        let mut p = Ds4ExpertPager::new(1);
        let a = ExpertKey { layer: 0, expert: 1, role: ExpertBlobRole::GateUp };
        let b = ExpertKey { layer: 1, expert: 1, role: ExpertBlobRole::GateUp };
        let c = ExpertKey { layer: 0, expert: 1, role: ExpertBlobRole::Down };
        assert_eq!(p.resolve_slot(a), (0, true));
        assert_eq!(p.resolve_slot(b), (0, true));
        assert_eq!(p.resolve_slot(c), (0, true));
        // None evicted each other.
        assert_eq!(p.resolve_slot(a).1, false);
        assert_eq!(p.resolve_slot(b).1, false);
        assert_eq!(p.resolve_slot(c).1, false);
    }

    #[test]
    fn hit_rate_tracks_reuse() {
        let mut p = Ds4ExpertPager::new(4);
        let k = ExpertKey { layer: 0, expert: 7, role: ExpertBlobRole::Down };
        p.resolve_slot(k);
        p.resolve_slot(k);
        p.resolve_slot(k);
        let (hits, misses) = p.stats();
        assert_eq!((hits, misses), (2, 1));
        assert!((p.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }
```

Run: `cargo test -p hipfire-arch-deepseek4 --lib expert_pager`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add crates/hipfire-arch-deepseek4/src/expert_pager.rs crates/hipfire-arch-deepseek4/src/lib.rs
git commit -m "feat(ds4): bounded routed-expert cache with LRU slot resolution"
```

---

### Task 4: Catalog — map every routed expert to its byte range in the HFQ

**Files:**
- Modify: `crates/hipfire-arch-deepseek4/src/expert_pager.rs`

**Interfaces:**
- Consumes: `hipfire_runtime::hfq::HfqFile` tensor index.
- Produces: `struct ExpertCatalog` with `fn byte_range(&self, key: ExpertKey) -> Option<(usize, usize)>` and `fn build(hfq: &HfqFile, n_layers: usize, n_experts: usize) -> Result<ExpertCatalog, PagerError>`.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn catalog_reports_missing_expert_rather_than_guessing() {
        let mut c = ExpertCatalog::empty();
        let k = ExpertKey { layer: 3, expert: 9, role: ExpertBlobRole::GateUp };
        assert!(c.byte_range(k).is_none());
        c.insert(k, 1024, 2_359_296);
        assert_eq!(c.byte_range(k), Some((1024, 2_359_296)));
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p hipfire-arch-deepseek4 --lib expert_pager::tests::catalog_reports_missing_expert_rather_than_guessing`
Expected: FAIL — `ExpertCatalog` not found.

- [ ] **Step 3: Implement `ExpertCatalog`**

```rust
/// (file_offset, byte_len) for every pageable routed expert.
///
/// Built once at load from the HFQ tensor index. A missing entry is an ERROR at
/// build time, never a silent zero at first use.
#[derive(Debug, Default)]
pub struct ExpertCatalog {
    ranges: HashMap<ExpertKey, (usize, usize)>,
}

impl ExpertCatalog {
    pub fn empty() -> Self {
        Self { ranges: HashMap::new() }
    }

    pub fn insert(&mut self, key: ExpertKey, offset: usize, len: usize) {
        self.ranges.insert(key, (offset, len));
    }

    pub fn byte_range(&self, key: ExpertKey) -> Option<(usize, usize)> {
        self.ranges.get(&key).copied()
    }

    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}
```

- [ ] **Step 4: Run the test**

Run: `cargo test -p hipfire-arch-deepseek4 --lib expert_pager`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-arch-deepseek4/src/expert_pager.rs
git commit -m "feat(ds4): expert byte-range catalog with fail-closed lookup"
```

---

### Task 5: Wire paging into the ds4 MoE forward behind an env knob

**Files:**
- Modify: `crates/hipfire-arch-deepseek4/src/arch.rs` (blob allocation at the routed-expert upload, doc comment at line 151, writes at 269 and 361)
- Modify: `crates/hipfire-arch-deepseek4/src/forward.rs` (MoE dispatch; top-k lives in `state.moe_topk_indices`, see lines 3627 and 3788)

**Interfaces:**
- Consumes: `plan_slots`, `Ds4ExpertPager::resolve_slot`, `ExpertCatalog::byte_range`, `Transport::fetch_into`.
- Produces: env knob `HIPFIRE_DEEPSEEK4_EXPERT_CACHE_GB` (unset = fully resident, current behaviour).

- [ ] **Step 1: Add the sizing entry point and its test**

```rust
    #[test]
    fn cache_gb_env_absent_means_fully_resident() {
        assert_eq!(expert_cache_budget_bytes(None), None);
    }

    #[test]
    fn cache_gb_env_parses_to_bytes() {
        assert_eq!(expert_cache_budget_bytes(Some("40")), Some(40 * 1024 * 1024 * 1024));
    }

    #[test]
    fn cache_gb_env_rejects_garbage_rather_than_defaulting() {
        assert_eq!(expert_cache_budget_bytes(Some("banana")), None);
    }
```

Implementation in `expert_pager.rs`:

```rust
/// Parse `HIPFIRE_DEEPSEEK4_EXPERT_CACHE_GB`. `None` = page nothing (fully
/// resident, today's behaviour). Unparseable input yields `None` so a typo
/// degrades to the safe path rather than to an arbitrary budget.
pub fn expert_cache_budget_bytes(raw: Option<&str>) -> Option<u64> {
    let v = raw?.trim().parse::<u64>().ok()?;
    if v == 0 {
        return None;
    }
    Some(v * 1024 * 1024 * 1024)
}
```

Run: `cargo test -p hipfire-arch-deepseek4 --lib expert_pager`
Expected: 10 passed.

- [ ] **Step 1b: Auto-size the budget from available memory, then enforce it hard**

The spec requires `min(configured, MemAvailable − non_routed − kv_and_scratch − headroom)`: auto-size for convenience, hard-enforce for determinism. Add the tests:

```rust
    #[test]
    fn auto_budget_subtracts_reservations_from_available() {
        // 100 GiB available, 10 GiB reserved => 90 GiB usable.
        let got = auto_budget_bytes(100 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024);
        assert_eq!(got, 90 * 1024 * 1024 * 1024);
    }

    #[test]
    fn auto_budget_saturates_at_zero_rather_than_underflowing() {
        // Reservations exceed what's available: must not wrap around.
        assert_eq!(auto_budget_bytes(4 * 1024 * 1024 * 1024, 9 * 1024 * 1024 * 1024), 0);
    }

    #[test]
    fn effective_budget_takes_the_smaller_of_configured_and_available() {
        let avail = 90u64 * 1024 * 1024 * 1024;
        // Configured smaller than available => configured wins.
        assert_eq!(effective_budget_bytes(Some(40 * 1024 * 1024 * 1024), avail), 40 * 1024 * 1024 * 1024);
        // Configured larger than available => clamped to available.
        assert_eq!(effective_budget_bytes(Some(200 * 1024 * 1024 * 1024), avail), avail);
        // Nothing configured => use all available.
        assert_eq!(effective_budget_bytes(None, avail), avail);
    }
```

Implementation:

```rust
/// Bytes usable for the slot pool, given MemAvailable and everything the pager
/// does NOT own (non-routed weights, KV/SWA caches, per-step scratch, headroom).
/// Saturates at zero so an over-subscribed box yields a clean sizing error from
/// `plan_slots` rather than an underflowed, enormous budget.
pub fn auto_budget_bytes(mem_available_bytes: u64, reserved_bytes: u64) -> u64 {
    mem_available_bytes.saturating_sub(reserved_bytes)
}

/// Final budget: the smaller of what the user asked for and what actually fits.
/// Auto-size for convenience; the result is then fixed for the process lifetime.
pub fn effective_budget_bytes(configured: Option<u64>, auto: u64) -> u64 {
    match configured {
        Some(c) => c.min(auto),
        None => auto,
    }
}

/// Read MemAvailable (kB) from /proc/meminfo, in bytes. `None` if unreadable,
/// in which case the caller must fall back to the configured budget alone.
pub fn mem_available_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}
```

Run: `cargo test -p hipfire-arch-deepseek4 --lib expert_pager`
Expected: 13 passed.

- [ ] **Step 2: Allocate the slot pool instead of the full blob when paging is on**

At the routed-expert upload in `arch.rs` (see the doc comment at line 151 and the blob writes at 269/361): when `expert_cache_budget_bytes(...)` is `Some(budget)`, call `plan_slots(effective_budget_bytes(configured, auto_budget_bytes(mem_available_bytes().unwrap_or(0), reserved)), n_layers, gate_up_stride, w2_stride, cfg.num_experts_per_tok)`. On `Ok(plan)`, allocate each blob as `plan.slots_per_blob * stride` bytes and upload only the first `plan.slots_per_blob` experts (the rest arrive by paging). On `Err(e)`, return the error so the load fails cleanly with the sizing message.

When the env is unset, take the existing path unchanged — allocate `n_routed_experts * stride` and upload all experts.

- [ ] **Step 3: Page on demand in the MoE dispatch**

In the MoE dispatch in `forward.rs`, before the indexed GEMV, when the pager is active:
1. D2H `state.moe_topk_indices` for this token (`num_experts_per_tok` u32s — see the existing device handle at `forward.rs:3627` for the routed path and `:3788` for the hash-routed path),
2. for each routed expert and each of `ExpertBlobRole::{GateUp, Down}`, call `resolve_slot`; on a miss look up `catalog.byte_range(key)` and `fetch_into` the slot at `slot_index * stride`, propagating any error with layer/expert/offset context,
3. write the slot's device pointer into the pointer table entry for that expert using the two-F32-slot encoding pinned in Task 1, and upload the patched table.

- [ ] **Step 4: Verify the default path is untouched**

Run: `cargo test -p hipfire-arch-deepseek4 --lib`
Run: `cargo build --release --example daemon`
Expected: both succeed; no behaviour change with the env unset.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-arch-deepseek4/src/arch.rs crates/hipfire-arch-deepseek4/src/forward.rs crates/hipfire-arch-deepseek4/src/expert_pager.rs
git commit -m "feat(ds4): page routed experts on demand behind HIPFIRE_DEEPSEEK4_EXPERT_CACHE_GB"
```

---

### Task 6: Output-neutrality gate

Paging is pure memory management over read-only weights, so it must not change output. Greedy ds4 decode is deterministic, which makes that directly assertable.

**Files:**
- Create: `scripts/paging-neutrality-gate.sh`

**Interfaces:**
- Consumes: `target/release/examples/daemon`, `HIPFIRE_DEEPSEEK4_EXPERT_CACHE_GB`, `HIPFIRE_EMIT_TOKEN_IDS=1`.
- Produces: exit 0 when paged output is bit-identical to resident output; exit 1 otherwise.

- [ ] **Step 1: Write the gate**

```bash
#!/usr/bin/env bash
# Paging must be output-neutral: identical committed token IDs with the expert
# cache large enough to hold everything AND with it small enough to thrash.
# Compares committed token IDs (not text) because BPE can mask a divergence.
set -u
cd "$(dirname "$0")/.."
MODEL="${HIPFIRE_DS4_MODEL:-$HOME/.hipfire/models/deepseek-v4-flash-0731.mq2lloyd}"
EXE=./target/release/examples/daemon
PROMPT="List three primary colours, comma separated."

run() { # $1 = cache GB ("" = fully resident)
  local env_prefix=""
  [ -n "$1" ] && env_prefix="HIPFIRE_DEEPSEEK4_EXPERT_CACHE_GB=$1"
  printf '%s\n%s\n%s\n' \
    "{\"type\":\"load\",\"model\":\"$MODEL\",\"params\":{\"max_seq\":4096,\"dspark_mode\":\"off\",\"mtp_mode\":\"off\"}}" \
    "{\"type\":\"generate\",\"id\":\"g\",\"prompt\":\"$PROMPT\",\"temperature\":0.0,\"max_tokens\":64,\"repeat_penalty\":1.0,\"reasoning_effort\":\"high\"}" \
    '{"type":"unload"}' \
  | env $env_prefix HIPFIRE_EMIT_TOKEN_IDS=1 "$EXE" 2>/dev/null \
  | grep -a '"type": *"committed"' | sed 's/.*"tok_id": *\([0-9]*\).*/\1/' | tr '\n' ' '
}

echo "== resident (no paging) =="; A=$(run ""); echo "$A" | head -c 200; echo
echo "== paged, large cache ==";   B=$(run "70"); echo "$B" | head -c 200; echo
echo "== paged, thrashing cache =="; C=$(run "8"); echo "$C" | head -c 200; echo

rc=0
[ -n "$A" ] || { echo "FAIL: resident run produced no tokens"; rc=1; }
[ "$A" = "$B" ] || { echo "FAIL: large-cache paged output differs from resident"; rc=1; }
[ "$A" = "$C" ] || { echo "FAIL: thrashing-cache paged output differs from resident"; rc=1; }
[ $rc -eq 0 ] && echo "PASS: paging is output-neutral"
exit $rc
```

- [ ] **Step 2: Make it executable and syntax-check**

```bash
chmod +x scripts/paging-neutrality-gate.sh
bash -n scripts/paging-neutrality-gate.sh
```
Expected: no output (syntax OK).

- [ ] **Step 3: Run the gate**

Run: `./scripts/paging-neutrality-gate.sh`
Expected: `PASS: paging is output-neutral`.

If the thrashing case differs but the large-cache case matches, the bug is in eviction or pointer-table patching (Task 3 / Task 5 step 3), not in the read path.

- [ ] **Step 4: Verify the sizing floor fails closed**

Run: `HIPFIRE_DEEPSEEK4_EXPERT_CACHE_GB=1 ./scripts/paging-neutrality-gate.sh`
Expected: the load errors with the "expert cache budget too small" message and the gate reports FAIL rather than producing degraded output.

- [ ] **Step 5: Commit**

```bash
git add scripts/paging-neutrality-gate.sh
git commit -m "test(ds4): output-neutrality gate for routed-expert paging"
```

---

## Definition of done

- `./scripts/paging-neutrality-gate.sh` passes.
- `cargo test -p hipfire-arch-deepseek4 --lib` and `cargo test -p hipfire-runtime --lib` green.
- With the env unset, behaviour and throughput are unchanged from today.
- Hit rate is logged so the budget/throughput curve is empirical.
- Follow-up (NOT this plan): load an MQ3 trunk under paging, measure PPL, then `KLD(MQ3 ‖ MQ2)` — the measurement that motivated the work.
