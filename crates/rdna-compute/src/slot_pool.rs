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

// `Debug` is required so tests can call `.unwrap_err()` on
// `Result<SlotPool, String>` (`unwrap_err` requires `T: Debug` because it
// formats the Ok value into the panic message on the failure path).
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
