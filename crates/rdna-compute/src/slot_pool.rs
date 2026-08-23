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
    pub fn new(n_slots: usize, cap_tokens: usize, per_pos_bytes: usize) -> Result<Self, String> {
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

    /// Build an ELASTIC pool: a total token budget shared across slots with a
    /// big primary slab.
    ///
    /// Layout (per K or V arena, `stride = per_pos_bytes`):
    ///   slot 0        — `pool_total - (n_slots-1)*reserve` tokens at offset 0
    ///   slot 1..n-1   — `reserve` tokens each, packed after the primary
    ///
    /// A solo request admitted to the primary can use nearly the whole pool
    /// budget (e.g. ~96k of 100k with 2 slots and an 8k reserve); concurrent
    /// requests fall back to the reserved slabs. Addressing is unchanged —
    /// every kernel resolves `k_base + pos*stride` from the descriptor, and
    /// attention masks on `desc.seq_len`, so variable caps need no kernel
    /// changes. Sum of caps equals `pool_total` (rounded up per slab).
    pub fn new_elastic(
        n_slots: usize,
        reserve_per_slot: usize,
        pool_total: usize,
        per_pos_bytes: usize,
    ) -> Result<Self, String> {
        assert!(n_slots > 0, "n_slots must be positive");
        assert!(per_pos_bytes > 0, "per_pos_bytes must be positive");
        assert!(reserve_per_slot > 0, "reserve_per_slot must be positive");
        let reserve = reserve_per_slot.div_ceil(PAGE_TOKENS) * PAGE_TOKENS;
        let primary_raw = pool_total
            .checked_sub(reserve * (n_slots - 1))
            .filter(|p| *p >= reserve)
            .ok_or_else(|| {
                format!(
                    "SlotPool: pool_total {pool_total} too small for {n_slots} slots \
                     with reserve {reserve}"
                )
            })?;
        let primary = primary_raw.div_ceil(PAGE_TOKENS) * PAGE_TOKENS;

        // One contiguous arena of pool_total (rounded) tokens per K or V.
        let arena_tokens = primary + reserve * (n_slots - 1);
        let arena_bytes = (arena_tokens * per_pos_bytes) as u64;
        let total = arena_bytes
            .checked_mul(2)
            .ok_or_else(|| "SlotPool: arena size overflows u64".to_string())?;
        preflight_alloc(total, R9700_VRAM_BYTES, "SlotPool arena")?;

        let stride = per_pos_bytes as u64;
        let mut descs = Vec::with_capacity(n_slots);
        let mut base: u64 = 0;
        for i in 0..n_slots {
            let cap = if i == 0 { primary } else { reserve };
            descs.push(KvSlotDesc {
                k_base: base,
                v_base: base,
                seq_len: 0,
                cap: cap as i32,
            });
            base += (cap as u64) * stride;
        }
        debug_assert_eq!(base as usize, arena_bytes as usize);

        Ok(Self {
            descs,
            in_use: vec![false; n_slots],
            cap_tokens: primary,
            per_pos_bytes,
            dirty: true,
        })
    }

    /// Capacity of a specific slot (caps vary under `new_elastic`).
    pub fn slot_cap(&self, id: SlotId) -> usize {
        self.descs[id.0].cap as usize
    }

    /// Build a DYNAMIC-RANGE pool: one arena of `total_tokens` with `n_slots`
    /// descriptors whose bases/caps are assigned at admission time via
    /// [`Self::set_desc_range`]. Arena bytes equal the whole budget; nothing
    /// is reserved per-slot.
    pub fn new_dynamic(n_slots: usize, total_tokens: usize, per_pos_bytes: usize)
        -> Result<Self, String>
    {
        assert!(n_slots > 0);
        let total_rounded = total_tokens.div_ceil(PAGE_TOKENS) * PAGE_TOKENS;
        let arena_bytes = (total_rounded * per_pos_bytes) as u64;
        preflight_alloc(arena_bytes.checked_mul(2).ok_or("overflow")?,
                        R9700_VRAM_BYTES, "SlotPool arena")?;
        // Descriptors start invalid (cap 0); admission assigns them.
        let descs = (0..n_slots).map(|i| KvSlotDesc {
            k_base: 0,
            v_base: 0,
            seq_len: 0,
            cap: 0,
        }).collect();
        Ok(Self { descs, in_use: vec![false; n_slots],
                  cap_tokens: total_rounded, per_pos_bytes, dirty: true })
    }

    /// Point a lane at an allocated KV range and mark it in-use.
    /// `range.byte_off` is a byte offset into each arena.
    pub fn bind_range(&mut self, id: SlotId, byte_off: u64, cap_tokens: usize) {
        self.descs[id.0] = KvSlotDesc {
            k_base: byte_off,
            v_base: byte_off,
            seq_len: 0,
            cap: cap_tokens as i32,
        };
        self.in_use[id.0] = true;
        self.dirty = true;
    }

    /// Release a lane and invalidate its descriptor.
    pub fn release_lane(&mut self, id: SlotId) {
        self.reset(id);
        self.in_use[id.0] = false;
    }

    /// Largest per-slot capacity in the pool.
    pub fn max_cap(&self) -> usize {
        self.cap_tokens
    }

    /// Take a free slot, or `None` when the pool is full. Admission control
    /// lives in SP4; this only reports capacity.
    pub fn acquire(&mut self) -> Option<SlotId> {
        let i = self.in_use.iter().position(|&u| !u)?;
        self.in_use[i] = true;
        self.reset(SlotId(i));
        Some(SlotId(i))
    }

    /// Take the SMALLEST free slot whose cap fits `needed_tokens`. Under an
    /// elastic pool this preserves the big primary slab for requests that
    /// actually need it; equal-slab pools behave identically to `acquire`.
    pub fn acquire_fitting(&mut self, needed_tokens: usize) -> Option<SlotId> {
        let mut best: Option<usize> = None;
        let mut best_cap = usize::MAX;
        for (i, (&u, d)) in self.in_use.iter().zip(self.descs.iter()).enumerate() {
            if u {
                continue;
            }
            let cap = d.cap as usize;
            if cap >= needed_tokens && cap < best_cap {
                best = Some(i);
                best_cap = cap;
            }
        }
        let i = best?;
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
        // Per-slot cap, not pool-wide max: under `new_elastic` reserved slots
        // are much smaller than the primary and must be bounded by their own
        // slab capacity.
        let cap = self.descs[id.0].cap as usize;
        if seq_len > cap {
            return Err(format!(
                "SlotPool: slot {} seq_len {} exceeds cap {}",
                id.0, seq_len, cap
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
            assert_eq!(
                d[i].k_base,
                prev_end,
                "slab {i} must start where {} ended",
                i - 1
            );
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
        assert_eq!(
            p.descriptors()[id2.0].seq_len,
            0,
            "reused slot must start empty"
        );
    }

    #[test]
    fn dirty_flag_tracks_descriptor_changes() {
        let mut p = SlotPool::new(1, 128, PPB).unwrap();
        p.mark_uploaded();
        assert!(!p.descriptors_dirty());
        let id = p.acquire().unwrap();
        p.set_seq_len(id, 10).unwrap();
        assert!(
            p.descriptors_dirty(),
            "a seq_len change must dirty the table"
        );
        p.mark_uploaded();
        assert!(!p.descriptors_dirty());
    }

    #[test]
    fn elastic_pool_gives_primary_the_budget_minus_reserves() {
        let p = SlotPool::new_elastic(2, 8192, 100_000, PPB).unwrap();
        let d = p.descriptors();
        assert_eq!(d.len(), 2);
        // primary = 100k - 8192 (rounded), reserve = 8192
        let expect_primary: usize = ((100_000usize - 8_192).div_ceil(128)) * 128; // 91_904
        assert_eq!(d[0].cap as usize, expect_primary);
        assert_eq!(d[1].cap as usize, 8192);
        // Non-overlapping: primary slab ends where reserve starts.
        assert_eq!(
            d[1].k_base as usize,
            d[0].cap as usize * PPB
        );
        // Solo request can use the primary fully.
        assert!(p.max_cap() >= 90_000);
    }

    #[test]
    fn elastic_pool_refuses_budget_smaller_than_reserves() {
        let e = SlotPool::new_elastic(4, 8192, 10_000, PPB).unwrap_err();
        assert!(e.contains("too small"), "unexpected: {e}");
    }

    #[test]
    fn elastic_pool_seq_len_respects_per_slot_cap() {
        let mut p = SlotPool::new_elastic(2, 8192, 100_000, PPB).unwrap();
        let primary = p.acquire().unwrap();
        // Primary can hold ~92k.
        let big = p.slot_cap(primary) - 1;
        assert!(p.set_seq_len(primary, big).is_ok());
        // Reserve slot cannot exceed 8192.
        let res = p.acquire().unwrap();
        assert!(p.set_seq_len(res, 8192).is_ok());
        assert!(p.set_seq_len(res, 8193).is_err());
    }

    #[test]
    fn oversized_pool_is_refused_not_allocated() {
        // 8 slots x 4M tokens x 1088 B/pos x 2 (K and V) = ~69.6 GB, over the
        // 32 GiB target budget.
        //
        // Was 1M tokens, which is ~17.4 GB once K and V are counted -- under
        // the budget, so `new` correctly returned Ok and the `unwrap_err` here
        // panicked. The test's comment said "8.7 TB", off by 1000x; the
        // refusal it is checking was never actually being exercised.
        let e = SlotPool::new(8, 4_000_000, PPB).unwrap_err();
        assert!(e.contains("budget") || e.contains("GiB"), "unexpected: {e}");
    }
}
