// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Bounded, never-growing routed-expert cache for DeepSeek V4.
//!
//! ds4 uploads routed experts as per-layer contiguous blobs plus a device-side
//! pointer table the indexed MoE GEMV dereferences. Only the routed
//! `num_experts_per_tok` entries are dereferenced per token, so we can allocate
//! blobs with K cache slots instead of `n_routed_experts` and repoint table
//! entries as experts become resident. The expert→slot indirection already
//! exists for expert-parallel sharding (see `upload_layer_routed_experts` in
//! `arch.rs`); paging only changes the resident subset from static to dynamic.
//! The layout this depends on is pinned by `tests/expert_blob_contract.rs`.
//!
//! MEMORY SAFETY: the slot pool is allocated ONCE by the caller at load and
//! never grows. [`Ds4ExpertPager::resolve_slot`] on a miss evicts an LRU slot
//! and hands its index back to be overwritten in place — there is no path from
//! a miss to an allocator.

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
    /// Cache slots per (layer, blob). Equal to `n_routed_experts` means the
    /// cache can never miss, i.e. fully resident.
    pub slots_per_blob: usize,
    /// Total bytes the slot pool will occupy.
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagerSizingError {
    /// Budget cannot hold even one token's working set.
    BelowMinimum {
        needed_slots: usize,
        got_slots: usize,
    },
}

impl std::fmt::Display for PagerSizingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PagerSizingError::BelowMinimum {
                needed_slots,
                got_slots,
            } => write!(
                f,
                "expert cache budget too small: fits {got_slots} slots/blob, \
                 need at least {needed_slots} (num_experts_per_tok) to make progress"
            ),
        }
    }
}

impl std::error::Error for PagerSizingError {}

/// Hard cap: never plan more slots than DeepSeek V4 Flash has routed experts.
///
/// This is an upper bound, not the live count — a REAP-pruned checkpoint
/// lowers `cfg.n_routed_experts` below this. The caller clamps the plan to the
/// actual count (and treats "slots >= experts" as fully resident, skipping
/// paging entirely), so this const only stops an absurd budget from planning a
/// pool larger than any ds4 model could ever need.
const MAX_EXPERTS: usize = 256;

/// Decide how many cache slots per blob fit in `budget_bytes`.
///
/// Fails closed when the budget cannot hold one token's working set, so an
/// undersized configuration errors at LOAD rather than stalling mid-forward:
/// with fewer slots than a token routes to, the experts for a single token
/// would evict each other before that token finished.
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
        self.occupant
            .insert((key.layer, key.role, slot), key.expert);
        (slot, true)
    }
}

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
            matches!(
                err,
                PagerSizingError::BelowMinimum {
                    needed_slots: 6,
                    ..
                }
            ),
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

    #[test]
    fn evicts_lru_and_forgets_the_victim() {
        let mut p = Ds4ExpertPager::new(2);
        let k = |e: u16| ExpertKey {
            layer: 0,
            expert: e,
            role: ExpertBlobRole::GateUp,
        };
        assert_eq!(p.resolve_slot(k(1)), (0, true));
        assert_eq!(p.resolve_slot(k(2)), (1, true));
        // Hit on 1 makes 2 the LRU.
        assert_eq!(p.resolve_slot(k(1)), (0, false));
        // 3 evicts 2, taking its slot.
        assert_eq!(p.resolve_slot(k(3)), (1, true));
        // 2 is gone: re-requesting it is a miss.
        assert!(p.resolve_slot(k(2)).1);
    }

    #[test]
    fn buckets_are_independent_per_layer_and_role() {
        let mut p = Ds4ExpertPager::new(1);
        let a = ExpertKey {
            layer: 0,
            expert: 1,
            role: ExpertBlobRole::GateUp,
        };
        let b = ExpertKey {
            layer: 1,
            expert: 1,
            role: ExpertBlobRole::GateUp,
        };
        let c = ExpertKey {
            layer: 0,
            expert: 1,
            role: ExpertBlobRole::Down,
        };
        assert_eq!(p.resolve_slot(a), (0, true));
        assert_eq!(p.resolve_slot(b), (0, true));
        assert_eq!(p.resolve_slot(c), (0, true));
        // None evicted each other.
        assert!(!p.resolve_slot(a).1);
        assert!(!p.resolve_slot(b).1);
        assert!(!p.resolve_slot(c).1);
    }

    #[test]
    fn hit_rate_tracks_reuse() {
        let mut p = Ds4ExpertPager::new(4);
        let k = ExpertKey {
            layer: 0,
            expert: 7,
            role: ExpertBlobRole::Down,
        };
        p.resolve_slot(k);
        p.resolve_slot(k);
        p.resolve_slot(k);
        let (hits, misses) = p.stats();
        assert_eq!((hits, misses), (2, 1));
        assert!((p.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn a_full_bucket_never_hands_out_a_slot_outside_the_pool() {
        // The pool is allocated once; a slot index >= slots_per_blob would
        // write past the end of the blob. Churn well past capacity and prove
        // every index stays in range and the map never outgrows the pool.
        let mut p = Ds4ExpertPager::new(3);
        for e in 0..64u16 {
            let (slot, _) = p.resolve_slot(ExpertKey {
                layer: 5,
                expert: e,
                role: ExpertBlobRole::GateUp,
            });
            assert!(slot < 3, "slot {slot} outside pool of 3 at expert {e}");
        }
        assert_eq!(p.resident.len(), 3, "residency map outgrew the slot pool");
        assert_eq!(p.occupant.len(), 3, "occupant map outgrew the slot pool");
    }

    #[test]
    fn one_tokens_working_set_stays_resident_at_the_sizing_floor() {
        // plan_slots' floor exists so a token's experts cannot evict each
        // other. At exactly num_experts_per_tok slots, re-touching the same 6
        // experts must be all hits.
        let mut p = Ds4ExpertPager::new(6);
        let routed = [11u16, 42, 7, 200, 3, 99];
        for &e in &routed {
            assert!(
                p.resolve_slot(ExpertKey {
                    layer: 2,
                    expert: e,
                    role: ExpertBlobRole::Down
                })
                .1,
                "first touch of {e} should miss"
            );
        }
        for &e in &routed {
            assert!(
                !p.resolve_slot(ExpertKey {
                    layer: 2,
                    expert: e,
                    role: ExpertBlobRole::Down
                })
                .1,
                "expert {e} was evicted within one token's working set"
            );
        }
    }
}
