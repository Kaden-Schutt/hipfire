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

/// One contiguous run of HFQ bytes to copy into a cache slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSegment {
    /// Byte offset in the HFQ file.
    pub offset: usize,
    /// Byte length.
    pub len: usize,
}

/// Anything that can go wrong building or using the catalog.
///
/// Every variant carries enough context to name the offending tensor. A read
/// error must NEVER degrade to a zero, stale, or wrong expert: the shard path's
/// zeroed dummy makes a bad pointer produce silence rather than a fault (see
/// `tests/expert_blob_contract.rs`), so paging has to fail loudly instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PagerError {
    /// A routed-expert tensor named by the catalog is absent from the HFQ.
    MissingTensor { name: String },
    /// Experts are not a uniform size, so `slot_index * stride` addressing —
    /// which both the blob layout and the pointer table assume — is invalid.
    StrideMismatch {
        name: String,
        got: usize,
        want: usize,
    },
    /// An expert was requested that the catalog has no byte range for.
    NotCatalogued { key: ExpertKey },
    /// The slot pool could not be sized.
    Sizing(PagerSizingError),
    /// The on-demand read failed. Carries layer/expert/offset context.
    Read {
        key: ExpertKey,
        offset: usize,
        len: usize,
        detail: String,
    },
}

impl std::fmt::Display for PagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PagerError::MissingTensor { name } => {
                write!(f, "expert pager: missing routed-expert tensor '{name}'")
            }
            PagerError::StrideMismatch { name, got, want } => write!(
                f,
                "expert pager: '{name}' size {got} != expert stride {want}; \
                 routed experts must be a uniform stride for slot addressing"
            ),
            PagerError::NotCatalogued { key } => write!(
                f,
                "expert pager: no byte range catalogued for layer {} expert {} {:?}",
                key.layer, key.expert, key.role
            ),
            PagerError::Sizing(e) => write!(f, "{e}"),
            PagerError::Read {
                key,
                offset,
                len,
                detail,
            } => write!(
                f,
                "expert pager: read of layer {} expert {} {:?} \
                 ({len} B at offset {offset}) failed: {detail}",
                key.layer, key.expert, key.role
            ),
        }
    }
}

impl std::error::Error for PagerError {}

impl From<PagerSizingError> for PagerError {
    fn from(e: PagerSizingError) -> Self {
        PagerError::Sizing(e)
    }
}

/// The HFQ byte ranges backing every pageable routed expert.
///
/// Built once at load from the HFQ tensor index. A missing entry is an ERROR at
/// build time, never a silent zero at first use.
///
/// A key maps to a *list* of segments, not a single range: ds4 fuses w1 and w3
/// into one `gate_up` slot, but they are two separate tensors at unrelated file
/// offsets, so filling a GateUp slot means two reads written back-to-back.
/// `Down` (w2) is a single segment.
#[derive(Debug, Default)]
pub struct ExpertCatalog {
    ranges: HashMap<ExpertKey, Vec<ExpertSegment>>,
    /// Bytes one slot occupies, per role. Uniform across layers and experts —
    /// enforced at build time because slot addressing depends on it.
    gate_up_slot_len: Option<usize>,
    down_slot_len: Option<usize>,
}

impl ExpertCatalog {
    pub fn empty() -> Self {
        Self {
            ranges: HashMap::new(),
            gate_up_slot_len: None,
            down_slot_len: None,
        }
    }

    /// Record a single-segment entry. Used for `Down`, and by tests.
    pub fn insert(&mut self, key: ExpertKey, offset: usize, len: usize) {
        self.ranges.insert(key, vec![ExpertSegment { offset, len }]);
    }

    /// The segments to read, in blob order, to fill this expert's slot.
    /// `None` means the expert was never catalogued — the caller must error,
    /// not substitute anything.
    pub fn segments(&self, key: ExpertKey) -> Option<&[ExpertSegment]> {
        self.ranges.get(&key).map(|v| v.as_slice())
    }

    /// Convenience for single-segment entries (`Down`). Returns `None` both for
    /// an unknown key and for a multi-segment entry such as `GateUp` — callers
    /// that must handle both roles use [`ExpertCatalog::segments`].
    pub fn byte_range(&self, key: ExpertKey) -> Option<(usize, usize)> {
        match self.ranges.get(&key)?.as_slice() {
            [seg] => Some((seg.offset, seg.len)),
            _ => None,
        }
    }

    /// Bytes one cache slot of this role occupies.
    pub fn slot_len(&self, role: ExpertBlobRole) -> Option<usize> {
        match role {
            ExpertBlobRole::GateUp => self.gate_up_slot_len,
            ExpertBlobRole::Down => self.down_slot_len,
        }
    }

    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Build from a name→(offset, len) resolver.
    ///
    /// `layers` pairs the layer id used in [`ExpertKey`] with its HFQ tensor
    /// prefix (`layers.{L}`, or `mtp.0` for the MTP block). `src` maps a
    /// compact expert slot to its ORIGINAL index, mirroring the REAP keep-map
    /// in `upload_layer_routed_experts`; pass the identity when no keep-map is
    /// active. Every expert of every layer must resolve at a uniform stride or
    /// the build fails — a hole here would be a wrong-weights read later.
    pub fn build_from<F>(
        layers: &[(u16, String)],
        n_experts: usize,
        src: impl Fn(usize) -> usize,
        lookup: F,
    ) -> Result<Self, PagerError>
    where
        F: Fn(&str) -> Option<(usize, usize)>,
    {
        let mut cat = ExpertCatalog::empty();
        let mut part_stride: Option<usize> = None;
        for (layer_id, prefix) in layers {
            for slot in 0..n_experts {
                let orig = src(slot);
                let mut fetch = |part: &str| -> Result<ExpertSegment, PagerError> {
                    let name = format!("{prefix}.ffn.experts.{orig}.{part}.weight");
                    let (offset, len) = lookup(&name)
                        .ok_or_else(|| PagerError::MissingTensor { name: name.clone() })?;
                    match part_stride {
                        None => part_stride = Some(len),
                        Some(want) if want != len => {
                            return Err(PagerError::StrideMismatch {
                                name,
                                got: len,
                                want,
                            })
                        }
                        Some(_) => {}
                    }
                    Ok(ExpertSegment { offset, len })
                };
                let w1 = fetch("w1")?;
                let w3 = fetch("w3")?;
                let w2 = fetch("w2")?;
                cat.ranges.insert(
                    ExpertKey {
                        layer: *layer_id,
                        expert: slot as u16,
                        role: ExpertBlobRole::GateUp,
                    },
                    vec![w1, w3],
                );
                cat.ranges.insert(
                    ExpertKey {
                        layer: *layer_id,
                        expert: slot as u16,
                        role: ExpertBlobRole::Down,
                    },
                    vec![w2],
                );
            }
        }
        if let Some(s) = part_stride {
            cat.gate_up_slot_len = Some(2 * s);
            cat.down_slot_len = Some(s);
        }
        Ok(cat)
    }

    /// Build from a real HFQ tensor index.
    pub fn build(
        hfq: &hipfire_runtime::hfq::HfqFile,
        layers: &[(u16, String)],
        n_experts: usize,
        keep: Option<&[u32]>,
    ) -> Result<Self, PagerError> {
        let src = |slot: usize| keep.map(|k| k[slot] as usize).unwrap_or(slot);
        Self::build_from(layers, n_experts, src, |name| {
            hfq.find_tensor_info(name)
                .map(|i| (i.data_offset, i.data_size))
        })
    }
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

    /// A fake HFQ index: `name -> (data_offset, data_size)`.
    fn fake_index(prefixes: &[&str], n_exp: usize, stride: usize) -> Vec<(String, (usize, usize))> {
        let mut v = Vec::new();
        let mut off = 4096usize;
        for p in prefixes {
            for e in 0..n_exp {
                for part in ["w1", "w3", "w2"] {
                    v.push((format!("{p}.ffn.experts.{e}.{part}.weight"), (off, stride)));
                    off += stride;
                }
            }
        }
        v
    }

    fn lookup_from(
        idx: &[(String, (usize, usize))],
    ) -> impl Fn(&str) -> Option<(usize, usize)> + '_ {
        move |name: &str| idx.iter().find(|(n, _)| n == name).map(|(_, r)| *r)
    }

    #[test]
    fn catalog_reports_missing_expert_rather_than_guessing() {
        let mut c = ExpertCatalog::empty();
        let k = ExpertKey {
            layer: 3,
            expert: 9,
            role: ExpertBlobRole::GateUp,
        };
        assert!(c.byte_range(k).is_none());
        c.insert(k, 1024, 2_359_296);
        assert_eq!(c.byte_range(k), Some((1024, 2_359_296)));
    }

    #[test]
    fn gate_up_holds_two_segments_because_w1_and_w3_are_separate_tensors() {
        // The fused gate_up slot is w1 ‖ w3, and the two live at unrelated
        // offsets in the HFQ. A single range cannot describe it, so a GateUp
        // entry carries both segments in blob order.
        let stride = 2_359_296usize;
        let idx = fake_index(&["layers.0"], 2, stride);
        let cat =
            ExpertCatalog::build_from(&[(0u16, "layers.0".into())], 2, |s| s, lookup_from(&idx))
                .expect("builds");
        let gu = cat
            .segments(ExpertKey {
                layer: 0,
                expert: 1,
                role: ExpertBlobRole::GateUp,
            })
            .expect("gate_up present");
        assert_eq!(gu.len(), 2, "gate_up must be w1 ‖ w3");
        let w1 = idx
            .iter()
            .find(|(n, _)| n == "layers.0.ffn.experts.1.w1.weight")
            .unwrap()
            .1;
        let w3 = idx
            .iter()
            .find(|(n, _)| n == "layers.0.ffn.experts.1.w3.weight")
            .unwrap()
            .1;
        assert_eq!((gu[0].offset, gu[0].len), w1);
        assert_eq!((gu[1].offset, gu[1].len), w3);
        assert_eq!(cat.slot_len(ExpertBlobRole::GateUp), Some(2 * stride));
        assert_eq!(cat.slot_len(ExpertBlobRole::Down), Some(stride));
    }

    #[test]
    fn build_errors_on_a_missing_tensor_rather_than_skipping_it() {
        // A hole in the catalog is a wrong-weights read at first use, so it
        // must be a load-time error with the tensor named.
        let stride = 2_359_296usize;
        let mut idx = fake_index(&["layers.0"], 3, stride);
        idx.retain(|(n, _)| n != "layers.0.ffn.experts.2.w3.weight");
        let err =
            ExpertCatalog::build_from(&[(0u16, "layers.0".into())], 3, |s| s, lookup_from(&idx))
                .expect_err("must fail closed");
        let msg = err.to_string();
        assert!(
            msg.contains("experts.2.w3.weight"),
            "error must name the missing tensor, got: {msg}"
        );
    }

    #[test]
    fn build_errors_when_experts_are_not_a_uniform_stride() {
        // Paging indexes a slot as `slot_index * stride`. A ragged expert size
        // makes that arithmetic wrong, so reject it at load.
        let stride = 2_359_296usize;
        let mut idx = fake_index(&["layers.0"], 3, stride);
        for (n, r) in idx.iter_mut() {
            if n == "layers.0.ffn.experts.2.w2.weight" {
                r.1 = stride - 128;
            }
        }
        let err =
            ExpertCatalog::build_from(&[(0u16, "layers.0".into())], 3, |s| s, lookup_from(&idx))
                .expect_err("must fail closed");
        let msg = err.to_string();
        assert!(
            msg.contains("stride"),
            "error must explain the stride mismatch, got: {msg}"
        );
    }

    #[test]
    fn build_follows_the_reap_keep_map() {
        // With a REAP keep-map, compact slot `s` must read ORIGINAL expert
        // `keep[s]` — reading slot `s` directly would load the wrong weights.
        let stride = 2_359_296usize;
        let idx = fake_index(&["layers.0"], 8, stride);
        let keep = [5usize, 2, 7];
        let cat = ExpertCatalog::build_from(
            &[(0u16, "layers.0".into())],
            3,
            |s| keep[s],
            lookup_from(&idx),
        )
        .expect("builds");
        let got = cat
            .byte_range(ExpertKey {
                layer: 0,
                expert: 1,
                role: ExpertBlobRole::Down,
            })
            .expect("slot 1 present");
        let want = idx
            .iter()
            .find(|(n, _)| n == "layers.0.ffn.experts.2.w2.weight")
            .unwrap()
            .1;
        assert_eq!(got, want, "compact slot 1 must map to original expert 2");
    }

    #[test]
    fn catalog_covers_every_expert_of_every_layer() {
        let stride = 2_359_296usize;
        let idx = fake_index(&["layers.0", "layers.1", "mtp.0"], 4, stride);
        let layers = [
            (0u16, "layers.0".to_string()),
            (1u16, "layers.1".to_string()),
            (2u16, "mtp.0".to_string()),
        ];
        let cat = ExpertCatalog::build_from(&layers, 4, |s| s, lookup_from(&idx)).expect("builds");
        assert_eq!(cat.len(), 3 * 4 * 2, "3 layers x 4 experts x 2 roles");
        assert!(!cat.is_empty());
    }

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
