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

/// Encode a device pointer into the two-F32-slot table representation used by
/// the indexed MoE GEMV. Pinned by `tests/expert_blob_contract.rs`.
#[inline]
pub fn encode_ptr_slots(out: &mut [f32], expert: usize, p: u64) {
    out[expert * 2] = f32::from_bits((p & 0xffff_ffff) as u32);
    out[expert * 2 + 1] = f32::from_bits((p >> 32) as u32);
}

/// What one dispatch must do for a single routed expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotFill {
    pub key: ExpertKey,
    /// Slot index within this (layer, role) blob.
    pub slot: usize,
    /// True when the slot does not already hold this expert and must be read.
    pub needs_read: bool,
}

/// Everything paging needs at runtime, allocated once at load.
///
/// Split from [`Ds4ExpertPager`] so the bookkeeping stays unit-testable
/// without a GPU: [`Ds4PagingRuntime::plan_dispatch`] is pure and does the
/// fail-closed catalog check BEFORE any I/O happens.
pub struct Ds4PagingRuntime {
    pager: Ds4ExpertPager,
    catalog: ExpertCatalog,
    /// Host shadow of every (layer, role) device pointer table, `n_exp` u64s.
    /// Patched in place, then re-uploaded when a dispatch changes it.
    shadow: HashMap<(u16, ExpertBlobRole), Vec<u64>>,
    /// Reusable encode buffer for one pointer table (`2 * n_exp` f32 slots).
    encode_scratch: Vec<f32>,
    /// Device base pointer of each (layer, role) slot pool, snapshotted at
    /// construction. A dispatch that hands over a different blob is aiming at
    /// a pool this pager does not own — see [`Ds4PagingRuntime::plan_dispatch`].
    expected_base: HashMap<(u16, ExpertBlobRole), u64>,
    n_exp: usize,
}

impl Ds4PagingRuntime {
    /// Build the runtime. `layers` are the (layer id, HFQ prefix) pairs whose
    /// experts are pageable; `initial_ptrs` is what the loader wrote into each
    /// table, so a table that is never touched keeps its load-time contents.
    pub fn new(
        pager: Ds4ExpertPager,
        catalog: ExpertCatalog,
        n_exp: usize,
        initial_ptrs: impl IntoIterator<Item = ((u16, ExpertBlobRole), Vec<u64>)>,
    ) -> Result<Self, PagerError> {
        let shadow: HashMap<_, _> = initial_ptrs.into_iter().collect();
        for ((layer, role), v) in shadow.iter() {
            if v.len() != n_exp {
                return Err(PagerError::StrideMismatch {
                    name: format!("layer {layer} {role:?} pointer table"),
                    got: v.len(),
                    want: n_exp,
                });
            }
        }
        // Every entry starts aimed at the pool's base, so entry 0 IS the base.
        let expected_base = shadow
            .iter()
            .filter_map(|(&k, v)| v.first().map(|&b| (k, b)))
            .collect();
        Ok(Self {
            pager,
            catalog,
            shadow,
            encode_scratch: vec![0f32; 2 * n_exp],
            expected_base,
            n_exp,
        })
    }

    pub fn slots_per_blob(&self) -> usize {
        self.pager.slots_per_blob()
    }

    pub fn stats(&self) -> (u64, u64) {
        self.pager.stats()
    }

    pub fn hit_rate(&self) -> f64 {
        self.pager.hit_rate()
    }

    pub fn catalog(&self) -> &ExpertCatalog {
        &self.catalog
    }

    /// Host shadow of a pointer table, for tests and for re-upload.
    pub fn shadow(&self, layer: u16, role: ExpertBlobRole) -> Option<&[u64]> {
        self.shadow.get(&(layer, role)).map(|v| v.as_slice())
    }

    /// Resolve every expert in `experts` to a slot and patch the host shadow.
    ///
    /// Pure bookkeeping: no allocation, no I/O. `out` is filled with one
    /// [`SlotFill`] per requested expert, in request order; the caller reads
    /// the bytes for entries with `needs_read` before dispatching.
    ///
    /// Fails closed BEFORE mutating anything if the working set cannot fit
    /// (more distinct experts than slots would make them evict each other
    /// mid-dispatch) or if any expert is absent from the catalog.
    pub fn plan_dispatch(
        &mut self,
        layer: u16,
        role: ExpertBlobRole,
        experts: &[u16],
        stride: usize,
        blob_base: u64,
        out: &mut Vec<SlotFill>,
    ) -> Result<bool, PagerError> {
        out.clear();
        // Whether any pointer-table entry actually changed. A dispatch whose
        // experts are all still in the slots they were last in changes
        // nothing, and re-uploading an identical table costs a host-to-device
        // copy per layer per role per token — 86 per token on ds4, on the path
        // whose per-dispatch overhead dominates paged decode.
        let mut dirty = false;
        // The blob must be the pool this pager was built over. Layer ids are
        // not globally unique across ds4's layer-shaped blocks — a DSpark
        // stage is also `s = 0, 1, 2` — so a mis-wired caller could otherwise
        // hand over a fully-resident block's blob and have us page layer 0's
        // experts into it at layer 0's offsets. Check identity, not index.
        match self.expected_base.get(&(layer, role)) {
            Some(&want) if want == blob_base => {}
            _ => {
                return Err(PagerError::NotCatalogued {
                    key: ExpertKey {
                        layer,
                        expert: 0,
                        role,
                    },
                })
            }
        }
        // Distinct-expert count for THIS dispatch. Every one of them must be
        // resident simultaneously, so the pool has to hold all of them.
        let mut distinct = 0usize;
        for (i, &e) in experts.iter().enumerate() {
            if !experts[..i].contains(&e) {
                distinct += 1;
            }
        }
        if distinct > self.pager.slots_per_blob() {
            return Err(PagerError::Sizing(PagerSizingError::BelowMinimum {
                needed_slots: distinct,
                got_slots: self.pager.slots_per_blob(),
            }));
        }
        // Verify catalog coverage for every expert BEFORE touching residency,
        // so a bad request cannot leave half-updated bookkeeping behind.
        for &e in experts {
            let key = ExpertKey {
                layer,
                expert: e,
                role,
            };
            if e as usize >= self.n_exp {
                return Err(PagerError::NotCatalogued { key });
            }
            if self.catalog.segments(key).is_none() {
                return Err(PagerError::NotCatalogued { key });
            }
        }
        let shadow = self
            .shadow
            .get_mut(&(layer, role))
            .ok_or(PagerError::NotCatalogued {
                key: ExpertKey {
                    layer,
                    expert: 0,
                    role,
                },
            })?;
        for &e in experts {
            let key = ExpertKey {
                layer,
                expert: e,
                role,
            };
            let (slot, needs_read) = self.pager.resolve_slot(key);
            // Repoint unconditionally, hits included: cheap, and it removes
            // any dependence on a previous dispatch having left the entry
            // correct. A routed expert must never be left aimed at whatever
            // the loader wrote (in the EP shard path that is a ZEROED dummy,
            // which produces silence rather than an error).
            let want = blob_base + (slot * stride) as u64;
            if shadow[e as usize] != want {
                shadow[e as usize] = want;
                dirty = true;
            }
            out.push(SlotFill {
                key,
                slot,
                needs_read,
            });
        }
        Ok(dirty)
    }

    /// Encode the host shadow into the two-F32-slot device representation.
    /// Returns the slice to upload into the layer's pointer table.
    pub fn encoded_ptr_table(&mut self, layer: u16, role: ExpertBlobRole) -> Option<&[f32]> {
        let shadow = self.shadow.get(&(layer, role))?;
        for (e, &p) in shadow.iter().enumerate() {
            encode_ptr_slots(&mut self.encode_scratch, e, p);
        }
        Some(&self.encode_scratch)
    }

    /// Segments to read for a miss, in blob order.
    pub fn segments_for(&self, key: ExpertKey) -> Result<&[ExpertSegment], PagerError> {
        self.catalog
            .segments(key)
            .ok_or(PagerError::NotCatalogued { key })
    }
}

/// Optional access-trace sink for the offline policy simulator
/// (`crate::expert_policy`). `HIPFIRE_DEEPSEEK4_EXPERT_TRACE=<path>` records
/// one `seq,layer,role,expert` row per requested expert.
///
/// This is the P0 instrumentation from Kaden's weight-pager spec: decide
/// whether the eviction policy is worth changing by replaying real routing
/// offline, rather than by arguing about locality. Off by default.
struct ExpertTrace {
    out: std::io::BufWriter<std::fs::File>,
    seq: u64,
}

impl ExpertTrace {
    fn open() -> Option<Self> {
        let path = std::env::var("HIPFIRE_DEEPSEEK4_EXPERT_TRACE").ok()?;
        match std::fs::File::create(&path) {
            Ok(f) => {
                eprintln!("deepseek4: expert access trace -> {path}");
                Some(Self {
                    out: std::io::BufWriter::new(f),
                    seq: 0,
                })
            }
            Err(e) => {
                eprintln!("deepseek4: cannot open expert trace {path}: {e}");
                None
            }
        }
    }

    fn record(&mut self, layer: u16, role: ExpertBlobRole, experts: &[u16]) {
        use std::io::Write;
        let r = match role {
            ExpertBlobRole::GateUp => 'g',
            ExpertBlobRole::Down => 'd',
        };
        for &e in experts {
            let _ = writeln!(self.out, "{},{layer},{r},{e}", self.seq);
        }
        self.seq += 1;
    }
}

/// The GPU-touching half of paging: owns the transport and every scratch
/// buffer, so a cache miss on the forward path allocates nothing.
pub struct Ds4ExpertPaging {
    rt: Ds4PagingRuntime,
    transport: hipfire_runtime::weight_pager::PreadH2DTransport,
    /// Scratch, all pre-sized at construction.
    fills: Vec<SlotFill>,
    experts: Vec<u16>,
    ptr_bytes: Vec<u8>,
    topk_bytes: Vec<u8>,
    tile_bytes: Vec<u8>,
    tile_experts: Vec<i32>,
    /// Bytes actually read from the HFQ. With the hit rate this is the whole
    /// budget/throughput story: a bigger pool is only worth its memory if it
    /// moves these numbers.
    trace: Option<ExpertTrace>,
    bytes_read: u64,
    dispatches: u64,
    table_uploads: u64,
    table_uploads_skipped: u64,
}

/// How many `ensure_resident` calls between hit-rate reports. One decode token
/// makes `n_layers * 2` of them (43 * 2 = 86 on ds4), so this is roughly every
/// 100 tokens — often enough to watch a run, rare enough not to spam.
const STATS_EVERY_DEFAULT: u64 = 8192;

/// Dispatches between hit-rate reports. `HIPFIRE_DEEPSEEK4_CACHE_STATS_EVERY`
/// lowers it for short measurement runs — a 192-token generation only makes
/// ~2.3k dispatches, so the default would never print.
fn stats_every() -> u64 {
    use std::sync::OnceLock;
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("HIPFIRE_DEEPSEEK4_CACHE_STATS_EVERY")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(STATS_EVERY_DEFAULT)
    })
}

impl Ds4ExpertPaging {
    /// `max_working_set` is the largest number of experts a single dispatch
    /// can ask for (`k_top` for decode, the chunk union for prefill); scratch
    /// is sized for it once so the forward path never grows a buffer.
    pub fn new(
        rt: Ds4PagingRuntime,
        mut transport: hipfire_runtime::weight_pager::PreadH2DTransport,
        n_exp: usize,
        max_working_set: usize,
        max_expert_bytes: usize,
    ) -> Self {
        transport.reserve_staging(max_expert_bytes);
        Self {
            rt,
            transport,
            fills: Vec::with_capacity(max_working_set),
            experts: Vec::with_capacity(max_working_set),
            ptr_bytes: vec![0u8; n_exp * 8],
            topk_bytes: vec![0u8; max_working_set * 4],
            tile_bytes: Vec::new(),
            tile_experts: Vec::new(),
            trace: ExpertTrace::open(),
            bytes_read: 0,
            dispatches: 0,
            table_uploads: 0,
            table_uploads_skipped: 0,
        }
    }

    pub fn runtime(&self) -> &Ds4PagingRuntime {
        &self.rt
    }

    pub fn stats(&self) -> (u64, u64) {
        self.rt.stats()
    }

    pub fn hit_rate(&self) -> f64 {
        self.rt.hit_rate()
    }

    /// Bytes read from the HFQ since load.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    pub fn slots_per_blob(&self) -> usize {
        self.rt.slots_per_blob()
    }

    /// Read `count` i32 top-k indices from a device buffer into a host Vec of
    /// expert ids, clamped to the routed-expert range. Reuses scratch.
    pub fn read_topk(
        &mut self,
        topk_indices: &rdna_compute::GpuTensor,
        count: usize,
        n_exp: usize,
        gpu: &rdna_compute::Gpu,
    ) -> Result<&[u16], String> {
        let need = count * 4;
        if self.topk_bytes.len() < need {
            // Only reachable if a caller exceeds the max_working_set it
            // declared at construction; grow rather than read out of bounds.
            self.topk_bytes.resize(need, 0);
        }
        gpu.bind_thread().map_err(|e| format!("{e:?}"))?;
        gpu.hip
            .memcpy_dtoh(&mut self.topk_bytes[..need], &topk_indices.buf)
            .map_err(|e| format!("d2h topk indices: {e:?}"))?;
        self.experts.clear();
        for c in self.topk_bytes[..need].chunks_exact(4) {
            let v = i32::from_ne_bytes([c[0], c[1], c[2], c[3]]);
            // A negative or out-of-range index would index outside the
            // pointer table. Clamp exactly as the hash-router fallback does.
            let e = v.clamp(0, n_exp as i32 - 1) as u16;
            self.experts.push(e);
        }
        Ok(&self.experts)
    }

    /// Read `n_tiles` i32 expert ids from the scatter's `expert_tile_ids`.
    ///
    /// That tensor is `DType::Raw` with its shape in BYTES, so `sub_offset` +
    /// `download_f32` reads the wrong length — copy the bytes and reinterpret.
    /// Reuses scratch so a per-layer call allocates nothing after the first.
    pub fn read_tile_experts(
        &mut self,
        expert_tile_ids: &rdna_compute::GpuTensor,
        n_tiles: usize,
        gpu: &rdna_compute::Gpu,
    ) -> Result<&[i32], String> {
        let need = n_tiles * 4;
        if expert_tile_ids.byte_size() < need {
            return Err(format!(
                "expert_tile_ids holds {} B, need {need} B for {n_tiles} tiles",
                expert_tile_ids.byte_size()
            ));
        }
        if self.tile_bytes.len() < need {
            self.tile_bytes.resize(need, 0);
        }
        gpu.bind_thread().map_err(|e| format!("{e:?}"))?;
        gpu.hip
            .memcpy_dtoh(&mut self.tile_bytes[..need], &expert_tile_ids.buf)
            .map_err(|e| format!("d2h expert_tile_ids: {e:?}"))?;
        self.tile_experts.clear();
        for c in self.tile_bytes[..need].chunks_exact(4) {
            self.tile_experts
                .push(i32::from_ne_bytes([c[0], c[1], c[2], c[3]]));
        }
        Ok(&self.tile_experts)
    }

    /// Whether this layer's experts are paged. False for layers left fully
    /// resident (the MTP head and any DSpark sidecar), which must take the
    /// unchanged path.
    pub fn pages_layer(&self, layer: u16) -> bool {
        self.rt.shadow(layer, ExpertBlobRole::GateUp).is_some()
    }

    /// Page in everything one MoE dispatch needs: read the routed expert ids
    /// from the device top-k buffer, make them resident in BOTH blobs, and
    /// repoint both pointer tables.
    ///
    /// This is the whole forward-path contract. Call it after routing and
    /// before the expert GEMVs; on return every routed expert of this layer is
    /// resident and its table entry aims at the slot holding it.
    #[allow(clippy::too_many_arguments)]
    pub fn page_dispatch(
        &mut self,
        layer: u16,
        topk_indices: &rdna_compute::GpuTensor,
        count: usize,
        n_exp: usize,
        gate_up_blob: &rdna_compute::GpuTensor,
        gate_up_ptrs: &rdna_compute::GpuTensor,
        gate_up_stride: usize,
        w2_blob: &rdna_compute::GpuTensor,
        w2_ptrs: &rdna_compute::GpuTensor,
        w2_stride: usize,
        gpu: &mut rdna_compute::Gpu,
    ) -> Result<(), PagerError> {
        self.read_topk(topk_indices, count, n_exp, gpu)
            .map_err(|detail| PagerError::Read {
                key: ExpertKey {
                    layer,
                    expert: 0,
                    role: ExpertBlobRole::GateUp,
                },
                offset: 0,
                len: count * 4,
                detail,
            })?;
        // Move the id list out so `self` can be borrowed mutably below; put it
        // back on every exit so the buffer is reused, never reallocated.
        let experts = std::mem::take(&mut self.experts);
        let r = self.page_experts_both(
            layer,
            &experts,
            gate_up_blob,
            gate_up_ptrs,
            gate_up_stride,
            w2_blob,
            w2_ptrs,
            w2_stride,
            gpu,
        );
        self.experts = experts;
        r
    }

    /// [`Ds4ExpertPaging::page_dispatch`] over a caller-supplied expert list —
    /// used by prefill, which computes the union across a window of tokens
    /// rather than reading one token's top-k.
    #[allow(clippy::too_many_arguments)]
    pub fn page_experts_both(
        &mut self,
        layer: u16,
        experts: &[u16],
        gate_up_blob: &rdna_compute::GpuTensor,
        gate_up_ptrs: &rdna_compute::GpuTensor,
        gate_up_stride: usize,
        w2_blob: &rdna_compute::GpuTensor,
        w2_ptrs: &rdna_compute::GpuTensor,
        w2_stride: usize,
        gpu: &mut rdna_compute::Gpu,
    ) -> Result<(), PagerError> {
        self.ensure_resident(
            layer,
            ExpertBlobRole::GateUp,
            experts,
            gate_up_blob,
            gate_up_ptrs,
            gate_up_stride,
            gpu,
        )?;
        self.ensure_resident(
            layer,
            ExpertBlobRole::Down,
            experts,
            w2_blob,
            w2_ptrs,
            w2_stride,
            gpu,
        )
    }

    /// Make every expert in `experts` resident in this (layer, role) blob and
    /// repoint the device pointer table at their slots.
    ///
    /// Allocation-free: slots are reused in place and every buffer is scratch
    /// sized at construction. Any read failure is reported with layer, expert
    /// and file offset rather than leaving a stale or zeroed expert behind.
    #[allow(clippy::too_many_arguments)]
    pub fn ensure_resident(
        &mut self,
        layer: u16,
        role: ExpertBlobRole,
        experts: &[u16],
        blob: &rdna_compute::GpuTensor,
        ptr_table: &rdna_compute::GpuTensor,
        stride: usize,
        gpu: &mut rdna_compute::Gpu,
    ) -> Result<(), PagerError> {
        use hipfire_runtime::weight_pager::Transport;

        if let Some(t) = self.trace.as_mut() {
            t.record(layer, role, experts);
        }
        let base = blob.buf.as_ptr() as u64;
        // Move scratch out so `self.rt` can borrow mutably alongside it.
        let mut fills = std::mem::take(&mut self.fills);
        let dirty = match self
            .rt
            .plan_dispatch(layer, role, experts, stride, base, &mut fills)
        {
            Ok(d) => d,
            Err(e) => {
                self.fills = fills;
                return Err(e);
            }
        };
        for f in &fills {
            if !f.needs_read {
                continue;
            }
            // Copy the segment list out (max 2: w1 ‖ w3) so the catalog borrow
            // ends before the transport borrow begins.
            let mut segbuf = [ExpertSegment { offset: 0, len: 0 }; 2];
            let n_seg = {
                let segs = match self.rt.segments_for(f.key) {
                    Ok(s) => s,
                    Err(e) => {
                        self.fills = fills;
                        return Err(e);
                    }
                };
                let n = segs.len().min(segbuf.len());
                segbuf[..n].copy_from_slice(&segs[..n]);
                n
            };
            let mut dst = f.slot * stride;
            for seg in &segbuf[..n_seg] {
                if let Err(e) = self
                    .transport
                    .fetch_into(seg.offset, seg.len, blob, dst, gpu)
                {
                    let err = PagerError::Read {
                        key: f.key,
                        offset: seg.offset,
                        len: seg.len,
                        detail: format!("{e:?}"),
                    };
                    self.fills = fills;
                    return Err(err);
                }
                self.bytes_read += seg.len as u64;
                dst += seg.len;
            }
        }
        self.fills = fills;

        // Periodic hit rate + bytes read. The budget/throughput curve should
        // be measured, not argued about, so make the inputs to it visible on
        // any paged run rather than behind another env knob.
        self.dispatches += 1;
        if self.dispatches % stats_every() == 0 {
            let (hits, misses) = self.rt.stats();
            eprintln!(
                "deepseek4: expert cache — {:.1}% hits ({hits} hit / {misses} miss), \
                 {:.1} GiB read from disk, {} slots/blob",
                self.rt.hit_rate() * 100.0,
                self.bytes_read as f64 / (1024.0 * 1024.0 * 1024.0),
                self.rt.slots_per_blob(),
            );
            eprintln!(
                "deepseek4: expert cache — pointer-table uploads {} done / {} skipped ({:.1}% skipped)",
                self.table_uploads,
                self.table_uploads_skipped,
                100.0 * self.table_uploads_skipped as f64
                    / (self.table_uploads + self.table_uploads_skipped).max(1) as f64,
            );
        }

        if !dirty {
            // Every requested expert already points at the slot holding it, so
            // the device table is already correct and the copy is pure cost.
            self.table_uploads_skipped += 1;
            return Ok(());
        }
        self.table_uploads += 1;

        // Publish the patched table. Uploaded as native u64 bytes, exactly the
        // encoding the loader writes — the kernel reinterprets each u64 as two
        // F32 slots (pinned by `ptr_table_u64_bytes_match_the_f32_slot_encoding`).
        let shadow = self
            .rt
            .shadow(layer, role)
            .ok_or(PagerError::NotCatalogued {
                key: ExpertKey {
                    layer,
                    expert: 0,
                    role,
                },
            })?;
        self.ptr_bytes.clear();
        for p in shadow {
            self.ptr_bytes.extend_from_slice(&p.to_ne_bytes());
        }
        gpu.memcpy_htod_auto(&ptr_table.buf, &self.ptr_bytes)
            .map_err(|e| PagerError::Read {
                key: ExpertKey {
                    layer,
                    expert: 0,
                    role,
                },
                offset: 0,
                len: self.ptr_bytes.len(),
                detail: format!("pointer-table upload: {e:?}"),
            })?;
        Ok(())
    }
}

/// Split a prefill chunk into token windows whose routed-expert union fits the
/// slot pool.
///
/// A prefill chunk of B tokens routes to up to `B * k_top` distinct experts,
/// far more than a bounded pool holds, so the MoE dispatch runs one window at
/// a time. Windows are greedy and contiguous, which keeps each one a plain
/// row range of the batch tensors (so a window is a `sub_offset` view, not a
/// gather) — and contiguity is what makes a window's output identical to the
/// same rows inside the full batch.
///
/// Returns `(start, len)` pairs covering `0..batch`. Errors only if one single
/// token needs more slots than exist, which `plan_slots`' floor already rules
/// out at load.
pub fn plan_prefill_windows(
    topk: &[u16],
    batch: usize,
    k_top: usize,
    slots: usize,
) -> Result<Vec<(usize, usize)>, PagerError> {
    let mut windows = Vec::new();
    let mut start = 0usize;
    while start < batch {
        let mut union: Vec<u16> = Vec::with_capacity(slots);
        let mut end = start;
        while end < batch {
            let row = &topk[end * k_top..(end + 1) * k_top];
            let mut added: Vec<u16> = Vec::new();
            for &e in row {
                if !union.contains(&e) && !added.contains(&e) {
                    added.push(e);
                }
            }
            if union.len() + added.len() > slots {
                break;
            }
            union.extend_from_slice(&added);
            end += 1;
        }
        if end == start {
            // One token alone exceeds the pool. plan_slots' floor makes this
            // unreachable, but fail closed rather than loop forever.
            let distinct = {
                let row = &topk[start * k_top..(start + 1) * k_top];
                let mut u: Vec<u16> = Vec::new();
                for &e in row {
                    if !u.contains(&e) {
                        u.push(e);
                    }
                }
                u.len()
            };
            return Err(PagerError::Sizing(PagerSizingError::BelowMinimum {
                needed_slots: distinct,
                got_slots: slots,
            }));
        }
        windows.push((start, end - start));
        start = end;
    }
    Ok(windows)
}

/// Distinct routed experts across a window of tokens, in first-seen order.
pub fn window_expert_union(topk: &[u16], start: usize, len: usize, k_top: usize) -> Vec<u16> {
    let mut u: Vec<u16> = Vec::new();
    for t in start..start + len {
        for &e in &topk[t * k_top..(t + 1) * k_top] {
            if !u.contains(&e) {
                u.push(e);
            }
        }
    }
    u
}

/// One expert band: a contiguous tile range of the scatter's expert-ordered
/// slots, plus the experts it covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertBand {
    pub tile_begin: usize,
    pub tile_count: usize,
    pub experts: Vec<u16>,
}

/// Group the scatter's per-tile expert ids into bands of at most `slots`
/// distinct experts.
///
/// `tile_experts[t]` is the expert that tile `t` belongs to, in the scatter's
/// order — experts are contiguous, so a band is a tile RANGE and needs no
/// gather. This is what turns paged prefill from "re-read experts per token
/// window" (redundancy grows with prompt length) into "read each expert once
/// per chunk" (redundancy 1.0).
///
/// Tiles with a negative/sentinel expert id are padding and are folded into
/// the current band without counting against its budget.
pub fn plan_expert_bands(tile_experts: &[i32], n_exp: usize, slots: usize) -> Vec<ExpertBand> {
    let mut bands: Vec<ExpertBand> = Vec::new();
    if slots == 0 {
        return bands;
    }
    let mut begin = 0usize;
    let mut experts: Vec<u16> = Vec::new();
    for (t, &e) in tile_experts.iter().enumerate() {
        let valid = e >= 0 && (e as usize) < n_exp;
        if valid && !experts.contains(&(e as u16)) && experts.len() == slots {
            // This tile starts an expert that will not fit — close the band
            // here so every band's working set fits the pool.
            bands.push(ExpertBand {
                tile_begin: begin,
                tile_count: t - begin,
                experts: std::mem::take(&mut experts),
            });
            begin = t;
        }
        if valid && !experts.contains(&(e as u16)) {
            experts.push(e as u16);
        }
    }
    if begin < tile_experts.len() {
        bands.push(ExpertBand {
            tile_begin: begin,
            tile_count: tile_experts.len() - begin,
            experts,
        });
    }
    bands
}

/// `HIPFIRE_DEEPSEEK4_PREFILL_WORK_TRACE=1` — per-layer paged-prefill work
/// accounting (windows, distinct experts, actual loads, redundancy factor).
///
/// Counts, not timings: wall-clock on a shared box is confounded by anything
/// else touching the same disk, but "how many expert reads did we issue
/// versus the minimum possible" is not.
pub fn prefill_work_trace() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("HIPFIRE_DEEPSEEK4_PREFILL_WORK_TRACE").as_deref() == Ok("1"))
}

/// Environment knob that turns paging on. Unset (or unparseable, or `0`) keeps
/// today's fully-resident behaviour.
pub const EXPERT_CACHE_GB_ENV: &str = "HIPFIRE_DEEPSEEK4_EXPERT_CACHE_GB";

/// Parse [`EXPERT_CACHE_GB_ENV`]. `None` = page nothing (fully resident,
/// today's behaviour). Unparseable input yields `None` so a typo degrades to
/// the safe path rather than to an arbitrary budget.
pub fn expert_cache_budget_bytes(raw: Option<&str>) -> Option<u64> {
    let v = raw?.trim().parse::<u64>().ok()?;
    if v == 0 {
        return None;
    }
    Some(v * 1024 * 1024 * 1024)
}

/// Bytes usable for the slot pool, given MemAvailable and everything the pager
/// does NOT own (non-routed weights, KV/SWA caches, per-step scratch, headroom).
/// Saturates at zero so an over-subscribed box yields a clean sizing error from
/// [`plan_slots`] rather than an underflowed, enormous budget.
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

    /// Test pool base. `plan_dispatch` checks blob identity, so every test
    /// must hand over the same base the runtime was built with.
    const TEST_BASE: u64 = 0x7f00_0000_0000;

    fn runtime_with(n_exp: usize, slots: usize, layers: &[u16]) -> Ds4PagingRuntime {
        let stride = 1024usize;
        let prefixes: Vec<String> = layers.iter().map(|l| format!("layers.{l}")).collect();
        let idx = fake_index(
            &prefixes.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            n_exp,
            stride,
        );
        let pairs: Vec<(u16, String)> =
            layers.iter().map(|&l| (l, format!("layers.{l}"))).collect();
        let cat = ExpertCatalog::build_from(&pairs, n_exp, |s| s, lookup_from(&idx)).expect("cat");
        let mut init = Vec::new();
        for &l in layers {
            for role in [ExpertBlobRole::GateUp, ExpertBlobRole::Down] {
                init.push(((l, role), vec![TEST_BASE; n_exp]));
            }
        }
        Ds4PagingRuntime::new(Ds4ExpertPager::new(slots), cat, n_exp, init).expect("runtime")
    }

    #[test]
    fn plan_dispatch_reads_on_first_touch_and_reuses_on_the_second() {
        let mut rt = runtime_with(16, 8, &[0]);
        let mut out = Vec::with_capacity(8);
        rt.plan_dispatch(
            0,
            ExpertBlobRole::GateUp,
            &[3, 9, 1],
            2048,
            TEST_BASE,
            &mut out,
        )
        .expect("plans");
        assert!(out.iter().all(|f| f.needs_read), "first touch must read");
        rt.plan_dispatch(
            0,
            ExpertBlobRole::GateUp,
            &[3, 9, 1],
            2048,
            TEST_BASE,
            &mut out,
        )
        .expect("plans");
        assert!(
            out.iter().all(|f| !f.needs_read),
            "second touch must be all hits"
        );
    }

    #[test]
    fn plan_dispatch_repoints_every_requested_expert_including_hits() {
        // A routed expert left pointing at the loader's entry reads the wrong
        // weights (or, in the EP shard path, a zeroed dummy that yields
        // silence). Every requested expert must be repointed every dispatch.
        let mut rt = runtime_with(16, 8, &[0]);
        let mut out = Vec::with_capacity(8);
        let (stride, base) = (2048usize, TEST_BASE);
        rt.plan_dispatch(0, ExpertBlobRole::Down, &[5, 2], stride, base, &mut out)
            .expect("plans");
        rt.plan_dispatch(0, ExpertBlobRole::Down, &[5, 2], stride, base, &mut out)
            .expect("plans");
        let shadow = rt.shadow(0, ExpertBlobRole::Down).unwrap();
        for f in &out {
            assert_eq!(
                shadow[f.key.expert as usize],
                base + (f.slot * stride) as u64,
                "expert {} not aimed at its slot",
                f.key.expert
            );
        }
        // Experts nobody asked for keep the loader's entry untouched.
        assert_eq!(shadow[7], base, "unrequested expert was repointed");
    }

    #[test]
    fn plan_dispatch_refuses_a_working_set_larger_than_the_pool() {
        // 4 slots cannot hold 5 distinct experts at once; they would evict
        // each other mid-dispatch and the kernel would read the wrong weights.
        let mut rt = runtime_with(16, 4, &[0]);
        let mut out = Vec::new();
        let err = rt
            .plan_dispatch(
                0,
                ExpertBlobRole::GateUp,
                &[0, 1, 2, 3, 4],
                1024,
                TEST_BASE,
                &mut out,
            )
            .expect_err("must fail closed");
        assert!(
            matches!(
                err,
                PagerError::Sizing(PagerSizingError::BelowMinimum {
                    needed_slots: 5,
                    got_slots: 4
                })
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn plan_dispatch_counts_duplicates_once() {
        // Prefill unions can repeat an expert; the pool only needs it once.
        let mut rt = runtime_with(16, 2, &[0]);
        let mut out = Vec::new();
        rt.plan_dispatch(
            0,
            ExpertBlobRole::GateUp,
            &[7, 7, 7, 1, 1],
            1024,
            TEST_BASE,
            &mut out,
        )
        .expect("2 distinct experts fit in 2 slots");
        assert_eq!(out.len(), 5, "one fill per request, duplicates included");
        assert_eq!(out[0].slot, out[1].slot);
        assert!(out[0].needs_read && !out[1].needs_read);
    }

    #[test]
    fn plan_dispatch_rejects_an_uncatalogued_expert_without_mutating_state() {
        let mut rt = runtime_with(8, 4, &[0]);
        let mut out = Vec::new();
        let err = rt
            .plan_dispatch(0, ExpertBlobRole::GateUp, &[1, 99], 1024, 0x1000, &mut out)
            .expect_err("must fail closed");
        assert!(
            matches!(err, PagerError::NotCatalogued { .. }),
            "got {err:?}"
        );
        // The valid expert in the same request must not have been paged in:
        // a rejected dispatch leaves no half-applied bookkeeping behind.
        assert_eq!(rt.stats(), (0, 0), "residency mutated on a rejected plan");
        assert_eq!(rt.shadow(0, ExpertBlobRole::GateUp).unwrap()[1], TEST_BASE);
    }

    #[test]
    fn encoded_ptr_table_round_trips_the_shadow() {
        let mut rt = runtime_with(4, 2, &[0]);
        let mut out = Vec::new();
        let (stride, base) = (4096usize, TEST_BASE);
        rt.plan_dispatch(0, ExpertBlobRole::Down, &[3], stride, base, &mut out)
            .expect("plans");
        let want = rt.shadow(0, ExpertBlobRole::Down).unwrap().to_vec();
        let enc = rt.encoded_ptr_table(0, ExpertBlobRole::Down).unwrap();
        assert_eq!(enc.len(), 2 * 4);
        for (e, &p) in want.iter().enumerate() {
            let lo = enc[e * 2].to_bits() as u64;
            let hi = enc[e * 2 + 1].to_bits() as u64;
            assert_eq!((hi << 32) | lo, p, "expert {e} pointer did not round-trip");
        }
    }

    #[test]
    fn plan_dispatch_rejects_a_blob_this_pager_does_not_own() {
        // Layer ids repeat across ds4's layer-shaped blocks (a DSpark stage is
        // also s=0), so a mis-wired caller could hand over a fully-resident
        // block's blob. Paging into it would corrupt weights silently.
        let mut rt = runtime_with(8, 4, &[0]);
        let mut out = Vec::new();
        let err = rt
            .plan_dispatch(0, ExpertBlobRole::GateUp, &[1], 1024, 0xDEAD_0000, &mut out)
            .expect_err("must reject a foreign blob");
        assert!(
            matches!(err, PagerError::NotCatalogued { .. }),
            "got {err:?}"
        );
        assert_eq!(rt.stats(), (0, 0), "residency mutated on a rejected plan");
    }

    #[test]
    fn ptr_table_u64_bytes_match_the_f32_slot_encoding() {
        // `ensure_resident` uploads the u64 shadow as native bytes, which is
        // what the loader writes; the kernel reads it as two F32 slots per
        // pointer. Those must be the same bytes, or every paged pointer is
        // garbage. (True on little-endian, which the loader already assumes.)
        let ptrs: [u64; 3] = [0x7f00_0000_1000, 0, u64::MAX];
        let mut via_u64 = Vec::new();
        for p in &ptrs {
            via_u64.extend_from_slice(&p.to_ne_bytes());
        }
        let mut slots = vec![0f32; ptrs.len() * 2];
        for (e, &p) in ptrs.iter().enumerate() {
            encode_ptr_slots(&mut slots, e, p);
        }
        let mut via_f32 = Vec::new();
        for s in &slots {
            via_f32.extend_from_slice(&s.to_ne_bytes());
        }
        assert_eq!(via_u64, via_f32);
    }

    #[test]
    fn layers_do_not_share_pointer_tables() {
        let stride = 1024usize;
        let mut rt = runtime_with(8, 4, &[0, 1]);
        let mut out = Vec::new();
        // Two experts so the second lands in slot 1 — a repoint that is
        // distinguishable from the load-time value, which IS slot 0's address.
        rt.plan_dispatch(
            0,
            ExpertBlobRole::GateUp,
            &[2, 5],
            stride,
            TEST_BASE,
            &mut out,
        )
        .expect("plans");
        assert_eq!(
            rt.shadow(0, ExpertBlobRole::GateUp).unwrap()[5],
            TEST_BASE + stride as u64,
            "layer 0 expert 5 should hold slot 1"
        );
        assert_eq!(
            rt.shadow(1, ExpertBlobRole::GateUp).unwrap()[5],
            TEST_BASE,
            "layer 0 dispatch leaked into layer 1's table"
        );
    }

    #[test]
    fn bands_cover_every_tile_exactly_once() {
        // A dropped or duplicated tile is a silently wrong prefill.
        let tiles: Vec<i32> = vec![0, 0, 1, 2, 2, 2, 3, 4, 5, 5, 6, 7];
        let bands = plan_expert_bands(&tiles, 256, 3);
        assert!(bands.len() > 1, "expected several bands, got {bands:?}");
        let mut next = 0usize;
        for b in &bands {
            assert_eq!(b.tile_begin, next, "bands must be contiguous: {bands:?}");
            assert!(b.tile_count > 0);
            next = b.tile_begin + b.tile_count;
        }
        assert_eq!(next, tiles.len(), "bands must cover every tile");
    }

    #[test]
    fn bands_never_exceed_the_pool() {
        let tiles: Vec<i32> = (0..40).map(|i| i / 2).collect();
        for slots in [1usize, 3, 7, 16] {
            for b in plan_expert_bands(&tiles, 256, slots) {
                assert!(b.experts.len() <= slots, "band {b:?} exceeds {slots} slots");
            }
        }
    }

    #[test]
    fn bands_read_each_expert_once() {
        // The whole point: across all bands, every distinct expert appears in
        // exactly one band, so it is paged in exactly once per chunk.
        let tiles: Vec<i32> = vec![0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 8];
        let bands = plan_expert_bands(&tiles, 256, 2);
        let mut seen: Vec<u16> = Vec::new();
        for b in &bands {
            for &e in &b.experts {
                assert!(!seen.contains(&e), "expert {e} paged in twice: {bands:?}");
                seen.push(e);
            }
        }
        let mut distinct: Vec<u16> = tiles.iter().map(|&e| e as u16).collect();
        distinct.sort_unstable();
        distinct.dedup();
        seen.sort_unstable();
        assert_eq!(seen, distinct, "every expert must be covered once");
    }

    #[test]
    fn bands_fold_padding_tiles_without_spending_budget() {
        // Scatter pads with sentinel tiles; they must not consume a slot.
        let tiles: Vec<i32> = vec![0, -1, -1, 1, -1, 2];
        let bands = plan_expert_bands(&tiles, 256, 3);
        assert_eq!(
            bands.len(),
            1,
            "padding should not force a new band: {bands:?}"
        );
        assert_eq!(bands[0].experts, vec![0, 1, 2]);
        assert_eq!(bands[0].tile_count, 6);
    }

    #[test]
    fn bands_of_one_slot_still_make_progress() {
        let tiles: Vec<i32> = vec![0, 1, 2, 3];
        let bands = plan_expert_bands(&tiles, 256, 1);
        assert_eq!(bands.len(), 4);
        assert!(bands.iter().all(|b| b.tile_count == 1));
    }

    #[test]
    fn prefill_windows_cover_every_token_exactly_once() {
        // 10 tokens, k=2, distinct experts everywhere: 20 distinct ids and a
        // pool of 6 forces several windows. Coverage must still be exact —
        // a dropped or repeated token is a silently wrong prefill.
        let k = 2usize;
        let topk: Vec<u16> = (0..20u16).collect();
        let w = plan_prefill_windows(&topk, 10, k, 6).expect("plans");
        assert!(w.len() > 1, "expected several windows, got {w:?}");
        let mut next = 0usize;
        for &(start, len) in &w {
            assert_eq!(start, next, "windows must be contiguous: {w:?}");
            assert!(len > 0);
            next = start + len;
        }
        assert_eq!(next, 10, "windows must cover the whole batch");
    }

    #[test]
    fn prefill_windows_never_exceed_the_pool() {
        let k = 3usize;
        let slots = 5usize;
        // Deliberately churny routing so unions grow fast.
        let topk: Vec<u16> = (0..36u16).map(|i| (i * 7) % 32).collect();
        let batch = topk.len() / k;
        let w = plan_prefill_windows(&topk, batch, k, slots).expect("plans");
        for &(start, len) in &w {
            let u = window_expert_union(&topk, start, len, k);
            assert!(
                u.len() <= slots,
                "window ({start},{len}) needs {} slots, pool has {slots}",
                u.len()
            );
        }
    }

    #[test]
    fn prefill_windows_take_the_whole_batch_when_it_fits() {
        // A generous pool must not fragment the batch — fragmenting costs
        // grouped-GEMM efficiency for nothing.
        let k = 2usize;
        let topk: Vec<u16> = vec![1, 2, 1, 2, 3, 1, 2, 3];
        let w = plan_prefill_windows(&topk, 4, k, 64).expect("plans");
        assert_eq!(w, vec![(0, 4)]);
    }

    #[test]
    fn prefill_windows_fail_closed_when_one_token_cannot_fit() {
        // 4 distinct experts for one token, 3 slots: no window can be formed.
        // Must error rather than emit a zero-length window and spin forever.
        let topk: Vec<u16> = vec![1, 2, 3, 4];
        let err = plan_prefill_windows(&topk, 1, 4, 3).expect_err("must fail closed");
        assert!(
            matches!(
                err,
                PagerError::Sizing(PagerSizingError::BelowMinimum {
                    needed_slots: 4,
                    got_slots: 3
                })
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn window_union_is_first_seen_order_without_duplicates() {
        let topk: Vec<u16> = vec![5, 5, 2, 2, 5, 9];
        let u = window_expert_union(&topk, 0, 3, 2);
        assert_eq!(u, vec![5, 2, 9]);
    }

    #[test]
    fn cache_gb_env_absent_means_fully_resident() {
        assert_eq!(expert_cache_budget_bytes(None), None);
    }

    #[test]
    fn cache_gb_env_parses_to_bytes() {
        assert_eq!(
            expert_cache_budget_bytes(Some("40")),
            Some(40 * 1024 * 1024 * 1024)
        );
    }

    #[test]
    fn cache_gb_env_rejects_garbage_rather_than_defaulting() {
        assert_eq!(expert_cache_budget_bytes(Some("banana")), None);
        // Zero is "off", not a zero-byte cache that could never make progress.
        assert_eq!(expert_cache_budget_bytes(Some("0")), None);
    }

    #[test]
    fn auto_budget_subtracts_reservations_from_available() {
        // 100 GiB available, 10 GiB reserved => 90 GiB usable.
        let got = auto_budget_bytes(100 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024);
        assert_eq!(got, 90 * 1024 * 1024 * 1024);
    }

    #[test]
    fn auto_budget_saturates_at_zero_rather_than_underflowing() {
        // Reservations exceed what's available: must not wrap around.
        assert_eq!(
            auto_budget_bytes(4 * 1024 * 1024 * 1024, 9 * 1024 * 1024 * 1024),
            0
        );
    }

    #[test]
    fn effective_budget_takes_the_smaller_of_configured_and_available() {
        let avail = 90u64 * 1024 * 1024 * 1024;
        // Configured smaller than available => configured wins.
        assert_eq!(
            effective_budget_bytes(Some(40 * 1024 * 1024 * 1024), avail),
            40 * 1024 * 1024 * 1024
        );
        // Configured larger than available => clamped to available.
        assert_eq!(
            effective_budget_bytes(Some(200 * 1024 * 1024 * 1024), avail),
            avail
        );
        // Nothing configured => use all available.
        assert_eq!(effective_budget_bytes(None, avail), avail);
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
