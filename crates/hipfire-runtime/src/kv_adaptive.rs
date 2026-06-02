//! Adaptive KV: runtime VRAM-fit downshift of K/V cache precision.
//! See docs/plans/2026-05-31-adaptive-kv-design.md.
//!
//! Capacity model (corrected): K and V are SEPARATE fixed-size buffers, each
//! sized at its own floor tier (`k_buf = max_seq * nkv * k_floor_bph`,
//! `v_buf = max_seq * nkv * v_floor_bph`). The usable token capacity at a given
//! (cur_k, cur_v) is `min` over the two buffers of how many positions each
//! holds at its current stride — NOT a shared pool. The shared-pool formula
//! over-estimates capacity in lopsided states (e.g. K=fwht4 while V=q8) and
//! would let seq_pos overflow the binding buffer. n_kv_heads cancels in each
//! per-buffer ratio, so the caps are `max_seq * floor_bph / cur_bph`.
use crate::llama::VMode;

/// K-cache tier. Mirrors VMode for the V side. fwht4/fwht2 rotate 128-wide,
/// fwht3 rotates 256-wide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KMode { Fwht4, Fwht3, Fwht2 }

impl KMode {
    /// bytes-per-head at a given head_dim.
    pub fn bytes_per_head(self, head_dim: usize) -> usize {
        match self {
            KMode::Fwht4 => 4 + head_dim / 2,        // 132 @256
            KMode::Fwht3 => 4 + (head_dim * 3) / 8,  // 100 @256
            KMode::Fwht2 => 4 + head_dim / 4,        // 68  @256
        }
    }
    /// FWHT rotation width.
    pub fn rot_width(self) -> usize { match self { KMode::Fwht3 => 256, _ => 128 } }
    /// Quantization bit-width of the tier (Fwht4→4, Fwht3→3, Fwht2→2).
    pub fn bits(self) -> u32 { match self { KMode::Fwht4 => 4, KMode::Fwht3 => 3, KMode::Fwht2 => 2 } }
}

/// V bytes-per-head (mirrors KvCache::v_bytes_per_pos per-head logic).
pub fn v_bytes_per_head(v: VMode, head_dim: usize) -> usize {
    match v {
        VMode::Q8 => (head_dim / 32) * 34,                 // 272 @256
        VMode::Lloyd2 | VMode::Lloyd3 | VMode::Lloyd4 => 4 + (head_dim * v.bits() as usize) / 8,
    }
}

/// Per-layer byte size of the K buffer (sized at the K floor for all max_seq).
pub fn k_buf_bytes_per_layer(max_seq: usize, n_kv_heads: usize, head_dim: usize, k_floor: KMode) -> usize {
    max_seq * n_kv_heads * k_floor.bytes_per_head(head_dim)
}

/// Per-layer byte size of the V buffer (sized at the V floor for all max_seq).
pub fn v_buf_bytes_per_layer(max_seq: usize, n_kv_heads: usize, head_dim: usize, v_floor: VMode) -> usize {
    max_seq * n_kv_heads * v_bytes_per_head(v_floor, head_dim)
}

/// Usable token capacity at (cur_k, cur_v) = min of the two SEPARATE buffers.
/// Each buffer is sized at its floor; `cap_buffer = max_seq * floor_bph / cur_bph`
/// (n_kv_heads cancels). The binding (smaller) buffer determines when the next
/// downshift must fire.
pub fn cap_min(max_seq: usize, head_dim: usize, k_floor: KMode, v_floor: VMode,
               cur_k: KMode, cur_v: VMode) -> usize {
    let k_cap = max_seq * k_floor.bytes_per_head(head_dim) / cur_k.bytes_per_head(head_dim);
    let v_cap = max_seq * v_bytes_per_head(v_floor, head_dim) / v_bytes_per_head(cur_v, head_dim);
    k_cap.min(v_cap)
}

/// One downshift step: drop ONE cache by one tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step { V(VMode), K(KMode) }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset { Conservative, Balanced, Aggressive }

pub struct KvAdaptive {
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub k_floor: KMode,
    pub v_floor: VMode,
    pub cur_k: KMode,
    pub cur_v: VMode,
    pub steps: Vec<Step>,        // ordered remaining steps
    pub next_step: usize,        // index into steps
    pub thresholds: Vec<usize>,  // seq_pos at which steps[i] fires
    pub margin: usize,           // fire this many tokens before the cap
}

impl KvAdaptive {
    /// Build the default `balanced` step order: V q8→l4→l3, K f4→f2, V l3→l2.
    /// (Keeps the K/V bit-gap ≤ 1 tier; finalized empirically in Task 8.)
    fn balanced_steps(k_floor: KMode, v_floor: VMode) -> Vec<Step> {
        let mut s = Vec::new();
        // descend V to lloyd3 first (biggest byte win up front)
        if v_floor != VMode::Q8 { s.push(Step::V(VMode::Lloyd4)); }
        if matches!(v_floor, VMode::Lloyd3 | VMode::Lloyd2) { s.push(Step::V(VMode::Lloyd3)); }
        // K step (cheap same-width fwht4→fwht2, or fwht4→fwht3 re-rotation) once
        // V is at lloyd3. fwht4 floor ⇒ K never downshifts (no step).
        if k_floor == KMode::Fwht2 { s.push(Step::K(KMode::Fwht2)); }
        else if k_floor == KMode::Fwht3 { s.push(Step::K(KMode::Fwht3)); }
        // final V step to the floor
        if v_floor == VMode::Lloyd2 { s.push(Step::V(VMode::Lloyd2)); }
        s
    }

    pub fn from_preset(p: Preset, max_seq: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let (k_floor, v_floor) = match p {
            Preset::Conservative => (KMode::Fwht4, VMode::Lloyd4),
            Preset::Balanced     => (KMode::Fwht2, VMode::Lloyd2),
            Preset::Aggressive   => (KMode::Fwht2, VMode::Lloyd2),
        };
        Self::new(max_seq, n_kv_heads, head_dim, k_floor, v_floor)
    }

    /// Advanced: caller picks K and V floors independently.
    pub fn new(max_seq: usize, n_kv_heads: usize, head_dim: usize,
               k_floor: KMode, v_floor: VMode) -> Self {
        let steps = Self::balanced_steps(k_floor, v_floor);
        let mut s = Self {
            n_kv_heads, head_dim, max_seq, k_floor, v_floor,
            cur_k: KMode::Fwht4, cur_v: VMode::Q8, steps, next_step: 0,
            // margin MUST be >= the largest single seq_pos advance between two
            // maybe_downshift calls. The prefill loop advances by up to
            // PREFILL_MAX_BATCH per chunk and only downshifts BETWEEN chunks, so
            // the threshold has to fire a full chunk early — otherwise the chunk
            // that trips it would already have written past `cap` (the
            // floor-sized buffer's start-tier ceiling), overflowing it before the
            // post-chunk downshift can free room. Decode advances 1/token, where
            // 256 is comfortably conservative.
            thresholds: Vec::new(), margin: crate::llama::PREFILL_MAX_BATCH,
        };
        s.recompute_thresholds();
        s
    }

    /// Per-layer K / V buffer byte sizes (sized at the floors).
    pub fn k_buf_bytes_per_layer(&self) -> usize {
        k_buf_bytes_per_layer(self.max_seq, self.n_kv_heads, self.head_dim, self.k_floor)
    }
    pub fn v_buf_bytes_per_layer(&self) -> usize {
        v_buf_bytes_per_layer(self.max_seq, self.n_kv_heads, self.head_dim, self.v_floor)
    }

    /// Usable token capacity at the CURRENT (cur_k, cur_v) tier — the binding
    /// floor-sized buffer's position count. Smallest at the q8/fwht4 start tier
    /// (each position is largest there), growing as precision steps down. The
    /// load path checks the start-tier value against PREFILL_MAX_BATCH before
    /// engaging: a single prefill chunk must fit at the start tier, or the very
    /// first chunk would overflow the floor buffer before any downshift fires.
    pub fn current_cap(&self) -> usize {
        cap_min(self.max_seq, self.head_dim, self.k_floor, self.v_floor, self.cur_k, self.cur_v)
    }

    /// threshold[i] = cap(state BEFORE applying steps[i]) - margin: the seq_pos
    /// at which we must apply steps[i] before the binding buffer overflows.
    /// Thresholds are non-decreasing (coincident when two steps relax the same
    /// binding point — `maybe_downshift` applies all crossed steps in one call).
    fn recompute_thresholds(&mut self) {
        let mut k = KMode::Fwht4; let mut v = VMode::Q8;
        self.thresholds.clear();
        for st in &self.steps {
            let cap = cap_min(self.max_seq, self.head_dim, self.k_floor, self.v_floor, k, v);
            self.thresholds.push(cap.saturating_sub(self.margin));
            match *st { Step::V(nv) => v = nv, Step::K(nk) => k = nk }
        }
    }

    /// Reset the tier state to the start (Q8/fwht4) position.  Call this
    /// whenever the KV cache is cold-reset (context-full rollover, explicit
    /// "reset" command) so the threshold sequence restarts from the beginning
    /// instead of staying pinned at the floor tier permanently.
    pub fn reset(&mut self) {
        self.cur_k = KMode::Fwht4;
        self.cur_v = crate::llama::VMode::Q8;
        self.next_step = 0;
    }

    /// Apply ALL downshift steps whose threshold seq_pos has crossed (handles
    /// coincident thresholds — e.g. V→lloyd3 and K→fwht2 sharing a binding
    /// point). Called after each committed token write at the same site as
    /// `maybe_evict`. The common case (no threshold crossed) is a single integer
    /// compare returning an empty Vec. Returns the steps applied this call.
    pub fn maybe_downshift(&mut self, gpu: &mut rdna_compute::Gpu,
                           kv: &mut crate::llama::KvCache, seq_pos: usize)
        -> hip_bridge::HipResult<Vec<Step>> {
        let mut applied = Vec::new();
        while self.next_step < self.steps.len() && seq_pos >= self.thresholds[self.next_step] {
            match self.steps[self.next_step] {
                Step::V(nv) => { kv.transcode_v_step(gpu, nv, seq_pos)?; self.cur_v = nv; }
                Step::K(nk) => { kv.transcode_k_step(gpu, nk.bits(), seq_pos)?; self.cur_k = nk; }
            }
            applied.push(self.steps[self.next_step]);
            self.next_step += 1;
        }
        Ok(applied)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::VMode;
    #[test]
    fn byte_tables_match_design_256() {
        assert_eq!(KMode::Fwht4.bytes_per_head(256), 132);
        assert_eq!(KMode::Fwht3.bytes_per_head(256), 100);
        assert_eq!(KMode::Fwht2.bytes_per_head(256), 68);
        assert_eq!(v_bytes_per_head(VMode::Q8, 256), 272);
        assert_eq!(v_bytes_per_head(VMode::Lloyd4, 256), 132);
        assert_eq!(v_bytes_per_head(VMode::Lloyd3, 256), 100);
        assert_eq!(v_bytes_per_head(VMode::Lloyd2, 256), 68);
    }
    #[test]
    fn cap_at_floor_is_max_seq() {
        // At the floor (both buffers at their floor tier) capacity == max_seq:
        // the "max_seq = floor-tier context guarantee" contract.
        assert_eq!(cap_min(1000, 256, KMode::Fwht2, VMode::Lloyd2, KMode::Fwht2, VMode::Lloyd2), 1000);
        assert_eq!(cap_min(1000, 256, KMode::Fwht4, VMode::Lloyd4, KMode::Fwht4, VMode::Lloyd4), 1000);
    }
    #[test]
    fn cap_is_min_of_two_buffers_and_v_bound_at_start() {
        // Balanced floors (K=fwht2=68, V=lloyd2=68). Start state K4(132)/q8(272):
        // K buffer holds 1000*68/132 = 515 positions, V buffer holds
        // 1000*68/272 = 250 → min = 250 (V is the binding constraint at start).
        let c_start = cap_min(1000, 256, KMode::Fwht2, VMode::Lloyd2, KMode::Fwht4, VMode::Q8);
        assert_eq!(c_start, 250, "start cap is V-bound = 0.25*max_seq, not the shared-pool fiction");
        let c_floor = cap_min(1000, 256, KMode::Fwht2, VMode::Lloyd2, KMode::Fwht2, VMode::Lloyd2);
        assert!(c_floor > c_start * 2, "floor ({c_floor}) should fit >2x start ({c_start})");
    }
    #[test]
    fn balanced_pattern_shape_and_thresholds() {
        let a = KvAdaptive::from_preset(Preset::Balanced, 10_000, 4, 256);
        assert_eq!(a.steps, vec![
            Step::V(VMode::Lloyd4), Step::V(VMode::Lloyd3),
            Step::K(KMode::Fwht2), Step::V(VMode::Lloyd2),
        ]);
        // thresholds non-decreasing (coincident allowed when steps share a
        // binding point — here V→l3 and K→f2 both at ~0.515*max_seq).
        for w in a.thresholds.windows(2) { assert!(w[1] >= w[0], "thresholds {:?}", a.thresholds); }
        // first threshold = start-tier cap (V-bound, 0.25*max_seq=2500) - margin.
        // Asserted relative to a.margin so it tracks the chunk-safety value.
        let start_cap = cap_min(10_000, 256, KMode::Fwht2, VMode::Lloyd2, KMode::Fwht4, VMode::Q8);
        assert_eq!(start_cap, 2500);
        assert_eq!(a.current_cap(), start_cap, "current_cap at construction == start-tier cap");
        assert_eq!(a.thresholds[0], start_cap - a.margin, "first threshold = start_cap - margin");
    }
    #[test]
    fn margin_at_least_one_prefill_chunk() {
        // SAFETY INVARIANT: the prefill loop advances seq_pos by up to
        // PREFILL_MAX_BATCH per chunk and only downshifts between chunks, so the
        // threshold must fire >= one chunk early or the tripping chunk writes
        // past the floor buffer. See daemon.rs prefill loop + the load-time
        // start_cap >= PREFILL_MAX_BATCH guard.
        let a = KvAdaptive::from_preset(Preset::Balanced, 100_000, 4, 256);
        assert!(a.margin >= crate::llama::PREFILL_MAX_BATCH,
            "margin {} must be >= PREFILL_MAX_BATCH {}", a.margin, crate::llama::PREFILL_MAX_BATCH);
    }
    #[test]
    fn conservative_only_v_to_lloyd4() {
        let a = KvAdaptive::from_preset(Preset::Conservative, 10_000, 4, 256);
        assert_eq!(a.steps, vec![Step::V(VMode::Lloyd4)]);
    }
    #[test]
    fn advanced_k_fwht3_floor() {
        let a = KvAdaptive::new(10_000, 4, 256, KMode::Fwht3, VMode::Lloyd2);
        assert!(a.steps.contains(&Step::K(KMode::Fwht3)));
    }
    #[test]
    fn advanced_v_only_floor_has_no_k_step() {
        // k_floor = fwht4 ⇒ K never downshifts; pure V chain (used by Task 5
        // end-to-end V-only proof: advanced k=fwht4,v=lloyd2).
        let a = KvAdaptive::new(10_000, 4, 256, KMode::Fwht4, VMode::Lloyd2);
        assert_eq!(a.steps, vec![Step::V(VMode::Lloyd4), Step::V(VMode::Lloyd3), Step::V(VMode::Lloyd2)]);
    }
}
