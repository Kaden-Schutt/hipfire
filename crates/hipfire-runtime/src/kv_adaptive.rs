//! Adaptive KV: runtime VRAM-fit downshift of K/V cache precision.
//! See docs/plans/2026-05-31-adaptive-kv-design.md.
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
}

/// V bytes-per-head (mirrors KvCache::v_bytes_per_pos per-head logic).
pub fn v_bytes_per_head(v: VMode, head_dim: usize) -> usize {
    match v {
        VMode::Q8 => (head_dim / 32) * 34,                 // 272 @256
        VMode::Lloyd2 | VMode::Lloyd3 | VMode::Lloyd4 => 4 + (head_dim * v.bits() as usize) / 8,
    }
}

/// Token capacity of the floor-sized buffer at a given (K,V) tier.
pub fn cap_tokens(budget_bytes_per_layer: usize, n_kv_heads: usize, head_dim: usize,
                  k: KMode, v: VMode) -> usize {
    let per_tok = n_kv_heads * (k.bytes_per_head(head_dim) + v_bytes_per_head(v, head_dim));
    if per_tok == 0 { 0 } else { budget_bytes_per_layer / per_tok }
}

/// Floor-sized per-layer byte budget = capacity for `max_seq` tokens at the floor.
pub fn budget_bytes_per_layer(max_seq: usize, n_kv_heads: usize, head_dim: usize,
                              k_floor: KMode, v_floor: VMode) -> usize {
    max_seq * n_kv_heads * (k_floor.bytes_per_head(head_dim) + v_bytes_per_head(v_floor, head_dim))
}

/// One downshift step: drop ONE cache by one tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step { V(VMode), K(KMode) }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset { Conservative, Balanced, Aggressive }

pub struct KvAdaptive {
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub budget_bytes_per_layer: usize,
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
        // K step (cheap same-width fwht4→fwht2) once V is at lloyd3
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
        let budget = budget_bytes_per_layer(max_seq, n_kv_heads, head_dim, k_floor, v_floor);
        let steps = Self::balanced_steps(k_floor, v_floor);
        let mut s = Self {
            n_kv_heads, head_dim, budget_bytes_per_layer: budget,
            cur_k: KMode::Fwht4, cur_v: VMode::Q8, steps, next_step: 0,
            thresholds: Vec::new(), margin: 64,
        };
        s.recompute_thresholds();
        s
    }

    /// threshold[i] = cap(state AFTER applying steps[0..=i-1]) - margin,
    /// i.e. the seq_pos at which we must apply steps[i] before overflowing the
    /// cap of the CURRENT (pre-step-i) tier.
    fn recompute_thresholds(&mut self) {
        let mut k = KMode::Fwht4; let mut v = VMode::Q8;
        self.thresholds.clear();
        for st in &self.steps {
            let cap = cap_tokens(self.budget_bytes_per_layer, self.n_kv_heads, self.head_dim, k, v);
            self.thresholds.push(cap.saturating_sub(self.margin));
            match *st { Step::V(nv) => v = nv, Step::K(nk) => k = nk }
        }
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
    fn floor_budget_gives_max_seq_at_floor() {
        let max_seq = 1000;
        let b = budget_bytes_per_layer(max_seq, 4, 256, KMode::Fwht2, VMode::Lloyd2);
        assert_eq!(cap_tokens(b, 4, 256, KMode::Fwht2, VMode::Lloyd2), max_seq);
    }
    #[test]
    fn cap_grows_as_precision_drops() {
        let b = budget_bytes_per_layer(1000, 4, 256, KMode::Fwht2, VMode::Lloyd2);
        let c_start = cap_tokens(b, 4, 256, KMode::Fwht4, VMode::Q8);
        let c_floor = cap_tokens(b, 4, 256, KMode::Fwht2, VMode::Lloyd2);
        assert!(c_floor > c_start * 2, "floor should fit >2x start ({c_floor} vs {c_start})");
        // design table: start K4/q8 ≈ 0.337*max_seq
        assert!((330..=345).contains(&c_start), "start cap {c_start}");
    }
    #[test]
    fn balanced_pattern_shape() {
        let a = KvAdaptive::from_preset(Preset::Balanced, 10_000, 4, 256);
        assert_eq!(a.steps, vec![
            Step::V(VMode::Lloyd4), Step::V(VMode::Lloyd3),
            Step::K(KMode::Fwht2), Step::V(VMode::Lloyd2),
        ]);
        // thresholds strictly increasing (each tier fits more before the next shift)
        for w in a.thresholds.windows(2) { assert!(w[1] > w[0], "thresholds {:?}", a.thresholds); }
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
}
