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
}
