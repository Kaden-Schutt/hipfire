//! Per-architecture capability defaults.
//!
//! Single source of truth for "is feature X default-on for arch Y" questions.
//! Adding a new arch = one-line edit per feature.

/// MQ4G128 in-engine encoding of LinearAttention in_proj_a / in_proj_b weights
/// at model load time. Default-on for gfx1151 (validated on shisa-Qwen3.6-35B-A3B-PARO).
/// Override via env var `HIPFIRE_PARO_LA_GATES_MQ4G128={0|1}`.
// NOTE: `const fn` with `&str` pattern matching is not yet stable (PartialEq not const).
// Using plain `pub fn` instead — semantically identical at runtime.
pub fn paro_la_gates_mq4g128_default(arch: &str) -> bool {
    matches!(arch, "gfx1151")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_table() {
        assert!(paro_la_gates_mq4g128_default("gfx1151"));
        assert!(!paro_la_gates_mq4g128_default("gfx1100"));
        assert!(!paro_la_gates_mq4g128_default("gfx1010"));
        assert!(!paro_la_gates_mq4g128_default("gfx906"));
        assert!(!paro_la_gates_mq4g128_default(""));
    }
}
