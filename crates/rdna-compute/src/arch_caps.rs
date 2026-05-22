//! Per-architecture capability defaults.
//!
//! Single source of truth for "is feature X default-on for arch Y" questions.
//! Adding a new arch = one-line edit per feature.

/// MQ4G128 in-engine encoding of LinearAttention in_proj_a / in_proj_b weights
/// at model load time.
///
/// **Default-off pending fused rotation-GEMV kernel.** Bench on 2026-05-22
/// (`.scratch/rocprof-2026-05-22-mq4g128-on/`) measured −0.5% decode tok/s vs
/// baseline on shisa-Qwen3.6-35B-A3B-PARO / gfx1151. Root cause: alpha/beta
/// have M=16 shape; the existing dispatch chain (`mq_rotate_x_128` +
/// `gemv_mq4g128_prerotated` → `gemv_hfq4g128`) doubles launch count vs the
/// single F32 GEMV, and at small M the GPU kernel time (~4 µs) is dwarfed
/// by launch overhead. K-split variant was tried and also regressed.
///
/// The lever can only be made net-positive by FUSING the FWHT-128 rotation
/// into the GEMV in a single kernel launch. See
/// `docs/superpowers/specs/2026-05-22-lever1-fused-mq4g128-design.md` for
/// the design.
///
/// The infrastructure (DType variant, kernel, dispatch wrappers, codec) is
/// kept in place since it's correct end-to-end (round-trip + smoke + KLD
/// equivalent argmax verified). Opt-in via `HIPFIRE_PARO_LA_GATES_MQ4G128=1`.
// NOTE: `const fn` with `&str` pattern matching is not yet stable (PartialEq not const).
// Using plain `pub fn` instead — semantically identical at runtime.
pub fn paro_la_gates_mq4g128_default(_arch: &str) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_table() {
        // All archs default-off pending fused rotation-GEMV kernel.
        assert!(!paro_la_gates_mq4g128_default("gfx1151"));
        assert!(!paro_la_gates_mq4g128_default("gfx1100"));
        assert!(!paro_la_gates_mq4g128_default("gfx1010"));
        assert!(!paro_la_gates_mq4g128_default(""));
    }
}
