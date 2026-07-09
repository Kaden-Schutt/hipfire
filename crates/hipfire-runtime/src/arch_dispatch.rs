// crates/hipfire-runtime/src/arch_dispatch.rs
//! The AR-decode arch abstraction — the decode-loop analog of the load-time
//! `Carrier` and the TP/PP `DenseServed`. Introduced by the god-struct-collapse
//! refactor (docs/superpowers/specs/2026-07-09-daemon-god-struct-archdispatch-design.md).
//! Inc 0 defines only the arch-invariant + reset surface; AR-phase hooks land in Inc 1.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingDefaults {
    pub temp: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ArchFeatures {
    pub supports_think: bool,
    pub supports_stop_seq: bool,
    pub supports_grammar: bool,
    pub supports_vision: bool,
}

/// One trait object per loaded model. Owns arch-specific decode behavior so the
/// daemon's dispatch stops branching on `arch_id`.
pub trait ArchDispatch {
    fn arch_id(&self) -> u32;
    fn eos_token(&self) -> u32;
    fn sampling_defaults(&self) -> SamplingDefaults;
    fn features(&self) -> ArchFeatures;
    /// TOTAL recurrent/KV reset for a fresh context. The #462 lever: a new arch
    /// cannot ship without an impl, so no reset site can be forgotten.
    fn reset(&mut self, gpu: &mut rdna_compute::Gpu);
    /// Bridge to the existing spec-decode verify seam; None for AR-only arches.
    fn as_spec_target(&mut self) -> Option<&mut dyn crate::spec::SpecTarget> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_features_default_is_all_false() {
        let f = ArchFeatures::default();
        assert!(!f.supports_think && !f.supports_stop_seq && !f.supports_grammar && !f.supports_vision);
    }

    #[test]
    fn sampling_defaults_are_plain_data() {
        let d = SamplingDefaults { temp: 0.3, top_p: 0.8, repeat_penalty: 1.0 };
        assert_eq!(d, SamplingDefaults { temp: 0.3, top_p: 0.8, repeat_penalty: 1.0 });
    }
}
