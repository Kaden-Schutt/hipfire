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

/// Grammar constraint for structured decoding (tool-call / DSML).
///
/// The trait is expressed in terms of primitive/std types so the runtime
/// library can name it without taking a dependency on any arch crate (which
/// would introduce a cycle: arch crates already depend on hipfire-runtime).
/// Arch-specific `Matcher` types implement this via a newtype in daemon.rs
/// (the example crate, which has both sides as dev-deps and is therefore
/// free of the cycle).
///
/// Method signatures mirror the real `hipfire_arch_qwen35::grammar::Matcher`
/// exactly:
///   - `token_mask(&self, vocab: &[String], out: &mut [bool])` — populate a
///     per-token boolean allow-mask over the decoded vocab slice.
///   - `advance(&mut self, text: &str)` — commit a decoded token's text,
///     advancing grammar state.
///   - `is_free(&self) -> bool` — true when no constraint is active at the
///     current position (the caller may skip the mask scan).
pub trait GrammarMatcher {
    fn token_mask(&self, vocab: &[String], out: &mut [bool]);
    fn advance(&mut self, text: &str);
    fn is_free(&self) -> bool;
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

    // ── AR-phase forward hooks (Inc 1, Task 1.4a) ─────────────────────────

    /// Run the prefill forward pass for `chunk` tokens starting at `seq_pos`.
    /// Equivalent to the arch-specific `forward_prefill_batch` call in each
    /// generate arm. Default returns Err so un-ported arches fail loudly if
    /// `ar_generate` is accidentally routed through them.
    #[allow(dead_code)]
    fn prefill_forward(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        chunk: &[u32],
        seq_pos: usize,
    ) -> Result<(), String> {
        let _ = (gpu, chunk, seq_pos);
        Err("prefill_forward not implemented for this arch".into())
    }

    /// Run a single decode-step forward pass for `token` at `seq_pos`.
    /// Equivalent to the arch-specific `forward_scratch` call in each generate
    /// arm. Default returns Err so un-ported arches fail loudly.
    #[allow(dead_code)]
    fn decode_step_forward(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        token: u32,
        seq_pos: usize,
    ) -> Result<(), String> {
        let _ = (gpu, token, seq_pos);
        Err("decode_step_forward not implemented for this arch".into())
    }

    /// Build a grammar matcher for the given tool schemas, or return None for
    /// arches without grammar support.
    ///
    /// `tool_schemas` is a slice of `(name, required_fields)` pairs — the
    /// primitive representation used to cross the runtime→arch crate boundary
    /// without naming arch-specific types here. The qwen35 impl converts each
    /// pair to a `hipfire_arch_qwen35::grammar::ToolSchema` before constructing
    /// the `Matcher` newtype.
    #[allow(dead_code)]
    fn init_grammar(
        &self,
        tool_schemas: &[(String, Vec<String>)],
    ) -> Option<Box<dyn GrammarMatcher>> {
        let _ = tool_schemas;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_features_default_is_all_false() {
        let f = ArchFeatures::default();
        assert!(
            !f.supports_think && !f.supports_stop_seq && !f.supports_grammar && !f.supports_vision
        );
    }

    #[test]
    fn sampling_defaults_are_plain_data() {
        let d = SamplingDefaults {
            temp: 0.3,
            top_p: 0.8,
            repeat_penalty: 1.0,
        };
        assert_eq!(
            d,
            SamplingDefaults {
                temp: 0.3,
                top_p: 0.8,
                repeat_penalty: 1.0
            }
        );
    }
}
