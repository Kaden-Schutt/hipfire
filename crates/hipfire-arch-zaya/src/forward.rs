//! ZAYA1 forward-pass entry points (stubs).
//!
//! Phase 1 status: declarations only. Bodies return Err. Real forward
//! lands incrementally:
//!   - Phase 2: free components (config, RMSNorm, SwiGLU, GQA reuse,
//!     partial-RoPE, scale_residual_merge, MLP router, top-1).
//!   - Phase 6 decision: where the CCA recurrent-state primitive lives.
//!   - Phase 3 (later): CCA scalar reference + HIP kernel.
//!   - Phase 4 (later): MoD per-token routing (gen-loop change).
//!
//! Forward signatures intentionally NOT on the Architecture trait
//! (per `hipfire-runtime::arch` doc-block: forward-pass dispatch is
//! static and arch-specific). The runtime's daemon calls these via the
//! concrete `Zaya` type.

use crate::state::ZayaState;
use crate::weights::ZayaWeights;
use rdna_compute::Gpu;

/// Prefill: process a prompt of length S, populate the KV cache and the
/// CCA recurrent buffers (`conv_states`, `prev_hs`) such that the next
/// `decode_step` can pick up coherently.
///
/// Returns the next-token logits over the vocab.
pub fn prefill(
    _gpu: &mut Gpu,
    _weights: &ZayaWeights,
    _state: &mut ZayaState,
    _input_ids: &[u32],
) -> Result<Vec<f32>, String> {
    Err(
        "ZAYA1 prefill not implemented (Phase 1 scaffold). Free \
         components land in Phase 2; CCA forward in Phase 3 after \
         Phase 6 design doc. See docs/investigations/2026-05-07-zaya1-port-intake/."
            .to_string(),
    )
}

/// Single decode step: takes one new token id, advances the recurrent
/// state and KV cache by one position, returns the next-token logits.
///
/// MoD per-token layer-skip routing (Phase 4) lives inside this function;
/// the call site does not see whether a token skipped layers.
pub fn decode_step(
    _gpu: &mut Gpu,
    _weights: &ZayaWeights,
    _state: &mut ZayaState,
    _input_id: u32,
) -> Result<Vec<f32>, String> {
    Err(
        "ZAYA1 decode_step not implemented (Phase 1 scaffold). See \
         docs/investigations/2026-05-07-zaya1-port-intake/."
            .to_string(),
    )
}
