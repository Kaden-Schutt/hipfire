//! ZayaState: per-decode GPU scratch for ZAYA1.
//!
//! Holds the standard KV cache PLUS the two CCA recurrent state buffers
//! (`conv_states`, `prev_hs`) per layer per sequence. The recurrent
//! buffers are the load-bearing structural addition vs every prior
//! hipfire arch (qwen35 hybrid LA aside).
//!
//! Phase 1 status: TYPED FIELDS ONLY. `new` returns Err — actual GPU
//! allocation lands after Phase 6 design doc resolves whether recurrent
//! state lives in `hipfire-runtime` (Option B) or stays per-arch in this
//! crate's State (Option A). See
//! docs/investigations/2026-05-07-zaya1-port-intake/00-cca-disambiguation.md
//! for the trade-off.

use crate::config::ZayaConfig;
use rdna_compute::Gpu;

/// Per-decode GPU scratch for a single ZAYA1 sequence.
///
/// Field shapes (for ZAYA1-8B, num_layers=80, hidden_size=2048):
///   - `kv_cache_bytes`: standard KV cache, sized by max context. Owned
///     by the runtime's existing KV pager (placeholder field today).
///   - `cca_conv_states`: `[num_layers, B, in_out_ch=1280, conv_kernel_size=2]`
///     fp16. Per-step roll-and-write update from
///     `ZayaDynamicCache.update_conv_state`. ~205 KB per sequence.
///   - `cca_prev_hs`: `[num_layers, B, hidden_size=2048]` fp16. Per-step
///     overwrite from `prev_hs[layer].copy_(hs[-1, :, :])`. ~328 KB per
///     sequence.
///
/// Total CCA recurrent state per sequence: ~533 KB at fp16. Trivial vs
/// KV cache; large vs nothing. Fits comfortably in a single HBM block.
///
/// RDNA ISA notes (gfx1201 R9700 target):
///   - in_out_ch=1280 is divisible by 64 → wave32 × 2-VGPR-per-lane
///     packed-fp16 covers it in 20 lane-groups.
///   - conv_kernel_size=2 means each output position is a 2-element MAC
///     per channel; ideal for v_pk_fma_f16 packed math.
///   - Roll(-1) + write[-1] becomes a single uint32 swap if we lay the
///     two time slots out as consecutive fp16 pairs in HBM.
pub struct ZayaState {
    pub config: ZayaConfig,

    // -- KV cache ----------------------------------------------------------
    /// Placeholder. Real impl threads through `hipfire-runtime`'s KV pager.
    pub kv_cache_bytes: usize,

    // -- CCA recurrent state -----------------------------------------------
    /// Total bytes for `conv_states` across all layers, computed from cfg.
    pub cca_conv_states_bytes: usize,
    /// Total bytes for `prev_hs` across all layers, computed from cfg.
    pub cca_prev_hs_bytes: usize,
    /// `has_previous_state` flag from `ZayaDynamicCache`. False until the
    /// first decode step writes the cache. Toggled by the forward pass.
    pub has_previous_state: bool,
}

impl ZayaState {
    /// Allocate per-sequence GPU scratch for ZAYA1.
    ///
    /// Phase 1 status: returns Err. The recurrent-state allocation
    /// strategy is the headline Phase 6 deliverable; do not silently
    /// allocate anything until that decision lands.
    pub fn new(_gpu: &mut Gpu, _cfg: &ZayaConfig) -> Result<Self, String> {
        Err(
            "ZayaState::new not implemented (Phase 1 scaffold). The CCA \
             recurrent state's GPU layout is the Phase 6 deliverable; \
             see docs/investigations/2026-05-07-zaya1-port-intake/. Two \
             options under consideration: extend per-arch State (Option A) \
             vs first-class recurrent-cache primitive in hipfire-runtime \
             (Option B). Decision REQUIRES-KADEN-DECISION."
                .to_string(),
        )
    }

    /// Compute the per-sequence CCA recurrent state size in bytes.
    /// Used by the Phase 6 design doc to size the runtime allocator
    /// budget regardless of which option lands.
    pub fn cca_state_bytes_per_seq(cfg: &ZayaConfig) -> usize {
        let dtype_bytes = 2; // fp16, per ZayaDynamicCache default
        let in_out_ch = cfg.cca_num_q_heads * cfg.head_dim
            + cfg.num_query_groups * cfg.head_dim;
        let conv_kernel_size = cfg.cca_time0.max(cfg.cca_time1); // both =2 in 8B
        let conv = cfg.num_hidden_layers * in_out_ch * conv_kernel_size * dtype_bytes;
        let prev_hs = cfg.num_hidden_layers * cfg.hidden_size * dtype_bytes;
        conv + prev_hs
    }

    /// Reset both recurrent buffers to zero (matches
    /// `ZayaDynamicCache.reset()` semantics). Called by the daemon on
    /// session end / context reset.
    pub fn reset_cca(&mut self) {
        self.has_previous_state = false;
        // Real impl zeroes the GPU buffers via `gpu.memset(...)`.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cca_state_bytes_8b_under_1mb() {
        let cfg = ZayaConfig::default();
        let bytes = ZayaState::cca_state_bytes_per_seq(&cfg);
        // Expected for 8B (80 layers, in_out_ch=1280, kernel=2, hidden=2048):
        //   conv:    80 * 1280 * 2 * 2 = 409_600
        //   prev_hs: 80 * 2048 * 2     = 327_680
        //   total = 737_280 = ~720 KB
        assert!(bytes > 700_000 && bytes < 800_000, "got {bytes} bytes");
    }
}
