//! GPU-side iu4 activation calibration data.
//!
//! Engine-side sidecar lives at `engine::quant::iu4_calibration::Iu4Calibration`.
//! When the engine wants to enable calibrated iu4 dispatch on gfx12, it
//! uploads each site's mu/inv_s/bias FP16 vectors into device buffers and
//! hands the resulting `GpuIu4Calibration` to `Gpu::load_iu4_calibration`.
//!
//! The dispatcher then consults `Gpu::iu4_calibration` per call to look up
//! the active site's tables and runs:
//!
//!   1. preshift kernel: `x_centered[t][c] = (x[t][c] - mu[c]) * inv_s[c]`
//!   2. existing `ensure_q4_1_x` on `x_centered`
//!   3. existing `gemm_hfq4g256_residual_iu4_gfx12` (UNCHANGED — just with a
//!      weight whose per-row scales were pre-multiplied by group-mean(s_a)
//!      at calibration-load time on the host)
//!   4. broadcast-add of `w_mu_bias[m]` to the GEMM output (existing
//!      `bias_add_f32`)

use hip_bridge::DeviceBuffer;

/// Per-site GPU-resident calibration tables. Keep them owned by the Gpu
/// struct (lives as long as the model is loaded).
pub struct GpuIu4CalSite {
    /// Logical layer index (informational; the dispatcher keys by call order).
    pub layer_idx: u32,
    /// 0=wo, 1=w_down.
    pub proj_id: u32,
    /// Input channel dim (= k).
    pub n_channels: u32,
    /// Output dim (= weight rows).
    pub n_output_rows: u32,
    /// HFQ4-G256 group count along K (= k / 256).
    pub groups_per_row: u32,
    /// Per-input-channel mu_a, FP16, n_channels half-words.
    pub mu_a: DeviceBuffer,
    /// Per-input-channel 1 / s_a, FP16, n_channels half-words. The s_a
    /// stored here is GROUP-CONSTANT (= `s_group[group(c)]` repeated
    /// across the 256 channels of each group) so the activation-side
    /// preshift composes EXACTLY with the weight-side bake. See
    /// `engine::quant::iu4_calibration` for the math.
    pub inv_s_a: DeviceBuffer,
    /// Per-K=256-group activation scale, FP16, groups_per_row half-words.
    /// Used once at first-touch per weight to multiply the HFQ4 group
    /// scale fields in-place (or in a clone — see `iu4_baked_weight_cache`).
    pub s_group: DeviceBuffer,
    /// Per-output-row W·mu_a precomputed bias, FP32 to match the GEMM
    /// output dtype directly (no FP16 → FP32 cast in the bias_add path).
    pub w_mu_bias_f32: DeviceBuffer,
}

pub struct GpuIu4Calibration {
    pub sites: Vec<GpuIu4CalSite>,
}

impl GpuIu4Calibration {
    pub fn empty() -> Self {
        Self { sites: Vec::new() }
    }

    pub fn site_at(&self, call_idx: usize) -> Option<&GpuIu4CalSite> {
        self.sites.get(call_idx)
    }

    pub fn n_sites(&self) -> usize {
        self.sites.len()
    }
}
