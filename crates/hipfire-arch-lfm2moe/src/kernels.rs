// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! LFM2.5-8B-A1B-only GPU kernels.
//!
//! These launchers are deliberately owned by the LFM architecture crate rather
//! than `rdna-compute`: their model shapes are not valid shared-MoE contracts.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

pub const LFM2_A1B_HIDDEN: usize = 2048;
pub const LFM2_A1B_MOE_INTERMEDIATE: usize = 1792;
pub const LFM2_A1B_TOP_K: usize = 4;
pub const LFM2_A1B_DOWN_ARCHES: &[&str] = &["gfx1201", "gfx1151", "gfx1100", "gfx1030", "gfx1010"];

const MODULE: &str = "lfm2_a1b_moe_down_hfq4g256_k1792_wave32";
const SYMBOL: &str = "lfm2_a1b_moe_down_hfq4g256_k1792_wave32";
const SOURCE: &str = include_str!("../kernels/lfm2_a1b_moe_down_hfq4g256_k1792_wave32.hip");

const CONV_SCAN_MODULE: &str = "conv1d_gated_scan_n_gfx1201";
const CONV_SCAN_SYMBOL: &str = "conv1d_gated_scan_n_f32";
const CONV_SCAN_SOURCE: &str =
    include_str!("../../../kernels/src/conv1d_gated_scan_n.gfx1201.hip");

fn validate_contract(
    arch: &str,
    dtype: DType,
    m: usize,
    k: usize,
    k_top: usize,
) -> Result<(), String> {
    if !LFM2_A1B_DOWN_ARCHES.contains(&arch) {
        return Err(format!(
            "lfm2 A1B MoE down requires one of {}, got {arch}",
            LFM2_A1B_DOWN_ARCHES.join(", ")
        ));
    }
    if (m, k, k_top) != (LFM2_A1B_HIDDEN, LFM2_A1B_MOE_INTERMEDIATE, LFM2_A1B_TOP_K) {
        return Err(format!(
            "lfm2 A1B MoE down requires M={} K={} top_k={}, got M={m} K={k} top_k={k_top}",
            LFM2_A1B_HIDDEN, LFM2_A1B_MOE_INTERMEDIATE, LFM2_A1B_TOP_K
        ));
    }
    if !matches!(dtype, DType::HFQ4G256 | DType::MQ4G256) {
        return Err(format!(
            "lfm2 A1B MoE down requires HFQ4G256 or MQ4G256 weights, got {dtype:?}"
        ));
    }
    Ok(())
}

/// Decode-only LFM2.5-8B-A1B routed-expert down projection on supported
/// wave32 RDNA architectures.
///
/// The exact model/hardware contract is checked before launch so this path
/// cannot be selected by Qwen, DeepSeek, MiniMax, another LFM shape, or an
/// unsupported GPU architecture.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_a1b_moe_down(
    gpu: &mut Gpu,
    expert_ptrs: &GpuTensor,
    topk_indices: &GpuTensor,
    rot_batch: &GpuTensor,
    expert_outputs: &GpuTensor,
    down_dtype: DType,
    m: usize,
    k: usize,
    k_top: usize,
) -> Result<(), String> {
    validate_contract(&gpu.arch, down_dtype, m, k, k_top)?;

    gpu.ensure_kernel_public(MODULE, SOURCE, SYMBOL)
        .map_err(|e| format!("compile LFM2 A1B MoE down: {e:?}"))?;

    let expert_ptr = expert_ptrs.buf.as_ptr();
    let topk_ptr = topk_indices.buf.as_ptr();
    let rot_ptr = rot_batch.buf.as_ptr();
    let output_ptr = expert_outputs.buf.as_ptr();
    let mut params = vec![
        &expert_ptr as *const _ as *mut c_void,
        &topk_ptr as *const _ as *mut c_void,
        &rot_ptr as *const _ as *mut c_void,
        &output_ptr as *const _ as *mut c_void,
    ];
    let mut blob = hip_bridge::KernargBlob::new();
    blob.push_ptr(expert_ptr);
    blob.push_ptr(topk_ptr);
    blob.push_ptr(rot_ptr);
    blob.push_ptr(output_ptr);
    gpu.launch_external_kernel(
        SYMBOL,
        [LFM2_A1B_HIDDEN as u32, LFM2_A1B_TOP_K as u32, 1],
        [32, 1, 1],
        0,
        &mut params,
        blob,
    )
    .map_err(|e| format!("launch LFM2 A1B MoE down: {e:?}"))
}

pub fn conv1d_gated_scan_n(
    gpu: &mut Gpu,
    bcx: &GpuTensor,
    state: &GpuTensor,
    weight: &GpuTensor,
    out_y: &GpuTensor,
    n_tokens: usize,
    channels: usize,
) -> hip_bridge::HipResult<()> {
    if !gpu.arch_caps.is_gfx1201() {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "conv1d_gated_scan_n_f32 requires gfx1201, got {}",
                gpu.arch
            ),
        ));
    }
    let n_tokens_i32 = i32::try_from(n_tokens).map_err(|_| {
        hip_bridge::HipError::new(0, "conv1d_gated_scan_n_f32 n_tokens exceeds i32")
    })?;
    let channels_i32 = i32::try_from(channels).map_err(|_| {
        hip_bridge::HipError::new(0, "conv1d_gated_scan_n_f32 channels exceeds i32")
    })?;
    if n_tokens_i32 <= 0 || channels_i32 <= 0 {
        return Err(hip_bridge::HipError::new(
            0,
            "conv1d_gated_scan_n_f32 requires non-zero n_tokens and channels",
        ));
    }

    gpu.ensure_kernel_public(CONV_SCAN_MODULE, CONV_SCAN_SOURCE, CONV_SCAN_SYMBOL)?;

    let bcx_ptr = bcx.buf.as_ptr();
    let state_ptr = state.buf.as_ptr();
    let weight_ptr = weight.buf.as_ptr();
    let out_ptr = out_y.buf.as_ptr();
    let mut params = vec![
        &bcx_ptr as *const _ as *mut c_void,
        &state_ptr as *const _ as *mut c_void,
        &weight_ptr as *const _ as *mut c_void,
        &out_ptr as *const _ as *mut c_void,
        &n_tokens_i32 as *const _ as *mut c_void,
        &channels_i32 as *const _ as *mut c_void,
    ];
    let mut blob = hip_bridge::KernargBlob::new();
    blob.push_ptr(bcx_ptr);
    blob.push_ptr(state_ptr);
    blob.push_ptr(weight_ptr);
    blob.push_ptr(out_ptr);
    blob.push_i32(n_tokens_i32);
    blob.push_i32(channels_i32);
    gpu.launch_external_kernel(
        CONV_SCAN_SYMBOL,
        [(channels as u32).div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
        &mut params,
        blob,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_accepts_a1b_on_supported_arches() {
        for dtype in [DType::HFQ4G256, DType::MQ4G256] {
            for &arch in LFM2_A1B_DOWN_ARCHES {
                assert!(
                    validate_contract(arch, dtype, 2048, 1792, 4).is_ok(),
                    "{arch} must support {dtype:?} LFM A1B down weights"
                );
            }
        }
        assert!(validate_contract("gfx1200", DType::MQ4G256, 2048, 1792, 4).is_err());
        assert!(validate_contract("gfx906", DType::MQ4G256, 2048, 1792, 4).is_err());
        assert!(validate_contract("gfx1201", DType::MQ4G256, 256, 1792, 4).is_err());
        assert!(validate_contract("gfx1201", DType::MQ4G256, 2048, 512, 4).is_err());
        assert!(validate_contract("gfx1201", DType::MQ4G256, 2048, 1792, 8).is_err());
        assert!(validate_contract("gfx1201", DType::MQ3G256, 2048, 1792, 4).is_err());
        assert!(validate_contract("gfx1201", DType::HFQ6G256, 2048, 1792, 4).is_err());
    }
}
