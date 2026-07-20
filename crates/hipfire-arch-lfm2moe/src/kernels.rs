// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! LFM-only GPU kernels (A1B MoE down + 350M MQ4 gfx1201 WMMA BT).
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
const CONV_SCAN_SOURCE: &str = include_str!("../../../kernels/src/conv1d_gated_scan_n.gfx1201.hip");

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
            &format!("conv1d_gated_scan_n_f32 requires gfx1201, got {}", gpu.arch),
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

// ---------------------------------------------------------------------------
// LFM2.5-350M-MQ4 / gfx1201 ONLY — HFQ4G256 WMMA batch-tiled gate_up + residual.
// Not shared substrate: Qwen/DeepSeek/MiniMax must keep the 1-acc gfx12 path.
// ---------------------------------------------------------------------------

const LFM2_350M_HIDDEN: usize = 1024;
const LFM2_350M_INTERMEDIATE: usize = 4608;
/// Matches HIPFIRE_LFM2_PREFILL_MAX_BATCH hard cap in lfm2moe.rs.
const LFM2_350M_MAX_BATCH: usize = 512;

const LFM_GATE_UP_1ACC_MODULE: &str = "lfm2_350m_mq4_gfx1201_gate_up_hfq4g256_wmma";
const LFM_GATE_UP_1ACC_SYMBOL: &str = "lfm2_350m_mq4_gfx1201_gate_up_hfq4g256_wmma";
const LFM_GATE_UP_1ACC_SRC: &str =
    include_str!("../../../kernels/src/lfm2_350m_mq4_gfx1201_gate_up_hfq4g256_wmma.hip");

const LFM_GATE_UP_BT_MODULE: &str = "lfm2_350m_mq4_gfx1201_gate_up_hfq4g256_wmma_bt";
const LFM_GATE_UP_BT_SRC: &str =
    include_str!("../../../kernels/src/lfm2_350m_mq4_gfx1201_gate_up_hfq4g256_wmma_bt.hip");

const LFM_RESID_1ACC_MODULE: &str = "lfm2_350m_mq4_gfx1201_residual_hfq4g256_wmma";
const LFM_RESID_1ACC_SYMBOL: &str = "lfm2_350m_mq4_gfx1201_residual_hfq4g256_wmma";
const LFM_RESID_1ACC_SRC: &str =
    include_str!("../../../kernels/src/lfm2_350m_mq4_gfx1201_residual_hfq4g256_wmma.hip");

const LFM_RESID_BT_MODULE: &str = "lfm2_350m_mq4_gfx1201_residual_hfq4g256_wmma_bt";
const LFM_RESID_BT_SRC: &str =
    include_str!("../../../kernels/src/lfm2_350m_mq4_gfx1201_residual_hfq4g256_wmma_bt.hip");

/// Adaptive B policy for LFM 350M MQ4 gfx1201 prefill BT selection.
/// batch<64=>1; %192=>12; %128=>8; %64=>4; >=192=>12; >=128=>8; else 4.
fn lfm2_350m_adaptive_bt_b(batch_size: usize) -> usize {
    if batch_size < 64 {
        1
    } else if batch_size % 192 == 0 {
        12
    } else if batch_size % 128 == 0 {
        8
    } else if batch_size % 64 == 0 {
        4
    } else if batch_size >= 192 {
        12
    } else if batch_size >= 128 {
        8
    } else {
        4
    }
}

fn validate_lfm2_350m_gfx1201(gpu: &Gpu, batch_size: usize) -> Result<(), String> {
    if !gpu.arch_caps.is_gfx1201() {
        return Err(format!(
            "lfm2_350m mq4 wmma requires gfx1201, got {}",
            gpu.arch
        ));
    }
    if batch_size == 0 || batch_size > LFM2_350M_MAX_BATCH {
        return Err(format!(
            "lfm2_350m mq4 wmma batch_size {batch_size} out of range 1..={LFM2_350M_MAX_BATCH}"
        ));
    }
    Ok(())
}

fn validate_lfm2_350m_gate_up_shape(gate_m: usize, up_m: usize, k: usize) -> Result<(), String> {
    // Dense FFN gate/up: [intermediate, hidden] = [4608, 1024].
    if gate_m != LFM2_350M_INTERMEDIATE || up_m != LFM2_350M_INTERMEDIATE || k != LFM2_350M_HIDDEN {
        return Err(format!(
            "lfm2_350m gate_up requires gate_m=up_m={LFM2_350M_INTERMEDIATE} k={LFM2_350M_HIDDEN}, \
             got gate_m={gate_m} up_m={up_m} k={k}"
        ));
    }
    if k % 256 != 0 {
        return Err(format!(
            "lfm2_350m gate_up requires K multiple of 256, got {k}"
        ));
    }
    Ok(())
}

/// Exact LFM 350M residual projection shapes admitted on the MQ4 prefill path:
/// - conv in_proj:  M=3*hidden=3072, K=hidden=1024
/// - conv out_proj: M=hidden=1024,    K=hidden=1024
/// - attn out:      M=hidden=1024,    K=q_dim=1024
/// - ffn down:      M=hidden=1024,    K=intermediate=4608
fn validate_lfm2_350m_residual_shape(m: usize, k: usize) -> Result<(), String> {
    let ok = matches!((m, k), (3072, 1024) | (1024, 1024) | (1024, 4608));
    if !ok {
        return Err(format!(
            "lfm2_350m residual requires one of \
             (M,K) in {{(3072,1024),(1024,1024),(1024,4608)}}, got M={m} K={k}"
        ));
    }
    if k % 256 != 0 {
        return Err(format!(
            "lfm2_350m residual requires K multiple of 256, got {k}"
        ));
    }
    Ok(())
}

/// LFM2.5-350M-MQ4 / gfx1201 dense gate+up WMMA with adaptive BT{1,4,8,12}.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_350m_gate_up_wmma_gfx1201(
    gpu: &mut Gpu,
    a_gate: &GpuTensor,
    a_up: &GpuTensor,
    x: &GpuTensor,
    y_gate: &GpuTensor,
    y_up: &GpuTensor,
    gate_m: usize,
    up_m: usize,
    k: usize,
    batch_size: usize,
) -> Result<(), String> {
    validate_lfm2_350m_gfx1201(gpu, batch_size)?;
    validate_lfm2_350m_gate_up_shape(gate_m, up_m, k)?;

    let bt_b = lfm2_350m_adaptive_bt_b(batch_size);
    let (module, source, symbol) = match bt_b {
        12 => (
            LFM_GATE_UP_BT_MODULE,
            LFM_GATE_UP_BT_SRC,
            "lfm2_350m_mq4_gfx1201_gate_up_hfq4g256_wmma_bt12",
        ),
        8 => (
            LFM_GATE_UP_BT_MODULE,
            LFM_GATE_UP_BT_SRC,
            "lfm2_350m_mq4_gfx1201_gate_up_hfq4g256_wmma_bt8",
        ),
        4 => (
            LFM_GATE_UP_BT_MODULE,
            LFM_GATE_UP_BT_SRC,
            "lfm2_350m_mq4_gfx1201_gate_up_hfq4g256_wmma_bt4",
        ),
        _ => (
            LFM_GATE_UP_1ACC_MODULE,
            LFM_GATE_UP_1ACC_SRC,
            LFM_GATE_UP_1ACC_SYMBOL,
        ),
    };
    gpu.ensure_kernel_public(module, source, symbol)
        .map_err(|e| format!("compile LFM2 350M gate_up {symbol}: {e:?}"))?;
    let x_f16 = gpu
        .ensure_fp16_x_public(x, batch_size * k)
        .map_err(|e| format!("LFM2 350M gate_up fp16 x: {e:?}"))?;

    let ag = a_gate.buf.as_ptr();
    let au = a_up.buf.as_ptr();
    let xp = x_f16;
    let yg = y_gate.buf.as_ptr();
    let yu = y_up.buf.as_ptr();
    let g_m = gate_m as i32;
    let u_m = up_m as i32;
    let k_val = k as i32;
    let n_val = batch_size as i32;
    let mut params = vec![
        &ag as *const _ as *mut c_void,
        &au as *const _ as *mut c_void,
        &xp as *const _ as *mut c_void,
        &yg as *const _ as *mut c_void,
        &yu as *const _ as *mut c_void,
        &g_m as *const _ as *mut c_void,
        &u_m as *const _ as *mut c_void,
        &k_val as *const _ as *mut c_void,
        &n_val as *const _ as *mut c_void,
    ];
    let mut blob = hip_bridge::KernargBlob::new();
    blob.push_ptr(ag);
    blob.push_ptr(au);
    blob.push_ptr(xp);
    blob.push_ptr(yg);
    blob.push_ptr(yu);
    blob.push_i32(g_m);
    blob.push_i32(u_m);
    blob.push_i32(k_val);
    blob.push_i32(n_val);

    let total_m = gate_m + up_m;
    let row_tiles = (total_m + 15) / 16;
    let batch_tiles = (batch_size + 16 * bt_b - 1) / (16 * bt_b);
    gpu.launch_external_kernel(
        symbol,
        [row_tiles as u32, batch_tiles as u32, 1],
        [32, 1, 1],
        0,
        &mut params,
        blob,
    )
    .map_err(|e| format!("launch LFM2 350M gate_up {symbol}: {e:?}"))
}

/// LFM2.5-350M-MQ4 / gfx1201 residual WMMA with adaptive BT{1,4,8,12}.
/// Covers conv in/out, attention out, and FFN down on the admitted MQ4 path.
pub fn lfm2_350m_residual_wmma_gfx1201(
    gpu: &mut Gpu,
    a_raw: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    batch_size: usize,
) -> Result<(), String> {
    validate_lfm2_350m_gfx1201(gpu, batch_size)?;
    validate_lfm2_350m_residual_shape(m, k)?;

    let bt_b = lfm2_350m_adaptive_bt_b(batch_size);
    let (module, source, symbol) = match bt_b {
        12 => (
            LFM_RESID_BT_MODULE,
            LFM_RESID_BT_SRC,
            "lfm2_350m_mq4_gfx1201_residual_hfq4g256_wmma_bt12",
        ),
        8 => (
            LFM_RESID_BT_MODULE,
            LFM_RESID_BT_SRC,
            "lfm2_350m_mq4_gfx1201_residual_hfq4g256_wmma_bt8",
        ),
        4 => (
            LFM_RESID_BT_MODULE,
            LFM_RESID_BT_SRC,
            "lfm2_350m_mq4_gfx1201_residual_hfq4g256_wmma_bt4",
        ),
        _ => (
            LFM_RESID_1ACC_MODULE,
            LFM_RESID_1ACC_SRC,
            LFM_RESID_1ACC_SYMBOL,
        ),
    };
    gpu.ensure_kernel_public(module, source, symbol)
        .map_err(|e| format!("compile LFM2 350M residual {symbol}: {e:?}"))?;
    let x_f16 = gpu
        .ensure_fp16_x_public(x, batch_size * k)
        .map_err(|e| format!("LFM2 350M residual fp16 x: {e:?}"))?;

    let a_ptr = a_raw.buf.as_ptr();
    let x_ptr = x_f16;
    let y_ptr = y.buf.as_ptr();
    let m_val = m as i32;
    let k_val = k as i32;
    let bs_val = batch_size as i32;
    let mut params = vec![
        &a_ptr as *const _ as *mut c_void,
        &x_ptr as *const _ as *mut c_void,
        &y_ptr as *const _ as *mut c_void,
        &m_val as *const _ as *mut c_void,
        &k_val as *const _ as *mut c_void,
        &bs_val as *const _ as *mut c_void,
    ];
    let mut blob = hip_bridge::KernargBlob::new();
    blob.push_ptr(a_ptr);
    blob.push_ptr(x_ptr);
    blob.push_ptr(y_ptr);
    blob.push_i32(m_val);
    blob.push_i32(k_val);
    blob.push_i32(bs_val);

    let row_tiles = (m + 15) / 16;
    let batch_tiles = (batch_size + 16 * bt_b - 1) / (16 * bt_b);
    gpu.launch_external_kernel(
        symbol,
        [row_tiles as u32, batch_tiles as u32, 1],
        [32, 1, 1],
        0,
        &mut params,
        blob,
    )
    .map_err(|e| format!("launch LFM2 350M residual {symbol}: {e:?}"))
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

    #[test]
    fn adaptive_bt_policy_matches_lfm_route() {
        assert_eq!(lfm2_350m_adaptive_bt_b(1), 1);
        assert_eq!(lfm2_350m_adaptive_bt_b(63), 1);
        assert_eq!(lfm2_350m_adaptive_bt_b(64), 4);
        assert_eq!(lfm2_350m_adaptive_bt_b(128), 8);
        assert_eq!(lfm2_350m_adaptive_bt_b(192), 12);
        assert_eq!(lfm2_350m_adaptive_bt_b(96), 4);
        assert_eq!(lfm2_350m_adaptive_bt_b(160), 8);
        assert_eq!(lfm2_350m_adaptive_bt_b(200), 12);
    }

    #[test]
    fn residual_shape_admits_350m_projections() {
        assert!(validate_lfm2_350m_residual_shape(3072, 1024).is_ok());
        assert!(validate_lfm2_350m_residual_shape(1024, 1024).is_ok());
        assert!(validate_lfm2_350m_residual_shape(1024, 4608).is_ok());
        assert!(validate_lfm2_350m_residual_shape(2048, 1024).is_err());
        assert!(validate_lfm2_350m_gate_up_shape(4608, 4608, 1024).is_ok());
        assert!(validate_lfm2_350m_gate_up_shape(8192, 8192, 2048).is_err());
    }
}
