// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-FP16 projection-input producers for S3-f16-projection-inputs
//! (DFlash launch fusion, gfx1100 only).
//!
//! For the 48 LA qkvza, 16 FA qkv, and 64 gate/up inputs, the old path is
//! `fused_rmsnorm_rotate_mq[_awq]_batched` (F32 `x_rot`) followed by a
//! `convert_f32_to_f16` launch feeding the `*_mq4g256v2_wmma` base GEMMs.
//! This module emits the identical F16 bytes directly:
//!
//! - [`Gpu::fused_rmsnorm_rotate_mq_f16_batched`] /
//!   [`Gpu::fused_rmsnorm_rotate_mq_awq_f16_batched`]: operation-order-exact
//!   clones of the F32 producers (see
//!   `kernels/src/fused_rmsnorm_mq_rotate_f16.gfx1100.hip`) storing
//!   `(_Float16)` directly into the caller-owned F16 sidecar.
//! - [`Gpu::gemm_qkvza_mq4g256v2_wmma_f16`] /
//!   [`Gpu::gemm_qkv_mq4g256v2_wmma_f16`] /
//!   [`Gpu::gemm_gate_up_mq4g256v2_wmma_f16`]: the historical base GEMM
//!   launch bodies with the F16 pointer consumed directly — they validate
//!   `DType::F16` and never call `ensure_fp16_x`, never consult or update
//!   `fp16_x_source_ptr`.
//!
//! Route contract (mirrored by the prefill hook predicate): exact gfx1100,
//! `DflashFusionCtx::ChainVerify`, N<=16, MQ4G256V2 weights, graph-off and
//! no active replay recording, `HIPFIRE_MQ_F16_PROJECTION_OFF != 1`. Every
//! failed predicate runs the pre-change path; these entries return
//! `Err` on a non-gfx1100 arch or non-F16 input rather than silently
//! falling back. New kernels use `launch_maybe_blob` with the inline
//! `KernargBlob` builder (capture-safe ABI, same as the baselines).

use std::ffi::c_void;

use crate::dispatch::{DType, Gpu, GpuTensor};
use hip_bridge::HipResult;

/// Self-contained source: this module never touches the shared `kernels.rs`
/// registry (owned by no slice — the prescaffold reservation did not land),
/// so concurrent slices cannot conflict here.
pub const FUSED_RMSNORM_MQ_ROTATE_F16_SRC: &str =
    include_str!("../../../kernels/src/fused_rmsnorm_mq_rotate_f16.gfx1100.hip");

impl Gpu {
    /// Fused RMSNorm + FWHT rotation writing exact FP16 directly.
    ///
    /// Bit contract: every stored element equals the historical
    /// `fused_rmsnorm_rotate_mq_batched` F32 output followed by
    /// `convert_f32_to_f16`. Same grid/block/shared reservation as the
    /// baseline launcher; same `ensure_mq_signs` inputs.
    pub fn fused_rmsnorm_rotate_mq_f16_batched(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot_f16: &GpuTensor,
        k: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                0,
                "fused_rmsnorm_rotate_mq_f16_batched: exact gfx1100 only",
            ));
        }
        if x_rot_f16.dtype != DType::F16 {
            return Err(hip_bridge::HipError::new(
                0,
                "fused_rmsnorm_rotate_mq_f16_batched: x_rot_f16 must be DType::F16",
            ));
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_rmsnorm_mq_rotate_f16",
            FUSED_RMSNORM_MQ_ROTATE_F16_SRC,
            "fused_rmsnorm_mq_rotate_f16",
        )?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let mut xp = x.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut xrp = x_rot_f16.buf.as_ptr();
        let mut kv = k as i32;
        let mut eps_v = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut eps_v as *mut _ as *mut c_void,
        ];
        let block_size = 256u32;
        let shared_mem = ((k + 256) * 4) as u32;
        let bytes = (k * 4 * 3 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_rmsnorm_rotate_mq_f16_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "fused_rmsnorm_mq_rotate_f16",
            [batch_size as u32, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(wp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(xrp);
                b.push_i32(kv);
                b.push_f32(eps_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        // Deliberately no invalidate_x_caches_for: the F16 sidecar is never
        // consulted via fp16_x_source_ptr, and the F16 GEMM entries below
        // never populate that cache — the shared F32 oracle path is untouched.
        result
    }

    /// AWQ exact-FP16 producer. Bit contract: every stored element equals the
    /// historical `fused_rmsnorm_rotate_mq_awq_batched` F32 output (identical
    /// for the base and the gfx1100-direct AWQ kernels — same value operation
    /// order) followed by `convert_f32_to_f16`.
    pub fn fused_rmsnorm_rotate_mq_awq_f16_batched(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot_f16: &GpuTensor,
        k: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                0,
                "fused_rmsnorm_rotate_mq_awq_f16_batched: exact gfx1100 only",
            ));
        }
        if x_rot_f16.dtype != DType::F16 {
            return Err(hip_bridge::HipError::new(
                0,
                "fused_rmsnorm_rotate_mq_awq_f16_batched: x_rot_f16 must be DType::F16",
            ));
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_rmsnorm_mq_rotate_awq_f16",
            FUSED_RMSNORM_MQ_ROTATE_F16_SRC,
            "fused_rmsnorm_mq_rotate_awq_f16",
        )?;
        let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let mut xp = x.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut awp = awq_scale.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut xrp = x_rot_f16.buf.as_ptr();
        let mut kv = k as i32;
        let mut eps_v = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut awp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut eps_v as *mut _ as *mut c_void,
        ];
        let block_size = 256u32;
        // Direct-structure kernel: reduce[256] only, like the gfx1100-direct
        // AWQ launcher.
        let shared_mem = (256 * 4) as u32;
        let bytes = (k * 4 * 4 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_rmsnorm_rotate_mq_awq_f16_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "fused_rmsnorm_mq_rotate_awq_f16",
            [batch_size as u32, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(wp);
                b.push_ptr(awp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(xrp);
                b.push_i32(kv);
                b.push_f32(eps_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ4V2 qkvza base GEMM consuming a caller-owned F16 activation.
    ///
    /// Launch-body-exact copy of the `gemm_qkvza_mq4g256v2_wmma` historical
    /// base path (same module/symbol, grid, block, kernarg order, byte
    /// accounting) except `xp` is the validated F16 pointer — no
    /// `ensure_fp16_x`, no `fp16_x_source_ptr` traffic. The MMQ/BT perf
    /// policies of the base launcher are intentionally absent: callers
    /// guarantee the exact route (gfx1100, N<=16, graph-off, no recording),
    /// where the base launcher itself falls through to this same base
    /// kernel. Calibration taps mirror the `FusedQkvzaMq4G256V2` run-arm.
    pub fn gemm_qkvza_mq4g256v2_wmma_f16(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x_f16: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                0,
                "gemm_qkvza_mq4g256v2_wmma_f16: exact gfx1100 only",
            ));
        }
        if x_f16.dtype != DType::F16 {
            return Err(hip_bridge::HipError::new(
                0,
                "gemm_qkvza_mq4g256v2_wmma_f16: x_f16 must be DType::F16",
            ));
        }
        self.maybe_capture_activation(a_qkv, x_f16, batch_size, k);
        self.maybe_capture_activation(a_z, x_f16, batch_size, k);
        self.maybe_capture_activation(a_beta, x_f16, batch_size, k);
        self.maybe_capture_activation(a_alpha, x_f16, batch_size, k);
        self.bind_thread()?;
        let kname = "gemm_qkvza_mq4g256v2_wmma";
        let ksrc = crate::kernels::GEMM_QKVZA_MQ4G256V2_WMMA_SRC;
        self.ensure_kernel(kname, ksrc, kname)?;
        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16.buf.as_ptr();
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];
        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_mq4g256v2_wmma_f16", bytes);
        let result = self.launch_maybe_blob(
            kname,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m);
                b.push_i32(z_m_val);
                b.push_i32(b_m);
                b.push_i32(a_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ4V2 qkv base GEMM consuming a caller-owned F16 activation.
    ///
    /// Same contract as [`Gpu::gemm_qkvza_mq4g256v2_wmma_f16`]: launch-body
    /// copy of the `gemm_qkv_mq4g256v2_wmma` historical base path with the
    /// validated F16 pointer. Taps mirror the `FusedQkvMq4G256V2` run-arm.
    pub fn gemm_qkv_mq4g256v2_wmma_f16(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x_f16: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                0,
                "gemm_qkv_mq4g256v2_wmma_f16: exact gfx1100 only",
            ));
        }
        if x_f16.dtype != DType::F16 {
            return Err(hip_bridge::HipError::new(
                0,
                "gemm_qkv_mq4g256v2_wmma_f16: x_f16 must be DType::F16",
            ));
        }
        self.maybe_capture_activation(a_q, x_f16, batch_size, k);
        self.maybe_capture_activation(a_k, x_f16, batch_size, k);
        self.maybe_capture_activation(a_v, x_f16, batch_size, k);
        self.bind_thread()?;
        let kname = "gemm_qkv_mq4g256v2_wmma";
        let ksrc = crate::kernels::GEMM_QKV_MQ4G256V2_WMMA_SRC;
        self.ensure_kernel(kname, ksrc, kname)?;
        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16.buf.as_ptr();
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];
        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_mq4g256v2_wmma_f16", bytes);
        let result = self.launch_maybe_blob(
            kname,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ4V2 gate/up base GEMM consuming a caller-owned F16 activation.
    ///
    /// Same contract as [`Gpu::gemm_qkvza_mq4g256v2_wmma_f16`]: launch-body
    /// copy of the `gemm_gate_up_mq4g256v2_wmma` historical base path with
    /// the validated F16 pointer. Taps mirror the `FusedGateUpMq4G256V2`
    /// run-arm.
    pub fn gemm_gate_up_mq4g256v2_wmma_f16(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x_f16: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                0,
                "gemm_gate_up_mq4g256v2_wmma_f16: exact gfx1100 only",
            ));
        }
        if x_f16.dtype != DType::F16 {
            return Err(hip_bridge::HipError::new(
                0,
                "gemm_gate_up_mq4g256v2_wmma_f16: x_f16 must be DType::F16",
            ));
        }
        self.maybe_capture_activation(a_gate, x_f16, batch_size, k);
        self.maybe_capture_activation(a_up, x_f16, batch_size, k);
        self.bind_thread()?;
        let kname = "gemm_gate_up_mq4g256v2_wmma";
        let ksrc = crate::kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_SRC;
        self.ensure_kernel(kname, ksrc, kname)?;
        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16.buf.as_ptr();
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];
        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_mq4g256v2_wmma_f16",
            bytes,
        );
        let result = self.launch_maybe_blob(
            kname,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(g_m);
                b.push_i32(u_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
