// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fused kernels (gate-up, QKV+Z+A, rmsnorm+rope+rotate fusions). Pure move (Phase 1 M3).

use super::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// Fused silu(gate)*up + per-channel scale + krot rounds of Givens
    /// rotation. Single-launch replacement for the
    /// `silu_mul_f32 + givens_rotate` pair used by the ParoQuant routed
    /// gate→down hop. Same shared-memory + grid contract as
    /// `givens_rotate`, plus two additional input pointers (gate, up)
    /// and a separate output pointer.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_silu_mul_givens_rotate_f32(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        out: &GpuTensor,
        pairs: &GpuTensor,
        theta: &GpuTensor,
        channel_scales: &GpuTensor,
        seq_len: usize,
        hidden_dim: usize,
        krot: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_silu_mul_givens_rotate_f32",
            kernels::FUSED_SILU_MUL_GIVENS_ROTATE_SRC,
            "fused_silu_mul_givens_rotate_f32",
        )?;

        let cta_m: u32 = 4;
        let group_size: u32 = 128;
        let groups_per_row = (hidden_dim as u32 + group_size - 1) / group_size;
        let grid_x = ((seq_len as u32) + cta_m - 1) / cta_m;

        let gate_ptr = gate.buf.as_ptr();
        let up_ptr = up.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let pairs_ptr = pairs.buf.as_ptr();
        let theta_ptr = theta.buf.as_ptr();
        let cs_ptr = channel_scales.buf.as_ptr();
        let seq_val = seq_len as i32;
        let dim_val = hidden_dim as i32;
        let krot_val = krot as i32;

        let smem = (cta_m * group_size * 4) as u32;

        // Bytes: read gate (seq × dim × 4) + read up (seq × dim × 4) + write out
        // (seq × dim × 4) + read pairs/theta/scales (krot × dim × 2 + dim × 2).
        let bytes = seq_len * hidden_dim * 4 * 3 + krot * hidden_dim * 2 + hidden_dim * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_silu_mul_givens_rotate_f32",
            bytes,
        );
        let result = self.launch_kernargs(
            "fused_silu_mul_givens_rotate_f32",
            [grid_x, groups_per_row, 1],
            [group_size / 2, 1, 1],
            smem,
            &kernargs![ptr gate_ptr, ptr up_ptr, ptr out_ptr, ptr pairs_ptr, ptr theta_ptr, ptr cs_ptr, i32 seq_val, i32 dim_val, i32 krot_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Lever 1 — Fused RMSNorm + PARO4G128T per-group Givens rotation.
    ///
    /// Replaces `rmsnorm_f32(x, weight) -> x_norm` followed by
    /// `paro4g128t_rotate(A, x_norm) -> x_rot` with a single launch:
    /// `fused_rmsnorm_paro4g128t_rotate(A, x_pre, weight) -> x_rot, x_norm`.
    /// Math identity is `(x * weight * rms) * channel_scales -> KROT Givens`,
    /// numerically equivalent to the separated path within FP16 epsilon
    /// (float mul reorder is the only difference).
    ///
    /// When `x_norm` is `Some`, also emits the post-rmsnorm activation so
    /// subsequent paro linears in the same residual block can apply their
    /// own rotation (each linear has different pairs/theta/channel_scales).
    /// When `None`, x_norm write is skipped — useful when this is the last
    /// linear in a block, or for byte-equivalence smoke tests.
    ///
    /// Layout: 1 workgroup, 256 threads, dynamic LDS = (K + 256) * 4 bytes.
    /// K must be a multiple of 128 (PARO group size). Engine layout only
    /// (PARO4G128T, qtype 29) — the kernel assumes the precomputed sincos
    /// trig payload.
    pub fn fused_rmsnorm_paro4g128t_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        x_norm: Option<&GpuTensor>,
        m: usize,
        k: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "fused_rmsnorm_paro4g128t_rotate requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "fused_rmsnorm_paro4g128t_rotate requires K multiple of 128, got {k}"
        );
        assert!(
            x_rot.buf.size() / 4 >= k,
            "fused_rmsnorm_paro4g128t_rotate x_rot scratch too small: {} floats for K={k}",
            x_rot.buf.size() / 4
        );
        if let Some(xn) = x_norm {
            assert!(
                xn.buf.size() / 4 >= k,
                "fused_rmsnorm_paro4g128t_rotate x_norm scratch too small: {} floats for K={k}",
                xn.buf.size() / 4
            );
        }
        self.ensure_kernel(
            "fused_rmsnorm_paro4g128t_rotate",
            kernels::FUSED_RMSNORM_PARO4G128T_ROTATE_SRC,
            "fused_rmsnorm_paro4g128t_rotate",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let weight_ptr = weight.buf.as_ptr();
        let x_rot_ptr = x_rot.buf.as_ptr();
        let x_norm_ptr = x_norm
            .map(|t| t.buf.as_ptr())
            .unwrap_or(std::ptr::null_mut());
        let m_val = m as i32;
        let k_val = k as i32;
        let eps_val = eps;

        let block_size = 256u32;
        // LDS: x_shared[K] + reduce[256]
        let shared_mem = ((k + 256) * 4) as u32;
        // BW estimate: paro rotate bytes + extra x + weight read; if x_norm emit, +K floats write
        let bytes = crate::profile::paro4g128t_rotate_bytes(m, k)
            + k * 4
            + if x_norm.is_some() { k * 4 } else { 0 };
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_rmsnorm_paro4g128t_rotate",
            bytes,
        );
        let result = self.launch_kernargs(
            "fused_rmsnorm_paro4g128t_rotate",
            [1, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &kernargs![ptr a_ptr, ptr x_ptr, ptr weight_ptr, ptr x_rot_ptr, ptr x_norm_ptr, i32 m_val, i32 k_val, f32 eps_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(x_rot_ptr);
        if x_norm.is_some() {
            self.invalidate_x_caches_for(x_norm_ptr);
        }
        result
    }
    /// PARO4-G128T fused gate/up decode path. Gate and up have distinct
    /// Paro rotations, so this still rotates both, but batches the two
    /// rotations and the two pack4 GEMVs into two launches instead of four.
    pub fn fused_gate_up_paro4g128t(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        x_rot_gate: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128T fused gate/up requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128T fused gate/up requires K multiple of 128, got {k}"
        );
        assert!(
            x_rot_gate.buf.size() / 4 >= k,
            "PARO4G128T fused gate/up gate scratch too small: {} floats for K={k}",
            x_rot_gate.buf.size() / 4
        );
        self.ensure_mq_signs()?;
        let x_rot_up = GpuTensor {
            buf: unsafe { self.mq_x_rot.as_ref().unwrap().buf.alias() },
            shape: vec![self.mq_x_rot.as_ref().unwrap().buf.size() / 4],
            dtype: DType::F32,
        };
        assert!(
            x_rot_up.buf.size() / 4 >= k,
            "PARO4G128T fused gate/up up scratch too small: {} floats for K={k}",
            x_rot_up.buf.size() / 4
        );

        let shared_pairs = std::env::var_os("HIPFIRE_PARO_SHARED_PAIRS").is_some();
        let rotate_kernel = if shared_pairs {
            "paro4g128t_dual_rotate_shared_pairs"
        } else {
            "paro4g128t_dual_rotate"
        };
        let use_pack2 = std::env::var_os("HIPFIRE_PARO_FUSED_PACK2").is_some();
        let gemv_kernel = if use_pack2 {
            "fused_gate_up_paro4g128t_pack2"
        } else {
            "fused_gate_up_paro4g128t_pack4"
        };
        self.ensure_kernel("gemv_paro4g128", kernels::GEMV_PARO4G128_SRC, rotate_kernel)?;
        self.ensure_kernel("gemv_paro4g128", kernels::GEMV_PARO4G128_SRC, gemv_kernel)?;

        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let xrg = x_rot_gate.buf.as_ptr();
        let xru = x_rot_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let groups = (k / 128) as u32;
        let rotate_bytes = crate::profile::paro4g128t_rotate_bytes(m, k) * 2;
        let rotate_timer =
            crate::profile::begin_timer(&self.hip, "format", rotate_kernel, rotate_bytes);
        let rotate_result = self.launch_kernargs(
            rotate_kernel,
            [groups, if shared_pairs { 1 } else { 2 }, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr ag, ptr au, ptr xp, ptr xrg, ptr xru, i32 m_val, i32 k_val],
        );
        if let Some(t) = rotate_timer {
            t.finish(&self.hip);
        }
        rotate_result?;
        self.invalidate_x_caches_for(xrg);
        self.invalidate_x_caches_for(xru);

        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let pack_multiplier = if use_pack2 { 8 } else { 4 };
        let gemv_bytes = crate::profile::gemv_paro4g128_prerotated_bytes(m, k) * pack_multiplier;
        let gemv_timer = crate::profile::begin_timer(&self.hip, "gemv", gemv_kernel, gemv_bytes);
        let gemv_result = self.launch_kernargs(
            gemv_kernel,
            [(m / if use_pack2 { 2 } else { 4 }) as u32, 2, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr ag, ptr au, ptr xrg, ptr xru, ptr yg, ptr yu, i32 m_val, i32 k_val],
        );
        if let Some(t) = gemv_timer {
            t.finish(&self.hip);
        }
        gemv_result
    }
    /// PARO4-G128T fused LA projection path. The four Paro projections have
    /// distinct rotations, so this batches four rotates and four pack4 GEMVs
    /// into two launches.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkvza_paro4g128t(
        &mut self,
        a0: &GpuTensor,
        a1: &GpuTensor,
        a2: &GpuTensor,
        a3: &GpuTensor,
        x: &GpuTensor,
        y0: &GpuTensor,
        y1: &GpuTensor,
        y2: &GpuTensor,
        y3: &GpuTensor,
        x_rot0: &GpuTensor,
        x_rot1: &GpuTensor,
        x_rot2: &GpuTensor,
        x_rot3: &GpuTensor,
        m0: usize,
        m1: usize,
        m2: usize,
        m3: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        for (label, m) in [("m0", m0), ("m1", m1), ("m2", m2), ("m3", m3)] {
            assert_eq!(
                m % 8,
                0,
                "PARO4G128T fused LA {label} requires M multiple of 8, got {m}"
            );
        }
        assert_eq!(
            k % 128,
            0,
            "PARO4G128T fused LA requires K multiple of 128, got {k}"
        );
        for (label, scratch) in [
            ("x_rot0", x_rot0),
            ("x_rot1", x_rot1),
            ("x_rot2", x_rot2),
            ("x_rot3", x_rot3),
        ] {
            assert!(
                scratch.buf.size() / 4 >= k,
                "PARO4G128T fused LA {label} scratch too small: {} floats for K={k}",
                scratch.buf.size() / 4
            );
        }
        let shared_pairs = std::env::var_os("HIPFIRE_PARO_SHARED_PAIRS").is_some();
        let rotate_kernel = if shared_pairs {
            "paro4g128t_quad_rotate_shared_pairs"
        } else {
            "paro4g128t_quad_rotate"
        };
        let use_pack2 = std::env::var_os("HIPFIRE_PARO_FUSED_PACK2").is_some();
        let gemv_kernel = if use_pack2 {
            "fused_qkvza_paro4g128t_pack2"
        } else {
            "fused_qkvza_paro4g128t_pack4"
        };
        self.ensure_kernel("gemv_paro4g128", kernels::GEMV_PARO4G128_SRC, rotate_kernel)?;
        self.ensure_kernel("gemv_paro4g128", kernels::GEMV_PARO4G128_SRC, gemv_kernel)?;

        let a0p = a0.buf.as_ptr();
        let a1p = a1.buf.as_ptr();
        let a2p = a2.buf.as_ptr();
        let a3p = a3.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let xr0p = x_rot0.buf.as_ptr();
        let xr1p = x_rot1.buf.as_ptr();
        let xr2p = x_rot2.buf.as_ptr();
        let xr3p = x_rot3.buf.as_ptr();
        let y0p = y0.buf.as_ptr();
        let y1p = y1.buf.as_ptr();
        let y2p = y2.buf.as_ptr();
        let y3p = y3.buf.as_ptr();
        let m0v = m0 as i32;
        let m1v = m1 as i32;
        let m2v = m2 as i32;
        let m3v = m3 as i32;
        let kv = k as i32;

        let groups = (k / 128) as u32;
        let rotate_bytes = crate::profile::paro4g128t_rotate_bytes(m0, k)
            + crate::profile::paro4g128t_rotate_bytes(m1, k)
            + crate::profile::paro4g128t_rotate_bytes(m2, k)
            + crate::profile::paro4g128t_rotate_bytes(m3, k);
        let rotate_timer =
            crate::profile::begin_timer(&self.hip, "format", rotate_kernel, rotate_bytes);
        let rotate_result = self.launch_kernargs(
            rotate_kernel,
            [groups, if shared_pairs { 1 } else { 4 }, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr a0p, ptr a1p, ptr a2p, ptr a3p, ptr xp, ptr xr0p, ptr xr1p, ptr xr2p, ptr xr3p, i32 m0v, i32 m1v, i32 m2v, i32 m3v, i32 kv],
        );
        if let Some(t) = rotate_timer {
            t.finish(&self.hip);
        }
        rotate_result?;
        for ptr in [xr0p, xr1p, xr2p, xr3p] {
            self.invalidate_x_caches_for(ptr);
        }

        let max_m = m0.max(m1).max(m2).max(m3);
        let pack_multiplier = if use_pack2 { 4 } else { 2 };
        let gemv_bytes = (crate::profile::gemv_paro4g128_prerotated_bytes(m0, k)
            + crate::profile::gemv_paro4g128_prerotated_bytes(m1, k)
            + crate::profile::gemv_paro4g128_prerotated_bytes(m2, k)
            + crate::profile::gemv_paro4g128_prerotated_bytes(m3, k))
            * pack_multiplier;
        let gemv_timer = crate::profile::begin_timer(&self.hip, "gemv", gemv_kernel, gemv_bytes);
        let gemv_result = self.launch_kernargs(
            gemv_kernel,
            [(max_m / if use_pack2 { 2 } else { 4 }) as u32, 4, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr a0p, ptr a1p, ptr a2p, ptr a3p, ptr xr0p, ptr xr1p, ptr xr2p, ptr xr3p, ptr y0p, ptr y1p, ptr y2p, ptr y3p, i32 m0v, i32 m1v, i32 m2v, i32 m3v, i32 kv],
        );
        if let Some(t) = gemv_timer {
            t.finish(&self.hip);
        }
        gemv_result
    }
    /// Fused Gate+Up MQ4-Lloyd: two GEMVs in one launch. Mirrors
    /// fused_gate_up_mq3g256_lloyd. Caller is responsible for pre-rotating x.
    pub fn fused_gate_up_mq4g256_lloyd(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_gate_up_mq4g256_lloyd_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "fused_gate_up_mq4g256_lloyd")?;
        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let gm = gate_m as i32;
        let um = up_m as i32;
        let kv = k as i32;
        let total = (gate_m + up_m) as u32;
        let bytes = crate::profile::gemv_mq4g256_lloyd_bytes(gate_m, k)
            + crate::profile::gemv_mq4g256_lloyd_bytes(up_m, k)
            - k * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_gate_up_mq4g256_lloyd", bytes);
        let result = self.launch_kernargs(
            "fused_gate_up_mq4g256_lloyd",
            [total, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr ag, ptr au, ptr xp, ptr yg, ptr yu, i32 gm, i32 um, i32 kv],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Fused QKVZA MQ4-Lloyd: 4 LA-preamble GEMVs in one launch.
    pub fn fused_qkvza_mq4g256_lloyd(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_qkvza_mq4g256_lloyd_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "fused_qkvza_mq4g256_lloyd")?;
        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let q_m_i = qkv_m as i32;
        let z_m_i = z_m as i32;
        let b_m_i = beta_m as i32;
        let a_m_i = alpha_m as i32;
        let k_i = k as i32;
        let total = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let bytes = crate::profile::gemv_mq4g256_lloyd_bytes(qkv_m, k)
            + crate::profile::gemv_mq4g256_lloyd_bytes(z_m, k)
            + crate::profile::gemv_mq4g256_lloyd_bytes(beta_m, k)
            + crate::profile::gemv_mq4g256_lloyd_bytes(alpha_m, k)
            - 3 * (k * 4);
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_qkvza_mq4g256_lloyd", bytes);
        let result = self.launch_kernargs(
            "fused_qkvza_mq4g256_lloyd",
            [total, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr aq, ptr az, ptr ab, ptr aa, ptr xp, ptr yq, ptr yz, ptr yb, ptr ya, i32 q_m_i, i32 z_m_i, i32 b_m_i, i32 a_m_i, i32 k_i],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Fused QKV MQ4-Lloyd: 3 FA-preamble GEMVs in one launch.
    pub fn fused_qkv_mq4g256_lloyd(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_qkv_mq4g256_lloyd_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "fused_qkv_mq4g256_lloyd")?;
        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_i = q_m as i32;
        let k_m_i = k_m as i32;
        let v_m_i = v_m as i32;
        let k_i = k as i32;
        let total = (q_m + k_m + v_m) as u32;
        let bytes = crate::profile::gemv_mq4g256_lloyd_bytes(q_m, k)
            + crate::profile::gemv_mq4g256_lloyd_bytes(k_m, k)
            + crate::profile::gemv_mq4g256_lloyd_bytes(v_m, k)
            - 2 * (k * 4);
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_qkv_mq4g256_lloyd", bytes);
        let result = self.launch_kernargs(
            "fused_qkv_mq4g256_lloyd",
            [total, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr aq, ptr ak, ptr av, ptr xp, ptr yq, ptr yk, ptr yv, i32 q_m_i, i32 k_m_i, i32 v_m_i, i32 k_i],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Fused Gate+Up MQ3-Lloyd: two GEMVs in one launch. Mirrors
    /// `fused_gate_up_hfq4g256` for the Lloyd-MQ3 dtype. Caller is
    /// responsible for pre-rotating x (FWHT) before invoking; the kernel
    /// itself only does the GEMV. Both `a_gate` and `a_up` must be MQ3-Lloyd
    /// matrices with the same K and codebook layout.
    pub fn fused_gate_up_mq3g256_lloyd(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_gate_up_mq3g256_lloyd_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "fused_gate_up_mq3g256_lloyd")?;
        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let gm = gate_m as i32;
        let um = up_m as i32;
        let kv = k as i32;
        let total = (gate_m + up_m) as u32;
        // Bandwidth: A_gate + A_up read, x read once, y_gate + y_up written.
        let bytes = crate::profile::gemv_mq3g256_lloyd_bytes(gate_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(up_m, k)
            - k * 4; // x is shared, don't double-count
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_gate_up_mq3g256_lloyd", bytes);
        let result = self.launch_kernargs(
            "fused_gate_up_mq3g256_lloyd",
            [total, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr ag, ptr au, ptr xp, ptr yg, ptr yu, i32 gm, i32 um, i32 kv],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Fused QKVZA MQ3-Lloyd: 4 LA-preamble GEMVs in one launch. Used by
    /// qwen35.rs DeltaNet decode when wqkv + wz + w_beta + w_alpha are
    /// all MQ3G256Lloyd. Mirrors `fused_qkvza_hfq4g256` — same routing
    /// (grid = qkv_m + z_m + beta_m + alpha_m, block picks A by gid),
    /// Lloyd K4+LDS body on gfx1100. Caller is responsible for
    /// pre-rotating x (FWHT); the kernel only does the GEMVs.
    pub fn fused_qkvza_mq3g256_lloyd(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_qkvza_mq3g256_lloyd_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "fused_qkvza_mq3g256_lloyd")?;
        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let q_m_i = qkv_m as i32;
        let z_m_i = z_m as i32;
        let b_m_i = beta_m as i32;
        let a_m_i = alpha_m as i32;
        let k_i = k as i32;
        let total = (qkv_m + z_m + beta_m + alpha_m) as u32;
        // Bandwidth: 4 weight matrices read once each, x shared (read once).
        let bytes = crate::profile::gemv_mq3g256_lloyd_bytes(qkv_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(z_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(beta_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(alpha_m, k)
            - 3 * (k * 4); // x is shared, don't quadruple-count
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_qkvza_mq3g256_lloyd", bytes);
        let result = self.launch_kernargs(
            "fused_qkvza_mq3g256_lloyd",
            [total, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr aq, ptr az, ptr ab, ptr aa, ptr xp, ptr yq, ptr yz, ptr yb, ptr ya, i32 q_m_i, i32 z_m_i, i32 b_m_i, i32 a_m_i, i32 k_i],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Fused QKV MQ3-Lloyd: 3 FA-preamble GEMVs in one launch. Used by
    /// qwen35.rs FullAttention decode when wq + wk + wv are all
    /// MQ3G256Lloyd. Sibling of `fused_qkvza_mq3g256_lloyd` for the
    /// 3-projection FA case (vs LA's 4-projection QKVZA). Caller is
    /// responsible for pre-rotating x; the kernel only does the GEMVs.
    pub fn fused_qkv_mq3g256_lloyd(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::fused_qkv_mq3g256_lloyd_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "fused_qkv_mq3g256_lloyd")?;
        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_i = q_m as i32;
        let k_m_i = k_m as i32;
        let v_m_i = v_m as i32;
        let k_i = k as i32;
        let total = (q_m + k_m + v_m) as u32;
        // Bandwidth: 3 weight matrices read once each, x shared (read once).
        let bytes = crate::profile::gemv_mq3g256_lloyd_bytes(q_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(k_m, k)
            + crate::profile::gemv_mq3g256_lloyd_bytes(v_m, k)
            - 2 * (k * 4); // x is shared, don't triple-count
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_qkv_mq3g256_lloyd", bytes);
        let result = self.launch_kernargs(
            "fused_qkv_mq3g256_lloyd",
            [total, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr aq, ptr ak, ptr av, ptr xp, ptr yq, ptr yk, ptr yv, i32 q_m_i, i32 k_m_i, i32 v_m_i, i32 k_i],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Fused RMSNorm + MagnumQuant FWHT rotation. Replaces the
    /// `rmsnorm_f32` + `rotate_x_mq` sequence with a single kernel launch.
    /// Reads unnormalized `x` + rmsnorm `weight`, computes rmsnorm in LDS,
    /// applies the same per-256-element FWHT as `mq_rotate_x`, and writes
    /// the rotated normalized vector into `x_rot`.
    ///
    /// Preconditions:
    /// - `k` is a multiple of 256 (enforced by callers via `config.dim`)
    /// - `k` ≤ 16384 (LDS ceiling; 16K floats = 64KB minus reduce buffer)
    pub fn fused_rmsnorm_rotate_mq(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx94x split: opt-in via HIPFIRE_GFX942_RMSNORM_SPLIT=1.
        // Two-kernel path (reduce + rotate) gives 5× more in-flight wave64s
        // on prefill scale; modest decode change. Math byte-identical.
        if self.flags.gfx942_rmsnorm_split {
            return self.fused_rmsnorm_rotate_mq_split_gfx942(x, weight, x_rot, k, eps, 1);
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        let (module_name, kernel_src, kernel_name, timer_name, reduce_slots) =
            if self.arch_caps.is_gfx1151() {
                (
                    "fused_rmsnorm_mq_rotate_gfx1151",
                    kernels::FUSED_RMSNORM_MQ_ROTATE_GFX1151_SRC,
                    "fused_rmsnorm_mq_rotate_gfx1151",
                    "fused_rmsnorm_mq_rotate_gfx1151",
                    8usize,
                )
            } else {
                (
                    "fused_rmsnorm_mq_rotate",
                    kernels::FUSED_RMSNORM_MQ_ROTATE_SRC,
                    "fused_rmsnorm_mq_rotate",
                    "fused_rmsnorm_mq_rotate",
                    256usize,
                )
            };
        self.ensure_kernel(module_name, kernel_src, kernel_name)?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let xp = x.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let eps_v = eps;

        let block_size = 256u32;
        // Dynamic LDS: K floats for x_shared plus the selected reduction
        // scratch (generic=256 floats, gfx1151=8 wave sums).
        let shared_mem = ((k + reduce_slots) * 4) as u32;

        // Bandwidth: read x (K*4) + weight (K*4) + signs (2*256*4) + write x_rot (K*4)
        let bytes = k * 4 * 3 + 2 * 256 * 4;
        let timer = crate::profile::begin_timer(&self.hip, "fused", timer_name, bytes);
        let result = self.launch_kernargs(
            kernel_name,
            [1, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &kernargs![ptr xp, ptr wp, ptr s1, ptr s2, ptr xrp, i32 kv, f32 eps_v],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }
    /// Phase A Stage A — AWQ-aware variant of fused_rmsnorm_rotate_mq.
    ///
    /// After computing the RMSNorm output, divides element-wise by
    /// `awq_scale[i]` BEFORE the FWHT rotation. Completes the AWQ math
    /// `(W·s) · (x/s) = W·x` where W·s is baked at quantize time.
    ///
    /// Use when the upcoming linear layer's WeightTensor carries
    /// `awq_scale = Some(...)`; otherwise call the non-AWQ variant.
    ///
    /// awq_scale: 1D FP32 GpuTensor of length K (host-side F16 → F32
    /// conversion happens in the loader; see hfq.rs::load_awq_scale).
    ///
    /// Backward-compatible: kernel is separate, no behavioral change for
    /// the standard fused_rmsnorm_rotate_mq path.
    pub fn fused_rmsnorm_rotate_mq_awq(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_rmsnorm_mq_rotate_awq",
            kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_SRC,
            "fused_rmsnorm_mq_rotate_awq",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let xp = x.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let awp = awq_scale.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let eps_v = eps;

        let block_size = 256u32;
        let shared_mem = ((k + 256) * 4) as u32;
        // Bandwidth: read x + weight + awq_scale + signs + write x_rot.
        let bytes = k * 4 * 4 + 2 * 256 * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_rmsnorm_mq_rotate_awq", bytes);
        let result = self.launch_kernargs(
            "fused_rmsnorm_mq_rotate_awq",
            [1, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &kernargs![ptr xp, ptr wp, ptr awp, ptr s1, ptr s2, ptr xrp, i32 kv, f32 eps_v],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }
    /// Batched `fused_rmsnorm_rotate_mq`. Grid.x is the batch dim — processes
    /// N tokens' [N × K] x into [N × K] x_rot in a single launch. Byte-exact
    /// against calling `fused_rmsnorm_rotate_mq` N times on separate x/x_rot
    /// buffers. Weight/signs are shared across the batch.
    /// Phase A Stage A — batched AWQ variant. Mirrors
    /// fused_rmsnorm_rotate_mq_batched but takes an additional
    /// `awq_scale: &GpuTensor` (length K, FP32) and dispatches the
    /// AWQ kernel. Caller selects based on the upcoming linear
    /// layer's WeightTensor.awq_scale being Some.
    pub fn fused_rmsnorm_rotate_mq_awq_batched(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_rmsnorm_mq_rotate_awq",
            kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_SRC,
            "fused_rmsnorm_mq_rotate_awq",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let xp = x.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let awp = awq_scale.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let eps_v = eps;
        let block_size = 256u32;
        let shared_mem = ((k + 256) * 4) as u32;
        let bytes = (k * 4 * 4 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_rmsnorm_mq_rotate_awq_batched",
            bytes,
        );
        let result = self.launch_kernargs(
            "fused_rmsnorm_mq_rotate_awq",
            [batch_size as u32, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &kernargs![ptr xp, ptr wp, ptr awp, ptr s1, ptr s2, ptr xrp, i32 kv, f32 eps_v],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }
    pub fn fused_rmsnorm_rotate_mq_batched(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx94x split — see fused_rmsnorm_rotate_mq docstring.
        if self.flags.gfx942_rmsnorm_split {
            return self.fused_rmsnorm_rotate_mq_split_gfx942(x, weight, x_rot, k, eps, batch_size);
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        let (module_name, kernel_src, kernel_name, timer_name, reduce_slots) =
            if self.arch_caps.is_gfx1151() {
                (
                    "fused_rmsnorm_mq_rotate_gfx1151",
                    kernels::FUSED_RMSNORM_MQ_ROTATE_GFX1151_SRC,
                    "fused_rmsnorm_mq_rotate_gfx1151",
                    "fused_rmsnorm_mq_rotate_gfx1151_batched",
                    8usize,
                )
            } else {
                (
                    "fused_rmsnorm_mq_rotate",
                    kernels::FUSED_RMSNORM_MQ_ROTATE_SRC,
                    "fused_rmsnorm_mq_rotate",
                    "fused_rmsnorm_mq_rotate_batched",
                    256usize,
                )
            };
        self.ensure_kernel(module_name, kernel_src, kernel_name)?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let xp = x.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let eps_v = eps;
        let block_size = 256u32;
        let shared_mem = ((k + reduce_slots) * 4) as u32;
        let bytes = (k * 4 * 3 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fused", timer_name, bytes);
        let result = self.launch_kernargs(
            kernel_name,
            [batch_size as u32, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &kernargs![ptr xp, ptr wp, ptr s1, ptr s2, ptr xrp, i32 kv, f32 eps_v],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }
    /// Fused SwiGLU + FWHT rotation. Reads gate/up, computes
    /// silu(gate[k])*up[k] on the fly, applies FWHT rotation, writes x_rot.
    /// Used as the w_down input stage for MQ4 — replaces the pair
    /// silu_mul_f32 + mq_rotate_x with one launch.
    pub fn fused_silu_mul_rotate_mq(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_silu_mul_mq_rotate",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_SRC,
            "fused_silu_mul_mq_rotate",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let gp = gate.buf.as_ptr();
        let up_p = up.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        // Bandwidth: read gate + up, 2x256 signs, write x_rot.
        let bytes = k * 4 * 3 + 2 * 256 * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_silu_mul_mq_rotate", bytes);
        let result = self.launch_kernargs(
            "fused_silu_mul_mq_rotate",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr gp, ptr up_p, ptr s1, ptr s2, ptr xrp, i32 kv],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }
    /// Batched `fused_silu_mul_rotate_mq`. Grid.y is the batch dim — processes
    /// N tokens' [N × K] gate/up/x_rot in a single launch.
    pub fn fused_silu_mul_rotate_mq_batched(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_silu_mul_mq_rotate",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_SRC,
            "fused_silu_mul_mq_rotate",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let gp = gate.buf.as_ptr();
        let up_p = up.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let bytes = (k * 4 * 3 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_silu_mul_mq_rotate_batched",
            bytes,
        );
        let result = self.launch_kernargs(
            "fused_silu_mul_mq_rotate",
            [n_groups, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr gp, ptr up_p, ptr s1, ptr s2, ptr xrp, i32 kv],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }
    /// Phase A Stage A — F2 AWQ-aware variant of `fused_silu_mul_rotate_mq`.
    ///
    /// After computing silu(gate)*up, divides element-wise by `awq_scale[i]`
    /// BEFORE the FWHT rotation. Completes the AWQ math
    /// `(W·s) · (silu(g)*u / s) = W·silu(g)*u` where W·s is baked at
    /// quantize time for the down_proj / w_down weights.
    ///
    /// Use when the down_proj `WeightTensor` carries `awq_scale = Some(...)`;
    /// otherwise call the non-AWQ variant.
    ///
    /// awq_scale: 1D FP32 GpuTensor of length K (host-side F16 → F32
    /// conversion happens in the loader; see hfq.rs::load_awq_scale).
    pub fn fused_silu_mul_rotate_mq_awq(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_silu_mul_mq_rotate_awq",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_SRC,
            "fused_silu_mul_mq_rotate_awq",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let gp = gate.buf.as_ptr();
        let up_p = up.buf.as_ptr();
        let awp = awq_scale.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        // Bandwidth: read gate + up + awq_scale, 2x256 signs, write x_rot.
        let bytes = k * 4 * 4 + 2 * 256 * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_silu_mul_mq_rotate_awq", bytes);
        let result = self.launch_kernargs(
            "fused_silu_mul_mq_rotate_awq",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr gp, ptr up_p, ptr awp, ptr s1, ptr s2, ptr xrp, i32 kv],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }
    /// Phase A Stage A — F2 batched AWQ variant of `fused_silu_mul_rotate_mq`.
    /// Grid.y is the batch dim — processes [N × K] gate/up/x_rot.
    pub fn fused_silu_mul_rotate_mq_awq_batched(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_silu_mul_mq_rotate_awq",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_SRC,
            "fused_silu_mul_mq_rotate_awq",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let gp = gate.buf.as_ptr();
        let up_p = up.buf.as_ptr();
        let awp = awq_scale.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let bytes = (k * 4 * 4 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_silu_mul_mq_rotate_awq_batched",
            bytes,
        );
        let result = self.launch_kernargs(
            "fused_silu_mul_mq_rotate_awq",
            [n_groups, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr gp, ptr up_p, ptr awp, ptr s1, ptr s2, ptr xrp, i32 kv],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }
    /// dp4a-port of fused_qkv_hfq4g256 for gfx906. Pre-quantizes x to
    /// Q8_1 via the shared MMQ scratch, then runs the dp4a-based GEMV.
    /// Math is identical modulo Q8_1 quant noise. Targets gfx906's
    /// memory-bound regime per the per-kernel PMC pass at 2026-05-05.
    pub fn fused_qkv_hfq4g256_dp4a(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_qkv_hfq4g256_wave64_dp4a",
            kernels::FUSED_QKV_HFQ4G256_WAVE64_DP4A_SRC,
            "fused_qkv_hfq4g256_wave64_dp4a",
        )?;

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_val = q_m as i32;
        let k_m_val = k_m as i32;
        let v_m_val = v_m as i32;
        let k_val = k as i32;
        let total = (q_m + k_m + v_m) as u32;
        let xq = xq_ptr;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k);
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_qkv_hfq4g256_dp4a", bytes);
        let result = self.launch_kernargs(
            "fused_qkv_hfq4g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1],
            [64, 1, 1],
            0,
            &kernargs![ptr aq, ptr ak, ptr av, ptr xq, ptr yq, ptr yk, ptr yv, i32 q_m_val, i32 k_m_val, i32 v_m_val, i32 k_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx906 dp4a-port — see fused_gate_up_hfq6g256_wave64_dp4a.hip for
    /// the math derivation. Plan §3.1.1 item 3 / v3.2.2 §5.1 item 1c.
    pub fn fused_qkv_hfq6g256_dp4a(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_qkv_hfq6g256_wave64_dp4a",
            kernels::FUSED_QKV_HFQ6G256_WAVE64_DP4A_SRC,
            "fused_qkv_hfq6g256_wave64_dp4a",
        )?;

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_val = q_m as i32;
        let k_m_val = k_m as i32;
        let v_m_val = v_m as i32;
        let k_val = k as i32;
        let total = (q_m + k_m + v_m) as u32;
        let xq = xq_ptr;

        self.launch_kernargs(
            "fused_qkv_hfq6g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1],
            [64, 1, 1],
            0,
            &kernargs![ptr aq, ptr ak, ptr av, ptr xq, ptr yq, ptr yk, ptr yv, i32 q_m_val, i32 k_m_val, i32 v_m_val, i32 k_val],
        )
    }
    /// gfx906 dp4a-port — 4-output deltanet QKV preamble.
    pub fn fused_qkvza_hfq6g256_dp4a(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_qkvza_hfq6g256_wave64_dp4a",
            kernels::FUSED_QKVZA_HFQ6G256_WAVE64_DP4A_SRC,
            "fused_qkvza_hfq6g256_wave64_dp4a",
        )?;

        let aqkv = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let yqkv = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let qkv_m_val = qkv_m as i32;
        let z_m_val = z_m as i32;
        let beta_m_val = beta_m as i32;
        let alpha_m_val = alpha_m as i32;
        let k_val = k as i32;
        let total = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let xq = xq_ptr;

        self.launch_kernargs(
            "fused_qkvza_hfq6g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1],
            [64, 1, 1],
            0,
            &kernargs![ptr aqkv, ptr az, ptr ab, ptr aa, ptr xq, ptr yqkv, ptr yz, ptr yb, ptr ya, i32 qkv_m_val, i32 z_m_val, i32 beta_m_val, i32 alpha_m_val, i32 k_val],
        )
    }
    /// gfx906 dp4a-port — 2-output FFN gate+up projection.
    pub fn fused_gate_up_hfq6g256_dp4a(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_gate_up_hfq6g256_wave64_dp4a",
            kernels::FUSED_GATE_UP_HFQ6G256_WAVE64_DP4A_SRC,
            "fused_gate_up_hfq6g256_wave64_dp4a",
        )?;

        let agate = a_gate.buf.as_ptr();
        let aup = a_up.buf.as_ptr();
        let ygate = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let gate_m_val = gate_m as i32;
        let up_m_val = up_m as i32;
        let k_val = k as i32;
        let total = (gate_m + up_m) as u32;
        let xq = xq_ptr;

        self.launch_kernargs(
            "fused_gate_up_hfq6g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1],
            [64, 1, 1],
            0,
            &kernargs![ptr agate, ptr aup, ptr xq, ptr ygate, ptr yup, i32 gate_m_val, i32 up_m_val, i32 k_val],
        )
    }
    /// 3-way fused HFQ4-G256 projection — cross-arch.
    ///
    /// Performs y_q=A_q·x, y_k=A_k·x, y_v=A_v·x in a single kernel launch
    /// for the Qwen3.5 FullAttention layer preamble. Same rationale and
    /// tail-handling guarantees as `fused_qkvza_hfq4g256`.
    pub fn fused_qkv_hfq4g256(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.gemv_dp4a_enabled() {
            return self.fused_qkv_hfq4g256_dp4a(a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k);
        }

        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let gfx1151_2row = self.gfx1151_fused_hfq4_2row_enabled();
        let (func_name, block, grid_x) = if cdna_wave64 {
            // gfx94x v2: 2 wave64s = 4 rows/WG, +1.9% on AR decode
            // (commit 5bd75a69 sibling). Default ON; opt out via
            // HIPFIRE_GFX942_GEMV_V2=0.
            let is_gfx94x = self.arch_caps.is_cdna3();
            let v2_on = self.flags.gfx942_gemv_v2.unwrap_or(true);
            if is_gfx94x && v2_on {
                self.ensure_kernel(
                    "fused_qkv_hfq4g256_v2_gfx942",
                    kernels::FUSED_QKV_HFQ4G256_V2_GFX942_SRC,
                    "fused_qkv_hfq4g256_v2_gfx942",
                )?;
                let total = (q_m + k_m + v_m) as u32;
                (
                    "fused_qkv_hfq4g256_v2_gfx942",
                    [128u32, 1, 1],
                    (total + 3) / 4,
                )
            } else {
                self.ensure_kernel(
                    "fused_qkv_hfq4g256_wave64",
                    kernels::FUSED_QKV_HFQ4G256_WAVE64_SRC,
                    "fused_qkv_hfq4g256_wave64",
                )?;
                let total = (q_m + k_m + v_m) as u32;
                ("fused_qkv_hfq4g256_wave64", [64u32, 1, 1], (total + 1) / 2)
            }
        } else if gfx1151_2row {
            self.ensure_kernel(
                "fused_qkv_hfq4g256_wave64",
                kernels::FUSED_QKV_HFQ4G256_WAVE64_SRC,
                "fused_qkv_hfq4g256_wave64",
            )?;
            let total = (q_m + k_m + v_m) as u32;
            ("fused_qkv_hfq4g256_wave64", [64u32, 1, 1], (total + 1) / 2)
        } else {
            self.ensure_kernel(
                "fused_qkv_hfq4g256",
                kernels::FUSED_QKV_HFQ4G256_SRC,
                "fused_qkv_hfq4g256",
            )?;
            (
                "fused_qkv_hfq4g256",
                [32u32, 1, 1],
                (q_m + k_m + v_m) as u32,
            )
        };

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_val = q_m as i32;
        let k_m_val = k_m as i32;
        let v_m_val = v_m as i32;
        let k_val = k as i32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qkv_hfq4g256", bytes);
        let result = self.launch_kernargs(
            func_name,
            [grid_x, 1, 1],
            block,
            0,
            &kernargs![
                ptr aq, ptr ak, ptr av, ptr xp, ptr yq, ptr yk, ptr yv,
                i32 q_m_val, i32 k_m_val, i32 v_m_val, i32 k_val
            ],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// 4-way fused HFQ4-G256 projection — cross-arch.
    ///
    /// Performs y_qkv=A_qkv·x, y_z=A_z·x, y_beta=A_beta·x, y_alpha=A_alpha·x
    /// in a single kernel launch, where all four matrices share the same
    /// input `x` and the same K. Used by the Qwen3.5 DeltaNet LA layer
    /// preamble to collapse four launches (one per projection) into one.
    /// Bit-exact with four sequential `gemv_hfq4g256` calls.
    ///
    /// Works on every RDNA generation (gfx1010 / gfx1013 / gfx1030 /
    /// gfx1100+) because the inner loop and the standalone gemv_hfq4g256
    /// inner loop were unified onto the same 4-accumulator structure
    /// after commit 5302926.
    pub fn fused_qkvza_hfq4g256(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.gemv_dp4a_enabled() {
            return self.fused_qkvza_hfq4g256_dp4a(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k,
            );
        }
        // gfx906/gfx908/gfx94x wave64-native path:
        // 2 rows per block, halves grid count vs wave32 kernel which wastes half
        // the wave slot. This kernel uses no MFMA, just FMA + shfl_down within
        // wave64, so it is safe for Vega 20 as well as CDNA.
        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let gfx1151_2row = self.gfx1151_fused_hfq4_2row_enabled();
        let (func_name, block, grid_x) = if cdna_wave64 {
            // gfx94x v2: 2 wave64s = 4 rows/WG, +1.9% on AR decode
            // (commit 5bd75a69 sibling). Default ON; opt out via
            // HIPFIRE_GFX942_GEMV_V2=0.
            let is_gfx94x = self.arch_caps.is_cdna3();
            let v2_on = self.flags.gfx942_gemv_v2.unwrap_or(true);
            if is_gfx94x && v2_on {
                self.ensure_kernel(
                    "fused_qkvza_hfq4g256_v2_gfx942",
                    kernels::FUSED_QKVZA_HFQ4G256_V2_GFX942_SRC,
                    "fused_qkvza_hfq4g256_v2_gfx942",
                )?;
                let total = (qkv_m + z_m + beta_m + alpha_m) as u32;
                (
                    "fused_qkvza_hfq4g256_v2_gfx942",
                    [128u32, 1, 1],
                    (total + 3) / 4,
                )
            } else {
                self.ensure_kernel(
                    "fused_qkvza_hfq4g256_wave64",
                    kernels::FUSED_QKVZA_HFQ4G256_WAVE64_SRC,
                    "fused_qkvza_hfq4g256_wave64",
                )?;
                let total = (qkv_m + z_m + beta_m + alpha_m) as u32;
                (
                    "fused_qkvza_hfq4g256_wave64",
                    [64u32, 1, 1],
                    (total + 1) / 2,
                )
            }
        } else if gfx1151_2row {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_wave64",
                kernels::FUSED_QKVZA_HFQ4G256_WAVE64_SRC,
                "fused_qkvza_hfq4g256_wave64",
            )?;
            let total = (qkv_m + z_m + beta_m + alpha_m) as u32;
            (
                "fused_qkvza_hfq4g256_wave64",
                [64u32, 1, 1],
                (total + 1) / 2,
            )
        } else {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256",
                kernels::FUSED_QKVZA_HFQ4G256_SRC,
                "fused_qkvza_hfq4g256",
            )?;
            (
                "fused_qkvza_hfq4g256",
                [32u32, 1, 1],
                (qkv_m + z_m + beta_m + alpha_m) as u32,
            )
        };
        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let q_m_i = qkv_m as i32;
        let z_m_i = z_m as i32;
        let b_m_i = beta_m as i32;
        let a_m_i = alpha_m as i32;
        let k_i = k as i32;

        let grid = [grid_x, 1, 1];

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qkvza_hfq4g256", bytes);

        let result = self.launch_kernargs(
            func_name,
            grid,
            block,
            0,
            &kernargs![
                ptr aq, ptr az, ptr ab, ptr aa, ptr xp, ptr yq, ptr yz, ptr yb, ptr ya,
                i32 q_m_i, i32 z_m_i, i32 b_m_i, i32 a_m_i, i32 k_i
            ],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// dp4a-port of fused_qkvza_hfq4g256 for gfx906. Pre-quantizes x to
    /// Q8_1 via the shared MMQ scratch, then runs the dp4a-based GEMV.
    /// Math is identical modulo Q8_1 quant noise. Targets gfx906's
    /// memory-bound regime per the per-kernel PMC pass at 2026-05-05.
    pub fn fused_qkvza_hfq4g256_dp4a(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_qkvza_hfq4g256_wave64_dp4a",
            kernels::FUSED_QKVZA_HFQ4G256_WAVE64_DP4A_SRC,
            "fused_qkvza_hfq4g256_wave64_dp4a",
        )?;

        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let q_m_i = qkv_m as i32;
        let z_m_i = z_m as i32;
        let b_m_i = beta_m as i32;
        let a_m_i = alpha_m as i32;
        let k_i = k as i32;
        let total = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let xq = xq_ptr;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k);
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_qkvza_hfq4g256_dp4a", bytes);

        let result = self.launch_kernargs(
            "fused_qkvza_hfq4g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1],
            [64, 1, 1],
            0,
            &kernargs![ptr aq, ptr az, ptr ab, ptr aa, ptr xq, ptr yq, ptr yz, ptr yb, ptr ya, i32 q_m_i, i32 z_m_i, i32 b_m_i, i32 a_m_i, i32 k_i],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Fused QKV: three Q4_K GEMVs in one launch (saves 2 kernel launches per layer).
    /// q = Wq * x, k = Wk * x, v = Wv * x — all read the same input x.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_q4k(
        &mut self,
        wq: &GpuTensor,
        wk: &GpuTensor,
        wv: &GpuTensor,
        x: &GpuTensor,
        yq: &GpuTensor,
        yk: &GpuTensor,
        yv: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("fused_qkv_q4k", kernels::FUSED_QKV_Q4K_SRC, "fused_qkv_q4k")?;
        let func = &self.functions["fused_qkv_q4k"];

        let mut aq = wq.buf.as_ptr();
        let mut ak = wk.buf.as_ptr();
        let mut av = wv.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yqp = yq.buf.as_ptr();
        let mut ykp = yk.buf.as_ptr();
        let mut yvp = yv.buf.as_ptr();
        let mut qm = q_m as i32;
        let mut km = k_m as i32;
        let mut vm = v_m as i32;
        let mut kk = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yqp as *mut _ as *mut c_void,
            &mut ykp as *mut _ as *mut c_void,
            &mut yvp as *mut _ as *mut c_void,
            &mut qm as *mut _ as *mut c_void,
            &mut km as *mut _ as *mut c_void,
            &mut vm as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];

        let grid = (q_m + k_m + v_m) as u32;
        unsafe {
            self.hip
                .launch_kernel(func, [grid, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }
    /// Fused Gate+Up: two Q4_K GEMVs in one launch (saves 1 kernel launch per layer).
    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_q4k(
        &mut self,
        w_gate: &GpuTensor,
        w_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_gate_up_q4k",
            kernels::FUSED_GATE_UP_Q4K_SRC,
            "fused_gate_up_q4k",
        )?;
        let func = &self.functions["fused_gate_up_q4k"];

        let mut ag = w_gate.buf.as_ptr();
        let mut au = w_up.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut ygp = y_gate.buf.as_ptr();
        let mut yup = y_up.buf.as_ptr();
        let mut gm = gate_m as i32;
        let mut um = up_m as i32;
        let mut kk = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut ygp as *mut _ as *mut c_void,
            &mut yup as *mut _ as *mut c_void,
            &mut gm as *mut _ as *mut c_void,
            &mut um as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];

        let grid = (gate_m + up_m) as u32;
        unsafe {
            self.hip
                .launch_kernel(func, [grid, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }
    /// Fused Gate+Up HFQ4-G256: two GEMVs in one launch.
    pub fn fused_gate_up_hfq4g256(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx906 dp4a opt-in: pre-quantize x to Q8_1 and use the
        // v_dot4_i32_i8 path. PMC at 2026-05-05 showed this kernel
        // was memory-bound; dp4a's 75% x-traffic reduction lands on
        // the actual bottleneck.
        if self.arch_caps.gemv_dp4a_enabled() {
            return self
                .fused_gate_up_hfq4g256_dp4a(a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k);
        }

        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_x) = if cdna_wave64 {
            // gfx94x v2: 2 wave64s = 4 rows/WG, +1.9% on AR decode
            // (commit 5bd75a69 sibling). Default ON; opt out via
            // HIPFIRE_GFX942_GEMV_V2=0.
            let is_gfx94x = self.arch_caps.is_cdna3();
            let v2_on = self.flags.gfx942_gemv_v2.unwrap_or(true);
            if is_gfx94x && v2_on {
                self.ensure_kernel(
                    "fused_gate_up_hfq4g256_v2_gfx942",
                    kernels::FUSED_GATE_UP_HFQ4G256_V2_GFX942_SRC,
                    "fused_gate_up_hfq4g256_v2_gfx942",
                )?;
                let total = (gate_m + up_m) as u32;
                (
                    "fused_gate_up_hfq4g256_v2_gfx942",
                    [128u32, 1, 1],
                    (total + 3) / 4,
                )
            } else {
                self.ensure_kernel(
                    "fused_gate_up_hfq4g256_wave64",
                    kernels::FUSED_GATE_UP_HFQ4G256_WAVE64_SRC,
                    "fused_gate_up_hfq4g256_wave64",
                )?;
                let total = (gate_m + up_m) as u32;
                (
                    "fused_gate_up_hfq4g256_wave64",
                    [64u32, 1, 1],
                    (total + 1) / 2,
                )
            }
        } else {
            self.ensure_kernel(
                "fused_gate_up_hfq4g256",
                kernels::FUSED_GATE_UP_HFQ4G256_SRC,
                "fused_gate_up_hfq4g256",
            )?;
            (
                "fused_gate_up_hfq4g256",
                [32u32, 1, 1],
                (gate_m + up_m) as u32,
            )
        };
        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let gm = gate_m as i32;
        let um = up_m as i32;
        let kv = k as i32;
        self.launch_kernargs(
            func_name,
            [grid_x, 1, 1],
            block,
            0,
            &kernargs![ptr ag, ptr au, ptr xp, ptr yg, ptr yu, i32 gm, i32 um, i32 kv],
        )
    }
    /// dp4a-port of fused_gate_up_hfq4g256 for gfx906. Pre-quantizes
    /// `x` to Q8_1 (block_q8_1_mmq, 144 B per 128-K block) using the
    /// shared MMQ x-scratch buffer, then runs the dp4a-based GEMV. Math
    /// is identical modulo Q8_1 quant noise (~1 % per-element relative).
    /// Targeted at gfx906 where the FP wave64 fused_gate_up sat at
    /// 41 % VALUBusy + 3.86 % MemUnitStalled — memory-bound, so dp4a's
    /// 75 % x-traffic reduction lands on the actual bottleneck.
    pub fn fused_gate_up_hfq4g256_dp4a(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Quantize x → Xq[K/128] block_q8_1_mmq via the existing shared
        // scratch path. Batch=1 for GEMV.
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_gate_up_hfq4g256_wave64_dp4a",
            kernels::FUSED_GATE_UP_HFQ4G256_WAVE64_DP4A_SRC,
            "fused_gate_up_hfq4g256_wave64_dp4a",
        )?;

        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let gm = gate_m as i32;
        let um = up_m as i32;
        let kv = k as i32;
        let total = (gate_m + up_m) as u32;
        let xq = xq_ptr;
        self.launch_kernargs(
            "fused_gate_up_hfq4g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1],
            [64, 1, 1],
            0,
            &kernargs![ptr ag, ptr au, ptr xq, ptr yg, ptr yu, i32 gm, i32 um, i32 kv],
        )
    }
    /// Fused sigmoid(dn_beta) + alpha_gate(dn_alpha). Both ops are element-wise
    /// scalar transforms applied to independent buffers of size n_v_heads in the
    /// DeltaNet preamble. Saves one launch per linear-attention layer.
    #[cfg(feature = "deltanet")]
    pub fn fused_sigmoid_alpha_gate_f32(
        &mut self,
        beta: &GpuTensor,
        alpha: &GpuTensor,
        dt_bias: &GpuTensor,
        a_log: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_sigmoid_alpha_gate",
            kernels::FUSED_SIGMOID_ALPHA_GATE_SRC,
            "fused_sigmoid_alpha_gate_f32",
        )?;
        let bp = beta.buf.as_ptr();
        let ap = alpha.buf.as_ptr();
        let dp = dt_bias.buf.as_ptr();
        let lp = a_log.buf.as_ptr();
        let nn = n as i32;
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = n * 4 * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_sigmoid_alpha_gate_f32", bytes);
        let result = self.launch_kernargs(
            "fused_sigmoid_alpha_gate_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &kernargs![ptr bp, ptr ap, ptr dp, ptr lp, i32 nn],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched `fused_sigmoid_alpha_gate_f32`. Grid.y is the batch dim.
    #[cfg(feature = "deltanet")]
    pub fn fused_sigmoid_alpha_gate_f32_batched(
        &mut self,
        beta: &GpuTensor,
        alpha: &GpuTensor,
        dt_bias: &GpuTensor,
        a_log: &GpuTensor,
        n: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_sigmoid_alpha_gate",
            kernels::FUSED_SIGMOID_ALPHA_GATE_SRC,
            "fused_sigmoid_alpha_gate_f32",
        )?;
        let bp = beta.buf.as_ptr();
        let ap = alpha.buf.as_ptr();
        let dp = dt_bias.buf.as_ptr();
        let lp = a_log.buf.as_ptr();
        let nn = n as i32;
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = n * 4 * 4 * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_sigmoid_alpha_gate_f32_batched",
            bytes,
        );
        let result = self.launch_kernargs(
            "fused_sigmoid_alpha_gate_f32",
            [grid, batch_size as u32, 1],
            [block, 1, 1],
            0,
            &kernargs![ptr bp, ptr ap, ptr dp, ptr lp, i32 nn],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// gfx1151 single-token DeltaNet decode fusion. On other arches this
    /// preserves the existing two-launch sequence.
    #[cfg(feature = "deltanet")]
    #[allow(clippy::too_many_arguments)]
    pub fn fused_sigmoid_alpha_gate_conv1d_silu_split_f32(
        &mut self,
        beta: &GpuTensor,
        alpha: &GpuTensor,
        dt_bias: &GpuTensor,
        a_log: &GpuTensor,
        q_out: &GpuTensor,
        k_out: &GpuTensor,
        v_out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        state: &GpuTensor,
        n_heads: usize,
        k_dim: usize,
        v_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1151() {
            self.fused_sigmoid_alpha_gate_f32(beta, alpha, dt_bias, a_log, n_heads)?;
            return self
                .conv1d_silu_split_f32(q_out, k_out, v_out, input, weight, state, k_dim, v_dim);
        }

        let kernel_name = "fused_sigmoid_alpha_gate_conv1d_silu_split_f32_gfx1151";
        self.ensure_kernel(
            "fused_sigmoid_alpha_gate_conv1d_silu_split_gfx1151",
            kernels::FUSED_SIGMOID_ALPHA_GATE_CONV1D_SILU_SPLIT_GFX1151_SRC,
            kernel_name,
        )?;

        let bp = beta.buf.as_ptr();
        let ap = alpha.buf.as_ptr();
        let dp = dt_bias.buf.as_ptr();
        let lp = a_log.buf.as_ptr();
        let qp = q_out.buf.as_ptr();
        let kp = k_out.buf.as_ptr();
        let vp = v_out.buf.as_ptr();
        let ip = input.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let sp = state.buf.as_ptr();
        let nh = n_heads as i32;
        let kd = k_dim as i32;
        let vd = v_dim as i32;

        let n_channels = 2 * k_dim + v_dim;
        let elems = n_channels.max(n_heads);
        let block = 256u32;
        let grid = ((elems as u32) + block - 1) / block;
        let bytes = n_heads * 4 * 4 + crate::profile::conv1d_silu_bytes(n_channels);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "fused_sigmoid_alpha_gate_conv1d_silu_split_f32_gfx1151",
            bytes,
        );
        let result = self.launch_kernargs(
            kernel_name,
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &kernargs![ptr bp, ptr ap, ptr dp, ptr lp, ptr qp, ptr kp, ptr vp, ptr ip, ptr wp, ptr sp, i32 nh, i32 kd, i32 vd],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Fused L2-norm(Q) + L2-norm(K) + scale(Q). Replaces three back-to-back
    /// launches in DeltaNet's attention path with one — ~2 launches saved per
    /// linear-attention layer, so on Qwen3.5 (18-32 LA layers) we shave ~36-64
    /// launches per forward.
    #[cfg(feature = "deltanet")]
    pub fn fused_qk_l2_norm_scale_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        q_scale: f32,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_qk_l2_norm_scale",
            kernels::FUSED_QK_L2_NORM_SCALE_SRC,
            "fused_qk_l2_norm_scale_f32",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let qs = q_scale;
        let ep = eps;
        // Covers both Q and K reads/writes.
        let bytes = crate::profile::elementwise1_bytes(n_heads * head_dim) * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_qk_l2_norm_scale_f32", bytes);
        let result = self.launch_kernargs(
            "fused_qk_l2_norm_scale_f32",
            [n_heads as u32, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr qp, ptr kp, i32 nh, i32 hd, f32 qs, f32 ep],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched `fused_qk_l2_norm_scale_f32`. Grid.y is the batch dim.
    #[cfg(feature = "deltanet")]
    pub fn fused_qk_l2_norm_scale_f32_batched(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        q_scale: f32,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_qk_l2_norm_scale",
            kernels::FUSED_QK_L2_NORM_SCALE_SRC,
            "fused_qk_l2_norm_scale_f32",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let qs = q_scale;
        let ep = eps;
        let bytes = crate::profile::elementwise1_bytes(n_heads * head_dim) * 2 * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_qk_l2_norm_scale_f32_batched",
            bytes,
        );
        let result = self.launch_kernargs(
            "fused_qk_l2_norm_scale_f32",
            [n_heads as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr qp, ptr kp, i32 nh, i32 hd, f32 qs, f32 ep],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Fused L2-norm(Q) + scale(Q) + L2-norm(K) + repeat-interleave(Q,K).
    /// Replaces fused_qk_l2_norm_scale_f32_batched +
    /// repeat_interleave_qk_f32_batched (2 launches → 1). Each block
    /// (key_head, batch) computes norms once and replicates across the
    /// `ratio` value-head slots. Used only when n_key_heads < n_v_heads.
    ///
    /// `q_src`/`k_src`: [N × n_key_heads × head_dim] (unchanged on exit).
    /// `q_dst`/`k_dst`: [N × n_value_heads × head_dim] (n_value = n_key*ratio).
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qk_l2_norm_scale_interleave_f32_batched(
        &mut self,
        q_src: &GpuTensor,
        k_src: &GpuTensor,
        q_dst: &GpuTensor,
        k_dst: &GpuTensor,
        n_key_heads: usize,
        ratio: usize,
        head_dim: usize,
        q_scale: f32,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_qk_l2_norm_scale_interleave_f32_batched",
            kernels::FUSED_QK_L2_NORM_SCALE_INTERLEAVE_F32_BATCHED_SRC,
            "fused_qk_l2_norm_scale_interleave_f32_batched",
        )?;
        let qsp = q_src.buf.as_ptr();
        let ksp = k_src.buf.as_ptr();
        let qdp = q_dst.buf.as_ptr();
        let kdp = k_dst.buf.as_ptr();
        let nkh = n_key_heads as i32;
        let r_val = ratio as i32;
        let hd = head_dim as i32;
        let qs = q_scale;
        let ep = eps;
        let bytes =
            crate::profile::elementwise1_bytes(n_key_heads * ratio * head_dim) * 2 * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_qk_l2_norm_scale_interleave_f32_batched",
            bytes,
        );
        let result = self.launch_kernargs(
            "fused_qk_l2_norm_scale_interleave_f32_batched",
            [n_key_heads as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr qsp, ptr ksp, ptr qdp, ptr kdp, i32 nkh, i32 r_val, i32 hd, f32 qs, f32 ep],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_q8_0(
        &mut self,
        _w_gate: &GpuTensor,
        _w_up: &GpuTensor,
        _x: &GpuTensor,
        _gate: &GpuTensor,
        _up: &GpuTensor,
        _m_gate: usize,
        _m_up: usize,
        _k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — unimplemented stub (no GPU work; returns Err)
        Err(hip_bridge::HipError::new(801, "not yet implemented"))
    }
    /// Strict superset of `fused_rmsnorm_rotate_mq`: also writes the plain
    /// (non-FWHT) RMSNormed output to `x_plain`. Saves the follow-up
    /// `rmsnorm_f32` launch on DeepSeek V4 decode FFN paths that consume both
    /// representations (MQ4 GEMV reads x_rot, Q8/F16 GEMV reads x_plain).
    pub fn fused_rmsnorm_rotate_mq_plain(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        x_plain: &GpuTensor,
        k: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_rmsnorm_mq_rotate_plain",
            kernels::FUSED_RMSNORM_MQ_ROTATE_PLAIN_SRC,
            "fused_rmsnorm_mq_rotate_plain",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let xp = x.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let xpp = x_plain.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let eps_v = eps;

        let block_size = 256u32;
        let shared_mem = ((k + 256) * 4) as u32;
        let bytes = k * 4 * 4 + 2 * 256 * 4; // +1 K*4 for x_plain write
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_rmsnorm_mq_rotate_plain", bytes);
        let result = self.launch_kernargs(
            "fused_rmsnorm_mq_rotate_plain",
            [1, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &kernargs![ptr xp, ptr wp, ptr s1, ptr s2, ptr xrp, ptr xpp, i32 kv, f32 eps_v],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        self.invalidate_x_caches_for(xpp);
        result
    }
    /// Batched twin of `fused_rmsnorm_rotate_mq_plain`. Grid.x = batch_size.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_rmsnorm_rotate_mq_plain_batched(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        x_plain: &GpuTensor,
        k: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_rmsnorm_mq_rotate_plain",
            kernels::FUSED_RMSNORM_MQ_ROTATE_PLAIN_SRC,
            "fused_rmsnorm_mq_rotate_plain",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let xp = x.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let xpp = x_plain.buf.as_ptr();
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let eps_v = eps;
        let block_size = 256u32;
        let shared_mem = ((k + 256) * 4) as u32;
        let bytes = (k * 4 * 4 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_rmsnorm_mq_rotate_plain_batched",
            bytes,
        );
        let result = self.launch_kernargs(
            "fused_rmsnorm_mq_rotate_plain",
            [batch_size as u32, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &kernargs![ptr xp, ptr wp, ptr s1, ptr s2, ptr xrp, ptr xpp, i32 kv, f32 eps_v],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        self.invalidate_x_caches_for(xpp);
        result
    }
    /// Opus Quant W4A4 fused Gate+Up: gate_proj + up_proj grouped-iu4 GEMMs in
    /// one launch sharing the int4 activation (`xq`/`xs`). Each weight buffer is
    /// the combined `[nibbles | f32 scales]` layout (scales addressed internally
    /// at offset M*(K/2)). Outputs `y_gate` [B,gate_m], `y_up` [B,up_m].
    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_oq4_wmma(
        &mut self,
        w_gate: &GpuTensor,
        w_up: &GpuTensor,
        xq: &GpuTensor,
        xs: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            k % group,
            0,
            "fused_gate_up_oq4_wmma: K must be a multiple of group"
        );
        assert_eq!(
            group % 16,
            0,
            "fused_gate_up_oq4_wmma: group must be a multiple of 16"
        );
        self.ensure_kernel(
            "fused_gate_up_oq4_wmma",
            kernels::FUSED_GATE_UP_OQ4_WMMA_SRC,
            "fused_gate_up_oq4_wmma",
        )?;
        let wgp = w_gate.buf.as_ptr();
        let wup = w_up.buf.as_ptr();
        let xqp = xq.buf.as_ptr();
        let xsp = xs.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let mut gmi = gate_m as i32;
        let mut umi = up_m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wgp as *const _ as *mut c_void,
            &wup as *const _ as *mut c_void,
            &xqp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &mut gmi as *mut _ as *mut c_void,
            &mut umi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let grid_x = (((gate_m + 15) / 16) + ((up_m + 15) / 16)) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        let func = &self.functions["fused_gate_up_oq4_wmma"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Opus Quant W4A4 fused QKVZA: in_proj_qkv/z/beta/alpha grouped-iu4 GEMMs in
    /// one launch sharing the int4 activation. Each weight buffer is the combined
    /// `[nibbles | f32 scales]` layout. Outputs the four projections separately.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkvza_oq4_wmma(
        &mut self,
        w_qkv: &GpuTensor,
        w_z: &GpuTensor,
        w_beta: &GpuTensor,
        w_alpha: &GpuTensor,
        xq: &GpuTensor,
        xs: &GpuTensor,
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
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            k % group,
            0,
            "fused_qkvza_oq4_wmma: K must be a multiple of group"
        );
        assert_eq!(
            group % 16,
            0,
            "fused_qkvza_oq4_wmma: group must be a multiple of 16"
        );
        self.ensure_kernel(
            "fused_qkvza_oq4_wmma",
            kernels::FUSED_QKVZA_OQ4_WMMA_SRC,
            "fused_qkvza_oq4_wmma",
        )?;
        let p_wqkv = w_qkv.buf.as_ptr();
        let p_wz = w_z.buf.as_ptr();
        let p_wbeta = w_beta.buf.as_ptr();
        let p_walpha = w_alpha.buf.as_ptr();
        let p_xq = xq.buf.as_ptr();
        let p_xs = xs.buf.as_ptr();
        let p_yqkv = y_qkv.buf.as_ptr();
        let p_yz = y_z.buf.as_ptr();
        let p_ybeta = y_beta.buf.as_ptr();
        let p_yalpha = y_alpha.buf.as_ptr();
        let mut qm = qkv_m as i32;
        let mut zm = z_m as i32;
        let mut bm = beta_m as i32;
        let mut am = alpha_m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &p_wqkv as *const _ as *mut c_void,
            &p_wz as *const _ as *mut c_void,
            &p_wbeta as *const _ as *mut c_void,
            &p_walpha as *const _ as *mut c_void,
            &p_xq as *const _ as *mut c_void,
            &p_xs as *const _ as *mut c_void,
            &p_yqkv as *const _ as *mut c_void,
            &p_yz as *const _ as *mut c_void,
            &p_ybeta as *const _ as *mut c_void,
            &p_yalpha as *const _ as *mut c_void,
            &mut qm as *mut _ as *mut c_void,
            &mut zm as *mut _ as *mut c_void,
            &mut bm as *mut _ as *mut c_void,
            &mut am as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let tiles = |m: usize| (m + 15) / 16;
        let grid_x = (tiles(qkv_m) + tiles(z_m) + tiles(beta_m) + tiles(alpha_m)) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        let func = &self.functions["fused_qkvza_oq4_wmma"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// OQ4+ interleaved-layout fused QKVZA decode (4-way demux, one launch).
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkvza_oq4_interleaved(
        &mut self,
        w_qkv: &GpuTensor,
        w_z: &GpuTensor,
        w_beta: &GpuTensor,
        w_alpha: &GpuTensor,
        x_f32: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(group, 256, "fused_qkvza_oq4_interleaved: group must be 256");
        self.ensure_kernel(
            "fused_qkvza_oq4_interleaved",
            kernels::FUSED_QKVZA_OQ4_INTERLEAVED_SRC,
            "fused_qkvza_oq4_interleaved",
        )?;
        let p_wqkv = w_qkv.buf.as_ptr();
        let p_wz = w_z.buf.as_ptr();
        let p_wbeta = w_beta.buf.as_ptr();
        let p_walpha = w_alpha.buf.as_ptr();
        let p_x = x_f32.buf.as_ptr();
        let p_yqkv = y_qkv.buf.as_ptr();
        let p_yz = y_z.buf.as_ptr();
        let p_ybeta = y_beta.buf.as_ptr();
        let p_yalpha = y_alpha.buf.as_ptr();
        let qm = qkv_m as i32;
        let zm = z_m as i32;
        let bm = beta_m as i32;
        let am = alpha_m as i32;
        let ki = k as i32;
        let gi = group as i32;
        let grid_x = (qkv_m + z_m + beta_m + alpha_m) as u32;
        self.launch_kernargs(
            "fused_qkvza_oq4_interleaved",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr p_wqkv, ptr p_wz, ptr p_wbeta, ptr p_walpha, ptr p_x, ptr p_yqkv, ptr p_yz, ptr p_ybeta, ptr p_yalpha, i32 qm, i32 zm, i32 bm, i32 am, i32 ki, i32 gi],
        )
    }

    /// OQ4+ interleaved-layout fused gate+up decode (2-way demux, one launch).
    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_oq4_interleaved(
        &mut self,
        w_gate: &GpuTensor,
        w_up: &GpuTensor,
        x_f32: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            group, 256,
            "fused_gate_up_oq4_interleaved: group must be 256"
        );
        self.ensure_kernel(
            "fused_gate_up_oq4_interleaved",
            kernels::FUSED_GATE_UP_OQ4_INTERLEAVED_SRC,
            "fused_gate_up_oq4_interleaved",
        )?;
        let wgp = w_gate.buf.as_ptr();
        let wup = w_up.buf.as_ptr();
        let xp = x_f32.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let gmi = gate_m as i32;
        let umi = up_m as i32;
        let ki = k as i32;
        let gi = group as i32;
        let grid_x = (gate_m + up_m) as u32;
        self.launch_kernargs(
            "fused_gate_up_oq4_interleaved",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr wgp, ptr wup, ptr xp, ptr ygp, ptr yup, i32 gmi, i32 umi, i32 ki, i32 gi],
        )
    }

    /// OQ4+ W4A16 fused gate+up decode (B=1): two GEMVs in one launch.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_oq4_gemv(
        &mut self,
        w_gate: &GpuTensor,
        w_up: &GpuTensor,
        x_f32: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(group, 256, "fused_gate_up_oq4_gemv: group must be 256");
        self.ensure_kernel(
            "fused_gate_up_oq4_gemv",
            kernels::FUSED_GATE_UP_OQ4_GEMV_SRC,
            "fused_gate_up_oq4_gemv",
        )?;
        let wgp = w_gate.buf.as_ptr();
        let wup = w_up.buf.as_ptr();
        let xp = x_f32.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let gmi = gate_m as i32;
        let umi = up_m as i32;
        let ki = k as i32;
        let gi = group as i32;
        let grid_x = (gate_m + up_m) as u32;
        self.launch_kernargs(
            "fused_gate_up_oq4_gemv",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr wgp, ptr wup, ptr xp, ptr ygp, ptr yup, i32 gmi, i32 umi, i32 ki, i32 gi],
        )
    }

    /// OQ4+ fused QKVZA decode (B=1) W4A8 DP4A over a quantized activation.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkvza_oq4_dp4a(
        &mut self,
        w_qkv: &GpuTensor,
        w_z: &GpuTensor,
        w_beta: &GpuTensor,
        w_alpha: &GpuTensor,
        xq: &GpuTensor,
        xs: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(group, 256, "fused_qkvza_oq4_dp4a: group must be 256");
        let src = kernels::fused_qkvza_oq4_dp4a_for_arch(&self.arch);
        self.ensure_kernel("fused_qkvza_oq4_dp4a", src, "fused_qkvza_oq4_dp4a")?;
        let p_wqkv = w_qkv.buf.as_ptr();
        let p_wz = w_z.buf.as_ptr();
        let p_wbeta = w_beta.buf.as_ptr();
        let p_walpha = w_alpha.buf.as_ptr();
        let p_xq = xq.buf.as_ptr();
        let p_xs = xs.buf.as_ptr();
        let p_yqkv = y_qkv.buf.as_ptr();
        let p_yz = y_z.buf.as_ptr();
        let p_ybeta = y_beta.buf.as_ptr();
        let p_yalpha = y_alpha.buf.as_ptr();
        let qm = qkv_m as i32;
        let zm = z_m as i32;
        let bm = beta_m as i32;
        let am = alpha_m as i32;
        let ki = k as i32;
        let gi = group as i32;
        let grid_x = (qkv_m + z_m + beta_m + alpha_m) as u32;
        self.launch_kernargs(
            "fused_qkvza_oq4_dp4a",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr p_wqkv, ptr p_wz, ptr p_wbeta, ptr p_walpha, ptr p_xq, ptr p_xs, ptr p_yqkv, ptr p_yz, ptr p_ybeta, ptr p_yalpha, i32 qm, i32 zm, i32 bm, i32 am, i32 ki, i32 gi],
        )
    }

    /// OQ+ W8A8 fused gate+up prefill.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_oq8_wmma(
        &mut self,
        w_gate: &GpuTensor,
        w_up: &GpuTensor,
        xq: &GpuTensor,
        xs: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            k % group,
            0,
            "fused_gate_up_oq8_wmma: K must be a multiple of group"
        );
        assert_eq!(
            group % 16,
            0,
            "fused_gate_up_oq8_wmma: group must be a multiple of 16"
        );
        self.ensure_kernel(
            "fused_gate_up_oq8_wmma",
            kernels::FUSED_GATE_UP_OQ8_WMMA_SRC,
            "fused_gate_up_oq8_wmma",
        )?;
        let wgp = w_gate.buf.as_ptr();
        let wup = w_up.buf.as_ptr();
        let xqp = xq.buf.as_ptr();
        let xsp = xs.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let mut gmi = gate_m as i32;
        let mut umi = up_m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wgp as *const _ as *mut c_void,
            &wup as *const _ as *mut c_void,
            &xqp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &mut gmi as *mut _ as *mut c_void,
            &mut umi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let grid_x = (gate_m.div_ceil(16) + up_m.div_ceil(16)) as u32;
        let grid_b = batch_size.div_ceil(16) as u32;
        let func = &self.functions["fused_gate_up_oq8_wmma"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// OQ+ W8A8 fused QKVZA prefill.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkvza_oq8_wmma(
        &mut self,
        w_qkv: &GpuTensor,
        w_z: &GpuTensor,
        w_beta: &GpuTensor,
        w_alpha: &GpuTensor,
        xq: &GpuTensor,
        xs: &GpuTensor,
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
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            k % group,
            0,
            "fused_qkvza_oq8_wmma: K must be a multiple of group"
        );
        assert_eq!(
            group % 16,
            0,
            "fused_qkvza_oq8_wmma: group must be a multiple of 16"
        );
        self.ensure_kernel(
            "fused_qkvza_oq8_wmma",
            kernels::FUSED_QKVZA_OQ8_WMMA_SRC,
            "fused_qkvza_oq8_wmma",
        )?;
        let p_wqkv = w_qkv.buf.as_ptr();
        let p_wz = w_z.buf.as_ptr();
        let p_wbeta = w_beta.buf.as_ptr();
        let p_walpha = w_alpha.buf.as_ptr();
        let p_xq = xq.buf.as_ptr();
        let p_xs = xs.buf.as_ptr();
        let p_yqkv = y_qkv.buf.as_ptr();
        let p_yz = y_z.buf.as_ptr();
        let p_ybeta = y_beta.buf.as_ptr();
        let p_yalpha = y_alpha.buf.as_ptr();
        let mut qm = qkv_m as i32;
        let mut zm = z_m as i32;
        let mut bm = beta_m as i32;
        let mut am = alpha_m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &p_wqkv as *const _ as *mut c_void,
            &p_wz as *const _ as *mut c_void,
            &p_wbeta as *const _ as *mut c_void,
            &p_walpha as *const _ as *mut c_void,
            &p_xq as *const _ as *mut c_void,
            &p_xs as *const _ as *mut c_void,
            &p_yqkv as *const _ as *mut c_void,
            &p_yz as *const _ as *mut c_void,
            &p_ybeta as *const _ as *mut c_void,
            &p_yalpha as *const _ as *mut c_void,
            &mut qm as *mut _ as *mut c_void,
            &mut zm as *mut _ as *mut c_void,
            &mut bm as *mut _ as *mut c_void,
            &mut am as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let tiles = |m: usize| m.div_ceil(16);
        let grid_x = (tiles(qkv_m) + tiles(z_m) + tiles(beta_m) + tiles(alpha_m)) as u32;
        let grid_b = batch_size.div_ceil(16) as u32;
        let func = &self.functions["fused_qkvza_oq8_wmma"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// OQ+ W8A16 fused QKVZA decode: one wave32 per output row, blockIdx demuxes
    /// across the four projections in one launch.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkvza_oq8_gemv(
        &mut self,
        w_qkv: &GpuTensor,
        w_z: &GpuTensor,
        w_beta: &GpuTensor,
        w_alpha: &GpuTensor,
        x_f32: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(group, 256, "fused_qkvza_oq8_gemv: group must be 256");
        assert_eq!(
            k % group,
            0,
            "fused_qkvza_oq8_gemv: K must be a multiple of group"
        );
        self.ensure_kernel(
            "fused_qkvza_oq8_gemv",
            kernels::FUSED_QKVZA_OQ8_GEMV_SRC,
            "fused_qkvza_oq8_gemv",
        )?;
        let p_wqkv = w_qkv.buf.as_ptr();
        let p_wz = w_z.buf.as_ptr();
        let p_wbeta = w_beta.buf.as_ptr();
        let p_walpha = w_alpha.buf.as_ptr();
        let p_x = x_f32.buf.as_ptr();
        let p_yqkv = y_qkv.buf.as_ptr();
        let p_yz = y_z.buf.as_ptr();
        let p_ybeta = y_beta.buf.as_ptr();
        let p_yalpha = y_alpha.buf.as_ptr();
        let qm = qkv_m as i32;
        let zm = z_m as i32;
        let bm = beta_m as i32;
        let am = alpha_m as i32;
        let ki = k as i32;
        let gi = group as i32;
        let grid_x = (qkv_m + z_m + beta_m + alpha_m) as u32;
        self.launch_kernargs(
            "fused_qkvza_oq8_gemv",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr p_wqkv, ptr p_wz, ptr p_wbeta, ptr p_walpha, ptr p_x, ptr p_yqkv, ptr p_yz, ptr p_ybeta, ptr p_yalpha, i32 qm, i32 zm, i32 bm, i32 am, i32 ki, i32 gi],
        )
    }

    /// OQ+ W8A16 fused gate+up decode: one wave32 per output row, blockIdx
    /// demuxes gate versus up in one launch.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_oq8_gemv(
        &mut self,
        w_gate: &GpuTensor,
        w_up: &GpuTensor,
        x_f32: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(group, 256, "fused_gate_up_oq8_gemv: group must be 256");
        assert_eq!(
            k % group,
            0,
            "fused_gate_up_oq8_gemv: K must be a multiple of group"
        );
        self.ensure_kernel(
            "fused_gate_up_oq8_gemv",
            kernels::FUSED_GATE_UP_OQ8_GEMV_SRC,
            "fused_gate_up_oq8_gemv",
        )?;
        let wgp = w_gate.buf.as_ptr();
        let wup = w_up.buf.as_ptr();
        let xp = x_f32.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let gmi = gate_m as i32;
        let umi = up_m as i32;
        let ki = k as i32;
        let gi = group as i32;
        let grid_x = (gate_m + up_m) as u32;
        self.launch_kernargs(
            "fused_gate_up_oq8_gemv",
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr wgp, ptr wup, ptr xp, ptr ygp, ptr yup, i32 gmi, i32 umi, i32 ki, i32 gi],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkvza_f16_xf32(
        &mut self,
        wqkv: &GpuTensor,
        wz: &GpuTensor,
        wbeta: &GpuTensor,
        walpha: &GpuTensor,
        x: &GpuTensor,
        yqkv: &GpuTensor,
        yz: &GpuTensor,
        ybeta: &GpuTensor,
        yalpha: &GpuTensor,
        mqkv: usize,
        mz: usize,
        mbeta: usize,
        malpha: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_qkvza_f16_xf32",
            kernels::FUSED_QKVZA_F16_XF32_SRC,
            "fused_qkvza_f16_xf32",
        )?;

        let qkvp = wqkv.buf.as_ptr();
        let zp = wz.buf.as_ptr();
        let bp = wbeta.buf.as_ptr();
        let ap = walpha.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yqkvp = yqkv.buf.as_ptr();
        let yzp = yz.buf.as_ptr();
        let ybp = ybeta.buf.as_ptr();
        let yap = yalpha.buf.as_ptr();
        let mqkv_val = mqkv as i32;
        let mz_val = mz as i32;
        let mbeta_val = mbeta as i32;
        let malpha_val = malpha as i32;
        let k_val = k as i32;

        let total = mqkv + mz + mbeta + malpha;
        let r = self.launch_kernargs(
            "fused_qkvza_f16_xf32",
            [total as u32, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr qkvp, ptr zp, ptr bp, ptr ap, ptr xp, ptr yqkvp, ptr yzp, ptr ybp, ptr yap, i32 mqkv_val, i32 mz_val, i32 mbeta_val, i32 malpha_val, i32 k_val],
        );
        // All four fused projections share input x → capture x per weight.
        self.maybe_capture_activation(wqkv, x, 1, k);
        self.maybe_capture_activation(wz, x, 1, k);
        self.maybe_capture_activation(wbeta, x, 1, k);
        self.maybe_capture_activation(walpha, x, 1, k);
        r
    }
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkvza_f16_xf32_batched(
        &mut self,
        wqkv: &GpuTensor,
        wz: &GpuTensor,
        wbeta: &GpuTensor,
        walpha: &GpuTensor,
        x: &GpuTensor,
        yqkv: &GpuTensor,
        yz: &GpuTensor,
        ybeta: &GpuTensor,
        yalpha: &GpuTensor,
        mqkv: usize,
        mz: usize,
        mbeta: usize,
        malpha: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_qkvza_f16_xf32_batched",
            kernels::FUSED_QKVZA_F16_XF32_BATCHED_SRC,
            "fused_qkvza_f16_xf32_batched",
        )?;

        let qkvp = wqkv.buf.as_ptr();
        let zp = wz.buf.as_ptr();
        let bp = wbeta.buf.as_ptr();
        let ap = walpha.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yqkvp = yqkv.buf.as_ptr();
        let yzp = yz.buf.as_ptr();
        let ybp = ybeta.buf.as_ptr();
        let yap = yalpha.buf.as_ptr();
        let mqkv_val = mqkv as i32;
        let mz_val = mz as i32;
        let mbeta_val = mbeta as i32;
        let malpha_val = malpha as i32;
        let k_val = k as i32;
        let b_val = batch_size as i32;

        let total = mqkv + mz + mbeta + malpha;
        self.launch_kernargs(
            "fused_qkvza_f16_xf32_batched",
            [total as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr qkvp, ptr zp, ptr bp, ptr ap, ptr xp, ptr yqkvp, ptr yzp, ptr ybp, ptr yap, i32 mqkv_val, i32 mz_val, i32 mbeta_val, i32 malpha_val, i32 k_val, i32 b_val],
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_f16_xf32(
        &mut self,
        wgate: &GpuTensor,
        wup: &GpuTensor,
        x: &GpuTensor,
        ygate: &GpuTensor,
        yup: &GpuTensor,
        mgate: usize,
        mup: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_gate_up_f16_xf32",
            kernels::FUSED_GATE_UP_F16_XF32_SRC,
            "fused_gate_up_f16_xf32",
        )?;

        let gp = wgate.buf.as_ptr();
        let up = wup.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = ygate.buf.as_ptr();
        let yup_p = yup.buf.as_ptr();
        let mgate_val = mgate as i32;
        let mup_val = mup as i32;
        let k_val = k as i32;

        let total = mgate + mup;
        let r = self.launch_kernargs(
            "fused_gate_up_f16_xf32",
            [total as u32, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr gp, ptr up, ptr xp, ptr ygp, ptr yup_p, i32 mgate_val, i32 mup_val, i32 k_val],
        );
        // gate + up share input x → capture x per weight.
        self.maybe_capture_activation(wgate, x, 1, k);
        self.maybe_capture_activation(wup, x, 1, k);
        r
    }
    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_f16_xf32_batched(
        &mut self,
        wgate: &GpuTensor,
        wup: &GpuTensor,
        x: &GpuTensor,
        ygate: &GpuTensor,
        yup: &GpuTensor,
        mgate: usize,
        mup: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_gate_up_f16_xf32_batched",
            kernels::FUSED_GATE_UP_F16_XF32_BATCHED_SRC,
            "fused_gate_up_f16_xf32_batched",
        )?;

        let gp = wgate.buf.as_ptr();
        let up = wup.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let ygp = ygate.buf.as_ptr();
        let yup_p = yup.buf.as_ptr();
        let mgate_val = mgate as i32;
        let mup_val = mup as i32;
        let k_val = k as i32;
        let b_val = batch_size as i32;

        let total = mgate + mup;
        self.launch_kernargs(
            "fused_gate_up_f16_xf32_batched",
            [total as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr gp, ptr up, ptr xp, ptr ygp, ptr yup_p, i32 mgate_val, i32 mup_val, i32 k_val, i32 b_val],
        )
    }
}
