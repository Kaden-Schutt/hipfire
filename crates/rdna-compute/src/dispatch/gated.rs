// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gated-norm + gated-delta-net (DeltaNet/GLA) dispatch. Pure move (Phase 1 M2).

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// Gated output norm: rmsnorm(x) * silu(z). Fused kernel.
    #[cfg(feature = "deltanet")]
    pub fn gated_norm_f32(
        &mut self,
        x: &GpuTensor,
        z: &GpuTensor,
        weight: &GpuTensor,
        out: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gated_norm", kernels::GATED_NORM_SRC, "gated_norm_f32")?;
        let xp = x.buf.as_ptr();
        let zp = z.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let op = out.buf.as_ptr();
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &zp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &ep as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gated_norm_bytes(n_heads * head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "gated_norm_f32", bytes);
        let result = self.launch_maybe_blob(
            "gated_norm_f32",
            [n_heads as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(zp);
                b.push_ptr(wp);
                b.push_ptr(op);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_f32(ep);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched `gated_norm_f32`. Grid.y is the batch dim.
    #[cfg(feature = "deltanet")]
    pub fn gated_norm_f32_batched(
        &mut self,
        x: &GpuTensor,
        z: &GpuTensor,
        weight: &GpuTensor,
        out: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gated_norm", kernels::GATED_NORM_SRC, "gated_norm_f32")?;
        let mut xp = x.buf.as_ptr();
        let mut zp = z.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut zp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::gated_norm_bytes(n_heads * head_dim) * batch_size;
        let timer =
            crate::profile::begin_timer(&self.hip, "rmsnorm", "gated_norm_f32_batched", bytes);
        let result = self.launch_maybe_blob(
            "gated_norm_f32",
            [n_heads as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(zp);
                b.push_ptr(wp);
                b.push_ptr(op);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_f32(ep);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Gated Delta Net recurrence. S matrix in LDS. Processes all tokens sequentially.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        state: &GpuTensor,
        output: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        Self::ensure_gdn_hd128(head_dim)?;
        self.ensure_kernel(
            "gated_delta_net",
            kernels::GATED_DELTA_NET_SRC,
            "gated_delta_net_f32",
        )?;
        let func = &self.functions["gated_delta_net_f32"];
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut bp = beta.buf.as_ptr();
        let mut sp = state.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        // 32 threads, tiled S in LDS (4KB per tile). Grid: [n_heads, 128/8=16].
        let n_tiles = (128 / 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, n_tiles, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Batched FP32-state Gated Delta Net recurrence. Processes all tokens
    /// sequentially inside the kernel and advances the FP32 state in place.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_f32_batch_seq(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        state: &GpuTensor,
        output: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        Self::ensure_gdn_hd128(head_dim)?;
        self.ensure_kernel(
            "gated_delta_net",
            kernels::GATED_DELTA_NET_SRC,
            "gated_delta_net_f32",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let vp = v.buf.as_ptr();
        let gp = gate.buf.as_ptr();
        let bp = beta.buf.as_ptr();
        let sp = state.buf.as_ptr();
        let op = output.buf.as_ptr();
        let nt = n_tokens as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &nt as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gated_delta_net_f32_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_f32_batch_seq",
            bytes,
        );
        let n_tiles = (128 / 4) as u32;
        let result = self.launch_maybe_blob(
            "gated_delta_net_f32",
            [n_heads as u32, n_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(gp);
                b.push_ptr(bp);
                b.push_ptr(sp);
                b.push_ptr(op);
                b.push_i32(nt);
                b.push_i32(nh);
                b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_f32_routed_batch_seq(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        state_ptrs: &GpuTensor,
        row_session_indices: &GpuTensor,
        output: &GpuTensor,
        ptr_layer_stride: usize,
        delta_layer_index: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        n_sessions: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        Self::ensure_gdn_hd128(head_dim)?;
        self.ensure_kernel(
            "gated_delta_net_f32_routed_batch_seq",
            kernels::GATED_DELTA_NET_F32_ROUTED_BATCH_SEQ_SRC,
            "gated_delta_net_f32_routed_batch_seq",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let vp = v.buf.as_ptr();
        let gp = gate.buf.as_ptr();
        let bp = beta.buf.as_ptr();
        let spp = state_ptrs.buf.as_ptr();
        let rsp = row_session_indices.buf.as_ptr();
        let op = output.buf.as_ptr();
        let ptr_stride = ptr_layer_stride as i32;
        let layer = delta_layer_index as i32;
        let nt = n_tokens as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &spp as *const _ as *mut c_void,
            &rsp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &ptr_stride as *const _ as *mut c_void,
            &layer as *const _ as *mut c_void,
            &nt as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gated_delta_net_f32_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_f32_routed_batch_seq",
            bytes,
        );
        let n_tiles = (128 / 4) as u32;
        let result = self.launch_maybe_blob(
            "gated_delta_net_f32_routed_batch_seq",
            [n_heads as u32, n_tiles, n_sessions as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(gp);
                b.push_ptr(bp);
                b.push_ptr(spp);
                b.push_ptr(rsp);
                b.push_ptr(op);
                b.push_i32(ptr_stride);
                b.push_i32(layer);
                b.push_i32(nt);
                b.push_i32(nh);
                b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    #[cfg(feature = "deltanet")]
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_net_q8_routed_batch_seq(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        state_ptrs: &GpuTensor,
        scale_ptrs: &GpuTensor,
        row_session_indices: &GpuTensor,
        output: &GpuTensor,
        ptr_layer_stride: usize,
        delta_layer_index: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        n_sessions: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        Self::ensure_gdn_hd128(head_dim)?;
        self.ensure_kernel(
            "gated_delta_net_q8_routed_batch_seq",
            kernels::GATED_DELTA_NET_Q8_ROUTED_BATCH_SEQ_SRC,
            "gated_delta_net_q8_routed_batch_seq",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let vp = v.buf.as_ptr();
        let gp = gate.buf.as_ptr();
        let bp = beta.buf.as_ptr();
        let spp = state_ptrs.buf.as_ptr();
        let scpp = scale_ptrs.buf.as_ptr();
        let rsp = row_session_indices.buf.as_ptr();
        let op = output.buf.as_ptr();
        let ptr_stride = ptr_layer_stride as i32;
        let layer = delta_layer_index as i32;
        let nt = n_tokens as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let fr = super::GDN_REQUANT_FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed) as i32;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &spp as *const _ as *mut c_void,
            &scpp as *const _ as *mut c_void,
            &rsp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &ptr_stride as *const _ as *mut c_void,
            &layer as *const _ as *mut c_void,
            &nt as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &fr as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::gated_delta_net_q8_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_q8_routed_batch_seq",
            bytes,
        );
        let n_tiles = (128 / 4) as u32;
        let result = self.launch_maybe_blob(
            "gated_delta_net_q8_routed_batch_seq",
            [n_heads as u32, n_tiles, n_sessions as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(gp);
                b.push_ptr(bp);
                b.push_ptr(spp);
                b.push_ptr(scpp);
                b.push_ptr(rsp);
                b.push_ptr(op);
                b.push_i32(ptr_stride);
                b.push_i32(layer);
                b.push_i32(nt);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_i32(fr);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// GDN recurrence with Q8-quantized S state — tiled LDS + warp-shuffle.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_q8(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        s_q8: &GpuTensor,
        s_scales: &GpuTensor,
        output: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        Self::ensure_gdn_hd128(head_dim)?;
        if self.gdn_q8_reg_gfx1151_enabled() {
            return self.gated_delta_net_q8_reg_gfx1151(
                q, k, v, gate, beta, s_q8, s_scales, output, n_tokens, n_heads, head_dim,
            );
        }
        self.ensure_kernel(
            "gated_delta_net_q8",
            kernels::GATED_DELTA_NET_Q8_SRC,
            "gated_delta_net_q8",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let vp = v.buf.as_ptr();
        let gp = gate.buf.as_ptr();
        let bp = beta.buf.as_ptr();
        let sp = s_q8.buf.as_ptr();
        let scp = s_scales.buf.as_ptr();
        let op = output.buf.as_ptr();
        let nt = n_tokens as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let fr = super::GDN_REQUANT_FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed) as i32;
        let ef_null: *const c_void = std::ptr::null();
        let rqt: i32 = 0; // single-end requant (MQ4/HFQ4 fast path; per-token=1 for PARO)
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &scp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &nt as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &fr as *const _ as *mut c_void,
            &ef_null as *const _ as *mut c_void,
            &rqt as *const _ as *mut c_void,
        ];
        let n_tiles = (128 / 4) as u32;
        let bytes = crate::profile::gated_delta_net_q8_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "deltanet", "gated_delta_net_q8", bytes);
        let result = self.launch_maybe_blob(
            "gated_delta_net_q8",
            [n_heads as u32, n_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(gp);
                b.push_ptr(bp);
                b.push_ptr(sp);
                b.push_ptr(scp);
                b.push_ptr(op);
                b.push_i32(nt);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_i32(fr);
                b.push_ptr(ef_null);
                b.push_i32(rqt);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched sequential `gated_delta_net_q8` for prefill.
    ///
    /// Launches the single-token kernel N times with offset pointers into
    /// [N × stride]-laid-out Q/K/V/gate/beta/output buffers. This preserves
    /// bit-exact semantics with N × `gated_delta_net_q8(n_tokens=1)` calls
    /// (i.e., dequant→update→requant per token, with stochastic rounding
    /// applied each step) — critical for byte-exact quality gate compliance.
    ///
    /// Why not just call the kernel once with `n_tokens=N`? The existing
    /// kernel dequants S_q8 once at start, runs N updates in FP32 inside
    /// LDS, and requants once at end. That collapses N rounding steps into
    /// one, producing numerically different output from sequential calls —
    /// diverges from the decode-path baseline.
    ///
    /// Q/K/V/output are [N × n_heads × 128] row-major. The `head_dim`
    /// argument is retained for call-site clarity and profiling, but this
    /// wrapper rejects any value other than 128 before launching the kernel.
    /// gate/beta are [N × n_heads] row-major.
    /// S_q8 / s_scales are the shared state (advanced N steps).
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_q8_batch_seq(
        &mut self,
        q_batch: &GpuTensor,
        k_batch: &GpuTensor,
        v_batch: &GpuTensor,
        gate_batch: &GpuTensor,
        beta_batch: &GpuTensor,
        s_q8: &GpuTensor,
        s_scales: &GpuTensor,
        output_batch: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        Self::ensure_gdn_hd128(head_dim)?;
        if self.gdn_q8_reg_gfx1151_enabled() {
            return self.gated_delta_net_q8_reg_gfx1151(
                q_batch,
                k_batch,
                v_batch,
                gate_batch,
                beta_batch,
                s_q8,
                s_scales,
                output_batch,
                n_tokens,
                n_heads,
                head_dim,
            );
        }
        self.ensure_kernel(
            "gated_delta_net_q8",
            kernels::GATED_DELTA_NET_Q8_SRC,
            "gated_delta_net_q8",
        )?;

        let n_tiles = (128 / 4) as u32;

        let mut qp = q_batch.buf.as_ptr();
        let mut kp = k_batch.buf.as_ptr();
        let mut vp = v_batch.buf.as_ptr();
        let mut gp = gate_batch.buf.as_ptr();
        let mut bp = beta_batch.buf.as_ptr();
        let mut sp = s_q8.buf.as_ptr();
        let mut scp = s_scales.buf.as_ptr();
        let mut op = output_batch.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut fr =
            super::GDN_REQUANT_FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed) as i32;
        let ef_null: *const c_void = std::ptr::null();
        let mut rqt: i32 = 0; // single-end requant (MQ4/HFQ4 fast path)
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut scp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut fr as *mut _ as *mut c_void,
            &ef_null as *const _ as *mut c_void,
            &mut rqt as *mut _ as *mut c_void,
        ];

        let bytes = crate::profile::gated_delta_net_q8_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_q8_batch_seq",
            bytes,
        );
        // Single launch — the kernel loops over n_tokens internally,
        // keeping state in F32 LDS across all tokens. Q8 quantization
        // happens once at the end instead of per-token, reducing noise
        // accumulation. Not byte-exact with N×1 decode calls but
        // strictly higher quality.
        let result = self.launch_maybe_blob(
            "gated_delta_net_q8",
            [n_heads as u32, n_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(gp);
                b.push_ptr(bp);
                b.push_ptr(sp);
                b.push_ptr(scp);
                b.push_ptr(op);
                b.push_i32(nt);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_i32(fr);
                b.push_ptr(ef_null);
                b.push_i32(rqt);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Tree-aware variant of `gated_delta_net_q8_batch_seq`. Per-token
    /// S-tile persist-write so sibling tokens read the parent's post-update
    /// state via `s_tape_q8[parent_indices[t]]`. `parent_indices[t] < 0`
    /// means "read pre-block initial state from `s_q8_init`".
    ///
    /// Does NOT advance persistent `s_q8_init` / `s_scales_init` (those
    /// are the pre-block snapshot, read-only). Caller runs linear replay
    /// on the accepted spine post-acceptance to commit the trajectory.
    ///
    /// Tape layout (caller responsibility):
    /// - `s_tape_q8`:     `[n_tokens × n_heads × HD × HD]` i8 (scratch)
    /// - `s_tape_scales`: `[n_tokens × n_heads × HD]` f32 (scratch)
    /// - `parent_indices`: `[n_tokens]` i32 (host materialized by
    ///   `ddtree::linearize_tree`; spine topology is [-1, 0, 1, 2, ...])
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_q8_tree_batch_seq(
        &mut self,
        q_batch: &GpuTensor,
        k_batch: &GpuTensor,
        v_batch: &GpuTensor,
        gate_batch: &GpuTensor,
        beta_batch: &GpuTensor,
        s_q8_init: &GpuTensor,
        s_scales_init: &GpuTensor,
        s_tape_q8: &GpuTensor,
        s_tape_scales: &GpuTensor,
        parent_indices: &GpuTensor,
        output_batch: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        Self::ensure_gdn_hd128(head_dim)?;
        self.ensure_kernel(
            "gated_delta_net_q8_tree",
            kernels::GATED_DELTA_NET_Q8_TREE_SRC,
            "gated_delta_net_q8_tree",
        )?;

        let n_tiles = (128 / 4) as u32;

        let mut qp = q_batch.buf.as_ptr();
        let mut kp = k_batch.buf.as_ptr();
        let mut vp = v_batch.buf.as_ptr();
        let mut gp = gate_batch.buf.as_ptr();
        let mut bp = beta_batch.buf.as_ptr();
        let mut sip = s_q8_init.buf.as_ptr();
        let mut scip = s_scales_init.buf.as_ptr();
        let mut stp = s_tape_q8.buf.as_ptr();
        let mut stsp = s_tape_scales.buf.as_ptr();
        let mut pp = parent_indices.buf.as_ptr();
        let mut op = output_batch.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut sip as *mut _ as *mut c_void,
            &mut scip as *mut _ as *mut c_void,
            &mut stp as *mut _ as *mut c_void,
            &mut stsp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];

        let bytes = crate::profile::gated_delta_net_q8_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_q8_tree_batch_seq",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gated_delta_net_q8_tree",
            [n_heads as u32, n_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(gp);
                b.push_ptr(bp);
                b.push_ptr(sip);
                b.push_ptr(scip);
                b.push_ptr(stp);
                b.push_ptr(stsp);
                b.push_ptr(pp);
                b.push_ptr(op);
                b.push_i32(nt);
                b.push_i32(nh);
                b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// GDN recurrence with Q4-quantized S state.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_q4(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        s_q4: &GpuTensor,
        s_scales: &GpuTensor,
        output: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        Self::ensure_gdn_hd128(head_dim)?;
        self.ensure_kernel(
            "gated_delta_net_q4",
            kernels::GATED_DELTA_NET_Q4_SRC,
            "gated_delta_net_q4",
        )?;
        let func = &self.functions["gated_delta_net_q4"];
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut bp = beta.buf.as_ptr();
        let mut sp = s_q4.buf.as_ptr();
        let mut scp = s_scales.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut scp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [128, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Gated linear-recurrence scan forward (fp32). `g`,`u`,`h_out`: `[seq*D]`
    /// row-major (time-major: index `t*D+c`). `h[t]=g[t]*h[t-1]+(1-g[t])*u[t]`,
    /// `h[-1]=0`. One thread per channel `c`, sequential over time; no shared mem.
    pub fn gated_scan_fwd(
        &mut self,
        g: &GpuTensor,
        u: &GpuTensor,
        h_out: &GpuTensor,
        seq: usize,
        d: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gated_scan_fwd",
            kernels::GATED_SCAN_TRAIN_SRC,
            "gated_scan_fwd",
        )?;
        let func = &self.functions["gated_scan_fwd"];
        let mut gp = g.buf.as_ptr();
        let mut up = u.buf.as_ptr();
        let mut hp = h_out.buf.as_ptr();
        let mut s = seq as i32;
        let mut dd = d as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut gp as *mut _ as *mut c_void,
            &mut up as *mut _ as *mut c_void,
            &mut hp as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut dd as *mut _ as *mut c_void,
        ];
        let blocks = (d as u32).div_ceil(256);
        unsafe {
            self.hip.launch_kernel(
                func,
                [blocks, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Gated linear-recurrence scan backward (fp32). Given `d_hout`=dL/dh[t] for
    /// every t, produces `d_g`,`d_u` (`[seq*D]`). Reverse scan, one thread per
    /// channel; `h_out` is the forward output (needed for `h[t-1]`). No shared mem.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_scan_bwd(
        &mut self,
        g: &GpuTensor,
        u: &GpuTensor,
        h_out: &GpuTensor,
        d_hout: &GpuTensor,
        d_g: &GpuTensor,
        d_u: &GpuTensor,
        seq: usize,
        d: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gated_scan_bwd",
            kernels::GATED_SCAN_TRAIN_SRC,
            "gated_scan_bwd",
        )?;
        let func = &self.functions["gated_scan_bwd"];
        let mut gp = g.buf.as_ptr();
        let mut up = u.buf.as_ptr();
        let mut hp = h_out.buf.as_ptr();
        let mut dhp = d_hout.buf.as_ptr();
        let mut dgp = d_g.buf.as_ptr();
        let mut dup = d_u.buf.as_ptr();
        let mut s = seq as i32;
        let mut dd = d as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut gp as *mut _ as *mut c_void,
            &mut up as *mut _ as *mut c_void,
            &mut hp as *mut _ as *mut c_void,
            &mut dhp as *mut _ as *mut c_void,
            &mut dgp as *mut _ as *mut c_void,
            &mut dup as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut dd as *mut _ as *mut c_void,
        ];
        let blocks = (d as u32).div_ceil(256);
        unsafe {
            self.hip.launch_kernel(
                func,
                [blocks, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
}
