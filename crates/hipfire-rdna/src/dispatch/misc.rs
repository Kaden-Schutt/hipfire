// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Misc op kernels not in a larger family: residual-quant, standalone Paro, Givens rotation, deinterleave, cross-entropy, cast, attn-bias, transpose, scatter, scale, l2-norm, qkv-split. Pure move (Phase 1 M7).

use super::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipResult};
use std::ffi::c_void;

impl Gpu {
    /// H-Neurons CETT GPU reduction for one transformer layer. Computes the
    /// per-response-token down_proj output norm, then accumulates
    /// `|act[t,j]|·col_norm[j]/(‖out[t]‖+1e-8)` over the response tokens into
    /// `sums[j]` (this layer's `[intermediate]` accumulator slice). Replaces the
    /// per-layer host download+reduce that forced ~2 device syncs per layer and
    /// serialized the capture forward. No LDS (gfx1103 fault class).
    #[allow(clippy::too_many_arguments)]
    pub fn cett_accumulate_layer(
        &mut self,
        act: &GpuTensor,      // [positions * intermediate] down_proj input
        col_norm: &GpuTensor, // [intermediate] this layer's column norms
        down_out: &GpuTensor, // [positions * hidden] down_proj output
        out_norm: &GpuTensor, // [positions] scratch
        sums: &GpuTensor,     // [intermediate] this layer's accumulator
        positions: usize,
        intermediate: usize,
        hidden: usize,
        resp_start: usize,
        resp_end: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("cett_reduce", kernels::CETT_REDUCE_SRC, "cett_out_norm")?;
        // Pass 1: per-token output norms directly from the materialized down_out.
        {
            let d_ptr = down_out.buf.as_ptr();
            let o_ptr = out_norm.buf.as_ptr();
            let p = positions as i32;
            let h = hidden as i32;
            let mut params: Vec<*mut c_void> = vec![
                &d_ptr as *const _ as *mut c_void,
                &o_ptr as *const _ as *mut c_void,
                &p as *const _ as *mut c_void,
                &h as *const _ as *mut c_void,
            ];
            let block = 64u32;
            let grid = (positions as u32).div_ceil(block).max(1);
            let func = &self.functions["cett_out_norm"];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [grid, 1, 1],
                    [block, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }
        self.cett_accumulate_pass(
            act,
            col_norm,
            out_norm,
            sums,
            positions,
            intermediate,
            resp_start,
            resp_end,
        )
    }

    /// Like [`Gpu::cett_accumulate_layer`] but the down_proj output is recovered
    /// from a residual snapshot: `down_out = x_after - x_before`. Used on the fast
    /// serving prefill path (`forward_prefill_chunk`), which fuses down_proj into
    /// the residual so the output isn't separately materialized.
    #[allow(clippy::too_many_arguments)]
    pub fn cett_accumulate_layer_residual(
        &mut self,
        act: &GpuTensor,      // [positions * intermediate] down_proj input
        col_norm: &GpuTensor, // [intermediate] this layer's column norms
        x_after: &GpuTensor,  // [positions * hidden] residual after the fused down add
        x_before: &GpuTensor, // [positions * hidden] residual snapshot before it
        out_norm: &GpuTensor, // [positions] scratch
        sums: &GpuTensor,     // [intermediate] this layer's accumulator
        positions: usize,
        intermediate: usize,
        hidden: usize,
        resp_start: usize,
        resp_end: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "cett_reduce",
            kernels::CETT_REDUCE_SRC,
            "cett_out_norm_diff",
        )?;
        // Pass 1: per-token output norms from the residual delta.
        {
            let a_ptr = x_after.buf.as_ptr();
            let b_ptr = x_before.buf.as_ptr();
            let o_ptr = out_norm.buf.as_ptr();
            let p = positions as i32;
            let h = hidden as i32;
            let mut params: Vec<*mut c_void> = vec![
                &a_ptr as *const _ as *mut c_void,
                &b_ptr as *const _ as *mut c_void,
                &o_ptr as *const _ as *mut c_void,
                &p as *const _ as *mut c_void,
                &h as *const _ as *mut c_void,
            ];
            let block = 64u32;
            let grid = (positions as u32).div_ceil(block).max(1);
            let func = &self.functions["cett_out_norm_diff"];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [grid, 1, 1],
                    [block, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }
        self.cett_accumulate_pass(
            act,
            col_norm,
            out_norm,
            sums,
            positions,
            intermediate,
            resp_start,
            resp_end,
        )
    }

    /// Shared Pass 2 for the CETT reduction: accumulate per-neuron
    /// `|act|·col_norm/(‖out‖+1e-8)` over the response tokens into `sums`.
    #[allow(clippy::too_many_arguments)]
    fn cett_accumulate_pass(
        &mut self,
        act: &GpuTensor,
        col_norm: &GpuTensor,
        out_norm: &GpuTensor,
        sums: &GpuTensor,
        positions: usize,
        intermediate: usize,
        resp_start: usize,
        resp_end: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("cett_reduce", kernels::CETT_REDUCE_SRC, "cett_accumulate")?;
        let a_ptr = act.buf.as_ptr();
        let c_ptr = col_norm.buf.as_ptr();
        let o_ptr = out_norm.buf.as_ptr();
        let s_ptr = sums.buf.as_ptr();
        let p = positions as i32;
        let i = intermediate as i32;
        let rs = resp_start as i32;
        let re = resp_end as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &c_ptr as *const _ as *mut c_void,
            &o_ptr as *const _ as *mut c_void,
            &s_ptr as *const _ as *mut c_void,
            &p as *const _ as *mut c_void,
            &i as *const _ as *mut c_void,
            &rs as *const _ as *mut c_void,
            &re as *const _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = (intermediate as u32).div_ceil(block).max(1);
        let func = &self.functions["cett_accumulate"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )?;
        }
        Ok(())
    }

    /// H-Neurons intervention gain for one transformer layer. Scales the FFN
    /// neuron activations in place by this layer's per-neuron gain slice
    /// (`gain[j]` = the swept scalar for an H-Neuron, else 1.0):
    /// `ffn[t,j] *= gain[j]` over all `positions × inter` elements. Applied to
    /// the down_proj input, on every position (prefill + decode). No LDS
    /// (gfx1103 fault class).
    pub fn hneuron_gain_layer(
        &mut self,
        ffn: &GpuTensor,  // [positions * inter] down_proj input, modified in place
        gain: &GpuTensor, // [inter] this layer's per-neuron gain
        positions: usize,
        inter: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("hneuron_gain", kernels::HNEURON_GAIN_SRC, "hneuron_gain")?;
        let f_ptr = ffn.buf.as_ptr();
        let m_ptr = gain.buf.as_ptr();
        let p = positions as i32;
        let i = inter as i32;
        let mut params: Vec<*mut c_void> = vec![
            &f_ptr as *const _ as *mut c_void,
            &m_ptr as *const _ as *mut c_void,
            &p as *const _ as *mut c_void,
            &i as *const _ as *mut c_void,
        ];
        let total = (positions.max(1) * inter.max(1)) as u32;
        let block = 256u32;
        let grid = total.div_ceil(block).max(1);
        let func = &self.functions["hneuron_gain"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )?;
        }
        Ok(())
    }

    /// ParoQuant Givens rotation: apply learned pairwise rotations + channel
    /// scaling to activation vector x in-place. Called before GEMV on
    /// ParoQ4G128 weights.
    ///
    /// x: [seq_len, hidden_dim] F16 (modified in place)
    /// pairs: [krot, hidden_dim] I16
    /// theta: [krot, hidden_dim/2] F16
    /// channel_scales: [hidden_dim] F16
    pub fn givens_rotate(
        &mut self,
        x: &GpuTensor,
        pairs: &GpuTensor,
        theta: &GpuTensor,
        channel_scales: &GpuTensor,
        seq_len: usize,
        hidden_dim: usize,
        krot: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "givens_rotate_f32",
            kernels::GIVENS_ROTATE_SRC,
            "givens_rotate_f32",
        )?;

        let cta_m: u32 = 4;
        let group_size: u32 = 128;
        let groups_per_row = (hidden_dim as u32 + group_size - 1) / group_size;
        let grid_x = ((seq_len as u32) + cta_m - 1) / cta_m;

        let x_ptr = x.buf.as_ptr();
        let pairs_ptr = pairs.buf.as_ptr();
        let theta_ptr = theta.buf.as_ptr();
        let cs_ptr = channel_scales.buf.as_ptr();
        let seq_val = seq_len as i32;
        let dim_val = hidden_dim as i32;
        let krot_val = krot as i32;

        let smem = (cta_m * group_size * 4) as u32; // CTA_M * GROUP_SIZE * sizeof(float)

        // Bytes: read+write activation (2 × seq × dim × 4) + read pairs/theta/scales
        // (krot × dim × 2 for pairs+theta packed, dim × 2 for scales).
        let bytes = seq_len * hidden_dim * 4 * 2 + krot * hidden_dim * 2 + hidden_dim * 2;
        let timer = crate::profile::begin_timer(&self.hip, "rotate", "givens_rotate_f32", bytes);
        let result = self.launch_kernargs(
            "givens_rotate_f32",
            [grid_x, groups_per_row, 1],
            [group_size / 2, 1, 1],
            smem,
            &kernargs![ptr x_ptr, ptr pairs_ptr, ptr theta_ptr, ptr cs_ptr, i32 seq_val, i32 dim_val, i32 krot_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Out-of-place Givens rotation. Reads `x_in`, writes rotated
    /// activations to `x_out`. Replaces the
    /// `copy_d2d + givens_rotate` pair used by `rotate_x_paro_for` —
    /// one graph node + one inter-node dependency removed.
    #[allow(clippy::too_many_arguments)]
    pub fn givens_rotate_to(
        &mut self,
        x_in: &GpuTensor,
        x_out: &GpuTensor,
        pairs: &GpuTensor,
        theta: &GpuTensor,
        channel_scales: &GpuTensor,
        seq_len: usize,
        hidden_dim: usize,
        krot: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "givens_rotate_to_f32",
            kernels::GIVENS_ROTATE_TO_SRC,
            "givens_rotate_to_f32",
        )?;

        let cta_m: u32 = 4;
        let group_size: u32 = 128;
        let groups_per_row = (hidden_dim as u32 + group_size - 1) / group_size;
        let grid_x = ((seq_len as u32) + cta_m - 1) / cta_m;

        let in_ptr = x_in.buf.as_ptr();
        let out_ptr = x_out.buf.as_ptr();
        let pairs_ptr = pairs.buf.as_ptr();
        let theta_ptr = theta.buf.as_ptr();
        let cs_ptr = channel_scales.buf.as_ptr();
        let seq_val = seq_len as i32;
        let dim_val = hidden_dim as i32;
        let krot_val = krot as i32;

        let smem = (cta_m * group_size * 4) as u32;

        // Bytes: read x_in (seq × dim × 4) + write x_out (seq × dim × 4)
        // + read pairs/theta/scales (krot × dim × 2 + dim × 2).
        let bytes = seq_len * hidden_dim * 4 * 2 + krot * hidden_dim * 2 + hidden_dim * 2;
        let timer = crate::profile::begin_timer(&self.hip, "rotate", "givens_rotate_to_f32", bytes);
        let result = self.launch_kernargs(
            "givens_rotate_to_f32",
            [grid_x, groups_per_row, 1],
            [group_size / 2, 1, 1],
            smem,
            &kernargs![ptr in_ptr, ptr out_ptr, ptr pairs_ptr, ptr theta_ptr, ptr cs_ptr, i32 seq_val, i32 dim_val, i32 krot_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// PARO4-G128 activation pre-rotation. This materializes the ParoQuant
    /// channel-scale + pair-rotation transform once per projection so the
    /// packed GEMV does not repeat it for every 8-output pack.
    pub fn paro4g128_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128 rotate requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128 rotate requires K multiple of 128, got {k}"
        );
        assert!(
            x_rot.buf.size() / 4 >= k,
            "PARO4G128 rotate scratch too small: {} floats for K={k}",
            x_rot.buf.size() / 4
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "paro4g128_rotate",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let x_rot_ptr = x_rot.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let groups = (k / 128) as u32;
        let bytes = crate::profile::paro4g128t_rotate_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "format", "paro4g128_rotate", bytes);
        let result = self.launch_kernargs(
            "paro4g128_rotate",
            [groups, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr a_ptr, ptr x_ptr, ptr x_rot_ptr, i32 m_val, i32 k_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(x_rot_ptr);
        result
    }
    /// PARO4-G128 fused SwiGLU activation + Paro pre-rotation. This is the
    /// useful fused shape for down projection: `x_rot = rotate(silu(gate)*up)`.
    pub fn paro4g128_swiglu_rotate(
        &mut self,
        a_raw: &GpuTensor,
        gate: &GpuTensor,
        up: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128 SwiGLU rotate requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128 SwiGLU rotate requires K multiple of 128, got {k}"
        );
        assert!(
            x_rot.buf.size() / 4 >= k,
            "PARO4G128 SwiGLU rotate scratch too small: {} floats for K={k}",
            x_rot.buf.size() / 4
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "paro4g128_swiglu_rotate",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let gate_ptr = gate.buf.as_ptr();
        let up_ptr = up.buf.as_ptr();
        let x_rot_ptr = x_rot.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let groups = (k / 128) as u32;
        let bytes = crate::profile::paro4g128t_rotate_bytes(m, k) + k * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "format", "paro4g128_swiglu_rotate", bytes);
        let result = self.launch_kernargs(
            "paro4g128_swiglu_rotate",
            [groups, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr a_ptr, ptr gate_ptr, ptr up_ptr, ptr x_rot_ptr, i32 m_val, i32 k_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(x_rot_ptr);
        result
    }
    /// PARO4-G128T activation pre-rotation. Same math as PARO4-G128, but
    /// theta is stored as precomputed f16 sin/cos pairs in the payload.
    pub fn paro4g128t_rotate(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128T rotate requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128T rotate requires K multiple of 128, got {k}"
        );
        assert!(
            x_rot.buf.size() / 4 >= k,
            "PARO4G128T rotate scratch too small: {} floats for K={k}",
            x_rot.buf.size() / 4
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "paro4g128t_rotate",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let x_rot_ptr = x_rot.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let groups = (k / 128) as u32;
        let bytes = crate::profile::paro4g128t_rotate_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "format", "paro4g128t_rotate", bytes);
        let result = self.launch_kernargs(
            "paro4g128t_rotate",
            [groups, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr a_ptr, ptr x_ptr, ptr x_rot_ptr, i32 m_val, i32 k_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(x_rot_ptr);
        result
    }
    /// PARO4-G128T fused SwiGLU activation + Paro pre-rotation.
    pub fn paro4g128t_swiglu_rotate(
        &mut self,
        a_raw: &GpuTensor,
        gate: &GpuTensor,
        up: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            m % 8,
            0,
            "PARO4G128T SwiGLU rotate requires M multiple of 8, got {m}"
        );
        assert_eq!(
            k % 128,
            0,
            "PARO4G128T SwiGLU rotate requires K multiple of 128, got {k}"
        );
        assert!(
            x_rot.buf.size() / 4 >= k,
            "PARO4G128T SwiGLU rotate scratch too small: {} floats for K={k}",
            x_rot.buf.size() / 4
        );
        self.ensure_kernel(
            "gemv_paro4g128",
            kernels::GEMV_PARO4G128_SRC,
            "paro4g128t_swiglu_rotate",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let gate_ptr = gate.buf.as_ptr();
        let up_ptr = up.buf.as_ptr();
        let x_rot_ptr = x_rot.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;

        let groups = (k / 128) as u32;
        let bytes = crate::profile::paro4g128t_rotate_bytes(m, k) + k * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "format", "paro4g128t_swiglu_rotate", bytes);
        let result = self.launch_kernargs(
            "paro4g128t_swiglu_rotate",
            [groups, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr a_ptr, ptr gate_ptr, ptr up_ptr, ptr x_rot_ptr, i32 m_val, i32 k_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(x_rot_ptr);
        result
    }
    /// RoughQuant reader gather: `dst[j] = src[idx[j]]` for j<n_idx, 0 for the
    /// power-of-2 padding up to n_out. `idx` is an i32 GpuTensor of length n_idx.
    pub fn rq_gather_f32(
        &mut self,
        src: &GpuTensor,
        idx: &GpuTensor,
        dst: &GpuTensor,
        n_idx: usize,
        n_out: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("rq_correction", kernels::RQ_CORRECTION_SRC, "rq_gather_f32")?;
        let src_ptr = src.buf.as_ptr();
        let idx_ptr = idx.buf.as_ptr();
        let dst_ptr = dst.buf.as_ptr();
        let ni = n_idx as i32;
        let no = n_out as i32;
        let block = 256u32;
        let grid = ((n_out as u32) + block - 1) / block;
        self.launch_kernargs(
            "rq_gather_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &kernargs![ptr src_ptr, ptr idx_ptr, ptr dst_ptr, i32 ni, i32 no],
        )
    }
    /// RoughQuant writer scatter-add: `y[idx[j]] += c[j]` for j<n_idx.
    pub fn rq_scatter_add_f32(
        &mut self,
        y: &GpuTensor,
        idx: &GpuTensor,
        c: &GpuTensor,
        n_idx: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rq_correction",
            kernels::RQ_CORRECTION_SRC,
            "rq_scatter_add_f32",
        )?;
        let y_ptr = y.buf.as_ptr();
        let idx_ptr = idx.buf.as_ptr();
        let c_ptr = c.buf.as_ptr();
        let ni = n_idx as i32;
        let block = 256u32;
        let grid = ((n_idx as u32) + block - 1) / block;
        self.launch_kernargs(
            "rq_scatter_add_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &kernargs![ptr y_ptr, ptr idx_ptr, ptr c_ptr, i32 ni],
        )
    }
    pub fn scatter_session_last_logits_f32(
        &mut self,
        batch_logits: &GpuTensor,
        logits_ptrs: &GpuTensor,
        session_last_row_indices: &GpuTensor,
        vocab_size: usize,
        sessions: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "scatter_session_last_logits",
            kernels::SCATTER_SESSION_LAST_LOGITS_SRC,
            "scatter_session_last_logits_f32",
        )?;

        let batch_logits_ptr = batch_logits.buf.as_ptr();
        let logits_ptrs_ptr = logits_ptrs.buf.as_ptr();
        let session_last_row_indices_ptr = session_last_row_indices.buf.as_ptr();
        let vocab = vocab_size as i32;
        let ns = sessions as i32;
        let block = 256u32;
        let grid_x = (vocab_size as u32 + block - 1) / block;
        self.launch_kernargs(
            "scatter_session_last_logits_f32",
            [grid_x, sessions as u32, 1],
            [block, 1, 1],
            0,
            &kernargs![ptr batch_logits_ptr, ptr logits_ptrs_ptr, ptr session_last_row_indices_ptr, i32 vocab, i32 ns],
        )
    }
    /// Split a fused interleaved `[n_patches, 3 * hidden]` QKV buffer
    /// into three separate `[n_patches, hidden]` Q, K, V buffers.
    /// Used by the dots.ocr vision encoder when feeding the
    /// non-causal `attention_dflash_f32` kernel (which expects Q/K/V
    /// as separate flat buffers).
    ///
    /// `hidden` here is `n_heads * head_dim` — the second axis of each
    /// of Q, K, V within the fused buffer.
    pub fn qkv_split_interleaved_f32(
        &mut self,
        qkv: &GpuTensor,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        n_patches: usize,
        hidden: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            n_patches > 0,
            "qkv_split_interleaved_f32: n_patches must be > 0"
        );
        assert!(hidden > 0, "qkv_split_interleaved_f32: hidden must be > 0");
        self.ensure_kernel(
            "qkv_split_interleaved",
            kernels::QKV_SPLIT_INTERLEAVED_SRC,
            "qkv_split_interleaved_f32",
        )?;

        let qkvp = qkv.buf.as_ptr();
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let vp = v.buf.as_ptr();
        let np = n_patches as i32;
        let hd = hidden as i32;

        let block_size = 256u32;
        let grid_y = ((hidden as u32) + block_size - 1) / block_size;
        let grid = [n_patches as u32, grid_y, 1];
        let block = [block_size, 1, 1];
        // Bytes-touched estimate: 3 reads + 3 writes per (patch, j) thread.
        let bytes = n_patches * hidden * 4 * 6;
        let timer =
            crate::profile::begin_timer(&self.hip, "qkv_split", "qkv_split_interleaved_f32", bytes);
        let result = self.launch_kernargs(
            "qkv_split_interleaved_f32",
            grid,
            block,
            0,
            &kernargs![ptr qkvp, ptr qp, ptr kp, ptr vp, i32 np, i32 hd],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Deinterleave: split [A_h0(hd), B_h0(hd), A_h1(hd), B_h1(hd), ...] into A and B.
    /// Replaces per-head memcpy loop (n_heads × 2 ioctls → 1 dispatch).
    pub fn deinterleave_f32(
        &mut self,
        interleaved: &GpuTensor,
        out_a: &GpuTensor,
        out_b: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deinterleave",
            kernels::DEINTERLEAVE_SRC,
            "deinterleave_f32",
        )?;
        let inp = interleaved.buf.as_ptr();
        let ap = out_a.buf.as_ptr();
        let bp = out_b.buf.as_ptr();
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let total = (n_heads * head_dim) as u32;
        let block = 256u32;
        let grid = (total + block - 1) / block;
        let bytes = n_heads * head_dim * 4 * 3; // read interleaved, write both outputs
        let timer =
            crate::profile::begin_timer(&self.hip, "elementwise", "deinterleave_f32", bytes);
        let result = self.launch_kernargs(
            "deinterleave_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &kernargs![ptr inp, ptr ap, ptr bp, i32 nh, i32 hd],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched deinterleave: split [N × n_heads × head_dim × 2] interleaved
    /// Q+Gate into separate [N × n_heads × head_dim] Q and Gate tensors.
    /// Replaces the per-token gather/deinterleave/scatter loop in the FA
    /// batched prefill path.
    pub fn deinterleave_f32_batched(
        &mut self,
        interleaved: &GpuTensor,
        out_q: &GpuTensor,
        out_gate: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deinterleave_batched",
            kernels::DEINTERLEAVE_BATCHED_SRC,
            "deinterleave_f32_batched",
        )?;
        let inp = interleaved.buf.as_ptr();
        let qp = out_q.buf.as_ptr();
        let gp = out_gate.buf.as_ptr();
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let nn = n as i32;
        let total = (n_heads * head_dim) as u32;
        let block = 256u32;
        let grid_x = (total + block - 1) / block;
        let bytes = n * n_heads * head_dim * 4 * 3;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "deinterleave_f32_batched",
            bytes,
        );
        let result = self.launch_kernargs(
            "deinterleave_f32_batched",
            [grid_x, n as u32, 1],
            [block, 1, 1],
            0,
            &kernargs![ptr inp, ptr qp, ptr gp, i32 nh, i32 hd, i32 nn],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// L2 normalization per head, in-place. One warp per head.
    #[cfg(feature = "deltanet")]
    pub fn l2_norm_f32(
        &mut self,
        x: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("l2_norm", kernels::L2_NORM_SRC, "l2_norm_f32")?;
        let func = &self.functions["l2_norm_f32"];
        let mut xp = x.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::elementwise1_bytes(n_heads * head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "l2_norm_f32", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Scale vector by constant: x[i] *= scale. Replaces 48µs CPU roundtrip.
    #[cfg(feature = "deltanet")]
    pub fn scale_f32(&mut self, x: &GpuTensor, scale: f32) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("scale_f32", kernels::SCALE_F32_SRC, "scale_f32")?;
        let func = &self.functions["scale_f32"];
        let n = x.numel();
        let mut xp = x.buf.as_ptr();
        let mut nv = n as i32;
        let mut sv = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
            &mut sv as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise1_bytes(n);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "scale_f32", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Compute cross-entropy loss for a single token on GPU.
    /// Returns -log(softmax(logits)[target]). Downloads 4 bytes instead of 600KB.
    pub fn cross_entropy_loss(
        &mut self,
        logits: &GpuTensor,
        target_buf: &DeviceBuffer,
        loss_buf: &GpuTensor,
        vocab_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "cross_entropy_loss",
            kernels::CROSS_ENTROPY_LOSS_SRC,
            "cross_entropy_loss",
        )?;
        let func = &self.functions["cross_entropy_loss"];
        let mut lp = logits.buf.as_ptr();
        let mut tp = target_buf.as_ptr();
        let mut op = loss_buf.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
        ];
        let block_size = 256u32;
        let shared_mem = (block_size * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Fused cross-entropy fwd+bwd (fp32). `logits`,`d_logits`: `[rows*v]`;
    /// `targets`,`loss`: `[rows]` (targets integer-valued f32). `d_logits` is the
    /// SUM-reduction gradient (divide by valid-token count for mean).
    #[allow(clippy::too_many_arguments)]
    pub fn cross_entropy_train(
        &mut self,
        logits: &GpuTensor,
        targets: &GpuTensor,
        loss: &GpuTensor,
        d_logits: &GpuTensor,
        rows: usize,
        v: usize,
        ignore_index: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "cross_entropy_train",
            kernels::CROSS_ENTROPY_TRAIN_SRC,
            "cross_entropy_train",
        )?;
        let func = &self.functions["cross_entropy_train"];
        let mut lp = logits.buf.as_ptr();
        let mut tp = targets.buf.as_ptr();
        let mut losp = loss.buf.as_ptr();
        let mut dlp = d_logits.buf.as_ptr();
        let mut rowsi = rows as i32;
        let mut vi = v as i32;
        let mut ign = ignore_index;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut tp as *mut _ as *mut c_void,
            &mut losp as *mut _ as *mut c_void,
            &mut dlp as *mut _ as *mut c_void,
            &mut rowsi as *mut _ as *mut c_void,
            &mut vi as *mut _ as *mut c_void,
            &mut ign as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [rows as u32, 1, 1],
                [64, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Transpose [rows, cols] → [cols, rows]
    pub fn transpose_f32(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        rows: usize,
        cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("transpose_f32", kernels::TRANSPOSE_SRC, "transpose_f32")?;
        let func = &self.functions["transpose_f32"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut ri = rows as i32;
        let mut ci = cols as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut ri as *mut _ as *mut c_void,
            &mut ci as *mut _ as *mut c_void,
        ];
        let total = rows * cols;
        let blocks = ((total + 255) / 256) as u32;
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
    /// Cast an F32 tensor to BF16 (round-to-nearest-even, top 16 bits). Mirrors
    /// [`Self::cast_f32_to_f16`]; used to stage activations (e.g. the fused qkv)
    /// into bf16 for the bf16 attention/GEMM path.
    pub fn cast_f32_to_bf16(&mut self, src: &GpuTensor, dst: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(src.dtype, DType::F32, "cast_f32_to_bf16: src must be F32");
        assert_eq!(dst.dtype, DType::BF16, "cast_f32_to_bf16: dst must be BF16");
        let n_src: usize = src.shape.iter().product();
        let n_dst: usize = dst.shape.iter().product();
        assert_eq!(
            n_src, n_dst,
            "cast_f32_to_bf16: element counts must match (src={n_src}, dst={n_dst})"
        );
        self.ensure_kernel(
            "convert_f32_to_bf16",
            kernels::CONVERT_F32_TO_BF16_SRC,
            "convert_f32_to_bf16",
        )?;
        let in_ptr = src.buf.as_ptr();
        let out_ptr = dst.buf.as_ptr();
        let n_val = n_src as i32;
        let grid = ((n_src + 255) / 256) as u32;
        self.launch_kernargs(
            "convert_f32_to_bf16",
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &kernargs![ptr in_ptr, ptr out_ptr, i32 n_val],
        )
    }
    /// Split a fused f32 qkv `[N, 3*hidden]` into head-dim-padded q(f32),
    /// k(f16), v(f16) `[N, num_heads, hdp]` (dims `[head_dim, hdp)` zero-filled)
    /// — the layout the f16-KV WMMA flash kernels (head_dim=128) consume.
    pub fn attn_split_pad_f16kv(
        &mut self,
        qkv: &GpuTensor,
        q_pad: &GpuTensor,
        k_pad: &GpuTensor,
        v_pad: &GpuTensor,
        n: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
        hdp: usize,
        // Pre-scale applied to Q so a fixed-1/sqrt(hdp) downstream flash kernel
        // gets the correct 1/sqrt(head_dim) softmax scale: pass
        // sqrt(hdp/head_dim) when head_dim != hdp, else 1.0.
        q_scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attn_split_pad_f16kv",
            kernels::ATTN_PAD_F16KV_SRC,
            "attn_split_pad_f16kv",
        )?;
        let qkvp = qkv.buf.as_ptr();
        let qp = q_pad.buf.as_ptr();
        let kp = k_pad.buf.as_ptr();
        let vp = v_pad.buf.as_ptr();
        let (ni, hi, nh, hd, hp) = (
            n as i32,
            hidden as i32,
            num_heads as i32,
            head_dim as i32,
            hdp as i32,
        );
        let qs = q_scale;
        let total = (n * num_heads * hdp) as u32;
        let grid = (total + 255) / 256;
        self.launch_kernargs(
            "attn_split_pad_f16kv",
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &kernargs![ptr qkvp, ptr qp, ptr kp, ptr vp, i32 ni, i32 hi, i32 nh, i32 hd, i32 hp, f32 qs],
        )
    }
    /// Inverse of [`Self::attn_split_pad_f16kv`]'s padding: attention output
    /// `[N, num_heads, hdp]` → contiguous `[N, num_heads, head_dim]` = `[N, hidden]`.
    pub fn attn_unpad(
        &mut self,
        input: &GpuTensor,
        out: &GpuTensor,
        n: usize,
        num_heads: usize,
        head_dim: usize,
        hdp: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attn_unpad", kernels::ATTN_PAD_F16KV_SRC, "attn_unpad")?;
        let ip = input.buf.as_ptr();
        let op = out.buf.as_ptr();
        let (ni, nh, hd, hp) = (n as i32, num_heads as i32, head_dim as i32, hdp as i32);
        let total = (n * num_heads * head_dim) as u32;
        let grid = (total + 255) / 256;
        self.launch_kernargs(
            "attn_unpad",
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &kernargs![ptr ip, ptr op, i32 ni, i32 nh, i32 hd, i32 hp],
        )
    }
    /// f32 → f16 elementwise cast. `src` must be `DType::F32`, `dst`
    /// must be `DType::F16`, both with the same logical length. Single
    /// pass over the buffer; block [256], grid `ceil(n / 256)`.
    pub fn cast_f32_to_f16(&mut self, src: &GpuTensor, dst: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(src.dtype, DType::F32, "cast_f32_to_f16: src must be F32");
        assert_eq!(dst.dtype, DType::F16, "cast_f32_to_f16: dst must be F16");
        let n_src: usize = src.shape.iter().product();
        let n_dst: usize = dst.shape.iter().product();
        assert_eq!(
            n_src, n_dst,
            "cast_f32_to_f16: src and dst element counts must match (src={n_src}, dst={n_dst})",
        );
        self.ensure_kernel(
            "cast_f32_to_f16",
            kernels::CAST_F32_TO_F16_SRC,
            "cast_f32_to_f16",
        )?;
        let in_ptr = src.buf.as_ptr();
        let out_ptr = dst.buf.as_ptr();
        let n_val = n_src as i32;
        let grid = ((n_src + 255) / 256) as u32;
        let bytes = n_src * 6;
        let timer = crate::profile::begin_timer(&self.hip, "format", "cast_f32_to_f16", bytes);
        let result = self.launch_kernargs(
            "cast_f32_to_f16",
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &kernargs![ptr in_ptr, ptr out_ptr, i32 n_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
