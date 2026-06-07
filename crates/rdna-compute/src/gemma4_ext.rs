// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Gemma4-specific GPU dispatch method stubs.
//!
//! Signatures match the gemma4.rs call sites from
//! `feat/gemma4-128k-ring-buffer`. Real implementations will be ported
//! in Phases 1b (attention/kv), 1d (rope/softcap), and 4 (MoE).

use crate::GpuTensor;
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipError, HipResult, KernargBlob};

// ─── rope_partial_halved_f32 ───────────────────────────────────────────

impl crate::Gpu {
    #[allow(unused_variables)]
    pub fn rope_partial_halved_f32(
        &mut self,
        q: &GpuTensor, k: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_heads: usize, n_kv: usize, head_dim: usize,
        n_rot_pairs: usize, rope_theta: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rope_partial_halved",
            kernels::ROPE_PARTIAL_HALVED_SRC,
            "rope_partial_halved_f32",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let pp = pos_buf.as_ptr();
        let nhq = n_heads as i32;
        let nhk = n_kv as i32;
        let hd = head_dim as i32;
        let nrp = n_rot_pairs as i32;
        let fb = rope_theta;
        let n_pairs = n_rot_pairs as u32;
        let block = 32u32.min(n_pairs.max(1));
        let grid = [(n_pairs + block - 1) / block, 1, 1];
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            &qp as *const _ as *mut std::ffi::c_void,
            &kp as *const _ as *mut std::ffi::c_void,
            &pp as *const _ as *mut std::ffi::c_void,
            &nhq as *const _ as *mut std::ffi::c_void,
            &nhk as *const _ as *mut std::ffi::c_void,
            &hd as *const _ as *mut std::ffi::c_void,
            &nrp as *const _ as *mut std::ffi::c_void,
            &fb as *const _ as *mut std::ffi::c_void,
        ];
        self.launch_maybe_blob(
            "rope_partial_halved_f32",
            grid,
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(pp);
                b.push_i32(nhq);
                b.push_i32(nhk);
                b.push_i32(hd);
                b.push_i32(nrp);
                b.push_f32(fb);
                b
            },
        )
    }

    #[allow(unused_variables)]
    pub fn logit_softcap_f32(
        &mut self, x: &GpuTensor, n: usize, cap: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "logit_softcap_f32",
            kernels::LOGIT_SOFTCAP_SRC,
            "logit_softcap_f32",
        )?;
        let xp = x.buf.as_ptr();
        let n_i32 = n as i32;
        let cap_f = cap;
        let block = 256u32;
        let grid = [((n as u32) + block - 1) / block, 1, 1];
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            &xp as *const _ as *mut std::ffi::c_void,
            &n_i32 as *const _ as *mut std::ffi::c_void,
            &cap_f as *const _ as *mut std::ffi::c_void,
        ];
        self.launch_maybe_blob(
            "logit_softcap_f32",
            grid,
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_i32(n_i32);
                b.push_f32(cap_f);
                b
            },
        )
    }
}

// ─── MoE GPU method stubs ──────────────────────────────────────────────

impl crate::Gpu {
    #[allow(unused_variables)]
    pub fn gemv_mq4g256_moe_gate_up_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor, topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        m: usize, k: usize,
    ) -> HipResult<()> {
        Err(HipError::new(0, "MoE kernel not yet ported (Phase 4)"))
    }

    #[allow(unused_variables)]
    pub fn gemv_q8_0_moe_down_residual_scaled_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor, topk_indices: &GpuTensor,
        topk_weights: &GpuTensor, per_expert_scale: &GpuTensor,
        hidden_batch: &GpuTensor, x_residual: &GpuTensor,
        m: usize, k: usize,
    ) -> HipResult<()> {
        Err(HipError::new(0, "MoE kernel not yet ported (Phase 4)"))
    }

    #[allow(unused_variables)]
    pub fn gemv_hfq4g128_moe_down_residual_scaled_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor, topk_indices: &GpuTensor,
        topk_weights: &GpuTensor, per_expert_scale: &GpuTensor,
        hidden_batch: &GpuTensor, x_residual: &GpuTensor,
        m: usize, k: usize,
    ) -> HipResult<()> {
        Err(HipError::new(0, "MoE kernel not yet ported (Phase 4)"))
    }

    #[allow(unused_variables)]
    pub fn gemv_hfq4g128_moe_down_residual_scaled_k8_indexed_batched(
        &mut self,
        expert_ptrs: &GpuTensor, topk_indices: &GpuTensor,
        topk_weights: &GpuTensor, per_expert_scale: &GpuTensor,
        hidden_batch: &GpuTensor, x_residual: &GpuTensor,
        m: usize, k: usize, k_top: usize, batch_size: usize,
    ) -> HipResult<()> {
        Err(HipError::new(0, "MoE kernel not yet ported (Phase 4)"))
    }

    #[allow(unused_variables)]
    pub fn gemv_mq4g256_moe_gate_up_bucketed(
        &mut self,
        expert_ptrs: &GpuTensor, expert_offsets: &GpuTensor,
        expert_token_list: &GpuTensor, x_rot: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        m: usize, k: usize, k_top: usize, n_exp: usize,
    ) -> HipResult<()> {
        Err(HipError::new(0, "MoE kernel not yet ported (Phase 4)"))
    }

    #[allow(unused_variables)]
    pub fn gemv_hfq4g256_moe_gate_up_bucketed(
        &mut self,
        expert_ptrs: &GpuTensor, expert_offsets: &GpuTensor,
        expert_token_list: &GpuTensor, x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        m: usize, k: usize, k_top: usize, n_exp: usize,
    ) -> HipResult<()> {
        Err(HipError::new(0, "MoE kernel not yet ported (Phase 4)"))
    }

    #[allow(unused_variables)]
    pub fn gemv_hfq4g128_moe_down_residual_scaled_bucketed(
        &mut self,
        expert_ptrs: &GpuTensor, expert_offsets: &GpuTensor,
        expert_token_list: &GpuTensor,
        topk_weights: &GpuTensor, per_expert_scale: &GpuTensor,
        hidden_batch: &GpuTensor, x_residual: &GpuTensor,
        m: usize, k: usize, k_top: usize, n_exp: usize,
    ) -> HipResult<()> {
        Err(HipError::new(0, "MoE kernel not yet ported (Phase 4)"))
    }

    #[allow(unused_variables)]
    pub fn moe_bucket_build(
        &mut self,
        topk_indices: &GpuTensor,
        expert_offsets: &GpuTensor,
        expert_token_list: &GpuTensor,
        n_batch: usize, k_top: usize, n_exp: usize,
    ) -> HipResult<()> {
        Err(HipError::new(0, "MoE kernel not yet ported (Phase 4)"))
    }
}

// ─── Sliding-window attention variant stubs ────────────────────────────

impl crate::Gpu {
    #[allow(unused_variables)]
    pub fn attention_flash_asym3_window(
        &mut self,
        q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize,
        head_dim: usize, max_seq: usize, partials: &GpuTensor,
        window_size: u32, cache_capacity: u32,
    ) -> HipResult<()> {
        // Delegate to regular flash attention (window/cache_capacity ignored
        // until sliding-window kernels are ported in Phase 1b).
        let _ = (window_size, cache_capacity);
        self.attention_flash_asym3(
            q, k_cache, v_cache, out, pos_buf,
            cos_theta, sin_theta, seq_len_hint,
            n_heads, n_kv_heads, head_dim, max_seq, partials,
        )
    }

    #[allow(unused_variables)]
    pub fn attention_flash_asym4_window(
        &mut self,
        q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize,
        head_dim: usize, max_seq: usize, partials: &GpuTensor,
        window_size: u32, cache_capacity: u32,
    ) -> HipResult<()> {
        let _ = (window_size, cache_capacity);
        self.attention_flash_asym4(
            q, k_cache, v_cache, out, pos_buf,
            cos_theta, sin_theta, seq_len_hint,
            n_heads, n_kv_heads, head_dim, max_seq, partials,
        )
    }

    #[allow(unused_variables)]
    pub fn attention_flash_asym2_window(
        &mut self,
        q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize,
        head_dim: usize, max_seq: usize, partials: &GpuTensor,
        window_size: u32, cache_capacity: u32,
    ) -> HipResult<()> {
        let _ = (window_size, cache_capacity);
        self.attention_flash_asym2(
            q, k_cache, v_cache, out, pos_buf,
            cos_theta, sin_theta, seq_len_hint,
            n_heads, n_kv_heads, head_dim, max_seq, partials,
        )
    }

    #[allow(unused_variables)]
    pub fn attention_flash_q8_0_window(
        &mut self,
        q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize,
        head_dim: usize, max_seq: usize, partials: &GpuTensor,
        window_size: u32, cache_capacity: u32,
    ) -> HipResult<()> {
        let _ = (window_size, cache_capacity);
        self.attention_flash_q8_0(
            q, k_cache, v_cache, out, pos_buf,
            seq_len_hint, n_heads, n_kv_heads, head_dim, max_seq, partials,
        )
    }

    #[allow(unused_variables)]
    pub fn attention_flash_asym3_batched_window(
        &mut self,
        q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, n_batch: usize,
        partials: &GpuTensor,
        window_size: u32, cache_capacity: u32,
    ) -> HipResult<()> {
        let _ = (window_size, cache_capacity);
        // Delegate to batched flash attention (window ignored).
        // Note: attention_flash_asym3_batched doesn't exist on dispatch branch;
        // the batched-masked variant is used. For Phase 1a scaffold, use the
        // batched (non-masked) path with dummy tree params.
        self.attention_flash_asym3_batched(
            q, k_cache, v_cache, out, positions,
            cos_theta, sin_theta,
            n_heads, n_kv_heads, head_dim,
            max_seq, max_ctx_len, n_batch, partials,
        )
    }
}
