// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt, Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.
//! Gemma4-specific GPU dispatch methods.
//!
//! Ported from `feat/gemma4-128k-ring-buffer`. Includes hd512 attention,
//! proportional partial RoPE, logit softcap, and MoE stubs (Phase 4).

use crate::{GpuTensor, Gpu};
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipError, HipResult, KernargBlob};

// ─── rope_partial_halved_f32 ───────────────────────────────────────────

impl Gpu {
    pub fn rope_partial_halved_f32(
        &mut self,
        q: &GpuTensor, k: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_heads: usize, n_kv: usize, head_dim: usize,
        n_rot_pairs: usize, rope_theta: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("rope_partial_halved", kernels::ROPE_PARTIAL_HALVED_SRC, "rope_partial_halved_f32")?;
        let qp = q.buf.as_ptr(); let kp = k.buf.as_ptr(); let pp = pos_buf.as_ptr();
        let nhq = n_heads as i32; let nhk = n_kv as i32; let hd = head_dim as i32;
        let nrp = n_rot_pairs as i32; let fb = rope_theta;
        let n_pairs = n_rot_pairs as u32;
        let block = 32u32.min(n_pairs.max(1));
        let grid = [(n_pairs + block - 1) / block, 1, 1];
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            &qp as *const _ as *mut std::ffi::c_void, &kp as *const _ as *mut std::ffi::c_void,
            &pp as *const _ as *mut std::ffi::c_void, &nhq as *const _ as *mut std::ffi::c_void,
            &nhk as *const _ as *mut std::ffi::c_void, &hd as *const _ as *mut std::ffi::c_void,
            &nrp as *const _ as *mut std::ffi::c_void, &fb as *const _ as *mut std::ffi::c_void,
        ];
        self.launch_maybe_blob("rope_partial_halved_f32", grid, [block, 1, 1], 0, &mut params, || {
            let mut b = KernargBlob::new();
            b.push_ptr(qp); b.push_ptr(kp); b.push_ptr(pp);
            b.push_i32(nhq); b.push_i32(nhk); b.push_i32(hd); b.push_i32(nrp); b.push_f32(fb);
            b
        })
    }

    pub fn logit_softcap_f32(&mut self, x: &GpuTensor, n: usize, cap: f32) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("logit_softcap_f32", kernels::LOGIT_SOFTCAP_SRC, "logit_softcap_f32")?;
        let xp = x.buf.as_ptr(); let n_i32 = n as i32; let cap_f = cap;
        let block = 256u32;
        let grid = [((n as u32) + block - 1) / block, 1, 1];
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            &xp as *const _ as *mut std::ffi::c_void,
            &n_i32 as *const _ as *mut std::ffi::c_void,
            &cap_f as *const _ as *mut std::ffi::c_void,
        ];
        self.launch_maybe_blob("logit_softcap_f32", grid, [block, 1, 1], 0, &mut params, || {
            let mut b = KernargBlob::new();
            b.push_ptr(xp); b.push_i32(n_i32); b.push_f32(cap_f);
            b
        })
    }
}

// ─── hd512 attention + KV write (full-attention layers) ─────────────────

impl Gpu {
    /// Single-token hd512 flash attention for asym3 KV cache (Gemma4 full-attn layers).
    pub fn attention_flash_asym3_hd512(
        &mut self,
        q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize,
        head_dim: usize, max_seq: usize, partials: &GpuTensor,
    ) -> HipResult<()> {
        debug_assert_eq!(head_dim, 512, "attention_flash_asym3_hd512 requires head_dim=512");
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "attention_flash_asym3_tile_hd512",
            kernels::ATTENTION_FLASH_ASYM3_TILE_HD512_SRC,
            "attention_flash_asym3_tile_hd512",
        )?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.graphs.capture_mode { max_tiles } else { actual_tiles };
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // Phase 1: tile kernel → unnormalized per-tile partials.
        {
            let func = &self.functions["attention_flash_asym3_tile_hd512"];
            let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
            let mut vp = v_cache.buf.as_ptr(); let mut pp = partials.buf.as_ptr();
            let mut posp = pos_buf.as_ptr(); let mut ctp = cos_theta.buf.as_ptr();
            let mut stp = sin_theta.buf.as_ptr();
            let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32; let mut ms = max_seq as i32;
            let mut sc = scale; let mut ts = TILE_SIZE as i32; let mut mt = max_tiles as i32;
            let mut ws: i32 = 0; // window_size=0 → full causal (no sliding on full layers)
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut qp as *mut _ as *mut std::ffi::c_void, &mut kp as *mut _ as *mut std::ffi::c_void,
                &mut vp as *mut _ as *mut std::ffi::c_void, &mut pp as *mut _ as *mut std::ffi::c_void,
                &mut posp as *mut _ as *mut std::ffi::c_void, &mut ctp as *mut _ as *mut std::ffi::c_void,
                &mut stp as *mut _ as *mut std::ffi::c_void, &mut nh as *mut _ as *mut std::ffi::c_void,
                &mut nkv as *mut _ as *mut std::ffi::c_void, &mut hd as *mut _ as *mut std::ffi::c_void,
                &mut ms as *mut _ as *mut std::ffi::c_void, &mut sc as *mut _ as *mut std::ffi::c_void,
                &mut ts as *mut _ as *mut std::ffi::c_void, &mut mt as *mut _ as *mut std::ffi::c_void,
                &mut ws as *mut _ as *mut std::ffi::c_void,
            ];
            let grid = [n_heads as u32, launch_tiles as u32, 1];
            let shared = ((TILE_SIZE + head_dim) * 4) as u32;
            unsafe {
                self.hip.launch_kernel(func, grid, [32, 1, 1], shared, self.stream_ref(), &mut params)?;
            }
        }
        // Phase 2: reduce partials → out. WITHOUT THIS, attn_out is never written
        // and full-attention layers read stale data from the prior sliding layer.
        // Mirrors the hd256 attention_flash_asym3 reduce (attention.rs). The reduce
        // kernel handles hd512 unchanged (n_halves = head_dim/128 = 4); partials
        // stride (2 + head_dim) = 514 matches the tile kernel's per-tile layout.
        self.ensure_kernel(
            "attention_flash_q8_0_reduce",
            kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC,
            "attention_flash_q8_0_reduce",
        )?;
        {
            let func = &self.functions["attention_flash_q8_0_reduce"];
            let mut p_ptr = partials.buf.as_ptr();
            let mut o_ptr = out.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut hd = head_dim as i32;
            let mut pos_ptr = pos_buf.as_ptr();
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut p_ptr as *mut _ as *mut std::ffi::c_void,
                &mut o_ptr as *mut _ as *mut std::ffi::c_void,
                &mut nh as *mut _ as *mut std::ffi::c_void,
                &mut hd as *mut _ as *mut std::ffi::c_void,
                &mut pos_ptr as *mut _ as *mut std::ffi::c_void,
                &mut ts as *mut _ as *mut std::ffi::c_void,
                &mut mt as *mut _ as *mut std::ffi::c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_heads as u32, 1, 1],
                    [32, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }
        Ok(())
    }

    /// Single-token hd512 KV cache write for asym3 (Gemma4 full-attn layers).
    pub fn kv_cache_write_asym3_hd512(
        &mut self,
        k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        debug_assert_eq!(head_dim, 512, "kv_cache_write_asym3_hd512 requires head_dim=512");
        self.bind_thread()?;
        // K: rotated 3-bit hd512
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_givens3_hd512",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_HD512_SRC,
            "kv_cache_write_asym_k_givens3_hd512",
        )?;
        {
            let func = &self.functions["kv_cache_write_asym_k_givens3_hd512"];
            let mut kdp = k_dst.buf.as_ptr(); let mut ksp = k_src.buf.as_ptr();
            let mut pp = pos_buf.as_ptr(); let mut ctp = cos_theta.buf.as_ptr();
            let mut stp = sin_theta.buf.as_ptr();
            let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut kdp as *mut _ as *mut std::ffi::c_void, &mut ksp as *mut _ as *mut std::ffi::c_void,
                &mut pp as *mut _ as *mut std::ffi::c_void, &mut ctp as *mut _ as *mut std::ffi::c_void,
                &mut stp as *mut _ as *mut std::ffi::c_void, &mut nkv as *mut _ as *mut std::ffi::c_void,
                &mut hd as *mut _ as *mut std::ffi::c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            unsafe {
                self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared_mem,
                    self.stream_ref(), &mut params)?;
            }
        }
        // V: standard Q8_0
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }
}

// ─── MoE GPU method stubs (Phase 4) ────────────────────────────────────

impl Gpu {
    #[allow(unused_variables)]
    pub fn gemv_mq4g256_moe_gate_up_k8_indexed(&mut self, expert_ptrs: &GpuTensor, topk_indices: &GpuTensor, x_rot: &GpuTensor, y_gate: &GpuTensor, y_up: &GpuTensor, m: usize, k: usize) -> HipResult<()> { Err(HipError::new(0, "MoE kernel not yet ported (Phase 4)")) }
    #[allow(unused_variables)]
    pub fn gemv_q8_0_moe_down_residual_scaled_k8_indexed(&mut self, expert_ptrs: &GpuTensor, topk_indices: &GpuTensor, topk_weights: &GpuTensor, per_expert_scale: &GpuTensor, hidden_batch: &GpuTensor, x_residual: &GpuTensor, m: usize, k: usize) -> HipResult<()> { Err(HipError::new(0, "MoE kernel not yet ported (Phase 4)")) }
    #[allow(unused_variables)]
    pub fn gemv_hfq4g128_moe_down_residual_scaled_k8_indexed(&mut self, expert_ptrs: &GpuTensor, topk_indices: &GpuTensor, topk_weights: &GpuTensor, per_expert_scale: &GpuTensor, hidden_batch: &GpuTensor, x_residual: &GpuTensor, m: usize, k: usize) -> HipResult<()> { Err(HipError::new(0, "MoE kernel not yet ported (Phase 4)")) }
    #[allow(unused_variables)]
    pub fn gemv_hfq4g128_moe_down_residual_scaled_k8_indexed_batched(&mut self, expert_ptrs: &GpuTensor, topk_indices: &GpuTensor, topk_weights: &GpuTensor, per_expert_scale: &GpuTensor, hidden_batch: &GpuTensor, x_residual: &GpuTensor, m: usize, k: usize, k_top: usize, batch_size: usize) -> HipResult<()> { Err(HipError::new(0, "MoE kernel not yet ported (Phase 4)")) }
    #[allow(unused_variables)]
    pub fn gemv_mq4g256_moe_gate_up_bucketed(&mut self, expert_ptrs: &GpuTensor, expert_offsets: &GpuTensor, expert_token_list: &GpuTensor, x_rot: &GpuTensor, y_gate: &GpuTensor, y_up: &GpuTensor, m: usize, k: usize, k_top: usize, n_exp: usize) -> HipResult<()> { Err(HipError::new(0, "MoE kernel not yet ported (Phase 4)")) }
    #[allow(unused_variables)]
    pub fn gemv_hfq4g256_moe_gate_up_bucketed(&mut self, expert_ptrs: &GpuTensor, expert_offsets: &GpuTensor, expert_token_list: &GpuTensor, x: &GpuTensor, y_gate: &GpuTensor, y_up: &GpuTensor, m: usize, k: usize, k_top: usize, n_exp: usize) -> HipResult<()> { Err(HipError::new(0, "MoE kernel not yet ported (Phase 4)")) }
    #[allow(unused_variables)]
    pub fn gemv_hfq4g128_moe_down_residual_scaled_bucketed(&mut self, expert_ptrs: &GpuTensor, expert_offsets: &GpuTensor, expert_token_list: &GpuTensor, topk_weights: &GpuTensor, per_expert_scale: &GpuTensor, hidden_batch: &GpuTensor, x_residual: &GpuTensor, m: usize, k: usize, k_top: usize, n_exp: usize) -> HipResult<()> { Err(HipError::new(0, "MoE kernel not yet ported (Phase 4)")) }
    #[allow(unused_variables)]
    pub fn moe_bucket_build(&mut self, topk_indices: &GpuTensor, expert_offsets: &GpuTensor, expert_token_list: &GpuTensor, n_batch: usize, k_top: usize, n_exp: usize) -> HipResult<()> { Err(HipError::new(0, "MoE kernel not yet ported (Phase 4)")) }
}

// ─── Sliding-window attention wrappers (route hd512 → hd512 kernels) ───

impl Gpu {
    pub fn attention_flash_asym3_window(
        &mut self,
        q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize,
        head_dim: usize, max_seq: usize, partials: &GpuTensor,
        window_size: u32, cache_capacity: u32,
    ) -> HipResult<()> {
        let _ = (window_size, cache_capacity);
        if head_dim == 512 {
            self.attention_flash_asym3_hd512(q, k_cache, v_cache, out, pos_buf,
                cos_theta, sin_theta, seq_len_hint, n_heads, n_kv_heads, head_dim, max_seq, partials)
        } else {
            self.attention_flash_asym3(q, k_cache, v_cache, out, pos_buf,
                cos_theta, sin_theta, seq_len_hint, n_heads, n_kv_heads, head_dim, max_seq, partials)
        }
    }

    pub fn attention_flash_asym4_window(&mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor, out: &GpuTensor, pos_buf: &DeviceBuffer, cos_theta: &GpuTensor, sin_theta: &GpuTensor, seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize, partials: &GpuTensor, window_size: u32, cache_capacity: u32) -> HipResult<()> {
        let _ = (window_size, cache_capacity);
        self.attention_flash_asym4(q, k_cache, v_cache, out, pos_buf, cos_theta, sin_theta, seq_len_hint, n_heads, n_kv_heads, head_dim, max_seq, partials)
    }

    pub fn attention_flash_asym2_window(&mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor, out: &GpuTensor, pos_buf: &DeviceBuffer, cos_theta: &GpuTensor, sin_theta: &GpuTensor, seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize, partials: &GpuTensor, window_size: u32, cache_capacity: u32) -> HipResult<()> {
        let _ = (window_size, cache_capacity);
        self.attention_flash_asym2(q, k_cache, v_cache, out, pos_buf, cos_theta, sin_theta, seq_len_hint, n_heads, n_kv_heads, head_dim, max_seq, partials)
    }

    pub fn attention_flash_q8_0_window(&mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor, out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize, partials: &GpuTensor, window_size: u32, cache_capacity: u32) -> HipResult<()> {
        let _ = (window_size, cache_capacity);
        self.attention_flash_q8_0(q, k_cache, v_cache, out, pos_buf, seq_len_hint, n_heads, n_kv_heads, head_dim, max_seq, partials)
    }

    pub fn attention_flash_asym3_batched_window(&mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor, out: &GpuTensor, positions: &GpuTensor, cos_theta: &GpuTensor, sin_theta: &GpuTensor, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize, max_ctx_len: usize, n_batch: usize, partials: &GpuTensor, window_size: u32, cache_capacity: u32) -> HipResult<()> {
        let _ = (window_size, cache_capacity);
        self.attention_flash_asym3_batched(q, k_cache, v_cache, out, positions, cos_theta, sin_theta, n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, n_batch, partials)
    }
}
