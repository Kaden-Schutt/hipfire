// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.
//! Gemma4-specific GPU dispatch methods.
//!
//! Ported from `feat/gemma4-128k-ring-buffer`. Includes hd512 attention,
//! proportional partial RoPE, logit softcap, and MoE stubs (Phase 4).

use crate::kernels;
use crate::{Gpu, GpuTensor};
use hip_bridge::{DeviceBuffer, HipResult};

// rope_partial_halved_f32 / logit_softcap_f32 live in norm.rs (master copies
// with profiling timers) — the ported duplicates were removed in the union merge.

// ─── hd512 attention + KV write (full-attention layers) ─────────────────

#[rustfmt::skip]
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
        let launch_tiles = if self.graphs.capture_mode || self.replay.is_recording() {
            max_tiles
        } else {
            actual_tiles
        };
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // Phase 1: tile kernel → unnormalized per-tile partials.
        {
            let qp = q.buf.as_ptr(); let kp = k_cache.buf.as_ptr();
            let vp = v_cache.buf.as_ptr(); let pp = partials.buf.as_ptr();
            let posp = pos_buf.as_ptr(); let ctp = cos_theta.buf.as_ptr();
            let stp = sin_theta.buf.as_ptr();
            let nh = n_heads as i32; let nkv = n_kv_heads as i32;
            let hd = head_dim as i32; let ms = max_seq as i32;
            let sc = scale; let ts = TILE_SIZE as i32; let mt = max_tiles as i32;
            let ws: i32 = 0; // window_size=0 → full causal (no sliding on full layers)
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &qp as *const _ as *mut _, &kp as *const _ as *mut _, &vp as *const _ as *mut _,
                &pp as *const _ as *mut _, &posp as *const _ as *mut _, &ctp as *const _ as *mut _,
                &stp as *const _ as *mut _, &nh as *const _ as *mut _, &nkv as *const _ as *mut _,
                &hd as *const _ as *mut _, &ms as *const _ as *mut _, &sc as *const _ as *mut _,
                &ts as *const _ as *mut _, &mt as *const _ as *mut _, &ws as *const _ as *mut _,
            ];
            let grid = [n_heads as u32, launch_tiles as u32, 1];
            let shared = ((TILE_SIZE + head_dim) * 4) as u32;
            self.launch_maybe_blob_position_grid(
                "attention_flash_asym3_tile_hd512", grid, [32, 1, 1], shared,
                &mut params, 1, 1, TILE_SIZE as u32,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(qp); b.push_ptr(kp); b.push_ptr(vp); b.push_ptr(pp);
                    b.push_ptr(posp); b.push_ptr(ctp); b.push_ptr(stp);
                    b.push_i32(nh); b.push_i32(nkv); b.push_i32(hd); b.push_i32(ms);
                    b.push_f32(sc); b.push_i32(ts); b.push_i32(mt); b.push_i32(ws); b
                },
            )?;
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
            let p_ptr = partials.buf.as_ptr(); let o_ptr = out.buf.as_ptr();
            let nh = n_heads as i32; let hd = head_dim as i32;
            let pos_ptr = pos_buf.as_ptr(); let ts = TILE_SIZE as i32;
            let mt = max_tiles as i32;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &p_ptr as *const _ as *mut _, &o_ptr as *const _ as *mut _,
                &nh as *const _ as *mut _, &hd as *const _ as *mut _,
                &pos_ptr as *const _ as *mut _, &ts as *const _ as *mut _,
                &mt as *const _ as *mut _,
            ];
            self.launch_maybe_blob(
                "attention_flash_q8_0_reduce", [n_heads as u32, 1, 1], [256, 1, 1],
                (max_tiles * std::mem::size_of::<f32>()) as u32, &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(p_ptr); b.push_ptr(o_ptr); b.push_i32(nh); b.push_i32(hd);
                    b.push_ptr(pos_ptr); b.push_i32(ts); b.push_i32(mt); b
                },
            )?;
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
            let kdp = k_dst.buf.as_ptr(); let ksp = k_src.buf.as_ptr();
            let pp = pos_buf.as_ptr(); let ctp = cos_theta.buf.as_ptr();
            let stp = sin_theta.buf.as_ptr();
            let nkv = n_kv_heads as i32; let hd = head_dim as i32;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &kdp as *const _ as *mut _, &ksp as *const _ as *mut _, &pp as *const _ as *mut _,
                &ctp as *const _ as *mut _, &stp as *const _ as *mut _, &nkv as *const _ as *mut _,
                &hd as *const _ as *mut _,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            self.launch_maybe_blob(
                "kv_cache_write_asym_k_givens3_hd512", [n_kv_heads as u32, 1, 1],
                [32, 1, 1], shared_mem, &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(kdp); b.push_ptr(ksp); b.push_ptr(pp); b.push_ptr(ctp);
                    b.push_ptr(stp); b.push_i32(nkv); b.push_i32(hd); b
                },
            )?;
        }
        // V: standard Q8_0
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }

    /// Single-token hd512 flash attention for fwht3 KV cache (Gemma4 full-attn layers).
    /// Signs1/signs2 are 512-element +-1 float arrays (same buffers used at K-write time).
    /// v_mode_bits: 8 = Q8_0 V (normal space, default).
    pub fn attention_flash_fwht3_hd512(
        &mut self,
        q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize,
        head_dim: usize, max_seq: usize, partials: &GpuTensor,
        v_mode_bits: i32,
    ) -> HipResult<()> {
        debug_assert_eq!(head_dim, 512, "attention_flash_fwht3_hd512 requires head_dim=512");
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "attention_flash_fwht3_tile_hd512",
            kernels::ATTENTION_FLASH_FWHT3_TILE_HD512_SRC,
            "attention_flash_fwht3_tile_hd512",
        )?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.graphs.capture_mode { max_tiles } else { actual_tiles };
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // Phase 1: tile kernel.
        {
            let func = &self.functions["attention_flash_fwht3_tile_hd512"];
            let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
            let mut vp = v_cache.buf.as_ptr(); let mut pp = partials.buf.as_ptr();
            let mut posp = pos_buf.as_ptr(); let mut s1p = signs1.buf.as_ptr();
            let mut s2p = signs2.buf.as_ptr();
            let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32; let mut ms = max_seq as i32;
            let mut sc = scale; let mut ts = TILE_SIZE as i32; let mut mt = max_tiles as i32;
            // window_size=0: Gemma4 full-attention layers are globally causal (no SWA).
            // MUST precede vm in the params vec — kernel signature is:
            //   ..., max_tiles, window_size, v_mode  (16 params total).
            // Omitting window_size shifts vm into kernel slot 14 (window_size), activating
            // an 8-token SWA window; v_mode lands on uninitialized stack -> Phase D V = zeros.
            let mut ws = 0i32;
            let mut vm = v_mode_bits;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut qp as *mut _ as *mut std::ffi::c_void, &mut kp as *mut _ as *mut std::ffi::c_void,
                &mut vp as *mut _ as *mut std::ffi::c_void, &mut pp as *mut _ as *mut std::ffi::c_void,
                &mut posp as *mut _ as *mut std::ffi::c_void, &mut s1p as *mut _ as *mut std::ffi::c_void,
                &mut s2p as *mut _ as *mut std::ffi::c_void, &mut nh as *mut _ as *mut std::ffi::c_void,
                &mut nkv as *mut _ as *mut std::ffi::c_void, &mut hd as *mut _ as *mut std::ffi::c_void,
                &mut ms as *mut _ as *mut std::ffi::c_void, &mut sc as *mut _ as *mut std::ffi::c_void,
                &mut ts as *mut _ as *mut std::ffi::c_void, &mut mt as *mut _ as *mut std::ffi::c_void,
                &mut ws as *mut _ as *mut std::ffi::c_void,
                &mut vm as *mut _ as *mut std::ffi::c_void,
            ];
            let grid = [n_heads as u32, launch_tiles as u32, 1];
            let shared = ((TILE_SIZE + head_dim) * 4) as u32;
            unsafe {
                self.hip.launch_kernel(func, grid, [32, 1, 1], shared, self.stream_ref(), &mut params)?;
            }
        }
        // Phase 2: reduce partials (q8_0_reduce handles n_halves=4 for hd=512 unchanged).
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
                    [256, 1, 1],
                    (max_tiles * std::mem::size_of::<f32>()) as u32,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }
        Ok(())
    }

    /// Single-token hd512 KV cache write for fwht3 (Gemma4 full-attn layers).
    /// K is written via FWHT-512 rotation + 3-bit quantize; V via standard Q8_0.
    /// Signs1/signs2 are 512-element +-1 float arrays (same buffers used at read time).
    pub fn kv_cache_write_fwht3_hd512(
        &mut self,
        k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        debug_assert_eq!(head_dim, 512, "kv_cache_write_fwht3_hd512 requires head_dim=512");
        self.bind_thread()?;
        // K: FWHT-512 rotated 3-bit.
        self.ensure_givens4_kernel(
            "kv_cache_write_fwht3_hd512",
            kernels::KV_CACHE_WRITE_FWHT3_HD512_SRC,
            "kv_cache_write_fwht3_hd512",
        )?;
        {
            let func = &self.functions["kv_cache_write_fwht3_hd512"];
            let mut kdp = k_dst.buf.as_ptr(); let mut ksp = k_src.buf.as_ptr();
            let mut pp = pos_buf.as_ptr(); let mut s1p = signs1.buf.as_ptr();
            let mut s2p = signs2.buf.as_ptr();
            let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut kdp as *mut _ as *mut std::ffi::c_void, &mut ksp as *mut _ as *mut std::ffi::c_void,
                &mut pp as *mut _ as *mut std::ffi::c_void, &mut s1p as *mut _ as *mut std::ffi::c_void,
                &mut s2p as *mut _ as *mut std::ffi::c_void, &mut nkv as *mut _ as *mut std::ffi::c_void,
                &mut hd as *mut _ as *mut std::ffi::c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            unsafe {
                self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared_mem,
                    self.stream_ref(), &mut params)?;
            }
        }
        // V: standard Q8_0.
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }
}

// ─── MoE GPU method stubs (Phase 4) ────────────────────────────────────

// These pre-modular stubs duplicate production implementations in gemm.rs.
#[cfg(any())]
#[rustfmt::skip]
impl Gpu {
    /// Indexed MoE gate_up GEMV for MQ4G256 expert weights.
    /// MQ4G256 has the same 136-byte/group layout as HFQ4G256, so this
    /// delegates to the same kernel — the only difference is that the
    /// input must be FWHT-pre-rotated (x_rot) by the caller.
    pub fn gemv_mq4g256_moe_gate_up_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        use std::os::raw::c_void;
        self.bind_thread()?;
        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_x) = if cdna_wave64 {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_gate_up_indexed_wave64",
                crate::kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_SRC,
                "gemv_hfq4g256_moe_gate_up_k8_indexed_wave64",
            )?;
            (
                "gemv_hfq4g256_moe_gate_up_k8_indexed_wave64",
                [64u32, 1, 1],
                ((m as u32) + 1) / 2,
            )
        } else {
            self.ensure_kernel(
                "gemv_hfq4g256_moe_gate_up_indexed",
                crate::kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_SRC,
                "gemv_hfq4g256_moe_gate_up_k8_indexed",
            )?;
            (
                "gemv_hfq4g256_moe_gate_up_k8_indexed",
                [32u32, 1, 1],
                m as u32,
            )
        };
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x_rot.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let bytes = 8 * (crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip, "gemv", "gemv_mq4g256_moe_gate_up_k8_indexed", bytes,
        );
        let result =
            self.launch_maybe_blob(func_name, [grid_x, 8, 1], block, 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Indexed MoE gate_up GEMV for Q8_0 expert weights (no FWHT rotation needed).
    /// Reads expert pointers from device, computes gate + up projections,
    /// writes split outputs to y_gate and y_up.
    pub fn gemv_q8_0_moe_gate_up_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_q8_0_moe_gate_up_k8_indexed",
            kernels::GEMV_Q8_0_MOE_GATE_UP_K8_INDEXED_SRC,
            "gemv_q8_0_moe_gate_up_k8_indexed",
        )?;
        let func = &self.functions["gemv_q8_0_moe_gate_up_k8_indexed"];
        let mut pp = expert_ptrs.buf.as_ptr();
        let mut ip = topk_indices.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut ygp = y_gate.buf.as_ptr();
        let mut yup = y_up.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            &mut pp as *mut _ as *mut std::ffi::c_void,
            &mut ip as *mut _ as *mut std::ffi::c_void,
            &mut xp as *mut _ as *mut std::ffi::c_void,
            &mut ygp as *mut _ as *mut std::ffi::c_void,
            &mut yup as *mut _ as *mut std::ffi::c_void,
            &mut m_val as *mut _ as *mut std::ffi::c_void,
            &mut k_val as *mut _ as *mut std::ffi::c_void,
        ];
        let result =
            self.launch_maybe_blob(
                "gemv_q8_0_moe_gate_up_k8_indexed",
                [m as u32, 8, 1],
                [32u32, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(pp);
                    b.push_ptr(ip);
                    b.push_ptr(xp);
                    b.push_ptr(ygp);
                    b.push_ptr(yup);
                    b.push_i32(m_val);
                    b.push_i32(k_val);
                    b
                },
            );
        result
    }
    #[allow(unused_variables)]
    /// Indexed MoE down-projection for Q8_0 expert weights, with fused
    /// scaled atomicAdd into the residual stream.
    pub fn gemv_q8_0_moe_down_residual_scaled_k8_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        per_expert_scale: &GpuTensor,
        hidden_batch: &GpuTensor,
        x_residual: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_q8_0_moe_down_residual_scaled_k8_indexed",
            kernels::GEMV_Q8_0_MOE_DOWN_RESIDUAL_SCALED_K8_INDEXED_SRC,
            "gemv_q8_0_moe_down_residual_scaled_k8_indexed",
        )?;
        let mut pp = expert_ptrs.buf.as_ptr();
        let mut ip = topk_indices.buf.as_ptr();
        let mut wp = topk_weights.buf.as_ptr();
        let mut sp = per_expert_scale.buf.as_ptr();
        let mut hbp = hidden_batch.buf.as_ptr();
        let mut xrp = x_residual.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            &mut pp as *mut _ as *mut std::ffi::c_void,
            &mut ip as *mut _ as *mut std::ffi::c_void,
            &mut wp as *mut _ as *mut std::ffi::c_void,
            &mut sp as *mut _ as *mut std::ffi::c_void,
            &mut hbp as *mut _ as *mut std::ffi::c_void,
            &mut xrp as *mut _ as *mut std::ffi::c_void,
            &mut m_val as *mut _ as *mut std::ffi::c_void,
            &mut k_val as *mut _ as *mut std::ffi::c_void,
        ];
        let result =
            self.launch_maybe_blob(
                "gemv_q8_0_moe_down_residual_scaled_k8_indexed",
                [m as u32, 8, 1],
                [32u32, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(pp);
                    b.push_ptr(ip);
                    b.push_ptr(wp);
                    b.push_ptr(sp);
                    b.push_ptr(hbp);
                    b.push_ptr(xrp);
                    b.push_i32(m_val);
                    b.push_i32(k_val);
                    b
                },
            );
        result
    }
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

// These wrappers target retired *_cap APIs; routing now lives in dispatch.
#[cfg(any())]
#[rustfmt::skip]
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
        if head_dim == 512 {
            // Full (global) layers: no sliding window (full causal).
            self.attention_flash_asym3_hd512(q, k_cache, v_cache, out, pos_buf,
                cos_theta, sin_theta, seq_len_hint, n_heads, n_kv_heads, head_dim, max_seq, partials)
        } else {
            // Sliding layers: window + optional ring-buffer wrap.
            self.attention_flash_asym3_cap(q, k_cache, v_cache, out, pos_buf,
                cos_theta, sin_theta, seq_len_hint, n_heads, n_kv_heads, head_dim, max_seq, partials,
                window_size as usize, cache_capacity)
        }
    }

    pub fn attention_flash_asym4_window(&mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor, out: &GpuTensor, pos_buf: &DeviceBuffer, cos_theta: &GpuTensor, sin_theta: &GpuTensor, seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize, partials: &GpuTensor, window_size: u32, cache_capacity: u32) -> HipResult<()> {
        self.attention_flash_asym4_cap(q, k_cache, v_cache, out, pos_buf,
            cos_theta, sin_theta, seq_len_hint, n_heads, n_kv_heads, head_dim,
            max_seq, partials, window_size, cache_capacity)
    }

    pub fn attention_flash_asym2_window(&mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor, out: &GpuTensor, pos_buf: &DeviceBuffer, cos_theta: &GpuTensor, sin_theta: &GpuTensor, seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize, partials: &GpuTensor, window_size: u32, cache_capacity: u32) -> HipResult<()> {
        self.attention_flash_asym2_cap(q, k_cache, v_cache, out, pos_buf,
            cos_theta, sin_theta, seq_len_hint, n_heads, n_kv_heads, head_dim,
            max_seq, partials, window_size, cache_capacity)
    }

    pub fn attention_flash_q8_0_window(&mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor, out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize, partials: &GpuTensor, window_size: u32, cache_capacity: u32) -> HipResult<()> {
        self.attention_flash_q8_0_cap(q, k_cache, v_cache, out, pos_buf, seq_len_hint, n_heads, n_kv_heads, head_dim, max_seq, partials, window_size, cache_capacity)
    }

    pub fn attention_flash_asym3_batched_window(&mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor, out: &GpuTensor, positions: &GpuTensor, cos_theta: &GpuTensor, sin_theta: &GpuTensor, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize, max_ctx_len: usize, n_batch: usize, partials: &GpuTensor, window_size: u32, cache_capacity: u32) -> HipResult<()> {
        let _ = (window_size, cache_capacity);
        self.attention_flash_asym3_batched(q, k_cache, v_cache, out, positions, cos_theta, sin_theta, n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, n_batch, partials)
    }
}
