// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Attention dispatch: flash-decode / batched-prefill / GQA, sliding-window (SWA), TriAttention eviction, partial-flash (PFlash), and ViT attention. Pure move (Phase 1 M4).

use super::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipResult};
use std::ffi::c_void;

impl Gpu {
    /// Batched RMSNorm: normalize `batch` vectors of length `n` independently.
    /// x and out can be the same buffer (in-place). Weight is [n], applied per vector.
    /// TriAttention sidecar calibration: accumulate band statistics for one
    /// chunk's Q tensor (batched across all tokens in the chunk).
    ///
    /// q_batch: [n_tokens, n_heads, head_dim] f32 pre-RoPE Q (already on GPU).
    /// accs_sum_re/im/abs: [n_layers * n_heads * n_bands] f64 accumulators.
    /// accs_count: [n_layers * n_heads * n_bands] u64 sample counters.
    /// All accs_* buffers persist across calls; the kernel ADDS into them.
    ///
    /// Grid = [n_heads, n_bands, 1]. Block = [64, 1, 1]. Zero cross-block
    /// contention since each (layer, head, band) is written by exactly one
    /// block at a time (called sequentially per layer per chunk).
    pub fn triattn_accumulate(
        &mut self,
        q_batch: &DeviceBuffer,
        accs_sum_re: &DeviceBuffer,
        accs_sum_im: &DeviceBuffer,
        accs_sum_abs: &DeviceBuffer,
        accs_count: &DeviceBuffer,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        layer_idx: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "triattn_accumulate",
            kernels::TRIATTN_ACCUMULATE_SRC,
            "triattn_accumulate_f32",
        )?;

        let n_bands = head_dim / 2;

        let mut q_ptr = q_batch.as_ptr();
        let mut sre_ptr = accs_sum_re.as_ptr();
        let mut sim_ptr = accs_sum_im.as_ptr();
        let mut sab_ptr = accs_sum_abs.as_ptr();
        let mut cnt_ptr = accs_count.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut li = layer_idx as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut sre_ptr as *mut _ as *mut c_void,
            &mut sim_ptr as *mut _ as *mut c_void,
            &mut sab_ptr as *mut _ as *mut c_void,
            &mut cnt_ptr as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "triattn_accumulate_f32",
            [n_heads as u32, n_bands as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(sre_ptr);
                b.push_ptr(sim_ptr);
                b.push_ptr(sab_ptr);
                b.push_ptr(cnt_ptr);
                b.push_i32(nt);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_i32(li);
                b
            },
        )
    }
    /// GPU-side GQA attention.
    /// pos_buf: GPU buffer with single i32 position. Kernel computes seq_len = pos_buf[0] + 1.
    /// seq_len_hint: host-side seq_len for shared memory sizing (= pos + 1).
    pub fn attention_f32(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx1103 (Phoenix): no-LDS flash-decode variant (one wave32/head,
        // register online softmax) removes the scores[seq_len] LDS ceiling.
        let use_gfx1103 = self.arch_caps.is_gfx1103() && head_dim % 32 == 0;
        let (module, src, kname) = if use_gfx1103 {
            (
                "attention_f32_gfx1103",
                kernels::ATTENTION_F32_GFX1103_SRC,
                "attention_f32_gfx1103",
            )
        } else {
            ("attention", kernels::ATTENTION_SRC, "attention_f32")
        };
        self.ensure_kernel(module, src, kname)?;
        let func = &self.functions[kname];

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;

        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        // When a stream is active (graph capture mode), use max_seq for shared mem
        // so the captured graph works for all sequence lengths.
        let (block_size, shared_mem) = if use_gfx1103 {
            (32u32, 0u32)
        } else {
            let effective_seq = if self.active_stream.is_some() {
                max_seq
            } else {
                seq_len_hint
            };
            let block_size = (effective_seq.max(head_dim) as u32)
                .next_power_of_two()
                .min(256);
            let shared_mem = ((effective_seq + block_size as usize) * 4) as u32;
            (block_size, shared_mem)
        };

        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Flash-decoding attention: split KV scan for long sequences.
    /// Automatically chooses single-block or multi-block based on seq_len.
    pub fn attention_flash(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        partials: &GpuTensor,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Choose chunk size: aim for 4-16 chunks
        let chunk_size = if seq_len <= 128 { seq_len } else { 128 };
        let n_chunks = (seq_len + chunk_size - 1) / chunk_size;

        // Phase 1: compute partial attention per chunk
        self.ensure_kernel(
            "attention_flash_partial",
            kernels::ATTENTION_FLASH_SRC,
            "attention_flash_partial",
        )?;

        let q_ptr = q.buf.as_ptr();
        let k_ptr = k_cache.buf.as_ptr();
        let v_ptr = v_cache.buf.as_ptr();
        let p_ptr = partials.buf.as_ptr();
        let sl = seq_len as i32;
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let ms = max_seq as i32;
        let sc = scale;
        let cs = chunk_size as i32;

        let mut params1: Vec<*mut c_void> = vec![
            &q_ptr as *const _ as *mut c_void,
            &k_ptr as *const _ as *mut c_void,
            &v_ptr as *const _ as *mut c_void,
            &p_ptr as *const _ as *mut c_void,
            &sl as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &ms as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
            &cs as *const _ as *mut c_void,
        ];

        let block_size = 128u32.min(chunk_size as u32).next_power_of_two();
        let shared_mem = ((chunk_size + block_size as usize) * 4) as u32;

        self.launch_maybe_blob(
            "attention_flash_partial",
            [n_heads as u32, n_chunks as u32, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params1,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(k_ptr);
                b.push_ptr(v_ptr);
                b.push_ptr(p_ptr);
                b.push_i32(sl);
                b.push_i32(nh);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(ms);
                b.push_f32(sc);
                b.push_i32(cs);
                b
            },
        )?;

        // Phase 2: reduce partials
        self.ensure_kernel(
            "attention_flash_reduce",
            kernels::ATTENTION_FLASH_SRC,
            "attention_flash_reduce",
        )?;

        let p_ptr2 = partials.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let nh2 = n_heads as i32;
        let nc = n_chunks as i32;
        let hd2 = head_dim as i32;

        let mut params2: Vec<*mut c_void> = vec![
            &p_ptr2 as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &nh2 as *const _ as *mut c_void,
            &nc as *const _ as *mut c_void,
            &hd2 as *const _ as *mut c_void,
        ];

        let reduce_block = head_dim.min(256) as u32;
        self.launch_maybe_blob(
            "attention_flash_reduce",
            [n_heads as u32, 1, 1],
            [reduce_block, 1, 1],
            0,
            &mut params2,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(p_ptr2);
                b.push_ptr(out_ptr);
                b.push_i32(nh2);
                b.push_i32(nc);
                b.push_i32(hd2);
                b
            },
        )
    }
    /// GQA-aware split-K flash decode: one phase-1 block per (kv_head, chunk)
    /// reuses a single K/V load across its query-head group (n_heads/n_kv_heads),
    /// so the KV cache is traversed n_kv_heads× not n_heads×. Phase-2 reuses
    /// `attention_flash_reduce`. Same partials buffer as `attention_flash`.
    pub fn attention_flash_gqa(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        partials: &GpuTensor,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let cs_cap = std::env::var("HIPFIRE_GQA_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);
        let chunk_size = if seq_len <= cs_cap { seq_len } else { cs_cap };
        let n_chunks = (seq_len + chunk_size - 1) / chunk_size;

        self.ensure_kernel(
            "attention_flash_gqa_partial",
            kernels::ATTENTION_FLASH_GQA_SRC,
            "attention_flash_gqa_partial",
        )?;
        let q_ptr = q.buf.as_ptr();
        let k_ptr = k_cache.buf.as_ptr();
        let v_ptr = v_cache.buf.as_ptr();
        let p_ptr = partials.buf.as_ptr();
        let sl = seq_len as i32;
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let ms = max_seq as i32;
        let sc = scale;
        let cs = chunk_size as i32;
        let mut p1: Vec<*mut c_void> = vec![
            &q_ptr as *const _ as *mut c_void,
            &k_ptr as *const _ as *mut c_void,
            &v_ptr as *const _ as *mut c_void,
            &p_ptr as *const _ as *mut c_void,
            &sl as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &ms as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
            &cs as *const _ as *mut c_void,
        ];
        let block = 128u32.min(chunk_size as u32).next_power_of_two();
        let shmem = ((chunk_size + block as usize) * 4) as u32;
        self.launch_maybe_blob(
            "attention_flash_gqa_partial",
            [n_kv_heads as u32, n_chunks as u32, 1],
            [block, 1, 1],
            shmem,
            &mut p1,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(k_ptr);
                b.push_ptr(v_ptr);
                b.push_ptr(p_ptr);
                b.push_i32(sl);
                b.push_i32(nh);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(ms);
                b.push_f32(sc);
                b.push_i32(cs);
                b
            },
        )?;

        self.ensure_kernel(
            "attention_flash_reduce",
            kernels::ATTENTION_FLASH_SRC,
            "attention_flash_reduce",
        )?;
        let p2_ptr = partials.buf.as_ptr();
        let o_ptr = out.buf.as_ptr();
        let nh2 = n_heads as i32;
        let nc = n_chunks as i32;
        let hd2 = head_dim as i32;
        let mut p2: Vec<*mut c_void> = vec![
            &p2_ptr as *const _ as *mut c_void,
            &o_ptr as *const _ as *mut c_void,
            &nh2 as *const _ as *mut c_void,
            &nc as *const _ as *mut c_void,
            &hd2 as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "attention_flash_reduce",
            [n_heads as u32, 1, 1],
            [head_dim.min(256) as u32, 1, 1],
            0,
            &mut p2,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(p2_ptr);
                b.push_ptr(o_ptr);
                b.push_i32(nh2);
                b.push_i32(nc);
                b.push_i32(hd2);
                b
            },
        )
    }
    /// Single-launch GQA decode: one block per kv_head streams all KV once,
    /// accumulates online-softmax for the group in LDS, writes O. No partials,
    /// no reduce. Grid = n_kv_heads. Probe of launch-vs-occupancy floor.
    pub fn attention_flash_gqa_fused(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        self.ensure_kernel(
            "attention_flash_gqa_fused",
            kernels::ATTENTION_FLASH_GQA_FUSED_SRC,
            "attention_flash_gqa_fused",
        )?;
        let f = &self.functions["attention_flash_gqa_fused"];
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut o_ptr = out.buf.as_ptr();
        let mut sl = seq_len as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let kv_group = n_heads / n_kv_heads;
        let block = 128u32;
        let shmem = ((kv_group * head_dim + block as usize) * 4) as u32;
        let mut p: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut o_ptr as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                f,
                [n_kv_heads as u32, 1, 1],
                [block, 1, 1],
                shmem,
                self.stream_ref(),
                &mut p,
            )
        }
    }
    /// Attention with HFQ4 KV blocks (72 bytes per head, co-located).
    pub fn attention_hfq4_kv(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let use_gfx1103 = self.arch_caps.is_gfx1103() && head_dim % 32 == 0;
        let (module, src, kname) = if use_gfx1103 {
            (
                "attention_q4_affine_kv_gfx1103",
                kernels::ATTENTION_Q4_AFFINE_KV_GFX1103_SRC,
                "attention_hfq4_kv_gfx1103",
            )
        } else {
            (
                "attention_hfq4_kv",
                kernels::ATTENTION_HFQ4_KV_SRC,
                "attention_hfq4_kv",
            )
        };
        self.ensure_kernel(module, src, kname)?;
        let func = &self.functions[kname];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr();
        let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let (block_size, shared_mem) = if use_gfx1103 {
            (32u32, 0u32)
        } else {
            let block_size = (seq_len_hint.max(head_dim) as u32)
                .next_power_of_two()
                .min(256);
            // scores[seq_len] + ws[block_size] + q_shared[head_dim]
            (
                block_size,
                ((seq_len_hint + block_size as usize + head_dim) * 4) as u32,
            )
        };
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn attention_int8c_f16_kv(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let use_gfx1103 = self.arch_caps.is_gfx1103() && head_dim % 32 == 0;
        let (module, src, kname) = if use_gfx1103 {
            (
                "attention_int8c_f16_kv_gfx1103",
                kernels::ATTENTION_INT8C_F16_KV_GFX1103_SRC,
                "attention_int8c_f16_kv_gfx1103",
            )
        } else {
            (
                "attention_int8c_f16_kv",
                kernels::ATTENTION_INT8C_F16_KV_SRC,
                "attention_int8c_f16_kv",
            )
        };
        self.ensure_kernel(module, src, kname)?;
        let func = &self.functions[kname];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr();
        let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let (block_size, shared_mem) = if use_gfx1103 {
            (32u32, 0u32)
        } else {
            let block_size = (seq_len_hint.max(head_dim) as u32)
                .next_power_of_two()
                .min(256);
            (block_size, ((seq_len_hint + block_size as usize) * 4) as u32)
        };
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Attention with INT8 co-located KV blocks.
    pub fn attention_int8c_kv(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let use_gfx1103 = self.arch_caps.is_gfx1103() && head_dim % 32 == 0;
        let (module, src, kname) = if use_gfx1103 {
            (
                "attention_int8c_kv_gfx1103",
                kernels::ATTENTION_INT8C_KV_GFX1103_SRC,
                "attention_int8c_kv_gfx1103",
            )
        } else {
            (
                "attention_int8c_kv",
                kernels::ATTENTION_INT8C_KV_SRC,
                "attention_int8c_kv",
            )
        };
        self.ensure_kernel(module, src, kname)?;
        let func = &self.functions[kname];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr();
        let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let (block_size, shared_mem) = if use_gfx1103 {
            (32u32, 0u32)
        } else {
            let block_size = (seq_len_hint.max(head_dim) as u32)
                .next_power_of_two()
                .min(256);
            (
                block_size,
                ((seq_len_hint + block_size as usize + head_dim) * 4) as u32,
            )
        };
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Attention with HFQ8 KV cache.
    pub fn attention_hfq8_kv(
        &mut self,
        q: &GpuTensor,
        k_data: &GpuTensor,
        k_scales: &GpuTensor,
        v_data: &GpuTensor,
        v_scales: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let use_gfx1103 = self.arch_caps.is_gfx1103() && head_dim % 32 == 0;
        let (module, src, kname) = if use_gfx1103 {
            (
                "attention_hfq8_kv_gfx1103",
                kernels::ATTENTION_HFQ8_KV_GFX1103_SRC,
                "attention_hfq8_kv_gfx1103",
            )
        } else {
            (
                "attention_hfq8_kv",
                kernels::ATTENTION_HFQ8_KV_SRC,
                "attention_hfq8_kv",
            )
        };
        self.ensure_kernel(module, src, kname)?;
        let func = &self.functions[kname];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr();
        let mut kd = k_data.buf.as_ptr();
        let mut ks = k_scales.buf.as_ptr();
        let mut vd = v_data.buf.as_ptr();
        let mut vs = v_scales.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kd as *mut _ as *mut c_void,
            &mut ks as *mut _ as *mut c_void,
            &mut vd as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let (block_size, shared_mem) = if use_gfx1103 {
            (32u32, 0u32)
        } else {
            let block_size = (seq_len_hint.max(head_dim) as u32)
                .next_power_of_two()
                .min(256);
            (block_size, ((seq_len_hint + block_size as usize) * 4) as u32)
        };
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Attention with INT8 KV (separate scale array).
    pub fn attention_int8_kv(
        &mut self,
        q: &GpuTensor,
        k_vals: &GpuTensor,
        k_scales: &GpuTensor,
        v_vals: &GpuTensor,
        v_scales: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let use_gfx1103 = self.arch_caps.is_gfx1103() && head_dim % 32 == 0;
        let (module, src, kname) = if use_gfx1103 {
            (
                "attention_int8_kv_gfx1103",
                kernels::ATTENTION_INT8_KV_GFX1103_SRC,
                "attention_int8_kv_gfx1103",
            )
        } else {
            (
                "attention_int8_kv",
                kernels::ATTENTION_INT8_KV_SRC,
                "attention_int8_kv",
            )
        };
        self.ensure_kernel(module, src, kname)?;
        let func = &self.functions[kname];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut kv_ptr = k_vals.buf.as_ptr();
        let mut ks_ptr = k_scales.buf.as_ptr();
        let mut vv_ptr = v_vals.buf.as_ptr();
        let mut vs_ptr = v_scales.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut kv_ptr as *mut _ as *mut c_void,
            &mut ks_ptr as *mut _ as *mut c_void,
            &mut vv_ptr as *mut _ as *mut c_void,
            &mut vs_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let (block_size, shared_mem) = if use_gfx1103 {
            (32u32, 0u32)
        } else {
            let block_size = (seq_len_hint.max(head_dim) as u32)
                .next_power_of_two()
                .min(256);
            (block_size, ((seq_len_hint + block_size as usize) * 4) as u32)
        };
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Batched causal attention: all query positions in one launch.
    /// Q: [seq_len × n_heads × head_dim], K/V: [seq_len × n_kv_heads × head_dim].
    pub fn attention_causal_batched(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        out: &GpuTensor,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_causal_batched",
            kernels::ATTENTION_CAUSAL_BATCHED_SRC,
            "attention_causal_batched",
        )?;
        let func = &self.functions["attention_causal_batched"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut sl = seq_len as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut causal = 0i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut causal as *mut _ as *mut c_void,
        ];
        // Block size: enough threads to cover head_dim and seq_len
        let block_size = 128u32.min((seq_len.max(head_dim) as u32).next_power_of_two());
        // Shared: scores[seq_len] + workspace[block_size]
        let shared_mem = ((seq_len + block_size as usize) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, seq_len as u32, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Batched causal attention with Q8_0 quantized KV cache. Processes N
    /// queries in one launch; each query b has its own causal window read
    /// from positions[b] (i.e. attend to 0..positions[b]+1). Q and out are
    /// [batch_size × n_heads × head_dim] row-major; K/V caches are the same
    /// layout as `attention_q8_0_kv` and must already contain the prefix
    /// through positions[batch_size-1].
    ///
    /// Byte-exact with N single-token calls at batch_size=1, positions[0]=pos.
    ///
    /// `max_ctx_len` is the maximum seq_len = max(positions[b]) + 1 across
    /// the batch; used to size the shared memory allocation for scores[].
    pub fn attention_q8_0_kv_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_q8_0_kv_batched_masked(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            None,
            0,
            0,
        )
    }
    /// Batched causal attention with unquantized FP32 KV cache. Processes N
    /// queries in one launch; each query b has its own causal window read
    /// from positions[b].
    #[allow(clippy::too_many_arguments)]
    pub fn attention_f32_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates immediately to attention_f32_batched_masked.
        self.attention_f32_batched_masked(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            None,
            0,
            0,
        )
    }
    /// Tree-mask variant of `attention_f32_batched`. Normal batched prefill
    /// passes `tree_bias=None`; DDTree-style callers may pass a visibility
    /// bias with the same contract as `attention_q8_0_kv_batched_masked`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_f32_batched_masked(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_f32_batched",
            kernels::ATTENTION_F32_BATCHED_SRC,
            "attention_f32_batched",
        )?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let q_ptr = q.buf.as_ptr();
        let k_ptr = k_cache.buf.as_ptr();
        let v_ptr = v_cache.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let pos_ptr = positions.buf.as_ptr();
        let bias_ptr: *mut std::ffi::c_void = match tree_bias {
            Some(t) => t.buf.as_ptr(),
            None => std::ptr::null_mut(),
        };
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let ms = max_seq as i32;
        let sc = scale;
        let bs = block_start as i32;
        let bc = block_cols as i32;
        let mut params: Vec<*mut c_void> = vec![
            &q_ptr as *const _ as *mut c_void,
            &k_ptr as *const _ as *mut c_void,
            &v_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &pos_ptr as *const _ as *mut c_void,
            &bias_ptr as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &ms as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
            &bs as *const _ as *mut c_void,
            &bc as *const _ as *mut c_void,
        ];
        let block_size = (max_ctx_len.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        let shared_mem = ((max_ctx_len + block_size as usize + head_dim) * 4) as u32;
        let bytes =
            crate::profile::attention_f32_kv_bytes(n_heads, n_kv_heads, head_dim, max_ctx_len)
                * batch_size;
        let timer =
            crate::profile::begin_timer(&self.hip, "attention", "attention_f32_batched", bytes);
        let result = self.launch_maybe_blob(
            "attention_f32_batched",
            [n_heads as u32, batch_size as u32, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(k_ptr);
                b.push_ptr(v_ptr);
                b.push_ptr(out_ptr);
                b.push_ptr(pos_ptr);
                b.push_ptr(bias_ptr);
                b.push_i32(nh);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(ms);
                b.push_f32(sc);
                b.push_i32(bs);
                b.push_i32(bc);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    pub fn attention_f32_routed_batched(
        &mut self,
        q: &GpuTensor,
        k_ptrs: &GpuTensor,
        v_ptrs: &GpuTensor,
        out: &GpuTensor,
        row_session_indices: &GpuTensor,
        positions: &GpuTensor,
        ptr_layer_stride: usize,
        layer_index: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_f32_routed_batched",
            kernels::ATTENTION_F32_ROUTED_BATCHED_SRC,
            "attention_f32_routed_batched",
        )?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let q_ptr = q.buf.as_ptr();
        let k_ptrs_ptr = k_ptrs.buf.as_ptr();
        let v_ptrs_ptr = v_ptrs.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let row_session_indices_ptr = row_session_indices.buf.as_ptr();
        let pos_ptr = positions.buf.as_ptr();
        let ptr_stride = ptr_layer_stride as i32;
        let layer = layer_index as i32;
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let ms = max_seq as i32;
        let sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &q_ptr as *const _ as *mut c_void,
            &k_ptrs_ptr as *const _ as *mut c_void,
            &v_ptrs_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &row_session_indices_ptr as *const _ as *mut c_void,
            &pos_ptr as *const _ as *mut c_void,
            &ptr_stride as *const _ as *mut c_void,
            &layer as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &ms as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
        ];
        let block_size = (max_ctx_len.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        let shared_mem = ((max_ctx_len + block_size as usize + head_dim) * 4) as u32;
        let bytes =
            crate::profile::attention_f32_kv_bytes(n_heads, n_kv_heads, head_dim, max_ctx_len)
                * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "attention",
            "attention_f32_routed_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "attention_f32_routed_batched",
            [n_heads as u32, batch_size as u32, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(k_ptrs_ptr);
                b.push_ptr(v_ptrs_ptr);
                b.push_ptr(out_ptr);
                b.push_ptr(row_session_indices_ptr);
                b.push_ptr(pos_ptr);
                b.push_i32(ptr_stride);
                b.push_i32(layer);
                b.push_i32(nh);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(ms);
                b.push_f32(sc);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    pub fn attention_q8_0_routed_batched(
        &mut self,
        q: &GpuTensor,
        k_ptrs: &GpuTensor,
        v_ptrs: &GpuTensor,
        out: &GpuTensor,
        row_session_indices: &GpuTensor,
        positions: &GpuTensor,
        ptr_layer_stride: usize,
        layer_index: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_q8_0_routed_batched",
            kernels::ATTENTION_Q8_0_ROUTED_BATCHED_SRC,
            "attention_q8_0_routed_batched",
        )?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let q_ptr = q.buf.as_ptr();
        let k_ptrs_ptr = k_ptrs.buf.as_ptr();
        let v_ptrs_ptr = v_ptrs.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let row_session_indices_ptr = row_session_indices.buf.as_ptr();
        let pos_ptr = positions.buf.as_ptr();
        let ptr_stride = ptr_layer_stride as i32;
        let layer = layer_index as i32;
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let ms = max_seq as i32;
        let sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &q_ptr as *const _ as *mut c_void,
            &k_ptrs_ptr as *const _ as *mut c_void,
            &v_ptrs_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &row_session_indices_ptr as *const _ as *mut c_void,
            &pos_ptr as *const _ as *mut c_void,
            &ptr_stride as *const _ as *mut c_void,
            &layer as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &ms as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
        ];
        let block_size = (max_ctx_len.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        let shared_mem = ((max_ctx_len + block_size as usize + head_dim) * 4) as u32;
        let bytes =
            crate::profile::attention_q8_0_kv_bytes(n_heads, n_kv_heads, head_dim, max_ctx_len)
                * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "attention",
            "attention_q8_0_routed_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "attention_q8_0_routed_batched",
            [n_heads as u32, batch_size as u32, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(k_ptrs_ptr);
                b.push_ptr(v_ptrs_ptr);
                b.push_ptr(out_ptr);
                b.push_ptr(row_session_indices_ptr);
                b.push_ptr(pos_ptr);
                b.push_i32(ptr_stride);
                b.push_i32(layer);
                b.push_i32(nh);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(ms);
                b.push_f32(sc);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Routed batched attention with a KVarN K cache + Q8_0 V (microbatching).
    /// Per-row session selects its caches from session-major pointer tables
    /// (`rec_ptrs` = 4-bit K records, `win_ptrs` = f32 recent window, `v_ptrs` =
    /// Q8_0 V); each row's `n_full_blocks` is derived from `positions[row]`.
    /// Mirrors `attention_q8_0_routed_batched`; K dequant is in place.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kvarn_routed_batched(
        &mut self,
        q: &GpuTensor,
        rec_ptrs: &GpuTensor,
        win_ptrs: &GpuTensor,
        v_ptrs: &GpuTensor,
        out: &GpuTensor,
        row_session_indices: &GpuTensor,
        positions: &GpuTensor,
        ptr_layer_stride: usize,
        layer_index: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_kvarn_routed_batched",
            kernels::ATTENTION_KVARN_ROUTED_BATCHED_SRC,
            "attention_kvarn_routed_batched",
        )?;
        const GROUP: usize = 128;
        let rec_bytes = (head_dim * GROUP).div_ceil(2) + head_dim * 2 * 2 + GROUP * 2;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let q_ptr = q.buf.as_ptr();
        let rec_ptrs_ptr = rec_ptrs.buf.as_ptr();
        let win_ptrs_ptr = win_ptrs.buf.as_ptr();
        let v_ptrs_ptr = v_ptrs.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let rsi_ptr = row_session_indices.buf.as_ptr();
        let pos_ptr = positions.buf.as_ptr();
        let ptr_stride = ptr_layer_stride as i32;
        let layer = layer_index as i32;
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let ms = max_seq as i32;
        let sc = scale;
        let rb = rec_bytes as i32;
        let gp = GROUP as i32;
        let mut params: Vec<*mut c_void> = vec![
            &q_ptr as *const _ as *mut c_void,
            &rec_ptrs_ptr as *const _ as *mut c_void,
            &win_ptrs_ptr as *const _ as *mut c_void,
            &v_ptrs_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &rsi_ptr as *const _ as *mut c_void,
            &pos_ptr as *const _ as *mut c_void,
            &ptr_stride as *const _ as *mut c_void,
            &layer as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &ms as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
            &rb as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
        ];
        let block_size = (max_ctx_len.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        let shared_mem = ((max_ctx_len + block_size as usize + head_dim) * 4) as u32;
        self.launch_maybe_blob(
            "attention_kvarn_routed_batched",
            [n_heads as u32, batch_size as u32, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut bb = hip_bridge::KernargBlob::new();
                bb.push_ptr(q_ptr);
                bb.push_ptr(rec_ptrs_ptr);
                bb.push_ptr(win_ptrs_ptr);
                bb.push_ptr(v_ptrs_ptr);
                bb.push_ptr(out_ptr);
                bb.push_ptr(rsi_ptr);
                bb.push_ptr(pos_ptr);
                bb.push_i32(ptr_stride);
                bb.push_i32(layer);
                bb.push_i32(nh);
                bb.push_i32(nkv);
                bb.push_i32(hd);
                bb.push_i32(ms);
                bb.push_f32(sc);
                bb.push_i32(rb);
                bb.push_i32(gp);
                bb
            },
        )
    }
    /// FP32 causal attention specialized for GQA groups where four query heads
    /// share one KV head. This is a full-precision KLD prefill fast path: it
    /// preserves FP32 score/output arithmetic and only reduces redundant K/V
    /// traffic across grouped query heads.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_f32_batched_gqa4(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx_len: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            n_kv_heads > 0 && n_heads % n_kv_heads == 0,
            "attention_f32_batched_gqa4: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        let kv_group = n_heads / n_kv_heads;
        assert!(
            kv_group >= 4 && kv_group % 4 == 0,
            "attention_f32_batched_gqa4: GQA group {kv_group} must be a multiple of 4",
        );
        self.ensure_kernel(
            "attention_f32_batched_gqa4",
            kernels::ATTENTION_F32_BATCHED_SRC,
            "attention_f32_batched_gqa4",
        )?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let q_ptr = q.buf.as_ptr();
        let k_ptr = k_cache.buf.as_ptr();
        let v_ptr = v_cache.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let pos_ptr = positions.buf.as_ptr();
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let mcl = max_ctx_len as i32;
        let sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &q_ptr as *const _ as *mut c_void,
            &k_ptr as *const _ as *mut c_void,
            &v_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &pos_ptr as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &mcl as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
        ];
        let block_size = (max_ctx_len.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        let shared_floats = 4usize * max_ctx_len + 4usize * block_size as usize + 4usize * head_dim;
        let shared_mem = (shared_floats * 4) as u32;
        assert!(
            shared_mem <= 64 * 1024,
            "attention_f32_batched_gqa4: shared memory {shared_mem} exceeds 64 KiB",
        );
        let grid_x = n_kv_heads * (kv_group / 4);
        let bytes =
            crate::profile::attention_f32_kv_bytes(n_heads, n_kv_heads, head_dim, max_ctx_len)
                * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "attention",
            "attention_f32_batched_gqa4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "attention_f32_batched_gqa4",
            [grid_x as u32, batch_size as u32, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(k_ptr);
                b.push_ptr(v_ptr);
                b.push_ptr(out_ptr);
                b.push_ptr(pos_ptr);
                b.push_i32(nh);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(mcl);
                b.push_f32(sc);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Tree-mask variant of `attention_q8_0_kv_batched`. When `tree_bias` is
    /// `Some`, the kernel ignores the causal cutoff and iterates over
    /// `[0, block_start + block_cols)`, applying an additive bias from
    /// `tree_bias[b × block_cols + (t - block_start)]` for in-block keys.
    /// Caller passes `-inf` on non-ancestor slots and `0.0` on ancestors
    /// (see `hipfire_runtime::ddtree::linearize_tree`).
    ///
    /// When `tree_bias` is `None`, `block_start` / `block_cols` are ignored
    /// and behavior is byte-identical to the legacy causal path.
    ///
    /// Shared memory: the tree-mode `seq_len` is always `block_start +
    /// block_cols`. Caller must pass `max_ctx_len` ≥ that value so the
    /// scores[] LDS slice is sized correctly.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_q8_0_kv_batched_masked(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_q8_0_kv_batched",
            kernels::ATTENTION_Q8_0_KV_BATCHED_SRC,
            "attention_q8_0_kv_batched",
        )?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = positions.buf.as_ptr();
        // tree_bias = null when None; the kernel branches on bias != nullptr.
        let mut bias_ptr: *mut std::ffi::c_void = match tree_bias {
            Some(t) => t.buf.as_ptr(),
            None => std::ptr::null_mut(),
        };
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut bs = block_start as i32;
        let mut bc = block_cols as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut bias_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut bc as *mut _ as *mut c_void,
        ];
        let block_size = (max_ctx_len.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        // Shared memory must accommodate the LARGEST batch row's seq_len for
        // scores[], plus nthreads workspace and head_dim q_shared.
        let shared_mem = ((max_ctx_len + block_size as usize + head_dim) * 4) as u32;
        let bytes =
            crate::profile::attention_q8_0_kv_bytes(n_heads, n_kv_heads, head_dim, max_ctx_len)
                * batch_size;
        let timer =
            crate::profile::begin_timer(&self.hip, "attention", "attention_q8_0_kv_batched", bytes);
        let bias_raw = bias_ptr; // alias for move into closure
        let result = self.launch_maybe_blob(
            "attention_q8_0_kv_batched",
            [n_heads as u32, batch_size as u32, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(k_ptr);
                b.push_ptr(v_ptr);
                b.push_ptr(out_ptr);
                b.push_ptr(pos_ptr);
                b.push_ptr(bias_raw);
                b.push_i32(nh);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(ms);
                b.push_f32(sc);
                b.push_i32(bs);
                b.push_i32(bc);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched flash attention for Q8_0 KV cache.
    ///
    /// This is the no-LDS-cap replacement for the old per-position
    /// `attention_flash_q8_0` loop in long-context prefill. The Q8 tile kernel
    /// shares the asym-family batched launcher ABI; the cos/sin slots are
    /// ignored by the kernel, so `q` is passed as a harmless non-null tensor.
    ///
    /// TODO: route `HIPFIRE_Q8_TOKPAR` / `HIPFIRE_Q8_DP4A*` benchmark variants
    /// through their specialized kernels. This wrapper intentionally restores
    /// the production tile-batched path first so examples and no-GPU CI compile.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_q8_0_batched_masked(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_q8_0_tile_batched",
            kernels::ATTENTION_FLASH_Q8_0_TILE_BATCHED_SRC,
            "attention_flash_q8_0_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            q,
            q,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            tree_bias,
            block_start,
            block_cols,
        )
    }
    /// Flash attention with an f16 K cache + Q8_0 V cache (KVarN v1 read path).
    /// Same tile+reduce machinery as `attention_flash_q8_0_batched_masked`, but
    /// `k_cache` is a token-major `[max_seq × kv_dim]` f16 shadow (materialized
    /// by `kvarn_build_kcache`) read directly instead of dequantizing Q8 blocks.
    /// V stays Q8_0. cos/sin are unused (K unrotated) — `q` is passed for them
    /// to satisfy the shared dispatcher ABI.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_f16k_q8v_batched_masked(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_f16k_q8v_tile_batched",
            kernels::ATTENTION_FLASH_F16K_Q8V_TILE_BATCHED_SRC,
            "attention_flash_f16k_q8v_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            q,
            q,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            tree_bias,
            block_start,
            block_cols,
        )
    }
    /// Fused KVarN flash (Phase D2): like `attention_flash_f16k_q8v_batched_masked`
    /// but Phase A dequants the 4-bit K records IN PLACE (`records` =
    /// `k_gpu[layer]`) for the `n_full_blocks` full tiles + reads the f32
    /// `window` for the trailing partial tile — no f16 shadow K, no build pass.
    /// V stays Q8_0 (`v_cache`). Tile+reduce path; shares the asym reduce kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_kvarn_batched_masked(
        &mut self,
        q: &GpuTensor,
        records: &GpuTensor,
        window: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
        n_full_blocks: usize,
        rec_bytes: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128; // == KVARN_GROUP
        let max_tiles = max_ctx_len.div_ceil(TILE_SIZE);
        let stride = 2 + head_dim;
        let per_pos_bytes = n_heads * max_tiles * stride * 4;
        let partials_capacity = partials.numel() * 4;
        let sub_batch = if per_pos_bytes > 0 {
            (partials_capacity / per_pos_bytes).max(1).min(batch_size)
        } else {
            batch_size
        };
        self.ensure_kernel(
            "attention_flash_kvarn_tile_batched",
            kernels::ATTENTION_FLASH_KVARN_TILE_BATCHED_SRC,
            "attention_flash_kvarn_tile_batched",
        )?;
        self.ensure_kernel(
            "attention_flash_asym_reduce_batched",
            kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC,
            "attention_flash_asym_reduce_batched",
        )?;
        let q_dim = n_heads * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut offset = 0usize;
        while offset < batch_size {
            let chunk = (batch_size - offset).min(sub_batch);
            {
                let q_ptr =
                    unsafe { (q.buf.as_ptr() as *mut u8).add(offset * q_dim * 4) as *mut c_void };
                let rec_ptr = records.buf.as_ptr();
                let win_ptr = window.buf.as_ptr();
                let v_ptr = v_cache.buf.as_ptr();
                let p_ptr = partials.buf.as_ptr();
                let pos_ptr = positions.buf.as_ptr();
                let bias_ptr: *mut std::ffi::c_void = match tree_bias {
                    Some(t) => t.buf.as_ptr(),
                    None => std::ptr::null_mut(),
                };
                let nh = n_heads as i32;
                let nkv = n_kv_heads as i32;
                let hd = head_dim as i32;
                let ms = max_seq as i32;
                let sc = scale;
                let ts = TILE_SIZE as i32;
                let mt = max_tiles as i32;
                let bo = offset as i32;
                let bs = block_start as i32;
                let bc = block_cols as i32;
                let nfb = n_full_blocks as i32;
                let rb = rec_bytes as i32;
                let mut params: Vec<*mut c_void> = vec![
                    &q_ptr as *const _ as *mut c_void,
                    &rec_ptr as *const _ as *mut c_void,
                    &win_ptr as *const _ as *mut c_void,
                    &v_ptr as *const _ as *mut c_void,
                    &p_ptr as *const _ as *mut c_void,
                    &pos_ptr as *const _ as *mut c_void,
                    &bias_ptr as *const _ as *mut c_void,
                    &nh as *const _ as *mut c_void,
                    &nkv as *const _ as *mut c_void,
                    &hd as *const _ as *mut c_void,
                    &ms as *const _ as *mut c_void,
                    &sc as *const _ as *mut c_void,
                    &ts as *const _ as *mut c_void,
                    &mt as *const _ as *mut c_void,
                    &bo as *const _ as *mut c_void,
                    &bs as *const _ as *mut c_void,
                    &bc as *const _ as *mut c_void,
                    &nfb as *const _ as *mut c_void,
                    &rb as *const _ as *mut c_void,
                ];
                self.launch_maybe_blob(
                    "attention_flash_kvarn_tile_batched",
                    [n_heads as u32, max_tiles as u32, chunk as u32],
                    [32, 1, 1],
                    (TILE_SIZE * 4) as u32,
                    &mut params,
                    || {
                        let mut b = hip_bridge::KernargBlob::new();
                        b.push_ptr(q_ptr);
                        b.push_ptr(rec_ptr);
                        b.push_ptr(win_ptr);
                        b.push_ptr(v_ptr);
                        b.push_ptr(p_ptr);
                        b.push_ptr(pos_ptr);
                        b.push_ptr(bias_ptr);
                        b.push_i32(nh);
                        b.push_i32(nkv);
                        b.push_i32(hd);
                        b.push_i32(ms);
                        b.push_f32(sc);
                        b.push_i32(ts);
                        b.push_i32(mt);
                        b.push_i32(bo);
                        b.push_i32(bs);
                        b.push_i32(bc);
                        b.push_i32(nfb);
                        b.push_i32(rb);
                        b
                    },
                )?;
            }
            {
                let p_ptr = partials.buf.as_ptr();
                let o_ptr =
                    unsafe { (out.buf.as_ptr() as *mut u8).add(offset * q_dim * 4) as *mut c_void };
                let pos_ptr = positions.buf.as_ptr();
                let nh = n_heads as i32;
                let hd = head_dim as i32;
                let ts = TILE_SIZE as i32;
                let mt = max_tiles as i32;
                let bo = offset as i32;
                let bs = block_start as i32;
                let bc = block_cols as i32;
                let mut params: Vec<*mut c_void> = vec![
                    &p_ptr as *const _ as *mut c_void,
                    &o_ptr as *const _ as *mut c_void,
                    &pos_ptr as *const _ as *mut c_void,
                    &nh as *const _ as *mut c_void,
                    &hd as *const _ as *mut c_void,
                    &ts as *const _ as *mut c_void,
                    &mt as *const _ as *mut c_void,
                    &bo as *const _ as *mut c_void,
                    &bs as *const _ as *mut c_void,
                    &bc as *const _ as *mut c_void,
                ];
                self.launch_maybe_blob(
                    "attention_flash_asym_reduce_batched",
                    [n_heads as u32, chunk as u32, 1],
                    [32, 1, 1],
                    0,
                    &mut params,
                    || {
                        let mut b = hip_bridge::KernargBlob::new();
                        b.push_ptr(p_ptr);
                        b.push_ptr(o_ptr);
                        b.push_ptr(pos_ptr);
                        b.push_i32(nh);
                        b.push_i32(hd);
                        b.push_i32(ts);
                        b.push_i32(mt);
                        b.push_i32(bo);
                        b.push_i32(bs);
                        b.push_i32(bc);
                        b
                    },
                )?;
            }
            offset += chunk;
        }
        Ok(())
    }
    /// Flash attention with Q8_0 KV cache — tile + reduce two-kernel path.
    /// Tiles seq_len into chunks of `tile_size`, launches [n_heads, n_tiles]
    /// blocks for the tile kernel, then [n_heads] blocks for the reduce.
    /// Requires a pre-allocated `partials` buffer of size
    /// n_heads * max_tiles * (2 + head_dim) floats.
    pub fn attention_flash_q8_0(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        // Graph-safe: use max_tiles so the grid is position-independent.
        // The tile kernel exits early for tiles beyond actual seq_len.
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        // For profiling / non-graph code paths, the actual tile count:
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.capture_mode {
            max_tiles
        } else {
            actual_tiles
        };

        // ── Tile kernel ──
        self.ensure_kernel(
            "attention_flash_q8_0_tile",
            kernels::ATTENTION_FLASH_Q8_0_TILE_SRC,
            "attention_flash_q8_0_tile",
        )?;
        {
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let q_ptr = q.buf.as_ptr();
            let k_ptr = k_cache.buf.as_ptr();
            let v_ptr = v_cache.buf.as_ptr();
            let p_ptr = partials.buf.as_ptr();
            let pos_ptr = pos_buf.as_ptr();
            let nh = n_heads as i32;
            let nkv = n_kv_heads as i32;
            let hd = head_dim as i32;
            let ms = max_seq as i32;
            let sc = scale;
            let ts = TILE_SIZE as i32;
            let grid = [n_heads as u32, launch_tiles as u32, 1];
            let shared = ((TILE_SIZE + head_dim) * 4) as u32;
            let mut params: Vec<*mut c_void> = vec![
                &q_ptr as *const _ as *mut c_void,
                &k_ptr as *const _ as *mut c_void,
                &v_ptr as *const _ as *mut c_void,
                &p_ptr as *const _ as *mut c_void,
                &pos_ptr as *const _ as *mut c_void,
                &nh as *const _ as *mut c_void,
                &nkv as *const _ as *mut c_void,
                &hd as *const _ as *mut c_void,
                &ms as *const _ as *mut c_void,
                &sc as *const _ as *mut c_void,
                &ts as *const _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                "attention_flash_q8_0_tile",
                grid,
                [32, 1, 1],
                shared,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(q_ptr);
                    b.push_ptr(k_ptr);
                    b.push_ptr(v_ptr);
                    b.push_ptr(p_ptr);
                    b.push_ptr(pos_ptr);
                    b.push_i32(nh);
                    b.push_i32(nkv);
                    b.push_i32(hd);
                    b.push_i32(ms);
                    b.push_f32(sc);
                    b.push_i32(ts);
                    b
                },
            )?;
        }

        // ── Reduce kernel (reads seq_len from pos_buf, computes n_tiles) ──
        self.ensure_kernel(
            "attention_flash_q8_0_reduce",
            kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC,
            "attention_flash_q8_0_reduce",
        )?;
        {
            let p_ptr = partials.buf.as_ptr();
            let o_ptr = out.buf.as_ptr();
            let nh = n_heads as i32;
            let hd = head_dim as i32;
            let pos_ptr = pos_buf.as_ptr();
            let ts = TILE_SIZE as i32;
            let mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &p_ptr as *const _ as *mut c_void,
                &o_ptr as *const _ as *mut c_void,
                &nh as *const _ as *mut c_void,
                &hd as *const _ as *mut c_void,
                &pos_ptr as *const _ as *mut c_void,
                &ts as *const _ as *mut c_void,
                &mt as *const _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                "attention_flash_q8_0_reduce",
                [n_heads as u32, 1, 1],
                [32, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(p_ptr);
                    b.push_ptr(o_ptr);
                    b.push_i32(nh);
                    b.push_i32(hd);
                    b.push_ptr(pos_ptr);
                    b.push_i32(ts);
                    b.push_i32(mt);
                    b
                },
            )?;
        }
        Ok(())
    }
    /// Batched flash attention for asym4 (K 4-bit rotated + V Q8_0).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym4_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_flash_asym4_batched_masked(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            cos_theta,
            sin_theta,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            None,
            0,
            0,
        )
    }
    /// Tree-mask variant of `attention_flash_asym4_batched`. See
    /// `attention_q8_0_kv_batched_masked` and `ddtree::linearize_tree` for the
    /// bias layout. Passes `tree_bias` / `block_start` / `block_cols` into the
    /// tile + reduce kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym4_batched_masked(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_asym4_tile_batched",
            kernels::ATTENTION_FLASH_ASYM4_TILE_BATCHED_SRC,
            "attention_flash_asym4_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            cos_theta,
            sin_theta,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            tree_bias,
            block_start,
            block_cols,
        )
    }
    /// Batched flash attention for fwht4 (K FWHT-rotated 4-bit + V Q8_0).
    /// `signs1` and `signs2` occupy the same slots as cos_theta/sin_theta on
    /// the asym4 path — the helper passes them opaquely to the tile kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_fwht4_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_flash_fwht4_batched_masked(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            signs1,
            signs2,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            None,
            0,
            0,
            0,
        )
    }
    /// Tree-mask variant of `attention_flash_fwht4_batched`. Mirrors the asym4
    /// path one-for-one; the FA tile kernel is the only difference.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_fwht4_batched_masked(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
        _v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_fwht4_tile_batched",
            kernels::ATTENTION_FLASH_FWHT4_TILE_BATCHED_SRC,
            "attention_flash_fwht4_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            signs1,
            signs2,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            tree_bias,
            block_start,
            block_cols,
        )
    }
    /// Batched flash attention for asym2 (K 2-bit rotated + V Q8_0).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym2_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_asym2_tile_batched",
            kernels::ATTENTION_FLASH_ASYM2_TILE_BATCHED_SRC,
            "attention_flash_asym2_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            cos_theta,
            sin_theta,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            None,
            0,
            0,
        )
    }
    /// Batched flash attention for fwht2 (K FWHT-rotated 2-bit + V Q8_0).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_fwht2_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        _v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_fwht2_tile_batched",
            kernels::ATTENTION_FLASH_FWHT2_TILE_BATCHED_SRC,
            "attention_flash_fwht2_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            signs1,
            signs2,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            None,
            0,
            0,
        )
    }
    /// Batched flash attention for asym3 KV.
    /// Grid: [n_heads, max_tiles, sub_batch] tile + [n_heads, sub_batch] reduce,
    /// chunked by partials buffer capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym3_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_flash_asym3_batched_masked(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            cos_theta,
            sin_theta,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            None,
            0,
            0,
        )
    }
    /// Tree-mask variant of `attention_flash_asym3_batched`. asym3 is the
    /// default live KV path on 9B MQ4 — this is the primary target for
    /// DDTree batched verify on the hybrid arch.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym3_batched_masked(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_asym3_tile_batched",
            kernels::ATTENTION_FLASH_ASYM3_TILE_BATCHED_SRC,
            "attention_flash_asym3_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            cos_theta,
            sin_theta,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            tree_bias,
            block_start,
            block_cols,
        )
    }
    /// Batched flash attention for fwht3 (K FWHT-rotated 3-bit + V Q8_0).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_fwht3_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_flash_fwht3_batched_masked(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            signs1,
            signs2,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            None,
            0,
            0,
            0,
        )
    }
    /// Tree-mask variant of `attention_flash_fwht3_batched`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_fwht3_batched_masked(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
        _v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_fwht3_tile_batched",
            kernels::ATTENTION_FLASH_FWHT3_TILE_BATCHED_SRC,
            "attention_flash_fwht3_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            signs1,
            signs2,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            tree_bias,
            block_start,
            block_cols,
        )
    }
    /// Flash attention for asym3 KV (K at 3-bit rotated, V at Q8_0).
    /// Reuses Q8_0 flash reduce (output in normal space — V was un-rotated).
    /// Flash attention for fwht3 KV (K FWHT-rotated 3-bit, V at Q8_0).
    pub fn attention_flash_fwht3(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
        _v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.capture_mode {
            max_tiles
        } else {
            actual_tiles
        };

        self.ensure_givens4_kernel(
            "attention_flash_fwht3_tile",
            kernels::ATTENTION_FLASH_FWHT3_TILE_SRC,
            "attention_flash_fwht3_tile",
        )?;
        {
            let func = &self.functions["attention_flash_fwht3_tile"];
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let mut q_ptr = q.buf.as_ptr();
            let mut k_ptr = k_cache.buf.as_ptr();
            let mut v_ptr = v_cache.buf.as_ptr();
            let mut p_ptr = partials.buf.as_ptr();
            let mut pos_ptr = pos_buf.as_ptr();
            let mut s1_ptr = signs1.buf.as_ptr();
            let mut s2_ptr = signs2.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut ms = max_seq as i32;
            let mut sc = scale;
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut q_ptr as *mut _ as *mut c_void,
                &mut k_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut s1_ptr as *mut _ as *mut c_void,
                &mut s2_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut ms as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_heads as u32, launch_tiles as u32, 1],
                    [32, 1, 1],
                    (TILE_SIZE * 4) as u32,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

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
            let mut params: Vec<*mut c_void> = vec![
                &mut p_ptr as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
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
    pub fn attention_flash_asym3(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.capture_mode {
            max_tiles
        } else {
            actual_tiles
        };

        self.ensure_givens4_kernel(
            "attention_flash_asym3_tile",
            kernels::ATTENTION_FLASH_ASYM3_TILE_SRC,
            "attention_flash_asym3_tile",
        )?;
        {
            let func = &self.functions["attention_flash_asym3_tile"];
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let mut q_ptr = q.buf.as_ptr();
            let mut k_ptr = k_cache.buf.as_ptr();
            let mut v_ptr = v_cache.buf.as_ptr();
            let mut p_ptr = partials.buf.as_ptr();
            let mut pos_ptr = pos_buf.as_ptr();
            let mut ct_ptr = cos_theta.buf.as_ptr();
            let mut st_ptr = sin_theta.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut ms = max_seq as i32;
            let mut sc = scale;
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut q_ptr as *mut _ as *mut c_void,
                &mut k_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ct_ptr as *mut _ as *mut c_void,
                &mut st_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut ms as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_heads as u32, launch_tiles as u32, 1],
                    [32, 1, 1],
                    (TILE_SIZE * 4) as u32,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

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
            let mut params: Vec<*mut c_void> = vec![
                &mut p_ptr as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
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
    /// Flash attention for asym4 KV (K at rotated 4-bit, V at Q8_0 normal space).
    /// Reuses the Q8_0 flash reduce since V was un-rotated — no inverse rotation needed.
    pub fn attention_flash_asym4(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.capture_mode {
            max_tiles
        } else {
            actual_tiles
        };

        // Tile kernel
        self.ensure_givens4_kernel(
            "attention_flash_asym4_tile",
            kernels::ATTENTION_FLASH_ASYM4_TILE_SRC,
            "attention_flash_asym4_tile",
        )?;
        {
            let func = &self.functions["attention_flash_asym4_tile"];
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let mut q_ptr = q.buf.as_ptr();
            let mut k_ptr = k_cache.buf.as_ptr();
            let mut v_ptr = v_cache.buf.as_ptr();
            let mut p_ptr = partials.buf.as_ptr();
            let mut pos_ptr = pos_buf.as_ptr();
            let mut ct_ptr = cos_theta.buf.as_ptr();
            let mut st_ptr = sin_theta.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut ms = max_seq as i32;
            let mut sc = scale;
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut q_ptr as *mut _ as *mut c_void,
                &mut k_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ct_ptr as *mut _ as *mut c_void,
                &mut st_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut ms as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_heads as u32, launch_tiles as u32, 1],
                    [32, 1, 1],
                    (TILE_SIZE * 4) as u32, // scores[tile_size]
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

        // Reuse Q8_0 flash reduce (output already in normal space).
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
            let mut params: Vec<*mut c_void> = vec![
                &mut p_ptr as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
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
    /// Flash attention for fwht4 KV (K FWHT-rotated 4-bit, V at Q8_0 normal space).
    /// Same launch geometry + Q8_0 reduce as asym4 — only the tile kernel differs.
    pub fn attention_flash_fwht4(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
        _v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.capture_mode {
            max_tiles
        } else {
            actual_tiles
        };

        self.ensure_givens4_kernel(
            "attention_flash_fwht4_tile",
            kernels::ATTENTION_FLASH_FWHT4_TILE_SRC,
            "attention_flash_fwht4_tile",
        )?;
        {
            let func = &self.functions["attention_flash_fwht4_tile"];
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let mut q_ptr = q.buf.as_ptr();
            let mut k_ptr = k_cache.buf.as_ptr();
            let mut v_ptr = v_cache.buf.as_ptr();
            let mut p_ptr = partials.buf.as_ptr();
            let mut pos_ptr = pos_buf.as_ptr();
            let mut s1_ptr = signs1.buf.as_ptr();
            let mut s2_ptr = signs2.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut ms = max_seq as i32;
            let mut sc = scale;
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut q_ptr as *mut _ as *mut c_void,
                &mut k_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut s1_ptr as *mut _ as *mut c_void,
                &mut s2_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut ms as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_heads as u32, launch_tiles as u32, 1],
                    [32, 1, 1],
                    (TILE_SIZE * 4) as u32,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

        // Reuse Q8_0 flash reduce (output already in normal space, same as asym4).
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
            let mut params: Vec<*mut c_void> = vec![
                &mut p_ptr as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
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
    /// Flash attention for asym2 KV (K at rotated 2-bit, V at Q8_0 normal space).
    /// Flash attention for fwht2 KV (K FWHT-rotated 2-bit, V at Q8_0).
    pub fn attention_flash_fwht2(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
        _v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.capture_mode {
            max_tiles
        } else {
            actual_tiles
        };

        self.ensure_givens4_kernel(
            "attention_flash_fwht2_tile",
            kernels::ATTENTION_FLASH_FWHT2_TILE_SRC,
            "attention_flash_fwht2_tile",
        )?;
        {
            let func = &self.functions["attention_flash_fwht2_tile"];
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let mut q_ptr = q.buf.as_ptr();
            let mut k_ptr = k_cache.buf.as_ptr();
            let mut v_ptr = v_cache.buf.as_ptr();
            let mut p_ptr = partials.buf.as_ptr();
            let mut pos_ptr = pos_buf.as_ptr();
            let mut s1_ptr = signs1.buf.as_ptr();
            let mut s2_ptr = signs2.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut ms = max_seq as i32;
            let mut sc = scale;
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut q_ptr as *mut _ as *mut c_void,
                &mut k_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut s1_ptr as *mut _ as *mut c_void,
                &mut s2_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut ms as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_heads as u32, launch_tiles as u32, 1],
                    [32, 1, 1],
                    (TILE_SIZE * 4) as u32,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

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
            let mut params: Vec<*mut c_void> = vec![
                &mut p_ptr as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
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
    pub fn attention_flash_asym2(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.capture_mode {
            max_tiles
        } else {
            actual_tiles
        };

        self.ensure_givens4_kernel(
            "attention_flash_asym2_tile",
            kernels::ATTENTION_FLASH_ASYM2_TILE_SRC,
            "attention_flash_asym2_tile",
        )?;
        {
            let func = &self.functions["attention_flash_asym2_tile"];
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let mut q_ptr = q.buf.as_ptr();
            let mut k_ptr = k_cache.buf.as_ptr();
            let mut v_ptr = v_cache.buf.as_ptr();
            let mut p_ptr = partials.buf.as_ptr();
            let mut pos_ptr = pos_buf.as_ptr();
            let mut ct_ptr = cos_theta.buf.as_ptr();
            let mut st_ptr = sin_theta.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut ms = max_seq as i32;
            let mut sc = scale;
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut q_ptr as *mut _ as *mut c_void,
                &mut k_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ct_ptr as *mut _ as *mut c_void,
                &mut st_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut ms as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_heads as u32, launch_tiles as u32, 1],
                    [32, 1, 1],
                    (TILE_SIZE * 4) as u32,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

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
            let mut params: Vec<*mut c_void> = vec![
                &mut p_ptr as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
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
    /// Attention with Q8_0 quantized KV cache.
    pub fn attention_q8_0_kv(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx1103 (Phoenix): no-LDS flash-decode variant (one wave32 per head,
        // register-resident online softmax). Removes the `scores[seq_len]` LDS
        // buffer and its context-length ceiling. Needs head_dim % 32 == 0 for
        // the per-lane block layout; otherwise fall through to the generic path.
        let use_gfx1103 = self.arch_caps.is_gfx1103() && head_dim % 32 == 0;
        let (module, src, kname) = if use_gfx1103 {
            (
                "attention_q8_0_kv_gfx1103",
                kernels::ATTENTION_Q8_0_KV_GFX1103_SRC,
                "attention_q8_0_kv_gfx1103",
            )
        } else {
            (
                "attention_q8_0_kv",
                kernels::ATTENTION_Q8_0_KV_SRC,
                "attention_q8_0_kv",
            )
        };
        self.ensure_kernel(module, src, kname)?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        // hipGraph capture: `block_size` and `shared_mem` are launch-time host
        // scalars baked into the captured node. They are sized by `seq_len_hint`
        // (= current position + 1) on the direct path, which would lock a
        // captured graph to its capture position and under-allocate scores[] on
        // replay at a later position. Under capture, size both to `max_seq` so
        // ONE captured graph replays correctly at EVERY later position: the
        // kernel recomputes `seq_len = pos_buf[0]+1` from the DEVICE pos buffer
        // (updated direct before each replay) and self-adjusts its internal
        // scores[]/q_sh offsets — only the *allocated* shared-mem must be large
        // enough, and `max_seq` always is. block_size strides safely at any
        // nthreads, so the larger capture value stays correct. Mirrors the
        // `if self.capture_mode { max_tiles }` pattern used elsewhere here.
        let sizing_seq = if self.capture_mode {
            max_seq
        } else {
            seq_len_hint
        };
        // gfx1103 no-LDS variant: one wave32 per head, zero shared memory, and
        // no seq-len-dependent launch sizing (so it captures at any position).
        let (block_size, shared_mem) = if use_gfx1103 {
            (32u32, 0u32)
        } else {
            let block_size = (sizing_seq.max(head_dim) as u32)
                .next_power_of_two()
                .min(256);
            // Extra shared mem for Q head vector preloaded into shared memory
            let shared_mem = ((sizing_seq + block_size as usize + head_dim) * 4) as u32;
            (block_size, shared_mem)
        };
        let bytes =
            crate::profile::attention_q8_0_kv_bytes(n_heads, n_kv_heads, head_dim, seq_len_hint);
        let timer = crate::profile::begin_timer(&self.hip, "attention", "attention_q8_0_kv", bytes);
        let result = self.launch_maybe_blob(
            kname,
            [n_heads as u32, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(k_ptr);
                b.push_ptr(v_ptr);
                b.push_ptr(out_ptr);
                b.push_ptr(pos_ptr);
                b.push_i32(nh);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(ms);
                b.push_f32(sc);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Phase-instrumented variant of attention_q8_0_kv. Identical to the
    /// baseline kernel but additionally writes per-head cycle counts for
    /// each internal phase into `cycle_counts` (layout: [n_heads * 3],
    /// per-head order = phase1(QK^T), phase2(softmax), phase3(V-weighted)).
    ///
    /// Uses __builtin_amdgcn_s_memrealtime() which returns a wall-clock
    /// counter. On gfx1100 the tick rate is approximately 1e8 Hz (10 ns
    /// per tick); confirm empirically by comparing against the kernel's
    /// total elapsed time from event timing.
    ///
    /// Use only for diagnostic profiling — the memrealtime reads serialize
    /// execution and inflate total time slightly.
    pub fn attention_q8_0_kv_timed(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        cycle_counts: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_q8_0_kv_timed",
            kernels::ATTENTION_Q8_0_KV_TIMED_SRC,
            "attention_q8_0_kv_timed",
        )?;
        let func = &self.functions["attention_q8_0_kv_timed"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut cc_ptr = cycle_counts.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cc_ptr as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        let shared_mem = ((seq_len_hint + block_size as usize + head_dim) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// TriAttention importance scoring over a Q8 post-RoPE K cache.
    /// Produces one score per cached position per query head; caller picks
    /// top-B for eviction (see arXiv:2604.04921 §4).
    ///
    /// `centers`: `[n_heads × n_bands × 3]` float32 packed as
    /// `(Re(E[q_f]), Im(E[q_f]), E[||q_f||])`. `scores`: `[n_heads × seq_len]`
    /// float32 output. One block per (pos, head); 32 threads reduce across
    /// the head's frequency bands.
    pub fn triattn_score_q8(
        &mut self,
        k_cache: &GpuTensor,
        centers: &GpuTensor,
        scores: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        p_q: f32,
        seq_len: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "triattn_score_q8",
            kernels::TRIATTN_SCORE_Q8_SRC,
            "triattn_score_q8",
        )?;
        let func = &self.functions["triattn_score_q8"];
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut c_ptr = centers.buf.as_ptr();
        let mut s_ptr = scores.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut nr = n_rot as i32;
        let mut th = rope_theta;
        let mut pq = p_q;
        let mut sl = seq_len as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut k_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut s_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut th as *mut _ as *mut c_void,
            &mut pq as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [seq_len as u32, n_heads as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// TriAttention importance scoring over an asym2 post-RoPE K cache.
    /// Same shape as `triattn_score_asym3` but reads the 2-bit packed
    /// layout (4 indices per byte) and the TURBO_C2_256 codebook.
    pub fn triattn_score_asym2(
        &mut self,
        k_cache: &GpuTensor,
        centers: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        scores: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        p_q: f32,
        seq_len: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "triattn_score_asym2",
            kernels::TRIATTN_SCORE_ASYM2_SRC,
            "triattn_score_asym2",
        )?;
        let func = &self.functions["triattn_score_asym2"];
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut c_ptr = centers.buf.as_ptr();
        let mut ct_ptr = cos_theta.buf.as_ptr();
        let mut st_ptr = sin_theta.buf.as_ptr();
        let mut s_ptr = scores.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut nr = n_rot as i32;
        let mut th = rope_theta;
        let mut pq = p_q;
        let mut sl = seq_len as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut k_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut ct_ptr as *mut _ as *mut c_void,
            &mut st_ptr as *mut _ as *mut c_void,
            &mut s_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut th as *mut _ as *mut c_void,
            &mut pq as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [seq_len as u32, n_heads as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// TriAttention importance scoring over an asym4 post-RoPE K cache.
    /// Same shape as `triattn_score_asym3` but reads the 4-bit nibble
    /// layout and the TURBO_C4 codebook.
    pub fn triattn_score_asym4(
        &mut self,
        k_cache: &GpuTensor,
        centers: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        scores: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        p_q: f32,
        seq_len: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "triattn_score_asym4",
            kernels::TRIATTN_SCORE_ASYM4_SRC,
            "triattn_score_asym4",
        )?;
        let func = &self.functions["triattn_score_asym4"];
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut c_ptr = centers.buf.as_ptr();
        let mut ct_ptr = cos_theta.buf.as_ptr();
        let mut st_ptr = sin_theta.buf.as_ptr();
        let mut s_ptr = scores.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut nr = n_rot as i32;
        let mut th = rope_theta;
        let mut pq = p_q;
        let mut sl = seq_len as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut k_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut ct_ptr as *mut _ as *mut c_void,
            &mut st_ptr as *mut _ as *mut c_void,
            &mut s_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut th as *mut _ as *mut c_void,
            &mut pq as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [seq_len as u32, n_heads as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// TriAttention importance scoring over an asym3 post-RoPE K cache.
    /// Same contract as `triattn_score_q8` but reads asym3's Givens-rotated
    /// 3-bit layout and applies the inverse Givens rotation on the fly to
    /// recover post-RoPE K per band.
    pub fn triattn_score_asym3(
        &mut self,
        k_cache: &GpuTensor,
        centers: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        scores: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        p_q: f32,
        seq_len: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "triattn_score_asym3",
            kernels::TRIATTN_SCORE_ASYM3_SRC,
            "triattn_score_asym3",
        )?;
        let func = &self.functions["triattn_score_asym3"];
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut c_ptr = centers.buf.as_ptr();
        let mut ct_ptr = cos_theta.buf.as_ptr();
        let mut st_ptr = sin_theta.buf.as_ptr();
        let mut s_ptr = scores.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut nr = n_rot as i32;
        let mut th = rope_theta;
        let mut pq = p_q;
        let mut sl = seq_len as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut k_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut ct_ptr as *mut _ as *mut c_void,
            &mut st_ptr as *mut _ as *mut c_void,
            &mut s_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut th as *mut _ as *mut c_void,
            &mut pq as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [seq_len as u32, n_heads as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Attention with Q8 quantized KV cache.
    pub fn attention_q8kv(
        &mut self,
        q: &GpuTensor,
        k_cache_q8: &GpuTensor,
        v_cache_q8: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_q8kv",
            kernels::ATTENTION_Q8KV_SRC,
            "attention_q8kv",
        )?;
        let func = &self.functions["attention_q8kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache_q8.buf.as_ptr();
        let mut v_ptr = v_cache_q8.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Attention with quantized HFQ4 KV cache — dequantizes K/V on the fly.
    pub fn attention_q4kv(
        &mut self,
        q: &GpuTensor,
        k_cache_q4: &GpuTensor,
        v_cache_q4: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let use_gfx1103 = self.arch_caps.is_gfx1103() && head_dim % 32 == 0;
        let (module, src, kname) = if use_gfx1103 {
            (
                "attention_q4_affine_kv_gfx1103",
                kernels::ATTENTION_Q4_AFFINE_KV_GFX1103_SRC,
                "attention_q4kv_gfx1103",
            )
        } else {
            (
                "attention_q4kv",
                kernels::ATTENTION_Q4KV_SRC,
                "attention_q4kv",
            )
        };
        self.ensure_kernel(module, src, kname)?;
        let func = &self.functions[kname];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache_q4.buf.as_ptr();
        let mut v_ptr = v_cache_q4.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let (block_size, shared_mem) = if use_gfx1103 {
            (32u32, 0u32)
        } else {
            let block_size = (seq_len_hint.max(head_dim) as u32)
                .next_power_of_two()
                .min(256);
            (block_size, ((seq_len_hint + block_size as usize) * 4) as u32)
        };
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// PFlash per-block cosine-importance forward (fp32 training twin).
    /// `k`:`[n_pos*kv_dim]`, `scores`:`[n_blocks]`. score[b] = cosine(block_mean_K,
    /// last_token_K) over the full kv_dim (matches production `pflash_score_q8_kv`).
    #[allow(clippy::too_many_arguments)]
    pub fn pflash_score_f32_fwd(
        &mut self,
        k: &GpuTensor,
        scores: &GpuTensor,
        n_pos: usize,
        kv_dim: usize,
        block_size: usize,
        n_blocks: usize,
        last_pos: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "pflash_score_f32_fwd",
            kernels::PFLASH_SCORE_F32_TRAIN_SRC,
            "pflash_score_f32_fwd",
        )?;
        let func = &self.functions["pflash_score_f32_fwd"];
        let mut kp = k.buf.as_ptr();
        let mut sp = scores.buf.as_ptr();
        let mut a = n_pos as i32;
        let mut b = kv_dim as i32;
        let mut c = block_size as i32;
        let mut d = n_blocks as i32;
        let mut e = last_pos as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut a as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut c as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut e as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_blocks as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// PFlash importance backward (fp32 training twin). `dscores`:`[n_blocks]`,
    /// `dk`:`[n_pos*kv_dim]` — MUST be zeroed before the call (accumulated via
    /// atomics; the last-token K is shared across all blocks).
    #[allow(clippy::too_many_arguments)]
    pub fn pflash_score_f32_bwd(
        &mut self,
        k: &GpuTensor,
        dscores: &GpuTensor,
        dk: &GpuTensor,
        n_pos: usize,
        kv_dim: usize,
        block_size: usize,
        n_blocks: usize,
        last_pos: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "pflash_score_f32_bwd",
            kernels::PFLASH_SCORE_F32_TRAIN_SRC,
            "pflash_score_f32_bwd",
        )?;
        let func = &self.functions["pflash_score_f32_bwd"];
        let mut kp = k.buf.as_ptr();
        let mut dsp = dscores.buf.as_ptr();
        let mut dkp = dk.buf.as_ptr();
        let mut a = n_pos as i32;
        let mut b = kv_dim as i32;
        let mut c = block_size as i32;
        let mut d = n_blocks as i32;
        let mut e = last_pos as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kp as *mut _ as *mut c_void,
            &mut dsp as *mut _ as *mut c_void,
            &mut dkp as *mut _ as *mut c_void,
            &mut a as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut c as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut e as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_blocks as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Fused ViT self-attention: reads QKV [N, 3*hidden], writes out [N, hidden].
    pub fn vit_attention_f32(
        &mut self,
        qkv: &GpuTensor,
        out: &GpuTensor,
        n: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "vit_attention_f32",
            kernels::VIT_ATTENTION_SRC,
            "vit_attention_f32",
        )?;
        let func = &self.functions["vit_attention_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = qkv.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut ni = n as i32;
        let mut hi = hidden as i32;
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut hi as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = std::cmp::min(256, std::cmp::max(n, head_dim)) as u32;
        let block_size = block_size.next_power_of_two();
        // Shared memory: scores[N] + workspace[block_size]
        let shared_mem = ((n + block_size as usize) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [num_heads as u32, n as u32, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Optimized vision attention with tiled K/V loading and 4 queries per block.
    /// ~3-5x faster than vit_attention_f32 via shared memory reuse.
    /// Grid=[num_heads, ceil(N/4)], Block=[256].
    pub fn vit_attention_opt(
        &mut self,
        qkv: &GpuTensor,
        out: &GpuTensor,
        n: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "vit_attention_opt",
            kernels::VIT_ATTENTION_OPT_SRC,
            "vit_attention_opt",
        )?;
        let func = &self.functions["vit_attention_opt"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = qkv.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut ni = n as i32;
        let mut hi = hidden as i32;
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut hi as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let qpb = 2u32;
        let grid_y = ((n as u32 + qpb - 1) / qpb) as u32;
        // LDS: k_tile[K_TILE*head_dim] + scores[N] + ws[256] + q_sh[head_dim]
        let k_tile = 64u32;
        let shared_mem =
            (k_tile * head_dim as u32 * 4) + (n as u32 * 4) + (256 * 4) + (head_dim as u32 * 4);
        unsafe {
            self.hip.launch_kernel(
                func,
                [num_heads as u32, grid_y, 1],
                [256, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Generic bidirectional flash attention over a fused **bf16** qkv
    /// `[N, 3*hidden]` → f32 `out [N, hidden]`. Online softmax, f32 accumulation,
    /// no causal mask. The bf16 vision tower's attention (replaces the f32
    /// vit_attention_opt). `head_dim` must be ≤ 128 (the block width). See
    /// `kernels/src/flash_attn_bf16.hip`.
    pub fn flash_attn_bf16(
        &mut self,
        qkv_bf16: &GpuTensor,
        out: &GpuTensor,
        n: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            head_dim <= 128,
            "flash_attn_bf16: head_dim={head_dim} must be <= 128 (block width)"
        );
        self.ensure_kernel(
            "flash_attn_bf16",
            kernels::FLASH_ATTN_BF16_SRC,
            "flash_attn_bf16",
        )?;
        let func = &self.functions["flash_attn_bf16"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = qkv_bf16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut ni = n as i32;
        let mut hi = hidden as i32;
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut hi as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        // LDS: q_sh[head_dim] + red[BLK=128].
        let shared_mem = (head_dim as u32 + 128) * 4;
        unsafe {
            self.hip.launch_kernel(
                func,
                [num_heads as u32, n as u32, 1],
                [128, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DFlash draft cross-attention: `B` queries attend to `L` keys/values
    /// with NO causal mask (bidirectional). Supports GQA; `n_heads` must be
    /// a multiple of `n_kv_heads`. See `kernels/src/attention_dflash.hip`
    /// for the full contract.
    ///
    /// Layouts:
    ///   q : [B * n_heads    * head_dim]
    ///   k : [L * n_kv_heads * head_dim]
    ///   v : [L * n_kv_heads * head_dim]
    ///   out: [B * n_heads    * head_dim]
    pub fn attention_dflash_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_dflash_f32",
            kernels::ATTENTION_DFLASH_SRC,
            "attention_dflash_f32",
        )?;
        let func = &self.functions["attention_dflash_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // Tiled online-softmax (FlashAttention-style). LDS layout:
        //   tile_scores[tile_size] + ws[block_size] + out_run[head_dim]
        //
        // tile_size is chosen to keep LDS ≤ 56 KB (8 KB margin under gfx1100's
        // 64 KB hard limit for kernel launch overhead). Single-tile case
        // (l ≤ tile_size) is mathematically equivalent to the prior
        // single-pass softmax up to FP order; multi-tile carries (max, sum,
        // out) running state across tiles. Replaces the prior `scores[L]`
        // allocation that overflowed LDS at l > ~16128.
        let block_size = std::cmp::min(256, std::cmp::max(l, head_dim)) as u32;
        let block_size = block_size.next_power_of_two();
        const LDS_BUDGET_F32: usize = 14_336; // 56 KB / 4 bytes
        let fixed = block_size as usize + head_dim;
        let max_tile_room = LDS_BUDGET_F32.saturating_sub(fixed);
        let tile_size = std::cmp::min(l.max(1), max_tile_room.max(1));
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut ts = tile_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut ts as *mut _ as *mut c_void,
        ];
        let shared_mem = ((tile_size + block_size as usize + head_dim) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, b as u32, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// WMMA-accelerated FlashAttention-style non-causal attention for
    /// the **large-B / large-L** case. Same Q/K/V layout and contract as
    /// [`Self::attention_dflash_f32`] — drop-in replacement.
    ///
    /// Grid:  `[n_heads, ceil(B / 16), 1]` (one block per (head, 16-Q-tile))
    /// Block: 32 threads (1 wave32 warp)
    /// LDS:   `(32 * head_dim + 256 + 48) * 4` bytes
    ///        — ≈ 17 KB for `head_dim=128`, fits comfortably under the
    ///        64 KB RDNA3 budget.
    ///
    /// Intended for `B >= 16` and `head_dim` a multiple of 16. The
    /// caller is responsible for picking between this and the scalar
    /// `attention_dflash_f32` based on workload shape.
    pub fn attention_dflash_wmma_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // The gfx11 M16 WMMA kernel is not launch-stable on RDNA3.5 today;
        // keep gfx115x on the scalar tiled kernel until it has its own port.
        if self.arch_caps.is_rdna3p5() {
            return self.attention_dflash_f32(q, k, v, out, b, l, n_heads, n_kv_heads, head_dim);
        }
        assert!(
            head_dim % 16 == 0,
            "attention_dflash_wmma_f32: head_dim={head_dim} must be a multiple of 16 \
             (WMMA tiles K-axis in 16-element chunks)",
        );
        assert!(
            head_dim <= 256,
            "attention_dflash_wmma_f32: head_dim={head_dim} exceeds the 256 cap \
             — LDS budget is `3 * 16 * head_dim + 304` f32 slots, which overflows \
             the 64 KB RDNA3 wave32 limit above head_dim=256. Use \
             attention_dflash_f32 (scalar) for larger head_dim, or split this \
             kernel's LDS layout (drop Q_lds or O_lds) in a future variant.",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            "attention_dflash_wmma_f32",
            kernels::ATTENTION_DFLASH_WMMA_SRC,
            "attention_dflash_wmma_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // LDS layout (in f32 slots):
        //   Q_lds[16 * head_dim] + V_lds[16 * head_dim] + O_lds[16 * head_dim]
        //   + S_lds[16 * 16]
        //   + m_lds[16] + l_lds[16] + alpha_lds[16]
        let lds_f32 = 3 * 16 * head_dim + 16 * 16 + 16 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 15) / 16;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [32, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Causal text-prefill sibling of [`Self::attention_dflash_wmma_f32`].
    /// Uses direct Q/K/V tensors for a self-contained chunk instead of writing
    /// K/V to the runtime KV cache and reading them back through the scalar
    /// `attention_f32_batched` kernel. Intended for KLD reference generation
    /// where every chunk starts at position 0 and no decode follows.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_dflash_wmma_causal_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            head_dim % 16 == 0,
            "attention_dflash_wmma_causal_f32: head_dim={head_dim} must be a multiple of 16",
        );
        assert!(
            head_dim <= 256,
            "attention_dflash_wmma_causal_f32: head_dim={head_dim} exceeds the 256 cap",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_causal_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            "attention_dflash_wmma_f32",
            kernels::ATTENTION_DFLASH_WMMA_SRC,
            "attention_dflash_wmma_f32",
        )?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let lds_f32 = 3 * 16 * head_dim + 16 * 16 + 16 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut causal = 1i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut causal as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 15) / 16;
        let bytes = b * n_heads * head_dim * 8 + l * n_kv_heads * head_dim * 8;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "attention",
            "attention_dflash_wmma_causal_f32",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "attention_dflash_wmma_f32",
            [n_heads as u32, q_tiles as u32, 1],
            [32, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(qp);
                blob.push_ptr(kp);
                blob.push_ptr(vp);
                blob.push_ptr(op);
                blob.push_i32(bi);
                blob.push_i32(li);
                blob.push_i32(nh);
                blob.push_i32(nkv);
                blob.push_i32(hd);
                blob.push_f32(sc);
                blob.push_i32(causal);
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// FlashAttention-style WMMA with M=32 query tile (vs M=16 in
    /// `attention_dflash_wmma_f32`). Two waves per block; doubles the
    /// queries served per K-tile load, halving global-memory K
    /// traffic at vision-encoder shapes (large B, large L, head_dim ≤
    /// 128). Same head_dim ≤ 128 ceiling here — LDS budget is
    /// `(2*32 + 16) * head_dim + 32*16 + 96` f32 slots = 43 KB at
    /// hd=128, which is the largest tile that fits the 64 KB RDNA3
    /// wave32 SLM cap with full Q_lds + O_lds + V_lds.
    ///
    /// Caller responsibility: dispatch this when `B >= 32` AND
    /// `head_dim ≤ 128`; fall back to the M=16 variant or the scalar
    /// `attention_dflash_f32` otherwise.
    pub fn attention_dflash_wmma_m32_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            head_dim % 16 == 0,
            "attention_dflash_wmma_m32_f32: head_dim={head_dim} must be a multiple of 16",
        );
        assert!(
            head_dim <= 128,
            "attention_dflash_wmma_m32_f32: head_dim={head_dim} exceeds the 128 cap — \
             LDS budget at head_dim=160 is 53.4 KB and at head_dim=256 is 84 KB which \
             exceeds the 64 KB RDNA3 wave32 limit. Fall back to attention_dflash_wmma_f32 \
             (M=16) for larger head_dim.",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_m32_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            "attention_dflash_wmma_m32_f32",
            kernels::ATTENTION_DFLASH_WMMA_M32_SRC,
            "attention_dflash_wmma_m32_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_m32_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // LDS layout (in f32 slots):
        //   Q_lds[32 * head_dim] + V_lds[16 * head_dim] + O_lds[32 * head_dim]
        //   + S_lds[32 * 16]
        //   + m_lds[32] + l_lds[32] + alpha_lds[32]
        let lds_f32 = (2 * 32 + 16) * head_dim + 32 * 16 + 32 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 31) / 32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [64, 1, 1], // 2 waves
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// FlashAttention-style WMMA with M=32 query tile and **N=64 K-tile
    /// width** (vs N=16 in `attention_dflash_wmma_m32_f32`). Q lives in
    /// registers across all K-tiles within a block; phase C fuses the
    /// alpha-scale of O with the SV epilogue.
    ///
    /// Targets the vision-encoder regime (large B, large L,
    /// head_dim ≤ 128) where rocprof shows the M=32 baseline is
    /// per-tile-fixed-cost bound (1220 K-tile visits at N=16 → 305 at
    /// N=64 means 4× fewer syncs / softmax passes / O-scaling passes).
    ///
    /// LDS at hd=128: V_lds[64*128] + O_lds[32*128] + S_lds[32*64] +
    /// scalars = 57.7 KB (under 64 KB RDNA3 wave32 cap). VGPR per lane
    /// ≈ 130 (Q_frags + s_acc + scratch).
    ///
    /// Caller responsibility: dispatch when `head_dim % 32 == 0`,
    /// `head_dim ≤ 128`. Falls back to M=32 or M=16 otherwise.
    pub fn attention_dflash_wmma_n64_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_n64_f32: head_dim={head_dim} but this kernel is \
             hard-coded to head_dim==128. The dc loop is fully unrolled with d_chunks=8 \
             so Q_frags[] gets register-promoted instead of spilled to scratch — making \
             it variable would re-introduce the 544 B/lane private segment that defeats \
             the Q-in-registers optimization (the v1 attempt regressed +19%). Fall back \
             to attention_dflash_wmma_m32_f32 (head_dim <= 128) or attention_dflash_wmma_f32 \
             (head_dim <= 256) for other head dims.",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_n64_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            "attention_dflash_wmma_n64_f32",
            kernels::ATTENTION_DFLASH_WMMA_N64_SRC,
            "attention_dflash_wmma_n64_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_n64_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // LDS layout (in f32 slots):
        //   V_lds[64 * head_dim] + O_lds[32 * head_dim]
        //   + S_lds[32 * 64]
        //   + m_lds[32] + l_lds[32] + alpha_lds[32]
        let lds_f32 = (64 + 32) * head_dim + 32 * 64 + 32 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 31) / 32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [64, 1, 1], // 2 waves
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// FlashAttention-style WMMA, M=32 query tile and N=64 K-tile,
    /// with **K and V already stored as f16 in DRAM** (Q and output
    /// stay f32). Halves the attention kernel's DRAM traffic for K and
    /// V — the dominant cost on memory-bound vision-encoder shapes.
    /// Caller must cast K and V to f16 once (via `cast_f32_to_f16`)
    /// before invoking this kernel.
    ///
    /// Same head_dim==128 restriction as `attention_dflash_wmma_n64_f32`
    /// (Q_frags register-promotion requires the dc loop fully unrolled).
    pub fn attention_dflash_wmma_n64_f16kv_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            q.dtype,
            DType::F32,
            "attention_dflash_wmma_n64_f16kv_f32: q must be F32"
        );
        assert_eq!(
            k_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_n64_f16kv_f32: k must be F16"
        );
        assert_eq!(
            v_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_n64_f16kv_f32: v must be F16"
        );
        assert_eq!(
            out.dtype,
            DType::F32,
            "attention_dflash_wmma_n64_f16kv_f32: out must be F32"
        );
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_n64_f16kv_f32: head_dim={head_dim} but this kernel is \
             hard-coded to head_dim==128 (same constraint as the f32-K/V sibling — the dc \
             loop is fully unrolled with d_chunks=8 so Q_frags register-promotes).",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_n64_f16kv_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            "attention_dflash_wmma_n64_f16kv_f32",
            kernels::ATTENTION_DFLASH_WMMA_N64_F16KV_SRC,
            "attention_dflash_wmma_n64_f16kv_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_n64_f16kv_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // LDS layout same as the f32-K/V sibling: V_lds stays f32 so
        // phase C is byte-identical between the two kernels.
        let lds_f32 = (64 + 32) * head_dim + 32 * 64 + 32 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 31) / 32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [64, 1, 1], // 2 waves
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// FlashAttention WMMA, M=32 query tile and **N=128 K-tile**, K and
    /// V f16 in DRAM, V_lds and S_lds in f16. Same shape as
    /// `attention_dflash_wmma_n64_f16kv_f32` but twice the K-tile width.
    /// Halves outer-loop iterations → halves __syncthreads / softmax /
    /// alpha-scale overhead per attention call.
    ///
    /// LDS at hd=128 ≈ 56.4 KB: V_lds[128*128] f16 (32 KB) +
    /// O_lds[32*128] f32 (16 KB) + S_lds[32*128] f16 (8 KB) + scalars.
    pub fn attention_dflash_wmma_n128_f16kv_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            q.dtype,
            DType::F32,
            "attention_dflash_wmma_n128_f16kv_f32: q must be F32"
        );
        assert_eq!(
            k_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_n128_f16kv_f32: k must be F16"
        );
        assert_eq!(
            v_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_n128_f16kv_f32: v must be F16"
        );
        assert_eq!(
            out.dtype,
            DType::F32,
            "attention_dflash_wmma_n128_f16kv_f32: out must be F32"
        );
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_n128_f16kv_f32: head_dim={head_dim} but this kernel is \
             hard-coded to head_dim==128 (d_chunks=8 unroll for register-promoted Q_frags).",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_n128_f16kv_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            "attention_dflash_wmma_n128_f16kv_f32",
            kernels::ATTENTION_DFLASH_WMMA_N128_F16KV_SRC,
            "attention_dflash_wmma_n128_f16kv_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_n128_f16kv_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // LDS layout (in f32-equivalent slots — V_lds and S_lds are f16
        // so they take half the slot count of their nominal element
        // count):
        //   V_lds[128 * head_dim] f16     = 128 * head_dim / 2 f32 slots
        //   O_lds[32  * head_dim] f32     =  32 * head_dim     f32 slots
        //   S_lds[32  * 128]      f16     =  32 * 128 / 2      f32 slots
        //   m_lds + l_lds + alpha_lds     =  96                f32 slots
        let lds_f32 = (128 * head_dim) / 2 + 32 * head_dim + (32 * 128) / 2 + 32 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 31) / 32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [64, 1, 1], // 2 waves
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// FlashAttention WMMA, **M=64** query tile + N=128 K-tile, K/V f16
    /// in DRAM, V_lds and S_lds in f16, **O register-resident**.
    /// 4-wave block (128 threads). Halves the query-block count vs
    /// M=32, which halves K and V DRAM traffic per attention call —
    /// the dominant cost on this DRAM-bound workload.
    ///
    /// LDS at hd=128 ≈ 48.8 KB: V_lds[128*128] f16 + S_lds[64*128] f16
    /// + scalars. No O_lds — O lives in per-lane register arrays
    /// (8 float8_t = 64 VGPRs/lane in WMMA frag_c layout).
    pub fn attention_dflash_wmma_m64_n128_f16kv_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            q.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_f32: q must be F32"
        );
        assert_eq!(
            k_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_f32: k must be F16"
        );
        assert_eq!(
            v_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_f32: v must be F16"
        );
        assert_eq!(
            out.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_f32: out must be F32"
        );
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_m64_n128_f16kv_f32: head_dim={head_dim} but this kernel is \
             hard-coded to head_dim==128 (d_chunks=8 unroll, O_frags[8] register array).",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_m64_n128_f16kv_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            "attention_dflash_wmma_m64_n128_f16kv_f32",
            kernels::ATTENTION_DFLASH_WMMA_M64_N128_F16KV_SRC,
            "attention_dflash_wmma_m64_n128_f16kv_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_m64_n128_f16kv_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // LDS layout (in f32-equivalent slots; V_lds and S_lds are f16
        // so they take half their nominal element count):
        //   V_lds[128 * head_dim]  f16   = 128 * head_dim / 2 f32 slots
        //   S_lds[64  * 128]       f16   =  64 * 128 / 2      f32 slots
        //   m + l + alpha (64 each)      = 192                f32 slots
        let lds_f32 = (128 * head_dim) / 2 + (64 * 128) / 2 + 64 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 63) / 64;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [128, 1, 1], // 4 waves
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// V3 of `attention_dflash_wmma_m64_n128_f16kv_f32`. Same shape
    /// as v2 (M=64, N=128, 4-wave block, f16 K/V, O in registers,
    /// padded S_lds, cooperative softmax) but with phase C reordered
    /// to outer c / inner dc so each `a_reg_sm` row chunk is read
    /// once per c instead of once per (dc, c). 8× reduction in phase
    /// C S_lds reads.
    pub fn attention_dflash_wmma_m64_n128_f16kv_v3_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            q.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_v3_f32: q must be F32"
        );
        assert_eq!(
            k_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_v3_f32: k must be F16"
        );
        assert_eq!(
            v_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_v3_f32: v must be F16"
        );
        assert_eq!(
            out.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_v3_f32: out must be F32"
        );
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_m64_n128_f16kv_v3_f32: head_dim={head_dim} but this kernel is \
             hard-coded to head_dim==128.",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_m64_n128_f16kv_v3_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            "attention_dflash_wmma_m64_n128_f16kv_v3_f32",
            kernels::ATTENTION_DFLASH_WMMA_M64_N128_F16KV_V3_SRC,
            "attention_dflash_wmma_m64_n128_f16kv_v3_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_m64_n128_f16kv_v3_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // Same LDS layout as v2 (padded S_lds stride 130).
        let lds_f32 = (128 * head_dim) / 2 + (64 * 130) / 2 + 64 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 63) / 64;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [128, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Causal variant of `attention_dflash_wmma_m64_n128_f16kv_v3_f32`.
    /// Same tile shape (M=64, N=128, f16 K/V, 4-wave block, padded S_lds,
    /// cooperative softmax, phase C hoisted) but applies a causal mask
    /// during Phase A: S[q, k] = -inf when k > q. Tiles where all keys
    /// are in the future (kt_start >= q_start + m_tile) are skipped
    /// entirely.
    ///
    /// Intended for text-decoder prefill (causal self-attention with GQA).
    pub fn attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kernel_module, kernel_src) = if self.arch_caps.has_wmma_w32_gfx12() {
            (
                "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32_rdna4",
                kernels::ATTENTION_DFLASH_WMMA_M64_N128_F16KV_V3_CAUSAL_GFX12_SRC,
            )
        } else if self.arch_caps.has_wmma_w32() {
            (
                "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32_rdna3",
                kernels::ATTENTION_DFLASH_WMMA_M64_N128_F16KV_V3_CAUSAL_SRC,
            )
        } else {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32 requires wave32 WMMA; \
                     arch={} has neither gfx11 has_wmma_w32 nor gfx12 has_wmma_w32_gfx12. \
                     Use attention_causal_batched for this arch.",
                    self.arch
                ),
            ));
        };
        assert_eq!(
            q.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32: q must be F32"
        );
        assert_eq!(
            k_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32: k must be F16"
        );
        assert_eq!(
            v_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32: v must be F16"
        );
        assert_eq!(
            out.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32: out must be F32"
        );
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32: head_dim={head_dim} but this kernel is \
             hard-coded to head_dim==128.",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            kernel_module,
            kernel_src,
            "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32",
        )?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let lds_f32 = (128 * head_dim) / 2 + (64 * 130) / 2 + 64 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut causal = 1i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut causal as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 63) / 64;
        let bytes = b * n_heads * head_dim * 8 + l * n_kv_heads * head_dim * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "attention",
            "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32",
            [n_heads as u32, q_tiles as u32, 1],
            [128, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(qp);
                blob.push_ptr(kp);
                blob.push_ptr(vp);
                blob.push_ptr(op);
                blob.push_i32(bi);
                blob.push_i32(li);
                blob.push_i32(nh);
                blob.push_i32(nkv);
                blob.push_i32(hd);
                blob.push_f32(sc);
                blob.push_i32(causal);
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// V2 of `attention_dflash_wmma_m64_n128_f16kv_f32`. Same shape
    /// (M=64, N=128, 4-wave block, f16 K/V, O in registers) but adds
    /// (a) S_lds row stride 130 (was 128) to break a 16-way LDS bank
    /// conflict in phase C's S_lds reads, and (b) cooperative wave-32
    /// softmax via __shfl_xor butterfly.
    pub fn attention_dflash_wmma_m64_n128_f16kv_v2_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            q.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_v2_f32: q must be F32"
        );
        assert_eq!(
            k_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_v2_f32: k must be F16"
        );
        assert_eq!(
            v_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_v2_f32: v must be F16"
        );
        assert_eq!(
            out.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_v2_f32: out must be F32"
        );
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_m64_n128_f16kv_v2_f32: head_dim={head_dim} but this kernel is \
             hard-coded to head_dim==128 (d_chunks=8 unroll, O_frags[8] register array).",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_m64_n128_f16kv_v2_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            "attention_dflash_wmma_m64_n128_f16kv_v2_f32",
            kernels::ATTENTION_DFLASH_WMMA_M64_N128_F16KV_V2_SRC,
            "attention_dflash_wmma_m64_n128_f16kv_v2_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_m64_n128_f16kv_v2_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // LDS layout (f32-equivalent slots):
        //   V_lds[128 * head_dim] f16  = 128 * head_dim / 2 f32 slots
        //   S_lds[64  * 130]      f16  = 64 * 130 / 2       f32 slots (padded stride)
        //   m + l + alpha (64 each f32) = 192               f32 slots
        let lds_f32 = (128 * head_dim) / 2 + (64 * 130) / 2 + 64 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 63) / 64;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [128, 1, 1], // 4 waves
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Per-block scoring on Q8_0 K cache. Reads `k_cache` (one layer's
    /// K-cache backing memory; the buffer must be the Q8_0-formatted slab
    /// produced by `KvCache::new_gpu_q8`) for the first `n_pos` positions,
    /// computes per-block mean K and cosine similarity vs the K at
    /// `last_pos`, and writes `n_blocks` f32 scores into `scores_out`.
    ///
    /// One workgroup per output block, 256 threads per workgroup. Each
    /// thread strides through `kv_dim` doing inline f16-scale + i8-value
    /// dequant; a 256-thread shared-memory reduction folds the partial
    /// (dot, ||block||^2, ||last||^2) fragments into one cosine score.
    ///
    /// Phase 2.1 of #93. Replaces the CPU-side dequant + mean-pool +
    /// cosine in `pflash::compute_scores_batched`.
    #[allow(clippy::too_many_arguments)]
    pub fn pflash_score_q8_kv(
        &mut self,
        k_cache: &GpuTensor,
        scores_out: &GpuTensor,
        n_pos: usize,
        n_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        n_blocks: usize,
        last_pos: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            head_dim % 32 == 0,
            "head_dim must be a multiple of 32 for Q8 KV cache"
        );
        assert!(n_blocks > 0 && block_size > 0 && n_pos > 0);
        assert!(last_pos < n_pos, "last_pos {last_pos} >= n_pos {n_pos}");
        self.ensure_kernel(
            "pflash_score_q8_kv",
            kernels::PFLASH_SCORE_Q8_KV_SRC,
            "pflash_score_q8_kv_blocks",
        )?;
        let func = &self.functions["pflash_score_q8_kv_blocks"];

        let k_ptr = k_cache.buf.as_ptr();
        let s_ptr = scores_out.buf.as_ptr();
        let mut np = n_pos as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = block_size as i32;
        let mut nb = n_blocks as i32;
        let mut lp = last_pos as i32;

        let mut params: Vec<*mut c_void> = vec![
            &k_ptr as *const _ as *mut c_void,
            &s_ptr as *const _ as *mut c_void,
            &mut np as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
            &mut lp as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip.launch_kernel(
                func,
                [n_blocks as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// PFlash per-block scoring — fwht3 K-cache variant.
    ///
    /// Same input/output contract as `pflash_score_q8_kv`: takes a K
    /// cache buffer and emits one f32 cosine score per block. Only the
    /// K dequant path differs (fwht3 vs Q8). Used by
    /// `pflash::compute_scores_batched_gpu` when the drafter runs with
    /// fwht3 KV — that path's no-LDS-cap batched flash unblocks the >15K
    /// ctx regime that Q8 batched flash falls out of.
    ///
    /// Header prepend: the kernel uses `TURBO_C3_256` from
    /// `turbo_common.h`. Reusing `ensure_givens4_kernel` since it already
    /// prepends `turbo_common.h` + `givens_common.h`. The unused
    /// givens_common include is harmless (no symbols are referenced from
    /// it), and avoids adding another `ensure_*` variant.
    #[allow(clippy::too_many_arguments)]
    pub fn pflash_score_fwht3_kv(
        &mut self,
        k_cache: &GpuTensor,
        scores_out: &GpuTensor,
        n_pos: usize,
        n_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        n_blocks: usize,
        last_pos: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to pflash_score_fwht_kv_impl which binds
        self.pflash_score_fwht_kv_impl(
            "pflash_score_fwht3_kv",
            kernels::PFLASH_SCORE_FWHT3_KV_SRC,
            "pflash_score_fwht3_kv_blocks",
            8, // alignment: 8 dims per thread group (3-bit codes × 8 = 24 bits = 3 bytes)
            k_cache,
            scores_out,
            n_pos,
            n_kv_heads,
            head_dim,
            block_size,
            n_blocks,
            last_pos,
        )
    }
    /// PFlash per-block scoring — fwht4 K-cache variant.
    /// 4-bit codes packed into nibbles, two FWHT-128 halves per head at
    /// head_dim=256. Higher precision than fwht3 / larger K storage
    /// (132 B/head vs 100 B). Ablation variant.
    #[allow(clippy::too_many_arguments)]
    pub fn pflash_score_fwht4_kv(
        &mut self,
        k_cache: &GpuTensor,
        scores_out: &GpuTensor,
        n_pos: usize,
        n_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        n_blocks: usize,
        last_pos: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to pflash_score_fwht_kv_impl which binds
        self.pflash_score_fwht_kv_impl(
            "pflash_score_fwht4_kv",
            kernels::PFLASH_SCORE_FWHT4_KV_SRC,
            "pflash_score_fwht4_kv_blocks",
            // fwht4 thread-group = 4 dims (4-bit × 4 = 16 bits = 2 bytes)
            // plus head_dim must accommodate two FWHT-128 halves.
            4,
            k_cache,
            scores_out,
            n_pos,
            n_kv_heads,
            head_dim,
            block_size,
            n_blocks,
            last_pos,
        )
    }
    /// PFlash per-block scoring — fwht2 K-cache variant.
    /// 2-bit codes packed 4 per byte, two FWHT-128 halves per head at
    /// head_dim=256. Smallest K storage in the family (68 B/head).
    /// Ablation / lower-bound variant — likely NIAH-marginal.
    #[allow(clippy::too_many_arguments)]
    pub fn pflash_score_fwht2_kv(
        &mut self,
        k_cache: &GpuTensor,
        scores_out: &GpuTensor,
        n_pos: usize,
        n_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        n_blocks: usize,
        last_pos: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to pflash_score_fwht_kv_impl which binds
        self.pflash_score_fwht_kv_impl(
            "pflash_score_fwht2_kv",
            kernels::PFLASH_SCORE_FWHT2_KV_SRC,
            "pflash_score_fwht2_kv_blocks",
            4, // fwht2 thread-group = 4 dims (2-bit × 4 = 8 bits = 1 byte)
            k_cache,
            scores_out,
            n_pos,
            n_kv_heads,
            head_dim,
            block_size,
            n_blocks,
            last_pos,
        )
    }
    /// Shared launch body for fwht{2,3,4} scoring — same grid +
    /// argument shape, only the kernel binary + per-thread-group
    /// alignment vary.
    #[allow(clippy::too_many_arguments)]
    fn pflash_score_fwht_kv_impl(
        &mut self,
        cache_key: &str,
        src: &str,
        func_name: &str,
        tg_align: i32,
        k_cache: &GpuTensor,
        scores_out: &GpuTensor,
        n_pos: usize,
        n_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        n_blocks: usize,
        last_pos: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            head_dim as i32 % tg_align == 0,
            "head_dim must be a multiple of {tg_align} for this fwht K cache layout",
        );
        assert!(n_blocks > 0 && block_size > 0 && n_pos > 0);
        assert!(last_pos < n_pos, "last_pos {last_pos} >= n_pos {n_pos}");
        self.ensure_givens4_kernel(cache_key, src, func_name)?;
        let func = &self.functions[func_name];

        let k_ptr = k_cache.buf.as_ptr();
        let s_ptr = scores_out.buf.as_ptr();
        let mut np = n_pos as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = block_size as i32;
        let mut nb = n_blocks as i32;
        let mut lp = last_pos as i32;

        let mut params: Vec<*mut c_void> = vec![
            &k_ptr as *const _ as *mut c_void,
            &s_ptr as *const _ as *mut c_void,
            &mut np as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
            &mut lp as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip.launch_kernel(
                func,
                [n_blocks as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn attention_dflash_wmma_m64_n32_f16kv_v5_f32(
        &mut self,
        _q: &GpuTensor,
        _k: &GpuTensor,
        _v: &GpuTensor,
        _out: &GpuTensor,
        _n: usize,
        _seq_len: usize,
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — unimplemented stub (no GPU work; returns Err)
        Err(hip_bridge::HipError::new(801, "not yet implemented"))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym4_wmma_tile_batched(
        &mut self,
        _q: &GpuTensor,
        _k_cache: &GpuTensor,
        _v_cache: &GpuTensor,
        _out: &GpuTensor,
        _positions: &GpuTensor,
        _ct: &GpuTensor,
        _st: &GpuTensor,
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
        _physical_cap: usize,
        _max_ctx_len: usize,
        _batch_size: usize,
        _partials: &GpuTensor,
        _tree_bias: Option<&GpuTensor>,
        _block_start: usize,
        _block_cols: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — unimplemented stub (no GPU work; returns Err)
        Err(hip_bridge::HipError::new(801, "not yet implemented"))
    }
    /// Cold-slot decode attention (deferred-hierarchical KV, Phase 2b): one query
    /// `q` [n_heads × 256] over the compacted cold tier `k`/`v`
    /// [n_kv_heads × n_slots × 256] (dequantized f32, all slots visible, GQA) →
    /// `out` [n_heads × 256]. Zero LDS, one wave per q-head. head_dim must be 256.
    /// Parity oracle: `hipfire_kvquant::ColdTier::two_tier_attend(.., n_hot=0)`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_cold_slots(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        out: &GpuTensor,
        m_out: &GpuTensor, // [n_heads] flash max — for hot/cold tier merge
        l_out: &GpuTensor, // [n_heads] flash denom
        n_heads: usize,
        n_kv_heads: usize,
        n_slots: usize,
        scale: f32,
        // 0 = slot-major [nkv×stride×hd]; 1 = channel-major [nkv×hd×stride]
        // (the kvarn_dequant_tile output layout — lets the cold tier read straight
        // off its dequantized 4-bit records).
        kv_layout: usize,
        // Per-kv-head slot row stride (the padded tile width); attend the first
        // `n_slots` slots. Pass 0 to default to n_slots (dense, no padding).
        slot_stride: usize,
        // Optional per-slot attention-mass accumulator [n_slots] (CASK importance).
        mass_out: Option<&GpuTensor>,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_cold_slots",
            kernels::ATTENTION_COLD_SLOTS_SRC,
            "attention_cold_slots",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let vp = v.buf.as_ptr();
        let op = out.buf.as_ptr();
        let mop = m_out.buf.as_ptr();
        let lop = l_out.buf.as_ptr();
        let massp: *mut std::ffi::c_void = match mass_out {
            Some(t) => t.buf.as_ptr(),
            None => std::ptr::null_mut(),
        };
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut ns = n_slots as i32;
        let mut sc = scale;
        let mut kl = kv_layout as i32;
        let mut sstride = slot_stride as i32;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mop as *const _ as *mut c_void,
            &lop as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut ns as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut kl as *mut _ as *mut c_void,
            &mut sstride as *mut _ as *mut c_void,
            &massp as *const _ as *mut c_void,
        ];
        let func = &self.functions["attention_cold_slots"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Phase 2b hot+cold merge: fold two flash tiers' (out,m,l) partials into one
    /// via online softmax. `hot` = (out_a, m_a, l_a), `cold` = (out_b, m_b, l_b);
    /// writes merged normalized `out` and (chainable) combined `m_out`,`l_out`.
    /// One wave per q-head, zero LDS. See flash_tier_merge.hip.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_tier_merge(
        &mut self,
        out_a: &GpuTensor,
        m_a: &GpuTensor,
        l_a: &GpuTensor,
        out_b: &GpuTensor,
        m_b: &GpuTensor,
        l_b: &GpuTensor,
        out: &GpuTensor,
        m_out: &GpuTensor,
        l_out: &GpuTensor,
        n_heads: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "flash_tier_merge",
            kernels::FLASH_TIER_MERGE_SRC,
            "flash_tier_merge",
        )?;
        let oap = out_a.buf.as_ptr();
        let map = m_a.buf.as_ptr();
        let lap = l_a.buf.as_ptr();
        let obp = out_b.buf.as_ptr();
        let mbp = m_b.buf.as_ptr();
        let lbp = l_b.buf.as_ptr();
        let op = out.buf.as_ptr();
        let mop = m_out.buf.as_ptr();
        let lop = l_out.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut params: Vec<*mut c_void> = vec![
            &oap as *const _ as *mut c_void,
            &map as *const _ as *mut c_void,
            &lap as *const _ as *mut c_void,
            &obp as *const _ as *mut c_void,
            &mbp as *const _ as *mut c_void,
            &lbp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mop as *const _ as *mut c_void,
            &lop as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
        ];
        let func = &self.functions["flash_tier_merge"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Phase 2b: extract the hot KVarN/asym flash's final softmax (m,l) from the
    /// per-tile `partials` buffer it already filled, so the hot tier can feed
    /// `flash_tier_merge`. `max_tiles` and the seq-len convention (positions vs
    /// block_start+block_cols) MUST match the flash that produced `partials`.
    /// Writes `m_out`,`l_out` = [sub_batch × n_heads]. Zero LDS, one thread/(head,pos).
    #[allow(clippy::too_many_arguments)]
    pub fn flash_partials_ml(
        &mut self,
        partials: &GpuTensor,
        positions: &GpuTensor,
        m_out: &GpuTensor,
        l_out: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        tile_size: usize,
        max_tiles: usize,
        sub_batch: usize,
        batch_offset: usize,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "flash_partials_ml",
            kernels::FLASH_PARTIALS_ML_SRC,
            "flash_partials_ml",
        )?;
        let pp = partials.buf.as_ptr();
        let posp = positions.buf.as_ptr();
        let mop = m_out.buf.as_ptr();
        let lop = l_out.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut ts = tile_size as i32;
        let mut mt = max_tiles as i32;
        let mut bo = batch_offset as i32;
        let mut bs = block_start as i32;
        let mut bc = block_cols as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &posp as *const _ as *mut c_void,
            &mop as *const _ as *mut c_void,
            &lop as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ts as *mut _ as *mut c_void,
            &mut mt as *mut _ as *mut c_void,
            &mut bo as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut bc as *mut _ as *mut c_void,
        ];
        let func = &self.functions["flash_partials_ml"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, sub_batch as u32, 1],
                [1, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// SWA ring write — BATCHED. For each batch position b at
    /// `start_pos + b`, writes `kv_batch[b, :]` into the ring at slot
    /// `(start_pos + b) % window`. Called at chunk-end to advance the
    /// ring so future decode/chunk calls see the latest history.
    #[allow(clippy::too_many_arguments)]
    pub fn swa_ring_write_batched_f32(
        &mut self,
        kv_batch: &GpuTensor, // [B, head_dim]
        cache: &GpuTensor,    // [n_kv_heads, head_dim, window]
        n_kv_heads: i32,
        head_dim: i32,
        window: i32,
        start_pos: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "swa_ring_write_batched",
            kernels::SWA_RING_WRITE_BATCHED_SRC,
            "swa_ring_write_batched_f32",
        )?;
        let func = &self.functions["swa_ring_write_batched_f32"];
        let kp = kv_batch.buf.as_ptr();
        let cp = cache.buf.as_ptr();
        let mut nh = n_kv_heads;
        let mut hd = head_dim;
        let mut w = window;
        let mut sp = start_pos;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &kp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut w as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [((head_dim + 255) / 256) as u32, batch_size as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HIP-graphs-safe variant of `swa_ring_write_f32`: reads `slot`
    /// from a device buffer instead of an i32 kernarg. Use this in
    /// captured-region code paths where the position changes between
    /// graph replays — the host updates `slot_buf` (stable Box-backed
    /// memory) before each replay and the captured launch re-reads it.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn swa_ring_write_f32_buf(
        &mut self,
        kv: &GpuTensor,
        cache: &GpuTensor,
        slot_buf: &GpuTensor,
        n_kv_heads: i32,
        head_dim: i32,
        window: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "swa_ring_write_f32_buf",
            kernels::SWA_RING_WRITE_BUF_SRC,
            "swa_ring_write_f32_buf",
        )?;
        let kp = kv.buf.as_ptr();
        let cp = cache.buf.as_ptr();
        let sb = slot_buf.buf.as_ptr();
        let mut nh = n_kv_heads;
        let mut hd = head_dim;
        let mut wn = window;
        let mut params: Vec<*mut c_void> = vec![
            &kp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &sb as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut wn as *mut _ as *mut c_void,
        ];
        let grid = ((head_dim + 255) / 256) as u32;
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(kp);
            b.push_ptr(cp);
            b.push_ptr(sb);
            b.push_i32(nh);
            b.push_i32(hd);
            b.push_i32(wn);
            b
        };
        self.launch_maybe_blob(
            "swa_ring_write_f32_buf",
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    /// SWA visibility staging — BATCHED. For each batch position b at
    /// absolute position `start_pos + b`, build a contiguous visibility
    /// window from the pre-chunk SWA ring + within-chunk `kv_batch`.
    /// Output `[B, head_dim, swa_window]` feeds the batched attention
    /// kernels (deepseek4_attn_swa_topk_batched / deepseek4_attn_swa_batched).
    ///
    /// Each batch row's effective length is `min(start_pos + b + 1,
    /// swa_window)`; trailing slots beyond that are left uninitialised
    /// since the attention kernel masks them via `n_valid_swa_arr[b]`.
    #[allow(clippy::too_many_arguments)]
    pub fn swa_visibility_stage_batched(
        &mut self,
        ring: &GpuTensor,     // [head_dim, swa_window] pre-chunk
        kv_batch: &GpuTensor, // [B, head_dim] within-chunk
        staged: &GpuTensor,   // [B, head_dim, swa_window] output
        start_pos: i32,
        swa_window: i32,
        head_dim: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "swa_visibility_stage_batched",
            kernels::SWA_VISIBILITY_STAGE_BATCHED_SRC,
            "swa_visibility_stage_batched",
        )?;
        let func = &self.functions["swa_visibility_stage_batched"];
        let rp = ring.buf.as_ptr();
        let kp = kv_batch.buf.as_ptr();
        let sp = staged.buf.as_ptr();
        let mut sp_i = start_pos;
        let mut sw = swa_window;
        let mut hd = head_dim;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &rp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut sp_i as *mut _ as *mut c_void,
            &mut sw as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [head_dim as u32, batch_size as u32, 1],
                [swa_window as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
}
