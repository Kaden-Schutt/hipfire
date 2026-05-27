// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Attention, KV cache, DFlash, pflash, triattn, kv_compact, and vision attention dispatch.

use crate::kernels;
use crate::GpuTensor;
use crate::DType;
use crate::Gpu;
use hip_bridge::{DeviceBuffer, HipResult};
use std::ffi::c_void;

impl Gpu {
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
        n_tokens: usize, n_heads: usize, head_dim: usize,
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
                b.push_ptr(q_ptr); b.push_ptr(sre_ptr); b.push_ptr(sim_ptr);
                b.push_ptr(sab_ptr); b.push_ptr(cnt_ptr);
                b.push_i32(nt); b.push_i32(nh); b.push_i32(hd); b.push_i32(li);
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
        self.ensure_kernel("attention", kernels::ATTENTION_SRC, "attention_f32")?;
        let func = &self.functions["attention_f32"];

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
        let effective_seq = if self.active_stream.is_some() { max_seq } else { seq_len_hint };
        let block_size = (effective_seq.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((effective_seq + block_size as usize) * 4) as u32;

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
        self.ensure_kernel("attention_flash_partial", kernels::ATTENTION_FLASH_SRC, "attention_flash_partial")?;

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
                b.push_ptr(q_ptr); b.push_ptr(k_ptr); b.push_ptr(v_ptr); b.push_ptr(p_ptr);
                b.push_i32(sl); b.push_i32(nh); b.push_i32(nkv);
                b.push_i32(hd); b.push_i32(ms); b.push_f32(sc); b.push_i32(cs);
                b
            },
        )?;

        // Phase 2: reduce partials
        self.ensure_kernel("attention_flash_reduce", kernels::ATTENTION_FLASH_SRC, "attention_flash_reduce")?;

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
                b.push_ptr(p_ptr2); b.push_ptr(out_ptr);
                b.push_i32(nh2); b.push_i32(nc); b.push_i32(hd2);
                b
            },
        )
    }

    /// GQA-aware split-K flash decode: one phase-1 block per (kv_head, chunk)
    /// reuses a single K/V load across its query-head group (n_heads/n_kv_heads),
    /// so the KV cache is traversed n_kv_heads× not n_heads×. Phase-2 reuses
    /// `attention_flash_reduce`. Same partials buffer as `attention_flash`.
    pub fn attention_flash_gqa(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, partials: &GpuTensor, seq_len: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let cs_cap = std::env::var("HIPFIRE_GQA_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
        let chunk_size = if seq_len <= cs_cap { seq_len } else { cs_cap };
        let n_chunks = (seq_len + chunk_size - 1) / chunk_size;

        self.ensure_kernel("attention_flash_gqa_partial", kernels::ATTENTION_FLASH_GQA_SRC, "attention_flash_gqa_partial")?;
        let q_ptr = q.buf.as_ptr(); let k_ptr = k_cache.buf.as_ptr();
        let v_ptr = v_cache.buf.as_ptr(); let p_ptr = partials.buf.as_ptr();
        let sl = seq_len as i32; let nh = n_heads as i32; let nkv = n_kv_heads as i32;
        let hd = head_dim as i32; let ms = max_seq as i32; let sc = scale; let cs = chunk_size as i32;
        let mut p1: Vec<*mut c_void> = vec![
            &q_ptr as *const _ as *mut c_void, &k_ptr as *const _ as *mut c_void,
            &v_ptr as *const _ as *mut c_void, &p_ptr as *const _ as *mut c_void,
            &sl as *const _ as *mut c_void, &nh as *const _ as *mut c_void, &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void, &ms as *const _ as *mut c_void, &sc as *const _ as *mut c_void, &cs as *const _ as *mut c_void,
        ];
        let block = 128u32.min(chunk_size as u32).next_power_of_two();
        let shmem = ((chunk_size + block as usize) * 4) as u32;
        self.launch_maybe_blob(
            "attention_flash_gqa_partial",
            [n_kv_heads as u32, n_chunks as u32, 1], [block, 1, 1], shmem, &mut p1,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr); b.push_ptr(k_ptr); b.push_ptr(v_ptr); b.push_ptr(p_ptr);
                b.push_i32(sl); b.push_i32(nh); b.push_i32(nkv);
                b.push_i32(hd); b.push_i32(ms); b.push_f32(sc); b.push_i32(cs);
                b
            },
        )?;

        self.ensure_kernel("attention_flash_reduce", kernels::ATTENTION_FLASH_SRC, "attention_flash_reduce")?;
        let p2_ptr = partials.buf.as_ptr(); let o_ptr = out.buf.as_ptr();
        let nh2 = n_heads as i32; let nc = n_chunks as i32; let hd2 = head_dim as i32;
        let mut p2: Vec<*mut c_void> = vec![
            &p2_ptr as *const _ as *mut c_void, &o_ptr as *const _ as *mut c_void,
            &nh2 as *const _ as *mut c_void, &nc as *const _ as *mut c_void, &hd2 as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "attention_flash_reduce",
            [n_heads as u32, 1, 1], [head_dim.min(256) as u32, 1, 1], 0, &mut p2,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(p2_ptr); b.push_ptr(o_ptr);
                b.push_i32(nh2); b.push_i32(nc); b.push_i32(hd2);
                b
            },
        )
    }

    /// Single-launch GQA decode: one block per kv_head streams all KV once,
    /// accumulates online-softmax for the group in LDS, writes O. No partials,
    /// no reduce. Grid = n_kv_heads. Probe of launch-vs-occupancy floor.
    pub fn attention_flash_gqa_fused(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, seq_len: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        self.ensure_kernel("attention_flash_gqa_fused", kernels::ATTENTION_FLASH_GQA_FUSED_SRC, "attention_flash_gqa_fused")?;
        let f = &self.functions["attention_flash_gqa_fused"];
        let mut q_ptr = q.buf.as_ptr(); let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr(); let mut o_ptr = out.buf.as_ptr();
        let mut sl = seq_len as i32; let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let kv_group = n_heads / n_kv_heads;
        let block = 128u32;
        let shmem = ((kv_group * head_dim + block as usize) * 4) as u32;
        let mut p: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void, &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void, &mut o_ptr as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(f, [n_kv_heads as u32, 1, 1], [block, 1, 1], shmem, self.stream_ref(), &mut p) }
    }

    /// Write KV to HFQ4 co-located block (72 bytes per head: scale+zero+nibbles).
    pub fn kv_cache_write_hfq4(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_hfq4", kernels::KV_CACHE_WRITE_HFQ4_SRC, "kv_cache_write_hfq4")?;
        let func = &self.functions["kv_cache_write_hfq4"];
        let mut d = dst.buf.as_ptr(); let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Attention with HFQ4 KV blocks (72 bytes per head, co-located).
    pub fn attention_hfq4_kv(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_hfq4_kv", kernels::ATTENTION_HFQ4_KV_SRC, "attention_hfq4_kv")?;
        let func = &self.functions["attention_hfq4_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr(); let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        // scores[seq_len] + ws[block_size] + q_shared[head_dim]
        let shared_mem = ((seq_len_hint + block_size as usize + head_dim) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// INT8 co-located with f16 scale (matches Q8_0 precision, one block per head).
    pub fn kv_cache_write_int8c_f16(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_int8c_f16", kernels::KV_CACHE_WRITE_INT8C_F16_SRC, "kv_cache_write_int8c_f16")?;
        let func = &self.functions["kv_cache_write_int8c_f16"];
        let mut d = dst.buf.as_ptr(); let mut s = src.buf.as_ptr(); let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    pub fn attention_int8c_f16_kv(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_int8c_f16_kv", kernels::ATTENTION_INT8C_F16_KV_SRC, "attention_int8c_f16_kv")?;
        let func = &self.functions["attention_int8c_f16_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr(); let mut op = out.buf.as_ptr(); let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Write KV to INT8 co-located block (f32 scale + int8 data, symmetric).
    pub fn kv_cache_write_int8c(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_int8c", kernels::KV_CACHE_WRITE_INT8C_SRC, "kv_cache_write_int8c")?;
        let func = &self.functions["kv_cache_write_int8c"];
        let mut d = dst.buf.as_ptr(); let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Attention with INT8 co-located KV blocks.
    pub fn attention_int8c_kv(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_int8c_kv", kernels::ATTENTION_INT8C_KV_SRC, "attention_int8c_kv")?;
        let func = &self.functions["attention_int8c_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr(); let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize + head_dim) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Write KV to HFQ8 cache (FP32 scale+zero, contiguous uint8).
    pub fn kv_cache_write_hfq8(
        &mut self, dst_data: &GpuTensor, dst_scales: &GpuTensor, src: &GpuTensor,
        pos_buf: &DeviceBuffer, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_hfq8", kernels::KV_CACHE_WRITE_HFQ8_SRC, "kv_cache_write_hfq8")?;
        let func = &self.functions["kv_cache_write_hfq8"];
        let mut dd = dst_data.buf.as_ptr(); let mut ds = dst_scales.buf.as_ptr();
        let mut s = src.buf.as_ptr(); let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dd as *mut _ as *mut c_void, &mut ds as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void, &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Attention with HFQ8 KV cache.
    pub fn attention_hfq8_kv(
        &mut self, q: &GpuTensor,
        k_data: &GpuTensor, k_scales: &GpuTensor,
        v_data: &GpuTensor, v_scales: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_hfq8_kv", kernels::ATTENTION_HFQ8_KV_SRC, "attention_hfq8_kv")?;
        let func = &self.functions["attention_hfq8_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr();
        let mut kd = k_data.buf.as_ptr(); let mut ks = k_scales.buf.as_ptr();
        let mut vd = v_data.buf.as_ptr(); let mut vs = v_scales.buf.as_ptr();
        let mut op = out.buf.as_ptr(); let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kd as *mut _ as *mut c_void, &mut ks as *mut _ as *mut c_void,
            &mut vd as *mut _ as *mut c_void, &mut vs as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void, &mut pp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Write KV to INT8 cache (separate scale array).
    pub fn kv_cache_write_int8(
        &mut self, dst_vals: &GpuTensor, dst_scales: &GpuTensor, src: &GpuTensor,
        pos_buf: &DeviceBuffer, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_int8", kernels::KV_CACHE_WRITE_INT8_SRC, "kv_cache_write_int8")?;
        let func = &self.functions["kv_cache_write_int8"];
        let mut dv = dst_vals.buf.as_ptr(); let mut ds = dst_scales.buf.as_ptr();
        let mut s = src.buf.as_ptr(); let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dv as *mut _ as *mut c_void, &mut ds as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void, &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Attention with INT8 KV (separate scale array).
    pub fn attention_int8_kv(
        &mut self, q: &GpuTensor,
        k_vals: &GpuTensor, k_scales: &GpuTensor,
        v_vals: &GpuTensor, v_scales: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_int8_kv", kernels::ATTENTION_INT8_KV_SRC, "attention_int8_kv")?;
        let func = &self.functions["attention_int8_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut kv_ptr = k_vals.buf.as_ptr(); let mut ks_ptr = k_scales.buf.as_ptr();
        let mut vv_ptr = v_vals.buf.as_ptr(); let mut vs_ptr = v_scales.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr(); let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut kv_ptr as *mut _ as *mut c_void, &mut ks_ptr as *mut _ as *mut c_void,
            &mut vv_ptr as *mut _ as *mut c_void, &mut vs_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void, &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Batched causal attention: all query positions in one launch.
    /// Q: [seq_len × n_heads × head_dim], K/V: [seq_len × n_kv_heads × head_dim].
    pub fn attention_causal_batched(
        &mut self, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, out: &GpuTensor,
        seq_len: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_causal_batched", kernels::ATTENTION_CAUSAL_BATCHED_SRC, "attention_causal_batched")?;
        let func = &self.functions["attention_causal_batched"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr(); let mut op = out.buf.as_ptr();
        let mut sl = seq_len as i32; let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        // Block size: enough threads to cover head_dim and seq_len
        let block_size = 128u32.min((seq_len.max(head_dim) as u32).next_power_of_two());
        // Shared: scores[seq_len] + workspace[block_size]
        let shared_mem = ((seq_len + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, seq_len as u32, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Batched Q8_0 KV cache write: quantize multiple positions in one launch.
    pub fn kv_cache_write_q8_0_batched(
        &mut self, dst: &GpuTensor, src: &GpuTensor, positions: &GpuTensor,
        n_kv_heads: usize, head_dim: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_q8_0_batched", kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC, "kv_cache_write_q8_0_batched")?;
        let mut d = dst.buf.as_ptr(); let mut s = src.buf.as_ptr();
        let mut p = positions.buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32; let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut bs as *mut _ as *mut c_void,
        ];
        let total_blocks = (n_kv_heads * head_dim / 32) as u32;
        self.launch_maybe_blob(
            "kv_cache_write_q8_0_batched",
            [total_blocks, batch_size as u32, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(d); b.push_ptr(s); b.push_ptr(p);
                b.push_i32(nkv); b.push_i32(hd); b.push_i32(bs);
                b
            },
        )
    }

    /// Write KV vector to Q8_0 quantized cache (same format as GGML Q8_0).
    pub fn kv_cache_write_q8_0(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_q8_0", kernels::KV_CACHE_WRITE_Q8_0_SRC, "kv_cache_write_q8_0")?;
        let d = dst.buf.as_ptr();
        let s = src.buf.as_ptr();
        let p = pos_buf.as_ptr();
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &d as *const _ as *mut c_void, &s as *const _ as *mut c_void,
            &p as *const _ as *mut c_void, &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
        ];
        let total_blocks = (n_kv_heads * head_dim / 32) as u32;
        let bytes = crate::profile::kv_cache_write_q8_0_bytes(n_kv_heads, head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "kv_write", "kv_cache_write_q8_0", bytes);
        let result = self.launch_maybe_blob(
            "kv_cache_write_q8_0",
            [total_blocks, 1, 1], [32, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(d); b.push_ptr(s); b.push_ptr(p);
                b.push_i32(nkv); b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
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
            q, k_cache, v_cache, out, positions,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size,
            None, 0, 0,
        )
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
        let block_size = (max_ctx_len.max(head_dim) as u32).next_power_of_two().min(256);
        // Shared memory must accommodate the LARGEST batch row's seq_len for
        // scores[], plus nthreads workspace and head_dim q_shared.
        let shared_mem = ((max_ctx_len + block_size as usize + head_dim) * 4) as u32;
        let bytes = crate::profile::attention_q8_0_kv_bytes(n_heads, n_kv_heads, head_dim, max_ctx_len) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "attention", "attention_q8_0_kv_batched", bytes);
        let bias_raw = bias_ptr; // alias for move into closure
        let result = self.launch_maybe_blob(
            "attention_q8_0_kv_batched",
            [n_heads as u32, batch_size as u32, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr); b.push_ptr(k_ptr); b.push_ptr(v_ptr);
                b.push_ptr(out_ptr); b.push_ptr(pos_ptr); b.push_ptr(bias_raw);
                b.push_i32(nh); b.push_i32(nkv); b.push_i32(hd); b.push_i32(ms);
                b.push_f32(sc); b.push_i32(bs); b.push_i32(bc);
                b
            },
        );
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Flash attention with Q8_0 KV cache — tile + reduce two-kernel path.
    /// Tiles seq_len into chunks of `tile_size`, launches [n_heads, n_tiles]
    /// blocks for the tile kernel, then [n_heads] blocks for the reduce.
    /// Requires a pre-allocated `partials` buffer of size
    /// n_heads * max_tiles * (2 + head_dim) floats.
    pub fn attention_flash_q8_0(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        // Graph-safe: use max_tiles so the grid is position-independent.
        // The tile kernel exits early for tiles beyond actual seq_len.
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        // For profiling / non-graph code paths, the actual tile count:
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.graphs.capture_mode { max_tiles } else { actual_tiles };

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
            let nh = n_heads as i32; let nkv = n_kv_heads as i32;
            let hd = head_dim as i32; let ms = max_seq as i32;
            let sc = scale; let ts = TILE_SIZE as i32;
            let grid = [n_heads as u32, launch_tiles as u32, 1];
            let shared = ((TILE_SIZE + head_dim) * 4) as u32;
            let mut params: Vec<*mut c_void> = vec![
                &q_ptr as *const _ as *mut c_void, &k_ptr as *const _ as *mut c_void,
                &v_ptr as *const _ as *mut c_void, &p_ptr as *const _ as *mut c_void,
                &pos_ptr as *const _ as *mut c_void, &nh as *const _ as *mut c_void,
                &nkv as *const _ as *mut c_void, &hd as *const _ as *mut c_void,
                &ms as *const _ as *mut c_void, &sc as *const _ as *mut c_void,
                &ts as *const _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                "attention_flash_q8_0_tile", grid, [32, 1, 1], shared, &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(q_ptr); b.push_ptr(k_ptr); b.push_ptr(v_ptr);
                    b.push_ptr(p_ptr); b.push_ptr(pos_ptr);
                    b.push_i32(nh); b.push_i32(nkv); b.push_i32(hd); b.push_i32(ms);
                    b.push_f32(sc); b.push_i32(ts);
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
                &p_ptr as *const _ as *mut c_void, &o_ptr as *const _ as *mut c_void,
                &nh as *const _ as *mut c_void, &hd as *const _ as *mut c_void,
                &pos_ptr as *const _ as *mut c_void, &ts as *const _ as *mut c_void,
                &mt as *const _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                "attention_flash_q8_0_reduce", [n_heads as u32, 1, 1], [32, 1, 1], 0, &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(p_ptr); b.push_ptr(o_ptr);
                    b.push_i32(nh); b.push_i32(hd);
                    b.push_ptr(pos_ptr); b.push_i32(ts); b.push_i32(mt);
                    b
                },
            )?;
        }
        Ok(())
    }

    /// Compile a givens4 kernel — prepends turbo_common + givens_common headers.
    fn ensure_givens4_kernel(&mut self, name: &str, body_src: &str, func_name: &str) -> HipResult<()> {
        if self.functions.contains_key(func_name) {
            return Ok(());
        }
        let stripped = body_src
            .replace("#include \"turbo_common.h\"", "")
            .replace("#include \"givens_common.h\"", "");
        let full_src = format!("{}\n{}\n{}", kernels::TURBO_COMMON_H, kernels::GIVENS_COMMON_SRC, stripped);
        let obj_path = self.compiler.compile(name, &full_src)?;
        let obj_path_str = obj_path.to_str().unwrap().to_string();
        if !self.modules.contains_key(name) {
            let module = self.hip.module_load(&obj_path_str)?;
            self.modules.insert(name.to_string(), module);
        }
        let module = &self.modules[name];
        let func = self.hip.module_get_function(module, func_name)?;
        self.functions.insert(func_name.to_string(), func);
        Ok(())
    }


    /// Fused K+V write for asym4: K at givens4 (rotated 4-bit), V at Q8_0 (normal space).
    /// Launches two kernels — K-only givens4 writer + standard Q8_0 writer.
    pub fn kv_cache_write_asym4_fused(
        &mut self, k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // K: rotated 4-bit
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_givens4",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS4_SRC,
            "kv_cache_write_asym_k_givens4",
        )?;
        {
            let func = &self.functions["kv_cache_write_asym_k_givens4"];
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = pos_buf.as_ptr();
            let mut ctp = cos_theta.buf.as_ptr();
            let mut stp = sin_theta.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut ctp as *mut _ as *mut c_void,
                &mut stp as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            unsafe {
                self.hip.launch_kernel(
                    func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared_mem,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        // V: standard Q8_0
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }

    /// Fused K+V write for fwht4: K at signed-FWHT-rotated 4-bit, V at Q8_0.
    /// Byte-identical storage to asym4_fused — only the K-write kernel differs.
    /// `signs1` and `signs2` are 128-element FP32 ±1 vectors (occupy the same
    /// `givens_cos`/`givens_sin` slots on KvCache when `quant_fwht == true`).
    pub fn kv_cache_write_fwht4_fused(
        &mut self, k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_fwht4",
            kernels::KV_CACHE_WRITE_ASYM_K_FWHT4_SRC,
            "kv_cache_write_asym_k_fwht4",
        )?;
        {
            let func = &self.functions["kv_cache_write_asym_k_fwht4"];
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = pos_buf.as_ptr();
            let mut s1p = signs1.buf.as_ptr();
            let mut s2p = signs2.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut s1p as *mut _ as *mut c_void,
                &mut s2p as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            unsafe {
                self.hip.launch_kernel(
                    func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared_mem,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        // V: standard Q8_0 (same as asym4)
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }

    /// Fused K+V write for asym3: K at 3-bit rotated (RotorQuant "planar3"), V at Q8_0.
    /// Best-quality rotated K per RotorQuant paper. Head geometry: 32 threads × 8
    /// values = 256 dims single-pass. 100 bytes/head for hd=256.
    pub fn kv_cache_write_asym3_fused(
        &mut self, k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_givens3",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_SRC,
            "kv_cache_write_asym_k_givens3",
        )?;
        {
            let func = &self.functions["kv_cache_write_asym_k_givens3"];
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = pos_buf.as_ptr();
            let mut ctp = cos_theta.buf.as_ptr();
            let mut stp = sin_theta.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut ctp as *mut _ as *mut c_void,
                &mut stp as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            unsafe {
                self.hip.launch_kernel(
                    func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared_mem,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }

    /// Fused K+V write for fwht3: K at signed-FWHT-256 rotated 3-bit, V at Q8_0.
    /// Byte-identical storage to asym3 — only the K-write kernel differs.
    pub fn kv_cache_write_fwht3_fused(
        &mut self, k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_fwht3",
            kernels::KV_CACHE_WRITE_ASYM_K_FWHT3_SRC,
            "kv_cache_write_asym_k_fwht3",
        )?;
        {
            let func = &self.functions["kv_cache_write_asym_k_fwht3"];
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = pos_buf.as_ptr();
            let mut s1p = signs1.buf.as_ptr();
            let mut s2p = signs2.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut s1p as *mut _ as *mut c_void,
                &mut s2p as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            unsafe {
                self.hip.launch_kernel(
                    func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared_mem,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }

    /// Shared helper: launch a batched K-only rotated write kernel.
    fn launch_asym_k_batched(
        &mut self, kernel_key: &str, src_const: &'static str, func_name: &'static str,
        k_dst: &GpuTensor, k_src: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_kv_heads: usize, head_dim: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_givens4_kernel(kernel_key, src_const, func_name)?;
        let mut kdp = k_dst.buf.as_ptr();
        let mut ksp = k_src.buf.as_ptr();
        let mut pp = positions.buf.as_ptr();
        let mut ctp = cos_theta.buf.as_ptr();
        let mut stp = sin_theta.buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kdp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut ctp as *mut _ as *mut c_void,
            &mut stp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        self.launch_maybe_blob(
            func_name,
            [n_kv_heads as u32, batch_size as u32, 1],
            [32, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(kdp); b.push_ptr(ksp); b.push_ptr(pp);
                b.push_ptr(ctp); b.push_ptr(stp);
                b.push_i32(nkv); b.push_i32(hd); b.push_i32(bs);
                b
            },
        )
    }

    /// Shared helper: launch a batched asym flash tile + the shared asym reduce.
    ///
    /// `tree_bias` / `block_start` / `block_cols` activate DDTree tree-attention
    /// mode (bias added to in-block qk scores; seq_len extends to full cache
    /// including the tree block). When `tree_bias` is None and `block_cols` is
    /// 0, behavior is byte-identical to the legacy causal path.
    #[allow(clippy::too_many_arguments)]
    fn launch_asym_flash_batched(
        &mut self,
        tile_key: &'static str, tile_src: &'static str, tile_func_name: &'static str,
        q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_ctx_len + TILE_SIZE - 1) / TILE_SIZE;
        let stride = 2 + head_dim;
        let per_pos_bytes = n_heads * max_tiles * stride * 4;
        let partials_capacity = partials.numel() * 4;
        let sub_batch = if per_pos_bytes > 0 {
            (partials_capacity / per_pos_bytes).max(1).min(batch_size)
        } else {
            batch_size
        };

        self.ensure_givens4_kernel(tile_key, tile_src, tile_func_name)?;
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
                let q_ptr = unsafe {
                    (q.buf.as_ptr() as *mut u8).add(offset * q_dim * 4) as *mut c_void
                };
                let k_ptr = k_cache.buf.as_ptr();
                let v_ptr = v_cache.buf.as_ptr();
                let p_ptr = partials.buf.as_ptr();
                let pos_ptr = positions.buf.as_ptr();
                let ct_ptr = cos_theta.buf.as_ptr();
                let st_ptr = sin_theta.buf.as_ptr();
                let bias_ptr: *mut std::ffi::c_void = match tree_bias {
                    Some(t) => t.buf.as_ptr(),
                    None => std::ptr::null_mut(),
                };
                let nh = n_heads as i32; let nkv = n_kv_heads as i32;
                let hd = head_dim as i32; let ms = max_seq as i32;
                let sc = scale; let ts = TILE_SIZE as i32;
                let mt = max_tiles as i32; let bo = offset as i32;
                let bs = block_start as i32; let bc = block_cols as i32;
                let mut params: Vec<*mut c_void> = vec![
                    &q_ptr as *const _ as *mut c_void,
                    &k_ptr as *const _ as *mut c_void,
                    &v_ptr as *const _ as *mut c_void,
                    &p_ptr as *const _ as *mut c_void,
                    &pos_ptr as *const _ as *mut c_void,
                    &ct_ptr as *const _ as *mut c_void,
                    &st_ptr as *const _ as *mut c_void,
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
                ];
                self.launch_maybe_blob(
                    tile_func_name,
                    [n_heads as u32, max_tiles as u32, chunk as u32],
                    [32, 1, 1],
                    (TILE_SIZE * 4) as u32,
                    &mut params,
                    || {
                        let mut b = hip_bridge::KernargBlob::new();
                        b.push_ptr(q_ptr); b.push_ptr(k_ptr); b.push_ptr(v_ptr);
                        b.push_ptr(p_ptr); b.push_ptr(pos_ptr);
                        b.push_ptr(ct_ptr); b.push_ptr(st_ptr); b.push_ptr(bias_ptr);
                        b.push_i32(nh); b.push_i32(nkv); b.push_i32(hd); b.push_i32(ms);
                        b.push_f32(sc); b.push_i32(ts); b.push_i32(mt); b.push_i32(bo);
                        b.push_i32(bs); b.push_i32(bc);
                        b
                    },
                )?;
            }
            {
                let p_ptr = partials.buf.as_ptr();
                let o_ptr = unsafe {
                    (out.buf.as_ptr() as *mut u8).add(offset * q_dim * 4) as *mut c_void
                };
                let pos_ptr = positions.buf.as_ptr();
                let nh = n_heads as i32; let hd = head_dim as i32;
                let ts = TILE_SIZE as i32; let mt = max_tiles as i32;
                let bo = offset as i32;
                let bs = block_start as i32; let bc = block_cols as i32;
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
                        b.push_ptr(p_ptr); b.push_ptr(o_ptr); b.push_ptr(pos_ptr);
                        b.push_i32(nh); b.push_i32(hd); b.push_i32(ts); b.push_i32(mt);
                        b.push_i32(bo); b.push_i32(bs); b.push_i32(bc);
                        b
                    },
                )?;
            }
            offset += chunk;
        }
        Ok(())
    }

    /// Batched K+V write for asym4 (K 4-bit rotated + V Q8_0).
    pub fn kv_cache_write_asym4_batched(
        &mut self,
        k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_kv_heads: usize, head_dim: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_k_batched(
            "kv_cache_write_asym_k_givens4_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS4_BATCHED_SRC,
            "kv_cache_write_asym_k_givens4_batched",
            k_dst, k_src, positions, cos_theta, sin_theta,
            n_kv_heads, head_dim, batch_size,
        )?;
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }

    /// Batched K+V write for fwht4 (K FWHT-rotated 4-bit + V Q8_0).
    /// Same launch geometry as asym4_batched; only the kernel name + sign-vector
    /// param semantics differ.
    pub fn kv_cache_write_fwht4_batched(
        &mut self,
        k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, positions: &GpuTensor,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_kv_heads: usize, head_dim: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_k_batched(
            "kv_cache_write_asym_k_fwht4_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_FWHT4_BATCHED_SRC,
            "kv_cache_write_asym_k_fwht4_batched",
            k_dst, k_src, positions, signs1, signs2,
            n_kv_heads, head_dim, batch_size,
        )?;
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }

    /// Batched K+V write for asym2 (K 2-bit rotated + V Q8_0).
    pub fn kv_cache_write_asym2_batched(
        &mut self,
        k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_kv_heads: usize, head_dim: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_k_batched(
            "kv_cache_write_asym_k_givens2_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS2_BATCHED_SRC,
            "kv_cache_write_asym_k_givens2_batched",
            k_dst, k_src, positions, cos_theta, sin_theta,
            n_kv_heads, head_dim, batch_size,
        )?;
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }

    /// Batched K+V write for fwht2 (K FWHT-rotated 2-bit + V Q8_0).
    pub fn kv_cache_write_fwht2_batched(
        &mut self,
        k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, positions: &GpuTensor,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_kv_heads: usize, head_dim: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_k_batched(
            "kv_cache_write_asym_k_fwht2_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_FWHT2_BATCHED_SRC,
            "kv_cache_write_asym_k_fwht2_batched",
            k_dst, k_src, positions, signs1, signs2,
            n_kv_heads, head_dim, batch_size,
        )?;
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }

    /// Batched flash attention for asym4 (K 4-bit rotated + V Q8_0).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym4_batched(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_flash_asym4_batched_masked(
            q, k_cache, v_cache, out, positions, cos_theta, sin_theta,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size, partials,
            None, 0, 0,
        )
    }

    /// Tree-mask variant of `attention_flash_asym4_batched`. See
    /// `attention_q8_0_kv_batched_masked` and `ddtree::linearize_tree` for the
    /// bias layout. Passes `tree_bias` / `block_start` / `block_cols` into the
    /// tile + reduce kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym4_batched_masked(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
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
            q, k_cache, v_cache, out, positions, cos_theta, sin_theta,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size, partials,
            tree_bias, block_start, block_cols,
        )
    }

    /// Batched flash attention for fwht4 (K FWHT-rotated 4-bit + V Q8_0).
    /// `signs1` and `signs2` occupy the same slots as cos_theta/sin_theta on
    /// the asym4 path — the helper passes them opaquely to the tile kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_fwht4_batched(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_flash_fwht4_batched_masked(
            q, k_cache, v_cache, out, positions, signs1, signs2,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size, partials,
            None, 0, 0,
        )
    }

    /// Tree-mask variant of `attention_flash_fwht4_batched`. Mirrors the asym4
    /// path one-for-one; the FA tile kernel is the only difference.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_fwht4_batched_masked(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_fwht4_tile_batched",
            kernels::ATTENTION_FLASH_FWHT4_TILE_BATCHED_SRC,
            "attention_flash_fwht4_tile_batched",
            q, k_cache, v_cache, out, positions, signs1, signs2,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size, partials,
            tree_bias, block_start, block_cols,
        )
    }

    /// Batched flash attention for asym2 (K 2-bit rotated + V Q8_0).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym2_batched(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_asym2_tile_batched",
            kernels::ATTENTION_FLASH_ASYM2_TILE_BATCHED_SRC,
            "attention_flash_asym2_tile_batched",
            q, k_cache, v_cache, out, positions, cos_theta, sin_theta,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size, partials,
            None, 0, 0,
        )
    }

    /// Batched flash attention for fwht2 (K FWHT-rotated 2-bit + V Q8_0).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_fwht2_batched(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_fwht2_tile_batched",
            kernels::ATTENTION_FLASH_FWHT2_TILE_BATCHED_SRC,
            "attention_flash_fwht2_tile_batched",
            q, k_cache, v_cache, out, positions, signs1, signs2,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size, partials,
            None, 0, 0,
        )
    }

    /// Batched K+V write for asym3 — processes N positions in one launch.
    /// K-only givens3 write (batched) + Q8_0 V write (batched).
    pub fn kv_cache_write_asym3_batched(
        &mut self,
        k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_kv_heads: usize, head_dim: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // K: batched 3-bit rotated write.
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_givens3_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_BATCHED_SRC,
            "kv_cache_write_asym_k_givens3_batched",
        )?;
        {
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = positions.buf.as_ptr();
            let mut ctp = cos_theta.buf.as_ptr();
            let mut stp = sin_theta.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut bs = batch_size as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut ctp as *mut _ as *mut c_void,
                &mut stp as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut bs as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            self.launch_maybe_blob(
                "kv_cache_write_asym_k_givens3_batched",
                [n_kv_heads as u32, batch_size as u32, 1],
                [32, 1, 1],
                shared_mem,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(kdp); b.push_ptr(ksp); b.push_ptr(pp);
                    b.push_ptr(ctp); b.push_ptr(stp);
                    b.push_i32(nkv); b.push_i32(hd); b.push_i32(bs);
                    b
                },
            )?;
        }
        // V: batched Q8_0 write.
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }

    /// Batched K+V write for fwht3 (K FWHT-rotated 3-bit + V Q8_0).
    pub fn kv_cache_write_fwht3_batched(
        &mut self,
        k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, positions: &GpuTensor,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_kv_heads: usize, head_dim: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_fwht3_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_FWHT3_BATCHED_SRC,
            "kv_cache_write_asym_k_fwht3_batched",
        )?;
        {
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = positions.buf.as_ptr();
            let mut s1p = signs1.buf.as_ptr();
            let mut s2p = signs2.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut bs = batch_size as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut s1p as *mut _ as *mut c_void,
                &mut s2p as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut bs as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            self.launch_maybe_blob(
                "kv_cache_write_asym_k_fwht3_batched",
                [n_kv_heads as u32, batch_size as u32, 1],
                [32, 1, 1],
                shared_mem,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(kdp); b.push_ptr(ksp); b.push_ptr(pp);
                    b.push_ptr(s1p); b.push_ptr(s2p);
                    b.push_i32(nkv); b.push_i32(hd); b.push_i32(bs);
                    b
                },
            )?;
        }
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }

    /// Batched flash attention for asym3 KV.
    /// Grid: [n_heads, max_tiles, sub_batch] tile + [n_heads, sub_batch] reduce,
    /// chunked by partials buffer capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym3_batched(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_flash_asym3_batched_masked(
            q, k_cache, v_cache, out, positions, cos_theta, sin_theta,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size, partials,
            None, 0, 0,
        )
    }

    /// Tree-mask variant of `attention_flash_asym3_batched`. asym3 is the
    /// default live KV path on 9B MQ4 — this is the primary target for
    /// DDTree batched verify on the hybrid arch.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym3_batched_masked(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
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
            q, k_cache, v_cache, out, positions, cos_theta, sin_theta,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size, partials,
            tree_bias, block_start, block_cols,
        )
    }

    /// Batched flash attention for fwht3 (K FWHT-rotated 3-bit + V Q8_0).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_fwht3_batched(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_flash_fwht3_batched_masked(
            q, k_cache, v_cache, out, positions, signs1, signs2,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size, partials,
            None, 0, 0,
        )
    }

    /// Tree-mask variant of `attention_flash_fwht3_batched`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_fwht3_batched_masked(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, positions: &GpuTensor,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, max_ctx_len: usize, batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_fwht3_tile_batched",
            kernels::ATTENTION_FLASH_FWHT3_TILE_BATCHED_SRC,
            "attention_flash_fwht3_tile_batched",
            q, k_cache, v_cache, out, positions, signs1, signs2,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size, partials,
            tree_bias, block_start, block_cols,
        )
    }

    /// Flash attention for asym3 KV (K at 3-bit rotated, V at Q8_0).
    /// Reuses Q8_0 flash reduce (output in normal space — V was un-rotated).
    /// Flash attention for fwht3 KV (K FWHT-rotated 3-bit, V at Q8_0).
    pub fn attention_flash_fwht3(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.graphs.capture_mode { max_tiles } else { actual_tiles };

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
            let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32; let mut ms = max_seq as i32;
            let mut sc = scale; let mut ts = TILE_SIZE as i32;
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
                    func, [n_heads as u32, 1, 1], [32, 1, 1], 0,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        Ok(())
    }

    pub fn attention_flash_asym3(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.graphs.capture_mode { max_tiles } else { actual_tiles };

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
            let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32; let mut ms = max_seq as i32;
            let mut sc = scale; let mut ts = TILE_SIZE as i32;
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
                    func, [n_heads as u32, 1, 1], [32, 1, 1], 0,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        Ok(())
    }

    /// Fused K+V write for asym2: K at givens2 (rotated 2-bit), V at Q8_0 (normal space).
    pub fn kv_cache_write_asym2_fused(
        &mut self, k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_givens2",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS2_SRC,
            "kv_cache_write_asym_k_givens2",
        )?;
        {
            let func = &self.functions["kv_cache_write_asym_k_givens2"];
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = pos_buf.as_ptr();
            let mut ctp = cos_theta.buf.as_ptr();
            let mut stp = sin_theta.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut ctp as *mut _ as *mut c_void,
                &mut stp as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            unsafe {
                self.hip.launch_kernel(
                    func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared_mem,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }

    /// Fused K+V write for fwht2: K at FWHT-rotated 2-bit, V at Q8_0.
    pub fn kv_cache_write_fwht2_fused(
        &mut self, k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_fwht2",
            kernels::KV_CACHE_WRITE_ASYM_K_FWHT2_SRC,
            "kv_cache_write_asym_k_fwht2",
        )?;
        {
            let func = &self.functions["kv_cache_write_asym_k_fwht2"];
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = pos_buf.as_ptr();
            let mut s1p = signs1.buf.as_ptr();
            let mut s2p = signs2.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut s1p as *mut _ as *mut c_void,
                &mut s2p as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            unsafe {
                self.hip.launch_kernel(
                    func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared_mem,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }

    /// Flash attention for asym4 KV (K at rotated 4-bit, V at Q8_0 normal space).
    /// Reuses the Q8_0 flash reduce since V was un-rotated — no inverse rotation needed.
    pub fn attention_flash_asym4(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.graphs.capture_mode { max_tiles } else { actual_tiles };

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
            let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32; let mut ms = max_seq as i32;
            let mut sc = scale; let mut ts = TILE_SIZE as i32;
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
                    func, [n_heads as u32, 1, 1], [32, 1, 1], 0,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        Ok(())
    }

    /// Flash attention for fwht4 KV (K FWHT-rotated 4-bit, V at Q8_0 normal space).
    /// Same launch geometry + Q8_0 reduce as asym4 — only the tile kernel differs.
    pub fn attention_flash_fwht4(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.graphs.capture_mode { max_tiles } else { actual_tiles };

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
            let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32; let mut ms = max_seq as i32;
            let mut sc = scale; let mut ts = TILE_SIZE as i32;
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
                    func, [n_heads as u32, 1, 1], [32, 1, 1], 0,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        Ok(())
    }

    /// Flash attention for asym2 KV (K at rotated 2-bit, V at Q8_0 normal space).
    /// Flash attention for fwht2 KV (K FWHT-rotated 2-bit, V at Q8_0).
    pub fn attention_flash_fwht2(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.graphs.capture_mode { max_tiles } else { actual_tiles };

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
            let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32; let mut ms = max_seq as i32;
            let mut sc = scale; let mut ts = TILE_SIZE as i32;
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
                    func, [n_heads as u32, 1, 1], [32, 1, 1], 0,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        Ok(())
    }

    pub fn attention_flash_asym2(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor, sin_theta: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.graphs.capture_mode { max_tiles } else { actual_tiles };

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
            let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32; let mut ms = max_seq as i32;
            let mut sc = scale; let mut ts = TILE_SIZE as i32;
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
                    func, [n_heads as u32, 1, 1], [32, 1, 1], 0,
                    self.stream_ref(), &mut params,
                )?;
            }
        }
        Ok(())
    }

    /// Attention with Q8_0 quantized KV cache.
    pub fn attention_q8_0_kv(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_q8_0_kv", kernels::ATTENTION_Q8_0_KV_SRC, "attention_q8_0_kv")?;
        let func = &self.functions["attention_q8_0_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr(); let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr(); let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void, &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void, &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        // Extra shared mem for Q head vector preloaded into shared memory
        let shared_mem = ((seq_len_hint + block_size as usize + head_dim) * 4) as u32;
        let bytes = crate::profile::attention_q8_0_kv_bytes(n_heads, n_kv_heads, head_dim, seq_len_hint);
        let timer = crate::profile::begin_timer(&self.hip, "attention", "attention_q8_0_kv", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
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
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
        cycle_counts: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_q8_0_kv_timed", kernels::ATTENTION_Q8_0_KV_TIMED_SRC, "attention_q8_0_kv_timed")?;
        let func = &self.functions["attention_q8_0_kv_timed"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr(); let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr(); let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut cc_ptr = cycle_counts.buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void, &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void, &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cc_ptr as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
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

    /// Compact a KV cache row by row: writes `dst[new_pos] = src[retain[new_pos]]`
    /// for `new_pos` in `[0, budget)`. Works for any byte-addressable cache
    /// layout — pass the layout's bytes-per-position.
    ///
    /// `retain_indices` must live on the device. Caller allocates `dst` with
    /// at least `budget × bytes_per_pos` bytes of capacity.
    pub fn kv_compact_gather(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        retain_indices: &GpuTensor,
        bytes_per_pos: usize,
        budget: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_compact_gather",
            kernels::KV_COMPACT_GATHER_SRC,
            "kv_compact_gather",
        )?;
        let func = &self.functions["kv_compact_gather"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut rp = retain_indices.buf.as_ptr();
        let mut bpp = bytes_per_pos as i32;
        let mut b = budget as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut bpp as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        // Choose thread count to saturate per-row bandwidth: ~1 thread per
        // 16-byte chunk, capped at 256 threads per block.
        let threads = ((bytes_per_pos / 16) as u32).clamp(32, 256);
        unsafe {
            self.hip.launch_kernel(
                func,
                [budget as u32, 1, 1],
                [threads, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// CASK m-folding merge for Q8_0 KV cache (arXiv:2604.10900).
    ///
    /// Computes `budget` output rows from `budget × m` source rows via
    /// weighted average + per-block requantization. Core (singleton)
    /// slots are handled uniformly by the caller: set `src_indices[s×m]`
    /// to the core source position and `src_weights[s×m] = 1.0`, rest = 0.
    ///
    /// All tensors live on the device. Caller allocates `dst` with at
    /// least `budget × n_kv × n_blocks × 34` bytes.
    pub fn kv_fold_q8(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        src_indices: &GpuTensor,     // [budget × m] i32
        src_weights: &GpuTensor,     // [budget × m] f32
        n_kv: usize,
        n_blocks: usize,
        m: usize,
        budget: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_fold_q8",
            kernels::KV_FOLD_Q8_SRC,
            "kv_fold_q8",
        )?;
        let func = &self.functions["kv_fold_q8"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut ip = src_indices.buf.as_ptr();
        let mut wp = src_weights.buf.as_ptr();
        let mut nkv = n_kv as i32;
        let mut nb = n_blocks as i32;
        let mut mi = m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [budget as u32, n_kv as u32, n_blocks as u32],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// CASK m-folding merge for asym3 K (givens3).
    ///
    /// Same calling convention as `kv_fold_q8` but takes `head_dim` (whole head)
    /// since asym3 doesn't block-wise split. One thread block per
    /// (slot, kv_head), 32 threads.
    pub fn kv_fold_asym3(
        &mut self,
        src: &GpuTensor, dst: &GpuTensor,
        src_indices: &GpuTensor, src_weights: &GpuTensor,
        n_kv: usize, head_dim: usize, m: usize, budget: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel("kv_fold_asym3", kernels::KV_FOLD_ASYM3_SRC, "kv_fold_asym3")?;
        let func = &self.functions["kv_fold_asym3"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut ip = src_indices.buf.as_ptr();
        let mut wp = src_weights.buf.as_ptr();
        let mut nkv = n_kv as i32;
        let mut hd = head_dim as i32;
        let mut mi = m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func, [budget as u32, n_kv as u32, 1],
                [32, 1, 1], 0, self.stream_ref(), &mut params,
            )
        }
    }

    /// CASK m-folding merge for asym4 K (givens4).
    pub fn kv_fold_asym4(
        &mut self,
        src: &GpuTensor, dst: &GpuTensor,
        src_indices: &GpuTensor, src_weights: &GpuTensor,
        n_kv: usize, head_dim: usize, m: usize, budget: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel("kv_fold_asym4", kernels::KV_FOLD_ASYM4_SRC, "kv_fold_asym4")?;
        let func = &self.functions["kv_fold_asym4"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut ip = src_indices.buf.as_ptr();
        let mut wp = src_weights.buf.as_ptr();
        let mut nkv = n_kv as i32;
        let mut hd = head_dim as i32;
        let mut mi = m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func, [budget as u32, n_kv as u32, 1],
                [32, 1, 1], 0, self.stream_ref(), &mut params,
            )
        }
    }

    /// CASK m-folding merge for asym2 K (givens2).
    pub fn kv_fold_asym2(
        &mut self,
        src: &GpuTensor, dst: &GpuTensor,
        src_indices: &GpuTensor, src_weights: &GpuTensor,
        n_kv: usize, head_dim: usize, m: usize, budget: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel("kv_fold_asym2", kernels::KV_FOLD_ASYM2_SRC, "kv_fold_asym2")?;
        let func = &self.functions["kv_fold_asym2"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut ip = src_indices.buf.as_ptr();
        let mut wp = src_weights.buf.as_ptr();
        let mut nkv = n_kv as i32;
        let mut hd = head_dim as i32;
        let mut mi = m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func, [budget as u32, n_kv as u32, 1],
                [32, 1, 1], 0, self.stream_ref(), &mut params,
            )
        }
    }

    /// Write KV vector to Q8 (int8 symmetric) quantized cache.
    pub fn kv_cache_write_q8(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_q8", kernels::KV_CACHE_WRITE_Q8_SRC, "kv_cache_write_q8")?;
        let func = &self.functions["kv_cache_write_q8"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let block = 64u32.min(head_dim as u32);
        let shared = (block * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [block, 1, 1], shared, self.stream_ref(), &mut params) }
    }

    /// Attention with Q8 quantized KV cache.
    pub fn attention_q8kv(
        &mut self, q: &GpuTensor, k_cache_q8: &GpuTensor, v_cache_q8: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_q8kv", kernels::ATTENTION_Q8KV_SRC, "attention_q8kv")?;
        let func = &self.functions["attention_q8kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr(); let mut k_ptr = k_cache_q8.buf.as_ptr();
        let mut v_ptr = v_cache_q8.buf.as_ptr(); let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void, &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void, &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Write KV vector to quantized HFQ4 cache.
    pub fn kv_cache_write_q4(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_q4", kernels::KV_CACHE_WRITE_Q4_SRC, "kv_cache_write_q4")?;
        let func = &self.functions["kv_cache_write_q4"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let block = 64u32.min(head_dim as u32);
        let shared = (block * 2 * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [block, 1, 1], shared, self.stream_ref(), &mut params) }
    }

    /// Attention with quantized HFQ4 KV cache — dequantizes K/V on the fly.
    pub fn attention_q4kv(
        &mut self, q: &GpuTensor, k_cache_q4: &GpuTensor, v_cache_q4: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_q4kv", kernels::ATTENTION_Q4KV_SRC, "attention_q4kv")?;
        let func = &self.functions["attention_q4kv"];
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
            &mut q_ptr as *mut _ as *mut c_void, &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void, &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// GPU-side KV cache write. Copies kv_dim floats from src to dst[pos_buf[0] * kv_dim].
    pub fn kv_cache_write(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        kv_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write", kernels::KV_CACHE_WRITE_SRC, "kv_cache_write")?;
        let func = &self.functions["kv_cache_write"];

        let dst_ptr = dst.buf.as_ptr();
        let src_ptr = src.buf.as_ptr();
        let pos_ptr = pos_buf.as_ptr();
        let kd = kv_dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &dst_ptr as *const _ as *mut c_void,
            &src_ptr as *const _ as *mut c_void,
            &pos_ptr as *const _ as *mut c_void,
            &kd as *const _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = (kv_dim as u32 + block - 1) / block;

        self.launch_maybe_blob(
            "kv_cache_write", [grid, 1, 1], [block, 1, 1], 0, &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(dst_ptr); b.push_ptr(src_ptr); b.push_ptr(pos_ptr);
                b.push_i32(kd);
                b
            },
        )
    }

    /// Batched F32 KV-cache write: scatter `batch_size` rows of `src`
    /// (`[batch_size * kv_dim]`) into the F32 cache at the absolute
    /// positions in `positions` (`[batch_size]` i32), in one launch.
    /// Batched-prefill replacement for the per-position `kv_cache_write`.
    pub fn kv_cache_write_f32_batched(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        positions: &GpuTensor,
        kv_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_cache_write_f32_batched", kernels::KV_CACHE_WRITE_F32_BATCHED_SRC, "kv_cache_write_f32_batched")?;
        let func = &self.functions["kv_cache_write_f32_batched"];

        let mut dst_ptr = dst.buf.as_ptr();
        let mut src_ptr = src.buf.as_ptr();
        let mut pos_ptr = positions.buf.as_ptr();
        let mut kd = kv_dim as i32;
        let mut bs = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut dst_ptr as *mut _ as *mut c_void,
            &mut src_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut kd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid_x = (kv_dim as u32 + block - 1) / block;

        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_size as u32, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Fused ViT self-attention: reads QKV [N, 3*hidden], writes out [N, hidden].
    pub fn vit_attention_f32(
        &mut self, qkv: &GpuTensor, out: &GpuTensor,
        n: usize, hidden: usize, num_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("vit_attention_f32", kernels::VIT_ATTENTION_SRC, "vit_attention_f32")?;
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
        unsafe { self.hip.launch_kernel(func, [num_heads as u32, n as u32, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Apply 2D rotary positional embedding to the Q and K halves of a packed
    /// QKV buffer for the Qwen3.5-VL vision tower (V is left untouched).
    ///
    /// `cos_t` and `sin_t` are shaped `[N, head_dim/2]` and are looked up
    /// per-(token, d) pair; the kernel reuses the same scalar for both
    /// `d < head_dim/2` and `d + head_dim/2` halves (HF concatenates
    /// `(rotary_pos_emb, rotary_pos_emb)` along the last dim before the
    /// trig table, so the two halves see the same angle).
    ///
    /// Grid=[num_heads, N], Block=[head_dim/2].
    pub fn apply_rope_2d_vision_f32(
        &mut self,
        qkv: &GpuTensor, cos_t: &GpuTensor, sin_t: &GpuTensor,
        n: usize, hidden: usize, num_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("apply_rope_2d_vision", kernels::APPLY_ROPE_2D_VISION_SRC, "apply_rope_2d_vision_f32")?;
        let func = &self.functions["apply_rope_2d_vision_f32"];
        let mut qp = qkv.buf.as_ptr();
        let mut cp = cos_t.buf.as_ptr();
        let mut sp = sin_t.buf.as_ptr();
        let mut ni = n as i32;
        let mut hi = hidden as i32;
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut cp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut hi as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let half = (head_dim / 2) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [num_heads as u32, n as u32, 1],
                [half, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Optimized vision attention with tiled K/V loading and 4 queries per block.
    /// ~3-5x faster than vit_attention_f32 via shared memory reuse.
    /// Grid=[num_heads, ceil(N/4)], Block=[256].
    pub fn vit_attention_opt(
        &mut self, qkv: &GpuTensor, out: &GpuTensor,
        n: usize, hidden: usize, num_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("vit_attention_opt", kernels::VIT_ATTENTION_OPT_SRC, "vit_attention_opt")?;
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
        // LDS: K_TILE * head_dim * 4 + N * 4 + 256 * 4
        let k_tile = 64u32;
        let shared_mem = (k_tile * head_dim as u32 * 4) + (n as u32 * 4) + (256 * 4);
        unsafe { self.hip.launch_kernel(func, [num_heads as u32, grid_y, 1], [256, 1, 1], shared_mem, self.stream_ref(), &mut params) }
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
        q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, out: &GpuTensor,
        b: usize, l: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention_dflash_f32", kernels::ATTENTION_DFLASH_SRC, "attention_dflash_f32")?;
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
        q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, out: &GpuTensor,
        b: usize, l: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
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
        q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, out: &GpuTensor,
        b: usize, l: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
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
                [64, 1, 1],   // 2 waves
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
        q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, out: &GpuTensor,
        b: usize, l: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
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
                [64, 1, 1],   // 2 waves
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
        q: &GpuTensor, k_f16: &GpuTensor, v_f16: &GpuTensor, out: &GpuTensor,
        b: usize, l: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(q.dtype, DType::F32, "attention_dflash_wmma_n64_f16kv_f32: q must be F32");
        assert_eq!(k_f16.dtype, DType::F16, "attention_dflash_wmma_n64_f16kv_f32: k must be F16");
        assert_eq!(v_f16.dtype, DType::F16, "attention_dflash_wmma_n64_f16kv_f32: v must be F16");
        assert_eq!(out.dtype, DType::F32, "attention_dflash_wmma_n64_f16kv_f32: out must be F32");
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
                [64, 1, 1],   // 2 waves
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
        q: &GpuTensor, k_f16: &GpuTensor, v_f16: &GpuTensor, out: &GpuTensor,
        b: usize, l: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(q.dtype, DType::F32, "attention_dflash_wmma_n128_f16kv_f32: q must be F32");
        assert_eq!(k_f16.dtype, DType::F16, "attention_dflash_wmma_n128_f16kv_f32: k must be F16");
        assert_eq!(v_f16.dtype, DType::F16, "attention_dflash_wmma_n128_f16kv_f32: v must be F16");
        assert_eq!(out.dtype, DType::F32, "attention_dflash_wmma_n128_f16kv_f32: out must be F32");
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
                [64, 1, 1],   // 2 waves
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
        q: &GpuTensor, k_f16: &GpuTensor, v_f16: &GpuTensor, out: &GpuTensor,
        b: usize, l: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(q.dtype, DType::F32, "attention_dflash_wmma_m64_n128_f16kv_f32: q must be F32");
        assert_eq!(k_f16.dtype, DType::F16, "attention_dflash_wmma_m64_n128_f16kv_f32: k must be F16");
        assert_eq!(v_f16.dtype, DType::F16, "attention_dflash_wmma_m64_n128_f16kv_f32: v must be F16");
        assert_eq!(out.dtype, DType::F32, "attention_dflash_wmma_m64_n128_f16kv_f32: out must be F32");
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
                [128, 1, 1],   // 4 waves
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
        q: &GpuTensor, k_f16: &GpuTensor, v_f16: &GpuTensor, out: &GpuTensor,
        b: usize, l: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(q.dtype, DType::F32, "attention_dflash_wmma_m64_n128_f16kv_v3_f32: q must be F32");
        assert_eq!(k_f16.dtype, DType::F16, "attention_dflash_wmma_m64_n128_f16kv_v3_f32: k must be F16");
        assert_eq!(v_f16.dtype, DType::F16, "attention_dflash_wmma_m64_n128_f16kv_v3_f32: v must be F16");
        assert_eq!(out.dtype, DType::F32, "attention_dflash_wmma_m64_n128_f16kv_v3_f32: out must be F32");
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
        q: &GpuTensor, k_f16: &GpuTensor, v_f16: &GpuTensor, out: &GpuTensor,
        b: usize, l: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
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
        assert_eq!(q.dtype, DType::F32, "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32: q must be F32");
        assert_eq!(k_f16.dtype, DType::F16, "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32: k must be F16");
        assert_eq!(v_f16.dtype, DType::F16, "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32: v must be F16");
        assert_eq!(out.dtype, DType::F32, "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32: out must be F32");
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
        let func = &self.functions["attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32"];
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

    /// V2 of `attention_dflash_wmma_m64_n128_f16kv_f32`. Same shape
    /// (M=64, N=128, 4-wave block, f16 K/V, O in registers) but adds
    /// (a) S_lds row stride 130 (was 128) to break a 16-way LDS bank
    /// conflict in phase C's S_lds reads, and (b) cooperative wave-32
    /// softmax via __shfl_xor butterfly.
    pub fn attention_dflash_wmma_m64_n128_f16kv_v2_f32(
        &mut self,
        q: &GpuTensor, k_f16: &GpuTensor, v_f16: &GpuTensor, out: &GpuTensor,
        b: usize, l: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(q.dtype, DType::F32, "attention_dflash_wmma_m64_n128_f16kv_v2_f32: q must be F32");
        assert_eq!(k_f16.dtype, DType::F16, "attention_dflash_wmma_m64_n128_f16kv_v2_f32: k must be F16");
        assert_eq!(v_f16.dtype, DType::F16, "attention_dflash_wmma_m64_n128_f16kv_v2_f32: v must be F16");
        assert_eq!(out.dtype, DType::F32, "attention_dflash_wmma_m64_n128_f16kv_v2_f32: out must be F32");
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
                [128, 1, 1],   // 4 waves
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    // ═══════════════════════════════════════════════════════════════════════════
    // PFlash scoring
    // ═══════════════════════════════════════════════════════════════════════════

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
        assert!(head_dim % 32 == 0, "head_dim must be a multiple of 32 for Q8 KV cache");
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
            k_cache, scores_out, n_pos, n_kv_heads, head_dim,
            block_size, n_blocks, last_pos,
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
            k_cache, scores_out, n_pos, n_kv_heads, head_dim,
            block_size, n_blocks, last_pos,
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
            k_cache, scores_out, n_pos, n_kv_heads, head_dim,
            block_size, n_blocks, last_pos,
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

}
