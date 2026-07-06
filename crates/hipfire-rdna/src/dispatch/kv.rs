// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! KV cache write/read dispatch (all quant formats) + KVarN sliding-window. Pure move (Phase 1 M3).

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipResult};
use std::ffi::c_void;

impl Gpu {
    /// Write KV to HFQ4 co-located block (72 bytes per head: scale+zero+nibbles).
    pub fn kv_cache_write_hfq4(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_hfq4",
            kernels::KV_CACHE_WRITE_HFQ4_SRC,
            "kv_cache_write_hfq4",
        )?;
        let func = &self.functions["kv_cache_write_hfq4"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// INT8 co-located with f16 scale (matches Q8_0 precision, one block per head).
    pub fn kv_cache_write_int8c_f16(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_int8c_f16",
            kernels::KV_CACHE_WRITE_INT8C_F16_SRC,
            "kv_cache_write_int8c_f16",
        )?;
        let func = &self.functions["kv_cache_write_int8c_f16"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Write KV to INT8 co-located block (f32 scale + int8 data, symmetric).
    pub fn kv_cache_write_int8c(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_int8c",
            kernels::KV_CACHE_WRITE_INT8C_SRC,
            "kv_cache_write_int8c",
        )?;
        let func = &self.functions["kv_cache_write_int8c"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Write KV to HFQ8 cache (FP32 scale+zero, contiguous uint8).
    pub fn kv_cache_write_hfq8(
        &mut self,
        dst_data: &GpuTensor,
        dst_scales: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_hfq8",
            kernels::KV_CACHE_WRITE_HFQ8_SRC,
            "kv_cache_write_hfq8",
        )?;
        let func = &self.functions["kv_cache_write_hfq8"];
        let mut dd = dst_data.buf.as_ptr();
        let mut ds = dst_scales.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dd as *mut _ as *mut c_void,
            &mut ds as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Write KV to INT8 cache (separate scale array).
    pub fn kv_cache_write_int8(
        &mut self,
        dst_vals: &GpuTensor,
        dst_scales: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_int8",
            kernels::KV_CACHE_WRITE_INT8_SRC,
            "kv_cache_write_int8",
        )?;
        let func = &self.functions["kv_cache_write_int8"];
        let mut dv = dst_vals.buf.as_ptr();
        let mut ds = dst_scales.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dv as *mut _ as *mut c_void,
            &mut ds as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Batched Q8_0 KV cache write: quantize multiple positions in one launch.
    pub fn kv_cache_write_q8_0_batched(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        positions: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_q8_0_batched",
            kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC,
            "kv_cache_write_q8_0_batched",
        )?;
        let d = dst.buf.as_ptr();
        let s = src.buf.as_ptr();
        let p = positions.buf.as_ptr();
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let bs = batch_size as i32;
        let total_blocks = (n_kv_heads * head_dim / 32) as u32;
        self.launch_kernargs(
            "kv_cache_write_q8_0_batched",
            [total_blocks, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr d, ptr s, ptr p, i32 nkv, i32 hd, i32 bs],
        )
    }
    /// Write KV vector to Q8_0 quantized cache (same format as GGML Q8_0).
    pub fn kv_cache_write_q8_0(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_q8_0",
            kernels::KV_CACHE_WRITE_Q8_0_SRC,
            "kv_cache_write_q8_0",
        )?;
        let d = dst.buf.as_ptr();
        let s = src.buf.as_ptr();
        let p = pos_buf.as_ptr();
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let total_blocks = (n_kv_heads * head_dim / 32) as u32;
        let bytes = crate::profile::kv_cache_write_q8_0_bytes(n_kv_heads, head_dim);
        let timer =
            crate::profile::begin_timer(&self.hip, "kv_write", "kv_cache_write_q8_0", bytes);
        let result = self.launch_kernargs(
            "kv_cache_write_q8_0",
            [total_blocks, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr d, ptr s, ptr p, i32 nkv, i32 hd],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Fused K+V write for asym4: K at givens4 (rotated 4-bit), V at Q8_0 (normal space).
    /// Launches two kernels — K-only givens4 writer + standard Q8_0 writer.
    pub fn kv_cache_write_asym4_fused(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
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
                    func,
                    [n_kv_heads as u32, 1, 1],
                    [32, 1, 1],
                    shared_mem,
                    self.stream_ref(),
                    &mut params,
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
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        _v_mode_bits: i32,
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
                    func,
                    [n_kv_heads as u32, 1, 1],
                    [32, 1, 1],
                    shared_mem,
                    self.stream_ref(),
                    &mut params,
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
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
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
                    func,
                    [n_kv_heads as u32, 1, 1],
                    [32, 1, 1],
                    shared_mem,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }
    /// Fused K+V write for fwht3: K at signed-FWHT-256 rotated 3-bit, V at Q8_0.
    /// Byte-identical storage to asym3 — only the K-write kernel differs.
    pub fn kv_cache_write_fwht3_fused(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        _v_mode_bits: i32,
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
                    func,
                    [n_kv_heads as u32, 1, 1],
                    [32, 1, 1],
                    shared_mem,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }
    /// Batched K+V write for asym4 (K 4-bit rotated + V Q8_0).
    pub fn kv_cache_write_asym4_batched(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_k_batched(
            "kv_cache_write_asym_k_givens4_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS4_BATCHED_SRC,
            "kv_cache_write_asym_k_givens4_batched",
            k_dst,
            k_src,
            positions,
            cos_theta,
            sin_theta,
            n_kv_heads,
            head_dim,
            batch_size,
        )?;
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }
    /// Batched K+V write for fwht4 (K FWHT-rotated 4-bit + V Q8_0).
    /// Same launch geometry as asym4_batched; only the kernel name + sign-vector
    /// param semantics differ.
    pub fn kv_cache_write_fwht4_batched(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
        _v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_k_batched(
            "kv_cache_write_asym_k_fwht4_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_FWHT4_BATCHED_SRC,
            "kv_cache_write_asym_k_fwht4_batched",
            k_dst,
            k_src,
            positions,
            signs1,
            signs2,
            n_kv_heads,
            head_dim,
            batch_size,
        )?;
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }
    /// Batched K+V write for asym2 (K 2-bit rotated + V Q8_0).
    pub fn kv_cache_write_asym2_batched(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_k_batched(
            "kv_cache_write_asym_k_givens2_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS2_BATCHED_SRC,
            "kv_cache_write_asym_k_givens2_batched",
            k_dst,
            k_src,
            positions,
            cos_theta,
            sin_theta,
            n_kv_heads,
            head_dim,
            batch_size,
        )?;
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }
    /// Batched K+V write for fwht2 (K FWHT-rotated 2-bit + V Q8_0).
    pub fn kv_cache_write_fwht2_batched(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
        _v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_k_batched(
            "kv_cache_write_asym_k_fwht2_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_FWHT2_BATCHED_SRC,
            "kv_cache_write_asym_k_fwht2_batched",
            k_dst,
            k_src,
            positions,
            signs1,
            signs2,
            n_kv_heads,
            head_dim,
            batch_size,
        )?;
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }
    /// Batched K+V write for asym3 — processes N positions in one launch.
    /// K-only givens3 write (batched) + Q8_0 V write (batched).
    pub fn kv_cache_write_asym3_batched(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // K: batched 3-bit rotated write.
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_givens3_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_BATCHED_SRC,
            "kv_cache_write_asym_k_givens3_batched",
        )?;
        {
            let kdp = k_dst.buf.as_ptr();
            let ksp = k_src.buf.as_ptr();
            let pp = positions.buf.as_ptr();
            let ctp = cos_theta.buf.as_ptr();
            let stp = sin_theta.buf.as_ptr();
            let nkv = n_kv_heads as i32;
            let hd = head_dim as i32;
            let bs = batch_size as i32;
            let shared_mem = ((head_dim + 32) * 4) as u32;
            self.launch_kernargs(
                "kv_cache_write_asym_k_givens3_batched",
                [n_kv_heads as u32, batch_size as u32, 1],
                [32, 1, 1],
                shared_mem,
                &kernargs![ptr kdp, ptr ksp, ptr pp, ptr ctp, ptr stp, i32 nkv, i32 hd, i32 bs],
            )?;
        }
        // V: batched Q8_0 write.
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }
    /// Batched K+V write for fwht3 (K FWHT-rotated 3-bit + V Q8_0).
    pub fn kv_cache_write_fwht3_batched(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
        _v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_fwht3_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_FWHT3_BATCHED_SRC,
            "kv_cache_write_asym_k_fwht3_batched",
        )?;
        {
            let kdp = k_dst.buf.as_ptr();
            let ksp = k_src.buf.as_ptr();
            let pp = positions.buf.as_ptr();
            let s1p = signs1.buf.as_ptr();
            let s2p = signs2.buf.as_ptr();
            let nkv = n_kv_heads as i32;
            let hd = head_dim as i32;
            let bs = batch_size as i32;
            let shared_mem = ((head_dim + 32) * 4) as u32;
            self.launch_kernargs(
                "kv_cache_write_asym_k_fwht3_batched",
                [n_kv_heads as u32, batch_size as u32, 1],
                [32, 1, 1],
                shared_mem,
                &kernargs![ptr kdp, ptr ksp, ptr pp, ptr s1p, ptr s2p, i32 nkv, i32 hd, i32 bs],
            )?;
        }
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }
    /// Fused K+V write for asym2: K at givens2 (rotated 2-bit), V at Q8_0 (normal space).
    pub fn kv_cache_write_asym2_fused(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
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
                    func,
                    [n_kv_heads as u32, 1, 1],
                    [32, 1, 1],
                    shared_mem,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }
    /// Fused K+V write for fwht2: K at FWHT-rotated 2-bit, V at Q8_0.
    pub fn kv_cache_write_fwht2_fused(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        _v_mode_bits: i32,
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
                    func,
                    [n_kv_heads as u32, 1, 1],
                    [32, 1, 1],
                    shared_mem,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
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
        src_indices: &GpuTensor, // [budget × m] i32
        src_weights: &GpuTensor, // [budget × m] f32
        n_kv: usize,
        n_blocks: usize,
        m: usize,
        budget: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_fold_q8", kernels::KV_FOLD_Q8_SRC, "kv_fold_q8")?;
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
        src: &GpuTensor,
        dst: &GpuTensor,
        src_indices: &GpuTensor,
        src_weights: &GpuTensor,
        n_kv: usize,
        head_dim: usize,
        m: usize,
        budget: usize,
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
                func,
                [budget as u32, n_kv as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// CASK m-folding merge for asym4 K (givens4).
    pub fn kv_fold_asym4(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        src_indices: &GpuTensor,
        src_weights: &GpuTensor,
        n_kv: usize,
        head_dim: usize,
        m: usize,
        budget: usize,
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
                func,
                [budget as u32, n_kv as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// CASK m-folding merge for asym2 K (givens2).
    pub fn kv_fold_asym2(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        src_indices: &GpuTensor,
        src_weights: &GpuTensor,
        n_kv: usize,
        head_dim: usize,
        m: usize,
        budget: usize,
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
                func,
                [budget as u32, n_kv as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Write KV vector to Q8 (int8 symmetric) quantized cache.
    pub fn kv_cache_write_q8(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_q8",
            kernels::KV_CACHE_WRITE_Q8_SRC,
            "kv_cache_write_q8",
        )?;
        let func = &self.functions["kv_cache_write_q8"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let block = 64u32.min(head_dim as u32);
        let shared = (block * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, 1, 1],
                [block, 1, 1],
                shared,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Write KV vector to quantized HFQ4 cache.
    pub fn kv_cache_write_q4(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_q4",
            kernels::KV_CACHE_WRITE_Q4_SRC,
            "kv_cache_write_q4",
        )?;
        let func = &self.functions["kv_cache_write_q4"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let block = 64u32.min(head_dim as u32);
        let shared = (block * 2 * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, 1, 1],
                [block, 1, 1],
                shared,
                self.stream_ref(),
                &mut params,
            )
        }
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
        self.ensure_kernel(
            "kv_cache_write",
            kernels::KV_CACHE_WRITE_SRC,
            "kv_cache_write",
        )?;

        let dst_ptr = dst.buf.as_ptr();
        let src_ptr = src.buf.as_ptr();
        let pos_ptr = pos_buf.as_ptr();
        let kd = kv_dim as i32;

        let block = 256u32;
        let grid = (kv_dim as u32 + block - 1) / block;

        self.launch_kernargs(
            "kv_cache_write",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &kernargs![ptr dst_ptr, ptr src_ptr, ptr pos_ptr, i32 kd],
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
        self.ensure_kernel(
            "kv_cache_write_f32_batched",
            kernels::KV_CACHE_WRITE_F32_BATCHED_SRC,
            "kv_cache_write_f32_batched",
        )?;

        let dst_ptr = dst.buf.as_ptr();
        let src_ptr = src.buf.as_ptr();
        let pos_ptr = positions.buf.as_ptr();
        let kd = kv_dim as i32;
        let bs = batch_size as i32;

        let block = 256u32;
        let grid_x = (kv_dim as u32 + block - 1) / block;
        self.launch_kernargs(
            "kv_cache_write_f32_batched",
            [grid_x, batch_size as u32, 1],
            [block, 1, 1],
            0,
            &kernargs![ptr dst_ptr, ptr src_ptr, ptr pos_ptr, i32 kd, i32 bs],
        )
    }
    /// Routed batched F32 KV-cache write: scatter `batch_size` rows of `src`
    /// (`[batch_size * kv_dim]`) into per-session F32 cache pointers selected
    /// by `row_session_indices` (`[batch_size]` i32), at absolute `positions`
    /// (`[batch_size]` i32), in one launch.
    ///
    /// Session-batched prefill uses this instead of `kv_cache_write_f32_batched`
    /// so rows from independent request sessions do not share one KV cache.
    pub fn kv_cache_write_f32_routed_batched(
        &mut self,
        dst_ptrs: &GpuTensor,
        src: &GpuTensor,
        row_session_indices: &GpuTensor,
        positions: &GpuTensor,
        ptr_layer_stride: usize,
        layer_index: usize,
        kv_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_f32_routed_batched",
            kernels::KV_CACHE_WRITE_F32_ROUTED_BATCHED_SRC,
            "kv_cache_write_f32_routed_batched",
        )?;

        let dst_ptrs_ptr = dst_ptrs.buf.as_ptr();
        let src_ptr = src.buf.as_ptr();
        let row_session_indices_ptr = row_session_indices.buf.as_ptr();
        let pos_ptr = positions.buf.as_ptr();
        let ptr_stride = ptr_layer_stride as i32;
        let layer = layer_index as i32;
        let kd = kv_dim as i32;
        let bs = batch_size as i32;

        let block = 256u32;
        let grid_x = (kv_dim as u32 + block - 1) / block;
        self.launch_kernargs(
            "kv_cache_write_f32_routed_batched",
            [grid_x, batch_size as u32, 1],
            [block, 1, 1],
            0,
            &kernargs![
                ptr dst_ptrs_ptr, ptr src_ptr, ptr row_session_indices_ptr, ptr pos_ptr,
                i32 ptr_stride, i32 layer, i32 kd, i32 bs
            ],
        )
    }
    /// Routed batched Q8_0 KV-cache write: quantize `batch_size` rows of
    /// `src` into per-session Q8_0 cache pointers selected by
    /// `row_session_indices`, at absolute `positions`.
    pub fn kv_cache_write_q8_0_routed_batched(
        &mut self,
        dst_ptrs: &GpuTensor,
        src: &GpuTensor,
        row_session_indices: &GpuTensor,
        positions: &GpuTensor,
        ptr_layer_stride: usize,
        layer_index: usize,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_q8_0_routed_batched",
            kernels::KV_CACHE_WRITE_Q8_0_ROUTED_BATCHED_SRC,
            "kv_cache_write_q8_0_routed_batched",
        )?;

        let dst_ptrs_ptr = dst_ptrs.buf.as_ptr();
        let src_ptr = src.buf.as_ptr();
        let row_session_indices_ptr = row_session_indices.buf.as_ptr();
        let pos_ptr = positions.buf.as_ptr();
        let ptr_stride = ptr_layer_stride as i32;
        let layer = layer_index as i32;
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let bs = batch_size as i32;

        let total_blocks = (n_kv_heads * head_dim / 32) as u32;
        self.launch_kernargs(
            "kv_cache_write_q8_0_routed_batched",
            [total_blocks, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![
                ptr dst_ptrs_ptr, ptr src_ptr, ptr row_session_indices_ptr, ptr pos_ptr,
                i32 ptr_stride, i32 layer, i32 nkv, i32 hd, i32 bs
            ],
        )
    }
    /// KVarN tile quantizer: variance-normalize (Sinkhorn) + 4-bit affine + pack
    /// to the on-device KVarN record. `tiles` = [n_tiles, r_dim*c_dim] f32; `recs`
    /// = [n_tiles, record_bytes] (kvarn_record_bytes(r,c)). c_dim must be even.
    /// One block per tile. Validated vs the kvarn.rs CPU oracle.
    pub fn kvarn_quantize_tile(
        &mut self,
        tiles: &GpuTensor,
        recs: &GpuTensor,
        n_tiles: usize,
        r_dim: usize,
        c_dim: usize,
        record_bytes: usize,
        bits: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            r_dim <= 256 && c_dim <= 256,
            "kvarn_quantize_tile: r,c must be <= 256"
        );
        assert!(
            matches!(bits, 2 | 4 | 8),
            "kvarn_quantize_tile: bits must be 2, 4, or 8"
        );
        assert_eq!(
            (c_dim * bits) % 8,
            0,
            "kvarn_quantize_tile: c_dim*bits must be a multiple of 8 (rows own whole bytes)"
        );
        self.ensure_kernel(
            "kvarn_quantize_tile",
            kernels::KVARN_QUANTIZE_TILE_SRC,
            "kvarn_quantize_tile",
        )?;
        let tp = tiles.buf.as_ptr();
        let rp = recs.buf.as_ptr();
        let mut nt = n_tiles as i32;
        let mut rd = r_dim as i32;
        let mut cd = c_dim as i32;
        let mut rb = record_bytes as i32;
        let mut bt = bits as i32;
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut rd as *mut _ as *mut c_void,
            &mut cd as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
        ];
        let func = &self.functions["kvarn_quantize_tile"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_tiles as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// KVarN tile dequantizer: unpack records → f16 tiles for the reused
    /// asym4/q8 flash attention. `recs` = [n_tiles, record_bytes]; `out` = f16
    /// [n_tiles, r_dim*c_dim]. One block per tile, zero LDS.
    pub fn kvarn_dequant_tile(
        &mut self,
        recs: &GpuTensor,
        out: &GpuTensor,
        n_tiles: usize,
        r_dim: usize,
        c_dim: usize,
        record_bytes: usize,
        bits: usize, // bits per code: 4 = legacy nibble layout, 2 = packed 2-bit
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kvarn_dequant_tile",
            kernels::KVARN_DEQUANT_TILE_SRC,
            "kvarn_dequant_tile",
        )?;
        let rp = recs.buf.as_ptr();
        let op = out.buf.as_ptr();
        let mut nt = n_tiles as i32;
        let mut rd = r_dim as i32;
        let mut cd = c_dim as i32;
        let mut rb = record_bytes as i32;
        let mut bt = bits as i32;
        let mut params: Vec<*mut c_void> = vec![
            &rp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut rd as *mut _ as *mut c_void,
            &mut cd as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
        ];
        let func = &self.functions["kvarn_dequant_tile"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_tiles as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// KVarN write-side gather: transpose a contiguous run of `n_blocks` token-
    /// major K blocks (`k` = [n_blocks*group, kv_dim] f32) into the channel-major
    /// `[head_dim × group]` tiles `kvarn_quantize_tile` expects (`tiles` =
    /// [n_blocks*n_kv_heads, head_dim*group] f32). Caller then runs
    /// `kvarn_quantize_tile` over `tiles` to fill the records. One block per tile.
    pub fn kvarn_gather_k_tiles(
        &mut self,
        k: &GpuTensor,
        tiles: &GpuTensor,
        n_blocks: usize,
        n_kv_heads: usize,
        head_dim: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kvarn_gather_k_tiles",
            kernels::KVARN_GATHER_K_TILES_SRC,
            "kvarn_gather_k_tiles",
        )?;
        let kp = k.buf.as_ptr();
        let tp = tiles.buf.as_ptr();
        let mut nb = n_blocks as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut gp = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &kp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
        ];
        let func = &self.functions["kvarn_gather_k_tiles"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [(n_blocks * n_kv_heads) as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// KVarN read-side build: materialize a token-major f16 shadow K cache
    /// (`out` = [n_full_blocks*group + tail_len, kv_dim] f16) from the block-tiled
    /// records `recs` (full blocks) + the f32 recent-window `window` (tail). The
    /// chosen v1 read path feeds this shadow into the f16-K / Q8-V flash kernel.
    /// One block per output token, zero LDS.
    #[allow(clippy::too_many_arguments)]
    pub fn kvarn_build_kcache(
        &mut self,
        recs: &GpuTensor,
        window: &GpuTensor,
        out: &GpuTensor,
        n_full_blocks: usize,
        tail_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
        group: usize,
        record_bytes: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kvarn_build_kcache",
            kernels::KVARN_BUILD_KCACHE_SRC,
            "kvarn_build_kcache",
        )?;
        let rp = recs.buf.as_ptr();
        let wp = window.buf.as_ptr();
        let op = out.buf.as_ptr();
        let mut nfb = n_full_blocks as i32;
        let mut tl = tail_len as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut gp = group as i32;
        let mut rb = record_bytes as i32;
        let mut params: Vec<*mut c_void> = vec![
            &rp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut nfb as *mut _ as *mut c_void,
            &mut tl as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
        ];
        let n_out_tokens = n_full_blocks * group + tail_len;
        let func = &self.functions["kvarn_build_kcache"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_out_tokens as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// End-to-end KVarN KV-write + attention for a contiguous run of `n` tokens
    /// at absolute positions `[start_pos, start_pos+n)`. Unifies prefill (`n>1`)
    /// and decode (`n==1`):
    ///   1. V → Q8_0 by position (reused `kv_cache_write_q8_0_batched`).
    ///   2. K → append rows to the f32 recent-window (slot = pos % GROUP); each
    ///      time a 128-token block completes, gather+quantize it into the
    ///      block-tiled records (`records` = `k_gpu[layer]`). The window holds
    ///      the trailing partial block.
    ///   3. Build a token-major f16 shadow K `[seq_len × kv_dim]` from the full
    ///      blocks (records) + the window tail (`kvarn_build_kcache`).
    ///   4. f16-K / Q8-V flash (`attention_flash_f16k_q8v_batched_masked`).
    ///
    /// `records`/`window` are F32-typed buffers (byte-addressed by the kernels).
    /// Scratch (shadow K, gather tiles) is pooled per call — v1 rebuilds the full
    /// history each step (correct, not yet perf-tuned; Phase-2 fuses dequant into
    /// the flash). Tree-verify (`tree_bias`) is passed through but the block
    /// write assumes contiguous causal positions — callers guard tree mode off.
    #[allow(clippy::too_many_arguments)]
    pub fn kvarn_attend(
        &mut self,
        records: &GpuTensor,
        window: &GpuTensor,
        v_cache: &GpuTensor,
        fa_q: &GpuTensor,
        fa_k: &GpuTensor,
        fa_v: &GpuTensor,
        positions: &GpuTensor,
        out: &GpuTensor,
        flash_partials: &GpuTensor,
        tiles: &GpuTensor,
        n: usize,
        start_pos: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        physical_cap: usize,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
        bits: usize,
    ) -> HipResult<()> {
        const GROUP: usize = 128;
        let kv_dim = n_kv_heads * head_dim;
        // kvarn_record_bytes_bits(head_dim, GROUP, bits): 8/bits K codes/byte +
        // fp16 scale_abs/zp_abs (per channel) + fp16 s_col (per token). Always a
        // multiple of 4 → expressible as an F32-typed byte-addressed buffer.
        let cpb = 8 / bits;
        let rec_bytes = (head_dim * GROUP).div_ceil(cpb) + head_dim * 2 * 2 + GROUP * 2;
        let seq_len = start_pos + n;

        // 1. V write (Q8_0) by absolute position.
        self.kv_cache_write_q8_0_batched(v_cache, fa_v, positions, n_kv_heads, head_dim, n)?;

        // 2. K write: append to window, flush each completed 128-token block.
        // `tiles`/`shadow` are caller-owned reusable scratch (see KvCache::kvarn_*).
        let mut written = 0usize;
        while written < n {
            let t = start_pos + written;
            let slot = t % GROUP;
            let block = t / GROUP;
            let take = (GROUP - slot).min(n - written);
            // Contiguous append: fa_k rows [written, written+take) → window slots
            // [slot, slot+take) (both token-major, kv_dim stride).
            self.memcpy_dtod_at_auto(
                &window.buf,
                slot * kv_dim * 4,
                &fa_k.buf,
                written * kv_dim * 4,
                take * kv_dim * 4,
            )?;
            written += take;
            if slot + take == GROUP {
                // Block complete in the window → gather + variance-norm 4-bit pack
                // into records[block].
                self.kvarn_gather_k_tiles(window, tiles, 1, n_kv_heads, head_dim, GROUP)?;
                let rec_off_elems = block * n_kv_heads * rec_bytes / 4;
                let rec_view = records.sub_offset(rec_off_elems, n_kv_heads * rec_bytes / 4);
                self.kvarn_quantize_tile(
                    tiles, &rec_view, n_kv_heads, head_dim, GROUP, rec_bytes, bits,
                )?;
            }
        }

        // 3. Fused KVarN flash over [0, seq_len): dequant the 4-bit records in
        // place for the `n_full_blocks` full tiles + read the f32 window for the
        // trailing partial tile. No f16 shadow build (Phase D2). Causal masking
        // via positions.
        let n_full_blocks = seq_len / GROUP;
        self.attention_flash_kvarn_batched_masked(
            fa_q,
            records,
            window,
            v_cache,
            out,
            positions,
            n_heads,
            n_kv_heads,
            head_dim,
            physical_cap,
            seq_len,
            n,
            flash_partials,
            tree_bias,
            block_start,
            block_cols,
            n_full_blocks,
            rec_bytes,
            bits,
        )
    }
}
