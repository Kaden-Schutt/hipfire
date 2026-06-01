// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-side embedding lookup dispatch methods (HFQ4G256, HFQ4G128, Q8, Q4K).

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;

impl Gpu {
    /// GPU-side embedding lookup: copy row `token_id` from embedding table to output.
    /// Avoids downloading the entire embedding table to CPU.
    pub fn embedding_lookup(
        &self,
        table: &GpuTensor,  // [vocab_size * dim] F32
        output: &GpuTensor, // [dim] F32
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let byte_offset = (token_id as usize) * dim * 4;
        let byte_size = dim * 4;
        self.hip.memcpy_dtod_offset(&output.buf, &table.buf, byte_offset, byte_size)
    }

    /// Q4_LUT GEMV: 4-bit with LDS codebook lookup. 48 bytes per 32 elements.

    /// Wave-cooperative Q4 GEMV (Q4_F16_G32 format, 0.625 B/w). Shuffle-based nibble distribution.

    /// Q4-as-Q8 GEMV: 4-bit precision stored in Q8_0 format (1.0625 B/w). Gets Q8 occupancy.

    /// Q8_0 embedding lookup: dequantize one row on GPU, output F32.
    pub fn embedding_lookup_q8(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("embedding_q8", kernels::EMBEDDING_Q8_SRC, "embedding_q8")?;
        let func = &self.functions["embedding_q8"];

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tid = token_id as i32;
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, None, &mut params)
        }
    }

    /// Q4_K embedding lookup: dequantize one row on GPU, output F32.
    /// table is raw Q4_K bytes on GPU, output is [dim] F32.
    pub fn embedding_lookup_q4k(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("embedding_q4k", kernels::EMBEDDING_Q4K_SRC, "embedding_q4k")?;
        let func = &self.functions["embedding_q4k"];

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tid = token_id as i32;
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, None, &mut params)
        }
    }

    /// HFQ4-G256 embedding lookup: dequantize one row on GPU, output F32.
    pub fn embedding_lookup_hfq4g256(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("embedding_hfq4g256", kernels::EMBEDDING_HFQ4G256_SRC, "embedding_hfq4g256")?;
        let func = &self.functions["embedding_hfq4g256"];

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tid = token_id as i32;
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        let bytes = crate::profile::embedding_hfq4g256_bytes(dim);
        let timer = crate::profile::begin_timer(&self.hip, "embedding", "embedding_lookup_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched Q8_0 embedding lookup. Same hipGraph-captureable pattern as
    /// the HFQ4G256 variant. `output` shape: `[n × dim]` row-major.
    pub fn embedding_lookup_q8_batched(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_ids: &GpuTensor,
        n: usize,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "embedding_q8_batched",
            kernels::EMBEDDING_Q8_BATCHED_SRC,
            "embedding_q8_batched",
        )?;

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tidp = token_ids.buf.as_ptr();
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tidp as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "embedding_q8_batched",
            [n as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp); b.push_ptr(op); b.push_ptr(tidp); b.push_i32(d);
                b
            },
        )
    }

    /// Batched HFQ4-G256 embedding lookup. Dequantizes N rows in a single
    /// launch, reading token ids from a device buffer. hipGraph-capture-safe:
    /// callers update `token_ids` between replays and replay the same graph.
    ///
    /// `output` shape: `[n × dim]` row-major. `token_ids` shape: `[n]` i32.
    pub fn embedding_lookup_hfq4g256_batched(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_ids: &GpuTensor,
        n: usize,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "embedding_hfq4g256_batched",
            kernels::EMBEDDING_HFQ4G256_BATCHED_SRC,
            "embedding_hfq4g256_batched",
        )?;

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tidp = token_ids.buf.as_ptr();
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tidp as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "embedding_hfq4g256_batched",
            [n as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp); b.push_ptr(op); b.push_ptr(tidp); b.push_i32(d);
                b
            },
        )
    }

    /// HFQ4-G128 embedding lookup: dequantize one row on GPU, output F32.
    pub fn embedding_lookup_hfq4g128(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("embedding_hfq4g128", kernels::EMBEDDING_HFQ4G128_SRC, "embedding_hfq4g128")?;
        let func = &self.functions["embedding_hfq4g128"];

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tid = token_id as i32;
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params)
        }
    }
}
