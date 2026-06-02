// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sampling and reduction dispatch methods
//! (argmax, top-k, top-p, log-sum-exp).

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;

impl Gpu {
    /// Compute max softmax probability on GPU. Downloads 4 bytes instead of vocab×4.
    pub fn max_prob(
        &mut self, logits: &GpuTensor, result: &GpuTensor, vocab_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("max_prob", kernels::MAX_PROB_SRC, "max_prob")?;
        let func = &self.functions["max_prob"];
        let mut lp = logits.buf.as_ptr();
        let mut rp = result.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void, &mut rp as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let shared = (block * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [1, 1, 1], [block, 1, 1], shared, self.stream_ref(), &mut params) }
    }

    /// GPU-side batched argmax: writes one i32 index per row into `result`
    /// (shape `[batch_size]`). Avoids downloading `batch_size × n` floats
    /// to the host — only `batch_size × 4` bytes land on PCIe.
    pub fn argmax_f32_batched(
        &mut self,
        data: &GpuTensor,
        result: &GpuTensor,
        n: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "argmax_f32_batched",
            kernels::ARGMAX_BATCHED_SRC,
            "argmax_f32_batched",
        )?;

        let mut dp = data.buf.as_ptr();
        let mut rp = result.buf.as_ptr();
        let mut nn = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut dp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared = block_size * 8; // f32 + i32 per thread
        self.launch_maybe_blob(
            "argmax_f32_batched",
            [batch_size as u32, 1, 1],
            [block_size, 1, 1],
            shared,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(dp); b.push_ptr(rp); b.push_i32(nn);
                b
            },
        )
    }

    /// GPU-side argmax: returns index of max value. Avoids downloading full logits.
    pub fn argmax_f32(&mut self, data: &GpuTensor, n: usize) -> HipResult<u32> {
        self.bind_thread()?;
        self.ensure_kernel("argmax_f32", kernels::ARGMAX_SRC, "argmax_f32")?;
        let func = &self.functions["argmax_f32"];

        let result_buf = self.hip.malloc(4)?; // single int
        self.hip.memset(&result_buf, 0, 4)?;

        let mut dp = data.buf.as_ptr();
        let mut rp = result_buf.as_ptr();
        let mut nn = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut dp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared = block_size * 8; // float + int per thread
        unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [block_size, 1, 1], shared, None, &mut params)?;
        }

        let mut result = [0i32];
        let result_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, 4)
        };
        self.hip.memcpy_dtoh(result_bytes, &result_buf)?;
        self.hip.free(result_buf)?;
        Ok(result[0] as u32)
    }


    /// GPU-side top-K + top-P sampling. Returns (token_id, new_rng_state).
    /// Eliminates 600KB logits download per token.
    pub fn sample_top_p(
        &mut self,
        logits: &GpuTensor,
        result_buf: &GpuTensor,
        repeat_buf: &GpuTensor,
        vocab_size: usize,
        temperature: f32,
        top_p: f32,
        rng_state: u32,
        repeat_window: usize,
        repeat_penalty: f32,
    ) -> HipResult<(u32, u32)> {
        // Back-compat shim: no presence/frequency penalties (byte-identical
        // to the pre-PF kernel, which had `if (repeat_penalty > 1.0f)`).
        self.sample_top_p_pf(
            logits, result_buf, repeat_buf, vocab_size, temperature, top_p,
            rng_state, repeat_window, repeat_penalty, 0.0, 0.0,
        )
    }

    /// Like [`sample_top_p`], plus OpenAI-style subtractive `presence_penalty`
    /// and `frequency_penalty` applied over the same `repeat_window`. Passing
    /// `0.0` for both is byte-identical to `sample_top_p`. These flat (non
    /// recency-weighted) penalties break block-level repetition loops the
    /// recency-weighted multiplicative repeat penalty cannot — provided the
    /// `repeat_buf` window is large enough to span a full loop period.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_top_p_pf(
        &mut self,
        logits: &GpuTensor,
        result_buf: &GpuTensor,
        repeat_buf: &GpuTensor,
        vocab_size: usize,
        temperature: f32,
        top_p: f32,
        rng_state: u32,
        repeat_window: usize,
        repeat_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
    ) -> HipResult<(u32, u32)> {
        self.bind_thread()?;
        self.ensure_kernel("sample_top_p", kernels::SAMPLE_TOP_P_SRC, "sample_top_p")?;
        let func = &self.functions["sample_top_p"];

        let mut logits_ptr = logits.buf.as_ptr();
        let mut result_ptr = result_buf.buf.as_ptr();
        let mut repeat_ptr = repeat_buf.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut temp = temperature;
        let mut tp = top_p;
        let mut rng = rng_state;
        let mut rw = repeat_window as i32;
        let mut rp = repeat_penalty;
        let mut pp = presence_penalty;
        let mut fp = frequency_penalty;

        let mut params: Vec<*mut std::ffi::c_void> = vec![
            &mut logits_ptr as *mut _ as *mut std::ffi::c_void,
            &mut result_ptr as *mut _ as *mut std::ffi::c_void,
            &mut repeat_ptr as *mut _ as *mut std::ffi::c_void,
            &mut vs as *mut _ as *mut std::ffi::c_void,
            &mut temp as *mut _ as *mut std::ffi::c_void,
            &mut tp as *mut _ as *mut std::ffi::c_void,
            &mut rng as *mut _ as *mut std::ffi::c_void,
            &mut rw as *mut _ as *mut std::ffi::c_void,
            &mut rp as *mut _ as *mut std::ffi::c_void,
            &mut pp as *mut _ as *mut std::ffi::c_void,
            &mut fp as *mut _ as *mut std::ffi::c_void,
        ];

        let block_size = 256u32;
        // topk_val[nthreads*20] + topk_idx[nthreads*20] = 256*20*4 + 256*20*4 = 40960 bytes
        let shared_mem = 256u32 * 20 * 4 * 2;

        unsafe {
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )?;
        }

        let mut out = [0u8; 8];
        self.hip.memcpy_dtoh(&mut out, &result_buf.buf)?;
        let token_id = u32::from_ne_bytes([out[0], out[1], out[2], out[3]]);
        let new_rng = u32::from_ne_bytes([out[4], out[5], out[6], out[7]]);
        Ok((token_id, new_rng))
    }

    /// Launch sampling kernel only (no readback). For use during graph capture.
    pub fn sample_top_p_launch(
        &mut self,
        logits: &GpuTensor,
        result_buf: &GpuTensor,
        repeat_buf: &GpuTensor,
        vocab_size: usize,
        temperature: f32,
        top_p: f32,
        rng_state: u32,
        repeat_window: usize,
        repeat_penalty: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("sample_top_p", kernels::SAMPLE_TOP_P_SRC, "sample_top_p")?;
        let func = &self.functions["sample_top_p"];

        let mut logits_ptr = logits.buf.as_ptr();
        let mut result_ptr = result_buf.buf.as_ptr();
        let mut repeat_ptr = repeat_buf.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut temp = temperature;
        let mut tp = top_p;
        let mut rng = rng_state;
        let mut rw = repeat_window as i32;
        let mut rp = repeat_penalty;
        // Graph-capture path does not expose presence/frequency penalties.
        let mut pp = 0.0f32;
        let mut fp = 0.0f32;

        let mut params: Vec<*mut std::ffi::c_void> = vec![
            &mut logits_ptr as *mut _ as *mut std::ffi::c_void,
            &mut result_ptr as *mut _ as *mut std::ffi::c_void,
            &mut repeat_ptr as *mut _ as *mut std::ffi::c_void,
            &mut vs as *mut _ as *mut std::ffi::c_void,
            &mut temp as *mut _ as *mut std::ffi::c_void,
            &mut tp as *mut _ as *mut std::ffi::c_void,
            &mut rng as *mut _ as *mut std::ffi::c_void,
            &mut rw as *mut _ as *mut std::ffi::c_void,
            &mut rp as *mut _ as *mut std::ffi::c_void,
            &mut pp as *mut _ as *mut std::ffi::c_void,
            &mut fp as *mut _ as *mut std::ffi::c_void,
        ];

        let block_size = 256u32;
        // topk_val[nthreads*20] + topk_idx[nthreads*20] = 256*20*4 + 256*20*4 = 40960 bytes
        let shared_mem = 256u32 * 20 * 4 * 2;

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


    /// Top-K=1024 extraction over a logits vector. Populates an 8 KB
    /// buffer with [1024 × u32 indices | 1024 × f32 values]. One
    /// device→host copy pulls the whole thing. The host then runs its
    /// existing top-20 min-tracking loop over the 1024 candidates.
    ///
    /// Previous version used 1 wave of 32 threads and measured at ~1.4 ms
    /// because the compiler couldn't pipeline loads through the branchy
    /// min-tracking path. Current version uses 256 threads (8 waves) on
    /// a single workgroup — roughly 10× faster.
    pub fn topk_logits_f32(
        &mut self,
        logits: &GpuTensor,
        topk_buf: &GpuTensor,   // DType::F32 shape [2048] = 8192 bytes
        vocab_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("topk_logits", kernels::TOPK_LOGITS_SRC, "topk_logits_f32")?;
        let func = &self.functions["topk_logits_f32"];
        let mut lp = logits.buf.as_ptr();
        let mut bp = topk_buf.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
        ];
        let bytes = vocab_size * 4 + 8192;
        let timer = crate::profile::begin_timer(&self.hip, "sampling", "topk_logits_f32", bytes);
        let result = unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Per-row top-K + log-sum-exp over `[B × vocab]` f32 logits.
    /// Writes `top_idx[B × K]` and `top_logp[B × K]` where `top_logp[r,k] =
    /// logit[r, top_idx[r,k]] - log_z[r]` with `log_z` = row-wise
    /// log-sum-exp. Replaces 20 ms of CPU sort + log_z per DDTree cycle.
    ///
    /// Constraints: K ≤ 8 (kernel-enforced). For larger K, extend MAX_K in
    /// the kernel source and the per-thread arrays.
    pub fn topk_logsumexp_batched_f32(
        &mut self,
        logits: &GpuTensor,   // [B × vocab] f32
        top_idx: &GpuTensor,  // [B × K] i32 (we use f32 tensor for storage — caller reinterprets)
        top_logp: &GpuTensor, // [B × K] f32
        vocab: usize,
        k: usize,
        b: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(k >= 1 && k <= 8, "topk_logsumexp_batched: K={} must be in [1,8]", k);
        self.ensure_kernel(
            "topk_logsumexp_batched",
            kernels::TOPK_LOGSUMEXP_BATCHED_SRC,
            "topk_logsumexp_batched_f32",
        )?;
        let func = &self.functions["topk_logsumexp_batched_f32"];
        let mut lp = logits.buf.as_ptr();
        let mut ti = top_idx.buf.as_ptr();
        let mut tl = top_logp.buf.as_ptr();
        let mut vs = vocab as i32;
        let mut kk = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut ti as *mut _ as *mut c_void,
            &mut tl as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];
        // LDS: (nth_warps=8 floats) + (nth × MAX_K × 2 floats). At nth=256,
        // MAX_K=8: 32 + 4096 = 4128 floats = 16,512 bytes. Fits in 64 KB LDS.
        const MAX_K: u32 = 8;
        let nth: u32 = 256;
        let lds = ((32 + nth * MAX_K * 2) * 4) as u32;
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [b as u32, 1, 1],
                [nth, 1, 1],
                lds,
                self.stream_ref(),
                &mut params,
            )
        };
        result
    }

    pub fn argmax_token_chain_f32(
        &mut self,
        data: &GpuTensor,
        argmax_out: &GpuTensor,
        token_chain: &GpuTensor,
        vocab_map: Option<&GpuTensor>,
        n: usize,
        dst_slot: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "argmax_token_chain",
            kernels::ARGMAX_TOKEN_CHAIN_SRC,
            "argmax_token_chain_f32",
        )?;

        let mut dp = data.buf.as_ptr();
        let mut ap = argmax_out.buf.as_ptr();
        let mut cp = token_chain.buf.as_ptr();
        let mut vp = vocab_map
            .map(|t| t.buf.as_ptr())
            .unwrap_or(std::ptr::null_mut::<c_void>());
        let mut nn = n as i32;
        let mut ds = dst_slot as i32;
        let mut use_map = i32::from(vocab_map.is_some());

        let mut params: Vec<*mut c_void> = vec![
            &mut dp as *mut _ as *mut c_void,
            &mut ap as *mut _ as *mut c_void,
            &mut cp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
            &mut ds as *mut _ as *mut c_void,
            &mut use_map as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared = block_size * 8; // f32 + i32 per thread
        self.launch_maybe_blob(
            "argmax_token_chain_f32",
            [1, 1, 1],
            [block_size, 1, 1],
            shared,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(dp);
                b.push_ptr(ap);
                b.push_ptr(cp);
                b.push_ptr(vp);
                b.push_i32(nn);
                b.push_i32(ds);
                b.push_i32(use_map);
                b
            },
        )
    }

    /// Device-side greedy accept prefix scan over verify argmaxes and MTP
    /// candidates. `result[0]` is accept_count; `result[1]` is the bonus
    /// token, or -1 if an accepted candidate was EOS and no bonus is present.
    pub fn greedy_accept_from_argmax_i32(
        &mut self,
        argmax_per_pos: &GpuTensor,
        candidates: &GpuTensor,
        result: &GpuTensor,
        drafts_generated: usize,
        eos_token_id: u32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "greedy_accept",
            kernels::GREEDY_ACCEPT_SRC,
            "greedy_accept_from_argmax_i32",
        )?;

        let mut ap = argmax_per_pos.buf.as_ptr();
        let mut cp = candidates.buf.as_ptr();
        let mut rp = result.buf.as_ptr();
        let mut dg = drafts_generated as i32;
        let mut eos = eos_token_id as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut cp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut dg as *mut _ as *mut c_void,
            &mut eos as *mut _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "greedy_accept_from_argmax_i32",
            [1, 1, 1],
            [1, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(cp);
                b.push_ptr(rp);
                b.push_i32(dg);
                b.push_i32(eos);
                b
            },
        )
    }
}
