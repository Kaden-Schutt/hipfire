// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sampling and reduction dispatch methods
//! (argmax, top-k, top-p, log-sum-exp).

use std::ffi::c_void;

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult};

/// Whether the multi-workgroup parallel sampler is enabled (default ON).
/// `HIPFIRE_SAMPLE_PARALLEL=0` forces the legacy single-block kernel (for
/// byte-exact A/B and as a fallback). Read once, cached.
fn sample_parallel_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_SAMPLE_PARALLEL")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Whether the tie-safe fast reducer is enabled (default ON). This keeps the
/// exact legacy parallel reducer available as a same-binary correctness and
/// performance control without falling all the way back to the single-block
/// sampler.
fn sample_fast_stable_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_SAMPLE_FAST")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// HIP source for the default parallel sampler module (`sample_top_p_parallel`,
/// TOP_K 20). Shared by runtime ensure_kernel and `precompile_qwen35` so the
/// cache hash is identical.
pub(crate) fn sample_top_p_parallel_src() -> String {
    kernels::SAMPLE_TOP_P_PARALLEL_SRC.replace("#define TOP_K 64", "#define TOP_K 20")
}

/// HIP source for the wide parallel sampler (`sample_top_p_parallel_w64`).
/// Renames the three entry points; leaves TOP_K at the on-disk default 64.
pub(crate) fn sample_top_p_parallel_w64_src() -> String {
    kernels::SAMPLE_TOP_P_PARALLEL_SRC
        .replace(
            "sample_apply_repeat_penalty",
            "sample_apply_repeat_penalty_w64",
        )
        .replace("sample_topk_partial", "sample_topk_partial_w64")
        .replace("sample_topk_finalize", "sample_topk_finalize_w64")
}

/// HIP source for a fast-stable parallel sampler module.
/// `top_k_width` is 21 (fast21) or 65 (fast65); `suffix` is the symbol suffix
/// (`fast21` / `fast65`). Must stay byte-identical to the runtime rewrite.
pub(crate) fn sample_top_p_parallel_fast_src(top_k_width: usize, suffix: &str) -> String {
    let top_k_define = format!(
        "#define TOP_K {}\n#define SAMPLE_FAST_STABLE 1",
        top_k_width
    );
    let fn_penalty = format!("sample_apply_repeat_penalty_{suffix}");
    let fn_partial = format!("sample_topk_partial_{suffix}");
    let fn_finalize = format!("sample_topk_finalize_{suffix}");
    kernels::SAMPLE_TOP_P_PARALLEL_SRC
        .replace("#define TOP_K 64", &top_k_define)
        .replace("sample_apply_repeat_penalty", &fn_penalty)
        .replace("sample_topk_partial", &fn_partial)
        .replace("sample_topk_finalize", &fn_finalize)
}

/// All exact parallel-sampler module identities used by `sample_top_p_pf`,
/// for admission into `precompile_qwen35`'s compile_batch.
pub(crate) fn sample_top_p_parallel_precompile_specs() -> [(&'static str, String); 4] {
    [
        ("sample_top_p_parallel", sample_top_p_parallel_src()),
        ("sample_top_p_parallel_w64", sample_top_p_parallel_w64_src()),
        (
            "sample_top_p_parallel_fast21",
            sample_top_p_parallel_fast_src(21, "fast21"),
        ),
        (
            "sample_top_p_parallel_fast65",
            sample_top_p_parallel_fast_src(65, "fast65"),
        ),
    ]
}

impl Gpu {
    /// Compute max softmax probability on GPU. Downloads 4 bytes instead of vocab×4.
    pub fn max_prob(
        &mut self,
        logits: &GpuTensor,
        result: &GpuTensor,
        vocab_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("max_prob", kernels::MAX_PROB_SRC, "max_prob")?;
        let func = &self.functions["max_prob"];
        let mut lp = logits.buf.as_ptr();
        let mut rp = result.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let shared = (block * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
                [block, 1, 1],
                shared,
                self.stream_ref(),
                &mut params,
            )
        }
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
                b.push_ptr(dp);
                b.push_ptr(rp);
                b.push_i32(nn);
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
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
                [block_size, 1, 1],
                shared,
                None,
                &mut params,
            )?;
        }

        let mut result = [0i32];
        let result_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, 4) };
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
            logits,
            result_buf,
            repeat_buf,
            vocab_size,
            temperature,
            top_p,
            rng_state,
            repeat_window,
            repeat_penalty,
            0.0,
            0.0,
            None,
            None,
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
        top_k: Option<u32>,
        min_p: Option<f32>,
    ) -> HipResult<(u32, u32)> {
        self.bind_thread()?;
        // Request-driven candidate caps. None preserves legacy behavior exactly:
        // top_k → 20 (== kernel TOP_K, no cut), min_p → 0.0 (disabled).
        let top_k_req = top_k.map(|k| k as i32).unwrap_or(20);
        let min_p_val = min_p.unwrap_or(0.0);
        // Multi-workgroup parallel sampler (default ON): splits the 150K-vocab
        // top-K scan across N blocks instead of the single-block kernel that
        // idles 95 of 96 CUs (~305 us/token on gfx1100 A3B decode). Byte-
        // identical token for distinct logits. Opt out: HIPFIRE_SAMPLE_PARALLEL=0.
        if sample_parallel_enabled() {
            return self.sample_top_p_parallel_impl(
                logits,
                result_buf,
                repeat_buf,
                vocab_size,
                temperature,
                top_p,
                rng_state,
                repeat_window,
                repeat_penalty,
                presence_penalty,
                frequency_penalty,
                top_k_req,
                min_p_val,
            );
        }
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
        let mut tk = top_k_req;
        let mut mp = min_p_val;

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
            &mut tk as *mut _ as *mut std::ffi::c_void,
            &mut mp as *mut _ as *mut std::ffi::c_void,
        ];

        // W7 P2b: gather TOP_K widened 20→64. LDS = nthreads*TOP_K*8; with
        // TOP_K=64 the block must be 128 threads (128*64*8 = 64 KiB, the RDNA
        // wave32 group-segment limit; 256 would request 128 KiB and fail).
        let block_size = 128u32;
        let shared_mem = block_size * 64 * 4 * 2;

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

    /// Multi-workgroup parallel implementation of [`sample_top_p_pf`]. Three
    /// stream-ordered launches: an in-place penalty prepass (only when a
    /// penalty is active), `sample_topk_partial` (vocab top-K split across
    /// `N_BLOCKS` workgroups → partials scratch), and `sample_topk_finalize`
    /// (merge partials → global top-K + the verbatim softmax/sort/top-p/RNG
    /// tail). Byte-identical token to the single-block kernel for distinct
    /// logits.
    #[allow(clippy::too_many_arguments)]
    fn sample_top_p_parallel_impl(
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
        top_k_req: i32,
        min_p_val: f32,
    ) -> HipResult<(u32, u32)> {
        const N_BLOCKS: u32 = 128;
        let any_penalty = repeat_penalty > 1.0 || presence_penalty > 0.0 || frequency_penalty > 0.0;
        // ARCHBLEED FIX (was d3472d9e): the gather budget is REQUEST-SELECTED,
        // not a global hardcode. W7 P2b widened TOP_K 20→64 (so minimax top_k=40
        // is honored) and dropped the block 256→128 to fit LDS — but did so
        // UNCONDITIONALLY, costing ~6.5% AR decode on every non-minimax model
        // (gfx1100 qwen3.6-27b). Here the common case (top_k<=20, incl. None)
        // uses a 20-wide / 256-block variant that is byte-identical to 43c3129c;
        // only top_k>20 compiles+uses the 64-wide / 128-block variant. LDS =
        // block*TOP_K*8 must stay <= 64 KiB: 256*20*8=40K and 128*64*8=64K both
        // fit; 256*64*8=128K would fail to launch — so width and block move
        // together. minimax keeps its full top_k=40 support via the wide path.
        let wide = top_k_req > 20;
        let top_k: usize = if wide { 64 } else { 20 };
        let block: u32 = if wide { 128 } else { 256 };
        // smem: topk_val[block*TOP_K] + topk_idx[block*TOP_K], 4 bytes each.
        let shared_mem = block * top_k as u32 * 4 * 2;

        // One templated source → two compiled variants. `self.functions` is
        // keyed by function name and skips reload once a name is present (see
        // compile_and_load_kernel), so the wide variant SUFFIXES its entry
        // points to coexist with the default without clobbering it.
        let (m, fn_penalty, fn_partial, fn_finalize) = if wide {
            (
                "sample_top_p_parallel_w64",
                "sample_apply_repeat_penalty_w64",
                "sample_topk_partial_w64",
                "sample_topk_finalize_w64",
            )
        } else {
            (
                "sample_top_p_parallel",
                "sample_apply_repeat_penalty",
                "sample_topk_partial",
                "sample_topk_finalize",
            )
        };
        // `ensure_kernel` caches compiled functions, but constructing/replacing
        // the full HIP source used to happen before that cache check on every
        // generated token. Only materialize source when a function is missing.
        if !self.functions.contains_key(fn_penalty)
            || !self.functions.contains_key(fn_partial)
            || !self.functions.contains_key(fn_finalize)
        {
            let src: String = if wide {
                sample_top_p_parallel_w64_src()
            } else {
                sample_top_p_parallel_src()
            };
            self.ensure_kernel(m, &src, fn_penalty)?;
            self.ensure_kernel(m, &src, fn_partial)?;
            self.ensure_kernel(m, &src, fn_finalize)?;
        }

        // Partials scratch: [N_BLOCKS*TOP_K] f32 vals then [N_BLOCKS*TOP_K] i32 idx.
        let n_cand = N_BLOCKS as usize * top_k;
        let val_bytes = n_cand * 4;
        let mut logits_ptr = logits.buf.as_ptr();
        let mut repeat_ptr = repeat_buf.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut rw = repeat_window as i32;
        let mut rp = repeat_penalty;
        let mut pp = presence_penalty;
        let mut fp = frequency_penalty;

        // 1) Penalty prepass (in-place on logits), only when active. It runs
        // before the fast attempt so an ambiguity fallback can reuse the
        // adjusted logits without applying the penalty twice.
        if any_penalty && repeat_window > 0 {
            let mut params: Vec<*mut c_void> = vec![
                &mut logits_ptr as *mut _ as *mut c_void,
                &mut repeat_ptr as *mut _ as *mut c_void,
                &mut vs as *mut _ as *mut c_void,
                &mut rw as *mut _ as *mut c_void,
                &mut rp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut fp as *mut _ as *mut c_void,
            ];
            let func = &self.functions[fn_penalty];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [1, 1, 1],
                    [block, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

        // Fast-stable reducer: TOP_K is request_cap+1 so the last slot is the
        // tie-boundary sentinel. Different internal order is accepted only when
        // ordering is provably unambiguous; token_id==u32::MAX falls through to
        // the exact reducer below. Penalty already ran exactly once for both.
        // width 21 covers top_k<=20; width 65 covers 21..=64 (vocab-guarded).
        if sample_fast_stable_enabled()
            && top_k_req > 0
            && top_k_req <= 64
            && vocab_size <= N_BLOCKS as usize * 256 * 16
        {
            if let Some(result) = self.sample_top_p_fast_stable_impl(
                logits,
                result_buf,
                vocab_size,
                temperature,
                top_p,
                rng_state,
                top_k_req,
                min_p_val,
            )? {
                return Ok(result);
            }
        }

        let partial_base = self
            .scratch
            .ensure_sample_partials(&self.hip, val_bytes * 2)?;
        let partial_val_ptr = partial_base;
        let partial_idx_ptr =
            unsafe { (partial_base as *mut u8).add(val_bytes) as *mut std::ffi::c_void };

        let mut result_ptr = result_buf.buf.as_ptr();
        let mut nb = N_BLOCKS as i32;
        let mut ncand = n_cand as i32;
        let mut temp = temperature;
        let mut tp = top_p;
        let mut rng = rng_state;
        let mut pval = partial_val_ptr;
        let mut pidx = partial_idx_ptr;
        let mut tk = top_k_req;
        let mut mp = min_p_val;

        // 2) Per-block partial top-K over vocab strips.
        {
            let mut params: Vec<*mut c_void> = vec![
                &mut logits_ptr as *mut _ as *mut c_void,
                &mut vs as *mut _ as *mut c_void,
                &mut nb as *mut _ as *mut c_void,
                &mut pval as *mut _ as *mut c_void,
                &mut pidx as *mut _ as *mut c_void,
            ];
            let func = &self.functions[fn_partial];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [N_BLOCKS, 1, 1],
                    [block, 1, 1],
                    shared_mem,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

        // 3) Merge partials → global top-K, then softmax/top-p/RNG sample.
        {
            let mut params: Vec<*mut c_void> = vec![
                &mut pval as *mut _ as *mut c_void,
                &mut pidx as *mut _ as *mut c_void,
                &mut ncand as *mut _ as *mut c_void,
                &mut result_ptr as *mut _ as *mut c_void,
                &mut temp as *mut _ as *mut c_void,
                &mut tp as *mut _ as *mut c_void,
                &mut rng as *mut _ as *mut c_void,
                &mut tk as *mut _ as *mut c_void,
                &mut mp as *mut _ as *mut c_void,
            ];
            let func = &self.functions[fn_finalize];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [1, 1, 1],
                    [block, 1, 1],
                    shared_mem,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

        let mut out = [0u8; 8];
        self.hip.memcpy_dtoh(&mut out, &result_buf.buf)?;
        let token_id = u32::from_ne_bytes([out[0], out[1], out[2], out[3]]);
        let new_rng = u32::from_ne_bytes([out[4], out[5], out[6], out[7]]);
        Ok((token_id, new_rng))
    }

    /// Fast-stable top-k+1 reducer. Selects width 21 for top_k_req<=20 and
    /// width 65 for 21..=64; the extra candidate is the tie-boundary sentinel.
    /// Any requested penalty is applied by the caller before entry. Returns
    /// `None` when the kernel detects a probability tie whose stable ordering
    /// could differ from the legacy reduction; the caller then runs legacy on
    /// the same adjusted logits.
    #[allow(clippy::too_many_arguments)]
    fn sample_top_p_fast_stable_impl(
        &mut self,
        logits: &GpuTensor,
        result_buf: &GpuTensor,
        vocab_size: usize,
        temperature: f32,
        top_p: f32,
        rng_state: u32,
        top_k_req: i32,
        min_p_val: f32,
    ) -> HipResult<Option<(u32, u32)>> {
        const N_BLOCKS: u32 = 128;
        const PARTIAL_BLOCK: u32 = 256;
        const FINALIZE_BLOCK: u32 = 128;
        // TOP_K = request_cap + 1 (boundary sentinel). Narrow path stays at 21
        // so top_k<=20 remains byte-identical; wide path uses 65 for 21..=64.
        let top_k_width: usize = if top_k_req <= 20 { 21 } else { 65 };
        let (module, _fn_penalty, fn_partial, fn_finalize) = if top_k_width == 21 {
            (
                "sample_top_p_parallel_fast21",
                "sample_apply_repeat_penalty_fast21",
                "sample_topk_partial_fast21",
                "sample_topk_finalize_fast21",
            )
        } else {
            (
                "sample_top_p_parallel_fast65",
                "sample_apply_repeat_penalty_fast65",
                "sample_topk_partial_fast65",
                "sample_topk_finalize_fast65",
            )
        };

        if !self.functions.contains_key(fn_partial) || !self.functions.contains_key(fn_finalize) {
            let suffix = if top_k_width == 21 {
                "fast21"
            } else {
                "fast65"
            };
            let src = sample_top_p_parallel_fast_src(top_k_width, suffix);
            self.ensure_kernel(module, &src, fn_partial)?;
            self.ensure_kernel(module, &src, fn_finalize)?;
        }

        let n_cand = N_BLOCKS as usize * top_k_width;
        let val_bytes = n_cand * 4;
        let partial_base = self
            .scratch
            .ensure_sample_partials(&self.hip, val_bytes * 2)?;
        let partial_val_ptr = partial_base;
        let partial_idx_ptr =
            unsafe { (partial_base as *mut u8).add(val_bytes) as *mut std::ffi::c_void };

        let mut logits_ptr = logits.buf.as_ptr();
        let mut result_ptr = result_buf.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut nb = N_BLOCKS as i32;
        let mut ncand = n_cand as i32;
        let mut temp = temperature;
        let mut tp = top_p;
        let mut rng = rng_state;
        let mut pval = partial_val_ptr;
        let mut pidx = partial_idx_ptr;
        let mut tk = top_k_req;
        let mut mp = min_p_val;

        {
            let mut params: Vec<*mut c_void> = vec![
                &mut logits_ptr as *mut _ as *mut c_void,
                &mut vs as *mut _ as *mut c_void,
                &mut nb as *mut _ as *mut c_void,
                &mut pval as *mut _ as *mut c_void,
                &mut pidx as *mut _ as *mut c_void,
            ];
            let func = &self.functions[fn_partial];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [N_BLOCKS, 1, 1],
                    [PARTIAL_BLOCK, 1, 1],
                    // 2048B LDS: PARTIAL_BLOCK * 4 * 2
                    PARTIAL_BLOCK * 4 * 2,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }
        {
            let mut params: Vec<*mut c_void> = vec![
                &mut pval as *mut _ as *mut c_void,
                &mut pidx as *mut _ as *mut c_void,
                &mut ncand as *mut _ as *mut c_void,
                &mut result_ptr as *mut _ as *mut c_void,
                &mut temp as *mut _ as *mut c_void,
                &mut tp as *mut _ as *mut c_void,
                &mut rng as *mut _ as *mut c_void,
                &mut tk as *mut _ as *mut c_void,
                &mut mp as *mut _ as *mut c_void,
            ];
            let func = &self.functions[fn_finalize];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [1, 1, 1],
                    [FINALIZE_BLOCK, 1, 1],
                    // 1536B LDS: FINALIZE_BLOCK * 4 * 3
                    FINALIZE_BLOCK * 4 * 3,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

        let mut out = [0u8; 8];
        self.hip.memcpy_dtoh(&mut out, &result_buf.buf)?;
        let token_id = u32::from_ne_bytes([out[0], out[1], out[2], out[3]]);
        let new_rng = u32::from_ne_bytes([out[4], out[5], out[6], out[7]]);
        if token_id == u32::MAX {
            Ok(None)
        } else {
            Ok(Some((token_id, new_rng)))
        }
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
        // Graph-capture path uses legacy candidate caps (no cut, min_p off).
        let mut tk = 20i32;
        let mut mp = 0.0f32;

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
            &mut tk as *mut _ as *mut std::ffi::c_void,
            &mut mp as *mut _ as *mut std::ffi::c_void,
        ];

        // W7 P2b: gather TOP_K widened 20→64. LDS = nthreads*TOP_K*8; with
        // TOP_K=64 the block must be 128 threads (128*64*8 = 64 KiB, the RDNA
        // wave32 group-segment limit; 256 would request 128 KiB and fail).
        let block_size = 128u32;
        let shared_mem = block_size * 64 * 4 * 2;

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
        topk_buf: &GpuTensor, // DType::F32 shape [2048] = 8192 bytes
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
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
                [256, 1, 1],
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

    /// Per-row top-K + log-sum-exp over `[B × vocab]` f32 logits.
    /// Writes `top_idx[B × K]` and `top_logp[B × K]` where `top_logp[r,k] =
    /// logit[r, top_idx[r,k]] - log_z[r]` with `log_z` = row-wise
    /// log-sum-exp. Replaces 20 ms of CPU sort + log_z per DDTree cycle.
    ///
    /// Constraints: K ≤ 16 (kernel-enforced). For larger K, extend MAX_K in
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
        self.launch_topk_batched_f32(logits, top_idx, top_logp, vocab, k, b, /*raw_values=*/ false)
    }

    /// Per-row top-K over `[B × vocab]` f32 logits, returning raw top logits
    /// (not log-probabilities). Same selection/order as
    /// [`Self::topk_logsumexp_batched_f32`]; skips the full-vocab logsumexp
    /// pass. Used by DFlash2 CandidateSelector.
    ///
    /// Constraints: K ≤ 16 (kernel-enforced).
    pub fn topk_values_batched_f32(
        &mut self,
        logits: &GpuTensor,    // [B × vocab] f32
        top_idx: &GpuTensor,   // [B × K] i32 (f32 tensor storage — caller reinterprets)
        top_values: &GpuTensor, // [B × K] f32 raw top logits
        vocab: usize,
        k: usize,
        b: usize,
    ) -> HipResult<()> {
        self.launch_topk_batched_f32(logits, top_idx, top_values, vocab, k, b, /*raw_values=*/ true)
    }

    fn launch_topk_batched_f32(
        &mut self,
        logits: &GpuTensor,
        top_idx: &GpuTensor,
        top_out: &GpuTensor,
        vocab: usize,
        k: usize,
        b: usize,
        raw_values: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            k >= 1 && k <= 16,
            "topk_batched: K={} must be in [1,16]",
            k
        );
        self.ensure_kernel(
            "topk_logsumexp_batched",
            kernels::TOPK_LOGSUMEXP_BATCHED_SRC,
            "topk_logsumexp_batched_f32",
        )?;
        let func = &self.functions["topk_logsumexp_batched_f32"];
        let mut lp = logits.buf.as_ptr();
        let mut ti = top_idx.buf.as_ptr();
        let mut to = top_out.buf.as_ptr();
        let mut vs = vocab as i32;
        let mut kk = k as i32;
        let mut raw = i32::from(raw_values);
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut ti as *mut _ as *mut c_void,
            &mut to as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
            &mut raw as *mut _ as *mut c_void,
        ];
        // LDS: (nth_warps=8 floats) + (nth × MAX_K × 2 floats). At nth=256,
        // MAX_K=16: 32 + 8192 = 8224 floats = 32,896 bytes. Fits in 64 KB LDS.
        const MAX_K: u32 = 16;
        let nth: u32 = 256;
        let lds = ((32 + nth * MAX_K * 2) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [b as u32, 1, 1],
                [nth, 1, 1],
                lds,
                self.stream_ref(),
                &mut params,
            )
        }
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

    /// Fused on-GPU sample+accept for DSpark (deepseek4) temp>0 spec-decode
    /// verify. Over the resident batched target logits `[n × vocab]` (produced
    /// by one batched lm-head weight read), samples every verify position on the
    /// device — replaying the single-block `sample_top_p` draw per row, threading
    /// the xorshift32 RNG across positions, and LAZILY early-exiting on the first
    /// token that mismatches its drafted successor (`draft[pos+1]`). Replaces the
    /// per-position `sample_top_p_pf` host loop (one 8-byte D2H + stream sync per
    /// position) with one launch and one `(n+1)×4`-byte D2H.
    ///
    /// `out_buf` must hold at least `n + 1` u32. Returns `(ids, new_rng)` where
    /// `ids` has length `n` (sampled tokens, `u32::MAX` after the first mismatch —
    /// the same vector the per-position path produced) and `new_rng` is the
    /// advanced RNG state to thread into the next window. `top_k = None` maps to
    /// the legacy nucleus (20), byte-identical to `sample_top_p_pf(.., None, ..)`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_accept_lazy_f32(
        &mut self,
        logits_batch: &GpuTensor, // [n × vocab] resident target logits
        draft: &GpuTensor,        // [n] u32 drafted block tokens
        out_buf: &GpuTensor,      // [n + 1] u32 scratch (ids + new rng)
        n: usize,
        vocab_size: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        rng_state: u32,
        cactus_delta: f32, // >0 → CACTUS acceptance boost (bench-only, deliberately lossy)
    ) -> HipResult<(Vec<u32>, u32)> {
        self.bind_thread()?;
        self.ensure_kernel(
            "dspark_sample_accept_lazy_f32",
            kernels::DSPARK_SAMPLE_ACCEPT_LAZY_SRC,
            "dspark_sample_accept_lazy_f32",
        )?;
        // None preserves legacy behavior: top_k → 20 (kernel TOP_K, no extra cut).
        let top_k_req = top_k.map(|k| k as i32).unwrap_or(20);

        let mut lp = logits_batch.buf.as_ptr();
        let mut dp = draft.buf.as_ptr();
        let mut op = out_buf.buf.as_ptr();
        let mut nn = n as i32;
        let mut vs = vocab_size as i32;
        let mut temp = temperature;
        let mut tp = top_p;
        let mut rng = rng_state;
        let mut tk = top_k_req;
        let mut cd = cactus_delta;

        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
            &mut temp as *mut _ as *mut c_void,
            &mut tp as *mut _ as *mut c_void,
            &mut rng as *mut _ as *mut c_void,
            &mut tk as *mut _ as *mut c_void,
            &mut cd as *mut _ as *mut c_void,
        ];

        // 1 block × 64 threads; LDS = 64 * 64 * 8 = 32 KiB. The single-block
        // sample_top_p uses 128 threads / 64 KiB, but gfx1151's usable dynamic
        // LDS is < 64 KiB (its parallel sampler tops out at 40 KiB), so 128 here
        // aborts with INVALID_ALLOCATION. 64 threads keeps us well under that and
        // is byte-identical: the top-K gather + tree reduction pick the same
        // global top-64 regardless of thread count (the RNG draw is thread-0
        // only). TOP_K stays 64 to honor top_k up to 40.
        let block_size = 64u32;
        let shared_mem = block_size * 64 * 4 * 2;
        self.launch_maybe_blob(
            "dspark_sample_accept_lazy_f32",
            [1, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(lp);
                b.push_ptr(dp);
                b.push_ptr(op);
                b.push_i32(nn);
                b.push_i32(vs);
                b.push_f32(temp);
                b.push_f32(tp);
                b.push_u32(rng);
                b.push_i32(tk);
                b.push_f32(cd);
                b
            },
        )?;

        // One D2H: (n+1) u32 — sampled ids [0..n) then the advanced rng at [n].
        let mut host = vec![0u32; n + 1];
        let bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, (n + 1) * 4) };
        self.hip.memcpy_dtoh(bytes, &out_buf.buf)?;
        let new_rng = host[n];
        host.truncate(n);
        Ok((host, new_rng))
    }

    /// Product-semantics sampling for independent continuous-batch lanes.
    ///
    /// Each row owns one workgroup and one RNG state. Repeat histories are
    /// lane-major with a fixed `repeat_stride`; `repeat_lengths` selects the
    /// chronological suffix populated for each lane. The compact readback is
    /// `[token, advanced_rng]` per row.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_rows_pf_f32(
        &mut self,
        logits_batch: &GpuTensor,
        repeat_tokens: &GpuTensor,
        repeat_lengths: &GpuTensor,
        rng_states: &GpuTensor,
        out_buf: &GpuTensor,
        batch_size: usize,
        vocab_size: usize,
        repeat_stride: usize,
        temperature: f32,
        top_p: f32,
        repeat_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
        top_k: Option<u32>,
        min_p: Option<f32>,
    ) -> HipResult<Vec<(u32, u32)>> {
        if batch_size == 0
            || repeat_stride == 0
            || logits_batch.numel() < batch_size * vocab_size
            || repeat_tokens.numel() < batch_size * repeat_stride
            || repeat_lengths.numel() < batch_size
            || rng_states.numel() < batch_size
            || out_buf.numel() < batch_size * 2
        {
            return Err(HipError::new(
                0,
                "sample_rows_pf_f32 buffers do not cover the requested batch shape",
            ));
        }
        self.bind_thread()?;
        self.ensure_kernel(
            "sample_rows_pf",
            kernels::SAMPLE_ROWS_PF_SRC,
            "sample_rows_pf_f32",
        )?;
        let func = &self.functions["sample_rows_pf_f32"];
        let mut logits_ptr = logits_batch.buf.as_ptr();
        let mut repeat_ptr = repeat_tokens.buf.as_ptr();
        let mut lengths_ptr = repeat_lengths.buf.as_ptr();
        let mut rng_ptr = rng_states.buf.as_ptr();
        let mut out_ptr = out_buf.buf.as_ptr();
        let mut batch = batch_size as i32;
        let mut vocab = vocab_size as i32;
        let mut stride = repeat_stride as i32;
        let mut temp = temperature;
        let mut nucleus = top_p;
        let mut repeat = repeat_penalty;
        let mut presence = presence_penalty;
        let mut frequency = frequency_penalty;
        let mut topk = top_k.map(|value| value as i32).unwrap_or(20);
        let mut minp = min_p.unwrap_or(0.0);
        let mut params: Vec<*mut c_void> = vec![
            &mut logits_ptr as *mut _ as *mut c_void,
            &mut repeat_ptr as *mut _ as *mut c_void,
            &mut lengths_ptr as *mut _ as *mut c_void,
            &mut rng_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut batch as *mut _ as *mut c_void,
            &mut vocab as *mut _ as *mut c_void,
            &mut stride as *mut _ as *mut c_void,
            &mut temp as *mut _ as *mut c_void,
            &mut nucleus as *mut _ as *mut c_void,
            &mut repeat as *mut _ as *mut c_void,
            &mut presence as *mut _ as *mut c_void,
            &mut frequency as *mut _ as *mut c_void,
            &mut topk as *mut _ as *mut c_void,
            &mut minp as *mut _ as *mut c_void,
        ];
        const BLOCK: u32 = 64;
        const TOP_K: u32 = 64;
        unsafe {
            self.hip.launch_kernel(
                func,
                [batch_size as u32, 1, 1],
                [BLOCK, 1, 1],
                BLOCK * TOP_K * 8,
                self.stream_ref(),
                &mut params,
            )?;
        }
        let mut words = vec![0u32; batch_size * 2];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, words.len() * 4)
        };
        self.hip.memcpy_dtoh(bytes, &out_buf.buf)?;
        Ok(words
            .chunks_exact(2)
            .map(|pair| (pair[0], pair[1]))
            .collect())
    }
    /// Per-row Gumbel-top-k SWOR sampler: draws `k` tokens WITHOUT replacement
    /// from `softmax(logits/temp)` per row of `[batch × vocab]`, returning the
    /// draw-ordered token ids (`top_idx`) and their true log-probs (`top_logp`),
    /// both `[batch × k]`. Keeps the draft logits device-resident — only B×k come
    /// back, vs the prior [B × vocab] D2H for host Gumbel sampling.
    #[allow(clippy::too_many_arguments)]
    pub fn ddtree_gumbel_topk_batched_f32(
        &mut self,
        logits: &GpuTensor,   // [batch × vocab]
        top_idx: &GpuTensor,  // [batch × k] i32
        top_logp: &GpuTensor, // [batch × k] f32
        vocab: usize,
        k: usize,
        batch: usize,
        temp: f32,
        seed: u64,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!((1..=8).contains(&k), "gumbel_topk: k={k} must be in [1,8]");
        self.ensure_kernel(
            "ddtree_gumbel_topk_batched",
            kernels::DDTREE_GUMBEL_TOPK_BATCHED_SRC,
            "ddtree_gumbel_topk_batched_f32",
        )?;
        let func = &self.functions["ddtree_gumbel_topk_batched_f32"];
        let mut lp = logits.buf.as_ptr();
        let mut ti = top_idx.buf.as_ptr();
        let mut tl = top_logp.buf.as_ptr();
        let mut vs = vocab as i32;
        let mut kk = k as i32;
        let mut tp = temp;
        let mut sd = (seed | 1) as u32;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut ti as *mut _ as *mut c_void,
            &mut tl as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
            &mut tp as *mut _ as *mut c_void,
            &mut sd as *mut _ as *mut c_void,
        ];
        const MAX_K: u32 = 8;
        let nth: u32 = 256;
        let lds = ((32 + nth * MAX_K * 2) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [batch as u32, 1, 1],
                [nth, 1, 1],
                lds,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Fused on-device SWOR tree-verify walk. One workgroup runs the whole
    /// sequential descent; each per-slot vocab sweep (target softmax, draft
    /// softmax, recursive `relu(p−q)` residual, renorm, categorical draw) is
    /// block-parallel. Replaces the host O(vocab·k)/slot loop and the q D2H.
    /// `out` (i32 `[2 + num_pos]`): `out[0]`=accept_len, `out[1]`=bonus token,
    /// `out[2+i]`=accepted child node index.
    #[allow(clippy::too_many_arguments)]
    pub fn ddtree_swor_walk_f32(
        &mut self,
        target_logits: &GpuTensor, // [n_slots * vocab]
        draft_logits: &GpuTensor,  // [num_pos * vocab]
        pos_cands: &GpuTensor,     // [num_pos * k] i32
        slot_depth: &GpuTensor,    // [n_slots] i32
        child_of_cand: &GpuTensor, // [n_slots * k] i32
        p_res: &GpuTensor,         // scratch [vocab]
        q_pos: &GpuTensor,         // scratch [vocab]
        out: &GpuTensor,           // [2 + num_pos] i32
        temp: f32,
        k: usize,
        vocab: usize,
        n_slots: usize,
        num_pos: usize,
        seed: u64,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!((1..=8).contains(&k), "swor_walk: k={k} must be in [1,8]");
        self.ensure_kernel(
            "ddtree_swor_walk",
            kernels::DDTREE_SWOR_WALK_SRC,
            "ddtree_swor_walk_f32",
        )?;
        let mut tl = target_logits.buf.as_ptr();
        let mut dl = draft_logits.buf.as_ptr();
        let mut pc = pos_cands.buf.as_ptr();
        let mut sd = slot_depth.buf.as_ptr();
        let mut cc = child_of_cand.buf.as_ptr();
        let mut pr = p_res.buf.as_ptr();
        let mut qp = q_pos.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut tp = temp;
        let mut kk = k as i32;
        let mut vs = vocab as i32;
        let mut ns = n_slots as i32;
        let mut np = num_pos as i32;
        let mut sd_seed = (seed | 1) as u32;
        let mut params: Vec<*mut c_void> = vec![
            &mut tl as *mut _ as *mut c_void,
            &mut dl as *mut _ as *mut c_void,
            &mut pc as *mut _ as *mut c_void,
            &mut sd as *mut _ as *mut c_void,
            &mut cc as *mut _ as *mut c_void,
            &mut pr as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tp as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
            &mut ns as *mut _ as *mut c_void,
            &mut np as *mut _ as *mut c_void,
            &mut sd_seed as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "ddtree_swor_walk_f32",
            [1, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tl);
                b.push_ptr(dl);
                b.push_ptr(pc);
                b.push_ptr(sd);
                b.push_ptr(cc);
                b.push_ptr(pr);
                b.push_ptr(qp);
                b.push_ptr(op);
                b.push_f32(tp);
                b.push_i32(kk);
                b.push_i32(vs);
                b.push_i32(ns);
                b.push_i32(np);
                b.push_u32(sd_seed);
                b
            },
        )
    }

    /// Stage 3a: on-GPU ddtree attention-mask builder.
    ///
    /// Reads `parent_indices[big_n]` (i32, device-resident) and fills
    /// `attn_bias[big_n * big_n]` (f32, row-major). Thread `i` walks the
    /// parent chain from `i` up to the root (-1 sentinel), setting 0.0 for
    /// each visited ancestor and -INF everywhere else. Exactly mirrors the
    /// host `visibility` bottom-up pass + row-major flatten in ddtree.rs.
    ///
    /// Grid: [big_n, 1, 1]. Block: [big_n, 1, 1]. At big_n ≤ 61 this is
    /// 61 threads total — occupancy is trivial; the kernel runs < 1 µs.
    pub fn ddtree_build_attn_mask_f32(
        &mut self,
        parent_indices: &GpuTensor, // [big_n] i32 (stored as Raw, 4*big_n bytes)
        attn_bias: &GpuTensor,      // [big_n * big_n] f32
        big_n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // The kernel launches blockDim.x = big_n (one thread per tree row); AMD
        // hardware caps blockDim.x at 1024. In practice big_n = 1 + max_budget ≤ 61,
        // so 1024 is already generous — a larger value would otherwise fail the
        // launch with an opaque invalid-configuration error instead of this assert.
        assert!(
            big_n >= 1 && big_n <= 1024,
            "ddtree_build_attn_mask: big_n={big_n} exceeds the 1024 blockDim.x cap (big_n = 1 + max_budget, normally ≤ 61)"
        );
        self.ensure_kernel(
            "ddtree_build_attn_mask",
            kernels::DDTREE_BUILD_ATTN_MASK_SRC,
            "ddtree_build_attn_mask_f32",
        )?;
        let func = &self.functions["ddtree_build_attn_mask_f32"];
        let mut pi = parent_indices.buf.as_ptr();
        let mut ab = attn_bias.buf.as_ptr();
        let mut nn = big_n as i32;
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            &mut pi as *mut _ as *mut std::ffi::c_void,
            &mut ab as *mut _ as *mut std::ffi::c_void,
            &mut nn as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [big_n as u32, 1, 1],
                [big_n as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// C8 Kernel 0: batched categorical sampler over already-softmax'd probs.
    ///
    /// For each of `batch` rows in `probs[batch * vocab]`, applies the
    /// top-p truncation `(p >= tau_cut[r]) ? p / z[r] : 0`, draws one
    /// categorical sample (LCG seeded per-row from `seed ^ row | 1`), and
    /// writes the sampled token id and its effective probability.
    ///
    /// D2H after this call: `batch * 8 bytes` (token + prob per row),
    /// replacing the `batch * vocab * 4` download that the FAST_SAMPLE path
    /// previously required.
    ///
    /// `probs` is left unmodified — it is also consumed by
    /// `chain_accept_spec_f32` as the draft prob buffer.
    ///
    /// `tau_cut` and `z` come from `softmax_temp_topp_batched_into_f32`.
    /// Pass zero-filled buffers (tau=0, z=1) when top-p is disabled.
    #[allow(clippy::too_many_arguments)]
    pub fn batched_categorical_sample_f32(
        &mut self,
        probs: &GpuTensor,      // [batch * vocab] f32 — softmax output
        tau_cut: &GpuTensor,    // [batch] f32 — top-p threshold per row
        z: &GpuTensor,          // [batch] f32 — kept mass per row
        out_tokens: &GpuTensor, // [batch] i32 — sampled token ids
        out_probs: &GpuTensor,  // [batch] f32 — prob at sampled token
        vocab: usize,
        batch: usize,
        seed: u32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "batched_categorical_sample",
            kernels::BATCHED_CATEGORICAL_SAMPLE_SRC,
            "batched_categorical_sample_f32",
        )?;

        let pp = probs.buf.as_ptr();
        let tp = tau_cut.buf.as_ptr();
        let zp = z.buf.as_ptr();
        let vs = vocab as i32;
        let mut sd = seed;
        let ot = out_tokens.buf.as_ptr();
        let op = out_probs.buf.as_ptr();

        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &zp as *const _ as *mut c_void,
            &vs as *const _ as *mut c_void,
            &mut sd as *mut _ as *mut c_void,
            &ot as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
        ];

        // One block per row; 256 threads per block.
        self.launch_maybe_blob(
            "batched_categorical_sample_f32",
            [batch as u32, 1, 1],
            [256, 1, 1],
            256 * 4 * 2, // s_red[256] + s_total_mass + s_pick + s_pick_prob ≈ 2 KB
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(tp);
                b.push_ptr(zp);
                b.push_i32(vs);
                b.push_u32(sd);
                b.push_ptr(ot);
                b.push_ptr(op);
                b
            },
        )
    }

    /// C8 Kernel 1: on-GPU chain rejection-sampling accept loop.
    ///
    /// Runs the entire spec-decode accept chain (Chen & Leviathan 2023,
    /// Algorithm 1) on-device over `b` speculated positions.  Replaces the
    /// host loop in `speculative.rs` that required two ~9 MB D2H transfers.
    ///
    /// `tgt_probs` must have `(b + 1) * vocab` elements: rows 0..b are the
    /// target probs for the drafted positions; row `b` is used for the bonus
    /// draw when all b positions are accepted.
    ///
    /// `dft_probs` must have `b * vocab` elements (draft side only).
    ///
    /// Returns the 16-byte output buffer contents as `[accept_len, bonus_token,
    /// rejected_at, new_rng_state]` (all i32/u32 words).  The caller reads the
    /// first three as i32 and the last as u32 for RNG bookkeeping.
    ///
    /// `cactus_delta = 0.0` disables the CACTUS boost (plain rejection sampling).
    #[allow(clippy::too_many_arguments)]
    pub fn chain_accept_spec_f32(
        &mut self,
        tgt_probs: &GpuTensor,        // [(b+1) * vocab] f32
        dft_probs: &GpuTensor,        // [b * vocab] f32
        draft_tokens: &GpuTensor,     // [b] i32
        draft_p_at_token: &GpuTensor, // [b] f32
        tau_t: &GpuTensor,            // [(b+1)] f32 — target topp tau per row
        z_t: &GpuTensor,              // [(b+1)] f32 — target topp Z per row
        tau_d: &GpuTensor,            // [b] f32 — draft topp tau per row
        z_d: &GpuTensor,              // [b] f32 — draft topp Z per row
        out: &GpuTensor,              // [4] i32 output buffer
        b: usize,
        vocab: usize,
        rng_seed: u32,
        cactus_delta: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "chain_accept_spec",
            kernels::CHAIN_ACCEPT_SPEC_SRC,
            "chain_accept_spec_f32",
        )?;

        let tgt_p = tgt_probs.buf.as_ptr();
        let dft_p = dft_probs.buf.as_ptr();
        let dtok = draft_tokens.buf.as_ptr();
        let dpat = draft_p_at_token.buf.as_ptr();
        let tt = tau_t.buf.as_ptr();
        let zt = z_t.buf.as_ptr();
        let td = tau_d.buf.as_ptr();
        let zd = z_d.buf.as_ptr();
        let outp = out.buf.as_ptr();
        let bv = b as i32;
        let vs = vocab as i32;
        let mut sd = rng_seed;
        let mut cd = cactus_delta;

        let mut params: Vec<*mut c_void> = vec![
            &tgt_p as *const _ as *mut c_void,
            &dft_p as *const _ as *mut c_void,
            &dtok as *const _ as *mut c_void,
            &dpat as *const _ as *mut c_void,
            &tt as *const _ as *mut c_void,
            &zt as *const _ as *mut c_void,
            &td as *const _ as *mut c_void,
            &zd as *const _ as *mut c_void,
            &bv as *const _ as *mut c_void,
            &vs as *const _ as *mut c_void,
            &mut sd as *mut _ as *mut c_void,
            &mut cd as *mut _ as *mut c_void,
            &outp as *const _ as *mut c_void,
        ];

        // Single block of 256 threads — the accept chain is sequential.
        self.launch_maybe_blob(
            "chain_accept_spec_f32",
            [1, 1, 1],
            [256, 1, 1],
            256 * 4 + 32, // s_red[256] + small shared scalars ≈ 1056 bytes
            &mut params,
            || {
                let mut bl = hip_bridge::KernargBlob::new();
                bl.push_ptr(tgt_p);
                bl.push_ptr(dft_p);
                bl.push_ptr(dtok);
                bl.push_ptr(dpat);
                bl.push_ptr(tt);
                bl.push_ptr(zt);
                bl.push_ptr(td);
                bl.push_ptr(zd);
                bl.push_i32(bv);
                bl.push_i32(vs);
                bl.push_u32(sd);
                bl.push_f32(cd);
                bl.push_ptr(outp);
                bl
            },
        )
    }

    /// Sample one token per slot from `[n_slots x vocab]` logits.
    ///
    /// Takes the existing `argmax_f32_batched` fast path when every slot is
    /// greedy; otherwise samples each slot with its own parameters. Per-slot
    /// dispatch is correct but not optimal — a fused kernel is a later
    /// optimisation, and SP2 is explicitly components-not-performance.
    pub fn sample_per_slot(
        &mut self,
        logits: &GpuTensor,
        params: &[SlotSampleParams],
        n_slots: usize,
        vocab: usize,
        out_tokens: &GpuTensor,
    ) -> HipResult<()> {
        assert_eq!(
            params.len(),
            n_slots,
            "sample_per_slot: one SlotSampleParams per slot required"
        );
        if all_greedy(params) {
            // NOTE: argmax_f32_batched's real signature is
            // (data, result, n=reduction_dim, batch_size), i.e. vocab comes
            // before n_slots — the brief's illustrative call had these
            // swapped; confirmed against call sites in
            // crates/hipfire-arch-deepseek4/src/forward.rs and others.
            return self.argmax_f32_batched(logits, out_tokens, vocab, n_slots);
        }
        for (i, p) in params.iter().enumerate() {
            self.sample_slot_row(logits, i, vocab, p, out_tokens)?;
        }
        Ok(())
    }

    /// Sample a single slot's row `i` from `[n_slots x vocab]` logits, writing
    /// the resulting token id into `out_tokens[i]`.
    ///
    /// A greedy row (`temperature == 0.0`) within an otherwise-mixed batch
    /// takes the exact single-row argmax rather than routing through
    /// `sample_top_p_pf` with temperature 0 — this keeps a greedy slot's
    /// result identical to what it would get on the all-greedy fast path,
    /// regardless of which other slots in the batch are sampling.
    fn sample_slot_row(
        &mut self,
        logits: &GpuTensor,
        i: usize,
        vocab: usize,
        p: &SlotSampleParams,
        out_tokens: &GpuTensor,
    ) -> HipResult<()> {
        let row = logits.sub_offset(i * vocab, vocab);
        let out_row = out_tokens.sub_offset(i, 1);

        let token_id: u32 = if p.temperature == 0.0 {
            self.argmax_f32(&row, vocab)?
        } else {
            // Existing sample_top_p kernel, dispatched per row. `sample_top_p`
            // itself is a fixed-top_k=20 shim over `sample_top_p_pf`; calling
            // `_pf` directly lets each slot's own top_k through rather than
            // silently discarding it, without introducing a new kernel.
            let result_buf = self.alloc_tensor(&[2], DType::F32)?;
            let repeat_buf = self.alloc_tensor(&[1], DType::F32)?;
            let top_k = if p.top_k > 0 {
                Some(p.top_k as u32)
            } else {
                None
            };
            let sample_result = self.sample_top_p_pf(
                &row,
                &result_buf,
                &repeat_buf,
                vocab,
                p.temperature,
                p.top_p,
                p.seed,
                0,   // repeat_window: no cross-slot repetition history here
                1.0, // repeat_penalty: 1.0 == disabled (kernel checks `> 1.0`)
                0.0, // presence_penalty: disabled
                0.0, // frequency_penalty: disabled
                top_k,
                None, // min_p: disabled
            );
            self.free_tensor(result_buf)?;
            self.free_tensor(repeat_buf)?;
            sample_result?.0
        };

        self.hip.memcpy_htod(&out_row.buf, &token_id.to_ne_bytes())
    }
}

/// Per-slot sampling parameters, uploaded as a table like `KvSlotDesc`.
///
/// A single scalar temperature across a batch is wrong as soon as two agents
/// differ, and a uniform-parameter test cannot see that bug — hence a table
/// rather than scalars.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotSampleParams {
    /// 0.0 means greedy/argmax for this slot.
    pub temperature: f32,
    pub top_p: f32,
    /// 0 disables top-k for this slot.
    pub top_k: i32,
    pub seed: u32,
}

/// True when every slot is greedy, so the batch can take the argmax fast path.
/// One sampling slot disables it for the whole batch.
pub fn all_greedy(params: &[SlotSampleParams]) -> bool {
    params.iter().all(|p| p.temperature == 0.0)
}

#[cfg(test)]
mod slot_sample_tests {
    use super::*;

    #[test]
    fn params_struct_is_16_bytes_repr_c() {
        // Uploaded straight to the GPU as a table, like KvSlotDesc.
        assert_eq!(std::mem::size_of::<SlotSampleParams>(), 16);
        assert_eq!(std::mem::align_of::<SlotSampleParams>(), 4);
    }

    #[test]
    fn all_greedy_is_detectable_as_a_fast_path() {
        let greedy = vec![
            SlotSampleParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: 1,
            },
            SlotSampleParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: 2,
            },
        ];
        assert!(all_greedy(&greedy));
        let mixed = vec![
            SlotSampleParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: 1,
            },
            SlotSampleParams {
                temperature: 0.7,
                top_p: 0.95,
                top_k: 20,
                seed: 2,
            },
        ];
        assert!(
            !all_greedy(&mixed),
            "one sampling slot must disable the greedy fast path"
        );
    }
}
