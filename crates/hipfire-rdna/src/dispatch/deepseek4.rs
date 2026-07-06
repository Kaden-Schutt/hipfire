// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek V4 Flash cluster: hyper-connections, NSA-style indexer/compressor, head-compute, hash routing, sqrt/softplus, int4 nibble-expand. Pure move (Phase 1 M7).

use super::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// Softplus activation, in-place.
    #[cfg(feature = "deltanet")]
    pub fn softplus_f32(&mut self, x: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("softplus", kernels::SOFTPLUS_SRC, "softplus_f32")?;
        let func = &self.functions["softplus_f32"];
        let mut xp = x.buf.as_ptr();
        let mut n = x.numel() as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 compressor batched aligned compress events. Per-event
    /// inputs come from `(prev_kv, prev_score)` for event 0 and from
    /// `(kv_batch, score_batch)` for events 1..N-1.
    /// Writes N_events × head_dim floats into `kv_cache_out` (caller
    /// supplies the slot-offset pointer).
    #[allow(clippy::too_many_arguments)]
    pub fn compressor_compress_aligned_batched_f32(
        &mut self,
        prev_kv: &GpuTensor,
        prev_score: &GpuTensor,
        kv_batch: &GpuTensor,
        score_batch: &GpuTensor,
        kv_cache_out: &GpuTensor,
        r: i32,
        head_dim: i32,
        n_events: i32,
        overlap: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "compressor_compress_aligned_batched_f32",
            kernels::COMPRESSOR_COMPRESS_ALIGNED_BATCHED_SRC,
            "compressor_compress_aligned_batched_f32",
        )?;
        let func = &self.functions["compressor_compress_aligned_batched_f32"];
        let pk = prev_kv.buf.as_ptr();
        let ps = prev_score.buf.as_ptr();
        let kb = kv_batch.buf.as_ptr();
        let sb = score_batch.buf.as_ptr();
        let yo = kv_cache_out.buf.as_ptr();
        let mut rr = r;
        let mut hd = head_dim;
        let mut ne = n_events;
        let mut ov = overlap;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &pk as *const _ as *mut c_void,
            &ps as *const _ as *mut c_void,
            &kb as *const _ as *mut c_void,
            &sb as *const _ as *mut c_void,
            &yo as *const _ as *mut c_void,
            &mut rr as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut ov as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let grid_x = ((head_dim + 255) / 256) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, n_events as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 compressor per-slot APE add over a batched score buffer.
    /// In-place: `score_batch[b, d] += ape[(start_pos + b) % ratio, d]`.
    /// Mirrors the per-position add in `compressor_forward_impl` so the
    /// batched-prefill compress path produces kv_cache entries with the
    /// same APE-applied scores as the sequential per-position path.
    #[allow(clippy::too_many_arguments)]
    pub fn compressor_add_ape_batched_f32(
        &mut self,
        score_batch: &GpuTensor, // [B, proj_dim] F32, in-place
        ape: &GpuTensor,         // [ratio, proj_dim] F32
        batch_size: i32,
        proj_dim: i32,
        ratio: i32,
        start_pos: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "compressor_add_ape_batched",
            kernels::COMPRESSOR_ADD_APE_BATCHED_SRC,
            "compressor_add_ape_batched_f32",
        )?;
        let func = &self.functions["compressor_add_ape_batched_f32"];
        let sb = score_batch.buf.as_ptr();
        let ap = ape.buf.as_ptr();
        let mut bs = batch_size;
        let mut pd = proj_dim;
        let mut rr = ratio;
        let mut sp = start_pos;
        let mut params: Vec<*mut c_void> = vec![
            &sb as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut pd as *mut _ as *mut c_void,
            &mut rr as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
        ];
        let grid_x = ((proj_dim + 255) / 256) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_size as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 Compressor overlap-transform concat (overlap=true / ratio=4).
    /// Reads [2*ratio, 2*head_dim] kv_state and writes [2*ratio, head_dim]
    /// dst by taking first half-cols for old window rows and second
    /// half-cols for current window rows.
    pub fn compressor_overlap_concat_f32(
        &mut self,
        src: &GpuTensor, // [2*ratio, 2*head_dim] F32
        dst: &GpuTensor, // [2*ratio, head_dim] F32
        ratio: i32,
        head_dim: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "compressor_overlap_concat",
            kernels::COMPRESSOR_OVERLAP_CONCAT_SRC,
            "compressor_overlap_concat_f32",
        )?;
        let func = &self.functions["compressor_overlap_concat_f32"];
        let sp = src.buf.as_ptr();
        let dp = dst.buf.as_ptr();
        let mut rv = ratio;
        let mut hd = head_dim;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &mut rv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [(2 * ratio) as u32, 1, 1],
                [head_dim as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 compressor batched ring-buffer write. Single launch scatters
    /// B positions into `kv_state[slot]` / `score_state[slot]` where
    /// `slot = (slot_base + b) % R + (overlap ? R : 0)`.
    #[allow(clippy::too_many_arguments)]
    pub fn compressor_ring_write_batched_f32(
        &mut self,
        kv_batch: &GpuTensor,
        score_batch: &GpuTensor,
        kv_state: &GpuTensor,
        score_state: &GpuTensor,
        batch_size: i32,
        proj_dim: i32,
        r: i32,
        slot_base: i32,
        overlap: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "compressor_ring_write_batched_f32",
            kernels::COMPRESSOR_RING_WRITE_BATCHED_SRC,
            "compressor_ring_write_batched_f32",
        )?;
        let func = &self.functions["compressor_ring_write_batched_f32"];
        let kb = kv_batch.buf.as_ptr();
        let sb = score_batch.buf.as_ptr();
        let ks = kv_state.buf.as_ptr();
        let ss = score_state.buf.as_ptr();
        let mut bsv = batch_size;
        let mut pd = proj_dim;
        let mut rr = r;
        let mut sbase = slot_base;
        let mut ov = overlap;
        let mut params: Vec<*mut c_void> = vec![
            &kb as *const _ as *mut c_void,
            &sb as *const _ as *mut c_void,
            &ks as *const _ as *mut c_void,
            &ss as *const _ as *mut c_void,
            &mut bsv as *mut _ as *mut c_void,
            &mut pd as *mut _ as *mut c_void,
            &mut rr as *mut _ as *mut c_void,
            &mut sbase as *mut _ as *mut c_void,
            &mut ov as *mut _ as *mut c_void,
        ];
        let grid_x = ((proj_dim + 255) / 256) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_size as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn compressor_softmax_pool_f32(
        &mut self,
        kv_state: &GpuTensor,    // [T, head_dim] F32
        score_state: &GpuTensor, // [T, head_dim] F32
        output: &GpuTensor,      // [head_dim] F32
        t: i32,
        head_dim: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "compressor_softmax_pool",
            kernels::COMPRESSOR_SOFTMAX_POOL_SRC,
            "compressor_softmax_pool_f32",
        )?;
        let func = &self.functions["compressor_softmax_pool_f32"];
        let kp = kv_state.buf.as_ptr();
        let sp = score_state.buf.as_ptr();
        let op = output.buf.as_ptr();
        let mut tv = t;
        let mut hd = head_dim;
        let mut params: Vec<*mut c_void> = vec![
            &kp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut tv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((head_dim as u32) + block - 1) / block;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 Compressor softmax-weighted pool. Compresses `T` window
    /// positions of (kv_state, score_state) into one `head_dim` output:
    ///   output[d] = sum_t softmax_t(score_state[:, d])[t] * kv_state[t, d]
    /// HIP-graphs-safe twin of `compressor_softmax_pool_f32`: reads the
    /// destination slot index from `slot_buf` (sentinel: -1 → no-op).
    /// Writes to `kv_cache + slot * head_dim`. Captured graphs include
    /// the commit kernel at every replay; the host sets slot to -1 on
    /// non-commit positions so the kernel is a no-op there.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn compressor_softmax_pool_f32_buf(
        &mut self,
        kv_state: &GpuTensor,
        score_state: &GpuTensor,
        kv_cache: &GpuTensor, // base ptr [max_slots, head_dim]
        slot_buf: &GpuTensor,
        t: i32,
        head_dim: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "compressor_softmax_pool_f32_buf",
            kernels::COMPRESSOR_SOFTMAX_POOL_BUF_SRC,
            "compressor_softmax_pool_f32_buf",
        )?;
        let kp = kv_state.buf.as_ptr();
        let sp = score_state.buf.as_ptr();
        let cp = kv_cache.buf.as_ptr();
        let sb = slot_buf.buf.as_ptr();
        let tv = t;
        let hd = head_dim;
        let block = 256u32;
        let grid = ((head_dim as u32) + block - 1) / block;
        self.launch_kernargs(
            "compressor_softmax_pool_f32_buf",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &kernargs![ptr kp, ptr sp, ptr cp, ptr sb, i32 tv, i32 hd],
        )
    }
    /// Bulk F32→F16 conversion. dst must hold at least `n` F16s.
    ///
    /// Registered as `deepseek4_convert_f32_to_f16` to avoid collision with the
    /// embedded `convert_f32_to_f16` helper in master's
    /// `GEMM_HFQ4G256_RESIDUAL_FP16_SRC` (different ABI: block=256, int n).
    /// First-call-wins kernel registration would otherwise launch with one
    /// ABI against the other's binary, corrupting the FP16 scratch and
    /// crashing the next launch (`hipErrorIllegalAddress`).
    pub fn deepseek4_convert_f32_to_f16(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        n: i64,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_convert_f32_to_f16",
            kernels::V4F_CONVERT_F32_TO_F16_SRC,
            "deepseek4_convert_f32_to_f16",
        )?;
        let func = &self.functions["deepseek4_convert_f32_to_f16"];
        let sp = src.buf.as_ptr();
        let dp = dst.buf.as_ptr();
        let mut nn = n;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];
        let n_wgs = ((n + 127) / 128) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_wgs, 1, 1],
                [128, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Reference kernel layer: expand packed int4 weights `wp_i4` [M,K/2] → int8
    /// `w8_i8` [M,K] (sign-extended), the W4A8 input to the iu8 WMMA core.
    pub fn nibble_expand_int4_to_int8(
        &mut self,
        wp_i4: &GpuTensor,
        w8_i8: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "nibble_expand_int4_to_int8",
            kernels::NIBBLE_EXPAND_INT4_TO_INT8_SRC,
            "nibble_expand_int4_to_int8",
        )?;
        let pp = wp_i4.buf.as_ptr();
        let op = w8_i8.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
        ];
        let n = m * k;
        let grid = ((n + 255) / 256) as u32;
        let func = &self.functions["nibble_expand_int4_to_int8"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 hash-routed MoE: GPU-side tid2eid lookup + score gather +
    /// softmax-normalize + route_scale multiply. Replaces the d2h+host+h2d
    /// round-trip in `ffn_hash_routed`.
    ///
    /// `tid2eid`: pre-uploaded `[vocab_size, k]` u32 lookup table for the
    /// hash layer. `scores`: `[n_exp]` router output (already on device).
    /// `topk_idx`/`topk_w`: `[k]` outputs (i32 / f32). On degenerate
    /// (non-positive score-sum) the weights are zeroed — caller still
    /// dispatches the MoE accumulator, which multiplies by 0.
    ///
    /// `token_id` is a kernarg (i32) — for HIP-graph capture, a
    /// `_buf` variant reading from a device buffer is the natural
    /// follow-up.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn hash_router_normalize_f32(
        &mut self,
        tid2eid: &GpuTensor,
        scores: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        token_id: i32,
        n_exp: i32,
        k: i32,
        route_scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hash_router_normalize_f32",
            kernels::HASH_ROUTER_NORMALIZE_SRC,
            "hash_router_normalize_f32",
        )?;
        let tp = tid2eid.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let tid = token_id;
        let ne = n_exp;
        let kv = k;
        let rs = route_scale;
        self.launch_kernargs(
            "hash_router_normalize_f32",
            [1, 1, 1],
            [1, 1, 1],
            0,
            &kernargs![ptr tp, ptr sp, ptr ip, ptr wp, i32 tid, i32 ne, i32 kv, f32 rs],
        )
    }
    /// Batched twin of `hash_router_normalize_f32_buf` — for the prefill
    /// `ffn_batched` hash-routed path. Single launch over batch positions:
    /// reads token_id from `token_ids[b]`, looks up tid2eid, gathers
    /// scores[b, eid], normalize + route_scale; writes `topk_idx[B, k]`
    /// and `topk_w[B, k]`. Eliminates the per-layer d2h(scores) + CPU
    /// loop + 2× h2d (idx+w) round-trip in batched prefill.
    #[allow(clippy::too_many_arguments)]
    pub fn hash_router_normalize_f32_batched(
        &mut self,
        tid2eid: &GpuTensor,
        scores: &GpuTensor,
        token_ids: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        n_exp: i32,
        k: i32,
        route_scale: f32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hash_router_normalize_f32_batched",
            kernels::HASH_ROUTER_NORMALIZE_BATCHED_SRC,
            "hash_router_normalize_f32_batched",
        )?;
        let tp = tid2eid.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let tb = token_ids.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let ne = n_exp;
        let kv = k;
        let rs = route_scale;
        let bs = batch_size;
        self.launch_kernargs(
            "hash_router_normalize_f32_batched",
            [batch_size as u32, 1, 1],
            [1, 1, 1],
            0,
            &kernargs![ptr tp, ptr sp, ptr tb, ptr ip, ptr wp, i32 ne, i32 kv, f32 rs, i32 bs],
        )
    }
    /// HIP-graphs-safe twin of `hash_router_normalize_f32` — reads
    /// `token_id` from `token_id_buf[0]` (device-resident) instead of
    /// a kernarg, so the captured graph re-reads it on every replay.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn hash_router_normalize_f32_buf(
        &mut self,
        tid2eid: &GpuTensor,
        scores: &GpuTensor,
        token_id_buf: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        n_exp: i32,
        k: i32,
        route_scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hash_router_normalize_f32_buf",
            kernels::HASH_ROUTER_NORMALIZE_BUF_SRC,
            "hash_router_normalize_f32_buf",
        )?;
        let tp = tid2eid.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let tb = token_id_buf.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let ne = n_exp;
        let kv = k;
        let rs = route_scale;
        self.launch_kernargs(
            "hash_router_normalize_f32_buf",
            [1, 1, 1],
            [1, 1, 1],
            0,
            &kernargs![ptr tp, ptr sp, ptr tb, ptr ip, ptr wp, i32 ne, i32 kv, f32 rs],
        )
    }
    /// Phase 3 — Apply α scaling to the 24-element HC control vector
    /// after `hc_compute_control` has run (which produces α=1 output).
    /// Rescales c[i] = α[seg(i)] · (c[i] - base[i]) + base[i] so each
    /// of the three segments (Ã/B̃/C̃) gets its proper α^pre/res/post.
    #[allow(dead_code)]
    pub fn hc_apply_alpha(
        &mut self,
        c: &GpuTensor,
        alpha: &GpuTensor,
        base: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_apply_alpha",
            kernels::HC_APPLY_ALPHA_SRC,
            "hc_apply_alpha",
        )?;
        let func = &self.functions["hc_apply_alpha"];
        let cp = c.buf.as_ptr();
        let ap = alpha.buf.as_ptr();
        let bp = base.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
                [24, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HC α-scaling — BATCHED. Per-batch in-place rescale of c[b, 0..24]
    /// using the shared 3-segment α + base. Byte-identical to
    /// `hc_apply_alpha` at batch_size == 1.
    #[allow(dead_code)]
    pub fn hc_apply_alpha_batched(
        &mut self,
        c: &GpuTensor,
        alpha: &GpuTensor,
        base: &GpuTensor,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_apply_alpha_batched",
            kernels::HC_APPLY_ALPHA_BATCHED_SRC,
            "hc_apply_alpha_batched",
        )?;
        let func = &self.functions["hc_apply_alpha_batched"];
        let cp = c.buf.as_ptr();
        let ap = alpha.buf.as_ptr();
        let bp = base.buf.as_ptr();
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [batch_size as u32, 1, 1],
                [24, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Phase 3 — `c = W_fn · x_flat + base`. Small GEMV producing the
    /// control vector that feeds Sinkhorn normalisation.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn hc_compute_control(
        &mut self,
        x_flat: &GpuTensor, // [x_dim] fp16
        w_fn: &GpuTensor,   // [n_ctrl, x_dim] fp16
        base: &GpuTensor,   // [n_ctrl] fp16
        c_out: &GpuTensor,  // [n_ctrl] fp32
        n_ctrl: i32,
        x_dim: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_compute_control",
            kernels::HC_COMPUTE_CONTROL_SRC,
            "hc_compute_control",
        )?;
        let func = &self.functions["hc_compute_control"];
        let xp = x_flat.buf.as_ptr();
        let wp = w_fn.buf.as_ptr();
        let bp = base.buf.as_ptr();
        let cp = c_out.buf.as_ptr();
        let mut nc = n_ctrl;
        let mut xd = x_dim;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut xd as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_ctrl as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HC control vector — BATCHED. Per batch row b reads
    /// `x_flat[b, :]`, dots against the shared `w_fn` rows, divides by
    /// rsqrt(mean(x^2)+eps), adds `base[ctrl]` → `c[b, ctrl]`. Byte-
    /// identical to `hc_compute_control` at batch_size == 1.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_compute_control_batched(
        &mut self,
        x_flat: &GpuTensor, // [batch, x_dim]
        w_fn: &GpuTensor,   // [n_ctrl, x_dim] fp16
        base: &GpuTensor,   // [n_ctrl] fp16
        c_out: &GpuTensor,  // [batch, n_ctrl] fp32
        n_ctrl: i32,
        x_dim: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_compute_control_batched",
            kernels::HC_COMPUTE_CONTROL_BATCHED_SRC,
            "hc_compute_control_batched",
        )?;
        let func = &self.functions["hc_compute_control_batched"];
        let xp = x_flat.buf.as_ptr();
        let wp = w_fn.buf.as_ptr();
        let bp = base.buf.as_ptr();
        let cp = c_out.buf.as_ptr();
        let mut nc = n_ctrl;
        let mut xd = x_dim;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut xd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_ctrl as u32, batch_size as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 head HC mix — compute the per-stream `pre` weights for the
    /// 4-stream → hidden projection before lm_head. Matches upstream
    /// `ParallelHead.hc_head`.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_head_compute_pre(
        &mut self,
        x_flat: &GpuTensor,  // [hc_mult * hidden] F32
        w_fn: &GpuTensor,    // [hc_mult, hc_mult * hidden] F16
        base: &GpuTensor,    // [hc_mult] F16
        pre_out: &GpuTensor, // [hc_mult] F32
        hc_mult: i32,
        x_dim: i32,
        scale: f32, // hc_head_scale (scalar)
        norm_eps: f32,
        hc_eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_head_compute_pre",
            kernels::HC_HEAD_COMPUTE_PRE_SRC,
            "hc_head_compute_pre",
        )?;
        let func = &self.functions["hc_head_compute_pre"];
        let xp = x_flat.buf.as_ptr();
        let wp = w_fn.buf.as_ptr();
        let bp = base.buf.as_ptr();
        let pp = pre_out.buf.as_ptr();
        let mut hm = hc_mult;
        let mut xd = x_dim;
        let mut sv = scale;
        let mut ne = norm_eps;
        let mut he = hc_eps;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &mut hm as *mut _ as *mut c_void,
            &mut xd as *mut _ as *mut c_void,
            &mut sv as *mut _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut he as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [hc_mult as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Phase 3 — Input mapping: x_in[d] = sum_s(A[s] * streams[s, d]).
    /// A is sigmoid-bounded [0, 1].
    #[allow(dead_code)]
    pub fn hc_input_map_4stream(
        &mut self,
        a_vec: &GpuTensor,
        streams: &GpuTensor,
        x_out: &GpuTensor,
        hidden: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_input_map_4stream",
            kernels::HC_INPUT_MAP_SRC,
            "hc_input_map_4stream",
        )?;
        let func = &self.functions["hc_input_map_4stream"];
        let ap = a_vec.buf.as_ptr();
        let sp = streams.buf.as_ptr();
        let op = x_out.buf.as_ptr();
        let mut h = hidden;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [((hidden + 255) / 256) as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Phase A5 — Batched HC input mapping. Per batch position b:
    /// `x_out[b, d] = sum_s(a_vec[b, s] * streams[b, s, d])`.
    /// At batch_size == 1, byte-identical to hc_input_map_4stream.
    #[allow(dead_code)]
    pub fn hc_input_map_4stream_batched(
        &mut self,
        a_vec: &GpuTensor,   // [batch, HC_MULT]
        streams: &GpuTensor, // [batch, HC_MULT, hidden]
        x_out: &GpuTensor,   // [batch, hidden]
        hidden: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_input_map_4stream_batched",
            kernels::HC_INPUT_MAP_BATCHED_SRC,
            "hc_input_map_4stream_batched",
        )?;
        let func = &self.functions["hc_input_map_4stream_batched"];
        let ap = a_vec.buf.as_ptr();
        let sp = streams.buf.as_ptr();
        let op = x_out.buf.as_ptr();
        let mut h = hidden;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [((hidden + 255) / 256) as u32, batch_size as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Phase 3 — Mix 4 residual streams via gating matrix + transform output.
    /// `x_out[s, d] = sum_t(A[s, t] * x_in[t, d]) + scale[s] * transform_out[d]`.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn hc_mix_4stream(
        &mut self,
        x_in: &GpuTensor,          // [4, hidden] fp16
        a_matrix: &GpuTensor,      // [4, 4] fp32 (post-Sinkhorn)
        scale: &GpuTensor,         // [4] fp32
        transform_out: &GpuTensor, // [hidden] fp16
        x_out: &GpuTensor,         // [4, hidden] fp16
        hidden: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_mix_4stream",
            kernels::HC_MIX_4STREAM_SRC,
            "hc_mix_4stream",
        )?;
        let func = &self.functions["hc_mix_4stream"];
        let xi = x_in.buf.as_ptr();
        let am = a_matrix.buf.as_ptr();
        let sc = scale.buf.as_ptr();
        let to = transform_out.buf.as_ptr();
        let xo = x_out.buf.as_ptr();
        let mut h = hidden;
        let mut params: Vec<*mut c_void> = vec![
            &xi as *const _ as *mut c_void,
            &am as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
            &to as *const _ as *mut c_void,
            &xo as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [((hidden + 255) / 256) as u32, 4, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Phase A5 — Batched HC 4-stream residual mix. Per batch position b:
    /// `x_out[b, s, d] = sum_t(A[b, s, t] * x_in[b, t, d]) + scale[b, s] * transform_out[b, d]`.
    /// At batch_size == 1, byte-identical to hc_mix_4stream.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn hc_mix_4stream_batched(
        &mut self,
        x_in: &GpuTensor,          // [batch, 4, hidden]
        a_matrix: &GpuTensor,      // [batch, 4, 4]
        scale: &GpuTensor,         // [batch, 4]
        transform_out: &GpuTensor, // [batch, hidden]
        x_out: &GpuTensor,         // [batch, 4, hidden]
        hidden: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_mix_4stream_batched",
            kernels::HC_MIX_4STREAM_BATCHED_SRC,
            "hc_mix_4stream_batched",
        )?;
        let func = &self.functions["hc_mix_4stream_batched"];
        let xi = x_in.buf.as_ptr();
        let am = a_matrix.buf.as_ptr();
        let sc = scale.buf.as_ptr();
        let to = transform_out.buf.as_ptr();
        let xo = x_out.buf.as_ptr();
        let mut h = hidden;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &xi as *const _ as *mut c_void,
            &am as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
            &to as *const _ as *mut c_void,
            &xo as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [((hidden + 255) / 256) as u32, 4, batch_size as u32],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    #[cfg(feature = "deltanet")]
    /// DeepSeek V4 mHC fused: hc_c[0..4] = sigmoid(hc_c[0..4]) + hc_eps;
    /// hc_c[4..8] = post_scale * sigmoid(hc_c[4..8]); hc_c[8..] unchanged.
    /// Replaces 3 element-wise launches (sigmoid(pre), sigmoid(post),
    /// scale(post)) with one 8-thread launch — saves 2 launches per
    /// mhc_pre call, ~860 μs/decode on 43-layer DeepSeek V4.
    #[allow(dead_code)]
    pub fn hc_pre_post_sigmoid_scale_f32(
        &mut self,
        hc_c: &GpuTensor,
        hc_eps: f32,
        post_scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_pre_post_sigmoid_scale_f32",
            kernels::HC_PRE_POST_SIGMOID_SCALE_SRC,
            "hc_pre_post_sigmoid_scale_f32",
        )?;
        let xp = hc_c.buf.as_ptr();
        let eps = hc_eps;
        let ps = post_scale;
        self.launch_kernargs(
            "hc_pre_post_sigmoid_scale_f32",
            [1, 1, 1],
            [8, 1, 1],
            0,
            &kernargs![ptr xp, f32 eps, f32 ps],
        )
    }
    /// Phase 3 — Sinkhorn-normalise a 4×4 gating matrix (in place).
    /// `matrix` is row-major 16 floats; `iters` = `hc_sinkhorn_iters`
    /// from DeepSeek V4 config (= 20). `eps` = `hc_eps` (= 1e-6).
    #[allow(dead_code)]
    pub fn hc_sinkhorn_4x4(&mut self, matrix: &GpuTensor, eps: f32, iters: i32) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(matrix.numel(), 16, "hc_sinkhorn_4x4 expects a 4x4 matrix");
        self.ensure_kernel(
            "hc_sinkhorn_4x4",
            kernels::HC_SINKHORN_4X4_SRC,
            "hc_sinkhorn_4x4",
        )?;
        let func = &self.functions["hc_sinkhorn_4x4"];
        let m_ptr = matrix.buf.as_ptr();
        let mut eps_v = eps;
        let mut iters_v = iters;
        let mut params: Vec<*mut c_void> = vec![
            &m_ptr as *const _ as *mut c_void,
            &mut eps_v as *mut _ as *mut c_void,
            &mut iters_v as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
                [32, 1, 1], // single-warp variant: 16 active lanes for the 4x4 matrix
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HC Sinkhorn 4×4 — BATCHED. Per-batch independent Sinkhorn
    /// iterations on each 4×4 matrix slot at `matrix[b * 16..]`. Byte-
    /// identical to `hc_sinkhorn_4x4` at batch_size == 1.
    #[allow(dead_code)]
    pub fn hc_sinkhorn_4x4_batched(
        &mut self,
        matrix: &GpuTensor,
        eps: f32,
        iters: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_sinkhorn_4x4_batched",
            kernels::HC_SINKHORN_4X4_BATCHED_SRC,
            "hc_sinkhorn_4x4_batched",
        )?;
        let func = &self.functions["hc_sinkhorn_4x4_batched"];
        let m_ptr = matrix.buf.as_ptr();
        let mut eps_v = eps;
        let mut iters_v = iters;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &m_ptr as *const _ as *mut c_void,
            &mut eps_v as *mut _ as *mut c_void,
            &mut iters_v as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [batch_size as u32, 1, 1],
                [32, 1, 1], // single-warp variant: 16 active lanes per batch row
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HC split + finalize — BATCHED. Per-batch position: applies
    /// sigmoid to c[b, 0..4] → pre[b], applies post_scale·sigmoid to
    /// c[b, 4..8] → post[b], and copies c[b, 8..24] → comb[b]. Avoids
    /// strided sigmoid_f32 launches on the [B, 24] layout.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_split_finalize_batched(
        &mut self,
        c: &GpuTensor,    // [B, 24]
        pre: &GpuTensor,  // [B, 4]
        post: &GpuTensor, // [B, 4]
        comb: &GpuTensor, // [B, 16]
        post_scale: f32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_split_finalize_batched",
            kernels::HC_SPLIT_FINALIZE_BATCHED_SRC,
            "hc_split_finalize_batched",
        )?;
        let func = &self.functions["hc_split_finalize_batched"];
        let cp = c.buf.as_ptr();
        let prp = pre.buf.as_ptr();
        let pop = post.buf.as_ptr();
        let cop = comb.buf.as_ptr();
        let mut ps = post_scale;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &prp as *const _ as *mut c_void,
            &pop as *const _ as *mut c_void,
            &cop as *const _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [batch_size as u32, 1, 1],
                [24, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Phase B2 — Broadcast batched embed `[B, hidden]` into all `hc_mult`
    /// slots of the residual-streams buffer `[B, hc_mult, hidden]`. Single
    /// kernel launch in place of B × hc_mult d2d memcpys.
    #[allow(dead_code)]
    pub fn hc_streams_init_from_embed_batched(
        &mut self,
        embed: &GpuTensor,
        streams: &GpuTensor,
        hidden: i32,
        hc_mult: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_streams_init_from_embed_batched",
            kernels::HC_STREAMS_INIT_FROM_EMBED_BATCHED_SRC,
            "hc_streams_init_from_embed_batched",
        )?;
        let func = &self.functions["hc_streams_init_from_embed_batched"];
        let ep = embed.buf.as_ptr();
        let sp = streams.buf.as_ptr();
        let mut h = hidden;
        let mut hm = hc_mult;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut hm as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [((hidden + 255) / 256) as u32, batch_size as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Phase 2 — Compressed-K scoring (Q · K^T over indexer-compressed positions).
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn indexer_compressed_k_score(
        &mut self,
        q_idx: &GpuTensor,       // [H, D] fp16
        k_idx_cache: &GpuTensor, // [H, D, N] fp16
        scores: &GpuTensor,      // [H, N] fp32
        n_idx_heads: i32,
        idx_head_dim: i32,
        n_compressed: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "indexer_compressed_k_score",
            kernels::INDEXER_COMPRESSED_K_SCORE_SRC,
            "indexer_compressed_k_score",
        )?;
        let func = &self.functions["indexer_compressed_k_score"];
        let qp = q_idx.buf.as_ptr();
        let kp = k_idx_cache.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let mut h = n_idx_heads;
        let mut d = idx_head_dim;
        let mut nc = n_compressed;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
        ];
        // grid.x = heads, grid.y = ceil(N / TILE_POSITIONS=8)
        let grid_y = ((n_compressed + 7) / 8).max(1) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_idx_heads as u32, grid_y, 1],
                [64, 1, 1], // THREADS_PER_BLOCK
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Phase 2 — Gather raw K/V rows from main cache at indexer indices.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn indexer_kv_gather(
        &mut self,
        k_main_cache: &GpuTensor,
        v_main_cache: &GpuTensor,
        unique_indices: &GpuTensor,
        k_gathered: &GpuTensor,
        v_gathered: &GpuTensor,
        n_kv_heads: i32,
        head_dim: i32,
        max_seq: i32,
        n_unique: i32,
        compress_ratio: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "indexer_kv_gather",
            kernels::INDEXER_KV_GATHER_SRC,
            "indexer_kv_gather",
        )?;
        let func = &self.functions["indexer_kv_gather"];
        let kc = k_main_cache.buf.as_ptr();
        let vc = v_main_cache.buf.as_ptr();
        let ui = unique_indices.buf.as_ptr();
        let kg = k_gathered.buf.as_ptr();
        let vg = v_gathered.buf.as_ptr();
        let mut nh = n_kv_heads;
        let mut hd = head_dim;
        let mut ms = max_seq;
        let mut nu = n_unique;
        let mut cr = compress_ratio;
        let mut params: Vec<*mut c_void> = vec![
            &kc as *const _ as *mut c_void,
            &vc as *const _ as *mut c_void,
            &ui as *const _ as *mut c_void,
            &kg as *const _ as *mut c_void,
            &vg as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void,
            &mut cr as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_unique as u32, n_kv_heads as u32, 1],
                [64, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 indexer score — BATCHED. Per batch position b scores every
    /// compressed slot against `q[b, :, :]` using `weights[b, :]`. The
    /// k_cache is shared across batch. `n_per_batch[b]` gives the per-
    /// batch causal cutoff; cache slots ≥ n_per_batch[b] are written
    /// with -inf so top-K skips them (handles within-chunk commits
    /// that batch row b shouldn't see).
    #[allow(clippy::too_many_arguments)]
    pub fn indexer_relu_score_batched_f32(
        &mut self,
        q: &GpuTensor,           // [B, H, D]
        k_cache: &GpuTensor,     // [N_max, D] shared
        weights: &GpuTensor,     // [B, H]
        n_per_batch: &GpuTensor, // [B] i32
        scores: &GpuTensor,      // [B, N_max] output
        n_idx_heads: i32,        // H
        idx_head_dim: i32,       // D
        n_max: i32,              // N_max (cache slots considered)
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "indexer_relu_score_batched",
            kernels::INDEXER_RELU_SCORE_BATCHED_SRC,
            "indexer_relu_score_batched_f32",
        )?;
        let func = &self.functions["indexer_relu_score_batched_f32"];
        let qp = q.buf.as_ptr();
        let kp = k_cache.buf.as_ptr();
        let wp = weights.buf.as_ptr();
        let np = n_per_batch.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let mut h = n_idx_heads;
        let mut d = idx_head_dim;
        let mut nc = n_max;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &np as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_max as u32, batch_size as u32, 1],
                [n_idx_heads as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// WMMA-accelerated BATCHED indexer scoring (Phase C1 of the
    /// deepseek4 prefill catch-up plan). Same math as
    /// `indexer_relu_score_batched_f32` but with a 16×16×16 WMMA tile
    /// of Q·K^T per warp + LDS reduction across heads.
    ///
    /// Requires H = 64 and idx_head_dim = 128 (DeepSeek V4 indexer
    /// shape — caller asserts before dispatch). gfx1100+ WMMA only.
    ///
    /// Grid:  [batch_size, ceil(N_max / 16), 1]
    /// Block: [128, 1, 1] (4 warps × 32 threads)
    #[allow(clippy::too_many_arguments)]
    pub fn indexer_relu_score_wmma_batched_f32(
        &mut self,
        q: &GpuTensor,           // [B, H, D]
        k_cache: &GpuTensor,     // [N_max, D] shared
        weights: &GpuTensor,     // [B, H]
        n_per_batch: &GpuTensor, // [B] i32
        scores: &GpuTensor,      // [B, N_max] output
        n_idx_heads: i32,
        idx_head_dim: i32,
        n_max: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            n_idx_heads, 64,
            "indexer_relu_score_wmma: requires H=64 (got {n_idx_heads})"
        );
        assert_eq!(
            idx_head_dim, 128,
            "indexer_relu_score_wmma: requires idx_head_dim=128 (got {idx_head_dim})"
        );
        self.ensure_kernel(
            "indexer_relu_score_wmma_batched",
            kernels::INDEXER_RELU_SCORE_WMMA_BATCHED_SRC,
            "indexer_relu_score_wmma_batched_f32",
        )?;
        let func = &self.functions["indexer_relu_score_wmma_batched_f32"];
        let qp = q.buf.as_ptr();
        let kp = k_cache.buf.as_ptr();
        let wp = weights.buf.as_ptr();
        let np = n_per_batch.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let mut h = n_idx_heads;
        let mut d = idx_head_dim;
        let mut nc = n_max;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &np as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let grid_n = (n_max as u32 + 15) / 16;
        unsafe {
            self.hip.launch_kernel(
                func,
                [batch_size as u32, grid_n, 1],
                [128, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 indexer scoring — combined across heads with relu gating.
    /// `scores[n] = sum_h relu(q[h, :] · k_cache[n, :]) * weights[h]`.
    /// Block per slot N, threads-per-block = H (one head per thread),
    /// LDS reduction across heads.
    /// HIP-graphs-safe twin of `indexer_relu_score_f32`. Reads `N` from
    /// a device buffer and launches with a FIXED grid sized to `max_n`
    /// (typically `HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS = 2048`). Blocks beyond
    /// N_buf[0] write a `-inf` sentinel into `scores` so the downstream
    /// `indexer_top_k_buf` (which also reads N from buf) ignores them.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn indexer_relu_score_f32_buf(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        weights: &GpuTensor,
        scores: &GpuTensor,
        n_buf: &GpuTensor,
        max_n: i32,
        h: i32,
        d: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "indexer_relu_score_f32_buf",
            kernels::INDEXER_RELU_SCORE_BUF_SRC,
            "indexer_relu_score_f32_buf",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k_cache.buf.as_ptr();
        let wp = weights.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let nbp = n_buf.buf.as_ptr();
        let hi = h;
        let di = d;
        self.launch_kernargs(
            "indexer_relu_score_f32_buf",
            [max_n as u32, 1, 1],
            [h as u32, 1, 1],
            0,
            &kernargs![ptr qp, ptr kp, ptr wp, ptr sp, ptr nbp, i32 hi, i32 di],
        )
    }
    /// Phase 2 — Per-head top-k selection.
    pub fn indexer_top_k(
        &mut self,
        scores: &GpuTensor,      // [H, N] fp32
        top_indices: &GpuTensor, // [H, K] i32
        n_idx_heads: i32,
        n_compressed: i32,
        k: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("indexer_top_k", kernels::INDEXER_TOP_K_SRC, "indexer_top_k")?;
        let func = &self.functions["indexer_top_k"];
        let sp = scores.buf.as_ptr();
        let ti = top_indices.buf.as_ptr();
        let mut h = n_idx_heads;
        let mut nc = n_compressed;
        let mut kk = k;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &ti as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];
        // shared mem = n_compressed bytes for the `taken` flag array.
        let smem = n_compressed as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_idx_heads as u32, 1, 1],
                [1, 1, 1], // stub single-thread per head
                smem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Phase A3 — Batched per-head top-K selection. Processes `batch_size`
    /// independent (batch_row, head) pairs in one launch. Each (b, h)
    /// block reads from `scores[b * H * N + h * N + ..]` and writes to
    /// `top_indices[b * H * K + h * K + ..]`. Byte-identical to
    /// `indexer_top_k` at batch_size == 1.
    pub fn indexer_top_k_batched(
        &mut self,
        scores: &GpuTensor,      // [B, H, N_stride] fp32
        top_indices: &GpuTensor, // [B, H, K_stride] i32
        n_idx_heads: i32,
        n_stride: i32, // score storage row stride
        n_iter: i32,   // actual iteration bound (≤ n_stride)
        k_stride: i32, // top_indices storage row stride
        k_fill: i32,   // ranks to fill (rest get -1)
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "indexer_top_k_batched",
            kernels::INDEXER_TOP_K_BATCHED_SRC,
            "indexer_top_k_batched",
        )?;
        let func = &self.functions["indexer_top_k_batched"];
        let sp = scores.buf.as_ptr();
        let ti = top_indices.buf.as_ptr();
        let mut h = n_idx_heads;
        let mut ns = n_stride;
        let mut ni = n_iter;
        let mut ks = k_stride;
        let mut kf = k_fill;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &ti as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut ns as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut ks as *mut _ as *mut c_void,
            &mut kf as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let smem = n_iter as u32;
        // Block sized to parallelise the fast-path identity write of
        // up to k_stride indices across threads (each thread writes
        // k_stride/128 slots via stride). Slow path serialises on
        // thread 0 — extra threads early-return.
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_idx_heads as u32, batch_size as u32, 1],
                [128, 1, 1],
                smem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HIP-graphs-safe twin of `indexer_top_k`. Reads `n_compressed` and
    /// `k` from device buffers; the output `top_indices` is sized to
    /// `max_k` per head and ranks ≥ k are filled with `-1` sentinels.
    /// Shared memory is sized for `max_n_compressed` to keep capture-
    /// time launch constant.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn indexer_top_k_buf(
        &mut self,
        scores: &GpuTensor,
        top_indices: &GpuTensor,
        n_compressed_buf: &GpuTensor,
        k_buf: &GpuTensor,
        n_idx_heads: i32,
        max_n_compressed: i32,
        max_k: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "indexer_top_k_buf",
            kernels::INDEXER_TOP_K_BUF_SRC,
            "indexer_top_k_buf",
        )?;
        let sp = scores.buf.as_ptr();
        let ti = top_indices.buf.as_ptr();
        let nbp = n_compressed_buf.buf.as_ptr();
        let kbp = k_buf.buf.as_ptr();
        let h = n_idx_heads;
        let mk = max_k;
        let smem = max_n_compressed as u32;
        // Block sized to parallelise the fast-path identity write of
        // up to max_k indices across threads (each thread writes
        // multiple slots via stride). The slow-path selection-sort
        // still serialises on thread 0 only — the extra threads
        // early-return in that branch.
        self.launch_kernargs(
            "indexer_top_k_buf",
            [n_idx_heads as u32, 1, 1],
            [128, 1, 1],
            smem,
            &kernargs![ptr sp, ptr ti, ptr nbp, ptr kbp, i32 h, i32 mk],
        )
    }
    /// DeepSeek V4 MoE routing affinity: sqrt(softplus(x)) elementwise in-place.
    #[allow(dead_code)]
    pub fn sqrt_softplus_f32(&mut self, x: &GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "sqrt_softplus_f32",
            kernels::SQRT_SOFTPLUS_F32_SRC,
            "sqrt_softplus_f32",
        )?;
        let func = &self.functions["sqrt_softplus_f32"];
        let n = x.numel() as i32;
        let xp = x.buf.as_ptr();
        let mut nv = n;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
        ];
        let grid_x = ((n + 255) / 256) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 position-0 attention: per-head sigmoid-of-(Q·K + attn_sink),
    /// times V, reduced over o_groups.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn deepseek4_attn_pos0(
        &mut self,
        q: &GpuTensor,
        kv: &GpuTensor,
        attn_sink: &GpuTensor,
        attn_out: &GpuTensor,
        n_heads: i32,
        head_dim: i32,
        o_groups: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_attn_pos0",
            kernels::V4F_ATTN_POS0_SRC,
            "deepseek4_attn_pos0",
        )?;
        let func = &self.functions["deepseek4_attn_pos0"];
        let qp = q.buf.as_ptr();
        let kp = kv.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut og = o_groups;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut og as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [head_dim as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn deepseek4_attn_swa(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        attn_sink: &GpuTensor,
        attn_out: &GpuTensor,
        n_heads: i32,
        head_dim: i32,
        o_groups: i32,
        n_valid: i32,
        window: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_attn_swa",
            kernels::V4F_ATTN_SWA_SRC,
            "deepseek4_attn_swa",
        )?;
        let func = &self.functions["deepseek4_attn_swa"];
        let qp = q.buf.as_ptr();
        let kp = k_cache.buf.as_ptr();
        let vp = v_cache.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut og = o_groups;
        let mut nv = n_valid;
        let mut wn = window;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut og as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
            &mut wn as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [head_dim as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 batched pure-SWA attention. Twin of `deepseek4_attn_swa_topk_batched_f32`
    /// for layers without an indexer top-K path. At batch_size == 1 the
    /// math is byte-identical to `deepseek4_attn_swa`.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn deepseek4_attn_swa_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        attn_sink: &GpuTensor,
        n_valid_arr: &GpuTensor,
        attn_out: &GpuTensor,
        n_heads: i32,
        head_dim: i32,
        o_groups: i32,
        window: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_attn_swa_batched",
            kernels::V4F_ATTN_SWA_BATCHED_SRC,
            "deepseek4_attn_swa_batched",
        )?;
        let func = &self.functions["deepseek4_attn_swa_batched"];
        let qp = q.buf.as_ptr();
        let kp = k_cache.buf.as_ptr();
        let vp = v_cache.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let nvp = n_valid_arr.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut og = o_groups;
        let mut wn = window;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &nvp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut og as *mut _ as *mut c_void,
            &mut wn as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, batch_size as u32, 1],
                [head_dim as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DEBUG twin of deepseek4_attn_swa_batched. Same compute; ALSO writes
    /// per-(h, b) max_score and sum_exp into the debug scratch tensors.
    /// Used to bisect non-determinism inside the kernel (commit b1e8aad
    /// trail). Not part of the production path.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn deepseek4_attn_swa_batched_debug(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        attn_sink: &GpuTensor,
        n_valid_arr: &GpuTensor,
        attn_out: &GpuTensor,
        debug_max: &GpuTensor,    // [batch, n_heads] f32
        debug_sumexp: &GpuTensor, // [batch, n_heads] f32
        n_heads: i32,
        head_dim: i32,
        o_groups: i32,
        window: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_attn_swa_batched_debug",
            kernels::V4F_ATTN_SWA_BATCHED_DEBUG_SRC,
            "deepseek4_attn_swa_batched_debug",
        )?;
        let func = &self.functions["deepseek4_attn_swa_batched_debug"];
        let qp = q.buf.as_ptr();
        let kp = k_cache.buf.as_ptr();
        let vp = v_cache.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let nvp = n_valid_arr.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let dmp = debug_max.buf.as_ptr();
        let dsp = debug_sumexp.buf.as_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut og = o_groups;
        let mut wn = window;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &nvp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &dmp as *const _ as *mut c_void,
            &dsp as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut og as *mut _ as *mut c_void,
            &mut wn as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, batch_size as u32, 1],
                [head_dim as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 SWA-windowed attention with attn_sink (multi-position).
    /// Generalises `deepseek4_attn_pos0` to attend over a cache of up to
    /// `window` past KV positions.
    #[allow(dead_code, clippy::too_many_arguments)]
    /// HIP-graphs-safe twin of `deepseek4_attn_swa`: reads `n_valid` from a
    /// device buffer instead of an i32 kernarg.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn deepseek4_attn_swa_buf(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        attn_sink: &GpuTensor,
        attn_out: &GpuTensor,
        n_valid_buf: &GpuTensor,
        n_heads: i32,
        head_dim: i32,
        o_groups: i32,
        window: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_attn_swa_buf",
            kernels::V4F_ATTN_SWA_BUF_SRC,
            "deepseek4_attn_swa_buf",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k_cache.buf.as_ptr();
        let vp = v_cache.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let nvp = n_valid_buf.buf.as_ptr();
        let nh = n_heads;
        let hd = head_dim;
        let og = o_groups;
        let wn = window;
        self.launch_kernargs(
            "deepseek4_attn_swa_buf",
            [n_heads as u32, 1, 1],
            [head_dim as u32, 1, 1],
            0,
            &kernargs![ptr qp, ptr kp, ptr vp, ptr sp, ptr op, ptr nvp, i32 nh, i32 hd, i32 og, i32 wn],
        )
    }
    /// DeepSeek V4 batched indexer-extended SWA attention. Processes B query
    /// positions in one launch. Each batch row has its own SWA K/V slice
    /// (`[batch, head_dim, swa_window]`), top-K K/V slice (`[batch,
    /// head_dim, topk_window]`), and per-row valid-count scalars
    /// (`n_valid_swa_arr[batch]`, `n_active_topk_arr[batch]`, i32 GPU
    /// buffers). attn_sink and the host-side scalars are shared across
    /// the batch. At batch_size == 1 the math is byte-identical to
    /// `deepseek4_attn_swa_topk_f32`.
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek4_attn_swa_topk_batched_f32(
        &mut self,
        q: &GpuTensor,
        swa_k: &GpuTensor,
        swa_v: &GpuTensor,
        topk_k: &GpuTensor,
        topk_v: &GpuTensor,
        attn_sink: &GpuTensor,
        n_valid_swa_arr: &GpuTensor,
        n_active_topk_arr: &GpuTensor,
        attn_out: &GpuTensor,
        n_heads: i32,
        head_dim: i32,
        swa_window: i32,
        topk_window: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_attn_swa_topk_batched",
            kernels::V4F_ATTN_SWA_TOPK_BATCHED_SRC,
            "deepseek4_attn_swa_topk_batched_f32",
        )?;
        let func = &self.functions["deepseek4_attn_swa_topk_batched_f32"];
        let qp = q.buf.as_ptr();
        let kp = swa_k.buf.as_ptr();
        let vp = swa_v.buf.as_ptr();
        let tkp = topk_k.buf.as_ptr();
        let tvp = topk_v.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let nvp = n_valid_swa_arr.buf.as_ptr();
        let nap = n_active_topk_arr.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut sw = swa_window;
        let mut tw = topk_window;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &tkp as *const _ as *mut c_void,
            &tvp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &nvp as *const _ as *mut c_void,
            &nap as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sw as *mut _ as *mut c_void,
            &mut tw as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, batch_size as u32, 1],
                [head_dim as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek4_attn_swa_topk_direct_batched_f32(
        &mut self,
        q: &GpuTensor,
        swa_k: &GpuTensor,
        swa_v: &GpuTensor,
        kv_cache: &GpuTensor,
        topk_idx: &GpuTensor,
        attn_sink: &GpuTensor,
        n_valid_swa_arr: &GpuTensor,
        n_active_topk_arr: &GpuTensor,
        attn_out: &GpuTensor,
        n_heads: i32,
        head_dim: i32,
        swa_window: i32,
        topk_window: i32,
        n_compressed: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_attn_swa_topk_direct_batched",
            kernels::V4F_ATTN_SWA_TOPK_DIRECT_BATCHED_SRC,
            "deepseek4_attn_swa_topk_direct_batched_f32",
        )?;
        let func = &self.functions["deepseek4_attn_swa_topk_direct_batched_f32"];
        let qp = q.buf.as_ptr();
        let kp = swa_k.buf.as_ptr();
        let vp = swa_v.buf.as_ptr();
        let cp = kv_cache.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let nvp = n_valid_swa_arr.buf.as_ptr();
        let nap = n_active_topk_arr.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut sw = swa_window;
        let mut tw = topk_window;
        let mut nc = n_compressed;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &nvp as *const _ as *mut c_void,
            &nap as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sw as *mut _ as *mut c_void,
            &mut tw as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, batch_size as u32, 1],
                [head_dim as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 indexer-extended SWA attention. Reads from the SWA ring
    /// buffer (`swa_k/v` [n_kv=1, head_dim, swa_window]) AND the
    /// indexer-gathered top-K K/V (`topk_k/v` [n_kv=1, head_dim,
    /// topk_window]) under a single joint softmax with `attn_sink` as
    /// an extra entry.
    /// HIP-graphs-safe twin of `deepseek4_attn_swa_topk_f32`. Reads
    /// `n_valid_swa` + `n_active_topk` from device buffers. Grid is
    /// fixed at `n_heads` so capture sees constant launch shape.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn deepseek4_attn_swa_topk_f32_buf(
        &mut self,
        q: &GpuTensor,
        swa_k: &GpuTensor,
        swa_v: &GpuTensor,
        topk_k: &GpuTensor,
        topk_v: &GpuTensor,
        attn_sink: &GpuTensor,
        attn_out: &GpuTensor,
        n_valid_swa_buf: &GpuTensor,
        n_active_topk_buf: &GpuTensor,
        n_heads: i32,
        head_dim: i32,
        swa_window: i32,
        topk_window: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_attn_swa_topk_f32_buf",
            kernels::V4F_ATTN_SWA_TOPK_BUF_SRC,
            "deepseek4_attn_swa_topk_f32_buf",
        )?;
        let qp = q.buf.as_ptr();
        let kp = swa_k.buf.as_ptr();
        let vp = swa_v.buf.as_ptr();
        let tkp = topk_k.buf.as_ptr();
        let tvp = topk_v.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let nvp = n_valid_swa_buf.buf.as_ptr();
        let nap = n_active_topk_buf.buf.as_ptr();
        let nh = n_heads;
        let hd = head_dim;
        let sw = swa_window;
        let tw = topk_window;
        self.launch_kernargs(
            "deepseek4_attn_swa_topk_f32_buf",
            [n_heads as u32, 1, 1],
            [head_dim as u32, 1, 1],
            0,
            &kernargs![ptr qp, ptr kp, ptr vp, ptr tkp, ptr tvp, ptr sp, ptr op, ptr nvp, ptr nap, i32 nh, i32 hd, i32 sw, i32 tw],
        )
    }
    /// DeepSeek V4-asymmetric-clamped fused SwiGLU + FWHT rotation. Replaces
    /// the DeepSeek V4 decode pair `deepseek4_silu_mul_clamp_f32 + rotate_x_mq` with
    /// one launch. Same shape contract as `fused_silu_mul_rotate_mq`
    /// plus a `swiglu_limit` scalar; pass 0.0 to disable the clamp
    /// (degenerates to fused_silu_mul_rotate_mq behavior modulo
    /// numerical noise).
    pub fn deepseek4_fused_silu_mul_clamp_mq_rotate(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        swiglu_limit: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "deepseek4_fused_silu_mul_clamp_mq_rotate",
            kernels::V4F_FUSED_SILU_MUL_CLAMP_MQ_ROTATE_SRC,
            "deepseek4_fused_silu_mul_clamp_mq_rotate",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let gp = gate.buf.as_ptr();
        let up_p = up.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let kv = k as i32;
        let lim = swiglu_limit;
        let bytes = k * 4 * 3 + 2 * 256 * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "deepseek4_fused_silu_mul_clamp_mq_rotate",
            bytes,
        );
        let result = self.launch_kernargs(
            "deepseek4_fused_silu_mul_clamp_mq_rotate",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr gp, ptr up_p, ptr s1_ptr, ptr s2_ptr, ptr xrp, i32 kv, f32 lim],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }
    /// DeepSeek V4 fused MoE down GEMV with scaled residual add. Atomically
    /// accumulates Σ_k topk_weights[k] * (W_down · rot_batch[k]) into
    /// x_residual. One launch replaces k_top per-expert calls.
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,  // [k_top × K]
        x_residual: &GpuTensor, // [M]
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq2g256_lloyd_moe_down_indexed",
            kernels::GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_SRC,
            "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        // MQ2-Lloyd: 72 bytes / 256-weight group.
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = (k_top as usize) * (mq2_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed",
            bytes,
        );
        let result = self.launch_kernargs(
            "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed",
            [m as u32, k_top as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr pp, ptr ip, ptr wp, ptr rbp, ptr xrp, i32 m_val, i32 k_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// MiniMax-M2 fused MoE gate_up GEMV for MQ3-Lloyd experts (3-bit +
    /// 8-entry codebook, 112 B/group). Sibling of
    /// `deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed` — identical grid +
    /// expert-ptrs dispatch; only the per-group byte stride differs (112 vs 72).
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek4_gemv_mq3g256_lloyd_moe_gate_up_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,  // [n_exp] u64 device pointers
        topk_indices: &GpuTensor, // [k_top] i32
        x_rot: &GpuTensor,        // [K] FWHT-rotated
        y_gate: &GpuTensor,       // [k_top × M/2]
        y_up: &GpuTensor,         // [k_top × M/2]
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq3g256_lloyd_moe_gate_up_indexed",
            kernels::GEMV_MQ3G256_LLOYD_MOE_GATE_UP_INDEXED_SRC,
            "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x_rot.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        // MQ3-Lloyd: 112 bytes / 256-weight group.
        let mq3_weight_bytes = m * (k / 256) * 112;
        let bytes = (k_top as usize) * (mq3_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq3g256_lloyd_moe_gate_up_indexed",
            bytes,
        );
        let result = self.launch_kernargs(
            "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed",
            [m as u32, k_top as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr pp, ptr ip, ptr xp, ptr ygp, ptr yup, i32 m_val, i32 k_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// MiniMax-M2 fused MoE down GEMV with scaled residual add for MQ3-Lloyd
    /// experts (3-bit + 8-entry codebook, 112 B/group). Sibling of
    /// `deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed` — only
    /// the per-group byte stride differs (112 vs 72).
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,  // [k_top × K]
        x_residual: &GpuTensor, // [M]
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq3g256_lloyd_moe_down_indexed",
            kernels::GEMV_MQ3G256_LLOYD_MOE_DOWN_INDEXED_SRC,
            "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        // MQ3-Lloyd: 112 bytes / 256-weight group.
        let mq3_weight_bytes = m * (k / 256) * 112;
        let bytes = (k_top as usize) * (mq3_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed",
            bytes,
        );
        let result = self.launch_kernargs(
            "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed",
            [m as u32, k_top as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr pp, ptr ip, ptr wp, ptr rbp, ptr xrp, i32 m_val, i32 k_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// K4-unrolled variant of
    /// `deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched`.
    /// 4 independent accumulators per thread for ILP. FMA-order-epsilon
    /// drift from the single-acc variant; not bit-identical.
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched_k4(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,
        x_residual: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq2g256_lloyd_moe_down_indexed_batched_k4",
            kernels::GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_BATCHED_K4_SRC,
            "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let rbp = rot_batch.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = batch_size * (k_top as usize) * (mq2_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched_k4",
            bytes,
        );
        let result = self.launch_kernargs(
            "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &kernargs![ptr pp, ptr ip, ptr wp, ptr rbp, ptr xrp, i32 m_val, i32 k_val, i32 kt_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// DeepSeek V4 fused MoE gate_up GEMV — one launch dispatches all top-K
    /// experts via per-layer expert pointer table + per-token topk
    /// indices. K_top is parameterised (DeepSeek V4 uses 6; kernel name's "_k8_"
    /// is from the Qwen35 sibling — the kernel body uses `krank =
    /// blockIdx.y` and accepts any k_top).
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,  // [n_exp] u64 device pointers
        topk_indices: &GpuTensor, // [k_top] i32
        x_rot: &GpuTensor,        // [K] FWHT-rotated
        y_gate: &GpuTensor,       // [k_top × M/2]
        y_up: &GpuTensor,         // [k_top × M/2]
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq2g256_lloyd_moe_gate_up_indexed",
            kernels::GEMV_MQ2G256_LLOYD_MOE_GATE_UP_INDEXED_SRC,
            "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x_rot.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        // MQ2-Lloyd: 72 bytes / 256-weight group.
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = (k_top as usize) * (mq2_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed",
            bytes,
        );
        let result = self.launch_kernargs(
            "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed",
            [m as u32, k_top as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr pp, ptr ip, ptr xp, ptr ygp, ptr yup, i32 m_val, i32 k_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// DeepSeek V4 MoE gate_up — POSITION-BATCHED MQ2-Lloyd K4-UNROLLED variant.
    /// Same shape contract and grid as
    /// `deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched` but the
    /// kernel uses 4 independent accumulators per thread for better
    /// instruction-level parallelism. Output drifts within FMA-order
    /// epsilon from the single-acc variant; not bit-identical.
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4",
            kernels::GEMV_MQ2G256_LLOYD_MOE_GATE_UP_INDEXED_BATCHED_K4_SRC,
            "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = x_rot.buf.as_ptr();
        let ygp = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = batch_size * (k_top as usize) * (mq2_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4",
            bytes,
        );
        let result = self.launch_kernargs(
            "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &kernargs![ptr pp, ptr ip, ptr xp, ptr ygp, ptr yup, i32 m_val, i32 k_val, i32 kt_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// DeepSeek V4 MoE router top-K — POSITION-BATCHED. Per-batch row runs
    /// the same bias-aware top-K + normalize + route_scale logic as
    /// the sequential `deepseek4_moe_topk_bias_aware_f32`. Block per batch row.
    /// Byte-identical to sequential at batch_size == 1.
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek4_moe_topk_bias_aware_batched_f32(
        &mut self,
        scores: &GpuTensor,  // [B, n_exp]
        bias: &GpuTensor,    // [n_exp]
        indices: &GpuTensor, // [B, k_top]
        weights: &GpuTensor, // [B, k_top]
        n_exp: i32,
        k_top: i32,
        route_scale: f32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_moe_topk_bias_aware_batched",
            kernels::V4F_MOE_TOPK_BIAS_AWARE_BATCHED_SRC,
            "deepseek4_moe_topk_bias_aware_batched_f32",
        )?;
        let func = &self.functions["deepseek4_moe_topk_bias_aware_batched_f32"];
        let sp = scores.buf.as_ptr();
        let bp = bias.buf.as_ptr();
        let ip = indices.buf.as_ptr();
        let wp = weights.buf.as_ptr();
        let mut ne = n_exp;
        let mut kt = k_top;
        let mut rs = route_scale;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut kt as *mut _ as *mut c_void,
            &mut rs as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [batch_size as u32, 1, 1],
                [n_exp as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 MoE router: GPU-side bias-aware top-K + normalized weights.
    /// Replaces the per-layer D2H scores → CPU top-K → H2D
    /// indices+weights round trip. `bias` may be a zero buffer for
    /// hash-routed layers. `indices` is written as i32 (its GpuTensor
    /// dtype is F32 because hipfire's tensor-shape machinery only carries
    /// f32 buffers, but kernels see the raw i32 bytes).
    pub fn deepseek4_moe_topk_bias_aware_f32(
        &mut self,
        scores: &GpuTensor,  // [n_exp] fp32
        bias: &GpuTensor,    // [n_exp] fp32 (zero if hash-routed)
        indices: &GpuTensor, // [k_top] i32 (typed as F32; raw bytes)
        weights: &GpuTensor, // [k_top] fp32
        n_exp: i32,
        k_top: i32,
        route_scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_moe_topk_bias_aware",
            kernels::V4F_MOE_TOPK_BIAS_AWARE_SRC,
            "deepseek4_moe_topk_bias_aware_f32",
        )?;
        let func = &self.functions["deepseek4_moe_topk_bias_aware_f32"];
        let sp = scores.buf.as_ptr();
        let bp = bias.buf.as_ptr();
        let ip = indices.buf.as_ptr();
        let wp = weights.buf.as_ptr();
        let mut ne = n_exp;
        let mut kt = k_top;
        let mut rs = route_scale;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut kt as *mut _ as *mut c_void,
            &mut rs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
                [n_exp as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 SwiGLU with swiglu_limit clamp.
    /// out[i] = silu(min(gate[i], L)) * clamp(up[i], -L, +L), where L = swiglu_limit.
    pub fn deepseek4_silu_mul_clamp_f32(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        out: &GpuTensor,
        swiglu_limit: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_silu_mul_clamp",
            kernels::V4F_SILU_MUL_CLAMP_SRC,
            "deepseek4_silu_mul_clamp_f32",
        )?;

        let n = gate.numel() as i32;
        let gate_ptr = gate.buf.as_ptr();
        let up_ptr = up.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let n_val = n;
        let limit_val = swiglu_limit;

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise_bytes(n as usize);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "deepseek4_silu_mul_clamp_f32",
            bytes,
        );
        let result = self.launch_kernargs(
            "deepseek4_silu_mul_clamp_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &kernargs![ptr gate_ptr, ptr up_ptr, ptr out_ptr, i32 n_val, f32 limit_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched DeepSeek V4 SwiGLU+clamp. `gate`, `up`, `out` each hold `batch`
    /// independent streams of length `n` laid out contiguously (stride =
    /// n). Per-stream math is byte-identical to `deepseek4_silu_mul_clamp_f32`;
    /// the kernel reads `batch_off = blockIdx.y * n` and indexes within
    /// the stream. Used by the DeepSeek V4 MoE expert loop to collapse a
    /// k_top-sized launch sequence into one.
    pub fn deepseek4_silu_mul_clamp_f32_batched(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        out: &GpuTensor,
        n: usize,
        batch: usize,
        swiglu_limit: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_silu_mul_clamp",
            kernels::V4F_SILU_MUL_CLAMP_SRC,
            "deepseek4_silu_mul_clamp_f32",
        )?;

        let n_i32 = n as i32;
        let gate_ptr = gate.buf.as_ptr();
        let up_ptr = up.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let n_val = n_i32;
        let limit_val = swiglu_limit;

        let block = 256u32;
        let grid = ((n_i32 as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise_bytes(n) * batch;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "deepseek4_silu_mul_clamp_f32_batched",
            bytes,
        );
        let result = self.launch_kernargs(
            "deepseek4_silu_mul_clamp_f32",
            [grid, batch as u32, 1],
            [block, 1, 1],
            0,
            &kernargs![ptr gate_ptr, ptr up_ptr, ptr out_ptr, i32 n_val, f32 limit_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// DeepSeek V4 top-K K/V gather — BATCHED. Per batch position b uses its
    /// own top-K index list `topk_idx[b, :]` (typically produced by
    /// `indexer_top_k_batched`) and writes into its own slice of the
    /// staged output `[B, head_dim, out_stride]`. Sentinel indices < 0
    /// or ≥ n_compressed write zeros.
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek4_topk_kv_gather_batched_f32(
        &mut self,
        kv_cache: &GpuTensor, // [N_compressed, head_dim] shared
        topk_idx: &GpuTensor, // [B, K] i32
        out: &GpuTensor,      // [B, head_dim, out_stride]
        k_active: i32,
        head_dim: i32,
        n_compressed: i32,
        out_stride: i32,
        col_offset: i32,
        scale: f32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_topk_kv_gather_batched",
            kernels::V4F_TOPK_KV_GATHER_BATCHED_SRC,
            "deepseek4_topk_kv_gather_batched_f32",
        )?;
        let func = &self.functions["deepseek4_topk_kv_gather_batched_f32"];
        let cp = kv_cache.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let op = out.buf.as_ptr();
        let mut k = k_active;
        let mut hd = head_dim;
        let mut nc = n_compressed;
        let mut os = out_stride;
        let mut co = col_offset;
        let mut sc = scale;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
            &mut co as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [k_active as u32, batch_size as u32, 1],
                [head_dim as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 indexer-extended attention K/V gather. Reads from
    /// `main_kv_cache` [N_compressed, head_dim] at the indices given by
    /// `topk_idx` [K] and writes into columns `[col_offset, col_offset+K)`
    /// of an [n_kv=1, head_dim, out_stride] output tensor — letting the
    /// caller stage the gather into a buffer whose first `col_offset`
    /// columns hold raw SWA window K/V. Sentinel `topk_idx[k] = -1` (or
    /// out-of-range) writes zeros. The `scale` parameter multiplies the
    /// gathered values; pass 1.0 for pass-through, larger to compensate
    /// for compressor.norm undershoot.
    #[allow(clippy::too_many_arguments)]
    /// HIP-graphs-safe twin of `deepseek4_topk_kv_gather_f32`: reads K_buf[0]
    /// and N_compressed_buf[0] from device buffers. Launches with a
    /// FIXED grid sized to `max_k` (so capture sees a constant grid);
    /// blocks beyond K_buf[0] early-return.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn deepseek4_topk_kv_gather_f32_buf(
        &mut self,
        kv_cache: &GpuTensor,
        topk_idx: &GpuTensor,
        out: &GpuTensor,
        k_buf: &GpuTensor,
        n_compressed_buf: &GpuTensor,
        max_k: i32, // upper bound on K — sets the captured grid size
        head_dim: i32,
        out_stride: i32,
        col_offset: i32,
        scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_topk_kv_gather_f32_buf",
            kernels::V4F_TOPK_KV_GATHER_BUF_SRC,
            "deepseek4_topk_kv_gather_f32_buf",
        )?;
        let cp = kv_cache.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let op = out.buf.as_ptr();
        let kbp = k_buf.buf.as_ptr();
        let ncp = n_compressed_buf.buf.as_ptr();
        let hd = head_dim;
        let os = out_stride;
        let co = col_offset;
        let sc = scale;
        self.launch_kernargs(
            "deepseek4_topk_kv_gather_f32_buf",
            [max_k as u32, 1, 1],
            [head_dim as u32, 1, 1],
            0,
            &kernargs![ptr cp, ptr ip, ptr op, ptr kbp, ptr ncp, i32 hd, i32 os, i32 co, f32 sc],
        )
    }
    /// DeepSeek V4 identity gather — BATCHED. For ratio=128 layers without an
    /// indexer: copies the same `kv_cache[0..K, :]` into every batch
    /// row's slab. Same shape as deepseek4_topk_kv_gather_batched_f32 but
    /// without the per-batch index lookup.
    #[allow(dead_code)]
    pub fn deepseek4_topk_kv_gather_identity_batched_f32(
        &mut self,
        kv_cache: &GpuTensor, // [N_compressed, head_dim] shared
        out: &GpuTensor,      // [B, head_dim, out_stride]
        k_active: i32,
        head_dim: i32,
        out_stride: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_topk_kv_gather_identity_batched",
            kernels::V4F_TOPK_KV_GATHER_IDENTITY_BATCHED_SRC,
            "deepseek4_topk_kv_gather_identity_batched_f32",
        )?;
        let func = &self.functions["deepseek4_topk_kv_gather_identity_batched_f32"];
        let cp = kv_cache.buf.as_ptr();
        let op = out.buf.as_ptr();
        let mut k = k_active;
        let mut hd = head_dim;
        let mut os = out_stride;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [k_active as u32, batch_size as u32, 1],
                [head_dim as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// HIP-graphs-safe twin of `deepseek4_topk_kv_gather_identity_f32`.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn deepseek4_topk_kv_gather_identity_f32_buf(
        &mut self,
        kv_cache: &GpuTensor,
        out: &GpuTensor,
        k_buf: &GpuTensor,
        max_k: i32,
        head_dim: i32,
        out_stride: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_topk_kv_gather_identity_f32_buf",
            kernels::V4F_TOPK_KV_GATHER_IDENTITY_BUF_SRC,
            "deepseek4_topk_kv_gather_identity_f32_buf",
        )?;
        let cp = kv_cache.buf.as_ptr();
        let op = out.buf.as_ptr();
        let kbp = k_buf.buf.as_ptr();
        let hd = head_dim;
        let os = out_stride;
        self.launch_kernargs(
            "deepseek4_topk_kv_gather_identity_f32_buf",
            [max_k as u32, 1, 1],
            [head_dim as u32, 1, 1],
            0,
            &kernargs![ptr cp, ptr op, ptr kbp, i32 hd, i32 os],
        )
    }
    /// DeepSeek V4 per-group O-LoRA batched GEMV (F32 weights). Block-diagonal:
    /// `y[b, g, r] = sum_k wo_a[g, r, k] * x_in[b, g, k]`. Single
    /// launch processes B batch positions × G groups × M output rows
    /// — replaces `B * G` separate gemv_f32 calls. F32-only for now;
    /// Q8/MQ4 weights need separate batched-per-group kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn wo_per_group_batched_f32(
        &mut self,
        wo_a: &GpuTensor,  // [G, M, K] F32
        x_in: &GpuTensor,  // [B, G, K]
        y_out: &GpuTensor, // [B, G, M]
        g: i32,
        m: i32,
        k: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "wo_per_group_batched_f32",
            kernels::WO_PER_GROUP_BATCHED_F32_SRC,
            "wo_per_group_batched_f32",
        )?;
        let func = &self.functions["wo_per_group_batched_f32"];
        let wp = wo_a.buf.as_ptr();
        let xp = x_in.buf.as_ptr();
        let yp = y_out.buf.as_ptr();
        let mut g_i = g;
        let mut m_i = m;
        let mut k_i = k;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut g_i as *mut _ as *mut c_void,
            &mut m_i as *mut _ as *mut c_void,
            &mut k_i as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_size as u32, g as u32],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 per-group O-LoRA batched GEMV — HFQ4G256-packed wo_a.
    /// Sibling of `wo_per_group_batched_f32` for the MQ4 case. Input
    /// `x_in` must be FWHT-pre-rotated. Single launch in place of B×G
    /// gemv_mq4g256_prerotated calls.
    #[allow(clippy::too_many_arguments)]
    pub fn wo_per_group_batched_hfq4g256(
        &mut self,
        wo_a: &GpuTensor,  // [G * M * K / 256 * 136] bytes
        x_in: &GpuTensor,  // [B, G, K] FWHT-rotated
        y_out: &GpuTensor, // [B, G, M]
        g: i32,
        m: i32,
        k: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "wo_per_group_batched_hfq4g256",
            kernels::WO_PER_GROUP_BATCHED_HFQ4G256_SRC,
            "wo_per_group_batched_hfq4g256",
        )?;
        let func = &self.functions["wo_per_group_batched_hfq4g256"];
        let wp = wo_a.buf.as_ptr();
        let xp = x_in.buf.as_ptr();
        let yp = y_out.buf.as_ptr();
        let mut g_i = g;
        let mut m_i = m;
        let mut k_i = k;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut g_i as *mut _ as *mut c_void,
            &mut m_i as *mut _ as *mut c_void,
            &mut k_i as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_size as u32, g as u32],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// DeepSeek V4 per-group O-LoRA batched GEMV — Q8_0-packed wo_a.
    /// Sibling of `wo_per_group_batched_hfq4g256` for the Q8 case
    /// (deepseek4-lloyd-mq2-q8 builds). Single launch in place of B × G
    /// per-position `gemv_q8_0` calls — collapses the per-(b, g) loop
    /// in `attention_block_batched_*` for Q8_0 wo_a.
    #[allow(clippy::too_many_arguments)]
    pub fn wo_per_group_batched_q8_0(
        &mut self,
        wo_a: &GpuTensor,  // [G * M * K / 32 * 34] bytes (Q8_0-packed)
        x_in: &GpuTensor,  // [B, G, K] plain F32 (no FWHT)
        y_out: &GpuTensor, // [B, G, M]
        g: i32,
        m: i32,
        k: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        // DeepSeek V4 prefill shape on gfx1151 (G=8, M=1024, K=4096,
        // B=1024): strided WMMA is ~10x faster than the scalar per-row
        // kernel. Env keeps a one-command fallback for bisects.
        let default_wmma = self.arch == "gfx1151" && k % 32 == 0 && m >= 64 && batch_size >= 64;
        let use_wmma = std::env::var("HIPFIRE_DEEPSEEK4_WO_Q8_WMMA")
            .map(|s| s != "0")
            .unwrap_or(default_wmma);
        if use_wmma && k % 32 == 0 {
            return self.wo_per_group_batched_q8_0_wmma_4w(wo_a, x_in, y_out, g, m, k, batch_size);
        }
        self.wo_per_group_batched_q8_0_1w(wo_a, x_in, y_out, g, m, k, batch_size)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn wo_per_group_batched_q8_0_1w(
        &mut self,
        wo_a: &GpuTensor,  // [G * M * K / 32 * 34] bytes (Q8_0-packed)
        x_in: &GpuTensor,  // [B, G, K] plain F32 (no FWHT)
        y_out: &GpuTensor, // [B, G, M]
        g: i32,
        m: i32,
        k: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "wo_per_group_batched_q8_0",
            kernels::WO_PER_GROUP_BATCHED_Q8_0_SRC,
            "wo_per_group_batched_q8_0",
        )?;
        let func = &self.functions["wo_per_group_batched_q8_0"];
        let wp = wo_a.buf.as_ptr();
        let xp = x_in.buf.as_ptr();
        let yp = y_out.buf.as_ptr();
        let mut g_i = g;
        let mut m_i = m;
        let mut k_i = k;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut g_i as *mut _ as *mut c_void,
            &mut m_i as *mut _ as *mut c_void,
            &mut k_i as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_size as u32, g as u32],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// `wo_per_group_batched_q8_0` but one block processes R output rows,
    /// hoisting x loads across rows. Grid = [ceil(M/R), B, G].
    /// `rows_per_block` must be 2 or 4.
    #[allow(clippy::too_many_arguments)]
    pub fn wo_per_group_batched_q8_0_multirow(
        &mut self,
        wo_a: &GpuTensor,
        x_in: &GpuTensor,
        y_out: &GpuTensor,
        g: i32,
        m: i32,
        k: i32,
        batch_size: i32,
        rows_per_block: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (name, grid_x) = match rows_per_block {
            2 => (
                "wo_per_group_batched_q8_0_multirow_r2",
                ((m as u32) + 1) / 2,
            ),
            4 => (
                "wo_per_group_batched_q8_0_multirow_r4",
                ((m as u32) + 3) / 4,
            ),
            _ => {
                return Err(hip_bridge::HipError::new(
                    1,
                    "wo_per_group_batched_q8_0_multirow: rows_per_block must be 2 or 4",
                ))
            }
        };
        self.ensure_kernel(name, kernels::WO_PER_GROUP_BATCHED_Q8_0_MULTIROW_SRC, name)?;
        let func = &self.functions[name];
        let wp = wo_a.buf.as_ptr();
        let xp = x_in.buf.as_ptr();
        let yp = y_out.buf.as_ptr();
        let mut g_i = g;
        let mut m_i = m;
        let mut k_i = k;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut g_i as *mut _ as *mut c_void,
            &mut m_i as *mut _ as *mut c_void,
            &mut k_i as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_size as u32, g as u32],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn wo_per_group_batched_q8_0_wmma_4w(
        &mut self,
        wo_a: &GpuTensor,  // [G * M * K / 32 * 34] bytes (Q8_0-packed)
        x_in: &GpuTensor,  // [B, G, K] plain F32 or F16
        y_out: &GpuTensor, // [B, G, M]
        g: i32,
        m: i32,
        k: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            k % 32,
            0,
            "wo_per_group_batched_q8_0_wmma_4w: K must divide 32"
        );
        self.ensure_kernel(
            "wo_per_group_batched_q8_0_wmma_4w",
            kernels::WO_PER_GROUP_BATCHED_Q8_0_WMMA_4W_SRC,
            "wo_per_group_batched_q8_0_wmma_4w",
        )?;
        let xp_owned = x_in.buf.as_ptr();
        let mut xp = if matches!(x_in.dtype, DType::F16) {
            xp_owned
        } else {
            // Production prefill reuses the same x_in tensor pointer every
            // layer with new contents, so pointer-keyed conversion caching
            // would read stale FP16 here.
            self.convert_fp16_x_uncached(x_in, batch_size as usize * g as usize * k as usize)?
        };
        let func = &self.functions["wo_per_group_batched_q8_0_wmma_4w"];
        let mut wp = wo_a.buf.as_ptr();
        let mut yp = y_out.buf.as_ptr();
        let mut g_i = g;
        let mut m_i = m;
        let mut k_i = k;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut g_i as *mut _ as *mut c_void,
            &mut m_i as *mut _ as *mut c_void,
            &mut k_i as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [
                    ((m + 63) / 64) as u32,
                    ((batch_size + 63) / 64) as u32,
                    g as u32,
                ],
                [128, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Atomic-free MQ2-Lloyd MoE down GEMV (K4-unrolled). Writes per-
    /// (token, krank) expert outputs to `expert_outputs[N × K_TOP × M]`
    /// — no atomicAdd. Pair with `moe_down_combine_k8_batched` to fold
    /// K_TOP outputs into x_residual deterministically.
    ///
    /// Same grid/block as the residual_scaled K4 variant, so this is a
    /// drop-in replacement for the scalar-path down GEMV when temp=0
    /// reproducibility is required.
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
        &mut self,
        expert_ptrs: &GpuTensor,    // [n_exp]
        topk_indices: &GpuTensor,   // [N × K_TOP]
        rot_batch: &GpuTensor,      // [N × K_TOP × K]
        expert_outputs: &GpuTensor, // [N × K_TOP × M] (written, no atomic)
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemv_mq2g256_lloyd_moe_down_expanded_k4",
            kernels::GEMV_MQ2G256_LLOYD_MOE_DOWN_EXPANDED_K4_SRC,
            "gemv_mq2g256_lloyd_moe_down_expanded_k4",
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = rot_batch.buf.as_ptr();
        let yp = expert_outputs.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let kt_val = k_top as i32;
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = batch_size * (k_top as usize) * (mq2_weight_bytes + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            "deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4",
            bytes,
        );
        let result = self.launch_kernargs(
            "gemv_mq2g256_lloyd_moe_down_expanded_k4",
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &kernargs![ptr pp, ptr ip, ptr xp, ptr yp, i32 m_val, i32 k_val, i32 kt_val],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
