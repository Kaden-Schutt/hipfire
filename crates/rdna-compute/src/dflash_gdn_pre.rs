// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S5 (launch-fusion): `Gpu` launchers for the single-launch GDN preambles
//! (`dflash_gdn_pre_capture_gfx1100` / `dflash_gdn_pre_replay_gfx1100`,
//! gfx1100-only).
//!
//! The kernel source is self-contained here via `include_str!` so no shared
//! registry (`kernels.rs` / `replay.rs`) changes are needed. Both launchers
//! go through `launch_maybe_blob` (blob retained through any graph-exec
//! lifetime) with `ensure_kernel` first, exactly like the kernels they
//! replace — so the fused launches are capture-safe wherever the old ones
//! were.
//!
//! Eligibility is strict and host-side: exact gfx1100, head_dim == 128,
//! consistent k/v dims, sequential N (capture) / n_steps (replay) in
//! 1..=16, and GQA ratio > 1 on capture (the interleave branch the fixture
//! takes) / >= 1 on replay (ratio == 1 matches the old memcpy path
//! byte-for-byte). Ineligible shapes return `Ok(false)` and the caller runs
//! the pre-change path. The `DflashFusionCtx`, kill switch, tree-exclusion,
//! and tape-presence gates live at the call sites (prefill hook /
//! `GdnTape::replay_gdn_inner`), which own that context.

use crate::dispatch::{Gpu, GpuTensor};
use hip_bridge::{HipResult, KernargBlob};
use std::ffi::c_void;

/// Kernel source for both [`Gpu::dflash_gdn_pre_capture_gfx1100`] and
/// [`Gpu::dflash_gdn_pre_replay_gfx1100`].
pub const DFLASH_GDN_PRE_GFX1100_SRC: &str =
    include_str!("../../../kernels/src/dflash_gdn_pre.gfx1100.hip");
/// Compiled-module key for the GDN-pre kernels.
pub const DFLASH_GDN_PRE_GFX1100_MODULE: &str = "dflash_gdn_pre_gfx1100";
/// Device symbol for the verify-side capture kernel.
pub const DFLASH_GDN_PRE_CAPTURE_GFX1100_SYMBOL: &str = "dflash_gdn_pre_capture_gfx1100";
/// Device symbol for the replay-side kernel.
pub const DFLASH_GDN_PRE_REPLAY_GFX1100_SYMBOL: &str = "dflash_gdn_pre_replay_gfx1100";
/// Threads per block (one Q/K head, one 256-wide V stripe, or prep per block).
pub const DFLASH_GDN_PRE_BLOCK: u32 = 256;
/// Only head_dim == 128 is fused (matches the `GDN_PRE_HD` staging).
pub const DFLASH_GDN_PRE_HEAD_DIM: usize = 128;
/// Sequential batch ceiling for the fused row loop (DFlash verify block).
pub const DFLASH_GDN_PRE_MAX_N: usize = 16;

impl Gpu {
    /// JIT the GDN-pre kernels (idempotent). Called on first fused launch;
    /// never JITs inside graph capture (callers warm up before capturing,
    /// like every other batched kernel).
    pub fn ensure_dflash_gdn_pre_gfx1100(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            DFLASH_GDN_PRE_GFX1100_MODULE,
            DFLASH_GDN_PRE_GFX1100_SRC,
            DFLASH_GDN_PRE_CAPTURE_GFX1100_SYMBOL,
        )?;
        self.ensure_kernel(
            DFLASH_GDN_PRE_GFX1100_MODULE,
            DFLASH_GDN_PRE_GFX1100_SRC,
            DFLASH_GDN_PRE_REPLAY_GFX1100_SYMBOL,
        )
    }

    /// Shared shape gate for both pre-kernels. Returns the
    /// `(n_key_heads, ratio, v_blocks)` triple on success, `Ok(None)` when
    /// the shapes must stay on the pre-change path.
    fn dflash_gdn_pre_eligible(
        &self,
        n_v_heads: usize,
        n_key_heads: usize,
        head_dim: usize,
        k_dim: usize,
        v_dim: usize,
        n: usize,
        need_gqa: bool,
    ) -> HipResult<Option<(u32, u32, u32)>> {
        if !self.arch_caps.is_gfx1100() {
            return Ok(None);
        }
        if head_dim != DFLASH_GDN_PRE_HEAD_DIM {
            return Ok(None);
        }
        if n_key_heads == 0 || n_v_heads == 0 || n_v_heads % n_key_heads != 0 {
            return Ok(None);
        }
        let ratio = n_v_heads / n_key_heads;
        if need_gqa && ratio <= 1 {
            return Ok(None);
        }
        if k_dim != n_key_heads * head_dim || v_dim != n_v_heads * head_dim {
            return Ok(None);
        }
        if n == 0 || n > DFLASH_GDN_PRE_MAX_N {
            return Ok(None);
        }
        let v_blocks = ((v_dim as u32) + DFLASH_GDN_PRE_BLOCK - 1) / DFLASH_GDN_PRE_BLOCK;
        if v_blocks == 0 {
            return Ok(None);
        }
        Ok(Some((n_key_heads as u32, ratio as u32, v_blocks)))
    }

    /// Verify-side fused GDN preamble: sigmoid(alpha/beta) + tape writes +
    /// conv + QK norm/interleave in one launch. Returns `Ok(true)` when the
    /// fused launch was issued, `Ok(false)` when the caller must run the
    /// pre-change sequence.
    ///
    /// Buffers (all F32, dense row-major): `beta`/`alpha` [N x n_v_heads]
    /// in/out; `qkv_in` [N x qkv_dim] raw projection (never modified);
    /// `conv_state` single-lane [n_channels x 3] (advanced exactly like the
    /// old batched conv); `q_raw`/`k_raw` [N x k_dim] receive conv outputs
    /// (old interleave-path postcondition); `v_out`/`q_dst`/`k_dst`
    /// [N x v_dim]; tape bufs receive rows at `tape_offset + t`.
    /// `q_scale` must be `1/sqrt(hd)` (host-computed, as before).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn dflash_gdn_pre_capture_gfx1100(
        &mut self,
        beta: &GpuTensor,
        alpha: &GpuTensor,
        dt_bias: &GpuTensor,
        a_log: &GpuTensor,
        qkv_in: &GpuTensor,
        conv_weight: &GpuTensor,
        conv_state: &GpuTensor,
        q_raw: &GpuTensor,
        k_raw: &GpuTensor,
        v_out: &GpuTensor,
        q_dst: &GpuTensor,
        k_dst: &GpuTensor,
        tape_qkv: &GpuTensor,
        tape_alpha: &GpuTensor,
        tape_beta: &GpuTensor,
        n_v_heads: usize,
        n_key_heads: usize,
        head_dim: usize,
        k_dim: usize,
        v_dim: usize,
        qkv_dim: usize,
        n_tokens: usize,
        tape_offset: usize,
        q_scale: f32,
        eps: f32,
    ) -> HipResult<bool> {
        let Some((nkh, ratio, v_blocks)) = self.dflash_gdn_pre_eligible(
            n_v_heads,
            n_key_heads,
            head_dim,
            k_dim,
            v_dim,
            n_tokens,
            /*need_gqa=*/ true,
        )?
        else {
            return Ok(false);
        };
        if qkv_dim != 2 * k_dim + v_dim {
            return Ok(false);
        }
        self.bind_thread()?;
        self.ensure_dflash_gdn_pre_gfx1100()?;

        let bp = beta.buf.as_ptr();
        let ap = alpha.buf.as_ptr();
        let dp = dt_bias.buf.as_ptr();
        let lp = a_log.buf.as_ptr();
        let ip = qkv_in.buf.as_ptr();
        let wp = conv_weight.buf.as_ptr();
        let sp = conv_state.buf.as_ptr();
        let qrp = q_raw.buf.as_ptr();
        let krp = k_raw.buf.as_ptr();
        let vp = v_out.buf.as_ptr();
        let qdp = q_dst.buf.as_ptr();
        let kdp = k_dst.buf.as_ptr();
        let tqp = tape_qkv.buf.as_ptr();
        let tap = tape_alpha.buf.as_ptr();
        let tbp = tape_beta.buf.as_ptr();
        let nvh = n_v_heads as i32;
        let nkh_i = nkh as i32;
        let ratio_i = ratio as i32;
        let kd = k_dim as i32;
        let vd = v_dim as i32;
        let qd = qkv_dim as i32;
        let nt = n_tokens as i32;
        let toff = tape_offset as i32;
        let qs = q_scale;
        let ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &bp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &lp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &qrp as *const _ as *mut c_void,
            &krp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &qdp as *const _ as *mut c_void,
            &kdp as *const _ as *mut c_void,
            &tqp as *const _ as *mut c_void,
            &tap as *const _ as *mut c_void,
            &tbp as *const _ as *mut c_void,
            &nvh as *const _ as *mut c_void,
            &nkh_i as *const _ as *mut c_void,
            &ratio_i as *const _ as *mut c_void,
            &kd as *const _ as *mut c_void,
            &vd as *const _ as *mut c_void,
            &qd as *const _ as *mut c_void,
            &nt as *const _ as *mut c_void,
            &toff as *const _ as *mut c_void,
            &qs as *const _ as *mut c_void,
            &ep as *const _ as *mut c_void,
        ];
        let grid = nkh + v_blocks + 1;
        let bytes = crate::profile::conv1d_silu_bytes(2 * k_dim + v_dim) * n_tokens
            + crate::profile::elementwise1_bytes(n_v_heads * head_dim) * 2 * n_tokens
            + qkv_dim * 4 * n_tokens;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            DFLASH_GDN_PRE_CAPTURE_GFX1100_SYMBOL,
            bytes,
        );
        let result = self.launch_maybe_blob(
            DFLASH_GDN_PRE_CAPTURE_GFX1100_SYMBOL,
            [grid, 1, 1],
            [DFLASH_GDN_PRE_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(bp);
                b.push_ptr(ap);
                b.push_ptr(dp);
                b.push_ptr(lp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(sp);
                b.push_ptr(qrp);
                b.push_ptr(krp);
                b.push_ptr(vp);
                b.push_ptr(qdp);
                b.push_ptr(kdp);
                b.push_ptr(tqp);
                b.push_ptr(tap);
                b.push_ptr(tbp);
                b.push_i32(nvh);
                b.push_i32(nkh_i);
                b.push_i32(ratio_i);
                b.push_i32(kd);
                b.push_i32(vd);
                b.push_i32(qd);
                b.push_i32(nt);
                b.push_i32(toff);
                b.push_f32(qs);
                b.push_f32(ep);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result.map(|()| true)
    }

    /// Replay-side fused GDN preamble: conv (from taped raw qkv) + QK
    /// norm/interleave in one launch. Returns `Ok(true)` when issued,
    /// `Ok(false)` for the pre-change path. `q_raw`/`k_raw` keep the old
    /// in-place-norm postcondition (normed values); `q_dst`/`k_dst` are the
    /// repeated outputs. `alpha`/`beta` are never touched (the GDN kernels
    /// read them from tape directly).
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_gdn_pre_replay_gfx1100(
        &mut self,
        qkv_tape: &GpuTensor,
        conv_weight: &GpuTensor,
        conv_state: &GpuTensor,
        q_raw: &GpuTensor,
        k_raw: &GpuTensor,
        v_out: &GpuTensor,
        q_dst: &GpuTensor,
        k_dst: &GpuTensor,
        n_v_heads: usize,
        n_key_heads: usize,
        head_dim: usize,
        k_dim: usize,
        v_dim: usize,
        qkv_dim: usize,
        n_steps: usize,
        q_scale: f32,
        eps: f32,
    ) -> HipResult<bool> {
        let Some((nkh, ratio, v_blocks)) = self.dflash_gdn_pre_eligible(
            n_v_heads,
            n_key_heads,
            head_dim,
            k_dim,
            v_dim,
            n_steps,
            /*need_gqa=*/ false,
        )?
        else {
            return Ok(false);
        };
        if qkv_dim != 2 * k_dim + v_dim {
            return Ok(false);
        }
        self.bind_thread()?;
        self.ensure_dflash_gdn_pre_gfx1100()?;

        let ip = qkv_tape.buf.as_ptr();
        let wp = conv_weight.buf.as_ptr();
        let sp = conv_state.buf.as_ptr();
        let qrp = q_raw.buf.as_ptr();
        let krp = k_raw.buf.as_ptr();
        let vp = v_out.buf.as_ptr();
        let qdp = q_dst.buf.as_ptr();
        let kdp = k_dst.buf.as_ptr();
        let nvh = n_v_heads as i32;
        let nkh_i = nkh as i32;
        let ratio_i = ratio as i32;
        let kd = k_dim as i32;
        let vd = v_dim as i32;
        let qd = qkv_dim as i32;
        let ns = n_steps as i32;
        let qs = q_scale;
        let ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &qrp as *const _ as *mut c_void,
            &krp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &qdp as *const _ as *mut c_void,
            &kdp as *const _ as *mut c_void,
            &nvh as *const _ as *mut c_void,
            &nkh_i as *const _ as *mut c_void,
            &ratio_i as *const _ as *mut c_void,
            &kd as *const _ as *mut c_void,
            &vd as *const _ as *mut c_void,
            &qd as *const _ as *mut c_void,
            &ns as *const _ as *mut c_void,
            &qs as *const _ as *mut c_void,
            &ep as *const _ as *mut c_void,
        ];
        let grid = nkh + v_blocks;
        let bytes = crate::profile::conv1d_silu_bytes(2 * k_dim + v_dim) * n_steps
            + crate::profile::elementwise1_bytes(n_v_heads * head_dim) * 2 * n_steps;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            DFLASH_GDN_PRE_REPLAY_GFX1100_SYMBOL,
            bytes,
        );
        let result = self.launch_maybe_blob(
            DFLASH_GDN_PRE_REPLAY_GFX1100_SYMBOL,
            [grid, 1, 1],
            [DFLASH_GDN_PRE_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(sp);
                b.push_ptr(qrp);
                b.push_ptr(krp);
                b.push_ptr(vp);
                b.push_ptr(qdp);
                b.push_ptr(kdp);
                b.push_i32(nvh);
                b.push_i32(nkh_i);
                b.push_i32(ratio_i);
                b.push_i32(kd);
                b.push_i32(vd);
                b.push_i32(qd);
                b.push_i32(ns);
                b.push_f32(qs);
                b.push_f32(ep);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result.map(|()| true)
    }
}
