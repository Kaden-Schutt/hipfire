// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fused MQ4G256V2 LM-head GEMM + greedy DDTree top-K/log-sum-exp selection,
//! exact gfx1100 only (S8-topk-direct-lmhead).
//!
//! Replaces the `[B, vocab]` logits-materializing `gemm_mq4g256v2_batched_lmhead`
//! + `topk_logsumexp_batched_f32` pair on the `sample == None` DDTree proposal
//! path with ONE resident-workgroup kernel
//! (`mq4v2_lmhead_topk_direct_gfx1100`). Logits never exist in memory: each
//! resident single-wave workgroup folds its WMMA tile outputs straight into a
//! register-resident running top-K and an online (max, sumexp) pair; a
//! grid-wide generation barrier (legal because the host-computed grid is
//! statically proven resident) hands off to a per-row reduce that writes the
//! `[N, K]` ids/log-probs directly.
//!
//! Contracts:
//! - exact gfx1100 (`arch_caps.is_gfx1100() && arch == "gfx1100"`), N <= 16,
//!   K % 256 == 0, K_TOP in 1..=8. Every failed predicate is an `Err` so the
//!   caller runs the pre-change GEMM+topk path.
//! - Occupancy proof is static: the kernel carries `__launch_bounds__(32, 8)`
//!   (8 co-resident one-wave blocks per CU; 96 CU * 8 = 768 >= grid), zero
//!   LDS, and the launcher clamps the grid to `min(m_tiles, 768)`.
//! - Launch goes through `launch_maybe_blob` with a `KernargBlob`, stable
//!   caller-owned scratch pointers, no host read/alloc in the launch body.
//! - Counter/generation words live in `ctl` (zeroed once at scratch
//!   allocation); the kernel self-resets them before returning, and HIP
//!   stream serialization guarantees a quiescent gap between launches, so
//!   back-to-back calls need no host-side reinit.
//! - Kill switch: `HIPFIRE_DDTREE_TOPK_DIRECT_OFF=1`
//!   (`flags.ddtree_topk_direct_off`) — the *caller* checks it and stays on
//!   the old path; this launcher also refuses to run when the flag is set.

use std::ffi::c_void;

use hip_bridge::HipResult;

use crate::dispatch::{Gpu, GpuTensor};

/// Kernel-side constants, mirrored from the HIP source. Scratch sizing and
/// the launcher's grid clamp depend on these; keep in sync.
pub const TDK_MAX_K: usize = 8;
pub const TDK_N_MAX: usize = 16;
/// 96 CU x 8 resident one-wave blocks (static `__launch_bounds__(32, 8)`).
pub const TDK_WG_MAX: usize = 768;

const MODULE: &str = "mq4v2_lmhead_topk_direct_gfx1100";
const FUNC: &str = "mq4v2_lmhead_topk_direct_gfx1100";
const SRC: &str = include_str!("../../../kernels/src/mq4v2_lmhead_topk_direct.gfx1100.hip");

/// Bytes required for the `partials` scratch tensor.
pub fn ddtree_topk_partials_bytes() -> usize {
    // Per-(wg, lane) lists over all 32 lanes (each column's tile rows split
    // across the lane-parity halves) + per-(wg, lane) online (max, sumexp).
    (TDK_WG_MAX * 32 * TDK_MAX_K * 2 + TDK_WG_MAX * 32 * 2) * 4
}

impl Gpu {
    /// Fused LM-head + greedy top-K/log-sum-exp for the DDTree proposal path.
    ///
    /// - `a_raw`: `[m, k]` MQ4G256V2 (qt=44) packed lm_head weights.
    /// - `x_f32`: `[n, k]` F32 FWHT-rotated hidden rows; converted to fp16
    ///   here (cache-stomped first, mirroring `gemm_mq4g256v2_batched_lmhead`).
    /// - `partials`: Raw scratch of at least [`ddtree_topk_partials_bytes`].
    /// - `ctl`: Raw scratch of at least 8 bytes (arrival counter, generation),
    ///   zeroed once before first use.
    /// - `top_idx`: `[n * k_top]` f32-storage tensor; kernel writes i32 ids.
    /// - `top_logp`: `[n * k_top]` f32 log-probs (`logit - log_z`).
    ///
    /// Selection (ids and their order for distinct values) is exact vs the
    /// `gemm + topk_logsumexp_batched_f32` baseline; log-prob accumulation is
    /// reassociated (fp32 online tree vs the oracle's f64 accumulator) and is
    /// gated against an f64 floor by `test_mq4v2_topk_direct_gfx1100`.
    #[allow(clippy::too_many_arguments)]
    pub fn mq4v2_lmhead_topk_direct_gfx1100(
        &mut self,
        a_raw: &GpuTensor,
        x_f32: &GpuTensor,
        partials: &GpuTensor,
        ctl: &GpuTensor,
        top_idx: &GpuTensor,
        top_logp: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
        k_top: usize,
    ) -> HipResult<()> {
        if self.flags.ddtree_topk_direct_off {
            return Err(hip_bridge::HipError::new(
                1,
                "mq4v2_lmhead_topk_direct_gfx1100: HIPFIRE_DDTREE_TOPK_DIRECT_OFF=1",
            ));
        }
        if !(self.arch_caps.is_gfx1100() && self.arch == "gfx1100") {
            return Err(hip_bridge::HipError::new(
                1,
                &format!(
                    "mq4v2_lmhead_topk_direct_gfx1100: exact gfx1100 required (got {})",
                    self.arch
                ),
            ));
        }
        if n == 0 || n > TDK_N_MAX {
            return Err(hip_bridge::HipError::new(
                1,
                &format!("mq4v2_lmhead_topk_direct_gfx1100: N must be in [1,{TDK_N_MAX}] (got {n})"),
            ));
        }
        if k == 0 || k % 256 != 0 {
            return Err(hip_bridge::HipError::new(
                1,
                &format!("mq4v2_lmhead_topk_direct_gfx1100: K must be a nonzero multiple of 256 (got {k})"),
            ));
        }
        // Mirror of `Gpu::residual_ksplit_kw` (gemm.rs; kept private there):
        // the baseline DDTree draft lm_head runs the ks{2,4,8} tier and the
        // kernel reproduces its exact split-K association, so the fused route
        // must use the SAME kw the baseline would dispatch. `None` here means
        // the baseline itself would fall off the tier — the caller falls back.
        let kw = {
            let g = k / 256;
            let want = if k <= 8192 { 4 } else { 8 };
            [want, 4, 2]
                .into_iter()
                .filter(|&kw| kw <= want)
                .find(|&kw| g >= kw && g % kw == 0)
        };
        let Some(kw) = kw else {
            return Err(hip_bridge::HipError::new(
                1,
                &format!("mq4v2_lmhead_topk_direct_gfx1100: no ks{{2,4,8}} split for K={k}"),
            ));
        };
        if !(1..=TDK_MAX_K).contains(&k_top) {
            return Err(hip_bridge::HipError::new(
                1,
                &format!("mq4v2_lmhead_topk_direct_gfx1100: K_TOP must be in [1,{TDK_MAX_K}] (got {k_top})"),
            ));
        }
        if partials.buf.size() < ddtree_topk_partials_bytes() {
            return Err(hip_bridge::HipError::new(
                1,
                &format!(
                    "mq4v2_lmhead_topk_direct_gfx1100: partials scratch {} < {} bytes",
                    partials.buf.size(),
                    ddtree_topk_partials_bytes()
                ),
            ));
        }
        if ctl.buf.size() < 8 {
            return Err(hip_bridge::HipError::new(
                1,
                "mq4v2_lmhead_topk_direct_gfx1100: ctl scratch must be >= 8 bytes",
            ));
        }

        self.bind_thread()?;
        self.ensure_kernel(MODULE, SRC, FUNC)?;

        // Mirror `gemm_mq4g256v2_batched_lmhead`: force the fp16 conversion
        // (the rotated scratch is rewritten every call) and convert here.
        self.scratch.fp16_x_source_ptr = std::ptr::null_mut();
        let x_f16_ptr = self.ensure_fp16_x(x_f32, n * k)?;

        let m_tiles = m.div_ceil(16);
        // Host-computed resident grid: statically bounded by
        // __launch_bounds__(32, 8) → 8 blocks/CU × 96 CU = 768.
        let wgs = m_tiles.min(TDK_WG_MAX) as u32;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
        let mut p_ptr = partials.buf.as_ptr();
        let mut c_ptr = ctl.buf.as_ptr();
        let mut ti_ptr = top_idx.buf.as_ptr();
        let mut tl_ptr = top_logp.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = n as i32;
        let mut kt_val = k_top as i32;
        let mut kw_val = kw as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut p_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut ti_ptr as *mut _ as *mut c_void,
            &mut tl_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut kt_val as *mut _ as *mut c_void,
            &mut kw_val as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + n * k * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", FUNC, bytes);
        let result = self.launch_maybe_blob(
            FUNC,
            [wgs, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(p_ptr);
                b.push_ptr(c_ptr);
                b.push_ptr(ti_ptr);
                b.push_ptr(tl_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(kt_val);
                b.push_i32(kw_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
