// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S7 (dflash draft launch collapse) GPU launchers, gfx1100-only.
//!
//! The kernels live in `kernels/src/dflash_draft_collapse.gfx1100.hip` and
//! are self-contained here via `include_str!` (no shared-registry edits).
//! Every launcher uses `launch_maybe_blob` + `KernargBlob` so the fast path
//! stays hipGraph-capturable (draft FFN graph mode included).

use crate::{Gpu, GpuTensor};
use hip_bridge::HipResult;
use std::ffi::c_void;

const COLLAPSE_SRC: &str = include_str!("../../../kernels/src/dflash_draft_collapse.gfx1100.hip");

/// Which overwrite GEMM the S7 fast path may use for one MQ4G256 dispatch.
///
/// Mirrors the default variant selection in
/// [`Gpu::gemm_hfq4g256_residual_wmma`]: `m >= 8192` runs the k2 schedule,
/// smaller M runs deterministic ksplit. Any non-default policy (mw16,
/// ldsstage, explicit `HIPFIRE_WO_WMMA_VARIANT`) resolves to [`Off`](DraftCollapseGemm::Off)
/// so the caller keeps today's path.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DraftCollapseGemm {
    Off,
    OverwriteK2,
    OverwriteKsplitDet,
}

/// Which overwrite GEMM the S7 fast path may use for one MQ4G256V2 dispatch.
///
/// Mirrors the gfx1100 production tier in
/// [`Gpu::gemm_mq4g256v2_residual_wmma`]: non-replay, non-capture,
/// `batch <= 16`, default ksplit policy (`HIPFIRE_RESIDUAL_KSPLIT_OFF` and
/// opt-in `HIPFIRE_RESIDUAL_LDSSTAGE` both veto). Anything else resolves to
/// [`Off`](DraftCollapseV2::Off) so the caller keeps today's path.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DraftCollapseV2 {
    Off,
    OverwriteKsplit { kw: u32 },
}

/// Mirror of the private `residual_ksplit_kw` K-split picker in gemm.rs:
/// `kw` waves for K/256 groups (`want` 4 below K=8192, else 8), falling back
/// down the [want, 4, 2] ladder. `None` routes to the base kernel.
fn draft_collapse_ksplit_kw(k: usize) -> Option<usize> {
    if k % 256 != 0 || k == 0 {
        return None;
    }
    let g = k / 256;
    let want = if k <= 8192 { 4 } else { 8 };
    [want, 4, 2]
        .into_iter()
        .filter(|&kw| kw <= want)
        .find(|&kw| g >= kw && g % kw == 0)
}

impl Gpu {
    /// S7 route check for one MQ4G256 draft GEMM (`m` rows, `k` cols, `batch` rows).
    ///
    /// Fast path requires: exact gfx1100, `HIPFIRE_DRAFT_COLLAPSE_OFF` unset,
    /// `batch > 1` (the scalar batch-1 path has no convert/fill to remove),
    /// `k % 256 == 0` (FWHT rotate granularity), no AWQ sidecar (draft
    /// artifacts never carry one; the AWQ divide needs the old kernel), and
    /// the default k2/ksplit_det variant policy.
    pub fn draft_collapse_mq4_route(
        &self,
        m: usize,
        k: usize,
        batch: usize,
        has_awq: bool,
    ) -> DraftCollapseGemm {
        if !self.arch_caps.is_gfx1100() {
            return DraftCollapseGemm::Off;
        }
        if self.flags.draft_collapse_off {
            return DraftCollapseGemm::Off;
        }
        if batch <= 1 {
            return DraftCollapseGemm::Off;
        }
        if has_awq {
            return DraftCollapseGemm::Off;
        }
        if k % 256 != 0 {
            return DraftCollapseGemm::Off;
        }
        if self.flags.mw16 || self.flags.hfq4g256_ldsstage_wmma {
            return DraftCollapseGemm::Off;
        }
        if self.flags.wo_wmma_variant.is_some() {
            return DraftCollapseGemm::Off;
        }
        // Mirror the auto selection: HIPFIRE_DETERMINISTIC=1 forces k2 for
        // every shape; otherwise the M=8192 threshold splits k2/ksplit_det.
        if self.flags.deterministic || m >= 8192 {
            DraftCollapseGemm::OverwriteK2
        } else {
            DraftCollapseGemm::OverwriteKsplitDet
        }
    }
    /// S7 route check for one MQ4G256V2 draft GEMM (`k` cols, `batch` rows).
    ///
    /// Mirrors the gfx1100 ksplit tier of `gemm_mq4g256v2_residual_wmma`:
    /// exact gfx1100, kill switch unset, no replay recording, no graph
    /// capture (capture keeps the base-kernel contract), `2 <= batch <= 16`,
    /// default ksplit policy, resolvable split width, no AWQ sidecar.
    pub fn draft_collapse_mq4v2_route(
        &self,
        k: usize,
        batch: usize,
        has_awq: bool,
    ) -> DraftCollapseV2 {
        if !self.arch_caps.is_gfx1100() || self.arch != "gfx1100" {
            return DraftCollapseV2::Off;
        }
        if self.flags.draft_collapse_off {
            return DraftCollapseV2::Off;
        }
        if self.replay.is_recording() || self.graphs.capture_mode {
            return DraftCollapseV2::Off;
        }
        if batch <= 1 || batch > 16 {
            return DraftCollapseV2::Off;
        }
        if has_awq {
            return DraftCollapseV2::Off;
        }
        if self.flags.residual_ksplit_off || self.flags.residual_ldsstage {
            return DraftCollapseV2::Off;
        }
        match draft_collapse_ksplit_kw(k) {
            Some(kw) if kw == 2 || kw == 4 || kw == 8 => {
                DraftCollapseV2::OverwriteKsplit { kw: kw as u32 }
            }
            _ => DraftCollapseV2::Off,
        }
    }

    /// Overwrite split-K LDS MQ4G256V2 GEMM: `y = W @ x_f16` (no residual,
    /// no pre-zero fill, no fp16-cache convert). `x_f16` is caller-owned F16
    /// ([batch, k]); `y` is F32 ([batch, m]). `kw` is 2, 4, or 8.
    pub fn gemm_mq4g256v2_overwrite_ksplit_lds_dflash(
        &mut self,
        a_raw: &GpuTensor,
        x_f16: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        kw: u32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let sym: &str = match kw {
            2 => "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks2",
            4 => "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks4",
            8 => "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks8",
            _ => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "gemm_mq4g256v2_overwrite_ksplit_lds_dflash: kw must be 2, 4, or 8",
                ));
            }
        };
        // One module per symbol (repo convention); shared collapse source.
        self.ensure_kernel(sym, COLLAPSE_SRC, sym)?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];
        let row_tiles = ((m + 15) / 16) as u32;
        let batch_tiles = ((batch_size + 15) / 16) as u32;
        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "mq4v2_overwrite_ksplit_dflash", bytes);
        let result = self.launch_maybe_blob(
            sym,
            [row_tiles, batch_tiles, 1],
            [32 * kw, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(bs_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// S7 master switch for the non-GEMM fusions (dual RMSNorm, finish
    /// conv+add, batched noise embeddings): exact gfx1100 with the kill
    /// switch unset. Shape/dtype predicates live at the call sites.
    pub fn draft_collapse_fused_enabled(&self) -> bool {
        self.arch_caps.is_gfx1100() && !self.flags.draft_collapse_off
    }

    /// FWHT-rotate F32 `x` ([batch, k]) directly to F16 `x_rot_f16`.
    ///
    /// Launch geometry mirrors `rotate_x_mq_batched` (one block of 32 per
    /// 256-group per row). Bit-identical to rotate-f32 + `convert_f32_to_f16`
    /// (same f32 expression tree, single rn conversion at the store).
    pub fn mq_rotate_x_f16_dflash(
        &mut self,
        x: &GpuTensor,
        x_rot_f16: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const SYM: &str = "mq_rotate_x_f16_dflash_gfx1100";
        self.ensure_kernel(SYM, COLLAPSE_SRC, SYM)?;
        self.ensure_mq_signs()?;
        let s1 = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let xp = x.buf.as_ptr();
        let xrp = x_rot_f16.buf.as_ptr();
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fwht", "mq_rotate_x_f16_dflash", bytes);
        let result = self.launch_maybe_blob(
            SYM,
            [((k / 256) * batch_size) as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xrp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Overwrite k2-schedule MQ4G256 GEMM: `y = W @ x_f16` (no residual, no
    /// pre-zero fill, no fp16-cache convert). `x_f16` is caller-owned F16
    /// ([batch, k]); `y` is F32 ([batch, m]).
    pub fn gemm_hfq4g256_overwrite_wmma_k2_dflash(
        &mut self,
        a_raw: &GpuTensor,
        x_f16: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const SYM: &str = "gemm_hfq4g256_overwrite_wmma_k2_dflash_gfx1100";
        self.ensure_kernel(SYM, COLLAPSE_SRC, SYM)?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];
        let row_tiles = ((m + 15) / 16) as u32;
        let batch_tiles = ((batch_size + 15) / 16) as u32;
        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", SYM, bytes);
        let result = self.launch_maybe_blob(
            SYM,
            [row_tiles, batch_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(bs_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Overwrite deterministic-ksplit MQ4G256 GEMM: phase 1 reuses the
    /// existing `gemm_hfq4g256_residual_wmma_ksplit_det` partial kernel
    /// (plain store, F16 X, no residual); phase 2 is the S7 overwrite
    /// finalize (`y = sum(partials)`, no residual load, no pre-zero fill).
    /// Partials scratch comes from the shared `ensure_ksplit_det_partials`
    /// pool (same lifetime contract as the residual path).
    pub fn gemm_hfq4g256_overwrite_ksplit_det_dflash(
        &mut self,
        a_raw: &GpuTensor,
        x_f16: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const K_SPLITS: u32 = 4;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_wmma_ksplit_det",
            crate::kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_KSPLIT_DET_SRC,
            "gemm_hfq4g256_residual_wmma_ksplit_det",
        )?;
        const FIN: &str = "gemm_ksplit_det_overwrite_finalize_dflash_gfx1100";
        self.ensure_kernel(FIN, COLLAPSE_SRC, FIN)?;
        // Partials scratch: [K_SPLITS][batch_size][M] fp32.
        let n_cells = batch_size * m;
        let partials_ptr = self.ensure_ksplit_det_partials(K_SPLITS as usize * n_cells * 4)?;

        // ── Phase 1: per-split partials (plain store, no atomic) ──
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16.buf.as_ptr();
        let mut p_ptr = partials_ptr;
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;
        let mut params1: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut p_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];
        let row_tiles = ((m + 15) / 16) as u32;
        let batch_tiles = ((batch_size + 15) / 16) as u32;
        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_overwrite_ksplit_det_dflash",
            bytes,
        );
        self.launch_maybe_blob(
            "gemm_hfq4g256_residual_wmma_ksplit_det",
            [row_tiles, batch_tiles, K_SPLITS],
            [32, 1, 1],
            0,
            &mut params1,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(p_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(bs_val);
                b
            },
        )?;

        // ── Phase 2: fixed-order overwrite finalize (partials → Y) ──
        let mut y_ptr = y.buf.as_ptr();
        let mut p_ptr2 = partials_ptr;
        let mut bs_val2 = batch_size as i32;
        let mut m_val2 = m as i32;
        let mut params2: Vec<*mut c_void> = vec![
            &mut y_ptr as *mut _ as *mut c_void,
            &mut p_ptr2 as *mut _ as *mut c_void,
            &mut bs_val2 as *mut _ as *mut c_void,
            &mut m_val2 as *mut _ as *mut c_void,
        ];
        let fin_grid = ((n_cells + 255) / 256) as u32;
        let r = self.launch_maybe_blob(
            FIN,
            [fin_grid, 1, 1],
            [256, 1, 1],
            0,
            &mut params2,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(y_ptr);
                b.push_ptr(p_ptr2);
                b.push_i32(bs_val2);
                b.push_i32(m_val2);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        r
    }

    /// Dual-output RMSNorm: `residual = x` (bitwise) + `out = rmsnorm(x)`.
    /// Same grid/block/shared config and accumulation order as
    /// `rmsnorm_batched`. `x` must not alias `residual` or `out`.
    pub fn rmsnorm_residual_dual_dflash(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        residual: &GpuTensor,
        out: &GpuTensor,
        batch: usize,
        n: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const SYM: &str = "rmsnorm_residual_dual_gfx1100";
        self.ensure_kernel(SYM, COLLAPSE_SRC, SYM)?;

        let mut x_ptr = x.buf.as_ptr();
        let mut w_ptr = weight.buf.as_ptr();
        let mut res_ptr = residual.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n as i32;
        let mut eps_val = eps;

        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut res_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut eps_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32.min(n as u32);
        let shared_mem = block_size * 4;
        let bytes = crate::profile::rmsnorm_bytes(batch * n);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", SYM, bytes);
        let result = self.launch_maybe_blob(
            SYM,
            [batch as u32, 1, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_ptr(w_ptr);
                b.push_ptr(res_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(n_val);
                b.push_f32(eps_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Fused DFlash2 finish conv + residual add:
    /// `out = residual + dynconv(input)`. Same grid/block as
    /// `dynamic_causal_conv_f32`. `input`, `residual`, `output` must be
    /// pairwise distinct buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn dynamic_conv_residual_dflash(
        &mut self,
        input: &GpuTensor,
        base: &GpuTensor,
        dynamic: &GpuTensor,
        residual: &GpuTensor,
        output: &GpuTensor,
        rows: usize,
        hidden: usize,
        kernel_size: usize,
        group_size: usize,
        dynamic_row_stride: usize,
        dynamic_offset: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if rows == 0 || hidden == 0 || kernel_size == 0 || group_size == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "dynamic_conv_residual_dflash: rows/hidden/kernel_size/group_size must be > 0",
            ));
        }
        if hidden % group_size != 0 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "dynamic_conv_residual_dflash: hidden {hidden} must be divisible by group_size {group_size}"
                ),
            ));
        }
        let groups = hidden / group_size;
        for (name, t) in [
            ("input", input),
            ("base", base),
            ("dynamic", dynamic),
            ("residual", residual),
            ("output", output),
        ] {
            if t.dtype != crate::DType::F32 {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "dynamic_conv_residual_dflash: {name} dtype must be F32 (got {:?})",
                        t.dtype
                    ),
                ));
            }
        }
        const SYM: &str = "dynamic_conv_residual_gfx1100";
        self.ensure_kernel(SYM, COLLAPSE_SRC, SYM)?;
        let input_ptr = input.buf.as_ptr();
        let base_ptr = base.buf.as_ptr();
        let dynamic_ptr = dynamic.buf.as_ptr();
        let residual_ptr = residual.buf.as_ptr();
        let output_ptr = output.buf.as_ptr();
        let rows_i32 = rows as i32;
        let hidden_i32 = hidden as i32;
        let kernel_size_i32 = kernel_size as i32;
        let groups_i32 = groups as i32;
        let group_size_i32 = group_size as i32;
        let stride_i32 = dynamic_row_stride as i32;
        let offset_i32 = dynamic_offset as i32;
        let total = rows.checked_mul(hidden).unwrap();
        let block = 256u32;
        let grid = total.div_ceil(block as usize) as u32;
        let mut params: Vec<*mut c_void> = vec![
            &input_ptr as *const _ as *mut c_void,
            &base_ptr as *const _ as *mut c_void,
            &dynamic_ptr as *const _ as *mut c_void,
            &residual_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &rows_i32 as *const _ as *mut c_void,
            &hidden_i32 as *const _ as *mut c_void,
            &kernel_size_i32 as *const _ as *mut c_void,
            &groups_i32 as *const _ as *mut c_void,
            &group_size_i32 as *const _ as *mut c_void,
            &stride_i32 as *const _ as *mut c_void,
            &offset_i32 as *const _ as *mut c_void,
        ];
        let bytes = total * 4 * 2 + base.buf.size() + dynamic.buf.size();
        let timer = crate::profile::begin_timer(&self.hip, "dynamic_conv", SYM, bytes);
        let result =
            self.launch_maybe_blob(SYM, [grid, 1, 1], [block, 1, 1], 0, &mut params, || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(input_ptr);
                blob.push_ptr(base_ptr);
                blob.push_ptr(dynamic_ptr);
                blob.push_ptr(residual_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(rows_i32);
                blob.push_i32(hidden_i32);
                blob.push_i32(kernel_size_i32);
                blob.push_i32(groups_i32);
                blob.push_i32(group_size_i32);
                blob.push_i32(stride_i32);
                blob.push_i32(offset_i32);
                blob
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
