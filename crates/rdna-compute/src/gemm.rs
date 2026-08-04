// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched GEMM prefill methods for RDNA GPUs.

use crate::dispatch::{
    DType, Gpu, GpuTensor, FP8_WMMA_MIN_BATCH, LLOYD_MQ3_GROUP_BYTES, LLOYD_MQ4_GROUP_BYTES,
};
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipResult};
use std::ffi::c_void;
use std::sync::OnceLock;

/// Batch ceilings for the LDS-staged HFQ4-G256 GEMMs (`HIPFIRE_HFQ4G256_LDSSTAGE=1`).
/// The staged kernels win while the grid is small and lose once the added LDS
/// traffic plus the `__launch_bounds__(256,4)` occupancy cap outweigh the
/// coalesced loads. Every value below is MEASURED in-engine (`hipfire bench
/// --matrix`, matched A/B/B/A, runs 5, warmups 10, dense 27B MQ4), 2026-07-29/30:
///
///   gfx1201 (R9700, 64 CU, pinned 3.6-27B `86a5f80f…`):
///     B16 1.0902  B32 1.0489  B64 1.0300  B72 1.0914  B80 1.1079
///     B96 1.0409  B112 0.9977  B128 0.9096  B512 0.9465  B1024 0.9492
///   gfx1100 (7900 XTX, 96 CU, 3.5-27B):
///     B16 1.1044  B32 1.1324  B64 1.1719  B96 1.1786   (>=128 unreachable, see below)
///   gfx1151 (Strix Halo, 40 CU, 3.5-27B):
///     B16 1.1132  B24 1.0487  B32 1.0383  B48 1.0023  B64 0.9915
///
/// Turn points are NOT a function of CU count (gfx1201 turns at 96-112 with 64 CU,
/// gfx1151 at 32-48 with 40 CU) — an earlier CU-count reading was an artifact of a
/// coarse sweep grid. They are also NOT portable from a small model: a 0.8b canary
/// measured +11..+23% at batch 512-8192 where 27B measures -3.5..-5.4%
/// (.research/round5-verify-cost.md §17). Measure per arch on a real model.
///
/// On gfx11 the staged path is only REACHABLE below the MMQ cutoff: `gemm.rs`
/// routes `arch_caps.should_use_mmq(batch_size)` (128 on RDNA3/3.5) to the MMQ
/// kernel before the gfx11 WMMA branch, so batch >= 128 never reaches these
/// kernels there. gfx12 has no such diversion, which is why only gfx12 shows the
/// large-batch regression.
const LDSSTAGE_MAX_BATCH: usize = 96;

/// gfx11 (RDNA3 / RDNA3.5), both dGPU and iGPU. Re-measured on the PINNED 3.6
/// trunk `86a5f80f…` (the fixture rule: measure 3.6 when it exists), which
/// reproduced the earlier 3.5 numbers within +/-0.7 percentage points at every
/// shared batch on both parts:
///
///   gfx1100 (7900 XTX, 96 CU): B16 1.0992  B24 1.1150  B32 1.1355
///                              B48 1.1792  B64 1.1670  B96 1.1754   (all WIN)
///   gfx1151 (Strix Halo, 40 CU): B16 1.1155  B24 1.0449  B32 1.0452
///                              B48 1.0054  B64 0.9947  B96 1.0399
///
/// The iGPU has a wash band at 48-64 (worst case -0.5%, inside the 0.87-1.25%
/// run-to-run spread) and then wins again at 96 — reproducible on BOTH model
/// files. That is the non-staged kernel dipping at 96 (its OFF arm drops
/// 196.7 -> 190.3 tok/s while the staged arm stays flat at ~198), not the staged
/// kernel improving. One ceiling of 96 therefore serves both parts: it captures
/// every measured win and rides a sub-spread wash on the iGPU's 48-64 band. An
/// earlier dGPU/iGPU split (96 / 32) was collapsed once the iGPU's 96 win
/// reproduced on the correct fixture — the split bought nothing.
///
/// Batches >= 128 never reach these kernels on gfx11: `should_use_mmq` (cutoff
/// 128) diverts to MMQ before the gfx11 WMMA branch, so the whole reachable
/// domain is covered.
const LDSSTAGE_MAX_BATCH_GFX11: usize = 96;

impl Gpu {
    /// CDNA3-only: prefill GEMM used by `gemm_hfq4g256` rocBLAS path.
    ///
    /// Computes Y_rowmajor[N × M] = X_rowmajor[N × K] · W_transposed, where
    /// the weight is stored row-major [M × K] but the operation needs W^T.
    /// This matches the engine's convention (weight dotted with each row of X
    /// produces one output column per batch row).
    ///
    /// rocBLAS is column-major. A row-major [M × K] matrix is byte-identical
    /// to a column-major [K × M] matrix. So the call is:
    ///   col-major C[M × N] = op_A(W) · X_col[K × N]
    /// with op_A = T (transpose the col-major [K × M] view of W to get [M × K]).
    /// X_row[N × K] viewed col-major is [K × N] with ld=K. Y_row[N × M] viewed
    /// col-major is [M × N] with ld=M — so pointer+ld match C directly.
    pub fn rocblas_gemm_hfq4_prefill(
        &self,
        w_fp16: &DeviceBuffer, // row-major [M × K]
        x_fp16: &DeviceBuffer, // row-major [N × K]
        y_fp32: &DeviceBuffer, // row-major [N × M]
        m: usize,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rocblas_gemm_hfq4_generic(w_fp16, x_fp16, y_fp32, m, n, k, 1.0, 0.0)
    }

    /// gfx942 DeepSeek4 qt=35 dense-prefill path.
    ///
    /// The first call expands the frozen MFP4G32E8SOA weight into an FP16
    /// model-lifetime shadow; subsequent calls reuse that shadow and execute a
    /// native rocBLAS MFMA GEMM. Returns `false` when the exact DS4 MQ2R /
    /// CDNA / shape / batch contract is not met, so the architecture owner
    /// can retain its compressed fallback without exposing this route to
    /// other models. Discovery is opt-in with
    /// `HIPFIRE_DEEPSEEK4_GFX942_E8_ROCBLAS=1` until the production-shape
    /// numerical oracle and model coherence gate pass.
    pub fn rocblas_gemm_mfp4e8_soa_prefill_auto(
        &mut self,
        weight: &GpuTensor,
        x_rotated: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<bool> {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        static ADMISSION_LOGGED: OnceLock<()> = OnceLock::new();
        let enabled = *ENABLED.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_GFX942_E8_ROCBLAS").as_deref()
                == Ok("1")
        });
        if self.arch != "gfx942"
            || !enabled
            || weight.dtype != DType::MFP4G32E8SOA
            || batch_size < self.rocblas_min_batch()
            || k % 256 != 0
            || self.rocblas.is_none()
            || self.graphs.capture_mode
        {
            return Ok(false);
        }
        // DeepSeek creates its explicit prefill stream after `Gpu` initializes
        // rocBLAS. Rebind here so the dequant/activation conversions and GEMM
        // share one ordered stream; otherwise rocBLAS remains on the null
        // stream and can race both its inputs and downstream consumers.
        if let (Some(rb), Some(stream)) = (self.rocblas.as_ref(), self.active_stream.as_ref()) {
            rb.set_stream(stream).map_err(|e| {
                hip_bridge::HipError::new(
                    e.status,
                    &format!("rocblas_set_stream for DS4 gfx942 prefill: {}", e.context),
                )
            })?;
        }
        ADMISSION_LOGGED.get_or_init(|| {
            eprintln!("deepseek4 gfx942: admitted qt35 FP16-shadow + rocBLAS prefill route");
        });
        let Some(shadow_ptr) = self.ensure_mfp4e8_soa_fp16_shadow_gfx942(weight, m, k)? else {
            return Ok(false);
        };
        // DS4 rewrites stable PBS allocations between projections and layers.
        // Pointer-keyed `ensure_fp16_x` would therefore reuse stale contents.
        // Convert on every call until a content-generation-aware cache exists.
        let x_fp16 = self.convert_fp16_x_uncached(x_rotated, batch_size * k)?;
        // The uncached conversion overwrote the shared FP16 scratch. Prevent a
        // later generic `ensure_fp16_x` caller from treating its old source
        // marker as a valid cache hit. Keep this repair DS4-local so Qwen's
        // generic conversion policy is unchanged.
        self.scratch.fp16_x_source_ptr = std::ptr::null_mut();
        let shadow = unsafe { DeviceBuffer::from_raw(shadow_ptr, m * k * 2) };
        let x = unsafe { DeviceBuffer::from_raw(x_fp16, batch_size * k * 2) };
        // The timer begins after staging, so report the actual hot rocBLAS
        // traffic: expanded FP16 W + staged FP16 X + FP32 Y.
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_mfp4g32_e8_soa_rocblas_gfx942",
            bytes,
        );
        let result = self.rocblas_gemm_hfq4_prefill(&shadow, &x, &y.buf, m, batch_size, k);
        if let Some(timer) = timer {
            timer.finish(&self.hip);
        }
        result?;
        Ok(true)
    }

    /// Same op as `rocblas_gemm_hfq4_prefill` but with Y += alpha·(X·W^T) +
    /// beta·Y. Covers the residual-GEMM pattern (w_down on LA path, wo on
    /// attention path) where the existing hand-rolled kernels fuse the add.
    pub fn rocblas_gemm_hfq4_prefill_residual(
        &self,
        w_fp16: &DeviceBuffer,
        x_fp16: &DeviceBuffer,
        y_fp32: &DeviceBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.rocblas_gemm_hfq4_generic(w_fp16, x_fp16, y_fp32, m, n, k, 1.0, 1.0)
    }

    fn rocblas_gemm_hfq4_generic(
        &self,
        w_fp16: &DeviceBuffer,
        x_fp16: &DeviceBuffer,
        y_fp32: &DeviceBuffer,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> HipResult<()> {
        use hip_bridge::{RocblasDatatype, RocblasOperation};
        let rb = self
            .rocblas
            .as_ref()
            .expect("rocblas_gemm_hfq4: rocBLAS not initialized");
        unsafe {
            rb.gemm_ex(
                RocblasOperation::Transpose,
                RocblasOperation::None,
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const f32 as *const c_void,
                w_fp16.as_ptr(),
                RocblasDatatype::F16,
                k as i32,
                x_fp16.as_ptr(),
                RocblasDatatype::F16,
                k as i32,
                &beta as *const f32 as *const c_void,
                y_fp32.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                y_fp32.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                RocblasDatatype::F32,
            )
            .map_err(|e| {
                hip_bridge::HipError::new(e.status, &format!("rocblas_gemm: {}", e.context))
            })
        }
    }

    // ── hipGraph capture/replay ───────────────────────────────────────────
    // Moved to crate::graph::GraphState.

    /// Batched HFQ4-G128 GEMM. Same tiled approach as G256.
    ///
    /// gfx1151 i8 MMQ fast-path (default ON; opt out via HIPFIRE_HFQ4G128_MMQ=0):
    /// when batch_size and M are 16-tile aligned, pre-quantize X to Q8_1 and
    /// route to `gemm_hfq4g128_mmq_gfx1151`. Closes the rocprof finding that
    /// this kernel was 66% of pp256 prefill on A3B-PARO; A/B median +129.5%
    /// (427 → 980 tok/s pp256) on shisa-Qwen3.6-35B-A3B-PARO. Mirror of the
    /// routed-MoE MMQ k8 default-on flip at 949f51db.
    pub fn gemm_hfq4g128(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let use_mmq = self.arch.starts_with("gfx1151")
            && self.flags.hfq4g128_mmq
            && batch_size >= 16
            && batch_size % 16 == 0
            && m % 16 == 0
            && k % 128 == 0;
        if use_mmq {
            return self.gemm_hfq4g128_mmq_gfx1151(a_raw, x, y, m, k, batch_size);
        }
        self.ensure_kernel("gemm_hfq4g128", kernels::GEMM_HFQ4G128_SRC, "gemm_hfq4g128")?;
        let func = &self.functions["gemm_hfq4g128"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
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
        let batch_tiles = ((batch_size + 7) / 8) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// gfx1151 i8 MMQ dispatch helper for HFQ4-G128. Pre-quantizes X to
    /// Q8_1 mmq DS4 then launches `gemm_hfq4g128_mmq_gfx1151`. Caller must
    /// have already verified the alignment constraints.
    fn gemm_hfq4g128_mmq_gfx1151(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        self.ensure_kernel(
            "gemm_hfq4g128_mmq_gfx1151",
            kernels::GEMM_HFQ4G128_MMQ_GFX1151_SRC,
            "gemm_hfq4g128_mmq_gfx1151",
        )?;
        let a_ptr = a_raw.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let n_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_q8_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
        ];
        let bytes = (m * k / 2)               // HFQ4 weight (4 bits / elem)
                  + (batch_size * k * 4 / 3)  // Q8_1 activation (approx, includes ds4 headers)
                  + (batch_size * m * 4); // F32 output
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g128_mmq_gfx1151", bytes);
        let result = self.launch_maybe_blob(
            "gemm_hfq4g128_mmq_gfx1151",
            [(m / 16) as u32, (batch_size / 16) as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_q8_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ2-G256 GEMV. K must be multiple of 256.

    /// MQ2-Lloyd GEMV (2-bit + per-block 4-entry fp16 codebook). K must be a
    /// multiple of 256. Same launch shape as gemv_hfq2g256 — header is the
    /// only layout difference.

    /// MQ2-Lloyd GEMV with engine-side x rotation (matches `gemv_mq2g256_with_rotate`).

    /// MQ3-Lloyd GEMV (3-bit + per-block 8-entry fp16 codebook). K must be a
    /// multiple of 256. gfx1100/1101/1102 use the K4-unrolled + LDS-codebook
    /// variant; other archs fall back to the baseline switch-dispatch path.

    /// MQ3-Lloyd GEMV with engine-side x rotation.

    /// MQ4-Lloyd GEMV (4-bit + per-block 16-entry fp16 codebook). K must be a
    /// multiple of 256. gfx1100/1101/1102/1151 use the K4-unrolled + LDS-codebook
    /// variant (cooperative double-load for the 64-entry table). Other archs
    /// fall back to the chip-agnostic baseline switch-dispatch path.

    /// MQ4-Lloyd GEMV with engine-side x rotation.

    /// DIAGNOSTIC ONLY: K4 multi-accumulator MQ4-Lloyd GEMV. NOT for production.
    /// Used by examples/diag_mq4_lloyd_multiacc.rs to compare against the slow
    /// generic kernel on real model rows. See the kernel header for the
    /// open question this exists to investigate.

    /// MQ4-Lloyd WMMA-accelerated batched residual GEMM (Phase 5b / issue #182,
    /// Phase B1). Mirrors gemm_mq3g256_lloyd_residual_wmma's wiring, with 160 B/
    /// group + 16-entry codebook + nibble-pair decode. fp16-LDS staging — fp16
    /// won the MQ3 Phase A bench by 7.15% (decision inherited).
    /// gfx11/gfx12 wave32 WMMA; other archs fall through to baseline (which
    /// itself currently requires WMMA — caller must check arch first).
    pub fn gemm_mq4g256_lloyd_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — path selector; concrete launch path binds before HIP use
        // Phase D-A path selector: route to `_mb4` (16×64 output tile, 4× weight
        // reuse per WG) when shape clears the size gate. Bench (gfx1151,
        // benchmarks/results/devlog_20260509_mq4_lloyd_gfx1151_bench.md):
        // 1.40-2.24× speedup at production shapes (M ≥ 4096, batch ≥ 128);
        // small shapes regress (4× WG reduction + 106 VGPR leaves CUs idle).
        // Threshold tuning open — see Phase D plan §"Open questions" #3.
        // Env override: HIPFIRE_LLOYD_MB4=1 force-on, =0 force-off.
        let arch_supports_mb4 = self.arch_caps.supports_mq4_lloyd_mb4();
        let use_mb4 = match self.flags.lloyd_mb4 {
            Some(_) => arch_supports_mb4,
            None => arch_supports_mb4 && batch_size >= 128 && m >= 4096,
        };
        if use_mb4 {
            return self.gemm_mq4g256_lloyd_residual_wmma_mb4(a_raw, x, y, m, k, batch_size);
        }
        self.bind_thread()?;
        let (src, module) = kernels::gemm_mq4g256_lloyd_residual_wmma_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_mq4g256_lloyd_residual_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = m * (k / 256) * LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_mq4g256_lloyd_residual_wmma",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_mq4g256_lloyd_residual_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
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

    /// MQ4-Lloyd WMMA residual GEMM, 4× batch-tile fanout per WG (Phase D-A).
    /// Same args as `gemm_mq4g256_lloyd_residual_wmma`; only the grid shape and
    /// per-WG output tile differ (16×64 vs 16×16).
    ///
    /// Caller is responsible for the path-selection gate. This kernel is shipped
    /// dead-code-safe: parity test wires it directly; production matcher routing
    /// lands in Phase D-C.
    pub fn gemm_mq4g256_lloyd_residual_wmma_mb4(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_mq4g256_lloyd_residual_wmma_mb4_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_mq4g256_lloyd_residual_wmma_mb4")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 63) / 64;

        let weight_bytes = m * (k / 256) * LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_mq4g256_lloyd_residual_wmma_mb4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_mq4g256_lloyd_residual_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
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

    /// Phase D experiment: 16×32 fanout sibling of `gemm_mq4g256_lloyd_residual_wmma`.
    /// Half the per-WG weight reuse of mb4 but 2× the WG count and ~21 fewer
    /// VGPRs — targets the small-M residual case where mb4 is occupancy-bound.
    pub fn gemm_mq4g256_lloyd_residual_wmma_mb2(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_mq4g256_lloyd_residual_wmma_mb2_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_mq4g256_lloyd_residual_wmma_mb2")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 31) / 32;

        let weight_bytes = m * (k / 256) * LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_mq4g256_lloyd_residual_wmma_mb2",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_mq4g256_lloyd_residual_wmma_mb2",
            [row_tiles as u32, batch_tiles as u32, 1],
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

    /// MQ4-Lloyd WMMA fused QKVZA GEMM (LA preamble: qkv + z + beta + alpha).
    /// 4-way fused — one launch covers all four projections of the LA layer.
    /// Phase B1 sibling of `gemm_mq4g256_lloyd_residual_wmma` (kernels-only,
    /// dead-code-safe — wired via the consolidated parity test only; matcher
    /// updates land together with corruption-prevention in Phase B2).
    pub fn gemm_qkvza_mq4g256_lloyd_wmma(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — path selector; concrete launch path binds before HIP use
        // Phase D-B path selector — same gate as residual_mb4.
        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let arch_supports_mb4 = self.arch_caps.supports_mq4_lloyd_mb4();
        let use_mb4 = match self.flags.lloyd_mb4 {
            None => arch_supports_mb4 && n >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_qkvza_mq4g256_lloyd_wmma_mb4(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, n,
            );
        }
        self.bind_thread()?;
        let (src, module) = kernels::gemm_qkvza_mq4g256_lloyd_wmma_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_qkvza_mq4g256_lloyd_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_qkv_p = a_qkv.buf.as_ptr();
        let mut a_z_p = a_z.buf.as_ptr();
        let mut a_beta_p = a_beta.buf.as_ptr();
        let mut a_alpha_p = a_alpha.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_qkv_p = y_qkv.buf.as_ptr();
        let mut y_z_p = y_z.buf.as_ptr();
        let mut y_beta_p = y_beta.buf.as_ptr();
        let mut y_alpha_p = y_alpha.buf.as_ptr();
        let mut qkv_m_v = qkv_m as i32;
        let mut z_m_v = z_m as i32;
        let mut beta_m_v = beta_m as i32;
        let mut alpha_m_v = alpha_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_qkv_p as *mut _ as *mut c_void,
            &mut a_z_p as *mut _ as *mut c_void,
            &mut a_beta_p as *mut _ as *mut c_void,
            &mut a_alpha_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_qkv_p as *mut _ as *mut c_void,
            &mut y_z_p as *mut _ as *mut c_void,
            &mut y_beta_p as *mut _ as *mut c_void,
            &mut y_alpha_p as *mut _ as *mut c_void,
            &mut qkv_m_v as *mut _ as *mut c_void,
            &mut z_m_v as *mut _ as *mut c_void,
            &mut beta_m_v as *mut _ as *mut c_void,
            &mut alpha_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 15) / 16;
        let weight_bytes = total_m * (k / 256) * LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_mq4g256_lloyd_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_mq4g256_lloyd_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_qkv_p);
                b.push_ptr(a_z_p);
                b.push_ptr(a_beta_p);
                b.push_ptr(a_alpha_p);
                b.push_ptr(x_p);
                b.push_ptr(y_qkv_p);
                b.push_ptr(y_z_p);
                b.push_ptr(y_beta_p);
                b.push_ptr(y_alpha_p);
                b.push_i32(qkv_m_v);
                b.push_i32(z_m_v);
                b.push_i32(beta_m_v);
                b.push_i32(alpha_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ4-Lloyd WMMA fused QKV GEMM (FullAttention preamble: q + k + v).
    /// 3-way fused. Phase B1 sibling.
    pub fn gemm_qkv_mq4g256_lloyd_wmma(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — path selector; concrete launch path binds before HIP use
        let total_m = q_m + k_m + v_m;
        let arch_supports_mb4 = self.arch_caps.supports_mq4_lloyd_mb4();
        let use_mb4 = match self.flags.lloyd_mb4 {
            None => arch_supports_mb4 && n >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_qkv_mq4g256_lloyd_wmma_mb4(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, n,
            );
        }
        self.bind_thread()?;
        let (src, module) = kernels::gemm_qkv_mq4g256_lloyd_wmma_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_qkv_mq4g256_lloyd_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_q_p = a_q.buf.as_ptr();
        let mut a_k_p = a_k.buf.as_ptr();
        let mut a_v_p = a_v.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_q_p = y_q.buf.as_ptr();
        let mut y_k_p = y_k.buf.as_ptr();
        let mut y_v_p = y_v.buf.as_ptr();
        let mut q_m_v = q_m as i32;
        let mut k_m_v = k_m as i32;
        let mut v_m_v = v_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_q_p as *mut _ as *mut c_void,
            &mut a_k_p as *mut _ as *mut c_void,
            &mut a_v_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_q_p as *mut _ as *mut c_void,
            &mut y_k_p as *mut _ as *mut c_void,
            &mut y_v_p as *mut _ as *mut c_void,
            &mut q_m_v as *mut _ as *mut c_void,
            &mut k_m_v as *mut _ as *mut c_void,
            &mut v_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 15) / 16;
        let weight_bytes = total_m * (k / 256) * LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_mq4g256_lloyd_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_mq4g256_lloyd_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_q_p);
                b.push_ptr(a_k_p);
                b.push_ptr(a_v_p);
                b.push_ptr(x_p);
                b.push_ptr(y_q_p);
                b.push_ptr(y_k_p);
                b.push_ptr(y_v_p);
                b.push_i32(q_m_v);
                b.push_i32(k_m_v);
                b.push_i32(v_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ4-Lloyd WMMA fused gate+up GEMM (FFN preamble). 2-way fused.
    /// Phase B1 sibling.
    pub fn gemm_gate_up_mq4g256_lloyd_wmma(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — path selector; concrete launch path binds before HIP use
        let total_m = gate_m + up_m;
        let arch_supports_mb4 = self.arch_caps.supports_mq4_lloyd_mb4();
        let use_mb4 = match self.flags.lloyd_mb4 {
            None => arch_supports_mb4 && n >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_gate_up_mq4g256_lloyd_wmma_mb4(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, n,
            );
        }
        self.bind_thread()?;
        let (src, module, func_name) = if self.flags.gate_up_nosync {
            let (src, module) =
                kernels::gemm_gate_up_mq4g256_lloyd_wmma_nosync_for_arch(&self.arch_caps);
            (src, module, "gemm_gate_up_mq4g256_lloyd_wmma_nosync")
        } else {
            let (src, module) = kernels::gemm_gate_up_mq4g256_lloyd_wmma_for_arch(&self.arch_caps);
            (src, module, "gemm_gate_up_mq4g256_lloyd_wmma")
        };
        self.ensure_kernel(module, src, func_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_gate_p = a_gate.buf.as_ptr();
        let mut a_up_p = a_up.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_gate_p = y_gate.buf.as_ptr();
        let mut y_up_p = y_up.buf.as_ptr();
        let mut gate_m_v = gate_m as i32;
        let mut up_m_v = up_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_gate_p as *mut _ as *mut c_void,
            &mut a_up_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_gate_p as *mut _ as *mut c_void,
            &mut y_up_p as *mut _ as *mut c_void,
            &mut gate_m_v as *mut _ as *mut c_void,
            &mut up_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 15) / 16;
        let weight_bytes = total_m * (k / 256) * LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", func_name, bytes);
        let result = self.launch_maybe_blob(
            func_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_gate_p);
                b.push_ptr(a_up_p);
                b.push_ptr(x_p);
                b.push_ptr(y_gate_p);
                b.push_ptr(y_up_p);
                b.push_i32(gate_m_v);
                b.push_i32(up_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Phase D-B: 16×64 fanout sibling of `gemm_qkvza_mq4g256_lloyd_wmma`.
    pub fn gemm_qkvza_mq4g256_lloyd_wmma_mb4(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_qkvza_mq4g256_lloyd_wmma_mb4_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_qkvza_mq4g256_lloyd_wmma_mb4")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_qkv_p = a_qkv.buf.as_ptr();
        let mut a_z_p = a_z.buf.as_ptr();
        let mut a_beta_p = a_beta.buf.as_ptr();
        let mut a_alpha_p = a_alpha.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_qkv_p = y_qkv.buf.as_ptr();
        let mut y_z_p = y_z.buf.as_ptr();
        let mut y_beta_p = y_beta.buf.as_ptr();
        let mut y_alpha_p = y_alpha.buf.as_ptr();
        let mut qkv_m_v = qkv_m as i32;
        let mut z_m_v = z_m as i32;
        let mut beta_m_v = beta_m as i32;
        let mut alpha_m_v = alpha_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_qkv_p as *mut _ as *mut c_void,
            &mut a_z_p as *mut _ as *mut c_void,
            &mut a_beta_p as *mut _ as *mut c_void,
            &mut a_alpha_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_qkv_p as *mut _ as *mut c_void,
            &mut y_z_p as *mut _ as *mut c_void,
            &mut y_beta_p as *mut _ as *mut c_void,
            &mut y_alpha_p as *mut _ as *mut c_void,
            &mut qkv_m_v as *mut _ as *mut c_void,
            &mut z_m_v as *mut _ as *mut c_void,
            &mut beta_m_v as *mut _ as *mut c_void,
            &mut alpha_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 63) / 64;
        let weight_bytes = total_m * (k / 256) * LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkvza_mq4g256_lloyd_wmma_mb4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_qkvza_mq4g256_lloyd_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_qkv_p);
                b.push_ptr(a_z_p);
                b.push_ptr(a_beta_p);
                b.push_ptr(a_alpha_p);
                b.push_ptr(x_p);
                b.push_ptr(y_qkv_p);
                b.push_ptr(y_z_p);
                b.push_ptr(y_beta_p);
                b.push_ptr(y_alpha_p);
                b.push_i32(qkv_m_v);
                b.push_i32(z_m_v);
                b.push_i32(beta_m_v);
                b.push_i32(alpha_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Phase D-B: 16×64 fanout sibling of `gemm_qkv_mq4g256_lloyd_wmma`.
    pub fn gemm_qkv_mq4g256_lloyd_wmma_mb4(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_qkv_mq4g256_lloyd_wmma_mb4_for_arch(
            &self.arch_caps,
            self.flags.lloyd_force_baseline,
        );
        self.ensure_kernel(module, src, "gemm_qkv_mq4g256_lloyd_wmma_mb4")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_q_p = a_q.buf.as_ptr();
        let mut a_k_p = a_k.buf.as_ptr();
        let mut a_v_p = a_v.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_q_p = y_q.buf.as_ptr();
        let mut y_k_p = y_k.buf.as_ptr();
        let mut y_v_p = y_v.buf.as_ptr();
        let mut q_m_v = q_m as i32;
        let mut k_m_v = k_m as i32;
        let mut v_m_v = v_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_q_p as *mut _ as *mut c_void,
            &mut a_k_p as *mut _ as *mut c_void,
            &mut a_v_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_q_p as *mut _ as *mut c_void,
            &mut y_k_p as *mut _ as *mut c_void,
            &mut y_v_p as *mut _ as *mut c_void,
            &mut q_m_v as *mut _ as *mut c_void,
            &mut k_m_v as *mut _ as *mut c_void,
            &mut v_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 63) / 64;
        let weight_bytes = total_m * (k / 256) * LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkv_mq4g256_lloyd_wmma_mb4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_qkv_mq4g256_lloyd_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_q_p);
                b.push_ptr(a_k_p);
                b.push_ptr(a_v_p);
                b.push_ptr(x_p);
                b.push_ptr(y_q_p);
                b.push_ptr(y_k_p);
                b.push_ptr(y_v_p);
                b.push_i32(q_m_v);
                b.push_i32(k_m_v);
                b.push_i32(v_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Phase D-B: 16×64 fanout sibling of `gemm_gate_up_mq4g256_lloyd_wmma`.
    pub fn gemm_gate_up_mq4g256_lloyd_wmma_mb4(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module, func_name) = if self.flags.gate_up_nosync {
            let (src, module) =
                kernels::gemm_gate_up_mq4g256_lloyd_wmma_mb4_nosync_for_arch(&self.arch_caps);
            (src, module, "gemm_gate_up_mq4g256_lloyd_wmma_mb4_nosync")
        } else {
            let (src, module) = kernels::gemm_gate_up_mq4g256_lloyd_wmma_mb4_for_arch(
                &self.arch_caps,
                self.flags.lloyd_force_baseline,
            );
            (src, module, "gemm_gate_up_mq4g256_lloyd_wmma_mb4")
        };
        self.ensure_kernel(module, src, func_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_gate_p = a_gate.buf.as_ptr();
        let mut a_up_p = a_up.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_gate_p = y_gate.buf.as_ptr();
        let mut y_up_p = y_up.buf.as_ptr();
        let mut gate_m_v = gate_m as i32;
        let mut up_m_v = up_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_gate_p as *mut _ as *mut c_void,
            &mut a_up_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_gate_p as *mut _ as *mut c_void,
            &mut y_up_p as *mut _ as *mut c_void,
            &mut gate_m_v as *mut _ as *mut c_void,
            &mut up_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 63) / 64;
        let weight_bytes = total_m * (k / 256) * LLOYD_MQ4_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", func_name, bytes);
        let result = self.launch_maybe_blob(
            func_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_gate_p);
                b.push_ptr(a_up_p);
                b.push_ptr(x_p);
                b.push_ptr(y_gate_p);
                b.push_ptr(y_up_p);
                b.push_i32(gate_m_v);
                b.push_i32(up_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ4-Lloyd GEMV with fused residual add: y[row] += A[row] · x. Mirrors
    /// gemv_mq3g256_lloyd_residual; same single-acc bug fix applies.

    /// MQ4-Lloyd residual GEMV with engine-side x rotation.

    /// Fused Gate+Up MQ4-Lloyd: two GEMVs in one launch. Mirrors
    /// fused_gate_up_mq3g256_lloyd. Caller is responsible for pre-rotating x.

    /// Fused QKVZA MQ4-Lloyd: 4 LA-preamble GEMVs in one launch.

    /// Fused QKV MQ4-Lloyd: 3 FA-preamble GEMVs in one launch.

    /// MQ3-Lloyd GEMV with fused residual add: y[row] += A[row] · x. Used by
    /// `weight_gemv_residual` MQ3-Lloyd arm to eliminate the alloc + gemv +
    /// add_inplace_f32 + free fallback chain (saves ~4.4% of decode time on
    /// 9B Lloyd-MQ3, gfx1100, per the 2026-05-06 decode profile).

    /// MQ3-Lloyd residual GEMV with engine-side x rotation.

    /// MQ3-Lloyd WMMA residual GEMM (Phase 5 / issue #116, Phase B1).
    /// Mirrors `gemm_hfq3g256_residual_wmma` shape + grid; group stride is 112 B
    /// (16 B fp16 codebook + 96 B 3-bit indices) instead of HFQ3's 104. K must
    /// be a multiple of 256. gfx11/gfx12 wave32 WMMA; other archs fall through
    /// to the baseline kernel (which itself currently requires WMMA — caller
    /// must check arch before dispatching).
    /// Caller is responsible for pre-rotating X (FWHT) for the MQ3-Lloyd dtype;
    /// this dispatch mirrors `gemm_hfq3g256_residual_wmma` and does not rotate.
    /// fp16-LDS staging — fp16 won the Phase A bench by 7.15% (devlog
    /// 2026-05-07).
    pub fn gemm_mq3g256_lloyd_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // mb4 path selector — same gate as MQ4-Lloyd's mb4 family.
        let arch_supports_mb4 = self.arch_caps.supports_mq3_lloyd_mb4();
        let use_mb4 = match self.flags.mq3_mb4 {
            Some(_) => arch_supports_mb4,
            None => arch_supports_mb4 && batch_size >= 128 && m >= 4096,
        };
        if use_mb4 {
            return self.gemm_mq3g256_lloyd_residual_wmma_mb4(a_raw, x, y, m, k, batch_size);
        }
        let (src, module) = kernels::gemm_mq3g256_lloyd_residual_wmma_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_mq3g256_lloyd_residual_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = m * (k / 256) * LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_mq3g256_lloyd_residual_wmma",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_mq3g256_lloyd_residual_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
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

    /// MQ3-Lloyd WMMA residual mb4: 16×64 output tile per WG. Sibling of
    /// `gemm_mq4g256_lloyd_residual_wmma_mb4` ported to the MQ3 codebook
    /// (8 entries) + 3-bit cross-byte K-tile decode.
    pub fn gemm_mq3g256_lloyd_residual_wmma_mb4(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_mq3g256_lloyd_residual_wmma_mb4_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_mq3g256_lloyd_residual_wmma_mb4")?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 63) / 64;

        let weight_bytes = m * (k / 256) * LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_mq3g256_lloyd_residual_wmma_mb4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_mq3g256_lloyd_residual_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
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

    /// MQ3-Lloyd WMMA fused QKVZA GEMM (LA preamble: qkv + z + beta + alpha).
    /// 4-way fused — one launch covers all four projections of the LA layer.
    /// Caller pre-rotates X (FWHT) for MQ3-Lloyd dtype.
    pub fn gemm_qkvza_mq3g256_lloyd_wmma(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let arch_supports_mb4 = self.arch_caps.supports_mq3_lloyd_mb4();
        let use_mb4 = match self.flags.mq3_mb4 {
            None => arch_supports_mb4 && n >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_qkvza_mq3g256_lloyd_wmma_mb4(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, n,
            );
        }
        let (src, module) = kernels::gemm_qkvza_mq3g256_lloyd_wmma_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_qkvza_mq3g256_lloyd_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_qkv_p = a_qkv.buf.as_ptr();
        let mut a_z_p = a_z.buf.as_ptr();
        let mut a_beta_p = a_beta.buf.as_ptr();
        let mut a_alpha_p = a_alpha.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_qkv_p = y_qkv.buf.as_ptr();
        let mut y_z_p = y_z.buf.as_ptr();
        let mut y_beta_p = y_beta.buf.as_ptr();
        let mut y_alpha_p = y_alpha.buf.as_ptr();
        let mut qkv_m_v = qkv_m as i32;
        let mut z_m_v = z_m as i32;
        let mut beta_m_v = beta_m as i32;
        let mut alpha_m_v = alpha_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_qkv_p as *mut _ as *mut c_void,
            &mut a_z_p as *mut _ as *mut c_void,
            &mut a_beta_p as *mut _ as *mut c_void,
            &mut a_alpha_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_qkv_p as *mut _ as *mut c_void,
            &mut y_z_p as *mut _ as *mut c_void,
            &mut y_beta_p as *mut _ as *mut c_void,
            &mut y_alpha_p as *mut _ as *mut c_void,
            &mut qkv_m_v as *mut _ as *mut c_void,
            &mut z_m_v as *mut _ as *mut c_void,
            &mut beta_m_v as *mut _ as *mut c_void,
            &mut alpha_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 15) / 16;
        let weight_bytes = total_m * (k / 256) * LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_mq3g256_lloyd_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_mq3g256_lloyd_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_qkv_p);
                b.push_ptr(a_z_p);
                b.push_ptr(a_beta_p);
                b.push_ptr(a_alpha_p);
                b.push_ptr(x_p);
                b.push_ptr(y_qkv_p);
                b.push_ptr(y_z_p);
                b.push_ptr(y_beta_p);
                b.push_ptr(y_alpha_p);
                b.push_i32(qkv_m_v);
                b.push_i32(z_m_v);
                b.push_i32(beta_m_v);
                b.push_i32(alpha_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ3-Lloyd WMMA fused QKV GEMM (FA preamble: q + k + v).
    /// MQ3-Lloyd qkvza mb4 dispatch.
    pub fn gemm_qkvza_mq3g256_lloyd_wmma_mb4(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_qkvza_mq3g256_lloyd_wmma_mb4_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_qkvza_mq3g256_lloyd_wmma_mb4")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_qkv_p = a_qkv.buf.as_ptr();
        let mut a_z_p = a_z.buf.as_ptr();
        let mut a_beta_p = a_beta.buf.as_ptr();
        let mut a_alpha_p = a_alpha.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_qkv_p = y_qkv.buf.as_ptr();
        let mut y_z_p = y_z.buf.as_ptr();
        let mut y_beta_p = y_beta.buf.as_ptr();
        let mut y_alpha_p = y_alpha.buf.as_ptr();
        let mut qkv_m_v = qkv_m as i32;
        let mut z_m_v = z_m as i32;
        let mut beta_m_v = beta_m as i32;
        let mut alpha_m_v = alpha_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_qkv_p as *mut _ as *mut c_void,
            &mut a_z_p as *mut _ as *mut c_void,
            &mut a_beta_p as *mut _ as *mut c_void,
            &mut a_alpha_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_qkv_p as *mut _ as *mut c_void,
            &mut y_z_p as *mut _ as *mut c_void,
            &mut y_beta_p as *mut _ as *mut c_void,
            &mut y_alpha_p as *mut _ as *mut c_void,
            &mut qkv_m_v as *mut _ as *mut c_void,
            &mut z_m_v as *mut _ as *mut c_void,
            &mut beta_m_v as *mut _ as *mut c_void,
            &mut alpha_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 63) / 64;
        let weight_bytes = total_m * (k / 256) * LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkvza_mq3g256_lloyd_wmma_mb4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_qkvza_mq3g256_lloyd_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_qkv_p);
                b.push_ptr(a_z_p);
                b.push_ptr(a_beta_p);
                b.push_ptr(a_alpha_p);
                b.push_ptr(x_p);
                b.push_ptr(y_qkv_p);
                b.push_ptr(y_z_p);
                b.push_ptr(y_beta_p);
                b.push_ptr(y_alpha_p);
                b.push_i32(qkv_m_v);
                b.push_i32(z_m_v);
                b.push_i32(beta_m_v);
                b.push_i32(alpha_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    pub fn gemm_qkv_mq3g256_lloyd_wmma(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let total_m = q_m + k_m + v_m;
        let arch_supports_mb4 = self.arch_caps.supports_mq3_lloyd_mb4();
        let use_mb4 = match self.flags.mq3_mb4 {
            None => arch_supports_mb4 && n >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_qkv_mq3g256_lloyd_wmma_mb4(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, n,
            );
        }
        let (src, module) = kernels::gemm_qkv_mq3g256_lloyd_wmma_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_qkv_mq3g256_lloyd_wmma")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_q_p = a_q.buf.as_ptr();
        let mut a_k_p = a_k.buf.as_ptr();
        let mut a_v_p = a_v.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_q_p = y_q.buf.as_ptr();
        let mut y_k_p = y_k.buf.as_ptr();
        let mut y_v_p = y_v.buf.as_ptr();
        let mut q_m_v = q_m as i32;
        let mut k_m_v = k_m as i32;
        let mut v_m_v = v_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_q_p as *mut _ as *mut c_void,
            &mut a_k_p as *mut _ as *mut c_void,
            &mut a_v_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_q_p as *mut _ as *mut c_void,
            &mut y_k_p as *mut _ as *mut c_void,
            &mut y_v_p as *mut _ as *mut c_void,
            &mut q_m_v as *mut _ as *mut c_void,
            &mut k_m_v as *mut _ as *mut c_void,
            &mut v_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 15) / 16;
        let weight_bytes = total_m * (k / 256) * LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_mq3g256_lloyd_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_mq3g256_lloyd_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_q_p);
                b.push_ptr(a_k_p);
                b.push_ptr(a_v_p);
                b.push_ptr(x_p);
                b.push_ptr(y_q_p);
                b.push_ptr(y_k_p);
                b.push_ptr(y_v_p);
                b.push_i32(q_m_v);
                b.push_i32(k_m_v);
                b.push_i32(v_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ3-Lloyd WMMA fused gate+up GEMM (FFN preamble).
    /// MQ3-Lloyd qkv mb4 dispatch.
    pub fn gemm_qkv_mq3g256_lloyd_wmma_mb4(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module) = kernels::gemm_qkv_mq3g256_lloyd_wmma_mb4_for_arch(&self.arch_caps);
        self.ensure_kernel(module, src, "gemm_qkv_mq3g256_lloyd_wmma_mb4")?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_q_p = a_q.buf.as_ptr();
        let mut a_k_p = a_k.buf.as_ptr();
        let mut a_v_p = a_v.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_q_p = y_q.buf.as_ptr();
        let mut y_k_p = y_k.buf.as_ptr();
        let mut y_v_p = y_v.buf.as_ptr();
        let mut q_m_v = q_m as i32;
        let mut k_m_v = k_m as i32;
        let mut v_m_v = v_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_q_p as *mut _ as *mut c_void,
            &mut a_k_p as *mut _ as *mut c_void,
            &mut a_v_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_q_p as *mut _ as *mut c_void,
            &mut y_k_p as *mut _ as *mut c_void,
            &mut y_v_p as *mut _ as *mut c_void,
            &mut q_m_v as *mut _ as *mut c_void,
            &mut k_m_v as *mut _ as *mut c_void,
            &mut v_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 63) / 64;
        let weight_bytes = total_m * (k / 256) * LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkv_mq3g256_lloyd_wmma_mb4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_qkv_mq3g256_lloyd_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_q_p);
                b.push_ptr(a_k_p);
                b.push_ptr(a_v_p);
                b.push_ptr(x_p);
                b.push_ptr(y_q_p);
                b.push_ptr(y_k_p);
                b.push_ptr(y_v_p);
                b.push_i32(q_m_v);
                b.push_i32(k_m_v);
                b.push_i32(v_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    pub fn gemm_gate_up_mq3g256_lloyd_wmma(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let total_m = gate_m + up_m;
        let arch_supports_mb4 = self.arch_caps.supports_mq3_lloyd_mb4();
        let use_mb4 = match self.flags.mq3_mb4 {
            None => arch_supports_mb4 && n >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_gate_up_mq3g256_lloyd_wmma_mb4(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, n,
            );
        }
        let (src, module, func_name) = if self.flags.gate_up_nosync {
            let (src, module) =
                kernels::gemm_gate_up_mq3g256_lloyd_wmma_nosync_for_arch(&self.arch_caps);
            (src, module, "gemm_gate_up_mq3g256_lloyd_wmma_nosync")
        } else {
            let (src, module) = kernels::gemm_gate_up_mq3g256_lloyd_wmma_for_arch(&self.arch_caps);
            (src, module, "gemm_gate_up_mq3g256_lloyd_wmma")
        };
        self.ensure_kernel(module, src, func_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_gate_p = a_gate.buf.as_ptr();
        let mut a_up_p = a_up.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_gate_p = y_gate.buf.as_ptr();
        let mut y_up_p = y_up.buf.as_ptr();
        let mut gate_m_v = gate_m as i32;
        let mut up_m_v = up_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_gate_p as *mut _ as *mut c_void,
            &mut a_up_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_gate_p as *mut _ as *mut c_void,
            &mut y_up_p as *mut _ as *mut c_void,
            &mut gate_m_v as *mut _ as *mut c_void,
            &mut up_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 15) / 16;
        let weight_bytes = total_m * (k / 256) * LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", func_name, bytes);
        let result = self.launch_maybe_blob(
            func_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_gate_p);
                b.push_ptr(a_up_p);
                b.push_ptr(x_p);
                b.push_ptr(y_gate_p);
                b.push_ptr(y_up_p);
                b.push_i32(gate_m_v);
                b.push_i32(up_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ3-Lloyd gate_up mb4 dispatch.
    pub fn gemm_gate_up_mq3g256_lloyd_wmma_mb4(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (src, module, func_name) = if self.flags.gate_up_nosync {
            let (src, module) =
                kernels::gemm_gate_up_mq3g256_lloyd_wmma_mb4_nosync_for_arch(&self.arch_caps);
            (src, module, "gemm_gate_up_mq3g256_lloyd_wmma_mb4_nosync")
        } else {
            let (src, module) =
                kernels::gemm_gate_up_mq3g256_lloyd_wmma_mb4_for_arch(&self.arch_caps);
            (src, module, "gemm_gate_up_mq3g256_lloyd_wmma_mb4")
        };
        self.ensure_kernel(module, src, func_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x, n * k)?;

        let mut a_gate_p = a_gate.buf.as_ptr();
        let mut a_up_p = a_up.buf.as_ptr();
        let mut x_p = x_f16_ptr;
        let mut y_gate_p = y_gate.buf.as_ptr();
        let mut y_up_p = y_up.buf.as_ptr();
        let mut gate_m_v = gate_m as i32;
        let mut up_m_v = up_m as i32;
        let mut k_v = k as i32;
        let mut n_v = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_gate_p as *mut _ as *mut c_void,
            &mut a_up_p as *mut _ as *mut c_void,
            &mut x_p as *mut _ as *mut c_void,
            &mut y_gate_p as *mut _ as *mut c_void,
            &mut y_up_p as *mut _ as *mut c_void,
            &mut gate_m_v as *mut _ as *mut c_void,
            &mut up_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (n + 63) / 64;
        let weight_bytes = total_m * (k / 256) * LLOYD_MQ3_GROUP_BYTES;
        let bytes = weight_bytes + n * k * 2 + n * total_m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", func_name, bytes);
        let result = self.launch_maybe_blob(
            func_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_gate_p);
                b.push_ptr(a_up_p);
                b.push_ptr(x_p);
                b.push_ptr(y_gate_p);
                b.push_ptr(y_up_p);
                b.push_i32(gate_m_v);
                b.push_i32(up_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Fused Gate+Up MQ3-Lloyd: two GEMVs in one launch. Mirrors
    /// `fused_gate_up_hfq4g256` for the Lloyd-MQ3 dtype. Caller is
    /// responsible for pre-rotating x (FWHT) before invoking; the kernel
    /// itself only does the GEMV. Both `a_gate` and `a_up` must be MQ3-Lloyd
    /// matrices with the same K and codebook layout.

    /// Fused QKVZA MQ3-Lloyd: 4 LA-preamble GEMVs in one launch. Used by
    /// qwen35.rs DeltaNet decode when wqkv + wz + w_beta + w_alpha are
    /// all MQ3G256Lloyd. Mirrors `fused_qkvza_hfq4g256` — same routing
    /// (grid = qkv_m + z_m + beta_m + alpha_m, block picks A by gid),
    /// Lloyd K4+LDS body on gfx1100. Caller is responsible for
    /// pre-rotating x (FWHT); the kernel only does the GEMVs.

    /// Fused QKV MQ3-Lloyd: 3 FA-preamble GEMVs in one launch. Used by
    /// qwen35.rs FullAttention decode when wq + wk + wv are all
    /// MQ3G256Lloyd. Sibling of `fused_qkvza_mq3g256_lloyd` for the
    /// 3-projection FA case (vs LA's 4-projection QKVZA). Caller is
    /// responsible for pre-rotating x; the kernel only does the GEMVs.

    /// Lazily initialize MagnumQuant FWHT sign tables (256 floats each, seeds 42 and 1042).

    /// Lazily initialize MagnumQuant FWHT sign tables for G128 (128 floats each, seeds 43 and 1043).
    /// Also allocates the shared `mq_x_rot` scratch if not already present — the G256 path
    /// (`ensure_mq_signs`) normally owns that allocation, but the G128 path must be
    /// self-sufficient so models that carry only MQ4G128 weights still get the scratch buffer.

    /// MagnumQuant GEMV: FWHT-rotated HFQ4-G256. Rotates x per group via ds_swizzle,
    /// then standard 4-bit dot product. signs1/signs2 are the FWHT sign tables (256 floats each).

    /// HFP4-G32 GEMV — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale).
    ///
    /// v1 correctness anchor: no WMMA, no FP8, no rotation. K must be a multiple of 256
    /// (the kernel's 4-accumulator + tail-by-g%4 outer loop assumes the 256-element
    /// "iter window" stride; v2 will lift this to k%32==0). See `kernels/src/gemv_hfp4g32.hip`
    /// and `docs/quant-formats/hfp4.md`.

    /// Direct fallback entry point (F32 mul+fma chain). Useful for
    /// A/B benchmarking against the dot2/fp8 variants.

    /// gfx12 FP8-dot4 decode-path GEMV for HFP4G32. Uses
    /// `dot4_f32_fp8_fp8` to cut inner-loop ALU vs the dequant/FMA
    /// fallback. Activation X is consumed as FP8 (E4M3); when called
    /// via `gemv_hfp4g32` (env-gated routing for HFP4G32 weights, no
    /// rotation), this function calls `ensure_fp8_x` to pack F32 → FP8
    /// scratch. The MFP4G32 rotation path uses
    /// `rotate_x_mq_dual_fp8` + `gemv_hfp4g32_fp8_gfx12_with_fp8_ptr`
    /// instead so the FP8 pack is fused into the rotation kernel.

    /// Fused RMSNorm + MagnumQuant FWHT rotation. Replaces the
    /// `rmsnorm_f32` + `rotate_x_mq` sequence with a single kernel launch.
    /// Reads unnormalized `x` + rmsnorm `weight`, computes rmsnorm in LDS,
    /// applies the same per-256-element FWHT as `mq_rotate_x`, and writes
    /// the rotated normalized vector into `x_rot`.
    ///
    /// Preconditions:
    /// - `k` is a multiple of 256 (enforced by callers via `config.dim`)
    /// - `k` ≤ 16384 (LDS ceiling; 16K floats = 64KB minus reduce buffer)

    /// Phase A Stage A — AWQ-aware variant of fused_rmsnorm_rotate_mq.
    ///
    /// After computing the RMSNorm output, divides element-wise by
    /// `awq_scale[i]` BEFORE the FWHT rotation. Completes the AWQ math
    /// `(W·s) · (x/s) = W·x` where W·s is baked at quantize time.
    ///
    /// Use when the upcoming linear layer's WeightTensor carries
    /// `awq_scale = Some(...)`; otherwise call the non-AWQ variant.
    ///
    /// awq_scale: 1D FP32 GpuTensor of length K (host-side F16 → F32
    /// conversion happens in the loader; see hfq.rs::load_awq_scale).
    ///
    /// Backward-compatible: kernel is separate, no behavioral change for
    /// the standard fused_rmsnorm_rotate_mq path.

    /// Batched `fused_rmsnorm_rotate_mq`. Grid.x is the batch dim — processes
    /// N tokens' [N × K] x into [N × K] x_rot in a single launch. Byte-exact
    /// against calling `fused_rmsnorm_rotate_mq` N times on separate x/x_rot
    /// buffers. Weight/signs are shared across the batch.
    /// Phase A Stage A — batched AWQ variant. Mirrors
    /// fused_rmsnorm_rotate_mq_batched but takes an additional
    /// `awq_scale: &GpuTensor` (length K, FP32) and dispatches the
    /// AWQ kernel. Caller selects based on the upcoming linear
    /// layer's WeightTensor.awq_scale being Some.

    /// gfx942 two-kernel split: rmsnorm_reduce + rotate_with_rms.
    ///
    /// Replaces the single-WG-per-batch fused kernel with two kernels that
    /// each scale better on MI300X's 304 CUs. Kernel A computes rms per
    /// batch (1 WG/batch × 16 wave64s). Kernel B applies rmsnorm + FWHT
    /// per (group, batch) cell (K/256 × batch WGs × 1 wave64 each).
    ///
    /// For batch=256 K=5120: 20×256 = 5120 wave64s on Kernel B vs 1024 on
    /// the fused path → 5× more in-flight waves on prefill.
    ///
    /// Math byte-identical to fused_rmsnorm_mq_rotate.

    /// Fused SwiGLU + FWHT rotation. Reads gate/up, computes
    /// silu(gate[k])*up[k] on the fly, applies FWHT rotation, writes x_rot.
    /// Used as the w_down input stage for MQ4 — replaces the pair
    /// silu_mul_f32 + mq_rotate_x with one launch.

    /// Batched `fused_silu_mul_rotate_mq`. Grid.y is the batch dim — processes
    /// N tokens' [N × K] gate/up/x_rot in a single launch.

    /// Phase A Stage A — F2 AWQ-aware variant of `fused_silu_mul_rotate_mq`.
    ///
    /// After computing silu(gate)*up, divides element-wise by `awq_scale[i]`
    /// BEFORE the FWHT rotation. Completes the AWQ math
    /// `(W·s) · (silu(g)*u / s) = W·silu(g)*u` where W·s is baked at
    /// quantize time for the down_proj / w_down weights.
    ///
    /// Use when the down_proj `WeightTensor` carries `awq_scale = Some(...)`;
    /// otherwise call the non-AWQ variant.
    ///
    /// awq_scale: 1D FP32 GpuTensor of length K (host-side F16 → F32
    /// conversion happens in the loader; see hfq.rs::load_awq_scale).

    /// Phase A Stage A — F2 batched AWQ variant of `fused_silu_mul_rotate_mq`.
    /// Grid.y is the batch dim — processes [N × K] gate/up/x_rot.

    /// Invalidate any `ensure_*_x` caches whose source pointer matches
    /// `dst_ptr`. Must be called by any kernel that overwrites data at
    /// `dst_ptr` since the caches key on raw pointer equality and have
    /// no way to detect data changes otherwise.

    /// Standalone FWHT rotation for MagnumQuant (MQ4). Writes K floats into x_rot.

    /// Batched `rotate_x_mq`. Grid.y is the batch dim.

    /// FWHT-128 standalone rotation for MQ4G128 activations.
    ///
    /// Mirrors `rotate_x_mq` but targets G128 groups (32 threads × 4 elems).
    /// Grid: [k/128, 1, 1]. Block: [32, 1, 1].

    /// Phase A Stage A — F2 AWQ-aware variant of `rotate_x_mq`.
    ///
    /// Divides each input element by `awq_scale[i]` BEFORE the FWHT.
    ///
    /// awq_scale: 1D FP32 GpuTensor of length K.

    /// Phase A Stage A — F2 batched AWQ variant of `rotate_x_mq`.
    /// Grid.y is the batch dim — processes [N × K] x/x_rot.

    /// MagnumQuant MQ4: rotate x once, then GEMV against rotated x.
    /// MQ4 weights are stored in HFQ4-G256 format with FWHT pre-applied, so the GEMV
    /// inner loop is identical to standard HFQ4 — we reuse the arch-tuned HFQ4 kernel.

    /// MagnumQuant MQ4 with pre-rotated x. Skips the rotation step entirely —
    /// caller must have called `rotate_x_mq` into `x_rot` first.

    /// MagnumQuant MQ4-G128 with pre-rotated x. Skips the rotation step entirely —
    /// caller must have called `rotate_x_mq_128` into `x_rot` first.

    /// MFP4G32: rotate x once via FWHT, then HFP4G32 GEMV against rotated x.
    /// MFP4 weights are stored in HFP4G32 format (E2M1 + UE8M0 g32 + FP16 row scale)
    /// with the same 256-element FWHT pre-applied, so the GEMV inner loop is
    /// identical to standard HFP4 — we reuse `gemv_hfp4g32`.

    /// MFP4G32 with pre-rotated x. Skips the rotation step entirely — caller must
    /// have called `rotate_x_mq` into `x_rot` first.

    /// Fused FWHT rotation + FP8 pack for the decode FP8 path.
    /// Writes both F32 (into `x_rot`) and FP8 (into `mq_x_rot_fp8`
    /// sibling scratch) in one kernel launch. Returns the FP8 buffer's
    /// device pointer for the caller to feed directly to the FP8 GEMV.
    /// gfx12-only — uses cvt_pk_fp8_f32.

    /// gfx11 (RDNA3) v_dot2_f32_f16 decode-path GEMV for HFP4G32.
    /// Takes F32 x and converts to FP16 INLINE in the inner loop;
    /// `__builtin_amdgcn_fdot2` (v_dot2_f32_f16) does 2 FP16 muls +
    /// 1 FP32 add per VALU. Reduces inner-loop multiply count ~4×
    /// vs the fallback F32 mul+fma chain on ALU-bound shapes.
    /// Routed automatically from `gemv_hfp4g32` when on gfx11+ archs
    /// (gfx1100/1101/1102/1150/1151). NO ensure_fp16_x pre-pass —
    /// that's the v1 trap (eats the dot2 savings in production).

    /// FP8-dot4 GEMV variant that takes an FP8 device pointer directly
    /// (bypassing `ensure_fp8_x`). Used by `gemv_mfp4g32_with_rotate`
    /// after the fused rotation+pack kernel produces the FP8 buffer
    /// in-place.

    /// MagnumQuant MQ3: rotate x once, then HFQ3-G256 GEMV against rotated x.
    /// MQ3 weights are stored in HFQ3-G256 format (104 B/group) with FWHT pre-applied,
    /// so the GEMV inner loop is identical to standard HFQ3.

    /// MagnumQuant MQ3 with pre-rotated x.

    /// MagnumQuant MQ2: rotate x once, then HFQ2-G256 GEMV against rotated x.
    /// MQ2 weights are stored in HFQ2-G256 format (72 B/group) with FWHT pre-applied.

    /// MagnumQuant MQ2 with pre-rotated x.

    /// MagnumQuant MQ6: rotate x via FWHT, then HFQ6 GEMV.

    /// MagnumQuant MQ6 with pre-rotated x.

    /// Standalone MQ8 rotate + INT8 quantize of x into internal `mq_x_q8`/`mq_x_scales`.
    /// After this, `gemv_mq8g256_prerotated` can be called multiple times with the same x.

    /// MQ8 dp4a GEMV using pre-rotated+quantized x. Caller must have called
    /// `rotate_quantize_x_mq8(x, k)` first — results use the internal `mq_x_q8`/`mq_x_scales`.

    /// MagnumQuant MQ8: FWHT rotate + INT8 quantize x, then dp4a GEMV.

    /// HFQ3-G256 GEMV. K must be multiple of 256.
    /// Per-arch dispatch: gfx1100/1101/1102 uses the K4-unrolled
    /// 4-accumulator variant. The default kernel was re-ported to match
    /// the same ordering so non-RDNA3 archs (gfx1010, gfx1030, gfx12,
    /// gfx9xx) produce byte-exact results against the RDNA3 baseline.
    /// Uses `launch_maybe_blob` for HIPFIRE_GRAPH=1 capture safety.

    /// HFQ3-G256 GEMV with fused residual add: y[row] += A[row] dot x.
    /// Used by `weight_gemv_residual` MQ3 arm to eliminate the
    /// alloc+gemv+add+free fallback chain (saves ~3 launches per residual).
    /// gfx1100 selects the K4-unrolled chip-specific variant (commit 0003103,
    /// 9B MQ3 decode 114 to 141 tok/s); other archs use the K4-ported default
    /// (re-port in 9fdba4d keeps non-RDNA3 archs byte-exact with the prior
    /// gemv + add_inplace path). Uses launch_maybe_blob for HIPFIRE_GRAPH=1
    /// capture safety.

    /// MagnumQuant MQ3-G256 GEMV with fused residual add. The pre-rotation
    /// happens in a separate kernel via fused_silu_mul_mq_rotate or
    /// rotate_x_for_mq; this function just dispatches the underlying
    /// hfq3g256_residual against the already-rotated x.

    /// HFQ3-G128 GEMV. K must be multiple of 128. Finer granularity than G256.

    /// HFQ2-G128 GEMV. K must be multiple of 128. Finer granularity than G256.

    /// HFQ6-G256 GEMV with fused residual add: y[row] += A[row] . x.
    /// Same shape as gemv_hfq6g256; only the final write differs (+= vs =).
    /// Used for wo and w_down in HFQ6 / MQ6 forward paths so the
    /// add_inplace_f32 follow-up launch can be elided.

    /// HFQ6-G256 GEMV. K must be multiple of 256.

    /// HFQ8-G256 GEMV. K must be multiple of 256.

    /// HFQ4-G512 GEMV. K must be multiple of 512.

    /// HFQ4-G1024 GEMV. K must be multiple of 1024.

    /// HFQ4-G256 GEMV: flat 4-bit with 256-weight groups. K must be multiple of 256.

    /// dp4a-port of fused_qkv_hfq4g256 for gfx906. Pre-quantizes x to
    /// Q8_1 via the shared MMQ scratch, then runs the dp4a-based GEMV.
    /// Math is identical modulo Q8_1 quant noise. Targets gfx906's
    /// memory-bound regime per the per-kernel PMC pass at 2026-05-05.
    pub fn fused_qkv_hfq4g256_dp4a(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_qkv_hfq4g256_wave64_dp4a",
            kernels::FUSED_QKV_HFQ4G256_WAVE64_DP4A_SRC,
            "fused_qkv_hfq4g256_wave64_dp4a",
        )?;

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_val = q_m as i32;
        let k_m_val = k_m as i32;
        let v_m_val = v_m as i32;
        let k_val = k as i32;
        let total = (q_m + k_m + v_m) as u32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &ak as *const _ as *mut c_void,
            &av as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yk as *const _ as *mut c_void,
            &yv as *const _ as *mut c_void,
            &q_m_val as *const _ as *mut c_void,
            &k_m_val as *const _ as *mut c_void,
            &v_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k);
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_qkv_hfq4g256_dp4a", bytes);
        let result = self.launch_maybe_blob(
            "fused_qkv_hfq4g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xq);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    // HFQ2 GEMV dispatch already exists at line ~521 from the HFQ family

    /// gfx906 dp4a-port — see fused_gate_up_hfq6g256_wave64_dp4a.hip for
    /// the math derivation. Plan §3.1.1 item 3 / v3.2.2 §5.1 item 1c.
    pub fn fused_qkv_hfq6g256_dp4a(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_qkv_hfq6g256_wave64_dp4a",
            kernels::FUSED_QKV_HFQ6G256_WAVE64_DP4A_SRC,
            "fused_qkv_hfq6g256_wave64_dp4a",
        )?;

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_val = q_m as i32;
        let k_m_val = k_m as i32;
        let v_m_val = v_m as i32;
        let k_val = k as i32;
        let total = (q_m + k_m + v_m) as u32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &ak as *const _ as *mut c_void,
            &av as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yk as *const _ as *mut c_void,
            &yv as *const _ as *mut c_void,
            &q_m_val as *const _ as *mut c_void,
            &k_m_val as *const _ as *mut c_void,
            &v_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "fused_qkv_hfq6g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xq);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b
            },
        )
    }

    /// gfx906 dp4a-port — 4-output deltanet QKV preamble.
    pub fn fused_qkvza_hfq6g256_dp4a(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_qkvza_hfq6g256_wave64_dp4a",
            kernels::FUSED_QKVZA_HFQ6G256_WAVE64_DP4A_SRC,
            "fused_qkvza_hfq6g256_wave64_dp4a",
        )?;

        let aqkv = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let yqkv = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let qkv_m_val = qkv_m as i32;
        let z_m_val = z_m as i32;
        let beta_m_val = beta_m as i32;
        let alpha_m_val = alpha_m as i32;
        let k_val = k as i32;
        let total = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &aqkv as *const _ as *mut c_void,
            &az as *const _ as *mut c_void,
            &ab as *const _ as *mut c_void,
            &aa as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yqkv as *const _ as *mut c_void,
            &yz as *const _ as *mut c_void,
            &yb as *const _ as *mut c_void,
            &ya as *const _ as *mut c_void,
            &qkv_m_val as *const _ as *mut c_void,
            &z_m_val as *const _ as *mut c_void,
            &beta_m_val as *const _ as *mut c_void,
            &alpha_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "fused_qkvza_hfq6g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aqkv);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xq);
                b.push_ptr(yqkv);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(qkv_m_val);
                b.push_i32(z_m_val);
                b.push_i32(beta_m_val);
                b.push_i32(alpha_m_val);
                b.push_i32(k_val);
                b
            },
        )
    }

    /// gfx906 dp4a-port — 2-output FFN gate+up projection.
    pub fn fused_gate_up_hfq6g256_dp4a(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_gate_up_hfq6g256_wave64_dp4a",
            kernels::FUSED_GATE_UP_HFQ6G256_WAVE64_DP4A_SRC,
            "fused_gate_up_hfq6g256_wave64_dp4a",
        )?;

        let agate = a_gate.buf.as_ptr();
        let aup = a_up.buf.as_ptr();
        let ygate = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let gate_m_val = gate_m as i32;
        let up_m_val = up_m as i32;
        let k_val = k as i32;
        let total = (gate_m + up_m) as u32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &agate as *const _ as *mut c_void,
            &aup as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &ygate as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &gate_m_val as *const _ as *mut c_void,
            &up_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "fused_gate_up_hfq6g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(agate);
                b.push_ptr(aup);
                b.push_ptr(xq);
                b.push_ptr(ygate);
                b.push_ptr(yup);
                b.push_i32(gate_m_val);
                b.push_i32(up_m_val);
                b.push_i32(k_val);
                b
            },
        )
    }

    /// 3-way fused HFQ4-G256 projection — cross-arch.
    ///
    /// Performs y_q=A_q·x, y_k=A_k·x, y_v=A_v·x in a single kernel launch
    /// for the Qwen3.5 FullAttention layer preamble. Same rationale and
    /// tail-handling guarantees as `fused_qkvza_hfq4g256`.
    pub fn fused_qkv_hfq4g256(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.fused_qkv_hfq4g256_impl(a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, None)
    }

    /// Qwen2-only bias-fold entry. It uses a distinct kernel symbol and kernarg
    /// ABI; [`Self::fused_qkv_hfq4g256`] remains unchanged for Qwen3+/Redline.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_hfq4g256_with_bias(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        bias_q_ptr: *mut c_void,
        bias_k_ptr: *mut c_void,
        bias_v_ptr: *mut c_void,
    ) -> HipResult<()> {
        self.fused_qkv_hfq4g256_impl(
            a_q,
            a_k,
            a_v,
            x,
            y_q,
            y_k,
            y_v,
            q_m,
            k_m,
            v_m,
            k,
            Some((bias_q_ptr, bias_k_ptr, bias_v_ptr)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn fused_qkv_hfq4g256_impl(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        bias: Option<(*mut c_void, *mut c_void, *mut c_void)>,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.gemv_dp4a_enabled() {
            debug_assert!(bias.is_none(), "Qwen2 bias fold is disabled on dp4a");
            return self.fused_qkv_hfq4g256_dp4a(a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k);
        }

        static GFX1151_QKVZA_WAVE64: OnceLock<bool> = OnceLock::new();
        let gfx1151_wave64 = self.arch_caps.is_gfx1151()
            && *GFX1151_QKVZA_WAVE64.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_QKVZA_WAVE64").as_deref() == Ok("1")
            });
        static GFX1151_QKV_ALL_BUFFER_CPOL: OnceLock<String> = OnceLock::new();
        let gfx1151_all_buffer_cpol = if self.arch_caps.is_gfx1151() && k == 2_048 {
            GFX1151_QKV_ALL_BUFFER_CPOL
                .get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_GFX1151_QKV_ALL_BUFFER_CPOL")
                        .unwrap_or_default()
                        .to_ascii_lowercase()
                })
                .as_str()
        } else {
            ""
        };
        static GFX1151_QKV_X_BUFFER: OnceLock<bool> = OnceLock::new();
        let gfx1151_x_buffer = self.arch_caps.is_gfx1151()
            && k == 2_048
            && *GFX1151_QKV_X_BUFFER.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_QKV_X_BUFFER").as_deref() == Ok("1")
            });
        let cdna_wave64 = self.arch_caps.is_wave64_native()
            || (self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_qkv_wave64)
            || gfx1151_wave64;
        let (func_name, block, grid_x) = if bias.is_some() && cdna_wave64 {
            let total = (q_m + k_m + v_m) as u32;
            if self.arch_caps.is_cdna3() && self.flags.gfx942_gemv_v2.unwrap_or(true) {
                self.ensure_kernel(
                    "fused_qkv_hfq4g256_v2_gfx942_qwen2_bias",
                    kernels::FUSED_QKV_HFQ4G256_V2_GFX942_QWEN2_BIAS_SRC,
                    "fused_qkv_hfq4g256_v2_gfx942_qwen2_bias",
                )?;
                (
                    "fused_qkv_hfq4g256_v2_gfx942_qwen2_bias",
                    [128u32, 1, 1],
                    (total + 3) / 4,
                )
            } else {
                self.ensure_kernel(
                    "fused_qkv_hfq4g256_wave64_qwen2_bias",
                    kernels::FUSED_QKV_HFQ4G256_WAVE64_QWEN2_BIAS_SRC,
                    "fused_qkv_hfq4g256_wave64_qwen2_bias",
                )?;
                (
                    "fused_qkv_hfq4g256_wave64_qwen2_bias",
                    [64u32, 1, 1],
                    (total + 1) / 2,
                )
            }
        } else if bias.is_some() {
            self.ensure_kernel(
                "fused_qkv_hfq4g256_qwen2_bias",
                kernels::FUSED_QKV_HFQ4G256_QWEN2_BIAS_SRC,
                "fused_qkv_hfq4g256_qwen2_bias",
            )?;
            (
                "fused_qkv_hfq4g256_qwen2_bias",
                [32u32, 1, 1],
                (q_m + k_m + v_m) as u32,
            )
        } else if gfx1151_x_buffer {
            self.ensure_kernel(
                "fused_qkv_hfq4g256_k2048_x_buffer_gfx1151",
                kernels::FUSED_QKV_HFQ4G256_K2048_X_BUFFER_GFX1151_SRC,
                "fused_qkv_hfq4g256_k2048_x_buffer_gfx1151",
            )?;
            (
                "fused_qkv_hfq4g256_k2048_x_buffer_gfx1151",
                [32u32, 1, 1],
                (q_m + k_m + v_m) as u32,
            )
        } else if matches!(gfx1151_all_buffer_cpol, "temporal" | "slc") {
            let (module, source, function) = if gfx1151_all_buffer_cpol == "slc" {
                (
                    "fused_qkv_hfq4g256_k2048_all_buffer_slc_gfx1151",
                    kernels::FUSED_QKV_HFQ4G256_K2048_ALL_BUFFER_SLC_GFX1151_SRC,
                    "fused_qkv_hfq4g256_k2048_all_buffer_slc_gfx1151",
                )
            } else {
                (
                    "fused_qkv_hfq4g256_k2048_all_buffer_gfx1151",
                    kernels::FUSED_QKV_HFQ4G256_K2048_ALL_BUFFER_GFX1151_SRC,
                    "fused_qkv_hfq4g256_k2048_all_buffer_gfx1151",
                )
            };
            self.ensure_kernel(module, source, function)?;
            (function, [32u32, 1, 1], (q_m + k_m + v_m) as u32)
        } else if cdna_wave64 {
            // gfx94x v2: 2 wave64s = 4 rows/WG, +1.9% on AR decode
            // (commit 5bd75a69 sibling). Default ON; opt out via
            // HIPFIRE_GFX942_GEMV_V2=0.
            let is_gfx94x = self.arch_caps.is_cdna3();
            let v2_on = self.flags.gfx942_gemv_v2.unwrap_or(true);
            if is_gfx94x && v2_on {
                self.ensure_kernel(
                    "fused_qkv_hfq4g256_v2_gfx942",
                    kernels::FUSED_QKV_HFQ4G256_V2_GFX942_SRC,
                    "fused_qkv_hfq4g256_v2_gfx942",
                )?;
                let total = (q_m + k_m + v_m) as u32;
                (
                    "fused_qkv_hfq4g256_v2_gfx942",
                    [128u32, 1, 1],
                    (total + 3) / 4,
                )
            } else {
                self.ensure_kernel(
                    "fused_qkv_hfq4g256_wave64",
                    kernels::FUSED_QKV_HFQ4G256_WAVE64_SRC,
                    "fused_qkv_hfq4g256_wave64",
                )?;
                let total = (q_m + k_m + v_m) as u32;
                ("fused_qkv_hfq4g256_wave64", [64u32, 1, 1], (total + 1) / 2)
            }
        } else {
            self.ensure_kernel(
                "fused_qkv_hfq4g256",
                kernels::FUSED_QKV_HFQ4G256_SRC,
                "fused_qkv_hfq4g256",
            )?;
            (
                "fused_qkv_hfq4g256",
                [32u32, 1, 1],
                (q_m + k_m + v_m) as u32,
            )
        };

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_val = q_m as i32;
        let k_m_val = k_m as i32;
        let v_m_val = v_m as i32;
        let k_val = k as i32;
        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qkv_hfq4g256", bytes);
        let result = if let Some((bq, bk, bv)) = bias {
            let mut params: Vec<*mut c_void> = vec![
                &aq as *const _ as *mut c_void,
                &ak as *const _ as *mut c_void,
                &av as *const _ as *mut c_void,
                &xp as *const _ as *mut c_void,
                &yq as *const _ as *mut c_void,
                &yk as *const _ as *mut c_void,
                &yv as *const _ as *mut c_void,
                &q_m_val as *const _ as *mut c_void,
                &k_m_val as *const _ as *mut c_void,
                &v_m_val as *const _ as *mut c_void,
                &k_val as *const _ as *mut c_void,
                &bq as *const _ as *mut c_void,
                &bk as *const _ as *mut c_void,
                &bv as *const _ as *mut c_void,
            ];
            self.launch_maybe_blob(func_name, [grid_x, 1, 1], block, 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_ptr(bq);
                b.push_ptr(bk);
                b.push_ptr(bv);
                b
            })
        } else {
            let mut params: Vec<*mut c_void> = vec![
                &aq as *const _ as *mut c_void,
                &ak as *const _ as *mut c_void,
                &av as *const _ as *mut c_void,
                &xp as *const _ as *mut c_void,
                &yq as *const _ as *mut c_void,
                &yk as *const _ as *mut c_void,
                &yv as *const _ as *mut c_void,
                &q_m_val as *const _ as *mut c_void,
                &k_m_val as *const _ as *mut c_void,
                &v_m_val as *const _ as *mut c_void,
                &k_val as *const _ as *mut c_void,
            ];
            self.launch_maybe_blob(func_name, [grid_x, 1, 1], block, 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b
            })
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Per-row 3-way fused HFQ6/MQ6-G256 QKV decode (qwen2-family). Authored
    /// from `gemv_hfq6g256`'s row body; bit-exact with three sequential
    /// `gemv_hfq6g256` calls. Replaces the n=1 `gemm_qkv_hfq6g256` decode path
    /// (a deliberate decode-kernel change — see the FusedQkvHfq6G256 dispatch
    /// arm). Null bias = byte-identical to 3×gemv_hfq6g256.
    pub fn fused_qkv_hfq6g256(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.fused_qkv_hfq6g256_with_bias(
            a_q,
            a_k,
            a_v,
            x,
            y_q,
            y_k,
            y_v,
            q_m,
            k_m,
            v_m,
            k,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    }

    /// `fused_qkv_hfq6g256` with an optional Q/K/V bias folded into the kernel's
    /// lane-0 store (`HIPFIRE_FUSE_QKV_BIAS`). All-null = byte-identical.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_hfq6g256_with_bias(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        bias_q_ptr: *mut c_void,
        bias_k_ptr: *mut c_void,
        bias_v_ptr: *mut c_void,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_qkv_hfq6g256",
            kernels::FUSED_QKV_HFQ6G256_SRC,
            "fused_qkv_hfq6g256",
        )?;

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_i = q_m as i32;
        let k_m_i = k_m as i32;
        let v_m_i = v_m as i32;
        let k_i = k as i32;
        let bq = bias_q_ptr;
        let bk = bias_k_ptr;
        let bv = bias_v_ptr;
        let total = (q_m + k_m + v_m) as u32;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &ak as *const _ as *mut c_void,
            &av as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yk as *const _ as *mut c_void,
            &yv as *const _ as *mut c_void,
            &q_m_i as *const _ as *mut c_void,
            &k_m_i as *const _ as *mut c_void,
            &v_m_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
            &bq as *const _ as *mut c_void,
            &bk as *const _ as *mut c_void,
            &bv as *const _ as *mut c_void,
        ];

        let bytes = crate::profile::gemv_hfq6g256_bytes(q_m, k)
            + crate::profile::gemv_hfq6g256_bytes(k_m, k)
            + crate::profile::gemv_hfq6g256_bytes(v_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qkv_hfq6g256", bytes);
        let result = self.launch_maybe_blob(
            "fused_qkv_hfq6g256",
            [total, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_i);
                b.push_i32(k_m_i);
                b.push_i32(v_m_i);
                b.push_i32(k_i);
                b.push_ptr(bq);
                b.push_ptr(bk);
                b.push_ptr(bv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// 4-way fused HFQ4-G256 projection — cross-arch.
    ///
    /// Performs y_qkv=A_qkv·x, y_z=A_z·x, y_beta=A_beta·x, y_alpha=A_alpha·x
    /// in a single kernel launch, where all four matrices share the same
    /// input `x` and the same K. Used by the Qwen3.5 DeltaNet LA layer
    /// preamble to collapse four launches (one per projection) into one.
    /// Bit-exact with four sequential `gemv_hfq4g256` calls.
    ///
    /// Works on every RDNA generation (gfx1010 / gfx1013 / gfx1030 /
    /// gfx1100+) because the inner loop and the standalone gemv_hfq4g256
    /// inner loop were unified onto the same 4-accumulator structure
    /// after commit 5302926.
    pub fn fused_qkvza_hfq4g256(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.gemv_dp4a_enabled() {
            return self.fused_qkvza_hfq4g256_dp4a(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k,
            );
        }
        // gfx906/gfx908/gfx94x wave64-native path:
        // 2 rows per block, halves grid count vs wave32 kernel which wastes half
        // the wave slot. This kernel uses no MFMA, just FMA + shfl_down within
        // wave64, so it is safe for Vega 20 as well as CDNA.
        let total_m = qkv_m + z_m + beta_m + alpha_m;
        // gfx1100 has 96 CUs: the 1,281-row DeltaNet shape exposes only
        // 13.3 independent waves/CU, while the 12,352-row shape already has
        // ample scheduler freedom. Keep the cooperative split restricted to
        // the genuinely underfilled shape instead of paying its LDS join on
        // every QKVZA projection.
        let rdna3_2wave =
            self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_qkvza_2wave && total_m <= 2_048;
        let rdna3_wavepack4 = self.arch_caps.is_rdna3_dgpu()
            && self.flags.rdna3_hfq4_qkvza_wavepack4
            && total_m <= 2_048;
        let rdna3_ldsx8 =
            self.arch_caps.is_gfx1100() && self.flags.rdna3_hfq4_qkvza_ldsx8 && k == 2_048;
        let rdna3_reduce_chain =
            self.arch_caps.is_gfx1100() && self.flags.rdna3_hfq4_qkvza_reduce_chain;
        let rdna3_k2048 =
            self.arch_caps.is_gfx1100() && self.flags.rdna3_hfq4_qkvza_k2048 && k == 2_048;
        static GFX1151_WEIGHT_BUFFER_LOADS: OnceLock<bool> = OnceLock::new();
        let gfx1151_k2048_buffer = self.arch_caps.is_gfx1151()
            && k == 2_048
            && *GFX1151_WEIGHT_BUFFER_LOADS.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_WEIGHT_BUFFER_LOADS").as_deref()
                    == Ok("1")
                    || hipfire_config::developer_var("HIPFIRE_GFX1151_WEIGHT_BUFFER_QKVZA")
                        .as_deref()
                        == Ok("1")
            });
        let gfx1151_k2048_all_buffer = self.arch_caps.is_gfx1151() && k == 2_048;
        let gfx1151_k2048_hybrid_buffer =
            self.arch_caps.is_gfx1151() && k == 2_048 && total_m == 1_281;
        static GFX1151_QKVZA_X_BUFFER_LARGE: OnceLock<bool> = OnceLock::new();
        let gfx1151_k2048_x_buffer_large = self.arch_caps.is_gfx1151()
            && k == 2_048
            && total_m > 2_048
            && *GFX1151_QKVZA_X_BUFFER_LARGE.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_QKVZA_X_BUFFER_LARGE").as_deref()
                    == Ok("1")
            });
        static GFX1151_QKVZA_ALL_BUFFER_CPOL: OnceLock<String> = OnceLock::new();
        let gfx1151_k2048_all_buffer_cpol = if self.arch_caps.is_gfx1151() && k == 2_048 {
            GFX1151_QKVZA_ALL_BUFFER_CPOL
                .get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_GFX1151_QKVZA_ALL_BUFFER_CPOL")
                        .unwrap_or_default()
                        .to_ascii_lowercase()
                })
                .as_str()
        } else {
            ""
        };
        static GFX1151_QKVZA_LDSX8_BUFFER: OnceLock<bool> = OnceLock::new();
        let gfx1151_k2048_ldsx8_buffer = self.arch_caps.is_gfx1151()
            && k == 2_048
            && *GFX1151_QKVZA_LDSX8_BUFFER.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_QKVZA_LDSX8_BUFFER").as_deref()
                    == Ok("1")
            });
        static GFX1151_QKVZA_PAIR_BUFFER: OnceLock<bool> = OnceLock::new();
        let gfx1151_k2048_pair_buffer = self.arch_caps.is_gfx1151()
            && k == 2_048
            && *GFX1151_QKVZA_PAIR_BUFFER.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_QKVZA_PAIR_BUFFER").as_deref()
                    == Ok("1")
            });
        static GFX1151_QKVZA_K2048_HOIST: OnceLock<bool> = OnceLock::new();
        let gfx1151_k2048_hoist = self.arch_caps.is_gfx1151()
            && k == 2_048
            && *GFX1151_QKVZA_K2048_HOIST.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_QKVZA_K2048_HOIST").as_deref()
                    == Ok("1")
            });
        static GFX1151_QKVZA_R2: OnceLock<bool> = OnceLock::new();
        let gfx1151_k2048_r2 = self.arch_caps.is_gfx1151()
            && k == 2_048
            && *GFX1151_QKVZA_R2.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_QKVZA_R2").as_deref() == Ok("1")
            });
        static GFX1151_QKVZA_R2_BUFFER: OnceLock<bool> = OnceLock::new();
        let gfx1151_k2048_r2_buffer = self.arch_caps.is_gfx1151()
            && k == 2_048
            && *GFX1151_QKVZA_R2_BUFFER.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_QKVZA_R2_BUFFER").as_deref()
                    == Ok("1")
            });
        static GFX1151_QKVZA_R4_STREAM: OnceLock<bool> = OnceLock::new();
        let gfx1151_k2048_r4_stream = self.arch_caps.is_gfx1151()
            && k == 2_048
            && total_m > 2_048
            && qkv_m.is_multiple_of(4)
            && z_m.is_multiple_of(4)
            && beta_m.is_multiple_of(4)
            && alpha_m.is_multiple_of(4)
            && *GFX1151_QKVZA_R4_STREAM.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_QKVZA_R4_STREAM").as_deref()
                    == Ok("1")
            });
        static GFX1151_QKVZA_WAVE64_SHARE_X: OnceLock<bool> = OnceLock::new();
        let gfx1151_wave64_share_x = self.arch_caps.is_gfx1151()
            && k == 2_048
            && *GFX1151_QKVZA_WAVE64_SHARE_X.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_GFX1151_QKVZA_WAVE64_SHARE_X").as_deref()
                    == Ok("1")
            });
        static QKVZA_R2: OnceLock<bool> = OnceLock::new();
        let rdna3_k2048_r2 = rdna3_k2048
            && *QKVZA_R2.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_RDNA3_QKVZA_R2").as_deref() == Ok("1")
            });
        static QKVZA_CPOL_SLC: OnceLock<bool> = OnceLock::new();
        let rdna3_k2048_cpol_slc = rdna3_k2048
            && *QKVZA_CPOL_SLC.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_QKVZA_CPOL").as_deref() == Ok("slc")
            });
        let cdna_wave64 = self.arch_caps.is_wave64_native()
            || (self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_qkv_wave64);
        let (func_name, block, grid_x) = if rdna3_ldsx8 {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_ldsx8_gfx1100",
                kernels::FUSED_QKVZA_HFQ4G256_LDSX8_GFX1100_SRC,
                "fused_qkvza_hfq4g256_ldsx8",
            )?;
            (
                "fused_qkvza_hfq4g256_ldsx8",
                [256u32, 1, 1],
                (total_m as u32).div_ceil(8),
            )
        } else if rdna3_2wave {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_2wave",
                kernels::FUSED_QKVZA_HFQ4G256_2WAVE_GFX1100_SRC,
                "fused_qkvza_hfq4g256_2wave",
            )?;
            ("fused_qkvza_hfq4g256_2wave", [64u32, 1, 1], total_m as u32)
        } else if rdna3_wavepack4 {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_wavepack4",
                kernels::FUSED_QKVZA_HFQ4G256_WAVEPACK4_GFX1100_SRC,
                "fused_qkvza_hfq4g256_wavepack4",
            )?;
            (
                "fused_qkvza_hfq4g256_wavepack4",
                [128u32, 1, 1],
                (total_m as u32).div_ceil(4),
            )
        } else if rdna3_reduce_chain {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_reduce_chain_gfx1100",
                kernels::FUSED_QKVZA_HFQ4G256_REDUCE_CHAIN_GFX1100_SRC,
                "fused_qkvza_hfq4g256_reduce_chain",
            )?;
            (
                "fused_qkvza_hfq4g256_reduce_chain",
                [32u32, 1, 1],
                total_m as u32,
            )
        } else if rdna3_k2048_r2 {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_k2048_r2_gfx1100",
                kernels::FUSED_QKVZA_HFQ4G256_K2048_R2_GFX1100_SRC,
                "fused_qkvza_hfq4g256_k2048_r2",
            )?;
            (
                "fused_qkvza_hfq4g256_k2048_r2",
                [32u32, 1, 1],
                (total_m as u32).div_ceil(2),
            )
        } else if rdna3_k2048_cpol_slc {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_k2048_cpol_slc_gfx1100",
                kernels::FUSED_QKVZA_HFQ4G256_K2048_CPOL_SLC_GFX1100_SRC,
                "fused_qkvza_hfq4g256_k2048_cpol_slc",
            )?;
            (
                "fused_qkvza_hfq4g256_k2048_cpol_slc",
                [32u32, 1, 1],
                total_m as u32,
            )
        } else if rdna3_k2048 {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_k2048_gfx1100",
                kernels::FUSED_QKVZA_HFQ4G256_K2048_GFX1100_SRC,
                "fused_qkvza_hfq4g256_k2048",
            )?;
            ("fused_qkvza_hfq4g256_k2048", [32u32, 1, 1], total_m as u32)
        } else if gfx1151_wave64_share_x {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_wave64_share_x_gfx1151",
                kernels::FUSED_QKVZA_HFQ4G256_WAVE64_SHARE_X_GFX1151_SRC,
                "fused_qkvza_hfq4g256_wave64_share_x_gfx1151",
            )?;
            (
                "fused_qkvza_hfq4g256_wave64_share_x_gfx1151",
                [64u32, 1, 1],
                (total_m as u32).div_ceil(2),
            )
        } else if gfx1151_k2048_r4_stream {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_k2048_r4_stream_gfx1151",
                kernels::FUSED_QKVZA_HFQ4G256_K2048_R4_STREAM_GFX1151_SRC,
                "fused_qkvza_hfq4g256_k2048_r4_stream_gfx1151",
            )?;
            (
                "fused_qkvza_hfq4g256_k2048_r4_stream_gfx1151",
                [32u32, 1, 1],
                (total_m as u32).div_ceil(4),
            )
        } else if gfx1151_k2048_x_buffer_large {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_k2048_x_buffer_gfx1151",
                kernels::FUSED_QKVZA_HFQ4G256_K2048_X_BUFFER_GFX1151_SRC,
                "fused_qkvza_hfq4g256_k2048_x_buffer_gfx1151",
            )?;
            (
                "fused_qkvza_hfq4g256_k2048_x_buffer_gfx1151",
                [32u32, 1, 1],
                total_m as u32,
            )
        } else if gfx1151_k2048_ldsx8_buffer {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_ldsx8_buffer_gfx1151",
                kernels::FUSED_QKVZA_HFQ4G256_LDSX8_BUFFER_GFX1151_SRC,
                "fused_qkvza_hfq4g256_ldsx8_buffer_gfx1151",
            )?;
            (
                "fused_qkvza_hfq4g256_ldsx8_buffer_gfx1151",
                [256u32, 1, 1],
                (total_m as u32).div_ceil(8),
            )
        } else if matches!(gfx1151_k2048_all_buffer_cpol, "glc" | "slc" | "dlc") {
            let (module, source, function) = match gfx1151_k2048_all_buffer_cpol {
                "glc" => (
                    "fused_qkvza_hfq4g256_k2048_all_buffer_glc_gfx1151",
                    kernels::FUSED_QKVZA_HFQ4G256_K2048_ALL_BUFFER_GLC_GFX1151_SRC,
                    "fused_qkvza_hfq4g256_k2048_all_buffer_glc_gfx1151",
                ),
                "slc" => (
                    "fused_qkvza_hfq4g256_k2048_all_buffer_slc_gfx1151",
                    kernels::FUSED_QKVZA_HFQ4G256_K2048_ALL_BUFFER_SLC_GFX1151_SRC,
                    "fused_qkvza_hfq4g256_k2048_all_buffer_slc_gfx1151",
                ),
                "dlc" => (
                    "fused_qkvza_hfq4g256_k2048_all_buffer_dlc_gfx1151",
                    kernels::FUSED_QKVZA_HFQ4G256_K2048_ALL_BUFFER_DLC_GFX1151_SRC,
                    "fused_qkvza_hfq4g256_k2048_all_buffer_dlc_gfx1151",
                ),
                _ => unreachable!(),
            };
            self.ensure_kernel(module, source, function)?;
            (function, [32u32, 1, 1], total_m as u32)
        } else if gfx1151_k2048_hybrid_buffer {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_k2048_hybrid_buffer_gfx1151",
                kernels::FUSED_QKVZA_HFQ4G256_K2048_HYBRID_BUFFER_GFX1151_SRC,
                "fused_qkvza_hfq4g256_k2048_hybrid_buffer_gfx1151",
            )?;
            (
                "fused_qkvza_hfq4g256_k2048_hybrid_buffer_gfx1151",
                [32u32, 1, 1],
                total_m as u32,
            )
        } else if gfx1151_k2048_all_buffer && !gfx1151_k2048_pair_buffer {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151",
                kernels::FUSED_QKVZA_HFQ4G256_K2048_ALL_BUFFER_GFX1151_SRC,
                "fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151",
            )?;
            (
                "fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151",
                [32u32, 1, 1],
                total_m as u32,
            )
        } else if gfx1151_k2048_r2_buffer {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_k2048_r2_buffer_gfx1151",
                kernels::FUSED_QKVZA_HFQ4G256_K2048_R2_BUFFER_GFX1151_SRC,
                "fused_qkvza_hfq4g256_k2048_r2",
            )?;
            (
                "fused_qkvza_hfq4g256_k2048_r2",
                [32u32, 1, 1],
                (total_m as u32).div_ceil(2),
            )
        } else if gfx1151_k2048_pair_buffer {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_k2048_pair_buffer_gfx1151",
                kernels::FUSED_QKVZA_HFQ4G256_K2048_PAIR_BUFFER_GFX1151_SRC,
                "fused_qkvza_hfq4g256_k2048_pair_buffer_gfx1151",
            )?;
            (
                "fused_qkvza_hfq4g256_k2048_pair_buffer_gfx1151",
                [32u32, 1, 1],
                total_m as u32,
            )
        } else if gfx1151_k2048_r2 {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_k2048_r2_gfx1151",
                kernels::FUSED_QKVZA_HFQ4G256_K2048_R2_GFX1100_SRC,
                "fused_qkvza_hfq4g256_k2048_r2",
            )?;
            (
                "fused_qkvza_hfq4g256_k2048_r2",
                [32u32, 1, 1],
                (total_m as u32).div_ceil(2),
            )
        } else if gfx1151_k2048_hoist {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_k2048_hoist_x32_gfx1151",
                kernels::FUSED_QKVZA_HFQ4G256_K2048_HOIST_X32_GFX1151_SRC,
                "fused_qkvza_hfq4g256_k2048_hoist_x32_gfx1151",
            )?;
            (
                "fused_qkvza_hfq4g256_k2048_hoist_x32_gfx1151",
                [32u32, 1, 1],
                total_m as u32,
            )
        } else if gfx1151_k2048_buffer {
            self.ensure_kernel(
                "fused_qkvza_hfq4g256_k2048_buffer_gfx1151",
                kernels::FUSED_QKVZA_HFQ4G256_K2048_BUFFER_GFX1151_SRC,
                "fused_qkvza_hfq4g256_k2048_buffer_gfx1151",
            )?;
            (
                "fused_qkvza_hfq4g256_k2048_buffer_gfx1151",
                [32u32, 1, 1],
                total_m as u32,
            )
        } else if cdna_wave64 {
            // gfx94x v2: 2 wave64s = 4 rows/WG, +1.9% on AR decode
            // (commit 5bd75a69 sibling). Default ON; opt out via
            // HIPFIRE_GFX942_GEMV_V2=0.
            let is_gfx94x = self.arch_caps.is_cdna3();
            let v2_on = self.flags.gfx942_gemv_v2.unwrap_or(true);
            if is_gfx94x && v2_on {
                self.ensure_kernel(
                    "fused_qkvza_hfq4g256_v2_gfx942",
                    kernels::FUSED_QKVZA_HFQ4G256_V2_GFX942_SRC,
                    "fused_qkvza_hfq4g256_v2_gfx942",
                )?;
                let total = total_m as u32;
                (
                    "fused_qkvza_hfq4g256_v2_gfx942",
                    [128u32, 1, 1],
                    (total + 3) / 4,
                )
            } else {
                self.ensure_kernel(
                    "fused_qkvza_hfq4g256_wave64",
                    kernels::FUSED_QKVZA_HFQ4G256_WAVE64_SRC,
                    "fused_qkvza_hfq4g256_wave64",
                )?;
                let total = total_m as u32;
                (
                    "fused_qkvza_hfq4g256_wave64",
                    [64u32, 1, 1],
                    (total + 1) / 2,
                )
            }
        } else {
            let (module, source) =
                if self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_qkvza_hoist_x32 {
                    (
                        "fused_qkvza_hfq4g256_hoist_x32_gfx1100",
                        kernels::FUSED_QKVZA_HFQ4G256_HOIST_X32_GFX1100_SRC,
                    )
                } else {
                    ("fused_qkvza_hfq4g256", kernels::FUSED_QKVZA_HFQ4G256_SRC)
                };
            self.ensure_kernel(module, source, "fused_qkvza_hfq4g256")?;
            ("fused_qkvza_hfq4g256", [32u32, 1, 1], total_m as u32)
        };
        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let q_m_i = qkv_m as i32;
        let z_m_i = z_m as i32;
        let b_m_i = beta_m as i32;
        let a_m_i = alpha_m as i32;
        let k_i = k as i32;

        let grid = [grid_x, 1, 1];

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qkvza_hfq4g256", bytes);

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &az as *const _ as *mut c_void,
            &ab as *const _ as *mut c_void,
            &aa as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yz as *const _ as *mut c_void,
            &yb as *const _ as *mut c_void,
            &ya as *const _ as *mut c_void,
            &q_m_i as *const _ as *mut c_void,
            &z_m_i as *const _ as *mut c_void,
            &b_m_i as *const _ as *mut c_void,
            &a_m_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        let result = self.launch_maybe_blob(func_name, grid, block, 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(aq);
            b.push_ptr(az);
            b.push_ptr(ab);
            b.push_ptr(aa);
            b.push_ptr(xp);
            b.push_ptr(yq);
            b.push_ptr(yz);
            b.push_ptr(yb);
            b.push_ptr(ya);
            b.push_i32(q_m_i);
            b.push_i32(z_m_i);
            b.push_i32(b_m_i);
            b.push_i32(a_m_i);
            b.push_i32(k_i);
            b
        });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx1100/K=2048 QKVZA experiment that also prepares the DeltaNet beta
    /// and alpha scalars. The projection FMAs and reduction are identical to
    /// `fused_qkvza_hfq4g256`; only the two tiny output tails absorb the
    /// elementwise work from `fused_sigmoid_alpha_gate_f32`.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkvza_hfq4g256_scalar_prep_gfx1100(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        dt_bias: &GpuTensor,
        a_log: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1100() || k != 2_048 {
            return Err(hip_bridge::HipError::new(
                0,
                "scalar-prep QKVZA requires gfx1100 and K=2048",
            ));
        }

        const KERNEL: &str = "fused_qkvza_hfq4g256_k2048_scalar_prep";
        self.ensure_kernel(
            "fused_qkvza_hfq4g256_k2048_scalar_prep_gfx1100",
            kernels::FUSED_QKVZA_HFQ4G256_K2048_SCALAR_PREP_GFX1100_SRC,
            KERNEL,
        )?;

        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let db = dt_bias.buf.as_ptr();
        let al = a_log.buf.as_ptr();
        let q_m_i = qkv_m as i32;
        let z_m_i = z_m as i32;
        let b_m_i = beta_m as i32;
        let a_m_i = alpha_m as i32;
        let k_i = k as i32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
            + (beta_m + alpha_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "fused", KERNEL, bytes);

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &az as *const _ as *mut c_void,
            &ab as *const _ as *mut c_void,
            &aa as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yz as *const _ as *mut c_void,
            &yb as *const _ as *mut c_void,
            &ya as *const _ as *mut c_void,
            &db as *const _ as *mut c_void,
            &al as *const _ as *mut c_void,
            &q_m_i as *const _ as *mut c_void,
            &z_m_i as *const _ as *mut c_void,
            &b_m_i as *const _ as *mut c_void,
            &a_m_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        let grid = [(qkv_m + z_m + beta_m + alpha_m) as u32, 1, 1];
        let result = self.launch_maybe_blob(KERNEL, grid, [32, 1, 1], 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(aq);
            b.push_ptr(az);
            b.push_ptr(ab);
            b.push_ptr(aa);
            b.push_ptr(xp);
            b.push_ptr(yq);
            b.push_ptr(yz);
            b.push_ptr(yb);
            b.push_ptr(ya);
            b.push_ptr(db);
            b.push_ptr(al);
            b.push_i32(q_m_i);
            b.push_i32(z_m_i);
            b.push_i32(b_m_i);
            b.push_i32(a_m_i);
            b.push_i32(k_i);
            b
        });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// dp4a-port of fused_qkvza_hfq4g256 for gfx906. Pre-quantizes x to
    /// Q8_1 via the shared MMQ scratch, then runs the dp4a-based GEMV.
    /// Math is identical modulo Q8_1 quant noise. Targets gfx906's
    /// memory-bound regime per the per-kernel PMC pass at 2026-05-05.
    pub fn fused_qkvza_hfq4g256_dp4a(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_qkvza_hfq4g256_wave64_dp4a",
            kernels::FUSED_QKVZA_HFQ4G256_WAVE64_DP4A_SRC,
            "fused_qkvza_hfq4g256_wave64_dp4a",
        )?;

        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let q_m_i = qkv_m as i32;
        let z_m_i = z_m as i32;
        let b_m_i = beta_m as i32;
        let a_m_i = alpha_m as i32;
        let k_i = k as i32;
        let total = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let mut xq = xq_ptr;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k);
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_qkvza_hfq4g256_dp4a", bytes);

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &az as *const _ as *mut c_void,
            &ab as *const _ as *mut c_void,
            &aa as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yz as *const _ as *mut c_void,
            &yb as *const _ as *mut c_void,
            &ya as *const _ as *mut c_void,
            &q_m_i as *const _ as *mut c_void,
            &z_m_i as *const _ as *mut c_void,
            &b_m_i as *const _ as *mut c_void,
            &a_m_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        let result = self.launch_maybe_blob(
            "fused_qkvza_hfq4g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xq);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m_i);
                b.push_i32(z_m_i);
                b.push_i32(b_m_i);
                b.push_i32(a_m_i);
                b.push_i32(k_i);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched 4-way fused HFQ4-G256 GEMM for the LA preamble.
    ///
    /// Processes N tokens × four projections (wqkv + wz + w_beta + w_alpha)
    /// in one launch. Bitwise-identical output to calling `fused_qkvza_hfq4g256`
    /// N times on the same x[b] — 4-accumulator interleave + pairwise combine
    /// are preserved per batch element.
    ///
    /// `x`: [N × K] row-major activation batch.
    /// `y_*`: [N × *_m] row-major outputs (overwrite semantics).
    pub fn gemm_qkvza_hfq4g256(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // CDNA3 MFMA path — 4 back-to-back rocBLAS calls. The last two
        // matrices (beta, alpha) are tiny (n_v_heads = 128 on A3B) so we
        // could skip them and stay on the GEMV path, but dispatching all
        // four via rocBLAS keeps the codepath uniform. Amortizes well.
        if self.rocblas_arch_eligible()
            && batch_size >= self.rocblas_min_batch()
            && self.rocblas.is_some()
            && !self.graphs.capture_mode
        {
            let shadow_qkv = self.ensure_fp16_shadow(a_qkv, qkv_m, k)?;
            let shadow_z = self.ensure_fp16_shadow(a_z, z_m, k)?;
            let shadow_beta = self.ensure_fp16_shadow(a_beta, beta_m, k)?;
            let shadow_alpha = self.ensure_fp16_shadow(a_alpha, alpha_m, k)?;
            if let (Some(pq), Some(pz), Some(pb), Some(pa)) =
                (shadow_qkv, shadow_z, shadow_beta, shadow_alpha)
            {
                let x_fp16 = self.ensure_fp16_x(x, batch_size * k)?;
                let xb = unsafe { DeviceBuffer::from_raw(x_fp16, (batch_size * k) * 2) };
                let wq = unsafe { DeviceBuffer::from_raw(pq, (qkv_m * k) * 2) };
                let wz_b = unsafe { DeviceBuffer::from_raw(pz, (z_m * k) * 2) };
                let wb = unsafe { DeviceBuffer::from_raw(pb, (beta_m * k) * 2) };
                let wa = unsafe { DeviceBuffer::from_raw(pa, (alpha_m * k) * 2) };
                let timer = crate::profile::begin_timer(
                    &self.hip,
                    "gemm",
                    "gemm_qkvza_hfq4g256_rocblas",
                    crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                        + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                        + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                        + crate::profile::gemv_hfq4g256_bytes(alpha_m, k),
                );
                let r1 = self.rocblas_gemm_hfq4_prefill(&wq, &xb, &y_qkv.buf, qkv_m, batch_size, k);
                let r2 = if r1.is_ok() {
                    self.rocblas_gemm_hfq4_prefill(&wz_b, &xb, &y_z.buf, z_m, batch_size, k)
                } else {
                    Ok(())
                };
                let r3 = if r2.is_ok() {
                    self.rocblas_gemm_hfq4_prefill(&wb, &xb, &y_beta.buf, beta_m, batch_size, k)
                } else {
                    Ok(())
                };
                let r4 = if r3.is_ok() {
                    self.rocblas_gemm_hfq4_prefill(&wa, &xb, &y_alpha.buf, alpha_m, batch_size, k)
                } else {
                    Ok(())
                };
                std::mem::forget(xb);
                std::mem::forget(wq);
                std::mem::forget(wz_b);
                std::mem::forget(wb);
                std::mem::forget(wa);
                if let Some(t) = timer {
                    t.finish(&self.hip);
                }
                return r1.and(r2).and(r3).and(r4);
            }
        }
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !self.flags.fp16_disabled {
            // Wave64 FP16 hybrid — best of both worlds for gfx906 (MI50).
            if self.arch_caps.is_gcn5_wave64() {
                // gfx906 dp4a MMQ split: qkv + z route through the new MMQ
                // kernel (large-M outputs); beta + alpha keep the fused
                // wave64 kernel because their M (=linear_num_value_heads,
                // typically 32) is far below MMQ_Y=128 — bounds-checked
                // MMQ would waste ~75% of each row-tile.
                //
                // The fused wave64 kernel accepts qkv_m=0, z_m=0 to handle
                // the beta+alpha tail alone (its row-routing logic skips
                // the qkv/z branches when those Ms are zero). See
                // kernels/src/gemm_qkvza_hfq4g256_fp16_wave64.hip:54-61.
                //
                // Routes through MMQ at batch_size ≥ 16 (per
                // should_use_mmq's gfx906 default). Falls through to the
                // fused wave64 if any of qkv/z screening rejects (matches
                // gate_up's behavior in gemm_gate_up_hfq4g256).
                // gfx906 MMQ split — qkv + z through MMQ (large-M), beta + alpha
                // through a fused-projection kernel (tail M typically 32, below
                // MMQ_Y=128). Distinguishes two reasons MMQ might not fire:
                //   (a) batch_size below cutover → fall to dp4a 4-way fused
                //   (b) qkv or z screening rejected → fall to fp16 4-way fused
                //       (screen-reject path preserves higher-precision intent;
                //       dp4a shares Q8_1 quant step that MMQ failed on).
                let mut mmq_screen_rejected = false;
                if self.arch_caps.is_gfx906() && self.arch_caps.should_use_mmq(batch_size) {
                    let qz_safe = if self.mmq_screen.enabled {
                        self.mmq_screen_weight(a_qkv, qkv_m, k)
                            && self.mmq_screen_weight(a_z, z_m, k)
                    } else {
                        true
                    };
                    if qz_safe {
                        let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                        let (r1, r2) = if qkv_m % 128 == 0 && z_m % 128 == 0 {
                            (
                                self.gemm_gate_up_hfq4g256_mmq_gfx906_prequant(
                                    a_qkv, a_z, xq, y_qkv, y_z, qkv_m, z_m, k, batch_size,
                                ),
                                Ok(()),
                            )
                        } else {
                            let r_qkv = self.gemm_hfq4g256_mmq_set_gfx906(
                                a_qkv, xq, y_qkv, qkv_m, k, batch_size,
                            );
                            let r_z = if r_qkv.is_ok() {
                                self.gemm_hfq4g256_mmq_set_gfx906(a_z, xq, y_z, z_m, k, batch_size)
                            } else {
                                Ok(())
                            };
                            (r_qkv, r_z)
                        };
                        // Tail: beta+alpha. Use dp4a-prequant when available
                        // (reuses the Q8_1 scratch we just produced above, no
                        // re-quantize). Falls back to fp16_wave64 in capture
                        // mode (ensure_kernel first-use JIT is unsafe inside
                        // capture; the dp4a kernel may not be compiled yet on
                        // a fresh process).
                        let r3 = if r2.is_ok() {
                            if self.arch_caps.gemv_dp4a_enabled() && !self.graphs.capture_mode {
                                self.gemm_qkvza_hfq4g256_wave64_dp4a_prequant(
                                    a_qkv, a_z, a_beta, a_alpha, xq, y_qkv, y_z, y_beta, y_alpha,
                                    0, 0, beta_m, alpha_m, k, batch_size,
                                )
                            } else {
                                self.gemm_qkvza_hfq4g256_fp16_wave64(
                                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, 0,
                                    0, beta_m, alpha_m, k, batch_size,
                                )
                            }
                        } else {
                            Ok(())
                        };
                        return r1.and(r2).and(r3);
                    }
                    mmq_screen_rejected = self.mmq_screen.enabled;
                    // qkv or z screening rejected — fall through; screen-reject
                    // path goes to fp16, NOT dp4a.
                }
                // gfx906 dp4a 4-way fused (issue #276 Gap 2). Fires when
                // batch_size > 1 below the MMQ cutover or when capture mode
                // prevents MMQ. Skipped on screen-reject to preserve the
                // higher-precision fallback intent.
                if !mmq_screen_rejected
                    && self.arch_caps.gemv_dp4a_enabled()
                    && !self.graphs.capture_mode
                {
                    return self.gemm_qkvza_hfq4g256_wave64_dp4a(
                        a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                        beta_m, alpha_m, k, batch_size,
                    );
                }
                return self.gemm_qkvza_hfq4g256_fp16_wave64(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            if self.arch_caps.should_use_mmq(batch_size) {
                let use_mmq = if self.mmq_screen.enabled {
                    self.mmq_screen_weight(a_qkv, qkv_m, k)
                        && self.mmq_screen_weight(a_z, z_m, k)
                        && self.mmq_screen_weight(a_beta, beta_m, k)
                        && self.mmq_screen_weight(a_alpha, alpha_m, k)
                } else {
                    true
                };
                if use_mmq {
                    let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                    // RDNA3 dGPU opt-in: qkv/z are large aligned projections
                    // where the MMQ set path is profitable, while beta/alpha
                    // are tiny LA tails that waste most of a 128-row MMQ tile.
                    // Keep this off by default and do not widen it to gfx12 or
                    // gfx10; their MMQ/dot2 tradeoffs and source families differ.
                    if self.flags.qkvza_split_tail
                        && self.arch_caps.is_rdna3_dgpu()
                        && qkv_m % 128 == 0
                        && z_m % 128 == 0
                    {
                        let r1 = self
                            .gemm_hfq4g256_mmq_set_prequant(a_qkv, xq, y_qkv, qkv_m, k, batch_size);
                        let r2 = if r1.is_ok() {
                            self.gemm_hfq4g256_mmq_set_prequant(a_z, xq, y_z, z_m, k, batch_size)
                        } else {
                            Ok(())
                        };
                        let r3 = if r2.is_ok() {
                            self.gemm_gate_up_hfq4g256_dot2(
                                a_beta, a_alpha, x, y_beta, y_alpha, beta_m, alpha_m, k, batch_size,
                            )
                        } else {
                            Ok(())
                        };
                        return r1.and(r2).and(r3);
                    }
                    let r1 =
                        self.gemm_hfq4g256_mmq_set_prequant(a_qkv, xq, y_qkv, qkv_m, k, batch_size);
                    let r2 = if r1.is_ok() {
                        self.gemm_hfq4g256_mmq_set_prequant(a_z, xq, y_z, z_m, k, batch_size)
                    } else {
                        Ok(())
                    };
                    let r3 = if r2.is_ok() {
                        self.gemm_hfq4g256_mmq_set_prequant(
                            a_beta, xq, y_beta, beta_m, k, batch_size,
                        )
                    } else {
                        Ok(())
                    };
                    let r4 = if r3.is_ok() {
                        self.gemm_hfq4g256_mmq_set_prequant(
                            a_alpha, xq, y_alpha, alpha_m, k, batch_size,
                        )
                    } else {
                        Ok(())
                    };
                    return r1.and(r2).and(r3).and(r4);
                }
            }
            // HFQ4 wave32 MMQ RDNA2 path (issue #299 Phase 4). Three modes:
            //   (a) all 4 Ms aligned to MMQ_Y=128 → single 4-way fused MMQ kernel
            //   (b) qkv_m and z_m aligned but beta_m/alpha_m not (LinearAttention
            //       β+α are typically tiny, well below 128) → split routing:
            //       2-way gate_up MMQ on (wqkv, wz) + 2-way gate_up dot2 on
            //       (w_beta, w_alpha). Mirrors MQ3 phase-2 finding that
            //       gave +22% prefill on Qwen3.5 LA layers.
            //   (c) something else not aligned → fall through to dot2/wmma.
            if self.arch_caps.has_hfq4_mmq() {
                let all_aligned =
                    qkv_m % 128 == 0 && z_m % 128 == 0 && beta_m % 128 == 0 && alpha_m % 128 == 0;
                if all_aligned {
                    return self.gemm_qkvza_hfq4g256_mmq(
                        a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                        beta_m, alpha_m, k, batch_size,
                    );
                }
                if qkv_m % 128 == 0 && z_m % 128 == 0 {
                    let r1 = self.gemm_gate_up_hfq4g256_mmq(
                        a_qkv, a_z, x, y_qkv, y_z, qkv_m, z_m, k, batch_size,
                    );
                    let r2 = if r1.is_ok() {
                        self.gemm_gate_up_hfq4g256_dot2(
                            a_beta, a_alpha, x, y_beta, y_alpha, beta_m, alpha_m, k, batch_size,
                        )
                    } else {
                        Ok(())
                    };
                    return r1.and(r2);
                }
            }
            if self.arch_caps.has_wmma_w32_gfx12() {
                return self.gemm_qkvza_hfq4g256_wmma_gfx12(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            if self.arch_caps.has_wmma_w32() {
                return self.gemm_qkvza_hfq4g256_wmma(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkvza_hfq4g256_dot2(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_qkvza_hfq4g256_fp16(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemm_qkvza_hfq4g256_wave64",
                kernels::GEMM_QKVZA_HFQ4G256_WAVE64_SRC,
                "gemm_qkvza_hfq4g256_wave64",
            )?;
            ("gemm_qkvza_hfq4g256_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel(
                "gemm_qkvza_hfq4g256",
                kernels::GEMM_QKVZA_HFQ4G256_SRC,
                "gemm_qkvza_hfq4g256",
            )?;
            ("gemm_qkvza_hfq4g256", [32, 1, 1], 1)
        };
        let func = &self.functions[func_name];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let grid_x = (total_m + grid_div - 1) / grid_div;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                block,
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

    /// Batched 4-way fused HFQ3-G256 GEMM for the LA preamble (MQ3 path).
    ///
    /// HFQ3 sibling of `gemm_qkvza_hfq4g256` — single scalar variant only.
    /// Phase 1 of the gfx10 MQ3 prefill plan. Wires the dense Qwen3.5
    /// LA layer's 4-way fused projection (wqkv + wz + w_beta + w_alpha)
    /// onto the batched path; previously gfx10 MQ3 LA fell back to
    /// per-token forward_scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq3g256(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Phase 3 MMQ (auto-tile-selecting). Default-on for gfx10 sdot4 archs
        // (issue #300 MQ3 gate removal; escape hatch HIPFIRE_HFQ3_MMQ=0).
        // Auto-selector falls back to dot2 at small batch. Layer-gate
        // (HIPFIRE_HFQ3_MMQ_LAYER_{MIN,MAX}) is a no-op when unset; supports
        // per-layer KLD attribution sweeps (#302).
        if batch_size > 1 && self.arch_caps.has_hfq3_mmq() && self.flags.hfq3_mmq_layer_gate_pass()
        {
            return self.gemm_qkvza_hfq3g256_mmq(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        // FP16 fast paths — Phase 2b (dot2) + Phase 2c (fp16 fallback).
        // Layer-aware FP16 gate (#302): falls through to scalar when the
        // current layer falls in HIPFIRE_FP16_LAYER_MIN..=MAX. No-op when
        // those env vars are unset.
        if batch_size > 1 && !self.flags.fp16_disabled_for_current_layer() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkvza_hfq3g256_dot2(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            return self.gemm_qkvza_hfq3g256_fp16(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_qkvza_hfq3g256",
            kernels::GEMM_QKVZA_HFQ3G256_SRC,
            "gemm_qkvza_hfq3g256",
        )?;
        let func = &self.functions["gemm_qkvza_hfq3g256"];

        let mut aqkv = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yqkv = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aqkv as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yqkv as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let bytes = crate::profile::gemm_hfq3g256_bytes(qkv_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(z_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(beta_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(alpha_m, k, batch_size);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq3g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// v_dot2_f32_f16-accelerated batched 4-way fused HFQ3-G256 GEMM (qkv + z + beta + alpha).
    /// RDNA2 (gfx1011/1012/1030-1032) + RDNA3/4 fast path; HFQ3 sibling of
    /// `gemm_qkvza_hfq4g256_dot2`. Phase 2b of the gfx10 MQ3 prefill plan.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq3g256_dot2(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq3g256_dot2",
            kernels::GEMM_QKVZA_HFQ3G256_DOT2_SRC,
            "gemm_qkvza_hfq3g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq3g256_dot2"];

        let mut aqkv = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yqkv = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aqkv as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yqkv as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let bytes = crate::profile::gemm_hfq3g256_bytes(qkv_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(z_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(beta_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(alpha_m, k, batch_size)
            + batch_size * k * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq3g256_dot2", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// v_pk_fma_f16-accelerated batched 4-way fused HFQ3-G256 GEMM.
    /// Fallback for archs without the dot extension (gfx1010, gfx1013).
    /// Phase 2c of the gfx10 MQ3 prefill plan.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq3g256_fp16(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq3g256_fp16",
            kernels::GEMM_QKVZA_HFQ3G256_FP16_SRC,
            "gemm_qkvza_hfq3g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq3g256_fp16"];

        let mut aqkv = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yqkv = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aqkv as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yqkv as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let bytes = crate::profile::gemm_hfq3g256_bytes(qkv_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(z_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(beta_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(alpha_m, k, batch_size)
            + batch_size * k * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq3g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// FP16-packed batched 4-way fused HFQ4-G256 GEMM (qkv + z + beta + alpha).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq4g256_fp16(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq4g256_fp16",
            kernels::GEMM_QKVZA_HFQ4G256_FP16_SRC,
            "gemm_qkvza_hfq4g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq4g256_fp16"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (qkv_m + z_m + beta_m + alpha_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// Wave64 FP16 hybrid batched 4-way fused HFQ4-G256 GEMM (qkv + z + beta + alpha).
    /// Combines wave64 block structure (2 rows/block, full lane utilization) with
    /// FP16 packed arithmetic (__hfma2). Target: gfx906 (MI50) prefill optimization.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq4g256_fp16_wave64(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq4g256_fp16_wave64",
            kernels::GEMM_QKVZA_HFQ4G256_FP16_WAVE64_SRC,
            "gemm_qkvza_hfq4g256_fp16_wave64",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq4g256_fp16_wave64"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let grid_x = (total_m + 1) / 2;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (qkv_m + z_m + beta_m + alpha_m) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkvza_hfq4g256_fp16_wave64",
            bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                [64, 1, 1],
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

    /// v_dot2_f32_f16-accelerated batched 4-way fused HFQ4-G256 GEMM (qkv + z + beta + alpha).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `amd_mixed_dot`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq4g256_dot2(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq4g256_dot2",
            kernels::GEMM_QKVZA_HFQ4G256_DOT2_SRC,
            "gemm_qkvza_hfq4g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq4g256_dot2"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
            + batch_size * k * 2
            + batch_size * (qkv_m + z_m + beta_m + alpha_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256_dot2", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// Batched 3-way fused HFQ4-G256 GEMM for the FA preamble.
    ///
    /// Processes N tokens × three projections (wq + wk + wv) in one launch.
    /// Bitwise-identical to calling `fused_qkv_hfq4g256` N times on the same
    /// x[b] — 4-accumulator interleave + pairwise combine preserved per
    /// batch element.
    pub fn gemm_qkv_hfq4g256(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // CDNA3 MFMA path — 3 back-to-back rocBLAS calls for Q, K, V.
        if self.rocblas_arch_eligible()
            && batch_size >= self.rocblas_min_batch()
            && self.rocblas.is_some()
            && !self.graphs.capture_mode
        {
            let sq = self.ensure_fp16_shadow(a_q, q_m, k)?;
            let sk = self.ensure_fp16_shadow(a_k, k_m, k)?;
            let sv = self.ensure_fp16_shadow(a_v, v_m, k)?;
            if let (Some(pq), Some(pk), Some(pv)) = (sq, sk, sv) {
                let x_fp16 = self.ensure_fp16_x(x, batch_size * k)?;
                let xb = unsafe { DeviceBuffer::from_raw(x_fp16, (batch_size * k) * 2) };
                let wq = unsafe { DeviceBuffer::from_raw(pq, (q_m * k) * 2) };
                let wk = unsafe { DeviceBuffer::from_raw(pk, (k_m * k) * 2) };
                let wv = unsafe { DeviceBuffer::from_raw(pv, (v_m * k) * 2) };
                let timer = crate::profile::begin_timer(
                    &self.hip,
                    "gemm",
                    "gemm_qkv_hfq4g256_rocblas",
                    crate::profile::gemv_hfq4g256_bytes(q_m, k)
                        + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                        + crate::profile::gemv_hfq4g256_bytes(v_m, k),
                );
                let r1 = self.rocblas_gemm_hfq4_prefill(&wq, &xb, &y_q.buf, q_m, batch_size, k);
                let r2 = if r1.is_ok() {
                    self.rocblas_gemm_hfq4_prefill(&wk, &xb, &y_k.buf, k_m, batch_size, k)
                } else {
                    Ok(())
                };
                let r3 = if r2.is_ok() {
                    self.rocblas_gemm_hfq4_prefill(&wv, &xb, &y_v.buf, v_m, batch_size, k)
                } else {
                    Ok(())
                };
                std::mem::forget(xb);
                std::mem::forget(wq);
                std::mem::forget(wk);
                std::mem::forget(wv);
                if let Some(t) = timer {
                    t.finish(&self.hip);
                }
                return r1.and(r2).and(r3);
            }
        }
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !self.flags.fp16_disabled {
            // Wave64 FP16 hybrid — best of both worlds for gfx906 (MI50).
            if self.arch_caps.is_gcn5_wave64() {
                // gfx906 dp4a MMQ: route q+k+v through the new MMQ kernel.
                // Unlike qkvza, all three qkv outputs have M well above
                // MMQ_Y=128 (Qwen 9B full-attn: q_m=4096, k_m=v_m=1024),
                // so no tail kernel is needed — straight 3× MMQ-set.
                //
                // Routes through MMQ at batch_size ≥ 16 (per
                // should_use_mmq's gfx906 default). Falls through to the
                // fused wave64 if any of q/k/v screening rejects.
                let mut mmq_screen_rejected = false;
                if self.arch_caps.is_gfx906() && self.arch_caps.should_use_mmq(batch_size) {
                    let qkv_safe = if self.mmq_screen.enabled {
                        self.mmq_screen_weight(a_q, q_m, k)
                            && self.mmq_screen_weight(a_k, k_m, k)
                            && self.mmq_screen_weight(a_v, v_m, k)
                    } else {
                        true
                    };
                    if qkv_safe {
                        if q_m % 128 == 0 && k_m % 128 == 0 && v_m % 128 == 0 {
                            return self.gemm_qkv_hfq4g256_mmq_gfx906(
                                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                            );
                        }
                        let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                        let r1 =
                            self.gemm_hfq4g256_mmq_set_gfx906(a_q, xq, y_q, q_m, k, batch_size);
                        let r2 = if r1.is_ok() {
                            self.gemm_hfq4g256_mmq_set_gfx906(a_k, xq, y_k, k_m, k, batch_size)
                        } else {
                            Ok(())
                        };
                        let r3 = if r2.is_ok() {
                            self.gemm_hfq4g256_mmq_set_gfx906(a_v, xq, y_v, v_m, k, batch_size)
                        } else {
                            Ok(())
                        };
                        return r1.and(r2).and(r3);
                    }
                    mmq_screen_rejected = self.mmq_screen.enabled;
                    // q/k/v screening rejected — fall through; screen-reject
                    // path goes to fp16, NOT dp4a (preserves the screen's
                    // higher-precision fallback intent).
                }
                // gfx906 dp4a 3-way fused (issue #276 Gap 2). Fires when
                // batch_size > 1 below the MMQ cutover or in capture mode.
                // Skipped on screen-reject (dp4a shares Q8_1 quant step with
                // MMQ; routing rejected weights here would defeat the screen).
                if !mmq_screen_rejected
                    && self.arch_caps.gemv_dp4a_enabled()
                    && !self.graphs.capture_mode
                {
                    return self.gemm_qkv_hfq4g256_wave64_dp4a(
                        a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                    );
                }
                return self.gemm_qkv_hfq4g256_fp16_wave64(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            if self.arch_caps.should_use_mmq(batch_size) {
                let use_mmq = if self.mmq_screen.enabled {
                    self.mmq_screen_weight(a_q, q_m, k)
                        && self.mmq_screen_weight(a_k, k_m, k)
                        && self.mmq_screen_weight(a_v, v_m, k)
                } else {
                    true
                };
                if use_mmq {
                    let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                    let r1 = self.gemm_hfq4g256_mmq_set_prequant(a_q, xq, y_q, q_m, k, batch_size);
                    let r2 = if r1.is_ok() {
                        self.gemm_hfq4g256_mmq_set_prequant(a_k, xq, y_k, k_m, k, batch_size)
                    } else {
                        Ok(())
                    };
                    let r3 = if r2.is_ok() {
                        self.gemm_hfq4g256_mmq_set_prequant(a_v, xq, y_v, v_m, k, batch_size)
                    } else {
                        Ok(())
                    };
                    return r1.and(r2).and(r3);
                }
            }
            // HFQ4 wave32 MMQ RDNA2 path (issue #299 Phase 2). Routes
            // ahead of dot2/wmma fallbacks; default-on for the allowlist
            // arch set (issue #300 gate removal, escape hatch
            // HIPFIRE_HFQ4_MMQ_RDNA2=0). All q_m/k_m/v_m for the Qwen3.5
            // family are MMQ_Y(128)-aligned.
            if self.arch_caps.has_hfq4_mmq() && q_m % 128 == 0 && k_m % 128 == 0 && v_m % 128 == 0 {
                return self.gemm_qkv_hfq4g256_mmq(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            if self.arch_caps.has_wmma_w32_gfx12() {
                return self.gemm_qkv_hfq4g256_wmma_gfx12(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            if self.arch_caps.has_wmma_w32() {
                return self.gemm_qkv_hfq4g256_wmma(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkv_hfq4g256_dot2(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_qkv_hfq4g256_fp16(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemm_qkv_hfq4g256_wave64",
                kernels::GEMM_QKV_HFQ4G256_WAVE64_SRC,
                "gemm_qkv_hfq4g256_wave64",
            )?;
            ("gemm_qkv_hfq4g256_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel(
                "gemm_qkv_hfq4g256",
                kernels::GEMM_QKV_HFQ4G256_SRC,
                "gemm_qkv_hfq4g256",
            )?;
            ("gemm_qkv_hfq4g256", [32, 1, 1], 1)
        };
        let func = &self.functions[func_name];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;
        let grid_x = (total_m + grid_div - 1) / grid_div;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                block,
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

    /// Batched 3-way fused HFQ3-G256 GEMM for the FA preamble (MQ3 path).
    ///
    /// HFQ3 sibling of `gemm_qkv_hfq4g256` — same dispatch shape, 104 B
    /// group stride and 3-bit unpack. Single scalar variant only (no
    /// rocBLAS / wave64 / fp16 / dp4a fast paths yet) — Phase 1 of the
    /// gfx10 MQ3 prefill plan. Bitwise-identical to running the
    /// single-row HFQ3 GEMV N times for N=1.
    pub fn gemm_qkv_hfq3g256(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Phase 3 MMQ family (auto-tile-selecting). Default-on for gfx10
        // sdot4 archs (issue #300, escape hatch HIPFIRE_HFQ3_MMQ=0) when
        // q_m/k_m/v_m are MMQ_Y-aligned. The
        // auto-selector itself falls back to dot2 at batch ≤ 12, so it's
        // safe at any batch_size. Layer-gate is a no-op when unset (#302).
        if batch_size > 1
            && self.arch_caps.has_hfq3_mmq()
            && self.flags.hfq3_mmq_layer_gate_pass()
            && q_m % 128 == 0
            && k_m % 128 == 0
            && v_m % 128 == 0
        {
            return self.gemm_qkv_hfq3g256_mmq(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        // Phase 2 experimental: wave32 dp4a if HIPFIRE_HFQ3_DP4A=1.
        if batch_size > 1 && self.arch_caps.has_hfq3_dp4a() {
            return self.gemm_qkv_hfq3g256_dp4a(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        // FP16 fast paths — gfx10xx admits MQ3 via is_batchable_la, all of
        // these archs support FP16 ISA. Phase 2b (dot2) + Phase 2c (fp16).
        // Layer-aware FP16 gate (#302) falls through to scalar when layer
        // in HIPFIRE_FP16_LAYER_MIN..=MAX. No-op when those vars are unset.
        if batch_size > 1 && !self.flags.fp16_disabled_for_current_layer() {
            // v_dot2_f32_f16 on archs with the dot extension
            // (gfx1011/1012/1030-1032, gfx11/12).
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkv_hfq3g256_dot2(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            // v_pk_fma_f16 fallback for gfx1010 (Navi 10 / 5700 XT) and
            // gfx1013 (BC-250 APU), which lack the dot extension but have FP16.
            return self.gemm_qkv_hfq3g256_fp16(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_qkv_hfq3g256",
            kernels::GEMM_QKV_HFQ3G256_SRC,
            "gemm_qkv_hfq3g256",
        )?;
        let func = &self.functions["gemm_qkv_hfq3g256"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemm_hfq3g256_bytes(q_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(k_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(v_m, k, batch_size);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq3g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// v_dot2_f32_f16-accelerated batched 3-way fused HFQ3-G256 GEMM (Q + K + V).
    /// RDNA2 (gfx1011/1012/1030-1032) + RDNA3/4 fast path. HFQ3 sibling of
    /// `gemm_qkv_hfq4g256_dot2` — same dispatch shape, FP16 X via
    /// `ensure_fp16_x`, only the weight unpack differs (104 B/group, uint24
    /// byte-combine, 8 3-bit trits per group per thread). Phase 2b of the
    /// gfx10 MQ3 prefill plan.
    pub fn gemm_qkv_hfq3g256_dot2(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq3g256_dot2",
            kernels::GEMM_QKV_HFQ3G256_DOT2_SRC,
            "gemm_qkv_hfq3g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq3g256_dot2"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemm_hfq3g256_bytes(q_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(k_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(v_m, k, batch_size)
            + batch_size * k * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq3g256_dot2", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// v_pk_fma_f16-accelerated batched 3-way fused HFQ3-G256 GEMM (Q + K + V).
    /// Fallback for archs without the dot extension (gfx1010, gfx1013).
    /// Phase 2c of the gfx10 MQ3 prefill plan.
    pub fn gemm_qkv_hfq3g256_fp16(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq3g256_fp16",
            kernels::GEMM_QKV_HFQ3G256_FP16_SRC,
            "gemm_qkv_hfq3g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq3g256_fp16"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemm_hfq3g256_bytes(q_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(k_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(v_m, k, batch_size)
            + batch_size * k * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq3g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// Wave32+dp4a batched 3-way fused HFQ3-G256 GEMM (Q + K + V).
    /// Phase 2 experimental — port of `gemm_qkv_hfq4g256_wave64_dp4a` from
    /// gfx906 wave64 to wave32 + HFQ3 unpack. Available on the gfx10 sdot4
    /// subset. Gated by `HIPFIRE_HFQ3_DP4A=1`.
    pub fn gemm_qkv_hfq3g256_dp4a(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.has_hfq3_sdot4() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkv_hfq3g256_dot2(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            return self.gemm_qkv_hfq3g256_fp16(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_qkv_hfq3g256_dp4a",
            kernels::GEMM_QKV_HFQ3G256_DP4A_SRC,
            "gemm_qkv_hfq3g256_dp4a",
        )?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions["gemm_qkv_hfq3g256_dp4a"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 16;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemm_hfq3g256_bytes(q_m, k, batch_size)
                  + crate::profile::gemm_hfq3g256_bytes(k_m, k, batch_size)
                  + crate::profile::gemm_hfq3g256_bytes(v_m, k, batch_size)
                  + batch_size * k  // Q8_1 mmq X is ~1 byte per element
                  + batch_size * (q_m + k_m + v_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq3g256_dp4a", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// FP16-packed batched 3-way fused HFQ4-G256 GEMM (Q + K + V).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    pub fn gemm_qkv_hfq4g256_fp16(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq4g256_fp16",
            kernels::GEMM_QKV_HFQ4G256_FP16_SRC,
            "gemm_qkv_hfq4g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq4g256_fp16"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (q_m + k_m + v_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// Wave64 FP16 hybrid batched 3-way fused HFQ4-G256 GEMM (Q + K + V).
    /// Combines wave64 block structure (2 rows/block, full lane utilization) with
    /// FP16 packed arithmetic (__hfma2). Target: gfx906 (MI50) prefill optimization.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq4g256_fp16_wave64(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq4g256_fp16_wave64",
            kernels::GEMM_QKV_HFQ4G256_FP16_WAVE64_SRC,
            "gemm_qkv_hfq4g256_fp16_wave64",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq4g256_fp16_wave64"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;
        let grid_x = (total_m + 1) / 2;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (q_m + k_m + v_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_fp16_wave64", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                [64, 1, 1],
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

    /// v_dot2_f32_f16-accelerated batched 3-way fused HFQ4-G256 GEMM (Q + K + V).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `amd_mixed_dot`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    pub fn gemm_qkv_hfq4g256_dot2(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq4g256_dot2",
            kernels::GEMM_QKV_HFQ4G256_DOT2_SRC,
            "gemm_qkv_hfq4g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq4g256_dot2"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + batch_size * k * 2
            + batch_size * (q_m + k_m + v_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_dot2", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// Batched 2-way fused HFQ4-G256 GEMM for the FFN preamble (gate + up).
    ///
    /// Processes N tokens × both projections (w_gate + w_up) in one launch.
    /// Bitwise-identical to calling `fused_gate_up_hfq4g256` N times on the
    /// same x[b] — 4-accumulator interleave + pairwise combine preserved
    /// per batch element.
    pub fn gemm_gate_up_hfq4g256(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // CDNA3 MFMA path (task #130): two back-to-back rocBLAS calls against
        // the gate/up FP16 shadows. rocBLAS launch overhead is small compared
        // to the GEMM work at prefill batches, so fusing into a single
        // concatenated matrix isn't worth the extra kernel code tonight.
        let cdna3 = self.arch_caps.is_cdna3();
        if cdna3
            && batch_size >= self.rocblas_min_batch()
            && self.rocblas.is_some()
            && !self.graphs.capture_mode
        {
            if let Ok(Some(w_gate_ptr)) = self.ensure_fp16_shadow(a_gate, gate_m, k) {
                if let Ok(Some(w_up_ptr)) = self.ensure_fp16_shadow(a_up, up_m, k) {
                    let x_fp16 = self.ensure_fp16_x(x, batch_size * k)?;
                    let xb = unsafe { DeviceBuffer::from_raw(x_fp16, (batch_size * k) * 2) };
                    let wgate = unsafe { DeviceBuffer::from_raw(w_gate_ptr, (gate_m * k) * 2) };
                    let wup = unsafe { DeviceBuffer::from_raw(w_up_ptr, (up_m * k) * 2) };
                    let gate_bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k);
                    let up_bytes = crate::profile::gemv_hfq4g256_bytes(up_m, k);
                    let timer = crate::profile::begin_timer(
                        &self.hip,
                        "gemm",
                        "gemm_gate_up_hfq4g256_rocblas",
                        gate_bytes + up_bytes,
                    );
                    let r1 = self.rocblas_gemm_hfq4_prefill(
                        &wgate,
                        &xb,
                        &y_gate.buf,
                        gate_m,
                        batch_size,
                        k,
                    );
                    let r2 = if r1.is_ok() {
                        self.rocblas_gemm_hfq4_prefill(&wup, &xb, &y_up.buf, up_m, batch_size, k)
                    } else {
                        Ok(())
                    };
                    std::mem::forget(xb);
                    std::mem::forget(wgate);
                    std::mem::forget(wup);
                    if let Some(t) = timer {
                        t.finish(&self.hip);
                    }
                    return r1.and(r2);
                }
            }
        }
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !self.flags.fp16_disabled {
            // gfx906 dp4a MMQ — default-on at batch_size ≥ 8 (per
            // should_use_mmq's gfx906 default). Quantize X once, screen
            // both weights, dispatch MMQ for each in set mode (add=0).
            // See docs/plans/gfx906-mmq-prd.md for context.
            let mut mmq_screen_rejected = false;
            if self.arch_caps.is_gfx906() && self.arch_caps.should_use_mmq(batch_size) {
                let use_mmq = if self.mmq_screen.enabled {
                    self.mmq_screen_weight(a_gate, gate_m, k)
                        && self.mmq_screen_weight(a_up, up_m, k)
                } else {
                    true
                };
                if use_mmq {
                    if gate_m % 128 == 0 && up_m % 128 == 0 {
                        return self.gemm_gate_up_hfq4g256_mmq_gfx906(
                            a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                        );
                    }
                    let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                    let r1 = self
                        .gemm_hfq4g256_mmq_set_gfx906(a_gate, xq, y_gate, gate_m, k, batch_size);
                    let r2 = if r1.is_ok() {
                        self.gemm_hfq4g256_mmq_set_gfx906(a_up, xq, y_up, up_m, k, batch_size)
                    } else {
                        Ok(())
                    };
                    return r1.and(r2);
                }
                mmq_screen_rejected = self.mmq_screen.enabled;
                // screening rejected at least one weight — fall through; the
                // screen-reject path skips dp4a and lands on fp16 to preserve
                // the higher-precision fallback intent (dp4a shares the Q8_1
                // quant step that MMQ already failed on for this weight).
            }
            // gfx906 dp4a 2-way fused (issue #276 Gap 2). Fires for B>1
            // below the MMQ cutover or in capture mode. Skipped on
            // screen-reject.
            if !mmq_screen_rejected
                && self.arch_caps.gemv_dp4a_enabled()
                && !self.graphs.capture_mode
            {
                return self.gemm_gate_up_hfq4g256_wave64_dp4a(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            // Wave64 FP16 hybrid — best of both worlds for gfx906 (MI50).
            if self.arch_caps.is_gcn5_wave64() {
                return self.gemm_gate_up_hfq4g256_fp16_wave64(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            if self.arch_caps.should_use_mmq(batch_size) {
                let use_mmq = if self.mmq_screen.enabled {
                    self.mmq_screen_weight(a_gate, gate_m, k)
                        && self.mmq_screen_weight(a_up, up_m, k)
                } else {
                    true
                };
                if use_mmq {
                    let xq = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
                    let r1 = self
                        .gemm_hfq4g256_mmq_set_prequant(a_gate, xq, y_gate, gate_m, k, batch_size);
                    let r2 = if r1.is_ok() {
                        self.gemm_hfq4g256_mmq_set_prequant(a_up, xq, y_up, up_m, k, batch_size)
                    } else {
                        Ok(())
                    };
                    return r1.and(r2);
                }
            }
            // HFQ4 wave32 MMQ RDNA2 path (issue #299 Phase 3).
            if self.arch_caps.has_hfq4_mmq() && gate_m % 128 == 0 && up_m % 128 == 0 {
                return self.gemm_gate_up_hfq4g256_mmq(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            // WMMA on gfx12 (RDNA4)
            if self.arch_caps.has_wmma_w32_gfx12() {
                return self.gemm_gate_up_hfq4g256_wmma_gfx12(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            // WMMA on gfx11 (RDNA3)
            if self.arch_caps.has_wmma_w32() {
                return self.gemm_gate_up_hfq4g256_wmma(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_gate_up_hfq4g256_dot2(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_gate_up_hfq4g256_fp16(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_gate_up_hfq4g256",
            kernels::GEMM_GATE_UP_HFQ4G256_SRC,
            "gemm_gate_up_hfq4g256",
        )?;
        let func = &self.functions["gemm_gate_up_hfq4g256"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// v_dot2_f32_f16-accelerated batched 2-way fused HFQ4-G256 GEMM (gate + up).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `__ockl_fdot2`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    pub fn gemm_gate_up_hfq4g256_dot2(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq4g256_dot2",
            kernels::GEMM_GATE_UP_HFQ4G256_DOT2_SRC,
            "gemm_gate_up_hfq4g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq4g256_dot2"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k)
            + batch_size * k * 2
            + batch_size * (gate_m + up_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq4g256_dot2", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// Batched 2-way fused HFQ3-G256 GEMM for the FFN preamble (MQ3 path).
    ///
    /// HFQ3 sibling of `gemm_gate_up_hfq4g256` — single scalar variant only.
    /// Phase 1 of the gfx10 MQ3 prefill plan.
    pub fn gemm_gate_up_hfq3g256(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Phase 3 MMQ (auto-tile-selecting). Default-on for the supported
        // allowlist unless HIPFIRE_HFQ3_MMQ=0, and gate_m/up_m must be
        // MMQ_Y-aligned. Auto-selector falls back to dot2 at small batch.
        // Layer-gate is a no-op when unset (#302).
        if batch_size > 1
            && self.arch_caps.has_hfq3_mmq()
            && self.flags.hfq3_mmq_layer_gate_pass()
            && gate_m % 128 == 0
            && up_m % 128 == 0
        {
            return self.gemm_gate_up_hfq3g256_mmq(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        // Phase 2 experimental: wave32 dp4a if HIPFIRE_HFQ3_DP4A=1.
        if batch_size > 1 && self.arch_caps.has_hfq3_dp4a() {
            return self.gemm_gate_up_hfq3g256_dp4a(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        // FP16 fast paths — Phase 2b (dot2) + Phase 2c (fp16 fallback).
        // Layer-aware FP16 gate (#302).
        if batch_size > 1 && !self.flags.fp16_disabled_for_current_layer() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_gate_up_hfq3g256_dot2(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            return self.gemm_gate_up_hfq3g256_fp16(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_gate_up_hfq3g256",
            kernels::GEMM_GATE_UP_HFQ3G256_SRC,
            "gemm_gate_up_hfq3g256",
        )?;
        let func = &self.functions["gemm_gate_up_hfq3g256"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut gate_m_val = gate_m as i32;
        let mut up_m_val = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut gate_m_val as *mut _ as *mut c_void,
            &mut up_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;
        let bytes = crate::profile::gemm_hfq3g256_bytes(gate_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(up_m, k, batch_size);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq3g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// v_dot2_f32_f16-accelerated batched 2-way fused HFQ3-G256 GEMM (gate + up).
    /// HFQ3 sibling of `gemm_gate_up_hfq4g256_dot2`. Phase 2b.
    pub fn gemm_gate_up_hfq3g256_dot2(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq3g256_dot2",
            kernels::GEMM_GATE_UP_HFQ3G256_DOT2_SRC,
            "gemm_gate_up_hfq3g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq3g256_dot2"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut gate_m_val = gate_m as i32;
        let mut up_m_val = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut gate_m_val as *mut _ as *mut c_void,
            &mut up_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;
        let bytes = crate::profile::gemm_hfq3g256_bytes(gate_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(up_m, k, batch_size)
            + batch_size * k * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq3g256_dot2", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// v_pk_fma_f16-accelerated batched 2-way fused HFQ3-G256 GEMM (gate + up).
    /// Fallback for archs without the dot extension (gfx1010, gfx1013).
    /// Phase 2c of the gfx10 MQ3 prefill plan.
    pub fn gemm_gate_up_hfq3g256_fp16(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq3g256_fp16",
            kernels::GEMM_GATE_UP_HFQ3G256_FP16_SRC,
            "gemm_gate_up_hfq3g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq3g256_fp16"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut gate_m_val = gate_m as i32;
        let mut up_m_val = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut gate_m_val as *mut _ as *mut c_void,
            &mut up_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;
        let bytes = crate::profile::gemm_hfq3g256_bytes(gate_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(up_m, k, batch_size)
            + batch_size * k * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq3g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// Wave32+dp4a batched 2-way fused HFQ3-G256 GEMM (gate + up).
    /// Phase 2 experimental sibling of `gemm_qkv_hfq3g256_dp4a`.
    pub fn gemm_gate_up_hfq3g256_dp4a(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.has_hfq3_sdot4() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_gate_up_hfq3g256_dot2(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            return self.gemm_gate_up_hfq3g256_fp16(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_gate_up_hfq3g256_dp4a",
            kernels::GEMM_GATE_UP_HFQ3G256_DP4A_SRC,
            "gemm_gate_up_hfq3g256_dp4a",
        )?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions["gemm_gate_up_hfq3g256_dp4a"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut gate_m_val = gate_m as i32;
        let mut up_m_val = up_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut gate_m_val as *mut _ as *mut c_void,
            &mut up_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 16;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemm_hfq3g256_bytes(gate_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(up_m, k, batch_size)
            + batch_size * k
            + batch_size * (gate_m + up_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq3g256_dp4a", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// Wave32 MMQ residual kernel for HFQ3 on RDNA2+ — Phase 3 tile-size
    /// family auto-selector. Picks the best path per batch_size, falling
    /// back to `gemm_hfq3g256_residual_dot2` when MMQ would lose at small
    /// N. Gate boundaries from the microbench at
    /// `examples/bench_hfq3_mmq_sweep.rs` (m=4096, k=2048 on gfx1031):
    ///   batch ≤ 12       → dot2 (MMQ tile granularity wastes compute)
    ///   13 ≤ batch ≤ 127 → mmq_x=16 (best across this whole range,
    ///                       within ~5% of mmq_x=32 even at N=96)
    ///   batch ≥ 128      → mmq_x=32 (b128 LDS path pulls ahead +4-10%)
    /// Default-on on the supported allowlist unless `HIPFIRE_HFQ3_MMQ=0`.
    /// mmq_x=8 is never best in the
    /// sweep (lost to scalar/dot2 at small N, lost to mmq_x=16 at large
    /// N) so it's not in the auto-selector — kept available as
    /// `gemm_hfq3g256_residual_mmq_x8` for further experimentation.
    pub fn gemm_hfq3g256_residual_mmq(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to gemm_hfq3g256_residual_{dot2,mmq_xN} which bind.
        if !self.arch_caps.has_hfq3_sdot4() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_hfq3g256_residual_dot2(a_raw, x, y, m, k, batch_size);
            }
            return self.gemm_hfq3g256_residual_fp16(a_raw, x, y, m, k, batch_size);
        }
        if batch_size <= 12 {
            self.gemm_hfq3g256_residual_dot2(a_raw, x, y, m, k, batch_size)
        } else if batch_size <= 63 {
            self.gemm_hfq3g256_residual_mmq_x16(a_raw, x, y, m, k, batch_size)
        } else {
            self.gemm_hfq3g256_residual_mmq_x32_y64(a_raw, x, y, m, k, batch_size)
        }
    }

    fn launch_hfq3_mmq_tile(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        self.launch_hfq3_mmq_tile_with_y(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            mmq_x,
            128,
            kernel_name,
            src,
        )
    }

    /// MMQ_Y-parameterized variant of `launch_hfq3_mmq_tile`. The body.cuh
    /// lets wrappers override the row-tile size for occupancy probes while
    /// preserving the same x-quantization path.
    #[allow(clippy::too_many_arguments)]
    fn launch_hfq3_mmq_tile_with_y(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        mmq_y: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        // Inline the body .cuh — same pattern as the gfx906 MMQ family.
        let inlined = src.replace(
            "#include \"gemm_hfq3g256_residual_mmq_body.cuh\"",
            kernels::GEMM_HFQ3G256_RESIDUAL_MMQ_BODY_CUH,
        );
        self.ensure_kernel(kernel_name, &inlined, kernel_name)?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions[kernel_name];

        let mut ap = a_raw.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut yp = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        // LDS layout — must match the body.cuh constants:
        //   x_qs: mmq_y × X_STRIDE(40) ints + x_dm: mmq_y × float2
        //   tile_y: mmq_x × Y_STRIDE(36) ints
        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (mmq_y * X_STRIDE * 4 + mmq_y * 8 + mmq_x * Y_STRIDE * 4) as u32;

        let row_tiles = (m + mmq_y - 1) / mmq_y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let bytes = crate::profile::gemm_hfq3g256_bytes(m, k, batch_size)
            + batch_size * k
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ3 MMQ residual at mmq_x=8 (short-prefill tile).
    pub fn gemm_hfq3g256_residual_mmq_x8(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_hfq3_mmq_tile(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            8,
            "gemm_hfq3g256_residual_mmq_x8",
            kernels::GEMM_HFQ3G256_RESIDUAL_MMQ_X8_SRC,
        )
    }

    /// HFQ3 MMQ residual at mmq_x=16 (mid-prefill tile).
    pub fn gemm_hfq3g256_residual_mmq_x16(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_hfq3_mmq_tile(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            16,
            "gemm_hfq3g256_residual_mmq_x16",
            kernels::GEMM_HFQ3G256_RESIDUAL_MMQ_X16_SRC,
        )
    }

    /// HFQ3 MMQ residual at mmq_x=32 (long-prefill tile, b128 LDS path).
    pub fn gemm_hfq3g256_residual_mmq_x32(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_hfq3_mmq_tile(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            32,
            "gemm_hfq3g256_residual_mmq_x32",
            kernels::GEMM_HFQ3G256_RESIDUAL_MMQ_X32_SRC,
        )
    }

    /// HFQ3 MMQ residual experimental MMQ_Y=64 variant (mmq_x=32).
    pub fn gemm_hfq3g256_residual_mmq_x32_y64(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_hfq3_mmq_tile_with_y(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            32,
            64,
            "gemm_hfq3g256_residual_mmq_x32_y64",
            kernels::GEMM_HFQ3G256_RESIDUAL_MMQ_X32_Y64_SRC,
        )
    }

    /// HFQ3 MMQ residual experimental MMQ_Y=32 variant (mmq_x=32).
    pub fn gemm_hfq3g256_residual_mmq_x32_y32(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_hfq3_mmq_tile_with_y(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            32,
            32,
            "gemm_hfq3g256_residual_mmq_x32_y32",
            kernels::GEMM_HFQ3G256_RESIDUAL_MMQ_X32_Y32_SRC,
        )
    }

    // ── HFQ3 qkv MMQ family — 3-way fused (Q + K + V) ────────────────────
    //
    // Auto-selector picks tile size by batch_size, falling back to dot2 at
    // small N. Same gate boundaries as the residual family from the
    // bench_hfq3_mmq_sweep microbench.

    /// HFQ3 qkv MMQ auto-selector. Default-on unless `HIPFIRE_HFQ3_MMQ=0`.
    /// CALLER INVARIANT: q_m, k_m, v_m must each be multiples of 128.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq3g256_mmq(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to gemm_qkv_hfq3g256_{dot2,mmq_xN} which bind.
        if !self.arch_caps.has_hfq3_sdot4() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkv_hfq3g256_dot2(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            return self.gemm_qkv_hfq3g256_fp16(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        if batch_size <= 12 {
            self.gemm_qkv_hfq3g256_dot2(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            )
        } else if batch_size <= 127 {
            self.gemm_qkv_hfq3g256_mmq_x16(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            )
        } else {
            self.gemm_qkv_hfq3g256_mmq_x32(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_qkv_hfq3_mmq_tile(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        let inlined = src.replace(
            "#include \"gemm_qkv_hfq3g256_mmq_body.cuh\"",
            kernels::GEMM_QKV_HFQ3G256_MMQ_BODY_CUH,
        );
        self.ensure_kernel(kernel_name, &inlined, kernel_name)?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions[kernel_name];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const MMQ_Y: usize = 128;
        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (MMQ_Y * X_STRIDE * 4 + MMQ_Y * 8 + mmq_x * Y_STRIDE * 4) as u32;

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + MMQ_Y - 1) / MMQ_Y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let bytes = crate::profile::gemm_hfq3g256_bytes(q_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(k_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(v_m, k, batch_size)
            + batch_size * k
            + batch_size * total_m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ3 qkv MMQ at mmq_x=8 (short-prefill tile).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq3g256_mmq_x8(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkv_hfq3_mmq_tile(
            a_q,
            a_k,
            a_v,
            x,
            y_q,
            y_k,
            y_v,
            q_m,
            k_m,
            v_m,
            k,
            batch_size,
            8,
            "gemm_qkv_hfq3g256_mmq_x8",
            kernels::GEMM_QKV_HFQ3G256_MMQ_X8_SRC,
        )
    }

    /// HFQ3 qkv MMQ at mmq_x=16 (mid-prefill tile, auto-selector default for 13-127).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq3g256_mmq_x16(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkv_hfq3_mmq_tile(
            a_q,
            a_k,
            a_v,
            x,
            y_q,
            y_k,
            y_v,
            q_m,
            k_m,
            v_m,
            k,
            batch_size,
            16,
            "gemm_qkv_hfq3g256_mmq_x16",
            kernels::GEMM_QKV_HFQ3G256_MMQ_X16_SRC,
        )
    }

    /// HFQ3 qkv MMQ at mmq_x=32 (long-prefill tile, b128 LDS).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq3g256_mmq_x32(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkv_hfq3_mmq_tile(
            a_q,
            a_k,
            a_v,
            x,
            y_q,
            y_k,
            y_v,
            q_m,
            k_m,
            v_m,
            k,
            batch_size,
            32,
            "gemm_qkv_hfq3g256_mmq_x32",
            kernels::GEMM_QKV_HFQ3G256_MMQ_X32_SRC,
        )
    }

    // ── HFQ3 gate_up MMQ family — 2-way fused ─────────────────────────────

    /// HFQ3 gate_up MMQ auto-selector. Default-on unless `HIPFIRE_HFQ3_MMQ=0`.
    /// CALLER INVARIANT: gate_m and up_m must each be multiples of 128.
    pub fn gemm_gate_up_hfq3g256_mmq(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to gemm_gate_up_hfq3g256_{dot2,mmq_xN} which bind.
        if !self.arch_caps.has_hfq3_sdot4() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_gate_up_hfq3g256_dot2(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            return self.gemm_gate_up_hfq3g256_fp16(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        if batch_size <= 12 {
            self.gemm_gate_up_hfq3g256_dot2(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            )
        } else if batch_size <= 127 {
            self.gemm_gate_up_hfq3g256_mmq_x16(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            )
        } else {
            self.gemm_gate_up_hfq3g256_mmq_x32(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_gate_up_hfq3_mmq_tile(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        self.launch_gate_up_hfq3_mmq_tile_with_y(
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
            mmq_x,
            128,
            kernel_name,
            src,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_gate_up_hfq3_mmq_tile_with_y(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        mmq_y: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        let inlined = src.replace(
            "#include \"gemm_gate_up_hfq3g256_mmq_body.cuh\"",
            kernels::GEMM_GATE_UP_HFQ3G256_MMQ_BODY_CUH,
        );
        self.ensure_kernel(kernel_name, &inlined, kernel_name)?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions[kernel_name];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut gate_m_val = gate_m as i32;
        let mut up_m_val = up_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut gate_m_val as *mut _ as *mut c_void,
            &mut up_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (mmq_y * X_STRIDE * 4 + mmq_y * 8 + mmq_x * Y_STRIDE * 4) as u32;

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + mmq_y - 1) / mmq_y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let bytes = crate::profile::gemm_hfq3g256_bytes(gate_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(up_m, k, batch_size)
            + batch_size * k
            + batch_size * total_m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ3 gate_up MMQ at mmq_x=8.
    pub fn gemm_gate_up_hfq3g256_mmq_x8(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_gate_up_hfq3_mmq_tile(
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
            8,
            "gemm_gate_up_hfq3g256_mmq_x8",
            kernels::GEMM_GATE_UP_HFQ3G256_MMQ_X8_SRC,
        )
    }

    /// HFQ3 gate_up MMQ at mmq_x=16.
    pub fn gemm_gate_up_hfq3g256_mmq_x16(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_gate_up_hfq3_mmq_tile(
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
            16,
            "gemm_gate_up_hfq3g256_mmq_x16",
            kernels::GEMM_GATE_UP_HFQ3G256_MMQ_X16_SRC,
        )
    }

    /// HFQ3 gate_up MMQ at mmq_x=32.
    pub fn gemm_gate_up_hfq3g256_mmq_x32(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_gate_up_hfq3_mmq_tile(
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
            32,
            "gemm_gate_up_hfq3g256_mmq_x32",
            kernels::GEMM_GATE_UP_HFQ3G256_MMQ_X32_SRC,
        )
    }

    /// HFQ3 gate_up MMQ mmq_x=32, MMQ_Y=96.
    pub fn gemm_gate_up_hfq3g256_mmq_x32_y96(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_gate_up_hfq3_mmq_tile_with_y(
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
            32,
            96,
            "gemm_gate_up_hfq3g256_mmq_x32_y96",
            kernels::GEMM_GATE_UP_HFQ3G256_MMQ_X32_Y96_SRC,
        )
    }

    /// HFQ3 gate_up MMQ mmq_x=32, MMQ_Y=64.
    pub fn gemm_gate_up_hfq3g256_mmq_x32_y64(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_gate_up_hfq3_mmq_tile_with_y(
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
            32,
            64,
            "gemm_gate_up_hfq3g256_mmq_x32_y64",
            kernels::GEMM_GATE_UP_HFQ3G256_MMQ_X32_Y64_SRC,
        )
    }

    // ── HFQ3 qkvza MMQ family — 4-way fused LinearAttention preamble ─────

    /// HFQ3 qkvza MMQ auto-selector (wqkv + wz + w_beta + w_alpha). Default-on
    /// unless `HIPFIRE_HFQ3_MMQ=0`. CALLER INVARIANT: qkv_m, z_m, beta_m,
    /// alpha_m must each be multiples of 128.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq3g256_mmq(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to gemm_qkvza_hfq3g256_{dot2,mmq_xN} which bind.
        if !self.arch_caps.has_hfq3_sdot4() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkvza_hfq3g256_dot2(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            return self.gemm_qkvza_hfq3g256_fp16(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        if batch_size <= 12 {
            self.gemm_qkvza_hfq3g256_dot2(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            )
        } else if batch_size <= 127 {
            self.gemm_qkvza_hfq3g256_mmq_x16(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            )
        } else {
            self.gemm_qkvza_hfq3g256_mmq_x32(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_qkvza_hfq3_mmq_tile(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        let inlined = src.replace(
            "#include \"gemm_qkvza_hfq3g256_mmq_body.cuh\"",
            kernels::GEMM_QKVZA_HFQ3G256_MMQ_BODY_CUH,
        );
        self.ensure_kernel(kernel_name, &inlined, kernel_name)?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions[kernel_name];

        let mut aqkv = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut yqkv = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut qkv_m_val = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut beta_m_val = beta_m as i32;
        let mut alpha_m_val = alpha_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aqkv as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut yqkv as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut qkv_m_val as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut beta_m_val as *mut _ as *mut c_void,
            &mut alpha_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const MMQ_Y: usize = 128;
        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (MMQ_Y * X_STRIDE * 4 + MMQ_Y * 8 + mmq_x * Y_STRIDE * 4) as u32;

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + MMQ_Y - 1) / MMQ_Y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let bytes = crate::profile::gemm_hfq3g256_bytes(qkv_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(z_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(beta_m, k, batch_size)
            + crate::profile::gemm_hfq3g256_bytes(alpha_m, k, batch_size)
            + batch_size * k
            + batch_size * total_m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ3 qkvza MMQ at mmq_x=8.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq3g256_mmq_x8(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkvza_hfq3_mmq_tile(
            a_qkv,
            a_z,
            a_beta,
            a_alpha,
            x,
            y_qkv,
            y_z,
            y_beta,
            y_alpha,
            qkv_m,
            z_m,
            beta_m,
            alpha_m,
            k,
            batch_size,
            8,
            "gemm_qkvza_hfq3g256_mmq_x8",
            kernels::GEMM_QKVZA_HFQ3G256_MMQ_X8_SRC,
        )
    }

    /// HFQ3 qkvza MMQ at mmq_x=16 (auto-selector default for batch 13-127).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq3g256_mmq_x16(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkvza_hfq3_mmq_tile(
            a_qkv,
            a_z,
            a_beta,
            a_alpha,
            x,
            y_qkv,
            y_z,
            y_beta,
            y_alpha,
            qkv_m,
            z_m,
            beta_m,
            alpha_m,
            k,
            batch_size,
            16,
            "gemm_qkvza_hfq3g256_mmq_x16",
            kernels::GEMM_QKVZA_HFQ3G256_MMQ_X16_SRC,
        )
    }

    /// HFQ3 qkvza MMQ at mmq_x=32.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq3g256_mmq_x32(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkvza_hfq3_mmq_tile(
            a_qkv,
            a_z,
            a_beta,
            a_alpha,
            x,
            y_qkv,
            y_z,
            y_beta,
            y_alpha,
            qkv_m,
            z_m,
            beta_m,
            alpha_m,
            k,
            batch_size,
            32,
            "gemm_qkvza_hfq3g256_mmq_x32",
            kernels::GEMM_QKVZA_HFQ3G256_MMQ_X32_SRC,
        )
    }

    // ── HFQ4 RDNA2 tiled MMQ families (issue #299) ─────────────────────
    fn launch_hfq4_mmq_tile_with_y(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        mmq_y: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        let inlined = src.replace(
            "#include \"gemm_hfq4g256_residual_mmq_body.cuh\"",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_BODY_CUH,
        );
        self.ensure_kernel(kernel_name, &inlined, kernel_name)?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions[kernel_name];

        let mut ap = a_raw.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut yp = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (mmq_y * X_STRIDE * 4 + mmq_y * 8 + mmq_x * Y_STRIDE * 4) as u32;
        let row_tiles = (m + mmq_y - 1) / mmq_y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    pub fn gemm_hfq4g256_residual_mmq_x16(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_hfq4_mmq_tile_with_y(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            16,
            128,
            "gemm_hfq4g256_residual_mmq_x16",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_X16_SRC,
        )
    }

    pub fn gemm_hfq4g256_residual_mmq_x32(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_hfq4_mmq_tile_with_y(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            32,
            128,
            "gemm_hfq4g256_residual_mmq_x32",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_X32_SRC,
        )
    }

    pub fn gemm_hfq4g256_residual_mmq_x32_y64(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_hfq4_mmq_tile_with_y(
            a_raw,
            x,
            y,
            m,
            k,
            batch_size,
            32,
            64,
            "gemm_hfq4g256_residual_mmq_x32_y64",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_X32_Y64_SRC,
        )
    }

    pub fn gemm_hfq4g256_residual_mmq_rdna2_auto(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if batch_size <= 63 {
            self.gemm_hfq4g256_residual_mmq_x16(a_raw, x, y, m, k, batch_size)
        } else {
            self.gemm_hfq4g256_residual_mmq_x32_y64(a_raw, x, y, m, k, batch_size)
        }
    }

    // ── HFQ4 qkv MMQ family (3-way fused Q+K+V) — issue #299 Phase 2 ────
    #[allow(clippy::too_many_arguments)]
    fn launch_qkv_hfq4_mmq_tile(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        let inlined = src.replace(
            "#include \"gemm_qkv_hfq4g256_mmq_body.cuh\"",
            kernels::GEMM_QKV_HFQ4G256_MMQ_BODY_CUH,
        );
        self.ensure_kernel(kernel_name, &inlined, kernel_name)?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions[kernel_name];

        let mut a_q_p = a_q.buf.as_ptr();
        let mut a_k_p = a_k.buf.as_ptr();
        let mut a_v_p = a_v.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut y_q_p = y_q.buf.as_ptr();
        let mut y_k_p = y_k.buf.as_ptr();
        let mut y_v_p = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_q_p as *mut _ as *mut c_void,
            &mut a_k_p as *mut _ as *mut c_void,
            &mut a_v_p as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut y_q_p as *mut _ as *mut c_void,
            &mut y_k_p as *mut _ as *mut c_void,
            &mut y_v_p as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const MMQ_Y: usize = 128;
        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (MMQ_Y * X_STRIDE * 4 + MMQ_Y * 8 + mmq_x * Y_STRIDE * 4) as u32;
        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + MMQ_Y - 1) / MMQ_Y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + batch_size * k
            + batch_size * (q_m + k_m + v_m) * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    pub fn gemm_qkv_hfq4g256_mmq_x16(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkv_hfq4_mmq_tile(
            a_q,
            a_k,
            a_v,
            x,
            y_q,
            y_k,
            y_v,
            q_m,
            k_m,
            v_m,
            k,
            batch_size,
            16,
            "gemm_qkv_hfq4g256_mmq_x16",
            kernels::GEMM_QKV_HFQ4G256_MMQ_X16_SRC,
        )
    }

    pub fn gemm_qkv_hfq4g256_mmq_x32(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkv_hfq4_mmq_tile(
            a_q,
            a_k,
            a_v,
            x,
            y_q,
            y_k,
            y_v,
            q_m,
            k_m,
            v_m,
            k,
            batch_size,
            32,
            "gemm_qkv_hfq4g256_mmq_x32",
            kernels::GEMM_QKV_HFQ4G256_MMQ_X32_SRC,
        )
    }

    pub fn gemm_qkv_hfq4g256_mmq(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if batch_size <= 63 {
            self.gemm_qkv_hfq4g256_mmq_x16(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            )
        } else {
            self.gemm_qkv_hfq4g256_mmq_x32(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            )
        }
    }

    // ── HFQ4 gate_up MMQ family (2-way fused gate+up) — issue #299 Phase 3 ──
    #[allow(clippy::too_many_arguments)]
    fn launch_gate_up_hfq4_mmq_tile(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        let inlined = src.replace(
            "#include \"gemm_gate_up_hfq4g256_mmq_body.cuh\"",
            kernels::GEMM_GATE_UP_HFQ4G256_MMQ_BODY_CUH,
        );
        self.ensure_kernel(kernel_name, &inlined, kernel_name)?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions[kernel_name];

        let mut a_gate_p = a_gate.buf.as_ptr();
        let mut a_up_p = a_up.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut y_gate_p = y_gate.buf.as_ptr();
        let mut y_up_p = y_up.buf.as_ptr();
        let mut gate_m_val = gate_m as i32;
        let mut up_m_val = up_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_gate_p as *mut _ as *mut c_void,
            &mut a_up_p as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut y_gate_p as *mut _ as *mut c_void,
            &mut y_up_p as *mut _ as *mut c_void,
            &mut gate_m_val as *mut _ as *mut c_void,
            &mut up_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const MMQ_Y: usize = 128;
        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (MMQ_Y * X_STRIDE * 4 + MMQ_Y * 8 + mmq_x * Y_STRIDE * 4) as u32;
        let total_m = gate_m + up_m;
        let row_tiles = (total_m + MMQ_Y - 1) / MMQ_Y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k)
            + batch_size * k
            + batch_size * (gate_m + up_m) * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    pub fn gemm_gate_up_hfq4g256_mmq_x16(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_gate_up_hfq4_mmq_tile(
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
            16,
            "gemm_gate_up_hfq4g256_mmq_x16",
            kernels::GEMM_GATE_UP_HFQ4G256_MMQ_X16_SRC,
        )
    }

    pub fn gemm_gate_up_hfq4g256_mmq_x32(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_gate_up_hfq4_mmq_tile(
            a_gate,
            a_up,
            x,
            y_gate,
            y_up,
            gate_m,
            up_m,
            k,
            batch_size,
            32,
            "gemm_gate_up_hfq4g256_mmq_x32",
            kernels::GEMM_GATE_UP_HFQ4G256_MMQ_X32_SRC,
        )
    }

    pub fn gemm_gate_up_hfq4g256_mmq(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if batch_size <= 63 {
            self.gemm_gate_up_hfq4g256_mmq_x16(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            )
        } else {
            self.gemm_gate_up_hfq4g256_mmq_x32(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            )
        }
    }

    // ── HFQ4 qkvza MMQ family (4-way fused LA preamble) — issue #299 Phase 4 ──
    #[allow(clippy::too_many_arguments)]
    fn launch_qkvza_hfq4_mmq_tile(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
        mmq_x: usize,
        kernel_name: &'static str,
        src: &'static str,
    ) -> HipResult<()> {
        let inlined = src.replace(
            "#include \"gemm_qkvza_hfq4g256_mmq_body.cuh\"",
            kernels::GEMM_QKVZA_HFQ4G256_MMQ_BODY_CUH,
        );
        self.ensure_kernel(kernel_name, &inlined, kernel_name)?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions[kernel_name];

        let mut a_qkv_p = a_qkv.buf.as_ptr();
        let mut a_z_p = a_z.buf.as_ptr();
        let mut a_beta_p = a_beta.buf.as_ptr();
        let mut a_alpha_p = a_alpha.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut y_qkv_p = y_qkv.buf.as_ptr();
        let mut y_z_p = y_z.buf.as_ptr();
        let mut y_beta_p = y_beta.buf.as_ptr();
        let mut y_alpha_p = y_alpha.buf.as_ptr();
        let mut qkv_m_val = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut beta_m_val = beta_m as i32;
        let mut alpha_m_val = alpha_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_qkv_p as *mut _ as *mut c_void,
            &mut a_z_p as *mut _ as *mut c_void,
            &mut a_beta_p as *mut _ as *mut c_void,
            &mut a_alpha_p as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut y_qkv_p as *mut _ as *mut c_void,
            &mut y_z_p as *mut _ as *mut c_void,
            &mut y_beta_p as *mut _ as *mut c_void,
            &mut y_alpha_p as *mut _ as *mut c_void,
            &mut qkv_m_val as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut beta_m_val as *mut _ as *mut c_void,
            &mut alpha_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const MMQ_Y: usize = 128;
        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (MMQ_Y * X_STRIDE * 4 + MMQ_Y * 8 + mmq_x * Y_STRIDE * 4) as u32;
        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + MMQ_Y - 1) / MMQ_Y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
            + batch_size * k
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq4g256_mmq_x16(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkvza_hfq4_mmq_tile(
            a_qkv,
            a_z,
            a_beta,
            a_alpha,
            x,
            y_qkv,
            y_z,
            y_beta,
            y_alpha,
            qkv_m,
            z_m,
            beta_m,
            alpha_m,
            k,
            batch_size,
            16,
            "gemm_qkvza_hfq4g256_mmq_x16",
            kernels::GEMM_QKVZA_HFQ4G256_MMQ_X16_SRC,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq4g256_mmq_x32(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_qkvza_hfq4_mmq_tile(
            a_qkv,
            a_z,
            a_beta,
            a_alpha,
            x,
            y_qkv,
            y_z,
            y_beta,
            y_alpha,
            qkv_m,
            z_m,
            beta_m,
            alpha_m,
            k,
            batch_size,
            32,
            "gemm_qkvza_hfq4g256_mmq_x32",
            kernels::GEMM_QKVZA_HFQ4G256_MMQ_X32_SRC,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq4g256_mmq(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if batch_size <= 63 {
            self.gemm_qkvza_hfq4g256_mmq_x16(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            )
        } else {
            self.gemm_qkvza_hfq4g256_mmq_x32(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            )
        }
    }

    /// Wave32 MMQ residual kernel for HFQ4 on RDNA2+ — Phase 3 side-win probe.
    /// Same topology as the HFQ3 sibling; differs only in 4-bit nibble unpack
    /// (vs 3-bit trit). Default-on unless `HIPFIRE_HFQ4_MMQ_RDNA2=0`.
    pub fn gemm_hfq4g256_residual_mmq_rdna2(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Distinct module + function name from the pre-existing
        // `gemm_hfq4g256_residual_mmq` (llama.cpp-style, RDNA3+ via
        // HIPFIRE_WO_MMQ=1) to avoid kernel-cache collision.
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mmq_rdna2",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_RDNA2_SRC,
            "gemm_hfq4g256_residual_mmq_rdna2",
        )?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let func = &self.functions["gemm_hfq4g256_residual_mmq_rdna2"];

        let mut ap = a_raw.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut yp = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        const MMQ_Y: usize = 128;
        const MMQ_X: usize = 32;
        const X_STRIDE: usize = 40;
        const Y_STRIDE: usize = 36;
        let shared_mem = (MMQ_Y * X_STRIDE * 4 + MMQ_Y * 8 + MMQ_X * Y_STRIDE * 4) as u32;

        let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
        let col_tiles = (batch_size + MMQ_X - 1) / MMQ_X;

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_residual_mmq_rdna2",
            bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, col_tiles as u32, 1],
                [32, 4, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// FP16-packed batched 2-way fused HFQ4-G256 GEMM (gate + up).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    pub fn gemm_gate_up_hfq4g256_fp16(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq4g256_fp16",
            kernels::GEMM_GATE_UP_HFQ4G256_FP16_SRC,
            "gemm_gate_up_hfq4g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq4g256_fp16"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (gate_m + up_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq4g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// GCN5 wave64 FP16 hybrid batched 2-way fused HFQ4-G256 GEMM (gate + up).
    /// block=[64,1,1] with 2 rows/block via warp_id. Halves grid.x vs wave32.
    /// Default-on for gfx906; gfx908 opts in via HIPFIRE_GCN5_WAVE64_HYBRID=1.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq4g256_fp16_wave64(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq4g256_fp16_wave64",
            kernels::GEMM_GATE_UP_HFQ4G256_FP16_WAVE64_SRC,
            "gemm_gate_up_hfq4g256_fp16_wave64",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq4g256_fp16_wave64"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;
        let grid_x = (total_m + 1) / 2;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (gate_m + up_m) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_hfq4g256_fp16_wave64",
            bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                [64, 1, 1],
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

    /// WMMA-accelerated batched 5-way fused HFQ4-G256 GEMM (qkv + z + beta + alpha).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    pub fn gemm_qkvza_hfq4g256_wmma(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq4g256_wmma",
            kernels::GEMM_QKVZA_HFQ4G256_WMMA_SRC,
            "gemm_qkvza_hfq4g256_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq4g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m);
                b.push_i32(z_m_val);
                b.push_i32(b_m);
                b.push_i32(a_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFP4-G32 batched 4-way fused GEMM (qkv + z + beta + alpha) for
    /// the Qwen3.5 DeltaNet LA preamble. Routes gfx11 / gfx12. Used for
    /// HFP4G32 (raw X) and MFP4G32 (FWHT-rotated X handled upstream).
    pub fn gemm_qkvza_hfp4g32(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.gemm_qkvza_hfp4g32_wmma_gfx12(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        self.gemm_qkvza_hfp4g32_wmma(
            a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
            alpha_m, k, batch_size,
        )
    }

    pub fn gemm_qkvza_hfp4g32_wmma(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfp4g32_wmma",
            kernels::GEMM_QKVZA_HFP4G32_WMMA_SRC,
            "gemm_qkvza_hfp4g32_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfp4g32_bytes(qkv_m, k)
            + crate::profile::gemv_hfp4g32_bytes(z_m, k)
            + crate::profile::gemv_hfp4g32_bytes(beta_m, k)
            + crate::profile::gemv_hfp4g32_bytes(alpha_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfp4g32_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfp4g32_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m);
                b.push_i32(z_m_val);
                b.push_i32(b_m);
                b.push_i32(a_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    pub fn gemm_qkvza_hfp4g32_wmma_gfx12(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfp4g32_wmma_gfx12",
            kernels::GEMM_QKVZA_HFP4G32_WMMA_GFX12_SRC,
            "gemm_qkvza_hfp4g32_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfp4g32_bytes(qkv_m, k)
            + crate::profile::gemv_hfp4g32_bytes(z_m, k)
            + crate::profile::gemv_hfp4g32_bytes(beta_m, k)
            + crate::profile::gemv_hfp4g32_bytes(alpha_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfp4g32_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfp4g32_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m);
                b.push_i32(z_m_val);
                b.push_i32(b_m);
                b.push_i32(a_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ3-G256 sister of `gemm_qkvza_hfq4g256_wmma`. Same WMMA shape +
    /// lane decomposition; only the inner K-tile unpack differs (3-bit
    /// cross-byte vs 4-bit nibble) and the per-group byte stride is 104
    /// instead of 136. Used for MQ3 prefill via dispatch wrappers that
    /// pre-rotate `x` (see `gemm_qkvza_mq3g256_wmma` below). gfx11 K2
    /// unroll variant — gfx12 K4 to follow once K2 is validated.
    pub fn gemm_qkvza_hfq3g256_wmma(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // HFQ3 mb4 path selector. Only triggers on gfx11; gfx12 keeps its
        // existing fast path (line below) since mb4 sibling not ported.
        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let arch_supports_mb4 = self.arch_caps.supports_mq3_lloyd_mb4();
        let use_mb4 = match self.flags.mq3_mb4 {
            None => arch_supports_mb4 && batch_size >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_qkvza_hfq3g256_wmma_mb4(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.gemm_qkvza_hfq3g256_wmma_gfx12(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_qkvza_hfq3g256_wmma",
            kernels::GEMM_QKVZA_HFQ3G256_WMMA_SRC,
            "gemm_qkvza_hfq3g256_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        // HFQ3 storage = 104 B/group → ~3.06 bits/weight (vs HFQ4's 4.25).
        let weight_bytes = (qkv_m + z_m + beta_m + alpha_m) * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq3g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq3g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m);
                b.push_i32(z_m_val);
                b.push_i32(b_m);
                b.push_i32(a_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ3 qkvza mb4 dispatch: 16×64 output tile per WG.
    pub fn gemm_qkvza_hfq3g256_wmma_mb4(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq3g256_wmma_mb4",
            kernels::GEMM_QKVZA_HFQ3G256_WMMA_MB4_SRC,
            "gemm_qkvza_hfq3g256_wmma_mb4",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m_v = qkv_m as i32;
        let mut z_m_v = z_m as i32;
        let mut b_m_v = beta_m as i32;
        let mut a_m_v = alpha_m as i32;
        let mut k_v = k as i32;
        let mut n_v = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m_v as *mut _ as *mut c_void,
            &mut z_m_v as *mut _ as *mut c_void,
            &mut b_m_v as *mut _ as *mut c_void,
            &mut a_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 63) / 64;
        let bytes = total_m * (k / 256) * 104 + batch_size * k * 2 + batch_size * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq3g256_wmma_mb4", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq3g256_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m_v);
                b.push_i32(z_m_v);
                b.push_i32(b_m_v);
                b.push_i32(a_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ3 wrapper: rotates `x` via `mq_rotate_x` (FWHT with shared sign
    /// vectors) into the caller-provided `x_rot` scratch, then invokes
    /// `gemm_qkvza_hfq3g256_wmma`. Mirror of `gemm_qkvza_mq4g256_wmma`.
    /// Caller is responsible for `x_rot` being [batch × K] f32 scratch.
    pub fn gemm_qkvza_mq3g256_wmma(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Rotate batched x. mq_rotate_x_batched applies FWHT per-row.
        for b in 0..batch_size {
            let x_row = x.sub_offset(b * k, k);
            let x_rot_row = x_rot.sub_offset(b * k, k);
            self.rotate_x_mq(&x_row, &x_rot_row, k)?;
        }
        // Invalidate the fp16-conversion cache: `x_rot`'s pointer is stable
        // across consecutive MQ3 wrapper calls (same scratch buffer reused
        // per layer), but the underlying data was just rewritten by the
        // rotate loop above. Without this, `ensure_fp16_x` would see the
        // matching `fp16_x_source_ptr` and skip the f32→fp16 conversion,
        // and the kernel would read stale fp16 values from the previous
        // layer's rotation.
        self.scratch.fp16_x_source_ptr = std::ptr::null_mut();
        self.gemm_qkvza_hfq3g256_wmma(
            a_qkv, a_z, a_beta, a_alpha, x_rot, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
            alpha_m, k, batch_size,
        )
    }

    /// gfx12 (RDNA4) sister of `gemm_qkvza_hfq3g256_wmma`. K4-unrolled
    /// half8_t lane-split per `gemm_qkvza_hfq4g256_wmma_gfx12`. Wired via
    /// the `gemm_qkvza_hfq3g256_wmma` arch dispatch — direct callers can
    /// also use this if they know they're on gfx12.
    pub fn gemm_qkvza_hfq3g256_wmma_gfx12(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq3g256_wmma_gfx12",
            kernels::GEMM_QKVZA_HFQ3G256_WMMA_GFX12_SRC,
            "gemm_qkvza_hfq3g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = total_m * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq3g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq3g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m);
                b.push_i32(z_m_val);
                b.push_i32(b_m);
                b.push_i32(a_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx12 (RDNA4) sister of `gemm_qkvza_hfq4g256_wmma`. Same gfx12
    /// recipe as the other scaffolds (validated on R9700) extended to
    /// 4-output qkv/z/beta/alpha routing. Not yet wired into the public
    /// dispatch tree — exposed only for the channel-test harness.
    pub fn gemm_qkvza_hfq4g256_wmma_gfx12(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Adaptive-B batch-tile (env HIPFIRE_GATE_UP_BT, shared with gate_up/residual).
        let bt_b: usize = if hipfire_config::developer_var("HIPFIRE_GATE_UP_BT")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(true)
        {
            if batch_size < 64 {
                1
            } else if batch_size % 192 == 0 {
                12
            } else if batch_size % 128 == 0 {
                8
            } else if batch_size % 64 == 0 {
                4
            } else if batch_size >= 192 {
                12
            } else if batch_size >= 128 {
                8
            } else {
                4
            }
        } else {
            1
        };
        let (kname, ksrc): (&str, &str) = match bt_b {
            12 => (
                "gemm_qkvza_hfq4g256_wmma_gfx12_bt12",
                kernels::GEMM_QKVZA_HFQ4G256_WMMA_GFX12_BT_SRC,
            ),
            8 => (
                "gemm_qkvza_hfq4g256_wmma_gfx12_bt8",
                kernels::GEMM_QKVZA_HFQ4G256_WMMA_GFX12_BT_SRC,
            ),
            4 => (
                "gemm_qkvza_hfq4g256_wmma_gfx12_bt4",
                kernels::GEMM_QKVZA_HFQ4G256_WMMA_GFX12_BT_SRC,
            ),
            _ => (
                "gemm_qkvza_hfq4g256_wmma_gfx12",
                kernels::GEMM_QKVZA_HFQ4G256_WMMA_GFX12_SRC,
            ),
        };
        self.ensure_kernel(kname, ksrc, kname)?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 16 * bt_b - 1) / (16 * bt_b);

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kname, bytes);
        let result = self.launch_maybe_blob(
            kname,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m);
                b.push_i32(z_m_val);
                b.push_i32(b_m);
                b.push_i32(a_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// WMMA-accelerated batched 3-way fused HFQ4-G256 GEMM (Q + K + V).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    pub fn gemm_qkv_hfq4g256_wmma(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq4g256_wmma",
            kernels::GEMM_QKV_HFQ4G256_WMMA_SRC,
            "gemm_qkv_hfq4g256_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq4g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ3-G256 sister of `gemm_qkv_hfq4g256_wmma`. Same WMMA shape +
    /// lane decomposition; only the inner K-tile unpack differs (3-bit
    /// cross-byte vs 4-bit nibble) and the per-group byte stride is 104
    /// instead of 136. Used for MQ3 prefill via dispatch sites in
    /// qwen35.rs FullAttention branch (X is pre-rotated upstream).
    pub fn gemm_qkv_hfq3g256_wmma(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let total_m = q_m + k_m + v_m;
        let arch_supports_mb4 = self.arch_caps.supports_mq3_lloyd_mb4();
        let use_mb4 = match self.flags.mq3_mb4 {
            None => arch_supports_mb4 && batch_size >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_qkv_hfq3g256_wmma_mb4(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.gemm_qkv_hfq3g256_wmma_gfx12(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_qkv_hfq3g256_wmma",
            kernels::GEMM_QKV_HFQ3G256_WMMA_SRC,
            "gemm_qkv_hfq3g256_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = (q_m + k_m + v_m) * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq3g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq3g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ3 qkv mb4 dispatch: 16×64 output tile per WG.
    pub fn gemm_qkv_hfq3g256_wmma_mb4(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq3g256_wmma_mb4",
            kernels::GEMM_QKV_HFQ3G256_WMMA_MB4_SRC,
            "gemm_qkv_hfq3g256_wmma_mb4",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_v = q_m as i32;
        let mut k_m_v = k_m as i32;
        let mut v_m_v = v_m as i32;
        let mut k_v = k as i32;
        let mut n_v = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_v as *mut _ as *mut c_void,
            &mut k_m_v as *mut _ as *mut c_void,
            &mut v_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 63) / 64;
        let bytes = total_m * (k / 256) * 104 + batch_size * k * 2 + batch_size * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq3g256_wmma_mb4", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq3g256_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_v);
                b.push_i32(k_m_v);
                b.push_i32(v_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx12 (RDNA4) sister of `gemm_qkv_hfq4g256_wmma`. Identical signature
    /// and grid/block; only the kernel-side intrinsic + operand vector size
    /// differs. NOT yet wired into the public dispatch tree — exposed only
    /// for the channel-test (`test_wmma_qkv_gfx12`) that validates the
    /// gfx12 C-output mapping hypothesis on real RDNA4 silicon. See issue
    /// #54 and `.skills/hipfire-arch-port/wmma-matrix.md`.
    pub fn gemm_qkv_hfq4g256_wmma_gfx12(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq4g256_wmma_gfx12",
            kernels::GEMM_QKV_HFQ4G256_WMMA_GFX12_SRC,
            "gemm_qkv_hfq4g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq4g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// WMMA-accelerated batched 3-way fused HFP4-G32 GEMM (Q + K + V).
    /// Sister of `gemm_qkv_hfq4g256_wmma` for the FP4 (E2M1 + UE8M0 g32 +
    /// FP16 row scale) family. Routes to the gfx11 or gfx12 variant by
    /// arch. Asserts a WMMA-capable arch — callers must gate via
    /// `is_batchable_la` (which restricts HFP4G32 to gfx11+/gfx12 archs).
    ///
    /// Used for both HFP4G32 (raw, X is the rmsnormed activation) and
    /// MFP4G32 (X is the FWHT-rotated activation; rotation happens
    /// upstream via `mq_rotate_x` so the kernel is identical).
    pub fn gemm_qkv_hfp4g32(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // FP8 WMMA gate: only at batch sizes where the prefill bench
        // measured ≥1× vs FP16 WMMA. At small batches (decode FA QKV
        // calls this with batch_size=1) the FP8 path measures
        // 0.71-0.84×, so we keep the FP16 path there. Threshold is
        // conservative — see project_fp8_wmma_hfp4g32_2026_05_10.md
        // for the full N sweep. The decode-path FP8 win is on the
        // GEMV side (gemv_hfp4g32_fp8_gfx12), not WMMA.
        if self.arch_caps.has_wmma_w32_gfx12()
            && self.flags.fp8_wmma
            && batch_size >= FP8_WMMA_MIN_BATCH
        {
            return self.gemm_qkv_hfp4g32_wmma_fp8_gfx12(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.gemm_qkv_hfp4g32_wmma_gfx12(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        self.gemm_qkv_hfp4g32_wmma(
            a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
        )
    }

    /// gfx11 (RDNA3) variant of `gemm_qkv_hfp4g32`. Direct entry point
    /// for tests; production callers should use `gemm_qkv_hfp4g32` to
    /// pick up the gfx12 sister automatically.
    pub fn gemm_qkv_hfp4g32_wmma(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfp4g32_wmma",
            kernels::GEMM_QKV_HFP4G32_WMMA_SRC,
            "gemm_qkv_hfp4g32_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfp4g32_bytes(q_m, k)
            + crate::profile::gemv_hfp4g32_bytes(k_m, k)
            + crate::profile::gemv_hfp4g32_bytes(v_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfp4g32_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfp4g32_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx12 (RDNA4) variant of `gemm_qkv_hfp4g32`. half8_t lane-split +
    /// K4 unroll. Same C-output mapping as `gemm_qkv_hfq4g256_wmma_gfx12`.
    pub fn gemm_qkv_hfp4g32_wmma_gfx12(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfp4g32_wmma_gfx12",
            kernels::GEMM_QKV_HFP4G32_WMMA_GFX12_SRC,
            "gemm_qkv_hfp4g32_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfp4g32_bytes(q_m, k)
            + crate::profile::gemv_hfp4g32_bytes(k_m, k)
            + crate::profile::gemv_hfp4g32_bytes(v_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfp4g32_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfp4g32_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx12 FP8-WMMA variant of `gemm_qkv_hfp4g32_wmma_gfx12`. Same
    /// 16x16x16 tile shape, same C-mapping; weight LUT pre-converts
    /// E2M1->E4M3 bytes (no scale) and per-output-row row_scale * UE8M0
    /// is applied to the F32 accumulator after each WMMA pair via
    /// lane-shuffle. Activation is converted FP16->FP8 inline by
    /// cvt_pk_fp8_f32 (unscaled — post-RMSNorm magnitudes are bounded
    /// well below E4M3 saturation). Opt-in via HIPFIRE_FP8_WMMA=1.
    pub fn gemm_qkv_hfp4g32_wmma_fp8_gfx12(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfp4g32_wmma_fp8_gfx12",
            kernels::GEMM_QKV_HFP4G32_WMMA_FP8_GFX12_SRC,
            "gemm_qkv_hfp4g32_wmma_fp8_gfx12",
        )?;
        let x_fp8_ptr = self.ensure_fp8_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_fp8_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfp4g32_bytes(q_m, k)
            + crate::profile::gemv_hfp4g32_bytes(k_m, k)
            + crate::profile::gemv_hfp4g32_bytes(v_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkv_hfp4g32_wmma_fp8_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfp4g32_wmma_fp8_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFP4-G32 batched residual GEMM with fused += semantics.
    /// Sister of `gemm_hfq4g256_residual_wmma_k2`. Used for wo + w_down
    /// projections in the batched prefill path. Routes to gfx11/gfx12.
    /// Caller must initialize Y to the residual stream before this call.
    pub fn gemm_hfp4g32_residual(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.gemm_hfp4g32_residual_wmma_gfx12(a, x, y, m, k, batch_size);
        }
        self.gemm_hfp4g32_residual_wmma(a, x, y, m, k, batch_size)
    }

    pub fn gemm_hfp4g32_residual_wmma(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfp4g32_residual_wmma",
            kernels::GEMM_HFP4G32_RESIDUAL_WMMA_SRC,
            "gemm_hfp4g32_residual_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ap = a.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yp = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes =
            crate::profile::gemv_hfp4g32_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfp4g32_residual_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_hfp4g32_residual_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    pub fn gemm_hfp4g32_residual_wmma_gfx12(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfp4g32_residual_wmma_gfx12",
            kernels::GEMM_HFP4G32_RESIDUAL_WMMA_GFX12_SRC,
            "gemm_hfp4g32_residual_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ap = a.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yp = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes =
            crate::profile::gemv_hfp4g32_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfp4g32_residual_wmma_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_hfp4g32_residual_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFP4-G32 batched 2-way fused GEMM (gate + up). Routes gfx11/gfx12.
    pub fn gemm_gate_up_hfp4g32(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.gemm_gate_up_hfp4g32_wmma_gfx12(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        self.gemm_gate_up_hfp4g32_wmma(a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size)
    }

    pub fn gemm_gate_up_hfp4g32_wmma(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfp4g32_wmma",
            kernels::GEMM_GATE_UP_HFP4G32_WMMA_SRC,
            "gemm_gate_up_hfp4g32_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut gm_val = gate_m as i32;
        let mut um_val = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut gm_val as *mut _ as *mut c_void,
            &mut um_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfp4g32_bytes(gate_m, k)
            + crate::profile::gemv_hfp4g32_bytes(up_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfp4g32_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfp4g32_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(gm_val);
                b.push_i32(um_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    pub fn gemm_gate_up_hfp4g32_wmma_gfx12(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfp4g32_wmma_gfx12",
            kernels::GEMM_GATE_UP_HFP4G32_WMMA_GFX12_SRC,
            "gemm_gate_up_hfp4g32_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut gm_val = gate_m as i32;
        let mut um_val = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut gm_val as *mut _ as *mut c_void,
            &mut um_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfp4g32_bytes(gate_m, k)
            + crate::profile::gemv_hfp4g32_bytes(up_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_hfp4g32_wmma_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfp4g32_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(gm_val);
                b.push_i32(um_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// WMMA-accelerated batched 2-way fused HFQ4-G256 GEMM (gate + up).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    pub fn gemm_gate_up_hfq4g256_wmma(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // LDS-staged multi-wave gate_up (env HIPFIRE_HFQ4G256_LDSSTAGE=1).
        // gfx11 sister of the gfx12 ldsstage path. Reorders FP32 K accumulation;
        // K must be a multiple of 512. Batch ceiling: see the measured 3.6-trunk
        // tables on LDSSTAGE_MAX_BATCH_GFX11 (one value covers dGPU and iGPU).
        if self.flags.hfq4g256_ldsstage_wmma
            && k % 512 == 0
            && batch_size <= LDSSTAGE_MAX_BATCH_GFX11
        {
            let kname = "gemm_gate_up_hfq4g256_wmma_ldsstage";
            let ksrc = kernels::GEMM_GATE_UP_HFQ4G256_WMMA_SRC;
            self.ensure_kernel(kname, ksrc, kname)?;
            let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

            let mut ag = a_gate.buf.as_ptr();
            let mut au = a_up.buf.as_ptr();
            let mut xp = x_f16_ptr;
            let mut yg = y_gate.buf.as_ptr();
            let mut yu = y_up.buf.as_ptr();
            let mut g_m = gate_m as i32;
            let mut u_m = up_m as i32;
            let mut k_val = k as i32;
            let mut n_val = batch_size as i32;

            let mut params: Vec<*mut c_void> = vec![
                &mut ag as *mut _ as *mut c_void,
                &mut au as *mut _ as *mut c_void,
                &mut xp as *mut _ as *mut c_void,
                &mut yg as *mut _ as *mut c_void,
                &mut yu as *mut _ as *mut c_void,
                &mut g_m as *mut _ as *mut c_void,
                &mut u_m as *mut _ as *mut c_void,
                &mut k_val as *mut _ as *mut c_void,
                &mut n_val as *mut _ as *mut c_void,
            ];

            let total_m = gate_m + up_m;
            let row_tiles = (total_m + 15) / 16;
            let batch_tiles = (batch_size + 15) / 16;

            let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                + crate::profile::gemv_hfq4g256_bytes(up_m, k)
                + batch_size * k * 2
                + batch_size * total_m * 4 * 2;
            let timer = crate::profile::begin_timer(&self.hip, "gemm", kname, bytes);
            let result = self.launch_maybe_blob(
                kname,
                [row_tiles as u32, batch_tiles as u32, 1],
                [256, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(ag);
                    b.push_ptr(au);
                    b.push_ptr(xp);
                    b.push_ptr(yg);
                    b.push_ptr(yu);
                    b.push_i32(g_m);
                    b.push_i32(u_m);
                    b.push_i32(k_val);
                    b.push_i32(n_val);
                    b
                },
            );
            if let Some(t) = timer {
                t.finish(&self.hip);
            }
            return result;
        }

        // HIPFIRE_GATE_UP_VARIANT=ldsx routes to the LDS-staged X variant
        // (Gate 1 microbench, opt-in only, default off). See
        // docs/perf-checkpoints/2026-05-01-gate-up-lds-x-share-plan.md.
        let variant_override = self.flags.gate_up_variant.clone();
        // (kernel_name, kernel_src, m_tile, block_threads). m_tile is the
        // per-block row count; block_threads is the wave/block size.
        let (kernel_name, kernel_src, m_tile, block_threads) = match variant_override.as_deref() {
            Some("ldsx") => (
                "gemm_gate_up_hfq4g256_wmma_ldsx",
                kernels::GEMM_GATE_UP_HFQ4G256_WMMA_LDSX_SRC,
                16,
                32,
            ),
            // k4 = 4-tile pipeline (more in-flight B loads for better BW
            // utilization). Opt-in default-off; bench-measured 2026-05-21.
            Some("k4") => (
                "gemm_gate_up_hfq4g256_wmma_k4",
                kernels::GEMM_GATE_UP_HFQ4G256_WMMA_K4_SRC,
                16,
                32,
            ),
            // ldscoop = cooperative LDS weight staging for coalesced DRAM
            // loads. All 32 threads load one row's weights at a time
            // (128-byte coalesced cache lines), staged in LDS for the
            // WMMA loop. Targets the 32% peak BW seen in base kernel.
            Some("ldscoop") => (
                "gemm_gate_up_hfq4g256_wmma_ldscoop",
                kernels::GEMM_GATE_UP_HFQ4G256_WMMA_LDSCOOP_SRC,
                16,
                32,
            ),
            // 2tile = 32 rows × 16 cols per block, 2 wave32 waves.
            // Halves grid in M; both waves share the same X tile so
            // L0/L1 cache absorbs the second wave's loads cheaply.
            Some("2tile") => (
                "gemm_gate_up_hfq4g256_wmma_2tile",
                kernels::GEMM_GATE_UP_HFQ4G256_WMMA_2TILE_SRC,
                32,
                64,
            ),
            _ => {
                let def = if self.arch_caps.is_rdna3p5() {
                    // RDNA3.5 iGPU (gfx1150/1151/1152): narrow-BW LPDDR5; the
                    // ldscoop_nosync variant wins here. gfx1152 previously fell
                    // through to the RDNA4 `else` ldscoop arm because the old
                    // string-prefix test ("gfx1151"/"gfx1150" starts_with) did
                    // not match "gfx1152" — that was a misroute, now fixed by
                    // gating on the is_rdna3p5 capability molecule.
                    (
                        "gemm_gate_up_hfq4g256_wmma_ldscoop_nosync",
                        kernels::GEMM_GATE_UP_HFQ4G256_WMMA_LDSCOOP_NOSYNC_SRC,
                        16,
                        32,
                    )
                } else if self.arch_caps.is_rdna3_dgpu() {
                    // gfx1100/1101/1102 (RDNA3 dGPU): ldscoop's LDS weight-staging
                    // overhead exceeds its coalescing gain here. Falsified in
                    // 303d69e9 ("ldscoop variant FALSIFIED — LDS overhead exceeds
                    // coalescing gains") and re-confirmed by rocprofv3 2026-06-12:
                    // on the 27B DFlash batched-verify gate_up the ldscoop variant
                    // ran 0.343 ms/launch vs the plain WMMA 0.232 (+48%), the bulk
                    // of a ~14% DFlash decode regression vs the ca30ca21 baseline.
                    // e3232034 ("ldscoop on others") tuned the default for gfx1151
                    // (nosync) but dumped RDNA3 dGPUs into the falsified ldscoop.
                    // Restore the plain WMMA variant that ca30ca21 (dense-DFlash
                    // perfmaxx) shipped. RDNA4 (else arm) is left on ldscoop pending
                    // its own measurement on hiptrx/gfx1201.
                    (
                        "gemm_gate_up_hfq4g256_wmma",
                        kernels::GEMM_GATE_UP_HFQ4G256_WMMA_SRC,
                        16,
                        32,
                    )
                } else {
                    (
                        "gemm_gate_up_hfq4g256_wmma_ldscoop",
                        kernels::GEMM_GATE_UP_HFQ4G256_WMMA_LDSCOOP_SRC,
                        16,
                        32,
                    )
                };
                def
            }
        };
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + m_tile - 1) / m_tile;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [block_threads as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(g_m);
                b.push_i32(u_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ3-G256 sister of `gemm_gate_up_hfq4g256_wmma`. Same WMMA shape
    /// + lane decomposition; only the inner K-tile unpack differs (3-bit
    /// cross-byte vs 4-bit nibble) and the per-group byte stride is 104
    /// instead of 136. Used for MQ3 prefill via `gemm_gate_up_mq3g256_wmma`.
    /// gfx12 (RDNA4) sister of `gemm_qkv_hfq3g256_wmma`.
    pub fn gemm_qkv_hfq3g256_wmma_gfx12(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq3g256_wmma_gfx12",
            kernels::GEMM_QKV_HFQ3G256_WMMA_GFX12_SRC,
            "gemm_qkv_hfq3g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = total_m * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq3g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq3g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    pub fn gemm_gate_up_hfq3g256_wmma(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let total_m = gate_m + up_m;
        let arch_supports_mb4 = self.arch_caps.supports_mq3_lloyd_mb4();
        let use_mb4 = match self.flags.mq3_mb4 {
            None => arch_supports_mb4 && batch_size >= 128 && total_m >= 4096,
            Some(_) => arch_supports_mb4,
        };
        if use_mb4 {
            return self.gemm_gate_up_hfq3g256_wmma_mb4(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.gemm_gate_up_hfq3g256_wmma_gfx12(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_gate_up_hfq3g256_wmma",
            kernels::GEMM_GATE_UP_HFQ3G256_WMMA_SRC,
            "gemm_gate_up_hfq3g256_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = (gate_m + up_m) * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq3g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq3g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(g_m);
                b.push_i32(u_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ3 gate_up mb4 dispatch: 16×64 output tile per WG.
    pub fn gemm_gate_up_hfq3g256_wmma_mb4(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq3g256_wmma_mb4",
            kernels::GEMM_GATE_UP_HFQ3G256_WMMA_MB4_SRC,
            "gemm_gate_up_hfq3g256_wmma_mb4",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m_v = gate_m as i32;
        let mut u_m_v = up_m as i32;
        let mut k_v = k as i32;
        let mut n_v = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m_v as *mut _ as *mut c_void,
            &mut u_m_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 63) / 64;
        let bytes = total_m * (k / 256) * 104 + batch_size * k * 2 + batch_size * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq3g256_wmma_mb4", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq3g256_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(g_m_v);
                b.push_i32(u_m_v);
                b.push_i32(k_v);
                b.push_i32(n_v);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx12 (RDNA4) sister of `gemm_gate_up_hfq3g256_wmma`.
    pub fn gemm_gate_up_hfq3g256_wmma_gfx12(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq3g256_wmma_gfx12",
            kernels::GEMM_GATE_UP_HFQ3G256_WMMA_GFX12_SRC,
            "gemm_gate_up_hfq3g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = total_m * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_hfq3g256_wmma_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq3g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(g_m);
                b.push_i32(u_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ3 wrapper for `gemm_gate_up_hfq3g256_wmma`: pre-rotates X then
    /// dispatches the HFQ3 kernel. See `gemm_qkvza_mq3g256_wmma` for
    /// the cache-invalidation rationale.
    pub fn gemm_gate_up_mq3g256_wmma(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        for b in 0..batch_size {
            let x_row = x.sub_offset(b * k, k);
            let x_rot_row = x_rot.sub_offset(b * k, k);
            self.rotate_x_mq(&x_row, &x_rot_row, k)?;
        }
        self.scratch.fp16_x_source_ptr = std::ptr::null_mut();
        self.gemm_gate_up_hfq3g256_wmma(
            a_gate, a_up, x_rot, y_gate, y_up, gate_m, up_m, k, batch_size,
        )
    }

    /// gfx12 (RDNA4) sister of `gemm_gate_up_hfq4g256_wmma`. Same recipe
    /// as the QKV gfx12 scaffold (validated on R9700). Not yet wired into
    /// the public dispatch tree — exposed only for the channel-test
    /// harness. See issue #54.
    pub fn gemm_gate_up_hfq4g256_wmma_gfx12(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // LDS-staged multi-wave gate_up (env HIPFIRE_HFQ4G256_LDSSTAGE=1).
        // Reorders FP32 K accumulation; K must be a multiple of 512.
        //
        // SMALL-BATCH ONLY (<= LDSSTAGE_MAX_BATCH). Measured 2026-07-29 in-engine
        // on the pinned Qwen3.6-27B trunk (86a5f80f…), gfx1201 R9700, matched
        // A/B/B/A, `hipfire bench --matrix --runs 5 --warmups 10`:
        //
        //   batch:   16      32      64     128     256     512    1024
        //   ratio: 1.0902  1.0489  1.0313  0.9152  0.9450  0.9465  0.9492
        //
        // The win inverts between 64 and 128: cooperative LDS staging pays for
        // itself while the grid is small, and past that the added LDS traffic +
        // the __launch_bounds__(256,4) occupancy cap cost more than the coalesced
        // loads save. Spec-decode verify runs THIS path at B = trained block 16
        // (`speculative.rs` calls `forward_prefill_batch_single_chunk_captured_opts`),
        // so the gate keeps the verify win and leaves bulk prefill untouched.
        // Cross-validated: the B=16 row (1.0902x) reproduces the 27B DFlash golden
        // run's end-to-end +9.1% (254.2 -> 277.4 tok/s) from a separate instrument.
        //
        // A 0.8b canary is NOT valid for this decision — it measured +11..+23% at
        // batch 512-8192 where 27B measures -3.5..-5.4% (see
        // .research/round5-verify-cost.md §17).
        if self.flags.hfq4g256_ldsstage_wmma && k % 512 == 0 && batch_size <= LDSSTAGE_MAX_BATCH {
            let kname = "gemm_gate_up_hfq4g256_wmma_gfx12_ldsstage";
            let ksrc = kernels::GEMM_GATE_UP_HFQ4G256_WMMA_GFX12_SRC;
            self.ensure_kernel(kname, ksrc, kname)?;
            let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

            let mut ag = a_gate.buf.as_ptr();
            let mut au = a_up.buf.as_ptr();
            let mut xp = x_f16_ptr;
            let mut yg = y_gate.buf.as_ptr();
            let mut yu = y_up.buf.as_ptr();
            let mut g_m = gate_m as i32;
            let mut u_m = up_m as i32;
            let mut k_val = k as i32;
            let mut n_val = batch_size as i32;

            let mut params: Vec<*mut c_void> = vec![
                &mut ag as *mut _ as *mut c_void,
                &mut au as *mut _ as *mut c_void,
                &mut xp as *mut _ as *mut c_void,
                &mut yg as *mut _ as *mut c_void,
                &mut yu as *mut _ as *mut c_void,
                &mut g_m as *mut _ as *mut c_void,
                &mut u_m as *mut _ as *mut c_void,
                &mut k_val as *mut _ as *mut c_void,
                &mut n_val as *mut _ as *mut c_void,
            ];

            let total_m = gate_m + up_m;
            let row_tiles = (total_m + 15) / 16;
            let batch_tiles = (batch_size + 15) / 16;

            let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                + crate::profile::gemv_hfq4g256_bytes(up_m, k)
                + batch_size * k * 2
                + batch_size * total_m * 4 * 2;
            let timer = crate::profile::begin_timer(&self.hip, "gemm", kname, bytes);
            let result = self.launch_maybe_blob(
                kname,
                [row_tiles as u32, batch_tiles as u32, 1],
                [256, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(ag);
                    b.push_ptr(au);
                    b.push_ptr(xp);
                    b.push_ptr(yg);
                    b.push_ptr(yu);
                    b.push_i32(g_m);
                    b.push_i32(u_m);
                    b.push_i32(k_val);
                    b.push_i32(n_val);
                    b
                },
            );
            if let Some(t) = timer {
                t.finish(&self.hip);
            }
            return result;
        }
        // Adaptive-B batch-tile fast path (env HIPFIRE_GATE_UP_BT): B independent
        // accumulator chains hide the WMMA latency that caps the 1-acc kernel at ~19%
        // of peak. B = clamp(N/16, 1, 12), capped at 12 (B=16 spills VGPR). Byte-exact
        // vs the 1-acc kernel; +85% at N=192 on gfx1201.
        let bt_b: usize = if hipfire_config::developer_var("HIPFIRE_GATE_UP_BT")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(true)
        {
            if batch_size < 64 {
                1
            } else if batch_size % 192 == 0 {
                12
            } else if batch_size % 128 == 0 {
                8
            } else if batch_size % 64 == 0 {
                4
            } else if batch_size >= 192 {
                12
            } else if batch_size >= 128 {
                8
            } else {
                4
            }
        } else {
            1
        };
        let (kname, ksrc): (&str, &str) = match bt_b {
            12 => (
                "gemm_gate_up_hfq4g256_wmma_gfx12_bt12",
                kernels::GEMM_GATE_UP_HFQ4G256_WMMA_GFX12_BT_SRC,
            ),
            8 => (
                "gemm_gate_up_hfq4g256_wmma_gfx12_bt8",
                kernels::GEMM_GATE_UP_HFQ4G256_WMMA_GFX12_BT_SRC,
            ),
            4 => (
                "gemm_gate_up_hfq4g256_wmma_gfx12_bt4",
                kernels::GEMM_GATE_UP_HFQ4G256_WMMA_GFX12_BT_SRC,
            ),
            _ => (
                "gemm_gate_up_hfq4g256_wmma_gfx12",
                kernels::GEMM_GATE_UP_HFQ4G256_WMMA_GFX12_SRC,
            ),
        };
        self.ensure_kernel(kname, ksrc, kname)?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 16 * bt_b - 1) / (16 * bt_b);

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kname, bytes);
        let result = self.launch_maybe_blob(
            kname,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(g_m);
                b.push_i32(u_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx12 (RDNA4) sister of `gemm_hfq4g256_residual_wmma` (specifically
    /// the `_k2` variant — the gfx11 dispatch default for M >= 8192, with
    /// the validated C-output mapping).
    ///
    /// Closes the residual-GEMM gap on 9B prefill: before this kernel,
    /// gfx12 fell through to the dot2 fp16 fallback for the residual call
    /// site (attn-out + ffn-down), which accounted for ~42% of 9B prefill
    /// time on R9700. The other six gfx12 WMMA kernels shipped in PR #62.
    ///
    /// Same recipe as the qkv / qkvza / gate_up gfx12 ports: `_w32_gfx12`
    /// builtin, half8_t operands, K-split via `tid >> 4`, contiguous
    /// C-row mapping (`acc[j] = C[8*(tid>>4) + j][tid & 15]`). Validated
    /// on R9700 by the `test_wmma_residual_gfx12` channel-test against
    /// the dot2 reference path.
    pub fn gemm_hfq4g256_residual_wmma_gfx12(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // LDS-staged multi-wave residual (env HIPFIRE_HFQ4G256_LDSSTAGE=1).
        // Reorders FP32 K accumulation; K must be a multiple of 512.
        // SMALL-BATCH ONLY — see the measured batch/ratio table on
        // `gemm_gate_up_hfq4g256_wmma_gfx12` above. Same crossover applies: this
        // site serves BOTH o_proj and down_proj in dense prefill/verify
        // (`GemmHfq4G256Residual`), so an ungated enable regressed 27B prefill
        // 3.5-5.4% while the verify regime (B=16) gains ~9%.
        if self.flags.hfq4g256_ldsstage_wmma && k % 512 == 0 && batch_size <= LDSSTAGE_MAX_BATCH {
            let kname = "gemm_hfq4g256_residual_wmma_gfx12_ldsstage";
            let ksrc = kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_SRC;
            self.ensure_kernel(kname, ksrc, kname)?;
            let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

            let mut a_ptr = a_raw.buf.as_ptr();
            let mut x_ptr = x_f16_ptr;
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

            let row_tiles = (m + 15) / 16;
            let batch_tiles = (batch_size + 15) / 16;

            let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
                + batch_size * k * 2
                + batch_size * m * 4 * 2;
            let timer = crate::profile::begin_timer(&self.hip, "gemm", kname, bytes);
            let result = self.launch_maybe_blob(
                kname,
                [row_tiles as u32, batch_tiles as u32, 1],
                [256, 1, 1],
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
            return result;
        }
        // Adaptive-B batch-tile (env HIPFIRE_GATE_UP_BT, shared with gate_up/qkvza).
        let bt_b: usize = if hipfire_config::developer_var("HIPFIRE_GATE_UP_BT")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(true)
        {
            if batch_size < 64 {
                1
            } else if batch_size % 192 == 0 {
                12
            } else if batch_size % 128 == 0 {
                8
            } else if batch_size % 64 == 0 {
                4
            } else if batch_size >= 192 {
                12
            } else if batch_size >= 128 {
                8
            } else {
                4
            }
        } else {
            1
        };
        let (kname, ksrc): (&str, &str) = match bt_b {
            12 => (
                "gemm_hfq4g256_residual_wmma_gfx12_bt12",
                kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_BT_SRC,
            ),
            8 => (
                "gemm_hfq4g256_residual_wmma_gfx12_bt8",
                kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_BT_SRC,
            ),
            4 => (
                "gemm_hfq4g256_residual_wmma_gfx12_bt4",
                kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_BT_SRC,
            ),
            _ => (
                "gemm_hfq4g256_residual_wmma_gfx12",
                kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_SRC,
            ),
        };
        self.ensure_kernel(kname, ksrc, kname)?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 16 * bt_b - 1) / (16 * bt_b);

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kname, bytes);
        let result = self.launch_maybe_blob(
            kname,
            [row_tiles as u32, batch_tiles as u32, 1],
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

    /// HFQ4-G256 GEMV with fused residual add: y[row] += A[row] · x.
    /// Same math as `gemv_hfq4g256` but the final write accumulates into `y`
    /// instead of overwriting. Used for wo / w_down projections where the
    /// following step would have been `x += gemv_out` via add_inplace_f32.

    /// HFQ4-G256 GEMV with fused SCALED residual add, CPU-scalar variant:
    ///   y[row] += scale * (A[row] · x)
    /// where `scale` is host-supplied by kernarg. Replaces the three-kernel
    /// tail of the MoE routed-expert epilogue (gemv → scale → add_inplace)
    /// with a single launch. Bit-exact with gemv_hfq4g256_residual followed
    /// by scaled_add_inplace_cpu_scalar when the inputs are identical —
    /// same accumulator layout, same pairwise combine.

    /// HFQ4-G256 GEMV with fused SCALED residual add, GPU-scalar variant:
    ///   y[row] += c_buf[0] * (A[row] · x)
    /// Reads the scale from a 1-element device buffer. Used by the MoE
    /// shared-expert epilogue where `c_buf` holds sigmoid(gate · x) computed
    /// entirely on-device, avoiding a D2H sync.

    /// Same as `gemv_hfq4g256_residual_scaled_gpu` but applies sigmoid to
    /// `c_buf[0]` before scaling — lets the caller skip a separate
    /// `sigmoid_f32` launch on the 1-elem shared-expert gate scalar.
    /// Used by the A3B MoE FFN shared-expert down path.

    /// N-batched variant of `gemv_hfq4g256_residual_sigmoid_scaled_gpu`.
    /// `x_batch` is [N × K], `y_batch` is [N × M], `c_batch` is [N]. Each
    /// (row, token) block runs the HFQ4G256 GEMV body on its token's x
    /// row and atomicAdd's `sigmoid(c_batch[token]) * acc` into
    /// `y_batch[token × M + row]`. Used by the batched MoE FFN shared-
    /// expert down projection to eliminate N per-token launches.
    #[allow(clippy::too_many_arguments)]

    /// HFQ4-G128 batched GEMV with fused per-token sigmoid-scaled residual.
    ///
    /// y_batch[token, row] += sigmoid(c_batch[token]) * (A[row] · x_batch[token])
    ///
    /// HFQ4-G128 layout: 72 bytes per 128-element group (vs HFQ4-G256's
    /// 136 B/256-element group). Used by the PARO shared-expert down
    /// dispatch in `prefill_moe_ffn_body_batched` (Phase 2 — admit gated
    /// behind HIPFIRE_PARO_BATCHED=1). Same grid/block contract as the
    /// HFQ4-G256 sister: grid=[M × batch_size × 1], block=[32 × 1 × 1].

    /// HFQ6/MQ6 analogue of `gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched`.
    /// Same kernel shape (grid = `M × batch`, block = 32, one warp per
    /// `(row, token)`), but reads HFQ6's 200 B / group layout (4 B scale +
    /// 4 B zero + 192 B packed 6-bit nibbles). MQ6G256 shares storage with
    /// HFQ6G256 — caller applies the FWHT rotation upstream, same convention
    /// as MQ4 / HFQ4. Used by the batched MoE FFN shared-expert `down`
    /// projection in the AWQ-style mixed-precision path where shared.down
    /// is MQ6 (12 of 40 layers in AWQ A3B fall into this case).
    #[allow(clippy::too_many_arguments)]

    /// MoE fused gate_up GEMV: runs 8 top-K experts' HFQ4-G256 GEMV in a
    /// single launch. Caller passes the 8 selected experts' weight
    /// tensors (in top-K order); the kernel's grid.y picks which expert
    /// each block uses. Outputs are SPLIT into `y_gate` (first mi rows of
    /// each expert) and `y_up` (second mi rows), both `[k_top × mi]`
    /// row-major, so the next-stage batched silu_mul_rotate can consume
    /// them as plain [batch × K] buffers without extra strided reads.
    ///
    /// Bit-exact with running `gemv_hfq4g256` 8 times (same accumulator
    /// layout and pairwise final combine). `k_top` is currently hardcoded
    /// to 8 to match A3B; a generic path can follow alongside Phase 2b.
    #[allow(clippy::too_many_arguments)]

    /// MoE fused down GEMV with scaled residual: accumulates 8 top-K
    /// experts' weighted contributions into `x_residual` in a single
    /// kernel launch. Grid.y selects the expert; each block atomicAdds
    /// `s_rank * (W_rank[row] · rot_batch[rank, :])` into `x_residual[row]`.
    /// Replaces 8 separate `gemv_hfq4g256_residual_scaled_cpu` calls.
    ///
    /// Atomic-add summation order is non-deterministic, so bit-exactness
    /// across runs isn't guaranteed (vs the sequential per-expert path).
    /// For A3B the MoE contribution is added on top of a non-trivial base,
    /// so the ordering-dependent FP noise is tiny in practice and the
    /// smoke-test decode still matches the Phase 2c step 2 output.
    #[allow(clippy::too_many_arguments)]

    /// MoE router GPU softmax + top-K + (optional) renormalize. One
    /// workgroup, no D2H sync. Writes [k_top] i32 indices and [k_top]
    /// f32 weights to device buffers. Hardcoded k_top=8 to match A3B.

    /// MoE top-K + renorm given pre-softmaxed probs. Companion to the
    /// regular `softmax_f32`. The dispatch site runs `softmax_f32` first,
    /// then this kernel — same softmax math everywhere, no 1-ULP
    /// divergence between the routing path and a CPU reference.

    /// Index-aware MoE gate_up GEMV. Reads expert_ids from a device-side
    /// topk_indices buffer and weight bases from expert_ptrs[expert_id].
    /// hipGraph-capture-safe replacement for the kernarg-pointer variant.
    #[allow(clippy::too_many_arguments)]

    /// HFQ4G128 (ParoQuant) variant of the indexed MoE gate_up GEMV.
    /// wave32-only (gfx10/11/12) — no wave64 path yet because ParoQuant
    /// A3B is not currently validated on gfx94x.

    /// Index-aware MoE down GEMV with scaled residual. Same pattern as
    /// the indexed gate_up; also reads scales from a device topk_weights
    /// buffer and atomicAdds the contribution into x_residual.
    #[allow(clippy::too_many_arguments)]

    /// N-batched MoE softmax + top-K + renorm. Grid = (N, 1, 1); one
    /// workgroup per token. `logits` is [N × n_exp], `topk_idx` is
    /// [N × K_TOP] i32, `topk_w` is [N × K_TOP] f32.

    /// Batched companion of `moe_topk_renorm_k8` for the prefill path.
    /// Takes pre-softmaxed probs of shape `[batch_size × n_exp]` and writes
    /// `[batch_size × K_TOP]` indices and weights. Caller must run a batched
    /// softmax (`gpu.softmax_f32` on a [batch_size × n_exp] tensor) before
    /// calling this kernel.

    /// N-batched indexed MoE gate_up. Grid = (M, K_TOP, N). `x` is
    /// [N × K], `topk_indices` is [N × K_TOP] i32, `y_gate` and `y_up`
    /// are [N × K_TOP × MI] where MI = M / 2.
    #[allow(clippy::too_many_arguments)]

    /// N-batched indexed MoE down + scaled residual. Grid = (M, K_TOP, N).
    /// `rot_batch` is [N × K_TOP × K], `x_residual` is [N × M]; the kernel
    /// atomicAdd's per-token slices. `topk_indices` / `topk_weights` are
    /// [N × K_TOP].
    #[allow(clippy::too_many_arguments)]

    /// Atomic-free counterpart to
    /// `gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched`. Writes
    /// each (token, krank) result to its own row of `expert_outputs`
    /// ([N × K_TOP × M], f32) instead of atomicAdd'ing the scaled sum into
    /// `x_residual`. Pair with `moe_down_combine_k8_batched` to fold the
    /// K_TOP slots back into the residual with topk_weights applied.
    ///
    /// Observed lift on R9700/gfx1201: 387 → ~900 GiB/s for the down GEMV
    /// (no K_TOP-way atomic contention per output cell). Wave32-only
    /// (RDNA) for now — the CDNA wave64 path stays on the residual_scaled
    /// kernel; atomicAdd on HBM is faster there and the contention pattern
    /// is different.
    #[allow(clippy::too_many_arguments)]

    /// HFQ4G128 (ParoQuant) variant of the atomic-free batched indexed
    /// MoE down. Same expanded-output contract as the HFQ4G256 sibling;
    /// caller must follow with `moe_down_combine_k8_batched` to fold the
    /// K_TOP slots into x_residual with topk_weights applied. wave32-only.
    #[allow(clippy::too_many_arguments)]
    /// N-batched indexed MoE gate_up GEMV for HFQ4G128 (ParoQuant routed
    /// experts). Sister of `gemv_hfq4g256_moe_gate_up_k8_indexed_batched`
    /// with 72 B/group stride. The caller MUST pre-rotate x using the
    /// layer's shared `gate_up` Givens sidecar (givens_rotate_to into
    /// x_rot_batch) before calling — this kernel is rotation-agnostic and
    /// just reads HFQ4G128 nibbles. Grid: (M, K_TOP, N) wave32.
    #[allow(clippy::too_many_arguments)]

    /// Index-aware MoE gate_up GEMV for HFQ6G256-layout routed experts.
    /// Wave32 (RDNA) only — CDNA wave64 path stays on the residual_scaled
    /// kernel family. Used to keep mixed-kmap A3B (post-PR-199 alternating
    /// MQ4→MQ6 promotion) on the device-side top-K path under hipGraph
    /// capture.
    #[allow(clippy::too_many_arguments)]

    /// HFQ6G256 counterpart to `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded`.
    /// Atomic-free expand-then-combine for the MoE down step. Pairs with
    /// `moe_down_combine_k8_batched` (dtype-independent — operates on the
    /// f32 expanded buffer). Wave32 (RDNA) only.
    #[allow(clippy::too_many_arguments)]

    /// Path 2 grouped-WMMA-GEMM for MoE prefill (gate_up or down).
    /// Each WMMA tile picks its expert via `expert_tile_ids[tile_y]` and
    /// gathers its B-operand rows via `sorted_slot_index`; -1 padding
    /// lanes contribute zeros. Writes `Y_grouped[m_total × M]` direct.
    ///
    /// The companion combine kernel (Stage 3) fans Y_grouped back to the
    /// per-token gate_batch/up_batch streams (or applies topk_weights for
    /// the down combine).
    /// `x_row_div` selects the X gather layout:
    ///   gate_up: x_src = x_rot_batch [N × K], x_row_div = K_TOP
    ///   down:    x_src = rot_batch [N*K_TOP × K], x_row_div = 1
    /// `x_src_rows` is the number of rows in x_src (N or N*K_TOP).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_wmma_k2(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to FP16)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx1151 (Strix Halo iGPU) i8 MMQ port: lift the compute ceiling
        // from ~71 (FP16 WMMA) to ~140 TFLOPS (i8 WMMA). Opt-out via
        // HIPFIRE_MOE_GROUPED_I8=0; default ON for gfx1151 only.
        let use_i8_gfx1151 =
            self.arch_caps.is_gfx1151() && self.flags.moe_grouped_i8.unwrap_or(true);
        if use_i8_gfx1151 {
            // Optional deeper-pipeline variants (opt-IN, default OFF).
            // Same kernarg layout + scatter contract as the k2 default.
            // - k8: processes all 4 sub-blocks of one Q8_1 block per inner
            //   iteration (8 WMMAs into 4 independent int32 accumulators).
            // - k4: pairs adjacent Q8_1 sub-blocks (4 WMMAs into 2 accumulators).
            // - k2 (default): one sub-block per inner iteration.
            let use_k8 = self.flags.moe_grouped_i8_k8;
            let use_k4 = self.flags.moe_grouped_i8_k4;
            if use_k8 {
                return self.gemm_hfq4g256_moe_grouped_mmq_k8_gfx1151(
                    expert_weight_ptrs,
                    expert_tile_ids,
                    sorted_slot_index,
                    x_src,
                    y_grouped,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    x_src_rows,
                );
            }
            if use_k4 {
                return self.gemm_hfq4g256_moe_grouped_mmq_k4_gfx1151(
                    expert_weight_ptrs,
                    expert_tile_ids,
                    sorted_slot_index,
                    x_src,
                    y_grouped,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    x_src_rows,
                );
            }
            return self.gemm_hfq4g256_moe_grouped_mmq_gfx1151(
                expert_weight_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x_src,
                y_grouped,
                m,
                k,
                x_row_div,
                m_total,
                x_src_rows,
            );
        }
        // gfx11 dGPU i8 MMQ port (gfx1100/1101/1102/1103 — 7900 XTX, 7800/
        // 7700, 7600, Phoenix mobile). Same lift as gfx1151: doubles the
        // compute ceiling on this compute-bound grouped GEMM path.
        // Opt-out via HIPFIRE_MOE_GROUPED_I8=0; default ON for these archs.
        let use_i8_gfx11_dgpu = (self.arch.starts_with("gfx1100")
            || self.arch.starts_with("gfx1101")
            || self.arch.starts_with("gfx1102")
            || self.arch.starts_with("gfx1103"))
            && self.flags.moe_grouped_i8.unwrap_or(true);
        if use_i8_gfx11_dgpu {
            // k4 default ON: deeper K-tile pipeline gives +2.8% over k2 on
            // gfx1100 (A/B confirmed 2026-05-19 k9lin 7900 XTX); same
            // structural pattern as gfx1151's +4.6%. k2 alone was a wash vs
            // FP16, so k4 is what makes the dGPU i8 path actually worth
            // shipping. Opt out with HIPFIRE_MOE_GROUPED_I8_K4=0.
            let use_k4 = self.flags.moe_grouped_i8_k4;
            if use_k4 {
                return self.gemm_hfq4g256_moe_grouped_mmq_k4_gfx11_dgpu(
                    expert_weight_ptrs,
                    expert_tile_ids,
                    sorted_slot_index,
                    x_src,
                    y_grouped,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    x_src_rows,
                );
            }
            return self.gemm_hfq4g256_moe_grouped_mmq_gfx11_dgpu(
                expert_weight_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x_src,
                y_grouped,
                m,
                k,
                x_row_div,
                m_total,
                x_src_rows,
            );
        }
        // gfx12 (RDNA4 — R9700/gfx1201, gfx1200) i8 MMQ port. Correctness PASS
        // (NRMSE ~0.4% on A3B shapes vs FP16 reference) but empirical perf on
        // 2026-05-19 R9700 A3B prefill (256-token, 5-run median): 2960 → 2607
        // tok/s = **-11.6% regression**. Per-call kernel time 279µs (FP16) →
        // 408µs (i8) = +46% kernel slowdown. Theoretical 2× i8 WMMA FLOP rate
        // is offset by per-sub-block scale FMA dependency chain (8 INT→FLOAT
        // conversions + 16 FMAs per sub-block, fully serial after each WMMA
        // pair). Same pattern as documented synth-win → prod-falsify cases
        // (FP8 WMMA HFQ4G32 2026-05-10, gfx11 dot2 trickle-down 2026-05-11).
        // Shipped as opt-in research artifact; default OFF for gfx12.
        // Opt-in via HIPFIRE_MOE_GROUPED_I8=1 to evaluate on other shapes.
        let use_i8_gfx12 = self.arch_caps.is_rdna4() && self.flags.moe_grouped_i8.unwrap_or(false);
        if use_i8_gfx12 {
            // k4 variant: 4 sub-blocks paired per inner iteration, 8 WMMAs
            // into 4 independent int32 accumulators before scale-FMA chain
            // resolves. Experimental — separate gate from the gfx11_dgpu k4
            // (which is default-on) because the gfx12 i8 path itself is
            // default-off pending recovery from the -11.6% regression vs FP16.
            let use_k4 = self.flags.moe_grouped_i8_k4_gfx12;
            if use_k4 {
                return self.gemm_hfq4g256_moe_grouped_mmq_k4_gfx12(
                    expert_weight_ptrs,
                    expert_tile_ids,
                    sorted_slot_index,
                    x_src,
                    y_grouped,
                    m,
                    k,
                    x_row_div,
                    m_total,
                    x_src_rows,
                );
            }
            return self.gemm_hfq4g256_moe_grouped_mmq_gfx12(
                expert_weight_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x_src,
                y_grouped,
                m,
                k,
                x_row_div,
                m_total,
                x_src_rows,
            );
        }
        // gfx12 (RDNA4) needs the _gfx12 WMMA intrinsic; gfx11 (RDNA3) and
        // older RDNA archs use the base _w32 intrinsic from the k2 sibling.
        let is_gfx12 = self.arch_caps.is_rdna4();
        // 2×1 M-direction reg-blocked variant (gfx12 only for now). Env-gated.
        let use_m2 = is_gfx12 && self.flags.moe_grouped_m2;
        let (kernel_name, kernel_src) = if use_m2 {
            (
                "gemm_hfq4g256_moe_grouped_wmma_m2_gfx12",
                kernels::GEMM_HFQ4G256_MOE_GROUPED_WMMA_M2_GFX12_SRC,
            )
        } else if is_gfx12 {
            (
                "gemm_hfq4g256_moe_grouped_wmma_gfx12",
                kernels::GEMM_HFQ4G256_MOE_GROUPED_WMMA_GFX12_SRC,
            )
        } else {
            (
                "gemm_hfq4g256_moe_grouped_wmma_k2",
                kernels::GEMM_HFQ4G256_MOE_GROUPED_WMMA_K2_SRC,
            )
        };
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tile_stride = if use_m2 { 32 } else { 16 };
        let row_tiles = ((m + row_tile_stride - 1) / row_tile_stride) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: each tile loads one expert weight row band (m_total/16 tiles
        // share the same expert avg ~ m_total/E times) + gathers X + writes Y.
        let bytes =
            m_total * k * 2 + (m_total * m) * 4 + (crate::profile::gemv_hfq4g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Run the HFQ4/MQ4 grouped MoE GEMM through the FP16-WMMA route even on
    /// archs where the i8 MMQ shortcut is default-on. Used by mixed MQ6 A3B
    /// prefill, where gfx1151's HFQ4 i8 shortcut is model-level unsafe.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_wmma_k2_fp16(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let is_gfx12 = self.arch_caps.is_rdna4();
        let use_m2 = is_gfx12 && self.flags.moe_grouped_m2;
        let (kernel_name, kernel_src) = if use_m2 {
            (
                "gemm_hfq4g256_moe_grouped_wmma_m2_gfx12",
                kernels::GEMM_HFQ4G256_MOE_GROUPED_WMMA_M2_GFX12_SRC,
            )
        } else if is_gfx12 {
            (
                "gemm_hfq4g256_moe_grouped_wmma_gfx12",
                kernels::GEMM_HFQ4G256_MOE_GROUPED_WMMA_GFX12_SRC,
            )
        } else {
            (
                "gemm_hfq4g256_moe_grouped_wmma_k2",
                kernels::GEMM_HFQ4G256_MOE_GROUPED_WMMA_K2_SRC,
            )
        };
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tile_stride = if use_m2 { 32 } else { 16 };
        let row_tiles = ((m + row_tile_stride - 1) / row_tile_stride) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let bytes = m_total * k * 2 + (m_total * m) * 4 + crate::profile::gemv_hfq4g256_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// SGLang-style grouped-WMMA-GEMM for HFQ4G128 (ParoQuant) routed
    /// experts. Sister of `gemm_hfq4g256_moe_grouped_wmma_k2` with the
    /// 72 B/group HFQ4G128 stride. F32 x_src is auto-converted to F16
    /// via `ensure_fp16_x` (same convention as the G256 sister). Used
    /// by the Path 2 routed-expert dispatch in
    /// `prefill_moe_ffn_body_batched` on gfx11/gfx12 when ParoQ4G128
    /// experts are admitted (HIPFIRE_PARO_BATCHED=1). No i8 MMQ variant
    /// today — HFQ4G128 doesn't have a Q8_1 prequant pipeline; if needed
    /// later this would parallel `gemm_hfq4g256_moe_grouped_mmq_gfx1151`.
    ///
    /// `x_src_rows` is the number of rows in x_src (N for gate_up,
    /// N*K_TOP for down).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_paro_q4g128_moe_grouped_wmma_k2(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to FP16)
        y_grouped: &GpuTensor,          // [m_total × M] f32
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_paro_q4g128_moe_grouped_wmma_k2",
            kernels::GEMM_PARO_Q4G128_MOE_GROUPED_WMMA_K2_SRC,
            "gemm_paro_q4g128_moe_grouped_wmma_k2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let bytes =
            m_total * k * 2 + (m_total * m) * 4 + (crate::profile::gemv_hfq4g128_bytes(m, k));
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_paro_q4g128_moe_grouped_wmma_k2",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_paro_q4g128_moe_grouped_wmma_k2",
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// k8 deepest-pipeline sibling of `gemm_paro_q4g128_moe_grouped_mmq_gfx1151`.
    /// 8 WMMAs into 4 independent int32 accumulators per HFQ4G128 group.
    /// Same kernarg layout + grid as k2. Used via HIPFIRE_MOE_PARO_I8_K8=1.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_paro_q4g128_moe_grouped_mmq_k8_gfx1151(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let kernel_name = "gemm_paro_q4g128_moe_grouped_mmq_k8_gfx1151";
        let kernel_src = kernels::GEMM_PARO_Q4G128_MOE_GROUPED_MMQ_K8_GFX1151_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_src, x_src_rows, k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let xsr_val = x_src_rows as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &xsr_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g128_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b.push_i32(xsr_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx1151 i8 WMMA MMQ MoE grouped GEMM for HFQ4G128 (ParoQuant). Same
    /// scatter contract + per-sub-block scale-FMA convention as the
    /// HFQ4G256 sister `gemm_hfq4g256_moe_grouped_mmq_gfx1151`. Auto-
    /// quantizes F32 x_src to Q8_1 via `ensure_q8_1_mmq_x` (shared
    /// scratch). Compute-bound regime: ~140 TFLOPS i8 WMMA vs ~71 TFLOPS
    /// FP16 WMMA. gfx1151-only — the kernel guards on `__gfx1151__` and
    /// is a no-op stub on other archs.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_paro_q4g128_moe_grouped_mmq_gfx1151(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to Q8_1)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let kernel_name = "gemm_paro_q4g128_moe_grouped_mmq_gfx1151";
        let kernel_src = kernels::GEMM_PARO_Q4G128_MOE_GROUPED_MMQ_GFX1151_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_src, x_src_rows, k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let xsr_val = x_src_rows as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &xsr_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: Q8_1 X reads + HFQ4G128 weights + Y writes.
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g128_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b.push_i32(xsr_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx1151 (Strix Halo iGPU) i8 MMQ MoE grouped GEMM. Ports the i8
    /// WMMA MMQ pattern from `gemm_hfq4g256_residual_mmq` to the SGLang
    /// grouped scatter dispatch. X is pre-quantized to Q8_1 via
    /// `ensure_q8_1_mmq_x` (same buffer/scratch as the residual MMQ path).
    ///
    /// Kernarg layout matches the FP16 sister except the X pointer is the
    /// Q8_1 packed scratch (not the FP16 conversion buffer) and there is
    /// one extra `x_src_rows` arg (Q8_1 layout is `[K/128 × x_src_rows]`,
    /// so the kernel needs `x_src_rows` to compute the row stride).
    ///
    /// Used as a drop-in replacement for `gemm_hfq4g256_moe_grouped_wmma_k2`
    /// on gfx1151 when `HIPFIRE_MOE_GROUPED_I8 != "0"` (default ON for
    /// gfx1151). The FP16 sister still owns gfx12/gfx11-non-1151 paths.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_mmq_gfx1151(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to Q8_1)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let kernel_name = "gemm_hfq4g256_moe_grouped_mmq_gfx1151";
        let kernel_src = kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_GFX1151_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        // Q8_1 pre-pass (reuses the shared MMQ X scratch).
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_src, x_src_rows, k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let xsr_val = x_src_rows as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &xsr_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: Q8_1 X reads + HFQ4 weights + Y writes. Q8_1 = ~1B/elem
        // (slightly more for the per-sub-block (d,sum) metadata) vs FP16 = 2B/elem.
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b.push_i32(xsr_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx1151 i8 MMQ MoE grouped GEMM for MQ2-Lloyd weights. Drop-in for the
    /// FP16 `gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2` (same args + trailing
    /// `x_src_rows`): X is auto-quantized to Q8_1, the 2-bit Lloyd index is
    /// decoded via an in-kernel int8-codebook LUT, and i8 WMMA runs at ~2x the
    /// FP16 rate. Default path on gfx1151 (no env gate).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1151(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to Q8_1)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch != "gfx1151" {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1151 requires gfx1151; current arch = {}",
                    self.arch
                ),
            ));
        }
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1151";
        let kernel_src = kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_MMQ_GFX1151_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        // Q8_1 pre-pass (reuses the shared MMQ X scratch).
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_src, x_src_rows, k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let xsr_val = x_src_rows as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &xsr_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let bytes = (m_total * k) + (m_total * m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b.push_i32(xsr_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx1151 MQ2-Lloyd grouped MMQ candidate that replaces sixteen scalar
    /// codebook shifts per K tile with four byte-permute lookups.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq2g256_lloyd_moe_grouped_mmq_perm_gfx1151(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch != "gfx1151" {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_mq2g256_lloyd_moe_grouped_mmq_perm_gfx1151 \
                     requires gfx1151; current arch = {}",
                    self.arch
                ),
            ));
        }
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_mmq_perm_gfx1151";
        self.ensure_kernel(
            kernel_name,
            kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_MMQ_PERM_GFX1151_SRC,
            kernel_name,
        )?;
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_src, x_src_rows, k)?;
        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let xsr_val = x_src_rows as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &xsr_val as *const _ as *mut c_void,
        ];
        let row_tiles = m.div_ceil(16) as u32;
        let slot_tiles = m_total.div_ceil(16) as u32;
        let bytes = (m_total * k) + (m_total * m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b.push_i32(xsr_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Exact-gfx1030 MQ2-Lloyd grouped sdot4 probe. This method is deliberately
    /// absent from the production MoE selector, so A3B MQ2 admission and beta's
    /// Redline recording surface remain unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1030(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1030() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1030 requires exact gfx1030; current arch = {}",
                    self.arch
                ),
            ));
        }
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1030";
        self.ensure_kernel(
            kernel_name,
            kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_MMQ_GFX1030_SRC,
            kernel_name,
        )?;
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_src, x_src_rows, k)?;
        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let xsr_val = x_src_rows as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &xsr_val as *const _ as *mut c_void,
        ];
        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let bytes = (m_total * k) + (m_total * m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        // Probe-only raw HIP launch: never enter the Redline recording seam.
        let func = &self.functions[kernel_name];
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles, slot_tiles, 1],
                [32, 1, 1],
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

    /// gfx1151 i8 MMQ MoE grouped GEMM for MQ3-Lloyd weights. Drop-in for the
    /// FP16 `gemm_mq3g256_lloyd_moe_grouped_wmma` (same args + trailing
    /// `x_src_rows`): X is auto-quantized to Q8_1, the 3-bit Lloyd index is
    /// decoded via an in-kernel 8-entry int8-codebook LUT, and i8 WMMA runs at
    /// ~2x the FP16 rate. MQ3-Lloyd twin of `gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1151`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq3g256_lloyd_moe_grouped_mmq_gfx1151(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to Q8_1)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch != "gfx1151" {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_mq3g256_lloyd_moe_grouped_mmq_gfx1151 requires gfx1151; current arch = {}",
                    self.arch
                ),
            ));
        }
        let kernel_name = "gemm_mq3g256_lloyd_moe_grouped_mmq_gfx1151";
        let kernel_src = kernels::GEMM_MQ3G256_LLOYD_MOE_GROUPED_MMQ_GFX1151_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        // Q8_1 pre-pass (reuses the shared MMQ X scratch).
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_src, x_src_rows, k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let xsr_val = x_src_rows as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &xsr_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let bytes = (m_total * k) + (m_total * m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b.push_i32(xsr_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Arch-resolved MQ2-Lloyd grouped MoE GEMM (the dispatcher entry — callers
    /// stay arch-agnostic). Picks the i8 MMQ kernel on gfx1151 (~1.7x FP16) when
    /// the shape qualifies (M%16, K%256), else the FP16 WMMA 4w_k2 kernel.
    /// Same args as both: trailing `x_src_rows` is the Q8_1/F16 source row count.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq2g256_lloyd_moe_grouped(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        if self.arch_caps.is_gfx942() && m % 16 == 0 && k % 256 == 0 {
            self.gemm_mq2g256_lloyd_moe_grouped_mfma_gfx942(
                expert_weight_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x_src,
                y_grouped,
                m,
                k,
                x_row_div,
                m_total,
                x_src_rows,
            )
        } else if self.arch_caps.is_gfx1151() && m % 16 == 0 && k % 256 == 0 {
            self.gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1151(
                expert_weight_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x_src,
                y_grouped,
                m,
                k,
                x_row_div,
                m_total,
                x_src_rows,
            )
        } else {
            self.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2(
                expert_weight_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x_src,
                y_grouped,
                m,
                k,
                x_row_div,
                m_total,
                x_src_rows,
            )
        }
    }

    /// gfx942/CDNA3 grouped MQ2-Lloyd MoE prefill using native wave64 MFMA.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq2g256_lloyd_moe_grouped_mfma_gfx942(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx942() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_mq2g256_lloyd_moe_grouped_mfma_gfx942 requires gfx942; current arch = {}",
                    self.arch
                ),
            ));
        }
        debug_assert_eq!(m % 16, 0);
        debug_assert_eq!(k % 256, 0);
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_mfma_gfx942";
        self.ensure_kernel(
            kernel_name,
            kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_MFMA_GFX942_SRC,
            kernel_name,
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = m_total * k * 2 + m_total * m * 4 + mq2_weight_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Arch-resolved MQ3-Lloyd grouped MoE GEMM (dispatcher entry). Picks the i8
    /// MMQ kernel on gfx1151 (M%16, K%256) else the FP16 WMMA dispatcher
    /// (`gemm_mq3g256_lloyd_moe_grouped_wmma`, which itself selects gfx12 vs k2).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq3g256_lloyd_moe_grouped(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        if self.arch_caps.is_gfx1151() && m % 16 == 0 && k % 256 == 0 {
            self.gemm_mq3g256_lloyd_moe_grouped_mmq_gfx1151(
                expert_weight_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x_src,
                y_grouped,
                m,
                k,
                x_row_div,
                m_total,
                x_src_rows,
            )
        } else {
            self.gemm_mq3g256_lloyd_moe_grouped_wmma(
                expert_weight_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x_src,
                y_grouped,
                m,
                k,
                x_row_div,
                m_total,
                x_src_rows,
            )
        }
    }

    /// gfx1151 (Strix Halo iGPU) i8 MMQ MoE grouped GEMM — k4 (deeper
    /// K-tile pipeline) variant. Drop-in for `gemm_hfq4g256_moe_grouped_mmq_gfx1151`
    /// — same kernarg layout, same grid/block geometry, same scatter
    /// contract. The kernel pairs adjacent Q8_1 sub-blocks so each inner
    /// iteration issues 4 WMMAs into 2 independent int32 accumulators
    /// before the per-sub-block scale FMA resolves. Output is
    /// numerically equivalent to k2 modulo int32 summation-order
    /// (commutative; integer-addition reductions are exact).
    ///
    /// Opt-IN via `HIPFIRE_MOE_GROUPED_I8_K4=1` (default OFF). Routes
    /// through the same wrapper as k2 (`gemm_hfq4g256_moe_grouped_wmma_k2`),
    /// which gates on `HIPFIRE_MOE_GROUPED_I8 != "0"` first.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_mmq_k4_gfx1151(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to Q8_1)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let kernel_name = "gemm_hfq4g256_moe_grouped_mmq_k4_gfx1151";
        let kernel_src = kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_K4_GFX1151_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        // Q8_1 pre-pass (reuses the shared MMQ X scratch).
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_src, x_src_rows, k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let xsr_val = x_src_rows as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &xsr_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: same as the k2 sibling — Q8_1 X reads + HFQ4 weights
        // + Y writes. k4 is a pure unroll-depth change, no extra memory traffic.
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b.push_i32(xsr_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx1151 (Strix Halo iGPU) i8 MMQ MoE grouped GEMM — k8 (deepest
    /// K-tile pipeline) variant. Drop-in for `gemm_hfq4g256_moe_grouped_mmq_gfx1151`
    /// — same kernarg layout, same grid/block geometry, same scatter
    /// contract. The kernel processes all 4 sub-blocks of one Q8_1 block
    /// per inner iteration — 8 WMMAs into 4 independent int32 accumulators
    /// before the per-sub-block scale FMA resolves. Output is numerically
    /// equivalent to k2/k4 modulo int32 summation-order (commutative;
    /// integer-addition reductions are exact).
    ///
    /// Opt-IN via `HIPFIRE_MOE_GROUPED_I8_K8=1` (default OFF). Routes
    /// through the same wrapper as k2/k4 (`gemm_hfq4g256_moe_grouped_wmma_k2`),
    /// which gates on `HIPFIRE_MOE_GROUPED_I8 != "0"` first; k8 takes
    /// priority over k4 if both env vars are set.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_mmq_k8_gfx1151(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to Q8_1)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let kernel_name = "gemm_hfq4g256_moe_grouped_mmq_k8_gfx1151";
        let kernel_src = kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_K8_GFX1151_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        // Q8_1 pre-pass (reuses the shared MMQ X scratch).
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_src, x_src_rows, k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let xsr_val = x_src_rows as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &xsr_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: same as the k2/k4 siblings — Q8_1 X reads + HFQ4 weights
        // + Y writes. k8 is a pure unroll-depth change, no extra memory traffic.
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b.push_i32(xsr_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// k4 (deeper K-tile pipeline) variant for gfx11 dGPUs. Mirrors the
    /// gfx1151 k4 design (validated +4.6% over k2 there with zero spills).
    /// Pairs adjacent Q8_1 sub-blocks for 4 WMMAs into 2 independent int32
    /// accumulators per inner iteration; numerically equivalent to k2
    /// modulo int32 summation order. Opt-IN via
    /// `HIPFIRE_MOE_GROUPED_I8_K4=1` (default OFF on dGPU — k2 was no-op
    /// vs FP16, so k4 is the real test of the pipeline-depth hypothesis).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_mmq_k4_gfx11_dgpu(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let kernel_name = "gemm_hfq4g256_moe_grouped_mmq_k4_gfx11_dgpu";
        let kernel_src = kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_K4_GFX11_DGPU_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_src, x_src_rows, k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let xsr_val = x_src_rows as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &xsr_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b.push_i32(xsr_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx11 dGPU i8 MMQ MoE grouped GEMM (gfx1100/1101/1102/1103 — 7900 XTX,
    /// 7800/7700, 7600, Phoenix mobile). Same kernarg layout as the gfx1151
    /// i8 sister (10-arg variant with explicit `x_src_rows` for the Q8_1
    /// K-block stride). X pre-quantized to Q8_1 via `ensure_q8_1_mmq_x`.
    ///
    /// Used as a drop-in replacement for `gemm_hfq4g256_moe_grouped_wmma_k2`
    /// on gfx11 dGPUs when `HIPFIRE_MOE_GROUPED_I8 != "0"` (default ON for
    /// gfx1100/1101/1102/1103). Roughly 2× the FLOP rate of the FP16 sister
    /// on this compute-bound grouped MoE GEMM path.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_mmq_gfx11_dgpu(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to Q8_1)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let kernel_name = "gemm_hfq4g256_moe_grouped_mmq_gfx11_dgpu";
        let kernel_src = kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_GFX11_DGPU_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        // Q8_1 pre-pass (reuses the shared MMQ X scratch).
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_src, x_src_rows, k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let xsr_val = x_src_rows as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &xsr_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: Q8_1 X reads (~1 B/elem incl. (d,sum) metadata) +
        // HFQ4 weights + Y writes. Distinct from the FP16 sister (which
        // uses 2 B/elem for X).
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b.push_i32(xsr_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx12 (RDNA4 — R9700/gfx1201, gfx1200) i8 MMQ MoE grouped GEMM. Ports
    /// the i8 WMMA MMQ pattern to the SGLang grouped scatter dispatch using
    /// the gfx12-specific WMMA intrinsic (`wmma_i32_16x16x16_iu8_w32_gfx12`)
    /// at 2× the FLOP rate of FP16 WMMA on gfx12. X is pre-quantized to Q8_1
    /// via `ensure_q8_1_mmq_x` (same scratch buffer as the residual MMQ path).
    ///
    /// Kernarg layout matches the gfx1151 sister: FP16 args + `x_src_rows`
    /// extra arg for the Q8_1 layout stride (`[K/128 × x_src_rows]`).
    ///
    /// Used as a drop-in replacement for `gemm_hfq4g256_moe_grouped_wmma_k2`
    /// on gfx12 when `HIPFIRE_MOE_GROUPED_I8 != "0"` (default ON for gfx12).
    /// The FP16 sister still owns the env=0 opt-out path.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_mmq_gfx12(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to Q8_1)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let kernel_name = "gemm_hfq4g256_moe_grouped_mmq_gfx12";
        let kernel_src = kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_GFX12_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        // Q8_1 pre-pass (reuses the shared MMQ X scratch).
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_src, x_src_rows, k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let xsr_val = x_src_rows as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &xsr_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: Q8_1 X reads (~1B/elem + ds4 metadata) + HFQ4 weights + Y writes.
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b.push_i32(xsr_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx12 (RDNA4 — R9700/gfx1201, gfx1200) i8 MMQ MoE grouped GEMM —
    /// k4 (deeper K-tile pipeline) variant. Drop-in for
    /// `gemm_hfq4g256_moe_grouped_mmq_gfx12` — same kernarg layout, same
    /// grid/block geometry, same scatter contract. Processes all 4
    /// sub-blocks of one Q8_1 block per inner iteration (8 WMMAs into 4
    /// independent int32 accumulators) before the per-sub-block scale FMA
    /// chain resolves. Numerically equivalent to k2 modulo floating-point
    /// summation order on the scale FMA chain.
    ///
    /// Opt-IN via `HIPFIRE_MOE_GROUPED_I8=1 HIPFIRE_MOE_GROUPED_I8_K4_GFX12=1`
    /// (both default OFF). Routes through the same wrapper as k2
    /// (`gemm_hfq4g256_moe_grouped_wmma_k2`), which gates on
    /// `HIPFIRE_MOE_GROUPED_I8 == "1"` for gfx12 first.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq4g256_moe_grouped_mmq_k4_gfx12(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to Q8_1)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let kernel_name = "gemm_hfq4g256_moe_grouped_mmq_k4_gfx12";
        let kernel_src = kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_K4_GFX12_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        // Q8_1 pre-pass (reuses the shared MMQ X scratch).
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_src, x_src_rows, k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let xsr_val = x_src_rows as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &xsr_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: same as the k2 sibling. k4 is a pure unroll-depth
        // change, no extra memory traffic.
        let bytes = (m_total * k) + (m_total * m) * 4 + (crate::profile::gemv_hfq4g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b.push_i32(xsr_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx1151 i8-WMMA MMQ grouped GEMM for HFQ6/MQ6 MoE experts. Same
    /// scatter contract as the FP16-WMMA sister, but prequantizes X to Q8_1
    /// and applies the HFQ6 correction term after each 32-K sub-block.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq6g256_moe_grouped_mmq_gfx1151(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let kernel_name = "gemm_hfq6g256_moe_grouped_mmq_gfx1151";
        self.ensure_kernel(
            kernel_name,
            kernels::GEMM_HFQ6G256_MOE_GROUPED_MMQ_GFX1151_SRC,
            kernel_name,
        )?;
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x_src, x_src_rows, k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;
        let xsr_val = x_src_rows as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &xsr_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let bytes = (m_total * k) + (m_total * m) * 4 + crate::profile::gemv_hfq6g256_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b.push_i32(xsr_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ6/MQ6 sister of `gemm_hfq4g256_moe_grouped_wmma_k2`. Same kernarg
    /// layout + grouped dispatch contract; differs only in the 200 B/group
    /// HFQ6 dequant inner loop. Unblocks AWQ A3B prefill (where ~50% of
    /// experts are MQ6 not MQ4 in the production AWQ A3B build at
    /// /mnt/nas/kaden/hipfire/mi300x-v3/qwen3-35b-a3b.mq4-awq).
    ///
    /// `x_row_div` selects the X gather layout (identical to the HFQ4 sister):
    ///   gate_up: x_src = x_rot_batch [N × K], x_row_div = K_TOP
    ///   down:    x_src = rot_batch [N*K_TOP × K], x_row_div = 1
    /// `x_src_rows` is the number of rows in x_src (N or N*K_TOP).
    ///
    /// gfx11 (RDNA3/3.5) and gfx12 (RDNA4). Dispatches the channel-tested
    /// `_gfx1151` WMMA kernel on gfx1151 (with an optional i8/MMQ fast-path
    /// under `HIPFIRE_MOE_HFQ6_I8`), the generic `_k2` kernel on other gfx11
    /// (RDNA3/3.5) archs, and the `_gfx12` (or `_v2_gfx12`) variant on gfx12.
    /// Returns an error on archs without WMMA support.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq6g256_moe_grouped_wmma(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to FP16)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let is_gfx12 = self.arch_caps.is_rdna4();
        if !is_gfx12 && !self.arch_caps.has_wmma_w32() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_hfq6g256_moe_grouped_wmma: requires WMMA (gfx11 or gfx12); \
                     current arch = {}",
                    self.arch
                ),
            ));
        }
        // gfx1151 i8/MMQ fast-path (master dfed8cc6): opt-in via
        // HIPFIRE_MOE_HFQ6_I8. Routes the Strix-Halo grouped MQ6 GEMM through
        // the channel-tested i8 MMQ kernel before WMMA kernel selection.
        if self.arch_caps.is_gfx1151() && self.flags.moe_hfq6_i8 {
            return self.gemm_hfq6g256_moe_grouped_mmq_gfx1151(
                expert_weight_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                x_src,
                y_grouped,
                m,
                k,
                x_row_div,
                m_total,
                x_src_rows,
            );
        }
        // v2 lever (gfx12 only): M-direction 2×1 reg-block, env-gated. Defaults
        // off; promotes when `HIPFIRE_MOE_HFQ6_V2=1`. Each warp covers 32 rows
        // × 16 slots (vs 16×16); B-load halved per output.
        let use_v2 = is_gfx12 && self.flags.moe_hfq6_v2;
        // Kernel selection (union of the 458 gfx11-widen + master gfx1151
        // channel-tested kernel): gfx1151 → `_gfx1151`; gfx12 v2 → `_v2_gfx12`;
        // gfx12 → `_gfx12`; other gfx11 (RDNA3/3.5 dGPU) → `_k2`.
        let (kernel_name, kernel_src, row_tile_stride) = if self.arch_caps.is_gfx1151() {
            (
                "gemm_hfq6g256_moe_grouped_wmma_gfx1151",
                kernels::GEMM_HFQ6G256_MOE_GROUPED_WMMA_GFX1151_SRC,
                16usize,
            )
        } else if use_v2 {
            (
                "gemm_hfq6g256_moe_grouped_wmma_v2_gfx12",
                kernels::GEMM_HFQ6G256_MOE_GROUPED_WMMA_V2_GFX12_SRC,
                32usize,
            )
        } else if is_gfx12 {
            (
                "gemm_hfq6g256_moe_grouped_wmma_gfx12",
                kernels::GEMM_HFQ6G256_MOE_GROUPED_WMMA_GFX12_SRC,
                16usize,
            )
        } else {
            (
                "gemm_hfq6g256_moe_grouped_wmma_k2",
                kernels::GEMM_HFQ6G256_MOE_GROUPED_WMMA_K2_SRC,
                16usize,
            )
        };
        let slot_tile_stride = 16usize;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + row_tile_stride - 1) / row_tile_stride) as u32;
        let slot_tiles = ((m_total + slot_tile_stride - 1) / slot_tile_stride) as u32;
        // BW estimate uses the HFQ6 weight footprint (200 B/group vs HFQ4's 136 B).
        let bytes =
            m_total * k * 2 + (m_total * m) * 4 + (crate::profile::gemv_hfq6g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx1151 mfp4-E8 grouped-WMMA-GEMM (Path 2 prefill) — the FP16-WMMA sister
    /// of the indexed E8 GEMV. Same scatter-pipeline contract as
    /// `gemm_hfq4g256_moe_grouped_wmma_k2`: each tile picks its expert via
    /// `expert_tile_ids[tile_y]` (-1 sentinel = early return) and gathers its
    /// B-operand rows via `sorted_slot_index` (-1 padding lanes = zeros), writing
    /// `Y_grouped[m_total × M]` direct. The A operand is dequantized from the E8
    /// lattice (bit-identical decode to `gemv_mfp4g32_e8`); X is auto-converted
    /// f32→f16 via `ensure_fp16_x`. This is the only E8 path that amortizes
    /// expert-weight reads — the memory-bound prefill / batched-verify lever.
    /// gfx1151-only (wave32 `wmma_f32_16x16x16_f16_w32`).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mfp4g32_e8_moe_grouped_wmma(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to FP16)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert!(
            self.arch_caps.has_wmma_w32() || self.arch_caps.is_rdna4(),
            "gemm_mfp4g32_e8_moe_grouped_wmma needs RDNA3 wave32-WMMA or RDNA4 (current arch = {})",
            self.arch
        );
        // gfx12 (RDNA4) uses the _gfx12 WMMA intrinsic sister; gfx1151 (and the
        // rest of RDNA3) use the original gfx1151 kernel — both carry the same
        // kernarg layout and grid formula so nothing else changes.
        let is_gfx12 = self.arch_caps.is_rdna4();
        let (kernel_name, kernel_src) = if is_gfx12 {
            (
                "gemm_mfp4g32_e8_moe_grouped_wmma_gfx12",
                kernels::GEMM_MFP4G32_E8_MOE_GROUPED_WMMA_GFX12_SRC,
            )
        } else {
            (
                "gemm_mfp4g32_e8_moe_grouped_wmma",
                kernels::GEMM_MFP4G32_E8_MOE_GROUPED_WMMA_GFX1151_SRC,
            )
        };
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let bytes = m_total * k * 2 + (m_total * m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Dense E8-SoA WMMA GEMM for gfx1151 prefill.
    ///
    /// Decodes each 16-row weight tile once per 16-token tile instead of
    /// replaying the decode GEMV for every prompt token. X is converted to
    /// f16 through the existing pointer-keyed scratch cache; Y remains f32.
    pub fn gemm_mfp4g32_e8_soa_wmma(
        &mut self,
        weight: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert!(
            self.arch_caps.is_gfx1151(),
            "dense E8-SoA WMMA is gfx1151-only"
        );
        assert!(k % 256 == 0, "dense E8-SoA WMMA requires K%256==0");
        const KERNEL: &str = "gemm_mfp4g32_e8_soa_wmma_gfx1151";
        self.ensure_kernel(
            KERNEL,
            kernels::GEMM_MFP4G32_E8_SOA_WMMA_GFX1151_SRC,
            KERNEL,
        )?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * k)?;
        let ap = weight.buf.as_ptr();
        let xp = x_f16;
        let yp = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let b_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &b_val as *const _ as *mut c_void,
        ];
        let row_tiles = m.div_ceil(16) as u32;
        let batch_tiles = batch_size.div_ceil(16) as u32;
        let bytes = weight.byte_size() + batch_size * (k * 2 + m * 4);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", KERNEL, bytes);
        let result = self.launch_maybe_blob(
            KERNEL,
            [row_tiles, batch_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(ap);
                blob.push_ptr(xp);
                blob.push_ptr(yp);
                blob.push_i32(m_val);
                blob.push_i32(k_val);
                blob.push_i32(b_val);
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Two-token-tile variant of [`Self::gemm_mfp4g32_e8_soa_wmma`].
    ///
    /// Each decoded weight fragment feeds two independent 16-token WMMA
    /// accumulators, amortizing E8 lattice decode across 32 prompt tokens.
    pub fn gemm_mfp4g32_e8_soa_wmma_b2(
        &mut self,
        weight: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert!(
            self.arch_caps.is_gfx1151(),
            "dense E8-SoA WMMA B2 is gfx1151-only"
        );
        assert!(k % 256 == 0, "dense E8-SoA WMMA B2 requires K%256==0");
        const KERNEL: &str = "gemm_mfp4g32_e8_soa_wmma_b2_gfx1151";
        self.ensure_kernel(
            KERNEL,
            kernels::GEMM_MFP4G32_E8_SOA_WMMA_B2_GFX1151_SRC,
            KERNEL,
        )?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * k)?;
        let ap = weight.buf.as_ptr();
        let xp = x_f16;
        let yp = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let b_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &b_val as *const _ as *mut c_void,
        ];
        let row_tiles = m.div_ceil(16) as u32;
        let batch_tiles = batch_size.div_ceil(32) as u32;
        let bytes = weight.byte_size() + batch_size * (k * 2 + m * 4);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", KERNEL, bytes);
        let result = self.launch_maybe_blob(
            KERNEL,
            [row_tiles, batch_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(ap);
                blob.push_ptr(xp);
                blob.push_ptr(yp);
                blob.push_i32(m_val);
                blob.push_i32(k_val);
                blob.push_i32(b_val);
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Four-token-tile E8-SoA WMMA candidate for gfx1151 prefill.
    pub fn gemm_mfp4g32_e8_soa_wmma_b4(
        &mut self,
        weight: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert!(
            self.arch_caps.is_gfx1151(),
            "dense E8-SoA WMMA B4 is gfx1151-only"
        );
        assert!(k % 256 == 0, "dense E8-SoA WMMA B4 requires K%256==0");
        const KERNEL: &str = "gemm_mfp4g32_e8_soa_wmma_b4_gfx1151";
        self.ensure_kernel(
            KERNEL,
            kernels::GEMM_MFP4G32_E8_SOA_WMMA_B4_GFX1151_SRC,
            KERNEL,
        )?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * k)?;
        let ap = weight.buf.as_ptr();
        let xp = x_f16;
        let yp = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let b_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &b_val as *const _ as *mut c_void,
        ];
        let row_tiles = m.div_ceil(16) as u32;
        let batch_tiles = batch_size.div_ceil(64) as u32;
        let bytes = weight.byte_size() + batch_size * (k * 2 + m * 4);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", KERNEL, bytes);
        let result = self.launch_maybe_blob(
            KERNEL,
            [row_tiles, batch_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(ap);
                blob.push_ptr(xp);
                blob.push_ptr(yp);
                blob.push_i32(m_val);
                blob.push_i32(k_val);
                blob.push_i32(b_val);
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Grouped block-diagonal E8-SoA WMMA GEMM for gfx1151 prefill.
    ///
    /// A is `[groups, m, k]`, X is `[batch, groups, k]`, and Y is
    /// `[batch, groups, m]`. One launch replaces the former
    /// `batch * groups` sequence of single-token GEMVs.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mfp4g32_e8_soa_grouped_wmma(
        &mut self,
        weight: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        groups: usize,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert!(
            self.arch_caps.is_gfx1151(),
            "grouped E8-SoA WMMA is gfx1151-only"
        );
        assert!(k % 256 == 0, "grouped E8-SoA WMMA requires K%256==0");
        const KERNEL: &str = "gemm_mfp4g32_e8_soa_grouped_wmma_gfx1151";
        self.ensure_kernel(
            KERNEL,
            kernels::GEMM_MFP4G32_E8_SOA_GROUPED_WMMA_GFX1151_SRC,
            KERNEL,
        )?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * groups * k)?;
        let ap = weight.buf.as_ptr();
        let xp = x_f16;
        let yp = y.buf.as_ptr();
        let g_val = groups as i32;
        let m_val = m as i32;
        let k_val = k as i32;
        let b_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &g_val as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &b_val as *const _ as *mut c_void,
        ];
        let row_tiles = m.div_ceil(16) as u32;
        let batch_tiles = batch_size.div_ceil(16) as u32;
        let bytes = weight.byte_size() + batch_size * groups * (k * 2 + m * 4);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", KERNEL, bytes);
        let result = self.launch_maybe_blob(
            KERNEL,
            [row_tiles, batch_tiles, groups as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(ap);
                blob.push_ptr(xp);
                blob.push_ptr(yp);
                blob.push_i32(g_val);
                blob.push_i32(m_val);
                blob.push_i32(k_val);
                blob.push_i32(b_val);
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Two-token-tile variant of the grouped block-diagonal E8-SoA GEMM.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mfp4g32_e8_soa_grouped_wmma_b2(
        &mut self,
        weight: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        groups: usize,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert!(
            self.arch_caps.is_gfx1151(),
            "grouped E8-SoA WMMA B2 is gfx1151-only"
        );
        assert!(k % 256 == 0, "grouped E8-SoA WMMA B2 requires K%256==0");
        const KERNEL: &str = "gemm_mfp4g32_e8_soa_grouped_wmma_b2_gfx1151";
        self.ensure_kernel(
            KERNEL,
            kernels::GEMM_MFP4G32_E8_SOA_GROUPED_WMMA_B2_GFX1151_SRC,
            KERNEL,
        )?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * groups * k)?;
        let ap = weight.buf.as_ptr();
        let xp = x_f16;
        let yp = y.buf.as_ptr();
        let g_val = groups as i32;
        let m_val = m as i32;
        let k_val = k as i32;
        let b_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &g_val as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &b_val as *const _ as *mut c_void,
        ];
        let row_tiles = m.div_ceil(16) as u32;
        let batch_tiles = batch_size.div_ceil(32) as u32;
        let bytes = weight.byte_size() + batch_size * groups * (k * 2 + m * 4);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", KERNEL, bytes);
        let result = self.launch_maybe_blob(
            KERNEL,
            [row_tiles, batch_tiles, groups as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(ap);
                blob.push_ptr(xp);
                blob.push_ptr(yp);
                blob.push_i32(g_val);
                blob.push_i32(m_val);
                blob.push_i32(k_val);
                blob.push_i32(b_val);
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ3/MQ3 sister of `gemm_hfq4g256_moe_grouped_wmma_k2` for the
    /// MoE Path-2 grouped-WMMA-GEMM. Same contract: each WMMA tile picks
    /// its expert via `expert_tile_ids[tile_y]` (-1 sentinel = early
    /// return) and gathers its B-operand rows via `sorted_slot_index`
    /// (-1 padding lanes contribute zeros). Writes `Y_grouped[m_total ×
    /// M]` direct.
    ///
    /// `x_row_div` selects the X gather layout:
    ///   gate_up: x_src = x_rot_batch [N × K], x_row_div = K_TOP
    ///   down:    x_src = rot_batch [N*K_TOP × K], x_row_div = 1
    /// `x_src_rows` is the number of rows in x_src (N or N*K_TOP).
    ///
    /// **gfx12 (RDNA4) only** for now. Other archs panic; integration
    /// with `is_batchable_la` is gated on arch=gfx12.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfq3g256_moe_grouped_wmma(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to FP16)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_rdna4() {
            panic!(
                "gemm_hfq3g256_moe_grouped_wmma: only gfx12 (RDNA4) is supported; \
                 caller must gate on arch.starts_with(\"gfx12\"). Arch: {}",
                self.arch
            );
        }
        let kernel_name = "gemm_hfq3g256_moe_grouped_wmma_gfx12";
        let kernel_src = kernels::GEMM_HFQ3G256_MOE_GROUPED_WMMA_GFX12_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: HFQ3 row footprint is groups_per_row × 104 B (vs
        // HFQ4's 136 B); reuse the gemv_hfq3g256_bytes profile helper.
        let bytes =
            m_total * k * 2 + (m_total * m) * 4 + (crate::profile::gemv_hfq3g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// MQ3-Lloyd sister of `gemm_hfq4g256_moe_grouped_wmma_k2` for MoE Path-2.
    /// Same scatter contract + 9-tuple kernarg. gfx11 dispatches the `_k2`
    /// (half16) kernel; gfx12 (RDNA4) the `.gfx12` (half8/k_grp) kernel.
    /// 112 B/group = 16 B (8 × fp16 codebook) + 96 B (256 × 3-bit indices).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq3g256_lloyd_moe_grouped_wmma(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to FP16)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let is_gfx12 = self.arch_caps.is_rdna4();
        if !is_gfx12 && !self.arch_caps.has_wmma_w32() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_mq3g256_lloyd_moe_grouped_wmma: requires WMMA (gfx11 or gfx12); \
                     current arch = {}",
                    self.arch
                ),
            ));
        }
        let (kernel_name, kernel_src) = if is_gfx12 {
            (
                "gemm_mq3g256_lloyd_moe_grouped_wmma_gfx12",
                kernels::GEMM_MQ3G256_LLOYD_MOE_GROUPED_WMMA_GFX12_SRC,
            )
        } else {
            (
                "gemm_mq3g256_lloyd_moe_grouped_wmma_k2",
                kernels::GEMM_MQ3G256_LLOYD_MOE_GROUPED_WMMA_K2_SRC,
            )
        };
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: MQ3-Lloyd row footprint = groups_per_row × 112 B.
        let bytes =
            m_total * k * 2 + (m_total * m) * 4 + (crate::profile::gemv_mq3g256_lloyd_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Merged dtype-tag MoE grouped-WMMA prefill kernel.
    ///
    /// Routes each workgroup to the correct per-expert dequant body via a
    /// block-uniform `dtype_tags[expert_id]` branch (zero warp divergence):
    ///   tag 0 = MQ6 (200 B/group), tag 1 = MQ2-Lloyd (72 B/group),
    ///   tag 2 = MQ4 (136 B/group),  tag 3 = MQ3-Lloyd (112 B/group).
    ///
    /// Kernarg layout differs from the uniform grouped launchers by inserting
    /// `dtype_tags` as the 2nd argument (after `expert_weight_ptrs`):
    ///   expert_weight_ptrs, dtype_tags, expert_tile_ids, sorted_slot_index,
    ///   X_src, Y_grouped, M, K, x_row_div, m_total.
    ///
    /// `x_row_div = k_top` for gate_up, `1` for down — the kernel uses this
    /// to construct the correct input row index, same as the uniform launchers.
    ///
    /// Arch dispatch: gfx1200/1201 (RDNA4) → `_gfx12` kernel (half8_t /
    /// K4-unroll / _w32_gfx12 WMMA); gfx1100/1101/1102/1150/1151 (RDNA3/3.5)
    /// → `_k2` kernel (half16_t / K2-unroll / _w32 WMMA). Requires WMMA on
    /// the current arch — returns an error for non-WMMA archs (gfx9*, gfx10*).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mixed_moe_grouped_wmma(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_dtype_tags: &GpuTensor,  // [E] u8 (DType::Raw) — tag 0/1/2/3
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to FP16)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let is_gfx12 = self.arch_caps.is_rdna4();
        if !is_gfx12 && !self.arch_caps.has_wmma_w32() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_mixed_moe_grouped_wmma: requires WMMA (gfx11 or gfx12); \
                     current arch = {}",
                    self.arch
                ),
            ));
        }
        // 4-warp LDS-tiled variant (gfx11/RDNA3 only). Env-gated; default OFF.
        let use_4w = !is_gfx12 && self.flags.moe_grouped_4w;
        let (kernel_name, kernel_src) = if is_gfx12 {
            (
                "gemm_mixed_moe_grouped_wmma_gfx12",
                kernels::GEMM_MIXED_MOE_GROUPED_WMMA_GFX12_SRC,
            )
        } else if use_4w {
            (
                "gemm_mixed_moe_grouped_wmma_4w_k2",
                kernels::GEMM_MIXED_MOE_GROUPED_WMMA_4W_K2_SRC,
            )
        } else {
            (
                "gemm_mixed_moe_grouped_wmma_k2",
                kernels::GEMM_MIXED_MOE_GROUPED_WMMA_K2_SRC,
            )
        };
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let dtp = expert_dtype_tags.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &dtp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        // 4w computes 64 rows/block (4 warps × 16) in a 128-thread block;
        // single-warp computes 16 rows/block in a 32-thread block.
        let row_tile_stride = if use_4w { 64 } else { 16 };
        let block_threads: u32 = if use_4w { 128 } else { 32 };
        let row_tiles = ((m + row_tile_stride - 1) / row_tile_stride) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: mixed row footprint — use MQ4 as a conservative middle estimate.
        let bytes =
            m_total * k * 2 + (m_total * m) * 4 + (crate::profile::gemv_hfq4g256_bytes(m, k));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [block_threads, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(dtp);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched HFQ4-G256 GEMM with fused residual add:
    ///   for b in 0..batch_size: y[b][row] += A[row] · x[b]
    ///
    /// Bitwise-identical output to calling `gemv_hfq4g256_residual` N times
    /// (preserves the 4-accumulator interleave and pairwise final combine),
    /// so safe to use in the quality-gated forward path. Each block handles
    /// one row × up to BATCH_TILE batch elements, amortizing the weight
    /// fetch across the batch loop.
    ///
    /// `x`: [batch_size × K] row-major, `y`: [batch_size × M] row-major.
    /// `y` must already hold the residual summand to accumulate into.
    pub fn gemm_hfq4g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx94x MFMA-direct opt-in: skips FP16 shadow + rocBLAS launch.
        // Opt-in via HIPFIRE_GFX942_MFMA_PREFILL=1 while validating; this
        // fires BEFORE the rocBLAS branch on purpose (rocBLAS goes through
        // FP16 dequant shadow, which is the cost we want to avoid).
        {
            let mfma_v = self.flags.gfx942_mfma_prefill.clone();
            let want = mfma_v.as_deref();
            if matches!(want, Some("1" | "2" | "3" | "4"))
                && self.arch_caps.is_cdna3()
                && batch_size >= 16
                && m % 16 == 0
                && k % 256 == 0
                && !self.graphs.capture_mode
            {
                if want == Some("4") && batch_size % 64 == 0 && m % 16 == 0 {
                    return self
                        .gemm_hfq4g256_residual_mfma_v4_gfx942(a_raw, x, y, m, k, batch_size);
                }
                if want == Some("3") && batch_size % 32 == 0 && m % 32 == 0 {
                    return self
                        .gemm_hfq4g256_residual_mfma_v3_gfx942(a_raw, x, y, m, k, batch_size);
                }
                if want == Some("2") && batch_size % 32 == 0 && m % 32 == 0 {
                    return self
                        .gemm_hfq4g256_residual_mfma_v2_gfx942(a_raw, x, y, m, k, batch_size);
                }
                return self.gemm_hfq4g256_residual_mfma_gfx942(a_raw, x, y, m, k, batch_size);
            }
        }
        // CDNA3 MFMA path — Y += X·W^T via rocBLAS with beta=1.
        if self.rocblas_arch_eligible()
            && batch_size >= self.rocblas_min_batch()
            && self.rocblas.is_some()
            && !self.graphs.capture_mode
        {
            if let Ok(Some(shadow_ptr)) = self.ensure_fp16_shadow(a_raw, m, k) {
                let x_fp16 = self.ensure_fp16_x(x, batch_size * k)?;
                let w_buf = unsafe { DeviceBuffer::from_raw(shadow_ptr, (m * k) * 2) };
                let x_buf = unsafe { DeviceBuffer::from_raw(x_fp16, (batch_size * k) * 2) };
                let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
                    + batch_size * k * 4
                    + batch_size * m * 4 * 2;
                let timer = crate::profile::begin_timer(
                    &self.hip,
                    "gemm",
                    "gemm_hfq4g256_residual_rocblas",
                    bytes,
                );
                let result = self
                    .rocblas_gemm_hfq4_prefill_residual(&w_buf, &x_buf, &y.buf, m, batch_size, k);
                std::mem::forget(w_buf);
                std::mem::forget(x_buf);
                if let Some(t) = timer {
                    t.finish(&self.hip);
                }
                return result;
            }
        }

        // HFQ4 wave32 MMQ residual on RDNA2+. Default-on for the allowlist
        // arch set (issue #300 gate removal — +210% prefill on gfx1031 4B
        // MQ4 pp128, KLD-neutral; escape hatch HIPFIRE_HFQ4_MMQ_RDNA2=0).
        // HFQ4's cheaper 4-bit nibble unpack lets MMQ beat the fp16
        // fallback. Env gate is OnceLock-cached.
        //
        // Issue #299 follow-up: route through the tile-size auto-selector
        // so narrow-batch calls pick mmq_x=16 and long-prefill picks
        // mmq_x=32_y64 (MQ3 phase-2 finding). All variants clamp M-tail
        // internally, so no alignment check needed.
        if batch_size > 1 && self.arch_caps.has_hfq4_mmq() {
            return self.gemm_hfq4g256_residual_mmq_rdna2_auto(a_raw, x, y, m, k, batch_size);
        }

        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !self.flags.fp16_disabled {
            // gfx906 dp4a MMQ residual path — default-on at batch ≥ 8 per
            // should_use_mmq's gfx906 default. Distinguishes two reasons
            // MMQ might NOT fire:
            //   (a) batch_size below cutover → fall to dp4a batched residual
            //   (b) mmq_screen_weight rejected the weight → fall to fp16
            //       (preserves screen's design intent: rejected weights go
            //       to a higher-precision fallback, NOT to dp4a which has
            //       the same Q8_1 quantization step that MMQ failed on).
            let mut mmq_screen_rejected = false;
            if self.arch_caps.is_gfx906() && self.arch_caps.should_use_mmq(batch_size) {
                let use_mmq = if self.mmq_screen.enabled {
                    self.mmq_screen_weight(a_raw, m, k)
                } else {
                    true
                };
                if use_mmq {
                    return self.gemm_hfq4g256_residual_mmq_gfx906(a_raw, x, y, m, k, batch_size);
                }
                mmq_screen_rejected = self.mmq_screen.enabled;
            }

            // gfx906 dp4a batched residual (issue #276 Gap 2, HFQ4 sibling of
            // HFQ6 Phase A.2). Fires for B>1 below the MMQ cutover (B ∈
            // [2, 7] on gfx906 by should_use_mmq's default). Wins on
            // per-call ALU (dp4a issues 4 int8 multiplies + 4 accumulates
            // per cycle, vs FP wave64 hybrid's hfma2 at 2 mul + 2 add per
            // cycle → 2× FLOPs/cycle) and reuses the existing Q8_1 scratch.
            //
            // Skipped when MMQ screening rejected (preserves screen's
            // higher-precision fallback intent — dp4a has the same Q8_1
            // quantization step that MMQ already failed on for this
            // weight).
            //
            // The `!self.graphs.capture_mode` guard: `ensure_q8_1_mmq_x` (and the
            // downstream `ensure_kernel` for this kernel) can fire `hipMalloc`
            // / JIT-compile on first use, both unsafe inside an active capture.
            // The internal Q8_1 quantize launch itself goes through
            // `launch_maybe_blob` and IS recorded into the captured graph;
            // the guard protects only first-use-only side effects.
            if !mmq_screen_rejected
                && self.arch_caps.gemv_dp4a_enabled()
                && !self.graphs.capture_mode
            {
                return self.gemm_hfq4g256_residual_wave64_dp4a(a_raw, x, y, m, k, batch_size);
            }

            // Wave64 FP16 hybrid — best of both worlds for gfx906 (MI50).
            // Also the safe fallback when MMQ screen rejected the weight.
            if self.arch_caps.is_gcn5_wave64() {
                return self.gemm_hfq4g256_residual_fp16_wave64(a_raw, x, y, m, k, batch_size);
            }

            // Opt-in MMQ path (RDNA3/3.5, HIPFIRE_MMQ=1 or HIPFIRE_WO_MMQ=1).
            if self.flags.wo_mmq || self.arch_caps.should_use_mmq(batch_size) {
                let use_mmq = if self.mmq_screen.enabled {
                    self.mmq_screen_weight(a_raw, m, k)
                } else {
                    true
                };
                if use_mmq {
                    return self.gemm_hfq4g256_residual_mmq(a_raw, x, y, m, k, batch_size);
                }
            }

            // WMMA on gfx12 (RDNA4): K2-unroll port
            if self.arch_caps.has_wmma_w32_gfx12() {
                return self.gemm_hfq4g256_residual_wmma_gfx12(a_raw, x, y, m, k, batch_size);
            }

            // WMMA on gfx11+ (RDNA3): 16×16 tiled, ~8-10× over scalar
            if self.arch_caps.has_wmma_w32() {
                return self.gemm_hfq4g256_residual_wmma(a_raw, x, y, m, k, batch_size);
            }

            // FP16 packed on all other RDNA: ~15% prefill improvement
            return self.gemm_hfq4g256_residual_fp16(a_raw, x, y, m, k, batch_size);
        }

        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemm_hfq4g256_residual_wave64",
                kernels::GEMM_HFQ4G256_RESIDUAL_WAVE64_SRC,
                "gemm_hfq4g256_residual_wave64",
            )?;
            ("gemm_hfq4g256_residual_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel(
                "gemm_hfq4g256_residual",
                kernels::GEMM_HFQ4G256_RESIDUAL_SRC,
                "gemm_hfq4g256_residual",
            )?;
            ("gemm_hfq4g256_residual", [32, 1, 1], 1)
        };
        let func = &self.functions[func_name];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let grid_x = (m as u32 + grid_div - 1) / grid_div;

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 4 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_residual", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                block,
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

    /// Batched HFQ3-G256 GEMM with fused residual add (MQ3 path).
    ///
    /// HFQ3 sibling of `gemm_hfq4g256_residual` — single scalar variant,
    /// 104 B group stride and 3-bit unpack. Phase 1 of the gfx10 MQ3
    /// prefill plan. Used for batched prefill of the post-attention
    /// (wo) and post-FFN (w_down) projections.
    ///
    /// `x`: [batch_size × K] row-major, `y`: [batch_size × M] row-major.
    /// `y` must already hold the residual summand to accumulate into.
    pub fn gemm_hfq3g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Phase 3 experimental: wave32 MMQ is default-on for the supported
        // allowlist unless HIPFIRE_HFQ3_MMQ=0. Layer-gate is a no-op when
        // unset (#302).
        if batch_size > 1 && self.arch_caps.has_hfq3_mmq() && self.flags.hfq3_mmq_layer_gate_pass()
        {
            return self.gemm_hfq3g256_residual_mmq(a_raw, x, y, m, k, batch_size);
        }
        // FP16 fast paths — Phase 2b (dot2) + Phase 2c (fp16 fallback).
        // Layer-aware FP16 gate (#302).
        if batch_size > 1 && !self.flags.fp16_disabled_for_current_layer() {
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_hfq3g256_residual_dot2(a_raw, x, y, m, k, batch_size);
            }
            return self.gemm_hfq3g256_residual_fp16(a_raw, x, y, m, k, batch_size);
        }
        self.ensure_kernel(
            "gemm_hfq3g256_residual",
            kernels::GEMM_HFQ3G256_RESIDUAL_SRC,
            "gemm_hfq3g256_residual",
        )?;
        let func = &self.functions["gemm_hfq3g256_residual"];

        let mut ap = a_raw.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let bytes = crate::profile::gemm_hfq3g256_bytes(m, k, batch_size);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq3g256_residual", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// v_dot2_f32_f16-accelerated batched HFQ3-G256 residual GEMM (Y += A·X).
    /// HFQ3 sibling of `gemm_hfq4g256_residual_fp16`, upgraded from
    /// v_pk_fma_f16 to v_dot2_f32_f16. Phase 2b.
    pub fn gemm_hfq3g256_residual_dot2(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq3g256_residual_dot2",
            kernels::GEMM_HFQ3G256_RESIDUAL_DOT2_SRC,
            "gemm_hfq3g256_residual_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_hfq3g256_residual_dot2"];

        let mut ap = a_raw.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yp = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let bytes = crate::profile::gemm_hfq3g256_bytes(m, k, batch_size) + batch_size * k * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq3g256_residual_dot2", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// v_pk_fma_f16-accelerated batched HFQ3-G256 residual GEMM (Y += A·X).
    /// Fallback for archs without the dot extension (gfx1010, gfx1013).
    /// Phase 2c of the gfx10 MQ3 prefill plan.
    pub fn gemm_hfq3g256_residual_fp16(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq3g256_residual_fp16",
            kernels::GEMM_HFQ3G256_RESIDUAL_FP16_SRC,
            "gemm_hfq3g256_residual_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_hfq3g256_residual_fp16"];

        let mut ap = a_raw.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yp = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let bytes = crate::profile::gemm_hfq3g256_bytes(m, k, batch_size) + batch_size * k * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq3g256_residual_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// FP16-input batched HFQ4-G256 GEMM with residual add.
    /// Converts X from FP32 to FP16 (halving X bandwidth), then runs the
    /// FP16-packed GEMM kernel. The conversion is a one-shot pass amortized
    /// across M rows.
    pub fn gemm_hfq4g256_residual_fp16(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor, // FP32 [batch_size × K]
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_fp16",
            kernels::GEMM_HFQ4G256_RESIDUAL_FP16_SRC,
            "gemm_hfq4g256_residual_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        // FP16 GEMM
        let func = &self.functions["gemm_hfq4g256_residual_fp16"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k * 2  // FP16 X (half bandwidth!)
            + batch_size * m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_residual_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// Wave64 FP16 hybrid batched HFQ4-G256 GEMM with fused residual add.
    /// Combines wave64 block structure (2 rows/block, full lane utilization) with
    /// FP16 packed arithmetic (__hfma2). Target: gfx906 (MI50) prefill optimization.
    #[allow(clippy::too_many_arguments)]
    /// MFMA-direct HFQ4G256 GEMM with residual add for gfx942 (MI300X CDNA3).
    /// Channel-test verified at max_rel_err = 2e-5 vs FP16 scalar reference.
    pub fn gemm_hfq4g256_residual_mfma_gfx942(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mfma_gfx942",
            kernels::GEMM_HFQ4G256_RESIDUAL_MFMA_GFX942_SRC,
            "gemm_hfq4g256_residual_mfma_gfx942",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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
        let grid_x = ((m as u32) + 15) / 16;
        let grid_y = ((batch_size as u32) + 15) / 16;
        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_residual_mfma_gfx942",
            bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(
                &self.functions["gemm_hfq4g256_residual_mfma_gfx942"],
                [grid_x, grid_y, 1],
                [64, 1, 1],
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

    pub fn gemm_hfq4g256_residual_mfma_v2_gfx942(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mfma_v2_gfx942",
            kernels::GEMM_HFQ4G256_RESIDUAL_MFMA_V2_GFX942_SRC,
            "gemm_hfq4g256_residual_mfma_v2_gfx942",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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
        let grid_x = ((m as u32) + 31) / 32;
        let grid_y = ((batch_size as u32) + 31) / 32;
        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_residual_mfma_v2_gfx942",
            bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(
                &self.functions["gemm_hfq4g256_residual_mfma_v2_gfx942"],
                [grid_x, grid_y, 1],
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

    pub fn gemm_hfq4g256_residual_mfma_v3_gfx942(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mfma_v3_gfx942",
            kernels::GEMM_HFQ4G256_RESIDUAL_MFMA_V3_GFX942_SRC,
            "gemm_hfq4g256_residual_mfma_v3_gfx942",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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
        let grid_x = ((m as u32) + 31) / 32;
        let grid_y = ((batch_size as u32) + 31) / 32;
        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_residual_mfma_v3_gfx942",
            bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(
                &self.functions["gemm_hfq4g256_residual_mfma_v3_gfx942"],
                [grid_x, grid_y, 1],
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

    pub fn gemm_hfq4g256_residual_mfma_v4_gfx942(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_mfma_v4_gfx942",
            kernels::GEMM_HFQ4G256_RESIDUAL_MFMA_V4_GFX942_SRC,
            "gemm_hfq4g256_residual_mfma_v4_gfx942",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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
        let grid_x = ((m as u32) + 15) / 16;
        let grid_y = ((batch_size as u32) + 63) / 64;
        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_residual_mfma_v4_gfx942",
            bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(
                &self.functions["gemm_hfq4g256_residual_mfma_v4_gfx942"],
                [grid_x, grid_y, 1],
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

    pub fn gemm_hfq4g256_residual_fp16_wave64(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor, // FP32 [batch_size × K]
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_fp16_wave64",
            kernels::GEMM_HFQ4G256_RESIDUAL_FP16_WAVE64_SRC,
            "gemm_hfq4g256_residual_fp16_wave64",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let func = &self.functions["gemm_hfq4g256_residual_fp16_wave64"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let grid_x = (m as u32 + 1) / 2;

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k * 2  // FP16 X (half bandwidth!)
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_residual_fp16_wave64",
            bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_tiles as u32, 1],
                [64, 1, 1],
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

    /// Experimental llama.cpp-style MMQ residual GEMM for HFQ4-G256.
    /// Opt-in only via `HIPFIRE_WO_MMQ=1` while the tiled path is validated.
    pub fn gemm_hfq4g256_residual_mmq(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        // gfx12 (RDNA4) uses a separate single-wave 16-row-tile MMQ port; the
        // RDNA3 LDS-tile source #if-excludes gfx12 (empty stub -> garbage).
        let is_gfx12 = self.arch_caps.is_rdna4();
        let kernel_name = if is_gfx12 {
            // gfx12 port: full_add when the tile is exactly filled (M and N
            // multiples of 16), else the bounds-clamped residual kernel.
            if m % 16 == 0 && batch_size % 16 == 0 {
                "gemm_hfq4g256_residual_mmq_full_add"
            } else {
                "gemm_hfq4g256_residual_mmq"
            }
        } else if m % 128 == 0 && batch_size % 128 == 0 {
            "gemm_hfq4g256_residual_mmq_full_add"
        } else {
            "gemm_hfq4g256_residual_mmq"
        };
        let src = if is_gfx12 {
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX12_SRC
        } else {
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_SRC
        };
        // Module name must be arch-distinct: `ensure_q8_1_mmq_x` already
        // compiled+loaded the RDNA3 source under module
        // "gemm_hfq4g256_residual_mmq" (to grab `quantize_q8_1_mmq_ds4`).
        // The compiler/module cache is keyed by module NAME only, so loading
        // the gfx12 source under the same name short-circuits to the RDNA3
        // module — whose body #if-excludes gfx12, leaving the gemm symbols as
        // empty stubs (output == input buffer, NRMSE ~100%). Use a distinct
        // module name for gfx12 while keeping the kernel SYMBOL names constant.
        let module_name = if is_gfx12 {
            "gemm_hfq4g256_residual_mmq_gfx12"
        } else {
            "gemm_hfq4g256_residual_mmq"
        };
        self.ensure_kernel(module_name, src, kernel_name)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut add_val = 1i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];

        // gfx12: single-wave 16-row × 16-col tile, block [32,1,1], LDS 0.
        // RDNA3: 128-row × 128-col LDS-staged tile, block [32,8,1].
        let (grid, block, shared_mem) = if is_gfx12 {
            let row_tiles = (m + 15) / 16;
            let col_tiles = (batch_size + 15) / 16;
            ([row_tiles as u32, col_tiles as u32, 1], [32u32, 1, 1], 0u32)
        } else {
            const MMQ_X: usize = 128;
            const MMQ_Y: usize = 128;
            const MMQ_TILE_Y_K: usize = 36;
            const MMQ_TILE_X_K: usize = 76;
            let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
            let batch_tiles = (batch_size + MMQ_X - 1) / MMQ_X;
            let shared_mem =
                ((MMQ_X * MMQ_TILE_Y_K + MMQ_Y * MMQ_TILE_X_K) * std::mem::size_of::<i32>()) as u32;
            (
                [row_tiles as u32, batch_tiles as u32, 1],
                [32u32, 8, 1],
                shared_mem,
            )
        };

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k + batch_size * m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_residual_mmq", bytes);
        let result =
            self.launch_maybe_blob(kernel_name, grid, block, shared_mem, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(add_val);
                b
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx906 dp4a MMQ residual GEMM. Wave-native topology (block 64×2,
    /// tile 128×64) per llama.cpp-gfx906 reference. Distinct from the
    /// RDNA3 i8-WMMA variant above — different block dim, different
    /// LDS layout, different kernel symbols.
    ///
    /// Phase 1 implementation; opt-in via `HIPFIRE_MMQ=1` while correctness
    /// is being validated. See plans/gfx906_mmq_plan.md and
    /// plans/p1.2_dp4a_mmq_design.md.
    pub fn gemm_hfq4g256_residual_mmq_gfx906(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Quantize activations to Q8_1.
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        // Diagnostic: HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY=1 isolates the cost of
        // the Q8_1 activation pre-quantize by running the FP16 wave64 path
        // *after* paying the quantize cost. The flag is read once at init
        // (see `Gpu::new`) so this check is a single bool load, not a
        // per-call env::var lookup.
        if self.flags.mmq_diag_quantize_only {
            let _ = x_q8_ptr;
            return self.gemm_hfq4g256_residual_fp16_wave64(a_raw, x, y, m, k, batch_size);
        }

        // Greedy mmq_x selection matching stock.
        let mmq_x = if batch_size <= 8 {
            8
        } else if batch_size <= 16 {
            16
        } else if batch_size <= 24 {
            24
        } else if batch_size <= 32 {
            32
        } else if batch_size <= 40 {
            40
        } else if batch_size <= 48 {
            48
        } else if batch_size <= 56 {
            56
        } else {
            64
        };

        // Pick variant name and source.
        let is_full = m % 128 == 0 && batch_size % mmq_x == 0;
        let base_name = "gemm_hfq4g256_residual_mmq_gfx906";
        let kernel_name = if is_full {
            format!("{}_full_add_x{}", base_name, mmq_x)
        } else {
            format!("{}_x{}", base_name, mmq_x)
        };

        let wrapper_src = match mmq_x {
            8 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X8_SRC,
            16 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X16_SRC,
            24 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X24_SRC,
            32 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X32_SRC,
            40 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X40_SRC,
            48 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X48_SRC,
            56 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X56_SRC,
            64 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X64_SRC,
            _ => unreachable!(),
        };
        // Inline the body .cuh: the runtime hipcc compiles from cache_dir,
        // which doesn't have kernels/src on its -I path. Strip the
        // `#include "..._body.cuh"` line and prepend the body content.
        let inlined = wrapper_src.replace(
            "#include \"gemm_hfq4g256_residual_mmq_gfx906_body.cuh\"",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_BODY_CUH,
        );

        self.ensure_kernel(&format!("{}_x{}", base_name, mmq_x), &inlined, &kernel_name)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut add_val = 1i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];

        // Option C streaming topology — KEEP IN SYNC WITH body.cuh:
        //   x_qs   : MMQ_Y * x_stride ints  (per-mmq_x: 40 if mmq_x≥32 else 33)
        //   x_dm   : MMQ_Y float2
        //   tile_y : mmq_x * Y_STRIDE ints
        const MMQ_Y: usize = 128;
        let x_stride: usize = if mmq_x >= 32 { 40 } else { 33 };
        const Y_STRIDE: usize = 36;
        const X_DM_HALF2: usize = 128;
        let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
        let batch_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let shared_mem =
            ((MMQ_Y * x_stride * 4) + (X_DM_HALF2 * 8) + (mmq_x * Y_STRIDE * 4)) as u32;
        // 2 WGs/CU on gfx906 needs ≤32 KiB/WG (64 KiB cap).
        debug_assert!(
            shared_mem as usize <= 32 * 1024,
            "gfx906 MMQ LDS budget exceeded: {} B > 32 KiB",
            shared_mem
        );

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", base_name, bytes);
        let result = self.launch_maybe_blob(
            &kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [64, 4, 1], // nwarps=4
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(add_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// HFQ4 qkv MMQ fused-projection kernel — gfx906 wave64. 3-way fused
    /// {Q, K, V} on a single launch, eliminating 2 of 3 launch overheads
    /// and amortizing L2 hits on the Q8_1 batch tile across the three
    /// outputs. See `kernels/src/gemm_qkv_hfq4g256_mmq_gfx906_body.cuh`
    /// for the kernel design and
    /// `experiments/gfx906-fused-mmq/probe-results.md` for the §6.1
    /// probe that motivated this work.
    ///
    /// Caller invariants:
    ///   - q_m, k_m, v_m must each be multiples of MMQ_Y(=128). Qwen3.5
    ///     family satisfies (9B: q_m=4096, k_m=v_m=1024; 4B: q_m=2048,
    ///     k_m=v_m=512).
    ///   - batch_size ≥ should_use_mmq cutover (gfx906 default: 8).
    ///   - x is the same activation tensor as the residual sibling
    ///     expects (raw fp16, ensure_q8_1_mmq_x is called internally).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq4g256_mmq_gfx906(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert!(
            self.arch_caps.should_use_mmq(batch_size) || self.graphs.capture_mode,
            "qkv_hfq4g256_mmq_gfx906 called at non-winning B={} (capture={})",
            batch_size,
            self.graphs.capture_mode,
        );
        debug_assert!(
            q_m % 128 == 0 && k_m % 128 == 0 && v_m % 128 == 0,
            "qkv_hfq4g256_mmq_gfx906 requires q_m/k_m/v_m multiples of MMQ_Y=128 (got q={q_m} k={k_m} v={v_m})",
        );

        // Same mmq_x sweep as the gfx906 single-output mmq_set path so
        // future MMQ_X tuning translates 1:1. Note: only the {8,16,32,64}
        // quartet is wired up initially (the most common batch buckets);
        // the in-between values fall up to the next available mmq_x.
        let mmq_x = if batch_size <= 8 {
            8
        } else if batch_size <= 16 {
            16
        } else if batch_size <= 32 {
            32
        } else {
            64
        };

        let is_full = q_m % 128 == 0 && k_m % 128 == 0 && v_m % 128 == 0 && batch_size % mmq_x == 0;
        let base_name = "gemm_qkv_hfq4g256_mmq_gfx906";
        let kernel_name = if is_full {
            format!("{}_full_set_x{}", base_name, mmq_x)
        } else {
            format!("{}_x{}", base_name, mmq_x)
        };

        let wrapper_src = match mmq_x {
            8 => kernels::GEMM_QKV_HFQ4G256_MMQ_GFX906_X8_SRC,
            16 => kernels::GEMM_QKV_HFQ4G256_MMQ_GFX906_X16_SRC,
            32 => kernels::GEMM_QKV_HFQ4G256_MMQ_GFX906_X32_SRC,
            64 => kernels::GEMM_QKV_HFQ4G256_MMQ_GFX906_X64_SRC,
            _ => unreachable!(),
        };
        let inlined = wrapper_src.replace(
            "#include \"gemm_qkv_hfq4g256_mmq_gfx906_body.cuh\"",
            kernels::GEMM_QKV_HFQ4G256_MMQ_GFX906_BODY_CUH,
        );
        self.ensure_kernel(&format!("{}_x{}", base_name, mmq_x), &inlined, &kernel_name)?;

        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        let mut a_q_p = a_q.buf.as_ptr();
        let mut a_k_p = a_k.buf.as_ptr();
        let mut a_v_p = a_v.buf.as_ptr();
        let mut xq = xq_ptr;
        let mut y_q_p = y_q.buf.as_ptr();
        let mut y_k_p = y_k.buf.as_ptr();
        let mut y_v_p = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_q_p as *mut _ as *mut c_void,
            &mut a_k_p as *mut _ as *mut c_void,
            &mut a_v_p as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut y_q_p as *mut _ as *mut c_void,
            &mut y_k_p as *mut _ as *mut c_void,
            &mut y_v_p as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        // Option C streaming topology — KEEP IN SYNC WITH body.cuh.
        // X_STRIDE varies with mmq_x (see body.cuh x_stride_for<>):
        //   mmq_x ≥ 32 → stride 40 (b128 path), mmq_x < 32 → stride 33 (b32).
        const MMQ_Y: usize = 128;
        let x_stride: usize = if mmq_x >= 32 { 40 } else { 33 };
        const Y_STRIDE: usize = 36;
        const X_DM_FLOAT2: usize = 128;
        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + MMQ_Y - 1) / MMQ_Y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;
        let shared_mem =
            ((MMQ_Y * x_stride * 4) + (X_DM_FLOAT2 * 8) + (mmq_x * Y_STRIDE * 4)) as u32;
        debug_assert!(
            shared_mem as usize <= 32 * 1024,
            "gfx906 qkv MMQ LDS budget exceeded: {} B > 32 KiB",
            shared_mem
        );

        // Total bytes = weight read (Q+K+V) + X read (Q8_1) + Y writes (3 outputs).
        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + batch_size * k
            + batch_size * (q_m + k_m + v_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_mmq_gfx906", bytes);
        let result = self.launch_maybe_blob(
            &kernel_name,
            [row_tiles as u32, col_tiles as u32, 1],
            [64, 4, 1], // wave64 native: 4 wave64s = 256 threads
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_q_p);
                b.push_ptr(a_k_p);
                b.push_ptr(a_v_p);
                b.push_ptr(xq);
                b.push_ptr(y_q_p);
                b.push_ptr(y_k_p);
                b.push_ptr(y_v_p);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
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

    /// HFQ4 gate_up MMQ fused-projection kernel — gfx906 wave64. 2-way
    /// fused {a, b} on a single launch. Generic naming so the same
    /// entry serves BOTH gate_up (a=gate, b=up) and LA QKVZA-head
    /// (a=qkv, b=z) dispatch sites. See
    /// `kernels/src/gemm_gate_up_hfq4g256_mmq_gfx906_body.cuh`.
    ///
    /// Caller invariants:
    ///   - m_a, m_b multiples of MMQ_Y(=128).
    ///   - batch_size ≥ should_use_mmq cutover (gfx906 default: 8).
    ///   - x is the raw fp16 activation (ensure_q8_1_mmq_x is called
    ///     internally; caller MAY pre-quantize via the prequant
    ///     sibling below if Xq is already on hand from another call
    ///     on the same x).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq4g256_mmq_gfx906(
        &mut self,
        a_a: &GpuTensor,
        a_b: &GpuTensor,
        x: &GpuTensor,
        y_a: &GpuTensor,
        y_b: &GpuTensor,
        m_a: usize,
        m_b: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        self.gemm_gate_up_hfq4g256_mmq_gfx906_prequant(
            a_a, a_b, xq_ptr, y_a, y_b, m_a, m_b, k, batch_size,
        )
    }

    /// Pre-quantized X variant — caller passes the Q8_1 scratch pointer
    /// produced by an earlier `ensure_q8_1_mmq_x(x, batch_size, k)` call.
    /// Used by the LA QKVZA-head site, which has already quantized X
    /// when it computed the qkv/z 2-way and then continues to the
    /// β+α tail. Mirrors the `_prequant` sibling on the dp4a path.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq4g256_mmq_gfx906_prequant(
        &mut self,
        a_a: &GpuTensor,
        a_b: &GpuTensor,
        x_q8_ptr: *mut c_void,
        y_a: &GpuTensor,
        y_b: &GpuTensor,
        m_a: usize,
        m_b: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert!(
            self.arch_caps.should_use_mmq(batch_size) || self.graphs.capture_mode,
            "gate_up_hfq4g256_mmq_gfx906 called at non-winning B={} (capture={})",
            batch_size,
            self.graphs.capture_mode,
        );

        // MMQ_Y selection. Y=64 is the higher-occupancy variant (plan §6.5);
        // wrappers only exist for {x16, x32}, so larger mmq_x at y64 falls
        // back to y128. Y=128 is the established default (matches the
        // residual sibling).
        let y64 = self.flags.hfq4_mmq_gfx906_y64_enabled();
        let mmq_y: usize = if y64 { 64 } else { 128 };

        debug_assert!(
            m_a % mmq_y == 0 && m_b % mmq_y == 0,
            "gate_up_hfq4g256_mmq_gfx906 requires m_a/m_b multiples of MMQ_Y={mmq_y} (got a={m_a} b={m_b})",
        );

        let mut mmq_x = if batch_size <= 8 {
            8
        } else if batch_size <= 16 {
            16
        } else if batch_size <= 32 {
            32
        } else {
            64
        };
        // Y=64 only has wrappers for x16 and x32; cap mmq_x at 32 when
        // y64 is requested. Falls through to the y128 path for tiny
        // batches (x8) since no x8_y64 wrapper exists.
        let use_y64 = y64 && mmq_x >= 16;
        if use_y64 && mmq_x > 32 {
            mmq_x = 32;
        }

        let is_full = m_a % mmq_y == 0 && m_b % mmq_y == 0 && batch_size % mmq_x == 0;
        let base_name = "gemm_gate_up_hfq4g256_mmq_gfx906";
        let y_suffix = if use_y64 { "_y64" } else { "" };
        let kernel_name = if is_full {
            format!("{}_full_set_x{}{}", base_name, mmq_x, y_suffix)
        } else {
            format!("{}_x{}{}", base_name, mmq_x, y_suffix)
        };

        let wrapper_src = match (mmq_x, use_y64) {
            (8, false) => kernels::GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X8_SRC,
            (16, false) => kernels::GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X16_SRC,
            (32, false) => kernels::GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X32_SRC,
            (64, false) => kernels::GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X64_SRC,
            (16, true) => kernels::GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X16_Y64_SRC,
            (32, true) => kernels::GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_X32_Y64_SRC,
            _ => unreachable!("no gate_up wrapper for mmq_x={mmq_x} y64={use_y64}"),
        };
        let inlined = wrapper_src.replace(
            "#include \"gemm_gate_up_hfq4g256_mmq_gfx906_body.cuh\"",
            kernels::GEMM_GATE_UP_HFQ4G256_MMQ_GFX906_BODY_CUH,
        );
        self.ensure_kernel(
            &format!("{}_x{}{}", base_name, mmq_x, y_suffix),
            &inlined,
            &kernel_name,
        )?;

        let mut a_a_p = a_a.buf.as_ptr();
        let mut a_b_p = a_b.buf.as_ptr();
        let mut xq = x_q8_ptr;
        let mut y_a_p = y_a.buf.as_ptr();
        let mut y_b_p = y_b.buf.as_ptr();
        let mut m_a_val = m_a as i32;
        let mut m_b_val = m_b as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_a_p as *mut _ as *mut c_void,
            &mut a_b_p as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut y_a_p as *mut _ as *mut c_void,
            &mut y_b_p as *mut _ as *mut c_void,
            &mut m_a_val as *mut _ as *mut c_void,
            &mut m_b_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        // KEEP IN SYNC WITH body.cuh: MMQ_Y is chosen by the wrapper
        // (`use_y64` above gates which wrapper is included). x_dm sizing
        // is MMQ_Y float2s. row_tiles uses the same MMQ_Y.
        let x_stride: usize = if mmq_x >= 32 { 40 } else { 33 };
        const Y_STRIDE: usize = 36;
        let x_dm_float2: usize = mmq_y;
        let total_m = m_a + m_b;
        let row_tiles = (total_m + mmq_y - 1) / mmq_y;
        let col_tiles = (batch_size + mmq_x - 1) / mmq_x;
        let shared_mem =
            ((mmq_y * x_stride * 4) + (x_dm_float2 * 8) + (mmq_x * Y_STRIDE * 4)) as u32;
        debug_assert!(
            shared_mem as usize <= 32 * 1024,
            "gfx906 gate_up MMQ LDS budget exceeded: {} B > 32 KiB",
            shared_mem
        );

        let bytes = crate::profile::gemv_hfq4g256_bytes(m_a, k)
            + crate::profile::gemv_hfq4g256_bytes(m_b, k)
            + batch_size * k
            + batch_size * (m_a + m_b) * 4;
        // Distinct timer label per Y variant so attribution shows the split.
        let timer_label: &'static str = if use_y64 {
            "gemm_gate_up_hfq4g256_mmq_gfx906_y64"
        } else {
            "gemm_gate_up_hfq4g256_mmq_gfx906"
        };
        let timer = crate::profile::begin_timer(&self.hip, "gemm", timer_label, bytes);
        let result = self.launch_maybe_blob(
            &kernel_name,
            [row_tiles as u32, col_tiles as u32, 1],
            [64, 4, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_a_p);
                b.push_ptr(a_b_p);
                b.push_ptr(xq);
                b.push_ptr(y_a_p);
                b.push_ptr(y_b_p);
                b.push_i32(m_a_val);
                b.push_i32(m_b_val);
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

    /// Set-mode (add=0) variant of the gfx906 MMQ kernel.
    pub fn gemm_hfq4g256_mmq_set_gfx906(
        &mut self,
        a_raw: &GpuTensor,
        x_q8_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Caller (fused dispatcher) is expected to gate via `should_use_mmq`;
        // the assert below enforces that contract so a future caller can't
        // silently route a non-winning batch through MMQ. Mirrors the MQ6
        // sibling's `hfq6_mmq_route` assert added in 5ea9050.
        debug_assert!(
            self.arch_caps.should_use_mmq(batch_size) || self.graphs.capture_mode,
            "_mmq_set_gfx906 called at non-winning B={} (capture={}) — \
             caller must gate via should_use_mmq first",
            batch_size,
            self.graphs.capture_mode,
        );
        let mmq_x = if batch_size <= 8 {
            8
        } else if batch_size <= 16 {
            16
        } else if batch_size <= 24 {
            24
        } else if batch_size <= 32 {
            32
        } else if batch_size <= 40 {
            40
        } else if batch_size <= 48 {
            48
        } else if batch_size <= 56 {
            56
        } else {
            64
        };

        let is_full = m % 128 == 0 && batch_size % mmq_x == 0;
        let base_name = "gemm_hfq4g256_residual_mmq_gfx906";
        let kernel_name = if is_full {
            format!("{}_full_set_x{}", base_name, mmq_x)
        } else {
            format!("{}_x{}", base_name, mmq_x)
        };

        let wrapper_src = match mmq_x {
            8 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X8_SRC,
            16 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X16_SRC,
            24 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X24_SRC,
            32 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X32_SRC,
            40 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X40_SRC,
            48 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X48_SRC,
            56 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X56_SRC,
            64 => kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_X64_SRC,
            _ => unreachable!(),
        };
        let inlined = wrapper_src.replace(
            "#include \"gemm_hfq4g256_residual_mmq_gfx906_body.cuh\"",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX906_BODY_CUH,
        );

        self.ensure_kernel(&format!("{}_x{}", base_name, mmq_x), &inlined, &kernel_name)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut add_val = 0i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];

        // Option C streaming topology — KEEP IN SYNC WITH body.cuh
        // (same layout invariant as residual variant above).
        const MMQ_Y: usize = 128;
        let x_stride: usize = if mmq_x >= 32 { 40 } else { 33 };
        const Y_STRIDE: usize = 36;
        const X_DM_HALF2: usize = 128;
        let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
        let batch_tiles = (batch_size + mmq_x - 1) / mmq_x;

        let shared_mem =
            ((MMQ_Y * x_stride * 4) + (X_DM_HALF2 * 8) + (mmq_x * Y_STRIDE * 4)) as u32;
        debug_assert!(
            shared_mem as usize <= 32 * 1024,
            "gfx906 MMQ LDS budget exceeded: {} B > 32 KiB",
            shared_mem
        );

        // bytes = weight read + X read (Q8_1, ~1 byte/element + scale) + Y write (set, no read).
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k + batch_size * m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_mmq_set_gfx906", bytes);
        let result = self.launch_maybe_blob(
            &kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [64, 4, 1], // nwarps=4
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(add_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    pub fn gemm_hfq4g256_mmq_set(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let is_gfx12 = self.arch_caps.is_rdna4();
        let kernel_name = if is_gfx12 {
            if m % 16 == 0 && batch_size % 16 == 0 {
                "gemm_hfq4g256_residual_mmq_full_set"
            } else {
                "gemm_hfq4g256_residual_mmq"
            }
        } else if m % 128 == 0 && batch_size % 128 == 0 {
            "gemm_hfq4g256_residual_mmq_full_set"
        } else {
            "gemm_hfq4g256_residual_mmq"
        };
        let src = if is_gfx12 {
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX12_SRC
        } else {
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_SRC
        };
        // Arch-distinct module name on gfx12 — see gemm_hfq4g256_residual_mmq
        // for the cache-collision rationale (RDNA3 source pre-loaded under the
        // shared name by ensure_q8_1_mmq_x → gfx12 source short-circuited →
        // empty-stub symbols). Symbol names stay constant; only the module
        // cache key differs.
        let module_name = if is_gfx12 {
            "gemm_hfq4g256_residual_mmq_gfx12"
        } else {
            "gemm_hfq4g256_residual_mmq"
        };
        self.ensure_kernel(module_name, src, kernel_name)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        // For gfx12 the non-full residual kernel also reads `add`; set=0.
        let mut add_val = 0i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];

        let (grid, block, shared_mem) = if is_gfx12 {
            let row_tiles = (m + 15) / 16;
            let col_tiles = (batch_size + 15) / 16;
            ([row_tiles as u32, col_tiles as u32, 1], [32u32, 1, 1], 0u32)
        } else {
            const MMQ_X: usize = 128;
            const MMQ_Y: usize = 128;
            const MMQ_TILE_Y_K: usize = 36;
            const MMQ_TILE_X_K: usize = 76;
            let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
            let batch_tiles = (batch_size + MMQ_X - 1) / MMQ_X;
            let shared_mem =
                ((MMQ_X * MMQ_TILE_Y_K + MMQ_Y * MMQ_TILE_X_K) * std::mem::size_of::<i32>()) as u32;
            (
                [row_tiles as u32, batch_tiles as u32, 1],
                [32u32, 8, 1],
                shared_mem,
            )
        };

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_mmq_set", bytes);
        let result =
            self.launch_maybe_blob(kernel_name, grid, block, shared_mem, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(add_val);
                b
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    pub fn gemm_hfq4g256_mmq_set_prequant(
        &mut self,
        a_raw: &GpuTensor,
        x_q8_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.is_gfx906() {
            // gfx906 has its own dispatcher (`gemm_hfq4g256_residual_mmq_gfx906`)
            // that handles its own quantize internally, called directly from
            // mmq_screen_weight on gfx906. _set_prequant is RDNA3/gfx12-only.
            return Err(hip_bridge::HipError::new(
                0,
                "gemm_hfq4g256_mmq_set_prequant is not supported on gfx906; \
                 callers should route to gemm_hfq4g256_residual_mmq_gfx906 directly",
            ));
        }
        let is_gfx12 = self.arch_caps.is_rdna4();
        let kernel_name = if is_gfx12 {
            if m % 16 == 0 && batch_size % 16 == 0 {
                "gemm_hfq4g256_residual_mmq_full_set"
            } else {
                "gemm_hfq4g256_residual_mmq"
            }
        } else if m % 128 == 0 && batch_size % 128 == 0 {
            "gemm_hfq4g256_residual_mmq_full_set"
        } else {
            "gemm_hfq4g256_residual_mmq"
        };
        let src = if is_gfx12 {
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_GFX12_SRC
        } else {
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_SRC
        };
        // Arch-distinct module name on gfx12 — see gemm_hfq4g256_residual_mmq
        // for the cache-collision rationale (RDNA3 source pre-loaded under the
        // shared name by ensure_q8_1_mmq_x → gfx12 source short-circuited →
        // empty-stub symbols). Symbol names stay constant; only the module
        // cache key differs.
        let module_name = if is_gfx12 {
            "gemm_hfq4g256_residual_mmq_gfx12"
        } else {
            "gemm_hfq4g256_residual_mmq"
        };
        self.ensure_kernel(module_name, src, kernel_name)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut xq_ptr = x_q8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut add_val = 0i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut add_val as *mut _ as *mut c_void,
        ];

        let (grid, block, shared_mem) = if is_gfx12 {
            let row_tiles = (m + 15) / 16;
            let col_tiles = (batch_size + 15) / 16;
            ([row_tiles as u32, col_tiles as u32, 1], [32u32, 1, 1], 0u32)
        } else {
            const MMQ_X: usize = 128;
            const MMQ_Y: usize = 128;
            const MMQ_TILE_Y_K: usize = 36;
            const MMQ_TILE_X_K: usize = 76;
            let row_tiles = (m + MMQ_Y - 1) / MMQ_Y;
            let batch_tiles = (batch_size + MMQ_X - 1) / MMQ_X;
            let shared_mem =
                ((MMQ_X * MMQ_TILE_Y_K + MMQ_Y * MMQ_TILE_X_K) * std::mem::size_of::<i32>()) as u32;
            (
                [row_tiles as u32, batch_tiles as u32, 1],
                [32u32, 8, 1],
                shared_mem,
            )
        };

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_mmq_set", bytes);
        let result =
            self.launch_maybe_blob(kernel_name, grid, block, shared_mem, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(add_val);
                b
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// WMMA-accelerated batched HFQ4-G256 GEMM with residual add.
    /// gfx1100+ only. 16×16 output tiles via wave32 WMMA.
    /// Converts X to FP16, then uses __builtin_amdgcn_wmma_f32_16x16x16_f16_w32.
    pub fn gemm_hfq4g256_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Compile both kernels (convert + WMMA GEMM share the FP16 convert)
        // Kernel variant selection
        // MW16 path: dequant weights to FP16 per-call, then run no-dequant WMMA
        if self.flags.mw16 {
            return self.gemm_mw16_residual_wmma_via_dequant(a_raw, x, y, m, k, batch_size);
        }
        // LDS-staged multi-wave residual (env HIPFIRE_HFQ4G256_LDSSTAGE=1).
        // gfx11 sister of the gfx12 ldsstage path. Reorders FP32 K accumulation;
        // K must be a multiple of 512. Batch ceiling: see the measured 3.6-trunk
        // tables on LDSSTAGE_MAX_BATCH_GFX11 (one value covers dGPU and iGPU).
        if self.flags.hfq4g256_ldsstage_wmma
            && k % 512 == 0
            && batch_size <= LDSSTAGE_MAX_BATCH_GFX11
        {
            let kname = "gemm_hfq4g256_residual_wmma_ldsstage";
            let ksrc = kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_SRC;
            self.ensure_kernel(kname, ksrc, kname)?;
            let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

            let mut a_ptr = a_raw.buf.as_ptr();
            let mut x_ptr = x_f16_ptr;
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

            let row_tiles = (m + 15) / 16;
            let batch_tiles = (batch_size + 15) / 16;

            let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
                + batch_size * k * 2
                + batch_size * m * 4 * 2;
            let timer = crate::profile::begin_timer(&self.hip, "gemm", kname, bytes);
            let result = self.launch_maybe_blob(
                kname,
                [row_tiles as u32, batch_tiles as u32, 1],
                [256, 1, 1],
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
            return result;
        }

        // Shape-aware default: ksplit only pays for itself when the un-split
        // grid is CU-starved (target wo_residual at M=5120 → 320 blocks,
        // ~3.3/CU on gfx1100 — ksplit 4×'s it to 13/CU). For draft-FFN shapes
        // (M=17408, K=5120, B=16) the un-split grid is already 1088 blocks
        // (~11/CU) and the atomicAdd reduce is pure overhead. k2 removes the
        // split + atomics and runs deterministically.
        //
        // Threshold picked at M=8192: covers M∈{5120,6144} (target wo) on the
        // ksplit side and M∈{17408} (draft gate/up/down) on the k2 side. lm_head
        // (M=vocab) is always way above threshold → k2.
        //
        // HIPFIRE_WO_WMMA_VARIANT=ksplit_det|ksplit|k2|k2x32|k4|wmma|wmma2
        // overrides the auto selection (applies to every call, target+draft).
        //   ksplit_det — K-split occupancy (grid.z) WITHOUT the racing
        //            atomicAdd: each split writes its partial to scratch, a
        //            fixed-order finalize sums them → bit-reproducible. Perf
        //            parity with ksplit across all benched batches (16..1024)
        //            on gfx1100; default for the CU-starved small-M case.
        //   ksplit — K-split + atomicAdd (non-deterministic accum order;
        //            kept as a perf-reference / debug variant — see ksplit_det)
        //   k2     — 2× K-tile pipeline (byte-exact accum order)
        //   k2x32  — 32-row block with shared X fragment per K-tile. Slower
        //            than k2 on gfx1100, but faster on gfx1151 Strix Halo for
        //            small-M residual projections at prefill-sized batches.
        //            DFlash verify/lm_head runs at B<=16 and large-M draft
        //            FFN/lm_head also prefer k2.
        //   k4     — 4× K-tile pipeline. Fixed 2026-05-01 (commit pending):
        //            output mapping was swapped relative to K2's canonical
        //            wave32 WMMA C-mapping. Channel-test passes at K∈{256,512,4096}
        //            × batch∈{1,2,4,16}. At m<8192 (9B residual at m=4096) K4
        //            ties K2 within FP drift but loses to ksplit by ~33%
        //            per-call at small batch (CU-starved grid: 3.3 vs 13
        //            blocks/CU); auto-dispatch correctly stays on ksplit. K4
        //            vs K2 at m≥8192 not yet benched on available models. See
        //            plans/k4_plan.md.
        //   wmma   — base WMMA         (output-mapping bug — debug only)
        //   wmma2  — 2-wave block, 32 rows × 16 batch (output-mapping bug — debug only)
        let is_gfx115x = self.arch_caps.is_rdna3p5();
        // The CU-starved small-M case (gfx11 RDNA3 discrete, the `else` arm
        // below) needs K-split occupancy for throughput. The original ksplit
        // got it via an atomicAdd reduction across K_SPLITS partials, which is
        // fp-non-associative — order varies with warp scheduling, so output
        // bytes drift between processes/runs. The drift is sub-argmax-margin
        // per call but cascades on long greedy decode (>50 tokens), breaking
        // bit-reproducibility (and multi-GPU pp=1 vs pp=2 parity). `ksplit_det`
        // keeps the identical grid.z occupancy but replaces the race with a
        // scratch + fixed-order finalize → deterministic at perf parity
        // (benched 16..1024 batch on gfx1100), so it is now the default.
        // HIPFIRE_DETERMINISTIC=1 still forces k2 (single-block K reduction)
        // for the strictest single-kernel byte-parity escape hatch.
        // Cached — getenv on every decode token would re-parse 6× per layer
        // × N layers per step. Read once at first dispatch.
        static FORCE_DET: OnceLock<bool> = OnceLock::new();
        let force_det = *FORCE_DET.get_or_init(|| self.flags.deterministic);
        let auto_variant = if force_det {
            "k2"
        } else if is_gfx115x && batch_size <= 16 {
            "k2"
        } else if is_gfx115x && m < 8192 {
            "k2x32"
        } else if m >= 8192 {
            "k2"
        } else {
            "ksplit_det"
        };
        let variant_override = self.flags.wo_wmma_variant.clone();
        let variant = variant_override.as_deref().unwrap_or(auto_variant);
        // Deterministic K-split: same grid.z occupancy as ksplit but writes
        // partials to scratch + a fixed-order finalize instead of racing
        // atomicAdd. Two-kernel path → handled separately.
        if variant == "ksplit_det" {
            return self.gemm_hfq4g256_residual_wmma_ksplit_det(a_raw, x, y, m, k, batch_size);
        }
        let (kernel_name, kernel_src, block_size, row_step, k_splits) = match variant {
            "k2" => (
                "gemm_hfq4g256_residual_wmma_k2",
                kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_K2_SRC,
                32u32,
                16usize,
                1u32,
            ),
            "k2x32" => (
                "gemm_hfq4g256_residual_wmma_k2x32",
                kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_K2X32_SRC,
                32u32,
                32usize,
                1u32,
            ),
            "k4" => (
                "gemm_hfq4g256_residual_wmma_k4",
                kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_K4_SRC,
                32u32,
                16usize,
                1u32,
            ),
            "wmma" => (
                "gemm_hfq4g256_residual_wmma",
                kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_SRC,
                32u32,
                16usize,
                1u32,
            ),
            "wmma2" => (
                "gemm_hfq4g256_residual_wmma2",
                kernels::GEMM_HFQ4G256_RESIDUAL_WMMA2_SRC,
                64u32,
                32usize,
                1u32,
            ),
            _ => (
                "gemm_hfq4g256_residual_wmma_ksplit",
                kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_KSPLIT_SRC,
                32u32,
                16usize,
                4u32,
            ),
        };
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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

        let row_tiles = (m + row_step - 1) / row_step;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        // HIPFIRE_GEMM_DUMP=1: per-call shape+wall-clock dump of this kernel.
        // Synchronously times only the ksplit kernel launch (not memset / convert).
        // Measures actual GPU execution time via device_synchronize pre+post —
        // costs latency vs async pipelining but gives shape-accurate µs.
        static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let dump = *DUMP.get_or_init(|| self.flags.gemm_dump);
        if dump {
            self.hip.device_synchronize()?;
        }
        let dump_start = if dump {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles as u32, batch_tiles as u32, k_splits],
            [block_size, 1, 1],
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
        if let Some(t) = dump_start {
            self.hip.device_synchronize()?;
            let us = t.elapsed().as_micros();
            let gbs = (bytes as f64) / (us.max(1) as f64) / 1000.0; // MB/ms == GB/s
            eprintln!(
                "[gemm-dump] {} M={} K={} B={} bytes={}KB us={} GB/s={:.1}",
                kernel_name,
                m,
                k,
                batch_size,
                bytes / 1024,
                us,
                gbs
            );
        }
        result
    }

    /// Deterministic K-split residual WMMA GEMM. Same K-split occupancy win
    /// as `ksplit` (grid.z = K_SPLITS = 4 → ~13 blocks/CU on gfx1100), but
    /// race-free: phase 1 writes each split's partial to its own scratch
    /// slice (plain store, no atomicAdd, no residual); phase 2 sums the
    /// K_SPLITS partials + residual into Y in fixed index order. Output is
    /// bit-reproducible across runs/processes. Caller must initialize Y with
    /// the residual stream before launching (same contract as ksplit/k2).
    pub fn gemm_hfq4g256_residual_wmma_ksplit_det(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const K_SPLITS: u32 = 4;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_wmma_ksplit_det",
            kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_KSPLIT_DET_SRC,
            "gemm_hfq4g256_residual_wmma_ksplit_det",
        )?;
        self.ensure_kernel(
            "gemm_ksplit_det_finalize",
            kernels::GEMM_KSPLIT_DET_FINALIZE_SRC,
            "gemm_ksplit_det_finalize",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        // Partials scratch: [K_SPLITS][batch_size][M] fp32.
        let n_cells = batch_size * m;
        let partials_ptr = self.ensure_ksplit_det_partials(K_SPLITS as usize * n_cells * 4)?;

        // ── Phase 1: per-split partials (plain store, no atomic) ──
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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
            "gemm_hfq4g256_residual_wmma_ksplit_det",
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

        // ── Phase 2: fixed-order finalize (residual + partials → Y) ──
        // Pass batch_size + m (not a pre-multiplied cell count): the kernel
        // computes the z-stride as batch_size*M in long long, matching the
        // partial kernel's split_base literal and dodging any i32 overflow.
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
            "gemm_ksplit_det_finalize",
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
        // Timer spans BOTH phases — finalize GPU time is included in the
        // ksplit_det perf accounting (else it would undercount vs ksplit).
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        r
    }

    /// HFQ3-G256 sister of `gemm_hfq4g256_residual_wmma` (basic WMMA
    /// variant). Same WMMA shape + lane decomposition; only the inner
    /// K-tile unpack differs (3-bit cross-byte vs 4-bit nibble) and the
    /// per-group byte stride is 104 instead of 136. Y += acc[j] (fused
    /// residual add — caller must initialize Y with the residual stream
    /// before launching).
    pub fn gemm_hfq3g256_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let arch_supports_mb4 = self.arch_caps.supports_mq3_lloyd_mb4();
        let use_mb4 = match self.flags.mq3_mb4 {
            Some(_) => arch_supports_mb4,
            None => arch_supports_mb4 && batch_size >= 128 && m >= 4096,
        };
        if use_mb4 {
            return self.gemm_hfq3g256_residual_wmma_mb4(a_raw, x, y, m, k, batch_size);
        }
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.gemm_hfq3g256_residual_wmma_gfx12(a_raw, x, y, m, k, batch_size);
        }
        self.ensure_kernel(
            "gemm_hfq3g256_residual_wmma",
            kernels::GEMM_HFQ3G256_RESIDUAL_WMMA_SRC,
            "gemm_hfq3g256_residual_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = m * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq3g256_residual_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_hfq3g256_residual_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
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

    /// HFQ3 residual mb4 dispatch: 16×64 output tile per WG.
    pub fn gemm_hfq3g256_residual_wmma_mb4(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq3g256_residual_wmma_mb4",
            kernels::GEMM_HFQ3G256_RESIDUAL_WMMA_MB4_SRC,
            "gemm_hfq3g256_residual_wmma_mb4",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 63) / 64;
        let bytes = m * (k / 256) * 104 + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq3g256_residual_wmma_mb4",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_hfq3g256_residual_wmma_mb4",
            [row_tiles as u32, batch_tiles as u32, 1],
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

    /// MQ3 wrapper for `gemm_hfq3g256_residual_wmma`: pre-rotates X then
    /// dispatches the HFQ3 kernel. See `gemm_qkvza_mq3g256_wmma` for
    /// the cache-invalidation rationale.
    /// gfx12 (RDNA4) sister of `gemm_hfq3g256_residual_wmma`.
    pub fn gemm_hfq3g256_residual_wmma_gfx12(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq3g256_residual_wmma_gfx12",
            kernels::GEMM_HFQ3G256_RESIDUAL_WMMA_GFX12_SRC,
            "gemm_hfq3g256_residual_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let weight_bytes = m * (k / 256) * 104;
        let bytes = weight_bytes + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq3g256_residual_wmma_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_hfq3g256_residual_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
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

    pub fn gemm_mq3g256_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        for b in 0..batch_size {
            let x_row = x.sub_offset(b * k, k);
            let x_rot_row = x_rot.sub_offset(b * k, k);
            self.rotate_x_mq(&x_row, &x_rot_row, k)?;
        }
        self.scratch.fp16_x_source_ptr = std::ptr::null_mut();
        self.gemm_hfq3g256_residual_wmma(a_raw, x_rot, y, m, k, batch_size)
    }

    /// MW16: dequant 4-bit weights to FP16, then run the no-dequant WMMA kernel.
    /// Per-call dequant (wasteful) — for benchmarking only. Production would
    /// dequant at model load time.
    fn gemm_mw16_residual_wmma_via_dequant(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(
            "dequant_hfq4g256_to_f16",
            kernels::DEQUANT_HFQ4G256_TO_F16_SRC,
            "dequant_hfq4g256_to_f16",
        )?;
        self.ensure_kernel(
            "gemm_mw16_residual_wmma",
            kernels::GEMM_MW16_RESIDUAL_WMMA_SRC,
            "gemm_mw16_residual_wmma",
        )?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * k)?;

        // Dequant weights to FP16 scratch
        let w_elems = m * k;
        let w_f16 = self.hip.malloc(w_elems * 2)?;
        {
            let f = &self.functions["dequant_hfq4g256_to_f16"];
            let groups = k / 256;
            let mut ap = a_raw.buf.as_ptr();
            let mut wp = w_f16.as_ptr();
            let mut mv = m as i32;
            let mut kv = k as i32;
            let mut p: Vec<*mut c_void> = vec![
                &mut ap as *mut _ as *mut c_void,
                &mut wp as *mut _ as *mut c_void,
                &mut mv as *mut _ as *mut c_void,
                &mut kv as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    f,
                    [m as u32, groups as u32, 1],
                    [32, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut p,
                )?;
            }
        }

        // MW16 WMMA GEMM
        let f = &self.functions["gemm_mw16_residual_wmma"];
        let mut wp = w_f16.as_ptr();
        let mut xp = x_f16;
        let mut yp = y.buf.as_ptr();
        let mut mv = m as i32;
        let mut kv = k as i32;
        let mut nv = batch_size as i32;
        let mut p: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
        ];
        let rows = (m + 15) / 16;
        let batches = (batch_size + 15) / 16;
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * m * 8;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_mw16_residual_wmma", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                f,
                [rows as u32, batches as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut p,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        drop(w_f16);
        result
    }

    /// Batched HFQ4-G256 GEMM: y[b][row] = A[row] · x[b] for all batch elements.
    /// x: [batch_size × K], y: [batch_size × M], both row-major.
    ///
    /// This is the portable scalar kernel — stays byte-exact with the AR
    /// greedy prefill's numerical baseline. For the DFlash lm_head fast
    /// path (batched, tolerates small FP16 drift for 8-10× speedup), use
    /// `gemm_hfq4g256_batched_lmhead` instead.
    pub fn gemm_hfq4g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx906 dp4a opt-in for the LM-head batched GEMM. PMC at 2026-05-06
        // showed gemm_hfq4g256_wave64 was 17 % of DFlash 27B steady-state
        // decode time on the FP wave64 path. The dp4a port pre-quantizes x
        // to Q8_1 (shared scratch with the prefill MMQ + the gate_up/qkv/qkvza
        // GEMV ports) and runs v_dot4_i32_i8.
        //
        // Only fires on gfx906 (other wave64-native archs have rocBLAS or
        // larger MFMA paths that beat dp4a at large batches). Skip in
        // capture mode (matches the rocBLAS branch's caveat — Q8_1
        // quantize launch must be reachable from the captured graph or
        // pre-baked).
        if self.arch_caps.gemv_dp4a_enabled() && !self.graphs.capture_mode {
            return self.gemm_hfq4g256_dp4a(a_raw, x, y, m, k, batch_size);
        }

        // CDNA3 MFMA path (task #130): when rocBLAS is loaded and batch is
        // big enough for the launch overhead to amortize, route through the
        // dequantize-once FP16 shadow + rocBLAS GEMM. Expected 20-100× over
        // the wave64 GEMV on prefill-heavy workloads (sidecar cal, DFlash
        // target verify). Falls back to wave64 GEMV on: single-token decode
        // (batch<4), capture mode (rocBLAS launches don't graph-capture
        // cleanly; revisit if hipGraph becomes critical for CDNA3 prefill),
        // or if the fp16 shadow alloc fails under VRAM pressure.
        if self.rocblas_arch_eligible()
            && batch_size >= self.rocblas_min_batch()
            && self.rocblas.is_some()
            && !self.graphs.capture_mode
        {
            if let Ok(Some(shadow_ptr)) = self.ensure_fp16_shadow(a_raw, m, k) {
                // Convert X to FP16 via the existing ensure_fp16_x helper.
                let x_fp16 = self.ensure_fp16_x(x, batch_size * k)?;
                // Wrap the raw device pointers as non-owning DeviceBuffers so
                // the rocBLAS helper's signature works. The underlying memory
                // is owned by the fp16 shadow cache / fp16_x_scratch / caller's
                // y GpuTensor — all live beyond this call.
                let w_buf = unsafe { DeviceBuffer::from_raw(shadow_ptr, (m * k) * 2) };
                let x_buf = unsafe { DeviceBuffer::from_raw(x_fp16, (batch_size * k) * 2) };
                let bytes = crate::profile::gemm_hfq4g256_bytes(m, k, batch_size);
                let timer =
                    crate::profile::begin_timer(&self.hip, "gemv", "gemm_hfq4g256_rocblas", bytes);
                let result =
                    self.rocblas_gemm_hfq4_prefill(&w_buf, &x_buf, &y.buf, m, batch_size, k);
                // Suppress the non-owning DeviceBuffer drop; HipError::Drop on
                // hip_free would clobber memory we don't own.
                std::mem::forget(w_buf);
                std::mem::forget(x_buf);
                if let Some(t) = timer {
                    t.finish(&self.hip);
                }
                return result;
            }
            // Shadow allocation failed — fall through to the GEMV path.
        }

        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_div): (&str, [u32; 3], u32) = if cdna_wave64 {
            self.ensure_kernel(
                "gemm_hfq4g256_wave64",
                kernels::GEMM_HFQ4G256_WAVE64_SRC,
                "gemm_hfq4g256_wave64",
            )?;
            ("gemm_hfq4g256_wave64", [64, 1, 1], 2)
        } else {
            self.ensure_kernel("gemm_hfq4g256", kernels::GEMM_HFQ4G256_SRC, "gemm_hfq4g256")?;
            ("gemm_hfq4g256", [32, 1, 1], 1)
        };

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            ((batch_size + BATCH_TILE - 1) / BATCH_TILE) as u32
        };
        let grid_x = (m as u32 + grid_div - 1) / grid_div;
        let bytes = crate::profile::gemm_hfq4g256_bytes(m, k, batch_size);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemm_hfq4g256", bytes);
        let result = self.launch_maybe_blob(
            func_name,
            [grid_x, batch_tiles, 1],
            block,
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

    /// dp4a-port of gemm_hfq4g256 for gfx906. Pre-quantizes x to Q8_1 via
    /// the shared MMQ x-scratch (kblock-major: `[K/128, batch_size]`),
    /// then dispatches the wave64 dp4a GEMM. Math is identical modulo
    /// Q8_1 quant noise.
    ///
    /// Targets the LM-head batched GEMM hot path on DFlash 27B (PMC at
    /// 2026-05-06 showed 17 % of decode time was here on the FP path).
    /// Same Q8_1 layout as the prefill MMQ kernel + the four PR-158
    /// fused GEMVs, so `ensure_q8_1_mmq_x` reuses the existing scratch.
    pub fn gemm_hfq4g256_dp4a(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Quantize x → Xq[K/128 * batch_size] block_q8_1_mmq via the
        // shared scratch. Stride layout: kblock-major (matches
        // quantize_q8_1_mmq_ds4 at gemm_hfq4g256_residual_mmq.hip:80).
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        self.ensure_kernel(
            "gemm_hfq4g256_wave64_dp4a",
            kernels::GEMM_HFQ4G256_WAVE64_DP4A_SRC,
            "gemm_hfq4g256_wave64_dp4a",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;
        let grid_x = (m as u32 + 1) / 2;
        const BATCH_TILE: usize = 8;
        let grid_y = ((batch_size + BATCH_TILE - 1) / BATCH_TILE) as u32;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        let bytes = crate::profile::gemm_hfq4g256_bytes(m, k, batch_size);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemm_hfq4g256_dp4a", bytes);
        let result = self.launch_maybe_blob(
            "gemm_hfq4g256_wave64_dp4a",
            [grid_x, grid_y, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq);
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

    /// Batched HFQ4-G256 residual GEMM with fused dp4a inner loop on gfx906.
    /// HFQ4 sibling of `gemm_hfq6g256_residual_wave64_dp4a` (HFQ6 Phase A.2,
    /// commit 1b9f3747 → merged via #187). Closes the dispatch gap where MQ4
    /// at gfx906 B>1 below the MMQ cutover (B ∈ [2, 7] per `should_use_mmq`'s
    /// gfx906 default) falls to `gemm_hfq4g256_residual_fp16_wave64`; the
    /// dp4a path wins on per-call ALU (sdot4 issues 4 int8 mul + 4 acc-add
    /// per cycle, vs FP wave64 hybrid's hfma2 at 2 mul + 2 add per cycle →
    /// ~2× FLOPs/cycle) and reuses the existing Q8_1 activation scratch
    /// (shared with HFQ4 MMQ + the GEMV-shape fused dp4a kernels).
    ///
    /// Issue #276 Gap 2. Ships with `BATCH_TILE = 16` from the start per the
    /// HFQ6 Phase B.1.1 measurement (commit ff9e2105: BT=8→16 halves A-reload
    /// trips per row, +7-17% per-call on the structurally identical HFQ6
    /// sibling). MUST stay in sync with the kernel's `#define BATCH_TILE 16`
    /// at `kernels/src/gemm_hfq4g256_residual_wave64_dp4a.hip:38`.
    pub fn gemm_hfq4g256_residual_wave64_dp4a(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        self.gemm_hfq4g256_residual_wave64_dp4a_prequant(a_raw, xq_ptr, y, m, k, batch_size)
    }

    /// Prequant entry point: caller has already populated the Q8_1 scratch
    /// (see `ensure_q8_1_mmq_x`). Skips the Q8_1 conversion. Use when X has
    /// just been quantized for a sibling kernel (e.g. MMQ split or fused
    /// QKVZA tail) to avoid a redundant ~k·batch_size byte memset+convert.
    pub fn gemm_hfq4g256_residual_wave64_dp4a_prequant(
        &mut self,
        a_raw: &GpuTensor,
        xq_ptr: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_residual_wave64_dp4a",
            kernels::GEMM_HFQ4G256_RESIDUAL_WAVE64_DP4A_SRC,
            "gemm_hfq4g256_residual_wave64_dp4a",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        // BATCH_TILE MUST match the kernel's `#define BATCH_TILE 16`.
        const BATCH_TILE: usize = 16;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let grid_x = ((m as u32) + 1) / 2;

        // bytes = weight (HFQ4: 136 B/group, 0.53 B/weight) + Q8_1 X scratch
        // (~33 B per Q8_1 block of 32 K-elems = ~1.03 B/element, but for
        // bandwidth accounting use the dominant int8 qs term: batch*k bytes)
        // + Y read+write (residual: batch*m*4 each way).
        let bytes =
            crate::profile::hfq4g256_weight_bytes(m, k) + batch_size * k + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_residual_wave64_dp4a",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_hfq4g256_residual_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq);
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

    /// Batched HFQ4-G256 fused 4-way QKVZA (qkv + z + beta + alpha) GEMM
    /// with dp4a inner loop on gfx906. HFQ4 sibling of
    /// `gemm_qkvza_hfq6g256_wave64_dp4a` (HFQ6 Phase A.3, merged via #187).
    /// Closes the dispatch fallthrough where MQ4 at gfx906 batched DeltaNet
    /// preamble (B>1) drops to `gemm_qkvza_hfq4g256_fp16_wave64`. Issue #276
    /// Gap 2 part 2. Uses `BATCH_TILE=16` matching the kernel's `#define`.
    pub fn gemm_qkvza_hfq4g256_wave64_dp4a(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        self.gemm_qkvza_hfq4g256_wave64_dp4a_prequant(
            a_qkv, a_z, a_beta, a_alpha, xq_ptr, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
            alpha_m, k, batch_size,
        )
    }

    /// Prequant entry point: caller has already populated the Q8_1 scratch.
    /// Skips the Q8_1 conversion. Use when X has just been quantized for a
    /// sibling kernel (e.g. the MMQ-split qkv+z path's beta+alpha tail) to
    /// avoid a redundant FP32→Q8_1 conversion of the entire X tensor.
    pub fn gemm_qkvza_hfq4g256_wave64_dp4a_prequant(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        xq_ptr: *mut c_void,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq4g256_wave64_dp4a",
            kernels::GEMM_QKVZA_HFQ4G256_WAVE64_DP4A_SRC,
            "gemm_qkvza_hfq4g256_wave64_dp4a",
        )?;

        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let qkv_m_val = qkv_m as i32;
        let z_m_val = z_m as i32;
        let beta_m_val = beta_m as i32;
        let alpha_m_val = alpha_m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &az as *const _ as *mut c_void,
            &ab as *const _ as *mut c_void,
            &aa as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yz as *const _ as *mut c_void,
            &yb as *const _ as *mut c_void,
            &ya as *const _ as *mut c_void,
            &qkv_m_val as *const _ as *mut c_void,
            &z_m_val as *const _ as *mut c_void,
            &beta_m_val as *const _ as *mut c_void,
            &alpha_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        // BATCH_TILE MUST match the kernel's `#define BATCH_TILE 16`.
        const BATCH_TILE: usize = 16;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let grid_x = (total_m + 1) / 2;

        // bytes = weight (4 matrices, 136 B/group each) + Q8_1 X read +
        // 4× Y writes (overwrite semantic, no read).
        let bytes = crate::profile::hfq4g256_weight_bytes(qkv_m, k)
            + crate::profile::hfq4g256_weight_bytes(z_m, k)
            + crate::profile::hfq4g256_weight_bytes(beta_m, k)
            + crate::profile::hfq4g256_weight_bytes(alpha_m, k)
            + batch_size * k
            + batch_size * (qkv_m + z_m + beta_m + alpha_m) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_qkvza_hfq4g256_wave64_dp4a",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq4g256_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xq);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(qkv_m_val);
                b.push_i32(z_m_val);
                b.push_i32(beta_m_val);
                b.push_i32(alpha_m_val);
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

    /// Batched HFQ4-G256 fused 3-way QKV GEMM with dp4a inner loop on
    /// gfx906. HFQ4 sibling of `gemm_qkv_hfq6g256_wave64_dp4a`. Closes the
    /// dispatch fallthrough where MQ4 at gfx906 batched FullAttention
    /// preamble drops to `gemm_qkv_hfq4g256_fp16_wave64`. Issue #276 Gap 2.
    pub fn gemm_qkv_hfq4g256_wave64_dp4a(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        self.gemm_qkv_hfq4g256_wave64_dp4a_prequant(
            a_q, a_k, a_v, xq_ptr, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
        )
    }

    /// Prequant entry point — see `gemm_qkvza_hfq4g256_wave64_dp4a_prequant`
    /// for rationale. Skips the FP32→Q8_1 conversion of X; caller must have
    /// populated the Q8_1 scratch beforehand.
    pub fn gemm_qkv_hfq4g256_wave64_dp4a_prequant(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        xq_ptr: *mut c_void,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq4g256_wave64_dp4a",
            kernels::GEMM_QKV_HFQ4G256_WAVE64_DP4A_SRC,
            "gemm_qkv_hfq4g256_wave64_dp4a",
        )?;

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_val = q_m as i32;
        let k_m_val = k_m as i32;
        let v_m_val = v_m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &ak as *const _ as *mut c_void,
            &av as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yk as *const _ as *mut c_void,
            &yv as *const _ as *mut c_void,
            &q_m_val as *const _ as *mut c_void,
            &k_m_val as *const _ as *mut c_void,
            &v_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 16;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (q_m + k_m + v_m) as u32;
        let grid_x = (total_m + 1) / 2;

        let bytes = crate::profile::hfq4g256_weight_bytes(q_m, k)
            + crate::profile::hfq4g256_weight_bytes(k_m, k)
            + crate::profile::hfq4g256_weight_bytes(v_m, k)
            + batch_size * k
            + batch_size * (q_m + k_m + v_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256_wave64_dp4a", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq4g256_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xq);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
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

    /// Batched HFQ4-G256 fused 2-way gate_up GEMM with dp4a inner loop on
    /// gfx906. HFQ4 sibling of `gemm_gate_up_hfq6g256_wave64_dp4a`. Closes
    /// the dispatch fallthrough where MQ4 at gfx906 batched FFN preamble
    /// drops to `gemm_gate_up_hfq4g256_fp16_wave64`. Issue #276 Gap 2.
    pub fn gemm_gate_up_hfq4g256_wave64_dp4a(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        self.gemm_gate_up_hfq4g256_wave64_dp4a_prequant(
            a_gate, a_up, xq_ptr, y_gate, y_up, gate_m, up_m, k, batch_size,
        )
    }

    /// Prequant entry point — see `gemm_qkvza_hfq4g256_wave64_dp4a_prequant`
    /// for rationale.
    pub fn gemm_gate_up_hfq4g256_wave64_dp4a_prequant(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        xq_ptr: *mut c_void,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq4g256_wave64_dp4a",
            kernels::GEMM_GATE_UP_HFQ4G256_WAVE64_DP4A_SRC,
            "gemm_gate_up_hfq4g256_wave64_dp4a",
        )?;

        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let gate_m_val = gate_m as i32;
        let up_m_val = up_m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &ag as *const _ as *mut c_void,
            &au as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yg as *const _ as *mut c_void,
            &yu as *const _ as *mut c_void,
            &gate_m_val as *const _ as *mut c_void,
            &up_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 16;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (gate_m + up_m) as u32;
        let grid_x = (total_m + 1) / 2;

        let bytes = crate::profile::hfq4g256_weight_bytes(gate_m, k)
            + crate::profile::hfq4g256_weight_bytes(up_m, k)
            + batch_size * k
            + batch_size * (gate_m + up_m) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_hfq4g256_wave64_dp4a",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq4g256_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xq);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(gate_m_val);
                b.push_i32(up_m_val);
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

    /// FP16-weight lm_head fast path for DFlash drafts that ship F16 (not
    /// quantized) weights. Routes through `gemm_mw16_residual_wmma` with the
    /// usual memset-then-atomicAdd residual pattern.
    ///
    /// Shape requirements: K must be a multiple of 32 (mw16 processes 32 K
    /// elements per WMMA iteration). All 27B/9B draft shapes satisfy this
    /// (hidden=5120, intermediate=17408, q_dim=4096, kv_dim=1024, fc-K=25600).
    ///
    /// Non-gfx11 falls through to row-by-row F16 GEMM so lm_head output keeps
    /// the `[batch, vocab]` layout expected by callers. Set
    /// `HIPFIRE_LM_HEAD_F16=f32` at load time to use the legacy F32-expanded
    /// storage and bypass this path entirely.
    pub fn gemm_f16_batched_lmhead(
        &mut self,
        w_f16: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.has_wmma_w32_gfx12() {
            // RDNA4 (gfx12): the mw16 WMMA kernel below is gfx11-builtin-only
            // (has_wmma_w32 = is_rdna3). Route through the validated gfx12
            // f16 WMMA kernel instead: same contract (F32 X, Y[batch, m]).
            // (Historically load-bearing for correctness too: the legacy
            // gemm_f16_tiled fallback miscomputed ~10x before its 2026-06
            // rewrite and corrupted the ds4 DSA compressor projections on
            // gfx1201, breaking EP tool-calling. The kernel is fixed now,
            // but WMMA remains the right path here for perf.)
            return self.gemm_f16_wmma_mb8(w_f16, x, y, m, k, batch_size);
        }
        if !self.arch_caps.has_wmma_w32() {
            // No mw16 WMMA on non-RDNA3. The generic F16 GEMM writes [M,N],
            // while lm_head consumers expect [N,M], so preserve layout by
            // launching one row at a time. gemm_f16_tiled covers K exactly
            // once per lane (disjoint stride-32 partition + tail loop; see
            // f16_gemv_parity for the cross-shape parity matrix).
            for b in 0..batch_size {
                let x_row = x.sub_offset(b * k, k);
                let y_row = y.sub_offset(b * m, m);
                self.gemm_f16_tiled(w_f16, &x_row, &y_row, m, k, 1)?;
            }
            return Ok(());
        }
        self.ensure_kernel(
            "gemm_mw16_residual_wmma",
            kernels::GEMM_MW16_RESIDUAL_WMMA_SRC,
            "gemm_mw16_residual_wmma",
        )?;
        // Pre-zero Y (residual WMMA does y += acc) and force FP16-X reconversion
        // (the draft reuses the same scratch pointer every cycle with new data).
        self.scratch.fp16_x_source_ptr = std::ptr::null_mut();
        match self.active_stream.as_ref() {
            Some(stream) => self
                .hip
                .memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
            None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
        }
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let wp = w_f16.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let ni = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mi as *const _ as *mut c_void,
            &ki as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        let rows = ((m + 15) / 16) as u32;
        let batches = ((batch_size + 15) / 16) as u32;
        // Bytes: FP16 weight + FP16 x + FP32 y (read+write).
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_mw16_residual_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_mw16_residual_wmma",
            [rows, batches, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(wp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mi);
                b.push_i32(ki);
                b.push_i32(ni);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// K2-pipelined WMMA GEMM: F16 weight × pre-converted F16 input → F32 output.
    ///
    /// Like `gemm_f16_batched_lmhead` (RDNA3 arm) but takes a caller-owned F16
    /// `x_f16` buffer instead of an F32 `x` that goes through `ensure_fp16_x`.
    /// This avoids polluting the shared `fp16_x_source_ptr` cache, which is
    /// critical when DSpark's `run_heads` runs immediately before the trunk's
    /// verify pass (both use `ensure_fp16_x` — cache poisoning by DSpark would
    /// force 36-layer × 2+ reconversions per verify window, +18ms regression).
    ///
    /// Pre-zeros `y` so the residual `+=` in `gemm_mw16_residual_wmma` collapses
    /// to `=` semantics (same as `gemm_f16_batched_lmhead`).
    /// Requires `has_wmma_w32()` (RDNA3/RDNA3.5). Falls back to
    /// `gemm_f16_x_f16_wmma` on non-WMMA arches (should not occur in practice).
    ///
    /// K must be a multiple of 32 (K2 loop step).
    pub fn gemm_mw16_residual_wmma_f16x(
        &mut self,
        w_f16: &GpuTensor,
        x_f16: &GpuTensor, // [batch_size, k] F16, caller-owned
        y: &GpuTensor,     // [batch_size, m] F32, output
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Pre-zero Y: gemm_mw16_residual_wmma does Y += acc.
        match self.active_stream.as_ref() {
            Some(stream) => self
                .hip
                .memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
            None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
        }
        // Ensure kernel is compiled.
        self.ensure_kernel(
            "gemm_mw16_residual_wmma",
            kernels::GEMM_MW16_RESIDUAL_WMMA_SRC,
            "gemm_mw16_residual_wmma",
        )?;
        let wp = w_f16.buf.as_ptr();
        let xp = x_f16.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let ni = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mi as *const _ as *mut c_void,
            &ki as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        let rows = ((m + 15) / 16) as u32;
        let batches = ((batch_size + 15) / 16) as u32;
        let bytes = m * k * 2 + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_mw16_residual_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_mw16_residual_wmma",
            [rows, batches, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(wp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mi);
                b.push_i32(ki);
                b.push_i32(ni);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// WMMA lm_head fast path for DFlash. Computes y = A @ x at batch>1 via
    /// the residual-WMMA kernel on pre-zeroed y — 8-10× faster than the
    /// scalar `gemm_hfq4g256` on 9B lm_head (batch=16, vocab=248K, k=2560).
    ///
    /// NOT numerically identical to `gemm_hfq4g256`. Uses FP16 tensor cores
    /// with the accumulators in FP32 the residual kernel ships. On the
    /// DFlash target-verify + draft-lm_head hot path this is a win (~13 ms
    /// saved per cycle), and the small FP16 drift doesn't meaningfully
    /// affect greedy acceptance. Do NOT use for AR greedy prefill — it will
    /// break byte-exact quality-gate reproducibility.
    ///
    /// Fallbacks: non-gfx11 or HIPFIRE_FP16=0 or HIPFIRE_LM_HEAD_WMMA=0 →
    /// routes to plain `gemm_hfq4g256`.
    ///
    /// Subtle: the residual-WMMA kernel goes through `ensure_fp16_x`, which
    /// caches the FP32→FP16 conversion keyed on source pointer. DFlash
    /// callers reuse the SAME hidden buffer pointer every cycle (draft
    /// scratch sub-offset, verify's persistent final_hidden) but with NEW
    /// data — so the cache entry is silently stale. Stomp the cache pointer
    /// before the dispatch to force reconversion.
    pub fn gemm_hfq4g256_batched_lmhead(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Gate covers gfx11 (RDNA3) AND gfx12 (RDNA4). The gfx12 family
        // ships its own residual_wmma kernel sibling
        // (gemm_hfq4g256_residual_wmma_gfx12); without this dispatch,
        // gfx12 falls through to the scalar `gemm_hfq4g256` and pays the
        // 8-10× per-call penalty (rocprof on R9700 / gfx1201 measured
        // ~26.68% of composition cycle wall in this scalar path).
        let arch = self.arch.as_str();
        let wmma_eligible = batch_size > 1
            && (self.arch_caps.has_wmma_w32() || self.arch_caps.has_wmma_w32_gfx12())
            && !self.flags.fp16_disabled
            && !self.flags.lm_head_wmma_disabled;
        if wmma_eligible {
            self.scratch.fp16_x_source_ptr = std::ptr::null_mut();
            match self.active_stream.as_ref() {
                Some(stream) => self
                    .hip
                    .memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
                None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
            }
            return if arch.starts_with("gfx12") {
                self.gemm_hfq4g256_residual_wmma_gfx12(a_raw, x, y, m, k, batch_size)
            } else {
                self.gemm_hfq4g256_residual_wmma(a_raw, x, y, m, k, batch_size)
            };
        }
        self.gemm_hfq4g256(a_raw, x, y, m, k, batch_size)
    }

    /// HFQ6-G256 sister of `gemm_hfq4g256_batched_lmhead`. Phase A.4
    /// (plan v3.2.3 §5.1 item 4). On gfx906 uses the dp4a residual GEMM
    /// (Phase A.2) with a zero-init of Y, mirroring the HFQ4 WMMA pattern
    /// at line 8019-8022. Lets the residual `+=` collapse to `=` semantics
    /// without needing a separate non-residual kernel.
    ///
    /// Caller is responsible for FWHT-rotating x first when the weights
    /// are MQ6 (FWHT-rotated at quant time).
    pub fn gemm_hfq6g256_batched_lmhead(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx906: dp4a residual + zero-init Y for `=` semantics.
        // Skip in capture mode (the residual kernel calls ensure_q8_1_mmq_x
        // which launches an internal quantize kernel — matches HFQ4 sibling).
        if batch_size > 1 && self.arch_caps.gemv_dp4a_enabled() && !self.graphs.capture_mode {
            match self.active_stream.as_ref() {
                Some(stream) => self
                    .hip
                    .memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
                None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
            }
            return self.gemm_hfq6g256_residual_wave64_dp4a(a_raw, x, y, m, k, batch_size);
        }
        // gfx11+ AND gfx12: WMMA residual + zero-init. Symmetric to the
        // HFQ4 fix (commit 48dd8ba4) — gfx12 sibling kernel already ships
        // (gemm_hfq6g256_residual_wmma_gfx12, see line ~15431 dispatch);
        // this wrapper just needed the gate widened.
        let arch_str = self.arch.as_str();
        let wmma_eligible = batch_size > 1
            && (self.arch_caps.has_wmma_w32() || self.arch_caps.has_wmma_w32_gfx12())
            && !self.flags.fp16_disabled
            && !self.flags.lm_head_wmma_disabled;
        if wmma_eligible {
            self.scratch.fp16_x_source_ptr = std::ptr::null_mut();
            match self.active_stream.as_ref() {
                Some(stream) => self
                    .hip
                    .memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
                None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
            }
            return if arch_str.starts_with("gfx12") {
                self.gemm_hfq6g256_residual_wmma_gfx12(a_raw, x, y, m, k, batch_size)
            } else {
                self.gemm_hfq6g256_residual_wmma(a_raw, x, y, m, k, batch_size)
            };
        }
        // Fallback: use the residual dispatcher with zero-init Y. This
        // routes to fp16-packed or scalar depending on arch.
        match self.active_stream.as_ref() {
            Some(stream) => self
                .hip
                .memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
            None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
        }
        self.gemm_hfq6g256_residual(a_raw, x, y, m, k, batch_size)
    }

    /// HFQ3-G256 sister of `gemm_hfq4g256_batched_lmhead`. Same FP16-X cache
    /// stomp + zero-init of Y, then `gemm_hfq3g256_residual_wmma` to compute
    /// y[b][row] = A[row] · x[b]. Used by `dflash::gemm_dispatch` for MQ3
    /// drafts so DFlash works with MQ3-quantized draft weights.
    ///
    /// Caller is responsible for FWHT-rotating x first when the weights are
    /// MQ3 (FWHT-rotated at quant time) — `dflash::gemm_dispatch` handles
    /// that via `rotate_x_mq_batched`. This wrapper is dtype-agnostic in
    /// the same sense as `gemm_hfq4g256_batched_lmhead`.
    pub fn gemm_hfq3g256_batched_lmhead(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // WMMA eligibility: any arch with an MQ3 WMMA family ported. Today
        // that's gfx11 (RDNA3, _w32 builtin) and gfx12 (RDNA4, _w32_gfx12
        // builtin) — `gemm_hfq3g256_residual_wmma` dispatches internally to
        // the correct variant per arch. Other archs (gfx10/906/94x) fall
        // through to the per-row GEMV path.
        let wmma_eligible = batch_size > 1
            && (self.arch_caps.has_wmma_w32() || self.arch_caps.has_wmma_w32_gfx12())
            && !self.flags.fp16_disabled
            && !self.flags.lm_head_wmma_disabled;
        if wmma_eligible {
            self.scratch.fp16_x_source_ptr = std::ptr::null_mut();
            match self.active_stream.as_ref() {
                Some(stream) => self
                    .hip
                    .memset_async(&y.buf, 0, batch_size * m * 4, stream)?,
                None => self.hip.memset(&y.buf, 0, batch_size * m * 4)?,
            }
            return self.gemm_hfq3g256_residual_wmma(a_raw, x, y, m, k, batch_size);
        }
        // Non-WMMA fallback: per-batch GEMV. Slow but functional. DFlash on
        // non-gfx11/gfx12 archs is already gated upstream by the daemon's
        // DFlash refusal guard (lm_head whitelist requires gfx11 or gfx12
        // for MQ3) — this fallback is reachable only via direct callers
        // that bypass the daemon (e.g., bench harnesses, channel tests).
        for b in 0..batch_size {
            let x_row = x.sub_offset(b * k, k);
            let y_row = y.sub_offset(b * m, m);
            self.gemv_hfq3g256(a_raw, &x_row, &y_row, m, k)?;
        }
        Ok(())
    }

    // ========================================================================
    // HFQ6-G256 GEMM variants (residual, fused)
    // ========================================================================

    /// gfx906 wave64+dp4a batched residual GEMM for HFQ6/MQ6.
    /// Phase A.2 (plan v3.2.3 §5.1 item 2). Pre-quantizes x to Q8_1 and
    /// dispatches the dp4a kernel; output is residual `+=` semantics.
    ///
    /// Math identity: same as the fused-GEMV dp4a kernels (plan §2.2
    /// Option A — HFQ6 unsigned weights, no zp shift correction).
    pub fn gemm_hfq6g256_residual_wave64_dp4a(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        self.ensure_kernel(
            "gemm_hfq6g256_residual_wave64_dp4a",
            kernels::GEMM_HFQ6G256_RESIDUAL_WAVE64_DP4A_SRC,
            "gemm_hfq6g256_residual_wave64_dp4a",
        )?;

        let a_ptr = a_raw.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 8;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let grid_x = ((m as u32) + 1) / 2;

        self.launch_maybe_blob(
            "gemm_hfq6g256_residual_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(xq);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(bs_val);
                b
            },
        )
    }

    /// Batched HFQ6-G256 GEMM with fused residual add:
    ///   for b in 0..batch_size: y[b][row] += A[row] · x[b]
    ///
    /// Auto-selects: gfx11 -> WMMA, gfx906 -> dp4a (Phase A.2),
    /// else -> FP16 packed, fallback -> FP32 scalar.
    pub fn gemm_hfq6g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !self.flags.fp16_disabled {
            // WMMA on gfx12 (RDNA4): _w32_gfx12 builtin (gfx11 builtin
            // does NOT pattern-match on gfx12 — see has_wmma_f16 comment).
            if self.arch_caps.has_wmma_w32_gfx12() {
                return self.gemm_hfq6g256_residual_wmma_gfx12(a_raw, x, y, m, k, batch_size);
            }
            // WMMA on gfx11+ (RDNA3): 16x16 tiled
            if self.arch_caps.has_wmma_w32() {
                return self.gemm_hfq6g256_residual_wmma(a_raw, x, y, m, k, batch_size);
            }
            // gfx906: dp4a + wave64 batched residual (Phase A.2, plan v3.2.3
            // §5.1 item 2). Pre-quantize x to Q8_1 and dispatch the dp4a
            // kernel. Mirror of the HFQ4 sibling pattern at gemm_hfq4g256_wave64_dp4a.
            // Skip in capture mode: ensure_q8_1_mmq_x launches an internal
            // quantize kernel that the captured graph may not record (matches
            // gemm_hfq4g256_dp4a's `&& !self.graphs.capture_mode` guard at line ~7889).
            if self.arch_caps.gemv_dp4a_enabled() && !self.graphs.capture_mode {
                return self.gemm_hfq6g256_residual_wave64_dp4a(a_raw, x, y, m, k, batch_size);
            }
            // FP16 packed on all other RDNA
            return self.gemm_hfq6g256_residual_fp16(a_raw, x, y, m, k, batch_size);
        }
        self.ensure_kernel(
            "gemm_hfq6g256_residual",
            kernels::GEMM_HFQ6G256_RESIDUAL_SRC,
            "gemm_hfq6g256_residual",
        )?;
        let func = &self.functions["gemm_hfq6g256_residual"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };

        // Bandwidth: weight (HFQ6: 200 bytes/group vs HFQ4: 136), per-batch x read, per-batch y RMW.
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)  // placeholder until hfq6 profiling added
            + batch_size * k * 4
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq6g256_residual", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// FP16-input batched HFQ6-G256 GEMM with residual add.
    /// Converts X from FP32 to FP16 (halving X bandwidth), then runs the
    /// FP16-packed GEMM kernel.
    pub fn gemm_hfq6g256_residual_fp16(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor, // FP32 [batch_size x K]
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq6g256_residual_fp16",
            kernels::GEMM_HFQ6G256_RESIDUAL_FP16_SRC,
            "gemm_hfq6g256_residual_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        // FP16 GEMM
        let func = &self.functions["gemm_hfq6g256_residual_fp16"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k * 2  // FP16 X (half bandwidth)
            + batch_size * m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq6g256_residual_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// WMMA-accelerated batched HFQ6-G256 GEMM with residual add.
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    pub fn gemm_hfq6g256_residual_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kernel_name, kernel_src, block_size, row_step) = (
            "gemm_hfq6g256_residual_wmma_k2",
            kernels::GEMM_HFQ6G256_RESIDUAL_WMMA_K2_SRC,
            32u32,
            16usize,
        );
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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

        let row_tiles = (m + row_step - 1) / row_step;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles as u32, batch_tiles as u32, 1],
            [block_size, 1, 1],
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

    /// gfx12 sister of `gemm_hfq6g256_residual_wmma` (wo / w_down post-projection
    /// for MQ6 LA/FA attention). Caller seeds Y with the residual; kernel does
    /// `Y += X @ A^T`. Mirrors `gemm_q8_0_residual_wmma_gfx12` kernarg layout
    /// and grid.
    pub fn gemm_hfq6g256_residual_wmma_gfx12(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            k % 256,
            0,
            "gemm_hfq6g256_residual_wmma_gfx12: K must be a multiple of 256 (got K={k})"
        );
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq6g256_residual_wmma_gfx12",
            kernels::GEMM_HFQ6G256_RESIDUAL_WMMA_GFX12_SRC,
            "gemm_hfq6g256_residual_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k * 2  // FP16 X
            + batch_size * m * 4 * 2; // residual read + write
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq6g256_residual_wmma_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_hfq6g256_residual_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
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

    /// Batched 4-way fused HFQ6-G256 GEMM (qkv + z + beta + alpha).
    /// Auto-selects: gfx11 -> WMMA, else -> scalar.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq6g256(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !self.flags.fp16_disabled {
            if self.arch_caps.has_wmma_w32_gfx12() {
                return self.gemm_qkvza_hfq6g256_wmma_gfx12(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            if self.arch_caps.has_wmma_w32() {
                return self.gemm_qkvza_hfq6g256_wmma(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            // gfx906: wave64+dp4a batched fused (Phase A.3, plan v3.2.3 §5.1
            // item 3). Pre-quantize x to Q8_1 and dispatch the dp4a kernel.
            // Skip in capture mode (Q8_1 quantize launch must be reachable
            // from captured graph or pre-baked) — matches HFQ4 sibling pattern.
            if self.arch_caps.gemv_dp4a_enabled() && !self.graphs.capture_mode {
                return self.gemm_qkvza_hfq6g256_wave64_dp4a(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkvza_hfq6g256_dot2(
                    a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                    beta_m, alpha_m, k, batch_size,
                );
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_qkvza_hfq6g256_fp16(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_qkvza_hfq6g256",
            kernels::GEMM_QKVZA_HFQ6G256_SRC,
            "gemm_qkvza_hfq6g256",
        )?;
        let func = &self.functions["gemm_qkvza_hfq6g256"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq6g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// FP16-packed batched 4-way fused HFQ6-G256 GEMM (qkv + z + beta + alpha).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq6g256_fp16(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq6g256_fp16",
            kernels::GEMM_QKVZA_HFQ6G256_FP16_SRC,
            "gemm_qkvza_hfq6g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq6g256_fp16"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (qkv_m + z_m + beta_m + alpha_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq6g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// v_dot2_f32_f16-accelerated batched 4-way fused HFQ6-G256 GEMM (qkv + z + beta + alpha).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `amd_mixed_dot`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    #[allow(clippy::too_many_arguments)]
    /// gfx906 wave64+dp4a batched 4-way fused QKVZA GEMM. Phase A.3
    /// (plan v3.2.3 §5.1 item 3). Uses Q8_1 activation pre-quantize
    /// (shared with A.1c GEMV-shape dp4a kernels) and HFQ6 6-bit
    /// unsigned weight unpack.
    pub fn gemm_qkvza_hfq6g256_wave64_dp4a(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        self.ensure_kernel(
            "gemm_qkvza_hfq6g256_wave64_dp4a",
            kernels::GEMM_QKVZA_HFQ6G256_WAVE64_DP4A_SRC,
            "gemm_qkvza_hfq6g256_wave64_dp4a",
        )?;

        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let qkv_m_val = qkv_m as i32;
        let z_m_val = z_m as i32;
        let beta_m_val = beta_m as i32;
        let alpha_m_val = alpha_m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &az as *const _ as *mut c_void,
            &ab as *const _ as *mut c_void,
            &aa as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yz as *const _ as *mut c_void,
            &yb as *const _ as *mut c_void,
            &ya as *const _ as *mut c_void,
            &qkv_m_val as *const _ as *mut c_void,
            &z_m_val as *const _ as *mut c_void,
            &beta_m_val as *const _ as *mut c_void,
            &alpha_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 8;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let grid_x = (total_m + 1) / 2;

        self.launch_maybe_blob(
            "gemm_qkvza_hfq6g256_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xq);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(qkv_m_val);
                b.push_i32(z_m_val);
                b.push_i32(beta_m_val);
                b.push_i32(alpha_m_val);
                b.push_i32(k_val);
                b.push_i32(bs_val);
                b
            },
        )
    }

    pub fn gemm_qkvza_hfq6g256_dot2(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq6g256_dot2",
            kernels::GEMM_QKVZA_HFQ6G256_DOT2_SRC,
            "gemm_qkvza_hfq6g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkvza_hfq6g256_dot2"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;

        unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// WMMA-accelerated batched 4-way fused HFQ6-G256 GEMM (qkv + z + beta + alpha).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_hfq6g256_wmma(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq6g256_wmma",
            kernels::GEMM_QKVZA_HFQ6G256_WMMA_SRC,
            "gemm_qkvza_hfq6g256_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq6g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq6g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m);
                b.push_i32(z_m_val);
                b.push_i32(b_m);
                b.push_i32(a_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx12 (RDNA4) sister of `gemm_qkvza_hfq6g256_wmma`. Pure scaffold
    /// composition (hfq6 dequant + 4-output qkvza routing, both validated
    /// on R9700). Not yet wired into the public dispatch tree — exposed
    /// only for the channel-test harness. See issue #54.
    pub fn gemm_qkvza_hfq6g256_wmma_gfx12(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkvza_hfq6g256_wmma_gfx12",
            kernels::GEMM_QKVZA_HFQ6G256_WMMA_GFX12_SRC,
            "gemm_qkvza_hfq6g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq6g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_hfq6g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m);
                b.push_i32(z_m_val);
                b.push_i32(b_m);
                b.push_i32(a_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched 3-way fused HFQ6-G256 GEMM for the FA preamble (Q + K + V).
    /// Auto-selects: gfx11 -> WMMA, else -> scalar.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq6g256(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !self.flags.fp16_disabled {
            if self.arch_caps.has_wmma_w32_gfx12() {
                return self.gemm_qkv_hfq6g256_wmma_gfx12(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            if self.arch_caps.has_wmma_w32() {
                return self.gemm_qkv_hfq6g256_wmma(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            // gfx906: wave64+dp4a batched fused (Phase A.3).
            // Skip in capture mode (Q8_1 quantize) — matches HFQ4 sibling.
            if self.arch_caps.gemv_dp4a_enabled() && !self.graphs.capture_mode {
                return self.gemm_qkv_hfq6g256_wave64_dp4a(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_qkv_hfq6g256_dot2(
                    a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
                );
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_qkv_hfq6g256_fp16(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_qkv_hfq6g256",
            kernels::GEMM_QKV_HFQ6G256_SRC,
            "gemm_qkv_hfq6g256",
        )?;
        let func = &self.functions["gemm_qkv_hfq6g256"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq6g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// FP16-packed batched 3-way fused HFQ6-G256 GEMM (Q + K + V).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq6g256_fp16(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq6g256_fp16",
            kernels::GEMM_QKV_HFQ6G256_FP16_SRC,
            "gemm_qkv_hfq6g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq6g256_fp16"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (q_m + k_m + v_m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq6g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// v_dot2_f32_f16-accelerated batched 3-way fused HFQ6-G256 GEMM (Q + K + V).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `amd_mixed_dot`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    #[allow(clippy::too_many_arguments)]
    /// gfx906 wave64+dp4a batched 3-way fused QKV GEMM. Phase A.3
    /// (plan v3.2.3 §5.1 item 3). Sibling of qkvza_wave64_dp4a.
    pub fn gemm_qkv_hfq6g256_wave64_dp4a(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        self.ensure_kernel(
            "gemm_qkv_hfq6g256_wave64_dp4a",
            kernels::GEMM_QKV_HFQ6G256_WAVE64_DP4A_SRC,
            "gemm_qkv_hfq6g256_wave64_dp4a",
        )?;

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_val = q_m as i32;
        let k_m_val = k_m as i32;
        let v_m_val = v_m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &ak as *const _ as *mut c_void,
            &av as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yk as *const _ as *mut c_void,
            &yv as *const _ as *mut c_void,
            &q_m_val as *const _ as *mut c_void,
            &k_m_val as *const _ as *mut c_void,
            &v_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 8;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (q_m + k_m + v_m) as u32;
        let grid_x = (total_m + 1) / 2;

        self.launch_maybe_blob(
            "gemm_qkv_hfq6g256_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xq);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(bs_val);
                b
            },
        )
    }

    pub fn gemm_qkv_hfq6g256_dot2(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq6g256_dot2",
            kernels::GEMM_QKV_HFQ6G256_DOT2_SRC,
            "gemm_qkv_hfq6g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_qkv_hfq6g256_dot2"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (q_m + k_m + v_m) as u32;

        unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// WMMA-accelerated batched 3-way fused HFQ6-G256 GEMM (Q + K + V).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_hfq6g256_wmma(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq6g256_wmma",
            kernels::GEMM_QKV_HFQ6G256_WMMA_SRC,
            "gemm_qkv_hfq6g256_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq6g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq6g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx12 (RDNA4) sister of `gemm_qkv_hfq6g256_wmma`. Same gfx12 recipe
    /// as the hfq4 QKV scaffold (validated on R9700) with the hfq6 dequant
    /// inner loop carried over. Not yet wired into the public dispatch
    /// tree — exposed only for the channel-test harness. See issue #54.
    pub fn gemm_qkv_hfq6g256_wmma_gfx12(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_qkv_hfq6g256_wmma_gfx12",
            kernels::GEMM_QKV_HFQ6G256_WMMA_GFX12_SRC,
            "gemm_qkv_hfq6g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq6g256_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_hfq6g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched 2-way fused HFQ6-G256 GEMM for the FFN preamble (gate + up).
    /// Auto-selects: gfx11 -> WMMA, else -> scalar.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq6g256(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Fast paths for prefill (batch_size > 1). Disable with HIPFIRE_FP16=0.
        if batch_size > 1 && !self.flags.fp16_disabled {
            if self.arch_caps.has_wmma_w32_gfx12() {
                return self.gemm_gate_up_hfq6g256_wmma_gfx12(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            if self.arch_caps.has_wmma_w32() {
                return self.gemm_gate_up_hfq6g256_wmma(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            // gfx906: wave64+dp4a batched fused (Phase A.3).
            // Skip in capture mode (Q8_1 quantize) — matches HFQ4 sibling.
            if self.arch_caps.gemv_dp4a_enabled() && !self.graphs.capture_mode {
                return self.gemm_gate_up_hfq6g256_wave64_dp4a(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            // v_dot2_f32_f16 on archs that have it (gfx1011/1012/1030-1032).
            // Excludes gfx1010 (Navi 10, 5700 XT) and gfx1013 (Van Gogh/BC-250 APU).
            if self.arch_caps.has_dot2_f32_f16() {
                return self.gemm_gate_up_hfq6g256_dot2(
                    a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
                );
            }
            // FP16 packed (v_pk_fma_f16) for gfx1010/1013 — 2× scalar FP32.
            return self.gemm_gate_up_hfq6g256_fp16(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        self.ensure_kernel(
            "gemm_gate_up_hfq6g256",
            kernels::GEMM_GATE_UP_HFQ6G256_SRC,
            "gemm_gate_up_hfq6g256",
        )?;
        let func = &self.functions["gemm_gate_up_hfq6g256"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq6g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// FP16-packed batched 2-way fused HFQ6-G256 GEMM (gate + up).
    /// RDNA1/2 fast path — v_pk_fma_f16 inner loop, 2× scalar FP32 throughput.
    /// Requires FP16-converted X (provided via ensure_fp16_x).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq6g256_fp16(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq6g256_fp16",
            kernels::GEMM_GATE_UP_HFQ6G256_FP16_SRC,
            "gemm_gate_up_hfq6g256_fp16",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq6g256_fp16"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k)
                  + batch_size * k * 2  // FP16 X
                  + batch_size * (gate_m + up_m) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq6g256_fp16", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
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

    /// v_dot2_f32_f16-accelerated batched 2-way fused HFQ6-G256 GEMM (gate + up).
    /// RDNA2 (gfx1011/1012/1030-1032) fast path using `amd_mixed_dot`.
    /// One instruction per half2 dot with FP32 accumulation — 1.2-1.5× over FP16 packed.
    #[allow(clippy::too_many_arguments)]
    /// gfx906 wave64+dp4a batched 2-way fused gate+up GEMM. Phase A.3.
    pub fn gemm_gate_up_hfq6g256_wave64_dp4a(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let xq_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        self.ensure_kernel(
            "gemm_gate_up_hfq6g256_wave64_dp4a",
            kernels::GEMM_GATE_UP_HFQ6G256_WAVE64_DP4A_SRC,
            "gemm_gate_up_hfq6g256_wave64_dp4a",
        )?;

        let agate = a_gate.buf.as_ptr();
        let aup = a_up.buf.as_ptr();
        let ygate = y_gate.buf.as_ptr();
        let yup = y_up.buf.as_ptr();
        let gate_m_val = gate_m as i32;
        let up_m_val = up_m as i32;
        let k_val = k as i32;
        let bs_val = batch_size as i32;
        let mut xq = xq_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &agate as *const _ as *mut c_void,
            &aup as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &ygate as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &gate_m_val as *const _ as *mut c_void,
            &up_m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &bs_val as *const _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 8;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (gate_m + up_m) as u32;
        let grid_x = (total_m + 1) / 2;

        self.launch_maybe_blob(
            "gemm_gate_up_hfq6g256_wave64_dp4a",
            [grid_x, batch_tiles as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(agate);
                b.push_ptr(aup);
                b.push_ptr(xq);
                b.push_ptr(ygate);
                b.push_ptr(yup);
                b.push_i32(gate_m_val);
                b.push_i32(up_m_val);
                b.push_i32(k_val);
                b.push_i32(bs_val);
                b
            },
        )
    }

    pub fn gemm_gate_up_hfq6g256_dot2(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq6g256_dot2",
            kernels::GEMM_GATE_UP_HFQ6G256_DOT2_SRC,
            "gemm_gate_up_hfq6g256_dot2",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
        let func = &self.functions["gemm_gate_up_hfq6g256_dot2"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = {
            const BATCH_TILE: usize = 8;
            (batch_size + BATCH_TILE - 1) / BATCH_TILE
        };
        let total_m = (gate_m + up_m) as u32;

        unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// WMMA-accelerated batched 2-way fused HFQ6-G256 GEMM (gate + up).
    /// gfx1100+ only. 16x16 output tiles via wave32 WMMA.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_hfq6g256_wmma(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq6g256_wmma",
            kernels::GEMM_GATE_UP_HFQ6G256_WMMA_SRC,
            "gemm_gate_up_hfq6g256_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq6g256_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq6g256_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(g_m);
                b.push_i32(u_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx12 (RDNA4) sister of `gemm_gate_up_hfq6g256_wmma`. Same gfx12
    /// recipe as the other scaffolds (validated on R9700). Not yet wired
    /// into the public dispatch tree — exposed only for the channel-test
    /// harness. See issue #54.
    pub fn gemm_gate_up_hfq6g256_wmma_gfx12(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_gate_up_hfq6g256_wmma_gfx12",
            kernels::GEMM_GATE_UP_HFQ6G256_WMMA_GFX12_SRC,
            "gemm_gate_up_hfq6g256_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_gate_up_hfq6g256_wmma_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_gate_up_hfq6g256_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(g_m);
                b.push_i32(u_m);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Fused QKV: three Q4_K GEMVs in one launch (saves 2 kernel launches per layer).
    /// q = Wq * x, k = Wk * x, v = Wv * x — all read the same input x.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_q4k(
        &mut self,
        wq: &GpuTensor,
        wk: &GpuTensor,
        wv: &GpuTensor,
        x: &GpuTensor,
        yq: &GpuTensor,
        yk: &GpuTensor,
        yv: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("fused_qkv_q4k", kernels::FUSED_QKV_Q4K_SRC, "fused_qkv_q4k")?;
        let func = &self.functions["fused_qkv_q4k"];

        let mut aq = wq.buf.as_ptr();
        let mut ak = wk.buf.as_ptr();
        let mut av = wv.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yqp = yq.buf.as_ptr();
        let mut ykp = yk.buf.as_ptr();
        let mut yvp = yv.buf.as_ptr();
        let mut qm = q_m as i32;
        let mut km = k_m as i32;
        let mut vm = v_m as i32;
        let mut kk = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yqp as *mut _ as *mut c_void,
            &mut ykp as *mut _ as *mut c_void,
            &mut yvp as *mut _ as *mut c_void,
            &mut qm as *mut _ as *mut c_void,
            &mut km as *mut _ as *mut c_void,
            &mut vm as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];
        let grid = (q_m + k_m + v_m) as u32;
        unsafe {
            self.hip
                .launch_kernel(func, [grid, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }

    /// `fused_qkv_q4k` with an optional Q/K/V bias folded into the kernel's
    /// lane-0 store (`HIPFIRE_FUSE_QKV_BIAS`). All-null = byte-identical to the
    /// unfused path.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_q4k_with_bias(
        &mut self,
        wq: &GpuTensor,
        wk: &GpuTensor,
        wv: &GpuTensor,
        x: &GpuTensor,
        yq: &GpuTensor,
        yk: &GpuTensor,
        yv: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        bias_q_ptr: *mut c_void,
        bias_k_ptr: *mut c_void,
        bias_v_ptr: *mut c_void,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_qkv_q4k_qwen2_bias",
            kernels::FUSED_QKV_Q4K_QWEN2_BIAS_SRC,
            "fused_qkv_q4k_qwen2_bias",
        )?;
        let func = &self.functions["fused_qkv_q4k_qwen2_bias"];

        let mut aq = wq.buf.as_ptr();
        let mut ak = wk.buf.as_ptr();
        let mut av = wv.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yqp = yq.buf.as_ptr();
        let mut ykp = yk.buf.as_ptr();
        let mut yvp = yv.buf.as_ptr();
        let mut qm = q_m as i32;
        let mut km = k_m as i32;
        let mut vm = v_m as i32;
        let mut kk = k as i32;
        let mut bq = bias_q_ptr;
        let mut bk = bias_k_ptr;
        let mut bv = bias_v_ptr;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yqp as *mut _ as *mut c_void,
            &mut ykp as *mut _ as *mut c_void,
            &mut yvp as *mut _ as *mut c_void,
            &mut qm as *mut _ as *mut c_void,
            &mut km as *mut _ as *mut c_void,
            &mut vm as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
            &mut bq as *mut _ as *mut c_void,
            &mut bk as *mut _ as *mut c_void,
            &mut bv as *mut _ as *mut c_void,
        ];

        let grid = (q_m + k_m + v_m) as u32;
        unsafe {
            self.hip
                .launch_kernel(func, [grid, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }

    /// Fused Gate+Up: two Q4_K GEMVs in one launch (saves 1 kernel launch per layer).
    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_q4k(
        &mut self,
        w_gate: &GpuTensor,
        w_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_gate_up_q4k",
            kernels::FUSED_GATE_UP_Q4K_SRC,
            "fused_gate_up_q4k",
        )?;
        let func = &self.functions["fused_gate_up_q4k"];

        let mut ag = w_gate.buf.as_ptr();
        let mut au = w_up.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut ygp = y_gate.buf.as_ptr();
        let mut yup = y_up.buf.as_ptr();
        let mut gm = gate_m as i32;
        let mut um = up_m as i32;
        let mut kk = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut ygp as *mut _ as *mut c_void,
            &mut yup as *mut _ as *mut c_void,
            &mut gm as *mut _ as *mut c_void,
            &mut um as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];

        let grid = (gate_m + up_m) as u32;
        unsafe {
            self.hip
                .launch_kernel(func, [grid, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }

    /// y = A_q8_0 * x (quantized GEMV for Q8_0)

    /// Y[batch, M] = X[batch, K] @ A_q8[M, K]^T — batched Q8_0 GEMM.
    /// One block per output row (32 threads, one wave). Each thread holds
    /// MAX_BATCH=16 per-batch accumulators and broadcasts each weight load.
    /// Drops the (batch_size − 1)× weight re-reads of the GEMV-loop path
    /// without splitting launches.
    pub fn gemm_q8_0_batched(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            batch_size <= 64,
            "gemm_q8_0_batched: batch_size {batch_size} exceeds kernel MAX_BATCH=64"
        );
        self.ensure_kernel(
            "gemm_q8_0_batched",
            kernels::GEMM_Q8_0_BATCHED_SRC,
            "gemm_q8_0_batched",
        )?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
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

        // Q8_0 weight bytes dominate: 34 B per 32-element group (32 int8 + f16
        // scale). This kernel carried NO profile timer, so `HIPFIRE_PROFILE`
        // silently under-reported the hottest weight path on CDNA3 (where the
        // WMMA Q8 kernels are unavailable and everything lands here).
        let bytes = m.saturating_mul(k) / 32 * 34 + batch_size.saturating_mul(k) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_q8_0_batched", bytes);
        let result = self.launch_maybe_blob(
            "gemm_q8_0_batched",
            [m as u32, 1, 1],
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

    /// Q8_0 batched GEMM driver that handles `n` rows by sub-batching at the
    /// kernel's MAX_BATCH=64. Y[n, m] = X[n, k] @ A_q8[m, k]^T.
    ///
    /// On wave32-WMMA archs with K % 32 == 0, routes the entire call through
    /// the WMMA Q8 GEMM (`gemm_q8_0_wmma`, or its gfx12 sibling) which is
    /// much faster than the scalar `gemm_q8_0_batched` per output. Opt out
    /// via HIPFIRE_Q8_BATCHED_LEGACY=1.
    pub fn gemm_q8_0_batched_chunked(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        static USE_LEGACY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_legacy = *USE_LEGACY.get_or_init(|| self.flags.q8_batched_legacy);
        // Route to WMMA on ALL wave32-WMMA arches (gfx11 RDNA3, gfx1151
        // RDNA3.5, gfx12 RDNA4), not just RDNA4. `gemm_q8_0_wmma` selects the
        // RDNA3 vs RDNA4 kernel internally and asserts `has_wmma()`, so this is
        // exactly its own precondition. The former `is_rdna4()` gate left gfx11
        // falling back to the scalar 1-wave-per-row `gemm_q8_0_batched` even
        // though the RDNA3 WMMA kernel is validated (parity test +
        // stale-fp16-cache fix, 49881383). On A3B mq4p this scalar path was
        // ~31% of the gfx11 MTP verify GPU time (rocprofv3 2026-06-16: 260ms /
        // 80 cycles); mq4 never hit it (4-bit hfq4 projections). Mirrors the
        // W3i MTP-head lm_head fix (5ac96a8f, +47% gfx12).
        if !use_legacy && self.arch_caps.has_wmma() && k % 32 == 0 && n > 0 {
            return self.gemm_q8_0_wmma(a_raw, x, y, m, k, n);
        }

        const MAX_BATCH: usize = 64;
        let mut off = 0;
        while off < n {
            let take = (n - off).min(MAX_BATCH);
            let x_sub = x.sub_offset(off * k, take * k);
            let y_sub = y.sub_offset(off * m, take * m);
            self.gemm_q8_0_batched(a_raw, &x_sub, &y_sub, m, k, take)?;
            off += take;
        }
        Ok(())
    }

    /// Resolve and dispatch the Q8_0 dense prefill GEMM (`Y[N,M] = X[N,K] @
    /// A_q8[M,K]^T`). Picks the i8-MMQ / f16-WMMA / scalar-chunked kernel from
    /// arch caps, feature flags, and shape — so arch forward passes carry no
    /// kernel-selection knowledge (mirrors the minimax dense path, which calls
    /// `gemm_q8_0_batched_chunked` blind).
    ///
    /// `x` is the plain F32 activation batch. `x_f16_scratch` is a caller-owned
    /// F16 staging tensor the f16-WMMA kernels require; the i8-MMQ kernel reads
    /// the int8 weights directly and quantizes `x` internally, and the scalar
    /// chunked fallback takes F32 — both ignore the scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q8_0_wmma_prefill_auto(
        &mut self,
        weight: &GpuTensor,
        x: &GpuTensor,
        x_f16_scratch: Option<&GpuTensor>,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        let wmma_on = !self.flags.deepseek4_q8_wmma_off;
        if wmma_on && self.arch_caps.is_rdna4() {
            // RDNA4 (gfx12): upstream-tuned gating.
            if let Some(scratch) = x_f16_scratch {
                let nn = (n * k) as i64;
                self.deepseek4_convert_f32_to_f16(x, scratch, nn)?;
                let use_4w = !self.flags.deepseek4_q8_4w_off
                    && n >= 256
                    && m >= 4096
                    && m % 64 == 0
                    && k % 32 == 0
                    && n % 64 == 0;
                if use_4w {
                    return self.gemm_q8_0_wmma_4w(weight, scratch, y, m, k, n);
                }
                return self.gemm_q8_0_wmma(weight, scratch, y, m, k, n);
            }
        } else if wmma_on && self.arch_caps.has_wmma() && m % 64 == 0 && k % 32 == 0 {
            // i8-MMQ 64×64 dense path (gfx1151): int8 weights direct + Q8_1
            // activations + i8 WMMA at ~2x. ~1.4-1.56x over the f16 64×64 kernel
            // at the q-LoRA projection shapes (M=4096, batch≥256).
            if self.arch == "gfx1151" && n % 64 == 0 && k % 128 == 0 {
                return self.gemm_q8_0_mmq_4w_gfx1151(weight, x, y, m, k, n);
            }
            // gfx11 / RDNA3.5 (gfx1151) Q8_0 WMMA prefill. The activation is
            // pre-converted to F16 in `scratch`; the kernels honor the F16 dtype
            // (no re-convert). 4-warp 64×64-tile kernel for batch%64==0 (~12%
            // over single-warp 16×16; weight-bandwidth-bound); the 4w opt-out
            // forces single-warp.
            if let Some(scratch) = x_f16_scratch {
                let nn = (n * k) as i64;
                self.deepseek4_convert_f32_to_f16(x, scratch, nn)?;
                if !self.flags.deepseek4_q8_4w_off && n >= 64 && n % 64 == 0 {
                    return self.gemm_q8_0_wmma_4w(weight, scratch, y, m, k, n);
                }
                return self.gemm_q8_0_wmma(weight, scratch, y, m, k, n);
            }
        }
        self.gemm_q8_0_batched_chunked(weight, x, y, m, k, n)
    }

    /// WMMA Q8_0 GEMM (no residual). Y[N, M] = X[N, K] @ A_q8[M, K]^T.
    /// gfx12 (RDNA4) only. Drop-in replacement for `gemm_q8_0_batched`;
    /// the scalar 1-wave-per-row kernel was 65% of A3B prefill GPU time
    /// per rocprofv3 2026-05-19. Mirrors `gemm_q8_0_residual_wmma_gfx12`
    /// without the residual load.
    pub fn gemm_q8_0_wmma(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_q8_0_wmma: K must be a multiple of 32 (got K={k})"
        );
        debug_assert!(
            self.arch_caps.has_wmma(),
            "gemm_q8_0_wmma: requires wave32 WMMA (gfx11+); got arch {}",
            self.arch
        );
        self.bind_thread()?;
        // RDNA3/RDNA3.5 and RDNA4 wave32 WMMA use different f32-accumulator
        // output layouts, so select the matching kernel source. The RDNA3
        // source (gfx1151-tuned) is the single-warp Q8_0 drop-in on gfx11 /
        // RDNA3.5; `_gfx12` is its RDNA4 sibling. Launch is identical.
        let (kname, ksrc): (&str, &str) = if self.arch_caps.is_rdna4() {
            ("gemm_q8_0_wmma_gfx12", kernels::GEMM_Q8_0_WMMA_GFX12_SRC)
        } else {
            ("gemm_q8_0_wmma", kernels::GEMM_Q8_0_WMMA_SRC)
        };
        self.ensure_kernel(kname, ksrc, kname)?;
        // Use unconditional F32→F16 conversion (no pointer-keyed cache).
        //
        // STALE-CACHE BUG (2026-06-16): the former `ensure_fp16_x` is
        // pointer-keyed — it skips reconversion when the source ptr matches
        // the last call's ptr. In the MTP decode path the `tmp_batched`
        // scratch tensor is a FIXED allocation (same device address) that
        // gets NEW content from `rmsnorm_batched` each step. After step 0
        // warms the cache, every subsequent step read stale FP16 from step 0,
        // producing wrong logits and collapsing τ from ~1.85 to ~1.01.
        // Using `convert_fp16_x_uncached` (always re-converts) is always
        // correct; the perf cost is one extra F32→F16 kernel per lm_head
        // call, negligible vs the GEMM itself.
        let x_f16_ptr = if matches!(x.dtype, DType::F16) {
            x.buf.as_ptr()
        } else {
            self.convert_fp16_x_uncached(x, batch_size * k)?
        };

        let mut a_p = a.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut y_p = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_p as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut y_p as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let bytes = m * (k / 32) * 34 + batch_size * k * 2 + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kname, bytes);
        let result = self.launch_maybe_blob(
            kname,
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_p);
                b.push_ptr(xp);
                b.push_ptr(y_p);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// WMMA 4-way fused Q8_0 GEMM (wqkv + wz + w_beta + w_alpha).
    /// DeltaNet LA preamble. Auto-routes to gfx12 sibling on RDNA4.
    pub fn gemm_qkvza_q8_0_wmma(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.is_rdna4() {
            return self.gemm_qkvza_q8_0_wmma_gfx12(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, batch_size,
            );
        }
        // Q8_0 packs 32 elements per block (34 bytes); the kernel iterates
        // `K/32` blocks per row and silently drops any tail if K is not a
        // multiple of 32. All current production shapes satisfy this; guard
        // here to catch future shape regressions before they corrupt output.
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_qkvza_q8_0_wmma: K must be a multiple of 32 (got K={k})"
        );
        self.ensure_kernel(
            "gemm_qkvza_q8_0_wmma",
            kernels::GEMM_QKVZA_Q8_0_WMMA_SRC,
            "gemm_qkvza_q8_0_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_qkv_p = a_qkv.buf.as_ptr();
        let mut a_z_p = a_z.buf.as_ptr();
        let mut a_beta_p = a_beta.buf.as_ptr();
        let mut a_alpha_p = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut y_qkv_p = y_qkv.buf.as_ptr();
        let mut y_z_p = y_z.buf.as_ptr();
        let mut y_beta_p = y_beta.buf.as_ptr();
        let mut y_alpha_p = y_alpha.buf.as_ptr();
        let mut qkv_m_val = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut beta_m_val = beta_m as i32;
        let mut alpha_m_val = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_qkv_p as *mut _ as *mut c_void,
            &mut a_z_p as *mut _ as *mut c_void,
            &mut a_beta_p as *mut _ as *mut c_void,
            &mut a_alpha_p as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut y_qkv_p as *mut _ as *mut c_void,
            &mut y_z_p as *mut _ as *mut c_void,
            &mut y_beta_p as *mut _ as *mut c_void,
            &mut y_alpha_p as *mut _ as *mut c_void,
            &mut qkv_m_val as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut beta_m_val as *mut _ as *mut c_void,
            &mut alpha_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let q8_bytes = |m: usize| m * (k / 32) * 34;
        let bytes = q8_bytes(qkv_m)
            + q8_bytes(z_m)
            + q8_bytes(beta_m)
            + q8_bytes(alpha_m)
            + batch_size * k * 2
            + batch_size * total_m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_q8_0_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_q8_0_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_qkv_p);
                b.push_ptr(a_z_p);
                b.push_ptr(a_beta_p);
                b.push_ptr(a_alpha_p);
                b.push_ptr(xp);
                b.push_ptr(y_qkv_p);
                b.push_ptr(y_z_p);
                b.push_ptr(y_beta_p);
                b.push_ptr(y_alpha_p);
                b.push_i32(qkv_m_val);
                b.push_i32(z_m_val);
                b.push_i32(beta_m_val);
                b.push_i32(alpha_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// WMMA 2-way fused Q8_0 GEMM (w_gate + w_up). FFN preamble.
    /// Auto-routes to gfx12 sibling on RDNA4.
    pub fn gemm_gate_up_q8_0_wmma(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.is_rdna4() {
            return self.gemm_gate_up_q8_0_wmma_gfx12(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, batch_size,
            );
        }
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_gate_up_q8_0_wmma: K must be a multiple of 32 (got K={k})"
        );
        self.ensure_kernel(
            "gemm_gate_up_q8_0_wmma",
            kernels::GEMM_GATE_UP_Q8_0_WMMA_SRC,
            "gemm_gate_up_q8_0_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_g = a_gate.buf.as_ptr();
        let mut a_u = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut y_g = y_gate.buf.as_ptr();
        let mut y_u = y_up.buf.as_ptr();
        let mut gate_m_val = gate_m as i32;
        let mut up_m_val = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_g as *mut _ as *mut c_void,
            &mut a_u as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut y_g as *mut _ as *mut c_void,
            &mut y_u as *mut _ as *mut c_void,
            &mut gate_m_val as *mut _ as *mut c_void,
            &mut up_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let q8_bytes = |m: usize| m * (k / 32) * 34;
        let bytes =
            q8_bytes(gate_m) + q8_bytes(up_m) + batch_size * k * 2 + batch_size * total_m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_q8_0_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_q8_0_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_g);
                b.push_ptr(a_u);
                b.push_ptr(xp);
                b.push_ptr(y_g);
                b.push_ptr(y_u);
                b.push_i32(gate_m_val);
                b.push_i32(up_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// WMMA Q8_0 GEMM with fused residual add (Y += X @ A^T).
    /// Caller seeds Y with the residual. Auto-routes to gfx12 sibling
    /// on RDNA4.
    pub fn gemm_q8_0_residual_wmma(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.is_rdna4() {
            return self.gemm_q8_0_residual_wmma_gfx12(a, x, y, m, k, batch_size);
        }
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_q8_0_residual_wmma: K must be a multiple of 32 (got K={k})"
        );
        self.ensure_kernel(
            "gemm_q8_0_residual_wmma",
            kernels::GEMM_Q8_0_RESIDUAL_WMMA_SRC,
            "gemm_q8_0_residual_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_p = a.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut y_p = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_p as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut y_p as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let bytes = m * (k / 32) * 34 + batch_size * k * 2 + batch_size * m * 4 * 2; // residual read + write
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_q8_0_residual_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_q8_0_residual_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_p);
                b.push_ptr(xp);
                b.push_ptr(y_p);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// WMMA-accelerated batched 3-way fused Q8_0 GEMM (Q + K + V projections).
    /// Auto-routes to the gfx12 sibling on RDNA4 archs; gfx11 path is the
    /// canonical implementation (X is converted from F32 to FP16 via
    /// `ensure_fp16_x`). Mirrors `gemm_qkv_hfq4g256_wmma`.
    pub fn gemm_qkv_q8_0_wmma(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.is_rdna4() {
            return self.gemm_qkv_q8_0_wmma_gfx12(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, batch_size,
            );
        }
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_qkv_q8_0_wmma: K must be a multiple of 32 (got K={k})"
        );
        self.ensure_kernel(
            "gemm_qkv_q8_0_wmma",
            kernels::GEMM_QKV_Q8_0_WMMA_SRC,
            "gemm_qkv_q8_0_wmma",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        // Byte accounting: per-weight Q8_0 = (k/32)*34 bytes/row × m rows,
        // plus X (fp16) and 3× Y (f32).
        let q8_bytes = |m: usize| m * (k / 32) * 34;
        let bytes = q8_bytes(q_m)
            + q8_bytes(k_m)
            + q8_bytes(v_m)
            + batch_size * k * 2
            + batch_size * total_m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_q8_0_wmma", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_q8_0_wmma",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx12 (RDNA4) sister of `gemm_qkv_q8_0_wmma`. Uses
    /// `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` (vs the gfx11 `_w32`)
    /// and half8_t operands with K split across 2 lane-groups. Mirrors the
    /// `gemm_qkv_hfq4g256_wmma_gfx12` pattern.
    pub fn gemm_qkv_q8_0_wmma_gfx12(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_qkv_q8_0_wmma_gfx12: K must be a multiple of 32 (got K={k})"
        );
        self.ensure_kernel(
            "gemm_qkv_q8_0_wmma_gfx12",
            kernels::GEMM_QKV_Q8_0_WMMA_GFX12_SRC,
            "gemm_qkv_q8_0_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = q_m + k_m + v_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let q8_bytes = |m: usize| m * (k / 32) * 34;
        let bytes = q8_bytes(q_m)
            + q8_bytes(k_m)
            + q8_bytes(v_m)
            + batch_size * k * 2
            + batch_size * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_q8_0_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkv_q8_0_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_val);
                b.push_i32(k_m_val);
                b.push_i32(v_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx12 sister of `gemm_qkvza_q8_0_wmma` (DeltaNet LA preamble).
    pub fn gemm_qkvza_q8_0_wmma_gfx12(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_qkvza_q8_0_wmma_gfx12: K must be a multiple of 32 (got K={k})"
        );
        self.ensure_kernel(
            "gemm_qkvza_q8_0_wmma_gfx12",
            kernels::GEMM_QKVZA_Q8_0_WMMA_GFX12_SRC,
            "gemm_qkvza_q8_0_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_qkv_p = a_qkv.buf.as_ptr();
        let mut a_z_p = a_z.buf.as_ptr();
        let mut a_beta_p = a_beta.buf.as_ptr();
        let mut a_alpha_p = a_alpha.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut y_qkv_p = y_qkv.buf.as_ptr();
        let mut y_z_p = y_z.buf.as_ptr();
        let mut y_beta_p = y_beta.buf.as_ptr();
        let mut y_alpha_p = y_alpha.buf.as_ptr();
        let mut qkv_m_val = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut beta_m_val = beta_m as i32;
        let mut alpha_m_val = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_qkv_p as *mut _ as *mut c_void,
            &mut a_z_p as *mut _ as *mut c_void,
            &mut a_beta_p as *mut _ as *mut c_void,
            &mut a_alpha_p as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut y_qkv_p as *mut _ as *mut c_void,
            &mut y_z_p as *mut _ as *mut c_void,
            &mut y_beta_p as *mut _ as *mut c_void,
            &mut y_alpha_p as *mut _ as *mut c_void,
            &mut qkv_m_val as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut beta_m_val as *mut _ as *mut c_void,
            &mut alpha_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let q8_bytes = |m: usize| m * (k / 32) * 34;
        let bytes = q8_bytes(qkv_m)
            + q8_bytes(z_m)
            + q8_bytes(beta_m)
            + q8_bytes(alpha_m)
            + batch_size * k * 2
            + batch_size * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_q8_0_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_qkvza_q8_0_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_qkv_p);
                b.push_ptr(a_z_p);
                b.push_ptr(a_beta_p);
                b.push_ptr(a_alpha_p);
                b.push_ptr(xp);
                b.push_ptr(y_qkv_p);
                b.push_ptr(y_z_p);
                b.push_ptr(y_beta_p);
                b.push_ptr(y_alpha_p);
                b.push_i32(qkv_m_val);
                b.push_i32(z_m_val);
                b.push_i32(beta_m_val);
                b.push_i32(alpha_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx12 sister of `gemm_gate_up_q8_0_wmma` (FFN preamble).
    pub fn gemm_gate_up_q8_0_wmma_gfx12(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_gate_up_q8_0_wmma_gfx12: K must be a multiple of 32 (got K={k})"
        );
        self.ensure_kernel(
            "gemm_gate_up_q8_0_wmma_gfx12",
            kernels::GEMM_GATE_UP_Q8_0_WMMA_GFX12_SRC,
            "gemm_gate_up_q8_0_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_g = a_gate.buf.as_ptr();
        let mut a_u = a_up.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut y_g = y_gate.buf.as_ptr();
        let mut y_u = y_up.buf.as_ptr();
        let mut gate_m_val = gate_m as i32;
        let mut up_m_val = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_g as *mut _ as *mut c_void,
            &mut a_u as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut y_g as *mut _ as *mut c_void,
            &mut y_u as *mut _ as *mut c_void,
            &mut gate_m_val as *mut _ as *mut c_void,
            &mut up_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let row_tiles = (total_m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let q8_bytes = |m: usize| m * (k / 32) * 34;
        let bytes =
            q8_bytes(gate_m) + q8_bytes(up_m) + batch_size * k * 2 + batch_size * total_m * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_q8_0_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_gate_up_q8_0_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_g);
                b.push_ptr(a_u);
                b.push_ptr(xp);
                b.push_ptr(y_g);
                b.push_ptr(y_u);
                b.push_i32(gate_m_val);
                b.push_i32(up_m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx12 sister of `gemm_q8_0_residual_wmma` (wo / w_down post-projection).
    /// Caller seeds Y with the residual; kernel does `Y += X @ A^T`.
    pub fn gemm_q8_0_residual_wmma_gfx12(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            k % 32,
            0,
            "gemm_q8_0_residual_wmma_gfx12: K must be a multiple of 32 (got K={k})"
        );
        self.ensure_kernel(
            "gemm_q8_0_residual_wmma_gfx12",
            kernels::GEMM_Q8_0_RESIDUAL_WMMA_GFX12_SRC,
            "gemm_q8_0_residual_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_p = a.buf.as_ptr();
        let mut xp = x_f16_ptr;
        let mut y_p = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_p as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut y_p as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;
        let bytes = m * (k / 32) * 34 + batch_size * k * 2 + batch_size * m * 4 * 2; // residual read + write
        let timer =
            crate::profile::begin_timer(&self.hip, "gemm", "gemm_q8_0_residual_wmma_gfx12", bytes);
        let result = self.launch_maybe_blob(
            "gemm_q8_0_residual_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_p);
                b.push_ptr(xp);
                b.push_ptr(y_p);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// y = A_q8hfq * x (split-metadata Q8 GEMV, row_stride = padded row bytes)

    /// y = A_q6k * x (quantized GEMV for Q6_K)

    /// y = A_q4f16 * x (RDNA-native Q4_F16 GEMV, group size 64)
    /// a_raw: raw Q4_F16_G64 bytes on GPU, x: F32 input, y: F32 output
    /// Block: 36 bytes per 64 elements. K must be multiple of 64.
    /// Uses 128 threads (4 warps) with shared memory reduction for increased MLP.

    /// y = A_q4f16 * x (256-thread wide variant for occupancy testing)
    /// Element-strided access pattern matching F32 GEMV. Shared memory reduction.

    /// y = A_q4f16 * x (RDNA-native Q4_F16 GEMV, group size 32)
    /// Block: 20 bytes per 32 elements. K must be multiple of 32.

    /// Fused Gate+Up HFQ4-G256: two GEMVs in one launch.
    pub fn fused_gate_up_hfq4g256(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // gfx906 dp4a opt-in: pre-quantize x to Q8_1 and use the
        // v_dot4_i32_i8 path. PMC at 2026-05-05 showed this kernel
        // was memory-bound; dp4a's 75% x-traffic reduction lands on
        // the actual bottleneck.
        if self.arch_caps.gemv_dp4a_enabled() {
            return self
                .fused_gate_up_hfq4g256_dp4a(a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k);
        }

        let cdna_wave64 = self.arch_caps.is_wave64_native();
        let (func_name, block, grid_x) = if cdna_wave64 {
            // gfx94x v2: 2 wave64s = 4 rows/WG, +1.9% on AR decode
            // (commit 5bd75a69 sibling). Default ON; opt out via
            // HIPFIRE_GFX942_GEMV_V2=0.
            let is_gfx94x = self.arch_caps.is_cdna3();
            let v2_on = self.flags.gfx942_gemv_v2.unwrap_or(true);
            if is_gfx94x && v2_on {
                self.ensure_kernel(
                    "fused_gate_up_hfq4g256_v2_gfx942",
                    kernels::FUSED_GATE_UP_HFQ4G256_V2_GFX942_SRC,
                    "fused_gate_up_hfq4g256_v2_gfx942",
                )?;
                let total = (gate_m + up_m) as u32;
                (
                    "fused_gate_up_hfq4g256_v2_gfx942",
                    [128u32, 1, 1],
                    (total + 3) / 4,
                )
            } else {
                self.ensure_kernel(
                    "fused_gate_up_hfq4g256_wave64",
                    kernels::FUSED_GATE_UP_HFQ4G256_WAVE64_SRC,
                    "fused_gate_up_hfq4g256_wave64",
                )?;
                let total = (gate_m + up_m) as u32;
                (
                    "fused_gate_up_hfq4g256_wave64",
                    [64u32, 1, 1],
                    (total + 1) / 2,
                )
            }
        } else if self.arch_caps.is_gfx1201() && k == 1024 {
            self.ensure_kernel(
                "fused_gate_up_hfq4g256_k1024_gfx1201",
                kernels::FUSED_GATE_UP_HFQ4G256_K1024_GFX1201_SRC,
                "fused_gate_up_hfq4g256_k1024_gfx1201",
            )?;
            (
                "fused_gate_up_hfq4g256_k1024_gfx1201",
                [32u32, 1, 1],
                (gate_m + up_m) as u32,
            )
        } else {
            self.ensure_kernel(
                "fused_gate_up_hfq4g256",
                kernels::FUSED_GATE_UP_HFQ4G256_SRC,
                "fused_gate_up_hfq4g256",
            )?;
            (
                "fused_gate_up_hfq4g256",
                [32u32, 1, 1],
                (gate_m + up_m) as u32,
            )
        };
        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let gm = gate_m as i32;
        let um = up_m as i32;
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ag as *const _ as *mut c_void,
            &au as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yg as *const _ as *mut c_void,
            &yu as *const _ as *mut c_void,
            &gm as *const _ as *mut c_void,
            &um as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(func_name, [grid_x, 1, 1], block, 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(ag);
            b.push_ptr(au);
            b.push_ptr(xp);
            b.push_ptr(yg);
            b.push_ptr(yu);
            b.push_i32(gm);
            b.push_i32(um);
            b.push_i32(kv);
            b
        })
    }

    /// Fused gate+up for Q8_0 weights: two Q8 GEMVs in one launch.
    /// Grid=[gate_m + up_m], block=[32].
    pub fn fused_gate_up_q8_0(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_gate_up_q8_0",
            kernels::FUSED_GATE_UP_Q8_0_SRC,
            "fused_gate_up_q8_0",
        )?;

        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let gm = gate_m as i32;
        let um = up_m as i32;
        let kv = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ag as *const _ as *mut c_void,
            &au as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yg as *const _ as *mut c_void,
            &yu as *const _ as *mut c_void,
            &gm as *const _ as *mut c_void,
            &um as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "fused_gate_up_q8_0",
            [(gate_m + up_m) as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xp);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(gm);
                b.push_i32(um);
                b.push_i32(kv);
                b
            },
        )
    }

    /// 4-way fused QKVZA for Q8_0 weights (DECODE, n=1).
    /// Bit-exact with four sequential gemv_q8_0 calls.
    /// Grid=[qkv_m + z_m + beta_m + alpha_m], block=[32].
    pub fn fused_qkvza_q8_0(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_qkvza_q8_0",
            kernels::FUSED_QKVZA_Q8_0_SRC,
            "fused_qkvza_q8_0",
        )?;

        let aq = a_qkv.buf.as_ptr();
        let az = a_z.buf.as_ptr();
        let ab = a_beta.buf.as_ptr();
        let aa = a_alpha.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_qkv.buf.as_ptr();
        let yz = y_z.buf.as_ptr();
        let yb = y_beta.buf.as_ptr();
        let ya = y_alpha.buf.as_ptr();
        let q_m_i = qkv_m as i32;
        let z_m_i = z_m as i32;
        let b_m_i = beta_m as i32;
        let a_m_i = alpha_m as i32;
        let k_i = k as i32;
        let total = (qkv_m + z_m + beta_m + alpha_m) as u32;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &az as *const _ as *mut c_void,
            &ab as *const _ as *mut c_void,
            &aa as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yz as *const _ as *mut c_void,
            &yb as *const _ as *mut c_void,
            &ya as *const _ as *mut c_void,
            &q_m_i as *const _ as *mut c_void,
            &z_m_i as *const _ as *mut c_void,
            &b_m_i as *const _ as *mut c_void,
            &a_m_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "fused_qkvza_q8_0",
            [total, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(az);
                b.push_ptr(ab);
                b.push_ptr(aa);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yz);
                b.push_ptr(yb);
                b.push_ptr(ya);
                b.push_i32(q_m_i);
                b.push_i32(z_m_i);
                b.push_i32(b_m_i);
                b.push_i32(a_m_i);
                b.push_i32(k_i);
                b
            },
        )
    }

    /// 3-way fused QKV for Q8_0 weights (DECODE, n=1).
    /// Bit-exact with three sequential gemv_q8_0 calls.
    /// Grid=[q_m + k_m + v_m], block=[32].
    pub fn fused_qkv_q8_0(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_qkv_q8_0",
            kernels::FUSED_QKV_Q8_0_SRC,
            "fused_qkv_q8_0",
        )?;

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_i = q_m as i32;
        let k_m_i = k_m as i32;
        let v_m_i = v_m as i32;
        let k_i = k as i32;
        let total = (q_m + k_m + v_m) as u32;
        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &ak as *const _ as *mut c_void,
            &av as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yk as *const _ as *mut c_void,
            &yv as *const _ as *mut c_void,
            &q_m_i as *const _ as *mut c_void,
            &k_m_i as *const _ as *mut c_void,
            &v_m_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "fused_qkv_q8_0",
            [total, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_i);
                b.push_i32(k_m_i);
                b.push_i32(v_m_i);
                b.push_i32(k_i);
                b
            },
        )
    }

    /// `fused_qkv_q8_0` with an optional Q/K/V bias folded into the kernel's
    /// lane-0 store (`HIPFIRE_FUSE_QKV_BIAS`). All-null = byte-identical to the
    /// unfused path.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_q8_0_with_bias(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        bias_q_ptr: *mut c_void,
        bias_k_ptr: *mut c_void,
        bias_v_ptr: *mut c_void,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_qkv_q8_0_qwen2_bias",
            kernels::FUSED_QKV_Q8_0_QWEN2_BIAS_SRC,
            "fused_qkv_q8_0_qwen2_bias",
        )?;

        let aq = a_q.buf.as_ptr();
        let ak = a_k.buf.as_ptr();
        let av = a_v.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yq = y_q.buf.as_ptr();
        let yk = y_k.buf.as_ptr();
        let yv = y_v.buf.as_ptr();
        let q_m_i = q_m as i32;
        let k_m_i = k_m as i32;
        let v_m_i = v_m as i32;
        let k_i = k as i32;
        let bq = bias_q_ptr;
        let bk = bias_k_ptr;
        let bv = bias_v_ptr;
        let total = (q_m + k_m + v_m) as u32;

        let mut params: Vec<*mut c_void> = vec![
            &aq as *const _ as *mut c_void,
            &ak as *const _ as *mut c_void,
            &av as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yq as *const _ as *mut c_void,
            &yk as *const _ as *mut c_void,
            &yv as *const _ as *mut c_void,
            &q_m_i as *const _ as *mut c_void,
            &k_m_i as *const _ as *mut c_void,
            &v_m_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
            &bq as *const _ as *mut c_void,
            &bk as *const _ as *mut c_void,
            &bv as *const _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "fused_qkv_q8_0_qwen2_bias",
            [total, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(aq);
                b.push_ptr(ak);
                b.push_ptr(av);
                b.push_ptr(xp);
                b.push_ptr(yq);
                b.push_ptr(yk);
                b.push_ptr(yv);
                b.push_i32(q_m_i);
                b.push_i32(k_m_i);
                b.push_i32(v_m_i);
                b.push_i32(k_i);
                b.push_ptr(bq);
                b.push_ptr(bk);
                b.push_ptr(bv);
                b
            },
        )
    }

    /// dp4a-port of fused_gate_up_hfq4g256 for gfx906. Pre-quantizes
    /// `x` to Q8_1 (block_q8_1_mmq, 144 B per 128-K block) using the
    /// shared MMQ x-scratch buffer, then runs the dp4a-based GEMV. Math
    /// is identical modulo Q8_1 quant noise (~1 % per-element relative).
    /// Targeted at gfx906 where the FP wave64 fused_gate_up sat at
    /// 41 % VALUBusy + 3.86 % MemUnitStalled — memory-bound, so dp4a's
    /// 75 % x-traffic reduction lands on the actual bottleneck.
    pub fn fused_gate_up_hfq4g256_dp4a(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Quantize x → Xq[K/128] block_q8_1_mmq via the existing shared
        // scratch path. Batch=1 for GEMV.
        let xq_ptr = self.ensure_q8_1_mmq_x(x, 1, k)?;

        self.ensure_kernel(
            "fused_gate_up_hfq4g256_wave64_dp4a",
            kernels::FUSED_GATE_UP_HFQ4G256_WAVE64_DP4A_SRC,
            "fused_gate_up_hfq4g256_wave64_dp4a",
        )?;

        let ag = a_gate.buf.as_ptr();
        let au = a_up.buf.as_ptr();
        let yg = y_gate.buf.as_ptr();
        let yu = y_up.buf.as_ptr();
        let gm = gate_m as i32;
        let um = up_m as i32;
        let kv = k as i32;
        let total = (gate_m + up_m) as u32;
        let mut xq = xq_ptr;
        let mut params: Vec<*mut c_void> = vec![
            &ag as *const _ as *mut c_void,
            &au as *const _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &yg as *const _ as *mut c_void,
            &yu as *const _ as *mut c_void,
            &gm as *const _ as *mut c_void,
            &um as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "fused_gate_up_hfq4g256_wave64_dp4a",
            [(total + 1) / 2, 1, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ag);
                b.push_ptr(au);
                b.push_ptr(xq);
                b.push_ptr(yg);
                b.push_ptr(yu);
                b.push_i32(gm);
                b.push_i32(um);
                b.push_i32(kv);
                b
            },
        )
    }

    /// Fused sigmoid(dn_beta) + alpha_gate(dn_alpha). Both ops are element-wise
    /// scalar transforms applied to independent buffers of size n_v_heads in the
    /// DeltaNet preamble. Saves one launch per linear-attention layer.
    #[cfg(feature = "deltanet")]
    pub fn fused_sigmoid_alpha_gate_f32(
        &mut self,
        beta: &GpuTensor,
        alpha: &GpuTensor,
        dt_bias: &GpuTensor,
        a_log: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_sigmoid_alpha_gate",
            kernels::FUSED_SIGMOID_ALPHA_GATE_SRC,
            "fused_sigmoid_alpha_gate_f32",
        )?;
        let bp = beta.buf.as_ptr();
        let ap = alpha.buf.as_ptr();
        let dp = dt_bias.buf.as_ptr();
        let lp = a_log.buf.as_ptr();
        let nn = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &bp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &lp as *const _ as *mut c_void,
            &nn as *const _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = n * 4 * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_sigmoid_alpha_gate_f32", bytes);
        let result = self.launch_maybe_blob(
            "fused_sigmoid_alpha_gate_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(bp);
                b.push_ptr(ap);
                b.push_ptr(dp);
                b.push_ptr(lp);
                b.push_i32(nn);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched `fused_sigmoid_alpha_gate_f32`. Grid.y is the batch dim.
    #[cfg(feature = "deltanet")]
    pub fn fused_sigmoid_alpha_gate_f32_batched(
        &mut self,
        beta: &GpuTensor,
        alpha: &GpuTensor,
        dt_bias: &GpuTensor,
        a_log: &GpuTensor,
        n: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_sigmoid_alpha_gate",
            kernels::FUSED_SIGMOID_ALPHA_GATE_SRC,
            "fused_sigmoid_alpha_gate_f32",
        )?;
        let mut bp = beta.buf.as_ptr();
        let mut ap = alpha.buf.as_ptr();
        let mut dp = dt_bias.buf.as_ptr();
        let mut lp = a_log.buf.as_ptr();
        let mut nn = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut bp as *mut _ as *mut c_void,
            &mut ap as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut lp as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = n * 4 * 4 * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_sigmoid_alpha_gate_f32_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "fused_sigmoid_alpha_gate_f32",
            [grid, batch_size as u32, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(bp);
                b.push_ptr(ap);
                b.push_ptr(dp);
                b.push_ptr(lp);
                b.push_i32(nn);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Fused L2-norm(Q) + L2-norm(K) + scale(Q). Replaces three back-to-back
    /// launches in DeltaNet's attention path with one — ~2 launches saved per
    /// linear-attention layer, so on Qwen3.5 (18-32 LA layers) we shave ~36-64
    /// launches per forward.
    #[cfg(feature = "deltanet")]
    pub fn fused_qk_l2_norm_scale_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        q_scale: f32,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_qk_l2_norm_scale",
            kernels::FUSED_QK_L2_NORM_SCALE_SRC,
            "fused_qk_l2_norm_scale_f32",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let qs = q_scale;
        let ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &qs as *const _ as *mut c_void,
            &ep as *const _ as *mut c_void,
        ];
        // Covers both Q and K reads/writes.
        let bytes = crate::profile::elementwise1_bytes(n_heads * head_dim) * 2;
        let timer =
            crate::profile::begin_timer(&self.hip, "fused", "fused_qk_l2_norm_scale_f32", bytes);
        let result = self.launch_maybe_blob(
            "fused_qk_l2_norm_scale_f32",
            [n_heads as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_f32(qs);
                b.push_f32(ep);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched `fused_qk_l2_norm_scale_f32`. Grid.y is the batch dim.
    #[cfg(feature = "deltanet")]
    pub fn fused_qk_l2_norm_scale_f32_batched(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        q_scale: f32,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_qk_l2_norm_scale",
            kernels::FUSED_QK_L2_NORM_SCALE_SRC,
            "fused_qk_l2_norm_scale_f32",
        )?;
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut qs = q_scale;
        let mut ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut qs as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::elementwise1_bytes(n_heads * head_dim) * 2 * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_qk_l2_norm_scale_f32_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "fused_qk_l2_norm_scale_f32",
            [n_heads as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_f32(qs);
                b.push_f32(ep);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Fused L2-norm(Q) + scale(Q) + L2-norm(K) + repeat-interleave(Q,K).
    /// Replaces fused_qk_l2_norm_scale_f32_batched +
    /// repeat_interleave_qk_f32_batched (2 launches → 1). Each block
    /// (key_head, batch) computes norms once and replicates across the
    /// `ratio` value-head slots. Used only when n_key_heads < n_v_heads.
    ///
    /// `q_src`/`k_src`: [N × n_key_heads × head_dim] (unchanged on exit).
    /// `q_dst`/`k_dst`: [N × n_value_heads × head_dim] (n_value = n_key*ratio).
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qk_l2_norm_scale_interleave_f32_batched(
        &mut self,
        q_src: &GpuTensor,
        k_src: &GpuTensor,
        q_dst: &GpuTensor,
        k_dst: &GpuTensor,
        n_key_heads: usize,
        ratio: usize,
        head_dim: usize,
        q_scale: f32,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "fused_qk_l2_norm_scale_interleave_f32_batched",
            kernels::FUSED_QK_L2_NORM_SCALE_INTERLEAVE_F32_BATCHED_SRC,
            "fused_qk_l2_norm_scale_interleave_f32_batched",
        )?;
        let qsp = q_src.buf.as_ptr();
        let ksp = k_src.buf.as_ptr();
        let qdp = q_dst.buf.as_ptr();
        let kdp = k_dst.buf.as_ptr();
        let nkh = n_key_heads as i32;
        let r_val = ratio as i32;
        let hd = head_dim as i32;
        let qs = q_scale;
        let ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &qsp as *const _ as *mut c_void,
            &ksp as *const _ as *mut c_void,
            &qdp as *const _ as *mut c_void,
            &kdp as *const _ as *mut c_void,
            &nkh as *const _ as *mut c_void,
            &r_val as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &qs as *const _ as *mut c_void,
            &ep as *const _ as *mut c_void,
        ];
        let bytes =
            crate::profile::elementwise1_bytes(n_key_heads * ratio * head_dim) * 2 * batch_size;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_qk_l2_norm_scale_interleave_f32_batched",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "fused_qk_l2_norm_scale_interleave_f32_batched",
            [n_key_heads as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qsp);
                b.push_ptr(ksp);
                b.push_ptr(qdp);
                b.push_ptr(kdp);
                b.push_i32(nkh);
                b.push_i32(r_val);
                b.push_i32(hd);
                b.push_f32(qs);
                b.push_f32(ep);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched GEMV (GEMM) for F16 weights: Y[M,N] = W_f16[M,K] @ X_f32[N,K]^T
    pub fn gemm_f16(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_f16", kernels::GEMM_F16_SRC, "gemm_f16")?;
        let func = &self.functions["gemm_f16"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, n as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// WMMA-accelerated batched GEMM for F16 weights × F32 activations (gfx1100+).
    /// Y[M,N] = W_f16[M,K] @ X_f32[N,K]^T.  Tiled 16×16 WMMA matrix multiply.
    /// Grid=[ceil(M/16), ceil(N/16)], Block=[32].  Replaces naive gemm_f16 for vision encoder.
    pub fn gemm_f16_wmma(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_f16_wmma", kernels::GEMM_F16_WMMA_SRC, "gemm_f16_wmma")?;
        let func = &self.functions["gemm_f16_wmma"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_n = ((n + 15) / 16) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_n, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Fused-transpose WMMA GEMM: Y[N,M] = W_f16[M,K] @ X_f32[N,K]^T.
    /// Writes transposed output directly with 4 N-subtiles per block.
    pub fn gemm_f16_wmma_mb4(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_f16_wmma_mb4",
            kernels::GEMM_F16_WMMA_MB4_SRC,
            "gemm_f16_wmma_mb4",
        )?;
        let func = &self.functions["gemm_f16_wmma_mb4"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_n = ((n + 63) / 64) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_n, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// MB=8 fused-transpose WMMA GEMM: 8 N-subtiles per block.
    pub fn gemm_f16_wmma_mb8(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kernel_name, kernel_src, symbol) = if self.arch_caps.has_wmma_w32_gfx12() {
            (
                "gemm_f16_wmma_mb8_gfx12",
                kernels::GEMM_F16_WMMA_MB8_GFX12_SRC,
                "gemm_f16_wmma_mb8_gfx12",
            )
        } else if self.arch_caps.has_wmma_w32() {
            (
                "gemm_f16_wmma_mb8",
                kernels::GEMM_F16_WMMA_MB8_SRC,
                "gemm_f16_wmma_mb8",
            )
        } else {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_f16_wmma_mb8 requires wave32 WMMA; arch={} does not support it.",
                    self.arch
                ),
            ));
        };
        self.ensure_kernel(kernel_name, kernel_src, symbol)?;
        let func = &self.functions[kernel_name];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_n = ((n + 127) / 128) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_n, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Tiled F16 GEMM — 8-way ILP unrolled, lane-coalesced stride-32 reads,
    /// no shared memory (high occupancy). Arbitrary M, K, N.
    /// Grid=[M, N], Block=[32], LDS=0.
    pub fn gemm_f16_tiled(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_f16_tiled",
            kernels::GEMM_F16_TILED_SRC,
            "gemm_f16_tiled",
        )?;
        let func = &self.functions["gemm_f16_tiled"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        // Same grid as naive: [M, N], block [32], no LDS
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, n as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Fused GEMM + bias: Y[N,M] = X[N,K] @ W_f16[M,K]^T + bias[M].
    /// Replaces gemm_f16 + transpose_f32 + bias_add_f32 (3 ops → 1).
    /// Grid=[N, 1], Block=[256].
    pub fn gemm_f16_bias(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        bias: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gemm_f16_bias", kernels::GEMM_F16_BIAS_SRC, "gemm_f16_bias")?;
        let func = &self.functions["gemm_f16_bias"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut bp = bias.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        // One block per row of X, 256 threads, no LDS
        unsafe {
            self.hip.launch_kernel(
                func,
                [n as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Batched GEMM for F32: Y[M,N] = A[M,K] @ B[N,K]^T
    pub fn gemm_f32_batched(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_f32_batched",
            kernels::GEMM_F32_SRC,
            "gemm_f32_batched",
        )?;
        let func = &self.functions["gemm_f32_batched"];
        let mut ap = a.buf.as_ptr();
        let mut bp = b.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, n as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// HFP4G32 / MFP4G32 grouped-WMMA-GEMM for MoE prefill — sister of
    /// `gemm_hfq4g256_moe_grouped_wmma_k2` but on the FP4G32 dequant. Same
    /// kernarg layout, same expert_tile_ids / sorted_slot_index contract.
    /// Tile (blockIdx.x, blockIdx.y) gathers row `sorted_slot_index[slot_start
    /// + m_lane]` (with -1 meaning "padding lane — zero B") and applies the
    /// weights of expert `expert_tile_ids[blockIdx.y]`. Sentinel `< 0` early-
    /// returns the tile so the dispatcher can launch up to m_total_max/16
    /// tiles without an m_total dtoh sync.
    ///
    /// `x_row_div` selects the X gather layout (same as HFQ4 sister):
    ///   gate_up: x_src = x_rot_batch [N × K], x_row_div = K_TOP
    ///   down:    x_src = rot_batch [N*K_TOP × K], x_row_div = 1
    /// `x_src_rows` is the number of rows in x_src (N or N*K_TOP).
    ///
    /// **gfx12 only.** gfx11 and older archs are not implemented and will
    /// panic — call sites should arch-gate before dispatching here.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_hfp4g32_moe_grouped_wmma(
        &mut self,
        expert_weight_ptrs: &GpuTensor, // [E] u64
        expert_tile_ids: &GpuTensor,    // [m_total / 16] i32
        sorted_slot_index: &GpuTensor,  // [m_total] i32
        x_src: &GpuTensor,              // [x_src_rows × K] f32 (auto-converted to FP16)
        y_grouped: &GpuTensor,          // [m_total × M] f32, written direct
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_rdna4() {
            panic!(
                "gemm_hfp4g32_moe_grouped_wmma: only gfx12 (RDNA4) is implemented; \
                 got arch={}. Add a wave32 sister kernel for non-gfx12 archs.",
                self.arch
            );
        }
        let kernel_name = "gemm_hfp4g32_moe_grouped_wmma_gfx12";
        let kernel_src = kernels::GEMM_HFP4G32_MOE_GROUPED_WMMA_GFX12_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW estimate: gather X (fp16) + weight rows (HFP4G32: 18 B/group, K/32 groups
        // per row, ~m_total/E shared per tile) + write Y. Use the existing helper.
        let bytes = m_total * k * 2 + (m_total * m) * 4 + crate::profile::gemv_hfp4g32_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    pub fn gemm_hfq4g256_lmhead_wmma_gfx12(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_lmhead_wmma_gfx12",
            kernels::GEMM_HFQ4G256_LMHEAD_WMMA_GFX12_SRC,
            "gemm_hfq4g256_lmhead_wmma_gfx12",
        )?;
        let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x_f16_ptr;
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

        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 15) / 16;

        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            "gemm_hfq4g256_lmhead_wmma_gfx12",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemm_hfq4g256_lmhead_wmma_gfx12",
            [row_tiles as u32, batch_tiles as u32, 1],
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
    pub fn gemm_q8_0_wmma_x64(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(k % 32, 0, "gemm_q8_0_wmma_x64: K must be %32");
        debug_assert_eq!(batch_size % 64, 0, "gemm_q8_0_wmma_x64: N must be %64");
        self.ensure_kernel(
            "gemm_q8_0_wmma_x64",
            kernels::GEMM_Q8_0_WMMA_X64_SRC,
            "gemm_q8_0_wmma_x64",
        )?;
        let xp_owned = x.buf.as_ptr();
        let mut xp = if matches!(x.dtype, DType::F16) {
            xp_owned
        } else {
            self.ensure_fp16_x(x, batch_size * k)?
        };
        let mut a_p = a.buf.as_ptr();
        let mut y_p = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_p as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut y_p as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];
        let row_tiles = (m + 15) / 16;
        let batch_tiles = (batch_size + 63) / 64;
        let bytes = m * (k / 32) * 34 + batch_size * k * 2 + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_q8_0_wmma_x64", bytes);
        let result = self.launch_maybe_blob(
            "gemm_q8_0_wmma_x64",
            [row_tiles as u32, batch_tiles as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_p);
                b.push_ptr(xp);
                b.push_ptr(y_p);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Exact-gfx1030 sdot4 MMQ Q8_0 probe. This is intentionally not selected
    /// by any production dispatch path: it exists only for direct channel
    /// validation while beta's model and Redline behavior remain immutable.
    pub fn gemm_q8_0_mmq_gfx1030(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1030() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_q8_0_mmq_gfx1030 requires exact gfx1030; current arch = {}",
                    self.arch
                ),
            ));
        }
        debug_assert_eq!(
            k % 128,
            0,
            "gemm_q8_0_mmq_gfx1030: K must be multiple of 128"
        );
        let kernel_name = "gemm_q8_0_mmq_gfx1030";
        self.ensure_kernel(kernel_name, kernels::GEMM_Q8_0_MMQ_GFX1030_SRC, kernel_name)?;
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;
        let ap = a.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let n_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
        ];
        let row_tiles = ((m + 31) / 32) as u32;
        let col_tiles = ((batch_size + 127) / 128) as u32;
        let bytes = m * (k / 32) * 34 + batch_size * (k / 128) * 144 + batch_size * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        // Probe-only raw HIP launch: never enter the Redline recording seam.
        let func = &self.functions[kernel_name];
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles, col_tiles, 1],
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

    /// Dense i8-WMMA MMQ GEMM for Q8_0 weights — 64x64 4-warp tile (gfx1151).
    /// Takes F32 `x` (auto-quantized to Q8_1), int8 WMMA at ~2x. Drop-in for
    /// `gemm_q8_0_wmma_4w`. Requires M%64==0, N%64==0, K%128==0.
    pub fn gemm_q8_0_mmq_4w_gfx1151(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch != "gfx1151" {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_q8_0_mmq_4w_gfx1151 requires gfx1151; current arch = {}",
                    self.arch
                ),
            ));
        }
        let kernel_name = "gemm_q8_0_mmq_4w_gfx1151";
        self.ensure_kernel(
            kernel_name,
            kernels::GEMM_Q8_0_MMQ_4W_GFX1151_SRC,
            kernel_name,
        )?;
        let x_q8_ptr = self.ensure_q8_1_mmq_x(x, batch_size, k)?;

        let ap = a.buf.as_ptr();
        let xp = x_q8_ptr;
        let yp = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let n_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
        ];
        let row_tiles = ((m + 63) / 64) as u32;
        let col_tiles = ((batch_size + 63) / 64) as u32;
        let bytes = (m * k) + (batch_size * m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, col_tiles, 1],
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    pub fn gemm_q8_0_wmma_4w(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(k % 32, 0, "gemm_q8_0_wmma_4w: K must be a multiple of 32");
        debug_assert_eq!(
            m % 64,
            0,
            "gemm_q8_0_wmma_4w: M must be a multiple of 64 (got {m})"
        );
        debug_assert_eq!(
            batch_size % 64,
            0,
            "gemm_q8_0_wmma_4w: N must be a multiple of 64 (got {batch_size})"
        );
        self.ensure_kernel(
            "gemm_q8_0_wmma_4w",
            kernels::GEMM_Q8_0_WMMA_4W_SRC,
            "gemm_q8_0_wmma_4w",
        )?;
        // Stage F32 → F16 input if needed.
        let xp_owned = x.buf.as_ptr();
        let mut xp = if matches!(x.dtype, DType::F16) {
            xp_owned
        } else {
            self.ensure_fp16_x(x, batch_size * k)?
        };

        let mut a_p = a.buf.as_ptr();
        let mut y_p = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_p as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut y_p as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];
        let row_tiles = (m + 63) / 64;
        let batch_tiles = (batch_size + 63) / 64;
        let func = &self.functions["gemm_q8_0_wmma_4w"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [row_tiles as u32, batch_tiles as u32, 1],
                [128, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn gemm_f16_x_f16_wmma(
        &mut self,
        a_f16: &GpuTensor,
        x_f16: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_f16_x_f16_wmma",
            kernels::GEMM_F16_X_F16_WMMA_SRC,
            "gemm_f16_x_f16_wmma",
        )?;
        let ap = a_f16.buf.as_ptr();
        let xp = x_f16.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        self.launch_maybe_blob(
            "gemm_f16_x_f16_wmma",
            [grid_m, grid_b, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mi);
                b.push_i32(ki);
                b.push_i32(bi);
                b
            },
        )
    }
    pub fn gemm_f32_register_tiled(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kname, src, batch_tile, block_x) = (
            "gemm_f32_register_tiled",
            kernels::GEMM_F32_REGISTER_TILED_SRC,
            8u32,
            32u32,
        );
        self.ensure_kernel(kname, src, kname)?;
        let func = &self.functions[kname];
        let mut ap = a.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let grid_y = (batch_size as u32 + batch_tile - 1) / batch_tile;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, grid_y, 1],
                [block_x, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn gemm_hfq4g256_wmma(
        &mut self,
        a_raw: &GpuTensor,
        x_f16: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gemm_hfq4g256_wmma",
            kernels::GEMM_HFQ4G256_WMMA_SRC,
            "gemm_hfq4g256_wmma",
        )?;
        let func = &self.functions["gemm_hfq4g256_wmma"];
        let ap = a_raw.buf.as_ptr();
        let xp = x_f16.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn gemm_mq2g256_lloyd_moe_grouped_wmma_k2(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_wmma_k2";
        let kernel_src = kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_K2_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 15) / 16) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW: MQ2-Lloyd weight is 72 B/group, half of HFQ4's 136 B/group.
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = m_total * k * 2 + (m_total * m) * 4 + mq2_weight_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            m % 64,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2: M must be a multiple of 64 (got {m})"
        );
        debug_assert_eq!(
            k % 256,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2: K must be a multiple of 256 (got {k})"
        );
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2";
        let kernel_src = kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        // Row tiles widen 16 → 64; slot tiles unchanged at 16.
        let row_tiles = ((m + 63) / 64) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        // BW unchanged from baseline: same total data movement, the
        // win is in B-fragment cache reuse across warps.
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = m_total * k * 2 + (m_total * m) * 4 + mq2_weight_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            m % 64,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload: M must be a multiple of 64 (got {m})"
        );
        debug_assert_eq!(
            k % 256,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload: K must be a multiple of 256 (got {k})"
        );
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload";
        let kernel_src = kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_MMQLOAD_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 63) / 64) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = m_total * k * 2 + (m_total * m) * 4 + mq2_weight_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload_nosync(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            m % 64,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload_nosync: M must be a multiple of 64 (got {m})"
        );
        debug_assert_eq!(
            k % 256,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload_nosync: K must be a multiple of 256 (got {k})"
        );
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload_nosync";
        let kernel_src = kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_MMQLOAD_NOSYNC_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 63) / 64) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = m_total * k * 2 + (m_total * m) * 4 + mq2_weight_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            m % 64,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32: M must be a multiple of 64 (got {m})"
        );
        debug_assert_eq!(
            k % 256,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32: K must be a multiple of 256 (got {k})"
        );
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32";
        let kernel_src = kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_N32_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 63) / 64) as u32;
        // Each block handles TWO 16-slot tiles → halve the slot-tile grid dim.
        let slot_tiles = (((m_total + 15) / 16 + 1) / 2) as u32;
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = m_total * k * 2 + (m_total * m) * 4 + mq2_weight_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_cnd(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            m % 64,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_cnd: M must be a multiple of 64 (got {m})"
        );
        debug_assert_eq!(
            k % 256,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_cnd: K must be a multiple of 256 (got {k})"
        );
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_cnd";
        let kernel_src = kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_4W_K2_CND_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 63) / 64) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = m_total * k * 2 + (m_total * m) * 4 + mq2_weight_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn gemm_mq2g256_lloyd_moe_grouped_wmma_8w_k2(
        &mut self,
        expert_weight_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        m_total: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            m % 64,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_8w_k2: M must be a multiple of 64 (got {m})"
        );
        debug_assert_eq!(
            k % 256,
            0,
            "gemm_mq2g256_lloyd_moe_grouped_wmma_8w_k2: K must be a multiple of 256 (got {k})"
        );
        let kernel_name = "gemm_mq2g256_lloyd_moe_grouped_wmma_8w_k2";
        let kernel_src = kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_8W_K2_SRC;
        self.ensure_kernel(kernel_name, kernel_src, kernel_name)?;
        let x_f16_ptr = self.ensure_fp16_x(x_src, x_src_rows * k)?;

        let ep = expert_weight_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_f16_ptr;
        let yp = y_grouped.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let xrd_val = x_row_div as i32;
        let mt_val = m_total as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &xrd_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];

        let row_tiles = ((m + 127) / 128) as u32;
        let slot_tiles = ((m_total + 15) / 16) as u32;
        let mq2_weight_bytes = m * (k / 256) * 72;
        let bytes = m_total * k * 2 + (m_total * m) * 4 + mq2_weight_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel_name, bytes);
        let result = self.launch_maybe_blob(
            kernel_name,
            [row_tiles, slot_tiles, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(xrd_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx942 BF16×BF16→FP32 MFMA GEMM — parent-checkpoint linear path.
    ///
    /// Computes `D[batch, M] = B[batch, K] @ A[M, K]^T` (i.e. `x @ W^T`) with
    /// BF16 operands and an FP32 accumulator, matching
    /// `kernels/src/gemm_bf16_mfma.gfx942.hip`.
    ///
    /// - `a`: row-major BF16 weights `[M, K]`
    /// - `b`: row-major BF16 activations `[batch, K]`
    /// - `d`: row-major FP32 output `[batch, M]` (overwritten)
    ///
    /// Launch geometry (fixed by the HIP source): block `[256,1,1]`,
    /// grid `[(M+31)/32, (batch+31)/32, 1]`, static LDS B-tile
    /// (`32 × 128 × 2 B`). Fail-closed on non-gfx942 and undersized buffers.
    pub fn gemm_bf16_mfma_gfx942(
        &mut self,
        a: &DeviceBuffer,
        b: &DeviceBuffer,
        d: &DeviceBuffer,
        m: usize,
        k: usize,
        batch: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx942() {
            return Err(hip_bridge::HipError::new(
                0,
                "gemm_bf16_mfma_gfx942 requires gfx942",
            ));
        }
        if m == 0 || k == 0 || batch == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_bf16_mfma_gfx942: M, K, batch must be positive (got M={m} K={k} batch={batch})"
                ),
            ));
        }
        let a_bytes = m
            .checked_mul(k)
            .and_then(|e| e.checked_mul(2))
            .ok_or_else(|| {
                hip_bridge::HipError::new(0, "gemm_bf16_mfma_gfx942: A size overflow")
            })?;
        let b_bytes = batch
            .checked_mul(k)
            .and_then(|e| e.checked_mul(2))
            .ok_or_else(|| {
                hip_bridge::HipError::new(0, "gemm_bf16_mfma_gfx942: B size overflow")
            })?;
        let d_bytes = batch
            .checked_mul(m)
            .and_then(|e| e.checked_mul(4))
            .ok_or_else(|| {
                hip_bridge::HipError::new(0, "gemm_bf16_mfma_gfx942: D size overflow")
            })?;
        if a.size() < a_bytes {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_bf16_mfma_gfx942: A buffer too small (have {} need {a_bytes})",
                    a.size()
                ),
            ));
        }
        if b.size() < b_bytes {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_bf16_mfma_gfx942: B buffer too small (have {} need {b_bytes})",
                    b.size()
                ),
            ));
        }
        if d.size() < d_bytes {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_bf16_mfma_gfx942: D buffer too small (have {} need {d_bytes})",
                    d.size()
                ),
            ));
        }

        const KERNEL: &str = "gemm_bf16_mfma_gfx942";
        self.ensure_kernel(KERNEL, kernels::GEMM_BF16_MFMA_GFX942_SRC, KERNEL)?;

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let d_ptr = d.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let batch_i32 = batch as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &b_ptr as *const _ as *mut c_void,
            &d_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
            &batch_i32 as *const _ as *mut c_void,
        ];
        let grid_x = ((m + 31) / 32) as u32;
        let grid_y = ((batch + 31) / 32) as u32;
        let bytes = a_bytes + b_bytes + d_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", KERNEL, bytes);
        let result = self.launch_maybe_blob(
            KERNEL,
            [grid_x, grid_y, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(a_ptr);
                blob.push_ptr(b_ptr);
                blob.push_ptr(d_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob.push_i32(batch_i32);
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// CDNA3 (gfx942) MFMA counterpart of [`Self::gemm_f16_x_f16_wmma`]:
    /// `Y[b, m] = Σ_k A[m, k] · X[b, k]` with F16 operands and an F32
    /// accumulator, matching `kernels/src/gemm_f16_x_f16_mfma.gfx942.hip`.
    ///
    /// The WMMA original is wave32-only and does not compile for gfx942, so
    /// this exists to give the DeepSeek V4 DSpark prefill path an F16 GEMM on
    /// MI300X. Identical signature, operand layout, and `[batch, M]` F32
    /// output layout as the WMMA version — call sites swap one for the other
    /// on `arch_caps.is_gfx942()`.
    ///
    /// - `a_f16`: row-major F16 weights `[M, K]`
    /// - `x_f16`: row-major F16 activations `[batch, K]`
    /// - `y_f32`: row-major F32 output `[batch, M]` (overwritten)
    ///
    /// Launch geometry (fixed by the HIP source): block `[64,1,1]` (wave64),
    /// grid `[(M+15)/16, (batch+15)/16, 1]`, no LDS. Fail-closed on non-gfx942,
    /// on `K % 16 != 0` (the kernel's K loop has no tail guard), and on
    /// undersized buffers.
    pub fn gemm_f16_x_f16_mfma_gfx942(
        &mut self,
        a_f16: &GpuTensor,
        x_f16: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx942() {
            return Err(hip_bridge::HipError::new(
                0,
                "gemm_f16_x_f16_mfma_gfx942 requires gfx942",
            ));
        }
        if m == 0 || k == 0 || batch_size == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_f16_x_f16_mfma_gfx942: M, K, batch must be positive (got M={m} K={k} batch={batch_size})"
                ),
            ));
        }
        if k % 16 != 0 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("gemm_f16_x_f16_mfma_gfx942: K must be a multiple of 16 (got K={k})"),
            ));
        }
        let a_bytes = m
            .checked_mul(k)
            .and_then(|e| e.checked_mul(2))
            .ok_or_else(|| {
                hip_bridge::HipError::new(0, "gemm_f16_x_f16_mfma_gfx942: A size overflow")
            })?;
        let x_bytes = batch_size
            .checked_mul(k)
            .and_then(|e| e.checked_mul(2))
            .ok_or_else(|| {
                hip_bridge::HipError::new(0, "gemm_f16_x_f16_mfma_gfx942: X size overflow")
            })?;
        let y_bytes = batch_size
            .checked_mul(m)
            .and_then(|e| e.checked_mul(4))
            .ok_or_else(|| {
                hip_bridge::HipError::new(0, "gemm_f16_x_f16_mfma_gfx942: Y size overflow")
            })?;
        if a_f16.buf.size() < a_bytes {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_f16_x_f16_mfma_gfx942: A buffer too small (have {} need {a_bytes})",
                    a_f16.buf.size()
                ),
            ));
        }
        if x_f16.buf.size() < x_bytes {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_f16_x_f16_mfma_gfx942: X buffer too small (have {} need {x_bytes})",
                    x_f16.buf.size()
                ),
            ));
        }
        if y_f32.buf.size() < y_bytes {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemm_f16_x_f16_mfma_gfx942: Y buffer too small (have {} need {y_bytes})",
                    y_f32.buf.size()
                ),
            ));
        }

        const KERNEL: &str = "gemm_f16_x_f16_mfma_gfx942";
        self.ensure_kernel(KERNEL, kernels::GEMM_F16_X_F16_MFMA_GFX942_SRC, KERNEL)?;

        let a_ptr = a_f16.buf.as_ptr();
        let x_ptr = x_f16.buf.as_ptr();
        let y_ptr = y_f32.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let batch_i32 = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
            &batch_i32 as *const _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        let bytes = a_bytes + x_bytes + y_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", KERNEL, bytes);
        let result = self.launch_maybe_blob(
            KERNEL,
            [grid_m, grid_b, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(a_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(y_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob.push_i32(batch_i32);
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// rocBLAS BF16×BF16→FP32 reference for the same orientation as
    /// [`Self::gemm_bf16_mfma_gfx942`]: `Y[N,M] = X[N,K] @ W[M,K]^T`.
    ///
    /// Returns `Err` when rocBLAS is not loaded. Used only as a numerical /
    /// throughput oracle for the hand-rolled MFMA kernel — not a production
    /// parent-linear path.
    pub fn rocblas_gemm_bf16_prefill(
        &self,
        w_bf16: &DeviceBuffer, // row-major [M × K]
        x_bf16: &DeviceBuffer, // row-major [N × K]
        y_fp32: &DeviceBuffer, // row-major [N × M]
        m: usize,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        use hip_bridge::{RocblasDatatype, RocblasOperation};
        let rb = self.rocblas.as_ref().ok_or_else(|| {
            hip_bridge::HipError::new(0, "rocblas_gemm_bf16_prefill: rocBLAS not initialized")
        })?;
        if let Some(stream) = self.active_stream.as_ref() {
            rb.set_stream(stream).map_err(|e| {
                hip_bridge::HipError::new(
                    e.status,
                    &format!("rocblas_set_stream for bf16 gemm: {}", e.context),
                )
            })?;
        }
        let alpha = 1.0f32;
        let beta = 0.0f32;
        unsafe {
            rb.gemm_ex(
                RocblasOperation::Transpose,
                RocblasOperation::None,
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const f32 as *const c_void,
                w_bf16.as_ptr(),
                RocblasDatatype::Bf16,
                k as i32,
                x_bf16.as_ptr(),
                RocblasDatatype::Bf16,
                k as i32,
                &beta as *const f32 as *const c_void,
                y_fp32.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                y_fp32.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                RocblasDatatype::F32,
            )
            .map_err(|e| {
                hip_bridge::HipError::new(e.status, &format!("rocblas_gemm_bf16: {}", e.context))
            })
        }
    }
}
