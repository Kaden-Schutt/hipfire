// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
use crate::context::DispatchCtx;
use crate::tables::KernelRegistry;
use crate::traits::KernelFamily;
use crate::types::*;
use rdna_compute::{Gpu, GpuTensor};

pub struct FusedQkvParams<'a> {
    pub kind: KernelKey,
    pub weights: &'a [&'a GpuTensor],
    pub x: &'a GpuTensor,
    pub outputs: &'a [&'a GpuTensor],
    pub m: &'a [usize],
    pub k: usize,
    /// Rotation scratch buffers for Paro fused-kernel dispatch.
    /// 4 × [k] F32 buffers for QKVZA (all 4) and 3-way QKV (first 3 + aliased 4th);
    /// for gate+up, only [0] is used as `x_rot_gate` (the kernel aliases `mq_x_rot`
    /// for `x_rot_up` internally). Empty slice for non-Paro keys; existing arms
    /// ignore it.
    pub rot_scratch: &'a [GpuTensor],
    /// Batched-prefill row count (`#397 Ship 5.2 slice 2`). `None` = single-token
    /// DECODE: gate+up arms dispatch to the `gpu.fused_gate_up_*` kernels (the
    /// historical behavior; the decode pipeline in `pipeline::steps` passes
    /// `None`). `Some(n)` = batched PREFILL: the 2-way gate+up arms instead
    /// dispatch to the batched `gpu.gemm_gate_up_*(.., n)` kernels — the IDENTICAL
    /// methods the qwen35 prefill call sites used directly — preserving each
    /// method's internal arch routing byte-for-byte. Only the gate+up arms read
    /// this field; QKV / QKVZA / Paro arms ignore it.
    pub batch_size: Option<usize>,
}

pub struct FusedQkvFamily {
    registry: KernelRegistry,
}

impl FusedQkvFamily {
    pub fn new() -> Self {
        let mut registry = KernelRegistry::new();
        super::super::tables::fused_qkv_table::populate(&mut registry);
        registry.validate().expect("fused_qkv kernel table has empty entries");
        Self { registry }
    }

    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }

    pub fn resolve(
        &self,
        key: KernelKey,
        ctx: &DispatchCtx,
        shape: Option<&ShapeInfo>,
    ) -> Result<&KernelVariant, DispatchError> {
        self.registry.resolve(key, ctx, shape)
    }

    pub fn run(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        params: &FusedQkvParams,
    ) -> Result<(), DispatchError> {
        self.resolve(params.kind, ctx, None)?;
        dispatch_fused_qkv(gpu, params)
    }
}

impl KernelFamily for FusedQkvFamily {
    fn name(&self) -> &'static str {
        "fused_qkv"
    }
}

macro_rules! hip {
    ($e:expr) => {
        $e.map_err(|e| DispatchError::Hip(e.to_string()))
    };
}

fn dispatch_fused_qkv(gpu: &mut Gpu, params: &FusedQkvParams) -> Result<(), DispatchError> {
    let x = params.x;
    let k = params.k;
    match params.kind {
        // ── 3-way Fused QKV ────────────────────────────────────
        //
        // Each arm is batch-aware via `params.batch_size` (mirrors the gate+up
        // arms below):
        //   None    → single-token DECODE → `gpu.fused_qkv_*` (historical;
        //             the decode pipeline in `pipeline::steps` passes `None`).
        //   Some(n) → batched PREFILL    → `gpu.gemm_qkv_*(.., n)`, the IDENTICAL
        //             batched method the qwen35 prefill call site used directly;
        //             each method keeps its own internal arch routing byte-for-byte.
        // `#397 Ship 5.2 slice 3` migrates the qwen35 prefill QKV sites onto the
        // `Some(n)` paths.
        KernelKey::FusedQkvHfq4G256 => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] = <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkv_hfq4g256(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n)),
                None => hip!(gpu.fused_qkv_hfq4g256(wq, wk, wv, x, q, kout, v, mq, mk, mv, k)),
            }
        }
        KernelKey::FusedQkvMq3G256Lloyd => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] = <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            match params.batch_size {
                // Prefill mq3-lloyd is WMMA-only (`gemm_qkv_mq3g256_lloyd_wmma`);
                // arch_required=HasWmma gates the entry.
                Some(n) => hip!(gpu.gemm_qkv_mq3g256_lloyd_wmma(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n)),
                None => hip!(gpu.fused_qkv_mq3g256_lloyd(wq, wk, wv, x, q, kout, v, mq, mk, mv, k)),
            }
        }
        KernelKey::FusedQkvMq4G256Lloyd => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] = <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkv_mq4g256_lloyd_wmma(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n)),
                None => hip!(gpu.fused_qkv_mq4g256_lloyd(wq, wk, wv, x, q, kout, v, mq, mk, mv, k)),
            }
        }
        KernelKey::FusedQkvHfq6G256 => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] = <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkv_hfq6g256(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n)),
                None if gpu.arch_caps.gemv_dp4a_enabled() => hip!(gpu.fused_qkv_hfq6g256_dp4a(wq, wk, wv, x, q, kout, v, mq, mk, mv, k)),
                None => hip!(gpu.gemm_qkv_hfq6g256(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, 1)),
            }
        }
        KernelKey::FusedQkvQ4K => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] = <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            hip!(gpu.fused_qkv_q4k(wq, wk, wv, x, q, kout, v, mq, mk, mv, k))
        }
        // ── Q8_0 fused QKV — prefill-only key (#397 Ship 5.2 slice 3) ──
        // No decode `fused_qkv_q8_0` exists; this key is batched-prefill only and
        // WMMA-only (entry gated HasWmma). `gpu.gemm_qkv_q8_0_wmma` routes the
        // gfx12 WMMA sibling on RDNA4 internally; no scalar fallback. The qwen35
        // non-WMMA arch case stays as three plain GemmQ8_0BatchedChunked GEMMs.
        KernelKey::FusedQkvQ8_0 => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] = <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let n = params.batch_size.ok_or(DispatchError::UnsupportedVariant {
                family: "fused_qkv", variant: "qkv", arch: "", quant: "q8_0 (prefill-only)",
            })?;
            hip!(gpu.gemm_qkv_q8_0_wmma(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n))
        }
        // ── HFQ3G256 fused QKV — prefill-only key (#397 Ship 5.2 slice 3) ──
        // No decode `fused_qkv_hfq3g256` exists; batched-prefill only. The qwen35
        // site picks `gemm_qkv_hfq3g256_wmma` on has_wmma() archs else the base
        // `gemm_qkv_hfq3g256` (full cross-arch ladder). We mirror that arch split
        // here so the same kernel runs (cf. FusedGateUpHfq3G256).
        KernelKey::FusedQkvHfq3G256 => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] = <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let n = params.batch_size.ok_or(DispatchError::UnsupportedVariant {
                family: "fused_qkv", variant: "qkv", arch: "", quant: "hfq3g256 (prefill-only)",
            })?;
            if gpu.arch_caps.has_wmma() {
                hip!(gpu.gemm_qkv_hfq3g256_wmma(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n))
            } else {
                hip!(gpu.gemm_qkv_hfq3g256(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n))
            }
        }
        // ── HFP4G32 fused QKV — prefill-only key (#397 Ship 5.2 FINAL) ──
        // WMMA-only (entry gated HasWmma); no decode `fused_qkv_hfp4g32` exists.
        // `gpu.gemm_qkv_hfp4g32` routes the gfx12 FP8/WMMA siblings on RDNA4 else
        // the gfx11 `_wmma` kernel internally; no scalar fallback. Mirrors the
        // FusedGateUpHfp4G32 arm.
        KernelKey::FusedQkvHfp4G32 => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] = <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let n = params.batch_size.ok_or(DispatchError::UnsupportedVariant {
                family: "fused_qkv", variant: "qkv", arch: "", quant: "hfp4g32 (prefill-only)",
            })?;
            hip!(gpu.gemm_qkv_hfp4g32(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n))
        }

        // ── 4-way Fused QKVZA (DeltaNet linear attention) ────
        //
        // Batch-aware via `params.batch_size` (same scheme as 3-way QKV):
        //   None    → DECODE  → `gpu.fused_qkvza_*` (historical).
        //   Some(n) → PREFILL → `gpu.gemm_qkvza_*(.., n)`, the IDENTICAL batched
        //             method the qwen35 prefill call site used directly.
        KernelKey::FusedQkvzaHfq4G256 => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] = <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkvza_hfq4g256(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k, n)),
                None => hip!(gpu.fused_qkvza_hfq4g256(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k)),
            }
        }
        KernelKey::FusedQkvzaMq3G256Lloyd => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] = <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkvza_mq3g256_lloyd_wmma(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k, n)),
                None => hip!(gpu.fused_qkvza_mq3g256_lloyd(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k)),
            }
        }
        KernelKey::FusedQkvzaMq4G256Lloyd => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] = <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkvza_mq4g256_lloyd_wmma(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k, n)),
                None => hip!(gpu.fused_qkvza_mq4g256_lloyd(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k)),
            }
        }
        KernelKey::FusedQkvzaHfq6G256 => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] = <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            match params.batch_size {
                // Batched prefill: cross-arch ladder (wmma_gfx12/wmma/dp4a/dot2/fp16/scalar).
                Some(n) => hip!(gpu.gemm_qkvza_hfq6g256(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k, n)),
                // Decode (n=1): gfx906 dp4a fused fast-path; cross-arch gemm (n=1,
                // scalar base) elsewhere so RDNA/CDNA decode doesn't hit the
                // gfx906-only dp4a kernel.
                None if gpu.arch_caps.gemv_dp4a_enabled() => hip!(gpu.fused_qkvza_hfq6g256_dp4a(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k)),
                None => hip!(gpu.gemm_qkvza_hfq6g256(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k, 1)),
            }
        }
        // ── Q8_0 fused QKVZA — prefill-only key (#397 Ship 5.2 slice 3) ──
        // WMMA-only (entry gated HasWmma); no decode `fused_qkvza_q8_0` exists.
        KernelKey::FusedQkvzaQ8_0 => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] = <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let n = params.batch_size.ok_or(DispatchError::UnsupportedVariant {
                family: "fused_qkv", variant: "qkvza", arch: "", quant: "q8_0 (prefill-only)",
            })?;
            hip!(gpu.gemm_qkvza_q8_0_wmma(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k, n))
        }
        // ── HFQ3G256 fused QKVZA — prefill-only key (#397 Ship 5.2 slice 3) ──
        // Arch-split mirror of the qwen35 call site (WMMA on has_wmma() else base
        // cross-arch ladder). No decode method exists.
        KernelKey::FusedQkvzaHfq3G256 => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] = <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let n = params.batch_size.ok_or(DispatchError::UnsupportedVariant {
                family: "fused_qkv", variant: "qkvza", arch: "", quant: "hfq3g256 (prefill-only)",
            })?;
            if gpu.arch_caps.has_wmma() {
                hip!(gpu.gemm_qkvza_hfq3g256_wmma(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k, n))
            } else {
                hip!(gpu.gemm_qkvza_hfq3g256(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k, n))
            }
        }
        // ── HFP4G32 fused QKVZA — prefill-only key (#397 Ship 5.2 FINAL) ──
        // WMMA-only (entry gated HasWmma); no decode `fused_qkvza_hfp4g32` exists.
        // `gpu.gemm_qkvza_hfp4g32` routes the gfx12 WMMA sibling on RDNA4 else the
        // gfx11 `_wmma` kernel internally; no scalar fallback. Mirrors FusedQkvHfp4G32.
        KernelKey::FusedQkvzaHfp4G32 => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] = <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let n = params.batch_size.ok_or(DispatchError::UnsupportedVariant {
                family: "fused_qkv", variant: "qkvza", arch: "", quant: "hfp4g32 (prefill-only)",
            })?;
            hip!(gpu.gemm_qkvza_hfp4g32(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k, n))
        }

        // ── 2-way Fused Gate+Up (FFN) ────────────────────────
        //
        // Each arm is batch-aware via `params.batch_size`:
        //   None      → single-token DECODE → `gpu.fused_gate_up_*` (historical).
        //   Some(n)   → batched PREFILL    → `gpu.gemm_gate_up_*(.., n)`, the
        //               IDENTICAL batched method the qwen35 prefill call site
        //               used; each method keeps its own internal arch routing.
        // `#397 Ship 5.2 slice 2` migrates the qwen35 prefill gate+up sites onto
        // the `Some(n)` paths.
        KernelKey::FusedGateUpHfq4G256 => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] = <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            match params.batch_size {
                Some(n) => hip!(gpu.gemm_gate_up_hfq4g256(w_gate, w_up, x, gate, up, mg, mu, k, n)),
                None => hip!(gpu.fused_gate_up_hfq4g256(w_gate, w_up, x, gate, up, mg, mu, k)),
            }
        }
        KernelKey::FusedGateUpMq3G256Lloyd => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] = <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            match params.batch_size {
                // Prefill mq3-lloyd is WMMA-only (`gemm_gate_up_mq3g256_lloyd_wmma`,
                // routed for_arch over RDNA3/RDNA4); arch_required=HasWmma gates entry.
                Some(n) => hip!(gpu.gemm_gate_up_mq3g256_lloyd_wmma(w_gate, w_up, x, gate, up, mg, mu, k, n)),
                None => hip!(gpu.fused_gate_up_mq3g256_lloyd(w_gate, w_up, x, gate, up, mg, mu, k)),
            }
        }
        KernelKey::FusedGateUpMq4G256Lloyd => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] = <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            hip!(gpu.fused_gate_up_mq4g256_lloyd(w_gate, w_up, x, gate, up, mg, mu, k))
        }
        KernelKey::FusedGateUpHfq6G256 => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] = <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            match params.batch_size {
                Some(n) => hip!(gpu.gemm_gate_up_hfq6g256(w_gate, w_up, x, gate, up, mg, mu, k, n)),
                None if gpu.arch_caps.gemv_dp4a_enabled() => hip!(gpu.fused_gate_up_hfq6g256_dp4a(w_gate, w_up, x, gate, up, mg, mu, k)),
                None => hip!(gpu.gemm_gate_up_hfq6g256(w_gate, w_up, x, gate, up, mg, mu, k, 1)),
            }
        }
        KernelKey::FusedGateUpQ4K => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] = <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            hip!(gpu.fused_gate_up_q4k(w_gate, w_up, x, gate, up, mg, mu, k))
        }
        KernelKey::FusedGateUpQ8_0 => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] = <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            match params.batch_size {
                // Prefill Q8 gate+up routes ONLY the WMMA arch case here
                // (`gemm_gate_up_q8_0_wmma`); the non-WMMA arch case stays as two
                // plain GemmQ8_0BatchedChunked GEMMs at the call site (slice 1).
                Some(n) => hip!(gpu.gemm_gate_up_q8_0_wmma(w_gate, w_up, x, gate, up, mg, mu, k, n)),
                None => hip!(gpu.fused_gate_up_q8_0(w_gate, w_up, x, gate, up, mg, mu, k)),
            }
        }
        // ── HFQ3G256 gate+up — prefill-only key (#397 Ship 5.2 slice 2) ──
        // No decode `fused_gate_up_hfq3g256` exists; this key is batched-prefill
        // only. The qwen35 site picks `gemm_gate_up_hfq3g256_wmma` on has_wmma()
        // archs else the base `gemm_gate_up_hfq3g256` (full cross-arch ladder).
        // We mirror that arch split here so the same kernel runs.
        KernelKey::FusedGateUpHfq3G256 => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] = <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let n = params.batch_size.ok_or(DispatchError::UnsupportedVariant {
                family: "fused_qkv", variant: "gate_up", arch: "", quant: "hfq3g256 (prefill-only)",
            })?;
            if gpu.arch_caps.has_wmma() {
                hip!(gpu.gemm_gate_up_hfq3g256_wmma(w_gate, w_up, x, gate, up, mg, mu, k, n))
            } else {
                hip!(gpu.gemm_gate_up_hfq3g256(w_gate, w_up, x, gate, up, mg, mu, k, n))
            }
        }
        // ── HFP4G32 gate+up — prefill-only key (#397 Ship 5.2 slice 2) ──
        // WMMA-only (entry gated HasWmma): `gemm_gate_up_hfp4g32` internally
        // routes gfx12 vs gfx11 WMMA siblings; no scalar fallback exists.
        KernelKey::FusedGateUpHfp4G32 => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] = <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let n = params.batch_size.ok_or(DispatchError::UnsupportedVariant {
                family: "fused_qkv", variant: "gate_up", arch: "", quant: "hfp4g32 (prefill-only)",
            })?;
            hip!(gpu.gemm_gate_up_hfp4g32(w_gate, w_up, x, gate, up, mg, mu, k, n))
        }

        // ── Paro fused Paro4G128T (dp4a) ────────────────────────────────
        // Gate+up: 1 explicit rotation scratch buffer (x_rot_gate) + kernel
        // internal mq_x_rot as x_rot_up. The kernel asserts mq_x_rot >= k
        // and x_rot_gate != mq_x_rot.
        KernelKey::FusedGateUpParo4G128T => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] = <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let rs = params.rot_scratch;
            assert!(rs.len() >= 1, "FusedGateUpParo4G128T needs >= 1 rotation scratch buffer, got {}", rs.len());
            assert!(mg % 8 == 0 && k % 128 == 0,
                "FusedGateUpParo4G128T requires m%8==0 and k%128==0, got m={} k={}", mg, k);
            hip!(gpu.fused_gate_up_paro4g128t(w_gate, w_up, x, gate, up, &rs[0], mg, k))
        }
        // QKVZA: 4 explicit rotation scratch buffers.
        KernelKey::FusedQkvzaParo4G128T => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] = <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let rs = params.rot_scratch;
            assert!(rs.len() >= 4, "FusedQkvzaParo4G128T needs >= 4 rotation scratch buffers, got {}", rs.len());
            for (label, m) in [("mqkv", mqkv), ("mz", mz), ("mbeta", mbeta), ("malpha", malpha)] {
                assert!(m % 8 == 0, "FusedQkvzaParo4G128T {} requires m%8==0, got {}", label, m);
            }
            assert!(k % 128 == 0, "FusedQkvzaParo4G128T requires k%128==0, got {}", k);
            hip!(gpu.fused_qkvza_paro4g128t(
                wqkv, wz, w_beta, w_alpha, x,
                qkv, z, beta, alpha,
                &rs[0], &rs[1], &rs[2], &rs[3],
                mqkv, mz, mbeta, malpha, k))
        }
        // QKV 3-way (FullAttn): synthesised via the 4-way kernel with m3=0.
        // a3/y3/x_rot3 are aliased to a0/y0/rs[0] — the kernel skips the 4th
        // projection because m3=0 guarantees no 4th write.
        KernelKey::FusedQkvParo4G128T => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] = <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let rs = params.rot_scratch;
            assert!(rs.len() >= 4, "FusedQkvParo4G128T needs >= 4 rotation scratch buffers (4th aliased for m3=0), got {}", rs.len());
            assert!(mq % 8 == 0 && mk % 8 == 0 && mv % 8 == 0,
                "FusedQkvParo4G128T requires m%8==0, got mq={}, mk={}, mv={}", mq, mk, mv);
            assert!(k % 128 == 0, "FusedQkvParo4G128T requires k%128==0, got {}", k);
            hip!(gpu.fused_qkvza_paro4g128t(
                wq, wk, wv, wq,  // a3 = wq (aliased)
                x,
                q, kout, v, q,   // y3 = q (aliased)
                &rs[0], &rs[1], &rs[2], &rs[0], // x_rot3 = rs[0] (aliased, unused)
                mq, mk, mv, 0,   // m3 = 0
                k))
        }
        _ => Err(DispatchError::UnsupportedVariant {
            family: "fused_qkv",
            variant: "",
            arch: "",
            quant: "",
        }),
    }
}

/// Build the dispatch error for a fused-projection call whose operand arity
/// (weights / outputs / m) did not match the kernel's expectation. The kernel
/// key already names the quant tier; we additionally report the fused-projection
/// *family* (qkv / qkvza / gate_up) so the diagnostic distinguishes a 3-way QKV
/// arity mismatch from a 4-way QKVZA or 2-way Gate+Up one (the three families
/// expect 3 / 4 / 2 operands respectively). `expected` is the operand count the
/// kernel arm tried to destructure into.
fn err_wrong_arity(kind: KernelKey, expected: usize) -> DispatchError {
    match fused_qkv_variant_for_key(kind) {
        Some(variant) => {
            let _ = expected; // family implies arity (qkv=3, qkvza=4, gate_up=2)
            let label = match variant {
                FusedQkvVariant::Qkv | FusedQkvVariant::QkvParo => "qkv",
                FusedQkvVariant::Qkvza | FusedQkvVariant::QkvzaParo => "qkvza",
                FusedQkvVariant::GateUp | FusedQkvVariant::GateUpParo => "gate_up",
            };
            DispatchError::UnsupportedVariant {
                family: "fused_qkv",
                variant: label,
                arch: "",
                quant: "",
            }
        }
        // Not a fused-projection key (should be unreachable from this family) —
        // fall back to the bare missing-impl report rather than mislabel it.
        None => DispatchError::MissingImpl { key: kind },
    }
}
