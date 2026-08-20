// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
use crate::context::DispatchCtx;
use crate::tables::KernelRegistry;
use crate::traits::KernelFamily;
use crate::types::*;
use rdna_compute::{DType, Gpu, GpuTensor};

fn is_fused_hfq4_key(key: KernelKey) -> bool {
    matches!(
        key,
        KernelKey::FusedQkvHfq4G256
            | KernelKey::FusedQkvzaHfq4G256
            | KernelKey::FusedGateUpHfq4G256
    )
}

fn is_fused_mq4v2_key(key: KernelKey) -> bool {
    matches!(
        key,
        KernelKey::FusedQkvMq4G256V2
            | KernelKey::FusedQkvzaMq4G256V2
            | KernelKey::FusedGateUpMq4G256V2
    )
}

fn is_fused_mq4cg256_key(key: KernelKey) -> bool {
    matches!(
        key,
        KernelKey::FusedQkvMq4CG256
            | KernelKey::FusedQkvzaMq4CG256
            | KernelKey::FusedGateUpMq4CG256
    )
}

fn is_fused_v2_key(key: KernelKey) -> bool {
    matches!(
        key,
        KernelKey::FusedQkvMq4G256V2
            | KernelKey::FusedQkvMq5G256V2
            | KernelKey::FusedQkvMq6G256V2
            | KernelKey::FusedQkvMq3G256V2
            | KernelKey::FusedQkvMq2G256V2
            | KernelKey::FusedQkvMq4CG256
            | KernelKey::FusedQkvzaMq4G256V2
            | KernelKey::FusedQkvzaMq5G256V2
            | KernelKey::FusedQkvzaMq6G256V2
            | KernelKey::FusedQkvzaMq3G256V2
            | KernelKey::FusedQkvzaMq2G256V2
            | KernelKey::FusedQkvzaMq4CG256
            | KernelKey::FusedGateUpMq4G256V2
            | KernelKey::FusedGateUpMq5G256V2
            | KernelKey::FusedGateUpMq6G256V2
            | KernelKey::FusedGateUpMq3G256V2
            | KernelKey::FusedGateUpMq2G256V2
            | KernelKey::FusedGateUpMq4CG256
    )
}

fn is_v2_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::MQ4G256V2
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256V2
            | DType::MQ2G256V2
            | DType::MQ4CG256
    )
}

fn guard_fused_qkv_dtype_key(weights: &[&GpuTensor], key: KernelKey) -> Result<(), DispatchError> {
    let is_v1 = is_fused_hfq4_key(key);
    let is_v2 = is_fused_v2_key(key);
    let is_mq4c = is_fused_mq4cg256_key(key);
    let is_mq4v2 = is_fused_mq4v2_key(key);
    for (idx, w) in weights.iter().enumerate() {
        let w_is_v2 = is_v2_dtype(w.dtype);
        let w_is_mq4v2 = w.dtype == DType::MQ4G256V2;
        if w_is_v2 && is_v1 {
            return Err(DispatchError::Hip(format!(
                "qt V2 weight[{}] (dtype {:?}) routed to v1 kernel key {:?}:                  v2 stores fp16 scale/zero per 128 weights (s0/z0 for 0..127, s1/z1 for 128..255)                  where v1 stores f32 scale/zero per 256, so the v1 kernel decodes every weight                  to ~1e-14. This is a missing v2 routing arm at the callsite, not a valid configuration.",
                idx, w.dtype, key
            )));
        }
        if (w.dtype == DType::HFQ4G256 || w.dtype == DType::MQ4G256) && is_v2 {
            return Err(DispatchError::Hip(format!(
                "v1 weight[{}] (dtype {:?}) routed to v2 kernel key {:?}:                  v2 expects fp16 s0/z0/s1/z1 per 128 weights while v1 stores f32 scale/zero per 256.                  Routing a v1 payload through a v2 kernel is equally wrong and silent.",
                idx, w.dtype, key
            )));
        }
        if w_is_v2 && is_v2 {
            let ok = match w.dtype {
                DType::MQ4G256V2 => matches!(
                    key,
                    KernelKey::FusedQkvMq4G256V2
                        | KernelKey::FusedQkvzaMq4G256V2
                        | KernelKey::FusedGateUpMq4G256V2
                ),
                DType::MQ6G256V2 => matches!(
                    key,
                    KernelKey::FusedQkvMq6G256V2
                        | KernelKey::FusedQkvzaMq6G256V2
                        | KernelKey::FusedGateUpMq6G256V2
                ),
                DType::MQ5G256V2 => matches!(
                    key,
                    KernelKey::FusedQkvMq5G256V2
                        | KernelKey::FusedQkvzaMq5G256V2
                        | KernelKey::FusedGateUpMq5G256V2
                ),
                DType::MQ3G256V2 => matches!(
                    key,
                    KernelKey::FusedQkvMq3G256V2
                        | KernelKey::FusedQkvzaMq3G256V2
                        | KernelKey::FusedGateUpMq3G256V2
                ),
                DType::MQ2G256V2 => matches!(
                    key,
                    KernelKey::FusedQkvMq2G256V2
                        | KernelKey::FusedQkvzaMq2G256V2
                        | KernelKey::FusedGateUpMq2G256V2
                ),
                DType::MQ4CG256 => is_mq4c,
                _ => false,
            };
            if !ok {
                return Err(DispatchError::Hip(format!(
                    "V2 weight[{}] (dtype {:?}) routed to incompatible V2 kernel key {:?}: cross-V2 group bytes differ (MQ6=200, MQ5=168, MQ4=136, MQ3=104, MQ2=72, MQ4C=136 packed) — mis-route decodes every group wrong.",
                    idx, w.dtype, key
                )));
            }
        }
        if (w.dtype == DType::HFQ4G256 || w.dtype == DType::MQ4G256 || w_is_mq4v2) && is_mq4c {
            return Err(DispatchError::Hip(format!(
                "non-mq4c weight[{}] (dtype {:?}) routed to mq4c kernel key {:?}:                  mq4c expects 136 B groups (packed fp16 scale/zero dword + 4 B pad + 128 B                  nibbles); v1/v2 payloads are also 136 B/group but with different headers, so                  equal stride is not interchangeable and would decode every group wrong.",
                idx, w.dtype, key
            )));
        }
    }
    Ok(())
}
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

/// Qwen2-only decode parameters for the Q/K/V-bias fold. This is deliberately
/// separate from [`FusedQkvParams`]: Qwen3+ and Redline retain their original
/// dispatch shape and kernel ABI.
pub struct FusedQkvBiasParams<'a> {
    pub kind: KernelKey,
    pub weights: [&'a GpuTensor; 3],
    pub x: &'a GpuTensor,
    pub outputs: [&'a GpuTensor; 3],
    pub m: [usize; 3],
    pub k: usize,
    pub bias: [&'a GpuTensor; 3],
}

pub struct FusedQkvFamily {
    registry: KernelRegistry,
}

impl FusedQkvFamily {
    pub fn new() -> Self {
        let mut registry = KernelRegistry::new();
        super::super::tables::fused_qkv_table::populate(&mut registry);
        registry
            .validate()
            .expect("fused_qkv kernel table has empty entries");
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

    pub fn run_with_qwen2_bias(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        params: &FusedQkvBiasParams,
    ) -> Result<(), DispatchError> {
        self.resolve(params.kind, ctx, None)?;
        dispatch_fused_qkv_with_qwen2_bias(gpu, params)
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
    // Guard: never route V2 bytes through a v1 kernel or vice-versa.
    guard_fused_qkv_dtype_key(params.weights, params.kind)?;
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
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] =
                <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wq, x, __cap_n, k);
            gpu.maybe_capture_activation(wk, x, __cap_n, k);
            gpu.maybe_capture_activation(wv, x, __cap_n, k);

            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkv_hfq4g256(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n)),
                None => hip!(gpu.fused_qkv_hfq4g256(wq, wk, wv, x, q, kout, v, mq, mk, mv, k)),
            }
        }
        KernelKey::FusedQkvMq4G256V2 => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] =
                <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wq, x, __cap_n, k);
            gpu.maybe_capture_activation(wk, x, __cap_n, k);
            gpu.maybe_capture_activation(wv, x, __cap_n, k);

            match params.batch_size {
                Some(n) => {
                    hip!(gpu.gemm_qkv_hfq4g256_mq4v2(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n))
                }
                None => {
                    hip!(gpu.fused_qkv_hfq4g256_mq4v2(wq, wk, wv, x, q, kout, v, mq, mk, mv, k))
                }
            }
        }
        KernelKey::FusedQkvMq6G256V2 => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] =
                <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wq, x, __cap_n, k);
            gpu.maybe_capture_activation(wk, x, __cap_n, k);
            gpu.maybe_capture_activation(wv, x, __cap_n, k);
            match params.batch_size {
                Some(n) => {
                    hip!(gpu.gemm_qkv_mq6g256v2_wmma(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n))
                }
                None => hip!(gpu.fused_qkv_mq6g256v2(wq, wk, wv, x, q, kout, v, mq, mk, mv, k)),
            }
        }
        KernelKey::FusedQkvMq5G256V2 => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] =
                <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wq, x, __cap_n, k);
            gpu.maybe_capture_activation(wk, x, __cap_n, k);
            gpu.maybe_capture_activation(wv, x, __cap_n, k);
            match params.batch_size {
                Some(n) => {
                    hip!(gpu.gemm_qkv_mq5g256v2_wmma(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n))
                }
                None => hip!(gpu.fused_qkv_mq5g256v2(wq, wk, wv, x, q, kout, v, mq, mk, mv, k)),
            }
        }
        KernelKey::FusedQkvMq3G256V2 => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] =
                <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wq, x, __cap_n, k);
            gpu.maybe_capture_activation(wk, x, __cap_n, k);
            gpu.maybe_capture_activation(wv, x, __cap_n, k);
            match params.batch_size {
                Some(n) => {
                    hip!(gpu.gemm_qkv_mq3g256v2_wmma(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n))
                }
                None => hip!(gpu.fused_qkv_mq3g256v2(wq, wk, wv, x, q, kout, v, mq, mk, mv, k)),
            }
        }
        KernelKey::FusedQkvMq2G256V2 => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] =
                <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wq, x, __cap_n, k);
            gpu.maybe_capture_activation(wk, x, __cap_n, k);
            gpu.maybe_capture_activation(wv, x, __cap_n, k);
            match params.batch_size {
                Some(n) => {
                    hip!(gpu.gemm_qkv_mq2g256v2_wmma(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n))
                }
                None => hip!(gpu.fused_qkv_mq2g256v2(wq, wk, wv, x, q, kout, v, mq, mk, mv, k)),
            }
        }
        KernelKey::FusedQkvMq4CG256 => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] =
                <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wq, x, __cap_n, k);
            gpu.maybe_capture_activation(wk, x, __cap_n, k);
            gpu.maybe_capture_activation(wv, x, __cap_n, k);

            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkv_mq4cg256(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n)),
                None => hip!(gpu.fused_qkv_mq4cg256(wq, wk, wv, x, q, kout, v, mq, mk, mv, k)),
            }
        }
        KernelKey::FusedQkvMq3G256Lloyd => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] =
                <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wq, x, __cap_n, k);
            gpu.maybe_capture_activation(wk, x, __cap_n, k);
            gpu.maybe_capture_activation(wv, x, __cap_n, k);

            match params.batch_size {
                // Prefill mq3-lloyd is WMMA-only (`gemm_qkv_mq3g256_lloyd_wmma`);
                // arch_required=HasWmma gates the entry.
                Some(n) => {
                    hip!(gpu
                        .gemm_qkv_mq3g256_lloyd_wmma(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n))
                }
                None => hip!(gpu.fused_qkv_mq3g256_lloyd(wq, wk, wv, x, q, kout, v, mq, mk, mv, k)),
            }
        }
        KernelKey::FusedQkvMq4G256Lloyd => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] =
                <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wq, x, __cap_n, k);
            gpu.maybe_capture_activation(wk, x, __cap_n, k);
            gpu.maybe_capture_activation(wv, x, __cap_n, k);

            match params.batch_size {
                Some(n) => {
                    hip!(gpu
                        .gemm_qkv_mq4g256_lloyd_wmma(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n))
                }
                None => hip!(gpu.fused_qkv_mq4g256_lloyd(wq, wk, wv, x, q, kout, v, mq, mk, mv, k)),
            }
        }
        KernelKey::FusedQkvHfq6G256 => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] =
                <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wq, x, __cap_n, k);
            gpu.maybe_capture_activation(wk, x, __cap_n, k);
            gpu.maybe_capture_activation(wv, x, __cap_n, k);

            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkv_hfq6g256(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n)),
                None if gpu.arch_caps.gemv_dp4a_enabled() => {
                    hip!(gpu.fused_qkv_hfq6g256_dp4a(wq, wk, wv, x, q, kout, v, mq, mk, mv, k))
                }
                None => hip!(gpu.gemm_qkv_hfq6g256(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, 1)),
            }
        }
        KernelKey::FusedQkvQ4K => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] =
                <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            // Calibration taps: one per constituent weight, shared input x (decode, n=1).
            gpu.maybe_capture_activation(wq, x, 1, k);
            gpu.maybe_capture_activation(wk, x, 1, k);
            gpu.maybe_capture_activation(wv, x, 1, k);

            hip!(gpu.fused_qkv_q4k(wq, wk, wv, x, q, kout, v, mq, mk, mv, k))
        }
        // ── Q8_0 fused QKV ──
        // None (decode, n=1): scalar `fused_qkv_q8_0` — cross-arch, no dp4a needed.
        // Some(n) (prefill):  WMMA-only `gemm_qkv_q8_0_wmma`; the qwen35 non-WMMA
        //   arch case stays as three plain GemmQ8_0BatchedChunked GEMMs at the call
        //   site. `gpu.gemm_qkv_q8_0_wmma` routes gfx12 WMMA sibling internally.
        KernelKey::FusedQkvQ8_0 => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] =
                <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wq, x, __cap_n, k);
            gpu.maybe_capture_activation(wk, x, __cap_n, k);
            gpu.maybe_capture_activation(wv, x, __cap_n, k);

            match params.batch_size {
                Some(n) => {
                    hip!(gpu.gemm_qkv_q8_0_wmma(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n))
                }
                None => hip!(gpu.fused_qkv_q8_0(wq, wk, wv, x, q, kout, v, mq, mk, mv, k)),
            }
        }
        // ── HFQ3G256 fused QKV — prefill-only key (#397 Ship 5.2 slice 3) ──
        // No decode `fused_qkv_hfq3g256` exists; batched-prefill only. The qwen35
        // site picks `gemm_qkv_hfq3g256_wmma` on has_wmma() archs else the base
        // `gemm_qkv_hfq3g256` (full cross-arch ladder). We mirror that arch split
        // here so the same kernel runs (cf. FusedGateUpHfq3G256).
        KernelKey::FusedQkvHfq3G256 => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] =
                <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let n = params.batch_size.ok_or(DispatchError::UnsupportedVariant {
                family: "fused_qkv",
                variant: "qkv",
                arch: "",
                quant: "hfq3g256 (prefill-only)",
            })?;
            gpu.maybe_capture_activation(wq, x, n, k);
            gpu.maybe_capture_activation(wk, x, n, k);
            gpu.maybe_capture_activation(wv, x, n, k);

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
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] =
                <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let n = params.batch_size.ok_or(DispatchError::UnsupportedVariant {
                family: "fused_qkv",
                variant: "qkv",
                arch: "",
                quant: "hfp4g32 (prefill-only)",
            })?;
            gpu.maybe_capture_activation(wq, x, n, k);
            gpu.maybe_capture_activation(wk, x, n, k);
            gpu.maybe_capture_activation(wv, x, n, k);

            hip!(gpu.gemm_qkv_hfp4g32(wq, wk, wv, x, q, kout, v, mq, mk, mv, k, n))
        }

        // ── 4-way Fused QKVZA (DeltaNet linear attention) ────
        //
        // Batch-aware via `params.batch_size` (same scheme as 3-way QKV):
        //   None    → DECODE  → `gpu.fused_qkvza_*` (historical).
        //   Some(n) → PREFILL → `gpu.gemm_qkvza_*(.., n)`, the IDENTICAL batched
        //             method the qwen35 prefill call site used directly.
        KernelKey::FusedQkvzaHfq4G256 => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] =
                <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wqkv, x, __cap_n, k);
            gpu.maybe_capture_activation(wz, x, __cap_n, k);
            gpu.maybe_capture_activation(w_beta, x, __cap_n, k);
            gpu.maybe_capture_activation(w_alpha, x, __cap_n, k);

            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkvza_hfq4g256(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k,
                    n
                )),
                None => hip!(gpu.fused_qkvza_hfq4g256(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k
                )),
            }
        }
        KernelKey::FusedQkvzaMq4G256V2 => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] =
                <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wqkv, x, __cap_n, k);
            gpu.maybe_capture_activation(wz, x, __cap_n, k);
            gpu.maybe_capture_activation(w_beta, x, __cap_n, k);
            gpu.maybe_capture_activation(w_alpha, x, __cap_n, k);

            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkvza_hfq4g256_mq4v2(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k,
                    n
                )),
                None => hip!(gpu.fused_qkvza_hfq4g256_mq4v2(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k
                )),
            }
        }
        KernelKey::FusedQkvzaMq5G256V2 => {
            let [wqkv, wz, wb, wa] = <[&GpuTensor; 4]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mb, ma] =
                <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wqkv, x, __cap_n, k);
            gpu.maybe_capture_activation(wz, x, __cap_n, k);
            gpu.maybe_capture_activation(wb, x, __cap_n, k);
            gpu.maybe_capture_activation(wa, x, __cap_n, k);
            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkvza_mq5g256v2_wmma(
                    wqkv, wz, wb, wa, x, qkv, z, beta, alpha, mqkv, mz, mb, ma, k, n
                )),
                None => hip!(gpu.fused_qkvza_mq5g256v2(
                    wqkv, wz, wb, wa, x, qkv, z, beta, alpha, mqkv, mz, mb, ma, k
                )),
            }
        }
        KernelKey::FusedQkvzaMq6G256V2 => {
            let [wqkv, wz, wb, wa] = <[&GpuTensor; 4]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mb, ma] =
                <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wqkv, x, __cap_n, k);
            gpu.maybe_capture_activation(wz, x, __cap_n, k);
            gpu.maybe_capture_activation(wb, x, __cap_n, k);
            gpu.maybe_capture_activation(wa, x, __cap_n, k);
            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkvza_mq6g256v2_wmma(
                    wqkv, wz, wb, wa, x, qkv, z, beta, alpha, mqkv, mz, mb, ma, k, n
                )),
                None => hip!(gpu.fused_qkvza_mq6g256v2(
                    wqkv, wz, wb, wa, x, qkv, z, beta, alpha, mqkv, mz, mb, ma, k
                )),
            }
        }
        KernelKey::FusedQkvzaMq3G256V2 => {
            let [wqkv, wz, wb, wa] = <[&GpuTensor; 4]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mb, ma] =
                <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wqkv, x, __cap_n, k);
            gpu.maybe_capture_activation(wz, x, __cap_n, k);
            gpu.maybe_capture_activation(wb, x, __cap_n, k);
            gpu.maybe_capture_activation(wa, x, __cap_n, k);
            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkvza_mq3g256v2_wmma(
                    wqkv, wz, wb, wa, x, qkv, z, beta, alpha, mqkv, mz, mb, ma, k, n
                )),
                None => hip!(gpu.fused_qkvza_mq3g256v2(
                    wqkv, wz, wb, wa, x, qkv, z, beta, alpha, mqkv, mz, mb, ma, k
                )),
            }
        }
        KernelKey::FusedQkvzaMq2G256V2 => {
            let [wqkv, wz, wb, wa] = <[&GpuTensor; 4]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mb, ma] =
                <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wqkv, x, __cap_n, k);
            gpu.maybe_capture_activation(wz, x, __cap_n, k);
            gpu.maybe_capture_activation(wb, x, __cap_n, k);
            gpu.maybe_capture_activation(wa, x, __cap_n, k);
            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkvza_mq2g256v2_wmma(
                    wqkv, wz, wb, wa, x, qkv, z, beta, alpha, mqkv, mz, mb, ma, k, n
                )),
                None => hip!(gpu.fused_qkvza_mq2g256v2(
                    wqkv, wz, wb, wa, x, qkv, z, beta, alpha, mqkv, mz, mb, ma, k
                )),
            }
        }
        KernelKey::FusedQkvzaMq4CG256 => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] =
                <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wqkv, x, __cap_n, k);
            gpu.maybe_capture_activation(wz, x, __cap_n, k);
            gpu.maybe_capture_activation(w_beta, x, __cap_n, k);
            gpu.maybe_capture_activation(w_alpha, x, __cap_n, k);

            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkvza_mq4cg256(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k,
                    n
                )),
                None => hip!(gpu.fused_qkvza_mq4cg256(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k
                )),
            }
        }
        KernelKey::FusedQkvzaMq3G256Lloyd => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] =
                <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wqkv, x, __cap_n, k);
            gpu.maybe_capture_activation(wz, x, __cap_n, k);
            gpu.maybe_capture_activation(w_beta, x, __cap_n, k);
            gpu.maybe_capture_activation(w_alpha, x, __cap_n, k);

            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkvza_mq3g256_lloyd_wmma(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k,
                    n
                )),
                None => hip!(gpu.fused_qkvza_mq3g256_lloyd(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k
                )),
            }
        }
        KernelKey::FusedQkvzaMq4G256Lloyd => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] =
                <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wqkv, x, __cap_n, k);
            gpu.maybe_capture_activation(wz, x, __cap_n, k);
            gpu.maybe_capture_activation(w_beta, x, __cap_n, k);
            gpu.maybe_capture_activation(w_alpha, x, __cap_n, k);

            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkvza_mq4g256_lloyd_wmma(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k,
                    n
                )),
                None => hip!(gpu.fused_qkvza_mq4g256_lloyd(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k
                )),
            }
        }
        KernelKey::FusedQkvzaHfq6G256 => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] =
                <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wqkv, x, __cap_n, k);
            gpu.maybe_capture_activation(wz, x, __cap_n, k);
            gpu.maybe_capture_activation(w_beta, x, __cap_n, k);
            gpu.maybe_capture_activation(w_alpha, x, __cap_n, k);

            match params.batch_size {
                // Batched prefill: cross-arch ladder (wmma_gfx12/wmma/dp4a/dot2/fp16/scalar).
                Some(n) => hip!(gpu.gemm_qkvza_hfq6g256(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k,
                    n
                )),
                // Decode (n=1): gfx906 dp4a fused fast-path; cross-arch gemm (n=1,
                // scalar base) elsewhere so RDNA/CDNA decode doesn't hit the
                // gfx906-only dp4a kernel.
                None if gpu.arch_caps.gemv_dp4a_enabled() => hip!(gpu.fused_qkvza_hfq6g256_dp4a(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k
                )),
                None => hip!(gpu.gemm_qkvza_hfq6g256(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k,
                    1
                )),
            }
        }
        // ── Q8_0 fused QKVZA ──
        // None (decode, n=1): scalar `fused_qkvza_q8_0` — cross-arch, no dp4a.
        //   Added 2026-06-14: Qwen3.5-A3B .mq4p has Q8_0 linear-attention projections.
        // Some(n) (prefill):  WMMA-only `gemm_qkvza_q8_0_wmma`; entry gated HasWmma.
        KernelKey::FusedQkvzaQ8_0 => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] =
                <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wqkv, x, __cap_n, k);
            gpu.maybe_capture_activation(wz, x, __cap_n, k);
            gpu.maybe_capture_activation(w_beta, x, __cap_n, k);
            gpu.maybe_capture_activation(w_alpha, x, __cap_n, k);

            match params.batch_size {
                Some(n) => hip!(gpu.gemm_qkvza_q8_0_wmma(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k,
                    n
                )),
                None => hip!(gpu.fused_qkvza_q8_0(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k
                )),
            }
        }
        // ── HFQ3G256 fused QKVZA — prefill-only key (#397 Ship 5.2 slice 3) ──
        // Arch-split mirror of the qwen35 call site (WMMA on has_wmma() else base
        // cross-arch ladder). No decode method exists.
        KernelKey::FusedQkvzaHfq3G256 => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] =
                <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let n = params.batch_size.ok_or(DispatchError::UnsupportedVariant {
                family: "fused_qkv",
                variant: "qkvza",
                arch: "",
                quant: "hfq3g256 (prefill-only)",
            })?;
            gpu.maybe_capture_activation(wqkv, x, n, k);
            gpu.maybe_capture_activation(wz, x, n, k);
            gpu.maybe_capture_activation(w_beta, x, n, k);
            gpu.maybe_capture_activation(w_alpha, x, n, k);

            if gpu.arch_caps.has_wmma() {
                hip!(gpu.gemm_qkvza_hfq3g256_wmma(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k,
                    n
                ))
            } else {
                hip!(gpu.gemm_qkvza_hfq3g256(
                    wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k,
                    n
                ))
            }
        }
        // ── HFP4G32 fused QKVZA — prefill-only key (#397 Ship 5.2 FINAL) ──
        // WMMA-only (entry gated HasWmma); no decode `fused_qkvza_hfp4g32` exists.
        // `gpu.gemm_qkvza_hfp4g32` routes the gfx12 WMMA sibling on RDNA4 else the
        // gfx11 `_wmma` kernel internally; no scalar fallback. Mirrors FusedQkvHfp4G32.
        KernelKey::FusedQkvzaHfp4G32 => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] =
                <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let n = params.batch_size.ok_or(DispatchError::UnsupportedVariant {
                family: "fused_qkv",
                variant: "qkvza",
                arch: "",
                quant: "hfp4g32 (prefill-only)",
            })?;
            gpu.maybe_capture_activation(wqkv, x, n, k);
            gpu.maybe_capture_activation(wz, x, n, k);
            gpu.maybe_capture_activation(w_beta, x, n, k);
            gpu.maybe_capture_activation(w_alpha, x, n, k);

            hip!(gpu.gemm_qkvza_hfp4g32(
                wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k, n
            ))
        }
        // MFP4G32E8 fused QKVZA — DECODE-ONLY (gfx1151 launch-fusion). The
        // producing guard (`guard_qkvza_mfp4g32e8`) only fires in execute_steps
        // decode, so batch_size is always None here; a Some(n) would mean a
        // prefill site wrongly emitted this key (no such site exists).
        KernelKey::FusedQkvzaMfp4G32E8 => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] =
                <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            if params.batch_size.is_some() {
                return Err(DispatchError::UnsupportedVariant {
                    family: "fused_qkv",
                    variant: "qkvza",
                    arch: "gfx1151",
                    quant: "mfp4g32e8 (decode-only)",
                });
            }
            // Calibration taps: one per constituent weight, shared input x (decode, n=1).
            gpu.maybe_capture_activation(wqkv, x, 1, k);
            gpu.maybe_capture_activation(wz, x, 1, k);
            gpu.maybe_capture_activation(w_beta, x, 1, k);
            gpu.maybe_capture_activation(w_alpha, x, 1, k);

            hip!(gpu.fused_qkvza_mfp4g32_e8(
                wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k
            ))
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
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] =
                <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(w_gate, x, __cap_n, k);
            gpu.maybe_capture_activation(w_up, x, __cap_n, k);

            match params.batch_size {
                Some(n) => hip!(gpu.gemm_gate_up_hfq4g256(w_gate, w_up, x, gate, up, mg, mu, k, n)),
                None => hip!(gpu.fused_gate_up_hfq4g256(w_gate, w_up, x, gate, up, mg, mu, k)),
            }
        }
        KernelKey::FusedGateUpMq4G256V2 => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] =
                <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(w_gate, x, __cap_n, k);
            gpu.maybe_capture_activation(w_up, x, __cap_n, k);

            match params.batch_size {
                Some(n) => {
                    hip!(gpu.gemm_gate_up_hfq4g256_mq4v2(w_gate, w_up, x, gate, up, mg, mu, k, n))
                }
                None => {
                    hip!(gpu.fused_gate_up_hfq4g256_mq4v2(w_gate, w_up, x, gate, up, mg, mu, k))
                }
            }
        }
        KernelKey::FusedGateUpMq5G256V2 => {
            let [wg, wu] = <[&GpuTensor; 2]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [g, u] = <[&GpuTensor; 2]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] =
                <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wg, x, __cap_n, k);
            gpu.maybe_capture_activation(wu, x, __cap_n, k);
            match params.batch_size {
                Some(n) => {
                    hip!(gpu.gemm_gate_up_mq5g256v2_wmma(wg, wu, x, g, u, mg, mu, k, n))
                }
                None => hip!(gpu.fused_gate_up_mq5g256v2(wg, wu, x, g, u, mg, mu, k)),
            }
        }
        KernelKey::FusedGateUpMq6G256V2 => {
            let [wg, wu] = <[&GpuTensor; 2]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [g, u] = <[&GpuTensor; 2]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] =
                <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wg, x, __cap_n, k);
            gpu.maybe_capture_activation(wu, x, __cap_n, k);
            match params.batch_size {
                Some(n) => {
                    hip!(gpu.gemm_gate_up_mq6g256v2_wmma(wg, wu, x, g, u, mg, mu, k, n))
                }
                None => hip!(gpu.fused_gate_up_mq6g256v2(wg, wu, x, g, u, mg, mu, k)),
            }
        }
        KernelKey::FusedGateUpMq3G256V2 => {
            let [wg, wu] = <[&GpuTensor; 2]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [g, u] = <[&GpuTensor; 2]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] =
                <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wg, x, __cap_n, k);
            gpu.maybe_capture_activation(wu, x, __cap_n, k);
            match params.batch_size {
                Some(n) => {
                    hip!(gpu.gemm_gate_up_mq3g256v2_wmma(wg, wu, x, g, u, mg, mu, k, n))
                }
                None => hip!(gpu.fused_gate_up_mq3g256v2(wg, wu, x, g, u, mg, mu, k)),
            }
        }
        KernelKey::FusedGateUpMq2G256V2 => {
            let [wg, wu] = <[&GpuTensor; 2]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [g, u] = <[&GpuTensor; 2]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] =
                <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(wg, x, __cap_n, k);
            gpu.maybe_capture_activation(wu, x, __cap_n, k);
            match params.batch_size {
                Some(n) => {
                    hip!(gpu.gemm_gate_up_mq2g256v2_wmma(wg, wu, x, g, u, mg, mu, k, n))
                }
                None => hip!(gpu.fused_gate_up_mq2g256v2(wg, wu, x, g, u, mg, mu, k)),
            }
        }
        KernelKey::FusedGateUpMq4CG256 => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] =
                <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(w_gate, x, __cap_n, k);
            gpu.maybe_capture_activation(w_up, x, __cap_n, k);

            match params.batch_size {
                Some(n) => hip!(gpu.gemm_gate_up_mq4cg256(w_gate, w_up, x, gate, up, mg, mu, k, n)),
                None => hip!(gpu.fused_gate_up_mq4cg256(w_gate, w_up, x, gate, up, mg, mu, k)),
            }
        }
        // MFP4G32E8 fused gate+up — DECODE-ONLY (gfx1151 launch-fusion). Same
        // rationale as FusedQkvzaMfp4G32E8: guard only fires in decode.
        KernelKey::FusedGateUpMfp4G32E8 => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] =
                <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            if params.batch_size.is_some() {
                return Err(DispatchError::UnsupportedVariant {
                    family: "fused_qkv",
                    variant: "gate_up",
                    arch: "gfx1151",
                    quant: "mfp4g32e8 (decode-only)",
                });
            }
            // Calibration taps: one per constituent weight, shared input x (decode, n=1).
            gpu.maybe_capture_activation(w_gate, x, 1, k);
            gpu.maybe_capture_activation(w_up, x, 1, k);

            hip!(gpu.fused_gate_up_mfp4g32_e8(w_gate, w_up, x, gate, up, mg, mu, k))
        }
        KernelKey::FusedGateUpMq3G256Lloyd => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] =
                <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(w_gate, x, __cap_n, k);
            gpu.maybe_capture_activation(w_up, x, __cap_n, k);

            match params.batch_size {
                // Prefill mq3-lloyd is WMMA-only (`gemm_gate_up_mq3g256_lloyd_wmma`,
                // routed for_arch over RDNA3/RDNA4); arch_required=HasWmma gates entry.
                Some(n) => {
                    hip!(gpu
                        .gemm_gate_up_mq3g256_lloyd_wmma(w_gate, w_up, x, gate, up, mg, mu, k, n))
                }
                None => hip!(gpu.fused_gate_up_mq3g256_lloyd(w_gate, w_up, x, gate, up, mg, mu, k)),
            }
        }
        KernelKey::FusedGateUpMq4G256Lloyd => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] =
                <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            // Calibration taps: one per constituent weight, shared input x (decode, n=1).
            gpu.maybe_capture_activation(w_gate, x, 1, k);
            gpu.maybe_capture_activation(w_up, x, 1, k);

            hip!(gpu.fused_gate_up_mq4g256_lloyd(w_gate, w_up, x, gate, up, mg, mu, k))
        }
        KernelKey::FusedGateUpHfq6G256 => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] =
                <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(w_gate, x, __cap_n, k);
            gpu.maybe_capture_activation(w_up, x, __cap_n, k);

            match params.batch_size {
                Some(n) => hip!(gpu.gemm_gate_up_hfq6g256(w_gate, w_up, x, gate, up, mg, mu, k, n)),
                None if gpu.arch_caps.gemv_dp4a_enabled() => {
                    hip!(gpu.fused_gate_up_hfq6g256_dp4a(w_gate, w_up, x, gate, up, mg, mu, k))
                }
                None => hip!(gpu.gemm_gate_up_hfq6g256(w_gate, w_up, x, gate, up, mg, mu, k, 1)),
            }
        }
        KernelKey::FusedGateUpQ4K => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] =
                <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            // Calibration taps: one per constituent weight, shared input x (decode, n=1).
            gpu.maybe_capture_activation(w_gate, x, 1, k);
            gpu.maybe_capture_activation(w_up, x, 1, k);

            hip!(gpu.fused_gate_up_q4k(w_gate, w_up, x, gate, up, mg, mu, k))
        }
        KernelKey::FusedGateUpQ8_0 => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] =
                <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            // Calibration taps: one per constituent weight, shared input x.
            let __cap_n = params.batch_size.unwrap_or(1);
            gpu.maybe_capture_activation(w_gate, x, __cap_n, k);
            gpu.maybe_capture_activation(w_up, x, __cap_n, k);

            match params.batch_size {
                // Prefill Q8 gate+up routes ONLY the WMMA arch case here
                // (`gemm_gate_up_q8_0_wmma`); the non-WMMA arch case stays as two
                // plain GemmQ8_0BatchedChunked GEMMs at the call site (slice 1).
                Some(n) => {
                    hip!(gpu.gemm_gate_up_q8_0_wmma(w_gate, w_up, x, gate, up, mg, mu, k, n))
                }
                None => hip!(gpu.fused_gate_up_q8_0(w_gate, w_up, x, gate, up, mg, mu, k)),
            }
        }
        // ── HFQ3G256 gate+up — prefill-only key (#397 Ship 5.2 slice 2) ──
        // No decode `fused_gate_up_hfq3g256` exists; this key is batched-prefill
        // only. The qwen35 site picks `gemm_gate_up_hfq3g256_wmma` on has_wmma()
        // archs else the base `gemm_gate_up_hfq3g256` (full cross-arch ladder).
        // We mirror that arch split here so the same kernel runs.
        KernelKey::FusedGateUpHfq3G256 => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] =
                <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let n = params.batch_size.ok_or(DispatchError::UnsupportedVariant {
                family: "fused_qkv",
                variant: "gate_up",
                arch: "",
                quant: "hfq3g256 (prefill-only)",
            })?;
            gpu.maybe_capture_activation(w_gate, x, n, k);
            gpu.maybe_capture_activation(w_up, x, n, k);

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
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] =
                <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let n = params.batch_size.ok_or(DispatchError::UnsupportedVariant {
                family: "fused_qkv",
                variant: "gate_up",
                arch: "",
                quant: "hfp4g32 (prefill-only)",
            })?;
            gpu.maybe_capture_activation(w_gate, x, n, k);
            gpu.maybe_capture_activation(w_up, x, n, k);

            hip!(gpu.gemm_gate_up_hfp4g32(w_gate, w_up, x, gate, up, mg, mu, k, n))
        }

        // ── Paro fused Paro4G128T (dp4a) ────────────────────────────────
        // Gate+up: 1 explicit rotation scratch buffer (x_rot_gate) + kernel
        // internal mq_x_rot as x_rot_up. The kernel asserts mq_x_rot >= k
        // and x_rot_gate != mq_x_rot.
        KernelKey::FusedGateUpParo4G128T => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] =
                <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let rs = params.rot_scratch;
            assert!(
                rs.len() >= 1,
                "FusedGateUpParo4G128T needs >= 1 rotation scratch buffer, got {}",
                rs.len()
            );
            assert!(
                mg % 8 == 0 && k % 128 == 0,
                "FusedGateUpParo4G128T requires m%8==0 and k%128==0, got m={} k={}",
                mg,
                k
            );
            // Calibration taps: one per constituent weight, shared input x (decode, n=1).
            gpu.maybe_capture_activation(w_gate, x, 1, k);
            gpu.maybe_capture_activation(w_up, x, 1, k);

            hip!(gpu.fused_gate_up_paro4g128t(w_gate, w_up, x, gate, up, &rs[0], mg, k))
        }
        // QKVZA: 4 explicit rotation scratch buffers.
        KernelKey::FusedQkvzaParo4G128T => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] =
                <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let rs = params.rot_scratch;
            assert!(
                rs.len() >= 4,
                "FusedQkvzaParo4G128T needs >= 4 rotation scratch buffers, got {}",
                rs.len()
            );
            for (label, m) in [
                ("mqkv", mqkv),
                ("mz", mz),
                ("mbeta", mbeta),
                ("malpha", malpha),
            ] {
                assert!(
                    m % 8 == 0,
                    "FusedQkvzaParo4G128T {} requires m%8==0, got {}",
                    label,
                    m
                );
            }
            assert!(
                k % 128 == 0,
                "FusedQkvzaParo4G128T requires k%128==0, got {}",
                k
            );
            // Calibration taps: one per constituent weight, shared input x (decode, n=1).
            gpu.maybe_capture_activation(wqkv, x, 1, k);
            gpu.maybe_capture_activation(wz, x, 1, k);
            gpu.maybe_capture_activation(w_beta, x, 1, k);
            gpu.maybe_capture_activation(w_alpha, x, 1, k);

            hip!(gpu.fused_qkvza_paro4g128t(
                wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, &rs[0], &rs[1], &rs[2], &rs[3],
                mqkv, mz, mbeta, malpha, k
            ))
        }
        // QKV 3-way (FullAttn): synthesised via the 4-way kernel with m3=0.
        // a3/y3/x_rot3 are aliased to a0/y0/rs[0] — the kernel skips the 4th
        // projection because m3=0 guarantees no 4th write.
        KernelKey::FusedQkvParo4G128T => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs)
                .map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] =
                <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let rs = params.rot_scratch;
            assert!(rs.len() >= 4, "FusedQkvParo4G128T needs >= 4 rotation scratch buffers (4th aliased for m3=0), got {}", rs.len());
            assert!(
                mq % 8 == 0 && mk % 8 == 0 && mv % 8 == 0,
                "FusedQkvParo4G128T requires m%8==0, got mq={}, mk={}, mv={}",
                mq,
                mk,
                mv
            );
            assert!(
                k % 128 == 0,
                "FusedQkvParo4G128T requires k%128==0, got {}",
                k
            );
            // Calibration taps: one per constituent weight, shared input x (decode, n=1).
            gpu.maybe_capture_activation(wq, x, 1, k);
            gpu.maybe_capture_activation(wk, x, 1, k);
            gpu.maybe_capture_activation(wv, x, 1, k);

            hip!(gpu.fused_qkvza_paro4g128t(
                wq, wk, wv, wq, // a3 = wq (aliased)
                x, q, kout, v, q, // y3 = q (aliased)
                &rs[0], &rs[1], &rs[2], &rs[0], // x_rot3 = rs[0] (aliased, unused)
                mq, mk, mv, 0, // m3 = 0
                k
            ))
        }
        _ => Err(DispatchError::UnsupportedVariant {
            family: "fused_qkv",
            variant: "",
            arch: "",
            quant: "",
        }),
    }
}

fn dispatch_fused_qkv_with_qwen2_bias(
    gpu: &mut Gpu,
    params: &FusedQkvBiasParams,
) -> Result<(), DispatchError> {
    // Guard: same v1/v2 cross-check as the non-bias path, but with the 3 bias weights.
    guard_fused_qkv_dtype_key(
        &[params.weights[0], params.weights[1], params.weights[2]],
        params.kind,
    )?;
    let [wq, wk, wv] = params.weights;
    let [q, kout, v] = params.outputs;
    let [mq, mk, mv] = params.m;
    let [bq, bk, bv] = params.bias;
    let k = params.k;
    let x = params.x;

    match params.kind {
        KernelKey::FusedQkvHfq4G256 => {
            // Calibration taps: one per constituent weight, shared input x (qwen2 bias, decode n=1).
            gpu.maybe_capture_activation(wq, x, 1, k);
            gpu.maybe_capture_activation(wk, x, 1, k);
            gpu.maybe_capture_activation(wv, x, 1, k);
            hip!(gpu.fused_qkv_hfq4g256_with_bias(
                wq,
                wk,
                wv,
                x,
                q,
                kout,
                v,
                mq,
                mk,
                mv,
                k,
                bq.buf.as_ptr(),
                bk.buf.as_ptr(),
                bv.buf.as_ptr()
            ))
        }
        KernelKey::FusedQkvMq4G256V2 => {
            // Calibration taps: one per constituent weight, shared input x (qwen2 bias, decode n=1).
            gpu.maybe_capture_activation(wq, x, 1, k);
            gpu.maybe_capture_activation(wk, x, 1, k);
            gpu.maybe_capture_activation(wv, x, 1, k);
            hip!(gpu.fused_qkv_hfq4g256_with_bias_mq4v2(
                wq,
                wk,
                wv,
                x,
                q,
                kout,
                v,
                mq,
                mk,
                mv,
                k,
                bq.buf.as_ptr(),
                bk.buf.as_ptr(),
                bv.buf.as_ptr()
            ))
        }
        KernelKey::FusedQkvMq5G256V2 => {
            return Err(DispatchError::Hip(
                "FusedQkvMq5G256V2 bias not implemented".to_string(),
            ))
        }
        KernelKey::FusedQkvMq6G256V2 => {
            return Err(DispatchError::Hip(
                "FusedQkvMq6G256V2 bias not implemented".to_string(),
            ))
        }
        KernelKey::FusedQkvMq4CG256 => {
            // Calibration taps: one per constituent weight, shared input x (qwen2 bias, decode n=1).
            gpu.maybe_capture_activation(wq, x, 1, k);
            gpu.maybe_capture_activation(wk, x, 1, k);
            gpu.maybe_capture_activation(wv, x, 1, k);
            hip!(gpu.fused_qkv_mq4cg256_with_bias(
                wq,
                wk,
                wv,
                x,
                q,
                kout,
                v,
                mq,
                mk,
                mv,
                k,
                bq.buf.as_ptr(),
                bk.buf.as_ptr(),
                bv.buf.as_ptr()
            ))
        }
        KernelKey::FusedQkvMq3G256Lloyd => {
            // Calibration taps: one per constituent weight, shared input x (qwen2 bias, decode n=1).
            gpu.maybe_capture_activation(wq, x, 1, k);
            gpu.maybe_capture_activation(wk, x, 1, k);
            gpu.maybe_capture_activation(wv, x, 1, k);
            hip!(gpu.fused_qkv_mq3g256_lloyd_with_bias(
                wq,
                wk,
                wv,
                x,
                q,
                kout,
                v,
                mq,
                mk,
                mv,
                k,
                bq.buf.as_ptr(),
                bk.buf.as_ptr(),
                bv.buf.as_ptr()
            ))
        }
        KernelKey::FusedQkvMq4G256Lloyd => {
            // Calibration taps: one per constituent weight, shared input x (qwen2 bias, decode n=1).
            gpu.maybe_capture_activation(wq, x, 1, k);
            gpu.maybe_capture_activation(wk, x, 1, k);
            gpu.maybe_capture_activation(wv, x, 1, k);
            hip!(gpu.fused_qkv_mq4g256_lloyd_with_bias(
                wq,
                wk,
                wv,
                x,
                q,
                kout,
                v,
                mq,
                mk,
                mv,
                k,
                bq.buf.as_ptr(),
                bk.buf.as_ptr(),
                bv.buf.as_ptr()
            ))
        }
        KernelKey::FusedQkvHfq6G256 => {
            // Calibration taps: one per constituent weight, shared input x (qwen2 bias, decode n=1).
            gpu.maybe_capture_activation(wq, x, 1, k);
            gpu.maybe_capture_activation(wk, x, 1, k);
            gpu.maybe_capture_activation(wv, x, 1, k);
            hip!(gpu.fused_qkv_hfq6g256_with_bias(
                wq,
                wk,
                wv,
                x,
                q,
                kout,
                v,
                mq,
                mk,
                mv,
                k,
                bq.buf.as_ptr(),
                bk.buf.as_ptr(),
                bv.buf.as_ptr()
            ))
        }
        KernelKey::FusedQkvQ4K => {
            // Calibration taps: one per constituent weight, shared input x (qwen2 bias, decode n=1).
            gpu.maybe_capture_activation(wq, x, 1, k);
            gpu.maybe_capture_activation(wk, x, 1, k);
            gpu.maybe_capture_activation(wv, x, 1, k);
            hip!(gpu.fused_qkv_q4k_with_bias(
                wq,
                wk,
                wv,
                x,
                q,
                kout,
                v,
                mq,
                mk,
                mv,
                k,
                bq.buf.as_ptr(),
                bk.buf.as_ptr(),
                bv.buf.as_ptr()
            ))
        }
        KernelKey::FusedQkvQ8_0 => {
            // Calibration taps: one per constituent weight, shared input x (qwen2 bias, decode n=1).
            gpu.maybe_capture_activation(wq, x, 1, k);
            gpu.maybe_capture_activation(wk, x, 1, k);
            gpu.maybe_capture_activation(wv, x, 1, k);
            hip!(gpu.fused_qkv_q8_0_with_bias(
                wq,
                wk,
                wv,
                x,
                q,
                kout,
                v,
                mq,
                mk,
                mv,
                k,
                bq.buf.as_ptr(),
                bk.buf.as_ptr(),
                bv.buf.as_ptr()
            ))
        }
        _ => Err(DispatchError::UnsupportedVariant {
            family: "fused_qkv",
            variant: "qwen2_bias",
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

#[cfg(test)]
mod dispatch_tests {
    use super::is_fused_v2_key;
    use crate::types::KernelKey;

    #[test]
    fn v2_fused_keys_never_hfq4() {
        let cases: &[(KernelKey, &str)] = &[
            (KernelKey::FusedQkvMq6G256V2, "qkv6"),
            (KernelKey::FusedQkvMq5G256V2, "qkv5"),
            (KernelKey::FusedQkvMq3G256V2, "qkv3"),
            (KernelKey::FusedQkvMq2G256V2, "qkv2"),
            (KernelKey::FusedQkvzaMq6G256V2, "qkvza6"),
            (KernelKey::FusedQkvzaMq5G256V2, "qkvza5"),
            (KernelKey::FusedQkvzaMq3G256V2, "qkvza3"),
            (KernelKey::FusedQkvzaMq2G256V2, "qkvza2"),
            (KernelKey::FusedGateUpMq6G256V2, "gate6"),
            (KernelKey::FusedGateUpMq5G256V2, "gate5"),
            (KernelKey::FusedGateUpMq3G256V2, "gate3"),
            (KernelKey::FusedGateUpMq2G256V2, "gate2"),
        ];
        for (key, _) in cases {
            assert!(is_fused_v2_key(*key), "V2 key not recognized {:?}", key);
            assert_ne!(*key, KernelKey::FusedQkvHfq4G256);
            assert_ne!(*key, KernelKey::FusedQkvzaHfq4G256);
            assert_ne!(*key, KernelKey::FusedGateUpHfq4G256);
        }
    }
}
