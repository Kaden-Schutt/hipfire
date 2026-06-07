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
        KernelKey::FusedQkvHfq4G256 => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] = <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            hip!(gpu.fused_qkv_hfq4g256(wq, wk, wv, x, q, kout, v, mq, mk, mv, k))
        }
        KernelKey::FusedQkvMq3G256Lloyd => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] = <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            hip!(gpu.fused_qkv_mq3g256_lloyd(wq, wk, wv, x, q, kout, v, mq, mk, mv, k))
        }
        KernelKey::FusedQkvMq4G256Lloyd => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] = <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            hip!(gpu.fused_qkv_mq4g256_lloyd(wq, wk, wv, x, q, kout, v, mq, mk, mv, k))
        }
        KernelKey::FusedQkvHfq6G256 => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] = <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            hip!(gpu.fused_qkv_hfq6g256_dp4a(wq, wk, wv, x, q, kout, v, mq, mk, mv, k))
        }
        KernelKey::FusedQkvQ4K => {
            let [wq, wk, wv] = <[&GpuTensor; 3]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [q, kout, v] = <[&GpuTensor; 3]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 3))?;
            let [mq, mk, mv] = <[usize; 3]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 3))?;
            hip!(gpu.fused_qkv_q4k(wq, wk, wv, x, q, kout, v, mq, mk, mv, k))
        }

        // ── 4-way Fused QKVZA (DeltaNet linear attention) ────
        KernelKey::FusedQkvzaHfq4G256 => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] = <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            hip!(gpu.fused_qkvza_hfq4g256(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k))
        }
        KernelKey::FusedQkvzaMq3G256Lloyd => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] = <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            hip!(gpu.fused_qkvza_mq3g256_lloyd(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k))
        }
        KernelKey::FusedQkvzaMq4G256Lloyd => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] = <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            hip!(gpu.fused_qkvza_mq4g256_lloyd(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k))
        }
        KernelKey::FusedQkvzaHfq6G256 => {
            let [wqkv, wz, w_beta, w_alpha] = <[&GpuTensor; 4]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [qkv, z, beta, alpha] = <[&GpuTensor; 4]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 4))?;
            let [mqkv, mz, mbeta, malpha] = <[usize; 4]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 4))?;
            hip!(gpu.fused_qkvza_hfq6g256_dp4a(wqkv, wz, w_beta, w_alpha, x, qkv, z, beta, alpha, mqkv, mz, mbeta, malpha, k))
        }

        // ── 2-way Fused Gate+Up (FFN) ────────────────────────
        KernelKey::FusedGateUpHfq4G256 => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] = <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            hip!(gpu.fused_gate_up_hfq4g256(w_gate, w_up, x, gate, up, mg, mu, k))
        }
        KernelKey::FusedGateUpMq3G256Lloyd => {
            let [w_gate, w_up] = <[&GpuTensor; 2]>::try_from(params.weights).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [gate, up] = <[&GpuTensor; 2]>::try_from(params.outputs).map_err(|_| err_wrong_arity(params.kind, 2))?;
            let [mg, mu] = <[usize; 2]>::try_from(params.m).map_err(|_| err_wrong_arity(params.kind, 2))?;
            hip!(gpu.fused_gate_up_mq3g256_lloyd(w_gate, w_up, x, gate, up, mg, mu, k))
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
            hip!(gpu.fused_gate_up_hfq6g256_dp4a(w_gate, w_up, x, gate, up, mg, mu, k))
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
            hip!(gpu.fused_gate_up_q8_0(w_gate, w_up, x, gate, up, mg, mu, k))
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

fn err_wrong_arity(kind: KernelKey, _expected: usize) -> DispatchError {
    DispatchError::MissingImpl { key: kind }
}
