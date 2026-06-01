// SPDX-License-Identifier: MIT OR Apache-2.0
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
    ) -> Result<KernelKey, DispatchError> {
        self.registry.resolve(key, ctx, None)
    }

    pub fn run(
        &self,
        ctx: &DispatchCtx,
        gpu: &mut Gpu,
        params: &FusedQkvParams,
    ) -> Result<(), DispatchError> {
        self.resolve(params.kind, ctx)?;
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

        // ── Paro variants need rotation scratch buffers ────
        KernelKey::FusedQkvParo4G128T
        | KernelKey::FusedQkvzaParo4G128T
        | KernelKey::FusedGateUpParo4G128T => Err(DispatchError::UnsupportedVariant {
            family: "fused_qkv",
            variant: "paro",
            arch: "",
            quant: "",
        }),

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
