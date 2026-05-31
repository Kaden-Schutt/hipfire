// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::tables::KernelRegistry;
use crate::types::*;
use rdna_compute::DType;

/// Register all GEMV kernel variants into the registry.
///
/// Covers all 4 variants: Plain, Prerotated, WithResidual, WithSwiGLUResidual.
/// Each entry pairs a KernelKey with the arch predicate that must be satisfied.
pub fn populate(registry: &KernelRegistry) {
    register_plain(registry);
    register_prerotated(registry);
    register_residual(registry);
    register_swiglu_residual(registry);
    register_fused(registry);
}

fn register_plain(registry: &KernelRegistry) {
    let dtypes: &[DType] = &[
        DType::F32,
        DType::F16,
        DType::Q8_0,
        DType::Q4K,
        DType::Q6K,
        DType::HFQ4G256,
        DType::HFQ4G128,
        DType::HFQ3G256,
        DType::HFQ3G128,
        DType::HFQ2G256,
        DType::HFQ2G128,
        DType::HFQ6G256,
        DType::MQ4G256,
        DType::MQ4G128,
        DType::MQ3G256,
        DType::MQ2G256,
        DType::MQ6G256,
        DType::MQ8G256,
        DType::MQ2G256Lloyd,
        DType::MQ3G256Lloyd,
        DType::MQ4G256Lloyd,
        DType::MFP4G32,
        DType::HFP4G32,
        DType::PARO4G128,
        DType::PARO4G128T,
        DType::ParoQ4G128,
        DType::Q4F16G64,
        DType::Q4F16G32,
    ];
    for &dtype in dtypes {
        let Ok(key) = KernelKey::for_gemv(dtype, GemvVariant::Plain, false) else {
            continue;
        };
        registry.register(KernelVariant {
            key,
            arch_required: KernelKey::dtype_arch_predicate(dtype),
            shape_gate: None,
            steps: KernelKey::gemv_steps(dtype, GemvVariant::Plain),
            has_awq: dtype == DType::MQ4G256,
        });
    }
}

fn register_prerotated(registry: &KernelRegistry) {
    let dtypes: &[DType] = &[
        DType::MQ4G256,
        DType::MQ3G256,
        DType::MQ2G256,
        DType::MQ6G256,
        DType::MQ8G256,
        DType::MQ2G256Lloyd,
        DType::MQ3G256Lloyd,
        DType::MQ4G256Lloyd,
        DType::MFP4G32,
    ];
    for &dtype in dtypes {
        let Ok(key) = KernelKey::for_gemv_prerotated(dtype) else {
            continue;
        };
        registry.register(KernelVariant {
            key,
            arch_required: KernelKey::dtype_arch_predicate(dtype),
            shape_gate: None,
            steps: KernelKey::gemv_steps(dtype, GemvVariant::Prerotated),
            has_awq: dtype == DType::MQ4G256,
        });
    }
}

fn register_residual(registry: &KernelRegistry) {
    let dtypes: &[DType] = &[
        DType::HFQ4G256,
        DType::HFQ3G256,
        DType::HFQ6G256,
        DType::MQ4G256,
        DType::MQ3G256,
        DType::MQ6G256,
        DType::MQ3G256Lloyd,
        DType::MQ4G256Lloyd,
        DType::PARO4G128,
        DType::PARO4G128T,
    ];
    for &dtype in dtypes {
        let Ok(key) = KernelKey::for_gemv_residual(dtype) else {
            continue;
        };
        registry.register(KernelVariant {
            key,
            arch_required: KernelKey::dtype_arch_predicate(dtype),
            shape_gate: None,
            steps: KernelKey::gemv_steps(dtype, GemvVariant::WithResidual),
            has_awq: dtype == DType::MQ4G256,
        });
    }
}

fn register_fused(registry: &KernelRegistry) {
    registry.register(KernelVariant {
        key: KernelKey::GemvMfp4G32Fused,
        arch_required: ArchPredicate::HasWmmaW32,
        shape_gate: None,
        steps: &[PipelineOp::RotateFwht, PipelineOp::Gemv],
        has_awq: false,
    });
}

fn register_swiglu_residual(registry: &KernelRegistry) {
    let dtypes: &[DType] = &[
        DType::HFQ4G256,
        DType::HFQ3G256,
        DType::HFQ6G256,
        DType::MQ4G256,
        DType::MQ3G256,
        DType::MQ6G256,
        DType::MQ3G256Lloyd,
        DType::MQ4G256Lloyd,
        DType::PARO4G128,
        DType::PARO4G128T,
    ];
    for &dtype in dtypes {
        let Ok(key) = KernelKey::for_gemv_swiglu_residual(dtype) else {
            continue;
        };
        registry.register(KernelVariant {
            key,
            arch_required: KernelKey::dtype_arch_predicate(dtype),
            shape_gate: None,
            steps: KernelKey::gemv_steps(dtype, GemvVariant::WithSwiGLUResidual),
            has_awq: dtype == DType::MQ4G256,
        });
    }
}

