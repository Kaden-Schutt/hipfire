// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! rdna-compute: Kernel compilation, caching, and dispatch for RDNA GPUs.

pub mod arch_caps;
pub mod attention;
pub mod cdna;
mod compiler;
mod dispatch;
pub mod dflash_hidden_scatter;
pub mod dflash_state_copy;
pub mod embedding;
pub mod feature_flags;
#[cfg(feature = "flash-attn-ck")]
pub mod flash_attn_ck;
pub mod gemm;
mod gemma4_ops;
pub mod gemv;
pub mod graph;
mod kernels;
pub mod kv_slots;
pub mod moe;
pub mod mq_f16_producers;
pub mod norm;
pub mod pool;
pub mod profile;
pub mod profile_rocprof;
pub mod profiler;
pub mod rdna;
pub mod replay;
pub mod sampling;
pub mod scratch;
pub mod slot_pool;

pub use compiler::KernelCompiler;
pub use dispatch::{
    gen_fwht_signs, ActivationCapture, BlockHessianAcc, DType, Gpu, GpuTensor, HessianCapture,
    GL_CB2, GL_CB3, GL_GROUP_SCALE_BYTES, GL_MQ2_GROUP_IDX_BYTES, GL_MQ3_GROUP_IDX_BYTES,
    LLOYD_MQ3_GROUP_BYTES, LLOYD_MQ4_GROUP_BYTES, MMQ_CURRENT_LAYER, MQ2G256V2_GROUP_BYTES,
    MQ3G256V2_GROUP_BYTES, MQ4C_GROUP_BYTES, MQ4V2_GROUP_BYTES, MQ5G256V2_GROUP_BYTES,
    MQ6G256V2_GROUP_BYTES,
};
pub use feature_flags::FeatureFlags;
pub use hip_bridge::{HipError, HipResult};
use std::sync::OnceLock;

/// Calibration-only override that forces native-BF16 teachers to stay BF16.
///
/// When `HIPFIRE_CALIB_BF16=1`, a qt=16 / BF16-source weight is uploaded as
/// raw 2-byte BF16 and tagged `DType::BF16` instead of being widened to F32,
/// and activations staged via `Gpu::convert_f32_to_bf16` reach
/// `KernelKey::GemmBf16Mfma` on gfx942 (CDNA3 MFMA `v_mfma_f32_16x16x16bf16_1k`)
/// instead of the scalar F32 GEMM (`GemmF32RegisterTiled` / `GemmF32Batched`
/// in `kernels/src/gemm_f32.hip` — one warp per output element, zero data
/// reuse, no LDS, no MFMA). OFF by default so shipped inference is
/// unaffected.
pub fn calib_force_bf16() -> bool {
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| std::env::var("HIPFIRE_CALIB_BF16").as_deref() == Ok("1"))
}
pub use kernels::GEMV_SRC;
/// whose gfx11 and gfx12 kernels are separate translation units. Exported so
/// no-GPU tests can assert the launcher resolves to a real entry point on both
/// arch legs (the `kernels` module itself stays private).
pub use kernels::{mq2g256_lloyd_moe_grouped_wmma_source, mq3g256_lloyd_moe_grouped_wmma_source};

#[cfg(test)]
mod tests {
    use super::calib_force_bf16;

    #[test]
    fn calib_force_bf16_false_when_unset() {
        // Spec requires false when HIPFIRE_CALIB_BF16 is not "1".
        // In CI the var is unset; if a prior test or env sets it to "1"
        // we skip rather than flake — the contract is OFF by default.
        if std::env::var("HIPFIRE_CALIB_BF16").as_deref() == Ok("1") {
            return;
        }
        // Ensure the var is not "1" for this assertion path.
        // When OnceLock is already cached as true (because some earlier
        // call saw "1"), the cached true would violate the unset
        // expectation — but that path is unreachable when the env is
        // unset at process start, which is the acceptance condition.
        assert!(
            !calib_force_bf16(),
            "calib_force_bf16() must be false when HIPFIRE_CALIB_BF16 != \"1\" (OFF by default)"
        );
    }
}
