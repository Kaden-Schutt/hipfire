// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
use crate::types::*;
use crate::tables::KernelRegistry;

/// Populate the registry with rotation kernel variants.
pub fn populate(registry: &mut KernelRegistry) {
    macro_rules! reg {
        ($key:ident, $arch:expr, $steps:expr, $awq:expr) => {
            registry.register(KernelVariant {
                key: KernelKey::$key,
                arch_required: $arch,
                shape_gate: None,
                steps: $steps,
                has_awq: $awq,
                tile: TileImpl::None,
            });
        };
    }

    // RotateMq — plain FWHT rotation (G256)
    reg!(RotateMq, ArchPredicate::Always, &[PipelineOp::RotateFwht], false);
    // RotateMqG128 — plain FWHT rotation with G128 sign tables
    reg!(RotateMqG128, ArchPredicate::Always, &[PipelineOp::RotateFwht], false);
    reg!(RotateMqAwq, ArchPredicate::Always, &[PipelineOp::AwqDivide, PipelineOp::RotateFwht], true);
    reg!(RotateMqBatched, ArchPredicate::Always, &[PipelineOp::RotateFwht], false);
    reg!(RotateMqAwqBatched, ArchPredicate::Always, &[PipelineOp::AwqDivide, PipelineOp::RotateFwht], true);

    // RmsnormRotateMq — fused RMSNorm + FWHT rotation
    reg!(RmsnormRotateMq, ArchPredicate::Always, &[PipelineOp::RotateFwht], false);
    reg!(RmsnormRotateMqAwq, ArchPredicate::Always, &[PipelineOp::AwqDivide, PipelineOp::RotateFwht], true);
    reg!(RmsnormRotateMqBatched, ArchPredicate::Always, &[PipelineOp::RotateFwht], false);
    reg!(RmsnormRotateMqAwqBatched, ArchPredicate::Always, &[PipelineOp::AwqDivide, PipelineOp::RotateFwht], true);

    // SiluMulRotateMq — fused SwiGLU + FWHT rotation
    reg!(SiluMulRotateMq, ArchPredicate::Always, &[PipelineOp::SiluMulRotate], false);
    reg!(SiluMulRotateMqAwq, ArchPredicate::Always, &[PipelineOp::AwqDivide, PipelineOp::SiluMulRotate], true);

    // RmsnormF32 — plain RMSNorm, no rotation (utility entry)
    reg!(RmsnormF32, ArchPredicate::Always, &[], false);
}
