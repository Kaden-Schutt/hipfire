// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! rdna-compute: Kernel compilation, caching, and dispatch for RDNA GPUs.

pub mod arch_caps;
pub mod attention;
mod compiler;
mod dispatch;
pub mod embedding;
pub mod feature_flags;
pub mod gemm;
pub mod gemv;
pub mod graph;
mod kernels;
pub mod moe;
pub mod norm;
pub mod pool;
pub mod profile;
pub mod profile_rocprof;
pub mod profiler;
pub mod replay;
pub mod sampling;
pub mod scratch;

pub use compiler::KernelCompiler;
pub use dispatch::{
    gen_fwht_signs, BlockHessianAcc, DType, Gpu, GpuTensor, HessianCapture, GL_CB2, GL_CB3,
    GL_GROUP_SCALE_BYTES, GL_MQ2_GROUP_IDX_BYTES, GL_MQ3_GROUP_IDX_BYTES, LLOYD_MQ3_GROUP_BYTES,
    LLOYD_MQ4_GROUP_BYTES, MMQ_CURRENT_LAYER,
};
pub use feature_flags::FeatureFlags;
pub use kernels::GEMV_SRC;
/// `(entry_point, source)` selectors for the uniform MoE grouped-WMMA GEMMs
/// whose gfx11 and gfx12 kernels are separate translation units. Exported so
/// no-GPU tests can assert the launcher resolves to a real entry point on both
/// arch legs (the `kernels` module itself stays private).
pub use kernels::{mq2g256_lloyd_moe_grouped_wmma_source, mq3g256_lloyd_moe_grouped_wmma_source};
