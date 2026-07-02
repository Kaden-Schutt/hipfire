// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! rdna-compute: Kernel compilation, caching, and dispatch for RDNA GPUs.

// `arch_caps::ArchCaps::dump_json` builds a large flat `serde_json::json!`
// object (60+ fields) — past the default macro recursion limit.
#![recursion_limit = "256"]

pub mod arch_caps;
pub mod attention;
pub mod bill_of_debt;
pub mod chip_profile;
mod compiler;
mod dispatch;
pub mod embedding;
pub mod feature_flags;
pub mod gemm;
pub mod gemv;
pub mod graph;
pub mod isa_histogram;
pub mod kernel_ledger;
mod kernels;
pub mod moe;
pub mod norm;
pub mod pool;
pub mod profile;
pub mod profile_rocprof;
pub mod profiler;
pub mod roofline;
pub mod sampling;
pub mod scratch;

pub use compiler::KernelCompiler;
pub use dispatch::{
    gen_fwht_signs, BlockHessianAcc, DType, Gpu, GpuTensor, HessianCapture, LLOYD_MQ3_GROUP_BYTES,
    LLOYD_MQ4_GROUP_BYTES, MMQ_CURRENT_LAYER,
};
pub use feature_flags::FeatureFlags;
pub use kernels::GEMV_SRC;
