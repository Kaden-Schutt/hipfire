// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! rdna-compute: Kernel compilation, caching, and dispatch for RDNA GPUs.

pub mod arch_caps;
pub mod graph;
mod compiler;
mod dispatch;
pub mod feature_flags;
mod kernels;
pub mod pool;
pub mod profile;
pub mod profile_rocprof;
pub mod profiler;

pub use compiler::KernelCompiler;
pub use dispatch::{
    gen_fwht_signs, DType, Gpu, GpuTensor, LLOYD_MQ4_GROUP_BYTES, MMQ_CURRENT_LAYER,
};
pub use feature_flags::FeatureFlags;
pub use kernels::GEMV_SRC;
