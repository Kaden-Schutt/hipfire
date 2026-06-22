// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! rdna-compute: Kernel compilation, caching, and dispatch for RDNA GPUs.

pub mod arch_caps;
mod compiler;
mod dispatch;
pub mod feature_flags;
pub mod generic_warn;
mod kernels;
pub mod pool;
pub mod profile;
pub mod profile_rocprof;
pub mod profiler;

pub use compiler::KernelCompiler;
pub use dispatch::{
    gen_fwht_signs, ActivationCapture, DType, Gpu, GpuTensor, LLOYD_MQ4_GROUP_BYTES,
    MMQ_CURRENT_LAYER,
};
pub use feature_flags::FeatureFlags;
pub use kernels::GEMV_SRC;
// Re-export the result/error types of `Gpu`'s public methods so downstream
// crates (e.g. hipfire-train) can name them without depending on hip-bridge.
pub use hip_bridge::{HipError, HipResult};
