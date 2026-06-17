// SPDX-License-Identifier: Apache-2.0
//! Forward+backward op pairs for the un-fused fp32 training graph.
//!
//! Each op exposes a `*_forward` and `*_backward` free function over raw
//! `GpuTensor`s (fp32, row-major). Backward takes the upstream gradient and
//! produces gradients for each differentiable input. All matmuls route through
//! `gemm_f32_train` (verified correct in rdna-compute).

pub mod linear;
