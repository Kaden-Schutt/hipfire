// SPDX-License-Identifier: Apache-2.0
//! Forward+backward op pairs for the un-fused fp32 training graph.
//!
//! Each op exposes a `*_forward` and `*_backward` free function over raw
//! `GpuTensor`s (fp32, row-major). Backward takes the upstream gradient and
//! produces gradients for each differentiable input. All matmuls route through
//! `gemm_f32_train` (verified correct in hipfire-rdna).

pub mod attention;
pub mod cross_entropy;
pub mod distill;
pub mod gated_scan;
pub mod linear;
pub mod lora;
pub mod pflash_score;
pub mod rmsnorm;
pub mod rope;
pub mod sigmoid;
pub mod softmax;
pub mod swiglu;
