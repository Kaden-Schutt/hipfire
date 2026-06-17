// SPDX-License-Identifier: Apache-2.0
// hipfire-train — training path for fine-tuning (quantized) models.
//
// Phase 0 (docs/plans/2026-06-17-hipfire-train-phase0.md): stand up the first
// backward pass + optimizer in hipfire and prove it numerically correct via a
// LoRA SFT overfit on Supra-50M, base weights in fp32.
//
// Design invariant: this crate does NOT differentiate the fused inference
// kernels (`fused_rmsnorm_rotate_mq`, …). It owns an *un-fused* fp32 forward —
// one clean op per node, each with a matching backward — built on the dedicated
// `gemm_f32_train` primitive (general transpose flags) in `rdna-compute`.

pub mod block;
pub mod config;
pub mod loader;
pub mod model;
pub mod ops;
pub mod optim;
pub mod tensor;

pub use config::LlamaConfig;
pub use loader::{load_llama_fp32, LlamaWeightsF32};
pub use tensor::TrainTensor;
