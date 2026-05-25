// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Stub model types for the toy arch.
//!
//! Every type here exists only to satisfy the [`Architecture`] trait's
//! associated-type slots. None of these structs do real compute. A real
//! arch replaces every body in this file with model logic; the *shapes*
//! (Config / Weights / State as separate types, owned by the arch crate,
//! constructable from an [`HfqFile`]) are what the trait expects.
//!
//! [`Architecture`]: hipfire_runtime::arch::Architecture
//! [`HfqFile`]: hipfire_runtime::hfq::HfqFile

use hipfire_runtime::hfq::HfqFile;

/// Toy config: tiny hardcoded constants. A real arch parses these out of
/// `HfqFile::metadata_json` (see `hipfire_arch_qwen35::qwen35::config_from_hfq`
/// for how Qwen3.5 walks the JSON tree, branches on `arch_id` for
/// dense-vs-MoE shape, and falls back to defaults for missing keys).
#[derive(Debug, Clone)]
pub struct ToyConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub layers: usize,
}

impl ToyConfig {
    /// In a real arch, this method reads `hfq.metadata_json` (a JSON
    /// blob) and returns either `Ok(config)` or `Err(reason)`. Here we
    /// ignore the input entirely and return hardcoded constants.
    pub fn from_hfq(_hfq: &HfqFile) -> Result<Self, String> {
        Ok(ToyConfig {
            vocab_size: 256,
            dim: 8,
            layers: 1,
        })
    }
}

/// Toy weights: a single embedding table, zero-initialized. A real arch
/// holds GPU-resident `WeightTensor` handles for every projection
/// (attention QKV, output, FFN gate/up/down, layernorm scales, embeddings,
/// LM head). See `Qwen35Weights` in `hipfire-arch-qwen35` for a complete
/// example.
pub struct ToyWeights {
    /// Stub embedding table. In a real arch this would be a
    /// `WeightTensor` (GPU-resident, possibly quantized). We keep a
    /// host-side `Vec<f32>` here just to demonstrate the slot exists.
    pub embeddings: Vec<f32>,
}

impl ToyWeights {
    /// Stub loader: ignores HFQ contents, returns a zero-initialized
    /// embedding table. A real arch would walk `hfq.tensor_info(name)`
    /// for each weight, dispatch to `WeightTensor::from_hfq_tensor` to
    /// upload it to GPU memory in the appropriate quant format
    /// (Q4F16G64 / F16 / F32), and assemble per-layer arrays.
    pub fn load(_hfq: &HfqFile, cfg: &ToyConfig) -> Result<Self, String> {
        Ok(ToyWeights {
            embeddings: vec![0.0; cfg.vocab_size * cfg.dim],
        })
    }
}

/// Toy state: a bare token counter. A real arch's state holds GPU
/// scratch buffers reused across decode steps, KV-cache handles,
/// recurrent-state tensors (for hybrid linear-attention archs), and
/// any per-step metadata the forward pass needs. See `DeltaNetState`
/// in `hipfire-arch-qwen35` (hybrid LA + FA scratch) and
/// `ForwardScratch` in `hipfire-runtime::llama` (dense FA scratch)
/// for the two reference shapes.
pub struct ToyState {
    pub token_count: usize,
}

impl ToyState {
    /// Stub state init: returns a bare counter. A real arch allocates
    /// GPU buffers via `gpu.alloc(...)`, sized by `cfg`.
    pub fn new(_cfg: &ToyConfig) -> Result<Self, String> {
        Ok(ToyState { token_count: 0 })
    }
}
