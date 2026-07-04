// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Toy-fixture vocabulary: the data an arch declares so a tiny, deterministic
//! random-init checkpoint can be synthesized for CI/gating.
//!
//! Follows the same declare-vs-do split as [`Ingest`](crate::Ingest): the arch
//! DESCRIBES its fixture (config JSON + tensor manifest) via
//! [`ToyModel::fixture`](crate::ToyModel::fixture); the offline quantizer owns the
//! actual byte generation + safetensors/tokenizer writing (seeded RNG, dtype
//! packing). So these types are plain data with no I/O and no serde dep — an arch's
//! `-spec` crate can build a [`ToyFixture`] with only `hipfire-arch-api` (+ whatever
//! it uses to render the config string).

/// Storage dtype for a fixture tensor. Weight matrices ship BF16 (source precision);
/// 1D norm/bias vectors ship F16 because several per-arch loaders reject BF16 for
/// norms/biases — real checkpoints keep those at F16/F32, never quantized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dt {
    Bf16,
    F16,
}

impl Dt {
    /// safetensors dtype tag.
    pub fn st_name(self) -> &'static str {
        match self {
            Dt::Bf16 => "BF16",
            Dt::F16 => "F16",
        }
    }
}

/// How a tensor's elements are initialized. The arch names the POLICY; the quantizer
/// realizes it against a seeded RNG (so bytes stay deterministic across machines).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Init {
    /// Zero-mean uniform in `[-scale, scale]` — generic projections.
    Uniform(f32),
    /// RMSNorm weights: ~1.0 + small jitter.
    NormOnes,
    /// Mamba/DeltaNet `A_log`: small negative so decay stays well-conditioned.
    ALog,
    /// Bias-like: zeros.
    Zeros,
}

/// One tensor in a fixture manifest: name, shape, init policy, storage dtype.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub init: Init,
    pub dt: Dt,
}

impl TensorSpec {
    /// BF16 tensor (default — weight matrices).
    pub fn new(name: impl Into<String>, shape: Vec<usize>, init: Init) -> Self {
        Self {
            name: name.into(),
            shape,
            init,
            dt: Dt::Bf16,
        }
    }

    /// F16 tensor — used for 1D norm/bias vectors (see [`Dt`]).
    pub fn f16(name: impl Into<String>, shape: Vec<usize>, init: Init) -> Self {
        Self {
            name: name.into(),
            shape,
            init,
            dt: Dt::F16,
        }
    }
}

/// A described toy fixture: the model's `config.json` (already rendered to a string,
/// so this crate stays serde-free) plus the tensor manifest. The quantizer turns this
/// into a loadable HF model dir (safetensors + config + shared tokenizer).
#[derive(Debug, Clone)]
pub struct ToyFixture {
    /// The fixture's `config.json` body.
    pub config_json: String,
    /// Every tensor the fixture emits, in declaration order.
    pub tensors: Vec<TensorSpec>,
}
