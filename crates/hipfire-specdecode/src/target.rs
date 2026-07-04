// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The `SpecDecodeTarget` model boundary.
//!
//! This is the trait a speculative-decode *target* or *draft* slot must satisfy
//! so the strategy families (dflash, ddtree, mtp) can drive draft→verify→accept
//! without naming a concrete architecture. An arch crate implements it for its
//! own slot type (e.g. `hipfire-arch-qwen35`'s `ModelSlot`), keeping the
//! arch-specific forward + recurrent-state mechanics on its side of the seam.
//!
//! The method set is the minimal boundary the strategies actually reach for
//! (measured against `speculative.rs`): output geometry, the lm_head weight and
//! logits buffer for the verify-graph GEMM, a single-token forward, sequence
//! reset, and the slot's own tokenizer. State snapshot / restore for rollback
//! replay (the arch-specific GDN/DeltaNet tape machinery) is threaded through in
//! a follow-up as an associated `StateSnapshot` type — see
//! `docs/specdecode-extraction-plan.md` (P2b).

use hip_bridge::HipResult;
use hipfire_model::tokenizer::{Tokenizer, TokenizerError};
use hipfire_rdna::{Gpu, GpuTensor};
use hipfire_runtime::weights::WeightTensor;

/// A target/draft model slot the speculative-decode strategies drive.
///
/// Implemented by an arch crate for its own slot type. Every method is a
/// narrow, arch-agnostic view onto the underlying model; the concrete
/// config/weights/scratch/recurrent-state types stay private to the arch.
pub trait SpecDecodeTarget {
    /// Human-readable slot label (e.g. `"target"` / `"draft"`), for diagnostics.
    fn name(&self) -> &str;

    /// Output vocabulary size — the width of the logits row.
    fn vocab_size(&self) -> usize;

    /// Number of transformer layers.
    fn num_layers(&self) -> usize;

    /// KV-cache geometry as `(n_kv_heads, head_dim)`.
    fn kv_geometry(&self) -> (usize, usize);

    /// The output projection (lm_head) weight, consumed by the verify-graph GEMM.
    fn lm_head_weight(&self) -> &WeightTensor;

    /// Device buffer holding this slot's most recent logits row(s).
    fn logits(&self) -> &GpuTensor;

    /// Single-token forward pass; writes logits into [`logits`](Self::logits).
    fn forward(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> HipResult<()>;

    /// Reset recurrent + KV write state to a clean sequence start (does not
    /// shrink the KV allocation).
    fn reset_state(&mut self, gpu: &mut Gpu);

    /// Load this slot's tokenizer from its own embedded model metadata.
    fn load_tokenizer(&self) -> Result<Tokenizer, TokenizerError>;
}
