// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait implementation for the LLaMA family.
//!
//! Mirrors PR 8's qwen35 pattern. Bring-up triple (`config_from_hfq`,
//! `load_weights`, `new_state`) goes through the trait so daemon and
//! examples can dispatch by `arch_id` without growing a `match` ladder.
//! Forward passes stay direct `llama::*` calls — the hot path doesn't
//! pay dyn dispatch overhead.
//!
//! See `crates/hipfire-arch-qwen35/src/arch.rs` for the canonical
//! design rationale; PR 11 just adds a second implementation of the
//! same trait surface for LLaMA-family bring-up.

use hip_bridge::HipResult;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::{self, HfqFile};
use hipfire_runtime::llama::{ForwardScratch, KvCache, LlamaConfig, LlamaWeights};
use rdna_compute::Gpu;

/// Type marker for the LLaMA family — covers `arch_id = 0` (LLaMA /
/// Mistral) and `arch_id = 1` (plain Qwen3 / Qwen2). All members of
/// this family share the dense-transformer forward pass owned by
/// [`hipfire_runtime::llama`].
///
/// Qwen3.5 / Qwen3.6 (hybrid DeltaNet, `arch_id = 5`) and Qwen3.5/3.6
/// MoE / Qwen3MoE (`arch_id = 6`) are NOT covered by this marker —
/// see [`hipfire_arch_qwen35::Qwen35`] for those.
pub struct Llama;

impl Architecture for Llama {
    type Weights = LlamaWeights;
    type State = ForwardScratch;
    type Config = LlamaConfig;

    fn arch_id() -> u32 {
        // `arch_id = 0` is the canonical LLaMA-family marker. The
        // actual arch_id loaded at runtime is on `HfqFile::arch_id`
        // and is either 0 (LLaMA / Mistral) or 1 (plain Qwen3 /
        // Qwen2); both share this trait impl. The qwen3-norm flag
        // is read off the HFQ metadata inside `config_from_hfq`,
        // so the bring-up triple does not need a separate marker
        // type per arch_id.
        0
    }

    fn name() -> &'static str {
        "llama"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        // `hfq::config_from_hfq` is the LLaMA-family HFQ metadata
        // parser — emits a `LlamaConfig` with the appropriate
        // `ModelArch` (Llama vs Qwen3) tag. It lives in the runtime
        // crate because the qwen35 hybrid path's pflash drafter also
        // calls it via `hfq::config_from_hfq` for its "Plain"
        // variant. See arch-llama/src/lib.rs for the colocation
        // rationale.
        hfq::config_from_hfq(hfq)
            .ok_or_else(|| "llama: failed to parse config from HFQ metadata".to_string())
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        // `hfq::load_weights_hfq` is the LLaMA-family HFQ tensor
        // loader. Same colocation reasoning as `config_from_hfq`.
        hfq::load_weights_hfq(hfq, cfg, gpu)
            .map_err(|e| format!("llama: load_weights_hfq failed: {e:?}"))
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        // The LLaMA-arch "state" is the `ForwardScratch` — persistent
        // GPU scratch buffers reused across decode steps. There is no
        // separate recurrent state (LLaMA is full-attention only).
        ForwardScratch::new(gpu, cfg)
            .map_err(|e| format!("llama: ForwardScratch::new failed: {e:?}"))
    }

    // Optional overrides: defaults from `hipfire_runtime::arch` already
    // assume Qwen3.5 family conventions. LLaMA / Mistral / Qwen3 don't
    // emit `<think>` blocks, but PR 11 keeps the override surface
    // empty here on purpose — the daemon's existing per-`arch_id`
    // policy choices stay unchanged. Future PRs that consolidate
    // policy through the trait can populate these (LLaMA: no
    // strip_think, no Qwen-specific blocked tokens).
}

// ── Dispatch integration ─────────────────────────────────────────
// When `feature = "new-dispatch"` is active, the crate builds with
// hipfire-dispatch and uses its centralized kernel selection tables
// instead of the inline match-on-DType trees in llama.rs.
//
// Migration pattern for each model forward function:
//
//   #[cfg(feature = "new-dispatch")]
//   fn forward(...) -> HipResult<...> {
//       ModelDispatch::new(gpu).forward_scratch_layers(gpu, weights, config, pos, ...)
//   }
//
//   #[cfg(not(feature = "new-dispatch"))]
//   fn forward(...) -> HipResult<...> {
//       llama::forward_scratch_layers(gpu, weights, config, pos, ...)
//   }
//
// The `ModelDispatch` struct (to be created in a follow-up) wraps all
// 6 families: rotation, gemv, gemm, fused_qkv, attention, moe.
// Each family selects kernel variant via (DType, variant, arch_caps),
// and the pipeline runner handles FWHT rotation, AWQ scaling, residual
// fusion automatically.
//
// Phase 1 proof of concept — see `crate::forward_dispatch` for the
// concrete RotationFamily integration.
//
// See `.opencode/plans/2026-05-30-hipfire-dispatch.md` for the full
// design and migration phases.
//
// ── Phase 1: RotationFamily integration ──────────────────────────

impl Llama {
    /// Forward pass — new-dispatch variant when the feature is active.
    ///
    /// Creates the dispatch context and delegates to the rotation-aware
    /// forward path in [`crate::forward_dispatch`]. Everything outside
    /// the rotation calls (attention, KV cache, sampling) is unchanged.
    #[cfg(feature = "new-dispatch")]
    pub fn forward_scratch_layers(
        gpu: &mut Gpu,
        weights: &LlamaWeights,
        config: &LlamaConfig,
        pos: usize,
        kv_cache: &mut KvCache,
        scratch: &ForwardScratch,
        temperature: f32,
        top_p: f32,
        rng_state: u32,
        repeat_window: usize,
        repeat_penalty: f32,
    ) -> HipResult<(u32, u32)> {
        crate::forward_dispatch::forward_scratch_layers(
            gpu, weights, config, pos, kv_cache, scratch,
            temperature, top_p, rng_state, repeat_window, repeat_penalty,
        )
    }

    /// Forward pass — legacy path (always available).
    #[cfg(not(feature = "new-dispatch"))]
    pub fn forward_scratch_layers(
        gpu: &mut Gpu,
        weights: &LlamaWeights,
        config: &LlamaConfig,
        pos: usize,
        kv_cache: &mut KvCache,
        scratch: &ForwardScratch,
        temperature: f32,
        top_p: f32,
        rng_state: u32,
        repeat_window: usize,
        repeat_penalty: f32,
    ) -> HipResult<(u32, u32)> {
        crate::llama::forward_scratch_layers(
            gpu, weights, config, pos, kv_cache, scratch,
            temperature, top_p, rng_state, repeat_window, repeat_penalty,
        )
    }
}
