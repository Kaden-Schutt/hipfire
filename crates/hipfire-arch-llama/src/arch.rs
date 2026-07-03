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

use hipfire_runtime::arch::{
    run_simple_ar, ArchCaps, Architecture, GenerateCtx, ServeOutcome, ServingBackend, SimpleAr,
};
use hipfire_runtime::hfq::{self, HfqFile};
use hipfire_runtime::kv::KvCache;
use hipfire_runtime::llama::{self, ForwardScratch, LlamaConfig, LlamaWeights};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{Gpu, GpuTensor};

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
        let weights = hfq::load_weights_hfq(hfq, cfg, gpu)
            .map_err(|e| format!("llama: load_weights_hfq failed: {e:?}"))?;
        // Pre-flight gate: refuse a model whose linear-weight dtype the GEMV path
        // cannot dispatch (e.g. an unsupported quant variant) BEFORE the forward,
        // with one legible error — instead of panicking deep in the lm_head GEMV.
        hipfire_runtime::weights::preflight_gemv_dtypes(&weights.linear_weight_dtypes())
            .map_err(|e| format!("llama: {e}"))?;
        Ok(weights)
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

/// Serving backend for the dense LLaMA / Mistral / plain-Qwen3 family (arch_id
/// 0/1) — routes through the shared `ServingBackend::serve` seam
/// (`run_simple_ar` → prefill → `decode_loop`) instead of the qwen35-shared
/// `generate()` path. Owns its config/weights/decode scratch/KV cache. P3.2.
pub struct LlamaBackend {
    pub arch_id: u32,
    pub config: LlamaConfig,
    pub weights: LlamaWeights,
    pub scratch: ForwardScratch,
    pub kv_cache: KvCache,
}

impl LlamaBackend {
    pub fn new(
        arch_id: u32,
        config: LlamaConfig,
        weights: LlamaWeights,
        scratch: ForwardScratch,
        kv_cache: KvCache,
    ) -> Self {
        Self {
            arch_id,
            config,
            weights,
            scratch,
            kv_cache,
        }
    }
}

impl SimpleAr for LlamaBackend {
    fn prefill(&mut self, gpu: &mut Gpu, tokens: &[u32]) -> Result<(), String> {
        // Leaves the final-position logits in `scratch.logits`; the batched path
        // falls back to a per-token embed+compute loop for short/ineligible
        // prompts, so any token slice is safe.
        llama::forward_prefill_batch(
            gpu,
            &self.weights,
            &self.config,
            tokens,
            0,
            &mut self.kv_cache,
            &self.scratch,
            None,
        )
        .map_err(|e| format!("llama prefill forward_prefill_batch: {e:?}"))
    }

    fn decode_step(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> Result<(), String> {
        // Logits-only decode: embed the token then run the layer stack, leaving
        // logits in `scratch.logits` for the daemon sampler (the sampling
        // `forward_scratch` is intentionally bypassed — `decode_loop` samples).
        llama::forward_scratch_embed(gpu, &self.weights, &self.config, token, pos, &self.scratch)
            .map_err(|e| format!("llama decode forward_scratch_embed: {e:?}"))?;
        llama::forward_scratch_compute(
            gpu,
            &self.weights,
            &self.config,
            pos,
            &mut self.kv_cache,
            &self.scratch,
        )
        .map_err(|e| format!("llama decode forward_scratch_compute: {e:?}"))
    }

    fn logits(&self) -> &GpuTensor {
        &self.scratch.logits
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

/// Dense-AR family: no fast-path caps; the shared `run_simple_ar` loop over the
/// `SimpleAr` impl drives generation.
impl ServingBackend for LlamaBackend {
    fn arch_id(&self) -> u32 {
        self.arch_id
    }

    fn caps(&self) -> ArchCaps {
        ArchCaps::default()
    }

    fn eos_token(&self) -> u32 {
        self.config.eos_token
    }

    fn serve(
        &mut self,
        gpu: &mut Gpu,
        tok: &Tokenizer,
        ctx: &mut GenerateCtx,
    ) -> Result<ServeOutcome, String> {
        let eos = self.config.eos_token;
        run_simple_ar(gpu, self, tok, eos, ctx)
    }

    fn reset_session(&mut self, _gpu: &mut Gpu, _session_id: &str) -> Result<(), String> {
        // Single-session bring-up: rewind the KV write cursor.
        self.kv_cache.compact_offset = 0;
        Ok(())
    }

    fn unload(self: Box<Self>, gpu: &mut Gpu) {
        let b = *self;
        b.weights.free_gpu(gpu);
        b.scratch.free_gpu(gpu);
        b.kv_cache.free_gpu(gpu);
    }
}
