// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Arch-crate bundle for Maple-Preview: the loaded (config, weights, state)
//! triple plus its `ArchModel` view.
//!
//! Mirrors `Cohere2MoeBundle`. The `ArchModel` impl is what lets the loader
//! hand a boxed, arch-agnostic model back to the daemon; `kv_cache_mut` stays
//! `None` because `MapleState` owns its own `KvCache` rather than the shared
//! `llama::KvCache` the FlashCASK eviction path expects.

use crate::config::MapleConfig;
use crate::maple::{MapleState, MapleWeights};
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

pub struct MapleBundle {
    pub config: MapleConfig,
    pub weights: MapleWeights,
    pub state: MapleState,
    pub eos_tok: u32,
}

impl ArchModel for MapleBundle {
    fn dim(&self) -> usize {
        self.config.hidden_size
    }

    fn n_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn arch_key(&self) -> &'static str {
        "maple"
    }

    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        // MapleState owns its KV; not the shared llama::KvCache.
        None
    }

    fn reset_session_state(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.state.reset(gpu)
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        let MapleBundle {
            config: _,
            weights,
            state,
            eos_tok: _,
        } = *self;
        state.free_gpu(gpu);
        weights.free_gpu(gpu);
    }
}

/// Maple's end-of-turn token. The checkpoint ships the Qwen tokenizer and
/// `config.json` declares `eos_token_id` 151645 (`<|im_end|>`), not the
/// `<|endoftext|>` (151643) that a vocab heuristic would pick.
pub const MAPLE_EOS_FALLBACK: u32 = 151645;

/// Resolve the end-of-turn id from the tokenizer, falling back to the
/// config-declared ChatML id.
pub fn resolve_eos(tokenizer: &hipfire_runtime::tokenizer::Tokenizer) -> u32 {
    for s in ["<|im_end|>", "<|endoftext|>"] {
        let ids = tokenizer.encode(s);
        if ids.len() == 1 {
            return ids[0];
        }
    }
    MAPLE_EOS_FALLBACK
}

/// Load a Maple bundle from an already-open HFQ file.
///
/// Split out from the carrier so an offline harness (the coherence example)
/// can build the same bundle without going through the loader registry.
/// `kv_mode_raw` is the UNRESOLVED request string (`--kv-mode`, `""` for the
/// default). It is resolved here rather than by the caller because this is the
/// first point where `config.head_dim` exists, and `resolve` takes it. Modes
/// outside `MAPLE_POLICY`'s accept set fall back to q8 with a warning.
pub fn load_maple_from_hfq(
    hfq: &mut HfqFile,
    gpu: &mut Gpu,
    max_seq: usize,
    kv_mode_raw: &str,
) -> Result<MapleBundle, String> {
    let config = MapleConfig::from_hfq(hfq)?;
    let weights = MapleWeights::load(hfq, &config, gpu)?;
    let hipfire_runtime::kv_mode::ResolveResult { mode, warning } =
        hipfire_runtime::kv_mode::resolve(
            kv_mode_raw,
            &hipfire_runtime::kv_mode::MAPLE_POLICY,
            config.head_dim,
        );
    if let Some(w) = warning {
        eprintln!("  KV cache: {w} (site maple)");
    }
    let state = MapleState::new_with_max_seq(gpu, &config, max_seq, mode)?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("maple: tokenizer not found: {e}"))?;
    let eos_tok = resolve_eos(&tokenizer);
    Ok(MapleBundle {
        config,
        weights,
        state,
        eos_tok,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eos_fallback_is_im_end_not_endoftext() {
        // config.json declares eos_token_id 151645. A vocab heuristic that
        // defaults to <|endoftext|> (151643) would end turns on the wrong
        // token and run to max_tokens on every reply.
        assert_eq!(MAPLE_EOS_FALLBACK, 151645);
        assert_ne!(MAPLE_EOS_FALLBACK, 151643);
    }
}
