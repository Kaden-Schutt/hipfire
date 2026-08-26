// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::config::Lfm2MoeConfig;
use crate::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
use crate::spec_impl::Lfm2MoeBundle;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use hipfire_runtime::model_source::ModelSource as ModelSourceTrait;

fn resolve_eos_tok(tokenizer: &hipfire_runtime::tokenizer::Tokenizer, candidates: &[&str]) -> u32 {
    for s in candidates {
        let ids = tokenizer.encode(s);
        if ids.len() == 1 {
            return ids[0];
        }
    }
    1
}

fn tokenizer_from_dir(
    source: &hipfire_runtime::safetensors_source::SafetensorsSource,
) -> Result<hipfire_runtime::tokenizer::Tokenizer, String> {
    if let Some(tok_path) = source.tokenizer_json_path() {
        hipfire_runtime::tokenizer::Tokenizer::from_tokenizer_json(&tok_path)
            .map_err(|e| format!("failed to parse tokenizer at {}: {e}", tok_path.display()))?
            .ok_or_else(|| format!("failed to load tokenizer from {}", tok_path.display()))
    } else {
        Err("no tokenizer.json found in model directory".into())
    }
}

/// Build the LFM2.5-MoE GPU bundle from an HFQ or safetensors-directory source.
///
/// Verbatim relocation of the `Lfm2MoeCarrier::load` model work. Preserves every
/// early-return and error string byte-for-byte. The per-source pp refusals,
/// config/weights seam (`Lfm2MoeConfig::from_hfq` / `config_from_source` and
/// `Lfm2MoeWeights::load` / `load_weights_from_source`), `maybe_screen_mmq`,
/// `Lfm2MoeState::new_with_max_seq`, and eos candidate list
/// `["<|im_end|>", "</s>", "<|endoftext|>"]` (fallback `1`) are arch knowledge
/// and live here. `dir_diag`, `resolve_source_meta`, `build_speculator`, and
/// `LoadedModel::skeleton` stay in the loader (loader-private).
pub fn load_lfm2moe_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Lfm2MoeBundle, String> {
    if ctx.pp > 1 {
        return Err(match &src {
            ModelSource::Hfq(_) => "lfm2moe: pipeline-parallel (pp>1) unsupported",
            ModelSource::Dir(_) => "lfm2moe: safetensors + pp>1 unsupported",
        }
        .into());
    }
    match src {
        ModelSource::Hfq(mut hfq) => {
            let config = Lfm2MoeConfig::from_hfq(&hfq)?;
            let weights = Lfm2MoeWeights::load(&mut hfq, &config, ctx.gpu)?;
            hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
            let state = Lfm2MoeState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
                .map_err(|e| format!("lfm2moe: Lfm2MoeState::new_with_max_seq failed: {e}"))?;
            let tokenizer =
                hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                    .map_err(|e| format!("tokenizer not found: {e}"))?;
            let eos_tok = resolve_eos_tok(&tokenizer, &["<|im_end|>", "</s>", "<|endoftext|>"]);
            Ok(Lfm2MoeBundle {
                config,
                weights,
                state,
                eos_tok,
                lfm2_decode_batch: None,
            })
        }
        ModelSource::Dir(source) => {
            let config = crate::config::config_from_source(&source).ok_or_else(|| {
                "lfm2moe: failed to parse config from safetensors".to_string()
            })?;
            let weights = crate::lfm2moe::load_weights_from_source(&source, &config, ctx.gpu)?;
            hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
            let state = Lfm2MoeState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
                .map_err(|e| format!("lfm2moe: Lfm2MoeState::new_with_max_seq failed: {e}"))?;
            let tokenizer = tokenizer_from_dir(&source)?;
            let eos_tok = resolve_eos_tok(&tokenizer, &["<|im_end|>", "</s>", "<|endoftext|>"]);
            Ok(Lfm2MoeBundle {
                config,
                weights,
                state,
                eos_tok,
                lfm2_decode_batch: None,
            })
        }
    }
}

