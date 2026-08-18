// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::cohere2moe::Cohere2MoeState;
use crate::config::Cohere2MoeConfig;
use crate::paro_dir;
use crate::spec_impl::Cohere2MoeBundle;
use hipfire_runtime::arch::Architecture;
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

/// Build the Cohere2-MoE GPU bundle from an HFQ or safetensors-directory source.
///
/// Verbatim relocation of the `Cohere2MoeCarrier::load` model work plus the
/// extracted `crate::load_cohere2moe` HFQ helper. Preserves every early-return
/// and error string byte-for-byte, including the per-source pp refusal
/// (`"cohere2moe: pp>1 unsupported via registry"`), the HFQ tokenizer error
/// (`"cohere2moe: tokenizer not found: {e}"`), the Dir config error
/// (`"failed to parse Cohere2-MoE config from config.json: {e}"`), and the
/// distinct eos fallbacks: HFQ `unwrap_or(255001)` vs Dir `resolve_eos_tok`
/// fallback `1` with candidates `["<|END_OF_TURN_TOKEN|>", "</s>", "<|endoftext|>"]`.
///
/// `dir_diag`, `resolve_source_meta`, `build_speculator`, `resolve_chat_template`,
/// and `LoadedModel::skeleton` stay in the loader (loader-private / cycle edge).
pub fn load_cohere2moe_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Cohere2MoeBundle, String> {
    if ctx.pp > 1 {
        return Err("cohere2moe: pp>1 unsupported via registry".into());
    }
    match src {
        ModelSource::Hfq(mut hfq) => {
            // HFQ path — mirrors `crate::load_cohere2moe` (config/weights/state/eos)
            // without the loader's `LoadedModel`/`chat_template` tail.
            let config = <crate::arch::Cohere2Moe as Architecture>::config_from_hfq(&hfq)?;
            let weights =
                <crate::arch::Cohere2Moe as Architecture>::load_weights(&mut hfq, &config, ctx.gpu)?;
            let state = Cohere2MoeState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
                .map_err(|e| format!("cohere2moe: new_with_max_seq failed: {e}"))?;
            let tokenizer =
                hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                    .map_err(|e| format!("cohere2moe: tokenizer not found: {e}"))?;
            // Preserve HFQ-specific eos fallback: 255001 (Dir uses 1 via resolve_eos_tok).
            let eos_tok: u32 = {
                let try_one = |s: &str| -> Option<u32> {
                    let ids = tokenizer.encode(s);
                    if ids.len() == 1 { Some(ids[0]) } else { None }
                };
                try_one("<|END_OF_TURN_TOKEN|>")
                    .or_else(|| try_one("</s>"))
                    .or_else(|| try_one("<|endoftext|>"))
                    .unwrap_or(255001)
            };
            Ok(Cohere2MoeBundle {
                config,
                weights,
                state,
                eos_tok,
            })
        }
        ModelSource::Dir(source) => {
            let config = Cohere2MoeConfig::from_safetensors(&source)
                .map_err(|e| format!("failed to parse Cohere2-MoE config from config.json: {e}"))?;
            let weights = paro_dir::load_from_source(&source, &config, ctx.gpu)?;
            let state = Cohere2MoeState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
                .map_err(|e| format!("cohere2moe: new_with_max_seq failed: {e}"))?;
            let tokenizer = tokenizer_from_dir(&source)?;
            let eos_tok = resolve_eos_tok(
                &tokenizer,
                &["<|END_OF_TURN_TOKEN|>", "</s>", "<|endoftext|>"],
            );
            Ok(Cohere2MoeBundle {
                config,
                weights,
                state,
                eos_tok,
            })
        }
    }
}

