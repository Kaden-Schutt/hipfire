// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::arch::MiniMaxM2;
use crate::minimax::{config_from_safetensors, load_weights_from_safetensors, MiniMaxState};
use crate::spec_impl::MiniMaxBundle;
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

/// Build the MiniMax-M2 GPU bundle from an HFQ or safetensors-directory source.
///
/// Verbatim relocation of the `MinimaxCarrier::load` model work. Preserves every
/// early-return and error string byte-for-byte. The per-source pp refusals,
/// config/weights seam, `maybe_screen_mmq`, `MiniMaxState::new_with_max_seq`,
/// and eos candidate list `["[e~[", "<|im_end|>", "</s>", "<|endoftext|>"]`
/// (fallback `1`) are arch knowledge and live here. `dir_diag`,
/// `resolve_source_meta`, `build_speculator`, and `LoadedModel::skeleton` stay
/// in the loader (loader-private).
pub fn load_minimax_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<MiniMaxBundle, String> {
    if ctx.pp > 1 {
        return Err(match &src {
            ModelSource::Hfq(_) => "minimax: pipeline-parallel (pp>1) unsupported",
            ModelSource::Dir(_) => "minimax: safetensors + pp>1 unsupported",
        }
        .into());
    }
    match src {
        ModelSource::Hfq(mut hfq_file) => {
            let config = <MiniMaxM2 as Architecture>::config_from_hfq(&hfq_file)?;
            let weights = <MiniMaxM2 as Architecture>::load_weights(&mut hfq_file, &config, ctx.gpu)?;
            hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
            let state = MiniMaxState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
                .map_err(|e| format!("minimax: MiniMaxState::new_with_max_seq failed: {e}"))?;
            let tokenizer =
                hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq_file.metadata_json)
                    .map_err(|e| format!("tokenizer not found: {e}"))?;
            let eos_tok = resolve_eos_tok(
                &tokenizer,
                &["[e~[", "<|im_end|>", "</s>", "<|endoftext|>"],
            );
            Ok(MiniMaxBundle {
                config,
                weights,
                state,
                eos_tok,
            })
        }
        ModelSource::Dir(source) => {
            let config = config_from_safetensors(&source)
                .map_err(|e| format!("failed to parse MiniMax config from config.json: {e}"))?;
            let weights = load_weights_from_safetensors(&source, &config, ctx.gpu)?;
            hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
            let state = MiniMaxState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
                .map_err(|e| format!("minimax: MiniMaxState::new_with_max_seq failed: {e}"))?;
            let tokenizer = tokenizer_from_dir(&source)?;
            let eos_tok = resolve_eos_tok(
                &tokenizer,
                &["[e~[", "<|im_end|>", "</s>", "<|endoftext|>"],
            );
            Ok(MiniMaxBundle {
                config,
                weights,
                state,
                eos_tok,
            })
        }
    }
}

