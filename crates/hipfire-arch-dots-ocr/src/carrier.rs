// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! dots.ocr carrier load body, relocated from `hipfire-loader`'s
//! `carriers.rs` so the arch crate owns its model-loading work. The loader
//! keeps the `Carrier` trait boilerplate, tokenizer/chat-template
//! resolution, the model-free speculator wiring, and `LoadedModel`
//! assembly (arch_id=8 stores config/weights/state as `LoadedModel`
//! side-fields, not a `ModelState` variant).

use crate::dots_ocr::{DotsOcrConfig, DotsOcrWeights};
use crate::spec_impl::DotsOcrBundle;
use crate::DotsOcr;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

/// Build the dots.ocr GPU bundle from an HFQ or safetensors-directory
/// source. Text-side load delegates to Qwen2; the vision tower loads via
/// the source loaders. Refusals owned by the loader (pp>1) stay there.
pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<DotsOcrBundle, String> {
    // ── source-varying seam: (config, weights) only ──
    let (config, weights) = match src {
        ModelSource::Hfq(mut hfq) => {
            let config = <DotsOcr as Architecture>::config_from_hfq(&hfq)?;
            let weights = <DotsOcr as Architecture>::load_weights(&mut hfq, &config, ctx.gpu)?;
            (config, weights)
        }
        ModelSource::Dir(source) => {
            let config = DotsOcrConfig::from_source(&source)?;
            let weights = DotsOcrWeights::load_weights_from_source(&source, &config, ctx.gpu)?;
            (config, weights)
        }
    };
    hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
    let state = hipfire_arch_qwen2::qwen2::Qwen2State::new_with_max_seq(
        ctx.gpu,
        &config.text,
        ctx.max_seq,
    )
    .map_err(|e| format!("dots-ocr: Qwen2State::new_with_max_seq failed: {e:?}"))?;
    Ok(DotsOcrBundle {
        config,
        weights,
        state,
    })
}
