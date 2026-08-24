// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `load_<arch>_bundle` entry point for the toy arch — the shape the loader's
//! `Carrier` impl calls into.
//!
//! A real arch's carrier (`hipfire-loader/src/carriers.rs`) does the
//! source-varying glue (tokenizer, chat template, speculator,
//! `LoadedModel::skeleton`) and delegates ALL model work to a function with
//! exactly this signature living in the arch crate — see
//! `hipfire-arch-minimax/src/carrier.rs` for the smallest current example.
//!
//! The toy is NEVER loadable. This body is an honest stub: it returns an
//! error naming what a real implementation does, because the toy has no
//! weights to load and `arch_id 0xFF` must never reach a carrier anyway.

use crate::arch_model::ToyBundle;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

/// Build the toy bundle from a model source.
///
/// ⚠️ DO NOT SHIP. Toy's `arch_id` is 0xFF, which is deliberately
/// UNREGISTERED: no carrier in `hipfire-loader/src/carriers.rs` claims it,
/// `load_model` fails closed with "no carrier for …", and the daemon can
/// never dispatch it. This function exists so the template demonstrates the
/// exact signature + call-site contract — not so the toy ever loads.
pub fn load_toy_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<ToyBundle, String> {
    let _ = (src, ctx);
    Err(
        "toy is a template, not a model: a real arch's load_<arch>_bundle \
         parses config from `src` (HFQ metadata_json or safetensors \
         config.json), uploads weights to `ctx.gpu`, allocates per-decode \
         GPU state sized by that config and `ctx.max_seq`, resolves eos, and \
         returns its Bundle. Toy has none of that — copy this crate and fill \
         it in (see README.md)."
            .into(),
    )
}
