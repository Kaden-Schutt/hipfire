// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Loader-facing bundle construction for Maple-Preview.
//!
//! Mirrors `hipfire_arch_cohere2moe::carrier`. HFQ only: there is no
//! safetensors-directory path, because the published checkpoint is 40 GB of
//! dequantized BF16 masters whose whole point is to be packed losslessly into
//! qt=51 first — serving it unpacked would need a BF16 MoE decode path that
//! does not exist and would be ~7× the memory.

use crate::bundle::{load_maple_from_hfq, MapleBundle};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

/// Build the Maple GPU bundle from a loader `ModelSource`.
pub fn load_maple_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<MapleBundle, String> {
    if ctx.pp > 1 {
        return Err("maple: pp>1 unsupported via registry".into());
    }
    match src {
        ModelSource::Hfq(mut hfq) => {
            // Same ladder the other carriers use: an explicit --kv-mode wins,
            // else the global config value. Resolution against MAPLE_POLICY
            // happens inside load_maple_from_hfq, where head_dim is known.
            // Before this, arch 15 hardcoded q8 and --kv-mode was a silent
            // no-op.
            let raw = ctx
                .kv_mode_override
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or_else(|| hipfire_runtime::config::get().kv_mode.clone());
            // A head overlay (hipfire-quantize --head-only) shadows the
            // base's lm_head, so one body serves several head carriers.
            crate::bundle::load_maple_from_hfq_with_head(
                &mut hfq,
                ctx.gpu,
                ctx.max_seq,
                &raw,
                ctx.head_path.map(std::path::Path::new),
            )
        }
        ModelSource::Dir(_) => Err(
            "maple: safetensors-directory loading is unsupported — convert first with \
             `hipfire-quantize --format maple --input <dir> --output <model.hfq>`, which packs \
             the native ternary weights losslessly into qt=51"
                .into(),
        ),
    }
}
