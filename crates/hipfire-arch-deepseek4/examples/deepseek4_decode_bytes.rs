// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
//
//! Estimate compulsory resident-model bytes read by one DeepSeek-V4 AR token.
//!
//! This is a model-traffic denominator, not a hardware-counter claim. It
//! resolves an attached REAP overlay by name, counts every non-MTP layer tensor
//! once, six of 256 routed experts per layer, one embedding row, and the full
//! output head. Divide by measured seconds/token for effective model bandwidth.

use hipfire_arch_deepseek4::DeepseekV4;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::model_source::ModelSource;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: deepseek4_decode_bytes <model.hfq>")?;
    let hfq = HfqFile::open(Path::new(&path))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;

    let mut dense_layer = 0_u128;
    let mut routed_all = 0_u128;
    let mut head = 0_u128;
    let mut embedding_row = 0_u128;
    let mut global = 0_u128;
    let mut counted = 0_usize;

    for name in hfq.tensor_names() {
        let Some(info) = hfq.find_tensor_info(name) else {
            continue;
        };
        let bytes = info.data_size as u128;
        if name.starts_with("mtp.") || name.starts_with("dspark.") {
            continue;
        }
        if name.contains(".ffn.experts.") {
            routed_all += bytes;
            counted += 1;
        } else if name == "head.weight" {
            head += bytes;
            counted += 1;
        } else if name == "embed.weight" || name.contains("embed_tokens") {
            let rows = info.shape.first().copied().unwrap_or(1).max(1) as u128;
            embedding_row += bytes.div_ceil(rows);
            counted += 1;
        } else if name.starts_with("layers.") {
            // All layer-resident tensors participate in forward or are tiny
            // routing/position constants. Counting the latter is conservative
            // and avoids silently omitting a newly added DS4 component.
            dense_layer += bytes;
            counted += 1;
        } else if name.contains("norm") {
            global += bytes;
            counted += 1;
        }
    }

    let routed_active = routed_all * cfg.num_experts_per_tok as u128 / cfg.n_routed_experts as u128;
    let total = dense_layer + routed_active + head + embedding_row + global;
    let gib = |bytes: u128| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("model={path}");
    println!("num_experts_per_tok={}", cfg.num_experts_per_tok);
    println!("n_routed_experts={}", cfg.n_routed_experts);
    println!("resolved_tensors_counted={counted}");
    println!(
        "dense_layer_bytes={dense_layer} ({:.6} GiB)",
        gib(dense_layer)
    );
    println!(
        "routed_active_bytes={routed_active} ({:.6} GiB; {}/{} of all routed payload)",
        gib(routed_active),
        cfg.num_experts_per_tok,
        cfg.n_routed_experts
    );
    println!("head_bytes={head} ({:.6} GiB)", gib(head));
    println!(
        "embedding_row_bytes={embedding_row} ({:.6} GiB)",
        gib(embedding_row)
    );
    println!("global_bytes={global} ({:.6} GiB)", gib(global));
    println!(
        "effective_model_bytes_per_token={total} ({:.6} GiB)",
        gib(total)
    );
    Ok(())
}
