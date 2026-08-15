// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.


#![allow(dead_code, unused_imports, unused_variables, non_snake_case, clippy::all)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::Write;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use clap::Parser;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};
use hipfire_quantize::hessian_io;
use crate::e8;
use crate::e8_gptq;
use crate::gguf_input;
use crate::reap_overlay;

// ─── Model Discovery ────────────────────────────────────────────────────────

pub(crate) fn find_safetensors(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    files.sort();
    files
}

/// Determine which tensors to quantize (weight matrices) vs keep as F16 (norms, embeddings)
pub(crate) fn should_quantize(name: &str) -> bool {
    // Vision encoder weights stay FP16 (only ~500M params, run once per image).
    // Qwen3.5-VL uses `model.visual.*` / `visual.*`; dots.ocr uses
    // `vision_tower.*`. Glimmer uses `model.vision_tower.*`,
    // `model.vision_adapter.*`, `model.vision_projection.*`. All vision stays
    // F16 during bring-up so the per-stage diff against the HF reference
    // activations (`benchmarks/references/<image>_activations/`) doesn't have
    // to absorb both forward-pass implementation noise AND quant noise — clean
    // attribution. See memory `feedback_dots_ocr_vision_f16_during_bringup`.
    if name.starts_with("model.visual.")
        || name.starts_with("visual.")
        || name.starts_with("vision_tower.")
        || name.starts_with("model.vision_tower.")
        || name.starts_with("model.vision_adapter.")
        || name.starts_with("model.vision_projection.")
    {
        return false;
    }
    if name.contains("norm") || name.contains("bias") {
        return false;
    }
    // Quantize everything including embeddings (Q8 embedding saves ~2.3GB for 8B models)
    name.contains("weight")
}

/// antirez ds4 reference keeps three classes at F16 because Q8 measurably
/// regresses PPL on DeepSeek V4: (1) attn compressor wkv + wgate, (2) indexer wq_b +
/// weights_proj, (3) indexer compressor wkv + wgate. All small (≤32 MiB
/// combined across 43 layers).
///
/// Router gate.weight (.ffn.gate.weight) is NOT kept at F16: antirez
/// actually ships it as MQ4G256, and the known-good DeepSeek V4 quant
/// matches. Falling back to the format's default (Q8F16 in deepseek4-q8-mtp)
/// is fine — the router is dispatched via `gemv_auto`.
///
/// `attn.indexer.compressor.*` is a substring of `attn.compressor.*` only
/// in the literal-prefix sense, so order doesn't matter — the substring
/// `.compressor.wkv.weight` matches both `.attn.compressor.wkv.weight` and
/// `.attn.indexer.compressor.wkv.weight` deliberately.
pub(crate) fn is_deepseek4_keep_f16(name: &str) -> bool {
    name.ends_with(".compressor.wkv.weight")
        || name.ends_with(".compressor.wgate.weight")
        || name.ends_with(".indexer.wq_b.weight")
        || name.ends_with(".indexer.weights_proj.weight")
}

/// Frozen MQ2RXT P3 replacement map.
///
/// This selects only tensors that are MFP4G32E8SOA in the released 0731
/// MQ2R artifact (554 trunk tensors, 24 DSpark tensors). The overlay builder
/// reads the original 0731 checkpoint and encodes these directly as MQ4G256;
/// it never dequantizes the E8 artifact. Routed experts and protected Q8/F16
/// tensors are deliberately absent from the overlay and remain byte-identical
/// to the 0731 MQ2R bases when baked.
pub(crate) fn is_deepseek4_mq2rxt_dense(name: &str) -> bool {
    if name == "head.weight" {
        return true;
    }
    let trunk = name.starts_with("layers.");
    let dspark = name.starts_with("mtp.");
    if !trunk && !dspark {
        return false;
    }
    if [
        ".attn.wq_a.weight",
        ".attn.wq_b.weight",
        ".attn.wkv.weight",
        ".attn.wo_a.weight",
        ".attn.wo_b.weight",
        ".ffn.shared_experts.w1.weight",
        ".ffn.shared_experts.w2.weight",
        ".ffn.shared_experts.w3.weight",
    ]
    .iter()
    .any(|suffix| name.ends_with(suffix))
    {
        return true;
    }
    trunk
        && [
            ".attn.compressor.wkv.weight",
            ".attn.compressor.wgate.weight",
            ".attn.indexer.wq_b.weight",
            ".attn.indexer.weights_proj.weight",
            ".attn.indexer.compressor.wkv.weight",
            ".attn.indexer.compressor.wgate.weight",
            ".ffn.gate.weight",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

pub(crate) fn stamp_deepseek4_mq2rxt_metadata(metadata_json: &str, sidecar: bool) -> Result<String, String> {
    let mut metadata: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|error| format!("MQ2RXT metadata is not valid JSON: {error}"))?;
    let object = metadata
        .as_object_mut()
        .ok_or_else(|| "MQ2RXT metadata must be a top-level object".to_owned())?;
    if object.contains_key("hipfire_quant_recipe") || object.contains_key("mq2rxt_sidecar") {
        return Err("MQ2RXT source metadata already carries a product recipe identity".to_owned());
    }
    object.insert(
        "hipfire_quant_recipe".to_owned(),
        serde_json::json!("deepseek4-mq2rxt-mq4-p3-v1"),
    );
    if sidecar {
        object.insert(
            "mq2rxt_sidecar".to_owned(),
            serde_json::json!({
                "target_recipe": "deepseek4-mq2rxt-mq4-p3-v1",
                "draft_head": "trunk_mq4g256_b4",
                "dense_tier": "MQ4G256",
                "built_by": "deepseek4-mq2rxt-v1",
            }),
        );
    }
    serde_json::to_string(&metadata).map_err(|error| format!("serialize MQ2RXT metadata: {error}"))
}

#[cfg(test)]
mod mq2rxt_recipe_tests {
    use super::{is_deepseek4_mq2rxt_dense, stamp_deepseek4_mq2rxt_metadata};

    #[test]
    pub(crate) fn selector_is_exactly_dense_p3_classes() {
        for name in [
            "head.weight",
            "layers.0.attn.wq_a.weight",
            "layers.42.attn.wo_b.weight",
            "layers.17.ffn.shared_experts.w3.weight",
            "layers.3.attn.compressor.wgate.weight",
            "layers.22.attn.indexer.weights_proj.weight",
            "layers.22.attn.indexer.compressor.wkv.weight",
            "layers.8.ffn.gate.weight",
            "mtp.0.attn.wkv.weight",
            "mtp.2.ffn.shared_experts.w2.weight",
        ] {
            assert!(is_deepseek4_mq2rxt_dense(name), "expected {name}");
        }
        for name in [
            "embed.weight",
            "layers.0.ffn.experts.0.w1.weight",
            "layers.0.attn.q_a_layernorm.weight",
            "mtp.0.ffn.gate.weight",
            "mtp.0.main_proj.weight",
            "mtp.2.confidence_head.proj.weight",
            "mtp.2.markov_head.markov_w1.weight",
        ] {
            assert!(!is_deepseek4_mq2rxt_dense(name), "rejected {name}");
        }
    }

    #[test]
    pub(crate) fn metadata_identity_is_distinct_and_sidecar_is_explicit() {
        let trunk =
            stamp_deepseek4_mq2rxt_metadata(r#"{"architecture":"deepseek4"}"#, false).unwrap();
        assert!(trunk.contains("deepseek4-mq2rxt-mq4-p3-v1"));
        assert!(!trunk.contains("mq2rxt_sidecar"));

        let sidecar =
            stamp_deepseek4_mq2rxt_metadata(r#"{"architecture":"deepseek4"}"#, true).unwrap();
        assert!(sidecar.contains("mq2rxt_sidecar"));
        assert!(sidecar.contains("trunk_mq4g256_b4"));
        assert!(stamp_deepseek4_mq2rxt_metadata(&sidecar, true).is_err());
    }
}

/// For mixed quant: should this tensor be Q8 (fast) or Q4 (compressed)?
/// Q8: attention weights, embeddings, lm_head (need occupancy)
/// Q4: FFN weights (bulk of model, benefits from compression)
/// Which fixed-tier classes a tensor belongs to, for `HIPFIRE_Q8_CLASSES`.
/// Ordered cheapest-to-keep first by measured per-token bytes on a3b:
/// router+gate 11.1 MB, lm_head 270 MB, attention 682 MB (of a 1031 MB fixed
/// tier at MQ4). Attention is 66% of the fixed tier, lm_head 26%.
pub(crate) fn q8_class_of(name: &str) -> Option<&'static str> {
    if name.contains("lm_head") {
        Some("lm_head")
    } else if name.contains("embed") {
        Some("embed")
    } else if name.ends_with("mlp.gate.weight")
        || name.ends_with("mlp.shared_expert_gate.weight")
        || name.ends_with("router.proj.weight")
    {
        Some("router")
    } else if name.contains("self_attn")
        || name.contains("attn_q")
        || name.contains("attn_k")
        || name.contains("attn_v")
        || name.contains("attn_output")
        || name.contains("q_proj")
        || name.contains("k_proj")
        || name.contains("v_proj")
        || name.contains("o_proj")
        // Qwen3.5 DeltaNet attention
        || name.contains("linear_attn")
    {
        Some("attn")
    } else {
        None
    }
}

/// Fixed-tier tensors held at Q8F16 regardless of `--format`.
///
/// `HIPFIRE_Q8_CLASSES=<comma list>` narrows this to a subset of
/// {`lm_head`, `embed`, `router`, `attn`} — the lever for attributing which
/// fixed class actually carries the quality. Measured 2026-08-04: dropping the
/// WHOLE fixed tier Q8 -> MQ4 costs **+35.2% KLD** (0.1742 -> 0.2356) while
/// buying 1.75x decode speed, so the tier is emphatically not free — but the
/// +35% is unattributed across classes whose byte costs differ by 25x.
/// `--no-q8-router` (all classes off) still wins if both are set.
///
/// Note the router (`mlp.gate.weight`) is small but precision-sensitive —
/// flat-routing on a quantized router shifts which experts a token sees — so
/// prefer keeping `router` in the set unless you are explicitly testing it.
pub(crate) fn is_q8_tensor(name: &str) -> bool {
    let Some(class) = q8_class_of(name) else {
        return false;
    };
    // A class named in HIPFIRE_FIXED_TIER is held above --format even if it is
    // not in HIPFIRE_Q8_CLASSES — it just lands on the named dtype instead of Q8.
    if fixed_tier_dtype_for(name).is_some() {
        return true;
    }
    match std::env::var("HIPFIRE_Q8_CLASSES") {
        Ok(list) => list.split(',').any(|c| c.trim() == class),
        Err(_) => true,
    }
}

/// `HIPFIRE_FIXED_TIER=lm_head:mfp4e8soa,attn:mq4` — per-class dtype for the
/// fixed tier. Returns the dtype token for `name`'s class, or `None` to fall
/// back to Q8F16 (the historic behaviour).
///
/// Accepted dtypes: `q8`, `mq4`, `mq3l`, `mfp4e8`, `mfp4e8soa`. Accepted
/// classes: `lm_head`, `embed`, `router`, `attn` (see `q8_class_of`).
pub(crate) fn fixed_tier_dtype_for(name: &str) -> Option<&'static str> {
    let class = q8_class_of(name)?;
    let spec = std::env::var("HIPFIRE_FIXED_TIER").ok()?;
    for entry in spec.split(',') {
        // NOT `?` — a `?` here aborts the whole lookup on the FIRST malformed
        // entry and silently returns None, i.e. every class quietly falls back
        // to Q8 and the encode looks like it worked. Fail loudly instead.
        let Some((c, d)) = entry.split_once(':') else {
            eprintln!(
                "error: HIPFIRE_FIXED_TIER: malformed entry '{entry}' \
                 (expected <class>:<dtype>, e.g. attn:mfp4e8soa)"
            );
            std::process::exit(2);
        };
        if c.trim() == class {
            return match d.trim() {
                "mfp4e8soa" => Some("mfp4e8soa"),
                "mfp4e8" => Some("mfp4e8"),
                "mq4" => Some("mq4"),
                "mq3l" => Some("mq3l"),
                "q8" => None, // explicit q8 == default
                other => {
                    eprintln!(
                        "error: HIPFIRE_FIXED_TIER: unknown dtype '{other}' \
                         (expected q8|mq4|mq3l|mfp4e8|mfp4e8soa)"
                    );
                    std::process::exit(2);
                }
            };
        }
    }
    None
}

/// Qwen3.5 DeltaNet conv1d weight: `{prefix}.linear_attn.conv1d.weight`,
/// shape [conv_channels, 1, 4]. Small (~32K elem) and runs every token —
/// Q8 is the safe default; lossy 4-bit FWHT formats (mq4/mq3) measurably
/// hurt the gated-delta path.
pub(crate) fn is_conv1d_tensor(name: &str) -> bool {
    name.ends_with("conv1d.weight")
}
