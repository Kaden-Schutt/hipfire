// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    non_snake_case,
    clippy::all
)]

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use crate::e8;
use crate::e8_gptq;
use crate::gguf_input;
use crate::reap_overlay;
use clap::Parser;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::hessian_io;
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};

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
    name.contains("weight")
}

pub(crate) fn is_deepseek4_keep_f16(name: &str) -> bool {
    name.ends_with(".compressor.wkv.weight")
        || name.ends_with(".compressor.wgate.weight")
        || name.ends_with(".indexer.wq_b.weight")
        || name.ends_with(".indexer.weights_proj.weight")
}

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

pub(crate) fn stamp_deepseek4_mq2rxt_metadata(
    metadata_json: &str,
    sidecar: bool,
) -> Result<String, String> {
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
/// `ssm_out` is the architecture-neutral name for `linear_attn.out_proj.weight`
/// — the Qwen3.8 DeltaNet recurrent SSM state — checked most-specifically before
/// the broad `linear_attn` -> `attn` bucket.
///
/// `attn_full` is a more-specific override class (not returned here) for names
/// containing `self_attn` only; see [`is_attn_full_tensor`]. `q8_class_of` still
/// returns `attn` for both full-attention and DeltaNet linear-attention tensors
/// so existing callers keep a single attention bucket.
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
    } else if name.contains("linear_attn.out_proj") || name.contains("ssm_out") {
        Some("ssm_out")
    } else if name.contains("self_attn")
        || name.contains("attn_q")
        || name.contains("attn_k")
        || name.contains("attn_v")
        || name.contains("attn_output")
        || name.contains("q_proj")
        || name.contains("k_proj")
        || name.contains("v_proj")
        || name.contains("o_proj")
        // Qwen3.5 DeltaNet attention — broad bucket, after ssm_out specificity
        || name.contains("linear_attn")
    {
        Some("attn")
    } else {
        None
    }
}

/// Full-attention tensors only (`self_attn`). Rejects DeltaNet `linear_attn`
/// and other generic attention name shapes that still map to class `attn`.
pub(crate) fn is_attn_full_tensor(name: &str) -> bool {
    name.contains("self_attn")
}

// ── Strict validation for env/CLI tokens ────────────────────────────────

pub(crate) const VALID_Q8_CLASSES: &[&str] =
    &["lm_head", "embed", "router", "attn", "attn_full", "ssm_out"];
pub(crate) const VALID_FIXED_DTYPES: &[&str] = &[
    "q8",
    "mq2v2",
    "mq3v2",
    "mq4v2",
    "mq5v2",
    "mq6v2",
    // legacy compat — still accepted but not advertised for new products
    "mq4",
    "mq3l",
    "mfp4e8",
    "mfp4e8soa",
];

pub(crate) fn validate_q8_classes_spec(spec: &str) -> Result<(), String> {
    for raw in spec.split(',') {
        let c = raw.trim();
        if c.is_empty() {
            continue;
        }
        if !VALID_Q8_CLASSES.contains(&c) {
            return Err(format!(
                "unknown HIPFIRE_Q8_CLASSES class '{c}' (expected one of lm_head,embed,router,attn,attn_full,ssm_out)"
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_fixed_tier_spec(spec: &str) -> Result<(), String> {
    for entry in spec.split(',') {
        let e = entry.trim();
        if e.is_empty() {
            continue;
        }
        let Some((c, d)) = e.split_once(':') else {
            return Err(format!(
                "malformed HIPFIRE_FIXED_TIER entry '{e}' (expected <class>:<dtype>, e.g. lm_head:mq6v2)"
            ));
        };
        let class = c.trim();
        let dtype = d.trim();
        if !VALID_Q8_CLASSES.contains(&class) {
            return Err(format!(
                "unknown HIPFIRE_FIXED_TIER class '{class}' (expected one of lm_head,embed,router,attn,attn_full,ssm_out)"
            ));
        }
        if !VALID_FIXED_DTYPES.contains(&dtype) {
            return Err(format!(
                "unknown HIPFIRE_FIXED_TIER dtype '{dtype}' (expected q8|mq2v2|mq3v2|mq4v2|mq5v2|mq6v2)"
            ));
        }
    }
    Ok(())
}

/// Fail before worker threads if any unknown class/dtype token appears in the
/// two fixed-tier env strings. Called at CLI boundary before rayon spawn.
pub(crate) fn validate_env_fixed_tier_or_exit() {
    if let Ok(spec) = std::env::var("HIPFIRE_Q8_CLASSES") {
        if let Err(e) = validate_q8_classes_spec(&spec) {
            eprintln!("error: HIPFIRE_Q8_CLASSES: {e}");
            std::process::exit(2);
        }
    }
    if let Ok(spec) = std::env::var("HIPFIRE_FIXED_TIER") {
        if let Err(e) = validate_fixed_tier_spec(&spec) {
            eprintln!("error: HIPFIRE_FIXED_TIER: {e}");
            std::process::exit(2);
        }
    }
}

/// Strict parser for HIPFIRE_Q8_CLASSES / --q8-classes. Returns trimmed list.
pub(crate) fn parse_q8_classes(spec: &str) -> Result<Vec<String>, String> {
    validate_q8_classes_spec(spec)?;
    Ok(spec
        .split(',')
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect())
}

/// Strict parser for HIPFIRE_FIXED_TIER / --fixed-tier. Returns class->dtype map.
pub(crate) fn parse_fixed_tier(spec: &str) -> Result<HashMap<String, String>, String> {
    validate_fixed_tier_spec(spec)?;
    let mut map = HashMap::new();
    for entry in spec.split(',') {
        let e = entry.trim();
        if e.is_empty() {
            continue;
        }
        let (c, d) = e.split_once(':').unwrap();
        map.insert(c.trim().to_string(), d.trim().to_string());
    }
    Ok(map)
}

/// Product tier for Qwen3.8 ladder. XT keeps lm_head at base codec, base lifts
/// lm_head, pro also lifts ssm_out (linear_attn.out_proj). Embed and conv1d stay
/// Q8 at every rung; structural norms stay F16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProductTier {
    Xt,
    Base,
    Pro,
}

impl ProductTier {
    pub(crate) fn from_flag(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "xt" => Some(Self::Xt),
            "base" => Some(Self::Base),
            "pro" => Some(Self::Pro),
            _ => None,
        }
    }
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::Xt => "xt",
            Self::Base => "base",
            Self::Pro => "pro",
        }
    }
    /// Which classes are held above base codec at this rung? Includes embed
    /// and ssm_out; conv1d is covered separately via is_conv1d_tensor.
    pub(crate) fn lifted_classes(&self) -> &'static [&'static str] {
        match self {
            Self::Xt => &["embed"],
            Self::Base => &["embed", "lm_head"],
            Self::Pro => &["embed", "lm_head", "ssm_out"],
        }
    }
    pub(crate) fn lifts(&self, class: &str) -> bool {
        self.lifted_classes().contains(&class)
    }
}

/// Fixed-tier tensors held at Q8F16 regardless of `--format`.
///
/// `HIPFIRE_Q8_CLASSES=<comma list>` narrows this to a subset of
/// {`lm_head`, `embed`, `router`, `attn`, `attn_full`, `ssm_out`} — the lever for
/// attributing which fixed class actually carries the quality. `attn_full`
/// retains only `self_attn` tensors at Q8; generic `attn` still covers both
/// full-attention and DeltaNet linear-attention when listed. Measured 2026-08-04:
/// dropping the WHOLE fixed tier Q8 -> MQ4 costs **+35.2% KLD** (0.1742 -> 0.2356)
/// while buying 1.75x decode speed, so the tier is emphatically not free — but the
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
    if fixed_tier_dtype_for(name).is_some() {
        return true;
    }
    match std::env::var("HIPFIRE_Q8_CLASSES") {
        Ok(list) => {
            // validated at startup; still handle empty list as no lift.
            // `attn_full` independently retains self-attention without linear_attn.
            list.split(',').any(|c| {
                let c = c.trim();
                c == class || (c == "attn_full" && is_attn_full_tensor(name))
            })
        }
        Err(_) => true,
    }
}

/// `HIPFIRE_FIXED_TIER=lm_head:mq6v2,ssm_out:mq5v2` — per-class dtype for the
/// fixed tier. Returns the dtype token for `name`'s class, or `None` to fall
/// back to Q8F16 (the historic behaviour).
///
/// Accepted dtypes: `q8`, `mq2v2`, `mq3v2`, `mq4v2`, `mq5v2`, `mq6v2` (plus legacy
/// `mq4`, `mq3l`, `mfp4e8`, `mfp4e8soa`). Accepted
/// classes: `lm_head`, `embed`, `router`, `attn`, `attn_full`, `ssm_out`
/// (see `q8_class_of` / `is_attn_full_tensor`).
///
/// For full-attention names, an `attn_full` map entry wins over generic `attn`.
/// Explicit `q8` still means no codec override (fall back to Q8F16).
pub(crate) fn fixed_tier_dtype_for(name: &str) -> Option<&'static str> {
    let class = q8_class_of(name)?;
    // Prefer CLI map if set (populated from --fixed-tier), else env.
    if let Some(map) = fixed_tier_map_cli() {
        if is_attn_full_tensor(name) {
            if let Some(dtype) = map.get("attn_full") {
                return intern_fixed_dtype(dtype);
            }
        }
        if let Some(dtype) = map.get(class) {
            return intern_fixed_dtype(dtype);
        }
        return None;
    }
    let spec = std::env::var("HIPFIRE_FIXED_TIER").ok()?;
    let mut attn_full_hit: Option<&'static str> = None;
    let mut class_hit: Option<&'static str> = None;
    // sentinel: distinguish "missing" from "explicit q8 => None"
    let mut saw_attn_full = false;
    let mut saw_class = false;
    for entry in spec.split(',') {
        let e = entry.trim();
        if e.is_empty() {
            continue;
        }
        let Some((c, d)) = e.split_once(':') else {
            eprintln!(
                "error: HIPFIRE_FIXED_TIER: malformed entry '{e}' \
                 (expected <class>:<dtype>, e.g. lm_head:mq6v2)"
            );
            std::process::exit(2);
        };
        let c = c.trim();
        let d = d.trim();
        if c == "attn_full" && is_attn_full_tensor(name) {
            saw_attn_full = true;
            attn_full_hit = match d {
                "q8" => None,
                other => Some(intern_fixed_dtype_or_exit(other)),
            };
        } else if c == class {
            saw_class = true;
            class_hit = match d {
                "q8" => None,
                other => Some(intern_fixed_dtype_or_exit(other)),
            };
        }
    }
    if saw_attn_full {
        return attn_full_hit;
    }
    if saw_class {
        return class_hit;
    }
    None
}

fn intern_fixed_dtype(dtype: &str) -> Option<&'static str> {
    match dtype {
        "q8" => None,
        "mfp4e8soa" => Some("mfp4e8soa"),
        "mfp4e8" => Some("mfp4e8"),
        "mq4" => Some("mq4"),
        "mq3l" => Some("mq3l"),
        "mq2v2" => Some("mq2v2"),
        "mq3v2" => Some("mq3v2"),
        "mq4v2" => Some("mq4v2"),
        "mq5v2" => Some("mq5v2"),
        "mq6v2" => Some("mq6v2"),
        _ => None,
    }
}

fn intern_fixed_dtype_or_exit(dtype: &str) -> &'static str {
    match dtype {
        "mfp4e8soa" => "mfp4e8soa",
        "mfp4e8" => "mfp4e8",
        "mq4" => "mq4",
        "mq3l" => "mq3l",
        "mq2v2" => "mq2v2",
        "mq3v2" => "mq3v2",
        "mq4v2" => "mq4v2",
        "mq5v2" => "mq5v2",
        "mq6v2" => "mq6v2",
        other => {
            eprintln!(
                "error: HIPFIRE_FIXED_TIER: unknown dtype '{other}' \
                 (expected q8|mq2v2|mq3v2|mq4v2|mq5v2|mq6v2)"
            );
            std::process::exit(2);
        }
    }
}

// CLI-provided fixed-tier map, set once at startup before threads.
static FIXED_TIER_CLI: OnceLock<Option<HashMap<String, String>>> = OnceLock::new();

pub(crate) fn set_fixed_tier_cli(spec: Option<String>) {
    let parsed = match spec {
        Some(s) => {
            if s.trim().is_empty() {
                Some(HashMap::new())
            } else {
                match parse_fixed_tier(&s) {
                    Ok(m) => Some(m),
                    Err(e) => {
                        eprintln!("error: --fixed-tier: {e}");
                        std::process::exit(2);
                    }
                }
            }
        }
        None => None,
    };
    let _ = FIXED_TIER_CLI.set(parsed);
}

pub(crate) fn fixed_tier_map_cli() -> Option<HashMap<String, String>> {
    FIXED_TIER_CLI.get().cloned().flatten()
}

pub(crate) fn is_fixed_tier_cli_set() -> bool {
    FIXED_TIER_CLI.get().is_some()
}

static PRODUCT_TIER_CLI: OnceLock<Option<ProductTier>> = OnceLock::new();

pub(crate) fn set_product_tier_cli(tier: Option<ProductTier>) {
    let _ = PRODUCT_TIER_CLI.set(tier);
}

pub(crate) fn product_tier_cli() -> Option<ProductTier> {
    PRODUCT_TIER_CLI.get().copied().flatten()
}

/// Qwen3.5 DeltaNet conv1d weight: `{prefix}.linear_attn.conv1d.weight`,
/// shape [conv_channels, 1, 4]. Small (~32K elem) and runs every token —
/// Q8 is the safe default; lossy 4-bit FWHT formats (mq4/mq3) measurably
/// hurt the gated-delta path.
pub(crate) fn is_conv1d_tensor(name: &str) -> bool {
    name.ends_with("conv1d.weight")
}

#[cfg(test)]
mod product_tier_tests {
    use super::*;

    #[test]
    fn q8_class_of_exact_roles() {
        // lm_head
        assert_eq!(q8_class_of("model.lm_head.weight"), Some("lm_head"));
        assert_eq!(q8_class_of("lm_head.weight"), Some("lm_head"));
        // embed
        assert_eq!(q8_class_of("model.embed_tokens.weight"), Some("embed"));
        assert_eq!(q8_class_of("embed.weight"), Some("embed"));
        // router
        assert_eq!(q8_class_of("layers.0.mlp.gate.weight"), Some("router"));
        // ssm_out most-specific before attn
        assert_eq!(
            q8_class_of("model.layers.0.linear_attn.out_proj.weight"),
            Some("ssm_out")
        );
        assert_eq!(q8_class_of("blk.0.ssm_out.weight"), Some("ssm_out"));
        // linear_attn other tensors should be attn, not ssm_out
        assert_eq!(
            q8_class_of("model.layers.0.linear_attn.in_proj.weight"),
            Some("attn")
        );
        assert_eq!(
            q8_class_of("model.layers.0.linear_attn.conv1d.weight"),
            Some("attn")
        );
        // self_attn should be attn
        assert_eq!(
            q8_class_of("model.layers.0.self_attn.q_proj.weight"),
            Some("attn")
        );
        assert_eq!(
            q8_class_of("model.layers.0.self_attn.o_proj.weight"),
            Some("attn")
        );
        // unrelated
        assert_eq!(q8_class_of("model.layers.0.mlp.down_proj.weight"), None);
        assert_eq!(q8_class_of("model.norm.weight"), None);
    }

    #[test]
    fn unknown_token_refusal() {
        assert!(validate_q8_classes_spec("lm_head,embed,attn").is_ok());
        assert!(validate_q8_classes_spec("lm_head,ssm_out").is_ok());
        assert!(validate_q8_classes_spec("attn,unknown").is_err());
        assert!(validate_q8_classes_spec("bogus").is_err());
        assert!(validate_fixed_tier_spec("lm_head:q8").is_ok());
        assert!(validate_fixed_tier_spec("lm_head:mq6v2,ssm_out:mq5v2").is_ok());
        assert!(validate_fixed_tier_spec("lm_head:mq6v2").is_ok());
        assert!(validate_fixed_tier_spec("lm_head:unknown").is_err());
        assert!(validate_fixed_tier_spec("unknown:q8").is_err());
        assert!(validate_fixed_tier_spec("lm_head-q8").is_err());
        assert!(validate_fixed_tier_spec("lm_head:mq4v2,embed:q8").is_ok());
        assert!(validate_q8_classes_spec("attn_full").is_ok());
        assert!(validate_q8_classes_spec("attn,attn_full,ssm_out").is_ok());
        assert!(validate_fixed_tier_spec("ssm_out:mq6v2,attn_full:mq6v2").is_ok());
        assert!(validate_fixed_tier_spec("attn_full:mq6v2").is_ok());
    }

    #[test]
    fn xt_base_pro_class_sets() {
        let xt = ProductTier::Xt;
        assert!(xt.lifts("embed"));
        assert!(!xt.lifts("lm_head"));
        assert!(!xt.lifts("ssm_out"));
        assert!(!xt.lifts("attn"));
        let base = ProductTier::Base;
        assert!(base.lifts("embed"));
        assert!(base.lifts("lm_head"));
        assert!(!base.lifts("ssm_out"));
        let pro = ProductTier::Pro;
        assert!(pro.lifts("embed"));
        assert!(pro.lifts("lm_head"));
        assert!(pro.lifts("ssm_out"));
        assert!(!pro.lifts("attn"));
        assert!(!pro.lifts("router"));
        assert!(!pro.lifts("attn_full"), "Pro must not lift attn_full");
    }

    #[test]
    fn independent_lm_head_ssm_out_lifts() {
        // lm_head:mq6v2 and ssm_out:mq5v2 should parse independently
        let m = parse_fixed_tier("lm_head:mq6v2,ssm_out:mq5v2").unwrap();
        assert_eq!(m.get("lm_head").map(|s| s.as_str()), Some("mq6v2"));
        assert_eq!(m.get("ssm_out").map(|s| s.as_str()), Some("mq5v2"));
        // single lifts
        let m2 = parse_fixed_tier("ssm_out:mq6v2").unwrap();
        assert_eq!(m2.get("ssm_out").unwrap(), "mq6v2");
        assert!(m2.get("lm_head").is_none());
        // q8 explicit
        let m3 = parse_fixed_tier("lm_head:q8").unwrap();
        assert_eq!(m3.get("lm_head").unwrap(), "q8");
    }

    #[test]
    fn no_accidental_promotion_of_self_attn_or_in_proj() {
        // self_attn and linear_attn.in_proj must be attn, not ssm_out, and Pro should not lift them
        assert_eq!(
            q8_class_of("model.layers.5.self_attn.q_proj.weight"),
            Some("attn")
        );
        assert_eq!(
            q8_class_of("model.layers.5.linear_attn.in_proj.weight"),
            Some("attn")
        );
        let pro = ProductTier::Pro;
        assert!(!pro.lifts("attn"), "Pro must not lift attn");
        // ssm_out is the only DeltaNet tensor Pro lifts
        assert_eq!(
            q8_class_of("model.layers.5.linear_attn.out_proj.weight"),
            Some("ssm_out")
        );
        assert!(pro.lifts("ssm_out"));
    }

    #[test]
    fn attn_full_predicate_separates_self_attn_from_linear() {
        assert!(is_attn_full_tensor(
            "model.layers.0.self_attn.q_proj.weight"
        ));
        assert!(is_attn_full_tensor(
            "model.layers.0.self_attn.o_proj.weight"
        ));
        assert!(!is_attn_full_tensor(
            "model.layers.0.linear_attn.in_proj.weight"
        ));
        assert!(!is_attn_full_tensor(
            "model.layers.0.linear_attn.out_proj.weight"
        ));
        assert!(!is_attn_full_tensor(
            "model.layers.0.linear_attn.conv1d.weight"
        ));
        // generic attn shapes without self_attn are not attn_full
        assert!(!is_attn_full_tensor("blk.0.attn_q.weight"));
        assert!(!is_attn_full_tensor("layers.0.q_proj.weight"));
        // q8_class_of still returns attn for self-attention
        assert_eq!(
            q8_class_of("model.layers.0.self_attn.q_proj.weight"),
            Some("attn")
        );
        assert_eq!(
            q8_class_of("model.layers.0.linear_attn.in_proj.weight"),
            Some("attn")
        );
    }

    #[test]
    fn attn_full_lookup_precedes_generic_attn() {
        let m = parse_fixed_tier("ssm_out:mq6v2,attn_full:mq6v2,attn:mq4v2").unwrap();
        assert_eq!(m.get("attn_full").map(|s| s.as_str()), Some("mq6v2"));
        assert_eq!(m.get("attn").map(|s| s.as_str()), Some("mq4v2"));
        assert_eq!(m.get("ssm_out").map(|s| s.as_str()), Some("mq6v2"));

        // Resolve via the same intern path CLI lookup uses.
        let self_attn = "model.layers.0.self_attn.q_proj.weight";
        let linear_attn = "model.layers.0.linear_attn.in_proj.weight";
        let ssm = "model.layers.0.linear_attn.out_proj.weight";

        // Prefer attn_full over attn for self_attn names.
        assert_eq!(resolve_fixed_tier_from_map(self_attn, &m), Some("mq6v2"));
        // linear_attn stays on generic attn — not selected by attn_full.
        assert_eq!(resolve_fixed_tier_from_map(linear_attn, &m), Some("mq4v2"));
        assert_eq!(resolve_fixed_tier_from_map(ssm, &m), Some("mq6v2"));

        // Generic attn alone still covers self_attn when attn_full is absent.
        let attn_only = parse_fixed_tier("attn:mq5v2").unwrap();
        assert_eq!(
            resolve_fixed_tier_from_map(self_attn, &attn_only),
            Some("mq5v2")
        );
        assert_eq!(
            resolve_fixed_tier_from_map(linear_attn, &attn_only),
            Some("mq5v2")
        );

        // attn_full alone does not select linear_attn.
        let full_only = parse_fixed_tier("attn_full:mq6v2").unwrap();
        assert_eq!(
            resolve_fixed_tier_from_map(self_attn, &full_only),
            Some("mq6v2")
        );
        assert_eq!(resolve_fixed_tier_from_map(linear_attn, &full_only), None);

        // explicit q8 means no codec override
        let q8_full = parse_fixed_tier("attn_full:q8,attn:mq4v2").unwrap();
        assert_eq!(resolve_fixed_tier_from_map(self_attn, &q8_full), None);
        assert_eq!(
            resolve_fixed_tier_from_map(linear_attn, &q8_full),
            Some("mq4v2")
        );
    }

    /// Map-path resolution mirroring `fixed_tier_dtype_for` CLI branch without
    /// touching process-global OnceLock / env state.
    fn resolve_fixed_tier_from_map(
        name: &str,
        map: &HashMap<String, String>,
    ) -> Option<&'static str> {
        let class = q8_class_of(name)?;
        if is_attn_full_tensor(name) {
            if let Some(dtype) = map.get("attn_full") {
                return intern_fixed_dtype(dtype);
            }
        }
        map.get(class).and_then(|d| intern_fixed_dtype(d))
    }

    #[test]
    fn pro_still_does_not_lift_attention() {
        let pro = ProductTier::Pro;
        assert_eq!(pro.lifted_classes(), &["embed", "lm_head", "ssm_out"]);
        assert!(!pro.lifts("attn"));
        assert!(!pro.lifts("attn_full"));
        assert!(is_attn_full_tensor(
            "model.layers.0.self_attn.q_proj.weight"
        ));
        assert_eq!(
            q8_class_of("model.layers.0.self_attn.q_proj.weight"),
            Some("attn")
        );
    }

    #[test]
    fn product_tier_from_flag() {
        assert_eq!(ProductTier::from_flag("xt"), Some(ProductTier::Xt));
        assert_eq!(ProductTier::from_flag("XT"), Some(ProductTier::Xt));
        assert_eq!(ProductTier::from_flag("base"), Some(ProductTier::Base));
        assert_eq!(ProductTier::from_flag("pro"), Some(ProductTier::Pro));
        assert_eq!(ProductTier::from_flag("unknown"), None);
    }
}
