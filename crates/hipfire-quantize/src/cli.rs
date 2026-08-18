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

#[derive(Debug, Parser)]
#[command(
    name = "hipfire-quantize",
    version,
    about = "Quantize Hugging Face safetensors or GGUF weights into Hipfire HFQ"
)]
pub(crate) struct QuantizeArgs {
    /// Hugging Face model directory, model ID, or GGUF file.
    #[arg(long, value_name = "PATH_OR_MODEL_ID")]
    pub input: String,

    /// Destination HFQ file.
    #[arg(long, value_name = "PATH")]
    pub output: String,

    /// Quantization recipe or wire format.
    #[arg(long, default_value = "q8f16")]
    pub format: String,

    /// Rayon worker threads (defaults to 80% of available cores).
    #[arg(long, env = "HIPFIRE_QUANT_THREADS", value_name = "N")]
    pub threads: Option<usize>,

    /// Override the architecture ID stamped into the HFQ header.
    #[arg(long, value_name = "ID")]
    pub arch_id: Option<u32>,

    /// Allow an architecture override to move Qwen3 off its pillar IDs.
    #[arg(long)]
    pub force_arch_id: bool,

    /// Reuse the source checkpoint's AWQ sidecars as an imatrix for the
    /// low-bit packers' column weighting. Value is the alpha the source was
    /// AWQ-built with (see `awq_col_weights`); defaults to the CLI's own
    /// --awq default. Off entirely when the flag is absent.
    #[arg(long, value_name = "ALPHA", num_args = 0..=1, default_missing_value = "0.55")]
    pub awq_imatrix: Option<f32>,

    /// Requantize an ordinary checkpoint down to ternary/binary anyway.
    /// See `lowbit_ptq_gate` — this is a measured collapse regime.
    #[arg(long, env = "HIPFIRE_ALLOW_LOWBIT_PTQ")]
    pub allow_lowbit_ptq: bool,

    /// Upstream URL recorded in the output's `hipfire_provenance`.
    #[arg(long, value_name = "URL")]
    pub source_url: Option<String>,

    /// SPDX license recorded in the output's `hipfire_provenance`.
    #[arg(long, value_name = "SPDX")]
    pub license: Option<String>,

    /// Write a ternary model even if the pack-health check says it is
    /// degenerate (see `check_ternary_pack_health`). Research escape hatch.
    #[arg(long, env = "HIPFIRE_ALLOW_DEGENERATE_TERNARY")]
    pub allow_degenerate_ternary: bool,

    /// Emit only tensors selected by a REAP plan.
    #[arg(long, value_name = "PLAN_DIR", conflicts_with = "reap_bake")]
    pub reap_overlay: Option<String>,

    /// Apply a REAP plan while baking a complete model.
    #[arg(long, value_name = "PLAN_DIR", conflicts_with = "reap_overlay")]
    pub reap_bake: Option<String>,

    /// Output path for a REAP overlay or baked model.
    #[arg(long, value_name = "PATH")]
    pub reap_out: Option<String>,

    /// Architecture family used to interpret a REAP plan.
    #[arg(long, value_name = "ARCH")]
    pub reap_arch: Option<String>,

    /// llama.cpp imatrix GGUF used for activation-aware quantization.
    #[arg(long, value_name = "PATH")]
    pub imatrix: Option<PathBuf>,

    /// Per-tensor Hessian directory used by GPTQ-E8 recipes.
    #[arg(long, value_name = "DIR")]
    pub hessian_dir: Option<PathBuf>,

    /// Fraction of hot layers assigned the higher-precision Lloyd tier.
    #[arg(long, env = "HIPFIRE_TIER_RATIO", default_value_t = 0.30)]
    pub tier_ratio: f64,

    /// Force router tensors to Q8.
    #[arg(long)]
    pub q8_router: bool,

    /// Disable the default Q8 protection for conv1d tensors.
    #[arg(long)]
    pub no_q8_conv1d: bool,

    /// Let the FIXED tier (attention / lm_head / embed / router) follow
    /// `--format` instead of being pinned to Q8F16. The fixed tier is ~66% of
    /// per-token decode bytes on a3b, so pinning it at Q8 (1.0625 B/w) rather
    /// than MQ4 (0.53125) doubles the dominant term — this is why `.mq2` reads
    /// 45% MORE bytes/token than `.mq4r` despite being 7 GB smaller on disk.
    /// Required to reproduce `.mq4r`.
    #[arg(long)]
    pub no_q8_router: bool,

    /// Disable K-map precision promotion.
    #[arg(long)]
    pub no_kmap: bool,

    /// Alias for --no-kmap for uniform quantization.
    #[arg(long)]
    pub uniform: bool,

    /// Enable AWQ pre-scaling with the default alpha.
    #[arg(long)]
    pub awq: bool,

    /// Enable AWQ pre-scaling with an explicit alpha.
    #[arg(long, value_name = "ALPHA")]
    pub awq_alpha: Option<f32>,

    /// Enable K-map promotion for dense models.
    #[arg(long)]
    pub kmap_dense: bool,

    /// K-map policy: full, alternating/alt, or typed.
    #[arg(long, default_value = "alternating", value_name = "MODE")]
    pub kmap_mode: String,

    /// Permit research-only uniform MQ2 output.
    #[arg(long)]
    pub allow_mq2: bool,

    /// Permit research-only MQ2-Lloyd output.
    #[arg(long)]
    pub allow_mq2_lloyd: bool,

    /// Permit research-only MQ3-Lloyd output.
    #[arg(long)]
    pub allow_mq3_lloyd: bool,

    /// Permit research-only MQ4-Lloyd output.
    #[arg(long)]
    pub allow_mq4_lloyd: bool,

    /// Include vision tensors that are skipped by default.
    #[arg(long)]
    pub include_vision: bool,

    /// Quantization recipe for included vision tensors.
    #[arg(long, default_value = "", value_name = "FORMAT")]
    pub vision_quant: String,

    /// Ingest only tensors whose names start with this prefix.
    #[arg(long, value_name = "PREFIX")]
    pub include_prefix: Option<String>,
}

/// Refuse an `--arch-id` override that strips a qwen3* model (auto-detected
/// arch 5 or 6) off the pillar arches, unless `--force-arch-id` is given.
/// Keeps the froggeric chat-template pillar + qwen35-crate dispatch intact by
/// construction. No-op for non-qwen models or overrides that stay in {5,6}.
pub(crate) fn guard_qwen3_arch_override(auto_arch_id: u32, arch_id: u32, force_arch_id: bool) {
    let auto_is_qwen3 = matches!(auto_arch_id, 5 | 6);
    let override_off_pillar = !matches!(arch_id, 5 | 6);
    if auto_is_qwen3 && arch_id != auto_arch_id && override_off_pillar && !force_arch_id {
        eprintln!(
            "error: --arch-id {arch_id} moves an auto-detected qwen3* model (arch {auto_arch_id}) \
             OFF the pillar arches {{5,6}}; this would break the froggeric chat-template pillar \
             and the qwen35-crate dispatch. Pass --force-arch-id to override anyway."
        );
        std::process::exit(1);
    }
}
