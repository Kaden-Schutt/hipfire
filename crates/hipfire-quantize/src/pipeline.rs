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

use crate::calibration::*;
use crate::cli::*;
use crate::dequant::*;
use crate::e8;
use crate::e8_gptq;
use crate::gguf_input;
use crate::hfq::*;
use crate::model_filter::*;
use crate::pipeline_deepseek::*;
use crate::pipeline_gguf::*;
use crate::quant_e8::*;
use crate::quant_fwht::*;
use crate::quant_hfp4::*;
use crate::quant_mq::*;
use crate::quant_q4::*;
use crate::reap_overlay;
use clap::Parser;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::hessian_io;
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};

// ── Per-tensor grouping for disposition helpers ──────────────────────────
struct PerTensorCtx<'a> {
    name: &'a str,
    file_idx: usize,
    shape: &'a [usize],
    n_elements: usize,
    arch_id: u32,
    dtype: &'a str,
    is_vision: bool,
}
struct MainQuantFlags {
    use_fast: bool,
    use_gptq_e8: bool,
    use_gptq_mfp2e8: bool,
    use_gptq_mfp3e8: bool,
    use_hfp4: bool,
    use_hfq2g128: bool,
    use_hfq2g256: bool,
    use_hfq3g128: bool,
    use_hfq3g256: bool,
    use_hfq4g256: bool,
    use_hfq6: bool,
    use_hfq_mixed: bool,
    use_mfp2e8_gptq_fmt: bool,
    use_mfp3e8_gptq_fmt: bool,
    use_mfp4: bool,
    use_mfp4e8: bool,
    use_mfp4e8soa: bool,
    use_mfp4l: bool,
    use_mfp4p: bool,
    use_mixed: bool,
    use_mq2g256: bool,
    use_mq2g256_lloyd: bool,
    use_mq3g256: bool,
    use_mq3g256_lloyd: bool,
    use_mq4_mq2glexp: bool,
    use_mq4_mq2lloyd_gptq_all: bool,
    use_mq4_mq2lloyd_imatrix: bool,
    use_mq4_mq2lloyd_kmap: bool,
    use_mq4_mq2lloyd_native: bool,
    use_mq4_mq2lloydexp: bool,
    use_mq4_mq3lloyd_kmap: bool,
    use_mq4_mq6exp: bool,
    use_mq4_mqlloyd_antirez: bool,
    use_mq4_mqlloyd_antirez_gptq: bool,
    use_mq4_mqlloyd_tiered: bool,
    use_mq4g256: bool,
    use_mq4v2: bool,
    use_mq4c: bool,
    use_mq4g256_lloyd: bool,
    use_mq5g256: bool,
    use_mq6g256: bool,
    use_mq5g256v2: bool,
    use_mq6g256v2: bool,
    use_mq3g256v2: bool,
    use_mq2g256v2: bool,
    use_mq8g256: bool,
    use_q4k_all: bool,
    use_q4k_q8embed: bool,
    use_q8: bool,
    use_q8hfq: bool,
    is_gemma4_family: bool,
    q8_conv1d_default: bool,
    q8_router: bool,
    arch_id: u32,
    vision_quant: String,
    product_tier: Option<crate::model_filter::ProductTier>,
}

struct MainQuantOuter<'a> {
    kmap: &'a HashMap<String, QuantLevel>,
    imatrix_gguf: &'a Option<gguf_input::GgufFile>,
    hessian_dir: &'a Option<PathBuf>,
}

struct MainQuantState<'a> {
    hfq_tensors: &'a mut Vec<HfqTensor>,
    quantized_params: &'a mut u64,
    total_quant_error: &'a mut f64,
    max_quant_error: &'a mut f32,
    _n_quant_groups: &'a mut u64,
    spill: &'a mut Option<TensorSpill>,
}

struct FormatFlags {
    use_f32_passthrough: bool,
    use_bf16: bool,
    use_q8: bool,
    use_mixed: bool,
    use_fast: bool,
    use_q8hfq: bool,
}

pub(crate) fn run() {
    let args = QuantizeArgs::parse();

    // ── Strict validation before worker threads ──────────────────────────
    // Unknown class/dtype tokens must fail before rayon spawn, and CLI/env
    // parsers must share the same strict set.
    crate::model_filter::validate_env_fixed_tier_or_exit();
    let product_tier: Option<crate::model_filter::ProductTier> = match args.tier.as_deref() {
        Some(s) => match crate::model_filter::ProductTier::from_flag(s) {
            Some(t) => Some(t),
            None => {
                eprintln!("error: --tier unknown tier '{s}' (expected xt|base|pro)");
                std::process::exit(2);
            }
        },
        None => None,
    };
    // CLI --fixed-tier overrides env; parsed and validated via the same strict parser.
    crate::model_filter::set_fixed_tier_cli(args.fixed_tier.clone());
    crate::model_filter::set_product_tier_cli(product_tier);
    if let Some(t) = product_tier {
        eprintln!(
            "product tier: {} (lifted classes: {})",
            t.label(),
            t.lifted_classes().join(",")
        );
        if let Some(map) = crate::model_filter::fixed_tier_map_cli() {
            let mut entries: Vec<String> = map.iter().map(|(k, v)| format!("{k}:{v}")).collect();
            entries.sort();
            eprintln!("fixed-tier overrides: {}", entries.join(","));
        }
    } else if crate::model_filter::fixed_tier_map_cli().is_some() {
        let map = crate::model_filter::fixed_tier_map_cli().unwrap();
        let mut entries: Vec<String> = map.iter().map(|(k, v)| format!("{k}:{v}")).collect();
        entries.sort();
        eprintln!("fixed-tier overrides (CLI): {}", entries.join(","));
    }

    setup_thread_pool(&args);

    let input_dir = args.input.as_str();
    let output_path = args.output.as_str();
    let format = args.format.as_str();

    if handle_early_special_formats(&args) {
        return;
    }

    // SP4 selective re-quant overlay mode. `--reap-overlay <plan-dir>` activates
    // it: instead of quantizing the whole model, only the tensors named by the
    // plan's `quant_overrides` are decoded from the original safetensors and
    // re-quantized into a small `overlay.hfq` (written to `--reap-out`).
    // `--reap-arch` overrides the auto-detected arch family used for
    // tensor-name matching. See reap_overlay.rs / SP4 plan Task 4.
    let reap_overlay_dir = args.reap_overlay.clone();
    let reap_out = args.reap_out.clone();
    let reap_arch_flag = args.reap_arch.clone();
    // SP4b bake mode. `--reap-bake <plan-dir>` runs the NORMAL whole-model
    // quantize to completion BUT with a per-tensor override hook active: any
    // tensor the plan's `quant_overrides` name is re-quantized to its override
    // tier; every other tensor keeps its arch-specific default quant. The whole
    // model is written via the usual `write_hfq` to `--reap-out` (or the normal
    // `--format` output path). Mutually exclusive with `--reap-overlay`.
    let reap_bake_dir = args.reap_bake.clone();
    if reap_bake_dir.is_some() && reap_overlay_dir.is_some() {
        eprintln!("reap: --reap-bake and --reap-overlay are mutually exclusive");
        std::process::exit(1);
    }

    // Optional imatrix (llama.cpp GGUF format with .in_sum2 / .counts per-tensor).
    // When provided, MQ2-Lloyd quantization uses per-column importance weights
    // to bias centroid placement. See `quantize_mq2g256_lloyd_weighted`.
    let imatrix_path: Option<&Path> = args.imatrix.as_deref();
    let imatrix_gguf: Option<gguf_input::GgufFile> = imatrix_path.map(|p| {
        eprintln!("Loading imatrix: {}", p.display());
        gguf_input::GgufFile::open(p).unwrap_or_else(|e| {
            eprintln!("imatrix open failed: {e}");
            std::process::exit(2);
        })
    });
    if let Some(ref gg) = imatrix_gguf {
        let n_in_sum2 = gg
            .tensors
            .iter()
            .filter(|t| t.name.ends_with(".in_sum2"))
            .count();
        let n_counts = gg
            .tensors
            .iter()
            .filter(|t| t.name.ends_with(".counts"))
            .count();
        eprintln!(
            "  imatrix: {} in_sum2 + {} counts tensors",
            n_in_sum2, n_counts
        );
    }
    // q8f16 = all weights Q8 (interleaved blocks)
    // q4f16 = all weights Q4_F16_G64
    // q8-mixed = Q8 attn + Q4_K FFN (best tok/s for VRAM-constrained)
    // q8-fast = Q8 attn + Q4-as-Q8 FFN (all Q8 occupancy, most VRAM)
    // q8hfq = all weights Q8_HFQ (split-metadata, 128B-aligned rows)
    let use_q8 = format == "q8f16" || format == "q8";
    // F32 oracle: full-precision passthrough. Every tensor stored
    // as QuantType::F32 (qt=2) -- weights, norms, embeddings. The bf16 source
    // is widened bf16->f32 (lossless), giving the engine a superset-precision
    // reference forward for self-sufficient KLD eval.
    let use_f32_passthrough = format == "f32" || format == "f32-passthrough" || format == "oracle";
    let use_mixed = format == "q8-mixed" || format == "mixed";
    let use_fast = format == "q8-fast" || format == "fast";
    let use_q8hfq = format == "q8hfq";
    let use_q4k_all = format == "q4k";
    let use_q4k_q8embed = format == "q4k-q8embed";
    let use_mq8g256 = format == "mq8" || format == "mq8g256";
    // DeepSeek V4 recipe (2026-05-20): routed experts → MQ2-Lloyd, every other
    // 2D weight → Q8F16, with norms/biases/HC matrices falling through
    // to the F16 fallback path via `should_quantize() == false`.
    // No K-map, no imatrix promotions, no source-dtype distinctions in
    // the quant branch — uniform Q8F16 for everything that's a real
    // matmul weight. Designed to re-quant DeepSeek-V4-Flash including
    // the MTP head at maximum precision for the dense path.
    let use_deepseek4_source_precision = format == "deepseek4-q8-mtp"
        || format == "deepseek4-q8"
        || format == "deepseek4-source-precision"
        || format == "deepseek4-source"
        || format == "deepseek4-mtp-precise"
        || format == "deepseek4-mq4lloyd"
        || format == "deepseek4-mq3lloyd";
    let use_deepseek4_mq2rxt_overlay = format == "deepseek4-mq2rxt-overlay";
    // deepseek4-mq4lloyd / deepseek4-mq3lloyd: identical recipe to deepseek4-q8
    // (non-expert 2D → Q8F16, norms/HC → F16) EXCEPT routed experts ship as
    // MQ4G256Lloyd (qt=30, 160 B/group) resp. MQ3G256Lloyd (qt=20, 112 B/group)
    // instead of MQ2G256Lloyd. Both require the matching MoE GEMV kernels in the
    // ds4 forward (MQ3-Lloyd kernels pre-existed; MQ4-Lloyd added alongside).
    let use_deepseek4_mq4_experts = format == "deepseek4-mq4lloyd";
    let use_deepseek4_mq3_experts = format == "deepseek4-mq3lloyd";
    // deepseek4-mtp-precise: addon-only build (use with --include-prefix mtp.) that
    // keeps every mtp.0.* DENSE weight at F16 instead of Q8F16. Doubles the
    // addon size (~2 GB → ~3 GB) but eliminates Q8 quant noise on the MTP
    // attn projections, e_proj, h_proj, and shared experts. MTP is small
    // enough that the precision matters disproportionately — V3 paper's
    // 60-80% acceptance benchmark assumes weights at training precision,
    // not 8-bit. Routed experts stay MQ2-Lloyd (no precision-upgrade option
    // available without a new MoE GEMV kernel).
    let use_mtp_precise = format == "deepseek4-mtp-precise";
    let use_mq4g256 = format == "mq4v1" || format == "mq4g256" || format == "magnum";
    let use_mq4v2 = format == "mq4v2" || format == "mq4" || format == "mq4g256v2";
    let use_mq4c = format == "mq4c" || format == "mq4cg256" || format == "mq4g256c";
    let use_hfq4g256 = format == "hfq4g256" || format == "hfq4" || format == "hf4";
    let use_hfq3g256 = format == "hfq3g256";
    let use_hfq3g128 = format == "hfq3g128" || format == "hfq3" || format == "hf3"; // default HF3 = G128
    let use_hfq2g256 = format == "hfq2g256";
    let use_hfq2g128 = format == "hfq2g128" || format == "hfq2" || format == "hf2";
    let use_hfq_mixed = format == "hfq-mixed"; // Q8 attn + HFQ4 FFN
    let use_mq6g256 = format == "mq6" || format == "mq6g256";
    let use_mq5g256 = format == "mq5" || format == "mq5g256";
    let use_mq6g256v2 = format == "mq6v2" || format == "mq6g256v2";
    let use_mq5g256v2 = format == "mq5v2" || format == "mq5g256v2";
    let use_mq3g256v2 = format == "mq3v2" || format == "mq3g256v2";
    let use_mq2g256v2 = format == "mq2v2" || format == "mq2g256v2";
    // Native-bf16 reference. Cohere2MoE and Qwen3.5 store matmul weights as
    // the exact downloaded BF16 bytes; `f16` is a lossy-reconvert alternative
    // tier, while the all-F32 `oracle` doubles storage.
    let use_bf16 = format == "bf16" || format == "bf16-passthrough" || format == "oracle";
    let use_f16 = format == "f16" || format == "f16-passthrough";
    // ── Graded per-expert mixed precision (HIPFIRE_MOE_GRADED) ────────────
    // When set, each routed 3D-MoE expert in a layer is assigned its OWN
    // dtype: the top `HIPFIRE_MOE_HOT_FRAC` (default 0.2) experts BY IMATRIX
    // ROUTING COUNT → MQ6 (hot), the rest → MQ2-Lloyd (cold). A single
    // parent (one layer's gate_up_proj or down_proj) therefore emits MIXED
    // per-expert dtypes; the runtime builds a per-expert dtype-tag table
    // from each expert's gpu_dtype and dispatches the merged MQ6/MQ2-Lloyd
    // decode kernel. Requires --imatrix (the .counts tensor). Mutually
    // exclusive with the AWQ / Lloyd-tier expert paths (graded is the first
    // arm in the rayon dispatch). Compose with --format mq4 --no-kmap so the
    // DENSE attn/shared weights stay MQ4 and only the 3D experts are graded.
    let use_moe_graded = hipfire_config::developer_var("HIPFIRE_MOE_GRADED")
        .ok()
        .as_deref()
        == Some("1");
    let moe_hot_frac: f64 = hipfire_config::developer_var("HIPFIRE_MOE_HOT_FRAC")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(|f| f.clamp(0.0, 1.0))
        .unwrap_or(0.2);
    if use_moe_graded && imatrix_path.is_none() {
        eprintln!(
            "error: HIPFIRE_MOE_GRADED=1 requires --imatrix <PATH> (uses per-expert .counts)"
        );
        std::process::exit(2);
    }
    if use_moe_graded {
        eprintln!(
            "note: HIPFIRE_MOE_GRADED=1 — top {:.0}% routed experts per layer (by\n\
             imatrix .counts) -> MQ6, rest -> MQ2-Lloyd. Emits MIXED per-expert\n\
             dtypes; requires the merged MQ6/MQ2-Lloyd decode kernel at runtime.",
            moe_hot_frac * 100.0
        );
    }
    // ── N-tier graded MoE (HIPFIRE_MOE_TIER_MAP) ─────────────────────────────
    // When set to a path, reads a file of "LAYER EXPERT DTYPE" lines and
    // builds a per-(layer,expert) tier assignment for BOTH gate_up and down
    // projections. Supported DTYPE values: MQ6, MQ4, MQ3L, MQ2L.
    // Placed BEFORE the graded_hot arm in the rayon dispatch (takes priority).
    // Does NOT require --imatrix; does NOT restrict to down_proj.
    // Compose with --format mq4 --no-kmap --awq (dense attn/shared keep AWQ-MQ4).
    let moe_tier_map: Option<std::collections::HashMap<(usize, usize), QuantType>> = if let Ok(
        path,
    ) =
        hipfire_config::developer_var("HIPFIRE_MOE_TIER_MAP")
    {
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("error: HIPFIRE_MOE_TIER_MAP={path}: {e}");
            std::process::exit(2);
        });
        let mut map = std::collections::HashMap::new();
        for (lineno, line) in content.lines().enumerate() {
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 3 {
                continue;
            }
            let lay: usize = cols[0].parse().unwrap_or_else(|_| {
                eprintln!("error: {path}:{}: bad layer '{}'", lineno + 1, cols[0]);
                std::process::exit(2);
            });
            let exp: usize = cols[1].parse().unwrap_or_else(|_| {
                eprintln!("error: {path}:{}: bad expert '{}'", lineno + 1, cols[1]);
                std::process::exit(2);
            });
            let qt = match cols[2] {
                "MQ6" => QuantType::MQ6G256,
                "MQ4" => QuantType::MQ4G256,
                "MQ3L" => QuantType::MQ3G256Lloyd,
                "MQ2L" => QuantType::MQ2G256Lloyd,
                // GL = global codebook (one per tensor, shipped as kernel scalar
                // args) + per-block fp16 scale, SoA. 2.0625 / 3.0625 bpw vs the
                // per-block Lloyd family's 2.25 / 3.5 -- 0.1875 bpw cheaper for a
                // measured +1.16% KLD and -0.08% decode. DECODE-ONLY: no grouped
                // or batched GL kernels exist, so a GL model prefills per-token.
                "MQ2GL" => QuantType::MQ2G256GL,
                "MQ3GL" => QuantType::MQ3G256GL,
                "E8" | "MFP4E8" | "MFP4G32E8" => QuantType::MFP4G32E8,
                "MFP3E8" | "MFP3G32E8" => QuantType::MFP3G32E8,
                "MFP2E8" | "MFP2G32E8" => QuantType::MFP2G32E8,
                other => {
                    eprintln!(
                        "error: {path}:{}: unknown dtype '{}' (expected MQ6/MQ4/MQ3L/MQ2L/MQ2GL/MQ3GL/E8/MFP3E8/MFP2E8)",
                        lineno + 1,
                        other
                    );
                    std::process::exit(2);
                }
            };
            map.insert((lay, exp), qt);
        }
        eprintln!(
            "note: HIPFIRE_MOE_TIER_MAP={path} — {} (layer,expert) tier assignments loaded.",
            map.len()
        );
        Some(map)
    } else {
        None
    };
    // Mixed: MQ4 for attention/shared-expert + MQ6 for routed experts only.
    // Saves ~15 GB vs full MQ6 on 122B-A10B (75 GB vs 90 GB), fits in 125 GB UMA.
    let use_mq4_mq6exp = format == "mq4-mq6exp" || format == "mq4-mq6experts";
    // Round-trip quality probe: route routed-MoE experts through MQ2-Lloyd
    // quantize → dequantize → re-quantize as HFQ4. The .hfq ships as plain
    // MQ4 (HFQ4G256), no runtime changes. Measures whether 2-bit noise on
    // routed experts survives the MoE sparse-usage rescue, before sinking
    // a week into new MoE-2bit GEMV kernels.
    let use_mq4_mq2lloydexp = format == "mq4-mq2lloydexp"
        || format == "mq4-mq2lloydexperts"
        || format == "mq4-mq2lloyd-exp";
    // GL twin of the probe above: identical pipeline, but routed experts go
    // through the GLOBAL-codebook codec (one tensor-wide Lloyd–Max Gaussian
    // codebook + per-block fp16 scale) instead of a per-block fitted codebook.
    // Ships as HFQ4G256 exactly like `mq4-mq2lloydexp`, so both probes land in
    // the same container and a KLD delta between them isolates the codec —
    // no engine, loader, or kernel changes on either arm.
    let use_mq4_mq2glexp =
        format == "mq4-mq2glexp" || format == "mq4-mq2glexperts" || format == "mq4-mq2gl-exp";
    if use_mq4_mq2glexp {
        eprintln!(
            "note: --format mq4-mq2glexp is a quality probe — routed MoE experts\n\
             go through the GLOBAL-codebook 2-bit codec (one tensor-wide\n\
             Lloyd–Max Gaussian codebook + per-block fp16 scale) round-trip\n\
             (quantize → dequantize) and ship as HFQ4G256. Identical container\n\
             to --format mq4-mq2lloydexp, so a KLD delta between the two\n\
             isolates the codec. No engine/loader/kernel changes on either arm."
        );
    }
    if use_mq4_mq2lloydexp {
        eprintln!(
            "note: --format mq4-mq2lloydexp is a quality probe — routed MoE\n\
             experts go through MQ2-Lloyd round-trip (quantize → dequantize)\n\
             before being re-quantized as MQ4. Output is shipped as plain\n\
             MQ4 (no runtime changes needed). Measures whether MoE sparse\n\
             usage rescues MQ2-Lloyd at the experts before investing in new\n\
             MoE-2bit GEMV kernels."
        );
    }
    // Native Phase-2 form: routed MoE experts ship as native MQ2G256Lloyd
    // (qt=19). Requires runtime support — the qwen35 MoE forward path must
    // dispatch the new gemv_mq2g256_lloyd_moe_*_indexed* kernels (or fall
    // through to weight_gemv's MQ2G256Lloyd arm for the slow per-expert
    // path).
    let use_mq4_mq2lloyd_native = format == "mq4-mq2lloyd-native"
        || format == "mq4-mq2lloydexp-native"
        || format == "mq4-mq2lloyd-routed";
    // kmap-respecting variant: like mq4-mq2lloyd-native, but routed-expert
    // tensors that the kmap flags as Promote6 stay at MQ6 (instead of being
    // demoted to MQ2-Lloyd). Reduces precision-loss on the ~30% of layers
    // that the alternating K-map identifies as important. Larger file
    // (extra MQ6 layers) but expected to recover quality on attractor-prone
    // prompts that mq4-mq2lloyd-native truncated early.
    let use_mq4_mq2lloyd_kmap = format == "mq4-mq2lloyd-kmap"
        || format == "mq4-mq2lloyd-respectkmap"
        || format == "mq4-mq2lloyd-kmap-promote";
    // Imatrix-weighted variant: like mq4-mq2lloyd-kmap, but the Lloyd
    // codebook for each non-promoted expert is fit with per-column
    // importance weights from a llama.cpp imatrix file (--imatrix flag).
    // The kmap-promoted ~30 % of expert layers still stay at MQ6.
    let use_mq4_mq2lloyd_imatrix = format == "mq4-mq2lloyd-imatrix"
        || format == "mq4-mq2lloyd-kmap-imatrix"
        || format == "mq4-mq2lloyd-imatrix-kmap";
    // MQ3-Lloyd-on-routed-experts: 3 bpw alternative when 2 bpw isn't enough.
    // Kmap-respecting: promoted experts → MQ6, rest → MQ3-Lloyd (qt=20).
    // No imatrix variant for MQ3 in this commit — MQ3-Lloyd is empirically
    // production-grade on Qwen3.5-MoE A3B, so uniform Lloyd is the baseline.
    let use_mq4_mq3lloyd_kmap = format == "mq4-mq3lloyd-kmap"
        || format == "mq4-mq3lloyd-routed"
        || format == "mq4-mq3lloyd-exp";
    let allow_mq3_lloyd_for_mixed = args.allow_mq3_lloyd
        || hipfire_config::developer_var("HIPFIRE_ALLOW_MQ3_LLOYD")
            .ok()
            .as_deref()
            == Some("1");
    if use_mq4_mq3lloyd_kmap && !allow_mq3_lloyd_for_mixed {
        eprintln!(
            "note: --format mq4-mq3lloyd-kmap requires --allow-mq3-lloyd or\n\
             HIPFIRE_ALLOW_MQ3_LLOYD=1 (same gate as bare --format mq3-lloyd)."
        );
        std::process::exit(2);
    }
    if use_mq4_mq3lloyd_kmap {
        eprintln!(
            "note: --format mq4-mq3lloyd-kmap ships routed experts as MQ3G256Lloyd\n\
             (qt=20, 112 B / 256 weights, 3.5 bpw). Promoted experts stay at MQ6.\n\
             3 bpw fallback when 2 bpw can't avoid attractors on code-gen."
        );
    }
    // Phase 5: importance-aware MQ2/MQ3 layer tiering. Requires --imatrix.
    // Per-layer aggregate counts rank layers by routing activity; the top
    // `tier_ratio` fraction of NON-PROMOTED layers gets MQ3-Lloyd (3.5 bpw)
    // for higher precision on hot layers, the bottom fraction gets
    // MQ2-Lloyd (2.25 bpw) for size. K-map-promoted layers stay at MQ6.
    //
    // Granularity is PER LAYER (not per expert within a layer) because the
    // MoE-indexed kernels require uniform dtype across experts within a
    // tensor — the kernel reads expert_ptrs and assumes a fixed byte
    // stride per group (72 B for MQ2 vs 112 B for MQ3).
    let use_mq4_mqlloyd_tiered = format == "mq4-mqlloyd-tiered"
        || format == "mq4-mqlloyd-tiered-imatrix"
        || format == "mqlloyd-tiered";
    // Phase 6: antirez-style asymmetric-tensor recipe. Routed-expert
    // gate_up_proj → MQ2-Lloyd (imatrix-weighted), routed-expert
    // down_proj → MQ3-Lloyd (no imatrix, fixed-precision protection of
    // the residual-write direction). K-map promoted layers still get
    // MQ6 on both tensors.
    //
    // Rationale: antirez (V4 Flash) uses IQ2_XXS on up/gate and Q2_K
    // on down. The empirical claim is that `down` is the more sensitive
    // direction because it writes back into the residual stream — gate/up
    // errors get partially absorbed by silu. Mirror that asymmetry in
    // MQ-family: 2-bit on gate_up, 3-bit on down.
    let use_mq4_mqlloyd_antirez =
        format == "mq4-mqlloyd-antirez" || format == "mq4-mqlloyd-asym" || format == "antirez-mq";
    // HIPFIRE_ROUTED_GL=1: keep the per-projection bit allocation but swap the
    // per-block fp16 codebook for a GLOBAL one (qt=38/39). Post-FWHT blocks are
    // Gaussian by CLT, so the optimal LEVEL SHAPE is identical in every block and
    // a per-block fit re-derives it ~4000x per tensor, differing only by scale —
    // which the fp16 per-block scale already carries. Costs 0.1875 bpw less
    // (2.0625/3.0625 vs 2.25/3.5) for a measured +1.16% KLD and -0.08% decode.
    //
    // DECODE-ONLY: GL ships five kernels, all single-token indexed MoE GEMVs
    // (gemv_mq{2,3}g256gl_moe_{gate_up,down}_indexed + the sym gate_up). There
    // is no grouped-WMMA GEMM and no batched indexed GEMV for the SoA
    // global-codebook layout, and the merged dtype-tag kernel has no GL branch,
    // so a GL model still takes the per-token prefill path. The per-block Lloyd
    // pair does NOT: MQ2G256Lloyd / MQ3G256Lloyd both have grouped-WMMA GEMMs on
    // gfx11 and gfx12 and are batched-prefill admissible. Choosing GL therefore
    // trades ~0.19 bpw against prefill throughput, not just KLD.
    let routed_gl = std::env::var("HIPFIRE_ROUTED_GL").ok().as_deref() == Some("1");
    if routed_gl {
        eprintln!(
            "note: HIPFIRE_ROUTED_GL=1 — routed experts ship the GLOBAL-codebook\n\
             variants (MQ2G256GL qt=38 / MQ3G256GL qt=39) instead of the per-block\n\
             Lloyd ones. Decode-only: batched prefill rejects GL (no grouped-WMMA\n\
             kernel exists for the SoA layout), so prefill runs the per-token\n\
             fallback. The per-block Lloyd pair DOES batch — prefer it when\n\
             prefill throughput matters."
        );
    }
    // Lever 2: same recipe as antirez but with sequential-GPTQ Lloyd
    // on the gate_up_proj path instead of plain imatrix-weighted Lloyd.
    // Aims to reduce attractor risk at 2 bpw — if successful, opens path
    // to ALL-MQ2 routed experts (no down=MQ3 compensation needed) and
    // a further size reduction.
    let use_mq4_mqlloyd_antirez_gptq = format == "mq4-mqlloyd-antirez-gptq"
        || format == "mq4-mqlloyd-asym-gptq"
        || format == "antirez-mq-gptq";
    if use_mq4_mqlloyd_antirez_gptq && imatrix_path.is_none() {
        eprintln!("error: --format mq4-mqlloyd-antirez-gptq requires --imatrix <PATH>");
        std::process::exit(2);
    }
    if use_mq4_mqlloyd_antirez_gptq && !allow_mq3_lloyd_for_mixed {
        eprintln!(
            "note: --format mq4-mqlloyd-antirez-gptq requires --allow-mq3-lloyd or\n\
             HIPFIRE_ALLOW_MQ3_LLOYD=1 (down_proj uses MQ3-Lloyd)."
        );
        std::process::exit(2);
    }
    if use_mq4_mqlloyd_antirez_gptq {
        eprintln!(
            "note: --format mq4-mqlloyd-antirez-gptq — same routed-expert split\n\
             as antirez (gate_up=MQ2-Lloyd, down=MQ3-Lloyd), but gate_up uses\n\
             SEQUENTIAL-error-feedback Lloyd (simplified GPTQ-LDLQ) for\n\
             reduced attractor risk at 2 bpw."
        );
    }
    // All-MQ2-GPTQ: route BOTH gate_up AND down through MQ2-Lloyd-GPTQ.
    // Tests whether sequential error feedback closes the attractor gap
    // enough to drop the down=MQ3 compensation antirez uses, saving
    // ~30 % more on routed-expert size.
    let use_mq4_mq2lloyd_gptq_all = format == "mq4-mq2lloyd-gptq-all"
        || format == "mq4-mq2lloyd-gptq"
        || format == "all-mq2-gptq";
    if use_mq4_mq2lloyd_gptq_all
        && imatrix_path.is_none()
        && hipfire_config::developer_var("HIPFIRE_ALLOW_UNIT_IMATRIX")
            .ok()
            .as_deref()
            != Some("1")
    {
        eprintln!("error: --format mq4-mq2lloyd-gptq-all requires --imatrix <PATH>");
        eprintln!(
            "       (DeepSeek V4: set HIPFIRE_ALLOW_UNIT_IMATRIX=1 to use unit column weights —"
        );
        eprintln!(
            "        captures GPTQ sequential error-feedback win without imatrix calibration.)"
        );
        std::process::exit(2);
    }
    if use_mq4_mq2lloyd_gptq_all {
        eprintln!(
            "note: --format mq4-mq2lloyd-gptq-all — ALL routed experts (both\n\
             gate_up AND down) at MQ2-Lloyd with sequential-GPTQ codebook\n\
             assignment. Tests the size-reduction hypothesis from Lever 2."
        );
    }
    if use_mq4_mqlloyd_antirez {
        if imatrix_path.is_none() {
            eprintln!("error: --format mq4-mqlloyd-antirez requires --imatrix <PATH>");
            std::process::exit(2);
        }
        if !allow_mq3_lloyd_for_mixed {
            eprintln!(
                "note: --format mq4-mqlloyd-antirez requires --allow-mq3-lloyd or\n\
                 HIPFIRE_ALLOW_MQ3_LLOYD=1 (down_proj uses MQ3-Lloyd)."
            );
            std::process::exit(2);
        }
        eprintln!(
            "note: --format mq4-mqlloyd-antirez ships routed experts as\n\
             gate_up_proj → MQ2-Lloyd (imatrix-weighted, qt=19), down_proj\n\
             → MQ3-Lloyd (qt=20). K-map-promoted layers stay at MQ6 on both.\n\
             Mirrors antirez/ds4 V4 Flash recipe (IQ2_XXS gate/up, Q2_K down).\n\
             Estimated DeepSeek V4 size: 70% × MQ2 + 20% × MQ3 + 10% × MQ4 ≈ 96 GB."
        );
    }
    let tier_ratio = args.tier_ratio;
    if use_mq4_mqlloyd_tiered {
        if imatrix_path.is_none() {
            eprintln!("error: --format mq4-mqlloyd-tiered requires --imatrix <PATH>");
            std::process::exit(2);
        }
        if !allow_mq3_lloyd_for_mixed {
            eprintln!(
                "note: --format mq4-mqlloyd-tiered requires --allow-mq3-lloyd or\n\
                 HIPFIRE_ALLOW_MQ3_LLOYD=1 (uses MQ3-Lloyd on the hot layers)."
            );
            std::process::exit(2);
        }
        eprintln!(
            "note: --format mq4-mqlloyd-tiered uses imatrix .counts to rank\n\
             routed-expert layers by aggregate activation. Top {:.0}% of\n\
             non-promoted layers go to MQ3-Lloyd (3.5 bpw); the rest go to\n\
             MQ2-Lloyd (2.25 bpw). K-map-promoted layers stay at MQ6.",
            tier_ratio * 100.0
        );
    }
    if use_mq4_mq2lloyd_imatrix {
        if imatrix_path.is_none() {
            eprintln!("error: --format mq4-mq2lloyd-imatrix requires --imatrix <PATH>");
            std::process::exit(2);
        }
        eprintln!(
            "note: --format mq4-mq2lloyd-imatrix uses per-column importance\n\
             weights from the supplied calibration imatrix. Promoted experts\n\
             still stay at MQ6 (kmap-respect). Falls back to uniform Lloyd\n\
             for any expert whose imatrix tensor is missing."
        );
    }
    if use_mq4_mq2lloyd_kmap {
        eprintln!(
            "note: --format mq4-mq2lloyd-kmap respects K-map promotion —\n\
             experts flagged Promote6 (~30 % of layers) stay at MQ6G256;\n\
             remaining ~70 % get MQ2G256Lloyd (qt=19). File size is larger\n\
             than mq4-mq2lloyd-native but quality on attractor-prone prompts\n\
             should be markedly better."
        );
    }
    if use_mq4_mq2lloyd_native {
        eprintln!(
            "note: --format mq4-mq2lloyd-native ships routed MoE experts as\n\
             native MQ2G256Lloyd (qt=19, 72 B/group). Runtime must support\n\
             the MQ2-Lloyd MoE dispatch (weight_gemv arm exists; indexed\n\
             fast path requires forward-path arms in hipfire-arch-qwen35)."
        );
    }
    if use_mq4_mq6exp {
        eprintln!(
            "warning: --format mq4-mq6exp is deprecated. Use --format mq4 instead — \
             K-map promotes expert FFNs (and edge layers) to MQ6 automatically. \
             Proceeding as --format mq4."
        );
    }
    let use_mq3g256 = format == "mq3" || format == "mq3g256";
    let use_mq2g256 = format == "mq2" || format == "mq2g256";
    let use_mq2g256_lloyd =
        format == "mq2-lloyd" || format == "mq2g256-lloyd" || format == "mq2lloyd";
    let use_mq3g256_lloyd =
        format == "mq3-lloyd" || format == "mq3g256-lloyd" || format == "mq3lloyd";
    let use_mq4g256_lloyd =
        format == "mq4-lloyd" || format == "mq4g256-lloyd" || format == "mq4lloyd";
    let use_hfq6 = format == "hfq6" || format == "hfq6g256" || format == "hf6";
    // HFP4G32 — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale). Spec at docs/quant-formats/hfp4.md.
    let use_hfp4 = format == "hfp4" || format == "hfp4g32" || format == "hf4p" || format == "fp4";
    // MFP4G32 — HFP4G32 + offline FWHT (drop-in MQ4 replacement). Same per-row layout
    // as HFP4G32 with format_flags bit 0 + bits 2-3 = 01 stamping the rotation kind.
    let use_mfp4 = format == "mfp4" || format == "mfp4g32" || format == "mf4p";
    let use_mfp4l = format == "mfp4l"
        || format == "mfp4-lloyd"
        || format == "mfp4g32-lloyd"
        || format == "mfp4lloyd";
    // mfp4+P — mfp4 with E4M3 (non-power-of-2) per-block scale. Byte layout
    // identical to mfp4 (no prefix); only the per-block scale byte's meaning differs.
    let use_mfp4p = format == "mfp4p" || format == "mfp4+p" || format == "mfp4-p";
    // mfp4-E8 — mfp4+P container with E8-lattice vector quantization.
    // The `-gptq` suffix activates Hessian-aware sequential rounding (LDLQ on
    // the E8 lattice) — output bytes are IDENTICAL format (same E4M3 scale + 4
    // E8 codewords); GPTQ only changes the lattice-point assignment.
    let use_gptq_e8 = format == "mfp4e8-gptq" || format == "mfp4-e8-gptq";
    let use_mfp4e8 = format == "mfp4e8" || format == "mfp4-e8" || format == "mfp4l8" || use_gptq_e8;
    let use_mfp4e8soa = format == "mfp4e8soa" || format == "mfp4-e8-soa" || format == "mfp4e8-soa";
    // mfp3-E8 and mfp2-E8: 3-bit and 2-bit narrowed E8 lattice variants.
    // The `-gptq` suffix activates LDLQ — output bytes are IDENTICAL format to
    // the corresponding RTN paths; GPTQ only changes the lattice-point assignment.
    let use_gptq_mfp3e8 = format == "mfp3e8-gptq" || format == "mfp3-e8-gptq";
    let use_mfp3e8_gptq_fmt = format == "mfp3e8" || format == "mfp3-e8" || use_gptq_mfp3e8;
    let use_gptq_mfp2e8 = format == "mfp2e8-gptq" || format == "mfp2-e8-gptq";
    let use_mfp2e8_gptq_fmt = format == "mfp2e8" || format == "mfp2-e8" || use_gptq_mfp2e8;
    // GPTQ-E8 Hessian directory: per-(tensor,expert) 256-block XX^T captured by
    // the collect_e8_hessian binary. Missing/degenerate Hessians silently fall
    // back to RTN per-block (never worse than baseline). REQUIRED when --format
    // mfp{2,3,4}e8-gptq is set.
    let hessian_dir = args.hessian_dir.clone();
    if use_gptq_e8 && hessian_dir.is_none() {
        eprintln!(
            "warning: --format mfp4e8-gptq without --hessian-dir; every tensor falls back to RTN E8 (== plain mfp4e8). Pass --hessian-dir <dir> to enable GPTQ."
        );
    }
    if use_gptq_mfp3e8 && hessian_dir.is_none() {
        eprintln!(
            "warning: --format mfp3e8-gptq without --hessian-dir; every tensor falls back to RTN mfp3-E8. Pass --hessian-dir <dir> to enable GPTQ."
        );
    }
    if use_gptq_mfp2e8 && hessian_dir.is_none() {
        eprintln!(
            "warning: --format mfp2e8-gptq without --hessian-dir; every tensor falls back to RTN mfp2-E8. Pass --hessian-dir <dir> to enable GPTQ."
        );
    }
    if let Some(hd) = &hessian_dir {
        if !hd.exists() {
            eprintln!("error: --hessian-dir not found: {}", hd.display());
            std::process::exit(1);
        }
    }
    let q8_router_flag = args.q8_router;
    // Conv1d (DeltaNet) defaults to Q8 regardless of --format — the tensor is
    // small (~32K elem) but runs every token and lossy 4-bit FWHT formats
    // measurably hurt the gated-delta path. Override with --no-q8-conv1d to
    // keep conv1d at the same quant as the rest of the model.
    let q8_conv1d_default = !args.no_q8_conv1d;
    let no_kmap = args.no_kmap || args.uniform;

    // ── imatrix loader (consumed by AWQ pre-scaling) ──
    // --imatrix <path>: load an llama-imatrix-produced GGUF (per `examples/
    // imatrix_collect.rs`). Populates the IMATRIX OnceLock with per-channel
    // `Σ_token act²` values keyed by ggml-style tensor name. Quantizer behavior
    // with no `--imatrix` is byte-equivalent to baseline.
    //
    // For Qwen3.5 hybrid layers, the mapper covers: ffn_{gate,up,down},
    // self_attn.{q,k,v,o}_proj (full-attention layers), and
    // linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}
    // (linear-attention layers via SSM-naming). Norms / biases / 1D scalars /
    // conv1d / lookup tables have no imatrix entry.
    let imatrix_path = args.imatrix.clone();
    if let Some(path) = &imatrix_path {
        if !path.exists() {
            eprintln!("error: --imatrix path not found: {}", path.display());
            std::process::exit(1);
        }
        let table = load_imatrix(path);
        IMATRIX
            .set(table)
            .expect("IMATRIX set twice — should not happen");
        eprintln!("imatrix loaded from {}", path.display());
    }

    // ── Phase A Stage A: AWQ (Activation-aware Weight Quantization) ──
    // --awq           → enable AWQ at default alpha=0.55
    // --awq-alpha <f> → enable AWQ at explicit alpha (overrides default)
    // Requires --imatrix (we derive RMS_act from imatrix's in_sum2 values).
    // Per-channel scaling: W' = W · diag(s) at quantize time, sidecar
    // 1D F16 tensor <weight>.awq_scale stored alongside the parent weight.
    // Runtime path divides activations by s before the rotation kernel —
    // separate change, not in this patch. Implementation reference:
    // docs/plans/awq_hipfire.md.
    //
    // Stage A targets MQ4G256 specifically (large g=256 → AWQ's outlier-
    // mitigation works; per Egiazarian et al 2509.23202 §3.2, small-group
    // formats (g=16/32 NVFP4/MXFP4) "provably neutralize traditional
    // outlier mitigation techniques" — MR-GPTQ is the right lever there,
    // tracked as Stage C). HFP4/MFP4 are explicitly NOT awq-pre-scaled
    // in this patch.
    let awq_enabled = args.awq || args.awq_alpha.is_some();
    let awq_alpha = args.awq_alpha.unwrap_or(0.55);
    if awq_enabled {
        if IMATRIX.get().is_none() {
            eprintln!(
                "error: --awq requires --imatrix (we derive RMS_act per channel from imatrix in_sum2 values)"
            );
            std::process::exit(1);
        }
        if !(0.0..=1.0).contains(&awq_alpha) {
            eprintln!(
                "warning: --awq-alpha {awq_alpha} outside typical [0, 1] range; using anyway"
            );
        }
        AWQ_ALPHA
            .set(awq_alpha)
            .expect("AWQ_ALPHA set twice — should not happen");
        eprintln!(
            "AWQ pre-scaling: ENABLED (alpha={awq_alpha}, formula: s[j]=(RMS_act[j])^alpha, geo-mean normalized to 1)"
        );
    }
    // K-map gate: applies to MoE models by default. Dense models opt in
    // via --kmap-dense (the K-map dense PPL effect is mixed: regression at
    // short context, win at long context — see benchmarks/results/
    // ppl_kmap_20260508.md). Maintainer directive 2026-05-08: "intends to
    // help ONLY (never on dense)" by default.
    let kmap_dense = args.kmap_dense;
    // K-map mode: 0=full (all candidates promoted), 1=alternating (edge + every 3rd),
    // 2=typed (ffn_down+attn_v everywhere). Default: alternating — same PPL as full
    // at 17% less model size on MoE (22.9 vs 27.7 GB, PPL 8K: 19.96 vs 20.07).
    let mut kmap_mode: u8 = match args.kmap_mode.as_str() {
        "full" | "0" => 0,
        "alternating" | "alt" | "1" => 1,
        "typed" | "2" => 2,
        "typed-gemma4" | "3" => 3,
        _ => {
            eprintln!(
                "warning: unknown --kmap-mode '{}', using alternating",
                args.kmap_mode
            );
            1
        }
    };

    // ── Sub-4-bit guards (2026-04-30 sweep) ─────────────────────────────
    // MQ2 with the current uniform 4-level codebook collapses at every
    // model size validated locally (0.8B / 4B / 9B Qwen 3.5 → multilingual
    // mojibake on all 4 coherence-gate prompts). Refuse by default until
    // Path D Lloyd-Max non-uniform codebooks land (PRD §5.2).
    let allow_mq2 = args.allow_mq2
        || hipfire_config::developer_var("HIPFIRE_ALLOW_MQ2")
            .ok()
            .as_deref()
            == Some("1");
    if use_mq2g256 && !allow_mq2 {
        eprintln!(
            "error: --format mq2 is reserved — empirical quality verdict is collapse on every model\n\
             size validated locally (0.8B / 4B / 9B Qwen 3.5 → mojibake / symbol soup on all 4\n\
             coherence-gate prompts). The current uniform 4-level codebook is fundamentally too\n\
             lossy; Path D Lloyd-Max non-uniform codebooks (per-block squared-error-minimising)\n\
             are the planned remediation per PRD §5.2.\n\
             \n\
             To opt in for research / ablation purposes anyway, pass --allow-mq2 or set\n\
             HIPFIRE_ALLOW_MQ2=1. Don't ship MQ2 artifacts to users until the codebook\n\
             improvement lands."
        );
        std::process::exit(1);
    }
    // MQ2-Lloyd: rescues uniform MQ2 by 41–55× (per benchmarks/results/
    // lloyd_max_findings_20260501.md) but still text-collapse — 9B ppl=2,163
    // vs 9B MQ4 ppl=10. Research-only: same opt-in gate so users don't
    // accidentally ship a 2-bpw model that won't produce coherent output.
    let allow_mq3_lloyd = args.allow_mq3_lloyd
        || hipfire_config::developer_var("HIPFIRE_ALLOW_MQ3_LLOYD")
            .ok()
            .as_deref()
            == Some("1");
    if use_mq3g256_lloyd && !allow_mq3_lloyd {
        eprintln!(
            "note: --format mq3-lloyd is research — Lloyd-Max 8-entry codebook +\n\
             3-bit indices (112 B/group, +7.7% over uniform MQ3). Hypothesis is\n\
             non-uniform codebook lifts sub-9B MQ3 out of collapse (#114) and\n\
             tightens 9B MQ3's 4× ppl gap vs MQ4. Ppl evidence pending — DO NOT\n\
             ship MQ3-Lloyd artifacts to users until quality is validated against\n\
             baseline MQ3/MQ4 ppl.\n\
             \n\
             To proceed, pass --allow-mq3-lloyd or set HIPFIRE_ALLOW_MQ3_LLOYD=1."
        );
        std::process::exit(1);
    }
    let allow_mq2_lloyd = args.allow_mq2_lloyd
        || hipfire_config::developer_var("HIPFIRE_ALLOW_MQ2_LLOYD")
            .ok()
            .as_deref()
            == Some("1");
    if (use_mq2g256_lloyd
        || use_mq4_mq2lloydexp
        || use_mq4_mq2glexp
        || use_mq4_mq2lloyd_native
        || use_mq4_mq2lloyd_kmap
        || use_mq4_mq2lloyd_imatrix
        || use_mq4_mq3lloyd_kmap
        || use_mq4_mq2lloyd_kmap
        || use_mq4_mqlloyd_tiered
        || use_mq4_mqlloyd_antirez
        || use_mq4_mqlloyd_antirez_gptq
        || use_mq4_mq2lloyd_gptq_all
        || use_deepseek4_source_precision)
        && !allow_mq2_lloyd
    {
        eprintln!(
            "error: --format mq2-lloyd is research-only — Lloyd-Max codebook lifts\n\
             uniform MQ2 by 41–55× ppl but absolute quality is still collapse\n\
             (9B Qwen 3.5 wikitext2-test ppl=2,163 vs MQ4=10, MQ3=42; 0.8B ppl=19,651).\n\
             2 bpw is fundamentally too aggressive for usable text; the format\n\
             is plumbed for follow-on Lloyd-Max MQ3 (qt=20) experiments only.\n\
             \n\
             To opt in for research anyway, pass --allow-mq2-lloyd or set\n\
             HIPFIRE_ALLOW_MQ2_LLOYD=1. Don't ship MQ2-Lloyd artifacts to users."
        );
        std::process::exit(1);
    }
    // MQ4-Lloyd: extension of MQ3-Lloyd to K=16 centroids. Conjectured to
    // narrow the MQ4 → MQ6 ppl gap at +17.6% bandwidth over uniform MQ4
    // (160 vs 136 B/group). Per
    // benchmarks/results/devlog_20260506_lloyd_mq4_extension.md the
    // 9B projection is ppl 8.0–9.3 (vs uniform MQ4 ppl 10.34, MQ6 ppl 9.36).
    // Quality not yet validated — same opt-in gate as MQ3-Lloyd until ppl
    // numbers land.
    let allow_mq4_lloyd = args.allow_mq4_lloyd
        || hipfire_config::developer_var("HIPFIRE_ALLOW_MQ4_LLOYD")
            .ok()
            .as_deref()
            == Some("1");
    if use_mq4g256_lloyd && !allow_mq4_lloyd {
        eprintln!(
            "note: --format mq4-lloyd is research — Lloyd-Max 16-entry codebook +\n\
             4-bit indices (160 B/group, +17.6% over uniform MQ4). Hypothesis is\n\
             non-uniform codebook narrows the MQ4 → MQ6 ppl gap at lower bandwidth\n\
             than uniform MQ6. Ppl evidence pending — DO NOT ship MQ4-Lloyd\n\
             artifacts to users until quality is validated against baseline\n\
             MQ4/MQ6 ppl on the target model.\n\
             \n\
             To proceed, pass --allow-mq4-lloyd or set HIPFIRE_ALLOW_MQ4_LLOYD=1."
        );
        std::process::exit(1);
    }
    // MQ3 quality threshold ≈ 9B from the same sweep — 27B + 9B fluent,
    // 4B partial-collapse (intent recognised, language drifts), 0.8B
    // gibberish. Print a soft advisory so users running --format mq3
    // against small models don't think the engine is broken.
    if use_mq3g256 {
        eprintln!(
            "note: MQ3 empirical quality threshold ≈ 9B params. 27B / 9B Qwen 3.5 produce\n\
             fluent output across the coherence-gate battery; 4B partially collapses\n\
             (intent recognised, language mixes / loops); 0.8B is incoherent. For models\n\
             below ~9B, prefer --format mq4 (same kernel family, ~30% larger but\n\
             reliably coherent).\n"
        );
    }

    // GGUF input branch: if --input is a `.gguf` file, run the GGUF
    // pipeline and exit. Tensor names are translated GGUF → safetensors
    // style. The 2D quantization target follows --format:
    //   hfq4 (default for GGUF) | hfq6 | mq4 | mq6
    // Per CLAUDE.md guidance: dense (non-DeltaNet) models should use
    // hfq4/hfq6. mq4/mq6 are calibrated for Qwen3.5+ — using them on a
    // Llama-style model produces correct output (the FWHT cancels in
    // `gemv_mq4g256_with_rotate`) but adds runtime rotation overhead
    // with no quality benefit.
    {
        let raw_input = Path::new(input_dir);
        if is_gguf_input(raw_input) {
            let gguf_format = GgufFormat::from_flag(format).unwrap_or_else(|| {
                eprintln!(
                    "GGUF input: --format '{format}' not recognized. \
                     Supported: hfq4 (default), hfq6, mq4, mq6. \
                     Falling back to hfq4."
                );
                GgufFormat::Hfq4
            });
            let out = Path::new(output_path);
            if let Err(e) = run_gguf_pipeline(
                raw_input,
                out,
                gguf_format,
                no_kmap,
                kmap_dense,
                kmap_mode,
                args.arch_id,
                args.force_arch_id,
            ) {
                eprintln!("GGUF pipeline failed: {e}");
                std::process::exit(2);
            }
            return;
        }
    }

    // Resolve input: local path or HuggingFace model ID (e.g. "Qwen/Qwen3-8B")
    let input_dir = resolve_model_path(input_dir);
    let input_dir = Path::new(&input_dir);
    let output_path = Path::new(output_path);

    // Read model config
    let config_path = input_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|_| panic!("Cannot read {}. If using a HuggingFace model ID, ensure it's downloaded: huggingface-cli download {}", config_path.display(), input_dir.display()));
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

    let arch_str = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");
    // Single source of truth for model_type -> arch_id lives in hipfire-runtime::arch_mapping.
    // Fail-closed on unknown model_type: error naming the type and listing supported ones,
    // unless an explicit --arch-id override is supplied (operator knows best).
    let auto_arch_id: u32 = match hipfire_runtime::arch_mapping::lookup_model_type(arch_str) {
        Some(id) => id,
        None => {
            if let Some(ov) = args.arch_id {
                eprintln!(
                    "warning: unknown model_type '{}' but --arch-id {} override supplied; proceeding with override",
                    arch_str, ov
                );
                ov
            } else {
                let supported = hipfire_runtime::arch_mapping::supported_model_types_display();
                eprintln!(
                    "error: unknown model_type '{}'; supported model_types are: [{}]. Hint: pass --arch-id <id> to override for this model",
                    arch_str, supported
                );
                std::process::exit(1);
            }
        }
    };
    // --arch-id <u32> overrides the auto-detected id. Use when the
    // model's family maps to a different crate than the default
    // (e.g. plain Qwen2 → arch_id=7 for the hipfire-arch-qwen2 crate
    // instead of the LLaMA-family default 1, which silently drops
    // Q/K/V bias on the LLaMA loader path). See docs/plans/
    // dots-ocr-devlog.md §7 (R1) for the bring-up context.
    let arch_id = args.arch_id.unwrap_or(auto_arch_id);
    guard_qwen3_arch_override(auto_arch_id, arch_id, args.force_arch_id);
    if arch_id != auto_arch_id {
        eprintln!(
            "Architecture: {arch_str} (auto id={auto_arch_id}, overridden via --arch-id to {arch_id})"
        );
    } else {
        eprintln!("Architecture: {arch_str} (id={arch_id})");
    }
    let is_moe = arch_id == 6;
    // DeepSeek V4 (arch_id=9 post-2026-05-26 upstream merge that promoted
    // Qwen2-dense to 7 and dots.ocr to 8) is also MoE but ships per-expert
    // separate 2D tensors (`layers.L.ffn.experts.E.{w1,w2,w3}.weight`)
    // instead of Qwen3.5's stacked 3D `mlp.experts.gate_up_proj`. Phase 1
    // ingest handles DeepSeek V4's per-expert tensors individually through
    // the standard 2D quant path; the routing fan-out into top-k experts
    // happens at forward time, not quant time.
    let is_deepseek4 = arch_id == 9;
    // LFM2.5 (arch_id 11): A1B routes per-expert w1/w2/w3 → MQ4G256, expert_bias
    // → F32, everything else → Q8; dense lfm2 (Lfm2ForCausalLM, e.g. 350M/1.2B)
    // has no experts so the ingest just Q8s every tensor (the loader's load_f32
    // dequantizes norms / conv-filter back to F32).
    let is_lfm2moe = arch_id == 11;
    // Cohere2-MoE (arch_id 12): per-expert pre-split tensors; experts carry the
    // bit-width knob (--format f16|q8|mq6|mq4), attention/dense/router stay Q8
    // (F16 in the oracle), tied embed stays Q8, norms -> F16.
    let is_cohere2moe = arch_id == 12;
    // MiniMax-M2 (arch_id=10): MoE like DeepSeek V4, ships per-expert pre-split
    // 2D tensors (`...block_sparse_moe.experts.E.{w1,w2,w3}.weight`). Quantized
    // as HFQ4G256 (the only 4-bit format with a complete indexed-MoE GEMV
    // kernel family). Raw HF tensor names are written verbatim (no rename);
    // the hipfire loader looks them up.
    let is_minimax = arch_id == 10;
    let is_gemma4 = arch_id == 13;
    // Covers both the dense arch-13 (12B/26B unified) and the EAGLE drafter
    // (arch-22). Both have the same AWQ-unsuitability: √d_model embedding scale
    // (not RMSNorm-anchored) corrupts AWQ saliency for FFN; embed/lm_head are
    // tied + scaled by √3840 making AWQ scale saliency meaningless there.
    let is_gemma4_family = arch_id == 13 || arch_id == 22;
    let is_moe_like =
        is_moe || is_deepseek4 || is_lfm2moe || is_minimax || is_cohere2moe || is_gemma4;
    if (use_mq6g256v2 || use_mq5g256v2 || use_mq3g256v2 || use_mq2g256v2) && is_moe_like {
        eprintln!(
            "error: --format mq{{2,3,5,6}}v2 is dense-only (gfx1201 Qwen3.8); MoE model (arch_id={arch_id}) is not supported with this format. Use legacy mq{{2,3,5,6}} or mq4/mq4v2/mq4c for MoE, or run on a dense checkpoint."
        );
        std::process::exit(2);
    }
    // Gemma4 (arch_id 13) defaults to kmap_mode=3 (typed-gemma4): promote down_proj,
    // v_proj, and edge-layer non-attn-qko tensors. Attn q/k/o are excluded even
    // in edge layers (dense attn promotion regresses PPL +3.1% on 27B).
    // The explicit --kmap-mode flag overrides this default.
    if is_gemma4 && args.kmap_mode == "alternating" {
        kmap_mode = 3;
    }
    // Q8 "router" — a misnomer: `is_q8_tensor` covers the whole FIXED tier
    // (attention q/k/v/o, linear_attn projections, conv1d, lm_head, embed, and
    // the MoE router), not just `mlp.gate.weight`. On for MoE-class models by
    // default, since the fixed tier is quality-critical and cheap relative to
    // the routed experts *by parameter count*.
    //
    // `--no-q8-router` restores the historic opt-out. It matters far more than
    // the name suggests: the fixed tier is **66% of per-token decode bytes** on
    // a3b (mq4r: 1030.8 MB fixed vs 534.8 MB routed), so forcing it to Q8
    // (1.0625 B/w) instead of MQ4 (0.53125 B/w) doubles the dominant term. That
    // is why `.mq2` reads 45% MORE bytes/token than `.mq4r` despite being 7 GB
    // smaller on disk, and why `.mq4r` — which needs this flag off — is not
    // byte-reproducible from HEAD without it.
    let no_q8_router_flag =
        args.no_q8_router || std::env::var("HIPFIRE_NO_Q8_ROUTER").ok().as_deref() == Some("1");
    let q8_router = (is_moe_like || q8_router_flag) && !no_q8_router_flag;
    // Muse Glimmer (arch 14): untied lm_head defaults to Q8, like embed.
    //
    // Glimmer sets `tie_word_embeddings=false`, so `lm_head.weight` is a
    // SEPARATE [202048, 6656] tensor rather than an alias of the embedding
    // table. `embed_tokens` stays Q8 through its own `is_embed` arm, but an
    // untied lm_head has no such arm — on a dense model `q8_router` is off by
    // default, so the K-map's Q8 verdict for lm_head is never reached and it
    // follows `--format` down to MQ4. That is a 4-bit output projection over a
    // 202k vocab, and nothing in the pipeline flags it. The first Glimmer MQ4
    // build shipped exactly that way while the K-map unit tests passed.
    //
    // Rather than widen the shared `is_moe_like` default (which would drag
    // attention into Q8 for every dense arch and change their artifacts), this
    // enables the fixed tier for arch 14 only and narrows it to the two classes
    // that must be Q8. `--no-q8-router` still wins, and an explicit
    // HIPFIRE_Q8_CLASSES still wins, so both levers stay available.
    //
    // Cost on the 30B: 15.51 GB -> 16.26 GB, decode 33.06 -> 31.62 tok/s
    // (gfx1201, 64 tok greedy). Both artifacts decode coherently.
    let glimmer_q8_head = arch_id == 14 && !no_q8_router_flag;
    if glimmer_q8_head {
        if std::env::var("HIPFIRE_Q8_CLASSES").is_err() {
            // SAFETY: single-threaded CLI setup, before any worker threads spawn.
            unsafe { std::env::set_var("HIPFIRE_Q8_CLASSES", "lm_head,embed") };
        }
        eprintln!(
            "note: muse_glimmer (arch 14) — untied lm_head + embed held at Q8F16;\n\
             all other tensors follow --format. Override with HIPFIRE_Q8_CLASSES\n\
             or disable with --no-q8-router."
        );
    }
    let q8_router = q8_router || glimmer_q8_head;
    if no_q8_router_flag {
        eprintln!(
            "note: --no-q8-router — the fixed tier (attention / lm_head / router /\n\
             embed / conv1d) follows --format instead of being forced to Q8F16.\n\
             This is the mq4r recipe and the lever for a sub-MQ4 fixed tier;\n\
             embed_tokens still stays Q8 via its own arm."
        );
    }
    if is_moe {
        eprintln!("  MoE detected — will split 3D expert tensors per-expert before quantization.");
    }
    if is_deepseek4 {
        eprintln!(
            "  DeepSeek V4 detected — per-expert tensors ship pre-split; quantizing each as 2D weight."
        );
    }
    if is_lfm2moe {
        eprintln!(
            "  LFM2.5 detected — experts → MQ4G256, expert_bias → F32, all else (conv/attn/dense/router/embed/norms) → Q8."
        );
    }
    if is_minimax {
        eprintln!(
            "  MiniMax-M2 detected — per-expert tensors ship pre-split; quantizing each as HFQ4G256 2D weight."
        );
    }
    if is_cohere2moe {
        eprintln!(
            "  Cohere2-MoE detected — experts → --format ({{f16|q8|mq6|mq4}}); attn/dense → Q8 (F16 in oracle); router/embed → Q8; norms → F16."
        );
    }

    // Extract layer count for K-map edge-layer promotion.
    // Qwen3.5+ nests config under "text_config"; try both paths.
    let n_layers: usize = config
        .get("num_hidden_layers")
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|tc| tc.get("num_hidden_layers"))
        })
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    if n_layers == 0 {
        eprintln!(
            "  warning: num_hidden_layers not found in config.json — edge-layer promotion disabled"
        );
    }

    // Read tokenizer if present
    let tokenizer_json = input_dir.join("tokenizer.json");
    let tokenizer_str = if tokenizer_json.exists() {
        std::fs::read_to_string(&tokenizer_json).ok()
    } else {
        None
    };

    // Read tokenizer_config.json (has chat_template)
    let tokenizer_config_path = input_dir.join("tokenizer_config.json");
    let tokenizer_config: Option<serde_json::Value> = if tokenizer_config_path.exists() {
        std::fs::read_to_string(&tokenizer_config_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    } else {
        None
    };

    // Some checkpoints (e.g. LFM2.5) ship the Jinja chat template in a separate
    // `chat_template.jinja` file rather than inside tokenizer_config.json. The
    // daemon extracts its template from `tokenizer_config.chat_template` (then
    // renders via minijinja); fold the sidecar in when tokenizer_config lacks
    // one, else the daemon falls back to Plain framing and a chat-tuned model
    // produces garbage (LFM2.5-350M bring-up, 2026-06-07).
    let tokenizer_config = {
        let mut tc = tokenizer_config;
        let jinja_path = input_dir.join("chat_template.jinja");
        if jinja_path.exists() {
            let has_template = tc
                .as_ref()
                .and_then(|v| v.get("chat_template"))
                .map(|v| !v.is_null())
                .unwrap_or(false);
            if !has_template {
                if let Ok(jinja) = std::fs::read_to_string(&jinja_path) {
                    let n = jinja.len();
                    let obj = tc.get_or_insert_with(|| serde_json::json!({}));
                    if let Some(map) = obj.as_object_mut() {
                        map.insert(
                            "chat_template".to_string(),
                            serde_json::Value::String(jinja),
                        );
                        eprintln!(
                            "  embedded chat_template.jinja into tokenizer_config ({n} bytes)"
                        );
                    }
                }
            }
        }
        tc
    };

    // Read generation_config.json. HF stores some sampler-side defaults
    // here (eos_token_id, pad_token_id, bos_token_id, do_sample, etc.)
    // separately from config.json. For most checkpoints these duplicate
    // config.json fields, but dots.ocr's config.json carries no
    // eos_token_id at all — the [151643, 151673] array lives only in
    // generation_config.json. Packing it here lets the arch-side parser
    // (e.g. `hipfire-arch-qwen2::Qwen2Config::from_hfq`) fall back to
    // generation_config when config.eos_token_id is absent. Resolves
    // R5 in docs/plans/dots-ocr-devlog.md §7.
    let generation_config_path = input_dir.join("generation_config.json");
    let generation_config: Option<serde_json::Value> = if generation_config_path.exists() {
        std::fs::read_to_string(&generation_config_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    } else {
        None
    };

    // Build metadata JSON for .hfq
    let metadata = serde_json::json!({
        "architecture": arch_str,
        "config": config,
        "tokenizer": tokenizer_str.as_deref().unwrap_or("{}"),
        "tokenizer_config": tokenizer_config,
        "generation_config": generation_config,
    });
    // `mut` so the SP4b bake-prune path can patch the routed-expert count down to
    // the kept count before write_hfq (so the baked model loads with the compact
    // count and NO env var). Untouched in every non-prune path.
    let mut metadata_json = serde_json::to_string(&metadata).unwrap();

    // Load all safetensors files
    let st_files: Vec<SafetensorsFile> = find_safetensors(input_dir)
        .iter()
        .map(|p| {
            eprintln!("Loading: {}", p.display());
            SafetensorsFile::open(p).unwrap()
        })
        .collect();

    // Collect all tensor names.
    //
    // DeepSeek V4 note: tensors come in `<name>.weight` (I8 = E4M3) + `<name>.scale`
    // (F8_E8M0) pairs. We index the `.scale` siblings into a side map
    // keyed by the weight tensor's full name and skip them in the main
    // iteration. When we encounter the `.weight` half we look up the
    // sibling and call `dequantize_e4m3_ue8m0_to_f32` to recover f32
    // before the existing MQ-family pipeline runs.
    let mut all_tensors: Vec<(&str, usize)> = Vec::new();
    let mut fp8_scale_for: HashMap<String, (usize, String)> = HashMap::new();
    for (fi, st) in st_files.iter().enumerate() {
        for name in st.tensor_names() {
            // MiniMax-M2 FP8: `<w>.weight` (e4m3) + `<w>.weight_scale_inv` (F32
            // block-[128,128] scale). Strip the longer suffix FIRST.
            if let Some(stem) = name.strip_suffix(".weight_scale_inv") {
                let w_name = format!("{stem}.weight");
                fp8_scale_for.insert(w_name, (fi, name.to_string()));
                continue;
            }
            if let Some(stem) = name.strip_suffix(".scale") {
                // FP8 scale siblings: `foo.scale` is the per-tensor scale for
                // `foo.weight`. Skip from quantization; attach at quant time.
                // Exception: Gemma4's `router.scale` is a real model weight
                // (multiplicative scale on router input), NOT an FP8 scale.
                if name.contains("router.scale") {
                    all_tensors.push((name, fi));
                } else {
                    let w_name = format!("{stem}.weight");
                    fp8_scale_for.insert(w_name, (fi, name.to_string()));
                }
                continue;
            }
            all_tensors.push((name, fi));
        }
    }
    all_tensors.sort_by_key(|(name, _)| name.to_string());
    eprintln!(
        "Found {} tensors ({} FP8 scale siblings indexed)",
        all_tensors.len(),
        fp8_scale_for.len()
    );

    // ── SP4: selective re-quant overlay mode ────────────────────────────────
    // When `--reap-overlay <plan-dir>` is set, this branch fully replaces the
    // normal whole-model quantize: it loads the reap plan, resolves the arch
    // family (auto from arch_id, or `--reap-arch` override), then iterates the
    // model tensors and — for ONLY the tensors the plan overrides — decodes
    // f32 and re-quantizes via `quantize_to_format`. Non-matched tensors skip
    // the (expensive) f32 decode entirely. The subset is written to
    // `--reap-out` via the existing `write_hfq`, keyed by original tensor name
    // so a load-time splice (SP3) can overlay them onto the base model.
    if let Some(plan_dir) = reap_overlay_dir.as_deref() {
        let reap_out_path = reap_out.as_deref().unwrap_or_else(|| {
            eprintln!("--reap-overlay requires --reap-out <overlay.hfq path>");
            std::process::exit(1);
        });
        // Resolve arch: explicit --reap-arch overrides the auto-detection.
        let arch: reap_overlay::ReapArch = match reap_arch_flag.as_deref() {
            Some(s) => reap_overlay::ReapArch::from_flag(s).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1);
            }),
            None => reap_overlay::ReapArch::from_arch_id(arch_id).unwrap_or_else(|| {
                eprintln!(
                    "reap overlay: could not auto-detect arch family from arch_id={arch_id}; \
                     pass --reap-arch <deepseek4|qwen35|lfm2moe|minimax>"
                );
                std::process::exit(1);
            }),
        };
        let plan = hipfire_reap::plan::ReapPlan::load_unchecked(plan_dir).unwrap_or_else(|e| {
            eprintln!("reap overlay: failed to load plan from {plan_dir}: {e}");
            std::process::exit(1);
        });
        eprintln!(
            "REAP overlay mode: arch={arch:?}, {} quant_overrides, out={reap_out_path}",
            plan.quant_overrides.len()
        );

        let mut hfq_tensors: Vec<HfqTensor> = Vec::new();
        for (name, file_idx) in &all_tensors {
            // Check the plan BEFORE decoding f32 — skipping the decode of
            // non-matched tensors is the whole point of an overlay build.
            if reap_overlay::reap_override_for(name, arch, &plan).is_none() {
                continue;
            }
            let (meta, raw_data) = st_files[*file_idx].tensor_data(name).unwrap();
            let f32 = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                &meta,
                &fp8_scale_for,
                &st_files,
            );
            let shape: Vec<usize> = meta.shape.clone();
            let tier = reap_overlay::reap_override_for(name, arch, &plan).unwrap();
            match reap_overlay::quantize_to_format(name, tier, &f32, &shape) {
                Ok(t) => {
                    eprintln!("  overlay: {name} → {tier} ({} bytes)", t.data.len());
                    hfq_tensors.push(t);
                }
                Err(e) => {
                    eprintln!("reap overlay: {e}");
                    std::process::exit(2);
                }
            }
        }

        if hfq_tensors.is_empty() {
            eprintln!(
                "reap overlay: no tensors matched the plan's quant_overrides \
                 (check arch/layer/expert names)"
            );
            std::process::exit(1);
        }

        eprintln!(
            "REAP overlay: {} tensors quantized; writing {reap_out_path}",
            hfq_tensors.len()
        );
        write_hfq(
            Path::new(reap_out_path),
            arch_id,
            &metadata_json,
            &hfq_tensors,
            None,
        )
        .unwrap_or_else(|e| {
            eprintln!("reap overlay: failed to write {reap_out_path}: {e}");
            std::process::exit(2);
        });
        eprintln!("REAP overlay written: {reap_out_path}");
        return;
    }

    // ── SP4b: bake-mode setup ────────────────────────────────────────────────
    // `--reap-bake <plan-dir>` keeps the normal whole-model quantize loop but
    // activates the per-tensor override hook (at the top of the loop below).
    // Resolve the plan + arch family up front; the loop reads `reap_bake_plan`
    // and `reap_arch`. When bake is inactive these are unused / None and the
    // loop is byte-identical to today. If `--reap-out` is given, the whole
    // baked model is written there instead of the normal `--output` path.
    let reap_bake_plan: Option<hipfire_reap::plan::ReapPlan> = match reap_bake_dir.as_deref() {
        Some(plan_dir) => Some(
            hipfire_reap::plan::ReapPlan::load_unchecked(plan_dir).unwrap_or_else(|e| {
                eprintln!("reap bake: failed to load plan from {plan_dir}: {e}");
                std::process::exit(1);
            }),
        ),
        None => None,
    };
    // Arch family for tensor-name matching: explicit --reap-arch overrides the
    // auto-detection from arch_id (only consulted when bake is active).
    let reap_arch: reap_overlay::ReapArch = if reap_bake_plan.is_some() {
        match reap_arch_flag.as_deref() {
            Some(s) => reap_overlay::ReapArch::from_flag(s).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1);
            }),
            None => reap_overlay::ReapArch::from_arch_id(arch_id).unwrap_or_else(|| {
                eprintln!(
                    "reap bake: could not auto-detect arch family from arch_id={arch_id}; \
                     pass --reap-arch <deepseek4|qwen35|lfm2moe|minimax>"
                );
                std::process::exit(1);
            }),
        }
    } else {
        // Placeholder (never read when reap_bake_plan is None).
        reap_overlay::ReapArch::Qwen35
    };
    // Redirect the whole-model output to --reap-out when baking with that flag.
    let bake_out_path = reap_bake_plan
        .as_ref()
        .and(reap_out.as_deref())
        .map(Path::new);
    let output_path: &Path = bake_out_path.unwrap_or(output_path);
    if let Some(plan) = &reap_bake_plan {
        eprintln!(
            "REAP bake mode: arch={reap_arch:?}, {} quant_overrides, out={}",
            plan.quant_overrides.len(),
            output_path.display()
        );
    }
    // Is expert pruning active? (A bake plan with a per-layer keep-map.) When
    // active, the loop's prune hook drops pruned per-expert tensors, the kept
    // per-expert tensors are recorded in `bake_rename` for a post-loop renumber
    // to compact slots, routers/biases are row-gathered to the kept set, and the
    // output metadata's expert count is patched to `kept_per_layer`.
    let bake_keep_active = reap_bake_plan
        .as_ref()
        .map(|p| p.keep.is_some())
        .unwrap_or(false);
    // original-name → compact-renamed-name for kept per-expert tensors
    // (ds4 score layers / lfm2 / minimax). Applied as a post-loop rename pass so
    // the per-expert quant branches keep using the ORIGINAL name to read source
    // bytes, then we rewrite `HfqTensor.name` to the compact slot before write.
    let mut bake_rename: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    // Task A0: original gate_proj name → fused `experts.{N}.gate_up_proj.weight`,
    // for ORNITH-class Qwen3.5-MoE checkpoints that ship experts un-stacked.
    // Applied as an UNCONDITIONAL post-loop rename pass (see below).
    let mut expert_fuse_rename: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    // ── K-map pre-pass ──────────────────────────────────────────────────────
    // Build per-tensor quant level map. Gated to MoE models by default
    // (maintainer directive 2026-05-08): K-map's dense PPL effect is mixed
    // (+1.5% to +2.5% at 2K, -4.8% at 8K — crossover at ~3K context). To
    // avoid silently changing dense quantization output, dense models opt
    // out by default and require `--kmap-dense` to enable. MoE models keep
    // the K-map default-on path because the routed-expert promotion is
    // the headline win and the empirical regression there is tighter
    // (+1.7% PPL at 2K, gated below the dense regression threshold).
    // K-map is enabled for: MoE models (default), gemma4 (arch_id 13,
    // default mode=2), or any dense model with --kmap-dense.
    // Suppress with --no-kmap / --uniform.
    let kmap: HashMap<String, QuantLevel> = if no_kmap || (!is_moe && !is_gemma4 && !kmap_dense) {
        HashMap::new()
    } else {
        let mut map = HashMap::new();
        let mut counts = [0u32; 4]; // F16, Q8, Promote6, Base
        for (name, _fi) in &all_tensors {
            let level = kmap_resolve_mode(name, n_layers, is_moe, kmap_mode);
            match level {
                QuantLevel::F16 => counts[0] += 1,
                QuantLevel::Q8 => counts[1] += 1,
                QuantLevel::Promote6 => counts[2] += 1,
                QuantLevel::Override(_) => counts[3] += 1,
                QuantLevel::Base => counts[3] += 1,
            }
            map.insert(name.to_string(), level);
        }
        if !map.is_empty() {
            let mode_label = match kmap_mode {
                0 => "full",
                1 => "alternating",
                2 => "typed",
                _ => "?",
            };
            eprintln!(
                "K-map plan ({format} base, {n_layers} layers{}, mode={mode_label}):",
                if is_moe { ", MoE" } else { "" }
            );
            eprintln!("  F16:       {:>4} tensors (norms, biases)", counts[0]);
            eprintln!(
                "  Q8:        {:>4} tensors (embed, lm_head, routers)",
                counts[1]
            );
            eprintln!("  Promote6:  {:>4} tensors", counts[2]);
            eprintln!("  Base:      {:>4} tensors (remaining)", counts[3]);
        }
        map
    };

    // Phase 5: per-layer tier set — which routed-expert layers go MQ3-Lloyd
    // vs MQ2-Lloyd. Only populated for `--format mq4-mqlloyd-tiered`.
    // Computed once from imatrix .counts; kmap-promoted layers are excluded
    // (they always go MQ6).
    let mq3_tier_layers: std::collections::HashSet<usize> = if use_mq4_mqlloyd_tiered {
        if let Some(ref gguf) = imatrix_gguf {
            if let Some(layer_counts) = imatrix_layer_activation_counts(gguf, n_layers) {
                // Indexes of layers NOT promoted by K-map. We need a name
                // representative of each layer's expert tensor to query
                // kmap; use the canonical safetensors name format.
                let candidates: Vec<usize> = (0..n_layers)
                    .filter(|&l| {
                        let probe_name =
                            format!("model.language_model.layers.{}.mlp.experts.gate_up_proj", l);
                        kmap.get(&probe_name) != Some(&QuantLevel::Promote6)
                    })
                    .collect();
                let mut ranked: Vec<(usize, f64)> = candidates
                    .iter()
                    .filter(|&&l| layer_counts[l].is_finite())
                    .map(|&l| (l, layer_counts[l]))
                    .collect();
                // Sort by count DESC (hot layers first).
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let n_mq3 = ((ranked.len() as f64) * tier_ratio).round() as usize;
                let n_mq3 = n_mq3.min(ranked.len());
                let set: std::collections::HashSet<usize> =
                    ranked.iter().take(n_mq3).map(|&(l, _)| l).collect();
                eprintln!(
                    "Tiered MQ-Lloyd: {} candidate non-promoted layers; \
                     {} (top {:.0}%) → MQ3-Lloyd, {} → MQ2-Lloyd",
                    ranked.len(),
                    set.len(),
                    tier_ratio * 100.0,
                    ranked.len().saturating_sub(set.len())
                );
                if set.len() <= 16 {
                    eprintln!(
                        "  MQ3-Lloyd layers (by count): {:?}",
                        ranked
                            .iter()
                            .take(n_mq3)
                            .map(|&(l, c)| (l, c as u64))
                            .collect::<Vec<_>>()
                    );
                }
                set
            } else {
                eprintln!("warning: imatrix has no ffn_gate_exps counts — tiering disabled");
                std::collections::HashSet::new()
            }
        } else {
            std::collections::HashSet::new()
        }
    } else {
        std::collections::HashSet::new()
    };

    // Quantize
    let mut hfq_tensors = Vec::new();
    let mut total_params = 0u64;
    let mut quantized_params = 0u64;
    // Spill file for large models — keeps peak RSS bounded by flushing
    // completed tensor data to disk when accumulated memory exceeds 32 GB.
    // HIPFIRE_SPILL_DIR overrides the spill location (default = output dir).
    // Point it at a RAM-backed tmpfs (e.g. /dev/shm) to keep peak DISK usage
    // = output size only, when disk is tight but RAM is ample.
    let spill_dir_override = std::env::var("HIPFIRE_SPILL_DIR").ok();
    let spill_dir = match spill_dir_override.as_deref() {
        Some(d) => Path::new(d),
        None => output_path.parent().unwrap_or(Path::new(".")),
    };
    // HIPFIRE_NO_SPILL=1 disables the disk spill entirely (hold all tensors in
    // RAM, write output directly). Needed for huge f32 oracles where spill+output
    // would be ~2x the output size on disk — but RAM is ample.
    let mut spill = if hipfire_config::developer_var("HIPFIRE_NO_SPILL")
        .ok()
        .as_deref()
        == Some("1")
    {
        None
    } else {
        TensorSpill::new(spill_dir).ok()
    };
    let mut total_quant_error = 0.0f64;
    let mut max_quant_error = 0.0f32;
    let mut _n_quant_groups = 0u64;

    let include_vision = args.include_vision;
    let vision_quant = args.vision_quant.as_str();
    // --include-prefix <prefix>: when set, ONLY tensors whose name starts
    // with this prefix are ingested; everything else is silently skipped.
    // Used to produce side-car HFQs (e.g. `--include-prefix mtp.` builds an
    // MTP-only addon that pairs with an existing base HFQ via the loader's
    // `.mtp-addon.hfq` discovery). When unset (default), all tensors pass
    // this gate and the usual mtp/vision skip rules below apply.
    let include_prefix = args.include_prefix.as_deref();
    if let Some(p) = include_prefix {
        eprintln!(
            "  [filter] --include-prefix {p:?} — only tensors with this prefix will be ingested"
        );
    }
    let mut skipped_params = 0u64;
    let mut mq2rxt_overlay_count = 0usize;
    // MiniMax AWQ: shared-per-layer expert scales, cached + sidecars emitted once.
    let mut mm_awq_cache: std::collections::HashMap<usize, Option<(Vec<f32>, Vec<f32>)>> =
        std::collections::HashMap::new();
    let mut mm_awq_emitted: std::collections::HashSet<usize> = std::collections::HashSet::new();
    // Task A0: name → shard index, so the pre-split expert fusion can fetch a
    // gate_proj's up_proj sibling (which may live in a different shard).
    let name_to_file: std::collections::HashMap<&str, usize> =
        all_tensors.iter().map(|(n, fi)| (*n, *fi)).collect();
    // Gemma4 (arch 13): unified multimodal checkpoints prefix the text decoder
    // with `model.language_model.`; text-only checkpoints (model_type
    // "gemma4_text") use flat `model.*` names. Only arm the tower-skip when the
    // multimodal prefix actually exists, else a text-only checkpoint would be
    // skipped wholesale.
    let gemma4_skip_non_lm = arch_id == 13
        && all_tensors
            .iter()
            .any(|(n, _)| n.starts_with("model.language_model."));

    for (name, file_idx) in &all_tensors {
        // --include-prefix filter (highest priority — runs before mtp/vision skips).
        if let Some(p) = include_prefix {
            if !name.starts_with(p) {
                let (meta, _) = st_files[*file_idx].tensor_data(name).unwrap();
                let n: usize = meta.shape.iter().product();
                skipped_params += n as u64;
                continue;
            }
        }
        // Skip MTP head; optionally include vision encoder for VL inference.
        // Qwen3.5-VL names vision tensors `model.visual.*` / `visual.*`;
        // dots.ocr names them `vision_tower.*`; Glimmer names them
        // `model.vision_tower.*`, `model.vision_adapter.*`,
        // `model.vision_projection.*`. All fall through to the F16 fallback
        // path (see should_quantize) when --include-vision is set.
        let is_vision = name.starts_with("model.visual.")
            || name.starts_with("visual.")
            || name.starts_with("vision_tower.")
            || name.starts_with("model.vision_tower.")
            || name.starts_with("model.vision_adapter.")
            || name.starts_with("model.vision_projection.");
        if is_vision && !include_vision {
            let (meta, _) = st_files[*file_idx].tensor_data(name).unwrap();
            let n: usize = meta.shape.iter().product();
            skipped_params += n as u64;
            continue;
        }
        // Gemma4 unified (arch 13): text-only bring-up — skip the vision/audio
        // towers + multimodal projectors; quantize only the text decoder.
        if gemma4_skip_non_lm && !name.starts_with("model.language_model.") {
            let (meta, _) = st_files[*file_idx].tensor_data(name).unwrap();
            let n: usize = meta.shape.iter().product();
            skipped_params += n as u64;
            continue;
        }
        // MTP (Multi-Token Prediction) head: pre-Phase-5 quants skipped these
        // because no forward path consumed them. deepseek4-q8-mtp is the first format
        // that ingests the MTP layer; v3 spec-decode requires it. For other
        // formats we still skip to avoid bloating the HFQ with unused tensors.
        if name.starts_with("mtp.")
            && !use_deepseek4_source_precision
            && !use_deepseek4_mq2rxt_overlay
        {
            let (meta, _) = st_files[*file_idx].tensor_data(name).unwrap();
            let n: usize = meta.shape.iter().product();
            skipped_params += n as u64;
            continue;
        }

        let (meta, raw_data) = st_files[*file_idx].tensor_data(name).unwrap();
        let mut n_elements: usize = meta.shape.iter().product();
        total_params += n_elements as u64;

        if use_deepseek4_mq2rxt_overlay {
            let sidecar = include_prefix.is_some_and(|prefix| prefix == "mtp.");
            let in_requested_artifact = if sidecar {
                name.starts_with("mtp.")
            } else {
                !name.starts_with("mtp.")
            };
            if !in_requested_artifact || !is_deepseek4_mq2rxt_dense(name) {
                skipped_params += n_elements as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                continue;
            }
            if meta.shape.len() != 2 || meta.shape[1] % 256 != 0 {
                eprintln!(
                    "MQ2RXT overlay: '{name}' must be rank-2 with K divisible by 256, got {:?}",
                    meta.shape
                );
                std::process::exit(2);
            }
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            let signs1 = gen_fwht_signs(42, 256);
            let signs2 = gen_fwht_signs(1042, 256);
            let data = quantize_mq4g256(&f32_data, &signs1, &signs2);
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::MQ4G256,
                shape: meta
                    .shape
                    .iter()
                    .map(|&dimension| dimension as u32)
                    .collect(),
                group_size: 256,
                data,
                spilled_len: 0,
            });
            mq2rxt_overlay_count += 1;
            quantized_params += n_elements as u64;
            st_files[*file_idx].drop_tensor_pages(name);
            if let Some(ref mut spill) = spill {
                maybe_spill(&mut hfq_tensors, spill, 2 * 1024 * 1024 * 1024);
            }
            continue;
        }

        // ── SP4b: bake prune hook ──────────────────────────────────────────────
        // BEFORE the override hook. When `--reap-bake`'s plan carries a keep-map,
        // prune routed experts not in `keep[L]`, renumber kept experts to compact
        // slots, and row-gather routers / per-expert biases to the kept set so the
        // baked model loads with the compact expert count and NO load-time
        // keep-map. `meta`/`raw_data` may be shadowed below with gathered owned
        // copies so the override hook + arch branches transparently quantize the
        // gathered tensor with the arch's normal encoder.
        //
        // Two owned holders are pre-declared so a gather can rebind the borrowed
        // (`meta`, `raw_data`) to point at gathered data for the rest of the body.
        let _gathered_meta: TensorMeta;
        let _gathered_bytes: Vec<u8>;
        let mut meta: &TensorMeta = meta;
        let mut raw_data: &[u8] = raw_data;
        if let Some(res) = handle_bake_keep_active(
            name,
            *file_idx,
            meta,
            raw_data,
            bake_keep_active,
            &reap_bake_plan,
            reap_arch,
            &mut bake_rename,
            &st_files,
        ) {
            match res {
                BakeKeepResult::Pruned => continue,
                BakeKeepResult::Gathered {
                    meta: gm,
                    bytes: gb,
                } => {
                    _gathered_meta = gm;
                    _gathered_bytes = gb;
                    meta = &_gathered_meta;
                    raw_data = &_gathered_bytes;
                }
            }
        }

        // ── SP4b: bake override hook ───────────────────────────────────────────
        // When `--reap-bake` is active and the plan overrides this tensor,
        // re-quantize it to the override tier and skip the arch-specific default
        // branch below. Non-overridden tensors fall through UNCHANGED. The hook
        // is entirely behind `if let Some(plan) = &reap_bake_plan`, so default
        // mode (no `--reap-bake`) is byte-identical to before. Bookkeeping
        // mirrors the arch branches: f32-decode → push → drop_tensor_pages →
        // quantized_params → maybe_spill → continue.
        if let Some(plan) = &reap_bake_plan {
            if let Some(fmt) = reap_overlay::reap_override_for(name, reap_arch, plan) {
                let f32 = tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw_data,
                    meta,
                    &fp8_scale_for,
                    &st_files,
                );
                let shape: Vec<usize> = meta.shape.clone();
                match reap_overlay::quantize_to_format(name, fmt, &f32, &shape) {
                    Ok(t) => {
                        eprintln!("  {:>8}: {} {:?} → {fmt}", "BAKE", name, meta.shape);
                        hfq_tensors.push(t);
                    }
                    Err(e) => {
                        eprintln!("reap bake: {e}");
                        std::process::exit(2);
                    }
                }
                quantized_params += n_elements as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut s) = spill {
                    maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }
        }

        // ── Task A0: Qwen3.5-MoE pre-split routed-expert fusion (arch_id 6) ──
        // Canonical Qwen3.5-MoE ships routed experts stacked-3D as
        // `mlp.experts.gate_up_proj` (the paths below split it per-expert).
        // ORNITH-class finetunes instead ship them UN-stacked as separate 2D
        // `mlp.experts.{N}.{gate,up,down}_proj.weight` (DeepSeek-V4 layout). The
        // qwen35 loader only knows the fused per-expert
        // `mlp.experts.{N}.gate_up_proj.weight` ([2*inter, hidden], gate||up), so
        // fuse gate+up here and rename the output post-loop; the normal quant
        // path below encodes the [2*inter, hidden] tensor (k-map still selects
        // the level by the gate_proj name). `down_proj` already matches the
        // loader name and takes the normal path unchanged; `shared_expert` (kept
        // un-fused by the loader) is excluded by the `.mlp.experts.` guard.
        let _fused_meta: TensorMeta;
        let _fused_bytes: Vec<u8>;
        if is_moe && meta.shape.len() == 2 && name.contains(".mlp.experts.") {
            if name.ends_with(".up_proj.weight") {
                // Consumed by its gate_proj sibling (fused below).
                st_files[*file_idx].drop_tensor_pages(name);
                continue;
            }
            if let Some(stem) = name.strip_suffix(".gate_proj.weight") {
                let up_name = format!("{stem}.up_proj.weight");
                let up_fi = match name_to_file.get(up_name.as_str()) {
                    Some(fi) => *fi,
                    None => {
                        eprintln!("qwen35 expert fusion: missing sibling {up_name} for {name}");
                        std::process::exit(2);
                    }
                };
                let (up_meta, up_raw) = st_files[up_fi].tensor_data(&up_name).unwrap();
                if up_meta.shape != meta.shape || up_meta.dtype != meta.dtype {
                    eprintln!(
                        "qwen35 expert fusion: gate {:?}/{} vs up {:?}/{} mismatch at {name}",
                        meta.shape, meta.dtype, up_meta.shape, up_meta.dtype
                    );
                    std::process::exit(2);
                }
                // gate rows first, then up rows → [2*inter, hidden]. Same source
                // dtype ⇒ a raw byte concat is lossless. Order is load-bearing:
                // loader stores gate_up = gate||up; forward is silu(gate)*up.
                let mut fused = Vec::with_capacity(raw_data.len() + up_raw.len());
                fused.extend_from_slice(raw_data);
                fused.extend_from_slice(up_raw);
                _fused_bytes = fused;
                _fused_meta = TensorMeta {
                    dtype: meta.dtype.clone(),
                    shape: vec![meta.shape[0] * 2, meta.shape[1]],
                    data_offsets: meta.data_offsets,
                };
                n_elements *= 2; // fused tensor carries gate + up params
                meta = &_fused_meta;
                raw_data = &_fused_bytes;
                expert_fuse_rename.insert(name.to_string(), format!("{stem}.gate_up_proj.weight"));
                st_files[up_fi].drop_tensor_pages(&up_name);
                eprintln!(
                    "  {:>8}: {name} + up_proj → {stem}.gate_up_proj.weight {:?}",
                    "FUSE", _fused_meta.shape
                );
            }
        }

        // ── F1 native-bf16 oracle passthrough ──────────────────────────────
        // Store EVERY tensor as F32 (qt=2): no quantization, bf16/f16->f32
        // widened losslessly. This bypasses every per-format branch below so
        // the produced .hfq is a full-precision reference the qwen35 loader
        // reads via its qt=2 arm and the engine forwards through the existing
        // F32 GEMV / attention_f32 path.
        {
            let __ctx = PerTensorCtx {
                name,
                file_idx: *file_idx,
                shape: &meta.shape,
                n_elements,
                arch_id,
                dtype: &meta.dtype,
                is_vision,
            };
            if handle_f32_passthrough(
                &__ctx,
                meta,
                raw_data,
                is_cohere2moe,
                is_moe,
                use_f32_passthrough,
                &fp8_scale_for,
                &st_files,
                &mut hfq_tensors,
                &mut quantized_params,
                &mut spill,
            ) {
                continue;
            }
        }

        // Source-precision BF16 passthrough for non-vision model tensors.
        // Unlike the F32 oracle above, this preserves the checkpoint's native
        // two-byte representation on disk. The qwen35 loader can consume
        // qt=16 losslessly; vision remains on the established F16 ingest path
        // because its kernels consume F16 matrices.
        {
            let __ctx = PerTensorCtx {
                name,
                file_idx: *file_idx,
                shape: &meta.shape,
                n_elements,
                arch_id,
                dtype: &meta.dtype,
                is_vision,
            };
            if handle_bf16_passthrough(
                &__ctx,
                meta,
                raw_data,
                use_bf16,
                arch_id,
                is_vision,
                is_moe,
                &fp8_scale_for,
                &st_files,
                &mut hfq_tensors,
                &mut quantized_params,
                &mut spill,
            ) {
                continue;
            }
        }

        // ── LFM2.5 ingest (arch_id 11) — extracted to try_handle_lfm2moe
        if try_handle_lfm2moe(
            is_lfm2moe,
            use_mq4g256,
            use_mq4v2,
            use_mq4c,
            name,
            meta,
            raw_data,
            &fp8_scale_for,
            &st_files,
            &mut hfq_tensors,
            &mut quantized_params,
            &mut spill,
            *file_idx,
            n_elements,
        ) {
            {
                continue;
            }
        }
        // ── Cohere2-MoE ingest (arch_id 12) ─────────────────────────────────
        // North-Mini-Code-1.0. Sweep tiers via --format: f16 (BF16-class oracle)
        // | q8 | mq6 | mq4. The EXPERTS carry the bit-width knob; attention/dense
        // stay Q8 (F16 in the oracle); the router (mlp.gate.weight) and the tied
        // embed_tokens stay Q8 (selection- / lookup-sensitive, held constant
        // across the sweep so KLD isolates expert/attention precision); all
        // *norm* tensors -> F16. Experts ship per-expert pre-split (gate_proj/
        // up_proj/down_proj); the loader byte-fuses gate_proj||up_proj.
        if handle_cohere2moe(
            name,
            meta,
            raw_data,
            n_elements,
            is_cohere2moe,
            use_bf16,
            use_f16,
            use_mq6g256,
            use_mq4g256,
            &fp8_scale_for,
            &st_files,
            &mut hfq_tensors,
            &mut quantized_params,
            &mut spill,
            *file_idx,
        ) {
            continue;
        }

        // DeepSeek V4's `tid2eid` hash-routing tables: source I64 in safetensors,
        // shape [vocab=129280, k=6]. The values are token-id × expert-id
        // pairs that all fit in i32 (vocab < 2^31, n_experts < 2^31), so
        // we downcast I64 → U32 (4 bytes/element) before write — antirez
        // does the same and the DeepSeek V4 loader at arch.rs reads them as U32
        // (`bytes.chunks_exact(4)`). Without these in the HFQ, the loader
        // sees an empty `tid2eid_host` and `ffn_hash_routed` falls back
        // to shared-only on the first `num_hash_layers` (3) layers —
        // measured 2× wikitext2 PPL regression on deepseek4-q8-mtp (21.85
        // vs 11.42 antirez) before this fix landed.
        //
        // QuantType=22 is "reserved-but-unused" in our enum (HFP4G16
        // ablation slot, never built); we use it for tid2eid storage to
        // stay byte-compatible with antirezQ8.hfq which also writes 22.
        // The loader is name-gated (looks for "tid2eid" substring), so
        // qt value doesn't actually steer dispatch — only matters for
        // cross-tooling identification.
        if meta.dtype == "I64" {
            if name.ends_with("tid2eid") {
                if n_elements * 8 != raw_data.len() {
                    panic!(
                        "tid2eid '{name}': expected {} bytes (8 × {}), got {}",
                        n_elements * 8,
                        n_elements,
                        raw_data.len()
                    );
                }
                let mut u32_bytes: Vec<u8> = Vec::with_capacity(n_elements * 4);
                for i in 0..n_elements {
                    let off = i * 8;
                    let v = i64::from_le_bytes(raw_data[off..off + 8].try_into().unwrap());
                    let v_u32 = v as u32; // downcast — values fit
                    u32_bytes.extend_from_slice(&v_u32.to_le_bytes());
                }
                let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
                eprintln!(
                    "  {:>8}: {} {:?} (I64 → U32, {} elements, {:.1} KB)",
                    "TID2EID",
                    name,
                    meta.shape,
                    n_elements,
                    u32_bytes.len() as f64 / 1024.0
                );
                quantized_params += n_elements as u64;
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::TidI32,
                    shape,
                    group_size: 0,
                    data: u32_bytes,
                    spilled_len: 0,
                });
                st_files[*file_idx].drop_tensor_pages(name);
                continue;
            }
            // Other I64 (none expected in DeepSeek V4): skip with explicit warning.
            eprintln!(
                "  [skip-I64] {} {:?} ({} elements) — unexpected I64 tensor, not ingested",
                name, meta.shape, n_elements
            );
            skipped_params += n_elements as u64;
            continue;
        }

        // ── MiniMax-M2 router: keep Q8 ─────────────────────────────────────────
        // The MoE router (`block_sparse_moe.gate.weight`) is precision-sensitive
        // (4-bit noise flips top-k on borderline tokens) but must NOT be F16:
        // weight_gemv's F16 arm dispatches gemm_f16_batched_lmhead, which is a
        // WMMA lm-head kernel that produces garbage for the router's tiny m
        // (=n_exp). Q8 (gemv_q8_0) is well-behaved at any m and ~0.4% noise.
        if is_minimax && name.ends_with("block_sparse_moe.gate.weight") {
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            let q = quantize_q8f16(&f32_data);
            eprintln!("  {:>8}: {} {:?} (router Q8)", "Q8-MM", name, meta.shape);
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::Q8F16,
                shape,
                group_size: 32,
                data: q,
                spilled_len: 0,
            });
            st_files[*file_idx].drop_tensor_pages(name);
            continue;
        }

        // ── MiniMax-M2 per-expert pre-split path ───────────────────────────────
        // Experts ship as 2D `...block_sparse_moe.experts.E.{w1,w2,w3}.weight`
        // (F32 in the tiny oracle; FP8 e4m3 + F32 weight_scale_inv in the 229B
        // ckpt — handled transparently by tensor_to_f32_with_optional_fp8_scale).
        // Quantize each as MQ4G256 (FWHT-pre-rotated 4-bit): byte-compatible with
        // the gemv_hfq4g256_moe_* indexed kernels — passing FWHT-rotated input to
        // those kernels is mathematically equivalent to gemv_mq4g256 (the exact
        // path qwen35's MoE uses). This IS the user-facing "mq4" format. Names
        // are written verbatim; the loader fuses w1||w3 into the gate_up blob.
        if is_minimax
            && name.contains(".block_sparse_moe.experts.")
            && name.ends_with(".weight")
            && meta.shape.len() == 2
        {
            let mut f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            let k = meta.shape[1];
            let m = meta.shape[0];
            if k % 256 == 0 {
                // AWQ shared-per-layer pre-scaling of the routed experts (--awq +
                // --imatrix). w1/w3 use s_gate_up (MoE-input channels), w2 uses
                // s_down (intermediate channels). Math W·s @ x/s = W·x is exact;
                // the forward divides the activation by experts[0]'s scale.
                if awq_enabled {
                    if let (Some(layer_n), Some(gg)) =
                        (minimax_layer_index(name), imatrix_gguf.as_ref())
                    {
                        let alpha = AWQ_ALPHA.get().copied().unwrap_or(0.55);
                        let entry = mm_awq_cache
                            .entry(layer_n)
                            .or_insert_with(|| minimax_layer_awq_scales(gg, layer_n, alpha));
                        if let Some((s_gu, s_dn)) = entry.as_ref() {
                            let scale = if name.ends_with(".w2.weight") {
                                s_dn
                            } else {
                                s_gu
                            };
                            if scale.len() == k {
                                awq_pre_scale_weights(&mut f32_data, m, k, scale);
                            } else {
                                eprintln!(
                                    "  minimax AWQ L{layer_n}: scale len {} != k {} ({name}); skipped",
                                    scale.len(),
                                    k
                                );
                            }
                            if mm_awq_emitted.insert(layer_n) {
                                let p = name.split(".block_sparse_moe.").next().unwrap();
                                hfq_tensors.push(HfqTensor {
                                    name: format!("{p}.block_sparse_moe.awq_scale_gate_up.weight"),
                                    quant_type: QuantType::F16,
                                    shape: vec![s_gu.len() as u32],
                                    group_size: 0,
                                    data: awq_scales_to_f16_bytes(s_gu),
                                    spilled_len: 0,
                                });
                                hfq_tensors.push(HfqTensor {
                                    name: format!("{p}.block_sparse_moe.awq_scale_down.weight"),
                                    quant_type: QuantType::F16,
                                    shape: vec![s_dn.len() as u32],
                                    group_size: 0,
                                    data: awq_scales_to_f16_bytes(s_dn),
                                    spilled_len: 0,
                                });
                                eprintln!("  AWQ-MM: emitted shared expert scales for L{layer_n}");
                            }
                        }
                    }
                }
                let signs1 = gen_fwht_signs(42, 256);
                let signs2 = gen_fwht_signs(1042, 256);
                // Expert format by --format: mq2-lloyd (MQ2G256Lloyd, hipx sub-4-bit
                // target — has deepseek4 indexed-MoE kernels), mq3-lloyd / mq6 (oracle
                // check / HIPFIRE_MINIMAX_EXPERT_*), else mq4 (MQ4G256, default + validated).
                let mm_mq6 = use_mq6g256
                    || hipfire_config::developer_var_os("HIPFIRE_MINIMAX_EXPERT_MQ6").is_some();
                let mm_mq2l = use_mq2g256_lloyd
                    || hipfire_config::developer_var_os("HIPFIRE_MINIMAX_EXPERT_MQ2L").is_some();
                let mm_mq3l = use_mq3g256_lloyd
                    || hipfire_config::developer_var_os("HIPFIRE_MINIMAX_EXPERT_MQ3L").is_some();
                // Per-layer mixed-precision promotion. HIPFIRE_MINIMAX_PROMOTE_MQ4 /
                // _MQ6 hold comma-separated layer ranges ("12-45,50") whose experts are
                // forced UP to MQ4 / MQ6 regardless of the base --format. The forward
                // dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model
                // carries an MQ2-Lloyd base with MQ4 on the quant-sensitive middle layers.
                let mm_layer = minimax_layer_index(name);
                let promote_mq6 = mm_layer.map_or(false, |l| {
                    minimax_layer_in_config_set("HIPFIRE_MINIMAX_PROMOTE_MQ6", l)
                });
                let promote_mq4 = mm_layer.map_or(false, |l| {
                    minimax_layer_in_config_set("HIPFIRE_MINIMAX_PROMOTE_MQ4", l)
                });
                let (q, qt, label) = if promote_mq6 {
                    (
                        quantize_mq6g256(&f32_data, &signs1, &signs2),
                        QuantType::MQ6G256,
                        "MQ6-PROMO",
                    )
                } else if promote_mq4 {
                    (
                        quantize_mq4g256(&f32_data, &signs1, &signs2),
                        QuantType::MQ4G256,
                        "MQ4-PROMO",
                    )
                } else if mm_mq3l {
                    (
                        quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2),
                        QuantType::MQ3G256Lloyd,
                        "MQ3L-MM",
                    )
                } else if mm_mq2l {
                    (
                        quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2),
                        QuantType::MQ2G256Lloyd,
                        "MQ2L-MM",
                    )
                } else if mm_mq6 {
                    (
                        quantize_mq6g256(&f32_data, &signs1, &signs2),
                        QuantType::MQ6G256,
                        "MQ6-MM",
                    )
                } else {
                    (
                        quantize_mq4g256(&f32_data, &signs1, &signs2),
                        QuantType::MQ4G256,
                        "MQ4-MM",
                    )
                };
                let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
                eprintln!(
                    "  {label:>8}: {} {:?} ({:.1} KB → {:.1} KB)",
                    name,
                    meta.shape,
                    raw_data.len() as f64 / 1024.0,
                    q.len() as f64 / 1024.0
                );
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: qt,
                    shape,
                    group_size: 256,
                    data: q,
                    spilled_len: 0,
                });
                quantized_params += (meta.shape[0] * meta.shape[1]) as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut s) = spill {
                    maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }
            // k not %256 → fall through to standard path (real MiniMax inter=1536,
            // hidden=3072 are both %256, so this only guards degenerate tinies).
        }

        // ── MoE 3D-stacked expert tensor split ─────────────────────────────────
        // Qwen3.5-MoE stores routed experts as 3D tensors:
        //   model.language_model.layers.{N}.mlp.experts.gate_up_proj
        //     shape: [num_experts, 2 * moe_intermediate, hidden_size]
        //   model.language_model.layers.{N}.mlp.experts.down_proj
        //     shape: [num_experts, hidden_size, moe_intermediate]
        // Note: no `.weight` suffix on these, so should_quantize() returns false
        // and the standard path would store them as F16 — defeating the purpose.
        // We split into per-expert 2D MQ4G256 quantized tensors named
        //   model.language_model.layers.{N}.mlp.experts.{X}.{base}.weight
        // so the engine loader can fish them out by expert index.
        // ── DeepSeek V4 per-expert tensor path ─────────────────────────────────────
        // DeepSeek V4 ships per-expert 2D tensors at `layers.L.ffn.experts.E.{w1,w2,w3}.weight`.
        // (Not 3D-stacked like Qwen3.5 MoE.) Route them through the MQ-family
        // quant path directly. No imatrix yet for DeepSeek V4 — pass unit column
        // weights so the underlying Lloyd codebook fit is uniform; the
        // GPTQ sequential error-feedback assignment still applies and is
        // worth +1-2 % coherence (project_gptq_lloyd_mq2_win.md).
        if is_deepseek4
            && name.contains(".ffn.experts.")
            && name.ends_with(".weight")
            && meta.shape.len() == 2
        {
            // DeepSeek V4 routed experts are FP4 (E2M1) per upstream `inference/
            // model.py:132-137` and config `expert_dtype:"fp4"`. Safetensors
            // shape is [out, in/2] with each byte packing two nibbles; the
            // paired scale tensor is `<name>.scale` UE8M0 with block size 32
            // along logical K.
            //
            // The outer condition `name.contains(".ffn.experts.")` already
            // excludes shared_experts (which use the non-routed `.shared_
            // experts.` infix). So everything reaching here is a routed
            // expert → unconditionally FP4 unpack. Logical K dim doubles.
            let name_owned = name.to_string();
            let (f32_data, logical_shape) = if (meta.dtype == "I8" || meta.dtype == "F8_E4M3")
                && fp8_scale_for.contains_key(&name_owned)
            {
                let (sfi, sname) = &fp8_scale_for[&name_owned];
                let (smeta, sbytes) = st_files[*sfi]
                    .tensor_data(sname)
                    .unwrap_or_else(|| panic!("FP scale tensor missing: {sname}"));
                dequantize_e2m1_ue8m0_to_f32(raw_data, &meta.shape, sbytes, &smeta.shape)
            } else {
                let vals = tensor_to_f32_with_optional_fp8_scale(
                    name,
                    raw_data,
                    meta,
                    &fp8_scale_for,
                    &st_files,
                );
                (vals, meta.shape.clone())
            };
            let k = logical_shape[1];
            if k % 256 == 0
                && (use_mq4_mq2lloyd_gptq_all
                    || use_mq4_mqlloyd_antirez_gptq
                    || use_mq4_mq2lloyd_native
                    || use_mq4_mq2lloyd_imatrix
                    || use_mq4_mqlloyd_antirez
                    || use_deepseek4_source_precision)
            {
                let signs1 = gen_fwht_signs(42, 256);
                let signs2 = gen_fwht_signs(1042, 256);
                let unit_col_weights: Vec<f32> = vec![1.0; k];
                let (q, expert_qt, expert_label): (Vec<u8>, QuantType, &str) =
                    if use_deepseek4_mq4_experts {
                        (
                            quantize_mq4g256_lloyd(&f32_data, &signs1, &signs2),
                            QuantType::MQ4G256Lloyd,
                            "MQ4L-DeepSeek V4",
                        )
                    } else if use_deepseek4_mq3_experts {
                        (
                            quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2),
                            QuantType::MQ3G256Lloyd,
                            "MQ3L-DeepSeek V4",
                        )
                    } else if use_mq4_mq2lloyd_gptq_all || use_mq4_mqlloyd_antirez_gptq {
                        (
                            quantize_mq2g256_lloyd_gptq(
                                &f32_data,
                                &unit_col_weights,
                                &signs1,
                                &signs2,
                            ),
                            QuantType::MQ2G256Lloyd,
                            "MQ2L-DeepSeek V4",
                        )
                    } else {
                        (
                            quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2),
                            QuantType::MQ2G256Lloyd,
                            "MQ2L-DeepSeek V4",
                        )
                    };
                let shape: Vec<u32> = logical_shape.iter().map(|&s| s as u32).collect();
                eprintln!(
                    "  {:>8}: {} storage{:?} → logical{:?} ({:.1} KB → {:.1} KB)",
                    expert_label,
                    name,
                    meta.shape,
                    logical_shape,
                    raw_data.len() as f64 / 1024.0,
                    q.len() as f64 / 1024.0
                );
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: expert_qt,
                    shape,
                    group_size: 256,
                    data: q,
                    spilled_len: 0,
                });
                quantized_params += (logical_shape[0] * logical_shape[1]) as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut s) = spill {
                    maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }
            // Fall through to standard path for non-MQ2 formats.
        }

        // Gemma 4 26B-A4B uses the SAME layout but at a different prefix
        // (no `mlp.` — tensors live directly under `.experts.`):
        //   model.language_model.layers.{N}.experts.gate_up_proj
        //   model.language_model.layers.{N}.experts.down_proj
        // Name-suffix match + shape check handles both qwen3.5 (mlp.experts.*)
        // and gemma4 (experts.*) without prefix-specific conditions.
        let is_moe_expert_3d = (is_moe || is_gemma4)
            && (name.ends_with("experts.gate_up_proj") || name.ends_with("experts.down_proj"))
            && meta.shape.len() == 3;
        if is_moe_expert_3d {
            let __ctx = PerTensorCtx {
                name,
                file_idx: *file_idx,
                shape: &meta.shape,
                n_elements,
                arch_id,
                dtype: &meta.dtype,
                is_vision,
            };
            if handle_moe_expert_3d(
                &__ctx,
                meta,
                raw_data,
                is_moe,
                is_gemma4,
                &kmap,
                &mq3_tier_layers,
                &imatrix_gguf,
                &moe_tier_map,
                use_moe_graded,
                moe_hot_frac,
                &hessian_dir,
                use_gptq_e8,
                use_gptq_mfp3e8,
                use_gptq_mfp2e8,
                use_mq6g256,
                use_mq4g256,
                use_mq4_mq6exp,
                use_mq4_mq2lloydexp,
                use_mq4_mq2glexp,
                use_mq4_mq2lloyd_native,
                use_mq4_mq2lloyd_kmap,
                use_mq4_mq2lloyd_imatrix,
                use_mq4_mq3lloyd_kmap,
                use_mq4_mqlloyd_tiered,
                use_mq4_mqlloyd_antirez,
                use_mq4_mqlloyd_antirez_gptq,
                use_mq4_mq2lloyd_gptq_all,
                use_mq5g256,
                use_hfq6,
                use_hfq4g256,
                use_hfq3g256,
                use_hfq3g128,
                use_hfq2g256,
                use_hfq2g128,
                use_hfq_mixed,
                use_mfp4,
                use_mfp4p,
                use_mfp4e8,
                use_mfp4e8soa,
                use_mfp3e8_gptq_fmt,
                use_mfp2e8_gptq_fmt,
                use_mq3g256,
                use_mq2g256,
                use_mq2g256_lloyd,
                use_mq3g256_lloyd,
                use_mq4g256_lloyd,
                use_hfp4,
                use_mfp4l,
                routed_gl,
                &imatrix_path,
                bake_keep_active,
                &reap_bake_plan,
                reap_arch,
                &st_files,
                &fp8_scale_for,
                &mut hfq_tensors,
                &mut quantized_params,
                &mut spill,
            ) {
                continue;
            }
        }

        // ── deepseek4-q8-mtp short-circuit ───────────────────────────────────────
        // Routed experts (.ffn.experts.*) were claimed by the MQ2-Lloyd
        // branch above. Here we handle everything else:
        //
        //   - antirez-precision-sensitive (compressor / indexer /
        //     router gate.weight): keep as F16 on disk. The compressor
        //     class alone regresses PPL +40-81% if dropped to MQ4
        //     (memory: project_deepseek4_compressor_must_stay_f16); F16 → Q8
        //     on these classes is a smaller hit but still unnecessary.
        //   - All other weights: uniform Q8F16.
        //   - Norms / biases / HC matrices: should_quantize() returns
        //     false → fall through to F16 fallback at the bottom.
        // deepseek4-mtp-precise: all mtp.0.* dense weights (anything that goes
        // through gemv_auto in mtp_forward — wq_a/b, wkv, wo_a/b, e_proj,
        // h_proj, shared experts, gate.weight) stay F16 to eliminate Q8
        // quant noise on the MTP block. Routed experts (".ffn.experts.")
        // are excluded — they MUST stay MQ2-Lloyd because the MoE GEMV
        // kernel (`deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed`) only
        // handles that format.
        let keep_f16_mtp = use_mtp_precise
            && name.starts_with("mtp.")
            && !name.contains(".ffn.experts.")
            && should_quantize(name);
        if (use_deepseek4_source_precision && is_deepseek4_keep_f16(name) || keep_f16_mtp)
            && n_elements >= 32
        {
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let src_dtype = meta.dtype.as_str();
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            quantized_params += n_elements as u64;
            let f16_bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect();
            eprintln!(
                "  {:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB) [src={src_dtype}, keep-F16]",
                "F16",
                name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0,
                f16_bytes.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::F16,
                shape,
                group_size: 0,
                data: f16_bytes,
                spilled_len: 0,
            });
            st_files[*file_idx].drop_tensor_pages(name);
            if let Some(ref mut s) = spill {
                maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
            }
            continue;
        }
        if use_deepseek4_source_precision && should_quantize(name) && n_elements >= 32 {
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let src_dtype = meta.dtype.as_str();
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            quantized_params += n_elements as u64;
            let q = quantize_q8f16(&f32_data);
            eprintln!(
                "  {:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB) [src={src_dtype}]",
                "Q8_F16",
                name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0,
                q.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::Q8F16,
                shape,
                group_size: 32,
                data: q,
                spilled_len: 0,
            });
            st_files[*file_idx].drop_tensor_pages(name);
            if let Some(ref mut s) = spill {
                maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
            }
            continue;
        }

        {
            let flags = MainQuantFlags {
                use_fast: use_fast,
                use_gptq_e8: use_gptq_e8,
                use_gptq_mfp2e8: use_gptq_mfp2e8,
                use_gptq_mfp3e8: use_gptq_mfp3e8,
                use_hfp4: use_hfp4,
                use_hfq2g128: use_hfq2g128,
                use_hfq2g256: use_hfq2g256,
                use_hfq3g128: use_hfq3g128,
                use_hfq3g256: use_hfq3g256,
                use_hfq4g256: use_hfq4g256,
                use_hfq6: use_hfq6,
                use_hfq_mixed: use_hfq_mixed,
                use_mfp2e8_gptq_fmt: use_mfp2e8_gptq_fmt,
                use_mfp3e8_gptq_fmt: use_mfp3e8_gptq_fmt,
                use_mfp4: use_mfp4,
                use_mfp4e8: use_mfp4e8,
                use_mfp4e8soa: use_mfp4e8soa,
                use_mfp4l: use_mfp4l,
                use_mfp4p: use_mfp4p,
                use_mixed: use_mixed,
                use_mq2g256: use_mq2g256,
                use_mq2g256_lloyd: use_mq2g256_lloyd,
                use_mq3g256: use_mq3g256,
                use_mq3g256_lloyd: use_mq3g256_lloyd,
                use_mq4_mq2glexp: use_mq4_mq2glexp,
                use_mq4_mq2lloyd_gptq_all: use_mq4_mq2lloyd_gptq_all,
                use_mq4_mq2lloyd_imatrix: use_mq4_mq2lloyd_imatrix,
                use_mq4_mq2lloyd_kmap: use_mq4_mq2lloyd_kmap,
                use_mq4_mq2lloyd_native: use_mq4_mq2lloyd_native,
                use_mq4_mq2lloydexp: use_mq4_mq2lloydexp,
                use_mq4_mq3lloyd_kmap: use_mq4_mq3lloyd_kmap,
                use_mq4_mq6exp: use_mq4_mq6exp,
                use_mq4_mqlloyd_antirez: use_mq4_mqlloyd_antirez,
                use_mq4_mqlloyd_antirez_gptq: use_mq4_mqlloyd_antirez_gptq,
                use_mq4_mqlloyd_tiered: use_mq4_mqlloyd_tiered,
                use_mq4g256: use_mq4g256,
                use_mq4v2: use_mq4v2,
                use_mq4c: use_mq4c,
                use_mq4g256_lloyd: use_mq4g256_lloyd,
                use_mq5g256: use_mq5g256,
                use_mq6g256: use_mq6g256,
                use_mq5g256v2: use_mq5g256v2,
                use_mq6g256v2: use_mq6g256v2,
                use_mq3g256v2: use_mq3g256v2,
                use_mq2g256v2: use_mq2g256v2,
                use_mq8g256: use_mq8g256,
                use_q4k_all: use_q4k_all,
                use_q4k_q8embed: use_q4k_q8embed,
                use_q8: use_q8,
                use_q8hfq: use_q8hfq,
                is_gemma4_family,
                q8_conv1d_default,
                q8_router,
                arch_id,
                vision_quant: vision_quant.to_string(),
                product_tier,
            };
            let outer = MainQuantOuter {
                kmap: &kmap,
                imatrix_gguf: &imatrix_gguf,
                hessian_dir: &hessian_dir,
            };
            let mut state = MainQuantState {
                hfq_tensors: &mut hfq_tensors,
                quantized_params: &mut quantized_params,
                total_quant_error: &mut total_quant_error,
                max_quant_error: &mut max_quant_error,
                _n_quant_groups: &mut _n_quant_groups,
                spill: &mut spill,
            };
            let ctx = PerTensorCtx {
                name,
                file_idx: *file_idx,
                shape: &meta.shape,
                n_elements,
                arch_id,
                dtype: &meta.dtype,
                is_vision,
            };
            handle_main_quant(
                &ctx,
                meta,
                raw_data,
                &flags,
                &outer,
                &mut state,
                &fp8_scale_for,
                &st_files,
            );
        } // Release source file page cache after each tensor to prevent
          // mmap'd pages from starving GPU allocations on UMA systems.
        st_files[*file_idx].drop_tensor_pages(name);
    }

    // Summary
    if use_deepseek4_mq2rxt_overlay {
        if arch_id != 9 {
            eprintln!("MQ2RXT overlay requires DeepSeek V4 arch_id=9, got {arch_id}");
            std::process::exit(2);
        }
        let sidecar = include_prefix.is_some_and(|prefix| prefix == "mtp.");
        let expected = if sidecar { 24 } else { 554 };
        if mq2rxt_overlay_count != expected {
            eprintln!(
                "MQ2RXT {} overlay selected {} tensors, expected {expected}; refusing partial recipe",
                if sidecar { "DSpark" } else { "trunk" },
                mq2rxt_overlay_count
            );
            std::process::exit(2);
        }
        eprintln!(
            "MQ2RXT {} overlay: exact {expected}-tensor P3 map encoded directly from the parent",
            if sidecar { "DSpark" } else { "trunk" }
        );
        metadata_json =
            stamp_deepseek4_mq2rxt_metadata(&metadata_json, sidecar).unwrap_or_else(|error| {
                eprintln!("MQ2RXT metadata: {error}");
                std::process::exit(2);
            });
    }
    let total_bytes: usize = hfq_tensors
        .iter()
        .map(|t| {
            if t.spilled_len > 0 {
                t.spilled_len as usize
            } else {
                t.data.len()
            }
        })
        .sum();
    {
        let fired = GPTQ_E8_FIRED.load(std::sync::atomic::Ordering::Relaxed);
        let fb = GPTQ_E8_FALLBACK.load(std::sync::atomic::Ordering::Relaxed);
        if fired + fb > 0 {
            eprintln!(
                "  GPTQ-on-E8: {fired} tensors FIRED (Hessian-aware LDLQ), {fb} RTN-fallback (missing/singular H). {:.1}% fired.",
                100.0 * fired as f64 / (fired + fb) as f64
            );
            if fired == 0 {
                eprintln!(
                    "  WARNING: 0 GPTQ tensors fired with --hessian-dir set — likely a KEY-MISMATCH (.hblk filenames != hessian_key), NOT a flat result."
                );
            }
        }
    }
    let mean_quant_error = if quantized_params > 0 {
        total_quant_error / quantized_params as f64
    } else {
        0.0
    };

    eprintln!("\n=== Quantization Summary ===");
    if skipped_params > 0 {
        eprintln!(
            "  Skipped params:   {skipped_params} (mtp/visual — use --include-vision for VL)"
        );
    }
    eprintln!("  Total params:     {total_params}");
    eprintln!(
        "  Quantized params: {quantized_params} ({:.1}%)",
        100.0 * quantized_params as f64 / total_params as f64
    );
    eprintln!("  Mean quant error: {mean_quant_error:.8}");
    eprintln!("  Max quant error:  {max_quant_error:.8}");
    eprintln!("  Output size:      {:.1} MB", total_bytes as f64 / 1e6);

    // ── Deterministic recipe census/metadata ─────────────────────────────
    {
        use std::collections::BTreeMap;
        let mut census: BTreeMap<String, usize> = BTreeMap::new();
        for t in &hfq_tensors {
            let label = format!("{:?}", t.quant_type);
            *census.entry(label).or_insert(0) += 1;
        }
        eprintln!("  Recipe census (deterministic):");
        for (k, v) in &census {
            eprintln!("    {k}: {v}");
        }
        if let Some(tier) = product_tier {
            eprintln!("  Product tier: {}", tier.label());
        }
        if let Some(map) = crate::model_filter::fixed_tier_map_cli() {
            let mut entries: Vec<String> = map.iter().map(|(k, v)| format!("{k}:{v}")).collect();
            entries.sort();
            eprintln!("  Fixed-tier: {}", entries.join(","));
        }
        // Inject into metadata_json deterministically sorted
        let mut meta_val: serde_json::Value =
            serde_json::from_str(&metadata_json).unwrap_or(serde_json::json!({}));
        if let Some(obj) = meta_val.as_object_mut() {
            obj.insert(
                "hipfire_recipe_census".to_string(),
                serde_json::to_value(&census).unwrap_or(serde_json::json!({})),
            );
            if let Some(tier) = product_tier {
                obj.insert("hipfire_product_tier".to_string(), tier.label().into());
            }
            if let Some(map) = crate::model_filter::fixed_tier_map_cli() {
                let mut sorted: BTreeMap<String, String> = BTreeMap::new();
                for (k, v) in map {
                    sorted.insert(k, v);
                }
                obj.insert(
                    "hipfire_fixed_tier".to_string(),
                    serde_json::to_value(sorted).unwrap(),
                );
            }
            obj.insert("hipfire_base_format".to_string(), format.to_string().into());
            metadata_json = serde_json::to_string(&meta_val).unwrap_or(metadata_json);
        }
    }

    // ── SP4b: bake prune finalize (rename kept per-expert tensors + patch count) ──
    // kept tensors (ds4 score layers / lfm2 / minimax) recorded during the loop to
    // their compact slots, then patches the output metadata's routed-expert count
    // to `kept_per_layer` so the baked model loads standalone (no env var, no
    // load-time keep-map). Spill preserves `.name`, so rename order vs. spill is
    // irrelevant. (Qwen3.5 stacked experts + all routers/biases were already
    // pruned/gathered in-loop.)
    // Task A0: apply Qwen3.5-MoE pre-split expert fusion renames. Unconditional
    // (unlike the bake rename below, which is gated on `bake_keep_active`):
    // rewrite each fused gate_proj output tensor to the loader's
    // `experts.{N}.gate_up_proj.weight`. The in-loop quant path kept the original
    // gate_proj name so k-map/encoding used it. Spill preserves `.name`, so this
    // works regardless of spill order. No-op (byte-identical) for every model
    // that isn't a pre-split Qwen3.5-MoE.
    if !expert_fuse_rename.is_empty() {
        for t in hfq_tensors.iter_mut() {
            if let Some(new_name) = expert_fuse_rename.get(&t.name) {
                t.name = new_name.clone();
            }
        }
        eprintln!(
            "qwen35 expert fusion: renamed {} fused gate_up_proj tensors",
            expert_fuse_rename.len()
        );
    }

    if bake_keep_active {
        let plan = reap_bake_plan.as_ref().unwrap();
        if !bake_rename.is_empty() {
            for t in hfq_tensors.iter_mut() {
                if let Some(new_name) = bake_rename.get(&t.name) {
                    t.name = new_name.clone();
                }
            }
            eprintln!(
                "REAP bake: renamed {} kept per-expert tensors to compact slots",
                bake_rename.len()
            );
        }
        let kept = plan.kept_per_layer();
        match patch_expert_count_metadata(&metadata_json, reap_arch, kept) {
            Ok(patched) => {
                metadata_json = patched;
                eprintln!("REAP bake: patched output metadata expert count → {kept}");
            }
            Err(e) => {
                eprintln!("reap bake: failed to patch expert-count metadata: {e}");
                std::process::exit(2);
            }
        }
    }

    // Write .hfq file
    eprintln!("\nWriting: {}", output_path.display());
    // Final spill before writing
    if let Some(ref mut s) = spill {
        maybe_spill(&mut hfq_tensors, s, 0); // spill everything remaining
    }
    write_hfq(
        output_path,
        arch_id,
        &metadata_json,
        &hfq_tensors,
        spill.as_mut(),
    )
    .unwrap();
    if let Some(s) = spill {
        s.cleanup();
    }

    let file_size = std::fs::metadata(output_path).unwrap().len();
    eprintln!("Done: {:.1} MB written", file_size as f64 / 1e6);
}

fn setup_thread_pool(args: &QuantizeArgs) {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let default_threads = ((cores * 8) / 10).max(1);
    let threads = args.threads.unwrap_or(default_threads);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();
    eprintln!(
        "Rayon: {threads} worker threads ({cores} cores available, default 80% = {default_threads})"
    );
}

fn handle_early_special_formats(args: &QuantizeArgs) -> bool {
    let input_dir = args.input.as_str();
    let output_path = args.output.as_str();
    let format = args.format.as_str();
    if matches!(
        format,
        "deepseek4-dense-mfp4e8soa-overlay" | "ds4-dense-e8soa-overlay"
    ) {
        if let Err(e) =
            build_deepseek4_dense_e8soa_overlay(Path::new(input_dir), Path::new(output_path))
        {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
        return true;
    }
    // ── deepseek4-dspark-e8soa: re-quantize an EXISTING DSpark/MTP sidecar's
    // dense projections Q8F16 -> MFP4-E8-SoA so the drafter matches its MQ2R
    // trunk. Input is the sidecar .hfq itself (NOT a checkpoint dir), so this
    // needs no source safetensors. Routed experts stay MQ2-Lloyd; the
    // `mq2r_sidecar` identity is stamped at build time.
    //   hipfire-quantize --format deepseek4-dspark-e8soa \
    //     --input <sidecar-in.mq2r> --output <sidecar-out.mq2r>
    if matches!(
        format,
        "deepseek4-dspark-e8soa" | "ds4-dspark-e8soa" | "deepseek4-dspark-mq2r"
    ) {
        if let Err(e) =
            build_deepseek4_dspark_e8soa_sidecar(Path::new(input_dir), Path::new(output_path))
        {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
        return true;
    }
    // ── qwen3-dspark-q8: Qwen3DSparkModel drafter sidecar emission ──────────
    if format == "qwen3-dspark-q8" || format == "qwen35-dspark-q8" {
        run_qwen3_dspark(args);
        return true;
    }
    false
}

fn run_qwen3_dspark(args: &QuantizeArgs) {
    let input_dir = Path::new(args.input.as_str());
    let output_path = Path::new(args.output.as_str());

    // Read config
    let config_path = input_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path).unwrap_or_else(|e| {
        eprintln!(
            "qwen3-dspark-q8: cannot read {}: {e}",
            config_path.display()
        );
        std::process::exit(1);
    });
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap_or_else(|e| {
        eprintln!("qwen3-dspark-q8: config.json parse error: {e}");
        std::process::exit(1);
    });

    // Verify architecture
    let archs = config.get("architectures").and_then(|v| v.as_array());
    let is_dspark = archs
        .map(|a| {
            a.iter().any(|v| {
                matches!(
                    v.as_str(),
                    Some("Qwen3DSparkModel" | "DSparkDraftModel" | "DSparkSpeculator")
                )
            })
        })
        .unwrap_or(false);
    if !is_dspark {
        eprintln!(
            "dspark-q8: architectures is not a DSpark drafter \
             (Qwen3DSparkModel / DSparkDraftModel / DSparkSpeculator); got {:?}",
            archs
        );
        std::process::exit(1);
    }

    // Read DSpark config fields. speculators v0.6.0 (DSparkDraftModel) nests
    // the body dims under `transformer_layer_config` and names the target
    // taps `aux_hidden_state_layer_ids`; the legacy Qwen3DSparkModel puts
    // dims / `target_layer_ids` at the top level. Handle both.
    let tlc = config.get("transformer_layer_config");
    let cfg_u64 = |k: &str, d: u64| -> u64 {
        config
            .get(k)
            .or_else(|| tlc.and_then(|t| t.get(k)))
            .and_then(|v| v.as_u64())
            .unwrap_or(d)
    };
    let block_size = config
        .get("block_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(7) as usize;
    let target_layer_ids: Vec<u64> = config
        .get("target_layer_ids")
        .or_else(|| config.get("aux_hidden_state_layer_ids"))
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
        .unwrap_or_else(|| vec![1, 9, 17, 25, 33]);
    let markov_rank = config
        .get("markov_rank")
        .and_then(|v| v.as_u64())
        .unwrap_or(256) as usize;
    let noise_token_id = config
        .get("mask_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(151669) as u32;
    let draft_vocab_size = cfg_u64("draft_vocab_size", 0);
    let confidence_with_markov = config
        .get("confidence_head_with_markov")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let hidden_size = cfg_u64("hidden_size", 2048);
    let head_dim = cfg_u64("head_dim", 128);
    let num_hidden_layers = cfg_u64("num_hidden_layers", 0);
    let num_attention_heads = cfg_u64("num_attention_heads", 0);
    let num_key_value_heads = cfg_u64("num_key_value_heads", 0);
    let intermediate_size = cfg_u64("intermediate_size", 0);
    let vocab_size = cfg_u64("vocab_size", 0);
    // rope params nest under transformer_layer_config.rope_parameters in v0.6.0.
    let rope = tlc
        .and_then(|t| t.get("rope_parameters"))
        .or_else(|| config.get("rope_parameters"));
    let partial_rotary_factor = rope
        .and_then(|r| r.get("partial_rotary_factor"))
        .or_else(|| config.get("partial_rotary_factor"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let rope_theta = rope
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(10000000.0);

    eprintln!(
        "qwen3-dspark-q8: block_size={block_size} target_layer_ids={target_layer_ids:?} \
         markov_rank={markov_rank} noise_token_id={noise_token_id}"
    );

    // Build metadata JSON — mirrors the keys DsparkConfig::from_metadata_json reads.
    let metadata = serde_json::json!({
        "architecture": "qwen3",
        "config": {
            "dspark_block_size": block_size,
            "dspark_target_layer_ids": target_layer_ids,
            "dspark_num_targets": target_layer_ids.len(),
            "dspark_markov_rank": markov_rank,
            "dspark_noise_token_id": noise_token_id,
            "dspark_enable_confidence": true,
            "dspark_confidence_with_markov": confidence_with_markov,
            "dspark_draft_vocab_size": draft_vocab_size,
            "dspark_hidden_size": hidden_size,
            "dspark_head_dim": head_dim,
            "dspark_num_hidden_layers": num_hidden_layers,
            "dspark_num_attention_heads": num_attention_heads,
            "dspark_num_key_value_heads": num_key_value_heads,
            "dspark_intermediate_size": intermediate_size,
            "dspark_vocab_size": vocab_size,
            "dspark_partial_rotary_factor": partial_rotary_factor,
            "dspark_rope_theta": rope_theta,
        },
    });
    let metadata_json = serde_json::to_string(&metadata).unwrap();

    // Load safetensors
    let st_paths = find_safetensors(input_dir);
    if st_paths.is_empty() {
        eprintln!(
            "qwen3-dspark-q8: no safetensors found in {}",
            input_dir.display()
        );
        std::process::exit(1);
    }
    let st_files: Vec<SafetensorsFile> = st_paths
        .iter()
        .map(|p| {
            eprintln!("Loading: {}", p.display());
            SafetensorsFile::open(p).unwrap()
        })
        .collect();

    let mut all_tensors: Vec<(&str, usize)> = Vec::new();
    for (fi, st) in st_files.iter().enumerate() {
        for name in st.tensor_names() {
            all_tensors.push((name, fi));
        }
    }
    all_tensors.sort_by_key(|(name, _)| name.to_string());
    eprintln!("qwen3-dspark-q8: {} tensors found", all_tensors.len());

    // Determine which 2D weights get Q8F16 (attn projections + MLP projections)
    let is_dspark_matmul_weight = |name: &str| -> bool {
        // Attn projections: q/k/v/o_proj
        let is_attn = name.contains("self_attn.")
            && (name.ends_with("q_proj.weight")
                || name.ends_with("k_proj.weight")
                || name.ends_with("v_proj.weight")
                || name.ends_with("o_proj.weight"));
        // MLP projections: gate/up/down_proj
        let is_mlp = name.contains("mlp.")
            && (name.ends_with("gate_proj.weight")
                || name.ends_with("up_proj.weight")
                || name.ends_with("down_proj.weight"));
        is_attn || is_mlp
    };

    let mut hfq_tensors: Vec<HfqTensor> = Vec::new();
    let mut total_params = 0u64;
    let mut q8_params = 0u64;
    let mut f16_params = 0u64;

    for (name, file_idx) in &all_tensors {
        let (meta, raw_data) = st_files[*file_idx].tensor_data(name).unwrap();
        let n_elements: usize = meta.shape.iter().product();
        total_params += n_elements as u64;

        // Map source tensor name → sidecar name
        let sidecar_name = if *name == "fc.weight" {
            "main_proj.weight".to_string()
        } else if *name == "hidden_norm.weight" {
            "main_norm.weight".to_string()
        } else {
            name.to_string()
        };

        let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();

        // Reduced-vocab maps: `d2t` (draft→target token id, I64) and `t2d`
        // (target→draft membership, BOOL). Store as F32 — token indices are
        // < 2^24 so exact; the DSpark loader casts d2t→u32, t2d→bool. The
        // float `to_f32` path can't read I64/BOOL.
        if *name == "d2t" || *name == "t2d" {
            let f32_data: Vec<f32> = if meta.dtype == "I64" {
                raw_data
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect()
            } else if meta.dtype == "BOOL" || meta.dtype == "U8" {
                raw_data
                    .iter()
                    .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                    .collect()
            } else {
                to_f32(raw_data, &meta.dtype)
            };
            let bytes: Vec<u8> = f32_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
            eprintln!(
                "  {:>8}: {} {:?} ({} elems) [reduced-vocab map]",
                "F32", sidecar_name, meta.shape, n_elements
            );
            f16_params += n_elements as u64;
            hfq_tensors.push(HfqTensor {
                name: sidecar_name,
                quant_type: QuantType::F32,
                shape,
                group_size: 0,
                data: bytes,
                spilled_len: 0,
            });
            continue;
        }

        if is_dspark_matmul_weight(name) && n_elements >= 32 {
            // 2D matmul weight → Q8F16 (body layers, trained precision preserved)
            let f32_data = to_f32(raw_data, &meta.dtype);
            let q = quantize_q8f16(&f32_data);
            eprintln!(
                "  {:>8}: {} {:?} ({} elems, {:.1} KB → {:.1} KB)",
                "Q8_F16",
                sidecar_name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0,
                q.len() as f64 / 1024.0
            );
            q8_params += n_elements as u64;
            hfq_tensors.push(HfqTensor {
                name: sidecar_name,
                quant_type: QuantType::Q8F16,
                shape,
                group_size: 32,
                data: q,
                spilled_len: 0,
            });
        } else {
            // Everything else → F16 (norms, embeds, main_proj, markov, confidence, lm_head)
            let f32_data = to_f32(raw_data, &meta.dtype);
            let f16_bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect();
            eprintln!(
                "  {:>8}: {} {:?} ({} elems, {:.1} KB → {:.1} KB)",
                "F16",
                sidecar_name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0,
                f16_bytes.len() as f64 / 1024.0
            );
            f16_params += n_elements as u64;
            hfq_tensors.push(HfqTensor {
                name: sidecar_name,
                quant_type: QuantType::F16,
                shape,
                group_size: 0,
                data: f16_bytes,
                spilled_len: 0,
            });
        }
    }

    eprintln!(
        "\n=== qwen3-dspark-q8 Summary ===\n\
         Total params:  {total_params}\n\
         Q8F16 params:  {q8_params} ({:.1}%)\n\
         F16 params:    {f16_params} ({:.1}%)\n\
         Tensors:       {}",
        100.0 * q8_params as f64 / total_params as f64,
        100.0 * f16_params as f64 / total_params as f64,
        hfq_tensors.len()
    );

    eprintln!("\nWriting: {}", output_path.display());
    write_hfq(output_path, 1u32, &metadata_json, &hfq_tensors, None).unwrap_or_else(|e| {
        eprintln!("qwen3-dspark-q8: write_hfq failed: {e}");
        std::process::exit(2);
    });

    let file_size = std::fs::metadata(output_path).unwrap().len();
    eprintln!("Done: {:.1} MB written", file_size as f64 / 1e6);
}

fn try_handle_lfm2moe(
    is_lfm2moe: bool,
    use_mq4g256: bool,
    use_mq4v2: bool,
    use_mq4c: bool,
    name: &str,
    meta: &TensorMeta,
    raw_data: &[u8],
    fp8_scale_for: &HashMap<String, (usize, String)>,
    st_files: &[SafetensorsFile],
    hfq_tensors: &mut Vec<HfqTensor>,
    quantized_params: &mut u64,
    spill: &mut Option<TensorSpill>,
    file_idx: usize,
    n_elements: usize,
) -> bool {
    // ── LFM2.5 ingest (arch_id 11) ─────────────────────────────────────────
    // Routed experts (A1B only) → MQ4G256; expert_bias → F32; everything else
    // (conv in/out_proj, conv depthwise filter, attn q/k/v/out_proj + qk-norm,
    // dense w1/w2/w3, router gate, operator/ffn/embedding norms, tied embed/
    // lm_head) → Q8 (qt=3 Q8F16). Dense lfm2 (350M/1.2B) has no experts, so
    // every tensor takes the final Q8 path. The loader's load_f32 dequantizes
    // Q8 norms / conv-filter back to F32 on load.
    if is_lfm2moe {
        let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
        if name.contains(".feed_forward.experts.")
            && (name.ends_with(".w1.weight")
                || name.ends_with(".w2.weight")
                || name.ends_with(".w3.weight"))
            && meta.shape.len() == 2
            && meta.shape[1] % 256 == 0
        {
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            let signs1 = gen_fwht_signs(42, 256);
            let signs2 = gen_fwht_signs(1042, 256);
            let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
            eprintln!(
                "  {:>8}: {} {:?} ({:.1} KB → {:.1} KB)",
                "MQ4-LFM",
                name,
                meta.shape,
                raw_data.len() as f64 / 1024.0,
                q.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::MQ4G256,
                shape,
                group_size: 256,
                data: q,
                spilled_len: 0,
            });
            *quantized_params += (meta.shape[0] * meta.shape[1]) as u64;
            st_files[file_idx].drop_tensor_pages(name);
            if let Some(s) = spill.as_mut() {
                maybe_spill(hfq_tensors, s, 2 * 1024 * 1024 * 1024);
            }
            return true;
        }
        if name.ends_with(".feed_forward.expert_bias") {
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            let mut bytes = Vec::with_capacity(f32_data.len() * 4);
            for v in &f32_data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            eprintln!(
                "  {:>8}: {} {:?} (expert_bias F32)",
                "F32-LFM", name, meta.shape
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::F32,
                shape,
                group_size: 1,
                data: bytes,
                spilled_len: 0,
            });
            st_files[file_idx].drop_tensor_pages(name);
            return true;
        }
        // Dense mq4 (--format mq4): route the big 2D proj/FFN weight matrices
        // (conv in/out_proj, attn q/k/v/out_proj, dense w1/w2/w3) → MQ4G256.
        // The loader's weight_gemv / weight_gemv_residual auto-FWHT-rotate
        // MQ4G256, so no forward change is needed. Keep the tied embed/lm_head
        // (model.embed_tokens.weight), the router gate, norms, and the depthwise
        // conv filter at Q8/F32 (small + precision-sensitive). Default (no mq4
        // format) keeps the full-precision Q8 bring-up recipe.
        if (use_mq4g256 || use_mq4v2 || use_mq4c)
            && meta.shape.len() == 2
            && meta.shape[1] % 256 == 0
            && !name.ends_with("embed_tokens.weight")
            && (name.ends_with("_proj.weight")
                || name.ends_with(".w1.weight")
                || name.ends_with(".w2.weight")
                || name.ends_with(".w3.weight"))
        {
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                &fp8_scale_for,
                &st_files,
            );
            let signs1 = gen_fwht_signs(42, 256);
            let signs2 = gen_fwht_signs(1042, 256);
            let (q, qt, label) = if use_mq4c {
                let m = meta.shape[0];
                let k = meta.shape[1];
                let qq = quantize_mq4cg256(&f32_data, m, k, &signs1, &signs2);
                (qq, QuantType::MQ4CG256, "MQ4C-LFM")
            } else if use_mq4v2 {
                let m = meta.shape[0];
                let k = meta.shape[1];
                let qq = quantize_mq4g256v2(&f32_data, m, k, &signs1, &signs2);
                (qq, QuantType::MQ4G256V2, "MQ4V2-LFM")
            } else {
                let qq = quantize_mq4g256(&f32_data, &signs1, &signs2);
                (qq, QuantType::MQ4G256, "MQ4-LFM")
            };
            eprintln!(
                "  {:>8}: {} {:?} ({:.1} KB → {:.1} KB)",
                label,
                name,
                meta.shape,
                raw_data.len() as f64 / 1024.0,
                q.len() as f64 / 1024.0
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: qt,
                shape,
                group_size: 256,
                data: q,
                spilled_len: 0,
            });
            *quantized_params += (meta.shape[0] * meta.shape[1]) as u64;
            st_files[file_idx].drop_tensor_pages(name);
            if let Some(s) = spill.as_mut() {
                maybe_spill(hfq_tensors, s, 2 * 1024 * 1024 * 1024);
            }
            return true;
        }

        // All remaining LFM2 tensors → Q8 (qt=3). quantize_q8f16 handles any
        // 1D/2D/3D shape elementwise (conv.conv.weight is [hidden,1,K]).
        let f32_data =
            tensor_to_f32_with_optional_fp8_scale(name, raw_data, meta, &fp8_scale_for, &st_files);
        let q = quantize_q8f16(&f32_data);
        eprintln!("  {:>8}: {} {:?} (Q8)", "Q8-LFM", name, meta.shape);
        hfq_tensors.push(HfqTensor {
            name: name.to_string(),
            quant_type: QuantType::Q8F16,
            shape,
            group_size: 32,
            data: q,
            spilled_len: 0,
        });
        *quantized_params += n_elements as u64;
        st_files[file_idx].drop_tensor_pages(name);
        return true;
    }

    false
}

enum BakeKeepResult {
    Pruned,
    Gathered { meta: TensorMeta, bytes: Vec<u8> },
}

fn handle_bake_keep_active(
    name: &str,
    file_idx: usize,
    meta: &TensorMeta,
    raw_data: &[u8],
    bake_keep_active: bool,
    reap_bake_plan: &Option<hipfire_reap::plan::ReapPlan>,
    reap_arch: reap_overlay::ReapArch,
    bake_rename: &mut HashMap<String, String>,
    st_files: &[SafetensorsFile],
) -> Option<BakeKeepResult> {
    if !bake_keep_active {
        return None;
    }
    let plan = reap_bake_plan.as_ref().unwrap();
    let keep = plan.keep.as_ref().unwrap();
    let layer = reap_overlay::bake_layer_of(name);
    if reap_overlay::expert_index_of(name, reap_arch).is_some() {
        let l = layer.unwrap_or_else(|| {
            eprintln!("reap bake: routed-expert tensor '{name}' has no parseable layer");
            std::process::exit(2);
        });
        if reap_arch == reap_overlay::ReapArch::Deepseek4 && l <= 2 {
            eprintln!(
                "reap bake: ds4 hash-layer (0-2) tid2eid remap not supported in bake;                  use the load-time keep-map for pruned ds4 hash layers"
            );
            std::process::exit(2);
        }
        if l >= keep.len() {
            eprintln!("reap bake: layer {l} for '{name}' out of keep-map range");
            std::process::exit(2);
        }
        match reap_overlay::bake_expert_rename(name, reap_arch, l, &keep[l]) {
            None => {
                st_files[file_idx].drop_tensor_pages(name);
                return Some(BakeKeepResult::Pruned);
            }
            Some(new_name) => {
                if &new_name != name {
                    bake_rename.insert(name.to_string(), new_name);
                }
            }
        }
    } else if let Some(l) = layer {
        let is_router_w = reap_overlay::is_reap_router_weight(name, reap_arch)
            && meta.shape.len() == 2
            && meta.shape[0] == plan.original_experts;
        let is_expert_bias = reap_overlay::is_reap_expert_bias(name, reap_arch)
            && meta.shape.len() == 1
            && meta.shape[0] == plan.original_experts;
        if is_router_w || is_expert_bias {
            if reap_arch == reap_overlay::ReapArch::Deepseek4 && l <= 2 {
                eprintln!(
                    "reap bake: ds4 hash-layer (0-2) router/tid2eid remap not supported                      in bake; use the load-time keep-map for pruned ds4 hash layers"
                );
                std::process::exit(2);
            }
            if l >= keep.len() {
                eprintln!("reap bake: layer {l} for router/bias '{name}' out of keep-map range");
                std::process::exit(2);
            }
            let keep_l = &keep[l];
            match hipfire_reap::gather::gather_rows(&meta.shape, raw_data, keep_l) {
                Ok((new_shape, gathered)) => {
                    let gathered_meta = TensorMeta {
                        dtype: meta.dtype.clone(),
                        shape: new_shape,
                        data_offsets: meta.data_offsets,
                    };
                    eprintln!(
                        "  {:>8}: {} {:?} → rows[{}] (kept {} of {})",
                        "GATHER",
                        name,
                        meta.shape,
                        keep_l.len(),
                        keep_l.len(),
                        plan.original_experts
                    );
                    return Some(BakeKeepResult::Gathered {
                        meta: gathered_meta,
                        bytes: gathered,
                    });
                }
                Err(e) => {
                    eprintln!("reap bake: router/bias gather '{name}': {e}");
                    std::process::exit(2);
                }
            }
        }
    }
    None
}

fn handle_f32_passthrough(
    ctx: &PerTensorCtx,
    meta: &TensorMeta,
    raw_data: &[u8],
    is_cohere2moe: bool,
    is_moe: bool,
    use_f32_passthrough: bool,
    fp8_scale_for: &HashMap<String, (usize, String)>,
    st_files: &[SafetensorsFile],
    hfq_tensors: &mut Vec<HfqTensor>,
    quantized_params: &mut u64,
    spill: &mut Option<TensorSpill>,
) -> bool {
    if !use_f32_passthrough || is_cohere2moe {
        return false;
    }
    if is_moe
        && ctx.name.contains("mlp.experts.")
        && (ctx.name.ends_with("gate_up_proj") || ctx.name.ends_with("down_proj"))
        && meta.shape.len() == 3
    {
        let n_exp = meta.shape[0];
        let inner_n: usize = meta.shape[1..].iter().product();
        let base_name = if ctx.name.ends_with("gate_up_proj") {
            "gate_up_proj"
        } else {
            "down_proj"
        };
        let parent = &ctx.name[..ctx.name.len() - base_name.len()];
        let inner_shape: Vec<u32> = meta.shape[1..].iter().map(|&d| d as u32).collect();
        let f32_all = tensor_to_f32_with_optional_fp8_scale(
            ctx.name,
            raw_data,
            meta,
            fp8_scale_for,
            st_files,
        );
        for x in 0..n_exp {
            let slice = &f32_all[x * inner_n..(x + 1) * inner_n];
            let bytes: Vec<u8> = slice.iter().flat_map(|&v| v.to_le_bytes()).collect();
            hfq_tensors.push(HfqTensor {
                name: format!("{parent}{x}.{base_name}.weight"),
                quant_type: QuantType::F32,
                shape: inner_shape.clone(),
                group_size: 0,
                data: bytes,
                spilled_len: 0,
            });
        }
        *quantized_params += ctx.n_elements as u64;
        eprintln!(
            "  {:>8}: {} {:?} -> {} per-expert F32 [oracle split]",
            "F32", ctx.name, meta.shape, n_exp
        );
        st_files[ctx.file_idx].drop_tensor_pages(ctx.name);
        if let Some(sp) = spill.as_mut() {
            maybe_spill(hfq_tensors, sp, 2 * 1024 * 1024 * 1024);
        }
        return true;
    }
    let f32_data =
        tensor_to_f32_with_optional_fp8_scale(ctx.name, raw_data, meta, fp8_scale_for, st_files);
    let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
    let bytes: Vec<u8> = f32_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
    *quantized_params += ctx.n_elements as u64;
    eprintln!(
        "  {:>8}: {} {:?} ({} elements, {:.1} KB -> {:.1} KB) [F32 oracle passthrough]",
        "F32",
        ctx.name,
        meta.shape,
        ctx.n_elements,
        raw_data.len() as f64 / 1024.0,
        bytes.len() as f64 / 1024.0
    );
    hfq_tensors.push(HfqTensor {
        name: ctx.name.to_string(),
        quant_type: QuantType::F32,
        shape,
        group_size: 0,
        data: bytes,
        spilled_len: 0,
    });
    st_files[ctx.file_idx].drop_tensor_pages(ctx.name);
    if let Some(sp) = spill.as_mut() {
        maybe_spill(hfq_tensors, sp, 2 * 1024 * 1024 * 1024);
    }
    true
}

fn handle_bf16_passthrough(
    ctx: &PerTensorCtx,
    meta: &TensorMeta,
    raw_data: &[u8],
    use_bf16: bool,
    arch_id: u32,
    is_vision: bool,
    is_moe: bool,
    fp8_scale_for: &HashMap<String, (usize, String)>,
    st_files: &[SafetensorsFile],
    hfq_tensors: &mut Vec<HfqTensor>,
    quantized_params: &mut u64,
    spill: &mut Option<TensorSpill>,
) -> bool {
    if !use_bf16 || !matches!(arch_id, 5 | 6) || is_vision {
        return false;
    }
    let bf16_bytes = if meta.dtype == "BF16" {
        raw_data.to_vec()
    } else {
        tensor_to_f32_with_optional_fp8_scale(ctx.name, raw_data, meta, fp8_scale_for, st_files)
            .iter()
            .flat_map(|&value| {
                let bits = value.to_bits();
                let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
                ((rounded >> 16) as u16).to_le_bytes()
            })
            .collect()
    };

    if is_moe
        && ctx.name.contains("mlp.experts.")
        && (ctx.name.ends_with("gate_up_proj") || ctx.name.ends_with("down_proj"))
        && meta.shape.len() == 3
    {
        let n_exp = meta.shape[0];
        let inner_n: usize = meta.shape[1..].iter().product();
        let base_name = if ctx.name.ends_with("gate_up_proj") {
            "gate_up_proj"
        } else {
            "down_proj"
        };
        let parent = &ctx.name[..ctx.name.len() - base_name.len()];
        let inner_shape: Vec<u32> = meta.shape[1..].iter().map(|&d| d as u32).collect();
        for expert in 0..n_exp {
            let start = expert * inner_n * 2;
            let end = start + inner_n * 2;
            hfq_tensors.push(HfqTensor {
                name: format!("{parent}{expert}.{base_name}.weight"),
                quant_type: QuantType::BF16,
                shape: inner_shape.clone(),
                group_size: 0,
                data: bf16_bytes[start..end].to_vec(),
                spilled_len: 0,
            });
        }
        eprintln!(
            "  {:>8}: {} {:?} -> {} per-expert BF16 [source split]",
            "BF16", ctx.name, meta.shape, n_exp
        );
    } else {
        eprintln!(
            "  {:>8}: {} {:?} ({} elements, {:.1} KB) [source passthrough]",
            "BF16",
            ctx.name,
            meta.shape,
            ctx.n_elements,
            bf16_bytes.len() as f64 / 1024.0
        );
        hfq_tensors.push(HfqTensor {
            name: ctx.name.to_string(),
            quant_type: QuantType::BF16,
            shape: meta.shape.iter().map(|&s| s as u32).collect(),
            group_size: 0,
            data: bf16_bytes,
            spilled_len: 0,
        });
    }
    *quantized_params += ctx.n_elements as u64;
    st_files[ctx.file_idx].drop_tensor_pages(ctx.name);
    if let Some(sp) = spill.as_mut() {
        maybe_spill(hfq_tensors, sp, 2 * 1024 * 1024 * 1024);
    }
    true
}

fn handle_cohere2moe(
    name: &str,
    meta: &TensorMeta,
    raw_data: &[u8],
    n_elements: usize,
    is_cohere2moe: bool,
    use_bf16: bool,
    use_f16: bool,
    use_mq6g256: bool,
    use_mq4g256: bool,
    fp8_scale_for: &HashMap<String, (usize, String)>,
    st_files: &[SafetensorsFile],
    hfq_tensors: &mut Vec<HfqTensor>,
    quantized_params: &mut u64,
    spill: &mut Option<TensorSpill>,
    file_idx: usize,
) -> bool {
    if !is_cohere2moe {
        return false;
    }
    let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
    if name.contains("norm") {
        let f32_data =
            tensor_to_f32_with_optional_fp8_scale(name, raw_data, meta, fp8_scale_for, st_files);
        let f16_bytes: Vec<u8> = f32_data
            .iter()
            .flat_map(|&v| f32_to_f16(v).to_le_bytes())
            .collect();
        eprintln!("  {:>8}: {} {:?} (norm F16)", "F16-COH", name, meta.shape);
        hfq_tensors.push(HfqTensor {
            name: name.to_string(),
            quant_type: QuantType::F16,
            shape,
            group_size: 0,
            data: f16_bytes,
            spilled_len: 0,
        });
        *quantized_params += n_elements as u64;
        st_files[file_idx].drop_tensor_pages(name);
        return true;
    }

    if name.ends_with("embed_tokens.weight") {
        let f32_data =
            tensor_to_f32_with_optional_fp8_scale(name, raw_data, meta, fp8_scale_for, st_files);
        let q = quantize_q8f16(&f32_data);
        eprintln!(
            "  {:>8}: {} {:?} (tied embed Q8)",
            "Q8-COH", name, meta.shape
        );
        hfq_tensors.push(HfqTensor {
            name: name.to_string(),
            quant_type: QuantType::Q8F16,
            shape,
            group_size: 32,
            data: q,
            spilled_len: 0,
        });
        *quantized_params += n_elements as u64;
        st_files[file_idx].drop_tensor_pages(name);
        return true;
    }

    if meta.shape.len() == 2 && meta.shape[1] % 256 == 0 {
        let is_expert = name.contains(".mlp.experts.");
        let is_router = name.ends_with(".mlp.gate.weight");
        if use_bf16 && !is_router && meta.dtype == "BF16" {
            eprintln!(
                "  {:>8}: {} {:?} (native bf16)",
                "BF16-COH", name, meta.shape
            );
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::BF16,
                shape,
                group_size: 0,
                data: raw_data.to_vec(),
                spilled_len: 0,
            });
            *quantized_params += (meta.shape[0] * meta.shape[1]) as u64;
            st_files[file_idx].drop_tensor_pages(name);
            if let Some(s) = spill.as_mut() {
                maybe_spill(hfq_tensors, s, 2 * 1024 * 1024 * 1024);
            }
            return true;
        }

        let dt = if is_router {
            QuantType::Q8F16
        } else if use_f16 {
            QuantType::F16
        } else if is_expert {
            if use_mq6g256 {
                QuantType::MQ6G256
            } else if use_mq4g256 {
                QuantType::MQ4G256
            } else {
                QuantType::Q8F16
            }
        } else {
            QuantType::Q8F16
        };
        let f32_data =
            tensor_to_f32_with_optional_fp8_scale(name, raw_data, meta, fp8_scale_for, st_files);
        let (data, gs, tag): (Vec<u8>, u32, &str) = match dt {
            QuantType::F16 => (
                f32_data
                    .iter()
                    .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                    .collect(),
                0,
                "F16-COH",
            ),
            QuantType::MQ6G256 => {
                let s1 = gen_fwht_signs(42, 256);
                let s2 = gen_fwht_signs(1042, 256);
                (quantize_mq6g256(&f32_data, &s1, &s2), 256, "MQ6-COH")
            }
            QuantType::MQ4G256 => {
                let s1 = gen_fwht_signs(42, 256);
                let s2 = gen_fwht_signs(1042, 256);
                (quantize_mq4g256(&f32_data, &s1, &s2), 256, "MQ4-COH")
            }
            _ => (quantize_q8f16(&f32_data), 32, "Q8-COH"),
        };
        eprintln!(
            "  {:>8}: {} {:?} ({:.1} KB -> {:.1} KB)",
            tag,
            name,
            meta.shape,
            raw_data.len() as f64 / 1024.0,
            data.len() as f64 / 1024.0
        );
        hfq_tensors.push(HfqTensor {
            name: name.to_string(),
            quant_type: dt,
            shape,
            group_size: gs,
            data,
            spilled_len: 0,
        });
        *quantized_params += (meta.shape[0] * meta.shape[1]) as u64;
        st_files[file_idx].drop_tensor_pages(name);
        if let Some(s) = spill.as_mut() {
            maybe_spill(hfq_tensors, s, 2 * 1024 * 1024 * 1024);
        }
        return true;
    }

    let f32_data =
        tensor_to_f32_with_optional_fp8_scale(name, raw_data, meta, fp8_scale_for, st_files);
    let q = quantize_q8f16(&f32_data);
    eprintln!("  {:>8}: {} {:?} (Q8)", "Q8-COH", name, meta.shape);
    hfq_tensors.push(HfqTensor {
        name: name.to_string(),
        quant_type: QuantType::Q8F16,
        shape,
        group_size: 32,
        data: q,
        spilled_len: 0,
    });
    *quantized_params += n_elements as u64;
    st_files[file_idx].drop_tensor_pages(name);
    true
}

fn handle_moe_expert_3d(
    ctx: &PerTensorCtx,
    meta: &TensorMeta,
    raw_data: &[u8],
    is_moe: bool,
    is_gemma4: bool,
    kmap: &HashMap<String, QuantLevel>,
    mq3_tier_layers: &std::collections::HashSet<usize>,
    imatrix_gguf: &Option<gguf_input::GgufFile>,
    moe_tier_map: &Option<std::collections::HashMap<(usize, usize), QuantType>>,
    use_moe_graded: bool,
    moe_hot_frac: f64,
    hessian_dir: &Option<PathBuf>,
    use_gptq_e8: bool,
    use_gptq_mfp3e8: bool,
    use_gptq_mfp2e8: bool,
    use_mq6g256: bool,
    use_mq4g256: bool,
    use_mq4_mq6exp: bool,
    use_mq4_mq2lloydexp: bool,
    use_mq4_mq2glexp: bool,
    use_mq4_mq2lloyd_native: bool,
    use_mq4_mq2lloyd_kmap: bool,
    use_mq4_mq2lloyd_imatrix: bool,
    use_mq4_mq3lloyd_kmap: bool,
    use_mq4_mqlloyd_tiered: bool,
    use_mq4_mqlloyd_antirez: bool,
    use_mq4_mqlloyd_antirez_gptq: bool,
    use_mq4_mq2lloyd_gptq_all: bool,
    use_mq5g256: bool,
    use_hfq6: bool,
    use_hfq4g256: bool,
    use_hfq3g256: bool,
    use_hfq3g128: bool,
    use_hfq2g256: bool,
    use_hfq2g128: bool,
    use_hfq_mixed: bool,
    use_mfp4: bool,
    use_mfp4p: bool,
    use_mfp4e8: bool,
    use_mfp4e8soa: bool,
    use_mfp3e8_gptq_fmt: bool,
    use_mfp2e8_gptq_fmt: bool,
    use_mq3g256: bool,
    use_mq2g256: bool,
    use_mq2g256_lloyd: bool,
    use_mq3g256_lloyd: bool,
    use_mq4g256_lloyd: bool,
    use_hfp4: bool,
    use_mfp4l: bool,
    routed_gl: bool,
    imatrix_path: &Option<PathBuf>,
    bake_keep_active: bool,
    reap_bake_plan: &Option<hipfire_reap::plan::ReapPlan>,
    reap_arch: reap_overlay::ReapArch,
    st_files: &[SafetensorsFile],
    fp8_scale_for: &HashMap<String, (usize, String)>,
    hfq_tensors: &mut Vec<HfqTensor>,
    quantized_params: &mut u64,
    spill: &mut Option<TensorSpill>,
) -> bool {
    let name = ctx.name;
    let file_idx = ctx.file_idx;
    let n_elements = ctx.n_elements;
    let arch_id = ctx.arch_id;
    let is_vision = ctx.is_vision;

    let n_experts = meta.shape[0];
    let inner_n: usize = meta.shape[1..].iter().product();
    let elem_size = match meta.dtype.as_str() {
        "F32" => 4,
        "F16" | "BF16" => 2,
        other => panic!("unsupported expert tensor dtype: {other}"),
    };
    let inner_bytes = inner_n * elem_size;
    let inner_shape: Vec<u32> = meta.shape[1..].iter().map(|&s| s as u32).collect();
    let base_name = if name.ends_with("gate_up_proj") {
        "gate_up_proj"
    } else {
        "down_proj"
    };
    // Strip the trailing base; what remains is the parent path with `experts.` already on the end
    let parent = &name[..name.len() - base_name.len()];

    // Inner quantization for experts — respects --format flag.
    // MQ6 reduces quantization error that compounds across 48 MoE
    // layers × 9 expert contributions per layer at the cost of ~50%
    // more VRAM per expert. MQ4 is the default for VRAM efficiency.
    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);
    let inner_k = inner_shape[1] as usize;
    let supports_g256 = inner_k % 256 == 0;
    // K-map: check the parent tensor name directly. The parent
    // (e.g. "...mlp.experts.gate_up_proj") contains "mlp.experts."
    // so kmap_resolve rule 4 matches it. The kmap HashMap was built
    // from all_tensors which has these parent names as keys.
    let kmap_promote = kmap.get(name) == Some(&QuantLevel::Promote6);
    // Phase 5 tiering decision needs the layer index for this parent.
    // Computed once here and reused by both expert_mq2lloyd_native
    // and expert_mq3lloyd_native below.
    let parent_layer: Option<usize> = {
        let marker = ".layers.";
        parent.rfind(marker).and_then(|i| {
            let rest = &parent[i + marker.len()..];
            rest.split('.').next().and_then(|s| s.parse().ok())
        })
    };
    let tiered_layer_is_mq3 = use_mq4_mqlloyd_tiered
        && !kmap_promote
        && parent_layer
            .map(|l| mq3_tier_layers.contains(&l))
            .unwrap_or(false);
    let tiered_layer_is_mq2 = use_mq4_mqlloyd_tiered
        && !kmap_promote
        && parent_layer
            .map(|l| !mq3_tier_layers.contains(&l))
            .unwrap_or(false);
    // Antirez-style: gate_up → MQ2, down → MQ3 (kmap-respecting).
    // Selects based on `base_name` ("gate_up_proj" vs "down_proj").
    let is_gate_up = base_name == "gate_up_proj";
    let antirez_mq3 =
        (use_mq4_mqlloyd_antirez || use_mq4_mqlloyd_antirez_gptq) && !kmap_promote && !is_gate_up;
    let antirez_mq2 =
        (use_mq4_mqlloyd_antirez || use_mq4_mqlloyd_antirez_gptq) && !kmap_promote && is_gate_up;
    // Lever 2: GPTQ-style sequential Lloyd specifically for the
    // gate_up MQ2 path. Sets a flag the inner quant dispatch will
    // honor (separate from the imatrix-only path).
    let use_gptq_for_gate_up = use_mq4_mqlloyd_antirez_gptq && antirez_mq2;
    // For the kmap-respecting MQ2-Lloyd variants, kmap_promote experts
    // get MQ6 instead of MQ2-Lloyd. Falls through to expert_mq6 below.
    let expert_mq6 = (use_mq6g256
        || use_mq4_mq6exp
        || (kmap_promote && use_mq4g256)
        || (kmap_promote && use_mq4_mq2lloyd_kmap)
        || (kmap_promote && use_mq4_mq2lloyd_imatrix)
        || (kmap_promote && use_mq4_mq2lloyd_gptq_all)
        || (kmap_promote && use_mq4_mq3lloyd_kmap))
        && supports_g256;
    // MQ5 routed experts: `--format mq5` ships ALL experts at MQ5
    // (mirrors expert_mq6's use_mq6g256 base-format case). The env-var
    // levers (HIPFIRE_MOE_EXPERTS_MQ5 / _DOWN_MQ5) below add the
    // gate_up-stays-MQ4 + down-only-MQ5 recipe via `down_mq5`.
    let expert_mq5 = use_mq5g256 && supports_g256;
    let expert_hfq6 = (use_hfq6 || (kmap_promote && use_hfq4g256)) && supports_g256;
    let expert_hfq4 = use_hfq4g256 && !kmap_promote && supports_g256;
    // HIPFIRE_MOE_DOWN_MQ6=1: promote ONLY the expert down_proj to MQ6
    // (gate_up stays MQ4) — the "mq6-down" precision lever, composable
    // with down-AWQ. Kept OUT of `expert_mq6` so `expert_awq_active` still
    // fires; the AWQ branch below switches its output format to MQ6.
    // HIPFIRE_MOE_EXPERTS_MQ6=1 promotes BOTH gate_up + down to MQ6 (the
    // experts-level "+P" / kmap-experts recipe, minus the gfx12-only dense
    // attn promotion). HIPFIRE_MOE_DOWN_MQ6=1 promotes only down. `down_mq6`
    // means "promote THIS expert tensor to MQ6" (gate_up or down).
    let experts_mq6_all = hipfire_config::developer_var("HIPFIRE_MOE_EXPERTS_MQ6")
        .ok()
        .as_deref()
        == Some("1");
    let down_mq6 = supports_g256
        && (experts_mq6_all
            || (hipfire_config::developer_var("HIPFIRE_MOE_DOWN_MQ6")
                .ok()
                .as_deref()
                == Some("1")
                && base_name == "down_proj"));
    // HIPFIRE_MOE_EXPERTS_MQ5=1 promotes BOTH gate_up + down to MQ5; the
    // experts-level 5-bit recipe (5.25 bpw, between MQ4 and MQ6).
    // HIPFIRE_MOE_DOWN_MQ5=1 promotes ONLY the expert down_proj to MQ5
    // (gate_up stays MQ4). Kept OUT of `expert_mq5` so `expert_awq_active`
    // still fires; the AWQ branch switches its output format to MQ5.
    let experts_mq5_all = hipfire_config::developer_var("HIPFIRE_MOE_EXPERTS_MQ5")
        .ok()
        .as_deref()
        == Some("1");
    let down_mq5 = supports_g256
        && (experts_mq5_all
            || (hipfire_config::developer_var("HIPFIRE_MOE_DOWN_MQ5")
                .ok()
                .as_deref()
                == Some("1")
                && base_name == "down_proj"));
    // mq4-mq2lloydexp round-trip probe: ALWAYS hits routed experts
    // (overrides any kmap promotion). The intent is to inject MQ2
    // noise specifically on the routed-expert tensors, so even
    // K-map "Promote6" experts get the MQ2-Lloyd round-trip here.
    let expert_mq2lloyd_roundtrip = use_mq4_mq2lloydexp && supports_g256;
    // GL twin — same "always hits routed experts" intent as above.
    let expert_mq2gl_roundtrip = use_mq4_mq2glexp && supports_g256;
    // Native MQ2-Lloyd: ship qt=19 bytes directly, no round-trip.
    // Requires runtime support for DType::MQ2G256Lloyd on experts.
    // For -native (no kmap respect): always MQ2-Lloyd on every expert.
    // For -kmap / -imatrix (kmap respect): only non-promoted experts
    // go MQ2-Lloyd; promoted ones hit `expert_mq6` above.
    // All-MQ2-GPTQ test: ALL routed experts at MQ2-Lloyd, both
    // gate_up and down. Respects kmap_promote (promoted layers
    // still get MQ6). Uses sequential-GPTQ Lloyd everywhere via
    // the `use_gptq_for_all_mq2` flag below.
    let all_mq2_gptq = use_mq4_mq2lloyd_gptq_all && !kmap_promote;
    let expert_mq2lloyd_native = (use_mq4_mq2lloyd_native
        || (use_mq4_mq2lloyd_kmap && !kmap_promote)
        || (use_mq4_mq2lloyd_imatrix && !kmap_promote)
        || tiered_layer_is_mq2
        || antirez_mq2
        || all_mq2_gptq)
        && supports_g256;
    // GPTQ assignment fires for both gate_up and down when in
    // all-MQ2-GPTQ mode (not just gate_up like the antirez split).
    let use_gptq_for_gate_up = use_gptq_for_gate_up || (all_mq2_gptq && imatrix_path.is_some());
    // MQ3-Lloyd asymmetric: non-promoted experts → qt=20 (3.5 bpw).
    // Promoted ones hit `expert_mq6` above (note: kmap_promote already
    // includes use_mq4_mq3lloyd_kmap via the expert_mq6 expression).
    //
    // Phase 5 tiered variant: also MQ3-Lloyd on hot non-promoted
    // layers (the ones in `mq3_tier_layers`, decided above by imatrix
    // .counts ranking).
    let expert_mq3lloyd_native =
        ((use_mq4_mq3lloyd_kmap && !kmap_promote) || tiered_layer_is_mq3 || antirez_mq3)
            && supports_g256;
    // Per-expert column-weights from the imatrix file, used only by
    // the imatrix variant. Built once per parent (cheap), then sliced
    // per expert inside the rayon loop. Falls back to None when the
    // imatrix tensor for this parent isn't found (e.g. a non-expert
    // tensor we accidentally route here, or a layer that wasn't in
    // the calibration set).
    let imatrix_lookup_name = format!("{}{}", parent, base_name);
    let imatrix_per_expert: Option<Vec<Vec<f32>>> = if (use_mq4_mq2lloyd_imatrix
        || use_mq4_mqlloyd_antirez
        || use_mq4_mqlloyd_antirez_gptq
        || use_mq4_mq2lloyd_gptq_all)
        && imatrix_gguf.is_some()
        && expert_mq2lloyd_native
    {
        imatrix_col_weights_for_parent(
            imatrix_gguf.as_ref().unwrap(),
            &imatrix_lookup_name,
            n_experts,
        )
    } else {
        None
    };
    if use_mq4_mq2lloyd_imatrix && expert_mq2lloyd_native && imatrix_per_expert.is_none() {
        eprintln!(
            "  imatrix: no entry for {} → falling back to uniform Lloyd",
            imatrix_lookup_name
        );
    }

    // ── SP4b bake prune (3D-stacked experts) ───────────────────────────
    // Qwen3.5-MoE (and any 3D-stacked MoE) ships routed experts as one
    // `[n_experts, ...]` tensor that this branch splits per-expert. Under
    // an active bake keep, emit ONLY kept slices, renumbered to compact
    // slots: `slots[slot] = orig_expert`. The slice offset + imatrix
    // lookup key off `orig`; the output name uses `slot`. No keep ⇒
    // identity (`slots[i] = (i, i)`), byte-identical to baseline.
    let bake_slots: Vec<(usize, usize)> = if bake_keep_active {
        let plan = reap_bake_plan.as_ref().unwrap();
        let keep = plan.keep.as_ref().unwrap();
        let l = parent_layer.unwrap_or_else(|| {
            eprintln!("reap bake: 3D-stacked expert tensor '{name}' has no parseable layer");
            std::process::exit(2);
        });
        if l >= keep.len() {
            eprintln!("reap bake: layer {l} for stacked experts '{name}' out of keep-map range");
            std::process::exit(2);
        }
        keep[l]
            .iter()
            .enumerate()
            .map(|(slot, &orig)| (slot, orig as usize))
            .collect()
    } else {
        (0..n_experts).map(|i| (i, i)).collect()
    };
    let n_out_experts = bake_slots.len();

    // ── Per-expert AWQ (Route A) ──────────────────────────────────────
    // When `--awq` is active with a GGUF imatrix, MQ4 experts get
    // activation-aware per-expert pre-scaling + a per-expert
    // `.awq_scale.weight` sidecar (length K). The runtime divides x by
    // the per-expert scale inside the indexed/grouped expert GEMM. Takes
    // priority over plain MQ4G256; the Lloyd branches above are mutually
    // exclusive (selected by their own flags), so AWQ only fires when
    // none of them claimed this expert.
    // HIPFIRE_AWQ_EXPERTS=down restricts expert AWQ to down_proj (the
    // sensitive residual-write projection + the free runtime kernel);
    // unset/=all does both gate_up and down (default).
    let awq_down_only = hipfire_config::developer_var("HIPFIRE_AWQ_EXPERTS")
        .ok()
        .as_deref()
        == Some("down");
    // HIPFIRE_AWQ_EXPERTS=none keeps DENSE AWQ (attn/lm_head) but emits
    // NO per-expert AWQ — the clean baseline for isolating the expert
    // contribution against an HIPFIRE_AWQ_EXPERTS=down treatment.
    let awq_experts_none = hipfire_config::developer_var("HIPFIRE_AWQ_EXPERTS")
        .ok()
        .as_deref()
        == Some("none");
    let expert_awq_active = AWQ_ALPHA.get().is_some()
                && !awq_experts_none
                && imatrix_gguf.is_some()
                && supports_g256
                && !(awq_down_only && base_name == "gate_up_proj")
                && !expert_mq3lloyd_native
                && !expert_mq2lloyd_native
                && !expert_mq2lloyd_roundtrip
                // GL twin of the line above. Without it, `--format mq4-mq2glexp
                // --awq --imatrix` takes the AWQ arm and the GL codec never runs,
                // so the probe silently measures AWQ-MQ4 instead of GL.
                && !expert_mq2gl_roundtrip
                && !expert_mq6
                && !expert_hfq6
                && !expert_hfq4;
    let awq_in_sum2_per_expert: Option<Vec<Vec<f32>>> = if expert_awq_active {
        imatrix_in_sum2_for_parent(
            imatrix_gguf.as_ref().unwrap(),
            &imatrix_lookup_name,
            n_experts,
        )
    } else {
        None
    };
    let awq_alpha_e = AWQ_ALPHA.get().copied().unwrap_or(0.5);
    let inner_m = inner_shape[0] as usize; // out features
    let inner_k_e = inner_shape[1] as usize; // in features (K, awq scale length)
    if expert_awq_active && awq_in_sum2_per_expert.is_none() {
        eprintln!(
            "  imatrix(awq): no entry for {} → plain MQ4G256 experts (no AWQ)",
            imatrix_lookup_name
        );
    }

    // ── Graded mixed-precision hot-set (HIPFIRE_MOE_GRADED) ───────────
    // Rank this parent's experts by imatrix routing count; the top
    // `moe_hot_frac` (DESC) get MQ6, the rest MQ2-Lloyd. Mirrors the
    // per-layer tier formula (n_hot = round(frac*n), sort DESC, take
    // top-n) but applied PER-PARENT over experts. Read-only; captured
    // by reference into the rayon closure below.
    // De-risk (Verify): the runtime wires the merged dtype-tag kernel
    // for the DOWN projection only — gate_up stays uniform MQ4. Grade
    // ONLY down_proj so the emitted file matches the wired decode path
    // (mixed MQ6/MQ2-Lloyd down, uniform MQ4 gate_up). Grading gate_up
    // would emit mixed bytes the single-dtype gate_up GEMV cannot read,
    // producing NaN logits.
    let graded_hot: Option<std::collections::HashSet<usize>> = if use_moe_graded
        && base_name == "down_proj"
    {
        let counts = imatrix_gguf
            .as_ref()
            .and_then(|g| imatrix_expert_counts_for_parent(g, &imatrix_lookup_name, n_experts));
        match counts {
            Some(c) => {
                let mut ranked: Vec<(usize, f32)> = c
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.is_finite())
                    .map(|(e, &v)| (e, v))
                    .collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let n_hot = ((n_experts as f64) * moe_hot_frac).round() as usize;
                let n_hot = n_hot.min(ranked.len());
                let set: std::collections::HashSet<usize> =
                    ranked.iter().take(n_hot).map(|&(e, _)| e).collect();
                eprintln!(
                    "  Graded {}{}: {} hot (MQ6) / {} cold (MQ2-Lloyd) experts",
                    parent,
                    base_name,
                    set.len(),
                    n_experts - set.len()
                );
                Some(set)
            }
            None => {
                eprintln!(
                    "  Graded: no imatrix .counts for {} → ALL experts MQ2-Lloyd",
                    imatrix_lookup_name
                );
                Some(std::collections::HashSet::new())
            }
        }
    } else {
        None
    };

    // Parallelize across the expert slices via rayon. Each slice
    // dequant→FWHT→quant→pack is a CPU-bound, self-contained job.
    // The outer Rayon pool size is set in main() before this runs.
    use rayon::prelude::*;
    let dtype = meta.dtype.clone();
    let parent_owned = parent.to_string();
    let inner_shape_clone = inner_shape.clone();
    let base_owned = base_name.to_string();
    // GPTQ-E8: borrow the Hessian dir into the rayon closure. Each
    // expert reads its own per-(tensor,expert) 256-block file; missing
    // -> RTN fallback. None unless --format mfp{2,3,4}e8-gptq + --hessian-dir.
    let hessian_dir_ref: Option<&Path> = if use_gptq_e8 || use_gptq_mfp3e8 || use_gptq_mfp2e8 {
        hessian_dir.as_deref()
    } else {
        None
    };
    let new_pairs: Vec<(HfqTensor, Option<HfqTensor>)> = bake_slots
        .into_par_iter()
        .map(|(slot, x)| {
            let slice_off = x * inner_bytes;
            let slice = &raw_data[slice_off..slice_off + inner_bytes];
            let f32_slice = to_f32(slice, &dtype);
            // Per-expert AWQ override (Route A): when this expert has a
            // raw in_sum2 row, pre-scale W·s and remember s for the
            // sidecar. Falls through to the format branches otherwise.
            let awq_scales: Option<Vec<f32>> = awq_in_sum2_per_expert
                .as_ref()
                .and_then(|t| t.get(x))
                .filter(|v| v.len() == inner_k_e)
                .map(|v| compute_awq_scales(v, awq_alpha_e));
            let (quantized, qt, gs) = if let (Some(tm), Some(lay)) =
                (moe_tier_map.as_ref(), parent_layer)
            {
                // N-tier TIER_MAP dispatch: look up (layer, expert) -> QuantType.
                // Fires for BOTH gate_up and down (unlike graded_hot which is down-only).
                // Falls back to uniform MQ4 for unmapped (layer,expert) pairs.
                // Uses the outer-scope signs1/signs2 captured by the rayon closure.
                match tm.get(&(lay, x)).copied().unwrap_or(QuantType::MQ4G256) {
                    QuantType::MQ6G256 => (
                        quantize_mq6g256(&f32_slice, &signs1, &signs2),
                        QuantType::MQ6G256,
                        256u32,
                    ),
                    QuantType::MQ4G256 => (
                        quantize_mq4g256(&f32_slice, &signs1, &signs2),
                        QuantType::MQ4G256,
                        256u32,
                    ),
                    QuantType::MQ3G256Lloyd => (
                        quantize_mq3g256_lloyd(&f32_slice, &signs1, &signs2),
                        QuantType::MQ3G256Lloyd,
                        256u32,
                    ),
                    QuantType::MQ2G256Lloyd => (
                        quantize_mq2g256_lloyd(&f32_slice, &signs1, &signs2),
                        QuantType::MQ2G256Lloyd,
                        256u32,
                    ),
                    // GL = GLOBAL codebook: one codebook for the whole tensor
                    // (GL_CB2/GL_CB3, passed to the kernel as scalar args) plus a
                    // per-block fp16 scale, in a two-region SoA blob. Saves the
                    // 0.1875 bpw the per-block fp16 codebook costs — measured at
                    // +1.16% KLD and -0.08% decode, i.e. size for free.
                    //
                    // NOTE these take the 2D (m, k) form like the E8 encoders, NOT
                    // the flat form the Lloyd ones use — the SoA layout needs the
                    // row count to place the scale region.
                    QuantType::MQ2G256GL => (
                        quantize_mq2g256gl(&f32_slice, inner_m, inner_k_e, &signs1, &signs2),
                        QuantType::MQ2G256GL,
                        256u32,
                    ),
                    QuantType::MQ3G256GL => (
                        quantize_mq3g256gl(&f32_slice, inner_m, inner_k_e, &signs1, &signs2),
                        QuantType::MQ3G256GL,
                        256u32,
                    ),
                    // T3-3L-E8 experiment: mfp4-E8 mid tier (4.25 bpw,
                    // MQ6-class quality) in place of MQ4. group_size 32.
                    QuantType::MFP4G32E8 => (
                        quantize_mfp4g32_e8_2d(&f32_slice, inner_m, inner_k_e, &signs1, &signs2),
                        QuantType::MFP4G32E8,
                        32u32,
                    ),
                    // [NaN-CRITICAL] mfp3-E8 cold tier: 3-bit lattice, 13 B/blk, 3.25 bpw.
                    // Drop-in for MQ3G256Lloyd (tag 3 → tag 5 in the kernel tag table).
                    QuantType::MFP3G32E8 => {
                        // GPTQ/LDLQ when a Hessian is available (graded cold
                        // tier), else RTN. Same per-tensor key + fallback
                        // accounting as the uniform mfp3e8-gptq path.
                        // Use the RAW --hessian-dir (not the format-gated
                        // hessian_dir_ref): graded base --format is mq4, so
                        // the gptq-format flags are off, but a passed Hessian
                        // still means "GPTQ the E8 cold tier".
                        let q = if let Some(hdir) = hessian_dir.as_deref() {
                            let tname = format!("{parent_owned}{x}.{base_owned}.weight");
                            let hblk = load_hessian_blocks(hdir, &tname);
                            if hblk.is_empty() {
                                GPTQ_E8_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            } else {
                                GPTQ_E8_FIRED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            quantize_mfp3g32_e8_gptq_2d(
                                &f32_slice, inner_m, inner_k_e, &signs1, &signs2, &hblk,
                            )
                        } else {
                            quantize_mfp3g32_e8_2d(&f32_slice, inner_m, inner_k_e, &signs1, &signs2)
                        };
                        (q, QuantType::MFP3G32E8, 32u32)
                    }
                    // [NaN-CRITICAL] mfp2-E8 cold tier: 2-bit lattice, 9 B/blk, 2.25 bpw.
                    // Drop-in for MQ2G256Lloyd (tag 1 → tag 6 in the kernel tag table).
                    QuantType::MFP2G32E8 => {
                        // GPTQ/LDLQ when a Hessian is available (graded cold
                        // tier), else RTN. Raw --hessian-dir (see MFP3 arm).
                        let q = if let Some(hdir) = hessian_dir.as_deref() {
                            let tname = format!("{parent_owned}{x}.{base_owned}.weight");
                            let hblk = load_hessian_blocks(hdir, &tname);
                            if hblk.is_empty() {
                                GPTQ_E8_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            } else {
                                GPTQ_E8_FIRED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            quantize_mfp2g32_e8_gptq_2d(
                                &f32_slice, inner_m, inner_k_e, &signs1, &signs2, &hblk,
                            )
                        } else {
                            quantize_mfp2g32_e8_2d(&f32_slice, inner_m, inner_k_e, &signs1, &signs2)
                        };
                        (q, QuantType::MFP2G32E8, 32u32)
                    }
                    // Any other QuantType in the map → MQ4 safe fallback
                    _ => (
                        quantize_mq4g256(&f32_slice, &signs1, &signs2),
                        QuantType::MQ4G256,
                        256u32,
                    ),
                }
            } else if let Some(hot) = graded_hot.as_ref() {
                // Graded mixed precision: hot expert -> MQ6, cold ->
                // MQ2-Lloyd. Each expert's HfqTensor carries its own qt
                // so this single parent emits MIXED dtypes; the runtime
                // builds the per-expert dtype-tag table from gpu_dtype.
                if hot.contains(&x) {
                    (
                        quantize_mq6g256(&f32_slice, &signs1, &signs2),
                        QuantType::MQ6G256,
                        256u32,
                    )
                } else {
                    (
                        quantize_mq2g256_lloyd(&f32_slice, &signs1, &signs2),
                        QuantType::MQ2G256Lloyd,
                        256u32,
                    )
                }
            } else if let Some(scales) = awq_scales.as_ref() {
                let mut scaled = f32_slice.clone();
                awq_pre_scale_weights(&mut scaled, inner_m, inner_k_e, scales);
                if down_mq6 {
                    (
                        quantize_mq6g256(&scaled, &signs1, &signs2),
                        QuantType::MQ6G256,
                        256u32,
                    )
                } else if down_mq5 || expert_mq5 {
                    (
                        quantize_mq5g256(&scaled, &signs1, &signs2),
                        QuantType::MQ5G256,
                        256u32,
                    )
                } else {
                    (
                        quantize_mq4g256(&scaled, &signs1, &signs2),
                        QuantType::MQ4G256,
                        256u32,
                    )
                }
            } else if expert_mq3lloyd_native && routed_gl {
                // GL swap: same 3-bit allocation, global codebook instead of
                // a per-block fp16 one. 3.0625 vs 3.5 bpw.
                let q = quantize_mq3g256gl(&f32_slice, inner_m, inner_k_e, &signs1, &signs2);
                (q, QuantType::MQ3G256GL, 256u32)
            } else if expert_mq3lloyd_native {
                let q = quantize_mq3g256_lloyd(&f32_slice, &signs1, &signs2);
                (q, QuantType::MQ3G256Lloyd, 256u32)
            } else if expert_mq2lloyd_native && routed_gl {
                // GL swap: 2.0625 vs 2.25 bpw. NOTE the imatrix-weighted and
                // GPTQ arms below are DELIBERATELY not mirrored here — the
                // weighted fit is provably inert after the FWHT (every
                // R[i][j]^2 = 1/256, so a rotated diagonal importance vector
                // is constant), so plain Lloyd is the honest baseline and
                // there is nothing to lose by taking it.
                let q = quantize_mq2g256gl(&f32_slice, inner_m, inner_k_e, &signs1, &signs2);
                (q, QuantType::MQ2G256GL, 256u32)
            } else if expert_mq2lloyd_native {
                // Native MQ2-Lloyd: ship qt=19 bytes (72 B / 256 weights).
                // Selection order:
                //   1. GPTQ-Lloyd (sequential error feedback) — Lever 2
                //      path, requires imatrix.
                //   2. Imatrix-weighted Lloyd — standard Phase 3b path.
                //   3. Uniform Lloyd — fallback when no imatrix available.
                let q = match imatrix_per_expert.as_ref() {
                    Some(table)
                        if x < table.len() && !table[x].is_empty() && use_gptq_for_gate_up =>
                    {
                        quantize_mq2g256_lloyd_gptq(&f32_slice, &table[x], &signs1, &signs2)
                    }
                    Some(table) if x < table.len() && !table[x].is_empty() => {
                        quantize_mq2g256_lloyd_weighted(&f32_slice, &table[x], &signs1, &signs2)
                    }
                    _ => quantize_mq2g256_lloyd(&f32_slice, &signs1, &signs2),
                };
                (q, QuantType::MQ2G256Lloyd, 256u32)
            } else if expert_mq2lloyd_roundtrip {
                // MQ2-Lloyd → F32 → HFQ4 round-trip. The MQ2 step injects
                // the 2-bit Lloyd-codebook noise; the HFQ4 step re-packs
                // for runtime. Final on-disk format is HFQ4G256, no
                // engine changes required.
                let mq2_bytes = quantize_mq2g256_lloyd(&f32_slice, &signs1, &signs2);
                let dequant =
                    dequantize_mq2g256_lloyd_to_f32(&mq2_bytes, f32_slice.len(), &signs1, &signs2);
                let q = quantize_hfq4g256(&dequant);
                (q, QuantType::HFQ4G256, 256u32)
            } else if expert_mq2gl_roundtrip {
                // MQ2-GL → F32 → HFQ4 round-trip. Identical to the arm
                // above except the 2-bit step uses ONE tensor-global
                // codebook + per-block fp16 scale rather than a
                // per-block fitted codebook. Same HFQ4G256 output, so
                // the two arms differ ONLY in the injected codec noise.
                let dequant = mq2g256gl_roundtrip_f32(&f32_slice, &signs1, &signs2);
                let q = quantize_hfq4g256(&dequant);
                (q, QuantType::HFQ4G256, 256u32)
            } else if expert_mq5 || down_mq5 {
                let q = quantize_mq5g256(&f32_slice, &signs1, &signs2);
                (q, QuantType::MQ5G256, 256u32)
            } else if expert_mq6 || down_mq6 {
                let q = quantize_mq6g256(&f32_slice, &signs1, &signs2);
                (q, QuantType::MQ6G256, 256u32)
            } else if expert_hfq6 {
                let q = quantize_hfq6g256(&f32_slice);
                (q, QuantType::HFQ6G256, 256u32)
            } else if expert_hfq4 {
                let q = quantize_hfq4g256(&f32_slice);
                (q, QuantType::HFQ4G256, 256u32)
            } else if use_mfp4 && supports_g256 {
                let q = quantize_mfp4g32_2d(&f32_slice, inner_m, inner_k_e, &signs1, &signs2);
                (q, QuantType::MFP4G32, 32u32)
            } else if use_mfp4p && supports_g256 {
                let q = quantize_mfp4g32_p_2d(&f32_slice, inner_m, inner_k_e, &signs1, &signs2);
                (q, QuantType::MFP4G32P, 32u32)
            } else if use_mfp4e8 && supports_g256 {
                let q = if let Some(hdir) = hessian_dir_ref {
                    let tname = format!("{parent_owned}{x}.{base_owned}.weight");
                    let hblk = load_hessian_blocks(hdir, &tname);
                    if hblk.is_empty() {
                        GPTQ_E8_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    } else {
                        GPTQ_E8_FIRED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    quantize_mfp4g32_e8_gptq_2d(
                        &f32_slice, inner_m, inner_k_e, &signs1, &signs2, &hblk,
                    )
                } else {
                    quantize_mfp4g32_e8_2d(&f32_slice, inner_m, inner_k_e, &signs1, &signs2)
                };
                (q, QuantType::MFP4G32E8, 32u32)
            } else if use_mfp3e8_gptq_fmt && supports_g256 {
                // mfp3e8-gptq: 3-bit E8 with LDLQ. Falls back to RTN if no Hessian.
                let q = if let Some(hdir) = hessian_dir_ref {
                    let tname = format!("{parent_owned}{x}.{base_owned}.weight");
                    let hblk = load_hessian_blocks(hdir, &tname);
                    if hblk.is_empty() {
                        GPTQ_E8_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    } else {
                        GPTQ_E8_FIRED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    quantize_mfp3g32_e8_gptq_2d(
                        &f32_slice, inner_m, inner_k_e, &signs1, &signs2, &hblk,
                    )
                } else {
                    quantize_mfp3g32_e8_2d(&f32_slice, inner_m, inner_k_e, &signs1, &signs2)
                };
                (q, QuantType::MFP3G32E8, 32u32)
            } else if use_mfp2e8_gptq_fmt && supports_g256 {
                // mfp2e8-gptq: 2-bit E8 with LDLQ. Falls back to RTN if no Hessian.
                let q = if let Some(hdir) = hessian_dir_ref {
                    let tname = format!("{parent_owned}{x}.{base_owned}.weight");
                    let hblk = load_hessian_blocks(hdir, &tname);
                    if hblk.is_empty() {
                        GPTQ_E8_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    } else {
                        GPTQ_E8_FIRED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    quantize_mfp2g32_e8_gptq_2d(
                        &f32_slice, inner_m, inner_k_e, &signs1, &signs2, &hblk,
                    )
                } else {
                    quantize_mfp2g32_e8_2d(&f32_slice, inner_m, inner_k_e, &signs1, &signs2)
                };
                (q, QuantType::MFP2G32E8, 32u32)
            } else if use_mfp4e8soa && supports_g256 {
                let q =
                    quantize_mfp4g32_e8_soa_2d(&f32_slice, inner_m, inner_k_e, &signs1, &signs2);
                (q, QuantType::MFP4G32E8SOA, 32u32)
            } else if supports_g256 {
                let q = quantize_mq4g256(&f32_slice, &signs1, &signs2);
                (q, QuantType::MQ4G256, 256u32)
            } else {
                let q = quantize_hfq4g128(&f32_slice);
                (q, QuantType::HFQ4G128, 128u32)
            };
            let weight = HfqTensor {
                name: format!("{parent_owned}{slot}.{base_owned}.weight"),
                quant_type: qt,
                shape: inner_shape_clone.clone(),
                group_size: gs,
                data: quantized,
                spilled_len: 0,
            };
            let sidecar = awq_scales.map(|s| HfqTensor {
                name: format!("{parent_owned}{slot}.{base_owned}.awq_scale.weight"),
                quant_type: QuantType::F16,
                shape: vec![inner_k_e as u32],
                group_size: 0,
                data: awq_scales_to_f16_bytes(&s),
                spilled_len: 0,
            });
            (weight, sidecar)
        })
        .collect();
    // Flatten weight+sidecar pairs; each AWQ expert emits two tensors.
    let n_awq = new_pairs.iter().filter(|(_, s)| s.is_some()).count();
    let mut new_tensors: Vec<HfqTensor> = Vec::with_capacity(new_pairs.len() + n_awq);
    for (w, s) in new_pairs {
        new_tensors.push(w);
        if let Some(sc) = s {
            new_tensors.push(sc);
        }
    }
    *quantized_params += inner_n as u64 * n_out_experts as u64;
    // Single eprintln to summarize the whole expert sweep.
    let label = if moe_tier_map.is_some() && parent_layer.is_some() {
        "TierMap"
    } else if graded_hot.is_some() {
        "Graded(MQ6/MQ2L)"
    } else if expert_awq_active && awq_in_sum2_per_expert.is_some() {
        if down_mq6 {
            "MQ6G256+AWQ"
        } else if down_mq5 || expert_mq5 {
            "MQ5G256+AWQ"
        } else {
            "MQ4G256+AWQ"
        }
    } else if expert_mq3lloyd_native {
        "MQ3G256L"
    } else if expert_mq2lloyd_native {
        if imatrix_per_expert.is_some() {
            "MQ2L+imatrix"
        } else {
            "MQ2G256L"
        }
    } else if expert_mq2lloyd_roundtrip {
        "MQ2L→HFQ4"
    } else if expert_mq5 || down_mq5 {
        "MQ5G256"
    } else if expert_mq6 || down_mq6 {
        "MQ6G256"
    } else if expert_hfq6 {
        "HFQ6G256"
    } else if expert_hfq4 {
        "HFQ4G256"
    } else if use_mfp4 && supports_g256 {
        "MFP4G32"
    } else if use_mfp4p && supports_g256 {
        "MFP4G32P"
    } else if use_mfp4e8 && supports_g256 {
        if use_gptq_e8 {
            "MFP4E8-GPTQ"
        } else {
            "MFP4G32E8"
        }
    } else if use_mfp3e8_gptq_fmt && supports_g256 {
        if use_gptq_mfp3e8 {
            "MFP3E8-GPTQ"
        } else {
            "MFP3G32E8"
        }
    } else if use_mfp2e8_gptq_fmt && supports_g256 {
        if use_gptq_mfp2e8 {
            "MFP2E8-GPTQ"
        } else {
            "MFP2G32E8"
        }
    } else if use_mfp4e8soa && supports_g256 {
        "MFP4G32E8SOA"
    } else if supports_g256 {
        "MQ4G256"
    } else {
        "HFQ4G128"
    };
    let bytes_per = new_tensors.first().map(|t| t.data.len()).unwrap_or(0);
    eprintln!(
                "  {label:>8}: {parent_owned}{{0..{n_out_experts}}}.{base_owned}.weight {:?} (×{n_out_experts} experts of {n_experts} || {:.1} KB/expert, parallel)",
                inner_shape,
                bytes_per as f64 / 1024.0
            );
    hfq_tensors.append(&mut new_tensors);
    // Drop source pages and spill quantized data after each expert batch.
    st_files[file_idx].drop_tensor_pages(name);
    if let Some(s) = spill.as_mut() {
        maybe_spill(hfq_tensors, s, 2 * 1024 * 1024 * 1024); // 2 GB threshold
    }
    true
}

fn handle_main_quant(
    ctx: &PerTensorCtx,
    meta: &TensorMeta,
    raw_data: &[u8],
    flags: &MainQuantFlags,
    outer: &MainQuantOuter,
    state: &mut MainQuantState,
    fp8_scale_for: &HashMap<String, (usize, String)>,
    st_files: &[SafetensorsFile],
) {
    let name = ctx.name;
    let n_elements = ctx.n_elements;
    let is_vision = ctx.is_vision;
    let arch_id = ctx.arch_id;
    let vision_quant = flags.vision_quant.as_str();
    let is_gemma4_family = flags.is_gemma4_family;
    let q8_conv1d_default = flags.q8_conv1d_default;
    if should_quantize(name) && n_elements >= 32 {
        let f32_data =
            tensor_to_f32_with_optional_fp8_scale(name, raw_data, meta, &fp8_scale_for, &st_files);
        *state.quantized_params += n_elements as u64;

        let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();

        // Q8HFQ path: split-metadata per-row layout (needs M and K)
        // Exclude embeddings — they use a lookup kernel, not GEMV
        if flags.use_q8hfq && meta.shape.len() == 2 && !name.contains("embed_tokens") {
            let m = meta.shape[0];
            let k = meta.shape[1];
            let (quantized, row_stride) = quantize_q8hfq(&f32_data, m, k);

            // Compute quantization error for Q8HFQ
            let n_groups = k / 32;
            let scales_bytes = n_groups * 2;
            for row in 0..m {
                let row_off = row * row_stride;
                for g in 0..n_groups {
                    let scale = f16_to_f32(u16::from_le_bytes([
                        quantized[row_off + g * 2],
                        quantized[row_off + g * 2 + 1],
                    ]));
                    for i in 0..32 {
                        let qval = quantized[row_off + scales_bytes + g * 32 + i] as i8;
                        let dequant = scale * qval as f32;
                        let orig_idx = row * k + g * 32 + i;
                        let err = (dequant - f32_data[orig_idx]).abs();
                        *state.total_quant_error += err as f64;
                        *state.max_quant_error = (*state.max_quant_error).max(err);
                    }
                    *state._n_quant_groups += 1;
                }
            }

            eprintln!(
                "  {:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB, stride={})",
                "Q8_HFQ",
                name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0,
                quantized.len() as f64 / 1024.0,
                row_stride
            );

            state.hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::Q8HFQ,
                shape,
                group_size: 32,
                data: quantized,
                spilled_len: 0,
            });
        } else {
            // ── K-map override ──────────────────────────────────────────────
            let kmap_level = outer.kmap.get(name).copied().unwrap_or(QuantLevel::Base);

            // AWQ sidecar scales for this tensor — populated only inside the
            // MQ4G256 arm when --awq is enabled and an imatrix entry exists
            // for this tensor's ggml-translated name. After the main tensor
            // push, we emit an `<name>.awq_scale` 1D F16 sidecar tensor so
            // the runtime can apply `x / s` before the rotation kernel at
            // inference time.
            let mut awq_sidecar_scales: Option<Vec<f32>> = None;

            let (quantized, qt, gs, label) = if flags.q8_conv1d_default && is_conv1d_tensor(name) {
                // DeltaNet conv1d defaults to Q8 (see --no-q8-conv1d to disable).
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if kmap_level == QuantLevel::Q8 {
                // K-map says Q8 (embed, lm_head, router)
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if kmap_level == QuantLevel::F16 {
                // K-map says F16 (should not normally reach here — should_quantize filters first)
                let f16_bytes: Vec<u8> = f32_data
                    .iter()
                    .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                    .collect();
                (f16_bytes, QuantType::F16, 0u32, "F16")
            } else if kmap_level == QuantLevel::Promote6 {
                // K-map says promote to 6-bit
                let k_dim = if meta.shape.len() == 2 {
                    meta.shape[1]
                } else {
                    n_elements
                };
                if (flags.use_mq4g256
                    || flags.use_mq4v2
                    || flags.use_mq4c
                    || flags.use_mq4_mq6exp
                    || flags.use_mq4_mq2lloydexp
                    || flags.use_mq4_mq2glexp
                    || flags.use_mq4_mq2lloyd_native
                    || flags.use_mq4_mq2lloyd_kmap
                    || flags.use_mq4_mq2lloyd_imatrix
                    || flags.use_mq4_mq3lloyd_kmap
                    || flags.use_mq4_mqlloyd_tiered
                    || flags.use_mq4_mqlloyd_antirez
                    || flags.use_mq4_mqlloyd_antirez_gptq
                    || flags.use_mq4_mq2lloyd_gptq_all
                    || flags.use_mq3g256
                    || flags.use_mq2g256
                    || flags.use_mq2g256_lloyd
                    || flags.use_mq3g256_lloyd)
                    && k_dim % 256 == 0
                {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                } else if (flags.use_hfq4g256
                    || flags.use_hfq3g256
                    || flags.use_hfq3g128
                    || flags.use_hfq2g256
                    || flags.use_hfq2g128)
                    && k_dim % 256 == 0
                {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                } else if flags.use_mq6g256 && k_dim % 256 == 0 {
                    // Already 6-bit MQ — no-op promotion
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                } else if flags.use_hfq6 && k_dim % 256 == 0 {
                    // Already 6-bit HFQ — no-op promotion
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                } else {
                    // Non-256-aligned fallback: Q8
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                }
            } else if let QuantLevel::Override(override_fmt) = kmap_level {
                // K-map says override (today: lm_head when --lm-head-format set).
                // Dispatch on the carried format. For MQ4 with AWQ enabled,
                // apply AWQ pre-scaling + emit a sidecar so the runtime
                // (once the CUDA-branch AWQ-aware lm_head dispatch lands)
                // sees scaled bytes and inverse-divides correctly. For any
                // other format, plain quantize (the AWQ wiring outside MQ4
                // is a follow-up).
                let k_dim = if meta.shape.len() == 2 {
                    meta.shape[1]
                } else {
                    n_elements
                };
                if k_dim % 256 == 0 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    // ── Gemma4 (arch 13/22): embed/lm_head MUST NOT reach AWQ ──
                    // They are always routed to Q8 by the K-map before this
                    // branch, so they cannot arrive here. Assert the invariant
                    // rather than leave it implicit: Gemma4's tied embed/lm_head
                    // carries an implicit sqrt(d_model) scaling with no RMSNorm
                    // anchor on the embedding dimension, which makes AWQ's
                    // imatrix-saliency ratio meaningless and the per-channel
                    // pre-scale actively harmful.
                    debug_assert!(
                        !(flags.is_gemma4_family
                            && (name.contains("embed_tokens") || name.contains("lm_head"))),
                        "gemma4 embed/lm_head reached the MQ4 AWQ path — the outer.kmap Q8 \
                             guard should have prevented this (arch {} tensor {})",
                        flags.arch_id,
                        name
                    );
                    match override_fmt {
                        GgufFormat::Mq4 => {
                            // Inline AWQ + MQ4 dance (mirrors the Base MQ4 arm).
                            let q = if let (Some(alpha), Some(im_weights)) =
                                (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                            {
                                if awq_eligible(name) {
                                    let scales = compute_awq_scales(im_weights, alpha);
                                    awq_sidecar_scales = Some(scales.clone());
                                    let m_dim = meta.shape[0];
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                    quantize_mq4g256(&scaled, &signs1, &signs2)
                                } else {
                                    quantize_mq4g256(&f32_data, &signs1, &signs2)
                                }
                            } else {
                                quantize_mq4g256(&f32_data, &signs1, &signs2)
                            };
                            (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                        }
                        GgufFormat::Mq4V2 => {
                            let q = if let (Some(alpha), Some(im_weights)) =
                                (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                            {
                                if awq_eligible(name) {
                                    let scales = compute_awq_scales(im_weights, alpha);
                                    awq_sidecar_scales = Some(scales.clone());
                                    let m_dim = meta.shape[0];
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                    quantize_mq4g256v2(&scaled, m_dim, k_dim, &signs1, &signs2)
                                } else {
                                    let m = meta.shape[0];
                                    let k = k_dim;
                                    quantize_mq4g256v2(&f32_data, m, k, &signs1, &signs2)
                                }
                            } else {
                                let m = meta.shape[0];
                                let k = k_dim;
                                quantize_mq4g256v2(&f32_data, m, k, &signs1, &signs2)
                            };
                            (q, QuantType::MQ4G256V2, 256u32, "MQ4G256V2")
                        }
                        GgufFormat::Mq4C => {
                            let q = if let (Some(alpha), Some(im_weights)) =
                                (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                            {
                                if awq_eligible(name) {
                                    let scales = compute_awq_scales(im_weights, alpha);
                                    awq_sidecar_scales = Some(scales.clone());
                                    let m_dim = meta.shape[0];
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                    quantize_mq4cg256(&scaled, m_dim, k_dim, &signs1, &signs2)
                                } else {
                                    let m = meta.shape[0];
                                    let k = k_dim;
                                    quantize_mq4cg256(&f32_data, m, k, &signs1, &signs2)
                                }
                            } else {
                                let m = meta.shape[0];
                                let k = k_dim;
                                quantize_mq4cg256(&f32_data, m, k, &signs1, &signs2)
                            };
                            (q, QuantType::MQ4CG256, 256u32, "MQ4CG256")
                        }
                        GgufFormat::Mq5 => {
                            // MQ5 + AWQ on lm_head: MQ5G256 is in
                            // DType::supports_awq_sidecar, so the runtime applies the
                            // inverse divide via rotate_x_mq. Same AWQ inline dance as MQ4.
                            let q = if let (Some(alpha), Some(im_weights)) =
                                (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                            {
                                if awq_eligible(name) {
                                    let scales = compute_awq_scales(im_weights, alpha);
                                    awq_sidecar_scales = Some(scales.clone());
                                    let m_dim = meta.shape[0];
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                    quantize_mq5g256(&scaled, &signs1, &signs2)
                                } else {
                                    quantize_mq5g256(&f32_data, &signs1, &signs2)
                                }
                            } else {
                                quantize_mq5g256(&f32_data, &signs1, &signs2)
                            };
                            (q, QuantType::MQ5G256, 256u32, "MQ5G256")
                        }
                        GgufFormat::Mq6 => {
                            let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                            (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                        }
                        GgufFormat::Mq6V2 => {
                            let m = meta.shape[0];
                            let k = k_dim;
                            let q = quantize_mq6g256v2(&f32_data, m, k, &signs1, &signs2);
                            (q, QuantType::MQ6G256V2, 256u32, "MQ6G256V2")
                        }
                        GgufFormat::Mq5V2 => {
                            let m = meta.shape[0];
                            let k = k_dim;
                            let q = quantize_mq5g256v2(&f32_data, m, k, &signs1, &signs2);
                            (q, QuantType::MQ5G256V2, 256u32, "MQ5G256V2")
                        }
                        GgufFormat::Mq3V2 => {
                            let m = meta.shape[0];
                            let k = k_dim;
                            let q = quantize_mq3g256v2(&f32_data, m, k, &signs1, &signs2);
                            (q, QuantType::MQ3G256V2, 256u32, "MQ3G256V2")
                        }
                        GgufFormat::Mq2V2 => {
                            let m = meta.shape[0];
                            let k = k_dim;
                            let q = quantize_mq2g256v2(&f32_data, m, k, &signs1, &signs2);
                            (q, QuantType::MQ2G256V2, 256u32, "MQ2G256V2")
                        }
                        GgufFormat::Mq3 => {
                            // DType::supports_awq_sidecar(MQ3G256)=true (per the
                            // fix/lm-head-awq-runtime branch). Wire the same AWQ
                            // inline-quantize dance as the MQ4 arm.
                            let q = if let (Some(alpha), Some(im_weights)) =
                                (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                            {
                                if awq_eligible(name) {
                                    let scales = compute_awq_scales(im_weights, alpha);
                                    awq_sidecar_scales = Some(scales.clone());
                                    let m_dim = meta.shape[0];
                                    let mut scaled = f32_data.clone();
                                    awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                    quantize_mq3g256(&scaled, &signs1, &signs2)
                                } else {
                                    quantize_mq3g256(&f32_data, &signs1, &signs2)
                                }
                            } else {
                                quantize_mq3g256(&f32_data, &signs1, &signs2)
                            };
                            (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                        }
                        GgufFormat::Hfq4 => {
                            let q = quantize_hfq4g256(&f32_data);
                            (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                        }
                        GgufFormat::Hfq6 => {
                            let q = quantize_hfq6g256(&f32_data);
                            (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                        }
                        // Other Override targets: not yet wired with AWQ;
                        // emit plain quantization. Used in Phase 0 sweeps
                        // for non-AWQ lm_head experiments.
                        GgufFormat::Mq2 => {
                            let q = quantize_mq2g256(&f32_data, &signs1, &signs2);
                            (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                        }
                        GgufFormat::Mq2Lloyd => {
                            let q = quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2);
                            (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                        }
                        GgufFormat::Mq3Lloyd => {
                            let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2);
                            (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                        }
                        GgufFormat::Mq4Lloyd => {
                            let q = quantize_mq4g256_lloyd(&f32_data, &signs1, &signs2);
                            (q, QuantType::MQ4G256Lloyd, 256u32, "MQ4G256Lloyd")
                        }
                        GgufFormat::Mfp4 => {
                            let m = if meta.shape.len() == 2 {
                                meta.shape[0]
                            } else {
                                1
                            };
                            let q = quantize_mfp4g32_2d(&f32_data, m, k_dim, &signs1, &signs2);
                            (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                        }
                        GgufFormat::Mfp4Lloyd => {
                            let m = if meta.shape.len() == 2 {
                                meta.shape[0]
                            } else {
                                1
                            };
                            let q =
                                quantize_mfp4g32_lloyd_2d(&f32_data, m, k_dim, &signs1, &signs2);
                            (q, QuantType::MFP4G32Lloyd, 32u32, "MFP4G32Lloyd")
                        }
                        GgufFormat::Mfp4P => {
                            let m = if meta.shape.len() == 2 {
                                meta.shape[0]
                            } else {
                                1
                            };
                            let q = quantize_mfp4g32_p_2d(&f32_data, m, k_dim, &signs1, &signs2);
                            (q, QuantType::MFP4G32P, 32u32, "MFP4G32P")
                        }
                        GgufFormat::Mfp4E8 => {
                            let m = if meta.shape.len() == 2 {
                                meta.shape[0]
                            } else {
                                1
                            };
                            let q = quantize_mfp4g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2);
                            (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                        }
                        GgufFormat::Mfp4E8Soa => {
                            let m = if meta.shape.len() == 2 {
                                meta.shape[0]
                            } else {
                                1
                            };
                            let q =
                                quantize_mfp4g32_e8_soa_2d(&f32_data, m, k_dim, &signs1, &signs2);
                            (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                        }
                        GgufFormat::Mfp3E8 => {
                            let m = if meta.shape.len() == 2 {
                                meta.shape[0]
                            } else {
                                1
                            };
                            let q = quantize_mfp3g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2);
                            (q, QuantType::MFP3G32E8, 32u32, "MFP3G32E8")
                        }
                        GgufFormat::Mfp2E8 => {
                            let m = if meta.shape.len() == 2 {
                                meta.shape[0]
                            } else {
                                1
                            };
                            let q = quantize_mfp2g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2);
                            (q, QuantType::MFP2G32E8, 32u32, "MFP2G32E8")
                        }
                        GgufFormat::Hfp4 => {
                            let m = if meta.shape.len() == 2 {
                                meta.shape[0]
                            } else {
                                1
                            };
                            let q = quantize_hfp4g32_2d(&f32_data, m, k_dim);
                            (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                        }
                        GgufFormat::Ternary => {
                            // Terminal low-bit: direct TQ2G128 — scale-only ternary g128, 34 B/blk.
                            // No rotation (RotationPlan::None) and no AWQ fold-out.
                            let q = quantize_tq2g128(&f32_data);
                            (q, QuantType::TQ2G128, 128u32, "TQ2G128")
                        }
                        GgufFormat::Binary => {
                            // Terminal low-bit: direct BQ1G128 — scale-only binary g128, 18 B/blk.
                            let q = quantize_bq1g128(&f32_data);
                            (q, QuantType::BQ1G128, 128u32, "BQ1G128")
                        }
                    }
                } else {
                    // Non-256-aligned override target: Q8 fallback.
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                }
            } else {
                // QuantLevel::Base — product-tier lift takes precedence over base format.
                // XT: embed only, Base: +lm_head, Pro: +ssm_out. Embed/conv1d are already
                // handled above via is_embed / q8_conv1d_default, but tier also declares them.
                // Fixed-tier per-class dtype overrides (e.g. lm_head:mq6v2) route through real encoder.
                // Lifted tensors have AWQ sidecars removed.
                // Choose quant format per tensor
                let this_q8 = if flags.use_q4k_all {
                    false // everything Q4_K
                } else if flags.use_q4k_q8embed {
                    name.contains("embed") || name.contains("lm_head") // only embed/output Q8
                } else if flags.use_mixed || flags.use_fast {
                    is_q8_tensor(name)
                } else {
                    flags.use_q8 || flags.use_q8hfq // 1D Q8HFQ tensors fall back to Q8F16
                };
                let this_q4as8 = flags.use_fast && !this_q8; // FFN tensors in q8-fast mode
                let this_q4k = flags.use_q4k_all || flags.use_q4k_q8embed || flags.use_mixed;

                // Embeddings stored as Q8 in HFQ4 mode — Q4 is too lossy for
                // large-dim models (9B: dim=4096, values ~0.016, Q4 step ~0.007)
                let is_embed = name.contains("embed_tokens");

                let tier_lift = flags
                    .product_tier
                    .is_some_and(|t| q8_class_of(name).is_some_and(|cls| t.lifts(cls)));
                if tier_lift {
                    awq_sidecar_scales = None;
                    if let Some(dt) = fixed_tier_dtype_for(name) {
                        let m = meta.shape[0];
                        let k = meta.shape[1];
                        if k % 256 != 0
                            && matches!(dt, "mq2v2" | "mq3v2" | "mq4v2" | "mq5v2" | "mq6v2")
                        {
                            eprintln!(
                                "error: fixed-tier dtype {dt} requires K%256==0 for {name} (K={k})"
                            );
                            std::process::exit(2);
                        }
                        let s1 = gen_fwht_signs(42, 256);
                        let s2 = gen_fwht_signs(1042, 256);
                        match dt {
                            "mq6v2" => {
                                let q = quantize_mq6g256v2(&f32_data, m, k, &s1, &s2);
                                (q, QuantType::MQ6G256V2, 256u32, "MQ6G256V2")
                            }
                            "mq5v2" => {
                                let q = quantize_mq5g256v2(&f32_data, m, k, &s1, &s2);
                                (q, QuantType::MQ5G256V2, 256u32, "MQ5G256V2")
                            }
                            "mq4v2" => {
                                let q = quantize_mq4g256v2(&f32_data, m, k, &s1, &s2);
                                (q, QuantType::MQ4G256V2, 256u32, "MQ4G256V2")
                            }
                            "mq3v2" => {
                                let q = quantize_mq3g256v2(&f32_data, m, k, &s1, &s2);
                                (q, QuantType::MQ3G256V2, 256u32, "MQ3G256V2")
                            }
                            "mq2v2" => {
                                let q = quantize_mq2g256v2(&f32_data, m, k, &s1, &s2);
                                (q, QuantType::MQ2G256V2, 256u32, "MQ2G256V2")
                            }
                            "mq4" => {
                                let q = quantize_mq4g256(&f32_data, &s1, &s2);
                                (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                            }
                            "mq3l" => {
                                let q = quantize_mq3g256_lloyd(&f32_data, &s1, &s2);
                                (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                            }
                            "mfp4e8" => {
                                let q = quantize_mfp4g32_e8_2d(&f32_data, m, k, &s1, &s2);
                                (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                            }
                            "mfp4e8soa" => {
                                let q = quantize_mfp4g32_e8_soa_2d(&f32_data, m, k, &s1, &s2);
                                (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                            }
                            "q8" => {
                                let q = quantize_q8f16(&f32_data);
                                (q, QuantType::Q8F16, 32u32, "Q8_F16")
                            }
                            _ => {
                                let q = quantize_mq4g256(&f32_data, &s1, &s2);
                                (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                            }
                        }
                    } else {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    }
                } else if flags.use_hfq_mixed {
                    // hfq-mixed: Q8 for attention, HFQ4 for FFN (fits 9B in 8GB VRAM)
                    let is_ffn = name.contains("mlp.") || name.contains("ffn");
                    if !is_ffn {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else {
                        let k_dim = if meta.shape.len() == 2 {
                            meta.shape[1]
                        } else {
                            n_elements
                        };
                        if k_dim % 256 == 0 {
                            let q = quantize_hfq4g256(&f32_data);
                            (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                        } else {
                            let q = quantize_hfq4g128(&f32_data);
                            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                        }
                    }
                } else if flags.use_hfq6 {
                    // HFQ6-G256: all weights 6-bit, embeddings Q8
                    if is_embed {
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    } else {
                        let q = quantize_hfq6g256(&f32_data);
                        (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                    }
                } else if (flags.use_hfq2g256 || flags.use_hfq2g128) && is_embed {
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if flags.use_hfq2g128 {
                    let q = quantize_hfq2g128(&f32_data);
                    (q, QuantType::HFQ2G128, 128u32, "HFQ2G128")
                } else if flags.use_hfq2g256 {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let q = quantize_hfq2g256(&f32_data);
                        (q, QuantType::HFQ2G256, 256u32, "HFQ2G256")
                    } else {
                        // Fallback to HFQ4 for non-256-aligned
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if flags.use_mq8g256 && is_embed {
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if flags.use_mq8g256 {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let q = quantize_mq8g256(&f32_data, &signs1, &signs2);
                        (q, QuantType::MQ8G256, 256u32, "MQ8G256")
                    } else {
                        // Fallback to Q8 for non-256-aligned
                        let q = quantize_q8f16(&f32_data);
                        (q, QuantType::Q8F16, 32u32, "Q8_F16")
                    }
                } else if flags.q8_router && is_q8_tensor(name) {
                    // Fixed tier (attention / lm_head / embed / router) held above
                    // --format. Default Q8F16; `HIPFIRE_FIXED_TIER=<class>:<dtype>`
                    // overrides per class — the bit-allocation lever.
                    //
                    // Why this matters: the fixed tier is 66% of per-token decode
                    // bytes on a3b, and dropping the WHOLE tier Q8 -> MQ4 measured
                    // +35.2% KLD (0.1742 -> 0.2356) for 1.75x speed. MFP4G32E8SOA
                    // is the interesting middle: ~same bytes as MQ4 (4.3 vs 4.25
                    // bpw) but E8 lattice VQ instead of scalar affine, and it is
                    // already dispatchable for lm_head (plain GEMV, gemv_table.rs
                    // registers it Plain + Prerotated). NOTE it is NOT a whole-tier
                    // replacement: FusedQkvza's E8 arm is gfx1151-decode-only and
                    // there is no E8 residual GEMV for o_proj.
                    match fixed_tier_dtype_for(name) {
                        Some(dt) => {
                            let s1 = gen_fwht_signs(42, 256);
                            let s2 = gen_fwht_signs(1042, 256);
                            let m = meta.shape[0];
                            let k = meta.shape[1];
                            if k % 256 != 0
                                && matches!(dt, "mq2v2" | "mq3v2" | "mq4v2" | "mq5v2" | "mq6v2")
                            {
                                eprintln!("error: fixed-tier dtype {dt} requires K%256==0 for {name} (K={k})");
                                std::process::exit(2);
                            }
                            match dt {
                                "mfp4e8soa" => {
                                    let q = quantize_mfp4g32_e8_soa_2d(&f32_data, m, k, &s1, &s2);
                                    (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                                }
                                "mfp4e8" => {
                                    let q = quantize_mfp4g32_e8_2d(&f32_data, m, k, &s1, &s2);
                                    (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                                }
                                "mq3l" => {
                                    let q = quantize_mq3g256_lloyd(&f32_data, &s1, &s2);
                                    (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256L")
                                }
                                "mq6v2" => {
                                    let q = quantize_mq6g256v2(&f32_data, m, k, &s1, &s2);
                                    (q, QuantType::MQ6G256V2, 256u32, "MQ6G256V2")
                                }
                                "mq5v2" => {
                                    let q = quantize_mq5g256v2(&f32_data, m, k, &s1, &s2);
                                    (q, QuantType::MQ5G256V2, 256u32, "MQ5G256V2")
                                }
                                "mq4v2" => {
                                    let q = quantize_mq4g256v2(&f32_data, m, k, &s1, &s2);
                                    (q, QuantType::MQ4G256V2, 256u32, "MQ4G256V2")
                                }
                                "mq3v2" => {
                                    let q = quantize_mq3g256v2(&f32_data, m, k, &s1, &s2);
                                    (q, QuantType::MQ3G256V2, 256u32, "MQ3G256V2")
                                }
                                "mq2v2" => {
                                    let q = quantize_mq2g256v2(&f32_data, m, k, &s1, &s2);
                                    (q, QuantType::MQ2G256V2, 256u32, "MQ2G256V2")
                                }
                                "q8" => {
                                    let q = quantize_q8f16(&f32_data);
                                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                                }
                                _ => {
                                    let q = quantize_mq4g256(&f32_data, &s1, &s2);
                                    (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                                }
                            }
                        }
                        None => {
                            let q = quantize_q8f16(&f32_data);
                            (q, QuantType::Q8F16, 32u32, "Q8_F16")
                        }
                    }
                } else if (flags.use_mq4g256
                    || flags.use_mq4v2
                    || flags.use_mq4c
                    || flags.use_mq4_mq6exp
                    || flags.use_mq4_mq2lloydexp
                    || flags.use_mq4_mq2glexp
                    || flags.use_mq4_mq2lloyd_native
                    || flags.use_mq4_mq2lloyd_kmap
                    || flags.use_mq4_mq2lloyd_imatrix
                    || flags.use_mq4_mq3lloyd_kmap
                    || flags.use_mq4_mqlloyd_tiered
                    || flags.use_mq4_mqlloyd_antirez
                    || flags.use_mq4_mqlloyd_antirez_gptq
                    || flags.use_mq4_mq2lloyd_gptq_all)
                    && is_embed
                {
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if flags.use_mq4g256
                    || flags.use_mq4_mq6exp
                    || flags.use_mq4_mq2lloydexp
                    || flags.use_mq4_mq2glexp
                    || flags.use_mq4_mq2lloyd_native
                    || flags.use_mq4_mq2lloyd_kmap
                    || flags.use_mq4_mq2lloyd_imatrix
                    || flags.use_mq4_mq3lloyd_kmap
                    || flags.use_mq4_mqlloyd_tiered
                    || flags.use_mq4_mqlloyd_antirez
                    || flags.use_mq4_mqlloyd_antirez_gptq
                    || flags.use_mq4_mq2lloyd_gptq_all
                {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        // Phase A Stage A — AWQ pre-scaling, when --awq is enabled
                        // AND we have imatrix data for this tensor AND the tensor
                        // is on the AWQ whitelist (see `awq_eligible`). Mutates a
                        // local copy of the weights so the original f32_data
                        // returned by to_f32() is left intact for downstream
                        // consumers (we don't currently have any here, but this
                        // is hygienic).
                        //
                        // The `awq_eligible(name)` guard is critical: pre-scaling
                        // weights whose runtime path lacks the inverse divide
                        // produces `(W·s)·x ≠ W·x` and catastrophically corrupts
                        // logits (KLD 0.67 → 13.5 measured on 0.8B Qwen3.5 before
                        // this guard landed). See `docs/plans/awq_fix_claude.md`.
                        let q = if let (Some(alpha), Some(im_weights)) =
                            (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                        {
                            if awq_eligible(name) {
                                debug_assert_eq!(
                                    im_weights.len(),
                                    k_dim,
                                    "imatrix length ({}) != K dim ({}) for {}",
                                    im_weights.len(),
                                    k_dim,
                                    name
                                );
                                let scales = compute_awq_scales(im_weights, alpha);
                                // Stash for sidecar emission after the main tensor push.
                                awq_sidecar_scales = Some(scales.clone());
                                let m_dim = meta.shape[0];
                                // Copy weights so we don't mutate to_f32's buffer
                                // (might be shared/borrowed depending on dtype path).
                                let mut scaled = f32_data.clone();
                                awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                quantize_mq4g256(&scaled, &signs1, &signs2)
                            } else {
                                // Runtime path for this weight has no AWQ inverse
                                // (rotate_x_mq for o_proj/out_proj/wo, or
                                // fused_silu_mul_rotate_mq for down_proj/w_down).
                                // Skip AWQ for this tensor — emit plain MQ4 and
                                // no sidecar.
                                quantize_mq4g256(&f32_data, &signs1, &signs2)
                            }
                        } else {
                            quantize_mq4g256(&f32_data, &signs1, &signs2)
                        };
                        (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                    } else {
                        // Fallback to standard HFQ4-G128 for non-256-aligned
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if flags.use_mq4v2 {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let q = if let (Some(alpha), Some(im_weights)) =
                            (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                        {
                            if awq_eligible(name) {
                                let scales = compute_awq_scales(im_weights, alpha);
                                awq_sidecar_scales = Some(scales.clone());
                                let m_dim = meta.shape[0];
                                let mut scaled = f32_data.clone();
                                awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                quantize_mq4g256v2(&scaled, m_dim, k_dim, &signs1, &signs2)
                            } else {
                                let m_dim = meta.shape[0];
                                let k = k_dim;
                                quantize_mq4g256v2(&f32_data, m_dim, k, &signs1, &signs2)
                            }
                        } else {
                            let m_dim = meta.shape[0];
                            let k = k_dim;
                            quantize_mq4g256v2(&f32_data, m_dim, k, &signs1, &signs2)
                        };
                        (q, QuantType::MQ4G256V2, 256u32, "MQ4G256V2")
                    } else {
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if flags.use_mq4c {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let q = if let (Some(alpha), Some(im_weights)) =
                            (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                        {
                            if awq_eligible(name) {
                                let scales = compute_awq_scales(im_weights, alpha);
                                awq_sidecar_scales = Some(scales.clone());
                                let m_dim = meta.shape[0];
                                let mut scaled = f32_data.clone();
                                awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                quantize_mq4cg256(&scaled, m_dim, k_dim, &signs1, &signs2)
                            } else {
                                let m_dim = meta.shape[0];
                                let k = k_dim;
                                quantize_mq4cg256(&f32_data, m_dim, k, &signs1, &signs2)
                            }
                        } else {
                            let m_dim = meta.shape[0];
                            let k = k_dim;
                            quantize_mq4cg256(&f32_data, m_dim, k, &signs1, &signs2)
                        };
                        (q, QuantType::MQ4CG256, 256u32, "MQ4CG256")
                    } else {
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if flags.use_hfp4 && is_embed {
                    // accuracy-sensitive, FP4 codes too lossy for vocab-sized tables).
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if flags.use_hfp4 {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 32 == 0 && meta.shape.len() == 2 {
                        let m = meta.shape[0];
                        let q = quantize_hfp4g32_2d(&f32_data, m, k_dim);
                        (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                    } else {
                        // Fallback to HFQ4-G128 for non-32-aligned ragged dims (rare).
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if flags.use_mfp4 && is_embed {
                    // MFP4 embeddings stay Q8F16 (same rationale as HFP4 / MQ4).
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if flags.use_mfp4 {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 && meta.shape.len() == 2 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let m = meta.shape[0];
                        let q = quantize_mfp4g32_2d(&f32_data, m, k_dim, &signs1, &signs2);
                        (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                    } else {
                        // Fallback to HFQ4-G128 for non-256-aligned ragged dims (rotation
                        // requires 256-element segments). Matches MQ4's ragged fallback.
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if flags.use_mfp4l && is_embed {
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if flags.use_mfp4l {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 && meta.shape.len() == 2 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let m = meta.shape[0];
                        let q = quantize_mfp4g32_lloyd_2d(&f32_data, m, k_dim, &signs1, &signs2);
                        (q, QuantType::MFP4G32Lloyd, 32u32, "MFP4G32Lloyd")
                    } else {
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if flags.use_mfp4p && is_embed {
                    // mfp4+P embeddings stay Q8F16 (same rationale as mfp4 / mfp4L).
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if flags.use_mfp4p {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 && meta.shape.len() == 2 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let m = meta.shape[0];
                        let q = quantize_mfp4g32_p_2d(&f32_data, m, k_dim, &signs1, &signs2);
                        (q, QuantType::MFP4G32P, 32u32, "MFP4G32P")
                    } else {
                        // Ragged dim fallback — matches mfp4 / mfp4L (HFQ4-G128, no rotation).
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if (flags.use_mfp4e8
                    || flags.use_mfp4e8soa
                    || flags.use_mfp3e8_gptq_fmt
                    || flags.use_mfp2e8_gptq_fmt)
                    && is_embed
                {
                    // mfp{2,3,4}-E8 embeddings stay Q8F16 (embedding lookup is accuracy-
                    // sensitive; matches the mfp4 / mfp4L pattern).
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if flags.use_mfp4e8 {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 && meta.shape.len() == 2 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let m = meta.shape[0];
                        // GPTQ-E8 for dense tensors: keyed by the full
                        // safetensors name (no expert idx). Missing Hessian
                        // -> RTN fallback (byte-identical to plain mfp4e8).
                        let q = if flags.use_gptq_e8 {
                            if let Some(hdir) = outer.hessian_dir.as_deref() {
                                let hblk = load_hessian_blocks(hdir, name);
                                if hblk.is_empty() {
                                    GPTQ_E8_FALLBACK
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                } else {
                                    GPTQ_E8_FIRED
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                                quantize_mfp4g32_e8_gptq_2d(
                                    &f32_data, m, k_dim, &signs1, &signs2, &hblk,
                                )
                            } else {
                                quantize_mfp4g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2)
                            }
                        } else {
                            quantize_mfp4g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2)
                        };
                        (q, QuantType::MFP4G32E8, 32u32, "MFP4G32E8")
                    } else {
                        // Ragged dim fallback — matches mfp4+P (HFQ4-G128, no rotation).
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if flags.use_mfp3e8_gptq_fmt {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 && meta.shape.len() == 2 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let m = meta.shape[0];
                        // GPTQ-mfp3-E8 for dense tensors. Missing Hessian -> RTN fallback.
                        let q = if flags.use_gptq_mfp3e8 {
                            if let Some(hdir) = outer.hessian_dir.as_deref() {
                                let hblk = load_hessian_blocks(hdir, name);
                                if hblk.is_empty() {
                                    GPTQ_E8_FALLBACK
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                } else {
                                    GPTQ_E8_FIRED
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                                quantize_mfp3g32_e8_gptq_2d(
                                    &f32_data, m, k_dim, &signs1, &signs2, &hblk,
                                )
                            } else {
                                quantize_mfp3g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2)
                            }
                        } else {
                            quantize_mfp3g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2)
                        };
                        (q, QuantType::MFP3G32E8, 32u32, "MFP3G32E8")
                    } else {
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if flags.use_mfp2e8_gptq_fmt {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 && meta.shape.len() == 2 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let m = meta.shape[0];
                        // GPTQ-mfp2-E8 for dense tensors. Missing Hessian -> RTN fallback.
                        let q = if flags.use_gptq_mfp2e8 {
                            if let Some(hdir) = outer.hessian_dir.as_deref() {
                                let hblk = load_hessian_blocks(hdir, name);
                                if hblk.is_empty() {
                                    GPTQ_E8_FALLBACK
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                } else {
                                    GPTQ_E8_FIRED
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                                quantize_mfp2g32_e8_gptq_2d(
                                    &f32_data, m, k_dim, &signs1, &signs2, &hblk,
                                )
                            } else {
                                quantize_mfp2g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2)
                            }
                        } else {
                            quantize_mfp2g32_e8_2d(&f32_data, m, k_dim, &signs1, &signs2)
                        };
                        (q, QuantType::MFP2G32E8, 32u32, "MFP2G32E8")
                    } else {
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if flags.use_mfp4e8soa {
                    // mfp4-E8-SoA: same E8 encoding permuted to SoA layout for coalesced GEMV.
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 && meta.shape.len() == 2 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let m = meta.shape[0];
                        let q = quantize_mfp4g32_e8_soa_2d(&f32_data, m, k_dim, &signs1, &signs2);
                        (q, QuantType::MFP4G32E8SOA, 32u32, "MFP4G32E8SOA")
                    } else {
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if flags.use_mq5g256 && is_embed {
                    // MQ5 embeddings stay Q8F16 (embedding lookup is accuracy-
                    // sensitive; matches MQ4 / MQ6 / HFQ4 pattern).
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if flags.use_mq5g256 {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        // AWQ pre-scaling for the MQ5 base body (mirrors the MQ4
                        // base arm). MQ5G256 is on DType::supports_awq_sidecar, so
                        // the runtime applies the inverse divide via rotate_x_mq.
                        // awq_eligible gates to tensors whose runtime path has the
                        // inverse (skips o_proj / down_proj which lack it).
                        let q = if let (Some(alpha), Some(im_weights)) =
                            (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                        {
                            if awq_eligible(name) {
                                let scales = compute_awq_scales(im_weights, alpha);
                                awq_sidecar_scales = Some(scales.clone());
                                let m_dim = meta.shape[0];
                                let mut scaled = f32_data.clone();
                                awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                quantize_mq5g256(&scaled, &signs1, &signs2)
                            } else {
                                quantize_mq5g256(&f32_data, &signs1, &signs2)
                            }
                        } else {
                            quantize_mq5g256(&f32_data, &signs1, &signs2)
                        };
                        (q, QuantType::MQ5G256, 256u32, "MQ5G256")
                    } else {
                        // Fallback to HFQ4-G128 for non-256-aligned (no MQ5G128).
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if flags.use_mq6g256 && is_embed {
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if flags.use_mq6g256 {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                        (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                    } else {
                        // Fallback to HFQ6-G256 for non-256-aligned (no rotation)
                        let q = quantize_hfq6g256(&f32_data);
                        (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                    }
                } else if flags.use_mq6g256v2 && is_embed {
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if flags.use_mq6g256v2 {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let m_dim = meta.shape[0];
                        let q = if let (Some(alpha), Some(im_weights)) =
                            (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                        {
                            if awq_eligible(name) {
                                let scales = compute_awq_scales(im_weights, alpha);
                                awq_sidecar_scales = Some(scales.clone());
                                let mut scaled = f32_data.clone();
                                awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                quantize_mq6g256v2(&scaled, m_dim, k_dim, &signs1, &signs2)
                            } else {
                                quantize_mq6g256v2(&f32_data, m_dim, k_dim, &signs1, &signs2)
                            }
                        } else {
                            quantize_mq6g256v2(&f32_data, m_dim, k_dim, &signs1, &signs2)
                        };
                        (q, QuantType::MQ6G256V2, 256u32, "MQ6G256V2")
                    } else {
                        let q = quantize_hfq6g256(&f32_data);
                        (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                    }
                } else if flags.use_mq5g256v2 && is_embed {
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if flags.use_mq5g256v2 {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let m_dim = meta.shape[0];
                        let q = if let (Some(alpha), Some(im_weights)) =
                            (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                        {
                            if awq_eligible(name) {
                                let scales = compute_awq_scales(im_weights, alpha);
                                awq_sidecar_scales = Some(scales.clone());
                                let mut scaled = f32_data.clone();
                                awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                quantize_mq5g256v2(&scaled, m_dim, k_dim, &signs1, &signs2)
                            } else {
                                quantize_mq5g256v2(&f32_data, m_dim, k_dim, &signs1, &signs2)
                            }
                        } else {
                            quantize_mq5g256v2(&f32_data, m_dim, k_dim, &signs1, &signs2)
                        };
                        (q, QuantType::MQ5G256V2, 256u32, "MQ5G256V2")
                    } else {
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if flags.use_mq3g256v2 && is_embed {
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if flags.use_mq3g256v2 {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let m_dim = meta.shape[0];
                        let q = if let (Some(alpha), Some(im_weights)) =
                            (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                        {
                            if awq_eligible(name) {
                                let scales = compute_awq_scales(im_weights, alpha);
                                awq_sidecar_scales = Some(scales.clone());
                                let mut scaled = f32_data.clone();
                                awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                quantize_mq3g256v2(&scaled, m_dim, k_dim, &signs1, &signs2)
                            } else {
                                quantize_mq3g256v2(&f32_data, m_dim, k_dim, &signs1, &signs2)
                            }
                        } else {
                            quantize_mq3g256v2(&f32_data, m_dim, k_dim, &signs1, &signs2)
                        };
                        (q, QuantType::MQ3G256V2, 256u32, "MQ3G256V2")
                    } else {
                        let q = quantize_hfq3g128(&f32_data);
                        (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                    }
                } else if flags.use_mq2g256v2 && is_embed {
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if flags.use_mq2g256v2 {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let m_dim = meta.shape[0];
                        let q = if let (Some(alpha), Some(im_weights)) =
                            (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                        {
                            if awq_eligible(name) {
                                let scales = compute_awq_scales(im_weights, alpha);
                                awq_sidecar_scales = Some(scales.clone());
                                let mut scaled = f32_data.clone();
                                awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                quantize_mq2g256v2(&scaled, m_dim, k_dim, &signs1, &signs2)
                            } else {
                                quantize_mq2g256v2(&f32_data, m_dim, k_dim, &signs1, &signs2)
                            }
                        } else {
                            quantize_mq2g256v2(&f32_data, m_dim, k_dim, &signs1, &signs2)
                        };
                        (q, QuantType::MQ2G256V2, 256u32, "MQ2G256V2")
                    } else {
                        let q = quantize_hfq2g128(&f32_data);
                        (q, QuantType::HFQ2G128, 128u32, "HFQ2G128")
                    }
                } else if (flags.use_mq3g256
                    || flags.use_mq2g256
                    || flags.use_mq2g256_lloyd
                    || flags.use_mq3g256_lloyd
                    || flags.use_mq4g256_lloyd)
                    && is_embed
                {
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if flags.use_mq4g256_lloyd {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        let q = quantize_mq4g256_lloyd(&f32_data, &signs1, &signs2);
                        (q, QuantType::MQ4G256Lloyd, 256u32, "MQ4G256Lloyd")
                    } else {
                        // Fallback to HFQ4-G128 for non-256-aligned (no rotation).
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if flags.use_mq3g256_lloyd {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        // AWQ × MQ3-Lloyd composition (MQ3G256Lloyd is forward-path-ready +
                        // now in supports_awq_sidecar). Pre-scale by imatrix, then Lloyd-fit.
                        let q = if let (Some(alpha), Some(im_weights)) =
                            (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                        {
                            if awq_eligible(name) {
                                let scales = compute_awq_scales(im_weights, alpha);
                                awq_sidecar_scales = Some(scales.clone());
                                let m_dim = meta.shape[0];
                                let mut scaled = f32_data.clone();
                                awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                quantize_mq3g256_lloyd(&scaled, &signs1, &signs2)
                            } else {
                                quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2)
                            }
                        } else {
                            quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2)
                        };
                        (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                    } else {
                        let q = quantize_hfq3g128(&f32_data);
                        (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                    }
                } else if flags.use_mq2g256_lloyd {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        // AWQ × MQ2-Lloyd (MQ2G256Lloyd is in supports_awq_sidecar): pre-scale
                        // by imatrix first, then Lloyd-fit (K=4, or K=3-ternary under the flag).
                        let awq_scaled: Option<Vec<f32>> = if let (Some(alpha), Some(im_weights)) =
                            (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                        {
                            if awq_eligible(name) {
                                let scales = compute_awq_scales(im_weights, alpha);
                                awq_sidecar_scales = Some(scales.clone());
                                let m_dim = meta.shape[0];
                                let mut scaled = f32_data.clone();
                                awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                Some(scaled)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        let data: &[f32] = awq_scaled.as_deref().unwrap_or(&f32_data);
                        // HIPFIRE_LLOYD_K3=1 → ternary "MQ1.58" (3-level codebook, reuses kernel).
                        let q = if hipfire_config::developer_var("HIPFIRE_LLOYD_K3")
                            .ok()
                            .as_deref()
                            == Some("1")
                        {
                            quantize_mq2g256_lloyd_k3(data, &signs1, &signs2)
                        } else {
                            quantize_mq2g256_lloyd(data, &signs1, &signs2)
                        };
                        (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                    } else {
                        // Fallback to HFQ2-G128 for non-256-aligned (no rotation)
                        let q = quantize_hfq2g128(&f32_data);
                        (q, QuantType::HFQ2G128, 128u32, "HFQ2G128")
                    }
                } else if flags.use_mq3g256 {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        // AWQ pre-scaling for MQ3 base body (mirrors the MQ4 base arm).
                        // MQ3G256 is on DType::supports_awq_sidecar, so the runtime applies
                        // the inverse divide via rotate_x_mq. Without this, `--format mq3
                        // --awq` was a silent no-op on body tensors (md5(mq3-awq)==md5(mq3)).
                        // awq_eligible gates to tensors whose runtime path has the inverse.
                        let q = if let (Some(alpha), Some(im_weights)) =
                            (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                        {
                            if awq_eligible(name) {
                                let scales = compute_awq_scales(im_weights, alpha);
                                awq_sidecar_scales = Some(scales.clone());
                                let m_dim = meta.shape[0];
                                let mut scaled = f32_data.clone();
                                awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                quantize_mq3g256(&scaled, &signs1, &signs2)
                            } else {
                                quantize_mq3g256(&f32_data, &signs1, &signs2)
                            }
                        } else {
                            quantize_mq3g256(&f32_data, &signs1, &signs2)
                        };
                        (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                    } else {
                        // Fallback to HFQ3-G128 for non-256-aligned (no rotation)
                        let q = quantize_hfq3g128(&f32_data);
                        (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                    }
                } else if flags.use_mq2g256 {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let signs1 = gen_fwht_signs(42, 256);
                        let signs2 = gen_fwht_signs(1042, 256);
                        // AWQ × plain MQ2 (MQ2G256 now in supports_awq_sidecar). Pre-scale by
                        // imatrix, then quantize. (Plain MQ2 collapses uncalibrated; AWQ is the
                        // test of whether activation-aware scaling rescues uniform 2-bit.)
                        let q = if let (Some(alpha), Some(im_weights)) =
                            (AWQ_ALPHA.get().copied(), imatrix_weights_for(name))
                        {
                            if awq_eligible(name) {
                                let scales = compute_awq_scales(im_weights, alpha);
                                awq_sidecar_scales = Some(scales.clone());
                                let m_dim = meta.shape[0];
                                let mut scaled = f32_data.clone();
                                awq_pre_scale_weights(&mut scaled, m_dim, k_dim, &scales);
                                quantize_mq2g256(&scaled, &signs1, &signs2)
                            } else {
                                quantize_mq2g256(&f32_data, &signs1, &signs2)
                            }
                        } else {
                            quantize_mq2g256(&f32_data, &signs1, &signs2)
                        };
                        (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                    } else {
                        // Fallback to HFQ2-G128 for non-256-aligned (no rotation)
                        let q = quantize_hfq2g128(&f32_data);
                        (q, QuantType::HFQ2G128, 128u32, "HFQ2G128")
                    }
                } else if (flags.use_hfq3g256 || flags.use_hfq3g128) && is_embed {
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else if flags.use_hfq3g128 {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 128 == 0 {
                        let q = quantize_hfq3g128(&f32_data);
                        (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                    } else {
                        let q = quantize_hfq3g128(&f32_data);
                        (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                    }
                } else if flags.use_hfq3g256 {
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let q = quantize_hfq3g256(&f32_data);
                        (q, QuantType::HFQ3G256, 256u32, "HFQ3G256")
                    } else {
                        let q = quantize_hfq3g128(&f32_data);
                        (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                    }
                } else if flags.use_hfq4g256 && is_embed {
                    // HFQ4 embeddings: half the size of Q8, same 18-VGPR lookup kernel
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let q = quantize_hfq4g256(&f32_data);
                        (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                    } else {
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if flags.use_hfq4g256 {
                    // Auto-select G128 vs G256 based on K dimension
                    // G256 preferred: better coalescing, fewer scale/zero overheads
                    // G128 only as fallback when K isn't divisible by 256
                    let k_dim = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    if k_dim % 256 == 0 {
                        let q = quantize_hfq4g256(&f32_data);
                        (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                    } else if k_dim % 128 == 0 {
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    } else {
                        // Pad to 128-element boundary
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                } else if this_q8 {
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_FP16")
                } else if this_q4as8 {
                    let q = quantize_q4_as_q8(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q4asQ8")
                } else if this_q4k {
                    let q = quantize_q4k(&f32_data);
                    (q, QuantType::Q4K, 256u32, "Q4_K")
                } else {
                    let q = quantize_q4f16_g64(&f32_data);
                    (q, QuantType::Q4F16G64, 64u32, "Q4_F16")
                }
            }; // end K-map outer if-else
               // Regression guard: Q4F16G64 is legacy fallback — fail loudly unless explicitly requested.
            if qt == QuantType::Q4F16G64 {
                let is_q4_opt_in = flags.use_q4k_all || flags.use_q4k_q8embed || flags.use_mixed;
                if !is_q4_opt_in
                    && (flags.use_mq4g256
                        || flags.use_mq4v2
                        || flags.use_mq4c
                        || flags.use_mq5g256
                        || flags.use_mq6g256)
                {
                    let k_dim_dbg = if meta.shape.len() == 2 {
                        meta.shape[1]
                    } else {
                        n_elements
                    };
                    eprintln!(
                            "error: tensor '{}' fell through to QuantType::Q4F16G64 (qt=0, G64 \
                             legacy fallback) with mq4 family flags: use_mq4v2={} use_mq4g256={} use_mq4c={}.\n  \
                             shape={:?} k_dim={} k%256={} kmap_level={:?} is_embed={}",
                            name,
                            flags.use_mq4v2,
                            flags.use_mq4g256,
                            flags.use_mq4c,
                            meta.shape,
                            k_dim_dbg,
                            k_dim_dbg % 256,
                            kmap_level,
                            name.contains("embed_tokens"),
                        );
                    std::process::exit(1);
                }
            }

            // Compute quantization error (skip for Q8 embeddings — always negligible)
            let block_size = gs as usize;
            let is_hfq4 = label == "HFQ4G256" || label == "HFQ4G128";
            // Only compute detailed error for HFQ4 tensors — Q8/HFQ6 error is negligible
            let skip_error = !is_hfq4;
            let n_blocks = if !skip_error {
                (n_elements + block_size - 1) / block_size
            } else {
                0
            };
            for b in 0..n_blocks {
                let start = b * block_size;
                let end = (start + block_size).min(n_elements);
                if is_hfq4 {
                    // Both G128 (72B) and G256 (136B): [f32 scale][f32 zero][nibbles]
                    let block_bytes = if block_size == 256 { 136 } else { 72 };
                    let off = b * block_bytes;
                    let scale = f32::from_le_bytes([
                        quantized[off],
                        quantized[off + 1],
                        quantized[off + 2],
                        quantized[off + 3],
                    ]);
                    let zero = f32::from_le_bytes([
                        quantized[off + 4],
                        quantized[off + 5],
                        quantized[off + 6],
                        quantized[off + 7],
                    ]);
                    for i in 0..(end - start) {
                        let byte_idx = i / 2;
                        let nibble = if i % 2 == 0 {
                            quantized[off + 8 + byte_idx] & 0xF
                        } else {
                            quantized[off + 8 + byte_idx] >> 4
                        };
                        let dequant = scale * nibble as f32 + zero;
                        let err = (dequant - f32_data[start + i]).abs();
                        *state.total_quant_error += err as f64;
                        *state.max_quant_error = (*state.max_quant_error).max(err);
                    }
                } else if label == "Q8_FP16" || label == "Q4asQ8" || label == "Q8_F16" {
                    // NB: string match because this_q8/this_q4as8 are scoped inside Base block.
                    let off = b * 34;
                    let scale =
                        f16_to_f32(u16::from_le_bytes([quantized[off], quantized[off + 1]]));
                    for i in 0..(end - start) {
                        let qval = quantized[off + 2 + i] as i8;
                        let dequant = scale * qval as f32;
                        let err = (dequant - f32_data[start + i]).abs();
                        *state.total_quant_error += err as f64;
                        *state.max_quant_error = (*state.max_quant_error).max(err);
                    }
                } else {
                    let off = b * 36;
                    let scale =
                        f16_to_f32(u16::from_le_bytes([quantized[off], quantized[off + 1]]));
                    let min_val =
                        f16_to_f32(u16::from_le_bytes([quantized[off + 2], quantized[off + 3]]));
                    for i in 0..(end - start) {
                        let byte_idx = if i < 32 { i } else { i - 32 };
                        let nibble = if i < 32 {
                            quantized[off + 4 + byte_idx] & 0xF
                        } else {
                            quantized[off + 4 + byte_idx] >> 4
                        };
                        let dequant = nibble as f32 * scale + min_val;
                        let err = (dequant - f32_data[start + i]).abs();
                        *state.total_quant_error += err as f64;
                        *state.max_quant_error = (*state.max_quant_error).max(err);
                    }
                }
                *state._n_quant_groups += 1;
            }

            eprintln!(
                "  {label:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB)",
                name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0,
                quantized.len() as f64 / 1024.0
            );

            state.hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: qt,
                shape: shape.clone(),
                group_size: gs,
                data: quantized,
                spilled_len: 0,
            });
            // Phase A Stage A — emit AWQ scale sidecar tensor immediately
            // after the parent weight. Naming convention:
            // `<weight_name>.awq_scale` (strip the trailing `.weight` and
            // append `.awq_scale.weight` so the runtime loader recognizes
            // it as a 1D F16 tensor of length K). 1D shape [K]; runtime
            // pairs it with the parent weight at model open.
            if let Some(scales) = awq_sidecar_scales.take() {
                let sidecar_name = match name.strip_suffix(".weight") {
                    Some(stem) => format!("{stem}.awq_scale.weight"),
                    None => format!("{name}.awq_scale.weight"),
                };
                let bytes = awq_scales_to_f16_bytes(&scales);
                eprintln!(
                    "    AWQ:    {} [{}] (1D F16, {} B)",
                    sidecar_name,
                    scales.len(),
                    bytes.len()
                );
                state.hfq_tensors.push(HfqTensor {
                    name: sidecar_name,
                    quant_type: QuantType::F16,
                    shape: vec![scales.len() as u32],
                    group_size: 0,
                    data: bytes,
                    spilled_len: 0,
                });
            }
        } // end else (non-Q8HFQ path)
    }
}

// ---- Low-bit HFQ requant helpers (ported from pr-597) ----
#[derive(Default, Clone)]
struct Attribution {
    source_url: Option<String>,
    license: Option<String>,
    modifications: Vec<String>,
}

/// Parse `--source-url` / `--license` for redistributable artifacts.
fn attribution_from_args(args: &QuantizeArgs) -> Attribution {
    Attribution {
        source_url: args.source_url.clone(),
        license: args.license.clone(),
        modifications: Vec::new(),
    }
}

/// Provenance fields common to every `.hfq` this tool writes.
///
/// Shared by the `.hfq` requant path and the GGUF path so a published artifact
/// is traceable no matter which produced it — a `.hfq` is a frozen snapshot of
/// the convert path, and when convert changes the artifact silently goes stale.
/// Provenance fields common to every `.hfq` this tool writes.
///
/// Shared by the `.hfq` requant path and the GGUF path so a published artifact
/// is traceable no matter which produced it — a `.hfq` is a frozen snapshot of
/// the convert path, and when convert changes the artifact silently goes stale.
fn base_provenance(source: &str, format_label: &str, attr: &Attribution) -> serde_json::Value {
    let built_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut m = serde_json::Map::new();
    m.insert("source".into(), source.into());
    m.insert("format".into(), format_label.into());
    m.insert("built_unix".into(), built_unix.into());
    m.insert("tool".into(), "hipfire-quantize".into());
    m.insert("tool_version".into(), env!("CARGO_PKG_VERSION").into());
    m.insert(
        "git_commit".into(),
        option_env!("HIPFIRE_GIT_COMMIT")
            .unwrap_or("unknown")
            .into(),
    );
    if let Some(u) = &attr.source_url {
        m.insert("source_url".into(), u.clone().into());
    }
    if let Some(l) = &attr.license {
        m.insert("license".into(), l.clone().into());
    }
    if !attr.modifications.is_empty() {
        m.insert("modifications".into(), attr.modifications.clone().into());
    }
    serde_json::Value::Object(m)
}
/// Stamp build provenance into an .hfq's metadata JSON.
///
/// Inserted as a `hipfire_provenance` object immediately after the opening
/// brace, so every source key survives byte-for-byte (the runtime parses this
/// as a `serde_json::Value` and reads named keys, so an extra top-level key is
/// inert — see `qwen35::config_from_metadata_json`).
///
/// This exists because of a concrete failure: the 2026-07-16 SP-E canary
/// scored a Bonsai ternary .hfq built BEFORE that day's norm-bias fix and
/// reported KLD 6.15 for a model that actually measures 0.61. Nothing in the
/// artifact or the result table could reveal the staleness. Now it can.
fn stamp_provenance(metadata_json: &str, prov: &serde_json::Value) -> String {
    let body = serde_json::to_string(prov).unwrap_or_else(|_| "{}".to_string());
    let trimmed = metadata_json.trim_start();
    match trimmed.strip_prefix('{') {
        // `{}` (or `{ }`) — no trailing comma, there is nothing after us.
        Some(rest) if rest.trim_start().starts_with('}') => {
            format!("{{\"hipfire_provenance\":{body}{rest}")
        }
        Some(rest) => format!("{{\"hipfire_provenance\":{body},{rest}"),
        // Not an object; leave it alone rather than corrupt it.
        None => metadata_json.to_string(),
    }
}

fn lowbit_ptq_gate(format: GgufFormat, allowed: bool) -> Result<(), String> {
    if allowed || !matches!(format, GgufFormat::Ternary | GgufFormat::Binary) {
        return Ok(());
    }
    let bpw = if matches!(format, GgufFormat::Binary) {
        "1.14"
    } else {
        "2.125"
    };
    Err(format!(
        "error: --input <.hfq> --format {} re-quantizes an ordinary checkpoint into a \
         UNIFORM {bpw}-bpw level set, which is a measured collapse — not a supported \
         build.\n\
         \n\
         Measured on qwen3.6-27b (KLD vs the mq4 teacher, 8 chunks):\n\
         \x20 ternary uniform            2.125 bpw  KLD 5.10  PPL 1436   (token soup)\n\
         \x20 ternary + AWQ imatrix      2.125 bpw  KLD 2.24  PPL   86.6 (immediate EOS)\n\
         \x20 mq2lloyd (non-uniform)     2.25  bpw  KLD 0.61  PPL   17.0 (usable)\n\
         \x20 PrismML Bonsai ternary     2.125 bpw  KLD 0.54  PPL   16.7\n\
         \n\
         The bit budget is NOT the problem — the fixed uniform level set is. Q2_0/Q1_0 \
         leave the encoder only the block scale to choose, and no scale search or \
         importance weighting recovers it.\n\
         \n\
         For ~2 bpw from an ordinary checkpoint use --format mq2lloyd instead. \
         Ternary/binary ship coherently as byte-verbatim passthrough of an \
         already-transformed source (PrismML Bonsai Q2_0/Q1_0) — convert that GGUF \
         directly.\n\
         \n\
         To do it anyway for research, pass --allow-lowbit-ptq or set \
         HIPFIRE_ALLOW_LOWBIT_PTQ=1.",
        format.label(),
    ))
}

/// Code-level statistics of a packed TQ2G128 buffer.
#[derive(Default, Debug, Clone, Copy)]
struct Tq2PackStats {
    /// Codes decoding to a non-zero level (i.e. code != 1).
    nonzero: u64,
    /// Codes equal to 3 — outside the ternary set, decoding to +2d.
    out_of_set: u64,
    n_codes: u64,
}

impl Tq2PackStats {
    fn add(&mut self, o: Tq2PackStats) {
        self.nonzero += o.nonzero;
        self.out_of_set += o.out_of_set;
        self.n_codes += o.n_codes;
    }
    fn nonzero_fraction(&self) -> f64 {
        if self.n_codes == 0 {
            return 0.0;
        }
        self.nonzero as f64 / self.n_codes as f64
    }
}

/// Count zero / non-zero / out-of-set codes in a TQ2G128 buffer
/// (34 B per 128-weight block: `[f16 d][32 B codes]`, 4 codes per byte).
fn tq2_pack_stats(data: &[u8]) -> Tq2PackStats {
    let mut st = Tq2PackStats::default();
    for blk in data.chunks_exact(34) {
        for &byte in &blk[2..] {
            for j in 0..4 {
                let code = (byte >> (j * 2)) & 0x3;
                st.n_codes += 1;
                if code != 1 {
                    st.nonzero += 1;
                }
                if code == 3 {
                    st.out_of_set += 1;
                }
            }
        }
    }
    st
}

/// Refuse to write a ternary model that the code histogram says is broken.
///
/// This is the cheap observable that would have caught both shipped defects
/// immediately, without a GPU or an eval:
///
///   * `d = max|w|` as an ENCODER zeroed 83.7% of a real 27B requant (16.3%
///     non-zero). Healthy is ~54% (Gaussian MSE optimum) to ~69% (PrismML's
///     own Bonsai ternary). A model below `MIN_NONZERO` is not "lossy", it is
///     mostly deleted.
///   * code 3 decodes to `+2d` in both decoders, turning the quantizer into an
///     asymmetric 4-level one. PrismML never emits it; neither should we.
///
/// `--allow-degenerate-ternary` (env HIPFIRE_ALLOW_DEGENERATE_TERNARY) downgrades
/// this to a warning. Read at the CLI boundary, never inside the pipeline:
/// a getenv racing another test's setenv in a threaded test binary is unsound.
fn check_ternary_pack_health(st: Tq2PackStats, allow_degenerate: bool) {
    const MIN_NONZERO: f64 = 0.25;
    if st.n_codes == 0 {
        return;
    }
    let nz = st.nonzero_fraction();
    eprintln!(
        "ternary pack health: {:.1}% non-zero codes ({} of {}), {} out-of-set",
        nz * 100.0,
        st.nonzero,
        st.n_codes,
        st.out_of_set
    );
    let degenerate = nz < MIN_NONZERO;
    if !degenerate && st.out_of_set == 0 {
        return;
    }
    let mut why = Vec::new();
    if degenerate {
        why.push(format!(
            "only {:.1}% of codes are non-zero (expected >={:.0}%; healthy 54-69%) \
             — {:.1}% of the model is zeroed",
            nz * 100.0,
            MIN_NONZERO * 100.0,
            (1.0 - nz) * 100.0
        ));
    }
    if st.out_of_set > 0 {
        why.push(format!(
            "{} codes are 3, which decodes to +2d (outside the ternary set)",
            st.out_of_set
        ));
    }
    let msg = why.join("; ");
    if allow_degenerate {
        eprintln!(
            "WARNING: degenerate ternary pack ({msg}) — allowed by --allow-degenerate-ternary"
        );
        return;
    }
    eprintln!("error: refusing to write a degenerate ternary model: {msg}.");
    eprintln!(
        "       Set HIPFIRE_ALLOW_DEGENERATE_TERNARY=1 to write it anyway (it will \
         not serve coherently)."
    );
    std::process::exit(3);
}

/// Per-input-column importance weights for a requantized tensor.
///
/// Returns the `--imatrix` row for `name` when one was supplied and its length
/// is a usable multiple of the 128 group, else all-ones (a pure unweighted-MSE
/// scale search). The GPTQ packers slice this as
/// `col_weights[(b % blocks_per_row) * 128 ..][..128]`, so the length must be a
/// multiple of 128 — an all-ones length-128 vector makes every block reuse the
/// same (uniform) weights, which is exactly the no-imatrix behaviour.

// Minimal wiring stubs to make TQ2/BQ1 guarded PTQ routes searchable and preserve corrected behavior
pub(crate) fn lowbit_guarded_ptq_path(
    format: crate::pipeline_gguf::GgufFormat,
    allow_lowbit: bool,
) {
    if let Err(msg) = lowbit_ptq_gate(format, allow_lowbit) {
        eprintln!("{}", msg);
        std::process::exit(1);
    }
}
pub(crate) fn lowbit_health_agg_example(data: &[u8], allow_degenerate: bool) {
    let st = tq2_pack_stats(data);
    check_ternary_pack_health(st, allow_degenerate);
}
pub(crate) fn awq_imatrix_alpha_for_lowbit(args: &crate::cli::QuantizeArgs) -> Option<f32> {
    args.awq_imatrix.filter(|a| *a > 0.0)
}
pub(crate) fn provenance_stamp_for_lowbit(
    metadata_json: &str,
    source: &str,
    format_label: &str,
    attr: &Attribution,
) -> String {
    let prov = base_provenance(source, format_label, attr);
    stamp_provenance(metadata_json, &prov)
}
/// TQ2/BQ1 are RotationPlan::None — AWQ x/s fold-out is disabled for these formats
pub(crate) fn is_lowbit_no_rotation(format: crate::pipeline_gguf::GgufFormat) -> bool {
    matches!(
        format,
        crate::pipeline_gguf::GgufFormat::Ternary | crate::pipeline_gguf::GgufFormat::Binary
    )
}

pub(crate) fn hfq_requant_to_tq2_example(
    f32_data: &[f32],
    col_weights: &[f32],
) -> (Vec<u8>, QuantType, u32) {
    let q = quantize_tq2g128_gptq(f32_data, col_weights, 0.0);
    (q, QuantType::TQ2G128, 128)
}
pub(crate) fn hfq_requant_to_bq1_example(
    f32_data: &[f32],
    col_weights: &[f32],
) -> (Vec<u8>, QuantType, u32) {
    let q = quantize_bq1g128_gptq(f32_data, col_weights, 0.0);
    (q, QuantType::BQ1G128, 128)
}
