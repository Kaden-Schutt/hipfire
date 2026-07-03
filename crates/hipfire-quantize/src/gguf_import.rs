// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GGUF → `.hfq` import pipeline.
//!
//! Reads a GGUF file (via the dedicated `hipfire-gguf` codec), re-quantizes its
//! weights through the shared quant codecs + K-map plan ([`super::quant_plan`]),
//! and writes a native `.hfq` through [`super::hfq_out`]. Owned by the library
//! (not the quantize binary) so the user-facing importer lives in
//! `hipfire-coexistence` (`import gguf`) per AGENTS.md, while re-quantization
//! stays with the codecs it depends on. The quantize binary keeps only a
//! deprecation shim for `--input *.gguf`.

use crate::codecs::*;
use crate::hfq_out::{
    insert_parameter_counts_metadata, metadata_with_quantization_hash, write_hfq, HfqTensor,
};
use crate::quant_plan::{kmap_resolve_mode, GgufFormat, QuantLevel};
use hipfire_gguf as gguf_input;
use hipfire_primitives::conv::f32_slice_to_f16_bytes;
use hipfire_primitives::fwht::gen_fwht_signs;
use hipfire_quant_format::QuantType;
use std::collections::HashMap;
use std::path::Path;

/// Read `--arch-id <u32>` from `std::env::args` if present. Used by
/// both the GGUF and safetensors entry paths to override the
/// auto-detected `arch_id` stamped into the HFQ header.
///
/// Why an override exists: the auto-detection maps every Qwen2 input
/// to `arch_id=1`, which the daemon dispatches through
/// `hipfire-arch-llama`. That loader doesn't read Q/K/V proj bias,
/// so a Qwen2 model loaded by default would produce wrong outputs.
/// Plain Qwen2 should be `arch_id=7` (hipfire-arch-qwen2) and Qwen2-VL
/// family (dots.ocr) should be `arch_id=8` (hipfire-arch-dots-ocr).
/// See docs/architecture-ids.md and docs/plans/
/// dots-ocr-devlog.md §7 (R1).
pub fn parse_arch_id_override() -> Option<u32> {
    let args: Vec<String> = std::env::args().collect();
    let pos = args.iter().position(|a| a == "--arch-id")?;
    let raw = args.get(pos + 1).unwrap_or_else(|| {
        eprintln!("error: --arch-id requires a u32 value");
        std::process::exit(1);
    });
    match raw.parse::<u32>() {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!("error: --arch-id value '{raw}' is not a valid u32: {e}");
            std::process::exit(1);
        }
    }
}

/// True if the path points to a `.gguf` file on disk.
pub fn is_gguf_input(p: &Path) -> bool {
    p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("gguf")
}

/// True if the GGUF tensor's name is a 1D norm / RMSNorm scaling vector.
/// These stay F16 in the .hfq (no benefit from quantization, precision-sensitive).
pub fn gguf_is_norm_tensor(name: &str) -> bool {
    name.contains("_norm") || name.contains("norm.weight")
}

/// True if the tensor is the token embedding. We Q8 these (matches the
/// safetensors path's `is_embed` rule — Q4 is too lossy for embedding tables).
pub fn gguf_is_embed_tensor(name: &str) -> bool {
    name == "token_embd.weight"
}

/// Convert a GGUF file to a hipfire `.hfq`. Per-format quantization target
/// applies to 2D weight matrices; the embedding table is always Q8F16
/// (Q4-grade is too lossy for embeddings) and 1D norms stay F16. Tensor
/// names are translated GGUF → safetensors style so the engine's existing
/// `load_weights_hfq` can consume the output.
pub fn run_gguf_pipeline(
    input: &Path,
    output: &Path,
    format: GgufFormat,
    format_label: &str,
    no_kmap: bool,
    kmap_dense: bool,
    kmap_mode: u8,
) -> std::io::Result<()> {
    eprintln!("=== GGUF → {} conversion ===", format.label());
    eprintln!("Input:  {}", input.display());
    eprintln!("Output: {}", output.display());

    let gguf = gguf_input::GgufFile::open(input)?;
    eprintln!("GGUF version: {}", gguf.version);
    eprintln!("Tensors: {}", gguf.tensors.len());

    let arch_str = gguf
        .meta_str("general.architecture")
        .unwrap_or("llama")
        .to_string();
    let auto_arch_id: u32 = match arch_str.as_str() {
        "llama" => 0,
        "qwen3" | "qwen2" => 1,
        "qwen3moe" => 6,
        other => {
            eprintln!("warning: unknown GGUF architecture '{other}', tagging as llama-compatible");
            0
        }
    };
    // --arch-id <u32> overrides the auto-detected id. Use when the
    // model's family maps to a different crate than the default
    // (e.g. plain Qwen2 → arch_id=7 for the hipfire-arch-qwen2 crate
    // instead of the LLaMA-family default 1, which silently drops
    // Q/K/V bias on the LLaMA loader path). See docs/plans/
    // dots-ocr-devlog.md §7 (R1) for the bring-up context.
    let arch_id: u32 = parse_arch_id_override().unwrap_or(auto_arch_id);
    if arch_id != auto_arch_id {
        eprintln!("Architecture: {arch_str} (auto id={auto_arch_id}, overridden via --arch-id to {arch_id})");
    } else {
        eprintln!("Architecture: {arch_str} (id={arch_id})");
    }

    // Metadata JSON: must populate `config.*` so engine's `config_from_hfq`
    // can reconstruct LlamaConfig at load time. Also keep the raw GGUF
    // metadata tree under `gguf_meta` for any consumer that wants original
    // values (chat template, vocab, scores, merges, etc.).
    let config_json = gguf_input::config_json_from_gguf(&gguf, &arch_str);
    let mut metadata = serde_json::json!({
        "architecture": arch_str,
        "source": "gguf",
        "quant_format": format_label,
        "config": config_json,
        "gguf_meta": gguf_input::gguf_meta_to_json(&gguf.metadata),
    });

    // FWHT signs — only used by MQ/MFP formats. Same seed pair as the
    // safetensors path so the engine's runtime FWHT inverse stays identical.
    let needs_signs = matches!(
        format,
        GgufFormat::Mq4
            | GgufFormat::Mq6
            | GgufFormat::Mq3
            | GgufFormat::Mq2
            | GgufFormat::Mq2Lloyd
            | GgufFormat::Mq3Lloyd
            | GgufFormat::Mq4Lloyd
            | GgufFormat::Mfp4
    );
    let signs1 = if needs_signs {
        gen_fwht_signs(42, 256)
    } else {
        Vec::new()
    };
    let signs2 = if needs_signs {
        gen_fwht_signs(1042, 256)
    } else {
        Vec::new()
    };

    // K-map setup for GGUF path
    let is_moe = arch_id == 6;
    let n_layers: usize = config_json
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    // Build K-map using translated (safetensors-style) names where available,
    // falling back to raw GGUF names for untranslated tensors.
    //
    // K-map is gated to MoE models only. On dense models the author's own
    // bench shows a mixed picture (PPL +1.5% to +2.5% at 2K context on 4B
    // and 27B; PPL -4.8% on 27B at 8K context — crossover at ~3K). The
    // ship-default is the conservative shape per maintainer directive
    // (2026-05-08): never silently change dense quantization. Users who
    // want K-map on dense pass `--kmap-dense` (see flag parsing below).
    let kmap: HashMap<String, QuantLevel> =
        if format == GgufFormat::F16 || no_kmap || (!is_moe && !kmap_dense) {
            HashMap::new()
        } else {
            let mut map = HashMap::new();
            let mut counts = [0u32; 4];
            for info in &gguf.tensors {
                let out_name =
                    gguf_input::gguf_to_safetensors_name(&info.name).unwrap_or_else(|| info.name.clone());
                let level = kmap_resolve_mode(&out_name, n_layers, is_moe, kmap_mode);
                match level {
                    QuantLevel::F16 => counts[0] += 1,
                    QuantLevel::Q8 => counts[1] += 1,
                    QuantLevel::Promote6 => counts[2] += 1,
                    QuantLevel::Override(_) => counts[3] += 1,
                    QuantLevel::Base => counts[3] += 1,
                }
                map.insert(out_name, level);
            }
            if !map.is_empty() {
                let mode_label = match kmap_mode {
                    0 => "full",
                    1 => "alternating",
                    2 => "typed",
                    _ => "?",
                };
                eprintln!(
                    "K-map plan ({} base, {n_layers} layers{}, mode={mode_label}):",
                    format.label(),
                    if is_moe { ", MoE" } else { "" }
                );
                eprintln!("  F16:       {:>4} tensors", counts[0]);
                eprintln!("  Q8:        {:>4} tensors", counts[1]);
                eprintln!("  Promote6:  {:>4} tensors", counts[2]);
                eprintln!("  Base:      {:>4} tensors", counts[3]);
            }
            map
        };

    let mut hfq_tensors: Vec<HfqTensor> = Vec::with_capacity(gguf.tensors.len());
    let mut total_params: u64 = 0;
    let mut quant_params: u64 = 0;
    let mut total_bytes_in: u64 = 0;
    let mut total_bytes_out: u64 = 0;

    for info in &gguf.tensors {
        let raw = gguf.tensor_data(info);
        let n_elements = info.numel();
        total_params += n_elements as u64;
        total_bytes_in += raw.len() as u64;

        let shape: Vec<u32> = info.shape.iter().map(|&s| s as u32).collect();

        // Tensor classification (uses the original GGUF name).
        let is_norm = gguf_is_norm_tensor(&info.name);
        let is_embed = gguf_is_embed_tensor(&info.name);
        let is_2d = info.shape.len() == 2;
        let k_dim = if is_2d { info.shape[0] } else { n_elements };

        // Translate to the safetensors-style name `hipfire_runtime::hfq::load_weights_hfq`
        // expects. If we don't have a translation, keep the original name —
        // the future loader can ignore unknown tensors.
        let out_name = gguf_input::gguf_to_safetensors_name(&info.name).unwrap_or_else(|| info.name.clone());

        let kmap_level = kmap.get(&out_name).copied().unwrap_or(QuantLevel::Base);

        let (data, quant_type, group_size, label) = if format == GgufFormat::F16
            || is_norm
            || !is_2d
        {
            // Full-F16 artifacts, norms, and 1D tensors use raw half precision.
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let f16_bytes = f32_slice_to_f16_bytes(&f32_data);
            (f16_bytes, QuantType::F16, 0u32, "F16")
        } else if kmap_level == QuantLevel::Q8 || is_embed {
            // K-map Q8 or embedding
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let q = quantize_q8f16(&f32_data);
            quant_params += n_elements as u64;
            (q, QuantType::Q8F16, 32u32, "Q8_F16")
        } else if kmap_level == QuantLevel::Promote6 && k_dim % 256 == 0 {
            // K-map promote to 6-bit
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            match format {
                GgufFormat::Mq4
                | GgufFormat::Mq3
                | GgufFormat::Mq2
                | GgufFormat::Mq2Lloyd
                | GgufFormat::Mq3Lloyd
                | GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Hfq4 | GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Hfp4 => {
                    // No HFP6 variant in v1. Promote6 for HFP4 stays at HFP4G32 (4.25 bpw).
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    // No MFP6 variant. Promote6 for MFP4 stays at MFP4G32 (4.25 bpw).
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
                GgufFormat::Mq4Lloyd => {
                    // Promote6 -> MQ6, consistent with default_promote_target
                    // (Mq4Lloyd -> Mq6) and the Lloyd siblings above.
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::F16 | GgufFormat::Bf16 => {
                    let f16_bytes = f32_slice_to_f16_bytes(&f32_data);
                    (f16_bytes, QuantType::F16, 0u32, "F16")
                }
            }
        } else if let (QuantLevel::Override(override_fmt), true) = (kmap_level, k_dim % 256 == 0) {
            // K-map says override (lm_head when --lm-head-format set).
            // GGUF pipeline has no AWQ wiring (AWQ is safetensors-only today),
            // so this is a plain quantize on the carried target format.
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            match override_fmt {
                GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Mq4 => {
                    let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                }
                GgufFormat::Hfq4 => {
                    let q = quantize_hfq4g256(&f32_data);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                }
                GgufFormat::Mq3 => {
                    let q = quantize_mq3g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                }
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
                GgufFormat::Hfp4 => {
                    let m = info.shape[0] as usize;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k_dim);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    let m = info.shape[0] as usize;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
                GgufFormat::F16 | GgufFormat::Bf16 => {
                    let f16_bytes = f32_slice_to_f16_bytes(&f32_data);
                    (f16_bytes, QuantType::F16, 0u32, "F16")
                }
            }
        } else if k_dim % 256 == 0 {
            // 256-aligned 2D weight — quantize per the chosen format (Base level).
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            match format {
                GgufFormat::Hfq4 => {
                    let q = quantize_hfq4g256(&f32_data);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                }
                GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Mq4 => {
                    let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                }
                GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Mq3 => {
                    let q = quantize_mq3g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                }
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
                GgufFormat::Hfp4 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
                GgufFormat::F16 | GgufFormat::Bf16 => {
                    let f16_bytes = f32_slice_to_f16_bytes(&f32_data);
                    (f16_bytes, QuantType::F16, 0u32, "F16")
                }
            }
        } else {
            // K not divisible by 256 — fall back to HFQ4-G128 (no rotation).
            // This branch fires for the rare ragged dim; ignores --format
            // (no G128 variant of mq4/mq6 exists).
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let q = quantize_hfq4g128(&f32_data);
            quant_params += n_elements as u64;
            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
        };

        total_bytes_out += data.len() as u64;
        eprintln!(
            "  {label:>9}: {} → {} {:?} ({} src={:?}, {:.1} KB → {:.1} KB)",
            info.name,
            out_name,
            info.shape,
            n_elements,
            info.dtype,
            raw.len() as f64 / 1024.0,
            data.len() as f64 / 1024.0,
        );

        hfq_tensors.push(HfqTensor {
            name: out_name,
            quant_type,
            shape,
            group_size,
            data,
            spilled_len: 0,
        });
    }

    eprintln!("\n=== GGUF → MQ4 Summary ===");
    eprintln!("  Tensors:        {}", hfq_tensors.len());
    eprintln!("  Total params:   {total_params}");
    eprintln!(
        "  Quant'd params: {quant_params} ({:.1}%)",
        100.0 * quant_params as f64 / total_params as f64
    );
    eprintln!("  Input size:     {:.1} MB", total_bytes_in as f64 / 1e6);
    eprintln!(
        "  Output size:    {:.1} MB ({:.1}% of input)",
        total_bytes_out as f64 / 1e6,
        100.0 * total_bytes_out as f64 / total_bytes_in as f64,
    );

    insert_parameter_counts_metadata(&mut metadata, &hfq_tensors, total_params, quant_params, 0);
    let metadata_json = metadata_with_quantization_hash(metadata, &hfq_tensors, None)?;
    write_hfq(output, arch_id, &metadata_json, &hfq_tensors, None)?;
    eprintln!("\nWrote: {}", output.display());
    Ok(())
}
