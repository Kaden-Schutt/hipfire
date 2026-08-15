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
use crate::hfq::*;
use crate::pipeline_gguf::{dequantize_hfq_q8f16, GgufFormat};
use crate::calibration::*;
use crate::model_filter::*;
use crate::quant_e8::*;
use crate::quant_mq::*;
use crate::quant_fwht::*;

pub(crate) fn build_deepseek4_dense_e8soa_overlay(input: &Path, output: &Path) -> Result<(), String> {
    let mut hfq = hipfire_runtime::hfq::HfqFile::open(input)
        .map_err(|e| format!("open source HFQ {}: {e}", input.display()))?;
    if hfq.arch_id != 9 {
        return Err(format!(
            "deepseek4 dense E8 overlay requires arch_id=9, got {}",
            hfq.arch_id
        ));
    }
    let metadata_json = hfq.metadata_json.clone();
    let metadata: serde_json::Value = serde_json::from_str(&metadata_json)
        .map_err(|e| format!("source HFQ metadata JSON: {e}"))?;
    let n_layers = metadata
        .pointer("/config/num_hidden_layers")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| "source HFQ metadata missing config.num_hidden_layers".to_string())?
        as usize;
    hfq.drop_mmap();

    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);
    let suffixes = [
        "attn.wq_a.weight",
        "attn.wq_b.weight",
        "attn.wkv.weight",
        "attn.wo_a.weight",
        "attn.wo_b.weight",
        "ffn.shared_experts.w1.weight",
        "ffn.shared_experts.w2.weight",
        "ffn.shared_experts.w3.weight",
    ];

    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create output dir {}: {e}", parent.display()))?;
    }
    let spill_dir = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut spill = TensorSpill::new(spill_dir).map_err(|e| format!("create tensor spill: {e}"))?;
    let mut tensors = Vec::with_capacity(n_layers * suffixes.len());
    let mut source_bytes = 0u64;
    let mut overlay_bytes = 0u64;

    for layer in 0..n_layers {
        for suffix in suffixes {
            let name = format!("layers.{layer}.{suffix}");
            let (info, bytes) = hfq
                .tensor_data_vec(&name)
                .ok_or_else(|| format!("source HFQ missing dense projection '{name}'"))?;
            if info.quant_type != QuantType::Q8F16 as u8 {
                return Err(format!(
                    "{name}: expected Q8F16 qt=3 source, got qt={}",
                    info.quant_type
                ));
            }
            if info.shape.len() != 2 {
                return Err(format!(
                    "{name}: expected rank-2 shape, got {:?}",
                    info.shape
                ));
            }
            let m = info.shape[0] as usize;
            let k = info.shape[1] as usize;
            if k % 256 != 0 {
                return Err(format!("{name}: E8-SoA requires K%256==0, got K={k}"));
            }
            let f32_data =
                dequantize_hfq_q8f16(&bytes, m * k).map_err(|e| format!("{name}: {e}"))?;
            let packed = quantize_mfp4g32_e8_soa_2d(&f32_data, m, k, &signs1, &signs2);
            source_bytes += bytes.len() as u64;
            overlay_bytes += packed.len() as u64;
            eprintln!(
                "E8-SoA {name}: [{m}, {k}] {:.2} MiB -> {:.2} MiB",
                bytes.len() as f64 / 1_048_576.0,
                packed.len() as f64 / 1_048_576.0
            );
            tensors.push(HfqTensor {
                name,
                quant_type: QuantType::MFP4G32E8SOA,
                shape: info.shape.clone(),
                group_size: 32,
                data: packed,
                spilled_len: 0,
            });
            maybe_spill(&mut tensors, &mut spill, 64 * 1024 * 1024);
        }
    }

    write_hfq(
        output,
        hfq.arch_id,
        &metadata_json,
        &tensors,
        Some(&mut spill),
    )
    .map_err(|e| format!("write overlay {}: {e}", output.display()))?;
    eprintln!(
        "deepseek4 dense E8-SoA overlay: {} tensors, {:.2} GiB Q8 -> {:.2} GiB E8 ({:.1}% of source)",
        tensors.len(),
        source_bytes as f64 / 1_073_741_824.0,
        overlay_bytes as f64 / 1_073_741_824.0,
        overlay_bytes as f64 * 100.0 / source_bytes as f64,
    );
    Ok(())
}

/// Re-quantize a DeepSeek V4 DSpark/MTP sidecar's DENSE projections from Q8F16
/// to MFP4-E8-SoA so the drafter MATCHES its MQ2R trunk's recipe.
///
/// Why: a drafter must predict what the TRUNK emits, not what the original
/// checkpoint would emit. `deepseek4-q8-mtp` ships the sidecar at Q8F16
/// (see the tier selection near `use_deepseek4_source_precision`), while an
/// MQ2R trunk is qt=35 MFP4G32E8SOA dense + qt=19 MQ2-Lloyd experts. That
/// leaves the draft 2-4x HIGHER precision than its target, so wherever the
/// trunk's quantization moves the argmax the draft confidently predicts the
/// un-quantized token and is rejected — systematically right about the wrong
/// model. Matching the recipes makes both share the same quantization error.
/// This is why DFlash's MQ4 drafts work against MQ4 targets: matched, not
/// merely cheap. Measured context: a DS4 draft stage weighs 1.05x a trunk
/// layer (2.00 GB vs 1.91 GB), so "small drafter, preserve precision" — the
/// rationale behind the Q8F16 tier — does not hold here.
///
/// Unlike `build_deepseek4_dense_e8soa_overlay` this emits a COMPLETE sidecar,
/// not a shadow overlay: converted dense tensors plus every other tensor
/// copied through byte-for-byte. Routed experts are deliberately untouched —
/// they are already MQ2-Lloyd, matching the trunk, and the MoE GEMV kernel
/// handles only that format.
///
/// Also stamps the `mq2r_sidecar` identity that
/// `DeepseekV4::validate_mq2r_dspark_sidecar` requires, so the artifact is
/// born valid instead of being patched afterwards by
/// `scripts/reap/hfq_metadata_stamp.rs`.
pub(crate) fn build_deepseek4_dspark_e8soa_sidecar(input: &Path, output: &Path) -> Result<(), String> {
    let mut hfq = hipfire_runtime::hfq::HfqFile::open(input)
        .map_err(|e| format!("open source sidecar {}: {e}", input.display()))?;
    if hfq.arch_id != 9 {
        return Err(format!(
            "deepseek4 DSpark E8 sidecar requires arch_id=9, got {}",
            hfq.arch_id
        ));
    }
    let metadata_json = hfq.metadata_json.clone();

    // Dense, per-token projections inside each `mtp.{stage}.*` block — the same
    // suffix set the trunk overlay converts, which is exactly the set that goes
    // through `gemv_auto` in the draft forward. `ffn.experts.` is NOT here.
    pub(crate) const DENSE_SUFFIXES: [&str; 8] = [
        "attn.wq_a.weight",
        "attn.wq_b.weight",
        "attn.wkv.weight",
        "attn.wo_a.weight",
        "attn.wo_b.weight",
        "ffn.shared_experts.w1.weight",
        "ffn.shared_experts.w2.weight",
        "ffn.shared_experts.w3.weight",
    ];
    let is_dense_target = |name: &str| -> bool {
        name.starts_with("mtp.")
            && !name.contains(".ffn.experts.")
            && DENSE_SUFFIXES.iter().any(|s| name.ends_with(s))
    };

    let names: Vec<String> = hfq.tensors().iter().map(|t| t.name.clone()).collect();
    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);

    let spill_dir = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut spill = TensorSpill::new(spill_dir).map_err(|e| format!("create tensor spill: {e}"))?;
    let mut tensors: Vec<HfqTensor> = Vec::with_capacity(names.len());
    let (mut n_conv, mut src_b, mut dst_b, mut n_skip) = (0usize, 0u64, 0u64, 0usize);

    for name in &names {
        let (info, bytes) = hfq
            .tensor_data_vec(name)
            .ok_or_else(|| format!("source sidecar missing tensor '{name}'"))?;
        let shape = info.shape.clone();
        let qt = info.quant_type;
        let gs = info.group_size;

        // Convert only rank-2 Q8F16 dense projections with K%256==0; anything
        // else (experts, norms, HC, ragged K) passes through untouched.
        let convertible = is_dense_target(name)
            && qt == QuantType::Q8F16 as u8
            && shape.len() == 2
            && (shape[1] as usize) % 256 == 0;
        if !convertible {
            if is_dense_target(name) {
                n_skip += 1;
                eprintln!("  passthrough (not convertible): {name} qt={qt} shape={shape:?}");
            }
            tensors.push(HfqTensor {
                name: name.clone(),
                quant_type: QuantType::from_u8(qt)
                    .ok_or_else(|| format!("{name}: unknown source qt={qt}"))?,
                shape,
                group_size: gs,
                data: bytes,
                spilled_len: 0,
            });
            maybe_spill(&mut tensors, &mut spill, 64 * 1024 * 1024);
            continue;
        }

        let m = shape[0] as usize;
        let k = shape[1] as usize;
        let f32_data = dequantize_hfq_q8f16(&bytes, m * k).map_err(|e| format!("{name}: {e}"))?;
        let packed = quantize_mfp4g32_e8_soa_2d(&f32_data, m, k, &signs1, &signs2);
        src_b += bytes.len() as u64;
        dst_b += packed.len() as u64;
        n_conv += 1;
        eprintln!(
            "E8-SoA {name}: [{m}, {k}] {:.2} MiB -> {:.2} MiB",
            bytes.len() as f64 / 1_048_576.0,
            packed.len() as f64 / 1_048_576.0
        );
        tensors.push(HfqTensor {
            name: name.clone(),
            quant_type: QuantType::MFP4G32E8SOA,
            shape,
            group_size: 32,
            data: packed,
            spilled_len: 0,
        });
        maybe_spill(&mut tensors, &mut spill, 64 * 1024 * 1024);
    }

    if n_conv == 0 {
        return Err(
            "no convertible mtp.* dense Q8F16 projections found — is this a DSpark sidecar?"
                .to_string(),
        );
    }
    if tensors.iter().any(|t| t.name == "draft_head.weight") {
        return Err(
            "sidecar carries draft_head.weight, which validate_mq2r_dspark_sidecar forbids"
                .to_string(),
        );
    }

    // Stamp the identity the loader enforces (arch.rs validate_mq2r_dspark_sidecar)
    // so this artifact is born valid rather than metadata-patched after the fact.
    let mut meta: serde_json::Value = serde_json::from_str(&metadata_json)
        .map_err(|e| format!("source sidecar metadata JSON: {e}"))?;
    if let Some(obj) = meta.as_object_mut() {
        obj.insert(
            "mq2r_sidecar".to_string(),
            serde_json::json!({
                "target_recipe": "deepseek4-mq2r-e8-p3-v1",
                "draft_head": "trunk_mfp4_e8_soa_b4",
                "dense_tier": "MFP4G32E8SOA",
                "built_by": "deepseek4-dspark-e8soa",
            }),
        );
    } else {
        return Err("source sidecar metadata is not a JSON object".to_string());
    }
    let out_meta = serde_json::to_string(&meta).map_err(|e| format!("re-encode metadata: {e}"))?;

    write_hfq(output, hfq.arch_id, &out_meta, &tensors, Some(&mut spill))
        .map_err(|e| format!("write sidecar {}: {e}", output.display()))?;
    eprintln!(
        "deepseek4 DSpark E8-SoA sidecar: {} tensors total, {n_conv} dense converted \
         ({n_skip} dense passed through), {:.2} GiB Q8 -> {:.2} GiB E8 ({:.1}% of converted source)",
        tensors.len(),
        src_b as f64 / 1_073_741_824.0,
        dst_b as f64 / 1_073_741_824.0,
        dst_b as f64 * 100.0 / src_b.max(1) as f64,
    );
    Ok(())
}
