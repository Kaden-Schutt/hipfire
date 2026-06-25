// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3 text-decoder calibration collection. See LICENSE / NOTICE.

use hip_bridge::{HipError, HipResult};
use hipfire_runtime::calibration::{logsumexp, topk_logits, CalibCollector};
use hipfire_runtime::hfq::{
    write_hfqm_package_streaming, HfqMemTensor, HfqPackage, HfqStreamEntry,
};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weights::WeightTensor;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};

use crate::config::Gemma3Config;
use crate::forward::{forward_step, Gemma3State};
use crate::weights::Gemma3Weights;

/// Options for [`collect_calibration_artifacts_text_only`].
pub struct CalibOpts {
    /// Capture the lm-head top-K logits + logZ per position (KLDREF reference).
    pub kldref: bool,
    pub kldref_topk: usize,
}

impl Default for CalibOpts {
    fn default() -> Self {
        Self {
            kldref: false,
            kldref_topk: 64,
        }
    }
}

/// Summary of a calibration pass after the `.calib.hfq` has been streamed.
pub struct CalibSummary {
    pub n_hessian: usize,
    pub n_imatrix: usize,
    pub max_consistency: f32,
}

fn put(
    m: &mut std::collections::HashMap<usize, String>,
    wt: &WeightTensor,
    name: impl Into<String>,
) {
    m.insert(wt.buf.buf.as_ptr() as usize, name.into());
}

/// Build the calibration capture map for the Gemma3 text decoder.
///
/// Names match the source HFQ tensor keys without the `.weight` suffix. Pure
/// text Gemma3 uses `prefix=""`; Gemma3-VL text-only collection uses
/// `prefix="language_model."`.
pub fn build_capture_names(
    weights: &Gemma3Weights,
    prefix: &str,
) -> std::collections::HashMap<usize, String> {
    build_capture_names_for_layers(weights, prefix, 0, weights.layers.len())
}

fn build_capture_names_for_layers(
    weights: &Gemma3Weights,
    prefix: &str,
    start_layer: usize,
    end_layer: usize,
) -> std::collections::HashMap<usize, String> {
    let mut m = std::collections::HashMap::new();
    for (i, layer) in weights
        .layers
        .iter()
        .enumerate()
        .skip(start_layer)
        .take(end_layer.saturating_sub(start_layer))
    {
        let p = format!("{prefix}model.layers.{i}");
        put(&mut m, &layer.wq, format!("{p}.self_attn.q_proj"));
        put(&mut m, &layer.wk, format!("{p}.self_attn.k_proj"));
        put(&mut m, &layer.wv, format!("{p}.self_attn.v_proj"));
        put(&mut m, &layer.wo, format!("{p}.self_attn.o_proj"));
        put(&mut m, &layer.w_gate, format!("{p}.mlp.gate_proj"));
        put(&mut m, &layer.w_up, format!("{p}.mlp.up_proj"));
        put(&mut m, &layer.w_down, format!("{p}.mlp.down_proj"));
    }
    m
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for &x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

fn layers_per_pass() -> usize {
    std::env::var("HIPFIRE_GEMMA3_CALIB_LAYERS_PER_PASS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(4)
}

fn part_path(output: &Path, group_idx: usize) -> PathBuf {
    let file_name = output
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("gemma3.calib.hfq");
    output.with_file_name(format!(".{file_name}.part-{group_idx:03}.hfq"))
}

fn run_text_forward_for_capture(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    config: &Gemma3Config,
    tokens: &[u32],
    opts: &CalibOpts,
    collect_kldref: bool,
    kldref: &mut Vec<(f32, Vec<(u32, f32)>)>,
) -> HipResult<()> {
    let mut state = Gemma3State::new(gpu, config)
        .map_err(|e| HipError::new(0, &format!("gemma3 calib state: {e}")))?;
    let mut result = Ok(());
    for &tok in tokens {
        if let Err(e) = forward_step(gpu, weights, config, &mut state, tok) {
            result = Err(e);
            break;
        }
        if collect_kldref && opts.kldref {
            match gpu.download_f32(&state.logits) {
                Ok(lg) => kldref.push((logsumexp(&lg), topk_logits(&lg, opts.kldref_topk))),
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
    }
    state.free_gpu(gpu);
    result
}

fn kldref_extra(kldref: &[(f32, Vec<(u32, f32)>)]) -> Vec<HfqMemTensor> {
    if kldref.is_empty() {
        return Vec::new();
    }
    let np = kldref.len();
    let kk = kldref[0].1.len();
    let (mut idx_v, mut lg_v, mut lz_v) = (Vec::new(), Vec::new(), Vec::new());
    for (logz, tk) in kldref {
        lz_v.push(*logz);
        for j in 0..kk {
            let (i, l) = tk.get(j).copied().unwrap_or((0, f32::NEG_INFINITY));
            idx_v.push(i as f32);
            lg_v.push(l);
        }
    }
    [
        ("lm_head.kldref_idx", vec![np as u32, kk as u32], idx_v),
        ("lm_head.kldref_logit", vec![np as u32, kk as u32], lg_v),
        ("lm_head.kldref_logz", vec![np as u32], lz_v),
    ]
    .into_iter()
    .map(|(name, shape, data)| HfqMemTensor {
        name: name.to_string(),
        quant_type: 2,
        shape,
        group_size: 0,
        data: f32_bytes(&data),
    })
    .collect()
}

/// Collect calibration Hessians/imatrices from the Gemma3 text decoder only.
///
/// For a Gemma3-VL artifact, pass `prefix="language_model."`; vision/projector
/// tensors are not loaded and cannot be captured.
pub fn collect_calibration_artifacts_text_only(
    gpu: &mut Gpu,
    weights: &Gemma3Weights,
    config: &Gemma3Config,
    _tokenizer: &Tokenizer,
    tokens: &[u32],
    opts: &CalibOpts,
    output: &std::path::Path,
    prefix: &str,
    provenance: &[(&str, serde_json::Value)],
) -> HipResult<CalibSummary> {
    let group = layers_per_pass();
    let mut kldref: Vec<(f32, Vec<(u32, f32)>)> = Vec::new();
    let mut part_paths = Vec::new();
    let mut all_descriptors = Vec::new();
    let mut max_consistency = 0.0f32;

    for (group_idx, start) in (0..config.num_hidden_layers).step_by(group).enumerate() {
        let end = (start + group).min(config.num_hidden_layers);
        eprintln!(
            "gemma3 calib: capturing layers {}..{} of {}",
            start, end, config.num_hidden_layers
        );
        let collector = std::sync::Arc::new(CalibCollector::default());
        gpu.capture_names = build_capture_names_for_layers(weights, prefix, start, end);
        gpu.active_capture = Some(collector.clone());

        let run_result = run_text_forward_for_capture(
            gpu,
            weights,
            config,
            tokens,
            opts,
            group_idx == 0,
            &mut kldref,
        );

        gpu.active_capture = None;
        gpu.capture_names = std::collections::HashMap::new();
        run_result?;

        let descriptors = collector.tensor_descriptors();
        if descriptors.is_empty() {
            return Err(HipError::new(
                0,
                &format!("gemma3 calib: no tensors captured for layers {start}..{end}"),
            ));
        }

        let part = part_path(output, group_idx);
        let part_meta = serde_json::json!({
            "artifact_kind": "calibration-part",
            "text_only": true,
            "text_prefix": prefix,
            "layer_start": start,
            "layer_end": end,
        })
        .to_string();
        let write_result = collector
            .write_streaming(gpu, &part, 0, &part_meta, &[])
            .map_err(|e| HipError::new(0, &format!("write part .calib.hfq: {e}")));
        collector.free_gpu(gpu);
        let consistency = write_result?;
        max_consistency = max_consistency.max(consistency);
        all_descriptors.extend(descriptors);
        part_paths.push(part);
    }

    let n_hessian = all_descriptors.iter().filter(|d| d.has_hessian).count();
    let n_imatrix = all_descriptors.len();
    let mut per_tensor_tokens = serde_json::Map::new();
    for d in &all_descriptors {
        per_tensor_tokens.insert(d.name.clone(), serde_json::json!(d.n_tokens));
    }

    let mut artifacts = vec![serde_json::json!("hessian"), serde_json::json!("imatrix")];
    let mut meta = serde_json::json!({
        "artifact_kind": "calibration",
        "text_only": true,
        "text_prefix": prefix,
        "layers_per_pass": group,
        "n_hessian": n_hessian,
        "n_imatrix": n_imatrix,
        "per_tensor_tokens": serde_json::Value::Object(per_tensor_tokens),
    });

    let extra = kldref_extra(&kldref);
    if !kldref.is_empty() {
        let np = kldref.len();
        let kk = kldref[0].1.len();
        meta.as_object_mut().unwrap().insert(
            "kldref".to_string(),
            serde_json::json!({ "n_positions": np, "top_k": kk }),
        );
        artifacts.push(serde_json::json!("kldref"));
    }

    if let Some(obj) = meta.as_object_mut() {
        obj.insert("artifacts".to_string(), serde_json::Value::Array(artifacts));
        for (k, v) in provenance {
            obj.insert((*k).to_string(), v.clone());
        }
    }
    let metadata_json = serde_json::to_string(&meta).unwrap();
    combine_parts(output, &metadata_json, &part_paths, &extra)
        .map_err(|e| HipError::new(0, &format!("combine .calib.hfq: {e}")))?;
    for part in part_paths {
        let _ = std::fs::remove_file(part);
    }

    Ok(CalibSummary {
        n_hessian,
        n_imatrix,
        max_consistency,
    })
}

fn combine_parts(
    output: &Path,
    metadata_json: &str,
    part_paths: &[PathBuf],
    extra: &[HfqMemTensor],
) -> std::io::Result<()> {
    enum Plan {
        Part { package_idx: usize, name: String },
        Extra { extra_idx: usize },
    }

    let mut packages = Vec::with_capacity(part_paths.len());
    let mut entries = Vec::new();
    let mut plan = Vec::new();
    for part in part_paths {
        let package = HfqPackage::open(part)?;
        let package_idx = packages.len();
        for e in package.entries() {
            entries.push(HfqStreamEntry {
                name: e.name.clone(),
                quant_type: e.quant_type,
                shape: e.shape.clone(),
                group_size: e.group_size,
                data_len: e.data_size as u64,
            });
            plan.push(Plan::Part {
                package_idx,
                name: e.name.clone(),
            });
        }
        packages.push(package);
    }
    for (extra_idx, t) in extra.iter().enumerate() {
        entries.push(HfqStreamEntry {
            name: t.name.clone(),
            quant_type: t.quant_type,
            shape: t.shape.clone(),
            group_size: t.group_size,
            data_len: t.data.len() as u64,
        });
        plan.push(Plan::Extra { extra_idx });
    }

    write_hfqm_package_streaming(output, 0, metadata_json, &entries, |i, w| match &plan[i] {
        Plan::Part { package_idx, name } => {
            let data = packages[*package_idx].blob_data(name).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("part tensor not found: {name}"),
                )
            })?;
            w.write_all(data)
        }
        Plan::Extra { extra_idx } => w.write_all(&extra[*extra_idx].data),
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn capture_names_keep_expected_prefixes() {
        assert_eq!(
            format!("{}model.layers.7.self_attn.q_proj", ""),
            "model.layers.7.self_attn.q_proj"
        );
        assert_eq!(
            format!("{}model.layers.7.mlp.down_proj", "language_model."),
            "language_model.model.layers.7.mlp.down_proj"
        );
    }
}
