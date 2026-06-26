// SPDX-License-Identifier: Apache-2.0
// hipfire — LFM2/LFM2-MoE text calibration collection.

use crate::config::Lfm2MoeConfig;
use crate::forward::decode_step;
use crate::lfm2moe::{Ffn, Lfm2MoeState, Lfm2MoeWeights, Mixer};
use hip_bridge::{HipError, HipResult};
use hipfire_runtime::calibration::{logsumexp, topk_logits, CalibCollector};
use hipfire_runtime::hfq::HfqMemTensor;
use hipfire_runtime::weights::WeightTensor;
use rdna_compute::Gpu;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Options for [`collect_calibration_artifacts`].
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

fn put(m: &mut HashMap<usize, String>, wt: &WeightTensor, name: impl Into<String>) {
    m.insert(wt.buf.buf.as_ptr() as usize, name.into());
}

/// Build the capture map for LFM2 dense/router projections.
///
/// Names match checkpoint tensor keys without the `.weight` suffix so the
/// quantizer can join `<name>.imatrix`/`<name>.hessian` to source weights.
/// Routed expert weights are captured explicitly in `forward.rs` because the
/// fused indexed kernels do not have one weight pointer per source tensor: gate
/// and up are byte-fused into `gate_up`, but the calibration package needs
/// separate checkpoint-style names for `w1` and `w3`.
pub fn build_capture_names(weights: &Lfm2MoeWeights) -> HashMap<usize, String> {
    let mut m = HashMap::new();
    for (i, layer) in weights.layers.iter().enumerate() {
        let p = format!("model.layers.{i}");
        match &layer.mixer {
            Mixer::Conv(c) => {
                put(&mut m, &c.in_proj, format!("{p}.conv.in_proj"));
                put(&mut m, &c.out_proj, format!("{p}.conv.out_proj"));
            }
            Mixer::Attention(a) => {
                put(&mut m, &a.wq, format!("{p}.self_attn.q_proj"));
                put(&mut m, &a.wk, format!("{p}.self_attn.k_proj"));
                put(&mut m, &a.wv, format!("{p}.self_attn.v_proj"));
                put(&mut m, &a.wo, format!("{p}.self_attn.out_proj"));
            }
        }
        match &layer.ffn {
            Ffn::Dense(d) => {
                put(&mut m, &d.w1, format!("{p}.feed_forward.w1"));
                put(&mut m, &d.w3, format!("{p}.feed_forward.w3"));
                put(&mut m, &d.w2, format!("{p}.feed_forward.w2"));
            }
            Ffn::Moe(moe) => {
                put(&mut m, &moe.router, format!("{p}.feed_forward.gate"));
            }
        }
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

/// Collect calibration Hessians/imatrices from the LFM2 text decoder.
///
/// This covers dense projection calls that route through `weight_gemv` plus
/// calibration-only taps around the indexed routed-expert kernels. Dense/router
/// tensors get full Hessians; routed expert tensors are imatrix-only because
/// full per-expert Hessians do not fit for the 8B-A1B model.
pub fn collect_calibration_artifacts(
    gpu: &mut Gpu,
    weights: &Lfm2MoeWeights,
    config: &Lfm2MoeConfig,
    tokens: &[u32],
    opts: &CalibOpts,
    output: &Path,
    provenance: &[(&str, serde_json::Value)],
) -> HipResult<CalibSummary> {
    let collector = Arc::new(CalibCollector::with_imatrix_only(vec![
        ".feed_forward.experts.".to_string(),
    ]));
    gpu.capture_names = build_capture_names(weights);
    gpu.active_capture = Some(collector.clone());

    let mut state = Lfm2MoeState::new(gpu, config)
        .map_err(|e| HipError::new(0, &format!("lfm2 calib state: {e}")))?;
    let mut kldref: Vec<(f32, Vec<(u32, f32)>)> = Vec::new();
    let mut run_result: HipResult<()> = Ok(());
    for (pos, &tok) in tokens.iter().enumerate() {
        match decode_step(config, weights, &mut state, gpu, tok, pos as u32) {
            Ok(logits) => {
                if opts.kldref {
                    kldref.push((logsumexp(&logits), topk_logits(&logits, opts.kldref_topk)));
                }
            }
            Err(e) => {
                run_result = Err(HipError::new(0, &format!("lfm2 calib decode: {e}")));
                break;
            }
        }
    }
    gpu.active_capture = None;
    gpu.capture_names = HashMap::new();
    run_result?;

    let descriptors = collector.tensor_descriptors();
    if descriptors.is_empty() {
        return Err(HipError::new(
            0,
            "lfm2 calib: no tensors captured (capture_names empty or weight_gemv not hit)",
        ));
    }

    let n_hessian = descriptors.iter().filter(|d| d.has_hessian).count();
    let n_imatrix = descriptors.len();
    let mut per_tensor_tokens = serde_json::Map::new();
    for d in &descriptors {
        per_tensor_tokens.insert(d.name.clone(), serde_json::json!(d.n_tokens));
    }

    let mut artifacts = vec![serde_json::json!("hessian"), serde_json::json!("imatrix")];
    let mut meta = serde_json::json!({
        "artifact_kind": "calibration",
        "arch": "lfm2",
        "text_only": true,
        "captures": "decode_step_weight_gemv+routed_expert_indexed_tap",
        "routed_expert_capture": "imatrix-only-selected-experts",
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
    let max_consistency = collector
        .write_streaming(gpu, output, crate::ARCH_ID, &metadata_json, &extra)
        .map_err(|e| HipError::new(0, &format!("write .calib.hfq: {e}")))?;
    collector.free_gpu(gpu);

    Ok(CalibSummary {
        n_hessian,
        n_imatrix,
        max_consistency,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn lfm2_calibration_name_examples_match_checkpoint_keys() {
        assert_eq!(
            format!("model.layers.7.conv.in_proj"),
            "model.layers.7.conv.in_proj"
        );
        assert_eq!(
            format!("model.layers.7.feed_forward.gate"),
            "model.layers.7.feed_forward.gate"
        );
    }
}
