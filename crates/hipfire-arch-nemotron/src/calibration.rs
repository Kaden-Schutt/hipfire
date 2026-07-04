// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
//! nemotron_h calibration: full-[K,K] Hessian + imatrix sidecar (+ optional
//! KLDREF) for `oq4+/oq4++/oq8+/oq8++` quantization. Thin arch driver over the
//! shared [`hipfire_runtime::calibration::collect`]: register each dense
//! projection's buffer → HFQ name ([`NemotronModel::build_capture_names`]), run
//! the capturing per-token decode forward (the `gpu.maybe_capture_activation`
//! taps fire inside `LinearWeight::gemv`), and the general driver streams the
//! HFQM `.calib.hfq`. Dense Nano-4B gets full Hessians on all linears; routed
//! MoE experts (Nano-30B, matched by `.experts.`) are imatrix-only — a follow-on
//! (currently skipped by `build_capture_names`).

use crate::model::NemotronModel;
use hip_bridge::HipResult;
use hipfire_runtime::calibration::{collect, logsumexp, topk_logits, CalibForward, CalibSummary};
use hipfire_runtime::hfq::HfqMemTensor;
use hipfire_rdna::Gpu;
use std::path::Path;

/// Calibration knobs. `kldref` taps the lm-head logits during the forward and
/// bakes the per-position `(logZ, top-k)` reference into the sidecar.
pub struct CalibOpts {
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

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for &x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// Pack the per-position `(logZ, top-k)` KLDREF reference into HFQ tensors
/// (`lm_head.kldref_{idx,logit,logz}`), matching the other arches' layout.
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

/// Per-token decode forward over `tokens` (the capture taps fire in
/// `LinearWeight::gemv`). With `kldref_topk`, also capture the per-position
/// `(logZ, top-k)` lm-head reference.
fn forward_calib(
    gpu: &mut Gpu,
    model: &mut NemotronModel,
    tokens: &[u32],
    kldref_topk: Option<usize>,
) -> HipResult<Vec<(f32, Vec<(u32, f32)>)>> {
    model.reset(gpu)?;
    let mut kldref = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        model.forward_gpu(gpu, tok, pos)?;
        if let Some(k) = kldref_topk {
            let lg = gpu.download_f32(model.logits_tensor())?;
            kldref.push((logsumexp(&lg), topk_logits(&lg, k)));
        }
    }
    Ok(kldref)
}

/// Run the calibration forward over `tokens` and write the HFQM Hessian/imatrix
/// sidecar to `output`. `provenance` is folded into the sidecar metadata.
pub fn collect_calibration_artifacts(
    gpu: &mut Gpu,
    model: &mut NemotronModel,
    tokens: &[u32],
    opts: &CalibOpts,
    output: &Path,
    provenance: &[(&str, serde_json::Value)],
) -> Result<CalibSummary, String> {
    if tokens.is_empty() {
        return Err("nemotron calib: empty calibration corpus".to_string());
    }
    let mut static_meta: Vec<(&str, serde_json::Value)> = provenance.to_vec();
    static_meta.push(("text_only", serde_json::json!(true)));
    let capture_names = model.build_capture_names();
    let kldref_topk = opts.kldref.then_some(opts.kldref_topk);
    collect(
        gpu,
        hipfire_model::ARCH_ID_NEMOTRON_H,
        capture_names,
        vec![".experts.".to_string()], // routed MoE experts: imatrix-only
        output,
        &static_meta,
        |gpu| {
            let kldref =
                forward_calib(gpu, model, tokens, kldref_topk).map_err(|e| e.to_string())?;
            let extra_tensors = kldref_extra(&kldref);
            let mut extra_meta: Vec<(String, serde_json::Value)> = Vec::new();
            if !kldref.is_empty() {
                let np = kldref.len();
                let kk = kldref[0].1.len();
                extra_meta.push((
                    "kldref".to_string(),
                    serde_json::json!({ "n_positions": np, "top_k": kk }),
                ));
                extra_meta.push((
                    "artifacts".to_string(),
                    serde_json::json!(["hessian", "imatrix", "kldref"]),
                ));
            }
            Ok(CalibForward {
                extra_tensors,
                extra_meta,
            })
        },
    )
}
