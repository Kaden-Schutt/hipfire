// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ZAYA1 calibration: generate the LDLQ full-[K,K] Hessian (+ imatrix) sidecar
//! for `oq4++`/`oq8++` quantization, plus an optional KLDREF reference. Thin
//! arch driver over the shared [`hipfire_runtime::calibration::collect`]:
//! register each linear's buffer → HFQ name (`build_capture_names`), run the
//! capturing forward ([`crate::gpu::gpu_forward_calib`]), and the general driver
//! streams the HFQM `.calib.hfq`.
//!
//! Dense projections (q/k/v/o, router fc/down/out, tied lm_head) get a full
//! Hessian — exactly the LDLQ-eligible set. Routed experts (matched by
//! `.experts.`) are imatrix-only (Σx², for the AWQ `+` scale — no [K,K]).
//! Experts not selected by the corpus under top-1 routing get no imatrix and
//! fall back to RTN. With `--kldref`, the capturing forward also taps the tied
//! lm-head logits and bakes the per-position top-k reference into the sidecar.

use crate::gpu::{build_capture_names, gpu_forward_calib, ZayaGpuWeights};
use crate::{ZayaConfig, ARCH_ID_ZAYA};
use hipfire_runtime::calibration::{collect, CalibForward};
use hipfire_runtime::hfq::HfqMemTensor;
use rdna_compute::Gpu;
use std::path::Path;

pub use hipfire_runtime::calibration::CalibSummary;

/// Calibration knobs. `kldref` taps the tied lm-head logits during the forward
/// and bakes the per-position `(logZ, top-k)` reference into the sidecar.
pub struct CalibOpts {
    pub kldref: bool,
    pub kldref_topk: usize,
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

/// Run the calibration forward over `tokens` and write the HFQM Hessian sidecar
/// to `output`. `provenance` is folded into the sidecar metadata.
pub fn collect_calibration_artifacts(
    gpu: &mut Gpu,
    weights: &ZayaGpuWeights,
    config: &ZayaConfig,
    tokens: &[u32],
    opts: &CalibOpts,
    output: &Path,
    provenance: &[(&str, serde_json::Value)],
) -> Result<CalibSummary, String> {
    if tokens.is_empty() {
        return Err("zaya calib: empty calibration corpus".to_string());
    }
    let mut static_meta: Vec<(&str, serde_json::Value)> = provenance.to_vec();
    static_meta.push(("text_only", serde_json::json!(true)));
    collect(
        gpu,
        ARCH_ID_ZAYA,
        build_capture_names(weights),
        vec![".experts.".to_string()],
        output,
        &static_meta,
        |gpu| {
            let kldref = gpu_forward_calib(
                gpu,
                weights,
                config,
                tokens,
                opts.kldref.then_some(opts.kldref_topk),
            )?;
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
