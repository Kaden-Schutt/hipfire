// SPDX-License-Identifier: Apache-2.0
// hipfire — MiniMax-M2 text calibration collection.

use crate::forward::decode_step;
use crate::minimax::{MiniMaxConfig, MiniMaxState, MiniMaxWeights};
use hip_bridge::{HipError, HipResult};
use hipfire_runtime::calibration::{collect, logsumexp, topk_logits, CalibForward};
use hipfire_runtime::hfq::HfqMemTensor;
use hipfire_runtime::weights::WeightTensor;
use rdna_compute::Gpu;
use std::collections::HashMap;
use std::path::Path;

pub use hipfire_runtime::calibration::CalibSummary;

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

fn put(m: &mut HashMap<usize, String>, wt: &WeightTensor, name: impl Into<String>) {
    m.insert(wt.buf.buf.as_ptr() as usize, name.into());
}

/// Build the capture map for MiniMax dense/attention/router projections.
///
/// Names match checkpoint tensor keys without the `.weight` suffix so the
/// quantizer can join `<name>.imatrix`/`<name>.hessian` to the source weights.
/// These projections all route through `weight_gemv`/`weight_gemv_residual`
/// (see `forward.rs`), where the `maybe_capture_activation` tap fires — so they
/// capture for free, keyed by each weight's GPU buffer pointer.
///
/// Routed experts are NOT captured here: they are byte-fused into the per-layer
/// `gate_up`/`down` blobs and dispatched through the indexed
/// `gemv_hfq4g256_moe_*` kernels, which do not pass one weight pointer per
/// source tensor to `weight_gemv`. Per-expert imatrix capture needs explicit
/// taps around those indexed kernels (the LFM2 `decode_step_capture` pattern)
/// and is a documented follow-on; the dense/attention/router/lm_head Hessians
/// captured here are exactly what oq4++ (LDLQ) needs for the non-expert path.
pub fn build_capture_names(weights: &MiniMaxWeights) -> HashMap<usize, String> {
    let mut m = HashMap::new();
    for (i, layer) in weights.layers.iter().enumerate() {
        let p = format!("model.layers.{i}");
        put(&mut m, &layer.wq, format!("{p}.self_attn.q_proj"));
        put(&mut m, &layer.wk, format!("{p}.self_attn.k_proj"));
        put(&mut m, &layer.wv, format!("{p}.self_attn.v_proj"));
        put(&mut m, &layer.wo, format!("{p}.self_attn.o_proj"));
        put(&mut m, &layer.router, format!("{p}.block_sparse_moe.gate"));
    }
    put(&mut m, &weights.lm_head, "lm_head");
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

/// Collect calibration Hessians/imatrices from the MiniMax-M2 text decoder.
///
/// Covers the dense attention projections (`q/k/v/o_proj`), the MoE router
/// (`block_sparse_moe.gate`), and `lm_head` — every call that routes through
/// `weight_gemv`. All get full [K,K] Hessians (the LDLQ signal oq4++ needs).
/// Routed experts are imatrix-only and currently NOT captured (see
/// [`build_capture_names`]); the `imatrix_only` list is therefore empty.
pub fn collect_calibration_artifacts(
    gpu: &mut Gpu,
    weights: &MiniMaxWeights,
    config: &MiniMaxConfig,
    tokens: &[u32],
    opts: &CalibOpts,
    output: &Path,
    provenance: &[(&str, serde_json::Value)],
) -> HipResult<CalibSummary> {
    let mut static_meta: Vec<(&str, serde_json::Value)> = provenance.to_vec();
    static_meta.push(("arch", serde_json::json!("minimax")));
    static_meta.push(("text_only", serde_json::json!(true)));
    static_meta.push((
        "captures",
        serde_json::json!("decode_step_weight_gemv:attn+router+lm_head"),
    ));
    static_meta.push((
        "routed_expert_capture",
        serde_json::json!("not-captured:indexed-moe-kernels-need-explicit-taps"),
    ));

    collect(
        gpu,
        crate::ARCH_ID,
        build_capture_names(weights),
        Vec::new(),
        output,
        &static_meta,
        |gpu| {
            let mut state =
                MiniMaxState::new(gpu, config).map_err(|e| format!("minimax calib state: {e}"))?;
            let mut kldref: Vec<(f32, Vec<(u32, f32)>)> = Vec::new();
            for (pos, &tok) in tokens.iter().enumerate() {
                let logits = decode_step(config, weights, &mut state, gpu, tok, pos as u32)
                    .map_err(|e| format!("minimax calib decode: {e}"))?;
                if opts.kldref {
                    kldref.push((logsumexp(&logits), topk_logits(&logits, opts.kldref_topk)));
                }
            }
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
    .map_err(|e| HipError::new(0, &e))
}

/// Thin `&weights`/`&config` adapter so the daemon's loose MiniMax resident
/// slots satisfy the calibration seam without a bundled backend type — mirrors
/// `lfm2moe::Lfm2MoeCalibBackend`.
pub struct MiniMaxCalibBackend<'a> {
    pub weights: &'a MiniMaxWeights,
    pub config: &'a MiniMaxConfig,
}

impl hipfire_runtime::calibration::CalibratableBackend for MiniMaxCalibBackend<'_> {
    fn collect_calibration(
        &self,
        gpu: &mut Gpu,
        _tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
        tokens: &[u32],
        kldref: bool,
        output: &Path,
        provenance: &[(&str, serde_json::Value)],
    ) -> Result<CalibSummary, String> {
        let opts = CalibOpts {
            kldref,
            kldref_topk: 64,
        };
        collect_calibration_artifacts(
            gpu,
            self.weights,
            self.config,
            tokens,
            &opts,
            output,
            provenance,
        )
        .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn minimax_calibration_names_match_checkpoint_keys() {
        // Sanity on the name shapes the quantizer joins `<name>.hessian` to.
        assert_eq!(
            format!("model.layers.{}.self_attn.q_proj", 7),
            "model.layers.7.self_attn.q_proj"
        );
        assert_eq!(
            format!("model.layers.{}.block_sparse_moe.gate", 7),
            "model.layers.7.block_sparse_moe.gate"
        );
    }

    #[test]
    fn arch_id_is_minimax() {
        assert_eq!(crate::ARCH_ID, 10);
    }
}
