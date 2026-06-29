// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3 text-decoder calibration collection. See LICENSE / NOTICE.

use hip_bridge::{HipError, HipResult};
use hipfire_runtime::calibration::{collect_grouped, logsumexp, topk_logits, CalibForward};
use hipfire_runtime::hfq::HfqMemTensor;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weights::WeightTensor;
use rdna_compute::Gpu;

pub use hipfire_runtime::calibration::CalibSummary;

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
    let mut static_meta: Vec<(&str, serde_json::Value)> = provenance.to_vec();
    static_meta.push(("text_only", serde_json::json!(true)));
    static_meta.push(("text_prefix", serde_json::json!(prefix)));

    // Gemma3 is dense (no MoE) so every captured tensor wants a full Hessian, but
    // all layers at once do not fit — the grouped driver re-runs the forward per
    // layer-group and concatenates the parts. KLDREF is captured once (group 0).
    collect_grouped(
        gpu,
        0,
        config.num_hidden_layers,
        layers_per_pass(),
        Vec::new(),
        output,
        &static_meta,
        |start, end| build_capture_names_for_layers(weights, prefix, start, end),
        |gpu, group_idx| {
            let mut kldref: Vec<(f32, Vec<(u32, f32)>)> = Vec::new();
            run_text_forward_for_capture(
                gpu,
                weights,
                config,
                tokens,
                opts,
                group_idx == 0,
                &mut kldref,
            )
            .map_err(|e| format!("gemma3 calib forward: {e}"))?;
            let mut extra_meta: Vec<(String, serde_json::Value)> = Vec::new();
            let extra_tensors = if group_idx == 0 {
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
                kldref_extra(&kldref)
            } else {
                Vec::new()
            };
            Ok(CalibForward {
                extra_tensors,
                extra_meta,
            })
        },
    )
    .map_err(|e| HipError::new(0, &e))
}

/// Daemon calibration seam: collect from the already-resident [`Gemma3Backend`]
/// (arch_id 12, dense text). Delegates to the text-only collector with an empty
/// prefix; the resident backend already holds bf16 weights + config. Gemma3-VL
/// (arch_id 13, `Gemma3VlBackend`, `language_model.` prefix) is not yet wired.
impl hipfire_runtime::calibration::CalibratableBackend for crate::arch::Gemma3Backend {
    fn collect_calibration(
        &self,
        gpu: &mut Gpu,
        tokenizer: &Tokenizer,
        tokens: &[u32],
        kldref: bool,
        output: &std::path::Path,
        provenance: &[(&str, serde_json::Value)],
    ) -> Result<CalibSummary, String> {
        let opts = CalibOpts {
            kldref,
            kldref_topk: 64,
        };
        collect_calibration_artifacts_text_only(
            gpu,
            &self.weights,
            &self.config,
            tokenizer,
            tokens,
            &opts,
            output,
            "",
            provenance,
        )
        .map_err(|e| e.to_string())
    }
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
