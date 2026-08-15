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
use crate::dequant::*;

pub(crate) static IMATRIX: OnceLock<HashMap<String, Vec<f32>>> = OnceLock::new();
pub(crate) static AWQ_ALPHA: OnceLock<f32> = OnceLock::new();


pub(crate) fn resolve_model_path(input: &str) -> String {
    let path = Path::new(input);

    // If it's already a valid local directory with config.json, use it directly
    if path.join("config.json").exists() {
        return input.to_string();
    }

    // Check if it looks like a HuggingFace model ID (contains exactly one /)
    if input.contains('/') && !input.contains(std::path::MAIN_SEPARATOR)
        || (cfg!(unix) && input.matches('/').count() == 1)
    {
        let parts: Vec<&str> = input.splitn(2, '/').collect();
        if parts.len() == 2 {
            let org = parts[0];
            let name = parts[1];

            // Check HF cache: ~/.cache/huggingface/hub/models--{org}--{name}/snapshots/*/
            let home = std::env::var("HOME").unwrap_or_default();
            let cache_dir = format!("{home}/.cache/huggingface/hub/models--{org}--{name}");
            let snapshots_dir = Path::new(&cache_dir).join("snapshots");

            if snapshots_dir.exists() {
                // Find the first snapshot directory
                if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                    for entry in entries.flatten() {
                        let snap_path = entry.path();
                        if snap_path.is_dir() && snap_path.join("config.json").exists() {
                            eprintln!("Resolved {input} -> {}", snap_path.display());
                            return snap_path.to_string_lossy().to_string();
                        }
                    }
                }
            }

            // Not in cache — try to download
            eprintln!("Model {input} not found locally. Downloading via huggingface-cli...");
            let status = std::process::Command::new("huggingface-cli")
                .args(["download", input])
                .status();

            match status {
                Ok(s) if s.success() => {
                    // Retry cache lookup after download
                    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                        for entry in entries.flatten() {
                            let snap_path = entry.path();
                            if snap_path.is_dir() && snap_path.join("config.json").exists() {
                                eprintln!("Downloaded {input} -> {}", snap_path.display());
                                return snap_path.to_string_lossy().to_string();
                            }
                        }
                    }
                }
                Ok(s) => eprintln!("huggingface-cli download failed with status {s}"),
                Err(e) => eprintln!(
                    "Failed to run huggingface-cli: {e}. Install with: pip install huggingface_hub"
                ),
            }
        }
    }

    // Fall through: return as-is, will fail at config.json read with a helpful error
    input.to_string()
}

// ─── GGUF input pipeline ────────────────────────────────────────────────────

/// True if the path points to a `.gguf` file on disk.
pub(crate) fn is_gguf_input(p: &Path) -> bool {
    p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("gguf")
}

/// Translate llama.cpp GGUF tensor names to the HuggingFace safetensors
/// names that `hipfire_runtime::hfq::load_weights_hfq` expects. The mapping is
/// the canonical llama.cpp ↔ HF convention.
///
/// Returns None for tensors that don't have a known safetensors equivalent
/// (we then keep them under their GGUF name; the future loader can decide
/// what to do, or they're skipped).
pub(crate) fn gguf_to_safetensors_name(gguf_name: &str) -> Option<String> {
    // Top-level tensors.
    match gguf_name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".to_string()),
        "output.weight" => return Some("lm_head.weight".to_string()),
        "output_norm.weight" => return Some("model.norm.weight".to_string()),
        _ => {}
    }
    // Per-layer: blk.{N}.<slot>.weight  →  model.layers.{N}.<slot>.weight
    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        // rest = "{N}.<slot>.weight"
        let dot = rest.find('.')?;
        let layer_idx = &rest[..dot];
        let slot_full = &rest[dot + 1..]; // "<slot>.weight"
                                          // Drop the trailing ".weight" so we can rewrite slots like "attn_q"→"self_attn.q_proj".
        let slot = slot_full.strip_suffix(".weight")?;
        let translated = match slot {
            "attn_norm" => "input_layernorm".to_string(),
            "ffn_norm" => "post_attention_layernorm".to_string(),
            "attn_q" => "self_attn.q_proj".to_string(),
            "attn_k" => "self_attn.k_proj".to_string(),
            "attn_v" => "self_attn.v_proj".to_string(),
            "attn_output" => "self_attn.o_proj".to_string(),
            "attn_q_norm" => "self_attn.q_norm".to_string(),
            "attn_k_norm" => "self_attn.k_norm".to_string(),
            "ffn_gate" => "mlp.gate_proj".to_string(),
            "ffn_up" => "mlp.up_proj".to_string(),
            "ffn_down" => "mlp.down_proj".to_string(),
            other => return Some(format!("model.layers.{layer_idx}.{other}.weight")),
        };
        return Some(format!("model.layers.{layer_idx}.{translated}.weight"));
    }
    None
}

/// True if the GGUF tensor's name is a 1D norm / RMSNorm scaling vector.
/// These stay F16 in the .hfq (no benefit from quantization, precision-sensitive).
pub(crate) fn gguf_is_norm_tensor(name: &str) -> bool {
    name.contains("_norm") || name.contains("norm.weight")
}

/// Translate a hipfire safetensors-style tensor name to the ggml-style name
/// used by llama.cpp's imatrix output (and the rest of llama.cpp's tooling).
///
/// Verified by shape-alignment on Qwen3.5-0.8B imatrix vs safetensors load log
/// (2026-05-11):
///   - K dims match for every covered tensor class (mlp.* , self_attn.* ,
///     linear_attn.in_proj_qkv/z/a/b, linear_attn.out_proj).
///   - Layer-pattern: FullAttention layers (3, 7, 11, ...) carry standard
///     `attn_q/k/v/output`; LinearAttention layers carry `attn_qkv`/
///     `attn_gate`/`ssm_alpha`/`ssm_beta`/`ssm_out` — the SSM-naming
///     convention llama.cpp uses for Mamba-style sub-blocks.
///
/// Returns `None` for tensors that don't have an imatrix counterpart
/// (norms / biases / 1D scalars / lookup-only tables). Those fall back to
/// non-imatrix-weighted quantization in the call site.
pub(crate) fn safetensors_to_ggml_name(name: &str) -> Option<String> {
    // Drop the architecture-specific "language_model." prefix (Qwen3.5
    // structure has model.language_model.layers.{N}.* — the linear-attn
    // crate uses this nested layout, llama.cpp flattens to blk.{N}.*).
    let normalized = name
        .strip_prefix("model.language_model.")
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name);

    // Top-level (currently no imatrix coverage; default is --process-output OFF).
    match normalized {
        "embed_tokens.weight" => return Some("token_embd.weight".to_string()),
        "lm_head.weight" => return Some("output.weight".to_string()),
        "norm.weight" => return Some("output_norm.weight".to_string()),
        _ => {}
    }

    // Per-layer: "layers.{N}.<slot>.weight"
    let rest = normalized.strip_prefix("layers.")?;
    let dot = rest.find('.')?;
    let layer_idx = &rest[..dot];
    let slot_full = &rest[dot + 1..];
    let slot = slot_full.strip_suffix(".weight")?;

    let translated = match slot {
        // MLP — present on every layer.
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        // FullAttention layer tensors (standard names).
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        // Glimmer gates attention output before o_proj under a name Qwen does
        // not use (see hipfire-arch-muse-glimmer lib.rs). llama.cpp exports it
        // as blk.{N}.attn_gate, so without this arm the 52 Glimmer gate tensors
        // silently miss AWQ despite `awq_eligible` matching `gate_proj.weight`
        // and the imatrix carrying the entry. No collision with the linear-attn
        // arm below: a layer is either full- or linear-attention, never both.
        "self_attn.gate_proj" => "attn_gate",
        // LinearAttention layer tensors (Mamba-2 / hybrid-arch SSM naming).
        "linear_attn.in_proj_qkv" => "attn_qkv",
        "linear_attn.in_proj_z" => "attn_gate",
        "linear_attn.in_proj_a" => "ssm_alpha",
        "linear_attn.in_proj_b" => "ssm_beta",
        "linear_attn.out_proj" => "ssm_out",
        // Unmapped: conv1d.weight (special-cased to HFQ4G128 at quantize
        // time; small, not multiplied by activation in the standard sense),
        // norm.weight, A_log, dt_bias (1D or scalars, no imatrix entry).
        _ => return None,
    };

    Some(format!("blk.{layer_idx}.{translated}.weight"))
}

/// Load an llama.cpp-compatible imatrix GGUF file and build a lookup
/// keyed by ggml-style tensor name. The GGUF stores per-linear-layer
/// pairs:
///   {name}.in_sum2     F32[k, n_mat]   sum of squared activations per channel
///   {name}.counts      F32[1, n_mat]   token count contributing per matrix
///
/// For non-MoE models n_mat=1; the [k] vector goes into the map directly.
/// For MoE we'd need per-expert handling — out of scope for Step 5a
/// (Qwen3.5 dense + Qwen3.6 dense are the first cohort targets; A3B MoE
/// is deferred to a future iteration that handles n_mat > 1).
///
/// Returns `HashMap<ggml_name, Vec<f32>>` with the .in_sum2 values keyed by
/// the BASE tensor name (the ".in_sum2" suffix stripped).
pub(crate) fn load_imatrix(path: &Path) -> HashMap<String, Vec<f32>> {
    use gguf_input::GgmlType;
    let gguf = gguf_input::GgufFile::open(path).unwrap_or_else(|e| {
        eprintln!("error: failed to open imatrix file {}: {e}", path.display());
        std::process::exit(1);
    });

    let mut map: HashMap<String, Vec<f32>> = HashMap::new();
    let mut total_entries = 0usize;
    let mut skipped_moe = 0usize;
    for t in &gguf.tensors {
        let name = match t.name.strip_suffix(".in_sum2") {
            Some(n) => n.to_string(),
            None => continue, // ignore .counts and any other entries
        };
        if t.dtype != GgmlType::F32 {
            eprintln!(
                "warning: imatrix entry {} has non-F32 dtype {:?}; skipping",
                t.name, t.dtype
            );
            continue;
        }
        // Shape is [k] (1D) for non-MoE; [k, n_mat] for MoE. Skip multi-mat
        // tensors with a warning — Step 5a doesn't handle them yet.
        let n_mat = if t.shape.len() >= 2 { t.shape[1] } else { 1 };
        if n_mat != 1 {
            skipped_moe += 1;
            continue;
        }
        let k = t.shape[0];

        // Read the F32 values from the tensor data segment.
        let data = gguf.tensor_data(t);
        let mut values = Vec::with_capacity(k);
        for i in 0..k {
            let off = i * 4;
            let v = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            values.push(v);
        }
        map.insert(name, values);
        total_entries += 1;
    }

    eprintln!(
        "imatrix: loaded {} entries from {} ({} MoE multi-matrix entries skipped — Step 5a is dense-only)",
        total_entries,
        path.display(),
        skipped_moe,
    );
    if total_entries == 0 {
        if skipped_moe > 0 {
            // MoE-only imatrix (e.g. MiniMax routed experts): no 1D dense
            // entries for the legacy dense-AWQ table, but the file IS valid.
            // The MiniMax AWQ-on-experts path reads the raw imatrix GGUF
            // (imatrix_gguf) directly, so an empty dense table is harmless —
            // dense tensors just fall back to non-imatrix quantization.
            eprintln!(
                "imatrix: 0 dense entries, {skipped_moe} MoE multi-matrix entries — \
                 dense table empty (MoE-only imatrix; expert AWQ uses the raw GGUF)"
            );
        } else {
            eprintln!("error: imatrix file contains no usable .in_sum2 entries");
            std::process::exit(1);
        }
    }
    map
}

/// Look up imatrix per-channel weights for a given safetensors tensor name.
/// Returns `None` (caller falls back to non-imatrix-weighted quantization) if:
///   - --imatrix wasn't passed (IMATRIX not initialized), OR
///   - the tensor name doesn't have a ggml-mapping (norms, small 1D, etc.), OR
///   - the imatrix file doesn't carry this tensor (rare; usually means the
///     tensor wasn't exercised by the calibration corpus).
pub(crate) fn imatrix_weights_for(safetensors_name: &str) -> Option<&'static [f32]> {
    let im = IMATRIX.get()?;
    // `load_imatrix` keys the map by the imatrix FILE's tensor names (`.in_sum2`
    // stripped). hipfire's `collect_imatrix` emits *safetensors* names
    // (`model.language_model.layers.N.linear_attn.in_proj_qkv.weight`), so try the
    // direct safetensors name FIRST — this was the AWQ no-op: the map is
    // safetensors-keyed but we only tried the GGML-converted name, which always
    // missed (and 27B-3.6 hybrid linear_attn names don't round-trip anyway).
    // Fall back to the GGML name for llama.cpp-style (blk.*) imatrices.
    if let Some(v) = im.get(safetensors_name) {
        return Some(v.as_slice());
    }
    let ggml_name = safetensors_to_ggml_name(safetensors_name)?;
    im.get(&ggml_name).map(|v| v.as_slice())
}

/// Compute AWQ per-channel scales `s[j]` for one linear-layer weight tensor.
///
/// Inputs:
///   - `in_sum2`: imatrix data — Σ_token act²[j] per input channel, length K.
///     Source: hipfire's `imatrix_collect` (llama.cpp `--imatrix` output).
///   - `alpha`: AWQ tuning parameter ∈ [0, 1]. Paper-original default = 0.5.
///
/// Output:
///   - `Vec<f32>` of length K, with geometric mean normalized to ≈ 1.0.
///
/// Formula (AWQ-paper-original simplified for hipfire's data shape):
///   1. RMS_act[j] = sqrt(in_sum2[j] / N_tok). The N_tok term is a global
///      constant for the tensor and gets absorbed by the geo-mean normalization
///      below, so we can omit it from the per-channel computation.
///      Equivalent: use sqrt(in_sum2[j]) directly.
///   2. s_raw[j] = (RMS_act[j])^alpha
///   3. Normalize: s[j] = s_raw[j] / exp(mean_j log(s_raw[j]))
///      This keeps the post-AWQ-scaled weight tensor's overall magnitude
///      in the same range as the input — important for the downstream MQ4
///      min-max scale fitter not to suddenly compress/expand its dynamic
///      range based on alpha.
///
/// Edge cases:
///   - Zero in_sum2[j] (channel never exercised by calibration): clamp to
///     a tiny floor (1e-12) before sqrt to avoid log(0). Practically rare;
///     would mean a channel is unused in the calibration corpus.
///   - alpha == 0 → all s[j] = 1.0 (AWQ disabled at this layer). Caller
///     can short-circuit before invoking this function.
///
/// Cost: O(K). For 9B Qwen3.5 ~32 calls × ~4096 elements = ~131K ops total
/// across the whole quantize. Negligible.
/// Parse the layer index N from a MiniMax expert tensor name
/// `…layers.N.block_sparse_moe.experts.E.wX.weight`.
pub(crate) fn minimax_layer_index(name: &str) -> Option<usize> {
    let after = name.split(".layers.").nth(1)?;
    after.split('.').next()?.parse::<usize>().ok()
}

/// True if layer `l` falls in the comma-separated range list held in process config `var`
/// (e.g. "12-45,50,55-60"; inclusive ranges or bare singles). Unset/empty →
/// false. Drives per-layer mixed-precision expert promotion for MiniMax.
pub(crate) fn minimax_layer_in_config_set(var: &str, l: usize) -> bool {
    let spec = match hipfire_config::developer_var(var) {
        Ok(v) => v,
        Err(_) => return false,
    };
    for tok in spec.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        if let Some((a, b)) = tok.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                if l >= a.min(b) && l <= a.max(b) {
                    return true;
                }
            }
        } else if let Ok(n) = tok.parse::<usize>() {
            if l == n {
                return true;
            }
        }
    }
    false
}

/// Shared-per-layer AWQ scales for MiniMax routed experts from an imatrix GGUF.
/// Aggregates per-expert activation energy (in_sum2) across ALL experts of
/// layer `n` into one shared per-input-channel scale: gate(w1)/up(w3) share the
/// MoE-input channels (s_gate_up, len hidden); down(w2) uses the intermediate
/// channels (s_down, len inter). The forward applies these via experts[0], so
/// one scale per layer is exactly what the runtime consumes. None if absent.
pub(crate) fn minimax_layer_awq_scales(
    gguf: &gguf_input::GgufFile,
    n: usize,
    alpha: f32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let agg = |kind: &str| -> Option<Vec<f32>> {
        let nm = format!("blk.{n}.ffn_{kind}_exps.weight.in_sum2");
        let t = gguf.tensors.iter().find(|t| t.name == nm)?;
        if t.shape.len() != 2 {
            return None;
        }
        let k = t.shape[0];
        let n_exp = t.shape[1];
        let flat: Vec<f32> = gguf
            .tensor_data(t)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if flat.len() != k * n_exp {
            return None;
        }
        let mut a = vec![0.0f32; k];
        for e in 0..n_exp {
            let off = e * k;
            for j in 0..k {
                a[j] += flat[off + j];
            }
        }
        Some(a)
    };
    let g = agg("gate")?;
    let gu: Vec<f32> = match agg("up") {
        Some(u) if u.len() == g.len() => g.iter().zip(&u).map(|(a, b)| a + b).collect(),
        _ => g.clone(),
    };
    let d = agg("down")?;
    Some((
        compute_awq_scales(&gu, alpha),
        compute_awq_scales(&d, alpha),
    ))
}

pub(crate) fn compute_awq_scales(in_sum2: &[f32], alpha: f32) -> Vec<f32> {
    let k = in_sum2.len();
    debug_assert!(k > 0, "empty imatrix vector");

    // Step 1+2: RMS_act^alpha, with the constant N_tok factor absorbed into
    // the geo-mean normalization. The sqrt and (·)^alpha combine into
    // (·)^(alpha/2) on the raw in_sum2 values.
    //
    // Implementation choice: compute log(s_raw) directly so we can do the
    // geo-mean normalization in log space (numerically more stable for
    // wide dynamic-range imatrix values).
    let half_alpha = (alpha as f64) * 0.5;
    let mut log_s_raw = Vec::with_capacity(k);
    let mut sum_log: f64 = 0.0;
    for &v in in_sum2 {
        // Floor dead channels to 1e-12 (NaN also maps here: f64::max returns the
        // non-NaN arg) AND cap non-finite / pathologically-large values to a
        // finite ceiling. An inf in_sum2 — f32 overflow during imatrix
        // collection, which the 27B tier1 imatrix actually contains — would
        // otherwise make this tensor's `mean_log = inf`, and then `l - mean_log`
        // = inf - inf = NaN for the inf channel. That NaN survives the output
        // clamp below (f32::clamp propagates NaN), poisoning the F16 sidecar and
        // NaN'ing the whole forward (37747 such values measured pre-fix).
        // Capping the input keeps mean_log finite; the output clamp then bounds
        // the final scale. 1e30 is well inside f64 range (ln ≈ 69).
        let v_clamped = (v as f64).max(1e-12).min(1e30);
        let log_s = half_alpha * v_clamped.ln(); // log(v^(alpha/2)) = (alpha/2) * log(v)
        log_s_raw.push(log_s);
        sum_log += log_s;
    }
    let mean_log = sum_log / (k as f64);

    // Step 3: subtract mean in log space, then exp back. After this,
    // geo_mean(s) = exp(0) = 1.0 exactly (within floating-point precision).
    //
    // Step 4 (CRITICAL — f16 safety): clamp to an f16-representable,
    // non-exploding range. The geo-mean is 1.0 by construction, so the bulk
    // of channels sit near 1; only pathological outliers reach the rails —
    // dead channels floored to 1e-12, or hot channels with huge activation
    // sums. Without this, exp() overflows to f32 inf and/or the F16 sidecar
    // under/overflows, and the inference-time `x / awq_scale` divide produces
    // inf → NaN. (Verified via dump_awq_scales on the 27B tier1 imatrix:
    // 49293 scales underflowed to 0.0 and 37747 stored as inf/NaN pre-clamp,
    // which NaN'd the whole forward — KLD 0.0 / PPL NaN on gfx11.)
    //
    // The SAME clamped vector is used for both the weight pre-scale (W*s) and
    // the emitted sidecar (x/s at inference), so the cancellation stays exact;
    // clamping only limits how aggressively pathological channels redistribute
    // quant difficulty. Real AWQ scales live in ~[0.2, 5]; [1e-2, 1e2] keeps
    // all genuine signal while removing the representability blow-ups.
    pub(crate) const AWQ_SCALE_MIN: f32 = 1e-2;
    pub(crate) const AWQ_SCALE_MAX: f32 = 1e2;
    log_s_raw
        .into_iter()
        .map(|l| ((l - mean_log).exp() as f32).clamp(AWQ_SCALE_MIN, AWQ_SCALE_MAX))
        .collect()
}

/// Apply AWQ pre-scaling to a row-major [m, k] weight tensor in place:
/// `W'[i,j] = W[i,j] * s[j]` for every (i, j).
///
/// AWQ scales are per-INPUT-channel (length K). The same s[j] vector
/// broadcasts across every output row i.
///
/// Done in-place to avoid allocating a second [m, k] buffer. The caller
/// owns the W slice and is responsible for ensuring this pre-scaling
/// happens BEFORE any subsequent transformation (e.g. FWHT rotation).
pub(crate) fn awq_pre_scale_weights(weights: &mut [f32], m: usize, k: usize, scales: &[f32]) {
    debug_assert_eq!(weights.len(), m * k, "weight buffer size mismatch");
    debug_assert_eq!(scales.len(), k, "AWQ scale vector must have length K");
    for r in 0..m {
        let row = &mut weights[r * k..(r + 1) * k];
        for j in 0..k {
            row[j] *= scales[j];
        }
    }
}

/// Helper: convert a `Vec<f32>` AWQ-scale vector into the F16 byte
/// payload that `HfqTensor` consumes for sidecar emission.
pub(crate) fn awq_scales_to_f16_bytes(scales: &[f32]) -> Vec<u8> {
    scales
        .iter()
        .flat_map(|&s| f32_to_f16(s).to_le_bytes())
        .collect()
}

/// AWQ pre-scaling is mathematically valid only for weights whose runtime
/// path applies the inverse divide-by-scale. As of F2 (2026-05-14), this
/// covers both the input-side projections (fed via the AWQ-aware variants
/// of `fused_rmsnorm_rotate_mq` from F1) AND the output-side projections
/// (`o_proj` / `out_proj` / `down_proj` / `w_down`, fed via the AWQ-aware
/// variants `rotate_x_mq_awq` and `fused_silu_mul_mq_rotate_awq` from F2).
///
/// Runtime path mapping for AWQ inverse divide-by-scale:
/// - `fused_rmsnorm_mq_rotate_awq`: post-RMSNorm input projections
///   (q/k/v/qkv, gate/up, in_proj_*, router, gate_up_proj)
/// - `rotate_x_mq_awq`: post-attention input to o_proj / out_proj
/// - `fused_silu_mul_mq_rotate_awq`: post-SwiGLU input to down_proj
///
/// Pre-F2 history: until 2026-05-14, output-side projections (o_proj /
/// out_proj / down_proj / w_down) were NOT on this whitelist because
/// their runtime path lacked AWQ-aware kernels. Pre-scaling them without
/// a runtime compensating divide produces `(W·s) · x ≠ W · x` — measured
/// 0.8B Qwen3.5 KLD blowup 0.6721 → 13.4893; see `awq_fix_claude.md`.
/// F2 added those kernels (`rotate_x_mq_awq` / `fused_silu_mul_mq_rotate_awq`)
/// plus `_for` helper routing in hipfire-runtime/llama.rs, so the whitelist
/// is now safe to expand.
///
/// Whitelist (vs blacklist) is still the safe default: a new tensor name
/// in a future arch fails closed (no AWQ) until someone confirms its
/// runtime path is AWQ-aware.
pub(crate) fn awq_eligible(name: &str) -> bool {
    // F1-vs-F2 A/B gate. When `HIPFIRE_AWQ_F1_ONLY=1` is set, the F2
    // additions below (o_proj / wo / out_proj / down_proj / w_down)
    // are excluded — produces an F1-equivalent quant for comparison
    // bench against the same binary's F2 quant. Default (env unset):
    // the full F2 whitelist applies.
    let f1_only = hipfire_config::developer_var("HIPFIRE_AWQ_F1_ONLY")
        .ok()
        .as_deref()
        == Some("1");
    let f1_match =
    // Full-attention input projections (HF naming + fused variants).
    name.ends_with("q_proj.weight")
        || name.ends_with("k_proj.weight")
        || name.ends_with("v_proj.weight")
        || name.ends_with("qkv_proj.weight")
        || name.ends_with("wqkv.weight")
        // MLP input projections (HF + hipfire-internal naming).
        || name.ends_with("gate_proj.weight")
        || name.ends_with("up_proj.weight")
        || name.ends_with("w_gate.weight")
        || name.ends_with("w_up.weight")
        // MoE fused expert gate+up projection (Qwen3-MoE convention —
        // experts.gate_up_proj is [num_experts, 2*intermediate, hidden]
        // with rows split between gate and up halves). Same input-side
        // semantics as gate_proj/up_proj: post-RMSNorm hidden state
        // routed via the MoE dispatch.
        || name.ends_with("gate_up_proj.weight")
        // Linear-attention input projections (Qwen3.5 Gated-DeltaNet).
        // Suffix varies (in_proj_qkv / _z / _a / _b); the substring is
        // anchored enough that no non-linear-attn tensor name should match.
        || name.contains(".in_proj_")
        // MoE router (HF naming for Qwen3-MoE / DeepSeek family — single
        // linear projecting post-RMSNorm hidden state to num_experts
        // logits). The quantizer's q8_router rule (set when is_moe)
        // promotes this to Q8 before reaching the MQ4G256 branch, so
        // this match is effectively dead code today. Kept for intent:
        // if Q8 auto-promotion is ever disabled, this preserves
        // correctness. `router.weight` would be a non-HF naming an
        // arch might choose; kept for safety.
        || name.ends_with("mlp.gate.weight")
        // MiniMax-M2 MoE router (block_sparse_moe.gate.weight). Same intent
        // as mlp.gate.weight: q8_router (set for is_minimax via is_moe_like)
        // keeps the router at Q8 so HFQ4 noise can't flip top-k selection.
        || name.ends_with("block_sparse_moe.gate.weight")
        || name.ends_with("router.weight")
        // Gemma4 26B-A4B MoE router: `router.proj.weight` (hidden_size × num_experts).
        // Same precision-sensitivity as Qwen3.5's `mlp.gate.weight`.
        || name.ends_with("router.proj.weight");
    if f1_only {
        return f1_match;
    }
    let f2_match =
        // ── F2 (2026-05-14): output-side projections ────────────────────
        // These now have AWQ-aware runtime kernels (rotate_x_mq_awq for
        // o_proj/out_proj/wo; fused_silu_mul_mq_rotate_awq for down_proj/w_down).
        // Runtime dispatch routes through _for helpers in llama.rs based on
        // WeightTensor.awq_scale.
        //
        // FullAttention output projection (HF + hipfire-internal naming).
        name.ends_with("o_proj.weight")
        || name.ends_with("wo.weight")
        // LinearAttention output projection (Qwen3.5 Gated-DeltaNet).
        || name.ends_with("out_proj.weight")
        // MLP down projection (HF + hipfire-internal naming).
        || name.ends_with("down_proj.weight")
        || name.ends_with("w_down.weight");
    f1_match || f2_match
}

/// True if the tensor is the token embedding. We Q8 these (matches the
/// safetensors path's `is_embed` rule — Q4 is too lossy for embedding tables).
pub(crate) fn gguf_is_embed_tensor(name: &str) -> bool {
    name == "token_embd.weight"
}

/// Build the `config` JSON object that `hipfire_runtime::hfq::config_from_hfq`
/// reads. Mirrors the field names HuggingFace uses in `config.json` for
/// LlamaForCausalLM / Qwen3ForCausalLM, populated from the GGUF
/// `<arch>.*` metadata keys.
pub(crate) fn config_json_from_gguf(gguf: &gguf_input::GgufFile, arch_str: &str) -> serde_json::Value {
    // GGUF prefixes its model hyperparameters with the architecture name —
    // e.g. for `general.architecture=llama` the keys live under `llama.*`.
    let prefix = arch_str;

    let read_u = |k: &str| -> Option<u64> {
        gguf.metadata.get(k).and_then(|v| match v {
            gguf_input::MetaValue::U8(x) => Some(*x as u64),
            gguf_input::MetaValue::I8(x) => Some(*x as u64),
            gguf_input::MetaValue::U16(x) => Some(*x as u64),
            gguf_input::MetaValue::I16(x) => Some(*x as u64),
            gguf_input::MetaValue::U32(x) => Some(*x as u64),
            gguf_input::MetaValue::I32(x) => Some(*x as u64),
            gguf_input::MetaValue::U64(x) => Some(*x),
            gguf_input::MetaValue::I64(x) => Some(*x as u64),
            _ => None,
        })
    };
    let read_f = |k: &str| -> Option<f64> {
        gguf.metadata.get(k).and_then(|v| match v {
            gguf_input::MetaValue::F32(x) => Some(*x as f64),
            gguf_input::MetaValue::F64(x) => Some(*x),
            _ => None,
        })
    };

    let dim = read_u(&format!("{prefix}.embedding_length"));
    let n_layers = read_u(&format!("{prefix}.block_count"));
    let n_heads = read_u(&format!("{prefix}.attention.head_count"));
    let n_kv_heads = read_u(&format!("{prefix}.attention.head_count_kv")).or(n_heads);
    let hidden_dim = read_u(&format!("{prefix}.feed_forward_length"));
    // vocab_size: prefer metadata, fall back to token_embd shape[1].
    let vocab_size = read_u(&format!("{prefix}.vocab_size")).or_else(|| {
        gguf.tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .and_then(|t| t.shape.get(1).map(|&s| s as u64))
    });
    let max_seq_len = read_u(&format!("{prefix}.context_length"));
    let rope_theta = read_f(&format!("{prefix}.rope.freq_base"));
    let rms_eps = read_f(&format!("{prefix}.attention.layer_norm_rms_epsilon"));
    let head_dim = read_u(&format!("{prefix}.attention.key_length")).or_else(|| {
        // Fall back: head_dim = dim / n_heads.
        dim.zip(n_heads).map(|(d, h)| if h > 0 { d / h } else { d })
    });
    let bos = read_u("tokenizer.ggml.bos_token_id").unwrap_or(1);
    let eos = read_u("tokenizer.ggml.eos_token_id").unwrap_or(2);

    let mut cfg = serde_json::Map::new();
    cfg.insert(
        "model_type".to_string(),
        serde_json::Value::from(arch_str.to_string()),
    );
    if let Some(v) = dim {
        cfg.insert("hidden_size".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = n_layers {
        cfg.insert("num_hidden_layers".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = n_heads {
        cfg.insert(
            "num_attention_heads".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = n_kv_heads {
        cfg.insert(
            "num_key_value_heads".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = hidden_dim {
        cfg.insert("intermediate_size".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = vocab_size {
        cfg.insert("vocab_size".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = max_seq_len {
        cfg.insert(
            "max_position_embeddings".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = rope_theta {
        cfg.insert("rope_theta".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = rms_eps {
        cfg.insert("rms_norm_eps".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = head_dim {
        cfg.insert("head_dim".to_string(), serde_json::Value::from(v));
    }
    cfg.insert("bos_token_id".to_string(), serde_json::Value::from(bos));
    cfg.insert("eos_token_id".to_string(), serde_json::Value::from(eos));
    serde_json::Value::Object(cfg)
}

/// Translate the GGUF metadata HashMap into a JSON object that ends up in
/// the `.hfq` header's metadata blob. A future engine-side `from_hfq` for
/// Llama-style models can read these fields the same way the existing
/// `from_gguf` reads them today.
pub(crate) fn gguf_meta_to_json(meta: &HashMap<String, gguf_input::MetaValue>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (k, v) in meta {
        let json_v = mv_to_json(v);
        map.insert(k.clone(), json_v);
    }
    serde_json::Value::Object(map)
}

pub(crate) fn mv_to_json(v: &gguf_input::MetaValue) -> serde_json::Value {
    use gguf_input::MetaValue as MV;
    match v {
        MV::U8(x) => serde_json::Value::from(*x),
        MV::I8(x) => serde_json::Value::from(*x),
        MV::U16(x) => serde_json::Value::from(*x),
        MV::I16(x) => serde_json::Value::from(*x),
        MV::U32(x) => serde_json::Value::from(*x),
        MV::I32(x) => serde_json::Value::from(*x),
        MV::F32(x) => serde_json::Value::from(*x),
        MV::Bool(x) => serde_json::Value::from(*x),
        MV::String(s) => serde_json::Value::from(s.clone()),
        MV::U64(x) => serde_json::Value::from(*x),
        MV::I64(x) => serde_json::Value::from(*x),
        MV::F64(x) => serde_json::Value::from(*x),
        // Tokenizer arrays (tokens, scores, merges, ...) can be huge —
        // serialize them as JSON arrays so the engine side can re-parse.
        MV::Array(arr) => serde_json::Value::Array(arr.iter().map(mv_to_json).collect()),
    }
}