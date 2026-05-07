//! Phase 2 CPU-only validator for free ZAYA1 components.
//!
//! Bypasses the HFQ writer entirely (which doesn't exist for ZAYA1
//! yet). Reads the model's bf16 safetensors directly + the PyTorch
//! reference activation dump, runs CPU implementations of the free
//! components, and checks per-tensor NRMSE against the reference.
//!
//! Components validated tonight:
//!   1. ZayaRMSNorm  (modeling_zaya.py:167) - on layer 0 input_norm.
//!   2. ResidualScaling (modeling_zaya.py:895) - on layer 0 + layer 1.
//!
//! Methodology: the canonical PyTorch oracle pattern from CLAUDE.md.
//! Per-component NRMSE (= ||a - b|| / ||b||) clearing the bf16 ULP
//! threshold (5e-3) means the implementation matches the reference at
//! the precision the underlying datatype permits.
//!
//! Usage:
//!   cargo run --release --example cpu_validate_phase2 -p hipfire-arch-zaya -- \
//!       --weights /tmp/zaya-port/refs/zaya1_phase2_subset.safetensors \
//!       --refs    /tmp/zaya-port/refs/refs-canonical-v3
//!
//! The weight subset file is produced by
//! `scripts/arch-intake/extract_phase2_subset.py` running on hiptrx.
//! The reference dump is produced by
//! `scripts/arch-intake/dump_zaya_reference.py`.

use half::bf16;
use safetensors::SafeTensors;
use std::fs;
use std::path::{Path, PathBuf};

const ULP_THRESHOLD_BF16: f32 = 5e-3;

// =====================================================================
// Arg parsing (no clap dependency; tiny flag-pair extractor).
// =====================================================================

struct Args {
    weights: PathBuf,
    refs: PathBuf,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut weights = None;
    let mut refs = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => {
                weights = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--refs" => {
                refs = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: cpu_validate_phase2 --weights <path.safetensors> --refs <dir>"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown flag: {other}");
                std::process::exit(2);
            }
        }
    }
    Args {
        weights: weights.expect("--weights is required"),
        refs: refs.expect("--refs is required"),
    }
}

// =====================================================================
// Tensor decoding helpers.
// =====================================================================

/// Decode a bf16 tensor from raw safetensors bytes into a Vec<f32>.
/// bf16 is little-endian on x86: 2 bytes per element, with the same
/// upper-16-bits-of-f32 representation. `half::bf16::from_le_bytes` +
/// `f32::from` gives us the right value.
fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 2 == 0, "bf16 byte length must be even");
    bytes
        .chunks_exact(2)
        .map(|p| f32::from(bf16::from_le_bytes([p[0], p[1]])))
        .collect()
}

/// Decode an fp32 tensor from raw safetensors bytes into a Vec<f32>.
fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 4 == 0, "f32 byte length must be a multiple of 4");
    bytes
        .chunks_exact(4)
        .map(|p| f32::from_le_bytes([p[0], p[1], p[2], p[3]]))
        .collect()
}

/// Load a single-tensor safetensors file (key="x", as written by the
/// dump script) into a Vec<f32>.
fn load_ref_tensor(path: &Path) -> Vec<f32> {
    let buf = fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let st = SafeTensors::deserialize(&buf).expect("ref safetensors deserialize");
    let view = st.tensor("x").expect("ref tensor 'x' missing");
    decode_f32(view.data())
}

/// Load a named bf16 weight tensor from the subset safetensors file.
fn load_bf16_weight(st: &SafeTensors<'_>, name: &str) -> Vec<f32> {
    let view = st.tensor(name).unwrap_or_else(|_| panic!("missing weight: {name}"));
    decode_bf16(view.data())
}

// =====================================================================
// NRMSE.
// =====================================================================

/// Normalized root-mean-square error: ||a - b|| / ||b||.
/// Returns NaN if ||b|| == 0 (which would mean reference is all-zeros).
fn nrmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "NRMSE length mismatch: {} vs {}", a.len(), b.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let d = (ai - bi) as f64;
        num += d * d;
        den += (bi as f64) * (bi as f64);
    }
    (num.sqrt() / den.sqrt()) as f32
}

fn report(label: &str, value: f32) -> bool {
    let pass = value < ULP_THRESHOLD_BF16;
    println!(
        "  {} NRMSE = {value:.3e}  {}",
        if pass { "PASS" } else { "FAIL" },
        label,
    );
    pass
}

// =====================================================================
// Component implementations (CPU, fp32 accumulator).
// These are the "this is what hipfire's RDNA kernel will compute"
// references; if they pass against PyTorch here, the kernel's job is
// to match these on GPU.
// =====================================================================

/// ZayaRMSNorm forward.
///
/// Math: hs_f32 = hs.to(f32); var = hs_f32.pow(2).mean(-1, keepdim);
///       hs_f32 = hs_f32 * rsqrt(var + eps);
///       out    = weight * hs_f32.to(input_dtype)
///
/// CPU impl always works in f32; the cast-back-to-bf16 step at the end
/// of PyTorch's forward is what bounds the achievable NRMSE to ~bf16
/// ULP. We mimic that to match the reference output.
fn zaya_rmsnorm_cpu(input: &[f32], weight_bf16_as_f32: &[f32], hidden_size: usize, eps: f32) -> Vec<f32> {
    assert!(input.len() % hidden_size == 0, "input not divisible by hidden_size");
    let n_rows = input.len() / hidden_size;
    let mut out = vec![0.0f32; input.len()];
    for r in 0..n_rows {
        let row = &input[r * hidden_size..(r + 1) * hidden_size];
        // Variance in f32 (matches PyTorch)
        let mut sum_sq = 0.0f64;
        for &x in row {
            sum_sq += (x as f64) * (x as f64);
        }
        let var = (sum_sq / hidden_size as f64) as f32;
        let inv_rms = 1.0f32 / (var + eps).sqrt();
        // Multiply, then cast-to-bf16-and-back, then multiply by weight.
        // PyTorch does: (hs_f32 * inv_rms).to(input_dtype) * weight
        // (input_dtype = bf16 in this case; weight is also bf16). The
        // cast happens BEFORE the weight multiply.
        for c in 0..hidden_size {
            let scaled = row[c] * inv_rms;
            // simulate the bf16 round-trip
            let scaled_bf16 = bf16::from_f32(scaled);
            let scaled_back = f32::from(scaled_bf16);
            out[r * hidden_size + c] = scaled_back * weight_bf16_as_f32[c];
        }
    }
    out
}

/// ResidualScaling forward.
///
/// Math (from modeling_zaya.py:907-914):
///   hidden_states = (hidden_states + hidden_states_bias) * hidden_states_scale
///   if not_first_layer:
///       residual = (residual + residual_bias) * residual_scale
///   return (residual, hidden_states)
///
/// First layer: residual is passed through unchanged (no res params).
fn residual_scaling_cpu(
    input: &[f32],
    bias: &[f32],
    scale: &[f32],
    hidden_size: usize,
) -> Vec<f32> {
    assert!(input.len() % hidden_size == 0);
    assert_eq!(bias.len(), hidden_size);
    assert_eq!(scale.len(), hidden_size);
    let n_rows = input.len() / hidden_size;
    let mut out = vec![0.0f32; input.len()];
    for r in 0..n_rows {
        for c in 0..hidden_size {
            out[r * hidden_size + c] = (input[r * hidden_size + c] + bias[c]) * scale[c];
        }
    }
    out
}

// =====================================================================
// Top-level harness.
// =====================================================================

fn main() {
    let args = parse_args();
    println!("[zaya-cpu-validate]");
    println!("  weights: {}", args.weights.display());
    println!("  refs   : {}", args.refs.display());

    let wbuf = fs::read(&args.weights).expect("read weights");
    let weights = SafeTensors::deserialize(&wbuf).expect("weights safetensors deserialize");

    let mut all_pass = true;
    let hidden = 2048;
    let eps = 1e-5f32;

    // -----------------------------------------------------------------
    // 1. ZayaRMSNorm on layer 0 input_norm.
    // -----------------------------------------------------------------
    println!("\n=== ZayaRMSNorm @ layer 0 input_norm ===");
    let w_norm0 = load_bf16_weight(&weights, "model.layers.0.input_norm.weight");
    assert_eq!(w_norm0.len(), hidden);
    let in_norm0 = load_ref_tensor(&args.refs.join("layer_00/prefill.input_norm.in.safetensors"));
    let out_norm0 = load_ref_tensor(&args.refs.join("layer_00/prefill.input_norm.out.safetensors"));
    let my_out0 = zaya_rmsnorm_cpu(&in_norm0, &w_norm0, hidden, eps);
    all_pass &= report("layer_00 input_norm", nrmse(&my_out0, &out_norm0));

    // Layer 1 input_norm (sanity check second layer matches too).
    println!("\n=== ZayaRMSNorm @ layer 1 input_norm ===");
    let w_norm1 = load_bf16_weight(&weights, "model.layers.1.input_norm.weight");
    let in_norm1 = load_ref_tensor(&args.refs.join("layer_01/prefill.input_norm.in.safetensors"));
    let out_norm1 = load_ref_tensor(&args.refs.join("layer_01/prefill.input_norm.out.safetensors"));
    let my_out1 = zaya_rmsnorm_cpu(&in_norm1, &w_norm1, hidden, eps);
    all_pass &= report("layer_01 input_norm", nrmse(&my_out1, &out_norm1));

    // -----------------------------------------------------------------
    // 2. ResidualScaling @ layer 0 (hidden_states only; first layer
    //    skips residual transform per `not_first_layer=False`).
    // -----------------------------------------------------------------
    println!("\n=== ResidualScaling @ layer 0 (hidden_states only) ===");
    let bias_h0 = load_bf16_weight(&weights, "model.layers.0.res_scale.hidden_states_bias");
    let scale_h0 = load_bf16_weight(&weights, "model.layers.0.res_scale.hidden_states_scale");
    let in_res0 = load_ref_tensor(&args.refs.join("layer_00/prefill.res_scale.in1.safetensors"));
    let out_res0 = load_ref_tensor(&args.refs.join("layer_00/prefill.res_scale.out1.safetensors"));
    let my_res0 = residual_scaling_cpu(&in_res0, &bias_h0, &scale_h0, hidden);
    all_pass &= report("layer_00 res_scale.hidden_states", nrmse(&my_res0, &out_res0));

    // -----------------------------------------------------------------
    // 3. ResidualScaling @ layer 1 (BOTH residual and hidden_states
    //    paths; `not_first_layer=True`).
    // -----------------------------------------------------------------
    println!("\n=== ResidualScaling @ layer 1 (residual path) ===");
    let bias_r1 = load_bf16_weight(&weights, "model.layers.1.res_scale.residual_bias");
    let scale_r1 = load_bf16_weight(&weights, "model.layers.1.res_scale.residual_scale");
    let in_res1_r = load_ref_tensor(&args.refs.join("layer_01/prefill.res_scale.in0.safetensors"));
    let out_res1_r = load_ref_tensor(&args.refs.join("layer_01/prefill.res_scale.out0.safetensors"));
    let my_res1_r = residual_scaling_cpu(&in_res1_r, &bias_r1, &scale_r1, hidden);
    all_pass &= report("layer_01 res_scale.residual", nrmse(&my_res1_r, &out_res1_r));

    println!("\n=== ResidualScaling @ layer 1 (hidden_states path) ===");
    let bias_h1 = load_bf16_weight(&weights, "model.layers.1.res_scale.hidden_states_bias");
    let scale_h1 = load_bf16_weight(&weights, "model.layers.1.res_scale.hidden_states_scale");
    let in_res1_h = load_ref_tensor(&args.refs.join("layer_01/prefill.res_scale.in1.safetensors"));
    let out_res1_h = load_ref_tensor(&args.refs.join("layer_01/prefill.res_scale.out1.safetensors"));
    let my_res1_h = residual_scaling_cpu(&in_res1_h, &bias_h1, &scale_h1, hidden);
    all_pass &= report("layer_01 res_scale.hidden_states", nrmse(&my_res1_h, &out_res1_h));

    // -----------------------------------------------------------------
    // 4. ZayaRMSNorm on final_norm (sanity).
    // -----------------------------------------------------------------
    println!("\n=== ZayaRMSNorm @ final_norm ===");
    let w_final = load_bf16_weight(&weights, "model.final_norm.weight");
    let in_final = load_ref_tensor(&args.refs.join("final/prefill.final_norm.in.safetensors"));
    let out_final = load_ref_tensor(&args.refs.join("final/prefill.final_norm.out.safetensors"));
    let my_final = zaya_rmsnorm_cpu(&in_final, &w_final, hidden, eps);
    all_pass &= report("final_norm", nrmse(&my_final, &out_final));

    println!();
    if all_pass {
        println!("=== ALL PASS ({} threshold = {:.0e}) ===", "bf16-ULP", ULP_THRESHOLD_BF16);
        std::process::exit(0);
    } else {
        eprintln!("=== FAILURES detected (threshold {:.0e}) ===", ULP_THRESHOLD_BF16);
        std::process::exit(1);
    }
}
