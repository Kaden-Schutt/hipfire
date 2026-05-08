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
// Math primitives (CPU, fp32).
// =====================================================================

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// PyTorch torch.nn.functional.gelu default = "tanh" approximation? No,
/// default is the EXACT erf-based gelu. Zyphra's router_mlp uses
/// `nn.GELU()` (line 978), which defaults to `approximate="none"` = the
/// exact erf form: 0.5 * x * (1 + erf(x / sqrt(2))).
#[inline]
fn gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
}

/// Stable softmax over the last axis.
fn softmax_last(rows: &mut [f32], cols: usize) {
    assert!(rows.len() % cols == 0);
    let nrows = rows.len() / cols;
    for r in 0..nrows {
        let row = &mut rows[r * cols..(r + 1) * cols];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

/// y[m, n] = x[m, k] @ W[n, k].T + bias?  (PyTorch nn.Linear style.)
/// Inputs are row-major. `weight` has shape [n_out, n_in].
fn linear(x: &[f32], weight: &[f32], bias: Option<&[f32]>, m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(x.len(), m * k);
    assert_eq!(weight.len(), n * k);
    if let Some(b) = bias {
        assert_eq!(b.len(), n);
    }
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            let xi = i * k;
            let wj = j * k;
            for kk in 0..k {
                acc += (x[xi + kk] as f64) * (weight[wj + kk] as f64);
            }
            let mut v = acc as f32;
            if let Some(b) = bias {
                v += b[j];
            }
            out[i * n + j] = v;
        }
    }
    out
}

/// argmax of every row (used for top-1 routing).
fn argmax_rows(rows: &[f32], cols: usize) -> Vec<usize> {
    rows.chunks_exact(cols)
        .map(|row| {
            let mut best = 0usize;
            let mut bv = row[0];
            for (i, &v) in row.iter().enumerate().skip(1) {
                if v > bv {
                    bv = v;
                    best = i;
                }
            }
            best
        })
        .collect()
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

/// SwiGLU activation chunk: silu(x[..N/2]) * x[N/2..]. Operates in place
/// on the input rows; returns a new Vec of half-width.
fn swiglu_chunk(input: &[f32], full_dim: usize) -> Vec<f32> {
    assert_eq!(full_dim % 2, 0);
    let half = full_dim / 2;
    assert_eq!(input.len() % full_dim, 0);
    let n_rows = input.len() / full_dim;
    let mut out = vec![0.0f32; n_rows * half];
    for r in 0..n_rows {
        let src = &input[r * full_dim..(r + 1) * full_dim];
        let dst = &mut out[r * half..(r + 1) * half];
        for c in 0..half {
            dst[c] = silu(src[c]) * src[half + c];
        }
    }
    out
}

/// ZAYA1 partial RoPE per modeling_zaya.py:455 (apply_rotary_pos_emb).
///   rotary_dim = cos.shape[-1]
///   q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
///   q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
///   q_embed = cat((q_rot, q_pass), dim=-1)
/// rotate_half: x1 = x[..N/2], x2 = x[N/2..]; return cat((-x2, x1)).
///
/// Operates per-row; row layout is [head_dim] contiguous (caller is
/// responsible for unbroadcasting cos/sin appropriately if heads share).
fn partial_rope_apply(
    q: &[f32],
    cos_per_row: &[f32],
    sin_per_row: &[f32],
    rotary_dim: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(q.len() % head_dim, 0);
    assert_eq!(cos_per_row.len() % rotary_dim, 0);
    assert_eq!(sin_per_row.len() % rotary_dim, 0);
    assert_eq!(rotary_dim % 2, 0);
    let n_rows = q.len() / head_dim;
    let cos_rows = cos_per_row.len() / rotary_dim;
    assert!(cos_rows == n_rows || cos_rows == 1, "cos rows {} vs q rows {}", cos_rows, n_rows);
    let half = rotary_dim / 2;
    let mut out = vec![0.0f32; q.len()];
    for r in 0..n_rows {
        let qrow = &q[r * head_dim..(r + 1) * head_dim];
        let crow = if cos_rows == n_rows {
            &cos_per_row[r * rotary_dim..(r + 1) * rotary_dim]
        } else {
            &cos_per_row[..rotary_dim]
        };
        let srow = if cos_rows == n_rows {
            &sin_per_row[r * rotary_dim..(r + 1) * rotary_dim]
        } else {
            &sin_per_row[..rotary_dim]
        };
        let orow = &mut out[r * head_dim..(r + 1) * head_dim];
        // q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
        // rotate_half: x[..half] -> -x[half..], x[half..] -> x[..half]
        for i in 0..half {
            let a = qrow[i];
            let b = qrow[half + i];
            orow[i] = a * crow[i] + (-b) * srow[i];
            orow[half + i] = b * crow[half + i] + a * srow[half + i];
        }
        // q_pass: copy unchanged
        for i in rotary_dim..head_dim {
            orow[i] = qrow[i];
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

    // -----------------------------------------------------------------
    // 5. o_proj @ layer 0 (post-attention output projection).
    //    The CCA-fed attention path's tail. o_proj input is 1024-dim
    //    (8q heads x 128 head_dim, the "//2 query compression"); output
    //    is 2048 = hidden_size. With no bias on o_proj.
    // -----------------------------------------------------------------
    println!("\n=== o_proj @ layer 0 (1024 -> 2048) ===");
    let w_oproj0 = load_bf16_weight(&weights, "model.layers.0.self_attn.o_proj.weight");
    let in_oproj0 = load_ref_tensor(&args.refs.join("layer_00/prefill.o_proj.in.safetensors"));
    let out_oproj0 = load_ref_tensor(&args.refs.join("layer_00/prefill.o_proj.out.safetensors"));
    // o_proj.weight shape: [out=2048, in=1024]. PyTorch nn.Linear y = x @ W.T
    let n_tokens = in_oproj0.len() / 1024;
    let my_oproj0 = linear(&in_oproj0, &w_oproj0, None, n_tokens, 1024, 2048);
    all_pass &= report("layer_00 o_proj", nrmse(&my_oproj0, &out_oproj0));
    // o_proj.out should equal self_attn.out0 (o_proj is the last op).
    let self_attn_out0 = load_ref_tensor(&args.refs.join("layer_00/prefill.self_attn.out0.safetensors"));
    all_pass &= report("layer_00 o_proj == self_attn.out0", nrmse(&out_oproj0, &self_attn_out0));

    // -----------------------------------------------------------------
    // 6. MLP-based MoE router @ layer 1 (no EDA on first MoE layer).
    //    ZayaRouter.forward (modeling_zaya.py:992):
    //      hs = down_proj(input)              # [B, S, 256]
    //      router_hidden_states_next = hs.clone()
    //      hs_norm = rmsnorm_eda(hs)          # [B, S, 256]
    //      logits  = router_mlp(hs_norm)      # [B, S, 17]
    //      expert_prob = softmax(logits, -1)
    //      biased = expert_prob.f32() + balancing_biases
    //      expert_choice = topk(biased, k=1)  # [B, S, 1]
    //      route_prob = gather(expert_prob, dim=2, index=expert_choice)
    //      return (route_prob_flat, expert_choice_flat, router_hidden_states_next)
    // -----------------------------------------------------------------
    println!("\n=== MLP-based MoE router @ layer 1 (top-1, no EDA) ===");
    let mlp_exp = 256;
    let n_experts_with_skip = 17; // 16 experts + 1 MoD skip slot
    let dp_w = load_bf16_weight(&weights, "model.layers.1.zaya_block.router.down_proj.weight");
    let dp_b = load_bf16_weight(&weights, "model.layers.1.zaya_block.router.down_proj.bias");
    let rmsnorm_eda_w = load_bf16_weight(&weights, "model.layers.1.zaya_block.router.rmsnorm_eda.weight");
    let mlp0_w = load_bf16_weight(&weights, "model.layers.1.zaya_block.router.router_mlp.0.weight");
    let mlp0_b = load_bf16_weight(&weights, "model.layers.1.zaya_block.router.router_mlp.0.bias");
    let mlp2_w = load_bf16_weight(&weights, "model.layers.1.zaya_block.router.router_mlp.2.weight");
    let mlp2_b = load_bf16_weight(&weights, "model.layers.1.zaya_block.router.router_mlp.2.bias");
    let mlp4_w = load_bf16_weight(&weights, "model.layers.1.zaya_block.router.router_mlp.4.weight");
    let bal_biases = load_bf16_weight(&weights, "model.layers.1.zaya_block.router.balancing_biases");

    let router_in = load_ref_tensor(&args.refs.join("layer_01/prefill.router.in.safetensors"));
    let n_tok = router_in.len() / hidden;

    // 1) hs = down_proj(input)
    let hs_post_dp = linear(&router_in, &dp_w, Some(&dp_b), n_tok, hidden, mlp_exp);
    // 2) router_hidden_states_next = hs.clone() (validates against router.out2)
    let ref_router_out2 = load_ref_tensor(&args.refs.join("layer_01/prefill.router.out2.safetensors"));
    all_pass &= report("layer_01 router_hidden_states_next (down_proj)", nrmse(&hs_post_dp, &ref_router_out2));

    // 3) hs_norm = rmsnorm(hs)
    let hs_norm = zaya_rmsnorm_cpu(&hs_post_dp, &rmsnorm_eda_w, mlp_exp, eps);
    // 4) router_mlp: Linear(256,256)+bias -> GELU -> Linear(256,256)+bias -> GELU -> Linear(256,17)
    let h0 = linear(&hs_norm, &mlp0_w, Some(&mlp0_b), n_tok, mlp_exp, mlp_exp);
    let h0_act: Vec<f32> = h0.iter().map(|&x| gelu_exact(x)).collect();
    let h2 = linear(&h0_act, &mlp2_w, Some(&mlp2_b), n_tok, mlp_exp, mlp_exp);
    let h2_act: Vec<f32> = h2.iter().map(|&x| gelu_exact(x)).collect();
    let logits = linear(&h2_act, &mlp4_w, None, n_tok, mlp_exp, n_experts_with_skip);
    // 5) softmax
    let mut expert_prob = logits.clone();
    softmax_last(&mut expert_prob, n_experts_with_skip);
    // 6) biased = expert_prob (already f32) + balancing_biases
    let mut biased = expert_prob.clone();
    for r in 0..n_tok {
        for c in 0..n_experts_with_skip {
            biased[r * n_experts_with_skip + c] += bal_biases[c];
        }
    }
    // 7) top-1
    let my_expert_choice: Vec<usize> = argmax_rows(&biased, n_experts_with_skip);
    // 8) route_prob = gather expert_prob @ expert_choice
    let my_route_prob: Vec<f32> = my_expert_choice.iter().enumerate()
        .map(|(r, &c)| expert_prob[r * n_experts_with_skip + c])
        .collect();

    // Compare to dump
    let ref_route_prob = load_ref_tensor(&args.refs.join("layer_01/prefill.router.out0.safetensors"));
    let ref_expert_choice = load_ref_tensor(&args.refs.join("layer_01/prefill.router.out1.safetensors"));
    all_pass &= report("layer_01 router route_prob", nrmse(&my_route_prob, &ref_route_prob));

    let ref_choices_int: Vec<usize> = ref_expert_choice.iter().map(|&v| v as usize).collect();
    let mismatches = my_expert_choice.iter().zip(ref_choices_int.iter()).filter(|(a, b)| a != b).count();
    println!("  {} expert_choice exact match: {}/{} mismatches",
        if mismatches == 0 { "PASS" } else { "FAIL" },
        mismatches, n_tok);
    all_pass &= mismatches == 0;
    println!("  per-token expert assignment: {:?}", &my_expert_choice);

    // -----------------------------------------------------------------
    // 7. SwiGLU + Expert 0 MLP @ layer 1.
    //    Expert MLP per modeling_zaya.py:1063 (no biases; gated_linear_unit=true):
    //      x = linear_fc1(input)   # [N, ffn_hidden=4096]
    //      gated = silu(x[:, :2048]) * x[:, 2048:]    # [N, 2048]
    //      out = linear_fc2(gated)  # [N, hidden=2048]
    //    `experts.in0` is sorted by expert_choice; tokens going to
    //    expert 0 occupy the first `tokens_per_expert[0]` rows.
    // -----------------------------------------------------------------
    println!("\n=== SwiGLU + expert 0 MLP @ layer 1 ===");
    let n_to_expert_0 = ref_choices_int.iter().filter(|&&c| c == 0).count();
    if n_to_expert_0 == 0 {
        println!("  SKIP no tokens routed to expert 0 in this prompt; cannot validate");
    } else {
        let fc1_w = load_bf16_weight(&weights, "model.layers.1.zaya_block.experts.local_experts.0.linear_fc1.weight");
        let fc2_w = load_bf16_weight(&weights, "model.layers.1.zaya_block.experts.local_experts.0.linear_fc2.weight");
        let experts_in = load_ref_tensor(&args.refs.join("layer_01/prefill.experts.in0.safetensors"));
        let experts_out = load_ref_tensor(&args.refs.join("layer_01/prefill.experts.out0.safetensors"));
        let ffn_hidden = 4096;
        // experts.in0 is [N_total, 2048] sorted by expert_choice. Expert 0 is
        // the FIRST bin in the sort because expert ids are non-negative ints.
        let in_e0 = &experts_in[..n_to_expert_0 * hidden];
        let out_e0 = &experts_out[..n_to_expert_0 * hidden];

        // x = linear_fc1(input)
        let post_fc1 = linear(in_e0, &fc1_w, None, n_to_expert_0, hidden, ffn_hidden);
        // SwiGLU
        let gated = swiglu_chunk(&post_fc1, ffn_hidden);
        // linear_fc2
        let my_out_e0 = linear(&gated, &fc2_w, None, n_to_expert_0, ffn_hidden / 2, hidden);
        all_pass &= report(&format!("layer_01 expert 0 MLP ({} tokens)", n_to_expert_0), nrmse(&my_out_e0, out_e0));
    }

    // -----------------------------------------------------------------
    // 8. partial-RoPE math (unit test, no PyTorch ref needed).
    //    Validates that the impl matches PyTorch's apply_rotary_pos_emb
    //    semantics (modeling_zaya.py:455) for partial_rotary_factor=0.5.
    //    Three properties tested:
    //      a. cos=1, sin=0 : identity on first 64 dims.
    //      b. last 64 dims always pass through unchanged.
    //      c. cos=0, sin=1 : 90-deg rotation matches rotate_half formula.
    // -----------------------------------------------------------------
    println!("\n=== partial-RoPE math (head_dim=128, rotary_dim=64) ===");
    {
        let head_dim = 128;
        let rotary_dim = 64; // partial_rotary_factor=0.5
        let n_pos = 4;
        // Synthetic Q: per row [r*100 + 0, r*100 + 1, ..., r*100 + 127]
        let q: Vec<f32> = (0..n_pos)
            .flat_map(|r| (0..head_dim).map(move |c| (r * 100 + c) as f32))
            .collect();

        // (a) identity check
        let cos_id = vec![1.0f32; n_pos * rotary_dim];
        let sin_id = vec![0.0f32; n_pos * rotary_dim];
        let q_rot_id = partial_rope_apply(&q, &cos_id, &sin_id, rotary_dim, head_dim);
        let mut identity_first64_err = 0.0f32;
        for r in 0..n_pos {
            for c in 0..rotary_dim {
                let e = (q[r * head_dim + c] - q_rot_id[r * head_dim + c]).abs();
                if e > identity_first64_err { identity_first64_err = e; }
            }
        }
        let pass_a = identity_first64_err == 0.0;
        println!("  {} (a) cos=1,sin=0 identity on first 64 dims (max abs err {:.3e})",
                 if pass_a { "PASS" } else { "FAIL" }, identity_first64_err);
        all_pass &= pass_a;

        // (b) passthrough on last 64 dims
        let mut pass_b = true;
        for r in 0..n_pos {
            for c in rotary_dim..head_dim {
                if q[r * head_dim + c] != q_rot_id[r * head_dim + c] {
                    pass_b = false;
                }
            }
        }
        println!("  {} (b) last 64 dims unchanged (passthrough)", if pass_b { "PASS" } else { "FAIL" });
        all_pass &= pass_b;

        // (c) cos=0, sin=1 -> rotate_half formula
        // rotate_half: x[..32] -> -x[32..64], x[32..64] -> x[..32]
        // result[..32] = -x[32..64], result[32..64] = x[..32]
        let cos_z = vec![0.0f32; n_pos * rotary_dim];
        let sin_one = vec![1.0f32; n_pos * rotary_dim];
        let q_rot_90 = partial_rope_apply(&q, &cos_z, &sin_one, rotary_dim, head_dim);
        let half = rotary_dim / 2;
        let mut max_err_c = 0.0f32;
        for r in 0..n_pos {
            for c in 0..half {
                let want_first = -q[r * head_dim + half + c];
                let want_second = q[r * head_dim + c];
                max_err_c = max_err_c.max((q_rot_90[r * head_dim + c] - want_first).abs());
                max_err_c = max_err_c.max((q_rot_90[r * head_dim + half + c] - want_second).abs());
            }
        }
        let pass_c = max_err_c == 0.0;
        println!("  {} (c) cos=0,sin=1 rotate_half formula (max abs err {:.3e})",
                 if pass_c { "PASS" } else { "FAIL" }, max_err_c);
        all_pass &= pass_c;
    }

    println!();
    if all_pass {
        println!("=== ALL PASS ({} threshold = {:.0e}) ===", "bf16-ULP", ULP_THRESHOLD_BF16);
        std::process::exit(0);
    } else {
        eprintln!("=== FAILURES detected (threshold {:.0e}) ===", ULP_THRESHOLD_BF16);
        std::process::exit(1);
    }
}
