//! SpinQuant R1 Phase 2: **learn** the residual rotation and show it beats the
//! fixed Hadamard on the A4 activation grid of a real model.
//!
//! Capture the deployed (fold-only) residual activations `X` and reader weights
//! `W` from one forward, then Cayley-SGD a rotation `R` minimizing the per-element
//! 4th moment (kurtosis) of `X Rᵀ` — the differentiable incoherence proxy. Since
//! a rotation is a pure basis change of the captured tensors, every candidate is
//! evaluated host-side: for rotation `R`, the deployed activation is `X Rᵀ` and
//! weight `W Rᵀ`, and the end-to-end A4 output SNR is
//!   SNR( (X Rᵀ)(W Rᵀ)ᵀ ,  A4(X Rᵀ)(W Rᵀ)ᵀ )   [ = SNR(X Wᵀ, …), R-invariant fp ].
//! Compares identity (fold only) vs fixed Hadamard vs learned R, for q_proj and
//! gate_proj (mean over layers). Expected: learned ≥ Hadamard > identity.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire lock acquire "learned-r1-probe"
//!   cargo run -p hipfire-train --release --example learned_r1_probe
//!   hipfire lock release

use hipfire_train::a4_quant::{a4_simquant, snr_db};
use hipfire_train::learn_rotation::{kurtosis_objective, learn_rotation_kurtosis};
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{model_forward, LlamaModel};
use hipfire_train::rotation::{apply_r1, rotate_rows, Rotation};
use hipfire_rdna::Gpu;
use std::path::Path;

const DEFAULT_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const SEQ: usize = 16;

/// `y = a Wᵀ`, `a [rows,h]`, `w [out,h]`.
fn matmul_t(a: &[f32], w: &[f32], rows: usize, h: usize, out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out];
    for r in 0..rows {
        for o in 0..out {
            let mut acc = 0.0f32;
            for k in 0..h {
                acc += a[r * h + k] * w[o * h + k];
            }
            y[r * out + o] = acc;
        }
    }
    y
}

/// Mean A4 output SNR (dB) over layers for one projection, in rotation `rot`.
fn proj_snr(acts: &[Vec<f32>], ws: &[Vec<f32>], rot: &Rotation, h: usize, out: usize) -> f32 {
    let mut sum = 0.0f32;
    for (a, w) in acts.iter().zip(ws) {
        let ar = rotate_rows(a, rot, SEQ); // X Rᵀ
        let wr = rotate_rows(w, rot, out); // W Rᵀ
        let y = matmul_t(&ar, &wr, SEQ, h, out);
        let yq = matmul_t(&a4_simquant(&ar, SEQ, h), &wr, SEQ, h, out);
        sum += snr_db(&y, &yq);
    }
    sum / acts.len().max(1) as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_DIR.to_string());
    let dir = Path::new(&dir);
    if !dir.exists() {
        return Err(format!("model dir not found: {} (argv[1])", dir.display()).into());
    }
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}  model: {}", gpu.arch, dir.display());

    // One fold-only forward → capture deployed residual activations + weights.
    let (cfg, w) = load_llama_fp32(&mut gpu, dir).map_err(|e| format!("load: {e}"))?;
    let h = cfg.hidden_size;
    if !h.is_power_of_two() {
        return Err(format!("hidden {h} not power-of-two (Hadamard warm start)").into());
    }
    let mut model = LlamaModel::from_f32_weights(&mut gpu, &cfg, w, SEQ, 2, 1.0)?;
    apply_r1(&mut gpu, &mut model, &Rotation::identity(h))?; // fold only
    let qd = model.dims.q_dim();
    let inter = model.dims.inter;
    let tokens: Vec<u32> = (0..SEQ)
        .map(|t| (13 + t * 97) as u32 % cfg.vocab_size as u32)
        .collect();
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();
    let acts = model_forward(&mut gpu, &model, &tokens, &pos)?;

    let mut xn1 = Vec::new();
    let mut wq = Vec::new();
    let mut xn2 = Vec::new();
    let mut wgate = Vec::new();
    for (i, (lw, _)) in model.layers.iter().enumerate() {
        xn1.push(gpu.download_f32(&acts.layer_acts[i].xn1)?);
        wq.push(gpu.download_f32(&lw.wq)?);
        xn2.push(gpu.download_f32(&acts.layer_acts[i].xn2)?);
        wgate.push(gpu.download_f32(&lw.wgate)?);
    }

    // Learning set: all residual-read activations (xn1 + xn2) stacked [rows,h].
    let nl = model.layers.len();
    let rows = nl * 2 * SEQ;
    let mut xstack = Vec::with_capacity(rows * h);
    for m in xn1.iter().chain(xn2.iter()) {
        xstack.extend_from_slice(m);
    }

    println!("\n  learning R1 (Cayley-SGD, kurtosis surrogate) on {rows} rows × {h} …",);
    let had = Rotation::hadamard(h, 1);
    let k_id = kurtosis_objective(&xstack, &Rotation::identity(h), rows);
    let k_had = kurtosis_objective(&xstack, &had, rows);
    // Warm-start from the Hadamard (already a good incoherence basis) and refine.
    let learned = learn_rotation_kurtosis(&xstack, rows, h, had.clone(), 120, 0.05, 6);
    let k_learn = kurtosis_objective(&xstack, &learned, rows);
    println!(
        "  kurtosis Σx⁴:  identity={k_id:.3e}  hadamard={k_had:.3e}  learned={k_learn:.3e}  \
         (orthonormality {:.1e})",
        learned.orthonormality_error()
    );

    println!("\n  A4 output SNR (mean over layers):");
    println!("  rotation            q_proj      gate_proj");
    for (name, rot) in [
        ("identity (fold)", Rotation::identity(h)),
        ("fixed Hadamard", had.clone()),
        ("learned R1", learned.clone()),
    ] {
        let q = proj_snr(&xn1, &wq, &rot, h, qd);
        let g = proj_snr(&xn2, &wgate, &rot, h, inter);
        println!("  {name:<16} {q:8.2} dB   {g:8.2} dB");
    }

    let q_had = proj_snr(&xn1, &wq, &had, h, qd);
    let q_learn = proj_snr(&xn1, &wq, &learned, h, qd);
    println!(
        "\n  learned − fixed-Hadamard q_proj A4 SNR: {:+.2} dB",
        q_learn - q_had
    );
    if q_learn >= q_had {
        println!("PHASE 2: learned R1 matches/beats the fixed Hadamard on the A4 grid.");
    } else {
        println!("NOTE: learned did not beat fixed here (small model / already near-Gaussian).");
    }
    Ok(())
}
