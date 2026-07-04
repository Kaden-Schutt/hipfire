//! SpinQuant R1 Phase 1: measure how a residual rotation improves **A4
//! activation** quality on a *real* model. For each rotation (identity=fold-only,
//! Hadamard, random orthonormal) we bake `R1` into the weights, run the fp32
//! forward, and at each layer take the deployed residual activation feeding
//! q_proj (`xn1`) and gate_proj (`xn2`) and its weight, then measure the
//! end-to-end output SNR of the int4 activation round-trip:
//!
//!     y  = a · Wᵀ                 (fp reference, in the rotated basis)
//!     ŷ  = A4(a) · Wᵀ             (int4 per-256-group activation)
//!     SNR = 10·log10(‖y‖²/‖y−ŷ‖²)
//!
//! Both `a` and `W` are the *deployed* rotated tensors (α folded in, `R` applied)
//! — so identity vs Hadamard isolates R1's effect on the A4 grid. Rotation
//! disperses the outlier channels that otherwise inflate the shared int4 scale
//! and crush the bulk, so Hadamard/random should beat fold-only.
//!
//! Run (defaults to Supra-50M; pass a dense-llama dir as argv[1]):
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire lock acquire "rotation-a4-snr-probe"
//!   cargo run -p hipfire-train --release --example rotation_a4_snr_probe
//!   hipfire lock release

use hipfire_train::a4_quant::{a4_simquant, snr_db};
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{model_forward, LlamaModel};
use hipfire_train::rotation::{apply_r1, Rotation};
use hipfire_rdna::{Gpu, HipResult};
use std::path::Path;

const DEFAULT_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const SEQ: usize = 16;

/// `y = a Wᵀ`, `a [rows,h]`, `w [out,h]` row-major.
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

/// Mean end-to-end A4 output SNR (dB) of one projection over all layers: the
/// activation `a [seq,h]` (a per-layer norm output) quantized to int4 per-256-
/// group, pushed through the deployed weight `w [out,h]`.
fn proj_snr(acts_and_w: &[(Vec<f32>, Vec<f32>)], h: usize, out: usize) -> f32 {
    let mut sum = 0.0f32;
    for (a, w) in acts_and_w {
        let y = matmul_t(a, w, SEQ, h, out);
        let aq = a4_simquant(a, SEQ, h);
        let yq = matmul_t(&aq, w, SEQ, h, out);
        sum += snr_db(&y, &yq);
    }
    sum / acts_and_w.len().max(1) as f32
}

fn run_rotation(
    gpu: &mut Gpu,
    dir: &Path,
    rot: &Rotation,
    tokens: &[u32],
) -> HipResult<(f32, f32)> {
    // Fresh weights each time — apply_r1 mutates in place.
    let (cfg, w) = load_llama_fp32(gpu, dir).expect("load model");
    let model = LlamaModel::from_f32_weights(gpu, &cfg, w, SEQ, 2, 1.0)?;
    let mut model = model;
    apply_r1(gpu, &mut model, rot)?;
    let (h, qd, inter) = (model.dims.h, model.dims.q_dim(), model.dims.inter);
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();
    let acts = model_forward(gpu, &model, tokens, &pos)?;

    // Capture deployed xn1 (→q_proj) and xn2 (→gate_proj) + their weights.
    let mut q_data = Vec::with_capacity(model.layers.len());
    let mut g_data = Vec::with_capacity(model.layers.len());
    for (i, (lw, _)) in model.layers.iter().enumerate() {
        let xn1 = gpu.download_f32(&acts.layer_acts[i].xn1)?;
        let wq = gpu.download_f32(&lw.wq)?;
        q_data.push((xn1, wq));
        let xn2 = gpu.download_f32(&acts.layer_acts[i].xn2)?;
        let wgate = gpu.download_f32(&lw.wgate)?;
        g_data.push((xn2, wgate));
    }
    let q_snr = proj_snr(&q_data, h, qd);
    let g_snr = proj_snr(&g_data, h, inter);
    Ok((q_snr, g_snr))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_DIR.to_string());
    let dir = Path::new(&dir);
    if !dir.exists() {
        return Err(format!(
            "model dir not found: {} (pass a dense-llama dir as argv[1])",
            dir.display()
        )
        .into());
    }

    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);
    println!("model: {}", dir.display());

    // Arbitrary but valid token ids (< vocab); the outlier channels that drive
    // A4 damage are largely weight-driven, not sensitive to the exact prompt.
    let tokens: Vec<u32> = (0..SEQ).map(|t| (13 + t * 97) as u32 % 32000).collect();

    let h = {
        let (cfg, _) = hipfire_train::config::LlamaConfig::from_dir(dir)
            .map(|c| (c, ()))
            .map_err(|e| format!("config: {e}"))?;
        cfg.hidden_size
    };
    if !h.is_power_of_two() {
        return Err(
            format!("hidden {h} not a power of two — Hadamard R1 needs power-of-two h").into(),
        );
    }

    println!("\n  rotation            q_proj SNR   gate_proj SNR");
    let cases: [(&str, Rotation); 3] = [
        ("identity (fold only)", Rotation::identity(h)),
        ("hadamard R1", Rotation::hadamard(h, 1)),
        ("random R1", Rotation::random(h, 1)),
    ];
    let mut ident_q = 0.0f32;
    let mut had_q = 0.0f32;
    for (name, rot) in &cases {
        let (q, g) = run_rotation(&mut gpu, dir, rot, &tokens)?;
        println!("  {name:<20} {q:8.2} dB   {g:8.2} dB");
        if *name == "identity (fold only)" {
            ident_q = q;
        }
        if *name == "hadamard R1" {
            had_q = q;
        }
    }

    let delta = had_q - ident_q;
    println!("\n  Hadamard−identity q_proj A4 SNR gain: {delta:+.2} dB");
    if delta > 0.0 {
        println!("PHASE 1 A4-SNR: rotation improves the int4-activation grid (as expected).");
    } else {
        println!(
            "NOTE: no gain here — activations may lack strong outliers at this scale; \
             try a larger checkpoint."
        );
    }
    Ok(())
}
