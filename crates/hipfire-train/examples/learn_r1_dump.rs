//! SpinQuant item 2 (deploy): learn the residual rotation R1 on a real model and
//! **dump it to disk** as the input to `hipfire-quantize --rotate`.
//!
//! Emits the raw learned rotation `M` (row-major `f32 [h*h]`, little-endian) — NOT
//! the baked `FᵀM`. The quantizer re-derives `F = block_fwht` from its own codec
//! seeds and composes `Fᵀ M` itself, so the on-disk artifact is codec-agnostic:
//! one file, `<magic><h:u32>` header then `h*h` f32. The quantizer applies
//! `apply_r1(FᵀM)`'s host math to the weights before the Oq4G256 quantize, whose
//! per-group FWHT then cancels the `Fᵀ`, leaving the int4 grid in the learned `M`
//! basis (the +1.7 dB optimum from `learned_r1_w4a4_probe`).
//!
//! This is offline tooling (learns on captured activations, writes a file); the
//! deploy-time rotation merge lives in `hipfire-quantize` per the AGENTS invariant.
//!
//! Run:
//!   source ./scripts/rocm-env.sh
//!   hipfire lock acquire "learn-r1-dump"
//!   cargo run -p hipfire-train --release --example learn_r1_dump -- <model_dir> <out.r1>
//!   hipfire lock release

use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{model_forward, LlamaModel};
use hipfire_train::rotation::{apply_r1, bake_for_oq4_recipe, Rotation};
use rdna_compute::Gpu;
use std::io::Write;
use std::path::Path;

const DEFAULT_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const SEQ: usize = 16;
/// File magic: "HFR1" (hipfire rotation, R1). Followed by `h: u32 LE`, then `h*h`
/// f32 LE (row-major). Kept trivially simple — the quantizer reads the same shape.
const MAGIC: [u8; 4] = *b"HFR1";

/// Capture the residual-read activations (xn1 + xn2 per layer) that R1 flattens,
/// mirroring `learned_r1_w4a4_probe`, and learn the kurtosis-minimizing rotation.
fn learn_r1(
    gpu: &mut Gpu,
    model: &LlamaModel,
    tokens: &[u32],
    pos: &[f32],
    h: usize,
) -> Result<Rotation, Box<dyn std::error::Error>> {
    let acts = model_forward(gpu, model, tokens, pos)?;
    let nl = model.layers.len();
    let mut xres = Vec::with_capacity(nl * 2 * SEQ * h);
    for i in 0..nl {
        xres.extend_from_slice(&gpu.download_f32(&acts.layer_acts[i].xn1)?);
    }
    for i in 0..nl {
        xres.extend_from_slice(&gpu.download_f32(&acts.layer_acts[i].xn2)?);
    }
    let rows = nl * 2 * SEQ;
    Ok(hipfire_train::learn_rotation::learn_rotation_kurtosis(
        &xres,
        rows,
        h,
        Rotation::hadamard(h, 1),
        120,
        0.05,
        6,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut argv = std::env::args().skip(1);
    let dir = argv.next().unwrap_or_else(|| DEFAULT_DIR.to_string());
    let out = argv.next().unwrap_or_else(|| "supra50m.r1".to_string());
    let dir = Path::new(&dir);
    if !dir.exists() {
        return Err(format!("model dir not found: {} (argv[1])", dir.display()).into());
    }
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}  model: {}", gpu.arch, dir.display());

    let (cfg, w) = load_llama_fp32(&mut gpu, dir).map_err(|e| format!("load: {e}"))?;
    let h = cfg.hidden_size;
    if h % 256 != 0 || !h.is_power_of_two() {
        return Err(
            format!("hidden {h} must be power-of-two & %256 for the codec FWHT bake").into(),
        );
    }
    let mut model = LlamaModel::from_f32_weights(&mut gpu, &cfg, w, SEQ, 2, 1.0)?;
    apply_r1(&mut gpu, &mut model, &Rotation::identity(h))?; // fold only (untie); R1=I basis

    let tokens: Vec<u32> = (0..SEQ)
        .map(|t| (13 + t * 97) as u32 % cfg.vocab_size as u32)
        .collect();
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();

    println!("learning R1 (kurtosis on residual xn1+xn2, hidden dim {h}) …");
    let r1 = learn_r1(&mut gpu, &model, &tokens, &pos, h)?;
    println!("  orthonormality {:.1e}", r1.orthonormality_error());

    // Sanity: the baked FᵀM stays orthonormal and apply_r1(FᵀM) leaves the fp
    // forward invariant (the deployment carries this bake; the codec FWHT cancels
    // the Fᵀ). We prove invariance on a *fresh* model against the fold-only logits.
    let baked = bake_for_oq4_recipe(&r1);
    println!(
        "  baked FᵀM orthonormality {:.1e}",
        baked.orthonormality_error()
    );
    let acts_fold = model_forward(&mut gpu, &model, &tokens, &pos)?;
    let logits_fold = gpu.download_f32(&acts_fold.logits)?;
    let (cfg2, w2) = load_llama_fp32(&mut gpu, dir).map_err(|e| format!("reload: {e}"))?;
    let mut m_baked = LlamaModel::from_f32_weights(&mut gpu, &cfg2, w2, SEQ, 2, 1.0)?;
    apply_r1(&mut gpu, &mut m_baked, &baked)?;
    let acts_baked = model_forward(&mut gpu, &m_baked, &tokens, &pos)?;
    let logits_baked = gpu.download_f32(&acts_baked.logits)?;
    let worst = logits_fold
        .iter()
        .zip(&logits_baked)
        .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
    println!("  apply_r1(FᵀM) fp invariance vs fold-only: max|Δlogit| {worst:.2e}");
    if worst > 5e-3 {
        return Err(format!("bake invariance failed (max|Δ| {worst:.2e}) — not writing").into());
    }

    // Write raw learned M (the quantizer composes FᵀM itself).
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out)?);
    f.write_all(&MAGIC)?;
    f.write_all(&(h as u32).to_le_bytes())?;
    let mut bytes = Vec::with_capacity(h * h * 4);
    for &v in &r1.r {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&bytes)?;
    f.flush()?;
    println!(
        "wrote {out}  ({} bytes: HFR1 + h={h} + {}×f32)",
        8 + h * h * 4,
        h * h
    );
    Ok(())
}
