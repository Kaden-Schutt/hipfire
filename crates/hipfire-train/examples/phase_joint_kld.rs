//! Phase-joint α-sweep, END-TO-END decode metric (the definitive test the per-tensor
//! SQNR sweep couldn't give). Exploits `apply_r1`'s proven fp-invariance: for a baked
//! rotation `FᵀM`, `apply_r1` produces a model that computes identical logits but
//! holds the *deployment* weights (int4-target, in the M basis). Fake-quantizing those
//! h-dim reader weights to symmetric int4 per-256-group and running the FULL forward
//! yields exactly the end-to-end **W-int4 / A-f16 decode** logits — all layers,
//! compounding, softmax. KLD vs the fp reference is the decode quality. Since R1 only
//! rotates the residual (h) dim, down_proj (reads on `inter`) stays fp and cancels, so
//! the KLD delta across α isolates the M effect. This is the metric that showed the
//! +22% regression for the act-only rotation; the sweep asks whether a joint α removes
//! it while keeping the prefill win (one buffer for both phases).
//!
//!   source ./scripts/rocm-env.sh
//!   export ROCM_PATH=$HOME/.venv/lib/python3.14/site-packages/_rocm_sdk_core
//!   hipfire lock acquire "phase-joint-kld"
//!   cargo run -p hipfire-train --release --example phase_joint_kld
//!   hipfire lock release

use hipfire_train::learn_rotation::{learn_rotation_kurtosis, learn_rotation_phase_joint};
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{model_forward, LlamaModel};
use hipfire_train::rotation::{apply_r1, bake_for_oq4_recipe, Rotation};
use hipfire_rdna::{Gpu, GpuTensor, HipResult};
use std::path::Path;

const DEFAULT_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const SEQ: usize = 16;
const GROUP: usize = 256;

/// Fake-quant a weight `[out, dim]` (dim = contraction) to symmetric int4 per
/// 256-group with clip-search, in place on the GPU (download → quantize → upload).
fn fake_quant_int4(gpu: &mut Gpu, t: &GpuTensor, out: usize, dim: usize) -> HipResult<()> {
    const GRID: [f32; 9] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6];
    let mut w = gpu.download_f32(t)?;
    for r in 0..out {
        for g in 0..(dim / GROUP) {
            let g0 = r * dim + g * GROUP;
            let grp = &w[g0..g0 + GROUP];
            let amax = grp.iter().fold(1e-12f32, |a, &v| a.max(v.abs()));
            let (mut bs, mut be) = (amax / 7.0, f32::INFINITY);
            for &cl in &GRID {
                let s = (cl * amax / 7.0).max(1e-12);
                let e: f32 = grp
                    .iter()
                    .map(|&v| {
                        let d = v - (v / s).round().clamp(-7.0, 7.0) * s;
                        d * d
                    })
                    .sum();
                if e < be {
                    be = e;
                    bs = s;
                }
            }
            for j in 0..GROUP {
                let v = w[g0 + j];
                w[g0 + j] = (v / bs).round().clamp(-7.0, 7.0) * bs;
            }
        }
    }
    let bytes = unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) };
    gpu.memcpy_htod_auto(&t.buf, bytes)
}

/// Mean per-position KLD( softmax(ref) || softmax(test) ) over the sequence.
fn kld(refl: &[f32], testl: &[f32], seq: usize, vocab: usize) -> f64 {
    let mut acc = 0.0f64;
    for s in 0..seq {
        let r = &refl[s * vocab..s * vocab + vocab];
        let t = &testl[s * vocab..s * vocab + vocab];
        let rmax = r.iter().cloned().fold(f32::MIN, f32::max);
        let tmax = t.iter().cloned().fold(f32::MIN, f32::max);
        let (mut rz, mut tz) = (0.0f64, 0.0f64);
        for v in 0..vocab {
            rz += ((r[v] - rmax) as f64).exp();
            tz += ((t[v] - tmax) as f64).exp();
        }
        for v in 0..vocab {
            let pr = ((r[v] - rmax) as f64).exp() / rz;
            let pt = ((t[v] - tmax) as f64).exp() / tz;
            if pr > 1e-12 {
                acc += pr * (pr.ln() - pt.max(1e-30).ln());
            }
        }
    }
    acc / seq as f64
}

/// Load a fresh model, `apply_r1(bake(rot))`, fake-quant the h-dim readers to int4,
/// forward, and return the logits. `rot` is the learned `M` (raw); `bake` composes
/// `FᵀM` so the codec FWHT cancels leaving the M-basis int4 grid.
fn decode_logits(
    gpu: &mut Gpu,
    dir: &Path,
    rot: &Rotation,
    tokens: &[u32],
    pos: &[f32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let (cfg, w) = load_llama_fp32(gpu, dir).map_err(|e| format!("reload: {e}"))?;
    let h = cfg.hidden_size;
    let mut m = LlamaModel::from_f32_weights(gpu, &cfg, w, SEQ, 2, 1.0)?;
    apply_r1(gpu, &mut m, &bake_for_oq4_recipe(rot))?;
    let (qd, kvd, inter) = (m.dims.q_dim(), m.dims.kv_dim(), m.dims.inter);
    // Fake-quant every h-dim (contraction=h, %256) reader to int4 — the tensors the
    // learned M rotates. down_proj (contraction=inter) is left fp: M-invariant.
    for (lw, _) in m.layers.iter() {
        fake_quant_int4(gpu, &lw.wq, qd, h)?;
        fake_quant_int4(gpu, &lw.wk, kvd, h)?;
        fake_quant_int4(gpu, &lw.wv, kvd, h)?;
        fake_quant_int4(gpu, &lw.wo, h, h)?;
        fake_quant_int4(gpu, &lw.wgate, inter, h)?;
        fake_quant_int4(gpu, &lw.wup, inter, h)?;
    }
    let acts = model_forward(gpu, &m, tokens, pos)?;
    Ok(gpu.download_f32(&acts.logits)?)
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

    let (cfg, w) = load_llama_fp32(&mut gpu, dir).map_err(|e| format!("load: {e}"))?;
    let h = cfg.hidden_size;
    if h % GROUP != 0 || !h.is_power_of_two() {
        return Err(format!("hidden {h} must be pow2 & %256").into());
    }
    let vocab = cfg.vocab_size;
    let tokens: Vec<u32> = (0..SEQ)
        .map(|t| (13 + t * 97) as u32 % cfg.vocab_size as u32)
        .collect();
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();

    // Reference: fold-only (R1=I), NO weight quant — the fp32 decode target.
    let mut model = LlamaModel::from_f32_weights(&mut gpu, &cfg, w, SEQ, 2, 1.0)?;
    apply_r1(&mut gpu, &mut model, &Rotation::identity(h))?;
    let ref_acts = model_forward(&mut gpu, &model, &tokens, &pos)?;
    let refl = gpu.download_f32(&ref_acts.logits)?;

    // Learn the rotations: activation set = residual xn1+xn2; H, weight set for joint.
    let acts = model_forward(&mut gpu, &model, &tokens, &pos)?;
    let nl = model.layers.len();
    let mut xres = Vec::new();
    for i in 0..nl {
        xres.extend_from_slice(&gpu.download_f32(&acts.layer_acts[i].xn1)?);
    }
    for i in 0..nl {
        xres.extend_from_slice(&gpu.download_f32(&acts.layer_acts[i].xn2)?);
    }
    let rows = nl * 2 * SEQ;
    let mut hess = vec![0.0f32; h * h];
    for r in 0..rows {
        let xr = &xres[r * h..r * h + h];
        for i in 0..h {
            let xi = xr[i];
            if xi == 0.0 {
                continue;
            }
            let hrow = &mut hess[i * h..i * h + h];
            for (o, &xj) in hrow.iter_mut().zip(xr.iter()) {
                *o += xi * xj;
            }
        }
    }
    let mut wstack = Vec::new();
    let mut rows_wt = 0usize;
    for (lw, _) in model.layers.iter() {
        for t in [&lw.wq, &lw.wo] {
            let m = gpu.download_f32(t)?;
            let mr = m.len() / h;
            let mut r = 0;
            while r < mr {
                wstack.extend_from_slice(&m[r * h..r * h + h]);
                rows_wt += 1;
                r += 24;
            }
        }
    }

    // Rotations to score end-to-end: plain FWHT (M=I), act-only (α=0), joint α, weight-only.
    let plain = Rotation::identity(h);
    let act_only = learn_rotation_kurtosis(&xres, rows, h, Rotation::hadamard(h, 1), 100, 0.05, 6);
    let joint = |a: f32| {
        learn_rotation_phase_joint(
            &xres,
            rows,
            &wstack,
            rows_wt,
            &hess,
            h,
            GROUP,
            4,
            Rotation::hadamard(h, 1),
            100,
            0.05,
            6,
            a,
        )
    };

    println!("\n  end-to-end decode KLD (W-int4 h-readers / A-f16, full forward vs fp):");
    println!("   rotation             | decode KLD");
    println!("  ----------------------+-----------");
    let cases: Vec<(String, Rotation)> = vec![
        ("plain FWHT (baseline)".into(), plain),
        ("learned act-only (α=0)".into(), act_only),
        ("phase-joint α=0.5".into(), joint(0.5)),
        ("phase-joint α=0.75".into(), joint(0.75)),
        ("weight-only (α=1)".into(), joint(1.0)),
    ];
    for (name, rot) in &cases {
        let l = decode_logits(&mut gpu, dir, rot, &tokens, &pos)?;
        let k = kld(&refl, &l, SEQ, vocab);
        println!("   {name:<20} | {k:.5}");
    }
    println!(
        "\n  (per-tensor SQNR was flat across α; this end-to-end KLD is the metric that\n   showed +22% for act-only. A joint α at/below plain FWHT = one-buffer decode win.)"
    );
    Ok(())
}
