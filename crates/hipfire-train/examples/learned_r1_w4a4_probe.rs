//! SpinQuant R1 Phase 2 follow-up: does the **learned** rotation beat the
//! deployed per-group FWHT in the *full* W4A4 recipe (both operands int4)?
//!
//! Phase 1b showed a *fixed* global Hadamard R1 adds ~nothing over the Oq4
//! recipe's per-256-group FWHT (both are fixed, data-agnostic). The recipe's
//! FWHT is itself a rotation `F`, so with a *free* learned R1 the pipeline
//! "R1 then FWHT" collapses to one rotation — meaning we can compare rotations
//! head-to-head on the identical int4 pipeline. For each rotation `M` we quantize
//! the deployed activation `X Mᵀ` and weight `W Mᵀ` to symmetric int4 per
//! 256-group (weight clip-search / act absmax — the Oq4G256 grids), run the real
//! `iu4·iu4` GEMM (the r1 kernel copy), rescale, and score SQNR vs the fp
//! reference `X Wᵀ` (rotation-invariant). `M ∈ {I, per-group FWHT, global
//! Hadamard, learned}`; the learned `M` minimizes the kurtosis surrogate on the
//! captured residual activations.
//!
//! Run (needs a JIT-capable toolchain for the r1 kernel):
//!   source ./scripts/rocm-env.sh
//!   export ROCM_PATH=$HOME/.venv/lib/python3.14/site-packages/_rocm_sdk_core
//!   hipfire lock acquire "learned-r1-w4a4"
//!   cargo run -p hipfire-train --release --example learned_r1_w4a4_probe
//!   hipfire lock release

use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
use hipfire_train::learn_rotation::{learn_rotation_joint, learn_rotation_kurtosis};
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{model_forward, LlamaModel};
use hipfire_train::rotation::{apply_r1, rotate_rows, Rotation};
use rdna_compute::{Gpu, HipResult};
use std::path::Path;

const DEFAULT_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const SEQ: usize = 16;
const GROUP: usize = 256;

/// How to rotate a captured tensor before int4.
enum Rot<'a> {
    None,
    Fwht(&'a [f32], &'a [f32]), // per-256-group block Hadamard (F)
    Full(&'a Rotation),         // a dense [h,h] rotation (global Hadamard / learned)
}

fn rotate(src: &[f32], rows: usize, h: usize, mode: &Rot) -> Vec<f32> {
    match mode {
        Rot::None => src.to_vec(),
        Rot::Full(r) => rotate_rows(src, r, rows),
        Rot::Fwht(s1, s2) => {
            let mut m = src.to_vec();
            let mut buf = [0.0f32; GROUP];
            for r in 0..rows {
                for seg in 0..(h / GROUP) {
                    let base = r * h + seg * GROUP;
                    buf.copy_from_slice(&m[base..base + GROUP]);
                    cpu_fwht_256(&mut buf, s1, s2);
                    m[base..base + GROUP].copy_from_slice(&buf);
                }
            }
            m
        }
    }
}

/// Symmetric int4 [-7,7] per 256-group. clip=true ⇒ clip-search (weight); else
/// absmax/7 (activation). Returns (q [rows,h] i8, scales [rows, h/256]).
fn quant_int4(src: &[f32], rows: usize, h: usize, clip: bool) -> (Vec<i8>, Vec<f32>) {
    const GRID: [f32; 9] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6];
    let ng = h / GROUP;
    let mut q = vec![0i8; rows * h];
    let mut sc = vec![0f32; rows * ng];
    for r in 0..rows {
        for g in 0..ng {
            let g0 = g * GROUP;
            let grp = &src[r * h + g0..r * h + g0 + GROUP];
            let amax = grp.iter().fold(1e-12f32, |a, &v| a.max(v.abs()));
            let scale = if clip {
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
                bs
            } else {
                (amax / 7.0).max(1e-12)
            };
            sc[r * ng + g] = scale;
            for (c, &v) in grp.iter().enumerate() {
                q[r * h + g0 + c] = (v / scale).round().clamp(-7.0, 7.0) as i8;
            }
        }
    }
    (q, sc)
}

fn pack_group(q: &[i8], rows: usize, h: usize, g: usize) -> Vec<u8> {
    let g0 = g * GROUP;
    let mut out = vec![0u8; rows * (GROUP / 2)];
    for r in 0..rows {
        for j in (0..GROUP).step_by(2) {
            let lo = (q[r * h + g0 + j] as u8) & 0xf;
            let hi = (q[r * h + g0 + j + 1] as u8) & 0xf;
            out[r * (GROUP / 2) + j / 2] = lo | (hi << 4);
        }
    }
    out
}

fn sqnr(rec: &[f32], yref: &[f32]) -> f32 {
    let (mut sig, mut noise) = (0.0f64, 0.0f64);
    for (&r, &o) in rec.iter().zip(yref) {
        sig += (o as f64) * (o as f64);
        let d = o as f64 - r as f64;
        noise += d * d;
    }
    (10.0 * (sig / noise.max(1e-30)).log10()) as f32
}

/// Full W4A4 (both operands int4) of `y=a·Wᵀ` under rotation `mode`, via the r1
/// iu4 kernel. Returns SQNR dB vs the fp reference.
fn w4a4(gpu: &mut Gpu, a: &[f32], w: &[f32], out: usize, h: usize, mode: &Rot) -> HipResult<f32> {
    let mut yref = vec![0.0f32; SEQ * out];
    for b in 0..SEQ {
        for o in 0..out {
            let mut acc = 0.0f32;
            for k in 0..h {
                acc += a[b * h + k] * w[o * h + k];
            }
            yref[b * out + o] = acc;
        }
    }
    let af = rotate(a, SEQ, h, mode);
    let wf = rotate(w, out, h, mode);
    let (qw, sw) = quant_int4(&wf, out, h, true);
    let (qx, sx) = quant_int4(&af, SEQ, h, false);
    let ng = h / GROUP;
    let mut ygpu = vec![0.0f32; SEQ * out];
    for g in 0..ng {
        let wd = gpu.upload_raw(&pack_group(&qw, out, h, g), &[out, GROUP / 2])?;
        let xd = gpu.upload_raw(&pack_group(&qx, SEQ, h, g), &[SEQ, GROUP / 2])?;
        let yd = gpu.upload_raw(&vec![0u8; SEQ * out * 4], &[SEQ, out])?;
        gpu.gemm_iu4_i32_wmma_r1(&wd, &xd, &yd, out, GROUP, SEQ)?;
        gpu.device_synchronize()?;
        let yb = gpu.download_raw(&yd, SEQ * out * 4)?;
        for b in 0..SEQ {
            let sxg = sx[b * ng + g];
            for o in 0..out {
                let isum = i32::from_le_bytes([
                    yb[(b * out + o) * 4],
                    yb[(b * out + o) * 4 + 1],
                    yb[(b * out + o) * 4 + 2],
                    yb[(b * out + o) * 4 + 3],
                ]);
                ygpu[b * out + o] += isum as f32 * sw[o * ng + g] * sxg;
            }
        }
        gpu.free_tensor(wd)?;
        gpu.free_tensor(xd)?;
        gpu.free_tensor(yd)?;
    }
    Ok(sqnr(&ygpu, &yref))
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
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP: {} lacks wave32 WMMA", gpu.arch);
        return Ok(());
    }
    println!("arch: {}  model: {}", gpu.arch, dir.display());

    let (cfg, w) = load_llama_fp32(&mut gpu, dir).map_err(|e| format!("load: {e}"))?;
    let h = cfg.hidden_size;
    if h % GROUP != 0 || !h.is_power_of_two() {
        return Err(format!("hidden {h} must be power-of-two & %256").into());
    }
    let mut model = LlamaModel::from_f32_weights(&mut gpu, &cfg, w, SEQ, 2, 1.0)?;
    apply_r1(&mut gpu, &mut model, &Rotation::identity(h))?; // fold only
    let qd = model.dims.q_dim();
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
    let nl = model.layers.len();
    let inter = model.dims.inter;

    // Activation set for learning: stacked residual-read activations (xn1+xn2).
    let rows = nl * 2 * SEQ;
    let mut xstack = Vec::with_capacity(rows * h);
    for m in xn1.iter().chain(xn2.iter()) {
        xstack.extend_from_slice(m);
    }
    // Weight set: reader weights (wq + wgate), the tensors the deployed W4A4
    // quantizes in the rotated basis alongside the activations. Row-subsampled
    // (stride) to ~a few thousand rows — the kurtosis statistics converge fine on
    // a representative sample, and it keeps the offline learn cheap.
    let all_wt: Vec<&Vec<f32>> = wq.iter().chain(wgate.iter()).collect();
    let total_wt_rows = nl * (qd + inter);
    let stride = (total_wt_rows / 4096).max(1);
    let mut wstack = Vec::new();
    let mut rows_wt = 0usize;
    for m in &all_wt {
        let mrows = m.len() / h;
        let mut r = 0;
        while r < mrows {
            wstack.extend_from_slice(&m[r * h..r * h + h]);
            rows_wt += 1;
            r += stride;
        }
    }

    println!("\n  learning rotations (act-only, then joint act+weight) …");
    let learned = learn_rotation_kurtosis(&xstack, rows, h, Rotation::hadamard(h, 1), 120, 0.05, 6);
    let learned_joint = learn_rotation_joint(
        &xstack,
        rows,
        &wstack,
        rows_wt,
        h,
        Rotation::hadamard(h, 1),
        120,
        0.05,
        6,
        0.5,
    );
    println!(
        "  (orthonormality  act-only {:.1e}  joint {:.1e})",
        learned.orthonormality_error(),
        learned_joint.orthonormality_error()
    );

    let s1 = gen_fwht_signs(42, GROUP);
    let s2 = gen_fwht_signs(1042, GROUP);
    let had = Rotation::hadamard(h, 1);

    // Mean full-W4A4 q_proj SQNR over layers, per rotation.
    let mean_snr = |gpu: &mut Gpu, mode: &Rot| -> HipResult<f32> {
        let mut s = 0.0f32;
        for i in 0..nl {
            s += w4a4(gpu, &xn1[i], &wq[i], qd, h, mode)?;
        }
        Ok(s / nl as f32)
    };

    println!("\n  full-W4A4 q_proj SQNR (both operands int4, per-256-group, mean over layers):");
    let naive = mean_snr(&mut gpu, &Rot::None)?;
    let fwht = mean_snr(&mut gpu, &Rot::Fwht(&s1, &s2))?;
    let hada = mean_snr(&mut gpu, &Rot::Full(&had))?;
    let lrn = mean_snr(&mut gpu, &Rot::Full(&learned))?;
    let lrn_j = mean_snr(&mut gpu, &Rot::Full(&learned_joint))?;
    println!("  naive (no rotation)          {naive:8.2} dB");
    println!("  per-group FWHT (deployed)    {fwht:8.2} dB   <- baseline");
    println!("  global Hadamard              {hada:8.2} dB");
    println!("  learned (act only)           {lrn:8.2} dB");
    println!("  learned (joint act+weight)   {lrn_j:8.2} dB");
    println!(
        "\n  vs per-group-FWHT baseline:  act-only {:+.2} dB   joint {:+.2} dB",
        lrn - fwht,
        lrn_j - fwht
    );
    let best = lrn.max(lrn_j);
    if best > fwht + 0.5 {
        println!("PHASE 2: the LEARNED rotation beats the deployed per-group FWHT in full W4A4.");
    } else if best >= fwht - 0.5 {
        println!("RESULT: learned ≈ per-group FWHT here (small model; the FWHT is already near-optimal).");
    } else {
        println!("RESULT: learned below FWHT — the kurtosis surrogate underperforms the block Hadamard here.");
    }
    Ok(())
}
