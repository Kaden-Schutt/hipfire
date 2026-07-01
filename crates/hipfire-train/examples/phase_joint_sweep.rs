//! Joint-M α-sweep: can ONE learned rotation (one stored int4 weight buffer) serve
//! both the compute-bound prefill (W4A4 — both operands int4) and the
//! bandwidth-bound decode (W4A16 — int4 weight, f16 activation)? For each blend
//! `α ∈ [0,1]` we learn `M(α)` via [`learn_rotation_phase_joint`] (α=0 = activation
//! kurtosis only = today's `--rotate`; α=1 = Hessian-weighted weight-quant only =
//! decode-optimal) and score BOTH phases on real Supra q_proj:
//!   • prefill  = full W4A4 SQNR via the real iu4·iu4 GEMM (higher = better).
//!   • decode   = W4A16 SQNR: int4 weight `Q4(W Mᵀ)` × exact f16 activation `X Mᵀ`.
//! The deployed per-group FWHT is the baseline for both. A joint α that keeps decode
//! ≥ FWHT while lifting prefill proves the single-buffer path; otherwise it's two
//! buffers.
//!
//!   source ./scripts/rocm-env.sh
//!   export ROCM_PATH=$HOME/.venv/lib/python3.14/site-packages/_rocm_sdk_core
//!   hipfire lock acquire "phase-joint-sweep"
//!   cargo run -p hipfire-train --release --example phase_joint_sweep
//!   hipfire lock release

use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
use hipfire_train::learn_rotation::learn_rotation_phase_joint;
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{model_forward, LlamaModel};
use hipfire_train::rotation::{apply_r1, rotate_rows, Rotation};
use rdna_compute::{Gpu, HipResult};
use std::path::Path;

const DEFAULT_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const SEQ: usize = 16;
const GROUP: usize = 256;

enum Rot<'a> {
    Fwht(&'a [f32], &'a [f32]),
    Full(&'a Rotation),
}

fn rotate(src: &[f32], rows: usize, h: usize, mode: &Rot) -> Vec<f32> {
    match mode {
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

/// Symmetric int4 [-7,7] per 256-group. clip=true ⇒ clip-search (weight).
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

fn yref(a: &[f32], w: &[f32], out: usize, h: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; SEQ * out];
    for b in 0..SEQ {
        for o in 0..out {
            let mut acc = 0.0f32;
            for k in 0..h {
                acc += a[b * h + k] * w[o * h + k];
            }
            y[b * out + o] = acc;
        }
    }
    y
}

/// Prefill: full W4A4 (both operands int4) via the r1 iu4 kernel. SQNR dB.
fn w4a4(gpu: &mut Gpu, a: &[f32], w: &[f32], out: usize, h: usize, mode: &Rot) -> HipResult<f32> {
    let yr = yref(a, w, out, h);
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
    Ok(sqnr(&ygpu, &yr))
}

/// Decode: W4A16 — int4 weight `Q4(W Mᵀ)` × exact (f16≈f32) activation `X Mᵀ`.
/// Pure CPU (no activation quant). SQNR dB vs the fp reference.
fn w4a16(a: &[f32], w: &[f32], out: usize, h: usize, mode: &Rot) -> f32 {
    let yr = yref(a, w, out, h);
    let af = rotate(a, SEQ, h, mode); // exact rotated activation (f16)
    let wf = rotate(w, out, h, mode);
    let (qw, sw) = quant_int4(&wf, out, h, true);
    let ng = h / GROUP;
    let mut y = vec![0.0f32; SEQ * out];
    for b in 0..SEQ {
        for o in 0..out {
            let mut acc = 0.0f32;
            for g in 0..ng {
                let s = sw[o * ng + g];
                let g0 = g * GROUP;
                for c in 0..GROUP {
                    let wdq = qw[o * h + g0 + c] as f32 * s;
                    acc += wdq * af[b * h + g0 + c];
                }
            }
            y[b * out + o] = acc;
        }
    }
    sqnr(&y, &yr)
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
    for (i, (lw, _)) in model.layers.iter().enumerate() {
        xn1.push(gpu.download_f32(&acts.layer_acts[i].xn1)?);
        wq.push(gpu.download_f32(&lw.wq)?);
    }
    let nl = model.layers.len();

    // Learning activation set: stacked q_proj reader activations (xn1).
    let rows = nl * SEQ;
    let mut xstack = Vec::with_capacity(rows * h);
    for m in xn1.iter() {
        xstack.extend_from_slice(m);
    }
    // Hessian H = XᵀX [h,h] over the reader activations (the calib collector's H).
    let mut hess = vec![0.0f32; h * h];
    for r in 0..rows {
        let xr = &xstack[r * h..r * h + h];
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
    // Weight set for the weight-quant term: q_proj readers, row-subsampled cheap.
    let stride = ((nl * qd) / 384).max(1);
    let mut wstack = Vec::new();
    let mut rows_wt = 0usize;
    for m in wq.iter() {
        let mut r = 0;
        while r < qd {
            wstack.extend_from_slice(&m[r * h..r * h + h]);
            rows_wt += 1;
            r += stride;
        }
    }
    println!("  hidden={h} q_dim={qd} layers={nl}  learn rows_act={rows} rows_wt={rows_wt}");

    let s1 = gen_fwht_signs(42, GROUP);
    let s2 = gen_fwht_signs(1042, GROUP);
    let fwht = Rot::Fwht(&s1, &s2);

    // Deployed per-group FWHT baseline for BOTH phases.
    let mut pre_fwht = 0.0f32;
    let mut dec_fwht = 0.0f32;
    for i in 0..nl {
        pre_fwht += w4a4(&mut gpu, &xn1[i], &wq[i], qd, h, &fwht)?;
        dec_fwht += w4a16(&xn1[i], &wq[i], qd, h, &fwht);
    }
    pre_fwht /= nl as f32;
    dec_fwht /= nl as f32;
    println!(
        "\n  baseline per-group FWHT:   prefill W4A4 {pre_fwht:6.2} dB   decode W4A16 {dec_fwht:6.2} dB"
    );

    println!("\n   alpha | prefill W4A4 | decode W4A16 | note");
    println!("  -------+--------------+--------------+-----------------------------");
    for &alpha in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
        let m = learn_rotation_phase_joint(
            &xstack,
            rows,
            &wstack,
            rows_wt,
            &hess,
            h,
            GROUP,
            4,
            Rotation::hadamard(h, 1),
            80,
            0.05,
            6,
            alpha,
        );
        let mode = Rot::Full(&m);
        let mut pre = 0.0f32;
        let mut dec = 0.0f32;
        for i in 0..nl {
            pre += w4a4(&mut gpu, &xn1[i], &wq[i], qd, h, &mode)?;
            dec += w4a16(&xn1[i], &wq[i], qd, h, &mode);
        }
        pre /= nl as f32;
        dec /= nl as f32;
        let note = if alpha == 0.0 {
            "act-only (= today's --rotate)"
        } else if alpha == 1.0 {
            "weight-only (decode-optimal)"
        } else if dec >= dec_fwht && pre > pre_fwht {
            "<- single-buffer WIN"
        } else {
            ""
        };
        println!("   {alpha:4.2}  |   {pre:7.2}    |   {dec:7.2}    | {note}");
    }
    println!(
        "\n  target: an alpha with decode >= {dec_fwht:.2} dB (FWHT) AND prefill > {pre_fwht:.2} dB\n  => one int4 buffer serves both phases; else two buffers."
    );
    Ok(())
}
