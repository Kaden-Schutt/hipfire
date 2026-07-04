//! SpinQuant R1 Phase 1b: run the **real W4A4 int4 matrix-core path** end-to-end
//! on a real model and show R1 improves output quality. For each rotation
//! (identity=fold-only, Hadamard) we bake `R1` into the weights, run the fp32
//! forward, then per layer take the deployed q_proj activation `xn1 [seq,h]` and
//! weight `wq [qd,h]` and push them through the deployed Oq4G256 W4A4 recipe:
//!
//!   per 256-group over h:  FWHT-256 (signs 42/1042) → symmetric int4
//!     (weight: clip-search scale, à la quantize_oq4g256; act: absmax/7, à la
//!      quantize_act_oq4) → the fused iu4 WMMA GEMM (the R1 kernel copy) → f32
//!      rescale by scale_w·scale_x, accumulate across groups.
//!
//! Compares the reconstructed W4A4 output SQNR (vs the fp reference `xn1·wqᵀ`)
//! for identity vs Hadamard — the concrete W4A4 payoff of the rotation, on the
//! actual matrix-core kernel. A CPU sim of the identical integer scheme is
//! reported alongside (GPU int math is exact; only the f32 rescale order differs).
//!
//! Run (defaults to Supra-50M; point ROCM_PATH at a toolchain that can JIT if the
//! r1 kernel is not precompiled):
//!   source ./scripts/rocm-env.sh
//!   export ROCM_PATH=$HOME/.venv/lib/python3.14/site-packages/_rocm_sdk_core
//!   hipfire lock acquire "w4a4-r1-probe"
//!   cargo run -p hipfire-train --release --example w4a4_r1_probe
//!   hipfire lock release

use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{model_forward, LlamaModel};
use hipfire_train::rotation::{apply_r1, Rotation};
use hipfire_rdna::{Gpu, HipResult};
use std::path::Path;

const DEFAULT_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const SEQ: usize = 16;
const GROUP: usize = 256;

/// FWHT-256 rotate each 256-group of the `h` dim of a `[rows,h]` matrix in place
/// (the Oq4G256 codec's per-group Hadamard; `h % 256 == 0`).
fn fwht_rows(m: &mut [f32], rows: usize, h: usize, s1: &[f32], s2: &[f32]) {
    let mut buf = [0.0f32; GROUP];
    for r in 0..rows {
        for seg in 0..(h / GROUP) {
            let base = r * h + seg * GROUP;
            buf.copy_from_slice(&m[base..base + GROUP]);
            cpu_fwht_256(&mut buf, s1, s2);
            m[base..base + GROUP].copy_from_slice(&buf);
        }
    }
}

/// Symmetric int4 [-7,7] per 256-group. `clip=true` ⇒ clip-search scale (weight,
/// matches quantize_oq4g256); `clip=false` ⇒ absmax/7 (activation, matches
/// quantize_act_oq4). Returns (q [rows,h] as i8, scales [rows, h/256]).
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

/// Pack group `g` (256 cols) of a [rows,h] int4 matrix → [rows,128] bytes
/// (`k_even | k_odd<<4`, signed two's-complement).
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
    let mut sig = 0.0f64;
    let mut noise = 0.0f64;
    for (&r, &o) in rec.iter().zip(yref) {
        sig += (o as f64) * (o as f64);
        let d = o as f64 - r as f64;
        noise += d * d;
    }
    (10.0 * (sig / noise.max(1e-30)).log10()) as f32
}

/// Full W4A4 of one projection `y = a·Wᵀ`, `a [seq,h]`, `w [out,h]`, through the
/// r1 iu4 kernel. Returns (gpu_sqnr, cpu_sqnr) dB vs the fp reference.
fn w4a4_proj(
    gpu: &mut Gpu,
    a: &[f32],
    w: &[f32],
    out: usize,
    h: usize,
    s1: &[f32],
    s2: &[f32],
    fwht: bool,
) -> HipResult<(f32, f32)> {
    // fp reference.
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
    // Front-end: optional FWHT-256 (the Oq4 recipe's per-group Hadamard), then
    // symmetric int4 per group. `fwht=false` is the naive-W4A4 floor.
    let mut af = a.to_vec();
    let mut wf = w.to_vec();
    if fwht {
        fwht_rows(&mut af, SEQ, h, s1, s2);
        fwht_rows(&mut wf, out, h, s1, s2);
    }
    let (qw, sw) = quant_int4(&wf, out, h, true); // weight: clip-search
    let (qx, sx) = quant_int4(&af, SEQ, h, false); // act: absmax
    let ng = h / GROUP;

    // CPU sim (exact integer, f32 rescale).
    let mut ycpu = vec![0.0f32; SEQ * out];
    for b in 0..SEQ {
        for o in 0..out {
            let mut acc = 0.0f32;
            for g in 0..ng {
                let g0 = g * GROUP;
                let mut isum = 0i32;
                for c in g0..g0 + GROUP {
                    isum += qw[o * h + c] as i32 * qx[b * h + c] as i32;
                }
                acc += isum as f32 * sw[o * ng + g] * sx[b * ng + g];
            }
            ycpu[b * out + o] = acc;
        }
    }

    // GPU: per-group fused iu4 GEMM (r1 copy), rescale + accumulate in f32.
    let mut ygpu = vec![0.0f32; SEQ * out];
    for g in 0..ng {
        let wpk = pack_group(&qw, out, h, g);
        let xpk = pack_group(&qx, SEQ, h, g);
        let wd = gpu.upload_raw(&wpk, &[out, GROUP / 2])?;
        let xd = gpu.upload_raw(&xpk, &[SEQ, GROUP / 2])?;
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
    Ok((sqnr(&ygpu, &yref), sqnr(&ycpu, &yref)))
}

fn run_rotation(
    gpu: &mut Gpu,
    dir: &Path,
    rot: &Rotation,
    tokens: &[u32],
    s1: &[f32],
    s2: &[f32],
    fwht: bool,
) -> HipResult<(f32, f32)> {
    let (cfg, w) = load_llama_fp32(gpu, dir).expect("load model");
    let mut model = LlamaModel::from_f32_weights(gpu, &cfg, w, SEQ, 2, 1.0)?;
    apply_r1(gpu, &mut model, rot)?;
    let (h, qd) = (model.dims.h, model.dims.q_dim());
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();
    let acts = model_forward(gpu, &model, tokens, &pos)?;

    let (mut gpu_sum, mut cpu_sum) = (0.0f32, 0.0f32);
    let nl = model.layers.len();
    for i in 0..nl {
        let xn1 = gpu.download_f32(&acts.layer_acts[i].xn1)?;
        let wq = gpu.download_f32(&model.layers[i].0.wq)?;
        let (gp, cp) = w4a4_proj(gpu, &xn1, &wq, qd, h, s1, s2, fwht)?;
        gpu_sum += gp;
        cpu_sum += cp;
    }
    Ok((gpu_sum / nl as f32, cpu_sum / nl as f32))
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

    let cfg =
        hipfire_train::config::LlamaConfig::from_dir(dir).map_err(|e| format!("config: {e}"))?;
    if cfg.hidden_size % GROUP != 0 || !cfg.hidden_size.is_power_of_two() {
        return Err(format!("hidden {} must be power-of-two & %256", cfg.hidden_size).into());
    }
    let s1 = gen_fwht_signs(42, GROUP);
    let s2 = gen_fwht_signs(1042, GROUP);
    let tokens: Vec<u32> = (0..SEQ)
        .map(|t| (13 + t * 97) as u32 % cfg.vocab_size as u32)
        .collect();

    println!("\n  W4A4 q_proj SQNR (mean over layers) via the iu4·iu4 kernel copy:");
    println!("  configuration                         GPU        CPU-sim");
    // (name, rotation, apply per-group FWHT). The naive floor has neither the
    // recipe's per-group Hadamard nor R1; the recipe tier adds the FWHT; the R1
    // tier adds a global fixed Hadamard on top of the FWHT.
    let ident = Rotation::identity(cfg.hidden_size);
    let had = Rotation::hadamard(cfg.hidden_size, 1);
    let cases: [(&str, &Rotation, bool); 3] = [
        ("naive W4A4 (no rotation)", &ident, false),
        ("Oq4 recipe (per-group FWHT)", &ident, true),
        ("FWHT + fixed Hadamard R1", &had, true),
    ];
    let mut recipe = 0.0f32;
    let mut r1 = 0.0f32;
    for (name, rot, fwht) in cases {
        let (g, c) = run_rotation(&mut gpu, dir, rot, &tokens, &s1, &s2, fwht)?;
        println!("  {name:<34} {g:8.2} dB {c:8.2} dB");
        if name == "Oq4 recipe (per-group FWHT)" {
            recipe = g;
        } else if name == "FWHT + fixed Hadamard R1" {
            r1 = g;
        }
    }
    println!(
        "\n  fixed-R1 gain over the existing per-group FWHT recipe: {:+.2} dB",
        r1 - recipe
    );
    println!(
        "  Finding: the Oq4 recipe's per-group FWHT already captures the fixed-rotation\n  \
         (QuaRot) tier, so a second *fixed* Hadamard R1 adds ~nothing. The +6.8 dB SpinQuant\n  \
         payoff needs a *learned* R1 (Phase 2) — this probe is the baseline it must beat."
    );
    Ok(())
}
