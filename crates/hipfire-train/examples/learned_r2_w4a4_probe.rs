//! SpinQuant R2 (head-wise) follow-up: does a **learned** per-head value rotation
//! beat identity / a fixed per-head Hadamard in the *full* W4A4 recipe on the
//! **o_proj** path (both operands int4)?
//!
//! R1 (`learned_r1_w4a4_probe`) rotates the hidden/residual dim and covers the
//! q/k/v/gate/up **readers**. It never touches the value → o_proj GEMM: that
//! contraction runs over `q_dim = n_heads·head_dim`, and the quant-relevant axis
//! there is the **head_dim** subspace of each attention head. R2 is the pair that
//! conditions it — merged into `v_proj` (writer, `Wv → R2·Wv` per KV head) and
//! `o_proj` (reader, `Wo → Wo·R2ᵀ` per query head); attention is linear in V so
//! the fp output is invariant while the int4 grid sees a flatter per-head basis.
//!
//! This probe mirrors the R1 one on the head_dim axis. For a per-head rotation
//! `R2 [head_dim, head_dim]` we form the block-diagonal `[q_dim, q_dim]` rotation
//! (R2 on each head), rotate the o_proj input `ctx` and weight `Wo` in that basis,
//! quantize both to symmetric int4 per 256-group, run the real `iu4·iu4` GEMM (the
//! r1 kernel copy), rescale, and score SQNR vs the fp reference `ctx·Woᵀ`
//! (rotation-invariant). `R2 ∈ {I, per-head Hadamard, learned}`; the learned `R2`
//! minimizes the kurtosis surrogate on the captured **value** activations (the
//! SpinQuant target), on the actually-quantized **ctx**, and jointly on
//! ctx+o_proj-weight.
//!
//! The per-256-group FWHT (the Oq4 codec default) is shown as a reference line,
//! but note it mixes across 4 heads (256 = 4·64) — it is *not* a per-head,
//! `apply_r2`-mergeable rotation. R2's payoff is that it is head-local and merges
//! into the weights at zero runtime cost, and it is the rotation the future
//! 4-bit-KV / R3 path needs on the value cache.
//!
//! Run (needs a JIT-capable toolchain for the r1 kernel):
//!   source ./scripts/rocm-env.sh
//!   export ROCM_PATH=$HOME/.venv/lib/python3.14/site-packages/_rocm_sdk_core
//!   hipfire lock acquire "learned-r2-w4a4"
//!   cargo run -p hipfire-train --release --example learned_r2_w4a4_probe
//!   hipfire lock release

use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
use hipfire_train::learn_rotation::{learn_rotation_joint, learn_rotation_kurtosis};
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{model_forward, LlamaModel};
use hipfire_train::rotation::{apply_r1, rotate_rows, Rotation};
use hipfire_rdna::{Gpu, HipResult};
use std::path::Path;

const DEFAULT_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const SEQ: usize = 16;
const GROUP: usize = 256;

/// How to rotate a captured tensor before int4 (on its contraction axis `h`).
enum Rot<'a> {
    None,
    Fwht(&'a [f32], &'a [f32]), // per-256-group block Hadamard (codec default)
    Full(&'a Rotation),         // a dense [h,h] rotation (block-diagonal R2)
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
/// iu4 kernel. `a [SEQ,h]`, `w [out,h]`, contraction `h`. Returns SQNR dB.
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

/// Expand a per-head `R2 [hd,hd]` into the block-diagonal `[hd·n, hd·n]` rotation
/// that acts on `q_dim` (R2 on each head's head_dim sub-vector; heads unmixed).
fn block_diag(r2: &Rotation, n_blocks: usize) -> Rotation {
    let hd = r2.h;
    let qd = hd * n_blocks;
    let mut r = vec![0.0f32; qd * qd];
    for b in 0..n_blocks {
        let off = b * hd;
        for i in 0..hd {
            for j in 0..hd {
                r[(off + i) * qd + (off + j)] = r2.r[i * hd + j];
            }
        }
    }
    Rotation { h: qd, r }
}

/// Stack a `[outer, n_blocks·hd]` tensor into `[outer·n_blocks, hd]` rows — each
/// head's head_dim sub-vector becomes one row of the kurtosis learning set.
fn head_rows(src: &[f32], outer: usize, n_blocks: usize, hd: usize) -> Vec<f32> {
    let stride = n_blocks * hd;
    let mut out = Vec::with_capacity(outer * n_blocks * hd);
    for r in 0..outer {
        for b in 0..n_blocks {
            let base = r * stride + b * hd;
            out.extend_from_slice(&src[base..base + hd]);
        }
    }
    out
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
    let mut model = LlamaModel::from_f32_weights(&mut gpu, &cfg, w, SEQ, 2, 1.0)?;
    apply_r1(&mut gpu, &mut model, &Rotation::identity(h))?; // fold only (untie); R1 = I
    let (n_heads, n_kv, hd) = (model.dims.n_heads, model.dims.n_kv, model.dims.head_dim);
    let qd = model.dims.q_dim();
    if qd % GROUP != 0 || !qd.is_power_of_two() {
        return Err(format!("q_dim {qd} must be power-of-two & %256 for this probe").into());
    }
    if !hd.is_power_of_two() {
        return Err(
            format!("head_dim {hd} must be power-of-two for the Hadamard warm start").into(),
        );
    }
    println!("  h={h}  n_heads={n_heads}  n_kv={n_kv}  head_dim={hd}  q_dim={qd}");

    let tokens: Vec<u32> = (0..SEQ)
        .map(|t| (13 + t * 97) as u32 % cfg.vocab_size as u32)
        .collect();
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();
    let acts = model_forward(&mut gpu, &model, &tokens, &pos)?;

    // Per-layer capture: value activation v [seq,kvd], o_proj input ctx [seq,qd],
    // o_proj weight Wo [h,qd].
    let mut vv = Vec::new();
    let mut ctxs = Vec::new();
    let mut wo = Vec::new();
    for (i, (lw, _)) in model.layers.iter().enumerate() {
        vv.push(gpu.download_f32(&acts.layer_acts[i].v)?);
        ctxs.push(gpu.download_f32(&acts.layer_acts[i].ctx)?);
        wo.push(gpu.download_f32(&lw.wo)?);
    }
    let nl = model.layers.len();

    // Learning sets, all as [rows, head_dim]:
    //  X_v   — value activations per KV head (the SpinQuant R2 target).
    //  X_ctx — the o_proj input per query head (the tensor W4A4 actually quantizes).
    //  W_o   — o_proj weight columns per query head (the co-quantized reader).
    let mut xv = Vec::new();
    let mut xctx = Vec::new();
    for m in &vv {
        xv.extend_from_slice(&head_rows(m, SEQ, n_kv, hd));
    }
    for m in &ctxs {
        xctx.extend_from_slice(&head_rows(m, SEQ, n_heads, hd));
    }
    let rows_v = nl * SEQ * n_kv;
    let rows_ctx = nl * SEQ * n_heads;
    // o_proj weight rows [h·n_heads, hd] per layer, subsampled to ~4096 total.
    let total_wt_rows = nl * h * n_heads;
    let stride = (total_wt_rows / 4096).max(1);
    let mut wstack = Vec::new();
    let mut rows_wt = 0usize;
    for m in &wo {
        let cols = head_rows(m, h, n_heads, hd); // [h·n_heads, hd]
        let mrows = h * n_heads;
        let mut r = 0;
        while r < mrows {
            wstack.extend_from_slice(&cols[r * hd..r * hd + hd]);
            rows_wt += 1;
            r += stride;
        }
    }

    println!("\n  learning R2 (value-only, ctx-only, then joint ctx+weight) …");
    let iters = 200;
    let lr = 0.05;
    let learned_v =
        learn_rotation_kurtosis(&xv, rows_v, hd, Rotation::hadamard(hd, 1), iters, lr, 6);
    let learned_ctx =
        learn_rotation_kurtosis(&xctx, rows_ctx, hd, Rotation::hadamard(hd, 1), iters, lr, 6);
    let learned_joint = learn_rotation_joint(
        &xctx,
        rows_ctx,
        &wstack,
        rows_wt,
        hd,
        Rotation::hadamard(hd, 1),
        iters,
        lr,
        6,
        0.5,
    );
    println!(
        "  (orthonormality  value {:.1e}  ctx {:.1e}  joint {:.1e})",
        learned_v.orthonormality_error(),
        learned_ctx.orthonormality_error(),
        learned_joint.orthonormality_error()
    );

    let s1 = gen_fwht_signs(42, GROUP);
    let s2 = gen_fwht_signs(1042, GROUP);
    let had = block_diag(&Rotation::hadamard(hd, 1), n_heads);
    let bd_v = block_diag(&learned_v, n_heads);
    let bd_ctx = block_diag(&learned_ctx, n_heads);
    let bd_joint = block_diag(&learned_joint, n_heads);

    // Mean full-W4A4 o_proj SQNR over layers, per rotation (a=ctx, w=Wo, contract qd).
    let mean_snr = |gpu: &mut Gpu, mode: &Rot| -> HipResult<f32> {
        let mut s = 0.0f32;
        for i in 0..nl {
            s += w4a4(gpu, &ctxs[i], &wo[i], h, qd, mode)?;
        }
        Ok(s / nl as f32)
    };

    println!(
        "\n  full-W4A4 o_proj SQNR (ctx·Woᵀ, both operands int4, per-256-group, mean over layers):"
    );
    let naive = mean_snr(&mut gpu, &Rot::None)?;
    let fwht = mean_snr(&mut gpu, &Rot::Fwht(&s1, &s2))?;
    let hada = mean_snr(&mut gpu, &Rot::Full(&had))?;
    let lv = mean_snr(&mut gpu, &Rot::Full(&bd_v))?;
    let lc = mean_snr(&mut gpu, &Rot::Full(&bd_ctx))?;
    let lj = mean_snr(&mut gpu, &Rot::Full(&bd_joint))?;
    println!("  naive (no rotation)              {naive:8.2} dB");
    println!("  per-head Hadamard R2 (fixed)     {hada:8.2} dB   <- baseline");
    println!("  learned R2 (value activations)   {lv:8.2} dB");
    println!("  learned R2 (ctx activations)     {lc:8.2} dB");
    println!("  learned R2 (joint ctx+weight)    {lj:8.2} dB");
    println!("  per-256-group FWHT (codec, crosses heads, not R2-mergeable)  {fwht:8.2} dB");
    println!(
        "\n  vs fixed per-head Hadamard:  value {:+.2}  ctx {:+.2}  joint {:+.2} dB",
        lv - hada,
        lc - hada,
        lj - hada
    );
    let best = lv.max(lc).max(lj);
    if best > hada + 0.5 {
        println!("RESULT: the LEARNED per-head R2 beats the fixed per-head Hadamard on the o_proj W4A4 path.");
    } else if best >= hada - 0.5 {
        println!("RESULT: learned R2 ≈ fixed per-head Hadamard (head_dim small; the Hadamard is already near-optimal per head).");
    } else {
        println!("RESULT: learned R2 below the fixed Hadamard — the kurtosis surrogate underperforms here.");
    }
    Ok(())
}
