//! SpinQuant joint {R1, R2}: do the independently-learned residual rotation R1
//! and head-wise value rotation R2 **compose** across the attention block with no
//! interference?
//!
//! R1 (`learned_r1_w4a4_probe`) conditions the q/k/v/gate/up readers on the hidden
//! dim; R2 (`learned_r2_w4a4_probe`) conditions the o_proj / value path on the
//! head_dim subspace of q_dim. They live on **orthogonal axes and different GEMMs**
//! — no single int4 GEMM sees both — so the doc's plan is to learn each
//! independently and confirm the combination adds. This probe does exactly that,
//! in two parts:
//!
//! Part 1 — int4 W4A4 (both operands int4, per-256-group, via the r1 kernel copy),
//!   on fold-only captures with each candidate rotation applied in the measurement:
//!     q_proj (contract h):  naive / codec FWHT / learned R1
//!     o_proj (contract qd): naive / codec FWHT / per-head Hadamard / learned R2
//!   The "attention block under {R1,R2}" is the pair {q_proj@R1, o_proj@R2}.
//!
//! Part 2 — fp composition proof, *through the model*. Bake BOTH rotations into a
//!   fresh model (`apply_r1(R1)` then `apply_r2(R2)`), re-forward, and check the
//!   jointly-rotated activations against the analytic single-rotation predictions:
//!     • logits invariant vs fold-only         (whole stack fp-invariant under both)
//!     • xn1_joint == R1·xn1_fold              (R2 leaves the q_proj input untouched)
//!     • ctx_joint == blockdiag(R2)·ctx_fold   (R1 leaves the value/o_proj input untouched)
//!   These three equalities are the empirical statement that R1 and R2 commute and
//!   compose, so each's Part-1 gain survives in the joint model.
//!
//! Run (needs a JIT-capable toolchain for the r1 kernel):
//!   source ./scripts/rocm-env.sh
//!   export ROCM_PATH=$HOME/.venv/lib/python3.14/site-packages/_rocm_sdk_core
//!   hipfire lock acquire "learned-joint-r1r2"
//!   cargo run -p hipfire-train --release --example learned_joint_r1r2_probe
//!   hipfire lock release

use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
use hipfire_train::learn_rotation::learn_rotation_kurtosis;
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{model_forward, LlamaModel};
use hipfire_train::rotation::{apply_r1, apply_r2, rotate_rows, Rotation};
use rdna_compute::{Gpu, HipResult};
use std::path::Path;

const DEFAULT_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const SEQ: usize = 16;
const GROUP: usize = 256;

/// How to rotate a captured tensor before int4 (on its contraction axis `h`).
enum Rot<'a> {
    None,
    Fwht(&'a [f32], &'a [f32]), // per-256-group block Hadamard (codec default)
    Full(&'a Rotation),         // a dense [h,h] rotation (learned R1 / block-diag R2)
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

/// Expand a per-head `R2 [hd,hd]` into the block-diagonal `[hd·n, hd·n]` rotation.
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

/// Stack a `[outer, n_blocks·hd]` tensor into `[outer·n_blocks, hd]` rows.
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

/// max |a − b| over two equal-length slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()))
}

/// Load Supra-50M fp32 and build a fresh trainable model (fold applied by caller).
fn load(gpu: &mut Gpu, dir: &Path) -> Result<(LlamaModel, usize), Box<dyn std::error::Error>> {
    let (cfg, w) = load_llama_fp32(gpu, dir).map_err(|e| format!("load: {e}"))?;
    let vocab = cfg.vocab_size;
    let model = LlamaModel::from_f32_weights(gpu, &cfg, w, SEQ, 2, 1.0)?;
    Ok((model, vocab))
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

    // ── Fold-only model: the R1=I, R2=I basis. Learn + Part-1 captures come here.
    let (mut m_fold, vocab) = load(&mut gpu, dir)?;
    let h = m_fold.dims.h;
    let (n_heads, n_kv, hd) = (m_fold.dims.n_heads, m_fold.dims.n_kv, m_fold.dims.head_dim);
    let qd = m_fold.dims.q_dim();
    if h % GROUP != 0 || !h.is_power_of_two() || qd % GROUP != 0 || !qd.is_power_of_two() {
        return Err(format!("need h {h} and q_dim {qd} power-of-two & %256").into());
    }
    if !hd.is_power_of_two() {
        return Err(
            format!("head_dim {hd} must be power-of-two for the Hadamard warm start").into(),
        );
    }
    println!("  h={h}  n_heads={n_heads}  n_kv={n_kv}  head_dim={hd}  q_dim={qd}");
    apply_r1(&mut gpu, &mut m_fold, &Rotation::identity(h))?; // fold only (untie); R1=R2=I

    let tokens: Vec<u32> = (0..SEQ)
        .map(|t| (13 + t * 97) as u32 % vocab as u32)
        .collect();
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();
    let acts = model_forward(&mut gpu, &m_fold, &tokens, &pos)?;

    let nl = m_fold.layers.len();
    let mut xn1 = Vec::new();
    let mut xn2 = Vec::new();
    let mut ctxs = Vec::new();
    let mut wq = Vec::new();
    let mut wo = Vec::new();
    for (i, (lw, _)) in m_fold.layers.iter().enumerate() {
        xn1.push(gpu.download_f32(&acts.layer_acts[i].xn1)?);
        xn2.push(gpu.download_f32(&acts.layer_acts[i].xn2)?);
        ctxs.push(gpu.download_f32(&acts.layer_acts[i].ctx)?);
        wq.push(gpu.download_f32(&lw.wq)?);
        wo.push(gpu.download_f32(&lw.wo)?);
    }
    let logits_fold = gpu.download_f32(&acts.logits)?;

    // ── Learn R1 (residual xn1+xn2 stack, hidden dim) and R2 (ctx, head_dim). ───
    println!("\n  learning R1 (residual, hidden dim) and R2 (ctx, head dim) …");
    let mut xres = Vec::with_capacity(nl * 2 * SEQ * h);
    for m in xn1.iter().chain(xn2.iter()) {
        xres.extend_from_slice(m);
    }
    let r1 = learn_rotation_kurtosis(
        &xres,
        nl * 2 * SEQ,
        h,
        Rotation::hadamard(h, 1),
        120,
        0.05,
        6,
    );
    let mut xctx = Vec::new();
    for m in &ctxs {
        xctx.extend_from_slice(&head_rows(m, SEQ, n_heads, hd));
    }
    let r2 = learn_rotation_kurtosis(
        &xctx,
        nl * SEQ * n_heads,
        hd,
        Rotation::hadamard(hd, 1),
        200,
        0.05,
        6,
    );
    let r2_bd = block_diag(&r2, n_heads);
    println!(
        "  (orthonormality  R1 {:.1e}  R2 {:.1e})",
        r1.orthonormality_error(),
        r2.orthonormality_error()
    );

    // ── Part 1: int4 W4A4 of the two attention GEMMs, mean over layers. ─────────
    let s1 = gen_fwht_signs(42, GROUP);
    let s2 = gen_fwht_signs(1042, GROUP);
    let had_h = Rotation::hadamard(h, 1);
    let had_hd = block_diag(&Rotation::hadamard(hd, 1), n_heads);

    let mean_q = |gpu: &mut Gpu, mode: &Rot| -> HipResult<f32> {
        let mut s = 0.0f32;
        for i in 0..nl {
            s += w4a4(gpu, &xn1[i], &wq[i], qd, h, mode)?;
        }
        Ok(s / nl as f32)
    };
    let mean_o = |gpu: &mut Gpu, mode: &Rot| -> HipResult<f32> {
        let mut s = 0.0f32;
        for i in 0..nl {
            s += w4a4(gpu, &ctxs[i], &wo[i], h, qd, mode)?;
        }
        Ok(s / nl as f32)
    };

    println!("\n  Part 1 — full-W4A4 SQNR (both operands int4, per-256-group, mean over layers):");
    println!("  q_proj (R1 domain, contract h):");
    println!(
        "    naive              {:8.2} dB",
        mean_q(&mut gpu, &Rot::None)?
    );
    println!(
        "    codec FWHT         {:8.2} dB",
        mean_q(&mut gpu, &Rot::Fwht(&s1, &s2))?
    );
    println!(
        "    global Hadamard    {:8.2} dB",
        mean_q(&mut gpu, &Rot::Full(&had_h))?
    );
    let q_r1 = mean_q(&mut gpu, &Rot::Full(&r1))?;
    println!("    learned R1         {q_r1:8.2} dB   <- R1");
    println!("  o_proj (R2 domain, contract q_dim):");
    println!(
        "    naive              {:8.2} dB",
        mean_o(&mut gpu, &Rot::None)?
    );
    println!(
        "    codec FWHT         {:8.2} dB",
        mean_o(&mut gpu, &Rot::Fwht(&s1, &s2))?
    );
    println!(
        "    per-head Hadamard  {:8.2} dB",
        mean_o(&mut gpu, &Rot::Full(&had_hd))?
    );
    let o_r2 = mean_o(&mut gpu, &Rot::Full(&r2_bd))?;
    println!("    learned R2         {o_r2:8.2} dB   <- R2");

    // ── Part 2: compose R1 and R2 into a fresh model, prove non-interference. ───
    println!("\n  Part 2 — fp composition proof (apply_r1(R1) ∘ apply_r2(R2), through the model):");
    let (mut m_joint, _) = load(&mut gpu, dir)?;
    apply_r1(&mut gpu, &mut m_joint, &r1)?;
    apply_r2(&mut gpu, &mut m_joint, &r2)?;
    let acts_j = model_forward(&mut gpu, &m_joint, &tokens, &pos)?;
    let logits_j = gpu.download_f32(&acts_j.logits)?;

    // Analytic single-rotation predictions from the fold-only captures.
    let mut worst_xn1 = 0.0f32; // xn1_joint  vs  R1·xn1_fold        (R2 must not disturb)
    let mut worst_ctx = 0.0f32; // ctx_joint  vs  blockdiag(R2)·ctx_fold (R1 must not disturb)
    for i in 0..nl {
        let xn1_j = gpu.download_f32(&acts_j.layer_acts[i].xn1)?;
        let ctx_j = gpu.download_f32(&acts_j.layer_acts[i].ctx)?;
        let xn1_pred = rotate_rows(&xn1[i], &r1, SEQ);
        let ctx_pred = rotate_rows(&ctxs[i], &r2_bd, SEQ);
        worst_xn1 = worst_xn1.max(max_abs_diff(&xn1_j, &xn1_pred));
        worst_ctx = worst_ctx.max(max_abs_diff(&ctx_j, &ctx_pred));
    }
    let worst_logit = max_abs_diff(&logits_j, &logits_fold);

    println!("    logits joint vs fold-only            max|Δ| {worst_logit:.2e}  (whole-stack fp invariance)");
    println!("    xn1  joint vs R1·xn1_fold            max|Δ| {worst_xn1:.2e}  (R2 leaves q_proj input untouched)");
    println!("    ctx  joint vs blockdiag(R2)·ctx_fold max|Δ| {worst_ctx:.2e}  (R1 leaves o_proj input untouched)");

    let ok = worst_logit < 5e-3 && worst_xn1 < 5e-3 && worst_ctx < 5e-3;
    println!();
    if ok {
        println!(
            "RESULT: R1 and R2 COMPOSE — orthogonal axes, no interference. The joint attention block \
             keeps q_proj@R1 {q_r1:.2} dB and o_proj@R2 {o_r2:.2} dB simultaneously."
        );
    } else {
        println!(
            "RESULT: composition check FAILED (max|Δ| logit {worst_logit:.1e} xn1 {worst_xn1:.1e} \
             ctx {worst_ctx:.1e}) — R1/R2 are interfering; investigate the merge."
        );
    }
    Ok(())
}
