//! W×A precision matrix: side-by-side weight/activation quant quality for the Opus
//! grid, W ∈ {int3, int4} × A ∈ {int4, int8, f16} × calib ∈ {RTN, LDLQ}, plus a
//! learned-R1 (SpinQuant) overlay for the A4 column and an end-to-end decode-KLD
//! anchor for the A16 column. Uniform: only tensors whose contraction dim is %256 are
//! scored (the valid Oq{3,4}G256 targets); down_proj (inter=1408) is excluded.
//!
//! METRIC 1 — per-tensor END-TO-END OUTPUT SNR, energy-weighted over every eligible
//! tensor of a real Supra-50M forward. For weight W [out,feat] and captured input
//! x [rows,feat]: ŷ = A_q(R x)·(R W)_qᵀ vs y = xWᵀ, both operands quantized in the
//! rotated domain R. R = per-256-group FWHT (codec floor) or the learned dense R1
//! (residual readers only; o_proj stays FWHT since R1 is residual-only). Orthogonality
//! ⇒ (Rx)(RW)ᵀ = xWᵀ, so no inverse rotation is needed.
//!
//! METRIC 2 — end-to-end decode KLD (A16): fake-quant every eligible weight to the
//! original-domain codec round-trip (FWHT→int→[OBS]→inv-FWHT), run the FULL forward
//! (activations stay f16), KLD(softmax fp ‖ softmax quant). Anchors that the per-tensor
//! SNR deltas track true compounding quality.
//!
//!   source ./scripts/rocm-env.sh
//!   hipfire lock acquire "wa-matrix"; cargo run -p hipfire-train --release --example wa_matrix_sweep; hipfire lock release

use faer::prelude::Solve;
use faer::{Mat, Side};
use hipfire_model::tokenizer::Tokenizer;
use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
use hipfire_train::block::BlockAdjoints;
use hipfire_train::learn_rotation::learn_rotation_kurtosis;
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{model_forward, model_guided_adjoints, LlamaModel};
use hipfire_train::qtip_quant::{build_codebook, qtip_group_requant};
use hipfire_train::rotation::{rotate_rows, Rotation};
use rayon::prelude::*;
use rdna_compute::{Gpu, GpuTensor, HipResult};
use std::path::Path;

const DEFAULT_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
// calibration window = held-out eval window length. The training model keeps a full
// un-freed activation tape per forward (attention probs ~seq²); the run does ~8 forwards,
// so seq must stay modest on this UMA box — 1024 (4× the 256 baseline) fits ~45 GB.
const SEQ: usize = 256; // calib+eval window (3 held-out windows fit Supra's 1024 ctx)
const EVAL_ROWS: usize = 256; // held-out rows used for the SNR/KLD metrics (<= SEQ)
const GROUP: usize = 256;

fn signs() -> (Vec<f32>, Vec<f32>) {
    (gen_fwht_signs(42, GROUP), gen_fwht_signs(1042, GROUP))
}

/// Per-256-group FWHT of each row of `[rows, feat]`; `inv` swaps the sign order
/// (the orthogonal inverse). feat % 256 == 0.
fn fwht_rows(x: &[f32], rows: usize, feat: usize, inv: bool) -> Vec<f32> {
    let (s1, s2) = signs();
    let (a, b) = if inv { (&s2, &s1) } else { (&s1, &s2) };
    let mut out = x.to_vec();
    for r in 0..rows {
        for g in (0..feat).step_by(GROUP) {
            let mut buf = [0.0f32; GROUP];
            buf.copy_from_slice(&out[r * feat + g..r * feat + g + GROUP]);
            cpu_fwht_256(&mut buf, a, b);
            out[r * feat + g..r * feat + g + GROUP].copy_from_slice(&buf);
        }
    }
    out
}

fn symmetric_clipsearch(group: &[f32], qmax: f32) -> f32 {
    const GRID: [f32; 9] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6];
    let amax = group.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let (mut bs, mut be) = ((amax / qmax).max(1e-12), f32::INFINITY);
    for &c in &GRID {
        let s = (c * amax / qmax).max(1e-12);
        let inv = 1.0 / s;
        let e: f32 = group
            .iter()
            .map(|&v| {
                let d = v - (v * inv).round().clamp(-qmax, qmax) * s;
                d * d
            })
            .sum();
        if e < be {
            be = e;
            bs = s;
        }
    }
    bs
}

/// Per-256-group symmetric-int activation round-trip (absmax — the online grid).
/// `qmax <= 0` ⇒ f16 passthrough (A16). Input already rotated.
fn aquant(x: &[f32], rows: usize, feat: usize, qmax: f32) -> Vec<f32> {
    if qmax <= 0.0 {
        return x.to_vec();
    }
    let mut out = x.to_vec();
    for r in 0..rows {
        for g in (0..feat).step_by(GROUP) {
            let grp = &x[r * feat + g..r * feat + g + GROUP];
            let amax = grp.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let s = (amax / qmax).max(1e-12);
            let inv = 1.0 / s;
            for i in 0..GROUP {
                out[r * feat + g + i] = (grp[i] * inv).round().clamp(-qmax, qmax) * s;
            }
        }
    }
    out
}

const QTIP_BEAM: usize = 16;

/// Weight codec: symmetric-int (Oq) vs trellis (QTIP3, holds the prebuilt codebook).
/// Both operate on an already-rotated 256-group; the LDLQ OBS + rotation + Hessian are
/// codec-agnostic, so swapping this is all that differs between Oq3 and QTIP3.
enum Codec {
    Int,
    Qtip(Vec<f32>),
}

fn qmax_of(wbits: u32) -> f32 {
    if wbits == 3 {
        3.0
    } else {
        7.0
    }
}

/// Quantize→dequant one already-rotated 256-group per the codec (int rounding or
/// trellis). `wbits` = 3 or 4.
fn quant_group(grp: &[f32], wbits: u32, codec: &Codec) -> Vec<f32> {
    match codec {
        Codec::Int => {
            let qmax = qmax_of(wbits);
            let scale = symmetric_clipsearch(grp, qmax);
            let inv = 1.0 / scale;
            grp.iter()
                .map(|&v| (v * inv).round().clamp(-qmax, qmax) * scale)
                .collect()
        }
        Codec::Qtip(cb) => qtip_group_requant(grp, wbits, QTIP_BEAM, cb),
    }
}

/// Per-256-group RTN quant of an ALREADY-ROTATED weight `[out,feat]` via the codec;
/// returns dequant in the same (rotated) domain.
fn wquant_pergroup(rw: &[f32], out: usize, feat: usize, wbits: u32, codec: &Codec) -> Vec<f32> {
    let mut res = vec![0.0f32; out * feat];
    for r in 0..out {
        for g in (0..feat).step_by(GROUP) {
            let grp = &rw[r * feat + g..r * feat + g + GROUP];
            let dq = quant_group(grp, wbits, codec);
            res[r * feat + g..r * feat + g + GROUP].copy_from_slice(&dq);
        }
    }
    res
}

/// LDLQ OBS error-feedback quant of an ALREADY-ROTATED weight (the `residual`),
/// `l` = inv-Cholesky of the Hessian in the SAME rotated domain. Bit-parametric
/// mirror of hipfire-quantize::ldlq::oq{3,4}_ldlq_pack; returns rotated dequant.
fn ldlq_obs(
    mut residual: Vec<f64>,
    out: usize,
    feat: usize,
    wbits: u32,
    codec: &Codec,
    l: &Mat<f64>,
) -> Vec<f32> {
    let nb = feat / GROUP;
    let mut deq = vec![0.0f32; out * feat];
    for blk in 0..nb {
        let c0 = blk * GROUP;
        let c1 = c0 + GROUP;
        let mut errs = vec![0.0f64; out * GROUP];
        for r in 0..out {
            let grp: Vec<f32> = (0..GROUP)
                .map(|c| residual[r * feat + c0 + c] as f32)
                .collect();
            let dq = quant_group(&grp, wbits, codec); // trellis or int, per codec
            for c in 0..GROUP {
                deq[r * feat + c0 + c] = dq[c];
                let ucc = l[(c0 + c, c0 + c)];
                errs[r * GROUP + c] = if ucc > 0.0 {
                    (grp[c] as f64 - dq[c] as f64) / ucc
                } else {
                    0.0
                };
            }
        }
        if c1 < feat {
            for r in 0..out {
                for c in 0..GROUP {
                    let ec = errs[r * GROUP + c];
                    if ec == 0.0 {
                        continue;
                    }
                    let col = c0 + c;
                    for f in c1..feat {
                        let usf = l[(f, col)];
                        if usf != 0.0 {
                            residual[r * feat + f] -= ec * usf;
                        }
                    }
                }
            }
        }
    }
    deq
}

fn matmul_t(x: &[f32], w: &[f32], rows: usize, feat: usize, out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out];
    y.par_chunks_mut(out).enumerate().for_each(|(r, yr)| {
        let xr = &x[r * feat..r * feat + feat];
        for o in 0..out {
            let wr = &w[o * feat..o * feat + feat];
            yr[o] = xr.iter().zip(wr).map(|(&a, &b)| a * b).sum();
        }
    });
    y
}

fn energy(yref: &[f32], yhat: &[f32]) -> (f64, f64) {
    let (mut sig, mut noise) = (0.0f64, 0.0f64);
    for (&a, &b) in yref.iter().zip(yhat) {
        sig += (a as f64) * (a as f64);
        let d = (a - b) as f64;
        noise += d * d;
    }
    (sig, noise)
}

/// Input Hessian H̄ = Σ_rows w[r]·xᵀx  [feat×feat]. `w=None` ⇒ plain XᵀX (w≡1);
/// `w=Some` ⇒ GuidedQuant Fisher-weighted.
fn hessian(x: &[f32], rows: usize, feat: usize, w: Option<&[f32]>) -> Vec<f64> {
    let mut h = vec![0.0f64; feat * feat];
    for r in 0..rows {
        let wr = w.map_or(1.0f64, |w| w[r] as f64);
        if wr == 0.0 {
            continue;
        }
        let xr = &x[r * feat..r * feat + feat];
        for i in 0..feat {
            let xi = wr * xr[i] as f64;
            if xi == 0.0 {
                continue;
            }
            let hrow = &mut h[i * feat..i * feat + feat];
            for (o, &xj) in hrow.iter_mut().zip(xr.iter()) {
                *o += xi * xj as f64;
            }
        }
    }
    h
}

/// GuidedQuant per-token Fisher weight from a tensor's output adjoint d [rows,out]:
/// w[n] = mean_c d[n,c]², normalized so mean(w)=1 (scale-invariant for LDLQ, keeps
/// H̄ ~ plain-H magnitude so the damping is comparable).
fn fisher(d: &[f32], rows: usize, out: usize) -> Vec<f32> {
    let mut w: Vec<f32> = (0..rows)
        .map(|n| {
            d[n * out..n * out + out]
                .iter()
                .map(|&v| v * v)
                .sum::<f32>()
                / out as f32
        })
        .collect();
    let mean = w.iter().sum::<f32>() / rows as f32;
    if mean > 0.0 {
        for x in &mut w {
            *x /= mean;
        }
    }
    w
}

/// The output-adjoint slice + its channel count for eligible tensor `ti`
/// (build order: per layer wq,wk,wv,wo,wgate,wup).
fn adjoint_of<'a>(
    adj: &'a [BlockAdjoints],
    ti: usize,
    qd: usize,
    kvd: usize,
    h: usize,
    inter: usize,
) -> (&'a [f32], usize) {
    let a = &adj[ti / 6];
    match ti % 6 {
        0 => (&a.d_q, qd),
        1 => (&a.d_k, kvd),
        2 => (&a.d_v, kvd),
        3 => (&a.d_attn, h),
        4 => (&a.d_gate, inter),
        _ => (&a.d_up, inter),
    }
}

/// H ← R H Rᵀ for R = per-256-group FWHT (row-pass then col-pass).
fn rotate_hessian_fwht(h: &mut [f64], k: usize) {
    let (s1, s2) = signs();
    let nb = k / GROUP;
    let mut buf = [0.0f32; GROUP];
    for r in 0..k {
        for b in 0..nb {
            for c in 0..GROUP {
                buf[c] = h[r * k + b * GROUP + c] as f32;
            }
            cpu_fwht_256(&mut buf, &s1, &s2);
            for c in 0..GROUP {
                h[r * k + b * GROUP + c] = buf[c] as f64;
            }
        }
    }
    for col in 0..k {
        for b in 0..nb {
            for r in 0..GROUP {
                buf[r] = h[(b * GROUP + r) * k + col] as f32;
            }
            cpu_fwht_256(&mut buf, &s1, &s2);
            for r in 0..GROUP {
                h[(b * GROUP + r) * k + col] = buf[r] as f64;
            }
        }
    }
}

fn transpose(a: &[f32], n: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}

/// H ← M H Mᵀ for a dense orthonormal M (via rotate_rows = x·Mᵀ, twice).
fn rotate_hessian_dense(h: &[f64], k: usize, m: &Rotation) -> Vec<f64> {
    let hf: Vec<f32> = h.iter().map(|&v| v as f32).collect();
    let a = rotate_rows(&hf, m, k); // H Mᵀ
    let at = transpose(&a, k); // (H Mᵀ)ᵀ = M H  (H symmetric)
    let hm = rotate_rows(&at, m, k); // M H Mᵀ
    hm.iter().map(|&v| v as f64).collect()
}

fn inv_cholesky_lower_rotated(h: &[f64], k: usize, damp: f64) -> Option<Mat<f64>> {
    let base = damp.max(1e-12);
    for mult in [1.0, 10.0, 100.0, 1000.0, 10000.0] {
        let lambda = base * mult;
        let hd = Mat::<f64>::from_fn(k, k, |i, j| {
            h[i * k + j] + if i == j { lambda } else { 0.0 }
        });
        let Ok(chol) = hd.llt(Side::Lower) else {
            continue;
        };
        let hinv = chol.solve(Mat::<f64>::identity(k, k));
        let Ok(chol2) = hinv.llt(Side::Lower) else {
            continue;
        };
        return Some(chol2.L().to_owned());
    }
    None
}

fn l_for(h_rot: &[f64], feat: usize) -> Option<Mat<f64>> {
    let diag: f64 = (0..feat).map(|i| h_rot[i * feat + i]).sum();
    let damp = 0.01 * (diag / feat as f64).max(1e-12);
    inv_cholesky_lower_rotated(h_rot, feat, damp)
}

fn upload(gpu: &mut Gpu, t: &GpuTensor, w: &[f32]) -> HipResult<()> {
    let bytes = unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) };
    gpu.memcpy_htod_auto(&t.buf, bytes)
}

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

struct Tensor {
    w: Vec<f32>, // original [out, feat]
    yref: Vec<f32>,
    out: usize,
    feat: usize,
    xkey: usize,    // index into inputs
    residual: bool, // input is residual stream (R1 applies) vs o_proj ctx
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Args: [model_dir] [--text <file>]. --text = real-text calibration via the model's
    // tokenizer.json (meaningful Fisher weights); without it, synthetic tokens.
    let mut dir_s: Option<String> = None;
    let mut text_path: Option<String> = None;
    let mut eval_offset: usize = 0; // shift the held-out eval window (multi-window via reruns)
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--text" => text_path = it.next(),
            "--eval-offset" => eval_offset = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            _ => {
                if dir_s.is_none() {
                    dir_s = Some(a);
                }
            }
        }
    }
    let dir_s = dir_s.unwrap_or_else(|| DEFAULT_DIR.to_string());
    let dir = Path::new(&dir_s);
    if !dir.exists() {
        return Err(format!("model dir not found: {} (argv[1])", dir.display()).into());
    }
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    let (cfg, w) = load_llama_fp32(&mut gpu, dir).map_err(|e| format!("load: {e}"))?;
    let (h, vocab) = (cfg.hidden_size, cfg.vocab_size);
    println!(
        "arch: {}  model: {}  h={h} layers={}",
        gpu.arch,
        dir.display(),
        cfg.num_hidden_layers
    );

    // Tokens: a CALIB window [0,SEQ) to calibrate quant (Hessians/Fisher/R1) and a DISJOINT
    // EVAL window [SEQ,2·SEQ) to measure quality — held-out, so no in-sample optimism.
    let need = SEQ * (2 + eval_offset); // calib [0,SEQ) + eval window at offset
    let tokens_all: Vec<u32> = if let Some(tp) = &text_path {
        let tok = Tokenizer::from_tokenizer_json(&dir.join("tokenizer.json"))?
            .ok_or("no tokenizer.json in model dir")?;
        let text = std::fs::read_to_string(tp)?;
        let ids = tok.encode(&text);
        if ids.len() < need {
            return Err(format!("text too short: {} < {} tokens", ids.len(), need).into());
        }
        println!("real-text: {} chars -> {} tokens", text.len(), ids.len());
        ids[..need].to_vec()
    } else {
        println!("synthetic tokens");
        (0..need)
            .map(|t| (13 + t * 97) as u32 % cfg.vocab_size as u32)
            .collect()
    };
    let calib_tokens = tokens_all[..SEQ].to_vec();
    let e0 = SEQ * (1 + eval_offset);
    let eval_tokens = tokens_all[e0..e0 + SEQ].to_vec();
    println!(
        "calib [0,{SEQ})  eval [{e0},{}) (offset {eval_offset})",
        e0 + SEQ
    );
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();
    let model = LlamaModel::from_f32_weights(&mut gpu, &cfg, w, SEQ, 2, 1.0)?;
    let (qd, kvd, inter) = (model.dims.q_dim(), model.dims.kv_dim(), model.dims.inter);

    // CALIB forward + guided backward → Fisher adjoints; capture full-SEQ inputs for Hessians.
    let calib_acts = model_forward(&mut gpu, &model, &calib_tokens, &pos)?;
    let targets: Vec<f32> = (0..SEQ)
        .map(|n| {
            if n + 1 < SEQ {
                calib_tokens[n + 1] as f32
            } else {
                -100.0
            }
        })
        .collect();
    let (gloss, adjoints) = model_guided_adjoints(&mut gpu, &model, &calib_acts, &targets, -100)?;
    println!("calib CE loss/tok = {:.3}", gloss / (SEQ - 1) as f32);
    let mut calib_inputs: Vec<(Vec<f32>, usize, usize)> = Vec::new();
    let mut residual_input = Vec::new();
    for i in 0..model.layers.len() {
        let a = &calib_acts.layer_acts[i];
        calib_inputs.push((gpu.download_f32(&a.xn1)?, SEQ, h));
        residual_input.push(true);
        calib_inputs.push((gpu.download_f32(&a.ctx)?, SEQ, qd));
        residual_input.push(false);
        calib_inputs.push((gpu.download_f32(&a.xn2)?, SEQ, h));
        residual_input.push(true);
    }

    // EVAL forward → held-out fp logits (KLD ref) + first-EVAL_ROWS inputs (SNR).
    let eval_acts = model_forward(&mut gpu, &model, &eval_tokens, &pos)?;
    let refl = gpu.download_f32(&eval_acts.logits)?; // [SEQ,vocab]; KLD uses first EVAL_ROWS
    let mut eval_inputs: Vec<(Vec<f32>, usize, usize)> = Vec::new();
    for i in 0..model.layers.len() {
        let a = &eval_acts.layer_acts[i];
        for (g, feat) in [(&a.xn1, h), (&a.ctx, qd), (&a.xn2, h)] {
            let full = gpu.download_f32(g)?;
            eval_inputs.push((full[..EVAL_ROWS * feat].to_vec(), EVAL_ROWS, feat));
        }
    }

    // Eligible tensors (contraction %256); yref from the held-out EVAL activations.
    let mut tensors: Vec<Tensor> = Vec::new();
    for (i, (lw, _)) in model.layers.iter().enumerate() {
        let base = i * 3;
        for (t, out, feat, xk) in [
            (&lw.wq, qd, h, base),
            (&lw.wk, kvd, h, base),
            (&lw.wv, kvd, h, base),
            (&lw.wo, h, qd, base + 1),
            (&lw.wgate, inter, h, base + 2),
            (&lw.wup, inter, h, base + 2),
        ] {
            if feat % GROUP != 0 {
                continue;
            }
            let wv = gpu.download_f32(t)?;
            let (ex, erows, _) = &eval_inputs[xk];
            let yref = matmul_t(ex, &wv, *erows, feat, out);
            tensors.push(Tensor {
                w: wv,
                yref,
                out,
                feat,
                xkey: xk,
                residual: residual_input[xk],
            });
        }
    }
    let nt = tensors.len();
    println!("eligible tensors: {nt}; calib SEQ={SEQ} eval rows={EVAL_ROWS}; down_proj inter={inter} excluded\n");

    // Learn dense R1 (h×h) on CALIB residual activations.
    let do_learned = std::env::var("HIPFIRE_WA_LEARNED")
        .map(|v| v != "0")
        .unwrap_or(true);
    let r1 = if do_learned {
        let mut xres = Vec::new();
        let mut rres = 0usize;
        for (idx, (x, rows, feat)) in calib_inputs.iter().enumerate() {
            if !residual_input[idx] || *feat != h {
                continue;
            }
            let mut r = 0;
            while r < *rows {
                xres.extend_from_slice(&x[r * h..r * h + h]);
                rres += 1;
                r += 16; // ~2048 learn rows — keeps the (partly-serial) Cayley-SGD tractable
            }
        }
        // learn_rotation_kurtosis is now rayon-parallel (matmuls over output rows), so a
        // heavier M is affordable at h=2048. HIPFIRE_WA_R1_ITERS tunes it (default 100).
        let iters: usize = std::env::var("HIPFIRE_WA_R1_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        println!("learning R1 (h={h}) on {rres} calib residual rows ({iters} iters)...");
        learn_rotation_kurtosis(&xres, rres, h, Rotation::hadamard(h, 1), iters, 0.05, 6)
    } else {
        println!("learned-R1 overlay OFF (HIPFIRE_WA_LEARNED=0)");
        Rotation::identity(h)
    };

    // Per-input caches (parallel): rotated EVAL activations (rotx, for SNR) + inv-Cholesky
    // of the CALIB Hessian (L, for LDLQ), in the FWHT and R1 domains.
    let per: Vec<(Vec<f32>, Option<Mat<f64>>, Vec<f32>, Option<Mat<f64>>)> = (0..calib_inputs
        .len())
        .into_par_iter()
        .map(|idx| {
            let (cx, crows, feat) = &calib_inputs[idx];
            let (ex, erows, _) = &eval_inputs[idx];
            let rx_f = fwht_rows(ex, *erows, *feat, false);
            let mut hf = hessian(cx, *crows, *feat, None);
            rotate_hessian_fwht(&mut hf, *feat);
            let l_f = l_for(&hf, *feat);
            let (rx_m, l_m) = if do_learned && residual_input[idx] && *feat == h {
                let rxm = rotate_rows(ex, &r1, *erows);
                let hm = rotate_hessian_dense(&hessian(cx, *crows, *feat, None), *feat, &r1);
                (rxm, l_for(&hm, *feat))
            } else {
                (rx_f.clone(), l_f.clone())
            };
            (rx_f, l_f, rx_m, l_m)
        })
        .collect();
    let (mut rotx_f, mut lf, mut rotx_m, mut lm) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (a, b, c, d) in per {
        rotx_f.push(a);
        lf.push(b);
        rotx_m.push(c);
        lm.push(d);
    }

    // Rotated weights per tensor (weight-only; calib-independent).
    let rotw_f: Vec<Vec<f32>> = tensors
        .par_iter()
        .map(|t| fwht_rows(&t.w, t.out, t.feat, false))
        .collect();
    let rotw_m: Vec<Vec<f32>> = tensors
        .par_iter()
        .map(|t| {
            if do_learned && t.residual && t.feat == h {
                rotate_rows(&t.w, &r1, t.out)
            } else {
                fwht_rows(&t.w, t.out, t.feat, false)
            }
        })
        .collect();

    // GuidedQuant per-tensor guided L from the CALIB Fisher-weighted Hessian.
    let lg: Vec<Option<Mat<f64>>> = (0..tensors.len())
        .into_par_iter()
        .map(|ti| {
            let t = &tensors[ti];
            let (x, rows, feat) = &calib_inputs[t.xkey];
            let (d, outa) = adjoint_of(&adjoints, ti, qd, kvd, h, inter);
            let w = fisher(d, *rows, outa);
            let mut hh = hessian(x, *rows, *feat, Some(&w));
            rotate_hessian_fwht(&mut hh, *feat);
            l_for(&hh, *feat)
        })
        .collect();

    // HIPFIRE_WA_CODEC=qtip3 swaps the symmetric-int weight quant for the QTIP3 trellis;
    // rotation / LDLQ / guided-Hessian are codec-agnostic, so this closes SpinQuant+
    // GuidedQuant onto QTIP3 by reusing the whole harness.
    let codec = match std::env::var("HIPFIRE_WA_CODEC").ok().as_deref() {
        Some("qtip3") | Some("qtip") => Codec::Qtip(build_codebook()),
        _ => Codec::Int,
    };
    let codec_name = if matches!(codec, Codec::Qtip(_)) {
        "QTIP3-trellis"
    } else {
        "Oq-int"
    };
    println!("weight codec: {codec_name}\n");
    let aqmax = |bits: u32| match bits {
        4 => 7.0,
        8 => 127.0,
        _ => 0.0,
    };

    // Weight fake-quant in a rotation domain × calib (RTN / LDLQ plain-XᵀX / LDLQ-G
    // GuidedQuant Fisher). LDLQ-G is FWHT-domain (learned overlay stays plain LDLQ).
    let wq_rotated = |ti: usize, learned: bool, calib: &str, wbits: u32| -> Vec<f32> {
        let t = &tensors[ti];
        let rw = if learned { &rotw_m[ti] } else { &rotw_f[ti] };
        let use_l: Option<&Mat<f64>> = match calib {
            "LDLQ" => (if learned { &lm[t.xkey] } else { &lf[t.xkey] }).as_ref(),
            "LDLQ-G" => lg[ti].as_ref(),
            _ => None,
        };
        match use_l {
            Some(l) => {
                let res: Vec<f64> = rw.iter().map(|&v| v as f64).collect();
                ldlq_obs(res, t.out, t.feat, wbits, &codec, l)
            }
            None => wquant_pergroup(rw, t.out, t.feat, wbits, &codec),
        }
    };

    // ---- METRIC 1: per-tensor output SNR matrix ----
    println!("\n  per-tensor end-to-end output SNR (dB), energy-weighted over {nt} tensors:");
    println!("  rot     calib   Wbits |     A4      A8     A16");
    println!("  ---------------------+------------------------");
    let rots: &[(&str, bool)] = if do_learned {
        &[("FWHT", false), ("R1", true)]
    } else {
        &[("FWHT", false)]
    };
    for &(rot_name, learned) in rots {
        let calibs: &[&str] = if learned {
            &["RTN", "LDLQ"]
        } else {
            &["RTN", "LDLQ", "LDLQ-G"]
        };
        for &calib in calibs {
            for wbits in [3u32, 4] {
                let wqs: Vec<Vec<f32>> = (0..nt)
                    .into_par_iter()
                    .map(|ti| wq_rotated(ti, learned, calib, wbits))
                    .collect();
                let abits_list: &[u32] = if learned { &[4] } else { &[4, 8, 16] }; // R1 overlay: A4 only
                let mut row = format!("  {rot_name:<6}  {calib:<6}  W{wbits}  |");
                for &abits in abits_list {
                    let (mut sig, mut noise) = (0.0f64, 0.0f64);
                    for ti in 0..nt {
                        let t = &tensors[ti];
                        let (_, rows, feat) = &eval_inputs[t.xkey];
                        let rx = if learned {
                            &rotx_m[t.xkey]
                        } else {
                            &rotx_f[t.xkey]
                        };
                        let xq = aquant(rx, *rows, *feat, aqmax(abits));
                        let yhat = matmul_t(&xq, &wqs[ti], *rows, *feat, t.out);
                        let (s, n) = energy(&t.yref, &yhat);
                        sig += s;
                        noise += n;
                    }
                    row.push_str(&format!(
                        " {:6.2} ",
                        10.0 * (sig / noise.max(1e-30)).log10()
                    ));
                }
                if learned {
                    row.push_str("   —       —    (R1 = residual readers; A4 only)");
                }
                println!("{row}");
            }
        }
    }

    // ---- METRIC 2: end-to-end decode KLD (A16), FWHT, in-place weight fake-quant ----
    println!("\n  end-to-end decode KLD (A16, full forward vs fp; lower = better):");
    println!("  calib   Wbits |   KLD");
    println!("  -------------+--------");
    for calib in ["RTN", "LDLQ", "LDLQ-G"] {
        for wbits in [3u32, 4] {
            // fake-quant each eligible weight in original domain (parallel), upload serially
            let origs: Vec<Vec<f32>> = (0..nt)
                .into_par_iter()
                .map(|ti| {
                    let t = &tensors[ti];
                    let rotq = wq_rotated(ti, false, calib, wbits); // FWHT-domain quant
                    fwht_rows(&rotq, t.out, t.feat, true) // inverse FWHT → original domain
                })
                .collect();
            for ti in 0..nt {
                upload_tensor(&mut gpu, &model, ti, &origs[ti])?;
            }
            let a = model_forward(&mut gpu, &model, &eval_tokens, &pos)?;
            let l = gpu.download_f32(&a.logits)?;
            println!(
                "  {calib:<6}  W{wbits}  | {:.5}",
                kld(&refl, &l, EVAL_ROWS, vocab)
            );
            // restore originals
            for ti in 0..nt {
                upload_tensor(&mut gpu, &model, ti, &tensors[ti].w)?;
            }
        }
    }
    println!("\n  (SNR: W3=int3[-3,3] W4=int4[-7,7]; A4/A8 per-256-group absmax, A16=f16. KLD anchor\n   quantizes only the %256 tensors — down_proj stays fp — so it isolates the same set.)");
    Ok(())
}

/// Upload `w` into the `ti`-th eligible model weight (build-order: per layer
/// wq,wk,wv,wo,wgate,wup, all eligible since every contraction dim is %256).
fn upload_tensor(gpu: &mut Gpu, model: &LlamaModel, ti: usize, w: &[f32]) -> HipResult<()> {
    let per = 6;
    let (layer, slot) = (ti / per, ti % per);
    let (lw, _) = &model.layers[layer];
    let t = match slot {
        0 => &lw.wq,
        1 => &lw.wk,
        2 => &lw.wv,
        3 => &lw.wo,
        4 => &lw.wgate,
        _ => &lw.wup,
    };
    upload(gpu, t, w)
}
