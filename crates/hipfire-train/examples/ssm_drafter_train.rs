//! Train the GLA-lite / minimal-selective-SSM PFlash drafter to reproduce the
//! qwen3.5 target's MID-layer block ranking, and measure it against the same
//! shallow-K bar + the +0.47 ATTENTION-drafter ceiling (P5). Same labels, same
//! production scoring, same ListNet/AdamW loop as `pflash_drafter_train` — only
//! the drafter body differs (gated recurrence vs attention).
//!
//! Requires daemon labels (teacher/student split; real qwen3.5 target):
//!   HIPFIRE_PFLASH_DAEMON_LABELS=/tmp/pflash_q35_labels.jsonl \
//!   cargo run -p hipfire-train --release --example ssm_drafter_train
//!
//! Env knobs: HIPFIRE_PFLASH_{EPOCHS,TAU,LR,WD}, HIPFIRE_SSM_LAYERS, HIPFIRE_SSM_H.

use hipfire_train::optim::AdamW;
use hipfire_train::ssm_drafter::{
    free_ssm_drafter_acts, free_ssm_drafter_grads, ssm_drafter_backward, ssm_drafter_forward_train,
    SsmDrafter, SsmDrafterConfig,
};
use rdna_compute::{DType, Gpu};

const SEQ: usize = 512;
const BLOCK: usize = 64;
const N_TRAIN: usize = 32;
const N_EVAL: usize = 8;
const EPOCHS: usize = 300;
const TAU: f32 = 0.1;
const EVAL_EVERY: usize = 15;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}
fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn rank(a: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..a.len()).collect();
    idx.sort_by(|&i, &j| a[i].partial_cmp(&a[j]).unwrap());
    let mut r = vec![0.0f32; a.len()];
    for (pos, &i) in idx.iter().enumerate() {
        r[i] = pos as f32;
    }
    r
}
fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f32>() as f64 / n, b.iter().sum::<f32>() as f64 / n);
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (da, db) = (a[i] as f64 - ma, b[i] as f64 - mb);
        c += da * db;
        va += da * da;
        vb += db * db;
    }
    if va == 0.0 || vb == 0.0 { 0.0 } else { (c / (va.sqrt() * vb.sqrt())) as f32 }
}
fn spearman(a: &[f32], b: &[f32]) -> f32 {
    pearson(&rank(a), &rank(b))
}
fn softmax_t(x: &[f32], tau: f32) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|&v| ((v - m) / tau).exp()).collect();
    let z: f32 = e.iter().sum();
    e.into_iter().map(|v| v / z).collect()
}

/// Daemon `pflash_labels` JSONL + `<path>.embed.bin` (QEMB) sidecar.
#[allow(clippy::type_complexity)]
fn load_daemon_labels(
    gpu: &mut Gpu,
    jsonl: &str,
) -> Result<(Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, rdna_compute::GpuTensor, usize, usize), Box<dyn std::error::Error>>
{
    let text = std::fs::read_to_string(jsonl)?;
    let (mut chunks, mut label_mid, mut base_shallow) = (Vec::new(), Vec::new(), Vec::new());
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let arr_u32 = |k: &str| -> Vec<u32> {
            v[k].as_array().map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect()).unwrap_or_default()
        };
        let arr_f32 = |k: &str| -> Vec<f32> {
            v[k].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect()).unwrap_or_default()
        };
        let toks = arr_u32("tokens");
        assert_eq!(toks.len(), SEQ, "daemon label chunk len {} != SEQ {SEQ}", toks.len());
        chunks.push(toks);
        label_mid.push(arr_f32("mid_scores"));
        base_shallow.push(arr_f32("shallow_scores"));
    }
    let bytes = std::fs::read(format!("{jsonl}.embed.bin"))?;
    if &bytes[0..4] != b"QEMB" {
        return Err("daemon embed sidecar: bad magic".into());
    }
    let vocab = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    let dim = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
    let data: Vec<f32> =
        bytes[12..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(data.len(), vocab * dim, "embed sidecar size mismatch");
    let embed = gpu.upload_f32(&data, &[vocab, dim])?;
    println!("daemon labels: {} chunks, embed [{vocab}×{dim}]", chunks.len());
    Ok((chunks, label_mid, base_shallow, embed, dim, vocab))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    let nb = SEQ / BLOCK;
    let last = SEQ - 1;
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();

    let dlpath = std::env::var("HIPFIRE_PFLASH_DAEMON_LABELS")
        .map_err(|_| "set HIPFIRE_PFLASH_DAEMON_LABELS=<jsonl> (real qwen3.5 labels)")?;
    println!("arch: {}  SEQ={SEQ} BLOCK={BLOCK} blocks={nb} train={N_TRAIN} eval={N_EVAL}", gpu.arch);
    println!("labels: daemon source {dlpath} (real qwen3.5 target)");
    let (chunks, label_mid, base_shallow, embed, h_t, vocab) = load_daemon_labels(&mut gpu, &dlpath)?;

    let n_chunks = chunks.len();
    assert!(n_chunks == N_TRAIN + N_EVAL, "expected {} chunks, got {n_chunks}", N_TRAIN + N_EVAL);
    let scores_dev = gpu.zeros(&[nb], DType::F32)?;
    let bar: f32 = (N_TRAIN..n_chunks).map(|i| spearman(&base_shallow[i], &label_mid[i])).sum::<f32>()
        / N_EVAL as f32;

    // ── SSM drafter + training ──
    let n_layers = env_usize("HIPFIRE_SSM_LAYERS", 3);
    let h_draft = env_usize("HIPFIRE_SSM_H", 512);
    let mut dcfg = SsmDrafterConfig::tiny(10000.0, 1e-5);
    dcfg.n_layers = n_layers;
    dcfg.h_draft = h_draft;
    let kvd_d = dcfg.kv_dim();
    let drafter = SsmDrafter::new(&mut gpu, embed, h_t, vocab, dcfg, SEQ)?;
    let sizes = drafter.param_sizes();
    let nparams: usize = sizes.iter().sum();
    let epochs = env_usize("HIPFIRE_PFLASH_EPOCHS", EPOCHS);
    let tau = env_f32("HIPFIRE_PFLASH_TAU", TAU);
    let lr = env_f32("HIPFIRE_PFLASH_LR", 1e-3);
    let wd = env_f32("HIPFIRE_PFLASH_WD", 0.0);
    let mut opt = AdamW::new(&mut gpu, &sizes, lr, 0.9, 0.999, 1e-8, wd)?;
    println!(
        "SSM drafter: h={} layers={} inter={} kv={}×{}  params={} ({:.2}M)  epochs={epochs} tau={tau} lr={lr} wd={wd}",
        dcfg.h_draft, dcfg.n_layers, dcfg.inter, dcfg.n_kv, dcfg.head_dim, sizes.len(),
        nparams as f32 / 1e6
    );

    let eval = |gpu: &mut Gpu, d: &SsmDrafter| -> f32 {
        let sc = gpu.zeros(&[nb], DType::F32).unwrap();
        let mut s = 0.0;
        for i in N_TRAIN..n_chunks {
            let a = ssm_drafter_forward_train(gpu, d, &chunks[i], &pos).unwrap();
            pflash_score_fwd(gpu, &a.score_k, &sc, kvd_d, nb, last);
            let pred = gpu.download_f32(&sc).unwrap();
            s += spearman(&pred, &label_mid[i]);
            free_ssm_drafter_acts(gpu, a).unwrap();
        }
        let _ = gpu.free_tensor(sc);
        s / N_EVAL as f32
    };

    println!("\n  bar  Spearman(shallow, mid)   [eval] = {bar:+.3}  ← drafter must beat this");
    println!("  ref  attention-drafter ceiling       ≈ +0.47  (P5, tuning-resistant)");
    println!("  init Spearman(ssm-drafter, mid)[eval] = {:+.3}\n", eval(&mut gpu, &drafter));

    let mut best_corr = f32::NEG_INFINITY;
    let mut best_ep = 0usize;
    for ep in 0..epochs {
        let mut ep_loss = 0.0f32;
        for i in 0..N_TRAIN {
            let acts = ssm_drafter_forward_train(&mut gpu, &drafter, &chunks[i], &pos)?;
            pflash_score_fwd(&mut gpu, &acts.score_k, &scores_dev, kvd_d, nb, last);
            let pred = gpu.download_f32(&scores_dev)?;
            let pl = softmax_t(&label_mid[i], tau);
            let pp = softmax_t(&pred, tau);
            let mut ds = vec![0.0f32; nb];
            let mut l = 0.0f32;
            for b in 0..nb {
                l -= pl[b] * pp[b].max(1e-12).ln();
                ds[b] = (pp[b] - pl[b]) / tau;
            }
            ep_loss += l;
            let dscores = gpu.upload_f32(&ds, &[nb])?;
            let grads = ssm_drafter_backward(&mut gpu, &drafter, &acts, &dscores, BLOCK, nb, last)?;
            opt.step(&mut gpu, &drafter.params(), &grads.flat())?;
            free_ssm_drafter_acts(&mut gpu, acts)?;
            free_ssm_drafter_grads(&mut gpu, grads)?;
            gpu.free_tensor(dscores)?;
        }
        if ep % EVAL_EVERY == 0 || ep == epochs - 1 {
            let corr = eval(&mut gpu, &drafter);
            if corr > best_corr {
                best_corr = corr;
                best_ep = ep;
            }
            if ep % 30 == 0 || ep == epochs - 1 {
                println!(
                    "  ep {ep:>3}  train_loss {:.4}  eval {:+.3}  (best {:+.3} @ ep {})",
                    ep_loss / N_TRAIN as f32, corr, best_corr, best_ep
                );
            }
        }
    }

    println!("\n── SSM drafter result ──");
    println!("  shallow bar       : {bar:+.3}");
    println!("  attn ceiling (P5) : ≈ +0.47");
    println!("  SSM drafter BEST  : {best_corr:+.3} @ ep {best_ep}");
    if best_corr > bar {
        println!("  ✓ SSM drafter BEATS the shallow bar");
    } else if best_corr > 0.47 {
        println!("  ~ SSM drafter beats the attn ceiling but not the shallow bar");
    } else {
        println!("  ✗ SSM drafter did not beat the attn ceiling — ablate up (conv1d / delta rule)");
    }
    Ok(())
}

/// Local helper: PFlash forward (last_pos = SEQ-1) without re-importing the op
/// path each call site.
fn pflash_score_fwd(gpu: &mut Gpu, k: &rdna_compute::GpuTensor, sc: &rdna_compute::GpuTensor, kvd: usize, nb: usize, last: usize) {
    hipfire_train::ops::pflash_score::pflash_score_forward(gpu, k, sc, SEQ, kvd, BLOCK, nb, last).unwrap();
}
