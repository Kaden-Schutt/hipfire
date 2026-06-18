//! P3 — train the PFlash drafter to reproduce the target's MID-layer block
//! ranking, and measure it against the shallow-K baseline PFlash uses today.
//!
//! Labels and drafter scores both use the SAME production scoring
//! (`pflash_score_forward` = full-kv_dim cosine of block_mean vs last-token K),
//! so a win is genuinely drop-in. Listwise (ListNet top-1) loss, AdamW.
//!
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   source ./scripts/gpu-lock.sh && gpu_acquire "pflash-train"
//!   cargo run -p hipfire-train --release --example pflash_drafter_train

use hipfire_model::tokenizer::Tokenizer;
use hipfire_train::drafter::{
    drafter_backward, drafter_forward_train, free_drafter_acts, free_drafter_grads, Drafter,
    DrafterConfig,
};
use hipfire_train::block::free_block_acts;
use hipfire_train::checkpoint::{load_drafter, load_labels, save_drafter, save_labels};
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{model_block_activations, LlamaModel};
use hipfire_train::ops::pflash_score::pflash_score_forward;
use hipfire_train::optim::AdamW;
use rdna_compute::{DType, Gpu};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

const LABELS_PATH: &str = "target/pflash_labels.bin";
const CKPT_PATH: &str = "target/pflash_drafter_ckpt.bin";
const CKPT_BEST_PATH: &str = "target/pflash_drafter_best.bin";
const CKPT_EVERY: usize = 30;
const EVAL_EVERY: usize = 15;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}
fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

const HF: &str = "/srv/huggingface";
const TARGET: &str = "models--meta-llama--Llama-3.2-3B-Instruct";
const SEQ: usize = 512;
const BLOCK: usize = 64;
const SHALLOW: usize = 1;
const N_TRAIN: usize = 12;
const N_EVAL: usize = 4;
const EPOCHS: usize = 300;
const TAU: f32 = 0.1;

fn snapshot_dir(repo: &str) -> Option<PathBuf> {
    let snaps = Path::new(HF).join(repo).join("snapshots");
    std::fs::read_dir(&snaps).ok()?.flatten().map(|e| e.path()).find(|p| p.is_dir())
}

fn corpus_tokens(tok: &Tokenizer) -> Vec<u32> {
    let mut stack = vec![PathBuf::from("docs"), PathBuf::from("crates")];
    let mut text = String::new();
    while let Some(d) = stack.pop() {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().map_or(false, |x| x == "md" || x == "rs") {
                    if let Ok(t) = std::fs::read_to_string(&p) {
                        text.push_str(&t);
                        text.push('\n');
                    }
                }
                if text.len() > 2_000_000 {
                    break;
                }
            }
        }
    }
    tok.encode(&text)
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
        c += da * db; va += da * da; vb += db * db;
    }
    if va == 0.0 || vb == 0.0 { 0.0 } else { (c / (va.sqrt() * vb.sqrt())) as f32 }
}
fn spearman(a: &[f32], b: &[f32]) -> f32 { pearson(&rank(a), &rank(b)) }

fn softmax_t(x: &[f32], tau: f32) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|&v| ((v - m) / tau).exp()).collect();
    let z: f32 = e.iter().sum();
    e.into_iter().map(|v| v / z).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    let nb = SEQ / BLOCK;
    let last = SEQ - 1;
    println!("arch: {}  SEQ={SEQ} BLOCK={BLOCK} blocks={nb} train={N_TRAIN} eval={N_EVAL}", gpu.arch);

    let dir = snapshot_dir(TARGET).ok_or("target not found")?;
    let tok = Tokenizer::from_hf_json(&std::fs::read_to_string(dir.join("tokenizer.json"))?)
        .map_err(|e| format!("tok: {e:?}"))?;
    let toks = corpus_tokens(&tok);
    let n_chunks = N_TRAIN + N_EVAL;
    if toks.len() < n_chunks * SEQ {
        return Err(format!("corpus too small: {} toks < {}", toks.len(), n_chunks * SEQ).into());
    }
    let mut chunks: Vec<Vec<u32>> =
        (0..n_chunks).map(|i| toks[i * SEQ..(i + 1) * SEQ].to_vec()).collect();
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();

    // ── P1: capture target mid-layer labels + shallow baseline per chunk ──────
    let (cfg, w) = load_llama_fp32(&mut gpu, &dir)?;
    let (n_kv_t, hd_t, h_t, vocab) =
        (cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size, cfg.vocab_size);
    let kvd_t = n_kv_t * hd_t;
    let (rope_base, eps, n_layers) = (cfg.rope_theta, cfg.rms_norm_eps, cfg.num_hidden_layers);
    let mid = n_layers / 2;
    let mut target = LlamaModel::from_f32_weights(&mut gpu, &cfg, w, SEQ, 4, 8.0)?;
    println!("target {TARGET}: h_t={h_t} layers={n_layers} mid=L{mid} vocab={vocab}");

    // Geometry-only key (NOT corpus content): the cache stores its own chunks, so
    // a HIT reuses the exact captured corpus regardless of live docs/+crates churn.
    let key = {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        TARGET.hash(&mut h);
        (SEQ, BLOCK, mid, n_chunks).hash(&mut h);
        h.finish()
    };
    let scores_dev = gpu.zeros(&[nb], DType::F32)?;
    let (label_mid, base_shallow) = if let Some((cached, lm, bs)) = load_labels(LABELS_PATH, key) {
        println!("labels: cache HIT {LABELS_PATH} ({} chunks) — skipping 3B capture", lm.len());
        chunks = cached; // use the exact corpus the labels were computed from
        (lm, bs)
    } else {
        let max_id = chunks.iter().flatten().copied().max().unwrap_or(0);
        assert!((max_id as usize) < vocab, "token id {max_id} ≥ vocab {vocab} → OOB embedding read");
        println!("labels: cache miss — capturing (mid-layer partial forward)…");
        let mut label_mid: Vec<Vec<f32>> = Vec::new();
        let mut base_shallow: Vec<Vec<f32>> = Vec::new();
        for (ci, ids) in chunks.iter().enumerate() {
            // partial forward to the mid layer only — skips the final norm + the
            // huge [seq×vocab] logit GEMM that label capture doesn't need.
            let acts = model_block_activations(&mut gpu, &target, ids, &pos, mid)?;
            pflash_score_forward(&mut gpu, &acts[mid].k_r, &scores_dev, SEQ, kvd_t, BLOCK, nb, last)?;
            label_mid.push(gpu.download_f32(&scores_dev)?);
            pflash_score_forward(&mut gpu, &acts[SHALLOW].k_r, &scores_dev, SEQ, kvd_t, BLOCK, nb, last)?;
            base_shallow.push(gpu.download_f32(&scores_dev)?);
            for b in acts {
                free_block_acts(&mut gpu, b)?;
            }
            eprintln!("  captured labels {}/{}", ci + 1, n_chunks);
        }
        save_labels(LABELS_PATH, key, &chunks, &label_mid, &base_shallow)
            .map_err(|e| format!("save_labels: {e}"))?;
        println!("labels: cached → {LABELS_PATH}");
        (label_mid, base_shallow)
    };
    // shallow baseline vs mid label (the bar to beat), eval split
    let bar: f32 = (N_TRAIN..n_chunks)
        .map(|i| spearman(&base_shallow[i], &label_mid[i]))
        .sum::<f32>() / N_EVAL as f32;

    // move the embedding into the drafter (target not used again)
    let embed = std::mem::replace(&mut target.embed, gpu.zeros(&[1], DType::F32)?);

    // ── P2/P3: drafter + training ─────────────────────────────────────────────
    let dcfg = DrafterConfig::tiny(rope_base, eps);
    let kvd_d = dcfg.kv_dim();
    let drafter = Drafter::new(&mut gpu, embed, h_t, vocab, dcfg, SEQ)?;
    let sizes = drafter.param_sizes();
    let nparams: usize = sizes.iter().sum();
    let epochs = env_usize("HIPFIRE_PFLASH_EPOCHS", EPOCHS);
    let tau = env_f32("HIPFIRE_PFLASH_TAU", TAU);
    let lr = env_f32("HIPFIRE_PFLASH_LR", 1e-3);
    let wd = env_f32("HIPFIRE_PFLASH_WD", 0.0);
    let mut opt = AdamW::new(&mut gpu, &sizes, lr, 0.9, 0.999, 1e-8, wd)?;
    println!(
        "drafter: h={} layers={} kv={}×{}  params={} ({:.2}M)  epochs={epochs} tau={tau} lr={lr} wd={wd}",
        dcfg.h_draft, dcfg.n_layers, dcfg.n_kv, dcfg.head_dim, sizes.len(), nparams as f32 / 1e6
    );

    // resume: reload weights + AdamW state from the checkpoint unless FRESH=1.
    let fresh = std::env::var("HIPFIRE_PFLASH_FRESH").is_ok();
    let start_ep = if fresh {
        0
    } else {
        match load_drafter(&mut gpu, CKPT_PATH, &drafter, &mut opt)? {
            Some(e) => {
                println!("resume: loaded {CKPT_PATH} → continuing from epoch {e}");
                e as usize
            }
            None => 0,
        }
    };
    println!();

    let eval = |gpu: &mut Gpu, d: &Drafter| -> f32 {
        let sc = gpu.zeros(&[nb], DType::F32).unwrap();
        let mut s = 0.0;
        for i in N_TRAIN..n_chunks {
            let a = drafter_forward_train(gpu, d, &chunks[i], &pos).unwrap();
            pflash_score_forward(gpu, &a.score_k, &sc, SEQ, kvd_d, BLOCK, nb, last).unwrap();
            let pred = gpu.download_f32(&sc).unwrap();
            s += spearman(&pred, &label_mid[i]);
            free_drafter_acts(gpu, a).unwrap();
        }
        s / N_EVAL as f32
    };

    println!("  bar  Spearman(shallow, mid) [eval] = {bar:+.3}  ← drafter must beat this");
    println!("  init Spearman(drafter, mid) [eval] = {:+.3}\n", eval(&mut gpu, &drafter));

    // best-eval checkpointing: the drafter overfits past its peak, so keep the
    // best-generalizing weights rather than the final (decayed) ones.
    let mut best_corr = f32::NEG_INFINITY;
    let mut best_ep = start_ep;
    for ep in start_ep..epochs {
        let mut ep_loss = 0.0f32;
        for i in 0..N_TRAIN {
            let acts = drafter_forward_train(&mut gpu, &drafter, &chunks[i], &pos)?;
            pflash_score_forward(&mut gpu, &acts.score_k, &scores_dev, SEQ, kvd_d, BLOCK, nb, last)?;
            let pred = gpu.download_f32(&scores_dev)?;
            // ListNet top-1: L = -Σ p_label log p_pred ; dL/dpred = (p_pred - p_label)/τ
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
            let grads = drafter_backward(&mut gpu, &drafter, &acts, &dscores, BLOCK, nb, last)?;
            opt.step(&mut gpu, &drafter.params(), &grads.flat())?;
            free_drafter_acts(&mut gpu, acts)?;
            free_drafter_grads(&mut gpu, grads)?;
            gpu.free_tensor(dscores)?;
        }
        if ep % EVAL_EVERY == 0 || ep == epochs - 1 {
            let corr = eval(&mut gpu, &drafter);
            if corr > best_corr {
                best_corr = corr;
                best_ep = ep;
                save_drafter(&mut gpu, CKPT_BEST_PATH, &drafter, &opt, ep as u32)?;
            }
            if ep % 30 == 0 || ep == epochs - 1 {
                println!(
                    "  ep {ep:>3}  train_loss {:.4}  eval {:+.3}  (best {:+.3} @ ep {})",
                    ep_loss / N_TRAIN as f32, corr, best_corr, best_ep
                );
            }
        }
        // checkpoint periodically + on the last epoch (epoch+1 = next to run)
        if (ep + 1) % CKPT_EVERY == 0 || ep == epochs - 1 {
            save_drafter(&mut gpu, CKPT_PATH, &drafter, &opt, (ep + 1) as u32)?;
        }
    }

    println!("\n── P3 result ──");
    println!("  shallow bar  : {bar:+.3}");
    println!("  drafter final: {:+.3}", eval(&mut gpu, &drafter));
    println!("  drafter BEST : {best_corr:+.3} @ ep {best_ep}  → {CKPT_BEST_PATH}");
    if best_corr > bar {
        println!("  ✓ drafter BEATS the shallow baseline (best-eval checkpoint)");
    } else {
        println!("  ✗ best did not beat the shallow baseline — tune wd/τ/lr/data");
    }
    Ok(())
}
