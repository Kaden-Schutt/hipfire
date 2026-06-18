//! PFlash drafter prototype — P1 (label capture) + P2 (arch scaffold).
//!
//! P1: run a LOADABLE Llama stand-in target, capture its MID-layer K, compute the
//!     per-block cosine ranking = training label (the strong +0.81 M0b signal).
//!     Also capture the target's SHALLOW-layer ranking = PFlash's current baseline
//!     (the bar a trained drafter must beat).
//! P2: build the shared-embedding tiny-attention drafter, run it UNTRAINED, and
//!     measure its block ranking vs (a) the target mid-layer label and (b) the
//!     shallow baseline. Untrained ≈ chance — this establishes the P2 starting
//!     point that P3 training has to move.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   source ./scripts/gpu-lock.sh && gpu_acquire "pflash-drafter"
//!   cargo run -p hipfire-train --release --example pflash_drafter_proto
//!   gpu_release

use hipfire_model::tokenizer::Tokenizer;
use hipfire_train::drafter::{drafter_forward, Drafter, DrafterConfig};
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{model_forward, LlamaModel};
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};

const HF: &str = "/srv/huggingface";
const TARGET: &str = "models--meta-llama--Llama-3.2-3B-Instruct";
const SEQ: usize = 512;
const BLOCK: usize = 64;
const SHALLOW: usize = 1;
const NEEDLE_BLOCK: usize = 1;

fn snapshot_dir(repo: &str) -> Option<PathBuf> {
    let snaps = Path::new(HF).join(repo).join("snapshots");
    std::fs::read_dir(&snaps).ok()?.flatten().map(|e| e.path()).find(|p| p.is_dir())
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
fn rank_of(a: &[f32], i: usize) -> usize {
    a.iter().filter(|&&v| v > a[i]).count() + 1
}

/// Per-block head-averaged cosine(block_mean_K, last_token_K). k:[seq*n_kv*hd].
fn block_scores(k: &[f32], seq: usize, n_kv: usize, hd: usize, block: usize) -> Vec<f32> {
    let kvd = n_kv * hd;
    let nb = seq / block;
    let last = &k[(seq - 1) * kvd..seq * kvd];
    let cos = |a: &[f32], b: &[f32]| {
        let (mut d, mut na, mut nb_) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..a.len() {
            d += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb_ += (b[i] as f64).powi(2);
        }
        if na == 0.0 || nb_ == 0.0 { 0.0 } else { d / (na.sqrt() * nb_.sqrt()) }
    };
    let mut scores = vec![0.0f32; nb];
    for b in 0..nb {
        let mut mean = vec![0.0f32; kvd];
        for t in b * block..(b + 1) * block {
            for j in 0..kvd {
                mean[j] += k[t * kvd + j];
            }
        }
        for v in mean.iter_mut() {
            *v /= block as f32;
        }
        let mut s = 0.0f64;
        for h in 0..n_kv {
            s += cos(&mean[h * hd..(h + 1) * hd], &last[h * hd..(h + 1) * hd]);
        }
        scores[b] = (s / n_kv as f64) as f32;
    }
    scores
}

fn build_needle(tok: &Tokenizer) -> Vec<u32> {
    let filler = tok.encode(
        "The committee reviewed the quarterly logistics report in the main hall and noted no issues. ",
    );
    let needle = tok.encode("Critical fact to remember: the secret access word is tungsten. ");
    let query = tok.encode(" In summary, the one secret access word requested above is");
    let f0 = *filler.first().unwrap_or(&0);
    let mut ids = vec![f0; SEQ];
    for t in 0..SEQ {
        ids[t] = filler[t % filler.len()];
    }
    let base = NEEDLE_BLOCK * BLOCK;
    for (j, &t) in needle.iter().enumerate() {
        if base + j < SEQ {
            ids[base + j] = t;
        }
    }
    let qstart = SEQ - query.len();
    for (j, &t) in query.iter().enumerate() {
        ids[qstart + j] = t;
    }
    ids
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}  SEQ={SEQ} BLOCK={BLOCK}\n", gpu.arch);

    let dir = snapshot_dir(TARGET).ok_or("target snapshot not found")?;
    let tok = Tokenizer::from_hf_json(&std::fs::read_to_string(dir.join("tokenizer.json"))?)
        .map_err(|e| format!("tokenizer: {e:?}"))?;
    let ids = build_needle(&tok);
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();
    let nb = SEQ / BLOCK;

    // ── P1: capture target labels ────────────────────────────────────────────
    let (cfg, w) = load_llama_fp32(&mut gpu, &dir)?;
    let (n_kv_t, hd_t, h_t, vocab) =
        (cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size, cfg.vocab_size);
    let rope_base = cfg.rope_theta;
    let eps = cfg.rms_norm_eps;
    let n_layers = cfg.num_hidden_layers;
    let mid = n_layers / 2;
    let mut target = LlamaModel::from_f32_weights(&mut gpu, &cfg, w, SEQ, 4, 8.0)?;

    println!("target: {TARGET}  h_t={h_t} layers={n_layers} mid=L{mid} kv={n_kv_t}×{hd_t}");
    let acts = model_forward(&mut gpu, &target, &ids, &pos)?;
    let k_mid = gpu.download_f32(&acts.layer_acts[mid].k_r)?;
    let k_shallow = gpu.download_f32(&acts.layer_acts[SHALLOW].k_r)?;
    let label_mid = block_scores(&k_mid, SEQ, n_kv_t, hd_t, BLOCK); // training target
    let base_shallow = block_scores(&k_shallow, SEQ, n_kv_t, hd_t, BLOCK); // bar to beat

    println!("\n── P1 labels (target) ──");
    println!(
        "  needle block rank — mid(label):{}/{nb}  shallow(baseline):{}/{nb}",
        rank_of(&label_mid, NEEDLE_BLOCK),
        rank_of(&base_shallow, NEEDLE_BLOCK)
    );
    println!(
        "  Spearman(shallow_baseline, mid_label) = {:+.3}  (PFlash's current shallow K vs the strong mid signal)",
        spearman(&base_shallow, &label_mid)
    );

    // free the target forward acts; move the embedding into the drafter (shared).
    // (We keep `target.embed` by moving it out — target isn't used again.)
    drop(acts);
    let embed = std::mem::replace(
        &mut target.embed,
        gpu.zeros(&[1], rdna_compute::DType::F32)?,
    );

    // ── P2: untrained drafter ────────────────────────────────────────────────
    let dcfg = DrafterConfig::tiny(rope_base, eps);
    let (n_kv_d, hd_d) = (dcfg.n_kv, dcfg.head_dim);
    let drafter = Drafter::new(&mut gpu, embed, h_t, vocab, dcfg, SEQ)?;
    println!(
        "\n── P2 drafter (UNTRAINED) — h={} layers={} kv={}×{} ──",
        dcfg.h_draft, dcfg.n_layers, n_kv_d, hd_d
    );
    let dk = drafter_forward(&mut gpu, &drafter, &ids, &pos)?;
    let dk_host = gpu.download_f32(&dk)?;
    let draft_scores = block_scores(&dk_host, SEQ, n_kv_d, hd_d, BLOCK);

    println!(
        "  needle block rank — drafter:{}/{nb}   (mid label:{}/{nb})",
        rank_of(&draft_scores, NEEDLE_BLOCK),
        rank_of(&label_mid, NEEDLE_BLOCK)
    );
    println!(
        "  Spearman(drafter, mid_label)   = {:+.3}   ← P3 training must push this UP",
        spearman(&draft_scores, &label_mid)
    );
    println!(
        "  Spearman(drafter, shallow_base)= {:+.3}",
        spearman(&draft_scores, &base_shallow)
    );
    println!(
        "\nP2 done: pipeline runs end-to-end (shared embed → in_proj → {} small blocks → K →\n\
         cosine block scores). Untrained Spearman≈0 is expected. P3 = train the drafter to\n\
         match mid_label (target to beat: shallow baseline {:+.3}).",
        dcfg.n_layers,
        spearman(&base_shallow, &label_mid)
    );
    Ok(())
}
