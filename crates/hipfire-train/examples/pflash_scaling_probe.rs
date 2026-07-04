#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

//! PFlash scaling-trend probe (hipfire-native dense-Llama ladder).
//!
//! Answers the 405B question honestly: does PFlash's shallow-layer cosine-K
//! importance signal carry REAL information beyond recency, and does that signal
//! GROW with model scale? M0 showed on Supra-50M the ranking is ~pure recency.
//! Here we (1) replace shallow-vs-deep self-correlation (trivially recency-bound)
//! with a CAUSAL keep/drop oracle, (2) plant a DISTANT dependency (a needle in an
//! early block that the final query needs), and (3) sweep a dense-Llama ladder.
//!
//! Per model:
//!   - baseline forward → last-token logits + shallow-layer K.
//!   - oracle: for each context block, replace its tokens with filler, re-run,
//!     importance = KL(baseline_last ‖ ablated_last). Causal "does this block
//!     matter for the next token".
//!   - metric: PFlash cosine(block_mean_K, last_token_K) at the shallow layer.
//!   - recency: block index.
//! Report Spearman(metric,oracle), Spearman(recency,oracle), and the
//! recency-PARTIALLED Spearman(metric,oracle | recency) — the number that says
//! whether the shallow K beats recency at finding what causally matters. Plus the
//! planted-needle block's rank under each, and whether the model retrieved it.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "pflash-scaling"
//!   cargo run -p hipfire-train --release --example pflash_scaling_probe
//!   hipfire gpu-lock release

#![allow(clippy::doc_lazy_continuation, clippy::needless_range_loop)]

use hipfire_model::tokenizer::Tokenizer;
use hipfire_train::block::BlockActivations;
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{model_forward, LlamaModel, ModelActivations};
use hipfire_rdna::Gpu;
use std::path::{Path, PathBuf};

/// Return every tensor of a forward's activations to the pool. `GpuTensor` has no
/// Drop (buffers only return via `free_tensor`), so the ablation loop must free
/// explicitly or VRAM climbs ~2GB/forward and OOMs on the larger models.
fn free_block(gpu: &mut Gpu, b: BlockActivations) {
    let BlockActivations {
        xn1,
        rinv1,
        hq,
        hv,
        q_r,
        k_r,
        v,
        p_all,
        ctx,
        x_mid,
        xn2,
        rinv2,
        gate,
        up,
        act,
        pos,
    } = b;
    for t in [
        xn1, rinv1, hq, hv, q_r, k_r, v, p_all, ctx, x_mid, xn2, rinv2, gate, up, act, pos,
    ] {
        let _ = gpu.free_tensor(t);
    }
}
fn free_acts(gpu: &mut Gpu, a: ModelActivations) {
    let ModelActivations {
        layer_inputs,
        layer_acts,
        x_last,
        rinv_final,
        xf,
        logits,
    } = a;
    for t in layer_inputs {
        let _ = gpu.free_tensor(t);
    }
    for b in layer_acts {
        free_block(gpu, b);
    }
    for t in [x_last, rinv_final, xf, logits] {
        let _ = gpu.free_tensor(t);
    }
}

const HF: &str = "/srv/huggingface";
const SEQ: usize = 512;
const BLOCK: usize = 64; // → 8 blocks
const SHALLOW: usize = 1; // PFlash scores at the shallowest full-attn layer
const NEEDLE_BLOCK: usize = 1; // early, distant from the tail query

/// (repo, tied?) — tied is informational; loader reads it from config.
const LADDER: &[(&str, &str)] = &[
    ("Supra-50M", "models--SupraLabs--Supra-50M-Instruct"),
    ("Llama-3.2-1B", "models--meta-llama--Llama-3.2-1B"),
    ("Llama-3.2-3B", "models--meta-llama--Llama-3.2-3B-Instruct"),
    ("Llama-3.1-8B", "models--meta-llama--Llama-3.1-8B-Instruct"),
];

fn snapshot_dir(repo: &str) -> Option<PathBuf> {
    let snaps = Path::new(HF).join(repo).join("snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.is_dir())
}

// ── stats helpers ──────────────────────────────────────────────────────────
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
    let (ma, mb) = (
        a.iter().sum::<f32>() as f64 / n,
        b.iter().sum::<f32>() as f64 / n,
    );
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (da, db) = (a[i] as f64 - ma, b[i] as f64 - mb);
        c += da * db;
        va += da * da;
        vb += db * db;
    }
    if va == 0.0 || vb == 0.0 {
        0.0
    } else {
        (c / (va.sqrt() * vb.sqrt())) as f32
    }
}
fn spearman(a: &[f32], b: &[f32]) -> f32 {
    pearson(&rank(a), &rank(b))
}
/// Partial Spearman of x,y controlling z (rank-based).
fn partial_spearman(x: &[f32], y: &[f32], z: &[f32]) -> f32 {
    let (rxy, rxz, ryz) = (spearman(x, y), spearman(x, z), spearman(y, z));
    let denom = ((1.0 - rxz * rxz) * (1.0 - ryz * ryz)).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (rxy - rxz * ryz) / denom
    }
}
/// 1-based rank of index `i` in `a` (1 = largest value).
fn rank_of(a: &[f32], i: usize) -> usize {
    a.iter().filter(|&&v| v > a[i]).count() + 1
}

/// Per-block head-averaged cosine(block_mean_K, last_token_K). k: [seq*n_kv*hd].
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
        if na == 0.0 || nb_ == 0.0 {
            0.0
        } else {
            d / (na.sqrt() * nb_.sqrt())
        }
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

fn softmax(logits: &[f32]) -> Vec<f64> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<f64> = logits.iter().map(|&l| (l as f64 - m).exp()).collect();
    let z: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / z).collect()
}
fn kl(p_logits: &[f32], q_logits: &[f32]) -> f32 {
    let (p, q) = (softmax(p_logits), softmax(q_logits));
    let mut s = 0.0f64;
    for i in 0..p.len() {
        if p[i] > 1e-12 {
            s += p[i] * (p[i] / q[i].max(1e-12)).ln();
        }
    }
    s as f32
}

/// Build a SEQ-token input with a planted distant dependency.
/// Returns (ids, filler_token, answer_token).
fn build_needle(tok: &Tokenizer) -> (Vec<u32>, u32, u32) {
    let filler = tok.encode(
        "The committee reviewed the quarterly logistics report in the main hall and noted no issues. ",
    );
    let needle = tok.encode("Critical fact to remember: the secret access word is tungsten. ");
    let query = tok.encode(" In summary, the one secret access word requested above is");
    let answer = tok.encode(" tungsten");
    let filler_tok = *filler.first().unwrap_or(&0);
    let answer_tok = *answer.first().unwrap_or(&0);

    let mut ids = vec![filler_tok; SEQ];
    for t in 0..SEQ {
        ids[t] = filler[t % filler.len()];
    }
    // plant needle at the start of NEEDLE_BLOCK
    let base = NEEDLE_BLOCK * BLOCK;
    for (j, &t) in needle.iter().enumerate() {
        if base + j < SEQ {
            ids[base + j] = t;
        }
    }
    // place query at the very end
    let qstart = SEQ - query.len();
    for (j, &t) in query.iter().enumerate() {
        ids[qstart + j] = t;
    }
    (ids, filler_tok, answer_tok)
}

struct Row {
    label: String,
    layers: usize,
    // shallow-layer metric (PFlash's actual scoring layer)
    sp_metric: f32,
    partial: f32,
    needle_rank_metric: usize,
    // mid-layer metric (depth control: is the signal there, just not shallow?)
    mid_layer: usize,
    partial_mid: f32,
    needle_rank_mid: usize,
    sp_recency: f32,
    needle_rank_recency: usize,
    needle_rank_oracle: usize,
    n_ctx: usize,
    retrieved_prob: f64,
    retrieved_argmax: bool,
}

fn run_model(gpu: &mut Gpu, label: &str, dir: &Path) -> Result<Row, String> {
    let tok = Tokenizer::from_hf_json(
        &std::fs::read_to_string(dir.join("tokenizer.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("tokenizer: {e:?}"))?;
    let (cfg, w) = load_llama_fp32(gpu, dir)?;
    let (n_kv, hd, vocab, n_layers) = (
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.vocab_size,
        cfg.num_hidden_layers,
    );
    let model =
        LlamaModel::from_f32_weights(gpu, &cfg, w, SEQ, 4, 8.0).map_err(|e| format!("{e:?}"))?;
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();

    let (ids, filler_tok, answer_tok) = build_needle(&tok);

    // baseline
    let base = model_forward(gpu, &model, &ids, &pos).map_err(|e| format!("{e:?}"))?;
    let logits_all = gpu
        .download_f32(&base.logits)
        .map_err(|e| format!("{e:?}"))?;
    let base_last = logits_all[(SEQ - 1) * vocab..SEQ * vocab].to_vec();
    let probs = softmax(&base_last);
    let retrieved_prob = probs[answer_tok as usize];
    let retrieved_argmax = base_last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        == Some(answer_tok);

    let mid_layer = (n_layers / 2).min(base.layer_acts.len() - 1);
    let k_shallow = gpu
        .download_f32(&base.layer_acts[SHALLOW].k_r)
        .map_err(|e| format!("{e:?}"))?;
    let k_mid = gpu
        .download_f32(&base.layer_acts[mid_layer].k_r)
        .map_err(|e| format!("{e:?}"))?;
    let metric_all = block_scores(&k_shallow, SEQ, n_kv, hd, BLOCK);
    let metric_mid_all = block_scores(&k_mid, SEQ, n_kv, hd, BLOCK);
    // everything we need from the baseline is now on host (base_last, k_*); free
    // its device buffers before the ablation loop so big models (8B) fit in VRAM.
    free_acts(gpu, base);

    // causal oracle over CONTEXT blocks only (exclude the final query block)
    let nb = SEQ / BLOCK;
    let n_ctx = nb - 1;
    let mut oracle = vec![0.0f32; n_ctx];
    for b in 0..n_ctx {
        let mut ab = ids.clone();
        for t in b * BLOCK..(b + 1) * BLOCK {
            ab[t] = filler_tok;
        }
        let abf = model_forward(gpu, &model, &ab, &pos).map_err(|e| format!("{e:?}"))?;
        let abl = gpu
            .download_f32(&abf.logits)
            .map_err(|e| format!("{e:?}"))?;
        let ab_last = &abl[(SEQ - 1) * vocab..SEQ * vocab];
        oracle[b] = kl(&base_last, ab_last);
        free_acts(gpu, abf); // return this forward's buffers to the pool
    }

    let metric = metric_all[..n_ctx].to_vec();
    let metric_mid = metric_mid_all[..n_ctx].to_vec();
    let recency: Vec<f32> = (0..n_ctx).map(|b| b as f32).collect();

    Ok(Row {
        label: label.to_string(),
        layers: n_layers,
        sp_metric: spearman(&metric, &oracle),
        partial: partial_spearman(&metric, &oracle, &recency),
        needle_rank_metric: rank_of(&metric, NEEDLE_BLOCK),
        mid_layer,
        partial_mid: partial_spearman(&metric_mid, &oracle, &recency),
        needle_rank_mid: rank_of(&metric_mid, NEEDLE_BLOCK),
        sp_recency: spearman(&recency, &oracle),
        needle_rank_recency: rank_of(&recency, NEEDLE_BLOCK),
        needle_rank_oracle: rank_of(&oracle, NEEDLE_BLOCK),
        n_ctx,
        retrieved_prob,
        retrieved_argmax,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Optional argv filter: run ONE model per process (frees all VRAM on exit,
    // sidestepping the non-releasing device allocator). e.g. `... -- Llama-3.2-3B`.
    let filter = std::env::args().nth(1);
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!(
        "arch: {}  SEQ={SEQ} BLOCK={BLOCK} needle@block{NEEDLE_BLOCK} shallow=L{SHALLOW}\n",
        gpu.arch
    );

    for (label, repo) in LADDER {
        if let Some(f) = &filter {
            if f != label {
                continue;
            }
        }
        let dir = match snapshot_dir(repo) {
            Some(d) => d,
            None => {
                println!("[skip] {label}: no snapshot under {repo}");
                continue;
            }
        };
        println!("── {label} ({}) ──", dir.display());
        match run_model(&mut gpu, label, &dir) {
            Ok(r) => {
                println!(
                    "  layers={} mid=L{} ctx_blocks={}  retrieved: argmax={} p(answer)={:.4}",
                    r.layers, r.mid_layer, r.n_ctx, r.retrieved_argmax, r.retrieved_prob
                );
                println!(
                    "  needle block rank (1=best) — oracle:{}/{n}  recency:{}/{n}  shallowK:{}/{n}  midK:{}/{n}",
                    r.needle_rank_oracle, r.needle_rank_recency, r.needle_rank_metric, r.needle_rank_mid, n = r.n_ctx
                );
                println!(
                    "  corr(K,oracle) — shallow:{:+.3}  partial(shallow|recency):{:+.3}  mid partial:{:+.3}  sp(recency,oracle)={:+.3}",
                    r.sp_metric, r.partial, r.partial_mid, r.sp_recency
                );
                // single machine-greppable summary line for the bash loop to collect
                println!(
                    "ROW\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}\t{:.4}",
                    r.label,
                    r.layers,
                    r.mid_layer,
                    r.sp_metric,
                    r.partial,
                    r.partial_mid,
                    r.needle_rank_oracle,
                    r.needle_rank_recency,
                    r.needle_rank_metric,
                    r.needle_rank_mid,
                    r.retrieved_prob
                );
                println!();
            }
            Err(e) => println!("  [error] {e}\n"),
        }
    }
    Ok(())
}
