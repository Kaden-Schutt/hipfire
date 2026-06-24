// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! FU4 loader-compat validation: load Nano-4B both as f32 (safetensors) and as
//! quantized HFQ (`NemotronModel::from_hfq`), run the **same** token sequence
//! through both, and compare final logits. The HFQ forward should track the f32
//! forward within quantization error (mq4/hfq4/q8) — same argmax, high cosine,
//! small relative error. This is the only correctness check available while
//! coherent generation is blocked on the missing CUDA-kernel HF reference.
//!
//!   hipfire lock acquire hfq_vs_f32 --watch-pid $$
//!   NANO4B_DIR=<snap> cargo run --release -p hipfire-arch-nemotron \
//!       --example hfq_vs_f32 -- /tmp/nano4b-mq4.hfq
//!
//! Loads the two models sequentially (free f32 before loading HFQ) to keep peak
//! memory down on the dev APU.

use hipfire_arch_nemotron::loader::load_nemotron_weights;
use hipfire_arch_nemotron::model::NemotronModel;
use hipfire_arch_nemotron::NemotronHConfig;
use hipfire_model::ModelSource;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};

const DEFAULT_DIR: &str = "/srv/huggingface/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f";
const DEFAULT_HFQ: &str = "/tmp/nano4b-mq4.hfq";
// "The capital of France is" (matches the bisect/HF dump). Override via
// NEMO_TOKENS=comma,separated,ids.
const TOKENS: [u32; 5] = [1784, 8961, 1307, 5498, 1395];

fn tokens() -> Vec<u32> {
    match std::env::var("NEMO_TOKENS") {
        Ok(s) => s.split(',').map(|x| x.trim().parse().unwrap()).collect(),
        Err(_) => TOKENS.to_vec(),
    }
}

/// Run the prompt (state built over `toks[..last]`) and return the final-position
/// per-block residual captures + `[vocab]` logits.
fn final_capture(
    model: &mut NemotronModel,
    gpu: &mut Gpu,
    toks: &[u32],
) -> (Vec<Vec<f32>>, Vec<f32>) {
    for (pos, &t) in toks.iter().enumerate().take(toks.len() - 1) {
        model.forward_gpu(gpu, t, pos).unwrap();
    }
    let last = toks.len() - 1;
    model.forward_capture(gpu, toks[last], last).unwrap()
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        d += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    d / (na.sqrt() * nb.sqrt())
}

fn argmax(v: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..v.len() {
        if v[i] > v[bi] {
            bi = i;
        }
    }
    bi
}

fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
    idx.truncate(k);
    idx
}

fn main() {
    let hfq_path = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| DEFAULT_HFQ.to_string()),
    );
    let dir =
        PathBuf::from(std::env::var("NANO4B_DIR").unwrap_or_else(|_| DEFAULT_DIR.to_string()));
    if !dir.join("config.json").exists() {
        eprintln!("SKIP: checkpoint not found at {}", dir.display());
        return;
    }
    if !hfq_path.exists() {
        eprintln!("SKIP: hfq not found at {}", hfq_path.display());
        return;
    }

    let cfg_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let cfg = NemotronHConfig::from_json(&cfg_json).unwrap();
    let toks = tokens();
    let max_seq = (toks.len() + 4).max(16);

    let mut gpu = Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);

    // ── f32 reference ────────────────────────────────────────────────────────
    eprintln!("loading f32 (safetensors)…");
    let src = SafetensorsSource::open(&dir).unwrap();
    assert_eq!(src.arch_id(), 14);
    let weights = load_nemotron_weights(&src, &cfg).unwrap();
    let mut f32_model = NemotronModel::new(&mut gpu, cfg.clone(), &weights, max_seq).unwrap();
    let (f32_caps, f32_logits) = final_capture(&mut f32_model, &mut gpu, &toks);
    f32_model.free(&mut gpu);
    drop(weights);
    eprintln!("f32 done; freed");

    // ── HFQ (quantized) ──────────────────────────────────────────────────────
    eprintln!("loading hfq {}…", hfq_path.display());
    let hfq = HfqFile::open(Path::new(&hfq_path)).unwrap();
    let mut hfq_model = NemotronModel::from_hfq(&mut gpu, &hfq, cfg.clone(), max_seq).unwrap();
    let (hfq_caps, hfq_logits) = final_capture(&mut hfq_model, &mut gpu, &toks);
    hfq_model.free(&mut gpu);
    eprintln!("hfq done");

    // ── per-layer divergence (caps[0]=embedding, caps[l+1]=after block l) ─────
    eprintln!("per-block cosine (f32 vs hfq):");
    for (i, (a, b)) in f32_caps.iter().zip(hfq_caps.iter()).enumerate() {
        let label = if i == 0 {
            "embed".to_string()
        } else {
            format!("blk{:>2} {:?}", i - 1, cfg.blocks[i - 1])
        };
        let c = cos(a, b);
        let mark = if c < 0.98 { "  <-- DIVERGE" } else { "" };
        eprintln!("  cap{i:>2} {label:<16} cos={c:.5}{mark}");
    }

    // ── compare ──────────────────────────────────────────────────────────────
    assert_eq!(f32_logits.len(), hfq_logits.len());
    let n = f32_logits.len();
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for i in 0..n {
        let (a, b) = (f32_logits[i] as f64, hfq_logits[i] as f64);
        dot += a * b;
        na += a * a;
        nb += b * b;
        let d = (f32_logits[i] - hfq_logits[i]).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
    }
    let cosine = dot / (na.sqrt() * nb.sqrt());
    let mean_abs = sum_abs / n as f64;

    let a_f32 = argmax(&f32_logits);
    let a_hfq = argmax(&hfq_logits);
    let t5_f32 = top_k(&f32_logits, 5);
    let t5_hfq = top_k(&hfq_logits, 5);
    let overlap = t5_f32.iter().filter(|x| t5_hfq.contains(x)).count();

    eprintln!("tokens: {toks:?}");
    eprintln!(
        "argmax  f32={a_f32}  hfq={a_hfq}  {}",
        if a_f32 == a_hfq { "MATCH" } else { "DIFFER" }
    );
    eprintln!("top5    f32={t5_f32:?}");
    eprintln!("        hfq={t5_hfq:?}  overlap={overlap}/5");
    eprintln!("cosine  {cosine:.6}");
    eprintln!("mean|Δ| {mean_abs:.5}   max|Δ| {max_abs:.5}");

    // Quantized 4-bit weights against an f32 forward: argmax should agree and the
    // logit direction should be near-identical. Cosine ≥ 0.99 + argmax match is a
    // solid "the quantized path is wired correctly" bar (a mis-loaded weight,
    // wrong rotation, or skipped out_proj rescale tanks cosine far below this).
    // Over 42 layers of mixed Q8 (in/up/q/k/v/lm_head/embed) + MQ4G256
    // (out/down) the residual stream accrues ~1% cosine error — argmax agreement
    // plus cosine ≥ 0.98 is the "wired correctly" bar (a mis-loaded weight, wrong
    // rotation, or skipped out_proj rescale collapses cosine to ~0).
    if a_f32 == a_hfq && cosine >= 0.98 {
        println!("PASS: hfq forward tracks f32 (argmax match, cosine {cosine:.4})");
    } else {
        println!("FAIL: hfq diverges from f32 (argmax {a_f32}/{a_hfq}, cosine {cosine:.4})");
        std::process::exit(1);
    }
}
