// SPDX-License-Identifier: Apache-2.0
// hipfire — Tier-1 native single-load artifact collector (thin CLI).
//
//! Loads a bf16 `.hfq` once and runs `qwen35::collect_calibration_artifacts`
//! (the lib-ified driver), writing a unified `<model>.calib.hfq` bundling the
//! per-tensor Hessian + imatrix (+ MoE router histogram for MoE models, +
//! KLDREF with `--kldref`). All collection logic lives in the engine + the
//! hipfire_runtime::calibration lib; this is just argv + load + write, so the
//! daemon `Collect` op reuses the exact same driver.
//!
//! Run:
//!   cargo run --release -p hipfire-runtime --example collect_artifacts -- \
//!     --model ~/.hipfire/models/qwen3.5-0.8b-bf16.hfq \
//!     --corpus benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \
//!     --output /tmp/qwen3.5-0.8b.calib.hfq --max-tokens 256 [--kldref]

use hipfire_arch_qwen35::qwen35::{self, CalibOpts};
use rdna_compute::Gpu;
use std::path::Path;

fn arg(flag: &str, default: Option<String>) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == flag)
        .and_then(|i| a.get(i + 1).cloned())
        .or(default)
}

fn main() {
    let model = arg("--model", None).expect("--model required");
    let corpus = arg("--corpus", None).expect("--corpus required");
    let output = arg("--output", Some("/tmp/native.calib.hfq".into())).unwrap();
    let max_tokens: usize = arg("--max-tokens", Some("512".into()))
        .unwrap()
        .parse()
        .unwrap();
    let want_kldref = std::env::args().any(|a| a == "--kldref");

    let mut hfq = hipfire_runtime::hfq::HfqFile::open(Path::new(&model)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let tokenizer =
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tok");

    let raw = std::fs::read(&corpus).expect("read corpus");
    let take = (max_tokens * 8).min(raw.len());
    let text = String::from_utf8_lossy(&raw[..take]).to_string();
    let all: Vec<u32> = tokenizer.encode(&text);
    let n_tok = all.len().min(max_tokens);
    let tokens = &all[..n_tok];
    eprintln!("calibrating on {n_tok} tokens (kldref={want_kldref})");

    let mut gpu = Gpu::init().expect("gpu");
    eprintln!("GPU: {}", gpu.arch);
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");

    let opts = CalibOpts {
        kldref: want_kldref,
        kldref_topk: 64,
    };
    // Provenance keys (caller-known) layered onto the driver's technical metadata.
    let provenance = [
        ("source_model", serde_json::json!(model)),
        ("corpus", serde_json::json!(corpus)),
        ("n_calib_tokens", serde_json::json!(n_tok)),
    ];
    let t0 = std::time::Instant::now();
    // Streams the package to `output` one tensor at a time (no full-RAM
    // materialization), returning only a summary.
    let summary = qwen35::collect_calibration_artifacts(
        &mut gpu,
        &weights,
        &config,
        tokens,
        &opts,
        Path::new(&output),
        &provenance,
    )
    .expect("collect");
    eprintln!(
        "collected {} hessian + {} imatrix tensors in {:.1}s; max diag(H)-vs-Σx² rel-err = {:.3e} {}",
        summary.n_hessian,
        summary.n_imatrix,
        t0.elapsed().as_secs_f64(),
        summary.max_consistency,
        if summary.max_consistency < 1e-4 {
            "[CONSISTENT]"
        } else {
            "[MISMATCH]"
        }
    );
    eprintln!("wrote calib HFQ: {output}");
    if summary.max_consistency >= 1e-4 {
        std::process::exit(1);
    }
}
