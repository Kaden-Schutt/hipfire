// SPDX-License-Identifier: Apache-2.0
// hipfire — Tier-1 native single-load artifact collector (thin CLI).
//
//! Loads a bf16 `.hfq` once and runs the matching arch collector, writing a
//! unified `<model>.calib.hfq` bundling the per-tensor Hessian + imatrix (+
//! MoE router histogram for MoE models, + KLDREF with `--kldref`). Gemma3-VL
//! (`arch_id=13`) is collected text-only through the `language_model.` prefix.
//!
//! Run:
//!   cargo run --release -p hipfire-runtime --example collect_artifacts -- \
//!     --model ~/.hipfire/models/qwen3.5-0.8b-bf16.hfq \
//!     --corpus benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \
//!     --output /tmp/qwen3.5-0.8b.calib.hfq --max-tokens 256 [--kldref]

use hipfire_arch_gemma3::calibration as gemma3_calib;
use hipfire_arch_gemma3::weights as gemma3_weights;
use hipfire_arch_gemma3::{self as gemma3};
use hipfire_arch_qwen35::qwen35::{self, CalibOpts as QwenCalibOpts};
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
    let source_arch_id = hfq.arch_id;
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

    // Provenance keys (caller-known) layered onto the driver's technical metadata.
    let provenance = [
        ("source_model", serde_json::json!(model)),
        ("corpus", serde_json::json!(corpus)),
        ("n_calib_tokens", serde_json::json!(n_tok)),
        ("source_arch_id", serde_json::json!(source_arch_id)),
    ];
    let t0 = std::time::Instant::now();
    let (n_hessian, n_imatrix, max_consistency, mode) = match source_arch_id {
        5 | 6 => {
            let config = qwen35::config_from_hfq(&hfq).expect("qwen35 config");
            let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");
            let opts = QwenCalibOpts {
                kldref: want_kldref,
                kldref_topk: 64,
            };
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
            (
                summary.n_hessian,
                summary.n_imatrix,
                summary.max_consistency,
                "qwen35",
            )
        }
        12 | 13 => {
            let prefix = if source_arch_id == 13 {
                "language_model."
            } else {
                ""
            };
            let config = gemma3::config_from_hfq(&hfq).expect("gemma3 config");
            let weights =
                gemma3_weights::load_weights_prefixed(&mut hfq, &config, &mut gpu, prefix)
                    .expect("load_weights");
            let opts = gemma3_calib::CalibOpts {
                kldref: want_kldref,
                kldref_topk: 64,
            };
            let summary = gemma3_calib::collect_calibration_artifacts_text_only(
                &mut gpu,
                &weights,
                &config,
                &tokenizer,
                tokens,
                &opts,
                Path::new(&output),
                prefix,
                &provenance,
            )
            .expect("collect");
            (
                summary.n_hessian,
                summary.n_imatrix,
                summary.max_consistency,
                if source_arch_id == 13 {
                    "gemma3-vl-text-only"
                } else {
                    "gemma3-text"
                },
            )
        }
        other => panic!("collect_artifacts: unsupported arch_id {other}; handled 5/6/12/13"),
    };
    eprintln!(
        "collected {n_hessian} hessian + {n_imatrix} imatrix tensors in {:.1}s; mode={mode}; max diag(H)-vs-Σx² rel-err = {:.3e} {}",
        t0.elapsed().as_secs_f64(),
        max_consistency,
        if max_consistency < 1e-4 {
            "[CONSISTENT]"
        } else {
            "[MISMATCH]"
        }
    );
    eprintln!("wrote calib HFQ: {output}");
    if max_consistency >= 1e-4 {
        std::process::exit(1);
    }
}
