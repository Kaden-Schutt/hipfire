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
use hipfire_arch_lfm2moe::calibration as lfm2_calib;
use hipfire_arch_lfm2moe::{Lfm2MoeConfig, Lfm2MoeWeights};
use hipfire_arch_minimax::calibration as minimax_calib;
use hipfire_arch_minimax::{MiniMaxConfig, MiniMaxWeights};
use hipfire_arch_nemotron::{calibration as nemotron_calib, model::NemotronModel, NemotronHConfig};
use hipfire_arch_qwen35::qwen35::{self, CalibOpts as QwenCalibOpts};
use hipfire_arch_zaya::{calibration as zaya_calib, ZayaConfig};
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
    // Optional under `--synthetic-tokens` (no corpus is read in that mode).
    let corpus = arg("--corpus", Some(String::new())).unwrap();
    let output = arg("--output", Some("/tmp/native.calib.hfq".into())).unwrap();
    let max_tokens: usize = arg("--max-tokens", Some("512".into()))
        .unwrap()
        .parse()
        .unwrap();
    let want_kldref = std::env::args().any(|a| a == "--kldref");
    // Tiny seeded fixtures (`hipfire-quantize --emit-fixture <family>`) carry a
    // synthetic tokenizer with no real `model`, so the corpus-encode path can't
    // run. `--synthetic-tokens` skips the tokenizer and feeds seeded random ids
    // in `[0, vocab)` — enough to exercise the capturing forward + streamed
    // collector for pipeline validation. Real models still use `--corpus`.
    let synthetic = std::env::args().any(|a| a == "--synthetic-tokens");
    let seed: u64 = arg("--seed", Some("0".into())).unwrap().parse().unwrap();

    let mut hfq = hipfire_runtime::hfq::HfqFile::open(Path::new(&model)).expect("open model");
    // `--arch <id>` overrides the hfq's stored arch_id. Needed for hfqs that
    // predate proper arch tagging (e.g. some qwen3 MoE bf16 hfqs are stamped
    // arch_id=0/llama but load fine through the qwen35 backend at 5/6).
    let source_arch_id = arg("--arch", None)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(hfq.arch_id);

    // Loaded lazily — synthetic mode has no usable tokenizer; only the gemma3
    // text-only arm actually consumes it (asserts Some below).
    let tokenizer: Option<hipfire_runtime::tokenizer::Tokenizer> = if synthetic {
        None
    } else {
        Some(
            hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .expect("tok"),
        )
    };

    let tokens_owned: Vec<u32> = if synthetic {
        // Parse vocab_size from the hfq metadata (flat or under `config`).
        let meta: serde_json::Value =
            serde_json::from_str(&hfq.metadata_json).expect("metadata json");
        let vocab = meta
            .get("vocab_size")
            .or_else(|| meta.get("config").and_then(|c| c.get("vocab_size")))
            .and_then(|v| v.as_u64())
            .expect("vocab_size in metadata") as u32;
        // xorshift64 — matches the tiny-gate token generator's intent.
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15) | 1;
        (0..max_tokens)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % vocab as u64) as u32
            })
            .collect()
    } else {
        let raw = std::fs::read(&corpus).expect("read corpus");
        let take = (max_tokens * 8).min(raw.len());
        let text = String::from_utf8_lossy(&raw[..take]).to_string();
        tokenizer.as_ref().unwrap().encode(&text)
    };
    let n_tok = tokens_owned.len().min(max_tokens);
    let tokens = &tokens_owned[..n_tok];
    eprintln!("calibrating on {n_tok} tokens (kldref={want_kldref}, synthetic={synthetic})");

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
                tokenizer
                    .as_ref()
                    .expect("gemma3 text-only collect needs a tokenizer (not --synthetic-tokens)"),
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
        11 => {
            let config = Lfm2MoeConfig::from_hfq(&hfq).expect("lfm2 config");
            let weights = Lfm2MoeWeights::load(&mut hfq, &config, &mut gpu).expect("lfm2 weights");
            let opts = lfm2_calib::CalibOpts {
                kldref: want_kldref,
                kldref_topk: 64,
            };
            let summary = lfm2_calib::collect_calibration_artifacts(
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
                "lfm2-text",
            )
        }
        16 => {
            let meta: serde_json::Value =
                serde_json::from_str(&hfq.metadata_json).expect("zaya metadata");
            let cfg_json = meta.get("config").unwrap_or(&meta);
            let config = ZayaConfig::from_json(cfg_json).expect("zaya config");
            let weights = hipfire_arch_zaya::gpu::ZayaGpuWeights::load(&hfq, &mut gpu, &config)
                .expect("zaya weights");
            let opts = zaya_calib::CalibOpts {
                kldref: want_kldref,
                kldref_topk: 64,
            };
            let summary = zaya_calib::collect_calibration_artifacts(
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
                "zaya",
            )
        }
        10 => {
            // MiniMax-M2 (MoE: GQA attention + indexed-expert MoE). Attention
            // q/k/v/o, the MoE router, and lm_head route through `weight_gemv`
            // and get full Hessians; routed experts are not captured yet
            // (indexed MoE kernels need explicit taps — documented follow-on).
            let config = MiniMaxConfig::from_hfq(&hfq).expect("minimax config");
            let weights =
                MiniMaxWeights::load(&mut hfq, &config, &mut gpu, None).expect("minimax weights");
            let opts = minimax_calib::CalibOpts {
                kldref: want_kldref,
                kldref_topk: 64,
            };
            let summary = minimax_calib::collect_calibration_artifacts(
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
                "minimax",
            )
        }
        14 => {
            // Dense nemotron_h (Nano-4B). Config lives in the hfq metadata's
            // `config` key (same as serving). MoE Nano-30B experts are
            // imatrix-only — a follow-on (build_capture_names skips them).
            let meta: serde_json::Value =
                serde_json::from_str(&hfq.metadata_json).expect("nemotron metadata parse");
            let cfg_json = meta
                .get("config")
                .expect("nemotron metadata_json missing 'config'");
            let config = NemotronHConfig::from_json(cfg_json).expect("nemotron config");
            let mut model = NemotronModel::from_hfq(&mut gpu, &hfq, config, n_tok + 16)
                .expect("nemotron from_hfq");
            let opts = nemotron_calib::CalibOpts {
                kldref: want_kldref,
                kldref_topk: 64,
            };
            let summary = nemotron_calib::collect_calibration_artifacts(
                &mut gpu,
                &mut model,
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
                "nemotron-h",
            )
        }
        other => {
            panic!("collect_artifacts: unsupported arch_id {other}; handled 5/6/10/11/12/13/14/16")
        }
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
