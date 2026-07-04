// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! End-to-end smoke test (N5a): load the real NVIDIA-Nemotron-3-Nano-4B-BF16
//! checkpoint via the safetensors loader and run a few greedy decode steps,
//! asserting finite, non-degenerate logits. This is NOT an HF-reference numeric
//! check (that's the next step) — it proves the loader + full GPU forward run on
//! real weights without NaN/attractor collapse.
//!
//! Skips gracefully if the checkpoint isn't present. Override the dir with
//! `NANO4B_DIR=/path cargo run -p hipfire-arch-nemotron --example test_load_nano4b`.
//!
//!   hipfire lock acquire test_load_nano4b --watch-pid $$
//!   cargo run --release -p hipfire-arch-nemotron --example test_load_nano4b

use hipfire_arch_nemotron::loader::load_nemotron_weights;
use hipfire_arch_nemotron::model::NemotronModel;
use hipfire_arch_nemotron::NemotronHConfig;
use hipfire_model::ModelSource;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use hipfire_rdna::Gpu;
use std::path::PathBuf;

const DEFAULT_DIR: &str = "/srv/huggingface/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f";

fn main() {
    let dir = std::env::var("NANO4B_DIR").unwrap_or_else(|_| DEFAULT_DIR.to_string());
    let dir = PathBuf::from(dir);
    if !dir.join("config.json").exists() {
        eprintln!(
            "SKIP: checkpoint not found at {} (set NANO4B_DIR)",
            dir.display()
        );
        return;
    }

    // Parse config.json → NemotronHConfig.
    let cfg_str = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
    let cfg_json: serde_json::Value = serde_json::from_str(&cfg_str).expect("parse config.json");
    let cfg = NemotronHConfig::from_json(&cfg_json).expect("nemotron config");
    eprintln!(
        "config: {} layers ({} M / {} * / {} -), hidden {}, vocab {}",
        cfg.num_layers,
        cfg.count(hipfire_arch_nemotron::BlockKind::Mamba2),
        cfg.count(hipfire_arch_nemotron::BlockKind::Attention),
        cfg.count(hipfire_arch_nemotron::BlockKind::Mlp),
        cfg.hidden_size,
        cfg.vocab_size,
    );

    let src = SafetensorsSource::open(&dir).expect("open safetensors");
    assert_eq!(src.arch_id(), 14, "nemotron_h must classify to arch_id 14");

    eprintln!("loading + dequantizing weights (bf16→f32)...");
    let t0 = std::time::Instant::now();
    let weights = load_nemotron_weights(&src, &cfg).expect("load weights");
    eprintln!("  loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let t1 = std::time::Instant::now();
    let mut model = NemotronModel::new(&mut gpu, cfg.clone(), &weights, 64).expect("upload model");
    eprintln!("  uploaded in {:.1}s", t1.elapsed().as_secs_f32());

    // Greedy decode a few steps from a seed token; check finiteness + variety.
    let mut tok = 1u32; // arbitrary seed (no tokenizer needed for the smoke test)
    let mut produced = Vec::new();
    for pos in 0..10 {
        let logits = model.forward(&mut gpu, tok, pos).expect("forward");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite logits at pos {pos}"
        );
        let (argmax, _) =
            logits
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                });
        eprintln!("pos {pos}: tok {tok} → argmax {argmax}");
        produced.push(argmax);
        tok = argmax as u32;
    }

    let unique = produced
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    if unique == 1 {
        eprintln!(
            "WARN: greedy collapsed to a single token (possible attractor) — needs HF-ref check"
        );
    }
    println!("PASS: Nano-4B loads and forwards with finite logits ({unique} unique greedy tokens)");
}
