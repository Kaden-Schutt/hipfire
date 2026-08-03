// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU parity test for Task 2: per-extract-layer residual-hidden capture in the
//! `LlamaBundle` `SpecTarget` path (DFlash drafter conditioning).
//!
//! Loads `bielik-minitron-7b.mq4` (40-layer dense LLaMA, hidden=4096), sets
//! `dflash_extract_layers = [1, k/2, k-1]` = `[1, 20, 39]`, prefills a short
//! token sequence
//! through `SpecTarget::spec_advance` with a capture sink, and asserts:
//!   - captured row count == prompt token count
//!   - row width == `num_extract * hidden` (3 * 4096 = 12288)
//!   - bit-identical on a second (reset) run
//!
//! Usage:
//!   cargo run --release --example llama_dflash_hidden_capture
//!   cargo run --release --example llama_dflash_hidden_capture -- <model.mq4>

use hipfire_arch_llama::load_llama_bundle;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
use hipfire_runtime::spec::{SpecAdvance, SpecTarget};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let default_path = format!(
        "{}/.hipfire/models/bielik-minitron-7b.mq4",
        std::env::var("HOME").unwrap_or_else(|_| ".".into())
    );
    let model_path = args.get(1).cloned().unwrap_or(default_path);

    eprintln!("[task2] opening {model_path}");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");

    let src = ModelSource::from_path(&model_path).expect("open model source");
    let cask = CaskConfig::default();
    let mut ctx = LoadCtx {
        path: &model_path,
        max_seq: 2048,
        draft_path: None,
        kv_mode_override: None,
        kv_backend: hipfire_runtime::kv_backend::KvBackend::Contiguous,
        kv_adaptive_override: None,
        state_quant_override: None,
        cask: &cask,
        pp: 1,
        spec: SpecLoadCfg::default(),
        gpu: &mut gpu,
    };

    let mut bundle = load_llama_bundle(src, &mut ctx).expect("load llama bundle");
    let hidden = bundle.config.dim;
    let n_layers = bundle.config.n_layers;
    eprintln!("[task2] loaded: hidden={hidden}, n_layers={n_layers}");

    // 3 of Bielik's 40 layers, ascending (the brief's [1, k/2, k-1] pattern).
    let extract: Vec<usize> = vec![1, n_layers / 2, n_layers - 1];
    let num_extract = extract.len();
    bundle.set_dflash_extract_layers(extract.clone());
    assert_eq!(
        bundle.dflash_extract_layers(),
        Some(extract.as_slice()),
        "dflash_extract_layers should reflect the configured set"
    );

    // A short prompt of fixed token ids (>= MIN_BATCH=4 so the batched path is
    // eligible). Real token text is irrelevant to the capture shape; we use raw
    // ids well inside the vocab.
    let prompt: Vec<u32> = vec![1u32, 2, 415, 920, 31, 678, 12, 99];
    let n_pos = prompt.len();
    let expected_width = num_extract * hidden;
    eprintln!(
        "[task2] prompt_len={n_pos}, extract={extract:?}, expected row width={expected_width}"
    );

    let run =
        |bundle: &mut hipfire_arch_llama::LlamaBundle, gpu: &mut rdna_compute::Gpu| -> Vec<f32> {
            let mut hidden_out: Vec<f32> = Vec::new();
            let no_abort = || false;
            let adv = bundle
                .spec_advance(gpu, &prompt, 0, true, &no_abort, Some(&mut hidden_out))
                .expect("spec_advance");
            match adv {
                SpecAdvance::Ready { last_argmax, .. } => {
                    eprintln!("[task2]   spec_advance Ready, last_argmax={last_argmax}");
                }
                other => panic!("expected SpecAdvance::Ready, got {other:?}"),
            }
            hidden_out
        };

    let cap1 = run(&mut bundle, &mut gpu);
    let cap2 = run(&mut bundle, &mut gpu);

    let mut ok = true;

    // 1. Row count == prompt token count, total len == n_pos * num_extract * hidden.
    let total_expected = n_pos * expected_width;
    if cap1.len() != total_expected {
        eprintln!(
            "[task2] FAIL: captured {} floats, expected {} ({}×{}×{})",
            cap1.len(),
            total_expected,
            n_pos,
            num_extract,
            hidden
        );
        ok = false;
    } else {
        let rows = cap1.len() / expected_width;
        eprintln!(
            "[task2] OK: row_count={rows} (== prompt_len {n_pos}), row_width={expected_width}"
        );
    }

    // 2. Not all zero (capture actually ran).
    if cap1.iter().all(|&v| v == 0.0) {
        eprintln!("[task2] FAIL: all captured values are zero (capture did not run)");
        ok = false;
    }
    if cap1.iter().any(|v| !v.is_finite()) {
        eprintln!("[task2] FAIL: captured values contain non-finite floats");
        ok = false;
    }

    // 3. Bit-identical on re-run.
    if cap1 != cap2 {
        let diffs = cap1.iter().zip(cap2.iter()).filter(|(a, b)| a != b).count();
        eprintln!(
            "[task2] FAIL: re-run not bit-identical ({diffs}/{} floats differ)",
            cap1.len()
        );
        ok = false;
    } else {
        eprintln!("[task2] OK: re-run bit-identical ({} floats)", cap1.len());
    }

    // Process exits immediately after; GPU teardown handled by driver.
    let _ = &bundle;

    if ok {
        println!("PASS");
    } else {
        println!("FAIL");
        std::process::exit(1);
    }
}
