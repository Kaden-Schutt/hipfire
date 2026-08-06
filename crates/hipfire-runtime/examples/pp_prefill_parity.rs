// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! P-C PC-4: **dense pipeline-parallel BATCHED-PREFILL parity** — run a fixed
//! prompt through `PpModel::prefill` (one batched forward per stage, banded over
//! layers, `[n×dim]` residual `boundary_copy` across each seam) and assert the
//! last-position logits match single-GPU `llama::prefill_forward` (the same
//! banded kernels, whole-stack). Both sides use mq4 weights + Q8 KV; only the
//! byte-exact F32 residual boundary-copy differs, so expect argmax-identical and
//! `max|Δ| ~ 0..1e-4` (cf. PC-1 `pp_full_model_parity` which got max|Δ|=0).
//!
//! Emulated Pp-2 (gfx1151).
//!
//! Run: HIPFIRE_EMULATE_GPUS=2 HIPFIRE_DETERMINISTIC=1 \
//!   cargo run -p hipfire-runtime --release --example pp_prefill_parity -- --model model.mq4

use hipfire_hardware::{DeviceMesh, DimKind};
use hipfire_loader::model_parallel::PipelineImpl;
use hipfire_loader::ModelParallel;
use hipfire_runtime::llama::{self, KvCache, LlamaConfig};

const MAX_SEQ: usize = 512;

// Fixed prompt (≤256 tokens after tokenization). md5(PROMPT) = a7fad5c2aca9f03cd751904c3fe606f8
// (recorded so any whitespace edit that would change tokenization is caught in review).
const PROMPT: &str = "The pipeline-parallel prefill splits the transformer layers across \
two stages. Stage zero embeds every prompt token in one batched forward and runs its layer \
band, writing the Q8 KV cache for all positions; the residual hidden state is then copied \
across the device boundary to stage one, which runs the remaining layers. Explain, in a few \
sentences, why the last-position logits after this banded batched prefill must be identical \
to running the whole model on a single device.";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = concat!(env!("HOME"), "/.hipfire/models/qwen3-0.6b-llama.mq4").to_string();
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => {
                if let Some(v) = it.next() {
                    model_path = v.clone();
                }
            }
            other => model_path = other.to_string(),
        }
    }
    let pp = 2usize;

    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");

    let hfq =
        hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(&model_path)).expect("open model");
    let config: LlamaConfig = hipfire_runtime::hfq::config_from_hfq(&hfq).expect("config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let toks = tokenizer.encode(PROMPT);
    assert!(!toks.is_empty(), "empty prompt");
    assert!(
        toks.len() <= llama::PREFILL_MAX_BATCH,
        "prompt {} toks > PREFILL_MAX_BATCH {} — pick a shorter fixed prompt",
        toks.len(),
        llama::PREFILL_MAX_BATCH
    );
    eprintln!(
        "model: layers={} | prompt={} toks, pp={pp} (batched prefill)",
        config.n_layers,
        toks.len()
    );

    // ── Reference: single-GPU batched prefill (last-position logits). Scoped so
    // its Gpu drops before PpModel brings up the emulated Gpus. ──
    let ref_logits: Vec<f32> = {
        let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
        gpu.bind_thread().unwrap();
        let weights =
            hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, &mut gpu).expect("load_weights");
        let mut kv = KvCache::new_gpu_q8(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            MAX_SEQ,
        )
        .unwrap();
        llama::prefill_forward(&mut gpu, &weights, &config, &toks, &mut kv).expect("ref prefill")
    };

    // ── PP path: PpModel batched prefill (via loader). ──
    let mesh = DeviceMesh::rect(&[(DimKind::Pp, pp)]);
    let loaded =
        match hipfire_loader::load_model_pp(&model_path, MAX_SEQ, &mesh, Default::default()) {
            Ok(m) => m,
            Err(e) => {
                println!("pp_prefill_parity: SKIPPED (load_model_pp: {e})");
                return;
            }
        };
        let mut model = loaded
        .pp_model
        .expect("expected dense PP model (pp_model carrier)");
    model.prefill(&toks).expect("pp prefill");
    let pp_logits = model.logits().expect("pp logits");

    assert_eq!(
        ref_logits.len(),
        pp_logits.len(),
        "logits length mismatch: ref={} pp={}",
        ref_logits.len(),
        pp_logits.len()
    );
    let ref_argmax = llama::argmax(&ref_logits);
    let pp_argmax = llama::argmax(&pp_logits);
    let max_delta = ref_logits
        .iter()
        .zip(&pp_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "[pp-prefill] toks={} ref_argmax={ref_argmax} pp_argmax={pp_argmax} max|Δ|={max_delta:.3e}",
        toks.len()
    );
    eprintln!("ref next-token: {:?}", tokenizer.decode(&[ref_argmax]));
    eprintln!(" pp next-token: {:?}", tokenizer.decode(&[pp_argmax]));

    assert_eq!(
        ref_argmax, pp_argmax,
        "PP batched prefill argmax diverged from single-GPU prefill_forward: \
         pp={pp_argmax} ref={ref_argmax} (max|Δ|={max_delta:.3e})"
    );
    assert!(
        max_delta < 1e-3,
        "PP batched prefill max|Δ|={max_delta:.3e} >= 1e-3 (expected byte-exact residual copy)"
    );
    println!(
        "pp_prefill_parity: dense PP batched prefill last-position logits == single-GPU \
         prefill_forward (argmax-exact, max|Δ|={max_delta:.3e}) — PC-4 validated"
    );
}
