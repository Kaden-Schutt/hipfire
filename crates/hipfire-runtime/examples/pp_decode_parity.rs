// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! P-C PC-2: **dense pipeline-parallel DECODE parity** — prefill a prompt and
//! greedy-decode N tokens pipeline-parallel (Pp-2) through `PpModel`, and assert
//! the token stream is IDENTICAL to single-GPU `llama::forward_scratch`. The PP
//! analog of `tp_decode_parity`: where `pp_full_model_parity` proved a single
//! position, this proves the real generation loop — prefill + multi-token decode
//! with a per-stage KV growing across positions (each stage's band writes its own
//! KV; the residual is `boundary_copy`'d across the seam per token).
//!
//! PP is exact, so the argmax matches at every step (max|Δ|=0 per token → identical
//! greedy stream). Per-token prefill (no batched prefill), matching the TP arc.
//! Emulated Pp-2 (gfx1151).
//!
//! Run: HIP_VISIBLE_DEVICES=0 HIPFIRE_DETERMINISTIC=1 \
//!   cargo run -p hipfire-runtime --release --example pp_decode_parity [model.mq4] [steps] [prompt]

use hipfire_hardware::{DeviceMesh, DimKind};
use hipfire_loader::model_parallel::PipelineImpl;
use hipfire_loader::ModelParallel;
use hipfire_runtime::llama::{self, ForwardScratch, KvCache, LlamaConfig};

const MAX_SEQ: usize = 512;

fn fnv1a(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &id in ids {
        for b in id.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(String::as_str).unwrap_or(concat!(
        env!("HOME"),
        "/.hipfire/models/qwen3-0.6b-llama.mq4"
    ));
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);
    let prompt = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("The capital of France is");
    let pp = 2usize;

    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");

    let hfq =
        hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(model_path)).expect("open model");
    let config: LlamaConfig = hipfire_runtime::hfq::config_from_hfq(&hfq).expect("config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let prompt_tokens = tokenizer.encode(prompt);
    assert!(!prompt_tokens.is_empty(), "empty prompt");
    eprintln!(
        "model: layers={} | prompt={} toks, decode {steps} steps, pp={pp}",
        config.n_layers,
        prompt_tokens.len()
    );

    // ── Reference: single-GPU forward_scratch (prefill + greedy decode). Scoped
    // so its Gpu drops before PpModel brings up the emulated Gpus. ──
    let ref_ids: Vec<u32> = {
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
        let scratch = ForwardScratch::new_with_max_seq(&mut gpu, &config, MAX_SEQ).unwrap();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            llama::forward_scratch(
                &mut gpu, &weights, &config, tok, pos, &mut kv, &scratch, 0.0, 1.0, 0, 0, 1.0,
            )
            .expect("ref prefill");
        }
        let mut ids = Vec::with_capacity(steps + 1);
        let mut next = llama::argmax(&gpu.download_f32(&scratch.logits).unwrap());
        ids.push(next);
        for step in 0..steps {
            let pos = prompt_tokens.len() + step;
            llama::forward_scratch(
                &mut gpu, &weights, &config, next, pos, &mut kv, &scratch, 0.0, 1.0, 0, 0, 1.0,
            )
            .expect("ref decode");
            next = llama::argmax(&gpu.download_f32(&scratch.logits).unwrap());
            ids.push(next);
        }
        ids
    };

    // ── PP path: PpModel prefill + greedy decode. ──
    let mesh = DeviceMesh::rect(&[(DimKind::Pp, pp)]);
    let loaded = match hipfire_loader::load_model_pp(model_path, MAX_SEQ, &mesh, Default::default())
    {
        Ok(m) => m,
        Err(e) => {
            println!("pp_decode_parity: SKIPPED (load_model_pp: {e})");
            return;
        }
    };
        let mut model = loaded
        .pp_model
        .expect("expected dense PP model (pp_model carrier)");
    let eos = model.eos_token();
    let mut pp_ids = Vec::with_capacity(steps + 1);
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        model.forward_token(tok, pos).expect("pp prefill");
    }
    let mut next = llama::argmax(&model.logits().expect("pp logits"));
    pp_ids.push(next);
    for step in 0..steps {
        if next == eos {
            break;
        }
        let pos = prompt_tokens.len() + step;
        model.forward_token(next, pos).expect("pp decode");
        next = llama::argmax(&model.logits().expect("pp logits"));
        pp_ids.push(next);
    }

    // Trim the reference to the PP length if PP stopped early on eos.
    let ref_trunc = &ref_ids[..pp_ids.len().min(ref_ids.len())];
    let first_div = pp_ids.iter().zip(ref_trunc).position(|(a, b)| a != b);
    let ref_fnv = fnv1a(ref_trunc);
    let pp_fnv = fnv1a(&pp_ids[..ref_trunc.len().min(pp_ids.len())]);
    eprintln!("ref text: {:?}", tokenizer.decode(ref_trunc));
    eprintln!(" pp text: {:?}", tokenizer.decode(&pp_ids));
    println!(
        "[pp-decode] steps={} ref_fnv={ref_fnv:016x} pp_fnv={pp_fnv:016x} first_div={first_div:?}",
        pp_ids.len()
    );
    assert_eq!(
        first_div, None,
        "PP decode diverged from single-GPU at step {first_div:?}: pp={pp_ids:?} ref={ref_ids:?}"
    );
    assert_eq!(pp_fnv, ref_fnv, "token-stream FNV mismatch");
    println!(
        "pp_decode_parity: dense PP prefill+decode token stream == single-GPU forward_scratch \
         ({} tokens, argmax-exact) — PC-2 validated",
        pp_ids.len()
    );
}
