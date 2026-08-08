// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Standalone single-GPU driver for the RUNTIME llama forward
//! (`hipfire_runtime::llama::forward_scratch`) — the reference harness for
//! PB-TP4c full-model logit parity.
//!
//! Every existing *_parity / logit_dump example drives the qwen35 arch path;
//! none drives the runtime llama forward standalone. This one does: it loads a
//! llama-family HFQ (arch_id 0/1) via the HFQ WeightSource, prefills a prompt,
//! then greedily decodes, dumping the argmax token sequence + a logit
//! fingerprint per step so a later TP forward can be diffed against it.
//!
//! Usage: llama_logit_dump <model.mq4|.hfq> [out_dir] [n_gen]

use hipfire_runtime::llama::{self, ForwardScratch, KvCache, LlamaWeights};
use hipfire_runtime::tokenizer::Tokenizer;
use std::io::Write;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(String::as_str).unwrap_or(concat!(
        env!("HOME"),
        "/.hipfire/models/qwen3-0.6b-llama.mq4"
    ));
    let out_dir = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("/home/bjoern/hipfire-tp-ref");
    let n_gen: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(48);
    std::fs::create_dir_all(out_dir).unwrap();

    let prompt_text = "The quick brown fox jumps over the lazy dog. Explain how a four-stroke combustion engine works.";

    let hfq =
        hipfire_runtime::hfq::HfqFile::open(Path::new(model_path)).expect("failed to open model");
    eprintln!("HFQ arch_id = {}", hfq.arch_id);
    let config = hipfire_runtime::hfq::config_from_hfq(&hfq).expect("config_from_hfq failed");
    let tokenizer =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("need tokenizer in hfq metadata");

    eprintln!(
        "Config: arch={:?} dim={} layers={} heads={} kv_heads={} head_dim={} vocab={}",
        config.arch,
        config.dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        config.vocab_size,
    );

    let prompt_tokens = tokenizer.encode(prompt_text);
    eprintln!("Prompt: {} tokens", prompt_tokens.len());

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let weights: LlamaWeights = hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, &mut gpu)
        .expect("load_weights_hfq failed");

    let max_seq = 2048usize;
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        max_seq,
    )
    .expect("kv cache alloc failed");
    let scratch = ForwardScratch::new(&mut gpu, &config).expect("scratch alloc failed");

    let mut tok_file = std::fs::File::create(format!("{out_dir}/token_sequence.txt")).unwrap();
    let mut txt_file = std::fs::File::create(format!("{out_dir}/generated_text.txt")).unwrap();

    // Greedy: temp=0, top_p=1, no repeat penalty. We read scratch.logits and run
    // our own argmax so the reference is deterministic regardless of the
    // sampler's internal RNG plumbing.
    let (temp, top_p, rng, rw, rp) = (0.0f32, 1.0f32, 0u32, 0usize, 1.0f32);

    eprintln!("Prefilling {} tokens...", prompt_tokens.len());
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        llama::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            token,
            pos,
            &mut kv_cache,
            &scratch,
            temp,
            top_p,
            rng,
            rw,
            rp,
        )
        .expect("prefill forward failed");
    }

    let logits = gpu.download_f32(&scratch.logits).unwrap();
    let mut next = llama::argmax(&logits);
    eprintln!(
        "prefill last-pos logit fp: argmax={next} val={:.6} fnv={:016x}",
        logits[next as usize],
        logit_fnv(&logits)
    );

    eprintln!("Generating {n_gen} tokens (greedy)...");
    for step in 0..n_gen {
        let text = tokenizer.decode(&[next]);
        writeln!(tok_file, "{next}").unwrap();
        write!(txt_file, "{text}").unwrap();
        if step < 12 {
            eprintln!("  step {step:3}: token={next:6} {:?}", text);
        }
        if next == config.eos_token {
            eprintln!("  <eos> at step {step}");
            break;
        }
        let pos = prompt_tokens.len() + step;
        if pos >= max_seq {
            break;
        }
        llama::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            next,
            pos,
            &mut kv_cache,
            &scratch,
            temp,
            top_p,
            rng,
            rw,
            rp,
        )
        .expect("gen forward failed");
        let l = gpu.download_f32(&scratch.logits).unwrap();
        next = llama::argmax(&l);
    }

    tok_file.flush().unwrap();
    txt_file.flush().unwrap();
    eprintln!("\nDumped to {out_dir}/ — standalone runtime-llama forward OK");
}

/// FNV-1a over the raw logit bytes — a cheap fingerprint for cross-run diffing.
fn logit_fnv(logits: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in logits {
        for b in v.to_ne_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}
