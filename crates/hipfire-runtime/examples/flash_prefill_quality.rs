// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
//! Real-text quality harness for the batched prefill attention path.
//!
//! `dump_logits_qwen35` compares dispatch paths on a *deterministic fake*
//! prompt (token ids 0,1,2,...). That yields a near-uniform output
//! distribution (entropy ~9-10 nats on a 248K vocab), so its KLD and top-1
//! numbers are dominated by ties between effectively-equal tokens and cannot
//! answer "does this path change model behaviour". This harness runs the same
//! comparison on real text.
//!
//! For each window it prefills `ctx` real tokens through
//! `qwen35::forward_prefill_batch` (the batched path, so batched-attention
//! kernels are actually exercised) and scores the true next token from the
//! last-position logits. Reports NLL/token and perplexity, and dumps every
//! window's logits so a second run on a different dispatch path can be
//! compared with KLD / top-k agreement.
//!
//! Usage:
//!   flash_prefill_quality <model.hfq> <corpus.txt> <out.f32>
//!                         [--ctx N] [--windows W]
//!
//! Compare two runs that differ ONLY in dispatch (e.g. HIPFIRE_FLASH_PREFILL).

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use std::io::Write;
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "Usage: flash_prefill_quality <model.hfq> <corpus.txt> <out.f32> \
             [--ctx N] [--windows W]"
        );
        std::process::exit(2);
    }
    let model_path = args[1].clone();
    let corpus_path = args[2].clone();
    let out_path = args[3].clone();
    let mut ctx: usize = 4096;
    let mut windows: usize = 8;
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--ctx" => {
                ctx = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--windows" => {
                windows = args[i + 1].parse().unwrap();
                i += 2;
            }
            _ => i += 1,
        }
    }

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    eprintln!(
        "flash_prefill_quality: arch={} ctx={} windows={} flash={:?}",
        gpu.arch,
        ctx,
        windows,
        std::env::var("HIPFIRE_FLASH_PREFILL").unwrap_or_else(|_| "unset".into())
    );

    let tokenizer =
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .expect("tokenizer");
    let corpus = std::fs::read_to_string(&corpus_path).expect("read corpus");
    let all_tokens: Vec<u32> = tokenizer.encode(&corpus);
    let need = windows * ctx + 1;
    assert!(
        all_tokens.len() >= need,
        "corpus has {} tokens, need {need} for {windows}x{ctx}",
        all_tokens.len()
    );

    let weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("load weights");

    let kv_seq = (ctx + 16).max(512);
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        kv_seq,
    )
    .unwrap();
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 256, kv_seq).unwrap();

    let mut out = std::fs::File::create(&out_path).expect("create out");
    let mut total_nll = 0.0f64;

    for w in 0..windows {
        let start = w * ctx;
        let toks: Vec<u32> = all_tokens[start..start + ctx].to_vec();
        let next_tok = all_tokens[start + ctx] as usize;

        // Fresh recurrent state per window; the KV cache is overwritten from
        // position 0 so windows stay independent.
        let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
        qwen35::forward_prefill_batch(
            &mut gpu,
            &weights,
            &config,
            &toks,
            0,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            None,
            None,
            None,
            None,
        )
        .expect("prefill forward failed");
        gpu.hip.device_synchronize().expect("sync");

        let logits = gpu.download_f32(&scratch.logits).expect("download logits");
        let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse: f64 = logits.iter().map(|&x| ((x - maxl) as f64).exp()).sum::<f64>().ln()
            + maxl as f64;
        let nll = lse - logits[next_tok] as f64;
        total_nll += nll;
        eprintln!(
            "  window {w}: nll={nll:.4} next_tok={next_tok} argmax={}",
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        );

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                logits.as_ptr() as *const u8,
                logits.len() * std::mem::size_of::<f32>(),
            )
        };
        out.write_all(bytes).expect("write logits");
    }

    let mean_nll = total_nll / windows as f64;
    println!(
        "QUALITY windows={windows} ctx={ctx} mean_nll={mean_nll:.6} ppl={:.4}",
        mean_nll.exp()
    );
}
