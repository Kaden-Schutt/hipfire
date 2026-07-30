// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
//! Real-text, all-position quality harness for the batched prefill path.
//!
//! Why not the existing tools:
//!   * `perplexity.rs` scores via `forward_scratch` at batch_size=1, so it
//!     never reaches the batched attention kernels at all — it would report a
//!     perfect null result for any change to them.
//!   * `dump_logits_qwen35` compares dispatch paths on a deterministic FAKE
//!     prompt (token ids 0,1,2,...). Measured output entropy there is 9-10
//!     nats on a 248K vocab, so its KLD/top-1 numbers are dominated by ties
//!     between effectively-equal tokens.
//!   * `build_kld_ref_native` + `eval_hipfire` are the right shape but the
//!     reference builder sets HIPFIRE_KV_MODE=f32, which the current config
//!     validator rejects (`memory.kv_cache` enum has no "f32"), so it cannot
//!     run.
//!
//! This scores EVERY position (subject to `--stride`) in the second half of
//! each chunk, matching llama-perplexity chunking semantics: DeltaNet state
//! reset per chunk, KV overwritten from 0, scored window [n_ctx/2, n_ctx-1).
//! The transformer stack runs through `forward_prefill_batch` (so batched
//! attention IS exercised) with `per_token_hidden_out` capturing one row per
//! scored token; logits are then recovered per row via the lm_head gemv.
//!
//! Emits a fixed-size record per scored position so two runs differing only
//! in dispatch can be compared exactly and pairwise.
//!
//! Usage:
//!   flash_prefill_quality <model.hfq> <corpus.txt> <out.bin>
//!                         [--ctx N] [--chunks C] [--stride S]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::{gemv_family, DispatchCtx, KvCache};
    use rdna_compute::DType;
    use std::io::Write;
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "Usage: flash_prefill_quality <model.hfq> <corpus.txt> <out.bin> \
             [--ctx N] [--chunks C] [--stride S]"
        );
        std::process::exit(2);
    }
    let model_path = args[1].clone();
    let corpus_path = args[2].clone();
    let out_path = args[3].clone();
    let mut n_ctx: usize = 4096;
    let mut chunks: usize = 8;
    let mut stride: usize = 8;
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--ctx" => {
                n_ctx = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--chunks" => {
                chunks = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--stride" => {
                stride = args[i + 1].parse().unwrap();
                i += 2;
            }
            _ => i += 1,
        }
    }

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let scoring_start = n_ctx / 2;
    let scored_per_chunk = n_ctx - 1 - scoring_start;
    eprintln!(
        "flash_prefill_quality: arch={} ctx={} chunks={} stride={} scored/chunk={} flash={:?}",
        gpu.arch,
        n_ctx,
        chunks,
        stride,
        scored_per_chunk.div_ceil(stride),
        std::env::var("HIPFIRE_FLASH_PREFILL").unwrap_or_else(|_| "unset".into())
    );

    let tokenizer =
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .expect("tokenizer");
    let corpus = std::fs::read_to_string(&corpus_path).expect("read corpus");
    let all_tokens: Vec<u32> = tokenizer.encode(&corpus);
    let need = chunks * n_ctx + 1;
    assert!(
        all_tokens.len() >= need,
        "corpus has {} tokens, need {need}",
        all_tokens.len()
    );

    let weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("load weights");

    let kv_seq = (n_ctx + 16).max(512);
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        kv_seq,
    )
    .unwrap();
    // Allocated ONCE and reset in place per chunk: DeltaNetState has no Drop
    // impl, so per-chunk allocation leaks ~6 MB x n_la_layers.
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 256, kv_seq).unwrap();
    let h_buf = gpu
        .alloc_tensor(&[scored_per_chunk, config.dim], DType::F32)
        .expect("alloc hidden buf");

    let mut out = std::fs::File::create(&out_path).expect("create out");
    let mut total_nll = 0.0f64;
    let mut n_scored = 0usize;

    for c in 0..chunks {
        let base = c * n_ctx;
        let toks: Vec<u32> = all_tokens[base..base + n_ctx].to_vec();
        dn_state.reset(&mut gpu);

        // Prefix [0, scoring_start): builds KV, no capture.
        qwen35::forward_prefill_batch(
            &mut gpu, &weights, &config, &toks[0..scoring_start], 0, &mut kv_cache,
            &mut dn_state, &scratch, None, None, None, None,
        )
        .expect("prefix prefill");

        // Scored region [scoring_start, n_ctx-1): capture one hidden row per token.
        qwen35::forward_prefill_batch(
            &mut gpu,
            &weights,
            &config,
            &toks[scoring_start..(n_ctx - 1)],
            scoring_start,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            None,
            Some(&h_buf),
            None,
            None,
        )
        .expect("scored prefill");
        gpu.hip.device_synchronize().expect("sync");

        let mut j = 0usize;
        while j < scored_per_chunk {
            let row = h_buf.sub_offset(j * config.dim, config.dim);
            {
                let ctx_d = DispatchCtx::new(&gpu);
                gemv_family()
                    .run_auto(
                        &ctx_d,
                        &mut gpu,
                        &weights.output.dispatch_ref(),
                        &row,
                        &scratch.logits,
                    )
                    .expect("gemv lm_head");
            }
            let logits = gpu.download_f32(&scratch.logits).expect("download logits");
            let v = config.vocab_size.min(logits.len());
            let lg = &logits[..v];

            let maxl = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse: f64 =
                lg.iter().map(|&x| ((x - maxl) as f64).exp()).sum::<f64>().ln() + maxl as f64;
            let next_tok = toks[scoring_start + j + 1] as usize;
            let nll = lse - lg[next_tok] as f64;
            total_nll += nll;
            n_scored += 1;

            // top-8 indices, descending.
            let mut idx: Vec<u32> = (0..v as u32).collect();
            idx.select_nth_unstable_by(8, |&a, &b| {
                lg[b as usize].partial_cmp(&lg[a as usize]).unwrap()
            });
            let mut top8: Vec<u32> = idx[..8].to_vec();
            top8.sort_by(|&a, &b| lg[b as usize].partial_cmp(&lg[a as usize]).unwrap());

            // Record: next_tok u32 | nll f32 | lse f32 | top8 u32[8] = 44 bytes
            out.write_all(&(next_tok as u32).to_le_bytes()).unwrap();
            out.write_all(&(nll as f32).to_le_bytes()).unwrap();
            out.write_all(&(lse as f32).to_le_bytes()).unwrap();
            for t in &top8 {
                out.write_all(&t.to_le_bytes()).unwrap();
            }
            j += stride;
        }
        eprintln!(
            "  chunk {c}/{chunks}: scored={n_scored} running_ppl={:.4}",
            (total_nll / n_scored as f64).exp()
        );
    }

    let mean_nll = total_nll / n_scored as f64;
    println!(
        "QUALITY scored={n_scored} ctx={n_ctx} chunks={chunks} stride={stride} \
         mean_nll={mean_nll:.6} ppl={:.4}",
        mean_nll.exp()
    );
}
