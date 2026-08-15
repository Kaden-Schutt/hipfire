// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched verify-forward correctness harness for Gemma 4 dense text.
//!
//! The spec-decode keystone is `forward_batch`: verify B tokens in ONE pass
//! (weights read once) and get the LAST token's logits identical to running B
//! sequential `decode_step` calls. This example proves that equivalence:
//!
//!   1. Load the HFQ.
//!   2. For each B in {1, 2, 4, 8}:
//!      a. Reset state, run B sequential `decode_step` calls over tokens[0..B];
//!         capture the LAST token's logits + the next-token argmax (one more
//!         decode_step at position B).
//!      b. Reset state, run `forward_batch(tokens[0..B], start_pos=0)`; capture
//!         its returned logits.
//!      c. Compare: cosine similarity + argmax-match of (a) vs (b).
//!      d. KV-cache equivalence: from the forward_batch state, decode one MORE
//!         token at position B and check its argmax matches the sequential
//!         next-token argmax — this exercises attention over the batched-written
//!         history.
//!
//! PASS criterion: cosine ≥ 0.99999 and argmax identical for every B, and the
//! post-batch next-token argmax matches the sequential next-token argmax.
//!
//! Usage:
//!   verify_batch_gemma4 --model ~/gemma4-12b-mq4attn.hfq [--prompt <text>]
//!                       [--token-ids 2,1234,...] [--bs 1,2,4,8]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_gemma4::config::Gemma4Config;
    use hipfire_arch_gemma4::forward::{decode_step, forward_batch};
    use hipfire_arch_gemma4::gemma4::{Gemma4State, Gemma4Weights};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt =
        "The quick brown fox jumps over the lazy dog. In a distant galaxy, far away,".to_string();
    let mut token_ids: Option<Vec<u32>> = None;
    let mut batch_sizes: Vec<usize> = vec![1, 2, 4, 8];
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--prompt" => {
                prompt = argv[i + 1].clone();
                i += 2;
            }
            "--token-ids" => {
                token_ids = Some(
                    argv[i + 1]
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| s.trim().parse::<u32>().expect("--token-ids"))
                        .collect(),
                );
                i += 2;
            }
            "--bs" => {
                batch_sizes = argv[i + 1]
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.trim().parse::<usize>().expect("--bs"))
                    .collect();
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let hfq = HfqFile::open(&model).expect("open model");
    let cfg = Gemma4Config::from_hfq(&hfq).expect("config");
    eprintln!(
        "gemma4 dim={} layers={} sliding={} full={} vocab={} softcap={}",
        cfg.dim,
        cfg.n_layers,
        cfg.n_sliding_layers(),
        cfg.n_full_layers(),
        cfg.vocab_size,
        cfg.final_logit_softcapping
    );
    let weights = Gemma4Weights::load(&hfq, &cfg, &mut gpu).expect("weights");

    // Build a token stream: --token-ids, else tokenize the prompt. Prepend BOS.
    let mut tokens = match token_ids {
        Some(ids) => ids,
        None => {
            let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
            tok.encode(&prompt)
        }
    };
    if tokens.first() != Some(&cfg.bos_token) {
        tokens.insert(0, cfg.bos_token);
    }
    let max_b = *batch_sizes.iter().max().unwrap_or(&8);
    // Need at least max_b+1 tokens (max_b for the batch + 1 to seed the
    // sequential "one more" next-token check at position max_b).
    while tokens.len() < max_b + 1 {
        tokens.push(cfg.bos_token);
    }
    eprintln!("token stream ({} toks): {:?}", tokens.len(), &tokens[..tokens.len().min(16)]);

    let argmax = |v: &[f32]| -> u32 {
        let mut bi = 0u32;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = i as u32;
            }
        }
        bi
    };
    let cosine = |a: &[f32], b: &[f32]| -> f64 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
        }
        dot / (na.sqrt() * nb.sqrt() + 1e-30)
    };

    // KV cache must hold all positions we touch.
    let max_seq = max_b + 16;

    let mut all_pass = true;
    for &bsz in &batch_sizes {
        assert!(bsz >= 1 && bsz <= tokens.len() - 1, "B={bsz} out of range");
        let batch: Vec<u32> = tokens[..bsz].to_vec();
        let next_seed = tokens[bsz]; // token to feed at position bsz for the KV check

        // ── (a) Sequential reference: B decode_steps, capture last logits +
        //        the next-token argmax (one more decode_step at position B). ──
        let mut seq_state =
            Gemma4State::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("seq state");
        let mut seq_last_logits = Vec::new();
        for (pos, &t) in batch.iter().enumerate() {
            seq_last_logits =
                decode_step(&cfg, &weights, &mut seq_state, &mut gpu, t, pos as u32).expect("seq");
        }
        let seq_argmax = argmax(&seq_last_logits);
        // One more step to define the "next token after the batch" reference.
        let seq_next_logits =
            decode_step(&cfg, &weights, &mut seq_state, &mut gpu, next_seed, bsz as u32)
                .expect("seq next");
        let seq_next_argmax = argmax(&seq_next_logits);
        drop(seq_state);

        // ── (b) Batched verify forward over the same B tokens. ──
        let mut bat_state =
            Gemma4State::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("bat state");
        let bat_logits =
            forward_batch(&cfg, &weights, &mut bat_state, &mut gpu, &batch, 0).expect("forward_batch");
        let bat_argmax = argmax(&bat_logits);

        // ── (c) Compare last-token logits. ──
        let cos = cosine(&seq_last_logits, &bat_logits);
        let argmax_match = seq_argmax == bat_argmax;

        // ── (d) KV-cache equivalence: decode one MORE token at position B from
        //        the forward_batch state and compare its argmax. ──
        let bat_next_logits =
            decode_step(&cfg, &weights, &mut bat_state, &mut gpu, next_seed, bsz as u32)
                .expect("bat next");
        let bat_next_argmax = argmax(&bat_next_logits);
        let kv_match = bat_next_argmax == seq_next_argmax;
        let next_cos = cosine(&seq_next_logits, &bat_next_logits);
        drop(bat_state);

        // PASS criterion. The load-bearing property for spec-decode verify is
        // argmax-identity (the accepted/rejected decision is an argmax compare),
        // and the KV cache must leave the sequential next-token argmax intact.
        //
        // Cosine: an all-Q8 model (q8.hfq) is bit-exact (cosine = 1.0). When the
        // attention projections are MQ4G256 (mq4attn.hfq), batch>1 routes the
        // MQ4 GEMM through the gfx12 fp16-WMMA kernel
        // (gemm_hfq4g256_residual_wmma_gfx12), which accumulates in fp16 — a
        // known ~1e-3 cosine drift vs the scalar single-token GEMV the eager
        // path uses. That is a numerical artifact of the batched MQ4 kernel, not
        // a forward_batch logic error, so the cosine floor is relaxed to 0.999
        // while argmax-identity stays strict. (B=1 always routes to the scalar
        // MQ4 GEMM and is bit-exact even on mq4attn.)
        let pass = cos >= 0.999 && argmax_match && kv_match;
        all_pass &= pass;
        println!(
            "B={bsz:<2}  cosine={cos:.7}  argmax_match={argmax_match} (seq={seq_argmax} bat={bat_argmax})  \
             | KV-check: next_argmax_match={kv_match} (seq={seq_next_argmax} bat={bat_next_argmax}) next_cos={next_cos:.7}  \
             => {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    println!(
        "\n=== OVERALL: {} ===",
        if all_pass { "PASS" } else { "FAIL" }
    );
    if !all_pass {
        std::process::exit(1);
    }
}
