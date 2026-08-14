// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Reproduce the non-zero-position overlap used by Gemma 4 EAGLE verify.
//!
//! Eagerly prefills a prompt, then rewrites its final token at `L - 1` with
//! `forward_batch_spec`.  Row zero must reproduce the eager prefill logits and
//! post-final-norm hidden regardless of the causally-masked suffix rows.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_gemma4::config::Gemma4Config;
    use hipfire_arch_gemma4::forward::{decode_step, forward_batch_spec};
    use hipfire_arch_gemma4::gemma4::{Gemma4State, Gemma4Weights};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use rdna_compute::DType;
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt = "Explain why a causal attention mask prevents future tokens from changing the current token.".to_string();
    let mut batch_sizes = vec![1usize, 2, 4, 6];
    let mut append_only = false;
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
            "--bs" => {
                batch_sizes = argv[i + 1]
                    .split(',')
                    .map(|s| s.parse::<usize>().expect("--bs"))
                    .collect();
                i += 2;
            }
            "--append-only" => {
                append_only = true;
                i += 1;
            }
            other => panic!("unknown argument {other}"),
        }
    }

    let model = model.expect("--model required");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let hfq = HfqFile::open(&model).expect("open model");
    let cfg = Gemma4Config::from_hfq(&hfq).expect("config");
    let weights = Gemma4Weights::load(&hfq, &cfg, &mut gpu).expect("weights");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let mut prompt_tokens = tokenizer.encode(&prompt);
    if prompt_tokens.first() != Some(&cfg.bos_token) {
        prompt_tokens.insert(0, cfg.bos_token);
    }
    assert!(
        prompt_tokens.len() >= 2,
        "prompt must contain at least two tokens"
    );

    let max_b = *batch_sizes.iter().max().expect("non-empty --bs");
    let max_seq = prompt_tokens.len() + max_b + 8;
    let last_pos = prompt_tokens.len() - 1;

    let argmax = |xs: &[f32]| -> u32 {
        xs.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap()
    };
    let cosine = |a: &[f32], b: &[f32]| -> f64 {
        let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            dot += f64::from(x) * f64::from(y);
            aa += f64::from(x) * f64::from(x);
            bb += f64::from(y) * f64::from(y);
        }
        dot / (aa.sqrt() * bb.sqrt() + 1e-30)
    };

    let mut all_pass = true;
    for &bsz in &batch_sizes {
        assert!((1..=64).contains(&bsz), "B={bsz} out of range");
        let mut state = Gemma4State::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("state");
        let mut eager_logits = Vec::new();
        for (pos, &token) in prompt_tokens.iter().enumerate() {
            eager_logits = decode_step(&cfg, &weights, &mut state, &mut gpu, token, pos as u32)
                .expect("prefill");
        }
        let eager_hidden = gpu.download_f32(&state.tmp).expect("download eager hidden");
        let eager_argmax = argmax(&eager_logits);

        // Row zero is the last prompt token.  Future rows deliberately vary,
        // but a correct causal mask makes them irrelevant to row zero.
        let mut overlap_pass = true;
        if !append_only {
            let mut block = Vec::with_capacity(bsz);
            block.push(prompt_tokens[last_pos]);
            for n in 1..bsz {
                block.push(((cfg.bos_token as usize + 17 * n) % cfg.vocab_size) as u32);
            }
            let hidden = gpu
                .alloc_tensor(&[bsz * cfg.dim], DType::F32)
                .expect("hidden output");
            let mut per_pos_argmax = Vec::new();
            forward_batch_spec(
                &cfg,
                &weights,
                &mut state,
                &mut gpu,
                &block,
                last_pos,
                Some(&hidden),
                Some(&mut per_pos_argmax),
            )
            .expect("overlap verify");
            let overlap_hidden = gpu.download_f32(&hidden).expect("download overlap hidden");
            let row0 = &overlap_hidden[..cfg.dim];
            let cos = cosine(&eager_hidden, row0);
            let max_abs = eager_hidden
                .iter()
                .zip(row0)
                .map(|(&x, &y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            let overlap_argmax = per_pos_argmax[0];
            overlap_pass = overlap_argmax == eager_argmax && cos >= 0.999;
            println!(
                "overlap B={bsz:<2} row0_argmax={} eager={} match={} hidden_cos={cos:.8} max_abs={max_abs:.6e} => {}",
                overlap_argmax,
                eager_argmax,
                overlap_argmax == eager_argmax,
                if overlap_pass { "PASS" } else { "FAIL" }
            );
        }

        // Control arm: append a genuinely new token at L.  This separates an
        // overlap/rewrite defect from a general B>1 batched-forward defect.
        let append_token = eager_argmax;
        let mut eager_append =
            Gemma4State::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("append eager state");
        for (pos, &token) in prompt_tokens.iter().enumerate() {
            decode_step(
                &cfg,
                &weights,
                &mut eager_append,
                &mut gpu,
                token,
                pos as u32,
            )
            .expect("append eager prefill");
        }
        let append_logits = decode_step(
            &cfg,
            &weights,
            &mut eager_append,
            &mut gpu,
            append_token,
            prompt_tokens.len() as u32,
        )
        .expect("append eager token");
        let append_hidden = gpu
            .download_f32(&eager_append.tmp)
            .expect("download append eager hidden");
        let append_argmax = argmax(&append_logits);

        let mut batch_append =
            Gemma4State::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("append batch state");
        for (pos, &token) in prompt_tokens.iter().enumerate() {
            decode_step(
                &cfg,
                &weights,
                &mut batch_append,
                &mut gpu,
                token,
                pos as u32,
            )
            .expect("append batch prefill");
        }
        let mut append_block = Vec::with_capacity(bsz);
        append_block.push(append_token);
        for n in 1..bsz {
            append_block.push(((cfg.bos_token as usize + 29 * n) % cfg.vocab_size) as u32);
        }
        let append_hidden_out = gpu
            .alloc_tensor(&[bsz * cfg.dim], DType::F32)
            .expect("append hidden output");
        let mut append_argmax_rows = Vec::new();
        forward_batch_spec(
            &cfg,
            &weights,
            &mut batch_append,
            &mut gpu,
            &append_block,
            prompt_tokens.len(),
            Some(&append_hidden_out),
            Some(&mut append_argmax_rows),
        )
        .expect("append batch verify");
        let append_batch_hidden = gpu
            .download_f32(&append_hidden_out)
            .expect("download append batch hidden");
        let append_row0 = &append_batch_hidden[..cfg.dim];
        let append_cos = cosine(&append_hidden, append_row0);
        let append_max_abs = append_hidden
            .iter()
            .zip(append_row0)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let append_match = append_argmax_rows[0] == append_argmax;
        let append_pass = append_match && append_cos >= 0.999;
        all_pass &= overlap_pass && append_pass;
        println!(
            "append  B={bsz:<2} row0_argmax={} eager={} match={} hidden_cos={append_cos:.8} max_abs={append_max_abs:.6e} => {}",
            append_argmax_rows[0],
            append_argmax,
            append_match,
            if append_pass { "PASS" } else { "FAIL" }
        );
    }

    println!("OVERALL: {}", if all_pass { "PASS" } else { "FAIL" });
    if !all_pass {
        std::process::exit(1);
    }
}
