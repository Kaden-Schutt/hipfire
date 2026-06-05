// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Greedy decode with per-step top-5 logit dump.
//!
//! Runs the same forward pass as greedy_dump (same chat wrapping, same
//! prefill, same argmax generation) but also records top-5 logit IDs +
//! values per step to a CSV next to the token output. Used as a
//! divergence diagnostic: compare two runs' CSVs to see whether an
//! argmax flip is a near-tie (ULP-scale gap between top-1 and top-2 =
//! FP drift) or a wide gap (= structural numerical error).
//!
//! Usage: greedy_dump_top5 <model.hfq> <out_prefix> [--max-gen N] [--kv-mode MODE] [prompt...]
//!   writes  <out_prefix>.tokens  — one token ID per line
//!           <out_prefix>.top5.csv — step,rank1_id,rank1_logit,...,rank5_id,rank5_logit

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::{self, KvCache};
    use std::io::Write;
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: greedy_dump_top5 <model.hfq> <out_prefix> [--max-gen N] [--kv-mode MODE] [prompt...]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let out_prefix = &args[2];
    let mut max_gen_override: Option<usize> = None;
    let mut kv_mode = std::env::var("HIPFIRE_KV_MODE").unwrap_or_else(|_| "q8".to_string());
    let mut prompt_parts = Vec::new();
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--max-gen" => {
                i += 1;
                max_gen_override = Some(
                    args.get(i)
                        .expect("--max-gen requires N")
                        .parse()
                        .expect("parse --max-gen"),
                );
            }
            "--kv-mode" => {
                i += 1;
                kv_mode = args.get(i).expect("--kv-mode requires MODE").clone();
            }
            other => prompt_parts.push(other.to_string()),
        }
        i += 1;
    }
    let prompt_text = if prompt_parts.is_empty() {
        "Write a 500-word essay about Federalist No. 10 by James Madison.".to_string()
    } else {
        prompt_parts.join(" ")
    };

    let mode = std::env::var("PROMPT_MODE").unwrap_or_else(|_| "thinking".to_string());
    eprintln!("greedy_dump_top5: {model_path} mode={mode}");

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    let tokenizer =
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tok");

    let mut prompt_tokens: Vec<u32> = match mode.as_str() {
        "raw" => tokenizer.encode(&prompt_text),
        _ => {
            let im_start = tokenizer.encode("<|im_start|>");
            let im_end = tokenizer.encode("<|im_end|>");
            let user = tokenizer.encode("user");
            let asst = tokenizer.encode("assistant");
            let nl = tokenizer.encode("\n");
            let user_body = tokenizer.encode(&prompt_text);
            let mut chat = Vec::new();
            chat.extend_from_slice(&im_start);
            chat.extend_from_slice(&user);
            chat.extend_from_slice(&nl);
            chat.extend_from_slice(&user_body);
            chat.extend_from_slice(&im_end);
            chat.extend_from_slice(&nl);
            chat.extend_from_slice(&im_start);
            chat.extend_from_slice(&asst);
            chat.extend_from_slice(&nl);
            if mode == "thinking" {
                chat.extend_from_slice(&tokenizer.encode("<think>"));
                chat.extend_from_slice(&nl);
            }
            chat
        }
    };
    eprintln!("prompt: {} tokens", prompt_tokens.len());

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");

    let kv_seq = 2048usize;
    eprintln!("greedy_dump_top5: kv_mode={kv_mode}");
    let mut kv_cache = match kv_mode.as_str() {
        "q8" => KvCache::new_gpu_q8(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            kv_seq,
        )
        .unwrap(),
        "asym4" | "turbo4" => KvCache::new_gpu_asym4(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            kv_seq,
        )
        .unwrap(),
        "asym3" | "turbo3" | "turbo" => KvCache::new_gpu_asym3(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            kv_seq,
        )
        .unwrap(),
        "asym2" | "turbo2" => KvCache::new_gpu_asym2(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            kv_seq,
        )
        .unwrap(),
        "fwht4" => KvCache::new_gpu_fwht4(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            kv_seq,
        )
        .unwrap(),
        "fwht3" => {
            let is_kv_layer = vec![true; config.n_layers];
            KvCache::new_gpu_fwht3_filtered(
                &mut gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
            )
            .unwrap()
        }
        "fwht2" => {
            let is_kv_layer = vec![true; config.n_layers];
            KvCache::new_gpu_fwht2_filtered(
                &mut gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
            )
            .unwrap()
        }
        other => panic!(
            "unknown --kv-mode/HIPFIRE_KV_MODE: {other} (q8|asym4|asym3|asym2|fwht4|fwht3|fwht2)"
        ),
    };
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    let max_gen =
        max_gen_override.unwrap_or_else(|| kv_seq.saturating_sub(prompt_tokens.len() + 8));
    let mut out_tokens =
        std::fs::File::create(format!("{out_prefix}.tokens")).expect("create out.tokens");
    let mut out_csv =
        std::fs::File::create(format!("{out_prefix}.top5.csv")).expect("create out.top5.csv");
    writeln!(out_csv, "step,r1_id,r1_logit,r2_id,r2_logit,r3_id,r3_logit,r4_id,r4_logit,r5_id,r5_logit,margin_top12").ok();

    // Helper: sort indices by logit desc and take top 5.
    fn top5(logits: &[f32]) -> [(u32, f32); 5] {
        // Partial top-5 via simple linear scan keeping a sorted window.
        let mut best: [(u32, f32); 5] = [(0, f32::NEG_INFINITY); 5];
        for (i, &v) in logits.iter().enumerate() {
            if v <= best[4].1 {
                continue;
            }
            best[4] = (i as u32, v);
            // Bubble up
            for j in (1..5).rev() {
                if best[j].1 > best[j - 1].1 {
                    best.swap(j, j - 1);
                } else {
                    break;
                }
            }
        }
        best
    }

    // Prefill
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            token,
            pos,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
        )
        .expect("prefill forward failed");
    }

    // First token after prefill
    let mut logits = gpu.download_f32(&scratch.logits).unwrap();
    let mut next_token = llama::argmax(&logits);
    writeln!(out_tokens, "{next_token}").ok();
    {
        let t = top5(&logits);
        let margin = t[0].1 - t[1].1;
        writeln!(
            out_csv,
            "0,{},{:.8},{},{:.8},{},{:.8},{},{:.8},{},{:.8},{:.8}",
            t[0].0, t[0].1, t[1].0, t[1].1, t[2].0, t[2].1, t[3].0, t[3].1, t[4].0, t[4].1, margin
        )
        .ok();
    }
    prompt_tokens.push(next_token);

    for step in 1..max_gen {
        let pos = prompt_tokens.len() - 1;
        if pos >= kv_seq {
            break;
        }
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            next_token,
            pos,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
        )
        .expect("forward failed");
        logits = gpu.download_f32(&scratch.logits).unwrap();
        next_token = llama::argmax(&logits);
        writeln!(out_tokens, "{next_token}").ok();
        let t = top5(&logits);
        let margin = t[0].1 - t[1].1;
        writeln!(
            out_csv,
            "{step},{},{:.8},{},{:.8},{},{:.8},{},{:.8},{},{:.8},{:.8}",
            t[0].0, t[0].1, t[1].0, t[1].1, t[2].0, t[2].1, t[3].0, t[3].1, t[4].0, t[4].1, margin
        )
        .ok();
        prompt_tokens.push(next_token);
        if next_token == config.eos_token {
            break;
        }
        if step % 100 == 0 {
            eprintln!("  step {step:4}");
        }
    }
    out_tokens.flush().ok();
    out_csv.flush().ok();
    eprintln!("done");
}
