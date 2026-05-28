// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! End-to-end OCR validation for dots-ocr (Strategy A, step 2).
//!
//! Loads the smoke image, runs our vision encoder, splices merger output
//! into a captured-from-HF prompt token list at `<|imgpad|>` positions,
//! greedy-decodes until EOS, prints the OCR JSON to stdout. The result
//! is then scored against `dots_ocr_smoke_001_vllm.json` by
//! `scripts/grade_dots_ocr_e2e.py`.
//!
//! This is intentionally MVP: prompt-building is delegated to a
//! pre-captured `input_token_ids` artifact (`dots_ocr_smoke_001.json`)
//! to skip the dots-ocr chat-template implementation. Once daemon-side
//! splicing lands (phase 3), this example will be obsoleted by a
//! proper serving path.
//!
//! Usage:
//!     export PATH=/opt/rocm-7.12/bin:$PATH
//!     export LD_LIBRARY_PATH=/opt/rocm-7.12/lib:$LD_LIBRARY_PATH
//!     cargo run --release -p hipfire-arch-dots-ocr --example ocr_e2e -- \
//!         --hfq ~/.hipfire/models/dots-ocr.q8.hfq \
//!         --image benchmarks/images/dots_ocr_smoke_001.jpg \
//!         --prompt-json benchmarks/references/dots_ocr_smoke_001.json \
//!         --max-tokens 16384 > /tmp/our_ocr.txt

use std::path::PathBuf;
use std::time::Instant;

use hipfire_arch_dots_ocr::{dots_ocr, image as preprocess};
use hipfire_arch_qwen2::qwen2::{self, Qwen2State, Qwen2Weights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{profile, Gpu};

struct Args {
    hfq: PathBuf,
    image: PathBuf,
    prompt_json: PathBuf,
    max_tokens: usize,
    max_seq: usize,
    prefill: String,
    profile_decode: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut hfq: Option<PathBuf> = None;
    let mut image: Option<PathBuf> = None;
    let mut prompt_json: Option<PathBuf> = None;
    let mut max_tokens: usize = 16384;
    let mut max_seq: usize = 8192;
    let mut prefill = "batch".to_string();
    let mut profile_decode = false;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--hfq" => hfq = Some(it.next().ok_or("--hfq needs value")?.into()),
            "--image" => image = Some(it.next().ok_or("--image needs value")?.into()),
            "--prompt-json" => prompt_json = Some(it.next().ok_or("--prompt-json needs value")?.into()),
            "--max-tokens" => max_tokens = it.next().ok_or("--max-tokens needs value")?
                .parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--max-seq" => max_seq = it.next().ok_or("--max-seq needs value")?
                .parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--prefill" => prefill = it.next().ok_or("--prefill needs batch|seq")?,
            "--profile-decode" => profile_decode = true,
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(Args {
        hfq: hfq.ok_or("--hfq required")?,
        image: image.ok_or("--image required")?,
        prompt_json: prompt_json.ok_or("--prompt-json required")?,
        max_tokens, max_seq, prefill, profile_decode,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    eprintln!("=== dots-ocr end-to-end OCR validation ===");
    eprintln!("HFQ:         {}", args.hfq.display());
    eprintln!("Image:       {}", args.image.display());
    eprintln!("Prompt JSON: {}", args.prompt_json.display());

    // 1. Load HFQ + configs.
    let mut hfq = HfqFile::open(&args.hfq)?;
    let dots_cfg = dots_ocr::DotsOcrConfig::from_hfq(&hfq)
        .map_err(|e| format!("config parse failed: {e}"))?;
    let text_cfg = dots_cfg.text.clone();
    eprintln!("Text decoder: layers={}, hidden={}, vocab={}",
        text_cfg.num_hidden_layers, text_cfg.hidden_size, text_cfg.vocab_size);
    eprintln!("Vision tower: layers={}, embed={}, out_hidden={}",
        dots_cfg.vision.num_hidden_layers, dots_cfg.vision.embed_dim, dots_cfg.vision.out_hidden_size);
    assert_eq!(text_cfg.hidden_size, dots_cfg.vision.out_hidden_size,
        "merger out_hidden_size must equal text decoder hidden_size for splicing");

    // 2. Load captured prompt token IDs.
    let prompt_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&args.prompt_json)?)?;
    let prompt_ids: Vec<u32> = prompt_json["input_token_ids"].as_array()
        .ok_or("input_token_ids missing or not array")?
        .iter()
        .map(|v| v.as_u64().map(|u| u as u32).ok_or("non-u32 token"))
        .collect::<Result<Vec<u32>, _>>()?;
    let n_imgpad = prompt_ids.iter().filter(|&&t| t == dots_ocr::IMGPAD_ID).count();
    eprintln!("Prompt: {} tokens ({} imgpad slots)", prompt_ids.len(), n_imgpad);
    if prompt_ids.len() + args.max_tokens > args.max_seq {
        eprintln!("warning: prompt ({}) + max_tokens ({}) > max_seq ({}). \
                   Bumping max_seq to fit.", prompt_ids.len(), args.max_tokens, args.max_seq);
    }
    let max_seq = (prompt_ids.len() + args.max_tokens).max(args.max_seq);

    // 3. Tokenizer (from HFQ metadata) for detokenization.
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer load failed: {e:?}"))?;

    // 4. GPU init.
    let mut gpu = Gpu::init()?;

    // 5. Vision pipeline: load weights → preprocess image → forward → download.
    eprintln!("\n[vision] loading weights...");
    let t = Instant::now();
    let vis_weights = dots_ocr::load_vision_weights(&hfq, &dots_cfg.vision, &mut gpu)?;
    gpu.hip.device_synchronize()?;
    eprintln!("[vision] weights loaded in {:.1}s", t.elapsed().as_secs_f32());

    let img = preprocess::preprocess_image(&args.image)?;
    let n_patches = img.n_patches();
    let n_visual_tokens = img.n_visual_tokens();
    eprintln!("[vision] preprocessed: resized {}x{}, grid {}x{}, {} patches → {} visual tokens",
        img.resized_h, img.resized_w, img.grid_h, img.grid_w, n_patches, n_visual_tokens);
    assert_eq!(n_visual_tokens, n_imgpad,
        "vision n_visual_tokens={n_visual_tokens} != prompt n_imgpad={n_imgpad} — \
         image grid does not match the captured prompt");

    let patches_gpu = gpu.upload_f32(&img.patches,
        &[n_patches, img.patches.len() / n_patches])?;
    eprintln!("[vision] running encoder...");
    let t = Instant::now();
    let merged_gpu = dots_ocr::vision_forward(&mut gpu, &vis_weights, &dots_cfg.vision,
        &patches_gpu, img.grid_h, img.grid_w)?;
    gpu.free_tensor(patches_gpu)?;
    gpu.hip.device_synchronize()?;
    eprintln!("[vision] encoder done in {:.1}s", t.elapsed().as_secs_f32());
    let merged: Vec<f32> = gpu.download_f32(&merged_gpu)?;
    gpu.free_tensor(merged_gpu)?;
    vis_weights.free_gpu(&mut gpu);
    assert_eq!(merged.len(), n_visual_tokens * text_cfg.hidden_size);

    // 6. Load text weights.
    eprintln!("\n[text] loading weights...");
    let t = Instant::now();
    let text_weights = Qwen2Weights::load(&mut hfq, &text_cfg, &mut gpu)
        .map_err(|e| format!("text weight load failed: {e}"))?;
    gpu.hip.device_synchronize()?;
    eprintln!("[text] weights loaded in {:.1}s", t.elapsed().as_secs_f32());

    let mut text_state = Qwen2State::new_with_max_seq(&mut gpu, &text_cfg, max_seq)
        .map_err(|e| format!("text state alloc failed: {e:?}"))?;

    // 7. Prefill: splice merger output at IMGPAD positions.
    eprintln!("\n[prefill] {} positions ({} visual + {} text)...",
        prompt_ids.len(), n_imgpad, prompt_ids.len() - n_imgpad);
    let t = Instant::now();
    let dim = text_cfg.hidden_size;
    let mut visual_idx = 0usize;
    if args.prefill == "batch" {
        // One batched call: build [batch, dim] embeds (visual rows from merger,
        // text rows from embed table), then forward_prefill_batch_embeds. Hits
        // the WMMA causal+GQA path when hd==128 && batch>=64.
        let mut embeds = vec![0.0f32; prompt_ids.len() * dim];
        for (pos, &token) in prompt_ids.iter().enumerate() {
            if token == dots_ocr::IMGPAD_ID {
                embeds[pos*dim..(pos+1)*dim].copy_from_slice(&merged[visual_idx*dim..(visual_idx+1)*dim]);
                visual_idx += 1;
            } else {
                let row = qwen2::embed_token_row(&mut gpu, &text_weights, &text_cfg, &mut text_state, token)?;
                embeds[pos*dim..(pos+1)*dim].copy_from_slice(&row);
            }
        }
        qwen2::forward_prefill_batch_embeds(&mut gpu, &text_weights, &text_cfg, &mut text_state, &embeds)?;
    } else {
        for (pos, &token) in prompt_ids.iter().enumerate() {
            if token == dots_ocr::IMGPAD_ID {
                let emb = &merged[visual_idx * dim..(visual_idx + 1) * dim];
                qwen2::forward_step_with_embed(&mut gpu, &text_weights, &text_cfg, &mut text_state, emb)?;
                visual_idx += 1;
            } else {
                qwen2::forward_step(&mut gpu, &text_weights, &text_cfg, &mut text_state, token)?;
            }
            if pos > 0 && pos % 500 == 0 {
                let so_far = t.elapsed().as_secs_f32();
                eprintln!("  [prefill] pos {}/{}  {:.1}s  ({:.1} tok/s)",
                    pos, prompt_ids.len(), so_far, (pos as f32) / so_far);
            }
        }
    }
    assert_eq!(visual_idx, n_visual_tokens, "spliced {visual_idx}/{n_visual_tokens} visual tokens");
    eprintln!("[prefill] mode={}", args.prefill);
    let prefill_s = t.elapsed().as_secs_f32();
    eprintln!("[prefill] done in {:.1}s ({:.1} tok/s)",
        prefill_s, prompt_ids.len() as f32 / prefill_s);

    // 8. Generate greedy. First token = argmax(logits) from the final prefill step.
    eprintln!("\n[generate] greedy, max_tokens={}, eos={}", args.max_tokens, text_cfg.eos_token_id);
    let t = Instant::now();
    let mut output_ids: Vec<u32> = Vec::with_capacity(args.max_tokens);
    let mut next = gpu.argmax_f32(&text_state.logits, text_cfg.vocab_size)?;
    let eos_set: Vec<u32> = if text_cfg.eos_token_ids.is_empty() {
        vec![text_cfg.eos_token_id]
    } else {
        text_cfg.eos_token_ids.clone()
    };
    for step in 0..args.max_tokens {
        if eos_set.contains(&next) {
            eprintln!("[generate] hit EOS ({next}) at step {step}");
            break;
        }
        output_ids.push(next);

        if args.profile_decode && step == 50 {
            profile::start();
        }
        if args.profile_decode && step == 55 {
            let entries = profile::stop().unwrap();
            let mut by_cat: std::collections::HashMap<&str, (f64, usize)> = std::collections::HashMap::new();
            let mut total_us = 0.0f64;
            for e in &entries {
                let (t, n) = by_cat.entry(e.category).or_default();
                *t += e.time_us;
                *n += 1;
                total_us += e.time_us;
            }
            let mut cats: Vec<_> = by_cat.into_iter().collect();
            cats.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
            eprintln!("\n=== DECODE PROFILE (5 steps, {} kernels) ===", entries.len());
            eprintln!("{:<20} {:>10} {:>10} {:>8} {:>10}", "category", "time(ms)", "launches", "avg(us)", "share%");
            for (cat, (time_us, n)) in &cats {
                eprintln!("{:<20} {:>10.2} {:>10} {:>8.1} {:>9.1}%",
                    cat, time_us / 1000.0, n, time_us / *n as f64, time_us / total_us * 100.0);
            }
            eprintln!("{:<20} {:>10.2} {:>10} {:>8.1}", "TOTAL", total_us / 1000.0, entries.len(), total_us / entries.len() as f64);
            let per_step = total_us / 5.0;
            eprintln!("Per decode step: {:.1} ms ({:.0} tok/s theoretical)",
                per_step / 1000.0, 1_000_000.0 / per_step);
        }

        next = qwen2::forward_step_greedy(&mut gpu, &text_weights, &text_cfg, &mut text_state, next)?;
        if step > 0 && step % 200 == 0 {
            let so_far = t.elapsed().as_secs_f32();
            eprintln!("  [generate] step {step}  {:.1}s  ({:.1} tok/s)",
                so_far, (step as f32) / so_far);
        }
    }
    let gen_s = t.elapsed().as_secs_f32();
    eprintln!("[generate] {} tokens in {:.1}s ({:.1} tok/s)",
        output_ids.len(), gen_s, output_ids.len() as f32 / gen_s);

    // 9. Decode + emit to stdout (the grading script consumes this).
    let decoded = tokenizer.decode(&output_ids);
    println!("{decoded}");

    eprintln!("\n[done] total: prefill {:.1}s + gen {:.1}s", prefill_s, gen_s);
    text_weights.free_gpu(&mut gpu);
    Ok(())
}
