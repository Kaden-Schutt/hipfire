// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! maple_coherence — load a Maple-Preview `.hfq`, prefill a prompt and greedily
//! decode, printing the text.
//!
//! This is the coherence gate for the arch-15 port. An exactness number alone
//! has never been sufficient evidence on this class of work: the packing can be
//! bit-perfect while the forward pass rotates the wrong activation, applies
//! RoPE on a NoPE layer, or normalises q/k in the wrong order — all of which
//! produce a model that loads, runs at full speed, and emits garbage.
//!
//! Usage:
//!   maple_coherence --model <model.hfq> [--prompt "..."] [--max-tokens N]
//!                   [--raw] [--kv-mode q8|bf16]
//!
//! `--kv-mode bf16` swaps the Q8_0 KV cache for the flat BF16 tier. Both run
//! the same sliding-window kernels with the same dim mapping and FMA order, so
//! a q8-vs-bf16 diff isolates KV storage precision from everything else.
//!
//! `HIPFIRE_MAPLE_PER_TOKEN_PREFILL=1` forces the per-token prefill path, so the
//! batched path can be A/B'd against it from one binary on one machine.
//!
//! `--raw` skips the ChatML frame and feeds the prompt verbatim (base-model
//! continuation check). Per-layer hidden-state dumping for cosine comparison
//! against the HF reference is a separate follow-up; it needs a capture hook
//! inside `decode_step_body`, which this harness deliberately does not have.

use hipfire_arch_maple::bundle::load_maple_from_hfq;
use hipfire_arch_maple::forward::decode_step;
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

struct Args {
    model: String,
    prompt: String,
    max_tokens: usize,
    raw: bool,
    kv_mode: String,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut model = None;
    let mut prompt = "The capital of France is".to_string();
    let mut max_tokens = 64usize;
    let mut raw = false;
    // "" = MAPLE_POLICY's default (q8). "bf16" selects the flat BF16 KV tier.
    let mut kv_mode = String::new();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(argv[i + 1].clone());
                i += 2;
            }
            "--prompt" => {
                prompt = argv[i + 1].clone();
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = argv[i + 1].parse().expect("--max-tokens");
                i += 2;
            }
            // Skip the ChatML frame and feed the prompt verbatim. Useful for a
            // base-model continuation check.
            "--raw" => {
                raw = true;
                i += 1;
            }
            // KV storage tier: "q8" (default) or "bf16". Anything else warns
            // and falls back to q8 via MAPLE_POLICY.
            "--kv-mode" => {
                kv_mode = argv[i + 1].clone();
                i += 2;
            }
            other => panic!("unknown arg {other}"),
        }
    }
    Args {
        model: model.expect("--model <model.hfq> is required"),
        prompt,
        max_tokens,
        raw,
        kv_mode,
    }
}

fn main() {
    let args = parse_args();
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(Path::new(&args.model)).expect("open model");
    assert_eq!(
        hfq.arch_id, 15,
        "maple_coherence: expected arch_id 15, got {}",
        hfq.arch_id
    );

    // Tokenize BEFORE allocating state: the KV cache must hold prompt +
    // generation. Sizing it from `max_tokens` alone under-allocates on any
    // prompt longer than the slack, and the KV write then runs off the end of
    // the cache.
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");

    let text = if args.raw {
        args.prompt.clone()
    } else {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            args.prompt
        )
    };
    let prompt_toks = tokenizer.encode(&text);

    let max_seq = prompt_toks.len() + args.max_tokens + 64;
    let mut b =
        load_maple_from_hfq(&mut hfq, &mut gpu, max_seq, &args.kv_mode).expect("load maple bundle");
    eprintln!(
        "maple: hidden={} layers={} experts={}/{} moe_inter={} vocab={} eos={} max_seq={}",
        b.config.hidden_size,
        b.config.num_hidden_layers,
        b.config.num_experts,
        b.config.num_experts_per_tok,
        b.config.moe_intermediate_size,
        b.config.vocab_size,
        b.eos_tok,
        max_seq,
    );
    eprintln!("prompt: {} token(s)", prompt_toks.len());

    // Prefill: batched when the checkpoint's tier supports it, else the
    // per-token path (the original zero-risk baseline) unchanged. The fallback
    // is not dead code — `forward_batch_supported` is false for any checkpoint
    // whose attention/expert weights are not uniformly qt51, or whose router
    // has no F16 mirror.
    let t0 = std::time::Instant::now();
    let mut logits = Vec::new();
    // `HIPFIRE_MAPLE_PER_TOKEN_PREFILL=1` forces the old per-token path. Both
    // arms then run from ONE binary on ONE machine, so the batched-vs-per-token
    // speedup is a controlled A/B rather than a comparison against a figure
    // recorded in an earlier session under unknown load. Under CPU contention
    // both arms are depressed together and the RATIO stays meaningful.
    let force_per_token =
        hipfire_config::developer_var("HIPFIRE_MAPLE_PER_TOKEN_PREFILL").as_deref() == Ok("1");
    if hipfire_arch_maple::forward::forward_batch_supported(&b.weights) && !force_per_token {
        for (start, len) in hipfire_arch_maple::batch::prefill_chunks(
            prompt_toks.len(),
            hipfire_arch_maple::batch::MAPLE_PREFILL_CHUNK,
        ) {
            logits = hipfire_arch_maple::forward::forward_batch(
                &b.config,
                &b.weights,
                &mut b.state,
                &mut gpu,
                &prompt_toks[start..start + len],
                start,
            )
            .expect("forward_batch");
        }
    } else {
        for (p, &tok) in prompt_toks.iter().enumerate() {
            logits = decode_step(&b.config, &b.weights, &mut b.state, &mut gpu, tok, p as u32)
                .expect("decode_step (prefill)");
        }
    }
    let mut pos = prompt_toks.len() as u32;
    let prefill_s = t0.elapsed().as_secs_f64();
    eprintln!(
        "prefill: {:.2}s ({:.1} tok/s)",
        prefill_s,
        prompt_toks.len() as f64 / prefill_s
    );

    // Greedy decode.
    let mut out = String::new();
    let t1 = std::time::Instant::now();
    let mut n_gen = 0usize;
    for _ in 0..args.max_tokens {
        let (best, _) =
            logits
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                    if v > acc.1 {
                        (i, v)
                    } else {
                        acc
                    }
                });
        let tok = best as u32;
        if tok == b.eos_tok {
            eprintln!("[eos]");
            break;
        }
        let piece = tokenizer.decode(&[tok]);
        out.push_str(&piece);
        print!("{piece}");
        use std::io::Write;
        let _ = std::io::stdout().flush();
        n_gen += 1;
        logits = decode_step(&b.config, &b.weights, &mut b.state, &mut gpu, tok, pos)
            .expect("decode_step (decode)");
        pos += 1;
    }
    let dec_s = t1.elapsed().as_secs_f64();
    println!();
    eprintln!(
        "decode: {n_gen} tok in {dec_s:.2}s ({:.1} tok/s)",
        n_gen as f64 / dec_s.max(1e-9)
    );

    // A model that emits the same token forever is the classic
    // wrong-forward-pass signature; call it out rather than letting a human
    // eyeball a wall of repeats.
    let distinct: std::collections::HashSet<char> = out.chars().collect();
    if n_gen >= 8 && distinct.len() <= 2 {
        eprintln!("WARNING: output is degenerate ({} distinct chars) — this is what a wrong rotation / RoPE-on-NoPE-layer / QK-norm-order bug looks like", distinct.len());
        std::process::exit(3);
    }
}
