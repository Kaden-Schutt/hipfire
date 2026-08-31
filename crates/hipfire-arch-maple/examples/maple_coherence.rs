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
//!                   [--temp T] [--top-p P] [--seed N]
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
    /// 0.0 = greedy (default, unchanged behaviour). > 0 = sample.
    temp: f32,
    top_p: f32,
    seed: u64,
}

/// SplitMix64 — a 64-bit mixer used here as the sampling RNG.
///
/// Deliberately self-contained and NOT the engine's sampler: this harness needs
/// a stream that depends only on `--seed`, so two runs of the same arm are
/// reproducible and two different seeds are genuinely independent draws. It is
/// not trying to match production sampling numerics.
struct SplitMix64(u64);
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1). 53 bits of mantissa, so the quantisation is far finer
    /// than any probability this is used to compare against.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Temperature + top-p (nucleus) sampling.
///
/// Softmax is computed in f64 max-shifted so the exponentials cannot overflow;
/// the vocab is 151,936 wide and the raw logit range is large enough that the
/// naive form does overflow in f32.
fn sample_top_p(logits: &[f32], temp: f32, top_p: f32, rng: &mut SplitMix64) -> u32 {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let t = temp.max(1e-6) as f64;
    let mut p: Vec<f64> = logits
        .iter()
        .map(|&v| ((v as f64 - max) / t).exp())
        .collect();
    let sum: f64 = p.iter().sum();
    for v in p.iter_mut() {
        *v /= sum;
    }
    // Descending by probability, then keep the smallest prefix whose mass
    // reaches top_p. The prefix always keeps at least one token, so a
    // degenerate top_p cannot produce an empty nucleus.
    idx.sort_unstable_by(|&a, &b| {
        p[b as usize]
            .partial_cmp(&p[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cum = 0.0;
    let mut cut = idx.len();
    for (n, &i) in idx.iter().enumerate() {
        cum += p[i as usize];
        if cum >= top_p as f64 {
            cut = n + 1;
            break;
        }
    }
    let nucleus = &idx[..cut.max(1)];
    let mass: f64 = nucleus.iter().map(|&i| p[i as usize]).sum();
    let mut r = rng.next_f64() * mass;
    for &i in nucleus {
        r -= p[i as usize];
        if r <= 0.0 {
            return i;
        }
    }
    nucleus[nucleus.len() - 1]
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut model = None;
    let mut prompt = "The capital of France is".to_string();
    let mut max_tokens = 64usize;
    let mut raw = false;
    // "" = MAPLE_POLICY's default (bf16). "q8" selects the block-quantized tier.
    let mut kv_mode = String::new();
    let mut temp = 0.0f32;
    let mut top_p = 0.95f32;
    let mut seed = 0u64;
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
            "--temp" => {
                temp = argv[i + 1].parse().expect("--temp");
                i += 2;
            }
            "--top-p" => {
                top_p = argv[i + 1].parse().expect("--top-p");
                i += 2;
            }
            "--seed" => {
                seed = argv[i + 1].parse().expect("--seed");
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
        temp,
        top_p,
        seed,
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
            // The trailing "<think>\n" is REQUIRED and was missing until
            // 2026-08-31. Maple's embedded jinja template ends its generation
            // prompt with `'<|im_start|>assistant\n<think>\n'`, and the vendor's
            // llama.cpp README calls out `--jinja` as applying the template
            // "exactly, including its thinking prefix".
            //
            // Without it the model has to emit the opening <think> itself, so
            // every generation starts off-distribution INSIDE the reasoning
            // block — which is exactly where this model's degenerate loops
            // occur. Any loop-rate measurement taken without this prefix is
            // measuring a prompt frame the model was never trained on.
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
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

    // Decode. `--temp 0` (the default) is greedy and bit-for-bit reproduces the
    // previous behaviour; `--temp > 0` samples with top-p and an EXPLICIT seed.
    //
    // The seed is what makes a loop-rate measurement possible at all: greedy
    // gives exactly ONE draw per (prompt, model), so sample size can only grow
    // with the prompt set and prompt dominates the variance. With a seed, the
    // same prompt can be redrawn N times and the arms compared on equal terms.
    let mut out = String::new();
    let t1 = std::time::Instant::now();
    let mut n_gen = 0usize;
    let mut rng = SplitMix64::new(args.seed);
    for _ in 0..args.max_tokens {
        let tok = if args.temp <= 0.0 {
            logits
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                    if v > acc.1 {
                        (i, v)
                    } else {
                        acc
                    }
                })
                .0 as u32
        } else {
            sample_top_p(&logits, args.temp, args.top_p, &mut rng)
        };
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
