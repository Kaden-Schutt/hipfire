// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Perplexity / NLL eval for Maple-Preview (arch_id=15).
//!
//! This is the arch-15 analog of `deepseek4_perplexity` and `saddle-lab`'s
//! `perplexity`, and it computes PPL the SAME way so the number is comparable
//! to the other hipfire PPL figures rather than being a bespoke definition:
//!
//!   * take a window of `--ctx` tokens starting at `--offset`,
//!   * run the per-token `decode_step` over positions `0..ctx-1`,
//!   * score positions `pos` in `[warmup, ctx-1)` against target
//!     `window[pos + 1]`,
//!   * NLL is a NATURAL log of the softmax probability, and
//!     `PPL = exp(total_nll / scored)`.
//!
//! The only addition over the ds4 template is `--windows` / `--stride`, which
//! repeats that exact per-window procedure over several disjoint (or
//! overlapping) windows and pools the NLL. With the default `--windows 1` the
//! computation is identical to the ds4 definition token-for-token; with more
//! windows each one is an independent context (`MapleState::reset` zeroes the
//! KV and rewinds the cursor between them), which is the standard
//! "non-overlapping chunks" estimator and simply scores more tokens for a
//! tighter number.
//!
//! WHY THE PER-TOKEN PATH: `forward_batch` returns only the LAST position's
//! logits, so it cannot score every position — PPL needs a logprob at every
//! scored position. `decode_step` is the already-verified path and is used for
//! the whole run, prefill included. That costs a full forward per token, so a
//! 2048-token window is ~2048 forwards; budget accordingly.
//!
//! TOKENIZATION: pass `--corpus <text>` to tokenize with the model's OWN
//! tokenizer (embedded in the HFQ metadata). `--tokens <json>` accepts a
//! pre-tokenized JSON array of u32 ids, but it is GATED: any id >= the model's
//! `vocab_size` proves the ids came from a different vocab, and the run aborts
//! rather than silently reporting a meaningless number. A pre-tokenized file
//! that passes the gate still gets a decode sample printed so the ids can be
//! eyeballed against the model's tokenizer.
//!
//! Usage:
//!   maple_perplexity --model <model.hfq> --corpus <corpus.txt> \
//!       [--ctx 2048] [--warmup 8] [--offset 0] [--windows 1] [--stride N]
//!   maple_perplexity --model <model.hfq> --tokens <tokens.json> [...]

use hipfire_arch_maple::bundle::load_maple_from_hfq;
use hipfire_arch_maple::forward::decode_step;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use std::path::Path;
use std::time::Instant;

struct Args {
    model: String,
    corpus: Option<String>,
    tokens: Option<String>,
    ctx: usize,
    warmup: usize,
    offset: usize,
    windows: usize,
    stride: Option<usize>,
}

const USAGE: &str = "usage: maple_perplexity --model <model.hfq> \
                     (--corpus <corpus.txt> | --tokens <tokens.json>) \
                     [--ctx N] [--warmup N] [--offset N] [--windows N] [--stride N]";

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args {
        model: String::new(),
        corpus: None,
        tokens: None,
        ctx: 2048,
        warmup: 8,
        offset: 0,
        windows: 1,
        stride: None,
    };
    let mut i = 1;
    while i < argv.len() {
        let flag = argv[i].as_str();
        let val = || {
            argv.get(i + 1)
                .unwrap_or_else(|| panic!("{flag} missing value"))
                .clone()
        };
        match flag {
            "--model" => a.model = val(),
            "--corpus" => a.corpus = Some(val()),
            "--tokens" => a.tokens = Some(val()),
            "--ctx" => a.ctx = val().parse().expect("--ctx"),
            "--warmup" => a.warmup = val().parse().expect("--warmup"),
            "--offset" => a.offset = val().parse().expect("--offset"),
            "--windows" => a.windows = val().parse().expect("--windows"),
            "--stride" => a.stride = Some(val().parse().expect("--stride")),
            other => panic!("unknown arg {other}\n{USAGE}"),
        }
        i += 2;
    }
    assert!(!a.model.is_empty(), "{USAGE}");
    assert!(
        a.corpus.is_some() != a.tokens.is_some(),
        "exactly one of --corpus / --tokens is required\n{USAGE}"
    );
    assert!(
        a.ctx > a.warmup + 4,
        "--ctx must exceed --warmup by enough to score"
    );
    assert!(a.windows >= 1, "--windows must be >= 1");
    a
}

fn main() {
    let args = parse_args();
    let stride = args.stride.unwrap_or(args.ctx);

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch.clone());
    let mut hfq = HfqFile::open(Path::new(&args.model)).expect("open model");
    assert_eq!(
        hfq.arch_id, 15,
        "maple_perplexity: expected arch_id 15, got {}",
        hfq.arch_id
    );
    let tokenizer =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer from HFQ metadata");

    eprintln!("Loading weights from {}...", args.model);
    let t_load = Instant::now();
    let mut b = load_maple_from_hfq(&mut hfq, &mut gpu, args.ctx, "").expect("load maple bundle");
    eprintln!("Loaded in {:.1}s", t_load.elapsed().as_secs_f64());
    eprintln!(
        "maple: hidden={} layers={} experts={}/{} vocab={}",
        b.config.hidden_size,
        b.config.num_hidden_layers,
        b.config.num_experts,
        b.config.num_experts_per_tok,
        b.config.vocab_size,
    );

    // ---- corpus -> token ids -------------------------------------------
    let (all_tokens, source) = match (&args.corpus, &args.tokens) {
        (Some(path), None) => {
            let text = std::fs::read_to_string(path).expect("read corpus");
            eprintln!("Corpus: {} bytes from {path}", text.len());
            let t = Instant::now();
            let ids = tokenizer.encode(&text);
            eprintln!(
                "Tokenized with the MODEL'S OWN tokenizer: {} tokens in {:.2}s",
                ids.len(),
                t.elapsed().as_secs_f64()
            );
            (ids, format!("{path} (tokenized in-process)"))
        }
        (None, Some(path)) => {
            let raw = std::fs::read_to_string(path).expect("read tokens json");
            let ids: Vec<u32> = serde_json::from_str(&raw).expect("tokens json: expected [u32]");
            let max_id = ids.iter().copied().max().unwrap_or(0);
            eprintln!(
                "Pre-tokenized: {} ids from {path} (max id {max_id}, model vocab {})",
                ids.len(),
                b.config.vocab_size
            );
            // Hard gate: ids outside this model's vocab prove a different
            // tokenizer. Reporting a PPL from those would be meaningless, so
            // refuse rather than warn.
            assert!(
                (max_id as usize) < b.config.vocab_size,
                "REFUSING to score: --tokens file contains id {max_id} but this model's vocab \
                 is {} — these ids were produced by a DIFFERENT tokenizer. Re-run with \
                 --corpus <text> to tokenize with the model's own tokenizer.",
                b.config.vocab_size
            );
            let sample = &ids[..ids.len().min(32)];
            eprintln!("  decode sample: {:?}", tokenizer.decode(sample));
            (ids, format!("{path} (pre-tokenized, vocab gate passed)"))
        }
        _ => unreachable!("arg validation"),
    };

    let need = args.offset + (args.windows - 1) * stride + args.ctx;
    assert!(
        all_tokens.len() >= need,
        "corpus has {} tokens but offset={} windows={} stride={stride} ctx={} needs {need}",
        all_tokens.len(),
        args.offset,
        args.windows,
        args.ctx
    );

    let per_window_scored = args.ctx - args.warmup - 1;
    eprintln!(
        "Definition: {} window(s) of ctx={} at stride={stride} from offset={}, \
         warmup={} (unscored), scoring positions [{}, {}) => {} tok/window, {} total",
        args.windows,
        args.ctx,
        args.offset,
        args.warmup,
        args.warmup,
        args.ctx - 1,
        per_window_scored,
        per_window_scored * args.windows,
    );

    // ---- score ----------------------------------------------------------
    let mut total_nll: f64 = 0.0;
    let mut scored: usize = 0;
    let t0 = Instant::now();

    for w in 0..args.windows {
        let start = args.offset + w * stride;
        let window = &all_tokens[start..start + args.ctx];
        // Each window is an independent context: zero the KV and rewind the
        // cursor so window k cannot attend to window k-1.
        b.state.reset(&mut gpu).expect("state reset");

        for (pos, &tok) in window.iter().enumerate().take(window.len() - 1) {
            let logits = decode_step(
                &b.config,
                &b.weights,
                &mut b.state,
                &mut gpu,
                tok,
                pos as u32,
            )
            .expect("decode_step");
            if pos < args.warmup {
                continue;
            }
            let target = window[pos + 1] as usize;
            let nll = neg_log_softmax_at(&logits, target);
            if !nll.is_finite() {
                eprintln!(
                    "  warn: non-finite NLL at window={w} pos={pos} target={target}, skipping"
                );
                continue;
            }
            total_nll += nll as f64;
            scored += 1;

            if scored == 1 || scored % 128 == 0 {
                let avg = total_nll / scored as f64;
                let el = t0.elapsed().as_secs_f64();
                eprintln!(
                    "  win={w} pos={pos:5} scored={scored:6} nll/tok={avg:.4} ppl={:.3} ({:.2} tok/s)",
                    avg.exp(),
                    scored as f64 / el.max(1e-9),
                );
            }
        }
    }

    let avg_nll = if scored > 0 {
        total_nll / scored as f64
    } else {
        0.0
    };
    let ppl = avg_nll.exp();
    let elapsed = t0.elapsed().as_secs_f64();

    println!();
    println!("Model:    {}", args.model);
    println!("Corpus:   {source}");
    println!(
        "Window:   ctx={} warmup={} offset={} windows={} stride={stride}",
        args.ctx, args.warmup, args.offset, args.windows
    );
    println!("Scored:   {scored} tokens");
    println!("NLL/tok:  {avg_nll:.10}  (natural log)");
    println!("PPL:      {ppl:.4}");
    println!(
        "Elapsed:  {:.1}s ({:.2} tok/s, per-token decode_step path)",
        elapsed,
        scored as f64 / elapsed.max(1e-9)
    );

    // Sanity band. A coherent 20B on wikitext lands in high single digits to
    // low teens; anything far outside that is a harness bug (wrong tokenizer,
    // off-by-one between logit position and target, log base), not a real
    // quality result. Say so loudly instead of letting the number be quoted.
    if !(2.0..=100.0).contains(&ppl) {
        eprintln!();
        eprintln!(
            "*** RED FLAG: PPL={ppl:.4} is outside the plausible band [2, 100] for a coherent \
             20B model on natural text. Suspect the harness, not the model: wrong tokenizer, \
             an off-by-one between the logit position and the target token, or a log-base \
             confusion. Do NOT quote this as a quality figure without investigating. ***"
        );
        std::process::exit(3);
    }
}

/// -log softmax(logits)[target], in NATURAL log, computed in f64 from a
/// max-shifted sum so the exponentials cannot overflow.
fn neg_log_softmax_at(logits: &[f32], target: usize) -> f32 {
    if target >= logits.len() {
        return f32::NAN;
    }
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f64;
    for &v in logits {
        sum += ((v - max) as f64).exp();
    }
    let log_sum = max as f64 + sum.ln();
    (log_sum - logits[target] as f64) as f32
}
