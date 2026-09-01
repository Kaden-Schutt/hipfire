// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! KL divergence between hipfire's Maple-Preview (arch_id=15) output
//! distribution and a full-precision BF16 reference, on IDENTICAL tokens under
//! an IDENTICAL protocol.
//!
//! WHY THIS EXISTS. hipfire's Maple scores a wikitext PPL roughly 10x worse
//! than a comparable anchor. The harness has been cleared (tokenizer checked
//! against an independent BPE, alignment and KV proven by an induction test,
//! logit scale ruled out, generation coherent), so the number is real but its
//! CAUSE is not attributed. Either the model is genuinely weak at raw-text LM
//! (~1B active params, natively ternary, a "Preview" reasoning model), or there
//! is an arch-15 numerics defect that blurs the distribution without moving the
//! greedy argmax. Coherence and induction cannot tell those apart; KL against a
//! full-precision reference can.
//!
//! Weight packing for this model is separately verified VALUE-EXACT, so any
//! divergence measured here is RUNTIME numerics (the Q8 KV cache, F16
//! activations into the WMMA GEMMs), not weight loss. The reference is BF16 and
//! also finite-precision, so a small nonzero floor is expected even from a
//! perfect implementation.
//!
//! PROTOCOL. One unbroken window from position 0, per-token `decode_step` (the
//! verified path — `forward_batch` returns only the last position's logits and
//! so cannot score every position). Row `i` of both sides is the next-token
//! distribution after consuming `tokens[0..=i]`, i.e. it predicts
//! `tokens[i+1]`; the final row has no target. PPL is `exp(mean NLL)` in
//! NATURAL log over positions `[0, n-1)`, which is exactly how the reference's
//! own number was produced, so the two are directly comparable.
//!
//! REFERENCE FILE LAYOUT. `u32 n_positions`, `u32 vocab_size`, then
//! `n_positions * vocab_size` f32 little-endian, position-major. Logits are
//! RAW: this example applies its own log-softmax over the full row.
//!
//! MEMORY. A full side is `n_positions * vocab_size * 4` bytes — 1.2 GB at
//! 2048 x 151936. Nothing here holds a whole side: the reference is read one
//! row at a time by seek, and hipfire's row is consumed as soon as it is
//! produced. `--dump` optionally streams hipfire's logits to disk in the same
//! layout so a re-analysis does not have to re-run 2048 forwards.
//!
//! Usage:
//!   maple_kld --model <model.hfq> --tokens <tokens.json> --ref <ref.bin> \
//!       [--dump <hip_logits.bin>] [--per-pos <stats.csv>] [--limit N]

use hipfire_arch_maple::bundle::load_maple_from_hfq;
use hipfire_arch_maple::forward::decode_step;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Instant;

const USAGE: &str = "usage: maple_kld --model <model.hfq> --tokens <tokens.json> \
                     --ref <ref_logits.bin> [--dump <out.bin>] [--per-pos <out.csv>] [--limit N] \\
                     [--kv-mode q8|bf16]";

struct Args {
    model: String,
    tokens: String,
    reference: String,
    dump: Option<String>,
    per_pos: Option<String>,
    limit: Option<usize>,
    /// KV storage tier request, resolved through MAPLE_POLICY. "" = q8.
    /// This is what lets the Q8-KV contribution to the measured KL be
    /// SUBTRACTED rather than assumed: run the same tokens and the same
    /// reference under q8 and bf16 and diff the results.
    kv_mode: String,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args {
        model: String::new(),
        tokens: String::new(),
        reference: String::new(),
        dump: None,
        per_pos: None,
        limit: None,
        kv_mode: String::new(),
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
            "--tokens" => a.tokens = val(),
            "--ref" => a.reference = val(),
            "--dump" => a.dump = Some(val()),
            "--per-pos" => a.per_pos = Some(val()),
            "--limit" => a.limit = Some(val().parse().expect("--limit")),
            "--kv-mode" => a.kv_mode = val(),
            other => panic!("unknown arg {other}\n{USAGE}"),
        }
        i += 2;
    }
    assert!(!a.model.is_empty(), "{USAGE}");
    assert!(!a.tokens.is_empty(), "{USAGE}");
    assert!(!a.reference.is_empty(), "{USAGE}");
    a
}

/// Log-softmax of `logits` into `out`, in f64, max-shifted so the
/// exponentials cannot overflow. Returns `(argmax, sum_of_probs)`; the caller
/// checks the sum against 1.0 as a numerical-hygiene gate.
fn log_softmax_f64(logits: &[f32], out: &mut [f64]) -> (usize, f64) {
    debug_assert_eq!(logits.len(), out.len());
    let mut max = f32::NEG_INFINITY;
    let mut argmax = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > max {
            max = v;
            argmax = i;
        }
    }
    let maxf = max as f64;
    let mut sum = 0.0f64;
    for (o, &v) in out.iter_mut().zip(logits.iter()) {
        let e = (v as f64 - maxf).exp();
        *o = e;
        sum += e;
    }
    let log_sum = sum.ln();
    // out currently holds exp(x - max); convert to log p = (x - max) - log_sum.
    let mut prob_sum = 0.0f64;
    for (o, &v) in out.iter_mut().zip(logits.iter()) {
        prob_sum += *o / sum;
        *o = (v as f64 - maxf) - log_sum;
    }
    (argmax, prob_sum)
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    let args = parse_args();

    // ---- reference header ------------------------------------------------
    let mut ref_file = File::open(&args.reference).expect("open reference logits");
    let mut hdr = [0u8; 8];
    ref_file
        .read_exact(&mut hdr)
        .expect("read reference header");
    let ref_positions = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
    let ref_vocab = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let expect_len = 8u64 + (ref_positions as u64) * (ref_vocab as u64) * 4;
    let actual_len = ref_file.metadata().expect("stat reference").len();
    assert_eq!(
        actual_len, expect_len,
        "reference logits file is {actual_len} bytes but the header \
         (n_positions={ref_positions}, vocab={ref_vocab}) implies {expect_len} — refusing to \
         score against a truncated or mislabelled file"
    );
    eprintln!(
        "Reference: {} ({ref_positions} positions x {ref_vocab} vocab, raw f32)",
        args.reference
    );

    // ---- tokens ----------------------------------------------------------
    let raw = std::fs::read_to_string(&args.tokens).expect("read tokens json");
    let all_tokens: Vec<u32> = serde_json::from_str(&raw).expect("tokens json: expected [u32]");
    let n = args.limit.unwrap_or(all_tokens.len()).min(all_tokens.len());
    assert!(n >= 2, "need at least 2 tokens to score anything");
    assert!(
        n <= ref_positions,
        "asked to score {n} positions but the reference only has {ref_positions}"
    );
    let tokens = &all_tokens[..n];

    // ---- model -----------------------------------------------------------
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch.clone());
    let mut hfq = HfqFile::open(Path::new(&args.model)).expect("open model");
    assert_eq!(
        hfq.arch_id, 15,
        "maple_kld: expected arch_id 15, got {}",
        hfq.arch_id
    );
    let tokenizer =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer from HFQ metadata");

    eprintln!("Loading weights from {}...", args.model);
    let t_load = Instant::now();
    let mut b =
        load_maple_from_hfq(&mut hfq, &mut gpu, n, &args.kv_mode).expect("load maple bundle");
    eprintln!("Loaded in {:.1}s", t_load.elapsed().as_secs_f64());
    eprintln!(
        "maple: hidden={} layers={} experts={}/{} vocab={}",
        b.config.hidden_size,
        b.config.num_hidden_layers,
        b.config.num_experts,
        b.config.num_experts_per_tok,
        b.config.vocab_size,
    );

    // A vocab mismatch means the two sides are not describing the same
    // distribution and every downstream number would be nonsense.
    assert_eq!(
        b.config.vocab_size, ref_vocab,
        "vocab mismatch: model has {} but the reference has {ref_vocab}",
        b.config.vocab_size
    );
    let max_id = tokens.iter().copied().max().unwrap_or(0);
    assert!(
        (max_id as usize) < b.config.vocab_size,
        "REFUSING to score: token id {max_id} is outside this model's vocab {} — these ids came \
         from a DIFFERENT tokenizer",
        b.config.vocab_size
    );
    eprintln!("Tokens: {n} ids from {} (max id {max_id})", args.tokens);
    eprintln!(
        "  decode sample: {:?}",
        tokenizer.decode(&tokens[..tokens.len().min(24)])
    );

    // ---- run -------------------------------------------------------------
    let vocab = ref_vocab;
    let row_bytes = vocab * 4;
    let mut ref_bytes = vec![0u8; row_bytes];
    let mut ref_logits = vec![0f32; vocab];
    let mut lp_ref = vec![0f64; vocab];
    let mut lp_hip = vec![0f64; vocab];

    let mut dump = args
        .dump
        .as_ref()
        .map(|p| BufWriter::with_capacity(1 << 22, File::create(p).expect("create dump")));
    if let Some(w) = dump.as_mut() {
        w.write_all(&(n as u32).to_le_bytes()).expect("dump header");
        w.write_all(&(vocab as u32).to_le_bytes())
            .expect("dump header");
    }
    let mut per_pos = args
        .per_pos
        .as_ref()
        .map(|p| BufWriter::new(File::create(p).expect("create per-pos csv")));
    if let Some(w) = per_pos.as_mut() {
        writeln!(
            w,
            "pos,target,kl_ref_hip,nll_ref,nll_hip,top1_ref,top1_hip,agree"
        )
        .expect("csv header");
    }

    let mut kls: Vec<f64> = Vec::with_capacity(n);
    let mut kl_by_pos: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut total_nll_hip = 0.0f64;
    let mut total_nll_ref = 0.0f64;
    let mut scored = 0usize;
    let mut agree = 0usize;
    let mut compared = 0usize;
    let mut top1_hit_hip = 0usize;
    let mut top1_hit_ref = 0usize;
    let mut worst_sum_err = 0.0f64;
    let mut identical_rows = 0usize;

    b.state.reset(&mut gpu).expect("state reset");
    let t0 = Instant::now();

    for (pos, &tok) in tokens.iter().enumerate() {
        let logits = decode_step(
            &b.config,
            &b.weights,
            &mut b.state,
            &mut gpu,
            tok,
            pos as u32,
        )
        .expect("decode_step");
        assert_eq!(
            logits.len(),
            vocab,
            "decode_step returned {} logits, expected vocab {vocab}",
            logits.len()
        );

        if let Some(w) = dump.as_mut() {
            // Reinterpreting the f32 slice as bytes is sound on any
            // little-endian host, which is the only target this runs on.
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(logits.as_ptr() as *const u8, logits.len() * 4)
            };
            w.write_all(bytes).expect("dump row");
        }

        // Reference row `pos`.
        ref_file
            .seek(SeekFrom::Start(8 + (pos as u64) * (row_bytes as u64)))
            .expect("seek reference");
        ref_file.read_exact(&mut ref_bytes).expect("read ref row");
        for (o, c) in ref_logits.iter_mut().zip(ref_bytes.chunks_exact(4)) {
            *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }

        let (argmax_ref, sum_ref) = log_softmax_f64(&ref_logits, &mut lp_ref);
        let (argmax_hip, sum_hip) = log_softmax_f64(&logits, &mut lp_hip);
        worst_sum_err = worst_sum_err
            .max((sum_ref - 1.0).abs())
            .max((sum_hip - 1.0).abs());

        // KL(reference || hipfire) = sum_v p_ref(v) * (log p_ref(v) - log p_hip(v)).
        let mut kl = 0.0f64;
        let mut bitwise_same = true;
        for v in 0..vocab {
            let d = lp_ref[v] - lp_hip[v];
            if d != 0.0 {
                bitwise_same = false;
            }
            kl += lp_ref[v].exp() * d;
        }
        if bitwise_same {
            identical_rows += 1;
        }
        kls.push(kl);
        kl_by_pos.push((pos, kl));

        compared += 1;
        if argmax_ref == argmax_hip {
            agree += 1;
        }

        // Scoring: row `pos` predicts tokens[pos + 1]; the last row has no
        // target and is scored by neither side.
        let (mut nll_ref, mut nll_hip, mut target) = (f64::NAN, f64::NAN, usize::MAX);
        if pos + 1 < n {
            target = tokens[pos + 1] as usize;
            nll_ref = -lp_ref[target];
            nll_hip = -lp_hip[target];
            if nll_ref.is_finite() && nll_hip.is_finite() {
                total_nll_ref += nll_ref;
                total_nll_hip += nll_hip;
                scored += 1;
                if argmax_ref == target {
                    top1_hit_ref += 1;
                }
                if argmax_hip == target {
                    top1_hit_hip += 1;
                }
            } else {
                eprintln!("  warn: non-finite NLL at pos={pos} target={target}, skipping");
            }
        }

        if let Some(w) = per_pos.as_mut() {
            writeln!(
                w,
                "{pos},{},{kl:.9},{nll_ref:.9},{nll_hip:.9},{argmax_ref},{argmax_hip},{}",
                if target == usize::MAX {
                    -1i64
                } else {
                    target as i64
                },
                u8::from(argmax_ref == argmax_hip),
            )
            .expect("csv row");
        }

        if pos == 0 || (pos + 1) % 128 == 0 || pos + 1 == n {
            let el = t0.elapsed().as_secs_f64();
            let mean_kl = kls.iter().sum::<f64>() / kls.len() as f64;
            let (ph, pr) = if scored > 0 {
                (
                    (total_nll_hip / scored as f64).exp(),
                    (total_nll_ref / scored as f64).exp(),
                )
            } else {
                (f64::NAN, f64::NAN)
            };
            eprintln!(
                "  pos={pos:5} scored={scored:5} ppl_hip={ph:.3} ppl_ref={pr:.3} \
                 mean_kl={mean_kl:.5} agree={:.1}% ({:.2} tok/s)",
                100.0 * agree as f64 / compared as f64,
                (pos + 1) as f64 / el.max(1e-9),
            );
        }
    }

    if let Some(w) = dump.as_mut() {
        w.flush().expect("flush dump");
    }
    if let Some(w) = per_pos.as_mut() {
        w.flush().expect("flush csv");
    }

    // ---- report ----------------------------------------------------------
    let elapsed = t0.elapsed().as_secs_f64();
    let mean_nll_hip = total_nll_hip / scored as f64;
    let mean_nll_ref = total_nll_ref / scored as f64;
    let ppl_hip = mean_nll_hip.exp();
    let ppl_ref = mean_nll_ref.exp();

    let mut sorted = kls.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_kl = kls.iter().sum::<f64>() / kls.len() as f64;

    let mut worst = kl_by_pos.clone();
    worst.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!();
    println!("Model:      {}", args.model);
    println!("Reference:  {}", args.reference);
    println!(
        "Tokens:     {} ({n} positions, one unbroken window from 0)",
        args.tokens
    );
    println!(
        "Scored:     {scored} positions (0..{}), logits[i] vs tokens[i+1]",
        n - 1
    );
    println!();
    println!("PPL (hipfire):    {ppl_hip:.4}   (mean NLL {mean_nll_hip:.6})");
    println!("PPL (reference):  {ppl_ref:.4}   (mean NLL {mean_nll_ref:.6})");
    println!(
        "Top-1 accuracy:   hipfire {:.2}%  reference {:.2}%",
        100.0 * top1_hit_hip as f64 / scored as f64,
        100.0 * top1_hit_ref as f64 / scored as f64,
    );
    println!();
    println!("KL(reference || hipfire) over {compared} positions, full {vocab}-wide vocab:");
    println!("  mean    {mean_kl:.6}");
    println!("  median  {:.6}", percentile(&sorted, 0.50));
    println!("  p90     {:.6}", percentile(&sorted, 0.90));
    println!("  p99     {:.6}", percentile(&sorted, 0.99));
    println!("  min     {:.6}", sorted[0]);
    println!("  max     {:.6}", sorted[sorted.len() - 1]);
    println!(
        "Top-1 agreement (argmax ref == argmax hipfire): {:.2}% ({agree}/{compared})",
        100.0 * agree as f64 / compared as f64
    );
    println!();
    println!("Worst positions by KL:");
    for &(p, k) in worst.iter().take(10) {
        let tgt = if p + 1 < n { tokens[p + 1] as i64 } else { -1 };
        println!("  pos={p:5} kl={k:.6} target={tgt}");
    }
    println!();
    println!("Numerical hygiene:");
    println!("  max |softmax_sum - 1|:   {worst_sum_err:.3e}");
    println!("  bitwise-identical rows:  {identical_rows} / {compared}");
    println!(
        "Elapsed: {elapsed:.1}s ({:.2} tok/s)",
        n as f64 / elapsed.max(1e-9)
    );

    // Self-gate. A KL that is essentially zero on a path that includes a Q8 KV
    // cache is far more likely to mean the two sides are the same buffer than
    // to mean the implementation is perfect; a huge KL with matching argmax
    // points at a scale/temperature mismatch rather than a quality difference.
    println!();
    if identical_rows > 0 {
        println!(
            "*** REGIME: SELF-COMPARISON SUSPECTED — {identical_rows} rows are bitwise identical \
             between the two sides. Verify the reference file is not hipfire's own dump. ***"
        );
    } else if mean_kl < 1e-4 {
        println!(
            "*** REGIME: SUSPICIOUSLY LOW (mean KL {mean_kl:.3e} < 1e-4 nats) on a path with a \
             Q8 KV cache. Check the two sides are genuinely different runs. ***"
        );
    } else if mean_kl > 1.0 && (agree as f64 / compared as f64) > 0.9 {
        println!(
            "*** REGIME: SCALE/TEMPERATURE MISMATCH SUSPECTED — mean KL {mean_kl:.4} > 1 nat but \
             top-1 agrees {:.1}% of the time. That pattern is a logit-scale difference, not a \
             quality difference. ***",
            100.0 * agree as f64 / compared as f64
        );
    } else {
        println!(
            "*** REGIME: PLAUSIBLE QUALITY MEASUREMENT — mean KL {mean_kl:.4} nats, top-1 \
             agreement {:.1}%, no identical rows. ***",
            100.0 * agree as f64 / compared as f64
        );
    }
}
