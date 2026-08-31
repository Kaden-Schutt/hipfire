// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! maple_decode_profile — measure where a Maple-Preview decode token goes.
//!
//! The question this exists to settle is whether arch-15 decode is
//! BANDWIDTH-bound (it reads ~245 MB of ternary active weights plus a dense
//! BF16 lm_head over 151,936 rows every token) or LAUNCH-bound (~16 kernels per
//! layer × 24 layers plus the head). Those two diagnoses imply opposite work —
//! quantize the head vs. capture the graph — so guessing is not an option.
//!
//! Five phases run from one binary, on one machine, in one run, so nothing has
//! to be compared against a figure recorded in an earlier session under unknown
//! load. Phases 2b and 2c are the ones that actually decide the question; 1–3
//! exist to give them a denominator and to keep the instruments honest.
//!
//! Three numbers come out of one binary, on one machine, in one run:
//!
//!   1. `wall`      — untouched decode wall-clock, profiling OFF. The number
//!                    any optimisation has to move.
//!   2. `gpu_sum`   — sum of per-kernel hipEvent times under
//!                    `rdna_compute::profile`. NOTE: that profiler
//!                    SYNCHRONISES after every launch (see the module doc on
//!                    `profile.rs`), so the wall-clock of the profiled phase is
//!                    inflated and must never be quoted as decode speed. Only
//!                    the per-kernel *sum* and the *counts* are meaningful.
//!   3. `d2h`       — the per-token logits download. `decode_step` ends with
//!                    `download_f32(&state.logits)`, which for a 151,936-entry
//!                    vocab is a 608 KB blocking copy per token. It is neither
//!                    a kernel nor launch overhead, so it would otherwise be
//!                    silently attributed to whichever of the two hypotheses
//!                    the reader already believed.
//!
//! `wall - gpu_sum - d2h` is the launch-overhead residual. It is an upper
//! bound, not a point estimate: any real kernel that carries no
//! `profile::begin_timer` also lands in it. Cross-check the launch count
//! against rocprofv3 (`scripts/rocprof-wrap.sh`) before quoting the residual as
//! overhead — that is exactly the blindspot documented in
//! `docs/methodology/rocprof-coverage.md`, and on arch 15 it is not
//! hypothetical: the internal profiler sees 283 of the 478 dispatches
//! rocprofv3 counts, and misses the lm_head, every attention projection, flash
//! attention, softmax and the router GEMVs.
//!
//! Then the two phases that settle it:
//!
//!   2b. the lm_head GEMV priced DIRECTLY (no profiler, one event pair over a
//!       100-launch burst), alongside Q8 / MQ4 / MQ2 arms at the same shape —
//!       so "quantize the head" is a measured number, not an extrapolation.
//!   2c. the marginal cost of one more dispatch, measured IN SITU by injecting
//!       extra null launches into each decode step and fitting the slope. This
//!       stands in for a graph-capture A/B, which cannot be run because Maple's
//!       forward pass has no hipGraph wiring to enable.
//!
//! Usage:
//!   maple_decode_profile --model <model.hfq> [--prompt "..."] [--warmup N]
//!                        [--gen N] [--profile-gen N]
//!
//! `--gen` is the unprofiled timed phase; `--profile-gen` the profiled one.
//! Set `--profile-gen 0` to skip profiling entirely and get a clean wall-clock
//! baseline for the artefact cross-check.

use hipfire_arch_maple::bundle::load_maple_from_hfq;
use hipfire_arch_maple::forward::decode_step;
use hipfire_runtime::hfq::HfqFile;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

struct Args {
    model: String,
    prompt: String,
    warmup: usize,
    gen: usize,
    profile_gen: usize,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut model = None;
    let mut prompt = "Explain, step by step, how a four-stroke engine works.".to_string();
    let mut warmup = 8usize;
    let mut gen = 128usize;
    let mut profile_gen = 8usize;
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
            "--warmup" => {
                warmup = argv[i + 1].parse().expect("--warmup");
                i += 2;
            }
            "--gen" => {
                gen = argv[i + 1].parse().expect("--gen");
                i += 2;
            }
            "--profile-gen" => {
                profile_gen = argv[i + 1].parse().expect("--profile-gen");
                i += 2;
            }
            other => panic!("unknown arg {other}"),
        }
    }
    Args {
        model: model.expect("--model <model.hfq> is required"),
        prompt,
        warmup,
        gen,
        profile_gen,
    }
}

/// Greedy argmax over the logits row. Kept on the host so the harness measures
/// the same `decode_step` boundary a real sampler would.
fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best as u32
}

/// Bucket a kernel name into the coarse phases the verdict is stated in.
///
/// The buckets are chosen so that "is the head 72% of the time, matching its
/// 72% of the bytes?" is answerable by reading one row. `lm_head` is
/// deliberately its OWN bucket rather than being folded into a generic
/// "gemv" bucket — folding it in is how a 622 MB/token read hides.
fn bucket(kernel: &str, category: &str) -> &'static str {
    let k = kernel.to_ascii_lowercase();
    if k.contains("lm_head") {
        return "lm_head";
    }
    if k.contains("moe") || k.contains("silu") || k.contains("expert") {
        return "moe_expert";
    }
    if k.contains("topk") || k.contains("softmax") || k.contains("router") {
        return "router";
    }
    if k.contains("attention")
        || k.contains("attn")
        || k.contains("flash")
        || k.contains("kv_cache")
    {
        return "attention";
    }
    if k.contains("rope") {
        return "rope";
    }
    if k.contains("norm") {
        return "norm";
    }
    if k.contains("embed") {
        return "embed";
    }
    if k.contains("gemv") || k.contains("gemm") {
        return "attn_proj_gemv";
    }
    match category {
        "gemv" | "gemm" => "attn_proj_gemv",
        _ => "other",
    }
}

#[derive(Clone, Copy)]
enum HeadArm {
    /// The checkpoint's own BF16 head, driven through the same `weight_gemv`
    /// the forward pass uses — not a re-implementation of it.
    Real,
    Q8,
    Mq4,
    Mq2,
}

/// Median ms per lm_head-shaped GEMV, amortised over a launch burst.
///
/// Cache honesty: the smallest arm here (MQ2, ~87 MB) is still far larger than
/// gfx1151's 32 MB MALL, so a single re-streamed buffer measures DRAM, not
/// cache. No slot rotation is needed at these shapes — but that is a property
/// of `vocab × hidden` being enormous, not a general licence.
fn time_head(
    gpu: &mut rdna_compute::Gpu,
    b: &mut hipfire_arch_maple::bundle::MapleBundle,
    arm: HeadArm,
) -> f64 {
    let vocab = b.config.vocab_size;
    let hidden = b.config.hidden_size;
    let n_warm = 20usize;
    let n_iter = 100usize;

    // Deterministic filler: the kernels are data-independent (no branches on
    // weight values), so the bit pattern only has to be legal, not meaningful.
    let synth = |gpu: &mut rdna_compute::Gpu, per_group: usize, group: usize| {
        let n = vocab * hidden / group * per_group;
        let mut v = vec![0u8; n];
        for (i, byte) in v.iter_mut().enumerate() {
            *byte = (i % 251) as u8;
        }
        gpu.upload_raw(&v, &[n]).expect("upload synthetic head")
    };

    let x = &b.state.final_norm_buf;
    let y = &b.state.logits;
    let (w, run): (Option<rdna_compute::GpuTensor>, u8) = match arm {
        HeadArm::Real => (None, 0),
        HeadArm::Q8 => (Some(synth(gpu, 34, 32)), 1),
        HeadArm::Mq4 => (Some(synth(gpu, 160, 256)), 2),
        HeadArm::Mq2 => (Some(synth(gpu, 72, 256)), 3),
    };
    let mut once = |gpu: &mut rdna_compute::Gpu| match run {
        0 => {
            hipfire_runtime::llama::weight_gemv(gpu, &b.weights.lm_head, x, y).expect("lm_head");
        }
        1 => {
            gpu.gemv_q8_0(w.as_ref().unwrap(), x, y, vocab, hidden)
                .expect("q8 head");
        }
        2 => {
            gpu.gemv_mq4g256_lloyd(w.as_ref().unwrap(), x, y, vocab, hidden)
                .expect("mq4 head");
        }
        _ => {
            gpu.gemv_mq2g256_lloyd(w.as_ref().unwrap(), x, y, vocab, hidden)
                .expect("mq2 head");
        }
    };
    for _ in 0..n_warm {
        once(gpu);
    }
    // One event pair around the whole burst: no per-launch sync, so this is the
    // pipelined steady-state cost, directly comparable to the decode loop.
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter {
        once(gpu);
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 / n_iter as f64;
    let _ = gpu.hip.event_destroy(start);
    let _ = gpu.hip.event_destroy(stop);
    ms
}

/// One row of the head table. `real_ms` is the BF16 head's measured cost, so
/// the last column is `median - real + this` — an AMDAHL PROJECTION that holds
/// every other kernel constant and swaps only the head. It is not a measured
/// end-to-end speed, and quoting it as one is how a 40%-of-time component turns
/// into a promised 2x. Pass `real_ms == ms` for the BF16 row itself.
fn report_head(name: &str, bytes: f64, ms: f64, median_ms: f64, real_ms: f64) {
    let projected = median_ms - real_ms + ms;
    println!(
        "{name:<26} {:>10.1} {ms:>10.3} {:>9.1} {:>9.1}% {:>12.1}",
        bytes / 1e6,
        bytes / 1e9 / (ms / 1e3),
        ms / median_ms * 100.0,
        1e3 / projected,
    );
}

fn main() {
    let args = parse_args();
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("arch: {}", gpu.arch);
    let mut hfq = HfqFile::open(Path::new(&args.model)).expect("open model");
    assert_eq!(hfq.arch_id, 15, "expected arch_id 15, got {}", hfq.arch_id);

    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let text = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        args.prompt
    );
    let prompt_toks = tokenizer.encode(&text);
    // +160 covers phase 2c's four 40-token dispatch-cost probes. Sizing the KV
    // from the headline token counts alone silently under-allocates and the
    // first symptom is an illegal-access fault in an unrelated kernel.
    let max_seq = prompt_toks.len() + args.warmup + args.gen + args.profile_gen + 160 + 64;
    let mut b = load_maple_from_hfq(&mut hfq, &mut gpu, max_seq, "").expect("load maple bundle");
    eprintln!(
        "maple: hidden={} layers={} experts={}/{} moe_inter={} vocab={} max_seq={}",
        b.config.hidden_size,
        b.config.num_hidden_layers,
        b.config.num_experts,
        b.config.num_experts_per_tok,
        b.config.moe_intermediate_size,
        b.config.vocab_size,
        max_seq,
    );

    // ── Prefill ─────────────────────────────────────────────────────────────
    let mut logits = Vec::new();
    if hipfire_arch_maple::forward::forward_batch_supported(&b.weights) {
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

    // ── Warmup: kernel JIT, DPM clock ramp, cache fill ──────────────────────
    // Discarded entirely. The first few tokens of any run on this box are
    // depressed by module load and by the clocks still being at idle DPM.
    for _ in 0..args.warmup {
        let tok = argmax(&logits);
        logits = decode_step(&b.config, &b.weights, &mut b.state, &mut gpu, tok, pos)
            .expect("decode_step (warmup)");
        pos += 1;
    }

    // ── Phase 1: unprofiled wall clock ──────────────────────────────────────
    // Per-token samples are retained so the report can quote a median as well
    // as a mean: on a machine with an unrelated CPU hog the mean is dragged by
    // a handful of scheduler stalls that no GPU change can fix.
    let mut per_tok_ms: Vec<f64> = Vec::with_capacity(args.gen);
    let t_wall = Instant::now();
    for _ in 0..args.gen {
        let tok = argmax(&logits);
        let t0 = Instant::now();
        logits = decode_step(&b.config, &b.weights, &mut b.state, &mut gpu, tok, pos)
            .expect("decode_step");
        per_tok_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        pos += 1;
    }
    let wall_s = t_wall.elapsed().as_secs_f64();
    per_tok_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = per_tok_ms[per_tok_ms.len() / 2];
    let p10 = per_tok_ms[per_tok_ms.len() / 10];
    let mean_ms = wall_s * 1e3 / args.gen as f64;
    println!("\n=== phase 1: decode wall clock (profiling OFF) ===");
    println!("tokens          : {}", args.gen);
    println!("wall            : {wall_s:.3} s");
    println!(
        "ms/token mean   : {mean_ms:.3}  ({:.1} tok/s)",
        1e3 / mean_ms
    );
    println!(
        "ms/token median : {median_ms:.3}  ({:.1} tok/s)",
        1e3 / median_ms
    );
    println!("ms/token p10    : {p10:.3}  ({:.1} tok/s)", 1e3 / p10);

    // ── Phase 2: price the logits D2H on its own ────────────────────────────
    // `decode_step` ends in `download_f32(&state.logits)`. That is a blocking
    // 4·vocab-byte copy that belongs to NEITHER hypothesis, so it is measured
    // separately rather than being left to inflate whichever bucket the reader
    // is rooting for. Timed after the compute phase so the clocks are hot.
    let n_d2h = 200usize;
    let t_d2h = Instant::now();
    for _ in 0..n_d2h {
        let _ = gpu.download_f32(&b.state.logits).expect("logits d2h");
    }
    let d2h_ms = t_d2h.elapsed().as_secs_f64() * 1e3 / n_d2h as f64;
    println!(
        "\n=== phase 2: logits D2H (vocab={}) ===",
        b.config.vocab_size
    );
    println!(
        "d2h/token       : {d2h_ms:.3} ms  ({:.1}% of median token)",
        d2h_ms / median_ms * 100.0
    );

    // ── Phase 2b: price the lm_head directly, and price its alternatives ────
    //
    // This is the decisive test of the "decode is bandwidth-bound and the head
    // is 72% of the bytes" hypothesis, and it deliberately does NOT go through
    // either profiler. One event pair spans the whole loop, so there is no
    // per-launch synchronisation and no instrument offset to argue about; the
    // only assumption is that 300 back-to-back launches of the same kernel
    // amortise the dispatch cost, which is the same assumption every GB/s
    // number in this tree already makes.
    //
    // The quantized arms use SYNTHETIC weights at the real (vocab × hidden)
    // shape. They measure what the hardware would do with that many bytes in
    // that layout — which is the whole question for "should we quantize the
    // head" — and say nothing about output quality, which is a separate KL
    // measurement. Data-independent kernels, so synthetic values are legitimate
    // here in a way they would not be for an accuracy claim.
    let vocab = b.config.vocab_size;
    let hidden = b.config.hidden_size;
    println!("\n=== phase 2b: lm_head GEMV priced directly (m={vocab} k={hidden}) ===");
    println!(
        "{:<26} {:>10} {:>10} {:>9} {:>10} {:>12}",
        "head format", "MB", "ms/call", "GB/s", "% median", "tok/s if..."
    );
    let head_bytes_bf16 = (vocab * hidden * 2) as f64;
    let real_ms = time_head(&mut gpu, &mut b, HeadArm::Real);
    report_head(
        "BF16 (in-model)",
        head_bytes_bf16,
        real_ms,
        median_ms,
        real_ms,
    );

    // Synthetic arms. Byte-per-weight: Q8_0 34 B/32; MQ4-Lloyd 160 B/256;
    // MQ2-Lloyd 72 B/256 (the format Maple's own experts already use).
    for (name, arm, bytes_per_weight) in [
        ("Q8_0", HeadArm::Q8, 34.0 / 32.0),
        ("MQ4-Lloyd (g256)", HeadArm::Mq4, 160.0 / 256.0),
        ("MQ2-Lloyd (g256)", HeadArm::Mq2, 72.0 / 256.0),
    ] {
        let ms = time_head(&mut gpu, &mut b, arm);
        report_head(
            name,
            (vocab * hidden) as f64 * bytes_per_weight,
            ms,
            median_ms,
            real_ms,
        );
    }
    println!(
        "reference: BF16 head is {:.1}% of the token's bytes and {:.1}% of its time",
        head_bytes_bf16 / (head_bytes_bf16 + 241e6) * 100.0,
        real_ms / median_ms * 100.0
    );

    // ── Phase 2c: marginal cost of one more dispatch, IN SITU ───────────────
    //
    // Maple's forward pass has no hipGraph wiring (nothing in the
    // `decode_step` → `execute_steps` chain consults `HIPFIRE_GRAPH`), so the
    // launch-bound hypothesis cannot be tested by capturing the graph and
    // diffing — there is nothing to capture yet. This is the next best thing
    // and it needs no new kernel: inject `k` extra near-empty dispatches into
    // each decode step and fit ms/token against k. The slope is what ONE
    // dispatch costs inside the real, dependency-chained, already-pipelined
    // decode stream.
    //
    // Read it as an UPPER BOUND on what removing a launch can buy. The
    // injected nulls are appended to the step rather than interleaved between
    // dependent kernels, so if the host is comfortably ahead of the GPU (it is
    // — ~1 ms of submit against ~7.5 ms of GPU work) a burst of nulls exposes
    // host submit cost that a real, spread-out launch has hidden. Slope ×
    // launches/token therefore over-states the graph-capture prize; it does
    // not under-state it, which is the direction that matters before spending
    // a week on capture.
    let one = gpu.upload_f32(&[1.0], &[1]).expect("null a");
    let zero = gpu.upload_f32(&[0.0], &[1]).expect("null b");
    println!("\n=== phase 2c: marginal dispatch cost in situ ===");
    println!(
        "{:>8} {:>12} {:>14}",
        "extra/tok", "ms/token", "delta vs k=0"
    );
    let mut fit: Vec<(f64, f64)> = Vec::new();
    for k in [0usize, 120, 240, 480] {
        let n_probe = 40usize;
        let mut samples = Vec::with_capacity(n_probe);
        for _ in 0..n_probe {
            let tok = argmax(&logits);
            let t0 = Instant::now();
            logits = decode_step(&b.config, &b.weights, &mut b.state, &mut gpu, tok, pos)
                .expect("decode_step (dispatch probe)");
            for _ in 0..k {
                gpu.add_inplace_f32(&one, &zero).expect("null launch");
            }
            // The next `decode_step` ends in a blocking logits D2H, so the
            // injected launches are drained inside the following sample. Sync
            // here instead, so each sample owns its own cost.
            gpu.hip.device_synchronize().expect("sync");
            samples.push(t0.elapsed().as_secs_f64() * 1e3);
            pos += 1;
            if pos as usize + 8 >= b.state.max_seq {
                break;
            }
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = samples[samples.len() / 2];
        fit.push((k as f64, m));
        println!("{k:>8} {m:>12.3} {:>14.3}", m - fit[0].1);
    }
    // Least-squares slope through the four points.
    let n = fit.len() as f64;
    let sx: f64 = fit.iter().map(|p| p.0).sum();
    let sy: f64 = fit.iter().map(|p| p.1).sum();
    let sxx: f64 = fit.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = fit.iter().map(|p| p.0 * p.1).sum();
    let slope_ms = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    println!(
        "slope           : {:.3} us per extra dispatch",
        slope_ms * 1e3
    );
    println!(
        "→ 478 real dispatches ≤ {:.3} ms/token = {:.1}% of the {median_ms:.3} ms token",
        slope_ms * 478.0,
        slope_ms * 478.0 / median_ms * 100.0
    );

    if args.profile_gen == 0 {
        println!("\n(--profile-gen 0: skipping per-kernel profile)");
        return;
    }

    // ── Phase 3: per-kernel profile ─────────────────────────────────────────
    // The wall clock of THIS phase is meaningless (the profiler syncs on every
    // launch). Counts and the per-kernel time sum are what it is for.
    rdna_compute::profile::start();
    let t_prof = Instant::now();
    for _ in 0..args.profile_gen {
        let tok = argmax(&logits);
        logits = decode_step(&b.config, &b.weights, &mut b.state, &mut gpu, tok, pos)
            .expect("decode_step (profiled)");
        pos += 1;
    }
    let prof_wall_ms = t_prof.elapsed().as_secs_f64() * 1e3 / args.profile_gen as f64;
    let entries = rdna_compute::profile::stop().unwrap_or_default();

    let n = args.profile_gen as f64;
    #[derive(Default, Clone)]
    struct Agg {
        count: usize,
        us: f64,
        bytes: u128,
        category: String,
    }
    let mut by_kernel: HashMap<String, Agg> = HashMap::new();
    let mut by_bucket: HashMap<&'static str, Agg> = HashMap::new();
    for e in &entries {
        let a = by_kernel.entry(e.kernel.to_string()).or_default();
        a.count += 1;
        a.us += e.time_us;
        a.bytes += e.bytes as u128;
        a.category = e.category.to_string();
        let bkt = by_bucket.entry(bucket(e.kernel, e.category)).or_default();
        bkt.count += 1;
        bkt.us += e.time_us;
        bkt.bytes += e.bytes as u128;
    }
    let total_us: f64 = entries.iter().map(|e| e.time_us).sum();
    let total_bytes: u128 = entries.iter().map(|e| e.bytes as u128).sum();

    println!(
        "\n=== phase 3: per-kernel profile ({} tokens) ===",
        args.profile_gen
    );
    println!("NOTE: the profiler syncs after every launch; its wall clock");
    println!(
        "      ({prof_wall_ms:.3} ms/tok vs {median_ms:.3} unprofiled) is instrument, not signal."
    );
    println!("launches/token  : {:.1}", entries.len() as f64 / n);
    println!("distinct kernels: {}", by_kernel.len());
    println!("gpu_sum/token   : {:.3} ms", total_us / n / 1e3);
    println!(
        "bytes/token     : {:.1} MB  (implied {:.1} GB/s at the unprofiled median)",
        total_bytes as f64 / n / 1e6,
        total_bytes as f64 / n / 1e9 / (median_ms / 1e3)
    );
    let residual = median_ms - total_us / n / 1e3 - d2h_ms;
    println!(
        "residual        : {residual:.3} ms/token  ({:.1}% of median) = median - gpu_sum - d2h",
        residual / median_ms * 100.0
    );
    println!("                  upper bound on launch overhead + untimed kernels");

    // Bucket table — this is the row the verdict is read off.
    println!("\n--- by bucket ---");
    println!(
        "{:<16} {:>7} {:>10} {:>9} {:>12} {:>9}",
        "bucket", "launch", "ms/tok", "% median", "MB/tok", "% bytes"
    );
    let mut buckets: Vec<_> = by_bucket.into_iter().collect();
    buckets.sort_by(|a, b| b.1.us.partial_cmp(&a.1.us).unwrap());
    for (name, a) in &buckets {
        println!(
            "{:<16} {:>7.1} {:>10.3} {:>8.1}% {:>12.1} {:>8.1}%",
            name,
            a.count as f64 / n,
            a.us / n / 1e3,
            a.us / n / 1e3 / median_ms * 100.0,
            a.bytes as f64 / n / 1e6,
            a.bytes as f64 / total_bytes as f64 * 100.0
        );
    }

    let mut rows: Vec<(String, Agg)> = by_kernel.into_iter().collect();

    println!("\n--- top 20 kernels by total time ---");
    rows.sort_by(|a, b| b.1.us.partial_cmp(&a.1.us).unwrap());
    println!(
        "{:<52} {:>7} {:>9} {:>8} {:>9} {:>9}",
        "kernel", "n/tok", "ms/tok", "% total", "us/call", "GB/s"
    );
    for (k, a) in rows.iter().take(20) {
        let s = a.us / 1e6;
        println!(
            "{:<52} {:>7.1} {:>9.3} {:>7.1}% {:>9.1} {:>9.1}",
            k,
            a.count as f64 / n,
            a.us / n / 1e3,
            a.us / total_us * 100.0,
            a.us / a.count as f64,
            if s > 0.0 {
                a.bytes as f64 / 1e9 / s
            } else {
                0.0
            }
        );
    }

    println!("\n--- top 20 kernels by launch count ---");
    rows.sort_by_key(|(_, a)| std::cmp::Reverse(a.count));
    println!(
        "{:<52} {:>7} {:>9} {:>9}",
        "kernel", "n/tok", "ms/tok", "us/call"
    );
    for (k, a) in rows.iter().take(20) {
        println!(
            "{:<52} {:>7.1} {:>9.3} {:>9.1}",
            k,
            a.count as f64 / n,
            a.us / n / 1e3,
            a.us / a.count as f64
        );
    }
}
