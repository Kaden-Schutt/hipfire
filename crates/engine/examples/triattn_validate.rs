//! TriAttention reconstruction-correlation harness (arXiv:2604.04921, §3.3).
//!
//! 1. Run forward on a small calibration corpus, collecting pre-RoPE Q at
//!    each FA layer via the triattn tap. Finalize into BandCenters and save
//!    to a `.triattn.bin` sidecar.
//! 2. Run forward on a validation prompt with a full-capture tap that stores
//!    pre-RoPE Q *and* K per token per FA layer.
//! 3. For each FA layer, for each query head, predict attention logits
//!    using the TriAttn scoring function (S_trig + S_norm) and compare
//!    against ground-truth Q·K dot products computed on the host.
//!    Report Pearson correlation per layer (the paper's `r̄` metric).
//!
//! This doesn't exercise the KV cache or eviction path; the scoring is
//! entirely host-side to validate the math end-to-end before we wire a
//! GPU kernel.

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::llama::KvCache;
    use engine::qwen35::{self, DeltaNetState, LayerType, Qwen35Scratch};
    use engine::tokenizer::Tokenizer;
    use engine::triattn::{self, BandCenter, TriAttnCalibState, TriAttnCapture, TriAttnCenters};
    use std::path::Path;
    use std::time::Instant;

    // Per-chunk CSV timing breakdown for calibration optimization sessions.
    // Enable with HIPFIRE_CALIB_PROFILE=1; emits to stderr.
    let calib_profile = std::env::var("HIPFIRE_CALIB_PROFILE")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // ── Parse args ─────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut model_path: Option<String> = None;
    let mut sidecar_path: Option<String> = None;
    let mut corpus_path: Option<String> = None;
    let mut max_tokens: usize = 4000;
    let mut chunk_len: usize = 256;
    let mut validation_prompt = String::from(
        "James Madison wrote Federalist No. 10 arguing that a large republic would curb the effects of factions better than a small one.",
    );
    let mut load_sidecar = false;
    // GPU calibration path is now DEFAULT (Phase 2, 2026-04-28).
    // Rationale: forward_prefill_batch is hard-capped at kv_seq.saturating_sub(4)
    // = 508 tokens per call regardless of `chunk_len`, so production calibration
    // never sees the avg ~3 tok/chunk corpus that drove the original 40% GPU
    // regression. The kernel header for triattn_accumulate.hip annotates a
    // 5-8× speedup on MI300X. Verified Phase 2 R̄ within ±0.005 of CPU baseline.
    // Opt out with --cpu-calib when stress-testing the CPU fallback path
    // (e.g. for FP64 vs FP32 reference comparisons).
    let mut gpu_calib = true;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sidecar" => { sidecar_path = Some(args[i + 1].clone()); i += 2; }
            "--corpus" => { corpus_path = Some(args[i + 1].clone()); i += 2; }
            "--max-tokens" => { max_tokens = args[i + 1].parse().unwrap(); i += 2; }
            "--chunk-len" => { chunk_len = args[i + 1].parse().unwrap(); i += 2; }
            "--val-prompt" => { validation_prompt = args[i + 1].clone(); i += 2; }
            "--load-sidecar" => { load_sidecar = true; i += 1; }
            "--gpu-calib" => { gpu_calib = true; i += 1; }
            "--cpu-calib" => { gpu_calib = false; i += 1; }
            s if !s.starts_with("--") && model_path.is_none() => { model_path = Some(s.to_string()); i += 1; }
            other => {
                eprintln!("unknown arg: {other}\nUsage: triattn_validate <model.mq4> [--sidecar PATH] [--corpus TXT] [--max-tokens N] [--chunk-len N] [--val-prompt STR] [--load-sidecar] [--gpu-calib | --cpu-calib]");
                std::process::exit(1);
            }
        }
    }
    let model_path = model_path.expect("need <model.mq4> positional arg");
    let sidecar_path = sidecar_path.unwrap_or_else(|| format!("{model_path}.triattn.bin"));

    // Calibration corpus: either chunks from --corpus file or 8 built-in
    // sentences (quick-iterate mode).
    //
    // ⚠️ caveat: two of these prompts (Federalist #10, Constitution)
    // lexically overlap the default validation prompt below. When running
    // with no --corpus AND no --val-prompt, expect r̄ to be INFLATED vs a
    // disjoint-corpus run on the same model — that inflation is
    // contamination, not quality. Use needle_eval (or a disjoint --corpus /
    // --val-prompt pair) to actually rank sidecars.
    let builtin_prompts: Vec<String> = [
        "The quick brown fox jumps over the lazy dog.",
        // ⚠ overlaps default --val-prompt (James Madison / Federalist No. 10);
        //   drives a spurious +0.10 r̄ bump vs a disjoint Wikipedia corpus.
        "Federalist No. 10 addresses the problem of factions in a republic.",
        "RoPE encodes positional information via geometric frequencies applied to Q/K.",
        "Speculative decoding verifies many draft tokens in one forward pass.",
        "Attention heads in a transformer specialize during training.",
        // ⚠ "Constitution" also surfaces in the default Madison val-prompt
        //   context; lesser contaminator than the Federalist line above.
        "The Constitution of the United States was ratified in 1788.",
        "Shakespeare wrote thirty-seven plays and over a hundred sonnets.",
        "Pythagoras proved that a squared plus b squared equals c squared.",
    ].iter().map(|s| s.to_string()).collect();
    let calibration_chunks: Vec<String> = if let Some(path) = &corpus_path {
        let text = std::fs::read_to_string(path).expect("read corpus");
        // Chunk by paragraphs first, then merge small paragraphs up to chunk_len.
        let paras: Vec<&str> = text.split("\n\n").map(|p| p.trim()).filter(|p| !p.is_empty()).collect();
        let mut out: Vec<String> = Vec::new();
        for p in paras {
            // Very rough token estimate: 4 chars/token. Keep chunks comfortably
            // under chunk_len tokens so we never have to truncate mid-paragraph.
            let est_tokens = p.len() / 4;
            if est_tokens <= chunk_len {
                out.push(p.to_string());
            } else {
                // Split long paragraphs on sentences.
                let mut cur = String::new();
                for s in p.split(". ") {
                    let s_trim = s.trim();
                    if s_trim.is_empty() { continue; }
                    let cand_len = (cur.len() + s_trim.len() + 2) / 4;
                    if cand_len > chunk_len && !cur.is_empty() {
                        out.push(cur.trim().to_string());
                        cur = String::new();
                    }
                    cur.push_str(s_trim);
                    cur.push_str(". ");
                }
                if !cur.trim().is_empty() { out.push(cur.trim().to_string()); }
            }
        }
        eprintln!("corpus: {} chunks from {path}", out.len());
        out
    } else {
        builtin_prompts
    };
    let calibration_prompts: Vec<&str> = calibration_chunks.iter().map(|s| s.as_str()).collect();

    // Contamination warning: r̄ is NOT a fair cross-corpus ranking metric
    // when the calibration corpus lexically overlaps the validation prompt.
    // The builtin seed prompts include a Federalist No. 10 line which
    // directly overlaps the default validation prompt below. Fire a loud
    // one-shot warning if we detect both defaults in play so operators
    // don't mistake the inflated r̄ for a real quality signal — empirically,
    // a disjoint Wikipedia corpus gives WORSE r̄ but BETTER downstream
    // needle-recall on the same model.
    let default_val_prompt = validation_prompt.contains("James Madison")
        && validation_prompt.contains("Federalist No. 10");
    let using_builtin_corpus = corpus_path.is_none() && !load_sidecar;
    if using_builtin_corpus && default_val_prompt {
        eprintln!(
            "[triattn_validate] ⚠ r̄ CONTAMINATION: builtin calibration corpus contains a Federalist No. 10 line that overlaps the default validation prompt. r̄ reported below will be INFLATED by ~0.10 vs a disjoint corpus (empirically on qwen3.5-9b: builtin r̄≈0.400 vs Wikipedia r̄≈0.304, yet Wikipedia gave BETTER downstream needle recall). For fair sidecar ranking, pass --corpus <disjoint_text> AND --val-prompt <unrelated_text>, and validate against an external long-context recall test."
        );
    }

    // ── Load model ─────────────────────────────────────────────────────
    let hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("weights");

    let kv_seq = 512usize;
    let mut kv = KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
    ).expect("kv q8");
    let mut dn = DeltaNetState::new(&mut gpu, &config).expect("dn");
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");

    // ── Phase 1: Calibrate (or load existing sidecar) ─────────────────
    let centers = if load_sidecar {
        eprintln!("loading sidecar from {sidecar_path}");
        let c = TriAttnCenters::load(Path::new(&sidecar_path)).expect("load sidecar");
        eprintln!(
            "loaded: n_layers={} n_heads={} n_bands={} head_dim={}",
            c.n_layers, c.n_heads, c.n_bands(), c.head_dim,
        );
        c
    } else {
        eprintln!(
            "calibration: {} prompts, {} FA layers × {} heads × {} bands",
            calibration_prompts.len(),
            config.layer_types.iter().filter(|t| **t == LayerType::FullAttention).count(),
            config.n_heads,
            config.head_dim / 2,
        );
        // Default CPU path — faster in practice on short-chunk corpora
        // because the kernel-launch overhead of the HIP reduce dominates
        // the savings (verified 2026-04-19). --gpu-calib opts into the
        // numerically-equivalent HIP path for long-chunk corpora.
        let using_gpu_tap = gpu_calib;
        if using_gpu_tap {
            eprintln!("calibration path: GPU (kernel triattn_accumulate_f32) [opt-in]");
            let gpu_state = triattn::TriAttnCalibStateGpu::new(
                &mut gpu,
                config.n_layers, config.n_heads, config.head_dim,
                config.rope_theta, config.partial_rotary_factor,
            ).expect("alloc GPU calib state");
            triattn::install_tap_gpu(gpu_state);
        } else {
            eprintln!("calibration path: CPU (--cpu-calib)");
            let calib_state = TriAttnCalibState::new(
                config.n_layers, config.n_heads, config.head_dim,
                config.rope_theta, config.partial_rotary_factor,
            );
            triattn::install_tap(calib_state);
        }

        // Pre-tokenize chunks once. Per-chunk encode in the original hot
        // loop accounted for ~44% of wall time on MI300X (measured
        // 2026-04-28 Phase 1 baseline: 456ms encode vs 580ms forward per
        // 508-token chunk). Pre-tokenization runs in parallel across the
        // EPYC cores via rayon (Phase 2.5+: serial pretok still cost 89s
        // on a 100k-token run, ~50% of remaining wall). Stop early at
        // max_tokens worth of effective samples (each chunk contributes
        // min(chunk_tokens, kv_seq-4)).
        // Pretokenize chunks once. Per-chunk encode in the original hot
        // loop accounted for ~44% of wall time on MI300X (Phase 1
        // baseline 2026-04-28). Hoisting it out trades a few seconds of
        // upfront work for a several-minute savings on 1M-token corpora.
        // Hermes/blended corpora contain individual ChatML conversations
        // of 4-10k tokens each; encoding them whole and then slicing to
        // 508 wastes 90%+ of tokenizer wall time, so truncate input
        // text to ~max_eff*5 chars before encoding. Iterates batches
        // until covered_tokens reaches max_tokens — short ChatML
        // chunks average <max_eff effective tokens, so a single
        // parallel pass would underdeliver the corpus the user asked
        // for.
        let pretok_t0 = Instant::now();
        let prompt_tokens: Vec<Vec<u32>> = {
            use rayon::prelude::*;
            let max_eff = kv_seq.saturating_sub(4).max(1);
            let max_chars = max_eff.saturating_mul(5).max(64);
            let mut acc: Vec<Vec<u32>> = Vec::new();
            let mut covered = 0usize;
            let mut start = 0usize;
            while covered < max_tokens && start < calibration_prompts.len() {
                let remaining = max_tokens.saturating_sub(covered);
                let est = remaining.div_ceil(max_eff).max(1);
                // Oversize by 2× to absorb short chunks (ChatML system
                // turns are often 30-100 tokens, well under max_eff).
                let take = (est.saturating_mul(2)).min(calibration_prompts.len() - start);
                if take == 0 { break; }
                let end = start + take;
                let mut new_chunks: Vec<Vec<u32>> = calibration_prompts[start..end]
                    .par_iter()
                    .map(|p| {
                        let s: &str = if p.len() > max_chars {
                            let mut bend = max_chars;
                            while bend > 0 && !p.is_char_boundary(bend) { bend -= 1; }
                            &p[..bend]
                        } else {
                            &p[..]
                        };
                        let mut t = tok.encode(s);
                        if t.len() > max_eff { t.truncate(max_eff); }
                        t
                    })
                    .collect();
                start = end;
                for toks in new_chunks.drain(..) {
                    let effective = toks.len().min(max_eff);
                    if effective == 0 { continue; }
                    covered = covered.saturating_add(effective);
                    acc.push(toks);
                    if covered >= max_tokens { break; }
                }
            }
            acc
        };
        let covered_tokens = prompt_tokens.iter()
            .map(|t| t.len().min(kv_seq.saturating_sub(4)))
            .sum::<usize>();
        let pretok_ms = pretok_t0.elapsed().as_secs_f64() * 1000.0;
        if calib_profile {
            eprintln!(
                "[CALIB_PROFILE] PRETOK total_ms={pretok_ms:.1} n_chunks={} covered_tokens={covered_tokens}",
                prompt_tokens.len(),
            );
        }
        if calib_profile {
            eprintln!("[CALIB_PROFILE] chunk_idx,n_tokens,memset_ms,forward_ms,total_ms,cumulative_tokens");
        }
        let calib_t0 = Instant::now();
        let mut total_tokens = 0usize;
        'outer: for (pi, tokens) in prompt_tokens.iter().enumerate() {
            let chunk_t0 = Instant::now();
            let memset_t0 = Instant::now();
            for buf in kv.k_gpu.iter() { let _ = gpu.hip.memset(&buf.buf, 0, buf.buf.size()); }
            for buf in kv.v_gpu.iter() { let _ = gpu.hip.memset(&buf.buf, 0, buf.buf.size()); }
            for t in &dn.s_matrices { let _ = gpu.hip.memset(&t.buf, 0, t.buf.size()); }
            for t in &dn.s_scales { let _ = gpu.hip.memset(&t.buf, 0, t.buf.size()); }
            for t in &dn.conv_states { let _ = gpu.hip.memset(&t.buf, 0, t.buf.size()); }
            let memset_ms = memset_t0.elapsed().as_secs_f64() * 1000.0;
            let max_len = tokens.len().min(kv_seq.saturating_sub(4));
            let remaining = max_tokens.saturating_sub(total_tokens);
            let take_len = max_len.min(remaining);
            if take_len == 0 { break 'outer; }
            let fwd_t0 = Instant::now();
            qwen35::forward_prefill_batch(
                &mut gpu, &weights, &config, &tokens[..take_len], 0,
                &mut kv, &mut dn, &scratch,
                None, None, None, None,
            ).expect("calib batched forward");
            // Force the device to drain so the timing reflects real GPU work,
            // not just queued kernels — Phase 1 attribution depends on this.
            if calib_profile { let _ = gpu.hip.device_synchronize(); }
            let forward_ms = fwd_t0.elapsed().as_secs_f64() * 1000.0;
            let total_ms = chunk_t0.elapsed().as_secs_f64() * 1000.0;
            total_tokens += take_len;
            if calib_profile {
                eprintln!(
                    "[CALIB_PROFILE] {pi},{take_len},{memset_ms:.3},{forward_ms:.3},{total_ms:.3},{total_tokens}",
                );
            } else if pi % 10 == 0 || pi + 1 == calibration_prompts.len() {
                eprintln!("  chunk {}/{}: cumulative {} tokens", pi + 1, calibration_prompts.len(), total_tokens);
            }
            if total_tokens >= max_tokens { break 'outer; }
        }
        let calib_loop_ms = calib_t0.elapsed().as_secs_f64() * 1000.0;

        eprintln!("total calibration samples: {total_tokens} tokens × FA layers");
        let finalize_t0 = Instant::now();
        let c = if using_gpu_tap {
            let gpu_state = triattn::take_tap_gpu().expect("GPU tap still installed");
            gpu_state.finalize(&mut gpu).expect("finalize GPU calib")
        } else {
            let calib = triattn::take_tap().expect("tap still installed");
            calib.finalize()
        };
        let finalize_ms = finalize_t0.elapsed().as_secs_f64() * 1000.0;
        if calib_profile {
            eprintln!(
                "[CALIB_PROFILE] SUMMARY total_tokens={total_tokens} loop_ms={calib_loop_ms:.1} finalize_ms={finalize_ms:.1} effective_tok_per_sec={:.1}",
                (total_tokens as f64) * 1000.0 / calib_loop_ms.max(1.0),
            );
        }
        c.save(Path::new(&sidecar_path)).expect("save sidecar");
        eprintln!("saved sidecar: {sidecar_path}");
        c
    };

    report_mrl_distribution(&centers);

    // ── Phase 2: Full capture on validation prompt ─────────────────────
    // Fresh KV + DN state so the captured Qs line up with token positions
    // starting at 0.
    for buf in kv.k_gpu.iter() { let _ = gpu.hip.memset(&buf.buf, 0, buf.buf.size()); }
    for buf in kv.v_gpu.iter() { let _ = gpu.hip.memset(&buf.buf, 0, buf.buf.size()); }
    for t in &dn.s_matrices { let _ = gpu.hip.memset(&t.buf, 0, t.buf.size()); }
    for t in &dn.s_scales { let _ = gpu.hip.memset(&t.buf, 0, t.buf.size()); }
    for t in &dn.conv_states { let _ = gpu.hip.memset(&t.buf, 0, t.buf.size()); }

    let cap = TriAttnCapture::new(config.n_heads, config.n_kv_heads, config.head_dim);
    triattn::install_capture(cap);

    let val_tokens = tok.encode(&validation_prompt);
    let val_len = val_tokens.len().min(kv_seq.saturating_sub(4));
    eprintln!("validation: {val_len} tokens");
    for (pos, tid) in val_tokens.iter().take(val_len).enumerate() {
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, *tid, pos,
            &mut kv, &mut dn, &scratch,
        ).expect("val forward");
        triattn::capture_finish_token();
    }
    let capture = triattn::take_capture().expect("capture still installed");
    assert_eq!(capture.q_samples.len(), val_len, "token count mismatch");

    // ── Phase 3: Reconstruction correlation ────────────────────────────
    // For each FA layer, for each query head, predict attention logits via
    // the TriAttn scoring vs ground-truth Q·K dot products. Because we
    // want to cover short-to-long distances evenly the way the paper does,
    // we score against the LAST token's query.
    let last = val_len - 1;
    let p_q = last as f32;
    let n_bands = config.head_dim / 2;
    let d_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;

    // Map kv_head h_kv to the set of query heads sharing it.
    let kv_group = config.n_heads / config.n_kv_heads;

    let mut per_layer_r: Vec<f32> = Vec::new();
    for (fa_pos, &layer_idx) in capture.layer_ids_per_token[last].iter().enumerate() {
        // Ignore layers that aren't FA (those arrays are empty).
        if capture.q_samples[last][fa_pos].is_empty() { continue; }
        let q_last = &capture.q_samples[last][fa_pos];
        assert_eq!(q_last.len(), config.n_heads * config.head_dim);

        let mut per_head_r = Vec::with_capacity(config.n_heads);
        for h in 0..config.n_heads {
            let h_kv = h / kv_group;
            let q_head = &q_last[h * config.head_dim..(h + 1) * config.head_dim];

            // Post-RoPE Q for the last position.
            let q_post = apply_rope(q_head, p_q, d_rot, config.rope_theta);

            let mut predicted = Vec::with_capacity(val_len);
            let mut actual = Vec::with_capacity(val_len);

            for i in 0..val_len {
                if capture.k_samples[i].is_empty() { continue; }
                let k_row = &capture.k_samples[i][fa_pos];
                if k_row.is_empty() { continue; }
                let k_head = &k_row[h_kv * config.head_dim..(h_kv + 1) * config.head_dim];

                // Ground truth: dot product of post-RoPE Q and post-RoPE K
                // (standard attention, no softmax — softmax is monotonic
                // so correlation of logits ≈ correlation of softmaxed).
                let k_post = apply_rope(k_head, i as f32, d_rot, config.rope_theta);
                let mut dot = 0.0f32;
                for d in 0..config.head_dim { dot += q_post[d] * k_post[d]; }
                actual.push(dot);

                // TriAttn prediction (post-RoPE path, uses stored post-RoPE K).
                let centers_slice = center_slice(&centers, layer_idx, h);
                let k_post_bands = triattn::kpost_per_band(&k_post);
                let s = triattn::s_total(centers_slice, &k_post_bands, p_q, |f| centers.omega(f));
                predicted.push(s);
            }

            let r = triattn::pearson(&actual, &predicted);
            per_head_r.push(r);
        }

        let mean_r: f32 = per_head_r.iter().sum::<f32>() / per_head_r.len() as f32;
        let (min_r, max_r) = per_head_r.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &r| (lo.min(r), hi.max(r)));
        let pct_above_05 = per_head_r.iter().filter(|&&r| r > 0.5).count() as f32 * 100.0 / per_head_r.len() as f32;
        eprintln!(
            "layer {layer_idx:2}: r̄={mean_r:.3}  [{min_r:.3}, {max_r:.3}]  {pct_above_05:.0}% of heads > 0.5",
        );
        per_layer_r.push(mean_r);
    }

    if !per_layer_r.is_empty() {
        let overall: f32 = per_layer_r.iter().sum::<f32>() / per_layer_r.len() as f32;
        eprintln!("\n=== overall mean r̄ across FA layers: {overall:.3} ===");
        eprintln!("paper target: ≈0.5 (Figure 3 mean), per-head 0.6-0.9 common; calibration corpus is tiny here");
        eprintln!("note: r̄ plateaus by ~20k calibration tokens under running-mean aggregation — more data does NOT improve r̄ past that point. r̄ is also validation-prompt-dependent; for sidecar quality decisions, validate against a downstream long-context recall test rather than this number alone.");
        if using_builtin_corpus && default_val_prompt {
            eprintln!("⚠ r̄ above is CONTAMINATED by corpus/val-prompt overlap (see warning at startup). Do not compare this number against runs with a disjoint corpus.");
        }
    }

    fn apply_rope(x: &[f32], pos: f32, d_rot: usize, theta: f32) -> Vec<f32> {
        let mut out = x.to_vec();
        for f in 0..(d_rot / 2) {
            let exponent = -2.0f32 * f as f32 / d_rot as f32;
            let w = theta.powf(exponent);
            let angle = w * pos;
            let c = angle.cos();
            let s = angle.sin();
            let xr = x[2 * f];
            let xi = x[2 * f + 1];
            out[2 * f] = xr * c - xi * s;
            out[2 * f + 1] = xr * s + xi * c;
        }
        out
    }

    fn center_slice(c: &TriAttnCenters, layer: usize, head: usize) -> &[BandCenter] {
        let n_bands = c.n_bands();
        let base = layer * c.n_heads * n_bands + head * n_bands;
        &c.centers[base..base + n_bands]
    }

    fn report_mrl_distribution(c: &TriAttnCenters) {
        let n_bands = c.n_bands();
        let mut mrls = Vec::new();
        for l in 0..c.n_layers {
            for h in 0..c.n_heads {
                for f in 0..n_bands {
                    let bc = c.get(l, h, f);
                    if bc.e_abs_q > 1e-10 {
                        mrls.push(bc.mrl());
                    }
                }
            }
        }
        if mrls.is_empty() { return; }
        mrls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean: f32 = mrls.iter().sum::<f32>() / mrls.len() as f32;
        let median = mrls[mrls.len() / 2];
        let pct_above_095 = mrls.iter().filter(|&&r| r > 0.95).count() as f32 * 100.0 / mrls.len() as f32;
        eprintln!(
            "Mean Resultant Length R_f across all (layer, head, band): mean={mean:.3}, median={median:.3}, {pct_above_095:.1}% > 0.95",
        );
        eprintln!("paper target: ~90% of heads R > 0.95 (Figure 2C)");
        eprintln!("note: R_f is validation-prompt-INDEPENDENT (a plus) but has LOW DYNAMIC RANGE — empirically on qwen3.5-9b, calibration corpora of 20k / 50k / 100k Wikipedia tokens all land at R_f≈0.74 despite the 5× size difference. Treat R_f as a floor check (is aggregation converging at all?), not as a ranking metric.");
    }
}
