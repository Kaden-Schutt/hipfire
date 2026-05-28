// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Native Qwen3.5 MTP serving-style greedy benchmark.
//!
//! Build with:
//!   cargo build --release --features deltanet --example mtp_probe -p hipfire-runtime
//!
//! Requires an HFQ produced with `hipfire-quantize --include-mtp`.
//!
//! Usage:
//!   mtp_probe <model-with-mtp.hfq> [prompt] [steps] [mtp_batch]
//!   mtp_probe <model-with-mtp.hfq> --probe-only [prompt] [steps] [mtp_batch]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn print_usage() {
    println!(
        "\
Usage:
  mtp_probe <model-with-mtp.hfq> [prompt] [steps] [mtp_batch]
  mtp_probe <model-with-mtp.hfq> --probe-only [prompt] [steps] [mtp_batch]

Arguments:
  model-with-mtp.hfq  HFQ model produced with hipfire-quantize --include-mtp
  prompt              Prompt text to seed greedy decode [default: Hello]
  steps               Maximum generated tokens [default: 64]
  mtp_batch           Proposal batch size, clamped to at least 1 [default: 4]

Options:
  --probe-only        Run the simpler probe path instead of production-step MTP
  -h, --help          Show this help text

Environment:
  HIPFIRE_GRAPH=0       Forced by mtp_probe until MTP graph replay is proven
  HIPFIRE_MTP_PROBE_CTX  KV-cache sequence length [default: 1024]
  HIPFIRE_MTP_PROBE_F32_KV=1
                          Use F32 MTP KV cache instead of Q8
  HIPFIRE_MTP_PROBE_NORM_HIDDEN=1
                          Feed post-output-norm target hidden to MTP diagnostics
  HIPFIRE_MTP_TRACE=1     Print proposals/acceptance per speculative cycle
  HIPFIRE_MTP_SEED_PROBE=1
                          Probe target-next-token seeding after prompt prefill
  HIPFIRE_MTP_SEED_POS_DELTA=<n>
                          Add n to the seed-probe MTP position [default: 0]
  HIPFIRE_MTP_DUMP_PREFIX=<path>
                          Dump target hidden and MTP logits after prompt prefill
"
    );
}

#[cfg(feature = "deltanet")]
fn top_k_ids(logits: &[f32], k: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    let keep = k.min(ids.len());
    if keep == 0 {
        return Vec::new();
    }
    ids.select_nth_unstable_by(keep.saturating_sub(1), |&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ids.truncate(keep);
    ids.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ids
}

#[cfg(feature = "deltanet")]
fn top_k_overlap(a: &[usize], b: &[usize]) -> usize {
    a.iter().filter(|id| b.contains(id)).count()
}

#[cfg(feature = "deltanet")]
fn argmax_one(
    gpu: &mut rdna_compute::Gpu,
    logits: &rdna_compute::GpuTensor,
    result: &rdna_compute::GpuTensor,
    vocab: usize,
) -> u32 {
    gpu.argmax_f32_batched(logits, result, vocab, 1)
        .expect("gpu argmax");
    let mut out = [0i32];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &result.buf)
        .expect("download argmax");
    out[0] as u32
}

#[cfg(feature = "deltanet")]
#[allow(clippy::too_many_arguments)]
fn probe_only_step(
    gpu: &mut rdna_compute::Gpu,
    target: &mut hipfire_arch_qwen35::speculative::ModelSlot,
    mtp_kv_cache: &mut hipfire_runtime::llama::KvCache,
    mtp_scratch: &hipfire_arch_qwen35::qwen35::Qwen35MtpScratch,
    position: usize,
    block_size: usize,
    use_norm_hidden: bool,
) -> hipfire_arch_qwen35::speculative::MtpSpecStepResult {
    use hipfire_arch_qwen35::qwen35;
    let vocab = target.config.vocab_size;
    let t_mtp_start = std::time::Instant::now();
    let mut drafted = Vec::with_capacity(block_size);
    drafted.push(argmax_one(
        gpu,
        &mtp_scratch.logits,
        &mtp_scratch.argmax,
        vocab,
    ));
    for j in 1..block_size {
        let prev = drafted[j - 1];
        qwen35::mtp_forward_dense_gpu_with_scratch(
            gpu,
            &target.weights,
            &target.config,
            prev,
            &mtp_scratch.x,
            position + j - 1,
            j,
            mtp_kv_cache,
            mtp_scratch,
        )
        .expect("probe-only mtp proposal");
        drafted.push(argmax_one(
            gpu,
            &mtp_scratch.logits,
            &mtp_scratch.argmax,
            vocab,
        ));
    }
    let mtp_propose_us = t_mtp_start.elapsed().as_micros();

    let t_replay_start = std::time::Instant::now();
    let mut mtp_repair_us = 0u128;
    let mut accepted = 0usize;
    let mut committed = Vec::with_capacity(block_size);
    for (j, &proposal) in drafted.iter().enumerate() {
        let target_next = argmax_one(gpu, &target.scratch.logits, &mtp_scratch.argmax, vocab);
        let hit = proposal == target_next;
        let tok = if hit {
            accepted += 1;
            proposal
        } else {
            target_next
        };
        let pos = position + committed.len();
        target
            .forward_no_graph(gpu, tok, pos)
            .expect("probe-only target decode");
        let t_mtp = std::time::Instant::now();
        qwen35::mtp_forward_dense_gpu_with_scratch(
            gpu,
            &target.weights,
            &target.config,
            tok,
            if use_norm_hidden {
                &target.scratch.tmp
            } else {
                &target.scratch.x
            },
            pos,
            j,
            mtp_kv_cache,
            mtp_scratch,
        )
        .expect("probe-only mtp repair");
        mtp_repair_us += t_mtp.elapsed().as_micros();
        committed.push(tok);
        if !hit {
            break;
        }
    }
    let replay_us = t_replay_start.elapsed().as_micros();

    hipfire_arch_qwen35::speculative::MtpSpecStepResult {
        accepted,
        bonus_token: *committed.last().unwrap_or(&0),
        drafted,
        committed,
        proposal_count: block_size,
        mtp_propose_us,
        verify_us: 0,
        replay_us,
        mtp_repair_us,
    }
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35MtpScratch, Qwen35Scratch};
    use hipfire_arch_qwen35::speculative::{
        spec_step_mtp_greedy, DeltaNetSnapshot, ModelSlot, ModelSlotConfig, VerifyScratch,
    };
    use hipfire_runtime::llama::KvCache;
    use std::path::Path;
    use std::time::{Duration, Instant};

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return;
    }
    let probe_only = if let Some(idx) = args.iter().position(|a| a == "--probe-only") {
        args.remove(idx);
        true
    } else {
        false
    };
    let model_path = args.get(0).unwrap_or_else(|| {
        eprintln!(
            "Usage: mtp_probe <model-with-mtp.hfq> [--probe-only] [prompt] [steps] [mtp_batch]"
        );
        std::process::exit(1);
    });
    let prompt = args.get(1).map(String::as_str).unwrap_or("Hello");
    let steps = args
        .get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(64);
    let mtp_batch = args
        .get(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4)
        .max(1);

    std::env::set_var("HIPFIRE_GRAPH", "0");
    eprintln!("mtp_probe: forcing HIPFIRE_GRAPH=0 for the dense MTP contract");

    let kv_seq = std::env::var("HIPFIRE_MTP_PROBE_CTX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1024);
    let use_norm_hidden = std::env::var("HIPFIRE_MTP_PROBE_NORM_HIDDEN")
        .ok()
        .as_deref()
        == Some("1");
    let use_f32_mtp_kv = std::env::var("HIPFIRE_MTP_PROBE_F32_KV").ok().as_deref() == Some("1");
    let trace_mtp = std::env::var("HIPFIRE_MTP_TRACE").ok().as_deref() == Some("1");
    let seed_probe = std::env::var("HIPFIRE_MTP_SEED_PROBE").ok().as_deref() == Some("1");
    let seed_pos_delta = std::env::var("HIPFIRE_MTP_SEED_POS_DELTA")
        .ok()
        .and_then(|s| s.parse::<isize>().ok())
        .unwrap_or(0);
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut target = ModelSlot::load(
        &mut gpu,
        Path::new(model_path),
        "target+mtp",
        ModelSlotConfig {
            max_seq: kv_seq,
            ..ModelSlotConfig::default()
        },
    )
    .expect("load model");
    let tokenizer = target.load_tokenizer().expect("parse tokenizer");
    let prompt_tokens = tokenizer.encode(prompt);
    if prompt_tokens.is_empty() {
        eprintln!("prompt encoded to zero tokens");
        std::process::exit(2);
    }
    let mtp_layers = target
        .weights
        .mtp
        .as_ref()
        .map(|m| m.layers.len())
        .unwrap_or(0);
    if mtp_layers == 0 {
        eprintln!("model has no loaded MTP weights; re-quantize with --include-mtp");
        std::process::exit(2);
    }

    eprintln!(
        "mtp_probe: dim={} layers={} mtp_layers={} prompt_tokens={} mtp_batch={} mode={}",
        target.config.dim,
        target.config.n_layers,
        mtp_layers,
        prompt_tokens.len(),
        mtp_batch,
        if probe_only {
            "probe-only"
        } else {
            "production-step"
        },
    );

    let mut mtp_kv_cache = if use_f32_mtp_kv {
        KvCache::new_gpu(
            &mut gpu,
            mtp_layers,
            target.config.n_kv_heads,
            target.config.head_dim,
            kv_seq,
        )
        .expect("mtp kv")
    } else {
        KvCache::new_gpu_q8(
            &mut gpu,
            mtp_layers,
            target.config.n_kv_heads,
            target.config.head_dim,
            kv_seq,
        )
        .expect("mtp kv")
    };
    let mtp_scratch = Qwen35MtpScratch::new(&mut gpu, &target.config).expect("mtp scratch");

    let mut prompt_target_time = Duration::ZERO;
    let mut prompt_mtp_time = Duration::ZERO;
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        let t0 = Instant::now();
        target
            .forward_no_graph(&mut gpu, tok, pos)
            .expect("target prompt prefill");
        gpu.hip.device_synchronize().expect("sync target prompt");
        prompt_target_time += t0.elapsed();

        let t0 = Instant::now();
        qwen35::mtp_forward_dense_gpu_with_scratch(
            &mut gpu,
            &target.weights,
            &target.config,
            tok,
            if use_norm_hidden {
                &target.scratch.tmp
            } else {
                &target.scratch.x
            },
            pos,
            0,
            &mut mtp_kv_cache,
            &mtp_scratch,
        )
        .expect("mtp prompt prefill");
        gpu.hip.device_synchronize().expect("sync mtp prompt");
        prompt_mtp_time += t0.elapsed();
    }

    if let Ok(prefix) = std::env::var("HIPFIRE_MTP_DUMP_PREFIX") {
        let hidden = gpu
            .download_f32(&target.scratch.x)
            .expect("download target hidden");
        let hidden_norm = gpu
            .download_f32(&target.scratch.tmp)
            .expect("download target norm hidden");
        let mtp_logits = gpu
            .download_f32(&mtp_scratch.logits)
            .expect("download mtp logits");
        let hidden_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(hidden.as_ptr() as *const u8, hidden.len() * 4) };
        let hidden_norm_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(hidden_norm.as_ptr() as *const u8, hidden_norm.len() * 4)
        };
        let logits_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(mtp_logits.as_ptr() as *const u8, mtp_logits.len() * 4)
        };
        std::fs::write(format!("{prefix}.target_hidden.f32"), hidden_bytes)
            .expect("write target hidden");
        std::fs::write(
            format!("{prefix}.target_hidden_norm.f32"),
            hidden_norm_bytes,
        )
        .expect("write target norm hidden");
        std::fs::write(format!("{prefix}.mtp_logits.f32"), logits_bytes).expect("write mtp logits");
        std::fs::write(
            format!("{prefix}.prompt_tokens.txt"),
            prompt_tokens
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        )
        .expect("write prompt tokens");
    }

    if seed_probe {
        let seed = argmax_one(
            &mut gpu,
            &target.scratch.logits,
            &mtp_scratch.argmax,
            target.config.vocab_size,
        );
        let pos = (prompt_tokens.len() as isize + seed_pos_delta).max(0) as usize;
        qwen35::mtp_forward_dense_gpu_with_scratch(
            &mut gpu,
            &target.weights,
            &target.config,
            seed,
            if use_norm_hidden {
                &target.scratch.tmp
            } else {
                &target.scratch.x
            },
            pos,
            0,
            &mut mtp_kv_cache,
            &mtp_scratch,
        )
        .expect("mtp seed probe");
        let pred = argmax_one(
            &mut gpu,
            &mtp_scratch.logits,
            &mtp_scratch.argmax,
            target.config.vocab_size,
        );
        eprintln!("mtp_seed_probe seed={seed} pred={pred} pos={pos}");
    }

    let mut ar_kv_cache = KvCache::new_gpu_q8(
        &mut gpu,
        target.config.n_layers,
        target.config.n_kv_heads,
        target.config.head_dim,
        kv_seq,
    )
    .expect("ar kv");
    let mut ar_dn_state = DeltaNetState::new(&mut gpu, &target.config).expect("ar deltanet");
    let ar_scratch = Qwen35Scratch::new(&mut gpu, &target.config, 128).expect("ar scratch");
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        qwen35::forward_scratch_no_graph(
            &mut gpu,
            &target.weights,
            &target.config,
            tok,
            pos,
            &mut ar_kv_cache,
            &mut ar_dn_state,
            &ar_scratch,
        )
        .expect("ar prompt");
    }
    gpu.hip.device_synchronize().expect("sync ar prompt");
    let ar_start = Instant::now();
    let mut ar_tokens = Vec::new();
    for _ in 0..steps {
        let tok = argmax_one(
            &mut gpu,
            &ar_scratch.logits,
            &mtp_scratch.argmax,
            target.config.vocab_size,
        );
        if tokenizer.is_terminator(tok) {
            break;
        }
        let pos = prompt_tokens.len() + ar_tokens.len();
        qwen35::forward_scratch_no_graph(
            &mut gpu,
            &target.weights,
            &target.config,
            tok,
            pos,
            &mut ar_kv_cache,
            &mut ar_dn_state,
            &ar_scratch,
        )
        .expect("ar decode");
        gpu.hip.device_synchronize().expect("sync ar decode");
        ar_tokens.push(tok);
    }
    let ar_elapsed = ar_start.elapsed();

    let verify_scratch = VerifyScratch::with_prefill(
        &mut gpu,
        mtp_batch.max(1),
        target.config.dim,
        target.config.vocab_size,
        target.weights.output.k,
        &target.config,
    )
    .expect("verify scratch");
    let mut target_snap =
        DeltaNetSnapshot::new_for(&mut gpu, &target.dn_state).expect("target snapshot");

    let mut generated = Vec::new();
    let mut all_tokens = prompt_tokens.clone();
    let mut cycles = 0usize;
    let mut accepted = 0usize;
    let mut proposals = 0usize;
    let mut argmax_matches = 0usize;
    let mut top10_overlap_sum = 0usize;
    let mut top10_overlap_cycles = 0usize;
    let mut mtp_propose_us = 0u128;
    let mut verify_us = 0u128;
    let mut replay_us = 0u128;
    let mut mtp_repair_us = 0u128;
    let decode_start = Instant::now();
    while generated.len() < steps {
        cycles += 1;
        let remaining = steps - generated.len();
        let b = mtp_batch.min(remaining.max(1));
        let target_logits = gpu
            .download_f32(&target.scratch.logits)
            .expect("download target logits for MTP diagnostics");
        let mtp_logits = gpu
            .download_f32(&mtp_scratch.logits)
            .expect("download MTP logits for diagnostics");
        let target_top10 = top_k_ids(&target_logits, 10);
        let mtp_top10 = top_k_ids(&mtp_logits, 10);
        let target_argmax = target_top10[0] as u32;
        let mtp_argmax = mtp_top10[0] as u32;
        let overlap10 = top_k_overlap(&target_top10, &mtp_top10);
        argmax_matches += usize::from(target_argmax == mtp_argmax);
        top10_overlap_sum += overlap10;
        top10_overlap_cycles += 1;
        let step = if probe_only {
            probe_only_step(
                &mut gpu,
                &mut target,
                &mut mtp_kv_cache,
                &mtp_scratch,
                all_tokens.len(),
                b,
                use_norm_hidden,
            )
        } else {
            spec_step_mtp_greedy(
                &mut gpu,
                &mut target,
                &mut mtp_kv_cache,
                &mtp_scratch,
                all_tokens.len(),
                b,
                &mut target_snap,
                &verify_scratch,
            )
            .expect("mtp spec step")
        };
        gpu.hip.device_synchronize().expect("sync mtp step");
        if trace_mtp {
            eprintln!(
                "cycle={cycles} pos={} target_argmax={} mtp_argmax={} top10_overlap={} drafted={:?} accepted={} committed={:?}",
                all_tokens.len(),
                target_argmax,
                mtp_argmax,
                overlap10,
                step.drafted,
                step.accepted,
                step.committed
            );
        }

        accepted += step.accepted;
        proposals += step.proposal_count;
        mtp_propose_us += step.mtp_propose_us;
        verify_us += step.verify_us;
        replay_us += step.replay_us;
        mtp_repair_us += step.mtp_repair_us;
        for tok in step.committed {
            if tokenizer.is_terminator(tok) || generated.len() >= steps {
                break;
            }
            all_tokens.push(tok);
            generated.push(tok);
        }
        if generated
            .last()
            .is_some_and(|t| tokenizer.is_terminator(*t))
        {
            break;
        }
    }
    let decode_elapsed = decode_start.elapsed();

    let seconds = |d: Duration| d.as_secs_f64().max(1e-9);
    let ar_prefix_matches = generated
        .iter()
        .zip(ar_tokens.iter())
        .take(generated.len().min(ar_tokens.len()))
        .filter(|(a, b)| a == b)
        .count();
    println!("mtp_batch={mtp_batch}");
    println!("cycles={cycles}");
    println!("emitted_tokens={}", generated.len());
    println!("proposals_made={proposals}");
    println!("accepted_tokens={accepted}");
    println!("argmax_matches={argmax_matches}");
    println!(
        "top10_overlap_avg={:.4}",
        top10_overlap_sum as f64 / (top10_overlap_cycles.max(1) as f64)
    );
    println!(
        "accepted_tokens_per_cycle={:.4}",
        accepted as f64 / (cycles.max(1) as f64)
    );
    println!("prompt_tokens={}", prompt_tokens.len());
    println!(
        "prompt_target_ms={:.3}",
        prompt_target_time.as_secs_f64() * 1000.0
    );
    println!(
        "prompt_mtp_ms={:.3}",
        prompt_mtp_time.as_secs_f64() * 1000.0
    );
    println!(
        "decode_elapsed_ms={:.3}",
        decode_elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "decode_mtp_propose_ms={:.3}",
        mtp_propose_us as f64 / 1000.0
    );
    println!("decode_verify_ms={:.3}", verify_us as f64 / 1000.0);
    println!("decode_replay_ms={:.3}", replay_us as f64 / 1000.0);
    println!(
        "decode_mtp_exact_repair_ms={:.3}",
        mtp_repair_us as f64 / 1000.0
    );
    println!(
        "spec_effective_tok_s={:.3}",
        generated.len() as f64 / seconds(decode_elapsed)
    );
    println!("ar_baseline_tokens={}", ar_tokens.len());
    println!(
        "ar_baseline_elapsed_ms={:.3}",
        ar_elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "ar_baseline_tok_s={:.3}",
        ar_tokens.len() as f64 / seconds(ar_elapsed)
    );
    println!("ar_prefix_matches={ar_prefix_matches}");
    println!("generated_text={}", tokenizer.decode(&generated));

    if ar_prefix_matches != generated.len().min(ar_tokens.len()) {
        eprintln!(
            "ERROR: MTP output diverged from target-only greedy: matched {ar_prefix_matches}/{} prefix tokens",
            generated.len().min(ar_tokens.len())
        );
        std::process::exit(3);
    }

    verify_scratch.free_gpu(&mut gpu);
    mtp_scratch.free_gpu(&mut gpu);
    ar_scratch.free_gpu(&mut gpu);
}
