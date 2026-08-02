// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek V4 MQ2R long-context measurement probe.
//!
//! This is deliberately a measurement driver, not a serving path. It can either
//! prefill a real prefix (`--prefill N`) or set a synthetic absolute position
//! (`--position N`) to measure the allocation and O(N/ratio) decode wall without
//! spending hours constructing cache contents. The latter is valid for timing
//! and memory only; top-k sanity always requires `--prefill`.

use hipfire_arch_deepseek4::{
    forward::{decode_step, forward_prefill_batch_chunked, PrefillBatchScratch},
    DeepseekV4, DeepseekV4State,
};
use hipfire_runtime::{arch::Architecture, hfq::HfqFile, tokenizer::Tokenizer};
use rdna_compute::Gpu;
use std::{collections::HashSet, path::Path, time::Instant};

#[derive(Debug)]
struct Args {
    model: String,
    cap: usize,
    position: Option<usize>,
    prefill: Option<usize>,
    batch: usize,
    decode_reps: usize,
    topk_sanity: bool,
    generate: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let model = it.next().ok_or_else(|| {
        "usage: ds4_longctx_probe MODEL --cap N [--position N | --prefill N] \
         [--batch N] [--decode-reps N] [--topk-sanity] [--generate N]"
            .to_string()
    })?;
    let mut out = Args {
        model,
        cap: 2048,
        position: None,
        prefill: None,
        batch: 1024,
        decode_reps: 3,
        topk_sanity: false,
        generate: 0,
    };
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--cap" => out.cap = it.next().ok_or("--cap N")?.parse().map_err(|_| "bad --cap")?,
            "--position" => {
                out.position = Some(
                    it.next()
                        .ok_or("--position N")?
                        .parse()
                        .map_err(|_| "bad --position")?,
                )
            }
            "--prefill" => {
                out.prefill = Some(
                    it.next()
                        .ok_or("--prefill N")?
                        .parse()
                        .map_err(|_| "bad --prefill")?,
                )
            }
            "--batch" => {
                out.batch = it.next().ok_or("--batch N")?.parse().map_err(|_| "bad --batch")?
            }
            "--decode-reps" => {
                out.decode_reps = it
                    .next()
                    .ok_or("--decode-reps N")?
                    .parse()
                    .map_err(|_| "bad --decode-reps")?
            }
            "--topk-sanity" => out.topk_sanity = true,
            "--generate" => {
                out.generate = it.next().ok_or("--generate N")?.parse().map_err(|_| "bad --generate")?
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if out.position.is_some() && out.prefill.is_some() {
        return Err("--position and --prefill are mutually exclusive".to_string());
    }
    if out.topk_sanity && out.prefill.is_none() {
        return Err("--topk-sanity requires real --prefill cache contents".to_string());
    }
    if out.cap == 0 || out.batch == 0 {
        return Err("--cap and --batch must be non-zero".to_string());
    }
    Ok(out)
}

fn used_vram(gpu: &Gpu) -> Result<(usize, usize), String> {
    let (free, total) = gpu
        .hip
        .get_vram_info()
        .map_err(|e| format!("hipMemGetInfo: {e:?}"))?;
    Ok((total.saturating_sub(free), total))
}

fn print_vram(stage: &str, gpu: &Gpu, peak: &mut usize) -> Result<(), String> {
    let (used, total) = used_vram(gpu)?;
    *peak = (*peak).max(used);
    println!(
        "VRAM stage={stage} used_bytes={used} used_gib={:.3} total_bytes={total} peak_bytes={}",
        used as f64 / (1u64 << 30) as f64,
        *peak
    );
    Ok(())
}

fn argmax(xs: &[f32]) -> u32 {
    xs.iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index as u32)
        .unwrap_or(0)
}

fn prompt_tokens(tokenizer: &Tokenizer, n: usize) -> Vec<u32> {
    let early = tokenizer.encode(
        "EARLY CONTEXT FACT: The launch code is ORCHID-7391. Remember that exact code.\n\n",
    );
    let filler = tokenizer.encode(
        "This is a long-context systems measurement passage. The indexer scores compressed memory slots while sparse attention reads only its selected window. ",
    );
    let suffix = tokenizer.encode(
        "\n\nQuestion: What was the exact launch code stated at the very beginning? Answer with the code and one short sentence.\nAnswer:",
    );
    let mut out = Vec::with_capacity(n);
    out.extend(early.into_iter().take(n));
    while out.len() + suffix.len() < n {
        let take = (n - suffix.len() - out.len()).min(filler.len());
        out.extend_from_slice(&filler[..take]);
    }
    if out.len() < n {
        let take = (n - out.len()).min(suffix.len());
        out.extend_from_slice(&suffix[suffix.len() - take..]);
    }
    out.truncate(n);
    out
}

#[derive(Debug)]
struct TopkSnapshot {
    sets: Vec<(usize, HashSet<i32>)>,
    all_count_ok: bool,
    all_unique: bool,
    all_in_range: bool,
    prefix_layers: usize,
    active: usize,
    visible: usize,
}

fn topk_snapshot(
    cfg: &hipfire_arch_deepseek4::DeepseekV4Config,
    state: &DeepseekV4State,
    gpu: &Gpu,
    position: usize,
    cap: usize,
) -> Result<TopkSnapshot, String> {
    let visible = ((position + 1) / 4).min(cap);
    let active = cfg.index_topk.min(visible);
    let prefix: HashSet<i32> = (0..active as i32).collect();
    let mut sets = Vec::new();
    let mut all_count_ok = true;
    let mut all_unique = true;
    let mut all_in_range = true;
    let mut prefix_layers = 0usize;
    for (layer, &ratio) in cfg.compress_ratios.iter().enumerate() {
        if ratio != 4 {
            continue;
        }
        let tensor = state._indexer[layer]
            .topk_idx_indices
            .as_ref()
            .ok_or_else(|| format!("top-k tensor absent at ratio-4 layer {layer}"))?;
        let raw = gpu
            .download_f32(tensor)
            .map_err(|e| format!("download top-k layer {layer}: {e:?}"))?;
        let indices: Vec<i32> = raw.iter().map(|value| i32::from_ne_bytes(value.to_ne_bytes())).collect();
        let valid: Vec<i32> = indices
            .iter()
            .copied()
            .filter(|&index| index >= 0)
            .collect();
        let set: HashSet<i32> = valid.iter().copied().collect();
        all_count_ok &= valid.len() == active;
        all_unique &= set.len() == valid.len();
        all_in_range &= valid.iter().all(|&index| (index as usize) < visible);
        if set == prefix {
            prefix_layers += 1;
        }
        sets.push((layer, set));
    }
    Ok(TopkSnapshot {
        sets,
        all_count_ok,
        all_unique,
        all_in_range,
        prefix_layers,
        active,
        visible,
    })
}

fn report_topk_change(first: &TopkSnapshot, second: &TopkSnapshot) {
    let mut changed_layers = 0usize;
    let mut intersections = 0usize;
    let mut unions = 0usize;
    for ((layer_a, a), (layer_b, b)) in first.sets.iter().zip(&second.sets) {
        assert_eq!(layer_a, layer_b);
        if a != b {
            changed_layers += 1;
        }
        intersections += a.intersection(b).count();
        unions += a.union(b).count();
    }
    let jaccard = if unions == 0 {
        1.0
    } else {
        intersections as f64 / unions as f64
    };
    println!(
        "TOPK visible={} active={} layers={} count_ok={} unique_ok={} causal_in_range_ok={} prefix_layers_query_a={} prefix_layers_query_b={} changed_layers={} aggregate_jaccard={jaccard:.6}",
        first.visible,
        first.active,
        first.sets.len(),
        first.all_count_ok && second.all_count_ok,
        first.all_unique && second.all_unique,
        first.all_in_range && second.all_in_range,
        first.prefix_layers,
        second.prefix_layers,
        changed_layers,
    );
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    std::env::set_var("HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS", args.cap.to_string());
    std::env::set_var("HIPFIRE_DEEPSEEK4_GRAPH", "0");
    std::env::set_var("HIPFIRE_FORWARD_LOWERED", "0");

    let wall_start = Instant::now();
    println!(
        "PROBE model={} cap={} position={:?} prefill={:?} batch={} decode_reps={} topk_sanity={} generate={}",
        args.model,
        args.cap,
        args.position,
        args.prefill,
        args.batch,
        args.decode_reps,
        args.topk_sanity,
        args.generate
    );
    let mut hfq = HfqFile::open(Path::new(&args.model)).map_err(|e| format!("open model: {e:?}"))?;
    let mut cfg = DeepseekV4::config_from_hfq(&hfq)?;
    cfg.load_dspark = false;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer: {e:?}"))?;
    let ratio4_layers = cfg.compress_ratios.iter().filter(|&&ratio| ratio == 4).count();
    let ratio128_layers = cfg.compress_ratios.iter().filter(|&&ratio| ratio == 128).count();
    println!(
        "CONFIG layers={} ratio4_layers={} ratio128_layers={} index_topk={} max_position_embeddings={} effective_route_scale={}",
        cfg.num_hidden_layers,
        ratio4_layers,
        ratio128_layers,
        cfg.index_topk,
        cfg.max_position_embeddings,
        hipfire_arch_deepseek4::forward::effective_route_scale(cfg.routed_scaling_factor, cfg.mq2r),
    );

    let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e:?}"))?;
    let mut peak = 0usize;
    print_vram("gpu_init", &gpu, &mut peak)?;
    let load_start = Instant::now();
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    drop(hfq);
    gpu.hip.device_synchronize().map_err(|e| format!("load sync: {e:?}"))?;
    let load_s = load_start.elapsed().as_secs_f64();
    println!("LOAD wall_s={load_s:.6}");
    print_vram("weights_loaded", &gpu, &mut peak)?;

    let mut state = DeepseekV4State::new(&cfg)?;
    let mut current_pos = args.position.unwrap_or(0);
    let mut next_token = tokenizer.bos_id;
    let mut prefill_s = 0.0f64;

    let pbs = if let Some(n) = args.prefill {
        let scratch = PrefillBatchScratch::new(&mut gpu, &cfg, args.batch)
            .map_err(|e| format!("alloc prefill scratch batch={}: {e}", args.batch))?;
        print_vram("prefill_scratch", &gpu, &mut peak)?;
        let tokens = prompt_tokens(&tokenizer, n);
        gpu.hip.device_synchronize().map_err(|e| format!("prefill pre-sync: {e:?}"))?;
        let start = Instant::now();
        let logits = forward_prefill_batch_chunked(
            &cfg,
            &weights,
            &mut state,
            &mut gpu,
            &tokens,
            0,
            &scratch,
        )
        .map_err(|e| format!("prefill context={n}: {e}"))?;
        gpu.hip.device_synchronize().map_err(|e| format!("prefill post-sync: {e:?}"))?;
        prefill_s = start.elapsed().as_secs_f64();
        current_pos = n;
        state.n_tokens = n as u64;
        next_token = argmax(&logits);
        println!(
            "PREFILL context={} wall_s={prefill_s:.6} tok_s={:.6}",
            n,
            n as f64 / prefill_s.max(1e-12)
        );
        print_vram("prefill_done", &gpu, &mut peak)?;
        Some(scratch)
    } else {
        // Materialize every lazy buffer at a real position-0 boundary before
        // jumping to the requested synthetic position. Several DS4 scratch
        // views are initialized by the first forward and cannot safely be
        // created for the first time at an arbitrary absolute position.
        state.n_tokens = 0;
        let bootstrap = match decode_step(
            &cfg,
            &weights,
            &mut state,
            &mut gpu,
            next_token,
            0,
        ) {
            Ok(logits) => logits,
            Err(error) => {
                let _ = print_vram("bootstrap_failure", &gpu, &mut peak);
                return Err(format!("synthetic bootstrap cap={}: {error}", args.cap));
            }
        };
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("bootstrap sync: {e:?}"))?;
        next_token = argmax(&bootstrap);
        print_vram("synthetic_bootstrap", &gpu, &mut peak)?;
        state.n_tokens = current_pos as u64;
        println!("SYNTHETIC_JUMP from_position=1 to_position={current_pos}");
        None
    };

    if args.topk_sanity {
        let logits_a = decode_step(&cfg, &weights, &mut state, &mut gpu, next_token, current_pos as u32)
            .map_err(|e| format!("top-k query A pos={current_pos}: {e}"))?;
        gpu.hip.device_synchronize().map_err(|e| format!("top-k A sync: {e:?}"))?;
        let first = topk_snapshot(&cfg, &state, &gpu, current_pos, args.cap)?;
        current_pos += 1;
        let token_b = argmax(&logits_a);
        let logits_b = decode_step(&cfg, &weights, &mut state, &mut gpu, token_b, current_pos as u32)
            .map_err(|e| format!("top-k query B pos={current_pos}: {e}"))?;
        gpu.hip.device_synchronize().map_err(|e| format!("top-k B sync: {e:?}"))?;
        let second = topk_snapshot(&cfg, &state, &gpu, current_pos, args.cap)?;
        report_topk_change(&first, &second);
        current_pos += 1;
        next_token = argmax(&logits_b);
        print_vram("topk_done", &gpu, &mut peak)?;
    }

    // One untimed decode materializes all lazy state and cache allocations.
    let warm_start = Instant::now();
    let warm_logits = match decode_step(
        &cfg,
        &weights,
        &mut state,
        &mut gpu,
        next_token,
        current_pos as u32,
    ) {
        Ok(logits) => logits,
        Err(error) => {
            let _ = print_vram("decode_failure", &gpu, &mut peak);
            return Err(format!(
                "decode warmup cap={} pos={current_pos}: {error}",
                args.cap
            ));
        }
    };
    gpu.hip.device_synchronize().map_err(|e| format!("warmup sync: {e:?}"))?;
    let warm_s = warm_start.elapsed().as_secs_f64();
    current_pos += 1;
    next_token = argmax(&warm_logits);
    let mut latest_logits = Some(warm_logits);
    println!("DECODE_WARMUP wall_s={warm_s:.6}");
    print_vram("decode_warm", &gpu, &mut peak)?;

    let mut decode_times = Vec::with_capacity(args.decode_reps);
    for rep in 0..args.decode_reps {
        gpu.hip.device_synchronize().map_err(|e| format!("decode pre-sync: {e:?}"))?;
        let start = Instant::now();
        let logits = decode_step(
            &cfg,
            &weights,
            &mut state,
            &mut gpu,
            next_token,
            current_pos as u32,
        )
        .map_err(|e| format!("decode rep={rep} cap={} pos={current_pos}: {e}", args.cap))?;
        gpu.hip.device_synchronize().map_err(|e| format!("decode post-sync: {e:?}"))?;
        let seconds = start.elapsed().as_secs_f64();
        println!("DECODE_REP rep={rep} position={current_pos} wall_s={seconds:.6} tok_s={:.6}", 1.0 / seconds);
        decode_times.push(seconds);
        current_pos += 1;
        next_token = argmax(&logits);
        latest_logits = Some(logits);
        print_vram("decode_rep", &gpu, &mut peak)?;
    }
    decode_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_s = decode_times
        .get(decode_times.len() / 2)
        .copied()
        .unwrap_or(f64::NAN);
    println!(
        "DECODE_SUMMARY cap={} effective_position={} reps={} median_s={median_s:.6} median_tok_s={:.6}",
        args.cap,
        current_pos,
        args.decode_reps,
        1.0 / median_s
    );

    if args.generate > 0 {
        let mut generated = Vec::with_capacity(args.generate);
        let mut token = latest_logits.as_deref().map(argmax).unwrap_or(next_token);
        for _ in 0..args.generate {
            generated.push(token);
            let logits = decode_step(&cfg, &weights, &mut state, &mut gpu, token, current_pos as u32)
                .map_err(|e| format!("generation pos={current_pos}: {e}"))?;
            current_pos += 1;
            token = argmax(&logits);
        }
        gpu.hip.device_synchronize().map_err(|e| format!("generation sync: {e:?}"))?;
        println!("GENERATED_TOKEN_IDS {generated:?}");
        println!("GENERATED_TEXT_BEGIN");
        println!("{}", tokenizer.decode(&generated));
        println!("GENERATED_TEXT_END");
        print_vram("generation_done", &gpu, &mut peak)?;
    }

    if let Some(scratch) = pbs {
        scratch.free_gpu(&mut gpu);
    }
    let total_s = wall_start.elapsed().as_secs_f64();
    println!(
        "RESULT status=pass cap={} prefill_context={} prefill_s={prefill_s:.6} median_decode_tok_s={:.6} peak_vram_bytes={} peak_vram_gib={:.3} total_wall_s={total_s:.6}",
        args.cap,
        args.prefill.unwrap_or(0),
        1.0 / median_s,
        peak,
        peak as f64 / (1u64 << 30) as f64,
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("RESULT status=fail error={error:?}");
        std::process::exit(2);
    }
}
