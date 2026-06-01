// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Focused perf benchmark for Qwen3.5 MQ4 forward pass.
//!
//! Separates prefill from generation, strips first-run kernel JIT overhead
//! via an explicit warmup phase, and reports per-token latency stats plus
//! an effective memory bandwidth estimate (weights_bytes × gen_tok/s).
//!
//! Usage: bench_qwen35_mq4 <model.hfq> [--prefill <N>] [--prefill-runs <N>] [--gen <N>] [--warmup <N>]

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::llama::{self, KvCache};
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench_qwen35_mq4 <model.hfq> [--prefill N] [--prefill-runs N] [--gen N] [--warmup N] [--emit-atlas <path.jsonl>]");
        std::process::exit(1);
    }
    let model_path = &args[1];

    // Defaults: 32-token prefill, 5-token warmup, 100-token bench.
    let mut prefill_len: usize = 32;
    let mut prefill_runs: usize = 1;
    let mut gen_len: usize = 100;
    let mut warmup_len: usize = 5;
    // Optional kernel-atlas emission: when set, write one typed AtlasRow
    // per timed phase (prefill, decode_ar) to this JSONL file. Replaces
    // stdout-scraping by external collectors like scripts/kernel_atlas.py.
    let mut atlas_out: Option<String> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prefill" => { prefill_len = args[i + 1].parse().unwrap(); i += 2; }
            "--prefill-runs" => { prefill_runs = args[i + 1].parse::<usize>().unwrap().max(1); i += 2; }
            "--gen"     => { gen_len     = args[i + 1].parse().unwrap(); i += 2; }
            "--warmup"  => { warmup_len  = args[i + 1].parse().unwrap(); i += 2; }
            "--emit-atlas" => { atlas_out = Some(args[i + 1].clone()); i += 2; }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }

    eprintln!("=== bench_qwen35_mq4 ===");
    eprintln!("Model: {model_path}");
    eprintln!("Phases: prefill={prefill_len} prefill_runs={prefill_runs} warmup={warmup_len} gen={gen_len}");

    let model_path_buf = Path::new(model_path);
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);

    // Auto-route safetensors directories (ParoQuant / AWQ / HF native) —
    // mirrors daemon.rs:1500-1504 and eval_hipfire's load path. HFQ files
    // take the canonical HFQ path below.
    let t_load = Instant::now();
    let (config, weights, model_bytes) = if model_path_buf.is_dir() {
        use hipfire_runtime::safetensors_source::SafetensorsSource;
        let source = SafetensorsSource::open(model_path_buf).expect("safetensors open");
        let config = qwen35::config_from_safetensors(&source)
            .expect("config_from_safetensors");
        eprintln!(
            "Config: dim={} layers={} heads={} kv_heads={} vocab={}",
            config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size
        );
        let bytes = std::fs::read_dir(model_path_buf)
            .ok()
            .map(|rd| rd.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
                .filter_map(|e| std::fs::metadata(e.path()).ok())
                .map(|m| m.len()).sum::<u64>())
            .unwrap_or(0);
        eprintln!("Model size: {:.3} GiB ({} bytes, safetensors)",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0), bytes);
        eprintln!("  loading via safetensors (ParoQuant path)");
        let weights = qwen35::load_weights_paroquant(&source, &config, &mut gpu)
            .expect("load_weights_paroquant");
        (config, weights, bytes)
    } else {
        let mut hfq = HfqFile::open(model_path_buf).expect("open model");
        let config = qwen35::config_from_hfq(&hfq).expect("read config");
        eprintln!(
            "Config: dim={} layers={} heads={} kv_heads={} vocab={}",
            config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size
        );
        let bytes = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
        eprintln!("Model size: {:.3} GiB ({} bytes)",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0), bytes);
        let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");
        (config, weights, bytes)
    };
    eprintln!("Weights loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let kv_seq = (prefill_len + warmup_len + gen_len + 16).max(512);
    // KV cache mode via HIPFIRE_KV_MODE env var:
    //   q8 (default) | asym4 | asym3 | asym2
    let kv_mode = std::env::var("HIPFIRE_KV_MODE").unwrap_or_else(|_| "q8".to_string());
    eprintln!("KV mode: {kv_mode}");
    let mut kv_cache = match kv_mode.as_str() {
        "q8" => KvCache::new_gpu_q8(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq
        ).unwrap(),
        "asym4" | "turbo4" => KvCache::new_gpu_asym4(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq
        ).unwrap(),
        "asym3" | "turbo3" | "turbo" => KvCache::new_gpu_asym3(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq
        ).unwrap(),
        "asym2" | "turbo2" => KvCache::new_gpu_asym2(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq
        ).unwrap(),
        other => panic!("unknown HIPFIRE_KV_MODE: {other}  (use q8|asym4|asym3|asym2)"),
    };
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, kv_seq).unwrap();

    // Deterministic fake-prompt: token 0, 1, 2, ... prefill_len-1. Keeps the
    // benchmark independent of tokenizer / chat template behaviour.
    let prompt_tokens: Vec<u32> = (0..prefill_len as u32).collect();

    // DPM warmup BEFORE prefill (issue #65). The default `HIPFIRE_DPM_WARMUP_SECS`
    // hook fires AFTER the warmup tokens, before the timed gen — useless for
    // prefill measurement when the GPU is in DPM step 0/1 from idle. This
    // mirrors that hook for the prefill phase: stabilizes clocks before the
    // timed forward_prefill_batch.
    if let Ok(secs_str) = std::env::var("HIPFIRE_DPM_WARMUP_SECS") {
        let secs: f32 = secs_str.parse().unwrap_or(0.0);
        if secs > 0.0 {
            eprintln!("\n=== DPM warmup ({secs:.1}s, pre-prefill) ===");
            gpu.dpm_warmup(secs).expect("dpm warmup");
        }
    }

    // === PREFILL ===
    // Route through forward_prefill_batch so the bench measures the production
    // prefill path (daemon + greedy_dump both go through it). Inside, this
    // takes the batched LA kernel path for MQ4 models and the FA gather/scatter
    // fallback for FA layers.
    let do_profile = std::env::var("HIPFIRE_PROFILE").ok().as_deref() == Some("1");
    // When HIPFIRE_ROCPROF_CSV=<path> is set AND profiling is active,
    // cross-check internal profile against rocprofv3 _kernel_stats.csv.
    // Stored here so the report is visible in both the logging block and
    // the atlas emission block below.
    let mut prefill_rocprof_report: Option<rdna_compute::profile_rocprof::ProfileReport> = None;
    // When profiling, do an unprofile warm-up prefill first to JIT all kernels,
    // then reset state and profile the second pass.
    if do_profile {
        eprintln!("\n=== warm-up prefill (JIT kernels) ===");
        qwen35::forward_prefill_batch(
            &mut gpu, &weights, &config, &prompt_tokens, 0,
            &mut kv_cache, &mut dn_state, &scratch,
            None, None, None, None,
        ).expect("warmup prefill failed");
        // Reset DeltaNet state for the profiled run
        dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
        kv_cache = match kv_mode.as_str() {
            "q8" => KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
            "asym4" | "turbo4" => KvCache::new_gpu_asym4(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
            "asym3" | "turbo3" | "turbo" => KvCache::new_gpu_asym3(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
            "asym2" | "turbo2" => KvCache::new_gpu_asym2(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
            _ => KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
        };
        eprintln!("  JIT complete, profiling next pass...");
        rdna_compute::profile::start();
    }
    eprintln!("\n=== prefill ({prefill_len} tokens) ===");
    let mut prefill_samples_ms = Vec::with_capacity(prefill_runs);

    // HIPFIRE_GRAPH_PREFILL=1: route the timed prefill loop through
    // hipGraph capture/replay. Reuses the DFlash verify_graph_cache
    // (single-slot keyed by b = prompt_tokens.len()). Run 0 warms,
    // run 1 captures + launches, runs 2+ replay. dn_state + kv_cache
    // are NOT re-created between runs (would invalidate baked-in
    // tensor pointers); state drift is accepted for perf measurement.
    let use_graph_prefill = std::env::var("HIPFIRE_GRAPH_PREFILL")
        .ok().as_deref() == Some("1");

    if use_graph_prefill {
        let pbs = scratch.prefill_batch.as_ref()
            .expect("HIPFIRE_GRAPH_PREFILL=1 requires PrefillBatchScratch");
        let b = prompt_tokens.len();
        if b > pbs.max_batch {
            panic!("HIPFIRE_GRAPH_PREFILL=1: prompt b={} exceeds pbs.max_batch={}",
                b, pbs.max_batch);
        }
        // Pre-upload tokens + positions ONCE — kernels read them from
        // device buffers, so subsequent replays use the same content.
        qwen35::upload_prefill_batch_inputs(&mut gpu, pbs, &prompt_tokens, 0)
            .expect("upload prefill batch inputs");
        if gpu.active_stream.is_none() {
            gpu.active_stream = Some(gpu.hip.stream_create().expect("stream"));
        }
        eprintln!("  [graph-prefill] b={}, max_batch={}", b, pbs.max_batch);

        for run in 0..prefill_runs {
            let mode;
            let t_prefill = Instant::now();
            if gpu.graphs.verify_has_graph(b) {
                mode = "replay";
                gpu.graphs.verify_graph_launch(
                    &gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap(), b,
                ).expect("graph replay");
            } else if gpu.graphs.verify_needs_warmup(b) {
                mode = "warmup";
                gpu.graphs.verify_mark_warmup_done(b);
                qwen35::forward_prefill_batch_single_chunk_captured(
                    &mut gpu, &weights, &config, &prompt_tokens, 0,
                    &mut kv_cache, &mut dn_state, &scratch, pbs,
                    None, None, None, None,
                ).expect("warmup prefill");
            } else {
                mode = "capture";
                gpu.graphs.begin_verify_graph_capture(
                    &gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap(), b,
                ).expect("begin capture");
                let r = qwen35::forward_prefill_batch_single_chunk_captured(
                    &mut gpu, &weights, &config, &prompt_tokens, 0,
                    &mut kv_cache, &mut dn_state, &scratch, pbs,
                    None, None, None, None,
                );
                if r.is_ok() {
                    let blob_count = gpu.graphs.capture_blobs.len();
                    gpu.graphs.end_verify_graph_capture(
                        &gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap(),
                    ).expect("end capture");
                    gpu.graphs.verify_graph_launch(
                        &gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap(), b,
                    ).expect("launch after capture");
                    eprintln!("  [graph-prefill] captured b={} ({} kernarg blobs)", b, blob_count);
                } else {
                    panic!("capture pass failed: {:?}", r);
                }
            }
            gpu.hip.device_synchronize().expect("sync");
            let ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
            prefill_samples_ms.push(ms);
            if prefill_runs > 1 {
                eprintln!("  run {:>2} ({}): {:.1}ms  {:.1} tok/s", run + 1, mode, ms, prefill_len as f64 / (ms / 1000.0));
            }
        }
    } else {
        for run in 0..prefill_runs {
            if run > 0 {
                dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
                kv_cache = match kv_mode.as_str() {
                    "q8" => KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
                    "asym4" | "turbo4" => KvCache::new_gpu_asym4(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
                    "asym3" | "turbo3" | "turbo" => KvCache::new_gpu_asym3(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
                    "asym2" | "turbo2" => KvCache::new_gpu_asym2(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
                    _ => KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
                };
            }
            let t_prefill = Instant::now();
            qwen35::forward_prefill_batch(
                &mut gpu, &weights, &config, &prompt_tokens, 0,
                &mut kv_cache, &mut dn_state, &scratch,
                None, None, None, None,
            ).expect("prefill forward failed");
            gpu.hip.device_synchronize().expect("sync after prefill");
            let ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
            prefill_samples_ms.push(ms);
            if prefill_runs > 1 {
                eprintln!("  run {:>2}: {:.1}ms  {:.1} tok/s", run + 1, ms, prefill_len as f64 / (ms / 1000.0));
            }
        }
    }
    let prefill_ms = *prefill_samples_ms.last().unwrap();
    // Captured outside `do_profile` so SUMMARY can split kernel vs wall.
    // None when profiling is disabled (HIPFIRE_PROFILE != 1).
    let mut prefill_kernel_ms: Option<f64> = None;
    if do_profile {
        let rocprof_csv_env = std::env::var("HIPFIRE_ROCPROF_CSV").ok();
        let report = if let Some(ref csv_path_str) = rocprof_csv_env {
            let csv_path = std::path::Path::new(csv_path_str.as_str());
            let r = rdna_compute::profile_rocprof::stop_with_rocprof(csv_path);
            if r.is_none() {
                eprintln!("WARN: HIPFIRE_ROCPROF_CSV set but internal profiler was not active");
            }
            r
        } else {
            rdna_compute::profile::stop().map(|entries| {
                rdna_compute::profile_rocprof::compute_coverage(&entries, &[])
            })
        };
        if let Some(ref report) = report {
            let entries = &report.internal;
            let mut by_kernel: std::collections::HashMap<&str, (f64, usize, usize)> = Default::default();
            for e in entries {
                let (time, count, bytes) = by_kernel.entry(e.kernel).or_default();
                *time += e.time_us;
                *count += 1;
                *bytes += e.bytes;
            }
            eprintln!("\n=== PROFILE ({} launches, {:.1}ms wall) ===", entries.len(), prefill_ms);
            let mut kerns: Vec<_> = by_kernel.iter().collect();
            kerns.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
            let total_us: f64 = kerns.iter().map(|(_, (t, _, _))| t).sum();
            for (kern, (us, n, bytes)) in &kerns {
                let gib_s = if *us > 0.0 {
                    (*bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (*us / 1_000_000.0)
                } else {
                    0.0
                };
                eprintln!("  {kern:45} {n:5}x  {:.1}ms  ({:.0}µs/call)  {:.1}%  {:.1} GiB/s",
                    us / 1000.0, us / *n as f64, us / total_us * 100.0, gib_s);
            }
            eprintln!("  {:45} {:5}   {:.1}ms", "TOTAL (serialized)", "", total_us / 1000.0);
            prefill_kernel_ms = Some(total_us / 1000.0);
            // rocprof cross-check output (only when CSV was loaded).
            if !report.rocprof.is_empty() {
                eprintln!("\n=== ROCPROF COVERAGE ({:.1}%) ===", report.coverage_pct);
                eprintln!("  rocprof total:   {:.1}ms  ({} kernels seen)",
                    report.rocprof_total_us / 1000.0, report.rocprof.len());
                eprintln!("  internal total:  {:.1}ms  ({} kernels tracked)",
                    report.internal_total_us / 1000.0, entries.len());
                eprintln!("  blindspot total: {:.1}ms  ({} un-tracked kernels)",
                    report.blindspot_total_us / 1000.0, report.blindspots.len());
                if !report.blindspots.is_empty() {
                    eprintln!("  BLINDSPOTS (rocprof saw; internal profile missed):");
                    for bs in &report.blindspots {
                        eprintln!("    {:55} {:5}x  {:.1}ms  {:.1}%",
                            bs.name, bs.calls, bs.duration_us / 1000.0, bs.percent);
                    }
                    if report.coverage_pct < 90.0 {
                        eprintln!("  WARNING: coverage {:.1}% < 90% -- investigate blindspot kernels", report.coverage_pct);
                    }
                }
            }
        }
        prefill_rocprof_report = report;
    }
    let prefill_tok_s = prefill_len as f64 / (prefill_ms / 1000.0);
    if prefill_samples_ms.len() > 1 {
        let mut sorted_prefill = prefill_samples_ms.clone();
        sorted_prefill.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = sorted_prefill[sorted_prefill.len() / 2];
        eprintln!("  median: {median_ms:.1}ms  {:.1} tok/s", prefill_len as f64 / (median_ms / 1000.0));
    }
    eprintln!("  total: {prefill_ms:.1}ms");
    eprintln!("  tok/s: {prefill_tok_s:.1}");
    eprintln!("  NOTE: first prefill run includes kernel JIT compile cost");

    // Emit prefill-only SUMMARY right here so prefill metrics survive any
    // failure in the gen phase that follows (the existing argmax-on-NaN
    // panic in graph-captured gen warmup blocks the end-of-run SUMMARY).
    eprintln!("PREFILL_SUMMARY  prefill_tok_s={prefill_tok_s:.1}  prefill_wall_ms={prefill_ms:.2}{}",
        split_prefill_summary(prefill_len, prefill_ms, prefill_kernel_ms));

    // Atlas row: typed prefill measurement. Emitted right here so it
    // survives any panic in the gen phase. Eliminates the stdout-scrape
    // round-trip the Python harness would otherwise do.
    if let Some(ref atlas_path) = atlas_out {
        let mut row = hipfire_atlas::AtlasRow::new("prefill", "bench_qwen35_mq4");
        row.set_metric_f64("prefill_tok_s", prefill_tok_s)
            .set_metric_f64("prefill_wall_ms", prefill_ms)
            .set_metric_u64("prefill_tokens", prefill_len as u64)
            .set_metric_u64("prefill_runs", prefill_runs as u64)
            .set_metric_str("kv_mode", &kv_mode)
            .set_metric_str("arch", &gpu.arch)
            .set_metric_str("model_path", model_path)
            .set_metric_u64("model_bytes", model_bytes)
            .set_extra("captured_at_unix_s", serde_json::Value::from(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            ));
        if let Some(kernel_ms) = prefill_kernel_ms {
            let prefill_tok_s_kernel = prefill_len as f64 / (kernel_ms / 1000.0);
            row.set_metric_f64("prefill_kernel_ms", kernel_ms);
            row.set_metric_f64("prefill_tok_s_kernel", prefill_tok_s_kernel);
            row.set_metric_f64("startup_overhead_ms", prefill_ms - kernel_ms);
            if prefill_ms > 0.0 {
                row.set_metric_f64("cold_overhead_pct", (prefill_ms - kernel_ms) / prefill_ms * 100.0);
            }
        }
        // Attach rocprofv3 coverage report when available.
        if let Some(ref report) = prefill_rocprof_report {
            if !report.rocprof.is_empty() {
                let mut rocprof_sorted: Vec<hipfire_atlas::AtlasRocprofKernel> =
                    report.rocprof.iter().map(|k| hipfire_atlas::AtlasRocprofKernel {
                        name: k.name.clone(),
                        calls: k.calls,
                        duration_us: k.duration_us,
                        percent: k.percent,
                    }).collect();
                rocprof_sorted.sort_by(|a, b| {
                    b.duration_us.partial_cmp(&a.duration_us).unwrap()
                });
                let blindspots_atlas: Vec<hipfire_atlas::AtlasRocprofKernel> =
                    report.blindspots.iter().map(|k| hipfire_atlas::AtlasRocprofKernel {
                        name: k.name.clone(),
                        calls: k.calls,
                        duration_us: k.duration_us,
                        percent: k.percent,
                    }).collect();
                let atlas_report = hipfire_atlas::AtlasProfileReport {
                    internal_total_us: report.internal_total_us,
                    rocprof_total_us: report.rocprof_total_us,
                    coverage_pct: report.coverage_pct,
                    blindspot_total_us: report.blindspot_total_us,
                    blindspots: blindspots_atlas,
                    rocprof_all: rocprof_sorted,
                };
                row.set_profile_report(&atlas_report);
            }
        }
        if let Err(e) = row.append_to_jsonl(atlas_path) {
            eprintln!("WARN: failed to append atlas row to {atlas_path}: {e}");
        }
    }

    // (deferred-conversion mode removed with givens — asym modes are natively
    //  batched so there's no prefill/decode cache swap to measure.)

    // Read logits to get a valid next token
    let logits = gpu.download_f32(&scratch.logits).unwrap();
    let mut next_token = llama::argmax(&logits);

    // === WARMUP ===
    eprintln!("\n=== warmup ({warmup_len} tokens — untimed, lets JIT settle) ===");
    let t_warmup = Instant::now();
    for step in 0..warmup_len {
        let pos = prefill_len + step;
        if pos >= kv_seq { break; }
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, next_token, pos,
            &mut kv_cache, &mut dn_state, &scratch,
        ).expect("warmup forward failed");
        let logits = gpu.download_f32(&scratch.logits).unwrap();
        next_token = llama::argmax(&logits);
    }
    let warmup_ms = t_warmup.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  total: {warmup_ms:.1}ms  avg: {:.2}ms/tok", warmup_ms / warmup_len as f64);

    // HIPFIRE_DPM_WARMUP_SECS: optional DPM-stabilization pass before the
    // timed decode. See crates/rdna-compute/src/dispatch.rs `dpm_warmup`.
    if let Ok(secs_str) = std::env::var("HIPFIRE_DPM_WARMUP_SECS") {
        let secs: f32 = secs_str.parse().unwrap_or(0.0);
        if secs > 0.0 {
            gpu.dpm_warmup(secs).expect("dpm warmup");
        }
    }

    // === GEN BENCHMARK ===
    // HIPFIRE_PROFILE_DECODE=1 wraps the timed gen loop in the per-kernel
    // profiler. Distinct from HIPFIRE_PROFILE=1 (which profiles prefill).
    // Decode is the steady-state hot path — this is the right surface to
    // attack for tok/s improvements.
    let do_profile_decode = std::env::var("HIPFIRE_PROFILE_DECODE").ok().as_deref() == Some("1");
    eprintln!("\n=== gen ({gen_len} tokens — timed) ===");
    let mut per_token_ms: Vec<f64> = Vec::with_capacity(gen_len);
    if do_profile_decode {
        rdna_compute::profile::start();
    }
    let t_gen_start = Instant::now();
    for step in 0..gen_len {
        let pos = prefill_len + warmup_len + step;
        if pos >= kv_seq { break; }
        let t = Instant::now();
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, next_token, pos,
            &mut kv_cache, &mut dn_state, &scratch,
        ).expect("gen forward failed");
        let logits = gpu.download_f32(&scratch.logits).unwrap();
        let t_ms = t.elapsed().as_secs_f64() * 1000.0;
        per_token_ms.push(t_ms);
        next_token = llama::argmax(&logits);
    }
    let gen_total_ms = t_gen_start.elapsed().as_secs_f64() * 1000.0;
    if do_profile_decode {
        if let Some(entries) = rdna_compute::profile::stop() {
            let mut by_kernel: std::collections::HashMap<&str, (f64, usize, usize)> = Default::default();
            for e in &entries {
                let (time, count, bytes) = by_kernel.entry(e.kernel).or_default();
                *time += e.time_us;
                *count += 1;
                *bytes += e.bytes;
            }
            eprintln!("\n=== DECODE PROFILE ({} launches, {:.1}ms wall) ===", entries.len(), gen_total_ms);
            let mut kerns: Vec<_> = by_kernel.iter().collect();
            kerns.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
            let total_us: f64 = kerns.iter().map(|(_, (t, _, _))| t).sum();
            for (kern, (us, n, bytes)) in &kerns {
                let gib_s = if *us > 0.0 {
                    (*bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (*us / 1_000_000.0)
                } else { 0.0 };
                eprintln!("  {kern:45} {n:5}x  {:.1}ms  ({:.0}µs/call)  {:.1}%  {:.1} GiB/s",
                    us / 1000.0, us / *n as f64, us / total_us * 100.0, gib_s);
            }
            eprintln!("  {:45} {:5}   {:.1}ms", "TOTAL (serialized)", "", total_us / 1000.0);
        }
    }

    // Stats
    let mut sorted = per_token_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n == 0 {
        eprintln!("  total: {gen_total_ms:.1}ms over 0 tokens");
        eprintln!("  tok/s (gen): 0.0");
        eprintln!();
        eprintln!("SUMMARY  gen_tok_s=0.0  bw_gib_s=0.0  prefill_tok_s={prefill_tok_s:.1}  avg_ms=0.00  p50_ms=0.00{}",
            split_prefill_summary(prefill_len, prefill_ms, prefill_kernel_ms));
        return;
    }
    let sum: f64 = sorted.iter().sum();
    let avg_ms = sum / n as f64;
    let min_ms = sorted[0];
    let max_ms = sorted[n - 1];
    let p50_ms = sorted[n / 2];
    let p90_ms = sorted[(n * 90) / 100];
    let p99_ms = sorted[(n.saturating_sub(1) * 99) / 100];
    let gen_tok_s = n as f64 / (gen_total_ms / 1000.0);

    // BW estimate: each gen token reads ~all weights (minus KV cache writes,
    // which are separate). Effective BW = model_bytes × tok/s.
    let bw_gbps = (model_bytes as f64 * gen_tok_s) / (1024.0 * 1024.0 * 1024.0);

    eprintln!("  total: {gen_total_ms:.1}ms over {n} tokens");
    eprintln!("  per-token ms:");
    eprintln!("    min={min_ms:.2}  p50={p50_ms:.2}  avg={avg_ms:.2}  p90={p90_ms:.2}  p99={p99_ms:.2}  max={max_ms:.2}");
    eprintln!("  tok/s (gen): {gen_tok_s:.1}");
    eprintln!("  effective BW: {bw_gbps:.1} GiB/s (model {:.2} GiB × {gen_tok_s:.1} tok/s)",
        model_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
    eprintln!();
    eprintln!("SUMMARY  gen_tok_s={gen_tok_s:.1}  bw_gib_s={bw_gbps:.1}  prefill_tok_s={prefill_tok_s:.1}  avg_ms={avg_ms:.2}  p50_ms={p50_ms:.2}{}",
        split_prefill_summary(prefill_len, prefill_ms, prefill_kernel_ms));

    // Decode atlas row (gen phase). Pairs with the prefill row emitted
    // earlier so a single bench invocation produces two phase-tagged rows.
    if let Some(ref atlas_path) = atlas_out {
        let mut row = hipfire_atlas::AtlasRow::new("decode_ar", "bench_qwen35_mq4");
        row.set_metric_f64("gen_tok_s", gen_tok_s)
            .set_metric_f64("bw_gib_s", bw_gbps)
            .set_metric_f64("avg_ms", avg_ms)
            .set_metric_f64("p50_ms", p50_ms)
            .set_metric_f64("p90_ms", p90_ms)
            .set_metric_f64("p99_ms", p99_ms)
            .set_metric_f64("min_ms", min_ms)
            .set_metric_f64("max_ms", max_ms)
            .set_metric_u64("gen_tokens", n as u64)
            .set_metric_str("kv_mode", &kv_mode)
            .set_metric_str("arch", &gpu.arch)
            .set_metric_str("model_path", model_path)
            .set_metric_u64("model_bytes", model_bytes);
        if let Err(e) = row.append_to_jsonl(atlas_path) {
            eprintln!("WARN: failed to append atlas row to {atlas_path}: {e}");
        }
    }
}

/// Emit the latency-class split for prefill when profiling captured the
/// per-kernel time. `prefill_tok_s` in the main SUMMARY is the wall-clock
/// figure (includes first-process JIT compile + graph capture). The
/// `*_kernel` figures here are steady-state kernel throughput — what every
/// call after the first one converges to once JIT amortizes. The gap is
/// `startup_overhead_ms`. AOT-shipped HSACOs should drop that gap to ~0.
///
/// Returns an empty string if profiling was disabled so the SUMMARY line
/// stays parseable by older tooling.
#[cfg(feature = "deltanet")]
fn split_prefill_summary(prefill_len: usize, prefill_ms: f64, prefill_kernel_ms: Option<f64>) -> String {
    if let Some(kernel_ms) = prefill_kernel_ms {
        let prefill_tok_s_kernel = prefill_len as f64 / (kernel_ms / 1000.0);
        let startup_overhead_ms = prefill_ms - kernel_ms;
        let cold_pct = if prefill_ms > 0.0 { (startup_overhead_ms / prefill_ms) * 100.0 } else { 0.0 };
        format!(
            "  prefill_tok_s_kernel={prefill_tok_s_kernel:.1}  prefill_kernel_ms={kernel_ms:.2}  startup_overhead_ms={startup_overhead_ms:.2}  cold_overhead_pct={cold_pct:.1}"
        )
    } else {
        String::new()
    }
}
