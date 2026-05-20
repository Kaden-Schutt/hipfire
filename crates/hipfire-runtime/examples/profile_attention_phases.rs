// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase-level breakdown of attention_q8_0_kv at long context.
//!
//! The normal profile_qwen35_mq4 only reports per-kernel totals — it shows
//! attention_q8_0_kv as the hot kernel at ctx=4096 but doesn't tell us
//! which internal phase dominates. This example uses the attention_q8_0_kv_timed
//! kernel (same code as baseline + __builtin_amdgcn_s_memrealtime() reads
//! around each phase) to split the time between:
//!
//!   phase 1: QK^T + local_max
//!   phase 2: softmax max/sum reductions
//!   phase 3: V-weighted sum
//!
//! Usage: profile_attention_phases <model.mq4> [--prefill N] [--repeats N]
//!
//! The workflow: prefill to target context via forward_prefill_batch, then
//! do a handful of single-token forward steps to settle the KV cache, then
//! directly invoke attention_q8_0_kv_timed N times on the populated cache
//! and average the per-phase cycle counts.

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::llama::{self, KvCache};
    use rdna_compute::{DType};
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: profile_attention_phases <model.mq4> [--prefill N] [--repeats N]");
        std::process::exit(1);
    }
    let model_path = &args[1];

    let mut prefill_len: usize = 4096;
    let mut repeats: usize = 20;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prefill" => { prefill_len = args[i + 1].parse().unwrap(); i += 2; }
            "--repeats" => { repeats = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }

    eprintln!("=== profile_attention_phases ===");
    eprintln!("Model: {model_path}");
    eprintln!("Prefill: {prefill_len}  Repeats: {repeats}");

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    eprintln!(
        "Config: dim={} layers={} n_heads={} n_kv_heads={} head_dim={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.head_dim,
    );

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);

    let t_load = Instant::now();
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");
    eprintln!("Weights loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // Find first FA layer so we know which layer's KV to probe.
    let fa_layer_idx = {
        let mut found = None;
        for (i, l) in weights.layers.iter().enumerate() {
            if matches!(l, hipfire_arch_qwen35::qwen35::LayerWeights::FullAttn(_)) { found = Some(i); break; }
        }
        found.expect("no FA layer in model")
    };
    eprintln!("First FA layer: {fa_layer_idx}");

    let kv_seq = (prefill_len + 32).max(512);
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq
    ).unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    // Prefill to target context
    let prompt: Vec<u32> = (0..prefill_len as u32).collect();
    eprintln!("\nPrefill {prefill_len} tokens (batched)...");
    let t_prefill = Instant::now();
    qwen35::forward_prefill_batch(
        &mut gpu, &weights, &config, &prompt, 0,
        &mut kv_cache, &mut dn_state, &scratch,
        None, None, None, None,
    ).expect("prefill failed");
    eprintln!("  prefill: {:.1}ms", t_prefill.elapsed().as_secs_f64() * 1000.0);

    let logits = gpu.download_f32(&scratch.logits).unwrap();
    let mut next_token = llama::argmax(&logits);

    // 5 warmup decode steps to populate KV cache beyond prefill (and ensure kernel JIT)
    for step in 0..5 {
        let pos = prefill_len + step;
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, next_token, pos,
            &mut kv_cache, &mut dn_state, &scratch,
        ).expect("warmup forward failed");
        let logits = gpu.download_f32(&scratch.logits).unwrap();
        next_token = llama::argmax(&logits);
    }
    let current_pos = prefill_len + 5;
    eprintln!("KV populated through pos {} (seq_len={})", current_pos - 1, current_pos);

    // Allocate buffers for direct timed-kernel launches:
    //   - q: f32 [n_heads × head_dim]  — fake deterministic pattern
    //   - attn_out: f32 [n_heads × head_dim]
    //   - pos_buf: raw int32 DeviceBuffer [1]
    //   - cycle_counts: raw [n_heads * 3 * 8] bytes (u64 per entry)
    // Restore real Q for production testing
    let q_values: Vec<f32> = (0..config.n_heads * config.head_dim)
        .map(|i| (i as f32 * 0.013).sin() * 0.5)
        .collect();
    let q_tensor = gpu.upload_f32(&q_values, &[config.n_heads, config.head_dim]).unwrap();
    let attn_out = gpu.zeros(&[config.n_heads, config.head_dim], DType::F32).unwrap();
    let pos_buf = gpu.hip.malloc(4).unwrap();
    let pos_i32 = (current_pos - 1) as i32;
    gpu.hip.memcpy_htod(&pos_buf, &pos_i32.to_ne_bytes()).unwrap();
    let cycle_tensor = gpu.zeros(&[config.n_heads * 3 * 8], DType::Raw).unwrap();

    // Grab k_cache, v_cache of the FA layer (indexed in KvCache)
    let k_cache = &kv_cache.k_gpu[fa_layer_idx];
    let v_cache = &kv_cache.v_gpu[fa_layer_idx];

    let seq_len = current_pos; // pos_buf has pos = current_pos - 1 → seq_len = pos + 1
    eprintln!("\n=== Timing attention_q8_0_kv_timed at seq_len={seq_len} ({repeats} repeats) ===");

    // Warmup once (first call triggers JIT for attention_q8_0_kv_timed)
    gpu.attention_q8_0_kv_timed(
        &q_tensor, k_cache, v_cache, &attn_out,
        &pos_buf, seq_len,
        config.n_heads, config.n_kv_heads, config.head_dim, kv_seq,
        &cycle_tensor,
    ).unwrap();
    gpu.hip.device_synchronize().unwrap();

    // Collect per-phase cycles across repeats. Each call overwrites cycle_tensor
    // so we download after each call. Also measure total elapsed ms via Instant.
    let mut p1_ticks_sum = 0u64;
    let mut p2_ticks_sum = 0u64;
    let mut p3_ticks_sum = 0u64;
    let mut wall_us_sum = 0.0f64;

    for _ in 0..repeats {
        let t = Instant::now();
        gpu.attention_q8_0_kv_timed(
            &q_tensor, k_cache, v_cache, &attn_out,
            &pos_buf, seq_len,
            config.n_heads, config.n_kv_heads, config.head_dim, kv_seq,
            &cycle_tensor,
        ).unwrap();
        gpu.hip.device_synchronize().unwrap();
        wall_us_sum += t.elapsed().as_secs_f64() * 1e6;

        // Download cycle_tensor as raw bytes and reinterpret as u64
        let mut cycles = vec![0u64; config.n_heads * 3];
        let dst_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                cycles.as_mut_ptr() as *mut u8,
                config.n_heads * 3 * 8,
            )
        };
        gpu.hip.memcpy_dtoh(dst_bytes, &cycle_tensor.buf).unwrap();

        // Sum across heads for this repeat (we want average per-head later)
        for h in 0..config.n_heads {
            p1_ticks_sum += cycles[h * 3 + 0];
            p2_ticks_sum += cycles[h * 3 + 1];
            p3_ticks_sum += cycles[h * 3 + 2];
        }
    }

    let n_samples = (config.n_heads * repeats) as f64;
    let p1_avg = p1_ticks_sum as f64 / n_samples;
    let p2_avg = p2_ticks_sum as f64 / n_samples;
    let p3_avg = p3_ticks_sum as f64 / n_samples;
    let total_avg = p1_avg + p2_avg + p3_avg;
    let wall_us_avg = wall_us_sum / repeats as f64;

    println!();
    println!("=== PHASE BREAKDOWN at seq_len={seq_len} ===");
    println!("(avg ticks per-head, memrealtime counter)");
    println!();
    println!("  {:<20} {:>14} {:>10}", "phase", "avg_ticks", "% total");
    println!("  {:-<46}", "");
    println!(
        "  {:<20} {:>14.0} {:>9.1}%",
        "phase 1 (QK^T)",  p1_avg, 100.0 * p1_avg / total_avg,
    );
    println!(
        "  {:<20} {:>14.0} {:>9.1}%",
        "phase 2 (softmax)", p2_avg, 100.0 * p2_avg / total_avg,
    );
    println!(
        "  {:<20} {:>14.0} {:>9.1}%",
        "phase 3 (V-sum)",  p3_avg, 100.0 * p3_avg / total_avg,
    );
    println!("  {:-<46}", "");
    println!("  {:<20} {:>14.0}", "total (sum)", total_avg);
    println!();
    println!("Wall clock per call: {wall_us_avg:.1} us");

    // Calibrate ticks → us via wall clock. This is approximate — wall_us
    // includes launch overhead, while total ticks don't — so ticks < wall.
    // But the ratio at least tells us the tick rate.
    let tick_rate_hz = total_avg / (wall_us_avg / 1e6);
    let ns_per_tick = 1e9 / tick_rate_hz;
    println!("Inferred tick rate: {:.2e} Hz  (~{:.2} ns/tick)", tick_rate_hz, ns_per_tick);
    println!();
    println!("Per-phase wall-clock estimate (ticks × ns/tick):");
    println!("  phase 1: {:.1} us", p1_avg * ns_per_tick / 1000.0);
    println!("  phase 2: {:.1} us", p2_avg * ns_per_tick / 1000.0);
    println!("  phase 3: {:.1} us", p3_avg * ns_per_tick / 1000.0);
    println!();
    println!(
        "Notes: tick→time calibration assumes phases account for all of wall time,"
    );
    println!(
        "  which is approximately true but not exactly (launch overhead, etc.)."
    );
    println!(
        "  Use the %-total column as the definitive answer for which phase dominates."
    );

    // ═══ Event-timed baseline (more accurate than wall_clock64-based phases) ═══
    eprintln!("\n=== Event-timed baseline (no memrealtime serialization) ===");
    let repeats_cmp = 50usize;

    for _ in 0..3 {
        gpu.attention_q8_0_kv(
            &q_tensor, k_cache, v_cache, &attn_out,
            &pos_buf, seq_len,
            config.n_heads, config.n_kv_heads, config.head_dim, kv_seq,
        ).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    let t_v1 = Instant::now();
    for _ in 0..repeats_cmp {
        gpu.attention_q8_0_kv(
            &q_tensor, k_cache, v_cache, &attn_out,
            &pos_buf, seq_len,
            config.n_heads, config.n_kv_heads, config.head_dim, kv_seq,
        ).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let v1_us = t_v1.elapsed().as_secs_f64() * 1e6 / repeats_cmp as f64;

    println!();
    println!("=== baseline attention_q8_0_kv timing at seq_len={seq_len} ===");
    println!("  wall clock (50 reps avg): {v1_us:7.1} us/call");

    // ═══ Flash attention comparison ═══
    eprintln!("\n=== Flash attention (tile+reduce) ===");
    const TILE_SIZE: usize = 128;
    let max_tiles = (kv_seq + TILE_SIZE - 1) / TILE_SIZE;
    let partials = gpu.zeros(
        &[config.n_heads * max_tiles * (2 + config.head_dim)],
        DType::F32,
    ).unwrap();

    // Warmup flash (forces JIT compile of both kernels)
    for _ in 0..3 {
        gpu.attention_flash_q8_0(
            &q_tensor, k_cache, v_cache, &attn_out,
            &pos_buf, seq_len,
            config.n_heads, config.n_kv_heads, config.head_dim, kv_seq,
            &partials,
        ).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    // Time flash
    let t_flash = Instant::now();
    for _ in 0..repeats_cmp {
        gpu.attention_flash_q8_0(
            &q_tensor, k_cache, v_cache, &attn_out,
            &pos_buf, seq_len,
            config.n_heads, config.n_kv_heads, config.head_dim, kv_seq,
            &partials,
        ).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let flash_us = t_flash.elapsed().as_secs_f64() * 1e6 / repeats_cmp as f64;

    // Correctness check
    gpu.attention_q8_0_kv(
        &q_tensor, k_cache, v_cache, &attn_out,
        &pos_buf, seq_len,
        config.n_heads, config.n_kv_heads, config.head_dim, kv_seq,
    ).unwrap();
    let out_baseline = gpu.download_f32(&attn_out).unwrap();

    gpu.attention_flash_q8_0(
        &q_tensor, k_cache, v_cache, &attn_out,
        &pos_buf, seq_len,
        config.n_heads, config.n_kv_heads, config.head_dim, kv_seq,
        &partials,
    ).unwrap();
    let out_flash = gpu.download_f32(&attn_out).unwrap();

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut max_val = 0.0f32;
    for (a, b) in out_baseline.iter().zip(out_flash.iter()) {
        let e = (a - b).abs();
        max_abs = max_abs.max(e);
        max_val = max_val.max(a.abs()).max(b.abs());
        if a.abs() > 1e-6 { max_rel = max_rel.max(e / a.abs()); }
    }

    println!();
    println!("=== baseline vs flash at seq_len={seq_len} ===");
    println!("  baseline:    {v1_us:7.1} us/call  (1.00x)");
    println!("  flash:       {flash_us:7.1} us/call  ({:.2}x)", v1_us / flash_us);
    println!();
    println!("  max_abs={max_abs:.2e}  max_rel={max_rel:.2e}  (out_max={max_val:.2e})");
    if max_abs / max_val.max(1e-30) > 1e-3 {
        println!("  WARNING: relative error > 1e-3");
        println!("  First 8 baseline: {:?}", &out_baseline[..8]);
        println!("  First 8 flash:    {:?}", &out_flash[..8]);
    }
}
