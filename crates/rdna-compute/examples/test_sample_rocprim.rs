//! E5 screening: rocPRIM top-K sampler vs production parallel path.
//!
//! (a) DETERMINISM GATE — distinct-logits fixtures; flag-ON tokens must equal
//!     flag-OFF exactly across temperature × top_k × top_p × seeds (512 draws).
//! (b) BENCH — median us/call over 2000 iters, both routes, K=20 and K=64.
//!
//! Flag is process-latched, so ON/OFF arms run as fresh subprocesses of this
//! same binary.
//!
//! Run:
//!   cargo run -p rdna-compute --release --features lab --example test_sample_rocprim
//!
//! Fallback demo:
//!   HIPFIRE_ROCM_PATH=/bogus HIPFIRE_SAMPLE_ROCPRIM=1 \
//!     cargo run -p rdna-compute --release --features lab --example test_sample_rocprim

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const VOCAB: usize = 248_320;
const N_DRAWS: usize = 512;
const N_BENCH: usize = 2000;
const N_WARMUP: usize = 50;

fn distinct_logits(seed: u64) -> Vec<f32> {
    let mut h = vec![0.0f32; VOCAB];
    let mut s = seed;
    for i in 0..VOCAB {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let base = ((s >> 40) & 0xFFFFFF) as f32 * (40.0 / 16_777_216.0) - 20.0;
        h[i] = base + i as f32 * 1.0e-7;
    }
    h
}

fn median_us(samples: &mut [f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        0.5 * (samples[n / 2 - 1] + samples[n / 2])
    }
}

fn sample_once(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    result: &GpuTensor,
    repeat: &GpuTensor,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    seed: u32,
) -> Result<(u32, u32), Box<dyn std::error::Error>> {
    Ok(gpu.sample_top_p_pf(
        logits,
        result,
        repeat,
        VOCAB,
        temperature,
        top_p,
        seed,
        0,
        1.0,
        0.0,
        0.0,
        Some(top_k),
        None,
    )?)
}

fn draw_matrix() -> Vec<(f32, f32, u32, u32)> {
    let temps = [0.0f32, 0.7, 1.0];
    let ks = [20u32, 64];
    let ps = [0.9f32, 1.0];
    let mut out = Vec::with_capacity(N_DRAWS);
    let mut seed: u32 = 0x13579bdf;
    while out.len() < N_DRAWS {
        for &t in &temps {
            for &k in &ks {
                for &p in &ps {
                    if out.len() >= N_DRAWS {
                        break;
                    }
                    out.push((t, p, k, seed));
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                }
            }
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("HIPFIRE_SAMPLE_FAST", "0");

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--dump-tokens") {
        return child_dump_tokens();
    }
    if args.iter().any(|a| a == "--bench-one") {
        let k = args
            .iter()
            .find_map(|a| a.strip_prefix("--k=").map(|s| s.parse::<u32>().ok()))
            .flatten()
            .unwrap_or(20);
        return child_bench_one(k);
    }

    let mut gpu = Gpu::init()?;
    let arch = gpu
        .hip
        .get_arch(gpu.device_id)
        .unwrap_or_else(|_| gpu.arch.clone());
    println!("gcnArchName={arch}");
    println!("device_id={} arch_field={}", gpu.device_id, gpu.arch);

    // ── (a) DETERMINISM GATE ──────────────────────────────────────────
    println!("=== DETERMINISM GATE (distinct logits, {N_DRAWS} draws) ===");
    let off_tokens = collect_tokens_subprocess(false)?;
    let on_tokens = collect_tokens_subprocess(true)?;
    if off_tokens.len() != on_tokens.len() {
        eprintln!(
            "DETERMINISM FAIL: length mismatch off={} on={}",
            off_tokens.len(),
            on_tokens.len()
        );
        std::process::exit(1);
    }
    let mut mismatches = 0usize;
    let mut first_fail: Option<(usize, u32, u32, (f32, f32, u32, u32))> = None;
    let matrix = draw_matrix();
    for (i, (a, b)) in off_tokens.iter().zip(on_tokens.iter()).enumerate() {
        if a != b {
            mismatches += 1;
            if first_fail.is_none() {
                first_fail = Some((i, *a, *b, matrix[i]));
            }
            if mismatches <= 8 {
                let (t, p, k, seed) = matrix[i];
                eprintln!(
                    "DETERMINISM FAIL draw={i} temp={t} top_p={p} top_k={k} seed={seed:#x} off={a} on={b}"
                );
            }
        }
    }
    if mismatches > 0 {
        if let Some((i, a, b, (t, p, k, seed))) = first_fail {
            eprintln!(
                "DETERMINISM GATE FAIL: {mismatches}/{N_DRAWS} mismatches; first i={i} temp={t} top_p={p} top_k={k} seed={seed:#x} off={a} on={b}"
            );
        }
        std::process::exit(2);
    }
    println!("DETERMINISM GATE PASS ({N_DRAWS} draws, all tokens match)");

    // ── (b) BENCH ─────────────────────────────────────────────────────
    println!("=== BENCH (median of {N_BENCH} iters, warmup {N_WARMUP}) ===");
    println!(
        "{:<6} {:>14} {:>16} {:>10}",
        "K", "prod_med_us", "rocprim_med_us", "speedup"
    );
    for &k in &[20u32, 64u32] {
        let off_us = bench_subprocess(false, k)?;
        let on_us = bench_subprocess(true, k)?;
        let speedup = if on_us > 0.0 { off_us / on_us } else { 0.0 };
        println!("{k:<6} {off_us:>14.3} {on_us:>16.3} {speedup:>9.3}x");
    }

    // In-process smoke under ambient env.
    let result = gpu.zeros(&[2], DType::F32)?;
    let repeat = gpu.zeros(&[1], DType::F32)?;
    let h = distinct_logits(0xC0FFEE);
    let logits = gpu.upload_f32(&h, &[VOCAB])?;
    let (tok, _) = sample_once(&mut gpu, &logits, &result, &repeat, 0.7, 0.9, 20, 0xA5A5)?;
    println!("smoke token={tok}");

    gpu.free_tensor(logits)?;
    gpu.free_tensor(repeat)?;
    gpu.free_tensor(result)?;
    Ok(())
}

fn child_dump_tokens() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("HIPFIRE_SAMPLE_FAST", "0");
    let mut gpu = Gpu::init()?;
    let result = gpu.zeros(&[2], DType::F32)?;
    let repeat = gpu.zeros(&[1], DType::F32)?;
    let h = distinct_logits(0xDEAD_BEEF_u64);
    let logits = gpu.upload_f32(&h, &[VOCAB])?;
    for (t, p, k, seed) in draw_matrix() {
        let (tok, _) = sample_once(&mut gpu, &logits, &result, &repeat, t, p, k, seed)?;
        println!("{tok}");
    }
    gpu.free_tensor(logits)?;
    gpu.free_tensor(repeat)?;
    gpu.free_tensor(result)?;
    Ok(())
}

fn child_bench_one(k: u32) -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("HIPFIRE_SAMPLE_FAST", "0");
    let mut gpu = Gpu::init()?;
    let result = gpu.zeros(&[2], DType::F32)?;
    let repeat = gpu.zeros(&[1], DType::F32)?;
    let h = distinct_logits(0xC0FFEE_u64 + k as u64);
    let logits = gpu.upload_f32(&h, &[VOCAB])?;

    for _ in 0..N_WARMUP {
        let _ = sample_once(&mut gpu, &logits, &result, &repeat, 0.7, 0.9, k, 0xA5A5)?;
    }
    // Drain GPU before timed loop.
    gpu.hip.device_synchronize()?;

    let mut samples = Vec::with_capacity(N_BENCH);
    for i in 0..N_BENCH {
        let seed = 0xA5A5u32.wrapping_add(i as u32);
        let t0 = Instant::now();
        let _ = sample_once(&mut gpu, &logits, &result, &repeat, 0.7, 0.9, k, seed)?;
        // Include D2H of the 8-byte result (already inside sample_top_p_pf).
        samples.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    let med = median_us(&mut samples);
    println!("BENCH_MED_US={med:.6}");

    gpu.free_tensor(logits)?;
    gpu.free_tensor(repeat)?;
    gpu.free_tensor(result)?;
    Ok(())
}

fn collect_tokens_subprocess(rocprim: bool) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let exe = std::env::current_exe()?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("--dump-tokens");
    cmd.env("HIPFIRE_SAMPLE_FAST", "0");
    if rocprim {
        cmd.env("HIPFIRE_SAMPLE_ROCPRIM", "1");
    } else {
        cmd.env("HIPFIRE_SAMPLE_ROCPRIM", "0");
    }
    let output = cmd.output()?;
    if !output.status.success() {
        return Err(format!(
            "subprocess rocprim={rocprim} failed: {}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut tokens = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if let Ok(t) = line.parse::<u32>() {
            tokens.push(t);
        }
    }
    if tokens.len() != N_DRAWS {
        return Err(format!(
            "subprocess rocprim={rocprim} produced {} tokens, expected {N_DRAWS}\nstderr:\n{}",
            tokens.len(),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(tokens)
}

fn bench_subprocess(rocprim: bool, k: u32) -> Result<f64, Box<dyn std::error::Error>> {
    let exe = std::env::current_exe()?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("--bench-one");
    cmd.arg(format!("--k={k}"));
    cmd.env("HIPFIRE_SAMPLE_FAST", "0");
    if rocprim {
        cmd.env("HIPFIRE_SAMPLE_ROCPRIM", "1");
    } else {
        cmd.env("HIPFIRE_SAMPLE_ROCPRIM", "0");
    }
    let output = cmd.output()?;
    if !output.status.success() {
        return Err(format!(
            "bench subprocess rocprim={rocprim} k={k} failed: {}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("BENCH_MED_US=") {
            return Ok(rest.trim().parse()?);
        }
    }
    Err(format!(
        "bench subprocess missing BENCH_MED_US=; stdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    )
    .into())
}
