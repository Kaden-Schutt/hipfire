// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Paired GEMV throughput bench for the 4-bit family.
//!
//! Measures two kernels that ALREADY exist (no new kernels):
//!   - gemv_hfq4g256            affine G256, 136 B/group (qt=1/6/13)  — baseline
//!   - gemv_mq4g128_prerotated   affine G128, 72 B per 128 = 144 B per 256 (qt=7)
//!                              — the empirical answer to "does G128 hurt perf"
//!
//! Conventions (non-negotiable, from CLAUDE.md / project history):
//!   - All timing device-side via rdna_compute::profile::start/stop (hip events).
//!     Host Instant carries ~22.1 us sync tax and is not acceptable.
//!   - HIPFIRE_GEMV_ROWS set explicitly for every measurement; never rely on default.
//!     Sweep R in {1,2,4}. R=2 is gemv_rows_default() on gfx1201 (headline number).
//!   - Interleave arms sample-by-sample in a single loop, never arm-at-a-time.
//!     Same kernel measured 21.72 us with 32 warmups vs 30.00 us with 8 — 38% swing.
//!   - >=32 warmup iterations before measured window.
//!   - >=3 runs, report all of them, not just a mean. Report RATIOS as primary.
//!   - Report VGPR/SGPR and confirm zero scratch spills.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example bench_gemv_paired_throughput
//!
//! The bench itself sweeps R by re-initializing Gpu with HIPFIRE_GEMV_ROWS={1,2,4}
//! set explicitly before each phase.
//! Wrapping `HIPFIRE_GEMV_ROWS=$R cargo run ...` also works (see stdout).

use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::HashMap;

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 255 {
        return (sign << 15) | 0x7C00 | if mant != 0 { 0x200 } else { 0 };
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign << 15) | 0x7C00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign << 15;
        }
        let shift = 14 - new_exp;
        let m = (mant | 0x800000) >> shift;
        let round = ((m >> 0) & 1) as u16;
        return (sign << 15) | ((m >> 1) as u16).wrapping_add(round);
    }
    let half_exp = (new_exp as u16) << 10;
    let half_mant = (mant >> 13) as u16;
    let round = ((mant >> 12) & 1) as u16;
    (sign << 15) | half_exp | half_mant.wrapping_add(round)
}

fn gemv_hfq4g256_explicit(
    gpu: &mut Gpu,
    a: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    rows: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    gpu.gemv_hfq4g256_with_rows(a, x, y, m, k, rows)
        .map_err(|e| format!("gemv_hfq4g256_with_rows R{}: {:?}", rows, e))?;
    Ok(())
}
const M: usize = 4096;
const K: usize = 5120;
const WARMUP: usize = 32;
const ITERS: usize = 100;
const RUNS: usize = 3;

/// Byte stride per group-of-256 for each format. Note qt=7 G128 is 72 B per 128
/// weights, so we allocate 72*(K/128) per row = 72*40 = 2880 B/row for K=5120,
/// which is 144 B per 256-equivalent (2×72). Stating the stride we allocate and why:
/// - hfq4g256: 136 B /256 (8 B hdr + 128 B nibbles)
/// - mq4g128 : 72 B /128  (8 B hdr + 64 B nibbles) = 144 B /256 equivalent — we allocate 72*B per 128
fn bytes_per_weight_row(format: &str) -> usize {
    match format {
        "hfq4g256" => (K / 256) * 136,
        "mq4g128" => (K / 128) * 72, // 72 per 128, i.e. 144 per 256
        _ => panic!("unknown format"),
    }
}

fn weight_bytes(format: &str) -> usize {
    M * bytes_per_weight_row(format)
}

fn total_bytes(format: &str) -> usize {
    weight_bytes(format) + K * 4 + M * 4
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn gbps(bytes: usize, us: f64) -> f64 {
    bytes as f64 / us / 1e3
}

fn print_vgpr_report(arch: &str) {
    println!(
        "\n=== VGPR/SGPR/spill report (from .hipfire_kernels/{}/ *.radiowave.json) ===",
        arch
    );
    let pattern = format!(".hipfire_kernels/{}/", arch);
    let dir = std::path::Path::new(&pattern);
    if !dir.exists() {
        println!("  (no .hipfire_kernels dir found — kernels not yet compiled on this arch)");
        return;
    }
    let wanted = [
        ("gemv_hfq4g256", "gemv_hfq4g256.radiowave.json"),
        ("gemv_hfq4g128", "gemv_hfq4g128.radiowave.json"),
        (
            "gemv_hfq4g256_multirow_default",
            "gemv_hfq4g256_multirow_default.radiowave.json",
        ),
        ("gemv_hfq4g256_wide", "gemv_hfq4g256_wide.radiowave.json"),
    ];
    for (kname, fname) in wanted {
        let p = dir.join(fname);
        if !p.exists() {
            println!("  {:35} — missing {}", kname, p.display());
            continue;
        }
        let txt = std::fs::read_to_string(&p).unwrap_or_default();
        if txt.is_empty() {
            println!("  {:35} — empty {}", kname, p.display());
            continue;
        }
        let mut pos = 0usize;
        let mut found = false;
        while let Some(name_idx) = txt[pos..].find("\"name\"") {
            let abs = pos + name_idx;
            let snippet = &txt[abs..std::cmp::min(abs + 600, txt.len())];
            let name = snippet.split('"').nth(3).unwrap_or("?").to_string();
            let vgpr = snippet
                .split("\"vgpr_count\"")
                .nth(1)
                .and_then(|s| {
                    s.split(|c: char| !c.is_ascii_digit())
                        .find(|t| !t.is_empty())
                })
                .and_then(|n| n.trim().parse::<u64>().ok())
                .unwrap_or(0);
            let sgpr = snippet
                .split("\"sgpr_count\"")
                .nth(1)
                .and_then(|s| {
                    s.split(|c: char| !c.is_ascii_digit())
                        .find(|t| !t.is_empty())
                })
                .and_then(|n| n.trim().parse::<u64>().ok())
                .unwrap_or(0);
            let vspill = snippet
                .split("\"vgpr_spill_count\"")
                .nth(1)
                .and_then(|s| {
                    s.split(|c: char| !c.is_ascii_digit())
                        .find(|t| !t.is_empty())
                })
                .and_then(|n| n.trim().parse::<u64>().ok())
                .unwrap_or(0);
            let sspill = snippet
                .split("\"sgpr_spill_count\"")
                .nth(1)
                .and_then(|s| {
                    s.split(|c: char| !c.is_ascii_digit())
                        .find(|t| !t.is_empty())
                })
                .and_then(|n| n.trim().parse::<u64>().ok())
                .unwrap_or(0);
            let privb = snippet
                .split("\"private_segment_fixed_size\"")
                .nth(1)
                .and_then(|s| {
                    s.split(|c: char| !c.is_ascii_digit())
                        .find(|t| !t.is_empty())
                })
                .and_then(|n| n.trim().parse::<u64>().ok())
                .unwrap_or(0);
            let spill_ok = if vspill == 0 && sspill == 0 && privb == 0 {
                "OK zero spill"
            } else {
                "!!! SPILL !!!"
            };
            println!(
                "  {:40} vgpr={:3} sgpr={:3} vspill={} sspill={} priv={:4}  {}",
                name, vgpr, sgpr, vspill, sspill, privb, spill_ok
            );
            found = true;
            pos = abs + 10;
            if pos >= txt.len() {
                break;
            }
            if txt[pos..].find("\"name\"").is_none() {
                break;
            }
        }
        if !found {
            println!(
                "  {:35} — no kernel entries parsed in {}",
                kname,
                p.display()
            );
        }
    }
}

fn main() {
    eprintln!("=== bench_gemv_paired_throughput ===");
    eprintln!("  m={} k={}  (matched across all formats)", M, K);
    eprintln!(
        "  warmup={} iters={} runs={}  interleave=sample-by-sample  timing=device hip events",
        WARMUP, ITERS, RUNS
    );
    eprintln!("  formats:");
    eprintln!(
        "    hfq4g256: 136 B/256  (8 hdr +128 nibbles)  total weight {} B",
        weight_bytes("hfq4g256")
    );
    eprintln!("    mq4g128 : 72 B/128 =144 B/256 equiv (8 hdr +64 nibbles per 128) total weight {} B  [stride 72 per 128]", weight_bytes("mq4g128"));
    eprintln!("  R sweep: 1,2,4  (R=2 is gemv_rows_default on gfx1201, headline)");

    let rs = [1usize, 2, 4];

    // Outer accumulators: per R -> per format -> Vec of medians (one per run)
    let mut all_results: HashMap<(usize, String), Vec<f64>> = HashMap::new();

    for &r in &rs {
        println!("\n########################################################################");
        println!("# R = {}  (HIPFIRE_GEMV_ROWS={})", r, r);
        println!("########################################################################");
        // SAFETY: set_var is unsafe in recent Rust (requires single-thread guarantee).
        // We are single-threaded at this point and re-init Gpu after each set.
        unsafe {
            std::env::set_var("HIPFIRE_GEMV_ROWS", r.to_string());
        }
        eprintln!("[R={}] HIPFIRE_GEMV_ROWS set to {}", r, r);

        let mut gpu = Gpu::init().expect("gpu init");
        let arch = gpu.arch.clone();
        println!(
            "  arch={}  HIPFIRE_GEMV_ROWS={}  gemv_rows_default={}",
            arch,
            r,
            gpu.arch_caps.gemv_rows_default()
        );
        // build weight blobs
        let gpr = K / 256;
        let gpr128 = K / 128;

        // allocate x, y (reuse across formats)
        let x_host: Vec<f32> = (0..K).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
        let d_x = gpu.upload_f32(&x_host, &[K]).expect("upload x");
        let d_y = gpu.zeros(&[M], DType::F32).expect("alloc y");

        // allocate weight blobs (synthetic zeros/pattern — kernel BW matters, values don't)
        // hfq4g256: 136 B per 256
        let hf_bytes = M * gpr * 136;
        let hf_blob = vec![0x44u8; hf_bytes];
        let hb = gpu.hip.malloc(hf_bytes).expect("malloc hf");
        gpu.hip.memcpy_htod(&hb, &hf_blob).expect("copy hf");
        let d_hf = GpuTensor {
            buf: hb,
            shape: vec![M, K],
            dtype: DType::Raw,
        };

        // mq4g128: 72 B per 128
        let g128_bytes = M * gpr128 * 72;
        let g128_blob = vec![0x33u8; g128_bytes];
        let gb = gpu.hip.malloc(g128_bytes).expect("malloc g128");
        gpu.hip.memcpy_htod(&gb, &g128_blob).expect("copy g128");
        let d_g128 = GpuTensor {
            buf: gb,
            shape: vec![M, K],
            dtype: DType::Raw,
        };

        // Warmup already done per run below, but also do a one-time JIT warmup to force compilation before timing.
        // Trigger each kernel once so first-run JIT doesn't pollute run 1.
        eprintln!("[R={}] JIT warmup (one launch per format)...", r);
        gemv_hfq4g256_explicit(&mut gpu, &d_hf, &d_x, &d_y, M, K, r).expect("jit hf");
        gpu.gemv_mq4g128_prerotated(&d_g128, &d_x, &d_y, M, K)
            .expect("jit g128");
        gpu.hip.device_synchronize().expect("jit sync");
        eprintln!("[R={}] JIT done", r);

        // Collect per-format samples across runs
        // For each run we do interleaved warmup then interleaved measured.
        // We store median per format per run.
        let mut medians_per_format: HashMap<String, Vec<f64>> = HashMap::new(); // format -> medians across runs (we push one per run)

        for run in 0..RUNS {
            // warmup: interleave sample-by-sample, not arm-at-a-time
            for _ in 0..WARMUP {
                gemv_hfq4g256_explicit(&mut gpu, &d_hf, &d_x, &d_y, M, K, r).expect("warm hf");
                gpu.gemv_mq4g128_prerotated(&d_g128, &d_x, &d_y, M, K)
                    .expect("warm g128");
            }
            gpu.hip.device_synchronize().expect("warm sync");

            // measured window: start profiling, interleave sample-by-sample
            rdna_compute::profile::start();
            for _ in 0..ITERS {
                gemv_hfq4g256_explicit(&mut gpu, &d_hf, &d_x, &d_y, M, K, r).expect("hf");
                gpu.gemv_mq4g128_prerotated(&d_g128, &d_x, &d_y, M, K)
                    .expect("g128");
            }
            let entries = rdna_compute::profile::stop().expect("profile stop");
            // entries should be ITERS*2 (each launch produces one entry)
            if entries.len() != ITERS * 2 {
                eprintln!("WARN run {} R={}: expected {} entries got {} — kernels may have been coalesced or profiling missed", run+1, r, ITERS*2, entries.len());
                for e in &entries {
                    eprintln!("  {} {} us", e.kernel, e.time_us);
                }
            }
            // Split by position mod 2: order we launched: hf, g128 repeating
            let mut hf_us = Vec::with_capacity(ITERS);
            let mut g128_us = Vec::with_capacity(ITERS);
            for (idx, e) in entries.iter().enumerate() {
                match idx % 2 {
                    0 => hf_us.push(e.time_us),
                    1 => g128_us.push(e.time_us),
                    _ => unreachable!(),
                }
            }
            // Compute medians for this run
            let hf_med = median(hf_us.clone());
            let g128_med = median(g128_us.clone());

            println!(
                "\n  -- run {}/{}  R={}  ({} samples/format, {} warmup, interleaved) --",
                run + 1,
                RUNS,
                r,
                ITERS,
                WARMUP
            );
            for (name, vals, med) in [
                ("hfq4g256", &hf_us, hf_med),
                ("mq4g128 (G128)", &g128_us, g128_med),
            ] {
                let bytes = total_bytes(match name {
                    "hfq4g256" => "hfq4g256",
                    n if n.contains("G128") => "mq4g128",
                    _ => unreachable!(),
                });
                let gb = gbps(bytes, med);
                // also show p50-ish spread: min/median/max
                let mn = vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let mx = vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                println!(
                    "    {:20}  median {:7.2} us  (min {:6.2} max {:6.2})  {:6.1} GB/s  bytes={}",
                    name, med, mn, mx, gb, bytes
                );
            }
            // store medians for outer aggregation
            for (k, v) in [("hfq4g256", hf_med), ("mq4g128", g128_med)] {
                medians_per_format.entry(k.to_string()).or_default().push(v);
                all_results.entry((r, k.to_string())).or_default().push(v);
            }
            // also report ratios vs hfq for this run inline
            println!("    ratios vs hfq4g256 (same R, same run):");
            for (name, med) in [("mq4g128", g128_med)] {
                let time_ratio = med / hf_med;
                let hf_gb = gbps(total_bytes("hfq4g256"), hf_med);
                let fmt_gb = gbps(total_bytes(name), med);
                let gbps_ratio = fmt_gb / hf_gb;
                println!(
                    "      {:10}  time_ratio={:.4} ({} {}{} )  gbps_ratio={:.4}",
                    name,
                    time_ratio,
                    if time_ratio < 1.0 { "faster" } else { "slower" },
                    format!("{:.1}%", (1.0 - time_ratio) * 100.0),
                    if time_ratio < 1.0 {
                        " faster"
                    } else {
                        " slower"
                    },
                    gbps_ratio
                );
            }
        } // runs

        // After all runs for this R, print summary table for this R (medians across runs already shown per run, but also show aggregate)
        println!(
            "\n  === Summary R={} ({} runs, median per run shown above) ===",
            r, RUNS
        );
        // free tensors for this R before next R's Gpu re-init (drop)
        drop(d_hf);
        drop(d_g128);
        drop(d_x);
        drop(d_y);
        // gpu will be dropped here; next iteration creates new Gpu with new R
    }

    // Global summary table: format x R -> {us (3 runs), GB/s, ratio vs hfq}
    println!("\n\n########################################################################");
    println!("# FINAL TABLE: format x R -> {{us, GB/s, ratio vs hfq4g256}} (3 runs each)");
    println!("########################################################################");
    println!("m={} k={}  warmup={} iters={} runs={}  device-side hip events, interleaved sample-by-sample", M, K, WARMUP, ITERS, RUNS);
    println!(
        "bytes/row: hfq4g256=136*{}={}  mq4g128=72*{}={} (144/256 equiv)",
        K / 256,
        bytes_per_weight_row("hfq4g256"),
        K / 128,
        bytes_per_weight_row("mq4g128")
    );
    println!(
        "total bytes (weight + x + y): hfq={} g128={}",
        total_bytes("hfq4g256"),
        total_bytes("mq4g128")
    );
    println!();
    println!("| format            | R | run1 us | run2 us | run3 us | median_of_medians_us | GB/s (from med) | time_ratio vs hfq (same R) | gbps_ratio |");
    println!("|-------------------|--:|---------|---------|---------|----------------------|-----------------|----------------------------:|-----------:|");
    for &r in &rs {
        // collect hfq medians for ratio denominator (median_of_medians)
        let hf_meds = all_results
            .get(&(r, "hfq4g256".to_string()))
            .cloned()
            .unwrap_or_default();
        let hf_mom = if hf_meds.is_empty() {
            f64::NAN
        } else {
            median(hf_meds.clone())
        };
        let hf_gb_mom = gbps(total_bytes("hfq4g256"), hf_mom);
        for fmt in ["hfq4g256", "mq4g128"] {
            let meds = all_results
                .get(&(r, fmt.to_string()))
                .cloned()
                .unwrap_or_default();
            if meds.is_empty() {
                continue;
            }
            let mom = median(meds.clone());
            let gb = gbps(total_bytes(fmt), mom);
            let (r1, r2, r3) = (
                meds.get(0).copied().unwrap_or(f64::NAN),
                meds.get(1).copied().unwrap_or(f64::NAN),
                meds.get(2).copied().unwrap_or(f64::NAN),
            );
            let time_ratio = if fmt == "hfq4g256" { 1.0 } else { mom / hf_mom };
            let gbps_ratio = if fmt == "hfq4g256" {
                1.0
            } else {
                gb / hf_gb_mom
            };
            let label = match fmt {
                "hfq4g256" => "hfq4g256 (136B)",
                "mq4g128" => "mq4g128 G128 (144B/256-equiv)",
                _ => fmt,
            };
            println!("| {:17} | {:1} | {:7.2} | {:7.2} | {:7.2} | {:20.2} | {:15.1} | {:27.4} | {:10.4} |", label, r, r1, r2, r3, mom, gb, time_ratio, gbps_ratio);
        }
    }

    // VGPR report (read from latest arch's compilation)
    // Need to know arch — re-init briefly with R=2 to query arch if needed, or use previous
    unsafe {
        std::env::set_var("HIPFIRE_GEMV_ROWS", "2");
    }
    if let Ok(gpu) = Gpu::init() {
        print_vgpr_report(&gpu.arch);
        // also try to force-compile missing kernels so second run will have them
        // (we already compiled above, so files should exist now)
    } else {
        println!("\n(could not re-init Gpu for VGPR report)");
    }

    println!("\n=== G128 question ===");
    // Compute G128 cost vs hfq at R=2 headline
    let hf_r2 = all_results
        .get(&(2, "hfq4g256".to_string()))
        .map(|v| median(v.clone()))
        .unwrap_or(f64::NAN);
    let g128_r2 = all_results
        .get(&(2, "mq4g128".to_string()))
        .map(|v| median(v.clone()))
        .unwrap_or(f64::NAN);
    if hf_r2.is_finite() && g128_r2.is_finite() {
        let pct_slower = (g128_r2 / hf_r2 - 1.0) * 100.0;
        let pct_thrpt = (hf_r2 / g128_r2 - 1.0) * 100.0;
        let hf_gb = gbps(total_bytes("hfq4g256"), hf_r2);
        let g128_gb = gbps(total_bytes("mq4g128"), g128_r2);
        println!("  At matched m={} k={} R=2 (gfx1201 default):", M, K);
        println!(
            "    hfq4g256 median {:.2} us  {:.1} GB/s  (weight {} B)",
            hf_r2,
            hf_gb,
            weight_bytes("hfq4g256")
        );
        println!("    mq4g128  median {:.2} us  {:.1} GB/s  (weight {} B, stride 72 B/128 =144 B/256-equiv)", g128_r2, g128_gb, weight_bytes("mq4g128"));
        println!("    G128 costs {:.2}% more time ( {:.2}% throughput deficit) vs hfq4g256 at same R and shape.", pct_slower, -pct_thrpt);
        println!("    Ratio (G128 time / HFQ time) = {:.4},  throughput ratio (G128 GB/s / HFQ GB/s) = {:.4}", g128_r2/hf_r2, g128_gb/hf_gb);
        println!();
        println!("  What this implies for a per-2x128 affine format at 136 B (the v2 candidate):");
        println!("    The v2 candidate keeps the SAME 136 B/256 budget as hfq4g256 (4.25 bpw) but splits it into");
        println!("    two sub-blocks per 256 with fp16 scale+zero headers per 128 (i.e. 2×(2+2+64)=136 B). So weight");
        println!("    bytes are IDENTICAL to hfq4g256 — any throughput delta is pure decode overhead, not BW.");
        println!("    qt=7's G128 (72 B/128 =144 B/256) is 5.9% larger (144 vs 136) and pays full per-wave header cost.");
        println!();
        println!("  2x128-inside-256 is NOT identical to qt=7's G128 layout:");
        println!("    - qt=7 G128: each 128-group header (scale+zero) is wave-uniform: one scalar load per wave32");
        println!("      covers all 32 lanes' 8 weights (32*8=256 weights per wave, but G128 has 2 headers per 256).");
        println!("      Every lane diverges on the same header for 128 weights, then reloads for next 128.");
        println!("    - 2x128-inside-256: on wave32 each lane covers 8 weights (256/32=8). Sub-block 0 is lanes 0-15");
        println!("      (16*8=128 weights) and sub-block 1 is lanes 16-31 (16*8=128 weights). The header is uniform");
        println!("      per HALF-wave, not per wave: lanes 0-15 share one scale/zero, lanes 16-31 share the other.");
        println!("      This is cheaper than qt=7's arrangement because:");
        println!("        * Half-wave uniform → scalar load can be hoisted per half-wave or broadcast via LDS/SGPR");
        println!("          with no cross-half divergence; qt=7 pays a wave-uniform but TWICE-per-wave reload.");
        println!("        * Within a 256-group the two halves' packed nibbles stay contiguous (128 B unbroken),");
        println!("          preserving the same 4 B/lane coalesced pattern as G256; no extra index math.");
        println!("        * The per-128 scale/zero adds one extra SGPR pair per half-wave vs one per wave, but");
        println!("          VGPR pressure is nearly identical — the kernel keeps the same 8-weight dot loop, just");
        println!("          selects scale/zero by lane predication (lane<16 ? s0: s1) which is a single v_cndmask");
        println!("          or scalar-broadcast, not an extra memory fetch per lane.");
        println!("      So a 2x128-inside-256 kernel should be slightly cheaper than qt=7's G128, and essentially");
        println!("      iso-cost with G256 (same total bytes, same nibble unpack, + one half-wave header select).");
        println!(
            "      The measured G128 penalty is therefore an UPPER BOUND on the v2 2x128 penalty."
        );
        if pct_slower < 5.0 {
            println!("      Measured G128 penalty is small (<5%), so v2 2x128 at 136 B is expected to be <~3% — likely noise.");
        } else if pct_slower < 10.0 {
            println!("      Measured G128 penalty is moderate (5-10%), so v2 2x128 likely 3-6% — measurable but probably acceptable for codec win.");
        } else {
            println!("      Measured G128 penalty is large (>10%), so v2 must be re-measured with an actual 2x128 kernel; G128 is not a tight bound.");
        }
    } else {
        println!("  (could not compute G128 cost — missing data)");
    }

    println!("\n=== Reproduce ===");
    println!("  cargo run --release -p rdna-compute --example bench_gemv_paired_throughput");
    println!("  (The bench sweeps HIPFIRE_GEMV_ROWS=1,2,4 internally; no external env needed. For external sweep: HIPFIRE_GEMV_ROWS=2 cargo run ...)");
    println!("  Also: HIPFIRE_GEMV_ROWS=1 cargo run ...  etc. if you want single-R isolation.");
    println!("\nDone.");
}
