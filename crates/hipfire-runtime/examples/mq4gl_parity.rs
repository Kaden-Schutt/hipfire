// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
//! GO/NO-GO gate for dense MQ4-G256-GL (qt=40) GEMV.
//!
//! Packs synthetic blobs against `rdna_compute::GL_CB4` (hipfire-quantize is a
//! bin crate, so the encoder is not linked — this validates kernel vs spec;
//! encoder-vs-spec is Slice 1's job), runs `Gpu::gemv_mq4g256gl`, and compares
//! against an f64 host reference in the rotated domain.
//!
//! Run:
//!   cargo build --release -p hipfire-runtime --example mq4gl_parity
//!   ./target/release/examples/mq4gl_parity

use half::f16;
use rdna_compute::{Gpu, GL_CB4, GL_GROUP_SCALE_BYTES, GL_MQ4_GROUP_IDX_BYTES};
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    eprintln!("arch={}", gpu.arch);
    eprintln!("GL_CB4={:?}", GL_CB4);

    let mut any_fail = false;

    // --- codebook-order probe (catches cb-arg permutation) ---
    {
        let (ok, detail) = codebook_order_probe(&mut gpu);
        eprintln!("codebook_order_probe: {} ({})", if ok { "PASS" } else { "FAIL" }, detail);
        any_fail |= !ok;
    }

    // --- edge cases ---
    any_fail |= !run_edge_all_codes(&mut gpu, 0, "edge_all_codes_0");
    any_fail |= !run_edge_all_codes(&mut gpu, 15, "edge_all_codes_15");
    any_fail |= !run_edge_zero_scale(&mut gpu);

    // --- shape parity ---
    let shapes = [(1usize, 256usize), (3, 512), (17, 2048), (4096, 5120)];
    for &(m, k) in &shapes {
        let (ok, max_nerr, peak) = run_shape_parity(&mut gpu, m, k);
        eprintln!(
            "parity m={m} k={k}: {}  max_norm_err={:.6e}  peak={:.6e}",
            if ok { "PASS" } else { "FAIL" },
            max_nerr,
            peak
        );
        any_fail |= !ok;
    }
    // Multirow parity — same reference, including M%8 !=0 tail shapes.
    let multirow_shapes = [(1usize, 256usize), (3, 512), (17, 2048), (4096, 5120), (17, 256), (4095, 5120)];
    for rows in [2usize, 4, 8] {
        for &(m, k) in &multirow_shapes {
            let (ok, max_nerr, peak) = run_shape_parity_multirow(&mut gpu, m, k, rows);
            eprintln!(
                "multirow parity R={rows} m={m} k={k}: {}  max_norm_err={:.6e}  peak={:.6e}",
                if ok { "PASS" } else { "FAIL" },
                max_nerr,
                peak
            );
            any_fail |= !ok;
        }
    }

    // --- warm median timing at largest shape ---
    let (m, k) = (4096usize, 5120usize);
    let gpr = k / 256;
    let gl_bytes = m * gpr * 130;
    let lloyd_bytes = m * gpr * 160;
    let uniform_bytes = m * gpr * 136;

    let (uni_ok, uni_max_nerr) = run_uniform_parity(&mut gpu, m, k);
    eprintln!(
        "uniform_parity m={m} k={k}: {}  max_norm_err={:.6e}",
        if uni_ok { "PASS" } else { "FAIL" },
        uni_max_nerr
    );
    any_fail |= !uni_ok;

    // Host `Instant` timing around launch + device_synchronize carries a
    // MEASURED ~22.1 us constant sync round-trip on gfx1201 (flat across a 24x
    // byte range; see examples/dispatch_floor_probe.rs). HIP's dispatch floor
    // is only ~2.13 us, so that term is sync, not submission. Differences
    // between arms here are single-digit microseconds, i.e. small compared to
    // an uncontrolled 22 us term, so host medians alone cannot rank these
    // arms. Collect GPU-event time in parallel: the gemv/gemm wrappers already
    // feed `profile::begin_timer` with a byte count, so a single
    // start/stop around every timed call yields per-kernel device time.
    rdna_compute::profile::start();

    let gl_med = time_gemv_gl(&mut gpu, m, k);
    let lloyd_med = time_gemv_lloyd(&mut gpu, m, k);
    let uni_med = time_gemv_uniform(&mut gpu, m, k);
    let gl_gbps = (gl_bytes as f64) / gl_med / 1e9;
    let lloyd_gbps = (lloyd_bytes as f64) / lloyd_med / 1e9;
    let uni_gbps = (uniform_bytes as f64) / uni_med / 1e9;
    eprintln!(
        "timing m={m} k={k}: mq4g256gl median={:.3} us  mq4g256_lloyd median={:.3} us  (n=30 after 5 warmup)",
        gl_med * 1e6,
        lloyd_med * 1e6
    );
    eprintln!(
        "timing m={m} k={k}: mq4g256_uniform median={:.3} us  (n=30 after 5 warmup, gemv_mq4g256_prerotated)",
        uni_med * 1e6
    );
    eprintln!(
        "bandwidth m={m} k={k}: gl bytes={gl_bytes} gbps={gl_gbps:.2}  lloyd bytes={lloyd_bytes} gbps={lloyd_gbps:.2}  uniform bytes={uniform_bytes} gbps={uni_gbps:.2}"
    );
    for rows in [2usize, 4, 8] {
        let med = time_gemv_gl_multirow(&mut gpu, m, k, rows);
        let gbps = (gl_bytes as f64) / med / 1e9;
        eprintln!(
            "timing multirow R={rows} m={m} k={k}: median={:.3} us  gbps={:.2}  bytes={gl_bytes}  (n=30 after 5 warmup, gemv_mq4g256gl_multirow_r{rows})",
            med * 1e6,
            gbps
        );
    }

    // Device-side truth. Median per kernel over every timed launch above, with
    // the byte count the wrapper recorded, so GB/s is not re-derived from a
    // host-side assumption. Host medians above are retained for comparison;
    // the difference is the sync round-trip, not kernel work.
    if let Some(entries) = rdna_compute::profile::stop() {
        use std::collections::BTreeMap;
        let mut by_kernel: BTreeMap<&str, (Vec<f64>, usize)> = BTreeMap::new();
        for e in &entries {
            let slot = by_kernel.entry(e.kernel).or_insert_with(|| (Vec::new(), e.bytes));
            slot.0.push(e.time_us);
        }
        eprintln!("\nGPU-event time (device-side, excludes ~22.1 us host sync round-trip):");
        eprintln!(
            "  {:38} {:>8} {:>11} {:>10} {:>9}",
            "kernel", "n", "bytes", "median_us", "GB/s"
        );
        for (kernel, (mut times, bytes)) in by_kernel {
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med_us = times[times.len() / 2];
            let gbps = if bytes > 0 && med_us > 0.0 {
                bytes as f64 / (med_us * 1e-6) / 1e9
            } else {
                f64::NAN
            };
            eprintln!(
                "  {:38} {:>8} {:>11} {:>10.3} {:>9.1}",
                kernel,
                times.len(),
                bytes,
                med_us,
                gbps
            );
        }
    }

    if any_fail {
        eprintln!("\n[FAIL] one or more mq4gl parity checks failed");
        std::process::exit(1);
    }
    eprintln!("\n[PASS] all mq4gl parity / probe / edge checks passed");
}

// ---------------------------------------------------------------------------
// Deterministic PRNG
// ---------------------------------------------------------------------------

fn fract_sin(x: f32) -> f32 {
    (x.sin() * 12345.6789f32).fract() * 2.0f32 - 1.0f32
}

// ---------------------------------------------------------------------------
// Spec packing (mirrors quantize_mq4g256gl layout; no FWHT needed when we
// hand-pack codes + scales, or when we pack already-rotated weights)
// ---------------------------------------------------------------------------

fn pack_mq4g256gl_from_codes(codes: &[u8], scales_f32: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let gpr = k / 256;
    assert_eq!(codes.len(), m * k);
    assert_eq!(scales_f32.len(), m * gpr);
    let idx_bytes = m * gpr * GL_MQ4_GROUP_IDX_BYTES;
    let mut out = vec![0u8; idx_bytes + m * gpr * GL_GROUP_SCALE_BYTES];
    for row in 0..m {
        for g in 0..gpr {
            let base = (row * gpr + g) * GL_MQ4_GROUP_IDX_BYTES;
            let c0 = (row * k + g * 256) as usize;
            for b in 0..128 {
                let lo = codes[c0 + 2 * b] & 0x0F;
                let hi = codes[c0 + 2 * b + 1] & 0x0F;
                out[base + b] = lo | (hi << 4);
            }
            let sbits = f16::from_f32(scales_f32[row * gpr + g]).to_bits();
            let soff = idx_bytes + (row * gpr + g) * 2;
            out[soff] = (sbits & 0xFF) as u8;
            out[soff + 1] = (sbits >> 8) as u8;
        }
    }
    out
}

/// Pack F32 weights already in the *rotated* domain (no FWHT). Nearest-level
/// encode against GL_CB4 with per-block fp16 RMS scale — same arithmetic as
/// `gl_encode_block` in hipfire-quantize.
fn pack_mq4g256gl_rotated(w_rot: &[f32], m: usize, k: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    assert_eq!(w_rot.len(), m * k);
    assert_eq!(k % 256, 0);
    let gpr = k / 256;
    let mut codes = vec![0u8; m * k];
    let mut scales = vec![0.0f32; m * gpr];
    for row in 0..m {
        for g in 0..gpr {
            let start = row * k + g * 256;
            let group = &w_rot[start..start + 256];
            let ss: f64 = group.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let rms = (ss / 256.0).sqrt() as f32;
            let scale = f16::from_f32(rms).to_f32();
            scales[row * gpr + g] = scale;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for j in 0..256 {
                let z = group[j] * inv;
                let mut best = 0usize;
                let mut best_d = (z - GL_CB4[0]).abs();
                for (qi, &c) in GL_CB4.iter().enumerate().skip(1) {
                    let d = (z - c).abs();
                    if d < best_d {
                        best_d = d;
                        best = qi;
                    }
                }
                codes[start + j] = best as u8;
            }
        }
    }
    let blob = pack_mq4g256gl_from_codes(&codes, &scales, m, k);
    (blob, codes, scales)
}

// ---------------------------------------------------------------------------
// f64 host reference in rotated domain
// ---------------------------------------------------------------------------

fn ref_gemv_f64(codes: &[u8], scales: &[f32], x_rot: &[f32], m: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let gpr = k / 256;
    let mut y = vec![0.0f64; m];
    let mut sum_abs = vec![0.0f64; m];
    for row in 0..m {
        let mut acc = 0.0f64;
        let mut sab = 0.0f64;
        for g in 0..gpr {
            let scale = scales[row * gpr + g] as f64;
            let base = g * 256;
            let c0 = row * k + base;
            for j in 0..256 {
                let q = codes[c0 + j] as usize;
                let w = scale * (GL_CB4[q] as f64);
                let term = w * (x_rot[base + j] as f64);
                acc += term;
                sab += term.abs();
            }
        }
        y[row] = acc;
        sum_abs[row] = sab;
    }
    (y, sum_abs)
}

fn norm_err(got: f32, want: f64, sum_abs: f64, peak: f64) -> f64 {
    let denom = want
        .abs()
        .max(1e-3 * peak)
        .max(1e-6 * sum_abs)
        .max(1e-12);
    ((got as f64) - want).abs() / denom
}

// ---------------------------------------------------------------------------
// GPU launch helper
// ---------------------------------------------------------------------------

fn gpu_gemv_gl(gpu: &mut Gpu, blob: &[u8], x_rot: &[f32], m: usize, k: usize) -> Vec<f32> {
    let d_a = gpu.upload_raw(blob, &[blob.len()]).unwrap();
    let d_x = gpu.upload_f32(x_rot, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    gpu.gemv_mq4g256gl(&d_a, &d_x, &d_y, m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.download_f32(&d_y).unwrap()
}

fn gpu_gemv_gl_multirow(
    gpu: &mut Gpu,
    blob: &[u8],
    x_rot: &[f32],
    m: usize,
    k: usize,
    rows: usize,
) -> Vec<f32> {
    let d_a = gpu.upload_raw(blob, &[blob.len()]).unwrap();
    let d_x = gpu.upload_f32(x_rot, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    gpu.gemv_mq4g256gl_multirow_with_rows(&d_a, &d_x, &d_y, m, k, rows)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.download_f32(&d_y).unwrap()
}

// ---------------------------------------------------------------------------
// Codebook-order probe
// ---------------------------------------------------------------------------

fn codebook_order_probe(gpu: &mut Gpu) -> (bool, String) {
    // One group, codes cycle 0..15 across 256 weights. One-hot x at each of
    // 16 representative positions (weight j=q for q in 0..15) so y = scale * cb[q].
    let m = 1usize;
    let k = 256usize;
    let scale = 1.25f32;
    let mut codes = vec![0u8; 256];
    for j in 0..256 {
        codes[j] = (j % 16) as u8;
    }
    let scales = vec![scale];
    let blob = pack_mq4g256gl_from_codes(&codes, &scales, m, k);

    let mut ok = true;
    let mut msgs = Vec::new();
    for q in 0..16usize {
        let mut x = vec![0.0f32; k];
        x[q] = 1.0; // weight index q has code q
        let y = gpu_gemv_gl(gpu, &blob, &x, m, k);
        let want = scale as f64 * (GL_CB4[q] as f64);
        let got = y[0] as f64;
        let err = (got - want).abs();
        let pass = err <= 1e-5 * want.abs().max(1.0);
        if !pass {
            ok = false;
        }
        msgs.push(format!(
            "q={q}: got={got:.6} want={want:.6} err={err:.3e} {}",
            if pass { "ok" } else { "BAD" }
        ));
    }
    (ok, msgs.join("; "))
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

fn run_edge_all_codes(gpu: &mut Gpu, code: u8, name: &str) -> bool {
    let m = 1usize;
    let k = 256usize;
    let scale = 0.75f32;
    let codes = vec![code; 256];
    let scales = vec![scale];
    let blob = pack_mq4g256gl_from_codes(&codes, &scales, m, k);
    let x: Vec<f32> = (0..k).map(|i| fract_sin(i as f32 * 0.17 + 3.1)).collect();
    let y = gpu_gemv_gl(gpu, &blob, &x, m, k);
    let want: f64 = x
        .iter()
        .map(|&xi| (scale as f64) * (GL_CB4[code as usize] as f64) * (xi as f64))
        .sum();
    let err = ((y[0] as f64) - want).abs();
    let tol = 1e-4 * want.abs().max(1.0);
    let ok = err <= tol;
    eprintln!(
        "{name}: {}  got={:.6e} want={:.6e} abs_err={:.3e}",
        if ok { "PASS" } else { "FAIL" },
        y[0],
        want,
        err
    );
    ok
}

fn run_edge_zero_scale(gpu: &mut Gpu) -> bool {
    let m = 2usize;
    let k = 256usize;
    // row0: nonzero scale + codes; row1: scale=0 must contribute exactly 0
    let mut codes = vec![0u8; m * k];
    for j in 0..k {
        codes[j] = (j % 16) as u8;
        codes[k + j] = 15;
    }
    let scales = vec![1.0f32, 0.0f32];
    let blob = pack_mq4g256gl_from_codes(&codes, &scales, m, k);
    let x: Vec<f32> = (0..k).map(|i| 0.5 + fract_sin(i as f32 * 0.09)).collect();
    let y = gpu_gemv_gl(gpu, &blob, &x, m, k);
    let ok0 = y[0].is_finite() && y[0].abs() > 1e-6;
    let ok1 = y[1] == 0.0;
    let ok = ok0 && ok1;
    eprintln!(
        "edge_zero_scale: {}  y0={:.6e} y1={:.6e} (y1 must be exact 0)",
        if ok { "PASS" } else { "FAIL" },
        y[0],
        y[1]
    );
    ok
}

// ---------------------------------------------------------------------------
// Shape parity
// ---------------------------------------------------------------------------

fn run_shape_parity(gpu: &mut Gpu, m: usize, k: usize) -> (bool, f64, f64) {
    // Work entirely in the rotated domain: generate "rotated" weights and x
    // directly. Kernel assumes x is already rotated.
    let w_rot: Vec<f32> = (0..m * k)
        .map(|i| fract_sin(i as f32 * 0.731 + 1.337) * 0.5)
        .collect();
    let x_rot: Vec<f32> = (0..k)
        .map(|i| fract_sin(i as f32 * 0.513 + 2.719))
        .collect();

    let (blob, codes, scales) = pack_mq4g256gl_rotated(&w_rot, m, k);
    let (want, sum_abs) = ref_gemv_f64(&codes, &scales, &x_rot, m, k);
    let peak = want.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let got = gpu_gemv_gl(gpu, &blob, &x_rot, m, k);

    let mut max_nerr = 0.0f64;
    let mut ok = true;
    for row in 0..m {
        let nerr = norm_err(got[row], want[row], sum_abs[row], peak);
        if nerr > max_nerr {
            max_nerr = nerr;
        }
        if nerr > 1e-4 {
            ok = false;
            if row < 8 {
                eprintln!(
                    "  row {row}: got={:.6e} want={:.6e} nerr={:.3e}",
                    got[row], want[row], nerr
                );
            }
        }
    }
    (ok, max_nerr, peak)
}

fn run_shape_parity_multirow(
    gpu: &mut Gpu,
    m: usize,
    k: usize,
    rows: usize,
) -> (bool, f64, f64) {
    let w_rot: Vec<f32> = (0..m * k)
        .map(|i| fract_sin(i as f32 * 0.731 + 1.337) * 0.5)
        .collect();
    let x_rot: Vec<f32> = (0..k)
        .map(|i| fract_sin(i as f32 * 0.513 + 2.719))
        .collect();
    let (blob, codes, scales) = pack_mq4g256gl_rotated(&w_rot, m, k);
    let (want, sum_abs) = ref_gemv_f64(&codes, &scales, &x_rot, m, k);
    let peak = want.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let got = gpu_gemv_gl_multirow(gpu, &blob, &x_rot, m, k, rows);
    let mut max_nerr = 0.0f64;
    let mut ok = true;
    for row in 0..m {
        let nerr = norm_err(got[row], want[row], sum_abs[row], peak);
        if nerr > max_nerr {
            max_nerr = nerr;
        }
        if nerr > 1e-4 {
            ok = false;
            if row < 8 {
                eprintln!(
                    "  multirow R={rows} row {row}: got={:.6e} want={:.6e} nerr={:.3e}",
                    got[row], want[row], nerr
                );
            }
        }
    }
    (ok, max_nerr, peak)
}


// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

fn time_gemv_gl(gpu: &mut Gpu, m: usize, k: usize) -> f64 {
    let w_rot: Vec<f32> = (0..m * k)
        .map(|i| fract_sin(i as f32 * 0.11 + 0.3) * 0.4)
        .collect();
    let x_rot: Vec<f32> = (0..k).map(|i| fract_sin(i as f32 * 0.07 + 1.1)).collect();
    let (blob, _, _) = pack_mq4g256gl_rotated(&w_rot, m, k);
    let d_a = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
    let d_x = gpu.upload_f32(&x_rot, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

    for _ in 0..5 {
        gpu.gemv_mq4g256gl(&d_a, &d_x, &d_y, m, k).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    let mut times = Vec::with_capacity(30);
    for _ in 0..30 {
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        gpu.gemv_mq4g256gl(&d_a, &d_x, &d_y, m, k).unwrap();
        gpu.hip.device_synchronize().unwrap();
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn pack_mq4_lloyd_for_ab(w_rot: &[f32], m: usize, k: usize) -> Vec<u8> {
    // Interleaved 160 B/group: 16×fp16 codebook (GL_CB4 * scale) + 128 B indices.
    // Uses the same codes/scales as GL so the A/B is bandwidth-dominated.
    assert_eq!(k % 256, 0);
    let gpr = k / 256;
    let mut out = vec![0u8; m * gpr * 160];
    for row in 0..m {
        for g in 0..gpr {
            let start = row * k + g * 256;
            let group = &w_rot[start..start + 256];
            let ss: f64 = group.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let rms = (ss / 256.0).sqrt() as f32;
            let scale = f16::from_f32(rms).to_f32();
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            let goff = (row * gpr + g) * 160;
            for qi in 0..16 {
                let bits = f16::from_f32(scale * GL_CB4[qi]).to_bits();
                out[goff + qi * 2] = (bits & 0xFF) as u8;
                out[goff + qi * 2 + 1] = (bits >> 8) as u8;
            }
            for b in 0..128 {
                let mut qlo = 0u8;
                let mut qhi = 0u8;
                {
                    let z = group[2 * b] * inv;
                    let mut best = 0usize;
                    let mut best_d = (z - GL_CB4[0]).abs();
                    for (qi, &c) in GL_CB4.iter().enumerate().skip(1) {
                        let d = (z - c).abs();
                        if d < best_d {
                            best_d = d;
                            best = qi;
                        }
                    }
                    qlo = best as u8;
                }
                {
                    let z = group[2 * b + 1] * inv;
                    let mut best = 0usize;
                    let mut best_d = (z - GL_CB4[0]).abs();
                    for (qi, &c) in GL_CB4.iter().enumerate().skip(1) {
                        let d = (z - c).abs();
                        if d < best_d {
                            best_d = d;
                            best = qi;
                        }
                    }
                    qhi = best as u8;
                }
                out[goff + 32 + b] = (qlo & 0x0F) | ((qhi & 0x0F) << 4);
            }
        }
    }
    out
}

fn time_gemv_lloyd(gpu: &mut Gpu, m: usize, k: usize) -> f64 {
    let w_rot: Vec<f32> = (0..m * k)
        .map(|i| fract_sin(i as f32 * 0.11 + 0.3) * 0.4)
        .collect();
    let x_rot: Vec<f32> = (0..k).map(|i| fract_sin(i as f32 * 0.07 + 1.1)).collect();
    let blob = pack_mq4_lloyd_for_ab(&w_rot, m, k);
    let d_a = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
    let d_x = gpu.upload_f32(&x_rot, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

    for _ in 0..5 {
        gpu.gemv_mq4g256_lloyd(&d_a, &d_x, &d_y, m, k).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    let mut times = Vec::with_capacity(30);
    for _ in 0..30 {
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        gpu.gemv_mq4g256_lloyd(&d_a, &d_x, &d_y, m, k).unwrap();
        gpu.hip.device_synchronize().unwrap();
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

// ---------------------------------------------------------------------------
// Uniform MQ4G256 (affine min-max, 136 B/group) — honest format peer
// ---------------------------------------------------------------------------

/// Pack already-rotated F32 weights into uniform MQ4G256 layout:
/// per group 136 B = [f32 scale][f32 min][128 B nibbles], groups contiguous,
/// row stride `gpr*136`. Genuine affine min-max fit — not GL codes reinterpreted.
fn pack_mq4g256_uniform_for_ab(w_rot: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(w_rot.len(), m * k);
    assert_eq!(k % 256, 0);
    let gpr = k / 256;
    let mut out = vec![0u8; m * gpr * 136];
    for row in 0..m {
        for g in 0..gpr {
            let start = row * k + g * 256;
            let group = &w_rot[start..start + 256];
            let mut mn = group[0];
            let mut mx = group[0];
            for &v in group.iter().skip(1) {
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
            }
            let scale = (mx - mn) / 15.0;
            let goff = (row * gpr + g) * 136;
            out[goff..goff + 4].copy_from_slice(&scale.to_le_bytes());
            out[goff + 4..goff + 8].copy_from_slice(&mn.to_le_bytes());
            for b in 0..128 {
                let qlo = if scale > 0.0 {
                    (((group[2 * b] - mn) / scale).round() as i32).clamp(0, 15) as u8
                } else {
                    0
                };
                let qhi = if scale > 0.0 {
                    (((group[2 * b + 1] - mn) / scale).round() as i32).clamp(0, 15) as u8
                } else {
                    0
                };
                out[goff + 8 + b] = (qlo & 0x0F) | ((qhi & 0x0F) << 4);
            }
        }
    }
    out
}

fn ref_gemv_uniform_f64(
    blob: &[u8],
    x_rot: &[f32],
    m: usize,
    k: usize,
) -> (Vec<f64>, Vec<f64>) {
    let gpr = k / 256;
    assert_eq!(blob.len(), m * gpr * 136);
    let mut y = vec![0.0f64; m];
    let mut sum_abs = vec![0.0f64; m];
    for row in 0..m {
        let mut acc = 0.0f64;
        let mut sab = 0.0f64;
        for g in 0..gpr {
            let goff = (row * gpr + g) * 136;
            let scale = f32::from_le_bytes(blob[goff..goff + 4].try_into().unwrap()) as f64;
            let minv = f32::from_le_bytes(blob[goff + 4..goff + 8].try_into().unwrap()) as f64;
            let base = g * 256;
            for b in 0..128 {
                let packed = blob[goff + 8 + b];
                let qlo = (packed & 0x0F) as f64;
                let qhi = ((packed >> 4) & 0x0F) as f64;
                let w0 = minv + qlo * scale;
                let w1 = minv + qhi * scale;
                let t0 = w0 * (x_rot[base + 2 * b] as f64);
                let t1 = w1 * (x_rot[base + 2 * b + 1] as f64);
                acc += t0 + t1;
                sab += t0.abs() + t1.abs();
            }
        }
        y[row] = acc;
        sum_abs[row] = sab;
    }
    (y, sum_abs)
}

fn gpu_gemv_uniform(gpu: &mut Gpu, blob: &[u8], x_rot: &[f32], m: usize, k: usize) -> Vec<f32> {
    let d_a = gpu.upload_raw(blob, &[blob.len()]).unwrap();
    let d_x = gpu.upload_f32(x_rot, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    // Pre-rotated path: isolates weight-decode (no rotation in this arm).
    gpu.gemv_mq4g256_prerotated(&d_a, &d_x, &d_y, m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.download_f32(&d_y).unwrap()
}

fn run_uniform_parity(gpu: &mut Gpu, m: usize, k: usize) -> (bool, f64) {
    let w_rot: Vec<f32> = (0..m * k)
        .map(|i| fract_sin(i as f32 * 0.731 + 1.337) * 0.5)
        .collect();
    let x_rot: Vec<f32> = (0..k)
        .map(|i| fract_sin(i as f32 * 0.513 + 2.719))
        .collect();

    let blob = pack_mq4g256_uniform_for_ab(&w_rot, m, k);
    let (want, sum_abs) = ref_gemv_uniform_f64(&blob, &x_rot, m, k);
    let peak = want.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let got = gpu_gemv_uniform(gpu, &blob, &x_rot, m, k);

    let mut max_nerr = 0.0f64;
    let mut ok = true;
    for row in 0..m {
        let nerr = norm_err(got[row], want[row], sum_abs[row], peak);
        if nerr > max_nerr {
            max_nerr = nerr;
        }
        if nerr > 1e-4 {
            ok = false;
            if row < 8 {
                eprintln!(
                    "  uniform row {row}: got={:.6e} want={:.6e} nerr={:.3e}",
                    got[row], want[row], nerr
                );
            }
        }
    }
    (ok, max_nerr)
}

fn time_gemv_uniform(gpu: &mut Gpu, m: usize, k: usize) -> f64 {
    let w_rot: Vec<f32> = (0..m * k)
        .map(|i| fract_sin(i as f32 * 0.11 + 0.3) * 0.4)
        .collect();
    let x_rot: Vec<f32> = (0..k).map(|i| fract_sin(i as f32 * 0.07 + 1.1)).collect();
    let blob = pack_mq4g256_uniform_for_ab(&w_rot, m, k);
    let d_a = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
    let d_x = gpu.upload_f32(&x_rot, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

    for _ in 0..5 {
        gpu.gemv_mq4g256_prerotated(&d_a, &d_x, &d_y, m, k).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    let mut times = Vec::with_capacity(30);
    for _ in 0..30 {
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        gpu.gemv_mq4g256_prerotated(&d_a, &d_x, &d_y, m, k).unwrap();
        gpu.hip.device_synchronize().unwrap();
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn time_gemv_gl_multirow(gpu: &mut Gpu, m: usize, k: usize, rows: usize) -> f64 {
    let w_rot: Vec<f32> = (0..m * k)
        .map(|i| fract_sin(i as f32 * 0.11 + 0.3) * 0.4)
        .collect();
    let x_rot: Vec<f32> = (0..k).map(|i| fract_sin(i as f32 * 0.07 + 1.1)).collect();
    let (blob, _, _) = pack_mq4g256gl_rotated(&w_rot, m, k);
    let d_a = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
    let d_x = gpu.upload_f32(&x_rot, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    for _ in 0..5 {
        gpu.gemv_mq4g256gl_multirow_with_rows(&d_a, &d_x, &d_y, m, k, rows)
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let mut times = Vec::with_capacity(30);
    for _ in 0..30 {
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        gpu.gemv_mq4g256gl_multirow_with_rows(&d_a, &d_x, &d_y, m, k, rows)
            .unwrap();
        gpu.hip.device_synchronize().unwrap();
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}
