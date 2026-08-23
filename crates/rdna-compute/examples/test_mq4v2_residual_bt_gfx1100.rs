// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! MQ4V2 residual BT parity on exact gfx1100 — batch-column reuse gate.
//!
//! Pack deterministic MQ4V2 residual weights with disjoint halves (half0 in
//! [-1,1], half1 in [96,160]) so a wrong half-header or lane test is
//! unmissable, and F32 X.  On exact gfx1100:
//!   1) run the historical base `gemm_mq4g256v2_residual_wmma` into reference
//!      outputs (force base by arming capture_mode so production BT is skipped),
//!   2) run the direct `gemm_mq4g256v2_residual_wmma_gfx1100_bt` BT4/6/8
//!      launchers into separate outputs for N=128 and N=256,
//!  and compare for raw f32 bit equality (`got.to_bits() == want.to_bits()` on
//!  every element) as a hard gate, plus finite/nondegenerate, relL2<=1e-5,
//!  cosine>=1-1e-6, reporting max abs. Preserves fused Y += W@X exactly,
//!  including nonzero deterministic initial residual in the harness.
//! On any other arch the harness SKIPs cleanly (exit 0, no GPU work).
//! Absolute target: hipfire-quant-quality only.

use rdna_compute::kv_slots::half_from_f32;
use rdna_compute::{DType, Gpu};

const GROUP: usize = 256;
const HALF: usize = 128;
const GROUP_BYTES: usize = 136;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let mut exp = ((bits >> 10) & 0x1f) as u32;
    let mut mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            exp = 127 - 15 + 1;
            while mant & 0x0400 == 0 {
                mant <<= 1;
                exp -= 1;
            }
            sign | (exp << 23) | ((mant & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

// Deterministic prng in [0,1).
fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

fn build_disjoint_halves(m: usize, k: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let salt = (r * 7919 + g * 104_729) as u32;
            for i in 0..HALF {
                w[base + i] = prng(i, salt) * 2.0 - 1.0;
            }
            for i in HALF..GROUP {
                w[base + i] = 96.0 + prng(i, salt ^ 0xA5A5_A5A5) * 64.0;
            }
        }
    }
    w
}

fn pack_mq4g256v2(w: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % GROUP, 0, "k must be multiple of 256");
    assert_eq!(w.len(), m * k);
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GROUP_BYTES];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * GROUP_BYTES;
            let mut codes = [0u8; GROUP];
            for h in 0..2 {
                let off = h * HALF;
                let slice = &w[src + off..src + off + HALF];
                let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
                let s_bits = if hi == lo { 0u16 } else { half_from_f32(step) };
                let z_bits = half_from_f32(lo);
                blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&s_bits.to_le_bytes());
                blob[dst + h * 4 + 2..dst + h * 4 + 4].copy_from_slice(&z_bits.to_le_bytes());
                let s_rt = f16_to_f32(s_bits);
                let z_rt = f16_to_f32(z_bits);
                if s_rt == 0.0 {
                    continue;
                }
                let inv = 1.0 / s_rt;
                for i in 0..HALF {
                    let q = ((slice[i] - z_rt) * inv + 0.5).floor().clamp(0.0, 15.0);
                    codes[off + i] = q as u8;
                }
            }
            for i in 0..HALF {
                let lo_q = codes[2 * i] & 0xF;
                let hi_q = codes[2 * i + 1] & 0xF;
                blob[dst + 8 + i] = lo_q | (hi_q << 4);
            }
        }
    }
    blob
}

fn is_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn variance(v: &[f32]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / v.len() as f64;
    var
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    if den == 0.0 {
        if num == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (num / den).sqrt()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x as f64 * *y as f64;
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn check_proj(label: &str, n: usize, bt: usize, got: &[f32], want: &[f32]) -> bool {
    let finite = is_finite(got) && is_finite(want);
    let var_got = variance(got);
    let var_want = variance(want);
    let nondeg = var_got > 1e-12 && var_want > 1e-12;
    let r = rel_l2(got, want);
    let c = cosine(got, want);
    let m = max_abs_diff(got, want);
    let bit_eq = got.len() == want.len()
        && got
            .iter()
            .zip(want.iter())
            .all(|(g, w)| g.to_bits() == w.to_bits());
    // Raw f32 bit equality is mandatory; numeric tolerances cannot substitute.
    let ok = bit_eq && finite && nondeg && r <= 1e-5 && c >= 1.0 - 1e-6;
    let status = if ok { "PASS" } else { "FAIL" };
    eprintln!(
        "  [{label} N={n} BT={bt}] bitEq={bit_eq} finite={finite} nondeg={nondeg} (var {var_got:.3e}/{var_want:.3e}) relL2={r:.3e} cosine={c:.9} maxAbs={m:.3e} [{status}]"
    );
    if !ok {
        eprintln!(
            "    thresholds: bitEq (raw f32 to_bits equality, all elements), relL2<=1e-5, cosine>=1-1e-6, finite, variance>1e-12"
        );
        if !bit_eq {
            let mism = got
                .iter()
                .zip(want.iter())
                .filter(|(g, w)| g.to_bits() != w.to_bits())
                .count();
            eprintln!(
                "    GATE FAIL: raw-bit equality violated — {mism} element(s) differ by to_bits() (includes +0/-0 and any other bit difference)"
            );
        }
        if r > 1e-5 {
            eprintln!("    relL2 violated: {r:.3e} > 1e-5");
        }
        if c < 1.0 - 1e-6 {
            eprintln!("    cosine violated: {c:.9} < {}", 1.0 - 1e-6);
        }
        if !finite {
            eprintln!("    non-finite values detected");
        }
        if !nondeg {
            eprintln!("    degenerate (near-zero variance) — possible zeroed output or tail bug");
        }
    }
    ok
}

fn main() {
    // Exact gfx1100 gate only.
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };

    // Exact arch check: only gfx1100 passes. Other arch SKIP cleanly.
    let arch = gpu.arch.clone();
    let is_gfx1100 = gpu.arch_caps.is_gfx1100() && arch == "gfx1100";
    if !is_gfx1100 {
        eprintln!("SKIP: arch {arch} is not exact gfx1100 — harness requires gfx1100 only");
        return;
    }
    eprintln!("arch {arch} confirmed exact gfx1100 — running residual BT parity (Y+=W@X)");

    // Capture check: if a capture is armed, skip to avoid polluting.
    if gpu.active_capture.is_some() {
        eprintln!("SKIP: active_capture is Some — BT harness requires no capture");
        return;
    }

    // Deterministic shapes: small but realistic, K multiple of 256, rows tiling 16.
    let m: usize = 64;
    let k: usize = 512;
    assert_eq!(k % 256, 0);
    assert_eq!(m % 16, 0, "m should tile 16 for clean dispatch");

    // Pack residual with disjoint halves.
    let w = build_disjoint_halves(m, k);
    let blob = pack_mq4g256v2(&w, m, k);
    assert_eq!(blob.len(), m * (k / GROUP) * GROUP_BYTES);
    eprintln!("packed residual {} B, K={k}, m={m}", blob.len());

    let d_a = gpu
        .upload_raw(&blob, &[blob.len()])
        .expect("upload residual");

    let ns = [128usize, 256usize];
    let bts = [4usize, 6, 8];

    let mut all_ok = true;

    for &n in &ns {
        eprintln!("\n=== N={n} ===");
        // Deterministic F32 X in [-1,1], row-major [N, K] as expected by GEMM.
        let x_host: Vec<f32> = (0..n * k)
            .map(|i| prng(i, 0xC0FF_EE00) * 2.0 - 1.0)
            .collect();
        assert_eq!(x_host.len(), n * k);
        // Deterministic nonzero initial residual Y in [-1,1] with offset to avoid zero.
        let y_init_host: Vec<f32> = (0..n * m)
            .map(|i| prng(i, 0xBEEF_1234) * 2.0 - 1.0 + 0.5)
            .collect();
        assert_eq!(y_init_host.len(), n * m);

        // Allocate X.
        let d_x = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x");
        gpu.hip
            .memcpy_htod(&d_x.buf, unsafe {
                std::slice::from_raw_parts(x_host.as_ptr() as *const u8, x_host.len() * 4)
            })
            .expect("htod x");
        gpu.hip.device_synchronize().expect("sync after htod x");

        // Reference Y.
        let d_y_ref = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y ref");
        gpu.hip
            .memcpy_htod(&d_y_ref.buf, unsafe {
                std::slice::from_raw_parts(y_init_host.as_ptr() as *const u8, y_init_host.len() * 4)
            })
            .expect("htod y_ref");
        gpu.hip.device_synchronize().expect("sync after htod y_ref");

        // Reference: historical base kernel. Production BT auto-selects via env
        // outside capture, so arm capture_mode for this launch only to retain
        // the base object as the arithmetic oracle.
        let saved_capture = gpu.graphs.capture_mode;
        gpu.graphs.capture_mode = true;
        gpu.hip.device_synchronize().unwrap();
        let t_ref = std::time::Instant::now();
        let ref_res = gpu.gemm_mq4g256v2_residual_wmma(&d_a, &d_x, &d_y_ref, m, k, n);
        gpu.graphs.capture_mode = saved_capture;
        ref_res.expect("base gemm_mq4g256v2_residual_wmma failed");
        gpu.hip.device_synchronize().expect("sync ref");
        let ref_us = t_ref.elapsed().as_secs_f64() * 1e6;
        eprintln!("  base ref N={n} done in {ref_us:.1} us");

        let y_ref = gpu.download_f32(&d_y_ref).expect("download ref");
        assert_eq!(y_ref.len(), n * m);

        // Quick sanity on ref: finite + nondegenerate.
        assert!(is_finite(&y_ref), "ref not finite N={n}");
        assert!(variance(&y_ref) > 1e-12, "ref degenerate N={n}");

        for &bt in &bts {
            let d_y_bt = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y bt");
            // Re-init BT Y with same deterministic residual.
            gpu.hip
                .memcpy_htod(&d_y_bt.buf, unsafe {
                    std::slice::from_raw_parts(
                        y_init_host.as_ptr() as *const u8,
                        y_init_host.len() * 4,
                    )
                })
                .expect("htod y_bt");
            gpu.hip.device_synchronize().expect("sync y_bt init");

            let res: Result<(), _> =
                gpu.gemm_mq4g256v2_residual_wmma_gfx1100_bt(&d_a, &d_x, &d_y_bt, m, k, n, bt);
            if let Err(e) = res {
                eprintln!("  BT{bt} wrapper launch failed N={n}: {e:?}");
                all_ok = false;
                continue;
            }
            gpu.hip.device_synchronize().expect("sync bt");

            let y_bt = gpu.download_f32(&d_y_bt).expect("download bt");

            let ok = check_proj("resid", n, bt, &y_bt, &y_ref);
            if !ok {
                eprintln!("  BT{bt} N={n} FAILED");
                all_ok = false;
            } else {
                eprintln!("  BT{bt} N={n} OK");
            }

            // Optional timing at N=256.
            if n == 256 {
                // Re-init for timing.
                gpu.hip
                    .memcpy_htod(&d_y_bt.buf, unsafe {
                        std::slice::from_raw_parts(
                            y_init_host.as_ptr() as *const u8,
                            y_init_host.len() * 4,
                        )
                    })
                    .unwrap();
                gpu.hip.device_synchronize().unwrap();
                let t = std::time::Instant::now();
                gpu.gemm_mq4g256v2_residual_wmma_gfx1100_bt(&d_a, &d_x, &d_y_bt, m, k, n, bt)
                    .expect("re-launch for timing");
                gpu.hip.device_synchronize().unwrap();
                let us = t.elapsed().as_secs_f64() * 1e6;
                eprintln!("    timing N256 BT{bt}: {us:.1} us (host Instant+sync)");
            }
        }
    }

    if all_ok {
        eprintln!("\nPASS: all N={{128,256}} x BT{{4,6,8}} raw-bit equal (f32::to_bits); finite/nondegenerate; relL2<=1e-5 cosine>=1-1e-6; maxAbs reported per variant; Y+=W@X preserved with nonzero init");
    } else {
        eprintln!(
            "\nFAIL: one or more parity checks violated raw-bit equality or numeric thresholds"
        );
        std::process::exit(1);
    }
}
