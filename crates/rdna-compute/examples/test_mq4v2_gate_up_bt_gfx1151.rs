// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! MQ4V2 gate+up BT parity on exact gfx1151 — batch-column reuse gate.
//!
//! Pack deterministic MQ4V2 gate/up weights with disjoint halves (half0 in
//! [-1,1], half1 in [96,160]) so a wrong half-header or lane test is
//! unmissable, and F32 X. Distinct deterministic weights per projection
//! (different seeds) so a swapped routing is caught. Row sizes are
//! routing-sensitive (gate=28/up=44 is not 16-aligned per projection; total
//! 72 has an 8-row tail) so straddling tiles and tail guards are exercised.
//! Distinct NaN sentinels per projection fill outputs before launch so an
//! unwritten tail is caught. On exact gfx1151:
//!   1) run the historical base `gemm_gate_up_mq4g256v2_wmma` into reference
//!      outputs (force base by arming capture_mode so production BT is skipped),
//!   2) run the direct `gemm_gate_up_mq4g256v2_wmma_gfx1151_bt` BT6/BT12
//!      launchers into separate outputs for N=128 and N=256,
//!  and compare both projections (gate and up) for raw f32 bit equality
//!  (`got.to_bits() == want.to_bits()` on every element) as a hard gate, plus
//!  finite/nondegenerate, relL2<=1e-5, cosine>=1-1e-6, reporting max abs.
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

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

fn build_disjoint_halves_seeded(m: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let salt = (r * 7919 + g * 104_729) as u32 ^ seed;
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
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };

    let arch = gpu.arch.clone();
    let is_gfx1151 = gpu.arch_caps.is_gfx1151() && arch == "gfx1151";
    if !is_gfx1151 {
        eprintln!("SKIP: arch {arch} is not exact gfx1151 — harness requires gfx1151 only");
        return;
    }
    eprintln!("arch {arch} confirmed exact gfx1151 — running gate+up BT parity");

    if gpu.active_capture.is_some() {
        eprintln!("SKIP: active_capture is Some — BT harness requires no capture");
        return;
    }

    // Boundary-crossing unequal dims: gate=28/up=44 are not 16-aligned (both
    // remainder 12), distinct, and total 72 has an 8-row tail so tail guards
    // and straddling are exercised. Distinct seeds catch swapped routing.
    let gate_m: usize = 28;
    let up_m: usize = 44;
    let k: usize = 512;
    assert_eq!(k % 256, 0);
    assert_eq!((gate_m + up_m) % 16, 8, "total rows should have 8-row tail");

    let w_gate = build_disjoint_halves_seeded(gate_m, k, 0x1111_2222);
    let w_up = build_disjoint_halves_seeded(up_m, k, 0x3333_4444);
    let blob_gate = pack_mq4g256v2(&w_gate, gate_m, k);
    let blob_up = pack_mq4g256v2(&w_up, up_m, k);
    assert_eq!(blob_gate.len(), gate_m * (k / GROUP) * GROUP_BYTES);
    assert_eq!(blob_up.len(), up_m * (k / GROUP) * GROUP_BYTES);
    eprintln!(
        "packed gate {} B, up {} B, K={k}, gate_m={gate_m}, up_m={up_m}",
        blob_gate.len(),
        blob_up.len()
    );

    let d_gate = gpu
        .upload_raw(&blob_gate, &[blob_gate.len()])
        .expect("upload gate");
    let d_up = gpu
        .upload_raw(&blob_up, &[blob_up.len()])
        .expect("upload up");

    let ns = [128usize, 256usize];
    let bts = [6usize, 12];

    let sentinel_gate = f32::from_bits(0x7FC0_0001);
    let sentinel_up = f32::from_bits(0x7FC0_0002);

    let mut all_ok = true;

    for &n in &ns {
        eprintln!("\n=== N={n} ===");
        let x_host: Vec<f32> = (0..n * k)
            .map(|i| prng(i, 0xC0FF_EE00) * 2.0 - 1.0)
            .collect();
        assert_eq!(x_host.len(), n * k);

        let d_x = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x");
        let d_y_gate_ref = gpu
            .alloc_tensor(&[n * gate_m], DType::F32)
            .expect("alloc y_gate ref");
        let d_y_up_ref = gpu
            .alloc_tensor(&[n * up_m], DType::F32)
            .expect("alloc y_up ref");

        gpu.hip
            .memcpy_htod(&d_x.buf, unsafe {
                std::slice::from_raw_parts(x_host.as_ptr() as *const u8, x_host.len() * 4)
            })
            .expect("htod x");
        gpu.hip.device_synchronize().expect("sync after htod");

        for (buf, sen, len) in [
            (&d_y_gate_ref, sentinel_gate, n * gate_m),
            (&d_y_up_ref, sentinel_up, n * up_m),
        ] {
            let host = vec![sen; len];
            gpu.hip
                .memcpy_htod(&buf.buf, unsafe {
                    std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
                })
                .expect("htod sentinel ref");
        }
        gpu.hip.device_synchronize().expect("sync sentinel ref");

        let saved_capture = gpu.graphs.capture_mode;
        gpu.graphs.capture_mode = true;
        gpu.hip.device_synchronize().unwrap();
        let t_ref = std::time::Instant::now();
        let ref_res = gpu.gemm_gate_up_mq4g256v2_wmma(
            &d_gate,
            &d_up,
            &d_x,
            &d_y_gate_ref,
            &d_y_up_ref,
            gate_m,
            up_m,
            k,
            n,
        );
        gpu.graphs.capture_mode = saved_capture;
        ref_res.expect("base gemm_gate_up_mq4g256v2_wmma failed");
        gpu.hip.device_synchronize().expect("sync ref");
        let ref_us = t_ref.elapsed().as_secs_f64() * 1e6;
        eprintln!("  base ref N={n} done in {ref_us:.1} us");

        let y_gate_ref = gpu.download_f32(&d_y_gate_ref).expect("download gate ref");
        let y_up_ref = gpu.download_f32(&d_y_up_ref).expect("download up ref");
        assert_eq!(y_gate_ref.len(), n * gate_m);
        assert_eq!(y_up_ref.len(), n * up_m);

        for (label, data, sen) in [
            ("gate_ref", &y_gate_ref, sentinel_gate),
            ("up_ref", &y_up_ref, sentinel_up),
        ] {
            let cnt = data.iter().filter(|x| x.to_bits() == sen.to_bits()).count();
            assert_eq!(cnt, 0, "ref {label} still contains sentinel NaN N={n}");
        }

        assert!(is_finite(&y_gate_ref), "ref gate not finite N={n}");
        assert!(is_finite(&y_up_ref), "ref up not finite N={n}");
        assert!(variance(&y_gate_ref) > 1e-12, "ref gate degenerate N={n}");
        assert!(variance(&y_up_ref) > 1e-12, "ref up degenerate N={n}");

        for &bt in &bts {
            let d_y_gate_bt = gpu
                .alloc_tensor(&[n * gate_m], DType::F32)
                .expect("alloc y_gate bt");
            let d_y_up_bt = gpu
                .alloc_tensor(&[n * up_m], DType::F32)
                .expect("alloc y_up bt");

            for (buf, sen, len) in [
                (&d_y_gate_bt, sentinel_gate, n * gate_m),
                (&d_y_up_bt, sentinel_up, n * up_m),
            ] {
                let host = vec![sen; len];
                gpu.hip
                    .memcpy_htod(&buf.buf, unsafe {
                        std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
                    })
                    .expect("htod sentinel bt");
            }
            gpu.hip.device_synchronize().expect("sync sentinel bt");

            let res: Result<(), _> = gpu.gemm_gate_up_mq4g256v2_wmma_gfx1151_bt(
                &d_gate,
                &d_up,
                &d_x,
                &d_y_gate_bt,
                &d_y_up_bt,
                gate_m,
                up_m,
                k,
                n,
                bt,
            );
            if let Err(e) = res {
                eprintln!("  BT{bt} wrapper launch failed N={n}: {e:?}");
                all_ok = false;
                continue;
            }
            gpu.hip.device_synchronize().expect("sync bt");

            let y_gate_bt = gpu.download_f32(&d_y_gate_bt).expect("download gate bt");
            let y_up_bt = gpu.download_f32(&d_y_up_bt).expect("download up bt");

            for (label, data, sen) in [
                ("gate_bt", &y_gate_bt, sentinel_gate),
                ("up_bt", &y_up_bt, sentinel_up),
            ] {
                let cnt = data.iter().filter(|x| x.to_bits() == sen.to_bits()).count();
                if cnt != 0 {
                    eprintln!("  BT{bt} {label} still contains {cnt} sentinel NaN(s) N={n}");
                    all_ok = false;
                }
            }

            let ok_gate = check_proj("gate", n, bt, &y_gate_bt, &y_gate_ref);
            let ok_up = check_proj("up  ", n, bt, &y_up_bt, &y_up_ref);
            if !ok_gate || !ok_up {
                eprintln!("  BT{bt} N={n} FAILED");
                all_ok = false;
            } else {
                eprintln!("  BT{bt} N={n} OK");
            }

            if n == 256 {
                let t = std::time::Instant::now();
                for (buf, sen, len) in [
                    (&d_y_gate_bt, sentinel_gate, n * gate_m),
                    (&d_y_up_bt, sentinel_up, n * up_m),
                ] {
                    let host = vec![sen; len];
                    gpu.hip
                        .memcpy_htod(&buf.buf, unsafe {
                            std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
                        })
                        .unwrap();
                }
                gpu.hip.device_synchronize().unwrap();
                gpu.gemm_gate_up_mq4g256v2_wmma_gfx1151_bt(
                    &d_gate,
                    &d_up,
                    &d_x,
                    &d_y_gate_bt,
                    &d_y_up_bt,
                    gate_m,
                    up_m,
                    k,
                    n,
                    bt,
                )
                .expect("re-launch for timing");
                gpu.hip.device_synchronize().unwrap();
                let us = t.elapsed().as_secs_f64() * 1e6;
                eprintln!("    timing N256 BT{bt}: {us:.1} us (host Instant+sync)");
            }
        }
    }

    if all_ok {
        eprintln!("\nPASS: all N={{128,256}} x BT{{6,12}} x {{gate,up}} raw-bit equal (f32::to_bits); finite/nondegenerate; relL2<=1e-5 cosine>=1-1e-6; maxAbs reported per variant");
    } else {
        eprintln!(
            "\nFAIL: one or more parity checks violated raw-bit equality or numeric thresholds"
        );
        std::process::exit(1);
    }
}
