// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! MQ4V2 qkv BT parity on exact gfx1201 — batch-column reuse candidate gate.
//!
//! Pack deterministic MQ4V2 q/k/v weights with distinct seeds and disjoint
//! halves. Routing-sensitive shapes q=40/k=32/v=48. Distinct NaN sentinels
//! per projection. On exact gfx1201:
//!   1) run current gfx12 production `gemm_qkv_hfq4g256_wmma_gfx12_mq4v2`
//!      as the arithmetic oracle,
//!   2) run direct `gemm_qkv_mq4g256v2_wmma_gfx1201_bt` BT4/8/12 for
//!      N∈{128,256,384},
//!  and compare all three projections for raw f32 bit equality plus
//!  finite/nondegenerate, relL2<=1e-5, cosine>=1-1e-6.
//! On any other arch the harness SKIPs cleanly (exit 0, no GPU work).

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

fn check_proj(label: &str, n: usize, variant: usize, got: &[f32], want: &[f32]) -> bool {
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
        "  [{label} N={n} var={variant}] bitEq={bit_eq} finite={finite} nondeg={nondeg} (var {var_got:.3e}/{var_want:.3e}) relL2={r:.3e} cosine={c:.9} maxAbs={m:.3e} [{status}]"
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
                "    GATE FAIL: raw-bit equality violated — {mism} element(s) differ by to_bits()"
            );
        }
        if r > 1e-5 {
            eprintln!("    relL2 violated: {r:.3e} > 1e-5");
        }
        if c < 1.0 - 1e-6 {
            eprintln!("    cosine violated: {c:.9} < {}", 1.0 - 1e-6);
        }
        if !finite {
            eprintln!("    non-finite values detected (unwritten tail or accidental += leaves quiet-NaN sentinel)");
        }
        if !nondeg {
            eprintln!("    degenerate (near-zero variance) — possible zeroed output or tail bug");
        }
    }
    ok
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

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };

    let arch = gpu.arch.clone();
    let is_gfx1201 = gpu.arch_caps.is_gfx1201() && arch == "gfx1201";
    if !is_gfx1201 {
        eprintln!("SKIP: arch {arch} is not exact gfx1201 — harness requires gfx1201 only");
        return;
    }
    eprintln!("arch {arch} confirmed exact gfx1201 — running qkv BT parity");

    if gpu.active_capture.is_some() {
        eprintln!("SKIP: active_capture is Some — BT harness requires no capture");
        return;
    }

    let q_m: usize = 40;
    let k_m: usize = 32;
    let v_m: usize = 48;
    let k: usize = 256; // not %512 so gfx12 ldsstage cannot steal the oracle
    assert_eq!(k % 256, 0);
    let total_m = q_m + k_m + v_m;
    eprintln!("q_m={q_m} k_m={k_m} v_m={v_m} total={total_m} K={k}");

    let w_q = build_disjoint_halves_seeded(q_m, k, 0x1111_2222);
    let w_k = build_disjoint_halves_seeded(k_m, k, 0x3333_4444);
    let w_v = build_disjoint_halves_seeded(v_m, k, 0x5555_6666);
    let blob_q = pack_mq4g256v2(&w_q, q_m, k);
    let blob_k = pack_mq4g256v2(&w_k, k_m, k);
    let blob_v = pack_mq4g256v2(&w_v, v_m, k);
    eprintln!(
        "packed q {} B, k {} B, v {} B",
        blob_q.len(),
        blob_k.len(),
        blob_v.len()
    );

    let d_q = gpu.upload_raw(&blob_q, &[blob_q.len()]).expect("upload q");
    let d_k = gpu.upload_raw(&blob_k, &[blob_k.len()]).expect("upload k");
    let d_v = gpu.upload_raw(&blob_v, &[blob_v.len()]).expect("upload v");

    let ns = [128usize, 256, 384];
    let bts = [4usize, 8, 12];
    let sentinel_q = f32::from_bits(0x7FC0_0001);
    let sentinel_k = f32::from_bits(0x7FC0_0002);
    let sentinel_v = f32::from_bits(0x7FC0_0003);
    let mut all_ok = true;

    for &n in &ns {
        eprintln!("\n=== N={n} ===");
        let x_host: Vec<f32> = (0..n * k)
            .map(|i| prng(i, 0xC0FF_EE00) * 2.0 - 1.0)
            .collect();

        let d_x = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x");
        let d_y_q_ref = gpu.alloc_tensor(&[n * q_m], DType::F32).expect("y_q ref");
        let d_y_k_ref = gpu.alloc_tensor(&[n * k_m], DType::F32).expect("y_k ref");
        let d_y_v_ref = gpu.alloc_tensor(&[n * v_m], DType::F32).expect("y_v ref");

        gpu.hip
            .memcpy_htod(&d_x.buf, unsafe {
                std::slice::from_raw_parts(x_host.as_ptr() as *const u8, x_host.len() * 4)
            })
            .expect("htod x");
        gpu.hip.device_synchronize().expect("sync x");

        for (buf, sen, len) in [
            (&d_y_q_ref, sentinel_q, n * q_m),
            (&d_y_k_ref, sentinel_k, n * k_m),
            (&d_y_v_ref, sentinel_v, n * v_m),
        ] {
            let host = vec![sen; len];
            gpu.hip
                .memcpy_htod(&buf.buf, unsafe {
                    std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
                })
                .expect("htod sentinel ref");
        }
        gpu.hip.device_synchronize().expect("sync sentinel ref");

        std::env::remove_var("HIPFIRE_MQ4V2_GFX1201_QKV_BT");
        let t_ref = std::time::Instant::now();
        let ref_res = gpu.gemm_qkv_hfq4g256_wmma_gfx12_mq4v2(
            &d_q, &d_k, &d_v, &d_x, &d_y_q_ref, &d_y_k_ref, &d_y_v_ref, q_m, k_m, v_m, k, n,
        );
        ref_res.expect("gfx12 base qkv oracle failed");
        gpu.hip.device_synchronize().expect("sync ref");
        let ref_us = t_ref.elapsed().as_secs_f64() * 1e6;
        eprintln!("  gfx12 base ref N={n} done in {ref_us:.1} us");

        let y_q_ref = gpu.download_f32(&d_y_q_ref).expect("download q ref");
        let y_k_ref = gpu.download_f32(&d_y_k_ref).expect("download k ref");
        let y_v_ref = gpu.download_f32(&d_y_v_ref).expect("download v ref");
        assert_eq!(y_q_ref.len(), n * q_m);
        assert_eq!(y_k_ref.len(), n * k_m);
        assert_eq!(y_v_ref.len(), n * v_m);

        for (label, data, sen) in [
            ("q_ref", &y_q_ref, sentinel_q),
            ("k_ref", &y_k_ref, sentinel_k),
            ("v_ref", &y_v_ref, sentinel_v),
        ] {
            let cnt = data.iter().filter(|x| x.to_bits() == sen.to_bits()).count();
            assert_eq!(cnt, 0, "ref {label} still contains sentinel NaN N={n}");
        }
        assert!(is_finite(&y_q_ref) && is_finite(&y_k_ref) && is_finite(&y_v_ref));
        assert!(
            variance(&y_q_ref) > 1e-12 && variance(&y_k_ref) > 1e-12 && variance(&y_v_ref) > 1e-12
        );

        for &bt in &bts {
            let d_y_q_bt = gpu.alloc_tensor(&[n * q_m], DType::F32).expect("y_q bt");
            let d_y_k_bt = gpu.alloc_tensor(&[n * k_m], DType::F32).expect("y_k bt");
            let d_y_v_bt = gpu.alloc_tensor(&[n * v_m], DType::F32).expect("y_v bt");
            for (buf, sen, len) in [
                (&d_y_q_bt, sentinel_q, n * q_m),
                (&d_y_k_bt, sentinel_k, n * k_m),
                (&d_y_v_bt, sentinel_v, n * v_m),
            ] {
                let host = vec![sen; len];
                gpu.hip
                    .memcpy_htod(&buf.buf, unsafe {
                        std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
                    })
                    .expect("htod sentinel bt");
            }
            gpu.hip.device_synchronize().expect("sync sentinel bt");

            let res = gpu.gemm_qkv_mq4g256v2_wmma_gfx1201_bt(
                &d_q, &d_k, &d_v, &d_x, &d_y_q_bt, &d_y_k_bt, &d_y_v_bt, q_m, k_m, v_m, k, n, bt,
            );
            if let Err(e) = res {
                eprintln!("  BT{bt} wrapper launch failed N={n}: {e:?}");
                all_ok = false;
                continue;
            }
            gpu.hip.device_synchronize().expect("sync bt");

            let y_q_bt = gpu.download_f32(&d_y_q_bt).expect("download q bt");
            let y_k_bt = gpu.download_f32(&d_y_k_bt).expect("download k bt");
            let y_v_bt = gpu.download_f32(&d_y_v_bt).expect("download v bt");

            for (label, data, sen) in [
                ("q_bt", &y_q_bt, sentinel_q),
                ("k_bt", &y_k_bt, sentinel_k),
                ("v_bt", &y_v_bt, sentinel_v),
            ] {
                let cnt = data.iter().filter(|x| x.to_bits() == sen.to_bits()).count();
                if cnt != 0 {
                    eprintln!("  BT{bt} {label} still contains {cnt} sentinel NaN(s) N={n}");
                    all_ok = false;
                }
            }

            let ok_q = check_proj("q", n, bt, &y_q_bt, &y_q_ref);
            let ok_k = check_proj("k", n, bt, &y_k_bt, &y_k_ref);
            let ok_v = check_proj("v", n, bt, &y_v_bt, &y_v_ref);
            if !ok_q || !ok_k || !ok_v {
                eprintln!("  BT{bt} N={n} FAILED");
                all_ok = false;
            } else {
                eprintln!("  BT{bt} N={n} OK");
            }

            if n == 256 {
                for (buf, sen, len) in [
                    (&d_y_q_bt, sentinel_q, n * q_m),
                    (&d_y_k_bt, sentinel_k, n * k_m),
                    (&d_y_v_bt, sentinel_v, n * v_m),
                ] {
                    let host = vec![sen; len];
                    gpu.hip
                        .memcpy_htod(&buf.buf, unsafe {
                            std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
                        })
                        .unwrap();
                }
                gpu.hip.device_synchronize().unwrap();
                let t = std::time::Instant::now();
                gpu.gemm_qkv_mq4g256v2_wmma_gfx1201_bt(
                    &d_q, &d_k, &d_v, &d_x, &d_y_q_bt, &d_y_k_bt, &d_y_v_bt, q_m, k_m, v_m, k, n,
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
        eprintln!(
            "\nPASS: all N={{128,256,384}} x BT{{4,8,12}} x {{q,k,v}} raw-bit equal vs gfx12 base; finite/nondegenerate; relL2<=1e-5 cosine>=1-1e-6"
        );
    } else {
        eprintln!(
            "\nFAIL: one or more qkv BT parity checks violated raw-bit equality or numeric thresholds"
        );
        std::process::exit(1);
    }
}
