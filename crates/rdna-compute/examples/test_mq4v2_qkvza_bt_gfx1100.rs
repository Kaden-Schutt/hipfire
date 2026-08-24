// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! MQ4V2 qkvza BT parity on exact gfx1100 — batch-column reuse gate.
//!
//! Pack deterministic MQ4V2 qkv/z/beta/alpha weights with disjoint halves
//! (half0 in [-1,1], half1 in [96,160]) so a wrong half-header or lane test is
//! unmissable, and F32 X. Distinct deterministic weights per projection
//! (different seeds) so a swapped routing is caught. Row sizes are
//! routing-sensitive (qkv=40/z=28/beta=36/alpha=16 is not 16-aligned per
//! projection; total 120 has an 8-row tail) so straddling tiles and tail
//! guards are exercised. On exact gfx1100:
//!   1) run the historical base `gemm_qkvza_mq4g256v2_wmma` into reference
//!      outputs (force base by arming capture_mode so production BT policy is skipped),
//!   2) run the direct `gemm_qkvza_mq4g256v2_wmma_gfx1100_bt` launchers into
//!      separate outputs for N=128 and N=256 with production policy tiles
//!      BT∈{4,12},
//!  and compare all four projections (qkv,z,beta,alpha) for raw f32 bit equality
//!  (`got.to_bits() == want.to_bits()` on every element) as a hard gate, plus
//!  finite/nondegenerate, relL2<=1e-5, cosine>=1-1e-6, reporting max abs.
//!  Every reference/BT output buffer is initialized with four distinct
//!  quiet-NaN sentinels before launch so unwritten tails and accidental `+=`
//!  remain non-finite; finite + raw-bit equality proves routing/full
//!  overwrite/no accidental +=.
//! On any other arch the harness SKIPs cleanly (exit 0, no GPU work).
//! Absolute target: hipfire-quant-quality only.

use rdna_compute::kv_slots::half_from_f32;
use rdna_compute::{DType, Gpu, GpuTensor};

const GROUP: usize = 256;
const HALF: usize = 128;
const GROUP_BYTES: usize = 136;
/// Distinct quiet-NaN payloads per projection — mantissa differs.
const NAN_QKV_BITS: u32 = 0x7fc0_0001;
const NAN_Z_BITS: u32 = 0x7fc0_0002;
const NAN_BETA_BITS: u32 = 0x7fc0_0003;
const NAN_ALPHA_BITS: u32 = 0x7fc0_0004;
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

/// Fill an F32 device tensor with a quiet-NaN sentinel (sync HtoD).
fn fill_f32_quiet_nan(gpu: &mut Gpu, tensor: &GpuTensor, payload_bits: u32) {
    let n = tensor.numel();
    let host: Vec<u32> = vec![payload_bits; n];
    gpu.hip
        .memcpy_htod(&tensor.buf, unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
        })
        .expect("htod fill quiet NaN");
    gpu.hip
        .device_synchronize()
        .expect("sync after fill quiet NaN");
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
    let is_gfx1100 = gpu.arch_caps.is_gfx1100() && arch == "gfx1100";
    if !is_gfx1100 {
        eprintln!("SKIP: arch {arch} is not exact gfx1100 — harness requires gfx1100 only");
        return;
    }
    eprintln!("arch {arch} confirmed exact gfx1100 — running qkvza BT parity");

    if gpu.active_capture.is_some() {
        eprintln!("SKIP: active_capture is Some — BT harness requires no capture");
        return;
    }

    // Routing-sensitive shapes: qkv=40/z=28/beta=36/alpha=16 is not 16-aligned per projection;
    // total 120 has an 8-row tail tile so tail guards are exercised. Distinct seeds
    // per projection ensure a swapped routing is caught by bit mismatch.
    let qkv_m: usize = 40;
    let z_m: usize = 28;
    let beta_m: usize = 36;
    let alpha_m: usize = 16;
    let k: usize = 512;
    assert_eq!(k % GROUP, 0);
    let total_m = qkv_m + z_m + beta_m + alpha_m;
    eprintln!("qkv_m={qkv_m} z_m={z_m} beta_m={beta_m} alpha_m={alpha_m} total={total_m} K={k}");

    let w_qkv = build_disjoint_halves_seeded(qkv_m, k, 0x1111_2222);
    let w_z = build_disjoint_halves_seeded(z_m, k, 0x3333_4444);
    let w_beta = build_disjoint_halves_seeded(beta_m, k, 0x5555_6666);
    let w_alpha = build_disjoint_halves_seeded(alpha_m, k, 0x7777_8888);
    let blob_qkv = pack_mq4g256v2(&w_qkv, qkv_m, k);
    let blob_z = pack_mq4g256v2(&w_z, z_m, k);
    let blob_beta = pack_mq4g256v2(&w_beta, beta_m, k);
    let blob_alpha = pack_mq4g256v2(&w_alpha, alpha_m, k);
    assert_eq!(blob_qkv.len(), qkv_m * (k / GROUP) * GROUP_BYTES);
    assert_eq!(blob_z.len(), z_m * (k / GROUP) * GROUP_BYTES);
    assert_eq!(blob_beta.len(), beta_m * (k / GROUP) * GROUP_BYTES);
    assert_eq!(blob_alpha.len(), alpha_m * (k / GROUP) * GROUP_BYTES);
    // Distinct packed bytes: swapped routing must not bit-match.
    assert_ne!(
        blob_qkv, blob_z,
        "qkv/z packed blobs must differ (distinct salts)"
    );
    assert_ne!(blob_beta, blob_alpha, "beta/alpha packed blobs must differ");
    eprintln!(
        "packed qkv {} B, z {} B, beta {} B, alpha {} B",
        blob_qkv.len(),
        blob_z.len(),
        blob_beta.len(),
        blob_alpha.len()
    );

    let d_qkv = gpu
        .upload_raw(&blob_qkv, &[blob_qkv.len()])
        .expect("upload qkv");
    let d_z = gpu.upload_raw(&blob_z, &[blob_z.len()]).expect("upload z");
    let d_beta = gpu
        .upload_raw(&blob_beta, &[blob_beta.len()])
        .expect("upload beta");
    let d_alpha = gpu
        .upload_raw(&blob_alpha, &[blob_alpha.len()])
        .expect("upload alpha");

    let ns_bt = [128usize, 256usize];
    let bts = [4usize, 12];

    let mut all_ok = true;

    for &n in &ns_bt {
        eprintln!("\n=== BT N={n} ===");
        let x_host: Vec<f32> = (0..n * k)
            .map(|i| prng(i, 0xC0FF_EE00) * 2.0 - 1.0)
            .collect();
        assert_eq!(x_host.len(), n * k);

        let d_x = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x");
        let d_y_qkv_ref = gpu
            .alloc_tensor(&[n * qkv_m], DType::F32)
            .expect("alloc y_qkv ref");
        let d_y_z_ref = gpu
            .alloc_tensor(&[n * z_m], DType::F32)
            .expect("alloc y_z ref");
        let d_y_beta_ref = gpu
            .alloc_tensor(&[n * beta_m], DType::F32)
            .expect("alloc y_beta ref");
        let d_y_alpha_ref = gpu
            .alloc_tensor(&[n * alpha_m], DType::F32)
            .expect("alloc y_alpha ref");

        gpu.hip
            .memcpy_htod(&d_x.buf, unsafe {
                std::slice::from_raw_parts(x_host.as_ptr() as *const u8, x_host.len() * 4)
            })
            .expect("htod x");
        gpu.hip.device_synchronize().expect("sync after htod");

        // Reference: historical base kernel. Production BT auto-selects outside capture,
        // so arm capture_mode for this launch only to retain the base object as oracle.
        // Initialize with distinct quiet-NaN per projection so unwritten tails / accidental += remain non-finite.
        fill_f32_quiet_nan(&mut gpu, &d_y_qkv_ref, NAN_QKV_BITS);
        fill_f32_quiet_nan(&mut gpu, &d_y_z_ref, NAN_Z_BITS);
        fill_f32_quiet_nan(&mut gpu, &d_y_beta_ref, NAN_BETA_BITS);
        fill_f32_quiet_nan(&mut gpu, &d_y_alpha_ref, NAN_ALPHA_BITS);
        let saved_capture = gpu.graphs.capture_mode;
        gpu.graphs.capture_mode = true;
        gpu.hip.device_synchronize().unwrap();
        let t_ref = std::time::Instant::now();
        let ref_res = gpu.gemm_qkvza_mq4g256v2_wmma(
            &d_qkv,
            &d_z,
            &d_beta,
            &d_alpha,
            &d_x,
            &d_y_qkv_ref,
            &d_y_z_ref,
            &d_y_beta_ref,
            &d_y_alpha_ref,
            qkv_m,
            z_m,
            beta_m,
            alpha_m,
            k,
            n,
        );
        gpu.graphs.capture_mode = saved_capture;
        ref_res.expect("base gemm_qkvza_mq4g256v2_wmma failed");
        gpu.hip.device_synchronize().expect("sync ref");
        let ref_us = t_ref.elapsed().as_secs_f64() * 1e6;
        eprintln!("  base ref N={n} done in {ref_us:.1} us");

        let y_qkv_ref = gpu.download_f32(&d_y_qkv_ref).expect("download qkv ref");
        let y_z_ref = gpu.download_f32(&d_y_z_ref).expect("download z ref");
        let y_beta_ref = gpu.download_f32(&d_y_beta_ref).expect("download beta ref");
        let y_alpha_ref = gpu
            .download_f32(&d_y_alpha_ref)
            .expect("download alpha ref");
        assert_eq!(y_qkv_ref.len(), n * qkv_m);
        assert_eq!(y_z_ref.len(), n * z_m);
        assert_eq!(y_beta_ref.len(), n * beta_m);
        assert_eq!(y_alpha_ref.len(), n * alpha_m);

        assert!(is_finite(&y_qkv_ref), "ref qkv not finite N={n}");
        assert!(is_finite(&y_z_ref), "ref z not finite N={n}");
        assert!(is_finite(&y_beta_ref), "ref beta not finite N={n}");
        assert!(is_finite(&y_alpha_ref), "ref alpha not finite N={n}");
        assert!(variance(&y_qkv_ref) > 1e-12, "ref qkv degenerate N={n}");
        assert!(variance(&y_z_ref) > 1e-12, "ref z degenerate N={n}");
        assert!(variance(&y_beta_ref) > 1e-12, "ref beta degenerate N={n}");
        assert!(variance(&y_alpha_ref) > 1e-12, "ref alpha degenerate N={n}");

        for &bt in &bts {
            let d_y_qkv_bt = gpu
                .alloc_tensor(&[n * qkv_m], DType::F32)
                .expect("alloc y_qkv bt");
            let d_y_z_bt = gpu
                .alloc_tensor(&[n * z_m], DType::F32)
                .expect("alloc y_z bt");
            let d_y_beta_bt = gpu
                .alloc_tensor(&[n * beta_m], DType::F32)
                .expect("alloc y_beta bt");
            let d_y_alpha_bt = gpu
                .alloc_tensor(&[n * alpha_m], DType::F32)
                .expect("alloc y_alpha bt");

            fill_f32_quiet_nan(&mut gpu, &d_y_qkv_bt, NAN_QKV_BITS);
            fill_f32_quiet_nan(&mut gpu, &d_y_z_bt, NAN_Z_BITS);
            fill_f32_quiet_nan(&mut gpu, &d_y_beta_bt, NAN_BETA_BITS);
            fill_f32_quiet_nan(&mut gpu, &d_y_alpha_bt, NAN_ALPHA_BITS);

            let res = gpu.gemm_qkvza_mq4g256v2_wmma_gfx1100_bt(
                &d_qkv,
                &d_z,
                &d_beta,
                &d_alpha,
                &d_x,
                &d_y_qkv_bt,
                &d_y_z_bt,
                &d_y_beta_bt,
                &d_y_alpha_bt,
                qkv_m,
                z_m,
                beta_m,
                alpha_m,
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

            let y_qkv_bt = gpu.download_f32(&d_y_qkv_bt).expect("download qkv bt");
            let y_z_bt = gpu.download_f32(&d_y_z_bt).expect("download z bt");
            let y_beta_bt = gpu.download_f32(&d_y_beta_bt).expect("download beta bt");
            let y_alpha_bt = gpu.download_f32(&d_y_alpha_bt).expect("download alpha bt");

            let ok_qkv = check_proj("qkv  ", n, bt, &y_qkv_bt, &y_qkv_ref);
            let ok_z = check_proj("z    ", n, bt, &y_z_bt, &y_z_ref);
            let ok_beta = check_proj("beta ", n, bt, &y_beta_bt, &y_beta_ref);
            let ok_alpha = check_proj("alpha", n, bt, &y_alpha_bt, &y_alpha_ref);
            if !ok_qkv || !ok_z || !ok_beta || !ok_alpha {
                eprintln!("  BT{bt} N={n} FAILED");
                all_ok = false;
            } else {
                eprintln!("  BT{bt} N={n} OK");
            }

            if n == 256 {
                fill_f32_quiet_nan(&mut gpu, &d_y_qkv_bt, NAN_QKV_BITS);
                fill_f32_quiet_nan(&mut gpu, &d_y_z_bt, NAN_Z_BITS);
                fill_f32_quiet_nan(&mut gpu, &d_y_beta_bt, NAN_BETA_BITS);
                fill_f32_quiet_nan(&mut gpu, &d_y_alpha_bt, NAN_ALPHA_BITS);
                let t = std::time::Instant::now();
                gpu.gemm_qkvza_mq4g256v2_wmma_gfx1100_bt(
                    &d_qkv,
                    &d_z,
                    &d_beta,
                    &d_alpha,
                    &d_x,
                    &d_y_qkv_bt,
                    &d_y_z_bt,
                    &d_y_beta_bt,
                    &d_y_alpha_bt,
                    qkv_m,
                    z_m,
                    beta_m,
                    alpha_m,
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
        eprintln!("\nPASS: all N={{128,256}} x BT{{4,12}} x {{qkv,z,beta,alpha}} raw-bit equal (f32::to_bits) under distinct salts + quiet-NaN overwrite init (qkv=40/z=28/beta=36/alpha=16, total 120 tail 8) — finite+to_bits proves routing/full overwrite/no accidental +=; relL2<=1e-5 cosine>=1-1e-6; maxAbs reported per variant");
    } else {
        eprintln!(
            "\nFAIL: one or more parity checks violated raw-bit equality, overwrite (finite), routing, or numeric thresholds"
        );
        std::process::exit(1);
    }
}
