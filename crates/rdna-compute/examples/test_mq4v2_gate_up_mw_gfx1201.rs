// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! MQ4V2 gate+up MW_LDS parity on exact gfx1201 — multi-wave LDS candidate gate.
//!
//! Pack deterministic MQ4V2 gate/up weights with **distinct projection salts**
//! and disjoint halves. Row geometry stresses the fused stack (gate_m=40 /
//! up_m=53). Quiet-NaN sentinels prove full overwrite. On exact gfx1201:
//!   1) run current gfx12 production `gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2`
//!      as the arithmetic oracle (candidate env left unset),
//!   2) run direct `gemm_gate_up_mq4g256v2_wmma_gfx1201_mw_lds` MW4/8/12
//!      for N∈{191,192,383,384,511,512},
//!  and compare both projections for raw f32 bit equality plus
//!  finite/nondegenerate, relL2<=1e-5, cosine>=1-1e-6.
//! On any other arch the harness SKIPs cleanly (exit 0, no GPU work).

use rdna_compute::kv_slots::half_from_f32;
use rdna_compute::{DType, Gpu, GpuTensor};

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

const NAN_GATE_BITS: u32 = 0x7fc0_0001;
const NAN_UP_BITS: u32 = 0x7fc0_0002;
const SALT_GATE: u32 = 0x1111_2222;
const SALT_UP: u32 = 0x3333_4444;

fn build_disjoint_halves(m: usize, k: usize, proj_salt: u32) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let salt = ((r * 7919 + g * 104_729) as u32).wrapping_add(proj_salt);
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

fn fill_f32_quiet_nan(gpu: &mut Gpu, tensor: &GpuTensor, payload_bits: u32) {
    let v = f32::from_bits(payload_bits);
    assert!(v.is_nan(), "sentinel payload must be quiet NaN bits");
    gpu.fill_f32(tensor, v)
        .unwrap_or_else(|e| panic!("fill quiet-NaN sentinel 0x{payload_bits:08x}: {e:?}"));
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
    eprintln!("arch {arch} confirmed exact gfx1201 — running gate+up MW_LDS parity");

    if gpu.active_capture.is_some() {
        eprintln!("SKIP: active_capture is Some — MW harness requires no capture");
        return;
    }

    let gate_m: usize = 40;
    let up_m: usize = 53;
    let k: usize = 256; // not %512 so gfx12 ldsstage cannot steal the oracle
    assert_eq!(k % 256, 0);
    eprintln!(
        "geometry: gate_m={gate_m} up_m={up_m} total={} final_tile_rows={}",
        gate_m + up_m,
        (gate_m + up_m) % 16
    );

    let w_gate = build_disjoint_halves(gate_m, k, SALT_GATE);
    let w_up = build_disjoint_halves(up_m, k, SALT_UP);
    let blob_gate = pack_mq4g256v2(&w_gate, gate_m, k);
    let blob_up = pack_mq4g256v2(&w_up, up_m, k);
    assert_ne!(blob_gate, blob_up, "gate/up packed blobs must differ");
    eprintln!(
        "packed gate {} B, up {} B, K={k}",
        blob_gate.len(),
        blob_up.len()
    );

    let d_gate = gpu
        .upload_raw(&blob_gate, &[blob_gate.len()])
        .expect("upload gate");
    let d_up = gpu
        .upload_raw(&blob_up, &[blob_up.len()])
        .expect("upload up");

    // Below-threshold 191 exercises direct launcher tails; 192+ covers selector floor.
    let ns_mw = [191usize, 192, 383, 384, 511, 512];
    let mws = [4usize, 8, 12];
    let mut all_ok = true;

    for &n in &ns_mw {
        eprintln!("\n=== MW_LDS N={n} ===");
        let x_host: Vec<f32> = (0..n * k)
            .map(|i| prng(i, 0xC0FF_EE00) * 2.0 - 1.0)
            .collect();

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
        gpu.hip.device_synchronize().expect("sync after htod x");

        fill_f32_quiet_nan(&mut gpu, &d_y_gate_ref, NAN_GATE_BITS);
        fill_f32_quiet_nan(&mut gpu, &d_y_up_ref, NAN_UP_BITS);
        // Force gfx12 base (BT=1) as oracle so production BT selection cannot
        // alias the candidate under test.
        std::env::set_var("HIPFIRE_GATE_UP_BT", "0");
        std::env::remove_var("HIPFIRE_MQ4V2_GFX1201_GATE_UP_MW_LDS");
        gpu.hip.device_synchronize().unwrap();
        let t_ref = std::time::Instant::now();
        let ref_res = gpu.gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2(
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
        ref_res.expect("gfx12 base gate_up oracle failed");
        gpu.hip.device_synchronize().expect("sync ref");
        let ref_us = t_ref.elapsed().as_secs_f64() * 1e6;
        eprintln!("  gfx12 base ref N={n} done in {ref_us:.1} us");

        let y_gate_ref = gpu.download_f32(&d_y_gate_ref).expect("download gate ref");
        let y_up_ref = gpu.download_f32(&d_y_up_ref).expect("download up ref");
        assert_eq!(y_gate_ref.len(), n * gate_m);
        assert_eq!(y_up_ref.len(), n * up_m);
        assert!(is_finite(&y_gate_ref), "ref gate not finite N={n}");
        assert!(is_finite(&y_up_ref), "ref up not finite N={n}");
        assert!(variance(&y_gate_ref) > 1e-12, "ref gate degenerate N={n}");
        assert!(variance(&y_up_ref) > 1e-12, "ref up degenerate N={n}");

        for &waves in &mws {
            let d_y_gate_mw = gpu
                .alloc_tensor(&[n * gate_m], DType::F32)
                .expect("alloc y_gate mw");
            let d_y_up_mw = gpu
                .alloc_tensor(&[n * up_m], DType::F32)
                .expect("alloc y_up mw");
            fill_f32_quiet_nan(&mut gpu, &d_y_gate_mw, NAN_GATE_BITS);
            fill_f32_quiet_nan(&mut gpu, &d_y_up_mw, NAN_UP_BITS);
            let res = gpu.gemm_gate_up_mq4g256v2_wmma_gfx1201_mw_lds(
                &d_gate,
                &d_up,
                &d_x,
                &d_y_gate_mw,
                &d_y_up_mw,
                gate_m,
                up_m,
                k,
                n,
                waves,
            );
            if let Err(e) = res {
                eprintln!("  MW{waves} wrapper launch failed N={n}: {e:?}");
                all_ok = false;
                continue;
            }
            gpu.hip.device_synchronize().expect("sync mw");

            let y_gate_mw = gpu.download_f32(&d_y_gate_mw).expect("download gate mw");
            let y_up_mw = gpu.download_f32(&d_y_up_mw).expect("download up mw");
            let ok_gate = check_proj("gate-mw", n, waves, &y_gate_mw, &y_gate_ref);
            let ok_up = check_proj("up-mw  ", n, waves, &y_up_mw, &y_up_ref);
            if !ok_gate || !ok_up {
                eprintln!("  MW{waves} N={n} FAILED");
                all_ok = false;
            } else {
                eprintln!("  MW{waves} N={n} OK");
            }

            if n == 512 {
                fill_f32_quiet_nan(&mut gpu, &d_y_gate_mw, NAN_GATE_BITS);
                fill_f32_quiet_nan(&mut gpu, &d_y_up_mw, NAN_UP_BITS);
                let t = std::time::Instant::now();
                gpu.gemm_gate_up_mq4g256v2_wmma_gfx1201_mw_lds(
                    &d_gate,
                    &d_up,
                    &d_x,
                    &d_y_gate_mw,
                    &d_y_up_mw,
                    gate_m,
                    up_m,
                    k,
                    n,
                    waves,
                )
                .expect("re-launch mw for timing");
                gpu.hip.device_synchronize().unwrap();
                let us = t.elapsed().as_secs_f64() * 1e6;
                eprintln!("    timing N512 MW{waves}: {us:.1} us (host Instant+sync)");
            }
        }
    }

    if all_ok {
        eprintln!(
            "\nPASS: all N={{191,192,383,384,511,512}} x MW{{4,8,12}} x {{gate,up}} raw-bit equal vs gfx12 base; finite/nondegenerate; relL2<=1e-5 cosine>=1-1e-6"
        );
    } else {
        eprintln!(
            "\nFAIL: one or more parity checks violated raw-bit equality, overwrite, routing, or numeric thresholds"
        );
        std::process::exit(1);
    }
}
