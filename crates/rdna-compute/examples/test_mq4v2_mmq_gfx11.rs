// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! MQ4V2 Q8_1 MMQ set/add numerical oracle on exact gfx1100 / gfx1151.
//!
//! Synthetic pack: dual-fp16 disjoint half headers (half0 ∈ [-1,1], half1 ∈
//! [96,160]) + contiguous nibble payload so a wrong half-select is unmissable.
//! Deterministic F32 X. Shapes: K=512, M=32 (tail of the 128-row MMQ tile),
//! N ∈ {128, 512}. Four arms per arch: set/add × N128/N512.
//!
//! Base oracle: `gemm_mq4g256v2_residual_wmma` under `capture_mode` (forces the
//! historical F16 residual WMMA path; skips production BT/MW/MMQ routes).
//!   - set arms: zero Y
//!   - add arms: identical nonzero deterministic Y on base and candidate
//!
//! Candidate: `ensure_q8_1_mmq_x` then
//! `gemm_mq4g256v2_mmq_{set,add}_prequant`.
//!
//! Q8 path is approximate: rel-RMS ≤ 0.01, all finite, nondegenerate, canary
//! immediately after the Y region intact. No raw-bit requirement.
//! Other arches SKIP cleanly (exit 0).

use rdna_compute::kv_slots::half_from_f32;
use rdna_compute::Gpu;

const GROUP: usize = 256;
const HALF: usize = 128;
const GROUP_BYTES: usize = 136;
const K: usize = 512;
const M: usize = 128; // aligned MMQ_Y tile exercises _full_set/_full_add
const REL_RMS_LIMIT: f64 = 0.002;
const CANARY: f32 = 7.654_321;

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

/// MQ4V2 wire: 136 B/G256 = [fp16 s0,z0,s1,z1] + 128 B contiguous nibbles.
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
            // Contiguous nibble packing: even K → lo nibble, odd K → hi nibble.
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
    v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / v.len() as f64
}

/// rel-RMS = sqrt(sum (a-b)^2 / sum b^2)  (== RMS_err / RMS_ref).
fn rel_rms(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in got.iter().zip(want.iter()) {
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn check_arm(label: &str, n: usize, got: &[f32], want: &[f32], canary_ok: bool) -> bool {
    let finite = is_finite(got) && is_finite(want);
    let var_got = variance(got);
    let var_want = variance(want);
    let nondeg = var_got > 1e-12 && var_want > 1e-12;
    let r = rel_rms(got, want);
    let m = max_abs_diff(got, want);
    let max_finite = m.is_finite();
    let ok = canary_ok && finite && nondeg && max_finite && r <= REL_RMS_LIMIT;
    let status = if ok { "PASS" } else { "FAIL" };
    eprintln!(
        "  [{label} N={n} M={M} K={K}] canary={canary_ok} finite={finite} nondeg={nondeg} \
         (var {var_got:.3e}/{var_want:.3e}) relRMS={r:.6} maxAbs={m:.3e} maxFinite={max_finite} [{status}]"
    );
    if !ok {
        if !canary_ok {
            eprintln!("    GATE FAIL: canary immediately after Y corrupted");
        }
        if !finite {
            eprintln!("    non-finite values detected");
        }
        if !nondeg {
            eprintln!("    degenerate (near-zero variance) — possible zeroed/stub output");
        }
        if !max_finite {
            eprintln!("    max abs error is non-finite");
        }
        if r > REL_RMS_LIMIT {
            eprintln!("    rel-RMS violated: {r:.6} > {REL_RMS_LIMIT}");
        }
    }
    ok
}

/// Build host Y of length `n*m + 1` with payload then canary.
fn y_host_with_canary(n: usize, m: usize, init: impl Fn(usize) -> f32) -> Vec<f32> {
    let y_len = n * m;
    let mut y = vec![0.0f32; y_len + 1];
    for i in 0..y_len {
        y[i] = init(i);
    }
    y[y_len] = CANARY;
    y
}

fn run_arm(gpu: &mut Gpu, d_a: &rdna_compute::GpuTensor, n: usize, add: bool) -> bool {
    let label = if add { "mmq-add" } else { "mmq-set" };
    eprintln!("\n=== {label} N={n} ===");

    let x_host: Vec<f32> = (0..n * K)
        .map(|i| prng(i, 0xC0FF_EE00) * 2.0 - 1.0)
        .collect();
    let d_x = gpu.upload_f32(&x_host, &[n * K]).expect("upload x");
    gpu.hip.device_synchronize().expect("sync after htod x");

    // Y init: zero for set; identical nonzero for add (base + candidate).
    let y_init = y_host_with_canary(n, M, |i| {
        if add {
            prng(i, 0xBEEF_1234) * 2.0 - 1.0 + 0.5
        } else {
            0.0
        }
    });
    let y_len = n * M;

    // ── Base: F16 residual WMMA under capture_mode (skips BT/MW/MMQ) ──────
    let d_y_ref = gpu.upload_f32(&y_init, &[y_len + 1]).expect("upload y_ref");
    gpu.hip.device_synchronize().expect("sync y_ref");

    let saved_capture = gpu.graphs.capture_mode;
    gpu.graphs.capture_mode = true;
    gpu.hip.device_synchronize().unwrap();
    let t_ref = std::time::Instant::now();
    let ref_res = gpu.gemm_mq4g256v2_residual_wmma(d_a, &d_x, &d_y_ref, M, K, n);
    gpu.graphs.capture_mode = saved_capture;
    if let Err(e) = ref_res {
        eprintln!("  base residual_wmma failed N={n}: {e:?}");
        let _ = gpu.free_tensor(d_x);
        let _ = gpu.free_tensor(d_y_ref);
        return false;
    }
    gpu.hip.device_synchronize().expect("sync ref");
    let ref_us = t_ref.elapsed().as_secs_f64() * 1e6;
    eprintln!("  base F16 residual_wmma N={n} done in {ref_us:.1} us");

    let y_ref_full = gpu.download_f32(&d_y_ref).expect("download ref");
    let canary_ref_ok =
        y_ref_full.len() == y_len + 1 && y_ref_full[y_len].to_bits() == CANARY.to_bits();
    if !canary_ref_ok {
        eprintln!(
            "  WARN: base canary unexpected (len={} val={:?})",
            y_ref_full.len(),
            y_ref_full.get(y_len)
        );
    }
    let y_ref = &y_ref_full[..y_len];
    if !is_finite(y_ref) || variance(y_ref) <= 1e-12 {
        eprintln!(
            "  base ref invalid N={n}: finite={} var={:.3e}",
            is_finite(y_ref),
            variance(y_ref)
        );
        let _ = gpu.free_tensor(d_x);
        let _ = gpu.free_tensor(d_y_ref);
        return false;
    }

    // ── Candidate: Q8_1 prequant + MMQ set/add ────────────────────────────
    let d_y_mmq = gpu.upload_f32(&y_init, &[y_len + 1]).expect("upload y_mmq");
    gpu.hip.device_synchronize().expect("sync y_mmq");

    // Candidate must not run under capture_mode (base already restored).
    debug_assert!(!gpu.graphs.capture_mode);

    let t_mmq = std::time::Instant::now();
    let xq = match gpu.ensure_q8_1_mmq_x(&d_x, n, K) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  ensure_q8_1_mmq_x failed N={n}: {e:?}");
            let _ = gpu.free_tensor(d_x);
            let _ = gpu.free_tensor(d_y_ref);
            let _ = gpu.free_tensor(d_y_mmq);
            return false;
        }
    };
    gpu.hip.device_synchronize().expect("sync after q8 quant");

    let mmq_res = if add {
        gpu.gemm_mq4g256v2_mmq_add_prequant(d_a, xq, &d_y_mmq, M, K, n)
    } else {
        gpu.gemm_mq4g256v2_mmq_set_prequant(d_a, xq, &d_y_mmq, M, K, n)
    };
    if let Err(e) = mmq_res {
        eprintln!("  {label} prequant launch failed N={n}: {e:?}");
        let _ = gpu.free_tensor(d_x);
        let _ = gpu.free_tensor(d_y_ref);
        let _ = gpu.free_tensor(d_y_mmq);
        return false;
    }
    gpu.hip.device_synchronize().expect("sync mmq");
    let mmq_us = t_mmq.elapsed().as_secs_f64() * 1e6;
    eprintln!("  candidate {label} N={n} done in {mmq_us:.1} us (quant+gemm)");

    let y_mmq_full = gpu.download_f32(&d_y_mmq).expect("download mmq");
    let canary_ok =
        y_mmq_full.len() == y_len + 1 && y_mmq_full[y_len].to_bits() == CANARY.to_bits();
    let y_mmq = &y_mmq_full[..y_len.min(y_mmq_full.len())];

    // Stub signature: set/add wrote nothing → still equals init payload.
    if y_mmq.len() == y_len
        && y_mmq
            .iter()
            .zip(y_init[..y_len].iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    {
        eprintln!("  STUB SIGNATURE: {label} N={n} output == input Y (kernel wrote nothing)");
        let _ = gpu.free_tensor(d_x);
        let _ = gpu.free_tensor(d_y_ref);
        let _ = gpu.free_tensor(d_y_mmq);
        return false;
    }

    let ok = check_arm(label, n, y_mmq, y_ref, canary_ok);

    let _ = gpu.free_tensor(d_x);
    let _ = gpu.free_tensor(d_y_ref);
    let _ = gpu.free_tensor(d_y_mmq);
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
    let exact = matches!(arch.as_str(), "gfx1100" | "gfx1151");
    if !exact {
        eprintln!(
            "SKIP: arch {arch} is not exact gfx1100/gfx1151 — MQ4V2 Q8_1 MMQ harness requires those only"
        );
        return;
    }
    eprintln!(
        "arch {arch} confirmed — running MQ4V2 MMQ set/add parity (M={M} K={K} N={{128,512}})"
    );

    if gpu.active_capture.is_some() {
        eprintln!("SKIP: active_capture is Some — harness requires no capture");
        return;
    }

    assert_eq!(K % 256, 0);
    assert_eq!(M % 128, 0, "M must cover an aligned 128-row MMQ tile");

    let w = build_disjoint_halves(M, K);
    let blob = pack_mq4g256v2(&w, M, K);
    assert_eq!(blob.len(), M * (K / GROUP) * GROUP_BYTES);
    eprintln!(
        "packed MQ4V2 residual {} B (disjoint dual-fp16 headers, contiguous nibbles)",
        blob.len()
    );

    let d_a = gpu.upload_raw(&blob, &[blob.len()]).expect("upload A");

    let mut all_ok = true;
    for &n in &[128usize, 512usize] {
        for &add in &[false, true] {
            if !run_arm(&mut gpu, &d_a, n, add) {
                all_ok = false;
            }
        }
    }

    let _ = gpu.free_tensor(d_a);

    if all_ok {
        eprintln!(
            "\nPASS: all 4 arms (set/add × N{{128,512}}) on {arch}: rel-RMS<={REL_RMS_LIMIT}, \
             finite/nondegenerate, maxAbs finite, canary intact; base=F16 residual_wmma(capture_mode), \
             candidate=ensure_q8_1_mmq_x+mmq_{{set,add}}_prequant; M={M} K={K}"
        );
    } else {
        eprintln!("\nFAIL: one or more MMQ set/add arms violated rel-RMS/canary/finite gates");
        std::process::exit(1);
    }
}
