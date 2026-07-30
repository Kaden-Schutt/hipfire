// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Correctness gate for attention_q8_0_flash_prefill vs attention_q8_0_kv_batched.
// Env: NH, NKV, HD, N (query rows), CTX (max_ctx_len), BR, BC, POS.

use rdna_compute::{DType, Gpu};

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let nh = env_usize("NH", 8);
    let nkv = env_usize("NKV", 2);
    let hd = env_usize("HD", 256);
    let n = env_usize("N", 16);
    let ctx = env_usize("CTX", 32);
    let br = env_usize("BR", 16);
    let bc = env_usize("BC", 32);
    let mut gpu = Gpu::init().expect("gpu init");

    let bph = hd / 32;
    let bytes_per_pos = nkv * bph * 34;
    let cache_bytes = ctx * bytes_per_pos;

    // Deterministic pseudo-random KV: varied scales and codes so a wrong
    // dequant, wrong block stride or wrong GQA head cannot pass by symmetry.
    let mut kv = vec![0u8; cache_bytes];
    // KVEXACT=1 forces every block scale to 1.0, so the dequantised value is
    // just the int8 code — exactly representable in f16. Any residual WMMA
    // error under this mode therefore cannot come from K/V rounding.
    let kv_exact = std::env::var("KVEXACT").as_deref() == Ok("1");
    for (bi, blk) in kv.chunks_mut(34).enumerate() {
        // Powers of two in the SAME magnitude band as the normal scales
        // (0.031/0.016/0.008 vs 0.02-0.08), so only representability changes:
        // a power-of-two scale times an 8-bit code is exact in f16. Using
        // scale=1.0 instead would blow the score magnitudes up and collapse
        // the softmax to near one-hot — a different regime, not a controlled
        // precision isolation.
        let scale: f32 = if kv_exact {
            [0.03125f32, 0.015625, 0.0078125][bi % 3]
        } else {
            0.02 + ((bi % 13) as f32) * 0.005
        };
        let h = half_from_f32(scale);
        blk[0] = (h & 0xFF) as u8;
        blk[1] = (h >> 8) as u8;
        for (j, b) in blk[2..].iter_mut().enumerate() {
            *b = (((bi * 31 + j * 17) % 251) as i32 - 125) as i8 as u8;
        }
    }
    let k_cache = gpu.upload_raw(&kv, &[cache_bytes]).expect("k upload");
    let mut kv2 = kv.clone();
    for (i, b) in kv2.iter_mut().enumerate() {
        if i % 34 >= 2 {
            *b = (*b).wrapping_add(7);
        }
    }
    let v_cache = gpu.upload_raw(&kv2, &[cache_bytes]).expect("v upload");

    let q_data: Vec<f32> = (0..n * nh * hd)
        .map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.01)
        .collect();
    // QF16=1 isolates Q's rounding contribution: the REFERENCE keeps full-f32
    // Q while the CANDIDATE gets Q pre-rounded through f16. Pair with
    // KERNEL=scalar (otherwise an f32 path) so Q's precision is the ONLY
    // difference between the two arms.
    //
    // An earlier version rounded q_data before a SINGLE shared upload, so both
    // arms saw the same rounded Q and the comparison measured nothing — it
    // reported ~1e-6 and led me to wrongly clear Q as a contributor.
    let qf16 = std::env::var("QF16").as_deref() == Ok("1");
    let q = gpu.upload_f32(&q_data, &[n * nh * hd]).expect("q upload");
    let q_cand = if qf16 {
        let rounded: Vec<f32> = q_data.iter().map(|&v| f32::from_bits(round_f16(v))).collect();
        gpu.upload_f32(&rounded, &[n * nh * hd]).expect("q_cand upload")
    } else {
        gpu.upload_f32(&q_data, &[n * nh * hd]).expect("q_cand upload")
    };

    // POS=tail  : positions[b] = ctx - n + b   (contiguous tail chunk)
    // POS=ragged: every row gets a different, non-monotonic causal window,
    //             which exercises per-row masking and the per-tile seq_len max.
    let pos_mode = std::env::var("POS").unwrap_or_else(|_| "tail".into());
    let pos_data: Vec<i32> = match pos_mode.as_str() {
        "ragged" => (0..n)
            .map(|b| {
                let span = ctx.max(2) - 1;
                (((b * 7919) % span) + 1) as i32
            })
            .collect(),
        _ => (0..n).map(|b| (ctx - n + b) as i32).collect(),
    };
    let pos_bytes = unsafe { std::slice::from_raw_parts(pos_data.as_ptr() as *const u8, n * 4) };
    let positions = gpu.upload_raw(pos_bytes, &[n]).expect("pos upload");

    let out_ref = gpu.zeros(&[n * nh * hd], DType::F32).expect("out_ref");
    let out_new = gpu.zeros(&[n * nh * hd], DType::F32).expect("out_new");

    gpu.attention_q8_0_kv_batched_masked(
        &q, &k_cache, &v_cache, &out_ref, &positions, nh, nkv, hd, ctx, ctx, n, None, 0, 0,
    )
    .expect("reference kernel");

    // KERNEL=scalar (default) | wmma
    let kernel = std::env::var("KERNEL").unwrap_or_else(|_| "scalar".into());
    match kernel.as_str() {
        "wmma" => gpu
            .attention_q8_0_flash_prefill_wmma(
                &q_cand, &k_cache, &v_cache, &out_new, &positions, nh, nkv, hd, n,
            )
            .expect("wmma flash prefill kernel"),
        _ => gpu
            .attention_q8_0_flash_prefill(
                &q_cand, &k_cache, &v_cache, &out_new, &positions, nh, nkv, hd, ctx, n, br, bc,
            )
            .expect("flash prefill kernel"),
    }

    let a = gpu.download_f32(&out_ref).expect("dl ref");
    let b = gpu.download_f32(&out_new).expect("dl new");
    assert_eq!(a.len(), b.len());

    // Combined tolerance (numpy allclose form): |a-b| <= ATOL + RTOL*|a|.
    // A hard split on |a| would be discontinuous — the same absolute error
    // passes or fails depending on which side of the split |a| lands.
    const ATOL: f32 = 1e-5;
    const RTOL: f32 = 1e-4;
    let (mut max_abs_all, mut worst_ratio, mut worst_at) = (0.0f32, 0.0f32, 0usize);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let abs = (x - y).abs();
        max_abs_all = max_abs_all.max(abs);
        let budget = ATOL + RTOL * x.abs();
        let ratio = abs / budget;
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst_at = i;
        }
    }
    // Cosine similarity and relative L2 error per (query, head) output vector.
    // Relative L2 is the meaningful accuracy metric for a reduced-precision
    // kernel: per-element relative error explodes on outputs near zero, where
    // cancellation amplifies input rounding, even when the vector is correct.
    let mut min_cos = 1.0f32;
    let mut max_rel_l2 = 0.0f32;
    let mut compared = 0usize;
    let mut degenerate = 0usize;
    for vec_i in 0..(n * nh) {
        let s = vec_i * hd;
        let (mut dot, mut na, mut nb, mut nd) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for d in 0..hd {
            dot += (a[s + d] as f64) * (b[s + d] as f64);
            na += (a[s + d] as f64).powi(2);
            nb += (b[s + d] as f64).powi(2);
            nd += ((a[s + d] - b[s + d]) as f64).powi(2);
        }
        if na > 0.0 && nb > 0.0 {
            min_cos = min_cos.min((dot / (na.sqrt() * nb.sqrt())) as f32);
            max_rel_l2 = max_rel_l2.max((nd.sqrt() / na.sqrt()) as f32);
            compared += 1;
        } else if na > 0.0 {
            // Reference has signal but the candidate vector is all-zero.
            // Without this branch such a vector is silently SKIPPED, leaving
            // min_cos at 1.0 and rel_l2 at 0.0 — so an all-zero kernel output
            // passes VACUOUSLY. Observed for real: a broken SPLIT_Q variant
            // emitted all zeros and this suite reported PASS.
            degenerate += 1;
        }
    }
    assert!(
        b.iter().all(|v| v.is_finite()),
        "candidate output contains non-finite values"
    );
    assert!(
        degenerate == 0,
        "{degenerate} candidate vectors are all-zero while the reference is not"
    );
    assert!(
        compared == n * nh,
        "only {compared} of {} vectors were comparable",
        n * nh
    );
    println!("kernel={kernel} nh={nh} nkv={nkv} hd={hd} n={n} ctx={ctx} br={br} bc={bc} pos={pos_mode}");
    println!(
        "max_abs={max_abs_all:.3e} worst_tol_ratio={worst_ratio:.3} \
         (at {worst_at}: ref={:.6e} new={:.6e}) min_cos={min_cos:.9} rel_l2={max_rel_l2:.3e}",
        a[worst_at], b[worst_at]
    );
    // The WMMA kernel computes in f16 (~5e-4 relative input precision), so it
    // is held to a reduced-precision bar rather than the fp32-reassociation
    // one. Both bars are strict for their arithmetic: the scalar kernel uses
    // 6.3% of its budget, and f16 attention of this depth cannot do better
    // than ~1e-3 relative L2.
    if kernel == "wmma" {
        assert!(
            max_rel_l2 <= 5e-3,
            "wmma relative L2 {max_rel_l2:.3e} > 5e-3 — too large for f16 rounding"
        );
        assert!(min_cos >= 1.0 - 1e-5, "wmma min cosine {min_cos:.9} < 1-1e-5");
    } else {
        assert!(
            worst_ratio <= 1.0,
            "element {worst_at} exceeds ATOL+RTOL*|ref|: ref={:.6e} new={:.6e} \
             abs={:.3e} budget={:.3e}",
            a[worst_at],
            b[worst_at],
            (a[worst_at] - b[worst_at]).abs(),
            ATOL + RTOL * a[worst_at].abs()
        );
        assert!(min_cos >= 1.0 - 1e-6, "min cosine {min_cos:.9} < 1-1e-6");
    }
    println!("PASS");
}

/// Minimal f32 -> IEEE binary16 bit pattern (round-toward-zero mantissa).
/// Only needs to cover the small positive scales used above.
fn half_from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits & 0x007F_FFFF) >> 13;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    sign | ((exp as u16) << 10) | (mant as u16)
}

/// Round an f32 through IEEE binary16 and back, returning the f32 bit pattern.
/// Round-to-nearest-even on the mantissa; the magnitudes here never reach the
/// f16 exponent limits.
fn round_f16(x: f32) -> u32 {
    let b = x.to_bits();
    let sign = b & 0x8000_0000;
    let exp = ((b >> 23) & 0xFF) as i32;
    let mant = b & 0x007F_FFFF;
    if exp == 0 || exp == 0xFF {
        return b;
    }
    // f16 keeps 10 mantissa bits; round-to-nearest-even on bit 13.
    let keep = mant & !0x1FFF;
    let rem = mant & 0x1FFF;
    let mut out_mant = keep;
    let mut out_exp = exp;
    if rem > 0x1000 || (rem == 0x1000 && (keep & 0x2000) != 0) {
        out_mant = keep + 0x2000;
        if out_mant > 0x007F_FFFF {
            out_mant = 0;
            out_exp += 1;
        }
    }
    sign | ((out_exp as u32) << 23) | out_mant
}
