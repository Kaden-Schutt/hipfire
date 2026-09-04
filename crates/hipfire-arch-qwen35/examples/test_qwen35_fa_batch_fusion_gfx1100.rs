// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S6-fa-prep-q8-pair parity gate (gfx1100 only):
//! `qwen35_fa_prep_batched_gfx1100` vs deinterleave + Q/K rmsnorm + halfsplit
//! RoPE, and `kv_cache_write_q8_0_pair_batched_gfx1100` vs the two Q8 batched
//! writes. Requires q/gate/k F32 bit-equality and K/V cache byte equality
//! for N in {1, 2, 8, 16}, with noncontiguous positions, a nonzero RoPE
//! pos_offset (compaction phase), high cache slots, and canary bytes around
//! every written slot.
//!
//! Run: `cargo run --release -p hipfire-arch-qwen35
//!         --example test_qwen35_fa_batch_fusion_gfx1100`
//! (hipfire-arch-qwen35 enables `deltanet` by default; needs a gfx1100 GPU.)

use rdna_compute::Gpu;

const NQ: usize = 16;
const NK: usize = 2;
const HD: usize = 256;
const NROT: usize = 64;
const Q_DIM: usize = NQ * HD;
const KV_DIM: usize = NK * HD;
const EPS: f32 = 1e-6;
const THETA: f32 = 1_000_000.0;
const POS_OFFSET: i32 = 3;
const CAP: usize = 48;

/// Deterministic LCG in [-2, 2); seed-addressed so every buffer is stable.
fn fill_lcg(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f64) / (65536.0 * 32768.0);
            (u as f32 - 1.0) * 2.0
        })
        .collect()
}

fn upload_pos(gpu: &mut Gpu, vals: &[i32]) -> rdna_compute::GpuTensor {
    let t = gpu
        .alloc_tensor(&[vals.len()], rdna_compute::DType::F32)
        .expect("alloc positions");
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4)
    };
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("upload positions");
    t
}

fn assert_bits_eq(tag: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{tag}: length {} vs {}", a.len(), b.len());
    let mut bad = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            if bad < 8 {
                eprintln!("{tag}[{i}]: old={x:e} new={y:e}");
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{tag}: {bad} mismatched words");
}

fn test_prep(gpu: &mut Gpu, n: usize) {
    // Noncontiguous physical slots; tree-depth-like gaps included.
    let base: Vec<i32> = (0..n)
        .map(|b| (5 + b * 3 + (b % 3) * 7) as i32)
        .collect();
    let pos = upload_pos(gpu, &base);

    let inter = gpu
        .upload_f32(&fill_lcg(n * Q_DIM * 2, 0x11 + n as u64), &[n * Q_DIM * 2])
        .expect("upload inter");
    let k_in = fill_lcg(n * KV_DIM, 0x22 + n as u64);
    let qw = fill_lcg(HD, 0x33).iter().map(|&v| 0.5 + 0.02 * v).collect::<Vec<_>>();
    let kw = fill_lcg(HD, 0x44).iter().map(|&v| 0.5 + 0.02 * v).collect::<Vec<_>>();
    let qw_t = gpu.upload_f32(&qw, &[HD]).expect("upload qw");
    let kw_t = gpu.upload_f32(&kw, &[HD]).expect("upload kw");

    // Old path buffers.
    let q_old = gpu.upload_f32(&vec![0.0; n * Q_DIM], &[n * Q_DIM]).expect("q_old");
    let g_old = gpu.upload_f32(&vec![0.0; n * Q_DIM], &[n * Q_DIM]).expect("g_old");
    let k_old = gpu.upload_f32(&k_in, &[n * KV_DIM]).expect("k_old");
    gpu.deinterleave_f32_batched(&inter, &q_old, &g_old, NQ, HD, n)
        .expect("deinterleave");
    gpu.rmsnorm_batched(&q_old, &qw_t, &q_old, n * NQ, HD, EPS)
        .expect("q norm");
    gpu.rmsnorm_batched(&k_old, &kw_t, &k_old, n * NK, HD, EPS)
        .expect("k norm");
    gpu.rope_partial_interleaved_f32_batched(
        &q_old, &k_old, &pos, NQ, NK, HD, NROT, THETA, n, POS_OFFSET,
    )
    .expect("rope");

    // Fused path buffers.
    let q_new = gpu.upload_f32(&vec![0.0; n * Q_DIM], &[n * Q_DIM]).expect("q_new");
    let g_new = gpu.upload_f32(&vec![0.0; n * Q_DIM], &[n * Q_DIM]).expect("g_new");
    let k_new = gpu.upload_f32(&k_in, &[n * KV_DIM]).expect("k_new");
    gpu.qwen35_fa_prep_batched_gfx1100(
        &inter, &q_new, &g_new, &k_new, &qw_t, &kw_t, &pos, EPS, THETA, POS_OFFSET, n,
    )
    .expect("fused prep");

    let qo = gpu.download_f32(&q_old).expect("dl qo");
    let qn = gpu.download_f32(&q_new).expect("dl qn");
    let go = gpu.download_f32(&g_old).expect("dl go");
    let gn = gpu.download_f32(&g_new).expect("dl gn");
    let ko = gpu.download_f32(&k_old).expect("dl ko");
    let kn = gpu.download_f32(&k_new).expect("dl kn");
    assert_bits_eq(&format!("prep q N={n}"), &qo, &qn);
    assert_bits_eq(&format!("prep gate N={n}"), &go, &gn);
    assert_bits_eq(&format!("prep k N={n}"), &ko, &kn);
    // Non-triviality: norm+rope must actually change values (else both arms
    // could be no-ops and still agree).
    let inter_host = gpu.download_f32(&inter).expect("dl inter");
    assert!(
        qo.iter().zip(inter_host.iter()).any(|(&a, &b)| a.to_bits() != b.to_bits()),
        "prep N={n}: fused output looks untouched"
    );
    println!("  prep N={n}: q/gate/k bit-equal ({})", qo.len());

    for t in [inter, q_old, g_old, k_old, q_new, g_new, k_new, qw_t, kw_t, pos] {
        gpu.free_tensor(t).expect("free");
    }
}

fn test_kv_pair(gpu: &mut Gpu, n: usize) {
    // Unique noncontiguous slots spanning the arena (11 is coprime to 48).
    // Uniqueness is required: two rows sharing a slot race in the OLD kernel
    // too (concurrent blocks, one launch), so duplicates can never be
    // byte-compared across runs. Production batches always use distinct slots.
    let slots: Vec<i32> = (0..n).map(|b| ((b * 11 + 5) % CAP) as i32).collect();
    assert!(slots.iter().all(|&p| p >= 0 && (p as usize) < CAP));
    let pos = upload_pos(gpu, &slots);

    let per_pos_bytes = NK * (HD / 32) * 34;
    assert_eq!(per_pos_bytes % 4, 0);
    let words = CAP * per_pos_bytes / 4;
    let canary: Vec<f32> = (0..words)
        .map(|i| f32::from_bits(0xAB000000u32.wrapping_add(i as u32 * 2654435761)))
        .collect();

    let k_src = gpu
        .upload_f32(&fill_lcg(n * KV_DIM, 0x55 + n as u64), &[n * KV_DIM])
        .expect("k_src");
    let v_src = gpu
        .upload_f32(&fill_lcg(n * KV_DIM, 0x66 + n as u64), &[n * KV_DIM])
        .expect("v_src");

    let k_old = gpu.upload_f32(&canary, &[words]).expect("k_old");
    let v_old = gpu.upload_f32(&canary, &[words]).expect("v_old");
    gpu.kv_cache_write_q8_0_batched(&k_old, &k_src, &pos, NK, HD, n)
        .expect("k write");
    gpu.kv_cache_write_q8_0_batched(&v_old, &v_src, &pos, NK, HD, n)
        .expect("v write");

    let k_new = gpu.upload_f32(&canary, &[words]).expect("k_new");
    let v_new = gpu.upload_f32(&canary, &[words]).expect("v_new");
    gpu.kv_cache_write_q8_0_pair_batched(&k_new, &v_new, &k_src, &v_src, &pos, NK, HD, n)
        .expect("pair write");

    let ko = gpu.download_f32(&k_old).expect("dl ko");
    let kn = gpu.download_f32(&k_new).expect("dl kn");
    let vo = gpu.download_f32(&v_old).expect("dl vo");
    let vn = gpu.download_f32(&v_new).expect("dl vn");
    assert_bits_eq(&format!("kv K N={n}"), &ko, &kn);
    assert_bits_eq(&format!("kv V N={n}"), &vo, &vn);
    // Canary preservation: every unwritten word still holds the pattern, and
    // the written slots actually changed (else the test is vacuous).
    let written: std::collections::HashSet<usize> = slots.iter().map(|&p| p as usize).collect();
    let mut touched = 0usize;
    for slot in 0..CAP {
        let w0 = slot * per_pos_bytes / 4;
        let w1 = w0 + per_pos_bytes / 4;
        if written.contains(&slot) {
            if kn[w0..w1].iter().zip(&canary[w0..w1]).any(|(&a, &b)| a.to_bits() != b.to_bits()) {
                touched += 1;
            }
        } else {
            assert_bits_eq(&format!("kv K canary slot {slot} N={n}"), &kn[w0..w1], &canary[w0..w1]);
            assert_bits_eq(&format!("kv V canary slot {slot} N={n}"), &vn[w0..w1], &canary[w0..w1]);
        }
    }
    assert_eq!(touched, written.len(), "kv N={n}: some slot unwritten");
    println!("  kv-pair N={n}: K/V byte-equal, {touched} slots touched, canaries intact");

    for t in [pos, k_src, v_src, k_old, v_old, k_new, v_new] {
        gpu.free_tensor(t).expect("free");
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    if !gpu.arch_caps.is_gfx1100() {
        eprintln!("SKIP: test_qwen35_fa_batch_fusion_gfx1100 needs gfx1100");
        return;
    }
    println!("FA batch fusion parity (gfx1100):");
    for &n in &[1usize, 2, 8, 16] {
        test_prep(&mut gpu, n);
        test_kv_pair(&mut gpu, n);
    }
    println!("PASS: test_qwen35_fa_batch_fusion_gfx1100");
}
