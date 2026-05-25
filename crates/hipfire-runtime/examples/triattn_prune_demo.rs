// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! End-to-end prune demo for TriAttention (arXiv:2604.04921 §4.3).
//!
//! Populates a Q8_0 K-cache with synthetic data, runs `triattn_score_q8`
//! to rank positions, picks a top-B via z-score + max aggregation on
//! the host, calls `kv_compact_gather` to physically move the retained
//! rows to the front of a fresh cache buffer, and then rescoring on the
//! compacted cache must match the top-B scores of the original cache.
//!
//! This validates the full eviction pipeline without touching a live
//! model — the next step is to wire it into the per-token forward loop.

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::triattn::{self, BandCenter, TriAttnCenters};
    use rdna_compute::{DType, Gpu};

    // ── Config ─────────────────────────────────────────────────────────
    let n_heads = 16usize;
    let n_kv_heads = 4usize;
    let head_dim = 256usize;
    let n_bands = head_dim / 2;
    let rope_theta = 10_000_000.0f32;
    let partial_rotary_factor = 1.0f32;
    let n_rot = (head_dim as f32 * partial_rotary_factor) as usize;
    let seq_len = 128usize;
    let budget = 48usize;
    let p_q = seq_len as f32;

    // ── Synthetic centers and K cache ──────────────────────────────────
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (*seed >> 40) as u32;
        let uniform = bits as f32 / (1u32 << 24) as f32;
        uniform * 2.0 - 1.0
    }

    let mut centers = TriAttnCenters::new(1, n_heads, head_dim, rope_theta, partial_rotary_factor);
    let mut seed = 0xdeadbeefu64;
    for h in 0..n_heads {
        for f in 0..n_bands {
            centers.set(0, h, f, BandCenter {
                eq_re: 0.3 * lcg(&mut seed),
                eq_im: 0.3 * lcg(&mut seed),
                e_abs_q: 0.5 + 0.3 * lcg(&mut seed).abs(),
            });
        }
    }
    let mut centers_flat = Vec::with_capacity(n_heads * n_bands * 3);
    for h in 0..n_heads {
        for f in 0..n_bands {
            let c = centers.get(0, h, f);
            centers_flat.push(c.eq_re);
            centers_flat.push(c.eq_im);
            centers_flat.push(c.e_abs_q);
        }
    }

    let kv_dim = n_kv_heads * head_dim;
    let blocks_per_head = head_dim / 32;
    let bytes_per_pos = n_kv_heads * blocks_per_head * 34;
    let cache_floats = (seq_len * bytes_per_pos + 3) / 4;
    let compact_cache_floats = (budget * bytes_per_pos + 3) / 4;

    let mut gpu = Gpu::init().expect("gpu init");
    let k_cache = gpu.zeros(&[cache_floats], DType::F32).unwrap();
    let k_compact = gpu.zeros(&[compact_cache_floats], DType::F32).unwrap();
    let pos_dev = gpu.hip.malloc(4).unwrap();

    // Seed the cache row-by-row through the production Q8_0 writer so
    // the layout matches the attention kernels exactly.
    for pos in 0..seq_len {
        let row: Vec<f32> = (0..kv_dim).map(|_| 0.5 * lcg(&mut seed)).collect();
        let tmp = gpu.upload_f32(&row, &[kv_dim]).unwrap();
        let pos_bytes = (pos as i32).to_ne_bytes();
        gpu.hip.memcpy_htod(&pos_dev, &pos_bytes).unwrap();
        gpu.kv_cache_write_q8_0(&k_cache, &tmp, &pos_dev, n_kv_heads, head_dim).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    let centers_dev = gpu.upload_f32(&centers_flat, &[n_heads * n_bands * 3]).unwrap();

    // ── Score the full cache ───────────────────────────────────────────
    let scores_full = gpu.alloc_tensor(&[n_heads * seq_len], DType::F32).unwrap();
    gpu.triattn_score_q8(
        &k_cache, &centers_dev, &scores_full,
        n_heads, n_kv_heads, head_dim, n_rot, rope_theta, p_q, seq_len,
    ).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let scores_full_cpu = gpu.download_f32(&scores_full).unwrap();

    // ── Top-B selection on host ────────────────────────────────────────
    let retain = triattn::compute_retain_indices(&scores_full_cpu, n_heads, seq_len, budget);
    assert_eq!(retain.len(), budget);
    // Sanity: strictly ascending by construction.
    for w in retain.windows(2) { assert!(w[0] < w[1]); }
    eprintln!(
        "retain first 8: {:?}…  last 4: {:?}",
        &retain[..8.min(retain.len())],
        &retain[retain.len().saturating_sub(4)..],
    );

    let retain_bytes: Vec<u8> = retain.iter().flat_map(|&x| (x as i32).to_ne_bytes()).collect();
    let retain_dev = gpu.alloc_tensor(&[budget], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&retain_dev.buf, &retain_bytes).unwrap();

    // ── Physical compaction ────────────────────────────────────────────
    gpu.kv_compact_gather(&k_cache, &k_compact, &retain_dev, bytes_per_pos, budget).unwrap();
    gpu.hip.device_synchronize().unwrap();

    // ── Re-score the compacted cache ───────────────────────────────────
    //
    // Critical check: the score at cache-position j of the compacted
    // cache must equal the original score at position retain[j]. RoPE
    // phase was baked into each K when it was written, so compaction is
    // pure gather — post-RoPE K rows land at new cache indices without
    // any re-rotation. The scorer therefore reads identical K bytes and
    // must produce identical numbers (bit-for-bit within fp32 rounding).
    let scores_compact = gpu.alloc_tensor(&[n_heads * budget], DType::F32).unwrap();
    gpu.triattn_score_q8(
        &k_compact, &centers_dev, &scores_compact,
        n_heads, n_kv_heads, head_dim, n_rot, rope_theta, p_q, budget,
    ).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let scores_compact_cpu = gpu.download_f32(&scores_compact).unwrap();

    // ── Validate ───────────────────────────────────────────────────────
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for h in 0..n_heads {
        for j in 0..budget {
            let src_pos = retain[j] as usize;
            let a = scores_full_cpu[h * seq_len + src_pos];
            let b = scores_compact_cpu[h * budget + j];
            let diff = (a - b).abs();
            if diff > max_abs { max_abs = diff; }
            let denom = a.abs().max(b.abs()).max(1e-6);
            let rel = diff / denom;
            if rel > max_rel { max_rel = rel; }
        }
    }
    eprintln!(
        "compact re-score parity over {n_heads} × {budget} = {} entries: max |Δ|={:.2e}, max rel={:.2e}",
        n_heads * budget, max_abs, max_rel,
    );
    assert!(max_rel < 1e-4, "score mismatch after compaction");

    // Ranking sanity: averaged across heads, the compacted positions
    // should dominate the evicted ones.
    let mean_per_pos_full: Vec<f32> = (0..seq_len).map(|p| {
        let mut s = 0.0;
        for h in 0..n_heads { s += scores_full_cpu[h * seq_len + p]; }
        s / n_heads as f32
    }).collect();
    let retained_mean: f32 = retain.iter().map(|&r| mean_per_pos_full[r as usize]).sum::<f32>() / budget as f32;
    let mut evicted_mean = 0.0f32;
    let mut evicted_n = 0usize;
    for p in 0..seq_len {
        if !retain.contains(&(p as u32)) {
            evicted_mean += mean_per_pos_full[p];
            evicted_n += 1;
        }
    }
    evicted_mean /= evicted_n as f32;
    eprintln!(
        "mean score: retained={retained_mean:.3}  evicted={evicted_mean:.3}  (z-score+max ranks retained slightly over head-mean)"
    );

    eprintln!("✅ eviction pipeline end-to-end correct");
}
