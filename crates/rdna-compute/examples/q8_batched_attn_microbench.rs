// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Microbench for the no-LDS-cap batched Q8 flash attention introduced in
// fix/q8-batched-masked-no-lds-cap. Compares, at a single FA-layer scale:
//
//   (A) NEW  attention_flash_q8_0_batched_masked   — one batched launch
//   (B) OLD  attention_flash_q8_0 looped per query  — the >15k fallback it replaces
//
// at a controlled (n, max_ctx_len) shape so rocprof / wall timing isn't
// drowned by 64 layers × many prefill chunks. Reports wall ms (median of 5)
// for each. The point: confirm NEW ≤ OLD (the replacement is not a perf
// regression) at long context, where OLD launches `n` separate kernels.
//
// Shapes default to Qwen3.5-9B FA: n_heads=40, n_kv_heads=8, head_dim=256.
// Override via env: NH, NKV, HD, N (batch/query rows), CTX (max_ctx_len).
//
// Run (gfx906): cargo run --release --example q8_batched_attn_microbench

use rdna_compute::{DType, Gpu};

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let nh = env_usize("NH", 40);
    let nkv = env_usize("NKV", 8);
    let hd = env_usize("HD", 256);
    let n = env_usize("N", 512); // query rows in the prefill chunk
    let ctx = env_usize("CTX", 20000); // max_ctx_len — above the 15k cliff
    let iters = env_usize("ITERS", 5);

    assert!(hd % 32 == 0, "head_dim must be a multiple of 32");
    let mut gpu = Gpu::init().expect("gpu init");

    // Q8 K/V cache layout (matches kv_cache.k_gpu): per position,
    // n_kv_heads * (head_dim/32) blocks of 34 bytes (fp16 scale + 32 i8).
    let blocks_per_head = hd / 32;
    let bytes_per_pos = nkv * blocks_per_head * 34;
    let cache_bytes = ctx * bytes_per_pos;

    // Fill K/V with a POSITION-DEPENDENT pattern (scale=1.0 fp16 0x3C00,
    // codes vary by absolute position) so attention weights are non-uniform
    // — a meaningful correctness test, not a uniform-softmax degenerate one.
    let mut kv = vec![0u8; cache_bytes];
    for (pos, posrow) in kv.chunks_mut(bytes_per_pos).enumerate() {
        for blk in posrow.chunks_mut(34) {
            blk[0] = 0x00;
            blk[1] = 0x3C; // fp16 1.0
            for (j, b) in blk[2..].iter_mut().enumerate() {
                *b = (((j as i32 + pos as i32) % 13) - 6) as i8 as u8;
            }
        }
    }
    let k_cache = gpu.upload_raw(&kv, &[cache_bytes]).expect("k upload");
    let v_cache = gpu.upload_raw(&kv, &[cache_bytes]).expect("v upload");

    // Q: [n × n_heads × head_dim] f32.
    let q_data: Vec<f32> = (0..n * nh * hd).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let q = gpu.upload_f32(&q_data, &[n * nh * hd]).expect("q upload");
    let out = gpu.zeros(&[n * nh * hd], DType::F32).expect("out");

    // positions: i32 bits in f32 slot — positions[b] = ctx - n + b (the
    // queries sit at the tail of the context, as in real tail-chunk prefill).
    let pos_data: Vec<i32> = (0..n).map(|b| (ctx - n + b) as i32).collect();
    let pos_bytes = unsafe {
        std::slice::from_raw_parts(pos_data.as_ptr() as *const u8, n * 4)
    };
    let positions = gpu.upload_raw(pos_bytes, &[n]).expect("pos upload");

    // flash_partials: [sub_batch × n_heads × max_tiles × (2+head_dim)].
    // Size it for the full batch so sub_batch == n (single chunk).
    const TILE: usize = 128;
    let max_tiles = ctx.div_ceil(TILE);
    let partials_numel = n * nh * max_tiles * (2 + hd);
    let partials = gpu.zeros(&[partials_numel], DType::F32).expect("partials");

    eprintln!(
        "shape: nh={nh} nkv={nkv} hd={hd} n={n} ctx={ctx} | cache={:.1} MiB partials={:.1} MiB",
        cache_bytes as f64 / 1048576.0,
        partials_numel as f64 * 4.0 / 1048576.0,
    );

    // VERIFY=1: run the token-parallel path vs the tile_batched fallback on
    // identical inputs and report max abs diff. tile_batched is NIAH-PASS
    // (known-good); tokpar must match it within FP noise.
    if std::env::var("VERIFY").as_deref() == Ok("1") {
        let out_ref = gpu.zeros(&[n * nh * hd], DType::F32).expect("out_ref");
        let out_tp = gpu.zeros(&[n * nh * hd], DType::F32).expect("out_tp");
        // Force tile_batched (fallback) into out_ref.
        std::env::set_var("HIPFIRE_Q8_TOKPAR", "0");
        gpu.attention_flash_q8_0_batched_masked(
            &q, &k_cache, &v_cache, &out_ref, &positions,
            nh, nkv, hd, ctx, ctx, n, &partials, None, 0, 0,
        ).expect("ref");
        // Force tokpar into out_tp.
        std::env::set_var("HIPFIRE_Q8_TOKPAR", "1");
        gpu.attention_flash_q8_0_batched_masked(
            &q, &k_cache, &v_cache, &out_tp, &positions,
            nh, nkv, hd, ctx, ctx, n, &partials, None, 0, 0,
        ).expect("tokpar");
        gpu.hip.device_synchronize().unwrap();
        let a = gpu.download_f32(&out_ref).unwrap();
        let b = gpu.download_f32(&out_tp).unwrap();
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            max_abs = max_abs.max(d);
            let den = x.abs().max(1e-6);
            max_rel = max_rel.max(d / den);
        }
        let ref_norm: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("\n=== VERIFY tokpar vs tile_batched ===");
        println!("max abs diff: {max_abs:.3e}   max rel diff: {max_rel:.3e}   ref L2 norm: {ref_norm:.3e}");
        println!("{}", if max_abs < 1e-2 { "PASS (within FP noise)" } else { "FAIL — math mismatch" });
        std::env::remove_var("HIPFIRE_Q8_TOKPAR");
        return;
    }

    let time = |gpu: &mut Gpu, f: &dyn Fn(&mut Gpu)| -> f64 {
        f(gpu); // warmup
        gpu.hip.device_synchronize().unwrap();
        let mut ts = vec![];
        for _ in 0..iters {
            let t0 = std::time::Instant::now();
            f(gpu);
            gpu.hip.device_synchronize().unwrap();
            ts.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts[ts.len() / 2]
    };

    // (A0) tile_batched (scalar per-token-serial fallback).
    std::env::set_var("HIPFIRE_Q8_TOKPAR", "0");
    let tile_ms = time(&mut gpu, &|g: &mut Gpu| {
        g.attention_flash_q8_0_batched_masked(
            &q, &k_cache, &v_cache, &out, &positions,
            nh, nkv, hd, ctx, ctx, n, &partials, None, 0, 0,
        ).expect("tile_batched");
    });
    // (A) NEW token-parallel (default on gfx906/gfx1031).
    std::env::set_var("HIPFIRE_Q8_TOKPAR", "1");
    let new_ms = time(&mut gpu, &|g: &mut Gpu| {
        g.attention_flash_q8_0_batched_masked(
            &q, &k_cache, &v_cache, &out, &positions,
            nh, nkv, hd, ctx, ctx, n, &partials, None, 0, 0,
        ).expect("tokpar");
    });
    std::env::remove_var("HIPFIRE_Q8_TOKPAR");

    // (B) OLD per-position loop — replicate the fallback this PR removed.
    let pos_single: Vec<Vec<u8>> = (0..n)
        .map(|b| ((ctx - n + b) as i32).to_ne_bytes().to_vec())
        .collect();
    let pos_bufs: Vec<_> = pos_single.iter()
        .map(|bytes| gpu.upload_raw(bytes, &[1]).expect("pos1"))
        .collect();
    let old_ms = time(&mut gpu, &|g: &mut Gpu| {
        for b in 0..n {
            let q_b = q.sub_offset(b * nh * hd, nh * hd);
            let out_b = out.sub_offset(b * nh * hd, nh * hd);
            let seq_len = ctx - n + b + 1;
            g.attention_flash_q8_0(
                &q_b, &k_cache, &v_cache, &out_b,
                &pos_bufs[b].buf, seq_len, nh, nkv, hd, ctx, &partials,
            ).expect("old per-pos");
        }
    });

    // DRAM-bound check. The NEW kernel grid is [n_heads, tiles, batch]:
    // each of n_heads blocks reads its kv-head's K+V over the full ctx
    // once, so total K/V DRAM reads ≈ n_heads × ctx × bytes_per_kvhead
    // (K+V) — the GQA reload factor is baked in (n_heads, not n_kv_heads).
    // Causal halves it on average (queries at the tail see most of ctx,
    // but tiles past a query are skipped), so use ctx as an upper bound.
    let bytes_per_kvhead = blocks_per_head * 34; // one kv-head's K row (V same)
    let new_kv_reads = (nh as f64) * (ctx as f64) * (bytes_per_kvhead as f64) * 2.0; // K+V
    let new_gibs = new_kv_reads / (new_ms / 1000.0) / 1.073741824e9;
    // gfx906 (MI50) HBM2 peak ≈ 1024 GB/s ≈ 954 GiB/s.
    const PEAK_GIBS: f64 = 954.0;

    println!("\n=== Q8 long-ctx attention ===");
    println!("TOKPAR  (token-parallel, new)            : {new_ms:8.2} ms");
    println!("TILE    (scalar per-token-serial)        : {tile_ms:8.2} ms");
    println!("OLD     per-position loop (n={n})         : {old_ms:8.2} ms");
    println!("speedup TOKPAR vs TILE                    : {:.2}x", tile_ms / new_ms);
    println!("speedup TOKPAR vs OLD                     : {:.2}x", old_ms / new_ms);
    println!("--- NEW kernel BW (upper-bound K/V reads) ---");
    println!("K/V DRAM read (n_heads×ctx, K+V)         : {:.1} MiB",
        new_kv_reads / 1048576.0);
    println!("achieved BW                               : {new_gibs:.1} GiB/s ({:.0}% of ~{PEAK_GIBS:.0} GiB/s peak)",
        100.0 * new_gibs / PEAK_GIBS);
    println!("(>~60% peak ⇒ DRAM-bound ⇒ GQA-reuse is the lever)");
}
