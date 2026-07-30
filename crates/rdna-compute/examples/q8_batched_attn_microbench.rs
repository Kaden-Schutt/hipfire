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
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let nh = env_usize("NH", 40);
    let nkv = env_usize("NKV", 8);
    let hd = env_usize("HD", 256);
    let n = env_usize("N", 512); // query rows in the prefill chunk
    let ctx = env_usize("CTX", 20000); // max_ctx_len — above the 15k cliff
    let warmups = env_usize("WARMUPS", 3);
    let iters = env_usize("ITERS", 5);

    assert!(hd % 32 == 0, "head_dim must be a multiple of 32");
    let mut gpu = Gpu::init().expect("gpu init");

    // Q8 K/V cache layout (matches kv_cache.k_gpu): per position,
    // n_kv_heads * (head_dim/32) blocks of 34 bytes (fp16 scale + 32 i8).
    let blocks_per_head = hd / 32;
    let bytes_per_pos = nkv * blocks_per_head * 34;
    let cache_bytes = ctx * bytes_per_pos;

    // Fill K/V with a plausible-magnitude pattern: scale=1.0 (fp16 0x3C00),
    // codes = small ramp. Not numerically meaningful — we time, not verify
    // (correctness is the NIAH gate on the 32k fixture).
    let mut kv = vec![0u8; cache_bytes];
    for blk in kv.chunks_mut(34) {
        blk[0] = 0x00;
        blk[1] = 0x3C; // fp16 1.0 little-endian
        for (j, b) in blk[2..].iter_mut().enumerate() {
            *b = ((j as i32 % 7) - 3) as i8 as u8;
        }
    }
    let k_cache = gpu.upload_raw(&kv, &[cache_bytes]).expect("k upload");
    let v_cache = gpu.upload_raw(&kv, &[cache_bytes]).expect("v upload");

    // Q: [n × n_heads × head_dim] f32.
    let q_data: Vec<f32> = (0..n * nh * hd)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let q = gpu.upload_f32(&q_data, &[n * nh * hd]).expect("q upload");
    let out = gpu.zeros(&[n * nh * hd], DType::F32).expect("out");

    // positions: i32 bits in f32 slot — positions[b] = ctx - n + b (the
    // queries sit at the tail of the context, as in real tail-chunk prefill).
    let pos_data: Vec<i32> = (0..n).map(|b| (ctx - n + b) as i32).collect();
    let pos_bytes = unsafe { std::slice::from_raw_parts(pos_data.as_ptr() as *const u8, n * 4) };
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

    let time = |gpu: &mut Gpu, f: &dyn Fn(&mut Gpu)| -> f64 {
        for _ in 0..warmups {
            f(gpu);
        }
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

    // Windowed batched flash. WINDOW=0 is full causal; WINDOW>0 clips to the
    // sliding window. The grid (max_tiles) AND the reduce (n_tiles) are sized by
    // max_ctx_len regardless of window — so sweeping CTX at a FIXED window tests
    // whether windowing actually reduces prefill cost, or only skips the dots
    // while still paying O(ctx) tile-launch + reduce overhead.
    let window = env_usize("WINDOW", 0) as i32;
    let new_ms = time(&mut gpu, &|g: &mut Gpu| {
        if window == 0 {
            g.attention_flash_q8_0_batched_masked(
                &q, &k_cache, &v_cache, &out, &positions, nh, nkv, hd, ctx, ctx, n, &partials,
                None, 0, 0,
            )
            .expect("non-windowed batched");
        } else {
            g.attention_flash_q8_0_batched_masked_windowed(
                &q, &k_cache, &v_cache, &out, &positions, nh, nkv, hd, ctx, ctx, n, &partials,
                None, 0, 0, window,
            )
            .expect("windowed batched");
        }
    });

    println!(
        "WINDOW={window:6} CTX={ctx:6} N={n:4} HD={hd} : batched flash {new_ms:8.2} ms  ({:6.1} us/query-row)",
        new_ms * 1000.0 / n as f64
    );

    // Query-tiled flash prefill. BR/BC swept via env; LDS is independent of ctx.
    let br = env_usize("BR", 16);
    let bc = env_usize("BC", 32);
    let flash_ms = time(&mut gpu, &|g: &mut Gpu| {
        g.attention_q8_0_flash_prefill(
            &q, &k_cache, &v_cache, &out, &positions, nh, nkv, hd, ctx, n, br, bc,
        )
        .expect("flash prefill");
    });
    println!(
        "flash_prefill br={br} bc={bc} CTX={ctx} N={n}: {flash_ms:8.2} ms  \
         ({:6.1} us/query-row)  speedup_vs_tiled={:.2}x",
        flash_ms * 1000.0 / n as f64,
        new_ms / flash_ms
    );

    // WMMA (matrix-core) variant of the query-tiled kernel. Fixed 16x16 tiles.
    let wmma_ms = time(&mut gpu, &|g: &mut Gpu| {
        g.attention_q8_0_flash_prefill_wmma(
            &q, &k_cache, &v_cache, &out, &positions, nh, nkv, hd, n,
        )
        .expect("wmma flash prefill");
    });
    println!(
        "flash_wmma       CTX={ctx} N={n}: {wmma_ms:8.2} ms  \
         ({:6.1} us/query-row)  vs_tiled={:.2}x  vs_scalar_flash={:.2}x",
        wmma_ms * 1000.0 / n as f64,
        new_ms / wmma_ms,
        flash_ms / wmma_ms
    );

    // The legacy LDS-backed kernel is only launchable while
    // (max_ctx_len + block + head_dim) * 4 <= 64 KB; above that it cannot run
    // at all, which is exactly why dispatch crosses over at 8192.
    let legacy_lds = (ctx + 256 + hd) * 4;
    if legacy_lds <= 64 * 1024 {
        let legacy_ms = time(&mut gpu, &|g: &mut Gpu| {
            g.attention_q8_0_kv_batched_masked(
                &q, &k_cache, &v_cache, &out, &positions, nh, nkv, hd, ctx, ctx, n, None, 0, 0,
            )
            .expect("legacy lds kernel");
        });
        println!(
            "legacy_lds       CTX={ctx} N={n}: {legacy_ms:8.2} ms  \
             ({:6.1} us/query-row)  flash_speedup_vs_legacy={:.2}x",
            legacy_ms * 1000.0 / n as f64,
            legacy_ms / flash_ms
        );
    } else {
        println!("legacy_lds       CTX={ctx}: N/A (needs {legacy_lds} B LDS > 64 KB)");
    }
}
