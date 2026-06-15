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

    // (A) NEW batched.
    let new_ms = time(&mut gpu, &|g: &mut Gpu| {
        g.attention_flash_q8_0_batched_masked(
            &q, &k_cache, &v_cache, &out, &positions,
            nh, nkv, hd, ctx, ctx, n, &partials, None, 0, 0,
        ).expect("new batched");
    });

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

    println!("\n=== Q8 long-ctx attention: NEW batched vs OLD per-position ===");
    println!("NEW attention_flash_q8_0_batched_masked : {new_ms:8.2} ms");
    println!("OLD per-position loop (n={n} launches)   : {old_ms:8.2} ms");
    println!("speedup (OLD/NEW)                         : {:.2}x", old_ms / new_ms);
}
