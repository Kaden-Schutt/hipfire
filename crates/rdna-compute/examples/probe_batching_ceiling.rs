// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Task 0 of the SP1 batched-attention plan: measure the batching ceiling
// empirically instead of trusting roofline arithmetic.
//
// Method: whole-step decode latency at batch 1 across context lengths, then a
// least-squares fit of t(ctx) = a + b*ctx. `a` is the context-independent term
// (weights, DeltaNet, dense projections); `b*ctx` is the KV/attention term.
// Predicted batched step time at N slots is a_amortised + N*b*ctx.
//
// There is deliberately NO per-operation device_synchronize here: per-op syncs
// fabricate GPU speedups and would corrupt the fit. Only whole-step wall time
// is measured.
//
// Env: CTXS (comma-separated context lengths), ITERS, WARMUPS.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// Least-squares fit of y = a + b*x. Returns (a, b).
fn linfit(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| x * y).sum();
    let denom = n * sxx - sx * sx;
    assert!(denom.abs() > 1e-9, "context lengths must not all be equal");
    let b = (n * sxy - sx * sy) / denom;
    let a = (sy - b * sx) / n;
    (a, b)
}

fn main() {
    let ctxs: Vec<usize> = std::env::var("CTXS")
        .unwrap_or_else(|_| "4096,16384,32768,65536".into())
        .split(',')
        .map(|s| s.trim().parse().expect("CTXS must be integers"))
        .collect();
    let iters = env_usize("ITERS", 9);
    let warmups = env_usize("WARMUPS", 3);

    // Attention-only proxy for a decode step: one FA layer's worth of work at
    // batch 1, repeated `layers` times. Shapes default to qwen3.6-35b-a3b's
    // full-attention layers (nh=16, nkv=2, hd=256, 10 FA layers).
    let nh = env_usize("NH", 16);
    let nkv = env_usize("NKV", 2);
    let hd = env_usize("HD", 256);
    let layers = env_usize("LAYERS", 10);

    let mut gpu = Gpu::init().expect("gpu init");
    let blocks_per_head = hd / 32;
    let bytes_per_pos = nkv * blocks_per_head * 34;

    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for &ctx in &ctxs {
        let cache_bytes = ctx * bytes_per_pos;
        let mut kv = vec![0u8; cache_bytes];
        for blk in kv.chunks_mut(34) {
            blk[0] = 0x00;
            blk[1] = 0x3C; // fp16 1.0
            for (j, b) in blk[2..].iter_mut().enumerate() {
                *b = ((j as i32 % 7) - 3) as i8 as u8;
            }
        }
        let k_cache = gpu.upload_raw(&kv, &[cache_bytes]).expect("k upload");
        let v_cache = gpu.upload_raw(&kv, &[cache_bytes]).expect("v upload");

        let q_data: Vec<f32> = (0..nh * hd)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let q = gpu.upload_f32(&q_data, &[nh * hd]).expect("q upload");
        let out = gpu.zeros(&[nh * hd], DType::F32).expect("out");

        // positions are i32 bits uploaded through upload_raw — there is no
        // upload_i32 on Gpu. This matches q8_batched_attn_microbench.rs.
        let pos_data: Vec<i32> = vec![(ctx - 1) as i32];
        let pos_bytes = unsafe { std::slice::from_raw_parts(pos_data.as_ptr() as *const u8, 4) };
        let positions = gpu.upload_raw(pos_bytes, &[1]).expect("pos upload");

        let stride = 2 + hd;
        let max_tiles = ctx.div_ceil(128);
        let partials = gpu
            .zeros(&[nh * max_tiles * stride], DType::F32)
            .expect("partials");

        let mut run = |g: &mut Gpu| {
            for _ in 0..layers {
                g.attention_flash_q8_0_batched_masked(
                    &q, &k_cache, &v_cache, &out, &positions, nh, nkv, hd, ctx, ctx, 1, &partials,
                    None, 0, 0,
                )
                .expect("attn");
            }
        };

        for _ in 0..warmups {
            run(&mut gpu);
        }
        gpu.hip.device_synchronize().unwrap();

        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            run(&mut gpu);
            // One sync per WHOLE measured block, never per kernel. Per-op syncs
            // fabricate GPU speedups and would corrupt the slope fit.
            gpu.hip.device_synchronize().unwrap();
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = samples[samples.len() / 2];

        println!("ctx={ctx:>7}  median_ms={median:8.3}");
        xs.push(ctx as f64);
        ys.push(median);
    }

    let (a, b) = linfit(&xs, &ys);
    println!();
    println!("fit: t(ctx) = {a:.4} ms + {:.6} ms per 1K ctx", b * 1000.0);
    println!("  a (context-independent, does amortise across slots): {a:.4} ms");
    println!(
        "  b (KV term, does NOT amortise across slots):         {:.6} ms/1K",
        b * 1000.0
    );
    for n in [2usize, 4, 8] {
        for &ctx in &ctxs {
            let seq = (a + b * ctx as f64) * n as f64;
            let bat = a + b * ctx as f64 * n as f64;
            println!(
                "  N={n} ctx={ctx:>7}: seq={seq:8.3}ms batched={bat:8.3}ms speedup={:.2}x",
                seq / bat
            );
        }
    }
}
