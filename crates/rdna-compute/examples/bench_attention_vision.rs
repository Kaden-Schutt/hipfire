//! Focused bench harness for vision-encoder attention shapes.
//!
//! Runs `attention_dflash_wmma_f32` (M=16) and `attention_dflash_wmma_m32_f32`
//! at the dots.ocr smoke-image shape (B=L=19520, head_dim=128, n_heads=12,
//! n_kv_heads=12 — vision is self-attention with full per-head KV). Warm-up
//! iter + N timed iters. Designed for `rocprofv3 --kernel-include-regex` to
//! pin down the actual hot kernel without the parity-sweep noise.
//!
//! Usage:
//!     ./target/release/examples/bench_attention_vision [--iters N]
//!
//! Default 3 iters of each kernel. With `rocprofv3 --pmc <list> --` the
//! counters are summed over all kernel invocations matching the kernel
//! regex; pick whichever counters you care about.

use rdna_compute::{DType, Gpu};

fn lcg_data(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let u = (s >> 16) & 0x7fff;
            (u as f32 / 32_768.0 - 0.5) * 0.2
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut iters: usize = 3;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" => { iters = args[i + 1].parse().expect("--iters needs int"); i += 2; }
            other => panic!("unknown arg: {other}"),
        }
    }

    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    // dots.ocr smoke-image shape.
    let b = 19520usize;
    let l = 19520usize;
    let n_heads = 12usize;
    let n_kv_heads = 12usize;
    let hd = 128usize;
    eprintln!("shape: B={b} L={l} n_heads={n_heads} hd={hd}");

    let q = lcg_data(0xa5a5_a5a5, b * n_heads * hd);
    let k = lcg_data(0xc3c3_c3c3, l * n_kv_heads * hd);
    let v = lcg_data(0x9696_9696, l * n_kv_heads * hd);

    let d_q = gpu.upload_f32(&q, &[b * n_heads * hd]).unwrap();
    let d_k = gpu.upload_f32(&k, &[l * n_kv_heads * hd]).unwrap();
    let d_v = gpu.upload_f32(&v, &[l * n_kv_heads * hd]).unwrap();
    let d_out = gpu.zeros(&[b * n_heads * hd], DType::F32).unwrap();

    // f16 K/V scratch + one-shot cast (amortised across all iters).
    let d_k_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
    let d_v_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
    gpu.cast_f32_to_f16(&d_k, &d_k_f16).unwrap();
    gpu.cast_f32_to_f16(&d_v, &d_v_f16).unwrap();

    if gpu.arch_caps.has_wmma_w32_gfx12() {
        eprintln!("gfx12: running production dots.ocr v5 path; older gfx11 experiment variants are skipped");
        gpu.attention_dflash_wmma_m64_n32_f16kv_v5_f32(
            &d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd,
        ).unwrap();
        gpu.hip.device_synchronize().unwrap();

        let t = std::time::Instant::now();
        for _ in 0..iters {
            gpu.attention_dflash_wmma_m64_n32_f16kv_v5_f32(
                &d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd,
            ).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        eprintln!("M=64 N=32  v5 gfx12 (V_tile=32): {:.1} ms / iter ({iters} iters)", t.elapsed().as_secs_f32() * 1000.0 / iters as f32);

        gpu.free_tensor(d_q).unwrap();
        gpu.free_tensor(d_k).unwrap();
        gpu.free_tensor(d_v).unwrap();
        gpu.free_tensor(d_k_f16).unwrap();
        gpu.free_tensor(d_v_f16).unwrap();
        gpu.free_tensor(d_out).unwrap();
        return;
    }

    // Warm-up.
    gpu.attention_dflash_wmma_f32(&d_q, &d_k, &d_v, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    gpu.attention_dflash_wmma_m32_f32(&d_q, &d_k, &d_v, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    gpu.attention_dflash_wmma_n64_f32(&d_q, &d_k, &d_v, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    gpu.attention_dflash_wmma_n64_f16kv_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    gpu.attention_dflash_wmma_n128_f16kv_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    gpu.attention_dflash_wmma_m64_n128_f16kv_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    gpu.attention_dflash_wmma_m64_n128_f16kv_v2_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    gpu.attention_dflash_wmma_m64_n128_f16kv_v3_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    gpu.attention_dflash_wmma_m64_n128_f16kv_v4_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    gpu.attention_dflash_wmma_m64_n32_f16kv_v5_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    gpu.attention_dflash_wmma_m64_n32_f16kv_v6_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    gpu.attention_dflash_wmma_m128_n32_f16kv_v7_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    gpu.attention_dflash_wmma_m128_n32_f16kv_v7b_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    gpu.hip.device_synchronize().unwrap();

    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_dflash_wmma_f32(&d_q, &d_k, &d_v, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("M=16               wmma: {:.1} ms / iter ({iters} iters)", t.elapsed().as_secs_f32() * 1000.0 / iters as f32);

    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_dflash_wmma_m32_f32(&d_q, &d_k, &d_v, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("M=32               wmma: {:.1} ms / iter ({iters} iters)", t.elapsed().as_secs_f32() * 1000.0 / iters as f32);

    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_dflash_wmma_n64_f32(&d_q, &d_k, &d_v, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("M=32 N=64          wmma: {:.1} ms / iter ({iters} iters)", t.elapsed().as_secs_f32() * 1000.0 / iters as f32);

    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_dflash_wmma_n64_f16kv_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("M=32 N=64  f16-K/V wmma: {:.1} ms / iter ({iters} iters)", t.elapsed().as_secs_f32() * 1000.0 / iters as f32);

    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_dflash_wmma_n128_f16kv_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("M=32 N=128 f16-K/V wmma:     {:.1} ms / iter ({iters} iters)", t.elapsed().as_secs_f32() * 1000.0 / iters as f32);

    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_dflash_wmma_m64_n128_f16kv_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("M=64 N=128 f16-K/V O-reg wmma:    {:.1} ms / iter ({iters} iters)", t.elapsed().as_secs_f32() * 1000.0 / iters as f32);

    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_dflash_wmma_m64_n128_f16kv_v2_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("M=64 N=128 v2 (pad+coop softmax): {:.1} ms / iter ({iters} iters)", t.elapsed().as_secs_f32() * 1000.0 / iters as f32);

    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_dflash_wmma_m64_n128_f16kv_v3_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("M=64 N=128 v3 (hoisted S_lds):    {:.1} ms / iter ({iters} iters)", t.elapsed().as_secs_f32() * 1000.0 / iters as f32);

    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_dflash_wmma_m64_n128_f16kv_v4_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("M=64 N=128 v4 (V_lds_T):          {:.1} ms / iter ({iters} iters)", t.elapsed().as_secs_f32() * 1000.0 / iters as f32);

    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_dflash_wmma_m64_n32_f16kv_v5_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("M=64 N=32  v5 (V_tile=32):        {:.1} ms / iter ({iters} iters)", t.elapsed().as_secs_f32() * 1000.0 / iters as f32);

    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_dflash_wmma_m64_n32_f16kv_v6_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("M=64 N=32  v6 (V_lds_T):          {:.1} ms / iter ({iters} iters)", t.elapsed().as_secs_f32() * 1000.0 / iters as f32);

    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_dflash_wmma_m128_n32_f16kv_v7_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("M=128 N=32 v7 (sub-tile):         {:.1} ms / iter ({iters} iters)", t.elapsed().as_secs_f32() * 1000.0 / iters as f32);

    let t = std::time::Instant::now();
    for _ in 0..iters {
        gpu.attention_dflash_wmma_m128_n32_f16kv_v7b_f32(&d_q, &d_k_f16, &d_v_f16, &d_out, b, l, n_heads, n_kv_heads, hd).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    eprintln!("M=128 N=32 v7b (seq, no-share):   {:.1} ms / iter ({iters} iters)", t.elapsed().as_secs_f32() * 1000.0 / iters as f32);

    gpu.free_tensor(d_q).unwrap();
    gpu.free_tensor(d_k).unwrap();
    gpu.free_tensor(d_v).unwrap();
    gpu.free_tensor(d_k_f16).unwrap();
    gpu.free_tensor(d_v_f16).unwrap();
    gpu.free_tensor(d_out).unwrap();
}
