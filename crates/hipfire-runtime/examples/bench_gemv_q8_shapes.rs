// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Self-contained Q8_0 GEMV bandwidth microbench at MiniMax-M2.7 decode shapes.
//! No model/GGUF dependency — synthetic random Q8_0 data. Measures achieved
//! GB/s per dense-projection shape so we know how close gemv_q8_0 already is to
//! the gfx1151 memory roofline (decides whether the multirow ILP lever has room).
//!
//! Run on hipx with HIP_VISIBLE_DEVICES=1 (gfx1151).

fn make_q8(m: usize, k: usize) -> Vec<u8> {
    // Q8_0 block = 2-byte f16 scale + 32 int8 = 34 bytes / 32 weights.
    let nblk = m * (k / 32);
    let mut buf = vec![0u8; nblk * 34];
    // scale = 1.0 in f16 = 0x3C00; qvals = deterministic sawtooth (data-independent timing).
    for blk in 0..nblk {
        let o = blk * 34;
        buf[o] = 0x00;
        buf[o + 1] = 0x3C;
        for i in 0..32 {
            buf[o + 2 + i] = ((blk + i) % 17) as u8;
        }
    }
    buf
}

fn bench_shape(gpu: &mut rdna_compute::Gpu, name: &str, m: usize, k: usize, roofline: f64) {
    let x: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let d_x = gpu.upload_f32(&x, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    let bytes = (m * (k / 32) * 34 + k * 4) as f64;
    let shape_bytes = m * (k / 32) * 34;

    // WARM: one buffer reused (cache-resident if it fits MALL) — the naive/inflated number.
    let q8 = make_q8(m, k);
    let d_q8 = gpu.upload_raw(&q8, &[q8.len()]).unwrap();
    let n_warm = 50;
    let n_iter = 300;
    for _ in 0..n_warm {
        gpu.gemv_q8_0(&d_q8, &d_x, &d_y, m, k).unwrap();
    }
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter {
        gpu.gemv_q8_0(&d_q8, &d_x, &d_y, m, k).unwrap();
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms_w = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let us_w = ms_w * 1000.0 / n_iter as f32;
    let bw_w = bytes * n_iter as f64 / (ms_w as f64 / 1000.0) / 1e9;

    // COLD: cycle through enough distinct buffers (>~160MB total) that any one is
    // evicted from MALL before it's revisited → measures real per-token DRAM cost.
    let n_slots = (((192 * 1024 * 1024) / shape_bytes.max(1)) + 1).max(4);
    let slots: Vec<_> = (0..n_slots)
        .map(|s| {
            // perturb each slot's bytes so they're distinct allocations
            let mut q = make_q8(m, k);
            q[0] = (s & 0xff) as u8;
            gpu.upload_raw(&q, &[q.len()]).unwrap()
        })
        .collect();
    for i in 0..n_warm {
        gpu.gemv_q8_0(&slots[i % n_slots], &d_x, &d_y, m, k).unwrap();
    }
    gpu.hip.event_record(&start, None).unwrap();
    for i in 0..n_iter {
        gpu.gemv_q8_0(&slots[i % n_slots], &d_x, &d_y, m, k).unwrap();
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms_c = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let us_c = ms_c * 1000.0 / n_iter as f32;
    let bw_c = bytes * n_iter as f64 / (ms_c as f64 / 1000.0) / 1e9;
    let pct_c = bw_c / roofline * 100.0;

    eprintln!(
        "{:<10} M={:>6} K={:>5} | warm {:>7.1}us {:>6.1}GB/s | COLD {:>7.1}us {:>6.1}GB/s {:>5.1}%roof ({} slots)",
        name, m, k, us_w, bw_w, us_c, bw_c, pct_c, n_slots
    );
}

fn main() {
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    eprintln!("arch = {}", gpu.arch);
    // Roofline probe first: a huge-M shape is maximally BW-bound (tons of blocks),
    // so its GB/s ~ practical read roofline for this access pattern. Use it to
    // normalize the others.
    eprintln!("\n--- roofline probe (huge M, K=3072) ---");
    // measure raw, roofline=1 so pct prints as the GB/s number itself
    bench_shape(&mut gpu, "roof", 65536, 3072, 1.0);
    eprintln!("(read the 'GB/s' above as the practical roofline; re-run mentally)\n");

    // Use a conservative 256 GB/s LPDDR5x theoretical as the % denominator.
    let roof = 256.0;
    eprintln!("--- MiniMax-M2.7 decode dense-Q8 shapes (% of {:.0} GB/s theoretical) ---", roof);
    bench_shape(&mut gpu, "q_proj", 6144, 3072, roof);
    bench_shape(&mut gpu, "k_proj", 1024, 3072, roof);
    bench_shape(&mut gpu, "v_proj", 1024, 3072, roof);
    bench_shape(&mut gpu, "o_proj", 3072, 6144, roof);
    bench_shape(&mut gpu, "router", 256, 3072, roof);
    bench_shape(&mut gpu, "lm_head", 200064, 3072, roof);
}
