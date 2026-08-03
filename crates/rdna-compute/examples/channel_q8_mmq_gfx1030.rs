//! Channel test: gemm_q8_0_mmq_gfx1030 vs gemm_q8_0_batched (legacy scalar).
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

fn pack_q8_0(w: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 32, 0);
    let blocks = k / 32;
    let mut out = vec![0u8; m * blocks * 34];
    for row in 0..m {
        for bi in 0..blocks {
            let base = row * k + bi * 32;
            let slice = &w[base..base + 32];
            let amax = slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-8);
            let d = amax / 127.0;
            let off = (row * blocks + bi) * 34;
            // pack fp16 scale via software
            let bits = {
                // round-to-nearest even f32->f16
                let x = d;
                let b = x.to_bits();
                let sign = (b >> 16) & 0x8000;
                let exp = ((b >> 23) & 0xff) as i32;
                let mant = b & 0x7fffff;
                let h = if exp == 255 {
                    sign | 0x7c00 | (mant >> 13)
                } else if exp > 142 {
                    sign | 0x7c00
                } else if exp < 113 {
                    sign
                } else {
                    let e = (exp - 112) as u32;
                    let m = mant + 0x1000;
                    sign | (e << 10) | (m >> 13)
                };
                h as u16
            };
            out[off] = (bits & 0xff) as u8;
            out[off + 1] = (bits >> 8) as u8;
            for j in 0..32 {
                let q = (slice[j] / d).round().clamp(-127.0, 127.0) as i8;
                out[off + 2 + j] = q as u8;
            }
        }
    }
    out
}
fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    assert_eq!(gpu.arch, "gfx1030", "this channel test is exact-gfx1030 only");
    println!("arch={}", gpu.arch);
    // Shapes typical of LA qkv / router on A3B
    let cases = [
        (8192usize, 2048usize, 256usize), // LA wqkv-ish
        (4096, 2048, 256),
        (2048, 4096, 256),
        (512, 2048, 256),
        (256, 2048, 256),
        (32, 2048, 256),
        (17, 128, 15), // ragged tile edges
    ];
    for (m, k, n) in cases {
        let mut w = vec![0f32; m * k];
        let mut x = vec![0f32; n * k];
        for i in 0..w.len() {
            w[i] = (((i * 17) % 200) as f32 - 100.0) / 50.0;
        }
        for i in 0..x.len() {
            x[i] = (((i * 13) % 100) as f32 - 50.0) / 40.0;
        }
        let wq = pack_q8_0(&w, m, k);
        let w_gpu = {
            let b = gpu.hip.malloc(wq.len()).unwrap();
            gpu.hip.memcpy_htod(&b, &wq).unwrap();
            GpuTensor {
                buf: unsafe { hip_bridge::DeviceBuffer::from_raw(b.as_ptr(), wq.len()) },
                shape: vec![m, k],
                dtype: DType::Q8_0,
            }
        };
        let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x_gpu = {
            let b = gpu.hip.malloc(x_bytes.len()).unwrap();
            gpu.hip.memcpy_htod(&b, &x_bytes).unwrap();
            GpuTensor {
                buf: unsafe { hip_bridge::DeviceBuffer::from_raw(b.as_ptr(), x_bytes.len()) },
                shape: vec![n, k],
                dtype: DType::F32,
            }
        };
        let y_leg = {
            let b = gpu.hip.malloc(n * m * 4).unwrap();
            gpu.hip.memset(&b, 0, n * m * 4).unwrap();
            GpuTensor {
                buf: unsafe { hip_bridge::DeviceBuffer::from_raw(b.as_ptr(), n * m * 4) },
                shape: vec![n, m],
                dtype: DType::F32,
            }
        };
        let y_mmq = {
            let b = gpu.hip.malloc(n * m * 4).unwrap();
            gpu.hip.memset(&b, 0, n * m * 4).unwrap();
            GpuTensor {
                buf: unsafe { hip_bridge::DeviceBuffer::from_raw(b.as_ptr(), n * m * 4) },
                shape: vec![n, m],
                dtype: DType::F32,
            }
        };
        // Force legacy scalar path via env-less direct call
        {
            let mut off = 0;
            while off < n {
                let take = (n - off).min(64);
                let xs = x_gpu.sub_offset(off * k, take * k);
                let ys = y_leg.sub_offset(off * m, take * m);
                gpu.gemm_q8_0_batched(&w_gpu, &xs, &ys, m, k, take).unwrap();
                off += take;
            }
        }
        gpu.gemm_q8_0_mmq_gfx1030(&w_gpu, &x_gpu, &y_mmq, m, k, n).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let leg = gpu.download_f32(&y_leg).unwrap();
        let mmq = gpu.download_f32(&y_mmq).unwrap();
        let mut max_abs = 0f32;
        let mut sum_sq = 0f32;
        let mut sum_ref_sq = 0f32;
        for i in 0..leg.len() {
            let d = mmq[i] - leg[i];
            max_abs = max_abs.max(d.abs());
            sum_sq += d * d;
            sum_ref_sq += leg[i] * leg[i];
        }
        let rms = (sum_sq / leg.len() as f32).sqrt();
        let rel = rms / (sum_ref_sq / leg.len() as f32).sqrt().max(1e-8);
        println!(
            "m={m:<5} k={k:<5} n={n:<4}  max_abs={max_abs:.6} rms={rms:.6} rel={rel:.6}"
        );
        // timing
        for _ in 0..3 {
            gpu.gemm_q8_0_mmq_gfx1030(&w_gpu, &x_gpu, &y_mmq, m, k, n).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        let trials = 20;
        for _ in 0..trials {
            gpu.gemm_q8_0_mmq_gfx1030(&w_gpu, &x_gpu, &y_mmq, m, k, n).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() / trials as f64 * 1e6;
        // legacy timing
        for _ in 0..3 {
            let mut off = 0;
            while off < n {
                let take = (n - off).min(64);
                let xs = x_gpu.sub_offset(off * k, take * k);
                let ys = y_leg.sub_offset(off * m, take * m);
                gpu.gemm_q8_0_batched(&w_gpu, &xs, &ys, m, k, take).unwrap();
                off += take;
            }
        }
        gpu.hip.device_synchronize().unwrap();
        let t1 = Instant::now();
        for _ in 0..trials {
            let mut off = 0;
            while off < n {
                let take = (n - off).min(64);
                let xs = x_gpu.sub_offset(off * k, take * k);
                let ys = y_leg.sub_offset(off * m, take * m);
                gpu.gemm_q8_0_batched(&w_gpu, &xs, &ys, m, k, take).unwrap();
                off += take;
            }
        }
        gpu.hip.device_synchronize().unwrap();
        let us_leg = t1.elapsed().as_secs_f64() / trials as f64 * 1e6;
        println!("  time mmq={us:.1}us  legacy={us_leg:.1}us  speedup={:.1}x", us_leg / us);
        std::mem::forget(w_gpu);
        std::mem::forget(x_gpu);
        std::mem::forget(y_leg);
        std::mem::forget(y_mmq);
    }
}
