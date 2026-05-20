//! Production-realistic bench for the MFP4G32 decode GEMV path
//! (`gemv_mfp4g32_with_rotate`). Compares env=0 fallback (rotate +
//! gemv_hfp4g32 fallback) vs env=1 A3 (fused rotate+pack + FP8 GEMV).

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    if !arch.starts_with("gfx12") {
        eprintln!("=== SKIP === FP8 dot4 path is gfx12-only (arch={arch})");
        return;
    }
    eprintln!("=== gemv_mfp4g32_with_rotate (production path) ===");
    eprintln!("  arch={arch}");

    let shapes: Vec<(usize, usize, &str)> = vec![
        (2048,  2048,  "qkv-q     M=2048 K=2048"),
        (512,   2048,  "qkv-kv    M=512  K=2048"),
        (11008, 2048,  "gate_up   M=11008 K=2048"),
        (2048,  11008, "w_down    M=2048  K=11008"),
        (4096,  2048,  "med       M=4096 K=2048"),
        (1024,  2048,  "small     M=1024 K=2048"),
    ];

    let trials = 200;
    let warmup = 20;

    for (m, k, label) in &shapes {
        let (m, k) = (*m, *k);
        let row_bytes = 16 + (k / 32) * 17;
        let total_w_bytes = m * row_bytes;

        let w = gpu.upload_raw(&synth(m, k, 0xAA00 | (m as u64) ^ (k as u64)), &[total_w_bytes]).unwrap();
        let x = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let y = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        let x_rot = gpu.alloc_tensor(&[k], DType::F32).unwrap();

        let x_host = make_x(k, 0x1111);
        gpu.hip.memcpy_htod(&x.buf, bytes_of(&x_host)).unwrap();

        // Force a different `x` per call by overwriting from host between
        // each launch — mimics the production case where each gemv has
        // freshly-written x (which is what blocks the v1 src_ptr cache).
        for _ in 0..warmup {
            gpu.gemv_mfp4g32_with_rotate(&w, &x, &y, &x_rot, m, k).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        let t = Instant::now();
        for _ in 0..trials {
            gpu.gemv_mfp4g32_with_rotate(&w, &x, &y, &x_rot, m, k).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / trials as f64;

        eprintln!("  {label:30}  {us:6.2} µs/call");
    }
}

fn make_x(n: usize, seed: i64) -> Vec<f32> {
    (0..n).map(|i| ((i as i64).wrapping_mul(seed.wrapping_add(0x91c2_a73d)).wrapping_add(seed) & 0xFFFFFF) as f32 * 1e-7 - 0.5).collect()
}

fn synth(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let blocks_per_row = k / 32;
    let row_bytes = 16 + blocks_per_row * 17;
    let mut out = vec![0u8; m * row_bytes];
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        let row_off = row * row_bytes;
        let rs_f32 = 0.02f32 + ((next() & 0xFF) as f32) * 1e-4;
        let rs_f16 = f32_to_f16_bits(rs_f32);
        out[row_off..row_off + 2].copy_from_slice(&rs_f16.to_le_bytes());
        let bc = blocks_per_row as u16;
        out[row_off + 4..row_off + 6].copy_from_slice(&bc.to_le_bytes());
        for b in 0..blocks_per_row {
            let bp = row_off + 16 + b * 17;
            let e = 120 + (next() & 0x7) as u8;
            out[bp] = e;
            for i in 0..16 {
                out[bp + 1 + i] = (next() & 0xFF) as u8;
            }
        }
    }
    out
}

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp == 0 { return sign; }
    if exp >= 143 { return sign | 0x7C00; }
    if exp <= 112 { return sign; }
    let new_exp = (exp - 127 + 15) as u16;
    let new_mant = (mant >> 13) as u16;
    sign | (new_exp << 10) | new_mant
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
