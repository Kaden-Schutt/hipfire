// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — MFP4-G32-E8 vs MQ4G256 decode cost on the Qwen3.5-0.8B stack.
//
// MFP4-E8 was ~20% behind MQ4 on decode. MQ4 does NOT decode E4M3 at all (it
// carries an f32 scale + zero per 256-weight group), while MFP4-E8 decodes one
// E4M3 block scale per 32 weights — so the gfx12 native OCP FP8 conversion
// should move this gap and nothing else. This measures where it now sits.
//
// The two formats do NOT move the same bytes, so a raw time delta is not the
// whole story and the byte ratio is reported alongside:
//   MQ4G256    (K/256) * 136 B per row
//   MFP4G32E8  16 B row header + (K/32) * 17 B per row
// At K=1024 that is 544 B vs 560 B — E8 carries +2.9% weight traffic before a
// single instruction executes, which bounds how close it can get.
//
// Methodology matches rwq4_decode_cost: variants interleaved inside the timing
// loop (DPM drift otherwise aliases into the delta), launches batched B=64
// (these kernels are 4-10 us against 3-5 us of ROCm launch overhead), payload
// randomized (a memset fill hides bank conflicts), and shapes weighted by real
// per-token invocation counts rather than averaged flat.
//
// Run: cargo run --release -p rdna-compute --example mfp4e8_vs_mq4_decode

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const BATCH: usize = 64;
const ITERS: usize = 12;
const REPS: usize = 7;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn fill(&mut self, buf: &mut [u8]) {
        for c in buf.chunks_mut(8) {
            let v = self.next_u64().to_le_bytes();
            let n = c.len();
            c.copy_from_slice(&v[..n]);
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// MQ4G256 / HFQ4-G256: (K/256) groups of `[f32 scale][f32 zero][128 B nibbles]`.
fn build_mq4(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let groups = k / 256;
    let row = groups * 136;
    let mut v = vec![0u8; m * row];
    rng.fill(&mut v);
    for r in 0..m {
        for g in 0..groups {
            let o = r * row + g * 136;
            // Sane scale/zero so neither kernel trips denormal slow paths.
            v[o..o + 4].copy_from_slice(&0.01f32.to_le_bytes());
            v[o + 4..o + 8].copy_from_slice(&(-0.08f32).to_le_bytes());
        }
    }
    v
}

/// MFP4G32E8: `[16 B row header][(K/32) x (1 B E4M3 scale + 16 B E8 codewords)]`.
/// Header is row_scale_a:f16 @0, n_blocks:u16 @4, flags=0x05 @6.
fn build_mfp4e8(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let blocks = k / 32;
    let row = 16 + blocks * 17;
    let mut v = vec![0u8; m * row];
    rng.fill(&mut v);
    for r in 0..m {
        let o = r * row;
        // f16 0.05 = 0x2A66; keep the row scale small and positive.
        v[o..o + 2].copy_from_slice(&0x2A66u16.to_le_bytes());
        v[o + 2..o + 4].copy_from_slice(&0u16.to_le_bytes());
        v[o + 4..o + 6].copy_from_slice(&(blocks as u16).to_le_bytes());
        v[o + 6] = 0x05;
        v[o + 7..o + 16].fill(0);
        for b in 0..blocks {
            // 0x38 decodes to 1.0 in E4M3 — a reachable, positive scale code.
            v[o + 16 + b * 17] = 0x38;
        }
    }
    v
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("=== MFP4-G32-E8 vs MQ4G256 decode, Qwen3.5-0.8B per-token stack ===");
    eprintln!("  arch={}", gpu.arch);
    eprintln!("  batch={BATCH} iters={ITERS} reps={REPS}, variants interleaved");
    eprintln!("  negative delta = E8 faster than MQ4");
    eprintln!();

    // 24 layers, hidden 1024, one lm_head pass over the 151936 vocab.
    let shapes: &[(usize, usize, usize, &str)] = &[
        (3072, 1024, 24, "fused_qkv   M=3072   K=1024"),
        (1024, 1024, 24, "o_proj      M=1024   K=1024"),
        (6144, 1024, 24, "gate_up     M=6144   K=1024"),
        (1024, 3072, 24, "down_proj   M=1024   K=3072"),
        (151936, 1024, 1, "lm_head     M=151936 K=1024"),
    ];

    let mut rng = Rng(0x4D46_5034_9E37_79B9);
    let (mut tot_mq4, mut tot_e8) = (0.0f64, 0.0f64);
    let (mut nolm_mq4, mut nolm_e8) = (0.0f64, 0.0f64);
    let (mut bytes_mq4, mut bytes_e8) = (0.0f64, 0.0f64);

    println!(
        "{:<28} {:>5} {:>10} {:>10} {:>9} {:>9}",
        "shape", "n/tok", "MQ4 ms", "E8 ms", "delta", "E8 bytes"
    );
    println!("{}", "-".repeat(78));

    for &(m, k, per_token, name) in shapes {
        let a_mq4 = build_mq4(&mut rng, m, k);
        let a_e8 = build_mfp4e8(&mut rng, m, k);
        let br = a_e8.len() as f64 / a_mq4.len() as f64;

        let x_host: Vec<f32> = (0..k)
            .map(|_| ((rng.next_u64() >> 40) as f32 / 8388608.0) - 1.0)
            .collect();

        let w_mq4 = gpu.upload_raw(&a_mq4, &[a_mq4.len()]).expect("upload mq4");
        let w_e8 = gpu.upload_raw(&a_e8, &[a_e8.len()]).expect("upload e8");
        let x = gpu.upload_f32(&x_host, &[k]).expect("upload x");
        let y = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y");

        for _ in 0..80 {
            gpu.gemv_hfq4g256(&w_mq4, &x, &y, m, k).expect("mq4 warm");
            gpu.gemv_mfp4g32_e8(&w_e8, &x, &y, m, k).expect("e8 warm");
        }
        gpu.hip.device_synchronize().expect("sync");

        let mut med_a: Vec<f64> = Vec::with_capacity(REPS);
        let mut med_b: Vec<f64> = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let mut ta: Vec<f64> = Vec::with_capacity(ITERS);
            let mut tb: Vec<f64> = Vec::with_capacity(ITERS);
            for _ in 0..ITERS {
                let s = Instant::now();
                for _ in 0..BATCH {
                    gpu.gemv_hfq4g256(&w_mq4, &x, &y, m, k).expect("mq4");
                }
                gpu.hip.device_synchronize().expect("sync");
                ta.push(s.elapsed().as_secs_f64() * 1e3 / BATCH as f64);

                let s = Instant::now();
                for _ in 0..BATCH {
                    gpu.gemv_mfp4g32_e8(&w_e8, &x, &y, m, k).expect("e8");
                }
                gpu.hip.device_synchronize().expect("sync");
                tb.push(s.elapsed().as_secs_f64() * 1e3 / BATCH as f64);
            }
            med_a.push(median(&mut ta));
            med_b.push(median(&mut tb));
        }
        let a = median(&mut med_a);
        let b = median(&mut med_b);

        tot_mq4 += a * per_token as f64;
        tot_e8 += b * per_token as f64;
        bytes_mq4 += a_mq4.len() as f64 * per_token as f64;
        bytes_e8 += a_e8.len() as f64 * per_token as f64;
        if !name.contains("lm_head") {
            nolm_mq4 += a * per_token as f64;
            nolm_e8 += b * per_token as f64;
        }

        println!(
            "{name:<28} {per_token:>5} {a:>10.4} {b:>10.4} {:>+8.2}% {:>+8.2}%",
            (b / a - 1.0) * 100.0,
            (br - 1.0) * 100.0
        );
    }

    println!("{}", "-".repeat(78));
    let incl = (tot_e8 / tot_mq4 - 1.0) * 100.0;
    let excl = (nolm_e8 / nolm_mq4 - 1.0) * 100.0;
    let byte_pen = (bytes_e8 / bytes_mq4 - 1.0) * 100.0;
    println!();
    println!("PER-TOKEN WEIGHTED");
    println!("  incl. lm_head : MQ4 {tot_mq4:.4} ms   E8 {tot_e8:.4} ms   {incl:+.3}%");
    println!("  excl. lm_head : MQ4 {nolm_mq4:.4} ms   E8 {nolm_e8:.4} ms   {excl:+.3}%");
    println!();
    println!("  E8 weight traffic is {byte_pen:+.2}% vs MQ4 — that much of the gap is");
    println!("  format bytes, not decode. Excess over it is decode cost:");
    println!("    incl. lm_head : {:+.3}%", incl - byte_pen);
    println!("    excl. lm_head : {:+.3}%", excl - byte_pen);
}
