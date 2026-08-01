// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — RWQ4G256 vs MQ4G256 decode cost.
//
// RWQ4 is a drop-in successor to MQ4G256 at the IDENTICAL 136 B/group budget,
// so this is a like-for-like kernel comparison: same bytes moved, same shapes,
// same pre-rotated-x contract. The only difference is what the inner loop does
// with the group header (affine scale+min vs LDS codebook + factored E4M3
// sub-scale).
//
// Methodology — each of these changed the answer during earlier probes, so none
// of them are optional:
//   1. Variants are INTERLEAVED inside the timing loop. Running all of A then
//      all of B aliases DPM/thermal drift into the delta; a measurement of
//      identical code against itself read +6% that way.
//   2. Launches are BATCHED (B=64) per timed region. These kernels are 5-10 us
//      and ROCm launch overhead is 3-5 us, so unbatched timing measures the
//      launch path more than the kernel.
//   3. The weight payload is RANDOMIZED. A memset fill gives every lane the
//      same nibble and hides LDS bank conflicts in the codebook lookup.
//   4. Shapes are weighted by their real per-token invocation count in the
//      Qwen3.5-0.8B decode stack, so the headline number is what a token
//      actually pays rather than an unweighted shape average.
//
// Run: cargo run --release -p rdna-compute --example rwq4_decode_cost

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const GROUP: usize = 256;
const GROUP_BYTES: usize = 136;

/// Batched launches per timed region (see methodology note 2).
const BATCH: usize = 64;
/// Timed regions per repetition.
const ITERS: usize = 12;
/// Repetitions; the per-shape result is the median of these medians.
const REPS: usize = 7;

/// Deterministic xorshift64* so the payload is identical run to run.
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
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("=== RWQ4G256 vs MQ4G256 decode cost ===");
    eprintln!("  arch={}", gpu.arch);
    eprintln!("  identical 136 B/group budget — like-for-like kernel comparison");
    eprintln!("  batch={BATCH} iters={ITERS} reps={REPS}, variants interleaved");
    eprintln!();

    // (M, K, per-token invocations, name) for the Qwen3.5-0.8B decode stack:
    // 24 layers, hidden 1024, and one lm_head pass over the 151936 vocab.
    let shapes: &[(usize, usize, usize, &str)] = &[
        (3072, 1024, 24, "fused_qkv   M=3072  K=1024"),
        (1024, 1024, 24, "o_proj      M=1024  K=1024"),
        (6144, 1024, 24, "gate_up     M=6144  K=1024"),
        (1024, 3072, 24, "down_proj   M=1024  K=3072"),
        (151936, 1024, 1, "lm_head     M=151936 K=1024"),
    ];

    let mut rng = Rng(0x5257_5134_9E37_79B9);
    let mut deltas: Vec<f64> = Vec::new();
    let (mut tot_mq4, mut tot_rwq4) = (0.0f64, 0.0f64);
    let (mut nolm_mq4, mut nolm_rwq4) = (0.0f64, 0.0f64);

    println!(
        "{:<26} {:>5} {:>11} {:>11} {:>9} {:>9}",
        "shape", "n/tok", "MQ4 ms", "RWQ4 ms", "delta", "GB/s"
    );
    println!("{}", "-".repeat(76));

    for &(m, k, per_token, name) in shapes {
        let groups = k / GROUP;
        let total = m * groups * GROUP_BYTES;

        // Randomized payload: real nibble entropy so codebook lookups hit the
        // full LDS bank spread (methodology note 3).
        let mut packed = vec![0u8; total];
        for chunk in packed.chunks_mut(8) {
            let v = rng.next_u64().to_le_bytes();
            let n = chunk.len();
            chunk.copy_from_slice(&v[..n]);
        }
        // Force every group's f32 master/scale header to a sane positive value
        // so neither kernel trips denormal slow paths on garbage exponents.
        for g in 0..(total / GROUP_BYTES) {
            let off = g * GROUP_BYTES;
            packed[off..off + 4].copy_from_slice(&1.0f32.to_le_bytes());
            // MQ4 reads [4..8) as an f32 zero-point; RWQ4 reads it as 4 E4M3
            // sub-scale bytes. 0x38 decodes to 1.0 in E4M3 and to a small
            // normal f32 when the four bytes are read together — valid for both.
            packed[off + 4..off + 8].copy_from_slice(&[0x38, 0x38, 0x38, 0x38]);
        }

        let x_host: Vec<f32> = (0..k)
            .map(|_| ((rng.next_u64() >> 40) as f32 / 8388608.0) - 1.0)
            .collect();

        let w = gpu.upload_raw(&packed, &[total]).expect("upload weights");
        let x = gpu.upload_f32(&x_host, &[k]).expect("upload x");
        let y = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y");

        // Warmup both paths before any timing.
        for _ in 0..80 {
            gpu.gemv_mq4g256_prerotated(&w, &x, &y, m, k).expect("mq4 warmup");
            gpu.gemv_rwq4g256_prerotated(&w, &x, &y, m, k).expect("rwq4 warmup");
        }
        gpu.hip.device_synchronize().expect("sync");

        let mut med_mq4: Vec<f64> = Vec::with_capacity(REPS);
        let mut med_rwq4: Vec<f64> = Vec::with_capacity(REPS);

        for _ in 0..REPS {
            let mut t_mq4: Vec<f64> = Vec::with_capacity(ITERS);
            let mut t_rwq4: Vec<f64> = Vec::with_capacity(ITERS);
            // Interleaved A/B inside the loop (methodology note 1).
            for _ in 0..ITERS {
                let s = Instant::now();
                for _ in 0..BATCH {
                    gpu.gemv_mq4g256_prerotated(&w, &x, &y, m, k).expect("mq4");
                }
                gpu.hip.device_synchronize().expect("sync");
                t_mq4.push(s.elapsed().as_secs_f64() * 1e3 / BATCH as f64);

                let s = Instant::now();
                for _ in 0..BATCH {
                    gpu.gemv_rwq4g256_prerotated(&w, &x, &y, m, k).expect("rwq4");
                }
                gpu.hip.device_synchronize().expect("sync");
                t_rwq4.push(s.elapsed().as_secs_f64() * 1e3 / BATCH as f64);
            }
            med_mq4.push(median(&mut t_mq4));
            med_rwq4.push(median(&mut t_rwq4));
        }

        let a = median(&mut med_mq4);
        let b = median(&mut med_rwq4);
        let delta = (b / a - 1.0) * 100.0;
        let gbps = total as f64 / (b * 1e-3) / 1e9;
        deltas.push(delta);

        tot_mq4 += a * per_token as f64;
        tot_rwq4 += b * per_token as f64;
        if !name.contains("lm_head") {
            nolm_mq4 += a * per_token as f64;
            nolm_rwq4 += b * per_token as f64;
        }

        println!(
            "{name:<26} {per_token:>5} {a:>11.4} {b:>11.4} {delta:>+8.2}% {gbps:>9.1}"
        );
    }

    println!("{}", "-".repeat(76));
    let incl = (tot_rwq4 / tot_mq4 - 1.0) * 100.0;
    let excl = (nolm_rwq4 / nolm_mq4 - 1.0) * 100.0;
    println!();
    println!("PER-TOKEN WEIGHTED TOTAL (the number that matters)");
    println!("  incl. lm_head : MQ4 {tot_mq4:.4} ms   RWQ4 {tot_rwq4:.4} ms   {incl:+.3}%");
    println!("  excl. lm_head : MQ4 {nolm_mq4:.4} ms   RWQ4 {nolm_rwq4:.4} ms   {excl:+.3}%");
    println!();
    // The project's stated budget: RWQ4 may cost at most 1% of decode speed.
    let worst = incl.max(excl);
    if worst <= 1.0 {
        println!("PASS: within the 1% decode budget (worst weighted delta {worst:+.3}%)");
    } else {
        println!("FAIL: exceeds the 1% decode budget (worst weighted delta {worst:+.3}%)");
        std::process::exit(1);
    }
}
