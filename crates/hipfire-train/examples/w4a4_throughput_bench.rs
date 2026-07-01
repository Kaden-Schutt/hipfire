//! SpinQuant W4A4 payoff: is `iu4·iu4` (W4A4) actually ~2× `iu4·iu8` (W4A8) in the
//! compute-bound prefill/batch regime on gfx1151? This is the throughput half of
//! the SpinQuant premise ("W4A4 almost halves our compute") — the quality half is
//! the learned rotation (`learned_r1_w4a4_probe`, +1.73 dB SQNR). Decode (B=1) is
//! weight-bandwidth-bound, so it is shown only as a reference point where the ratio
//! is expected to be ~1 (compute isn't the bottleneck there).
//!
//! W4A8's `iu4·iu8` GEMM unpacks the int4 weight to int8 and runs the int8 matrix
//! core, so its compute cost == `gemm_iu8_i32_wmma`. W4A4 runs `gemm_iu4_i32_wmma`
//! (int4 matrix core). We time the two production kernels head-to-head at the SAME
//! logical GEMM `[M,K]·[B,K]ᵀ → [B,M]` and report GOP/s + the iu4/iu8 ratio.
//!
//! NOTE: this measures the matrix-core INSTRUCTION throughput only (random bytes,
//! no rotation/dequant/rescale) — the fair "does the 4-bit core actually run 2×"
//! question. RDNA3 (gfx1103/1151) does NOT always double-rate narrow types (see
//! memory rdna3_wmma_accumulate / fsr4_rdna3_perf_notes), so the answer is the
//! point of the bench.
//!
//! Run (production kernels, precompiled — no JIT toolchain needed):
//!   source ./scripts/rocm-env.sh
//!   hipfire lock acquire "w4a4-bench"
//!   cargo run -p hipfire-train --release --example w4a4_throughput_bench
//!   hipfire lock release

use rdna_compute::Gpu;
use std::time::Instant;

fn rand_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) as u8
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP: {} lacks wave32 WMMA", gpu.arch);
        return Ok(());
    }
    println!("arch: {}", gpu.arch);
    println!(
        "\n  W4A4 (iu4·iu4) vs W4A8 (iu4·iu8 = int8 matrix core) GEMM throughput\n  \
         logical GEMM [M,K]·[B,K]ᵀ → [B,M]; 2·M·K·B ops. iters=50, warmup=10.\n"
    );
    println!(
        "  {:>6} {:>6} {:>6} | {:>10} {:>10} | {:>10} {:>10} | {:>7}",
        "M", "K", "B", "iu4 ms", "iu4 GOP/s", "iu8 ms", "iu8 GOP/s", "iu4/iu8"
    );
    println!("  {}", "-".repeat(78));

    // (M, K, B): decode (B=1, bandwidth-bound ref) + prefill/batch (compute-bound).
    let shapes: &[(usize, usize, usize)] = &[
        (4096, 4096, 1),    // decode reference (compute NOT the bottleneck)
        (4096, 4096, 128),  // small prefill batch
        (4096, 4096, 512),  // prefill
        (4096, 4096, 2048), // large prefill / batched
        (2048, 2048, 512),
        (11008, 4096, 512), // FFN-up shape (llama-ish)
        (512, 512, 512),    // Supra-50M-ish (small — launch/occupancy bound)
    ];
    let iters = 50u32;
    let warmup = 10u32;

    for &(m, k, b) in shapes {
        assert_eq!(k % 16, 0);
        // iu4: weight [M, K/2], act [B, K/2], out [B, M] i32.
        let w4 = gpu.upload_raw(&rand_bytes(m * (k / 2), 1), &[m, k / 2])?;
        let x4 = gpu.upload_raw(&rand_bytes(b * (k / 2), 2), &[b, k / 2])?;
        let y4 = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m])?;
        // iu8: weight [M, K], act [B, K], out [B, M] i32.
        let w8 = gpu.upload_raw(&rand_bytes(m * k, 3), &[m, k])?;
        let x8 = gpu.upload_raw(&rand_bytes(b * k, 4), &[b, k])?;
        let y8 = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m])?;

        let ops = 2.0 * m as f64 * k as f64 * b as f64;

        // iu4
        for _ in 0..warmup {
            gpu.gemm_iu4_i32_wmma(&w4, &x4, &y4, m, k, b)?;
        }
        gpu.device_synchronize()?;
        let t0 = Instant::now();
        for _ in 0..iters {
            gpu.gemm_iu4_i32_wmma(&w4, &x4, &y4, m, k, b)?;
        }
        gpu.device_synchronize()?;
        let iu4_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let iu4_gops = ops / (iu4_ms * 1e-3) / 1e9;

        // iu8
        for _ in 0..warmup {
            gpu.gemm_iu8_i32_wmma(&w8, &x8, &y8, m, k, b)?;
        }
        gpu.device_synchronize()?;
        let t1 = Instant::now();
        for _ in 0..iters {
            gpu.gemm_iu8_i32_wmma(&w8, &x8, &y8, m, k, b)?;
        }
        gpu.device_synchronize()?;
        let iu8_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let iu8_gops = ops / (iu8_ms * 1e-3) / 1e9;

        println!(
            "  {:>6} {:>6} {:>6} | {:>10.4} {:>10.1} | {:>10.4} {:>10.1} | {:>6.2}×",
            m,
            k,
            b,
            iu4_ms,
            iu4_gops,
            iu8_ms,
            iu8_gops,
            iu8_ms / iu4_ms
        );

        for t in [w4, x4, y4, w8, x8, y8] {
            gpu.free_tensor(t)?;
        }
    }

    println!(
        "\n  iu4/iu8 > ~1.7× at large B ⇒ the 4-bit matrix core is compute-faster and\n  \
         W4A4 pays off in prefill/batch. ~1× everywhere (esp. B=1) ⇒ gfx1151 does not\n  \
         double-rate int4 WMMA; W4A4's win there is bandwidth (half the weight bytes),\n  \
         not compute — frame deployment accordingly."
    );
    Ok(())
}
