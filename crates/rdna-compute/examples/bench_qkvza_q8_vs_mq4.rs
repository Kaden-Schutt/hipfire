//! Decode fused-QKVZA throughput: Q8_0 (fused_qkvza_q8_0) vs MQ4/HFQ4G256
//! (fused_qkvza_hfq4g256) at the REAL Qwen3.6-A3B LA shape. Answers: is the Q8
//! fused kernel BYTE-FLOORED (time_q8/time_mq4 ≈ 2, both near the same GB/s — Q8
//! just reads 2× bytes) or INEFFICIENT (ratio ≫ 2 / Q8 GB/s ≪ MQ4 GB/s — tunable)?
//! Single-layer weights (~24 MB Q8) are L3-resident on gfx1100, so absolute GB/s
//! is L3-rate; the RATIO is the signal.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

// A3B linear-attention in_proj shapes (from query_tensor on .mq4p): hidden K=2048.
const QKV_M: usize = 8192; // in_proj_qkv [8192, 2048]
const Z_M: usize = 4096; // in_proj_z   [4096, 2048]
const B_M: usize = 32; // in_proj_b   [32, 2048]
const A_M: usize = 32; // in_proj_a   [32, 2048]
const K: usize = 2048;

/// Q8_0: per row, K/32 blocks of [2B fp16 scale | 32 int8] = 34 B/block.
fn synth_q8(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let blocks = k / 32;
    let row = blocks * 34;
    let mut out = vec![0u8; m * row];
    let mut st = seed;
    for b in out.iter_mut() {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (st >> 56) as u8;
    }
    // valid fp16 scales (avoid NaN/Inf in the scale slot)
    for r in 0..m {
        for blk in 0..blocks {
            let off = r * row + blk * 34;
            out[off] = 0x00;
            out[off + 1] = 0x3c; // fp16 1.0
        }
    }
    out
}

/// HFQ4G256 (mq4): K/256 groups of 136 B [f32 scale | f32 zp | 32×u32 nibbles].
fn synth_hfq4g256(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let groups = k / 256;
    let row = groups * 136;
    let mut out = vec![0u8; m * row];
    let mut st = seed;
    let mut rng = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (st >> 33) as u32
    };
    for r in 0..m {
        for g in 0..groups {
            let off = r * row + g * 136;
            out[off..off + 4].copy_from_slice(&(0.004f32).to_bits().to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&(-0.02f32).to_bits().to_le_bytes());
            for w in 0..32 {
                out[off + 8 + w * 4..off + 8 + w * 4 + 4].copy_from_slice(&rng().to_le_bytes());
            }
        }
    }
    out
}

fn main() {
    let mut gpu = Gpu::init().expect("init");
    eprintln!(
        "arch={}  A3B LA: qkv={QKV_M} z={Z_M} b={B_M} a={A_M} K={K}",
        gpu.arch
    );

    // Q8 weights + buffers
    let q_qkv = gpu
        .upload_raw(&synth_q8(QKV_M, K, 1), &[QKV_M * (K / 32) * 34])
        .unwrap();
    let q_z = gpu
        .upload_raw(&synth_q8(Z_M, K, 2), &[Z_M * (K / 32) * 34])
        .unwrap();
    let q_b = gpu
        .upload_raw(&synth_q8(B_M, K, 3), &[B_M * (K / 32) * 34])
        .unwrap();
    let q_a = gpu
        .upload_raw(&synth_q8(A_M, K, 4), &[A_M * (K / 32) * 34])
        .unwrap();
    // MQ4 weights
    let m_qkv = gpu
        .upload_raw(&synth_hfq4g256(QKV_M, K, 1), &[QKV_M * (K / 256) * 136])
        .unwrap();
    let m_z = gpu
        .upload_raw(&synth_hfq4g256(Z_M, K, 2), &[Z_M * (K / 256) * 136])
        .unwrap();
    let m_b = gpu
        .upload_raw(&synth_hfq4g256(B_M, K, 3), &[B_M * (K / 256) * 136])
        .unwrap();
    let m_a = gpu
        .upload_raw(&synth_hfq4g256(A_M, K, 4), &[A_M * (K / 256) * 136])
        .unwrap();

    let x: Vec<f32> = (0..K).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    let dx = gpu.upload_f32(&x, &[K]).unwrap();
    let (yq, yz, yb, ya) = (
        gpu.zeros(&[QKV_M], DType::F32).unwrap(),
        gpu.zeros(&[Z_M], DType::F32).unwrap(),
        gpu.zeros(&[B_M], DType::F32).unwrap(),
        gpu.zeros(&[A_M], DType::F32).unwrap(),
    );

    let q8_bytes = (QKV_M + Z_M + B_M + A_M) * (K / 32) * 34;
    let mq4_bytes = (QKV_M + Z_M + B_M + A_M) * (K / 256) * 136;
    let n = 1000usize;

    macro_rules! time_q8 {
        () => {{
            for _ in 0..30 {
                gpu.fused_qkvza_q8_0(
                    &q_qkv, &q_z, &q_b, &q_a, &dx, &yq, &yz, &yb, &ya, QKV_M, Z_M, B_M, A_M, K,
                )
                .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..n {
                gpu.fused_qkvza_q8_0(
                    &q_qkv, &q_z, &q_b, &q_a, &dx, &yq, &yz, &yb, &ya, QKV_M, Z_M, B_M, A_M, K,
                )
                .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            t0.elapsed().as_secs_f64() * 1e6 / n as f64
        }};
    }
    macro_rules! time_mq4 {
        () => {{
            for _ in 0..30 {
                gpu.fused_qkvza_hfq4g256(
                    &m_qkv, &m_z, &m_b, &m_a, &dx, &yq, &yz, &yb, &ya, QKV_M, Z_M, B_M, A_M, K,
                )
                .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..n {
                gpu.fused_qkvza_hfq4g256(
                    &m_qkv, &m_z, &m_b, &m_a, &dx, &yq, &yz, &yb, &ya, QKV_M, Z_M, B_M, A_M, K,
                )
                .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            t0.elapsed().as_secs_f64() * 1e6 / n as f64
        }};
    }

    let _ = (time_q8!(), time_mq4!()); // discard first
    let q8_us = time_q8!();
    let mq4_us = time_mq4!();
    let gbps = |bytes: usize, us: f64| bytes as f64 / us / 1e3;

    eprintln!("\n  fused_qkvza  bytes/call    µs/call    GB/s");
    eprintln!(
        "  Q8_0        {:>8.2} MiB  {:>7.1}  {:>6.0}",
        q8_bytes as f64 / 1048576.0,
        q8_us,
        gbps(q8_bytes, q8_us)
    );
    eprintln!(
        "  MQ4         {:>8.2} MiB  {:>7.1}  {:>6.0}",
        mq4_bytes as f64 / 1048576.0,
        mq4_us,
        gbps(mq4_bytes, mq4_us)
    );
    eprintln!(
        "\n  byte ratio (Q8/MQ4) = {:.2}   TIME ratio (Q8/MQ4) = {:.2}",
        q8_bytes as f64 / mq4_bytes as f64,
        q8_us / mq4_us
    );
    eprintln!(
        "  GB/s ratio (MQ4/Q8) = {:.2}",
        gbps(mq4_bytes, mq4_us) / gbps(q8_bytes, q8_us)
    );
    eprintln!("\n  READ: time ratio ≈ byte ratio (~2.0) AND similar GB/s → Q8 is BYTE-FLOORED");
    eprintln!("        (only MQ4 attention reaches mq4 decode parity). time ratio ≫ 2 OR");
    eprintln!("        Q8 GB/s ≪ MQ4 GB/s → Q8 kernel is INEFFICIENT (tunable on-box).");
}
