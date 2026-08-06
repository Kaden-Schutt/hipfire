//! WS1 KEYSTONE PROBE — indexed MoE gate_up GEMV at the REAL decode grid.
//!
//! WS1 measured the *standalone* gate_up GEMV at grid [M,1,1] and built a
//! byte-traffic model for the 150-vs-110-vs-102 tok/s spread. Both adversarial
//! reviewers flagged the same hole: decode runs the *indexed* kernel at
//! `[M, K_TOP=8, 1]` with a 256-expert pointer table, and the routed-GEMV's
//! FRACTION of the 6.67 ms decode token was never measured. This probe closes
//! that hole.
//!
//! Two questions, two regimes:
//!   1. FRACTION — what share of the decode token is the routed gate_up GEMV?
//!      routed_gu_us/token = us_per_launch × n_moe_layers(40); fraction vs the
//!      6667 us/token of mq4@150. If large (~30%+), the byte ceiling binds and
//!      WS1's 122/102 numbers are roughly real; if tiny (~5%), the byte story is
//!      a proxy and the spread lives elsewhere (shared expert / attn / graph).
//!   2. KERNEL-FIXABLE H2 — (c) merged-on-MQ4-data minus (a) mq4-uniform, on
//!      IDENTICAL mq4 buffers with all-2 (MQ4) tags, is the PURE kernel overhead
//!      of the merged kernel (lost quad-unroll + unconditional LDS). This is the
//!      actionable "restore the unroll" upside, measured at the real grid.
//!
//! Regimes:
//!   HOT   — fixed top-8 [0..7]; 8 experts (8.5 MiB) stay L3-resident → optimistic.
//!   COLD  — cycle 32 disjoint top-8 sets across all 256 experts; 256-expert
//!           working set (~272 MiB) >> 96 MiB L3 → VRAM-bound, the realistic
//!           decode regime (the synthesizer's "L3 pressure from 256 experts").
//!
//! Timing: N back-to-back enqueues on the default stream, ONE device_synchronize
//! → GPU-throughput us/launch (graph-replay-relevant; excludes per-call CPU
//! launch overhead, which the decode hipGraph also elides).
//!
//! Run (gfx1100 / RDNA3 dGPU):
//!   cargo run --release -p rdna-compute --example bench_indexed_moe_keystone

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

// A3B gate_up decode shape (qwen35.rs:168-170): hidden=2048, moe_intermediate=512.
const MI: usize = 512;
const M: usize = 2 * MI; // 1024 — kernel splits gate vs up at M/2
const K: usize = 2048; // hidden
const N_EXP: usize = 256;
const K_TOP: usize = 8;
const N_MOE_LAYERS: usize = 40; // num_hidden_layers (all sparse-MoE on A3B)
const FULL_DECODE_US: f64 = 1.0e6 / 150.0; // 6667 us/token at the mq4 150 tok/s anchor

fn upload_u8(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::Raw)
        .expect("alloc u8");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod u8");
    t
}
fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len()], DType::F32)
        .expect("alloc f32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod f32");
    t
}
fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len() * 4], DType::Raw)
        .expect("alloc i32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod i32");
    t
}
fn upload_u64(gpu: &mut Gpu, data: &[u64]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
    let t = gpu
        .alloc_tensor(&[data.len() * 8], DType::Raw)
        .expect("alloc u64");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod u64");
    t
}
fn alloc_f32_zeros(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let t = gpu.alloc_tensor(&[n], DType::F32).expect("alloc zeros");
    gpu.hip.memset(&t.buf, 0, n * 4).expect("memset");
    t
}

/// Standalone HFQ4-G256 (mq4) expert: groups = K/256 groups of 136 B
/// ([f32 scale][f32 zp][32×u32 nibbles]). Matches gemv_hfq4g256 + profile.
fn synth_hfq4g256(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let groups = k / 256;
    let row_bytes = groups * 136;
    let mut out = vec![0u8; m * row_bytes];
    let mut st = seed;
    let mut rng = || -> u32 {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (st >> 33) as u32
    };
    for row in 0..m {
        for g in 0..groups {
            let off = row * row_bytes + g * 136;
            let sc: f32 = 0.003 + (rng() & 0x3F) as f32 * 1e-4;
            out[off..off + 4].copy_from_slice(&sc.to_bits().to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&(-0.02f32).to_bits().to_le_bytes());
            for w in 0..32 {
                let pk = rng();
                out[off + 8 + w * 4..off + 8 + w * 4 + 4].copy_from_slice(&pk.to_le_bytes());
            }
        }
    }
    out
}

/// HFQ6-G256 (mq6) expert: same g256 structure, 200 B/group. Byte VALUES are
/// timing-irrelevant for this memory-bound GEMV, so size-correct random bytes
/// suffice (we measure throughput, not correctness here).
fn synth_hfq6g256(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let groups = k / 256;
    let row_bytes = groups * 200;
    let mut out = vec![0u8; m * row_bytes];
    let mut st = seed;
    for b in out.iter_mut() {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (st >> 56) as u8;
    }
    out
}

fn make_x(n: usize, seed: u64) -> Vec<f32> {
    let mut st = seed;
    (0..n)
        .map(|_| {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    eprintln!("=== WS1 keystone: indexed MoE gate_up at the real decode grid ===");
    eprintln!(
        "arch={arch}  shape M={M} K={K} (mi={MI})  n_exp={N_EXP} k_top={K_TOP}  layers={N_MOE_LAYERS}"
    );
    if !gpu.arch_caps.is_wave32() {
        eprintln!("SKIP — indexed decode probe requires an RDNA wave32 device.");
        return;
    }

    // ---- 256 mq4 expert buffers + ptr table (config a, and c reuses them) ----
    let mut keep: Vec<GpuTensor> = Vec::new();
    let mut ptrs_mq4: Vec<u64> = Vec::with_capacity(N_EXP);
    for e in 0..N_EXP {
        let t = upload_u8(&mut gpu, &synth_hfq4g256(M, K, 0x4D51 ^ e as u64));
        ptrs_mq4.push(t.buf.as_ptr() as u64);
        keep.push(t);
    }
    let expert_ptrs_mq4 = upload_u64(&mut gpu, &ptrs_mq4);

    // ---- 256 mq6 expert buffers + ptr table (config b) ----
    let mut ptrs_mq6: Vec<u64> = Vec::with_capacity(N_EXP);
    for e in 0..N_EXP {
        let t = upload_u8(&mut gpu, &synth_hfq6g256(M, K, 0x4D36 ^ e as u64));
        ptrs_mq6.push(t.buf.as_ptr() as u64);
        keep.push(t);
    }
    let expert_ptrs_mq6 = upload_u64(&mut gpu, &ptrs_mq6);

    // ---- tags = all-2 (MQ4 tier) for the merged kernel on the mq4 buffers ----
    let tags_all_mq4 = upload_i32(&mut gpu, &vec![2i32; N_EXP]);

    // ---- 32 disjoint top-8 sets covering all 256 experts (COLD regime) ----
    let topk_sets: Vec<GpuTensor> = (0..N_EXP / K_TOP)
        .map(|s| {
            let idx: Vec<i32> = (0..K_TOP).map(|j| (s * K_TOP + j) as i32).collect();
            upload_i32(&mut gpu, &idx)
        })
        .collect();
    let topk_hot = &topk_sets[0]; // fixed [0..7]

    let xr = upload_f32(&mut gpu, &make_x(K, 0xABCD));
    let gate_batch = alloc_f32_zeros(&mut gpu, K_TOP * MI);
    let up_batch = alloc_f32_zeros(&mut gpu, K_TOP * MI);

    // weight bytes read per launch = k_top × per-expert bytes (gate_up)
    let mq4_bytes_launch = (K_TOP * rdna_compute::profile::gemv_hfq4g256_bytes(M, K)) as f64;
    let mq6_bytes_launch = (K_TOP * rdna_compute::profile::gemv_hfq6g256_bytes(M, K)) as f64;

    let n: usize = std::env::var("HIPFIRE_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    // closures: one launch of each config against a chosen topk buffer
    macro_rules! launch_mq4 {
        ($tk:expr) => {
            gpu.gemv_hfq4g256_moe_gate_up_k8_indexed(
                &expert_ptrs_mq4,
                $tk,
                &xr,
                &gate_batch,
                &up_batch,
                M,
                K,
                K_TOP,
            )
            .expect("mq4 idx")
        };
    }
    macro_rules! launch_mq6 {
        ($tk:expr) => {
            gpu.gemv_hfq6g256_moe_gate_up_k8_indexed(
                &expert_ptrs_mq6,
                $tk,
                &xr,
                &gate_batch,
                &up_batch,
                M,
                K,
                K_TOP,
            )
            .expect("mq6 idx")
        };
    }
    macro_rules! launch_mixed {
        ($tk:expr) => {
            gpu.gemv_mixed_moe_gate_up_k8_indexed_batched(
                &expert_ptrs_mq4,
                &tags_all_mq4,
                $tk,
                &xr,
                &gate_batch,
                &up_batch,
                M,
                K,
                K_TOP,
                1,
            )
            .expect("mixed idx")
        };
    }

    // warm (JIT all three + fill caches)
    for _ in 0..30 {
        launch_mq4!(topk_hot);
        launch_mq6!(topk_hot);
        launch_mixed!(topk_hot);
    }
    gpu.hip.device_synchronize().expect("warm sync");

    // measure: returns us/launch
    macro_rules! time_hot {
        ($launch:ident) => {{
            let t0 = Instant::now();
            for _ in 0..n {
                $launch!(topk_hot);
            }
            gpu.hip.device_synchronize().expect("sync");
            t0.elapsed().as_secs_f64() * 1e6 / n as f64
        }};
    }
    macro_rules! time_cold {
        ($launch:ident) => {{
            let t0 = Instant::now();
            for i in 0..n {
                $launch!(&topk_sets[i % topk_sets.len()]);
            }
            gpu.hip.device_synchronize().expect("sync");
            t0.elapsed().as_secs_f64() * 1e6 / n as f64
        }};
    }

    // RUN1 dropped as warmup jitter, RUN2 reported.
    let _ = (time_hot!(launch_mq4), time_cold!(launch_mq4));
    let mq4_hot = time_hot!(launch_mq4);
    let mq4_cold = time_cold!(launch_mq4);
    let mq6_hot = time_hot!(launch_mq6);
    let mq6_cold = time_cold!(launch_mq6);
    let mix_hot = time_hot!(launch_mixed);
    let mix_cold = time_cold!(launch_mixed);

    let gbps = |bytes: f64, us: f64| bytes / us / 1e3; // B/us = GB/s
    let cell = |us: f64, bytes: f64| format!("{:7.1}us  {:5.0}GB/s", us, gbps(bytes, us));

    eprintln!();
    eprintln!("--- per-launch (gate_up only), N={n}, GPU-throughput (1 sync / N launches) ---");
    eprintln!(
        "  {:<22}  {:>22}  {:>22}",
        "config", "HOT (L3-resident 8)", "COLD (256-expert WS)"
    );
    eprintln!("  {}", "-".repeat(70));
    eprintln!(
        "  {:<22}  {:>22}  {:>22}",
        "(a) mq4-uniform",
        cell(mq4_hot, mq4_bytes_launch),
        cell(mq4_cold, mq4_bytes_launch)
    );
    eprintln!(
        "  {:<22}  {:>22}  {:>22}",
        "(b) mq6-uniform",
        cell(mq6_hot, mq6_bytes_launch),
        cell(mq6_cold, mq6_bytes_launch)
    );
    eprintln!(
        "  {:<22}  {:>22}  {:>22}",
        "(c) merged@MQ4 data",
        cell(mix_hot, mq4_bytes_launch),
        cell(mix_cold, mq4_bytes_launch)
    );
    eprintln!();

    // ---- H2: kernel-fixable overhead (c)-(a) on identical mq4 data ----
    let h2_hot = (mix_hot - mq4_hot) / mq4_hot * 100.0;
    let h2_cold = (mix_cold - mq4_cold) / mq4_cold * 100.0;
    eprintln!("H2 (merged kernel overhead vs uniform mq4, SAME data, all-MQ4 tags):");
    eprintln!(
        "   HOT  (c)/(a) = {:+.1}%   COLD (c)/(a) = {:+.1}%",
        h2_hot, h2_cold
    );
    eprintln!("   → this is the pure 'restore the quad-unroll' upside at the real indexed grid.");
    eprintln!();

    // ---- byte-transfer check: does mq6 +47% bytes → proportional slowdown? ----
    let byte_ratio = mq6_bytes_launch / mq4_bytes_launch;
    eprintln!(
        "Byte-transfer (mq6 vs mq4 uniform): bytes ×{:.2}",
        byte_ratio
    );
    eprintln!("   HOT  us ×{:.2}   COLD us ×{:.2}   (==byte_ratio → BW-bound; <ratio → compute/launch-bound)",
        mq6_hot / mq4_hot, mq6_cold / mq4_cold);
    eprintln!();

    // ---- FRACTION: routed gate_up share of the decode token ----
    // down ≈ 0.5× gate_up bytes (per-expert: down 557056 B vs gate_up 1114112 B);
    // routed/token ≈ (gu + down) × layers ≈ gu × 1.5 × 40 = gu × 60. Labeled est.
    let frac = |gu_us: f64| {
        let gu_tok = gu_us * N_MOE_LAYERS as f64;
        let routed_tok = gu_tok * 1.5; // + down (~0.5× gate_up)
        (gu_tok, routed_tok, routed_tok / FULL_DECODE_US * 100.0)
    };
    eprintln!(
        "FRACTION of the {:.0} us/token decode (mq4@150 anchor), COLD (realistic) rate:",
        FULL_DECODE_US
    );
    for (name, gu) in [
        ("mq4", mq4_cold),
        ("mq6", mq6_cold),
        ("merged@MQ4", mix_cold),
    ] {
        let (gu_tok, routed_tok, pct) = frac(gu);
        eprintln!(
            "   {:<12} gate_up ×40 = {:6.0}us  +down(est) = {:6.0}us  = {:4.1}% of token",
            name, gu_tok, routed_tok, pct
        );
    }
    eprintln!();
    eprintln!("READ: if routed % is LARGE (~30%+), WS1's byte ceilings (122/102) bind and the");
    eprintln!("graded gap is dominantly format-inherent. If SMALL (~5%), the 150-vs-110 spread");
    eprintln!("lives outside the routed GEMV (shared expert / attn / hipGraph) and the byte model");
    eprintln!(
        "is a proxy — WS2 'capture the 150' would then be a kernel/launch problem, not format."
    );

    // Deterministic output-bit hash: the global and buffer-addressing arms use
    // separate processes/JIT caches but must produce identical gate/up bits.
    launch_mq4!(topk_hot);
    gpu.hip.device_synchronize().expect("hash sync");
    let gate = gpu.download_f32(&gate_batch).expect("download gate");
    let up = gpu.download_f32(&up_batch).expect("download up");
    let mut output_hash = 0xcbf29ce484222325u64;
    for value in gate.iter().chain(up.iter()) {
        for byte in value.to_bits().to_le_bytes() {
            output_hash ^= byte as u64;
            output_hash = output_hash.wrapping_mul(0x100000001b3);
        }
    }
    eprintln!(
        "\nsanity: gate_batch[0] = {:.5} (finite={}) output_fnv64={output_hash:016x}",
        gate[0],
        gate.iter().chain(up.iter()).all(|value| value.is_finite())
    );
    drop(keep);
}
