//! MoE-down i64-tax probe — reproducible int64 vs FP32 accumulation at the
//! SAME batched-indexed kernel structure, DeepSeek-V4-Flash down shape.
//!
//! Both kernels are grid `[M, K_TOP, N]`, 32-thread blocks, K4 unroll — the
//! ONLY difference is the accumulator: `_residual_scaled_..._batched_k4`
//! accumulates in FP32 (`atomicAdd(float)`), while the `_residual_i64_...`
//! REPRO variant accumulates in int64 fixed-point (S-scaled, order-independent).
//! So `i64_us / f32_us` is the PURE cost of int64 fixed-point reproducibility on
//! this arch (gfx1151 has no native 64-bit ALU — i64 is emulated), isolated from
//! the GEMV-vs-grouped-WMMA structural difference.
//!
//! Why it matters: the D2c TP down uses the i64 kernel (bit-exact cross-rank);
//! single-GPU prefill uses an FP32 down. Unifying both on i64 would remove the
//! tp-vs-single-GPU logit delta but costs this i64 tax. A small tax ⇒ a
//! grouped-i64-WMMA kernel could plausibly reach grouped-FP32 speed; a large tax
//! ⇒ int64 is fundamentally expensive here and the delta is the price of
//! reproducible TP.
//!
//! Run (gfx1151):
//!   cargo run --release -p rdna-compute --example bench_moe_down_i64_vs_f32
//!   HIPFIRE_BENCH_N=500 HIPFIRE_BENCH_BATCH=256 cargo run --release ...

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

// DeepSeek-V4-Flash down shape (from tp_deepseek4 run: hidden=4096, inter=2048).
const M: usize = 4096; // down output = hidden
const K: usize = 2048; // down input  = moe_intermediate
const N_EXP: usize = 256;
const K_TOP: usize = 6; // num_experts_per_tok

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

fn f32_to_f16_le(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    let h: u16 = if exp == 0xff {
        (sign << 15) | (0x1f << 10) | if mant != 0 { 0x200 } else { 0 }
    } else if exp - 127 + 15 < 1 {
        sign << 15
    } else if exp - 127 + 15 > 30 {
        (sign << 15) | (0x1f << 10)
    } else {
        let new_exp = (exp - 127 + 15) as u16;
        (sign << 15) | (new_exp << 10) | ((mant >> 13) as u16)
    };
    h.to_le_bytes()
}

/// Size-correct MQ2-Lloyd expert: groups = K/256, 72 B/group (8 B fp16 codebook
/// [4 entries] + 64 B 2-bit packed indices). The 2-bit indices are random (byte
/// VALUES are timing-irrelevant for this memory-bound GEMV), but the fp16
/// codebook is BENIGN small finite values so the accumulated output stays finite
/// (a plain random codebook decodes NaN/inf and the sanity check trips).
fn synth_mq2g256(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let groups = k / 256;
    let cb: [f32; 4] = [0.01, 0.02, -0.01, -0.02];
    let mut out = vec![0u8; m * groups * 72];
    let mut st = seed;
    for row in 0..m {
        for g in 0..groups {
            let off = (row * groups + g) * 72;
            for (e, &c) in cb.iter().enumerate() {
                out[off + e * 2..off + e * 2 + 2].copy_from_slice(&f32_to_f16_le(c));
            }
            for b in out[off + 8..off + 72].iter_mut() {
                st = st
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *b = (st >> 56) as u8;
            }
        }
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

    let n_batch: usize = std::env::var("HIPFIRE_BENCH_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let n_iter: usize = std::env::var("HIPFIRE_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);

    eprintln!("=== MoE-down i64-tax probe (MQ2-Lloyd, ds4 down shape) ===");
    eprintln!(
        "arch={arch}  M={M} (hidden) K={K} (inter)  n_exp={N_EXP} k_top={K_TOP}  \
         batch={n_batch}  iters={n_iter}"
    );

    // 256 MQ2 expert weight buffers + pointer table.
    let mut keep: Vec<GpuTensor> = Vec::new();
    let mut ptrs: Vec<u64> = Vec::with_capacity(N_EXP);
    for e in 0..N_EXP {
        let t = upload_u8(&mut gpu, &synth_mq2g256(M, K, 0x4D32 ^ e as u64));
        ptrs.push(t.buf.as_ptr() as u64);
        keep.push(t);
    }
    let expert_ptrs = upload_u64(&mut gpu, &ptrs);

    // rot_batch [N × k_top × K], topk_weights [N × k_top].
    let rot = upload_f32(&mut gpu, &make_x(n_batch * K_TOP * K, 0xABCD));
    let wts = upload_f32(&mut gpu, &vec![1.0f32; n_batch * K_TOP]);

    // HOT: every token routes to the SAME k_top experts [0..K_TOP) → tiny
    // working set, L3-resident. COLD: token b routes to a rotating disjoint set
    // so all 256 experts are touched — the realistic prefill regime.
    let idx_hot: Vec<i32> = (0..n_batch)
        .flat_map(|_| (0..K_TOP).map(|j| j as i32))
        .collect();
    let n_sets = N_EXP / K_TOP; // 42 disjoint top-6 sets
    let idx_cold: Vec<i32> = (0..n_batch)
        .flat_map(|b| {
            let s = b % n_sets;
            (0..K_TOP).map(move |j| (s * K_TOP + j) as i32)
        })
        .collect();
    let topk_hot = upload_i32(&mut gpu, &idx_hot);
    let topk_cold = upload_i32(&mut gpu, &idx_cold);

    // Outputs: FP32 residual and int64 residual, both [N × M].
    let x_res_f32 = gpu
        .alloc_tensor(&[n_batch * M], DType::F32)
        .expect("f32 out");
    gpu.hip
        .memset(&x_res_f32.buf, 0, n_batch * M * 4)
        .expect("memset f32");
    let res_i64 = gpu
        .alloc_tensor(&[n_batch * M * 8], DType::Raw)
        .expect("i64 out");
    gpu.hip
        .memset(&res_i64.buf, 0, n_batch * M * 8)
        .expect("memset i64");

    macro_rules! launch_f32 {
        ($tk:expr) => {
            gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched_k4(
                &expert_ptrs,
                $tk,
                &wts,
                &rot,
                &x_res_f32,
                M,
                K,
                K_TOP,
                n_batch,
            )
            .expect("f32 down")
        };
    }
    macro_rules! launch_i64 {
        ($tk:expr) => {
            gpu.moe_down_mq2g256_lloyd_residual_i64_indexed_batched(
                &expert_ptrs,
                $tk,
                &wts,
                &rot,
                &res_i64,
                M,
                K,
                K_TOP,
                n_batch,
            )
            .expect("i64 down")
        };
    }

    // Warm: JIT both kernels + fill caches.
    for _ in 0..20 {
        launch_f32!(&topk_hot);
        launch_i64!(&topk_hot);
    }
    gpu.hip.device_synchronize().expect("warm sync");

    macro_rules! time {
        ($launch:ident, $tk:expr) => {{
            let t0 = Instant::now();
            for _ in 0..n_iter {
                $launch!($tk);
            }
            gpu.hip.device_synchronize().expect("sync");
            t0.elapsed().as_secs_f64() * 1e6 / n_iter as f64
        }};
    }

    // RUN1 dropped as warmup jitter, RUN2 reported (per perf methodology).
    let _ = (time!(launch_f32, &topk_hot), time!(launch_i64, &topk_hot));
    let f32_hot = time!(launch_f32, &topk_hot);
    let i64_hot = time!(launch_i64, &topk_hot);
    let f32_cold = time!(launch_f32, &topk_cold);
    let i64_cold = time!(launch_i64, &topk_cold);

    eprintln!();
    eprintln!("--- us/launch (GPU-throughput, 1 sync / {n_iter} launches) ---");
    eprintln!(
        "  {:<26}  {:>12}  {:>12}",
        "config", "HOT (8 exp)", "COLD (256 exp)"
    );
    eprintln!("  {}", "-".repeat(54));
    eprintln!(
        "  {:<26}  {:>10.1}us  {:>10.1}us",
        "f32 indexed-batched (k4)", f32_hot, f32_cold
    );
    eprintln!(
        "  {:<26}  {:>10.1}us  {:>10.1}us",
        "i64 indexed-batched (repro)", i64_hot, i64_cold
    );
    eprintln!();
    eprintln!(
        "  i64 TAX (i64/f32):  HOT ×{:.3}  COLD ×{:.3}",
        i64_hot / f32_hot,
        i64_cold / f32_cold
    );
    eprintln!();
    eprintln!("READ: COLD is the realistic prefill regime. Tax ≈1.0 ⇒ int64 fixed-point is");
    eprintln!("effectively free at this structure (a grouped-i64-WMMA kernel could reach");
    eprintln!("grouped-FP32 speed → no-delta unification is cheap). Tax >>1.0 ⇒ int64 is");
    eprintln!("fundamentally costly on this arch → keep the tp-vs-single-GPU delta.");

    // sanity (gross-setup guard)
    let s = {
        let mut v = vec![0f32; 1];
        let bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, 4) };
        gpu.hip.memcpy_dtoh(bytes, &x_res_f32.buf).expect("dtoh");
        v[0]
    };
    eprintln!("\nsanity: x_res_f32[0] = {s:.4} (finite={})", s.is_finite());
    drop(keep);
}
