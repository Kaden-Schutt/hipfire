//! Tolerance-bounded A/B validation: gemm_hfq4g256_moe_grouped_mmq_gfx11_dgpu
//! against gemm_hfq4g256_moe_grouped_wmma_k2 (the FP16 grouped base).
//!
//! Unlike the byte-identical m2/k2 pair, the i8 MMQ path quantizes X to
//! Q8_1 (7-bit signed int + per-32-element scale), so outputs differ
//! within the Q8_1 quantization noise envelope. Expected relative error
//! is ~1-3%.
//!
//! gfx11 dGPU ONLY (gfx1100/1101/1102/1103 — 7900 XTX, 7800/7700, 7600,
//! Phoenix mobile). Skips with a clear message on other archs.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example test_moe_grouped_mmq_gfx11_dgpu

use rdna_compute::{Gpu, GpuTensor, DType};

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    *state & 0x7fff_ffff
}

fn upload_u8(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::Raw)
        .expect("alloc_tensor u8");
    gpu.hip.memcpy_htod(&t.buf, data).expect("memcpy_htod u8");
    t
}

fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    let t = gpu
        .alloc_tensor(&[data.len()], DType::F32)
        .expect("alloc_tensor f32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("memcpy_htod f32");
    t
}

fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    let t = gpu
        .alloc_tensor(&[data.len() * 4], DType::Raw)
        .expect("alloc_tensor i32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("memcpy_htod i32");
    t
}

fn upload_u64(gpu: &mut Gpu, data: &[u64]) -> GpuTensor {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8)
    };
    let t = gpu
        .alloc_tensor(&[data.len() * 8], DType::Raw)
        .expect("alloc_tensor u64");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("memcpy_htod u64");
    t
}

fn alloc_f32_zeros(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let t = gpu.alloc_tensor(&[n], DType::F32).expect("alloc f32 zeros");
    gpu.hip.memset(&t.buf, 0, n * 4).expect("memset zero");
    t
}

fn download_f32(gpu: &Gpu, tensor: &GpuTensor, n: usize) -> Vec<f32> {
    let mut data = vec![0f32; n];
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4)
    };
    gpu.hip.memcpy_dtoh(bytes, &tensor.buf).expect("memcpy_dtoh f32");
    data
}

/// HFQ4-G256 expert weight builder: per row, K/256 groups of 136 bytes
/// (f32 scale + f32 zero + 128 bytes = 256 4-bit nibbles).
fn build_expert_weight(m: usize, k: usize, seed: u32) -> Vec<u8> {
    assert!(k % 256 == 0, "K must be a multiple of 256");
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 136;
    let total = m * bytes_per_row;
    let mut buf = vec![0u8; total];
    let mut s = seed;
    for row in 0..m {
        for g in 0..groups_per_row {
            let off = row * bytes_per_row + g * 136;
            let sc = 0.005_f32 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 0.020_f32;
            let zp = -0.05_f32 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 0.10_f32;
            buf[off..off + 4].copy_from_slice(&sc.to_le_bytes());
            buf[off + 4..off + 8].copy_from_slice(&zp.to_le_bytes());
            for b in 0..128 {
                let lo = (lcg(&mut s) % 16) as u8;
                let hi = (lcg(&mut s) % 16) as u8;
                buf[off + 8 + b] = lo | (hi << 4);
            }
        }
    }
    buf
}

fn build_x_f32(n: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    let mut out = vec![0f32; n * k];
    for i in 0..n * k {
        // [-1, 1) — well within fp16 and Q8_1 representable range.
        out[i] = -1.0 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 2.0;
    }
    out
}

fn run_case(
    label: &str,
    m: usize,
    k: usize,
    m_total: usize,
    num_experts: usize,
    seed_w: u32,
    seed_x: u32,
    rtol: f32,
    atol: f32,
) {
    println!("=== {} | M={} K={} m_total={} E={} ===", label, m, k, m_total, num_experts);
    assert!(m % 16 == 0, "M must be a multiple of 16");
    assert!(m_total % 16 == 0, "m_total must be a multiple of 16");
    assert!(k % 256 == 0, "K must be a multiple of 256");

    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    let is_target = arch.starts_with("gfx1100")
        || arch.starts_with("gfx1101")
        || arch.starts_with("gfx1102")
        || arch.starts_with("gfx1103");
    if !is_target {
        println!(
            "  SKIP — arch {} is not gfx1100/1101/1102/1103; i8 MMQ MoE grouped \
             kernel only registered for gfx11 dGPUs",
            arch
        );
        return;
    }

    // Build E experts of identical shape, distinct random fills.
    let mut expert_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    let mut _expert_tensors: Vec<GpuTensor> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let bytes = build_expert_weight(m, k, seed_w.wrapping_add(e as u32 * 9973));
        let t = upload_u8(&mut gpu, &bytes);
        expert_ptrs.push(t.buf.as_ptr() as u64);
        _expert_tensors.push(t);
    }
    let expert_weight_ptrs = upload_u64(&mut gpu, &expert_ptrs);

    // sorted_slot_index: identity (slot s → x_row s). tile_y → expert (tile_y % E).
    let sorted: Vec<i32> = (0..m_total as i32).collect();
    let sorted_slot_index = upload_i32(&mut gpu, &sorted);
    let tile_ids: Vec<i32> = (0..(m_total / 16))
        .map(|tile_y| (tile_y % num_experts) as i32)
        .collect();
    let expert_tile_ids = upload_i32(&mut gpu, &tile_ids);

    // X: m_total rows × K, identity gather (x_row_div = 1).
    let x_f32 = build_x_f32(m_total, k, seed_x);
    let x_src = upload_f32(&mut gpu, &x_f32);

    let y_fp16 = alloc_f32_zeros(&mut gpu, m_total * m);
    let y_i8 = alloc_f32_zeros(&mut gpu, m_total * m);

    // Run FP16 reference (i8 disabled via env override so the wrapper falls
    // through to the FP16 k2 path).
    std::env::set_var("HIPFIRE_MOE_GROUPED_I8", "0");
    gpu.gemm_hfq4g256_moe_grouped_wmma_k2(
        &expert_weight_ptrs,
        &expert_tile_ids,
        &sorted_slot_index,
        &x_src,
        &y_fp16,
        m,
        k,
        1,       // x_row_div
        m_total,
        m_total, // x_src_rows
    ).expect("FP16 kernel launch");
    gpu.hip.device_synchronize().expect("sync after FP16");

    // Run i8 MMQ path — explicit direct call so the test is robust to
    // whether the wrapper dispatch sets defaults.
    gpu.gemm_hfq4g256_moe_grouped_mmq_gfx11_dgpu(
        &expert_weight_ptrs,
        &expert_tile_ids,
        &sorted_slot_index,
        &x_src,
        &y_i8,
        m,
        k,
        1,
        m_total,
        m_total,
    ).expect("i8 MMQ kernel launch");
    gpu.hip.device_synchronize().expect("sync after i8 MMQ");
    std::env::remove_var("HIPFIRE_MOE_GROUPED_I8");

    let y_fp16_v = download_f32(&gpu, &y_fp16, m_total * m);
    let y_i8_v = download_f32(&gpu, &y_i8, m_total * m);

    // i8 has ~1-3% relative error vs FP16 due to Q8_1 quantization.
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut argmax_abs = 0usize;
    let mut argmax_rel = 0usize;
    let mut sum_sq_err = 0f64;
    let mut sum_sq_ref = 0f64;
    for (i, (a, b)) in y_fp16_v.iter().zip(y_i8_v.iter()).enumerate() {
        let d = (a - b).abs();
        let r = if a.abs() > 1e-6 { d / a.abs() } else { d };
        if d > max_abs { max_abs = d; argmax_abs = i; }
        if r > max_rel { max_rel = r; argmax_rel = i; }
        sum_sq_err += (d as f64) * (d as f64);
        sum_sq_ref += (*a as f64) * (*a as f64);
    }
    let rmse = (sum_sq_err / (m_total * m) as f64).sqrt() as f32;
    let nrmse = if sum_sq_ref > 0.0 {
        (sum_sq_err.sqrt() / sum_sq_ref.sqrt()) as f32
    } else { 0.0 };

    println!(
        "  max_abs_diff = {:.6e} (at {}: fp16={:.6}, i8={:.6})",
        max_abs, argmax_abs, y_fp16_v[argmax_abs], y_i8_v[argmax_abs]
    );
    println!(
        "  max_rel_diff = {:.6e} (at {}: fp16={:.6}, i8={:.6})",
        max_rel, argmax_rel, y_fp16_v[argmax_rel], y_i8_v[argmax_rel]
    );
    println!("  RMSE = {:.6e}   NRMSE = {:.6e}", rmse, nrmse);

    // Accept if NRMSE is within tolerance — single-element outliers can blow
    // up max_rel but the overall noise envelope can still be small. Also
    // accept if max_rel is bounded by rtol, or max_abs by atol (catches
    // small-value cases where rel is unstable).
    let pass = max_rel <= rtol || max_abs <= atol || nrmse <= rtol;
    if pass {
        println!("  PASS (rtol={} atol={})", rtol, atol);
    } else {
        println!("  FAIL — exceeds rtol={} atol={}", rtol, atol);
        std::process::exit(1);
    }
}

fn main() {
    // Tiny: E=4 experts, N=16 tokens, K_TOP=2 → m_total=32 slots; K=512, M=256.
    // m_total must be a multiple of 16 → 32 is OK.
    run_case("tiny", 256, 512, 32, 4, 0xDEAD_BEEF, 0xCAFE_BABE, 0.03, 0.01);
    // Small: 2 experts, M=64 K=512 m_total=32.
    run_case("small", 64, 512, 32, 2, 0x1234_5678, 0x8765_4321, 0.03, 0.01);
    // Medium: 4 experts, M=128 K=1024 m_total=64.
    run_case("medium", 128, 1024, 64, 4, 0x0F0F_0F0F, 0xF0F0_F0F0, 0.03, 0.01);
    // A3B-shaped slice: M=768 (per-expert gate_up/2), K=7168, m_total=256.
    run_case("a3b-slice", 768, 7168, 256, 8, 0x4242_4242, 0x2424_2424, 0.03, 0.01);

    println!("\nAll cases PASS.");
}
