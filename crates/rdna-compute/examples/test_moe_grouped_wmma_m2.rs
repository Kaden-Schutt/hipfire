//! Byte-exact A/B validation: gemm_hfq4g256_moe_grouped_wmma_m2_gfx12
//! against gemm_hfq4g256_moe_grouped_wmma_gfx12 (the base kernel).
//!
//! Both kernels implement the same math (same dequant, same K-step order
//! inside each accumulator). The m2 variant just widens M-coverage per
//! warp to 32 rows; per-element accumulator updates run in identical
//! K-order, so outputs must be byte-identical.
//!
//! GFX12 ONLY. The m2 kernel is registered only as a gfx12 variant.
//! Skips on non-gfx12 archs with a SKIP message.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example test_moe_grouped_wmma_m2

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

/// Build a single MQ4-G256 expert weight matrix [M × K] with deterministic
/// random nibbles/scales. Each row has K/256 groups of 136 bytes:
///   [0..4]   f32 scale
///   [4..8]   f32 zero
///   [8..136] 128 bytes = 256 4-bit nibbles
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
            // Scale: small positive in [0.005, 0.025).
            let sc = 0.005_f32 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 0.020_f32;
            // Zero: small symmetric in [-0.05, 0.05).
            let zp = -0.05_f32 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 0.10_f32;
            buf[off..off + 4].copy_from_slice(&sc.to_le_bytes());
            buf[off + 4..off + 8].copy_from_slice(&zp.to_le_bytes());
            // 128 bytes = 256 nibbles.
            for b in 0..128 {
                let lo = (lcg(&mut s) % 16) as u8;
                let hi = (lcg(&mut s) % 16) as u8;
                buf[off + 8 + b] = lo | (hi << 4);
            }
        }
    }
    buf
}

/// Build an X tensor [N × K] in fp32 with deterministic random values
/// (will be auto-converted to fp16 by the dispatcher).
fn build_x_f32(n: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    let mut out = vec![0f32; n * k];
    for i in 0..n * k {
        // [-1, 1) tight range so fp16 conversion stays well-behaved.
        out[i] = -1.0 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 2.0;
    }
    out
}

fn run_case(label: &str, m: usize, k: usize, m_total: usize, num_experts: usize, seed_w: u32, seed_x: u32) {
    println!("=== {} | M={} K={} m_total={} E={} ===", label, m, k, m_total, num_experts);
    assert!(m % 32 == 0, "M must be a multiple of 32 (m2 stride)");
    assert!(m_total % 16 == 0, "m_total must be a multiple of 16");

    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    if !arch.starts_with("gfx12") {
        println!("  SKIP — arch {} is not gfx12; m2 kernel only registered for gfx12", arch);
        return;
    }

    // Build E experts of identical shape, with distinct random fills.
    let mut expert_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    let mut _expert_tensors: Vec<GpuTensor> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let bytes = build_expert_weight(m, k, seed_w.wrapping_add(e as u32 * 9973));
        let t = upload_u8(&mut gpu, &bytes);
        expert_ptrs.push(t.buf.as_ptr() as u64);
        _expert_tensors.push(t);
    }
    let expert_weight_ptrs = upload_u64(&mut gpu, &expert_ptrs);

    // sorted_slot_index: identity mapping. tile_y → expert (tile_y % E).
    let sorted: Vec<i32> = (0..m_total as i32).collect();
    let sorted_slot_index = upload_i32(&mut gpu, &sorted);
    let tile_ids: Vec<i32> = (0..(m_total / 16))
        .map(|tile_y| (tile_y % num_experts) as i32)
        .collect();
    let expert_tile_ids = upload_i32(&mut gpu, &tile_ids);

    // X: m_total rows × K, identity gather (x_row_div = 1).
    let x_f32 = build_x_f32(m_total, k, seed_x);
    let x_src = upload_f32(&mut gpu, &x_f32);

    let y_base = alloc_f32_zeros(&mut gpu, m_total * m);
    let y_m2 = alloc_f32_zeros(&mut gpu, m_total * m);

    // Run base kernel (env unset).
    std::env::remove_var("HIPFIRE_MOE_GROUPED_M2");
    gpu.gemm_hfq4g256_moe_grouped_wmma_k2(
        &expert_weight_ptrs,
        &expert_tile_ids,
        &sorted_slot_index,
        &x_src,
        &y_base,
        m,
        k,
        1, // x_row_div
        m_total,
        m_total, // x_src_rows
    ).expect("base kernel launch");
    gpu.hip.device_synchronize().expect("sync after base");

    // Run m2 kernel (env set).
    std::env::set_var("HIPFIRE_MOE_GROUPED_M2", "1");
    gpu.gemm_hfq4g256_moe_grouped_wmma_k2(
        &expert_weight_ptrs,
        &expert_tile_ids,
        &sorted_slot_index,
        &x_src,
        &y_m2,
        m,
        k,
        1,
        m_total,
        m_total,
    ).expect("m2 kernel launch");
    gpu.hip.device_synchronize().expect("sync after m2");
    std::env::remove_var("HIPFIRE_MOE_GROUPED_M2");

    let y_base_v = download_f32(&gpu, &y_base, m_total * m);
    let y_m2_v = download_f32(&gpu, &y_m2, m_total * m);

    // The two kernels should be byte-identical (same math, same K-order
    // per accumulator). Allow tiny ULP-level slop in case the compiler
    // reschedules FMA chains; >1e-4 abs / 1e-3 rel indicates a bug.
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut argmax_abs = 0usize;
    for (i, (a, b)) in y_base_v.iter().zip(y_m2_v.iter()).enumerate() {
        let d = (a - b).abs();
        let r = if a.abs() > 1e-6 { d / a.abs() } else { d };
        if d > max_abs { max_abs = d; argmax_abs = i; }
        if r > max_rel { max_rel = r; }
    }
    let base_sample = &y_base_v[argmax_abs];
    let m2_sample = &y_m2_v[argmax_abs];
    println!(
        "  max_abs_diff = {:.6e} (at {}: base={:.6}, m2={:.6})",
        max_abs, argmax_abs, base_sample, m2_sample
    );
    println!("  max_rel_diff = {:.6e}", max_rel);
    if max_abs > 1e-4 || max_rel > 1e-3 {
        println!("  FAIL — exceeds ULP-level slop");
        std::process::exit(1);
    } else {
        println!("  PASS");
    }
}

fn main() {
    // Toy: 1 expert, single tile_y, M=32 / K=256 / m_total=16.
    run_case("toy", 32, 256, 16, 1, 0xDEAD_BEEF, 0xCAFE_BABE);
    // Small: 2 experts, 2 tile_y, M=64 / K=512 / m_total=32.
    run_case("small", 64, 512, 32, 2, 0x1234_5678, 0x8765_4321);
    // Medium: 4 experts, 4 tile_y, M=128 / K=1024 / m_total=64.
    run_case("medium", 128, 1024, 64, 4, 0x0F0F_0F0F, 0xF0F0_F0F0);
    // A3B-shaped slice: M=768 (mirrors per-expert gate_up/2) , K=7168, m_total=256.
    run_case("a3b-slice", 768, 7168, 256, 8, 0x4242_4242, 0x2424_2424);

    println!("\nAll cases PASS.");
}
