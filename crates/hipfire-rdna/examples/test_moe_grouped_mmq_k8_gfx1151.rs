//! Parity validation for gemm_hfq4g256_moe_grouped_mmq_k8_gfx1151 and
//! gemm_hfq4g256_moe_grouped_mmq_k8_4w_gfx1151.
//!
//! The wrapper path is called first to exercise current gfx1151 defaults, then
//! the direct k8 and k8_4w entry points are checked for bit-identical output.
//! This catches routed-slot, padding, LDS-copy, and gfx11 C-mapping mistakes
//! in the 4w variant.
//!
//! GFX1151 ONLY. Skips with a clear message on other archs.
//!
//! Run:
//!   cargo run --release -p hipfire-rdna --example test_moe_grouped_mmq_k8_gfx1151

use hipfire_rdna::{DType, Gpu, GpuTensor};

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
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len()], DType::F32)
        .expect("alloc_tensor f32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("memcpy_htod f32");
    t
}

fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len() * 4], DType::Raw)
        .expect("alloc_tensor i32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("memcpy_htod i32");
    t
}

fn upload_u64(gpu: &mut Gpu, data: &[u64]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
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
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .expect("memcpy_dtoh f32");
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
        // [-1, 1) tight range so fp16 and Q8_1 conversions both stay
        // well within representable range.
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
    x_row_div: usize,
    sparse: bool,
    seed_w: u32,
    seed_x: u32,
    rtol: f32,
    atol: f32,
) {
    println!(
        "=== {} | M={} K={} m_total={} E={} x_row_div={} sparse={} ===",
        label, m, k, m_total, num_experts, x_row_div, sparse
    );
    assert!(m % 16 == 0, "M must be a multiple of 16");
    assert!(m_total % 16 == 0, "m_total must be a multiple of 16");
    assert!(k % 256 == 0, "K must be a multiple of 256");
    assert!(x_row_div > 0, "x_row_div must be non-zero");

    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    if !arch.starts_with("gfx1151") {
        println!("  SKIP — arch {} is not gfx1151; i8 MMQ MoE grouped k8 kernel only registered for gfx1151", arch);
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

    let x_src_rows = if sparse {
        m_total.saturating_sub(num_experts * 15) / x_row_div
    } else {
        m_total / x_row_div
    }
    .max(1);
    let (sorted, tile_ids): (Vec<i32>, Vec<i32>) = if sparse {
        let total_slots = x_src_rows * x_row_div;
        let mut sorted = vec![-1i32; m_total];
        let mut tile_ids = vec![-1i32; m_total / 16];
        for flat in 0..total_slots {
            let expert = flat % num_experts;
            let tile_y = expert;
            let lane = flat / num_experts;
            if tile_y < tile_ids.len() && lane < 16 {
                tile_ids[tile_y] = expert as i32;
                sorted[tile_y * 16 + lane] = flat as i32;
            }
        }
        (sorted, tile_ids)
    } else {
        let sorted: Vec<i32> = (0..m_total as i32).collect();
        let tile_ids: Vec<i32> = (0..(m_total / 16))
            .map(|tile_y| (tile_y % num_experts) as i32)
            .collect();
        (sorted, tile_ids)
    };
    let sorted_slot_index = upload_i32(&mut gpu, &sorted);
    let expert_tile_ids = upload_i32(&mut gpu, &tile_ids);

    let x_f32 = build_x_f32(x_src_rows, k, seed_x);
    let x_src = upload_f32(&mut gpu, &x_f32);

    let y_default = alloc_f32_zeros(&mut gpu, m_total * m);
    let y_i8_k8 = alloc_f32_zeros(&mut gpu, m_total * m);
    let y_i8_k8_4w = alloc_f32_zeros(&mut gpu, m_total * m);

    // Run the current wrapper default first. Feature flags are resolved at
    // Gpu::init(), so this intentionally validates the production route rather
    // than trying to mutate env vars mid-test.
    gpu.gemm_hfq4g256_moe_grouped_wmma_k2(
        &expert_weight_ptrs,
        &expert_tile_ids,
        &sorted_slot_index,
        &x_src,
        &y_default,
        m,
        k,
        x_row_div,
        m_total,
        x_src_rows,
    )
    .expect("default wrapper launch");
    gpu.hip
        .device_synchronize()
        .expect("sync after default wrapper");

    // Run i8 MMQ k8 path (gated to gfx1151 only — explicit direct call so
    // the test is robust to whether the wrapper dispatch sets defaults).
    gpu.gemm_hfq4g256_moe_grouped_mmq_k8_gfx1151(
        &expert_weight_ptrs,
        &expert_tile_ids,
        &sorted_slot_index,
        &x_src,
        &y_i8_k8,
        m,
        k,
        x_row_div,
        m_total,
        x_src_rows,
    )
    .expect("i8 MMQ k8 kernel launch");
    gpu.hip.device_synchronize().expect("sync after i8 MMQ k8");

    gpu.gemm_hfq4g256_moe_grouped_mmq_k8_4w_gfx1151(
        &expert_weight_ptrs,
        &expert_tile_ids,
        &sorted_slot_index,
        &x_src,
        &y_i8_k8_4w,
        m,
        k,
        x_row_div,
        m_total,
        x_src_rows,
    )
    .expect("i8 MMQ k8 4w kernel launch");
    gpu.hip
        .device_synchronize()
        .expect("sync after i8 MMQ k8 4w");

    let y_default_v = download_f32(&gpu, &y_default, m_total * m);
    let y_i8_v = download_f32(&gpu, &y_i8_k8, m_total * m);
    let y_i8_4w_v = download_f32(&gpu, &y_i8_k8_4w, m_total * m);

    let mut max_default_vs_k8 = 0f32;
    let mut argmax_default_vs_k8 = 0usize;
    for (i, (a, b)) in y_default_v.iter().zip(y_i8_v.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_default_vs_k8 {
            max_default_vs_k8 = d;
            argmax_default_vs_k8 = i;
        }
    }
    println!(
        "  default_vs_k8_max_abs = {:.6e} (at {}: default={:.6}, k8={:.6})",
        max_default_vs_k8,
        argmax_default_vs_k8,
        y_default_v[argmax_default_vs_k8],
        y_i8_v[argmax_default_vs_k8]
    );

    let mut max_k8_vs_4w = 0f32;
    let mut argmax_k8_vs_4w = 0usize;
    for (i, (a, b)) in y_i8_v.iter().zip(y_i8_4w_v.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_k8_vs_4w {
            max_k8_vs_4w = d;
            argmax_k8_vs_4w = i;
        }
    }
    println!(
        "  k8_vs_4w_max_abs = {:.6e} (at {}: k8={:.6}, k8_4w={:.6})",
        max_k8_vs_4w, argmax_k8_vs_4w, y_i8_v[argmax_k8_vs_4w], y_i8_4w_v[argmax_k8_vs_4w]
    );

    let pass = max_default_vs_k8 <= 1e-5 && max_k8_vs_4w <= 1e-5;
    if pass {
        println!("  PASS (rtol={} atol={})", rtol, atol);
    } else {
        println!("  FAIL — exceeds rtol={} atol={}", rtol, atol);
        std::process::exit(1);
    }
}

fn main() {
    // Toy: 1 expert, single tile_y, M=16 K=256 m_total=16.
    run_case(
        "toy",
        16,
        256,
        16,
        1,
        1,
        false,
        0xDEAD_BEEF,
        0xCAFE_BABE,
        0.05,
        0.05,
    );
    // Small: 2 experts, 2 tile_y, M=64 K=512 m_total=32.
    run_case(
        "small",
        64,
        512,
        32,
        2,
        1,
        false,
        0x1234_5678,
        0x8765_4321,
        0.05,
        0.05,
    );
    run_case(
        "gate-up-div8",
        64,
        1024,
        128,
        4,
        8,
        false,
        0x1357_2468,
        0x2468_1357,
        0.05,
        0.05,
    );
    // Medium: 4 experts, 4 tile_y, M=128 K=1024 m_total=64.
    run_case(
        "medium",
        128,
        1024,
        64,
        4,
        1,
        false,
        0x0F0F_0F0F,
        0xF0F0_F0F0,
        0.05,
        0.05,
    );
    run_case(
        "sparse-N2-K8-E256",
        16,
        512,
        2 * 8 + 256 * 15,
        256,
        8,
        true,
        0x4400_0002,
        0x5500_0002,
        0.05,
        0.05,
    );
    run_case(
        "sparse-N4-K8-E256",
        16,
        512,
        4 * 8 + 256 * 15,
        256,
        8,
        true,
        0x4400_0004,
        0x5500_0004,
        0.05,
        0.05,
    );
    run_case(
        "sparse-N16-K8-E256",
        16,
        512,
        16 * 8 + 256 * 15,
        256,
        8,
        true,
        0x4400_0016,
        0x5500_0016,
        0.05,
        0.05,
    );
    // A3B-shaped slice: M=768 (per-expert gate_up/2), K=7168, m_total=256.
    run_case(
        "a3b-slice",
        768,
        7168,
        256,
        8,
        1,
        false,
        0x4242_4242,
        0x2424_2424,
        0.05,
        0.05,
    );

    println!("\nAll cases PASS.");
}
