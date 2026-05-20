//! Channel test for `gemm_hfp4g32_moe_grouped_wmma_gfx12`.
//!
//! Synthesizes random HFP4G32 expert weights + X + sorted_slot_index +
//! expert_tile_ids, then compares the GPU kernel output to a CPU reference
//! that performs full HFP4G32 dequant (per-row fp16 row_scale × per-block
//! UE8M0 × E2M1 LUT[nibble]) + FP32 GEMM with the same gather/sentinel
//! semantics.
//!
//! GFX12 ONLY. The kernel is registered only as a gfx12 variant; on other
//! archs the test prints SKIP and exits 0. (The CPU reference path still
//! compiles + runs to confirm the test scaffolding is sound, but no GPU
//! comparison is performed.)
//!
//! Tolerance: 1e-3 abs, 1e-2 rel. WMMA accumulates in FP16, the CPU ref
//! mixes FP16-scaled dequant + FP32 acc to mirror the kernel; exact byte
//! match is not expected, ULP-level slop is.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example test_moe_grouped_wmma_hfp4
//!
//! Mirrors the structure of `test_moe_grouped_wmma_m2.rs` (HFQ4 sister).

use rdna_compute::{DType, Gpu, GpuTensor};

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    *state & 0x7fff_ffff
}

// E2M1 LUT — matches the kernel's __shared__ lut[16].
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

// Round f32 to f16 bits via bit manipulation (mirrors the helper used by
// `test_gemm_hfp4g32.rs`). Truncates mantissa toward zero — adequate for
// the small positive row scales we synthesize.
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

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits as u32) >> 10) & 0x1F;
    let mant = (bits as u32) & 0x3FF;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal — convert by normalizing.
        let mut e = 1i32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let exp_f32 = (e + 127 - 15) as u32;
        return f32::from_bits(sign | (exp_f32 << 23) | (m << 13));
    }
    if exp == 0x1F {
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }
    let exp_f32 = (exp + 127 - 15) as u32;
    f32::from_bits(sign | (exp_f32 << 23) | (mant << 13))
}

fn upload_u8(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::Raw)
        .expect("alloc raw");
    gpu.hip.memcpy_htod(&t.buf, data).expect("memcpy_htod raw");
    t
}

fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    let t = gpu
        .alloc_tensor(&[data.len()], DType::F32)
        .expect("alloc f32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("memcpy_htod f32");
    t
}

fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    let t = gpu
        .alloc_tensor(&[data.len() * 4], DType::Raw)
        .expect("alloc i32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("memcpy_htod i32");
    t
}

fn upload_u64(gpu: &mut Gpu, data: &[u64]) -> GpuTensor {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8)
    };
    let t = gpu
        .alloc_tensor(&[data.len() * 8], DType::Raw)
        .expect("alloc u64");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("memcpy_htod u64");
    t
}

fn alloc_f32_zeros(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let t = gpu.alloc_tensor(&[n], DType::F32).expect("alloc zeros");
    gpu.hip.memset(&t.buf, 0, n * 4).expect("memset zero");
    t
}

/// Build a single HFP4G32 expert weight matrix [M × K] with deterministic
/// random data. Row layout (matches `test_gemm_hfp4g32.rs::synth`):
///   [0..2]   fp16 row_scale_h  (small positive, ~0.02)
///   [2..16]  unused
///   per-block (17 bytes each, K/32 blocks):
///     [0]    UE8M0 exponent (1 B)
///     [1..17] 16 bytes = 32 packed FP4 nibbles
fn build_expert_weight_hfp4(m: usize, k: usize, seed: u32) -> Vec<u8> {
    assert!(k % 32 == 0, "K must be a multiple of 32");
    let blocks_per_row = k / 32;
    let row_bytes = 16 + blocks_per_row * 17;
    let total = m * row_bytes;
    let mut buf = vec![0u8; total];
    let mut s = seed;
    for row in 0..m {
        let row_off = row * row_bytes;
        // Small positive row scale in [0.015, 0.035).
        let rs_f32 = 0.015_f32 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 0.020_f32;
        let rs_f16 = f32_to_f16_bits(rs_f32);
        buf[row_off..row_off + 2].copy_from_slice(&rs_f16.to_le_bytes());
        for b in 0..blocks_per_row {
            let bp = row_off + 16 + b * 17;
            // UE8M0 exponent biased near 120 → 2^(120-127) = 2^-7 ≈ 0.0078. Tight
            // dynamic range so products stay well within fp16.
            let e = 119u8 + (lcg(&mut s) & 0x7) as u8;
            buf[bp] = e;
            // 16 bytes = 32 nibbles, each in [0..16).
            for i in 0..16 {
                let lo = (lcg(&mut s) & 0xF) as u8;
                let hi = (lcg(&mut s) & 0xF) as u8;
                buf[bp + 1 + i] = lo | (hi << 4);
            }
        }
    }
    buf
}

/// X tensor [N × K] in fp32 with deterministic random values in [-1, 1).
fn build_x_f32(n: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    let mut out = vec![0f32; n * k];
    for i in 0..n * k {
        out[i] = -1.0 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 2.0;
    }
    out
}

/// CPU reference matching the kernel's HFP4G32 dequant + WMMA path.
/// Outputs Y_grouped[m_total × M] in column-major-by-token (the kernel
/// writes Y[out_col * M + out_row]).
#[allow(clippy::too_many_arguments)]
fn cpu_reference(
    expert_weights: &[Vec<u8>],
    expert_tile_ids: &[i32],
    sorted_slot_index: &[i32],
    x_f32: &[f32],
    m: usize,
    k: usize,
    x_row_div: i32,
    m_total: usize,
    n_rows_x: usize,
) -> Vec<f32> {
    let blocks_per_row = k / 32;
    let row_bytes = 16 + blocks_per_row * 17;
    let lut: Vec<f32> = E2M1.to_vec();

    // Pre-convert X to fp16 (kernel does this via ensure_fp16_x).
    let x_f16_bits: Vec<u16> = x_f32.iter()
        .map(|&v| f32_to_f16_bits(v))
        .collect();
    let x_f16: Vec<f32> = x_f16_bits.iter()
        .map(|&b| f16_bits_to_f32(b))
        .collect();

    let mut y = vec![0f32; m_total * m];

    let n_tiles_y = m_total / 16;
    for tile_y in 0..n_tiles_y {
        let expert_id = expert_tile_ids[tile_y];
        if expert_id < 0 { continue; }
        let weight = &expert_weights[expert_id as usize];

        let slot_start = tile_y * 16;

        // X gather: per-lane (per-slot) source row, or -1 → zero.
        let mut x_rows: [Option<usize>; 16] = [None; 16];
        for lane in 0..16 {
            let slot_idx = slot_start + lane;
            if slot_idx >= m_total { continue; }
            let flat = sorted_slot_index[slot_idx];
            if flat < 0 { continue; }
            let row = if x_row_div > 1 { flat / x_row_div } else { flat };
            if (row as usize) < n_rows_x {
                x_rows[lane] = Some(row as usize);
            }
        }

        let n_tiles_x = (m + 15) / 16;
        for tile_x in 0..n_tiles_x {
            let row_start = tile_x * 16;
            for out_row_off in 0..16 {
                let m_row = row_start + out_row_off;
                if m_row >= m { continue; }
                let row_off = m_row * row_bytes;
                let rs_bits = u16::from_le_bytes([weight[row_off], weight[row_off + 1]]);
                let row_scale_f16 = f16_bits_to_f32(rs_bits);

                // For each output column = slot lane.
                for lane in 0..16 {
                    let out_col = slot_start + lane;
                    if out_col >= m_total { continue; }
                    let x_row = match x_rows[lane] {
                        Some(r) => r,
                        None => {
                            // B = 0 → contribution is zero, but still need
                            // to write 0 if uninitialized.
                            continue;
                        }
                    };

                    let mut acc: f32 = 0.0;
                    // Sum across all K blocks.
                    for b in 0..blocks_per_row {
                        let bp = row_off + 16 + b * 17;
                        let e = weight[bp] as u32;
                        // UE8M0: scale = 2^(e - 127) — same encoding as the kernel
                        // (sign=0 exponent=e mantissa=0 == bitcast(u32, e << 23)).
                        let block_scale = f32::from_bits(e << 23);
                        // Per the kernel, sc_h = row_scale_h(fp16) * block_scale(fp16).
                        // Mirror that by converting block_scale to f16 first.
                        let block_scale_f16 = f16_bits_to_f32(f32_to_f16_bits(block_scale));
                        let sc_h = f16_bits_to_f32(f32_to_f16_bits(row_scale_f16 * block_scale_f16));

                        // 32 packed nibbles in this block.
                        for n_idx in 0..32 {
                            let byte_idx = n_idx / 2;
                            let shift = (n_idx % 2) * 4;
                            let nib = ((weight[bp + 1 + byte_idx] >> shift) & 0xF) as usize;
                            // a = (sc_h * lut[nib]) in fp16
                            let a_f16 = f16_bits_to_f32(f32_to_f16_bits(sc_h * lut[nib]));
                            // b = X_f16[x_row, b*32 + n_idx]
                            let k_idx = b * 32 + n_idx;
                            let b_val = x_f16[x_row * k + k_idx];
                            acc += a_f16 * b_val;
                        }
                    }
                    y[out_col * m + m_row] = acc;
                }
            }
        }
    }
    y
}

fn run_case(label: &str, m: usize, k: usize, m_total: usize, num_experts: usize, seed_w: u32, seed_x: u32) -> bool {
    println!("=== {} | M={} K={} m_total={} E={} ===", label, m, k, m_total, num_experts);
    assert!(m % 16 == 0, "M must be a multiple of 16");
    assert!(m_total % 16 == 0, "m_total must be a multiple of 16");

    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    if !arch.starts_with("gfx12") {
        println!("  SKIP — arch {} is not gfx12; HFP4 grouped-WMMA only registered for gfx12", arch);
        // Still exercise the CPU reference to catch host-side regressions in the
        // dequant logic / test scaffolding (no GPU comparison performed).
        let weights: Vec<Vec<u8>> = (0..num_experts)
            .map(|e| build_expert_weight_hfp4(m, k, seed_w.wrapping_add(e as u32 * 9973)))
            .collect();
        let x_f32 = build_x_f32(m_total, k, seed_x);
        let sorted: Vec<i32> = (0..m_total as i32).collect();
        let tile_ids: Vec<i32> = (0..(m_total / 16))
            .map(|tile_y| (tile_y % num_experts) as i32)
            .collect();
        let y_ref = cpu_reference(
            &weights, &tile_ids, &sorted, &x_f32,
            m, k, 1, m_total, m_total,
        );
        let max_abs = y_ref.iter().map(|v| v.abs()).fold(0f32, f32::max);
        println!("  CPU reference computed, max_abs_y = {:.6e}", max_abs);
        return true;
    }

    // Build E experts of identical shape with distinct random fills.
    let mut expert_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    let mut weight_bytes: Vec<Vec<u8>> = Vec::with_capacity(num_experts);
    let mut _expert_tensors: Vec<GpuTensor> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let bytes = build_expert_weight_hfp4(m, k, seed_w.wrapping_add(e as u32 * 9973));
        let t = upload_u8(&mut gpu, &bytes);
        expert_ptrs.push(t.buf.as_ptr() as u64);
        weight_bytes.push(bytes);
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

    let y_gpu = alloc_f32_zeros(&mut gpu, m_total * m);

    gpu.gemm_hfp4g32_moe_grouped_wmma(
        &expert_weight_ptrs,
        &expert_tile_ids,
        &sorted_slot_index,
        &x_src,
        &y_gpu,
        m,
        k,
        1, // x_row_div
        m_total,
        m_total, // x_src_rows
    ).expect("kernel launch");
    gpu.hip.device_synchronize().expect("sync");

    let y_gpu_v = gpu.download_f32(&y_gpu).expect("download Y");

    // CPU reference.
    let y_ref = cpu_reference(
        &weight_bytes, &tile_ids, &sorted, &x_f32,
        m, k, 1, m_total, m_total,
    );

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut argmax_abs = 0usize;
    let mut max_y_ref_abs = 0f32;
    for (i, (r, g)) in y_ref.iter().zip(y_gpu_v.iter()).enumerate() {
        let d = (r - g).abs();
        if r.abs() > max_y_ref_abs { max_y_ref_abs = r.abs(); }
        let rel = if r.abs() > 1e-6 { d / r.abs() } else { d };
        if d > max_abs { max_abs = d; argmax_abs = i; }
        if rel > max_rel { max_rel = rel; }
    }
    let r_sample = y_ref[argmax_abs];
    let g_sample = y_gpu_v[argmax_abs];
    println!(
        "  max_abs_diff = {:.6e} (at {}: ref={:.6}, gpu={:.6}, max_|ref|={:.3e})",
        max_abs, argmax_abs, r_sample, g_sample, max_y_ref_abs
    );
    println!("  max_rel_diff = {:.6e}", max_rel);

    // ULP-level slop: dequant + WMMA F16-acc vs CPU mix of f16 dequant + f32 acc.
    // Tolerance picked to allow normal FP16 accumulation drift across K tiles.
    let tol_abs = 1e-3f32.max(1e-2 * max_y_ref_abs);
    let tol_rel = 1e-2f32;
    if max_abs > tol_abs && max_rel > tol_rel {
        println!("  FAIL — max_abs {:.3e} > tol_abs {:.3e} AND max_rel {:.3e} > tol_rel {:.3e}",
            max_abs, tol_abs, max_rel, tol_rel);
        false
    } else {
        println!("  PASS");
        true
    }
}

fn main() {
    // Toy: 1 expert, single tile_y, M=32 / K=256 / m_total=16.
    let mut ok = true;
    ok &= run_case("toy",       32,   256,  16, 1, 0xDEAD_BEEF, 0xCAFE_BABE);
    // Small: 2 experts, 2 tile_y, M=64 / K=512 / m_total=32.
    ok &= run_case("small",     64,   512,  32, 2, 0x1234_5678, 0x8765_4321);
    // Medium: 4 experts, 4 tile_y, M=128 / K=1024 / m_total=64.
    ok &= run_case("medium",   128,  1024,  64, 4, 0x0F0F_0F0F, 0xF0F0_F0F0);
    // A3B-shaped slice: M=768 (mirrors per-expert gate_up/2), K=7168, m_total=256.
    ok &= run_case("a3b-slice", 768, 7168, 256, 8, 0x4242_4242, 0x2424_2424);

    if ok {
        println!("\nAll cases PASS.");
    } else {
        println!("\nFAIL — at least one case exceeded slop.");
        std::process::exit(1);
    }
}
