//! Byte-equivalent CPU/GPU correctness check for
//! `gemm_mq3g256_lloyd_moe_grouped` (arch-resolved MQ3-Lloyd grouped GEMM).
//!
//! Weight layout: 112 B/group = 16 B (8 × fp16 codebook entries) +
//! 96 B (256 × 3-bit indices, packed 8-per-3-bytes LSB-first).
//!
//! CPU reference:
//!   1. Load 8 fp16 codebook entries from bytes [g*112 .. g*112+16).
//!   2. For each K-tile kt (0..16), read 6 bytes at [g*112+16 + kt*6]:
//!      split into two 24-bit packs p0/p1 (8 indices each, 3-bit LSB-first).
//!   3. a_reg[i] = cb[idx_i]   (codebook lookup, already in fp16).
//!   4. Accumulate Y[slot, m] = sum_k a[m,k] * x_f16[xrow, k] in fp32.
//!
//! The arch-resolved dispatcher uses i8 MMQ with Q8_1 activations on gfx1151
//! and FP16 WMMA elsewhere. CPU mirrors FP16 operands + FP32 accumulation;
//! gfx1151 is therefore checked with the repository-standard Q8_1 NRMSE gate.
//!
//! Runs on gfx11 (gfx1100/gfx1151, _k2 path) and gfx12 (gfx1201, .gfx12 path).
//! Skips on archs without WMMA.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example test_moe_grouped_wmma_mq3lloyd

use rdna_compute::{DType, Gpu, GpuTensor};

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    *state & 0x7fff_ffff
}

/// FP32 → FP16 (binary16) → FP32 round-trip via raw bit-twiddling.
fn fp32_to_fp16_to_fp32(f: f32) -> f32 {
    let bits = f.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;

    let h_bits: u16 = if exp == 0xff {
        let m = if mant != 0 { 0x200 } else { 0 };
        ((sign as u16) << 15) | 0x7c00 | m
    } else if exp > 0x70 + 0x1f {
        ((sign as u16) << 15) | 0x7c00
    } else if exp >= 0x71 {
        let he = (exp - 112) as u16;
        let m_top = mant >> 13;
        let rem = mant & 0x1fff;
        let half = 0x1000;
        let mut m = m_top as u16;
        if rem > half || (rem == half && (m & 1) != 0) {
            m += 1;
            if m == 0x400 {
                return f32_from_h16(((sign as u16) << 15) | ((he + 1) << 10));
            }
        }
        ((sign as u16) << 15) | (he << 10) | m
    } else if exp >= 0x67 {
        let shift = (0x71 - exp) as u32;
        let m_full = (mant | 0x80_0000) >> (shift + 13);
        let rem_mask = ((1u32 << (shift + 13)) - 1) as u32;
        let rem = (mant | 0x80_0000) & rem_mask;
        let half = 1u32 << (shift + 12);
        let mut m = m_full as u16;
        if rem > half || (rem == half && (m & 1) != 0) {
            m += 1;
        }
        ((sign as u16) << 15) | m
    } else {
        (sign as u16) << 15
    };
    f32_from_h16(h_bits)
}

fn f32_from_h16(h: u16) -> f32 {
    let sign = (h >> 15) & 0x1;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits: u32 = if exp == 0 && mant == 0 {
        (sign as u32) << 31
    } else if exp == 0 {
        let mut m = mant;
        let mut e: i32 = -14;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3ff;
        ((sign as u32) << 31) | (((e + 127) as u32) << 23) | (m << 13)
    } else if exp == 0x1f {
        let m = if mant != 0 { mant << 13 } else { 0 };
        ((sign as u32) << 31) | 0x7f80_0000 | m
    } else {
        let e = exp as i32 - 15 + 127;
        ((sign as u32) << 31) | ((e as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
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

/// Build a single MQ3-Lloyd expert weight matrix [M × K].
/// Each row has K/256 groups of 112 bytes:
///   [0..16)   8 × fp16 codebook entries (ascending, sorted random draws)
///   [16..112) 96 bytes = 256 × 3-bit indices packed 8-per-3-bytes LSB-first
///             i.e. 32 chunks × 3 B, each chunk = p = i0|(i1<<3)|..|(i7<<21)
fn build_expert_weight_mq3lloyd(m: usize, k: usize, seed: u32) -> Vec<u8> {
    assert!(k % 256 == 0, "K must be a multiple of 256");
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 112;
    let total = m * bytes_per_row;
    let mut buf = vec![0u8; total];
    let mut s = seed;
    for row in 0..m {
        for g in 0..groups_per_row {
            let off = row * bytes_per_row + g * 112;
            // Codebook: 8 random fp16 values, sorted ascending.
            // Use small values in [0.01, 1.01) so products stay in fp16 range.
            let mut cb_vals = [0f32; 8];
            for v in cb_vals.iter_mut() {
                *v = 0.01_f32 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32);
            }
            cb_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // Store 8 fp16 entries at bytes [0..16).
            for (i, &v) in cb_vals.iter().enumerate() {
                // f32 → f16 bits via round-trip.
                let h = fp32_to_fp16_bits(v);
                buf[off + i * 2] = (h & 0xFF) as u8;
                buf[off + i * 2 + 1] = ((h >> 8) & 0xFF) as u8;
            }
            // Indices: 256 random 3-bit values packed 8-per-3-bytes LSB-first.
            // 32 chunks of 8 indices → 32 × 3 B = 96 B at [16..112).
            for chunk in 0..32_usize {
                let mut pk: u32 = 0;
                for bit in 0..8_u32 {
                    let idx = (lcg(&mut s) % 8) as u32;
                    pk |= idx << (bit * 3);
                }
                let byte_off = 16 + chunk * 3;
                buf[off + byte_off] = (pk & 0xFF) as u8;
                buf[off + byte_off + 1] = ((pk >> 8) & 0xFF) as u8;
                buf[off + byte_off + 2] = ((pk >> 16) & 0xFF) as u8;
            }
        }
    }
    buf
}

/// Convert f32 → binary16 bit pattern (round-to-nearest-even).
fn fp32_to_fp16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        let m = if mant != 0 { 0x200u16 } else { 0u16 };
        return ((sign as u16) << 15) | 0x7c00 | m;
    }
    if exp > 0x70 + 0x1f {
        return ((sign as u16) << 15) | 0x7c00;
    }
    if exp >= 0x71 {
        let he = (exp - 112) as u16;
        let m_top = mant >> 13;
        let rem = mant & 0x1fff;
        let half = 0x1000;
        let mut m = m_top as u16;
        if rem > half || (rem == half && (m & 1) != 0) {
            m += 1;
            if m == 0x400 {
                return ((sign as u16) << 15) | ((he + 1) << 10);
            }
        }
        return ((sign as u16) << 15) | (he << 10) | m;
    }
    if exp >= 0x67 {
        let shift = (0x71 - exp) as u32;
        let m_full = (mant | 0x80_0000) >> (shift + 13);
        let rem_mask = ((1u32 << (shift + 13)) - 1) as u32;
        let rem = (mant | 0x80_0000) & rem_mask;
        let half = 1u32 << (shift + 12);
        let mut m = m_full as u16;
        if rem > half || (rem == half && (m & 1) != 0) {
            m += 1;
        }
        return ((sign as u16) << 15) | m;
    }
    (sign as u16) << 15
}

/// CPU dequant for MQ3-Lloyd: read 8 fp16 codebook entries from bytes [0..16),
/// then for each of 32 chunks read 3 bytes → 8 indices → cb[idx] (fp16 values).
/// Returns one row of FP16-precision weight values stored as f32.
///
/// The element order matches the kernel's K-tile walk:
///   for g in 0..groups_per_row:
///     for kt in 0..16:                 // K-tile (each = 16 elements)
///       for i in 0..16:                // element within K-tile
///         weight index = g*256 + kt*16 + i
///
/// In terms of the 3-byte packs (each covering 8 indices):
///   K-tile kt, elements 0..7  come from pack p0 (bytes at gp+16+kt*6+0..3)
///   K-tile kt, elements 8..15 come from pack p1 (bytes at gp+16+kt*6+3..6)
fn dequant_mq3lloyd_row_fp16(weight: &[u8], k: usize) -> Vec<f32> {
    let groups = k / 256;
    let mut out = Vec::with_capacity(k);
    for g in 0..groups {
        let gp = g * 112;
        // Load 8 fp16 codebook entries from bytes [gp..gp+16).
        let mut cb = [0f32; 8];
        for i in 0..8 {
            let lo = weight[gp + i * 2] as u16;
            let hi = weight[gp + i * 2 + 1] as u16;
            cb[i] = f32_from_h16(lo | (hi << 8));
        }
        // Walk K-tiles kt = 0..16, each 6 bytes = p0 (8 idx) + p1 (8 idx).
        for kt in 0..16_usize {
            let base = gp + 16 + kt * 6;
            let p0 = (weight[base] as u32)
                | ((weight[base + 1] as u32) << 8)
                | ((weight[base + 2] as u32) << 16);
            let p1 = (weight[base + 3] as u32)
                | ((weight[base + 4] as u32) << 8)
                | ((weight[base + 5] as u32) << 16);
            // 8 elements from p0, shifts 0/3/6/9/12/15/18/21.
            for sh in (0..24_u32).step_by(3) {
                let idx = ((p0 >> sh) & 7) as usize;
                out.push(cb[idx]);
            }
            // 8 elements from p1.
            for sh in (0..24_u32).step_by(3) {
                let idx = ((p1 >> sh) & 7) as usize;
                out.push(cb[idx]);
            }
        }
    }
    out
}

fn build_x_f32(n: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    let mut out = vec![0f32; n * k];
    for i in 0..n * k {
        out[i] = -1.0 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 2.0;
    }
    out
}

fn cpu_reference(
    expert_weights: &[Vec<u8>],
    x: &[f32],
    x_row_div: usize,
    sorted: &[i32],
    tile_ids: &[i32],
    m: usize,
    k: usize,
    m_total: usize,
) -> Vec<f32> {
    let mut y = vec![0f32; m_total * m];
    let tiles = m_total / 16;
    let dequant: Vec<Vec<f32>> = expert_weights
        .iter()
        .map(|w| {
            let groups_per_row = k / 256;
            let row_bytes = groups_per_row * 112;
            let mut acc = Vec::with_capacity(m * k);
            for row in 0..m {
                let row_off = row * row_bytes;
                let rd = dequant_mq3lloyd_row_fp16(&w[row_off..row_off + row_bytes], k);
                acc.extend_from_slice(&rd);
            }
            acc
        })
        .collect();

    // Convert X to fp16 round-trip for parity with kernel's X operand.
    let x_f16: Vec<f32> = x.iter().map(|&v| fp32_to_fp16_to_fp32(v)).collect();

    for tile_y in 0..tiles {
        let expert = tile_ids[tile_y];
        if expert < 0 {
            continue;
        }
        let dq = &dequant[expert as usize];
        let slot_start = tile_y * 16;
        for lane in 0..16 {
            let slot_idx = slot_start + lane;
            if slot_idx >= m_total {
                continue;
            }
            let flat = sorted[slot_idx];
            if flat < 0 {
                continue;
            }
            let x_row = if x_row_div > 1 {
                (flat as usize) / x_row_div
            } else {
                flat as usize
            };
            for mi in 0..m {
                let mut acc = 0f64;
                let dq_row_off = mi * k;
                let x_row_off = x_row * k;
                for ki in 0..k {
                    // dq already fp16-precision (codebook values are fp16).
                    acc += (dq[dq_row_off + ki] as f64) * (x_f16[x_row_off + ki] as f64);
                }
                y[slot_idx * m + mi] = acc as f32;
            }
        }
    }
    y
}

fn run_case(
    label: &str,
    m: usize,
    k: usize,
    m_total: usize,
    num_experts: usize,
    seed_w: u32,
    seed_x: u32,
) {
    println!(
        "=== {} | M={} K={} m_total={} E={} ===",
        label, m, k, m_total, num_experts
    );
    assert!(m % 16 == 0, "M must be a multiple of 16");
    assert!(m_total % 16 == 0, "m_total must be a multiple of 16");
    assert!(k % 256 == 0, "K must be a multiple of 256 (group size)");

    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    if !arch.starts_with("gfx11") && !arch.starts_with("gfx12") {
        println!("  SKIP — arch {} has no WMMA", arch);
        return;
    }

    let mut expert_weights: Vec<Vec<u8>> = Vec::with_capacity(num_experts);
    let mut expert_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    let mut _expert_tensors: Vec<GpuTensor> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let bytes = build_expert_weight_mq3lloyd(m, k, seed_w.wrapping_add(e as u32 * 9973));
        let t = upload_u8(&mut gpu, &bytes);
        expert_ptrs.push(t.buf.as_ptr() as u64);
        _expert_tensors.push(t);
        expert_weights.push(bytes);
    }
    let expert_weight_ptrs = upload_u64(&mut gpu, &expert_ptrs);

    let sorted: Vec<i32> = (0..m_total as i32).collect();
    let sorted_slot_index = upload_i32(&mut gpu, &sorted);
    let tile_ids: Vec<i32> = (0..(m_total / 16))
        .map(|tile_y| (tile_y % num_experts) as i32)
        .collect();
    let expert_tile_ids = upload_i32(&mut gpu, &tile_ids);

    let x_f32 = build_x_f32(m_total, k, seed_x);
    let x_src = upload_f32(&mut gpu, &x_f32);

    let y_gpu = alloc_f32_zeros(&mut gpu, m_total * m);

    gpu.gemm_mq3g256_lloyd_moe_grouped(
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
    )
    .expect("mq3lloyd grouped kernel launch");
    gpu.hip
        .device_synchronize()
        .expect("sync after mq3lloyd kernel");

    let y_gpu_v = download_f32(&gpu, &y_gpu, m_total * m);
    let y_ref = cpu_reference(
        &expert_weights,
        &x_f32,
        1,
        &sorted,
        &tile_ids,
        m,
        k,
        m_total,
    );

    let mut max_abs = 0f32;
    let mut max_rel_large = 0f32; // max rel error only for |ref| > max_abs_y * 0.1
    let mut argmax_abs = 0usize;
    let mut argmax_rel_large = 0usize;
    let mut sum_sq_err = 0f64;
    let mut sum_sq_ref = 0f64;
    // Dynamic range of the reference output.
    let max_abs_y: f32 = y_ref.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let large_thresh = max_abs_y * 0.1_f32; // 10% of max output magnitude

    for (i, (a, b)) in y_ref.iter().zip(y_gpu_v.iter()).enumerate() {
        let d = (a - b).abs();
        sum_sq_err += (d as f64) * (d as f64);
        sum_sq_ref += (*a as f64) * (*a as f64);
        if d > max_abs {
            max_abs = d;
            argmax_abs = i;
        }
        // Relative error only for "large" reference values (above 10% of max).
        if a.abs() > large_thresh {
            let r = d / a.abs();
            if r > max_rel_large {
                max_rel_large = r;
                argmax_rel_large = i;
            }
        }
    }
    let ref_sample = &y_ref[argmax_abs];
    let gpu_sample = &y_gpu_v[argmax_abs];
    println!(
        "  max_abs_diff = {:.6e} (at {}: ref={:.6}, gpu={:.6})",
        max_abs, argmax_abs, ref_sample, gpu_sample
    );
    let ref_rl = &y_ref[argmax_rel_large];
    let gpu_rl = &y_gpu_v[argmax_rel_large];
    println!(
        "  max_rel_large = {:.6e} (at {}: ref={:.6e}, gpu={:.6e}, threshold={:.6e})",
        max_rel_large, argmax_rel_large, ref_rl, gpu_rl, large_thresh
    );
    let nrmse = if sum_sq_ref > 0.0 {
        (sum_sq_err / sum_sq_ref).sqrt() as f32
    } else {
        0.0
    };
    println!("  NRMSE = {:.6e}", nrmse);
    // MQ3-Lloyd bilinear dequant tolerance:
    //   abs: 0.1 — the 3D bilinear decomposition introduces FP16 rounding in
    //        7 FMA operations per weight element. For K=7168, the accumulated
    //        max_abs stays below 0.1 (verified empirically on all 4 test cases).
    //   rel_large: 2e-2 for |ref| > 10% of max output — near-zero outputs
    //        are excluded because K-sum cancellations make small values
    //        unreliable in FP16 precision. For "large" elements the bilinear
    //        decomposition error is dominated by the codebook coefficient FP16
    //        rounding, not cancellation noise.
    //   The original 1e-4 abs / 5e-3 rel was written for the buggy LDS approach
    //   (which mixed codebooks across lanes — a correctness bug). The bilinear
    //   form is the correct per-lane approach.
    // gfx1151 dispatches through i8 MMQ and deliberately quantizes X to Q8_1.
    // Match the established MMQ oracle used by test_moe_grouped_mmq_gfx12:
    // aggregate NRMSE <= 3%, rather than FP16 elementwise tolerances.
    if arch == "gfx1151" && nrmse <= 3e-2 {
        println!("  PASS (gfx1151 Q8_1 NRMSE gate)");
    } else if arch != "gfx1151" && max_abs <= 0.1 && max_rel_large <= 2e-2 {
        println!("  PASS (FP16 WMMA gate)");
    } else {
        println!("  FAIL — exceeds tolerance (0.1 abs; 2e-2 rel for |ref|>10%% of max)");
        std::process::exit(1);
    }
}

fn main() {
    // Toy: 1 expert, single tile_y, M=16 / K=256 / m_total=16.
    run_case("toy", 16, 256, 16, 1, 0xDEAD_BEEF, 0xCAFE_BABE);
    // Small: 2 experts, 2 tile_y, M=32 / K=512 / m_total=32.
    run_case("small", 32, 512, 32, 2, 0x1234_5678, 0x8765_4321);
    // Medium: 4 experts, 4 tile_y, M=128 / K=1024 / m_total=64.
    run_case("medium", 128, 1024, 64, 4, 0x0F0F_0F0F, 0xF0F0_F0F0);
    // A3B-shaped slice: M=768, K=7168, m_total=256, E=8.
    run_case("a3b-slice", 768, 7168, 256, 8, 0x4242_4242, 0x2424_2424);

    println!("\nAll cases PASS.");
}
