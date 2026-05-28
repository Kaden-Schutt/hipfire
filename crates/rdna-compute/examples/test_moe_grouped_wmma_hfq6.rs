//! Byte-equivalent CPU/GPU correctness check for
//! `gemm_hfq6g256_moe_grouped_wmma` (gfx12 and gfx1151 variants).
//!
//! Unlike the HFQ4 m2 test (which A/Bs against the HFQ4 base kernel), the
//! HFQ6 grouped kernel is new — there is no peer GPU kernel to compare to.
//! We instead compute a full FP32 reference on the host:
//!   1. Dequantize each expert weight (HFQ6 200 B/group) to FP32.
//!   2. For each tile_y × m_lane, look up its expert via expert_tile_ids
//!      and its X-row via sorted_slot_index (using x_row_div).
//!   3. Compute Y[slot_idx, m] = sum_k A_dq[m, k] * X[x_row, k].
//!
//! The kernel runs in FP16 with WMMA accumulation; the CPU ref runs in
//! FP32, so we allow ULP-level slop (>1e-3 abs / 1e-2 rel = bug). On
//! K=7168 the bound is dominated by HFQ6 dequant precision (scale +
//! zero in FP16) plus the 4 K-tile partial-sum order — empirically the
//! HFQ4 sister sees ~1e-3 abs at K=7168, and HFQ6 is comparable.
//!
//! GFX12/GFX1151 ONLY. The wrapper routes to the arch-specific kernel.
//! Skips on other archs with a SKIP message.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example test_moe_grouped_wmma_hfq6

use rdna_compute::{Gpu, GpuTensor, DType};

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    *state & 0x7fff_ffff
}

/// FP32 → FP16 (binary16) → FP32 round-trip via raw bit-twiddling.
/// Matches `_Float16` semantics: round-to-nearest-even, saturates to
/// ±inf on overflow, flushes to ±0 on underflow below the subnormal
/// boundary. Used to model the kernel's a_reg = (_Float16)(...) cast
/// and the FP16 X operand on the CPU reference side.
fn fp32_to_fp16_to_fp32(f: f32) -> f32 {
    let bits = f.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;

    let h_bits: u16 = if exp == 0xff {
        // NaN or inf
        let m = if mant != 0 { 0x200 } else { 0 };
        ((sign as u16) << 15) | 0x7c00 | m
    } else if exp > 0x70 + 0x1f {
        // Overflow → ±inf
        ((sign as u16) << 15) | 0x7c00
    } else if exp >= 0x71 {
        // Normal half: half_exp = exp - 127 + 15 in [1..30]
        let he = (exp - 112) as u16;
        // Round-to-nearest-even on the 13 dropped mantissa bits.
        let m_top = mant >> 13;
        let rem = mant & 0x1fff;
        let half = 0x1000;
        let mut m = m_top as u16;
        if rem > half || (rem == half && (m & 1) != 0) {
            m += 1;
            if m == 0x400 {
                // Mantissa overflow → bump exponent.
                return f32_from_h16(((sign as u16) << 15) | ((he + 1) << 10));
            }
        }
        ((sign as u16) << 15) | (he << 10) | m
    } else if exp >= 0x67 {
        // Subnormal half (he == 0, mantissa shift varies).
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
        // Underflow → ±0
        (sign as u16) << 15
    };
    f32_from_h16(h_bits)
}

/// Decode an IEEE binary16 bit pattern back to FP32 (exact).
fn f32_from_h16(h: u16) -> f32 {
    let sign = (h >> 15) & 0x1;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits: u32 = if exp == 0 && mant == 0 {
        (sign as u32) << 31
    } else if exp == 0 {
        // Subnormal: normalize.
        let mut m = mant;
        let mut e: i32 = -14;
        while (m & 0x400) == 0 { m <<= 1; e -= 1; }
        m &= 0x3ff;
        ((sign as u32) << 31) | (((e + 127) as u32) << 23) | (m << 13)
    } else if exp == 0x1f {
        // inf or nan
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

/// Build a single HFQ6-G256 expert weight matrix [M × K] with
/// deterministic random 6-bit values. Each row has K/256 groups of 200
/// bytes:
///   [0..4]    f32 scale
///   [4..8]    f32 zero
///   [8..200]  192 bytes = 256 × 6-bit values (4 values per 3 bytes)
///
/// The 6-bit packing matches both the CPU dequant in qwen35.rs (case 8)
/// and the kernel inner loop (DQ6_8 macro): the 4-values-per-3-bytes
/// CPU layout is bit-equivalent to the kernel's overlapping-uint32 read
/// pattern (d0 = bytes 0..3, d1 = bytes 3..6, shifts 0/6/12/18 × mask 63).
fn build_expert_weight_hfq6(m: usize, k: usize, seed: u32) -> Vec<u8> {
    assert!(k % 256 == 0, "K must be a multiple of 256");
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 200;
    let total = m * bytes_per_row;
    let mut buf = vec![0u8; total];
    let mut s = seed;
    for row in 0..m {
        for g in 0..groups_per_row {
            let off = row * bytes_per_row + g * 200;
            // Scale: small positive in [0.001, 0.011). Smaller than HFQ4
            // because HFQ6 has 4× the codebook range (0..63 vs 0..15).
            let sc = 0.001_f32 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 0.010_f32;
            // Zero: small symmetric in [-0.05, 0.05).
            let zp = -0.05_f32 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 0.10_f32;
            buf[off..off + 4].copy_from_slice(&sc.to_le_bytes());
            buf[off + 4..off + 8].copy_from_slice(&zp.to_le_bytes());
            // 192 bytes = 256 × 6-bit, packed as 4 values per 3 bytes.
            // CPU encoding (inverse of the qwen35.rs dequant case 8):
            //   q0 = b0 & 0x3F
            //   q1 = (b0 >> 6) | (b1 << 2) & 0x3F
            //   q2 = (b1 >> 4) | (b2 << 4) & 0x3F
            //   q3 = (b2 >> 2) & 0x3F
            // Equivalently pack 4 6-bit values into 24 bits:
            //   bits[0..6]   = q0
            //   bits[6..12]  = q1
            //   bits[12..18] = q2
            //   bits[18..24] = q3
            for i in (0..256).step_by(4) {
                let byte_off = 8 + (i / 4) * 3;
                let q0 = (lcg(&mut s) % 64) as u32;
                let q1 = (lcg(&mut s) % 64) as u32;
                let q2 = (lcg(&mut s) % 64) as u32;
                let q3 = (lcg(&mut s) % 64) as u32;
                let packed: u32 = q0 | (q1 << 6) | (q2 << 12) | (q3 << 18);
                buf[off + byte_off]     = (packed & 0xFF) as u8;
                buf[off + byte_off + 1] = ((packed >> 8) & 0xFF) as u8;
                buf[off + byte_off + 2] = ((packed >> 16) & 0xFF) as u8;
            }
        }
    }
    buf
}

/// CPU dequant matching the kernel's FP16-precision a_reg formation:
///   sc_h = (f16) scale; zp_h = (f16) zero
///   a_reg[i] = sc_h * (f16)(float) q_i  + zp_h    (all in f16)
/// Returns one row of FP16-equivalent a_reg values stored as f32. Used
/// to match `gemm_hfq6g256_moe_grouped_wmma`'s inner-loop
/// precision; the WMMA op consumes a_reg as half8_t and accumulates in
/// FP32, which we mirror with f32 accumulation against an f16 X.
fn dequant_hfq6_row_fp16(weight: &[u8], k: usize) -> Vec<f32> {
    let groups = k / 256;
    let mut out = Vec::with_capacity(k);
    for g in 0..groups {
        let off = g * 200;
        let scale = f32::from_le_bytes([weight[off], weight[off+1], weight[off+2], weight[off+3]]);
        let zero  = f32::from_le_bytes([weight[off+4], weight[off+5], weight[off+6], weight[off+7]]);
        let sc_h = fp32_to_fp16_to_fp32(scale);
        let zp_h = fp32_to_fp16_to_fp32(zero);
        for i in (0..256).step_by(4) {
            let byte_off = 8 + (i / 4) * 3;
            let b0 = weight[off + byte_off]     as u32;
            let b1 = weight[off + byte_off + 1] as u32;
            let b2 = weight[off + byte_off + 2] as u32;
            let q0 = (b0 & 0x3F) as f32;
            let q1 = (((b0 >> 6) | (b1 << 2)) & 0x3F) as f32;
            let q2 = (((b1 >> 4) | (b2 << 4)) & 0x3F) as f32;
            let q3 = ((b2 >> 2) & 0x3F) as f32;
            // (f16)(float) q_i — round each q to f16. Integer 0..63 fits
            // exactly in f16, but expressed for parity with the kernel
            // (_Float16)(float)((dw>>sh) & 63u) cast.
            let q0_h = fp32_to_fp16_to_fp32(q0);
            let q1_h = fp32_to_fp16_to_fp32(q1);
            let q2_h = fp32_to_fp16_to_fp32(q2);
            let q3_h = fp32_to_fp16_to_fp32(q3);
            // a_reg = sc_h * q_h + zp_h, with FP16 arithmetic.
            let a0 = fp32_to_fp16_to_fp32(fp32_to_fp16_to_fp32(sc_h * q0_h) + zp_h);
            let a1 = fp32_to_fp16_to_fp32(fp32_to_fp16_to_fp32(sc_h * q1_h) + zp_h);
            let a2 = fp32_to_fp16_to_fp32(fp32_to_fp16_to_fp32(sc_h * q2_h) + zp_h);
            let a3 = fp32_to_fp16_to_fp32(fp32_to_fp16_to_fp32(sc_h * q3_h) + zp_h);
            out.push(a0); out.push(a1); out.push(a2); out.push(a3);
        }
    }
    out
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

/// Reference Y = A·X in FP16 to match the kernel's inner precision.
/// Kernel: A is FP16-dequantized; X is FP16; accumulate in FP32.
/// CPU ref mirrors that: convert dequant to f16 and X to f16, but
/// accumulate in f32. Output Y[slot, m] = sum_k a[m,k] * x[xrow, k].
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
    // Cache the per-expert dequant rows lazily. We compute the FP32
    // dequant of each expert once. For tight memory we could dequant
    // per-slot, but cases are small enough to materialize fully.
    let dequant: Vec<Vec<f32>> = expert_weights.iter()
        .map(|w| {
            // Full M × K dequant in FP16 precision (matches kernel a_reg).
            let groups_per_row = k / 256;
            let row_bytes = groups_per_row * 200;
            let mut acc = Vec::with_capacity(m * k);
            for row in 0..m {
                let row_off = row * row_bytes;
                let rd = dequant_hfq6_row_fp16(&w[row_off..row_off + row_bytes], k);
                acc.extend_from_slice(&rd);
            }
            acc
        })
        .collect();

    // Convert X to fp16 round-trip for closer kernel parity.
    let x_f16: Vec<f32> = x.iter().map(|&v| fp32_to_fp16_to_fp32(v)).collect();

    for tile_y in 0..tiles {
        let expert = tile_ids[tile_y];
        if expert < 0 { continue; }
        let dq = &dequant[expert as usize];
        let slot_start = tile_y * 16;
        for lane in 0..16 {
            let slot_idx = slot_start + lane;
            if slot_idx >= m_total { continue; }
            let flat = sorted[slot_idx];
            if flat < 0 { continue; }
            let x_row = if x_row_div > 1 { (flat as usize) / x_row_div } else { flat as usize };
            // Compute one column of Y: y[slot_idx, :] = dq * x[x_row, :].
            for mi in 0..m {
                let mut acc = 0f64;
                let dq_row_off = mi * k;
                let x_row_off = x_row * k;
                for ki in 0..k {
                    // dq[..] is already FP16-precision (see dequant_hfq6_row_fp16).
                    let a_f16 = dq[dq_row_off + ki];
                    acc += (a_f16 as f64) * (x_f16[x_row_off + ki] as f64);
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
    x_row_div: usize,
    seed_w: u32,
    seed_x: u32,
) {
    println!(
        "=== {} | M={} K={} m_total={} E={} x_row_div={} ===",
        label, m, k, m_total, num_experts, x_row_div
    );
    assert!(m % 16 == 0, "M must be a multiple of 16");
    assert!(m_total % 16 == 0, "m_total must be a multiple of 16");

    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    if !(arch.starts_with("gfx12") || arch.starts_with("gfx1151")) {
        println!(
            "  SKIP - arch {} is not gfx12/gfx1151; HFQ6 grouped kernel is only registered there",
            arch
        );
        return;
    }

    // Build E experts with distinct random fills.
    let mut expert_weights: Vec<Vec<u8>> = Vec::with_capacity(num_experts);
    let mut expert_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    let mut _expert_tensors: Vec<GpuTensor> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let bytes = build_expert_weight_hfq6(m, k, seed_w.wrapping_add(e as u32 * 9973));
        let t = upload_u8(&mut gpu, &bytes);
        expert_ptrs.push(t.buf.as_ptr() as u64);
        _expert_tensors.push(t);
        expert_weights.push(bytes);
    }
    let expert_weight_ptrs = upload_u64(&mut gpu, &expert_ptrs);

    // sorted_slot_index: identity. With x_row_div=K_TOP this mirrors gate/up:
    // slots 0..K_TOP-1 gather X row 0, etc. tile_y -> expert (tile_y % E).
    let sorted: Vec<i32> = (0..m_total as i32).collect();
    let sorted_slot_index = upload_i32(&mut gpu, &sorted);
    let tile_ids: Vec<i32> = (0..(m_total / 16))
        .map(|tile_y| (tile_y % num_experts) as i32)
        .collect();
    let expert_tile_ids = upload_i32(&mut gpu, &tile_ids);

    // X rows depend on the gather mode: down uses one X row per slot
    // (x_row_div=1), gate/up uses one X row per token and sorted flat slot
    // index divided by K_TOP.
    let x_rows = if x_row_div > 1 {
        (m_total + x_row_div - 1) / x_row_div
    } else {
        m_total
    };
    let x_f32 = build_x_f32(x_rows, k, seed_x);
    let x_src = upload_f32(&mut gpu, &x_f32);

    let y_gpu = alloc_f32_zeros(&mut gpu, m_total * m);

    gpu.gemm_hfq6g256_moe_grouped_wmma(
        &expert_weight_ptrs,
        &expert_tile_ids,
        &sorted_slot_index,
        &x_src,
        &y_gpu,
        m,
        k,
        x_row_div,
        m_total,
        x_rows,
    ).expect("hfq6 grouped kernel launch");
    gpu.hip.device_synchronize().expect("sync after hfq6 kernel");

    let y_gpu_v = download_f32(&gpu, &y_gpu, m_total * m);
    let y_ref = cpu_reference(&expert_weights, &x_f32, x_row_div, &sorted, &tile_ids, m, k, m_total);

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut argmax_abs = 0usize;
    let mut argmax_rel = 0usize;
    let mut sum_sq_err = 0f64;
    let mut sum_sq_ref = 0f64;
    let mut bad_count = 0usize;
    for (i, (a, b)) in y_ref.iter().zip(y_gpu_v.iter()).enumerate() {
        let d = (a - b).abs();
        let r = if a.abs() > 1e-6 { d / a.abs() } else { d };
        if d > max_abs { max_abs = d; argmax_abs = i; }
        if r > max_rel { max_rel = r; argmax_rel = i; }
        sum_sq_err += (d as f64) * (d as f64);
        sum_sq_ref += (*a as f64) * (*a as f64);
        if d > 5e-3 && r > 1e-2 {
            bad_count += 1;
        }
    }
    let rmse = (sum_sq_err / (m_total * m) as f64).sqrt() as f32;
    let nrmse = if sum_sq_ref > 0.0 {
        (sum_sq_err.sqrt() / sum_sq_ref.sqrt()) as f32
    } else { 0.0 };
    let gpu_nonzero = y_gpu_v.iter().any(|v| v.abs() > 1e-8);
    let ref_sample = &y_ref[argmax_abs];
    let gpu_sample = &y_gpu_v[argmax_abs];
    println!(
        "  max_abs_diff = {:.6e} (at {}: ref={:.6}, gpu={:.6})",
        max_abs, argmax_abs, ref_sample, gpu_sample
    );
    println!(
        "  max_rel_diff = {:.6e} (at {}: ref={:.6}, gpu={:.6})",
        max_rel, argmax_rel, y_ref[argmax_rel], y_gpu_v[argmax_rel]
    );
    let bad_limit = ((m_total * m) / 1000).max(8);
    println!(
        "  RMSE = {:.6e}   NRMSE = {:.6e}   bad = {} / limit {}",
        rmse, nrmse, bad_count, bad_limit
    );
    // FP16 WMMA accumulator + FP32 CPU ref -> ULP slop scales with K.
    // Treat near-zero relative outliers as diagnostics, not a hard gate.
    // The optional HFQ6 i8-MMQ route quantizes X to Q8_1 before the grouped
    // GEMM, so it must use the same Q8_1 noise envelope as the HFQ4 i8 tests.
    let use_i8_mmq = std::env::var("HIPFIRE_MOE_HFQ6_I8").as_deref() == Ok("1");
    let pass = if use_i8_mmq {
        gpu_nonzero && max_abs <= 2.5e-1 && nrmse <= 6e-3
    } else {
        gpu_nonzero && max_abs <= 5e-2 && nrmse <= 1e-3
    };
    if pass {
        if use_i8_mmq {
            println!("  PASS (Q8_1 i8-MMQ envelope)");
        } else {
            println!("  PASS");
        }
    } else {
        println!("  FAIL - exceeds FP16 WMMA slop");
        std::process::exit(1);
    }
}

fn main() {
    // Toy: 1 expert, single tile_y, M=16 / K=256 / m_total=16.
    run_case("toy-down", 16, 256, 16, 1, 1, 0xDEAD_BEEF, 0xCAFE_BABE);
    // Toy gate/up gather: the 16 routed slots collapse to two token rows
    // when x_row_div=8.
    run_case("toy-gate-up", 16, 256, 16, 1, 8, 0xDEAD_BEEF, 0xCAFE_BABE);
    // Small: 2 experts, 2 tile_y, M=32 / K=512 / m_total=32.
    run_case("small-down", 32, 512, 32, 2, 1, 0x1234_5678, 0x8765_4321);
    run_case("small-gate-up", 32, 512, 32, 2, 8, 0x1234_5678, 0x8765_4321);
    // Medium: 4 experts, 4 tile_y, M=128 / K=1024 / m_total=64.
    run_case("medium-down", 128, 1024, 64, 4, 1, 0x0F0F_0F0F, 0xF0F0_F0F0);
    run_case("medium-gate-up", 128, 1024, 64, 4, 8, 0x0F0F_0F0F, 0xF0F0_F0F0);
    // A3B-shaped slice: M=768 (mirrors per-expert gate_up/2), K=7168, m_total=256, E=8.
    run_case("a3b-slice-down", 768, 7168, 256, 8, 1, 0x4242_4242, 0x2424_2424);
    run_case("a3b-slice-gate-up", 1536, 2048, 256, 8, 8, 0x4242_4242, 0x2424_2424);

    println!("\nAll cases PASS.");
}
