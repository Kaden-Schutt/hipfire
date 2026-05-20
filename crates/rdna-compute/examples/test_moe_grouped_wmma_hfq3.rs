//! Byte-exact channel test for `gemm_hfq3g256_moe_grouped_wmma_gfx12`
//! against a CPU reference. The CPU ref mirrors the per-element math
//! the WMMA kernel does (FP16 dequant + FP16 X load → FP32 accumulate)
//! at GROUP granularity; we permit ULP-level slop to absorb the
//! WMMA-vs-scalar FMA-order divergence (per-group inputs are bounded
//! so the absolute slop floor is loose-but-non-trivial).
//!
//! GFX12 ONLY. The kernel is registered only as a gfx12 variant.
//! Skips on non-gfx12 archs with a SKIP message.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example test_moe_grouped_wmma_hfq3

use rdna_compute::{Gpu, GpuTensor, DType};

// FP32 -> FP16 -> FP32 round trip via IEEE 754 binary16 (round to even).
// Matches the implicit conversion done by `ensure_fp16_x` in the
// dispatcher + the FP16-storage X load inside the WMMA kernel.
fn f16_round_trip(f: f32) -> f32 {
    let bits = f.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    // NaN / Inf
    if exp == 0xFF {
        if mant != 0 {
            // NaN
            return f32::from_bits((sign << 31) | (0xFF << 23) | 0x400000);
        }
        // Inf
        return f32::from_bits((sign << 31) | (0xFF << 23));
    }

    let unbiased = exp - 127;

    // Overflow to fp16 Inf
    if unbiased > 15 {
        return f32::from_bits((sign << 31) | (0xFF << 23));
    }
    // Underflow to fp16 subnormal or zero
    if unbiased < -14 {
        if unbiased < -24 {
            // Underflow to signed zero
            return f32::from_bits(sign << 31);
        }
        // Subnormal: shift mantissa
        let m = mant | 0x800000;
        let shift = -unbiased - 14 + 13;
        let half_mant = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let half_bit = 1 << (shift - 1);
        let round_up = rem > half_bit || (rem == half_bit && (half_mant & 1) != 0);
        let hm = half_mant + if round_up { 1 } else { 0 };
        // Promotion to normal if mantissa overflows
        if hm >= 0x400 {
            // Becomes smallest normal fp16: exp=1, mant=0
            let f16_bits: u32 = (sign << 15) | (1 << 10);
            return f16_to_f32(f16_bits as u16);
        }
        let f16_bits: u32 = (sign << 15) | (hm & 0x3FF);
        return f16_to_f32(f16_bits as u16);
    }
    // Normal fp16
    let half_mant = mant >> 13;
    let rem = mant & 0x1FFF;
    let half_bit = 0x1000;
    let round_up = rem > half_bit || (rem == half_bit && (half_mant & 1) != 0);
    let mut hm = half_mant + if round_up { 1 } else { 0 };
    let mut hexp = (unbiased + 15) as u32;
    if hm >= 0x400 {
        hm = 0;
        hexp += 1;
        if hexp >= 31 {
            // Overflow to Inf
            return f32::from_bits((sign << 31) | (0xFF << 23));
        }
    }
    let f16_bits: u32 = (sign << 15) | (hexp << 10) | (hm & 0x3FF);
    f16_to_f32(f16_bits as u16)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign: u32 = ((h >> 15) & 0x1) as u32;
    let exp: u32 = ((h >> 10) & 0x1F) as u32;
    let mant: u32 = (h & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal — normalize
        let mut m = mant;
        let mut e: i32 = 1;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f_exp = (e - 15 + 127) as u32;
        return f32::from_bits((sign << 31) | (f_exp << 23) | (m << 13));
    }
    if exp == 31 {
        if mant == 0 {
            return f32::from_bits((sign << 31) | (0xFF << 23));
        }
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }
    let f_exp = exp - 15 + 127;
    f32::from_bits((sign << 31) | (f_exp << 23) | (mant << 13))
}

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

/// Build a single HFQ3-G256 expert weight matrix [M × K] with
/// deterministic random scales/zeros and 3-bit weight nibbles. Each row
/// is K/256 groups × 104 bytes:
///   [0..4]   f32 scale
///   [4..8]   f32 zero
///   [8..104] 96 bytes = 32 chunks × 3 B, each chunk encodes 8 weights
///            across 24 bits ((pk >> (3*i)) & 7u)
fn build_expert_weight(m: usize, k: usize, seed: u32) -> Vec<u8> {
    assert!(k % 256 == 0, "K must be a multiple of 256");
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 104;
    let total = m * bytes_per_row;
    let mut buf = vec![0u8; total];
    let mut s = seed;
    for row in 0..m {
        for g in 0..groups_per_row {
            let off = row * bytes_per_row + g * 104;
            // Scale: small positive in [0.005, 0.025).
            let sc = 0.005_f32 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 0.020_f32;
            // Zero: small symmetric in [-0.05, 0.05).
            let zp = -0.05_f32 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 0.10_f32;
            buf[off..off + 4].copy_from_slice(&sc.to_le_bytes());
            buf[off + 4..off + 8].copy_from_slice(&zp.to_le_bytes());
            // 96 bytes = 32 × 3 B chunks. Each chunk is 24 bits packing 8 × 3-bit weights.
            // Build chunk-by-chunk so the bit layout matches the unpack in
            // the kernel and `gemv_hfq3g256.hip:64-79`.
            for c in 0..32 {
                let mut pk: u32 = 0;
                for i in 0..8 {
                    let w = (lcg(&mut s) % 8) as u32; // 3-bit value
                    pk |= w << (3 * i);
                }
                // pk uses 24 bits; little-endian byte layout matches
                // `unsigned int pk = d[0] | d[1]<<8 | d[2]<<16` in the
                // gfx1100 GEMV reference.
                buf[off + 8 + c * 3 + 0] = (pk & 0xFF) as u8;
                buf[off + 8 + c * 3 + 1] = ((pk >> 8) & 0xFF) as u8;
                buf[off + 8 + c * 3 + 2] = ((pk >> 16) & 0xFF) as u8;
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

/// CPU reference: dequant each (row, k) weight to FP16, multiply by the
/// FP16-rounded X[slot_row, k], accumulate in FP32. Same dataflow
/// granularity as the WMMA kernel (modulo K-tile ordering — we sum K
/// monotonically here; the kernel sums in groups of 16 K via WMMA's
/// internal reduction tree, then group-by-group across K via FP32 add).
/// At input-magnitude ~[-0.05, 0.05] × [-1, 1] over K up to 7168, total
/// abs is bounded by ~360, so 1e-3 absolute slop comfortably covers
/// the WMMA-vs-scalar-FMA reorder slop.
fn cpu_ref(
    weights: &[Vec<u8>],
    expert_tile_ids: &[i32],
    sorted_slot_index: &[i32],
    x: &[f32],
    m: usize,
    k: usize,
    x_row_div: usize,
    m_total: usize,
) -> Vec<f32> {
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 104;
    let mut y = vec![0f32; m_total * m];

    // Convert X to fp16-rounded fp32 once.
    let x_fp16_rounded: Vec<f32> = x.iter().map(|v| f16_round_trip(*v)).collect();

    for tile_y in 0..(m_total / 16) {
        let expert_id = expert_tile_ids[tile_y];
        if expert_id < 0 { continue; }
        let a = &weights[expert_id as usize];

        for m_lane in 0..16 {
            let slot_idx = tile_y * 16 + m_lane;
            let flat = sorted_slot_index[slot_idx];
            if flat < 0 { continue; }
            let x_row = if x_row_div > 1 { (flat as usize) / x_row_div } else { flat as usize };

            for row_start in (0..m).step_by(16) {
                for j_lane in 0..16 {
                    let my_row = row_start + j_lane;
                    if my_row >= m { continue; }

                    let row_off = my_row * bytes_per_row;
                    let mut acc: f32 = 0.0;

                    for g in 0..groups_per_row {
                        let off = row_off + g * 104;
                        let sc_bytes: [u8; 4] = a[off..off + 4].try_into().unwrap();
                        let zp_bytes: [u8; 4] = a[off + 4..off + 8].try_into().unwrap();
                        let sc = f32::from_le_bytes(sc_bytes);
                        let zp = f32::from_le_bytes(zp_bytes);
                        let sc_h = f16_round_trip(sc);
                        let zp_h = f16_round_trip(zp);

                        // 256 k-values per group; iterate chunk-by-chunk
                        // matching the kernel's K-tile structure (kt 0..16,
                        // each kt has 2 chunks (k_grp 0,1), each chunk = 8 weights).
                        for kt in 0..16 {
                            for k_grp in 0..2 {
                                let chunk_idx = kt * 2 + k_grp;
                                let dp = off + 8 + chunk_idx * 3;
                                let b0 = a[dp + 0] as u32;
                                let b1 = a[dp + 1] as u32;
                                let b2 = a[dp + 2] as u32;
                                // Cross-byte 3-bit unpack matching kernel macro.
                                let unpack = [
                                     b0        & 7,
                                    (b0 >> 3)  & 7,
                                   ((b0 >> 6) | (b1 << 2)) & 7,
                                    (b1 >> 1)  & 7,
                                    (b1 >> 4)  & 7,
                                   ((b1 >> 7) | (b2 << 1)) & 7,
                                    (b2 >> 2)  & 7,
                                    (b2 >> 5)  & 7,
                                ];
                                for i in 0..8 {
                                    let k_idx = g * 256 + kt * 16 + k_grp * 8 + i;
                                    let w_h = f16_round_trip(sc_h * (unpack[i] as f32) + zp_h);
                                    let x_val = x_fp16_rounded[x_row * k + k_idx];
                                    acc += w_h * x_val;
                                }
                            }
                        }
                    }

                    let out_col = tile_y * 16 + m_lane;
                    y[out_col * m + my_row] = acc;
                }
            }
        }
    }
    y
}

fn run_case(label: &str, m: usize, k: usize, m_total: usize, num_experts: usize, seed_w: u32, seed_x: u32) {
    println!("=== {} | M={} K={} m_total={} E={} ===", label, m, k, m_total, num_experts);
    assert!(m % 16 == 0, "M must be a multiple of 16");
    assert!(m_total % 16 == 0, "m_total must be a multiple of 16");
    assert!(k % 256 == 0, "K must be a multiple of 256");

    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    if !arch.starts_with("gfx12") {
        println!("  SKIP — arch {} is not gfx12; HFQ3 grouped WMMA only registered for gfx12", arch);
        return;
    }

    // Build E experts of identical shape, with distinct random fills.
    let mut expert_bytes: Vec<Vec<u8>> = Vec::with_capacity(num_experts);
    let mut expert_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    let mut _expert_tensors: Vec<GpuTensor> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let bytes = build_expert_weight(m, k, seed_w.wrapping_add(e as u32 * 9973));
        let t = upload_u8(&mut gpu, &bytes);
        expert_ptrs.push(t.buf.as_ptr() as u64);
        expert_bytes.push(bytes);
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

    gpu.gemm_hfq3g256_moe_grouped_wmma(
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
    ).expect("HFQ3 grouped WMMA launch");
    gpu.hip.device_synchronize().expect("sync after HFQ3 launch");

    let y_gpu_v = download_f32(&gpu, &y_gpu, m_total * m);
    let y_ref_v = cpu_ref(&expert_bytes, &tile_ids, &sorted, &x_f32, m, k, 1, m_total);

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut argmax_abs = 0usize;
    for (i, (a, b)) in y_gpu_v.iter().zip(y_ref_v.iter()).enumerate() {
        let d = (a - b).abs();
        let r = if b.abs() > 1e-6 { d / b.abs() } else { d };
        if d > max_abs { max_abs = d; argmax_abs = i; }
        if r > max_rel { max_rel = r; }
    }
    let gpu_sample = y_gpu_v[argmax_abs];
    let ref_sample = y_ref_v[argmax_abs];
    println!(
        "  max_abs_diff = {:.6e} (at {}: gpu={:.6}, ref={:.6})",
        max_abs, argmax_abs, gpu_sample, ref_sample
    );
    println!("  max_rel_diff = {:.6e}", max_rel);
    // Tolerances: WMMA's reduction tree differs from sequential scalar
    // FMA, so absolute drift scales with sqrt(K) × |x|×|w|. For K=7168
    // and inputs ~[-0.05, 0.05], an empirical 1e-3 abs / 1e-2 rel
    // envelope covers the worst-case slop while still catching real
    // bugs (full-row flips show 1+ orders of magnitude over this).
    if max_abs > 1e-3 || max_rel > 1e-2 {
        println!("  FAIL — exceeds tolerance");
        std::process::exit(1);
    } else {
        println!("  PASS");
    }
}

fn main() {
    // Toy: 1 expert, single tile_y, M=16 / K=256 / m_total=16.
    run_case("toy", 16, 256, 16, 1, 0xDEAD_BEEF, 0xCAFE_BABE);
    // Small: 2 experts, 2 tile_y, M=32 / K=512 / m_total=32.
    run_case("small", 32, 512, 32, 2, 0x1234_5678, 0x8765_4321);
    // Medium: 4 experts, 4 tile_y, M=64 / K=1024 / m_total=64.
    run_case("medium", 64, 1024, 64, 4, 0x0F0F_0F0F, 0xF0F0_F0F0);
    // A3B-shaped slice: M=768 (per-expert gate_up/2), K=7168, m_total=256.
    run_case("a3b-slice", 768, 7168, 256, 8, 0x4242_4242, 0x2424_2424);

    println!("\nAll cases PASS.");
}
