//! CPU/GPU correctness parity for `gemm_mixed_moe_grouped_wmma` —
//! the merged dtype-tag MoE grouped-WMMA prefill kernel.
//!
//! Each expert in the test is assigned `tag = e % 4`:
//!   tag 0 = MQ6   (200 B/group, 6-bit affine)
//!   tag 1 = MQ2-Lloyd (72 B/group, 4-entry codebook)
//!   tag 2 = MQ4   (136 B/group, 4-bit affine)
//!   tag 3 = MQ3-Lloyd (112 B/group, 8-entry codebook)
//!
//! Two sub-cases per shape: x_row_div = k_top (gate_up mode) and
//! x_row_div = 1 (down mode). CPU reference dispatches per-expert on
//! the tag, using FP16-precision dequant (fp32_to_fp16_to_fp32 applied
//! to affine result / codebook values) to match the kernel's operand
//! precision.
//!
//! Runs on gfx11 (gfx1100/gfx1151) and gfx12 (gfx1201).
//! Skips on archs without WMMA.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example test_moe_grouped_wmma_mixed

use rdna_compute::{DType, Gpu, GpuTensor};

// ── Utilities (verbatim from test_moe_grouped_wmma_mq3lloyd.rs) ───────────

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    *state & 0x7fff_ffff
}

fn fp32_to_fp16_to_fp32(f: f32) -> f32 {
    f32_from_h16(fp32_to_fp16_bits(f))
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

fn build_x_f32(n: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    let mut out = vec![0f32; n * k];
    for i in 0..n * k {
        out[i] = -1.0 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 2.0;
    }
    out
}

// ── Per-tag weight builders ────────────────────────────────────────────────

/// MQ6: 200 B/group = 4B f32-bits scale + 4B f32-bits zp + 192B 6-bit packed.
/// 256 elements × 6 bits = 1536 bits = 192 bytes.
/// Packing: groups of 6 elements share 4.5 bytes; kernel uses 4-overlapping
/// uint32 reads at offsets +0/+3/+6/+9, extracting 6-bit fields by shifting.
fn build_expert_weight_mq6(m: usize, k: usize, seed: u32) -> Vec<u8> {
    assert!(k % 256 == 0);
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 200;
    let mut buf = vec![0u8; m * bytes_per_row];
    let mut s = seed;
    for row in 0..m {
        for g in 0..groups_per_row {
            let off = row * bytes_per_row + g * 200;
            // scale: random positive f32 in [0.01, 0.5)
            let sc: f32 = 0.01 + (lcg(&mut s) as f32 / 0x7fff_ffff_u32 as f32) * 0.49;
            // zp: random f32 in [-16, 0)
            let zp: f32 = -16.0 + (lcg(&mut s) as f32 / 0x7fff_ffff_u32 as f32) * 16.0;
            let sc_bits = sc.to_bits().to_le_bytes();
            let zp_bits = zp.to_bits().to_le_bytes();
            buf[off..off + 4].copy_from_slice(&sc_bits);
            buf[off + 4..off + 8].copy_from_slice(&zp_bits);
            // 256 random 6-bit quant values packed LSB-first.
            // Pack 8 elements into 6 bytes (each element consumes 6 bits).
            // Layout mirrors the kernel's DQ6 macro which reads overlapping
            // uint32s at bytes gp+8+kt*12 to gp+8+kt*12+12 (12 bytes per
            // K-tile of 16 elements = 96 bits = 6 bits × 16).
            // Simple dense 6-bit packing: bit i goes to byte[i*6/8], bit[i*6%8].
            let payload_off = off + 8;
            let mut bit_pos: usize = 0;
            for _ in 0..256 {
                let q = (lcg(&mut s) % 64) as u8; // 6-bit value [0..63]
                let byte_idx = bit_pos / 8;
                let bit_off = bit_pos % 8;
                buf[payload_off + byte_idx] |= q << bit_off;
                if bit_off > 2 {
                    // Spans two bytes.
                    buf[payload_off + byte_idx + 1] |= q >> (8 - bit_off);
                }
                bit_pos += 6;
            }
        }
    }
    buf
}

/// CPU dequant for MQ6: sc_h * (q & 63) + zp_h, fp16-precision.
/// K-tile kt: 16 elements at bits kt*96..(kt+1)*96.
fn dequant_mq6_row(weight: &[u8], k: usize) -> Vec<f32> {
    let groups = k / 256;
    let bytes_per_row = groups * 200;
    let mut out = Vec::with_capacity(k);
    for g in 0..groups {
        let gp = g * 200;
        // Load sc and zp as f32 from the first 8 bytes.
        let sc_bits = u32::from_le_bytes(weight[gp..gp + 4].try_into().unwrap());
        let zp_bits = u32::from_le_bytes(weight[gp + 4..gp + 8].try_into().unwrap());
        let sc_f32 = f32::from_bits(sc_bits);
        let zp_f32 = f32::from_bits(zp_bits);
        // Apply fp16 round-trip (kernel stores as f32 bits but uses _Float16 arithmetic).
        let sc_h = fp32_to_fp16_to_fp32(sc_f32);
        let zp_h = fp32_to_fp16_to_fp32(zp_f32);
        let payload_off = gp + 8;
        // 256 elements packed as 6-bit LSB-first.
        let mut bit_pos: usize = 0;
        for _ in 0..256 {
            let byte_idx = bit_pos / 8;
            let bit_off = bit_pos % 8;
            let raw = weight[payload_off + byte_idx] as u32;
            let raw2 = if byte_idx + 1 < 192 {
                weight[payload_off + byte_idx + 1] as u32
            } else {
                0
            };
            let q = ((raw >> bit_off) | (raw2 << (8 - bit_off))) & 63;
            bit_pos += 6;
            let val = sc_h * (q as f32) + zp_h;
            out.push(fp32_to_fp16_to_fp32(val));
        }
    }
    out
}

/// MQ4: 136 B/group = 4B f32-bits scale + 4B f32-bits zp + 128B 4-bit packed.
fn build_expert_weight_mq4(m: usize, k: usize, seed: u32) -> Vec<u8> {
    assert!(k % 256 == 0);
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 136;
    let mut buf = vec![0u8; m * bytes_per_row];
    let mut s = seed;
    for row in 0..m {
        for g in 0..groups_per_row {
            let off = row * bytes_per_row + g * 136;
            let sc: f32 = 0.01 + (lcg(&mut s) as f32 / 0x7fff_ffff_u32 as f32) * 0.49;
            let zp: f32 = -8.0 + (lcg(&mut s) as f32 / 0x7fff_ffff_u32 as f32) * 8.0;
            let sc_bits = sc.to_bits().to_le_bytes();
            let zp_bits = zp.to_bits().to_le_bytes();
            buf[off..off + 4].copy_from_slice(&sc_bits);
            buf[off + 4..off + 8].copy_from_slice(&zp_bits);
            // 256 nibbles packed 2-per-byte little-endian.
            let payload_off = off + 8;
            for byte_idx in 0..128 {
                let lo = (lcg(&mut s) % 16) as u8;
                let hi = (lcg(&mut s) % 16) as u8;
                buf[payload_off + byte_idx] = lo | (hi << 4);
            }
        }
    }
    buf
}

/// CPU dequant for MQ4: sc_h * (q & 0xF) + zp_h, fp16-precision.
fn dequant_mq4_row(weight: &[u8], k: usize) -> Vec<f32> {
    let groups = k / 256;
    let bytes_per_row = groups * 136;
    let mut out = Vec::with_capacity(k);
    for g in 0..groups {
        let gp = g * 136;
        let sc_bits = u32::from_le_bytes(weight[gp..gp + 4].try_into().unwrap());
        let zp_bits = u32::from_le_bytes(weight[gp + 4..gp + 8].try_into().unwrap());
        let sc_h = fp32_to_fp16_to_fp32(f32::from_bits(sc_bits));
        let zp_h = fp32_to_fp16_to_fp32(f32::from_bits(zp_bits));
        let payload_off = gp + 8;
        for byte_idx in 0..128 {
            let b = weight[payload_off + byte_idx];
            let lo = (b & 0xF) as f32;
            let hi = ((b >> 4) & 0xF) as f32;
            out.push(fp32_to_fp16_to_fp32(sc_h * lo + zp_h));
            out.push(fp32_to_fp16_to_fp32(sc_h * hi + zp_h));
        }
    }
    out
}

/// MQ2-Lloyd: 72 B/group = 8B (4 × fp16 codebook, ascending) + 64B 2-bit packed.
fn build_expert_weight_mq2lloyd(m: usize, k: usize, seed: u32) -> Vec<u8> {
    assert!(k % 256 == 0);
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 72;
    let mut buf = vec![0u8; m * bytes_per_row];
    let mut s = seed;
    for row in 0..m {
        for g in 0..groups_per_row {
            let off = row * bytes_per_row + g * 72;
            // 4 ascending fp16 codebook entries in [0.01, 1.01).
            let mut cb_vals = [0f32; 4];
            for v in cb_vals.iter_mut() {
                *v = 0.01_f32 + (lcg(&mut s) as f32 / 0x7fff_ffff_u32 as f32);
            }
            cb_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (i, &v) in cb_vals.iter().enumerate() {
                let h = fp32_to_fp16_bits(v);
                buf[off + i * 2] = (h & 0xFF) as u8;
                buf[off + i * 2 + 1] = ((h >> 8) & 0xFF) as u8;
            }
            // 256 × 2-bit indices packed 8-per-byte (4 pairs per byte, 2 bits each).
            let payload_off = off + 8;
            for byte_idx in 0..64 {
                let mut b: u8 = 0;
                for bit in 0..4u8 {
                    let q = (lcg(&mut s) % 4) as u8;
                    b |= q << (bit * 2);
                }
                buf[payload_off + byte_idx] = b;
            }
        }
    }
    buf
}

/// CPU dequant for MQ2-Lloyd: bilinear decomposition with per-lane register
/// codebook (matches the kernel's per-lane register form — NOT LDS broadcast).
fn dequant_mq2lloyd_row(weight: &[u8], k: usize) -> Vec<f32> {
    let groups = k / 256;
    let bytes_per_row = groups * 72;
    let mut out = Vec::with_capacity(k);
    for g in 0..groups {
        let gp = g * 72;
        // Load 4 fp16 codebook entries.
        let mut cb = [0f32; 4];
        for i in 0..4 {
            let lo = weight[gp + i * 2] as u16;
            let hi = weight[gp + i * 2 + 1] as u16;
            cb[i] = f32_from_h16(lo | (hi << 8));
        }
        // Bilinear coefficients (per-lane register form):
        //   DQ = cb0 + q_lo * d01 + q_hi * d02 + (q_lo * q_hi) * d_xor
        // where index = q_hi*2 + q_lo (2-bit value, q_lo = bit 0, q_hi = bit 1).
        let cb0 = fp32_to_fp16_to_fp32(cb[0]);
        let cb1 = fp32_to_fp16_to_fp32(cb[1]);
        let cb2 = fp32_to_fp16_to_fp32(cb[2]);
        let cb3 = fp32_to_fp16_to_fp32(cb[3]);
        let d01 = fp32_to_fp16_to_fp32(cb1 - cb0);
        let d02 = fp32_to_fp16_to_fp32(cb2 - cb0);
        let d_xor = fp32_to_fp16_to_fp32(cb3 - cb2 - cb1 + cb0);
        let payload_off = gp + 8;
        // 256 elements from 64 bytes, 4 elements per byte, 2 bits each LSB-first.
        for byte_idx in 0..64 {
            let b = weight[payload_off + byte_idx];
            for bit in 0..4 {
                let q = (b >> (bit * 2)) & 3;
                let q_lo = (q & 1) as f32;
                let q_hi = ((q >> 1) & 1) as f32;
                let val = cb0 + q_lo * d01 + q_hi * d02 + (q_lo * q_hi) * d_xor;
                out.push(fp32_to_fp16_to_fp32(val));
            }
        }
    }
    out
}

/// MQ3-Lloyd: 112 B/group = 16B (8 × fp16 codebook) + 96B (256 × 3-bit, 8-per-3B LSB).
fn build_expert_weight_mq3lloyd(m: usize, k: usize, seed: u32) -> Vec<u8> {
    assert!(k % 256 == 0, "K must be a multiple of 256");
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 112;
    let mut buf = vec![0u8; m * bytes_per_row];
    let mut s = seed;
    for row in 0..m {
        for g in 0..groups_per_row {
            let off = row * bytes_per_row + g * 112;
            let mut cb_vals = [0f32; 8];
            for v in cb_vals.iter_mut() {
                *v = 0.01_f32 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32);
            }
            cb_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (i, &v) in cb_vals.iter().enumerate() {
                let h = fp32_to_fp16_bits(v);
                buf[off + i * 2] = (h & 0xFF) as u8;
                buf[off + i * 2 + 1] = ((h >> 8) & 0xFF) as u8;
            }
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

/// CPU dequant for MQ3-Lloyd: 3D bilinear (per-lane register form).
fn dequant_mq3lloyd_row(weight: &[u8], k: usize) -> Vec<f32> {
    let groups = k / 256;
    let mut out = Vec::with_capacity(k);
    for g in 0..groups {
        let gp = g * 112;
        let mut cb = [0f32; 8];
        for i in 0..8 {
            let lo = weight[gp + i * 2] as u16;
            let hi = weight[gp + i * 2 + 1] as u16;
            cb[i] = f32_from_h16(lo | (hi << 8));
        }
        for kt in 0..16_usize {
            let base = gp + 16 + kt * 6;
            let p0 = (weight[base] as u32)
                | ((weight[base + 1] as u32) << 8)
                | ((weight[base + 2] as u32) << 16);
            let p1 = (weight[base + 3] as u32)
                | ((weight[base + 4] as u32) << 8)
                | ((weight[base + 5] as u32) << 16);
            for sh in (0..24_u32).step_by(3) {
                let idx = ((p0 >> sh) & 7) as usize;
                out.push(cb[idx]);
            }
            for sh in (0..24_u32).step_by(3) {
                let idx = ((p1 >> sh) & 7) as usize;
                out.push(cb[idx]);
            }
        }
    }
    out
}

// ── Mixed CPU reference ────────────────────────────────────────────────────

/// Return the byte stride per row for the given tag.
fn group_bytes_for_tag(tag: u8) -> usize {
    match tag {
        0 => 200, // MQ6
        1 => 72,  // MQ2-Lloyd
        2 => 136, // MQ4
        3 => 112, // MQ3-Lloyd
        _ => panic!("bad tag {}", tag),
    }
}

fn cpu_reference_mixed(
    expert_weights: &[Vec<u8>],
    dtype_tags: &[u8],
    x: &[f32],
    x_row_div: usize,
    sorted: &[i32],
    tile_ids: &[i32],
    m: usize,
    k: usize,
    m_total: usize,
) -> Vec<f32> {
    let mut y = vec![0f32; m_total * m];
    // Dequant ALL M rows for each expert according to its tag.
    // The dequant_*_row functions take a row slice (one row's worth of bytes).
    let dequant: Vec<Vec<f32>> = expert_weights
        .iter()
        .enumerate()
        .map(|(e, w)| {
            let tag = dtype_tags[e];
            let groups_per_row = k / 256;
            let row_bytes = groups_per_row * group_bytes_for_tag(tag);
            let mut acc = Vec::with_capacity(m * k);
            for row in 0..m {
                let row_off = row * row_bytes;
                let row_slice = &w[row_off..row_off + row_bytes];
                let rd = match tag {
                    0 => dequant_mq6_row(row_slice, k),
                    1 => dequant_mq2lloyd_row(row_slice, k),
                    2 => dequant_mq4_row(row_slice, k),
                    3 => dequant_mq3lloyd_row(row_slice, k),
                    _ => unreachable!(),
                };
                acc.extend_from_slice(&rd);
            }
            acc
        })
        .collect();

    // Convert X to fp16 round-trip for parity with kernel's X operand.
    let x_f16: Vec<f32> = x.iter().map(|&v| fp32_to_fp16_to_fp32(v)).collect();

    let tiles = m_total / 16;
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
                    acc += (dq[dq_row_off + ki] as f64) * (x_f16[x_row_off + ki] as f64);
                }
                y[slot_idx * m + mi] = acc as f32;
            }
        }
    }
    y
}

// ── Run one test case ──────────────────────────────────────────────────────

fn check(label: &str, y_ref: &[f32], y_gpu: &[f32]) -> bool {
    let max_abs_y: f32 = y_ref.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let large_thresh = max_abs_y * 0.1_f32;
    let mut max_abs = 0f32;
    let mut max_rel_large = 0f32;
    let mut argmax_abs = 0usize;
    let mut argmax_rel_large = 0usize;
    for (i, (a, b)) in y_ref.iter().zip(y_gpu.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
            argmax_abs = i;
        }
        if a.abs() > large_thresh {
            let r = d / a.abs();
            if r > max_rel_large {
                max_rel_large = r;
                argmax_rel_large = i;
            }
        }
    }
    println!(
        "    {} max_abs={:.4e} (i={}: ref={:.4}, gpu={:.4})  max_rel_large={:.4e} (i={})",
        label,
        max_abs,
        argmax_abs,
        y_ref[argmax_abs],
        y_gpu[argmax_abs],
        max_rel_large,
        argmax_rel_large,
    );
    if max_abs > 0.1 || max_rel_large > 2e-2 {
        println!("    FAIL — exceeds tolerance (0.1 abs; 2e-2 rel for |ref|>10%% of max)");
        false
    } else {
        println!("    PASS");
        true
    }
}

fn run_case(
    label: &str,
    m: usize,
    k: usize,
    m_total: usize,
    num_experts: usize,
    k_top: usize,
    seed_w: u32,
    seed_x: u32,
) -> bool {
    println!(
        "=== {} | M={} K={} m_total={} E={} k_top={} ===",
        label, m, k, m_total, num_experts, k_top
    );
    assert!(m % 16 == 0);
    assert!(m_total % 16 == 0);
    assert!(k % 256 == 0);

    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    if !arch.starts_with("gfx11") && !arch.starts_with("gfx12") {
        println!("  SKIP — arch {} has no WMMA", arch);
        return true;
    }

    // Assign dtype_tags cycling e % 4 so all four tags are exercised.
    let dtype_tags_host: Vec<u8> = (0..num_experts as u8).map(|e| e % 4).collect();

    // Build per-expert weights according to their tag.
    let mut expert_weights: Vec<Vec<u8>> = Vec::with_capacity(num_experts);
    let mut expert_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    let mut _expert_tensors: Vec<GpuTensor> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let tag = dtype_tags_host[e];
        let seed = seed_w.wrapping_add(e as u32 * 9973);
        let bytes = match tag {
            0 => build_expert_weight_mq6(m, k, seed),
            1 => build_expert_weight_mq2lloyd(m, k, seed),
            2 => build_expert_weight_mq4(m, k, seed),
            3 => build_expert_weight_mq3lloyd(m, k, seed),
            _ => unreachable!(),
        };
        let t = upload_u8(&mut gpu, &bytes);
        expert_ptrs.push(t.buf.as_ptr() as u64);
        _expert_tensors.push(t);
        expert_weights.push(bytes);
    }
    let expert_weight_ptrs = upload_u64(&mut gpu, &expert_ptrs);
    let dtype_tags_gpu = upload_u8(&mut gpu, &dtype_tags_host);

    let sorted: Vec<i32> = (0..m_total as i32).collect();
    let sorted_slot_index = upload_i32(&mut gpu, &sorted);
    let tile_ids: Vec<i32> = (0..(m_total / 16))
        .map(|tile_y| (tile_y % num_experts) as i32)
        .collect();
    let expert_tile_ids = upload_i32(&mut gpu, &tile_ids);

    // x_src_rows for gate_up = m_total (x_row_div = k_top; each slot maps to a batch row).
    // x_src_rows for down = m_total (x_row_div = 1; each slot is its own row).
    // We allocate enough rows for the gate_up case.
    let x_src_rows = m_total;
    let x_f32 = build_x_f32(x_src_rows, k, seed_x);
    let x_src = upload_f32(&mut gpu, &x_f32);

    let mut all_pass = true;

    // Sub-case 1: gate_up (x_row_div = k_top).
    {
        let y_gpu = alloc_f32_zeros(&mut gpu, m_total * m);
        gpu.gemm_mixed_moe_grouped_wmma(
            &expert_weight_ptrs,
            &dtype_tags_gpu,
            &expert_tile_ids,
            &sorted_slot_index,
            &x_src,
            &y_gpu,
            m,
            k,
            k_top,
            m_total,
            x_src_rows,
        )
        .expect("mixed grouped kernel launch (gate_up)");
        gpu.hip
            .device_synchronize()
            .expect("sync after mixed kernel (gate_up)");
        let y_gpu_v = download_f32(&gpu, &y_gpu, m_total * m);
        let y_ref = cpu_reference_mixed(
            &expert_weights,
            &dtype_tags_host,
            &x_f32,
            k_top,
            &sorted,
            &tile_ids,
            m,
            k,
            m_total,
        );
        all_pass &= check("gate_up", &y_ref, &y_gpu_v);
    }

    // Sub-case 2: down (x_row_div = 1).
    {
        let y_gpu = alloc_f32_zeros(&mut gpu, m_total * m);
        gpu.gemm_mixed_moe_grouped_wmma(
            &expert_weight_ptrs,
            &dtype_tags_gpu,
            &expert_tile_ids,
            &sorted_slot_index,
            &x_src,
            &y_gpu,
            m,
            k,
            1,
            m_total,
            x_src_rows,
        )
        .expect("mixed grouped kernel launch (down)");
        gpu.hip
            .device_synchronize()
            .expect("sync after mixed kernel (down)");
        let y_gpu_v = download_f32(&gpu, &y_gpu, m_total * m);
        let y_ref = cpu_reference_mixed(
            &expert_weights,
            &dtype_tags_host,
            &x_f32,
            1,
            &sorted,
            &tile_ids,
            m,
            k,
            m_total,
        );
        all_pass &= check("down", &y_ref, &y_gpu_v);
    }

    all_pass
}

fn main() {
    let mut all_pass = true;

    // Smoke case: M=64, K=512, m_total=32, E=4 — one expert per tag, fast.
    all_pass &= run_case("smoke", 64, 512, 32, 4, 4, 0xDEAD_BEEF, 0xCAFE_BABE);

    // 8-expert case (2 experts per tag): small-ish M+K.
    all_pass &= run_case("small", 32, 512, 32, 8, 8, 0x1234_5678, 0x8765_4321);

    // Realistic A3B-shaped slice: M=768, K=7168, m_total=256, E=8.
    all_pass &= run_case("a3b-slice", 768, 7168, 256, 8, 8, 0x4242_4242, 0x2424_2424);

    if all_pass {
        println!("\nAll cases PASS.");
    } else {
        println!("\nSome cases FAILED.");
        std::process::exit(1);
    }
}
