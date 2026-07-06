//! FP32-reference correctness check for `gemm_mfp4g32_e8_moe_grouped_wmma`
//! (gfx1151). Adapted from `test_moe_grouped_wmma_hfq6.rs`.
//!
//! Random E8-encoded expert weights (any 32-bit word is a valid E8 codeword;
//! any byte is a valid E4M3 block scale), random X. The CPU reference
//! dequantizes each expert to FP16-precision `a_reg` values exactly as the
//! kernel does (row_scale × cvt_e4m3(block) × 0.88 × e8_decode(codeword)),
//! then computes Y[slot,m] = Σ_k a[m,k]·x_f16[xrow,k] in f32. The kernel runs
//! FP16 WMMA, so we allow ULP-level slop (garbage from a layout/index bug
//! shows as ~100% rel error, not 1e-2).
//!
//! GFX1151 ONLY (the kernel uses the wave32 f16 WMMA builtin and is registered
//! only for Strix Halo). Skips on other archs.
//!
//! Run: HIP_VISIBLE_DEVICES=1 cargo run --release -p rdna-compute \
//!        --example test_moe_grouped_wmma_e8

use rdna_compute::{DType, Gpu, GpuTensor};

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    *state & 0x7fff_ffff
}

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

// ── E8 decode (bit-identical to the kernel device functions) ──
fn cvt_e4m3_scale_to_f32(b: u8) -> f32 {
    let exp = ((b >> 3) & 0xF) as i32;
    let mant = (b & 0x7) as u32;
    if exp == 0 {
        return 0.015625 * (mant as f32) * 0.125;
    }
    if exp == 0xF && mant == 7 {
        return 448.0;
    }
    let pow2 = f32::from_bits(((exp + 120) as u32) << 23);
    pow2 * (1.0 + (mant as f32) * 0.125)
}

fn e8_decode_index(idx: u32) -> [f32; 8] {
    let coset = ((idx >> 31) & 1) as i32;
    let mut e = [0u32; 8];
    let mut sl = 0u32;
    for i in 0..7 {
        e[i] = (idx >> (4 * i as u32)) & 0xF;
        sl = sl.wrapping_add(e[i]);
    }
    let e7_high = (idx >> 28) & 0x7;
    let p7 = e7_high << 1;
    let lsb = (sl.wrapping_add(p7)) & 1;
    e[7] = p7 | lsb;
    let mut p = [0f32; 8];
    for i in 0..8 {
        let c = (e[i] as i32 - 7) as f32;
        p[i] = if coset != 0 { c + 0.5 } else { c };
    }
    p
}

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
fn download_f32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<f32> {
    let mut data = vec![0f32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    data
}

/// Build one E8 expert weight [M × K]: per row 16-byte header (f16 row_scale at
/// [0..2]) + (K/32) × 17-byte blocks (1 E4M3 scale byte + 4 × 32-bit codewords).
fn build_expert_weight_e8(m: usize, k: usize, seed: u32) -> Vec<u8> {
    assert!(k % 256 == 0, "K must be a multiple of 256");
    let n_blocks = k / 32;
    let row_bytes = 16 + n_blocks * 17;
    let mut buf = vec![0u8; m * row_bytes];
    let mut s = seed;
    // Exact-f16 row scales (0.5, 1.0, 1.5, 2.0).
    let rs_bits = [0x3800u16, 0x3C00, 0x3E00, 0x4000];
    for row in 0..m {
        let row_off = row * row_bytes;
        let rs = rs_bits[row % 4];
        buf[row_off] = (rs & 0xff) as u8;
        buf[row_off + 1] = (rs >> 8) as u8;
        for b in 0..n_blocks {
            let boff = row_off + 16 + b * 17;
            // E4M3 byte in ~[0.5, 2] (exp 6..8) to keep magnitudes f16-sane.
            buf[boff] = (0x30 + (lcg(&mut s) & 0xf)) as u8;
            for c in 0..4 {
                let cw = lcg(&mut s).wrapping_mul(2654435761) ^ lcg(&mut s);
                let cwoff = boff + 1 + c * 4;
                buf[cwoff] = (cw & 0xff) as u8;
                buf[cwoff + 1] = ((cw >> 8) & 0xff) as u8;
                buf[cwoff + 2] = ((cw >> 16) & 0xff) as u8;
                buf[cwoff + 3] = ((cw >> 24) & 0xff) as u8;
            }
        }
    }
    buf
}

/// CPU dequant of one E8 row to FP16-precision a_reg values (f32 storage).
fn dequant_e8_row_fp16(row: &[u8], k: usize) -> Vec<f32> {
    let row_scale = f32_from_h16(u16::from_le_bytes([row[0], row[1]]));
    let blocks = &row[16..];
    let n_blocks = k / 32;
    let mut out = vec![0f32; k];
    for blk in 0..n_blocks {
        let boff = blk * 17;
        let ssc = row_scale * cvt_e4m3_scale_to_f32(blocks[boff]) * 0.88;
        for cw_idx in 0..4 {
            let cwoff = boff + 1 + cw_idx * 4;
            let cw = u32::from_le_bytes([
                blocks[cwoff],
                blocks[cwoff + 1],
                blocks[cwoff + 2],
                blocks[cwoff + 3],
            ]);
            let p = e8_decode_index(cw);
            for j in 0..8 {
                out[blk * 32 + cw_idx * 8 + j] = fp32_to_fp16_to_fp32(p[j] * ssc);
            }
        }
    }
    out
}

fn build_x_f32(n: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    let mut out = vec![0f32; n * k];
    for v in out.iter_mut() {
        *v = -1.0 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 2.0;
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
    let row_bytes = 16 + (k / 32) * 17;
    let dequant: Vec<Vec<f32>> = expert_weights
        .iter()
        .map(|w| {
            let mut acc = Vec::with_capacity(m * k);
            for row in 0..m {
                let off = row * row_bytes;
                acc.extend_from_slice(&dequant_e8_row_fp16(&w[off..off + row_bytes], k));
            }
            acc
        })
        .collect();
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
                let dq_off = mi * k;
                let x_off = x_row * k;
                for ki in 0..k {
                    acc += (dq[dq_off + ki] as f64) * (x_f16[x_off + ki] as f64);
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
) -> bool {
    println!(
        "=== {} | M={} K={} m_total={} E={} ===",
        label, m, k, m_total, num_experts
    );
    assert!(m % 16 == 0 && m_total % 16 == 0);

    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    // gfx1151 RDNA3 (wave32-WMMA builtin) + gfx1200/1201 RDNA4 (the .gfx12
    // sibling kernel, selected by the launcher on is_rdna4()). Skip elsewhere.
    let is_rdna4 = matches!(arch.as_str(), "gfx1200" | "gfx1201");
    if !gpu.arch_caps.has_wmma_w32() && !is_rdna4 {
        println!("  SKIP — arch {} lacks RDNA3/RDNA4 wave32-WMMA", arch);
        return true;
    }

    let mut expert_weights: Vec<Vec<u8>> = Vec::with_capacity(num_experts);
    let mut expert_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    let mut _keep: Vec<GpuTensor> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let bytes = build_expert_weight_e8(m, k, seed_w.wrapping_add(e as u32 * 9973));
        let t = upload_u8(&mut gpu, &bytes);
        expert_ptrs.push(t.buf.as_ptr() as u64);
        _keep.push(t);
        expert_weights.push(bytes);
    }
    let expert_weight_ptrs = upload_u64(&mut gpu, &expert_ptrs);

    let sorted: Vec<i32> = (0..m_total as i32).collect();
    let sorted_slot_index = upload_i32(&mut gpu, &sorted);
    let tile_ids: Vec<i32> = (0..(m_total / 16))
        .map(|t| (t % num_experts) as i32)
        .collect();
    let expert_tile_ids = upload_i32(&mut gpu, &tile_ids);

    let x_f32 = build_x_f32(m_total, k, seed_x);
    let x_src = upload_f32(&mut gpu, &x_f32);
    let y_gpu = alloc_f32_zeros(&mut gpu, m_total * m);

    gpu.gemm_mfp4g32_e8_moe_grouped_wmma(
        &expert_weight_ptrs,
        &expert_tile_ids,
        &sorted_slot_index,
        &x_src,
        &y_gpu,
        m,
        k,
        1,
        m_total,
        m_total,
    )
    .expect("e8 grouped kernel launch");
    gpu.hip.device_synchronize().expect("sync");

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

    // Combined abs+rel tolerance: an element is a violation only when its error
    // exceeds BOTH bands. The kernel runs FP32-accumulated WMMA on f16 inputs;
    // the ref uses the SAME f16 inputs with f64 accumulation, so the residual is
    // f32-vs-f64 accumulation drift (~1e-3..1e-2 abs at K≈2048). A near-zero
    // output (catastrophic cancellation of ±7.5 lattice terms) can show large
    // REL on a tiny ABS — that is not a bug, so it must clear the abs band too.
    // A real layout/index error makes max_abs comparable to the value scale.
    const ABS_TOL: f32 = 2e-2;
    const REL_TOL: f32 = 3e-2;
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut argmax = 0usize;
    let mut violations = 0usize;
    for (i, (a, b)) in y_ref.iter().zip(y_gpu_v.iter()).enumerate() {
        let d = (a - b).abs();
        let r = if a.abs() > 1e-4 { d / a.abs() } else { 0.0 };
        if d > max_abs {
            max_abs = d;
            argmax = i;
        }
        if r > max_rel {
            max_rel = r;
        }
        if d > ABS_TOL + REL_TOL * a.abs() {
            violations += 1;
        }
    }
    println!(
        "  max_abs_diff = {:.4e} (at {}: ref={:.4}, gpu={:.4})",
        max_abs, argmax, y_ref[argmax], y_gpu_v[argmax]
    );
    println!(
        "  max_rel_diff = {:.4e}  (near-zero cancellation, not error scale)",
        max_rel
    );
    println!(
        "  violations (d > {ABS_TOL:.0e} + {REL_TOL:.0e}|a|) = {violations} / {}",
        y_ref.len()
    );
    let ok = violations == 0;
    println!(
        "  {}",
        if ok {
            "PASS"
        } else {
            "FAIL — abs+rel violations (layout/index bug?)"
        }
    );
    ok
}

fn main() {
    let mut all_ok = true;
    all_ok &= run_case("toy", 16, 256, 16, 1, 0xDEAD_BEEF, 0xCAFE_BABE);
    all_ok &= run_case("small", 32, 512, 32, 2, 0x1234_5678, 0x8765_4321);
    all_ok &= run_case("medium", 128, 1024, 64, 4, 0x0F0F_0F0F, 0xF0F0_F0F0);
    all_ok &= run_case("a3b-slice", 768, 2048, 256, 8, 0x4242_4242, 0x2424_2424);
    if all_ok {
        println!("\nAll cases PASS.");
    } else {
        println!("\nFAILURES present.");
        std::process::exit(1);
    }
}
