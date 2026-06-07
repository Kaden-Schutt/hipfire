//! Channel test for gfx1151 raw F16/BF16 routed-MoE grouped WMMA kernels.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example test_moe_grouped_wmma_f16_bf16

use rdna_compute::{DType, Gpu, GpuTensor};

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    *state & 0x7fff_ffff
}

fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased < -14 {
        if unbiased < -24 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (-unbiased - 14 + 13) as u32;
        let mut hm = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (hm & 1) != 0) {
            hm += 1;
        }
        return sign | hm;
    }
    let mut hexp = (unbiased + 15) as u16;
    let mut hm = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (hm & 1) != 0) {
        hm += 1;
        if hm == 0x400 {
            hm = 0;
            hexp += 1;
        }
    }
    if hexp >= 31 {
        return sign | 0x7c00;
    }
    sign | (hexp << 10) | hm
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e = -14i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn f16_round(f: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(f))
}

fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb);
    (rounded >> 16) as u16
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn bf16_round(f: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(f))
}

fn upload_raw(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::Raw)
        .expect("alloc raw");
    gpu.hip.memcpy_htod(&t.buf, data).expect("upload raw");
    t
}

fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len()], DType::F32)
        .expect("alloc f32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("upload f32");
    t
}

fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    upload_raw(gpu, bytes)
}

fn upload_u64(gpu: &mut Gpu, data: &[u64]) -> GpuTensor {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
    upload_raw(gpu, bytes)
}

fn alloc_f32_zeros(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let t = gpu.alloc_tensor(&[n], DType::F32).expect("alloc y");
    gpu.hip.memset(&t.buf, 0, n * 4).expect("zero y");
    t
}

fn download_f32(gpu: &Gpu, tensor: &GpuTensor, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .expect("download f32");
    out
}

fn build_weight_bytes(m: usize, k: usize, seed: u32, bf16: bool) -> (Vec<u8>, Vec<f32>) {
    let mut s = seed;
    let mut bytes = Vec::with_capacity(m * k * 2);
    let mut rounded = Vec::with_capacity(m * k);
    for _ in 0..m * k {
        let v = -0.125 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 0.25;
        let bits = if bf16 {
            let b = f32_to_bf16_bits(v);
            rounded.push(bf16_bits_to_f32(b));
            b
        } else {
            let h = f32_to_f16_bits(v);
            rounded.push(f16_bits_to_f32(h));
            h
        };
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    (bytes, rounded)
}

fn build_x(rows: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..rows * k)
        .map(|_| -1.0 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 2.0)
        .collect()
}

fn cpu_ref(
    weights: &[Vec<f32>],
    sorted: &[i32],
    tile_ids: &[i32],
    x: &[f32],
    m: usize,
    k: usize,
    x_row_div: usize,
    m_total: usize,
    bf16: bool,
) -> Vec<f32> {
    let mut y = vec![0f32; m_total * m];
    let x_rounded: Vec<f32> = x
        .iter()
        .map(|v| if bf16 { bf16_round(*v) } else { f16_round(*v) })
        .collect();
    for tile_y in 0..(m_total / 16) {
        let expert_id = tile_ids[tile_y];
        if expert_id < 0 {
            continue;
        }
        let w = &weights[expert_id as usize];
        for lane in 0..16 {
            let slot = tile_y * 16 + lane;
            let flat = sorted[slot];
            if flat < 0 {
                continue;
            }
            let x_row = if x_row_div > 1 {
                flat as usize / x_row_div
            } else {
                flat as usize
            };
            for row in 0..m {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += w[row * k + kk] * x_rounded[x_row * k + kk];
                }
                y[slot * m + row] = acc;
            }
        }
    }
    y
}

fn run_case(
    label: &str,
    bf16: bool,
    m: usize,
    k: usize,
    m_total: usize,
    experts: usize,
    x_row_div: usize,
    sparse: bool,
) -> bool {
    println!(
        "=== {} {} M={} K={} m_total={} E={} x_row_div={} sparse={} ===",
        if bf16 { "BF16" } else { "F16" },
        label,
        m,
        k,
        m_total,
        experts,
        x_row_div,
        sparse
    );
    assert_eq!(k % 16, 0);
    assert_eq!(m_total % 16, 0);

    let mut gpu = match Gpu::init() {
        Ok(gpu) => gpu,
        Err(err) => {
            println!("  SKIP - Gpu::init failed: {err:?}");
            return false;
        }
    };
    if gpu.arch != "gfx1151" {
        println!("  SKIP - arch {} is not gfx1151", gpu.arch);
        return false;
    }

    let mut expert_ptrs = Vec::with_capacity(experts);
    let mut expert_weights = Vec::with_capacity(experts);
    let mut expert_tensors = Vec::with_capacity(experts);
    for e in 0..experts {
        let (bytes, rounded) = build_weight_bytes(m, k, 0x1000 + e as u32 * 7919, bf16);
        let t = upload_raw(&mut gpu, &bytes);
        expert_ptrs.push(t.buf.as_ptr() as u64);
        expert_weights.push(rounded);
        expert_tensors.push(t);
    }
    let expert_weight_ptrs = upload_u64(&mut gpu, &expert_ptrs);

    let x_rows = (m_total / x_row_div).max(1);
    let (sorted, tile_ids) = if sparse {
        let total_slots = (x_rows * x_row_div).min(experts * 16);
        let mut sorted = vec![-1i32; m_total];
        let mut tile_ids = vec![-1i32; m_total / 16];
        for flat in 0..total_slots {
            let expert = flat % experts;
            let lane = flat / experts;
            if lane < 16 {
                tile_ids[expert] = expert as i32;
                sorted[expert * 16 + lane] = flat as i32;
            }
        }
        (sorted, tile_ids)
    } else {
        (
            (0..m_total as i32).collect::<Vec<_>>(),
            (0..m_total / 16)
                .map(|tile| (tile % experts) as i32)
                .collect::<Vec<_>>(),
        )
    };
    let sorted_gpu = upload_i32(&mut gpu, &sorted);
    let tile_ids_gpu = upload_i32(&mut gpu, &tile_ids);
    let x = build_x(x_rows, k, 0xfeed_0000 ^ (m_total as u32));
    let x_gpu = upload_f32(&mut gpu, &x);
    let y_gpu = alloc_f32_zeros(&mut gpu, m_total * m);

    if bf16 {
        gpu.gemm_bf16_moe_grouped_wmma_gfx1151(
            &expert_weight_ptrs,
            &tile_ids_gpu,
            &sorted_gpu,
            &x_gpu,
            &y_gpu,
            m,
            k,
            x_row_div,
            m_total,
            x_rows,
        )
        .expect("BF16 grouped launch");
    } else {
        gpu.gemm_f16_moe_grouped_wmma_gfx1151(
            &expert_weight_ptrs,
            &tile_ids_gpu,
            &sorted_gpu,
            &x_gpu,
            &y_gpu,
            m,
            k,
            x_row_div,
            m_total,
            x_rows,
        )
        .expect("F16 grouped launch");
    }
    gpu.hip.device_synchronize().expect("sync");
    let got = download_f32(&gpu, &y_gpu, m_total * m);
    let want = cpu_ref(
        &expert_weights,
        &sorted,
        &tile_ids,
        &x,
        m,
        k,
        x_row_div,
        m_total,
        bf16,
    );

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut argmax = 0usize;
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let d = (a - b).abs();
        let r = if b.abs() > 1e-5 { d / b.abs() } else { d };
        if d > max_abs {
            max_abs = d;
            argmax = i;
        }
        max_rel = max_rel.max(r);
    }
    println!(
        "  max_abs={:.6e} max_rel={:.6e} at {} gpu={:.6} ref={:.6}",
        max_abs, max_rel, argmax, got[argmax], want[argmax]
    );
    let abs_tol = if bf16 { 6e-2 } else { 8e-3 };
    let rel_tol = if bf16 { 8e-2 } else { 2e-2 };
    if max_abs > abs_tol && max_rel > rel_tol {
        eprintln!("  FAIL");
        std::process::exit(1);
    }
    drop(expert_tensors);
    println!("  PASS");
    true
}

fn main() {
    let mut ran = 0usize;
    for bf16 in [false, true] {
        ran += run_case("down-dense", bf16, 32, 64, 32, 2, 1, false) as usize;
        ran += run_case("gate-up-div8", bf16, 64, 128, 128, 4, 8, false) as usize;
        ran += run_case("sparse-sentinel", bf16, 16, 64, 16 * 4, 4, 8, true) as usize;
        ran += run_case("padded-m", bf16, 48, 80, 48, 3, 1, false) as usize;
    }
    if ran == 0 {
        println!("\nNo F16/BF16 grouped MoE cases ran on this host.");
    } else {
        println!("\nAll executed F16/BF16 grouped MoE cases PASS.");
    }
}
