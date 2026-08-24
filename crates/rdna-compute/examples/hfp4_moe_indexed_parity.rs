// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Numerical parity for the HFP4G32 indexed MoE GEMVs.
//!
//! A wrong MoE GEMV does not error — it produces plausible-looking garbage that
//! only shows up as degraded generations, so these kernels need a reference
//! check before anything trusts them.
//!
//! Builds HFP4G32 expert blobs on the host, computes the expected gate/up and
//! down results in f64 on the CPU, then compares against the kernels. The
//! expert-pointer indirection is exercised too: pointers are shuffled so a
//! kernel that ignored `topk_indices` would read the wrong expert.

use rdna_compute::{DType, Gpu};

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn f32_to_f16_bits(v: f32) -> u16 {
    // Minimal round-to-nearest-even f32->f16 for the row scale.
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = b & 0x7F_FFFF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    let mut m = (mant >> 13) as u16;
    if (mant & 0x1000) != 0 && ((mant & 0x0FFF) != 0 || (m & 1) != 0) {
        m += 1;
        if m == 0x400 {
            m = 0;
            exp += 1;
        }
    }
    sign | ((exp as u16) << 10) | m
}

/// One HFP4G32 row: 16 B header + n_blocks * 17 B.
fn make_row(k: usize, row_scale: f32, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let n_blocks = k / 32;
    let mut bytes = vec![0u8; 16 + n_blocks * 17];
    bytes[0..2].copy_from_slice(&f32_to_f16_bits(row_scale).to_le_bytes());
    bytes[4..6].copy_from_slice(&(n_blocks as u16).to_le_bytes());
    bytes[6] = 0; // no rotation
    let mut vals = vec![0.0f32; k];
    let mut st = seed | 1;
    let mut next = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        st
    };
    // row_scale is read back through f16, so use the rounded value for the
    // reference or the comparison is off by the f16 rounding alone.
    let rs = half_to_f32(f32_to_f16_bits(row_scale));
    for b in 0..n_blocks {
        let e = (110 + (next() % 30)) as u8;
        let po = 16 + b * 17;
        bytes[po] = e;
        let bscale = (e as i32 - 127) as f32;
        let bscale = bscale.exp2();
        for i in 0..16 {
            let lo = (next() % 16) as u8;
            let hi = (next() % 16) as u8;
            bytes[po + 1 + i] = (lo & 0x0F) | ((hi & 0x0F) << 4);
            vals[b * 32 + 2 * i] = rs * bscale * E2M1[lo as usize];
            vals[b * 32 + 2 * i + 1] = rs * bscale * E2M1[hi as usize];
        }
    }
    (bytes, vals)
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut e = -1i32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (((e + 127 + 1) as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 31 {
        (sign << 31) | 0x7F80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn main() {
    let mut gpu = match Gpu::init_with_device(0) {
        Ok(g) => g,
        Err(e) => {
            println!("hfp4_moe_indexed_parity: no GPU ({e:?}) — SKIP");
            return;
        }
    };

    const K: usize = 256; // must be %32; keep small so the CPU ref is cheap
    const M: usize = 8; // gate rows 0..4, up rows 4..8
    const N_EXP: usize = 4;
    const K_TOP: usize = 2;
    let row_bytes = 16 + (K / 32) * 17;

    // Build N_EXP experts, each M rows.
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    let mut refs: Vec<Vec<Vec<f32>>> = Vec::new();
    for e in 0..N_EXP {
        let mut blob = Vec::with_capacity(M * row_bytes);
        let mut rows = Vec::new();
        for r in 0..M {
            let (b, v) = make_row(K, 0.75 + 0.1 * e as f32, (e * 131 + r * 17 + 1) as u64);
            blob.extend_from_slice(&b);
            rows.push(v);
        }
        blobs.push(blob);
        refs.push(rows);
    }

    let dev_blobs: Vec<_> = blobs
        .iter()
        .map(|b| gpu.upload_raw(b, &[b.len()]).expect("upload expert"))
        .collect();
    // Pointer table: expert id -> device pointer.
    let ptrs: Vec<u64> = dev_blobs.iter().map(|t| t.buf.as_ptr() as u64).collect();
    let ptr_bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let d_ptrs = gpu.upload_raw(&ptr_bytes, &[N_EXP]).expect("upload ptrs");

    // Deliberately non-identity so ignoring topk_indices fails.
    let topk: Vec<i32> = vec![3, 1];
    let topk_bytes: Vec<u8> = topk.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let d_topk = gpu.upload_raw(&topk_bytes, &[K_TOP]).expect("upload topk");

    let x: Vec<f32> = (0..K).map(|i| ((i % 13) as f32 - 6.0) * 0.125).collect();
    let d_x = gpu.upload_f32(&x, &[K]).expect("upload x");

    let mi = M / 2;
    let d_gate = gpu.alloc_tensor(&[K_TOP * mi], DType::F32).expect("gate");
    let d_up = gpu.alloc_tensor(&[K_TOP * mi], DType::F32).expect("up");

    gpu.deepseek4_gemv_hfp4g32_moe_gate_up_indexed(
        &d_ptrs, &d_topk, &d_x, &d_gate, &d_up, M, K, K_TOP,
    )
    .expect("gate_up launch");
    let got_gate = gpu.download_f32(&d_gate).expect("dl gate");
    let got_up = gpu.download_f32(&d_up).expect("dl up");

    let mut worst = 0.0f64;
    for (kr, &eid) in topk.iter().enumerate() {
        for r in 0..M {
            let w = &refs[eid as usize][r];
            let want: f64 = (0..K).map(|i| w[i] as f64 * x[i] as f64).sum();
            let got = if r < mi {
                got_gate[kr * mi + r]
            } else {
                got_up[kr * mi + (r - mi)]
            } as f64;
            let rel = (want - got).abs() / want.abs().max(1e-6);
            worst = worst.max(rel);
        }
    }
    println!("  gate_up worst relative error: {worst:.3e}");
    assert!(worst < 1e-5, "gate_up mismatch: {worst:.3e}");

    // ---- down: route-scaled residual accumulate ----
    let weights: Vec<f32> = vec![0.6, 0.4];
    let w_bytes: Vec<u8> = weights.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let d_w = gpu.upload_raw(&w_bytes, &[K_TOP]).expect("upload w");
    let xb: Vec<f32> = (0..K_TOP * K)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.25)
        .collect();
    let d_xb = gpu.upload_f32(&xb, &[K_TOP * K]).expect("upload xb");
    let resid = vec![0.0f32; M];
    let d_res = gpu.upload_f32(&resid, &[M]).expect("upload resid");

    gpu.deepseek4_gemv_hfp4g32_moe_down_residual_scaled_indexed(
        &d_ptrs, &d_topk, &d_w, &d_xb, &d_res, M, K, K_TOP,
    )
    .expect("down launch");
    let got_res = gpu.download_f32(&d_res).expect("dl resid");

    let mut worst_d = 0.0f64;
    for r in 0..M {
        let mut want = 0.0f64;
        for (kr, &eid) in topk.iter().enumerate() {
            let w = &refs[eid as usize][r];
            let dot: f64 = (0..K).map(|i| w[i] as f64 * xb[kr * K + i] as f64).sum();
            want += weights[kr] as f64 * dot;
        }
        let rel = (want - got_res[r] as f64).abs() / want.abs().max(1e-6);
        worst_d = worst_d.max(rel);
    }
    println!("  down worst relative error:    {worst_d:.3e}");
    assert!(worst_d < 1e-5, "down mismatch: {worst_d:.3e}");

    // ---- batched (prefill) variants ----
    // N>1 with DIFFERENT experts per position, so a kernel that ignored the
    // batch stride on topk/x/y would produce the right answer for bid=0 and
    // garbage after — the failure mode a single-position test cannot see.
    const N: usize = 3;
    let topk_b: Vec<i32> = vec![3, 1, 0, 2, 2, 3]; // [N x K_TOP]
    let tb_bytes: Vec<u8> = topk_b.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let d_topk_b = gpu.upload_raw(&tb_bytes, &[N * K_TOP]).expect("topk_b");

    let xb2: Vec<f32> = (0..N * K).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
    let d_xb2 = gpu.upload_f32(&xb2, &[N * K]).expect("xb2");
    let d_gate_b = gpu
        .alloc_tensor(&[N * K_TOP * mi], DType::F32)
        .expect("gate_b");
    let d_up_b = gpu
        .alloc_tensor(&[N * K_TOP * mi], DType::F32)
        .expect("up_b");

    gpu.deepseek4_gemv_hfp4g32_moe_gate_up_indexed_batched(
        &d_ptrs, &d_topk_b, &d_xb2, &d_gate_b, &d_up_b, M, K, K_TOP, N,
    )
    .expect("batched gate_up");
    let gb = gpu.download_f32(&d_gate_b).expect("dl gate_b");
    let ub = gpu.download_f32(&d_up_b).expect("dl up_b");

    let mut worst_bg = 0.0f64;
    for bid in 0..N {
        for kr in 0..K_TOP {
            let eid = topk_b[bid * K_TOP + kr] as usize;
            for r in 0..M {
                let w = &refs[eid][r];
                let want: f64 = (0..K).map(|i| w[i] as f64 * xb2[bid * K + i] as f64).sum();
                let stride = K_TOP * mi;
                let got = if r < mi {
                    gb[bid * stride + kr * mi + r]
                } else {
                    ub[bid * stride + kr * mi + (r - mi)]
                } as f64;
                worst_bg = worst_bg.max((want - got).abs() / want.abs().max(1e-6));
            }
        }
    }
    println!("  batched gate_up worst relative error: {worst_bg:.3e}");
    assert!(worst_bg < 1e-5, "batched gate_up mismatch: {worst_bg:.3e}");

    let wb: Vec<f32> = vec![0.5, 0.5, 0.7, 0.3, 0.25, 0.75];
    let wb_bytes: Vec<u8> = wb.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let d_wb = gpu.upload_raw(&wb_bytes, &[N * K_TOP]).expect("wb");
    let xdb: Vec<f32> = (0..N * K_TOP * K)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.2)
        .collect();
    let d_xdb = gpu.upload_f32(&xdb, &[N * K_TOP * K]).expect("xdb");
    let d_resb = gpu
        .upload_f32(&vec![0.0f32; N * M], &[N * M])
        .expect("resb");

    gpu.deepseek4_gemv_hfp4g32_moe_down_residual_scaled_indexed_batched(
        &d_ptrs, &d_topk_b, &d_wb, &d_xdb, &d_resb, M, K, K_TOP, N,
    )
    .expect("batched down");
    let rb = gpu.download_f32(&d_resb).expect("dl resb");

    let mut worst_bd = 0.0f64;
    for bid in 0..N {
        for r in 0..M {
            let mut want = 0.0f64;
            for kr in 0..K_TOP {
                let eid = topk_b[bid * K_TOP + kr] as usize;
                let w = &refs[eid][r];
                let base = (bid * K_TOP + kr) * K;
                let dot: f64 = (0..K).map(|i| w[i] as f64 * xdb[base + i] as f64).sum();
                want += wb[bid * K_TOP + kr] as f64 * dot;
            }
            worst_bd = worst_bd.max((want - rb[bid * M + r] as f64).abs() / want.abs().max(1e-6));
        }
    }
    println!("  batched down worst relative error:    {worst_bd:.3e}");
    assert!(worst_bd < 1e-5, "batched down mismatch: {worst_bd:.3e}");

    println!("\nhfp4_moe_indexed_parity: PASS");
}
