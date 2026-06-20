// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Numeric parity for the generic-library GEMV tier (gfx1103 wave32, zero LDS):
//! f16→f32, f16→f16, bf16→f32, bf16→bf16, iu8→i32, iu4→i32.
//! y[m] = sum_k W[m,k] * x[k]. Floats accumulate in F32 (ULP tolerance on the
//! 16-bit store); ints are EXACT. Run on RDNA3 (gfx1103/1100/1151).
//!
//!   cargo run -p rdna-compute --example parity_gemv_generic [M K]

use rdna_compute::Gpu;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            ((s >> 16) & 0x7fff) as f32 / 32768.0 - 0.5
        })
        .collect()
}
fn lcg_i8(seed: u32, n: usize) -> Vec<i8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (((s >> 16) & 0xff) as i32 - 128).clamp(-127, 127) as i8
        })
        .collect()
}
fn lcg_i4(seed: u32, n: usize) -> Vec<i8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (((s >> 16) & 0xf) as i32 - 8) as i8
        })
        .collect()
}

fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        return (b >> 16) as u16;
    }
    let bias = 0x7fff + ((b >> 16) & 1);
    ((b + bias) >> 16) as u16
}
fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}
fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0xff {
        return sign | 0x7c00 | (if mant != 0 { 0x200 } else { 0 });
    }
    let ne = exp - 127 + 15;
    if ne >= 0x1f {
        return sign | 0x7c00;
    }
    if ne <= 0 {
        if ne < -10 {
            return sign;
        }
        let m = mant | 0x800000;
        let sh = (14 - ne) as u32;
        return sign | ((m + (1 << (sh - 1))) >> sh) as u16;
    }
    let hm = (mant >> 13) as u16;
    let mut h = sign | ((ne as u16) << 10) | hm;
    if (mant >> 12) & 1 == 1 {
        h = h.wrapping_add(1);
    }
    h
}
fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            let mut ee = -1i32;
            let mut mm = m;
            while mm & 0x400 == 0 {
                mm <<= 1;
                ee -= 1;
            }
            (s << 31) | (((127 - 15 + 1 + ee) as u32) << 23) | ((mm & 0x3ff) << 13)
        }
    } else if e == 0x1f {
        (s << 31) | 0x7f80_0000 | (m << 13)
    } else {
        (s << 31) | (((e as i32 - 15 + 127) as u32) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

fn bytes16<F: Fn(f32) -> u16>(v: &[f32], f: F) -> Vec<u8> {
    v.iter().flat_map(|&x| f(x).to_le_bytes()).collect()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(256);

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP gemv generic parity: {} lacks wave32 WMMA", gpu.arch);
        return;
    }
    println!("gemv generic parity M={m} K={k} on {}:", gpu.arch);
    let mut all_pass = true;

    // ---- float reference inputs ----
    let w = lcg(1, m * k);
    let x = lcg(2, k);

    // f16 inputs (shared bytes both sides)
    let wr16: Vec<f32> = w.iter().map(|&v| f16_to_f32(f32_to_f16(v))).collect();
    let xr16: Vec<f32> = x.iter().map(|&v| f16_to_f32(f32_to_f16(v))).collect();
    let wrb: Vec<f32> = w.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
    let xrb: Vec<f32> = x.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
    let dot = |wv: &[f32], xv: &[f32], row: usize| {
        let mut s = 0.0f32;
        for ki in 0..k {
            s += wv[row * k + ki] * xv[ki];
        }
        s
    };

    let w_f16 = gpu.upload_raw(&bytes16(&w, f32_to_f16), &[m, k]).unwrap();
    let x_f16 = gpu.upload_raw(&bytes16(&x, f32_to_f16), &[k]).unwrap();
    let w_bf16 = gpu.upload_raw(&bytes16(&w, f32_to_bf16), &[m, k]).unwrap();
    let x_bf16 = gpu.upload_raw(&bytes16(&x, f32_to_bf16), &[k]).unwrap();

    // f16 -> f32
    {
        let y = gpu.upload_raw(&vec![0u8; m * 4], &[m]).unwrap();
        gpu.gemv_f16_f32(&w_f16, &x_f16, &y, m, k).unwrap();
        gpu.device_synchronize().unwrap();
        let yg = gpu.download_f32(&y).unwrap();
        let mut mx = 0.0f32;
        let mut mag = 0.0f32;
        for r in 0..m {
            let rf = dot(&wr16, &xr16, r);
            mx = mx.max((yg[r] - rf).abs());
            mag = mag.max(rf.abs());
        }
        report("f16->f32 ", mx, 3.0 * mag * 2f32.powi(-10) + 1e-4, &mut all_pass);
    }
    // f16 -> f16
    {
        let y = gpu.upload_raw(&vec![0u8; m * 2], &[m]).unwrap();
        gpu.gemv_f16_f16(&w_f16, &x_f16, &y, m, k).unwrap();
        gpu.device_synchronize().unwrap();
        let yb = gpu.download_raw(&y, m * 2).unwrap();
        let mut mx = 0.0f32;
        let mut mag = 0.0f32;
        for r in 0..m {
            let g = f16_to_f32(u16::from_le_bytes([yb[r * 2], yb[r * 2 + 1]]));
            let rf = f16_to_f32(f32_to_f16(dot(&wr16, &xr16, r)));
            mx = mx.max((g - rf).abs());
            mag = mag.max(rf.abs());
        }
        report("f16->f16 ", mx, 3.0 * mag * 2f32.powi(-10) + 1e-4, &mut all_pass);
    }
    // bf16 -> f32
    {
        let y = gpu.upload_raw(&vec![0u8; m * 4], &[m]).unwrap();
        gpu.gemv_bf16_f32(&w_bf16, &x_bf16, &y, m, k).unwrap();
        gpu.device_synchronize().unwrap();
        let yg = gpu.download_f32(&y).unwrap();
        let mut mx = 0.0f32;
        let mut mag = 0.0f32;
        for r in 0..m {
            let rf = dot(&wrb, &xrb, r);
            mx = mx.max((yg[r] - rf).abs());
            mag = mag.max(rf.abs());
        }
        report("bf16->f32", mx, 3.0 * mag * 2f32.powi(-8) + 1e-4, &mut all_pass);
    }
    // bf16 -> bf16
    {
        let y = gpu.upload_raw(&vec![0u8; m * 2], &[m]).unwrap();
        gpu.gemv_bf16_bf16(&w_bf16, &x_bf16, &y, m, k).unwrap();
        gpu.device_synchronize().unwrap();
        let yb = gpu.download_raw(&y, m * 2).unwrap();
        let mut mx = 0.0f32;
        let mut mag = 0.0f32;
        for r in 0..m {
            let g = bf16_to_f32(u16::from_le_bytes([yb[r * 2], yb[r * 2 + 1]]));
            let rf = bf16_to_f32(f32_to_bf16(dot(&wrb, &xrb, r)));
            mx = mx.max((g - rf).abs());
            mag = mag.max(rf.abs());
        }
        report("bf16->bf16", mx, 3.0 * mag * 2f32.powi(-8) + 1e-4, &mut all_pass);
    }
    // iu8 -> i32 (exact)
    {
        let wi = lcg_i8(3, m * k);
        let xi = lcg_i8(4, k);
        let wd = gpu
            .upload_raw(&wi.iter().map(|&v| v as u8).collect::<Vec<_>>(), &[m, k])
            .unwrap();
        let xd = gpu
            .upload_raw(&xi.iter().map(|&v| v as u8).collect::<Vec<_>>(), &[k])
            .unwrap();
        let y = gpu.upload_raw(&vec![0u8; m * 4], &[m]).unwrap();
        gpu.gemv_iu8_i32(&wd, &xd, &y, m, k).unwrap();
        gpu.device_synchronize().unwrap();
        let yb = gpu.download_raw(&y, m * 4).unwrap();
        let mut bad = 0;
        for r in 0..m {
            let g = i32::from_le_bytes([yb[r * 4], yb[r * 4 + 1], yb[r * 4 + 2], yb[r * 4 + 3]]);
            let rf: i64 = (0..k).map(|ki| wi[r * k + ki] as i64 * xi[ki] as i64).sum();
            if g as i64 != rf {
                bad += 1;
            }
        }
        report_int("iu8->i32 ", bad, &mut all_pass);
    }
    // iu4 -> i32 (exact)
    {
        let wi = lcg_i4(5, m * k);
        let xi = lcg_i4(6, k);
        let pack = |v: &[i8], rows: usize| -> Vec<u8> {
            let mut o = vec![0u8; rows * (k / 2)];
            for r in 0..rows {
                for kk in (0..k).step_by(2) {
                    let lo = (v[r * k + kk] as u8) & 0xf;
                    let hi = (v[r * k + kk + 1] as u8) & 0xf;
                    o[r * (k / 2) + kk / 2] = lo | (hi << 4);
                }
            }
            o
        };
        let wd = gpu.upload_raw(&pack(&wi, m), &[m, k / 2]).unwrap();
        let xd = gpu.upload_raw(&pack(&xi, 1), &[k / 2]).unwrap();
        let y = gpu.upload_raw(&vec![0u8; m * 4], &[m]).unwrap();
        gpu.gemv_iu4_i32(&wd, &xd, &y, m, k).unwrap();
        gpu.device_synchronize().unwrap();
        let yb = gpu.download_raw(&y, m * 4).unwrap();
        let mut bad = 0;
        for r in 0..m {
            let g = i32::from_le_bytes([yb[r * 4], yb[r * 4 + 1], yb[r * 4 + 2], yb[r * 4 + 3]]);
            let rf: i64 = (0..k).map(|ki| wi[r * k + ki] as i64 * xi[ki] as i64).sum();
            if g as i64 != rf {
                bad += 1;
            }
        }
        report_int("iu4->i32 ", bad, &mut all_pass);
    }

    if !all_pass {
        std::process::exit(1);
    }
}

fn report(name: &str, max_abs: f32, tol: f32, all_pass: &mut bool) {
    let pass = max_abs <= tol;
    *all_pass &= pass;
    println!(
        "  {name}: max_abs={max_abs:.5} tol={tol:.5} -> {}",
        if pass { "PASS" } else { "FAIL" }
    );
}
fn report_int(name: &str, mismatches: usize, all_pass: &mut bool) {
    let pass = mismatches == 0;
    *all_pass &= pass;
    println!(
        "  {name}: {} -> {}",
        if pass {
            "EXACT".to_string()
        } else {
            format!("{mismatches} mismatches")
        },
        if pass { "PASS" } else { "FAIL" }
    );
}
