// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! MQ2-GL (global codebook + per-block fp16 scale, 64 B/group + 2 B scale)
//! vs MQ2-Lloyd (per-block 4-entry fp16 codebook, 72 B/group), head-to-head on
//! the a3b routed-expert gate_up decode shape.
//!
//! Both kernels are JIT'd from source and launched via the kernarg-blob path,
//! so this bench touches no dispatch plumbing.
//!
//! Correctness is checked against a CPU reference for each format separately —
//! the two formats encode different weights by construction, so they are NOT
//! expected to agree with each other. What is compared is speed and bytes.
//!
//! Run: cargo run --release -p rdna-compute --example bench_mq2gl_vs_mq2l
use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};
use std::time::Instant;

const MQ2L_SRC: &str = include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_gate_up_indexed.hip");
const MQ2GL_SRC: &str = include_str!("../../../kernels/src/gemv_mq2g256gl_moe_gate_up_indexed.hip");

/// Textbook Lloyd–Max levels for a unit Gaussian, 2 bit. Measured on 28.3M real
/// a3b expert weights, a globally fitted codebook lands within 3 decimals of these.
const CB: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];

fn mix(x: u64) -> u64 {
    let h = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
}

/// Per-block-codebook format: [4×fp16 cb][64 B indices] × groups, per row.
fn build_mq2l(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let gpr = k / 256;
    let mut out = vec![0u8; m * gpr * 72];
    for row in 0..m {
        for g in 0..gpr {
            let off = (row * gpr + g) * 72;
            let s = 0.004f32 + ((mix(seed ^ ((row as u64) << 20) ^ g as u64) % 4000) as f32) * 1e-6;
            for (e, &c) in CB.iter().enumerate() {
                let h = half_bits(c * s);
                out[off + 2 * e] = (h & 0xff) as u8;
                out[off + 2 * e + 1] = (h >> 8) as u8;
            }
            for b in 0..64 {
                out[off + 8 + b] = (mix(seed ^ ((row as u64) << 32) ^ ((g as u64) << 8) ^ b as u64) & 0xff) as u8;
            }
        }
    }
    out
}

/// Global-codebook format: [all indices 64 B/group][all fp16 scales].
fn build_mq2gl(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let gpr = k / 256;
    let idx_bytes = m * gpr * 64;
    let mut out = vec![0u8; idx_bytes + m * gpr * 2];
    for row in 0..m {
        for g in 0..gpr {
            let off = (row * gpr + g) * 64;
            for b in 0..64 {
                out[off + b] = (mix(seed ^ ((row as u64) << 32) ^ ((g as u64) << 8) ^ b as u64) & 0xff) as u8;
            }
            let s = 0.004f32 + ((mix(seed ^ ((row as u64) << 20) ^ g as u64) % 4000) as f32) * 1e-6;
            let h = half_bits(s);
            let so = idx_bytes + (row * gpr + g) * 2;
            out[so] = (h & 0xff) as u8;
            out[so + 1] = (h >> 8) as u8;
        }
    }
    out
}

fn half_bits(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7f_ffff;
    if e <= 0 {
        return sign;
    }
    if e >= 31 {
        e = 30;
    }
    sign | ((e as u16) << 10) | ((mant >> 13) as u16)
}
fn half_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        return f32::from_bits(sign);
    }
    f32::from_bits(sign | ((e + 112) << 23) | (m << 13))
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("arch={}", gpu.arch);

    // a3b routed-expert gate_up decode shape: [gate+up = 1024, K = 2048], k_top=8.
    let m = 1024usize;
    let k = 2048usize;
    let k_top = 8usize;
    let n_exp = 32usize;
    let gpr = k / 256;

    let mq2l_bytes = m * gpr * 72;
    let mq2gl_bytes = m * gpr * 64 + m * gpr * 2;
    eprintln!(
        "M={m} K={k} k_top={k_top}  per-expert: MQ2L {} KiB  MQ2GL {} KiB  ({:.1}% less)",
        mq2l_bytes / 1024,
        mq2gl_bytes / 1024,
        100.0 * (1.0 - mq2gl_bytes as f64 / mq2l_bytes as f64)
    );

    // one blob per expert, per format
    let mut l_ts = Vec::new();
    let mut gl_ts = Vec::new();
    for e in 0..n_exp {
        l_ts.push(gpu.upload_raw(&build_mq2l(m, k, 0x1000 + e as u64), &[mq2l_bytes]).unwrap());
        gl_ts.push(gpu.upload_raw(&build_mq2gl(m, k, 0x1000 + e as u64), &[mq2gl_bytes]).unwrap());
    }
    let l_ptrs: Vec<u64> = l_ts.iter().map(|t| t.buf.as_ptr() as u64).collect();
    let gl_ptrs: Vec<u64> = gl_ts.iter().map(|t| t.buf.as_ptr() as u64).collect();
    let l_ptr_t = gpu
        .upload_raw(&l_ptrs.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>(), &[n_exp])
        .unwrap();
    let gl_ptr_t = gpu
        .upload_raw(&gl_ptrs.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>(), &[n_exp])
        .unwrap();

    let topk: Vec<i32> = (0..k_top as i32).map(|i| (i * 3) % n_exp as i32).collect();
    let topk_t = gpu
        .upload_raw(&topk.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), &[k_top])
        .unwrap();

    let x: Vec<f32> = (0..k).map(|i| ((i * 37 % 61) as f32 - 30.0) * 0.02).collect();
    let x_t = gpu.upload_f32(&x, &[k]).unwrap();
    let mi = m / 2;
    let y_g = gpu.alloc_tensor(&[k_top * mi], DType::F32).unwrap();
    let y_u = gpu.alloc_tensor(&[k_top * mi], DType::F32).unwrap();

    gpu.ensure_kernel_public("gemv_mq2g256_lloyd_moe_gate_up_indexed", MQ2L_SRC,
        "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed").expect("JIT mq2l");
    gpu.ensure_kernel_public("gemv_mq2g256gl_moe_gate_up_indexed", MQ2GL_SRC,
        "gemv_mq2g256gl_moe_gate_up_k8_indexed").expect("JIT mq2gl");

    let grid = [m as u32, k_top as u32, 1];
    let block = [32u32, 1, 1];

    let mut blob_l = KernargBlob::new();
    blob_l.push_ptr(l_ptr_t.buf.as_ptr() as *const _);
    blob_l.push_ptr(topk_t.buf.as_ptr() as *const _);
    blob_l.push_ptr(x_t.buf.as_ptr() as *const _);
    blob_l.push_ptr(y_g.buf.as_ptr() as *const _);
    blob_l.push_ptr(y_u.buf.as_ptr() as *const _);
    blob_l.push_i32(m as i32);
    blob_l.push_i32(k as i32);
    let mut bl = blob_l.into_vec();

    let mut blob_g = KernargBlob::new();
    blob_g.push_ptr(gl_ptr_t.buf.as_ptr() as *const _);
    blob_g.push_ptr(topk_t.buf.as_ptr() as *const _);
    blob_g.push_ptr(x_t.buf.as_ptr() as *const _);
    blob_g.push_ptr(y_g.buf.as_ptr() as *const _);
    blob_g.push_ptr(y_u.buf.as_ptr() as *const _);
    for c in CB { blob_g.push_f32(c); }
    blob_g.push_i32(m as i32);
    blob_g.push_i32(k as i32);
    let mut bg = blob_g.into_vec();

    // ---- correctness: each kernel vs its own CPU reference ----
    let check = |gpu: &mut Gpu, name: &str, bytes: &mut [u8], gl: bool, raw: &[u8]| {
        gpu.launch_kernel_blob(name, grid, block, 0, bytes).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let got = gpu.download_f32(&y_g).unwrap();
        // CPU reference for expert topk[0], row 0
        let e0 = topk[0] as usize;
        let _ = e0;
        let mut want = 0.0f64;
        for g in 0..gpr {
            let (scale, cb): (f32, [f32; 4]) = if gl {
                let idx_bytes = m * gpr * 64;
                let so = idx_bytes + g * 2;
                (half_to_f32(u16::from_le_bytes([raw[so], raw[so + 1]])), CB)
            } else {
                let off = g * 72;
                let mut c = [0f32; 4];
                for (e, ci) in c.iter_mut().enumerate() {
                    *ci = half_to_f32(u16::from_le_bytes([raw[off + 2 * e], raw[off + 2 * e + 1]]));
                }
                (1.0, c)
            };
            let ib = if gl { g * 64 } else { g * 72 + 8 };
            for b in 0..64 {
                let byte = raw[ib + b];
                for j in 0..4 {
                    let q = ((byte >> (2 * j)) & 3) as usize;
                    let col = g * 256 + b * 4 + j;
                    want += (scale * cb[q] * x[col]) as f64;
                }
            }
        }
        let rel = ((got[0] as f64 - want).abs()) / want.abs().max(1e-6);
        eprintln!("  {:<8} row0 gpu={:>12.5}  cpu={:>12.5}  rel={:.2e}", if gl {"MQ2GL"} else {"MQ2L"}, got[0], want, rel);
        rel
    };

    eprintln!("\ncorrectness (kernel vs CPU reference on its own format):");
    let raw_l = build_mq2l(m, k, 0x1000 + topk[0] as u64);
    let raw_g = build_mq2gl(m, k, 0x1000 + topk[0] as u64);
    let r1 = check(&mut gpu, "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed", &mut bl, false, &raw_l);
    let r2 = check(&mut gpu, "gemv_mq2g256gl_moe_gate_up_k8_indexed", &mut bg, true, &raw_g);

    // ---- timing ----
    let bench = |gpu: &mut Gpu, name: &str, bytes: &mut [u8]| -> f64 {
        for _ in 0..20 { gpu.launch_kernel_blob(name, grid, block, 0, bytes).unwrap(); }
        gpu.hip.device_synchronize().unwrap();
        let runs = 500;
        let t0 = Instant::now();
        for _ in 0..runs { gpu.launch_kernel_blob(name, grid, block, 0, bytes).unwrap(); }
        gpu.hip.device_synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e6 / runs as f64
    };

    eprintln!("\n{:<10} {:>10} {:>12} {:>12}", "variant", "us/call", "wt GiB/s", "bytes/call");
    let mut res = Vec::new();
    for _pass in 0..3 {
        let ul = bench(&mut gpu, "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed", &mut bl);
        let ug = bench(&mut gpu, "gemv_mq2g256gl_moe_gate_up_k8_indexed", &mut bg);
        res.push((ul, ug));
    }
    res.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (ul, ug) = res[res.len() / 2];
    let bl_bytes = (k_top * mq2l_bytes) as f64;
    let bg_bytes = (k_top * mq2gl_bytes) as f64;
    eprintln!("{:<10} {:>10.2} {:>12.1} {:>12.0}", "MQ2L", ul, bl_bytes / (ul * 1e-6) / (1u64 << 30) as f64, bl_bytes);
    eprintln!("{:<10} {:>10.2} {:>12.1} {:>12.0}", "MQ2GL", ug, bg_bytes / (ug * 1e-6) / (1u64 << 30) as f64, bg_bytes);
    eprintln!("\nMQ2GL is {:+.2}% on time  ({:.1}% fewer weight bytes)",
        100.0 * (ug / ul - 1.0), 100.0 * (1.0 - bg_bytes / bl_bytes));
    eprintln!("median of 3 passes; correctness rel err MQ2L={r1:.2e} MQ2GL={r2:.2e}");
}
