// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Correctness channel test: gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1030.
//!
//! Two oracles:
//!   1. Bit-faithful CPU path that emulates quantize_q8_1_mmq_ds4 + int8
//!      Lloyd codebook + signed sdot4. Tight layout/indexing check
//!      (rms_rel < 1e-4 expected on small shapes).
//!   2. Exact f32×codebook quality check (rms_rel < 0.05, matching the
//!      gfx1151 MQ2L MMQ channel-test envelope for Q8_1 noise).
//!
//! Shapes target Qwen3.5 A3B MoE (hidden=2048, mi=512).
//!
//!   HIP_VISIBLE_DEVICES=2 cargo run --release -p rdna-compute \
//!     --example test_mq2g256_lloyd_moe_grouped_mmq_gfx1030

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const WARMUP: usize = 2;
const TRIALS: usize = 10;

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (((bits >> 23) & 0xff) as i32) - 127 + 15;
    let mant = (bits & 0x7fffff) as u32;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as u32;
    if exp == 0 {
        return if sign == 0 { 0.0 } else { -0.0 };
    }
    if exp == 31 {
        return if mant == 0 {
            if sign == 0 {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            }
        } else {
            f32::NAN
        };
    }
    let f_exp = (exp - 15 + 127) as u32;
    f32::from_bits((sign << 31) | (f_exp << 23) | (mant << 13))
}

fn wrap_buf(
    raw_ptr: *mut std::ffi::c_void,
    bytes: usize,
    shape: Vec<usize>,
    dtype: DType,
) -> GpuTensor {
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(raw_ptr, bytes) },
        shape,
        dtype,
    }
}

fn quantize_mq2_lloyd(k: usize, rows: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let groups_per_row = k / 256;
    let mut out = Vec::with_capacity(rows * groups_per_row * 72);
    let mut rng = seed;
    for _ in 0..rows {
        for _ in 0..groups_per_row {
            for &v in &[-3.0f32, -1.0, 1.0, 3.0] {
                out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
            for _ in 0..64 {
                let mut byte = 0u8;
                for nibble in 0..4 {
                    rng = rng
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let idx = ((rng >> 48) & 0x3) as u8;
                    byte |= idx << (nibble * 2);
                }
                out.push(byte);
            }
        }
    }
    out
}

/// Emulate quantize_q8_1_mmq_ds4: per 32-element sub-block,
/// d = amax/127, qs = rint(clamp(x/d, -127, 127)), store sum of original floats.
/// Layout: [K/128][N] of block_q8_1_mmq (144 B each).
struct BlockQ81 {
    d: [f32; 4],     // scale per 32-el sub-block
    sum: [f32; 4],   // sum of original floats (unused by MQ2L, used by HFQ4)
    qs: [i8; 128],
}

fn quantize_q8_1_mmq_cpu(x: &[f32], n_rows: usize, k: usize) -> Vec<BlockQ81> {
    assert_eq!(k % 128, 0);
    let blocks_k = k / 128;
    let mut out = Vec::with_capacity(blocks_k * n_rows);
    for kb in 0..blocks_k {
        for row in 0..n_rows {
            let base = row * k + kb * 128;
            let mut blk = BlockQ81 {
                d: [0.0; 4],
                sum: [0.0; 4],
                qs: [0; 128],
            };
            for sb in 0..4 {
                let off = sb * 32;
                let mut amax = 0f32;
                let mut sum = 0f32;
                for t in 0..32 {
                    let v = x[base + off + t];
                    amax = amax.max(v.abs());
                    sum += v;
                }
                let d = if amax > 1e-20 { amax / 127.0 } else { 0.0 };
                let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                blk.d[sb] = d;
                blk.sum[sb] = sum;
                for t in 0..32 {
                    let q = (x[base + off + t] * id).round().clamp(-127.0, 127.0) as i32;
                    blk.qs[off + t] = q as i8;
                }
            }
            out.push(blk);
        }
    }
    out
}

/// Bit-faithful oracle: Q8_1 X + int8 Lloyd codebook + signed dots.
/// Y layout [m_total × M] matching the kernel.
fn cpu_oracle_q8_mq2l(
    weights: &[u8],
    x_q8: &[BlockQ81],
    m: usize,
    k: usize,
    m_total: usize,
    n_rows: usize,
) -> Vec<f32> {
    let groups = k / 256;
    let mut y = vec![0f32; m_total * m];
    for col in 0..m_total {
        // x_row_div=1 identity: x_row = col (live slots only; col < n_rows)
        if col >= n_rows {
            continue;
        }
        for row in 0..m {
            let mut acc = 0f32;
            let row_base = row * groups * 72;
            for g in 0..groups {
                let gp = row_base + g * 72;
                let mut cb = [0f32; 4];
                for i in 0..4 {
                    let h = u16::from_le_bytes([weights[gp + 2 * i], weights[gp + 2 * i + 1]]);
                    cb[i] = f16_bits_to_f32(h);
                }
                let amax = cb.iter().map(|v| v.abs()).fold(0f32, f32::max);
                let sc = amax / 127.0;
                let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                let mut cb_i8 = [0i32; 4];
                for i in 0..4 {
                    cb_i8[i] = (cb[i] * inv).round() as i32;
                }
                let data = &weights[gp + 8..gp + 72];
                // 2 × 128-K windows × 4 sub-blocks of 32
                for kb_in_group in 0..2 {
                    let kb_q8 = 2 * g + kb_in_group;
                    let blk = &x_q8[kb_q8 * n_rows + col];
                    for sb in 0..4 {
                        let d_x = blk.d[sb];
                        let mut sumi: i32 = 0;
                        let base_t = kb_in_group * 128 + sb * 32;
                        for t in 0..32 {
                            let byte = data[(base_t + t) / 4];
                            let q = ((byte >> (2 * ((base_t + t) % 4))) & 0x3) as usize;
                            let a = cb_i8[q];
                            let b = blk.qs[sb * 32 + t] as i32;
                            sumi += a * b;
                        }
                        acc += sc * d_x * (sumi as f32);
                    }
                }
            }
            y[col * m + row] = acc;
        }
    }
    y
}

/// Exact f32×codebook (quality envelope; ignores Q8_1).
fn cpu_oracle_f32_mq2l(weights: &[u8], x: &[f32], m: usize, k: usize, m_total: usize) -> Vec<f32> {
    let groups = k / 256;
    let mut y = vec![0f32; m_total * m];
    for col in 0..m_total {
        let x_row = &x[col * k..(col + 1) * k];
        for row in 0..m {
            let mut acc = 0f32;
            let row_base = row * groups * 72;
            for g in 0..groups {
                let gp = row_base + g * 72;
                let mut cb = [0f32; 4];
                for i in 0..4 {
                    let h = u16::from_le_bytes([weights[gp + 2 * i], weights[gp + 2 * i + 1]]);
                    cb[i] = f16_bits_to_f32(h);
                }
                let data = &weights[gp + 8..gp + 72];
                for t in 0..256 {
                    let byte = data[t / 4];
                    let q = ((byte >> (2 * (t % 4))) & 0x3) as usize;
                    acc += cb[q] * x_row[g * 256 + t];
                }
            }
            y[col * m + row] = acc;
        }
    }
    y
}

fn rms_rel(got: &[f32], exp: &[f32], n: usize) -> (f64, usize) {
    let mut num = 0f64;
    let mut den = 0f64;
    let mut nan = 0usize;
    for i in 0..n {
        if !got[i].is_finite() {
            nan += 1;
            continue;
        }
        let d = (got[i] - exp[i]) as f64;
        num += d * d;
        den += (exp[i] as f64).powi(2);
    }
    ((num / den.max(1e-12)).sqrt(), nan)
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("Arch: {}", gpu.arch);
    assert_eq!(gpu.arch, "gfx1030", "this channel test is exact-gfx1030 only");

    const TOP_K: usize = 8;
    let shapes: &[(usize, usize, usize, &str)] = &[
        (1024, 2048, 16, "gate/up B=16"),
        (1024, 2048, 64, "gate/up B=64"),
        (1024, 2048, 256, "gate/up B=256"),
        (2048, 512, 16, "down B=16"),
        (2048, 512, 64, "down B=64"),
        (2048, 512, 256, "down B=256"),
    ];

    let mut all_ok = true;
    for &(m, k, batch, label) in shapes {
        let m_total = batch * TOP_K;
        let m_total_pad = ((m_total + 15) / 16) * 16;
        println!("\n=== {label} | M={m} K={k} batch={batch} m_total={m_total} pad={m_total_pad} ===");
        if m % 16 != 0 || k % 256 != 0 {
            println!("  SKIP shape");
            continue;
        }

        let weight_bytes = quantize_mq2_lloyd(k, m, 0xA3Bu64);
        let x_f32: Vec<f32> = (0..m_total * k)
            .map(|i| ((i % 11) as f32 - 5.0) / 5.0)
            .collect();
        let x_f32_bytes: Vec<u8> = x_f32.iter().flat_map(|v| v.to_le_bytes()).collect();

        let w_gpu = gpu.hip.malloc(weight_bytes.len()).expect("malloc W");
        let x_gpu = gpu.hip.malloc(x_f32_bytes.len()).expect("malloc X");
        let y_gpu = gpu.hip.malloc(m_total_pad * m * 4).expect("malloc Y");
        gpu.hip.memcpy_htod(&w_gpu, &weight_bytes).expect("htod W");
        gpu.hip.memcpy_htod(&x_gpu, &x_f32_bytes).expect("htod X");
        let zeros = vec![0u8; m_total_pad * m * 4];
        gpu.hip.memcpy_htod(&y_gpu, &zeros).expect("htod Y0");

        let w_ptr_u64 = w_gpu.as_ptr() as u64;
        let ep_bytes = w_ptr_u64.to_le_bytes().to_vec();
        let ep_gpu = gpu.hip.malloc(8).expect("malloc EP");
        gpu.hip.memcpy_htod(&ep_gpu, &ep_bytes).expect("htod EP");

        let slot_tiles = m_total_pad / 16;
        let tile_ids_bytes: Vec<u8> = (0..slot_tiles)
            .flat_map(|_| 0i32.to_le_bytes().to_vec())
            .collect();
        let tp_gpu = gpu.hip.malloc(tile_ids_bytes.len()).expect("malloc TP");
        gpu.hip.memcpy_htod(&tp_gpu, &tile_ids_bytes).expect("htod TP");

        let perm_bytes: Vec<u8> = (0..m_total_pad)
            .flat_map(|i| {
                let v = if i < m_total { i as i32 } else { -1i32 };
                v.to_le_bytes().to_vec()
            })
            .collect();
        let sp_gpu = gpu.hip.malloc(perm_bytes.len()).expect("malloc SP");
        gpu.hip.memcpy_htod(&sp_gpu, &perm_bytes).expect("htod SP");

        let ep_t = wrap_buf(ep_gpu.as_ptr(), 8, vec![1], DType::F32);
        let tp_t = wrap_buf(tp_gpu.as_ptr(), tile_ids_bytes.len(), vec![slot_tiles], DType::F32);
        let sp_t = wrap_buf(sp_gpu.as_ptr(), perm_bytes.len(), vec![m_total_pad], DType::F32);
        let x_t = wrap_buf(x_gpu.as_ptr(), x_f32_bytes.len(), vec![m_total, k], DType::F32);
        let y_t = wrap_buf(y_gpu.as_ptr(), m_total_pad * m * 4, vec![m_total_pad, m], DType::F32);

        gpu.gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1030(
            &ep_t, &tp_t, &sp_t, &x_t, &y_t, m, k, 1, m_total_pad, m_total,
        )
        .expect("kernel launch");
        gpu.hip.device_synchronize().expect("sync");

        let mut y_bytes = vec![0u8; m_total_pad * m * 4];
        gpu.hip.memcpy_dtoh(&mut y_bytes, &y_gpu).expect("dtoh Y");
        let y_gpu_f: Vec<f32> = y_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // live slice only
        let mut y_live = vec![0f32; m_total * m];
        for col in 0..m_total {
            for row in 0..m {
                y_live[col * m + row] = y_gpu_f[col * m + row];
            }
        }

        // (1) bit-faithful Q8_1 oracle
        let x_q8 = quantize_q8_1_mmq_cpu(&x_f32, m_total, k);
        let y_q8 = cpu_oracle_q8_mq2l(&weight_bytes, &x_q8, m, k, m_total, m_total);
        let (rms_q8, nan_q8) = rms_rel(&y_live, &y_q8, m_total * m);

        // (2) quality envelope vs exact f32 codebook
        let y_f32 = cpu_oracle_f32_mq2l(&weight_bytes, &x_f32, m, k, m_total);
        let (rms_f32, nan_f32) = rms_rel(&y_live, &y_f32, m_total * m);

        // Layout gate: bit-faithful should be near-zero (fp32 accum order only).
        // Q8_1 d is stored as f16 in the real kernel so allow a little slack.
        let layout_ok = nan_q8 == 0 && rms_q8 < 5e-3;
        // Quality gate: match gfx1151 MQ2L MMQ envelope.
        let quality_ok = nan_f32 == 0 && rms_f32 < 0.05;
        let ok = layout_ok && quality_ok;
        println!(
            "  rms_q8={rms_q8:.6} (layout)  rms_f32={rms_f32:.4} (quality)  \
             nan_q8={nan_q8} nan_f32={nan_f32} {}",
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            all_ok = false;
            // dump a few mismatches for forensics
            let mut shown = 0;
            for i in 0..(m_total * m) {
                if (y_live[i] - y_q8[i]).abs() > 1e-2 {
                    println!(
                        "    mismatch i={i} gpu={:.6} q8={:.6} f32={:.6}",
                        y_live[i], y_q8[i], y_f32[i]
                    );
                    shown += 1;
                    if shown >= 4 {
                        break;
                    }
                }
            }
        }

        for _ in 0..WARMUP {
            gpu.gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1030(
                &ep_t, &tp_t, &sp_t, &x_t, &y_t, m, k, 1, m_total_pad, m_total,
            )
            .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..TRIALS {
            gpu.gemm_mq2g256_lloyd_moe_grouped_mmq_gfx1030(
                &ep_t, &tp_t, &sp_t, &x_t, &y_t, m, k, 1, m_total_pad, m_total,
            )
            .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() / TRIALS as f64 * 1e6;
        let flops = 2.0 * m as f64 * k as f64 * m_total as f64;
        let gflops = flops / us / 1e3;
        println!("  {us:>8.1} µs  {gflops:>7.0} GFLOPS");

        std::mem::forget(ep_t);
        std::mem::forget(tp_t);
        std::mem::forget(sp_t);
        std::mem::forget(x_t);
        std::mem::forget(y_t);
    }

    if all_ok {
        println!("\nALL SHAPES OK");
    } else {
        println!("\nSOME SHAPES FAILED");
        std::process::exit(1);
    }
}
