// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 2 — GPU E4M3/UE8M0 and E2M1/UE8M0 decode vs the independent CPU oracle.
//!
//! Proves the four gfx942 parent-checkpoint kernels execute correctly on real
//! MI300X silicon by bit-exact comparison against `parent::codec`:
//!
//! - `dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942`
//! - `dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942`
//! - `act_quant_fp8_ue8m0_inplace_gfx942` (block ∈ {64, 128})
//! - `act_quant_fp4_ue8m0_g32_inplace_gfx942`
//!
//! Plus a real-checkpoint sample path over
//! `/mnt/scratch/models/DeepSeek-V4-Flash-0731`.
//!
//! Usage (on mi300x):
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_codec_gate \
//!   [--ckpt-dir /mnt/scratch/models/DeepSeek-V4-Flash-0731] \
//!   [--tensor-dir /tmp/ds4_codec_gate_tensors]
//! ```
//!
//! Exit code 0 only if every check PASSes. Failures print index, inputs,
//! CPU value, and GPU value — never "close enough".

use hipfire_arch_deepseek4::parent::codec::{
    act_quant_fp4_inplace_ref, act_quant_fp8_inplace_ref, dequant_dense_fp8_block128,
    dequant_expert_fp4_g32, e2m1_to_f32, e4m3_to_f32, fast_round_scale, round_to_bf16,
    ue8m0_to_f32, E2M1_LUT, FP4_E2M1_MAX, FP8_E4M3_MAX,
};
use rdna_compute::{Gpu, GpuTensor};
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

// ── BF16 host helpers ────────────────────────────────────────────────────────

#[inline]
fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        let sign = ((bits >> 16) & 0x8000) as u16;
        return sign | 0x7fc0;
    }
    let lsb = (bits >> 16) & 1;
    let lower = bits & 0xffff;
    let round_bit = (lower >> 15) & 1;
    let sticky = if (lower & 0x7fff) != 0 { 1 } else { 0 };
    let mut top = bits >> 16;
    if round_bit == 1 && (sticky == 1 || lsb == 1) {
        top = top.wrapping_add(1);
    }
    top as u16
}

#[inline]
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn pack_f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
    }
    out
}

fn unpack_bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 2 == 0);
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let b = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(bf16_bits_to_f32(b));
    }
    out
}

fn download_bf16_f32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Result<Vec<f32>, String> {
    let mut bytes = vec![0u8; n * 2];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &t.buf)
        .map_err(|e| format!("memcpy_dtoh: {e:?}"))?;
    Ok(unpack_bf16_bytes_to_f32(&bytes))
}

fn upload_bytes(gpu: &Gpu, data: &[u8]) -> Result<GpuTensor, String> {
    gpu.upload_raw(data, &[data.len()])
        .map_err(|e| format!("upload_raw: {e:?}"))
}

fn free(gpu: &mut Gpu, t: GpuTensor) {
    let _ = gpu.free_tensor(t);
}

// ── Check bookkeeping ────────────────────────────────────────────────────────

struct Check {
    name: &'static str,
    part: char,
    ok: bool,
    detail: String,
}

struct Gate {
    checks: Vec<Check>,
}

impl Gate {
    fn new() -> Self {
        Self { checks: Vec::new() }
    }

    fn record(&mut self, part: char, name: &'static str, ok: bool, detail: impl Into<String>) {
        let detail = detail.into();
        if ok {
            println!("  PASS  [{part}] {name}");
        } else {
            println!("  FAIL  [{part}] {name}");
            if !detail.is_empty() {
                println!("        {detail}");
            }
        }
        self.checks.push(Check {
            name,
            part,
            ok,
            detail,
        });
    }

    fn all_ok(&self) -> bool {
        self.checks.iter().all(|c| c.ok)
    }

    fn print_table(&self) {
        println!();
        println!("════════════════════════════════════════════════════════════");
        println!(" Gate 2 codec summary");
        println!("════════════════════════════════════════════════════════════");
        println!(
            "{:<6} {:<4} {:<48} {}",
            "STATUS", "PART", "CHECK", "DETAIL"
        );
        println!("{}", "─".repeat(96));
        for c in &self.checks {
            let st = if c.ok { "PASS" } else { "FAIL" };
            let det = if c.detail.is_empty() {
                String::new()
            } else {
                // single-line truncate
                let d = c.detail.replace('\n', " | ");
                if d.len() > 40 {
                    format!("{}…", &d[..37])
                } else {
                    d
                }
            };
            println!("{st:<6} {part:<4} {name:<48} {det}", part = c.part, name = c.name);
        }
        let n_pass = self.checks.iter().filter(|c| c.ok).count();
        let n_fail = self.checks.len() - n_pass;
        println!("{}", "─".repeat(96));
        println!(
            "total={}  pass={}  fail={}",
            self.checks.len(),
            n_pass,
            n_fail
        );
    }
}

/// Compare two f32 slices for bit-exact BF16-domain agreement.
/// Finite values must match bits exactly after both are viewed as the BF16
/// payload already written (caller is responsible for BF16 rounding on CPU).
/// NaN matches NaN (payload may differ); Inf matches Inf with sign.
fn compare_bf16_domain(
    cpu: &[f32],
    gpu: &[f32],
    label: &str,
    max_print: usize,
) -> Result<(), String> {
    if cpu.len() != gpu.len() {
        return Err(format!(
            "{label}: length mismatch cpu={} gpu={}",
            cpu.len(),
            gpu.len()
        ));
    }
    let mut mismatches = 0usize;
    let mut first: Option<String> = None;
    for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let ok = if c.is_nan() && g.is_nan() {
            true
        } else if c.to_bits() == g.to_bits() {
            true
        } else {
            false
        };
        if !ok {
            mismatches += 1;
            if first.is_none() {
                first = Some(format!(
                    "idx={i}: cpu={c:?} (bits=0x{:08x}) gpu={g:?} (bits=0x{:08x})",
                    c.to_bits(),
                    g.to_bits()
                ));
            }
            if mismatches >= max_print && first.is_some() {
                // keep counting
            }
        }
    }
    if mismatches == 0 {
        Ok(())
    } else {
        Err(format!(
            "{label}: {mismatches}/{} mismatches; first: {}",
            cpu.len(),
            first.unwrap_or_default()
        ))
    }
}

// ── Part A: dense FP8 dequant ────────────────────────────────────────────────

fn part_a_exhaustive(gpu: &mut Gpu, gate: &mut Gate) -> Result<(), String> {
    const M: usize = 256;
    const K: usize = 256;
    // Weight covers all 256 E4M3 codes many times (256*256 / 256 = 256× each).
    let mut w = vec![0u8; M * K];
    for i in 0..M * K {
        w[i] = (i % 256) as u8;
    }
    // Scale grid is [ceil(256/128), ceil(256/128)] = [2, 2].
    // Wide UE8M0 spread including extremes (not 0xFF — that is the NaN case).
    let s = vec![
        100u8, // 2^-27
        120,   // 2^-7
        127,   // 2^0
        140,   // 2^13
    ];
    assert_eq!(s.len(), 2 * 2);

    let mut cpu = vec![0f32; M * K];
    dequant_dense_fp8_block128(&w, &s, M, K, &mut cpu).map_err(|e| e)?;
    // GPU writes BF16; round the CPU product the same way.
    for v in cpu.iter_mut() {
        *v = round_to_bf16(*v);
    }

    let d_w = upload_bytes(gpu, &w)?;
    let d_s = upload_bytes(gpu, &s)?;
    let out_bytes = M * K * 2;
    let d_out = upload_bytes(gpu, &vec![0u8; out_bytes])?;

    gpu.dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942(&d_w.buf, &d_s.buf, &d_out.buf, M, K)
        .map_err(|e| format!("dequant_fp8 exhaustive launch: {e:?}"))?;

    let gpu_out = download_bf16_f32(gpu, &d_out, M * K)?;
    free(gpu, d_w);
    free(gpu, d_s);
    free(gpu, d_out);

    match compare_bf16_domain(&cpu, &gpu_out, "A-exhaustive", 5) {
        Ok(()) => {
            gate.record(
                'A',
                "fp8_dequant_exhaustive_256x256",
                true,
                format!("{} elems bit-exact", M * K),
            );
            Ok(())
        }
        Err(e) => {
            // Loud first-mismatch dump with inputs.
            let mut extra = e.clone();
            for i in 0..M * K {
                if cpu[i].to_bits() != gpu_out[i].to_bits()
                    && !(cpu[i].is_nan() && gpu_out[i].is_nan())
                {
                    let m = i / K;
                    let k = i % K;
                    let wb = w[i];
                    let sb = s[(m / 128) * 2 + (k / 128)];
                    extra = format!(
                        "{e} | first raw: m={m} k={k} W=0x{wb:02x} S=0x{sb:02x} \
                         e4m3={:?} scale={:?} product_f32={:?} cpu_bf16={:?} gpu_bf16={:?}",
                        e4m3_to_f32(wb),
                        ue8m0_to_f32(sb),
                        e4m3_to_f32(wb) * ue8m0_to_f32(sb),
                        cpu[i],
                        gpu_out[i]
                    );
                    break;
                }
            }
            gate.record('A', "fp8_dequant_exhaustive_256x256", false, extra);
            Ok(())
        }
    }
}

fn part_a_ragged(gpu: &mut Gpu, gate: &mut Gate) -> Result<(), String> {
    // Ragged in BOTH dimensions against the 128×128 scale grid.
    const M: usize = 260;
    const K: usize = 300;
    let bm = M.div_ceil(128); // 3
    let bk = K.div_ceil(128); // 3
    let mut w = vec![0u8; M * K];
    // Deterministic but varied codes.
    for i in 0..M * K {
        w[i] = ((i.wrapping_mul(131) + 17) % 256) as u8;
        // Avoid NaN codes in the ragged sweep; NaN is a dedicated case.
        if w[i] == 0x7f || w[i] == 0xff {
            w[i] = 0x3c; // 1.0
        }
    }
    let mut s = vec![0u8; bm * bk];
    for i in 0..bm * bk {
        // Spread of finite UE8M0, never 0xFF.
        s[i] = (110 + (i as u8).wrapping_mul(3)) % 200;
        if s[i] == 0 {
            s[i] = 1;
        }
    }

    let mut cpu = vec![0f32; M * K];
    dequant_dense_fp8_block128(&w, &s, M, K, &mut cpu)?;
    for v in cpu.iter_mut() {
        *v = round_to_bf16(*v);
    }

    let d_w = upload_bytes(gpu, &w)?;
    let d_s = upload_bytes(gpu, &s)?;
    let d_out = upload_bytes(gpu, &vec![0u8; M * K * 2])?;
    gpu.dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942(&d_w.buf, &d_s.buf, &d_out.buf, M, K)
        .map_err(|e| format!("dequant_fp8 ragged launch: {e:?}"))?;
    let gpu_out = download_bf16_f32(gpu, &d_out, M * K)?;
    free(gpu, d_w);
    free(gpu, d_s);
    free(gpu, d_out);

    match compare_bf16_domain(&cpu, &gpu_out, "A-ragged", 5) {
        Ok(()) => {
            gate.record(
                'A',
                "fp8_dequant_ragged_260x300",
                true,
                format!("scale grid {bm}x{bk}, {n} elems", n = M * K),
            );
        }
        Err(e) => {
            // Highlight a cell in a trailing partial block (m>=256 or k>=256).
            let mut extra = e.clone();
            for m in [0usize, 128, 256, 259] {
                for k in [0usize, 128, 256, 299] {
                    let i = m * K + k;
                    if i < cpu.len()
                        && cpu[i].to_bits() != gpu_out[i].to_bits()
                        && !(cpu[i].is_nan() && gpu_out[i].is_nan())
                    {
                        let wb = w[i];
                        let sb = s[(m / 128) * bk + (k / 128)];
                        extra = format!(
                            "{e} | corner m={m} k={k} W=0x{wb:02x} S=0x{sb:02x} \
                             cpu={:?} gpu={:?}",
                            cpu[i], gpu_out[i]
                        );
                        break;
                    }
                }
            }
            gate.record('A', "fp8_dequant_ragged_260x300", false, extra);
        }
    }
    Ok(())
}

fn part_a_nan(gpu: &mut Gpu, gate: &mut Gate) -> Result<(), String> {
    // Tiny 128×128 so one scale cell covers the whole tile.
    const M: usize = 128;
    const K: usize = 128;
    // Case 1: scale byte 0xFF → every output in the block must be NaN.
    {
        let mut w = vec![0x3cu8; M * K]; // 1.0 codes
        w[0] = 0x00;
        w[1] = 0x3c;
        let s = vec![0xffu8]; // NaN scale
        let mut cpu = vec![0f32; M * K];
        dequant_dense_fp8_block128(&w, &s, M, K, &mut cpu)?;
        for v in cpu.iter_mut() {
            *v = round_to_bf16(*v);
        }
        let d_w = upload_bytes(gpu, &w)?;
        let d_s = upload_bytes(gpu, &s)?;
        let d_out = upload_bytes(gpu, &vec![0u8; M * K * 2])?;
        gpu.dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942(&d_w.buf, &d_s.buf, &d_out.buf, M, K)
            .map_err(|e| format!("nan-scale launch: {e:?}"))?;
        let gpu_out = download_bf16_f32(gpu, &d_out, M * K)?;
        free(gpu, d_w);
        free(gpu, d_s);
        free(gpu, d_out);

        let cpu_all_nan = cpu.iter().all(|v| v.is_nan());
        let gpu_all_nan = gpu_out.iter().all(|v| v.is_nan());
        let ok = cpu_all_nan && gpu_all_nan;
        let detail = if ok {
            format!("all {n} outputs NaN (scale=0xFF)", n = M * K)
        } else {
            format!(
                "cpu_all_nan={cpu_all_nan} gpu_all_nan={gpu_all_nan} \
                 cpu[0]={:?} gpu[0]={:?} cpu_nan_frac={:.4} gpu_nan_frac={:.4}",
                cpu[0],
                gpu_out[0],
                cpu.iter().filter(|v| v.is_nan()).count() as f32 / cpu.len() as f32,
                gpu_out.iter().filter(|v| v.is_nan()).count() as f32 / gpu_out.len() as f32,
            )
        };
        gate.record('A', "fp8_dequant_nan_scale_0xFF", ok, detail);
    }

    // Case 2: E4M3 NaN codes 0x7F / 0xFF with finite scale → NaN outputs.
    {
        let mut w = vec![0x3cu8; M * K];
        w[0] = 0x7f; // +NaN
        w[1] = 0xff; // -NaN
        w[2] = 0x3c; // finite control
        let s = vec![127u8]; // scale = 1.0
        let mut cpu = vec![0f32; M * K];
        dequant_dense_fp8_block128(&w, &s, M, K, &mut cpu)?;
        for v in cpu.iter_mut() {
            *v = round_to_bf16(*v);
        }
        let d_w = upload_bytes(gpu, &w)?;
        let d_s = upload_bytes(gpu, &s)?;
        let d_out = upload_bytes(gpu, &vec![0u8; M * K * 2])?;
        gpu.dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942(&d_w.buf, &d_s.buf, &d_out.buf, M, K)
            .map_err(|e| format!("nan-e4m3 launch: {e:?}"))?;
        let gpu_out = download_bf16_f32(gpu, &d_out, M * K)?;
        free(gpu, d_w);
        free(gpu, d_s);
        free(gpu, d_out);

        let ok0 = cpu[0].is_nan() && gpu_out[0].is_nan();
        let ok1 = cpu[1].is_nan() && gpu_out[1].is_nan();
        let ok2 = !cpu[2].is_nan()
            && !gpu_out[2].is_nan()
            && cpu[2].to_bits() == gpu_out[2].to_bits();
        let ok = ok0 && ok1 && ok2;
        let detail = if ok {
            "0x7F→NaN, 0xFF→NaN, finite control matches".into()
        } else {
            format!(
                "cpu=[{:?},{:?},{:?}] gpu=[{:?},{:?},{:?}] ok0={ok0} ok1={ok1} ok2={ok2}",
                cpu[0], cpu[1], cpu[2], gpu_out[0], gpu_out[1], gpu_out[2]
            )
        };
        gate.record('A', "fp8_dequant_nan_e4m3_0x7F_0xFF", ok, detail);
    }
    Ok(())
}

// ── Part B: expert FP4 dequant ───────────────────────────────────────────────

fn part_b_exhaustive(gpu: &mut Gpu, gate: &mut Gate) -> Result<(), String> {
    // Logical [64, 512]; stored W [64, 256]; scale [64, 16].
    const M: usize = 64;
    const K: usize = 512; // multiple of 32
    let k_stored = K / 2;
    let k_scale = K / 32;
    let mut w = vec![0u8; M * k_stored];
    // Cover all 16 E2M1 codes in both nibble positions repeatedly.
    for i in 0..M * k_stored {
        let lo = (i % 16) as u8;
        let hi = ((i / 16) % 16) as u8;
        w[i] = lo | (hi << 4);
    }
    let mut s = vec![0u8; M * k_scale];
    for i in 0..M * k_scale {
        // Varied finite scales.
        s[i] = 110 + ((i * 7) % 40) as u8;
        if s[i] == 0 {
            s[i] = 1;
        }
    }

    let mut cpu = vec![0f32; M * K];
    dequant_expert_fp4_g32(&w, &s, M, K, &mut cpu)?;
    for v in cpu.iter_mut() {
        *v = round_to_bf16(*v);
    }

    let d_w = upload_bytes(gpu, &w)?;
    let d_s = upload_bytes(gpu, &s)?;
    let d_out = upload_bytes(gpu, &vec![0u8; M * K * 2])?;
    gpu.dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942(&d_w.buf, &d_s.buf, &d_out.buf, M, K)
        .map_err(|e| format!("dequant_fp4 exhaustive launch: {e:?}"))?;
    let gpu_out = download_bf16_f32(gpu, &d_out, M * K)?;
    free(gpu, d_w);
    free(gpu, d_s);
    free(gpu, d_out);

    match compare_bf16_domain(&cpu, &gpu_out, "B-exhaustive", 5) {
        Ok(()) => {
            gate.record(
                'B',
                "fp4_dequant_exhaustive_64x512",
                true,
                format!("{} logical elems bit-exact", M * K),
            );
        }
        Err(e) => {
            let mut extra = e.clone();
            for i in 0..M * K {
                if cpu[i].to_bits() != gpu_out[i].to_bits()
                    && !(cpu[i].is_nan() && gpu_out[i].is_nan())
                {
                    let m = i / K;
                    let k = i % K;
                    let byte = w[m * k_stored + k / 2];
                    let nibble = if k & 1 == 0 { byte & 0x0f } else { byte >> 4 };
                    let sb = s[m * k_scale + k / 32];
                    extra = format!(
                        "{e} | m={m} k={k} byte=0x{byte:02x} nibble=0x{nibble:x} \
                         S=0x{sb:02x} e2m1={:?} scale={:?} cpu={:?} gpu={:?}",
                        e2m1_to_f32(nibble),
                        ue8m0_to_f32(sb),
                        cpu[i],
                        gpu_out[i]
                    );
                    break;
                }
            }
            gate.record('B', "fp4_dequant_exhaustive_64x512", false, extra);
        }
    }
    Ok(())
}

fn part_b_nibble_order(gpu: &mut Gpu, gate: &mut Gate) -> Result<(), String> {
    // One row, K=32. Byte0 = 0x42 → low=0x2 (=1.0), high=0x4 (=2.0).
    // Contract: logical[0]=1.0, logical[1]=2.0 (low nibble = even k).
    const M: usize = 1;
    const K: usize = 32;
    let mut w = vec![0u8; K / 2];
    w[0] = 0x42;
    // Fill remaining bytes with a distinct pattern so a full-row swap is visible.
    for j in 1..w.len() {
        w[j] = 0x10; // lo=0 (=0.0), hi=1 (=0.5)
    }
    let s = vec![127u8; K / 32]; // scale = 1.0

    let mut cpu = vec![0f32; M * K];
    dequant_expert_fp4_g32(&w, &s, M, K, &mut cpu)?;
    for v in cpu.iter_mut() {
        *v = round_to_bf16(*v);
    }

    let d_w = upload_bytes(gpu, &w)?;
    let d_s = upload_bytes(gpu, &s)?;
    let d_out = upload_bytes(gpu, &vec![0u8; M * K * 2])?;
    gpu.dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942(&d_w.buf, &d_s.buf, &d_out.buf, M, K)
        .map_err(|e| format!("nibble-order launch: {e:?}"))?;
    let gpu_out = download_bf16_f32(gpu, &d_out, M * K)?;
    free(gpu, d_w);
    free(gpu, d_s);
    free(gpu, d_out);

    let expect0 = round_to_bf16(1.0);
    let expect1 = round_to_bf16(2.0);
    let ok_cpu = cpu[0].to_bits() == expect0.to_bits() && cpu[1].to_bits() == expect1.to_bits();
    let ok_gpu =
        gpu_out[0].to_bits() == expect0.to_bits() && gpu_out[1].to_bits() == expect1.to_bits();
    let ok_match = compare_bf16_domain(&cpu, &gpu_out, "B-nibble", 3).is_ok();
    let ok = ok_cpu && ok_gpu && ok_match;
    let detail = if ok {
        "byte 0x42 → logical[0]=1.0 (lo), logical[1]=2.0 (hi)".into()
    } else {
        format!(
            "expect [1.0, 2.0]; cpu=[{:?}, {:?}] gpu=[{:?}, {:?}] \
             (if swapped → high-nibble-first, CONTRACT BUG)",
            cpu[0], cpu[1], gpu_out[0], gpu_out[1]
        )
    };
    gate.record('B', "fp4_nibble_order_low_even_k", ok, detail);

    // Also assert LUT sanity for the codes we used.
    assert!((E2M1_LUT[2] - 1.0).abs() < 1e-6);
    assert!((E2M1_LUT[4] - 2.0).abs() < 1e-6);
    Ok(())
}

// ── Part C: activation quant ─────────────────────────────────────────────────

/// Build a multi-group activation row covering the delicate fast_round_scale cases.
fn build_fp8_act_cases(block: usize, n_rows: usize) -> Vec<f32> {
    assert!(block == 64 || block == 128);
    let mut x = vec![0.0f32; n_rows * block * 8]; // 8 groups per row
    for r in 0..n_rows {
        let base = r * block * 8;
        // g0: amax exactly a power of two (16.0)
        for i in 0..block {
            x[base + i] = if i == 0 { 16.0 } else { 1.0 };
        }
        // g1: amax just above a power of two (16.0 + eps)
        for i in 0..block {
            x[base + block + i] = if i == 0 { 16.0 + 1e-3 } else { 0.5 };
        }
        // g2: amax below the 1e-4 floor
        for i in 0..block {
            x[base + 2 * block + i] = 1e-6 * if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        // g3: all zeros
        for i in 0..block {
            x[base + 3 * block + i] = 0.0;
        }
        // g4: single huge outlier
        for i in 0..block {
            x[base + 4 * block + i] = if i == block / 2 { 400.0 } else { 0.01 };
        }
        // g5: E4M3 RNE midpoints. After /s the values should land on midpoints.
        // Use scale-friendly amax so s=1 (amax in (224, 448] → ceil log2(amax/448)=0).
        // amax=300 → s = 2^ceil(log2(300/448)) = 2^ceil(log2(0.669)) = 2^0 = 1.
        // Midpoints of E4M3 bins around small normals:
        //   between 0 and 2^-9 (=0.001953125) midpoint = 2^-10 = 0.0009765625
        //   between 1.0 (exp=7,mant=0) and 1.125 (exp=7,mant=1): mid = 1.0625
        // Place a few known midpoints + amax anchor.
        for i in 0..block {
            x[base + 5 * block + i] = match i % 8 {
                0 => 300.0,          // amax anchor → s=1
                1 => 1.0625,         // midpoint 1.0 ↔ 1.125
                2 => -1.0625,
                3 => 0.0009765625,   // midpoint 0 ↔ smallest subnormal
                4 => 2.25,           // midpoint 2.0 ↔ 2.5?  E4M3 at exp for 2: codes
                5 => 3.0 + 0.0625, // near 3.0
                6 => -0.5,
                _ => 0.25,
            };
        }
        // g6: mixed signs, amax just below FP8 max so s chosen delicately
        for i in 0..block {
            let t = (i as f32) / (block as f32);
            x[base + 6 * block + i] = (t - 0.5) * 200.0;
        }
        // g7: values that hit exact E4M3 codepoints (should be identity after round-trip at s=1)
        for i in 0..block {
            let code = (i % 128) as u8; // avoid NaN half of the space somewhat
            let c = if code == 0x7f { 0x3c } else { code };
            x[base + 7 * block + i] = e4m3_to_f32(c);
        }
        // Ensure g7 amax forces s such that clamp doesn't destroy — bump one to 400.
        x[base + 7 * block] = 400.0;
    }
    let _ = FP8_E4M3_MAX; // referenced for readability / future asserts
    x
}

fn build_fp4_act_cases(n_rows: usize) -> Vec<f32> {
    const BLOCK: usize = 32;
    let n_groups = 8;
    let mut x = vec![0.0f32; n_rows * BLOCK * n_groups];
    for r in 0..n_rows {
        let base = r * BLOCK * n_groups;
        // g0: amax exact power of two
        for i in 0..BLOCK {
            x[base + i] = if i == 0 { 4.0 } else { 0.5 };
        }
        // g1: amax just above power of two
        for i in 0..BLOCK {
            x[base + BLOCK + i] = if i == 0 { 4.0 + 1e-3 } else { 0.25 };
        }
        // g2: below FP4 floor 6*2^-126
        let tiny = 6.0 * 2.0f32.powi(-130);
        for i in 0..BLOCK {
            x[base + 2 * BLOCK + i] = if i % 2 == 0 { tiny } else { -tiny };
        }
        // g3: all zeros
        for i in 0..BLOCK {
            x[base + 3 * BLOCK + i] = 0.0;
        }
        // g4: single huge outlier
        for i in 0..BLOCK {
            x[base + 4 * BLOCK + i] = if i == 3 { 5.5 } else { 0.01 };
        }
        // g5: E2M1 RNE midpoints at s=1 (amax in (3,6] → s=1)
        // midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
        let mids = [0.25f32, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -2.5, -5.0];
        for i in 0..BLOCK {
            x[base + 5 * BLOCK + i] = if i == 0 {
                5.5 // amax anchor → s=1
            } else {
                mids[i % mids.len()]
            };
        }
        // g6: mixed
        for i in 0..BLOCK {
            x[base + 6 * BLOCK + i] = ((i as f32) - 16.0) * 0.3;
        }
        // g7: exact E2M1 codepoints
        for i in 0..BLOCK {
            x[base + 7 * BLOCK + i] = e2m1_to_f32((i % 16) as u8);
        }
        x[base + 7 * BLOCK] = FP4_E2M1_MAX; // ensure finite amax
    }
    x
}

fn run_act_quant_fp8(
    gpu: &mut Gpu,
    gate: &mut Gate,
    block: usize,
    name: &'static str,
) -> Result<(), String> {
    let n_rows = 4usize;
    let x_f32 = build_fp8_act_cases(block, n_rows);
    let last_dim = x_f32.len() / n_rows;
    assert_eq!(last_dim % block, 0);

    // CPU oracle rounds to BF16 internally before amax (kernel.py in_dtype=BF16).
    let mut cpu = x_f32.clone();
    act_quant_fp8_inplace_ref(&mut cpu, last_dim, block)?;

    // GPU path: upload BF16-rounded inputs (matching the kernel's BF16 domain).
    let in_bf16: Vec<f32> = x_f32.iter().copied().map(round_to_bf16).collect();
    // Re-run CPU from the same BF16 inputs — must bit-agree with f32-in path.
    let mut cpu_from_bf16 = in_bf16.clone();
    act_quant_fp8_inplace_ref(&mut cpu_from_bf16, last_dim, block)?;

    let bytes = pack_f32_to_bf16_bytes(&in_bf16);
    let d_x = upload_bytes(gpu, &bytes)?;
    gpu.act_quant_fp8_ue8m0_inplace_gfx942(&d_x.buf, n_rows, last_dim, block)
        .map_err(|e| format!("act_quant_fp8 block={block} launch: {e:?}"))?;
    let gpu_out = download_bf16_f32(gpu, &d_x, n_rows * last_dim)?;
    free(gpu, d_x);

    // f32-in vs bf16-in host paths must agree (oracle BF16-domain contract).
    if let Err(e) = compare_bf16_domain(&cpu, &cpu_from_bf16, "cpu-f32in-vs-bf16in", 4) {
        gate.record(
            'C',
            name,
            false,
            format!("host f32-in vs bf16-in disagree (BF16-domain contract broken): {e}"),
        );
        return Ok(());
    }

    match compare_bf16_domain(&cpu_from_bf16, &gpu_out, name, 8) {
        Ok(()) => {
            gate.record(
                'C',
                name,
                true,
                format!("rows={n_rows} last_dim={last_dim} block={block} f32_in_agree=true"),
            );
        }
        Err(e) => {
            let mut extra = e.clone();
            for i in 0..cpu_from_bf16.len() {
                if cpu_from_bf16[i].to_bits() != gpu_out[i].to_bits()
                    && !(cpu_from_bf16[i].is_nan() && gpu_out[i].is_nan())
                {
                    let row = i / last_dim;
                    let col = i % last_dim;
                    let group = col / block;
                    extra = format!(
                        "{e} | row={row} col={col} group={group} \
                         xin_bf16={:?} cpu={:?} gpu={:?}",
                        in_bf16[i], cpu_from_bf16[i], gpu_out[i]
                    );
                    break;
                }
            }
            gate.record('C', name, false, extra);
        }
    }
    Ok(())
}

/// Full-precision f32 activations shaped like the compressor post-RoPE site
/// (`model.py:378`: `kv[..., :-rd]`, block 64, last_dim = 448 = 512−64).
///
/// Values carry bottom-16-bit mantissa junk (f64-accum → f32 cast, not a BF16
/// residual) and deliberately plant amaxes just above power-of-two
/// `fast_round_scale` boundaries so BF16 rounding flips the scale by 2×.
/// Gate 2 previously only exercised BF16-clean synthetics and missed this
/// class — the coverage hole behind the compressor finding.
fn build_fp8_post_rope_like_cases(n_rows: usize) -> Vec<f32> {
    const BLOCK: usize = 64;
    const LAST_DIM: usize = 448; // 7 groups × 64 — compressor non-RoPE slice
    // Just-above boundaries of amax/448 at powers of two. BF16 rounds each
    // back onto the boundary, flipping fast_round_scale by exactly one exp.
    let near = [
        224.4f32, // s_f32=1.0  vs s_bf16=0.5
        112.3,    // 0.5 vs 0.25
        56.2,     // 0.25 vs 0.125
        28.1,     // 0.125 vs 0.0625
        14.05,    // 0.0625 vs 0.03125
        7.02,     // 0.03125 vs 0.015625
        448.4,    // 2.0 vs 1.0
    ];
    let mut x = vec![0.0f32; n_rows * LAST_DIM];
    for r in 0..n_rows {
        for g in 0..(LAST_DIM / BLOCK) {
            let base = r * LAST_DIM + g * BLOCK;
            for i in 0..BLOCK {
                // Plant full-f32 mantissa noise (bottom 16 bits set).
                let bits = (0.37f32 * (i as f32 + 1.0 + r as f32)).to_bits() | 0x0000_a5a5;
                let v = f32::from_bits(bits);
                x[base + i] = if i % 2 == 0 { v } else { -v };
            }
            x[base] = near[g % near.len()];
        }
    }
    x
}

fn run_act_quant_fp8_post_rope_like(gpu: &mut Gpu, gate: &mut Gate) -> Result<(), String> {
    const BLOCK: usize = 64;
    let n_rows = 4usize;
    let x_f32 = build_fp8_post_rope_like_cases(n_rows);
    let last_dim = x_f32.len() / n_rows;
    assert_eq!(last_dim, 448);
    assert_eq!(last_dim % BLOCK, 0);

    // Prove the coverage hole is real: f32-domain amax scale ≠ BF16-domain
    // scale on at least one planted group (otherwise the case is vacuous).
    let mut n_scale_flip = 0usize;
    for r in 0..n_rows {
        for g in 0..(last_dim / BLOCK) {
            let base = r * last_dim + g * BLOCK;
            let mut amax_f = 0.0f32;
            let mut amax_b = 0.0f32;
            for i in 0..BLOCK {
                let v = x_f32[base + i];
                amax_f = amax_f.max(v.abs());
                amax_b = amax_b.max(round_to_bf16(v).abs());
            }
            let s_f = fast_round_scale(amax_f.max(1e-4), 1.0 / 448.0);
            let s_b = fast_round_scale(amax_b.max(1e-4), 1.0 / 448.0);
            if s_f != s_b {
                n_scale_flip += 1;
            }
        }
    }
    if n_scale_flip == 0 {
        gate.record(
            'C',
            "act_quant_fp8_post_rope_like_block64",
            false,
            "vacuous: no group had f32-vs-bf16 scale flip — case construction broken",
        );
        return Ok(());
    }

    // Host oracle on full-f32 (internally BF16-rounds) vs GPU on BF16 buffer.
    let mut cpu = x_f32.clone();
    act_quant_fp8_inplace_ref(&mut cpu, last_dim, BLOCK)?;

    let in_bf16: Vec<f32> = x_f32.iter().copied().map(round_to_bf16).collect();
    let bytes = pack_f32_to_bf16_bytes(&in_bf16);
    let d_x = upload_bytes(gpu, &bytes)?;
    gpu.act_quant_fp8_ue8m0_inplace_gfx942(&d_x.buf, n_rows, last_dim, BLOCK)
        .map_err(|e| format!("act_quant_fp8 post-rope-like launch: {e:?}"))?;
    let gpu_out = download_bf16_f32(gpu, &d_x, n_rows * last_dim)?;
    free(gpu, d_x);

    match compare_bf16_domain(&cpu, &gpu_out, "act_quant_fp8_post_rope_like_block64", 8) {
        Ok(()) => {
            gate.record(
                'C',
                "act_quant_fp8_post_rope_like_block64",
                true,
                format!(
                    "rows={n_rows} last_dim={last_dim} block={BLOCK} \
                     scale_flips={n_scale_flip} (f32-domain≠bf16-domain amax groups)"
                ),
            );
        }
        Err(e) => {
            let mut extra = e.clone();
            for i in 0..cpu.len() {
                if cpu[i].to_bits() != gpu_out[i].to_bits()
                    && !(cpu[i].is_nan() && gpu_out[i].is_nan())
                {
                    let row = i / last_dim;
                    let col = i % last_dim;
                    let group = col / BLOCK;
                    extra = format!(
                        "{e} | row={row} col={col} group={group} \
                         xin_f32={:?} xin_bf16={:?} cpu={:?} gpu={:?}",
                        x_f32[i], in_bf16[i], cpu[i], gpu_out[i]
                    );
                    break;
                }
            }
            gate.record('C', "act_quant_fp8_post_rope_like_block64", false, extra);
        }
    }
    Ok(())
}

fn run_act_quant_fp4(gpu: &mut Gpu, gate: &mut Gate) -> Result<(), String> {
    let n_rows = 4usize;
    let x_f32 = build_fp4_act_cases(n_rows);
    let last_dim = x_f32.len() / n_rows;
    assert_eq!(last_dim % 32, 0);

    let in_bf16: Vec<f32> = x_f32.iter().copied().map(round_to_bf16).collect();
    let mut cpu = in_bf16.clone();
    act_quant_fp4_inplace_ref(&mut cpu, last_dim)?;

    let bytes = pack_f32_to_bf16_bytes(&in_bf16);
    let d_x = upload_bytes(gpu, &bytes)?;
    gpu.act_quant_fp4_ue8m0_g32_inplace_gfx942(&d_x.buf, n_rows, last_dim)
        .map_err(|e| format!("act_quant_fp4 launch: {e:?}"))?;
    let gpu_out = download_bf16_f32(gpu, &d_x, n_rows * last_dim)?;
    free(gpu, d_x);

    match compare_bf16_domain(&cpu, &gpu_out, "C-fp4", 8) {
        Ok(()) => {
            gate.record(
                'C',
                "act_quant_fp4_g32",
                true,
                format!("rows={n_rows} last_dim={last_dim}"),
            );
        }
        Err(e) => {
            let mut extra = e.clone();
            for i in 0..cpu.len() {
                if cpu[i].to_bits() != gpu_out[i].to_bits()
                    && !(cpu[i].is_nan() && gpu_out[i].is_nan())
                {
                    let row = i / last_dim;
                    let col = i % last_dim;
                    extra = format!(
                        "{e} | row={row} col={col} xin={:?} cpu={:?} gpu={:?}",
                        in_bf16[i], cpu[i], gpu_out[i]
                    );
                    break;
                }
            }
            gate.record('C', "act_quant_fp4_g32", false, extra);
        }
    }
    Ok(())
}

// ── Part D: real checkpoint samples ──────────────────────────────────────────

#[derive(Debug)]
struct Stats {
    min: f32,
    max: f32,
    mean: f32,
    std: f32,
    frac_zero: f32,
    n: usize,
    n_nan: usize,
}

fn stats(xs: &[f32]) -> Stats {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut n_finite = 0usize;
    let mut n_zero = 0usize;
    let mut n_nan = 0usize;
    for &v in xs {
        if v.is_nan() {
            n_nan += 1;
            continue;
        }
        n_finite += 1;
        min = min.min(v);
        max = max.max(v);
        sum += v as f64;
        sumsq += (v as f64) * (v as f64);
        if v == 0.0 {
            n_zero += 1;
        }
    }
    let mean = if n_finite > 0 {
        (sum / n_finite as f64) as f32
    } else {
        f32::NAN
    };
    let var = if n_finite > 1 {
        (sumsq / n_finite as f64) - (mean as f64) * (mean as f64)
    } else {
        0.0
    };
    Stats {
        min,
        max,
        mean,
        std: (var.max(0.0).sqrt()) as f32,
        frac_zero: if n_finite > 0 {
            n_zero as f32 / n_finite as f32
        } else {
            f32::NAN
        },
        n: xs.len(),
        n_nan,
    }
}

fn read_bin(path: &Path) -> Result<Vec<u8>, String> {
    let mut f = fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(buf)
}

/// Locate a tensor inside a safetensors shard (stdlib-free host path used when
/// pre-extracted dumps are absent). Returns (dtype, shape, bytes).
fn load_from_shard(
    shard_path: &Path,
    tensor_name: &str,
) -> Result<(String, Vec<usize>, Vec<u8>), String> {
    let data = fs::read(shard_path).map_err(|e| format!("read shard: {e}"))?;
    if data.len() < 8 {
        return Err("shard too small".into());
    }
    let hdr_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    if 8 + hdr_len > data.len() {
        return Err("header overruns file".into());
    }
    let hdr: serde_json::Value = serde_json::from_slice(&data[8..8 + hdr_len])
        .map_err(|e| format!("header json: {e}"))?;
    let meta = hdr
        .get(tensor_name)
        .ok_or_else(|| format!("tensor {tensor_name} not in {}", shard_path.display()))?;
    let dtype = meta
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or("dtype")?
        .to_string();
    let shape: Vec<usize> = meta
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or("shape")?
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let offs = meta
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .ok_or("data_offsets")?;
    let o0 = offs[0].as_u64().unwrap() as usize;
    let o1 = offs[1].as_u64().unwrap() as usize;
    let body = &data[8 + hdr_len..];
    if o1 > body.len() {
        return Err(format!("offset {o1} past body {}", body.len()));
    }
    Ok((dtype, shape, body[o0..o1].to_vec()))
}

fn resolve_tensor(
    tensor_dir: Option<&Path>,
    ckpt_dir: &Path,
    shard_name: &str,
    dump_name: &str,
    tensor_name: &str,
) -> Result<(String, Vec<usize>, Vec<u8>), String> {
    if let Some(dir) = tensor_dir {
        let p = dir.join(dump_name);
        if p.exists() {
            let bytes = read_bin(&p)?;
            // Infer shape/dtype from known tensors.
            let (dtype, shape): (String, Vec<usize>) = match tensor_name {
                "layers.3.attn.wq_a.weight" => ("F8_E4M3".into(), vec![1024, 4096]),
                "layers.3.attn.wq_a.scale" => ("F8_E8M0".into(), vec![8, 32]),
                "layers.3.ffn.experts.0.w1.weight" => ("I8".into(), vec![2048, 2048]),
                "layers.3.ffn.experts.0.w1.scale" => ("F8_E8M0".into(), vec![2048, 128]),
                _ => return Err(format!("unknown dump tensor {tensor_name}")),
            };
            let expect: usize = shape.iter().product();
            if bytes.len() != expect {
                return Err(format!(
                    "dump {} len {} != expected {expect} for {shape:?}",
                    p.display(),
                    bytes.len()
                ));
            }
            return Ok((dtype, shape, bytes));
        }
    }
    let shard = ckpt_dir.join(shard_name);
    load_from_shard(&shard, tensor_name)
}

fn part_d_real(gpu: &mut Gpu, gate: &mut Gate, ckpt_dir: &Path, tensor_dir: Option<&Path>) {
    println!();
    println!("── Part D: real checkpoint samples ──");

    // Dense: layers.3.attn.wq_a.weight F8_E4M3 [1024,4096] + scale [8,32]
    let dense = (|| -> Result<(), String> {
        let (wd, ws, wbytes) = resolve_tensor(
            tensor_dir,
            ckpt_dir,
            "model-00005-of-00048.safetensors",
            "layers_3_attn_wq_a_weight.bin",
            "layers.3.attn.wq_a.weight",
        )?;
        let (sd, ss, sbytes) = resolve_tensor(
            tensor_dir,
            ckpt_dir,
            "model-00005-of-00048.safetensors",
            "layers_3_attn_wq_a_scale.bin",
            "layers.3.attn.wq_a.scale",
        )?;
        println!(
            "  dense weight: dtype={wd} shape={ws:?} bytes={}",
            wbytes.len()
        );
        println!(
            "  dense scale:  dtype={sd} shape={ss:?} bytes={}",
            sbytes.len()
        );
        let m = ws[0];
        let k = ws[1];
        assert_eq!(ws, vec![1024, 4096]);
        assert_eq!(ss, vec![8, 32]);

        let mut cpu = vec![0f32; m * k];
        dequant_dense_fp8_block128(&wbytes, &sbytes, m, k, &mut cpu)?;
        for v in cpu.iter_mut() {
            *v = round_to_bf16(*v);
        }

        let d_w = upload_bytes(gpu, &wbytes)?;
        let d_s = upload_bytes(gpu, &sbytes)?;
        let d_out = upload_bytes(gpu, &vec![0u8; m * k * 2])?;
        gpu.dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942(&d_w.buf, &d_s.buf, &d_out.buf, m, k)
            .map_err(|e| format!("dense real launch: {e:?}"))?;
        let gpu_out = download_bf16_f32(gpu, &d_out, m * k)?;
        free(gpu, d_w);
        free(gpu, d_s);
        free(gpu, d_out);

        let st = stats(&cpu);
        println!(
            "  dense decoded stats: n={} min={:.6} max={:.6} mean={:.6} std={:.6} \
             frac_zero={:.6} n_nan={}",
            st.n, st.min, st.max, st.mean, st.std, st.frac_zero, st.n_nan
        );
        // Print a few sample values for human eyeballing.
        print!("  dense sample [0..8]:");
        for i in 0..8 {
            print!(" {:+.5}", cpu[i]);
        }
        println!();

        match compare_bf16_domain(&cpu, &gpu_out, "D-dense", 5) {
            Ok(()) => {
                let looks_trained = st.n_nan == 0
                    && st.min > -50.0
                    && st.max < 50.0
                    && st.frac_zero < 0.5
                    && st.std > 1e-4
                    && st.std < 10.0;
                gate.record(
                    'D',
                    "real_dense_wq_a_bitexact",
                    true,
                    format!(
                        "min={:.4} max={:.4} mean={:.4} std={:.4} zero={:.4} trained_like={looks_trained}",
                        st.min, st.max, st.mean, st.std, st.frac_zero
                    ),
                );
                gate.record(
                    'D',
                    "real_dense_wq_a_stats_sane",
                    looks_trained,
                    format!(
                        "assessment: {}",
                        if looks_trained {
                            "looks like trained weights (bounded, non-degenerate, non-noise)"
                        } else {
                            "SUSPICIOUS stats — investigate"
                        }
                    ),
                );
            }
            Err(e) => {
                gate.record('D', "real_dense_wq_a_bitexact", false, e);
                gate.record(
                    'D',
                    "real_dense_wq_a_stats_sane",
                    false,
                    "skipped (bitexact failed)",
                );
            }
        }
        Ok(())
    })();
    if let Err(e) = dense {
        gate.record('D', "real_dense_wq_a_bitexact", false, e);
        gate.record('D', "real_dense_wq_a_stats_sane", false, "load failed");
    }

    // Expert: layers.3.ffn.experts.0.w1.weight I8 [2048,2048] logical [2048,4096]
    //         + scale [2048,128]
    let expert = (|| -> Result<(), String> {
        let (wd, ws, wbytes) = resolve_tensor(
            tensor_dir,
            ckpt_dir,
            "model-00005-of-00048.safetensors",
            "layers_3_ffn_experts_0_w1_weight.bin",
            "layers.3.ffn.experts.0.w1.weight",
        )?;
        let (sd, ss, sbytes) = resolve_tensor(
            tensor_dir,
            ckpt_dir,
            "model-00005-of-00048.safetensors",
            "layers_3_ffn_experts_0_w1_scale.bin",
            "layers.3.ffn.experts.0.w1.scale",
        )?;
        println!(
            "  expert weight: dtype={wd} shape={ws:?} bytes={}",
            wbytes.len()
        );
        println!(
            "  expert scale:  dtype={sd} shape={ss:?} bytes={}",
            sbytes.len()
        );
        let m = ws[0];
        let k_stored = ws[1];
        let k = k_stored * 2; // logical
        assert_eq!(ws, vec![2048, 2048]);
        assert_eq!(ss, vec![2048, 128]);
        assert_eq!(k, 4096);

        let mut cpu = vec![0f32; m * k];
        dequant_expert_fp4_g32(&wbytes, &sbytes, m, k, &mut cpu)?;
        for v in cpu.iter_mut() {
            *v = round_to_bf16(*v);
        }

        let d_w = upload_bytes(gpu, &wbytes)?;
        let d_s = upload_bytes(gpu, &sbytes)?;
        let d_out = upload_bytes(gpu, &vec![0u8; m * k * 2])?;
        gpu.dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942(&d_w.buf, &d_s.buf, &d_out.buf, m, k)
            .map_err(|e| format!("expert real launch: {e:?}"))?;
        let gpu_out = download_bf16_f32(gpu, &d_out, m * k)?;
        free(gpu, d_w);
        free(gpu, d_s);
        free(gpu, d_out);

        let st = stats(&cpu);
        println!(
            "  expert decoded stats: n={} min={:.6} max={:.6} mean={:.6} std={:.6} \
             frac_zero={:.6} n_nan={}",
            st.n, st.min, st.max, st.mean, st.std, st.frac_zero, st.n_nan
        );
        print!("  expert sample [0..8]:");
        for i in 0..8 {
            print!(" {:+.5}", cpu[i]);
        }
        println!();
        // Nibble-order smoke on real bytes: decode first byte both ways.
        let b0 = wbytes[0];
        let lo = e2m1_to_f32(b0 & 0x0f) * ue8m0_to_f32(sbytes[0]);
        let hi = e2m1_to_f32(b0 >> 4) * ue8m0_to_f32(sbytes[0]);
        println!(
            "  expert nibble check: byte0=0x{b0:02x} lo*s={lo:?} hi*s={hi:?} \
             cpu[0]={:?} cpu[1]={:?} (expect cpu[0]≈lo, cpu[1]≈hi)",
            cpu[0], cpu[1]
        );

        match compare_bf16_domain(&cpu, &gpu_out, "D-expert", 5) {
            Ok(()) => {
                let looks_trained = st.n_nan == 0
                    && st.min > -50.0
                    && st.max < 50.0
                    && st.frac_zero < 0.6
                    && st.std > 1e-4
                    && st.std < 10.0;
                gate.record(
                    'D',
                    "real_expert_w1_bitexact",
                    true,
                    format!(
                        "min={:.4} max={:.4} mean={:.4} std={:.4} zero={:.4} trained_like={looks_trained}",
                        st.min, st.max, st.mean, st.std, st.frac_zero
                    ),
                );
                gate.record(
                    'D',
                    "real_expert_w1_stats_sane",
                    looks_trained,
                    format!(
                        "assessment: {}",
                        if looks_trained {
                            "looks like trained weights"
                        } else {
                            "SUSPICIOUS stats — investigate"
                        }
                    ),
                );
            }
            Err(e) => {
                gate.record('D', "real_expert_w1_bitexact", false, e);
                gate.record(
                    'D',
                    "real_expert_w1_stats_sane",
                    false,
                    "skipped (bitexact failed)",
                );
            }
        }
        Ok(())
    })();
    if let Err(e) = expert {
        gate.record('D', "real_expert_w1_bitexact", false, e);
        gate.record('D', "real_expert_w1_stats_sane", false, "load failed");
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

fn parse_args() -> (PathBuf, Option<PathBuf>) {
    let mut ckpt = PathBuf::from("/mnt/scratch/models/DeepSeek-V4-Flash-0731");
    let mut tensor_dir: Option<PathBuf> = None;
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--ckpt-dir" => {
                if let Some(v) = args.next() {
                    ckpt = PathBuf::from(v);
                }
            }
            "--tensor-dir" => {
                if let Some(v) = args.next() {
                    tensor_dir = Some(PathBuf::from(v));
                }
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: ds4_parent_codec_gate [--ckpt-dir DIR] [--tensor-dir DIR]"
                );
                std::process::exit(0);
            }
            other => {
                // Bare path → ckpt dir (back-compat).
                ckpt = PathBuf::from(other);
            }
        }
    }
    // Default pre-extracted dumps if present.
    if tensor_dir.is_none() {
        let def = PathBuf::from("/tmp/ds4_codec_gate_tensors");
        if def.join("layers_3_attn_wq_a_weight.bin").exists() {
            tensor_dir = Some(def);
        }
    }
    (ckpt, tensor_dir)
}

fn main() -> ExitCode {
    println!("ds4_parent_codec_gate — Gate 2 GPU↔CPU bit-exact codec verification");
    println!("host target: gfx942 (MI300X)\n");

    let (ckpt_dir, tensor_dir) = parse_args();
    println!("ckpt_dir:    {}", ckpt_dir.display());
    println!(
        "tensor_dir:  {}",
        tensor_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(read shards directly)".into())
    );

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("FATAL: Gpu::init failed: {e:?}");
            return ExitCode::from(2);
        }
    };
    println!(
        "gpu arch:    {}  (gfx942 required)",
        gpu.arch
    );
    if !gpu.arch_caps.is_gfx942() {
        eprintln!(
            "FATAL: this binary must run on gfx942; got arch={}",
            gpu.arch
        );
        return ExitCode::from(2);
    }

    let mut gate = Gate::new();

    // ── Part A ──
    println!("\n── Part A: dense FP8 dequant ──");
    if let Err(e) = part_a_exhaustive(&mut gpu, &mut gate) {
        gate.record('A', "fp8_dequant_exhaustive_256x256", false, e);
    }
    if let Err(e) = part_a_ragged(&mut gpu, &mut gate) {
        gate.record('A', "fp8_dequant_ragged_260x300", false, e);
    }
    if let Err(e) = part_a_nan(&mut gpu, &mut gate) {
        gate.record('A', "fp8_dequant_nan_cases", false, e);
    }

    // ── Part B ──
    println!("\n── Part B: expert FP4 dequant ──");
    if let Err(e) = part_b_exhaustive(&mut gpu, &mut gate) {
        gate.record('B', "fp4_dequant_exhaustive_64x512", false, e);
    }
    if let Err(e) = part_b_nibble_order(&mut gpu, &mut gate) {
        gate.record('B', "fp4_nibble_order_low_even_k", false, e);
    }

    // ── Part C ──
    println!("\n── Part C: activation quant ──");
    if let Err(e) = run_act_quant_fp8(&mut gpu, &mut gate, 128, "act_quant_fp8_block128") {
        gate.record('C', "act_quant_fp8_block128", false, e);
    }
    if let Err(e) = run_act_quant_fp8(&mut gpu, &mut gate, 64, "act_quant_fp8_block64") {
        gate.record('C', "act_quant_fp8_block64", false, e);
    }
    if let Err(e) = run_act_quant_fp8_post_rope_like(&mut gpu, &mut gate) {
        gate.record('C', "act_quant_fp8_post_rope_like_block64", false, e);
    }
    if let Err(e) = run_act_quant_fp4(&mut gpu, &mut gate) {
        gate.record('C', "act_quant_fp4_g32", false, e);
    }

    // ── Part D ──
    part_d_real(
        &mut gpu,
        &mut gate,
        &ckpt_dir,
        tensor_dir.as_deref(),
    );

    gate.print_table();

    if gate.all_ok() {
        println!("\nGATE 2 RESULT: ALL PASS");
        ExitCode::SUCCESS
    } else {
        println!("\nGATE 2 RESULT: FAIL (see mismatches above — do not loosen tolerances)");
        ExitCode::from(1)
    }
}
