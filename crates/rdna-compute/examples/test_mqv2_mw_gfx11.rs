// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! MQ{3,4,5,6}V2 generic gfx11 multi-wave LDS raw-bit parity.
//!
//! For gate/up overwrite, residual `Y += W@X`, QKV (q/k/v) and QKVZA
//! (qkv/z/beta/alpha) compare the direct generic MW launchers
//!   `gemm_gate_up_mqv2_wmma_gfx11_mw_lds` (MW4 @ N=384, MW8 @ N=512)
//!   `gemm_mqv2_residual_wmma_gfx11_mw_lds` (MW4 @ N=416, MW8 @ N=464)
//!   `gemm_qkv_mqv2_wmma_gfx11_mw_lds` (MW4 @ N=512/511, MW8 @ N=512/128 + M-tail)
//!   `gemm_qkvza_mqv2_wmma_gfx11_mw_lds` (MW4 @ N=512/511, MW8 @ N=512/128 + M-tail)
//! against the exact per-format base entrypoints under forced `capture_mode`
//! (so production BT/MW policy cannot hijack the oracle).
//!
//! Synthetic V2 writer: `group_bytes = 8 + 32*bits`, deliberately disjoint dual
//! FP16 header halves (half0 ∈ [-1,1], half1 ∈ [96,160]), codes packed
//! contiguously LSB-first. Deterministic nondegenerate F32 X, boundary-sensitive
//! projection sizes, K=512. Residual arms start from identical nonzero Y.
//! Quiet-NaN sentinels on gate/up/QKV/QKVZA prove full overwrite (no leftover
//! NaN / no accidental `+=`; residual preserves `Y += W@X`).
//!
//! QKV/QKVZA fixtures mirror `test_mqv2_bt_gfx11.rs`:
//!   QKV   q_m=40 k_m=32 v_m=48 (total 120 = 7×16+8 tail)
//!   QKVZA qkv_m=40 z_m=28 beta_m=36 alpha_m=16 (total 120 = 7×16+8 tail)
//! plus one M-tail variant (q/qkv +1 → total 121 = 7×16+9) when the base
//! entrypoints support non-16-aligned M (they do — `div_ceil(16)` row tiles).
//!
//! Arch policy:
//!   - exact gfx1151: bits ∈ {3,4,5,6} → 4 bits × 12 ops = 48 arms + 8 M-tail
//!   - exact gfx1100: bits ∈ {4,5,6}   → 3 bits × 12 ops = 36 arms + 6 M-tail (MQ3 skipped)
//!   - other arch / no GPU: SKIP exit 0
//! Cross-arch totals: 56+42=98 with M-tail (28 gate/resid only without QKV).
//! Exit 0 PASS / 1 FAIL.

use rdna_compute::kv_slots::half_from_f32;
use rdna_compute::{DType, Gpu, GpuTensor};

const GROUP: usize = 256;
const HALF: usize = 128;
const K: usize = 512;

// Boundary / tail-sensitive projection sizes.
// gate_m=40 → gate/up boundary row 40 lies inside 16-row tile [32, 48);
// up_m=53  → total 93 = 5×16 + 13-row partial final tile.
const GATE_M: usize = 40;
const UP_M: usize = 53;
// residual 56 → final 16-row tile has 8 live rows.
const RESID_M: usize = 56;

// QKV / QKVZA projection sizes (mirror test_mqv2_bt_gfx11.rs).
// total 120 = 7×16+8 tail — already M-tail; extra +1 variant below for
// explicit 121 = 7×16+9 tail read of row-boundary + partial tile.
const QKVZA_QKV_M: usize = 40;
const QKVZA_Z_M: usize = 28;
const QKVZA_BETA_M: usize = 36;
const QKVZA_ALPHA_M: usize = 16;
const QKV_Q_M: usize = 40;
const QKV_K_M: usize = 32;
const QKV_V_M: usize = 48;
// M-tail variants (+1 on lead projection → total 121).
const QKVZA_QKV_M_TAIL: usize = 41;
const QKV_Q_M_TAIL: usize = 41;

// Production-boundary N paired 1:1 with MW waves (not a cross product).
const GATE_NS: [(usize, usize); 2] = [(384, 4), (512, 8)];
const RESID_NS: [(usize, usize); 2] = [(416, 4), (464, 8)];
// QKV/QKVZA MW matrix: waves 4 @ 512+511 (partial final-wave cols),
// waves 8 @ 512+128 (whole tail waves). Shared for both ops.
const QKV_MW_NS: [(usize, usize); 4] = [(512, 4), (511, 4), (512, 8), (128, 8)];
const QKVZA_MW_NS: [(usize, usize); 4] = [(512, 4), (511, 4), (512, 8), (128, 8)];

const NAN_GATE_BITS: u32 = 0x7fc0_0001;
const NAN_UP_BITS: u32 = 0x7fc0_0002;
const SALT_GATE: u32 = 0x1111_2222;
const SALT_UP: u32 = 0x3333_4444;
const SALT_RESID: u32 = 0xC0DE_F00D;

fn group_bytes(bits: u8) -> usize {
    8 + 32 * bits as usize
}

fn max_q(bits: u8) -> u8 {
    match bits {
        3 => 7,
        4 => 15,
        5 => 31,
        6 => 63,
        _ => unreachable!(),
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let mut exp = ((bits >> 10) & 0x1f) as u32;
    let mut mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            exp = 127 - 15 + 1;
            while mant & 0x0400 == 0 {
                mant <<= 1;
                exp -= 1;
            }
            sign | (exp << 23) | ((mant & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

fn build_disjoint_halves_seeded(m: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let salt = (r * 7919 + g * 104_729) as u32 ^ seed;
            for i in 0..HALF {
                w[base + i] = prng(i, salt) * 2.0 - 1.0;
            }
            for i in HALF..GROUP {
                w[base + i] = 96.0 + prng(i, salt ^ 0xA5A5_A5A5) * 64.0;
            }
        }
    }
    w
}

/// Synthetic V2 packer: dual FP16 headers + contiguous LSB-first integer codes.
fn pack_mqv2(bits: u8, w: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert!(matches!(bits, 3 | 4 | 5 | 6));
    assert_eq!(k % GROUP, 0);
    assert_eq!(w.len(), m * k);
    let gb = group_bytes(bits);
    let mq = max_q(bits) as f32;
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * gb];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * gb;
            let mut codes = [0u8; GROUP];
            for h in 0..2 {
                let off = h * HALF;
                let slice = &w[src + off..src + off + HALF];
                let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / mq } else { 0.0 };
                let s_bits = if hi == lo { 0u16 } else { half_from_f32(step) };
                let z_bits = half_from_f32(lo);
                blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&s_bits.to_le_bytes());
                blob[dst + h * 4 + 2..dst + h * 4 + 4].copy_from_slice(&z_bits.to_le_bytes());
                let s_rt = f16_to_f32(s_bits);
                let z_rt = f16_to_f32(z_bits);
                if s_rt == 0.0 {
                    continue;
                }
                let inv = 1.0 / s_rt;
                for i in 0..HALF {
                    let q = ((slice[i] - z_rt) * inv + 0.5).floor().clamp(0.0, mq);
                    codes[off + i] = q as u8;
                }
            }
            match bits {
                3 => {
                    for chunk in 0..32 {
                        let ci = chunk * 8;
                        let qq = [
                            codes[ci] & 7,
                            codes[ci + 1] & 7,
                            codes[ci + 2] & 7,
                            codes[ci + 3] & 7,
                            codes[ci + 4] & 7,
                            codes[ci + 5] & 7,
                            codes[ci + 6] & 7,
                            codes[ci + 7] & 7,
                        ];
                        let b0 = qq[0] | (qq[1] << 3) | ((qq[2] & 3) << 6);
                        let b1 =
                            ((qq[2] >> 2) & 1) | (qq[3] << 1) | (qq[4] << 4) | ((qq[5] & 1) << 7);
                        let b2 = ((qq[5] >> 1) & 3) | (qq[6] << 2) | (qq[7] << 5);
                        let bo = dst + 8 + chunk * 3;
                        blob[bo] = b0;
                        blob[bo + 1] = b1;
                        blob[bo + 2] = b2;
                    }
                }
                4 => {
                    for i in 0..HALF {
                        let lo_q = codes[2 * i] & 0xF;
                        let hi_q = codes[2 * i + 1] & 0xF;
                        blob[dst + 8 + i] = lo_q | (hi_q << 4);
                    }
                }
                5 => {
                    for i in (0..256).step_by(8) {
                        let bo = dst + 8 + (i / 8) * 5;
                        let q0 = codes[i] & 31;
                        let q1 = codes[i + 1] & 31;
                        let q2 = codes[i + 2] & 31;
                        let q3 = codes[i + 3] & 31;
                        let q4 = codes[i + 4] & 31;
                        let q5 = codes[i + 5] & 31;
                        let q6 = codes[i + 6] & 31;
                        let q7 = codes[i + 7] & 31;
                        blob[bo] = q0 | (q1 << 5);
                        blob[bo + 1] = (q1 >> 3) | (q2 << 2) | (q3 << 7);
                        blob[bo + 2] = (q3 >> 1) | (q4 << 4);
                        blob[bo + 3] = (q4 >> 4) | (q5 << 1) | (q6 << 6);
                        blob[bo + 4] = (q6 >> 2) | (q7 << 3);
                    }
                }
                6 => {
                    for i in (0..256).step_by(4) {
                        let bo = dst + 8 + (i / 4) * 3;
                        let q0 = codes[i] & 63;
                        let q1 = codes[i + 1] & 63;
                        let q2 = codes[i + 2] & 63;
                        let q3 = codes[i + 3] & 63;
                        blob[bo] = q0 | (q1 << 6);
                        blob[bo + 1] = (q1 >> 2) | (q2 << 4);
                        blob[bo + 2] = (q2 >> 4) | (q3 << 2);
                    }
                }
                _ => unreachable!(),
            }
        }
    }
    blob
}

fn is_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn variance(v: &[f32]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
    v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / v.len() as f64
}

fn check_raw_bits(
    label: &str,
    bits: u8,
    n: usize,
    waves: usize,
    got: &[f32],
    want: &[f32],
) -> bool {
    let finite = is_finite(got) && is_finite(want);
    let var_got = variance(got);
    let var_want = variance(want);
    let nondeg = var_got > 1e-12 && var_want > 1e-12;
    let bit_eq = got.len() == want.len()
        && got
            .iter()
            .zip(want.iter())
            .all(|(g, w)| g.to_bits() == w.to_bits());
    let ok = bit_eq && finite && nondeg;
    let status = if ok { "PASS" } else { "FAIL" };
    let mism = if bit_eq {
        0
    } else {
        got.iter()
            .zip(want.iter())
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count()
    };
    eprintln!(
        "  [bits={bits} {label} N={n} MW{waves}] bitEq={bit_eq} mism={mism} finite={finite} nondeg={nondeg} (var {var_got:.3e}/{var_want:.3e}) [{status}]"
    );
    ok
}

fn htod_f32(gpu: &Gpu, t: &GpuTensor, host: &[f32]) {
    gpu.hip
        .memcpy_htod(&t.buf, unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
        })
        .expect("htod f32");
}

fn fill_f32(gpu: &Gpu, t: &GpuTensor, len: usize, val: f32) {
    let host = vec![val; len];
    htod_f32(gpu, t, &host);
}

fn fill_f32_quiet_nan(gpu: &mut Gpu, tensor: &GpuTensor, payload_bits: u32) {
    let v = f32::from_bits(payload_bits);
    assert!(v.is_nan(), "sentinel payload must be quiet NaN bits");
    gpu.fill_f32(tensor, v)
        .unwrap_or_else(|e| panic!("fill quiet-NaN sentinel 0x{payload_bits:08x}: {e:?}"));
}

fn upload_blob(gpu: &Gpu, blob: &[u8]) -> GpuTensor {
    gpu.upload_raw(blob, &[blob.len()]).expect("upload_raw")
}

fn with_base_path<R>(gpu: &mut Gpu, f: impl FnOnce(&mut Gpu) -> R) -> R {
    let saved = gpu.graphs.capture_mode;
    gpu.graphs.capture_mode = true;
    gpu.hip.device_synchronize().unwrap();
    let out = f(gpu);
    gpu.graphs.capture_mode = saved;
    out
}

fn launch_base_gate_up(
    gpu: &mut Gpu,
    bits: u8,
    a_gate: &GpuTensor,
    a_up: &GpuTensor,
    x: &GpuTensor,
    y_gate: &GpuTensor,
    y_up: &GpuTensor,
    n: usize,
) -> Result<(), String> {
    let r = match bits {
        3 => {
            gpu.gemm_gate_up_mq3g256v2_wmma_gfx11(a_gate, a_up, x, y_gate, y_up, GATE_M, UP_M, K, n)
        }
        4 => gpu.gemm_gate_up_mq4g256v2_wmma(a_gate, a_up, x, y_gate, y_up, GATE_M, UP_M, K, n),
        5 => {
            gpu.gemm_gate_up_mq5g256v2_wmma_gfx11(a_gate, a_up, x, y_gate, y_up, GATE_M, UP_M, K, n)
        }
        6 => {
            gpu.gemm_gate_up_mq6g256v2_wmma_gfx11(a_gate, a_up, x, y_gate, y_up, GATE_M, UP_M, K, n)
        }
        _ => unreachable!(),
    };
    r.map_err(|e| format!("{e:?}"))
}

fn launch_base_residual(
    gpu: &mut Gpu,
    bits: u8,
    a: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    n: usize,
) -> Result<(), String> {
    let r = match bits {
        3 => gpu.gemm_mq3g256v2_residual_wmma_gfx11(a, x, y, RESID_M, K, n),
        4 => gpu.gemm_mq4g256v2_residual_wmma(a, x, y, RESID_M, K, n),
        5 => gpu.gemm_mq5g256v2_residual_wmma_gfx11(a, x, y, RESID_M, K, n),
        6 => gpu.gemm_mq6g256v2_residual_wmma_gfx11(a, x, y, RESID_M, K, n),
        _ => unreachable!(),
    };
    r.map_err(|e| format!("{e:?}"))
}

fn launch_base_qkvza(
    gpu: &mut Gpu,
    bits: u8,
    a_qkv: &GpuTensor,
    a_z: &GpuTensor,
    a_beta: &GpuTensor,
    a_alpha: &GpuTensor,
    x: &GpuTensor,
    y_qkv: &GpuTensor,
    y_z: &GpuTensor,
    y_beta: &GpuTensor,
    y_alpha: &GpuTensor,
    n: usize,
) -> Result<(), String> {
    let r = match bits {
        3 => gpu.gemm_qkvza_mq3g256v2_wmma_gfx11(
            a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, QKVZA_QKV_M,
            QKVZA_Z_M, QKVZA_BETA_M, QKVZA_ALPHA_M, K, n,
        ),
        4 => gpu.gemm_qkvza_mq4g256v2_wmma(
            a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, QKVZA_QKV_M,
            QKVZA_Z_M, QKVZA_BETA_M, QKVZA_ALPHA_M, K, n,
        ),
        5 => gpu.gemm_qkvza_mq5g256v2_wmma_gfx11(
            a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, QKVZA_QKV_M,
            QKVZA_Z_M, QKVZA_BETA_M, QKVZA_ALPHA_M, K, n,
        ),
        6 => gpu.gemm_qkvza_mq6g256v2_wmma_gfx11(
            a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, QKVZA_QKV_M,
            QKVZA_Z_M, QKVZA_BETA_M, QKVZA_ALPHA_M, K, n,
        ),
        _ => unreachable!(),
    };
    r.map_err(|e| format!("{e:?}"))
}

fn launch_base_qkvza_tail(
    gpu: &mut Gpu,
    bits: u8,
    a_qkv: &GpuTensor,
    a_z: &GpuTensor,
    a_beta: &GpuTensor,
    a_alpha: &GpuTensor,
    x: &GpuTensor,
    y_qkv: &GpuTensor,
    y_z: &GpuTensor,
    y_beta: &GpuTensor,
    y_alpha: &GpuTensor,
    n: usize,
) -> Result<(), String> {
    let r = match bits {
        3 => gpu.gemm_qkvza_mq3g256v2_wmma_gfx11(
            a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, QKVZA_QKV_M_TAIL,
            QKVZA_Z_M, QKVZA_BETA_M, QKVZA_ALPHA_M, K, n,
        ),
        4 => gpu.gemm_qkvza_mq4g256v2_wmma(
            a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, QKVZA_QKV_M_TAIL,
            QKVZA_Z_M, QKVZA_BETA_M, QKVZA_ALPHA_M, K, n,
        ),
        5 => gpu.gemm_qkvza_mq5g256v2_wmma_gfx11(
            a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, QKVZA_QKV_M_TAIL,
            QKVZA_Z_M, QKVZA_BETA_M, QKVZA_ALPHA_M, K, n,
        ),
        6 => gpu.gemm_qkvza_mq6g256v2_wmma_gfx11(
            a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, QKVZA_QKV_M_TAIL,
            QKVZA_Z_M, QKVZA_BETA_M, QKVZA_ALPHA_M, K, n,
        ),
        _ => unreachable!(),
    };
    r.map_err(|e| format!("{e:?}"))
}

fn launch_base_qkv(
    gpu: &mut Gpu,
    bits: u8,
    a_q: &GpuTensor,
    a_k: &GpuTensor,
    a_v: &GpuTensor,
    x: &GpuTensor,
    y_q: &GpuTensor,
    y_k: &GpuTensor,
    y_v: &GpuTensor,
    n: usize,
) -> Result<(), String> {
    let r = match bits {
        3 => gpu.gemm_qkv_mq3g256v2_wmma_gfx11(
            a_q, a_k, a_v, x, y_q, y_k, y_v, QKV_Q_M, QKV_K_M, QKV_V_M, K, n,
        ),
        4 => gpu.gemm_qkv_mq4g256v2_wmma(a_q, a_k, a_v, x, y_q, y_k, y_v, QKV_Q_M, QKV_K_M, QKV_V_M, K, n),
        5 => gpu.gemm_qkv_mq5g256v2_wmma_gfx11(
            a_q, a_k, a_v, x, y_q, y_k, y_v, QKV_Q_M, QKV_K_M, QKV_V_M, K, n,
        ),
        6 => gpu.gemm_qkv_mq6g256v2_wmma_gfx11(
            a_q, a_k, a_v, x, y_q, y_k, y_v, QKV_Q_M, QKV_K_M, QKV_V_M, K, n,
        ),
        _ => unreachable!(),
    };
    r.map_err(|e| format!("{e:?}"))
}

fn launch_base_qkv_tail(
    gpu: &mut Gpu,
    bits: u8,
    a_q: &GpuTensor,
    a_k: &GpuTensor,
    a_v: &GpuTensor,
    x: &GpuTensor,
    y_q: &GpuTensor,
    y_k: &GpuTensor,
    y_v: &GpuTensor,
    n: usize,
) -> Result<(), String> {
    let r = match bits {
        3 => gpu.gemm_qkv_mq3g256v2_wmma_gfx11(
            a_q, a_k, a_v, x, y_q, y_k, y_v, QKV_Q_M_TAIL, QKV_K_M, QKV_V_M, K, n,
        ),
        4 => gpu.gemm_qkv_mq4g256v2_wmma(
            a_q, a_k, a_v, x, y_q, y_k, y_v, QKV_Q_M_TAIL, QKV_K_M, QKV_V_M, K, n,
        ),
        5 => gpu.gemm_qkv_mq5g256v2_wmma_gfx11(
            a_q, a_k, a_v, x, y_q, y_k, y_v, QKV_Q_M_TAIL, QKV_K_M, QKV_V_M, K, n,
        ),
        6 => gpu.gemm_qkv_mq6g256v2_wmma_gfx11(
            a_q, a_k, a_v, x, y_q, y_k, y_v, QKV_Q_M_TAIL, QKV_K_M, QKV_V_M, K, n,
        ),
        _ => unreachable!(),
    };
    r.map_err(|e| format!("{e:?}"))
}

fn run_gate_up(gpu: &mut Gpu, bits: u8, n: usize, waves: usize, x_host: &[f32]) -> bool {
    let gb = group_bytes(bits);
    let w_gate = build_disjoint_halves_seeded(GATE_M, K, SALT_GATE ^ (bits as u32) << 4);
    let w_up = build_disjoint_halves_seeded(UP_M, K, SALT_UP ^ (bits as u32) << 4);
    let b_gate = pack_mqv2(bits, &w_gate, GATE_M, K);
    let b_up = pack_mqv2(bits, &w_up, UP_M, K);
    assert_eq!(b_gate.len(), GATE_M * (K / GROUP) * gb);
    assert_eq!(b_up.len(), UP_M * (K / GROUP) * gb);
    assert_ne!(b_gate, b_up, "gate/up packed blobs must differ");
    let d_gate = upload_blob(gpu, &b_gate);
    let d_up = upload_blob(gpu, &b_up);
    let d_x = gpu.alloc_tensor(&[n * K], DType::F32).expect("x");
    htod_f32(gpu, &d_x, x_host);

    let sen_g = f32::from_bits(NAN_GATE_BITS);
    let sen_u = f32::from_bits(NAN_UP_BITS);
    let d_yg_ref = gpu.alloc_tensor(&[n * GATE_M], DType::F32).expect("yg ref");
    let d_yu_ref = gpu.alloc_tensor(&[n * UP_M], DType::F32).expect("yu ref");
    fill_f32_quiet_nan(gpu, &d_yg_ref, NAN_GATE_BITS);
    fill_f32_quiet_nan(gpu, &d_yu_ref, NAN_UP_BITS);
    gpu.hip.device_synchronize().unwrap();

    if let Err(e) = with_base_path(gpu, |gpu| {
        launch_base_gate_up(gpu, bits, &d_gate, &d_up, &d_x, &d_yg_ref, &d_yu_ref, n)
    }) {
        eprintln!("  [bits={bits} gate_up N={n} MW{waves}] base launch FAIL: {e}");
        return false;
    }
    gpu.hip.device_synchronize().unwrap();
    let yg_ref = gpu.download_f32(&d_yg_ref).expect("dl g ref");
    let yu_ref = gpu.download_f32(&d_yu_ref).expect("dl u ref");

    let d_yg_mw = gpu.alloc_tensor(&[n * GATE_M], DType::F32).expect("yg mw");
    let d_yu_mw = gpu.alloc_tensor(&[n * UP_M], DType::F32).expect("yu mw");
    fill_f32_quiet_nan(gpu, &d_yg_mw, NAN_GATE_BITS);
    fill_f32_quiet_nan(gpu, &d_yu_mw, NAN_UP_BITS);
    gpu.hip.device_synchronize().unwrap();
    if let Err(e) = gpu.gemm_gate_up_mqv2_wmma_gfx11_mw_lds(
        bits, waves, &d_gate, &d_up, &d_x, &d_yg_mw, &d_yu_mw, GATE_M, UP_M, K, n,
    ) {
        eprintln!("  [bits={bits} gate_up N={n} MW{waves}] candidate launch FAIL: {e:?}");
        return false;
    }
    gpu.hip.device_synchronize().unwrap();
    let yg_mw = gpu.download_f32(&d_yg_mw).expect("dl g mw");
    let yu_mw = gpu.download_f32(&d_yu_mw).expect("dl u mw");

    let mut ok = true;
    for (lab, got, want, sen) in [
        ("gate_up.gate", &yg_mw, &yg_ref, sen_g),
        ("gate_up.up", &yu_mw, &yu_ref, sen_u),
    ] {
        let cnt = got.iter().filter(|x| x.to_bits() == sen.to_bits()).count();
        if cnt != 0 {
            eprintln!("  [bits={bits} {lab} N={n} MW{waves}] {cnt} sentinel(s) remain");
            ok = false;
        }
        if !check_raw_bits(lab, bits, n, waves, got, want) {
            ok = false;
        }
    }
    ok
}

fn run_residual(gpu: &mut Gpu, bits: u8, n: usize, waves: usize, x_host: &[f32]) -> bool {
    let gb = group_bytes(bits);
    let w = build_disjoint_halves_seeded(RESID_M, K, SALT_RESID ^ (bits as u32));
    let blob = pack_mqv2(bits, &w, RESID_M, K);
    assert_eq!(blob.len(), RESID_M * (K / GROUP) * gb);
    let d_a = upload_blob(gpu, &blob);
    let d_x = gpu.alloc_tensor(&[n * K], DType::F32).expect("x");
    htod_f32(gpu, &d_x, x_host);

    // Identical nonzero Y init for base and candidate (Y += W@X).
    let y_init: Vec<f32> = (0..n * RESID_M)
        .map(|i| prng(i, 0xBEEF_1234 ^ (bits as u32) ^ (n as u32)) * 2.0 - 1.0 + 0.5)
        .collect();
    let sen = f32::from_bits(0x7FC0_00FF);

    let d_y_ref = gpu.alloc_tensor(&[n * RESID_M], DType::F32).expect("y ref");
    fill_f32(gpu, &d_y_ref, n * RESID_M, sen);
    htod_f32(gpu, &d_y_ref, &y_init);
    gpu.hip.device_synchronize().unwrap();

    if let Err(e) = with_base_path(gpu, |gpu| {
        launch_base_residual(gpu, bits, &d_a, &d_x, &d_y_ref, n)
    }) {
        eprintln!("  [bits={bits} residual N={n} MW{waves}] base launch FAIL: {e}");
        return false;
    }
    gpu.hip.device_synchronize().unwrap();
    let y_ref = gpu.download_f32(&d_y_ref).expect("dl ref");
    if y_ref.iter().any(|x| x.to_bits() == sen.to_bits()) {
        eprintln!("  [bits={bits} residual N={n} MW{waves}] base left sentinel");
        return false;
    }

    let d_y_mw = gpu.alloc_tensor(&[n * RESID_M], DType::F32).expect("y mw");
    fill_f32(gpu, &d_y_mw, n * RESID_M, sen);
    htod_f32(gpu, &d_y_mw, &y_init);
    gpu.hip.device_synchronize().unwrap();
    if let Err(e) =
        gpu.gemm_mqv2_residual_wmma_gfx11_mw_lds(bits, waves, &d_a, &d_x, &d_y_mw, RESID_M, K, n)
    {
        eprintln!("  [bits={bits} residual N={n} MW{waves}] candidate launch FAIL: {e:?}");
        return false;
    }
    gpu.hip.device_synchronize().unwrap();
    let y_mw = gpu.download_f32(&d_y_mw).expect("dl mw");
    let cnt = y_mw.iter().filter(|x| x.to_bits() == sen.to_bits()).count();
    let mut ok = true;
    if cnt != 0 {
        eprintln!("  [bits={bits} residual N={n} MW{waves}] {cnt} sentinel(s) remain");
        ok = false;
    }
    if !check_raw_bits("residual", bits, n, waves, &y_mw, &y_ref) {
        ok = false;
    }
    ok
}

fn run_qkvza(gpu: &mut Gpu, bits: u8, n: usize, waves: usize, x_host: &[f32]) -> bool {
    let gb = group_bytes(bits);
    let w_qkv = build_disjoint_halves_seeded(QKVZA_QKV_M, K, 0x1111_2222 ^ (bits as u32) << 16);
    let w_z = build_disjoint_halves_seeded(QKVZA_Z_M, K, 0x3333_4444 ^ (bits as u32) << 16);
    let w_beta = build_disjoint_halves_seeded(QKVZA_BETA_M, K, 0x5555_6666 ^ (bits as u32) << 16);
    let w_alpha = build_disjoint_halves_seeded(QKVZA_ALPHA_M, K, 0x7777_8888 ^ (bits as u32) << 16);
    let b_qkv = pack_mqv2(bits, &w_qkv, QKVZA_QKV_M, K);
    let b_z = pack_mqv2(bits, &w_z, QKVZA_Z_M, K);
    let b_beta = pack_mqv2(bits, &w_beta, QKVZA_BETA_M, K);
    let b_alpha = pack_mqv2(bits, &w_alpha, QKVZA_ALPHA_M, K);
    assert_eq!(b_qkv.len(), QKVZA_QKV_M * (K / GROUP) * gb);
    let d_qkv = upload_blob(gpu, &b_qkv);
    let d_z = upload_blob(gpu, &b_z);
    let d_beta = upload_blob(gpu, &b_beta);
    let d_alpha = upload_blob(gpu, &b_alpha);
    let d_x = gpu.alloc_tensor(&[n * K], DType::F32).expect("x");
    htod_f32(gpu, &d_x, x_host);

    let sen = [
        f32::from_bits(0x7FC0_0001),
        f32::from_bits(0x7FC0_0002),
        f32::from_bits(0x7FC0_0003),
        f32::from_bits(0x7FC0_0004),
    ];
    let lens = [
        n * QKVZA_QKV_M,
        n * QKVZA_Z_M,
        n * QKVZA_BETA_M,
        n * QKVZA_ALPHA_M,
    ];
    let d_y_ref: [GpuTensor; 4] = std::array::from_fn(|i| {
        let t = gpu.alloc_tensor(&[lens[i]], DType::F32).expect("y ref");
        fill_f32_quiet_nan(gpu, &t, sen[i].to_bits());
        t
    });
    gpu.hip.device_synchronize().unwrap();

    if let Err(e) = with_base_path(gpu, |gpu| {
        launch_base_qkvza(
            gpu, bits, &d_qkv, &d_z, &d_beta, &d_alpha, &d_x, &d_y_ref[0], &d_y_ref[1],
            &d_y_ref[2], &d_y_ref[3], n,
        )
    }) {
        eprintln!("  [bits={bits} qkvza N={n} MW{waves}] base launch FAIL: {e}");
        return false;
    }
    gpu.hip.device_synchronize().unwrap();
    let y_ref: [Vec<f32>; 4] =
        std::array::from_fn(|i| gpu.download_f32(&d_y_ref[i]).expect("dl ref"));

    let d_y_mw: [GpuTensor; 4] = std::array::from_fn(|i| {
        let t = gpu.alloc_tensor(&[lens[i]], DType::F32).expect("y mw");
        fill_f32_quiet_nan(gpu, &t, sen[i].to_bits());
        t
    });
    gpu.hip.device_synchronize().unwrap();
    if let Err(e) = gpu.gemm_qkvza_mqv2_wmma_gfx11_mw_lds(
        bits,
        waves,
        &d_qkv,
        &d_z,
        &d_beta,
        &d_alpha,
        &d_x,
        &d_y_mw[0],
        &d_y_mw[1],
        &d_y_mw[2],
        &d_y_mw[3],
        QKVZA_QKV_M,
        QKVZA_Z_M,
        QKVZA_BETA_M,
        QKVZA_ALPHA_M,
        K,
        n,
    ) {
        eprintln!("  [bits={bits} qkvza N={n} MW{waves}] candidate launch FAIL: {e:?}");
        return false;
    }
    gpu.hip.device_synchronize().unwrap();
    let y_mw: [Vec<f32>; 4] = std::array::from_fn(|i| gpu.download_f32(&d_y_mw[i]).expect("dl mw"));

    let labels = ["qkvza.qkv", "qkvza.z", "qkvza.beta", "qkvza.alpha"];
    let mut ok = true;
    for i in 0..4 {
        let cnt = y_mw[i]
            .iter()
            .filter(|x| x.to_bits() == sen[i].to_bits())
            .count();
        if cnt != 0 {
            eprintln!(
                "  [bits={bits} {} N={n} MW{waves}] {cnt} sentinel(s) remain",
                labels[i]
            );
            ok = false;
        }
        if !check_raw_bits(labels[i], bits, n, waves, &y_mw[i], &y_ref[i]) {
            ok = false;
        }
    }
    ok
}

fn run_qkvza_m_tail(gpu: &mut Gpu, bits: u8, n: usize, waves: usize, x_host: &[f32]) -> bool {
    let gb = group_bytes(bits);
    let w_qkv = build_disjoint_halves_seeded(QKVZA_QKV_M_TAIL, K, 0x1111_2222 ^ (bits as u32) << 16);
    let w_z = build_disjoint_halves_seeded(QKVZA_Z_M, K, 0x3333_4444 ^ (bits as u32) << 16);
    let w_beta = build_disjoint_halves_seeded(QKVZA_BETA_M, K, 0x5555_6666 ^ (bits as u32) << 16);
    let w_alpha = build_disjoint_halves_seeded(QKVZA_ALPHA_M, K, 0x7777_8888 ^ (bits as u32) << 16);
    let b_qkv = pack_mqv2(bits, &w_qkv, QKVZA_QKV_M_TAIL, K);
    let b_z = pack_mqv2(bits, &w_z, QKVZA_Z_M, K);
    let b_beta = pack_mqv2(bits, &w_beta, QKVZA_BETA_M, K);
    let b_alpha = pack_mqv2(bits, &w_alpha, QKVZA_ALPHA_M, K);
    assert_eq!(b_qkv.len(), QKVZA_QKV_M_TAIL * (K / GROUP) * gb);
    let d_qkv = upload_blob(gpu, &b_qkv);
    let d_z = upload_blob(gpu, &b_z);
    let d_beta = upload_blob(gpu, &b_beta);
    let d_alpha = upload_blob(gpu, &b_alpha);
    let d_x = gpu.alloc_tensor(&[n * K], DType::F32).expect("x");
    htod_f32(gpu, &d_x, x_host);

    let sen = [
        f32::from_bits(0x7FC0_0001),
        f32::from_bits(0x7FC0_0002),
        f32::from_bits(0x7FC0_0003),
        f32::from_bits(0x7FC0_0004),
    ];
    let lens = [
        n * QKVZA_QKV_M_TAIL,
        n * QKVZA_Z_M,
        n * QKVZA_BETA_M,
        n * QKVZA_ALPHA_M,
    ];
    let d_y_ref: [GpuTensor; 4] = std::array::from_fn(|i| {
        let t = gpu.alloc_tensor(&[lens[i]], DType::F32).expect("y ref");
        fill_f32_quiet_nan(gpu, &t, sen[i].to_bits());
        t
    });
    gpu.hip.device_synchronize().unwrap();

    if let Err(e) = with_base_path(gpu, |gpu| {
        launch_base_qkvza_tail(
            gpu, bits, &d_qkv, &d_z, &d_beta, &d_alpha, &d_x, &d_y_ref[0], &d_y_ref[1],
            &d_y_ref[2], &d_y_ref[3], n,
        )
    }) {
        eprintln!("  [bits={bits} qkvza_tail N={n} MW{waves}] base launch FAIL: {e}");
        return false;
    }
    gpu.hip.device_synchronize().unwrap();
    let y_ref: [Vec<f32>; 4] =
        std::array::from_fn(|i| gpu.download_f32(&d_y_ref[i]).expect("dl ref"));

    let d_y_mw: [GpuTensor; 4] = std::array::from_fn(|i| {
        let t = gpu.alloc_tensor(&[lens[i]], DType::F32).expect("y mw");
        fill_f32_quiet_nan(gpu, &t, sen[i].to_bits());
        t
    });
    gpu.hip.device_synchronize().unwrap();
    if let Err(e) = gpu.gemm_qkvza_mqv2_wmma_gfx11_mw_lds(
        bits,
        waves,
        &d_qkv,
        &d_z,
        &d_beta,
        &d_alpha,
        &d_x,
        &d_y_mw[0],
        &d_y_mw[1],
        &d_y_mw[2],
        &d_y_mw[3],
        QKVZA_QKV_M_TAIL,
        QKVZA_Z_M,
        QKVZA_BETA_M,
        QKVZA_ALPHA_M,
        K,
        n,
    ) {
        eprintln!("  [bits={bits} qkvza_tail N={n} MW{waves}] candidate launch FAIL: {e:?}");
        return false;
    }
    gpu.hip.device_synchronize().unwrap();
    let y_mw: [Vec<f32>; 4] = std::array::from_fn(|i| gpu.download_f32(&d_y_mw[i]).expect("dl mw"));

    let labels = [
        "qkvza_tail.qkv",
        "qkvza_tail.z",
        "qkvza_tail.beta",
        "qkvza_tail.alpha",
    ];
    let mut ok = true;
    for i in 0..4 {
        let cnt = y_mw[i]
            .iter()
            .filter(|x| x.to_bits() == sen[i].to_bits())
            .count();
        if cnt != 0 {
            eprintln!(
                "  [bits={bits} {} N={n} MW{waves}] {cnt} sentinel(s) remain",
                labels[i]
            );
            ok = false;
        }
        if !check_raw_bits(labels[i], bits, n, waves, &y_mw[i], &y_ref[i]) {
            ok = false;
        }
    }
    ok
}

fn run_qkv(gpu: &mut Gpu, bits: u8, n: usize, waves: usize, x_host: &[f32]) -> bool {
    let gb = group_bytes(bits);
    let w_q = build_disjoint_halves_seeded(QKV_Q_M, K, 0xA111_2222 ^ (bits as u32) << 8);
    let w_k = build_disjoint_halves_seeded(QKV_K_M, K, 0xA333_4444 ^ (bits as u32) << 8);
    let w_v = build_disjoint_halves_seeded(QKV_V_M, K, 0xA555_6666 ^ (bits as u32) << 8);
    let b_q = pack_mqv2(bits, &w_q, QKV_Q_M, K);
    let b_k = pack_mqv2(bits, &w_k, QKV_K_M, K);
    let b_v = pack_mqv2(bits, &w_v, QKV_V_M, K);
    assert_eq!(b_q.len(), QKV_Q_M * (K / GROUP) * gb);
    let d_q = upload_blob(gpu, &b_q);
    let d_k = upload_blob(gpu, &b_k);
    let d_v = upload_blob(gpu, &b_v);
    let d_x = gpu.alloc_tensor(&[n * K], DType::F32).expect("x");
    htod_f32(gpu, &d_x, x_host);

    let sen = [
        f32::from_bits(0x7FC0_0011),
        f32::from_bits(0x7FC0_0012),
        f32::from_bits(0x7FC0_0013),
    ];
    let lens = [n * QKV_Q_M, n * QKV_K_M, n * QKV_V_M];
    let d_y_ref: [GpuTensor; 3] = std::array::from_fn(|i| {
        let t = gpu.alloc_tensor(&[lens[i]], DType::F32).expect("y ref");
        fill_f32_quiet_nan(gpu, &t, sen[i].to_bits());
        t
    });
    gpu.hip.device_synchronize().unwrap();

    if let Err(e) = with_base_path(gpu, |gpu| {
        launch_base_qkv(gpu, bits, &d_q, &d_k, &d_v, &d_x, &d_y_ref[0], &d_y_ref[1], &d_y_ref[2], n)
    }) {
        eprintln!("  [bits={bits} qkv N={n} MW{waves}] base launch FAIL: {e}");
        return false;
    }
    gpu.hip.device_synchronize().unwrap();
    let y_ref: [Vec<f32>; 3] =
        std::array::from_fn(|i| gpu.download_f32(&d_y_ref[i]).expect("dl ref"));

    let d_y_mw: [GpuTensor; 3] = std::array::from_fn(|i| {
        let t = gpu.alloc_tensor(&[lens[i]], DType::F32).expect("y mw");
        fill_f32_quiet_nan(gpu, &t, sen[i].to_bits());
        t
    });
    gpu.hip.device_synchronize().unwrap();
    if let Err(e) = gpu.gemm_qkv_mqv2_wmma_gfx11_mw_lds(
        bits, waves, &d_q, &d_k, &d_v, &d_x, &d_y_mw[0], &d_y_mw[1], &d_y_mw[2], QKV_Q_M,
        QKV_K_M, QKV_V_M, K, n,
    ) {
        eprintln!("  [bits={bits} qkv N={n} MW{waves}] candidate launch FAIL: {e:?}");
        return false;
    }
    gpu.hip.device_synchronize().unwrap();
    let y_mw: [Vec<f32>; 3] = std::array::from_fn(|i| gpu.download_f32(&d_y_mw[i]).expect("dl mw"));

    let labels = ["qkv.q", "qkv.k", "qkv.v"];
    let mut ok = true;
    for i in 0..3 {
        let cnt = y_mw[i]
            .iter()
            .filter(|x| x.to_bits() == sen[i].to_bits())
            .count();
        if cnt != 0 {
            eprintln!(
                "  [bits={bits} {} N={n} MW{waves}] {cnt} sentinel(s) remain",
                labels[i]
            );
            ok = false;
        }
        if !check_raw_bits(labels[i], bits, n, waves, &y_mw[i], &y_ref[i]) {
            ok = false;
        }
    }
    ok
}

fn run_qkv_m_tail(gpu: &mut Gpu, bits: u8, n: usize, waves: usize, x_host: &[f32]) -> bool {
    let gb = group_bytes(bits);
    let w_q = build_disjoint_halves_seeded(QKV_Q_M_TAIL, K, 0xA111_2222 ^ (bits as u32) << 8);
    let w_k = build_disjoint_halves_seeded(QKV_K_M, K, 0xA333_4444 ^ (bits as u32) << 8);
    let w_v = build_disjoint_halves_seeded(QKV_V_M, K, 0xA555_6666 ^ (bits as u32) << 8);
    let b_q = pack_mqv2(bits, &w_q, QKV_Q_M_TAIL, K);
    let b_k = pack_mqv2(bits, &w_k, QKV_K_M, K);
    let b_v = pack_mqv2(bits, &w_v, QKV_V_M, K);
    assert_eq!(b_q.len(), QKV_Q_M_TAIL * (K / GROUP) * gb);
    let d_q = upload_blob(gpu, &b_q);
    let d_k = upload_blob(gpu, &b_k);
    let d_v = upload_blob(gpu, &b_v);
    let d_x = gpu.alloc_tensor(&[n * K], DType::F32).expect("x");
    htod_f32(gpu, &d_x, x_host);

    let sen = [
        f32::from_bits(0x7FC0_0011),
        f32::from_bits(0x7FC0_0012),
        f32::from_bits(0x7FC0_0013),
    ];
    let lens = [n * QKV_Q_M_TAIL, n * QKV_K_M, n * QKV_V_M];
    let d_y_ref: [GpuTensor; 3] = std::array::from_fn(|i| {
        let t = gpu.alloc_tensor(&[lens[i]], DType::F32).expect("y ref");
        fill_f32_quiet_nan(gpu, &t, sen[i].to_bits());
        t
    });
    gpu.hip.device_synchronize().unwrap();

    if let Err(e) = with_base_path(gpu, |gpu| {
        launch_base_qkv_tail(gpu, bits, &d_q, &d_k, &d_v, &d_x, &d_y_ref[0], &d_y_ref[1], &d_y_ref[2], n)
    }) {
        eprintln!("  [bits={bits} qkv_tail N={n} MW{waves}] base launch FAIL: {e}");
        return false;
    }
    gpu.hip.device_synchronize().unwrap();
    let y_ref: [Vec<f32>; 3] =
        std::array::from_fn(|i| gpu.download_f32(&d_y_ref[i]).expect("dl ref"));

    let d_y_mw: [GpuTensor; 3] = std::array::from_fn(|i| {
        let t = gpu.alloc_tensor(&[lens[i]], DType::F32).expect("y mw");
        fill_f32_quiet_nan(gpu, &t, sen[i].to_bits());
        t
    });
    gpu.hip.device_synchronize().unwrap();
    if let Err(e) = gpu.gemm_qkv_mqv2_wmma_gfx11_mw_lds(
        bits, waves, &d_q, &d_k, &d_v, &d_x, &d_y_mw[0], &d_y_mw[1], &d_y_mw[2], QKV_Q_M_TAIL,
        QKV_K_M, QKV_V_M, K, n,
    ) {
        eprintln!("  [bits={bits} qkv_tail N={n} MW{waves}] candidate launch FAIL: {e:?}");
        return false;
    }
    gpu.hip.device_synchronize().unwrap();
    let y_mw: [Vec<f32>; 3] = std::array::from_fn(|i| gpu.download_f32(&d_y_mw[i]).expect("dl mw"));

    let labels = ["qkv_tail.q", "qkv_tail.k", "qkv_tail.v"];
    let mut ok = true;
    for i in 0..3 {
        let cnt = y_mw[i]
            .iter()
            .filter(|x| x.to_bits() == sen[i].to_bits())
            .count();
        if cnt != 0 {
            eprintln!(
                "  [bits={bits} {} N={n} MW{waves}] {cnt} sentinel(s) remain",
                labels[i]
            );
            ok = false;
        }
        if !check_raw_bits(labels[i], bits, n, waves, &y_mw[i], &y_ref[i]) {
            ok = false;
        }
    }
    ok
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };

    let arch = gpu.arch.clone();
    let is_gfx1100 = gpu.arch_caps.is_gfx1100() && arch == "gfx1100";
    let is_gfx1151 = gpu.arch_caps.is_gfx1151() && arch == "gfx1151";
    if !is_gfx1100 && !is_gfx1151 {
        eprintln!("SKIP: arch {arch} is not exact gfx1100 or gfx1151");
        return;
    }

    // gfx1100 forbids MQ3 runtime execution; gfx1151 covers MQ3.
    let bits: &[u8] = if is_gfx1151 {
        &[3, 4, 5, 6]
    } else {
        &[4, 5, 6]
    };
    // gate 2 + resid 2 + qkv 4 + qkvza 4 = 12 per bits; +2 per bits for M-tail (qkv/qkvza _tail)
    let tail_per_bits = 2;
    let expected_arms = bits.len() * (GATE_NS.len() + RESID_NS.len() + QKV_MW_NS.len() + QKVZA_MW_NS.len() + tail_per_bits);

    eprintln!(
        "arch {arch} confirmed — MQ V2 gfx11 MW_LDS raw-bit parity (bits={bits:?}, expected_arms={expected_arms})"
    );
    eprintln!(
        "shapes: gate_up=({GATE_M},{UP_M}) residual={RESID_M} qkv=({QKV_Q_M},{QKV_K_M},{QKV_V_M}) qkvza=({QKVZA_QKV_M},{QKVZA_Z_M},{QKVZA_BETA_M},{QKVZA_ALPHA_M}) K={K}"
    );
    eprintln!(
        "  N: gate{{384/MW4,512/MW8}} resid{{416/MW4,464/MW8}} qkv/qkvza{{512/MW4,511/MW4,512/MW8,128/MW8}} + M-tail(q+1) @512/MW4"
    );
    assert!(
        GATE_M > 32 && GATE_M < 48,
        "gate/up boundary row {GATE_M} must lie inside tile covering rows [32, 48)"
    );
    assert_eq!(GATE_M + UP_M, 93);
    assert_eq!((GATE_M + UP_M) % 16, 13);
    assert_eq!(RESID_M % 16, 8);

    if gpu.active_capture.is_some() {
        eprintln!("SKIP: active_capture is Some");
        return;
    }

    let mut all_ok = true;
    let mut arms = 0usize;

    for &b in bits {
        eprintln!(
            "\n======== bits={b} group_bytes={} ========",
            group_bytes(b)
        );

        for &(n, waves) in &GATE_NS {
            eprintln!("--- gate_up N={n} MW{waves} ---");
            let x_host: Vec<f32> = (0..n * K)
                .map(|i| {
                    prng(
                        i,
                        0xC0FF_EE00 ^ ((b as u32) << 20) ^ (n as u32) ^ ((waves as u32) << 8),
                    ) * 2.0
                        - 1.0
                })
                .collect();
            let ok = run_gate_up(&mut gpu, b, n, waves, &x_host);
            arms += 1;
            eprintln!(
                "  ARM bits={b} gate_up N={n} MW{waves}: {}",
                if ok { "PASS" } else { "FAIL" }
            );
            all_ok &= ok;
        }

        for &(n, waves) in &RESID_NS {
            eprintln!("--- residual N={n} MW{waves} ---");
            let x_host: Vec<f32> = (0..n * K)
                .map(|i| {
                    prng(
                        i,
                        0xC0FF_EE00 ^ ((b as u32) << 20) ^ (n as u32) ^ ((waves as u32) << 8),
                    ) * 2.0
                        - 1.0
                })
                .collect();
            let ok = run_residual(&mut gpu, b, n, waves, &x_host);
            arms += 1;
            eprintln!(
                "  ARM bits={b} residual N={n} MW{waves}: {}",
                if ok { "PASS" } else { "FAIL" }
            );
            all_ok &= ok;
        }

        for &(n, waves) in &QKV_MW_NS {
            eprintln!("--- qkv N={n} MW{waves} ---");
            let x_host: Vec<f32> = (0..n * K)
                .map(|i| {
                    prng(
                        i,
                        0xC0FF_EE00 ^ ((b as u32) << 20) ^ (n as u32) ^ ((waves as u32) << 8) ^ 0xA5A5_0000,
                    ) * 2.0
                        - 1.0
                })
                .collect();
            let ok = run_qkv(&mut gpu, b, n, waves, &x_host);
            arms += 1;
            eprintln!(
                "  ARM bits={b} qkv N={n} MW{waves}: {}",
                if ok { "PASS" } else { "FAIL" }
            );
            all_ok &= ok;
        }

        for &(n, waves) in &QKVZA_MW_NS {
            eprintln!("--- qkvza N={n} MW{waves} ---");
            let x_host: Vec<f32> = (0..n * K)
                .map(|i| {
                    prng(
                        i,
                        0xC0FF_EE00 ^ ((b as u32) << 20) ^ (n as u32) ^ ((waves as u32) << 8) ^ 0x5A5A_0000,
                    ) * 2.0
                        - 1.0
                })
                .collect();
            let ok = run_qkvza(&mut gpu, b, n, waves, &x_host);
            arms += 1;
            eprintln!(
                "  ARM bits={b} qkvza N={n} MW{waves}: {}",
                if ok { "PASS" } else { "FAIL" }
            );
            all_ok &= ok;
        }

        // M-tail: total 121 = 7×16+9 (base supports div_ceil → not 16-aligned-required).
        // One per projection type at N=512 MW4; oracle uses matching tail Ms.
        {
            let (n, waves) = (512, 4);
            eprintln!("--- qkv_tail N={n} MW{waves} (Q={QKV_Q_M_TAIL}) ---");
            let x_host: Vec<f32> = (0..n * K)
                .map(|i| {
                    prng(
                        i,
                        0xC0FF_EE00 ^ ((b as u32) << 20) ^ (n as u32) ^ ((waves as u32) << 8) ^ 0xA5A5_0000 ^ 0x1010,
                    ) * 2.0
                        - 1.0
                })
                .collect();
            let ok = run_qkv_m_tail(&mut gpu, b, n, waves, &x_host);
            arms += 1;
            eprintln!(
                "  ARM bits={b} qkv_tail N={n} MW{waves}: {}",
                if ok { "PASS" } else { "FAIL" }
            );
            all_ok &= ok;
        }
        {
            let (n, waves) = (512, 4);
            eprintln!("--- qkvza_tail N={n} MW{waves} (QKV={QKVZA_QKV_M_TAIL}) ---");
            let x_host: Vec<f32> = (0..n * K)
                .map(|i| {
                    prng(
                        i,
                        0xC0FF_EE00 ^ ((b as u32) << 20) ^ (n as u32) ^ ((waves as u32) << 8) ^ 0x5A5A_0000 ^ 0x1010,
                    ) * 2.0
                        - 1.0
                })
                .collect();
            let ok = run_qkvza_m_tail(&mut gpu, b, n, waves, &x_host);
            arms += 1;
            eprintln!(
                "  ARM bits={b} qkvza_tail N={n} MW{waves}: {}",
                if ok { "PASS" } else { "FAIL" }
            );
            all_ok &= ok;
        }
    }

    assert_eq!(
        arms, expected_arms,
        "expected {expected_arms} arms on {arch}"
    );
    if all_ok {
        eprintln!(
            "\nPASS: all {arms} arms on {arch} (bits{bits:?} × ops{{gate_up×2, residual×2, qkv×4, qkvza×4, tails×2}}) raw f32::to_bits equal; finite/nondegenerate; residual Y+= preserved; qkv/qkvza/gate_up overwrite"
        );
    } else {
        eprintln!("\nFAIL: one or more of {arms} raw-bit parity arms failed on {arch}");
        std::process::exit(1);
    }
}
