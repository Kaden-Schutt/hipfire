//! Runtime behavioral oracle for the V2 family: MQ2V2/MQ3V2/MQ5V2/MQ6V2.
//!
//! Exercises the real production wire layouts for all four new V2 formats
//! before model-scale quantization. Each format is tested via:
//!   * B=1 plain GEMV (`gemv_mq{n}g256v2` — production decode path)
//!   * B=16 batched lm-head GEMM (gfx11+gfx12 WMMA batched lm-head)
//!   * K=512 N=16 fused prefill WMMA via arch-aware production wrappers
//!     `gemm_qkv_mq{n}g256v2_wmma`, `gemm_qkvza_mq{n}g256v2_wmma`,
//!     `gemm_gate_up_mq{n}g256v2_wmma` (gfx12-first, then gfx11 has_wmma_w32)
//!
//! ## Why this oracle exists
//!
//! `mq4v2_parity` and `mq4v2_gemm_parity` guard qt44. The four new V2 siblings
//! (qt47 MQ6V2 200B, qt48 MQ5V2 168B, qt49 MQ3V2 104B, qt50 MQ2V2 72B) share the
//! same header contract — dual fp16 per 128 weights: `[s0 z0 s1 z1]` at
//! `[0..2) [2..4) [4..6) [6..8)`, payload at 8 — but have distinct bit-widths
//! and payload packings. A kernel that aliases one format to another (e.g. MQ2V2
//! decoded as MQ3V2) still runs at full speed, compiles, and passes byte-count
//! checks because the dispatch alias is a single enum variant swap.
//!
//! The discriminating construction makes s0/z0 differ from s1/z1 so a
//! legacy/single-half misroute fails. Fused prefill checks use independently
//! salted/disjoint weight blobs per output tensor so pointer swaps or
//! repeated-A mistakes cannot pass.
//!
//! ### Discriminating fixture
//!
//! For each group of 256 weights, half 0 (weights 0..127) occupies [-1, 1],
//! half 1 (weights 128..255) occupies [96, 160]. Consequences:
//!   * a kernel that reads only half 0's header reconstructs half 1 as ~0
//!     instead of ~128 — relative error >1.0, unmissable;
//!   * a kernel that reads the header as f32 scale/zero reinterprets four fp16
//!     fields as two f32s and produces noise;
//!   * a kernel that swaps halves reconstructs 128 where it should see 0.
//!
//! Two G256 groups per row (K=512) ensures group-stride errors also surface.
//! Payload packing is bit-exact per the production quantizers in
//! `crates/hipfire-quantize/src/quant_mq.rs` and `quant_fwht.rs` (without FWHT).
//!
//! ### What is verified
//!
//! For each format:
//!   1. Exact byte count: `m * gpr * GROUP_BYTES`.
//!   2. Header divergence: at least one group's s0 != s1 or z0 != z1.
//!   3. Sentinel/canary: y buffer has trailing sentinel that kernel must not clobber.
//!   4. B=1 plain GEMV vs CPU dequant+matmul (f64) — worst abs/rel error.
//!   5. B=16 batched GEMM (lm-head) vs CPU batched matmul — worst abs/rel error.
//!   6. Single-half bug baseline: error must be orders of magnitude larger than
//!      correct decode, otherwise fixture is not discriminating.
//!   7. Fused prefill WMMA (qkv / qkvza / gate_up) at K=512 N=16 vs independent
//!      f64 exact-dequant refs; pass iff every family output has rel-RMS <= 0.05.
//!      Launch failure is `FAIL_UNSUPPORTED` (never skipped).
//!
//! Output is machine-parseable (`fmt qt group_bytes m k gpr status ... fused_*`)
//! to preserve evidence.
//!
//! Run: `cargo run --release -p hipfire-runtime --example mqv2_family_parity`
//!
//! Style/helpers reused from `mq4v2_parity.rs`, `mq4v2_gemm_parity.rs`, and
//! `mq4v2_fused_parity.rs`.

use half::f16;
use rdna_compute::{DType, Gpu};

const GROUP: usize = 256;
const HALF: usize = 128;
const GPR_K512: usize = 2; // K=512 => 2 groups per row
const M: usize = 64;
const K: usize = 512;
const BATCH: usize = 16;
/// Fused prefill WMMA shape: K=512, N=16 (one 16-batch tile).
const FUSED_N: usize = 16;
const FUSED_ROW: usize = 16; // per-output M for q/k/v/gate/up/...
const CANARY_F32: f32 = 12345.6789;
const FUSED_TOL_RMS: f64 = 0.05;

// ── format descriptors ─────────────────────────────────────────────────────

struct Format {
    name: &'static str,
    qt: u8,
    group_bytes: usize,
    dtype: DType,
    bits: usize,
    max_q: usize,
}

const FORMATS: [Format; 4] = [
    Format {
        name: "MQ6V2",
        qt: 47,
        group_bytes: rdna_compute::MQ6G256V2_GROUP_BYTES,
        dtype: DType::MQ6G256V2,
        bits: 6,
        max_q: 63,
    },
    Format {
        name: "MQ5V2",
        qt: 48,
        group_bytes: rdna_compute::MQ5G256V2_GROUP_BYTES,
        dtype: DType::MQ5G256V2,
        bits: 5,
        max_q: 31,
    },
    Format {
        name: "MQ3V2",
        qt: 49,
        group_bytes: rdna_compute::MQ3G256V2_GROUP_BYTES,
        dtype: DType::MQ3G256V2,
        bits: 3,
        max_q: 7,
    },
    Format {
        name: "MQ2V2",
        qt: 50,
        group_bytes: rdna_compute::MQ2G256V2_GROUP_BYTES,
        dtype: DType::MQ2G256V2,
        bits: 2,
        max_q: 3,
    },
];

// ── deterministic PRNG (no rand dep) ───────────────────────────────────────

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

fn build_disjoint_halves(m: usize, k: usize) -> Vec<f32> {
    build_disjoint_halves_salted(m, k, 0)
}

/// Independently salted half-discriminating weights. Distinct `salt` per fused
/// output tensor so pointer swaps / repeated-A cannot pass parity.
fn build_disjoint_halves_salted(m: usize, k: usize, salt: u32) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let s = salt
                .wrapping_add((r as u32).wrapping_mul(7919))
                .wrapping_add((g as u32).wrapping_mul(104729));
            for i in 0..HALF {
                w[base + i] = prng(i, s) * 2.0 - 1.0;
            }
            for i in HALF..GROUP {
                w[base + i] = 96.0 + prng(i, s ^ 0xA5A5_A5A5) * 64.0;
            }
        }
    }
    w
}

fn build_x(k: usize, salt: u32) -> Vec<f32> {
    (0..k).map(|i| prng(i, salt) * 2.0 - 1.0).collect()
}

fn build_x_batched(k: usize, b: usize, salt: u32) -> Vec<f32> {
    (0..b * k).map(|i| prng(i, salt) * 2.0 - 1.0).collect()
}

// ── packing helpers (FWHT-free, mirrors production encoders) ───────────────

fn pack_blob(fmt: &Format, w: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % GROUP, 0);
    assert_eq!(w.len(), m * k);
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * fmt.group_bytes];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * fmt.group_bytes;
            // per-half scale/zero
            let mut s_bits = [0u16; 2];
            let mut z_bits = [0u16; 2];
            let mut s_rt = [0.0f32; 2];
            let mut z_rt = [0.0f32; 2];
            let mut degenerate = [false; 2];
            for h in 0..2 {
                let off = h * HALF;
                let slice = &w[src + off..src + off + HALF];
                let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo {
                    (hi - lo) / fmt.max_q as f32
                } else {
                    0.0
                };
                let sb = if hi == lo {
                    0u16
                } else {
                    f16::from_f32(step).to_bits()
                };
                let zb = f16::from_f32(lo).to_bits();
                s_bits[h] = sb;
                z_bits[h] = zb;
                s_rt[h] = f16::from_bits(sb).to_f32();
                z_rt[h] = f16::from_bits(zb).to_f32();
                degenerate[h] = hi == lo || step == 0.0 || s_rt[h] == 0.0;
            }
            // header: s0 z0 s1 z1 little-endian
            blob[dst..dst + 2].copy_from_slice(&s_bits[0].to_le_bytes());
            blob[dst + 2..dst + 4].copy_from_slice(&z_bits[0].to_le_bytes());
            blob[dst + 4..dst + 6].copy_from_slice(&s_bits[1].to_le_bytes());
            blob[dst + 6..dst + 8].copy_from_slice(&z_bits[1].to_le_bytes());

            // quantize per half against round-tripped header
            let mut q = [0u8; 256];
            for h in 0..2 {
                let off = h * HALF;
                if degenerate[h] {
                    for i in 0..HALF {
                        q[off + i] = 0;
                    }
                } else {
                    let inv = 1.0 / s_rt[h];
                    let z = z_rt[h];
                    for i in 0..HALF {
                        let v = w[src + off + i];
                        let qq = ((v - z) * inv + 0.5).floor().clamp(0.0, fmt.max_q as f32) as u8;
                        q[off + i] = qq;
                    }
                }
            }
            // payload packing per bit-width
            match fmt.bits {
                2 => {
                    for i in 0..64 {
                        let mut b = 0u8;
                        for j in 0..4 {
                            let qq = q[4 * i + j] & 3;
                            b |= qq << (j * 2);
                        }
                        blob[dst + 8 + i] = b;
                    }
                }
                3 => {
                    for chunk in 0..32 {
                        let ci = chunk * 8;
                        let qq = [
                            q[ci] & 7,
                            q[ci + 1] & 7,
                            q[ci + 2] & 7,
                            q[ci + 3] & 7,
                            q[ci + 4] & 7,
                            q[ci + 5] & 7,
                            q[ci + 6] & 7,
                            q[ci + 7] & 7,
                        ];
                        let b0 = (qq[0] & 7) | ((qq[1] & 7) << 3) | ((qq[2] & 3) << 6);
                        let b1 = ((qq[2] >> 2) & 1)
                            | ((qq[3] & 7) << 1)
                            | ((qq[4] & 7) << 4)
                            | ((qq[5] & 1) << 7);
                        let b2 = ((qq[5] >> 1) & 3) | ((qq[6] & 7) << 2) | ((qq[7] & 7) << 5);
                        let bo = dst + 8 + chunk * 3;
                        blob[bo] = b0;
                        blob[bo + 1] = b1;
                        blob[bo + 2] = b2;
                    }
                }
                5 => {
                    for i in (0..256).step_by(8) {
                        let bo = dst + 8 + (i / 8) * 5;
                        let q0 = q[i] & 31;
                        let q1 = q[i + 1] & 31;
                        let q2 = q[i + 2] & 31;
                        let q3 = q[i + 3] & 31;
                        let q4 = q[i + 4] & 31;
                        let q5 = q[i + 5] & 31;
                        let q6 = q[i + 6] & 31;
                        let q7 = q[i + 7] & 31;
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
                        let q0 = q[i] & 63;
                        let q1 = q[i + 1] & 63;
                        let q2 = q[i + 2] & 63;
                        let q3 = q[i + 3] & 63;
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

// ── CPU reference decode + matmul (independent of pack) ────────────────────

fn decode_and_gemv(fmt: &Format, blob: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f64> {
    let gpr = k / GROUP;
    let mut y = vec![0.0f64; m];
    for r in 0..m {
        let mut acc = 0.0f64;
        for g in 0..gpr {
            let base = (r * gpr + g) * fmt.group_bytes;
            // header
            let s0 = f16::from_bits(u16::from_le_bytes([blob[base], blob[base + 1]])).to_f32();
            let z0 = f16::from_bits(u16::from_le_bytes([blob[base + 2], blob[base + 3]])).to_f32();
            let s1 = f16::from_bits(u16::from_le_bytes([blob[base + 4], blob[base + 5]])).to_f32();
            let z1 = f16::from_bits(u16::from_le_bytes([blob[base + 6], blob[base + 7]])).to_f32();
            let sc = [s0, s1];
            let zp = [z0, z1];
            let payload = &blob[base + 8..base + fmt.group_bytes];
            // decode 256 weights for this group
            let mut w = [0.0f32; GROUP];
            match fmt.bits {
                2 => {
                    for i in 0..64 {
                        let b = payload[i];
                        for j in 0..4 {
                            let idx = i * 4 + j;
                            let q = ((b >> (j * 2)) & 3) as f32;
                            let h = idx / HALF;
                            w[idx] = zp[h] + sc[h] * q;
                        }
                    }
                }
                3 => {
                    for chunk in 0..32 {
                        let bo = chunk * 3;
                        let b0 = payload[bo] as u32;
                        let b1 = payload[bo + 1] as u32;
                        let b2 = payload[bo + 2] as u32;
                        let pk = b0 | (b1 << 8) | (b2 << 16);
                        for j in 0..8 {
                            let idx = chunk * 8 + j;
                            let q = ((pk >> (j * 3)) & 7) as f32;
                            let h = idx / HALF;
                            w[idx] = zp[h] + sc[h] * q;
                        }
                    }
                }
                5 => {
                    for chunk in 0..32 {
                        let bo = chunk * 5;
                        let b0 = payload[bo] as u32;
                        let b1 = payload[bo + 1] as u32;
                        let b2 = payload[bo + 2] as u32;
                        let b3 = payload[bo + 3] as u32;
                        let b4 = payload[bo + 4] as u32;
                        // linear 40-bit stream: q0 at bits0-4 etc.
                        let lo = (b0 as u64)
                            | ((b1 as u64) << 8)
                            | ((b2 as u64) << 16)
                            | ((b3 as u64) << 24)
                            | ((b4 as u64) << 32);
                        for j in 0..8 {
                            let idx = chunk * 8 + j;
                            let q = ((lo >> (j * 5)) & 0x1f) as f32;
                            let h = idx / HALF;
                            w[idx] = zp[h] + sc[h] * q;
                        }
                    }
                }
                6 => {
                    for i in (0..GROUP).step_by(4) {
                        let bo = (i / 4) * 3;
                        let b0 = payload[bo] as u32;
                        let b1 = payload[bo + 1] as u32;
                        let b2 = payload[bo + 2] as u32;
                        let pk = b0 | (b1 << 8) | (b2 << 16);
                        let qs = [
                            (pk & 63) as f32,
                            ((pk >> 6) & 63) as f32,
                            ((pk >> 12) & 63) as f32,
                            ((pk >> 18) & 63) as f32,
                        ];
                        for j in 0..4 {
                            let idx = i + j;
                            let h = idx / HALF;
                            w[idx] = zp[h] + sc[h] * qs[j];
                        }
                    }
                }
                _ => unreachable!(),
            }
            for i in 0..GROUP {
                acc += w[i] as f64 * x[g * GROUP + i] as f64;
            }
        }
        y[r] = acc;
    }
    y
}

fn decode_and_gemm_batched(
    fmt: &Format,
    blob: &[u8],
    x: &[f32],
    m: usize,
    k: usize,
    batch: usize,
) -> Vec<f64> {
    let gpr = k / GROUP;
    let mut y = vec![0.0f64; batch * m];
    for b in 0..batch {
        let xb = &x[b * k..(b + 1) * k];
        for r in 0..m {
            let mut acc = 0.0f64;
            for g in 0..gpr {
                let base = (r * gpr + g) * fmt.group_bytes;
                let s0 = f16::from_bits(u16::from_le_bytes([blob[base], blob[base + 1]])).to_f32();
                let z0 =
                    f16::from_bits(u16::from_le_bytes([blob[base + 2], blob[base + 3]])).to_f32();
                let s1 =
                    f16::from_bits(u16::from_le_bytes([blob[base + 4], blob[base + 5]])).to_f32();
                let z1 =
                    f16::from_bits(u16::from_le_bytes([blob[base + 6], blob[base + 7]])).to_f32();
                let sc = [s0, s1];
                let zp = [z0, z1];
                let payload = &blob[base + 8..base + fmt.group_bytes];
                let mut w = [0.0f32; GROUP];
                match fmt.bits {
                    2 => {
                        for i in 0..64 {
                            let by = payload[i];
                            for j in 0..4 {
                                let idx = i * 4 + j;
                                let q = ((by >> (j * 2)) & 3) as f32;
                                let h = idx / HALF;
                                w[idx] = zp[h] + sc[h] * q;
                            }
                        }
                    }
                    3 => {
                        for chunk in 0..32 {
                            let bo = chunk * 3;
                            let b0 = payload[bo] as u32;
                            let b1 = payload[bo + 1] as u32;
                            let b2 = payload[bo + 2] as u32;
                            let pk = b0 | (b1 << 8) | (b2 << 16);
                            for j in 0..8 {
                                let idx = chunk * 8 + j;
                                let q = ((pk >> (j * 3)) & 7) as f32;
                                let h = idx / HALF;
                                w[idx] = zp[h] + sc[h] * q;
                            }
                        }
                    }
                    5 => {
                        for chunk in 0..32 {
                            let bo = chunk * 5;
                            let b0 = payload[bo] as u64;
                            let b1 = payload[bo + 1] as u64;
                            let b2 = payload[bo + 2] as u64;
                            let b3 = payload[bo + 3] as u64;
                            let b4 = payload[bo + 4] as u64;
                            let lo = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32);
                            for j in 0..8 {
                                let idx = chunk * 8 + j;
                                let q = ((lo >> (j * 5)) & 0x1f) as f32;
                                let h = idx / HALF;
                                w[idx] = zp[h] + sc[h] * q;
                            }
                        }
                    }
                    6 => {
                        for i in (0..GROUP).step_by(4) {
                            let bo = (i / 4) * 3;
                            let b0 = payload[bo] as u32;
                            let b1 = payload[bo + 1] as u32;
                            let b2 = payload[bo + 2] as u32;
                            let pk = b0 | (b1 << 8) | (b2 << 16);
                            let qs = [
                                (pk & 63) as f32,
                                ((pk >> 6) & 63) as f32,
                                ((pk >> 12) & 63) as f32,
                                ((pk >> 18) & 63) as f32,
                            ];
                            for j in 0..4 {
                                let idx = i + j;
                                let h = idx / HALF;
                                w[idx] = zp[h] + sc[h] * qs[j];
                            }
                        }
                    }
                    _ => unreachable!(),
                }
                for i in 0..GROUP {
                    acc += w[i] as f64 * xb[g * GROUP + i] as f64;
                }
            }
            y[b * m + r] = acc;
        }
    }
    y
}

fn rel_rms(got: &[f32], want: &[f64]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&g, &w) in got.iter().zip(want.iter()) {
        let d = g as f64 - w;
        num += d * d;
        den += w * w;
    }
    (num / den.max(1e-30)).sqrt()
}

fn worst_errors(got: &[f32], want: &[f64]) -> (f64, f64, usize, usize) {
    let mut worst_abs: f64 = 0.0;
    let mut worst_rel: f64 = 0.0;
    let mut worst_abs_i = 0usize;
    let mut worst_rel_i = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let abs = ((g as f64) - w).abs();
        if abs > worst_abs {
            worst_abs = abs;
            worst_abs_i = i;
        }
        let denom = w.abs().max(1e-6);
        let rel = abs / denom;
        if rel > worst_rel {
            worst_rel = rel;
            worst_rel_i = i;
        }
    }
    (worst_abs, worst_rel, worst_abs_i, worst_rel_i)
}

fn ref_gemv_single_header_f64(
    fmt: &Format,
    blob: &[u8],
    x: &[f32],
    m: usize,
    k: usize,
) -> Vec<f64> {
    // Simulates kernel that reads only half0 header for all weights.
    let gpr = k / GROUP;
    let mut y = vec![0.0f64; m];
    for r in 0..m {
        let mut acc = 0.0f64;
        for g in 0..gpr {
            let base = (r * gpr + g) * fmt.group_bytes;
            let s0 = f16::from_bits(u16::from_le_bytes([blob[base], blob[base + 1]])).to_f32();
            let z0 = f16::from_bits(u16::from_le_bytes([blob[base + 2], blob[base + 3]])).to_f32();
            let payload = &blob[base + 8..base + fmt.group_bytes];
            // decode using s0/z0 for all indices
            let mut qvals = Vec::with_capacity(GROUP);
            match fmt.bits {
                2 => {
                    for i in 0..64 {
                        let b = payload[i];
                        for j in 0..4 {
                            qvals.push(((b >> (j * 2)) & 3) as f32);
                        }
                    }
                }
                3 => {
                    for chunk in 0..32 {
                        let bo = chunk * 3;
                        let pk = (payload[bo] as u32)
                            | ((payload[bo + 1] as u32) << 8)
                            | ((payload[bo + 2] as u32) << 16);
                        for j in 0..8 {
                            qvals.push(((pk >> (j * 3)) & 7) as f32);
                        }
                    }
                }
                5 => {
                    for chunk in 0..32 {
                        let bo = chunk * 5;
                        let lo = (payload[bo] as u64)
                            | ((payload[bo + 1] as u64) << 8)
                            | ((payload[bo + 2] as u64) << 16)
                            | ((payload[bo + 3] as u64) << 24)
                            | ((payload[bo + 4] as u64) << 32);
                        for j in 0..8 {
                            qvals.push(((lo >> (j * 5)) & 0x1f) as f32);
                        }
                    }
                }
                6 => {
                    for i in (0..GROUP).step_by(4) {
                        let bo = (i / 4) * 3;
                        let pk = (payload[bo] as u32)
                            | ((payload[bo + 1] as u32) << 8)
                            | ((payload[bo + 2] as u32) << 16);
                        qvals.push((pk & 63) as f32);
                        qvals.push(((pk >> 6) & 63) as f32);
                        qvals.push(((pk >> 12) & 63) as f32);
                        qvals.push(((pk >> 18) & 63) as f32);
                    }
                }
                _ => unreachable!(),
            }
            for i in 0..GROUP {
                let w = z0 + s0 * qvals[i];
                acc += w as f64 * x[g * GROUP + i] as f64;
            }
        }
        y[r] = acc;
    }
    y
}

fn headers_differ(blob: &[u8], fmt: &Format, m: usize, k: usize) -> bool {
    let gpr = k / GROUP;
    for r in 0..m {
        for g in 0..gpr {
            let base = (r * gpr + g) * fmt.group_bytes;
            let s0 = u16::from_le_bytes([blob[base], blob[base + 1]]);
            let z0 = u16::from_le_bytes([blob[base + 2], blob[base + 3]]);
            let s1 = u16::from_le_bytes([blob[base + 4], blob[base + 5]]);
            let z1 = u16::from_le_bytes([blob[base + 6], blob[base + 7]]);
            if s0 != s1 || z0 != z1 {
                return true;
            }
        }
    }
    false
}

/// One fused-family check outcome for machine-parseable stdout.
struct FusedFamilyResult {
    status: &'static str,
    /// Worst (max) rel-RMS across the family's output tensors.
    rel_rms: f64,
    canary_ok: bool,
}

fn fused_fail_unsupported() -> FusedFamilyResult {
    FusedFamilyResult {
        status: "FAIL_UNSUPPORTED",
        rel_rms: f64::INFINITY,
        canary_ok: false,
    }
}

fn upload_y_canary(
    gpu: &mut Gpu,
    n: usize,
    m: usize,
) -> Result<(rdna_compute::GpuTensor, usize), String> {
    let y_len = n * m + 1;
    let mut y_host = vec![0.0f32; y_len];
    y_host[y_len - 1] = CANARY_F32;
    let d_y = gpu
        .upload_f32(&y_host, &[y_len])
        .map_err(|e| format!("upload_f32 y canary: {e:?}"))?;
    Ok((d_y, y_len))
}

fn download_y_canary(
    gpu: &Gpu,
    d_y: &rdna_compute::GpuTensor,
    y_len: usize,
    n: usize,
    m: usize,
) -> Result<(Vec<f32>, bool), String> {
    let got_all = gpu
        .download_f32(d_y)
        .map_err(|e| format!("download y: {e:?}"))?;
    if got_all.len() != y_len {
        return Err(format!("download len {} != {}", got_all.len(), y_len));
    }
    let canary_ok = (got_all[y_len - 1] - CANARY_F32).abs() < 1e-6;
    Ok((got_all[..n * m].to_vec(), canary_ok))
}

/// Judge a multi-output fused family: all outputs must pass rel-RMS and canary.
fn judge_fused_outputs(
    label: &str,
    outs: &[(&str, Vec<f32>, Vec<f64>, bool)],
) -> FusedFamilyResult {
    let mut worst_rms = 0.0f64;
    let mut all_canary = true;
    let mut worst_name = "";
    for (name, got, want, canary_ok) in outs {
        let rms = rel_rms(got, want);
        let (wabs, wrel, wi_a, wi_r) = worst_errors(got, want);
        eprintln!(
            "  FUSED {label}/{name}: rel_rms {:.3e} worst_abs {:.3e}@{wi_a} worst_rel {:.3e}@{wi_r} canary_ok={canary_ok}",
            rms, wabs, wrel
        );
        if rms > worst_rms {
            worst_rms = rms;
            worst_name = name;
        }
        all_canary &= *canary_ok;
    }
    let pass = worst_rms <= FUSED_TOL_RMS && all_canary;
    let status = if pass { "PASS" } else { "FAIL" };
    if !pass {
        eprintln!(
            "  FUSED {label} FAIL: worst_rel_rms {:.3e} ({worst_name}) tol {:.2} canary_ok={all_canary}",
            worst_rms, FUSED_TOL_RMS
        );
    } else {
        eprintln!(
            "  FUSED {label} PASS: worst_rel_rms {:.3e} <= {:.2}",
            worst_rms, FUSED_TOL_RMS
        );
    }
    FusedFamilyResult {
        status,
        rel_rms: worst_rms,
        canary_ok: all_canary,
    }
}

fn run_fused_gate_up(gpu: &mut Gpu, fmt: &Format, x: &[f32]) -> FusedFamilyResult {
    let k = K;
    let n = FUSED_N;
    let gate_m = FUSED_ROW;
    let up_m = FUSED_ROW;
    let w_gate = build_disjoint_halves_salted(gate_m, k, 0x6A7E_0001);
    let w_up = build_disjoint_halves_salted(up_m, k, 0x6A7E_0002);

    let blob_gate = pack_blob(fmt, &w_gate, gate_m, k);
    let blob_up = pack_blob(fmt, &w_up, up_m, k);
    let want_gate = decode_and_gemm_batched(fmt, &blob_gate, x, gate_m, k, n);
    let want_up = decode_and_gemm_batched(fmt, &blob_up, x, up_m, k, n);

    let run = (|| -> Result<FusedFamilyResult, String> {
        let d_a_gate = gpu
            .upload_raw(&blob_gate, &[blob_gate.len()])
            .map_err(|e| format!("upload gate: {e:?}"))?;
        let d_a_up = gpu
            .upload_raw(&blob_up, &[blob_up.len()])
            .map_err(|e| format!("upload up: {e:?}"))?;
        let d_x = gpu
            .upload_f32(x, &[n * k])
            .map_err(|e| format!("upload x: {e:?}"))?;
        let (d_y_gate, y_gate_len) = upload_y_canary(gpu, n, gate_m)?;
        let (d_y_up, y_up_len) = upload_y_canary(gpu, n, up_m)?;
        let res = match fmt.qt {
            47 => gpu.gemm_gate_up_mq6g256v2_wmma(
                &d_a_gate, &d_a_up, &d_x, &d_y_gate, &d_y_up, gate_m, up_m, k, n,
            ),
            48 => gpu.gemm_gate_up_mq5g256v2_wmma(
                &d_a_gate, &d_a_up, &d_x, &d_y_gate, &d_y_up, gate_m, up_m, k, n,
            ),
            49 => gpu.gemm_gate_up_mq3g256v2_wmma(
                &d_a_gate, &d_a_up, &d_x, &d_y_gate, &d_y_up, gate_m, up_m, k, n,
            ),
            50 => gpu.gemm_gate_up_mq2g256v2_wmma(
                &d_a_gate, &d_a_up, &d_x, &d_y_gate, &d_y_up, gate_m, up_m, k, n,
            ),
            _ => unreachable!(),
        };
        res.map_err(|e| format!("gate_up launch: {e:?}"))?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;
        let (got_gate, c_gate) = download_y_canary(gpu, &d_y_gate, y_gate_len, n, gate_m)?;
        let (got_up, c_up) = download_y_canary(gpu, &d_y_up, y_up_len, n, up_m)?;
        Ok(judge_fused_outputs(
            "gate_up",
            &[
                ("gate", got_gate, want_gate, c_gate),
                ("up", got_up, want_up, c_up),
            ],
        ))
    })();
    match run {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  FUSED gate_up launch FAIL (unsupported route): {e}");
            fused_fail_unsupported()
        }
    }
}

fn run_fused_qkv(gpu: &mut Gpu, fmt: &Format, x: &[f32]) -> FusedFamilyResult {
    let k = K;
    let n = FUSED_N;
    let q_m = FUSED_ROW;
    let k_m = FUSED_ROW;
    let v_m = FUSED_ROW;
    let w_q = build_disjoint_halves_salted(q_m, k, 0x9CB0_0001);
    let w_k = build_disjoint_halves_salted(k_m, k, 0x9CB0_0002);
    let w_v = build_disjoint_halves_salted(v_m, k, 0x9CB0_0003);

    let blob_q = pack_blob(fmt, &w_q, q_m, k);
    let blob_k = pack_blob(fmt, &w_k, k_m, k);
    let blob_v = pack_blob(fmt, &w_v, v_m, k);
    let want_q = decode_and_gemm_batched(fmt, &blob_q, x, q_m, k, n);
    let want_k = decode_and_gemm_batched(fmt, &blob_k, x, k_m, k, n);
    let want_v = decode_and_gemm_batched(fmt, &blob_v, x, v_m, k, n);

    let run = (|| -> Result<FusedFamilyResult, String> {
        let d_a_q = gpu
            .upload_raw(&blob_q, &[blob_q.len()])
            .map_err(|e| format!("upload q: {e:?}"))?;
        let d_a_k = gpu
            .upload_raw(&blob_k, &[blob_k.len()])
            .map_err(|e| format!("upload k: {e:?}"))?;
        let d_a_v = gpu
            .upload_raw(&blob_v, &[blob_v.len()])
            .map_err(|e| format!("upload v: {e:?}"))?;
        let d_x = gpu
            .upload_f32(x, &[n * k])
            .map_err(|e| format!("upload x: {e:?}"))?;
        let (d_y_q, y_q_len) = upload_y_canary(gpu, n, q_m)?;
        let (d_y_k, y_k_len) = upload_y_canary(gpu, n, k_m)?;
        let (d_y_v, y_v_len) = upload_y_canary(gpu, n, v_m)?;
        let res = match fmt.qt {
            47 => gpu.gemm_qkv_mq6g256v2_wmma(
                &d_a_q, &d_a_k, &d_a_v, &d_x, &d_y_q, &d_y_k, &d_y_v, q_m, k_m, v_m, k, n,
            ),
            48 => gpu.gemm_qkv_mq5g256v2_wmma(
                &d_a_q, &d_a_k, &d_a_v, &d_x, &d_y_q, &d_y_k, &d_y_v, q_m, k_m, v_m, k, n,
            ),
            49 => gpu.gemm_qkv_mq3g256v2_wmma(
                &d_a_q, &d_a_k, &d_a_v, &d_x, &d_y_q, &d_y_k, &d_y_v, q_m, k_m, v_m, k, n,
            ),
            50 => gpu.gemm_qkv_mq2g256v2_wmma(
                &d_a_q, &d_a_k, &d_a_v, &d_x, &d_y_q, &d_y_k, &d_y_v, q_m, k_m, v_m, k, n,
            ),
            _ => unreachable!(),
        };
        res.map_err(|e| format!("qkv launch: {e:?}"))?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;
        let (got_q, c_q) = download_y_canary(gpu, &d_y_q, y_q_len, n, q_m)?;
        let (got_k, c_k) = download_y_canary(gpu, &d_y_k, y_k_len, n, k_m)?;
        let (got_v, c_v) = download_y_canary(gpu, &d_y_v, y_v_len, n, v_m)?;
        Ok(judge_fused_outputs(
            "qkv",
            &[
                ("q", got_q, want_q, c_q),
                ("k", got_k, want_k, c_k),
                ("v", got_v, want_v, c_v),
            ],
        ))
    })();
    match run {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  FUSED qkv launch FAIL (unsupported route): {e}");
            fused_fail_unsupported()
        }
    }
}

fn run_fused_qkvza(gpu: &mut Gpu, fmt: &Format, x: &[f32]) -> FusedFamilyResult {
    let k = K;
    let n = FUSED_N;
    let qkv_m = FUSED_ROW;
    let z_m = FUSED_ROW;
    let beta_m = FUSED_ROW;
    let alpha_m = FUSED_ROW;
    let w_qkv = build_disjoint_halves_salted(qkv_m, k, 0x2A1A_0001);
    let w_z = build_disjoint_halves_salted(z_m, k, 0x2A1A_0002);
    let w_beta = build_disjoint_halves_salted(beta_m, k, 0x2A1A_0003);
    let w_alpha = build_disjoint_halves_salted(alpha_m, k, 0x2A1A_0004);

    let blob_qkv = pack_blob(fmt, &w_qkv, qkv_m, k);
    let blob_z = pack_blob(fmt, &w_z, z_m, k);
    let blob_beta = pack_blob(fmt, &w_beta, beta_m, k);
    let blob_alpha = pack_blob(fmt, &w_alpha, alpha_m, k);
    let want_qkv = decode_and_gemm_batched(fmt, &blob_qkv, x, qkv_m, k, n);
    let want_z = decode_and_gemm_batched(fmt, &blob_z, x, z_m, k, n);
    let want_beta = decode_and_gemm_batched(fmt, &blob_beta, x, beta_m, k, n);
    let want_alpha = decode_and_gemm_batched(fmt, &blob_alpha, x, alpha_m, k, n);

    let run = (|| -> Result<FusedFamilyResult, String> {
        let d_a_qkv = gpu
            .upload_raw(&blob_qkv, &[blob_qkv.len()])
            .map_err(|e| format!("upload qkv: {e:?}"))?;
        let d_a_z = gpu
            .upload_raw(&blob_z, &[blob_z.len()])
            .map_err(|e| format!("upload z: {e:?}"))?;
        let d_a_beta = gpu
            .upload_raw(&blob_beta, &[blob_beta.len()])
            .map_err(|e| format!("upload beta: {e:?}"))?;
        let d_a_alpha = gpu
            .upload_raw(&blob_alpha, &[blob_alpha.len()])
            .map_err(|e| format!("upload alpha: {e:?}"))?;
        let d_x = gpu
            .upload_f32(x, &[n * k])
            .map_err(|e| format!("upload x: {e:?}"))?;
        let (d_y_qkv, y_qkv_len) = upload_y_canary(gpu, n, qkv_m)?;
        let (d_y_z, y_z_len) = upload_y_canary(gpu, n, z_m)?;
        let (d_y_beta, y_beta_len) = upload_y_canary(gpu, n, beta_m)?;
        let (d_y_alpha, y_alpha_len) = upload_y_canary(gpu, n, alpha_m)?;
        let res = match fmt.qt {
            47 => gpu.gemm_qkvza_mq6g256v2_wmma(
                &d_a_qkv, &d_a_z, &d_a_beta, &d_a_alpha, &d_x, &d_y_qkv, &d_y_z, &d_y_beta,
                &d_y_alpha, qkv_m, z_m, beta_m, alpha_m, k, n,
            ),
            48 => gpu.gemm_qkvza_mq5g256v2_wmma(
                &d_a_qkv, &d_a_z, &d_a_beta, &d_a_alpha, &d_x, &d_y_qkv, &d_y_z, &d_y_beta,
                &d_y_alpha, qkv_m, z_m, beta_m, alpha_m, k, n,
            ),
            49 => gpu.gemm_qkvza_mq3g256v2_wmma(
                &d_a_qkv, &d_a_z, &d_a_beta, &d_a_alpha, &d_x, &d_y_qkv, &d_y_z, &d_y_beta,
                &d_y_alpha, qkv_m, z_m, beta_m, alpha_m, k, n,
            ),
            50 => gpu.gemm_qkvza_mq2g256v2_wmma(
                &d_a_qkv, &d_a_z, &d_a_beta, &d_a_alpha, &d_x, &d_y_qkv, &d_y_z, &d_y_beta,
                &d_y_alpha, qkv_m, z_m, beta_m, alpha_m, k, n,
            ),
            _ => unreachable!(),
        };
        res.map_err(|e| format!("qkvza launch: {e:?}"))?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;
        let (got_qkv, c_qkv) = download_y_canary(gpu, &d_y_qkv, y_qkv_len, n, qkv_m)?;
        let (got_z, c_z) = download_y_canary(gpu, &d_y_z, y_z_len, n, z_m)?;
        let (got_beta, c_beta) = download_y_canary(gpu, &d_y_beta, y_beta_len, n, beta_m)?;
        let (got_alpha, c_alpha) = download_y_canary(gpu, &d_y_alpha, y_alpha_len, n, alpha_m)?;
        Ok(judge_fused_outputs(
            "qkvza",
            &[
                ("qkv", got_qkv, want_qkv, c_qkv),
                ("z", got_z, want_z, c_z),
                ("beta", got_beta, want_beta, c_beta),
                ("alpha", got_alpha, want_alpha, c_alpha),
            ],
        ))
    })();
    match run {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  FUSED qkvza launch FAIL (unsupported route): {e}");
            fused_fail_unsupported()
        }
    }
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("mqv2_family_parity: no GPU ({e}) — skipping");
            return;
        }
    };
    eprintln!(
        "mqv2_family_parity: arch={} has_wmma_w32={} has_wmma_w32_gfx12={} m={} k={} gpr={} batch={} fused_n={}",
        gpu.arch,
        gpu.arch_caps.has_wmma_w32(),
        gpu.arch_caps.has_wmma_w32_gfx12(),
        M,
        K,
        GPR_K512,
        BATCH,
        FUSED_N
    );
    if !(gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12()) {
        eprintln!(
            "warn: expected gfx11/gfx12 WMMA for fused prefill + batched lm-head, got {}",
            gpu.arch
        );
    }

    let w = build_disjoint_halves(M, K);
    let x_gemv = build_x(K, 0xC0FF_EE00);
    let x_batch = build_x_batched(K, BATCH, 0xBEEF_CAFE);
    let x_fused = build_x_batched(K, FUSED_N, 0xF05E_D000);

    // group bytes set to ensure no alias
    {
        let mut seen = std::collections::HashSet::new();
        for f in &FORMATS {
            assert!(
                seen.insert(f.group_bytes),
                "format {} aliases group_bytes {}",
                f.name,
                f.group_bytes
            );
        }
    }

    let mut overall_fail = false;

    for fmt in &FORMATS {
        eprintln!(
            "\n--- fmt={} qt={} group_bytes={} bits={} dtype={:?}",
            fmt.name, fmt.qt, fmt.group_bytes, fmt.bits, fmt.dtype
        );
        let blob = pack_blob(fmt, &w, M, K);
        let gpr = K / GROUP;
        let expected_bytes = M * gpr * fmt.group_bytes;
        let byte_ok = blob.len() == expected_bytes;
        eprintln!(
            "  byte_count: got={} expected={} ok={}",
            blob.len(),
            expected_bytes,
            byte_ok
        );
        if !byte_ok {
            eprintln!("  FAIL byte_count mismatch");
            overall_fail = true;
            println!(
                "fmt={} qt={} group_bytes={} m={} k={} gpr={} byte_ok={} canary_ok=false status=FAIL worst_abs=0 worst_rel=0 bug_rel=0",
                fmt.name, fmt.qt, fmt.group_bytes, M, K, gpr, byte_ok
            );
            continue;
        }
        // header divergence check
        let hdr_ok = headers_differ(&blob, fmt, M, K);
        eprintln!("  header_divergence s0/z0 != s1/z1 : {}", hdr_ok);
        if !hdr_ok {
            eprintln!("  FAIL headers do not differ — fixture not discriminating");
            overall_fail = true;
        }

        // want for gemv
        let want_gemv = decode_and_gemv(fmt, &blob, &x_gemv, M, K);
        let bug_gemv = ref_gemv_single_header_f64(fmt, &blob, &x_gemv, M, K);
        let bug_rel = {
            let bug_f32: Vec<f32> = bug_gemv.iter().map(|&v| v as f32).collect();
            let (_, r, _, _) = worst_errors(&bug_f32, &want_gemv);
            r
        };
        eprintln!(
            "  bug_baseline_rel (single-half vs correct): {:.3e}",
            bug_rel
        );
        if bug_rel < 1.0 {
            eprintln!("  WARN fixture separation <1.0 — may not discriminate half-select bug");
        }

        // ── B=1 plain GEMV (never residual) ─────────────────────────────────
        let gemv_result = (|| -> Result<(Vec<f32>, bool), String> {
            // allocate y with canary
            let y_len = M + 1;
            let mut y_host = vec![0.0f32; y_len];
            y_host[y_len - 1] = CANARY_F32;
            let d_a = gpu
                .upload_raw(&blob, &[blob.len()])
                .map_err(|e| format!("upload_raw: {e:?}"))?;
            let d_x = gpu
                .upload_f32(&x_gemv, &[K])
                .map_err(|e| format!("upload_f32 x: {e:?}"))?;
            // alloc y with extra canary
            let d_y = gpu
                .upload_f32(&y_host, &[y_len])
                .map_err(|e| format!("upload_f32 y: {e:?}"))?;
            // dispatch per format
            // Plain GEMV only — never substitute residual for the B=1 check.
            let res: Result<(), _> = match fmt.qt {
                47 => gpu.gemv_mq6g256v2(&d_a, &d_x, &d_y, M, K),
                48 => gpu.gemv_mq5g256v2(&d_a, &d_x, &d_y, M, K),
                49 => gpu.gemv_mq3g256v2(&d_a, &d_x, &d_y, M, K),
                50 => gpu.gemv_mq2g256v2(&d_a, &d_x, &d_y, M, K),
                _ => unreachable!(),
            };
            if let Err(e) = res {
                return Err(format!("gemv launch: {e:?}"));
            }
            gpu.hip
                .device_synchronize()
                .map_err(|e| format!("sync: {e:?}"))?;
            let got_all = gpu
                .download_f32(&d_y)
                .map_err(|e| format!("download: {e:?}"))?;
            if got_all.len() != y_len {
                return Err(format!("download len {} != {}", got_all.len(), y_len));
            }
            let canary_ok = (got_all[y_len - 1] - CANARY_F32).abs() < 1e-6;
            if !canary_ok {
                eprintln!(
                    "  GEMV canary FAIL: got {} want {}",
                    got_all[y_len - 1],
                    CANARY_F32
                );
            }
            let got = got_all[..M].to_vec();
            Ok((got, canary_ok))
        })();

        let (gemv_got, gemv_canary_ok, gemv_worst_abs, gemv_worst_rel, gemv_status) =
            match gemv_result {
                Ok((got, canary_ok)) => {
                    let (worst_abs, worst_rel, worst_abs_i, worst_rel_i) =
                        worst_errors(&got, &want_gemv);
                    eprintln!(
                    "  GEMV plain B=1: worst_abs {:.3e} at {} (got {:.6}, want {:.6}) worst_rel {:.3e} at {} canary_ok={}",
                    worst_abs, worst_abs_i, got[worst_abs_i], want_gemv[worst_abs_i], worst_rel, worst_rel_i, canary_ok
                );
                    let tol_rel: f64 = if fmt.bits <= 3 { 2e-3 } else { 1e-4 };
                    let tol_abs: f64 = 1e-2;
                    // WMMA fp16 may add small error, but GEMV is scalar — tighten
                    let pass =
                        worst_rel < tol_rel.max(tol_abs) && worst_rel < bug_rel * 0.01 && canary_ok;
                    let status = if pass { "PASS" } else { "FAIL" };
                    if !pass {
                        eprintln!(
                            "  GEMV FAIL: tol_rel {:.0e} bug_rel {:.3e}",
                            tol_rel, bug_rel
                        );
                    }
                    (Some(got), canary_ok, worst_abs, worst_rel, status)
                }
                Err(e) => {
                    eprintln!("  GEMV plain B=1 launch FAIL (unsupported route): {e}");
                    (
                        None,
                        false,
                        f64::INFINITY,
                        f64::INFINITY,
                        "FAIL_UNSUPPORTED",
                    )
                }
            };

        // ── B=16 batched lm-head GEMM (gfx11+gfx12 WMMA) ─────────────────────

        let want_batch = decode_and_gemm_batched(fmt, &blob, &x_batch, M, K, BATCH);
        let batch_result = (|| -> Result<(Vec<f32>, bool), String> {
            let y_len = BATCH * M + 1;
            let mut y_host = vec![0.0f32; y_len];
            y_host[y_len - 1] = CANARY_F32;
            let d_a = gpu
                .upload_raw(&blob, &[blob.len()])
                .map_err(|e| format!("upload_raw: {e:?}"))?;
            let d_x = gpu
                .upload_f32(&x_batch, &[BATCH * K])
                .map_err(|e| format!("upload_f32 x_batch: {e:?}"))?;
            let d_y = gpu
                .upload_f32(&y_host, &[y_len])
                .map_err(|e| format!("upload_f32 y: {e:?}"))?;
            let res: Result<(), _> = match fmt.qt {
                47 => gpu.gemm_mq6g256v2_batched_lmhead(&d_a, &d_x, &d_y, M, K, BATCH),
                48 => gpu.gemm_mq5g256v2_batched_lmhead(&d_a, &d_x, &d_y, M, K, BATCH),
                49 => gpu.gemm_mq3g256v2_batched_lmhead(&d_a, &d_x, &d_y, M, K, BATCH),
                50 => gpu.gemm_mq2g256v2_batched_lmhead(&d_a, &d_x, &d_y, M, K, BATCH),
                _ => unreachable!(),
            };
            if let Err(e) = res {
                return Err(format!("batched lmhead launch: {e:?}"));
            }
            gpu.hip
                .device_synchronize()
                .map_err(|e| format!("sync: {e:?}"))?;
            let got_all = gpu
                .download_f32(&d_y)
                .map_err(|e| format!("download: {e:?}"))?;
            if got_all.len() != y_len {
                return Err(format!("download len {} != {}", got_all.len(), y_len));
            }
            let canary_ok = (got_all[y_len - 1] - CANARY_F32).abs() < 1e-6;
            if !canary_ok {
                eprintln!(
                    "  BATCH canary FAIL: got {} want {}",
                    got_all[y_len - 1],
                    CANARY_F32
                );
            }
            let got = got_all[..BATCH * M].to_vec();
            Ok((got, canary_ok))
        })();

        let (batch_worst_abs, batch_worst_rel, batch_rel_rms, batch_canary_ok, batch_status) =
            match batch_result {
                Ok((got, canary_ok)) => {
                    let (worst_abs, worst_rel, worst_abs_i, worst_rel_i) =
                        worst_errors(&got, &want_batch);
                    let rms = rel_rms(&got, &want_batch);
                    eprintln!(
                    "  BATCH B=16: worst_abs {:.3e} at {} (got {:.6}, want {:.6}) worst_rel {:.3e} at {} rel_rms {:.3e} canary_ok={}",
                    worst_abs,
                    worst_abs_i,
                    got[worst_abs_i],
                    want_batch[worst_abs_i],
                    worst_rel,
                    worst_rel_i,
                    rms,
                    canary_ok
                );
                    // gfx12 WMMA batched lm-head does fp16 x conversion; judge by rel-RMS as in mq4v2_gemm_parity.rs (181-189,235-249)
                    // 0.05 is the defensible WMMA fp16 floor — worst_abs/rel retained as diagnostics
                    let tol_rms: f64 = 0.05;
                    let pass = rms <= tol_rms && canary_ok;
                    let status = if pass { "PASS" } else { "FAIL" };
                    if !pass {
                        eprintln!("  BATCH FAIL: rel_rms {:.3e} > {:.0e}", rms, tol_rms);
                    }
                    (worst_abs, worst_rel, rms, canary_ok, status)
                }
                Err(e) => {
                    eprintln!("  BATCH B=16 launch FAIL (unsupported route): {e}");
                    (
                        f64::INFINITY,
                        f64::INFINITY,
                        f64::INFINITY,
                        false,
                        "FAIL_UNSUPPORTED",
                    )
                }
            };

        // ── Fused prefill WMMA at K=512 N=16 (arch-aware gemm_*_mqNg256v2_wmma) ──
        let fused_gate_up = run_fused_gate_up(&mut gpu, fmt, &x_fused);
        let fused_qkv = run_fused_qkv(&mut gpu, fmt, &x_fused);
        let fused_qkvza = run_fused_qkvza(&mut gpu, fmt, &x_fused);

        let fmt_status = if gemv_status == "PASS"
            && batch_status == "PASS"
            && fused_gate_up.status == "PASS"
            && fused_qkv.status == "PASS"
            && fused_qkvza.status == "PASS"
            && hdr_ok
            && byte_ok
        {
            "PASS"
        } else {
            "FAIL"
        };
        if fmt_status == "FAIL" {
            overall_fail = true;
        }

        // machine-parseable per-format line (stdout for evidence)
        println!(
            "fmt={} qt={} group_bytes={} m={} k={} gpr={} fused_n={} byte_ok={} hdr_ok={} canary_gemv={} canary_batch={} canary_gate_up={} canary_qkv={} canary_qkvza={} gemv_status={} batch_status={} gate_up_status={} qkv_status={} qkvza_status={} status={} gemv_worst_abs={:.6e} gemv_worst_rel={:.6e} batch_worst_abs={:.6e} batch_worst_rel={:.6e} batch_rel_rms={:.6e} gate_up_rel_rms={:.6e} qkv_rel_rms={:.6e} qkvza_rel_rms={:.6e} bug_rel={:.6e}",
            fmt.name,
            fmt.qt,
            fmt.group_bytes,
            M,
            K,
            gpr,
            FUSED_N,
            byte_ok,
            hdr_ok,
            gemv_canary_ok,
            batch_canary_ok,
            fused_gate_up.canary_ok,
            fused_qkv.canary_ok,
            fused_qkvza.canary_ok,
            gemv_status,
            batch_status,
            fused_gate_up.status,
            fused_qkv.status,
            fused_qkvza.status,
            fmt_status,
            gemv_worst_abs,
            gemv_worst_rel,
            batch_worst_abs,
            batch_worst_rel,
            batch_rel_rms,
            fused_gate_up.rel_rms,
            fused_qkv.rel_rms,
            fused_qkvza.rel_rms,
            bug_rel
        );
        eprintln!(
            "  => {} : gemv rel {:.3e} batch rel_rms {:.3e} gate_up {:.3e} qkv {:.3e} qkvza {:.3e} {}",
            fmt.name,
            gemv_worst_rel,
            batch_rel_rms,
            fused_gate_up.rel_rms,
            fused_qkv.rel_rms,
            fused_qkvza.rel_rms,
            fmt_status
        );
        let _ = gemv_got;
    }

    if overall_fail {
        eprintln!("\nmqv2_family_parity: FAIL — one or more formats mismatched or unsupported");
        std::process::exit(1);
    } else {
        eprintln!("\nmqv2_family_parity: PASS — all 4 V2 formats: byte count, header divergence, sentinel, plain GEMV B=1, B=16 batched lm-head, and fused prefill WMMA (qkv/qkvza/gate_up @ K=512 N=16) match CPU dequant");
    }
}
