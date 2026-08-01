// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — RWQ4G256 correctness gate.
//
// Checks the shipped `gemv_rwq4g256_prerotated` kernel against an independent
// CPU model written straight from the frozen wire contract. The CPU side shares
// no code with the kernel or with the hipfire-quantize encoder on purpose: if
// two independent implementations agree, the layout is genuinely pinned down
// rather than two copies of the same mistake.
//
// Frozen contract — 136 B per 256-weight group:
//   [0..4)   f32 LE master scale
//   [4..8)   4 x u8 E4M3 sub-scale, one per 64 weights
//   [8..136) 128 B nibbles, 2 weights/byte, LOW nibble = even (first) weight
//   w[i] = CODEBOOK_4BIT[nibble[i]] * master * e4m3_decode(sub[i / 64])
//
// Run: cargo run --release -p rdna-compute --example rwq4_correctness

use rdna_compute::{DType, Gpu};

/// Lloyd-Max optimum for a unit Gaussian, normalized so `max|c| == 1`.
/// Transcribed from `crates/hipfire-quantize/src/mqn.rs::CODEBOOK_4BIT`.
const CODEBOOK: [f32; 16] = [
    -1.000_000_0,
    -0.757_163_55,
    -0.592_129_31,
    -0.459_721_87,
    -0.344_852_54,
    -0.240_343_13,
    -0.142_007_53,
    -0.046_986_58,
    0.046_986_58,
    0.142_007_53,
    0.240_343_13,
    0.344_852_54,
    0.459_721_87,
    0.592_129_31,
    0.757_163_55,
    1.000_000_0,
];

const GROUP: usize = 256;
const SUB: usize = 64;
const GROUP_BYTES: usize = 136;

/// Independent OCP e4m3fn decode, written from the format definition rather
/// than copied from `fp8.rs`. Sub-scales are UNSIGNED (bit 7 is never set by
/// the encoder), and code 0x7F is the NaN slot, which the codec clamps to the
/// max finite 448.0 — both mirrored here so the gate's edge semantics match
/// `turbo_common.h::cvt_e4m3_scale_to_f32_dq` and `fp8::e4m3_decode` exactly.
/// The independence that matters is the byte layout, nibble order, sub-scale
/// mapping and codebook indexing — not a second reading of the FP8 spec.
fn e4m3_decode(b: u8) -> f32 {
    let exp = ((b >> 3) & 0xF) as i32;
    let man = (b & 0x7) as f32;
    if exp == 0 {
        // subnormal: 2^-6 * (m / 8)
        return man * 2.0f32.powi(-9);
    }
    if exp == 0xF && man == 7.0 {
        return 448.0;
    }
    (1.0 + man / 8.0) * 2.0f32.powi(exp - 7)
}

/// Smallest E4M3 code whose decoded value is `>= v`, so a sub-scale never
/// clips the group it normalizes. Linear scan: correctness harness, not hot.
/// Stops at 126 because 0x7F is the NaN slot.
fn e4m3_encode_roundup(v: f32) -> u8 {
    if !(v > 0.0) {
        return 0;
    }
    for c in 1u8..=126 {
        if e4m3_decode(c) >= v {
            return c;
        }
    }
    126 // largest finite positive
}

/// Encode one 256-weight group into the frozen 136-byte layout.
fn encode_group(w: &[f32], out: &mut [u8]) {
    let mut sub_amax = [0.0f32; GROUP / SUB];
    for (i, &v) in w.iter().enumerate() {
        let a = v.abs();
        if a > sub_amax[i / SUB] {
            sub_amax[i / SUB] = a;
        }
    }
    let mut master = sub_amax.iter().copied().fold(0.0f32, f32::max);
    if master == 0.0 {
        master = 1.0;
    }

    out[0..4].copy_from_slice(&master.to_le_bytes());
    let mut sub_val = [0.0f32; GROUP / SUB];
    for s in 0..GROUP / SUB {
        let code = e4m3_encode_roundup(sub_amax[s] / master);
        out[4 + s] = code;
        sub_val[s] = e4m3_decode(code);
    }

    for (i, &v) in w.iter().enumerate() {
        let eff = master * sub_val[i / SUB];
        let t = if eff > 0.0 { v / eff } else { 0.0 };
        let mut best = 0usize;
        let mut best_d = (t - CODEBOOK[0]).abs();
        for (c, &cv) in CODEBOOK.iter().enumerate().skip(1) {
            let d = (t - cv).abs();
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        let byte = &mut out[8 + i / 2];
        if i % 2 == 0 {
            *byte = best as u8 & 0xF;
        } else {
            *byte = (*byte & 0x0F) | ((best as u8 & 0xF) << 4);
        }
    }
}

/// Independent dequant straight from the byte contract.
fn decode_group(g: &[u8], w: &mut [f32]) {
    let master = f32::from_le_bytes([g[0], g[1], g[2], g[3]]);
    for (i, dst) in w.iter_mut().enumerate() {
        let eff = master * e4m3_decode(g[4 + i / SUB]);
        let byte = g[8 + i / 2];
        let nib = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
        *dst = CODEBOOK[nib as usize] * eff;
    }
}

/// Deterministic Gaussian stream (Box-Muller over a xorshift64* core) so the
/// gate reproduces byte-for-byte across machines without a rand dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn unit(&mut self) -> f64 {
        // (0, 1] — Box-Muller must never see an exact zero.
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    fn gauss(&mut self) -> f32 {
        let u1 = self.unit();
        let u2 = self.unit();
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("=== RWQ4G256 correctness gate ===");
    eprintln!("  arch={}", gpu.arch);
    eprintln!("  kernel vs independent CPU model of the 136 B/group contract");
    eprintln!();

    // Shapes from the real Qwen3.5-0.8B per-token GEMV stack.
    let shapes: &[(usize, usize, &str)] = &[
        (1024, 1024, "o_proj      M=1024 K=1024"),
        (3072, 1024, "fused_qkv   M=3072 K=1024"),
        (6144, 1024, "gate_up     M=6144 K=1024"),
        (1024, 3072, "down_proj   M=1024 K=3072"),
    ];

    let mut rng = Rng(0x5257_5134_9E37_79B9); // 'RWQ4' | golden ratio
    let mut failures = 0usize;

    println!("{:<26} {:>13} {:>13} {:>9}", "shape", "max|abs err|", "rel err", "SNR dB");
    println!("{}", "-".repeat(65));

    for &(m, k, name) in shapes {
        let groups = k / GROUP;
        let row_bytes = groups * GROUP_BYTES;

        let w: Vec<f32> = (0..m * k).map(|_| rng.gauss()).collect();
        let mut packed = vec![0u8; m * row_bytes];
        for r in 0..m {
            for g in 0..groups {
                let src = &w[r * k + g * GROUP..r * k + (g + 1) * GROUP];
                let dst_off = r * row_bytes + g * GROUP_BYTES;
                encode_group(src, &mut packed[dst_off..dst_off + GROUP_BYTES]);
            }
        }

        // x is FWHT-rotated upstream; the kernel treats it as opaque, so a plain
        // Gaussian vector exercises the identical code path.
        let x_host: Vec<f32> = (0..k).map(|_| rng.gauss()).collect();

        // Host reference: dequant through the contract, then dot in f64.
        let mut reference = vec![0.0f32; m];
        let mut dq = vec![0.0f32; GROUP];
        for (r, dst) in reference.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for g in 0..groups {
                let off = r * row_bytes + g * GROUP_BYTES;
                decode_group(&packed[off..off + GROUP_BYTES], &mut dq);
                for (i, &d) in dq.iter().enumerate() {
                    acc += d as f64 * x_host[g * GROUP + i] as f64;
                }
            }
            *dst = acc as f32;
        }

        let weights = gpu.upload_raw(&packed, &[packed.len()]).expect("upload weights");
        let x = gpu.upload_f32(&x_host, &[k]).expect("upload x");
        let y = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y");

        gpu.gemv_rwq4g256_prerotated(&weights, &x, &y, m, k)
            .expect("gemv_rwq4g256_prerotated");
        let got = gpu.download_f32(&y).expect("download y");

        let mut max_abs = 0.0f64;
        let mut sig = 0.0f64;
        let mut err = 0.0f64;
        for (g, r) in got.iter().zip(reference.iter()) {
            let d = *g as f64 - *r as f64;
            max_abs = max_abs.max(d.abs());
            sig += (*r as f64) * (*r as f64);
            err += d * d;
        }
        let rel = max_abs / (sig / m as f64).sqrt();
        let snr = 10.0 * (sig / err.max(1e-30)).log10();

        // The wave-shuffle reduction and the sequential host loop associate the
        // f32 adds differently, so exact equality is not the bar. Above ~60 dB is
        // pure reassociation noise; below it is a real bug.
        let ok = snr > 60.0;
        println!(
            "{name:<26} {max_abs:>13.3e} {rel:>13.3e} {snr:>9.1} {}",
            if ok { "ok" } else { "FAIL" }
        );
        if !ok {
            failures += 1;
        }
    }

    println!("{}", "-".repeat(65));
    if failures > 0 {
        eprintln!("FAIL: {failures} shape(s) disagree with the CPU model");
        std::process::exit(1);
    }
    println!("PASS: kernel matches the independent CPU model on every shape");
}
