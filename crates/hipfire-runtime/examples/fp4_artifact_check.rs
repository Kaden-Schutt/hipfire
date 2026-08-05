// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Decode one HFP4G32 tensor straight out of an `.hfq` and print its leading
//! values plus summary statistics.
//!
//! Why this exists: the FP4 passthrough was verified by a roundtrip unit test,
//! which only proves the repack agrees with ITSELF. It cannot catch the
//! quantizer writing the wrong tensor's bytes, or a scale/nibble pairing that
//! is consistently wrong. Comparing these numbers against the same tensor
//! dequantized from the ORIGINAL safetensors is what closes that gap.
//!
//! Usage: fp4_artifact_check <path.hfq> <tensor_name> [n_values]

use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: fp4_artifact_check <hfq> <tensor> [n]");
    let target = args.next().expect("usage: fp4_artifact_check <hfq> <tensor> [n]");
    let n_show: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);

    let hfq = HfqFile::open(Path::new(&path)).expect("open .hfq");
    let info = hfq
        .find_tensor_info(&target)
        .unwrap_or_else(|| panic!("tensor not found: {target}"))
        .clone();
    println!("{target}\n  qt={} shape={:?} bytes={}", info.quant_type, info.shape, info.data_size);
    assert_eq!(info.quant_type, 21, "expected HFP4G32 (qt 21)");

    let (_, bytes) = hfq.tensor_data(&target).expect("tensor data");

    // Rows are `16 B header + n_blocks * 17 B`; K comes from the logical shape.
    let m = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    let n_blocks = k / 32;
    let row_bytes = 16 + n_blocks * 17;
    println!("  m={m} k={k} n_blocks={n_blocks} row_bytes={row_bytes} expected={}", m * row_bytes);

    let decode_row = |r: usize| -> Vec<f32> {
        let base = r * row_bytes;
        let row_scale = half_to_f32(u16::from_le_bytes([bytes[base], bytes[base + 1]]));
        let mut out = Vec::with_capacity(k);
        for b in 0..n_blocks {
            let bp = base + 16 + b * 17;
            let bscale = 2f32.powi(bytes[bp] as i32 - 127);
            for i in 0..16 {
                let byte = bytes[bp + 1 + i];
                out.push(row_scale * bscale * E2M1[(byte & 0x0f) as usize]);
                out.push(row_scale * bscale * E2M1[(byte >> 4) as usize]);
            }
        }
        out
    };

    let row0 = decode_row(0);
    println!("  row_scale_a(row0)={}", half_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])));
    print!("  first {n_show} of row 0:");
    for v in row0.iter().take(n_show) {
        print!(" {v:.5}");
    }
    println!();

    // Whole-tensor stats — a wrong scale/nibble pairing usually shows up as an
    // implausible magnitude or an all-zero row long before it shows up as text.
    let (mut mn, mut mx, mut sum, mut sumsq, mut zeros, mut n) = (f32::MAX, f32::MIN, 0f64, 0f64, 0usize, 0usize);
    for r in 0..m.min(64) {
        for v in decode_row(r) {
            mn = mn.min(v);
            mx = mx.max(v);
            sum += v as f64;
            sumsq += (v as f64) * (v as f64);
            if v == 0.0 {
                zeros += 1;
            }
            n += 1;
        }
    }
    let mean = sum / n as f64;
    let rms = (sumsq / n as f64).sqrt();
    println!("  over first {} rows: min={mn:.5} max={mx:.5} mean={mean:.6} rms={rms:.6} zeros={:.1}%",
             m.min(64), 100.0 * zeros as f64 / n as f64);
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if mant == 0 => sign << 31,
        0 => {
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (((127 - 15 + e + 1) as u32) << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => (sign << 31) | (0xff << 23) | (mant << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}
