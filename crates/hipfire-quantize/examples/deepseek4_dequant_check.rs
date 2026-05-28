// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Verify the FP8 E4M3 + UE8M0-scale dequantizer against a known DeepSeek V4
//! tensor. Cross-checked against a stdlib-Python reference that read
//! the same bytes.
//!
//! Usage:
//!   cargo run --release --example deepseek4_dequant_check
//!
//! Hardcodes the path to the local DeepSeek V4 snapshot. Prints the first 16
//! dequantized values of `layers.0.ffn.experts.0.w1.weight` row 0 plus
//! the stats over the first 4096 weights. Compare against:
//!
//!   E4M3 values (row 0, cols 0..15):
//!     [-0.3750 +12.0000 -0.1562 +3.0000 -6.0000 +12.0000 +96.0000
//!      +13.0000 +0.3125 +0.3750 -32.0000 -96.0000 +0.4062 -416.0000
//!      -0.0098 +2.5000]
//!   × scale (2^-7 ≈ 0.0078125):
//!     [-0.002930 +0.093750 -0.001221 +0.023438 -0.046875 +0.093750
//!      +0.750000 +0.101562 +0.002441 +0.002930 -0.250000 -0.750000
//!      +0.003174 -3.250000 -0.000076 +0.019531]
//!   first 4096: min=-5.5 p1=-2.0 median=0.0 p99=2.5 max=5.0

use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;

#[derive(Debug, Deserialize)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if (byte & 0x80) != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0xf) as i32;
    let mant = (byte & 0x7) as f32;
    if exp == 0xf && mant == 7.0 { return 0.0; }
    if exp == 0 {
        if mant == 0.0 { return 0.0; }
        return sign * (2.0f32.powi(-6)) * (mant / 8.0);
    }
    sign * (2.0f32.powi(exp - 7)) * (1.0 + mant / 8.0)
}

fn ue8m0_to_scale(byte: u8) -> f32 {
    2.0f32.powi(byte as i32 - 127)
}

fn main() {
    let snap = "/home/nick/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136";
    let shard = format!("{snap}/model-00002-of-00046.safetensors");

    let file = File::open(&shard).expect("open shard");
    let mmap = unsafe { Mmap::map(&file).expect("mmap shard") };
    let hdr_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let hdr_json: serde_json::Value =
        serde_json::from_slice(&mmap[8..8 + hdr_len]).expect("parse header");
    let body = &mmap[8 + hdr_len..];

    let mut metas: HashMap<String, TensorMeta> = HashMap::new();
    if let serde_json::Value::Object(map) = hdr_json {
        for (k, v) in map {
            if k == "__metadata__" { continue; }
            metas.insert(k, serde_json::from_value(v).unwrap());
        }
    }

    let w_name = "layers.0.ffn.experts.0.w1.weight";
    let s_name = "layers.0.ffn.experts.0.w1.scale";
    let wm = metas.get(w_name).expect("weight meta missing");
    let sm = metas.get(s_name).expect("scale meta missing");

    eprintln!("weight: shape {:?} dtype {} bytes {}",
        wm.shape, wm.dtype, wm.data_offsets[1] - wm.data_offsets[0]);
    eprintln!("scale:  shape {:?} dtype {} bytes {}",
        sm.shape, sm.dtype, sm.data_offsets[1] - sm.data_offsets[0]);

    let w_bytes = &body[wm.data_offsets[0]..wm.data_offsets[1]];
    let s_bytes = &body[sm.data_offsets[0]..sm.data_offsets[1]];

    // Row 0, cols 0..15 — same data Python printed
    eprintln!("\nrow 0, cols 0..15 raw bytes:");
    let hex: String = w_bytes[..16].iter().map(|b| format!("{:02x}", b)).collect();
    eprintln!("  {hex}");

    eprintln!("scale[0,0] = 0x{:02x} (= 2^{})", s_bytes[0], s_bytes[0] as i32 - 127);

    let e4m3: Vec<f32> = w_bytes[..16].iter().map(|&b| e4m3_to_f32(b)).collect();
    let scale = ue8m0_to_scale(s_bytes[0]);
    let dequant: Vec<f32> = e4m3.iter().map(|&v| v * scale).collect();

    eprint!("E4M3 values: ");
    for v in &e4m3 { eprint!("{:+.4} ", v); }
    eprintln!();
    eprint!("× scale:     ");
    for v in &dequant { eprint!("{:+.6} ", v); }
    eprintln!();

    // First 4096 dequantized — stats
    let (rows, cols) = (wm.shape[0], wm.shape[1]);
    let (_sr, sc) = (sm.shape[0], sm.shape[1]);
    let block_cols = cols / sc;
    let mut sample: Vec<f32> = (0..4096.min(rows * cols))
        .map(|i| {
            let r = i / cols;
            let c = i % cols;
            let s = s_bytes[r * sc + c / block_cols];
            e4m3_to_f32(w_bytes[i]) * ue8m0_to_scale(s)
        })
        .collect();
    sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sample.len();
    eprintln!("\nfirst 4096 dequantized:");
    eprintln!("  min={:.4} p1={:.4} median={:.4} p99={:.4} max={:.4}",
        sample[0], sample[n / 100], sample[n / 2], sample[99 * n / 100], sample[n - 1]);
    eprintln!("  fraction |w|<0.1: {:.3}",
        sample.iter().filter(|&&v| v.abs() < 0.1).count() as f32 / n as f32);
    eprintln!("  fraction |w|<1.0: {:.3}",
        sample.iter().filter(|&&v| v.abs() < 1.0).count() as f32 / n as f32);
}
