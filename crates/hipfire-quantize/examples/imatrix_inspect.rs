// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Inspect a llama.cpp-format imatrix GGUF file. Use existing `GgufFile`.

use hipfire_quantize::gguf_input::GgufFile;
use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).expect("usage: imatrix_inspect <path.gguf>");
    let gguf = GgufFile::open(Path::new(&path)).expect("open gguf");
    println!("version={} n_tensors={} kv_count={}",
        gguf.version, gguf.tensors.len(), gguf.metadata.len());
    println!("--- metadata ---");
    for (k, v) in &gguf.metadata {
        println!("  {}: {:?}", k, v);
    }
    println!("--- tensors (first 12) ---");
    for t in gguf.tensors.iter().take(12) {
        println!("  {:60} shape={:?} dtype={:?} offset={}",
            t.name, t.shape, t.dtype, t.offset);
    }
    println!("--- ffn_gate_exps + counts (layer 0) ---");
    for t in gguf.tensors.iter() {
        if t.name.starts_with("blk.0.ffn_gate_exps.weight.") {
            let bytes = gguf.tensor_data(t);
            let f32_data: Vec<f32> = bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]]))
                .collect();
            println!("  {} shape={:?} numel={} bytes={}",
                t.name, t.shape, f32_data.len(), bytes.len());
            println!("    first 16: {:?}", &f32_data[..16.min(f32_data.len())]);
            if f32_data.len() > 256 {
                println!("    at  256..272: {:?}", &f32_data[256..272]);
                println!("    at 2048..2064: {:?}", &f32_data[2048..2064.min(f32_data.len())]);
            }
            let mx = f32_data.iter().cloned().fold(f32::MIN, f32::max);
            let mn = f32_data.iter().cloned().fold(f32::MAX, f32::min);
            let mean: f64 = f32_data.iter().map(|&x| x as f64).sum::<f64>() / f32_data.len() as f64;
            println!("    min={:.3} max={:.3} mean={:.3}", mn, mx, mean);
        }
    }
}
