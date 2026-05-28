// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Inspect a .hfq/.mq4 file and report whether it carries AWQ scale
//! sidecars, GPTQ g_idx sidecars, or ParoQuant rotations.
//!
//! Run: ./target/release/examples/check_awq_gptq <path>

use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).expect("usage: check_awq_gptq <path>");
    let f = HfqFile::open(Path::new(&path)).expect("open");
    let tensors = f.tensors();
    let total = tensors.len();
    let awq = tensors.iter().filter(|t| t.name.contains("awq_scale")).count();
    let gptq = tensors.iter().filter(|t| t.name.contains("gptq") || t.name.contains("g_idx")).count();
    let paro = tensors.iter().filter(|t| t.name.contains("paro") || t.name.contains("rotation")).count();
    println!("file: {path}");
    println!("  total tensors:       {total}");
    println!("  awq_scale sidecars:  {awq}");
    println!("  gptq/g_idx sidecars: {gptq}");
    println!("  paro/rotation:       {paro}");
    if awq > 0 {
        println!("  sample awq tensors:");
        for t in tensors.iter().filter(|t| t.name.contains("awq_scale")).take(3) {
            println!("    {}", t.name);
        }
    }
    if gptq > 0 {
        println!("  sample gptq tensors:");
        for t in tensors.iter().filter(|t| t.name.contains("gptq") || t.name.contains("g_idx")).take(3) {
            println!("    {}", t.name);
        }
    }
    println!();
    println!("VERDICT:");
    if awq == 0 && gptq == 0 {
        println!("  PLAIN MQ4 — no AWQ/GPTQ calibration (LOWER quality, not recommended for benches)");
    } else if awq > 0 && gptq == 0 {
        println!("  AWQ-only — partial calibration");
    } else if awq == 0 && gptq > 0 {
        println!("  GPTQ-only — partial calibration");
    } else {
        println!("  AWQ+GPTQ stacked — full calibration (DEFAULT QUALITY)");
    }
}
