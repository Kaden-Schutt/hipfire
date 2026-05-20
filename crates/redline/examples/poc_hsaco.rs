// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Redline PoC: parse a real .hsaco kernel binary.
//! Extracts kernel name, register counts, LDS size — everything
//! needed to build PM4 dispatch packets.
//!
//! Usage: cargo run -p redline --example poc_hsaco -- kernels/compiled/gfx1010/add.hsaco

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: poc_hsaco <path/to/kernel.hsaco>");
        std::process::exit(1);
    });

    eprintln!("=== redline PoC: .hsaco parser ===\n");
    eprintln!("Parsing: {path}");

    let module = match redline::hsaco::HsacoModule::from_file(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("FAILED: {e}");
            std::process::exit(1);
        }
    };

    eprintln!("  .text offset: 0x{:x} ({} bytes)", module.text_offset, module.text_size);
    eprintln!("  Kernels found: {}\n", module.kernels.len());

    for k in &module.kernels {
        let vgprs = ((k.pgm_rsrc1 & 0x3F) + 1) * 4; // VGPR_COUNT field
        let sgprs = (((k.pgm_rsrc1 >> 6) & 0xF) + 1) * 8; // SGPR_COUNT field

        eprintln!("  Kernel: {}", k.name);
        eprintln!("    code_offset:           0x{:x}", k.code_offset);
        eprintln!("    pgm_rsrc1:             0x{:08x}", k.pgm_rsrc1);
        eprintln!("    pgm_rsrc2:             0x{:08x}", k.pgm_rsrc2);
        eprintln!("    VGPRs:                 {}", vgprs);
        eprintln!("    SGPRs:                 {}", sgprs);
        eprintln!("    group_segment (LDS):   {} bytes", k.group_segment_size);
        eprintln!("    private_segment:       {} bytes", k.private_segment_size);
        eprintln!("    kernarg_size:          {} bytes", k.kernarg_size);
        eprintln!();
    }

    if module.kernels.is_empty() {
        eprintln!("WARNING: no kernel descriptors found. The .hsaco may use a different symbol format.");
    } else {
        eprintln!("=== PASSED — kernel metadata extracted ===");
    }
}
