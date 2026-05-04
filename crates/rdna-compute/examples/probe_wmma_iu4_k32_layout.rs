//! gfx12 (RDNA4) iu4 K=32 WMMA layout-discovery probe — issue #136 part B.
//!
//! Runs three patterns through `__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12`
//! to confirm the wave32 WMMA layout before writing the production GEMM port.
//!
//! Confirmed layout (gfx12 iu4 K=32 wmma, wave32) — discovered 2026-05-03 on R9700:
//!   A input: lane l → row m = l & 0xF, k-half = l >> 4. v2i[0]/v2i[1] each
//!            pack 8 INT4 of that half-row.
//!   B input: lane l → col n = l & 0xF, k-half = l >> 4.
//!   acc:     lane l, slot j ∈ [0,7] → C[8*(l>>4) + j][l & 0xF].
//!
//! This is an **8-row-block** layout — DIFFERENT from the RDNA3
//! wmma_f32_16x16x16_f16 stride-2 interleave (slot j → row 2j + (l>>4)).
//! Lanes 0..15 own rows 0..7, lanes 16..31 own rows 8..15. The production
//! GEMM port writes acc-to-C using this 8-row-block formula.
//!
//! Patterns (run in one wave, dumped to consecutive 256-i32 regions):
//!   0 — A=ones, B=ones    → every entry = 32 (sanity)
//!   1 — A=row-id, B=ones  → acc[l, j] = 32 * (8*(l>>4) + j), independent of (l & 0xF)
//!   2 — A=ones, B=col-id  → acc[l, j] = 32 * (l & 0xF), same across all 8 slots
//!
//! Run on gfx1201:
//!   cargo run --release -p rdna-compute --example probe_wmma_iu4_k32_layout

use rdna_compute::{DType, Gpu};

fn pattern_label(p: usize) -> &'static str {
    match p {
        0 => "A=ones, B=ones",
        1 => "A=row-id, B=ones",
        2 => "A=ones, B=col-id",
        _ => "?",
    }
}

fn expected_pattern(p: usize, lane: usize, slot: usize) -> i32 {
    // 8-row-block acc layout: lane l, slot j → C[8*(l>>4) + j][l & 0xF].
    let row = (8 * (lane >> 4) + slot) as i32;
    let col = (lane & 0xF) as i32;
    match p {
        0 => 32,
        1 => 32 * row,
        2 => 32 * col,
        _ => 0,
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");

    if !(arch == "gfx1200" || arch == "gfx1201") {
        eprintln!(
            "SKIP: this probe requires gfx1200/gfx1201 (RDNA4). \
             Current arch: {arch}. The iu4 K=32 builtin only exists on RDNA4."
        );
        std::process::exit(0);
    }

    eprintln!("\n=== gfx12 iu4 K=32 WMMA layout-discovery probe ===");
    eprintln!(
        "Confirmed wave32 layout (locked in 2026-05-03 on R9700):\n  \
         A input: lane l → row m = l & 0xF, k-half = l >> 4\n  \
         B input: lane l → col n = l & 0xF, k-half = l >> 4\n  \
         acc:     lane l, slot j → C[8*(l>>4) + j][l & 0xF]   (8-row-block, NOT RDNA3 stride-2)"
    );

    // 3 patterns × 32 lanes × 8 slots = 768 i32. Allocate as F32, byte-reinterpret.
    let out = gpu.alloc_tensor(&[3 * 32 * 8], DType::F32).expect("alloc out");

    gpu.probe_wmma_iu4_k32_layout(&out)
        .expect("probe dispatch failed");

    let raw_f32 = gpu.download_f32(&out).expect("download out");
    let host: Vec<i32> = raw_f32.iter().map(|f| f.to_bits() as i32).collect();
    assert_eq!(host.len(), 3 * 256);

    let mut all_pass = true;

    for p in 0..3 {
        let region = &host[p * 256..(p + 1) * 256];
        let mut mismatches: Vec<(usize, usize, i32, i32)> = Vec::new();
        for lane in 0..32 {
            for slot in 0..8 {
                let v = region[lane * 8 + slot];
                let want = expected_pattern(p, lane, slot);
                if v != want {
                    mismatches.push((lane, slot, v, want));
                }
            }
        }
        let pat_pass = mismatches.is_empty();
        all_pass &= pat_pass;

        eprintln!("\n--- pattern {p} ({}) — {} ---",
                  pattern_label(p),
                  if pat_pass { "PASS" } else { "MISMATCH (assumed layout is wrong)" });
        if pat_pass {
            // Print first row of expected for context.
            eprintln!("  every (lane, slot) matched expected_pattern(p, lane, slot).");
        } else {
            eprintln!("  {} mismatches (showing first 32):", mismatches.len());
            for (lane, slot, got, want) in mismatches.iter().take(32) {
                eprintln!("    lane {lane:>2}, slot {slot}: got {got:>5}, want {want:>5}");
            }
            // Full per-lane dump for offline analysis.
            eprintln!("\n  full per-lane dump for pattern {p}:");
            for lane in 0..32 {
                let s = &region[lane * 8..(lane + 1) * 8];
                eprintln!("    lane {lane:>2}: {s:?}");
            }
        }
    }

    if all_pass {
        eprintln!(
            "\nPASS: all three patterns matched. gfx12 iu4 K=32 WMMA wave32 layout is:\n  \
             A input: lane l → row m = l & 0xF, k-half = l >> 4\n  \
             B input: lane l → col n = l & 0xF, k-half = l >> 4\n  \
             acc:     lane l, slot j → C[8*(l>>4) + j][l & 0xF]\n\
             Safe to write the production GEMM port against this layout."
        );
        std::process::exit(0);
    } else {
        eprintln!(
            "\nFAIL: at least one pattern did not match the assumed layout. \
             Inspect per-lane dump above to extract the actual layout map \
             before writing the production GEMM port."
        );
        std::process::exit(1);
    }
}
