//! gfx12 (RDNA4) iu8 K=16 WMMA layout-discovery probe — issue #136 part B.
//!
//! Probes `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` on gfx1201 to confirm
//! the wave32 layout before porting the existing RDNA3 MMQ kernel
//! (`gemm_hfq4g256_residual_mmq.hip`) to gfx12. The iu4 K=32 probe discovered
//! an 8-row-block acc layout — this probe checks whether iu8 K=16 follows
//! the same layout or differs.
//!
//! First-guess layout (same as iu4 K=32):
//!   A input: lane l → row m = l & 0xF, k-half = l >> 4 (8 INT8/half = int32x2 per lane)
//!   B input: lane l → col n = l & 0xF, k-half = l >> 4
//!   acc:     lane l, slot j ∈ [0,7] → C[8*(l>>4) + j][l & 0xF]
//!
//! Patterns (one wave per launch, dumped to consecutive 256-i32 regions):
//!   0 — A=1, B=1               → every entry = 16 (sanity)
//!   1 — A[m][k]=m+1, B=1       → acc[l, j] = 16 * (8*(l>>4) + j + 1)
//!   2 — A=1, B[k][n]=n+1       → acc[l, j] = 16 * ((l & 0xF) + 1), all 8 slots same
//!
//! Run on gfx1201:
//!   cargo run --release -p rdna-compute --example probe_wmma_iu8_k16_layout

use rdna_compute::{DType, Gpu};

fn pattern_label(p: usize) -> &'static str {
    match p {
        0 => "A=1, B=1",
        1 => "A=row-id+1, B=1",
        2 => "A=1, B=col-id+1",
        _ => "?",
    }
}

fn expected_pattern(p: usize, lane: usize, slot: usize) -> i32 {
    // First guess: 8-row-block acc layout (same as iu4 K=32).
    let row = (8 * (lane >> 4) + slot) as i32;
    let col = (lane & 0xF) as i32;
    match p {
        0 => 16,
        1 => 16 * (row + 1),
        2 => 16 * (col + 1),
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
             Current arch: {arch}. The iu8 K=16 builtin works on RDNA3 too \
             but with a different lane layout — see the RDNA3 MMQ kernel."
        );
        std::process::exit(0);
    }

    eprintln!("\n=== gfx12 iu8 K=16 WMMA layout-discovery probe ===");
    eprintln!(
        "First-guess wave32 layout (8-row-block, matching iu4 K=32):\n  \
         A input: lane l → row m = l & 0xF, k-half = l >> 4\n  \
         B input: lane l → col n = l & 0xF, k-half = l >> 4\n  \
         acc:     lane l, slot j → C[8*(l>>4) + j][l & 0xF]"
    );

    let out = gpu.alloc_tensor(&[3 * 32 * 8], DType::F32).expect("alloc out");

    gpu.probe_wmma_iu8_k16_layout(&out)
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
                  if pat_pass { "PASS" } else { "MISMATCH (first-guess layout is wrong)" });
        if pat_pass {
            eprintln!("  every (lane, slot) matched expected_pattern(p, lane, slot).");
        } else {
            eprintln!("  {} mismatches (showing first 32):", mismatches.len());
            for (lane, slot, got, want) in mismatches.iter().take(32) {
                eprintln!("    lane {lane:>2}, slot {slot}: got {got:>5}, want {want:>5}");
            }
            eprintln!("\n  full per-lane dump for pattern {p}:");
            for lane in 0..32 {
                let s = &region[lane * 8..(lane + 1) * 8];
                eprintln!("    lane {lane:>2}: {s:?}");
            }
        }
    }

    if all_pass {
        eprintln!(
            "\nPASS: all three patterns matched the first-guess layout. \
             gfx12 iu8 K=16 WMMA wave32 layout is the same 8-row-block as iu4 K=32:\n  \
             A input: lane l → row m = l & 0xF, k-half = l >> 4\n  \
             B input: lane l → col n = l & 0xF, k-half = l >> 4\n  \
             acc:     lane l, slot j → C[8*(l>>4) + j][l & 0xF]\n\
             The MMQ port can use the same acc-store path as the iu4 GEMM port."
        );
        std::process::exit(0);
    } else {
        eprintln!(
            "\nFAIL: at least one pattern mismatched. Inspect the per-lane dump \
             above to extract the actual layout map before porting the MMQ kernel."
        );
        std::process::exit(1);
    }
}
