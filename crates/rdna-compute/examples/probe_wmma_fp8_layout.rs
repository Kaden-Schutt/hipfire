//! gfx12 (RDNA4) FP8 (E4M3) WMMA layout-discovery probe — issue #136 part B.
//!
//! Probes `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` on gfx1201
//! to confirm the wave32 layout before writing an FP8 GEMM port. The iu4 K=32
//! and iu8 K=16 probes both discovered an 8-row-block acc layout — first guess
//! is FP8 wmma uses the same.
//!
//! Per CK header amd_wmma.hpp: signature is
//!   wmma_f32_16x16x16_fp8_fp8_w32_gfx12(int32x2 a, int32x2 b, float8 acc) -> float8
//! No clamp/sign args. Acc is 8 FP32 per lane.
//!
//! Patterns (one wave per launch, dumped to consecutive 256-FP32 regions):
//!   0 — A=B=1.0                 → every entry = 16.0 (sanity)
//!   1 — A[m][k]=(m+1).0, B=1.0  → acc[l, j] = 16 * (8*(l>>4) + j + 1)
//!   2 — A=1.0, B[k][n]=(n+1).0  → acc[l, j] = 16 * ((l & 0xF) + 1)
//!
//! Run on gfx1201:
//!   cargo run --release -p rdna-compute --example probe_wmma_fp8_layout

use rdna_compute::{DType, Gpu};

fn pattern_label(p: usize) -> &'static str {
    match p {
        0 => "A=1.0, B=1.0",
        1 => "A=row-id+1.0, B=1.0",
        2 => "A=1.0, B=col-id+1.0",
        _ => "?",
    }
}

fn expected_pattern(p: usize, lane: usize, slot: usize) -> f32 {
    // First guess: 8-row-block acc layout (same as iu4/iu8).
    let row = (8 * (lane >> 4) + slot) as f32;
    let col = (lane & 0xF) as f32;
    match p {
        0 => 16.0,
        1 => 16.0 * (row + 1.0),
        2 => 16.0 * (col + 1.0),
        _ => 0.0,
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");

    if !(arch == "gfx1200" || arch == "gfx1201") {
        eprintln!(
            "SKIP: this probe requires gfx1200/gfx1201 (RDNA4). \
             Current arch: {arch}. The FP8 wmma builtin only exists on RDNA4."
        );
        std::process::exit(0);
    }

    eprintln!("\n=== gfx12 FP8 (E4M3) WMMA layout-discovery probe ===");
    eprintln!(
        "First-guess wave32 layout (8-row-block, matching iu4/iu8):\n  \
         A input: int32x2/lane = 8 FP8 bytes = K-half of one row\n  \
         B input: int32x2/lane (same shape)\n  \
         acc:     float8/lane. lane l, slot j → C[8*(l>>4) + j][l & 0xF]"
    );

    let out = gpu.alloc_tensor(&[3 * 32 * 8], DType::F32).expect("alloc out");

    gpu.probe_wmma_fp8_layout(&out)
        .expect("probe dispatch failed");

    let host: Vec<f32> = gpu.download_f32(&out).expect("download out");
    assert_eq!(host.len(), 3 * 256);

    // FP8 → FP32 mul has perfect representation for integer values 1..16
    // (E4M3 represents these exactly). The accumulated dot product sums 16
    // exact integers, which fits exactly in FP32. So the comparison is
    // bit-exact in principle — use a tiny tolerance to absorb any
    // accumulator-order rounding.
    const TOL: f32 = 1e-4;

    let mut all_pass = true;

    for p in 0..3 {
        let region = &host[p * 256..(p + 1) * 256];
        let mut mismatches: Vec<(usize, usize, f32, f32)> = Vec::new();
        for lane in 0..32 {
            for slot in 0..8 {
                let v = region[lane * 8 + slot];
                let want = expected_pattern(p, lane, slot);
                if (v - want).abs() > TOL {
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
                eprintln!("    lane {lane:>2}, slot {slot}: got {got:>10.4}, want {want:>10.4}");
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
            "\nPASS: all three patterns matched. gfx12 FP8 (E4M3) wmma wave32 \
             layout is the same 8-row-block as iu4/iu8:\n  \
             A input: int32x2/lane (8 FP8 = K-half of one row)\n  \
             B input: int32x2/lane (same shape)\n  \
             acc:     float8/lane. lane l, slot j → C[8*(l>>4) + j][l & 0xF]\n\
             Safe to write the FP8 GEMM port using this layout — same acc-store \
             path as iu8 MMQ, just with FP32 acc instead of INT32."
        );
        std::process::exit(0);
    } else {
        eprintln!(
            "\nFAIL: at least one pattern mismatched. Inspect the per-lane dump \
             above to extract the actual layout map before writing the FP8 GEMM \
             port."
        );
        std::process::exit(1);
    }
}
