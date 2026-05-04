//! gfx12 (RDNA4) iu4 K=32 WMMA probe — POC for issue #136 part B.
//!
//! Validates that `__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12` works
//! correctly on gfx1201 silicon (R9700, 9070 XT). Probe-1 uses A=B=all-ones
//! (every INT4 = 1); expected output at every (m, n) in the 16x16 result is
//! `sum_{k=0..31} 1*1 = 32`. This is layout-independent — every lane should
//! read 32 in every accumulator slot — so it's a clean yes/no on whether
//! the intrinsic dispatches and accumulates correctly without first having
//! to discover the per-lane element layout.
//!
//! Run on gfx1201:
//!   cargo run --release -p rdna-compute --example probe_wmma_iu4_k32

use rdna_compute::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {}", arch);

    if !(arch == "gfx1200" || arch == "gfx1201") {
        eprintln!(
            "SKIP: this probe requires gfx1200/gfx1201 (RDNA4). \
             Current arch: {arch}. The iu4 K=32 builtin only exists on RDNA4."
        );
        std::process::exit(0);
    }

    eprintln!("\n=== gfx12 iu4 K=32 WMMA probe-1: all-ones ===");
    eprintln!(
        "Input A: 16x32 INT4, every value = 1 (packed as 0x11111111 per int)\n\
         Input B: 16x32 INT4, every value = 1\n\
         Expected C[m][n] = sum_{{k=0..31}} 1*1 = 32 for every (m, n)\n\
         Expected per-lane accumulator: every one of the 8 ints = 32"
    );

    // Output: 32 lanes x 8 ints = 256 i32. Allocate as F32 (same 4 bytes/element);
    // the kernel writes int32 and we reinterpret on the host side.
    let out = gpu.alloc_tensor(&[32 * 8], DType::F32).expect("alloc out");

    gpu.probe_wmma_iu4_k32_allones(&out)
        .expect("probe dispatch failed");

    // Sync, read back as i32 by reinterpreting the f32 download (the
    // download path doesn't have a dedicated i32 helper). We allocated
    // the tensor as I32 so the byte layout is correct; we just need to
    // get the bytes back.
    let raw_f32 = gpu.download_f32(&out).expect("download out");
    // Same-byte reinterpret f32 -> i32 element-wise.
    let out_host: Vec<i32> = raw_f32
        .iter()
        .map(|f| f.to_bits() as i32)
        .collect();

    let n_outputs = 32 * 8;
    let mut all_pass = true;
    let mut nonzero = 0usize;
    let mut histogram = std::collections::BTreeMap::<i32, usize>::new();
    for &v in &out_host {
        *histogram.entry(v).or_insert(0) += 1;
        if v != 0 {
            nonzero += 1;
        }
        if v != 32 {
            all_pass = false;
        }
    }

    eprintln!("\n--- result histogram (value : count out of {n_outputs}) ---");
    for (v, c) in &histogram {
        eprintln!("  {v:>10} : {c}");
    }
    eprintln!("non-zero entries: {nonzero}/{n_outputs}");

    if all_pass {
        eprintln!("\nPASS: every accumulator entry = 32. wmma_i32_16x16x32_iu4_w32_gfx12 works on {arch}.");
        std::process::exit(0);
    } else {
        eprintln!(
            "\nFAIL: not every entry equals 32. The intrinsic ran (output is {nonzero}/{n_outputs} non-zero) \
             but produced unexpected values. Per-lane dump below."
        );
        for lane in 0..32 {
            let slice = &out_host[lane * 8..(lane + 1) * 8];
            eprintln!("  lane {lane:>2}: {slice:?}");
        }
        std::process::exit(1);
    }
}
