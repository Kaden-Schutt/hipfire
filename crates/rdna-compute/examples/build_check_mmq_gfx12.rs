//! gfx12 (RDNA4) MMQ kernel build-check.
//!
//! Compile-only validation that hipcc accepts the new
//! `gemm_hfq4g256_residual_mmq.gfx12.hip` source on gfx1201 silicon. Calls
//! `Gpu::ensure_kernel_public` for both `quantize_q8_1_mmq_ds4_gfx12` and
//! `gemm_hfq4g256_residual_mmq_gfx12` — the function-load step exercises
//! hipcc/lld end-to-end without dispatching either kernel. Useful as a CI
//! gate before wiring the kernels into Rust dispatch (a separate task).
//!
//! Run on gfx1201:
//!   cargo run --release -p rdna-compute --example build_check_mmq_gfx12
//!
//! On non-gfx12 archs the kernels stub out via the `#if HIPFIRE_GFX12` guard,
//! so the example still verifies the source compiles cleanly through the
//! Rust-side hipcc pipeline (just doesn't exercise the WMMA codegen path).
//! On those archs we still attempt to load both functions to surface any
//! preprocessor / parse errors early.

use rdna_compute::Gpu;

// Embed the kernel source directly. The const in `rdna_compute::kernels` is
// not pub-reexported — examples consume the .hip file straight via
// include_str! (matches the pattern used by the gfx12 layout probes' rust
// callers, which only need the source string for ensure_kernel).
const GEMM_HFQ4G256_RESIDUAL_MMQ_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq.gfx12.hip");

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");

    let is_gfx12 = arch == "gfx1200" || arch == "gfx1201";
    if !is_gfx12 {
        eprintln!(
            "INFO: arch {arch} is not gfx12 — the kernels will stub out via \
             the #if HIPFIRE_GFX12 guard, but we still compile the source to \
             catch parse/preprocessor errors."
        );
    }

    eprintln!("\n--- compiling quantize_q8_1_mmq_ds4_gfx12 ---");
    gpu.ensure_kernel_public(
        "gemm_hfq4g256_residual_mmq_gfx12",
        GEMM_HFQ4G256_RESIDUAL_MMQ_GFX12_SRC,
        "quantize_q8_1_mmq_ds4_gfx12",
    )
    .expect("quantize_q8_1_mmq_ds4_gfx12 compile/load failed");
    eprintln!("  OK");

    eprintln!("\n--- compiling gemm_hfq4g256_residual_mmq_gfx12 ---");
    gpu.ensure_kernel_public(
        "gemm_hfq4g256_residual_mmq_gfx12",
        GEMM_HFQ4G256_RESIDUAL_MMQ_GFX12_SRC,
        "gemm_hfq4g256_residual_mmq_gfx12",
    )
    .expect("gemm_hfq4g256_residual_mmq_gfx12 compile/load failed");
    eprintln!("  OK");

    if is_gfx12 {
        eprintln!(
            "\nPASS: both kernels compiled and loaded cleanly on {arch}. \
             hipcc accepts the gfx12 MMQ source. Next step: wire dispatch \
             entry points and run a numeric correctness test against the \
             RDNA3 reference (separate task)."
        );
    } else {
        eprintln!(
            "\nPASS (stub mode): kernels compiled cleanly on {arch} via the \
             non-gfx12 stub branch. Re-run on gfx1200/gfx1201 to exercise \
             the WMMA codegen path."
        );
    }
}
