//! gfx12 (RDNA4) FP8 (E4M3) HFQ4 residual GEMM kernel build-check.
//!
//! Compile-only validation that hipcc accepts the new
//! `gemm_hfq4g256_residual_fp8.gfx12.hip` source on gfx1201 silicon. Calls
//! `Gpu::ensure_kernel_public` for `gemm_hfq4g256_residual_fp8_gfx12` —
//! the function-load step exercises hipcc/lld end-to-end (FP8 wmma builtin
//! lowering + cvt_pk_fp8_f32 codegen) without dispatching the kernel.
//! Useful as a CI gate before wiring the kernel into Rust dispatch (a
//! separate task — first cut leaves the dispatch fn opt-in only via direct
//! call, NOT routed from production prefill).
//!
//! Run on gfx1201:
//!   cargo run --release -p rdna-compute --example build_check_fp8_gfx12
//!
//! On non-gfx12 archs the kernel stubs out via the `#if HIPFIRE_GFX12` guard,
//! so the example still verifies the source compiles cleanly through the
//! Rust-side hipcc pipeline (just doesn't exercise the FP8 wmma codegen).

use rdna_compute::Gpu;

// Embed the kernel source directly. The const in `rdna_compute::kernels` is
// not pub-reexported — examples consume the .hip file straight via
// include_str! (matches the pattern in `build_check_mmq_gfx12.rs` and the
// gfx12 layout probes' rust callers).
const GEMM_HFQ4G256_RESIDUAL_FP8_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_fp8.gfx12.hip");

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");

    let is_gfx12 = arch == "gfx1200" || arch == "gfx1201";
    if !is_gfx12 {
        eprintln!(
            "INFO: arch {arch} is not gfx12 — the kernel will stub out via \
             the #if HIPFIRE_GFX12 guard, but we still compile the source to \
             catch parse/preprocessor errors."
        );
    }

    eprintln!("\n--- compiling gemm_hfq4g256_residual_fp8_gfx12 ---");
    gpu.ensure_kernel_public(
        "gemm_hfq4g256_residual_fp8_gfx12",
        GEMM_HFQ4G256_RESIDUAL_FP8_GFX12_SRC,
        "gemm_hfq4g256_residual_fp8_gfx12",
    )
    .expect("gemm_hfq4g256_residual_fp8_gfx12 compile/load failed");
    eprintln!("  OK");

    if is_gfx12 {
        eprintln!(
            "\nPASS: kernel compiled and loaded cleanly on {arch}. \
             hipcc accepts the gfx12 FP8 source (FP8 wmma builtin + \
             cvt_pk_fp8_f32 codegen exercised). Next step: wire dispatch \
             entry point through production routing and run a numeric \
             correctness test against the FP16 dequant->WMMA reference \
             (separate task)."
        );
    } else {
        eprintln!(
            "\nPASS (stub mode): kernel compiled cleanly on {arch} via the \
             non-gfx12 stub branch. Re-run on gfx1200/gfx1201 to exercise \
             the FP8 wmma codegen path."
        );
    }
}
