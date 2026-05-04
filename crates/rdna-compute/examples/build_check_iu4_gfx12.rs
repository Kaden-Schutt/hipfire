//! gfx12 (RDNA4) iu4 K=32 HFQ4 residual GEMM + Q4_1 quantizer build-check.
//!
//! Compile-only validation that hipcc accepts the new
//! `gemm_hfq4g256_residual_iu4.gfx12.hip` and `quantize_q4_1.gfx12.hip`
//! sources on gfx1201 silicon. Calls `Gpu::ensure_kernel_public` for both
//! `quantize_q4_1_mmq_ds4_gfx12` and `gemm_hfq4g256_residual_iu4_gfx12` —
//! the function-load step exercises hipcc/lld end-to-end (iu4 K=32 wmma
//! builtin lowering + INT4 byte-pack codegen) without dispatching either
//! kernel.
//!
//! Useful as a CI gate before wiring the kernel into Rust dispatch (a
//! separate task — first cut leaves `gemm_hfq4g256_residual_iu4_gfx12`
//! opt-in only via direct call, NOT routed from production prefill).
//!
//! Run on gfx1201:
//!   cargo run --release -p rdna-compute --example build_check_iu4_gfx12
//!
//! On non-gfx12 archs both kernels stub out via the `#if HIPFIRE_GFX12`
//! guard, so the example still verifies the source compiles cleanly through
//! the Rust-side hipcc pipeline (just doesn't exercise the iu4 wmma codegen).

use rdna_compute::Gpu;

// Embed the kernel sources directly. The const in `rdna_compute::kernels` is
// not pub-reexported — examples consume the .hip file straight via
// include_str! (matches the pattern in `build_check_fp8_gfx12.rs` and the
// gfx12 layout probes' rust callers).
const QUANTIZE_Q4_1_GFX12_SRC: &str =
    include_str!("../../../kernels/src/quantize_q4_1.gfx12.hip");
const GEMM_HFQ4G256_RESIDUAL_IU4_GFX12_SRC: &str =
    include_str!("../../../kernels/src/gemm_hfq4g256_residual_iu4.gfx12.hip");

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");

    let is_gfx12 = arch == "gfx1200" || arch == "gfx1201";
    if !is_gfx12 {
        eprintln!(
            "INFO: arch {arch} is not gfx12 — both kernels will stub out via \
             the #if HIPFIRE_GFX12 guard, but we still compile the source to \
             catch parse/preprocessor errors."
        );
    }

    eprintln!("\n--- compiling quantize_q4_1_mmq_ds4_gfx12 ---");
    gpu.ensure_kernel_public(
        "quantize_q4_1_mmq_ds4_gfx12",
        QUANTIZE_Q4_1_GFX12_SRC,
        "quantize_q4_1_mmq_ds4_gfx12",
    )
    .expect("quantize_q4_1_mmq_ds4_gfx12 compile/load failed");
    eprintln!("  OK");

    eprintln!("\n--- compiling gemm_hfq4g256_residual_iu4_gfx12 ---");
    gpu.ensure_kernel_public(
        "gemm_hfq4g256_residual_iu4_gfx12",
        GEMM_HFQ4G256_RESIDUAL_IU4_GFX12_SRC,
        "gemm_hfq4g256_residual_iu4_gfx12",
    )
    .expect("gemm_hfq4g256_residual_iu4_gfx12 compile/load failed");
    eprintln!("  OK");

    if is_gfx12 {
        eprintln!(
            "\nPASS: both kernels compiled and loaded cleanly on {arch}. \
             hipcc accepts the gfx12 iu4 K=32 source (iu4 wmma builtin + \
             INT4 byte-pack codegen exercised). Next step: run a numeric \
             correctness test against the FP16 dequant->WMMA reference \
             (separate task — wire ensure_q4_1_x + \
             gemm_hfq4g256_residual_iu4_gfx12 into a smoke harness)."
        );
    } else {
        eprintln!(
            "\nPASS (stub mode): both kernels compiled cleanly on {arch} via \
             the non-gfx12 stub branch. Re-run on gfx1200/gfx1201 to exercise \
             the iu4 K=32 wmma codegen path."
        );
    }
}
