//! Per-layer NRMSE verification of hipfire's ZAYA1 forward against a
//! PyTorch reference dump.
//!
//! Phase 1 status: SKELETON. The harness layout matches the pattern
//! described in the project's Gemma 4 / Qwen 3.5 port methodology
//! (PyTorch + safetensors as oracle, hipfire as system under test,
//! per-op NRMSE with ~5e-3 bf16 ULP threshold). Bodies return early
//! with a clear error until the forward pass and reference dumper
//! land.
//!
//! Workflow once forward exists:
//!   1. Run `scripts/arch-intake/dump_zaya_reference.py` on hiptrx
//!      with `HIP_VISIBLE_DEVICES=2` to populate
//!      `/tmp/zaya-port/refs/<prompt-hash>/layer_NN/<step>.<side>.safetensors`
//!      and `manifest.json`.
//!   2. Build hipfire with the dump-hooks env wired into ZAYA1's
//!      forward (HIPFIRE_DUMP_DIR / HIPFIRE_DUMP_POS).
//!   3. Run this example with `--ref /tmp/zaya-port/refs/<hash>` and
//!      `--out /tmp/zaya-port/hipfire/<hash>` and `--diff
//!      /tmp/zaya-port/diffs/<hash>`.
//!   4. Per-op NRMSE matrix prints; first divergence above 5e-3 is the
//!      first-broken op (see Gemma 4 V-norm story for the canonical
//!      bisection pattern).

fn main() {
    eprintln!(
        "verify_against_torch (zaya): SKELETON. Phase 1 scaffold returns \
         early. The forward pass, reference-dump generator, and diff \
         harness arrive incrementally. See \
         docs/investigations/2026-05-07-zaya1-port-intake/ for status."
    );
    std::process::exit(2);
}
