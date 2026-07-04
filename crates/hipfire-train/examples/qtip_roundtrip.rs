#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

//! QTIP quantize→dequant roundtrip sanity (Phase 2 Q0, CPU-only — no GPU).
//!
//! Verifies the vendored encoder: on synthetic Gaussian weights, the QTIP
//! reconstruction MSE should be (a) far below the signal variance (real
//! quantization, not noise) and (b) lower at 3-bit than 2-bit. Mirrors
//! hipfire-quantize/src/qtip.rs's own quality tests.
//!
//! Run: cargo run -p hipfire-train --release --example qtip_roundtrip

use hipfire_train::qtip_quant::qtip_quantize_dequant;

fn main() {
    // Deterministic standard-normal sample (Box–Muller over an LCG).
    let n = 256 * 8;
    let mut st: u64 = 0x1234_5678;
    let mut next = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 33) as f64) / (1u64 << 31) as f64
    };
    let mut w = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let u1 = (next()).max(1e-12);
        let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        w[i] = (r * (2.0 * std::f64::consts::PI * u2).cos()) as f32;
        if i + 1 < n {
            w[i + 1] = (r * (2.0 * std::f64::consts::PI * u2).sin()) as f32;
        }
        i += 2;
    }

    let var: f32 = w.iter().map(|x| x * x).sum::<f32>() / n as f32;
    let mse = |a: &[f32], b: &[f32]| -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len() as f32
    };

    let hat2 = qtip_quantize_dequant(&w, 2, 64);
    let hat3 = qtip_quantize_dequant(&w, 3, 64);
    let mse2 = mse(&w, &hat2);
    let mse3 = mse(&w, &hat3);

    println!("signal var          = {var:.4}");
    println!(
        "QTIP-2 recon MSE    = {mse2:.4}  (MSE/var = {:.3})",
        mse2 / var
    );
    println!(
        "QTIP-3 recon MSE    = {mse3:.4}  (MSE/var = {:.3})",
        mse3 / var
    );

    assert!(mse2 < var * 0.5, "QTIP-2 MSE not well below variance");
    assert!(mse3 < mse2, "QTIP-3 should beat QTIP-2");
    println!("\nPASS — vendored QTIP encoder reconstructs Gaussians; 3-bit beats 2-bit.");
}
