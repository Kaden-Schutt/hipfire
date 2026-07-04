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

//! Finite-difference gradcheck for the PFlash fp32 scoring head (fwd+bwd).
//! Never train on an unverified gradient.
//!
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "pflash-gradcheck"
//!   cargo run -p hipfire-train --release --example pflash_score_gradcheck

use hipfire_train::ops::pflash_score::{pflash_score_backward, pflash_score_forward};
use hipfire_rdna::{DType, Gpu};

const N_POS: usize = 8;
const KV_DIM: usize = 16;
const BLOCK: usize = 2; // → 4 blocks
const LAST: usize = N_POS - 1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    let nb = N_POS / BLOCK;
    println!("gradcheck pflash_score: n_pos={N_POS} kv_dim={KV_DIM} block={BLOCK} n_blocks={nb}");

    // deterministic pseudo-random k and loss weights w
    let mut s: u64 = 0x243F6A8885A308D3;
    let mut rng = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (1u64 << 31) as f32 - 1.0 // ~[-1,1)
    };
    let mut k: Vec<f32> = (0..N_POS * KV_DIM).map(|_| rng()).collect();
    let w: Vec<f32> = (0..nb).map(|_| rng()).collect();

    let scores_dev = gpu.zeros(&[nb], DType::F32)?;
    let dscores = gpu.upload_f32(&w, &[nb])?;

    // L(k) = Σ_b w_b * score_b
    let loss = |gpu: &mut Gpu, k: &[f32]| -> f32 {
        let kd = gpu.upload_f32(k, &[N_POS * KV_DIM]).unwrap();
        pflash_score_forward(gpu, &kd, &scores_dev, N_POS, KV_DIM, BLOCK, nb, LAST).unwrap();
        let sc = gpu.download_f32(&scores_dev).unwrap();
        sc.iter().zip(&w).map(|(a, b)| a * b).sum()
    };

    // analytic dk
    let kd = gpu.upload_f32(&k, &[N_POS * KV_DIM])?;
    pflash_score_forward(&mut gpu, &kd, &scores_dev, N_POS, KV_DIM, BLOCK, nb, LAST)?;
    let dk = pflash_score_backward(&mut gpu, &kd, &dscores, N_POS, KV_DIM, BLOCK, nb, LAST)?;
    let dk_host = gpu.download_f32(&dk)?;

    // FD over a spread of indices (include the last-token row, which every block
    // touches, and an interior block row).
    // torch.gradcheck-style tolerance: |a-fd| ≤ atol + rtol·|fd| (pure-relative
    // is unstable for small-magnitude components in fp32 FD).
    let h = 1e-3f32;
    let (atol, rtol) = (1e-3f32, 2e-2f32);
    let idxs = [
        0usize,
        1,
        5,
        17,
        33,
        LAST * KV_DIM,
        LAST * KV_DIM + 7,
        3 * KV_DIM + 2,
    ];
    let mut max_abs = 0.0f32;
    let mut all_ok = true;
    println!("\n  idx      analytic         fd        abs_err   tol      ok");
    for &i in &idxs {
        let orig = k[i];
        k[i] = orig + h;
        let lp = loss(&mut gpu, &k);
        k[i] = orig - h;
        let lm = loss(&mut gpu, &k);
        k[i] = orig;
        let fd = (lp - lm) / (2.0 * h);
        let a = dk_host[i];
        let abs = (a - fd).abs();
        let tol = atol + rtol * fd.abs();
        let ok = abs <= tol;
        all_ok &= ok;
        max_abs = max_abs.max(abs);
        println!(
            "  {i:>5} {a:>14.6} {fd:>12.6} {abs:>10.2e} {tol:>8.2e}   {}",
            if ok { "✓" } else { "✗" }
        );
    }
    println!("\n  max_abs_err={max_abs:.2e}  (atol={atol:.0e} rtol={rtol:.0e})");
    if all_ok {
        println!("  PASS ✓ (analytic gradient matches finite differences)");
        Ok(())
    } else {
        Err("gradcheck FAILED: some index exceeded atol+rtol·|fd|".into())
    }
}
