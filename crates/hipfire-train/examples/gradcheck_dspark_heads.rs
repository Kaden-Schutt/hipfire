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

//! Finite-difference gradcheck for the DSpark drafter HEADS (T2b):
//! lm-head + VanillaMarkov (`markov_w1`/`markov_w2`) + AcceptRatePredictor
//! (`confidence_proj`/`confidence_bias`). Loss
//!   L = Σ draft_logits ∘ G1 + Σ confidence_pred ∘ G2
//! ⇒ d_draft_logits = G1, d_confidence_pred = G2. Confirms the four head param
//! grads AND the `d_x_head` input grad match central differences, exercising
//! both the `d_draft_logits` seed (lm-head + markov_w2 + markov_w1 paths) and
//! the `d_confidence_pred` seed (sigmoid → proj/bias → x_head + markov_latent).
//!
//! DO NOT run on nix2 (LDS hazard); compile-gated by the build gate only.
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire lock acquire "gradcheck-dspark-heads"
//!   cargo run -p hipfire-train --release --example gradcheck_dspark_heads

use hipfire_rdna::{Gpu, HipResult};
use hipfire_train::dspark_drafter::{
    dspark_heads_backward, dspark_heads_forward, free_dspark_heads_acts, DsparkHeadsConfig,
    DsparkHeadsWeights,
};

const H: usize = 12;
const VOCAB: usize = 16;
const RANK: usize = 4;
const BLOCK: usize = 3;

fn cfg() -> DsparkHeadsConfig {
    DsparkHeadsConfig {
        h: H,
        vocab: VOCAB,
        markov_rank: RANK,
    }
}

/// Deterministic seeded fill; distinct `seed` per tensor so weights differ.
fn seeded(n: usize, seed: u64, scale: f32, off: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (1u64 << 31) as f32 - 1.0) * scale + off
        })
        .collect()
}

fn build(gpu: &mut Gpu) -> HipResult<DsparkHeadsWeights> {
    Ok(DsparkHeadsWeights {
        markov_w1: gpu.upload_f32(&seeded(VOCAB * RANK, 11, 0.08, 0.0), &[VOCAB, RANK])?,
        markov_w2: gpu.upload_f32(&seeded(VOCAB * RANK, 12, 0.06, 0.0), &[VOCAB, RANK])?,
        confidence_proj: gpu.upload_f32(&seeded(H + RANK, 13, 0.07, 0.0), &[1, H + RANK])?,
        confidence_bias: gpu.upload_f32(&seeded(1, 14, 0.05, 0.1), &[1])?,
    })
}

fn bytemuck_cast(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);
    let c = cfg();

    // prev_tokens must repeat a token so markov_w1 grad on that row is nonzero.
    let prev_tokens: Vec<u32> = vec![2, 7, 2];
    let xh: Vec<f32> = (0..BLOCK * H)
        .map(|i| ((i * 17 % 11) as f32) * 0.1 - 0.4)
        .collect();
    let lm: Vec<f32> = seeded(VOCAB * H, 21, 0.05, 0.0);
    let g1: Vec<f32> = (0..BLOCK * VOCAB)
        .map(|i| ((i * 7 % 5) as f32) * 0.05 - 0.1)
        .collect();
    let g2: Vec<f32> = (0..BLOCK)
        .map(|i| ((i * 3 % 4) as f32) * 0.2 - 0.3)
        .collect();

    let weights = build(&mut gpu)?;
    let lm_head = gpu.upload_f32(&lm, &[VOCAB * H])?;

    // Loss over both heads, rebuilding x_head each call (host-perturbable).
    let loss = |gpu: &mut Gpu, w: &DsparkHeadsWeights, xhv: &[f32]| -> HipResult<f32> {
        let x_head = gpu.upload_f32(xhv, &[BLOCK * H])?;
        let acts = dspark_heads_forward(gpu, &x_head, &prev_tokens, &lm_head, w, &c)?;
        let dl = gpu.download_f32(&acts.draft_logits)?;
        let cp = gpu.download_f32(&acts.confidence_pred)?;
        let l: f32 = dl.iter().zip(&g1).map(|(a, b)| a * b).sum::<f32>()
            + cp.iter().zip(&g2).map(|(a, b)| a * b).sum::<f32>();
        free_dspark_heads_acts(gpu, acts)?;
        gpu.free_tensor(x_head)?;
        Ok(l)
    };

    // Analytic grads (both seeds active).
    let x_head = gpu.upload_f32(&xh, &[BLOCK * H])?;
    let acts = dspark_heads_forward(&mut gpu, &x_head, &prev_tokens, &lm_head, &weights, &c)?;
    let d_draft_logits = gpu.upload_f32(&g1, &[BLOCK * VOCAB])?;
    let d_confidence_pred = gpu.upload_f32(&g2, &[BLOCK])?;
    let (d_x_head, grads) = dspark_heads_backward(
        &mut gpu,
        &d_draft_logits,
        &d_confidence_pred,
        &acts,
        &x_head,
        &lm_head,
        &weights,
        &c,
    )?;
    let gflat = grads.flat();
    let pflat = weights.params();
    assert_eq!(gflat.len(), pflat.len());
    let d_xh_a = gpu.download_f32(&d_x_head)?;

    let names = [
        "markov_w1",
        "markov_w2",
        "confidence_proj",
        "confidence_bias",
    ];
    // (param_index, element_index). markov_w1 idx 8 = token 2 (row 2) col 0.
    let probes: &[(usize, usize)] = &[(0, 8), (0, 9), (1, 5), (2, 3), (2, H + 1), (3, 0)];

    let hh = 1e-3f32;
    let (atol, rtol) = (2e-3f32, 3e-2f32);
    let mut all_ok = true;
    println!("  param                  idx   analytic         fd        abs_err   tol    ok");
    for &(pi, ei) in probes {
        let a = gpu.download_f32(gflat[pi])?[ei];
        let mut host = gpu.download_f32(pflat[pi])?;
        let orig = host[ei];
        host[ei] = orig + hh;
        gpu.memcpy_htod_auto(&pflat[pi].buf, bytemuck_cast(&host))?;
        let lp = loss(&mut gpu, &weights, &xh)?;
        host[ei] = orig - hh;
        gpu.memcpy_htod_auto(&pflat[pi].buf, bytemuck_cast(&host))?;
        let lm_ = loss(&mut gpu, &weights, &xh)?;
        host[ei] = orig;
        gpu.memcpy_htod_auto(&pflat[pi].buf, bytemuck_cast(&host))?;
        let fd = (lp - lm_) / (2.0 * hh);
        let abs = (a - fd).abs();
        let tol = atol + rtol * fd.abs();
        let ok = abs <= tol;
        all_ok &= ok;
        println!(
            "  {:<20} {:>4} {:>12.6} {:>12.6} {:>10.2e} {:>8.2e} {}",
            names[pi],
            ei,
            a,
            fd,
            abs,
            tol,
            if ok { "OK" } else { "XX" }
        );
    }

    // d_x_head input grad (sums lm-head path + confidence path).
    {
        let mut e = 0.0f32;
        for i in 0..xh.len() {
            let mut hp = xh.clone();
            hp[i] += hh;
            let mut hm = xh.clone();
            hm[i] -= hh;
            let lp = loss(&mut gpu, &weights, &hp)?;
            let lm_ = loss(&mut gpu, &weights, &hm)?;
            let fd = (lp - lm_) / (2.0 * hh);
            e = e.max((d_xh_a[i] - fd).abs());
        }
        let tol = atol + rtol * 1.0;
        let ok = e < 5e-2f32;
        all_ok &= ok;
        println!(
            "  {:<20} {:>4} {:>12} {:>12} {:>10.2e} {:>8.2e} {}",
            "x_head",
            "-",
            "",
            "",
            e,
            tol,
            if ok { "OK" } else { "XX" }
        );
    }

    free_dspark_heads_acts(&mut gpu, acts)?;
    gpu.free_tensor(d_x_head)?;

    if all_ok {
        println!("\n  PASS — DSpark drafter heads backward matches finite differences");
        Ok(())
    } else {
        Err("DSpark drafter heads gradcheck FAILED".into())
    }
}
