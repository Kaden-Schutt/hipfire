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

//! Finite-difference gradient check for masked/bidirectional SDPA — the DSpark
//! drafter's block-attention primitive (bidirectional over
//! `[context_KV ++ block_KV]`). Exercises the rectangular `seq_q != seq_k` path
//! with an additive bias, in two configurations:
//!   1. `bias = None`   — fully bidirectional (all keys attended).
//!   2. `bias = Some(m)` — partial 0/−inf mask (some keys dropped per query).
//!
//! Loss L = Σ CTX∘G ⇒ d_ctx = G. Checks analytic dQ, dK, dV vs central
//! differences.
//!
//! Run (LDS-wedge hazard on gfx1103 — do NOT run on nix2):
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire lock acquire "gradcheck-dspark-attn"
//!   cargo run -p hipfire-train --release --example gradcheck_dspark_attn
//!   hipfire lock release

use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};
use hipfire_train::ops::attention::{sdpa_backward_masked, sdpa_forward_masked};

const SEQ_Q: usize = 4; // block length (queries)
const SEQ_K: usize = 6; // context ++ block (keys)  → rectangular, seq_q != seq_k
const D: usize = 8;

fn scale() -> f32 {
    1.0 / (D as f32).sqrt()
}

/// Build a partial additive mask `[SEQ_Q*SEQ_K]`: query row `i` may not attend
/// key `j` when `(i + j) % 3 == 0` (arbitrary, deterministic). 0 keeps, −inf
/// drops. At least one key per row stays open so softmax is well-defined.
fn partial_mask() -> Vec<f32> {
    let mut m = vec![0.0f32; SEQ_Q * SEQ_K];
    for i in 0..SEQ_Q {
        for j in 0..SEQ_K {
            // keep j==i (diagonal-ish) always open; drop a deterministic subset
            if j != i && (i + j) % 3 == 0 {
                m[i * SEQ_K + j] = f32::NEG_INFINITY;
            }
        }
    }
    m
}

fn loss(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    bias: Option<&GpuTensor>,
    g: &[f32],
) -> HipResult<f32> {
    let scores = gpu.zeros(&[SEQ_Q * SEQ_K], DType::F32)?;
    let p = gpu.zeros(&[SEQ_Q * SEQ_K], DType::F32)?;
    let ctx = gpu.zeros(&[SEQ_Q * D], DType::F32)?;
    sdpa_forward_masked(
        gpu,
        q,
        k,
        v,
        &scores,
        &p,
        &ctx,
        SEQ_Q,
        SEQ_K,
        D,
        scale(),
        bias,
    )?;
    let cv = gpu.download_f32(&ctx)?;
    Ok(cv.iter().zip(g).map(|(a, b)| a * b).sum())
}

fn run_case(
    gpu: &mut Gpu,
    label: &str,
    qh: &[f32],
    kh: &[f32],
    vh: &[f32],
    gh: &[f32],
    bias_host: Option<&[f32]>,
) -> HipResult<f32> {
    let q = gpu.upload_f32(qh, &[SEQ_Q * D])?;
    let k = gpu.upload_f32(kh, &[SEQ_K * D])?;
    let v = gpu.upload_f32(vh, &[SEQ_K * D])?;
    let bias_t = match bias_host {
        Some(b) => Some(gpu.upload_f32(b, &[SEQ_Q * SEQ_K])?),
        None => None,
    };
    let bias = bias_t.as_ref();

    // Analytic
    let scores = gpu.zeros(&[SEQ_Q * SEQ_K], DType::F32)?;
    let p = gpu.zeros(&[SEQ_Q * SEQ_K], DType::F32)?;
    let ctx = gpu.zeros(&[SEQ_Q * D], DType::F32)?;
    sdpa_forward_masked(
        gpu,
        &q,
        &k,
        &v,
        &scores,
        &p,
        &ctx,
        SEQ_Q,
        SEQ_K,
        D,
        scale(),
        bias,
    )?;
    let d_ctx = gpu.upload_f32(gh, &[SEQ_Q * D])?;
    let dp = gpu.zeros(&[SEQ_Q * SEQ_K], DType::F32)?;
    let dsc = gpu.zeros(&[SEQ_Q * SEQ_K], DType::F32)?;
    let dq = gpu.zeros(&[SEQ_Q * D], DType::F32)?;
    let dk = gpu.zeros(&[SEQ_K * D], DType::F32)?;
    let dv = gpu.zeros(&[SEQ_K * D], DType::F32)?;
    sdpa_backward_masked(
        gpu,
        &d_ctx,
        &q,
        &k,
        &v,
        &p,
        &dp,
        &dsc,
        &dq,
        &dk,
        &dv,
        SEQ_Q,
        SEQ_K,
        D,
        scale(),
    )?;
    let dq_a = gpu.download_f32(&dq)?;
    let dk_a = gpu.download_f32(&dk)?;
    let dv_a = gpu.download_f32(&dv)?;

    let eps = 1e-3f32;
    // which: 0 = perturb Q, 1 = perturb K, 2 = perturb V
    let check = |gpu: &mut Gpu, host: &[f32], which: u8, ana: &[f32]| -> HipResult<f32> {
        let mut e = 0.0f32;
        for i in 0..host.len() {
            let mut hp = host.to_vec();
            hp[i] += eps;
            let mut hm = host.to_vec();
            hm[i] -= eps;
            let n = host.len();
            let pd = gpu.upload_f32(&hp, &[n])?;
            let md = gpu.upload_f32(&hm, &[n])?;
            let bt = match bias_host {
                Some(b) => Some(gpu.upload_f32(b, &[SEQ_Q * SEQ_K])?),
                None => None,
            };
            let bref = bt.as_ref();
            let (lp, lm) = match which {
                0 => (
                    loss(gpu, &pd, &k, &v, bref, gh)?,
                    loss(gpu, &md, &k, &v, bref, gh)?,
                ),
                1 => (
                    loss(gpu, &q, &pd, &v, bref, gh)?,
                    loss(gpu, &q, &md, &v, bref, gh)?,
                ),
                _ => (
                    loss(gpu, &q, &k, &pd, bref, gh)?,
                    loss(gpu, &q, &k, &md, bref, gh)?,
                ),
            };
            e = e.max(((lp - lm) / (2.0 * eps) - ana[i]).abs());
        }
        Ok(e)
    };

    let eq = check(gpu, qh, 0, &dq_a)?;
    let ek = check(gpu, kh, 1, &dk_a)?;
    let ev = check(gpu, vh, 2, &dv_a)?;

    println!("[{label}] dQ={eq:.2e} dK={ek:.2e} dV={ev:.2e}");
    Ok(eq.max(ek).max(ev))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let qh: Vec<f32> = (0..SEQ_Q * D)
        .map(|i| ((i * 17 % 13) as f32) * 0.15 - 0.8)
        .collect();
    let kh: Vec<f32> = (0..SEQ_K * D)
        .map(|i| ((i * 23 % 11) as f32) * 0.12 - 0.5)
        .collect();
    let vh: Vec<f32> = (0..SEQ_K * D)
        .map(|i| ((i * 7 % 9) as f32) * 0.2 - 0.7)
        .collect();
    let gh: Vec<f32> = (0..SEQ_Q * D)
        .map(|i| ((i * 13 % 5) as f32) * 0.25 - 0.4)
        .collect();

    // Case 1: fully bidirectional (bias = None).
    let e_bidir = run_case(&mut gpu, "bidirectional", &qh, &kh, &vh, &gh, None)?;
    // Case 2: partial 0/−inf mask.
    let mask = partial_mask();
    let e_masked = run_case(&mut gpu, "partial-mask", &qh, &kh, &vh, &gh, Some(&mask))?;

    let tol = 1e-2f32;
    if e_bidir < tol && e_masked < tol {
        println!(
            "\nGRADCHECK PASS — masked/bidirectional SDPA backward matches finite differences \
             (seq_q={SEQ_Q}, seq_k={SEQ_K})."
        );
        Ok(())
    } else {
        Err(format!("gradcheck FAIL: bidirectional {e_bidir:.2e}, masked {e_masked:.2e}").into())
    }
}
