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

//! Finite-difference gradcheck for the DSpark training loss (T3).
//!
//! Verifies the two analytic gradients against central differences:
//!
//! * `d_draft_logits` — checked against the finite difference of the
//!   `ce_alpha·ce + l1_alpha·l1` PARTIAL loss (the confidence term's dependence
//!   on the draft logits flows only through the DETACHED accept-rate target, so
//!   it contributes no draft gradient; excluding it is what "detached" means).
//! * `d_confidence_logit` — checked against the finite difference of the TOTAL
//!   loss (only the confidence term depends on `confidence_logit`).
//!
//! DO NOT run on nix2 (LDS hazard); this is compile-gated by the build gate.
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire lock acquire "gradcheck-dspark-loss"
//!   cargo run -p hipfire-train --release --example gradcheck_dspark_loss

use hipfire_rdna::{Gpu, GpuTensor, HipResult};
use hipfire_train::dspark_loss::{dspark_loss_forward_backward, DsparkLossCfg};

const N_BLOCKS: usize = 2;
const BLOCK: usize = 2;
const ROWS: usize = N_BLOCKS * BLOCK;
const V: usize = 5;

fn cfg() -> DsparkLossCfg {
    DsparkLossCfg::with_block_size(BLOCK)
}

/// Partial loss `ce_alpha·ce + l1_alpha·l1` (the draft-differentiable part).
fn partial_draft_loss(
    gpu: &mut Gpu,
    draft: &GpuTensor,
    conf: &GpuTensor,
    tgt_logits: &GpuTensor,
    tgt_tok: &GpuTensor,
    mask: &GpuTensor,
) -> HipResult<f32> {
    let c = cfg();
    let out = dspark_loss_forward_backward(gpu, draft, conf, tgt_logits, tgt_tok, mask, &c)?;
    Ok(c.ce_alpha * out.ce + c.l1_alpha * out.l1)
}

/// Full total loss (for the confidence-logit direction).
fn total_loss(
    gpu: &mut Gpu,
    draft: &GpuTensor,
    conf: &GpuTensor,
    tgt_logits: &GpuTensor,
    tgt_tok: &GpuTensor,
    mask: &GpuTensor,
) -> HipResult<f32> {
    let out = dspark_loss_forward_backward(gpu, draft, conf, tgt_logits, tgt_tok, mask, &cfg())?;
    Ok(out.total)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    // Deterministic pseudo-random-ish inputs.
    let draft_host: Vec<f32> = (0..ROWS * V)
        .map(|i| ((i * 31 % 13) as f32) * 0.3 - 1.7)
        .collect();
    let tgt_logits_host: Vec<f32> = (0..ROWS * V)
        .map(|i| ((i * 17 % 11) as f32) * 0.25 - 1.1)
        .collect();
    let conf_host: Vec<f32> = (0..ROWS).map(|i| (i as f32) * 0.4 - 0.7).collect();
    let tgt_tok_host: Vec<f32> = vec![2.0, 0.0, 4.0, 1.0];
    // Row 3 masked out to exercise the weight mask (its grad must be zero).
    let mask_host: Vec<f32> = vec![1.0, 1.0, 1.0, 0.0];

    let draft = gpu.upload_f32(&draft_host, &[ROWS, V])?;
    let tgt_logits = gpu.upload_f32(&tgt_logits_host, &[ROWS, V])?;
    let conf = gpu.upload_f32(&conf_host, &[ROWS])?;
    let tgt_tok = gpu.upload_f32(&tgt_tok_host, &[ROWS])?;
    let mask = gpu.upload_f32(&mask_host, &[ROWS])?;

    let out = dspark_loss_forward_backward(
        &mut gpu,
        &draft,
        &conf,
        &tgt_logits,
        &tgt_tok,
        &mask,
        &cfg(),
    )?;
    println!(
        "total={:.6}  ce={:.6}  l1={:.6}  conf={:.6}",
        out.total, out.ce, out.l1, out.conf
    );
    let d_draft = gpu.download_f32(&out.d_draft_logits)?;
    let d_conf = gpu.download_f32(&out.d_confidence_logit)?;

    // Masked row (3) must have exactly-zero gradients.
    if d_draft[3 * V..4 * V].iter().any(|x| *x != 0.0) {
        return Err("masked row draft grad not all zero".into());
    }
    if d_conf[3] != 0.0 {
        return Err("masked row confidence grad not zero".into());
    }

    let eps = 1e-3f32;

    // ---- d_draft_logits vs finite diff of the partial (ce+l1) loss ----------
    let mut max_err_draft = 0.0f32;
    for i in 0..ROWS * V {
        let mut lp = draft_host.clone();
        lp[i] += eps;
        let dp = gpu.upload_f32(&lp, &[ROWS, V])?;
        let hp = partial_draft_loss(&mut gpu, &dp, &conf, &tgt_logits, &tgt_tok, &mask)?;
        let mut lm = draft_host.clone();
        lm[i] -= eps;
        let dm = gpu.upload_f32(&lm, &[ROWS, V])?;
        let hm = partial_draft_loss(&mut gpu, &dm, &conf, &tgt_logits, &tgt_tok, &mask)?;
        max_err_draft = max_err_draft.max(((hp - hm) / (2.0 * eps) - d_draft[i]).abs());
    }

    // ---- d_confidence_logit vs finite diff of the total loss ----------------
    let mut max_err_conf = 0.0f32;
    for i in 0..ROWS {
        let mut cp = conf_host.clone();
        cp[i] += eps;
        let cpp = gpu.upload_f32(&cp, &[ROWS])?;
        let hp = total_loss(&mut gpu, &draft, &cpp, &tgt_logits, &tgt_tok, &mask)?;
        let mut cm = conf_host.clone();
        cm[i] -= eps;
        let cmm = gpu.upload_f32(&cm, &[ROWS])?;
        let hm = total_loss(&mut gpu, &draft, &cmm, &tgt_logits, &tgt_tok, &mask)?;
        max_err_conf = max_err_conf.max(((hp - hm) / (2.0 * eps) - d_conf[i]).abs());
    }

    println!("d_draft_logits     max|analytic-numeric| = {max_err_draft:.2e}");
    println!("d_confidence_logit max|analytic-numeric| = {max_err_conf:.2e}");

    let tol = 1e-2f32;
    if max_err_draft < tol && max_err_conf < tol {
        println!("\nGRADCHECK PASS — dspark_loss backward matches finite differences.");
        Ok(())
    } else {
        Err(format!(
            "gradcheck FAIL (tol {tol:.0e}): draft={max_err_draft:.2e} conf={max_err_conf:.2e}"
        )
        .into())
    }
}
