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

//! Smoke test: load Supra-50M into fp32 GPU tensors and report.
//!
//! First half of M0 (docs/plans/2026-06-17-hipfire-train-phase0.md). Verifies
//! the config parse, the bf16→f32 conversion, every named tensor, and the GPU
//! upload — before any forward/backward is wired.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "train-load"
//!   cargo run -p hipfire-train --release --example train_load_supra50m
//!   hipfire gpu-lock release

use hipfire_train::loader::load_llama_fp32;
use hipfire_rdna::Gpu;
use std::path::Path;

const MODEL_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new(MODEL_DIR);
    if !dir.exists() {
        return Err(format!("model dir not found: {MODEL_DIR}").into());
    }

    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let (cfg, w) = load_llama_fp32(&mut gpu, dir)?;
    println!("config: {cfg:?}");
    println!(
        "loaded: embed_tokens {:?}, {} layers, final_norm {:?}, lm_head tied={}",
        w.embed_tokens.shape,
        w.layers.len(),
        w.final_norm.shape,
        w.lm_head.is_none()
    );

    // Spot-check: pull a few values back and confirm they're finite & non-zero.
    let l0 = &w.layers[0];
    let q = gpu.download_f32(&l0.q_proj)?;
    let n_finite = q.iter().filter(|x| x.is_finite()).count();
    let n_nonzero = q.iter().filter(|x| **x != 0.0).count();
    println!(
        "layer0.q_proj: {} elems, {} finite, {} nonzero, first={:.5}",
        q.len(),
        n_finite,
        n_nonzero,
        q[0]
    );
    if n_finite != q.len() {
        return Err("layer0.q_proj has non-finite values".into());
    }
    if n_nonzero == 0 {
        return Err("layer0.q_proj is all zeros — conversion likely broken".into());
    }

    println!("\nOK — Supra-50M loaded into fp32 GPU tensors.");
    Ok(())
}
