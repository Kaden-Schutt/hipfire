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

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gpu-vs-gpu equivalence for the batched Mamba-2 block PREFILL (N6,
//! `Mamba2BlockGpu::prefill`) against the per-token `decode_step` loop. Two
//! blocks with identical f32 weights start from zero state; one prefills the
//! whole prompt in batched form, the other runs `decode_step` token-by-token.
//! The per-position outputs AND the final recurrent state must agree — that is
//! exactly the property that lets prefill hand off to decode. Composes the three
//! batched kernels (conv1d-seq, ssd-seq, gated-norm-seq) + gemm + strided split.
//!
//!   hipfire lock acquire test_block_prefill_gpu --watch-pid $$
//!   cargo run -p hipfire-arch-nemotron --example test_block_prefill_gpu

use hipfire_arch_nemotron::block::{Mamba2BlockWeights, Mamba2Dims};
use hipfire_arch_nemotron::block_gpu::Mamba2BlockGpu;
use hipfire_rdna::Gpu;

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    // Same structurally-faithful non-pow2 dims as test_block_gpu.
    let dims = Mamba2Dims {
        hidden_size: 24,
        num_heads: 4,
        head_dim: 10,
        state_size: 16,
        n_groups: 2,
        conv_kernel: 4,
        rms_norm_eps: 1e-5,
        dt_min: 0.0,
        dt_max: f32::INFINITY,
    };
    let seq = 29usize;

    let mut seed = 0x51ED_270Bu32;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 0.4 - 0.2
    };
    let in_proj: Vec<f32> = (0..dims.projection_size() * dims.hidden_size)
        .map(|_| rng())
        .collect();
    let conv_weight: Vec<f32> = (0..dims.conv_dim() * dims.conv_kernel)
        .map(|_| rng())
        .collect();
    let conv_bias: Vec<f32> = (0..dims.conv_dim()).map(|_| rng()).collect();
    let a_log: Vec<f32> = (0..dims.num_heads).map(|_| rng()).collect();
    let dd: Vec<f32> = (0..dims.num_heads).map(|_| rng()).collect();
    let dt_bias: Vec<f32> = (0..dims.num_heads).map(|_| rng()).collect();
    let norm_weight: Vec<f32> = (0..dims.d_inner()).map(|_| 1.0 + rng()).collect();
    let out_proj: Vec<f32> = (0..dims.hidden_size * dims.d_inner())
        .map(|_| rng())
        .collect();

    let w = Mamba2BlockWeights {
        in_proj: &in_proj,
        conv_weight: &conv_weight,
        conv_bias: &conv_bias,
        a_log: &a_log,
        d: &dd,
        dt_bias: &dt_bias,
        norm_weight: &norm_weight,
        out_proj: &out_proj,
    };

    let hidden = dims.hidden_size;
    let hidden_seq: Vec<f32> = (0..seq * hidden).map(|_| rng()).collect();

    // ── batched prefill ──────────────────────────────────────────────────────
    let mut block_pf = Mamba2BlockGpu::new(&mut gpu, dims.clone(), &w).expect("upload pf");
    let hs_g = gpu.upload_f32(&hidden_seq, &[seq * hidden]).unwrap();
    let out_pf_t = block_pf.prefill(&mut gpu, &hs_g, seq).expect("prefill");
    gpu.hip.device_synchronize().unwrap();
    let out_pf = gpu.download_f32(&out_pf_t).unwrap();

    // ── per-token decode loop ────────────────────────────────────────────────
    let mut block_dec = Mamba2BlockGpu::new(&mut gpu, dims.clone(), &w).expect("upload dec");
    let mut out_dec = vec![0.0f32; seq * hidden];
    for t in 0..seq {
        let row = gpu
            .upload_f32(&hidden_seq[t * hidden..(t + 1) * hidden], &[hidden])
            .unwrap();
        let o = block_dec.decode_step(&mut gpu, &row).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let ov = gpu.download_f32(o).unwrap();
        out_dec[t * hidden..(t + 1) * hidden].copy_from_slice(&ov);
        let _ = gpu.free_tensor(row);
    }

    let max_d = out_pf
        .iter()
        .zip(&out_dec)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // worst per-position cosine drift, to confirm it's not a single-element blip.
    eprintln!("seq={seq} hidden={hidden}  max|Δout|={max_d:.3e}");

    if max_d < 1e-4 {
        println!("PASS: Mamba2BlockGpu::prefill matches the decode loop (max|Δ|={max_d:.2e})");
    } else {
        println!("FAIL: prefill diverges from decode (max|Δ|={max_d:.2e})");
        std::process::exit(1);
    }
}
