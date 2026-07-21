// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

#![cfg(feature = "deltanet")]

//! Numerical smoke coverage for the canonical DeltaNet dispatch contract.
//! These tests are intentionally ignored: they require an AMD GPU and are
//! supplementary to the GPU-free route tests in `ops::delta_net`.

use hipfire_dispatch::ops::delta_net::{
    DeltaNetBatchParams, DeltaNetOps, DeltaNetTreeParams, StateQuant,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const HD: usize = 128;
const HEADS: usize = 1;
const TOKENS: usize = 3;

fn inputs(gpu: &mut Gpu) -> (GpuTensor, GpuTensor, GpuTensor, GpuTensor, GpuTensor) {
    let width = HEADS * HD;
    let q: Vec<f32> = (0..TOKENS * width)
        .map(|i| ((i * 17 % 101) as f32 - 50.0) * 0.0007)
        .collect();
    let k: Vec<f32> = (0..TOKENS * width)
        .map(|i| ((i * 29 % 113) as f32 - 56.0) * 0.0006)
        .collect();
    let v: Vec<f32> = (0..TOKENS * width)
        .map(|i| ((i * 43 % 127) as f32 - 63.0) * 0.0011)
        .collect();
    let gate = vec![-0.5f32; TOKENS * HEADS];
    let beta = vec![0.25f32; TOKENS * HEADS];
    (
        gpu.upload_f32(&q, &[TOKENS, width]).unwrap(),
        gpu.upload_f32(&k, &[TOKENS, width]).unwrap(),
        gpu.upload_f32(&v, &[TOKENS, width]).unwrap(),
        gpu.upload_f32(&gate, &[TOKENS, HEADS]).unwrap(),
        gpu.upload_f32(&beta, &[TOKENS, HEADS]).unwrap(),
    )
}

fn parents(gpu: &mut Gpu) -> GpuTensor {
    let values = [-1i32, 0, 1];
    let tensor = gpu.alloc_tensor(&[values.len() * 4], DType::Raw).unwrap();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) };
    gpu.hip.memcpy_htod(&tensor.buf, bytes).unwrap();
    tensor
}

fn assert_finite(gpu: &mut Gpu, tensor: &GpuTensor) {
    assert!(gpu
        .download_f32(tensor)
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
}

fn download_bytes(gpu: &mut Gpu, tensor: &GpuTensor) -> Vec<u8> {
    let mut bytes = vec![0u8; tensor.buf.size()];
    gpu.hip.memcpy_dtoh(&mut bytes, &tensor.buf).unwrap();
    bytes
}

fn clone_tensor(gpu: &mut Gpu, source: &GpuTensor, shape: &[usize], dtype: DType) -> GpuTensor {
    let clone = gpu.alloc_tensor(shape, dtype).unwrap();
    gpu.hip
        .memcpy_dtod(&clone.buf, &source.buf, source.buf.size())
        .unwrap();
    clone
}

#[test]
#[ignore = "requires an AMD GPU"]
fn canonical_dispatch_preserves_fp32_batch_tree_order_and_q8_ef() {
    let mut gpu = Gpu::init().expect("GPU init");
    let (q, k, v, gate, beta) = inputs(&mut gpu);
    let state = gpu.zeros(&[HEADS * HD * HD], DType::F32).unwrap();
    let output = gpu.zeros(&[TOKENS, HEADS * HD], DType::F32).unwrap();
    let batch = DeltaNetBatchParams {
        q_batch: &q,
        k_batch: &k,
        v_batch: &v,
        gate_batch: &gate,
        beta_batch: &beta,
        state: &state,
        s_scales: &state,
        output_batch: &output,
        ef_residual: None,
        n_tokens: TOKENS,
        n_heads: HEADS,
        head_dim: HD,
        quant: StateQuant::FP32,
    };
    ().run_delta_net_batch(&mut gpu, &batch).unwrap();
    let batch_output = gpu.download_f32(&output).unwrap();

    let tree_state = gpu.zeros(&[HEADS * HD * HD], DType::F32).unwrap();
    let tape = gpu.zeros(&[TOKENS * HEADS * HD * HD], DType::F32).unwrap();
    let tree_output = gpu.zeros(&[TOKENS, HEADS * HD], DType::F32).unwrap();
    let parent_indices = parents(&mut gpu);
    let tree = DeltaNetTreeParams::fp32(
        &q,
        &k,
        &v,
        &gate,
        &beta,
        &tree_state,
        &tape,
        &parent_indices,
        &tree_output,
        TOKENS,
        HEADS,
        HD,
    );
    ().run_delta_net_tree(&mut gpu, &tree).unwrap();
    let tree_output_host = gpu.download_f32(&tree_output).unwrap();
    assert!(
        batch_output
            .iter()
            .zip(tree_output_host.iter())
            .all(|(a, b)| (a - b).abs() < 1e-3),
        "FP32 batch/tree spine output ordering diverged"
    );

    let q8_state = gpu.zeros(&[HEADS * HD * HD], DType::Raw).unwrap();
    let q8_scales = gpu.zeros(&[HEADS * HD], DType::F32).unwrap();
    let ef_residual = gpu.zeros(&[HEADS * HD * HD], DType::F16).unwrap();
    let q8_output = gpu.zeros(&[TOKENS, HEADS * HD], DType::F32).unwrap();
    let batch_ef_before = download_bytes(&mut gpu, &ef_residual);
    let q8_batch = DeltaNetBatchParams {
        q_batch: &q,
        k_batch: &k,
        v_batch: &v,
        gate_batch: &gate,
        beta_batch: &beta,
        state: &q8_state,
        s_scales: &q8_scales,
        output_batch: &q8_output,
        ef_residual: Some(&ef_residual),
        n_tokens: TOKENS,
        n_heads: HEADS,
        head_dim: HD,
        quant: StateQuant::Q8,
    };
    ().run_delta_net_batch(&mut gpu, &q8_batch).unwrap();
    assert_finite(&mut gpu, &q8_output);
    let batch_ef_after = download_bytes(&mut gpu, &ef_residual);
    assert_ne!(
        batch_ef_after, batch_ef_before,
        "Q8 batch EF residual was not forwarded or written"
    );

    // Causal two-branch check: launch 1 establishes one post-launch state,
    // which is cloned before launch 2. Both branches remain EF-enabled; only
    // branch B has its carried residual zeroed. Compare post-launch-2 state,
    // scales, and residual rather than launch-2 output (EF is folded during
    // requant after that output is produced).
    let ef_state = gpu.zeros(&[HEADS * HD * HD], DType::Raw).unwrap();
    let ef_scales = gpu
        .upload_f32(&vec![1.0f32; HEADS * HD], &[HEADS * HD])
        .unwrap();
    let ef_residual_step = gpu.zeros(&[HEADS * HD * HD], DType::F16).unwrap();
    let ef_initial = download_bytes(&mut gpu, &ef_residual_step);

    let q1 = q.sub_offset(0, HEADS * HD);
    let k1 = k.sub_offset(0, HEADS * HD);
    let v1 = v.sub_offset(0, HEADS * HD);
    let gate1 = gate.sub_offset(0, HEADS);
    let beta1 = beta.sub_offset(0, HEADS);
    let ef_output1 = gpu.zeros(&[1, HEADS * HD], DType::F32).unwrap();
    let ef_step1 = hipfire_dispatch::ops::delta_net::DeltaNetStepParams {
        q: &q1,
        k: &k1,
        v: &v1,
        gate: &gate1,
        beta: &beta1,
        state: &ef_state,
        s_scales: &ef_scales,
        output: &ef_output1,
        ef_residual: Some(&ef_residual_step),
        n_heads: HEADS,
        head_dim: HD,
        quant: StateQuant::Q8,
    };
    ().run_delta_net_step(&mut gpu, &ef_step1).unwrap();
    let ef_after_first = download_bytes(&mut gpu, &ef_residual_step);
    assert_ne!(ef_after_first, ef_initial, "Q8 EF residual was not written");

    let branch_state = clone_tensor(&mut gpu, &ef_state, &[HEADS * HD * HD], DType::Raw);
    let branch_scales = clone_tensor(&mut gpu, &ef_scales, &[HEADS * HD], DType::F32);
    let zeroed_residual = gpu.zeros(&[HEADS * HD * HD], DType::F16).unwrap();

    let q2 = q.sub_offset(HEADS * HD, HEADS * HD);
    let k2 = k.sub_offset(HEADS * HD, HEADS * HD);
    let v2 = v.sub_offset(HEADS * HD, HEADS * HD);
    let gate2 = gate.sub_offset(HEADS, HEADS);
    let beta2 = beta.sub_offset(HEADS, HEADS);
    let ef_output2 = gpu.zeros(&[1, HEADS * HD], DType::F32).unwrap();
    let zeroed_output2 = gpu.zeros(&[1, HEADS * HD], DType::F32).unwrap();
    let ef_step2 = hipfire_dispatch::ops::delta_net::DeltaNetStepParams {
        q: &q2,
        k: &k2,
        v: &v2,
        gate: &gate2,
        beta: &beta2,
        state: &ef_state,
        s_scales: &ef_scales,
        output: &ef_output2,
        ef_residual: Some(&ef_residual_step),
        n_heads: HEADS,
        head_dim: HD,
        quant: StateQuant::Q8,
    };
    let zeroed_step2 = hipfire_dispatch::ops::delta_net::DeltaNetStepParams {
        q: &q2,
        k: &k2,
        v: &v2,
        gate: &gate2,
        beta: &beta2,
        state: &branch_state,
        s_scales: &branch_scales,
        output: &zeroed_output2,
        ef_residual: Some(&zeroed_residual),
        n_heads: HEADS,
        head_dim: HD,
        quant: StateQuant::Q8,
    };
    ().run_delta_net_step(&mut gpu, &ef_step2).unwrap();
    ().run_delta_net_step(&mut gpu, &zeroed_step2).unwrap();
    let preserved_state = download_bytes(&mut gpu, &ef_state);
    let zeroed_state = download_bytes(&mut gpu, &branch_state);
    let preserved_scales = download_bytes(&mut gpu, &ef_scales);
    let zeroed_scales = download_bytes(&mut gpu, &branch_scales);
    let preserved_residual = download_bytes(&mut gpu, &ef_residual_step);
    let zeroed_residual_bytes = download_bytes(&mut gpu, &zeroed_residual);
    assert!(
        preserved_state != zeroed_state
            || preserved_scales != zeroed_scales
            || preserved_residual != zeroed_residual_bytes,
        "preserved Q8 EF residual did not affect launch-2 state"
    );

    let q8_tree_state = gpu.zeros(&[HEADS * HD * HD], DType::Raw).unwrap();
    let q8_tree_scales = gpu
        .upload_f32(&vec![1.0f32; HEADS * HD], &[HEADS * HD])
        .unwrap();
    let q8_tape = gpu.zeros(&[TOKENS * HEADS * HD * HD], DType::Raw).unwrap();
    let q8_tape_scales = gpu.zeros(&[TOKENS * HEADS * HD], DType::F32).unwrap();
    let q8_tree_output = gpu.zeros(&[TOKENS, HEADS * HD], DType::F32).unwrap();
    let q8_parents = parents(&mut gpu);
    let q8_tree = DeltaNetTreeParams::q8(
        &q,
        &k,
        &v,
        &gate,
        &beta,
        &q8_tree_state,
        &q8_tree_scales,
        &q8_tape,
        &q8_tape_scales,
        &q8_parents,
        &q8_tree_output,
        TOKENS,
        HEADS,
        HD,
    );
    ().run_delta_net_tree(&mut gpu, &q8_tree).unwrap();
    assert_finite(&mut gpu, &q8_tree_output);
}

#[test]
#[ignore = "requires an AMD GPU"]
fn canonical_dispatch_preserves_q4_n_token_output_order() {
    let mut gpu = Gpu::init().expect("GPU init");
    let (q, k, v, gate, beta) = inputs(&mut gpu);
    let state = gpu.zeros(&[HEADS * HD * (HD / 2)], DType::Raw).unwrap();
    let scales = gpu.zeros(&[HEADS * HD], DType::F32).unwrap();
    let output = gpu.zeros(&[TOKENS, HEADS * HD], DType::F32).unwrap();
    let batch = DeltaNetBatchParams {
        q_batch: &q,
        k_batch: &k,
        v_batch: &v,
        gate_batch: &gate,
        beta_batch: &beta,
        state: &state,
        s_scales: &scales,
        output_batch: &output,
        ef_residual: None,
        n_tokens: TOKENS,
        n_heads: HEADS,
        head_dim: HD,
        quant: StateQuant::Q4,
    };
    ().run_delta_net_batch(&mut gpu, &batch).unwrap();
    let batch_output = gpu.download_f32(&output).unwrap();

    // The canonical Q4 N-token launch must have the same row ordering as
    // repeated single-token decode calls over the same caller-owned state.
    let reference_state = gpu.zeros(&[HEADS * HD * (HD / 2)], DType::Raw).unwrap();
    let reference_scales = gpu.zeros(&[HEADS * HD], DType::F32).unwrap();
    let reference_output = gpu.zeros(&[TOKENS, HEADS * HD], DType::F32).unwrap();
    for token in 0..TOKENS {
        let q1 = q.sub_offset(token * HEADS * HD, HEADS * HD);
        let k1 = k.sub_offset(token * HEADS * HD, HEADS * HD);
        let v1 = v.sub_offset(token * HEADS * HD, HEADS * HD);
        let gate1 = gate.sub_offset(token * HEADS, HEADS);
        let beta1 = beta.sub_offset(token * HEADS, HEADS);
        let output1 = reference_output.sub_offset(token * HEADS * HD, HEADS * HD);
        let step = hipfire_dispatch::ops::delta_net::DeltaNetStepParams {
            q: &q1,
            k: &k1,
            v: &v1,
            gate: &gate1,
            beta: &beta1,
            state: &reference_state,
            s_scales: &reference_scales,
            output: &output1,
            ef_residual: None,
            n_heads: HEADS,
            head_dim: HD,
            quant: StateQuant::Q4,
        };
        ().run_delta_net_step(&mut gpu, &step).unwrap();
    }
    let reference_output_host = gpu.download_f32(&reference_output).unwrap();
    assert!(
        batch_output
            .iter()
            .zip(reference_output_host.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5),
        "Q4 N-token output ordering diverged from sequential decode"
    );
}
