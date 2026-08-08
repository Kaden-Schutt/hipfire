// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU gate for the Step DeltaNet lowering.
//!
//! This deliberately compares the old raw oracle with the Step path in fresh
//! state/cache pairs.  Keep it ignored: it needs an AMD GPU and the canonical
//! local qwen3.5-0.8b fixture.

#![cfg(feature = "test-utils")]

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch, StateQuant};
use hipfire_arch_qwen35::speculative::GdnTape;
use hipfire_arch_qwen35::test_utils::{legacy_delta_net_sequence, with_raw_delta_net_scope};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::{Path, PathBuf};

const TOKENS: [u32; 4] = [1, 42, 314, 271];

#[derive(Clone, Debug, PartialEq, Eq)]
struct StateBytes {
    // The persistent DeltaNet state; KV cache storage is outside this seam.
    dn_s: Vec<Vec<u8>>,
    dn_scales: Vec<Vec<u8>>,
    dn_conv: Vec<Vec<u8>>,
    dn_ef: Vec<Vec<u8>>,
}

fn bytes(gpu: &Gpu, tensor: &GpuTensor) -> Vec<u8> {
    let mut out = vec![0u8; tensor.byte_size()];
    gpu.hip
        .memcpy_dtoh(&mut out, &tensor.buf)
        .expect("D2H snapshot");
    out
}

fn state_bytes(gpu: &Gpu, dn: &DeltaNetState) -> StateBytes {
    gpu.hip.device_synchronize().expect("state synchronize");
    let state_bytes = |tensor: &GpuTensor| {
        let nbytes = match dn.quant {
            StateQuant::FP32 => tensor.byte_size(),
            StateQuant::Q8 => tensor.numel(),
            StateQuant::Q4 => tensor.numel() / 2,
        };
        let mut out = vec![0u8; nbytes];
        gpu.hip
            .memcpy_dtoh(&mut out, &tensor.buf)
            .expect("D2H state snapshot");
        out
    };
    StateBytes {
        dn_s: dn.s_matrices.iter().map(state_bytes).collect(),
        dn_scales: dn.s_scales.iter().map(|t| bytes(gpu, t)).collect(),
        dn_conv: dn.conv_states.iter().map(|t| bytes(gpu, t)).collect(),
        dn_ef: dn.s_ef_residual.iter().map(|t| bytes(gpu, t)).collect(),
    }
}

fn assert_tensor_bytes(gpu: &Gpu, label: &str, lhs: &GpuTensor, rhs: &GpuTensor) {
    assert_eq!(bytes(gpu, lhs), bytes(gpu, rhs), "{label} differs");
}

fn assert_state_bytes(label: &str, lhs: StateBytes, rhs: StateBytes) {
    macro_rules! compare {
        ($name:literal, $left:expr, $right:expr) => {
            if $left != $right {
                let index = $left
                    .iter()
                    .zip(&$right)
                    .position(|(a, b)| a != b)
                    .unwrap_or(usize::MAX);
                let byte = if index == usize::MAX {
                    usize::MAX
                } else {
                    $left[index]
                        .iter()
                        .zip(&$right[index])
                        .position(|(a, b)| a != b)
                        .unwrap_or(usize::MAX)
                };
                panic!("{label} {} differs at tensor {index}, byte {byte}", $name);
            }
        };
    }
    compare!("dn_s", lhs.dn_s, rhs.dn_s);
    compare!("dn_scales", lhs.dn_scales, rhs.dn_scales);
    compare!("dn_conv", lhs.dn_conv, rhs.dn_conv);
    compare!("dn_ef", lhs.dn_ef, rhs.dn_ef);
}

fn fixture_path() -> PathBuf {
    std::env::var_os("HIPFIRE_DN_PARITY_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").expect("HOME"))
                .join(".hipfire/models/qwen3.5-0.8b.mq4")
        })
}

fn upload_i32(gpu: &Gpu, tensor: &GpuTensor, values: &[i32]) {
    assert_eq!(tensor.dtype, DType::Raw, "parent buffer must be raw bytes");
    let required = values.len() * std::mem::size_of::<i32>();
    assert!(
        tensor.buf.size() >= required,
        "parent buffer capacity {} < required {required}",
        tensor.buf.size()
    );
    assert!(
        tensor.byte_size() >= required,
        "parent tensor byte size {} < required {required}",
        tensor.byte_size()
    );
    let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, required) };
    gpu.hip
        .memcpy_htod(&tensor.buf, bytes)
        .expect("H2D i32 upload");
}

fn upload_f32(gpu: &Gpu, tensor: &GpuTensor, values: &[f32]) {
    assert_eq!(tensor.numel(), values.len(), "F32 upload shape mismatch");
    let bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) };
    gpu.hip
        .memcpy_htod(&tensor.buf, bytes)
        .expect("H2D F32 upload");
}

fn has_nonzero_bytes(buffers: &[Vec<u8>]) -> bool {
    buffers
        .iter()
        .any(|buffer| buffer.iter().any(|&byte| byte != 0))
}

fn assert_nonzero_tensor(gpu: &Gpu, label: &str, tensor: &GpuTensor) {
    let values = gpu.download_f32(tensor).expect("D2H F32 output");
    assert!(
        values.iter().any(|value| value.to_bits() != 0),
        "{label} must be nonzero"
    );
}

fn assert_q8_ef_allocated(dn: &DeltaNetState, label: &str) {
    assert_eq!(dn.quant, StateQuant::Q8, "{label}: expected Q8 state");
    assert_eq!(
        dn.s_ef_residual.len(),
        dn.s_matrices.len(),
        "{label}: Q8 EF allocation count mismatch"
    );
    assert!(
        !dn.s_ef_residual.is_empty(),
        "{label}: Q8 EF allocation is empty"
    );
}

fn new_kv(gpu: &mut Gpu, config: &qwen35::Qwen35Config) -> KvCache {
    KvCache::new_gpu_asym3_capped(
        gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        64,
        64,
    )
    .expect("asym3 KV cache")
}

fn compare_decode(
    gpu: &mut Gpu,
    weights: &qwen35::Qwen35Weights,
    config: &qwen35::Qwen35Config,
    quant: StateQuant,
    updates: usize,
) {
    let mut raw_kv = new_kv(gpu, config);
    let mut step_kv = new_kv(gpu, config);
    let mut raw_dn = DeltaNetState::new_with_quant(gpu, config, quant).expect("raw state");
    let mut step_dn = DeltaNetState::new_with_quant(gpu, config, quant).expect("step state");
    let raw_scratch = Qwen35Scratch::new_with_kv_max(gpu, config, 64, 64).expect("raw scratch");
    let step_scratch = Qwen35Scratch::new_with_kv_max(gpu, config, 64, 64).expect("step scratch");
    let q8_ef = quant == StateQuant::Q8;
    let mut raw_ef_after_update1 = None;
    let mut step_ef_after_update1 = None;
    if q8_ef {
        assert_q8_ef_allocated(&raw_dn, "raw decode");
        assert_q8_ef_allocated(&step_dn, "Step decode");
        assert!(
            !has_nonzero_bytes(&state_bytes(gpu, &raw_dn).dn_ef),
            "raw Q8 EF must start zero"
        );
        assert!(
            !has_nonzero_bytes(&state_bytes(gpu, &step_dn).dn_ef),
            "Step Q8 EF must start zero"
        );
    }

    for (i, &token) in TOKENS.iter().take(updates).enumerate() {
        legacy_delta_net_sequence(
            gpu,
            weights,
            config,
            token,
            i,
            &mut raw_kv,
            &mut raw_dn,
            &raw_scratch,
        )
        .expect("raw decode");
        qwen35::forward_scratch(
            gpu,
            weights,
            config,
            token,
            i,
            &mut step_kv,
            &mut step_dn,
            &step_scratch,
        )
        .expect("Step decode");

        assert_tensor_bytes(
            gpu,
            "decode logits",
            &raw_scratch.logits,
            &step_scratch.logits,
        );
        let raw_after = state_bytes(gpu, &raw_dn);
        let step_after = state_bytes(gpu, &step_dn);
        if q8_ef {
            if i == 0 {
                assert!(
                    has_nonzero_bytes(&raw_after.dn_ef),
                    "raw Q8 EF must mutate after update1"
                );
                assert!(
                    has_nonzero_bytes(&step_after.dn_ef),
                    "Step Q8 EF must mutate after update1"
                );
                raw_ef_after_update1 = Some(raw_after.dn_ef.clone());
                step_ef_after_update1 = Some(step_after.dn_ef.clone());
            } else if i == 1 {
                assert_ne!(
                    &raw_after.dn_ef,
                    raw_ef_after_update1
                        .as_ref()
                        .expect("raw Q8 EF update1 snapshot"),
                    "raw Q8 EF did not carry and update on update2"
                );
                assert_ne!(
                    &step_after.dn_ef,
                    step_ef_after_update1
                        .as_ref()
                        .expect("Step Q8 EF update1 snapshot"),
                    "Step Q8 EF did not carry and update on update2"
                );
            }
        }
        assert_state_bytes(
            &format!("decode state after update {i}"),
            raw_after,
            step_after,
        );
    }

    raw_scratch.free_gpu(gpu);
    step_scratch.free_gpu(gpu);
    raw_dn.free_gpu(gpu);
    step_dn.free_gpu(gpu);
    raw_kv.free_gpu(gpu);
    step_kv.free_gpu(gpu);
}

fn compare_prefill(
    gpu: &mut Gpu,
    weights: &qwen35::Qwen35Weights,
    config: &qwen35::Qwen35Config,
    tree: bool,
    quant: StateQuant,
) {
    let n = TOKENS.len();
    let mut raw_kv = new_kv(gpu, config);
    let mut step_kv = new_kv(gpu, config);
    let mut raw_dn = DeltaNetState::new_with_quant(gpu, config, quant).expect("raw state");
    let mut step_dn = DeltaNetState::new_with_quant(gpu, config, quant).expect("step state");
    let raw_scratch = Qwen35Scratch::new_with_kv_max(gpu, config, 64, 64).expect("raw scratch");
    let step_scratch = Qwen35Scratch::new_with_kv_max(gpu, config, 64, 64).expect("step scratch");
    let raw_pbs = qwen35::PrefillBatchScratch::new_opt(gpu, config, n, tree).expect("raw pbs");
    let step_pbs = qwen35::PrefillBatchScratch::new_opt(gpu, config, n, tree).expect("step pbs");
    let raw_hidden = gpu
        .zeros(&[n * config.dim], DType::F32)
        .expect("raw hidden");
    let step_hidden = gpu
        .zeros(&[n * config.dim], DType::F32)
        .expect("step hidden");

    if quant == StateQuant::Q8 {
        assert_q8_ef_allocated(&raw_dn, "raw prefill/tree");
        assert_q8_ef_allocated(&step_dn, "Step prefill/tree");
        assert!(!has_nonzero_bytes(&state_bytes(gpu, &raw_dn).dn_ef));
        assert!(!has_nonzero_bytes(&state_bytes(gpu, &step_dn).dn_ef));
    }

    let mut raw_parents = None;
    let mut step_parents = None;
    let mut raw_bias = None;
    let mut step_bias = None;
    let raw_ctx = if tree {
        let parent_bytes = n * std::mem::size_of::<i32>();
        raw_parents = Some(
            gpu.alloc_tensor(&[parent_bytes], DType::Raw)
                .expect("raw parents"),
        );
        raw_bias = Some(gpu.zeros(&[n * n], DType::F32).expect("raw tree bias"));
        upload_i32(gpu, raw_parents.as_ref().unwrap(), &[-1, 0, 0, 1]);
        Some(qwen35::TreeVerifyCtx {
            positions: &[0, 1, 1, 2],
            attn_bias: raw_bias.as_ref().unwrap(),
            parent_indices: Some(raw_parents.as_ref().unwrap()),
        })
    } else {
        None
    };
    let step_ctx = if tree {
        let parent_bytes = n * std::mem::size_of::<i32>();
        step_parents = Some(
            gpu.alloc_tensor(&[parent_bytes], DType::Raw)
                .expect("step parents"),
        );
        step_bias = Some(gpu.zeros(&[n * n], DType::F32).expect("step tree bias"));
        upload_i32(gpu, step_parents.as_ref().unwrap(), &[-1, 0, 0, 1]);
        Some(qwen35::TreeVerifyCtx {
            positions: &[0, 1, 1, 2],
            attn_bias: step_bias.as_ref().unwrap(),
            parent_indices: Some(step_parents.as_ref().unwrap()),
        })
    } else {
        None
    };

    with_raw_delta_net_scope(|| {
        qwen35::forward_prefill_batch_with_pbs(
            gpu,
            weights,
            config,
            &TOKENS,
            0,
            &mut raw_kv,
            &mut raw_dn,
            &raw_scratch,
            None,
            Some(&raw_hidden),
            None,
            raw_ctx,
            Some(&raw_pbs),
            None,
            None,
        )
    })
    .expect("raw prefill/tree");
    qwen35::forward_prefill_batch_with_pbs(
        gpu,
        weights,
        config,
        &TOKENS,
        0,
        &mut step_kv,
        &mut step_dn,
        &step_scratch,
        None,
        Some(&step_hidden),
        None,
        step_ctx,
        Some(&step_pbs),
        None,
        None,
    )
    .expect("Step prefill/tree");

    assert_tensor_bytes(
        gpu,
        "prefill/tree logits",
        &raw_scratch.logits,
        &step_scratch.logits,
    );
    assert_tensor_bytes(gpu, "prefill/tree hidden output", &raw_hidden, &step_hidden);
    let raw_after = state_bytes(gpu, &raw_dn);
    let step_after = state_bytes(gpu, &step_dn);
    if quant == StateQuant::Q8 && !tree {
        assert!(
            has_nonzero_bytes(&raw_after.dn_ef),
            "raw Q8 prefill EF must mutate"
        );
        assert!(
            has_nonzero_bytes(&step_after.dn_ef),
            "Step Q8 prefill EF must mutate"
        );
    }
    if quant == StateQuant::Q8 && tree {
        // The Q8 tree-tape kernel has no EF-residual operand. Q8 tree parity
        // is therefore covered, but EF mutation is intentionally not claimed.
        assert!(
            !has_nonzero_bytes(&raw_after.dn_ef),
            "raw Q8 tree EF unexpectedly mutated"
        );
        assert!(
            !has_nonzero_bytes(&step_after.dn_ef),
            "Step Q8 tree EF unexpectedly mutated"
        );
    }
    assert_state_bytes(
        if tree { "tree state" } else { "prefill state" },
        raw_after,
        step_after,
    );
    if tree {
        let state_elems = config.linear_num_value_heads
            * config.linear_value_head_dim
            * config.linear_value_head_dim;
        let scale_elems = config.linear_num_value_heads * config.linear_value_head_dim;
        for node in 0..n {
            match quant {
                StateQuant::FP32 => {
                    let raw_tape = raw_pbs
                        .dn_s_tape_f32
                        .as_ref()
                        .expect("raw FP32 tree state tape");
                    let step_tape = step_pbs
                        .dn_s_tape_f32
                        .as_ref()
                        .expect("Step FP32 tree state tape");
                    let raw_node = raw_tape.sub_offset(node * state_elems, state_elems);
                    let step_node = step_tape.sub_offset(node * state_elems, state_elems);
                    assert_tensor_bytes(
                        gpu,
                        &format!("FP32 tree state tape node {node}"),
                        &raw_node,
                        &step_node,
                    );
                }
                StateQuant::Q8 => {
                    let raw_tape = raw_pbs
                        .dn_s_tape_q8
                        .as_ref()
                        .expect("raw Q8 tree state tape");
                    let step_tape = step_pbs
                        .dn_s_tape_q8
                        .as_ref()
                        .expect("Step Q8 tree state tape");
                    let raw_node = raw_tape.sub_offset(node * state_elems, state_elems);
                    let step_node = step_tape.sub_offset(node * state_elems, state_elems);
                    assert_tensor_bytes(
                        gpu,
                        &format!("Q8 tree state tape node {node}"),
                        &raw_node,
                        &step_node,
                    );
                    let raw_scales = raw_pbs
                        .dn_s_tape_scales
                        .as_ref()
                        .expect("raw Q8 tree scales");
                    let step_scales = step_pbs
                        .dn_s_tape_scales
                        .as_ref()
                        .expect("Step Q8 tree scales");
                    let raw_scale_node = raw_scales.sub_offset(node * scale_elems, scale_elems);
                    let step_scale_node = step_scales.sub_offset(node * scale_elems, scale_elems);
                    assert_tensor_bytes(
                        gpu,
                        &format!("Q8 tree scale tape node {node}"),
                        &raw_scale_node,
                        &step_scale_node,
                    );
                }
                StateQuant::Q4 => unreachable!("Q4 tree parity is unsupported"),
            }
        }
    }

    raw_scratch.free_gpu(gpu);
    step_scratch.free_gpu(gpu);
    raw_pbs.free_gpu(gpu);
    step_pbs.free_gpu(gpu);
    raw_dn.free_gpu(gpu);
    step_dn.free_gpu(gpu);
    raw_kv.free_gpu(gpu);
    step_kv.free_gpu(gpu);
    let _ = gpu.free_tensor(raw_hidden);
    let _ = gpu.free_tensor(step_hidden);
    if tree {
        let _ = gpu.free_tensor(raw_parents.unwrap());
        let _ = gpu.free_tensor(step_parents.unwrap());
        let _ = gpu.free_tensor(raw_bias.unwrap());
        let _ = gpu.free_tensor(step_bias.unwrap());
    }
}

fn fill_replay_tape(gpu: &Gpu, tape: &GdnTape) {
    for (layer, tensor) in tape.qkv_bufs.iter().enumerate() {
        let values: Vec<f32> = (0..tensor.numel())
            .map(|index| 0.01 * (((index + layer * 3) % 31 + 1) as f32))
            .collect();
        upload_f32(gpu, tensor, &values);
    }
    for (layer, tensor) in tape.alpha_bufs.iter().enumerate() {
        let values: Vec<f32> = (0..tensor.numel())
            .map(|index| 0.65 + 0.01 * ((index + layer) % 7) as f32)
            .collect();
        upload_f32(gpu, tensor, &values);
    }
    for (layer, tensor) in tape.beta_bufs.iter().enumerate() {
        let values: Vec<f32> = (0..tensor.numel())
            .map(|index| 0.15 + 0.01 * ((index + layer) % 5) as f32)
            .collect();
        upload_f32(gpu, tensor, &values);
    }
}

fn compare_replay(
    gpu: &mut Gpu,
    weights: &qwen35::Qwen35Weights,
    config: &qwen35::Qwen35Config,
    quant: StateQuant,
) {
    let mut raw_dn = DeltaNetState::new_with_quant(gpu, config, quant).expect("raw replay state");
    let mut step_dn = DeltaNetState::new_with_quant(gpu, config, quant).expect("step replay state");
    let raw_tape = GdnTape::new_for_config(gpu, config, 2).expect("raw replay tape");
    let step_tape = GdnTape::new_for_config(gpu, config, 2).expect("step replay tape");
    fill_replay_tape(gpu, &raw_tape);
    fill_replay_tape(gpu, &step_tape);
    let raw_before = state_bytes(gpu, &raw_dn);
    let step_before = state_bytes(gpu, &step_dn);

    with_raw_delta_net_scope(|| raw_tape.replay_gdn(gpu, weights, config, &mut raw_dn, 2))
        .expect("raw replay");
    step_tape
        .replay_gdn(gpu, weights, config, &mut step_dn, 2)
        .expect("Step replay");

    assert_nonzero_tensor(gpu, "raw replay attention output", &raw_tape.attn_scratch);
    assert_nonzero_tensor(gpu, "Step replay attention output", &step_tape.attn_scratch);
    let raw_after = state_bytes(gpu, &raw_dn);
    let step_after = state_bytes(gpu, &step_dn);
    assert_ne!(raw_after, raw_before, "raw replay did not mutate state");
    assert_ne!(step_after, step_before, "Step replay did not mutate state");
    if quant == StateQuant::Q8 {
        assert_q8_ef_allocated(&raw_dn, "raw replay");
        assert_q8_ef_allocated(&step_dn, "Step replay");
        assert!(
            has_nonzero_bytes(&raw_after.dn_ef),
            "raw Q8 replay EF must mutate"
        );
        assert!(
            has_nonzero_bytes(&step_after.dn_ef),
            "Step Q8 replay EF must mutate"
        );
    }
    assert_tensor_bytes(
        gpu,
        "replay attention output",
        &raw_tape.attn_scratch,
        &step_tape.attn_scratch,
    );
    assert_state_bytes("replay state", raw_after, step_after);

    raw_tape.free_gpu(gpu);
    step_tape.free_gpu(gpu);
    raw_dn.free_gpu(gpu);
    step_dn.free_gpu(gpu);
}

#[test]
#[ignore = "requires AMD GPU and qwen3.5-0.8b.mq4"]
fn raw_vs_step_delta_net_gpu_parity() {
    std::env::set_var("HIPFIRE_GRAPH", "0");
    std::env::set_var("HIPFIRE_AR_GRAPH", "0");
    std::env::set_var("HIPFIRE_DN_STATE_EF", "1");

    let model = fixture_path();
    assert!(model.is_file(), "fixture not found: {}", model.display());
    let mut hfq = HfqFile::open(Path::new(&model)).expect("open fixture");
    let config = qwen35::config_from_hfq(&hfq).expect("fixture config");
    let mut gpu = Gpu::init().expect("Gpu::init");
    let mut source = qwen35::HfqSource::new(&mut hfq, &config);
    let layout = qwen35::Layout::single(config.n_layers);
    let weights = qwen35::load_weights(&mut source, std::slice::from_mut(&mut gpu), &layout)
        .expect("load fixture weights");

    eprintln!(
        "[delta-net-parity] model={} arch={}",
        model.display(),
        gpu.arch
    );
    eprintln!("[delta-net-parity] FP32 decode N=1");
    compare_decode(&mut gpu, &weights, &config, StateQuant::FP32, 1);
    eprintln!("[delta-net-parity] FP32 prefill N>1");
    compare_prefill(&mut gpu, &weights, &config, false, StateQuant::FP32);
    eprintln!("[delta-net-parity] FP32 tree");
    compare_prefill(&mut gpu, &weights, &config, true, StateQuant::FP32);
    eprintln!("[delta-net-parity] Q8 EF prefill N>1");
    compare_prefill(&mut gpu, &weights, &config, false, StateQuant::Q8);
    eprintln!("[delta-net-parity] Q8 tree parity (tree kernel has no EF operand)");
    compare_prefill(&mut gpu, &weights, &config, true, StateQuant::Q8);
    eprintln!("[delta-net-parity] FP32 replay N>1");
    compare_replay(&mut gpu, &weights, &config, StateQuant::FP32);
    eprintln!("[delta-net-parity] Q8 EF replay N>1");
    compare_replay(&mut gpu, &weights, &config, StateQuant::Q8);
    eprintln!("[delta-net-parity] Q8 EF two updates");
    compare_decode(&mut gpu, &weights, &config, StateQuant::Q8, 2);

    weights.free_gpu(&mut gpu);
    gpu.drain_pool();
    if config.num_experts == 0 {
        eprintln!("[delta-net-parity] MoE behavioral parity scope limit: fixture is dense (num_experts=0)");
    } else {
        eprintln!("[delta-net-parity] MoE behavioral parity exercised");
    }
    eprintln!("[delta-net-parity] PASS: FP32/Q8 decode/prefill/tree/replay + Q8 EF");
}
