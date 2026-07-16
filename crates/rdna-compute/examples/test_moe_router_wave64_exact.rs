// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bitwise differential test for the gfx1100 exact wave64 MoE router.
//!
//! Build/run on gfx1100:
//! `cargo run --release -p rdna-compute --example test_moe_router_wave64_exact`

use rdna_compute::{DType, Gpu};

const N_EXP: usize = 256;
const TOP_K: usize = 8;

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    let mut failures = 0usize;
    let mut cases = 0usize;

    for seed in 0..128u32 {
        let logits = make_logits(seed);
        failures += run_case(&mut gpu, &logits, true, seed);
        if failures != 0 {
            break;
        }
        failures += run_case(&mut gpu, &logits, false, seed);
        cases += 2;
        if failures != 0 {
            break;
        }
    }

    // Explicit ties exercise the production path's strict first-larger-wins
    // behavior across wave32 chunk boundaries.
    let mut ties = vec![-4.0f32; N_EXP];
    for &i in &[0usize, 31, 32, 63, 64, 127, 128, 255] {
        ties[i] = 7.0;
    }
    if failures == 0 {
        failures += run_case(&mut gpu, &ties, true, u32::MAX);
        cases += 1;
    }

    if failures != 0 {
        eprintln!("FAIL: {failures}/{cases} router cases differed");
        std::process::exit(1);
    }
    println!("PASS: {cases} cases matched indices and weights bit-for-bit");
}

fn run_case(gpu: &mut Gpu, logits: &[f32], norm_topk: bool, seed: u32) -> usize {
    let logits_ref = gpu.upload_f32(logits, &[N_EXP]).expect("upload ref logits");
    let logits_exact = gpu
        .upload_f32(logits, &[N_EXP])
        .expect("upload exact logits");
    let idx_ref = gpu.zeros(&[TOP_K], DType::F32).expect("reference indices");
    let weight_ref = gpu.zeros(&[TOP_K], DType::F32).expect("reference weights");
    let idx_exact = gpu.zeros(&[TOP_K], DType::F32).expect("exact indices");
    let weight_exact = gpu.zeros(&[TOP_K], DType::F32).expect("exact weights");

    gpu.softmax_f32(&logits_ref).expect("reference softmax");
    gpu.moe_topk_renorm_k8(&logits_ref, &idx_ref, &weight_ref, N_EXP, norm_topk)
        .expect("reference top-k");
    gpu.moe_router_softmax_topk_k8_wave64_exact(
        &logits_exact,
        &idx_exact,
        &weight_exact,
        N_EXP,
        norm_topk,
    )
    .expect("exact wave64 router");

    let ref_idx = gpu
        .download_f32(&idx_ref)
        .expect("download reference indices");
    let got_idx = gpu
        .download_f32(&idx_exact)
        .expect("download exact indices");
    let ref_weight = gpu
        .download_f32(&weight_ref)
        .expect("download reference weights");
    let got_weight = gpu
        .download_f32(&weight_exact)
        .expect("download exact weights");

    let idx_match = ref_idx
        .iter()
        .zip(&got_idx)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let weight_match = ref_weight
        .iter()
        .zip(&got_weight)
        .all(|(a, b)| a.to_bits() == b.to_bits());

    for tensor in [
        logits_ref,
        logits_exact,
        idx_ref,
        weight_ref,
        idx_exact,
        weight_exact,
    ] {
        gpu.free_tensor(tensor).expect("free tensor");
    }

    if idx_match && weight_match {
        return 0;
    }

    eprintln!("case seed={seed} norm={norm_topk} differs");
    for rank in 0..TOP_K {
        if ref_idx[rank].to_bits() != got_idx[rank].to_bits()
            || ref_weight[rank].to_bits() != got_weight[rank].to_bits()
        {
            eprintln!(
                "  rank {rank}: idx ref_bits={:#010x} got_bits={:#010x}; \
                 weight ref={:.9} ({:#010x}) got={:.9} ({:#010x})",
                ref_idx[rank].to_bits(),
                got_idx[rank].to_bits(),
                ref_weight[rank],
                ref_weight[rank].to_bits(),
                got_weight[rank],
                got_weight[rank].to_bits(),
            );
        }
    }
    1
}

fn make_logits(seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_add(1).wrapping_mul(0x9e37_79b9);
    (0..N_EXP)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let centered = (state & 0x00ff_ffff) as f32 / 8_388_608.0 - 1.0;
            centered * 12.0 + (i as f32 - 127.5) * 0.000_031_25
        })
        .collect()
}
