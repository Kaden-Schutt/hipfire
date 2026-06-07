// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.

//! Channel test for `kld_tile_topk_lse_f32`.
//!
//! Validates the exact per-2048 chunk top-256 contract used by the KLD
//! reference builder's GPU reducer path.

use rdna_compute::{DType, Gpu, GpuTensor};

const B: usize = 3;
const VOCAB_TILE: usize = 4096;
const CHUNK: usize = 2048;
const TOPK: usize = 256;
const GLOBAL_START: usize = 1234;

fn download_i32(gpu: &Gpu, tensor: &GpuTensor) -> Vec<i32> {
    gpu.bind_thread().expect("bind thread");
    let mut out = vec![0i32; tensor.numel()];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut u8,
            out.len() * std::mem::size_of::<i32>(),
        )
    };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .expect("download i32");
    out
}

fn cpu_chunk(logits: &[f32], row: usize, chunk: usize) -> (Vec<(i32, f32)>, f32, f32) {
    let base = row * VOCAB_TILE + chunk * CHUNK;
    let mut pairs: Vec<(i32, f32)> = (0..CHUNK)
        .map(|i| ((GLOBAL_START + chunk * CHUNK + i) as i32, logits[base + i]))
        .collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let max_v = pairs[0].1;
    let sum = logits[base..base + CHUNK]
        .iter()
        .map(|v| (*v - max_v).exp())
        .sum::<f32>();

    (pairs.into_iter().take(TOPK).collect(), max_v, sum)
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch={}", gpu.arch);

    let mut logits = vec![0.0f32; B * VOCAB_TILE];
    for row in 0..B {
        for i in 0..VOCAB_TILE {
            let wave = ((i * 37 + row * 101) % 997) as f32 * 0.001;
            let slope = -((i % CHUNK) as f32) * 0.00017;
            logits[row * VOCAB_TILE + i] = wave + slope + row as f32 * 0.03;
        }
        for chunk in 0..(VOCAB_TILE / CHUNK) {
            let base = row * VOCAB_TILE + chunk * CHUNK;
            logits[base + 17 + row] = 100.0 + row as f32;
            logits[base + 509] = 99.5 + chunk as f32 * 0.01;
            logits[base + 1023] = 99.0;
            logits[base + 1537] = 98.5;
            logits[base + 2047 - row] = 98.0;
        }
    }

    let d_logits = gpu
        .upload_f32(&logits, &[B, VOCAB_TILE])
        .expect("upload logits");
    let n_chunks = VOCAB_TILE / CHUNK;
    let top_vals = gpu
        .alloc_tensor(&[B, n_chunks, TOPK], DType::F32)
        .expect("alloc top vals");
    let top_idx = gpu
        .alloc_tensor(&[B, n_chunks, TOPK], DType::F32)
        .expect("alloc top idx");
    let chunk_max = gpu
        .alloc_tensor(&[B, n_chunks], DType::F32)
        .expect("alloc chunk max");
    let chunk_sum = gpu
        .alloc_tensor(&[B, n_chunks], DType::F32)
        .expect("alloc chunk sum");

    gpu.kld_tile_topk_lse_f32(
        &d_logits,
        &top_vals,
        &top_idx,
        &chunk_max,
        &chunk_sum,
        B,
        VOCAB_TILE,
        GLOBAL_START,
        n_chunks,
    )
    .expect("kld_tile_topk_lse_f32");
    gpu.hip.device_synchronize().expect("sync");

    let got_vals = gpu.download_f32(&top_vals).expect("download top vals");
    let got_idx = download_i32(&gpu, &top_idx);
    let got_max = gpu.download_f32(&chunk_max).expect("download chunk max");
    let got_sum = gpu.download_f32(&chunk_sum).expect("download chunk sum");

    let mut max_top_diff = 0.0f32;
    let mut max_sum_rel = 0.0f32;
    for row in 0..B {
        for chunk in 0..n_chunks {
            let (want_top, want_max, want_sum) = cpu_chunk(&logits, row, chunk);
            let stat = row * n_chunks + chunk;
            assert_eq!(
                got_max[stat], want_max,
                "chunk max mismatch row={row} chunk={chunk}"
            );
            let sum_rel = ((got_sum[stat] - want_sum) / want_sum).abs();
            max_sum_rel = max_sum_rel.max(sum_rel);
            assert!(
                sum_rel < 2.0e-6,
                "chunk sum mismatch row={row} chunk={chunk}: got={} want={} rel={}",
                got_sum[stat],
                want_sum,
                sum_rel
            );

            let off = stat * TOPK;
            for rank in 0..TOPK {
                let (want_idx, want_val) = want_top[rank];
                assert_eq!(
                    got_idx[off + rank],
                    want_idx,
                    "top idx mismatch row={row} chunk={chunk} rank={rank}"
                );
                let diff = (got_vals[off + rank] - want_val).abs();
                max_top_diff = max_top_diff.max(diff);
                assert!(
                    diff == 0.0,
                    "top val mismatch row={row} chunk={chunk} rank={rank}: got={} want={}",
                    got_vals[off + rank],
                    want_val
                );
            }
        }
    }

    println!("max_top_diff={max_top_diff:.6e} max_sum_rel={max_sum_rel:.6e}");
}
