// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Decode-shape channel screen for eliminating DS4's ratio-4 top-K KV gather.
//!
//! Compares the current graph-safe sequence
//!   deepseek4_topk_kv_gather_f32_buf -> deepseek4_attn_swa_topk_f32_buf
//! against the existing direct-main-KV scalar attention kernel. The fixtures
//! are the exact TP3 head partitions (24/24/16), batch 1, D=512, SWA=128,
//! K=512 and Ncompressed=513 at the canonical 2K-context waterline.

use rdna_compute::{DType, Gpu, GpuTensor};

const D: usize = 512;
const SWA: usize = 128;
const TOPK: usize = 512;
const N_COMPRESSED: usize = 513;
const WARMUP: usize = 20;
const ITERS: usize = 500;

fn u2f(x: u32) -> f32 {
    ((x >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
}

fn upload_i32(gpu: &mut Gpu, values: &[i32]) -> GpuTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    gpu.upload_raw(bytes, &[bytes.len()]).expect("upload i32")
}

#[allow(clippy::too_many_arguments)]
fn launch_gathered(
    gpu: &mut Gpu,
    q: &GpuTensor,
    swa_k: &GpuTensor,
    swa_v: &GpuTensor,
    kv: &GpuTensor,
    indices: &GpuTensor,
    sink: &GpuTensor,
    n_valid: &GpuTensor,
    n_active: &GpuTensor,
    n_compressed: &GpuTensor,
    gathered: &GpuTensor,
    out: &GpuTensor,
    heads: usize,
) {
    gpu.deepseek4_topk_kv_gather_f32_buf(
        kv,
        indices,
        gathered,
        n_active,
        n_compressed,
        TOPK as i32,
        D as i32,
        TOPK as i32,
        0,
        1.0,
    )
    .expect("gather");
    gpu.deepseek4_attn_swa_topk_f32_buf(
        false,
        q,
        swa_k,
        swa_v,
        gathered,
        gathered,
        sink,
        out,
        n_valid,
        n_active,
        heads as i32,
        D as i32,
        SWA as i32,
        TOPK as i32,
    )
    .expect("gathered attention");
}

#[allow(clippy::too_many_arguments)]
fn launch_direct(
    gpu: &mut Gpu,
    q: &GpuTensor,
    swa_k: &GpuTensor,
    swa_v: &GpuTensor,
    kv: &GpuTensor,
    indices: &GpuTensor,
    sink: &GpuTensor,
    n_valid: &GpuTensor,
    n_active: &GpuTensor,
    out: &GpuTensor,
    heads: usize,
) {
    gpu.deepseek4_attn_swa_topk_direct_batched_f32(
        q,
        swa_k,
        swa_v,
        kv,
        indices,
        sink,
        n_valid,
        n_active,
        out,
        heads as i32,
        D as i32,
        SWA as i32,
        TOPK as i32,
        N_COMPRESSED as i32,
        1,
    )
    .expect("direct attention");
}

fn run_shape(gpu: &mut Gpu, heads: usize, seed: &mut u32) {
    let mut next = || {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *seed
    };
    let q: Vec<f32> = (0..heads * D).map(|_| u2f(next())).collect();
    let swa_k: Vec<f32> = (0..D * SWA).map(|_| u2f(next())).collect();
    let swa_v: Vec<f32> = (0..D * SWA).map(|_| u2f(next())).collect();
    let kv: Vec<f32> = (0..N_COMPRESSED * D).map(|_| u2f(next())).collect();
    let sink: Vec<f32> = (0..heads).map(|_| u2f(next()) * 0.5).collect();

    // Deterministic permutation of 0..511; all indices are valid and exactly
    // one of the 513 compressed rows is excluded, as in the near-cap route.
    let indices: Vec<i32> = (0..TOPK)
        .map(|i| ((i * 313 + 97) % TOPK) as i32)
        .collect();
    let valid = [SWA as i32];
    let active = [TOPK as i32];
    let n_compressed = [N_COMPRESSED as i32];

    let d_q = gpu.upload_f32(&q, &[heads * D]).expect("q");
    let d_swa_k = gpu.upload_f32(&swa_k, &[D * SWA]).expect("swa k");
    let d_swa_v = gpu.upload_f32(&swa_v, &[D * SWA]).expect("swa v");
    let d_kv = gpu
        .upload_f32(&kv, &[N_COMPRESSED * D])
        .expect("main kv");
    let d_indices = upload_i32(gpu, &indices);
    let d_sink = gpu.upload_f32(&sink, &[heads]).expect("sink");
    let d_valid = upload_i32(gpu, &valid);
    let d_active = upload_i32(gpu, &active);
    let d_n_compressed = upload_i32(gpu, &n_compressed);
    let d_gathered = gpu.zeros(&[D * TOPK], DType::F32).expect("gathered");
    let d_gathered_out = gpu.zeros(&[heads * D], DType::F32).expect("gathered out");
    let d_direct_out = gpu.zeros(&[heads * D], DType::F32).expect("direct out");

    launch_gathered(
        gpu,
        &d_q,
        &d_swa_k,
        &d_swa_v,
        &d_kv,
        &d_indices,
        &d_sink,
        &d_valid,
        &d_active,
        &d_n_compressed,
        &d_gathered,
        &d_gathered_out,
        heads,
    );
    launch_direct(
        gpu,
        &d_q,
        &d_swa_k,
        &d_swa_v,
        &d_kv,
        &d_indices,
        &d_sink,
        &d_valid,
        &d_active,
        &d_direct_out,
        heads,
    );
    gpu.hip.device_synchronize().expect("initial synchronize");

    let gathered_out = gpu
        .download_f32(&d_gathered_out)
        .expect("download gathered");
    let direct_out = gpu.download_f32(&d_direct_out).expect("download direct");
    let mismatches = gathered_out
        .iter()
        .zip(&direct_out)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let max_abs = gathered_out
        .iter()
        .zip(&direct_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    for _ in 0..WARMUP {
        launch_gathered(
            gpu,
            &d_q,
            &d_swa_k,
            &d_swa_v,
            &d_kv,
            &d_indices,
            &d_sink,
            &d_valid,
            &d_active,
            &d_n_compressed,
            &d_gathered,
            &d_gathered_out,
            heads,
        );
        launch_direct(
            gpu,
            &d_q,
            &d_swa_k,
            &d_swa_v,
            &d_kv,
            &d_indices,
            &d_sink,
            &d_valid,
            &d_active,
            &d_direct_out,
            heads,
        );
    }
    gpu.hip.device_synchronize().expect("warmup synchronize");

    let e0 = gpu.hip.event_create().expect("event");
    let e1 = gpu.hip.event_create().expect("event");
    gpu.hip.event_record(&e0, None).expect("record");
    for _ in 0..ITERS {
        launch_gathered(
            gpu,
            &d_q,
            &d_swa_k,
            &d_swa_v,
            &d_kv,
            &d_indices,
            &d_sink,
            &d_valid,
            &d_active,
            &d_n_compressed,
            &d_gathered,
            &d_gathered_out,
            heads,
        );
    }
    gpu.hip.event_record(&e1, None).expect("record");
    gpu.hip.event_synchronize(&e1).expect("sync");
    let gathered_us = gpu.hip.event_elapsed_ms(&e0, &e1).expect("elapsed") as f64 * 1_000.0
        / ITERS as f64;

    let e2 = gpu.hip.event_create().expect("event");
    let e3 = gpu.hip.event_create().expect("event");
    gpu.hip.event_record(&e2, None).expect("record");
    for _ in 0..ITERS {
        launch_direct(
            gpu,
            &d_q,
            &d_swa_k,
            &d_swa_v,
            &d_kv,
            &d_indices,
            &d_sink,
            &d_valid,
            &d_active,
            &d_direct_out,
            heads,
        );
    }
    gpu.hip.event_record(&e3, None).expect("record");
    gpu.hip.event_synchronize(&e3).expect("sync");
    let direct_us = gpu.hip.event_elapsed_ms(&e2, &e3).expect("elapsed") as f64 * 1_000.0
        / ITERS as f64;

    eprintln!(
        "H={heads}: gather+attention={gathered_us:.3} us direct={direct_us:.3} us speedup={:.4}x saved={:.3} us raw_mismatches={mismatches}/{} max_abs={max_abs:.9e}",
        gathered_us / direct_us,
        gathered_us - direct_us,
        gathered_out.len(),
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!(
        "DS4 direct decode channel: arch={} B=1 D={D} SWA={SWA} K={TOPK} Ncompressed={N_COMPRESSED}",
        gpu.arch
    );
    let mut seed = 0xD54D_1201u32;
    run_shape(&mut gpu, 24, &mut seed);
    run_shape(&mut gpu, 16, &mut seed);
}
