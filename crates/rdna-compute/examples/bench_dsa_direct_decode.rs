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

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

const D: usize = 512;
const SWA: usize = 128;
const TOPK: usize = 512;
const N_COMPRESSED: usize = 513;
const WARMUP: usize = 20;
const ITERS: usize = 500;
const SPLIT_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_attn_swa_topk_split.gfx1201.hip");
const SPLIT_MODULE: &str = "deepseek4_attn_swa_topk_split_gfx1201_screen";
const SPLIT_PARTIAL: &str = "deepseek4_attn_swa_topk_split_partial_gfx1201";
const SPLIT_REDUCE: &str = "deepseek4_attn_swa_topk_split_reduce_gfx1201";

fn u2f(x: u32) -> f32 {
    ((x >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
}

fn upload_i32(gpu: &mut Gpu, values: &[i32]) -> GpuTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    gpu.upload_raw(bytes, &[bytes.len()]).expect("upload i32")
}

fn ensure_split_kernels(gpu: &mut Gpu) {
    gpu.ensure_kernel_public(SPLIT_MODULE, SPLIT_SRC, SPLIT_PARTIAL)
        .expect("compile split partial");
    gpu.ensure_kernel_public(SPLIT_MODULE, SPLIT_SRC, SPLIT_REDUCE)
        .expect("compile split reduce");
}

#[allow(clippy::too_many_arguments)]
fn launch_split(
    gpu: &Gpu,
    q: &GpuTensor,
    swa_k: &GpuTensor,
    swa_v: &GpuTensor,
    gathered: &GpuTensor,
    sink: &GpuTensor,
    n_valid: &GpuTensor,
    n_active: &GpuTensor,
    partials: &GpuTensor,
    out: &GpuTensor,
    heads: usize,
    chunk_size: usize,
) {
    let n_chunks = (SWA + TOPK).div_ceil(chunk_size);
    let mut partial = KernargBlob::new();
    partial.push_ptr(q.buf.as_ptr() as *const c_void);
    partial.push_ptr(swa_k.buf.as_ptr() as *const c_void);
    partial.push_ptr(swa_v.buf.as_ptr() as *const c_void);
    partial.push_ptr(gathered.buf.as_ptr() as *const c_void);
    partial.push_ptr(gathered.buf.as_ptr() as *const c_void);
    partial.push_ptr(sink.buf.as_ptr() as *const c_void);
    partial.push_ptr(partials.buf.as_ptr() as *const c_void);
    partial.push_ptr(n_valid.buf.as_ptr() as *const c_void);
    partial.push_ptr(n_active.buf.as_ptr() as *const c_void);
    partial.push_i32(heads as i32);
    partial.push_i32(D as i32);
    partial.push_i32(SWA as i32);
    partial.push_i32(TOPK as i32);
    partial.push_i32(n_chunks as i32);
    partial.push_i32(chunk_size as i32);
    partial.pad_to(16);
    gpu.launch_kernel_blob(
        SPLIT_PARTIAL,
        [heads as u32, n_chunks as u32, 1],
        [chunk_size as u32, 1, 1],
        0,
        partial.as_mut_slice(),
    )
    .expect("launch split partial");

    let mut reduce = KernargBlob::new();
    reduce.push_ptr(partials.buf.as_ptr() as *const c_void);
    reduce.push_ptr(out.buf.as_ptr() as *const c_void);
    reduce.push_i32(heads as i32);
    reduce.push_i32(D as i32);
    reduce.push_i32(n_chunks as i32);
    reduce.pad_to(16);
    gpu.launch_kernel_blob(
        SPLIT_REDUCE,
        [heads as u32, 1, 1],
        [512, 1, 1],
        0,
        reduce.as_mut_slice(),
    )
    .expect("launch split reduce");
}

#[allow(clippy::too_many_arguments)]
fn launch_gather(
    gpu: &mut Gpu,
    kv: &GpuTensor,
    indices: &GpuTensor,
    n_active: &GpuTensor,
    n_compressed: &GpuTensor,
    gathered: &GpuTensor,
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
    launch_gather(gpu, kv, indices, n_active, n_compressed, gathered);
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
    let indices: Vec<i32> = (0..TOPK).map(|i| ((i * 313 + 97) % TOPK) as i32).collect();
    let valid = [SWA as i32];
    let active = [TOPK as i32];
    let n_compressed = [N_COMPRESSED as i32];

    let d_q = gpu.upload_f32(&q, &[heads * D]).expect("q");
    let d_swa_k = gpu.upload_f32(&swa_k, &[D * SWA]).expect("swa k");
    let d_swa_v = gpu.upload_f32(&swa_v, &[D * SWA]).expect("swa v");
    let d_kv = gpu.upload_f32(&kv, &[N_COMPRESSED * D]).expect("main kv");
    let d_indices = upload_i32(gpu, &indices);
    let d_sink = gpu.upload_f32(&sink, &[heads]).expect("sink");
    let d_valid = upload_i32(gpu, &valid);
    let d_active = upload_i32(gpu, &active);
    let d_n_compressed = upload_i32(gpu, &n_compressed);
    let d_gathered = gpu.zeros(&[D * TOPK], DType::F32).expect("gathered");
    let d_gathered_out = gpu.zeros(&[heads * D], DType::F32).expect("gathered out");
    let d_direct_out = gpu.zeros(&[heads * D], DType::F32).expect("direct out");
    let max_chunks = (SWA + TOPK).div_ceil(64);
    let d_partials = gpu
        .zeros(&[heads * max_chunks * (2 + D)], DType::F32)
        .expect("split partials");
    let d_split_out = gpu.zeros(&[heads * D], DType::F32).expect("split out");

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
    let gathered_us =
        gpu.hip.event_elapsed_ms(&e0, &e1).expect("elapsed") as f64 * 1_000.0 / ITERS as f64;

    let eg0 = gpu.hip.event_create().expect("event");
    let eg1 = gpu.hip.event_create().expect("event");
    gpu.hip.event_record(&eg0, None).expect("record");
    for _ in 0..ITERS {
        launch_gather(
            gpu,
            &d_kv,
            &d_indices,
            &d_active,
            &d_n_compressed,
            &d_gathered,
        );
    }
    gpu.hip.event_record(&eg1, None).expect("record");
    gpu.hip.event_synchronize(&eg1).expect("sync");
    let gather_us =
        gpu.hip.event_elapsed_ms(&eg0, &eg1).expect("elapsed") as f64 * 1_000.0 / ITERS as f64;

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
    let direct_us =
        gpu.hip.event_elapsed_ms(&e2, &e3).expect("elapsed") as f64 * 1_000.0 / ITERS as f64;

    eprintln!(
        "H={heads}: gather={gather_us:.3} us gather+attention={gathered_us:.3} us direct={direct_us:.3} us speedup={:.4}x saved={:.3} us raw_mismatches={mismatches}/{} max_abs={max_abs:.9e}",
        gathered_us / direct_us,
        gathered_us - direct_us,
        gathered_out.len(),
    );

    for chunk_size in [64usize, 128, 256] {
        launch_split(
            gpu,
            &d_q,
            &d_swa_k,
            &d_swa_v,
            &d_gathered,
            &d_sink,
            &d_valid,
            &d_active,
            &d_partials,
            &d_split_out,
            heads,
            chunk_size,
        );
        gpu.hip
            .device_synchronize()
            .expect("split correctness sync");
        let split_out = gpu.download_f32(&d_split_out).expect("download split");
        let split_mismatches = gathered_out
            .iter()
            .zip(&split_out)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let split_max_abs = gathered_out
            .iter()
            .zip(&split_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        for _ in 0..WARMUP {
            launch_split(
                gpu,
                &d_q,
                &d_swa_k,
                &d_swa_v,
                &d_gathered,
                &d_sink,
                &d_valid,
                &d_active,
                &d_partials,
                &d_split_out,
                heads,
                chunk_size,
            );
        }
        gpu.hip.device_synchronize().expect("split warmup sync");
        let e4 = gpu.hip.event_create().expect("event");
        let e5 = gpu.hip.event_create().expect("event");
        gpu.hip.event_record(&e4, None).expect("record");
        for _ in 0..ITERS {
            launch_split(
                gpu,
                &d_q,
                &d_swa_k,
                &d_swa_v,
                &d_gathered,
                &d_sink,
                &d_valid,
                &d_active,
                &d_partials,
                &d_split_out,
                heads,
                chunk_size,
            );
        }
        gpu.hip.event_record(&e5, None).expect("record");
        gpu.hip.event_synchronize(&e5).expect("sync");
        let split_us =
            gpu.hip.event_elapsed_ms(&e4, &e5).expect("elapsed") as f64 * 1_000.0 / ITERS as f64;
        let split_plus_gather_us = split_us + gather_us;
        eprintln!(
            "H={heads} chunk={chunk_size}: split_attention={split_us:.3} us combined={split_plus_gather_us:.3} us vs current={gathered_us:.3} us speedup={:.4}x saved={:.3} us raw_mismatches={split_mismatches}/{} max_abs={split_max_abs:.9e}",
            gathered_us / split_plus_gather_us,
            gathered_us - split_plus_gather_us,
            split_out.len(),
        );
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    ensure_split_kernels(&mut gpu);
    eprintln!(
        "DS4 direct decode channel: arch={} B=1 D={D} SWA={SWA} K={TOPK} Ncompressed={N_COMPRESSED}",
        gpu.arch
    );
    let mut seed = 0xD54D_1201u32;
    run_shape(&mut gpu, 24, &mut seed);
    run_shape(&mut gpu, 16, &mut seed);
}
