// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1201 DeepSeek V4 compressed-attention score-grid screen.

use rdna_compute::{DType, Gpu, GpuTensor};

fn upload_i32(gpu: &mut Gpu, values: &[i32]) -> GpuTensor {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    gpu.upload_raw(&bytes, &[bytes.len()]).expect("upload i32")
}

fn elapsed_us<F>(gpu: &mut Gpu, repeats: usize, mut launch: F) -> f64
where
    F: FnMut(&mut Gpu),
{
    let start = gpu.hip.event_create().expect("start event");
    let stop = gpu.hip.event_create().expect("stop event");
    gpu.hip.event_record(&start, None).expect("record start");
    for _ in 0..repeats {
        launch(gpu);
    }
    gpu.hip.event_record(&stop, None).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("synchronize stop");
    gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed") as f64 * 1_000.0 / repeats as f64
}

fn run_shape(gpu: &mut Gpu, heads: usize, active_topk: usize) {
    const D: usize = 512;
    const SWA: usize = 128;
    const TOPK: usize = 512;
    const REPEATS: usize = 200;

    let mut seed = 0x1201_0731_u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((seed >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
    };
    let q = (0..heads * D).map(|_| next() * 0.125).collect::<Vec<_>>();
    let swa = (0..D * SWA).map(|_| next() * 0.125).collect::<Vec<_>>();
    let topk = (0..D * TOPK).map(|_| next() * 0.125).collect::<Vec<_>>();
    let sink = (0..heads).map(|_| next() * 0.25).collect::<Vec<_>>();

    let d_q = gpu.upload_f32(&q, &[heads, D]).expect("upload q");
    let d_swa = gpu.upload_f32(&swa, &[D, SWA]).expect("upload swa");
    let d_topk = gpu.upload_f32(&topk, &[D, TOPK]).expect("upload topk");
    let d_sink = gpu.upload_f32(&sink, &[heads]).expect("upload sink");
    let d_n_swa = upload_i32(gpu, &[SWA as i32]);
    let d_n_topk = upload_i32(gpu, &[active_topk as i32]);
    let d_reference = gpu.zeros(&[heads, D], DType::F32).expect("reference");
    let d_scoregrid = gpu.zeros(&[heads, D], DType::F32).expect("scoregrid");
    let d_split3 = gpu.zeros(&[heads, D], DType::F32).expect("split3");
    let d_split3_scratch = gpu
        .zeros(&[heads, SWA + TOPK], DType::F32)
        .expect("split3 scratch");

    let launch = |gpu: &mut Gpu, scoregrid: bool, output: &GpuTensor| {
        gpu.deepseek4_attn_swa_topk_f32_buf(
            scoregrid,
            &d_q,
            &d_swa,
            &d_swa,
            &d_topk,
            &d_topk,
            &d_sink,
            output,
            &d_n_swa,
            &d_n_topk,
            heads as i32,
            D as i32,
            SWA as i32,
            TOPK as i32,
        )
        .expect("attention launch");
    };

    launch(gpu, false, &d_reference);
    launch(gpu, true, &d_scoregrid);
    gpu.deepseek4_attn_swa_topk_split3_gfx1201(
        &d_q,
        &d_swa,
        &d_swa,
        &d_topk,
        &d_topk,
        &d_sink,
        &d_split3_scratch,
        &d_split3,
        &d_n_swa,
        &d_n_topk,
        heads as i32,
        D as i32,
        SWA as i32,
        TOPK as i32,
    )
    .expect("split3 launch");
    gpu.hip.device_synchronize().expect("correctness sync");

    let reference = gpu.download_f32(&d_reference).expect("reference output");
    let scoregrid = gpu.download_f32(&d_scoregrid).expect("scoregrid output");
    let split3 = gpu.download_f32(&d_split3).expect("split3 output");
    let mut raw_equal = 0usize;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (&expected, &actual) in reference.iter().zip(&scoregrid) {
        raw_equal += usize::from(expected.to_bits() == actual.to_bits());
        let abs = (expected - actual).abs();
        max_abs = max_abs.max(abs);
        if expected.abs() > 1.0e-6 {
            max_rel = max_rel.max(abs / expected.abs());
        }
    }
    let split3_mismatches = reference
        .iter()
        .zip(&split3)
        .filter(|(expected, actual)| expected.to_bits() != actual.to_bits())
        .count();

    for _ in 0..10 {
        launch(gpu, false, &d_reference);
        launch(gpu, true, &d_scoregrid);
        gpu.deepseek4_attn_swa_topk_split3_gfx1201(
            &d_q,
            &d_swa,
            &d_swa,
            &d_topk,
            &d_topk,
            &d_sink,
            &d_split3_scratch,
            &d_split3,
            &d_n_swa,
            &d_n_topk,
            heads as i32,
            D as i32,
            SWA as i32,
            TOPK as i32,
        )
        .expect("warm split3");
    }
    gpu.hip.device_synchronize().expect("warmup sync");

    let reference_us = elapsed_us(gpu, REPEATS, |gpu| launch(gpu, false, &d_reference));
    let scoregrid_us = elapsed_us(gpu, REPEATS, |gpu| launch(gpu, true, &d_scoregrid));
    let split3_us = elapsed_us(gpu, REPEATS, |gpu| {
        gpu.deepseek4_attn_swa_topk_split3_gfx1201(
            &d_q,
            &d_swa,
            &d_swa,
            &d_topk,
            &d_topk,
            &d_sink,
            &d_split3_scratch,
            &d_split3,
            &d_n_swa,
            &d_n_topk,
            heads as i32,
            D as i32,
            SWA as i32,
            TOPK as i32,
        )
        .expect("timed split3");
    });
    println!(
        "heads={heads} n_swa={SWA} n_topk={active_topk} topk_capacity={TOPK} raw_equal={raw_equal}/{} max_abs={max_abs:.9e} max_rel={max_rel:.9e} reference_us={reference_us:.3} scoregrid_us={scoregrid_us:.3} speedup_x={:.4} split3_us={split3_us:.3} split3_speedup_x={:.4} split3_raw_mismatches={split3_mismatches}",
        reference.len(),
        reference_us / scoregrid_us,
        reference_us / split3_us,
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx1201", "this screen requires exact gfx1201");
    for active_topk in [16, 512] {
        run_shape(&mut gpu, 24, active_topk);
        run_shape(&mut gpu, 16, active_topk);
    }
}
