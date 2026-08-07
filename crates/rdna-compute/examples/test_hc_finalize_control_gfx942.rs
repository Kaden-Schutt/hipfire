// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — raw-state admission gate for the gfx942/gfx1201 DS4 mHC finalizer.

use rdna_compute::{DType, Gpu, GpuTensor};

const HIDDEN: usize = 4096;
const GUARD: usize = 16;
const POISON: f32 = 12345.625;
const EPS: f32 = 1.0e-6;
const POST_SCALE: f32 = 1.5;
const SINKHORN_ITERS: i32 = 20;

fn upload_guarded_f16(gpu: &Gpu, values: &[u16]) -> (GpuTensor, GpuTensor) {
    const GUARD_BITS: u16 = 0x55aa;
    let mut words = vec![GUARD_BITS; values.len() + 2 * GUARD];
    words[GUARD..GUARD + values.len()].copy_from_slice(values);
    let bytes = unsafe {
        std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * size_of::<u16>())
    };
    let backing = gpu
        .upload_raw(bytes, &[bytes.len()])
        .expect("upload guarded f16 bytes");
    let mut view = backing.sub_offset(GUARD * size_of::<u16>(), values.len() * 2);
    view.shape = vec![values.len()];
    view.dtype = DType::F16;
    (backing, view)
}

fn download_bytes(gpu: &Gpu, tensor: &GpuTensor) -> Vec<u8> {
    let mut bytes = vec![0u8; tensor.byte_size()];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &tensor.buf)
        .expect("download raw bytes");
    bytes
}

fn make_streams(case: usize) -> Vec<f32> {
    (0..4 * HIDDEN)
        .map(|i| {
            let lane = i % HIDDEN;
            let stream = i / HIDDEN;
            if case == 0 {
                (((lane * 17 + stream * 31) % 257) as f32 - 128.0) * 0.000_976_562_5
            } else {
                match (lane + stream) & 7 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => 1.0,
                    3 => -1.0,
                    4 => 16.0,
                    5 => -16.0,
                    _ => 0.125,
                }
            }
        })
        .collect()
}

fn make_c(case: usize) -> Vec<f32> {
    (0..24)
        .map(|i| {
            if case == 0 {
                (i as f32 - 11.5) * 0.1875
            } else {
                [-32.0, -16.0, -1.0, -0.0, 0.0, 1.0, 16.0, 32.0][i & 7]
            }
        })
        .collect()
}

fn assert_f32_bits(label: &str, incumbent: &[f32], candidate: &[f32]) {
    let mismatch = incumbent
        .iter()
        .zip(candidate)
        .enumerate()
        .find_map(|(i, (a, b))| {
            (a.to_bits() != b.to_bits()).then_some((i, a.to_bits(), b.to_bits()))
        });
    assert!(
        mismatch.is_none(),
        "{label} raw-bit mismatch (index, incumbent, candidate): {mismatch:?}"
    );
}

fn assert_f32_guards(label: &str, values: &[f32], payload: usize) {
    assert!(
        values[..GUARD]
            .iter()
            .chain(&values[GUARD + payload..])
            .all(|v| v.to_bits() == POISON.to_bits()),
        "{label} guard overwritten"
    );
}

fn run_case(gpu: &mut Gpu, case: usize) {
    let alpha_bits = if case == 0 {
        [0x3000, 0x3800, 0x3b00]
    } else {
        [0x3c00, 0x3400, 0x0000]
    };
    let base_bits: Vec<u16> = (0..24)
        .map(|i| {
            if case == 0 {
                [0xbc00, 0xb800, 0x0000, 0x3400, 0x3800, 0x3c00][i % 6]
            } else {
                [0xc000, 0xbc00, 0x8000, 0x0000, 0x3c00, 0x4000][i % 6]
            }
        })
        .collect();
    let (alpha_backing, alpha) = upload_guarded_f16(gpu, &alpha_bits);
    let (base_backing, base) = upload_guarded_f16(gpu, &base_bits);
    let alpha_before = download_bytes(gpu, &alpha_backing);
    let base_before = download_bytes(gpu, &base_backing);

    let streams_host = make_streams(case);
    let streams = gpu
        .upload_f32(&streams_host, &[4, HIDDEN])
        .expect("upload streams");

    let mut guarded_c = vec![POISON; 24 + 2 * GUARD];
    guarded_c[GUARD..GUARD + 24].copy_from_slice(&make_c(case));
    let incumbent_c_backing = gpu
        .upload_f32(&guarded_c, &[guarded_c.len()])
        .expect("upload incumbent c");
    let candidate_c_backing = gpu
        .upload_f32(&guarded_c, &[guarded_c.len()])
        .expect("upload candidate c");
    let incumbent_c = incumbent_c_backing.sub_offset(GUARD, 24);
    let candidate_c = candidate_c_backing.sub_offset(GUARD, 24);

    let guarded_out = vec![POISON; HIDDEN + 2 * GUARD];
    let incumbent_out_backing = gpu
        .upload_f32(&guarded_out, &[guarded_out.len()])
        .expect("upload incumbent output");
    let candidate_out_backing = gpu
        .upload_f32(&guarded_out, &[guarded_out.len()])
        .expect("upload candidate output");
    let incumbent_out = incumbent_out_backing.sub_offset(GUARD, HIDDEN);
    let candidate_out = candidate_out_backing.sub_offset(GUARD, HIDDEN);

    gpu.hc_apply_alpha(&incumbent_c, &alpha, &base)
        .expect("incumbent alpha");
    gpu.hc_pre_post_sigmoid_scale_f32(&incumbent_c, EPS, POST_SCALE)
        .expect("incumbent sigmoid");
    gpu.hc_sinkhorn_4x4(&incumbent_c.sub_offset(8, 16), EPS, SINKHORN_ITERS)
        .expect("incumbent sinkhorn");
    gpu.hc_input_map_4stream(
        &incumbent_c.sub_offset(0, 4),
        &streams,
        &incumbent_out,
        HIDDEN as i32,
    )
    .expect("incumbent input map");

    gpu.hc_finalize_control(&candidate_c, &alpha, &base, EPS, POST_SCALE, SINKHORN_ITERS)
        .expect("candidate finalizer");
    gpu.hc_input_map_4stream(
        &candidate_c.sub_offset(0, 4),
        &streams,
        &candidate_out,
        HIDDEN as i32,
    )
    .expect("candidate input map");
    gpu.hip.device_synchronize().expect("parity sync");

    let incumbent_c_host = gpu
        .download_f32(&incumbent_c)
        .expect("download incumbent c");
    let candidate_c_host = gpu
        .download_f32(&candidate_c)
        .expect("download candidate c");
    assert_f32_bits("hc_c", &incumbent_c_host, &candidate_c_host);
    let incumbent_out_host = gpu
        .download_f32(&incumbent_out)
        .expect("download incumbent output");
    let candidate_out_host = gpu
        .download_f32(&candidate_out)
        .expect("download candidate output");
    assert_f32_bits("hc_x_in", &incumbent_out_host, &candidate_out_host);

    let incumbent_c_all = gpu
        .download_f32(&incumbent_c_backing)
        .expect("download incumbent guarded c");
    let candidate_c_all = gpu
        .download_f32(&candidate_c_backing)
        .expect("download candidate guarded c");
    let incumbent_out_all = gpu
        .download_f32(&incumbent_out_backing)
        .expect("download incumbent guarded output");
    let candidate_out_all = gpu
        .download_f32(&candidate_out_backing)
        .expect("download candidate guarded output");
    assert_f32_guards("incumbent c", &incumbent_c_all, 24);
    assert_f32_guards("candidate c", &candidate_c_all, 24);
    assert_f32_guards("incumbent output", &incumbent_out_all, HIDDEN);
    assert_f32_guards("candidate output", &candidate_out_all, HIDDEN);

    assert_eq!(
        download_bytes(gpu, &alpha_backing),
        alpha_before,
        "alpha mutated"
    );
    assert_eq!(
        download_bytes(gpu, &base_backing),
        base_before,
        "base mutated"
    );
    assert_f32_bits(
        "streams input",
        &streams_host,
        &gpu.download_f32(&streams).expect("download streams"),
    );
    println!(
        "PASS case={case}: hc_c 24/24 and hc_x_in {HIDDEN}/{HIDDEN} raw-bit identical; guards and inputs intact"
    );
}

fn elapsed_ms<F>(gpu: &mut Gpu, repeats: usize, mut launch: F) -> f32
where
    F: FnMut(&mut Gpu),
{
    let start = gpu.hip.event_create().expect("create start event");
    let stop = gpu.hip.event_create().expect("create stop event");
    gpu.hip.event_record(&start, None).expect("record start");
    for _ in 0..repeats {
        launch(gpu);
    }
    gpu.hip.event_record(&stop, None).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("wait stop");
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed ms") / repeats as f32;
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    ms
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert!(
        matches!(gpu.arch.as_str(), "gfx942" | "gfx1201"),
        "this admission gate requires gfx942 or gfx1201, got {}",
        gpu.arch
    );
    run_case(&mut gpu, 0);
    run_case(&mut gpu, 1);

    let c_values = make_c(0);
    let alpha_bits = [0x3000, 0x3800, 0x3b00];
    let base_bits: Vec<u16> = (0..24)
        .map(|i| [0xbc00, 0xb800, 0x0000, 0x3400, 0x3800, 0x3c00][i % 6])
        .collect();
    let (_, alpha) = upload_guarded_f16(&gpu, &alpha_bits);
    let (_, base) = upload_guarded_f16(&gpu, &base_bits);
    let incumbent = gpu.upload_f32(&c_values, &[24]).expect("timed incumbent c");
    let candidate = gpu.upload_f32(&c_values, &[24]).expect("timed candidate c");
    const REPEATS: usize = 512;
    let incumbent_ms = elapsed_ms(&mut gpu, REPEATS, |gpu| {
        gpu.hc_apply_alpha(&incumbent, &alpha, &base).unwrap();
        gpu.hc_pre_post_sigmoid_scale_f32(&incumbent, EPS, POST_SCALE)
            .unwrap();
        gpu.hc_sinkhorn_4x4(&incumbent.sub_offset(8, 16), EPS, SINKHORN_ITERS)
            .unwrap();
    });
    let candidate_ms = elapsed_ms(&mut gpu, REPEATS, |gpu| {
        gpu.hc_finalize_control(&candidate, &alpha, &base, EPS, POST_SCALE, SINKHORN_ITERS)
            .unwrap();
    });
    println!(
        "MICRO incumbent_3_launch_ms={incumbent_ms:.6} fused_1_launch_ms={candidate_ms:.6} speedup={:.3}x launches_saved=2",
        incumbent_ms / candidate_ms
    );
}
