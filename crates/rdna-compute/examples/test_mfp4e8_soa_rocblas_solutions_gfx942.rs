// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — channel-only exact-gfx942 rocBLAS solution tournament.

use hip_bridge::{Rocblas, RocblasDatatype, RocblasOperation, RocblasResult};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

const GUARD: usize = 64;
const POISON_BITS: u32 = 0x4f12_3456;

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn bytes_of_f32(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn build_e8_soa(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let blocks = k / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    let row_bytes = 16 + scale_padded + blocks * 16;
    let mut data = vec![0u8; m * row_bytes];
    let mut state = seed;
    const ROW_SCALES: [u16; 4] = [0x3400, 0x3800, 0x3c00, 0x4000];
    for row in 0..m {
        let off = row * row_bytes;
        data[off..off + 2].copy_from_slice(&ROW_SCALES[row % 4].to_le_bytes());
        data[off + 4..off + 6].copy_from_slice(&(blocks as u16).to_le_bytes());
        data[off + 6] = 0x06;
        for block in 0..blocks {
            data[off + 16 + block] = 0x38 | (lcg(&mut state) as u8 & 7);
            let cw_off = off + 16 + scale_padded + block * 16;
            for slot in 0..4 {
                data[cw_off + slot * 4..cw_off + slot * 4 + 4]
                    .copy_from_slice(&lcg(&mut state).to_le_bytes());
            }
        }
    }
    data
}

fn upload_weight(gpu: &Gpu, packed: &[u8], m: usize, k: usize) -> GpuTensor {
    let mut weight = gpu.upload_raw(packed, &[packed.len()]).expect("upload weight");
    weight.shape = vec![m, k];
    weight.dtype = DType::MFP4G32E8SOA;
    weight
}

fn make_x(k: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..k)
        .map(|_| (lcg(&mut state) as f32 / u32::MAX as f32 - 0.5) * 0.5)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn rocblas_call(
    rb: &Rocblas,
    shadow: &GpuTensor,
    x_f16: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    solution: Option<i32>,
) -> RocblasResult<()> {
    let alpha = 1.0f32;
    let beta = 0.0f32;
    unsafe {
        match solution {
            Some(id) => rb.gemm_ex_with_solution(
                RocblasOperation::Transpose,
                RocblasOperation::None,
                m as i32,
                1,
                k as i32,
                (&alpha as *const f32).cast::<c_void>(),
                shadow.buf.as_ptr(),
                RocblasDatatype::F16,
                k as i32,
                x_f16.buf.as_ptr(),
                RocblasDatatype::F16,
                k as i32,
                (&beta as *const f32).cast::<c_void>(),
                y.buf.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                y.buf.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                RocblasDatatype::F32,
                id,
            ),
            None => rb.gemm_ex(
                RocblasOperation::Transpose,
                RocblasOperation::None,
                m as i32,
                1,
                k as i32,
                (&alpha as *const f32).cast::<c_void>(),
                shadow.buf.as_ptr(),
                RocblasDatatype::F16,
                k as i32,
                x_f16.buf.as_ptr(),
                RocblasDatatype::F16,
                k as i32,
                (&beta as *const f32).cast::<c_void>(),
                y.buf.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                y.buf.as_ptr(),
                RocblasDatatype::F32,
                m as i32,
                RocblasDatatype::F32,
            ),
        }
    }
}

fn enumerate(
    rb: &Rocblas,
    shadow: &GpuTensor,
    x_f16: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
) -> Vec<i32> {
    let alpha = 1.0f32;
    let beta = 0.0f32;
    unsafe {
        rb.enumerate_gemm_ex_solutions(
            RocblasOperation::Transpose,
            RocblasOperation::None,
            m as i32,
            1,
            k as i32,
            (&alpha as *const f32).cast::<c_void>(),
            shadow.buf.as_ptr(),
            RocblasDatatype::F16,
            k as i32,
            x_f16.buf.as_ptr(),
            RocblasDatatype::F16,
            k as i32,
            (&beta as *const f32).cast::<c_void>(),
            y.buf.as_ptr(),
            RocblasDatatype::F32,
            m as i32,
            y.buf.as_ptr(),
            RocblasDatatype::F32,
            m as i32,
            RocblasDatatype::F32,
        )
        .expect("enumerate exact rocBLAS solutions")
        .expect("ROCm library lacks rocblas_gemm_ex_get_solutions")
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn time_rocblas(
    gpu: &Gpu,
    shadow: &GpuTensor,
    x_f16: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    solution: Option<i32>,
    repeats: usize,
) -> Result<f64, String> {
    let rb = gpu.rocblas.as_ref().unwrap();
    rocblas_call(rb, shadow, x_f16, y, m, k, solution).map_err(|e| e.to_string())?;
    gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
    let mut trials = Vec::with_capacity(3);
    for _ in 0..3 {
        let start = gpu.hip.event_create().map_err(|e| e.to_string())?;
        let stop = gpu.hip.event_create().map_err(|e| e.to_string())?;
        gpu.hip.event_record(&start, None).map_err(|e| e.to_string())?;
        for _ in 0..repeats {
            rocblas_call(rb, shadow, x_f16, y, m, k, solution)
                .map_err(|e| e.to_string())?;
        }
        gpu.hip.event_record(&stop, None).map_err(|e| e.to_string())?;
        gpu.hip.event_synchronize(&stop).map_err(|e| e.to_string())?;
        trials.push(gpu.hip.event_elapsed_ms(&start, &stop).map_err(|e| e.to_string())? as f64 / repeats as f64);
        gpu.hip.event_destroy(start).map_err(|e| e.to_string())?;
        gpu.hip.event_destroy(stop).map_err(|e| e.to_string())?;
    }
    Ok(median(trials))
}

fn time_compressed(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    repeats: usize,
) -> f64 {
    gpu.gemv_mfp4g32_e8_soa(weight, x, y, m, k).expect("warm compressed");
    gpu.hip.device_synchronize().expect("warm sync");
    let mut trials = Vec::with_capacity(3);
    for _ in 0..3 {
        let start = gpu.hip.event_create().unwrap();
        let stop = gpu.hip.event_create().unwrap();
        gpu.hip.event_record(&start, None).unwrap();
        for _ in 0..repeats {
            gpu.gemv_mfp4g32_e8_soa(weight, x, y, m, k).expect("compressed");
        }
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        trials.push(gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 / repeats as f64);
        gpu.hip.event_destroy(start).unwrap();
        gpu.hip.event_destroy(stop).unwrap();
    }
    median(trials)
}

fn validate(candidate: &[f32], reference: &[f32], guard: &[f32]) -> Result<(f64, f64), String> {
    if guard.iter().any(|v| v.to_bits() != POISON_BITS) {
        return Err("output guard modified".into());
    }
    let mut max_abs = 0.0f64;
    let mut err2 = 0.0f64;
    let mut ref2 = 0.0f64;
    for (&got, &want) in candidate.iter().zip(reference) {
        if !got.is_finite() || !want.is_finite() {
            return Err("nonfinite output".into());
        }
        let delta = got as f64 - want as f64;
        max_abs = max_abs.max(delta.abs());
        err2 += delta * delta;
        ref2 += (want as f64) * (want as f64);
    }
    let rel_l2 = (err2 / ref2.max(1.0e-30)).sqrt();
    if max_abs > 0.1 || rel_l2 > 1.0e-3 {
        return Err(format!("numerics max_abs={max_abs:.6e} rel_l2={rel_l2:.6e}"));
    }
    Ok((max_abs, rel_l2))
}

fn run_shape(gpu: &mut Gpu, m: usize, k: usize, seed: u64) {
    let packed = build_e8_soa(m, k, seed);
    let weight = upload_weight(gpu, &packed, m, k);
    let shadow = gpu.alloc_tensor(&[m * k], DType::F16).expect("FP16 shadow");
    gpu.dequantize_mfp4g32_e8_soa_to_f16_gfx942(&weight.buf, &shadow.buf, m, k)
        .expect("dequantize once");
    let x_host = make_x(k, seed ^ 0xa5a5);
    let x = gpu.upload_f32(&x_host, &[k]).expect("x");
    let x_f16 = gpu.alloc_tensor(&[k], DType::F16).expect("x f16");
    gpu.deepseek4_convert_f32_to_f16(&x, &x_f16, k as i64).expect("x f16 convert");
    let reference = gpu.zeros(&[m], DType::F32).expect("reference");
    gpu.gemv_mfp4g32_e8_soa(&weight, &x, &reference, m, k).expect("reference");
    gpu.hip.device_synchronize().expect("reference sync");
    let reference_host = gpu.download_f32(&reference).expect("reference download");

    let poison = f32::from_bits(POISON_BITS);
    let poison_host = vec![poison; m + GUARD];
    let backing = gpu.upload_f32(&poison_host, &[m + GUARD]).expect("guarded output");
    let y = backing.sub_offset(0, m);
    let ids = enumerate(gpu.rocblas.as_ref().unwrap(), &shadow, &x_f16, &y, m, k);
    println!("ENUM M={m} K={k} count={} ids={ids:?}", ids.len());
    let expanded_bytes = m * k * 2 + k * 2 + m * 4;
    let compressed_bytes = packed.len() + k * 4 + m * 4;
    let repeats = (512_000_000usize / expanded_bytes.max(1)).clamp(1, 64);
    let compressed_ms = time_compressed(gpu, &weight, &x, &reference, m, k, repeats);
    println!("RESULT M={m} K={k} route=compressed ms={compressed_ms:.6} model_bytes={compressed_bytes} gbps={:.3}", compressed_bytes as f64 / compressed_ms / 1.0e6);

    let mut choices = Vec::new();
    choices.push(("default".to_string(), None));
    choices.extend(ids.into_iter().filter(|id| *id > 0).map(|id| (format!("solution:{id}"), Some(id))));
    for (label, solution) in choices {
        gpu.hip.memcpy_htod(&backing.buf, bytes_of_f32(&poison_host)).expect("reset guard");
        let call = rocblas_call(gpu.rocblas.as_ref().unwrap(), &shadow, &x_f16, &y, m, k, solution);
        if let Err(error) = call {
            println!("REJECT M={m} K={k} route={label} reason=launch status={} context={:?}", error.status, error.context);
            continue;
        }
        gpu.hip.device_synchronize().expect("candidate sync");
        let got = gpu.download_f32(&backing).expect("candidate download");
        match validate(&got[..m], &reference_host, &got[m..]) {
            Ok((max_abs, rel_l2)) => match time_rocblas(gpu, &shadow, &x_f16, &y, m, k, solution, repeats) {
                Ok(ms) => println!("RESULT M={m} K={k} route={label} ms={ms:.6} model_bytes={expanded_bytes} gbps={:.3} speedup_vs_compressed={:.6} max_abs={max_abs:.6e} rel_l2={rel_l2:.6e}", expanded_bytes as f64 / ms / 1.0e6, compressed_ms / ms),
                Err(error) => println!("REJECT M={m} K={k} route={label} reason=timing error={error:?}"),
            },
            Err(error) => println!("REJECT M={m} K={k} route={label} reason={error:?}"),
        }
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx942", "exact-gfx942 channel only");
    assert!(gpu.rocblas.as_ref().is_some_and(Rocblas::has_gemm_ex_solution_enumeration), "ROCm rocBLAS solution enumeration required");
    run_shape(&mut gpu, 1024, 4096, 0x9421);
    run_shape(&mut gpu, 32768, 1024, 0x9422);
}
