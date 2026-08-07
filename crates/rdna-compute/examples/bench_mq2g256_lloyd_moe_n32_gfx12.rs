// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Product-shape screen for the gfx1201 MQ2-Lloyd grouped N=32 tile pair.
//!
//! Models B=1024, top-k=6 and the 256-expert padding contract exactly:
//! `m_total_max = 1024*6 + 256*16 = 10240`. Expert pointers alias one
//! deterministic weight matrix, while tile ids and padding preserve the
//! production same-expert/boundary mix.

use rdna_compute::{DType, Gpu, GpuTensor};

const BATCH: usize = 1024;
const TOP_K: usize = 6;
const EXPERTS: usize = 256;
const M_TOTAL: usize = BATCH * TOP_K + EXPERTS * 16;

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = (((bits >> 23) & 0xff) as i32) - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent <= 0 {
        return sign;
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
}

fn bytes_i32(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bytes_u64(values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bytes_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn wrap(raw: *mut std::ffi::c_void, bytes: usize, shape: Vec<usize>, dtype: DType) -> GpuTensor {
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(raw, bytes) },
        shape,
        dtype,
    }
}

fn build_weights(rows: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * (k / 256) * 72);
    let mut state = 0xc0ff_ee11_u32;
    for row in 0..rows {
        for group in 0..k / 256 {
            for value in [-3.0_f32, -1.0, 1.0, 3.0] {
                out.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
            }
            for byte in 0..64 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let mut packed = 0_u8;
                for index in 0..4 {
                    let code = ((state >> (index * 3)) as usize + row + group + byte) & 3;
                    packed |= (code as u8) << (2 * index);
                }
                out.push(packed);
            }
        }
    }
    out
}

fn product_tiles(x_row_div: usize) -> (Vec<i32>, Vec<i32>) {
    let mut tile_ids = Vec::with_capacity(M_TOTAL / 16);
    let mut slots = Vec::with_capacity(M_TOTAL);
    let mut flat = 0_i32;
    for expert in 0..EXPERTS {
        // 128 experts own two tiles and 128 own three: 640 total tiles, the
        // exact B1024 launch geometry. Each owns 24 live routed slots.
        let tile_count = if expert & 1 == 0 { 2 } else { 3 };
        for _ in 0..tile_count {
            tile_ids.push(expert as i32);
        }
        for local in 0..tile_count * 16 {
            if local < 24 {
                slots.push(if x_row_div > 1 {
                    // Preserve flat token/top-k encoding for gate/up.
                    (flat / TOP_K as i32) * TOP_K as i32 + flat % TOP_K as i32
                } else {
                    flat
                });
                flat += 1;
            } else {
                slots.push(-1);
            }
        }
    }
    assert_eq!(tile_ids.len(), M_TOTAL / 16);
    assert_eq!(slots.len(), M_TOTAL);
    assert_eq!(flat as usize, BATCH * TOP_K);
    (tile_ids, slots)
}

fn run_case(gpu: &mut Gpu, m: usize, k: usize, x_row_div: usize, rows: usize, label: &str) {
    let weights = build_weights(m, k);
    let x: Vec<f32> = (0..rows * k)
        .map(|index| ((index % 29) as f32 - 14.0) / 32.0)
        .collect();
    let (tile_ids, slots) = product_tiles(x_row_div);

    let d_weight = gpu.hip.malloc(weights.len()).unwrap();
    gpu.hip.memcpy_htod(&d_weight, &weights).unwrap();
    let pointer_values = vec![d_weight.as_ptr() as u64; EXPERTS];
    let pointer_bytes = bytes_u64(&pointer_values);
    let tile_bytes = bytes_i32(&tile_ids);
    let slot_bytes = bytes_i32(&slots);
    let x_bytes = bytes_f32(&x);

    let d_ptrs = gpu.hip.malloc(pointer_bytes.len()).unwrap();
    let d_tiles = gpu.hip.malloc(tile_bytes.len()).unwrap();
    let d_slots = gpu.hip.malloc(slot_bytes.len()).unwrap();
    let d_x = gpu.hip.malloc(x_bytes.len()).unwrap();
    let output_bytes = M_TOTAL * m * 4;
    let d_ref = gpu.hip.malloc(output_bytes).unwrap();
    let d_n32 = gpu.hip.malloc(output_bytes).unwrap();
    gpu.hip.memcpy_htod(&d_ptrs, &pointer_bytes).unwrap();
    gpu.hip.memcpy_htod(&d_tiles, &tile_bytes).unwrap();
    gpu.hip.memcpy_htod(&d_slots, &slot_bytes).unwrap();
    gpu.hip.memcpy_htod(&d_x, &x_bytes).unwrap();

    let ptrs = wrap(
        d_ptrs.as_ptr(),
        pointer_bytes.len(),
        vec![EXPERTS],
        DType::Raw,
    );
    let tiles = wrap(
        d_tiles.as_ptr(),
        tile_bytes.len(),
        vec![M_TOTAL / 16],
        DType::Raw,
    );
    let slot_index = wrap(
        d_slots.as_ptr(),
        slot_bytes.len(),
        vec![M_TOTAL],
        DType::Raw,
    );
    let x_tensor = wrap(d_x.as_ptr(), x_bytes.len(), vec![rows, k], DType::F32);
    let reference = wrap(d_ref.as_ptr(), output_bytes, vec![M_TOTAL, m], DType::F32);
    let candidate = wrap(d_n32.as_ptr(), output_bytes, vec![M_TOTAL, m], DType::F32);

    let incumbent = |gpu: &mut Gpu| {
        gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2(
            &ptrs,
            &tiles,
            &slot_index,
            &x_tensor,
            &reference,
            m,
            k,
            x_row_div,
            M_TOTAL,
            rows,
        )
        .unwrap();
    };
    let n32 = |gpu: &mut Gpu| {
        gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32(
            &ptrs,
            &tiles,
            &slot_index,
            &x_tensor,
            &candidate,
            m,
            k,
            x_row_div,
            M_TOTAL,
            rows,
        )
        .unwrap();
    };

    incumbent(gpu);
    n32(gpu);
    gpu.hip.device_synchronize().unwrap();
    let mut ref_bytes = vec![0_u8; output_bytes];
    let mut n32_bytes = vec![0_u8; output_bytes];
    gpu.hip.memcpy_dtoh(&mut ref_bytes, &d_ref).unwrap();
    gpu.hip.memcpy_dtoh(&mut n32_bytes, &d_n32).unwrap();
    let mismatches = ref_bytes
        .chunks_exact(4)
        .zip(n32_bytes.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(mismatches, 0, "{label}: raw-bit mismatch");

    for _ in 0..3 {
        incumbent(gpu);
        n32(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let time = |gpu: &mut Gpu, launch: &dyn Fn(&mut Gpu)| -> f64 {
        let start = gpu.hip.event_create().unwrap();
        let stop = gpu.hip.event_create().unwrap();
        gpu.hip.event_record(&start, None).unwrap();
        for _ in 0..10 {
            launch(gpu);
        }
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 * 100.0
    };
    let incumbent_us = time(gpu, &incumbent);
    let n32_us = time(gpu, &n32);
    println!(
        "{label}: M={m} K={k} m_total={M_TOTAL} incumbent={incumbent_us:.3}us n32={n32_us:.3}us speedup={:.3}x raw_mismatches={mismatches}",
        incumbent_us / n32_us
    );

    std::mem::forget(ptrs);
    std::mem::forget(tiles);
    std::mem::forget(slot_index);
    std::mem::forget(x_tensor);
    std::mem::forget(reference);
    std::mem::forget(candidate);
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert!(gpu.arch_caps.has_wmma_w32_gfx12(), "requires gfx12");
    println!("arch={}", gpu.arch);
    run_case(&mut gpu, 4096, 4096, TOP_K, BATCH, "gate_up");
    run_case(&mut gpu, 4096, 2048, 1, BATCH * TOP_K, "down");
}
