// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Micro-screen for removing repeated MQ2-Lloyd codebook quantization from
//! the gfx1151 grouped prefill GEMM. Production dispatch is not changed.
//!
//! The baseline consumes the shipping 72-byte group
//! `[4 x f16 codebook | 64 B indices]`. The candidate consumes an equal-size
//! derived group `[4 x int8 codebook | f32 scale | 64 B indices]`.
//!
//! Route-faithful screen shapes:
//! - gate_up: M=4096, K=4096, B=1024, top-k=6, x_row_div=6
//! - down:    M=4096, K=2048, B=1024, top-k=6, x_row_div=1
//! - 256 distinct expert address ranges, balanced over 384 slot tiles

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const EXPERTS: usize = 256;
const TOP_K: usize = 6;
const BATCH: usize = 1024;
const WARMUP: usize = 4;
const TRIALS_PER_ARM: usize = 10;
const ATOL: f32 = 1.0e-5;
const RTOL: f32 = 1.0e-5;

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (((bits >> 23) & 0xff) as i32) - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

fn expert_weight_pair(k: usize, rows: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(k % 256, 0);
    let groups = rows * (k / 256);
    let mut shipping = Vec::with_capacity(groups * 72);
    let mut prequant = Vec::with_capacity(groups * 72);
    let codebook = [-3.0f32, -1.0, 1.0, 3.0];
    let scale = 3.0f32 / 127.0;
    let q = [-127i8, -42, 42, 127];
    let mut rng = seed;

    for _ in 0..groups {
        for &v in &codebook {
            shipping.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
        }
        for &v in &q {
            prequant.push(v as u8);
        }
        prequant.extend_from_slice(&scale.to_le_bytes());

        for _ in 0..64 {
            let mut packed = 0u8;
            for lane in 0..4 {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                packed |= (((rng >> 48) & 3) as u8) << (2 * lane);
            }
            shipping.push(packed);
            prequant.push(packed);
        }
    }
    (shipping, prequant)
}

fn wrap_buf(
    raw_ptr: *mut std::ffi::c_void,
    bytes: usize,
    shape: Vec<usize>,
    dtype: DType,
) -> GpuTensor {
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(raw_ptr, bytes) },
        shape,
        dtype,
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    gpu: &mut Gpu,
    candidate: bool,
    expert_ptrs: &GpuTensor,
    tile_ids: &GpuTensor,
    slot_index: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    x_row_div: usize,
    m_total: usize,
    x_rows: usize,
) {
    if candidate {
        gpu.gemm_mq2g256_lloyd_moe_grouped_mmq_prequant_perm_gfx1151(
            expert_ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total,
            x_rows,
        )
        .expect("prequant candidate");
    } else {
        gpu.gemm_mq2g256_lloyd_moe_grouped_mmq_perm_gfx1151(
            expert_ptrs,
            tile_ids,
            slot_index,
            x,
            y,
            m,
            k,
            x_row_div,
            m_total,
            x_rows,
        )
        .expect("shipping perm baseline");
    }
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(f64::total_cmp);
    let mid = xs.len() / 2;
    if xs.len() % 2 == 0 {
        (xs[mid - 1] + xs[mid]) * 0.5
    } else {
        xs[mid]
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch={} expected=gfx1151", gpu.arch);
    assert_eq!(gpu.arch, "gfx1151", "this micro-screen is gfx1151-only");

    let shapes = [
        (4096usize, 4096usize, TOP_K, BATCH, "gate_up"),
        (4096usize, 2048usize, 1usize, BATCH * TOP_K, "down"),
    ];

    for (m, k, x_row_div, x_rows, label) in shapes {
        let m_total = BATCH * TOP_K;
        let slot_tiles = m_total.div_ceil(16);
        println!(
            "\n{label}: M={m} K={k} B={BATCH} top_k={TOP_K} \
             m_total={m_total} experts={EXPERTS} x_rows={x_rows} x_row_div={x_row_div}"
        );

        let (shipping_expert, prequant_expert) = expert_weight_pair(k, m, 0x8bad_f00d_dead_beef);
        let expert_bytes = shipping_expert.len();
        let shipping_weights = shipping_expert.repeat(EXPERTS);
        let prequant_weights = prequant_expert.repeat(EXPERTS);
        println!(
            "weight_bytes_per_arm={} ({:.3} GiB), group_stride=72 unchanged",
            shipping_weights.len(),
            shipping_weights.len() as f64 / (1u64 << 30) as f64
        );

        let w_shipping_gpu = gpu
            .hip
            .malloc(shipping_weights.len())
            .expect("malloc shipping weights");
        let w_prequant_gpu = gpu
            .hip
            .malloc(prequant_weights.len())
            .expect("malloc prequant weights");
        gpu.hip
            .memcpy_htod(&w_shipping_gpu, &shipping_weights)
            .expect("upload shipping weights");
        gpu.hip
            .memcpy_htod(&w_prequant_gpu, &prequant_weights)
            .expect("upload prequant weights");

        let shipping_base = w_shipping_gpu.as_ptr() as usize;
        let prequant_base = w_prequant_gpu.as_ptr() as usize;
        let shipping_ptr_bytes: Vec<u8> = (0..EXPERTS)
            .flat_map(|e| ((shipping_base + e * expert_bytes) as u64).to_le_bytes())
            .collect();
        let prequant_ptr_bytes: Vec<u8> = (0..EXPERTS)
            .flat_map(|e| ((prequant_base + e * expert_bytes) as u64).to_le_bytes())
            .collect();
        let ep_shipping_gpu = gpu.hip.malloc(EXPERTS * 8).expect("malloc ep shipping");
        let ep_prequant_gpu = gpu.hip.malloc(EXPERTS * 8).expect("malloc ep prequant");
        gpu.hip
            .memcpy_htod(&ep_shipping_gpu, &shipping_ptr_bytes)
            .expect("upload shipping ptrs");
        gpu.hip
            .memcpy_htod(&ep_prequant_gpu, &prequant_ptr_bytes)
            .expect("upload prequant ptrs");

        // Production scatter groups tiles by expert. Balance 384 slot tiles
        // over 256 expert address ranges: 1–2 tiles per expert.
        let tile_ids_bytes: Vec<u8> = (0..slot_tiles)
            .flat_map(|tile| ((tile * EXPERTS / slot_tiles) as i32).to_le_bytes())
            .collect();
        let tile_ids_gpu = gpu
            .hip
            .malloc(tile_ids_bytes.len())
            .expect("malloc tile ids");
        gpu.hip
            .memcpy_htod(&tile_ids_gpu, &tile_ids_bytes)
            .expect("upload tile ids");

        let slot_index_bytes: Vec<u8> = (0..m_total)
            .flat_map(|i| (i as i32).to_le_bytes())
            .collect();
        let slot_index_gpu = gpu
            .hip
            .malloc(slot_index_bytes.len())
            .expect("malloc slot index");
        gpu.hip
            .memcpy_htod(&slot_index_gpu, &slot_index_bytes)
            .expect("upload slot index");

        let x_f32: Vec<f32> = (0..x_rows * k)
            .map(|i| ((i.wrapping_mul(17) % 29) as f32 - 14.0) / 13.0)
            .collect();
        let x_bytes: Vec<u8> = x_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x_gpu = gpu.hip.malloc(x_bytes.len()).expect("malloc x");
        gpu.hip.memcpy_htod(&x_gpu, &x_bytes).expect("upload x");
        let y_bytes = m_total * m * 4;
        let y_shipping_gpu = gpu.hip.malloc(y_bytes).expect("malloc y shipping");
        let y_prequant_gpu = gpu.hip.malloc(y_bytes).expect("malloc y prequant");

        let ep_shipping = wrap_buf(
            ep_shipping_gpu.as_ptr(),
            EXPERTS * 8,
            vec![EXPERTS],
            DType::F32,
        );
        let ep_prequant = wrap_buf(
            ep_prequant_gpu.as_ptr(),
            EXPERTS * 8,
            vec![EXPERTS],
            DType::F32,
        );
        let tile_ids = wrap_buf(
            tile_ids_gpu.as_ptr(),
            tile_ids_bytes.len(),
            vec![slot_tiles],
            DType::F32,
        );
        let slot_index = wrap_buf(
            slot_index_gpu.as_ptr(),
            slot_index_bytes.len(),
            vec![m_total],
            DType::F32,
        );
        let x = wrap_buf(x_gpu.as_ptr(), x_bytes.len(), vec![x_rows, k], DType::F32);
        let y_shipping = wrap_buf(
            y_shipping_gpu.as_ptr(),
            y_bytes,
            vec![m_total, m],
            DType::F32,
        );
        let y_prequant = wrap_buf(
            y_prequant_gpu.as_ptr(),
            y_bytes,
            vec![m_total, m],
            DType::F32,
        );

        for candidate in [false, true] {
            for _ in 0..WARMUP {
                dispatch(
                    &mut gpu,
                    candidate,
                    if candidate {
                        &ep_prequant
                    } else {
                        &ep_shipping
                    },
                    &tile_ids,
                    &slot_index,
                    &x,
                    if candidate { &y_prequant } else { &y_shipping },
                    m,
                    k,
                    x_row_div,
                    m_total,
                    x_rows,
                );
            }
            gpu.hip.device_synchronize().expect("warmup sync");
        }

        // ABBA order. Each row is the mean of ten complete wrapper calls,
        // including the identical Q8_1 activation prepass on both arms.
        let mut base_samples = Vec::with_capacity(2);
        let mut cand_samples = Vec::with_capacity(2);
        for (name, candidate) in [("A", false), ("B", true), ("B", true), ("A", false)] {
            gpu.hip.device_synchronize().expect("pre-timing sync");
            let start = Instant::now();
            for _ in 0..TRIALS_PER_ARM {
                dispatch(
                    &mut gpu,
                    candidate,
                    if candidate {
                        &ep_prequant
                    } else {
                        &ep_shipping
                    },
                    &tile_ids,
                    &slot_index,
                    &x,
                    if candidate { &y_prequant } else { &y_shipping },
                    m,
                    k,
                    x_row_div,
                    m_total,
                    x_rows,
                );
            }
            gpu.hip.device_synchronize().expect("timing sync");
            let us = start.elapsed().as_secs_f64() * 1.0e6 / TRIALS_PER_ARM as f64;
            println!("arm={name} mean_us={us:.3} n={TRIALS_PER_ARM}");
            if candidate {
                cand_samples.push(us);
            } else {
                base_samples.push(us);
            }
        }

        dispatch(
            &mut gpu,
            false,
            &ep_shipping,
            &tile_ids,
            &slot_index,
            &x,
            &y_shipping,
            m,
            k,
            x_row_div,
            m_total,
            x_rows,
        );
        dispatch(
            &mut gpu,
            true,
            &ep_prequant,
            &tile_ids,
            &slot_index,
            &x,
            &y_prequant,
            m,
            k,
            x_row_div,
            m_total,
            x_rows,
        );
        gpu.hip.device_synchronize().expect("correctness sync");
        let mut shipping_out = vec![0u8; y_bytes];
        let mut prequant_out = vec![0u8; y_bytes];
        gpu.hip
            .memcpy_dtoh(&mut shipping_out, &y_shipping_gpu)
            .expect("download shipping output");
        gpu.hip
            .memcpy_dtoh(&mut prequant_out, &y_prequant_gpu)
            .expect("download prequant output");
        let shipping_f32 =
            unsafe { std::slice::from_raw_parts(shipping_out.as_ptr() as *const f32, m_total * m) };
        let prequant_f32 =
            unsafe { std::slice::from_raw_parts(prequant_out.as_ptr() as *const f32, m_total * m) };
        let mut bit_mismatch = 0usize;
        let mut bad = 0usize;
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (&a, &b) in shipping_f32.iter().zip(prequant_f32) {
            bit_mismatch += usize::from(a.to_bits() != b.to_bits());
            let abs = (a - b).abs();
            let rel = abs / a.abs().max(1.0e-12);
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
            bad += usize::from(abs > ATOL && rel > RTOL);
        }

        let base_us = median(&mut base_samples);
        let cand_us = median(&mut cand_samples);
        println!(
            "result label={label} baseline_us={base_us:.3} candidate_us={cand_us:.3} \
             speedup={:.5} delta_pct={:+.3}%",
            base_us / cand_us,
            (base_us / cand_us - 1.0) * 100.0
        );
        println!(
            "correctness elements={} bit_mismatch={} bad={} max_abs={max_abs:.8e} \
             max_rel={max_rel:.8e}",
            m_total * m,
            bit_mismatch,
            bad
        );
        assert_eq!(bad, 0, "candidate exceeded numeric tolerance");

        // The wrapping tensors alias the owning DeviceBuffers above.
        for tensor in [
            ep_shipping,
            ep_prequant,
            tile_ids,
            slot_index,
            x,
            y_shipping,
            y_prequant,
        ] {
            std::mem::forget(tensor);
        }
    }
}
