#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.
#![allow(
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::drop_non_drop,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::print_literal,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unusual_byte_groupings,
    clippy::useless_vec,
    clippy::unnecessary_cast
)]
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness + perf A/B for `gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2`
//! vs the single-warp baseline `gemm_mq2g256_lloyd_moe_grouped_wmma_k2`.
//!
//! Shapes target the DeepSeek V4 MoE hot path:
//!   gate/up: M=2048 (moe_intermediate), K=4096 (hidden)
//!   down:    M=4096 (hidden),           K=2048 (moe_intermediate)
//! m_total covers PP_BATCH ∈ {128, 256, 1024} × top_k=6 routed slots.
//!
//! E=1 single-expert setup keeps memory small; expert_tile_ids points
//! every slot to expert 0. Correctness check: 4w output must match
//! baseline output element-wise within FMA-reduction-order tolerance.

use hipfire_rdna::{DType, Gpu, GpuTensor};
use std::time::Instant;

const WARMUP: usize = 4;
const TRIALS: usize = 30;
const ATOL: f32 = 1e-2;
const RTOL: f32 = 1e-2;

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (((bits >> 23) & 0xff) as i32) - 127 + 15;
    let mant = (bits & 0x7fffff) as u32;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

#[allow(dead_code)]
fn f32_to_f16_bytes(f: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(f.len() * 2);
    for &v in f {
        out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
    }
    out
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
fn launch_variant(
    gpu: &mut Gpu,
    variant: &str,
    expert_weight_ptrs: &GpuTensor,
    expert_tile_ids: &GpuTensor,
    sorted_slot_index: &GpuTensor,
    x_src: &GpuTensor,
    y_grouped: &GpuTensor,
    m: usize,
    k: usize,
    x_row_div: usize,
    m_total: usize,
    x_src_rows: usize,
) {
    match variant {
        "1w16x16" => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_k2(
            expert_weight_ptrs,
            expert_tile_ids,
            sorted_slot_index,
            x_src,
            y_grouped,
            m,
            k,
            x_row_div,
            m_total,
            x_src_rows,
        ),
        "4w64x16" => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2(
            expert_weight_ptrs,
            expert_tile_ids,
            sorted_slot_index,
            x_src,
            y_grouped,
            m,
            k,
            x_row_div,
            m_total,
            x_src_rows,
        ),
        "4w64x16_mmqload" => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload(
            expert_weight_ptrs,
            expert_tile_ids,
            sorted_slot_index,
            x_src,
            y_grouped,
            m,
            k,
            x_row_div,
            m_total,
            x_src_rows,
        ),
        "4w64x16_mmqload_nosync" => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_mmqload_nosync(
            expert_weight_ptrs,
            expert_tile_ids,
            sorted_slot_index,
            x_src,
            y_grouped,
            m,
            k,
            x_row_div,
            m_total,
            x_src_rows,
        ),
        "4w64x32" => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32(
            expert_weight_ptrs,
            expert_tile_ids,
            sorted_slot_index,
            x_src,
            y_grouped,
            m,
            k,
            x_row_div,
            m_total,
            x_src_rows,
        ),
        "4w64x16_cnd" => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_cnd(
            expert_weight_ptrs,
            expert_tile_ids,
            sorted_slot_index,
            x_src,
            y_grouped,
            m,
            k,
            x_row_div,
            m_total,
            x_src_rows,
        ),
        "8w128x16" => gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_8w_k2(
            expert_weight_ptrs,
            expert_tile_ids,
            sorted_slot_index,
            x_src,
            y_grouped,
            m,
            k,
            x_row_div,
            m_total,
            x_src_rows,
        ),
        _ => panic!("unknown MQ2-Lloyd MoE variant {variant}"),
    }
    .expect("variant launch");
}

/// MQ2-Lloyd quant: per 256-element K-group, [4 × fp16 codebook | 64 B
/// packed 2-bit indices (4 per byte, LSB-first)]. Total 72 B/group.
/// Fixed codebook `[-3, -1, 1, 3]`; deterministic pseudo-random indices.
fn quantize_mq2_lloyd(k: usize, rows: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0, "K must be multiple of 256");
    let groups_per_row = k / 256;
    let mut out = Vec::with_capacity(rows * groups_per_row * 72);
    let mut rng = seed;
    for _ in 0..rows {
        for _ in 0..groups_per_row {
            // 4 fp16 codebook entries, sorted ascending.
            for &v in &[-3.0f32, -1.0, 1.0, 3.0] {
                out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
            // 256 × 2-bit indices packed 4 per byte → 64 bytes/group.
            for _ in 0..64 {
                let mut byte = 0u8;
                for nibble in 0..4 {
                    rng = rng
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let idx = ((rng >> 48) & 0x3) as u8;
                    byte |= idx << (nibble * 2);
                }
                out.push(byte);
            }
        }
    }
    out
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("Arch: {}", gpu.arch);

    // V4F MoE config: top_k = 6.
    const TOP_K: usize = 6;

    // (M, K, batch, label) — batch is PP_BATCH; m_total = batch × TOP_K.
    let shapes: &[(usize, usize, usize, &str)] = &[
        (2048, 4096, 128, "gate/up B=128"),
        (2048, 4096, 256, "gate/up B=256"),
        (2048, 4096, 1024, "gate/up B=1024 (V4F prefill default)"),
        (4096, 2048, 128, "down B=128"),
        (4096, 2048, 256, "down B=256"),
        (4096, 2048, 1024, "down B=1024 (V4F prefill default)"),
    ];

    for &(m, k, batch, label) in shapes {
        let m_total = batch * TOP_K;
        println!("\n=== {label} | M={m} K={k} batch={batch} m_total={m_total} ===");
        if m % 64 != 0 {
            println!("  SKIP — 4w kernel requires M%64==0 (got M={m})");
            continue;
        }
        if k % 256 != 0 {
            println!("  SKIP — both kernels require K%256==0");
            continue;
        }

        let weight_bytes = quantize_mq2_lloyd(k, m, 0xC0FFEEu64);
        // X is F32: the Lloyd dispatch wrapper calls ensure_fp16_x()
        // unconditionally, which treats input as F32 and converts to F16.
        // Passing F16 would be read OOB (2× the bytes). Production V4F
        // prefill passes F32 here too.
        let x_f32: Vec<f32> = (0..m_total * k)
            .map(|i| ((i % 11) as f32 - 5.0) / 5.0)
            .collect();
        let x_f32_bytes: Vec<u8> = x_f32
            .iter()
            .flat_map(|v| v.to_le_bytes().to_vec())
            .collect();

        let w_gpu = gpu.hip.malloc(weight_bytes.len()).expect("malloc W");
        let x_gpu = gpu.hip.malloc(x_f32_bytes.len()).expect("malloc X");
        let yref_gpu = gpu.hip.malloc(m_total * m * 4).expect("malloc Yref");
        let y4w_gpu = gpu.hip.malloc(m_total * m * 4).expect("malloc Y4w");
        gpu.hip.memcpy_htod(&w_gpu, &weight_bytes).expect("htod W");
        gpu.hip.memcpy_htod(&x_gpu, &x_f32_bytes).expect("htod X");

        // expert_weight_ptrs = [w_gpu.as_ptr() as u64]
        let w_ptr_u64 = w_gpu.as_ptr() as u64;
        let ep_bytes = w_ptr_u64.to_le_bytes().to_vec();
        let ep_gpu = gpu.hip.malloc(8).expect("malloc EP");
        gpu.hip.memcpy_htod(&ep_gpu, &ep_bytes).expect("htod EP");

        // expert_tile_ids = [0; m_total/16]
        let slot_tiles = (m_total + 15) / 16;
        let tile_ids_bytes: Vec<u8> = (0..slot_tiles)
            .flat_map(|_| 0i32.to_le_bytes().to_vec())
            .collect();
        let tp_gpu = gpu.hip.malloc(tile_ids_bytes.len()).expect("malloc TP");
        gpu.hip
            .memcpy_htod(&tp_gpu, &tile_ids_bytes)
            .expect("htod TP");

        // sorted_slot_index = identity
        let perm_bytes: Vec<u8> = (0..m_total)
            .flat_map(|i| (i as i32).to_le_bytes().to_vec())
            .collect();
        let sp_gpu = gpu.hip.malloc(perm_bytes.len()).expect("malloc SP");
        gpu.hip.memcpy_htod(&sp_gpu, &perm_bytes).expect("htod SP");

        // Wrap as GpuTensor for the dispatch fn.
        let ep_t = wrap_buf(ep_gpu.as_ptr(), 8, vec![1], DType::F32);
        let tp_t = wrap_buf(
            tp_gpu.as_ptr(),
            tile_ids_bytes.len(),
            vec![slot_tiles],
            DType::F32,
        );
        let sp_t = wrap_buf(sp_gpu.as_ptr(), perm_bytes.len(), vec![m_total], DType::F32);
        let x_t = wrap_buf(
            x_gpu.as_ptr(),
            x_f32_bytes.len(),
            vec![m_total, k],
            DType::F32,
        );
        let yref_t = wrap_buf(
            yref_gpu.as_ptr(),
            m_total * m * 4,
            vec![m_total, m],
            DType::F32,
        );
        let y4w_t = wrap_buf(
            y4w_gpu.as_ptr(),
            m_total * m * 4,
            vec![m_total, m],
            DType::F32,
        );

        let variants = [
            "4w64x16",
            "4w64x16_mmqload",
            "4w64x16_mmqload_nosync",
            "4w64x32",
            "4w64x16_cnd",
            "8w128x16",
        ];

        // ── Correctness pass: baseline vs each variant, element-wise compare.
        launch_variant(
            &mut gpu, "1w16x16", &ep_t, &tp_t, &sp_t, &x_t, &yref_t, m, k, 1, m_total, m_total,
        );
        gpu.hip.device_synchronize().expect("sync");
        let mut y_ref_bytes = vec![0u8; m_total * m * 4];
        gpu.hip
            .memcpy_dtoh(&mut y_ref_bytes, &yref_gpu)
            .expect("dtoh ref");
        let y_ref: &[f32] =
            unsafe { std::slice::from_raw_parts(y_ref_bytes.as_ptr() as *const f32, m_total * m) };

        let mut ok_variants = Vec::new();
        for variant in variants {
            launch_variant(
                &mut gpu, variant, &ep_t, &tp_t, &sp_t, &x_t, &y4w_t, m, k, 1, m_total, m_total,
            );
            gpu.hip.device_synchronize().expect("sync");

            let mut y_4w_bytes = vec![0u8; m_total * m * 4];
            gpu.hip
                .memcpy_dtoh(&mut y_4w_bytes, &y4w_gpu)
                .expect("dtoh variant");
            let y_4w: &[f32] = unsafe {
                std::slice::from_raw_parts(y_4w_bytes.as_ptr() as *const f32, m_total * m)
            };

            let mut max_abs = 0f32;
            let mut max_rel = 0f32;
            let mut bad = 0usize;
            let mut nan_count = 0usize;
            for i in 0..(m_total * m) {
                if !y_4w[i].is_finite() {
                    nan_count += 1;
                    continue;
                }
                let d = (y_ref[i] - y_4w[i]).abs();
                let r = d / y_ref[i].abs().max(1e-6);
                if d > max_abs {
                    max_abs = d;
                }
                if r > max_rel {
                    max_rel = r;
                }
                if d > ATOL && r > RTOL {
                    bad += 1;
                }
            }
            let ok = bad == 0 && nan_count == 0;
            println!(
                "  correctness {variant:>23}: max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  bad={bad}/{}  nan={nan_count}  {}",
                m_total * m,
                if ok { "OK" } else { "FAIL" },
            );
            if ok {
                ok_variants.push(variant);
            }
        }
        // ── Perf A/B
        for _ in 0..WARMUP {
            launch_variant(
                &mut gpu, "1w16x16", &ep_t, &tp_t, &sp_t, &x_t, &yref_t, m, k, 1, m_total, m_total,
            );
        }
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..TRIALS {
            launch_variant(
                &mut gpu, "1w16x16", &ep_t, &tp_t, &sp_t, &x_t, &yref_t, m, k, 1, m_total, m_total,
            );
        }
        gpu.hip.device_synchronize().unwrap();
        let ref_us = t0.elapsed().as_secs_f64() / TRIALS as f64 * 1e6;

        let flops = 2.0 * m as f64 * k as f64 * m_total as f64;
        let ref_gflops = flops / ref_us / 1e3;
        println!("  ref (1w16x16): {ref_us:>8.1} µs ({ref_gflops:>6.0} GFLOPS)");
        for variant in ok_variants {
            for _ in 0..WARMUP {
                launch_variant(
                    &mut gpu, variant, &ep_t, &tp_t, &sp_t, &x_t, &y4w_t, m, k, 1, m_total, m_total,
                );
            }
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..TRIALS {
                launch_variant(
                    &mut gpu, variant, &ep_t, &tp_t, &sp_t, &x_t, &y4w_t, m, k, 1, m_total, m_total,
                );
            }
            gpu.hip.device_synchronize().unwrap();
            let new_us = t0.elapsed().as_secs_f64() / TRIALS as f64 * 1e6;
            let new_gflops = flops / new_us / 1e3;
            println!(
                "  {variant:>23}: {new_us:>8.1} µs ({new_gflops:>6.0} GFLOPS)   speedup: {:.2}x",
                ref_us / new_us
            );
        }

        std::mem::forget(ep_t);
        std::mem::forget(tp_t);
        std::mem::forget(sp_t);
        std::mem::forget(x_t);
        std::mem::forget(yref_t);
        std::mem::forget(y4w_t);
    }
}
