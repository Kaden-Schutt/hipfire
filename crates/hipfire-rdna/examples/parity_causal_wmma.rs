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
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity: WMMA causal flash (f16 K/V) vs scalar attention_causal_batched.
//! Same prompt batch (b=l) the qwen2 text-prefill path uses. Verdict line PASS
//! if max-abs-diff under f16 tolerance.
use hipfire_rdna::{DType, Gpu};
use std::time::Instant;
fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            ((s >> 16) & 0x7fff) as f32 / 32768.0 - 0.5
        })
        .collect()
}
fn main() {
    let b: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let nh = 12;
    let nkv = 2;
    let hd = 128;
    let mut gpu = Gpu::init().unwrap();
    if !(gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12()) {
        println!(
            "SKIP causal WMMA parity: {} lacks wave32 WMMA; production should use attention_causal_batched",
            gpu.arch
        );
        return;
    }
    let q = gpu
        .upload_f32(&lcg(1, b * nh * hd), &[b * nh * hd])
        .unwrap();
    let k = gpu
        .upload_f32(&lcg(2, b * nkv * hd), &[b * nkv * hd])
        .unwrap();
    let v = gpu
        .upload_f32(&lcg(3, b * nkv * hd), &[b * nkv * hd])
        .unwrap();
    let o_scalar = gpu.zeros(&[b * nh * hd], DType::F32).unwrap();
    gpu.attention_causal_batched(&q, &k, &v, &o_scalar, b, nh, nkv, hd)
        .unwrap();
    let k16 = gpu.alloc_tensor(&[b * nkv * hd], DType::F16).unwrap();
    let v16 = gpu.alloc_tensor(&[b * nkv * hd], DType::F16).unwrap();
    gpu.cast_f32_to_f16(&k, &k16).unwrap();
    gpu.cast_f32_to_f16(&v, &v16).unwrap();
    let o_wmma = gpu.zeros(&[b * nh * hd], DType::F32).unwrap();
    gpu.attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32(
        &q, &k16, &v16, &o_wmma, b, b, nh, nkv, hd,
    )
    .unwrap();
    let a = gpu.download_f32(&o_scalar).unwrap();
    let c = gpu.download_f32(&o_wmma).unwrap();
    let d = a
        .iter()
        .zip(&c)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    println!(
        "max-abs-diff={d:.3e}  {}",
        if d < 5e-3 { "PASS" } else { "FAIL" }
    );

    gpu.hip.device_synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..iters {
        gpu.attention_causal_batched(&q, &k, &v, &o_scalar, b, nh, nkv, hd)
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let scalar_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        gpu.attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32(
            &q, &k16, &v16, &o_wmma, b, b, nh, nkv, hd,
        )
        .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let wmma_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

    println!(
        "b={b} iters={iters} scalar={scalar_us:.1} us/call wmma={wmma_us:.1} us/call speedup={:.2}x",
        scalar_us / wmma_us
    );
}
