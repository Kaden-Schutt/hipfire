// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Element-wise channel test for gfx1201 DS4 block-cyclic compressor reads.
//! This proves global score/gather ordering against the incumbent contiguous
//! kernels. It is not product throughput or serving evidence.

use rdna_compute::{DType, Gpu, GpuTensor};

const WORLD: usize = 3;
const BLOCK: usize = 256;
const B: usize = 4;
const H: usize = 64;
const D: usize = 128;
const N: usize = 640;
const K: usize = 64;

fn upload_i32(gpu: &mut Gpu, values: &[i32]) -> GpuTensor {
    let tensor = gpu
        .alloc_tensor(&[values.len() * 4], DType::Raw)
        .expect("alloc i32 tensor");
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    gpu.hip.memcpy_htod(&tensor.buf, bytes).expect("upload i32");
    tensor
}

fn assert_raw_f32_eq(gpu: &Gpu, label: &str, reference: &GpuTensor, candidate: &GpuTensor) {
    gpu.hip.device_synchronize().expect("sync");
    let left = gpu.download_f32(reference).expect("download reference");
    let right = gpu.download_f32(candidate).expect("download candidate");
    assert_eq!(left.len(), right.len(), "{label}: length");
    let mismatches: Vec<_> = left
        .iter()
        .zip(&right)
        .enumerate()
        .filter_map(|(index, (a, b))| (a.to_bits() != b.to_bits()).then_some((index, *a, *b)))
        .take(8)
        .collect();
    assert!(mismatches.is_empty(), "{label}: {mismatches:?}");
    println!("{label}: raw_bits_match={}", left.len());
}

fn shard_cache(cache: &[f32]) -> [Vec<f32>; 4] {
    let mut shards: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::new());
    for global_row in 0..N {
        let global_block = global_row / BLOCK;
        let owner = global_block % WORLD;
        let local_row = (global_block / WORLD) * BLOCK + global_row % BLOCK;
        let required = (local_row + 1) * D;
        if shards[owner].len() < required {
            shards[owner].resize(required, f32::NAN);
        }
        shards[owner][local_row * D..(local_row + 1) * D]
            .copy_from_slice(&cache[global_row * D..(global_row + 1) * D]);
    }
    // The unused fourth kernarg remains a valid allocation on TP3.
    shards[3].resize(D, f32::NAN);
    shards
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert!(gpu.arch_caps.is_gfx1201(), "requires exact gfx1201");

    let cache: Vec<f32> = (0..N * D)
        .map(|i| ((i.wrapping_mul(37) % 4093) as f32 - 2046.0) / 2048.0)
        .collect();
    let q: Vec<f32> = (0..B * H * D)
        .map(|i| ((i.wrapping_mul(19) % 1021) as f32 - 510.0) / 512.0)
        .collect();
    let weights: Vec<f32> = (0..B * H)
        .map(|i| 0.25 + (i.wrapping_mul(11) % 97) as f32 / 128.0)
        .collect();
    let valid = [N as i32, (N - 1) as i32, 513, 257];
    let indices: Vec<i32> = (0..B * K)
        .map(|i| {
            const PROBES: [usize; 12] = [0, 1, 254, 255, 256, 257, 510, 511, 512, 513, 638, 639];
            PROBES[(i * 7 + i / K) % PROBES.len()] as i32
        })
        .collect();

    let cache_t = gpu.upload_f32(&cache, &[N, D]).expect("cache");
    let shard_host = shard_cache(&cache);
    let shard_t: [GpuTensor; 4] = std::array::from_fn(|rank| {
        gpu.upload_f32(&shard_host[rank], &[shard_host[rank].len() / D, D])
            .expect("shard")
    });
    let cache_ptrs = std::array::from_fn(|rank| shard_t[rank].buf.as_ptr() as usize);

    let q_t = gpu.upload_f32(&q, &[B, H, D]).expect("q");
    let weights_t = gpu.upload_f32(&weights, &[B, H]).expect("weights");
    let valid_t = upload_i32(&mut gpu, &valid);
    let indices_t = upload_i32(&mut gpu, &indices);

    let score_ref = gpu.zeros(&[B, N], DType::F32).expect("score ref");
    let score_shard = gpu.zeros(&[B, N], DType::F32).expect("score shard");
    gpu.indexer_relu_score_wmma_batched_f32(
        &q_t, &cache_t, &weights_t, &valid_t, &score_ref, H as i32, D as i32, N as i32, B as i32,
    )
    .expect("contiguous WMMA score");
    gpu.indexer_relu_score_wmma_batched_sharded_gfx1201(
        &q_t,
        &cache_ptrs,
        &weights_t,
        &valid_t,
        &score_shard,
        H as i32,
        D as i32,
        N as i32,
        B as i32,
        WORLD as i32,
        BLOCK as i32,
    )
    .expect("sharded WMMA score");
    assert_raw_f32_eq(&gpu, "batched_score", &score_ref, &score_shard);

    let gather_ref = gpu.zeros(&[B, D, K], DType::F32).expect("gather ref");
    let gather_shard = gpu.zeros(&[B, D, K], DType::F32).expect("gather shard");
    gpu.deepseek4_topk_kv_gather_batched_tiled_gfx1201(
        &cache_t,
        &indices_t,
        &gather_ref,
        K as i32,
        D as i32,
        N as i32,
        K as i32,
        0,
        1.0,
        B as i32,
    )
    .expect("contiguous gather");
    gpu.deepseek4_topk_kv_gather_batched_tiled_sharded_gfx1201(
        &cache_ptrs,
        &indices_t,
        &gather_shard,
        K as i32,
        D as i32,
        N as i32,
        K as i32,
        0,
        1.0,
        B as i32,
        WORLD as i32,
        BLOCK as i32,
    )
    .expect("sharded gather");
    assert_raw_f32_eq(&gpu, "batched_topk_gather", &gather_ref, &gather_shard);

    let q_one = q_t.sub_offset(0, H * D);
    let weights_one = weights_t.sub_offset(0, H);
    let n_one = upload_i32(&mut gpu, &[N as i32]);
    let scalar_ref = gpu.zeros(&[N], DType::F32).expect("scalar ref");
    let scalar_shard = gpu.zeros(&[N], DType::F32).expect("scalar shard");
    gpu.indexer_relu_score_f32_buf(
        &q_one,
        &cache_t,
        &weights_one,
        &scalar_ref,
        &n_one,
        N as i32,
        H as i32,
        D as i32,
    )
    .expect("contiguous scalar score");
    gpu.indexer_relu_score_f32_buf_sharded_gfx1201(
        &q_one,
        &cache_ptrs,
        &weights_one,
        &scalar_shard,
        &n_one,
        N as i32,
        H as i32,
        D as i32,
        WORLD as i32,
        BLOCK as i32,
    )
    .expect("sharded scalar score");
    assert_raw_f32_eq(&gpu, "decode_score", &scalar_ref, &scalar_shard);

    let topk_one = indices_t.sub_offset(0, K);
    let k_one = upload_i32(&mut gpu, &[K as i32]);
    let decode_ref = gpu.zeros(&[D, K], DType::F32).expect("decode ref");
    let decode_shard = gpu.zeros(&[D, K], DType::F32).expect("decode shard");
    gpu.deepseek4_topk_kv_gather_f32_buf(
        &cache_t,
        &topk_one,
        &decode_ref,
        &k_one,
        &n_one,
        K as i32,
        D as i32,
        K as i32,
        0,
        1.0,
    )
    .expect("contiguous decode gather");
    gpu.deepseek4_topk_kv_gather_f32_buf_sharded_gfx1201(
        &cache_ptrs,
        &topk_one,
        &decode_shard,
        &k_one,
        &n_one,
        K as i32,
        D as i32,
        K as i32,
        0,
        1.0,
        WORLD as i32,
        BLOCK as i32,
    )
    .expect("sharded decode gather");
    assert_raw_f32_eq(&gpu, "decode_topk_gather", &decode_ref, &decode_shard);

    const K_ID: usize = 512;
    let identity_ref = gpu.zeros(&[B, D, K_ID], DType::F32).expect("identity ref");
    let identity_shard = gpu
        .zeros(&[B, D, K_ID], DType::F32)
        .expect("identity shard");
    gpu.deepseek4_topk_kv_gather_identity_batched_f32(
        &cache_t,
        &identity_ref,
        K_ID as i32,
        D as i32,
        K_ID as i32,
        B as i32,
    )
    .expect("contiguous identity batch");
    gpu.deepseek4_topk_kv_gather_identity_batched_sharded_gfx1201(
        &cache_ptrs,
        &identity_shard,
        K_ID as i32,
        D as i32,
        K_ID as i32,
        B as i32,
        WORLD as i32,
        BLOCK as i32,
    )
    .expect("sharded identity batch");
    assert_raw_f32_eq(
        &gpu,
        "batched_identity_gather",
        &identity_ref,
        &identity_shard,
    );

    let identity_buf_ref = gpu.zeros(&[D, K_ID], DType::F32).expect("identity buf ref");
    let identity_buf_shard = gpu
        .zeros(&[D, K_ID], DType::F32)
        .expect("identity buf shard");
    let k_id_buf = upload_i32(&mut gpu, &[K_ID as i32]);
    gpu.deepseek4_topk_kv_gather_identity_f32_buf(
        &cache_t,
        &identity_buf_ref,
        &k_id_buf,
        K_ID as i32,
        D as i32,
        K_ID as i32,
    )
    .expect("contiguous identity buf");
    gpu.deepseek4_topk_kv_gather_identity_f32_buf_sharded_gfx1201(
        &cache_ptrs,
        &identity_buf_shard,
        &k_id_buf,
        K_ID as i32,
        D as i32,
        K_ID as i32,
        WORLD as i32,
        BLOCK as i32,
    )
    .expect("sharded identity buf");
    assert_raw_f32_eq(
        &gpu,
        "decode_identity_gather",
        &identity_buf_ref,
        &identity_buf_shard,
    );

    println!("DS4_COMPRESSOR_SHARD_CHANNEL status=pass world={WORLD} block={BLOCK} N={N}");
}
