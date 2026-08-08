// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Model-free channel test for the selectable gfx1201 DeepSeek V4 F16
//! compressor cache. F32 reference kernels consume the exact values obtained by
//! widening the stored half bits, so raw-bit equality proves the new readers
//! change storage only, not arithmetic or indexing.

use rdna_compute::{DType, Gpu, GpuTensor};

const H: usize = 64;
const D: usize = 128;
const N: usize = 257;
const B: usize = 8;
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

fn download_u16(gpu: &Gpu, tensor: &GpuTensor) -> Vec<u16> {
    assert_eq!(tensor.dtype, DType::F16);
    let mut bytes = vec![0_u8; tensor.numel() * 2];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &tensor.buf)
        .expect("download f16");
    bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let fraction = (bits & 0x03ff) as u32;
    let out = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut mantissa = fraction;
            let mut shift = 0_u32;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                shift += 1;
            }
            mantissa &= 0x03ff;
            sign | ((113 - shift) << 23) | (mantissa << 13)
        }
        31 => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(out)
}

fn assert_raw_f32_eq(gpu: &Gpu, label: &str, reference: &GpuTensor, candidate: &GpuTensor) {
    gpu.hip.device_synchronize().expect("sync");
    let left = gpu.download_f32(reference).expect("download reference");
    let right = gpu.download_f32(candidate).expect("download candidate");
    let first = left
        .iter()
        .zip(&right)
        .enumerate()
        .find_map(|(slot, (a, b))| {
            (a.to_bits() != b.to_bits()).then_some((slot, a.to_bits(), b.to_bits()))
        });
    assert!(first.is_none(), "{label}: first raw mismatch {first:?}");
    println!("{label}: raw_bits_match={}", left.len());
}

fn assert_raw_f16_eq(gpu: &Gpu, label: &str, reference: &GpuTensor, candidate: &GpuTensor) {
    gpu.hip.device_synchronize().expect("sync");
    let left = download_u16(gpu, reference);
    let right = download_u16(gpu, candidate);
    let first = left
        .iter()
        .zip(&right)
        .enumerate()
        .find_map(|(slot, (a, b))| (a != b).then_some((slot, *a, *b)));
    assert!(first.is_none(), "{label}: first raw mismatch {first:?}");
    println!("{label}: raw_bits_match={}", left.len());
}

fn deterministic(len: usize, multiplier: usize, modulus: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let centered = (index.wrapping_mul(multiplier) % modulus) as i32
                - (modulus as i32 / 2);
            centered as f32 * scale
        })
        .collect()
}

fn test_staged_commit(gpu: &mut Gpu) {
    const T: usize = 4;
    const SLOTS: usize = 4;
    const SLOT: usize = 2;
    let kv_state = gpu
        .upload_f32(&deterministic(T * D, 37, 257, 1.0 / 256.0), &[T, D])
        .unwrap();
    let score_state = gpu
        .upload_f32(&deterministic(T * D, 19, 127, 1.0 / 128.0), &[T, D])
        .unwrap();
    let weight = gpu
        .upload_f32(&deterministic(D, 11, 31, 1.0 / 64.0), &[D])
        .unwrap();
    let slot = upload_i32(gpu, &[SLOT as i32]);
    let pos = upload_i32(gpu, &[123]);
    let reference_f32 = gpu.zeros(&[SLOTS, D], DType::F32).unwrap();
    let staged = gpu.zeros(&[D], DType::F32).unwrap();
    let candidate_f16 = gpu.zeros(&[SLOTS, D], DType::F16).unwrap();

    gpu.compressor_softmax_pool_f32_buf(
        &kv_state,
        &score_state,
        &reference_f32,
        &slot,
        T as i32,
        D as i32,
    )
    .unwrap();
    gpu.compressor_softmax_pool_f32_staged_buf(
        &kv_state,
        &score_state,
        &staged,
        &slot,
        T as i32,
        D as i32,
    )
    .unwrap();
    gpu.rmsnorm_f32_at_slot_buf(&reference_f32, &weight, &slot, D as i32, 1.0e-6)
        .unwrap();
    gpu.rmsnorm_f32_staged_buf(&staged, &weight, &slot, D as i32, 1.0e-6)
        .unwrap();
    gpu.rope_tail_yarn_interleaved_at_slot_buf(
        &reference_f32,
        &pos,
        &slot,
        D as i32,
        64,
        10_000.0,
        0.5,
        1.0,
        1.0,
        8.0,
        24.0,
    )
    .unwrap();
    gpu.rope_tail_yarn_interleaved_staged_buf_gfx1201(
        &staged,
        &pos,
        &slot,
        D as i32,
        64,
        10_000.0,
        0.5,
        1.0,
        1.0,
        8.0,
        24.0,
    )
    .unwrap();

    let reference_slot = reference_f32.sub_offset(SLOT * D, D);
    let reference_f16 = gpu.zeros(&[D], DType::F16).unwrap();
    gpu.cast_f32_to_f16(&reference_slot, &reference_f16)
        .unwrap();
    gpu.cast_f32_to_f16_at_slot_buf(&staged, &candidate_f16, &slot, D as i32)
        .unwrap();
    let candidate_slot = candidate_f16.sub_offset(SLOT * D, D);
    assert_raw_f16_eq(gpu, "staged_commit", &reference_f16, &candidate_slot);
}

fn test_readers(gpu: &mut Gpu) {
    let cache_f32 = gpu
        .upload_f32(&deterministic(N * D, 29, 4093, 1.0 / 2048.0), &[N, D])
        .unwrap();
    let cache_f16 = gpu.zeros(&[N, D], DType::F16).unwrap();
    gpu.cast_f32_to_f16(&cache_f32, &cache_f16).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let rounded_host: Vec<f32> = download_u16(gpu, &cache_f16)
        .into_iter()
        .map(f16_to_f32)
        .collect();
    let rounded_f32 = gpu.upload_f32(&rounded_host, &[N, D]).unwrap();

    let q = gpu
        .upload_f32(&deterministic(B * H * D, 17, 1021, 1.0 / 512.0), &[B, H, D])
        .unwrap();
    let weights = gpu
        .upload_f32(&deterministic(B * H, 13, 97, 1.0 / 128.0), &[B, H])
        .unwrap();
    let valid_host: Vec<i32> = (0..B).map(|b| (N - b * 7) as i32).collect();
    let valid = upload_i32(gpu, &valid_host);

    let q_one = q.sub_offset(0, H * D);
    let weights_one = weights.sub_offset(0, H);
    let n_one = upload_i32(gpu, &[N as i32]);
    let scalar_ref = gpu.zeros(&[N], DType::F32).unwrap();
    let scalar_f16 = gpu.zeros(&[N], DType::F32).unwrap();
    gpu.indexer_relu_score_f32_buf(
        &q_one,
        &rounded_f32,
        &weights_one,
        &scalar_ref,
        &n_one,
        N as i32,
        H as i32,
        D as i32,
    )
    .unwrap();
    gpu.indexer_relu_score_f16_buf(
        &q_one,
        &cache_f16,
        &weights_one,
        &scalar_f16,
        &n_one,
        N as i32,
        H as i32,
        D as i32,
    )
    .unwrap();
    assert_raw_f32_eq(gpu, "decode_score", &scalar_ref, &scalar_f16);

    let batch_ref = gpu.zeros(&[B, N], DType::F32).unwrap();
    let batch_f16 = gpu.zeros(&[B, N], DType::F32).unwrap();
    gpu.indexer_relu_score_batched_f32(
        &q,
        &rounded_f32,
        &weights,
        &valid,
        &batch_ref,
        H as i32,
        D as i32,
        N as i32,
        B as i32,
    )
    .unwrap();
    gpu.indexer_relu_score_batched_f16(
        &q,
        &cache_f16,
        &weights,
        &valid,
        &batch_f16,
        H as i32,
        D as i32,
        N as i32,
        B as i32,
    )
    .unwrap();
    assert_raw_f32_eq(gpu, "batched_score", &batch_ref, &batch_f16);

    let wmma_ref = gpu.zeros(&[B, N], DType::F32).unwrap();
    let wmma_f16 = gpu.zeros(&[B, N], DType::F32).unwrap();
    gpu.indexer_relu_score_wmma_batched_f32(
        &q,
        &rounded_f32,
        &weights,
        &valid,
        &wmma_ref,
        H as i32,
        D as i32,
        N as i32,
        B as i32,
    )
    .unwrap();
    gpu.indexer_relu_score_wmma_batched_f16(
        &q,
        &cache_f16,
        &weights,
        &valid,
        &wmma_f16,
        H as i32,
        D as i32,
        N as i32,
        B as i32,
    )
    .unwrap();
    assert_raw_f32_eq(gpu, "batched_score_wmma", &wmma_ref, &wmma_f16);

    let indices_host: Vec<i32> = (0..B * K)
        .map(|slot| ((slot * 73 + slot / K * 17) % N) as i32)
        .collect();
    let indices = upload_i32(gpu, &indices_host);
    let indices_one = indices.sub_offset(0, K);
    let k_one = upload_i32(gpu, &[K as i32]);
    let decode_ref = gpu.zeros(&[D, K], DType::F32).unwrap();
    let decode_f16 = gpu.zeros(&[D, K], DType::F32).unwrap();
    gpu.deepseek4_topk_kv_gather_f32_buf(
        &rounded_f32,
        &indices_one,
        &decode_ref,
        &k_one,
        &n_one,
        K as i32,
        D as i32,
        K as i32,
        0,
        0.75,
    )
    .unwrap();
    gpu.deepseek4_topk_kv_gather_f16_buf(
        &cache_f16,
        &indices_one,
        &decode_f16,
        &k_one,
        &n_one,
        K as i32,
        D as i32,
        K as i32,
        0,
        0.75,
    )
    .unwrap();
    assert_raw_f32_eq(gpu, "decode_gather", &decode_ref, &decode_f16);

    let gather_ref = gpu.zeros(&[B, D, K], DType::F32).unwrap();
    let gather_f16 = gpu.zeros(&[B, D, K], DType::F32).unwrap();
    gpu.deepseek4_topk_kv_gather_batched_tiled_gfx1201(
        &rounded_f32,
        &indices,
        &gather_ref,
        K as i32,
        D as i32,
        N as i32,
        K as i32,
        0,
        0.75,
        B as i32,
    )
    .unwrap();
    gpu.deepseek4_topk_kv_gather_batched_tiled_f16(
        &cache_f16,
        &indices,
        &gather_f16,
        K as i32,
        D as i32,
        N as i32,
        K as i32,
        0,
        0.75,
        B as i32,
    )
    .unwrap();
    assert_raw_f32_eq(gpu, "batched_gather", &gather_ref, &gather_f16);

    let identity_ref = gpu.zeros(&[B, D, K], DType::F32).unwrap();
    let identity_f16 = gpu.zeros(&[B, D, K], DType::F32).unwrap();
    gpu.deepseek4_topk_kv_gather_identity_batched_f32(
        &rounded_f32,
        &identity_ref,
        K as i32,
        D as i32,
        K as i32,
        B as i32,
    )
    .unwrap();
    gpu.deepseek4_topk_kv_gather_identity_batched_f16(
        &cache_f16,
        &identity_f16,
        K as i32,
        D as i32,
        K as i32,
        B as i32,
    )
    .unwrap();
    assert_raw_f32_eq(gpu, "batched_identity", &identity_ref, &identity_f16);

    let identity_decode_ref = gpu.zeros(&[D, K], DType::F32).unwrap();
    let identity_decode_f16 = gpu.zeros(&[D, K], DType::F32).unwrap();
    gpu.deepseek4_topk_kv_gather_identity_f32_buf(
        &rounded_f32,
        &identity_decode_ref,
        &k_one,
        K as i32,
        D as i32,
        K as i32,
    )
    .unwrap();
    gpu.deepseek4_topk_kv_gather_identity_f16_buf(
        &cache_f16,
        &identity_decode_f16,
        &k_one,
        K as i32,
        D as i32,
        K as i32,
    )
    .unwrap();
    assert_raw_f32_eq(
        gpu,
        "decode_identity",
        &identity_decode_ref,
        &identity_decode_f16,
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    assert_eq!(gpu.arch, "gfx1201", "requires exact gfx1201");
    println!("arch={} gate=PASS", gpu.arch);
    test_staged_commit(&mut gpu);
    test_readers(&mut gpu);
    println!("ds4_compressor_cache_f16_gfx1201=PASS");
}
