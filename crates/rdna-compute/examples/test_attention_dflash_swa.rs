// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! GPU parity test for Qwen3.6 DFlash causal sliding-window attention.

use rdna_compute::{DType, Gpu};

fn data(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.2
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k_ctx: &[f32],
    v_ctx: &[f32],
    k_noise: &[f32],
    v_noise: &[f32],
    positions_q: &[i32],
    positions_ctx: &[i32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
) -> Vec<f32> {
    let b = positions_q.len();
    let l = positions_ctx.len();
    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv_heads * head_dim;
    let repeat = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0; b * q_stride];

    for qi in 0..b {
        for head in 0..n_heads {
            let kv_head = head / repeat;
            let q_row = &q[qi * q_stride + head * head_dim..][..head_dim];
            let mut scores = Vec::new();
            let mut rows = Vec::new();
            for source_row in 0..l + b {
                let (key_pos, k, v, row) = if source_row < l {
                    (positions_ctx[source_row], k_ctx, v_ctx, source_row)
                } else {
                    let row = source_row - l;
                    (positions_q[row], k_noise, v_noise, row)
                };
                let delta = positions_q[qi] - key_pos;
                if delta < 0 || delta as usize >= window {
                    continue;
                }
                let offset = row * kv_stride + kv_head * head_dim;
                let score = q_row
                    .iter()
                    .zip(&k[offset..offset + head_dim])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
                scores.push(score);
                rows.push((&v[offset..offset + head_dim], score));
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom = scores.iter().map(|s| (*s - max).exp()).sum::<f32>();
            let out_offset = qi * q_stride + head * head_dim;
            for (v, score) in rows {
                let probability = (score - max).exp() / denom;
                for d in 0..head_dim {
                    out[out_offset + d] += probability * v[d];
                }
            }
        }
    }
    out
}

fn upload_i32(gpu: &mut Gpu, values: &[i32]) -> rdna_compute::GpuTensor {
    let tensor = gpu.alloc_tensor(&[values.len()], DType::F32).unwrap();
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    gpu.hip.memcpy_htod(&tensor.buf, bytes).unwrap();
    tensor
}

fn run_case(gpu: &mut Gpu, l: usize, b: usize, window: usize, compacted: bool) -> f32 {
    let (n_heads, n_kv_heads, head_dim) = (4, 2, 64);
    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv_heads * head_dim;
    let q = data(1, b * q_stride);
    let k_ctx = data(2, l * kv_stride);
    let v_ctx = data(3, l * kv_stride);
    let k_noise = data(4, b * kv_stride);
    let v_noise = data(5, b * kv_stride);
    let positions_ctx: Vec<i32> = if compacted {
        (0..l)
            .map(|i| if i < 8 { i as i32 } else { 10_000 + i as i32 })
            .collect()
    } else {
        (0..l as i32).collect()
    };
    let first_q = positions_ctx.last().copied().unwrap_or(0) + 1;
    let positions_q: Vec<i32> = (first_q..first_q + b as i32).collect();
    let expected = reference(
        &q,
        &k_ctx,
        &v_ctx,
        &k_noise,
        &v_noise,
        &positions_q,
        &positions_ctx,
        n_heads,
        n_kv_heads,
        head_dim,
        window,
    );

    let d_q = gpu.upload_f32(&q, &[q.len()]).unwrap();
    let d_k_ctx = gpu.upload_f32(&k_ctx, &[k_ctx.len()]).unwrap();
    let d_v_ctx = gpu.upload_f32(&v_ctx, &[v_ctx.len()]).unwrap();
    let d_k_noise = gpu.upload_f32(&k_noise, &[k_noise.len()]).unwrap();
    let d_v_noise = gpu.upload_f32(&v_noise, &[v_noise.len()]).unwrap();
    let d_positions_q = upload_i32(gpu, &positions_q);
    let d_positions_ctx = upload_i32(gpu, &positions_ctx);
    let d_out = gpu.zeros(&[b * q_stride], DType::F32).unwrap();
    gpu.attention_dflash_swa_f32(
        &d_q,
        &d_k_ctx,
        &d_v_ctx,
        &d_k_noise,
        &d_v_noise,
        &d_positions_q,
        &d_positions_ctx,
        &d_out,
        b,
        l,
        n_heads,
        n_kv_heads,
        head_dim,
        window,
    )
    .unwrap();
    let actual = gpu.download_f32(&d_out).unwrap();
    let max_error = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    for tensor in [
        d_q,
        d_k_ctx,
        d_v_ctx,
        d_k_noise,
        d_v_noise,
        d_positions_q,
        d_positions_ctx,
        d_out,
    ] {
        gpu.free_tensor(tensor).unwrap();
    }
    max_error
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    let cases = [(64, 16, 32, false), (2_050, 4, 2_048, false), (80, 8, 48, true)];
    for (l, b, window, compacted) in cases {
        let error = run_case(&mut gpu, l, b, window, compacted);
        println!(
            "L={l} B={b} window={window} compacted={compacted} max_abs_error={error:.3e}"
        );
        assert!(error < 1.0e-3, "DFlash SWA parity failed: {error}");
    }
}
