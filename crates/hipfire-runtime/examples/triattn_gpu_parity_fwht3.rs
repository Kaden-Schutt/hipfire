// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU↔CPU numeric oracle for adaptive FWHT3 TriAttention scoring.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::llama::KvCache;
    use hipfire_runtime::triattn::{self, BandCenter, TriAttnCenters};
    use rdna_compute::{DType, Gpu};

    const C3: [f32; 8] = [
        -0.134860, -0.083320, -0.046469, -0.015176, 0.015176, 0.046469, 0.083320, 0.134860,
    ];
    const N_HEADS: usize = 16;
    const N_KV_HEADS: usize = 4;
    const HEAD_DIM: usize = 256;
    const SEQ_LEN: usize = 32;
    const ROPE_THETA: f32 = 10_000_000.0;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as u32 as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }

    fn inverse_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
        for (value, sign) in x.iter_mut().zip(signs2) {
            *value *= sign;
        }
        let mut stride = 1;
        while stride < 256 {
            for base in (0..256).step_by(stride * 2) {
                for i in 0..stride {
                    let a = x[base + i];
                    let b = x[base + i + stride];
                    x[base + i] = a + b;
                    x[base + i + stride] = a - b;
                }
            }
            stride *= 2;
        }
        for i in 0..256 {
            x[i] *= 0.0625 * signs1[i];
        }
    }

    let n_bands = HEAD_DIM / 2;
    let kv_group = N_HEADS / N_KV_HEADS;
    let p_q = (SEQ_LEN - 1) as f32;
    let mut seed = 0x5eed_cafe_u64;
    let mut centers = TriAttnCenters::new(1, N_HEADS, HEAD_DIM, ROPE_THETA, 1.0);
    let mut centers_flat = Vec::with_capacity(N_HEADS * n_bands * 3);
    for h in 0..N_HEADS {
        for f in 0..n_bands {
            let center = BandCenter {
                eq_re: 0.3 * lcg(&mut seed),
                eq_im: 0.3 * lcg(&mut seed),
                e_abs_q: 0.5 + 0.3 * lcg(&mut seed).abs(),
            };
            centers.set(0, h, f, center);
            centers_flat.extend_from_slice(&[center.eq_re, center.eq_im, center.e_abs_q]);
        }
    }

    let mut gpu = Gpu::init().expect("gpu init");
    let kv = KvCache::new_gpu_fwht3_filtered(&mut gpu, &[true], N_KV_HEADS, HEAD_DIM, SEQ_LEN)
        .expect("fwht3 cache");
    let signs1 = kv.givens_cos.as_ref().expect("fwht signs1");
    let signs2 = kv.givens_sin.as_ref().expect("fwht signs2");
    let pos_dev = gpu.hip.malloc(4).expect("position buffer");
    let kv_dim = N_KV_HEADS * HEAD_DIM;
    for pos in 0..SEQ_LEN {
        let k_row: Vec<f32> = (0..kv_dim).map(|_| 0.5 * lcg(&mut seed)).collect();
        let v_row: Vec<f32> = (0..kv_dim).map(|_| 0.5 * lcg(&mut seed)).collect();
        let k_tmp = gpu.upload_f32(&k_row, &[kv_dim]).expect("upload K");
        let v_tmp = gpu.upload_f32(&v_row, &[kv_dim]).expect("upload V");
        gpu.hip
            .memcpy_htod(&pos_dev, &(pos as i32).to_ne_bytes())
            .expect("upload position");
        gpu.kv_cache_write_fwht3_fused(
            &kv.k_gpu[0],
            &kv.v_gpu[0],
            &k_tmp,
            &v_tmp,
            &pos_dev,
            signs1,
            signs2,
            N_KV_HEADS,
            HEAD_DIM,
            8,
        )
        .expect("write fwht3 row");
    }

    let centers_dev = gpu
        .upload_f32(&centers_flat, &[centers_flat.len()])
        .expect("upload centers");
    let scores_dev = gpu
        .alloc_tensor(&[N_HEADS * SEQ_LEN], DType::F32)
        .expect("score buffer");
    gpu.triattn_score_fwht(
        &kv.k_gpu[0],
        &centers_dev,
        signs1,
        signs2,
        &scores_dev,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        ROPE_THETA,
        p_q,
        SEQ_LEN,
        3,
    )
    .expect("FWHT3 score");
    gpu.hip.device_synchronize().expect("score synchronize");
    let gpu_scores = gpu.download_f32(&scores_dev).expect("download scores");
    let signs1_cpu = gpu.download_f32(signs1).expect("download signs1");
    let signs2_cpu = gpu.download_f32(signs2).expect("download signs2");
    let cache_bytes: Vec<u8> = gpu
        .download_f32(&kv.k_gpu[0])
        .expect("download K cache")
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect();

    let bytes_per_head = 4 + HEAD_DIM * 3 / 8;
    let bytes_per_pos = N_KV_HEADS * bytes_per_head;
    let mut cpu_scores = vec![0.0; N_HEADS * SEQ_LEN];
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for h in 0..N_HEADS {
        let kv_h = h / kv_group;
        let center_base = h * n_bands;
        let center_slice = &centers.centers[center_base..center_base + n_bands];
        for pos in 0..SEQ_LEN {
            let row = pos * bytes_per_pos + kv_h * bytes_per_head;
            let cnorm = f32::from_ne_bytes(cache_bytes[row..row + 4].try_into().unwrap());
            let mut k_post = vec![0.0f32; HEAD_DIM];
            for tid in 0..32 {
                let packed_row = row + 4 + tid * 3;
                let packed = cache_bytes[packed_row] as u32
                    | ((cache_bytes[packed_row + 1] as u32) << 8)
                    | ((cache_bytes[packed_row + 2] as u32) << 16);
                for i in 0..8 {
                    k_post[tid * 8 + i] = cnorm * C3[((packed >> (i * 3)) & 7) as usize];
                }
            }
            inverse_fwht_256(&mut k_post, &signs1_cpu, &signs2_cpu);
            let bands = triattn::kpost_per_band(&k_post);
            let cpu = triattn::s_total(center_slice, &bands, p_q, |f| centers.omega(f));
            let gpu_value = gpu_scores[h * SEQ_LEN + pos];
            cpu_scores[h * SEQ_LEN + pos] = cpu;
            let abs = (cpu - gpu_value).abs();
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(abs / cpu.abs().max(gpu_value.abs()).max(1e-6));
        }
    }

    let pearson = triattn::pearson(&cpu_scores, &gpu_scores);
    eprintln!(
        "FWHT3 TriAttention GPU/CPU: max_abs={max_abs:.3e} max_rel={max_rel:.3e} r={pearson:.7}"
    );
    assert!(pearson > 0.9999, "ranking correlation too low: {pearson}");
    assert!(max_rel < 5e-3, "relative error too high: {max_rel}");
}
