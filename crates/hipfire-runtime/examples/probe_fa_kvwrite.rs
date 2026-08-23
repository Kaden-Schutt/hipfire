// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Bitwise probe for the gfx1100 FA-prep KV-fold: runs the legacy sequence
//! (`qwen35_fa_prep_gfx1100` + `kv_cache_write_q8_0_pair`) and the folded
//! `qwen35_fa_prep_kvwrite_gfx1100` on identical synthetic input, then diffs
//! fa_q / fa_gate / fa_k AND every byte of both Q8_0 cache rows. Certified
//! shape only: 16Q/4K, head_dim 256, n_rot 64 (Qwen3.5-4B).
//!
//! Usage: probe_fa_kvwrite [out_prefix]

use rdna_compute::{DType, Gpu};

fn main() {
    let prefix = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/fakv".into());
    let mut gpu = Gpu::init().expect("GPU init failed");
    let n_q = 16usize;
    let n_kv = 4usize;
    let hd = 256usize;
    let n_rot = 64usize;
    let cap = 32usize;
    let blocks_per_head = hd / 32;
    let total_blocks = n_kv * blocks_per_head;
    let row_bytes = total_blocks * 34;
    let cache_bytes = cap * row_bytes;

    // Deterministic interleaved q/gate projection output: [16 heads x 512].
    let q_full: Vec<f32> = (0..n_q * hd * 2)
        .map(|i| ((i as f64 * 0.013).sin() * 1.7 + (i as f64 * 0.0071).cos() * 0.4) as f32)
        .collect();
    // Pre-norm K: [4 heads x 256]; include exact zeros and large magnitudes so
    // the amax==0 and clamp branches are exercised.
    let mut k_in: Vec<f32> = (0..n_kv * hd)
        .map(|i| ((i as f64 * 0.029).cos() * 1.3) as f32)
        .collect();
    for i in [5usize, 300, 777] {
        if i < k_in.len() {
            k_in[i] = 0.0;
        }
    }
    k_in[100] = -9.5e3;
    // V source: same treatment.
    let mut v_in: Vec<f32> = (0..n_kv * hd)
        .map(|i| ((i as f64 * 0.047).sin() * 2.9 + (i as f64 * 0.0031).cos()) as f32)
        .collect();
    for i in [11usize, 512] {
        if i < v_in.len() {
            v_in[i] = 0.0;
        }
    }
    v_in[900] = 7.25e4;
    let q_w: Vec<f32> = (0..hd)
        .map(|i| 0.8 + 0.4 * ((i as f64 * 0.01).sin()) as f32)
        .collect();
    let k_w: Vec<f32> = (0..hd)
        .map(|i| 0.9 + 0.2 * ((i as f64 * 0.013).cos()) as f32)
        .collect();

    let q_full_t = gpu.upload_f32(&q_full, &[n_q * hd * 2]).expect("upload");
    let v_t = gpu.upload_f32(&v_in, &[n_kv * hd]).expect("upload v");
    let qw_t = gpu.upload_f32(&q_w, &[hd]).expect("upload qw");
    let kw_t = gpu.upload_f32(&k_w, &[hd]).expect("upload kw");
    let pos: i32 = 17;
    let pos_buf = gpu.hip.malloc(4).expect("pos alloc");
    gpu.hip
        .memcpy_htod(&pos_buf, &pos.to_ne_bytes())
        .expect("pos copy");

    let fresh_caches = |gpu: &mut Gpu| -> (rdna_compute::GpuTensor, rdna_compute::GpuTensor) {
        (
            gpu.upload_raw(&vec![0u8; cache_bytes], &[cache_bytes])
                .expect("kc"),
            gpu.upload_raw(&vec![0u8; cache_bytes], &[cache_bytes])
                .expect("vc"),
        )
    };

    // ── Legacy arm ──
    let fa_q = gpu.alloc_tensor(&[n_q * hd], DType::F32).expect("q");
    let fa_gate = gpu.alloc_tensor(&[n_q * hd], DType::F32).expect("g");
    let fa_k = gpu.upload_f32(&k_in, &[n_kv * hd]).expect("k upload");
    gpu.deinterleave_f32(&q_full_t, &fa_q, &fa_gate, n_q, hd)
        .expect("deinterleave");
    gpu.rmsnorm_batched(&fa_q, &qw_t, &fa_q, n_q, hd, 1e-6)
        .expect("qnorm");
    gpu.rmsnorm_batched(&fa_k, &kw_t, &fa_k, n_kv, hd, 1e-6)
        .expect("knorm");
    gpu.rope_partial_interleaved_f32(&fa_q, &fa_k, &pos_buf, n_q, n_kv, hd, n_rot, 1_000_000.0)
        .expect("rope");
    let (kc_l, vc_l) = fresh_caches(&mut gpu);
    gpu.kv_cache_write_q8_0_pair(&kc_l, &vc_l, &fa_k, &v_t, &pos_buf, n_kv, hd)
        .expect("pair write");
    gpu.hip.device_synchronize().expect("sync");
    let l_q = gpu.download_f32(&fa_q).unwrap();
    let l_g = gpu.download_f32(&fa_gate).unwrap();
    let l_k = gpu.download_f32(&fa_k).unwrap();
    let l_kc = download_bytes(&gpu, &kc_l);
    let l_vc = download_bytes(&gpu, &vc_l);

    // ── Folded arm (fresh buffers + fresh caches) ──
    let fa_q2 = gpu.alloc_tensor(&[n_q * hd], DType::F32).expect("q2");
    let fa_g2 = gpu.alloc_tensor(&[n_q * hd], DType::F32).expect("g2");
    let fa_k2 = gpu.upload_f32(&k_in, &[n_kv * hd]).expect("k2");
    let (kc_f, vc_f) = fresh_caches(&mut gpu);
    gpu.qwen35_fa_prep_kvwrite_gfx1100(
        &q_full_t,
        &fa_q2,
        &fa_g2,
        &fa_k2,
        &v_t,
        &kc_f,
        &vc_f,
        &qw_t,
        &kw_t,
        &pos_buf,
        1e-6,
        1_000_000.0,
        n_q,
        n_kv,
    )
    .expect("folded prep+kvwrite");
    gpu.hip.device_synchronize().expect("sync2");
    let f_q = gpu.download_f32(&fa_q2).unwrap();
    let f_g = gpu.download_f32(&fa_g2).unwrap();
    let f_k = gpu.download_f32(&fa_k2).unwrap();
    let f_kc = download_bytes(&gpu, &kc_f);
    let f_vc = download_bytes(&gpu, &vc_f);

    let mut all_ok = true;
    for (name, a, b) in [
        ("fa_q", &l_q, &f_q),
        ("fa_gate", &l_g, &f_g),
        ("fa_k", &l_k, &f_k),
    ] {
        let diffs: Vec<usize> = a
            .iter()
            .zip(b.iter())
            .enumerate()
            .filter(|(_, (x, y))| x != y)
            .map(|(i, _)| i)
            .collect();
        if diffs.is_empty() {
            eprintln!("{name}: IDENTICAL ({} elems)", a.len());
        } else {
            all_ok = false;
            eprintln!("{name}: {} DIFFS; first idx {}", diffs.len(), diffs[0]);
        }
    }
    // Cache comparison: whole buffer (untouched rows must stay zero in BOTH).
    let row = pos as usize;
    for (name, a, b) in [("k_cache", &l_kc, &f_kc), ("v_cache", &l_vc, &f_vc)] {
        if a == b {
            let written: usize = a[row * row_bytes..(row + 1) * row_bytes]
                .iter()
                .filter(|&&x| x != 0)
                .count();
            eprintln!(
                "{name}: IDENTICAL ({cache_bytes} bytes; pos row has {written} nonzero bytes)"
            );
        } else {
            all_ok = false;
            let diffs: Vec<usize> = a
                .iter()
                .zip(b.iter())
                .enumerate()
                .filter(|(_, (x, y))| x != y)
                .map(|(i, _)| i)
                .collect();
            eprintln!(
                "{name}: {} BYTE DIFFS; first {} legacy={} folded={}",
                diffs.len(),
                diffs[0],
                a[diffs[0]],
                b[diffs[0]]
            );
        }
    }

    let _ = prefix; // reserved for future dump-and-exit debugging
    if all_ok {
        eprintln!("PROBE PASS: fold output + cache bytes bitwise identical to legacy");
    } else {
        eprintln!("PROBE FAIL");
        std::process::exit(1);
    }
}

fn download_bytes(gpu: &Gpu, t: &rdna_compute::GpuTensor) -> Vec<u8> {
    let mut data = vec![0u8; t.byte_size()];
    gpu.hip.memcpy_dtoh(&mut data, &t.buf).expect("dtoh");
    data
}
