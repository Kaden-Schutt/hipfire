// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Validation of the KVarN write path: `kvarn_gather_k_tiles` (token-major K →
//! channel-major `[head_dim × GROUP]` tiles) followed by the committed
//! `kvarn_quantize_tile`. Checks (a) the gather is a bit-exact transpose, and
//! (b) the gather+quantize pipeline reconstructs the original token-major K to
//! a high cos-sim per kv-head (the KVarN quality bar).
//!
//!   cargo run --release -p rdna-compute --example parity_kvarn_write_path

use rdna_compute::Gpu;

fn f16_to_f32(bits: u16) -> f32 {
    let s = (bits >> 15) & 1;
    let e = (bits >> 10) & 0x1f;
    let m = bits & 0x3ff;
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-24)
    } else if e == 31 {
        if m == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15)
    };
    if s == 1 {
        -v
    } else {
        v
    }
}

fn lcg_normal(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    let mut u = || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        (s as f32 + 0.5) / 2_147_483_648.0
    };
    (0..n)
        .map(|_| {
            let u1 = u().max(1e-7);
            let u2 = u();
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        })
        .collect()
}

fn cos_sim(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

fn main() {
    let head_dim = 128usize;
    let n_kv_heads = 2usize;
    let group = 128usize;
    let n_blocks = 2usize;
    let kv_dim = n_kv_heads * head_dim;
    let n_tokens = n_blocks * group;
    let n_tiles = n_blocks * n_kv_heads;
    let tile_elems = head_dim * group;

    let mut gpu = Gpu::init().unwrap();

    // Token-major K [n_tokens × kv_dim] with per-channel + per-token variance
    // spread (what KVarN targets). Channel c of head h scaled geometrically;
    // token t scaled geometrically.
    let base = lcg_normal(11, n_tokens * kv_dim);
    let mut k = vec![0.0f32; n_tokens * kv_dim];
    for t in 0..n_tokens {
        let tok_scale = 0.1f32 * 20f32.powf(t as f32 / n_tokens as f32);
        for j in 0..kv_dim {
            let ch = j % head_dim;
            let ch_scale = 0.05f32 * 40f32.powf(ch as f32 / head_dim as f32);
            k[t * kv_dim + j] = base[t * kv_dim + j] * tok_scale * ch_scale;
        }
    }

    // GPU gather.
    let kd = gpu
        .upload_raw(
            &k.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            &[n_tokens * kv_dim],
        )
        .unwrap();
    let td = gpu
        .upload_raw(
            &vec![0u8; n_tiles * tile_elems * 4],
            &[n_tiles * tile_elems],
        )
        .unwrap();
    gpu.kvarn_gather_k_tiles(&kd, &td, n_blocks, n_kv_heads, head_dim, group)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let tiles = gpu.download_f32(&td).unwrap();

    // (a) Bit-exact transpose check.
    let mut max_gather_err = 0.0f32;
    for b in 0..n_blocks {
        for h in 0..n_kv_heads {
            let tile = (b * n_kv_heads + h) * tile_elems;
            for ch in 0..head_dim {
                for tok in 0..group {
                    let got = tiles[tile + ch * group + tok];
                    let want = k[(b * group + tok) * kv_dim + h * head_dim + ch];
                    max_gather_err = max_gather_err.max((got - want).abs());
                }
            }
        }
    }

    // (b) Gather+quantize reconstruction. Quantize all tiles, host-dequant each
    // record, reconstruct token-major K, cos-sim vs the original.
    let record_bytes = (head_dim * group).div_ceil(2) + head_dim * 2 * 2 + group * 2;
    let rd = gpu
        .upload_raw(
            &vec![0u8; n_tiles * record_bytes],
            &[n_tiles * record_bytes],
        )
        .unwrap();
    gpu.kvarn_quantize_tile(&td, &rd, n_tiles, head_dim, group, record_bytes)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let recs = gpu.download_raw(&rd, n_tiles * record_bytes).unwrap();

    let qbytes = (head_dim * group).div_ceil(2);
    let off_scale = qbytes;
    let off_zp = off_scale + head_dim * 2;
    let off_scol = off_zp + head_dim * 2;
    let mut recon = vec![0.0f32; n_tokens * kv_dim];
    for b in 0..n_blocks {
        for h in 0..n_kv_heads {
            let rec = &recs[(b * n_kv_heads + h) * record_bytes..];
            let rd16 = |off: usize| f16_to_f32(u16::from_le_bytes([rec[off], rec[off + 1]]));
            for ch in 0..head_dim {
                let sa = rd16(off_scale + ch * 2);
                let za = rd16(off_zp + ch * 2);
                for tok in 0..group {
                    let gi = ch * group + tok;
                    let byte = rec[gi >> 1];
                    let q = if gi & 1 == 0 { byte & 0xf } else { byte >> 4 } as f32;
                    let sc = rd16(off_scol + tok * 2);
                    recon[(b * group + tok) * kv_dim + h * head_dim + ch] = (q * sa + za) * sc;
                }
            }
        }
    }
    let cs = cos_sim(&recon, &k);

    let pass = max_gather_err == 0.0 && cs >= 0.99;
    println!(
        "parity_kvarn_write_path on {}: gather-max-err={max_gather_err:.2e}  \
         gather+quantize cos-sim={cs:.5}  -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
