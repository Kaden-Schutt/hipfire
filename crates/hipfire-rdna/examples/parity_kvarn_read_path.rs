// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Validation of the KVarN read path: `kvarn_build_kcache` materializes a
//! token-major f16 shadow K from the block-tiled records (full blocks) + the
//! f32 recent-window (tail). Checks the full-block region matches a host dequant
//! of the records (transposed back to token-major) within f16 tolerance, and the
//! tail region equals f16(window). Records are produced by the validated
//! gather+quantize write path.
//!
//!   cargo run --release -p hipfire-rdna --example parity_kvarn_read_path

use hipfire_rdna::Gpu;

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

fn main() {
    let head_dim = 128usize;
    let n_kv_heads = 2usize;
    let group = 128usize;
    let n_full_blocks = 2usize;
    let tail_len = 40usize;
    let kv_dim = n_kv_heads * head_dim;
    let n_full = n_full_blocks * group;
    let n_tokens = n_full + tail_len;
    let n_tiles = n_full_blocks * n_kv_heads;
    let tile_elems = head_dim * group;

    let mut gpu = Gpu::init().unwrap();

    // Full-block token-major K, quantized to records via gather+quantize.
    let kbase = lcg_normal(11, n_full * kv_dim);
    let mut k = vec![0.0f32; n_full * kv_dim];
    for t in 0..n_full {
        let tok_scale = 0.1f32 * 20f32.powf(t as f32 / n_full as f32);
        for j in 0..kv_dim {
            let ch = j % head_dim;
            let ch_scale = 0.05f32 * 40f32.powf(ch as f32 / head_dim as f32);
            k[t * kv_dim + j] = kbase[t * kv_dim + j] * tok_scale * ch_scale;
        }
    }
    let kd = gpu
        .upload_raw(
            &k.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            &[n_full * kv_dim],
        )
        .unwrap();
    let td = gpu
        .upload_raw(
            &vec![0u8; n_tiles * tile_elems * 4],
            &[n_tiles * tile_elems],
        )
        .unwrap();
    gpu.kvarn_gather_k_tiles(&kd, &td, n_full_blocks, n_kv_heads, head_dim, group)
        .unwrap();
    let record_bytes = tile_elems.div_ceil(2) + head_dim * 2 * 2 + group * 2;
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

    // Window: [group × kv_dim] f32, first tail_len rows hold the tail tokens.
    let mut window = vec![0.0f32; group * kv_dim];
    let wbase = lcg_normal(23, tail_len * kv_dim);
    for t in 0..tail_len {
        for j in 0..kv_dim {
            window[t * kv_dim + j] = wbase[t * kv_dim + j] * 0.3;
        }
    }
    let wd = gpu
        .upload_raw(
            &window
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            &[group * kv_dim],
        )
        .unwrap();

    // Build the f16 shadow K.
    let outd = gpu
        .upload_raw(&vec![0u8; n_tokens * kv_dim * 2], &[n_tokens * kv_dim])
        .unwrap();
    gpu.kvarn_build_kcache(
        &rd,
        &wd,
        &outd,
        n_full_blocks,
        tail_len,
        n_kv_heads,
        head_dim,
        group,
        record_bytes,
    )
    .unwrap();
    gpu.device_synchronize().unwrap();
    let outb = gpu.download_raw(&outd, n_tokens * kv_dim * 2).unwrap();
    let out_f32: Vec<f32> = (0..n_tokens * kv_dim)
        .map(|i| f16_to_f32(u16::from_le_bytes([outb[i * 2], outb[i * 2 + 1]])))
        .collect();

    // Host dequant of records (token-major) for the full-block region.
    let qbytes = tile_elems.div_ceil(2);
    let off_scale = qbytes;
    let off_zp = off_scale + head_dim * 2;
    let off_scol = off_zp + head_dim * 2;
    let mut max_full_err = 0.0f32;
    for b in 0..n_full_blocks {
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
                    let want = (q * sa + za) * sc;
                    let got = out_f32[(b * group + tok) * kv_dim + h * head_dim + ch];
                    max_full_err = max_full_err.max((got - want).abs() / want.abs().max(1e-3));
                }
            }
        }
    }

    // Tail region: must equal f16(window).
    let mut max_tail_err = 0.0f32;
    for t in 0..tail_len {
        for j in 0..kv_dim {
            let want = f16_to_f32(crate_f32_to_f16(window[t * kv_dim + j]));
            let got = out_f32[(n_full + t) * kv_dim + j];
            max_tail_err = max_tail_err.max((got - want).abs());
        }
    }

    let pass = max_full_err < 5e-3 && max_tail_err == 0.0;
    println!(
        "parity_kvarn_read_path on {}: full-block max-rel-err={max_full_err:.2e}  \
         tail max-abs-err={max_tail_err:.2e}  -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}

// Round-to-nearest-even f32→f16 (matches HIP __float2half) for the tail check.
fn crate_f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp >= 0x1f {
        return sign | 0x7c00; // inf/overflow
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = mant | 0x80_0000;
        let shift = (14 - exp) as u32;
        let mut h = (mant >> shift) as u16;
        let round_bit = (mant >> (shift - 1)) & 1;
        if round_bit == 1 {
            let sticky = mant & ((1 << (shift - 1)) - 1);
            if sticky != 0 || (h & 1) == 1 {
                h += 1;
            }
        }
        return sign | h;
    }
    let mut h_mant = (mant >> 13) as u16;
    let round_bit = (mant >> 12) & 1;
    if round_bit == 1 {
        let sticky = mant & 0xfff;
        if sticky != 0 || (h_mant & 1) == 1 {
            h_mant += 1;
            if h_mant == 0x400 {
                h_mant = 0;
                exp += 1;
                if exp >= 0x1f {
                    return sign | 0x7c00;
                }
            }
        }
    }
    sign | ((exp as u16) << 10) | h_mant
}
