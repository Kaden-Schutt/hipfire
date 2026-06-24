// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Validation of the fused KVarN flash (Phase D2,
//! `attention_flash_kvarn_batched_masked`) against (a) an f64 host reference
//! flash over the EXACT dequantized records + f32 window + dequantized Q8 V, and
//! (b) the v1 read path (`kvarn_build_kcache` → f16-K/Q8-V flash). Sweeps several
//! (n_full_blocks, tail_len) configs incl. pure-window and exact-boundary. Both
//! GPU paths should track the host ref; whichever is farther has a bug.
//!
//!   cargo run --release -p rdna-compute --example parity_kvarn_fused_flash

use rdna_compute::{DType, Gpu};

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

fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = mant | 0x80_0000;
        let shift = (14 - exp) as u32;
        let mut h = (mant >> shift) as u16;
        if (mant >> (shift - 1)) & 1 == 1 {
            let sticky = mant & ((1 << (shift - 1)) - 1);
            if sticky != 0 || (h & 1) == 1 {
                h += 1;
            }
        }
        return sign | h;
    }
    let mut h_mant = (mant >> 13) as u16;
    if (mant >> 12) & 1 == 1 {
        let sticky = mant & 0xfff;
        if sticky != 0 || (h_mant & 1) == 1 {
            h_mant += 1;
            if h_mant == 0x400 {
                h_mant = 0;
                exp += 1;
            }
        }
    }
    sign | ((exp as u16) << 10) | h_mant
}

fn lcg(seed: u32, n: usize) -> Vec<f32> {
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
    let mut gpu = Gpu::init().unwrap();
    let mut all_pass = true;
    for &(nfb, tail) in &[
        (0usize, 50usize),
        (0, 127),
        (1, 0),
        (1, 13),
        (2, 40),
        (3, 1),
    ] {
        all_pass &= run_case(&mut gpu, nfb, tail);
    }
    if !all_pass {
        std::process::exit(1);
    }
}

fn run_case(gpu: &mut Gpu, n_full_blocks: usize, tail_len: usize) -> bool {
    let n_heads = 4usize;
    let n_kv_heads = 2usize;
    let head_dim = 256usize;
    let group = 128usize;
    let kv_dim = n_kv_heads * head_dim;
    let n_full = n_full_blocks * group;
    let seq_len = (n_full + tail_len).max(1);
    let max_seq = 768usize;
    let blocks_per_head = head_dim / 32;
    let v_row_stride = n_kv_heads * blocks_per_head * 34;
    let tile_elems = head_dim * group;
    let record_bytes = tile_elems.div_ceil(2) + head_dim * 2 * 2 + group * 2;

    // K for the full-block region.
    let kbase = lcg(11, n_full.max(1) * kv_dim);
    let mut k = vec![0.0f32; n_full * kv_dim];
    for t in 0..n_full {
        let tok_scale = 0.1f32 * 20f32.powf(t as f32 / n_full.max(1) as f32);
        for j in 0..kv_dim {
            let ch_scale = 0.05f32 * 40f32.powf((j % head_dim) as f32 / head_dim as f32);
            k[t * kv_dim + j] = kbase[t * kv_dim + j] * tok_scale * ch_scale;
        }
    }
    let n_blocks_alloc = max_seq.div_ceil(group);
    let rec_buf_bytes = (n_blocks_alloc * n_kv_heads * record_bytes).next_multiple_of(4);
    let rd = gpu
        .upload_raw(&vec![0u8; rec_buf_bytes], &[rec_buf_bytes / 4])
        .unwrap();
    if n_full_blocks > 0 {
        let kd = gpu
            .upload_raw(
                &k.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
                &[n_full * kv_dim],
            )
            .unwrap();
        let n_tiles = n_full_blocks * n_kv_heads;
        let td = gpu
            .upload_raw(
                &vec![0u8; n_tiles * tile_elems * 4],
                &[n_tiles * tile_elems],
            )
            .unwrap();
        gpu.kvarn_gather_k_tiles(&kd, &td, n_full_blocks, n_kv_heads, head_dim, group)
            .unwrap();
        gpu.kvarn_quantize_tile(&td, &rd, n_tiles, head_dim, group, record_bytes)
            .unwrap();
    }
    let recs = gpu.download_raw(&rd, rec_buf_bytes).unwrap();

    // Window [group × kv_dim] f32, first tail_len rows.
    let mut window = vec![0.0f32; group * kv_dim];
    let wbase = lcg(23, tail_len.max(1) * kv_dim);
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

    // Q8 V cache + host f32 dequant.
    let vbase = lcg(7, seq_len * kv_dim);
    let mut v_cache = vec![0u8; max_seq * v_row_stride];
    let mut v_deq = vec![0.0f32; seq_len * kv_dim];
    for t in 0..seq_len {
        for kvh in 0..n_kv_heads {
            for b in 0..blocks_per_head {
                let mut amax = 0.0f32;
                for e in 0..32 {
                    amax = amax.max((vbase[t * kv_dim + kvh * head_dim + b * 32 + e] * 0.3).abs());
                }
                let scale = (amax / 127.0).max(1e-8);
                let blk = t * v_row_stride + (kvh * blocks_per_head + b) * 34;
                v_cache[blk..blk + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
                for e in 0..32 {
                    let x = vbase[t * kv_dim + kvh * head_dim + b * 32 + e] * 0.3;
                    let qd = (x / scale).round().clamp(-127.0, 127.0) as i8;
                    v_cache[blk + 2 + e] = qd as u8;
                    v_deq[t * kv_dim + kvh * head_dim + b * 32 + e] = scale * qd as f32;
                }
            }
        }
    }
    let vd = gpu.upload_raw(&v_cache, &[max_seq * v_row_stride]).unwrap();

    // Host K (token-major f32) = dequant records (full blocks) + window (tail).
    let qbytes = tile_elems.div_ceil(2);
    let off_scale = qbytes;
    let off_zp = off_scale + head_dim * 2;
    let off_scol = off_zp + head_dim * 2;
    let mut k_host = vec![0.0f32; seq_len * kv_dim];
    for b in 0..n_full_blocks {
        for kvh in 0..n_kv_heads {
            let rec = &recs[(b * n_kv_heads + kvh) * record_bytes..];
            let rd16 = |o: usize| f16_to_f32(u16::from_le_bytes([rec[o], rec[o + 1]]));
            for ch in 0..head_dim {
                let sa = rd16(off_scale + ch * 2);
                let za = rd16(off_zp + ch * 2);
                for c in 0..group {
                    let gi = ch * group + c;
                    let byte = rec[gi >> 1];
                    let q = if gi & 1 == 0 { byte & 0xf } else { byte >> 4 } as f32;
                    let sc = rd16(off_scol + c * 2);
                    k_host[(b * group + c) * kv_dim + kvh * head_dim + ch] = (q * sa + za) * sc;
                }
            }
        }
    }
    for t in 0..tail_len {
        for j in 0..kv_dim {
            k_host[(n_full + t) * kv_dim + j] = window[t * kv_dim + j];
        }
    }

    // Host f64 flash, single query at last position.
    let q: Vec<f32> = lcg(3, n_heads * head_dim).iter().map(|v| v * 0.2).collect();
    let kv_group = n_heads / n_kv_heads;
    let scale_attn = 1.0f64 / (head_dim as f64).sqrt();
    let mut ref_out = vec![0.0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let kvh = h / kv_group;
        let mut sc = vec![0.0f64; seq_len];
        let mut mx = f64::NEG_INFINITY;
        for t in 0..seq_len {
            let mut s = 0.0f64;
            for d in 0..head_dim {
                s += q[h * head_dim + d] as f64 * k_host[t * kv_dim + kvh * head_dim + d] as f64;
            }
            s *= scale_attn;
            sc[t] = s;
            mx = mx.max(s);
        }
        let mut sum = 0.0f64;
        for t in 0..seq_len {
            sc[t] = (sc[t] - mx).exp();
            sum += sc[t];
        }
        for d in 0..head_dim {
            let mut acc = 0.0f64;
            for t in 0..seq_len {
                acc += sc[t] * v_deq[t * kv_dim + kvh * head_dim + d] as f64;
            }
            ref_out[h * head_dim + d] = (acc / sum) as f32;
        }
    }

    let qd = gpu
        .upload_raw(
            &q.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            &[n_heads * head_dim],
        )
        .unwrap();
    let posd = gpu
        .upload_raw(&((seq_len - 1) as i32).to_le_bytes(), &[1])
        .unwrap();
    let max_tiles = max_seq.div_ceil(group);
    let partials = gpu
        .zeros(&[n_heads * max_tiles * (2 + head_dim)], DType::F32)
        .unwrap();

    // v1 path.
    let shadow = gpu.zeros(&[seq_len * kv_dim], DType::F16).unwrap();
    gpu.kvarn_build_kcache(
        &rd,
        &wd,
        &shadow,
        n_full_blocks,
        tail_len,
        n_kv_heads,
        head_dim,
        group,
        record_bytes,
    )
    .unwrap();
    let out_a = gpu
        .upload_raw(&vec![0u8; n_heads * head_dim * 4], &[n_heads * head_dim])
        .unwrap();
    gpu.attention_flash_f16k_q8v_batched_masked(
        &qd, &shadow, &vd, &out_a, &posd, n_heads, n_kv_heads, head_dim, max_seq, seq_len, 1,
        &partials, None, 0, 0,
    )
    .unwrap();

    // fused path.
    let out_b = gpu
        .upload_raw(&vec![0u8; n_heads * head_dim * 4], &[n_heads * head_dim])
        .unwrap();
    gpu.attention_flash_kvarn_batched_masked(
        &qd,
        &rd,
        &wd,
        &vd,
        &out_b,
        &posd,
        n_heads,
        n_kv_heads,
        head_dim,
        max_seq,
        seq_len,
        1,
        &partials,
        None,
        0,
        0,
        n_full_blocks,
        record_bytes,
    )
    .unwrap();
    gpu.device_synchronize().unwrap();

    let a = gpu.download_f32(&out_a).unwrap();
    let b = gpu.download_f32(&out_b).unwrap();
    let err = |x: &[f32]| {
        let mut m = 0.0f32;
        for i in 0..x.len() {
            m = m.max((x[i] - ref_out[i]).abs());
        }
        m
    };
    let v1_err = err(&a);
    let fused_err = err(&b);
    let mut ab = 0.0f32;
    for i in 0..a.len() {
        ab = ab.max((a[i] - b[i]).abs());
    }
    let pass = fused_err < 2e-3 && fused_err <= v1_err * 3.0 + 1e-4;
    println!("  n_full={n_full_blocks} tail={tail_len} seq={seq_len}: v1-vs-host={v1_err:.2e}  fused-vs-host={fused_err:.2e}  fused-vs-v1={ab:.2e}  -> {}", if pass { "PASS" } else { "FAIL" });
    pass
}
