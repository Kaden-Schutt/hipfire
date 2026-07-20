// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Numerical suitability oracle: raw attention_q8_0_kv vs attention_flash_q8_0
// (tile+reduce) at the LFM2.5 350M full-attention shape on the host GPU.
//
// Shape under test:
//   n_heads=16, n_kv_heads=8, head_dim=64, Q8 KV, max_seq=2048
//
// Shared-kernel suitability contract (PASS requires all):
//   1. every raw and flash element is finite
//   2. global max |raw-flash| <= 1e-4
//   3. global RMS(raw-flash)  <= 1e-5
//   4. per-head max |raw-flash| <= 1e-4  (head failures cannot hide)
//
// Bit-identical is reported only. Raw serial QK vs flash wave-reduced FP
// association is known by source to differ, so bitwise equality is not a gate.
//
// Run:
//   flock /tmp/hipfire-gpu.lock cargo run -p rdna-compute --release \
//       --example lfm_q8_flash_parity_oracle

use rdna_compute::attention::q8_flash_tile_size;
use rdna_compute::{DType, Gpu};

const N_HEADS: usize = 16;
const N_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const MAX_SEQ: usize = 2048;
const Q8_BLOCK: usize = 34;
const Q8_GROUP: usize = 32;

/// Shared suitability ceilings.
const MAX_ABS_LIMIT: f32 = 1e-4;
const RMS_LIMIT: f32 = 1e-5;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) & 0x7fff) as f32 / 32_768.0 - 0.5
        })
        .collect()
}

fn unique_sorted(mut xs: Vec<usize>) -> Vec<usize> {
    xs.sort_unstable();
    xs.dedup();
    xs
}

fn main() {
    assert_eq!(
        N_HEADS % N_KV_HEADS,
        0,
        "GQA requires n_heads % n_kv_heads == 0"
    );
    assert_eq!(
        HEAD_DIM % Q8_GROUP,
        0,
        "head_dim must be a multiple of 32 for Q8_0"
    );
    assert!(MAX_SEQ > 0);

    let q_dim = N_HEADS * HEAD_DIM;
    let kv_dim = N_KV_HEADS * HEAD_DIM;
    let blocks_per_head = HEAD_DIM / Q8_GROUP;
    let total_blocks = N_KV_HEADS * blocks_per_head;
    let cache_bytes = MAX_SEQ * total_blocks * Q8_BLOCK;
    // Production Q8 cache pads byte capacity into F32 element slots.
    let cache_elems = (cache_bytes + 3) / 4;

    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(
        gpu.arch.as_str(),
        "gfx1201",
        "oracle requires exact gpu.arch gfx1201, got {}",
        gpu.arch
    );
    let tile_size = q8_flash_tile_size(&gpu.arch, N_HEADS, N_KV_HEADS, HEAD_DIM, MAX_SEQ);
    assert!(
        tile_size.is_power_of_two() && tile_size >= 16 && tile_size <= 256,
        "unexpected q8_flash_tile_size={tile_size}"
    );
    let max_tiles = MAX_SEQ.div_ceil(tile_size);
    let partials_len = N_HEADS * max_tiles * (2 + HEAD_DIM);

    eprintln!(
        "lfm_q8_flash_parity_oracle arch={} n_heads={} n_kv_heads={} head_dim={} max_seq={} tile_size={} max_tiles={}",
        gpu.arch, N_HEADS, N_KV_HEADS, HEAD_DIM, MAX_SEQ, tile_size, max_tiles
    );
    eprintln!(
        "contract: finite && max_abs<={MAX_ABS_LIMIT:.0e} && rms<={RMS_LIMIT:.0e} && per_head_max_abs<={MAX_ABS_LIMIT:.0e}"
    );
    eprintln!(
        "buffers: q_dim={} kv_dim={} cache_bytes={} cache_elems={} partials_len={}",
        q_dim, kv_dim, cache_bytes, cache_elems, partials_len
    );

    // Deterministic host tensors.
    let q_host = lcg(0xa5a5_0350, q_dim);
    let k_host = lcg(0xc3c3_0350, MAX_SEQ * kv_dim);
    let v_host = lcg(0x9696_0350, MAX_SEQ * kv_dim);
    assert_eq!(q_host.len(), q_dim);
    assert_eq!(k_host.len(), MAX_SEQ * kv_dim);
    assert_eq!(v_host.len(), MAX_SEQ * kv_dim);

    let d_q = gpu.upload_f32(&q_host, &[q_dim]).expect("upload q");
    let d_k_f32 = gpu
        .upload_f32(&k_host, &[MAX_SEQ * kv_dim])
        .expect("upload k f32");
    let d_v_f32 = gpu
        .upload_f32(&v_host, &[MAX_SEQ * kv_dim])
        .expect("upload v f32");

    // Q8 KV via the production quantize path.
    let d_k_q8 = gpu.zeros(&[cache_elems], DType::F32).expect("alloc k q8");
    let d_v_q8 = gpu.zeros(&[cache_elems], DType::F32).expect("alloc v q8");
    let pos_all_i32: Vec<i32> = (0..MAX_SEQ as i32).collect();
    let pos_all_bytes: Vec<u8> = pos_all_i32.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let pos_all_t = gpu.zeros(&[MAX_SEQ], DType::F32).expect("alloc pos_all");
    gpu.hip
        .memcpy_htod(&pos_all_t.buf, &pos_all_bytes)
        .expect("upload pos_all");

    gpu.kv_cache_write_q8_0_batched(&d_k_q8, &d_k_f32, &pos_all_t, N_KV_HEADS, HEAD_DIM, MAX_SEQ)
        .expect("quantize k");
    gpu.kv_cache_write_q8_0_batched(&d_v_q8, &d_v_f32, &pos_all_t, N_KV_HEADS, HEAD_DIM, MAX_SEQ)
        .expect("quantize v");
    gpu.hip.device_synchronize().expect("sync after kv write");

    let d_out_raw = gpu.zeros(&[q_dim], DType::F32).expect("out raw");
    let d_out_flash = gpu.zeros(&[q_dim], DType::F32).expect("out flash");
    let d_partials = gpu
        .zeros(&[partials_len], DType::F32)
        .expect("flash partials");
    assert_eq!(d_out_raw.numel(), q_dim);
    assert_eq!(d_out_flash.numel(), q_dim);
    assert_eq!(d_partials.numel(), partials_len);

    let pos_buf = gpu.hip.malloc(4).expect("pos_buf");

    // Positions spanning tile boundaries and the max-context tail.
    let t = tile_size;
    let positions = unique_sorted(vec![
        0,
        1,
        t.saturating_sub(2),
        t.saturating_sub(1),
        t,
        t + 1,
        2 * t - 2,
        2 * t - 1,
        2 * t,
        MAX_SEQ - 2,
        MAX_SEQ - 1,
    ]);
    for &p in &positions {
        assert!(p < MAX_SEQ, "position {p} out of range");
    }
    eprintln!("positions: {positions:?}");

    let mut global_bit_identical = true;
    let mut global_max_abs = 0.0f32;
    let mut global_max_rel = 0.0f32;
    let mut global_sum_sq = 0.0f64;
    let mut global_count = 0usize;
    let mut global_worst_pos = 0usize;
    let mut global_worst_idx = 0usize;
    let mut global_worst_raw = 0.0f32;
    let mut global_worst_flash = 0.0f32;
    let mut per_head_max_abs = vec![0.0f32; N_HEADS];
    let mut per_head_worst_pos = vec![0usize; N_HEADS];
    let mut per_head_worst_idx = vec![0usize; N_HEADS];
    let mut any_nonfinite = false;
    let mut total_raw_nonfinite = 0usize;
    let mut total_flash_nonfinite = 0usize;
    let mut failed = false;

    for &pos in &positions {
        let seq_len = pos + 1;
        gpu.hip
            .memcpy_htod(&pos_buf, &(pos as i32).to_ne_bytes())
            .expect("upload pos");

        gpu.attention_q8_0_kv(
            &d_q, &d_k_q8, &d_v_q8, &d_out_raw, &pos_buf, seq_len, N_HEADS, N_KV_HEADS, HEAD_DIM,
            MAX_SEQ,
        )
        .expect("raw attention_q8_0_kv");

        gpu.attention_flash_q8_0(
            &d_q,
            &d_k_q8,
            &d_v_q8,
            &d_out_flash,
            &pos_buf,
            seq_len,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            MAX_SEQ,
            &d_partials,
        )
        .expect("flash attention_flash_q8_0");

        gpu.hip.device_synchronize().expect("sync after attn");

        let raw = gpu.download_f32(&d_out_raw).expect("download raw");
        let flash = gpu.download_f32(&d_out_flash).expect("download flash");
        assert_eq!(raw.len(), q_dim);
        assert_eq!(flash.len(), q_dim);

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut sum_sq = 0.0f64;
        let mut worst_idx = 0usize;
        let mut worst_raw = 0.0f32;
        let mut worst_flash = 0.0f32;
        let mut bit_identical = true;
        let mut raw_nonfinite = 0usize;
        let mut flash_nonfinite = 0usize;
        let mut pos_head_max = vec![0.0f32; N_HEADS];

        for (i, (a, b)) in raw.iter().zip(flash.iter()).enumerate() {
            let head = i / HEAD_DIM;
            if !a.is_finite() {
                raw_nonfinite += 1;
            }
            if !b.is_finite() {
                flash_nonfinite += 1;
            }
            if a.to_bits() != b.to_bits() {
                bit_identical = false;
            }

            // Non-finite pairs count as +inf so NaNs surface as the worst error.
            let e = if a.is_finite() && b.is_finite() {
                (a - b).abs()
            } else {
                f32::INFINITY
            };
            if e > max_abs {
                max_abs = e;
                worst_idx = i;
                worst_raw = *a;
                worst_flash = *b;
            }
            if e > pos_head_max[head] {
                pos_head_max[head] = e;
            }
            if e > per_head_max_abs[head] {
                per_head_max_abs[head] = e;
                per_head_worst_pos[head] = pos;
                per_head_worst_idx[head] = i;
            }
            if a.is_finite() && b.is_finite() {
                let denom = a.abs().max(1e-6);
                max_rel = max_rel.max(e / denom);
                sum_sq += (e as f64) * (e as f64);
            } else {
                max_rel = f32::INFINITY;
                sum_sq = f64::INFINITY;
            }
        }

        let n = q_dim as f64;
        let rms = if sum_sq.is_finite() {
            (sum_sq / n).sqrt() as f32
        } else {
            f32::INFINITY
        };

        let nonfinite = raw_nonfinite > 0 || flash_nonfinite > 0;
        if nonfinite {
            any_nonfinite = true;
        }
        total_raw_nonfinite += raw_nonfinite;
        total_flash_nonfinite += flash_nonfinite;

        if !bit_identical {
            global_bit_identical = false;
        }
        if max_abs > global_max_abs {
            global_max_abs = max_abs;
            global_max_rel = max_rel;
            global_worst_pos = pos;
            global_worst_idx = worst_idx;
            global_worst_raw = worst_raw;
            global_worst_flash = worst_flash;
        }
        if sum_sq.is_finite() && global_sum_sq.is_finite() {
            global_sum_sq += sum_sq;
            global_count += q_dim;
        } else {
            global_sum_sq = f64::INFINITY;
        }

        let head_max_str: String = pos_head_max
            .iter()
            .enumerate()
            .map(|(h, v)| format!("h{h}={v:.3e}"))
            .collect::<Vec<_>>()
            .join(",");

        println!(
            "pos={pos:<5} seq_len={seq_len:<5} bit_identical={bit_identical} max_abs={max_abs:.6e} max_rel={max_rel:.6e} rms={rms:.6e} worst_idx={worst_idx} raw={worst_raw:.8e} flash={worst_flash:.8e} raw_nf={raw_nonfinite} flash_nf={flash_nonfinite} per_head=[{head_max_str}]"
        );

        if nonfinite {
            eprintln!("FAIL pos={pos}: non-finite value in raw or flash output");
            failed = true;
        }
        if !(max_abs <= MAX_ABS_LIMIT) {
            eprintln!(
                "FAIL pos={pos}: max_abs={max_abs:.6e} exceeds limit {MAX_ABS_LIMIT:.0e} (worst_idx={worst_idx})"
            );
            failed = true;
        }
        if !(rms <= RMS_LIMIT) {
            eprintln!("FAIL pos={pos}: rms={rms:.6e} exceeds limit {RMS_LIMIT:.0e}");
            failed = true;
        }
        for (h, &hm) in pos_head_max.iter().enumerate() {
            if !(hm <= MAX_ABS_LIMIT) {
                eprintln!(
                    "FAIL pos={pos}: per-head max_abs h{h}={hm:.6e} exceeds limit {MAX_ABS_LIMIT:.0e}"
                );
                failed = true;
            }
        }
    }

    let global_rms = if global_sum_sq.is_finite() && global_count > 0 {
        (global_sum_sq / global_count as f64).sqrt() as f32
    } else {
        f32::INFINITY
    };

    // Global per-head gate (aggregated across positions).
    for (h, &hm) in per_head_max_abs.iter().enumerate() {
        if !(hm <= MAX_ABS_LIMIT) {
            eprintln!(
                "FAIL global: per-head max_abs h{h}={hm:.6e} exceeds limit {MAX_ABS_LIMIT:.0e} (pos={} idx={})",
                per_head_worst_pos[h], per_head_worst_idx[h]
            );
            failed = true;
        }
    }
    if any_nonfinite {
        failed = true;
    }
    if !(global_max_abs <= MAX_ABS_LIMIT) {
        eprintln!("FAIL global: max_abs={global_max_abs:.6e} exceeds limit {MAX_ABS_LIMIT:.0e}");
        failed = true;
    }
    if !(global_rms <= RMS_LIMIT) {
        eprintln!("FAIL global: rms={global_rms:.6e} exceeds limit {RMS_LIMIT:.0e}");
        failed = true;
    }

    println!();
    println!("=== SUMMARY ===");
    println!("path=crates/rdna-compute/examples/lfm_q8_flash_parity_oracle.rs");
    println!("arch={}", gpu.arch);
    println!("tile_size={tile_size}");
    println!("max_tiles={max_tiles}");
    println!("positions={positions:?}");
    println!("contract_max_abs_limit={MAX_ABS_LIMIT:.0e}");
    println!("contract_rms_limit={RMS_LIMIT:.0e}");
    println!("bit_identical={global_bit_identical}");
    println!("max_abs={global_max_abs:.6e}");
    println!("max_rel={global_max_rel:.6e}");
    println!("rms={global_rms:.6e}");
    println!(
        "worst_pos={global_worst_pos} worst_idx={global_worst_idx} worst_raw={global_worst_raw:.8e} worst_flash={global_worst_flash:.8e}"
    );
    println!("any_nonfinite={any_nonfinite}");
    println!("total_raw_nonfinite={total_raw_nonfinite}");
    println!("total_flash_nonfinite={total_flash_nonfinite}");
    for h in 0..N_HEADS {
        println!(
            "per_head_max_abs h{h}={:.6e} worst_pos={} worst_idx={}",
            per_head_max_abs[h], per_head_worst_pos[h], per_head_worst_idx[h]
        );
    }

    if failed {
        eprintln!(
            "ORACLE FAIL: attention_flash_q8_0 unsuitable for LFM2.5-350M under shared contract (finite, max_abs<={MAX_ABS_LIMIT:.0e}, rms<={RMS_LIMIT:.0e}, per-head)"
        );
        std::process::exit(1);
    }
    println!(
        "ORACLE PASS: suitability contract met (finite, max_abs<={MAX_ABS_LIMIT:.0e}, rms<={RMS_LIMIT:.0e}, per-head); bit_identical={global_bit_identical}"
    );
}
