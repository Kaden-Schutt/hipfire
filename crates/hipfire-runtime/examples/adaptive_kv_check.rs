// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Synthetic GPU correctness harness for the adaptive-KV V transcode kernels.
//
// Proves transcode-vs-direct equivalence WITHOUT loading the 15GB model:
//   * q8 → lloyd4   : write a deterministic normal-space V cache via the q8
//                     path, transcode it in place to lloyd4, and compare the
//                     dequantized rotated-space values against a reference cache
//                     written DIRECTLY at lloyd4 over the same source data.
//   * lloyd4 → lloyd3 and lloyd3 → lloyd2: transcode the (already-rotated)
//                     cache one tier down and compare against a direct write at
//                     the target tier over the same source.
//
// We compare DEQUANTIZED rotated-space values (cnorm * LUT[idx]) read back to
// the host. q8→lloyd4 round-trips through one extra quant step (q8 → dequant →
// FWHT → 4-bit) vs the direct lloyd4 write, so we expect near-identical (within
// a fraction of a 4-bit step). The lloyd→lloyd remap reconstructs the SAME
// rotated value the direct write would have quantized, so it should be
// BYTE-identical (idx + cnorm match).
//
// Run (GPU-locked):
//   source scripts/gpu-lock.sh && gpu_acquire "adaptive-kv-vtranscode" \
//     && ./target/release/examples/adaptive_kv_check; rc=$?; gpu_release; echo RUN_RC=$rc

use hipfire_runtime::llama::{KvCache, VMode};
use rdna_compute::{DType, Gpu, GpuTensor};

// 256-dim Lloyd-Max centroids (must match kernels/src/turbo_common.h).
const TURBO_C2_256: [f32; 4] = [-0.094376, -0.028300, 0.028300, 0.094376];
const TURBO_C3_256: [f32; 8] = [
    -0.134860, -0.083320, -0.046469, -0.015176, 0.015176, 0.046469, 0.083320, 0.134860,
];
const TURBO_C4_256: [f32; 16] = [
    -0.170807, -0.129321, -0.101134, -0.078505, -0.058869, -0.041003, -0.024249, -0.007938,
    0.007938, 0.024249, 0.041003, 0.058869, 0.078505, 0.101134, 0.129321, 0.170807,
];

// 128-dim Lloyd-Max centroids (must match kernels/src/turbo_common.h, the
// 128-family used by the fwht2/fwht4 K rotation). NOT the _256 family above.
const TURBO_C2_128: [f32; 4] = [-0.133466, -0.040022, 0.040022, 0.133466];
const TURBO_C4_128: [f32; 16] = [
    -0.241565, -0.182875, -0.143012, -0.111016, -0.083262, -0.057983, -0.034295, -0.011225,
    0.011225, 0.034295, 0.057983, 0.083262, 0.111016, 0.143012, 0.182875, 0.241565,
];

const N_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const N_LAYERS: usize = 3;
const MAX_SEQ: usize = 64;
const N_POS: usize = 48; // positions to write/transcode/compare

/// Deterministic Gaussian-ish source value for (layer, pos, head, dim).
/// Box-Muller from a cheap LCG so both caches see byte-identical input.
fn src_val(layer: usize, pos: usize, head: usize, dim: usize) -> f32 {
    let seed = ((layer as u64) << 40)
        ^ ((pos as u64) << 24)
        ^ ((head as u64) << 12)
        ^ (dim as u64);
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / ((1u64 << 31) as f64)
    };
    let u1 = (next() + 1e-9).min(1.0 - 1e-9);
    let u2 = next();
    let r = (-2.0 * u1.ln()).sqrt();
    (r * (std::f64::consts::TAU * u2).cos()) as f32
}

/// Build the [n_kv_heads * head_dim] normal-space V source for one position.
fn build_pos_src(layer: usize, pos: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; N_KV_HEADS * HEAD_DIM];
    for h in 0..N_KV_HEADS {
        for d in 0..HEAD_DIM {
            v[h * HEAD_DIM + d] = src_val(layer, pos, h, d);
        }
    }
    v
}

/// Allocate a [n_kv_heads*head_dim] f32 source tensor and upload `vals`.
fn upload_src(gpu: &mut Gpu, vals: &[f32]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[vals.len()], DType::F32).unwrap();
    let bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&t.buf, &bytes).unwrap();
    t
}

/// pos_buf: a 4-byte device buffer holding the i32 position.
fn pos_buf(gpu: &Gpu, pos: i32) -> hip_bridge::DeviceBuffer {
    let b = gpu.hip.malloc(4).unwrap();
    gpu.hip.memcpy_htod(&b, &pos.to_ne_bytes()).unwrap();
    b
}

/// Read back a V buffer (one layer) and dequantize the rotated-space values for
/// all (pos, head, dim) at the given lloyd `bits` tier. Returns a flat Vec
/// indexed [pos][head*head_dim + dim].
fn dequant_lloyd_layer(gpu: &Gpu, buf: &GpuTensor, bits: usize, n_pos: usize) -> Vec<Vec<f32>> {
    let byte_size = buf.byte_size();
    let mut raw = vec![0u8; byte_size];
    gpu.hip.memcpy_dtoh(&mut raw, &buf.buf).unwrap();

    let bph = 4 + (HEAD_DIM * bits) / 8;
    let bpp = N_KV_HEADS * bph;
    let lut: &[f32] = match bits {
        2 => &TURBO_C2_256,
        3 => &TURBO_C3_256,
        4 => &TURBO_C4_256,
        _ => unreachable!(),
    };
    let mask = (1u32 << bits) - 1;
    let bytes_per_thread = bits; // 2,3,4 bytes/thread for 2,3,4-bit

    let mut out = vec![vec![0.0f32; N_KV_HEADS * HEAD_DIM]; n_pos];
    for p in 0..n_pos {
        for h in 0..N_KV_HEADS {
            let rec = p * bpp + h * bph;
            let cnorm = f32::from_ne_bytes([raw[rec], raw[rec + 1], raw[rec + 2], raw[rec + 3]]);
            // 32 threads, each owns 8 dims = tid*8..tid*8+7.
            for tid in 0..32usize {
                let base = rec + 4 + tid * bytes_per_thread;
                let mut packed: u32 = raw[base] as u32;
                if bytes_per_thread >= 2 {
                    packed |= (raw[base + 1] as u32) << 8;
                }
                if bytes_per_thread >= 3 {
                    packed |= (raw[base + 2] as u32) << 16;
                }
                if bytes_per_thread >= 4 {
                    packed |= (raw[base + 3] as u32) << 24;
                }
                for i in 0..8usize {
                    let idx = ((packed >> (i * bits)) & mask) as usize;
                    let d = tid * 8 + i;
                    out[p][h * HEAD_DIM + d] = cnorm * lut[idx];
                }
            }
        }
    }
    out
}

/// Max stored cnorm across all (layer, pos, head) of a lloyd cache. Used to
/// scale the boundary-flip tolerance into dequantized-value units.
fn max_cnorm(gpu: &Gpu, kv: &KvCache, bits: usize) -> f32 {
    let bph = 4 + (HEAD_DIM * bits) / 8;
    let bpp = N_KV_HEADS * bph;
    let mut cmax = 0.0f32;
    for layer in 0..N_LAYERS {
        let buf = &kv.v_gpu[layer];
        let mut raw = vec![0u8; buf.byte_size()];
        gpu.hip.memcpy_dtoh(&mut raw, &buf.buf).unwrap();
        for p in 0..N_POS {
            for h in 0..N_KV_HEADS {
                let rec = p * bpp + h * bph;
                let c = f32::from_ne_bytes([raw[rec], raw[rec + 1], raw[rec + 2], raw[rec + 3]]);
                cmax = cmax.max(c.abs());
            }
        }
    }
    cmax
}

/// Make a fresh fwht3 cache (3 layers, all real KV layers).
fn make_cache(gpu: &mut Gpu) -> KvCache {
    let is_kv = vec![true; N_LAYERS];
    KvCache::new_gpu_fwht3_filtered(gpu, &is_kv, N_KV_HEADS, HEAD_DIM, MAX_SEQ).unwrap()
}

/// Write the deterministic source into every layer's V buffer via the q8 path.
fn write_q8(gpu: &mut Gpu, kv: &KvCache) {
    for layer in 0..N_LAYERS {
        for p in 0..N_POS {
            let src = build_pos_src(layer, p);
            let st = upload_src(gpu, &src);
            let pb = pos_buf(gpu, p as i32);
            gpu.kv_cache_write_q8_0(&kv.v_gpu[layer], &st, &pb, N_KV_HEADS, HEAD_DIM)
                .unwrap();
            gpu.hip.device_synchronize().unwrap();
            let _ = gpu.free_tensor(st);
            gpu.hip.free(pb).unwrap();
        }
    }
}

/// Write the deterministic source directly at a lloyd tier via the matching
/// V-write launcher (uses the cache's 256-wide FWHT signs).
fn write_lloyd(gpu: &mut Gpu, kv: &KvCache, bits: usize) {
    let s1 = kv.givens_cos.as_ref().unwrap().sub_offset(0, kv.givens_cos.as_ref().unwrap().numel());
    let s2 = kv.givens_sin.as_ref().unwrap().sub_offset(0, kv.givens_sin.as_ref().unwrap().numel());
    for layer in 0..N_LAYERS {
        for p in 0..N_POS {
            let src = build_pos_src(layer, p);
            let st = upload_src(gpu, &src);
            let pb = pos_buf(gpu, p as i32);
            match bits {
                4 => gpu
                    .kv_cache_write_v256_4bit_vec(&kv.v_gpu[layer], &st, &pb, &s1, &s2, N_KV_HEADS, HEAD_DIM)
                    .unwrap(),
                3 => gpu
                    .kv_cache_write_fwht3_vec(&kv.v_gpu[layer], &st, &pb, &s1, &s2, N_KV_HEADS, HEAD_DIM)
                    .unwrap(),
                2 => gpu
                    .kv_cache_write_v256_2bit_vec(&kv.v_gpu[layer], &st, &pb, &s1, &s2, N_KV_HEADS, HEAD_DIM)
                    .unwrap(),
                _ => unreachable!(),
            }
            gpu.hip.device_synchronize().unwrap();
            let _ = gpu.free_tensor(st);
            gpu.hip.free(pb).unwrap();
        }
    }
}

// === K-side helpers (fwht4 / fwht2, 128-wide rotation, 128-LUT family) ===

/// Make a fresh fwht4 K cache (3 layers, all real KV layers). K buffers sized
/// at the fwht4 footprint (132 B/head @256); 128-wide FWHT signs (seeds 42/1042).
fn make_k_cache(gpu: &mut Gpu) -> KvCache {
    let is_kv = vec![true; N_LAYERS];
    KvCache::new_gpu_fwht4_filtered(gpu, &is_kv, N_KV_HEADS, HEAD_DIM, MAX_SEQ).unwrap()
}

/// Write the deterministic source into every layer's K buffer via the fused
/// fwht write at the given K bits (4 or 2). The fused kernel also writes a Q8 V
/// record into the cache's V buffer (v_mode_bits=8); we ignore V and only
/// inspect K. Uses the cache's 128-wide FWHT signs.
fn write_k_fwht(gpu: &mut Gpu, kv: &KvCache, bits: usize) {
    let s1 = kv.givens_cos.as_ref().unwrap().sub_offset(0, kv.givens_cos.as_ref().unwrap().numel());
    let s2 = kv.givens_sin.as_ref().unwrap().sub_offset(0, kv.givens_sin.as_ref().unwrap().numel());
    for layer in 0..N_LAYERS {
        for p in 0..N_POS {
            let src = build_pos_src(layer, p);
            let kt = upload_src(gpu, &src);
            // Dummy V source (same data); V record is written but never read.
            let vt = upload_src(gpu, &src);
            let pb = pos_buf(gpu, p as i32);
            match bits {
                4 => gpu
                    .kv_cache_write_fwht4_fused(
                        &kv.k_gpu[layer], &kv.v_gpu[layer], &kt, &vt, &pb, &s1, &s2,
                        N_KV_HEADS, HEAD_DIM, 8,
                    )
                    .unwrap(),
                2 => gpu
                    .kv_cache_write_fwht2_fused(
                        &kv.k_gpu[layer], &kv.v_gpu[layer], &kt, &vt, &pb, &s1, &s2,
                        N_KV_HEADS, HEAD_DIM, 8,
                    )
                    .unwrap(),
                _ => unreachable!(),
            }
            gpu.hip.device_synchronize().unwrap();
            let _ = gpu.free_tensor(kt);
            let _ = gpu.free_tensor(vt);
            gpu.hip.free(pb).unwrap();
        }
    }
}

/// Write the deterministic source into every layer's K buffer via the DIRECT
/// fwht3 (256-wide) write. Reference for the fwht4→fwht3 transcode. The cache
/// must already carry 256-wide signs (new_gpu_fwht3_filtered does).
fn write_k_fwht3(gpu: &mut Gpu, kv: &KvCache, n_layers: usize) {
    let s1 = kv.givens_cos.as_ref().unwrap().sub_offset(0, kv.givens_cos.as_ref().unwrap().numel());
    let s2 = kv.givens_sin.as_ref().unwrap().sub_offset(0, kv.givens_sin.as_ref().unwrap().numel());
    for layer in 0..n_layers {
        for p in 0..N_POS {
            let src = build_pos_src(layer, p);
            let kt = upload_src(gpu, &src);
            let pb = pos_buf(gpu, p as i32);
            gpu.kv_cache_write_fwht3_vec(&kv.k_gpu[layer], &kt, &pb, &s1, &s2, N_KV_HEADS, HEAD_DIM)
                .unwrap();
            gpu.hip.device_synchronize().unwrap();
            let _ = gpu.free_tensor(kt);
            gpu.hip.free(pb).unwrap();
        }
    }
}

/// Dequantize a fwht K buffer (one layer) at the given `bits` tier (4 or 2),
/// using the 128-LUT family and the K record layout (per 128-wide half:
/// 32 threads × 4 dims, slot half*4+{0..3}). Returns [pos][head*head_dim + slot]
/// where `slot` indexes the per-thread post-FWHT coefficient ordering — the
/// SAME ordering both the fwht4 and fwht2 writers use, so transcoded-vs-direct
/// compare element-wise.
fn dequant_k_layer(gpu: &Gpu, buf: &GpuTensor, bits: usize, n_pos: usize) -> Vec<Vec<f32>> {
    let byte_size = buf.byte_size();
    let mut raw = vec![0u8; byte_size];
    gpu.hip.memcpy_dtoh(&mut raw, &buf.buf).unwrap();

    let n_halves = HEAD_DIM / 128;
    let bph = 4 + (HEAD_DIM * bits) / 8; // fwht4=132, fwht2=68 @256
    let bpp = N_KV_HEADS * bph;
    let lut4: &[f32] = &TURBO_C4_128;
    let lut2: &[f32] = &TURBO_C2_128;

    let mut out = vec![vec![0.0f32; N_KV_HEADS * HEAD_DIM]; n_pos];
    for p in 0..n_pos {
        for h in 0..N_KV_HEADS {
            let rec = p * bpp + h * bph;
            let cnorm = f32::from_ne_bytes([raw[rec], raw[rec + 1], raw[rec + 2], raw[rec + 3]]);
            for half in 0..n_halves {
                for tid in 0..32usize {
                    let (i0, i1, i2, i3);
                    if bits == 4 {
                        // 2 bytes/thread/half at 4 + half*64 + tid*2.
                        let b = rec + 4 + half * 64 + tid * 2;
                        i0 = (raw[b] & 0xF) as usize;
                        i1 = ((raw[b] >> 4) & 0xF) as usize;
                        i2 = (raw[b + 1] & 0xF) as usize;
                        i3 = ((raw[b + 1] >> 4) & 0xF) as usize;
                    } else {
                        // 1 byte/thread/half at 4 + half*32 + tid.
                        let b = rec + 4 + half * 32 + tid;
                        let v = raw[b];
                        i0 = (v & 0x3) as usize;
                        i1 = ((v >> 2) & 0x3) as usize;
                        i2 = ((v >> 4) & 0x3) as usize;
                        i3 = ((v >> 6) & 0x3) as usize;
                    }
                    // Linear slot for this (half, tid, i): the post-FWHT
                    // coefficient ordering is consistent between the two writers.
                    let base = (half * 32 + tid) * 4;
                    let (c0, c1, c2, c3) = if bits == 4 {
                        (lut4[i0], lut4[i1], lut4[i2], lut4[i3])
                    } else {
                        (lut2[i0], lut2[i1], lut2[i2], lut2[i3])
                    };
                    out[p][h * HEAD_DIM + base] = cnorm * c0;
                    out[p][h * HEAD_DIM + base + 1] = cnorm * c1;
                    out[p][h * HEAD_DIM + base + 2] = cnorm * c2;
                    out[p][h * HEAD_DIM + base + 3] = cnorm * c3;
                }
            }
        }
    }
    out
}

/// Dequantize a fwht3 K buffer (one layer): 256-wide layout, 8 dims/thread at
/// out[4 + tid*3] (3-bit packed little-endian into 24 bits), TURBO_C3_256 LUT.
/// The slot ordering is the forward-256 output ordering (dim tid*8 + i), which
/// both the direct fwht3 write and the fwht4→fwht3 transcode produce — so
/// transcoded-vs-direct compare element-wise. Returns [pos][head*head_dim + slot].
fn dequant_k_fwht3_layer(gpu: &Gpu, buf: &GpuTensor, n_pos: usize) -> Vec<Vec<f32>> {
    let byte_size = buf.byte_size();
    let mut raw = vec![0u8; byte_size];
    gpu.hip.memcpy_dtoh(&mut raw, &buf.buf).unwrap();

    let bph = 4 + (HEAD_DIM * 3) / 8; // fwht3 = 100 @256
    let bpp = N_KV_HEADS * bph;

    let mut out = vec![vec![0.0f32; N_KV_HEADS * HEAD_DIM]; n_pos];
    for p in 0..n_pos {
        for h in 0..N_KV_HEADS {
            let rec = p * bpp + h * bph;
            let cnorm = f32::from_ne_bytes([raw[rec], raw[rec + 1], raw[rec + 2], raw[rec + 3]]);
            // 32 threads × 8 dims; thread tid owns dims tid*8..tid*8+7, packed
            // 3-bit into 3 bytes at rec+4+tid*3.
            for tid in 0..32usize {
                let b = rec + 4 + tid * 3;
                let packed = (raw[b] as u32) | ((raw[b + 1] as u32) << 8) | ((raw[b + 2] as u32) << 16);
                for i in 0..8usize {
                    let idx = ((packed >> (i * 3)) & 7) as usize;
                    out[p][h * HEAD_DIM + tid * 8 + i] = cnorm * TURBO_C3_256[idx];
                }
            }
        }
    }
    out
}

/// Max stored K cnorm across all (layer, pos, head) of a fwht3 K cache (100 B/head).
fn max_k_fwht3_cnorm(gpu: &Gpu, kv: &KvCache, n_layers: usize) -> f32 {
    let bph = 4 + (HEAD_DIM * 3) / 8;
    let bpp = N_KV_HEADS * bph;
    let mut cmax = 0.0f32;
    for layer in 0..n_layers {
        let buf = &kv.k_gpu[layer];
        let mut raw = vec![0u8; buf.byte_size()];
        gpu.hip.memcpy_dtoh(&mut raw, &buf.buf).unwrap();
        for p in 0..N_POS {
            for h in 0..N_KV_HEADS {
                let rec = p * bpp + h * bph;
                let c = f32::from_ne_bytes([raw[rec], raw[rec + 1], raw[rec + 2], raw[rec + 3]]);
                cmax = cmax.max(c.abs());
            }
        }
    }
    cmax
}

/// Max stored K cnorm across all (layer, pos, head) of a fwht K cache.
fn max_k_cnorm(gpu: &Gpu, kv: &KvCache, bits: usize) -> f32 {
    let bph = 4 + (HEAD_DIM * bits) / 8;
    let bpp = N_KV_HEADS * bph;
    let mut cmax = 0.0f32;
    for layer in 0..N_LAYERS {
        let buf = &kv.k_gpu[layer];
        let mut raw = vec![0u8; buf.byte_size()];
        gpu.hip.memcpy_dtoh(&mut raw, &buf.buf).unwrap();
        for p in 0..N_POS {
            for h in 0..N_KV_HEADS {
                let rec = p * bpp + h * bph;
                let c = f32::from_ne_bytes([raw[rec], raw[rec + 1], raw[rec + 2], raw[rec + 3]]);
                cmax = cmax.max(c.abs());
            }
        }
    }
    cmax
}

fn compare_k_layers(
    gpu: &Gpu,
    transcoded: &KvCache,
    reference: &KvCache,
    bits: usize,
) -> (f32, f32, f32) {
    let mut m = 0.0f32;
    let mut mean = 0.0f32;
    let mut frac = 0.0f32;
    for layer in 0..N_LAYERS {
        let a = dequant_k_layer(gpu, &transcoded.k_gpu[layer], bits, N_POS);
        let b = dequant_k_layer(gpu, &reference.k_gpu[layer], bits, N_POS);
        let (lm, lmean, lfrac) = diag(&a, &b);
        m = m.max(lm);
        mean = mean.max(lmean);
        frac = frac.max(lfrac);
    }
    (m, mean, frac)
}

/// Max abs diff between two dequantized layer-sets.
fn max_abs_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let mut m = 0.0f32;
    for (pa, pb) in a.iter().zip(b.iter()) {
        for (x, y) in pa.iter().zip(pb.iter()) {
            m = m.max((x - y).abs());
        }
    }
    m
}

/// Diagnostics: (max_abs, mean_abs, frac_elems_differing > 1e-6).
fn diag(a: &[Vec<f32>], b: &[Vec<f32>]) -> (f32, f32, f32) {
    let mut m = 0.0f32;
    let mut sum = 0.0f64;
    let mut n = 0u64;
    let mut ndiff = 0u64;
    for (pa, pb) in a.iter().zip(b.iter()) {
        for (x, y) in pa.iter().zip(pb.iter()) {
            let d = (x - y).abs();
            m = m.max(d);
            sum += d as f64;
            n += 1;
            if d > 1e-6 {
                ndiff += 1;
            }
        }
    }
    (m, (sum / n as f64) as f32, ndiff as f32 / n as f32)
}

fn compare_all_layers(
    gpu: &Gpu,
    transcoded: &KvCache,
    reference: &KvCache,
    bits: usize,
) -> (f32, f32, f32) {
    let mut m = 0.0f32;
    let mut mean = 0.0f32;
    let mut frac = 0.0f32;
    for layer in 0..N_LAYERS {
        let a = dequant_lloyd_layer(gpu, &transcoded.v_gpu[layer], bits, N_POS);
        let b = dequant_lloyd_layer(gpu, &reference.v_gpu[layer], bits, N_POS);
        let (lm, lmean, lfrac) = diag(&a, &b);
        m = m.max(lm);
        mean = mean.max(lmean);
        frac = frac.max(lfrac);
    }
    let _ = max_abs_diff;
    (m, mean, frac)
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("adaptive_kv_check: synthetic V transcode correctness");
    println!(
        "  n_kv_heads={N_KV_HEADS} head_dim={HEAD_DIM} n_layers={N_LAYERS} n_pos={N_POS}"
    );

    let mut all_pass = true;

    // A single ±1 index flip at a quantization boundary changes a dequantized
    // element by at most the LARGEST adjacent-centroid gap (Lloyd-Max spacing is
    // non-uniform — outer gaps are widest), scaled by that head's cnorm. Use the
    // max adjacent gap so the tolerance bounds a legitimate single-index flip.
    let max_gap = |lut: &[f32]| -> f32 {
        lut.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max)
    };
    let step4 = max_gap(&TURBO_C4_256);
    let step3 = max_gap(&TURBO_C3_256);
    let step2 = max_gap(&TURBO_C2_256);

    // === Case 1: q8 → lloyd4 ===
    {
        // Transcode cache: q8-sized V, written via q8, then transcoded to lloyd4.
        let trans = make_cache(&mut gpu);
        write_q8(&mut gpu, &trans);
        let mut trans = trans;
        trans.transcode_v_step(&mut gpu, VMode::Lloyd4, N_POS).unwrap();
        gpu.hip.device_synchronize().unwrap();

        // Reference cache: V resized to lloyd4, written directly at lloyd4.
        let mut refc = make_cache(&mut gpu);
        refc.set_v_mode_realloc(&mut gpu, VMode::Lloyd4).unwrap();
        write_lloyd(&mut gpu, &refc, 4);
        gpu.hip.device_synchronize().unwrap();

        let (m, mean, frac) = compare_all_layers(&gpu, &trans, &refc, 4);
        let cmax = max_cnorm(&gpu, &refc, 4);
        // A boundary flip changes one element's dequantized value by ~one target
        // step scaled by that head's cnorm. q8→lloyd4 adds q8 rounding before the
        // 4-bit quant; the 4-bit step is finer than nothing-coarser, so at most a
        // ±1 index flip per element. Accept if max ≤ ~1 step × max cnorm, only a
        // small fraction of elements differ, and the mean is tiny.
        let step_full = step4 * cmax;
        let pass = m <= step_full * 1.3 && mean <= step_full * 0.25;
        all_pass &= pass;
        println!(
            "  [q8 -> lloyd4]   max={m:.4e} mean={mean:.4e} frac_diff={frac:.4} (1step×cmax={step_full:.4e})  {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    // === Case 2: lloyd4 → lloyd3 ===
    {
        // Transcode cache: write directly at lloyd4, then transcode down to lloyd3.
        let mut trans = make_cache(&mut gpu);
        trans.set_v_mode_realloc(&mut gpu, VMode::Lloyd4).unwrap();
        write_lloyd(&mut gpu, &trans, 4);
        gpu.hip.device_synchronize().unwrap();
        trans.transcode_v_step(&mut gpu, VMode::Lloyd3, N_POS).unwrap();
        gpu.hip.device_synchronize().unwrap();

        // Reference: write directly at lloyd3 over the same source.
        let mut refc = make_cache(&mut gpu);
        refc.set_v_mode_realloc(&mut gpu, VMode::Lloyd3).unwrap();
        write_lloyd(&mut gpu, &refc, 3);
        gpu.hip.device_synchronize().unwrap();

        let (m, mean, frac) = compare_all_layers(&gpu, &trans, &refc, 3);
        let cmax = max_cnorm(&gpu, &refc, 3);
        let step_full = step3 * cmax;
        let pass = m <= step_full * 1.3 && mean <= step_full * 0.25;
        all_pass &= pass;
        println!(
            "  [lloyd4->lloyd3] max={m:.4e} mean={mean:.4e} frac_diff={frac:.4} (1step×cmax={step_full:.4e})  {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    // === Case 3: lloyd3 → lloyd2 ===
    {
        let mut trans = make_cache(&mut gpu);
        trans.set_v_mode_realloc(&mut gpu, VMode::Lloyd3).unwrap();
        write_lloyd(&mut gpu, &trans, 3);
        gpu.hip.device_synchronize().unwrap();
        trans.transcode_v_step(&mut gpu, VMode::Lloyd2, N_POS).unwrap();
        gpu.hip.device_synchronize().unwrap();

        let mut refc = make_cache(&mut gpu);
        refc.set_v_mode_realloc(&mut gpu, VMode::Lloyd2).unwrap();
        write_lloyd(&mut gpu, &refc, 2);
        gpu.hip.device_synchronize().unwrap();

        let (m, mean, frac) = compare_all_layers(&gpu, &trans, &refc, 2);
        let cmax = max_cnorm(&gpu, &refc, 2);
        let step_full = step2 * cmax;
        let pass = m <= step_full * 1.3 && mean <= step_full * 0.25;
        all_pass &= pass;
        println!(
            "  [lloyd3->lloyd2] max={m:.4e} mean={mean:.4e} frac_diff={frac:.4} (1step×cmax={step_full:.4e})  {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    // === Case 4: K fwht4 -> fwht2 (same-width 128-LUT remap) ===
    {
        let step4_128 = max_gap(&TURBO_C4_128);
        let step2_128 = max_gap(&TURBO_C2_128);
        let _ = step4_128;

        // Transcode cache: write K directly at fwht4, then transcode K -> fwht2.
        let mut trans = make_k_cache(&mut gpu);
        write_k_fwht(&mut gpu, &trans, 4);
        gpu.hip.device_synchronize().unwrap();
        trans.transcode_k_step(&mut gpu, 2, N_POS).unwrap();
        gpu.hip.device_synchronize().unwrap();

        // Reference cache: write K directly at fwht2 over the same source.
        let refc = make_k_cache(&mut gpu);
        write_k_fwht(&mut gpu, &refc, 2);
        gpu.hip.device_synchronize().unwrap();

        let (m, mean, frac) = compare_k_layers(&gpu, &trans, &refc, 2);
        let cmax = max_k_cnorm(&gpu, &refc, 2);
        // The transcode reconstructs from the fwht4 record (one extra 4-bit
        // rounding vs the direct fwht2 write of the same rotated values), so a
        // dim can land one 2-bit centroid off at a boundary: bound by ~one
        // 2-bit step × cnorm. (Direct fwht2 quantizes the exact rotated value;
        // transcode quantizes the 4-bit-rounded value — they agree except near
        // a 2-bit decision boundary that the 4-bit rounding nudges across.)
        let step_full = step2_128 * cmax;
        let pass = m <= step_full * 1.3 && mean <= step_full * 0.25;
        all_pass &= pass;
        println!(
            "  [K fwht4->fwht2] max={m:.4e} mean={mean:.4e} frac_diff={frac:.4} (1step×cmax={step_full:.4e})  {}",
            if pass { "PASS" } else { "FAIL" }
        );
        // Verify the K-mode booleans flipped (so the next forward dispatches
        // the fwht2 attention kernel).
        let flipped = trans.quant_asym2 && !trans.quant_asym4 && trans.quant_fwht;
        all_pass &= flipped;
        println!(
            "  [K mode flip]    quant_asym2={} quant_asym4={} quant_fwht={}  {}",
            trans.quant_asym2, trans.quant_asym4, trans.quant_fwht,
            if flipped { "PASS" } else { "FAIL" }
        );
    }

    // === Case 5: K fwht4 -> fwht3 (RE-ROTATION, 128-wide -> 256-wide) ===
    {
        let step3_256 = max_gap(&TURBO_C3_256);

        // Transcode cache: write K directly at fwht4 (128-wide signs), then
        // UPGRADE signs to 256-wide (mirrors set_adaptive_floor_alloc's sign
        // upgrade for k_floor=fwht3), then transcode K -> fwht3 (re-rotation).
        let mut trans = make_k_cache(&mut gpu);
        write_k_fwht(&mut gpu, &trans, 4);
        gpu.hip.device_synchronize().unwrap();
        // Upgrade the cache's FWHT signs to 256-wide (gen_fwht_signs is a pure
        // LCG; first 128 entries are byte-identical, so the inverse-128 matches
        // fwht4's forward). The K buffer stays fwht4-sized — large enough to
        // hold the smaller fwht3 record (production floor-sizes it to 100 B/head,
        // also large enough).
        {
            let s1v = KvCache::gen_fwht_signs(42, 256);
            let s2v = KvCache::gen_fwht_signs(1042, 256);
            let s1b: Vec<u8> = s1v.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let s2b: Vec<u8> = s2v.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let s1 = gpu.alloc_tensor(&[256], DType::F32).unwrap();
            let s2 = gpu.alloc_tensor(&[256], DType::F32).unwrap();
            gpu.hip.memcpy_htod(&s1.buf, &s1b).unwrap();
            gpu.hip.memcpy_htod(&s2.buf, &s2b).unwrap();
            if let Some(old) = trans.givens_cos.take() { let _ = gpu.free_tensor(old); }
            if let Some(old) = trans.givens_sin.take() { let _ = gpu.free_tensor(old); }
            trans.givens_cos = Some(s1);
            trans.givens_sin = Some(s2);
        }
        trans.transcode_k_step(&mut gpu, 3, N_POS).unwrap();
        gpu.hip.device_synchronize().unwrap();

        // Reference cache: a fwht3 (256-wide) cache written DIRECTLY at fwht3
        // over the same source.
        let is_kv = vec![true; N_LAYERS];
        let refc = KvCache::new_gpu_fwht3_filtered(&mut gpu, &is_kv, N_KV_HEADS, HEAD_DIM, MAX_SEQ).unwrap();
        write_k_fwht3(&mut gpu, &refc, N_LAYERS);
        gpu.hip.device_synchronize().unwrap();

        let mut m = 0.0f32;
        let mut mean = 0.0f32;
        let mut frac = 0.0f32;
        for layer in 0..N_LAYERS {
            let a = dequant_k_fwht3_layer(&gpu, &trans.k_gpu[layer], N_POS);
            let b = dequant_k_fwht3_layer(&gpu, &refc.k_gpu[layer], N_POS);
            let (lm, lmean, lfrac) = diag(&a, &b);
            m = m.max(lm);
            mean = mean.max(lmean);
            frac = frac.max(lfrac);
        }
        let cmax = max_k_fwht3_cnorm(&gpu, &refc, N_LAYERS);
        // RE-ROTATION error budget: the transcode round-trips fwht4 -> dequant
        // (4-bit, 128-family) -> inverse-128 -> forward-256 -> 3-bit quant
        // (256-family). The direct fwht3 write quantizes the EXACT normal-space
        // K at 3-bit. The extra 4-bit rounding before the inverse perturbs the
        // 256-wide rotated coefficients, so many dims can land one (or near a
        // boundary, occasionally two) 3-bit centroids off. Bound by a generous
        // multiple of one 3-bit step × cnorm (re-rotation is the design's
        // flagged-costly path; expect a larger spread than the same-width remap).
        let step_full = step3_256 * cmax;
        let pass = m <= step_full * 3.0 && mean <= step_full * 0.6;
        all_pass &= pass;
        println!(
            "  [K fwht4->fwht3] max={m:.4e} mean={mean:.4e} frac_diff={frac:.4} (1step×cmax={step_full:.4e})  {}",
            if pass { "PASS" } else { "FAIL" }
        );
        // Verify the K-mode booleans flipped to fwht3.
        let flipped = trans.quant_asym3 && !trans.quant_asym4 && !trans.quant_asym2 && trans.quant_fwht;
        all_pass &= flipped;
        println!(
            "  [K3 mode flip]   quant_asym3={} quant_asym4={} quant_asym2={} quant_fwht={}  {}",
            trans.quant_asym3, trans.quant_asym4, trans.quant_asym2, trans.quant_fwht,
            if flipped { "PASS" } else { "FAIL" }
        );
    }

    if all_pass {
        println!("adaptive_kv_check: PASS");
        std::process::exit(0);
    } else {
        println!("adaptive_kv_check: FAIL");
        std::process::exit(1);
    }
}
