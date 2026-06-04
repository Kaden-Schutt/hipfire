// Copyright (c) 2026 Kaden Schutt
// SPDX-License-Identifier: see LICENSE
//
//! fakequant_superblock — F6 super-block-linear quant *weight-codec* evaluator.
//!
//! Reads the all-F32 oracle .hfq, round-trips each 4-bit-"Base" weight tensor
//! through a chosen super-block-linear codec, and writes a new all-F32 .hfq
//! whose ONLY injected error is the weight-codec round-trip. The KLD of this
//! fake-quant candidate vs the faithful fp32-DN oracle (via the existing
//! eval_hipfire_fullvocab) is the isolated weight-codec quality of the codec.
//!
//! This is eval-only: it touches no forward/kernel/dispatch code. The candidate
//! runs through the proven all-F32 forward path. The reported `--report-bpw` is
//! the EXACT effective bits/weight of the codec's on-disk layout (256-elem
//! super-block, hierarchical fp16 d [+ fp16 dmin], per-sub-block 6-bit scale
//! [+ 6-bit min], 4-bit nibbles) — NOT the F32 storage of this fake-quant file.
//!
//! Tensor-class protection mirrors the production dense kmap (mode 0, is_moe=false):
//!   norms/bias -> kept F32 (lossless; would be F16 in prod, ~0 error)
//!   embed_tokens / lm_head -> kept F32 (Q8 in prod; held lossless here)
//!   DeltaNet conv1d -> kept F32 (Q8 in prod)
//!   edge layers (first2/last2) FFN -> kept F32 (6-bit in prod)
//!   everything else (Base 4-bit: attn q/k/v/o + middle MLP gate/up/down) ->
//!     round-tripped through the chosen super-block codec.
//! The `flat-g256` control codec re-derives the production-flat anchor under
//! THIS protection, calibrating the (small) lossless-protected-class offset so
//! the cross-codec deltas are trustworthy.
//!
//! Codecs (--codec):
//!   flat-g256          : flat HFQ4-G256 asym (scale+min, fp32 hdr), 136 B/256 = 4.25 bpw  [CONTROL]
//!   mq4-flat-g256      : flat-g256 + offline FWHT (seeds 42/1042)                          [CONTROL]
//!   sb-asym-g32        : 1a unrotated asym super-block, g32  -> 144 B/256 = 4.500 bpw
//!   sb-asym-g64        : 1a unrotated asym super-block, g64  -> 138 B/256 = 4.3125 bpw
//!   sb-fwht-sym-g32    : 1b FWHT + symmetric super-block, g32 -> 136 B/256 = 4.250 bpw
//!   sb-fwht-sym-g64    : 1b FWHT + symmetric super-block, g64 -> 133 B/256 = 4.1563 bpw

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

// ── f16 <-> f32 (bit-exact mirror of hipfire-quantize/src/main.rs) ──────────
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 { return f32::from_bits(sign << 31); }
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 { f <<= 1; e -= 1; }
        f &= 0x3FF;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13));
    }
    if exp == 31 {
        let frac32 = if frac == 0 { 0 } else { frac << 13 | 1 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | frac32);
    }
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
}
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0xFF {
        let f16_frac = if frac == 0 { 0 } else { (frac >> 13) | 1 };
        return ((sign << 15) | (0x1F << 10) | f16_frac) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 { return ((sign << 15) | (0x1F << 10)) as u16; }
    if new_exp <= 0 {
        if new_exp < -10 { return (sign << 15) as u16; }
        let f = frac | 0x800000;
        let shift = (1 - new_exp + 13) as u32;
        return ((sign << 15) | (f >> shift)) as u16;
    }
    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}
#[inline] fn f16rt(x: f32) -> f32 { f16_to_f32(f32_to_f16(x)) }

// ── FWHT (bit-exact mirror of cpu_fwht_256 + gen_fwht_signs) ────────────────
fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
        if (state >> 16) & 1 == 1 { 1.0f32 } else { -1.0f32 }
    }).collect()
}
fn fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert_eq!(x.len(), 256);
    for i in 0..256 { x[i] *= signs1[i]; }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 0.0625;
    for i in 0..256 { x[i] *= scale * signs2[i]; }
}
// Inverse FWHT: FWHT is its own inverse up to the orthonormal 1/16 scale. The
// forward applies signs1 -> H -> (1/16)*signs2. The exact inverse therefore is
// signs2 -> H -> (1/16)*signs1 (the unscaled H is symmetric & H*H = 256*I, so
// the two 1/16 factors compose with the 256 to give identity).
fn ifwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert_eq!(x.len(), 256);
    for i in 0..256 { x[i] *= signs2[i]; }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 0.0625;
    for i in 0..256 { x[i] *= scale * signs1[i]; }
}

// ── codecs: each takes a 256-element super-block (already padded) and returns
//    the round-tripped (dequantized) 256 F32 values. Mirrors the exact on-disk
//    quant arithmetic of the corresponding format. ───────────────────────────

// Flat HFQ4-G256 asymmetric: one (scale,min) for the whole 256 group, 4-bit.
fn rt_flat_g256(group: &[f32; 256]) -> [f32; 256] {
    let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;
    let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
    let inv = if range > 0.0 { 1.0 / scale } else { 0.0 };
    let mut out = [0.0f32; 256];
    for i in 0..256 {
        let q = (((group[i] - min_val) * inv + 0.5) as u8).min(15);
        out[i] = q as f32 * scale + min_val;
    }
    out
}

/// Per-sub-block FROZEN Q4K grid: the hierarchical-fp16-compressed
/// (eff_scale, eff_min) pair for each of the `256/gsub` sub-blocks of a
/// 256-element super-block. Bit-exact to the on-disk Q4K layout
/// (per-super-block fp16 d/dmin → 6-bit sub-scale/sub-min). `eff_min` is
/// the reconstructed `-min`, so dequant is `q*eff_scale - eff_min`.
fn gen_frozen_sb_grid(group: &[f32; 256], gsub: usize) -> Vec<(f32, f32)> {
    let n_sub = 256 / gsub;
    let mut sub_scale = vec![0.0f32; n_sub];
    let mut sub_min = vec![0.0f32; n_sub];
    for s in 0..n_sub {
        let g = &group[s * gsub..s * gsub + gsub];
        let mn = g.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = g.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let r = mx - mn;
        sub_scale[s] = if r > 0.0 { r / 15.0 } else { 0.0 };
        sub_min[s] = mn;
    }
    // hierarchical fp16 d / dmin (6-bit sub quantization of scale & -min)
    let max_scale = sub_scale.iter().cloned().fold(0.0f32, f32::max);
    let max_min = sub_min.iter().map(|m| -m).fold(0.0f32, f32::max);
    let d = f16rt(if max_scale > 0.0 { max_scale / 63.0 } else { 0.0 });
    let dmin = f16rt(if max_min > 0.0 { max_min / 63.0 } else { 0.0 });
    let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
    let inv_dmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };
    let mut grid = Vec::with_capacity(n_sub);
    for s in 0..n_sub {
        let sc_int = ((sub_scale[s] * inv_d + 0.5).min(63.0)) as u8 as f32;
        let mn_int = (((-sub_min[s]) * inv_dmin + 0.5).min(63.0)) as u8 as f32;
        let eff_scale = d * sc_int;
        let eff_min = dmin * mn_int; // = -reconstructed min
        grid.push((eff_scale, eff_min));
    }
    grid
}

/// Quantize one value to a frozen Q4K sub-block grid `(eff_scale, eff_min)`.
/// `q = round((val + eff_min)/eff_scale)` clamped [0,15]; recon = `q*eff_scale - eff_min`.
#[inline]
fn q4k_quant_element(val: f32, eff_scale: f32, eff_min: f32) -> f32 {
    if eff_scale <= 0.0 {
        return -eff_min;
    }
    let inv_s = 1.0 / eff_scale;
    let q = (((val + eff_min) * inv_s + 0.5).max(0.0).min(15.0)) as u8 as f32;
    q * eff_scale - eff_min
}

// Super-block ASYMMETRIC linear, sub-block size `gsub` (32 or 64). Hierarchical:
// per-sub-block (scale,min) at 6-bit, compressed by per-super-block fp16 d/dmin.
fn rt_sb_asym(group: &[f32; 256], gsub: usize) -> [f32; 256] {
    let grid = gen_frozen_sb_grid(group, gsub);
    let mut out = [0.0f32; 256];
    for s in 0..grid.len() {
        let (eff_scale, eff_min) = grid[s];
        for i in 0..gsub {
            let idx = s * gsub + i;
            out[idx] = q4k_quant_element(group[idx], eff_scale, eff_min);
        }
    }
    out
}

// Super-block FWHT + SYMMETRIC linear, sub-block size `gsub`. The group is
// already FWHT-rotated by the caller. Per-sub-block symmetric scale (amax/7),
// 4-bit signed [-8,7], compressed by per-super-block fp16 d. NO min.
fn rt_sb_fwht_sym(group: &[f32; 256], gsub: usize) -> [f32; 256] {
    let n_sub = 256 / gsub;
    let mut sub_scale = vec![0.0f32; n_sub];
    for s in 0..n_sub {
        let g = &group[s * gsub..s * gsub + gsub];
        let amax = g.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        sub_scale[s] = if amax > 0.0 { amax / 7.0 } else { 0.0 };
    }
    let max_scale = sub_scale.iter().cloned().fold(0.0f32, f32::max);
    let d = f16rt(if max_scale > 0.0 { max_scale / 63.0 } else { 0.0 });
    let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

    let mut out = [0.0f32; 256];
    for s in 0..n_sub {
        let sc_int = ((sub_scale[s] * inv_d + 0.5).min(63.0)) as u8 as f32;
        let eff_scale = d * sc_int;
        let inv_s = if eff_scale > 0.0 { 1.0 / eff_scale } else { 0.0 };
        for i in 0..gsub {
            let idx = s * gsub + i;
            let q = (group[idx] * inv_s).round().max(-8.0).min(7.0);
            out[idx] = q * eff_scale;
        }
    }
    out
}

struct T { name: String, qt: u8, shape: Vec<u32>, gs: u32, dlen: u64, doff: u64 }
impl AsTensor for T {
    fn name_b(&self) -> &[u8] { self.name.as_bytes() }
    fn shape_v(&self) -> &[u32] { &self.shape }
    fn gs_v(&self) -> u32 { self.gs }
}

#[derive(Clone, Copy, PartialEq)]
enum Codec { FlatG256, Mq4FlatG256, SbAsymG32, SbAsymG64, SbFwhtSymG32, SbFwhtSymG64 }

impl Codec {
    fn rotated(&self) -> bool {
        matches!(self, Codec::Mq4FlatG256 | Codec::SbFwhtSymG32 | Codec::SbFwhtSymG64)
    }
    fn bpw(&self) -> f64 {
        // bytes-per-256-element-super-block / 256 * 8
        let bytes_per_256: f64 = match self {
            Codec::FlatG256 | Codec::Mq4FlatG256 => 136.0, // 8 (f32 scale+min) + 128 nibbles
            Codec::SbAsymG32 => 144.0,   // 4 (f16 d+dmin) + 12 (8x 6b sc + 8x 6b min) + 128
            Codec::SbAsymG64 => 138.0,   // 4 + 6 (4x 6b sc + 4x 6b min) + 128
            Codec::SbFwhtSymG32 => 136.0,// 2 (f16 d) + 6 (8x 6b sc) + 128
            Codec::SbFwhtSymG64 => 133.0,// 2 + 3 (4x 6b sc) + 128
        };
        bytes_per_256 / 256.0 * 8.0
    }
}

// round-trip one full tensor (length = rows*cols, row-major) through codec.
fn roundtrip_tensor(data: &mut [f32], codec: Codec, signs1: &[f32], signs2: &[f32]) {
    let n = data.len();
    let mut b = 0;
    while b < n {
        let end = (b + 256).min(n);
        let mut group = [0.0f32; 256];
        let len = end - b;
        group[..len].copy_from_slice(&data[b..end]);
        let out = match codec {
            Codec::FlatG256 => rt_flat_g256(&group),
            Codec::Mq4FlatG256 => {
                let mut g = group; fwht_256(&mut g, signs1, signs2);
                let q = rt_flat_g256(&g);
                let mut r = q; ifwht_256(&mut r, signs1, signs2); r
            }
            Codec::SbAsymG32 => rt_sb_asym(&group, 32),
            Codec::SbAsymG64 => rt_sb_asym(&group, 64),
            Codec::SbFwhtSymG32 => {
                let mut g = group; fwht_256(&mut g, signs1, signs2);
                let q = rt_sb_fwht_sym(&g, 32);
                let mut r = q; ifwht_256(&mut r, signs1, signs2); r
            }
            Codec::SbFwhtSymG64 => {
                let mut g = group; fwht_256(&mut g, signs1, signs2);
                let q = rt_sb_fwht_sym(&g, 64);
                let mut r = q; ifwht_256(&mut r, signs1, signs2); r
            }
        };
        data[b..end].copy_from_slice(&out[..len]);
        b += 256;
    }
}

// ── production-dense kmap protection (mode 0, is_moe=false) ─────────────────
fn parse_layer_idx(name: &str) -> Option<usize> {
    // matches "...layers.<N>..." style names
    let parts: Vec<&str> = name.split('.').collect();
    for w in parts.windows(2) {
        if w[0] == "layers" { return w[1].parse().ok(); }
    }
    None
}
/// True if this tensor is a Base 4-bit weight (gets the codec). False = protected.
/// `v3_scope`=false: my conservative dense kmap (lm_head F32, edge-FFN F32, conv1d F32).
/// `v3_scope`=true: match the AWQ-GPTQ-v3 artifact's ACTUAL scope (only embed_tokens +
///   conv1d kept Q8/lossless; lm_head + DeltaNet in_proj + all MLP at 4-bit; no edge
///   promotion). This makes the comparison to v3's 0.073771 apples-to-apples.
fn is_base_4bit(name: &str, n_layers: usize, v3_scope: bool) -> bool {
    if name.contains("norm") || name.contains("bias") { return false; }
    if !name.contains("weight") { return false; }
    // 1D DeltaNet params (A_log, dt_bias) are not ".weight" so already excluded.
    if v3_scope {
        // v3: Q8 only embed_tokens + conv1d; everything else (incl lm_head) is 4-bit.
        if name.contains("embed_tokens") || name.contains("token_embd") { return false; }
        if name.contains("conv1d") || name.contains("conv_1d") { return false; }
        return true;
    }
    if name.contains("embed_tokens") || name.contains("token_embd")
        || name.contains("lm_head") || name.ends_with("output.weight") { return false; }
    if name.contains("conv1d") || name.contains("conv_1d") { return false; }
    if let Some(idx) = parse_layer_idx(name) {
        if idx < 2 || idx >= n_layers.saturating_sub(2) {
            if name.contains("mlp.") || name.contains("ffn") { return false; }
        }
    }
    true
}

/// Read AWQ per-column scales (sidecar `<name>.awq_scale.weight`, 1D F16) from an
/// HFQM file into a name->Vec<f32> map keyed by the BASE weight name.
fn read_awq_scales(path: &Path) -> std::collections::HashMap<String, Vec<f32>> {
    use std::io::Seek;
    let mut f = File::open(path).expect("open awq src");
    let mut hdr = [0u8; 32];
    f.read_exact(&mut hdr).unwrap();
    assert_eq!(&hdr[0..4], b"HFQM");
    let md_off = u64::from_le_bytes(hdr[16..24].try_into().unwrap());
    let data_off = u64::from_le_bytes(hdr[24..32].try_into().unwrap());
    let mut all = Vec::new();
    f.seek(std::io::SeekFrom::Start(0)).unwrap();
    f.read_to_end(&mut all).unwrap();
    let md_start = md_off as usize;
    // brace-match json
    let mut depth = 0i32; let mut in_str = false; let mut esc = false; let mut end = 0usize;
    for (k, &c) in all[md_start..].iter().enumerate() {
        if in_str { if esc { esc = false; } else if c == b'\\' { esc = true; } else if c == b'"' { in_str = false; } }
        else { match c { b'"' => in_str = true, b'{' => depth += 1, b'}' => { depth -= 1; if depth == 0 { end = k + 1; break; } }, _ => {} } }
    }
    let mut p = md_start + end;
    let cnt = u32::from_le_bytes(all[p..p+4].try_into().unwrap()) as usize; p += 4;
    let mut running = data_off;
    let mut map = std::collections::HashMap::new();
    for _ in 0..cnt {
        let nl = u16::from_le_bytes(all[p..p+2].try_into().unwrap()) as usize; p += 2;
        let name = String::from_utf8(all[p..p+nl].to_vec()).unwrap(); p += nl;
        let qt = all[p]; p += 1;
        let nd = all[p] as usize; p += 1;
        let mut shape = Vec::with_capacity(nd);
        for _ in 0..nd { shape.push(u32::from_le_bytes(all[p..p+4].try_into().unwrap())); p += 4; }
        let _gs = u32::from_le_bytes(all[p..p+4].try_into().unwrap()); p += 4;
        let dlen = u64::from_le_bytes(all[p..p+8].try_into().unwrap()); p += 8;
        let doff = running; running += dlen;
        if let Some(stem) = name.strip_suffix(".awq_scale.weight") {
            // F16 1D
            assert_eq!(qt, 1, "awq scale {name} not F16");
            let raw = &all[doff as usize .. (doff + dlen) as usize];
            let vals: Vec<f32> = raw.chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect();
            map.insert(format!("{stem}.weight"), vals);
        }
        let _ = shape;
    }
    map
}

/// hipfire AWQ scale formula (mirror of `compute_awq_scales` in
/// hipfire-quantize/src/main.rs): `s_c = (E[x_c^2])^(alpha/2)`, normalized
/// so the geometric mean of `s` is 1.0 (done in log space for stability).
fn awq_scales_from_imatrix(in_sum2: &[f64], alpha: f64) -> Vec<f32> {
    let k = in_sum2.len();
    let half_alpha = alpha * 0.5;
    let mut log_s = Vec::with_capacity(k);
    let mut sum_log = 0.0f64;
    for &v in in_sum2 {
        let vc = v.max(1e-12);
        let l = half_alpha * vc.ln();
        log_s.push(l);
        sum_log += l;
    }
    let mean_log = sum_log / k as f64;
    log_s.into_iter().map(|l| ((l - mean_log).exp()) as f32).collect()
}

/// Per-channel E[x_c^2] (imatrix diagonal) for a tensor, extracted from its
/// UN-rotated per-256-block Hessian: channel c lives in block c/256 at
/// diagonal index c%256. Length = n_groups*256 = K.
fn imatrix_from_unrot_hessian(h_blocks: &[Vec<f64>]) -> Vec<f64> {
    let mut diag = Vec::with_capacity(h_blocks.len() * 256);
    for blk in h_blocks {
        for c in 0..256 {
            diag.push(blk[c * 256 + c]);
        }
    }
    diag
}

/// Read the UN-rotated per-256-block Hessian sidecar (HUNR v1, produced by
/// `npz_to_unrot_hessian.py`). Format:
///   magic "HUNR"(4) | version u32=1 | n_tensors u32
///   per tensor: name_len u16 | name utf8 | n_groups u32 | (n_groups*256*256) f64 LE
/// Returns name -> Vec<[f64;256*256]> (one 256×256 block per group).
fn read_unrot_hessians(path: &Path) -> std::collections::HashMap<String, Vec<Vec<f64>>> {
    let mut f = File::open(path).expect("open hessian");
    let mut all = Vec::new();
    f.read_to_end(&mut all).expect("read hessian");
    assert_eq!(&all[0..4], b"HUNR", "bad hessian magic");
    let mut p = 8usize;
    let n_tensors = u32::from_le_bytes(all[p..p + 4].try_into().unwrap()) as usize;
    p += 4;
    let mut map = std::collections::HashMap::new();
    for _ in 0..n_tensors {
        let nl = u16::from_le_bytes(all[p..p + 2].try_into().unwrap()) as usize;
        p += 2;
        let name = String::from_utf8(all[p..p + nl].to_vec()).unwrap();
        p += nl;
        let n_groups = u32::from_le_bytes(all[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        let mut blocks = Vec::with_capacity(n_groups);
        for _ in 0..n_groups {
            let mut blk = vec![0.0f64; 256 * 256];
            for v in blk.iter_mut() {
                *v = f64::from_le_bytes(all[p..p + 8].try_into().unwrap());
                p += 8;
            }
            blocks.push(blk);
        }
        map.insert(name, blocks);
    }
    map
}

/// Damped inverse of a symmetric 256×256 Hessian block via Cholesky.
/// Mirrors `damped_inverse_hessian` (scripts/mq4_masked_calib.py): add
/// `damp*mult*mean(diag) + eps` to the diagonal, escalating mult on failure.
/// Returns the inverse as a row-major Vec<f64> length 256*256.
fn damped_inverse_256(h: &[f64]) -> Vec<f64> {
    const N: usize = 256;
    // symmetrize + mean positive diagonal
    let mut sym = vec![0.0f64; N * N];
    let mut sum_diag = 0.0f64;
    let mut cnt_diag = 0usize;
    for i in 0..N {
        for j in 0..N {
            sym[i * N + j] = 0.5 * (h[i * N + j] + h[j * N + i]);
        }
        let d = h[i * N + i];
        if d.is_finite() && d > 0.0 {
            sum_diag += d;
            cnt_diag += 1;
        }
    }
    let mean_diag = if cnt_diag > 0 { sum_diag / cnt_diag as f64 } else { 1.0 };
    for &mult in &[0.01f64, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0] {
        let lambda = mult * mean_diag + 1.0e-8;
        if let Some(inv) = chol_inverse(&sym, lambda) {
            return inv;
        }
    }
    // last-resort heavy damping
    let lambda = 100.0 * mean_diag + 1.0e-6;
    chol_inverse(&sym, lambda).unwrap_or_else(|| {
        // diagonal-only fallback (uncorrelated): inv = 1/(h_ii + lambda)
        let mut inv = vec![0.0f64; N * N];
        for i in 0..N {
            inv[i * N + i] = 1.0 / (sym[i * N + i] + lambda);
        }
        inv
    })
}

/// Cholesky factor of (H + lambda*I) then invert via L. Returns None if not SPD.
fn chol_inverse(h: &[f64], lambda: f64) -> Option<Vec<f64>> {
    const N: usize = 256;
    // L lower-triangular, L L^T = H + lambda I
    let mut l = vec![0.0f64; N * N];
    for i in 0..N {
        for j in 0..=i {
            let mut s = h[i * N + j];
            if i == j {
                s += lambda;
            }
            for k in 0..j {
                s -= l[i * N + k] * l[j * N + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * N + j] = s.sqrt();
            } else {
                l[i * N + j] = s / l[j * N + j];
            }
        }
    }
    // Invert L (lower-tri) -> Linv
    let mut linv = vec![0.0f64; N * N];
    for i in 0..N {
        linv[i * N + i] = 1.0 / l[i * N + i];
        for j in 0..i {
            let mut s = 0.0f64;
            for k in j..i {
                s += l[i * N + k] * linv[k * N + j];
            }
            linv[i * N + j] = -s * linv[i * N + i];
        }
    }
    // inv = Linv^T Linv  (since H^{-1} = (L L^T)^{-1} = L^{-T} L^{-1})
    let mut inv = vec![0.0f64; N * N];
    for i in 0..N {
        for j in 0..N {
            let mut s = 0.0f64;
            let kmin = i.max(j);
            for k in kmin..N {
                s += linv[k * N + i] * linv[k * N + j];
            }
            inv[i * N + j] = s;
        }
    }
    Some(inv)
}

/// GPTQ-on-Q4K for one AWQ-pre-scaled tensor (row-major `[rows, cols]`,
/// `cols % 256 == 0`). For each 256-super-block group, freezes the per-32
/// Q4K grid (per row), then runs column-sequential GPTQ error feedback
/// (within the group, natural column order) using the AWQ-rescaled
/// un-rotated 256×256 Hessian block's damped inverse. Mutates `vals` in
/// place to the GPTQ+Q4K dequantized weights.
///
/// `h_blocks[g]` is the UN-rotated 256×256 Hessian for column-group g
/// (length `cols/256`). `awq_s` is the AWQ per-column scale (length cols)
/// already applied to `vals` (so we rescale H by `1/(s_a s_b)` to match).
fn gptq_q4k_tensor(
    vals: &mut [f32],
    rows: usize,
    cols: usize,
    h_blocks: &[Vec<f64>],
    awq_s: Option<&[f32]>,
    gsub: usize,
) {
    use rayon::prelude::*;
    const N: usize = 256;
    let n_groups = cols / N;
    assert_eq!(h_blocks.len(), n_groups, "hessian groups vs cols/256 mismatch");

    for g in 0..n_groups {
        // AWQ-rescale the un-rotated Hessian block: H'[a,b] = H[a,b]/(s_a s_b).
        let mut hb = h_blocks[g].clone();
        if let Some(s) = awq_s {
            let base = g * N;
            for a in 0..N {
                let sa = s[base + a] as f64;
                for b in 0..N {
                    let sb = s[base + b] as f64;
                    let d = sa * sb;
                    if d != 0.0 {
                        hb[a * N + b] /= d;
                    }
                }
            }
        }
        let h_inv = damped_inverse_256(&hb);

        // Per-row independent GPTQ on this group's 256 columns.
        vals.par_chunks_mut(cols).for_each(|row| {
            let base = g * N;
            // Snapshot the group's original (AWQ-scaled) weights & freeze grid.
            let mut group_arr = [0.0f32; N];
            group_arr.copy_from_slice(&row[base..base + N]);
            let grid = gen_frozen_sb_grid(&group_arr, gsub); // (eff_scale, eff_min) per sub
            // working residual (f64 for OBS precision)
            let mut work = [0.0f64; N];
            for i in 0..N {
                work[i] = group_arr[i] as f64;
            }
            for i in 0..N {
                let sub = i / gsub;
                let (eff_scale, eff_min) = grid[sub];
                let qv = q4k_quant_element(work[i] as f32, eff_scale, eff_min) as f64;
                row[base + i] = qv as f32; // dequantized output
                let denom = {
                    let d = h_inv[i * N + i];
                    if d.abs() < 1.0e-12 || !d.is_finite() { 1.0 } else { d }
                };
                let err = (work[i] - qv) / denom;
                if err != 0.0 {
                    for j in (i + 1)..N {
                        work[j] -= err * h_inv[i * N + j];
                    }
                }
            }
        });
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut codec_s = String::new();
    let mut n_layers = 32usize;
    let mut report_only = false;
    let mut awq_from: Option<PathBuf> = None;
    let mut hessian_from: Option<PathBuf> = None;
    let mut v3_scope = false;
    // Step-2: derive AWQ scales from the un-rotated Hessian DIAGONAL (E[x^2],
    // i.e. a native imatrix from the same calib forward) for tensors that
    // have a Hessian, instead of the v3-embedded (unsloth-imatrix) scales.
    let mut awq_hessian_diag = false;
    // When set with --awq-hessian-diag: apply the native-imatrix AWQ
    // override ONLY to tensors that ALSO carry a v3 AWQ scale (the
    // 30-tensor overlap). The 37 Hessian-only tensors then get NO AWQ
    // (plain GPTQ), holding coverage at the v3-184 scope so the delta
    // vs the v3-anchor isolates DERIVATION (native vs unsloth) from
    // COVERAGE (184 -> 221). (c) 2026 Kaden Schutt.
    let mut awq_native_restrict_to_v3 = false;
    let mut awq_alpha = 0.5f64;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--input" => { input = Some(PathBuf::from(&argv[i+1])); i += 2; }
            "--output" => { output = Some(PathBuf::from(&argv[i+1])); i += 2; }
            "--codec" => { codec_s = argv[i+1].clone(); i += 2; }
            "--n-layers" => { n_layers = argv[i+1].parse().unwrap(); i += 2; }
            "--report-bpw" => { report_only = true; i += 1; }
            "--awq-from" => { awq_from = Some(PathBuf::from(&argv[i+1])); i += 2; }
            "--hessian-from" => { hessian_from = Some(PathBuf::from(&argv[i+1])); i += 2; }
            "--awq-hessian-diag" => { awq_hessian_diag = true; i += 1; }
            "--awq-native-restrict-to-v3" => { awq_native_restrict_to_v3 = true; i += 1; }
            "--awq-alpha" => { awq_alpha = argv[i+1].parse().unwrap(); i += 2; }
            "--match-v3-scope" => { v3_scope = true; i += 1; }
            o => { eprintln!("unknown arg {o}"); std::process::exit(1); }
        }
    }
    let codec = match codec_s.as_str() {
        "flat-g256" => Codec::FlatG256,
        "mq4-flat-g256" => Codec::Mq4FlatG256,
        "sb-asym-g32" => Codec::SbAsymG32,
        "sb-asym-g64" => Codec::SbAsymG64,
        "sb-fwht-sym-g32" => Codec::SbFwhtSymG32,
        "sb-fwht-sym-g64" => Codec::SbFwhtSymG64,
        o => { eprintln!("unknown codec {o}"); std::process::exit(1); }
    };
    eprintln!("codec={codec_s} EXACT-bpw={:.4}", codec.bpw());
    if report_only { return; }

    let input = input.expect("--input");
    let output = output.expect("--output");

    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);

    // ── read the oracle .hfq header + index ──
    let mut f = File::open(&input).expect("open input");
    let mut hdr = [0u8; 32];
    f.read_exact(&mut hdr).expect("hdr");
    assert_eq!(&hdr[0..4], b"HFQM", "bad magic");
    let arch = u32::from_le_bytes(hdr[8..12].try_into().unwrap());
    let n_tensors = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let metadata_offset = u64::from_le_bytes(hdr[16..24].try_into().unwrap());
    let data_offset = u64::from_le_bytes(hdr[24..32].try_into().unwrap());

    // read whole file (oracle ~36GB; mi300 has plenty of RAM)
    use std::io::Seek;
    f.seek(std::io::SeekFrom::Start(metadata_offset)).unwrap();
    // metadata bytes = from metadata_offset to index start. We don't know index
    // start directly; re-read the full file into a buffer and parse like the engine.
    let mut all = Vec::new();
    f.seek(std::io::SeekFrom::Start(0)).unwrap();
    f.read_to_end(&mut all).expect("read all");

    // parse metadata json: from metadata_offset, length until index. The index
    // count u32 sits right after metadata. We parse the index by walking from a
    // computed metadata length. The engine stores metadata as a json string whose
    // byte length = (index_offset - metadata_offset). Reconstruct index_offset by
    // scanning: metadata is valid UTF-8 JSON; index starts with a u32 tensor count
    // == n_tensors. We find it by trying: index immediately follows metadata, and
    // metadata length is recoverable because write_hfq put index right after the
    // json with no padding. Simplest: the engine's HfqFile parses metadata as the
    // bytes [metadata_offset .. first index byte]; but we can instead just locate
    // the index by knowing data_offset and walking index entries backward is hard.
    // Pragmatic: metadata json length is stored implicitly. Use the fact that the
    // json is a single object — find the matching closing brace from metadata_offset.
    let md_start = metadata_offset as usize;
    let md_json_len = {
        // brace-match scan (json has no unbalanced braces in strings here: arch
        // metadata is simple). Fallback robust: track string state.
        let bytes = &all[md_start..];
        let mut depth = 0i32; let mut in_str = false; let mut esc = false; let mut end = 0usize;
        for (k, &c) in bytes.iter().enumerate() {
            if in_str {
                if esc { esc = false; }
                else if c == b'\\' { esc = true; }
                else if c == b'"' { in_str = false; }
            } else {
                match c { b'"' => in_str = true, b'{' => depth += 1,
                    b'}' => { depth -= 1; if depth == 0 { end = k + 1; break; } }, _ => {} }
            }
        }
        end
    };
    let metadata_json = String::from_utf8(all[md_start..md_start + md_json_len].to_vec()).unwrap();
    let index_off = md_start + md_json_len;

    // ── parse index ──
    let mut p = index_off;
    let idx_count = u32::from_le_bytes(all[p..p+4].try_into().unwrap()) as usize;
    p += 4;
    assert_eq!(idx_count, n_tensors, "index count mismatch");
    let mut tensors: Vec<T> = Vec::with_capacity(n_tensors);
    let mut running = data_offset;
    for _ in 0..n_tensors {
        let nl = u16::from_le_bytes(all[p..p+2].try_into().unwrap()) as usize; p += 2;
        let name = String::from_utf8(all[p..p+nl].to_vec()).unwrap(); p += nl;
        let qt = all[p]; p += 1;
        let nd = all[p] as usize; p += 1;
        let mut shape = Vec::with_capacity(nd);
        for _ in 0..nd { shape.push(u32::from_le_bytes(all[p..p+4].try_into().unwrap())); p += 4; }
        let gs = u32::from_le_bytes(all[p..p+4].try_into().unwrap()); p += 4;
        let dlen = u64::from_le_bytes(all[p..p+8].try_into().unwrap()); p += 8;
        tensors.push(T { name, qt, shape, gs, dlen, doff: running });
        running += dlen;
    }

    // ── optional AWQ scales (Step-2 proxy) ──
    let awq_map = match &awq_from {
        Some(p) => { let m = read_awq_scales(p); eprintln!("loaded {} AWQ scales from {}", m.len(), p.display()); m }
        None => std::collections::HashMap::new(),
    };

    // ── optional un-rotated Hessian (Step-1: GPTQ-on-Q4K) ──
    // Only meaningful for the unrotated asym super-block codecs; the
    // Hessian is in the UNROTATED basis (matching Q4K's no-FWHT GEMV).
    let hess_gsub: Option<usize> = match codec {
        Codec::SbAsymG32 => Some(32),
        Codec::SbAsymG64 => Some(64),
        _ => None,
    };
    let hess_map = match &hessian_from {
        Some(p) => {
            assert!(hess_gsub.is_some(),
                "--hessian-from (GPTQ-on-Q4K) requires an unrotated asym codec (sb-asym-g32|sb-asym-g64), got {codec_s}");
            let m = read_unrot_hessians(p);
            eprintln!("loaded {} un-rotated Hessians from {}", m.len(), p.display());
            m
        }
        None => std::collections::HashMap::new(),
    };

    // ── round-trip Base tensors; collect new data ──
    let mut n_quanted = 0usize; let mut n_protected = 0usize; let mut n_awq = 0usize;
    let mut n_gptq = 0usize;
    let mut new_data: Vec<Vec<u8>> = Vec::with_capacity(n_tensors);
    let mut max_rt_err = 0.0f64; let mut sum_rt_err = 0.0f64; let mut cnt_err = 0u64;
    for t in &tensors {
        assert_eq!(t.qt, 2, "expected F32 oracle tensor, got qt={} for {}", t.qt, t.name);
        let raw = &all[t.doff as usize .. (t.doff + t.dlen) as usize];
        let mut vals: Vec<f32> = raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        if is_base_4bit(&t.name, n_layers, v3_scope) {
            // per-row: codec groups are within-row (K is the last dim). Tensors
            // are row-major [rows, cols]; group by cols so groups never straddle
            // rows (matches the production GEMV layout K%256==0 per row).
            let cols = if t.shape.len() == 2 { t.shape[1] as usize } else { vals.len() };
            let rows = if t.shape.len() == 2 { t.shape[0] as usize } else { 1 };
            // GPTQ-on-Q4K path: a per-256-block un-rotated Hessian for this
            // tensor exists AND the codec is an unrotated asym super-block.
            // The Hessian K must match `cols` and be a multiple of 256.
            let hess: Option<&Vec<Vec<f64>>> = hess_gsub
                .and_then(|_| hess_map.get(&t.name))
                .filter(|blocks| blocks.len() == cols / 256 && cols % 256 == 0);
            // AWQ per-column pre-scale: quantize W[:,j]*s[j], dequant, then /s[j].
            // s has length = cols (K). Skip if length mismatch (e.g. fused QKV
            // whose sidecar K differs — fall back to plain).
            // Step-2 native-imatrix override: when --awq-hessian-diag and this
            // tensor has a Hessian, derive s from the Hessian diagonal (E[x^2]).
            // Restrict-to-v3: only override tensors that ALSO have a v3 AWQ
            // scale (the 30-tensor overlap), so coverage stays at the v3-184
            // scope and the delta vs the v3-anchor is pure DERIVATION.
            let in_v3_awq = awq_map.contains_key(&t.name);
            let awq_native: Option<Vec<f32>> = if awq_hessian_diag
                && (!awq_native_restrict_to_v3 || in_v3_awq) {
                hess.map(|blocks| {
                    let im = imatrix_from_unrot_hessian(blocks);
                    awq_scales_from_imatrix(&im, awq_alpha)
                }).filter(|s| s.len() == cols)
            } else { None };
            let awq: Option<&Vec<f32>> = awq_native.as_ref()
                .or_else(|| awq_map.get(&t.name).filter(|s| s.len() == cols));
            if awq.is_some() { n_awq += 1; }
            let pre_all: Vec<f32> = vals.clone();
            if let (Some(gsub), Some(h_blocks)) = (hess_gsub, hess) {
                // AWQ pre-scale the whole tensor, run GPTQ-on-Q4K, un-scale.
                if let Some(s) = awq {
                    for r in 0..rows {
                        for j in 0..cols { vals[r*cols + j] *= s[j]; }
                    }
                }
                gptq_q4k_tensor(&mut vals, rows, cols, h_blocks, awq.map(|v| v.as_slice()), gsub);
                if let Some(s) = awq {
                    for r in 0..rows {
                        for j in 0..cols { if s[j] != 0.0 { vals[r*cols + j] /= s[j]; } }
                    }
                }
                n_gptq += 1;
            } else {
                for r in 0..rows {
                    let row = &mut vals[r*cols..(r+1)*cols];
                    if let Some(s) = awq { for j in 0..cols { row[j] *= s[j]; } }
                    roundtrip_tensor(row, codec, &signs1, &signs2);
                    if let Some(s) = awq { for j in 0..cols { if s[j] != 0.0 { row[j] /= s[j]; } } }
                }
            }
            for (a, b) in pre_all.iter().zip(vals.iter()) {
                let e = (*a - *b).abs() as f64; sum_rt_err += e; cnt_err += 1;
                if e > max_rt_err { max_rt_err = e; }
            }
            n_quanted += 1;
        } else {
            n_protected += 1;
        }
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        new_data.push(bytes);
    }
    eprintln!("tensors: {} total, {} quanted(base-4bit), {} protected(lossless-F32), {} with AWQ pre-scale ({}), {} with GPTQ-on-Q4K",
        n_tensors, n_quanted, n_protected, n_awq,
        if awq_hessian_diag { format!("native-imatrix alpha={awq_alpha} for Hessian tensors, v3 elsewhere") } else { "v3-embedded".to_string() },
        n_gptq);
    eprintln!("round-trip abs err: mean={:.6e} max={:.6e} over {} weights",
        sum_rt_err / cnt_err.max(1) as f64, max_rt_err, cnt_err);

    // ── write new all-F32 .hfq ──
    write_hfq(&output, arch, &metadata_json, &tensors, &new_data);
    eprintln!("wrote {}", output.display());
}

fn write_hfq(path: &Path, arch: u32, metadata_json: &str, tensors: &[impl AsTensor], data: &[Vec<u8>]) {
    let mut f = File::create(path).expect("create out");
    let md = metadata_json.as_bytes();
    let metadata_offset = 32u64;
    let index_offset = metadata_offset + md.len() as u64;
    let mut index = Vec::new();
    index.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for (i, t) in tensors.iter().enumerate() {
        let nb = t.name_b();
        index.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        index.extend_from_slice(nb);
        index.push(2u8); // F32
        index.push(t.shape_v().len() as u8);
        for &d in t.shape_v() { index.extend_from_slice(&d.to_le_bytes()); }
        index.extend_from_slice(&t.gs_v().to_le_bytes());
        index.extend_from_slice(&(data[i].len() as u64).to_le_bytes());
    }
    let data_start_unaligned = index_offset + index.len() as u64;
    let data_offset = (data_start_unaligned + 4095) & !4095;
    f.write_all(b"HFQM").unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    f.write_all(&arch.to_le_bytes()).unwrap();
    f.write_all(&(tensors.len() as u32).to_le_bytes()).unwrap();
    f.write_all(&metadata_offset.to_le_bytes()).unwrap();
    f.write_all(&data_offset.to_le_bytes()).unwrap();
    f.write_all(md).unwrap();
    f.write_all(&index).unwrap();
    let pad = (data_offset - data_start_unaligned) as usize;
    f.write_all(&vec![0u8; pad]).unwrap();
    for d in data { f.write_all(d).unwrap(); }
    f.flush().unwrap();
}

trait AsTensor { fn name_b(&self) -> &[u8]; fn shape_v(&self) -> &[u32]; fn gs_v(&self) -> u32; }
