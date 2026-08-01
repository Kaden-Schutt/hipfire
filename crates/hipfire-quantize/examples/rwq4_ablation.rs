// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Decompose RWQ4's quality win into its two independent changes.
//!
//! RWQ4G256 differs from MQ4G256 in two ways at the SAME 136 B/group:
//!
//!   (a) a Lloyd-Max Gaussian codebook instead of a uniform grid, and
//!   (b) four E4M3 sub-scales at g64 instead of one affine scale per 256.
//!
//! These have wildly different decode costs on RDNA. (b) is ~free: one extra
//! dword load per group, amortized over 8 weights, then pure VALU. (a) costs
//! eight `ds_load_b32` per group per lane — measured at ~+89% decode time
//! against the dispatched `gemv_hfq4g256` kernel, which is far outside the 1%
//! budget.
//!
//! So the question this answers is: how much of the +2.02 dB is (b) alone?
//! If most of it, a uniform-grid + sub-scale format keeps MQ4's exact FMA inner
//! loop and ships essentially free.
//!
//! SNR is reported in the FWHT-rotated domain, which equals the weight domain
//! because the transform is orthonormal.
//!
//! ```text
//! cargo run --release -p hipfire-quantize --example rwq4_ablation -- <model.safetensors> [max_tensors]
//! ```

use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32};
use hipfire_quantize::fp8::{e4m3_decode, e4m3_encode_roundup};
use hipfire_quantize::fwht::{fwht_256, gen_signs};
use hipfire_quantize::mqn::{self, Spec};
use hipfire_quantize::safetensors_file::SafetensorsFile;
use std::path::Path;

const GROUP: usize = 256;
const SUB: usize = 64;

/// Symmetric uniform 16-level grid, normalized so `max|c| == 1`. Decodes as
/// `(nibble - 7.5) * (eff / 7.5)` — pure VALU, no table lookup.
const UNIFORM_4BIT: [f32; 16] = [
    -1.0,
    -13.0 / 15.0,
    -11.0 / 15.0,
    -9.0 / 15.0,
    -7.0 / 15.0,
    -5.0 / 15.0,
    -3.0 / 15.0,
    -1.0 / 15.0,
    1.0 / 15.0,
    3.0 / 15.0,
    5.0 / 15.0,
    7.0 / 15.0,
    9.0 / 15.0,
    11.0 / 15.0,
    13.0 / 15.0,
    1.0,
];

fn to_f32(bytes: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F32" => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "F16" => bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "BF16" => bytes
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        _ => Vec::new(),
    }
}

/// Shipped MQ4G256: min/max affine over the rotated group, 16 uniform levels,
/// ONE scale + zero for all 256 weights.
fn legacy_sq_err(g: &[f32; GROUP]) -> f64 {
    let lo = g.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = g.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = hi - lo;
    let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
    let inv = if range > 0.0 { 1.0 / scale } else { 0.0 };
    g.iter()
        .map(|&v| {
            let q = (((v - lo) * inv + 0.5) as u8).min(15);
            let d = (v - (q as f32 * scale + lo)) as f64;
            d * d
        })
        .sum()
}

/// One scale for all 256 weights (no sub-scales), against an arbitrary
/// symmetric codebook. Isolates change (a) from change (b).
fn single_scale_sq_err(g: &[f32; GROUP], cb: &[f32; 16]) -> f64 {
    let amax = g.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if amax == 0.0 {
        return 0.0;
    }
    g.iter()
        .map(|&v| {
            let t = v / amax;
            let best = cb
                .iter()
                .map(|&c| (t - c).abs())
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0;
            let d = (v - cb[best] * amax) as f64;
            d * d
        })
        .sum()
}

/// Four E4M3 sub-scales at g64 over an arbitrary codebook — the RWQ4 header
/// shape. `cb == UNIFORM_4BIT` gives the free-decode candidate; `cb ==
/// CODEBOOK_4BIT` gives full RWQ4.
fn subscale_sq_err(g: &[f32; GROUP], cb: &[f32; 16]) -> f64 {
    let mut sub_amax = [0.0f32; GROUP / SUB];
    for (i, &v) in g.iter().enumerate() {
        sub_amax[i / SUB] = sub_amax[i / SUB].max(v.abs());
    }
    let master = sub_amax.iter().copied().fold(0.0f32, f32::max);
    if master == 0.0 {
        return 0.0;
    }
    let mut eff = [0.0f32; GROUP / SUB];
    for s in 0..GROUP / SUB {
        eff[s] = master * e4m3_decode(e4m3_encode_roundup(sub_amax[s] / master));
    }
    g.iter()
        .enumerate()
        .map(|(i, &v)| {
            let e = eff[i / SUB];
            if e == 0.0 {
                return (v as f64) * (v as f64);
            }
            let t = v / e;
            let best = cb
                .iter()
                .map(|&c| (t - c).abs())
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0;
            let d = (v - cb[best] * e) as f64;
            d * d
        })
        .sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: rwq4_ablation <model.safetensors> [max_tensors]");
        std::process::exit(2);
    }
    let max_tensors: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);

    let st = SafetensorsFile::open(Path::new(&args[1])).expect("open safetensors");
    let (s1, s2) = (gen_signs(0x9E37_79B9, 256), gen_signs(0x85EB_CA6B, 256));

    let lloyd: [f32; 16] = mqn::CODEBOOK_4BIT;
    let _ = Spec {
        bits: 4,
        group_bytes: 136,
        codebook: &mqn::CODEBOOK_4BIT,
    };

    let mut names: Vec<String> = st.tensor_names().into_iter().map(String::from).collect();
    names.sort();

    // Squared error per variant, accumulated over every sampled group.
    let mut e_legacy = 0.0f64; // uniform grid, 1 affine scale/256   (shipped MQ4)
    let mut e_uni_1 = 0.0f64; // uniform grid, 1 symmetric scale/256
    let mut e_lloyd_1 = 0.0f64; // Lloyd codebook, 1 scale/256        -> isolates (a)
    let mut e_uni_sub = 0.0f64; // uniform grid, 4x E4M3 g64          -> isolates (b)
    let mut e_lloyd_sub = 0.0f64; // Lloyd codebook, 4x E4M3 g64      -> full RWQ4
    let mut energy = 0.0f64;
    let mut n_used = 0usize;

    for name in names {
        if n_used >= max_tensors {
            break;
        }
        let Some((meta, bytes)) = st.tensor_data(&name) else {
            continue;
        };
        if meta.shape.len() != 2 || meta.shape[1] % 256 != 0 {
            continue;
        }
        if name.contains("norm") || name.contains("embed") {
            continue;
        }
        let w = to_f32(bytes, &meta.dtype);
        if w.is_empty() || w.len() < 256 * 8 {
            continue;
        }

        let n_groups = w.len() / 256;
        let stride = (n_groups / 2048).max(1);
        let mut b = 0usize;
        while b < n_groups {
            let mut g = [0.0f32; GROUP];
            g.copy_from_slice(&w[b * 256..b * 256 + 256]);
            fwht_256(&mut g, &s1, &s2);

            e_legacy += legacy_sq_err(&g);
            e_uni_1 += single_scale_sq_err(&g, &UNIFORM_4BIT);
            e_lloyd_1 += single_scale_sq_err(&g, &lloyd);
            e_uni_sub += subscale_sq_err(&g, &UNIFORM_4BIT);
            e_lloyd_sub += subscale_sq_err(&g, &lloyd);
            energy += g.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();

            b += stride;
        }
        n_used += 1;
    }

    let snr = |e: f64| 10.0 * (energy / e).log10();
    let base = snr(e_legacy);

    println!();
    println!("RWQ4 ablation over {n_used} tensors — all variants at 136 B/group (4.25 bpw)");
    println!();
    println!(
        "{:<38} {:>9} {:>9} {:>16}",
        "variant", "SNR dB", "vs MQ4", "decode cost"
    );
    println!("{}", "-".repeat(76));
    println!(
        "{:<38} {:>8.2}  {:>+8.2}  {:>16}",
        "MQ4G256 min/max affine (shipped)", base, 0.0, "baseline"
    );
    println!(
        "{:<38} {:>8.2}  {:>+8.2}  {:>16}",
        "uniform grid, 1 symmetric scale/256",
        snr(e_uni_1),
        snr(e_uni_1) - base,
        "= baseline"
    );
    println!(
        "{:<38} {:>8.2}  {:>+8.2}  {:>16}",
        "(a) Lloyd codebook, 1 scale/256",
        snr(e_lloyd_1),
        snr(e_lloyd_1) - base,
        "8x ds_load"
    );
    println!(
        "{:<38} {:>8.2}  {:>+8.2}  {:>16}",
        "(b) uniform grid, 4x E4M3 g64",
        snr(e_uni_sub),
        snr(e_uni_sub) - base,
        "~free (VALU)"
    );
    println!(
        "{:<38} {:>8.2}  {:>+8.2}  {:>16}",
        "(a)+(b) full RWQ4",
        snr(e_lloyd_sub),
        snr(e_lloyd_sub) - base,
        "8x ds_load"
    );
    println!("{}", "-".repeat(76));
    println!();
    let free = snr(e_uni_sub) - base;
    let full = snr(e_lloyd_sub) - base;
    println!(
        "  free-decode share of the win: {:.0}%  ({:+.2} dB of {:+.2} dB)",
        100.0 * free / full,
        free,
        full
    );
    println!(
        "  cost of the codebook half   : {:+.2} dB for ~+89% decode time",
        full - free
    );
}
