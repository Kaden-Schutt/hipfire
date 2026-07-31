// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compare MQ4N against the MQ4G256 min/max fit on real safetensors weights.
//!
//! Both encode 256 weights into 136 B, so this is a like-for-like quality
//! comparison at identical bandwidth. Reports SNR in the FWHT-rotated domain —
//! which equals the weight domain, because the transform is orthonormal.
//!
//! ```text
//! cargo run --release -p hipfire-quantize --example mq4n_quality -- <model.safetensors> [max_tensors]
//! ```

use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32};
use hipfire_quantize::fwht::{fwht_256, gen_signs};
use hipfire_quantize::mqn::{self, Spec};
use hipfire_quantize::safetensors_file::SafetensorsFile;
use std::path::Path;

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

/// Legacy MQ4G256: min/max affine over the rotated group, 16 uniform levels.
fn legacy_sq_err(g: &[f32; 256]) -> f64 {
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

fn mq4n_sq_err(g: &[f32; 256]) -> f64 {
    let mut enc = [0u8; mq4n::GROUP_BYTES];
    mq4n::encode_group(g, &mut enc);
    let mut dec = [0.0f32; mq4n::GROUP];
    mq4n::decode_group(&enc, &mut dec);
    g.iter()
        .zip(dec.iter())
        .map(|(&a, &b)| {
            let d = (a - b) as f64;
            d * d
        })
        .sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: mq4n_quality <model.safetensors> [max_tensors]");
        std::process::exit(2);
    }
    let max_tensors: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);

    let st = SafetensorsFile::open(Path::new(&args[1])).expect("open safetensors");
    let (s1, s2) = (gen_signs(0x9E37_79B9, 256), gen_signs(0x85EB_CA6B, 256));

    let mut names: Vec<String> = st.tensor_names().into_iter().map(String::from).collect();
    names.sort();

    println!(
        "{:<52} {:>10} {:>9} {:>9} {:>8}",
        "tensor", "shape", "legacy", "mq4n", "gain"
    );
    println!("{}", "-".repeat(92));

    let (mut tot_old, mut tot_new, mut n_used) = (0.0f64, 0.0f64, 0usize);
    let mut energy = 0.0f64;

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

        // Deterministic stride so big tensors stay cheap without biasing.
        let n_groups = w.len() / 256;
        let stride = (n_groups / 2048).max(1);
        let (mut t_old, mut t_new, mut t_e) = (0.0f64, 0.0f64, 0.0f64);
        let mut b = 0usize;
        while b < n_groups {
            let mut g = [0.0f32; 256];
            g.copy_from_slice(&w[b * 256..b * 256 + 256]);
            fwht_256(&mut g, &s1, &s2);
            t_old += legacy_sq_err(&g);
            t_new += mq4n_sq_err(&g);
            t_e += g.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
            b += stride;
        }
        let gain = 10.0 * (t_old / t_new).log10();
        println!(
            "{:<52} {:>10} {:>8.2}dB {:>7.2}dB {:>+7.2}",
            name.chars().take(52).collect::<String>(),
            format!("{}x{}", meta.shape[0], meta.shape[1]),
            10.0 * (t_e / t_old).log10(),
            10.0 * (t_e / t_new).log10(),
            gain
        );
        tot_old += t_old;
        tot_new += t_new;
        energy += t_e;
        n_used += 1;
        st.drop_tensor_pages(&name);
    }

    if n_used == 0 {
        eprintln!("no eligible 2-D tensors with K % 256 == 0");
        std::process::exit(1);
    }
    println!("{}", "-".repeat(92));
    let g = 10.0 * (tot_old / tot_new).log10();
    println!(
        "AGGREGATE over {n_used} tensors:  legacy {:.2} dB -> mq4n {:.2} dB",
        10.0 * (energy / tot_old).log10(),
        10.0 * (energy / tot_new).log10()
    );
    println!(
        "  gain {g:+.2} dB   squared error x{:.3}  ({:.1}% lower)  at IDENTICAL 136 B/group",
        10f64.powf(-g / 10.0),
        (1.0 - 10f64.powf(-g / 10.0)) * 100.0
    );
}
