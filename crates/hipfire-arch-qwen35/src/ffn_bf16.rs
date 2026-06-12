// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! CPU BF16 oracle for the dense Qwen3.5 FFN SwiGLU/down epilogue.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnBf16Mode {
    Off,
    Compare,
    Cpu,
    Xdna1,
}

#[derive(Debug)]
pub struct FfnBf16Config {
    pub mode: FfnBf16Mode,
    pub layer: LayerSelect,
    pub trace: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerSelect {
    One(usize),
    All,
}

#[derive(Debug)]
pub struct Bf16DownShadow {
    pub w_down: Vec<f32>,
    pub m: usize,
    pub k: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DiffStats {
    pub n: usize,
    pub max_abs: f32,
    pub mean_abs: f32,
    pub rms: f32,
    pub n_nan: usize,
    pub n_inf: usize,
}

pub fn config() -> &'static FfnBf16Config {
    static CONFIG: OnceLock<FfnBf16Config> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let mode = match std::env::var("HIPFIRE_QWEN35_FFN_BF16")
            .unwrap_or_else(|_| "off".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "compare" => FfnBf16Mode::Compare,
            "cpu" => FfnBf16Mode::Cpu,
            "xdna1" => FfnBf16Mode::Xdna1,
            "off" | "0" | "false" | "" => FfnBf16Mode::Off,
            other => {
                panic!("HIPFIRE_QWEN35_FFN_BF16 must be off|compare|cpu|xdna1, got {other:?}")
            }
        };
        let layer = match std::env::var("HIPFIRE_QWEN35_FFN_BF16_LAYER")
            .unwrap_or_else(|_| "0".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "all" => LayerSelect::All,
            s => LayerSelect::One(
                s.parse::<usize>()
                    .expect("HIPFIRE_QWEN35_FFN_BF16_LAYER must be <n>|all"),
            ),
        };
        let trace = std::env::var("HIPFIRE_QWEN35_FFN_BF16_TRACE")
            .map(|v| !matches!(v.trim(), "" | "0" | "false" | "False" | "FALSE"))
            .unwrap_or(false);
        FfnBf16Config { mode, layer, trace }
    })
}

pub fn enabled() -> bool {
    config().mode != FfnBf16Mode::Off
}

pub fn layer_selected(layer_idx: usize) -> bool {
    match config().layer {
        LayerSelect::One(n) => layer_idx == n,
        LayerSelect::All => true,
    }
}

pub fn f32_to_bf16_bits_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn round_f32_to_bf16(x: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits_rne(x))
}

pub fn decode_w_down_shadow(
    data: &[u8],
    quant_type: u8,
    m: usize,
    k: usize,
) -> Option<Bf16DownShadow> {
    let expected = m.checked_mul(k)?;
    let w_down = match quant_type {
        2 => {
            if data.len() != expected * 4 {
                return None;
            }
            data.chunks_exact(4)
                .map(|c| round_f32_to_bf16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect()
        }
        16 => {
            if data.len() != expected * 2 {
                return None;
            }
            data.chunks_exact(2)
                .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        _ => return None,
    };
    Some(Bf16DownShadow { w_down, m, k })
}

pub fn swiglu_down_bf16_cpu(
    gate: &[f32],
    up: &[f32],
    residual: &[f32],
    shadow: &Bf16DownShadow,
) -> Vec<f32> {
    assert_eq!(gate.len(), shadow.k);
    assert_eq!(up.len(), shadow.k);
    assert_eq!(residual.len(), shadow.m);

    let mut hidden = vec![0.0f32; shadow.k];
    for i in 0..shadow.k {
        let g = round_f32_to_bf16(gate[i]);
        let u = round_f32_to_bf16(up[i]);
        let silu = g / (1.0 + (-g).exp());
        hidden[i] = round_f32_to_bf16(silu * u);
    }

    let mut out = residual.to_vec();
    for row in 0..shadow.m {
        let w_row = &shadow.w_down[row * shadow.k..(row + 1) * shadow.k];
        let mut acc = 0.0f32;
        for col in 0..shadow.k {
            acc += w_row[col] * hidden[col];
        }
        out[row] += acc;
    }
    out
}

pub fn diff_stats(a: &[f32], b: &[f32]) -> DiffStats {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x.is_nan() || y.is_nan() {
            n_nan += 1;
            continue;
        }
        if x.is_infinite() || y.is_infinite() {
            n_inf += 1;
            continue;
        }
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        sum_sq += (d as f64) * (d as f64);
    }
    let n = a.len();
    DiffStats {
        n,
        max_abs,
        mean_abs: (sum_abs / n.max(1) as f64) as f32,
        rms: (sum_sq / n.max(1) as f64).sqrt() as f32,
        n_nan,
        n_inf,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_ties_to_even() {
        let one = 1.0f32;
        let half_ulp = f32::from_bits(one.to_bits() + 0x8000);
        assert_eq!(f32_to_bf16_bits_rne(half_ulp), 0x3f80);

        let odd_lsb = f32::from_bits(0x3f81_0000);
        let odd_half_ulp = f32::from_bits(odd_lsb.to_bits() + 0x8000);
        assert_eq!(f32_to_bf16_bits_rne(odd_half_ulp), 0x3f82);
    }

    #[test]
    fn decode_bf16_and_f32_shadow() {
        let f32_bytes: Vec<u8> = [1.0001f32, -2.25, 3.5, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f32_shadow = decode_w_down_shadow(&f32_bytes, 2, 2, 2).unwrap();
        assert_eq!(f32_shadow.w_down[0], round_f32_to_bf16(1.0001));

        let bf16_bytes: Vec<u8> = [0x3f80u16, 0xc000, 0x4060, 0x4080]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bf16_shadow = decode_w_down_shadow(&bf16_bytes, 16, 2, 2).unwrap();
        assert_eq!(bf16_shadow.w_down, vec![1.0, -2.0, 3.5, 4.0]);
    }

    #[test]
    fn tiny_swiglu_down_bf16_cpu() {
        let shadow = Bf16DownShadow {
            w_down: vec![1.0, 2.0, -1.0, 0.5],
            m: 2,
            k: 2,
        };
        let out = swiglu_down_bf16_cpu(&[0.0, 1.0], &[2.0, -3.0], &[0.5, -0.5], &shadow);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
