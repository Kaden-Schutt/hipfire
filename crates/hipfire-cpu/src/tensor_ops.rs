// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-reference tensor ops on [`CpuTensor`].
//!
//! The pure f32 linear / layer-norm / attention / conv / group-norm / upsample /
//! shape kernels that back a pipeline's CPU path and serve as the reference the
//! GPU path is checked against. Moved out of `hipfire-diffusion` so the CPU
//! backend owns the CPU tensor type + its reference math (the GPU/NPU backends
//! own theirs). Runtime-context-dispatched variants stay in the caller.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Error from a CPU tensor op — a shape/precondition violation carrying a
/// human-readable message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuError(pub String);

impl std::fmt::Display for CpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for CpuError {}

/// Result of a CPU tensor op.
pub type CpuResult<T> = Result<T, CpuError>;

/// A CPU tensor: a shape and its row-major f32 data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl CpuTensor {
    /// Zero-filled tensor of the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let len = shape.iter().product();
        Self {
            shape: shape.to_vec(),
            data: vec![0.0; len],
        }
    }

    /// `(rows, cols)` for a 2-D tensor, else an error.
    pub fn rows_cols(&self) -> CpuResult<(usize, usize)> {
        match self.shape.as_slice() {
            [rows, cols] => Ok((*rows, *cols)),
            _ => Err(CpuError(format!(
                "expected 2-D tensor, got shape {:?}",
                self.shape
            ))),
        }
    }
}

/// GELU activation (tanh approximation).
pub fn gelu(value: f32) -> f32 {
    0.5 * value
        * (1.0
            + (std::f32::consts::FRAC_2_SQRT_PI * (value + 0.044_715 * value * value * value))
                .tanh())
}

pub fn clip_causal_self_attention(
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    n_heads: usize,
) -> CpuResult<CpuTensor> {
    let (seq, hidden) = q.rows_cols()?;
    if k.shape.as_slice() != [seq, hidden] || v.shape.as_slice() != [seq, hidden] {
        return Err(CpuError(format!(
            "CLIP causal attention q/k/v shapes {:?}/{:?}/{:?} are incompatible",
            q.shape, k.shape, v.shape
        )));
    }
    if n_heads == 0 || hidden % n_heads != 0 {
        return Err(CpuError(format!(
            "CLIP hidden size {hidden} is not divisible by {n_heads} heads"
        )));
    }
    let head_dim = hidden / n_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut context = CpuTensor::zeros(&[seq, hidden]);
    for head in 0..n_heads {
        let head_off = head * head_dim;
        for i in 0..seq {
            let mut scores = vec![0.0f32; seq];
            for j in 0..seq {
                if j > i {
                    scores[j] = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0;
                for d in 0..head_dim {
                    dot += q.data[i * hidden + head_off + d] * k.data[j * hidden + head_off + d];
                }
                scores[j] = dot * scale;
            }
            softmax_in_place(&mut scores);
            for d in 0..head_dim {
                let mut acc = 0.0;
                for j in 0..seq {
                    acc += scores[j] * v.data[j * hidden + head_off + d];
                }
                context.data[i * hidden + head_off + d] = acc;
            }
        }
    }
    Ok(context)
}

pub fn linear(input: &CpuTensor, weight: &CpuTensor, bias: &CpuTensor) -> CpuResult<CpuTensor> {
    linear_optional_bias(input, weight, Some(bias))
}

pub fn linear_optional_bias(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
) -> CpuResult<CpuTensor> {
    let (rows, in_features) = input.rows_cols()?;
    let (out_features, weight_in) = weight.rows_cols()?;
    if in_features != weight_in {
        return Err(CpuError(format!(
            "linear input width {in_features} != weight input width {weight_in}"
        )));
    }
    if let Some(bias) = bias {
        if bias.shape.as_slice() != [out_features] {
            return Err(CpuError(format!(
                "linear bias shape {:?} != [{out_features}]",
                bias.shape
            )));
        }
    }
    let mut out = CpuTensor::zeros(&[rows, out_features]);
    for row in 0..rows {
        for out_col in 0..out_features {
            let mut acc = bias.map(|bias| bias.data[out_col]).unwrap_or(0.0);
            let weight_row = out_col * in_features;
            let input_row = row * in_features;
            for k in 0..in_features {
                acc += input.data[input_row + k] * weight.data[weight_row + k];
            }
            out.data[row * out_features + out_col] = acc;
        }
    }
    Ok(out)
}

pub fn layer_norm(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    eps: f32,
) -> CpuResult<CpuTensor> {
    let (rows, cols) = input.rows_cols()?;
    if weight.shape.as_slice() != [cols] || bias.shape.as_slice() != [cols] {
        return Err(CpuError(format!(
            "layer_norm weight/bias shapes {:?}/{:?} do not match width {cols}",
            weight.shape, bias.shape
        )));
    }
    let mut out = CpuTensor::zeros(&[rows, cols]);
    for row in 0..rows {
        let base = row * cols;
        let mean = input.data[base..base + cols].iter().sum::<f32>() / cols as f32;
        let var = input.data[base..base + cols]
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / cols as f32;
        let inv_std = (var + eps).sqrt().recip();
        for col in 0..cols {
            out.data[base + col] =
                (input.data[base + col] - mean) * inv_std * weight.data[col] + bias.data[col];
        }
    }
    Ok(out)
}

pub fn tensor_add(a: &CpuTensor, b: &CpuTensor) -> CpuResult<CpuTensor> {
    if a.shape != b.shape {
        return Err(CpuError(format!(
            "tensor_add shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        )));
    }
    Ok(CpuTensor {
        shape: a.shape.clone(),
        data: a.data.iter().zip(&b.data).map(|(a, b)| a + b).collect(),
    })
}

pub fn geglu_gate_3d(projected: &CpuTensor) -> CpuResult<CpuTensor> {
    let [batch, seq, width] = shape3(projected)?;
    if width % 2 != 0 {
        return Err(CpuError(format!(
            "GEGLU projection width {width} is not even"
        )));
    }
    let inner = width / 2;
    let mut gated = CpuTensor::zeros(&[batch, seq, inner]);
    for b in 0..batch {
        for s in 0..seq {
            let src = (b * seq + s) * width;
            let dst = (b * seq + s) * inner;
            for col in 0..inner {
                let value = projected.data[src + col];
                let gate = gelu(projected.data[src + inner + col]);
                gated.data[dst + col] = value * gate;
            }
        }
    }
    Ok(gated)
}

pub fn tensor_map(input: &CpuTensor, f: impl Fn(f32) -> f32) -> CpuTensor {
    CpuTensor {
        shape: input.shape.clone(),
        data: input.data.iter().copied().map(f).collect(),
    }
}

pub fn add_channel_bias_nchw(input: &mut CpuTensor, bias: &CpuTensor) -> CpuResult<()> {
    let [batch, channels, height, width] = shape4(input)?;
    if bias.shape.as_slice() != [batch, channels] {
        return Err(CpuError(format!(
            "channel bias shape {:?} != [{batch}, {channels}]",
            bias.shape
        )));
    }
    for b in 0..batch {
        for c in 0..channels {
            let value = bias.data[b * channels + c];
            for y in 0..height {
                for x in 0..width {
                    input.data[nchw_idx(b, c, y, x, channels, height, width)] += value;
                }
            }
        }
    }
    Ok(())
}

pub fn concat_channels_nchw(a: &CpuTensor, b: &CpuTensor) -> CpuResult<CpuTensor> {
    let [batch, a_channels, height, width] = shape4(a)?;
    let [b_batch, b_channels, b_height, b_width] = shape4(b)?;
    if batch != b_batch || height != b_height || width != b_width {
        return Err(CpuError(format!(
            "cannot concatenate NCHW tensors with shapes {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let out_channels = a_channels + b_channels;
    let mut out = CpuTensor::zeros(&[batch, out_channels, height, width]);
    for b_idx in 0..batch {
        for c in 0..a_channels {
            for y in 0..height {
                for x in 0..width {
                    out.data[nchw_idx(b_idx, c, y, x, out_channels, height, width)] =
                        a.data[nchw_idx(b_idx, c, y, x, a_channels, height, width)];
                }
            }
        }
        for c in 0..b_channels {
            for y in 0..height {
                for x in 0..width {
                    out.data[nchw_idx(b_idx, a_channels + c, y, x, out_channels, height, width)] =
                        b.data[nchw_idx(b_idx, c, y, x, b_channels, height, width)];
                }
            }
        }
    }
    Ok(out)
}

pub fn quick_gelu(value: f32) -> f32 {
    value / (1.0 + (-1.702 * value).exp())
}

pub fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum > 0.0 {
        for value in values {
            *value /= sum;
        }
    }
}

pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

pub fn matmul_vector(vector: &[f32], matrix: &CpuTensor) -> CpuResult<Vec<f32>> {
    let [rows, cols] = shape2(matrix)?;
    if vector.len() == rows {
        let mut out = vec![0.0; cols];
        for (row, value) in vector.iter().enumerate() {
            let base = row * cols;
            for col in 0..cols {
                out[col] += value * matrix.data[base + col];
            }
        }
        return Ok(out);
    }
    if vector.len() == cols {
        let mut out = vec![0.0; rows];
        for (row, out_value) in out.iter_mut().enumerate() {
            let base = row * cols;
            for (col, value) in vector.iter().enumerate() {
                *out_value += value * matrix.data[base + col];
            }
        }
        return Ok(out);
    }
    Err(CpuError(format!(
        "vector length {} does not match projection matrix shape {:?}",
        vector.len(),
        matrix.shape
    )))
}

pub fn conv2d_nchw(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
    padding: usize,
) -> CpuResult<CpuTensor> {
    conv2d_nchw_with_stride(input, weight, bias, padding, 1)
}

pub fn conv2d_nchw_with_stride(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
    padding: usize,
    stride: usize,
) -> CpuResult<CpuTensor> {
    if stride == 0 {
        return Err(CpuError(
            "conv2d stride must be positive".to_string(),
        ));
    }
    let [batch, in_channels, in_h, in_w] = shape4(input)?;
    let [out_channels, weight_in_channels, kernel_h, kernel_w] = shape4(weight)?;
    if in_channels != weight_in_channels {
        return Err(CpuError(format!(
            "conv2d input channels {in_channels} != weight input channels {weight_in_channels}"
        )));
    }
    if let Some(bias) = bias {
        if bias.shape.as_slice() != [out_channels] {
            return Err(CpuError(format!(
                "conv2d bias shape {:?} != [{out_channels}]",
                bias.shape
            )));
        }
    }
    let padded_h = in_h + 2 * padding;
    let padded_w = in_w + 2 * padding;
    if kernel_h > padded_h || kernel_w > padded_w {
        return Err(CpuError(format!(
            "conv2d kernel [{kernel_h}, {kernel_w}] is larger than padded input [{padded_h}, {padded_w}]"
        )));
    }
    let out_h = (padded_h - kernel_h) / stride + 1;
    let out_w = (padded_w - kernel_w) / stride + 1;
    let mut out = CpuTensor::zeros(&[batch, out_channels, out_h, out_w]);
    let plane_len = out_h * out_w;
    out.data
        .par_chunks_mut(plane_len)
        .enumerate()
        .for_each(|(plane_idx, out_plane)| {
            let b = plane_idx / out_channels;
            let oc = plane_idx % out_channels;
            if let Some(bias) = bias {
                out_plane.fill(bias.data[oc]);
            }
            for ic in 0..in_channels {
                let input_base = ((b * in_channels + ic) * in_h) * in_w;
                let weight_base = ((oc * in_channels + ic) * kernel_h) * kernel_w;
                for ky in 0..kernel_h {
                    for oy in 0..out_h {
                        let iy_with_pad = oy * stride + ky;
                        if iy_with_pad < padding || iy_with_pad >= in_h + padding {
                            continue;
                        }
                        let iy = iy_with_pad - padding;
                        let input_row = input_base + iy * in_w;
                        let output_row = oy * out_w;
                        for kx in 0..kernel_w {
                            let ix_offset = kx;
                            let weight_value = weight.data[weight_base + ky * kernel_w + kx];
                            if weight_value == 0.0 {
                                continue;
                            }
                            for ox in 0..out_w {
                                let ix_with_pad = ox * stride + ix_offset;
                                if ix_with_pad < padding || ix_with_pad >= in_w + padding {
                                    continue;
                                }
                                let ix = ix_with_pad - padding;
                                out_plane[output_row + ox] +=
                                    input.data[input_row + ix] * weight_value;
                            }
                        }
                    }
                }
            }
        });
    Ok(out)
}

pub fn group_norm_nchw(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &CpuTensor,
    groups: usize,
    eps: f32,
) -> CpuResult<CpuTensor> {
    let [batch, channels, height, width] = shape4(input)?;
    if groups == 0 || channels % groups != 0 {
        return Err(CpuError(format!(
            "group_norm channels {channels} not divisible by groups {groups}"
        )));
    }
    if weight.shape.as_slice() != [channels] || bias.shape.as_slice() != [channels] {
        return Err(CpuError(format!(
            "group_norm weight/bias shapes {:?}/{:?} != [{channels}]",
            weight.shape, bias.shape
        )));
    }
    let mut out = CpuTensor::zeros(&input.shape);
    let channels_per_group = channels / groups;
    let elems_per_group = channels_per_group * height * width;
    for b in 0..batch {
        for group in 0..groups {
            let c_start = group * channels_per_group;
            let c_end = c_start + channels_per_group;
            let mut sum = 0.0;
            for c in c_start..c_end {
                for y in 0..height {
                    for x in 0..width {
                        sum += input.data[nchw_idx(b, c, y, x, channels, height, width)];
                    }
                }
            }
            let mean = sum / elems_per_group as f32;
            let mut var_sum = 0.0;
            for c in c_start..c_end {
                for y in 0..height {
                    for x in 0..width {
                        let centered =
                            input.data[nchw_idx(b, c, y, x, channels, height, width)] - mean;
                        var_sum += centered * centered;
                    }
                }
            }
            let inv_std = (var_sum / elems_per_group as f32 + eps).sqrt().recip();
            for c in c_start..c_end {
                for y in 0..height {
                    for x in 0..width {
                        let idx = nchw_idx(b, c, y, x, channels, height, width);
                        out.data[idx] =
                            (input.data[idx] - mean) * inv_std * weight.data[c] + bias.data[c];
                    }
                }
            }
        }
    }
    Ok(out)
}

pub fn upsample_nearest2d_nchw(input: &CpuTensor, scale: usize) -> CpuResult<CpuTensor> {
    if scale == 0 {
        return Err(CpuError(
            "upsample scale must be positive".to_string(),
        ));
    }
    let [batch, channels, height, width] = shape4(input)?;
    let out_h = height * scale;
    let out_w = width * scale;
    let mut out = CpuTensor::zeros(&[batch, channels, out_h, out_w]);
    for b in 0..batch {
        for c in 0..channels {
            for oy in 0..out_h {
                let iy = oy / scale;
                for ox in 0..out_w {
                    let ix = ox / scale;
                    out.data[nchw_idx(b, c, oy, ox, channels, out_h, out_w)] =
                        input.data[nchw_idx(b, c, iy, ix, channels, height, width)];
                }
            }
        }
    }
    Ok(out)
}

/// Space-to-depth by `scale` (inverse of pixel-shuffle), NCHW:
/// `[N, C, H, W] -> [N, C*scale*scale, H/scale, W/scale]`. Output channel
/// `c*scale*scale + dy*scale + dx` gathers `input[n, c, oh*scale+dy, ow*scale+dx]`,
/// matching PyTorch/basicsr `pixel_unshuffle`. Used by the RealESRGAN x2 input
/// stage.
pub fn pixel_unshuffle_nchw(input: &CpuTensor, scale: usize) -> CpuResult<CpuTensor> {
    if scale == 0 {
        return Err(CpuError("pixel_unshuffle scale must be positive".to_string()));
    }
    let [batch, channels, height, width] = shape4(input)?;
    if height % scale != 0 || width % scale != 0 {
        return Err(CpuError(format!(
            "pixel_unshuffle input [{height}, {width}] not divisible by scale {scale}"
        )));
    }
    let out_h = height / scale;
    let out_w = width / scale;
    let out_channels = channels * scale * scale;
    let mut out = CpuTensor::zeros(&[batch, out_channels, out_h, out_w]);
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height {
                let dy = y % scale;
                let by = y / scale;
                for x in 0..width {
                    let dx = x % scale;
                    let bx = x / scale;
                    let c_out = c * scale * scale + dy * scale + dx;
                    out.data[nchw_idx(b, c_out, by, bx, out_channels, out_h, out_w)] =
                        input.data[nchw_idx(b, c, y, x, channels, height, width)];
                }
            }
        }
    }
    Ok(out)
}

pub fn shape4(tensor: &CpuTensor) -> CpuResult<[usize; 4]> {
    match tensor.shape.as_slice() {
        [a, b, c, d] => Ok([*a, *b, *c, *d]),
        _ => Err(CpuError(format!(
            "expected 4-D NCHW tensor, got {:?}",
            tensor.shape
        ))),
    }
}

pub fn shape2(tensor: &CpuTensor) -> CpuResult<[usize; 2]> {
    match tensor.shape.as_slice() {
        [a, b] => Ok([*a, *b]),
        _ => Err(CpuError(format!(
            "expected 2-D tensor, got {:?}",
            tensor.shape
        ))),
    }
}

pub fn shape3(tensor: &CpuTensor) -> CpuResult<[usize; 3]> {
    match tensor.shape.as_slice() {
        [a, b, c] => Ok([*a, *b, *c]),
        _ => Err(CpuError(format!(
            "expected 3-D tensor, got {:?}",
            tensor.shape
        ))),
    }
}

pub fn nchw_idx(
    batch: usize,
    channel: usize,
    y: usize,
    x: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> usize {
    (((batch * channels + channel) * height + y) * width) + x
}

/// Deterministic CPU-reference GEMM oracle for verifying quant/WMMA GPU kernels.
///
/// Computes `y[b*m + r] = Σ_k weight[r*k + k]·x[b*k + k]` — a weight × activation
/// product with row-major `weight` `[m, k]`, activation `x` `[n, k]`, and output
/// `y` `[n, m]` (the `[n_batch, m_rows]` layout the kernel benches expect). Shared
/// so kernel example/bench binaries stop each re-defining their own `cpu_gemm`.
pub fn cpu_reference_gemm(weight: &[f32], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n * m];
    for b in 0..n {
        for r in 0..m {
            let mut acc = 0.0f32;
            for c in 0..k {
                acc += weight[r * k + c] * x[b * k + c];
            }
            y[b * m + r] = acc;
        }
    }
    y
}
