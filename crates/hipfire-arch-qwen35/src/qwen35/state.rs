// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 DeltaNet recurrent state: `DeltaNetState`, its quantization mode,
//! and the sizing helpers that pick redundancy / precision from config.

use super::*;

/// Persistent state for DeltaNet layers across tokens.
/// State quantization mode for DeltaNet S matrix.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StateQuant {
    FP32,
    Q8,
    Q4,
}

/// Redundancy proxy for DeltaNet recurrent-state precision:
/// `linear_key_head_dim × linear_num_value_heads`. The recurrent state is the
/// most precision-sensitive tensor in the model — its quant error compounds
/// across the sequence — so low-redundancy models (small `head_dim × n_heads`)
/// lack the slack to absorb Q8 noise and develop attractors on long decode
/// (observed on 0.8B/2B). State traffic is only ~1–3% of per-token bandwidth,
/// so paying FP32 here is nearly free.
pub fn deltanet_state_redundancy(config: &Qwen35Config) -> usize {
    config.linear_key_head_dim * config.linear_num_value_heads
}

/// Redundancy threshold below which the DeltaNet state defaults to FP32.
/// Env-tunable via `HIPFIRE_DN_STATE_FP32_BELOW`; defaults to `usize::MAX`
/// ⇒ **FP32 for all current models** (2026-06-15 decision: Q8 caused
/// long-decode attractors on low-redundancy models and there is no FP16 state
/// kernel yet — see TODO.md). Lower it (e.g. 3000) to put 0.8B (redundancy
/// 2048) on FP32 while 9B (4096) / 27B (6144) use Q8.
pub fn deltanet_state_fp32_below() -> usize {
    std::env::var("HIPFIRE_DN_STATE_FP32_BELOW")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(usize::MAX)
}

/// Default DeltaNet state precision, gated on redundancy (`head_dim × n_heads`)
/// rather than parameter count. Below the threshold → FP32 (the numerical
/// anchor); at/above → Q8 (+error-feedback). Explicit operator overrides are
/// handled upstream (daemon `state_quant` param / `HIPFIRE_QWEN35_STATE_QUANT`).
///
/// NOTE: FP32 state is unsupported by the **tree** DeltaNet replay path
/// (`gated_delta_net_q8_tree_batch_seq`); tree-based spec-decode requires Q8.
/// Use the non-tree MTP/DFlash draft paths with FP32 state. See TODO.md.
pub fn default_state_quant(config: &Qwen35Config) -> StateQuant {
    if deltanet_state_redundancy(config) < deltanet_state_fp32_below() {
        StateQuant::FP32
    } else {
        StateQuant::Q8
    }
}

pub struct DeltaNetState {
    /// S matrix storage — FP32 or Q8 depending on quant mode
    pub s_matrices: Vec<GpuTensor>,
    /// Per-head scale factors (only used for Q8 mode)
    pub s_scales: Vec<GpuTensor>,
    /// Conv ring buffer: [n_deltanet_layers × conv_channels × (kernel_size-1)] FP32
    pub conv_states: Vec<GpuTensor>,
    /// Per-element f16 error-feedback residual for Q8 state requant (sigma-delta
    /// noise-shaping). Empty unless Q8 + `HIPFIRE_DN_STATE_EF`. Same element count
    /// as `s_matrices`; carries the previous step's quant error so the next
    /// requant cancels it — DeltaNet's contractive decay damps the shaped noise,
    /// yielding ~FP32-grade state at Q8's byte container.
    pub s_ef_residual: Vec<GpuTensor>,
    /// Current quantization mode
    pub quant: StateQuant,
}

impl DeltaNetState {
    /// EF residual for a delta-layer, if error-feedback is active (Q8 + flag).
    /// `None` ⇒ callers pass null ⇒ kernel uses the legacy stochastic-rounding requant.
    #[inline]
    pub fn ef_residual(&self, idx: usize) -> Option<&GpuTensor> {
        self.s_ef_residual.get(idx)
    }

    pub fn new(gpu: &mut Gpu, config: &Qwen35Config) -> HipResult<Self> {
        Self::new_with_quant(gpu, config, default_state_quant(config))
    }

    pub fn new_with_quant(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        quant: StateQuant,
    ) -> HipResult<Self> {
        let n_delta_layers = config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count();
        let s_dim = config.linear_key_head_dim; // 128
        let n_heads = config.linear_num_value_heads; // 16
        let s_size = n_heads * s_dim * s_dim; // 16 * 128 * 128 = 262144

        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_state_size = conv_channels * (config.conv_kernel_dim - 1);

        // Error-feedback (sigma-delta) requant for Q8 state — DEFAULT ON as of
        // 2026-06-08. q8_ef ≈ FP32 coherence at −0.7% decode vs FP32's −4.5% (best
        // spec-decode τ too), and far better than stochastic Q8 — DFlash 27b-prose
        // unique_ratio 0.625 vs 0.555, max_freq 0.055 vs 0.078. Also makes the DN
        // state DETERMINISTIC (no stochastic dither). Opt OUT with
        // HIPFIRE_DN_STATE_EF=0. Q8-only (FP32 has no requant; Q4 EF is future
        // work; the multi-GPU band split is still stochastic — new_with_quant_multi
        // leaves s_ef_residual empty). Residual is f16 per-element.
        let ef_enabled = quant == StateQuant::Q8
            && std::env::var("HIPFIRE_DN_STATE_EF")
                .map(|v| v != "0")
                .unwrap_or(true);

        let mut s_matrices = Vec::with_capacity(n_delta_layers);
        let mut s_scales = Vec::with_capacity(n_delta_layers);
        let mut conv_states = Vec::with_capacity(n_delta_layers);
        let mut s_ef_residual = Vec::with_capacity(if ef_enabled { n_delta_layers } else { 0 });
        for _ in 0..n_delta_layers {
            match quant {
                StateQuant::FP32 => {
                    s_matrices.push(gpu.zeros(&[s_size], DType::F32)?);
                    s_scales.push(gpu.zeros(&[n_heads], DType::F32)?);
                }
                StateQuant::Q8 => {
                    // int8 state: s_size bytes (1 byte each), per-row scales
                    let buf = gpu.hip.malloc(s_size)?;
                    gpu.hip.memset(&buf, 0, s_size)?;
                    s_matrices.push(GpuTensor {
                        buf,
                        shape: vec![s_size],
                        dtype: DType::F32,
                    });
                    s_scales.push(gpu.zeros(&[n_heads * s_dim], DType::F32)?);
                }
                StateQuant::Q4 => {
                    // 4-bit nibble-packed: s_size/2 bytes, per-row scales
                    let buf = gpu.hip.malloc(s_size / 2)?;
                    gpu.hip.memset(&buf, 0, s_size / 2)?;
                    s_matrices.push(GpuTensor {
                        buf,
                        shape: vec![s_size / 2],
                        dtype: DType::F32,
                    });
                    s_scales.push(gpu.zeros(&[n_heads * s_dim], DType::F32)?);
                }
            }
            if ef_enabled {
                s_ef_residual.push(gpu.zeros(&[s_size], DType::F16)?);
            }
            conv_states.push(gpu.zeros(&[conv_state_size], DType::F32)?);
        }
        Ok(Self {
            s_matrices,
            s_scales,
            conv_states,
            s_ef_residual,
            quant,
        })
    }

    /// Free all GPU tensors. Call before drop to return VRAM.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self.s_matrices {
            let _ = gpu.free_tensor(t);
        }
        for t in self.s_scales {
            let _ = gpu.free_tensor(t);
        }
        for t in self.conv_states {
            let _ = gpu.free_tensor(t);
        }
        for t in self.s_ef_residual {
            let _ = gpu.free_tensor(t);
        }
    }

    /// Reset all DeltaNet recurrent buffers to zero in place. Lets callers
    /// reuse a single `DeltaNetState` across independent chunks/sequences
    /// without allocating per chunk (which leaks since DeltaNetState has no
    /// Drop). Mirrors `ModelSlot::reset_state` in speculative.rs.
    pub fn reset(&mut self, gpu: &mut Gpu) {
        match gpu.active_stream.as_ref() {
            Some(stream) => {
                for s in &self.s_matrices {
                    let _ = gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream);
                }
                for s in &self.s_scales {
                    let _ = gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream);
                }
                for s in &self.conv_states {
                    let _ = gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream);
                }
                for s in &self.s_ef_residual {
                    let _ = gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream);
                }
            }
            None => {
                for s in &self.s_matrices {
                    let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                }
                for s in &self.s_scales {
                    let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                }
                for s in &self.conv_states {
                    let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                }
                for s in &self.s_ef_residual {
                    let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                }
            }
        }
    }

    /// Multi-GPU companion to `new_with_quant`. Each LA-layer's state is
    /// allocated on the device that owns the layer in the multi-GPU band
    /// split: `gpus.devices[gpus.device_for_layer(orig_layer_idx)]` for the
    /// `orig_layer_idx` of the LA-layer. Returns the state alongside the
    /// `la_to_device` mapping the daemon needs to route reset memsets to
    /// the correct device.
    pub fn new_with_quant_multi(
        gpus: &mut Gpus,
        config: &Qwen35Config,
        quant: StateQuant,
    ) -> HipResult<(Self, Vec<u8>)> {
        let s_dim = config.linear_key_head_dim;
        let n_heads = config.linear_num_value_heads;
        let s_size = n_heads * s_dim * s_dim;
        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_state_size = conv_channels * (config.conv_kernel_dim - 1);

        let mut s_matrices = Vec::new();
        let mut s_scales = Vec::new();
        let mut conv_states = Vec::new();
        let mut la_to_device: Vec<u8> = Vec::new();

        for (orig_layer_idx, lt) in config.layer_types.iter().enumerate() {
            if *lt != LayerType::LinearAttention {
                continue;
            }
            let dev_idx = gpus.device_for_layer(orig_layer_idx);
            la_to_device.push(dev_idx as u8);
            let g = &mut gpus.devices[dev_idx];
            // g.hip.malloc/memset bypass the Stage 2 bind_thread audit
            // (HipRuntime methods don't carry a device id). Bind explicitly
            // before any raw HIP ops so allocations land on the right device.
            g.bind_thread()?;
            match quant {
                StateQuant::FP32 => {
                    s_matrices.push(g.zeros(&[s_size], DType::F32)?);
                    s_scales.push(g.zeros(&[n_heads], DType::F32)?);
                }
                StateQuant::Q8 => {
                    let buf = g.hip.malloc(s_size)?;
                    g.hip.memset(&buf, 0, s_size)?;
                    s_matrices.push(GpuTensor {
                        buf,
                        shape: vec![s_size],
                        dtype: DType::F32,
                    });
                    s_scales.push(g.zeros(&[n_heads * s_dim], DType::F32)?);
                }
                StateQuant::Q4 => {
                    let buf = g.hip.malloc(s_size / 2)?;
                    g.hip.memset(&buf, 0, s_size / 2)?;
                    s_matrices.push(GpuTensor {
                        buf,
                        shape: vec![s_size / 2],
                        dtype: DType::F32,
                    });
                    s_scales.push(g.zeros(&[n_heads * s_dim], DType::F32)?);
                }
            }
            conv_states.push(g.zeros(&[conv_state_size], DType::F32)?);
        }
        Ok((
            Self {
                s_matrices,
                s_scales,
                conv_states,
                // EF residual not wired for the multi-GPU band split (would need
                // per-device residual alloc routed by device_for_layer); empty ⇒
                // ef_residual() returns None ⇒ kernel uses the stochastic path.
                s_ef_residual: Vec::new(),
                quant,
            },
            la_to_device,
        ))
    }

    /// Free per-LA-layer tensors on the devices listed in `la_to_device`
    /// (the second tuple element returned by `new_with_quant_multi`).
    pub fn free_gpu_multi(self, gpus: &mut Gpus, la_to_device: &[u8]) {
        for (i, t) in self.s_matrices.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
        for (i, t) in self.s_scales.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
        for (i, t) in self.conv_states.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
    }
}

/// Plug `DeltaNetState` into the neutral family-seam state container
/// (`hipfire_runtime::sequence_state`). The serving layer holds the recurrent
/// state as `Box<dyn RecurrentMixerState>` inside a `SequenceState`, and
/// recovers the concrete `&DeltaNetState` for its monomorphized hot path via
/// `SequenceState::recurrent_as::<DeltaNetState>()` — no per-token dyn cost.
/// See docs/plans/2026-06-23-seam-finish-and-mamba2.md (P2c, Slice 1).
impl hipfire_runtime::sequence_state::RecurrentMixerState for DeltaNetState {
    fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        // The inherent `reset` zeros every recurrent buffer (s_matrices /
        // s_scales / conv_states / s_ef_residual) and swallows memset errors,
        // so it is infallible — always report Ok.
        DeltaNetState::reset(self, gpu);
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}
