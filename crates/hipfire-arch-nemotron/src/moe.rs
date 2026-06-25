// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! nemotron_h routed MoE (`E`) block.
//!
//! This is the decode-first correctness path for Nano-30B A3B: sigmoid router
//! scores, top-k routed experts, normalized/scaled route weights, plus one
//! shared ReLU2 expert. The current implementation intentionally reuses the
//! existing single-expert ReLU2 MLP primitive and performs a small D2H transfer
//! for the selected expert ids/weights. A later FU6 optimization can replace the
//! selected-expert loop with the existing expert-indexed grouped kernels.

use crate::mlp::mlp_relu2;
use crate::mlp::MlpRelu2Gpu;
use crate::weight::LinearWeight;
use crate::MoeConfig;
use hip_bridge::{HipError, HipResult};
use rdna_compute::{DType, Gpu, GpuTensor};

#[derive(Clone, Debug)]
pub struct MoeExpertWeights {
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct MoeWeights {
    pub router: Vec<f32>,
    pub expert_bias: Vec<f32>,
    pub shared_up: Vec<f32>,
    pub shared_down: Vec<f32>,
    pub experts: Vec<MoeExpertWeights>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn matvec(w: &[f32], x: &[f32], out: usize, n_in: usize) -> Vec<f32> {
    (0..out)
        .map(|i| {
            w[i * n_in..i * n_in + n_in]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

fn topk_indices(scores_for_choice: &[f32], k: usize) -> Vec<usize> {
    let mut indexed = scores_for_choice
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

/// CPU reference for a routed ReLU2 MoE block.
pub fn moe_relu2(cfg: &MoeConfig, w: &MoeWeights, x: &[f32], hidden: usize) -> Vec<f32> {
    assert_eq!(cfg.n_group, 1, "nemotron MoE CPU oracle expects n_group=1");
    assert_eq!(
        cfg.topk_group, 1,
        "nemotron MoE CPU oracle expects topk_group=1"
    );
    assert_eq!(w.experts.len(), cfg.n_routed_experts);
    let logits = matvec(&w.router, x, cfg.n_routed_experts, hidden);
    let scores = logits.into_iter().map(sigmoid).collect::<Vec<_>>();
    let choice = scores
        .iter()
        .zip(&w.expert_bias)
        .map(|(score, bias)| score + bias)
        .collect::<Vec<_>>();
    let topk = topk_indices(&choice, cfg.num_experts_per_tok);
    let mut weights = topk.iter().map(|&i| scores[i]).collect::<Vec<_>>();
    if cfg.norm_topk_prob {
        let denom = weights.iter().sum::<f32>() + 1e-20;
        for weight in &mut weights {
            *weight /= denom;
        }
    }
    for weight in &mut weights {
        *weight *= cfg.routed_scaling_factor;
    }

    let mut out = mlp_relu2(
        &w.shared_up,
        &w.shared_down,
        x,
        hidden,
        cfg.shared_expert_intermediate_size,
    );
    for (&expert_idx, &weight) in topk.iter().zip(&weights) {
        let expert = &w.experts[expert_idx];
        let expert_out = mlp_relu2(&expert.up, &expert.down, x, hidden, cfg.intermediate_size);
        for i in 0..hidden {
            out[i] += weight * expert_out[i];
        }
    }
    out
}

pub struct MoeRelu2Gpu {
    cfg: MoeConfig,
    router: LinearWeight,
    expert_bias: GpuTensor,
    shared: MlpRelu2Gpu,
    experts: Vec<MlpRelu2Gpu>,
    router_scores: GpuTensor,
    topk_indices: GpuTensor,
    topk_weights: GpuTensor,
    out: GpuTensor,
}

impl MoeRelu2Gpu {
    pub fn new(
        gpu: &mut Gpu,
        hidden: usize,
        cfg: MoeConfig,
        weights: &MoeWeights,
    ) -> HipResult<Self> {
        let router =
            LinearWeight::F32(gpu.upload_f32(&weights.router, &[cfg.n_routed_experts, hidden])?);
        let shared = MlpRelu2Gpu::new(
            gpu,
            hidden,
            cfg.shared_expert_intermediate_size,
            &weights.shared_up,
            &weights.shared_down,
        )?;
        let mut experts = Vec::with_capacity(cfg.n_routed_experts);
        for expert in &weights.experts {
            experts.push(MlpRelu2Gpu::new(
                gpu,
                hidden,
                cfg.intermediate_size,
                &expert.up,
                &expert.down,
            )?);
        }
        Self::assemble(
            gpu,
            hidden,
            cfg,
            router,
            &weights.expert_bias,
            shared,
            experts,
        )
    }

    pub fn new_quant(
        gpu: &mut Gpu,
        hidden: usize,
        cfg: MoeConfig,
        router: LinearWeight,
        expert_bias: &[f32],
        shared: MlpRelu2Gpu,
        experts: Vec<MlpRelu2Gpu>,
    ) -> HipResult<Self> {
        Self::assemble(gpu, hidden, cfg, router, expert_bias, shared, experts)
    }

    fn assemble(
        gpu: &mut Gpu,
        hidden: usize,
        cfg: MoeConfig,
        router: LinearWeight,
        expert_bias: &[f32],
        shared: MlpRelu2Gpu,
        experts: Vec<MlpRelu2Gpu>,
    ) -> HipResult<Self> {
        if cfg.n_group != 1 || cfg.topk_group != 1 {
            return Err(HipError::unsupported(
                "nemotron MoE decode currently supports n_group=1 and topk_group=1",
            ));
        }
        if !cfg.norm_topk_prob {
            return Err(HipError::unsupported(
                "nemotron MoE decode currently expects norm_topk_prob=true",
            ));
        }
        if experts.len() != cfg.n_routed_experts {
            return Err(HipError::new(
                0,
                "nemotron MoE expert count does not match config",
            ));
        }
        Ok(Self {
            cfg,
            router,
            expert_bias: gpu.upload_f32(expert_bias, &[cfg.n_routed_experts])?,
            shared,
            experts,
            router_scores: gpu.zeros(&[cfg.n_routed_experts], DType::F32)?,
            topk_indices: gpu.zeros(&[cfg.num_experts_per_tok], DType::F32)?,
            topk_weights: gpu.zeros(&[cfg.num_experts_per_tok], DType::F32)?,
            out: gpu.zeros(&[hidden], DType::F32)?,
        })
    }

    pub fn forward(&mut self, gpu: &mut Gpu, x: &GpuTensor) -> HipResult<&GpuTensor> {
        gpu.fill_f32(&self.out, 0.0)?;

        self.router.gemv(gpu, x, &self.router_scores)?;
        gpu.sigmoid_f32(&self.router_scores)?;
        gpu.deepseek4_moe_topk_bias_aware_f32(
            &self.router_scores,
            &self.expert_bias,
            &self.topk_indices,
            &self.topk_weights,
            self.cfg.n_routed_experts as i32,
            self.cfg.num_experts_per_tok as i32,
            self.cfg.routed_scaling_factor,
        )?;

        let shared_out = self.shared.forward(gpu, x)?;
        gpu.add_inplace_f32(&self.out, shared_out)?;

        let raw_indices = gpu.download_raw(&self.topk_indices, self.cfg.num_experts_per_tok * 4)?;
        let weights = gpu.download_f32(&self.topk_weights)?;
        for (slot, weight) in weights.iter().enumerate() {
            let off = slot * 4;
            let idx = i32::from_le_bytes([
                raw_indices[off],
                raw_indices[off + 1],
                raw_indices[off + 2],
                raw_indices[off + 3],
            ]);
            if idx < 0 {
                continue;
            }
            let expert_idx = idx as usize;
            if expert_idx >= self.experts.len() {
                return Err(HipError::new(
                    0,
                    &format!("nemotron MoE router selected invalid expert {expert_idx}"),
                ));
            }
            let expert_out = self.experts[expert_idx].forward(gpu, x)?;
            gpu.scaled_add_inplace_cpu_scalar_f32(&self.out, expert_out, *weight)?;
        }
        Ok(&self.out)
    }

    pub fn free(self, gpu: &mut Gpu) {
        self.router.free(gpu);
        let _ = gpu.free_tensor(self.expert_bias);
        self.shared.free(gpu);
        for expert in self.experts {
            expert.free(gpu);
        }
        for t in [
            self.router_scores,
            self.topk_indices,
            self.topk_weights,
            self.out,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_moe_relu2_routes_topk_and_shared_expert() {
        let cfg = MoeConfig {
            n_routed_experts: 3,
            num_experts_per_tok: 2,
            intermediate_size: 2,
            n_shared_experts: 1,
            shared_expert_intermediate_size: 2,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 2.0,
        };
        let ident_up = vec![1.0, 0.0, 0.0, 1.0];
        let ident_down = ident_up.clone();
        let weights = MoeWeights {
            router: vec![2.0, 0.0, 0.0, 1.0, -1.0, 0.0],
            expert_bias: vec![0.0, 0.0, 0.0],
            shared_up: ident_up.clone(),
            shared_down: ident_down.clone(),
            experts: vec![
                MoeExpertWeights {
                    up: ident_up.clone(),
                    down: ident_down.clone(),
                },
                MoeExpertWeights {
                    up: vec![0.0, 1.0, 1.0, 0.0],
                    down: ident_down.clone(),
                },
                MoeExpertWeights {
                    up: ident_up.clone(),
                    down: ident_down.clone(),
                },
            ],
        };
        let out = moe_relu2(&cfg, &weights, &[1.0, 2.0], 2);
        assert!(out[0].is_finite());
        assert!(out[1].is_finite());
        assert!(out[0] > 1.0);
        assert!(out[1] > 4.0);
    }
}
