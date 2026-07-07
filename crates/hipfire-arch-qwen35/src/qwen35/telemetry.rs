// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MoE router selection histogram: thread-local telemetry for expert routing.

use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MoeRouterHistogram {
    pub num_experts: usize,
    pub k_top: usize,
    pub routed_tokens: u64,
    pub routed_slots: u64,
    pub top1_histogram: Vec<u64>,
    pub topk_histogram: Vec<u64>,
    pub weight_sums: Vec<f64>,
    pub dropped_indices: u64,
    pub per_layer: Vec<MoeRouterLayerHistogram>,
}

#[derive(Debug, Clone)]
pub struct MoeRouterLayerHistogram {
    pub layer_idx: usize,
    pub routed_tokens: u64,
    pub routed_slots: u64,
    pub top1_histogram: Vec<u64>,
    pub topk_histogram: Vec<u64>,
    pub weight_sums: Vec<f64>,
    pub dropped_indices: u64,
    pub cooccurrence: HashMap<u64, u64>,
}

impl MoeRouterHistogram {
    fn new(num_experts: usize, k_top: usize) -> Self {
        Self {
            num_experts,
            k_top,
            routed_tokens: 0,
            routed_slots: 0,
            top1_histogram: vec![0; num_experts],
            topk_histogram: vec![0; num_experts],
            weight_sums: vec![0.0; num_experts],
            dropped_indices: 0,
            per_layer: Vec::new(),
        }
    }

    fn ensure_layer(&mut self, layer_idx: usize) -> &mut MoeRouterLayerHistogram {
        while self.per_layer.len() <= layer_idx {
            let next = self.per_layer.len();
            self.per_layer.push(MoeRouterLayerHistogram {
                layer_idx: next,
                routed_tokens: 0,
                routed_slots: 0,
                top1_histogram: vec![0; self.num_experts],
                topk_histogram: vec![0; self.num_experts],
                weight_sums: vec![0.0; self.num_experts],
                dropped_indices: 0,
                cooccurrence: HashMap::new(),
            });
        }
        &mut self.per_layer[layer_idx]
    }
}

thread_local! {
    static MOE_ROUTER_HISTOGRAM: RefCell<Option<MoeRouterHistogram>> = const { RefCell::new(None) };
}

pub fn reset_moe_router_histogram(num_experts: usize, k_top: usize) {
    MOE_ROUTER_HISTOGRAM.with(|hist| {
        *hist.borrow_mut() = Some(MoeRouterHistogram::new(num_experts, k_top));
    });
}

pub fn take_moe_router_histogram() -> Option<MoeRouterHistogram> {
    MOE_ROUTER_HISTOGRAM.with(|hist| hist.borrow_mut().take())
}

pub(crate) fn record_moe_router_selection(layer_idx: usize, indices: &[usize], weights: &[f32]) {
    MOE_ROUTER_HISTOGRAM.with(|hist| {
        let mut hist = hist.borrow_mut();
        let Some(hist) = hist.as_mut() else {
            return;
        };
        hist.routed_tokens += 1;
        let num_experts = hist.num_experts;
        let k_top = hist.k_top;
        let mut valid_experts = Vec::with_capacity(k_top);
        let mut layer_updates = Vec::with_capacity(k_top);
        let mut dropped_indices = 0u64;
        for (rank, &expert_idx) in indices.iter().take(k_top).enumerate() {
            if expert_idx >= num_experts {
                hist.dropped_indices += 1;
                dropped_indices += 1;
                continue;
            }
            let weight = weights.get(rank).copied().unwrap_or(0.0);
            if rank == 0 {
                hist.top1_histogram[expert_idx] += 1;
            }
            hist.topk_histogram[expert_idx] += 1;
            hist.routed_slots += 1;
            hist.weight_sums[expert_idx] += weight as f64;
            valid_experts.push(expert_idx);
            layer_updates.push((rank, expert_idx, weight));
        }
        let layer = hist.ensure_layer(layer_idx);
        layer.routed_tokens += 1;
        layer.dropped_indices += dropped_indices;
        for (rank, expert_idx, weight) in layer_updates {
            if rank == 0 {
                layer.top1_histogram[expert_idx] += 1;
            }
            layer.topk_histogram[expert_idx] += 1;
            layer.routed_slots += 1;
            layer.weight_sums[expert_idx] += weight as f64;
        }
        for i in 0..valid_experts.len() {
            for j in (i + 1)..valid_experts.len() {
                let a = valid_experts[i].min(valid_experts[j]);
                let b = valid_experts[i].max(valid_experts[j]);
                let key = (a as u64) * (num_experts as u64) + b as u64;
                *layer.cooccurrence.entry(key).or_insert(0) += 1;
            }
        }
    });
}

pub(crate) fn moe_router_histogram_active() -> bool {
    MOE_ROUTER_HISTOGRAM.with(|hist| hist.borrow().is_some())
}

pub(crate) fn router_index_i32_to_usize(idx: i32) -> usize {
    usize::try_from(idx).unwrap_or(usize::MAX)
}
