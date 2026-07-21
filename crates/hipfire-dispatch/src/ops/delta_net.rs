// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
// DeltaNet linear-attention kernel extensions.
//
// These operations are unique to DeltaNet's linear attention layers:
// the gated linear recurrence with quantized/FP32 state, conv-state ring
// buffer management, and tree-batched speculative-decode variants. They
// don't fit into the standard dispatch families because the state is
// model-owned and the recurrence is an inherently sequential kernel.

use rdna_compute::{Gpu, GpuTensor};

// ── State quantization ─────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum StateQuant {
    FP32,
    Q8,
    Q4,
}

/// Intent controls the FP32 multi-token route. Normal prefill may opt into
/// chunking; speculative replay is always kept on batch-seq semantics.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DeltaNetBatchIntent {
    NormalPrefill,
    SpeculativeReplay,
}

// ── Parameter structs ──────────────────────────────────

/// Parameters for a single-token DeltaNet state update.
///
/// The gated delta net recurrence:
///   S' = gate · S + beta · (k ⊗ v)
///   output = S · q
///
/// where S is the recurrent state (n_heads × head_dim × head_dim),
/// quantized per the `quant` field.
pub struct DeltaNetStepParams<'a> {
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    pub state: &'a GpuTensor,
    pub s_scales: &'a GpuTensor,
    pub output: &'a GpuTensor,
    /// Optional caller-owned f16 error-feedback residual for Q8 state.
    pub ef_residual: Option<&'a GpuTensor>,
    pub n_heads: usize,
    pub head_dim: usize,
    pub quant: StateQuant,
}

/// Parameters for batched sequential DeltaNet updates (prefill path).
///
/// Q, K, V, gate, beta, and output are batched [n_tokens, n_heads, head_dim].
/// The state is updated in-place for all n_tokens.
pub struct DeltaNetBatchParams<'a> {
    pub q_batch: &'a GpuTensor,
    pub k_batch: &'a GpuTensor,
    pub v_batch: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub beta_batch: &'a GpuTensor,
    pub state: &'a GpuTensor,
    pub s_scales: &'a GpuTensor,
    pub output_batch: &'a GpuTensor,
    /// Optional f16 error-feedback residual for the Q8 requantizer. The
    /// residual is caller-owned state and is carried across batch launches.
    pub ef_residual: Option<&'a GpuTensor>,
    pub n_tokens: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub quant: StateQuant,
}

/// Quant-tagged caller-owned state for tree-batched DeltaNet.
///
/// The enum deliberately has no Q4 variant: Q4 tree verify has no tape
/// kernel, so an invalid state/tape combination cannot be constructed.
pub enum DeltaNetTreeState<'a> {
    F32 {
        initial: &'a GpuTensor,
        tape: &'a GpuTensor,
    },
    Q8 {
        initial: &'a GpuTensor,
        scales: &'a GpuTensor,
        tape: &'a GpuTensor,
        tape_scales: &'a GpuTensor,
    },
}

/// Parameters for tree-batched DeltaNet (speculative-decode path).
///
/// `state` owns the quantization tag and all state/tape buffers. The old
/// public shape exposed dummy Q8 fields alongside optional FP32 fields; use
/// [`DeltaNetTreeParams::new`] or the quant-specific constructors instead.
/// `hipfire-dispatch` is a feature-gated, workspace-internal 0.2.x crate, so
/// this source break is intentionally contained here until callers migrate.
pub struct DeltaNetTreeParams<'a> {
    pub q_batch: &'a GpuTensor,
    pub k_batch: &'a GpuTensor,
    pub v_batch: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub beta_batch: &'a GpuTensor,
    pub state: DeltaNetTreeState<'a>,
    pub parent_indices: &'a GpuTensor,
    pub output_batch: &'a GpuTensor,
    pub n_tokens: usize,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl<'a> DeltaNetTreeParams<'a> {
    /// Construct tree parameters from a quant-tagged state/tape bundle.
    // Keep this public constructor's established argument-level API; replacing
    // it with a parameter object would break existing DeltaNet callers.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_batch: &'a GpuTensor,
        k_batch: &'a GpuTensor,
        v_batch: &'a GpuTensor,
        gate_batch: &'a GpuTensor,
        beta_batch: &'a GpuTensor,
        state: DeltaNetTreeState<'a>,
        parent_indices: &'a GpuTensor,
        output_batch: &'a GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            q_batch,
            k_batch,
            v_batch,
            gate_batch,
            beta_batch,
            state,
            parent_indices,
            output_batch,
            n_tokens,
            n_heads,
            head_dim,
        }
    }

    /// Construct FP32 tree parameters without dummy Q8 buffers.
    // Keep this public constructor's established argument-level API; replacing
    // it with a parameter object would break existing DeltaNet callers.
    #[allow(clippy::too_many_arguments)]
    pub fn fp32(
        q_batch: &'a GpuTensor,
        k_batch: &'a GpuTensor,
        v_batch: &'a GpuTensor,
        gate_batch: &'a GpuTensor,
        beta_batch: &'a GpuTensor,
        initial: &'a GpuTensor,
        tape: &'a GpuTensor,
        parent_indices: &'a GpuTensor,
        output_batch: &'a GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self::new(
            q_batch,
            k_batch,
            v_batch,
            gate_batch,
            beta_batch,
            DeltaNetTreeState::F32 { initial, tape },
            parent_indices,
            output_batch,
            n_tokens,
            n_heads,
            head_dim,
        )
    }

    /// Construct Q8 tree parameters without dummy FP32 buffers.
    // Keep this public constructor's established argument-level API; replacing
    // it with a parameter object would break existing DeltaNet callers.
    #[allow(clippy::too_many_arguments)]
    pub fn q8(
        q_batch: &'a GpuTensor,
        k_batch: &'a GpuTensor,
        v_batch: &'a GpuTensor,
        gate_batch: &'a GpuTensor,
        beta_batch: &'a GpuTensor,
        initial: &'a GpuTensor,
        scales: &'a GpuTensor,
        tape: &'a GpuTensor,
        tape_scales: &'a GpuTensor,
        parent_indices: &'a GpuTensor,
        output_batch: &'a GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self::new(
            q_batch,
            k_batch,
            v_batch,
            gate_batch,
            beta_batch,
            DeltaNetTreeState::Q8 {
                initial,
                scales,
                tape,
                tape_scales,
            },
            parent_indices,
            output_batch,
            n_tokens,
            n_heads,
            head_dim,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchRoute {
    F32Chunked,
    F32BatchSeq,
    Q8BatchSeq,
    Q4NToken,
}

fn batch_route(
    quant: StateQuant,
    n_tokens: usize,
    chunked: bool,
    intent: DeltaNetBatchIntent,
) -> BatchRoute {
    match quant {
        StateQuant::FP32
            if matches!(intent, DeltaNetBatchIntent::NormalPrefill) && chunked && n_tokens > 1 =>
        {
            BatchRoute::F32Chunked
        }
        StateQuant::FP32 => BatchRoute::F32BatchSeq,
        StateQuant::Q8 => BatchRoute::Q8BatchSeq,
        StateQuant::Q4 => BatchRoute::Q4NToken,
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TreeRoute {
    FP32,
    Q8,
}

#[cfg(test)]
fn tree_route(quant: StateQuant) -> Result<TreeRoute, &'static str> {
    match quant {
        StateQuant::FP32 => Ok(TreeRoute::FP32),
        StateQuant::Q8 => Ok(TreeRoute::Q8),
        StateQuant::Q4 => Err(
            "Q4 DeltaNet state + tree-verify (DDTree) is unsupported: there is no Q4 tree-tape GDN kernel. Use Q8 or FP32 state for tree spec-decode.",
        ),
    }
}

/// Parameters for DeltaNet conv-state ring-buffer management.
pub struct ConvStateParams<'a> {
    pub state: &'a GpuTensor,
    pub input: &'a GpuTensor,
    pub conv_channels: usize,
    pub kernel_size: usize,
    pub position: usize,
}

// ── Trait ──────────────────────────────────────────────

pub trait DeltaNetOps {
    /// Run a single-token DeltaNet state update.
    ///
    /// Dispatches to `gated_delta_net_f32`, `gated_delta_net_q8`,
    /// or `gated_delta_net_q4` based on `params.quant`.
    fn run_delta_net_step(&self, gpu: &mut Gpu, params: &DeltaNetStepParams) -> Result<(), String>;

    /// Run batched sequential DeltaNet updates (prefill).
    ///
    /// Dispatches to the production FP32 batch/chunked kernels, the Q8
    /// batch-seq kernel (including optional error feedback), or the Q4
    /// N-token kernel.
    fn run_delta_net_batch(
        &self,
        gpu: &mut Gpu,
        params: &DeltaNetBatchParams,
    ) -> Result<(), String>;

    /// Run a batch with explicit prefill/replay route intent. Replay uses
    /// FP32 batch-seq even when normal prefill chunking is enabled.
    fn run_delta_net_batch_with_intent(
        &self,
        gpu: &mut Gpu,
        params: &DeltaNetBatchParams,
        intent: DeltaNetBatchIntent,
    ) -> Result<(), String> {
        // Keep third-party implementations source-compatible. The built-in
        // implementation below overrides this to honor replay semantics.
        let _ = intent;
        self.run_delta_net_batch(gpu, params)
    }

    /// Run tree-batched DeltaNet (speculative-decode path).
    ///
    /// Supports the FP32 and Q8 tree tape kernels. Q4 is refused explicitly;
    /// there is no Q4 tree-tape kernel.
    fn run_delta_net_tree(&self, gpu: &mut Gpu, params: &DeltaNetTreeParams) -> Result<(), String>;

    /// Zero the conv-state ring buffer.
    fn reset_conv_state(
        &self,
        gpu: &mut Gpu,
        state: &GpuTensor,
        conv_state_size: usize,
    ) -> Result<(), String>;
}

// ── Default implementations ────────────────────────────

impl DeltaNetOps for () {
    fn run_delta_net_step(&self, gpu: &mut Gpu, params: &DeltaNetStepParams) -> Result<(), String> {
        match params.quant {
            StateQuant::FP32 => gpu.gated_delta_net_f32(
                params.q,
                params.k,
                params.v,
                params.gate,
                params.beta,
                params.state,
                params.output,
                1,
                params.n_heads,
                params.head_dim,
            ),
            StateQuant::Q8 => gpu.gated_delta_net_q8(
                params.q,
                params.k,
                params.v,
                params.gate,
                params.beta,
                params.state,
                params.s_scales,
                params.output,
                1,
                params.n_heads,
                params.head_dim,
                params.ef_residual,
            ),
            StateQuant::Q4 => gpu.gated_delta_net_q4(
                params.q,
                params.k,
                params.v,
                params.gate,
                params.beta,
                params.state,
                params.s_scales,
                params.output,
                1,
                params.n_heads,
                params.head_dim,
            ),
        }
        .map_err(|e| format!("delta_net_step: {e:?}"))
    }

    fn run_delta_net_batch(
        &self,
        gpu: &mut Gpu,
        params: &DeltaNetBatchParams,
    ) -> Result<(), String> {
        self.run_delta_net_batch_with_intent(gpu, params, DeltaNetBatchIntent::NormalPrefill)
    }

    fn run_delta_net_batch_with_intent(
        &self,
        gpu: &mut Gpu,
        params: &DeltaNetBatchParams,
        intent: DeltaNetBatchIntent,
    ) -> Result<(), String> {
        match batch_route(
            params.quant,
            params.n_tokens,
            rdna_compute::norm::gdn_chunked(),
            intent,
        ) {
            BatchRoute::F32Chunked => gpu
                .gated_delta_net_f32_chunked(
                    params.q_batch,
                    params.k_batch,
                    params.v_batch,
                    params.gate_batch,
                    params.beta_batch,
                    params.state,
                    params.output_batch,
                    params.n_tokens,
                    params.n_heads,
                    params.head_dim,
                    rdna_compute::norm::gdn_chunk_size(),
                )
                .map_err(|e| format!("delta_net_batch: {e:?}")),
            BatchRoute::F32BatchSeq => gpu
                .gated_delta_net_f32_batch_seq(
                    params.q_batch,
                    params.k_batch,
                    params.v_batch,
                    params.gate_batch,
                    params.beta_batch,
                    params.state,
                    params.output_batch,
                    params.n_tokens,
                    params.n_heads,
                    params.head_dim,
                )
                .map_err(|e| format!("delta_net_batch: {e:?}")),
            BatchRoute::Q8BatchSeq => gpu
                .gated_delta_net_q8_batch_seq(
                    params.q_batch,
                    params.k_batch,
                    params.v_batch,
                    params.gate_batch,
                    params.beta_batch,
                    params.state,
                    params.s_scales,
                    params.output_batch,
                    params.n_tokens,
                    params.n_heads,
                    params.head_dim,
                    params.ef_residual,
                )
                .map_err(|e| format!("delta_net_batch: {e:?}")),
            BatchRoute::Q4NToken => gpu
                .gated_delta_net_q4(
                    params.q_batch,
                    params.k_batch,
                    params.v_batch,
                    params.gate_batch,
                    params.beta_batch,
                    params.state,
                    params.s_scales,
                    params.output_batch,
                    params.n_tokens,
                    params.n_heads,
                    params.head_dim,
                )
                .map_err(|e| format!("delta_net_batch: {e:?}")),
        }
    }

    fn run_delta_net_tree(&self, gpu: &mut Gpu, params: &DeltaNetTreeParams) -> Result<(), String> {
        match &params.state {
            DeltaNetTreeState::F32 { initial, tape } => gpu
                .gated_delta_net_f32_tree_batch_seq(
                    params.q_batch,
                    params.k_batch,
                    params.v_batch,
                    params.gate_batch,
                    params.beta_batch,
                    initial,
                    tape,
                    params.parent_indices,
                    params.output_batch,
                    params.n_tokens,
                    params.n_heads,
                    params.head_dim,
                )
                .map_err(|e| format!("delta_net_tree: {e:?}")),
            DeltaNetTreeState::Q8 {
                initial,
                scales,
                tape,
                tape_scales,
            } => gpu
                .gated_delta_net_q8_tree_batch_seq(
                    params.q_batch,
                    params.k_batch,
                    params.v_batch,
                    params.gate_batch,
                    params.beta_batch,
                    initial,
                    scales,
                    tape,
                    tape_scales,
                    params.parent_indices,
                    params.output_batch,
                    params.n_tokens,
                    params.n_heads,
                    params.head_dim,
                )
                .map_err(|e| format!("delta_net_tree: {e:?}")),
        }
    }

    fn reset_conv_state(
        &self,
        gpu: &mut Gpu,
        state: &GpuTensor,
        _conv_state_size: usize,
    ) -> Result<(), String> {
        gpu.hip
            .memset(&state.buf, 0, state.buf.size())
            .map_err(|e| format!("reset_conv_state: {e:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::{batch_route, tree_route, BatchRoute, DeltaNetBatchIntent, StateQuant, TreeRoute};

    #[test]
    fn fp32_batch_route_selects_chunked_only_for_multi_token_prefill() {
        assert_eq!(
            batch_route(
                StateQuant::FP32,
                8,
                true,
                DeltaNetBatchIntent::NormalPrefill,
            ),
            BatchRoute::F32Chunked
        );
        assert_eq!(
            batch_route(
                StateQuant::FP32,
                1,
                true,
                DeltaNetBatchIntent::NormalPrefill,
            ),
            BatchRoute::F32BatchSeq
        );
        assert_eq!(
            batch_route(
                StateQuant::FP32,
                8,
                false,
                DeltaNetBatchIntent::NormalPrefill,
            ),
            BatchRoute::F32BatchSeq
        );
        assert_eq!(
            batch_route(
                StateQuant::FP32,
                8,
                true,
                DeltaNetBatchIntent::SpeculativeReplay,
            ),
            BatchRoute::F32BatchSeq
        );
    }

    #[test]
    fn q8_and_q4_batch_routes_preserve_n_token_kernel_contract() {
        assert_eq!(
            batch_route(
                StateQuant::Q8,
                8,
                false,
                DeltaNetBatchIntent::SpeculativeReplay,
            ),
            BatchRoute::Q8BatchSeq
        );
        assert_eq!(
            batch_route(StateQuant::Q4, 8, false, DeltaNetBatchIntent::NormalPrefill,),
            BatchRoute::Q4NToken
        );
    }

    #[test]
    fn tree_route_accepts_fp32_and_q8_but_refuses_q4() {
        assert_eq!(tree_route(StateQuant::FP32), Ok(TreeRoute::FP32));
        assert_eq!(tree_route(StateQuant::Q8), Ok(TreeRoute::Q8));
        assert_eq!(
            tree_route(StateQuant::Q4),
            Err("Q4 DeltaNet state + tree-verify (DDTree) is unsupported: there is no Q4 tree-tape GDN kernel. Use Q8 or FP32 state for tree spec-decode.")
        );
    }

    #[test]
    fn delta_net_ops_is_implemented_without_a_model_caller() {
        fn needs_delta_net_ops<T: super::DeltaNetOps>() {}
        needs_delta_net_ops::<()>();
    }
}
