// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! VRAM/host memory accounting for a loaded model — artifact bytes plus
//! per-arch runtime scratch/KV/state byte tallies feeding the worker memory
//! view. Extracted verbatim from the former `main.rs` monolith (no behavior
//! change); helpers are `pub(crate)`.

use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::qwen35::DeltaNetState;
use hipfire_runtime::llama;
use hipfire_state::{ModelArtifactMemory, ModelWorkerMemoryView, SequenceStatePageDescriptor};

use crate::model::LoadedModel;

/// Artifact memory for a loaded HFQ: on-disk file size plus the summed
/// tensor-data bytes (resident weight footprint).
pub(crate) fn hfq_model_memory(
    path: &str,
    hfq: &hipfire_runtime::hfq::HfqFile,
) -> ModelArtifactMemory {
    ModelArtifactMemory {
        model_file_bytes: std::fs::metadata(path)
            .map(|metadata| metadata.len() as usize)
            .unwrap_or(0),
        model_weight_bytes: hfq
            .tensors()
            .iter()
            .map(|tensor| tensor.data_size)
            .sum::<usize>(),
    }
}

/// Artifact memory for a non-HFQ / unparsed model: file size only, weight bytes
/// reported as 0 (unknown) rather than guessed.
pub(crate) fn unknown_model_memory(path: &str) -> ModelArtifactMemory {
    ModelArtifactMemory {
        model_file_bytes: std::fs::metadata(path)
            .map(|metadata| metadata.len() as usize)
            .unwrap_or(0),
        model_weight_bytes: 0,
    }
}

/// Device-buffer byte size of one GPU tensor.
pub(crate) fn tensor_bytes(tensor: &rdna_compute::GpuTensor) -> usize {
    tensor.buf.size()
}

/// [`tensor_bytes`] for an optional tensor; 0 when `None`.
pub(crate) fn opt_tensor_bytes(tensor: Option<&rdna_compute::GpuTensor>) -> usize {
    tensor.map(tensor_bytes).unwrap_or(0)
}

/// Summed [`tensor_bytes`] over a tensor slice (e.g. per-layer KV vectors).
pub(crate) fn tensor_vec_bytes(tensors: &[rdna_compute::GpuTensor]) -> usize {
    tensors.iter().map(tensor_bytes).sum::<usize>()
}

/// Resident bytes of a KV cache: K/V tensors, their quant scales, and the
/// optional cached RoPE givens.
pub(crate) fn kv_cache_bytes(kv: &llama::KvCache) -> usize {
    tensor_vec_bytes(&kv.k_gpu)
        + tensor_vec_bytes(&kv.v_gpu)
        + tensor_vec_bytes(&kv.k_scales)
        + tensor_vec_bytes(&kv.v_scales)
        + opt_tensor_bytes(kv.givens_cos.as_ref())
        + opt_tensor_bytes(kv.givens_sin.as_ref())
}

/// Resident bytes of the DeltaNet linear-attention state: S-matrices, their
/// scales, and the short-conv states.
pub(crate) fn deltanet_state_bytes(dn: &DeltaNetState) -> usize {
    tensor_vec_bytes(&dn.s_matrices)
        + tensor_vec_bytes(&dn.s_scales)
        + tensor_vec_bytes(&dn.conv_states)
}

/// Resident bytes of the Qwen3.5 forward scratch: attention/DeltaNet/FlashAttn
/// working buffers, the FFN buffers, and the optional grouped-MoE scratch.
/// (`PrefillBatchScratch` has private fields and is reported as unknown / 0.)
pub(crate) fn qwen35_scratch_bytes(scratch: &qwen35::Qwen35Scratch) -> usize {
    let mut total = scratch.pos_buf.size();
    total += tensor_bytes(&scratch.x)
        + tensor_bytes(&scratch.tmp)
        + tensor_bytes(&scratch.dn_qkv)
        + tensor_bytes(&scratch.dn_z)
        + tensor_bytes(&scratch.dn_alpha)
        + tensor_bytes(&scratch.dn_beta)
        + tensor_bytes(&scratch.dn_conv_out)
        + tensor_bytes(&scratch.dn_q)
        + tensor_bytes(&scratch.dn_k)
        + tensor_bytes(&scratch.dn_v)
        + tensor_bytes(&scratch.dn_q_raw)
        + tensor_bytes(&scratch.dn_k_raw)
        + tensor_bytes(&scratch.dn_attn_out)
        + tensor_bytes(&scratch.dn_normed)
        + tensor_bytes(&scratch.fa_q_full)
        + tensor_bytes(&scratch.fa_q)
        + tensor_bytes(&scratch.fa_gate)
        + tensor_bytes(&scratch.fa_k)
        + tensor_bytes(&scratch.fa_v)
        + tensor_bytes(&scratch.fa_attn_out)
        + tensor_bytes(&scratch.o)
        + tensor_bytes(&scratch.gate_ffn)
        + tensor_bytes(&scratch.up)
        + tensor_bytes(&scratch.ffn_hidden)
        + tensor_bytes(&scratch.ffn_out)
        + tensor_bytes(&scratch.logits)
        + tensor_bytes(&scratch.sample_buf)
        + tensor_bytes(&scratch.repeat_buf)
        + tensor_bytes(&scratch.x_rot)
        + tensor_bytes(&scratch.flash_partials);
    total += opt_tensor_bytes(scratch.moe_router_logits.as_ref())
        + opt_tensor_bytes(scratch.moe_scalar_buf.as_ref())
        + opt_tensor_bytes(scratch.moe_x_rot.as_ref())
        + opt_tensor_bytes(scratch.moe_gate_up_buf.as_ref())
        + opt_tensor_bytes(scratch.moe_gate_buf.as_ref())
        + opt_tensor_bytes(scratch.moe_up_buf.as_ref())
        + opt_tensor_bytes(scratch.moe_ffn_hidden.as_ref())
        + opt_tensor_bytes(scratch.moe_ffn_out.as_ref())
        + opt_tensor_bytes(scratch.moe_gate_batch.as_ref())
        + opt_tensor_bytes(scratch.moe_up_batch.as_ref())
        + opt_tensor_bytes(scratch.moe_rot_batch.as_ref())
        + opt_tensor_bytes(scratch.moe_topk_indices.as_ref())
        + opt_tensor_bytes(scratch.moe_topk_weights.as_ref())
        + opt_tensor_bytes(scratch.moe_down_expanded.as_ref());
    // PrefillBatchScratch is an optional optimization scratch with private
    // fields. Report it as unknown in V1 instead of inventing an estimate.
    total
}

/// Resident bytes of the Qwen2 decode state, including its in-struct K/V cache.
pub(crate) fn qwen2_state_bytes(state: &qwen2::Qwen2State) -> usize {
    state.pos_buf.size()
        + tensor_bytes(&state.x)
        + tensor_bytes(&state.tmp)
        + tensor_bytes(&state.q)
        + tensor_bytes(&state.k)
        + tensor_bytes(&state.v)
        + tensor_bytes(&state.attn_out)
        + tensor_bytes(&state.o)
        + tensor_bytes(&state.gate)
        + tensor_bytes(&state.up)
        + tensor_bytes(&state.ffn_hidden)
        + tensor_bytes(&state.ffn_out)
        + tensor_bytes(&state.logits)
        + tensor_bytes(&state.attn_partials)
        + tensor_vec_bytes(&state.k_cache)
        + tensor_vec_bytes(&state.v_cache)
}

/// Resident bytes of the LLaMA/Qwen3 forward scratch (attention + FFN working
/// buffers). The KV cache is separate ([`kv_cache_bytes`]).
pub(crate) fn llama_scratch_bytes(scratch: &llama::ForwardScratch) -> usize {
    scratch.pos_buf.size()
        + tensor_bytes(&scratch.x)
        + tensor_bytes(&scratch.tmp)
        + tensor_bytes(&scratch.q)
        + tensor_bytes(&scratch.k)
        + tensor_bytes(&scratch.v)
        + tensor_bytes(&scratch.attn_out)
        + tensor_bytes(&scratch.o)
        + tensor_bytes(&scratch.gate)
        + tensor_bytes(&scratch.up)
        + tensor_bytes(&scratch.ffn_hidden)
        + tensor_bytes(&scratch.ffn_out)
        + tensor_bytes(&scratch.logits)
        + tensor_bytes(&scratch.sample_buf)
        + tensor_bytes(&scratch.repeat_buf)
        + tensor_bytes(&scratch.attn_partials)
        + tensor_bytes(&scratch.x_rot)
}

/// Resident bytes of the MiniMax-M2 decode state: its KV cache plus the
/// attention/FlashAttn, FFN, and MoE routing working buffers.
pub(crate) fn minimax_state_bytes(state: &hipfire_arch_minimax::MiniMaxState) -> usize {
    state.pos_buf.size()
        + kv_cache_bytes(&state.kv)
        + tensor_bytes(&state.tmp)
        + tensor_bytes(&state.x_rot)
        + tensor_bytes(&state.fa_q)
        + tensor_bytes(&state.fa_k)
        + tensor_bytes(&state.fa_v)
        + tensor_bytes(&state.fa_attn_out)
        + tensor_bytes(&state.flash_partials)
        + tensor_bytes(&state.h)
        + tensor_bytes(&state.ffn_tmp)
        + tensor_bytes(&state.ffn_x_rot)
        + tensor_bytes(&state.router_logits)
        + tensor_bytes(&state.topk_indices)
        + tensor_bytes(&state.topk_weights)
        + tensor_bytes(&state.gate_batch)
        + tensor_bytes(&state.up_batch)
        + tensor_bytes(&state.rot_batch)
        + tensor_bytes(&state.down_expanded)
        + tensor_bytes(&state.final_norm_buf)
        + tensor_bytes(&state.final_rot)
        + tensor_bytes(&state.logits)
}

/// Total resident runtime bytes for a loaded model (excludes weights/artifact):
/// sums whichever arch's KV cache, DeltaNet state, and forward scratch are
/// populated — the per-arch Option fields make this a simple `map().unwrap_or(0)`
/// tally.
pub(crate) fn loaded_model_runtime_base_bytes(m: &LoadedModel) -> usize {
    let mut total = 0usize;
    total += m.kv_cache.as_ref().map(kv_cache_bytes).unwrap_or(0);
    total += m.dn_state.as_ref().map(deltanet_state_bytes).unwrap_or(0);
    total += m
        .q35_scratch
        .as_ref()
        .map(qwen35_scratch_bytes)
        .unwrap_or(0);
    total += m
        .pp_scratch_set
        .as_ref()
        .map(|set| {
            set.per_device
                .iter()
                .map(qwen35_scratch_bytes)
                .sum::<usize>()
        })
        .unwrap_or(0);
    total += m.qwen2_state.as_ref().map(qwen2_state_bytes).unwrap_or(0);
    total += m.llama_kv.as_ref().map(kv_cache_bytes).unwrap_or(0);
    total += m
        .llama_scratch
        .as_ref()
        .map(llama_scratch_bytes)
        .unwrap_or(0);
    total += m
        .minimax_state
        .as_ref()
        .map(minimax_state_bytes)
        .unwrap_or(0);
    total
}

/// Assemble the client-facing [`ModelWorkerMemoryView`]: artifact bytes plus the
/// runtime base ([`loaded_model_runtime_base_bytes`]) and the per-session
/// resident bytes summed from the supplied state-page descriptors.
pub(crate) fn loaded_model_memory_view(
    m: &LoadedModel,
    state_page_descriptors: &[SequenceStatePageDescriptor],
) -> ModelWorkerMemoryView {
    let runtime_base_bytes = loaded_model_runtime_base_bytes(m);
    let runtime_session_bytes = state_page_descriptors
        .iter()
        .map(|descriptor| descriptor.resident_bytes)
        .sum::<usize>();
    m.memory
        .worker_memory_view(runtime_base_bytes, runtime_session_bytes)
}
