// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 batched prefill subsystem: `PrefillBatchScratch`, the DDTree
//! session-batch structures + pointer-table planning + per-layer session-batch
//! ops, and the `forward_prefill_batch*` entry points. Sits above
//! `forward_prefill_chunk` (in prefill_chunk.rs) on the prefill path.

use super::*;

/// Per-layer batched intermediates used by `forward_prefill_batch`. Each
/// row is one token in the batch; rows are contiguous [N × K] blocks so
/// all kernels can treat them as row-major matrices.
///
/// Allocated lazily on the first batched prefill call that takes the MQ4
/// fast path — models that never hit that path (HF4 weights, FA-only
/// models, short prompts) never pay the VRAM cost. Sized to `max_batch`;
/// longer prompts are processed in chunks of `max_batch`.
pub struct PrefillBatchScratch {
    pub max_batch: usize,

    // Residual stream and rotation scratch — all [N × dim]
    pub x_batch: GpuTensor,
    pub x_rot_batch: GpuTensor,
    // Rmsnorm-only scratch (no FWHT). Used by MoE prefill body for Q8_0
    // weights (router + shared_expert_gate) which were quantized against
    // un-rotated input. MQ4 sibling weights read `x_rot_batch` instead.
    // Mixed-dtype MoE layers populate both buffers per `prefill_moe_ffn_body_batched`.
    pub x_norm_batch: GpuTensor,

    // LA-layer projection outputs
    pub dn_qkv_batch: GpuTensor,      // [N × qkv_dim]
    pub dn_z_batch: GpuTensor,        // [N × v_dim]
    pub dn_alpha_batch: GpuTensor,    // [N × n_v_heads]
    pub dn_beta_batch: GpuTensor,     // [N × n_v_heads]
    pub dn_q_raw_batch: GpuTensor,    // [N × k_dim] (pre repeat-interleave)
    pub dn_k_raw_batch: GpuTensor,    // [N × k_dim]
    pub dn_v_batch: GpuTensor,        // [N × v_dim]
    pub dn_q_batch: GpuTensor,        // [N × v_dim] (post repeat-interleave)
    pub dn_k_batch: GpuTensor,        // [N × v_dim]
    pub dn_attn_out_batch: GpuTensor, // [N × v_dim]
    pub dn_normed_batch: GpuTensor,   // [N × v_dim]

    // FFN intermediates [N × hidden_dim]
    pub gate_ffn_batch: GpuTensor,
    pub up_batch: GpuTensor,
    // SwiGLU output (FWHT-rotated for MQ4) feeding w_down.
    pub ffn_hidden_batch: GpuTensor,

    // FWHT-rotated dn_normed [N × v_dim] feeding wo for MQ4 weights.
    // Decode path handles this via an internal mq_x_rot scratch inside
    // weight_gemv_residual; we need an explicit batched equivalent.
    pub dn_normed_rot_batch: GpuTensor,

    // ── FullAttention batched intermediates (when FA weights are MQ4G256) ──
    // Positions array: [max_batch] i32, absolute KV positions for this chunk.
    // Uploaded once at the start of each chunk and reused by rope + kv_write
    // + attention kernels.
    pub positions: GpuTensor,
    // Token-ids buffer feeding the batched embedding kernel. [max_batch] i32
    // stored as F32 (same dtype-cosmetic pattern as `positions`). Uploaded
    // once per batched forward and read by `embedding_lookup_hfq4g256_batched`.
    pub tokens: GpuTensor,
    // QKV projection outputs
    pub fa_q_full_batch: GpuTensor, // [N × n_heads × head_dim × 2] (Q + gate interleaved)
    pub fa_q_batch: GpuTensor,      // [N × n_heads × head_dim]
    pub fa_gate_batch: GpuTensor,   // [N × n_heads × head_dim]
    pub fa_k_batch: GpuTensor,      // [N × n_kv_heads × head_dim]
    pub fa_v_batch: GpuTensor,      // [N × n_kv_heads × head_dim]
    pub fa_attn_out_batch: GpuTensor, // [N × n_heads × head_dim]
    // FWHT-rotated fa_attn_out for feeding MQ4 wo.
    pub fa_attn_out_rot_batch: GpuTensor, // [N × n_heads × head_dim]

    // ── MoE batched intermediates (allocated only when num_experts > 0) ──
    // All outputs of the fused 4-way router + shared-gate GEMM, plus the
    // per-token routed-expert gate/up/rot buffers consumed by the N-batched
    // indexed MoE kernels. Sized as [max_batch × {n_exp, smi, k_top×mi}].
    pub moe_router_logits_batch: Option<GpuTensor>, // [N × num_experts]
    pub moe_shared_scalar_batch: Option<GpuTensor>, // [N × 1] — raw shared_expert_gate logit
    pub moe_shared_gate_batch: Option<GpuTensor>,   // [N × smi]
    pub moe_shared_up_batch: Option<GpuTensor>,     // [N × smi]
    pub moe_shared_rot_batch: Option<GpuTensor>,    // [N × smi] — FWHT(silu(gate) * up)
    pub moe_topk_indices_batch: Option<GpuTensor>,  // [N × k_top] i32 in F32 slots
    pub moe_topk_weights_batch: Option<GpuTensor>,  // [N × k_top]
    pub moe_gate_batch: Option<GpuTensor>,          // [N × k_top × mi]
    pub moe_up_batch: Option<GpuTensor>,            // [N × k_top × mi]
    pub moe_rot_batch: Option<GpuTensor>,           // [N × k_top × mi]
    // Atomic-free MoE down expansion buffer — [N × k_top × dim] f32.
    // Paired with `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` +
    // `moe_down_combine_k8_batched`: the down kernel writes each
    // (token, krank) result to its own row here (no atomic), then the
    // combine kernel folds K_TOP slots into x_batch with topk_weights
    // applied. RDNA-only (atomic on GDDR is slow); the wave64/CDNA path
    // stays on the residual_scaled atomic kernel.
    pub moe_down_expanded_batch: Option<GpuTensor>,

    // Path 2 (SGLang-style scatter + grouped-WMMA-GEMM) scratch. All
    // allocated when num_experts > 0; gated at runtime by
    // HIPFIRE_MOE_GROUPED_GEMM=1. m_total_max is tile-aligned:
    // align_up(max_batch * k_top + num_experts * (BLOCK_M - 1), BLOCK_M)
    // with BLOCK_M=16.
    //
    //   moe_expert_token_counts: [num_experts] i32 (raw → padded)
    //   moe_expert_offsets:      [num_experts + 1] i32 (exclusive prefix)
    //   moe_sorted_slot_index:   [m_total_max] i32 (flat slot or -1 padding)
    //   moe_expert_tile_ids:     [m_total_max / 16] i32 (per-tile expert id)
    //   moe_y_gate_up_grouped:   [m_total_max × (2*mi)] f32 (grouped GEMM output)
    pub moe_expert_token_counts: Option<GpuTensor>,
    pub moe_expert_offsets: Option<GpuTensor>,
    pub moe_sorted_slot_index: Option<GpuTensor>,
    pub moe_inverse_perm: Option<GpuTensor>, // [total_slots] i32: flat → sorted_pos
    pub moe_expert_tile_ids: Option<GpuTensor>,
    pub moe_y_gate_up_grouped: Option<GpuTensor>, // [m_total × (2*mi)]
    pub moe_y_down_grouped: Option<GpuTensor>,    // [m_total × dim] for the down step

    // ── Tree-aware LA scratch (Phase 3b of Task #101) ──
    // Per-token S-state tape consumed by gated_delta_net_q8_tree kernel
    // when TreeVerifyCtx.parent_indices is Some. Reused across LA layers
    // since LA dispatch is serial per-cycle. Only allocated when the model
    // has LA layers (linear_num_value_heads > 0). Call sites that pass
    // parent_indices must ensure these tensors exist.
    //
    // s_tape_q8:     [max_batch × n_v_heads × head_dim × head_dim] Raw/i8
    // s_tape_scales: [max_batch × n_v_heads × head_dim] f32
    //
    // At max_batch=22, n_v_heads=16, head_dim=128 → 5.77 MB + 180 KB total.
    pub dn_s_tape_q8: Option<GpuTensor>,
    pub dn_s_tape_scales: Option<GpuTensor>,
}

/// One independent dense-Qwen35 request/session row for the future fused
/// server-prefill worker.
///
/// This is intentionally NOT the same shape as `forward_prefill_batch`: that
/// function consumes one token stream, one KV cache, and one DeltaNet state.
/// Server microbatching needs multiple independent streams, each with its own
/// mutable recurrent state, while sharing weights and batched layer scratch.
pub struct DensePrefillSessionBatchRow<'a> {
    pub tokens: &'a [u32],
    pub start_pos: usize,
    pub kv_cache: &'a mut kv::KvCache,
    pub dn_state: &'a mut DeltaNetState,
    pub logits: &'a GpuTensor,
}

pub struct DensePrefillSessionBatchInput<'a> {
    pub tokens: &'a [u32],
    pub start_pos: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchRoundRow {
    pub session_index: usize,
    pub token_index: usize,
    pub token: u32,
    pub position: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePrefillSessionBatchRound {
    pub rows: Vec<DensePrefillSessionBatchRoundRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DensePrefillSessionBatchRoundStateRoute {
    SingleSession { session_index: usize },
    MultiSession { session_indices: Vec<usize> },
}

impl DensePrefillSessionBatchRound {
    pub fn state_route(&self) -> DensePrefillSessionBatchRoundStateRoute {
        let mut session_indices: Vec<usize> =
            self.rows.iter().map(|row| row.session_index).collect();
        session_indices.sort_unstable();
        session_indices.dedup();
        if session_indices.len() == 1 {
            DensePrefillSessionBatchRoundStateRoute::SingleSession {
                session_index: session_indices[0],
            }
        } else {
            DensePrefillSessionBatchRoundStateRoute::MultiSession { session_indices }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePrefillSessionBatchExecutionPlan {
    pub rounds: Vec<DensePrefillSessionBatchRound>,
    pub state_routes: Vec<DensePrefillSessionBatchRoundStateRoute>,
    pub total_rows: usize,
    pub max_rows_per_round: usize,
    pub multi_state_rounds: usize,
    pub multi_state_prefix_rounds: usize,
    pub multi_state_prefix_rows: usize,
    pub singleton_tail: Option<DensePrefillSessionBatchSingletonTail>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchSingletonTail {
    pub start_round: usize,
    pub session_index: usize,
    pub rows: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchRowShape {
    pub tokens: usize,
    pub logits_numel: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DensePrefillSessionBatchStateSignature {
    pub kv_physical_cap: usize,
    pub kv_compact_offset: usize,
    pub kv_quantized: bool,
    pub kv_quant_q8: bool,
    pub kv_quant_asym2: bool,
    pub kv_quant_asym3: bool,
    pub kv_quant_asym4: bool,
    pub kv_quant_fwht: bool,
    pub dn_quant: StateQuant,
}

pub struct DensePrefillSessionKvStateRoute<'a> {
    pub k_gpu: &'a [GpuTensor],
    pub v_gpu: &'a [GpuTensor],
    pub physical_cap: usize,
    pub compact_offset: usize,
}

pub struct DensePrefillSessionDeltaStateRoute<'a> {
    pub s_matrices: &'a [GpuTensor],
    pub s_scales: &'a [GpuTensor],
    pub conv_states: &'a [GpuTensor],
    pub quant: StateQuant,
}

pub struct DensePrefillSessionStateRoute<'a> {
    pub kv: DensePrefillSessionKvStateRoute<'a>,
    pub delta: DensePrefillSessionDeltaStateRoute<'a>,
    pub logits: &'a GpuTensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionStateRouteShape {
    pub kv_k_layers: usize,
    pub kv_v_layers: usize,
    pub dn_s_layers: usize,
    pub dn_scale_layers: usize,
    pub dn_conv_layers: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchPointerTableShape {
    pub sessions: usize,
    pub multi_state_prefix_rounds: usize,
    pub multi_state_prefix_rows: usize,
    pub max_rows_per_round: usize,
    pub kv_k_ptrs: usize,
    pub kv_v_ptrs: usize,
    pub dn_s_ptrs: usize,
    pub dn_scale_ptrs: usize,
    pub dn_conv_ptrs: usize,
    pub logits_ptrs: usize,
    pub session_last_row_indices: usize,
    pub row_session_indices: usize,
    pub row_tokens: usize,
    pub row_positions: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchPointerTableIndex {
    pub kv_k_offset: usize,
    pub kv_v_offset: usize,
    pub dn_s_offset: usize,
    pub dn_scale_offset: usize,
    pub dn_conv_offset: usize,
    pub logits_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchLayerPointerSlot {
    pub session_index: usize,
    pub layer_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchDeltaPointerSlot {
    pub session_index: usize,
    pub delta_layer_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchPrefixRowSlot {
    pub round_index: usize,
    pub round_row_index: usize,
    pub session_index: usize,
    pub token_index: usize,
    pub token: u32,
    pub position: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePrefillSessionBatchPointerTablePlan {
    pub shape: DensePrefillSessionBatchPointerTableShape,
    pub kv_layer_slots: Vec<DensePrefillSessionBatchLayerPointerSlot>,
    pub dn_layer_slots: Vec<DensePrefillSessionBatchDeltaPointerSlot>,
    pub logits_slots: Vec<usize>,
    pub prefix_rows: Vec<DensePrefillSessionBatchPrefixRowSlot>,
    pub session_last_row_indices: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePrefillSessionBatchHostPointerTables {
    pub kv_k_ptrs: Vec<u64>,
    pub kv_v_ptrs: Vec<u64>,
    pub dn_s_ptrs: Vec<u64>,
    pub dn_scale_ptrs: Vec<u64>,
    pub dn_conv_ptrs: Vec<u64>,
    pub logits_ptrs: Vec<u64>,
    pub session_last_row_indices: Vec<i32>,
    pub row_session_indices: Vec<i32>,
    pub row_tokens: Vec<i32>,
    pub row_positions: Vec<i32>,
}

pub struct DensePrefillSessionBatchDevicePointerTables {
    pub kv_k_ptrs: GpuTensor,
    pub kv_v_ptrs: GpuTensor,
    pub dn_s_ptrs: GpuTensor,
    pub dn_scale_ptrs: GpuTensor,
    pub dn_conv_ptrs: GpuTensor,
    pub logits_ptrs: GpuTensor,
    pub session_last_row_indices: GpuTensor,
    pub row_session_indices: GpuTensor,
    pub row_tokens: GpuTensor,
    pub row_positions: GpuTensor,
}

impl DensePrefillSessionBatchHostPointerTables {
    pub fn validate_shape(
        &self,
        shape: DensePrefillSessionBatchPointerTableShape,
    ) -> Result<(), String> {
        let checks = [
            ("kv_k_ptrs", self.kv_k_ptrs.len(), shape.kv_k_ptrs),
            ("kv_v_ptrs", self.kv_v_ptrs.len(), shape.kv_v_ptrs),
            ("dn_s_ptrs", self.dn_s_ptrs.len(), shape.dn_s_ptrs),
            (
                "dn_scale_ptrs",
                self.dn_scale_ptrs.len(),
                shape.dn_scale_ptrs,
            ),
            ("dn_conv_ptrs", self.dn_conv_ptrs.len(), shape.dn_conv_ptrs),
            ("logits_ptrs", self.logits_ptrs.len(), shape.logits_ptrs),
            (
                "session_last_row_indices",
                self.session_last_row_indices.len(),
                shape.session_last_row_indices,
            ),
            (
                "row_session_indices",
                self.row_session_indices.len(),
                shape.row_session_indices,
            ),
            ("row_tokens", self.row_tokens.len(), shape.row_tokens),
            (
                "row_positions",
                self.row_positions.len(),
                shape.row_positions,
            ),
        ];
        for (name, got, expected) in checks {
            if got != expected {
                return Err(format!(
                    "dense session prefill host pointer table {name} has {got} entries, expected {expected}",
                ));
            }
        }
        Ok(())
    }
}

impl DensePrefillSessionBatchDevicePointerTables {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.kv_k_ptrs);
        let _ = gpu.free_tensor(self.kv_v_ptrs);
        let _ = gpu.free_tensor(self.dn_s_ptrs);
        let _ = gpu.free_tensor(self.dn_scale_ptrs);
        let _ = gpu.free_tensor(self.dn_conv_ptrs);
        let _ = gpu.free_tensor(self.logits_ptrs);
        let _ = gpu.free_tensor(self.session_last_row_indices);
        let _ = gpu.free_tensor(self.row_session_indices);
        let _ = gpu.free_tensor(self.row_tokens);
        let _ = gpu.free_tensor(self.row_positions);
    }
}

fn u64_slice_as_bytes(values: &[u64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 8) }
}

pub(crate) fn i32_slice_as_bytes(values: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
}

fn alloc_and_upload_u64_table(gpu: &mut Gpu, values: &[u64]) -> HipResult<GpuTensor> {
    // F32 dtype is cosmetic here: it gives a 4-byte element size, so two
    // elements hold one raw device pointer. Kernels consume these buffers as
    // `const uint64_t*`.
    let tensor = gpu.alloc_tensor(&[values.len() * 2], DType::F32)?;
    gpu.hip
        .memcpy_htod(&tensor.buf, u64_slice_as_bytes(values))?;
    Ok(tensor)
}

fn alloc_and_upload_i32_table(gpu: &mut Gpu, values: &[i32]) -> HipResult<GpuTensor> {
    // F32 dtype is cosmetic here: the row-routing kernels consume these
    // buffers as `const int*`.
    let tensor = gpu.alloc_tensor(&[values.len()], DType::F32)?;
    gpu.hip
        .memcpy_htod(&tensor.buf, i32_slice_as_bytes(values))?;
    Ok(tensor)
}

pub fn upload_dense_prefill_session_batch_pointer_tables(
    gpu: &mut Gpu,
    shape: DensePrefillSessionBatchPointerTableShape,
    host: &DensePrefillSessionBatchHostPointerTables,
) -> HipResult<DensePrefillSessionBatchDevicePointerTables> {
    host.validate_shape(shape)
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    Ok(DensePrefillSessionBatchDevicePointerTables {
        kv_k_ptrs: alloc_and_upload_u64_table(gpu, &host.kv_k_ptrs)?,
        kv_v_ptrs: alloc_and_upload_u64_table(gpu, &host.kv_v_ptrs)?,
        dn_s_ptrs: alloc_and_upload_u64_table(gpu, &host.dn_s_ptrs)?,
        dn_scale_ptrs: alloc_and_upload_u64_table(gpu, &host.dn_scale_ptrs)?,
        dn_conv_ptrs: alloc_and_upload_u64_table(gpu, &host.dn_conv_ptrs)?,
        logits_ptrs: alloc_and_upload_u64_table(gpu, &host.logits_ptrs)?,
        session_last_row_indices: alloc_and_upload_i32_table(gpu, &host.session_last_row_indices)?,
        row_session_indices: alloc_and_upload_i32_table(gpu, &host.row_session_indices)?,
        row_tokens: alloc_and_upload_i32_table(gpu, &host.row_tokens)?,
        row_positions: alloc_and_upload_i32_table(gpu, &host.row_positions)?,
    })
}

pub fn dense_prefill_session_batch_write_f32_kv_layer(
    gpu: &mut Gpu,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    kv_layer_index: usize,
    k_src: &GpuTensor,
    v_src: &GpuTensor,
    kv_dim: usize,
    row_count: usize,
) -> HipResult<()> {
    if kv_layer_index >= route_shape.kv_k_layers || kv_layer_index >= route_shape.kv_v_layers {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "dense session prefill routed KV write layer {kv_layer_index} out of range for route shape {:?}",
                route_shape,
            ),
        ));
    }
    gpu.kv_cache_write_f32_routed_batched(
        &device_tables.kv_k_ptrs,
        k_src,
        &device_tables.row_session_indices,
        &device_tables.row_positions,
        route_shape.kv_k_layers,
        kv_layer_index,
        kv_dim,
        row_count,
    )?;
    gpu.kv_cache_write_f32_routed_batched(
        &device_tables.kv_v_ptrs,
        v_src,
        &device_tables.row_session_indices,
        &device_tables.row_positions,
        route_shape.kv_v_layers,
        kv_layer_index,
        kv_dim,
        row_count,
    )
}

pub fn prefill_session_batch_write_q8_kv_layer(
    gpu: &mut Gpu,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    kv_layer_index: usize,
    k_src: &GpuTensor,
    v_src: &GpuTensor,
    n_kv_heads: usize,
    head_dim: usize,
    row_count: usize,
) -> HipResult<()> {
    if kv_layer_index >= route_shape.kv_k_layers || kv_layer_index >= route_shape.kv_v_layers {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "grouped MoE session prefill routed Q8 KV write layer {kv_layer_index} out of range for route shape {:?}",
                route_shape,
            ),
        ));
    }
    gpu.kv_cache_write_q8_0_routed_batched(
        &device_tables.kv_k_ptrs,
        k_src,
        &device_tables.row_session_indices,
        &device_tables.row_positions,
        route_shape.kv_k_layers,
        kv_layer_index,
        n_kv_heads,
        head_dim,
        row_count,
    )?;
    gpu.kv_cache_write_q8_0_routed_batched(
        &device_tables.kv_v_ptrs,
        v_src,
        &device_tables.row_session_indices,
        &device_tables.row_positions,
        route_shape.kv_v_layers,
        kv_layer_index,
        n_kv_heads,
        head_dim,
        row_count,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn dense_prefill_session_batch_attention_f32_layer(
    gpu: &mut Gpu,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    kv_layer_index: usize,
    q_batch: &GpuTensor,
    out_batch: &GpuTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    max_ctx_len: usize,
    row_count: usize,
) -> HipResult<()> {
    if kv_layer_index >= route_shape.kv_k_layers || kv_layer_index >= route_shape.kv_v_layers {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "dense session prefill routed attention layer {kv_layer_index} out of range for route shape {:?}",
                route_shape,
            ),
        ));
    }
    gpu.attention_f32_routed_batched(
        q_batch,
        &device_tables.kv_k_ptrs,
        &device_tables.kv_v_ptrs,
        out_batch,
        &device_tables.row_session_indices,
        &device_tables.row_positions,
        route_shape.kv_k_layers,
        kv_layer_index,
        n_heads,
        n_kv_heads,
        head_dim,
        max_seq,
        max_ctx_len,
        row_count,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_session_batch_attention_q8_layer(
    gpu: &mut Gpu,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    kv_layer_index: usize,
    q_batch: &GpuTensor,
    out_batch: &GpuTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    max_ctx_len: usize,
    row_count: usize,
) -> HipResult<()> {
    if kv_layer_index >= route_shape.kv_k_layers || kv_layer_index >= route_shape.kv_v_layers {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "grouped MoE session prefill routed Q8 attention layer {kv_layer_index} out of range for route shape {:?}",
                route_shape,
            ),
        ));
    }
    gpu.attention_q8_0_routed_batched(
        q_batch,
        &device_tables.kv_k_ptrs,
        &device_tables.kv_v_ptrs,
        out_batch,
        &device_tables.row_session_indices,
        &device_tables.row_positions,
        route_shape.kv_k_layers,
        kv_layer_index,
        n_heads,
        n_kv_heads,
        head_dim,
        max_seq,
        max_ctx_len,
        row_count,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn grouped_moe_prefill_session_batch_gated_delta_net_q8_layer(
    gpu: &mut Gpu,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    sessions: usize,
    delta_layer_index: usize,
    q_batch: &GpuTensor,
    k_batch: &GpuTensor,
    v_batch: &GpuTensor,
    gate_batch: &GpuTensor,
    beta_batch: &GpuTensor,
    out_batch: &GpuTensor,
    row_count: usize,
    n_heads: usize,
    head_dim: usize,
) -> HipResult<()> {
    if delta_layer_index >= route_shape.dn_s_layers
        || delta_layer_index >= route_shape.dn_scale_layers
    {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "grouped MoE session prefill routed Q8 DeltaNet layer {delta_layer_index} out of range for route shape {:?}",
                route_shape,
            ),
        ));
    }
    gpu.gated_delta_net_q8_routed_batch_seq(
        q_batch,
        k_batch,
        v_batch,
        gate_batch,
        beta_batch,
        &device_tables.dn_s_ptrs,
        &device_tables.dn_scale_ptrs,
        &device_tables.row_session_indices,
        out_batch,
        route_shape.dn_s_layers,
        delta_layer_index,
        row_count,
        n_heads,
        head_dim,
        sessions,
    )
}

pub fn dense_prefill_session_batch_scatter_last_logits(
    gpu: &mut Gpu,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    batch_logits: &GpuTensor,
    vocab_size: usize,
    sessions: usize,
) -> HipResult<()> {
    gpu.scatter_session_last_logits_f32(
        batch_logits,
        &device_tables.logits_ptrs,
        &device_tables.session_last_row_indices,
        vocab_size,
        sessions,
    )
}

pub fn dense_prefill_session_batch_final_logits_full_precision(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    row_count: usize,
    sessions: usize,
) -> HipResult<()> {
    if row_count == 0 {
        return Err(hip_bridge::HipError::new(
            0,
            "dense session prefill final logits requires at least one prefix row",
        ));
    }
    if row_count > pbs.max_batch {
        return Err(hip_bridge::HipError::new(
            0,
            "dense session prefill final logits row_count exceeds PrefillBatchScratch max_batch",
        ));
    }

    let normed_rows = pbs.x_norm_batch.sub_offset(0, row_count * config.dim);
    gpu.rmsnorm_batched(
        &pbs.x_batch,
        &weights.output_norm,
        &normed_rows,
        row_count,
        config.dim,
        config.norm_eps,
    )?;

    let batch_logits = gpu.alloc_tensor(&[row_count * config.vocab_size], DType::F32)?;
    let result = match weights.output.gpu_dtype {
        DType::F32 => gpu.gemm_f32_register_tiled(
            &weights.output.buf,
            &normed_rows,
            &batch_logits,
            weights.output.m,
            weights.output.k,
            row_count,
        ),
        DType::F16 | DType::BF16 | DType::Raw => gemm_fp16_or_bf16_x_f32_wmma(
            gpu,
            &weights.output.buf,
            &normed_rows,
            &batch_logits,
            weights.output.m,
            weights.output.k,
            row_count,
        ),
        DType::Q8_0 => gpu.gemm_q8_0_batched_chunked(
            &weights.output.buf,
            &normed_rows,
            &batch_logits,
            weights.output.m,
            weights.output.k,
            row_count,
        ),
        DType::MQ4G256 => {
            let rot = gpu.alloc_tensor(&[row_count * weights.output.k], DType::F32)?;
            let rotated = gpu
                .rotate_x_mq_batched(&normed_rows, &rot, weights.output.k, row_count)
                .and_then(|()| {
                    gpu.gemm_hfq4g256(
                        &weights.output.buf,
                        &rot,
                        &batch_logits,
                        weights.output.m,
                        weights.output.k,
                        row_count,
                    )
                });
            let _ = gpu.free_tensor(rot);
            rotated
        }
        other => Err(hip_bridge::HipError::new(
            0,
            &format!(
                "dense session prefill final logits does not yet support lm_head dtype {other:?}; use serial_reference backend"
            ),
        )),
    }
    .and_then(|()| {
        dense_prefill_session_batch_scatter_last_logits(
            gpu,
            device_tables,
            &batch_logits,
            config.vocab_size,
            sessions,
        )
    });
    let _ = gpu.free_tensor(batch_logits);
    result
}

pub fn grouped_moe_prefill_session_batch_final_logits(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    row_count: usize,
    sessions: usize,
) -> HipResult<()> {
    if row_count == 0 {
        return Err(hip_bridge::HipError::new(
            0,
            "grouped MoE session prefill final logits requires at least one prefix row",
        ));
    }
    if row_count > pbs.max_batch {
        return Err(hip_bridge::HipError::new(
            0,
            "grouped MoE session prefill final logits row_count exceeds PrefillBatchScratch max_batch",
        ));
    }

    let normed_rows = pbs.x_norm_batch.sub_offset(0, row_count * config.dim);
    gpu.rmsnorm_batched(
        &pbs.x_batch,
        &weights.output_norm,
        &normed_rows,
        row_count,
        config.dim,
        config.norm_eps,
    )?;

    let batch_logits = gpu.alloc_owned(&[row_count * config.vocab_size], DType::F32)?;
    match weights.output.gpu_dtype {
        DType::F32 => gpu.gemm_f32_register_tiled(
            &weights.output.buf,
            &normed_rows,
            &batch_logits,
            weights.output.m,
            weights.output.k,
            row_count,
        ),
        DType::F16 | DType::Raw => gpu.gemm_f16_batched_lmhead(
            &weights.output.buf,
            &normed_rows,
            &batch_logits,
            weights.output.m,
            weights.output.k,
            row_count,
        ),
        DType::BF16 => gpu.gemm_bf16_x_bf16_wmma(
            &weights.output.buf,
            &normed_rows,
            &batch_logits,
            weights.output.m,
            weights.output.k,
            row_count,
        ),
        DType::Q8_0 => gpu.gemm_q8_0_batched_chunked(
            &weights.output.buf,
            &normed_rows,
            &batch_logits,
            weights.output.m,
            weights.output.k,
            row_count,
        ),
        DType::MQ4G256 => {
            let rotated = pbs.x_rot_batch.sub_offset(0, row_count * config.dim);
            rotate_x_mq_batched_for(
                gpu,
                &weights.output,
                &normed_rows,
                &rotated,
                config.dim,
                row_count,
            )
            .and_then(|()| {
                gpu.gemm_hfq4g256(
                    &weights.output.buf,
                    &rotated,
                    &batch_logits,
                    weights.output.m,
                    weights.output.k,
                    row_count,
                )
            })
        }
        DType::MQ6G256 => {
            let rotated = pbs.x_rot_batch.sub_offset(0, row_count * config.dim);
            rotate_x_mq_batched_for(
                gpu,
                &weights.output,
                &normed_rows,
                &rotated,
                config.dim,
                row_count,
            )
            .and_then(|()| {
                gpu.gemm_hfq6g256_batched_lmhead(
                    &weights.output.buf,
                    &rotated,
                    &batch_logits,
                    weights.output.m,
                    weights.output.k,
                    row_count,
                )
            })
        }
        DType::MQ3G256 => {
            let rotated = pbs.x_rot_batch.sub_offset(0, row_count * config.dim);
            rotate_x_mq_batched_for(
                gpu,
                &weights.output,
                &normed_rows,
                &rotated,
                config.dim,
                row_count,
            )
            .and_then(|()| {
                gpu.gemm_hfq3g256_batched_lmhead(
                    &weights.output.buf,
                    &rotated,
                    &batch_logits,
                    weights.output.m,
                    weights.output.k,
                    row_count,
                )
            })
        }
        other => Err(hip_bridge::HipError::new(
            0,
            &format!(
                "grouped MoE session prefill final logits does not yet support lm_head dtype {other:?}; use serial_reference backend"
            ),
        )),
    }?;
    dense_prefill_session_batch_scatter_last_logits(
        gpu,
        device_tables,
        &batch_logits,
        config.vocab_size,
        sessions,
    )?;
    // `batch_logits` (RAII `OwnedTensor`) returns to the pool on drop.
    drop(batch_logits);
    gpu.reclaim_pending();
    Ok(())
}

pub fn validate_dense_prefill_session_batch_fused_prefix_full_precision_contract(
    config: &Qwen35Config,
    signatures: &[DensePrefillSessionBatchStateSignature],
    execution_plan: &DensePrefillSessionBatchExecutionPlan,
) -> Result<(), String> {
    if config.num_experts != 0 || config.has_shared_expert {
        return Err(
            "dense session fused prefix currently supports dense Qwen35 only; MoE/A3B stays on serial_reference"
                .to_string(),
        );
    }
    if execution_plan.multi_state_prefix_rows == 0 {
        return Err(
            "dense session fused prefix requires at least one multi-session prefix row".to_string(),
        );
    }
    validate_dense_prefill_session_batch_state_signatures(signatures)?;
    for (idx, signature) in signatures.iter().enumerate() {
        if signature.kv_compact_offset != 0 {
            return Err(format!(
                "dense session fused prefix row {idx} has compacted KV offset {}; eviction/compaction is not fused yet",
                signature.kv_compact_offset,
            ));
        }
        // KV may be plain Q8 (Q8_0, inline per-block scale) or full precision —
        // the per-layer KV write + attention branch on `kv_q8` in
        // `forward_prefill_dense_session_batch_prefix_full_precision`. Asym/FWHT
        // KV and any other quantized-but-not-plain-Q8 state stay on
        // serial_reference (not fused). (Row uniformity is already enforced by
        // `validate_dense_prefill_session_batch_state_signatures`.)
        if signature.kv_quant_asym2
            || signature.kv_quant_asym3
            || signature.kv_quant_asym4
            || signature.kv_quant_fwht
            || (signature.kv_quantized && !signature.kv_quant_q8)
        {
            return Err(format!(
                "dense session fused prefix row {idx} has unsupported KV quantization; only plain Q8 or FP32 KV is fused"
            ));
        }
        if signature.dn_quant != StateQuant::FP32 {
            return Err(format!(
                "dense session fused prefix row {idx} has {:?} DeltaNet state; first fused target is FP32 DeltaNet state",
                signature.dn_quant,
            ));
        }
    }
    Ok(())
}

pub fn validate_grouped_moe_prefill_session_batch_q8_state_contract(
    config: &Qwen35Config,
    signatures: &[DensePrefillSessionBatchStateSignature],
    execution_plan: &DensePrefillSessionBatchExecutionPlan,
    arch: &str,
) -> Result<(), String> {
    if config.num_experts == 0 || !config.has_shared_expert {
        return Err(
            "grouped MoE session fused prefix requires Qwen35 MoE/A3B weights; dense Qwen35 should use fused_dense"
                .to_string(),
        );
    }
    if !arch.starts_with("gfx11") && !arch.starts_with("gfx12") {
        return Err(format!(
            "grouped MoE session fused prefix requires an RDNA grouped-MoE target, got arch={arch}"
        ));
    }
    if config.num_experts_per_tok != 8
        && !(config.paged_experts && config.num_experts_per_tok == 10)
    {
        return Err(format!(
            "grouped MoE session fused prefix currently requires K_TOP=8, or paged K_TOP=10, got {}",
            config.num_experts_per_tok,
        ));
    }
    if execution_plan.multi_state_prefix_rows == 0 {
        return Err(
            "grouped MoE session fused prefix requires at least one multi-session prefix row"
                .to_string(),
        );
    }
    validate_dense_prefill_session_batch_state_signatures(signatures)?;
    for (idx, signature) in signatures.iter().enumerate() {
        if signature.kv_compact_offset != 0 {
            return Err(format!(
                "grouped MoE session fused prefix row {idx} has compacted KV offset {}; eviction/compaction is not fused yet",
                signature.kv_compact_offset,
            ));
        }
        if !signature.kv_quantized || !signature.kv_quant_q8 {
            return Err(format!(
                "grouped MoE session fused prefix row {idx} must use Q8 KV state for the MQ4 control path"
            ));
        }
        if signature.kv_quant_asym2
            || signature.kv_quant_asym3
            || signature.kv_quant_asym4
            || signature.kv_quant_fwht
        {
            return Err(format!(
                "grouped MoE session fused prefix row {idx} has unsupported KV quantization flags; first MoE target is plain Q8 KV"
            ));
        }
        if signature.dn_quant != StateQuant::Q8 {
            return Err(format!(
                "grouped MoE session fused prefix row {idx} has {:?} DeltaNet state; first MoE target is Q8 DeltaNet state",
                signature.dn_quant,
            ));
        }
    }
    Ok(())
}

// Weight dtypes the dense fused prefill GEMM helpers can dispatch. Full precision
// (F32/F16/BF16/Raw) plus plain Q8_0 and MQ4G256 (quantized dense models). MQ6G256
// and other quant formats have no batched non-residual kernel yet, so models using
// them fall back to serial_reference via the contract. (Name kept to avoid churn.)
fn dense_prefill_weight_full_precision_supported(weight: &WeightTensor) -> bool {
    matches!(
        weight.gpu_dtype,
        DType::F32 | DType::F16 | DType::BF16 | DType::Raw | DType::Q8_0 | DType::MQ4G256
    )
}

pub fn validate_dense_prefill_session_batch_fused_prefix_full_precision_weights(
    weights: &Qwen35Weights,
) -> Result<(), String> {
    if !matches!(
        weights.embd_format,
        EmbeddingFormat::F32 | EmbeddingFormat::Q8_0 | EmbeddingFormat::HFQ4G256
    ) {
        return Err(format!(
            "dense session fused prefix does not support embedding format {:?} yet",
            weights.embd_format,
        ));
    }
    if !dense_prefill_weight_full_precision_supported(&weights.output) {
        return Err(format!(
            "dense session fused prefix does not support lm_head dtype {:?} yet",
            weights.output.gpu_dtype,
        ));
    }
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let supported = match layer {
            LayerWeights::DeltaNet(layer) => {
                dense_prefill_weight_full_precision_supported(&layer.wqkv)
                    && dense_prefill_weight_full_precision_supported(&layer.wz)
                    && dense_prefill_weight_full_precision_supported(&layer.w_alpha)
                    && dense_prefill_weight_full_precision_supported(&layer.w_beta)
                    && dense_prefill_weight_full_precision_supported(&layer.wo)
                    && dense_prefill_weight_full_precision_supported(&layer.w_gate)
                    && dense_prefill_weight_full_precision_supported(&layer.w_up)
                    && dense_prefill_weight_full_precision_supported(&layer.w_down)
            }
            LayerWeights::FullAttn(layer) => {
                dense_prefill_weight_full_precision_supported(&layer.wq)
                    && dense_prefill_weight_full_precision_supported(&layer.wk)
                    && dense_prefill_weight_full_precision_supported(&layer.wv)
                    && dense_prefill_weight_full_precision_supported(&layer.wo)
                    && dense_prefill_weight_full_precision_supported(&layer.w_gate)
                    && dense_prefill_weight_full_precision_supported(&layer.w_up)
                    && dense_prefill_weight_full_precision_supported(&layer.w_down)
            }
            LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_) => false,
        };
        if !supported {
            return Err(format!(
                "dense session fused prefix layer {layer_idx} has unsupported dense/MoE weight dtypes; first target is dense full-precision weights"
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dense_prefill_session_batch_gated_delta_net_f32_layer(
    gpu: &mut Gpu,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    sessions: usize,
    delta_layer_index: usize,
    q_batch: &GpuTensor,
    k_batch: &GpuTensor,
    v_batch: &GpuTensor,
    gate_batch: &GpuTensor,
    beta_batch: &GpuTensor,
    output_batch: &GpuTensor,
    row_count: usize,
    n_heads: usize,
    head_dim: usize,
) -> HipResult<()> {
    if delta_layer_index >= route_shape.dn_s_layers {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "dense session prefill routed DeltaNet layer {delta_layer_index} out of range for route shape {:?}",
                route_shape,
            ),
        ));
    }
    gpu.gated_delta_net_f32_routed_batch_seq(
        q_batch,
        k_batch,
        v_batch,
        gate_batch,
        beta_batch,
        &device_tables.dn_s_ptrs,
        &device_tables.row_session_indices,
        output_batch,
        route_shape.dn_s_layers,
        delta_layer_index,
        row_count,
        n_heads,
        head_dim,
        sessions,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn dense_prefill_session_batch_conv1d_silu_split_layer(
    gpu: &mut Gpu,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    sessions: usize,
    delta_layer_index: usize,
    q_out: &GpuTensor,
    k_out: &GpuTensor,
    v_out: &GpuTensor,
    input: &GpuTensor,
    weight: &GpuTensor,
    k_dim: usize,
    v_dim: usize,
    row_count: usize,
) -> HipResult<()> {
    if delta_layer_index >= route_shape.dn_conv_layers {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "dense session prefill routed conv layer {delta_layer_index} out of range for route shape {:?}",
                route_shape,
            ),
        ));
    }
    gpu.conv1d_silu_split_routed_f32_n(
        q_out,
        k_out,
        v_out,
        input,
        weight,
        &device_tables.dn_conv_ptrs,
        &device_tables.row_session_indices,
        route_shape.dn_conv_layers,
        delta_layer_index,
        k_dim,
        v_dim,
        row_count,
        sessions,
    )
}

impl DensePrefillSessionBatchPointerTableShape {
    pub fn index_for_session_layer(
        self,
        session_index: usize,
        kv_layer_index: usize,
        dn_layer_index: usize,
    ) -> Result<DensePrefillSessionBatchPointerTableIndex, String> {
        if session_index >= self.sessions {
            return Err(format!(
                "dense session prefill pointer table session_index {session_index} out of range for sessions={}",
                self.sessions,
            ));
        }
        let kv_layers = self.kv_k_ptrs.checked_div(self.sessions).unwrap_or(0);
        let dn_layers = self.dn_s_ptrs.checked_div(self.sessions).unwrap_or(0);
        if kv_layer_index >= kv_layers {
            return Err(format!(
                "dense session prefill pointer table kv_layer_index {kv_layer_index} out of range for kv_layers={kv_layers}",
            ));
        }
        if dn_layer_index >= dn_layers {
            return Err(format!(
                "dense session prefill pointer table dn_layer_index {dn_layer_index} out of range for dn_layers={dn_layers}",
            ));
        }
        Ok(DensePrefillSessionBatchPointerTableIndex {
            kv_k_offset: session_index * kv_layers + kv_layer_index,
            kv_v_offset: session_index * kv_layers + kv_layer_index,
            dn_s_offset: session_index * dn_layers + dn_layer_index,
            dn_scale_offset: session_index * dn_layers + dn_layer_index,
            dn_conv_offset: session_index * dn_layers + dn_layer_index,
            logits_offset: session_index,
        })
    }

    pub fn index_for_prefix_row(
        self,
        prefix_row_index: usize,
    ) -> Result<(usize, usize, usize), String> {
        if prefix_row_index >= self.multi_state_prefix_rows {
            return Err(format!(
                "dense session prefill pointer table prefix_row_index {prefix_row_index} out of range for multi_state_prefix_rows={}",
                self.multi_state_prefix_rows,
            ));
        }
        Ok((prefix_row_index, prefix_row_index, prefix_row_index))
    }

    pub fn validate_prefix_row_metadata(
        self,
        plan: &DensePrefillSessionBatchPointerTablePlan,
    ) -> Result<(), String> {
        if plan.prefix_rows.len() != self.multi_state_prefix_rows {
            return Err(format!(
                "dense session prefill pointer table has {} prefix rows, expected {}",
                plan.prefix_rows.len(),
                self.multi_state_prefix_rows,
            ));
        }
        if plan.session_last_row_indices.len() != self.sessions {
            return Err(format!(
                "dense session prefill pointer table has {} session-last-row entries, expected {}",
                plan.session_last_row_indices.len(),
                self.sessions,
            ));
        }
        for (session_index, &row_index) in plan.session_last_row_indices.iter().enumerate() {
            if row_index < 0 {
                return Err(format!(
                    "dense session prefill pointer table session {session_index} has no fused prefix row",
                ));
            }
            let row_index = row_index as usize;
            if row_index >= self.multi_state_prefix_rows {
                return Err(format!(
                    "dense session prefill pointer table session {session_index} last row {row_index} out of range for prefix rows {}",
                    self.multi_state_prefix_rows,
                ));
            }
            let row = &plan.prefix_rows[row_index];
            if row.session_index != session_index {
                return Err(format!(
                    "dense session prefill pointer table session {session_index} last row {row_index} belongs to session {}",
                    row.session_index,
                ));
            }
        }
        Ok(())
    }
}

pub fn dense_prefill_session_batch_pointer_table_plan(
    execution_plan: &DensePrefillSessionBatchExecutionPlan,
    route_shape: DensePrefillSessionStateRouteShape,
    sessions: usize,
) -> DensePrefillSessionBatchPointerTablePlan {
    let shape =
        dense_prefill_session_batch_pointer_table_shape(execution_plan, route_shape, sessions);
    let mut kv_layer_slots = Vec::with_capacity(shape.kv_k_ptrs);
    for session_index in 0..sessions {
        for layer_index in 0..route_shape.kv_k_layers {
            kv_layer_slots.push(DensePrefillSessionBatchLayerPointerSlot {
                session_index,
                layer_index,
            });
        }
    }
    let mut dn_layer_slots = Vec::with_capacity(shape.dn_s_ptrs);
    for session_index in 0..sessions {
        for delta_layer_index in 0..route_shape.dn_s_layers {
            dn_layer_slots.push(DensePrefillSessionBatchDeltaPointerSlot {
                session_index,
                delta_layer_index,
            });
        }
    }
    let logits_slots = (0..sessions).collect();
    let mut prefix_rows = Vec::with_capacity(shape.multi_state_prefix_rows);
    let mut session_last_row_indices = vec![-1; sessions];
    for (round_index, round) in execution_plan
        .rounds
        .iter()
        .take(execution_plan.multi_state_prefix_rounds)
        .enumerate()
    {
        for (round_row_index, row) in round.rows.iter().enumerate() {
            let prefix_row_index = prefix_rows.len() as i32;
            session_last_row_indices[row.session_index] = prefix_row_index;
            prefix_rows.push(DensePrefillSessionBatchPrefixRowSlot {
                round_index,
                round_row_index,
                session_index: row.session_index,
                token_index: row.token_index,
                token: row.token,
                position: row.position,
            });
        }
    }
    DensePrefillSessionBatchPointerTablePlan {
        shape,
        kv_layer_slots,
        dn_layer_slots,
        logits_slots,
        prefix_rows,
        session_last_row_indices,
    }
}

pub fn dense_prefill_session_batch_host_pointer_tables(
    plan: &DensePrefillSessionBatchPointerTablePlan,
    routes: &[DensePrefillSessionStateRoute<'_>],
) -> Result<DensePrefillSessionBatchHostPointerTables, String> {
    if routes.len() != plan.shape.sessions {
        return Err(format!(
            "dense session prefill pointer table has {} routes, expected {}",
            routes.len(),
            plan.shape.sessions,
        ));
    }
    let mut kv_k_ptrs = Vec::with_capacity(plan.shape.kv_k_ptrs);
    let mut kv_v_ptrs = Vec::with_capacity(plan.shape.kv_v_ptrs);
    for slot in &plan.kv_layer_slots {
        let route = routes.get(slot.session_index).ok_or_else(|| {
            format!(
                "dense session prefill KV slot references missing session {}",
                slot.session_index,
            )
        })?;
        let k = route.kv.k_gpu.get(slot.layer_index).ok_or_else(|| {
            format!(
                "dense session prefill KV K slot references missing layer {}",
                slot.layer_index,
            )
        })?;
        let v = route.kv.v_gpu.get(slot.layer_index).ok_or_else(|| {
            format!(
                "dense session prefill KV V slot references missing layer {}",
                slot.layer_index,
            )
        })?;
        kv_k_ptrs.push(k.buf.as_ptr() as u64);
        kv_v_ptrs.push(v.buf.as_ptr() as u64);
    }

    let mut dn_s_ptrs = Vec::with_capacity(plan.shape.dn_s_ptrs);
    let mut dn_scale_ptrs = Vec::with_capacity(plan.shape.dn_scale_ptrs);
    let mut dn_conv_ptrs = Vec::with_capacity(plan.shape.dn_conv_ptrs);
    for slot in &plan.dn_layer_slots {
        let route = routes.get(slot.session_index).ok_or_else(|| {
            format!(
                "dense session prefill DeltaNet slot references missing session {}",
                slot.session_index,
            )
        })?;
        let s = route
            .delta
            .s_matrices
            .get(slot.delta_layer_index)
            .ok_or_else(|| {
                format!(
                    "dense session prefill DeltaNet S slot references missing layer {}",
                    slot.delta_layer_index,
                )
            })?;
        let conv = route
            .delta
            .conv_states
            .get(slot.delta_layer_index)
            .ok_or_else(|| {
                format!(
                    "dense session prefill DeltaNet conv slot references missing layer {}",
                    slot.delta_layer_index,
                )
            })?;
        dn_s_ptrs.push(s.buf.as_ptr() as u64);
        dn_conv_ptrs.push(conv.buf.as_ptr() as u64);
        if plan.shape.dn_scale_ptrs != 0 {
            let scale = route
                .delta
                .s_scales
                .get(slot.delta_layer_index)
                .ok_or_else(|| {
                    format!(
                        "dense session prefill DeltaNet scale slot references missing layer {}",
                        slot.delta_layer_index,
                    )
                })?;
            dn_scale_ptrs.push(scale.buf.as_ptr() as u64);
        }
    }

    let mut logits_ptrs = Vec::with_capacity(plan.shape.logits_ptrs);
    for &session_index in &plan.logits_slots {
        let route = routes.get(session_index).ok_or_else(|| {
            format!("dense session prefill logits slot references missing session {session_index}")
        })?;
        logits_ptrs.push(route.logits.buf.as_ptr() as u64);
    }

    let row_session_indices = plan
        .prefix_rows
        .iter()
        .map(|row| row.session_index as i32)
        .collect();
    let row_tokens = plan
        .prefix_rows
        .iter()
        .map(|row| row.token as i32)
        .collect();
    let row_positions = plan
        .prefix_rows
        .iter()
        .map(|row| row.position as i32)
        .collect();

    let tables = DensePrefillSessionBatchHostPointerTables {
        kv_k_ptrs,
        kv_v_ptrs,
        dn_s_ptrs,
        dn_scale_ptrs,
        dn_conv_ptrs,
        logits_ptrs,
        session_last_row_indices: plan.session_last_row_indices.clone(),
        row_session_indices,
        row_tokens,
        row_positions,
    };
    tables.validate_shape(plan.shape)?;
    Ok(tables)
}

pub fn dense_prefill_session_batch_prefix_tokens_positions(
    plan: &DensePrefillSessionBatchPointerTablePlan,
) -> Result<(Vec<u32>, Vec<usize>), String> {
    plan.shape.validate_prefix_row_metadata(plan)?;
    let tokens = plan.prefix_rows.iter().map(|row| row.token).collect();
    let positions = plan.prefix_rows.iter().map(|row| row.position).collect();
    Ok((tokens, positions))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillSessionBatchShape {
    pub sessions: usize,
    pub total_tokens: usize,
    pub max_tokens_per_session: usize,
}

pub fn validate_dense_prefill_session_batch_shape(
    rows: &[DensePrefillSessionBatchRowShape],
    max_batch: usize,
) -> Result<DensePrefillSessionBatchShape, String> {
    if rows.len() < 2 {
        return Err(
            "dense session prefill batch requires at least two independent sessions".to_string(),
        );
    }
    let mut total_tokens = 0usize;
    let mut max_tokens_per_session = 0usize;
    for (idx, row) in rows.iter().enumerate() {
        if row.tokens == 0 {
            return Err(format!(
                "dense session prefill batch row {idx} has an empty token slice"
            ));
        }
        if row.tokens > max_batch {
            return Err(format!(
                "dense session prefill batch row {idx} has {} tokens, exceeding PrefillBatchScratch max_batch={}",
                row.tokens,
                max_batch,
            ));
        }
        if row.logits_numel == 0 {
            return Err(format!(
                "dense session prefill batch row {idx} has an empty logits tensor"
            ));
        }
        total_tokens += row.tokens;
        max_tokens_per_session = max_tokens_per_session.max(row.tokens);
    }
    Ok(DensePrefillSessionBatchShape {
        sessions: rows.len(),
        total_tokens,
        max_tokens_per_session,
    })
}

pub fn validate_dense_prefill_session_batch_state_signatures(
    signatures: &[DensePrefillSessionBatchStateSignature],
) -> Result<(), String> {
    if signatures.len() < 2 {
        return Err(
            "dense session prefill batch requires at least two independent session state signatures"
                .to_string(),
        );
    }
    let expected = signatures[0];
    for (idx, signature) in signatures.iter().enumerate().skip(1) {
        if *signature != expected {
            return Err(format!(
                "dense session prefill batch row {idx} has incompatible KV/DeltaNet state signature: expected {:?}, got {:?}",
                expected,
                signature,
            ));
        }
    }
    Ok(())
}

pub fn validate_dense_prefill_session_state_route_shapes(
    shapes: &[DensePrefillSessionStateRouteShape],
    expected_sessions: usize,
) -> Result<(), String> {
    if shapes.len() != expected_sessions {
        return Err(format!(
            "dense session prefill batch has {} state routes, expected {expected_sessions}",
            shapes.len(),
        ));
    }
    if shapes.len() < 2 {
        return Err(
            "dense session prefill batch requires at least two independent state routes"
                .to_string(),
        );
    }
    let expected = shapes[0];
    if expected.kv_k_layers == 0
        || expected.kv_v_layers == 0
        || expected.dn_s_layers == 0
        || expected.dn_conv_layers == 0
    {
        return Err(format!(
            "dense session prefill batch row 0 has incomplete KV/DeltaNet route shape: {:?}",
            expected,
        ));
    }
    if expected.kv_k_layers != expected.kv_v_layers {
        return Err(format!(
            "dense session prefill batch row 0 has mismatched KV K/V layers: {:?}",
            expected,
        ));
    }
    if expected.dn_s_layers != expected.dn_conv_layers {
        return Err(format!(
            "dense session prefill batch row 0 has mismatched DeltaNet S/conv layers: {:?}",
            expected,
        ));
    }
    if expected.dn_scale_layers != 0 && expected.dn_scale_layers != expected.dn_s_layers {
        return Err(format!(
            "dense session prefill batch row 0 has mismatched DeltaNet scale layers: {:?}",
            expected,
        ));
    }
    for (idx, shape) in shapes.iter().enumerate().skip(1) {
        if *shape != expected {
            return Err(format!(
                "dense session prefill batch row {idx} has incompatible state route shape: expected {:?}, got {:?}",
                expected,
                shape,
            ));
        }
    }
    Ok(())
}

pub fn expected_dense_prefill_session_state_route_shape(
    config: &Qwen35Config,
) -> DensePrefillSessionStateRouteShape {
    let dn_layers = config
        .layer_types
        .iter()
        .filter(|layer_type| **layer_type == LayerType::LinearAttention)
        .count();
    DensePrefillSessionStateRouteShape {
        kv_k_layers: config.n_layers,
        kv_v_layers: config.n_layers,
        dn_s_layers: dn_layers,
        dn_scale_layers: dn_layers,
        dn_conv_layers: dn_layers,
    }
}

pub fn validate_dense_prefill_session_state_route_shapes_for_config(
    shapes: &[DensePrefillSessionStateRouteShape],
    config: &Qwen35Config,
) -> Result<(), String> {
    validate_dense_prefill_session_state_route_shapes(shapes, shapes.len())?;
    let expected = expected_dense_prefill_session_state_route_shape(config);
    for (idx, shape) in shapes.iter().enumerate() {
        if *shape != expected {
            return Err(format!(
                "dense session prefill batch row {idx} has state route shape {:?}, expected model shape {:?}",
                shape,
                expected,
            ));
        }
    }
    Ok(())
}

pub fn dense_prefill_session_batch_pointer_table_shape(
    execution_plan: &DensePrefillSessionBatchExecutionPlan,
    route_shape: DensePrefillSessionStateRouteShape,
    sessions: usize,
) -> DensePrefillSessionBatchPointerTableShape {
    DensePrefillSessionBatchPointerTableShape {
        sessions,
        multi_state_prefix_rounds: execution_plan.multi_state_prefix_rounds,
        multi_state_prefix_rows: execution_plan.multi_state_prefix_rows,
        max_rows_per_round: execution_plan.max_rows_per_round,
        kv_k_ptrs: sessions * route_shape.kv_k_layers,
        kv_v_ptrs: sessions * route_shape.kv_v_layers,
        dn_s_ptrs: sessions * route_shape.dn_s_layers,
        dn_scale_ptrs: sessions * route_shape.dn_scale_layers,
        dn_conv_ptrs: sessions * route_shape.dn_conv_layers,
        logits_ptrs: sessions,
        session_last_row_indices: sessions,
        row_session_indices: execution_plan.multi_state_prefix_rows,
        row_tokens: execution_plan.multi_state_prefix_rows,
        row_positions: execution_plan.multi_state_prefix_rows,
    }
}

pub fn validate_dense_prefill_session_batch_rows(
    rows: &[DensePrefillSessionBatchRow<'_>],
    pbs: &PrefillBatchScratch,
) -> Result<DensePrefillSessionBatchShape, String> {
    let shapes: Vec<DensePrefillSessionBatchRowShape> = rows
        .iter()
        .map(|row| DensePrefillSessionBatchRowShape {
            tokens: row.tokens.len(),
            logits_numel: row.logits.numel(),
        })
        .collect();
    let shape = validate_dense_prefill_session_batch_shape(&shapes, pbs.max_batch)?;
    let signatures: Vec<DensePrefillSessionBatchStateSignature> = rows
        .iter()
        .map(|row| DensePrefillSessionBatchStateSignature {
            kv_physical_cap: row.kv_cache.physical_cap,
            kv_compact_offset: row.kv_cache.compact_offset,
            kv_quantized: row.kv_cache.quantized,
            kv_quant_q8: row.kv_cache.quant_q8,
            kv_quant_asym2: row.kv_cache.quant_asym2,
            kv_quant_asym3: row.kv_cache.quant_asym3,
            kv_quant_asym4: row.kv_cache.quant_asym4,
            kv_quant_fwht: row.kv_cache.quant_fwht,
            dn_quant: row.dn_state.quant,
        })
        .collect();
    validate_dense_prefill_session_batch_state_signatures(&signatures)?;
    let route_shapes: Vec<DensePrefillSessionStateRouteShape> = rows
        .iter()
        .map(|row| DensePrefillSessionStateRouteShape {
            kv_k_layers: row.kv_cache.k_gpu.len(),
            kv_v_layers: row.kv_cache.v_gpu.len(),
            dn_s_layers: row.dn_state.s_matrices.len(),
            dn_scale_layers: row.dn_state.s_scales.len(),
            dn_conv_layers: row.dn_state.conv_states.len(),
        })
        .collect();
    validate_dense_prefill_session_state_route_shapes(&route_shapes, rows.len())?;
    Ok(shape)
}

pub fn validate_dense_prefill_session_batch_rows_for_config(
    rows: &[DensePrefillSessionBatchRow<'_>],
    pbs: &PrefillBatchScratch,
    config: &Qwen35Config,
) -> Result<DensePrefillSessionBatchShape, String> {
    let shape = validate_dense_prefill_session_batch_rows(rows, pbs)?;
    let route_shapes: Vec<DensePrefillSessionStateRouteShape> = rows
        .iter()
        .map(|row| DensePrefillSessionStateRouteShape {
            kv_k_layers: row.kv_cache.k_gpu.len(),
            kv_v_layers: row.kv_cache.v_gpu.len(),
            dn_s_layers: row.dn_state.s_matrices.len(),
            dn_scale_layers: row.dn_state.s_scales.len(),
            dn_conv_layers: row.dn_state.conv_states.len(),
        })
        .collect();
    validate_dense_prefill_session_state_route_shapes_for_config(&route_shapes, config)?;
    Ok(shape)
}

pub fn build_dense_prefill_session_batch_rounds(
    inputs: &[DensePrefillSessionBatchInput<'_>],
    max_batch: usize,
) -> Result<Vec<DensePrefillSessionBatchRound>, String> {
    if inputs.len() < 2 {
        return Err(
            "dense session prefill batch requires at least two independent sessions".to_string(),
        );
    }
    if inputs.len() > max_batch {
        return Err(format!(
            "dense session prefill batch has {} sessions, exceeding PrefillBatchScratch max_batch={max_batch}",
            inputs.len(),
        ));
    }

    let mut max_tokens_per_session = 0usize;
    for (idx, input) in inputs.iter().enumerate() {
        if input.tokens.is_empty() {
            return Err(format!(
                "dense session prefill batch row {idx} has an empty token slice"
            ));
        }
        max_tokens_per_session = max_tokens_per_session.max(input.tokens.len());
    }

    let mut rounds = Vec::with_capacity(max_tokens_per_session);
    for token_index in 0..max_tokens_per_session {
        let mut rows = Vec::with_capacity(inputs.len());
        for (session_index, input) in inputs.iter().enumerate() {
            if let Some(&token) = input.tokens.get(token_index) {
                rows.push(DensePrefillSessionBatchRoundRow {
                    session_index,
                    token_index,
                    token,
                    position: input.start_pos + token_index,
                });
            }
        }
        if !rows.is_empty() {
            rounds.push(DensePrefillSessionBatchRound { rows });
        }
    }
    Ok(rounds)
}

pub fn build_dense_prefill_session_batch_execution_plan(
    inputs: &[DensePrefillSessionBatchInput<'_>],
    max_batch: usize,
) -> Result<DensePrefillSessionBatchExecutionPlan, String> {
    let rounds = build_dense_prefill_session_batch_rounds(inputs, max_batch)?;
    let mut total_rows = 0usize;
    let mut max_rows_per_round = 0usize;
    let mut multi_state_rounds = 0usize;
    let mut state_routes = Vec::with_capacity(rounds.len());
    let mut last_multi_state_round = None;
    for round in &rounds {
        total_rows += round.rows.len();
        max_rows_per_round = max_rows_per_round.max(round.rows.len());
        let route = round.state_route();
        if matches!(
            route,
            DensePrefillSessionBatchRoundStateRoute::MultiSession { .. }
        ) {
            multi_state_rounds += 1;
            last_multi_state_round = Some(state_routes.len());
        }
        state_routes.push(route);
    }
    let multi_state_prefix_rounds = last_multi_state_round.map(|idx| idx + 1).unwrap_or(0);
    let multi_state_prefix_rows: usize = rounds[..multi_state_prefix_rounds]
        .iter()
        .map(|round| round.rows.len())
        .sum();
    let singleton_tail =
        last_multi_state_round.and_then(|last_multi| {
            let start_round = last_multi + 1;
            if start_round >= state_routes.len() {
                return None;
            }
            let session_index = match state_routes[start_round] {
                DensePrefillSessionBatchRoundStateRoute::SingleSession { session_index } => {
                    session_index
                }
                DensePrefillSessionBatchRoundStateRoute::MultiSession { .. } => return None,
            };
            let mut rows = 0usize;
            for route in &state_routes[start_round..] {
                match route {
                    DensePrefillSessionBatchRoundStateRoute::SingleSession {
                        session_index: idx,
                    } if *idx == session_index => rows += 1,
                    _ => return None,
                }
            }
            Some(DensePrefillSessionBatchSingletonTail {
                start_round,
                session_index,
                rows,
            })
        });
    Ok(DensePrefillSessionBatchExecutionPlan {
        rounds,
        state_routes,
        total_rows,
        max_rows_per_round,
        multi_state_rounds,
        multi_state_prefix_rounds,
        multi_state_prefix_rows,
        singleton_tail,
    })
}

#[allow(clippy::too_many_arguments)]
fn forward_prefill_dense_session_batch_prefix_full_precision(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    row_count: usize,
    sessions: usize,
    max_ctx_len: usize,
    // Per-batch KV quant (uniform across rows — see the state-signature contract).
    // true = the sessions' KV caches are plain Q8 (Q8_0); the KV write +
    // attention use the Q8 path. false = full-precision F32 KV.
    kv_q8: bool,
) -> HipResult<()> {
    let dim = config.dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256_batched(
            &weights.token_embd,
            &pbs.x_batch,
            &pbs.tokens,
            row_count,
            dim,
        )?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8_batched(
            &weights.token_embd,
            &pbs.x_batch,
            &pbs.tokens,
            row_count,
            dim,
        )?,
        EmbeddingFormat::F32 => gpu.embedding_lookup_f32_batched(
            &weights.token_embd,
            &pbs.x_batch,
            &pbs.tokens,
            row_count,
            dim,
        )?,
        other => {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("dense session fused prefix does not support embedding format {other:?}"),
            ));
        }
    }

    let mut delta_layer_idx = 0usize;
    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                gpu.rmsnorm_batched(
                    &pbs.x_batch,
                    &layer.attn_norm,
                    &pbs.x_rot_batch,
                    row_count,
                    dim,
                    config.norm_eps,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.wqkv,
                    &pbs.x_rot_batch,
                    &pbs.dn_qkv_batch,
                    row_count,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.wz,
                    &pbs.x_rot_batch,
                    &pbs.dn_z_batch,
                    row_count,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.w_beta,
                    &pbs.x_rot_batch,
                    &pbs.dn_beta_batch,
                    row_count,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.w_alpha,
                    &pbs.x_rot_batch,
                    &pbs.dn_alpha_batch,
                    row_count,
                )?;

                gpu.fused_sigmoid_alpha_gate_f32_batched(
                    &pbs.dn_beta_batch,
                    &pbs.dn_alpha_batch,
                    &layer.dt_bias,
                    &layer.a_log,
                    n_v_heads,
                    row_count,
                )?;
                dense_prefill_session_batch_conv1d_silu_split_layer(
                    gpu,
                    device_tables,
                    route_shape,
                    sessions,
                    delta_layer_idx,
                    &pbs.dn_q_raw_batch,
                    &pbs.dn_k_raw_batch,
                    &pbs.dn_v_batch,
                    &pbs.dn_qkv_batch,
                    &layer.conv_weight,
                    k_dim,
                    v_dim,
                    row_count,
                )?;
                gpu.fused_qk_l2_norm_scale_f32_batched(
                    &pbs.dn_q_raw_batch,
                    &pbs.dn_k_raw_batch,
                    config.linear_num_key_heads,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                    row_count,
                )?;
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32_batched(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        config.linear_num_key_heads,
                        ratio,
                        hd,
                        row_count,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_q_batch.buf,
                        &pbs.dn_q_raw_batch.buf,
                        row_count * k_dim * 4,
                    )?;
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_k_batch.buf,
                        &pbs.dn_k_raw_batch.buf,
                        row_count * k_dim * 4,
                    )?;
                }
                dense_prefill_session_batch_gated_delta_net_f32_layer(
                    gpu,
                    device_tables,
                    route_shape,
                    sessions,
                    delta_layer_idx,
                    &pbs.dn_q_batch,
                    &pbs.dn_k_batch,
                    &pbs.dn_v_batch,
                    &pbs.dn_alpha_batch,
                    &pbs.dn_beta_batch,
                    &pbs.dn_attn_out_batch,
                    row_count,
                    n_v_heads,
                    config.linear_value_head_dim,
                )?;
                gpu.gated_norm_f32_batched(
                    &pbs.dn_attn_out_batch,
                    &pbs.dn_z_batch,
                    &layer.norm_weight,
                    &pbs.dn_normed_batch,
                    n_v_heads,
                    config.linear_value_head_dim,
                    config.norm_eps,
                    row_count,
                )?;
                dense_session_prefill_gemm_full_precision_residual(
                    gpu,
                    &layer.wo,
                    &pbs.dn_normed_batch,
                    &pbs.x_batch,
                    &pbs.x_rot_batch,
                    row_count,
                )?;

                gpu.rmsnorm_batched(
                    &pbs.x_batch,
                    &layer.ffn_norm,
                    &pbs.x_rot_batch,
                    row_count,
                    dim,
                    config.norm_eps,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.w_gate,
                    &pbs.x_rot_batch,
                    &pbs.gate_ffn_batch,
                    row_count,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.w_up,
                    &pbs.x_rot_batch,
                    &pbs.up_batch,
                    row_count,
                )?;
                gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
                dense_session_prefill_gemm_full_precision_residual(
                    gpu,
                    &layer.w_down,
                    &pbs.ffn_hidden_batch,
                    &pbs.x_batch,
                    &pbs.x_rot_batch,
                    row_count,
                )?;
                delta_layer_idx += 1;
            }
            (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                let kv_dim = config.n_kv_heads * config.head_dim;
                gpu.rmsnorm_batched(
                    &pbs.x_batch,
                    &layer.attn_norm,
                    &pbs.x_rot_batch,
                    row_count,
                    dim,
                    config.norm_eps,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.wq,
                    &pbs.x_rot_batch,
                    &pbs.fa_q_full_batch,
                    row_count,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.wk,
                    &pbs.x_rot_batch,
                    &pbs.fa_k_batch,
                    row_count,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.wv,
                    &pbs.x_rot_batch,
                    &pbs.fa_v_batch,
                    row_count,
                )?;
                qwen35_materialize_fa_q(
                    gpu,
                    config,
                    &pbs.fa_q_full_batch,
                    &pbs.fa_q_batch,
                    &pbs.fa_gate_batch,
                    row_count,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_q_batch,
                    &layer.q_norm,
                    &pbs.fa_q_batch,
                    row_count * config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_k_batch,
                    &layer.k_norm,
                    &pbs.fa_k_batch,
                    row_count * config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32_batched(
                    &pbs.fa_q_batch,
                    &pbs.fa_k_batch,
                    &pbs.positions,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    n_rot,
                    config.rope_theta,
                    row_count,
                    0,
                )?;
                if kv_q8 {
                    // Plain-Q8 KV: the routed write/attention helpers are shared
                    // with the grouped-MoE fused path — they are FFN-agnostic and
                    // operate on Q8_0 (inline-scale) KV buffers via the same
                    // device pointer tables.
                    prefill_session_batch_write_q8_kv_layer(
                        gpu,
                        device_tables,
                        route_shape,
                        layer_idx,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        config.n_kv_heads,
                        config.head_dim,
                        row_count,
                    )?;
                    prefill_session_batch_attention_q8_layer(
                        gpu,
                        device_tables,
                        route_shape,
                        layer_idx,
                        &pbs.fa_q_batch,
                        &pbs.fa_attn_out_batch,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        max_ctx_len,
                        max_ctx_len,
                        row_count,
                    )?;
                } else {
                    dense_prefill_session_batch_write_f32_kv_layer(
                        gpu,
                        device_tables,
                        route_shape,
                        layer_idx,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        kv_dim,
                        row_count,
                    )?;
                    dense_prefill_session_batch_attention_f32_layer(
                        gpu,
                        device_tables,
                        route_shape,
                        layer_idx,
                        &pbs.fa_q_batch,
                        &pbs.fa_attn_out_batch,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        max_ctx_len,
                        max_ctx_len,
                        row_count,
                    )?;
                }
                qwen35_apply_fa_gate(gpu, config, &pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
                dense_session_prefill_gemm_full_precision_residual(
                    gpu,
                    &layer.wo,
                    &pbs.fa_attn_out_batch,
                    &pbs.x_batch,
                    &pbs.x_rot_batch,
                    row_count,
                )?;

                gpu.rmsnorm_batched(
                    &pbs.x_batch,
                    &layer.ffn_norm,
                    &pbs.x_rot_batch,
                    row_count,
                    dim,
                    config.norm_eps,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.w_gate,
                    &pbs.x_rot_batch,
                    &pbs.gate_ffn_batch,
                    row_count,
                )?;
                dense_session_prefill_gemm_full_precision(
                    gpu,
                    &layer.w_up,
                    &pbs.x_rot_batch,
                    &pbs.up_batch,
                    row_count,
                )?;
                gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
                dense_session_prefill_gemm_full_precision_residual(
                    gpu,
                    &layer.w_down,
                    &pbs.ffn_hidden_batch,
                    &pbs.x_batch,
                    &pbs.x_rot_batch,
                    row_count,
                )?;
            }
            _ => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "dense session fused prefix encountered a layer that is not dense Qwen35",
                ));
            }
        }
    }

    dense_prefill_session_batch_final_logits_full_precision(
        gpu,
        weights,
        config,
        pbs,
        device_tables,
        row_count,
        sessions,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_dense_session_batch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    rows: &mut [DensePrefillSessionBatchRow<'_>],
    _scratch: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
) -> HipResult<DensePrefillSessionBatchShape> {
    let shape = validate_dense_prefill_session_batch_rows_for_config(rows, pbs, config)
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    let inputs: Vec<DensePrefillSessionBatchInput<'_>> = rows
        .iter()
        .map(|row| DensePrefillSessionBatchInput {
            tokens: row.tokens,
            start_pos: row.start_pos,
        })
        .collect();
    let execution_plan = build_dense_prefill_session_batch_execution_plan(&inputs, pbs.max_batch)
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    let signatures: Vec<DensePrefillSessionBatchStateSignature> = rows
        .iter()
        .map(|row| DensePrefillSessionBatchStateSignature {
            kv_physical_cap: row.kv_cache.physical_cap,
            kv_compact_offset: row.kv_cache.compact_offset,
            kv_quantized: row.kv_cache.quantized,
            kv_quant_q8: row.kv_cache.quant_q8,
            kv_quant_asym2: row.kv_cache.quant_asym2,
            kv_quant_asym3: row.kv_cache.quant_asym3,
            kv_quant_asym4: row.kv_cache.quant_asym4,
            kv_quant_fwht: row.kv_cache.quant_fwht,
            dn_quant: row.dn_state.quant,
        })
        .collect();
    validate_dense_prefill_session_batch_fused_prefix_full_precision_contract(
        config,
        &signatures,
        &execution_plan,
    )
    .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    validate_dense_prefill_session_batch_fused_prefix_full_precision_weights(weights)
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    let route_shape = expected_dense_prefill_session_state_route_shape(config);
    let pointer_table_plan =
        dense_prefill_session_batch_pointer_table_plan(&execution_plan, route_shape, rows.len());
    if execution_plan.multi_state_prefix_rows > pbs.max_batch {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "dense session prefill fused prefix has {} rows, exceeding PrefillBatchScratch max_batch={}",
                execution_plan.multi_state_prefix_rows, pbs.max_batch,
            ),
        ));
    }
    let (prefix_tokens, prefix_positions) =
        dense_prefill_session_batch_prefix_tokens_positions(&pointer_table_plan)
            .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    upload_prefill_batch_inputs_with_positions(gpu, pbs, &prefix_tokens, &prefix_positions)?;
    let routes: Vec<DensePrefillSessionStateRoute<'_>> = rows
        .iter()
        .map(|row| DensePrefillSessionStateRoute {
            kv: DensePrefillSessionKvStateRoute {
                k_gpu: &row.kv_cache.k_gpu,
                v_gpu: &row.kv_cache.v_gpu,
                physical_cap: row.kv_cache.physical_cap,
                compact_offset: row.kv_cache.compact_offset,
            },
            delta: DensePrefillSessionDeltaStateRoute {
                s_matrices: &row.dn_state.s_matrices,
                s_scales: &row.dn_state.s_scales,
                conv_states: &row.dn_state.conv_states,
                quant: row.dn_state.quant,
            },
            logits: row.logits,
        })
        .collect();
    let host_pointer_tables =
        dense_prefill_session_batch_host_pointer_tables(&pointer_table_plan, &routes)
            .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    drop(routes);
    let device_pointer_tables = upload_dense_prefill_session_batch_pointer_tables(
        gpu,
        pointer_table_plan.shape,
        &host_pointer_tables,
    )?;
    let max_ctx_len = prefix_positions
        .iter()
        .copied()
        .max()
        .map(|pos| pos + 1)
        .unwrap_or(1);
    for (idx, row) in rows.iter().enumerate() {
        let row_end = row.start_pos + row.tokens.len();
        if row_end > row.kv_cache.physical_cap {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "dense session fused prefix row {idx} ends at position {row_end}, exceeding KV physical_cap={}",
                    row.kv_cache.physical_cap,
                ),
            ));
        }
    }
    // Row signatures are uniform (state-signature contract), so row 0's KV quant
    // decides the per-layer KV write/attention path for the whole batch.
    let kv_q8 = signatures.first().map(|s| s.kv_quant_q8).unwrap_or(false);
    let result = forward_prefill_dense_session_batch_prefix_full_precision(
        gpu,
        weights,
        config,
        pbs,
        &device_pointer_tables,
        route_shape,
        execution_plan.multi_state_prefix_rows,
        rows.len(),
        max_ctx_len,
        kv_q8,
    );
    device_pointer_tables.free_gpu(gpu);
    result.map(|()| shape)
}

#[allow(clippy::too_many_arguments)]
fn forward_prefill_grouped_moe_session_batch_prefix_q8_control(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    device_tables: &DensePrefillSessionBatchDevicePointerTables,
    route_shape: DensePrefillSessionStateRouteShape,
    row_count: usize,
    sessions: usize,
    max_ctx_len: usize,
) -> HipResult<()> {
    let dim = config.dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let q8_wmma_arch = gpu.arch_caps.has_wmma();

    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256_batched(
            &weights.token_embd,
            &pbs.x_batch,
            &pbs.tokens,
            row_count,
            dim,
        )?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8_batched(
            &weights.token_embd,
            &pbs.x_batch,
            &pbs.tokens,
            row_count,
            dim,
        )?,
        EmbeddingFormat::F32 => gpu.embedding_lookup_f32_batched(
            &weights.token_embd,
            &pbs.x_batch,
            &pbs.tokens,
            row_count,
            dim,
        )?,
        other => {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "grouped MoE session fused prefix does not support embedding format {other:?}"
                ),
            ));
        }
    }

    let mut delta_layer_idx = 0usize;
    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                let attn_is_q8 = matches!(layer.wqkv.gpu_dtype, DType::Q8_0)
                    && matches!(layer.wz.gpu_dtype, DType::Q8_0)
                    && matches!(layer.w_alpha.gpu_dtype, DType::Q8_0)
                    && matches!(layer.w_beta.gpu_dtype, DType::Q8_0);
                let attn_is_mq4 = matches!(layer.wqkv.gpu_dtype, DType::MQ4G256)
                    && matches!(layer.wz.gpu_dtype, DType::MQ4G256)
                    && matches!(layer.w_alpha.gpu_dtype, DType::MQ4G256)
                    && matches!(layer.w_beta.gpu_dtype, DType::MQ4G256);
                let attn_is_mq6 = matches!(layer.wqkv.gpu_dtype, DType::MQ6G256)
                    && matches!(layer.wz.gpu_dtype, DType::MQ6G256)
                    && matches!(layer.w_alpha.gpu_dtype, DType::MQ6G256)
                    && matches!(layer.w_beta.gpu_dtype, DType::MQ6G256);
                if !attn_is_q8 && !attn_is_mq4 && !attn_is_mq6 {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "grouped MoE session fused prefix currently supports Q8, MQ4, or MQ6 DeltaNet-MoE attention weights only; use serial_reference",
                    ));
                }
                if attn_is_mq4 || attn_is_mq6 {
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &layer.wqkv,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        row_count,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_rot_batch,
                        row_count,
                        dim,
                        config.norm_eps,
                    )?;
                }
                if attn_is_mq4 {
                    gpu.gemm_qkvza_hfq4g256(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        row_count,
                    )?;
                } else if attn_is_mq6 {
                    gpu.gemm_qkvza_hfq6g256(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        row_count,
                    )?;
                } else if q8_wmma_arch {
                    gpu.gemm_qkvza_q8_0_wmma(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        row_count,
                    )?;
                } else {
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.wqkv.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        layer.wqkv.m,
                        layer.wqkv.k,
                        row_count,
                    )?;
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.wz.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_z_batch,
                        layer.wz.m,
                        layer.wz.k,
                        row_count,
                    )?;
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.w_beta.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_beta_batch,
                        layer.w_beta.m,
                        layer.w_beta.k,
                        row_count,
                    )?;
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_alpha_batch,
                        layer.w_alpha.m,
                        layer.w_alpha.k,
                        row_count,
                    )?;
                }
                gpu.fused_sigmoid_alpha_gate_f32_batched(
                    &pbs.dn_beta_batch,
                    &pbs.dn_alpha_batch,
                    &layer.dt_bias,
                    &layer.a_log,
                    n_v_heads,
                    row_count,
                )?;
                dense_prefill_session_batch_conv1d_silu_split_layer(
                    gpu,
                    device_tables,
                    route_shape,
                    sessions,
                    delta_layer_idx,
                    &pbs.dn_q_raw_batch,
                    &pbs.dn_k_raw_batch,
                    &pbs.dn_v_batch,
                    &pbs.dn_qkv_batch,
                    &layer.conv_weight,
                    k_dim,
                    v_dim,
                    row_count,
                )?;
                gpu.fused_qk_l2_norm_scale_f32_batched(
                    &pbs.dn_q_raw_batch,
                    &pbs.dn_k_raw_batch,
                    config.linear_num_key_heads,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                    row_count,
                )?;
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32_batched(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        config.linear_num_key_heads,
                        ratio,
                        hd,
                        row_count,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_q_batch.buf,
                        &pbs.dn_q_raw_batch.buf,
                        row_count * k_dim * 4,
                    )?;
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_k_batch.buf,
                        &pbs.dn_k_raw_batch.buf,
                        row_count * k_dim * 4,
                    )?;
                }
                grouped_moe_prefill_session_batch_gated_delta_net_q8_layer(
                    gpu,
                    device_tables,
                    route_shape,
                    sessions,
                    delta_layer_idx,
                    &pbs.dn_q_batch,
                    &pbs.dn_k_batch,
                    &pbs.dn_v_batch,
                    &pbs.dn_alpha_batch,
                    &pbs.dn_beta_batch,
                    &pbs.dn_attn_out_batch,
                    row_count,
                    n_v_heads,
                    config.linear_value_head_dim,
                )?;
                gpu.gated_norm_f32_batched(
                    &pbs.dn_attn_out_batch,
                    &pbs.dn_z_batch,
                    &layer.norm_weight,
                    &pbs.dn_normed_batch,
                    n_v_heads,
                    config.linear_value_head_dim,
                    config.norm_eps,
                    row_count,
                )?;
                if matches!(layer.wo.gpu_dtype, DType::MQ4G256) {
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.dn_normed_batch,
                        &pbs.dn_normed_rot_batch,
                        layer.wo.k,
                        row_count,
                    )?;
                    gpu.gemm_hfq4g256_residual(
                        &layer.wo.buf,
                        &pbs.dn_normed_rot_batch,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        row_count,
                    )?;
                } else if matches!(layer.wo.gpu_dtype, DType::MQ6G256) {
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.dn_normed_batch,
                        &pbs.dn_normed_rot_batch,
                        layer.wo.k,
                        row_count,
                    )?;
                    gpu.gemm_hfq6g256_residual(
                        &layer.wo.buf,
                        &pbs.dn_normed_rot_batch,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        row_count,
                    )?;
                } else if matches!(layer.wo.gpu_dtype, DType::Q8_0) && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, row_count * layer.wo.m);
                    gpu.gemm_q8_0_residual_wmma(
                        &layer.wo.buf,
                        &pbs.dn_normed_batch,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        row_count,
                    )?;
                } else if matches!(layer.wo.gpu_dtype, DType::Q8_0) {
                    let scratch = pbs
                        .dn_normed_rot_batch
                        .sub_offset(0, row_count * layer.wo.m);
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.wo.buf,
                        &pbs.dn_normed_batch,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        row_count,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, row_count * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "grouped MoE session fused prefix currently supports Q8 or MQ4 DeltaNet-MoE wo weights only; use serial_reference",
                    ));
                }
                let ctx = DispatchCtx::new(gpu);
                prefill_moe_ffn_body_batched(
                    gpu,
                    weights.pager.as_ref(),
                    &layer.ffn,
                    &layer.ffn_norm,
                    config,
                    pbs,
                    row_count,
                    layer_idx,
                    &ctx,
                    None,
                )?;
                delta_layer_idx += 1;
            }
            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) => {
                let attn_is_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0)
                    && matches!(layer.wk.gpu_dtype, DType::Q8_0)
                    && matches!(layer.wv.gpu_dtype, DType::Q8_0);
                let attn_is_mq4 = matches!(layer.wq.gpu_dtype, DType::MQ4G256)
                    && matches!(layer.wk.gpu_dtype, DType::MQ4G256)
                    && matches!(layer.wv.gpu_dtype, DType::MQ4G256);
                let attn_is_mq6 = matches!(layer.wq.gpu_dtype, DType::MQ6G256)
                    && matches!(layer.wk.gpu_dtype, DType::MQ6G256)
                    && matches!(layer.wv.gpu_dtype, DType::MQ6G256);
                if !attn_is_q8 && !attn_is_mq4 && !attn_is_mq6 {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "grouped MoE session fused prefix currently supports Q8, MQ4, or MQ6 FullAttention-MoE attention weights only; use serial_reference",
                    ));
                }
                if attn_is_mq4 || attn_is_mq6 {
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &layer.wq,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        row_count,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_rot_batch,
                        row_count,
                        dim,
                        config.norm_eps,
                    )?;
                }
                if attn_is_mq4 {
                    gpu.gemm_qkv_hfq4g256(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        row_count,
                    )?;
                } else if attn_is_mq6 {
                    gpu.gemm_qkv_hfq6g256(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        row_count,
                    )?;
                } else if q8_wmma_arch {
                    gpu.gemm_qkv_q8_0_wmma(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        row_count,
                    )?;
                } else {
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.wq.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        layer.wq.m,
                        layer.wq.k,
                        row_count,
                    )?;
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.wk.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        layer.wk.m,
                        layer.wk.k,
                        row_count,
                    )?;
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        layer.wv.m,
                        layer.wv.k,
                        row_count,
                    )?;
                }
                qwen35_materialize_fa_q(
                    gpu,
                    config,
                    &pbs.fa_q_full_batch,
                    &pbs.fa_q_batch,
                    &pbs.fa_gate_batch,
                    row_count,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_q_batch,
                    &layer.q_norm,
                    &pbs.fa_q_batch,
                    row_count * config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_k_batch,
                    &layer.k_norm,
                    &pbs.fa_k_batch,
                    row_count * config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                gpu.rope_partial_interleaved_f32_batched(
                    &pbs.fa_q_batch,
                    &pbs.fa_k_batch,
                    &pbs.positions,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    n_rot,
                    config.rope_theta,
                    row_count,
                    0,
                )?;
                prefill_session_batch_write_q8_kv_layer(
                    gpu,
                    device_tables,
                    route_shape,
                    layer_idx,
                    &pbs.fa_k_batch,
                    &pbs.fa_v_batch,
                    config.n_kv_heads,
                    config.head_dim,
                    row_count,
                )?;
                prefill_session_batch_attention_q8_layer(
                    gpu,
                    device_tables,
                    route_shape,
                    layer_idx,
                    &pbs.fa_q_batch,
                    &pbs.fa_attn_out_batch,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    max_ctx_len,
                    max_ctx_len,
                    row_count,
                )?;
                qwen35_apply_fa_gate(gpu, config, &pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
                if matches!(layer.wo.gpu_dtype, DType::MQ4G256) {
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.fa_attn_out_batch,
                        &pbs.fa_attn_out_rot_batch,
                        layer.wo.k,
                        row_count,
                    )?;
                    gpu.gemm_hfq4g256_residual(
                        &layer.wo.buf,
                        &pbs.fa_attn_out_rot_batch,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        row_count,
                    )?;
                } else if matches!(layer.wo.gpu_dtype, DType::MQ6G256) {
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.fa_attn_out_batch,
                        &pbs.fa_attn_out_rot_batch,
                        layer.wo.k,
                        row_count,
                    )?;
                    gpu.gemm_hfq6g256_residual(
                        &layer.wo.buf,
                        &pbs.fa_attn_out_rot_batch,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        row_count,
                    )?;
                } else if matches!(layer.wo.gpu_dtype, DType::Q8_0) && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, row_count * layer.wo.m);
                    gpu.gemm_q8_0_residual_wmma(
                        &layer.wo.buf,
                        &pbs.fa_attn_out_batch,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        row_count,
                    )?;
                } else if matches!(layer.wo.gpu_dtype, DType::Q8_0) {
                    let scratch = pbs
                        .fa_attn_out_rot_batch
                        .sub_offset(0, row_count * layer.wo.m);
                    gpu.gemm_q8_0_batched_chunked(
                        &layer.wo.buf,
                        &pbs.fa_attn_out_batch,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        row_count,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, row_count * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "grouped MoE session fused prefix currently supports Q8 or MQ4 FullAttention-MoE wo weights only; use serial_reference",
                    ));
                }
                let ctx = DispatchCtx::new(gpu);
                prefill_moe_ffn_body_batched(
                    gpu,
                    weights.pager.as_ref(),
                    &layer.ffn,
                    &layer.ffn_norm,
                    config,
                    pbs,
                    row_count,
                    layer_idx,
                    &ctx,
                    None,
                )?;
            }
            _ => {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "grouped MoE session fused prefix encountered unsupported layer {layer_idx}; use serial_reference"
                    ),
                ));
            }
        }
    }

    grouped_moe_prefill_session_batch_final_logits(
        gpu,
        weights,
        config,
        pbs,
        device_tables,
        row_count,
        sessions,
    )
}

pub fn forward_prefill_grouped_moe_session_batch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    rows: &mut [DensePrefillSessionBatchRow<'_>],
    _scratch: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
) -> HipResult<DensePrefillSessionBatchShape> {
    let shape = validate_dense_prefill_session_batch_rows_for_config(rows, pbs, config)
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    let inputs: Vec<DensePrefillSessionBatchInput<'_>> = rows
        .iter()
        .map(|row| DensePrefillSessionBatchInput {
            tokens: row.tokens,
            start_pos: row.start_pos,
        })
        .collect();
    let execution_plan = build_dense_prefill_session_batch_execution_plan(&inputs, pbs.max_batch)
        .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    let signatures: Vec<DensePrefillSessionBatchStateSignature> = rows
        .iter()
        .map(|row| DensePrefillSessionBatchStateSignature {
            kv_physical_cap: row.kv_cache.physical_cap,
            kv_compact_offset: row.kv_cache.compact_offset,
            kv_quantized: row.kv_cache.quantized,
            kv_quant_q8: row.kv_cache.quant_q8,
            kv_quant_asym2: row.kv_cache.quant_asym2,
            kv_quant_asym3: row.kv_cache.quant_asym3,
            kv_quant_asym4: row.kv_cache.quant_asym4,
            kv_quant_fwht: row.kv_cache.quant_fwht,
            dn_quant: row.dn_state.quant,
        })
        .collect();
    validate_grouped_moe_prefill_session_batch_q8_state_contract(
        config,
        &signatures,
        &execution_plan,
        gpu.arch.as_str(),
    )
    .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    let route_shape = expected_dense_prefill_session_state_route_shape(config);
    let pointer_table_plan =
        dense_prefill_session_batch_pointer_table_plan(&execution_plan, route_shape, rows.len());
    if execution_plan.multi_state_prefix_rows > pbs.max_batch {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "grouped MoE session prefill fused prefix has {} rows, exceeding PrefillBatchScratch max_batch={}",
                execution_plan.multi_state_prefix_rows, pbs.max_batch,
            ),
        ));
    }
    let (prefix_tokens, prefix_positions) =
        dense_prefill_session_batch_prefix_tokens_positions(&pointer_table_plan)
            .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    upload_prefill_batch_inputs_with_positions(gpu, pbs, &prefix_tokens, &prefix_positions)?;
    let routes: Vec<DensePrefillSessionStateRoute<'_>> = rows
        .iter()
        .map(|row| DensePrefillSessionStateRoute {
            kv: DensePrefillSessionKvStateRoute {
                k_gpu: &row.kv_cache.k_gpu,
                v_gpu: &row.kv_cache.v_gpu,
                physical_cap: row.kv_cache.physical_cap,
                compact_offset: row.kv_cache.compact_offset,
            },
            delta: DensePrefillSessionDeltaStateRoute {
                s_matrices: &row.dn_state.s_matrices,
                s_scales: &row.dn_state.s_scales,
                conv_states: &row.dn_state.conv_states,
                quant: row.dn_state.quant,
            },
            logits: row.logits,
        })
        .collect();
    let host_pointer_tables =
        dense_prefill_session_batch_host_pointer_tables(&pointer_table_plan, &routes)
            .map_err(|e| hip_bridge::HipError::new(0, &e))?;
    let device_pointer_tables = upload_dense_prefill_session_batch_pointer_tables(
        gpu,
        pointer_table_plan.shape,
        &host_pointer_tables,
    )?;
    let max_ctx_len = prefix_positions
        .iter()
        .copied()
        .max()
        .map(|pos| pos + 1)
        .unwrap_or(1);
    for (idx, row) in rows.iter().enumerate() {
        let row_end = row.start_pos + row.tokens.len();
        if row_end > row.kv_cache.physical_cap {
            device_pointer_tables.free_gpu(gpu);
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "grouped MoE session fused prefix row {idx} ends at position {row_end}, exceeding KV physical_cap={}",
                    row.kv_cache.physical_cap,
                ),
            ));
        }
    }
    let result = forward_prefill_grouped_moe_session_batch_prefix_q8_control(
        gpu,
        weights,
        config,
        pbs,
        &device_pointer_tables,
        route_shape,
        execution_plan.multi_state_prefix_rows,
        rows.len(),
        max_ctx_len,
    );
    device_pointer_tables.free_gpu(gpu);
    result.map(|()| shape)
}

impl PrefillBatchScratch {
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, max_batch: usize) -> HipResult<Self> {
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = k_dim * 2 + v_dim;
        let n_v_heads = config.linear_num_value_heads;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;

        Ok(Self {
            max_batch,
            x_batch: gpu.alloc_tensor(&[max_batch * dim], DType::F32)?,
            x_rot_batch: gpu.alloc_tensor(&[max_batch * dim], DType::F32)?,
            x_norm_batch: gpu.alloc_tensor(&[max_batch * dim], DType::F32)?,
            dn_qkv_batch: gpu.alloc_tensor(&[max_batch * qkv_dim], DType::F32)?,
            dn_z_batch: gpu.alloc_tensor(&[max_batch * v_dim], DType::F32)?,
            dn_alpha_batch: gpu.alloc_tensor(&[max_batch * n_v_heads], DType::F32)?,
            dn_beta_batch: gpu.alloc_tensor(&[max_batch * n_v_heads], DType::F32)?,
            dn_q_raw_batch: gpu.alloc_tensor(&[max_batch * k_dim], DType::F32)?,
            dn_k_raw_batch: gpu.alloc_tensor(&[max_batch * k_dim], DType::F32)?,
            dn_v_batch: gpu.alloc_tensor(&[max_batch * v_dim], DType::F32)?,
            dn_q_batch: gpu.alloc_tensor(&[max_batch * v_dim], DType::F32)?,
            dn_k_batch: gpu.alloc_tensor(&[max_batch * v_dim], DType::F32)?,
            dn_attn_out_batch: gpu.alloc_tensor(&[max_batch * v_dim], DType::F32)?,
            dn_normed_batch: gpu.alloc_tensor(&[max_batch * v_dim], DType::F32)?,
            gate_ffn_batch: gpu.alloc_tensor(&[max_batch * hidden_dim], DType::F32)?,
            up_batch: gpu.alloc_tensor(&[max_batch * hidden_dim], DType::F32)?,
            ffn_hidden_batch: gpu.alloc_tensor(&[max_batch * hidden_dim], DType::F32)?,
            dn_normed_rot_batch: gpu.alloc_tensor(&[max_batch * v_dim], DType::F32)?,
            // F32 dtype = 4 bytes/element, same layout as i32. The rope /
            // attention / kv_write kernels cast the pointer to `const int*`,
            // so dtype is cosmetic. Upload i32 bits via memcpy_htod.
            positions: gpu.alloc_tensor(&[max_batch], DType::F32)?,
            tokens: gpu.alloc_tensor(&[max_batch], DType::F32)?,
            fa_q_full_batch: gpu.alloc_tensor(&[max_batch * q_dim * 2], DType::F32)?,
            fa_q_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            fa_gate_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            fa_k_batch: gpu.alloc_tensor(&[max_batch * kv_dim], DType::F32)?,
            fa_v_batch: gpu.alloc_tensor(&[max_batch * kv_dim], DType::F32)?,
            fa_attn_out_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            fa_attn_out_rot_batch: gpu.alloc_tensor(&[max_batch * q_dim], DType::F32)?,
            moe_router_logits_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch * config.num_experts], DType::F32)?)
            } else {
                None
            },
            moe_shared_scalar_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch], DType::F32)?)
            } else {
                None
            },
            moe_shared_gate_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(
                    &[max_batch * config.shared_expert_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_shared_up_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(
                    &[max_batch * config.shared_expert_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_shared_rot_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(
                    &[max_batch * config.shared_expert_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_topk_indices_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch * config.num_experts_per_tok], DType::F32)?)
            } else {
                None
            },
            moe_topk_weights_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[max_batch * config.num_experts_per_tok], DType::F32)?)
            } else {
                None
            },
            moe_gate_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(
                    &[max_batch * config.num_experts_per_tok * config.moe_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_up_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(
                    &[max_batch * config.num_experts_per_tok * config.moe_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_rot_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(
                    &[max_batch * config.num_experts_per_tok * config.moe_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_down_expanded_batch: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(
                    &[max_batch * config.num_experts_per_tok * config.dim],
                    DType::F32,
                )?)
            } else {
                None
            },
            // Path 2 scatter + grouped-WMMA-GEMM scratch (gated at runtime by
            // HIPFIRE_MOE_GROUPED_GEMM=1). m_total_max = N*K_TOP + E*(BLOCK_M-1).
            // i32 buffers stored as Raw (4 bytes/elem matches; no DType::I32 yet).
            moe_expert_token_counts: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[config.num_experts * 4], DType::Raw)?)
            } else {
                None
            },
            moe_expert_offsets: if config.num_experts > 0 {
                Some(gpu.alloc_tensor(&[(config.num_experts + 1) * 4], DType::Raw)?)
            } else {
                None
            },
            moe_sorted_slot_index: if config.num_experts > 0 {
                let m_total_max = moe_grouped_m_total_max(
                    max_batch,
                    config.num_experts_per_tok,
                    config.num_experts,
                );
                Some(gpu.alloc_tensor(&[m_total_max * 4], DType::Raw)?)
            } else {
                None
            },
            moe_inverse_perm: if config.num_experts > 0 {
                let total_slots_max = max_batch * config.num_experts_per_tok;
                Some(gpu.alloc_tensor(&[total_slots_max * 4], DType::Raw)?)
            } else {
                None
            },
            moe_expert_tile_ids: if config.num_experts > 0 {
                let m_total_max = moe_grouped_m_total_max(
                    max_batch,
                    config.num_experts_per_tok,
                    config.num_experts,
                );
                Some(gpu.alloc_tensor(&[(m_total_max / MOE_GROUPED_BLOCK_M) * 4], DType::Raw)?)
            } else {
                None
            },
            moe_y_gate_up_grouped: if config.num_experts > 0 {
                let m_total_max = moe_grouped_m_total_max(
                    max_batch,
                    config.num_experts_per_tok,
                    config.num_experts,
                );
                Some(gpu.alloc_tensor(
                    &[m_total_max * 2 * config.moe_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_y_down_grouped: if config.num_experts > 0 {
                let m_total_max = moe_grouped_m_total_max(
                    max_batch,
                    config.num_experts_per_tok,
                    config.num_experts,
                );
                Some(gpu.alloc_tensor(&[m_total_max * config.dim], DType::F32)?)
            } else {
                None
            },
            dn_s_tape_q8: if config.linear_num_value_heads > 0 {
                let bytes = max_batch
                    * config.linear_num_value_heads
                    * config.linear_value_head_dim
                    * config.linear_value_head_dim;
                Some(gpu.alloc_tensor(&[bytes], DType::Raw)?)
            } else {
                None
            },
            dn_s_tape_scales: if config.linear_num_value_heads > 0 {
                Some(gpu.alloc_tensor(
                    &[max_batch * config.linear_num_value_heads * config.linear_value_head_dim],
                    DType::F32,
                )?)
            } else {
                None
            },
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.x_batch,
            self.x_rot_batch,
            self.x_norm_batch,
            self.dn_qkv_batch,
            self.dn_z_batch,
            self.dn_alpha_batch,
            self.dn_beta_batch,
            self.dn_q_raw_batch,
            self.dn_k_raw_batch,
            self.dn_v_batch,
            self.dn_q_batch,
            self.dn_k_batch,
            self.dn_attn_out_batch,
            self.dn_normed_batch,
            self.gate_ffn_batch,
            self.up_batch,
            self.ffn_hidden_batch,
            self.dn_normed_rot_batch,
            self.positions,
            self.tokens,
            self.fa_q_full_batch,
            self.fa_q_batch,
            self.fa_gate_batch,
            self.fa_k_batch,
            self.fa_v_batch,
            self.fa_attn_out_batch,
            self.fa_attn_out_rot_batch,
        ] {
            let _ = gpu.free_tensor(t);
        }
        for t in [
            self.moe_router_logits_batch,
            self.moe_shared_scalar_batch,
            self.moe_shared_gate_batch,
            self.moe_shared_up_batch,
            self.moe_shared_rot_batch,
            self.moe_topk_indices_batch,
            self.moe_topk_weights_batch,
            self.moe_gate_batch,
            self.moe_up_batch,
            self.moe_rot_batch,
            self.moe_down_expanded_batch,
            self.moe_expert_token_counts,
            self.moe_expert_offsets,
            self.moe_sorted_slot_index,
            self.moe_inverse_perm,
            self.moe_expert_tile_ids,
            self.moe_y_gate_up_grouped,
            self.moe_y_down_grouped,
            self.dn_s_tape_q8,
            self.dn_s_tape_scales,
        ]
        .into_iter()
        .flatten()
        {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Batched prefill entry point: processes N prompt tokens in one call,
/// writing the last token's logits into `scratch.logits` and leaving
/// the KV cache + DeltaNet state advanced by N positions.
///
/// Takes the batched kernel path when ALL linear-attention layer weights
/// are MQ4G256 (the batched element-wise kernels are MQ-specific).
/// Otherwise falls back to a per-token loop over `forward_scratch` that's
/// byte-identical to decode. FA layers always use a per-token gather/scatter
/// fallback — the FA causal attention kernel can't yet be batched (task #71).
///
/// `gated_delta_net_q8_batch_seq` runs one launch per LA layer; the kernel
/// loops over the N tokens internally and requants the Q8 state after every
/// token, matching the decode requant cadence (distributionally equivalent to
/// decode, not byte-identical — the stochastic-rounding frame differs).
///
/// `tokens`: slice of prompt tokens to prefill in order.
/// `start_pos`: first KV cache / DeltaNet position to write. Positions
/// `start_pos .. start_pos + tokens.len()` get populated.
/// On return, `scratch.logits` holds the logits for the *last* token
/// (position `start_pos + tokens.len() - 1`).
///
/// `hidden_rb`: if `Some`, post-layer residual hidden states are captured
/// into the ring buffer for the configured extract layers. Used by the
/// DFlash target-side verify path to batch `verify_dflash_block` into a
/// single forward launch (MVP does B per-token forwards — 88 ms on 4B;
/// this path drops it to ~40 ms with batched forward, further improvement
/// possible with batched lm_head). The per-token fallback also honors it,
/// so the fast-path eligibility doesn't change behavior.
///
/// `per_token_hidden_out`: if `Some`, writes post-output-norm hidden state
/// for each of the N tokens into the provided [N × dim] buffer. The caller
/// then loops `weight_gemv(weights.output, hidden_row, logits)` to recover
/// per-token logits. Required for DFlash verify (needs all B positions'
/// logits, not just the last). `None` preserves the existing "last token
/// only" semantics where logits land in `scratch.logits`.
///
/// `gdn_tape`: if `Some`, captures the post-processed `(q, k, v, α, β)` for
/// every DN (LinearAttention) layer and block position BEFORE the batched
/// `gated_delta_net_q8_batch_seq` call. Enables the DFlash rollback path
/// to replay GDN recurrence from a pre-verify S-state snapshot for
/// `accept_len + 1` steps — no full-target re-run needed.
#[allow(clippy::too_many_arguments)]
/// Upper bound on `forward_prefill_batch`'s per-chunk size. Exposed so
/// callers sizing `HiddenStateRingBuffer` staging can match the chunk
/// upper bound (staging that's smaller than a chunk will assert-fail
/// on prompt seeding of long prompts).
pub const PREFILL_MAX_BATCH: usize = 256;

pub(crate) const MOE_GROUPED_BLOCK_M: usize = 16;

#[inline]
pub(crate) fn prefill_should_emit_last_token_logits(
    has_per_token_hidden_out: bool,
    needs_last_token_logits: bool,
) -> bool {
    !has_per_token_hidden_out || needs_last_token_logits
}

#[inline]
pub(crate) fn align_up_usize(x: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (x + align - 1) & !(align - 1)
}

#[inline]
pub(crate) fn moe_grouped_m_total_max(max_batch: usize, k_top: usize, n_exp: usize) -> usize {
    // Every grouped-GEMM tile consumes 16 sorted slots. The scatter kernel
    // initializes sentinel tile ids up to this bound, so the bound itself must
    // be tile-aligned; otherwise the final launched tile can read an
    // uninitialized expert id.
    align_up_usize(
        max_batch * k_top + n_exp * (MOE_GROUPED_BLOCK_M - 1),
        MOE_GROUPED_BLOCK_M,
    )
}

#[inline]
pub(crate) fn moe_grouped_m_total_bound(total_slots: usize, n_exp: usize) -> usize {
    // Actual grouped rows are sum_e align_up(count_e, BLOCK_M). Only experts
    // that receive at least one slot can contribute padding, so small verify
    // batches do not need to launch the full all-experts worst case.
    let live_expert_bound = total_slots.min(n_exp);
    align_up_usize(
        total_slots + live_expert_bound * (MOE_GROUPED_BLOCK_M - 1),
        MOE_GROUPED_BLOCK_M,
    )
}

#[inline]
pub(crate) fn qwen35_f16_prefill_wmma_enabled(gpu: &Gpu) -> bool {
    gpu.arch_caps.has_wmma()
}

fn kld_direct_f16kv_attention_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let value = std::env::var("HIPFIRE_KLD_DIRECT_WMMA_ATTN")
            .or_else(|_| std::env::var("HIPFIRE_KLD_DIRECT_F16KV_ATTN"))
            .ok();
        matches!(
            value.as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
        )
    })
}

pub(crate) fn kld_direct_f16kv_attention_eligible(
    gpu: &Gpu,
    kv_cache: &kv::KvCache,
    config: &Qwen35Config,
    start_pos: usize,
    tree_verify: Option<&TreeVerifyCtx<'_>>,
) -> bool {
    let enabled = kld_direct_f16kv_attention_enabled();
    let eligible = enabled
        && start_pos == 0
        && kv_cache.compact_offset == 0
        && tree_verify.is_none()
        && !kv_cache.quant_q8
        && !kv_cache.quant_asym2
        && !kv_cache.quant_asym3
        && !kv_cache.quant_asym4
        && config.head_dim.is_multiple_of(16)
        && config.head_dim <= 256
        && gpu.arch_caps.has_wmma();
    if enabled && !eligible {
        static LOGGED: OnceLock<()> = OnceLock::new();
        LOGGED.get_or_init(|| {
            eprintln!(
                "HIPFIRE_KLD_DIRECT_WMMA_ATTN=1 but direct attention is ineligible: \
                 start_pos={} compact_offset={} tree={} quant_q8={} asym2={} asym3={} asym4={} \
                 head_dim={} has_wmma={}",
                start_pos,
                kv_cache.compact_offset,
                tree_verify.is_some(),
                kv_cache.quant_q8,
                kv_cache.quant_asym2,
                kv_cache.quant_asym3,
                kv_cache.quant_asym4,
                config.head_dim,
                gpu.arch_caps.has_wmma(),
            );
        });
    }
    eligible
}

fn kld_fp32_gqa4_attention_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let value = std::env::var("HIPFIRE_KLD_FP32_GQA4_ATTN").ok();
        !matches!(
            value.as_deref(),
            Some("0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO")
        )
    })
}

pub(crate) fn q8_fa_attention_row_loop_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HIPFIRE_Q8_FA_ATTENTION_ROW_LOOP")
                .ok()
                .as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
        )
    })
}

pub(crate) fn q8_fa_attention_scalar_loop_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HIPFIRE_Q8_FA_ATTENTION_SCALAR_LOOP")
                .ok()
                .as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
        )
    })
}

pub(crate) fn q8_fa_attention_serial_kv_loop_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP")
                .ok()
                .as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
        )
    })
}

pub(crate) fn q8_fa_attention_ignore_tree_bias_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HIPFIRE_Q8_FA_ATTENTION_IGNORE_TREE_BIAS")
                .ok()
                .as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
        )
    })
}

pub(crate) fn q8_gdn_verify_per_token_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HIPFIRE_Q8_GDN_VERIFY_PER_TOKEN")
                .ok()
                .as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
        )
    })
}

pub(crate) fn q8_gdn_verify_serial_frames_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HIPFIRE_Q8_GDN_VERIFY_SERIAL_FRAMES")
                .ok()
                .as_deref(),
            Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
        )
    })
}

pub(crate) fn kld_fp32_gqa4_attention_eligible(
    gpu: &Gpu,
    kv_cache: &kv::KvCache,
    config: &Qwen35Config,
    start_pos: usize,
    tree_verify: Option<&TreeVerifyCtx<'_>>,
    batch_len: usize,
) -> bool {
    let kv_group = if config.n_kv_heads == 0 {
        0
    } else {
        config.n_heads / config.n_kv_heads
    };
    let block_size = batch_len.max(config.head_dim).next_power_of_two().min(256);
    let shared_mem = (4usize * batch_len + 4usize * block_size + 4usize * config.head_dim) * 4usize;
    kld_fp32_gqa4_attention_enabled()
        && gpu.arch == "gfx1151"
        && start_pos == 0
        && kv_cache.compact_offset == 0
        && tree_verify.is_none()
        && !kv_cache.quant_q8
        && !kv_cache.quant_asym2
        && !kv_cache.quant_asym3
        && !kv_cache.quant_asym4
        && config.n_kv_heads > 0
        && config.n_heads.is_multiple_of(config.n_kv_heads)
        && kv_group >= 4
        && kv_group % 4 == 0
        && shared_mem <= 64 * 1024
}

fn gemm_f16_x_f32_wmma_residual_batched(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x: &GpuTensor,
    y_residual: &GpuTensor,
    scratch: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    let y_n = y_residual.sub_offset(0, n * m);
    let scratch_n = scratch.sub_offset(0, n * m);
    gpu.gemm_f16_x_f32_wmma(weight, x, &scratch_n, m, k, n)?;
    gpu.add_inplace_f32(&y_n, &scratch_n)
}

pub(crate) fn gemm_f32_residual_batched(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x: &GpuTensor,
    y_residual: &GpuTensor,
    scratch: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    let y_n = y_residual.sub_offset(0, n * m);
    let scratch_n = scratch.sub_offset(0, n * m);
    gpu.gemm_f32_register_tiled(weight, x, &scratch_n, m, k, n)?;
    gpu.add_inplace_f32(&y_n, &scratch_n)
}

fn gemm_bf16_x_bf16_wmma_residual_batched(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x: &GpuTensor,
    y_residual: &GpuTensor,
    scratch: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    let y_n = y_residual.sub_offset(0, n * m);
    let scratch_n = scratch.sub_offset(0, n * m);
    gpu.gemm_bf16_x_bf16_wmma(weight, x, &scratch_n, m, k, n)?;
    gpu.add_inplace_f32(&y_n, &scratch_n)
}

pub(crate) fn gemm_fp16_or_bf16_x_f32_wmma(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    match weight.dtype {
        DType::F16 | DType::Raw => gpu.gemm_f16_x_f32_wmma(weight, x, y, m, k, n),
        DType::BF16 => gpu.gemm_bf16_x_bf16_wmma(weight, x, y, m, k, n),
        other => panic!("expected F16/BF16 prefill weight, got {other:?}"),
    }
}

pub(crate) fn gemm_fp16_or_bf16_x_f32_wmma_residual_batched(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x: &GpuTensor,
    y_residual: &GpuTensor,
    scratch: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    match weight.dtype {
        DType::F16 | DType::Raw => {
            gemm_f16_x_f32_wmma_residual_batched(gpu, weight, x, y_residual, scratch, m, k, n)
        }
        DType::BF16 => {
            gemm_bf16_x_bf16_wmma_residual_batched(gpu, weight, x, y_residual, scratch, m, k, n)
        }
        other => panic!("expected F16/BF16 residual prefill weight, got {other:?}"),
    }
}

// Batched single-projection GEMM for the dense fused prefill. Despite the
// `_full_precision` name (kept to avoid churning ~10 call sites), this now
// dispatches plain Q8_0 and MQ4G256 weights too — quantized dense models route
// here. MQ4 needs the shared FWHT pre-rotation (rotate_x_mq_batched) into a
// scratch first; the rotation is allocated internally (prefill is one-shot, so
// the small per-GEMM alloc is acceptable). MQ6G256 non-residual has no batched
// kernel yet, so those weights fall back to serial_reference via the contract.
fn dense_session_prefill_gemm_full_precision(
    gpu: &mut Gpu,
    weight: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    match weight.gpu_dtype {
        DType::F32 => gpu.gemm_f32_register_tiled(&weight.buf, x, y, weight.m, weight.k, n),
        DType::F16 | DType::BF16 | DType::Raw => {
            gemm_fp16_or_bf16_x_f32_wmma(gpu, &weight.buf, x, y, weight.m, weight.k, n)
        }
        DType::Q8_0 => gpu.gemm_q8_0_batched_chunked(&weight.buf, x, y, weight.m, weight.k, n),
        DType::MQ4G256 => {
            let rot = gpu.alloc_tensor(&[n * weight.k], DType::F32)?;
            gpu.rotate_x_mq_batched(x, &rot, weight.k, n)?;
            let result = gpu.gemm_hfq4g256(&weight.buf, &rot, y, weight.m, weight.k, n);
            let _ = gpu.free_tensor(rot);
            result
        }
        other => Err(hip_bridge::HipError::new(
            0,
            &format!("dense session fused prefix GEMM does not support dtype {other:?}"),
        )),
    }
}

// Residual variant of [`dense_session_prefill_gemm_full_precision`] (adds the
// GEMM result into `y_residual`). Also dispatches plain Q8_0 + MQ4G256 now: Q8
// runs the chunked GEMM into `scratch` then adds it into the residual; MQ4 runs
// the FWHT-rotated `gemm_hfq4g256_residual` which accumulates directly. MQ6G256
// residual has a kernel (`gemm_hfq6g256_residual`) but is left for a follow-up so
// the contract gates MQ6 to serial uniformly with the non-residual path.
fn dense_session_prefill_gemm_full_precision_residual(
    gpu: &mut Gpu,
    weight: &WeightTensor,
    x: &GpuTensor,
    y_residual: &GpuTensor,
    scratch: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    match weight.gpu_dtype {
        DType::F32 => gemm_f32_residual_batched(
            gpu,
            &weight.buf,
            x,
            y_residual,
            scratch,
            weight.m,
            weight.k,
            n,
        ),
        DType::F16 | DType::BF16 | DType::Raw => gemm_fp16_or_bf16_x_f32_wmma_residual_batched(
            gpu,
            &weight.buf,
            x,
            y_residual,
            scratch,
            weight.m,
            weight.k,
            n,
        ),
        DType::Q8_0 => {
            let out = scratch.sub_offset(0, n * weight.m);
            gpu.gemm_q8_0_batched_chunked(&weight.buf, x, &out, weight.m, weight.k, n)?;
            let accum = y_residual.sub_offset(0, n * weight.m);
            gpu.add_inplace_f32(&accum, &out)
        }
        DType::MQ4G256 => {
            let rot = gpu.alloc_tensor(&[n * weight.k], DType::F32)?;
            gpu.rotate_x_mq_batched(x, &rot, weight.k, n)?;
            let result =
                gpu.gemm_hfq4g256_residual(&weight.buf, &rot, y_residual, weight.m, weight.k, n);
            let _ = gpu.free_tensor(rot);
            result
        }
        other => Err(hip_bridge::HipError::new(
            0,
            &format!("dense session fused prefix residual GEMM does not support dtype {other:?}"),
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MoeGroupedPath2Shape {
    pub(crate) total_slots: usize,
    pub(crate) m_total_bound: usize,
    pub(crate) gate_up_x_row_div: usize,
    pub(crate) gate_up_source_rows: usize,
    pub(crate) down_x_row_div: usize,
    pub(crate) down_source_rows: usize,
}

#[inline]
pub(crate) fn moe_grouped_path2_shape(
    n: usize,
    k_top: usize,
    n_exp: usize,
) -> MoeGroupedPath2Shape {
    let total_slots = n * k_top;
    MoeGroupedPath2Shape {
        total_slots,
        m_total_bound: moe_grouped_m_total_bound(total_slots, n_exp),
        // Gate/up consumes x_rot_batch [N x dim]. The grouped scatter's
        // sorted slot encodes token*K_TOP + expert-rank, so the kernel divides
        // source-row lookup by K_TOP to recover the token row.
        gate_up_x_row_div: k_top,
        gate_up_source_rows: n,
        // Down consumes rot_batch [N*K_TOP x mi]. Sorted slots already index
        // the flattened routed-expert rows, so no division is required.
        down_x_row_div: 1,
        down_source_rows: total_slots,
    }
}

pub(crate) struct PagedMoeExpertBucket {
    pub(crate) expert: u16,
    pub(crate) m_total: usize,
    pub(crate) sorted_slot_index: Vec<i32>,
    pub(crate) inverse_perm: Vec<i32>,
    pub(crate) expert_tile_ids: Vec<i32>,
}

pub(crate) fn build_paged_moe_expert_buckets(
    topk_indices: &[usize],
    n: usize,
    k_top: usize,
    n_exp: usize,
) -> HipResult<Vec<PagedMoeExpertBucket>> {
    let total_slots = n
        .checked_mul(k_top)
        .ok_or_else(|| HipError::new(0, "paged MoE expert bucket total_slots overflow"))?;
    if topk_indices.len() != total_slots {
        return Err(HipError::new(
            0,
            &format!(
                "paged MoE expert bucket topk length mismatch: got {}, expected {}",
                topk_indices.len(),
                total_slots
            ),
        ));
    }
    let mut slots_by_expert = vec![Vec::<i32>::new(); n_exp];
    for (flat, &expert) in topk_indices.iter().enumerate() {
        if expert >= n_exp {
            return Err(HipError::new(
                0,
                &format!("paged MoE router selected expert {expert}, but n_exp={n_exp}"),
            ));
        }
        slots_by_expert[expert].push(flat as i32);
    }

    let mut buckets = Vec::new();
    for (expert, slots) in slots_by_expert.into_iter().enumerate() {
        if slots.is_empty() {
            continue;
        }
        let m_total = align_up_usize(slots.len(), MOE_GROUPED_BLOCK_M);
        let mut sorted_slot_index = vec![-1i32; m_total];
        sorted_slot_index[..slots.len()].copy_from_slice(&slots);
        let mut inverse_perm = vec![-1i32; total_slots];
        for (sorted_pos, &flat) in slots.iter().enumerate() {
            inverse_perm[flat as usize] = sorted_pos as i32;
        }
        let expert_tile_ids = vec![expert as i32; m_total / MOE_GROUPED_BLOCK_M];
        buckets.push(PagedMoeExpertBucket {
            expert: expert as u16,
            m_total,
            sorted_slot_index,
            inverse_perm,
            expert_tile_ids,
        });
    }
    Ok(buckets)
}

pub(crate) fn upload_paged_moe_expert_bucket(
    gpu: &mut Gpu,
    bucket: &PagedMoeExpertBucket,
    sorted_slot_index: &GpuTensor,
    inverse_perm: &GpuTensor,
    expert_tile_ids: &GpuTensor,
) -> HipResult<()> {
    gpu.hip.memcpy_htod(
        &sorted_slot_index.buf,
        i32_slice_as_bytes(&bucket.sorted_slot_index),
    )?;
    gpu.hip
        .memcpy_htod(&inverse_perm.buf, i32_slice_as_bytes(&bucket.inverse_perm))?;
    gpu.hip.memcpy_htod(
        &expert_tile_ids.buf,
        i32_slice_as_bytes(&bucket.expert_tile_ids),
    )?;
    Ok(())
}

/// Host-side helper: upload token ids and positions to a `PrefillBatchScratch`
/// via sync `memcpy_htod`. Call this BEFORE entering a hipGraph capture to
/// pre-populate `pbs.tokens` and `pbs.positions`, then pass `pre_uploaded:
/// true` (or use `forward_prefill_chunk_captured_safe`) so the forward
/// does not issue any additional uploads inside the captured region.
pub fn upload_prefill_batch_inputs(
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    tokens: &[u32],
    start_pos: usize,
) -> HipResult<()> {
    let n = tokens.len();
    let positions_host: Vec<usize> = (0..n).map(|i| start_pos + i).collect();
    upload_prefill_batch_inputs_with_positions(gpu, pbs, tokens, &positions_host)
}

pub fn upload_prefill_batch_inputs_with_positions(
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    tokens: &[u32],
    positions: &[usize],
) -> HipResult<()> {
    let n = tokens.len();
    if positions.len() != n {
        return Err(hip_bridge::HipError::new(
            0,
            "upload_prefill_batch_inputs_with_positions: tokens and positions length mismatch",
        ));
    }
    if n > pbs.max_batch {
        return Err(hip_bridge::HipError::new(
            0,
            "upload_prefill_batch_inputs_with_positions: token count exceeds PrefillBatchScratch max_batch",
        ));
    }
    let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let tokens_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
    let positions_host: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
    let positions_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
    Ok(())
}

/// Capture-friendly entry point that runs the batched forward against a
/// SINGLE chunk (`tokens.len() <= pbs.max_batch`), skipping the internal
/// token/position upload and assuming the caller has already populated
/// `pbs.tokens` / `pbs.positions` via `upload_prefill_batch_inputs`.
///
/// This exists so `hipStreamBeginCapture` can wrap the forward without
/// the per-call `memcpy_htod` sync operations (which would either error
/// under capture or bake stale host data into the captured graph nodes).
///
/// Callers still must handle `hidden_rb.commit_staging_to_ring(gpu, n)`
/// AFTER the forward returns (outside any captured region) to scatter
/// staging writes to the ring at the current head.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_single_chunk_captured(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    hidden_rb: Option<&HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
) -> HipResult<()> {
    forward_prefill_batch_single_chunk_captured_opts(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        pbs,
        hidden_rb,
        per_token_hidden_out,
        gdn_tape,
        tree_verify,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_single_chunk_captured_opts(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    hidden_rb: Option<&HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    needs_last_token_logits: bool,
) -> HipResult<()> {
    let n = tokens.len();
    debug_assert!(
        n > 0 && n <= pbs.max_batch,
        "single_chunk_captured: n={} but pbs.max_batch={}",
        n,
        pbs.max_batch
    );

    // Defense-in-depth: this entry point bypasses the eligibility check
    // in `forward_prefill_batch_with_pbs`, so the caller is responsible
    // for ensuring the batched fast-path is valid. Two structural bypasses
    // could land here:
    //   1. MQ3-weighted dense model on an arch that lacks the gfx11 wave32
    //      WMMA builtin.
    //   2. MQ3 weights inside a MoE/A3B layer on an arch without the gfx12
    //      grouped-MoE HFQ3 kernels, or MQ3-Lloyd MoE, which is still unwired.
    // In production, `daemon.rs`'s DFlash refusal guard blocks both, but
    // dflash_spec_demo and other example callers go through ModelSlot::load
    // directly. We cross-check here so any caller is protected.
    let arch = gpu.arch.as_str();
    let mut mq3_in_dense = false;
    let mut mq3_in_moe = false;
    let mut lloyd_in_dense = false;
    let mut lloyd_in_moe = false;
    // The Lloyd dtype is treated identically to plain MQ3 in this guard:
    // both use 112-vs-104-byte stride that the MoE batched branches'
    // HFQ4-layout dispatch would corrupt, and both depend on the gfx11/12
    // WMMA family that other archs lack. Add Lloyd alongside MQ3 so the
    // refusal fires symmetrically and a future MQ3-Lloyd MoE model can't
    // silently land here without explicit MoE-Lloyd kernels.
    //
    // We also track `lloyd_in_dense` separately because Lloyd-MQ3 on
    // gfx12 ships behind an opt-in env gate (see is_batchable_la above) —
    // the gfx12 sibling kernels are runtime-unvalidated locally, so by
    // default a captured-path call with Lloyd-MQ3 weights on gfx1200/1201
    // must refuse rather than dispatch to an untested kernel.
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    let is_lloyd = |dt: DType| matches!(dt, DType::MQ3G256Lloyd);
    for lw in &weights.layers {
        match lw {
            LayerWeights::DeltaNet(l) => {
                if is_mq3_any(l.wqkv.gpu_dtype)
                    || is_mq3_any(l.wz.gpu_dtype)
                    || is_mq3_any(l.w_beta.gpu_dtype)
                    || is_mq3_any(l.w_alpha.gpu_dtype)
                    || is_mq3_any(l.wo.gpu_dtype)
                    || is_mq3_any(l.w_gate.gpu_dtype)
                    || is_mq3_any(l.w_up.gpu_dtype)
                    || is_mq3_any(l.w_down.gpu_dtype)
                {
                    mq3_in_dense = true;
                }
                if is_lloyd(l.wqkv.gpu_dtype)
                    || is_lloyd(l.wz.gpu_dtype)
                    || is_lloyd(l.w_beta.gpu_dtype)
                    || is_lloyd(l.w_alpha.gpu_dtype)
                    || is_lloyd(l.wo.gpu_dtype)
                    || is_lloyd(l.w_gate.gpu_dtype)
                    || is_lloyd(l.w_up.gpu_dtype)
                    || is_lloyd(l.w_down.gpu_dtype)
                {
                    lloyd_in_dense = true;
                }
            }
            LayerWeights::FullAttn(l) => {
                if is_mq3_any(l.wq.gpu_dtype)
                    || is_mq3_any(l.wk.gpu_dtype)
                    || is_mq3_any(l.wv.gpu_dtype)
                    || is_mq3_any(l.wo.gpu_dtype)
                    || is_mq3_any(l.w_gate.gpu_dtype)
                    || is_mq3_any(l.w_up.gpu_dtype)
                    || is_mq3_any(l.w_down.gpu_dtype)
                {
                    mq3_in_dense = true;
                }
                if is_lloyd(l.wq.gpu_dtype)
                    || is_lloyd(l.wk.gpu_dtype)
                    || is_lloyd(l.wv.gpu_dtype)
                    || is_lloyd(l.wo.gpu_dtype)
                    || is_lloyd(l.w_gate.gpu_dtype)
                    || is_lloyd(l.w_up.gpu_dtype)
                    || is_lloyd(l.w_down.gpu_dtype)
                {
                    lloyd_in_dense = true;
                }
            }
            LayerWeights::DeltaNetMoe(l) => {
                if is_mq3_any(l.wqkv.gpu_dtype)
                    || is_mq3_any(l.wz.gpu_dtype)
                    || is_mq3_any(l.w_beta.gpu_dtype)
                    || is_mq3_any(l.w_alpha.gpu_dtype)
                    || is_mq3_any(l.wo.gpu_dtype)
                    || moe_ffn_has_mq3(&l.ffn)
                {
                    mq3_in_moe = true;
                }
                if is_lloyd(l.wqkv.gpu_dtype)
                    || is_lloyd(l.wz.gpu_dtype)
                    || is_lloyd(l.w_beta.gpu_dtype)
                    || is_lloyd(l.w_alpha.gpu_dtype)
                    || is_lloyd(l.wo.gpu_dtype)
                    || moe_ffn_has_mq3_lloyd(&l.ffn)
                {
                    lloyd_in_moe = true;
                }
            }
            LayerWeights::FullAttnMoe(l) => {
                if is_mq3_any(l.wq.gpu_dtype)
                    || is_mq3_any(l.wk.gpu_dtype)
                    || is_mq3_any(l.wv.gpu_dtype)
                    || is_mq3_any(l.wo.gpu_dtype)
                    || moe_ffn_has_mq3(&l.ffn)
                {
                    mq3_in_moe = true;
                }
                if is_lloyd(l.wq.gpu_dtype)
                    || is_lloyd(l.wk.gpu_dtype)
                    || is_lloyd(l.wv.gpu_dtype)
                    || is_lloyd(l.wo.gpu_dtype)
                    || moe_ffn_has_mq3_lloyd(&l.ffn)
                {
                    lloyd_in_moe = true;
                }
            }
        }
    }
    let arch_has_wmma = matches!(
        arch,
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" | "gfx1200" | "gfx1201"
    );
    let mq3_moe_supported = arch == "gfx1151" || (arch.starts_with("gfx12") && !lloyd_in_moe);
    if mq3_in_moe && !mq3_moe_supported {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_prefill_batch_single_chunk_captured: model has MQ3G256 / \
             MQ3G256Lloyd weights inside a MoE/A3B layer (DeltaNetMoe or \
             FullAttnMoe), but MQ3-Lloyd MoE is only wired on gfx1151, and plain \
             MQ3G256 MoE is wired on gfx1151/gfx12. Use MQ4/MQ6 for other targets.",
        ));
    }
    if mq3_in_dense && !arch_has_wmma {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "forward_prefill_batch_single_chunk_captured: model contains MQ3G256 \
             weights but arch {arch} lacks the gfx11 wave32 WMMA builtin. The MQ3 \
             prefill kernels (gemm_*_hfq3g256_wmma) only compile on \
             gfx1100/1101/1102/1150/1151. Caller must use the non-captured \
             forward_prefill_batch path (which falls back to per-token \
             forward_scratch on this arch). gfx12 K4 variant for MQ3 is \
             a planned follow-up."
            ),
        ));
    }
    // Lloyd-MQ3 on gfx12 is opt-in (see is_batchable_la's gate). The
    // captured entry point bypasses is_batchable_la, so we replicate the
    // gate here: refuse Lloyd-on-gfx12 unless HIPFIRE_LLOYD_GFX12=1 is set.
    // Without this guard, a captured call would reach the dispatch arms
    // and try to load gfx12 kernels that are still community-CI-pending.
    let arch_is_gfx12 = matches!(arch, "gfx1200" | "gfx1201");
    let lloyd_gfx12_optin = std::env::var("HIPFIRE_LLOYD_GFX12").ok().as_deref() == Some("1");
    if lloyd_in_dense && arch_is_gfx12 && !lloyd_gfx12_optin {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "forward_prefill_batch_single_chunk_captured: model contains \
             MQ3G256Lloyd weights on arch {arch}, but the gfx12 (RDNA4) \
             sibling kernels (gemm_*_mq3g256_lloyd_wmma.gfx12.hip) are \
             runtime-unvalidated locally and ship behind an opt-in gate. \
             Set HIPFIRE_LLOYD_GFX12=1 to enable the gfx12 path for parity \
             testing, or use the non-captured forward_prefill_batch path \
             (which falls back to per-token forward_scratch on this arch \
             when the env var is unset)."
            ),
        ));
    }

    // Capture-mode contract: under hipStreamBeginCapture, the FA branch
    // bakes max_ctx_len = kv_cache.physical_cap (kernels read seq_len
    // per-row from a device buffer, but LDS is sized from this scalar).
    // For Q8 KV at physical_cap > 15000, the FA path enters the per-
    // position long-context fallback, which issues hip.malloc + per-row
    // memcpy_htod inside the layer loop. Both are capture-illegal — they
    // would either error at capture time or bake stale host bytes into
    // the kernarg blob. Asym2/3/4 KV use pure-batched flash kernels and
    // stay capture-safe at any context length, so reject only this exact
    // combination here.
    const LDS_CTX_LIMIT: usize = 15000;
    if kv_cache.quant_q8
        && !(kv_cache.quant_asym2 || kv_cache.quant_asym3 || kv_cache.quant_asym4)
        && kv_cache.physical_cap > LDS_CTX_LIMIT
    {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "forward_prefill_batch_single_chunk_captured: Q8 KV with \
             physical_cap {} > {} hits the per-position long-context \
             fallback, which issues hip.malloc + memcpy_htod inside the \
             captured region. Use asym3 KV for capture at long context, \
             or shrink physical_cap.",
                kv_cache.physical_cap, LDS_CTX_LIMIT,
            ),
        ));
    }

    let debug_max_layer = std::env::var("HIPFIRE_PREFILL_MAX_LAYER")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());

    forward_prefill_chunk(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        pbs,
        hidden_rb,
        per_token_hidden_out.map(|t| (t, 0)),
        gdn_tape,
        0,
        tree_verify,
        true, // pre_uploaded: caller must have run upload_prefill_batch_inputs
        None, // band: full-stack single-GPU path
        None, // mask_override: captured-prefill caller does not use the MTP probe hook
        None, // positions_override: captured-prefill uses linear positions
        needs_last_token_logits,
        debug_max_layer, // max_layer: default full stack; env is for graph-fault bisection only
        false,           // force_q8_gdn_per_token: captured verify preserves production policy
        None,            // routed_out: non-EP single-GPU path
    )
}

pub fn forward_prefill_batch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
) -> HipResult<()> {
    forward_prefill_batch_with_pbs(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        hidden_rb,
        per_token_hidden_out,
        gdn_tape,
        tree_verify,
        scratch.prefill_batch.as_ref(),
        None, // mask_override: MTP probe is the only consumer; default callers don't override
        None, // max_layer: pflash uses this; non-pflash default is full stack
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch_force_q8_gdn_per_token(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
) -> HipResult<()> {
    forward_prefill_batch_with_pbs_opts(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        hidden_rb,
        per_token_hidden_out,
        gdn_tape,
        tree_verify,
        scratch.prefill_batch.as_ref(),
        None,
        None,
        true,
        true,
    )
}

pub fn forward_prefill_batch_with_pbs(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pbs_in: Option<&PrefillBatchScratch>,
    mask_override: Option<MaskEmbedOverride<'_>>,
    max_layer: Option<usize>,
) -> HipResult<()> {
    forward_prefill_batch_with_pbs_opts(
        gpu,
        weights,
        config,
        tokens,
        start_pos,
        kv_cache,
        dn_state,
        scratch,
        hidden_rb,
        per_token_hidden_out,
        gdn_tape,
        tree_verify,
        pbs_in,
        mask_override,
        max_layer,
        true,  // preserve legacy post-condition: scratch.logits is last-token logits
        false, // force_q8_gdn_per_token: default callers preserve existing policy
    )
}

/// Like `forward_prefill_batch`, but accepts a caller-owned `PrefillBatchScratch`
/// so the ~25 per-cycle tensor allocations can be amortized across many calls.
///
/// `pbs = None` preserves the original behavior (per-call allocate + free);
/// `pbs = Some(&pbs)` reuses the provided scratch. The provided scratch's
/// `max_batch` determines the chunk size — `tokens` is processed in chunks of
/// up to `pbs.max_batch`. Callers driving DFlash verify should size `pbs`
/// to the maximum block size they'll ever request (e.g. `block_size` or
/// `1 + tree_budget`) so everything fits in one chunk.
///
/// `needs_last_token_logits = false` is only for callers that pass
/// `per_token_hidden_out` and compute their own logits from those hidden rows.
/// The default wrapper keeps this true to protect existing callers that rely on
/// `scratch.logits` being populated with the last token's logits.
pub fn forward_prefill_batch_with_pbs_opts(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    mut hidden_rb: Option<&mut HiddenStateRingBuffer>,
    per_token_hidden_out: Option<&GpuTensor>,
    mut gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pbs_in: Option<&PrefillBatchScratch>,
    mask_override: Option<MaskEmbedOverride<'_>>,
    max_layer: Option<usize>,
    needs_last_token_logits: bool,
    force_q8_gdn_per_token: bool,
) -> HipResult<()> {
    // Upper bound on the PrefillBatchScratch — large prompts get split
    // into chunks of this size and processed in a loop.
    //
    // Tuning note: each extra chunk pays full dispatch-overhead for the LA
    // preamble (rmsnorm, rotate, 4-way fused GEMM) and FFN (gate_up + down).
    // 256 costs ~80 MB of scratch on 9B vs 20 MB at 64 — trivial on modern
    // cards — and drops chunk count for pp2048 from 32 → 8. The inner
    // gated_delta_net_q8_batch_seq loop is still sequential per token, so
    // the per-chunk DeltaNet cost is linear in N either way; raising the
    // batch just amortizes the NON-DeltaNet kernels more.
    //
    // Exposed via PREFILL_MAX_BATCH so callers sizing `HiddenStateRingBuffer`
    // staging can match the chunk upper bound.
    let max_batch: usize = std::env::var("HIPFIRE_PREFILL_MAX_BATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v >= MIN_BATCH)
        .unwrap_or(PREFILL_MAX_BATCH);

    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }

    // Cross-path safety: only plain MQ3G256 MoE is admitted, and only on
    // gfx1151/gfx12 where the shared-expert HFQ3 kernels and routed grouped-WMMA
    // kernels are available. MQ3-Lloyd MoE remains rejected because its
    // routed expert kernels have not been wired into qwen35's MoE path.
    let arch = gpu.arch.as_str();
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    let is_lloyd = |dt: DType| matches!(dt, DType::MQ3G256Lloyd);
    let mq3_in_moe = weights.layers.iter().any(|lw| match lw {
        LayerWeights::DeltaNetMoe(l) => {
            is_mq3_any(l.wqkv.gpu_dtype)
                || is_mq3_any(l.wz.gpu_dtype)
                || is_mq3_any(l.w_beta.gpu_dtype)
                || is_mq3_any(l.w_alpha.gpu_dtype)
                || is_mq3_any(l.wo.gpu_dtype)
                || moe_ffn_has_mq3(&l.ffn)
        }
        LayerWeights::FullAttnMoe(l) => {
            is_mq3_any(l.wq.gpu_dtype)
                || is_mq3_any(l.wk.gpu_dtype)
                || is_mq3_any(l.wv.gpu_dtype)
                || is_mq3_any(l.wo.gpu_dtype)
                || moe_ffn_has_mq3(&l.ffn)
        }
        _ => false,
    });
    let lloyd_in_moe = weights.layers.iter().any(|lw| match lw {
        LayerWeights::DeltaNetMoe(l) => {
            is_lloyd(l.wqkv.gpu_dtype)
                || is_lloyd(l.wz.gpu_dtype)
                || is_lloyd(l.w_beta.gpu_dtype)
                || is_lloyd(l.w_alpha.gpu_dtype)
                || is_lloyd(l.wo.gpu_dtype)
                || moe_ffn_has_mq3_lloyd(&l.ffn)
        }
        LayerWeights::FullAttnMoe(l) => {
            is_lloyd(l.wq.gpu_dtype)
                || is_lloyd(l.wk.gpu_dtype)
                || is_lloyd(l.wv.gpu_dtype)
                || is_lloyd(l.wo.gpu_dtype)
                || moe_ffn_has_mq3_lloyd(&l.ffn)
        }
        _ => false,
    });
    let mq3_moe_supported = arch == "gfx1151" || (arch.starts_with("gfx12") && !lloyd_in_moe);
    if mq3_in_moe && !mq3_moe_supported {
        return Err(hip_bridge::HipError::new(
            0,
            "forward_prefill_batch: model has MQ3G256 / MQ3G256Lloyd weights \
             inside a MoE/A3B layer (DeltaNetMoe or FullAttnMoe), but only \
             MQ3-Lloyd MoE is only wired on gfx1151, and plain MQ3G256 MoE is \
             wired on gfx1151/gfx12. Use MQ4/MQ6 for other targets.",
        ));
    }

    // Tree-verify mode sanity checks — the downstream path can't silently
    // fall back to per-token FA (that's always causal and would ignore the
    // tree mask), and the positions/bias shapes must match the token count.
    if let Some(ctx) = tree_verify.as_ref() {
        assert_eq!(
            ctx.positions.len(),
            n,
            "TreeVerifyCtx.positions length {} must equal tokens.len() {}",
            ctx.positions.len(),
            n,
        );
        assert_eq!(
            ctx.attn_bias.numel(),
            n * n,
            "TreeVerifyCtx.attn_bias must be [{} × {}] f32 ({}), got numel {}",
            n,
            n,
            n * n,
            ctx.attn_bias.numel(),
        );
    }

    // Fast path requires (a) every LA layer's weights to be either MQ4G256
    // or HFQ4G256 (the batched GEMM kernels are dtype-agnostic but the LA
    // preamble's rmsnorm+rotate and SwiGLU+rotate kernels differ per dtype),
    // and (b) Q8 S-state for the GDN recurrence. Mixed-dtype layers are
    // allowed; each layer is routed to its own path. HFQ6/others fall back.
    let arch = gpu.arch.as_str();
    // Whether the tape-capturing batched (PBS) path runs for this call — the
    // single source of truth shared with spec-decode callers that later replay a
    // captured GDN tape. On `false` the forward drops to the tape-less per-token
    // loop below, leaving any passed tape stale (see `prefill_batch_pbs_eligible`).
    let moe_router_logits_present = pbs_in
        .map(|p| p.moe_router_logits_batch.is_some())
        .unwrap_or(true);
    let eligible = prefill_batch_pbs_eligible(
        weights,
        config,
        dn_state,
        n,
        arch,
        moe_router_logits_present,
    );
    // F4 guard: reject batched prefill when KV tier has no batched keys.
    // F32 KV has only BatchEq(1) → MissingImpl at resolve. asym2 + tree-verify
    // has no _batched_masked variant → UnsupportedTreeTier. Force per-token
    // fallback for these cases.
    let kv_f32 = !kv_cache.quantized && !kv_cache.quant_q8 && !kv_cache.quant_hfq4;
    let kv_asym2_tree = kv_cache.quant_asym2 && tree_verify.is_some();
    let pbs_eligible_base = eligible;
    let eligible = eligible && !kv_f32 && !kv_asym2_tree;
    if std::env::var("HIPFIRE_DEBUG_PREFILL_ELIGIBLE").as_deref() == Ok("1") {
        eprintln!(
            "[prefill-eligible] final={eligible} base={pbs_eligible_base} kv_f32={kv_f32} \
             kv_asym2_tree={kv_asym2_tree} dn_quant={:?} n={n} arch={arch} \
             kv(q8={} hfq4={} quantized={})",
            dn_state.quant, kv_cache.quant_q8, kv_cache.quant_hfq4, kv_cache.quantized
        );
    }

    if !eligible {
        assert!(
            tree_verify.is_none(),
            "tree-verify mode requires the batched-FA-eligible prefill path; \
             kv quant + FA weight dtypes do not match on this model",
        );
        // mask_override has nowhere to land on the per-token forward_scratch
        // fallback (it operates on `scratch.x`, not the batched `pbs.x_batch`,
        // and there's no shared "post-embed, pre-layer" hook). The MTP probe
        // is the only consumer today and runs on MQ4-quantized models that
        // always satisfy `eligible`, so hard-error rather than silently
        // ignoring the override.
        assert!(
            mask_override.is_none(),
            "MaskEmbedOverride requires the batched prefill path, but this \
             model fell through to the per-token fallback (likely non-MQ4 \
             weights, dn_state quant != Q8, or HIPFIRE_PREFILL_BATCHED=0).",
        );
        // Fallback: per-token loop, byte-identical to decode. If hidden
        // extraction is requested, use the with_hidden variant so the ring
        // buffer still gets populated correctly (each call advances head by 1).
        // When per-token hidden output is also requested, extract post-norm
        // hidden row-by-row into the caller's buffer.
        let dim = config.dim;
        let last_idx = tokens.len().saturating_sub(1);
        for (i, &tok) in tokens.iter().enumerate() {
            // lm_head (vocab-wide logits) only matters for the FINAL prefill
            // token — earlier prompt tokens' logits are never read. Computing it
            // every token was ~37% of prefill time on gfx1103 (rocprof). Skip
            // lm_head for all non-final tokens via the no-logits forward; the
            // last token still gets full logits in scratch.logits.
            let skip_logits = needs_last_token_logits && i != last_idx;
            if let Some(rb) = hidden_rb.as_mut() {
                forward_scratch_with_hidden(
                    gpu,
                    weights,
                    config,
                    tok,
                    start_pos + i,
                    kv_cache,
                    dn_state,
                    scratch,
                    rb,
                )?;
            } else if (per_token_hidden_out.is_some() && !needs_last_token_logits) || skip_logits {
                forward_scratch_no_logits(
                    gpu,
                    weights,
                    config,
                    tok,
                    start_pos + i,
                    kv_cache,
                    dn_state,
                    scratch,
                )?;
            } else {
                forward_scratch(
                    gpu,
                    weights,
                    config,
                    tok,
                    start_pos + i,
                    kv_cache,
                    dn_state,
                    scratch,
                )?;
            }
            if let Some(dst) = per_token_hidden_out {
                // scratch.tmp holds post-output-norm hidden after
                // forward_scratch_{with_hidden,layers} — it's the same buffer
                // lm_head reads from. Copy into the caller's output.
                gpu.hip
                    .memcpy_dtod_at(&dst.buf, i * dim * 4, &scratch.tmp.buf, 0, dim * 4)?;
            }
        }
        return Ok(());
    }

    // Tree-verify mode runs as a single chunk (tree is small, O(16) nodes);
    // chunk splitting would require slicing the mask by chunk rows which
    // is extra work for a case we don't need.
    if tree_verify.is_some() {
        assert!(
            n <= max_batch,
            "tree-verify tokens {} exceeds max_batch {}; tree budget must fit",
            n,
            max_batch,
        );
    }

    // Allocate the batch scratch once per call (or reuse a caller-owned one).
    // When `pbs_in` is Some, we neither allocate nor free — the caller retains
    // ownership across DFlash cycles to avoid ~25 per-cycle tensor alloc/free
    // pairs on the hot verify path. When None we fall back to the original
    // allocate-here / free-on-exit pattern so unmodified callers behave the
    // same. The chunk size is `pbs.max_batch` so a caller-owned scratch sized
    // to e.g. `block_size` or `1 + tree_budget` keeps DFlash verify in one
    // chunk without the full 256-row MAX_BATCH footprint.
    let mut own_pbs: Option<PrefillBatchScratch> = None;
    let result = (|| -> HipResult<()> {
        let pbs: &PrefillBatchScratch = match pbs_in {
            Some(p) => p,
            None => {
                own_pbs = Some(PrefillBatchScratch::new(gpu, config, max_batch)?);
                own_pbs.as_ref().unwrap()
            }
        };
        let chunk_batch = pbs.max_batch;
        let mut chunk_start = 0usize;
        while chunk_start < n {
            let chunk_end = (chunk_start + chunk_batch).min(n);
            let chunk = &tokens[chunk_start..chunk_end];
            let chunk_n = chunk.len();
            // The chunk only reads the ring buffer's head/dims to place its
            // writes. We advance the head AFTER the chunk returns, here in
            // the caller, to keep the mutable borrow scope tight.
            let pth_slot = per_token_hidden_out.map(|t| (t, chunk_start));
            // Reborrow the tape for this chunk so we keep the outer mut
            // after the chunk returns.
            let tape_for_chunk: Option<&mut crate::speculative::GdnTape> = gdn_tape.as_deref_mut();
            // Tree-verify was asserted to fit in one chunk above, so passing
            // the whole ctx through unconditionally is safe.
            let tv_for_chunk = tree_verify.as_ref().copied();
            // Apply mask_override only to the chunk that actually contains
            // its target slot, and rebase the slot index to chunk-local
            // coordinates. Out-of-range slots panic (caller error).
            let mo_for_chunk = mask_override.and_then(|ovr| {
                if ovr.slot >= chunk_start && ovr.slot < chunk_end {
                    Some(MaskEmbedOverride {
                        slot: ovr.slot - chunk_start,
                        embed: ovr.embed,
                    })
                } else {
                    None
                }
            });
            // Sanity: if caller provided an override, it MUST land in some
            // chunk. Detect "fell off the end" at the last chunk boundary.
            if let Some(override_) = mask_override.filter(|_| chunk_end == n) {
                let landed_anywhere = override_.slot < n;
                assert!(
                    landed_anywhere,
                    "MaskEmbedOverride.slot ({}) is out of range for tokens.len() ({})",
                    override_.slot, n,
                );
            }
            forward_prefill_chunk(
                gpu,
                weights,
                config,
                chunk,
                start_pos + chunk_start,
                kv_cache,
                dn_state,
                scratch,
                pbs,
                hidden_rb.as_deref(),
                pth_slot,
                tape_for_chunk,
                chunk_start,
                tv_for_chunk,
                false, // pre_uploaded: default path uploads inside
                None,  // band: full-stack single-GPU path
                mo_for_chunk,
                None, // positions_override: default path uses linear positions
                needs_last_token_logits,
                max_layer,
                force_q8_gdn_per_token,
                None, // routed_out: non-EP single-GPU path
            )?;
            if let Some(rb) = hidden_rb.as_mut() {
                // Scatter fixed-offset staging writes (done inside the chunk)
                // to the ring at the current head, then advance head by n.
                // This is the out-of-capture step: graph-captured writes went
                // to staging[0..n*h], this commit places them at head*h
                // where head is read from CPU state at call time (not baked
                // into a captured graph node).
                rb.commit_staging_to_ring(gpu, chunk_n)?;
            }
            chunk_start = chunk_end;
        }
        Ok(())
    })();
    if let Some(owned) = own_pbs {
        owned.free_gpu(gpu);
    }
    result
}
