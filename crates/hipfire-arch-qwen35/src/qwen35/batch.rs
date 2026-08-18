// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 continuous-batch state: `PrefillBatchScratch`, `Qwen35DecodeBatchState`,
//! lane-mask helpers, and the independent-lane batched decode entry points.

use super::config::LayerType;
use super::config::Qwen35Config;
use super::forward::Qwen35Scratch;
use super::prefill::forward_batch_chunk_impl;
use super::prefill::forward_prefill_batch;
use super::prefill::moe_grouped_m_total_max;
use super::prefill::run_plain_gemm_key;
use super::prefill::MOE_GROUPED_BLOCK_M;
use super::weights::DeltaNetState;
use super::weights::Qwen35Weights;
use super::weights::StateQuant;
use hip_bridge::HipError;
use hip_bridge::HipResult;
use hipfire_runtime::llama;
use hipfire_runtime::llama::EmbeddingFormat;
use hipfire_runtime::llama::WeightTensor;
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;

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
    // Depth-based RoPE angles for DDTree verify (39aa358 fix). Uploaded in
    // tree-verify mode; FA RoPE reads it instead of `positions` while KV
    // writes and attention seq_len keep the flat physical slots.
    pub rope_positions: GpuTensor,
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
    // FP32 per-node tape for the FP32 `StateQuant` tree-verify path. Same
    // element layout as `dn_s_tape_q8` but f32 (4×), no scales side-table.
    // TODO: gate allocation on state_quant (needs threading StateQuant into
    // `new`); currently always allocated when LA layers exist, like the Q8
    // tape. s_tape_f32: [max_batch × n_v_heads × head_dim × head_dim] f32.
    pub dn_s_tape_f32: Option<GpuTensor>,
}

impl PrefillBatchScratch {
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, max_batch: usize) -> HipResult<Self> {
        // Default: allocate the spec-decode DeltaNet S-tape (tree-verify reads it).
        Self::new_opt(gpu, config, max_batch, /*cap_gdn_tape=*/ true)
    }

    /// Like [`new`], but `cap_gdn_tape` controls allocation of the per-token
    /// DeltaNet S-state tape (`dn_s_tape_*`, sized `[max_batch × n_v_heads ×
    /// value_head_dim²]`). The tape is consumed ONLY by the tree-verify
    /// (spec-decode) GDN kernels; plain prefill (`tree_parents == None`)
    /// advances the recurrent state in place and never touches it. Pass `false`
    /// for plain prefill to skip the tape — on A3B (16 value heads × 128²) it is
    /// ~10 GB at an 8k batch (dn_s_tape_f32 8.2 GB + dn_s_tape_q8 2 GB), the
    /// difference between an 8k prefill fitting and OOMing. Callers that may run
    /// tree-verify MUST pass `true` (else the `.expect()` at the consumption site
    /// panics).
    pub fn new_opt(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        max_batch: usize,
        cap_gdn_tape: bool,
    ) -> HipResult<Self> {
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = k_dim * 2 + v_dim;
        let n_v_heads = config.linear_num_value_heads;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;

        // hunt3 H-E residual: this struct literal allocates ~40 GpuTensors via
        // `?` early-returns. PrefillBatchScratch has no Drop impl (GpuTensor
        // carries no Gpu handle; free_tensor needs &mut Gpu), so a `?` failure
        // partway through would drop the already-allocated tensors WITHOUT
        // freeing them on the device — the exact intra-`new` leak the
        // cross-band H-E recovery can't reach. OOM during new() is precisely
        // when a mid-literal failure is most likely. Fix: route every alloc
        // through a ledger and, on the first error, free everything allocated
        // so far before propagating. `alloc!` records mandatory tensors;
        // `alloc_opt!` records the inner tensor of an `if cond { Some(..) }`.
        //
        // The ledger stores non-owning aliases (DeviceBuffer has no Drop and
        // GpuTensor is not Clone), so on success the aliases drop as no-ops and
        // the real tensors live on in the struct (no double-free); on error we
        // free each alias once, which releases the same pool buffer the
        // partially-built (and about-to-be-dropped, never-freed) field held.
        let mut ledger: Vec<GpuTensor> = Vec::with_capacity(48);
        macro_rules! alloc {
            ($shape:expr, $dt:expr) => {
                match gpu.alloc_tensor($shape, $dt) {
                    Ok(t) => {
                        // SAFETY: alias lives only inside `new`; if used it is
                        // freed in the error arm below (the original field is
                        // dropped without freeing, no Drop on GpuTensor), and
                        // on success it is dropped untouched (no Drop on
                        // DeviceBuffer) while the original is moved into Self.
                        ledger.push(GpuTensor {
                            buf: unsafe { t.buf.alias() },
                            shape: t.shape.clone(),
                            dtype: t.dtype,
                        });
                        t
                    }
                    Err(e) => {
                        for prev in ledger.drain(..) {
                            let _ = gpu.free_tensor(prev);
                        }
                        return Err(e);
                    }
                }
            };
        }
        macro_rules! alloc_opt {
            ($cond:expr, $shape:expr, $dt:expr) => {
                if $cond {
                    Some(alloc!($shape, $dt))
                } else {
                    None
                }
            };
        }

        // Hoisted grouped-GEMM sizing (same value across the Path-2 fields).
        let grouped_m_total_max =
            moe_grouped_m_total_max(max_batch, config.num_experts_per_tok, config.num_experts);
        let grouped_total_slots_max = max_batch * config.num_experts_per_tok;

        Ok(Self {
            max_batch,
            x_batch: alloc!(&[max_batch * dim], DType::F32),
            x_rot_batch: alloc!(&[max_batch * dim], DType::F32),
            x_norm_batch: alloc!(&[max_batch * dim], DType::F32),
            dn_qkv_batch: alloc!(&[max_batch * qkv_dim], DType::F32),
            dn_z_batch: alloc!(&[max_batch * v_dim], DType::F32),
            dn_alpha_batch: alloc!(&[max_batch * n_v_heads], DType::F32),
            dn_beta_batch: alloc!(&[max_batch * n_v_heads], DType::F32),
            dn_q_raw_batch: alloc!(&[max_batch * k_dim], DType::F32),
            dn_k_raw_batch: alloc!(&[max_batch * k_dim], DType::F32),
            dn_v_batch: alloc!(&[max_batch * v_dim], DType::F32),
            dn_q_batch: alloc!(&[max_batch * v_dim], DType::F32),
            dn_k_batch: alloc!(&[max_batch * v_dim], DType::F32),
            dn_attn_out_batch: alloc!(&[max_batch * v_dim], DType::F32),
            dn_normed_batch: alloc!(&[max_batch * v_dim], DType::F32),
            gate_ffn_batch: alloc!(&[max_batch * hidden_dim], DType::F32),
            up_batch: alloc!(&[max_batch * hidden_dim], DType::F32),
            ffn_hidden_batch: alloc!(&[max_batch * hidden_dim], DType::F32),
            dn_normed_rot_batch: alloc!(&[max_batch * v_dim], DType::F32),
            // F32 dtype = 4 bytes/element, same layout as i32. The rope /
            // attention / kv_write kernels cast the pointer to `const int*`,
            // so dtype is cosmetic. Upload i32 bits via memcpy_htod.
            positions: alloc!(&[max_batch], DType::F32),
            // Depth-based RoPE angles for DDTree verify (39aa358 fix):
            // `positions` stays the flat linear KV slot index; this buffer
            // carries `base_pos + depth(node)` so FA-layer RoPE rotates Q/K
            // at the logically-correct phase while KV writes stay on
            // distinct linear slots. Uploaded per cycle in tree-verify mode
            // from `TreeVerifyCtx.positions`; FA RoPE kernels read it ONLY
            // when `tree_verify.is_some()`. Same i32-in-F32 cosmetic dtype
            // pattern as `positions`.
            rope_positions: alloc!(&[max_batch], DType::F32),
            tokens: alloc!(&[max_batch], DType::F32),
            fa_q_full_batch: alloc!(&[max_batch * q_dim * 2], DType::F32),
            fa_q_batch: alloc!(&[max_batch * q_dim], DType::F32),
            fa_gate_batch: alloc!(&[max_batch * q_dim], DType::F32),
            fa_k_batch: alloc!(&[max_batch * kv_dim], DType::F32),
            fa_v_batch: alloc!(&[max_batch * kv_dim], DType::F32),
            fa_attn_out_batch: alloc!(&[max_batch * q_dim], DType::F32),
            fa_attn_out_rot_batch: alloc!(&[max_batch * q_dim], DType::F32),
            moe_router_logits_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.num_experts],
                DType::F32
            ),
            moe_shared_scalar_batch: alloc_opt!(config.num_experts > 0, &[max_batch], DType::F32),
            moe_shared_gate_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.shared_expert_intermediate_size],
                DType::F32
            ),
            moe_shared_up_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.shared_expert_intermediate_size],
                DType::F32
            ),
            moe_shared_rot_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.shared_expert_intermediate_size],
                DType::F32
            ),
            moe_topk_indices_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.num_experts_per_tok],
                DType::F32
            ),
            moe_topk_weights_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.num_experts_per_tok],
                DType::F32
            ),
            moe_gate_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.num_experts_per_tok * config.moe_intermediate_size],
                DType::F32
            ),
            moe_up_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.num_experts_per_tok * config.moe_intermediate_size],
                DType::F32
            ),
            moe_rot_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.num_experts_per_tok * config.moe_intermediate_size],
                DType::F32
            ),
            moe_down_expanded_batch: alloc_opt!(
                config.num_experts > 0,
                &[max_batch * config.num_experts_per_tok * config.dim],
                DType::F32
            ),
            // Path 2 scatter + grouped-WMMA-GEMM scratch (gated at runtime by
            // HIPFIRE_MOE_GROUPED_GEMM=1). m_total_max = N*K_TOP + E*(BLOCK_M-1).
            // i32 buffers stored as Raw (4 bytes/elem matches; no DType::I32 yet).
            moe_expert_token_counts: alloc_opt!(
                config.num_experts > 0,
                &[config.num_experts * 4],
                DType::Raw
            ),
            moe_expert_offsets: alloc_opt!(
                config.num_experts > 0,
                &[(config.num_experts + 1) * 4],
                DType::Raw
            ),
            moe_sorted_slot_index: alloc_opt!(
                config.num_experts > 0,
                &[grouped_m_total_max * 4],
                DType::Raw
            ),
            moe_inverse_perm: alloc_opt!(
                config.num_experts > 0,
                &[grouped_total_slots_max * 4],
                DType::Raw
            ),
            moe_expert_tile_ids: alloc_opt!(
                config.num_experts > 0,
                &[(grouped_m_total_max / MOE_GROUPED_BLOCK_M) * 4],
                DType::Raw
            ),
            moe_y_gate_up_grouped: alloc_opt!(
                config.num_experts > 0,
                &[grouped_m_total_max * 2 * config.moe_intermediate_size],
                DType::F32
            ),
            moe_y_down_grouped: alloc_opt!(
                config.num_experts > 0,
                &[grouped_m_total_max * config.dim],
                DType::F32
            ),
            dn_s_tape_q8: alloc_opt!(
                cap_gdn_tape && config.linear_num_value_heads > 0,
                &[max_batch
                    * config.linear_num_value_heads
                    * config.linear_value_head_dim
                    * config.linear_value_head_dim],
                DType::Raw
            ),
            dn_s_tape_scales: alloc_opt!(
                cap_gdn_tape && config.linear_num_value_heads > 0,
                &[max_batch * config.linear_num_value_heads * config.linear_value_head_dim],
                DType::F32
            ),
            dn_s_tape_f32: alloc_opt!(
                cap_gdn_tape && config.linear_num_value_heads > 0,
                &[max_batch
                    * config.linear_num_value_heads
                    * config.linear_value_head_dim
                    * config.linear_value_head_dim],
                DType::F32
            ),
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> HipResult<()> {
        let mut first_err: Option<HipError> = None;
        let mut note = |r: HipResult<()>| {
            if let Err(e) = r {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        };
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
            self.rope_positions,
            self.tokens,
            self.fa_q_full_batch,
            self.fa_q_batch,
            self.fa_gate_batch,
            self.fa_k_batch,
            self.fa_v_batch,
            self.fa_attn_out_batch,
            self.fa_attn_out_rot_batch,
        ] {
            note(gpu.free_tensor(t));
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
            self.dn_s_tape_f32,
        ] {
            if let Some(t) = t {
                note(gpu.free_tensor(t));
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
/// Persistent fixed-slot state for independent-sequence Qwen3.5 decode.
///
/// Weights remain shared through [`Qwen35Weights`].  Only mutable sequence
/// state is multiplied by `max_batch`: Q8 KV, Q8 DeltaNet matrices, conv rings,
/// and the reusable batched scratch/logit buffers.  V1 intentionally uses Q8
/// KV/state because those are the production decode defaults and the first
/// independent attention/recurrent kernels target their exact layouts.
pub struct Qwen35DecodeBatchState {
    pub max_batch: usize,
    pub lane_capacity: usize,
    pub sample_repeat_capacity: usize,
    pub kv_cache: llama::KvCache,
    pub dn_state: DeltaNetState,
    pub pbs: PrefillBatchScratch,
    pub final_hidden: GpuTensor,
    pub logits: GpuTensor,
    pub lm_rot: GpuTensor,
    pub sample_out: GpuTensor,
    pub sample_repeat_tokens: GpuTensor,
    pub sample_repeat_lengths: GpuTensor,
    pub sample_rng_states: GpuTensor,
}

impl Qwen35DecodeBatchState {
    pub fn new(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        max_batch: usize,
        lane_capacity: usize,
        sample_repeat_capacity: usize,
    ) -> HipResult<Self> {
        if max_batch == 0 || lane_capacity == 0 || sample_repeat_capacity == 0 {
            return Err(HipError::new(
                0,
                "decode batch size, lane capacity, and repeat capacity must be non-zero",
            ));
        }
        // Fail closed before any GPU allocation: independent Q8 attention's
        // dynamic LDS is O(lane_capacity). Cap that the kernel cannot launch
        // must not be admitted into DecodeBatchState.
        gpu.ensure_attention_q8_0_kv_independent_lds(lane_capacity, config.head_dim)?;
        let total_capacity = max_batch
            .checked_mul(lane_capacity)
            .ok_or_else(|| HipError::new(0, "decode batch KV capacity multiplication overflow"))?;
        let repeat_tokens_len = max_batch
            .checked_mul(sample_repeat_capacity)
            .ok_or_else(|| {
                HipError::new(0, "decode batch repeat capacity multiplication overflow")
            })?;
        let is_kv_layer: Vec<bool> = config
            .layer_types
            .iter()
            .map(|t| *t == LayerType::FullAttention)
            .collect();

        // GpuTensor / KvCache / DeltaNetState / PrefillBatchScratch have no
        // freeing Drop (free needs &mut Gpu). A mid-`new` `?` would leak every
        // prior stage while the daemon falls back to sequential with the leak
        // still resident. Stage each compound owner, then ordinary tensors
        // through a ledger of non-owning aliases (same pattern as
        // PrefillBatchScratch::new_opt): on error free aliases + compound
        // owners before propagating; on success aliases drop as no-ops.
        let kv_cache = llama::KvCache::new_gpu_q8_filtered(
            gpu,
            &is_kv_layer,
            config.n_kv_heads,
            config.head_dim,
            total_capacity,
        )?;
        let dn_state =
            match DeltaNetState::new_batched_with_quant(gpu, config, StateQuant::Q8, max_batch) {
                Ok(s) => s,
                Err(e) => {
                    let _ = kv_cache.free_gpu(gpu);
                    return Err(e);
                }
            };
        let pbs = match PrefillBatchScratch::new_opt(gpu, config, max_batch, false) {
            Ok(p) => p,
            Err(e) => {
                dn_state.free_gpu(gpu);
                let _ = kv_cache.free_gpu(gpu);
                return Err(e);
            }
        };

        let mut ledger: Vec<GpuTensor> = Vec::with_capacity(7);
        macro_rules! zeros {
            ($shape:expr) => {
                match gpu.zeros($shape, DType::F32) {
                    Ok(t) => {
                        // SAFETY: alias lives only inside `new`. On error it is
                        // freed below (original field drops without freeing);
                        // on success it drops untouched while the original
                        // moves into Self.
                        ledger.push(GpuTensor {
                            buf: unsafe { t.buf.alias() },
                            shape: t.shape.clone(),
                            dtype: t.dtype,
                        });
                        t
                    }
                    Err(e) => {
                        for prev in ledger.drain(..) {
                            let _ = gpu.free_tensor(prev);
                        }
                        pbs.free_gpu(gpu);
                        dn_state.free_gpu(gpu);
                        let _ = kv_cache.free_gpu(gpu);
                        return Err(e);
                    }
                }
            };
        }

        let final_hidden = zeros!(&[max_batch * config.dim]);
        let logits = zeros!(&[max_batch * config.vocab_size]);
        let lm_rot = zeros!(&[max_batch * config.dim]);
        let sample_out = zeros!(&[max_batch * 2]);
        let sample_repeat_tokens = zeros!(&[repeat_tokens_len]);
        let sample_repeat_lengths = zeros!(&[max_batch]);
        let sample_rng_states = zeros!(&[max_batch]);
        Ok(Self {
            max_batch,
            lane_capacity,
            sample_repeat_capacity,
            kv_cache,
            dn_state,
            pbs,
            final_hidden,
            logits,
            lm_rot,
            sample_out,
            sample_repeat_tokens,
            sample_repeat_lengths,
            sample_rng_states,
        })
    }

    pub fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        self.kv_cache.clear_gpu(gpu)?;
        self.dn_state.reset(gpu)?;
        // Clear the batched scratch buffers (x_batch, etc.) — PrefillBatchScratch has no
        // clear_gpu in this branch, so memset its core tensors directly.
        for t in [
            &self.pbs.x_batch,
            &self.pbs.x_rot_batch,
            &self.pbs.x_norm_batch,
            &self.pbs.positions,
            &self.pbs.tokens,
            &self.final_hidden,
            &self.logits,
            &self.lm_rot,
            &self.sample_out,
            &self.sample_repeat_tokens,
            &self.sample_repeat_lengths,
            &self.sample_rng_states,
        ] {
            gpu.hip.memset(&t.buf, 0, t.buf.size())?;
        }
        Ok(())
    }

    pub fn reset_lane(
        &mut self,
        gpu: &mut Gpu,
        config: &Qwen35Config,
        lane: usize,
    ) -> HipResult<()> {
        let mut kv_lane = self.kv_cache.q8_lane_view(lane, self.lane_capacity)?;
        kv_lane.clear_gpu(gpu)?;
        let mut dn_lane = self.dn_state.q8_lane_view(config, lane, self.max_batch)?;
        dn_lane.reset(gpu)?;
        // Zero the lane's row in the batched output buffers.
        let hidden_lane = self.final_hidden.sub_offset(lane * config.dim, config.dim);
        gpu.hip
            .memset(&hidden_lane.buf, 0, hidden_lane.buf.size())?;
        let logits_lane = self
            .logits
            .sub_offset(lane * config.vocab_size, config.vocab_size);
        gpu.hip
            .memset(&logits_lane.buf, 0, logits_lane.buf.size())?;
        let rot_lane = self.lm_rot.sub_offset(lane * config.dim, config.dim);
        gpu.hip.memset(&rot_lane.buf, 0, rot_lane.buf.size())?;
        Ok(())
    }

    /// Seed one lane with a complete prompt through the production sequential
    /// prefill path.  The final prompt logits are copied into this lane's row
    /// of [`Self::logits`], so the caller can sample the first completion token
    /// for every seeded lane in one batch.
    pub fn prefill_lane(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        scratch: &Qwen35Scratch,
        lane: usize,
        tokens: &[u32],
    ) -> HipResult<()> {
        if lane >= self.max_batch || tokens.is_empty() || tokens.len() >= self.lane_capacity {
            return Err(HipError::new(
                0,
                "prefill lane/token count is invalid or leaves no decode capacity",
            ));
        }
        // Formats the independent decode-batch path can actually execute.
        // Must stay aligned with `lm_head_batched` + `prepare_decode_batch_inputs`
        // (and daemon `qwen_batch_weight_formats_supported`): reject unsupported
        // lm_head / F32 embedding before any lane work so capability is never
        // silently half-advertised at the first prefill.
        if !matches!(
            weights.embd_format,
            EmbeddingFormat::HFQ4G256 | EmbeddingFormat::Q8_0
        ) {
            return Err(HipError::new(
                0,
                "prefill_lane: F32/unsupported embedding is not supported for batched decode",
            ));
        }
        if !matches!(
            weights.output.gpu_dtype,
            DType::Q8_0
                | DType::HFQ4G256
                | DType::MQ4G256
                | DType::MQ4G256V2
                | DType::MQ4CG256
                | DType::HFQ6G256
                | DType::MQ6G256
                | DType::MQ3G256
        ) {
            return Err(HipError::new(
                0,
                &format!(
                    "prefill_lane: unsupported lm_head dtype {:?} for batched decode",
                    weights.output.gpu_dtype
                ),
            ));
        }
        let mut kv_lane = self.kv_cache.q8_lane_view(lane, self.lane_capacity)?;
        let mut dn_lane = self.dn_state.q8_lane_view(config, lane, self.max_batch)?;
        forward_prefill_batch(
            gpu,
            weights,
            config,
            tokens,
            0,
            &mut kv_lane,
            &mut dn_lane,
            scratch,
            None,
            None,
            None,
            None,
        )?;
        gpu.memcpy_dtod_at_auto(
            &self.logits.buf,
            lane * config.vocab_size * 4,
            &scratch.logits.buf,
            0,
            config.vocab_size * 4,
        )
    }

    pub fn sample(
        &self,
        gpu: &mut Gpu,
        config: &Qwen35Config,
        batch_size: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        rng_state: u32,
    ) -> HipResult<(Vec<u32>, u32)> {
        if batch_size == 0 || batch_size > self.max_batch {
            return Err(HipError::new(0, "sample batch size is out of range"));
        }
        // Emulate the former sample_rows_f32 via the product sampler with
        // empty histories and uniform RNG. This preserves the exact HIP kernel
        // used for product sampling while keeping the simple API.
        let vocab = config.vocab_size;
        let logits = self.logits.sub_offset(0, batch_size * vocab);
        // Zero histories
        gpu.hip.memset(
            &self.sample_repeat_tokens.buf,
            0,
            self.sample_repeat_tokens.buf.size(),
        )?;
        gpu.hip.memset(
            &self.sample_repeat_lengths.buf,
            0,
            self.sample_repeat_lengths.buf.size(),
        )?;
        let rng_vec = vec![rng_state; batch_size];
        let rng_bytes =
            unsafe { std::slice::from_raw_parts(rng_vec.as_ptr() as *const u8, rng_vec.len() * 4) };
        gpu.hip
            .memcpy_htod(&self.sample_rng_states.buf, rng_bytes)?;
        let out = gpu.sample_rows_pf_f32(
            &logits,
            &self.sample_repeat_tokens,
            &self.sample_repeat_lengths,
            &self.sample_rng_states,
            &self.sample_out,
            batch_size,
            vocab,
            self.sample_repeat_capacity,
            temperature,
            top_p,
            1.0,
            0.0,
            0.0,
            top_k,
            None,
        )?;
        // sample_rows_pf_f32 returns Vec<(token, rng)>, collapse to (tokens, next_rng)
        let tokens: Vec<u32> = out.iter().map(|(t, _)| *t).collect();
        let next_rng = out.first().map(|(_, r)| *r).unwrap_or(rng_state);
        Ok((tokens, next_rng))
    }

    /// Sample independent rows with the full product sampling surface and
    /// per-lane RNG state.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_product(
        &self,
        gpu: &mut Gpu,
        config: &Qwen35Config,
        batch_size: usize,
        repeat_tokens: &[u32],
        repeat_lengths: &[u32],
        rng_states: &[u32],
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        min_p: Option<f32>,
        repeat_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
    ) -> HipResult<Vec<(u32, u32)>> {
        if batch_size == 0
            || batch_size > self.max_batch
            || repeat_tokens.len() != batch_size * self.sample_repeat_capacity
            || repeat_lengths.len() != batch_size
            || rng_states.len() != batch_size
        {
            return Err(HipError::new(
                0,
                "product sampler inputs do not match the active batch shape",
            ));
        }
        let repeat_bytes = unsafe {
            std::slice::from_raw_parts(repeat_tokens.as_ptr() as *const u8, repeat_tokens.len() * 4)
        };
        let length_bytes = unsafe {
            std::slice::from_raw_parts(
                repeat_lengths.as_ptr() as *const u8,
                repeat_lengths.len() * 4,
            )
        };
        let rng_bytes = unsafe {
            std::slice::from_raw_parts(rng_states.as_ptr() as *const u8, rng_states.len() * 4)
        };
        gpu.hip
            .memcpy_htod(&self.sample_repeat_tokens.buf, repeat_bytes)?;
        gpu.hip
            .memcpy_htod(&self.sample_repeat_lengths.buf, length_bytes)?;
        gpu.hip
            .memcpy_htod(&self.sample_rng_states.buf, rng_bytes)?;
        let logits = self.logits.sub_offset(0, batch_size * config.vocab_size);
        gpu.sample_rows_pf_f32(
            &logits,
            &self.sample_repeat_tokens,
            &self.sample_repeat_lengths,
            &self.sample_rng_states,
            &self.sample_out,
            batch_size,
            config.vocab_size,
            self.sample_repeat_capacity,
            temperature,
            top_p,
            repeat_penalty,
            presence_penalty,
            frequency_penalty,
            top_k,
            min_p,
        )
    }

    /// Single-row refill sampler. Continuous batching uses this only when a
    /// completed lane is replaced; steady-state decode samples all rows with
    /// [`Self::sample`] and pays one D2H for the whole batch.
    pub fn sample_lane(
        &self,
        gpu: &mut Gpu,
        config: &Qwen35Config,
        lane: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        rng_state: u32,
    ) -> HipResult<(u32, u32)> {
        if lane >= self.max_batch {
            return Err(HipError::new(0, "sample lane is out of range"));
        }
        let logits = self
            .logits
            .sub_offset(lane * config.vocab_size, config.vocab_size);
        gpu.sample_top_p_pf(
            &logits,
            &self.sample_out,
            &self.sample_out,
            config.vocab_size,
            temperature,
            top_p,
            rng_state,
            0,
            1.0,
            0.0,
            0.0,
            top_k,
            None,
        )
    }

    /// Product-semantics refill sample for one lane. Refill is deliberately
    /// outside the steady-state batched draw, so a single-row sampler here
    /// does not serialize active lanes.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_lane_product(
        &self,
        gpu: &mut Gpu,
        config: &Qwen35Config,
        lane: usize,
        repeat_tokens: &[u32],
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        min_p: Option<f32>,
        rng_state: u32,
        repeat_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
    ) -> HipResult<(u32, u32)> {
        if lane >= self.max_batch || repeat_tokens.len() > self.sample_repeat_capacity {
            return Err(HipError::new(0, "sample lane/history is out of range"));
        }
        let repeat_bytes = unsafe {
            std::slice::from_raw_parts(repeat_tokens.as_ptr() as *const u8, repeat_tokens.len() * 4)
        };
        gpu.hip
            .memcpy_htod(&self.sample_repeat_tokens.buf, repeat_bytes)?;
        let logits = self
            .logits
            .sub_offset(lane * config.vocab_size, config.vocab_size);
        gpu.sample_top_p_pf(
            &logits,
            &self.sample_out,
            &self.sample_repeat_tokens,
            config.vocab_size,
            temperature,
            top_p,
            rng_state,
            repeat_tokens.len(),
            repeat_penalty,
            presence_penalty,
            frequency_penalty,
            top_k,
            min_p,
        )
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> HipResult<()> {
        let mut first_err: Option<HipError> = None;
        let mut note = |r: HipResult<()>| {
            if let Err(e) = r {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        };
        note(self.kv_cache.free_gpu(gpu));
        self.dn_state.free_gpu(gpu);
        note(self.pbs.free_gpu(gpu));
        note(gpu.free_tensor(self.final_hidden));
        note(gpu.free_tensor(self.logits));
        note(gpu.free_tensor(self.lm_rot));
        note(gpu.free_tensor(self.sample_out));
        note(gpu.free_tensor(self.sample_repeat_tokens));
        note(gpu.free_tensor(self.sample_repeat_lengths));
        note(gpu.free_tensor(self.sample_rng_states));
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
impl Qwen35DecodeBatchState {
    /// Checked projection mirroring `new` — returns total bytes that `new` would
    /// allocate, or an overflow error. Mirrors every allocation in `new` exactly
    /// (Q8 KV matrix+scales, DN Q8 matrix+scales+default F16 EF+conv, PBS, and the
    /// 7 output tensors) using *only* checked arithmetic. Zero or overflow is `Err`
    /// never panic. Used for admission before any VRAM commit.
    pub fn projected_allocation_bytes(
        config: &Qwen35Config,
        max_batch: usize,
        lane_capacity: usize,
        sample_repeat_capacity: usize,
    ) -> HipResult<u64> {
        if max_batch == 0 || lane_capacity == 0 || sample_repeat_capacity == 0 {
            return Err(HipError::new(
                0,
                "projected allocation: zero batch/capacity",
            ));
        }
        let total_cap = (max_batch as u64)
            .checked_mul(lane_capacity as u64)
            .ok_or_else(|| HipError::new(0, "projected KV capacity overflow"))?;
        let repeat_len = (max_batch as u64)
            .checked_mul(sample_repeat_capacity as u64)
            .ok_or_else(|| HipError::new(0, "projected repeat capacity overflow"))?;
        let l_fa = config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count() as u64;
        let l_dn = config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count() as u64;
        let hv = config.n_kv_heads as u64;
        if config.head_dim % 32 != 0 {
            return Err(HipError::new(0, "head_dim must be divisible by 32"));
        }
        let hd_div = (config.head_dim as u64) / 32;
        let kv = 68u64
            .checked_mul(total_cap)
            .and_then(|v| v.checked_mul(l_fa))
            .and_then(|v| v.checked_mul(hv))
            .and_then(|v| v.checked_mul(hd_div))
            .ok_or_else(|| HipError::new(0, "projected KV bytes overflow"))?;
        let n_heads = config.linear_num_value_heads as u64;
        let s_dim = config.linear_key_head_dim as u64;
        let s_elems = n_heads
            .checked_mul(s_dim)
            .and_then(|v| v.checked_mul(s_dim))
            .ok_or_else(|| HipError::new(0, "DN s_elems overflow"))?;
        let dn_s = l_dn
            .checked_mul(max_batch as u64)
            .and_then(|v| v.checked_mul(s_elems))
            .ok_or_else(|| HipError::new(0, "DN s bytes overflow"))?;
        let dn_scales = l_dn
            .checked_mul(max_batch as u64)
            .and_then(|v| v.checked_mul(n_heads))
            .and_then(|v| v.checked_mul(s_dim))
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| HipError::new(0, "DN scales bytes overflow"))?;
        let dn_ef = l_dn
            .checked_mul(max_batch as u64)
            .and_then(|v| v.checked_mul(s_elems))
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| HipError::new(0, "DN EF bytes overflow"))?;
        let k_heads = config.linear_num_key_heads as u64;
        let k_hd = config.linear_key_head_dim as u64;
        let v_heads = config.linear_num_value_heads as u64;
        let v_hd = config.linear_value_head_dim as u64;
        let k_part = k_heads
            .checked_mul(k_hd)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| HipError::new(0, "conv channels overflow"))?;
        let v_part = v_heads
            .checked_mul(v_hd)
            .ok_or_else(|| HipError::new(0, "conv channels overflow"))?;
        let conv_channels = k_part
            .checked_add(v_part)
            .ok_or_else(|| HipError::new(0, "conv channels overflow"))?;
        if config.conv_kernel_dim == 0 {
            return Err(HipError::new(0, "conv_kernel_dim must be >=1"));
        }
        let kernel_minus_one = (config.conv_kernel_dim as u64) - 1;
        let conv_elems = conv_channels
            .checked_mul(kernel_minus_one)
            .and_then(|v| v.checked_mul(max_batch as u64))
            .and_then(|v| v.checked_mul(l_dn))
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| HipError::new(0, "DN conv bytes overflow"))?;
        let pbs = PrefillBatchScratch::projected_allocation_bytes(config, max_batch, false)?;
        let dim = config.dim as u64;
        let vocab = config.vocab_size as u64;
        let out = (max_batch as u64)
            .checked_mul(4)
            .and_then(|v| {
                let inner = (2u64)
                    .checked_mul(dim)?
                    .checked_add(vocab)?
                    .checked_add(sample_repeat_capacity as u64)?
                    .checked_add(4)?;
                v.checked_mul(inner)
            })
            .ok_or_else(|| HipError::new(0, "output bytes overflow"))?;
        if (max_batch as u64)
            .checked_mul(sample_repeat_capacity as u64)
            .ok_or_else(|| HipError::new(0, "repeat overflow"))?
            != repeat_len
        {
            return Err(HipError::new(0, "repeat len mismatch"));
        }
        kv.checked_add(dn_s)
            .and_then(|v| v.checked_add(dn_scales))
            .and_then(|v| v.checked_add(dn_ef))
            .and_then(|v| v.checked_add(conv_elems))
            .and_then(|v| v.checked_add(pbs))
            .and_then(|v| v.checked_add(out))
            .ok_or_else(|| HipError::new(0, "total projected bytes overflow"))
    }
}

impl PrefillBatchScratch {
    /// Checked projection mirroring `new_opt`. Returns total bytes allocated
    /// for `max_batch` with/without the GDN tape. Mirrors every `alloc!` in
    /// `new_opt` exactly using only checked arithmetic.
    pub fn projected_allocation_bytes(
        config: &Qwen35Config,
        max_batch: usize,
        cap_gdn_tape: bool,
    ) -> HipResult<u64> {
        if max_batch == 0 {
            return Err(HipError::new(0, "PBS projected allocation: zero batch"));
        }
        let dim = config.dim as u64;
        let hd = config.hidden_dim as u64;
        let k_heads = config.linear_num_key_heads as u64;
        let k_hd = config.linear_key_head_dim as u64;
        let v_heads = config.linear_num_value_heads as u64;
        let v_hd = config.linear_value_head_dim as u64;
        let n_heads = config.n_heads as u64;
        let head_dim = config.head_dim as u64;
        let n_kv_heads = config.n_kv_heads as u64;
        let qkv_dim = k_heads
            .checked_mul(k_hd)
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| v_heads.checked_mul(v_hd).and_then(|w| v.checked_add(w)))
            .ok_or_else(|| HipError::new(0, "qkv_dim overflow"))?;
        let v_dim = v_heads
            .checked_mul(v_hd)
            .ok_or_else(|| HipError::new(0, "v_dim overflow"))?;
        let k_dim = k_heads
            .checked_mul(k_hd)
            .ok_or_else(|| HipError::new(0, "k_dim overflow"))?;
        let q_dim = n_heads
            .checked_mul(head_dim)
            .ok_or_else(|| HipError::new(0, "q_dim overflow"))?;
        let kv_dim = n_kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| HipError::new(0, "kv_dim overflow"))?;
        let n = max_batch as u64;
        let mut total: u64 = 0;
        let mut add = |elems: u64, bpe: u64| -> HipResult<()> {
            let bytes = elems
                .checked_mul(bpe)
                .ok_or_else(|| HipError::new(0, "PBS bytes overflow"))?;
            total = total
                .checked_add(bytes)
                .ok_or_else(|| HipError::new(0, "PBS total overflow"))?;
            Ok(())
        };
        let cm = |a: u64, b: u64| -> HipResult<u64> {
            a.checked_mul(b)
                .ok_or_else(|| HipError::new(0, "PBS checked_mul overflow"))
        };
        add(cm(n, dim)?, 4)?;
        add(cm(n, dim)?, 4)?;
        add(cm(n, dim)?, 4)?;
        add(cm(n, qkv_dim)?, 4)?;
        add(cm(n, v_dim)?, 4)?;
        add(cm(n, v_heads)?, 4)?;
        add(cm(n, v_heads)?, 4)?;
        add(cm(n, k_dim)?, 4)?;
        add(cm(n, k_dim)?, 4)?;
        add(cm(n, v_dim)?, 4)?;
        add(cm(n, v_dim)?, 4)?;
        add(cm(n, v_dim)?, 4)?;
        add(cm(n, v_dim)?, 4)?;
        add(cm(n, hd)?, 4)?;
        add(cm(n, hd)?, 4)?;
        add(cm(n, hd)?, 4)?;
        add(cm(n, v_dim)?, 4)?;
        add(n, 4)?;
        add(n, 4)?;
        add(n, 4)?;
        add(
            cm(n, q_dim)?
                .checked_mul(2)
                .ok_or_else(|| HipError::new(0, "PBS overflow"))?,
            4,
        )?;
        add(cm(n, q_dim)?, 4)?;
        add(cm(n, q_dim)?, 4)?;
        add(cm(n, kv_dim)?, 4)?;
        add(cm(n, kv_dim)?, 4)?;
        add(cm(n, q_dim)?, 4)?;
        add(cm(n, q_dim)?, 4)?;
        if config.num_experts > 0 {
            add(cm(n, config.num_experts as u64)?, 4)?;
            add(n, 4)?;
            add(cm(n, config.shared_expert_intermediate_size as u64)?, 4)?;
            add(cm(n, config.shared_expert_intermediate_size as u64)?, 4)?;
            add(cm(n, config.shared_expert_intermediate_size as u64)?, 4)?;
            add(cm(n, config.num_experts_per_tok as u64)?, 4)?;
            add(cm(n, config.num_experts_per_tok as u64)?, 4)?;
            add(
                cm(
                    cm(n, config.num_experts_per_tok as u64)?,
                    config.moe_intermediate_size as u64,
                )?,
                4,
            )?;
            add(
                cm(
                    cm(n, config.num_experts_per_tok as u64)?,
                    config.moe_intermediate_size as u64,
                )?,
                4,
            )?;
            add(
                cm(
                    cm(n, config.num_experts_per_tok as u64)?,
                    config.moe_intermediate_size as u64,
                )?,
                4,
            )?;
            add(
                cm(cm(n, config.num_experts_per_tok as u64)?, config.dim as u64)?,
                4,
            )?;
            let m_max =
                moe_grouped_m_total_max(max_batch, config.num_experts_per_tok, config.num_experts)
                    as u64;
            let total_slots = cm(n, config.num_experts_per_tok as u64)?;
            add(config.num_experts as u64 * 4, 1)?;
            add((config.num_experts as u64 + 1) * 4, 1)?;
            add(m_max * 4, 1)?;
            add(total_slots * 4, 1)?;
            add((m_max / 16) * 4, 1)?;
            add(cm(m_max, 2 * config.moe_intermediate_size as u64)?, 4)?;
            add(cm(m_max, config.dim as u64)?, 4)?;
            if cap_gdn_tape && config.linear_num_value_heads > 0 {
                let tape_elems = n
                    .checked_mul(v_heads)
                    .and_then(|v| v.checked_mul(v_hd))
                    .and_then(|v| v.checked_mul(v_hd))
                    .ok_or_else(|| HipError::new(0, "PBS tape elems overflow"))?;
                add(tape_elems, 1)?;
                add(
                    cm(n, v_heads)?
                        .checked_mul(v_hd)
                        .ok_or_else(|| HipError::new(0, "PBS tape overflow"))?,
                    4,
                )?;
                add(tape_elems, 4)?;
            }
        } else if cap_gdn_tape && config.linear_num_value_heads > 0 {
            let tape_elems = n
                .checked_mul(v_heads)
                .and_then(|v| v.checked_mul(v_hd))
                .and_then(|v| v.checked_mul(v_hd))
                .ok_or_else(|| HipError::new(0, "PBS tape elems overflow"))?;
            add(tape_elems, 1)?;
            add(
                cm(n, v_heads)?
                    .checked_mul(v_hd)
                    .ok_or_else(|| HipError::new(0, "PBS tape overflow"))?,
                4,
            )?;
            add(tape_elems, 4)?;
        }
        Ok(total)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum BatchSemantics<'a> {
    Sequential,
    Independent {
        positions: &'a [usize],
        lane_capacity: usize,
        active_mask: u64,
    },
}

impl BatchSemantics<'_> {
    #[inline]
    pub(crate) fn is_independent(self) -> bool {
        matches!(self, Self::Independent { .. })
    }
    #[inline]
    pub(crate) fn active_mask(self) -> Option<u64> {
        match self {
            Self::Independent { active_mask, .. } => Some(active_mask),
            Self::Sequential => None,
        }
    }
}
/// EP tick input residency: decode `pbs` per rank and `seed_pbs` are separate
/// allocations. `prefill_lane` correctly uses `seed_pbs` with `false,false`
/// and has no bearing on `forward_tick`'s per-rank decode `pbs`, which has
/// no external `prepare_decode_batch_inputs`. Therefore only band 0 on every
/// rank owns staging of host token/position/embedding inputs into `pbs`; later
/// bands must reuse the transformed residual already in `pbs.x_batch`.
#[inline]
pub(crate) fn ep_tick_inputs_prepared(layer_idx: usize) -> bool {
    layer_idx != 0
}

/// Central lane helpers — single source for max_batch bounds and shifts.
pub(crate) fn valid_lane_mask(max_batch: usize) -> HipResult<u64> {
    if max_batch == 0 || max_batch > 64 {
        return Err(HipError::new(0, "valid_lane_mask: max_batch must be 1..64"));
    }
    if max_batch >= 64 {
        Ok(u64::MAX)
    } else {
        Ok((1u64 << max_batch) - 1)
    }
}

#[inline]
pub(crate) fn lane_bit(lane: usize, max_batch: usize) -> HipResult<u64> {
    if lane >= max_batch {
        return Err(HipError::new(
            0,
            "lane_bit: lane out of range for max_batch",
        ));
    }
    if max_batch == 0 || max_batch > 64 {
        return Err(HipError::new(0, "lane_bit: max_batch must be 1..64"));
    }
    Ok(1u64 << lane)
}

/// Allocation-free iteration over contiguous active spans.
/// Calls `f(start, len)` for each span where mask has contiguous ones.
#[inline]
pub(crate) fn for_each_active_span<F>(mask: u64, max_batch: usize, mut f: F) -> HipResult<()>
where
    F: FnMut(usize, usize) -> HipResult<()>,
{
    valid_lane_mask(max_batch)?;
    if max_batch == 0 {
        return Ok(());
    }
    let valid = valid_lane_mask(max_batch)?;
    if mask & !valid != 0 {
        return Err(HipError::new(
            0,
            "for_each_active_span: mask has bits beyond max_batch",
        ));
    }
    let mut lane = 0usize;
    while lane < max_batch {
        if (mask >> lane) & 1 == 0 {
            lane += 1;
            continue;
        }
        let start = lane;
        while lane < max_batch && ((mask >> lane) & 1) == 1 {
            lane += 1;
        }
        f(start, lane - start)?;
    }
    Ok(())
}

pub(crate) fn lm_head_batched(
    gpu: &mut Gpu,
    output: &WeightTensor,
    hidden: &GpuTensor,
    rot: &GpuTensor,
    logits: &GpuTensor,
    batch_size: usize,
) -> HipResult<()> {
    match output.gpu_dtype {
        DType::Q8_0 => {
            gpu.gemm_q8_0_batched(&output.buf, hidden, logits, output.m, output.k, batch_size)
        }
        DType::HFQ4G256 => gpu.gemm_hfq4g256_batched_lmhead(
            &output.buf,
            hidden,
            logits,
            output.m,
            output.k,
            batch_size,
        ),
        DType::MQ4G256 => {
            llama::rotate_x_mq_batched_for(gpu, output, hidden, rot, output.k, batch_size)?;
            gpu.gemm_hfq4g256_batched_lmhead(
                &output.buf,
                rot,
                logits,
                output.m,
                output.k,
                batch_size,
            )
        }
        DType::MQ4G256V2 => {
            llama::rotate_x_mq_batched_for(gpu, output, hidden, rot, output.k, batch_size)?;
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmMq4G256V2BatchedLmhead,
                &output.buf,
                output.gpu_dtype,
                rot,
                logits,
                output.m,
                output.k,
                batch_size,
            )
        }
        DType::MQ4CG256 => {
            llama::rotate_x_mq_batched_for(gpu, output, hidden, rot, output.k, batch_size)?;
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmMq4CG256BatchedLmhead,
                &output.buf,
                output.gpu_dtype,
                rot,
                logits,
                output.m,
                output.k,
                batch_size,
            )
        }
        DType::HFQ6G256 => gpu.gemm_hfq6g256_batched_lmhead(
            &output.buf,
            hidden,
            logits,
            output.m,
            output.k,
            batch_size,
        ),
        DType::MQ6G256 => {
            llama::rotate_x_mq_batched_for(gpu, output, hidden, rot, output.k, batch_size)?;
            gpu.gemm_hfq6g256_batched_lmhead(
                &output.buf,
                rot,
                logits,
                output.m,
                output.k,
                batch_size,
            )
        }
        DType::MQ3G256 => {
            llama::rotate_x_mq_batched_for(gpu, output, hidden, rot, output.k, batch_size)?;
            gpu.gemm_hfq3g256_batched_lmhead(
                &output.buf,
                rot,
                logits,
                output.m,
                output.k,
                batch_size,
            )
        }
        other => Err(HipError::new(
            0,
            &format!("lm_head_batched: unsupported dtype {other:?}"),
        )),
    }
}

/// Populate the changing inputs that intentionally stay outside a retained
/// independent-batch replay: token embeddings and per-lane positions.
pub fn prepare_decode_batch_inputs(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    positions: &[usize],
    state: &Qwen35DecodeBatchState,
) -> HipResult<usize> {
    let n = tokens.len();
    if n == 0 {
        return Err(HipError::new(
            0,
            "independent batch input preparation requires at least one lane",
        ));
    }
    if n > state.max_batch || positions.len() != n {
        return Err(HipError::new(
            0,
            "decode batch inputs exceed max_batch or positions length differs",
        ));
    }
    if positions.iter().any(|&p| p >= state.lane_capacity) {
        return Err(HipError::new(
            0,
            "decode batch position exceeds lane capacity",
        ));
    }
    // Reject unsupported formats before any device work (aligned with
    // `prefill_lane` / daemon `qwen_batch_weight_formats_supported`).
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 | EmbeddingFormat::Q8_0 => {}
        EmbeddingFormat::F32 => {
            return Err(HipError::new(
                0,
                "prepare_decode_batch_inputs: F32 embedding not supported for batched decode",
            ));
        }
        _ => {
            return Err(HipError::new(
                0,
                "independent batch requires a batched HFQ4G256 or Q8 embedding",
            ));
        }
    }

    let tokens_host: Vec<i32> = tokens.iter().map(|&token| token as i32).collect();
    let token_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, tokens_host.len() * 4)
    };
    gpu.hip.memcpy_htod(&state.pbs.tokens.buf, token_bytes)?;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256_batched(
            &weights.token_embd,
            &state.pbs.x_batch,
            &state.pbs.tokens,
            n,
            config.dim,
        )?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8_batched(
            &weights.token_embd,
            &state.pbs.x_batch,
            &state.pbs.tokens,
            n,
            config.dim,
        )?,
        // Unreachable: guarded above.
        _ => unreachable!("embedding format pre-checked"),
    }

    let positions_host: Vec<i32> = positions.iter().map(|&position| position as i32).collect();
    let position_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            positions_host.as_ptr() as *const u8,
            positions_host.len() * 4,
        )
    };
    gpu.hip
        .memcpy_htod(&state.pbs.positions.buf, position_bytes)?;
    Ok(positions.iter().copied().max().unwrap_or(0))
}

/// Advance one token in each independent sequence lane and write
/// `[batch, vocab]` logits into `state.logits`.
pub fn forward_decode_batch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    positions: &[usize],
    state: &mut Qwen35DecodeBatchState,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }
    let position = prepare_decode_batch_inputs(gpu, weights, config, tokens, positions, state)?;
    forward_decode_batch_prepared(
        gpu, weights, config, tokens, positions, position, state, scratch,
    )
}

pub fn forward_decode_batch_prepared(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    positions: &[usize],
    position: usize,
    state: &mut Qwen35DecodeBatchState,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let n = tokens.len();
    if n == 0
        || n > state.max_batch
        || positions.len() != n
        || positions.iter().any(|&value| value >= state.lane_capacity)
        || position != positions.iter().copied().max().unwrap_or(0)
    {
        return Err(HipError::new(
            0,
            "prepared independent batch inputs do not match the admitted shape",
        ));
    }
    // Single-GPU all-active path: full low-bit mask preserves exact unmasked behavior.
    let active_mask = valid_lane_mask(n)?;
    let final_hidden = state.final_hidden.sub_offset(0, n * config.dim);
    forward_batch_chunk_impl(
        gpu,
        weights,
        config,
        tokens,
        0,
        &mut state.kv_cache,
        &mut state.dn_state,
        scratch,
        &state.pbs,
        None,
        Some((&final_hidden, 0)),
        None,
        0,
        None,
        true,
        true,
        None,
        None,
        false,
        None,
        None,
        BatchSemantics::Independent {
            positions,
            lane_capacity: state.lane_capacity,
            active_mask,
        },
    )?;

    let logits = state.logits.sub_offset(0, n * config.vocab_size);
    let lm_rot = state.lm_rot.sub_offset(0, n * config.dim);
    lm_head_batched(gpu, &weights.output, &final_hidden, &lm_rot, &logits, n)
}
