// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-request session-state and model-worker lifecycle for the daemon.
//!
//! Two tightly-interwoven concerns kept together because they call each other
//! and share the allocation-epoch state:
//!   - **Qwen3.5 session state** — `Qwen35RequestSessionState` plus the
//!     allocate/save/activate/fork/checkpoint/reset and prefix-hash-validation
//!     helpers that manage multi-turn KV/DeltaNet state per session id.
//!   - **Sequence-state arena + worker view** — the arch-agnostic
//!     `sequence_state_arena_*` dispatch, model-worker id/park/activate, and the
//!     resident-worker status JSON the daemon reports.
//!
//! Extracted verbatim from the former `main.rs` monolith (no behavior change);
//! items called from `main.rs` are `pub`.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "arch-lfm2moe")]
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::qwen35::{DeltaNetState, LayerType};
use hipfire_mixer::{MixerKind, MixerProfile};
use hipfire_model::{
    is_qwen35_family_arch_id, is_qwen35_moe_arch_id, parse_model_worker_id, AcceleratorDeviceInfo,
    AcceleratorInventory, ModelWorkerId, ARCH_ID_LFM2_MOE, ARCH_ID_MINIMAX_M2, ARCH_ID_NEMOTRON_H,
};
use hipfire_runtime::kv;
use hipfire_runtime::sequence_state::SequenceState;
use hipfire_state::{
    describe_sequence_state_descriptors, model_worker_runtime_view_json,
    parsed_handle_may_target_loaded_state, qwen35_sequence_state_handle,
    validate_checkpoint_logical_position, validate_checkpoint_prefix_hash,
    validate_checkpoint_source_resident, DescribedSequenceState, ModelWorkerRuntimeView,
    ParsedSequenceStateHandle, SequenceStateArenaBackend, SequenceStateArenaOperation,
    SequenceStateCheckpointRequest, SequenceStateForkRequest, SequenceStateHandle,
    SequenceStatePageDescriptor, SequenceStatePageKind, SequenceStatePrefixHash,
};

use crate::events::write_error;
use crate::memory::{
    kv_cache_bytes, loaded_model_memory_view, minimax_state_bytes, tensor_bytes, tensor_vec_bytes,
};
use crate::model::LoadedModel;

/// Synthetic session id used by the legacy single-session `generate` path (the
/// pre-multi-session code that didn't supply its own session id).
pub const QWEN35_LEGACY_SESSION_ID: &str = "__legacy_generate__";
#[cfg(feature = "arch-lfm2moe")]
pub const LFM2_LEGACY_SESSION_ID: &str = "__legacy_generate__";
/// Worker id assigned when a load message carries no explicit `worker_id`.
pub const DEFAULT_MODEL_WORKER_ID: &str = "__default__";
static QWEN35_STATE_ALLOCATION_EPOCH: AtomicU64 = AtomicU64::new(1);

/// Monotonic epoch stamped onto each allocated session state, so a stale handle
/// referencing freed/reallocated state can be detected and rejected.
pub fn next_qwen35_state_allocation_epoch() -> u64 {
    QWEN35_STATE_ALLOCATION_EPOCH.fetch_add(1, Ordering::Relaxed)
}

/// Saved Qwen3.5 multi-turn state for one session id: the KV cache, DeltaNet
/// linear-attention state, last-position logits, and the bookkeeping (KV cursor,
/// conversation tokens, prefix hash, prefilled-suffix length) needed to swap a
/// session out of and back into the single resident model slot. `allocation_epoch`
/// stamps the generation so stale handles are rejected.
pub struct Qwen35RequestSessionState {
    pub seq_pos: usize,
    pub conversation_tokens: Vec<u32>,
    pub prefix_hash: Option<SequenceStatePrefixHash>,
    /// Unified per-sequence decode state (KV cache + DeltaNet recurrent state),
    /// keyed by the qwen35 hybrid MixerProfile. P2c: replaces the former separate
    /// `kv_cache: KvCache` + `dn_state: DeltaNetState` fields. Simple read sites
    /// use the `kv_cache()`/`dn_state()` accessors; disjoint-borrow hot-path
    /// sites access `sequence_state.kv` / `sequence_state.recurrent` directly.
    pub sequence_state: SequenceState,
    pub logits: rdna_compute::GpuTensor,
    pub prefilled_generated_suffix_len: usize,
    pub allocation_epoch: u64,
}

impl Qwen35RequestSessionState {
    /// The session's KV cache (a qwen35 session always has one). For single
    /// reads/mutations; sites needing KV **and** DeltaNet simultaneously must
    /// use the disjoint `sequence_state.kv` / `sequence_state.recurrent` fields
    /// directly (a method borrows all of `self`).
    pub fn kv_cache(&self) -> &kv::KvCache {
        self.sequence_state
            .kv()
            .expect("qwen35 session always has KV")
    }
    /// Mutable KV cache (single-access only — see [`Self::kv_cache`]).
    pub fn kv_cache_mut(&mut self) -> &mut kv::KvCache {
        self.sequence_state
            .kv_mut()
            .expect("qwen35 session always has KV")
    }
    /// The session's DeltaNet recurrent state (concrete downcast).
    pub fn dn_state(&self) -> &DeltaNetState {
        self.sequence_state
            .recurrent_as::<DeltaNetState>()
            .expect("qwen35 session recurrent state is DeltaNetState")
    }
    /// Mutable DeltaNet recurrent state (single-access only).
    pub fn dn_state_mut(&mut self) -> &mut DeltaNetState {
        self.sequence_state
            .recurrent_as_mut::<DeltaNetState>()
            .expect("qwen35 session recurrent state is DeltaNetState")
    }

    /// Deep-copy one GPU tensor (fresh device allocation + device-to-device
    /// copy) — used to snapshot session state without aliasing the live buffers.
    pub fn clone_gpu_tensor(
        gpu: &mut rdna_compute::Gpu,
        tensor: &rdna_compute::GpuTensor,
        label: &str,
    ) -> Result<rdna_compute::GpuTensor, String> {
        let buffer_size = tensor.buf.size();
        gpu.bind_thread()
            .map_err(|e| format!("clone qwen35 checkpoint {label} bind gpu: {e:?}"))?;
        let buf = gpu
            .hip
            .malloc(buffer_size)
            .map_err(|e| format!("clone qwen35 checkpoint {label} alloc: {e:?}"))?;
        gpu.hip
            .memcpy_dtod_at(&buf, 0, &tensor.buf, 0, buffer_size)
            .map_err(|e| format!("clone qwen35 checkpoint {label} copy: {e:?}"))?;
        Ok(rdna_compute::GpuTensor {
            buf,
            shape: tensor.shape.clone(),
            dtype: tensor.dtype,
        })
    }

    /// [`clone_gpu_tensor`] over a slice of tensors (e.g. the per-layer KV
    /// vectors), returning a freshly-allocated `Vec`.
    pub fn clone_gpu_tensor_vec(
        gpu: &mut rdna_compute::Gpu,
        tensors: &[rdna_compute::GpuTensor],
        label: &str,
    ) -> Result<Vec<rdna_compute::GpuTensor>, String> {
        tensors
            .iter()
            .enumerate()
            .map(|(i, tensor)| Self::clone_gpu_tensor(gpu, tensor, &format!("{label}[{i}]")))
            .collect()
    }

    pub fn clone_kv_cache(
        gpu: &mut rdna_compute::Gpu,
        kv: &kv::KvCache,
    ) -> Result<kv::KvCache, String> {
        Ok(kv::KvCache {
            k_gpu: Self::clone_gpu_tensor_vec(gpu, &kv.k_gpu, "kv.k_gpu")?,
            v_gpu: Self::clone_gpu_tensor_vec(gpu, &kv.v_gpu, "kv.v_gpu")?,
            k_scales: Self::clone_gpu_tensor_vec(gpu, &kv.k_scales, "kv.k_scales")?,
            v_scales: Self::clone_gpu_tensor_vec(gpu, &kv.v_scales, "kv.v_scales")?,
            kv_dim: kv.kv_dim,
            max_seq: kv.max_seq,
            physical_cap: kv.physical_cap,
            n_kv_heads: kv.n_kv_heads,
            head_dim: kv.head_dim,
            quantized: kv.quantized,
            quant_q8: kv.quant_q8,
            quant_int8: kv.quant_int8,
            quant_hfq4: kv.quant_hfq4,
            quant_asym4: kv.quant_asym4,
            quant_asym3: kv.quant_asym3,
            quant_asym2: kv.quant_asym2,
            boundary_layers: kv.boundary_layers,
            givens_cos: kv
                .givens_cos
                .as_ref()
                .map(|tensor| Self::clone_gpu_tensor(gpu, tensor, "kv.givens_cos"))
                .transpose()?,
            givens_sin: kv
                .givens_sin
                .as_ref()
                .map(|tensor| Self::clone_gpu_tensor(gpu, tensor, "kv.givens_sin"))
                .transpose()?,
            quant_fwht: kv.quant_fwht,
            layer_is_boundary: kv.layer_is_boundary.clone(),
            compact_offset: kv.compact_offset,
            quant_kvarn: kv.quant_kvarn,
            k_window: Self::clone_gpu_tensor_vec(gpu, &kv.k_window, "kv.k_window")?,
            // Read-side scratch is lazily re-allocated on first KVarN attention.
            kvarn_shadow: None,
            kvarn_tiles: None,
            hier: None,
        })
    }

    pub fn clone_dn_state(
        gpu: &mut rdna_compute::Gpu,
        dn: &DeltaNetState,
    ) -> Result<DeltaNetState, String> {
        Ok(DeltaNetState {
            s_matrices: Self::clone_gpu_tensor_vec(gpu, &dn.s_matrices, "dn.s_matrices")?,
            s_scales: Self::clone_gpu_tensor_vec(gpu, &dn.s_scales, "dn.s_scales")?,
            conv_states: Self::clone_gpu_tensor_vec(gpu, &dn.conv_states, "dn.conv_states")?,
            s_ef_residual: Self::clone_gpu_tensor_vec(gpu, &dn.s_ef_residual, "dn.s_ef_residual")?,
            quant: dn.quant,
        })
    }

    /// Deep-copy an existing saved session into a new independent one (KV +
    /// DeltaNet + logits cloned), for branching a conversation without
    /// disturbing the source.
    pub fn fork_from(
        gpu: &mut rdna_compute::Gpu,
        source: &Qwen35RequestSessionState,
    ) -> Result<Self, String> {
        let kv = Self::clone_kv_cache(gpu, source.kv_cache())?;
        let dn = Self::clone_dn_state(gpu, source.dn_state())?;
        Ok(Self {
            seq_pos: source.seq_pos,
            conversation_tokens: source.conversation_tokens.clone(),
            prefix_hash: source.prefix_hash.clone(),
            sequence_state: SequenceState::new(
                source.sequence_state.profile.clone(),
                Some(kv),
                Some(Box::new(dn)),
            ),
            logits: Self::clone_gpu_tensor(gpu, &source.logits, "logits")?,
            prefilled_generated_suffix_len: source.prefilled_generated_suffix_len,
            allocation_epoch: next_qwen35_state_allocation_epoch(),
        })
    }

    /// Move the active model's live KV/DeltaNet/logits state out into an owned
    /// session snapshot (the "park" half of a session swap), leaving the slot
    /// ready to receive another session.
    pub fn take_from_loaded(
        m: &mut LoadedModel,
        gpu: &mut rdna_compute::Gpu,
    ) -> Result<Self, String> {
        if m.sequence_state.is_none() {
            return Err("qwen35 session missing decode state".to_string());
        }
        let scratch = m
            .q35_scratch
            .as_ref()
            .ok_or_else(|| "qwen35 session missing scratch/logits".to_string())?;
        let logits = gpu
            .alloc_tensor(&scratch.logits.shape, scratch.logits.dtype)
            .map_err(|e| format!("alloc qwen35 session logits snapshot: {e:?}"))?;
        gpu.memcpy_dtod_auto(&logits.buf, &scratch.logits.buf, scratch.logits.buf.size())
            .map_err(|e| format!("save qwen35 session logits snapshot: {e:?}"))?;
        Ok(Self {
            seq_pos: m.seq_pos,
            conversation_tokens: std::mem::take(&mut m.conversation_tokens),
            prefix_hash: None,
            sequence_state: m.sequence_state.take().expect("checked is_none above"),
            logits,
            prefilled_generated_suffix_len: m.q35_active_prefilled_generated_suffix_len,
            allocation_epoch: next_qwen35_state_allocation_epoch(),
        })
    }

    /// Install this saved session back into the active model slot (the
    /// "activate" half of a session swap), restoring its KV/DeltaNet/logits and
    /// the KV cursor so generation resumes mid-conversation.
    pub fn restore_into_loaded(
        self,
        m: &mut LoadedModel,
        gpu: &mut rdna_compute::Gpu,
    ) -> Result<(), String> {
        let allocation_epoch = self.allocation_epoch;
        if let Some(scratch) = m.q35_scratch.as_ref() {
            gpu.memcpy_dtod_auto(
                &scratch.logits.buf,
                &self.logits.buf,
                scratch.logits.buf.size(),
            )
            .map_err(|e| format!("restore qwen35 session logits snapshot: {e:?}"))?;
        }
        m.seq_pos = self.seq_pos;
        m.conversation_tokens = self.conversation_tokens;
        // Prefix hash metadata is kept with saved Qwen35 request sessions.
        // The loaded singleton path computes it when checkpointable prefill
        // sessions are saved back into the session map.
        restore_sequence_state_into_model(m, self.sequence_state);
        m.q35_active_state_allocation_epoch = allocation_epoch;
        m.q35_active_prefilled_generated_suffix_len = self.prefilled_generated_suffix_len;
        Ok(())
    }

    pub fn reset(&mut self, gpu: &mut rdna_compute::Gpu) {
        self.seq_pos = 0;
        self.conversation_tokens.clear();
        self.prefix_hash = None;
        self.prefilled_generated_suffix_len = 0;
        for s in &self.dn_state().s_matrices {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        for s in &self.dn_state().s_scales {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        for s in &self.dn_state().conv_states {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        self.kv_cache_mut().compact_offset = 0;
    }
}

#[cfg(feature = "arch-lfm2moe")]
pub struct Lfm2RequestSessionState {
    pub seq_pos: usize,
    pub conversation_tokens: Vec<u32>,
    pub prefix_hash: Option<SequenceStatePrefixHash>,
    pub state: lfm2moe::lfm2moe::Lfm2MoeState,
    pub dflash_target_hidden_host: Option<Vec<f32>>,
    pub allocation_epoch: u64,
}

#[cfg(feature = "arch-lfm2moe")]
impl Lfm2RequestSessionState {
    pub fn new(
        gpu: &mut rdna_compute::Gpu,
        config: &lfm2moe::config::Lfm2MoeConfig,
        max_seq: usize,
        physical_cap: usize,
    ) -> Result<Self, String> {
        let state = lfm2moe::lfm2moe::Lfm2MoeState::new_with_physical_cap(
            gpu,
            config,
            max_seq,
            physical_cap,
        )
        .map_err(|e| format!("lfm2 session state allocation failed: {e}"))?;
        Ok(Self {
            seq_pos: 0,
            conversation_tokens: Vec::new(),
            prefix_hash: None,
            state,
            dflash_target_hidden_host: None,
            allocation_epoch: next_qwen35_state_allocation_epoch(),
        })
    }

    pub fn take_from_loaded(m: &mut LoadedModel) -> Result<Self, String> {
        let state = m
            .lfm2moe_state
            .take()
            .ok_or_else(|| "lfm2 active session missing state".to_string())?;
        Ok(Self {
            seq_pos: state.n_tokens,
            conversation_tokens: std::mem::take(&mut m.conversation_tokens),
            prefix_hash: None,
            state,
            dflash_target_hidden_host: m
                .lfm2_dflash
                .as_mut()
                .map(|df| std::mem::take(&mut df.target_hidden_host)),
            allocation_epoch: m.lfm2_active_state_allocation_epoch,
        })
    }

    pub fn restore_into_loaded(self, m: &mut LoadedModel) {
        m.seq_pos = self.seq_pos;
        m.conversation_tokens = self.conversation_tokens;
        m.lfm2moe_state = Some(self.state);
        if let Some(df) = m.lfm2_dflash.as_mut() {
            df.target_hidden_host = self.dflash_target_hidden_host.unwrap_or_default();
        }
        m.lfm2_active_state_allocation_epoch = self.allocation_epoch;
    }

    pub fn reset(&mut self, gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
        self.state.reset(gpu)?;
        self.seq_pos = 0;
        self.conversation_tokens.clear();
        self.prefix_hash = None;
        if let Some(hidden) = self.dflash_target_hidden_host.as_mut() {
            hidden.clear();
        }
        Ok(())
    }

    fn clone_device_buffer(
        gpu: &mut rdna_compute::Gpu,
        buf: &hip_bridge::DeviceBuffer,
        label: &str,
    ) -> Result<hip_bridge::DeviceBuffer, String> {
        let buffer_size = buf.size();
        gpu.bind_thread()
            .map_err(|e| format!("clone lfm2 checkpoint {label} bind gpu: {e:?}"))?;
        let dst = gpu
            .hip
            .malloc(buffer_size)
            .map_err(|e| format!("clone lfm2 checkpoint {label} alloc: {e:?}"))?;
        gpu.hip
            .memcpy_dtod_at(&dst, 0, buf, 0, buffer_size)
            .map_err(|e| format!("clone lfm2 checkpoint {label} copy: {e:?}"))?;
        Ok(dst)
    }

    fn clone_state(
        gpu: &mut rdna_compute::Gpu,
        state: &lfm2moe::lfm2moe::Lfm2MoeState,
    ) -> Result<lfm2moe::lfm2moe::Lfm2MoeState, String> {
        Ok(lfm2moe::lfm2moe::Lfm2MoeState {
            kv: Qwen35RequestSessionState::clone_kv_cache(gpu, &state.kv)?,
            conv_states: Qwen35RequestSessionState::clone_gpu_tensor_vec(
                gpu,
                &state.conv_states,
                "lfm2.conv_states",
            )?,
            pos_buf: Self::clone_device_buffer(gpu, &state.pos_buf, "lfm2.pos_buf")?,
            graph_warmed_up: false,
            max_seq: state.max_seq,
            n_tokens: state.n_tokens,
            h: Qwen35RequestSessionState::clone_gpu_tensor(gpu, &state.h, "lfm2.h")?,
            tmp: Qwen35RequestSessionState::clone_gpu_tensor(gpu, &state.tmp, "lfm2.tmp")?,
            fa_q: Qwen35RequestSessionState::clone_gpu_tensor(gpu, &state.fa_q, "lfm2.fa_q")?,
            fa_k: Qwen35RequestSessionState::clone_gpu_tensor(gpu, &state.fa_k, "lfm2.fa_k")?,
            fa_v: Qwen35RequestSessionState::clone_gpu_tensor(gpu, &state.fa_v, "lfm2.fa_v")?,
            fa_attn_out: Qwen35RequestSessionState::clone_gpu_tensor(
                gpu,
                &state.fa_attn_out,
                "lfm2.fa_attn_out",
            )?,
            conv_bcx: Qwen35RequestSessionState::clone_gpu_tensor(
                gpu,
                &state.conv_bcx,
                "lfm2.conv_bcx",
            )?,
            conv_y: Qwen35RequestSessionState::clone_gpu_tensor(gpu, &state.conv_y, "lfm2.conv_y")?,
            ffn_tmp: Qwen35RequestSessionState::clone_gpu_tensor(
                gpu,
                &state.ffn_tmp,
                "lfm2.ffn_tmp",
            )?,
            ffn_x_rot: Qwen35RequestSessionState::clone_gpu_tensor(
                gpu,
                &state.ffn_x_rot,
                "lfm2.ffn_x_rot",
            )?,
            dense_gate: Qwen35RequestSessionState::clone_gpu_tensor(
                gpu,
                &state.dense_gate,
                "lfm2.dense_gate",
            )?,
            dense_up: Qwen35RequestSessionState::clone_gpu_tensor(
                gpu,
                &state.dense_up,
                "lfm2.dense_up",
            )?,
            dense_act: Qwen35RequestSessionState::clone_gpu_tensor(
                gpu,
                &state.dense_act,
                "lfm2.dense_act",
            )?,
            router_logits: Qwen35RequestSessionState::clone_gpu_tensor(
                gpu,
                &state.router_logits,
                "lfm2.router_logits",
            )?,
            topk_indices: Qwen35RequestSessionState::clone_gpu_tensor(
                gpu,
                &state.topk_indices,
                "lfm2.topk_indices",
            )?,
            topk_weights: Qwen35RequestSessionState::clone_gpu_tensor(
                gpu,
                &state.topk_weights,
                "lfm2.topk_weights",
            )?,
            gate_batch: Qwen35RequestSessionState::clone_gpu_tensor(
                gpu,
                &state.gate_batch,
                "lfm2.gate_batch",
            )?,
            up_batch: Qwen35RequestSessionState::clone_gpu_tensor(
                gpu,
                &state.up_batch,
                "lfm2.up_batch",
            )?,
            rot_batch: Qwen35RequestSessionState::clone_gpu_tensor(
                gpu,
                &state.rot_batch,
                "lfm2.rot_batch",
            )?,
            down_expanded: Qwen35RequestSessionState::clone_gpu_tensor(
                gpu,
                &state.down_expanded,
                "lfm2.down_expanded",
            )?,
            final_norm_buf: Qwen35RequestSessionState::clone_gpu_tensor(
                gpu,
                &state.final_norm_buf,
                "lfm2.final_norm_buf",
            )?,
            logits: Qwen35RequestSessionState::clone_gpu_tensor(gpu, &state.logits, "lfm2.logits")?,
        })
    }

    pub fn fork_from(
        gpu: &mut rdna_compute::Gpu,
        source: &Lfm2RequestSessionState,
    ) -> Result<Self, String> {
        Ok(Self {
            seq_pos: source.seq_pos,
            conversation_tokens: source.conversation_tokens.clone(),
            prefix_hash: source.prefix_hash.clone(),
            state: Self::clone_state(gpu, &source.state)?,
            dflash_target_hidden_host: source.dflash_target_hidden_host.clone(),
            allocation_epoch: next_qwen35_state_allocation_epoch(),
        })
    }
}

/// Install a session's unified `SequenceState` as the model's active state.
pub(crate) fn restore_sequence_state_into_model(m: &mut LoadedModel, ss: SequenceState) {
    m.sequence_state = Some(ss);
}

/// Build the model's active `SequenceState` from raw KV + DeltaNet parts (e.g.
/// tearing down a transient spec-decode `ModelSlot` back into the model). The
/// hybrid profile is derived from the loaded qwen35 config.
pub(crate) fn put_qwen35_state_into_model(m: &mut LoadedModel, kv: kv::KvCache, dn: DeltaNetState) {
    let profile = m
        .q35_config
        .as_ref()
        .map(|c| qwen35_mixer_profile(&c.layer_types))
        .expect("qwen35 config present when installing active state");
    m.sequence_state = Some(SequenceState::new(profile, Some(kv), Some(Box::new(dn))));
}

/// Take the model's active state out as raw KV + DeltaNet parts (e.g. to build a
/// transient spec-decode `ModelSlot`). Leaves `m.sequence_state == None`.
pub(crate) fn take_qwen35_state_from_model(
    seq: &mut Option<SequenceState>,
) -> (kv::KvCache, DeltaNetState) {
    let (kv, recurrent) = seq
        .take()
        .expect("qwen35 active state present")
        .into_parts();
    let kv = kv.expect("qwen35 active state has KV");
    let dn = *recurrent
        .expect("qwen35 active state has DeltaNet")
        .into_any()
        .downcast::<DeltaNetState>()
        .expect("qwen35 active recurrent state is DeltaNetState");
    (kv, dn)
}

pub fn qwen35_session_resident(m: &LoadedModel, session_id: &str) -> bool {
    m.q35_active_session_id.as_deref() == Some(session_id)
        || m.q35_sessions.contains_key(session_id)
}

pub fn qwen35_request_session_count(m: &LoadedModel) -> usize {
    let saved = m
        .q35_sessions
        .keys()
        .filter(|id| id.as_str() != QWEN35_LEGACY_SESSION_ID)
        .count();
    let active = usize::from(
        m.q35_active_session_id
            .as_deref()
            .is_some_and(|id| id != QWEN35_LEGACY_SESSION_ID),
    );
    saved + active
}

pub fn qwen35_state_page_descriptors(m: &LoadedModel) -> Vec<SequenceStatePageDescriptor> {
    let mut descriptors = Vec::new();
    let placement = format!("hip:arch{}:device0", m.arch_id);
    let mut push_session = |session_id: &str, session: &Qwen35RequestSessionState, role: &str| {
        if session_id == QWEN35_LEGACY_SESSION_ID {
            return;
        }
        let logical_position = session.seq_pos + session.kv_cache().compact_offset;
        let handle = qwen35_sequence_state_handle(session_id, session.allocation_epoch);
        let owns_pages = session.allocation_epoch != 0;
        let kv_bytes = session
            .kv_cache()
            .k_gpu
            .iter()
            .chain(session.kv_cache().v_gpu.iter())
            .chain(session.kv_cache().k_scales.iter())
            .chain(session.kv_cache().v_scales.iter())
            .map(|tensor| tensor.buf.size())
            .sum::<usize>();
        descriptors.push(SequenceStatePageDescriptor {
            session_id: session_id.to_string(),
            handle: handle.clone(),
            kind: SequenceStatePageKind::Kv,
            label: "qwen35.kv_cache".to_string(),
            logical_position,
            resident_bytes: kv_bytes,
            allocation_epoch: session.allocation_epoch,
            owns_pages,
            shape: vec![
                session.kv_cache().k_gpu.len(),
                session.kv_cache().physical_cap,
                session.kv_cache().n_kv_heads,
                session.kv_cache().head_dim,
            ],
            placement: placement.clone(),
            role: role.to_string(),
        });
        let dn_bytes = session
            .dn_state()
            .s_matrices
            .iter()
            .chain(session.dn_state().s_scales.iter())
            .chain(session.dn_state().conv_states.iter())
            .map(|tensor| tensor.buf.size())
            .sum::<usize>();
        descriptors.push(SequenceStatePageDescriptor {
            session_id: session_id.to_string(),
            handle: handle.clone(),
            kind: SequenceStatePageKind::DeltaNet,
            label: "qwen35.deltanet_state".to_string(),
            logical_position,
            resident_bytes: dn_bytes,
            allocation_epoch: session.allocation_epoch,
            owns_pages,
            shape: vec![
                session.dn_state().s_matrices.len(),
                session.dn_state().s_scales.len(),
                session.dn_state().conv_states.len(),
            ],
            placement: placement.clone(),
            role: role.to_string(),
        });
        descriptors.push(SequenceStatePageDescriptor {
            session_id: session_id.to_string(),
            handle: handle.clone(),
            kind: SequenceStatePageKind::Logits,
            label: "qwen35.logits_snapshot".to_string(),
            logical_position,
            resident_bytes: session.logits.buf.size(),
            allocation_epoch: session.allocation_epoch,
            owns_pages,
            shape: session.logits.shape.clone(),
            placement: placement.clone(),
            role: role.to_string(),
        });
        descriptors.push(SequenceStatePageDescriptor {
            session_id: session_id.to_string(),
            handle,
            kind: SequenceStatePageKind::BackendPrivate,
            label: "qwen35.prefix_metadata".to_string(),
            logical_position,
            resident_bytes: session
                .prefix_hash
                .as_ref()
                .map(|hash| hash.value.len() + hash.algorithm.len() + std::mem::size_of::<usize>())
                .unwrap_or(0),
            allocation_epoch: session.allocation_epoch,
            owns_pages,
            shape: vec![usize::from(session.prefix_hash.is_some())],
            placement: "host".to_string(),
            role: role.to_string(),
        });
    };
    for (session_id, session) in &m.q35_sessions {
        push_session(session_id, session, "resident");
    }
    if let Some(active_id) = m.q35_active_session_id.as_deref() {
        if active_id != QWEN35_LEGACY_SESSION_ID {
            let compact_offset = m.kv_cache().map(|kv| kv.compact_offset).unwrap_or(0);
            let logical_position = m.seq_pos + compact_offset;
            let allocation_epoch = m.q35_active_state_allocation_epoch;
            let owns_pages = allocation_epoch != 0;
            let handle = qwen35_sequence_state_handle(active_id, allocation_epoch);
            descriptors.push(SequenceStatePageDescriptor {
                session_id: active_id.to_string(),
                handle: handle.clone(),
                kind: SequenceStatePageKind::Kv,
                label: "qwen35.kv_cache.active".to_string(),
                logical_position,
                resident_bytes: m
                    .kv_cache()
                    .map(|kv| {
                        kv.k_gpu
                            .iter()
                            .chain(kv.v_gpu.iter())
                            .chain(kv.k_scales.iter())
                            .chain(kv.v_scales.iter())
                            .map(|tensor| tensor.buf.size())
                            .sum::<usize>()
                    })
                    .unwrap_or(0),
                shape: m
                    .kv_cache()
                    .map(|kv| vec![kv.k_gpu.len(), kv.physical_cap, kv.n_kv_heads, kv.head_dim])
                    .unwrap_or_default(),
                allocation_epoch,
                owns_pages,
                placement: placement.clone(),
                role: "active".to_string(),
            });
            descriptors.push(SequenceStatePageDescriptor {
                session_id: active_id.to_string(),
                handle: handle.clone(),
                kind: SequenceStatePageKind::DeltaNet,
                label: "qwen35.deltanet_state.active".to_string(),
                logical_position,
                resident_bytes: m
                    .dn_state()
                    .map(|dn| {
                        dn.s_matrices
                            .iter()
                            .chain(dn.s_scales.iter())
                            .chain(dn.conv_states.iter())
                            .map(|tensor| tensor.buf.size())
                            .sum::<usize>()
                    })
                    .unwrap_or(0),
                shape: m
                    .dn_state()
                    .map(|dn| vec![dn.s_matrices.len(), dn.s_scales.len(), dn.conv_states.len()])
                    .unwrap_or_default(),
                allocation_epoch,
                owns_pages,
                placement: placement.clone(),
                role: "active".to_string(),
            });
            descriptors.push(SequenceStatePageDescriptor {
                session_id: active_id.to_string(),
                handle: handle.clone(),
                kind: SequenceStatePageKind::Logits,
                label: "qwen35.logits_snapshot.active".to_string(),
                logical_position,
                resident_bytes: m
                    .q35_scratch
                    .as_ref()
                    .map(|scratch| scratch.logits.buf.size())
                    .unwrap_or(0),
                shape: m
                    .q35_scratch
                    .as_ref()
                    .map(|scratch| scratch.logits.shape.clone())
                    .unwrap_or_default(),
                allocation_epoch,
                owns_pages,
                placement,
                role: "active".to_string(),
            });
            descriptors.push(SequenceStatePageDescriptor {
                session_id: active_id.to_string(),
                handle,
                kind: SequenceStatePageKind::BackendPrivate,
                label: "qwen35.prefix_metadata.active".to_string(),
                logical_position,
                resident_bytes: 0,
                allocation_epoch,
                owns_pages,
                shape: Vec::new(),
                placement: "host".to_string(),
                role: "active".to_string(),
            });
        }
    }
    descriptors
}

#[cfg(feature = "arch-lfm2moe")]
pub fn lfm2_session_resident(m: &LoadedModel, session_id: &str) -> bool {
    m.lfm2_active_session_id.as_deref() == Some(session_id)
        || m.lfm2_sessions.contains_key(session_id)
}

#[cfg(feature = "arch-lfm2moe")]
pub fn lfm2_request_session_count(m: &LoadedModel) -> usize {
    let saved = m
        .lfm2_sessions
        .keys()
        .filter(|id| id.as_str() != LFM2_LEGACY_SESSION_ID)
        .count();
    let active = usize::from(
        m.lfm2_active_session_id
            .as_deref()
            .is_some_and(|id| id != LFM2_LEGACY_SESSION_ID),
    );
    saved + active
}

#[cfg(feature = "arch-lfm2moe")]
fn lfm2_push_state_descriptors(
    m: &LoadedModel,
    descriptors: &mut Vec<SequenceStatePageDescriptor>,
    session_id: &str,
    state: &lfm2moe::lfm2moe::Lfm2MoeState,
    logical_position: usize,
    allocation_epoch: u64,
    role: &str,
) {
    if session_id == LFM2_LEGACY_SESSION_ID {
        return;
    }
    let placement = format!("hip:arch{}:device0", m.arch_id);
    let handle = SequenceStateHandle {
        id: session_id.to_string(),
        kind: "lfm2_session".to_string(),
        generation: allocation_epoch,
    };
    descriptors.push(SequenceStatePageDescriptor {
        session_id: session_id.to_string(),
        handle: handle.clone(),
        kind: SequenceStatePageKind::Kv,
        label: "lfm2.kv_cache".to_string(),
        logical_position,
        resident_bytes: kv_cache_bytes(&state.kv),
        allocation_epoch,
        owns_pages: true,
        shape: vec![
            state.kv.k_gpu.len(),
            state.kv.physical_cap,
            state.kv.n_kv_heads,
            state.kv.head_dim,
        ],
        placement: placement.clone(),
        role: role.to_string(),
    });
    let conv_shape = m
        .lfm2moe_config
        .as_ref()
        .map(|cfg| {
            vec![
                state.conv_states.len(),
                cfg.hidden_size,
                cfg.conv_kernel_size - 1,
            ]
        })
        .unwrap_or_else(|| vec![state.conv_states.len()]);
    descriptors.push(SequenceStatePageDescriptor {
        session_id: session_id.to_string(),
        handle: handle.clone(),
        kind: SequenceStatePageKind::BackendPrivate,
        label: "lfm2.short_conv_state".to_string(),
        logical_position,
        resident_bytes: tensor_vec_bytes(&state.conv_states),
        allocation_epoch,
        owns_pages: true,
        shape: conv_shape,
        placement: placement.clone(),
        role: role.to_string(),
    });
    descriptors.push(SequenceStatePageDescriptor {
        session_id: session_id.to_string(),
        handle,
        kind: SequenceStatePageKind::Logits,
        label: "lfm2.logits".to_string(),
        logical_position,
        resident_bytes: tensor_bytes(&state.logits),
        allocation_epoch,
        owns_pages: true,
        shape: state.logits.shape.clone(),
        placement,
        role: role.to_string(),
    });
}

#[cfg(feature = "arch-lfm2moe")]
pub fn lfm2_session_page_descriptors(m: &LoadedModel) -> Vec<SequenceStatePageDescriptor> {
    let mut descriptors = Vec::new();
    for (session_id, session) in &m.lfm2_sessions {
        lfm2_push_state_descriptors(
            m,
            &mut descriptors,
            session_id,
            &session.state,
            session.state.n_tokens + session.state.kv.compact_offset,
            session.allocation_epoch,
            "resident",
        );
    }
    if let (Some(active_id), Some(state)) = (
        m.lfm2_active_session_id.as_deref(),
        m.lfm2moe_state.as_ref(),
    ) {
        lfm2_push_state_descriptors(
            m,
            &mut descriptors,
            active_id,
            state,
            state.n_tokens + state.kv.compact_offset,
            m.lfm2_active_state_allocation_epoch,
            "active",
        );
    }
    descriptors
}

#[cfg(not(feature = "arch-lfm2moe"))]
pub fn lfm2_request_session_count(_m: &LoadedModel) -> usize {
    0
}

#[cfg(not(feature = "arch-lfm2moe"))]
pub fn lfm2_session_page_descriptors(_m: &LoadedModel) -> Vec<SequenceStatePageDescriptor> {
    Vec::new()
}

pub fn backend_owned_session_id(m: &LoadedModel) -> &'static str {
    match m.arch_id {
        ARCH_ID_MINIMAX_M2 => "minimax:active",
        ARCH_ID_LFM2_MOE => "lfm2:active",
        ARCH_ID_NEMOTRON_H => "nemotron:active",
        _ => "backend-owned:active",
    }
}

fn backend_owned_sequence_state_handle(m: &LoadedModel) -> SequenceStateHandle {
    SequenceStateHandle {
        id: backend_owned_session_id(m).to_string(),
        kind: "backend_owned_session".to_string(),
        generation: 0,
    }
}

pub fn backend_owned_state_page_descriptors(m: &LoadedModel) -> Vec<SequenceStatePageDescriptor> {
    match m.arch_id {
        ARCH_ID_MINIMAX_M2 => minimax_state_page_descriptors(m),
        ARCH_ID_LFM2_MOE => lfm2_state_page_descriptors(m),
        ARCH_ID_NEMOTRON_H => nemotron_state_page_descriptors(m),
        _ => Vec::new(),
    }
}

pub fn minimax_state_page_descriptors(m: &LoadedModel) -> Vec<SequenceStatePageDescriptor> {
    let Some(state) = m.minimax_state.as_ref() else {
        return Vec::new();
    };
    let session_id = backend_owned_session_id(m).to_string();
    let handle = backend_owned_sequence_state_handle(m);
    let logical_position = m.seq_pos;
    let placement = format!("hip:arch{}:device0", m.arch_id);
    let kv_bytes = kv_cache_bytes(&state.kv);
    let logits_bytes = tensor_bytes(&state.logits);
    let private_bytes = minimax_state_bytes(state)
        .saturating_sub(kv_bytes)
        .saturating_sub(logits_bytes);
    vec![
        SequenceStatePageDescriptor {
            session_id: session_id.clone(),
            handle: handle.clone(),
            kind: SequenceStatePageKind::Kv,
            label: "minimax.kv_cache".to_string(),
            logical_position,
            resident_bytes: kv_bytes,
            allocation_epoch: handle.generation,
            owns_pages: false,
            shape: vec![
                state.kv.k_gpu.len(),
                state.kv.physical_cap,
                state.kv.n_kv_heads,
                state.kv.head_dim,
            ],
            placement: placement.clone(),
            role: "active_backend_owned".to_string(),
        },
        SequenceStatePageDescriptor {
            session_id: session_id.clone(),
            handle: handle.clone(),
            kind: SequenceStatePageKind::Logits,
            label: "minimax.logits".to_string(),
            logical_position,
            resident_bytes: logits_bytes,
            allocation_epoch: handle.generation,
            owns_pages: false,
            shape: state.logits.shape.clone(),
            placement: placement.clone(),
            role: "active_backend_owned".to_string(),
        },
        SequenceStatePageDescriptor {
            session_id,
            handle,
            kind: SequenceStatePageKind::BackendPrivate,
            label: "minimax.decode_scratch".to_string(),
            logical_position,
            resident_bytes: private_bytes,
            allocation_epoch: 0,
            owns_pages: false,
            shape: vec![1],
            placement,
            role: "active_backend_owned".to_string(),
        },
    ]
}

#[cfg(feature = "arch-lfm2moe")]
pub fn lfm2_state_page_descriptors(m: &LoadedModel) -> Vec<SequenceStatePageDescriptor> {
    let Some(state) = m.lfm2moe_state.as_ref() else {
        return Vec::new();
    };
    let session_id = backend_owned_session_id(m).to_string();
    let handle = backend_owned_sequence_state_handle(m);
    let logical_position = m.seq_pos;
    let placement = format!("hip:arch{}:device0", m.arch_id);
    let conv_bytes = tensor_vec_bytes(&state.conv_states);
    let conv_shape = m
        .lfm2moe_config
        .as_ref()
        .map(|cfg| {
            vec![
                state.conv_states.len(),
                cfg.hidden_size,
                cfg.conv_kernel_size - 1,
            ]
        })
        .unwrap_or_else(|| vec![state.conv_states.len()]);
    vec![
        SequenceStatePageDescriptor {
            session_id: session_id.clone(),
            handle: handle.clone(),
            kind: SequenceStatePageKind::Kv,
            label: "lfm2.kv_cache".to_string(),
            logical_position,
            resident_bytes: kv_cache_bytes(&state.kv),
            allocation_epoch: handle.generation,
            owns_pages: false,
            shape: vec![
                state.kv.k_gpu.len(),
                state.kv.physical_cap,
                state.kv.n_kv_heads,
                state.kv.head_dim,
            ],
            placement: placement.clone(),
            role: "active_backend_owned".to_string(),
        },
        SequenceStatePageDescriptor {
            session_id: session_id.clone(),
            handle: handle.clone(),
            kind: SequenceStatePageKind::BackendPrivate,
            label: "lfm2.short_conv_state".to_string(),
            logical_position,
            resident_bytes: conv_bytes,
            allocation_epoch: handle.generation,
            owns_pages: false,
            shape: conv_shape,
            placement: placement.clone(),
            role: "active_backend_owned".to_string(),
        },
        SequenceStatePageDescriptor {
            session_id,
            handle,
            kind: SequenceStatePageKind::Logits,
            label: "lfm2.logits".to_string(),
            logical_position,
            resident_bytes: tensor_bytes(&state.logits),
            allocation_epoch: 0,
            owns_pages: false,
            shape: state.logits.shape.clone(),
            placement,
            role: "active_backend_owned".to_string(),
        },
    ]
}

#[cfg(not(feature = "arch-lfm2moe"))]
pub fn lfm2_state_page_descriptors(_m: &LoadedModel) -> Vec<SequenceStatePageDescriptor> {
    Vec::new()
}

pub fn nemotron_state_page_descriptors(m: &LoadedModel) -> Vec<SequenceStatePageDescriptor> {
    let Some(model) = m.nemotron_backend.as_ref() else {
        return Vec::new();
    };
    let session_id = backend_owned_session_id(m).to_string();
    let handle = backend_owned_sequence_state_handle(m);
    let logical_position = m.seq_pos;
    let placement = format!("hip:arch{}:device0", m.arch_id);
    let mut descriptors = Vec::new();
    let mut push = |kind: SequenceStatePageKind,
                    label: &str,
                    resident_bytes: usize,
                    shape: Vec<usize>,
                    placement: String| {
        descriptors.push(SequenceStatePageDescriptor {
            session_id: session_id.clone(),
            handle: handle.clone(),
            kind,
            label: label.to_string(),
            logical_position,
            resident_bytes,
            allocation_epoch: handle.generation,
            owns_pages: false,
            shape,
            placement,
            role: "active_backend_owned".to_string(),
        });
    };
    if let Some((bytes, shape)) = model.attention_kv_state_summary() {
        push(
            SequenceStatePageKind::Kv,
            "nemotron.attention_kv",
            bytes,
            shape,
            placement.clone(),
        );
    }
    if let Some((bytes, shape)) = model.mamba_ssm_state_summary() {
        push(
            SequenceStatePageKind::MambaSsm,
            "nemotron.mamba_ssm",
            bytes,
            shape,
            placement.clone(),
        );
    }
    if let Some((bytes, shape)) = model.mamba_conv_state_summary() {
        push(
            SequenceStatePageKind::MambaConv,
            "nemotron.mamba_conv",
            bytes,
            shape,
            placement.clone(),
        );
    }
    let (bytes, shape) = model.logits_state_summary();
    push(
        SequenceStatePageKind::Logits,
        "nemotron.logits",
        bytes,
        shape,
        placement,
    );
    descriptors
}

/// Stable worker id for a loaded model, derived from its arch/pp/kv-mode parts.
pub fn loaded_model_worker_id(m: &LoadedModel) -> ModelWorkerId {
    ModelWorkerId::from_runtime_parts(m.arch_id, m.pp, m.q35_kv_mode.as_deref())
}

pub fn loaded_model_state_arena_backend(m: &LoadedModel) -> SequenceStateArenaBackend {
    SequenceStateArenaBackend::for_worker_parts(m.arch_id, m.pp)
}

/// Assemble the full runtime view the daemon reports for the active worker:
/// worker id, context limits, arena backend, resident-session descriptors, and
/// the memory view.
pub fn loaded_model_worker_runtime_view(m: &LoadedModel) -> ModelWorkerRuntimeView {
    let state_arena_backend = loaded_model_state_arena_backend(m);
    let state_arena_operations =
        loaded_model_state_arena_operations(m, state_arena_backend).to_vec();
    let resident_sessions = sequence_state_arena_resident_session_count(state_arena_backend, m);
    let state_page_descriptors = sequence_state_arena_page_descriptors(state_arena_backend, m);
    let memory = loaded_model_memory_view(m, &state_page_descriptors);
    ModelWorkerRuntimeView {
        worker_id: loaded_model_worker_id(m),
        max_seq: m.max_seq,
        physical_cap: m.physical_cap,
        max_resident_workers: 1,
        resident_workers: 1,
        state_arena_backend,
        state_arena_operations,
        resident_sessions,
        state_page_descriptors,
        memory,
    }
}

fn loaded_model_state_arena_operations(
    m: &LoadedModel,
    state_arena_backend: SequenceStateArenaBackend,
) -> &'static [SequenceStateArenaOperation] {
    if state_arena_backend == SequenceStateArenaBackend::BackendOwned
        && m.arch_id == ARCH_ID_LFM2_MOE
        && m.pp == 1
    {
        &[
            SequenceStateArenaOperation::AttachCheckpoint,
            SequenceStateArenaOperation::ForkCheckpoint,
            SequenceStateArenaOperation::ReleaseState,
            SequenceStateArenaOperation::DescribeState,
        ]
    } else {
        state_arena_backend.supported_operations()
    }
}

/// Extract the requested `worker_id` from a message, defaulting to
/// [`DEFAULT_MODEL_WORKER_ID`] when absent.
pub fn message_worker_id(msg: &serde_json::Value) -> String {
    parse_model_worker_id(msg, DEFAULT_MODEL_WORKER_ID).value
}

/// Park the currently-active model worker: save its live session out to the
/// resident-session map so a different worker/session can take the slot.
pub fn park_active_model(
    model: &mut Option<LoadedModel>,
    gpu: &mut rdna_compute::Gpu,
    active_worker_id: &str,
    resident_models: &mut std::collections::HashMap<String, LoadedModel>,
) -> Result<(), String> {
    if let Some(m) = model.as_mut() {
        if is_qwen35_family_arch_id(m.arch_id) && m.pp == 1 {
            qwen35_save_active_session(m, gpu)?;
        }
        #[cfg(feature = "arch-lfm2moe")]
        if m.arch_id == ARCH_ID_LFM2_MOE && m.pp == 1 {
            lfm2_save_active_session(m)?;
        }
    }
    if let Some(m) = model.take() {
        resident_models.insert(active_worker_id.to_string(), m);
    }
    Ok(())
}

pub fn validate_qwen35_fused_grouped_moe_prefill_model_capability(
    m: &LoadedModel,
    session_count: usize,
) -> Result<(), String> {
    if !is_qwen35_moe_arch_id(m.arch_id) {
        return Err(format!(
            "qwen35 grouped-MoE fused prefill-session batch worker requires arch_id=6, got {}",
            m.arch_id
        ));
    }
    if session_count < 2 {
        return Err(
            "qwen35 grouped-MoE fused prefill-session batch worker requires at least two sessions"
                .to_string(),
        );
    }
    let config = m
        .q35_config
        .as_ref()
        .ok_or_else(|| "qwen35 grouped-MoE fused prefill requires qwen35 config".to_string())?;
    if config.num_experts == 0 {
        return Err("qwen35 grouped-MoE fused prefill requires routed experts".to_string());
    }
    if !config.has_shared_expert {
        return Err("qwen35 grouped-MoE fused prefill requires a shared expert".to_string());
    }
    if config.num_experts_per_tok != 8
        && !(config.paged_experts && config.num_experts_per_tok == 10)
    {
        return Err(format!(
            "grouped MoE session fused prefix currently requires K_TOP=8, or paged K_TOP=10, got {}",
            config.num_experts_per_tok
        ));
    }
    if m.q35_scratch.is_none() {
        return Err("qwen35 grouped-MoE fused prefill requires qwen35 scratch".to_string());
    }
    if config.paged_experts {
        if let Some(weights) = m.q35_weights.as_ref() {
            qwen35::validate_paged_moe_decode_expert_cache(weights, config)?;
        }
    }
    Ok(())
}

/// Make the requested worker the active one, parking whatever was active first
/// — the single-resident-slot worker swap.
pub fn activate_model_worker(
    worker_id: &str,
    active_worker_id: &mut String,
    model: &mut Option<LoadedModel>,
    gpu: &mut rdna_compute::Gpu,
    resident_models: &mut std::collections::HashMap<String, LoadedModel>,
) -> Result<bool, String> {
    if active_worker_id == worker_id {
        return Ok(model.is_some());
    }
    if !resident_models.contains_key(worker_id) {
        return Ok(false);
    }
    park_active_model(model, gpu, active_worker_id, resident_models)?;
    if let Some(m) = resident_models.remove(worker_id) {
        *active_worker_id = worker_id.to_string();
        *model = Some(m);
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Build the `resident_worker_status` JSON the daemon emits: which workers are
/// resident, their runtime views, and accelerator inventory.
pub fn resident_worker_status_json(
    active_worker_id: &str,
    model: Option<&LoadedModel>,
    resident_models: &std::collections::HashMap<String, LoadedModel>,
) -> serde_json::Value {
    let mut workers = Vec::new();
    let mut total_model_weight_bytes = 0usize;
    let mut total_runtime_state_bytes = 0usize;
    let mut total_resident_bytes = 0usize;
    let mut total_evictable_state_bytes = 0usize;
    if let Some(m) = model {
        let worker = loaded_model_worker_runtime_view(m);
        total_model_weight_bytes += worker.memory.model_weight_bytes;
        total_runtime_state_bytes += worker.memory.runtime_state_bytes;
        total_resident_bytes += worker.memory.total_resident_bytes;
        total_evictable_state_bytes += worker.memory.evictable_state_bytes;
        let mut value = model_worker_runtime_view_json(&worker);
        value["worker_key_id"] = serde_json::json!(active_worker_id);
        value["active"] = serde_json::json!(true);
        value["model_path"] = serde_json::json!(m.model_path);
        workers.push(value);
    }
    for (worker_id, m) in resident_models {
        let worker = loaded_model_worker_runtime_view(m);
        total_model_weight_bytes += worker.memory.model_weight_bytes;
        total_runtime_state_bytes += worker.memory.runtime_state_bytes;
        total_resident_bytes += worker.memory.total_resident_bytes;
        total_evictable_state_bytes += worker.memory.evictable_state_bytes;
        let mut value = model_worker_runtime_view_json(&worker);
        value["worker_key_id"] = serde_json::json!(worker_id);
        value["active"] = serde_json::json!(false);
        value["model_path"] = serde_json::json!(m.model_path);
        workers.push(value);
    }
    serde_json::json!({
        "type": "worker_status",
        "resident_workers": workers.len(),
        "active_worker_key_id": active_worker_id,
        "total_model_weight_bytes": total_model_weight_bytes,
        "total_runtime_state_bytes": total_runtime_state_bytes,
        "total_resident_bytes": total_resident_bytes,
        "total_evictable_state_bytes": total_evictable_state_bytes,
        "workers": workers,
    })
}

pub fn daemon_accelerator_inventory(gpu: &mut rdna_compute::Gpu) -> AcceleratorInventory {
    let hip_runtime = gpu
        .hip
        .runtime_version()
        .ok()
        .map(|(major, minor)| format!("HIP {major}.{minor}"));
    let selected_device = gpu.device_id;
    let count = gpu.hip.device_count().unwrap_or(0).max(0);
    let mut devices = Vec::new();

    for ordinal in 0..count {
        let device_id = ordinal.to_string();
        if let Err(err) = gpu.hip.set_device(ordinal) {
            devices.push(AcceleratorDeviceInfo {
                kind: "hip".to_string(),
                device_id,
                ordinal: Some(ordinal as usize),
                available: false,
                selected: ordinal == selected_device,
                reason: Some(err.to_string()),
                ..Default::default()
            });
            continue;
        }

        let arch = gpu.hip.get_arch(ordinal).ok();
        let integrated = gpu.hip.is_integrated_device(ordinal).ok();
        let total_memory_bytes = gpu.hip.get_vram_info().ok().map(|(_, total)| total as u64);
        let mut device = AcceleratorDeviceInfo::hip(
            device_id,
            ordinal as usize,
            arch,
            total_memory_bytes,
            integrated,
            hip_runtime.clone(),
        );
        device.selected = ordinal == selected_device;
        devices.push(device);
    }

    if let Err(err) = gpu.hip.set_device(selected_device) {
        eprintln!(
            "WARNING: failed to restore HIP device {} after inventory probe: {}",
            selected_device, err
        );
    }

    devices.extend(hipfire_npu::xdna1_inventory_devices_from_env());

    AcceleratorInventory::from_devices("daemon", devices)
}

pub fn resident_state_reservation_budget_bytes() -> usize {
    std::env::var("HIPFIRE_DAEMON_RESIDENT_STATE_BUDGET_MB")
        .or_else(|_| std::env::var("HIPFIRE_SERVER_RESIDENT_STATE_BUDGET_MB"))
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|mb| mb.saturating_mul(1024 * 1024))
        .unwrap_or(16 * 1024 * 1024 * 1024)
}

pub fn describe_loaded_model_sequence_state(
    worker_id: &str,
    m: &LoadedModel,
    handle: &ParsedSequenceStateHandle,
) -> Option<DescribedSequenceState> {
    if !parsed_handle_may_target_loaded_state(handle) {
        return None;
    }
    let arena_backend = loaded_model_state_arena_backend(m);
    let descriptors = describe_sequence_state_descriptors(
        sequence_state_arena_page_descriptors(arena_backend, m),
        handle,
    )?;
    let state_arena_owns_pages = descriptors.iter().any(|descriptor| descriptor.owns_pages);
    let reserved_bytes = descriptors
        .iter()
        .map(|descriptor| descriptor.resident_bytes)
        .sum();
    Some(DescribedSequenceState {
        worker_id: worker_id.to_string(),
        handle: descriptors[0].handle.clone(),
        state_arena_owns_pages,
        reserved_bytes,
        state_page_descriptors: descriptors,
    })
}

pub fn describe_loaded_sequence_state(
    active_worker_id: &str,
    model: Option<&LoadedModel>,
    resident_models: &HashMap<String, LoadedModel>,
    handle: &ParsedSequenceStateHandle,
) -> Option<DescribedSequenceState> {
    if let Some(m) = model {
        if let Some(described) = describe_loaded_model_sequence_state(active_worker_id, m, handle) {
            return Some(described);
        }
    }
    for (worker_id, m) in resident_models {
        if let Some(described) = describe_loaded_model_sequence_state(worker_id, m, handle) {
            return Some(described);
        }
    }
    None
}

pub fn release_loaded_model_sequence_state_handles(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    handles: &[ParsedSequenceStateHandle],
) -> Result<(usize, usize), String> {
    let arena_backend = loaded_model_state_arena_backend(m);
    let mut released = 0usize;
    let mut released_bytes = 0usize;
    let mut released_session_ids = HashSet::new();
    for handle in handles {
        if !parsed_handle_may_target_loaded_state(handle)
            || released_session_ids.contains(&handle.id)
        {
            continue;
        }
        let Some(descriptors) = describe_sequence_state_descriptors(
            sequence_state_arena_page_descriptors(arena_backend, m),
            handle,
        ) else {
            continue;
        };
        let descriptor_bytes = descriptors
            .iter()
            .map(|descriptor| descriptor.resident_bytes)
            .sum::<usize>();
        let session_ids = vec![handle.id.clone()];
        let session_released =
            sequence_state_arena_release_sessions(arena_backend, m, gpu, &session_ids)?;
        if session_released > 0 {
            released += session_released;
            released_bytes = released_bytes.saturating_add(descriptor_bytes);
            released_session_ids.insert(handle.id.clone());
        }
    }
    Ok((released, released_bytes))
}

pub fn release_loaded_sequence_state_handles(
    model: &mut Option<LoadedModel>,
    resident_models: &mut HashMap<String, LoadedModel>,
    gpu: &mut rdna_compute::Gpu,
    handles: &[ParsedSequenceStateHandle],
) -> Result<(usize, usize), String> {
    let mut released = 0usize;
    let mut released_bytes = 0usize;
    if let Some(m) = model.as_mut() {
        let (count, bytes) = release_loaded_model_sequence_state_handles(m, gpu, handles)?;
        released += count;
        released_bytes = released_bytes.saturating_add(bytes);
    }
    for m in resident_models.values_mut() {
        let (count, bytes) = release_loaded_model_sequence_state_handles(m, gpu, handles)?;
        released += count;
        released_bytes = released_bytes.saturating_add(bytes);
    }
    Ok((released, released_bytes))
}

/// Drop the named saved sessions (freeing their GPU state); returns how many
/// were actually resident.
pub fn qwen35_release_sessions(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    session_ids: &[String],
) -> Result<usize, String> {
    if !is_qwen35_family_arch_id(m.arch_id) || m.pp != 1 {
        return Err(format!(
            "release_sessions currently supports single-GPU qwen35/qwen35-moe only (arch_id={} pp={})",
            m.arch_id, m.pp
        ));
    }

    let mut released = 0usize;
    for session_id in session_ids {
        if session_id == QWEN35_LEGACY_SESSION_ID {
            continue;
        }
        if m.q35_active_session_id.as_deref() == Some(session_id.as_str()) {
            qwen35_save_active_session(m, gpu)?;
        }
        if m.q35_sessions.remove(session_id).is_some() {
            released += 1;
        }
    }

    if m.q35_active_session_id.is_none() {
        let created = qwen35_activate_session(m, gpu, QWEN35_LEGACY_SESSION_ID)?;
        if created {
            qwen35_reset_active_session(m, gpu)?;
        }
    }

    Ok(released)
}

/// Absolute logical position (token count) of the active qwen35 session — the
/// resume point for the next prefill/decode.
pub fn qwen35_active_logical_position(m: &LoadedModel) -> Result<usize, String> {
    let compact_offset = m
        .kv_cache()
        .ok_or_else(|| "qwen35 active session missing KV cache".to_string())?
        .compact_offset;
    Ok(m.seq_pos + compact_offset)
}

/// Per-layer token-mixer profile for a qwen3.5 hybrid stack: `FullAttention`
/// layers are KV-backed [`MixerKind::FullAttn`], `LinearAttention` layers are
/// recurrent [`MixerKind::DeltaNet`] (no KV). The KV allocator consumes
/// [`MixerProfile::kv_layer_mask`] to skip the recurrent layers — the neutral
/// replacement for the hand-rolled `layer_types == FullAttention` mask. See
/// docs/plans/2026-06-23-seam-finish-and-mamba2.md (P2b).
pub(crate) fn qwen35_mixer_profile(layer_types: &[LayerType]) -> MixerProfile {
    MixerProfile::new(
        layer_types
            .iter()
            .map(|t| match t {
                LayerType::FullAttention => MixerKind::FullAttn,
                LayerType::LinearAttention => MixerKind::DeltaNet,
            })
            .collect(),
    )
}

/// Allocate (or reuse) the resident session-state slot for a session id,
/// parking any other active session first; the entry point that makes a session
/// the live one before prefill.
pub fn qwen35_allocate_session_state(
    m: &LoadedModel,
    gpu: &mut rdna_compute::Gpu,
) -> Result<Qwen35RequestSessionState, String> {
    let config = m
        .q35_config
        .as_ref()
        .ok_or_else(|| "qwen35 config missing".to_string())?;
    let kv_mode = m
        .q35_kv_mode
        .as_deref()
        .ok_or_else(|| "qwen35 KV mode missing; reload model before batch prefill".to_string())?;
    let kv_cache = match kv_mode {
        "fp32" | "f32" => {
            let is_kv_layer = qwen35_mixer_profile(&config.layer_types).kv_layer_mask();
            kv::KvCache::new_gpu_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                m.max_seq,
            )
            .map_err(|e| format!("{e}"))?
        }
        "q8" => kv::KvCache::new_gpu_q8_capped(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            m.max_seq,
            m.physical_cap,
        )
        .map_err(|e| format!("{e}"))?,
        "asym4" | "turbo4" => kv::KvCache::new_gpu_asym4_capped(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            m.max_seq,
            m.physical_cap,
        )
        .map_err(|e| format!("{e}"))?,
        "asym2" | "turbo2" => kv::KvCache::new_gpu_asym2_capped(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            m.max_seq,
            m.physical_cap,
        )
        .map_err(|e| format!("{e}"))?,
        "asym3" | "turbo3" | "turbo" if config.head_dim == 256 => {
            kv::KvCache::new_gpu_asym3_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                m.max_seq,
                m.physical_cap,
            )
            .map_err(|e| format!("{e}"))?
        }
        "auto" | "" if config.head_dim == 256 => kv::KvCache::new_gpu_asym3_capped(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            m.max_seq,
            m.physical_cap,
        )
        .map_err(|e| format!("{e}"))?,
        "auto" | "" => kv::KvCache::new_gpu_q8_capped(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            m.max_seq,
            m.physical_cap,
        )
        .map_err(|e| format!("{e}"))?,
        "asym3" | "turbo3" | "turbo" => {
            return Err(format!(
                "qwen35 batch-prefill KV mode {kv_mode} requires head_dim=256 (got {})",
                config.head_dim
            ));
        }
        other => {
            eprintln!("  batch-prefill KV cache: unrecognized '{other}', defaulting to asym3");
            kv::KvCache::new_gpu_asym3_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                m.max_seq,
                m.physical_cap,
            )
            .map_err(|e| format!("{e}"))?
        }
    };
    let dn_quant = m.q35_state_quant.ok_or_else(|| {
        "qwen35 DeltaNet state quant missing; reload model before batch prefill".to_string()
    })?;
    let dn_state = DeltaNetState::new_with_quant(gpu, config, dn_quant)
        .map_err(|e| format!("DeltaNetState::new_with_quant: {e:?}"))?;
    let sequence_state = SequenceState::new(
        qwen35_mixer_profile(&config.layer_types),
        Some(kv_cache),
        Some(Box::new(dn_state)),
    );
    Ok(Qwen35RequestSessionState {
        seq_pos: 0,
        conversation_tokens: Vec::new(),
        prefix_hash: None,
        sequence_state,
        logits: gpu
            .alloc_tensor(&[config.vocab_size], rdna_compute::DType::F32)
            .map_err(|e| format!("alloc qwen35 session logits snapshot: {e:?}"))?,
        prefilled_generated_suffix_len: 0,
        allocation_epoch: next_qwen35_state_allocation_epoch(),
    })
}

/// Snapshot the active session's live state back into the resident-session map
/// without giving up the slot (checkpoint without swap).
pub fn qwen35_save_active_session(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
) -> Result<(), String> {
    if let Some(active_id) = m.q35_active_session_id.take() {
        let session = Qwen35RequestSessionState::take_from_loaded(m, gpu)
            .map_err(|e| format!("failed to save active qwen35 session: {e}"))?;
        m.q35_sessions.insert(active_id, session);
        m.q35_active_state_allocation_epoch = 0;
    }
    Ok(())
}

/// Restore a saved session into the active slot (parking the current one),
/// resuming its multi-turn KV/DeltaNet state.
pub fn qwen35_activate_session(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    session_id: &str,
) -> Result<bool, String> {
    if m.q35_active_session_id.as_deref() == Some(session_id) {
        return Ok(false);
    }
    let existed = m.q35_sessions.contains_key(session_id);
    qwen35_save_active_session(m, gpu)?;
    let session = match m.q35_sessions.remove(session_id) {
        Some(session) => session,
        None => qwen35_allocate_session_state(m, gpu)?,
    };
    session.restore_into_loaded(m, gpu)?;
    m.q35_active_session_id = Some(session_id.to_string());
    Ok(!existed)
}

#[cfg(feature = "arch-lfm2moe")]
pub fn lfm2_save_active_session(m: &mut LoadedModel) -> Result<(), String> {
    if let Some(active_id) = m.lfm2_active_session_id.take() {
        let mut session = Lfm2RequestSessionState::take_from_loaded(m)
            .map_err(|e| format!("failed to save active lfm2 session: {e}"))?;
        session.seq_pos = session.state.n_tokens;
        m.lfm2_sessions.insert(active_id, session);
        m.lfm2_active_state_allocation_epoch = 0;
    }
    Ok(())
}

#[cfg(feature = "arch-lfm2moe")]
pub fn lfm2_allocate_session_state(
    m: &LoadedModel,
    gpu: &mut rdna_compute::Gpu,
) -> Result<Lfm2RequestSessionState, String> {
    let config = m
        .lfm2moe_config
        .as_ref()
        .ok_or_else(|| "lfm2 config missing".to_string())?;
    Lfm2RequestSessionState::new(gpu, config, m.max_seq, m.physical_cap)
}

#[cfg(feature = "arch-lfm2moe")]
pub fn lfm2_activate_session(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    session_id: &str,
) -> Result<bool, String> {
    if m.lfm2_active_session_id.as_deref() == Some(session_id) {
        return Ok(false);
    }
    let existed = m.lfm2_sessions.contains_key(session_id);
    lfm2_save_active_session(m)?;
    let session = match m.lfm2_sessions.remove(session_id) {
        Some(session) => session,
        None => lfm2_allocate_session_state(m, gpu)?,
    };
    session.restore_into_loaded(m);
    m.lfm2_active_session_id = Some(session_id.to_string());
    Ok(!existed)
}

#[cfg(feature = "arch-lfm2moe")]
pub fn lfm2_reset_active_session(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
) -> Result<(), String> {
    let state = m
        .lfm2moe_state
        .as_mut()
        .ok_or_else(|| "lfm2 active session missing state".to_string())?;
    state.reset(gpu)?;
    if let Some(df) = m.lfm2_dflash.as_mut() {
        df.target_hidden_host.clear();
    }
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    Ok(())
}

#[cfg(feature = "arch-lfm2moe")]
pub fn lfm2_active_logical_position(m: &LoadedModel) -> Result<usize, String> {
    let state = m
        .lfm2moe_state
        .as_ref()
        .ok_or_else(|| "lfm2 active session missing state".to_string())?;
    Ok(state.n_tokens + state.kv.compact_offset)
}

#[cfg(feature = "arch-lfm2moe")]
pub fn lfm2_release_sessions(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    session_ids: &[String],
) -> Result<usize, String> {
    if m.arch_id != ARCH_ID_LFM2_MOE || m.pp != 1 {
        return Err(format!(
            "release_sessions for LFM2 requires arch_id=11 pp=1 (arch_id={} pp={})",
            m.arch_id, m.pp
        ));
    }
    let mut released = 0usize;
    for session_id in session_ids {
        if session_id == LFM2_LEGACY_SESSION_ID {
            continue;
        }
        if m.lfm2_active_session_id.as_deref() == Some(session_id.as_str()) {
            lfm2_save_active_session(m)?;
        }
        if m.lfm2_sessions.remove(session_id).is_some() {
            released += 1;
        }
    }
    if m.lfm2_active_session_id.is_none() {
        let created = lfm2_activate_session(m, gpu, LFM2_LEGACY_SESSION_ID)?;
        if created {
            lfm2_reset_active_session(m, gpu)?;
        }
    }
    Ok(released)
}

#[cfg(feature = "arch-lfm2moe")]
pub fn lfm2_validate_prefix_hash(
    m: &LoadedModel,
    source_session_id: &str,
    requested: Option<&SequenceStatePrefixHash>,
) -> Result<(), String> {
    validate_checkpoint_source_resident(
        source_session_id,
        m.lfm2_sessions.contains_key(source_session_id),
    )?;
    let source = m
        .lfm2_sessions
        .get(source_session_id)
        .expect("source residency was validated");
    validate_checkpoint_prefix_hash(source_session_id, source.prefix_hash.as_ref(), requested)
}

#[cfg(feature = "arch-lfm2moe")]
pub fn lfm2_fork_session_state(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    request: SequenceStateForkRequest<'_>,
) -> Result<(), String> {
    lfm2_save_active_session(m)?;
    lfm2_validate_prefix_hash(m, request.source_session_id, request.requested_prefix_hash)?;
    if request.source_session_id == request.dest_session_id {
        return Ok(());
    }
    validate_checkpoint_source_resident(
        request.source_session_id,
        m.lfm2_sessions.contains_key(request.source_session_id),
    )?;
    let source = m
        .lfm2_sessions
        .get(request.source_session_id)
        .expect("source residency was validated");
    let forked = Lfm2RequestSessionState::fork_from(gpu, source)?;
    m.lfm2_sessions
        .insert(request.dest_session_id.to_string(), forked);
    Ok(())
}

#[cfg(feature = "arch-lfm2moe")]
pub fn lfm2_checkpoint_session_state(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    request: SequenceStateCheckpointRequest<'_>,
) -> Result<(), String> {
    if request.source_session_id == request.dest_session_id {
        return Ok(());
    }
    lfm2_save_active_session(m)?;
    {
        validate_checkpoint_source_resident(
            request.source_session_id,
            m.lfm2_sessions.contains_key(request.source_session_id),
        )?;
        let source = m
            .lfm2_sessions
            .get(request.source_session_id)
            .expect("source residency was validated");
        let logical_position = source.state.n_tokens + source.state.kv.compact_offset;
        validate_checkpoint_logical_position(
            request.source_session_id,
            request.expected_logical_position,
            logical_position,
        )?;
    }
    if let Some(prefix_hash) = request.checkpoint_prefix_hash {
        if let Some(source) = m.lfm2_sessions.get_mut(request.source_session_id) {
            source.prefix_hash = Some(prefix_hash.clone());
        }
    }
    lfm2_fork_session_state(
        m,
        gpu,
        SequenceStateForkRequest {
            source_session_id: request.source_session_id,
            dest_session_id: request.dest_session_id,
            requested_prefix_hash: request.requested_prefix_hash,
        },
    )
}

/// Fork a saved session into a new session id (deep-copying its state), so a
/// conversation can branch without disturbing the original.
pub fn qwen35_fork_session_state(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    request: SequenceStateForkRequest<'_>,
) -> Result<(), String> {
    if request.source_session_id == request.dest_session_id {
        return Ok(());
    }
    let source_is_active = m.q35_active_session_id.as_deref() == Some(request.source_session_id);
    if !source_is_active {
        qwen35_validate_prefix_hash(m, request.source_session_id, request.requested_prefix_hash)?;
    }
    qwen35_save_active_session(m, gpu)?;
    if source_is_active {
        if let Err(err) =
            qwen35_validate_prefix_hash(m, request.source_session_id, request.requested_prefix_hash)
        {
            let _ = qwen35_activate_session(m, gpu, request.source_session_id);
            return Err(err);
        }
    }
    validate_checkpoint_source_resident(
        request.source_session_id,
        m.q35_sessions.contains_key(request.source_session_id),
    )?;
    let source = m
        .q35_sessions
        .get(request.source_session_id)
        .expect("source residency was validated");
    let forked = Qwen35RequestSessionState::fork_from(gpu, source)?;
    m.q35_sessions
        .insert(request.dest_session_id.to_string(), forked);
    Ok(())
}

/// Checkpoint a session at a validated logical position / prefix hash, after
/// verifying the request matches the resident state (guards against stale or
/// mismatched checkpoint requests).
pub fn qwen35_checkpoint_session_state(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    request: SequenceStateCheckpointRequest<'_>,
) -> Result<(), String> {
    if request.source_session_id == request.dest_session_id {
        return Ok(());
    }
    qwen35_save_active_session(m, gpu)?;
    {
        validate_checkpoint_source_resident(
            request.source_session_id,
            m.q35_sessions.contains_key(request.source_session_id),
        )?;
        let source = m
            .q35_sessions
            .get(request.source_session_id)
            .expect("source residency was validated");
        let logical_position = source.seq_pos + source.kv_cache().compact_offset;
        validate_checkpoint_logical_position(
            request.source_session_id,
            request.expected_logical_position,
            logical_position,
        )?;
    }
    if let Some(prefix_hash) = request.checkpoint_prefix_hash {
        if let Some(source) = m.q35_sessions.get_mut(request.source_session_id) {
            source.prefix_hash = Some(prefix_hash.clone());
        }
    }
    qwen35_fork_session_state(
        m,
        gpu,
        SequenceStateForkRequest {
            source_session_id: request.source_session_id,
            dest_session_id: request.dest_session_id,
            requested_prefix_hash: request.requested_prefix_hash,
        },
    )
}

/// Check a request's claimed prefix hash against the session's recorded hash —
/// the prefix-cache safety check that prevents resuming on a divergent prompt.
pub fn qwen35_validate_prefix_hash(
    m: &LoadedModel,
    source_session_id: &str,
    requested: Option<&SequenceStatePrefixHash>,
) -> Result<(), String> {
    validate_checkpoint_source_resident(
        source_session_id,
        m.q35_sessions.contains_key(source_session_id),
    )?;
    let source = m
        .q35_sessions
        .get(source_session_id)
        .expect("source residency was validated");
    validate_checkpoint_prefix_hash(source_session_id, source.prefix_hash.as_ref(), requested)
}

/// Reset the active session's KV cursor to cold (rewind to position 0) without
/// freeing the allocation — a cheap O(1) restart for a fresh turn.
pub fn qwen35_reset_active_session(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
) -> Result<(), String> {
    let mut session = Qwen35RequestSessionState::take_from_loaded(m, gpu)
        .map_err(|e| format!("failed to reset qwen35 session: {e}"))?;
    session.reset(gpu);
    session.restore_into_loaded(m, gpu)?;
    Ok(())
}

// ── Sequence-state arena dispatch ──────────────────────────────────────────
// The `sequence_state_arena_*` functions below are thin arch-agnostic wrappers:
// each selects the backend for the loaded arch and forwards to the matching
// `qwen35_*` session op (or the generic arena), so the request loop can manage
// session state without branching on `arch_id` at every call site.

/// Error unless the given arena backend supports `op` on this build — the guard
/// every `sequence_state_arena_*` wrapper calls before dispatching.
pub fn ensure_sequence_state_arena_backend_supported(
    arena_backend: SequenceStateArenaBackend,
    m: &LoadedModel,
    op: &str,
) -> Result<(), String> {
    arena_backend.require_supported(m.arch_id, m.pp, op)
}

pub fn sequence_state_arena_resident_session_count(
    arena_backend: SequenceStateArenaBackend,
    m: &LoadedModel,
) -> usize {
    match arena_backend {
        SequenceStateArenaBackend::Qwen35Wrapped => qwen35_request_session_count(m),
        SequenceStateArenaBackend::BackendOwned => {
            #[cfg(feature = "arch-lfm2moe")]
            if m.arch_id == ARCH_ID_LFM2_MOE {
                let count = lfm2_request_session_count(m);
                if count > 0 {
                    return count;
                }
            }
            usize::from(!sequence_state_arena_page_descriptors(arena_backend, m).is_empty())
        }
        SequenceStateArenaBackend::Unsupported => 0,
    }
}

pub fn sequence_state_arena_page_descriptors(
    arena_backend: SequenceStateArenaBackend,
    m: &LoadedModel,
) -> Vec<SequenceStatePageDescriptor> {
    match arena_backend {
        SequenceStateArenaBackend::Qwen35Wrapped => qwen35_state_page_descriptors(m),
        SequenceStateArenaBackend::BackendOwned => {
            #[cfg(feature = "arch-lfm2moe")]
            if m.arch_id == ARCH_ID_LFM2_MOE && lfm2_request_session_count(m) > 0 {
                return lfm2_session_page_descriptors(m);
            }
            backend_owned_state_page_descriptors(m)
        }
        SequenceStateArenaBackend::Unsupported => Vec::new(),
    }
}

pub fn sequence_state_arena_is_session_resident(
    arena_backend: SequenceStateArenaBackend,
    m: &LoadedModel,
    session_id: &str,
) -> bool {
    match arena_backend {
        SequenceStateArenaBackend::Qwen35Wrapped => qwen35_session_resident(m, session_id),
        SequenceStateArenaBackend::BackendOwned => {
            #[cfg(feature = "arch-lfm2moe")]
            if m.arch_id == ARCH_ID_LFM2_MOE && lfm2_request_session_count(m) > 0 {
                return lfm2_session_resident(m, session_id);
            }
            session_id == backend_owned_session_id(m)
        }
        SequenceStateArenaBackend::Unsupported => false,
    }
}

pub fn sequence_state_arena_release_sessions(
    arena_backend: SequenceStateArenaBackend,
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    session_ids: &[String],
) -> Result<usize, String> {
    #[cfg(feature = "arch-lfm2moe")]
    if arena_backend == SequenceStateArenaBackend::BackendOwned && m.arch_id == ARCH_ID_LFM2_MOE {
        return lfm2_release_sessions(m, gpu, session_ids);
    }
    ensure_sequence_state_arena_backend_supported(arena_backend, m, "release_sessions")?;
    match arena_backend {
        SequenceStateArenaBackend::Qwen35Wrapped => qwen35_release_sessions(m, gpu, session_ids),
        SequenceStateArenaBackend::BackendOwned => {
            unreachable!("backend-owned arena rejected above")
        }
        SequenceStateArenaBackend::Unsupported => unreachable!("unsupported arena rejected above"),
    }
}

pub fn sequence_state_arena_activate_session(
    arena_backend: SequenceStateArenaBackend,
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    session_id: &str,
) -> Result<bool, String> {
    ensure_sequence_state_arena_backend_supported(arena_backend, m, "activate_session")?;
    match arena_backend {
        SequenceStateArenaBackend::Qwen35Wrapped => qwen35_activate_session(m, gpu, session_id),
        SequenceStateArenaBackend::BackendOwned => {
            unreachable!("backend-owned arena rejected above")
        }
        SequenceStateArenaBackend::Unsupported => unreachable!("unsupported arena rejected above"),
    }
}

pub fn sequence_state_arena_reset_active_session(
    arena_backend: SequenceStateArenaBackend,
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
) -> Result<(), String> {
    ensure_sequence_state_arena_backend_supported(arena_backend, m, "reset_active_session")?;
    match arena_backend {
        SequenceStateArenaBackend::Qwen35Wrapped => qwen35_reset_active_session(m, gpu),
        SequenceStateArenaBackend::BackendOwned => {
            unreachable!("backend-owned arena rejected above")
        }
        SequenceStateArenaBackend::Unsupported => unreachable!("unsupported arena rejected above"),
    }
}

pub fn sequence_state_arena_active_logical_position(
    arena_backend: SequenceStateArenaBackend,
    m: &LoadedModel,
) -> Result<usize, String> {
    ensure_sequence_state_arena_backend_supported(arena_backend, m, "active_logical_position")?;
    match arena_backend {
        SequenceStateArenaBackend::Qwen35Wrapped => qwen35_active_logical_position(m),
        SequenceStateArenaBackend::BackendOwned => {
            unreachable!("backend-owned arena rejected above")
        }
        SequenceStateArenaBackend::Unsupported => unreachable!("unsupported arena rejected above"),
    }
}

pub fn sequence_state_arena_fork_session_state(
    arena_backend: SequenceStateArenaBackend,
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    request: SequenceStateForkRequest<'_>,
) -> Result<(), String> {
    #[cfg(feature = "arch-lfm2moe")]
    if arena_backend == SequenceStateArenaBackend::BackendOwned && m.arch_id == ARCH_ID_LFM2_MOE {
        return lfm2_fork_session_state(m, gpu, request);
    }
    ensure_sequence_state_arena_backend_supported(arena_backend, m, "fork_session_state")?;
    match arena_backend {
        SequenceStateArenaBackend::Qwen35Wrapped => qwen35_fork_session_state(m, gpu, request),
        SequenceStateArenaBackend::BackendOwned => {
            unreachable!("backend-owned arena rejected above")
        }
        SequenceStateArenaBackend::Unsupported => unreachable!("unsupported arena rejected above"),
    }
}

pub fn sequence_state_arena_checkpoint_session_state(
    arena_backend: SequenceStateArenaBackend,
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    request: SequenceStateCheckpointRequest<'_>,
) -> Result<(), String> {
    #[cfg(feature = "arch-lfm2moe")]
    if arena_backend == SequenceStateArenaBackend::BackendOwned && m.arch_id == ARCH_ID_LFM2_MOE {
        return lfm2_checkpoint_session_state(m, gpu, request);
    }
    ensure_sequence_state_arena_backend_supported(arena_backend, m, "checkpoint_session_state")?;
    match arena_backend {
        SequenceStateArenaBackend::Qwen35Wrapped => {
            qwen35_checkpoint_session_state(m, gpu, request)
        }
        SequenceStateArenaBackend::BackendOwned => {
            unreachable!("backend-owned arena rejected above")
        }
        SequenceStateArenaBackend::Unsupported => unreachable!("unsupported arena rejected above"),
    }
}

/// Restore a session into the active slot, emitting a protocol error event
/// (rather than panicking) if the restore fails.
pub fn qwen35_restore_or_error(
    stdout: &mut std::io::Stdout,
    id: &str,
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    session: Qwen35RequestSessionState,
) {
    if let Err(e) = session.restore_into_loaded(m, gpu) {
        write_error(
            stdout,
            id,
            &format!("failed to restore qwen35 request session: {e}"),
        );
    }
}
