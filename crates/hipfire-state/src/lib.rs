// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared sequence-state handles, descriptors, and reservation helpers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct SessionStateReservation {
    pub worker_id: String,
    pub reserved_bytes: usize,
    pub handle: SequenceStateHandle,
    pub state_page_descriptors: Vec<SequenceStatePageDescriptor>,
    pub expires_at: Option<Instant>,
}

#[derive(Default)]
pub struct GenericSequenceStateArena {
    reservations: HashMap<String, SessionStateReservation>,
    next_generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelWorkerId {
    pub value: String,
}

impl ModelWorkerId {
    pub fn from_runtime_parts(arch_id: u32, pp: usize, kv_mode: Option<&str>) -> Self {
        Self {
            value: format!(
                "worker:arch{}:pp{}:{}",
                arch_id,
                pp,
                kv_mode.unwrap_or("unknown")
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SequenceStateArenaBackend {
    Qwen35Wrapped,
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SequenceStateArenaOperation {
    ReserveSessionState,
    AttachCheckpoint,
    ForkCheckpoint,
    ReleaseState,
    DescribeState,
}

impl SequenceStateArenaOperation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReserveSessionState => "reserve_session_state",
            Self::AttachCheckpoint => "attach_checkpoint",
            Self::ForkCheckpoint => "fork_checkpoint",
            Self::ReleaseState => "release_state",
            Self::DescribeState => "describe_state",
        }
    }
}

impl SequenceStateArenaBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Qwen35Wrapped => "qwen35_wrapped",
            Self::Unsupported => "unsupported",
        }
    }

    pub fn owns_state_pages(self) -> bool {
        match self {
            Self::Qwen35Wrapped => true,
            Self::Unsupported => false,
        }
    }

    pub fn supported_operations(self) -> &'static [SequenceStateArenaOperation] {
        match self {
            Self::Qwen35Wrapped => &[
                SequenceStateArenaOperation::ReserveSessionState,
                SequenceStateArenaOperation::AttachCheckpoint,
                SequenceStateArenaOperation::ForkCheckpoint,
                SequenceStateArenaOperation::ReleaseState,
                SequenceStateArenaOperation::DescribeState,
            ],
            Self::Unsupported => &[],
        }
    }

    pub fn for_worker_parts(arch_id: u32, pp: usize) -> Self {
        if matches!(arch_id, 5 | 6) && pp == 1 {
            Self::Qwen35Wrapped
        } else {
            Self::Unsupported
        }
    }

    pub fn require_supported(self, arch_id: u32, pp: usize, op: &str) -> Result<(), String> {
        match self {
            Self::Qwen35Wrapped => Ok(()),
            Self::Unsupported => Err(format!(
                "{op} requires a supported sequence-state arena (arch_id={arch_id} pp={pp})"
            )),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelWorkerRuntimeView {
    pub worker_id: ModelWorkerId,
    pub max_seq: usize,
    pub physical_cap: usize,
    pub max_resident_workers: usize,
    pub resident_workers: usize,
    pub state_arena_backend: SequenceStateArenaBackend,
    pub resident_sessions: usize,
    pub state_page_descriptors: Vec<SequenceStatePageDescriptor>,
    pub memory: ModelWorkerMemoryView,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SequenceStateHandle {
    pub id: String,
    pub kind: String,
    pub generation: u64,
}

pub fn qwen35_state_handle_kind(session_id: &str) -> &'static str {
    if session_id.starts_with("qwen35-checkpoint:") {
        "qwen35_checkpoint"
    } else {
        "qwen35_session"
    }
}

pub fn qwen35_sequence_state_handle(
    session_id: &str,
    allocation_epoch: u64,
) -> SequenceStateHandle {
    SequenceStateHandle {
        id: session_id.to_string(),
        kind: qwen35_state_handle_kind(session_id).to_string(),
        generation: allocation_epoch,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedSequenceStateHandle {
    pub id: String,
    pub kind: Option<String>,
    pub generation: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct SequenceStatePrefixHash {
    pub algorithm: String,
    pub value: String,
    pub prefix_len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceStateCheckpointRequest<'a> {
    pub source_session_id: &'a str,
    pub dest_session_id: &'a str,
    pub expected_logical_position: usize,
    pub requested_prefix_hash: Option<&'a SequenceStatePrefixHash>,
    pub checkpoint_prefix_hash: Option<&'a SequenceStatePrefixHash>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceStateForkRequest<'a> {
    pub source_session_id: &'a str,
    pub dest_session_id: &'a str,
    pub requested_prefix_hash: Option<&'a SequenceStatePrefixHash>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SequenceStateReservationRequest {
    pub worker_id: String,
    pub reservation_id: Option<String>,
    pub state_kinds: Vec<SequenceStatePageKind>,
    pub physical_cap: usize,
    pub ttl_ms: u64,
    pub budget_bytes: Option<usize>,
}

pub fn validate_checkpoint_source_resident(
    source_session_id: &str,
    resident: bool,
) -> Result<(), String> {
    if !resident {
        return Err(format!(
            "qwen35 checkpoint source session {source_session_id} is not resident"
        ));
    }
    Ok(())
}

pub fn validate_checkpoint_prefix_hash(
    source_session_id: &str,
    stored: Option<&SequenceStatePrefixHash>,
    requested: Option<&SequenceStatePrefixHash>,
) -> Result<(), String> {
    let Some(requested) = requested else {
        return Ok(());
    };
    let stored = stored.ok_or_else(|| {
        format!("qwen35 checkpoint source session {source_session_id} has no prefix hash")
    })?;
    if stored != requested {
        return Err(format!(
            "prefix hash mismatch for checkpoint {source_session_id}: request={} len={} stored={} len={}",
            requested.value,
            requested.prefix_len,
            stored.value,
            stored.prefix_len
        ));
    }
    Ok(())
}

pub fn validate_checkpoint_logical_position(
    source_session_id: &str,
    expected_logical_position: usize,
    resident_logical_position: usize,
) -> Result<(), String> {
    if resident_logical_position != expected_logical_position {
        return Err(format!(
            "qwen35 checkpoint source session {source_session_id} logical_position mismatch: expected={expected_logical_position} resident={resident_logical_position}"
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SequenceStatePageKind {
    Kv,
    DeltaNet,
    Logits,
    BackendPrivate,
}

impl SequenceStatePageKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Kv => "attention_kv",
            Self::DeltaNet => "deltanet_recurrent",
            Self::Logits => "logits",
            Self::BackendPrivate => "backend_private",
        }
    }

    pub fn from_state_kind(kind: &str) -> Option<Self> {
        match kind {
            "attention_kv" => Some(Self::Kv),
            "deltanet_recurrent" => Some(Self::DeltaNet),
            "logits" => Some(Self::Logits),
            "backend_private" | "architecture_specific" | "mamba_ssm" | "mamba_conv" => {
                Some(Self::BackendPrivate)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SequenceStatePageDescriptor {
    pub session_id: String,
    pub handle: SequenceStateHandle,
    pub kind: SequenceStatePageKind,
    pub label: String,
    pub logical_position: usize,
    pub resident_bytes: usize,
    pub allocation_epoch: u64,
    pub owns_pages: bool,
    pub shape: Vec<usize>,
    pub placement: String,
    pub role: String,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ModelWorkerMemoryView {
    pub model_file_bytes: usize,
    pub model_weight_bytes: usize,
    pub runtime_base_bytes: usize,
    pub runtime_session_bytes: usize,
    pub runtime_state_bytes: usize,
    pub total_resident_bytes: usize,
    pub evictable_state_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ModelArtifactMemory {
    pub model_file_bytes: usize,
    pub model_weight_bytes: usize,
}

impl ModelArtifactMemory {
    pub fn worker_memory_view(
        self,
        runtime_base_bytes: usize,
        runtime_session_bytes: usize,
    ) -> ModelWorkerMemoryView {
        let runtime_state_bytes = runtime_base_bytes.saturating_add(runtime_session_bytes);
        ModelWorkerMemoryView {
            model_file_bytes: self.model_file_bytes,
            model_weight_bytes: self.model_weight_bytes,
            runtime_base_bytes,
            runtime_session_bytes,
            runtime_state_bytes,
            total_resident_bytes: self.model_weight_bytes.saturating_add(runtime_state_bytes),
            evictable_state_bytes: runtime_session_bytes,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DescribedSequenceState {
    pub worker_id: String,
    pub handle: SequenceStateHandle,
    pub state_arena_owns_pages: bool,
    pub reserved_bytes: usize,
    pub state_page_descriptors: Vec<SequenceStatePageDescriptor>,
}

pub fn model_worker_runtime_view_json(worker: &ModelWorkerRuntimeView) -> serde_json::Value {
    let descriptor_bytes = worker
        .state_page_descriptors
        .iter()
        .map(|descriptor| descriptor.resident_bytes)
        .sum::<usize>();
    let descriptors: Vec<serde_json::Value> = worker
        .state_page_descriptors
        .iter()
        .map(sequence_state_page_descriptor_json)
        .collect();
    serde_json::json!({
        "id": worker.worker_id.value,
        "max_seq": worker.max_seq,
        "physical_cap": worker.physical_cap,
        "resident_workers": worker.resident_workers,
        "max_resident_workers": worker.max_resident_workers,
        "state_arena_backend": worker.state_arena_backend.as_str(),
        "state_arena_owns_pages": worker.state_arena_backend.owns_state_pages(),
        "state_arena_operations": worker
            .state_arena_backend
            .supported_operations()
            .iter()
            .map(|op| op.as_str())
            .collect::<Vec<_>>(),
        "resident_sessions": worker.resident_sessions,
        "state_page_descriptor_entries": worker.state_page_descriptors.len(),
        "state_page_descriptor_bytes": descriptor_bytes,
        "state_page_descriptors": descriptors,
        "model_file_bytes": worker.memory.model_file_bytes,
        "model_weight_bytes": worker.memory.model_weight_bytes,
        "runtime_base_bytes": worker.memory.runtime_base_bytes,
        "runtime_session_bytes": worker.memory.runtime_session_bytes,
        "runtime_state_bytes": worker.memory.runtime_state_bytes,
        "total_resident_bytes": worker.memory.total_resident_bytes,
        "evictable_state_bytes": worker.memory.evictable_state_bytes,
    })
}

pub fn sequence_state_page_descriptor_json(
    descriptor: &SequenceStatePageDescriptor,
) -> serde_json::Value {
    serde_json::json!({
        "session_id": &descriptor.session_id,
        "handle": {
            "id": &descriptor.handle.id,
            "kind": &descriptor.handle.kind,
            "generation": descriptor.handle.generation,
        },
        "state_kind": descriptor.kind.as_str(),
        "page_kind": descriptor.kind.as_str(),
        "label": &descriptor.label,
        "logical_position": descriptor.logical_position,
        "resident_bytes": descriptor.resident_bytes,
        "allocation_epoch": descriptor.allocation_epoch,
        "owns_pages": descriptor.owns_pages,
        "shape": &descriptor.shape,
        "placement": &descriptor.placement,
        "role": &descriptor.role,
    })
}

pub fn describe_state_done_json(
    id: &str,
    worker_id: &str,
    handle: &SequenceStateHandle,
    state_arena_owns_pages: bool,
    reserved_bytes: usize,
    state_page_descriptors: &[SequenceStatePageDescriptor],
) -> serde_json::Value {
    let state_page_descriptors = state_page_descriptors
        .iter()
        .map(sequence_state_page_descriptor_json)
        .collect::<Vec<_>>();
    serde_json::json!({
        "type": "describe_state_done",
        "id": id,
        "worker_key_id": worker_id,
        "runtime_state_handle": &handle.id,
        "handle": {
            "id": &handle.id,
            "kind": &handle.kind,
            "generation": handle.generation,
        },
        "state_arena_owns_pages": state_arena_owns_pages,
        "reserved_bytes": reserved_bytes,
        "state_page_descriptors": state_page_descriptors,
    })
}

pub fn session_state_reservation_describe_json(
    id: &str,
    reservation: &SessionStateReservation,
) -> serde_json::Value {
    describe_state_done_json(
        id,
        &reservation.worker_id,
        &reservation.handle,
        true,
        reservation.reserved_bytes,
        &reservation.state_page_descriptors,
    )
}

pub fn described_sequence_state_json(
    id: &str,
    described: &DescribedSequenceState,
) -> serde_json::Value {
    describe_state_done_json(
        id,
        &described.worker_id,
        &described.handle,
        described.state_arena_owns_pages,
        described.reserved_bytes,
        &described.state_page_descriptors,
    )
}

pub fn reserve_session_state_done_json(
    id: &str,
    reservation: &SessionStateReservation,
    current_session_bytes: usize,
    outstanding_reserved_bytes: usize,
    projected_reserved_bytes: usize,
    budget_bytes: usize,
) -> serde_json::Value {
    let state_page_descriptors = reservation
        .state_page_descriptors
        .iter()
        .map(sequence_state_page_descriptor_json)
        .collect::<Vec<_>>();
    serde_json::json!({
        "type": "reserve_session_state_done",
        "id": id,
        "worker_key_id": &reservation.worker_id,
        "reservation_id": &reservation.handle.id,
        "runtime_state_handle": &reservation.handle.id,
        "handle": {
            "id": &reservation.handle.id,
            "kind": &reservation.handle.kind,
            "generation": reservation.handle.generation,
        },
        "state_arena_owns_pages": true,
        "state_page_descriptors": state_page_descriptors,
        "reserved_bytes": reservation.reserved_bytes,
        "current_session_bytes": current_session_bytes,
        "outstanding_reserved_bytes": outstanding_reserved_bytes,
        "projected_reserved_bytes": projected_reserved_bytes,
        "budget_bytes": budget_bytes,
    })
}

pub fn reserve_session_state_rejected_json(
    id: &str,
    worker_id: &str,
    reserved_bytes: usize,
    current_session_bytes: usize,
    outstanding_reserved_bytes: usize,
    projected_reserved_bytes: usize,
    budget_bytes: usize,
) -> serde_json::Value {
    serde_json::json!({
        "type": "reserve_session_state_rejected",
        "id": id,
        "worker_key_id": worker_id,
        "reason": "memory_pressure",
        "reserved_bytes": reserved_bytes,
        "current_session_bytes": current_session_bytes,
        "outstanding_reserved_bytes": outstanding_reserved_bytes,
        "projected_reserved_bytes": projected_reserved_bytes,
        "budget_bytes": budget_bytes,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReleaseStateResponseKind {
    ReleaseState,
    ReleaseSessionStateReservation,
}

impl ReleaseStateResponseKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReleaseState => "release_state_done",
            Self::ReleaseSessionStateReservation => "release_session_state_reservation_done",
        }
    }
}

pub fn release_state_done_json(
    kind: ReleaseStateResponseKind,
    id: &str,
    generic_released: usize,
    generic_released_bytes: usize,
    loaded_released: usize,
    loaded_released_bytes: usize,
) -> serde_json::Value {
    let released = generic_released + loaded_released;
    let released_bytes = generic_released_bytes.saturating_add(loaded_released_bytes);
    serde_json::json!({
        "type": kind.as_str(),
        "id": id,
        "released": released,
        "released_bytes": released_bytes,
        "generic_released": generic_released,
        "loaded_released": loaded_released,
    })
}

pub fn release_sessions_done_json(
    id: &str,
    requested: usize,
    released: usize,
    resident_sessions: usize,
    model_worker: Option<&ModelWorkerRuntimeView>,
) -> serde_json::Value {
    let mut done = serde_json::json!({
        "type": "release_sessions_done",
        "id": id,
        "requested": requested,
        "released": released,
        "resident_sessions": resident_sessions,
    });
    if let Some(worker) = model_worker {
        done["model_worker"] = model_worker_runtime_view_json(worker);
    }
    done
}

pub fn unload_worker_done_json(
    id: &str,
    worker_id: &str,
    unloaded: bool,
    resident_workers: usize,
) -> serde_json::Value {
    serde_json::json!({
        "type": "unload_worker_done",
        "id": id,
        "worker_key_id": worker_id,
        "unloaded": unloaded,
        "resident_workers": resident_workers,
    })
}

impl GenericSequenceStateArena {
    pub fn new() -> Self {
        Self {
            reservations: HashMap::new(),
            next_generation: 1,
        }
    }

    pub fn purge_expired(&mut self) {
        let now = Instant::now();
        self.reservations.retain(|_, reservation| {
            reservation
                .expires_at
                .map(|expires_at| expires_at > now)
                .unwrap_or(true)
        });
    }

    pub fn release_worker(&mut self, worker_id: &str) {
        self.reservations
            .retain(|_, reservation| reservation.worker_id != worker_id);
    }

    pub fn clear(&mut self) {
        self.reservations.clear();
    }

    pub fn outstanding_bytes_for_worker(&self, worker_id: &str) -> usize {
        self.reservations
            .values()
            .filter(|reservation| reservation.worker_id == worker_id)
            .map(|reservation| reservation.reserved_bytes)
            .sum()
    }

    pub fn reserve(
        &mut self,
        worker_id: &str,
        reservation_id: String,
        state_kinds: &[SequenceStatePageKind],
        physical_cap: usize,
        reserved_bytes: usize,
        ttl_ms: u64,
    ) -> SessionStateReservation {
        let generation = self.next_generation;
        self.next_generation = self.next_generation.saturating_add(1).max(1);
        let handle = SequenceStateHandle {
            id: reservation_id.clone(),
            kind: "generic_reserved_state".to_string(),
            generation,
        };
        let state_page_descriptors = generic_state_reservation_descriptors(
            worker_id,
            &handle,
            state_kinds,
            physical_cap,
            reserved_bytes,
        );
        let reservation = SessionStateReservation {
            worker_id: worker_id.to_string(),
            reserved_bytes,
            handle,
            state_page_descriptors,
            expires_at: if ttl_ms == 0 {
                None
            } else {
                Some(Instant::now() + Duration::from_millis(ttl_ms))
            },
        };
        self.reservations
            .insert(reservation_id, reservation.clone());
        reservation
    }

    pub fn describe(
        &self,
        handle_id: &str,
        generation: Option<u64>,
    ) -> Option<&SessionStateReservation> {
        let reservation = self.reservations.get(handle_id)?;
        if generation
            .map(|generation| generation == reservation.handle.generation)
            .unwrap_or(true)
        {
            Some(reservation)
        } else {
            None
        }
    }

    pub fn release<I>(&mut self, handles: I) -> (usize, usize)
    where
        I: IntoIterator<Item = (String, Option<u64>)>,
    {
        let mut released = 0usize;
        let mut released_bytes = 0usize;
        for (handle_id, generation) in handles {
            let matches_generation = self
                .reservations
                .get(&handle_id)
                .map(|reservation| {
                    generation
                        .map(|generation| generation == reservation.handle.generation)
                        .unwrap_or(true)
                })
                .unwrap_or(false);
            if matches_generation {
                if let Some(reservation) = self.reservations.remove(&handle_id) {
                    released += 1;
                    released_bytes = released_bytes.saturating_add(reservation.reserved_bytes);
                }
            }
        }
        (released, released_bytes)
    }
}

pub fn sequence_state_handle_id(value: &serde_json::Value) -> Option<&str> {
    sequence_state_handle_parts(value).map(|(id, _)| id)
}

pub fn sequence_state_handle_parts(value: &serde_json::Value) -> Option<(&str, Option<u64>)> {
    if let Some(id) = value.as_str().filter(|s| !s.is_empty()) {
        return Some((id, None));
    }
    let id = value.get("id").and_then(|v| v.as_str())?.trim();
    if id.is_empty() {
        return None;
    }
    let generation = value
        .get("generation")
        .or_else(|| value.get("allocation_epoch"))
        .and_then(|v| v.as_u64());
    Some((id, generation))
}

pub fn parse_sequence_state_handle(value: &serde_json::Value) -> Option<ParsedSequenceStateHandle> {
    let (id, generation) = sequence_state_handle_parts(value)?;
    let kind = value
        .get("kind")
        .and_then(|v| v.as_str())
        .filter(|kind| !kind.is_empty())
        .map(|kind| kind.to_string());
    Some(ParsedSequenceStateHandle {
        id: id.to_string(),
        kind,
        generation,
    })
}

pub fn parse_sequence_state_handle_list(msg: &serde_json::Value) -> Vec<ParsedSequenceStateHandle> {
    msg.get("reservations")
        .or_else(|| msg.get("handles"))
        .and_then(|v| v.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(parse_sequence_state_handle)
                .collect()
        })
        .or_else(|| {
            msg.get("reservation_id")
                .and_then(parse_sequence_state_handle)
                .map(|handle| vec![handle])
        })
        .or_else(|| {
            msg.get("runtime_state_handle")
                .and_then(parse_sequence_state_handle)
                .map(|handle| vec![handle])
        })
        .or_else(|| {
            msg.get("handle")
                .and_then(parse_sequence_state_handle)
                .map(|handle| vec![handle])
        })
        .unwrap_or_default()
}

pub fn parse_reserve_session_state_kinds(
    msg: &serde_json::Value,
) -> Result<Vec<SequenceStatePageKind>, String> {
    let Some(value) = msg.get("state_kinds") else {
        return Ok(vec![
            SequenceStatePageKind::Kv,
            SequenceStatePageKind::DeltaNet,
        ]);
    };
    let values = value
        .as_array()
        .ok_or_else(|| "reserve_session_state.state_kinds must be an array".to_string())?;
    if values.is_empty() {
        return Err("reserve_session_state.state_kinds must not be empty".to_string());
    }
    let mut kinds = Vec::with_capacity(values.len());
    for value in values {
        let raw = value
            .as_str()
            .ok_or_else(|| "reserve_session_state.state_kinds must be strings".to_string())?;
        let kind = SequenceStatePageKind::from_state_kind(raw).ok_or_else(|| {
            format!("reserve_session_state.state_kinds contains unsupported kind {raw}")
        })?;
        if !kinds.contains(&kind) {
            kinds.push(kind);
        }
    }
    Ok(kinds)
}

pub fn parse_reserve_session_state_request(
    msg: &serde_json::Value,
    worker_id: &str,
) -> Result<SequenceStateReservationRequest, String> {
    let physical_cap = msg
        .get("physical_cap")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(0);
    if physical_cap == 0 {
        return Err("reserve_session_state.physical_cap must be > 0".to_string());
    }
    let state_kinds = parse_reserve_session_state_kinds(msg)?;
    let reservation_id = msg
        .get("reservation_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    let ttl_ms = msg.get("ttl_ms").and_then(|v| v.as_u64()).unwrap_or(30_000);
    let budget_bytes = msg
        .get("budget_bytes")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    Ok(SequenceStateReservationRequest {
        worker_id: worker_id.to_string(),
        reservation_id,
        state_kinds,
        physical_cap,
        ttl_ms,
        budget_bytes,
    })
}

pub fn generic_state_reservation_descriptors(
    worker_id: &str,
    handle: &SequenceStateHandle,
    kinds: &[SequenceStatePageKind],
    physical_cap: usize,
    reserved_bytes: usize,
) -> Vec<SequenceStatePageDescriptor> {
    let count = kinds.len().max(1);
    let base_bytes = reserved_bytes / count;
    let remainder = reserved_bytes % count;
    kinds
        .iter()
        .enumerate()
        .map(|(idx, &kind)| {
            let resident_bytes = base_bytes + usize::from(idx < remainder);
            let label = match kind {
                SequenceStatePageKind::Kv => "generic.attention_kv",
                SequenceStatePageKind::DeltaNet => "generic.deltanet_recurrent",
                SequenceStatePageKind::Logits => "generic.logits",
                SequenceStatePageKind::BackendPrivate => "generic.backend_private",
            };
            let shape = match kind {
                SequenceStatePageKind::Kv | SequenceStatePageKind::DeltaNet => vec![physical_cap],
                SequenceStatePageKind::Logits | SequenceStatePageKind::BackendPrivate => vec![1],
            };
            SequenceStatePageDescriptor {
                session_id: handle.id.clone(),
                handle: handle.clone(),
                kind,
                label: label.to_string(),
                logical_position: 0,
                resident_bytes,
                allocation_epoch: handle.generation,
                owns_pages: true,
                shape,
                placement: format!("host:reserved:{worker_id}"),
                role: "reserved".to_string(),
            }
        })
        .collect()
}

pub fn sequence_state_descriptor_matches_handle(
    descriptor: &SequenceStatePageDescriptor,
    handle: &ParsedSequenceStateHandle,
) -> bool {
    if descriptor.handle.id != handle.id {
        return false;
    }
    if let Some(kind) = handle.kind.as_deref() {
        if descriptor.handle.kind != kind {
            return false;
        }
    }
    handle
        .generation
        .map(|generation| descriptor.handle.generation == generation)
        .unwrap_or(true)
}

pub fn describe_sequence_state_descriptors(
    descriptors: Vec<SequenceStatePageDescriptor>,
    handle: &ParsedSequenceStateHandle,
) -> Option<Vec<SequenceStatePageDescriptor>> {
    let matched = descriptors
        .into_iter()
        .filter(|descriptor| sequence_state_descriptor_matches_handle(descriptor, handle))
        .collect::<Vec<_>>();
    if matched.is_empty() {
        None
    } else {
        Some(matched)
    }
}

pub fn parsed_handle_may_target_generic(handle: &ParsedSequenceStateHandle) -> bool {
    handle
        .kind
        .as_deref()
        .map(|kind| kind == "generic_reserved_state")
        .unwrap_or(true)
}

pub fn parsed_handle_may_target_loaded_state(handle: &ParsedSequenceStateHandle) -> bool {
    handle
        .kind
        .as_deref()
        .map(|kind| matches!(kind, "qwen35_session" | "qwen35_checkpoint"))
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_state_kinds_defaults_deduplicates_and_aliases() {
        let defaults = parse_reserve_session_state_kinds(&serde_json::json!({})).unwrap();
        assert_eq!(
            defaults,
            vec![SequenceStatePageKind::Kv, SequenceStatePageKind::DeltaNet]
        );

        let parsed = parse_reserve_session_state_kinds(&serde_json::json!({
            "state_kinds": [
                "attention_kv",
                "deltanet_recurrent",
                "mamba_ssm",
                "mamba_conv",
                "attention_kv"
            ]
        }))
        .unwrap();
        assert_eq!(
            parsed,
            vec![
                SequenceStatePageKind::Kv,
                SequenceStatePageKind::DeltaNet,
                SequenceStatePageKind::BackendPrivate
            ]
        );

        let err = parse_reserve_session_state_kinds(&serde_json::json!({
            "state_kinds": ["bogus"]
        }))
        .unwrap_err();
        assert!(err.contains("unsupported kind bogus"));
    }

    #[test]
    fn parse_reserve_session_state_request_preserves_daemon_defaults() {
        let request = parse_reserve_session_state_request(
            &serde_json::json!({
                "type": "reserve_session_state",
                "physical_cap": 4096
            }),
            "worker-a",
        )
        .unwrap();
        assert_eq!(request.worker_id, "worker-a");
        assert_eq!(request.reservation_id, None);
        assert_eq!(
            request.state_kinds,
            vec![SequenceStatePageKind::Kv, SequenceStatePageKind::DeltaNet]
        );
        assert_eq!(request.physical_cap, 4096);
        assert_eq!(request.ttl_ms, 30_000);
        assert_eq!(request.budget_bytes, None);
    }

    #[test]
    fn parse_reserve_session_state_request_preserves_optional_fields() {
        let request = parse_reserve_session_state_request(
            &serde_json::json!({
                "type": "reserve_session_state",
                "reservation_id": "reserve-a",
                "physical_cap": 8192,
                "ttl_ms": 0,
                "budget_bytes": 16384,
                "state_kinds": ["attention_kv", "mamba_ssm", "attention_kv"]
            }),
            "worker-b",
        )
        .unwrap();
        assert_eq!(request.worker_id, "worker-b");
        assert_eq!(request.reservation_id.as_deref(), Some("reserve-a"));
        assert_eq!(
            request.state_kinds,
            vec![
                SequenceStatePageKind::Kv,
                SequenceStatePageKind::BackendPrivate
            ]
        );
        assert_eq!(request.physical_cap, 8192);
        assert_eq!(request.ttl_ms, 0);
        assert_eq!(request.budget_bytes, Some(16384));
    }

    #[test]
    fn parse_reserve_session_state_request_rejects_missing_physical_cap_first() {
        let err = parse_reserve_session_state_request(
            &serde_json::json!({
                "type": "reserve_session_state",
                "state_kinds": ["bogus"]
            }),
            "worker-a",
        )
        .unwrap_err();
        assert_eq!(err, "reserve_session_state.physical_cap must be > 0");
    }

    #[test]
    fn model_worker_runtime_view_json_reports_state_page_descriptors() {
        let handle = SequenceStateHandle {
            id: "session-a".to_string(),
            kind: "qwen35_session".to_string(),
            generation: 7,
        };
        let worker = ModelWorkerRuntimeView {
            worker_id: ModelWorkerId {
                value: "worker:arch6:pp1:q8".to_string(),
            },
            max_seq: 4096,
            physical_cap: 2048,
            max_resident_workers: 1,
            resident_workers: 1,
            state_arena_backend: SequenceStateArenaBackend::Qwen35Wrapped,
            resident_sessions: 1,
            state_page_descriptors: vec![SequenceStatePageDescriptor {
                session_id: "session-a".to_string(),
                handle,
                kind: SequenceStatePageKind::Kv,
                label: "qwen35.kv_cache".to_string(),
                logical_position: 12,
                resident_bytes: 1024,
                allocation_epoch: 7,
                owns_pages: true,
                shape: vec![2, 2048, 8, 128],
                placement: "hip:arch6:device0".to_string(),
                role: "resident".to_string(),
            }],
            memory: ModelWorkerMemoryView {
                model_file_bytes: 11,
                model_weight_bytes: 22,
                runtime_base_bytes: 33,
                runtime_session_bytes: 44,
                runtime_state_bytes: 77,
                total_resident_bytes: 99,
                evictable_state_bytes: 44,
            },
        };

        let json = model_worker_runtime_view_json(&worker);
        assert_eq!(json["state_arena_backend"], "qwen35_wrapped");
        assert_eq!(
            json["state_arena_operations"],
            serde_json::json!([
                "reserve_session_state",
                "attach_checkpoint",
                "fork_checkpoint",
                "release_state",
                "describe_state"
            ])
        );
        assert_eq!(json["state_page_descriptor_entries"], 1);
        assert_eq!(json["state_page_descriptor_bytes"], 1024);
        assert_eq!(
            json["state_page_descriptors"][0]["state_kind"],
            "attention_kv"
        );
        assert_eq!(json["state_page_descriptors"][0]["handle"]["generation"], 7);
    }

    #[test]
    fn describe_state_done_json_preserves_daemon_wire_shape() {
        let handle = qwen35_sequence_state_handle("qwen35-checkpoint:batch:req:16", 41);
        let descriptor = SequenceStatePageDescriptor {
            session_id: handle.id.clone(),
            handle: handle.clone(),
            kind: SequenceStatePageKind::Kv,
            label: "qwen35.kv_cache".to_string(),
            logical_position: 16,
            resident_bytes: 1024,
            allocation_epoch: 41,
            owns_pages: true,
            shape: vec![2, 16, 4, 32],
            placement: "hip:arch6:device0".to_string(),
            role: "resident".to_string(),
        };
        let described = DescribedSequenceState {
            worker_id: "worker:arch6:pp1:q8".to_string(),
            handle,
            state_arena_owns_pages: true,
            reserved_bytes: 1024,
            state_page_descriptors: vec![descriptor],
        };

        let json = described_sequence_state_json("describe-1", &described);
        assert_eq!(json["type"], "describe_state_done");
        assert_eq!(json["id"], "describe-1");
        assert_eq!(json["worker_key_id"], "worker:arch6:pp1:q8");
        assert_eq!(
            json["runtime_state_handle"],
            "qwen35-checkpoint:batch:req:16"
        );
        assert_eq!(json["handle"]["kind"], "qwen35_checkpoint");
        assert_eq!(json["handle"]["generation"], 41);
        assert_eq!(json["state_arena_owns_pages"], true);
        assert_eq!(json["reserved_bytes"], 1024);
        assert_eq!(
            json["state_page_descriptors"][0]["state_kind"],
            "attention_kv"
        );
    }

    #[test]
    fn reservation_describe_json_preserves_generic_owned_shape() {
        let mut arena = GenericSequenceStateArena::new();
        let reservation = arena.reserve(
            "worker-a",
            "reserve-a".to_string(),
            &[SequenceStatePageKind::Kv],
            128,
            4096,
            0,
        );

        let json = session_state_reservation_describe_json("describe-2", &reservation);
        assert_eq!(json["type"], "describe_state_done");
        assert_eq!(json["id"], "describe-2");
        assert_eq!(json["worker_key_id"], "worker-a");
        assert_eq!(json["runtime_state_handle"], "reserve-a");
        assert_eq!(json["handle"]["kind"], "generic_reserved_state");
        assert_eq!(json["state_arena_owns_pages"], true);
        assert_eq!(json["reserved_bytes"], 4096);
        assert_eq!(json["state_page_descriptors"][0]["owns_pages"], true);
    }

    #[test]
    fn reserve_session_state_done_json_preserves_daemon_wire_shape() {
        let mut arena = GenericSequenceStateArena::new();
        let reservation = arena.reserve(
            "worker-a",
            "reserve-a".to_string(),
            &[SequenceStatePageKind::Kv, SequenceStatePageKind::DeltaNet],
            256,
            8192,
            0,
        );

        let json =
            reserve_session_state_done_json("reserve-1", &reservation, 1024, 2048, 11264, 16384);
        assert_eq!(json["type"], "reserve_session_state_done");
        assert_eq!(json["id"], "reserve-1");
        assert_eq!(json["worker_key_id"], "worker-a");
        assert_eq!(json["reservation_id"], "reserve-a");
        assert_eq!(json["runtime_state_handle"], "reserve-a");
        assert_eq!(json["handle"]["kind"], "generic_reserved_state");
        assert_eq!(json["handle"]["generation"], 1);
        assert_eq!(json["state_arena_owns_pages"], true);
        assert_eq!(json["state_page_descriptors"].as_array().unwrap().len(), 2);
        assert_eq!(json["reserved_bytes"], 8192);
        assert_eq!(json["current_session_bytes"], 1024);
        assert_eq!(json["outstanding_reserved_bytes"], 2048);
        assert_eq!(json["projected_reserved_bytes"], 11264);
        assert_eq!(json["budget_bytes"], 16384);
    }

    #[test]
    fn reserve_session_state_rejected_json_preserves_daemon_wire_shape() {
        let json = reserve_session_state_rejected_json(
            "reserve-2",
            "worker-a",
            8192,
            1024,
            2048,
            11264,
            4096,
        );
        assert_eq!(json["type"], "reserve_session_state_rejected");
        assert_eq!(json["id"], "reserve-2");
        assert_eq!(json["worker_key_id"], "worker-a");
        assert_eq!(json["reason"], "memory_pressure");
        assert_eq!(json["reserved_bytes"], 8192);
        assert_eq!(json["current_session_bytes"], 1024);
        assert_eq!(json["outstanding_reserved_bytes"], 2048);
        assert_eq!(json["projected_reserved_bytes"], 11264);
        assert_eq!(json["budget_bytes"], 4096);
    }

    #[test]
    fn release_state_done_json_preserves_daemon_wire_shape() {
        let json = release_state_done_json(
            ReleaseStateResponseKind::ReleaseState,
            "release-1",
            2,
            4096,
            1,
            2048,
        );
        assert_eq!(json["type"], "release_state_done");
        assert_eq!(json["id"], "release-1");
        assert_eq!(json["released"], 3);
        assert_eq!(json["released_bytes"], 6144);
        assert_eq!(json["generic_released"], 2);
        assert_eq!(json["loaded_released"], 1);
        assert!(json.get("generic_released_bytes").is_none());
        assert!(json.get("loaded_released_bytes").is_none());
    }

    #[test]
    fn release_session_state_reservation_done_json_uses_reservation_response_type() {
        let json = release_state_done_json(
            ReleaseStateResponseKind::ReleaseSessionStateReservation,
            "release-2",
            0,
            usize::MAX,
            1,
            8,
        );
        assert_eq!(json["type"], "release_session_state_reservation_done");
        assert_eq!(json["released"], 1);
        assert_eq!(json["released_bytes"], usize::MAX);
    }

    #[test]
    fn release_sessions_done_json_preserves_dummy_wire_shape() {
        let json = release_sessions_done_json("release-sessions-1", 3, 2, 1, None);
        assert_eq!(json["type"], "release_sessions_done");
        assert_eq!(json["id"], "release-sessions-1");
        assert_eq!(json["requested"], 3);
        assert_eq!(json["released"], 2);
        assert_eq!(json["resident_sessions"], 1);
        assert!(json.get("model_worker").is_none());
    }

    #[test]
    fn release_sessions_done_json_includes_model_worker_when_present() {
        let worker = ModelWorkerRuntimeView {
            worker_id: ModelWorkerId::from_runtime_parts(6, 1, Some("q8")),
            max_seq: 256,
            physical_cap: 128,
            resident_workers: 1,
            max_resident_workers: 2,
            state_arena_backend: SequenceStateArenaBackend::Qwen35Wrapped,
            resident_sessions: 4,
            state_page_descriptors: Vec::new(),
            memory: ModelWorkerMemoryView {
                model_file_bytes: 10,
                model_weight_bytes: 20,
                runtime_base_bytes: 30,
                runtime_session_bytes: 40,
                runtime_state_bytes: 70,
                total_resident_bytes: 90,
                evictable_state_bytes: 40,
            },
        };

        let json = release_sessions_done_json("release-sessions-2", 2, 1, 4, Some(&worker));
        assert_eq!(json["type"], "release_sessions_done");
        assert_eq!(json["requested"], 2);
        assert_eq!(json["released"], 1);
        assert_eq!(json["resident_sessions"], 4);
        assert_eq!(json["model_worker"]["id"], "worker:arch6:pp1:q8");
        assert_eq!(
            json["model_worker"]["state_arena_backend"],
            "qwen35_wrapped"
        );
        assert_eq!(json["model_worker"]["runtime_state_bytes"], 70);
    }

    #[test]
    fn unload_worker_done_json_preserves_daemon_wire_shape() {
        let json = unload_worker_done_json("unload-1", "worker:arch6:pp1:q8", true, 2);
        assert_eq!(json["type"], "unload_worker_done");
        assert_eq!(json["id"], "unload-1");
        assert_eq!(json["worker_key_id"], "worker:arch6:pp1:q8");
        assert_eq!(json["unloaded"], true);
        assert_eq!(json["resident_workers"], 2);
    }

    #[test]
    fn generic_reservation_arena_releases_by_generation() {
        let mut arena = GenericSequenceStateArena::new();
        let reservation = arena.reserve(
            "worker-a",
            "reserve-a".to_string(),
            &[SequenceStatePageKind::Kv],
            128,
            4096,
            0,
        );
        assert_eq!(reservation.handle.generation, 1);
        assert_eq!(arena.outstanding_bytes_for_worker("worker-a"), 4096);
        assert!(arena.describe("reserve-a", Some(1)).is_some());
        assert!(arena.describe("reserve-a", Some(99)).is_none());
        assert_eq!(
            arena.release(vec![("reserve-a".to_string(), Some(99))]),
            (0, 0)
        );
        assert_eq!(
            arena.release(vec![("reserve-a".to_string(), Some(1))]),
            (1, 4096)
        );
        assert_eq!(arena.outstanding_bytes_for_worker("worker-a"), 0);
    }

    #[test]
    fn handle_parsing_accepts_string_or_object() {
        assert_eq!(
            sequence_state_handle_id(&serde_json::json!("reserve-a")),
            Some("reserve-a")
        );
        assert_eq!(
            sequence_state_handle_id(&serde_json::json!({
                "id": "reserve-b",
                "generation": 2
            })),
            Some("reserve-b")
        );
        assert_eq!(
            sequence_state_handle_parts(&serde_json::json!({
                "id": "reserve-b",
                "allocation_epoch": 3
            })),
            Some(("reserve-b", Some(3)))
        );
        assert_eq!(sequence_state_handle_id(&serde_json::json!("")), None);
        assert_eq!(
            sequence_state_handle_id(&serde_json::json!({"kind": "missing_id"})),
            None
        );
    }

    #[test]
    fn qwen35_sequence_state_handles_classify_sessions_and_checkpoints() {
        let session = qwen35_sequence_state_handle("request-a", 7);
        assert_eq!(session.id, "request-a");
        assert_eq!(session.kind, "qwen35_session");
        assert_eq!(session.generation, 7);

        let checkpoint = qwen35_sequence_state_handle("qwen35-checkpoint:batch:req:16", 41);
        assert_eq!(checkpoint.id, "qwen35-checkpoint:batch:req:16");
        assert_eq!(checkpoint.kind, "qwen35_checkpoint");
        assert_eq!(checkpoint.generation, 41);
    }

    #[test]
    fn handle_target_helpers_respect_known_kinds() {
        let generic = parse_sequence_state_handle(&serde_json::json!({
            "id": "reserve-a",
            "kind": "generic_reserved_state",
            "generation": 1
        }))
        .unwrap();
        assert!(parsed_handle_may_target_generic(&generic));
        assert!(!parsed_handle_may_target_loaded_state(&generic));

        let qwen35 = parse_sequence_state_handle(&serde_json::json!({
            "id": "session-a",
            "kind": "qwen35_session"
        }))
        .unwrap();
        assert!(!parsed_handle_may_target_generic(&qwen35));
        assert!(parsed_handle_may_target_loaded_state(&qwen35));
    }

    #[test]
    fn worker_id_and_arena_policy_follow_runtime_shape() {
        assert_eq!(
            ModelWorkerId::from_runtime_parts(6, 1, Some("q8")).value,
            "worker:arch6:pp1:q8"
        );
        assert_eq!(
            ModelWorkerId::from_runtime_parts(5, 2, None).value,
            "worker:arch5:pp2:unknown"
        );
        assert_eq!(
            SequenceStateArenaBackend::for_worker_parts(5, 1),
            SequenceStateArenaBackend::Qwen35Wrapped
        );
        assert_eq!(
            SequenceStateArenaBackend::for_worker_parts(6, 1),
            SequenceStateArenaBackend::Qwen35Wrapped
        );
        assert_eq!(
            SequenceStateArenaBackend::for_worker_parts(5, 2),
            SequenceStateArenaBackend::Unsupported
        );
        assert!(SequenceStateArenaBackend::Qwen35Wrapped
            .require_supported(5, 1, "attach_checkpoint")
            .is_ok());
        let err = SequenceStateArenaBackend::Unsupported
            .require_supported(7, 1, "attach_checkpoint")
            .unwrap_err();
        assert!(err.contains("attach_checkpoint requires a supported sequence-state arena"));
        assert!(err.contains("arch_id=7 pp=1"));
    }

    #[test]
    fn checkpoint_prefix_hash_validation_accepts_matching_or_absent_request() {
        let stored = SequenceStatePrefixHash {
            algorithm: "xxh3_128".to_string(),
            value: "abc".to_string(),
            prefix_len: 12,
        };
        let requested = stored.clone();
        validate_checkpoint_prefix_hash("checkpoint-a", Some(&stored), None).unwrap();
        validate_checkpoint_prefix_hash("checkpoint-a", Some(&stored), Some(&requested)).unwrap();
    }

    #[test]
    fn checkpoint_prefix_hash_validation_reports_missing_stored_hash() {
        let requested = SequenceStatePrefixHash {
            algorithm: "xxh3_128".to_string(),
            value: "abc".to_string(),
            prefix_len: 12,
        };
        let err =
            validate_checkpoint_prefix_hash("checkpoint-a", None, Some(&requested)).unwrap_err();
        assert_eq!(
            err,
            "qwen35 checkpoint source session checkpoint-a has no prefix hash"
        );
    }

    #[test]
    fn checkpoint_prefix_hash_validation_reports_mismatch() {
        let stored = SequenceStatePrefixHash {
            algorithm: "xxh3_128".to_string(),
            value: "stored".to_string(),
            prefix_len: 10,
        };
        let requested = SequenceStatePrefixHash {
            algorithm: "xxh3_128".to_string(),
            value: "requested".to_string(),
            prefix_len: 12,
        };
        let err = validate_checkpoint_prefix_hash("checkpoint-a", Some(&stored), Some(&requested))
            .unwrap_err();
        assert_eq!(
            err,
            "prefix hash mismatch for checkpoint checkpoint-a: request=requested len=12 stored=stored len=10"
        );
    }

    #[test]
    fn checkpoint_logical_position_validation_accepts_match() {
        validate_checkpoint_logical_position("checkpoint-a", 16, 16).unwrap();
    }

    #[test]
    fn checkpoint_logical_position_validation_reports_mismatch() {
        let err = validate_checkpoint_logical_position("checkpoint-a", 16, 12).unwrap_err();
        assert_eq!(
            err,
            "qwen35 checkpoint source session checkpoint-a logical_position mismatch: expected=16 resident=12"
        );
    }

    #[test]
    fn checkpoint_source_resident_validation_accepts_resident_source() {
        validate_checkpoint_source_resident("checkpoint-a", true).unwrap();
    }

    #[test]
    fn checkpoint_source_resident_validation_reports_missing_source() {
        let err = validate_checkpoint_source_resident("checkpoint-a", false).unwrap_err();
        assert_eq!(
            err,
            "qwen35 checkpoint source session checkpoint-a is not resident"
        );
    }

    #[test]
    fn sequence_state_fork_request_preserves_attach_contract() {
        let hash = SequenceStatePrefixHash {
            algorithm: "xxh3_128".to_string(),
            value: "abc".to_string(),
            prefix_len: 12,
        };
        let request = SequenceStateForkRequest {
            source_session_id: "checkpoint-a",
            dest_session_id: "session-b",
            requested_prefix_hash: Some(&hash),
        };
        assert_eq!(request.source_session_id, "checkpoint-a");
        assert_eq!(request.dest_session_id, "session-b");
        assert_eq!(request.requested_prefix_hash.unwrap().prefix_len, 12);
    }
}
