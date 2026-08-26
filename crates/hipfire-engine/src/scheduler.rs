// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-side continuous-batch scheduler — architecture-neutral.
//!
//! Relocated verbatim from `crates/hipfire-daemon/src/main.rs` (wave 3).
//! No behaviour change.

use std::time::Instant;

use crate::terminal::{
    batch_announce_terminal, batch_apply_terminal_control, batch_bind_active, batch_check_abort,
    batch_clear_terminal, batch_mark_ready_with_pending, batch_poll_decision,
    batch_terminal_control, batch_transition_to_queued, AttemptKey, BatchRegistryState,
    ClientTerminalDecision, LaneTicket, CLIENT_TERMINAL_COMMIT_TIMEOUT,
};

// ── Batch sampling controls and cohort key ───────────────────────────────

/// Validated sampling controls for a single request. One active cohort has
/// exactly one key because `sample_product` accepts scalar controls but
/// per-row RNG/history.
#[derive(Debug, Clone)]
pub struct BatchSampling {
    pub temp: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repeat_window: usize,
}

impl BatchSampling {
    pub fn key(&self) -> BatchSamplingKey {
        BatchSamplingKey {
            temp_bits: self.temp.to_bits(),
            top_p_bits: self.top_p.to_bits(),
            top_k: self.top_k,
            min_p_bits: self.min_p.map(|v| v.to_bits()),
            repeat_bits: self.repeat_penalty.to_bits(),
            presence_bits: self.presence_penalty.to_bits(),
            frequency_bits: self.frequency_penalty.to_bits(),
            repeat_window: self.repeat_window,
        }
    }
}

/// Canonical sampling key made from validated values via `f32::to_bits`.
/// Derived Eq/Hash is bit-exact; no epsilon.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchSamplingKey {
    pub temp_bits: u32,
    pub top_p_bits: u32,
    pub top_k: Option<u32>,
    pub min_p_bits: Option<u32>,
    pub repeat_bits: u32,
    pub presence_bits: u32,
    pub frequency_bits: u32,
    pub repeat_window: usize,
}

// ── Typed lane state ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct QwenBatchLane {
    pub key: AttemptKey,
    pub ticket: LaneTicket,
    pub sampling: BatchSampling,
    pub prompt_len: usize,
    pub seq_pos: usize,
    pub next_token: Option<u32>,
    pub rng_state: u64,
    pub conversation_tokens: Vec<u32>,
    pub streamed_tokens: Vec<u32>,
    pub bytes_fed_to_filter: usize,
    pub created_at: Instant,
    /// Host Instant after successful lane prefill + first sample.
    pub prefill_done_at: Option<Instant>,
    /// Host Instant stamped immediately before the first classified token emit.
    pub first_token_at: Option<Instant>,
    /// Peak concurrent Running lanes observed while this lane was active.
    pub max_active_lanes: usize,
}

#[derive(Debug, Clone)]
pub struct BatchPendingTerminal {
    pub key: AttemptKey,
    pub ticket: LaneTicket,
    pub sampling: BatchSampling,
    pub prompt_len: usize,
    pub seq_pos: usize,
    pub pending_done: serde_json::Value,
    pub deadline: Instant,
}

#[derive(Debug)]
pub enum BatchLane {
    Empty { generation: u64 },
    Seeding(QwenBatchLane),
    Running(QwenBatchLane),
    AwaitingClient(BatchPendingTerminal),
}

impl BatchLane {
    pub fn generation(&self) -> u64 {
        match self {
            BatchLane::Empty { generation } => *generation,
            BatchLane::Seeding(l) => l.ticket.generation,
            BatchLane::Running(l) => l.ticket.generation,
            BatchLane::AwaitingClient(t) => t.ticket.generation,
        }
    }
    pub fn is_empty(&self) -> bool {
        matches!(self, BatchLane::Empty { .. })
    }
    pub fn is_awaiting(&self) -> bool {
        matches!(self, BatchLane::AwaitingClient(_))
    }
    pub fn key(&self) -> Option<&AttemptKey> {
        match self {
            BatchLane::Empty { .. } => None,
            BatchLane::Seeding(l) => Some(&l.key),
            BatchLane::Running(l) => Some(&l.key),
            BatchLane::AwaitingClient(t) => Some(&t.key),
        }
    }
}

/// Host-side continuous-batch scheduler (no GPU). Owns fixed slots, pending
/// inbox keyed by `(id, attempt_id)`, cohort sampling key, and per-lane
/// lifecycle. GPU batch state (`Qwen35DecodeBatchState`) is layered above
/// when available; tests drive this scheduler directly.
pub struct ContinuousBatchScheduler {
    pub max_batch: usize,
    pub lane_capacity: usize,
    pub lanes: Vec<BatchLane>,
    pub inbox: std::collections::VecDeque<AttemptKey>,
    pub pending: std::collections::HashMap<AttemptKey, BatchPendingRequest>,
    pub pending_sampling: std::collections::HashMap<AttemptKey, BatchSampling>,
    pub cohort_key: Option<BatchSamplingKey>,
    pub next_generation: u64,
}

#[derive(Debug, Clone)]
pub struct BatchPendingRequest {
    pub key: AttemptKey,
    /// The original generate request envelope, preserved verbatim so a
    /// think-barrier or cohort-parked request can be requeued to the daemon
    /// inbox byte/field-equivalently (never reconstructed field-by-field).
    pub original_msg: serde_json::Value,
    pub prompt: String,
    pub prompt_tokens: Vec<u32>,
    pub started_in_think: bool,
    pub system: Option<String>,
    pub assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    pub max_think_tokens: usize,
    pub max_tokens: usize,
    pub sampling: BatchSampling,
}

impl ContinuousBatchScheduler {
    /// Hard lane cap: every decode batch state addresses lanes with u64
    /// masks (`valid_lane_mask` in the archs), so more than 64 lanes can
    /// never be constructed. Staging refuses larger sizes before any GPU
    /// allocation; this constructor fails loudly rather than silently
    /// clamping or constructing a mask that cannot represent the lanes.
    pub const MAX_BATCH_LANES: usize = 64;

    pub fn new(max_batch: usize, lane_capacity: usize) -> Self {
        assert!(
            max_batch <= Self::MAX_BATCH_LANES,
            "continuous batch size {max_batch} exceeds the {} lane hard cap (u64 lane masks)",
            Self::MAX_BATCH_LANES
        );
        let lanes = (0..max_batch)
            .map(|_| BatchLane::Empty { generation: 0 })
            .collect();
        Self {
            max_batch,
            lane_capacity,
            lanes,
            inbox: std::collections::VecDeque::new(),
            pending: std::collections::HashMap::new(),
            pending_sampling: std::collections::HashMap::new(),
            cohort_key: None,
            next_generation: 1,
        }
    }

    pub fn capacity(&self) -> usize {
        self.max_batch
    }

    pub fn active_count(&self) -> usize {
        self.lanes
            .iter()
            .filter(|l| {
                matches!(
                    l,
                    BatchLane::Seeding(_) | BatchLane::Running(_) | BatchLane::AwaitingClient(_)
                )
            })
            .count()
    }
    pub fn running_count(&self) -> usize {
        self.lanes
            .iter()
            .filter(|l| matches!(l, BatchLane::Running(_)))
            .count()
    }
    pub fn awaiting_count(&self) -> usize {
        self.lanes
            .iter()
            .filter(|l| matches!(l, BatchLane::AwaitingClient(_)))
            .count()
    }

    pub fn empty_lanes(&self) -> Vec<usize> {
        self.lanes
            .iter()
            .enumerate()
            .filter(|(_, l)| l.is_empty())
            .map(|(i, _)| i)
            .collect()
    }

    pub fn enqueue(&mut self, req: BatchPendingRequest) -> bool {
        let key = req.key.clone();
        if self.pending.contains_key(&key) || self.inbox.contains(&key) {
            return false;
        }
        let sampling = req.sampling.clone();
        self.inbox.push_back(key.clone());
        self.pending.insert(key.clone(), req);
        self.pending_sampling.insert(key, sampling);
        true
    }

    pub fn cohort_key_for(&self, key: &AttemptKey) -> Option<BatchSamplingKey> {
        self.pending_sampling.get(key).map(|s| s.key())
    }

    /// The sampling cohort a NEW request must match to be admitted right
    /// now: the active cohort while lanes are live, else the front pending
    /// request's cohort (the next key to be assigned sets the cohort when
    /// the batch is idle). `None` means any cohort may join. Drivers use
    /// this to park cohort-mismatched requests BEFORE announcing/enqueueing
    /// them, so the mismatch never blocks the inbox head or emits a
    /// half-activated request.
    pub fn cohort_admission_key(&self) -> Option<BatchSamplingKey> {
        if let Some(key) = &self.cohort_key {
            return Some(key.clone());
        }
        let front = self.inbox.front()?;
        self.pending_sampling.get(front).map(|s| s.key())
    }

    pub fn try_assign_one(&mut self) -> Option<(AttemptKey, LaneTicket)> {
        let lane_idx = self.lanes.iter().position(|l| l.is_empty())?;
        let front_key = self.inbox.front()?.clone();
        let req_sampling_key = self.cohort_key_for(&front_key)?;
        if let Some(cohort) = &self.cohort_key {
            if cohort != &req_sampling_key {
                return None;
            }
        }
        let key = self.inbox.pop_front()?;
        let req = self.pending.get(&key)?.clone();
        let gen = self.next_generation;
        self.next_generation += 1;
        let ticket = LaneTicket {
            lane: lane_idx,
            generation: gen,
        };
        let lane = QwenBatchLane {
            key: key.clone(),
            ticket,
            sampling: req.sampling.clone(),
            prompt_len: req.prompt_tokens.len(),
            seq_pos: 0,
            next_token: None,
            rng_state: batch_rng_for_key(&key),
            conversation_tokens: Vec::new(),
            streamed_tokens: Vec::new(),
            bytes_fed_to_filter: 0,
            created_at: Instant::now(),
            prefill_done_at: None,
            first_token_at: None,
            max_active_lanes: 0,
        };
        self.lanes[lane_idx] = BatchLane::Running(lane);
        if self.cohort_key.is_none() {
            self.cohort_key = Some(req_sampling_key);
        }
        if batch_terminal_control()
            .mu
            .lock()
            .unwrap()
            .entries
            .contains_key(&key)
        {
            batch_transition_to_queued(&key.id, key.attempt_id);
            batch_bind_active(&key.id, key.attempt_id, ticket);
        }
        Some((key, ticket))
    }

    pub fn mark_awaiting_commit(&mut self, lane: usize, pending_done: serde_json::Value) -> bool {
        if lane >= self.lanes.len() {
            return false;
        }
        let lane_state =
            std::mem::replace(&mut self.lanes[lane], BatchLane::Empty { generation: 0 });
        match lane_state {
            BatchLane::Running(q) => {
                let key = q.key.clone();
                let ticket = q.ticket;
                let sampling = q.sampling.clone();
                let prompt_len = q.prompt_len;
                let seq_pos = q.seq_pos;
                let deadline = Instant::now() + CLIENT_TERMINAL_COMMIT_TIMEOUT;
                let term = BatchPendingTerminal {
                    key: key.clone(),
                    ticket,
                    sampling,
                    prompt_len,
                    seq_pos,
                    pending_done: pending_done.clone(),
                    deadline,
                };
                self.lanes[lane] = BatchLane::AwaitingClient(term);
                batch_mark_ready_with_pending(&key.id, key.attempt_id, ticket, pending_done);
                true
            }
            other => {
                self.lanes[lane] = other;
                false
            }
        }
    }

    pub fn commit_lane(&mut self, lane: usize, expected: &AttemptKey) -> bool {
        if lane >= self.lanes.len() {
            return false;
        }
        let is_awaiting =
            matches!(&self.lanes[lane], BatchLane::AwaitingClient(t) if &t.key == expected);
        if !is_awaiting {
            return false;
        }
        match batch_poll_decision(&expected.id, expected.attempt_id) {
            Some(ClientTerminalDecision::Commit) => {}
            _ => return false,
        }
        {
            let g = batch_terminal_control().mu.lock().unwrap();
            if let Some(e) = g.entries.get(expected) {
                if let BatchRegistryState::Ready { owner } = e.state {
                    let lane_gen = self.lanes[lane].generation();
                    if owner.generation != lane_gen {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
        let gen = self.lanes[lane].generation();
        self.lanes[lane] = BatchLane::Empty {
            generation: gen + 1,
        };
        self.pending.remove(expected);
        self.pending_sampling.remove(expected);
        batch_clear_terminal(&expected.id, expected.attempt_id);
        self.maybe_clear_cohort();
        true
    }

    pub fn abort_lane(&mut self, lane: usize, expected: &AttemptKey) -> bool {
        if lane >= self.lanes.len() {
            return false;
        }
        let lane_key = self.lanes[lane].key().cloned();
        if lane_key.as_ref() != Some(expected) {
            return false;
        }
        {
            let g = batch_terminal_control().mu.lock().unwrap();
            if let Some(e) = g.entries.get(expected) {
                match e.state {
                    BatchRegistryState::Active { owner } | BatchRegistryState::Ready { owner } => {
                        if owner.generation != self.lanes[lane].generation() {
                            return false;
                        }
                    }
                    _ => {}
                }
            }
        }
        let gen = self.lanes[lane].generation();
        self.lanes[lane] = BatchLane::Empty {
            generation: gen + 1,
        };
        self.pending.remove(expected);
        self.pending_sampling.remove(expected);
        self.inbox.retain(|k| k != expected);
        batch_clear_terminal(&expected.id, expected.attempt_id);
        self.maybe_clear_cohort();
        true
    }

    pub fn abort_queued(&mut self, key: &AttemptKey) -> bool {
        if !self.inbox.contains(key) {
            return false;
        }
        let state_ok = {
            let g = batch_terminal_control().mu.lock().unwrap();
            if let Some(e) = g.entries.get(key) {
                matches!(
                    e.state,
                    BatchRegistryState::Announced | BatchRegistryState::Queued
                )
            } else {
                false
            }
        };
        if !state_ok {
            return false;
        }
        if !batch_check_abort(&key.id, key.attempt_id) {
            return false;
        }
        self.inbox.retain(|k| k != key);
        self.pending.remove(key);
        self.pending_sampling.remove(key);
        batch_clear_terminal(&key.id, key.attempt_id);
        true
    }

    pub fn reset_lane_for_reuse(&mut self, lane: usize) -> bool {
        if lane >= self.lanes.len() {
            return false;
        }
        if !self.lanes[lane].is_empty() {
            return false;
        }
        true
    }

    pub fn fail_all_active(&mut self) -> Vec<AttemptKey> {
        let mut failed = Vec::new();
        for lane in &mut self.lanes {
            match lane {
                BatchLane::Running(q) | BatchLane::Seeding(q) => {
                    failed.push(q.key.clone());
                }
                BatchLane::AwaitingClient(t) => {
                    failed.push(t.key.clone());
                }
                BatchLane::Empty { .. } => {}
            }
            if !matches!(lane, BatchLane::Empty { .. }) {
                let gen = lane.generation();
                *lane = BatchLane::Empty {
                    generation: gen + 1,
                };
            }
        }
        for k in failed.iter() {
            self.pending.remove(k);
            self.pending_sampling.remove(k);
            batch_clear_terminal(&k.id, k.attempt_id);
        }
        for k in self.inbox.drain(..) {
            self.pending.remove(&k);
            self.pending_sampling.remove(&k);
            batch_clear_terminal(&k.id, k.attempt_id);
        }
        self.cohort_key = None;
        failed
    }

    pub fn maybe_clear_cohort(&mut self) {
        let has_active = self.lanes.iter().any(|l| {
            matches!(
                l,
                BatchLane::Seeding(_) | BatchLane::Running(_) | BatchLane::AwaitingClient(_)
            )
        });
        if !has_active {
            self.cohort_key = None;
        }
        if !has_active && self.inbox.is_empty() {
            self.cohort_key = None;
        }
    }

    pub fn find_lane_by_key(&self, key: &AttemptKey) -> Option<usize> {
        self.lanes
            .iter()
            .position(|l| l.key().is_some_and(|k| k == key))
    }
    pub fn pending_done_for(&self, lane: usize) -> Option<serde_json::Value> {
        match &self.lanes[lane] {
            BatchLane::AwaitingClient(t) => Some(t.pending_done.clone()),
            _ => None,
        }
    }
    pub fn deadline_for(&self, lane: usize) -> Option<Instant> {
        match &self.lanes[lane] {
            BatchLane::AwaitingClient(t) => Some(t.deadline),
            _ => None,
        }
    }
}
/// Deterministic per-request RNG derived from canonical base 0x13579BDF mixed
/// with the key's id bytes and attempt_id. Ensures each lane/request gets a
/// distinct stream rather than the same fixed seed.
pub fn batch_rng_for_key(key: &AttemptKey) -> u64 {
    let mut h = 0x13579BDFu64;
    h = h.wrapping_add(key.attempt_id.wrapping_mul(0x9E3779B97F4A7C15));
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 27;
    for &b in key.id.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
        h ^= h >> 33;
    }
    h = h.wrapping_mul(0xFF51AFD7ED558CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CEB9FE1A85EC53);
    h ^= h >> 33;
    if h == 0 {
        0x13579BDF
    } else {
        h
    }
}

/// Eligibility for continuous batching. Conservative: only single-GPU
/// exact HIP Qwen 5/6 and dense LFM 11 stateless text without excluded features.
pub fn is_batch_eligible(
    caps: &saddle_core::caps::ArchCaps,
    req: &saddle_core::caps::BatchEligibilityRequest,
) -> bool {
    if !req.serve_continuous_batch {
        return false;
    }
    if req.continuous_batch_size <= 1 {
        return false;
    }
    if !caps.supports_continuous_batch {
        return false;
    }
    if req.pp != 1 || req.ep_is_some {
        return false;
    }
    if req.has_image || req.has_tools || req.has_stop {
        return false;
    }
    if req.has_speculator || req.has_adaptive || req.has_pflash {
        return false;
    }
    if req.has_messages_history {
        return false;
    }
    if !req.think_mode_is_nonthink {
        return false;
    }
    true
}

/// Return the sole user message's text when the request has an eligible
/// single-user `messages` array. Prompt-only requests return `None`.
pub fn batch_single_user_content(msg: &serde_json::Value) -> Option<String> {
    let arr = msg.get("messages")?.as_array()?;
    if arr.len() != 1 {
        return None;
    }
    arr[0]
        .get("content")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
}

/// HTTP/OpenAI `messages` are batch-eligible only when absent (prompt path)
/// or exactly one user turn. Multi-turn, system/assistant/tool roles, and
/// tool_call payloads stay on the sequential route.
pub fn batch_messages_are_single_user(msg: &serde_json::Value) -> bool {
    let Some(messages) = msg.get("messages") else {
        return true;
    };
    let Some(arr) = messages.as_array() else {
        return false;
    };
    if arr.is_empty() {
        return true;
    }
    if arr.len() != 1 {
        return false;
    }
    let m0 = &arr[0];
    if m0.get("role").and_then(|v| v.as_str()) != Some("user") {
        return false;
    }
    // Shared contract with CLI: sole user content must be a plain string.
    // Multipart (array) and image content are sequential-only.
    if m0.get("content").and_then(|v| v.as_str()).is_none() {
        return false;
    }
    // Tool-call payloads on the sole message force sequential (batch v1 has
    // no tools path). Empty / missing tool_calls is fine.
    if let Some(tc) = m0.get("tool_calls") {
        if tc.as_array().is_some_and(|a| !a.is_empty()) || tc.is_object() {
            return false;
        }
    }
    true
}

pub fn parse_continuous_batch_size(params: Option<&serde_json::Value>) -> usize {
    params
        .and_then(|v| v.get("continuous_batch_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize
}

pub fn parse_serve_continuous_batch(msg: &serde_json::Value) -> bool {
    msg.get("params")
        .and_then(|p| p.get("serve_continuous_batch"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
        || msg
            .get("serve_continuous_batch")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
}

pub fn resolve_batch_sampling(
    msg: &serde_json::Value,
    m: &hipfire_loader::LoadedModel,
) -> BatchSampling {
    let defaults = hipfire_loader::carrier_for(m.arch_id)
        .map(|c| c.sampling_defaults())
        .unwrap_or_default();
    let (arch_default_temp, arch_default_top_p) = (defaults.temp, defaults.top_p);
    let default_temp = m
        .rec_temperature
        .map(|x| x as f64)
        .unwrap_or(arch_default_temp);
    let default_top_p = m.rec_top_p.map(|x| x as f64).unwrap_or(arch_default_top_p);
    let temp = msg
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(default_temp) as f32;
    let top_p = msg
        .get("top_p")
        .and_then(|v| v.as_f64())
        .unwrap_or(default_top_p) as f32;
    let default_repeat = defaults.repeat_penalty;
    let repeat_penalty = msg
        .get("repeat_penalty")
        .or_else(|| msg.get("repetition_penalty"))
        .and_then(|v| v.as_f64())
        .unwrap_or(default_repeat) as f32;
    let repeat_window = msg
        .get("repeat_window")
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as usize;
    let presence_penalty =
        (msg.get("presence_penalty")
            .and_then(|v| v.as_f64())
            .unwrap_or(m.rec_presence_penalty.unwrap_or(0.0) as f64) as f32)
            .max(0.0);
    let frequency_penalty = (msg
        .get("frequency_penalty")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32)
        .max(0.0);
    let top_k: Option<u32> = msg
        .get("top_k")
        .and_then(|v| v.as_u64())
        .map(|k| k as u32)
        .or_else(|| m.rec_top_k.map(|k| k as u32))
        .filter(|&k| k > 0);
    let min_p: Option<f32> = msg
        .get("min_p")
        .and_then(|v| v.as_f64())
        .map(|p| p as f32)
        .or(m.rec_min_p)
        .filter(|&p| p > 0.0);
    BatchSampling {
        temp,
        top_p,
        top_k,
        min_p,
        repeat_penalty,
        presence_penalty,
        frequency_penalty,
        repeat_window,
    }
}

pub fn lfm_fast_path_candidate_len(sched: &ContinuousBatchScheduler) -> usize {
    if sched.active_count() != 0 || sched.awaiting_count() != 0 {
        return 0;
    }
    if sched.inbox.len() < 2 {
        return 0;
    }
    // Peek front without mutating.
    let front_key = match sched.inbox.front() {
        Some(k) => k.clone(),
        None => return 0,
    };
    let front_req = match sched.pending.get(&front_key) {
        Some(r) => r,
        None => return 0,
    };
    // Front must be ordinary: non-thinking, max_tokens>0, within capacity, not aborted.
    if front_req.started_in_think {
        return 0;
    }
    if front_req.max_tokens == 0 {
        return 0;
    }
    if front_req.prompt_tokens.is_empty()
        || front_req.prompt_tokens.len() >= sched.lane_capacity
        || !crate::terminal::batch_lfm_admission_ok(
            front_req.prompt_tokens.len(),
            front_req.max_tokens,
            sched.lane_capacity,
        )
    {
        return 0;
    }
    if crate::terminal::batch_check_abort(&front_key.id, front_key.attempt_id) {
        return 0;
    }
    let first_len = front_req.prompt_tokens.len();
    let first_cohort = match sched.pending_sampling.get(&front_key) {
        Some(s) => s.key(),
        None => return 0,
    };
    let mut count = 1usize;
    for key in sched.inbox.iter().skip(1) {
        if count >= sched.max_batch {
            break;
        }
        let req = match sched.pending.get(key) {
            Some(r) => r,
            None => break,
        };
        if req.started_in_think {
            break;
        }
        if req.max_tokens == 0 {
            break;
        }
        if req.prompt_tokens.len() != first_len {
            break;
        }
        if req.prompt_tokens.is_empty()
            || req.prompt_tokens.len() >= sched.lane_capacity
            || !crate::terminal::batch_lfm_admission_ok(
                req.prompt_tokens.len(),
                req.max_tokens,
                sched.lane_capacity,
            )
        {
            break;
        }
        if crate::terminal::batch_check_abort(&key.id, key.attempt_id) {
            break;
        }
        let cohort = match sched.pending_sampling.get(key) {
            Some(s) => s.key(),
            None => break,
        };
        if cohort != first_cohort {
            break;
        }
        count += 1;
    }
    if count >= 2 {
        count
    } else {
        0
    }
}

/// Small helper that reuses the existing `sample_lane_product` contract and
/// populates the same lane fields as the sequential `while let try_assign_one` path.
pub fn lfm_populate_lane_after_sample(
    sched: &mut ContinuousBatchScheduler,
    lane_idx: usize,
    next_token: u32,
    next_rng: u32,
    prompt_len: usize,
) {
    if let BatchLane::Running(lane) = &mut sched.lanes[lane_idx] {
        lane.prompt_len = prompt_len;
        lane.seq_pos = prompt_len;
        lane.next_token = Some(next_token);
        lane.rng_state = next_rng as u64;
        lane.conversation_tokens = Vec::new();
        lane.streamed_tokens = Vec::new();
        lane.bytes_fed_to_filter = 0;
        lane.prefill_done_at = Some(Instant::now());
    }
}

// ── Daemon inbox with barrier pushback ─────────────────────────────────

/// Async inbox for DaemonMsg. The reader announces keys before queuing;
/// the main loop drains via `try_recv` between GPU ticks. A non-generate
/// or cohort-incompatible message is `push_front`ed as a barrier so it is
/// not lost and returns to the outer `inbox.recv()` for sequential handling.
pub struct DaemonInbox {
    pub rx: std::sync::mpsc::Receiver<DaemonMsg>,
    pub backlog: std::collections::VecDeque<DaemonMsg>,
}

impl DaemonInbox {
    pub fn new(rx: std::sync::mpsc::Receiver<DaemonMsg>) -> Self {
        Self {
            rx,
            backlog: std::collections::VecDeque::new(),
        }
    }
    pub fn recv(&mut self) -> Result<DaemonMsg, std::sync::mpsc::RecvError> {
        if let Some(msg) = self.backlog.pop_front() {
            return Ok(msg);
        }
        self.rx.recv()
    }
    pub fn try_recv(&mut self) -> Result<DaemonMsg, std::sync::mpsc::TryRecvError> {
        if let Some(msg) = self.backlog.pop_front() {
            return Ok(msg);
        }
        self.rx.try_recv()
    }
    pub fn recv_timeout(
        &mut self,
        timeout: std::time::Duration,
    ) -> Result<DaemonMsg, std::sync::mpsc::RecvTimeoutError> {
        if let Some(msg) = self.backlog.pop_front() {
            return Ok(msg);
        }
        self.rx.recv_timeout(timeout)
    }
    pub fn push_front(&mut self, msg: DaemonMsg) {
        self.backlog.push_front(msg);
    }
    /// Park a message at the back of the backlog. Drivers park barrier
    /// messages here so the outer daemon loop serves them between batch
    /// waves; the front stays free for control messages already in flight.
    pub fn push_back(&mut self, msg: DaemonMsg) {
        self.backlog.push_back(msg);
    }
}

/// Message types pushed from the stdin-reader thread to the main
/// processing loop. Abort/commit control messages are NOT forwarded —
/// they're handled inline in the reader thread via
/// [`apply_terminal_control`]. This is what lets the abort signal
/// interrupt a mid-flight prefill; the main loop is blocked on prefill
/// compute and would only see new stdin lines after that prefill completed.
#[derive(Debug, Clone)]
pub enum DaemonMsg {
    Regular(serde_json::Value),
    ParseError(String),
}

pub type CaskConfig = hipfire_runtime::loader_api::CaskConfig;

/// Error from the real GPU batch driver. Host stub never errors.
#[derive(Debug)]
pub enum BatchDriveError {
    Gpu(String),
    Poisoned(String),
}
impl std::fmt::Display for BatchDriveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchDriveError::Gpu(s) => write!(f, "batch gpu error: {}", s),
            BatchDriveError::Poisoned(s) => write!(f, "batch poisoned: {}", s),
        }
    }
}
impl std::error::Error for BatchDriveError {}

pub fn lane_max_tokens(key: &AttemptKey, sched: &ContinuousBatchScheduler) -> usize {
    sched.pending.get(key).map(|r| r.max_tokens).unwrap_or(4096)
}

pub fn batch_finite_rate(count: usize, seconds: f64) -> f64 {
    if seconds <= 0.0 || !seconds.is_finite() {
        return 0.0;
    }
    let rate = count as f64 / seconds;
    if rate.is_finite() {
        rate
    } else {
        0.0
    }
}

pub struct BatchLaneDoneMetrics {
    pub latency_ms: f64,
    pub ttft_ms: f64,
    pub prefill_ms: f64,
    pub prefill_tok_s: f64,
    pub tok_s: f64,
    pub decode_tok_s: f64,
}

pub fn batch_lane_done_metrics(
    created_at: Instant,
    prefill_done_at: Option<Instant>,
    first_token_at: Option<Instant>,
    finished_at: Instant,
    prompt_len: usize,
    generated: usize,
) -> BatchLaneDoneMetrics {
    let wall_s = finished_at
        .saturating_duration_since(created_at)
        .as_secs_f64();
    let latency_ms = if wall_s.is_finite() && wall_s > 0.0 {
        wall_s * 1000.0
    } else {
        0.0
    };

    let prefill_s = prefill_done_at
        .map(|t| t.saturating_duration_since(created_at).as_secs_f64())
        .filter(|s| s.is_finite() && *s > 0.0)
        .unwrap_or(0.0);
    let prefill_ms = prefill_s * 1000.0;
    let prefill_tok_s = batch_finite_rate(prompt_len, prefill_s);

    let ttft_s = first_token_at
        .map(|t| t.saturating_duration_since(created_at).as_secs_f64())
        .filter(|s| s.is_finite() && *s > 0.0)
        .unwrap_or(prefill_s);
    let ttft_ms = if ttft_s > 0.0 {
        ttft_s * 1000.0
    } else {
        prefill_ms
    };
    let tok_s = batch_finite_rate(generated, wall_s);

    // Decode rate over the post-first-token interval. Zero/one-token and missing
    // first-token stamps stay finite zero (no positive span to amortize).
    let decode_tok_s = match first_token_at {
        Some(ft) if generated > 0 => {
            let decode_s = finished_at.saturating_duration_since(ft).as_secs_f64();
            batch_finite_rate(generated, decode_s)
        }
        _ => 0.0,
    };

    BatchLaneDoneMetrics {
        latency_ms: if latency_ms.is_finite() {
            latency_ms
        } else {
            0.0
        },
        ttft_ms: if ttft_ms.is_finite() { ttft_ms } else { 0.0 },
        prefill_ms: if prefill_ms.is_finite() {
            prefill_ms
        } else {
            0.0
        },
        prefill_tok_s,
        tok_s,
        decode_tok_s,
    }
}

/// Attach actual continuous-batch route evidence to a batch-driver done envelope.
/// Only the real batch driver emits this; sequential fallbacks must not call it.
pub fn attach_continuous_batch_route_evidence(
    envelope: &mut serde_json::Value,
    slots: usize,
    lane: usize,
    lane_capacity: usize,
    max_active_lanes: usize,
) {
    envelope["execution_mode"] = serde_json::json!("continuous_batch_independent");
    envelope["continuous_batch"] = serde_json::json!({
        "executed": true,
        "slots": slots,
        "lane": lane,
        "lane_capacity": lane_capacity,
        "max_active_lanes": max_active_lanes,
        "refill": "continuous",
    });
}

/// CPU-side attractor blocker for the AR generate path.
pub fn block_attractor_unclosed_cpu(
    logits: &mut [f32],
    history: &[u32],
    open_id: u32,
    close_id: u32,
    window: usize,
    threshold: usize,
) {
    if window == 0 || threshold == 0 || open_id == close_id {
        return;
    }
    let start = history.len().saturating_sub(window);
    let mut depth: i32 = 0;
    for &t in &history[start..] {
        if t == open_id {
            depth += 1;
        } else if t == close_id && depth > 0 {
            depth -= 1;
        }
    }
    if depth >= threshold as i32 {
        if let Some(slot) = logits.get_mut(open_id as usize) {
            *slot = f32::NEG_INFINITY;
        }
    }
}
