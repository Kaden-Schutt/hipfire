// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Priority scheduling and session batching policy shared by control planes.

#[cfg(test)]
use hipfire_model::model_worker_key_id;
use hipfire_model::{normalize_model_worker_key, same_model_worker_key, ModelWorkerKey};
use hipfire_state::generate_state_kind_sets_match_exactly;
use std::collections::{BTreeMap, HashSet};

pub const SCHED_PRIORITY_REALTIME: u8 = 0;
pub const SCHED_PRIORITY_DEFAULT: u8 = 64;
pub const SCHED_PRIORITY_OPPORTUNISTIC: u8 = 255;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchedulerPriorityClass {
    Realtime,
    High,
    Interactive,
    Background,
    Bulk,
    Opportunistic,
}

impl SchedulerPriorityClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Realtime => "realtime",
            Self::High => "high",
            Self::Interactive => "interactive",
            Self::Background => "background",
            Self::Bulk => "bulk",
            Self::Opportunistic => "opportunistic",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchedulerPriorityPolicy {
    pub priority: u8,
    pub priority_class: SchedulerPriorityClass,
    pub coalesce_wait_ms: u64,
    pub max_batch_size: usize,
    pub resident_state_max: usize,
    pub spillable_batch_max: usize,
    pub disk_spill_allowed: bool,
    pub disk_spill_min_priority: u8,
    pub target_pair_tokens: usize,
    pub max_processing_ms: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SchedulerPolicyEnv {
    values: BTreeMap<String, String>,
}

impl SchedulerPolicyEnv {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self {
            values: pairs
                .into_iter()
                .map(|(key, value)| (key.into(), value.into()))
                .collect(),
        }
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpportunisticDispatchInput {
    pub compatible_queued_tokens: usize,
    pub schedule_clear: bool,
    pub target_pair_tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerPrefillPolicyControls {
    pub resident_state_cache: bool,
    pub resident_checkpoint_max: usize,
    pub state_cache_disk: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionStateHandle {
    pub worker_key: ModelWorkerKey,
    pub state_kinds: Vec<String>,
    pub logical_position: usize,
    pub cached_prefix_tokens: usize,
    pub runtime_state_handle: Option<String>,
    pub daemon_prefix_hash: Option<String>,
    pub daemon_prefix_len: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestSessionDraft {
    pub id: String,
    pub worker_key: ModelWorkerKey,
    pub priority: u8,
    pub prompt_tokens: Vec<u32>,
    pub suffix_tokens: Vec<u32>,
    pub cached_prefix_tokens: usize,
    pub state_handle: SessionStateHandle,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CreateRequestSessionInput {
    pub id: String,
    pub worker_key: ModelWorkerKey,
    pub prompt_tokens: Vec<u32>,
    pub cached_prefix_tokens: Option<usize>,
    pub priority: Option<i64>,
    pub state_kinds: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueuedPrefillRequest {
    pub session: RequestSessionDraft,
    pub enqueued_at_ms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NextBatchInput {
    pub now_ms: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreviewPrefillBatchInput {
    pub now_ms: u64,
    pub incoming_session: Option<RequestSessionDraft>,
    pub incoming_enqueued_at_ms: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillBatchSelection {
    pub sessions: Vec<RequestSessionDraft>,
    pub policy: SchedulerPriorityPolicy,
    pub total_prompt_tokens: usize,
    pub total_suffix_tokens: usize,
    pub max_prompt_tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActiveDecodeSession {
    pub id: String,
    pub worker_key_id: String,
    pub priority: u8,
    pub runtime_state_handle: String,
    pub logical_position: usize,
    pub cached_prefix_tokens: usize,
    pub generated_tokens: usize,
    pub max_tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeBatchSelection {
    pub sessions: Vec<ActiveDecodeSession>,
    pub policy: SchedulerPriorityPolicy,
}

fn parse_integer(value: Option<&str>, fallback: i64) -> i64 {
    let Some(value) = value else {
        return fallback;
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return fallback;
    }
    trimmed
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .map(|v| v.floor() as i64)
        .unwrap_or(fallback)
}

fn parse_boolean(value: Option<&str>, fallback: bool) -> bool {
    let Some(value) = value else {
        return fallback;
    };
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "on" | "yes"
    )
}

pub fn clamp_scheduler_priority(value: i64) -> u8 {
    value.clamp(0, 255) as u8
}

pub fn clamp_scheduler_priority_f64(value: f64) -> u8 {
    if !value.is_finite() {
        return SCHED_PRIORITY_DEFAULT;
    }
    clamp_scheduler_priority(value.floor() as i64)
}

pub fn parse_scheduler_priority(value: Option<&str>, fallback: u8) -> u8 {
    clamp_scheduler_priority(parse_integer(value, i64::from(fallback)))
}

pub fn parse_default_scheduler_priority(env: &SchedulerPolicyEnv) -> u8 {
    parse_scheduler_priority(
        env.get("HIPFIRE_SCHED_PRIORITY_DEFAULT"),
        SCHED_PRIORITY_DEFAULT,
    )
}

pub fn scheduler_priority_class(priority: u8) -> SchedulerPriorityClass {
    match priority {
        0 => SchedulerPriorityClass::Realtime,
        1..=63 => SchedulerPriorityClass::High,
        64..=127 => SchedulerPriorityClass::Interactive,
        128..=191 => SchedulerPriorityClass::Background,
        192..=254 => SchedulerPriorityClass::Bulk,
        255 => SchedulerPriorityClass::Opportunistic,
    }
}

pub fn parse_server_prefill_policy_controls(
    env: &SchedulerPolicyEnv,
) -> ServerPrefillPolicyControls {
    let resident_state_cache = parse_boolean(
        env.get("HIPFIRE_SERVER_PREFILL_STATE_CACHE"),
        parse_boolean(env.get("HIPFIRE_SCHED_STATE_CACHE_RESIDENT"), false),
    );
    let resident_checkpoint_max = parse_integer(
        env.get("HIPFIRE_STATE_CACHE_MAX_CHECKPOINTS")
            .or_else(|| env.get("HIPFIRE_SERVER_PREFILL_STATE_CACHE_MAX")),
        4,
    )
    .clamp(0, 64) as usize;
    let state_cache_disk = parse_boolean(
        env.get("HIPFIRE_SCHED_STATE_CACHE_DISK"),
        parse_boolean(
            env.get("HIPFIRE_SERVER_PREFILL_BATCH_STATE_CACHE_DISK"),
            false,
        ),
    );
    let legacy_state_cache_disk = parse_boolean(
        env.get("HIPFIRE_SERVER_PREFILL_BATCH_STATE_CACHE_DISK"),
        false,
    );
    ServerPrefillPolicyControls {
        resident_state_cache,
        resident_checkpoint_max,
        state_cache_disk: state_cache_disk || legacy_state_cache_disk,
    }
}

pub fn server_prefill_batch_enabled(env: &SchedulerPolicyEnv) -> bool {
    env.get("HIPFIRE_SERVER_PREFILL_BATCH")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "on" | "true"
            )
        })
        .unwrap_or(false)
}

pub fn server_prefill_batch_health_json(env: &SchedulerPolicyEnv) -> serde_json::Value {
    if !server_prefill_batch_enabled(env) {
        return serde_json::json!({ "enabled": false });
    }

    let priority = parse_default_scheduler_priority(env);
    let policy = scheduler_policy_for_priority(priority, env);
    let controls = parse_server_prefill_policy_controls(env);

    let mut payload = serde_json::json!({
        "enabled": true,
        "queued": 0,
        "eligible": 0,
        "selected": 0,
        "skipped": 0,
        "total_batches": 0,
        "fused_batches": 0,
        "fallback_batches": 0,
        "batch_size_histogram": {},
        "cache_hits": 0,
        "cache_misses": 0,
        "metadata_cache_hits": 0,
        "runtime_cache_hits": 0,
        "queue_size": 0,
        "pending_requests": 0,
        "resident_runtime_sessions": 0,
        "resident_decode_sessions": 0,
        "resident_checkpoints": 0,
        "resident_checkpoint_max": controls.resident_checkpoint_max,
        "resident_state_cache": controls.resident_state_cache,
        "resident_state_limit": policy.resident_state_max,
        "spillable_batch_max": policy.spillable_batch_max,
        "spillable_sessions": 0,
        "state_cache_disk": controls.state_cache_disk,
        "state_cache_disk_min_priority": policy.disk_spill_min_priority,
        "disk_spill_allowed": policy.disk_spill_allowed,
        "state_cache_evictions_total": 0,
        "state_cache_recompute_required_total": 0,
        "generate_batch_prefill_capability": "unknown",
        "generate_batch_prefill_capability_reason": "rust_server_daemon_capability_not_probed",
        "queue_wait_reason": "disabled",
        "fallback_reason": "rust_server_scheduler_metadata_only",
        "runtime_dispatch_skipped_reason": "rust_server_prefill_queue_not_enabled",
        "selected_batch_size": 0,
        "last_prefill_tokens": 0,
        "last_prefill_ms": 0,
        "last_prefill_tok_s": 0,
    });
    payload["policy"] = serde_json::json!({
        "priority": policy.priority,
        "priority_class": policy.priority_class.as_str(),
        "max_batch": policy.max_batch_size,
        "wait_ms": policy.coalesce_wait_ms,
        "target_pair_tokens": policy.target_pair_tokens,
        "max_processing_ms": policy.max_processing_ms,
    });
    payload
}

pub fn server_decode_batch_health_json(env: &SchedulerPolicyEnv) -> serde_json::Value {
    if !server_prefill_batch_enabled(env) {
        return serde_json::json!({ "enabled": false });
    }
    serde_json::json!({
        "enabled": true,
        "eligible": 0,
        "selected": 0,
        "skipped": 0,
        "active_sessions": 0,
        "selected_batch_size": 0,
        "total_batches": 0,
        "serial_batches": 0,
        "fused_batches": 0,
        "last_skipped_reason": "rust_server_decode_scheduler_not_enabled",
        "fallback_reason": "rust_server_scheduler_metadata_only",
    })
}

pub fn server_state_cache_health_json(env: &SchedulerPolicyEnv) -> serde_json::Value {
    if !server_prefill_batch_enabled(env) {
        return serde_json::json!({ "enabled": false });
    }
    let controls = parse_server_prefill_policy_controls(env);
    serde_json::json!({
        "enabled": controls.resident_state_cache || controls.state_cache_disk,
        "resident_enabled": controls.resident_state_cache,
        "resident_checkpoints": 0,
        "resident_checkpoint_max": controls.resident_checkpoint_max,
        "disk_enabled": controls.state_cache_disk,
        "daemon_prefix_hash": false,
        "daemon_prefix_hash_entries": 0,
        "semantic_boundary_checkpoints": false,
        "semantic_boundary_checkpoint_entries": 0,
        "prefix_hash_preflight_requests": 0,
        "prefix_hash_preflight_candidates": 0,
        "prefix_hash_preflight_matches": 0,
        "prefix_hash_preflight_boundary_matches": 0,
        "shared_prefix_fanout_groups": 0,
        "shared_prefix_fanout_followers": 0,
        "responses_previous_response_hits": 0,
        "responses_previous_response_misses": 0,
        "responses_stored_contexts": 0,
        "entries": 0,
        "bytes": 0,
        "metadata_hits": 0,
        "runtime_hits": 0,
        "evictions_total": 0,
        "recompute_required_total": 0,
    })
}

pub fn server_batch_health_json() -> serde_json::Value {
    serde_json::json!({ "enabled": false })
}

pub fn scheduler_policy_for_priority(
    priority: u8,
    env: &SchedulerPolicyEnv,
) -> SchedulerPriorityPolicy {
    let priority_class = scheduler_priority_class(priority);
    let max_batch_size = parse_integer(
        env.get("HIPFIRE_SCHED_PREFILL_BATCH_MAX")
            .or_else(|| env.get("HIPFIRE_SERVER_PREFILL_BATCH_MAX")),
        8,
    )
    .clamp(1, 64) as usize;
    let disk_spill_min_priority =
        parse_scheduler_priority(env.get("HIPFIRE_SCHED_STATE_CACHE_DISK_MIN_PRIORITY"), 128);
    let disk_spill_allowed = parse_server_prefill_policy_controls(env).state_cache_disk
        && priority >= disk_spill_min_priority;

    let state_policy_for_max = |effective_max_batch_size: usize| -> (usize, usize) {
        let resident_state_max = parse_integer(
            env.get("HIPFIRE_SCHED_RESIDENT_STATE_MAX"),
            effective_max_batch_size as i64,
        )
        .clamp(1, 64) as usize;
        let spillable_batch_max = parse_integer(
            env.get("HIPFIRE_SCHED_SPILLABLE_BATCH_MAX"),
            effective_max_batch_size as i64,
        )
        .clamp(resident_state_max as i64, 64) as usize;
        (resident_state_max, spillable_batch_max)
    };

    let legacy_interactive_wait = env.get("HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS");
    let realtime_wait =
        parse_integer(env.get("HIPFIRE_SCHED_PREFILL_WAIT_MS_REALTIME"), 0).max(0) as u64;
    let interactive_default = legacy_interactive_wait
        .map(|_| parse_integer(legacy_interactive_wait, 5))
        .unwrap_or(5);
    let interactive_wait = parse_integer(
        env.get("HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE"),
        interactive_default,
    )
    .max(0) as u64;
    let legacy_background_wait = legacy_interactive_wait
        .map(|_| parse_integer(legacy_interactive_wait, 25).max(0) * 2)
        .unwrap_or(25);
    let background_wait = parse_integer(
        env.get("HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND"),
        legacy_background_wait,
    )
    .max(0) as u64;
    let opportunistic_background_wait =
        parse_integer(env.get("HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND"), 25).max(0) as u64;
    let opportunistic_pair_tokens =
        parse_integer(env.get("HIPFIRE_SCHED_OPPORTUNISTIC_MIN_PAIR_TOKENS"), 256).max(1) as usize;

    let build = |coalesce_wait_ms, max_batch_size, target_pair_tokens, max_processing_ms| {
        let (resident_state_max, spillable_batch_max) = state_policy_for_max(max_batch_size);
        SchedulerPriorityPolicy {
            priority,
            priority_class,
            coalesce_wait_ms,
            max_batch_size,
            resident_state_max,
            spillable_batch_max,
            disk_spill_allowed,
            disk_spill_min_priority,
            target_pair_tokens,
            max_processing_ms,
        }
    };

    match priority_class {
        SchedulerPriorityClass::Realtime => build(realtime_wait, 1, 1, 25),
        SchedulerPriorityClass::High => {
            build(interactive_wait.min(2), max_batch_size.min(4), 32, 50)
        }
        SchedulerPriorityClass::Interactive => build(interactive_wait, max_batch_size, 64, 100),
        SchedulerPriorityClass::Background => build(background_wait, max_batch_size, 128, 250),
        SchedulerPriorityClass::Bulk => build(
            background_wait.saturating_mul(2),
            max_batch_size,
            opportunistic_pair_tokens,
            500,
        ),
        SchedulerPriorityClass::Opportunistic => build(
            opportunistic_background_wait.saturating_mul(4),
            max_batch_size,
            opportunistic_pair_tokens,
            1000,
        ),
    }
}

pub fn should_dispatch_opportunistic(input: OpportunisticDispatchInput) -> bool {
    input.schedule_clear || input.compatible_queued_tokens >= input.target_pair_tokens.max(1)
}

pub fn create_request_session_draft(input: CreateRequestSessionInput) -> RequestSessionDraft {
    let cached_prefix_tokens = input
        .cached_prefix_tokens
        .unwrap_or(0)
        .min(input.prompt_tokens.len());
    let worker_key = normalize_model_worker_key(&input.worker_key);
    let suffix_tokens = input.prompt_tokens[cached_prefix_tokens..].to_vec();
    let priority = input
        .priority
        .map(clamp_scheduler_priority)
        .unwrap_or(SCHED_PRIORITY_DEFAULT);
    RequestSessionDraft {
        id: input.id,
        worker_key: worker_key.clone(),
        priority,
        prompt_tokens: input.prompt_tokens,
        suffix_tokens,
        cached_prefix_tokens,
        state_handle: SessionStateHandle {
            worker_key,
            state_kinds: input.state_kinds,
            logical_position: cached_prefix_tokens,
            cached_prefix_tokens,
            runtime_state_handle: None,
            daemon_prefix_hash: None,
            daemon_prefix_len: None,
        },
    }
}

pub fn sessions_compatible_for_prefill(a: &RequestSessionDraft, b: &RequestSessionDraft) -> bool {
    if !same_model_worker_key(&a.worker_key, &b.worker_key) {
        return false;
    }
    generate_state_kind_sets_match_exactly(&a.state_handle.state_kinds, &b.state_handle.state_kinds)
}

#[derive(Clone, Debug)]
pub struct PriorityPrefillScheduler {
    env: SchedulerPolicyEnv,
    buckets: Vec<Vec<QueuedPrefillRequest>>,
    queued_ids: HashSet<String>,
    queued_count: usize,
}

impl Default for PriorityPrefillScheduler {
    fn default() -> Self {
        Self::new(SchedulerPolicyEnv::empty())
    }
}

impl PriorityPrefillScheduler {
    pub fn new(env: SchedulerPolicyEnv) -> Self {
        Self {
            env,
            buckets: (0..=255).map(|_| Vec::new()).collect(),
            queued_ids: HashSet::new(),
            queued_count: 0,
        }
    }

    pub fn size(&self) -> usize {
        self.queued_count
    }

    pub fn has_queued(&self, id: &str) -> bool {
        self.queued_ids.contains(id)
    }

    pub fn enqueue(
        &mut self,
        session: RequestSessionDraft,
        enqueued_at_ms: u64,
    ) -> Result<(), String> {
        let max_queued = self.max_queued_requests();
        if max_queued > 0 && self.queued_count >= max_queued {
            return Err(format!(
                "prefill scheduler backpressure: queued={} max={max_queued}",
                self.queued_count
            ));
        }
        if self.queued_ids.contains(&session.id) {
            return Err(format!("request session is already queued: {}", session.id));
        }
        let priority = session.priority as usize;
        let id = session.id.clone();
        self.buckets[priority].push(QueuedPrefillRequest {
            session,
            enqueued_at_ms,
        });
        self.queued_ids.insert(id);
        self.queued_count += 1;
        Ok(())
    }

    pub fn enqueue_if_absent(
        &mut self,
        session: RequestSessionDraft,
        enqueued_at_ms: u64,
    ) -> Result<bool, String> {
        if self.has_queued(&session.id) {
            return Ok(false);
        }
        self.enqueue(session, enqueued_at_ms)?;
        Ok(true)
    }

    pub fn cancel(&mut self, id: &str) -> bool {
        if !self.queued_ids.contains(id) {
            return false;
        }
        for bucket in &mut self.buckets {
            if let Some(index) = bucket.iter().position(|entry| entry.session.id == id) {
                bucket.remove(index);
                self.queued_ids.remove(id);
                self.queued_count = self.queued_count.saturating_sub(1);
                return true;
            }
        }
        self.queued_ids.remove(id);
        false
    }

    pub fn next_prefill_batch(&mut self, input: NextBatchInput) -> Option<PrefillBatchSelection> {
        if let Some(aged) = self.select_aged_candidate(input.now_ms) {
            self.remove_selected(&aged.sessions);
            return Some(aged);
        }

        for priority in 0..self.buckets.len() {
            if self.buckets[priority].is_empty() {
                continue;
            }
            let candidate =
                self.select_from_bucket(priority as u8, &self.buckets[priority], input.now_ms)?;
            self.remove_selected(&candidate.sessions);
            return Some(candidate);
        }
        None
    }

    pub fn preview_next_prefill_batch(
        &self,
        input: PreviewPrefillBatchInput,
    ) -> Option<PrefillBatchSelection> {
        for priority in 0..self.buckets.len() {
            let mut bucket = self.buckets[priority].clone();
            let already_queued_incoming = input
                .incoming_session
                .as_ref()
                .map(|incoming| bucket.iter().any(|entry| entry.session.id == incoming.id))
                .unwrap_or(false);
            if let Some(incoming) = input.incoming_session.as_ref() {
                if incoming.priority as usize == priority && !already_queued_incoming {
                    bucket.push(QueuedPrefillRequest {
                        session: incoming.clone(),
                        enqueued_at_ms: input.incoming_enqueued_at_ms.unwrap_or(input.now_ms),
                    });
                }
            }
            if bucket.is_empty() {
                continue;
            }
            let candidate = self.select_from_bucket(priority as u8, &bucket, input.now_ms)?;
            let Some(incoming) = input.incoming_session.as_ref() else {
                return Some(candidate);
            };
            if candidate
                .sessions
                .iter()
                .any(|session| session.id == incoming.id)
            {
                return Some(candidate);
            }
            return None;
        }
        None
    }

    fn select_from_bucket(
        &self,
        priority: u8,
        bucket: &[QueuedPrefillRequest],
        now_ms: u64,
    ) -> Option<PrefillBatchSelection> {
        let first = bucket.first()?;
        let policy = scheduler_policy_for_priority(first.session.priority, &self.env);
        let selection_limit = self.selection_limit(&policy);
        let compatible = bucket
            .iter()
            .filter(|entry| sessions_compatible_for_prefill(&first.session, &entry.session))
            .take(selection_limit)
            .cloned()
            .collect::<Vec<_>>();
        let total_suffix_tokens = compatible
            .iter()
            .map(|entry| entry.session.suffix_tokens.len())
            .sum();

        if policy.priority_class == SchedulerPriorityClass::Opportunistic {
            let dispatch = should_dispatch_opportunistic(OpportunisticDispatchInput {
                compatible_queued_tokens: total_suffix_tokens,
                schedule_clear: !self.has_queued_higher_priority(priority),
                target_pair_tokens: policy.target_pair_tokens,
            });
            return dispatch.then(|| self.selection(&compatible, policy));
        }

        let waited_ms = now_ms.saturating_sub(first.enqueued_at_ms);
        if compatible.len() >= selection_limit || waited_ms >= policy.coalesce_wait_ms {
            Some(self.selection(&compatible, policy))
        } else {
            None
        }
    }

    fn max_queued_requests(&self) -> usize {
        parse_integer(self.env.get("HIPFIRE_SCHED_PREFILL_MAX_QUEUED"), 256).max(0) as usize
    }

    fn aging_ms(&self) -> u64 {
        parse_integer(self.env.get("HIPFIRE_SCHED_DEADLINE_AGING_MS"), 0).max(0) as u64
    }

    fn select_aged_candidate(&self, now_ms: u64) -> Option<PrefillBatchSelection> {
        let aging_ms = self.aging_ms();
        if aging_ms == 0 {
            return None;
        }
        for bucket in &self.buckets {
            if bucket.is_empty() {
                continue;
            }
            let Some(first_aged) = bucket
                .iter()
                .find(|entry| now_ms.saturating_sub(entry.enqueued_at_ms) >= aging_ms)
            else {
                continue;
            };
            let policy = scheduler_policy_for_priority(first_aged.session.priority, &self.env);
            let selection_limit = self.selection_limit(&policy);
            let compatible = bucket
                .iter()
                .filter(|entry| {
                    sessions_compatible_for_prefill(&first_aged.session, &entry.session)
                })
                .take(selection_limit)
                .cloned()
                .collect::<Vec<_>>();
            return Some(self.selection(&compatible, policy));
        }
        None
    }

    fn has_queued_higher_priority(&self, priority: u8) -> bool {
        self.buckets[..priority as usize]
            .iter()
            .any(|bucket| !bucket.is_empty())
    }

    fn selection_limit(&self, policy: &SchedulerPriorityPolicy) -> usize {
        if policy.disk_spill_allowed {
            policy.max_batch_size.max(policy.spillable_batch_max)
        } else {
            policy.max_batch_size
        }
    }

    fn selection(
        &self,
        entries: &[QueuedPrefillRequest],
        policy: SchedulerPriorityPolicy,
    ) -> PrefillBatchSelection {
        let sessions = entries
            .iter()
            .map(|entry| entry.session.clone())
            .collect::<Vec<_>>();
        PrefillBatchSelection {
            total_prompt_tokens: sessions
                .iter()
                .map(|session| session.prompt_tokens.len())
                .sum(),
            total_suffix_tokens: sessions
                .iter()
                .map(|session| session.suffix_tokens.len())
                .sum(),
            max_prompt_tokens: sessions
                .iter()
                .map(|session| session.prompt_tokens.len())
                .max()
                .unwrap_or(0),
            sessions,
            policy,
        }
    }

    fn remove_selected(&mut self, sessions: &[RequestSessionDraft]) {
        for session in sessions {
            self.cancel(&session.id);
        }
    }
}

#[derive(Clone, Debug)]
pub struct PriorityDecodeScheduler {
    env: SchedulerPolicyEnv,
    buckets: Vec<Vec<ActiveDecodeSession>>,
    active_ids: HashSet<String>,
    active_count: usize,
}

impl Default for PriorityDecodeScheduler {
    fn default() -> Self {
        Self::new(SchedulerPolicyEnv::empty())
    }
}

impl PriorityDecodeScheduler {
    pub fn new(env: SchedulerPolicyEnv) -> Self {
        Self {
            env,
            buckets: (0..=255).map(|_| Vec::new()).collect(),
            active_ids: HashSet::new(),
            active_count: 0,
        }
    }

    pub fn size(&self) -> usize {
        self.active_count
    }

    pub fn has(&self, id: &str) -> bool {
        self.active_ids.contains(id)
    }

    pub fn enqueue(&mut self, session: ActiveDecodeSession) -> Result<(), String> {
        if self.active_ids.contains(&session.id) {
            return Err(format!("decode session is already active: {}", session.id));
        }
        let max_active = self.max_active_sessions();
        if max_active > 0 && self.active_count >= max_active {
            return Err(format!(
                "decode scheduler backpressure: active={} max={max_active}",
                self.active_count
            ));
        }
        let priority = session.priority as usize;
        let id = session.id.clone();
        self.buckets[priority].push(session);
        self.active_ids.insert(id);
        self.active_count += 1;
        Ok(())
    }

    pub fn cancel(&mut self, id: &str) -> bool {
        if !self.active_ids.contains(id) {
            return false;
        }
        for bucket in &mut self.buckets {
            if let Some(index) = bucket.iter().position(|session| session.id == id) {
                bucket.remove(index);
                self.active_ids.remove(id);
                self.active_count = self.active_count.saturating_sub(1);
                return true;
            }
        }
        self.active_ids.remove(id);
        self.active_count = self.active_count.saturating_sub(1);
        true
    }

    pub fn next_decode_batch(&mut self, _input: NextBatchInput) -> Option<DecodeBatchSelection> {
        for priority in 0..self.buckets.len() {
            let bucket = &self.buckets[priority];
            if bucket.is_empty() {
                continue;
            }
            let first = bucket.first()?;
            let policy = scheduler_policy_for_priority(first.priority, &self.env);
            let compatible = bucket
                .iter()
                .filter(|session| session.worker_key_id == first.worker_key_id)
                .take(policy.max_batch_size)
                .cloned()
                .collect::<Vec<_>>();
            if compatible.is_empty() {
                return None;
            }
            self.remove_selected(&compatible);
            return Some(DecodeBatchSelection {
                sessions: compatible,
                policy,
            });
        }
        None
    }

    fn max_active_sessions(&self) -> usize {
        parse_integer(self.env.get("HIPFIRE_SCHED_DECODE_MAX_ACTIVE"), 256).max(0) as usize
    }

    fn remove_selected(&mut self, sessions: &[ActiveDecodeSession]) {
        for session in sessions {
            self.cancel(&session.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> SchedulerPolicyEnv {
        SchedulerPolicyEnv::from_pairs(pairs.iter().copied())
    }

    fn qwen_worker() -> ModelWorkerKey {
        ModelWorkerKey {
            artifact_path: "/models/qwen3.6-35b-a3b-mq4.hfq".to_string(),
            artifact_digest: Some("sha256:qwen-a3b".to_string()),
            arch_id: "6".to_string(),
            quant_family: "mq4".to_string(),
            state_mode: "q8+deltanet".to_string(),
            max_seq_bucket: 4096,
            accelerator_kind: None,
            device_id: None,
            feature_flags: vec!["prefill_batch".to_string(), "qwen35".to_string()],
        }
    }

    fn nemotron_worker() -> ModelWorkerKey {
        ModelWorkerKey {
            artifact_path: "/models/nemotron-3-ultra-550b-a55b-bf16.hfq".to_string(),
            artifact_digest: Some("sha256:nemotron".to_string()),
            arch_id: "nemotron3".to_string(),
            quant_family: "bf16".to_string(),
            state_mode: "q8+mamba".to_string(),
            max_seq_bucket: 8192,
            accelerator_kind: None,
            device_id: None,
            feature_flags: vec!["mamba".to_string(), "prefill_batch".to_string()],
        }
    }

    fn session(id: &str, priority: u8, tokens: usize) -> RequestSessionDraft {
        session_with(
            id,
            priority,
            tokens,
            qwen_worker(),
            &["attention_kv", "deltanet_recurrent"],
            0,
        )
    }

    fn session_with(
        id: &str,
        priority: u8,
        tokens: usize,
        worker_key: ModelWorkerKey,
        state_kinds: &[&str],
        cached_prefix_tokens: usize,
    ) -> RequestSessionDraft {
        create_request_session_draft(CreateRequestSessionInput {
            id: id.to_string(),
            worker_key,
            prompt_tokens: (1..=tokens as u32).collect(),
            cached_prefix_tokens: Some(cached_prefix_tokens),
            priority: Some(i64::from(priority)),
            state_kinds: state_kinds.iter().map(|kind| kind.to_string()).collect(),
        })
    }

    fn ids(batch: Option<PrefillBatchSelection>) -> Vec<String> {
        batch
            .map(|batch| {
                batch
                    .sessions
                    .into_iter()
                    .map(|session| session.id)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn decode_ids(batch: Option<DecodeBatchSelection>) -> Vec<String> {
        batch
            .map(|batch| {
                batch
                    .sessions
                    .into_iter()
                    .map(|session| session.id)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn active(id: &str, worker_key_id: &str, priority: u8) -> ActiveDecodeSession {
        ActiveDecodeSession {
            id: id.to_string(),
            worker_key_id: worker_key_id.to_string(),
            priority,
            runtime_state_handle: format!("runtime-{id}"),
            logical_position: 8,
            cached_prefix_tokens: 8,
            generated_tokens: 0,
            max_tokens: 4,
        }
    }

    #[test]
    fn priority_parsing_and_classes_match_bun_policy() {
        assert_eq!(clamp_scheduler_priority(-1), 0);
        assert_eq!(clamp_scheduler_priority_f64(64.9), 64);
        assert_eq!(clamp_scheduler_priority(999), 255);
        assert_eq!(
            parse_default_scheduler_priority(&SchedulerPolicyEnv::empty()),
            64
        );
        assert_eq!(
            parse_default_scheduler_priority(&env(&[("HIPFIRE_SCHED_PRIORITY_DEFAULT", "192")])),
            192
        );
        assert_eq!(parse_scheduler_priority(Some("not-a-number"), 64), 64);
        assert_eq!(scheduler_priority_class(0).as_str(), "realtime");
        assert_eq!(scheduler_priority_class(63).as_str(), "high");
        assert_eq!(scheduler_priority_class(64).as_str(), "interactive");
        assert_eq!(scheduler_priority_class(191).as_str(), "background");
        assert_eq!(scheduler_priority_class(254).as_str(), "bulk");
        assert_eq!(scheduler_priority_class(255).as_str(), "opportunistic");
    }

    #[test]
    fn scheduler_policy_respects_waits_batch_limits_and_spill() {
        let policy = scheduler_policy_for_priority(
            64,
            &env(&[
                ("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "8"),
                ("HIPFIRE_SCHED_RESIDENT_STATE_MAX", "3"),
                ("HIPFIRE_SCHED_SPILLABLE_BATCH_MAX", "12"),
                ("HIPFIRE_SCHED_STATE_CACHE_DISK", "1"),
                ("HIPFIRE_SCHED_STATE_CACHE_DISK_MIN_PRIORITY", "64"),
                ("HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE", "7"),
            ]),
        );
        assert_eq!(policy.priority_class, SchedulerPriorityClass::Interactive);
        assert_eq!(policy.max_batch_size, 8);
        assert_eq!(policy.coalesce_wait_ms, 7);
        assert_eq!(policy.resident_state_max, 3);
        assert_eq!(policy.spillable_batch_max, 12);
        assert!(policy.disk_spill_allowed);
        assert_eq!(policy.disk_spill_min_priority, 64);

        let high =
            scheduler_policy_for_priority(1, &env(&[("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "16")]));
        assert_eq!(high.max_batch_size, 4);
        assert_eq!(high.resident_state_max, 4);
    }

    #[test]
    fn server_prefill_batch_health_is_disabled_by_default() {
        let payload = server_prefill_batch_health_json(&SchedulerPolicyEnv::empty());

        assert_eq!(payload, serde_json::json!({ "enabled": false }));
        assert!(!server_prefill_batch_enabled(&SchedulerPolicyEnv::empty()));
    }

    #[test]
    fn server_prefill_batch_health_uses_shared_scheduler_policy() {
        let payload = server_prefill_batch_health_json(&env(&[
            ("HIPFIRE_SERVER_PREFILL_BATCH", "1"),
            ("HIPFIRE_SCHED_PRIORITY_DEFAULT", "128"),
            ("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "8"),
            ("HIPFIRE_SCHED_RESIDENT_STATE_MAX", "3"),
            ("HIPFIRE_SCHED_SPILLABLE_BATCH_MAX", "12"),
            ("HIPFIRE_SCHED_STATE_CACHE_DISK", "1"),
            ("HIPFIRE_SCHED_STATE_CACHE_DISK_MIN_PRIORITY", "64"),
            ("HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND", "11"),
        ]));

        assert_eq!(payload["enabled"], true);
        assert_eq!(payload["policy"]["priority"], 128);
        assert_eq!(payload["policy"]["priority_class"], "background");
        assert_eq!(payload["policy"]["max_batch"], 8);
        assert_eq!(payload["policy"]["wait_ms"], 11);
        assert_eq!(payload["resident_state_limit"], 3);
        assert_eq!(payload["spillable_batch_max"], 12);
        assert_eq!(payload["state_cache_disk"], true);
        assert_eq!(payload["disk_spill_allowed"], true);
        assert_eq!(
            payload["generate_batch_prefill_capability_reason"],
            "rust_server_daemon_capability_not_probed"
        );
    }

    #[test]
    fn server_health_state_cache_uses_shared_scheduler_controls() {
        let payload = server_state_cache_health_json(&env(&[
            ("HIPFIRE_SERVER_PREFILL_BATCH", "true"),
            ("HIPFIRE_SERVER_PREFILL_STATE_CACHE", "1"),
            ("HIPFIRE_STATE_CACHE_MAX_CHECKPOINTS", "5"),
            ("HIPFIRE_SERVER_PREFILL_BATCH_STATE_CACHE_DISK", "1"),
        ]));

        assert_eq!(payload["enabled"], true);
        assert_eq!(payload["resident_enabled"], true);
        assert_eq!(payload["resident_checkpoint_max"], 5);
        assert_eq!(payload["disk_enabled"], true);
    }

    #[test]
    fn scheduler_policy_matches_realtime_and_interactive_bun_parity() {
        let realtime = scheduler_policy_for_priority(0, &SchedulerPolicyEnv::empty());
        let interactive = scheduler_policy_for_priority(64, &SchedulerPolicyEnv::empty());
        assert_eq!(realtime.priority_class, SchedulerPriorityClass::Realtime);
        assert_eq!(realtime.coalesce_wait_ms, 0);
        assert_eq!(realtime.max_batch_size, 1);
        assert_eq!(realtime.resident_state_max, 1);
        assert_eq!(realtime.spillable_batch_max, 1);
        assert!(realtime.max_processing_ms < interactive.max_processing_ms);

        let configured = scheduler_policy_for_priority(
            64,
            &env(&[
                ("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "16"),
                ("HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE", "7"),
            ]),
        );
        assert_eq!(
            configured.priority_class,
            SchedulerPriorityClass::Interactive
        );
        assert_eq!(configured.coalesce_wait_ms, 7);
        assert_eq!(configured.max_batch_size, 16);
        assert_eq!(configured.target_pair_tokens, 64);
    }

    #[test]
    fn scheduler_policy_matches_legacy_wait_and_opportunistic_bun_parity() {
        let legacy = env(&[("HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS", "9")]);
        let interactive = scheduler_policy_for_priority(64, &legacy);
        let background = scheduler_policy_for_priority(128, &legacy);
        assert_eq!(interactive.coalesce_wait_ms, 9);
        assert_eq!(background.coalesce_wait_ms, 18);

        let opportunistic = scheduler_policy_for_priority(
            255,
            &env(&[
                ("HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND", "20"),
                ("HIPFIRE_SCHED_OPPORTUNISTIC_MIN_PAIR_TOKENS", "512"),
            ]),
        );
        let default_background = scheduler_policy_for_priority(128, &SchedulerPolicyEnv::empty());
        assert_eq!(
            opportunistic.priority_class,
            SchedulerPriorityClass::Opportunistic
        );
        assert_eq!(opportunistic.coalesce_wait_ms, 80);
        assert_eq!(opportunistic.target_pair_tokens, 512);
        assert!(opportunistic.max_processing_ms > default_background.max_processing_ms);
    }

    #[test]
    fn scheduler_policy_matches_state_residency_and_spill_bun_parity() {
        let batch_env = env(&[("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "16")]);
        let realtime = scheduler_policy_for_priority(0, &batch_env);
        let high = scheduler_policy_for_priority(1, &batch_env);
        assert_eq!(realtime.max_batch_size, 1);
        assert_eq!(realtime.resident_state_max, 1);
        assert_eq!(realtime.spillable_batch_max, 1);
        assert_eq!(high.max_batch_size, 4);
        assert_eq!(high.resident_state_max, 4);
        assert_eq!(high.spillable_batch_max, 4);

        let disk_spill = env(&[("HIPFIRE_SCHED_STATE_CACHE_DISK", "1")]);
        assert!(!scheduler_policy_for_priority(64, &disk_spill).disk_spill_allowed);
        assert!(scheduler_policy_for_priority(128, &disk_spill).disk_spill_allowed);

        let legacy_disk_spill = env(&[("HIPFIRE_SERVER_PREFILL_BATCH_STATE_CACHE_DISK", "true")]);
        assert!(scheduler_policy_for_priority(255, &legacy_disk_spill).disk_spill_allowed);

        let clamped = scheduler_policy_for_priority(
            64,
            &env(&[
                ("HIPFIRE_SCHED_RESIDENT_STATE_MAX", "80"),
                ("HIPFIRE_SCHED_SPILLABLE_BATCH_MAX", "2"),
            ]),
        );
        assert_eq!(clamped.resident_state_max, 64);
        assert_eq!(clamped.spillable_batch_max, 64);
    }

    #[test]
    fn opportunistic_dispatch_waits_for_pairing_unless_clear() {
        assert!(!should_dispatch_opportunistic(OpportunisticDispatchInput {
            compatible_queued_tokens: 255,
            schedule_clear: false,
            target_pair_tokens: 256,
        }));
        assert!(should_dispatch_opportunistic(OpportunisticDispatchInput {
            compatible_queued_tokens: 256,
            schedule_clear: false,
            target_pair_tokens: 256,
        }));
        assert!(should_dispatch_opportunistic(OpportunisticDispatchInput {
            compatible_queued_tokens: 0,
            schedule_clear: true,
            target_pair_tokens: 256,
        }));
    }

    #[test]
    fn worker_keys_and_prefill_compatibility_match_session_policy() {
        let base = qwen_worker();
        let shuffled = ModelWorkerKey {
            feature_flags: vec!["qwen35".to_string(), "prefill_batch".to_string()],
            ..base.clone()
        };
        assert_eq!(model_worker_key_id(&base), model_worker_key_id(&shuffled));
        assert!(same_model_worker_key(&base, &shuffled));

        let a = session("a", 64, 3);
        let b = session_with(
            "b",
            64,
            2,
            shuffled,
            &["deltanet_recurrent", "attention_kv"],
            0,
        );
        let c = session_with(
            "c",
            64,
            1,
            nemotron_worker(),
            &["attention_kv", "mamba_ssm"],
            0,
        );
        assert!(sessions_compatible_for_prefill(&a, &b));
        assert!(!sessions_compatible_for_prefill(&a, &c));
    }

    #[test]
    fn prefill_scheduler_dispatches_priority_and_coalesces() {
        let mut scheduler = PriorityPrefillScheduler::new(env(&[
            ("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "3"),
            ("HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE", "5"),
        ]));
        scheduler
            .enqueue(session("interactive", 64, 16), 0)
            .unwrap();
        scheduler.enqueue(session("high", 1, 16), 0).unwrap();
        assert_eq!(
            ids(scheduler.next_prefill_batch(NextBatchInput { now_ms: 5 })),
            vec!["high"]
        );

        scheduler.enqueue(session("b", 64, 16), 2).unwrap();
        assert_eq!(
            ids(scheduler.next_prefill_batch(NextBatchInput { now_ms: 5 })),
            vec!["interactive", "b"]
        );
    }

    #[test]
    fn prefill_scheduler_respects_compatibility_spill_and_opportunistic_pairing() {
        let mut scheduler = PriorityPrefillScheduler::new(env(&[
            ("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "2"),
            ("HIPFIRE_SCHED_RESIDENT_STATE_MAX", "1"),
            ("HIPFIRE_SCHED_SPILLABLE_BATCH_MAX", "4"),
            ("HIPFIRE_SCHED_STATE_CACHE_DISK", "1"),
            ("HIPFIRE_SCHED_STATE_CACHE_DISK_MIN_PRIORITY", "128"),
            ("HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND", "0"),
        ]));
        for id in ["a", "b", "c", "d"] {
            scheduler.enqueue(session(id, 128, 16), 0).unwrap();
        }
        let batch = scheduler
            .next_prefill_batch(NextBatchInput { now_ms: 0 })
            .unwrap();
        assert_eq!(
            batch
                .sessions
                .iter()
                .map(|s| s.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b", "c", "d"]
        );
        assert_eq!(batch.policy.resident_state_max, 1);
        assert_eq!(batch.policy.spillable_batch_max, 4);

        let mut blocked = PriorityPrefillScheduler::new(env(&[
            ("HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE", "5"),
            ("HIPFIRE_SCHED_OPPORTUNISTIC_MIN_PAIR_TOKENS", "32"),
        ]));
        blocked.enqueue(session("interactive", 64, 8), 0).unwrap();
        blocked
            .enqueue(
                session_with(
                    "op-a",
                    255,
                    64,
                    qwen_worker(),
                    &["attention_kv", "deltanet_recurrent"],
                    56,
                ),
                0,
            )
            .unwrap();
        assert!(blocked
            .next_prefill_batch(NextBatchInput { now_ms: 1 })
            .is_none());
        blocked
            .enqueue(
                session_with(
                    "op-b",
                    255,
                    64,
                    qwen_worker(),
                    &["attention_kv", "deltanet_recurrent"],
                    40,
                ),
                1,
            )
            .unwrap();
        assert_eq!(
            ids(blocked.next_prefill_batch(NextBatchInput { now_ms: 5 })),
            vec!["interactive"]
        );
        let paired = blocked
            .next_prefill_batch(NextBatchInput { now_ms: 5 })
            .unwrap();
        assert_eq!(
            paired
                .sessions
                .iter()
                .map(|s| s.id.as_str())
                .collect::<Vec<_>>(),
            vec!["op-a", "op-b"]
        );
        assert_eq!(paired.total_suffix_tokens, 32);
    }

    #[test]
    fn prefill_scheduler_preview_cancel_aging_and_backpressure() {
        let mut scheduler = PriorityPrefillScheduler::new(env(&[
            ("HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE", "0"),
            ("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "2"),
        ]));
        scheduler.enqueue(session("a", 64, 16), 0).unwrap();
        let incoming = session("incoming", 64, 16);
        let preview = scheduler.preview_next_prefill_batch(PreviewPrefillBatchInput {
            now_ms: 30,
            incoming_session: Some(incoming.clone()),
            incoming_enqueued_at_ms: None,
        });
        assert_eq!(ids(preview), vec!["a", "incoming"]);
        assert_eq!(
            ids(scheduler.next_prefill_batch(NextBatchInput { now_ms: 30 })),
            vec!["a"]
        );

        assert!(scheduler.enqueue_if_absent(incoming.clone(), 10).unwrap());
        assert!(!scheduler.enqueue_if_absent(incoming, 20).unwrap());
        assert_eq!(scheduler.size(), 1);
        assert!(scheduler.cancel("incoming"));
        assert_eq!(scheduler.size(), 0);

        let mut aged = PriorityPrefillScheduler::new(env(&[
            ("HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE", "1000"),
            ("HIPFIRE_SCHED_DEADLINE_AGING_MS", "50"),
        ]));
        aged.enqueue(session("high-waiting", 32, 16), 100).unwrap();
        aged.enqueue(session("aged-low", 128, 16), 0).unwrap();
        assert_eq!(
            ids(aged.next_prefill_batch(NextBatchInput { now_ms: 60 })),
            vec!["aged-low"]
        );

        let mut capped =
            PriorityPrefillScheduler::new(env(&[("HIPFIRE_SCHED_PREFILL_MAX_QUEUED", "1")]));
        capped.enqueue(session("first", 64, 16), 0).unwrap();
        assert!(capped
            .enqueue(session("second", 64, 16), 0)
            .unwrap_err()
            .contains("backpressure"));
    }

    #[test]
    fn decode_scheduler_batches_by_worker_and_enforces_backpressure() {
        let mut scheduler =
            PriorityDecodeScheduler::new(env(&[("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "2")]));
        scheduler.enqueue(active("a", "worker-a", 64)).unwrap();
        scheduler.enqueue(active("b", "worker-a", 64)).unwrap();
        scheduler.enqueue(active("c", "worker-b", 64)).unwrap();
        assert_eq!(
            decode_ids(scheduler.next_decode_batch(NextBatchInput { now_ms: 0 })),
            vec!["a", "b"]
        );
        assert_eq!(
            decode_ids(scheduler.next_decode_batch(NextBatchInput { now_ms: 0 })),
            vec!["c"]
        );

        let mut capped =
            PriorityDecodeScheduler::new(env(&[("HIPFIRE_SCHED_DECODE_MAX_ACTIVE", "1")]));
        capped.enqueue(active("a", "worker-a", 64)).unwrap();
        assert!(capped
            .enqueue(active("b", "worker-a", 64))
            .unwrap_err()
            .contains("decode scheduler backpressure"));
    }
}
