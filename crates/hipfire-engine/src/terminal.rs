// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Terminal control plane for the daemon — single-request and continuous-batch
//! handshakes, active attempt TLS, and the two-phase `commit_ready` protocol.
//!
//! Relocated verbatim from `crates/hipfire-daemon/src/main.rs` (wave 3)
//! to break the `daemon -> loader -> daemon` cycle. No behaviour change.

use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Outcome of the two-phase client terminal handshake
/// (`commit_ready` → matching `commit` / `abort`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientTerminalDecision {
    /// Matching `commit` after readiness — producers may release tools/cache/done.
    Commit,
    /// Matching `abort`, timeout, or any fail-closed control outcome.
    Abort,
}

/// Control decision latched against the active generate transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalControlDecision {
    Abort,
    Commit,
}

/// Active generate terminal-control transaction keyed by exact
/// `(request id, attempt_id)`. The stdin reader posts matching
/// `abort` (any time) / `commit` (only after ready); producers wait
/// via [`await_client_terminal_commit`]. Terminal writers claim the
/// transaction before emitting a terminal so an abort/error race cannot
/// produce a second terminal.
pub struct ActiveTerminalControl {
    pub id: String,
    pub attempt_id: u64,
    pub ready: bool,
    pub decision: Option<TerminalControlDecision>,
    pub terminal_claimed: bool,
}
pub struct TerminalControlState {
    pub active: Option<ActiveTerminalControl>,
}

impl TerminalControlState {
    pub const fn new() -> Self {
        Self { active: None }
    }
}

pub struct TerminalControlCell {
    pub mu: Mutex<TerminalControlState>,
    pub cv: Condvar,
}

pub fn terminal_control() -> &'static TerminalControlCell {
    static CELL: OnceLock<TerminalControlCell> = OnceLock::new();
    CELL.get_or_init(|| TerminalControlCell {
        mu: Mutex::new(TerminalControlState::new()),
        cv: Condvar::new(),
    })
}

/// Bound on how long the daemon waits for a matching commit/abort after
/// emitting `commit_ready`. Timeout fails closed as Abort (never commit).
pub const CLIENT_TERMINAL_COMMIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Activate a fresh terminal-control transaction for this generate.
/// Clears any prior latch so a new request starts clean.
pub fn activate_terminal_control(id: &str, attempt_id: u64) {
    let cell = terminal_control();
    let mut g = cell.mu.lock().unwrap();
    g.active = Some(ActiveTerminalControl {
        id: id.to_string(),
        attempt_id,
        ready: false,
        decision: None,
        terminal_claimed: false,
    });
    cell.cv.notify_all();
}

/// Clear the active terminal-control transaction (request end / guard drop).
pub fn clear_terminal_control() {
    let cell = terminal_control();
    let mut g = cell.mu.lock().unwrap();
    g.active = None;
    cell.cv.notify_all();
}

/// Claim the sole terminal slot for a matching active attempt.
///
/// When no active transaction exists, returns `true` so pre-activation
/// validation errors retain their existing wire behavior (the explicit
/// uncorrelated writer `emit_uncorrelated_error` bypasses this entirely
/// with `attempt_id: 0`). When an active transaction exists, only the
/// exact `(id, attempt_id)` may claim; stale/mismatched keys return
/// `false` and do not consume the slot. Once the active exact key claims,
/// any further claim (even exact) returns `false`.
pub fn claim_terminal(id: &str, attempt_id: u64) -> bool {
    let cell = terminal_control();
    let mut g = cell.mu.lock().unwrap();
    let Some(active) = g.active.as_mut() else {
        return true;
    };
    if active.id != id || active.attempt_id != attempt_id {
        return false;
    }
    if active.terminal_claimed {
        return false;
    }
    active.terminal_claimed = true;
    true
}

/// Key for multiplexed terminal control and inbox, as required by the
/// continuous-batch contract: every lifecycle event is keyed by
/// `(id, attempt_id)` and unknown/stale keys fail closed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AttemptKey {
    pub id: String,
    pub attempt_id: u64,
}

impl AttemptKey {
    pub fn new(id: &str, attempt_id: u64) -> Self {
        Self {
            id: id.to_string(),
            attempt_id,
        }
    }
}

/// Generation-owned lane ticket. Prevents a stale control from releasing a
/// reused slot even when (id, attempt_id) would otherwise alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaneTicket {
    pub lane: usize,
    pub generation: u64,
}

// ── Keyed terminal registry ────────────────────────────────────────────

/// Registry state for a single `(id, attempt_id)`. Each variant can retain
/// an Abort; only Ready accepts Commit. Deadline is 30 s after Ready.
#[derive(Debug, Clone)]
pub enum BatchRegistryState {
    Announced,
    Queued,
    Active { owner: LaneTicket },
    Ready { owner: LaneTicket },
}

#[derive(Debug, Clone)]
pub struct BatchRegistryEntry {
    pub state: BatchRegistryState,
    pub abort_latched: bool,
    pub commit_latched: bool,
    pub pending_done: Option<serde_json::Value>,
    pub deadline: Option<Instant>,
}

pub struct BatchTerminalState {
    pub entries: std::collections::HashMap<AttemptKey, BatchRegistryEntry>,
}

impl BatchTerminalState {
    pub fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
        }
    }
}

pub struct BatchTerminalCell {
    pub mu: Mutex<BatchTerminalState>,
    pub cv: Condvar,
}

pub fn batch_terminal_control() -> &'static BatchTerminalCell {
    static CELL: OnceLock<BatchTerminalCell> = OnceLock::new();
    CELL.get_or_init(|| BatchTerminalCell {
        mu: Mutex::new(BatchTerminalState::new()),
        cv: Condvar::new(),
    })
}

/// Announce a generate key before queueing. Closes generate-then-immediate-
/// abort races for requests that arrive while GPU work is active. Returns
/// true if newly announced, false if already present.
pub fn batch_announce_terminal(id: &str, attempt_id: u64) -> bool {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    let key = AttemptKey::new(id, attempt_id);
    if g.entries.contains_key(&key) {
        return false;
    }
    g.entries.insert(
        key,
        BatchRegistryEntry {
            state: BatchRegistryState::Announced,
            abort_latched: false,
            commit_latched: false,
            pending_done: None,
            deadline: None,
        },
    );
    cell.cv.notify_all();
    true
}

/// Compatibility alias: current daemon generate arm still calls
/// `batch_activate_terminal`. Keep it as Announced insertion and do not
/// mutate the sequential singleton.
pub fn batch_activate_terminal(id: &str, attempt_id: u64) {
    batch_announce_terminal(id, attempt_id);
}

pub fn batch_transition_to_queued(id: &str, attempt_id: u64) -> bool {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    if let Some(e) = g.entries.get_mut(&AttemptKey::new(id, attempt_id)) {
        if matches!(e.state, BatchRegistryState::Announced) {
            e.state = BatchRegistryState::Queued;
            cell.cv.notify_all();
            return true;
        }
    }
    false
}

pub fn batch_bind_active(id: &str, attempt_id: u64, owner: LaneTicket) -> bool {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    if let Some(e) = g.entries.get_mut(&AttemptKey::new(id, attempt_id)) {
        if matches!(e.state, BatchRegistryState::Queued) {
            e.state = BatchRegistryState::Active { owner };
            cell.cv.notify_all();
            return true;
        }
    }
    false
}

pub fn batch_mark_ready_with_pending(
    id: &str,
    attempt_id: u64,
    owner: LaneTicket,
    pending_done: serde_json::Value,
) -> bool {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    if let Some(e) = g.entries.get_mut(&AttemptKey::new(id, attempt_id)) {
        match e.state {
            BatchRegistryState::Active { owner: o } if o == owner => {
                e.state = BatchRegistryState::Ready { owner };
                e.pending_done = Some(pending_done);
                e.deadline = Some(Instant::now() + CLIENT_TERMINAL_COMMIT_TIMEOUT);
                cell.cv.notify_all();
                return true;
            }
            _ => {}
        }
    }
    false
}

/// Legacy ready marker without payload. Transitions Active->Ready with a
/// deadline and empty pending_done. Preserved for host-only tests that do
/// not carry a full terminal payload yet.
pub fn batch_mark_ready(id: &str, attempt_id: u64) -> bool {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    if let Some(e) = g.entries.get_mut(&AttemptKey::new(id, attempt_id)) {
        match e.state {
            BatchRegistryState::Active { owner } => {
                e.state = BatchRegistryState::Ready { owner };
                e.deadline = Some(Instant::now() + CLIENT_TERMINAL_COMMIT_TIMEOUT);
                if e.pending_done.is_none() {
                    e.pending_done =
                        Some(serde_json::json!({"type":"done","id":id,"attempt_id":attempt_id}));
                }
                cell.cv.notify_all();
                return true;
            }
            _ => {}
        }
    }
    false
}

pub fn batch_clear_terminal(id: &str, attempt_id: u64) {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    g.entries.remove(&AttemptKey::new(id, attempt_id));
    cell.cv.notify_all();
}

pub fn batch_clear_all_terminals() {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    g.entries.clear();
    cell.cv.notify_all();
}

/// Apply abort/commit control. Abort latches in any state; Commit only in
/// Ready with matching owner. Stale or unknown keys are ignored and fail
/// closed elsewhere. Never mutates the sequential singleton.
pub fn batch_apply_terminal_control(kind: &str, id: &str, attempt_id: u64) {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    if let Some(e) = g.entries.get_mut(&AttemptKey::new(id, attempt_id)) {
        match kind {
            "abort" => {
                if !e.abort_latched {
                    e.abort_latched = true;
                    cell.cv.notify_all();
                }
            }
            "commit" => {
                if e.abort_latched {
                    return;
                }
                if matches!(e.state, BatchRegistryState::Ready { .. }) && !e.commit_latched {
                    e.commit_latched = true;
                    cell.cv.notify_all();
                }
            }
            _ => {}
        }
    }
}

pub fn batch_check_abort(id: &str, attempt_id: u64) -> bool {
    let cell = batch_terminal_control();
    let g = cell.mu.lock().unwrap();
    g.entries
        .get(&AttemptKey::new(id, attempt_id))
        .is_some_and(|e| e.abort_latched)
}

/// Non-mutating poll. Returns Commit only if Ready and commit latched and
/// not aborted; Abort if abort latched or deadline expired; None otherwise.
/// Never latches or mutates.
pub fn batch_poll_decision(id: &str, attempt_id: u64) -> Option<ClientTerminalDecision> {
    let cell = batch_terminal_control();
    let g = cell.mu.lock().unwrap();
    let e = g.entries.get(&AttemptKey::new(id, attempt_id))?;
    if e.abort_latched {
        return Some(ClientTerminalDecision::Abort);
    }
    if let Some(deadline) = e.deadline {
        if Instant::now() >= deadline {
            return Some(ClientTerminalDecision::Abort);
        }
    }
    if e.commit_latched {
        if matches!(e.state, BatchRegistryState::Ready { .. }) {
            return Some(ClientTerminalDecision::Commit);
        }
    }
    None
}

/// Blocking wait used by lane commit polling (30 s deadline). Unlike the
/// 5 ms host-sim poll, this waits on the condvar and respects the lane's
/// deadline. Returns Abort on timeout/expiry.
pub fn batch_wait_decision(id: &str, attempt_id: u64, timeout: Duration) -> ClientTerminalDecision {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    let deadline = Instant::now() + timeout;
    loop {
        let entry = g.entries.get(&AttemptKey::new(id, attempt_id)).cloned();
        match entry {
            None => return ClientTerminalDecision::Abort,
            Some(e) => {
                if e.abort_latched {
                    return ClientTerminalDecision::Abort;
                }
                if let Some(dl) = e.deadline {
                    if Instant::now() >= dl {
                        return ClientTerminalDecision::Abort;
                    }
                }
                if e.commit_latched && matches!(e.state, BatchRegistryState::Ready { .. }) {
                    return ClientTerminalDecision::Commit;
                }
            }
        }
        let now = Instant::now();
        if now >= deadline {
            return ClientTerminalDecision::Abort;
        }
        let remaining = deadline - now;
        let (guard, wait_res) = cell.cv.wait_timeout(g, remaining).unwrap();
        g = guard;
        if wait_res.timed_out() {
            continue;
        }
    }
}

/// If an announced request becomes a sequential barrier, transfer any
/// pre-latched Abort into the sequential singleton before invoking the
/// unchanged sequential route, then remove the keyed announcement. Early
/// Commit is ignored.
pub fn batch_transfer_abort_to_singleton_and_clear(id: &str, attempt_id: u64) -> bool {
    let had_abort = batch_check_abort(id, attempt_id);
    batch_clear_terminal(id, attempt_id);
    if had_abort {
        activate_terminal_control(id, attempt_id);
        apply_terminal_control("abort", id, attempt_id);
        return true;
    }
    false
}

/// Pure commit-teardown classifier: success `done` is allowed only after both
/// fallible GPU reset and host `commit_lane` succeed. Used by the driver and
/// covered by same-file tests so ordering cannot regress silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchCommitTeardownClass {
    /// `reset_lane` failed — use fail_all; never emit done.
    ResetFailed,
    /// Reset ok but `commit_lane` failed — fail closed for that key; no done.
    CommitFailed,
    /// Both transitions ok — emit the staged terminal done.
    EmitDone,
}

pub fn batch_commit_teardown_class(reset_ok: bool, commit_ok: bool) -> BatchCommitTeardownClass {
    if !reset_ok {
        BatchCommitTeardownClass::ResetFailed
    } else if !commit_ok {
        BatchCommitTeardownClass::CommitFailed
    } else {
        BatchCommitTeardownClass::EmitDone
    }
}

/// After the current token is committed, `seq_pos` is the next decode index.
/// A lane is at capacity when that index is no longer strictly below capacity.
pub fn batch_lane_at_capacity(seq_pos: usize, lane_capacity: usize) -> bool {
    seq_pos >= lane_capacity
}

/// Pure LFM capacity gate: `prompt_len + max_tokens` must fit strictly within
/// `lane_capacity`. Uses `saturating_add` so `u64::MAX` never wraps under the cap.
/// Returns `true` when the request exceeds capacity (must be rejected before
/// `gen_start`/GPU).
pub fn batch_lfm_exceeds_capacity(
    prompt_len: usize,
    max_tokens: usize,
    lane_capacity: usize,
) -> bool {
    prompt_len.saturating_add(max_tokens) > lane_capacity
}

/// Shared LFM admission decision: `true` when the request is valid and fits
/// within `lane_capacity` (including `max_tokens`). Invalid (empty or over-cap)
/// must emit validation/context error with no `gen_start`/GPU work.
pub fn batch_lfm_admission_ok(prompt_len: usize, max_tokens: usize, lane_capacity: usize) -> bool {
    if prompt_len == 0 {
        return false;
    }
    !batch_lfm_exceeds_capacity(prompt_len, max_tokens, lane_capacity)
}

/// Length-cap terminal when max_tokens or lane capacity is hit without a
/// competing stop cause (EOS / filter / loop guard).
pub fn batch_hit_length_cap(
    hit_max_tokens: bool,
    hit_lane_capacity: bool,
    is_eos: bool,
    stopped: bool,
    loop_hit: bool,
) -> bool {
    (hit_max_tokens || hit_lane_capacity) && !is_eos && !stopped && !loop_hit
}

pub fn batch_should_finish_decode(
    is_eos: bool,
    hit_max_tokens: bool,
    hit_lane_capacity: bool,
    stopped: bool,
    loop_hit: bool,
) -> bool {
    is_eos || hit_max_tokens || hit_lane_capacity || stopped || loop_hit
}

pub fn batch_pending_deadline(id: &str, attempt_id: u64) -> Option<Instant> {
    let cell = batch_terminal_control();
    let g = cell.mu.lock().unwrap();
    g.entries
        .get(&AttemptKey::new(id, attempt_id))
        .and_then(|e| e.deadline)
}

pub fn batch_is_ready(id: &str, attempt_id: u64) -> bool {
    let cell = batch_terminal_control();
    let g = cell.mu.lock().unwrap();
    matches!(
        g.entries.get(&AttemptKey::new(id, attempt_id)),
        Some(e) if matches!(e.state, BatchRegistryState::Ready { .. })
    )
}

// ── TLS active attempt id ───────────────────────────────────────────────

thread_local! {
    /// Active generate attempt_id for typed errors emitted during a request.
    /// Reset path parses attempt_id from the message directly.
    static ACTIVE_ATTEMPT_ID: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub fn active_attempt_id() -> u64 {
    ACTIVE_ATTEMPT_ID.with(|c| c.get())
}

pub fn set_active_attempt_id(id: u64) {
    ACTIVE_ATTEMPT_ID.with(|c| c.set(id));
}

pub struct ActiveAttemptGuard;
impl Drop for ActiveAttemptGuard {
    fn drop(&mut self) {
        set_active_attempt_id(0);
    }
}

/// Temporarily bind batch-lane emissions to their request attempt.
///
/// Continuous batching interleaves independent requests inside one outer
/// daemon command, so every lane-specific wire event must restore the
/// previously active attempt when its emission scope ends.
pub struct BatchAttemptScope {
    pub previous: u64,
}

impl BatchAttemptScope {
    pub fn enter(attempt_id: u64) -> Self {
        let previous = active_attempt_id();
        set_active_attempt_id(attempt_id);
        Self { previous }
    }
}

impl Drop for BatchAttemptScope {
    fn drop(&mut self) {
        set_active_attempt_id(self.previous);
    }
}

// ── singleton terminal control wrappers ──────────────────────────────────

/// Drop guard: clears the active terminal-control transaction.
pub struct TerminalControlGuard;
impl Drop for TerminalControlGuard {
    fn drop(&mut self) {
        clear_terminal_control();
    }
}

/// True if the in-flight request with `req_id` has been aborted for the
/// active attempt. Does not clear the latch (abort remains authoritative
/// through the rest of the turn / handshake).
pub fn check_abort(req_id: &str) -> bool {
    let cell = terminal_control();
    let g = cell.mu.lock().unwrap();
    match g.active.as_ref() {
        Some(active)
            if active.id == req_id
                && matches!(active.decision, Some(TerminalControlDecision::Abort)) =>
        {
            true
        }
        _ => false,
    }
}

/// Apply a control message from the stdin reader.
/// - `abort`: accepted throughout generation when `(id, attempt_id)` matches.
/// - `commit`: accepted only after readiness for the matching pair.
/// Stale / malformed controls are ignored without mutating state.
pub fn apply_terminal_control(kind: &str, id: &str, attempt_id: u64) {
    let cell = terminal_control();
    let mut g = cell.mu.lock().unwrap();
    let Some(active) = g.active.as_mut() else {
        return;
    };
    if active.id != id || active.attempt_id != attempt_id {
        return;
    }
    if active.decision.is_some() {
        return;
    }
    match kind {
        "abort" => {
            active.decision = Some(TerminalControlDecision::Abort);
            cell.cv.notify_all();
        }
        "commit" => {
            if active.ready {
                active.decision = Some(TerminalControlDecision::Commit);
                cell.cv.notify_all();
            }
            // Early commit before ready: ignore (must not commit).
        }
        _ => {}
    }
}

/// Mark the active transaction ready to accept `commit` for `(id, attempt)`.
/// Returns false if there is no matching active transaction.
pub fn mark_terminal_control_ready(id: &str, attempt_id: u64) -> bool {
    let cell = terminal_control();
    let mut g = cell.mu.lock().unwrap();
    match g.active.as_mut() {
        Some(active) if active.id == id && active.attempt_id == attempt_id => {
            active.ready = true;
            // Abort may already be latched; wake any waiter.
            cell.cv.notify_all();
            true
        }
        _ => false,
    }
}

/// Wait for a matching commit/abort decision on the active transaction.
/// Fail-closed: timeout or missing/mismatched active state → Abort.
pub fn wait_terminal_control_decision(
    id: &str,
    attempt_id: u64,
    timeout: Duration,
) -> ClientTerminalDecision {
    let cell = terminal_control();
    let mut g = cell.mu.lock().unwrap();
    let deadline = Instant::now() + timeout;
    loop {
        let matched = match g.active.as_ref() {
            Some(active) if active.id == id && active.attempt_id == attempt_id => {
                Some(active.decision)
            }
            // No matching active transaction: fail closed.
            _ => return ClientTerminalDecision::Abort,
        };
        if let Some(Some(decision)) = matched {
            return match decision {
                TerminalControlDecision::Commit => ClientTerminalDecision::Commit,
                TerminalControlDecision::Abort => ClientTerminalDecision::Abort,
            };
        }
        let now = Instant::now();
        if now >= deadline {
            // Timeout: latch Abort so subsequent check_abort sees it.
            if let Some(active) = g.active.as_mut() {
                if active.id == id && active.attempt_id == attempt_id && active.decision.is_none() {
                    active.decision = Some(TerminalControlDecision::Abort);
                }
            }
            cell.cv.notify_all();
            return ClientTerminalDecision::Abort;
        }
        let remaining = deadline - now;
        let (guard, wait_res) = cell.cv.wait_timeout(g, remaining).unwrap();
        g = guard;
        if wait_res.timed_out() {
            // Re-check once more under the lock before classifying timeout.
            continue;
        }
    }
}

/// Two-phase correlated terminal handshake.
///
/// After successful producer terminal classification and before tool-call
/// release / assistant-cache insertion / normal `done`:
/// 1. Mark the active `(id, attempt_id)` ready.
/// 2. Emit flushed `commit_ready` = clone of `pending_done` with only
///    `type` changed (`commit_ready`). All other fields (id, attempt_id,
///    finish_reason, usage/timing, route-specific terminals) are preserved.
/// 3. Wait for matching `commit` or `abort` (or bounded timeout → Abort).
///
/// On [`ClientTerminalDecision::Commit`] the caller must emit the same
/// `pending_done` value as `done` (payload-identical after normalizing type).
/// On Abort the caller must not emit normal done / cache / tool release.
///
/// Direct CLI generation auto-acks `commit_ready` on the engine side.
pub fn await_client_terminal_commit(
    stdout: &mut impl std::io::Write,
    id: &str,
    pending_done: &serde_json::Value,
) -> ClientTerminalDecision {
    let attempt_id = active_attempt_id();
    if !mark_terminal_control_ready(id, attempt_id) {
        return ClientTerminalDecision::Abort;
    }
    // Abort may already be latched before ready — still emit commit_ready so
    // the engine observes the handshake edge, then wait returns Abort.
    let mut envelope = pending_done.clone();
    if let Some(obj) = envelope.as_object_mut() {
        obj.insert(
            "type".to_string(),
            serde_json::Value::String("commit_ready".to_string()),
        );
    } else {
        return ClientTerminalDecision::Abort;
    }
    if writeln!(stdout, "{}", envelope).is_err() {
        return ClientTerminalDecision::Abort;
    }
    if stdout.flush().is_err() {
        return ClientTerminalDecision::Abort;
    }
    wait_terminal_control_decision(id, attempt_id, CLIENT_TERMINAL_COMMIT_TIMEOUT)
}

/// Emit a previously staged `done` envelope after Commit. Payload must be the
/// same value passed to [`await_client_terminal_commit`] as `pending_done`.
/// A matching active attempt may claim only one terminal envelope.
pub fn emit_staged_terminal_done(
    stdout: &mut impl std::io::Write,
    pending_done: &serde_json::Value,
) {
    if let Some(obj) = pending_done.as_object() {
        if let Some(id) = obj.get("id").and_then(|value| value.as_str()) {
            let attempt_id = obj
                .get("attempt_id")
                .and_then(|value| value.as_u64())
                .unwrap_or_else(active_attempt_id);
            if !claim_terminal(id, attempt_id) {
                return;
            }
        }
    }
    let _ = writeln!(stdout, "{}", pending_done);
    let _ = stdout.flush();
}

/// Force-answer target request ID, set by the stdin-reader thread on
/// `{type:"force_answer","id":"..."}`. Unlike `abort` (which kills the
/// turn), force-answer asks the decode loop to STOP THINKING and commit
/// to the answer — the model's `<think>` span is force-closed (the same
/// continuation the `max_think_tokens` budget splices) and generation
/// continues. The CLI sends this when a turn is taking too long so the
/// stream produces a real answer instead of the client timing out and
/// terminating mid-think.
pub fn force_answer_for_id() -> &'static Mutex<Option<String>> {
    static CELL: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(None))
}

/// True if the in-flight request `req_id` was asked to force-answer.
/// Clears on match (one-shot).
pub fn check_force_answer(req_id: &str) -> bool {
    let mut g = force_answer_for_id().lock().unwrap();
    if g.as_deref() == Some(req_id) {
        *g = None;
        true
    } else {
        false
    }
}

/// The text spliced into the stream to force-close a `<think>` span (on
/// either the `max_think_tokens` budget OR a CLI force-answer signal),
/// making the model commit to its answer. Default closes the think tag
/// per Qwen's trained post-think format; override with
/// `HIPFIRE_THINK_CONTINUATION` to inject a richer "now produce the
/// answer" nudge (keep it short — it's prepended to the visible answer).
pub fn think_continuation() -> String {
    std::env::var("HIPFIRE_THINK_CONTINUATION").unwrap_or_else(|_| "</think>\n\n".to_string())
}
