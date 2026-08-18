// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared semantic event and terminal carrier types for serve-hardening.
//!
//! Carriers freeze the internal vocabulary for:
//! - visible text (classifier-authorized client prose)
//! - buffered structured tool calls
//! - fail-closed malformed / truncated protocol
//! - exactly one terminal outcome per attempt
//! - attempt correlation
//!
//! Raw model token identity is preserved on a separate committed-token carrier
//! and is never treated as client-visible content.
//!
//! Minimal correlated JSONL wire value constructors live here so production
//! writers and CLI fold tests share one shape for gen_start / aborted terminals.

use crate::prompt_frame::ToolCall;
use serde::{Deserialize, Serialize};

/// Correlates all events belonging to one generation attempt.
///
/// Internal only: public completion IDs stay stable across retries; this id
/// tags attempt-local semantic traffic so stale events can be rejected later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AttemptId(u64);

impl AttemptId {
    /// Construct a new attempt id.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Raw numeric id.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Client-visible prose authorized by the runtime classifier/router.
///
/// Construction is sealed to the crate via [`VisibleText::from_classified`].
/// Fields are private and there is no public unchecked constructor from
/// arbitrary strings. Marker screening is **not** this carrier's job — the
/// classifier decides what prose is visible; authorized text may legitimately
/// quote protocol lexemes.
///
/// Runtime carriers are **emitted** (serialized) on trusted paths; they are
/// not inbound wire types. `Deserialize` is intentionally absent so external
/// crates cannot mint a [`VisibleText`] via `serde_json::from_str` (or any
/// other deserializer). Container types that embed this carrier likewise omit
/// public deserialization.
///
/// ```compile_fail
/// // External crates must not reconstruct VisibleText from arbitrary JSON.
/// let _ = serde_json::from_str::<hipfire_runtime::semantic::VisibleText>(r#""hi""#);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct VisibleText {
    text: String,
}

impl VisibleText {
    /// Build a visible-text carrier from classifier-authorized prose.
    ///
    /// Crate-private: only the runtime classifier/router boundary (and
    /// in-crate tests) may mint this type. No marker blacklist — the
    /// classifier already decided this string is client-visible.
    pub(crate) fn from_classified(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }

    /// Borrow the visible prose.
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// Consume into the owned string.
    pub fn into_string(self) -> String {
        self.text
    }
}

/// Fail-closed latch for malformed or truncated tool protocol.
///
/// When set, no buffered structured call may become executable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MalformedProtocol {
    detail: String,
}

impl MalformedProtocol {
    /// Record a human-readable fail-closed detail (not a retry class).
    pub fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }

    /// Borrow the detail string.
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

/// Why a generation attempt ended. Exactly one of these is admitted per attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalReason {
    /// Natural end without executable tool calls.
    Stop,
    /// ≥1 complete buffered call and a tool-safe end (not length/error/abort/malformed).
    ToolCalls,
    /// Hit max tokens; never tool-safe even if complete calls were buffered.
    Length,
    /// Client/server cancellation.
    Aborted,
    /// Request/runtime failure (non-protocol).
    Error,
    /// Malformed or truncated tool protocol (fail-closed).
    MalformedProtocol,
}

impl TerminalReason {
    /// Whether structured calls may be exposed as executable for this reason.
    pub const fn is_tool_safe(self) -> bool {
        matches!(self, Self::ToolCalls)
    }
}

/// Exactly-one terminal outcome for a correlated attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalOutcome {
    attempt_id: AttemptId,
    reason: TerminalReason,
}

impl TerminalOutcome {
    /// Build a terminal outcome (callers must enforce single-shot admission).
    pub fn new(attempt_id: AttemptId, reason: TerminalReason) -> Self {
        Self { attempt_id, reason }
    }

    /// Attempt this terminal belongs to.
    pub const fn attempt_id(&self) -> AttemptId {
        self.attempt_id
    }

    /// Terminal reason.
    pub const fn reason(&self) -> TerminalReason {
        self.reason
    }
}

/// Raw model token identity for KV/history/cache fidelity.
///
/// Not a client-visible content channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommittedToken {
    /// Tokenizer id as sampled/committed.
    pub id: u32,
    /// Output index within the attempt's committed stream.
    pub idx: usize,
}

/// Shared semantic event vocabulary (definitions only; no wire adapter here).
///
/// Serialize-only: events are emitted by the runtime. Public `Deserialize` is
/// omitted so embedding [`VisibleText`] cannot be bypassed by deserializing a
/// container from untrusted JSON.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SemanticEvent {
    /// Clean client-visible prose (canonical [`VisibleText`] carrier).
    VisibleText {
        text: VisibleText,
        attempt_id: AttemptId,
    },
    /// Think/reasoning channel text (also marker-free).
    Reasoning { text: String, attempt_id: AttemptId },
    /// Authoritative structured tool call (still subject to terminal gating).
    ToolCall {
        call: ToolCall,
        attempt_id: AttemptId,
    },
    /// Raw committed model token (debug / state fidelity; not content).
    Committed {
        token: CommittedToken,
        attempt_id: AttemptId,
    },
    /// Fail-closed malformed/truncated protocol notice prior to terminal latch.
    Malformed {
        detail: String,
        attempt_id: AttemptId,
    },
    /// Exactly-one terminal carrier for the attempt.
    Terminal { outcome: TerminalOutcome },
}

impl SemanticEvent {
    /// Attempt correlation on every event variant.
    pub const fn attempt_id(&self) -> AttemptId {
        match self {
            Self::VisibleText { attempt_id, .. }
            | Self::Reasoning { attempt_id, .. }
            | Self::ToolCall { attempt_id, .. }
            | Self::Committed { attempt_id, .. }
            | Self::Malformed { attempt_id, .. } => *attempt_id,
            Self::Terminal { outcome } => outcome.attempt_id(),
        }
    }
}

/// Errors from carrier mutation that must fail closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticCarrierError {
    /// A second terminal was requested after one was already latched.
    DuplicateTerminal,
    /// Further mutation after the attempt already terminated.
    AlreadyTerminated,
    /// `TerminalReason::ToolCalls` with an empty buffer.
    ToolCallsWithoutBufferedCall,
    /// `TerminalReason::MalformedProtocol` must use [`SemanticAttempt::terminate_malformed`].
    MalformedRequiresDedicatedTransition,
}

impl std::fmt::Display for SemanticCarrierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateTerminal => write!(f, "attempt already has a terminal outcome"),
            Self::AlreadyTerminated => write!(f, "attempt already terminated"),
            Self::ToolCallsWithoutBufferedCall => {
                write!(f, "tool_calls terminal requires at least one buffered call")
            }
            Self::MalformedRequiresDedicatedTransition => write!(
                f,
                "malformed_protocol terminal requires terminate_malformed with detail"
            ),
        }
    }
}

impl std::error::Error for SemanticCarrierError {}

/// Attempt-local accumulator proving the carrier invariants in unit tests.
///
/// Not a producer or OpenAI fold — only the shared state machine around
/// buffering, fail-closed malformed protocol, and single-terminal admission.
#[derive(Debug, Clone)]
pub struct SemanticAttempt {
    attempt_id: AttemptId,
    visible: VisibleText,
    buffered_calls: Vec<ToolCall>,
    committed: Vec<CommittedToken>,
    malformed: Option<MalformedProtocol>,
    terminal: Option<TerminalOutcome>,
}

impl SemanticAttempt {
    /// Open a new correlated attempt with empty buffers.
    pub fn new(attempt_id: AttemptId) -> Self {
        Self {
            attempt_id,
            // Empty prose is always classified-clean.
            visible: VisibleText {
                text: String::new(),
            },
            buffered_calls: Vec::new(),
            committed: Vec::new(),
            malformed: None,
            terminal: None,
        }
    }

    /// Attempt correlation id.
    pub const fn attempt_id(&self) -> AttemptId {
        self.attempt_id
    }

    /// Concatenated visible prose so far.
    pub fn visible_text(&self) -> &str {
        self.visible.as_str()
    }

    /// Borrow the canonical visible-text carrier.
    pub fn visible(&self) -> &VisibleText {
        &self.visible
    }

    /// Calls buffered but not yet (or never) exposed as executable.
    pub fn buffered_tool_calls(&self) -> &[ToolCall] {
        &self.buffered_calls
    }

    /// Raw committed tokens (model state fidelity).
    pub fn committed_tokens(&self) -> &[CommittedToken] {
        &self.committed
    }

    /// Malformed latch, if any.
    pub fn malformed(&self) -> Option<&MalformedProtocol> {
        self.malformed.as_ref()
    }

    /// The single terminal outcome, if latched.
    pub fn terminal(&self) -> Option<&TerminalOutcome> {
        self.terminal.as_ref()
    }

    /// Executable calls: only after a tool-safe terminal, never on fail-closed paths.
    pub fn executable_tool_calls(&self) -> &[ToolCall] {
        match &self.terminal {
            Some(t) if t.reason().is_tool_safe() && self.malformed.is_none() => {
                &self.buffered_calls
            }
            _ => &[],
        }
    }

    fn ensure_open(&self) -> Result<(), SemanticCarrierError> {
        if self.terminal.is_some() {
            Err(SemanticCarrierError::AlreadyTerminated)
        } else {
            Ok(())
        }
    }

    /// Append already-classified visible prose.
    pub fn push_visible_text(&mut self, text: VisibleText) -> Result<(), SemanticCarrierError> {
        self.ensure_open()?;
        self.visible.text.push_str(text.as_str());
        Ok(())
    }

    /// Buffer a complete structured tool call until terminal gating.
    pub fn buffer_tool_call(&mut self, call: ToolCall) -> Result<(), SemanticCarrierError> {
        self.ensure_open()?;
        if self.malformed.is_some() {
            // Fail-closed: once malformed is latched, further calls stay non-executable
            // and are not retained as candidates.
            return Ok(());
        }
        self.buffered_calls.push(call);
        Ok(())
    }

    /// Preserve a raw model token id (separate from visible text).
    pub fn push_committed_token(
        &mut self,
        id: u32,
        idx: usize,
    ) -> Result<(), SemanticCarrierError> {
        self.ensure_open()?;
        self.committed.push(CommittedToken { id, idx });
        Ok(())
    }

    /// Latch malformed/truncated protocol and emit the single terminal.
    pub fn terminate_malformed(
        &mut self,
        detail: impl Into<String>,
    ) -> Result<&TerminalOutcome, SemanticCarrierError> {
        self.ensure_open()?;
        self.malformed = Some(MalformedProtocol::new(detail));
        // Drop any previously buffered calls from the executable path by
        // keeping them non-tool-safe via the MalformedProtocol reason.
        self.terminal = Some(TerminalOutcome::new(
            self.attempt_id,
            TerminalReason::MalformedProtocol,
        ));
        Ok(self.terminal.as_ref().expect("just set"))
    }

    /// Admit exactly one non-malformed terminal reason.
    ///
    /// `TerminalReason::MalformedProtocol` is rejected here — use
    /// [`Self::terminate_malformed`] so the fail-closed detail latch and
    /// terminal are created together.
    pub fn terminate(
        &mut self,
        reason: TerminalReason,
    ) -> Result<&TerminalOutcome, SemanticCarrierError> {
        if self.terminal.is_some() {
            return Err(SemanticCarrierError::DuplicateTerminal);
        }
        if matches!(reason, TerminalReason::MalformedProtocol) {
            return Err(SemanticCarrierError::MalformedRequiresDedicatedTransition);
        }
        if matches!(reason, TerminalReason::ToolCalls) && self.buffered_calls.is_empty() {
            return Err(SemanticCarrierError::ToolCallsWithoutBufferedCall);
        }
        // A prior malformed latch (if ever exposed without terminal) still
        // forces the dedicated transition rather than a bare reason swap.
        if self.malformed.is_some() {
            return Err(SemanticCarrierError::MalformedRequiresDedicatedTransition);
        }
        self.terminal = Some(TerminalOutcome::new(self.attempt_id, reason));
        Ok(self.terminal.as_ref().expect("just set"))
    }
}

// --- Minimal correlated JSONL wire value constructors ----------------------
//
// Production daemon writers and hipfire-cli SemanticEventFold tests must use
// these exact helpers so gen_start / aborted correlation cannot drift.

/// Build a correlated `gen_start` envelope value.
pub fn wire_gen_start(
    id: &str,
    started_in_think: bool,
    attempt_id: u64,
    contract_version: Option<u32>,
) -> serde_json::Value {
    let mut envelope = serde_json::json!({
        "type": "gen_start",
        "id": id,
        "started_in_think": started_in_think,
        "attempt_id": attempt_id,
    });
    if let Some(v) = contract_version {
        envelope["contract_version"] = serde_json::json!(v);
    }
    envelope
}

/// Build a correlated mid-stream `aborted` control envelope.
pub fn wire_aborted(id: &str, reason: &str, attempt_id: u64) -> serde_json::Value {
    serde_json::json!({
        "type": "aborted",
        "id": id,
        "reason": reason,
        "attempt_id": attempt_id,
    })
}

/// Build a correlated `done` envelope with `finish_reason: "aborted"`.
///
/// Field set matches the Qwen AR cancellation writer (prompt/decode timings
/// zeroed; completion_tokens carries tokens generated before cancel).
pub fn wire_aborted_done(id: &str, completion_tokens: usize, attempt_id: u64) -> serde_json::Value {
    serde_json::json!({
        "type": "done",
        "id": id,
        "finish_reason": "aborted",
        "prompt_tokens": 0,
        "completion_tokens": completion_tokens,
        "prefill_ms": 0,
        "decode_ms": 0,
        "attempt_id": attempt_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt_frame::ToolCall;
    use serde_json::json;

    fn sample_call(name: &str) -> ToolCall {
        ToolCall {
            id: None,
            name: name.to_string(),
            arguments: json!({"path": "/tmp/x"}),
            rendered_body: None,
        }
    }

    #[test]
    fn visible_text_carrier_holds_prose_only() {
        let v = VisibleText::from_classified("hello world");
        assert_eq!(v.as_str(), "hello world");
    }

    #[test]
    fn structured_tool_call_carrier_buffers_until_terminal() {
        let mut attempt = SemanticAttempt::new(AttemptId::new(1));
        attempt
            .push_visible_text(VisibleText::from_classified("Before tools."))
            .expect("visible");
        attempt
            .buffer_tool_call(sample_call("read"))
            .expect("buffer");
        // Pre-terminal: no executable exposure.
        assert!(attempt.executable_tool_calls().is_empty());
        assert_eq!(attempt.buffered_tool_calls().len(), 1);

        let term = attempt
            .terminate(TerminalReason::ToolCalls)
            .expect("tool-safe terminal");
        assert_eq!(term.reason(), TerminalReason::ToolCalls);
        assert_eq!(attempt.executable_tool_calls().len(), 1);
        assert_eq!(attempt.executable_tool_calls()[0].name, "read");
    }

    #[test]
    fn malformed_protocol_is_fail_closed_no_executable_calls() {
        let mut attempt = SemanticAttempt::new(AttemptId::new(7));
        attempt
            .push_visible_text(VisibleText::from_classified("partial "))
            .expect("visible");
        attempt
            .buffer_tool_call(sample_call("bash"))
            .expect("buffer before malformed latch");

        let term = attempt
            .terminate_malformed("unclosed tool_call span")
            .expect("malformed terminal");
        assert_eq!(term.reason(), TerminalReason::MalformedProtocol);
        assert!(attempt.executable_tool_calls().is_empty());
        // Buffered calls must not leak as executable after fail-closed terminal.
        assert!(attempt.executable_tool_calls().is_empty());
        assert_eq!(
            attempt.malformed().map(|m| m.detail()),
            Some("unclosed tool_call span")
        );
    }

    #[test]
    fn truncated_or_length_terminal_exposes_no_executable_calls() {
        let mut attempt = SemanticAttempt::new(AttemptId::new(2));
        attempt
            .buffer_tool_call(sample_call("write"))
            .expect("complete call buffered before cap");
        let term = attempt
            .terminate(TerminalReason::Length)
            .expect("length terminal");
        assert_eq!(term.reason(), TerminalReason::Length);
        assert!(attempt.executable_tool_calls().is_empty());
    }

    #[test]
    fn exactly_one_terminal_outcome_per_attempt() {
        let mut attempt = SemanticAttempt::new(AttemptId::new(3));
        attempt
            .terminate(TerminalReason::Stop)
            .expect("first terminal");
        let err = attempt
            .terminate(TerminalReason::ToolCalls)
            .expect_err("second terminal must fail closed");
        assert!(matches!(err, SemanticCarrierError::DuplicateTerminal));
        assert_eq!(
            attempt.terminal().map(|t| t.reason()),
            Some(TerminalReason::Stop)
        );
    }

    #[test]
    fn attempt_correlation_carrier_tags_terminal() {
        let id = AttemptId::new(42);
        let mut attempt = SemanticAttempt::new(id);
        let term = attempt
            .terminate(TerminalReason::Aborted)
            .expect("abort terminal");
        assert_eq!(term.attempt_id(), id);
        assert_eq!(attempt.attempt_id(), id);
    }

    #[test]
    fn committed_raw_tokens_are_preserved_separately_from_visible_text() {
        let mut attempt = SemanticAttempt::new(AttemptId::new(9));
        // Raw model token stream (including what may later map to markers).
        attempt
            .push_committed_token(/*id=*/ 1001, /*idx=*/ 0)
            .expect("raw commit");
        attempt
            .push_committed_token(/*id=*/ 1002, /*idx=*/ 1)
            .expect("raw commit");
        // Client-visible channel stays clean.
        attempt
            .push_visible_text(VisibleText::from_classified("only prose"))
            .expect("visible");
        assert_eq!(attempt.visible_text(), "only prose");
        let committed = attempt.committed_tokens();
        assert_eq!(committed.len(), 2);
        assert_eq!(committed[0], CommittedToken { id: 1001, idx: 0 });
        assert_eq!(committed[1], CommittedToken { id: 1002, idx: 1 });
    }

    #[test]
    fn error_and_aborted_terminals_are_not_tool_safe() {
        for reason in [TerminalReason::Error, TerminalReason::Aborted] {
            let mut attempt = SemanticAttempt::new(AttemptId::new(11));
            attempt
                .buffer_tool_call(sample_call("read"))
                .expect("buffer");
            attempt.terminate(reason).expect("terminal");
            assert!(!reason.is_tool_safe(), "{reason:?} must not be tool-safe");
            assert!(attempt.executable_tool_calls().is_empty());
        }
    }

    #[test]
    fn tool_calls_terminal_requires_at_least_one_buffered_call() {
        let mut attempt = SemanticAttempt::new(AttemptId::new(5));
        let err = attempt
            .terminate(TerminalReason::ToolCalls)
            .expect_err("empty tool_calls terminal");
        assert!(matches!(
            err,
            SemanticCarrierError::ToolCallsWithoutBufferedCall
        ));
        assert!(attempt.terminal().is_none());
    }

    #[test]
    fn carriers_serde_roundtrip_snake_case() {
        let id = AttemptId::new(3);
        let v = serde_json::to_value(id).unwrap();
        assert_eq!(v, json!(3));
        let back: AttemptId = serde_json::from_value(v).unwrap();
        assert_eq!(back, id);

        let reason = TerminalReason::MalformedProtocol;
        let s = serde_json::to_string(&reason).unwrap();
        assert_eq!(s, "\"malformed_protocol\"");
        let back: TerminalReason = serde_json::from_str(&s).unwrap();
        assert_eq!(back, reason);

        // VisibleText / SemanticEvent are emit-only (Serialize). Public
        // Deserialize is intentionally absent — see compile_fail doctest on
        // VisibleText and visible_text_has_no_public_deserialize_path.
        let ev = SemanticEvent::VisibleText {
            text: VisibleText::from_classified("hi"),
            attempt_id: id,
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["type"], "visible_text");
        assert_eq!(v["text"], "hi");
        assert_eq!(v["attempt_id"], 3);
    }

    #[test]
    fn classifier_authorized_visible_text_serializes_on_wire() {
        // Classifier-authorized prose (including quoted marker lexemes) still
        // serializes as a plain JSON string for trusted emit paths.
        let prose = r#"literal "<tool_call>" is documentation, not a call"#;
        let v = VisibleText::from_classified(prose);
        assert_eq!(
            serde_json::to_value(&v).unwrap(),
            json!(prose),
            "transparent Serialize must emit the authorized prose"
        );

        let ev = SemanticEvent::VisibleText {
            text: VisibleText::from_classified(prose),
            attempt_id: AttemptId::new(19),
        };
        let wire = serde_json::to_value(&ev).unwrap();
        assert_eq!(wire["type"], "visible_text");
        assert_eq!(wire["text"], prose);
        assert_eq!(wire["attempt_id"], 19);
    }

    #[test]
    fn visible_text_has_no_public_deserialize_path() {
        // Emit-path proof complementary to the compile_fail doctest on
        // VisibleText: Serialize remains available for trusted wire output.
        // Public Deserialize is absent (removed from the derive list); external
        // crates cannot mint via serde_json::from_str (see doctest).
        fn assert_serialize<T: serde::Serialize>() {}
        assert_serialize::<VisibleText>();
        assert_serialize::<SemanticEvent>();

        // Sealed construction + serialize remains the only public mint→wire path.
        let v = VisibleText::from_classified("authorized emit");
        let s = serde_json::to_string(&v).expect("serialize");
        assert_eq!(s, "\"authorized emit\"");
    }

    #[test]
    fn malformed_terminal_must_use_fail_closed_transition() {
        // Generic terminate must not admit MalformedProtocol without detail.
        let mut attempt = SemanticAttempt::new(AttemptId::new(13));
        attempt
            .buffer_tool_call(sample_call("read"))
            .expect("buffer");
        let err = attempt
            .terminate(TerminalReason::MalformedProtocol)
            .expect_err("malformed must go through terminate_malformed");
        assert!(matches!(
            err,
            SemanticCarrierError::MalformedRequiresDedicatedTransition
        ));
        assert!(attempt.terminal().is_none());
        assert!(attempt.malformed().is_none());
        assert!(attempt.executable_tool_calls().is_empty());

        // Only the dedicated transition latches detail + fail-closed terminal.
        let term = attempt
            .terminate_malformed("truncated tool_call")
            .expect("dedicated malformed path");
        assert_eq!(term.reason(), TerminalReason::MalformedProtocol);
        assert_eq!(
            attempt.malformed().map(|m| m.detail()),
            Some("truncated tool_call")
        );
        assert!(attempt.executable_tool_calls().is_empty());
    }

    #[test]
    fn classifier_authorized_visible_text_may_include_marker_lexemes() {
        // Marker screening is not the carrier's job: once the classifier/router
        // authorizes prose, quoted protocol lexemes are legitimate client text.
        let prose = r#"See the docs: a literal "<tool_call>" tag is not an invocation."#;
        let v = VisibleText::from_classified(prose);
        assert_eq!(v.as_str(), prose);
        assert!(v.as_str().contains("<tool_call>"));

        let mut attempt = SemanticAttempt::new(AttemptId::new(17));
        attempt
            .push_visible_text(v)
            .expect("classifier-authorized visible");
        assert_eq!(attempt.visible_text(), prose);

        // Event vocabulary carries the same sealed VisibleText type.
        let ev = SemanticEvent::VisibleText {
            text: VisibleText::from_classified(prose),
            attempt_id: AttemptId::new(17),
        };
        match &ev {
            SemanticEvent::VisibleText { text, .. } => {
                assert_eq!(text.as_str(), prose);
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn visible_text_construction_is_crate_private() {
        // Construction is sealed at the crate boundary via `pub(crate)
        // from_classified`. Same-crate tests may call it; external crates
        // cannot (fields are private; no public new/try_from/From<&str> path).
        // Compile-time seal: the only constructor is crate-private.
        let v = VisibleText::from_classified(String::from("authorized"));
        assert_eq!(v.as_str(), "authorized");

        // No public marker-blacklist error path remains on the carrier —
        // classifier-authorized strings with protocol lexemes are accepted.
        let markers_ok = VisibleText::from_classified("<function=demo>");
        assert_eq!(markers_ok.as_str(), "<function=demo>");

        // Public surface is read-only accessors + Serialize emit, not minting
        // and not public Deserialize.
        assert_eq!(v.clone().into_string(), "authorized");
        assert_eq!(serde_json::to_value(&v).unwrap(), json!("authorized"));
    }

    #[test]
    fn wire_gen_start_and_aborted_helpers_are_correlated() {
        let gs = wire_gen_start("req", true, 7, Some(2));
        assert_eq!(gs["type"], "gen_start");
        assert_eq!(gs["id"], "req");
        assert_eq!(gs["started_in_think"], true);
        assert_eq!(gs["attempt_id"], 7);
        assert_eq!(gs["contract_version"], 2);

        let ab = wire_aborted("req", "client_cancelled", 7);
        assert_eq!(ab["type"], "aborted");
        assert_eq!(ab["reason"], "client_cancelled");
        assert_eq!(ab["attempt_id"], 7);

        let done = wire_aborted_done("req", 3, 7);
        assert_eq!(done["type"], "done");
        assert_eq!(done["finish_reason"], "aborted");
        assert_eq!(done["completion_tokens"], 3);
        assert_eq!(done["attempt_id"], 7);
        assert_eq!(done["prompt_tokens"], 0);
    }
}
