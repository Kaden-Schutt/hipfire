// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The single sealed transcript authority for a speculative assistant turn.
//!
//! Raw token bytes are quarantined before they are projected to UTF-8 text.
//! This is deliberately a small ownership type: callers may render the bytes
//! returned by [`OpenAssistantTurn::observe`], but cache/replay decisions are
//! made only from the consuming [`OpenAssistantTurn::seal`] result.

use crate::emit_text::extract_tool_calls_from_text;
use crate::prompt_frame::ToolCall;
use crate::stop_quarantine::{QuarantineOutcome, StopQuarantine};

/// The byte extent of one committed token in the raw decoded stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TokenByteSpan {
    pub token: u32,
    pub start: usize,
    pub end: usize,
}

/// Newly-safe visible bytes produced by one token, and whether that token
/// found a stop marker. After a stop, subsequent observations are inert.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AssistantTurnDelta {
    pub bytes: Vec<u8>,
    pub stopped: bool,
}

/// The owned, lossless decomposition of a finalized assistant turn.
#[derive(Debug, Clone)]
pub struct FinalizedAssistantTurnParts {
    text: String,
    tool_calls: Vec<ToolCall>,
    replay_tokens: Option<Vec<u32>>,
    diagnostic_tokens: Vec<u32>,
    terminal_delta: AssistantTurnDelta,
}

impl FinalizedAssistantTurnParts {
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    pub fn replay_tokens(&self) -> Option<&[u32]> {
        self.replay_tokens.as_deref()
    }

    pub fn diagnostic_tokens(&self) -> &[u32] {
        &self.diagnostic_tokens
    }

    pub fn terminal_delta(&self) -> &AssistantTurnDelta {
        &self.terminal_delta
    }
}

/// The immutable, consuming result of an assistant turn.
#[derive(Debug, Clone)]
pub struct FinalizedAssistantTurn {
    text: String,
    tool_calls: Vec<ToolCall>,
    /// Some only when the safe byte boundary is exactly a whole-token prefix.
    replay_tokens: Option<Vec<u32>>,
    /// Token IDs whose complete raw byte spans are in the sealed safe prefix.
    diagnostic_tokens: Vec<u32>,
    /// Visible bytes released only by the consuming seal (for example a false
    /// trailing `<think>` prefix that was held during streaming).
    terminal_delta: AssistantTurnDelta,
}

impl FinalizedAssistantTurn {
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    pub fn replay_tokens(&self) -> Option<&[u32]> {
        self.replay_tokens.as_deref()
    }

    pub fn diagnostic_tokens(&self) -> &[u32] {
        &self.diagnostic_tokens
    }

    pub fn terminal_delta(&self) -> &AssistantTurnDelta {
        &self.terminal_delta
    }

    /// Consume the finalized turn without losing any sealed bytes or metadata.
    pub fn into_parts(self) -> FinalizedAssistantTurnParts {
        FinalizedAssistantTurnParts {
            text: self.text,
            tool_calls: self.tool_calls,
            replay_tokens: self.replay_tokens,
            diagnostic_tokens: self.diagnostic_tokens,
            terminal_delta: self.terminal_delta,
        }
    }
}

#[derive(Debug)]
struct VisibleProjector {
    pending: Vec<u8>,
    visible: Vec<u8>,
    reasoning_open: bool,
}

impl VisibleProjector {
    fn new(reasoning_open: bool) -> Self {
        Self {
            pending: Vec::new(),
            visible: Vec::new(),
            reasoning_open,
        }
    }

    fn push(&mut self, bytes: &[u8], flush_ambiguous: bool) -> Vec<u8> {
        let visible_start = self.visible.len();
        self.pending.extend_from_slice(bytes);
        self.project(flush_ambiguous);
        self.visible_delta_since(visible_start)
    }

    fn visible_delta_since(&self, start: usize) -> Vec<u8> {
        assert!(
            start <= self.visible.len(),
            "visible projection moved backwards"
        );
        self.visible[start..].to_vec()
    }

    fn project(&mut self, flush_ambiguous: bool) {
        loop {
            let text =
                std::str::from_utf8(&self.pending).expect("projector input must be valid UTF-8");
            if self.reasoning_open {
                if let Some(close) = text.find("</think>") {
                    self.pending.drain(..close + "</think>".len());
                    self.reasoning_open = false;
                    continue;
                }
                let held = trailing_marker_prefix_len(text, "</think>");
                self.pending.drain(..text.len() - held);
                return;
            }

            if let Some(open) = text.find("<think>") {
                self.visible.extend_from_slice(&self.pending[..open]);
                self.pending.drain(..open + "<think>".len());
                self.reasoning_open = true;
                continue;
            }

            let held = trailing_marker_prefix_len(text, "<think>")
                .max(trailing_marker_prefix_len(text, "</think>"));
            let visible_len = if flush_ambiguous {
                text.len()
            } else {
                text.len() - held
            };
            self.visible.extend_from_slice(&self.pending[..visible_len]);
            self.pending.drain(..visible_len);
            return;
        }
    }
}

/// Owns all mutable state needed to construct one sealed assistant turn.
///
/// In particular, the stop quarantine, raw token spans, and canonical safe
/// bytes cannot be reconstructed independently by downstream consumers.
#[derive(Debug)]
pub struct OpenAssistantTurn {
    quarantine: StopQuarantine,
    token_spans: Vec<TokenByteSpan>,
    /// Raw bytes proven not to contain a stop marker. This is deliberately
    /// separate from the downstream valid-UTF-8 and visible projections.
    safe_raw_bytes: Vec<u8>,
    /// Bytes in `safe_raw_bytes` through the maximal contiguous valid UTF-8
    /// boundary. Bytes after this watermark are not part of the sealed turn.
    valid_raw_len: usize,
    projector: VisibleProjector,
    raw_len: usize,
    /// A user stop marker cut the raw byte stream. This terminal state owns
    /// the discard semantics: its unresolved quarantine suffix must not be
    /// recovered during sealing.
    byte_stopped: bool,
    /// A protocol/semantic terminal token ended generation. Unlike a byte
    /// stop, this must still let seal() finish the user-stop quarantine so a
    /// false trailing prefix is retained in the transcript.
    semantic_stopped: bool,
}

impl OpenAssistantTurn {
    /// Start a turn with literal stop markers. Empty markers are ignored.
    pub fn new<I, M>(markers: I) -> Self
    where
        I: IntoIterator<Item = M>,
        M: AsRef<[u8]>,
    {
        Self {
            quarantine: StopQuarantine::new(
                markers
                    .into_iter()
                    .map(|marker| marker.as_ref().to_vec())
                    .collect(),
            ),
            token_spans: Vec::new(),
            safe_raw_bytes: Vec::new(),
            valid_raw_len: 0,
            projector: VisibleProjector::new(false),
            raw_len: 0,
            byte_stopped: false,
            semantic_stopped: false,
        }
    }

    /// Start a turn whose assistant prefix already opened hidden reasoning.
    pub fn new_with_reasoning_open<I, M>(markers: I, reasoning_open: bool) -> Self
    where
        I: IntoIterator<Item = M>,
        M: AsRef<[u8]>,
    {
        let mut turn = Self::new(markers);
        turn.projector.reasoning_open = reasoning_open;
        turn
    }

    /// Observe one committed token's raw decoded bytes.
    pub fn observe(&mut self, token: u32, raw: &[u8]) -> AssistantTurnDelta {
        if self.stopped() {
            return AssistantTurnDelta {
                bytes: Vec::new(),
                stopped: true,
            };
        }

        let start = self.raw_len;
        self.raw_len += raw.len();
        self.token_spans.push(TokenByteSpan {
            token,
            start,
            end: self.raw_len,
        });

        let outcome = self.quarantine.push(raw);
        let (safe, stopped) = match outcome {
            QuarantineOutcome::Continue { bytes } => (bytes, false),
            QuarantineOutcome::Stop { bytes } => (bytes, true),
        };
        self.safe_raw_bytes.extend_from_slice(&safe);
        let mut delta = self.refresh_projection(false);
        if stopped {
            self.byte_stopped = true;
        }
        delta.stopped = stopped;
        delta
    }

    /// The maximal contiguous valid-UTF-8 prefix of the raw safe watermark.
    #[cfg(test)]
    fn canonical_safe_bytes(&self) -> &[u8] {
        &self.safe_raw_bytes[..self.valid_raw_len]
    }

    pub fn stopped(&self) -> bool {
        self.byte_stopped || self.semantic_stopped
    }

    /// Mark the turn semantically stopped without admitting another token.
    /// Semantic EOS belongs to the transcript owner, so its decoded marker
    /// must never pass through the visible projector. Unlike a user stop
    /// marker, this does not discard the quarantine's unresolved suffix.
    pub fn stop(&mut self) {
        self.semantic_stopped = true;
    }

    /// Consume the open turn without producing a result.
    ///
    /// Dropping an open turn has the same semantics; this named operation is
    /// useful at explicit abort/error branches and makes the no-seal contract
    /// visible at call sites.
    pub fn abort(self) {}

    /// Consume the turn and publish its one canonical result.
    pub fn seal(mut self) -> FinalizedAssistantTurn {
        if !self.byte_stopped {
            let safe = self.quarantine.finish();
            self.safe_raw_bytes.extend_from_slice(&safe);
        }
        let terminal_delta = self.refresh_projection(true);

        // A malformed interior byte ends the sealed cut; bytes after it are
        // not recoverable by dropping the byte and continuing. This keeps
        // diagnostics and replay on the same contiguous projection as text.
        let diagnostic_tokens: Vec<u32> = self
            .token_spans
            .iter()
            .filter(|span| span.end <= self.valid_raw_len)
            .map(|span| span.token)
            .collect();
        let whole_token_boundary = self.valid_raw_len == 0
            || self
                .token_spans
                .iter()
                .any(|span| span.end == self.valid_raw_len);
        let replay_tokens =
            if whole_token_boundary && self.valid_raw_len == self.safe_raw_bytes.len() {
                Some(diagnostic_tokens.clone())
            } else {
                None
            };
        let stopped = self.stopped();
        let text = String::from_utf8(self.projector.visible)
            .expect("OpenAssistantTurn only stores valid UTF-8 visible bytes");
        let tool_calls = extract_tool_calls_from_text(&text);
        FinalizedAssistantTurn {
            text,
            tool_calls,
            replay_tokens,
            diagnostic_tokens,
            terminal_delta: AssistantTurnDelta {
                bytes: terminal_delta.bytes,
                stopped,
            },
        }
    }

    fn refresh_projection(&mut self, flush_ambiguous: bool) -> AssistantTurnDelta {
        let next_valid_raw_len = valid_utf8_prefix_len(&self.safe_raw_bytes);
        assert!(
            next_valid_raw_len >= self.valid_raw_len,
            "valid UTF-8 watermark moved backwards"
        );
        let new_bytes = &self.safe_raw_bytes[self.valid_raw_len..next_valid_raw_len];
        self.valid_raw_len = next_valid_raw_len;
        AssistantTurnDelta {
            bytes: self.projector.push(new_bytes, flush_ambiguous),
            stopped: false,
        }
    }
}

fn valid_utf8_prefix_len(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(error) => error.valid_up_to(),
    }
}

fn trailing_marker_prefix_len(text: &str, marker: &str) -> usize {
    let max = text.len().min(marker.len().saturating_sub(1));
    (1..=max)
        .rev()
        .find(|&len| text.as_bytes().ends_with(&marker.as_bytes()[..len]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn() -> OpenAssistantTurn {
        OpenAssistantTurn::new([b"<stop>".as_slice()])
    }

    #[test]
    fn marker_within_token_seals_only_the_safe_prefix() {
        let mut turn = turn();
        let delta = turn.observe(7, b"safe<stop>tail");
        assert_eq!(delta.bytes, b"safe");
        assert!(delta.stopped);
        let final_turn = turn.seal();
        assert_eq!(final_turn.text(), "safe");
        assert_eq!(final_turn.replay_tokens(), None);
        assert!(final_turn.diagnostic_tokens().is_empty());
    }

    #[test]
    fn marker_spanning_tokens_never_becomes_visible() {
        let mut turn = turn();
        assert_eq!(turn.observe(1, b"safe<st").bytes, b"safe");
        let delta = turn.observe(2, b"op>tail");
        assert!(delta.stopped);
        assert!(delta.bytes.is_empty());
        let final_turn = turn.seal();
        assert_eq!(final_turn.text(), "safe");
        assert!(final_turn.replay_tokens().is_none());
        assert_eq!(final_turn.diagnostic_tokens(), &[] as &[u32]);
    }

    #[test]
    fn utf8_boundary_is_completed_across_tokens() {
        let mut turn = turn();
        assert_eq!(turn.observe(10, b"caf\xc3").bytes, b"caf");
        assert_eq!(turn.observe(11, b"\xa9").bytes, "é".as_bytes());
        let final_turn = turn.seal();
        assert_eq!(final_turn.text(), "café");
        assert_eq!(final_turn.replay_tokens(), Some([10, 11].as_slice()));
    }

    #[test]
    fn false_prefix_is_flushed_at_terminal_seal() {
        let mut turn = turn();
        let mut streamed = turn.observe(1, b"safe<sto").bytes;
        let final_turn = turn.seal();
        streamed.extend_from_slice(&final_turn.terminal_delta().bytes);
        assert_eq!(String::from_utf8(streamed).unwrap(), final_turn.text());
        assert_eq!(final_turn.terminal_delta().bytes, b"<sto");
        assert_eq!(final_turn.text(), "safe<sto");
        assert_eq!(final_turn.replay_tokens(), Some([1].as_slice()));
    }

    #[test]
    fn semantic_stop_flushes_trailing_user_stop_prefix() {
        let mut turn = turn();
        assert_eq!(turn.observe(1, b"safe<sto").bytes, b"safe");
        turn.stop();
        assert!(turn.stopped());
        assert!(!turn.byte_stopped);

        let final_turn = turn.seal();
        assert_eq!(final_turn.text(), "safe<sto");
        assert_eq!(final_turn.terminal_delta().bytes, b"<sto");
        assert_eq!(final_turn.replay_tokens(), Some([1].as_slice()));
    }

    #[test]
    fn exact_whole_token_seal_replays_exactly() {
        let mut turn = turn();
        turn.observe(21, b"hello ");
        turn.observe(22, b"world");
        let final_turn = turn.seal();
        assert_eq!(final_turn.replay_tokens(), Some([21, 22].as_slice()));
        assert_eq!(final_turn.diagnostic_tokens(), [21, 22].as_slice());
    }

    #[test]
    fn observe_remains_the_incremental_output_boundary() {
        let mut turn = turn();
        assert_eq!(turn.observe(25, b"hello ").bytes, b"hello ");
        assert_eq!(turn.observe(26, b"world").bytes, b"world");

        let final_turn = turn.seal();
        assert_eq!(final_turn.text(), "hello world");
        assert_eq!(final_turn.replay_tokens(), Some([25, 26].as_slice()));
    }

    #[test]
    fn intra_token_cut_preserves_text_but_is_not_replayable() {
        let mut turn = turn();
        turn.observe(31, b"hello<stop>");
        let final_turn = turn.seal();
        assert_eq!(final_turn.text(), "hello");
        assert_eq!(final_turn.replay_tokens(), None);
    }

    #[test]
    fn post_stop_observations_are_inert() {
        let mut turn = turn();
        turn.observe(40, b"ok<stop>");
        for token in 41..45 {
            assert_eq!(
                turn.observe(token, b"after"),
                AssistantTurnDelta {
                    bytes: Vec::new(),
                    stopped: true,
                }
            );
        }
        let final_turn = turn.seal();
        assert_eq!(final_turn.text(), "ok");
        assert_eq!(final_turn.diagnostic_tokens(), &[] as &[u32]);
    }

    #[test]
    fn stop_excludes_intra_token_and_later_tool_payload_everywhere() {
        let mut turn = turn();
        let first = turn.observe(
            80,
            br#"answer<stop><tool_call>{"name":"contaminate"}</tool_call>"#,
        );
        assert_eq!(first.bytes, b"answer");
        assert!(first.stopped);

        let later = turn.observe(81, b"<tool_call>{\"name\":\"later\"}</tool_call>");
        assert!(later.bytes.is_empty());
        assert!(later.stopped);

        let final_turn = turn.seal();
        assert_eq!(final_turn.text(), "answer");
        assert!(final_turn.tool_calls().is_empty());
        assert!(final_turn.replay_tokens().is_none());
        assert!(final_turn.diagnostic_tokens().is_empty());
        assert!(final_turn.terminal_delta().bytes.is_empty());
        assert!(final_turn.terminal_delta().stopped);
    }

    #[test]
    fn hidden_reasoning_advances_raw_watermark_but_not_visible_text() {
        let mut turn = turn();
        let delta = turn.observe(50, b"<think>private</think>answer");
        assert_eq!(delta.bytes, b"answer");
        assert_eq!(turn.canonical_safe_bytes(), b"<think>private</think>answer");

        let final_turn = turn.seal();
        assert_eq!(final_turn.text(), "answer");
        assert_eq!(final_turn.replay_tokens(), Some([50].as_slice()));
        assert_eq!(final_turn.diagnostic_tokens(), [50].as_slice());
    }

    #[test]
    fn split_think_opener_is_held_until_completed() {
        let mut turn = turn();
        let first = turn.observe(52, b"answer<th");
        let second = turn.observe(53, b"ink>private");
        let third = turn.observe(54, b"</think>tail");

        assert_eq!(first.bytes, b"answer");
        assert!(second.bytes.is_empty());
        assert_eq!(third.bytes, b"tail");
        let final_turn = turn.seal();
        assert_eq!(final_turn.text(), "answertail");
    }

    #[test]
    fn split_think_closer_is_held_until_completed() {
        let mut turn = OpenAssistantTurn::new_with_reasoning_open([b"<stop>".as_slice()], true);
        let first = turn.observe(55, b"private</th");
        let second = turn.observe(56, b"ink>answer");

        assert!(first.bytes.is_empty());
        assert_eq!(second.bytes, b"answer");
        assert_eq!(turn.seal().text(), "answer");
    }

    #[test]
    fn false_split_think_opener_flushes_as_literal_at_seal() {
        let mut turn = turn();
        let mut streamed = turn.observe(57, b"literal<th").bytes;
        let final_turn = turn.seal();
        streamed.extend_from_slice(&final_turn.terminal_delta().bytes);
        assert_eq!(String::from_utf8(streamed).unwrap(), final_turn.text());
        assert_eq!(final_turn.terminal_delta().bytes, b"<th");
        assert_eq!(final_turn.text(), "literal<th");
    }

    #[test]
    fn false_split_think_closer_flushes_as_literal_at_seal() {
        let mut turn = turn();
        let mut streamed = turn.observe(58, b"literal</th").bytes;
        let final_turn = turn.seal();
        streamed.extend_from_slice(&final_turn.terminal_delta().bytes);
        assert_eq!(String::from_utf8(streamed).unwrap(), final_turn.text());
        assert_eq!(final_turn.terminal_delta().bytes, b"</th");
        assert_eq!(final_turn.text(), "literal</th");
    }

    #[test]
    fn into_parts_is_lossless_including_terminal_delta() {
        let mut turn = turn();
        let mut streamed = turn.observe(59, b"literal<th").bytes;
        let final_turn = turn.seal();
        let parts = final_turn.into_parts();

        streamed.extend_from_slice(&parts.terminal_delta().bytes);
        assert_eq!(parts.terminal_delta().bytes, b"<th");
        assert_eq!(String::from_utf8(streamed).unwrap(), parts.text());
    }

    #[test]
    fn every_two_chunk_split_reconstructs_stream_and_terminal_text() {
        let input = b"prefix<think>hidden</think>answer<sto";
        for split in 0..=input.len() {
            let mut turn = turn();
            let first = turn.observe(90, &input[..split]);
            let second = turn.observe(91, &input[split..]);
            let final_turn = turn.seal();

            let mut streamed = first.bytes;
            streamed.extend_from_slice(&second.bytes);
            streamed.extend_from_slice(&final_turn.terminal_delta().bytes);
            assert_eq!(
                String::from_utf8(streamed).unwrap(),
                final_turn.text(),
                "split at {split}"
            );
        }
    }

    #[test]
    fn incomplete_utf8_at_seal_cannot_replay_or_diagnose_its_token() {
        let mut turn = turn();
        turn.observe(51, b"safe\xc3");
        let final_turn = turn.seal();

        assert_eq!(final_turn.text(), "safe");
        assert_eq!(final_turn.replay_tokens(), None);
        assert_eq!(final_turn.diagnostic_tokens(), &[] as &[u32]);
    }

    #[test]
    fn malformed_interior_byte_ends_the_contiguous_sealed_cut() {
        let mut turn = turn();
        turn.observe(60, b"safe");
        turn.observe(61, b"\xfftail");
        assert_eq!(turn.canonical_safe_bytes(), b"safe");
        let final_turn = turn.seal();

        assert_eq!(final_turn.text(), "safe");
        assert_eq!(final_turn.replay_tokens(), None);
        assert_eq!(final_turn.diagnostic_tokens(), [60].as_slice());
    }

    #[test]
    fn all_consumers_share_the_same_exact_safe_boundary() {
        let mut turn = turn();
        turn.observe(70, b"answer ");
        turn.observe(
            71,
            br#"<tool_call>{"name":"lookup","arguments":{"q":"rust"}}</tool_call>"#,
        );
        turn.observe(72, b"<stop>ignored");
        let final_turn = turn.seal();

        assert_eq!(
            final_turn.text(),
            "answer <tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"rust\"}}</tool_call>"
        );
        assert_eq!(final_turn.tool_calls().len(), 1);
        assert_eq!(final_turn.tool_calls()[0].name, "lookup");
        assert_eq!(final_turn.replay_tokens(), Some([70, 71].as_slice()));
        assert_eq!(final_turn.diagnostic_tokens(), [70, 71].as_slice());
    }
}
