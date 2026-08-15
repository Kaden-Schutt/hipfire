// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Output-stream filtering — applies hold-back, tag-strip, and
//! end-of-turn suppression to the decoded byte stream as tokens are
//! emitted. Single source for what reaches stdout / network.
//!
//! Each generation loop in `crates/hipfire-runtime/examples/daemon.rs` decodes
//! every newly-committed token to bytes and ships those bytes out the
//! wire. Per-arch quirks (Gemma 4's literal `<end_of_turn>` marker that
//! sometimes resolves to the compact-EOT special token id, Qwen-style
//! `<think>` blocks, and Qwen3's `<|im_end|>` / `<|endoftext|>` terminators)
//! used to be inlined in `daemon.rs` and had to be edited per arch port.
//! `EosFilter` consumes raw decoded bytes and emits one of:
//!
//! - `FilterAction::Emit(Vec<u8>)` — write these bytes to the consumer.
//! - `FilterAction::Hold` — buffer until the stream disambiguates (a
//!   trailing partial marker prefix, a UTF-8 boundary mid-codepoint,
//!   or bytes inside a `<think>` block while `strip_think=true`).
//! - `FilterAction::Stop` — generation should stop with no pending
//!   visible prose in this step.
//! - `FilterAction::EmitAndStop(Vec<u8>)` — emit the safe prose that
//!   precedes a stop marker in the same chunk, then stop without
//!   leaking the marker.
//!
//! Construction is config-only; no allocations until the first
//! `observe` call. The filter is `Send` and stateless across requests
//! after `reset()`.
//!
//! Think stripping is chunk-boundary invariant: split or fused observe
//! calls over the same byte stream produce the same visible concat.

/// Output action emitted by `EosFilter::observe`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterAction {
    /// Emit these bytes to the consumer.
    Emit(Vec<u8>),
    /// Hold these bytes; the filter is buffering until the stream
    /// disambiguates (e.g. partial marker prefix that may or may not
    /// be a stop token, or bytes inside an active `<think>` block).
    Hold,
    /// Generation should stop. No additional visible prose accompanies
    /// this step; any stop-marker bytes are discarded.
    Stop,
    /// Emit safe prose that precedes a stop marker, then stop. The
    /// stop marker itself is never included in the payload.
    EmitAndStop(Vec<u8>),
}

/// Configuration for `EosFilter`. All fields default to "filter does
/// nothing other than UTF-8-boundary-safe emit".
#[derive(Debug, Clone, Default)]
pub struct EosFilterConfig {
    /// Strip `<think>...</think>` blocks from emitted output. Bytes
    /// inside an open block are held; bytes after the close tag flow
    /// normally. The literal opener and closer (`<think>` /
    /// `</think>`) are never emitted in this mode.
    ///
    /// Outside a think span, an orphan `</think>` drops only the closer
    /// and preserves preceding prose. Paired blocks strip exactly the
    /// marker-delimited span while preserving prose before and after.
    pub strip_think: bool,
    /// When `strip_think` is on, start already inside a think block
    /// (assistant prefix / rendered tail opened `<think>` so the
    /// opener never appears in the generated stream). Content is held
    /// until the first `</think>` closer.
    pub started_in_think: bool,
    /// Byte sequences that signal end of turn. Generation stops at
    /// any match. Examples: `b"<|im_end|>"`, `b"<|endoftext|>"`,
    /// `b"<end_of_turn>"`, the compact-EOT marker that some Gemma 4
    /// GGUFs decode to.
    pub stop_at: Vec<Vec<u8>>,
    /// Byte prefixes that are ambiguous — buffer until disambiguated.
    /// Use for partial markers that may or may not be a stop token.
    /// On a true match, the buffered bytes are dropped (Stop).
    /// On a false match, the buffered bytes are flushed (Emit).
    pub holdback_prefixes: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Default)]
struct EosFilterState {
    /// Raw input bytes not yet fully consumed by the transform cursor.
    raw: Vec<u8>,
    /// Cursor into `raw`: bytes before this have been classified.
    cursor: usize,
    /// Visible prose produced by the transform, not yet returned to caller.
    visible: Vec<u8>,
    /// How many bytes of `visible` have already been returned via Emit.
    visible_emitted: usize,
    /// True while inside a stripped `<think>...</think>` span.
    in_think: bool,
    /// Generation has already stopped; further observe is a no-op Hold.
    stopped: bool,
}

/// Per-request output-stream filter. Construct from a
/// `EosFilterConfig` once per generation; feed each token's freshly
/// decoded bytes to `observe`. Reset between conversations / requests.
pub struct EosFilter {
    config: EosFilterConfig,
    state: EosFilterState,
}

impl EosFilter {
    /// Construct from a config. The empty default (`strip_think=false`,
    /// no `stop_at`, no `holdback_prefixes`) is the master daemon's
    /// pre-extraction behavior: a UTF-8-boundary-safe pass-through.
    pub fn new(config: EosFilterConfig) -> Self {
        let mut config = config;
        config
            .holdback_prefixes
            .sort_by(|a, b| b.len().cmp(&a.len()));
        config.stop_at.sort_by(|a, b| b.len().cmp(&a.len()));
        let in_think = config.strip_think && config.started_in_think;
        Self {
            config,
            state: EosFilterState {
                in_think,
                ..EosFilterState::default()
            },
        }
    }

    /// Reset between turns / requests. After this, the filter behaves
    /// as if freshly constructed from the same config (including
    /// `started_in_think`).
    pub fn reset(&mut self) {
        self.state = EosFilterState {
            in_think: self.config.strip_think && self.config.started_in_think,
            ..EosFilterState::default()
        };
    }

    /// Whether the filter currently has buffered bytes that have not
    /// been emitted. Useful for decisions like "did we drop content?"
    /// at end-of-stream. The caller can call `flush_pending` to drain.
    pub fn has_pending(&self) -> bool {
        if self.state.stopped {
            return false;
        }
        self.safe_visible_end() > self.state.visible_emitted
    }

    /// Whether the filter is currently inside a stripped think span
    /// (including `started_in_think` before the first closer).
    pub fn in_think(&self) -> bool {
        self.state.in_think
    }

    /// Drain any bytes currently held back due to UTF-8 boundary or
    /// marker-prefix buffering, *not* including bytes inside an open
    /// `<think>` block. At true token EOS / length, ordinary trailing
    /// watched-prefix prose (e.g. `answer <` that never completed a
    /// stop/think marker) is emitted unchanged. Completed stop markers
    /// remain suppressed (they set `stopped` and never enter visible).
    /// Intended for end-of-stream when the caller wants disambiguated
    /// safe prose. Returns the bytes that were held; caller emits them.
    pub fn flush_pending(&mut self) -> Vec<u8> {
        if self.state.stopped {
            return Vec::new();
        }
        // Force-classify remaining raw. At EOS, trailing partial marker
        // prefixes are ordinary prose and must be emitted; open-think
        // content stays suppressed inside `pump(at_eos=true)`.
        let _ = self.pump(true);
        if self.state.in_think {
            // Do not leak hidden reasoning; leave residual unread.
            return Vec::new();
        }
        let lo = self.state.visible_emitted;
        let hi = self.state.visible.len();
        if lo >= hi {
            return Vec::new();
        }
        let out = self.state.visible[lo..hi].to_vec();
        self.state.visible_emitted = hi;
        out
    }

    /// Feed newly-decoded bytes from a single token. Returns the next
    /// action.
    pub fn observe(&mut self, raw_bytes: &[u8]) -> FilterAction {
        if self.state.stopped {
            return FilterAction::Hold;
        }
        if raw_bytes.is_empty()
            && self.state.raw.is_empty()
            && self.state.visible_emitted >= self.state.visible.len()
        {
            return FilterAction::Hold;
        }
        self.state.raw.extend_from_slice(raw_bytes);

        if let Some(action) = self.pump(false) {
            return action;
        }

        // Emit newly available safe visible prose.
        let end = self.safe_visible_end();
        if end > self.state.visible_emitted {
            let out = self.state.visible[self.state.visible_emitted..end].to_vec();
            self.state.visible_emitted = end;
            FilterAction::Emit(out)
        } else {
            FilterAction::Hold
        }
    }

    /// Run the transform over `raw` from `cursor`. Returns `Some(Stop*)`
    /// if a stop marker was committed; otherwise `None`.
    ///
    /// `at_eos`: when true, trailing partial marker prefixes are treated
    /// as ordinary prose and flushed into visible (true end-of-stream
    /// disambiguation). Open-think content is still discarded. Incomplete
    /// UTF-8 tails are emitted as-is at EOS (no further bytes will arrive).
    fn pump(&mut self, at_eos: bool) -> Option<FilterAction> {
        const OPEN: &[u8] = b"<think>";
        const CLOSE: &[u8] = b"</think>";

        loop {
            if self.state.cursor >= self.state.raw.len() {
                return None;
            }
            let cursor = self.state.cursor;
            let raw_len = self.state.raw.len();

            if self.state.in_think {
                // Seek closer. Partial closer prefix at end → hold.
                let rest = &self.state.raw[cursor..raw_len];
                if let Some(idx) = memmem(rest, CLOSE) {
                    // Discard reasoning + closer.
                    self.state.cursor = cursor + idx + CLOSE.len();
                    self.state.in_think = false;
                    continue;
                }
                // No complete closer. Advance past bytes that cannot be
                // a closer prefix so the buffer does not grow unbound;
                // keep a trailing partial closer prefix.
                let keep_from = if at_eos {
                    rest.len()
                } else {
                    trailing_prefix_start(rest, CLOSE)
                };
                self.state.cursor = cursor + keep_from;
                return None;
            }

            // Outside think: find earliest of stop markers, optional think
            // open/close, subject to partial-prefix hold at the tail.
            let mut earliest: Option<Event> = None;
            {
                let rest = &self.state.raw[cursor..raw_len];
                for needle in &self.config.stop_at {
                    if needle.is_empty() {
                        continue;
                    }
                    if let Some(idx) = memmem(rest, needle) {
                        earliest = Event::earlier(
                            earliest,
                            Event {
                                at: idx,
                                kind: EventKind::Stop,
                                len: needle.len(),
                            },
                        );
                    }
                }

                if self.config.strip_think {
                    if let Some(idx) = memmem(rest, OPEN) {
                        earliest = Event::earlier(
                            earliest,
                            Event {
                                at: idx,
                                kind: EventKind::ThinkOpen,
                                len: OPEN.len(),
                            },
                        );
                    }
                    if let Some(idx) = memmem(rest, CLOSE) {
                        earliest = Event::earlier(
                            earliest,
                            Event {
                                at: idx,
                                kind: EventKind::ThinkClose,
                                len: CLOSE.len(),
                            },
                        );
                    }
                }
            }

            match earliest {
                None => {
                    // No complete event. Mid-stream: emit all but trailing
                    // partial marker / holdback / incomplete UTF-8 prefixes.
                    // At EOS: remaining bytes are ordinary prose (including
                    // watched-prefix tails that never completed a marker).
                    let take = {
                        let rest = &self.state.raw[cursor..raw_len];
                        if at_eos {
                            rest.len()
                        } else {
                            self.holdback_trim_len(rest)
                        }
                    };
                    if take > 0 {
                        let prose = self.state.raw[cursor..cursor + take].to_vec();
                        self.push_visible(&prose);
                        self.state.cursor = cursor + take;
                    }
                    return None;
                }
                Some(ev) => {
                    // Prose before the event is visible (UTF-8 safe later).
                    if ev.at > 0 {
                        let prose = self.state.raw[cursor..cursor + ev.at].to_vec();
                        self.push_visible(&prose);
                    }
                    self.state.cursor = cursor + ev.at + ev.len;
                    match ev.kind {
                        EventKind::Stop => {
                            // Commit stop. Drain any newly pushed visible
                            // that is safe, then Stop / EmitAndStop.
                            self.state.stopped = true;
                            // Compact raw (optional); keep state consistent.
                            self.state.raw.clear();
                            self.state.cursor = 0;
                            let end = self.safe_visible_end();
                            if end > self.state.visible_emitted {
                                let out =
                                    self.state.visible[self.state.visible_emitted..end].to_vec();
                                self.state.visible_emitted = self.state.visible.len();
                                return Some(FilterAction::EmitAndStop(out));
                            }
                            self.state.visible_emitted = self.state.visible.len();
                            return Some(FilterAction::Stop);
                        }
                        EventKind::ThinkOpen => {
                            self.state.in_think = true;
                            continue;
                        }
                        EventKind::ThinkClose => {
                            // Orphan closer: drop only the marker.
                            continue;
                        }
                    }
                }
            }
        }
    }

    fn push_visible(&mut self, bytes: &[u8]) {
        self.state.visible.extend_from_slice(bytes);
    }

    /// Largest end offset in `visible` that is safe to emit now.
    fn safe_visible_end(&self) -> usize {
        let lo = self.state.visible_emitted;
        let hi = self.state.visible.len();
        if lo >= hi {
            return lo;
        }
        let slice = &self.state.visible[lo..hi];
        let mut end = utf8_safe_end(slice) + lo;
        // Visible should not contain partial markers if pump is correct,
        // but still trim incomplete UTF-8 only (markers never enter visible).
        let _ = end;
        end
    }

    /// How many leading bytes of `rest` are safe to classify as prose
    /// given that no complete event was found — i.e. everything except a
    /// trailing partial prefix of stop/holdback/think markers.
    fn holdback_trim_len(&self, rest: &[u8]) -> usize {
        if rest.is_empty() {
            return 0;
        }
        let mut end = rest.len();
        // UTF-8: do not take a trailing incomplete codepoint into visible
        // as "committed" when more bytes may arrive — but visible path
        // also gates on utf8_safe_end. Still avoid stuffing incomplete
        // sequences that are only marker heads.
        let mut watch: Vec<&[u8]> = Vec::new();
        for p in &self.config.holdback_prefixes {
            if !p.is_empty() {
                watch.push(p.as_slice());
            }
        }
        for s in &self.config.stop_at {
            if !s.is_empty() {
                watch.push(s.as_slice());
            }
        }
        if self.config.strip_think {
            watch.push(b"<think>");
            watch.push(b"</think>");
        }
        if !watch.is_empty() {
            let mut max_trim = 0usize;
            for p in &watch {
                let max_k = p.len().saturating_sub(1).min(end);
                for k in (1..=max_k).rev() {
                    if k <= max_trim {
                        break;
                    }
                    if rest[end - k..end] == p[..k] {
                        max_trim = k;
                        break;
                    }
                }
            }
            end -= max_trim;
        }
        // Also hold incomplete UTF-8 at the tail so it pairs with the
        // next token rather than landing as a replacement later.
        end = utf8_safe_end(&rest[..end]);
        end
    }
}

#[derive(Clone, Copy)]
struct Event {
    at: usize,
    kind: EventKind,
    len: usize,
}

#[derive(Clone, Copy)]
enum EventKind {
    Stop,
    ThinkOpen,
    ThinkClose,
}

impl Event {
    fn earlier(best: Option<Event>, cand: Event) -> Option<Event> {
        match best {
            None => Some(cand),
            Some(b) if cand.at < b.at => Some(cand),
            Some(b) if cand.at == b.at && cand.len > b.len => Some(cand),
            // Prefer Stop over think markers at the same offset.
            Some(b)
                if cand.at == b.at
                    && matches!(cand.kind, EventKind::Stop)
                    && !matches!(b.kind, EventKind::Stop) =>
            {
                Some(cand)
            }
            other => other,
        }
    }
}

// --- helpers -------------------------------------------------------

/// Return the largest `k <= bytes.len()` such that `bytes[..k]` ends
/// on a UTF-8 codepoint boundary.
fn utf8_safe_end(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(e) => e.valid_up_to(),
    }
}

/// Naive substring search.
fn memmem(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Return the smallest `k` such that `bytes[k..]` is a non-empty
/// prefix of `needle`. If no such tail exists, returns `bytes.len()`.
fn trailing_prefix_start(bytes: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() || bytes.is_empty() {
        return bytes.len();
    }
    let max_k = needle.len().saturating_sub(1).min(bytes.len());
    for k in (1..=max_k).rev() {
        if bytes[bytes.len() - k..] == needle[..k] {
            return bytes.len() - k;
        }
    }
    bytes.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_default() -> EosFilterConfig {
        EosFilterConfig::default()
    }

    fn cfg_im_end() -> EosFilterConfig {
        EosFilterConfig {
            strip_think: false,
            started_in_think: false,
            stop_at: vec![b"<|im_end|>".to_vec()],
            holdback_prefixes: Vec::new(),
        }
    }

    fn cfg_strip_think() -> EosFilterConfig {
        EosFilterConfig {
            strip_think: true,
            started_in_think: false,
            stop_at: Vec::new(),
            holdback_prefixes: Vec::new(),
        }
    }

    fn cfg_gemma4_eot() -> EosFilterConfig {
        EosFilterConfig {
            strip_think: false,
            started_in_think: false,
            stop_at: vec![b"<end_of_turn>".to_vec()],
            holdback_prefixes: vec![b"<end_of_turn>".to_vec()],
        }
    }

    fn cfg_qwen_ar(started: bool) -> EosFilterConfig {
        EosFilterConfig {
            strip_think: true,
            started_in_think: started,
            stop_at: vec![b"<|im_end|>".to_vec(), b"<|endoftext|>".to_vec()],
            holdback_prefixes: Vec::new(),
        }
    }

    /// Concatenate all Emit payloads from a sequence of observe calls.
    fn drive(cfg: EosFilterConfig, chunks: &[&[u8]]) -> (Vec<u8>, bool) {
        let mut f = EosFilter::new(cfg);
        let mut out = Vec::new();
        let mut stopped = false;
        for c in chunks {
            match f.observe(c) {
                FilterAction::Emit(b) => out.extend_from_slice(&b),
                FilterAction::EmitAndStop(b) => {
                    out.extend_from_slice(&b);
                    stopped = true;
                    break;
                }
                FilterAction::Hold => {}
                FilterAction::Stop => {
                    stopped = true;
                    break;
                }
            }
        }
        if !stopped {
            out.extend_from_slice(&f.flush_pending());
        }
        (out, stopped)
    }

    #[test]
    fn empty_input_with_empty_state_holds() {
        let mut f = EosFilter::new(cfg_default());
        assert_eq!(f.observe(&[]), FilterAction::Hold);
    }

    #[test]
    fn single_ascii_byte_emits() {
        let mut f = EosFilter::new(cfg_default());
        assert_eq!(f.observe(b"a"), FilterAction::Emit(b"a".to_vec()));
        assert_eq!(f.observe(b"bc"), FilterAction::Emit(b"bc".to_vec()));
    }

    #[test]
    fn utf8_split_across_tokens_holds_then_emits() {
        let mut f = EosFilter::new(cfg_default());
        let smile = "😀".as_bytes();
        assert_eq!(smile.len(), 4);
        assert_eq!(f.observe(&smile[..2]), FilterAction::Hold);
        assert_eq!(f.observe(&smile[2..]), FilterAction::Emit(smile.to_vec()));
    }

    #[test]
    fn think_open_holds_until_close() {
        let mut f = EosFilter::new(cfg_strip_think());
        assert_eq!(f.observe(b"hello "), FilterAction::Emit(b"hello ".to_vec()));
        assert_eq!(f.observe(b"<think>reasoning"), FilterAction::Hold);
        assert_eq!(f.observe(b" more"), FilterAction::Hold);
        match f.observe(b"</think>answer") {
            FilterAction::Emit(bytes) => assert_eq!(bytes, b"answer"),
            other => panic!("expected Emit(\"answer\"), got {:?}", other),
        }
    }

    #[test]
    fn close_think_alone_resumes_emit() {
        let mut f = EosFilter::new(cfg_strip_think());
        assert_eq!(f.observe(b"<think>x"), FilterAction::Hold);
        assert_eq!(f.observe(b"</think>"), FilterAction::Hold);
        assert_eq!(f.observe(b" world"), FilterAction::Emit(b" world".to_vec()));
    }

    #[test]
    fn started_in_think_holds_until_closer() {
        let mut f = EosFilter::new(EosFilterConfig {
            strip_think: true,
            started_in_think: true,
            stop_at: Vec::new(),
            holdback_prefixes: Vec::new(),
        });
        assert_eq!(f.observe(b"hidden reasoning"), FilterAction::Hold);
        match f.observe(b"</think>visible") {
            FilterAction::Emit(bytes) => assert_eq!(bytes, b"visible"),
            other => panic!("expected Emit(visible), got {:?}", other),
        }
    }

    #[test]
    fn orphan_closer_preserves_preceding_prose() {
        let (out, _) = drive(cfg_strip_think(), &[b"hidden</think>answer"]);
        assert_eq!(out, b"hiddenanswer");
    }

    #[test]
    fn orphan_closer_split_chunks_preserves_prose() {
        let (out, _) = drive(cfg_strip_think(), &[b"hidden", b"</think>answer"]);
        assert_eq!(out, b"hiddenanswer");
    }

    #[test]
    fn paired_think_preserves_prose_before_and_after_one_chunk() {
        let (out, _) = drive(cfg_strip_think(), &[b"pre <think>secret</think> post"]);
        assert_eq!(out, b"pre  post");
    }

    #[test]
    fn paired_think_split_boundaries_invariant() {
        let cases: &[&[&[u8]]] = &[
            &[b"pre ", b"<think>secret</think>", b" post"],
            &[b"pre <th", b"ink>secret</th", b"ink> post"],
            &[b"pre <think>", b"secret", b"</think> post"],
            &[b"pre <think>secret</think> post"],
            &[b"p", b"re <", b"think>se", b"cret</", b"think> p", b"ost"],
        ];
        for chunks in cases {
            let (out, _) = drive(cfg_strip_think(), chunks);
            assert_eq!(
                out,
                b"pre  post",
                "chunks={:?}",
                chunks
                    .iter()
                    .map(|c| String::from_utf8_lossy(c).into_owned())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn orphan_closer_all_split_boundaries() {
        let cases: &[&[&[u8]]] = &[
            &[b"hidden</think>answer"],
            &[b"hidden", b"</think>answer"],
            &[b"hid", b"den</th", b"ink>ans", b"wer"],
            &[b"hidden</", b"think>", b"answer"],
        ];
        for chunks in cases {
            let (out, _) = drive(cfg_strip_think(), chunks);
            assert_eq!(
                out,
                b"hiddenanswer",
                "chunks={:?}",
                chunks
                    .iter()
                    .map(|c| String::from_utf8_lossy(c).into_owned())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn stop_at_does_not_emit_im_end_bytes() {
        let mut f = EosFilter::new(cfg_qwen_ar(false));
        assert_eq!(f.observe(b"hi"), FilterAction::Emit(b"hi".to_vec()));
        assert_eq!(f.observe(b"<|im_end|>"), FilterAction::Stop);
    }

    #[test]
    fn stop_at_endoftext_does_not_emit_marker() {
        let mut f = EosFilter::new(cfg_qwen_ar(false));
        assert_eq!(f.observe(b"hi"), FilterAction::Emit(b"hi".to_vec()));
        assert_eq!(f.observe(b"<|endoftext|>"), FilterAction::Stop);
    }

    #[test]
    fn stop_with_prose_same_chunk_emits_prose_then_stops() {
        let mut f = EosFilter::new(cfg_qwen_ar(false));
        match f.observe(b"hello<|im_end|>") {
            FilterAction::EmitAndStop(b) => assert_eq!(b, b"hello"),
            other => panic!("expected EmitAndStop(hello), got {:?}", other),
        }
    }

    #[test]
    fn stop_at_full_match_returns_stop() {
        let mut f = EosFilter::new(cfg_im_end());
        assert_eq!(f.observe(b"hi"), FilterAction::Emit(b"hi".to_vec()));
        assert_eq!(f.observe(b"<|im_end|>"), FilterAction::Stop);
    }

    #[test]
    fn partial_holdback_prefix_holds_then_flushes_on_false_match() {
        let mut f = EosFilter::new(cfg_gemma4_eot());
        assert_eq!(f.observe(b"<en"), FilterAction::Hold);
        match f.observe(b"glish") {
            FilterAction::Emit(bytes) => assert_eq!(bytes, b"<english"),
            other => panic!("expected Emit('<english'), got {:?}", other),
        }
    }

    #[test]
    fn partial_holdback_prefix_then_full_match_stops() {
        let mut f = EosFilter::new(cfg_gemma4_eot());
        assert_eq!(f.observe(b"<en"), FilterAction::Hold);
        assert_eq!(f.observe(b"d_of_turn>"), FilterAction::Stop);
    }

    #[test]
    fn reset_clears_state() {
        let mut f = EosFilter::new(cfg_strip_think());
        assert_eq!(f.observe(b"<think>"), FilterAction::Hold);
        f.reset();
        assert_eq!(f.observe(b"clean"), FilterAction::Emit(b"clean".to_vec()));
    }

    #[test]
    fn flush_pending_never_flushes_open_think() {
        let mut f = EosFilter::new(cfg_strip_think());
        assert_eq!(f.observe(b"<think>secret"), FilterAction::Hold);
        assert!(f.flush_pending().is_empty());
    }

    #[test]
    fn flush_pending_emits_ordinary_trailing_marker_prefix() {
        let mut f = EosFilter::new(cfg_qwen_ar(false));
        // Mid-stream: safe prose emits; trailing `<` is held as a think/stop prefix.
        match f.observe(b"answer <") {
            FilterAction::Emit(b) => assert_eq!(b, b"answer "),
            FilterAction::Hold => {}
            other => panic!("unexpected {other:?}"),
        }
        // True EOS: held watched-prefix prose must flush unchanged.
        assert_eq!(f.flush_pending(), b"<".to_vec());
        assert!(f.flush_pending().is_empty());
    }

    #[test]
    fn flush_pending_emits_partial_im_end_prefix() {
        let mut f = EosFilter::new(cfg_qwen_ar(false));
        match f.observe(b"hi<|im_") {
            FilterAction::Emit(b) => assert_eq!(b, b"hi"),
            FilterAction::Hold => {}
            other => panic!("unexpected {other:?}"),
        }
        // Incomplete stop marker at true EOS is ordinary prose — not a completed marker.
        let flushed = f.flush_pending();
        assert!(
            flushed == b"<|im_".to_vec() || flushed == b"hi<|im_".to_vec(),
            "flushed={flushed:?}"
        );
    }

    #[test]
    fn flush_pending_still_suppresses_completed_stop_marker() {
        let mut f = EosFilter::new(cfg_qwen_ar(false));
        assert_eq!(
            f.observe(b"hi<|im_end|>"),
            FilterAction::EmitAndStop(b"hi".to_vec())
        );
        assert!(f.flush_pending().is_empty());
    }

    #[test]
    fn stop_at_spanning_two_tokens_stops() {
        let mut f = EosFilter::new(cfg_im_end());
        assert_eq!(f.observe(b"<|im_"), FilterAction::Hold);
        assert_eq!(f.observe(b"end|>"), FilterAction::Stop);
    }

    #[test]
    fn started_in_think_split_closer_invariant() {
        let cases: &[&[&[u8]]] = &[
            &[b"hidden", b"</think>visible"],
            &[b"hidden</think>visible"],
            &[b"hid", b"den</th", b"ink>vis", b"ible"],
        ];
        for chunks in cases {
            let (out, _) = drive(cfg_qwen_ar(true), chunks);
            assert_eq!(
                out,
                b"visible",
                "chunks={:?}",
                chunks
                    .iter()
                    .map(|c| String::from_utf8_lossy(c).into_owned())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn endoftext_with_prose_same_chunk() {
        let mut f = EosFilter::new(cfg_qwen_ar(false));
        match f.observe(b"bye<|endoftext|>") {
            FilterAction::EmitAndStop(b) => assert_eq!(b, b"bye"),
            other => panic!("expected EmitAndStop(bye), got {:?}", other),
        }
    }

    /// Every nonempty proper prefix of every watched think/EOT marker.
    fn qwen_ar_watched_markers() -> &'static [&'static [u8]] {
        &[b"<think>", b"</think>", b"<|im_end|>", b"<|endoftext|>"]
    }

    fn nonempty_proper_prefixes(marker: &[u8]) -> Vec<&[u8]> {
        (1..marker.len()).map(|n| &marker[..n]).collect()
    }

    #[test]
    fn table_natural_eos_every_watched_marker_prefix() {
        // Fix round 5: filter-level proper-prefix flush at natural EOS only.
        // Dual terminal (natural EOS vs `hit_length_cap`) is owned by the
        // production `QwenArSemanticProducer::finish` table in daemon tests —
        // do not mirror two identical filter-only `drive` calls here.
        let prose = b"answer ";
        for marker in qwen_ar_watched_markers() {
            for prefix in nonempty_proper_prefixes(marker) {
                let mut f = EosFilter::new(cfg_qwen_ar(false));
                let mut chunk = prose.to_vec();
                chunk.extend_from_slice(prefix);
                match f.observe(&chunk) {
                    FilterAction::Emit(_) | FilterAction::Hold => {}
                    other => panic!(
                        "mid-stream unexpected for prefix {prefix:?} of {marker:?}: {other:?}"
                    ),
                }
                // Direct flush_pending also returns residual without panic.
                let _ = f.flush_pending();
                // Re-drive for exact concat assertion (natural EOS).
                let (out, stopped) = drive(cfg_qwen_ar(false), &[&chunk]);
                assert!(
                    !stopped,
                    "proper prefix must not complete stop: marker={marker:?} prefix={prefix:?}"
                );
                let mut want = prose.to_vec();
                want.extend_from_slice(prefix);
                assert_eq!(
                    out, want,
                    "natural EOS must flush proper prefix as prose: marker={marker:?} prefix={prefix:?}"
                );
            }

            // Completed marker suppression: full marker never appears in output.
            let mut full = prose.to_vec();
            full.extend_from_slice(marker);
            let (out, stopped) = drive(cfg_qwen_ar(false), &[&full]);
            let out_str = String::from_utf8_lossy(&out);
            let marker_str = String::from_utf8_lossy(marker);
            assert!(
                !out_str.contains(marker_str.as_ref()),
                "completed marker must be suppressed: marker={marker_str:?} out={out_str:?}"
            );
            // Think open starts strip (open-think hold); stop markers stop.
            if *marker == b"<think>" {
                // prose before open remains; marker and following held/discarded at EOS if open.
                assert!(
                    out.starts_with(prose) || out == prose || out.is_empty() || out == b"answer ",
                    "think open: out={out:?}"
                );
            } else if *marker == b"</think>" {
                // orphan closer drops closer, keeps prose
                assert!(
                    out_str.contains("answer"),
                    "orphan closer keeps prose: {out_str}"
                );
                assert!(!stopped, "orphan closer is not a stop marker");
            } else {
                assert!(stopped, "EOT completed marker must stop: {marker_str}");
                assert_eq!(out, prose, "EOT emits only preceding prose");
            }
        }
    }

    #[test]
    fn table_completed_marker_still_suppressed_after_prefix_hold() {
        // Hold a proper prefix mid-stream, then complete the marker → suppress.
        for marker in [b"<|im_end|>".as_slice(), b"<|endoftext|>".as_slice()] {
            for split in 1..marker.len() {
                let left = &marker[..split];
                let right = &marker[split..];
                let mut f = EosFilter::new(cfg_qwen_ar(false));
                assert_eq!(f.observe(b"hi"), FilterAction::Emit(b"hi".to_vec()));
                match f.observe(left) {
                    FilterAction::Hold | FilterAction::Emit(_) => {}
                    other => panic!("prefix hold: {other:?}"),
                }
                match f.observe(right) {
                    FilterAction::Stop | FilterAction::EmitAndStop(_) => {}
                    other => {
                        panic!("completed after split={split} marker={marker:?} got {other:?}")
                    }
                }
                assert!(
                    f.flush_pending().is_empty(),
                    "no residual after completed stop marker"
                );
            }
        }
    }
}
