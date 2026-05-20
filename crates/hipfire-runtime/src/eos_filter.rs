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
//! `<think>` blocks, and Qwen3's `<|im_end|>` ChatML terminator) used
//! to be inlined in `daemon.rs` and had to be edited per arch port.
//! `EosFilter` consumes raw decoded bytes and emits one of:
//!
//! - `FilterAction::Emit(Vec<u8>)` — write these bytes to the consumer.
//! - `FilterAction::Hold` — buffer until the stream disambiguates (a
//!   trailing partial marker prefix, a UTF-8 boundary mid-codepoint,
//!   or bytes inside a `<think>` block while `strip_think=true`).
//! - `FilterAction::Stop` — generation should stop. Any buffered bytes
//!   are discarded; the caller must not emit further output.
//!
//! Construction is config-only; no allocations until the first
//! `observe` call. The filter is `Send` and stateless across requests
//! after `reset()`.

use std::cmp::Ordering;

/// Output action emitted by `EosFilter::observe`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterAction {
    /// Emit these bytes to the consumer.
    Emit(Vec<u8>),
    /// Hold these bytes; the filter is buffering until the stream
    /// disambiguates (e.g. partial marker prefix that may or may not
    /// be a stop token, or bytes inside an active `<think>` block).
    Hold,
    /// Generation should stop. Any buffered bytes are discarded.
    Stop,
}

/// Configuration for `EosFilter`. All fields default to "filter does
/// nothing other than UTF-8-boundary-safe emit".
#[derive(Debug, Clone, Default)]
pub struct EosFilterConfig {
    /// Strip `<think>...</think>` blocks from emitted output. Bytes
    /// inside an open block are held; bytes after the close tag flow
    /// normally. The literal opener and closer (`<think>` /
    /// `</think>`) are never emitted in this mode.
    pub strip_think: bool,
    /// Byte sequences that signal end of turn. Generation stops at
    /// any match. Examples: `b"<|im_end|>"`, `b"<end_of_turn>"`, the
    /// compact-EOT marker that some Gemma 4 GGUFs decode to.
    pub stop_at: Vec<Vec<u8>>,
    /// Byte prefixes that are ambiguous — buffer until disambiguated.
    /// Use for partial markers that may or may not be a stop token.
    /// On a true match, the buffered bytes are dropped (Stop).
    /// On a false match, the buffered bytes are flushed (Emit).
    pub holdback_prefixes: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Default)]
struct EosFilterState {
    /// Bytes accumulated since the last full flush. Includes any bytes
    /// held back for marker-prefix disambiguation, UTF-8 boundary
    /// safety, or in-flight `<think>` content. Cleared by `reset()`.
    buf: Vec<u8>,
    /// True while we are inside a `<think>...</think>` block and
    /// `strip_think` is on. Set when the opener is seen, cleared on
    /// the closer.
    in_think: bool,
    /// Number of bytes already returned to the caller (in Emit
    /// actions). Used to compute the "new emit" delta on each call.
    emitted: usize,
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
        // Sort holdback_prefixes longest-first so prefix-match scans
        // pick the longest matching prefix, not the first one. Same
        // for stop_at (an early shorter match must not preempt a
        // longer one starting at the same offset).
        let mut config = config;
        config.holdback_prefixes.sort_by(|a, b| b.len().cmp(&a.len()));
        config.stop_at.sort_by(|a, b| b.len().cmp(&a.len()));
        Self {
            config,
            state: EosFilterState::default(),
        }
    }

    /// Reset between turns / requests. After this, the filter behaves
    /// as if freshly constructed from the same config.
    pub fn reset(&mut self) {
        self.state = EosFilterState::default();
    }

    /// Whether the filter currently has buffered bytes that have not
    /// been emitted. Useful for decisions like "did we drop content?"
    /// at end-of-stream. The caller can call `flush_pending` to drain.
    pub fn has_pending(&self) -> bool {
        self.state.emitted < self.state.buf.len()
    }

    /// Drain any bytes currently held back due to UTF-8 boundary or
    /// marker-prefix buffering, *not* including bytes inside an open
    /// `<think>` block. Intended for use at end-of-stream when the
    /// caller has already broken on a token-level stop signal and
    /// wants to flush any bytes the filter was holding pending
    /// disambiguation. Returns the bytes that were held; caller is
    /// responsible for emitting them.
    pub fn flush_pending(&mut self) -> Vec<u8> {
        if self.state.in_think {
            // We never flush mid-think content — that was the whole
            // point of strip_think. Drop pending and stay quiet.
            self.state.emitted = self.state.buf.len();
            return Vec::new();
        }
        let pending = self.state.buf[self.state.emitted..].to_vec();
        self.state.emitted = self.state.buf.len();
        pending
    }

    /// Feed newly-decoded bytes from a single token. Returns the next
    /// action.
    ///
    /// State machine, per call:
    /// 1. Append `raw_bytes` to the internal buffer.
    /// 2. Scan from `state.emitted` for any complete `stop_at` match.
    ///    If found, return `Stop` immediately — the held bytes plus
    ///    the new bytes are discarded together with the stop marker.
    /// 3. If `strip_think` is on:
    ///    - If we are inside a think block, scan for the closer
    ///      `</think>`. On hit, jump `emitted` past the closer and
    ///      clear `in_think`; otherwise advance `emitted` to the start
    ///      of any partial trailing closer prefix and Hold.
    ///    - If we are outside a think block, scan from `emitted` for
    ///      the opener `<think>`. On hit, the bytes before the opener
    ///      are emit candidates; jump `emitted` past the opener and
    ///      set `in_think`. Otherwise leave `emitted` where it is.
    /// 4. Compute the maximal "safe" emit prefix:
    ///    - It must end on a UTF-8 codepoint boundary.
    ///    - Its tail must not match any `holdback_prefix`.
    ///    Anything after that point stays buffered.
    /// 5. Return `Emit(prefix)` if non-empty, else `Hold`.
    pub fn observe(&mut self, raw_bytes: &[u8]) -> FilterAction {
        if raw_bytes.is_empty() && self.state.buf.is_empty() {
            // Nothing in flight. Pre-existing daemon behavior on
            // zero-byte tokens (e.g. "decode_bytes returned empty")
            // was to emit nothing — match it with Hold so the caller
            // does not write a JSON token frame for an empty payload.
            return FilterAction::Hold;
        }
        self.state.buf.extend_from_slice(raw_bytes);

        // (1) Stop-at scan. Look across the whole accumulated buffer
        //     so that a marker spanning two tokens still trips. We
        //     don't scan inside the already-emitted prefix repeatedly
        //     except for the last `max_stop_len - 1` bytes to catch
        //     boundary-spanning matches, but keeping it simple and
        //     scanning from 0 is correct (just O(buf.len())).
        if !self.config.stop_at.is_empty() {
            for needle in &self.config.stop_at {
                if needle.is_empty() {
                    continue;
                }
                if memmem(&self.state.buf, needle).is_some() {
                    // Discard everything; signal Stop.
                    return FilterAction::Stop;
                }
            }
        }

        // (2) Strip-think state machine.
        if self.config.strip_think {
            self.advance_think_state();
        }

        // If a strip moved emitted forward past where the buffer
        // currently ends (impossible, but defensively clamp), bail
        // with Hold rather than panicking on slice bounds.
        if self.state.emitted > self.state.buf.len() {
            self.state.emitted = self.state.buf.len();
        }

        // While inside a think block, hold everything.
        if self.state.in_think {
            return FilterAction::Hold;
        }

        // (3) Compute safe emit prefix from `emitted` to the start of
        //     any holdback-prefix tail (or open-think-tag tail when
        //     strip_think is on).
        let safe_end = self.compute_safe_end();
        if safe_end > self.state.emitted {
            let out = self.state.buf[self.state.emitted..safe_end].to_vec();
            self.state.emitted = safe_end;
            FilterAction::Emit(out)
        } else {
            FilterAction::Hold
        }
    }

    /// Internal: advance `state.emitted` and toggle `state.in_think`
    /// based on the buffer contents. Called only when
    /// `config.strip_think` is true.
    ///
    /// We loop because a single token may close one think block and
    /// open another (rare but legal). The loop terminates either by
    /// running out of input or by a partial trailing tag that needs
    /// the next token to disambiguate.
    fn advance_think_state(&mut self) {
        const OPEN: &[u8] = b"<think>";
        const CLOSE: &[u8] = b"</think>";

        loop {
            if self.state.in_think {
                // Inside a think block. Look for `</think>` anywhere
                // in the unscanned tail.
                if let Some(idx) = memmem(&self.state.buf[self.state.emitted..], CLOSE) {
                    // Skip the closer entirely. Bytes after it are
                    // emit candidates (subject to a possible new
                    // opener and to the holdback / UTF-8 trims below).
                    self.state.emitted += idx + CLOSE.len();
                    self.state.in_think = false;
                    continue;
                } else {
                    // No complete closer yet. Advance `emitted` up to
                    // the start of any partial trailing prefix of
                    // `</think>`, so when the next token completes it
                    // we recognize and skip the closer instead of
                    // re-scanning from 0. The bytes between the old
                    // `emitted` and the prefix start are inside the
                    // think block and stay un-emitted.
                    let cut = trailing_prefix_start(
                        &self.state.buf[self.state.emitted..],
                        CLOSE,
                    );
                    self.state.emitted += cut;
                    return;
                }
            } else {
                // Outside a think block. Look for `<think>`.
                if let Some(idx) = memmem(&self.state.buf[self.state.emitted..], OPEN) {
                    // Bytes before the opener are emit candidates;
                    // advance `emitted` to just after the opener and
                    // enter the think block.
                    // We do not move `emitted` past the pre-opener
                    // bytes here (those are the "emit" segment); we
                    // only mark the think transition in the buffer.
                    // The actual emit happens in `compute_safe_end`.
                    //
                    // To express this cleanly, copy the buffer head
                    // up to opener into a contiguous "to-emit" slice
                    // and advance `emitted` past the opener
                    // afterward.
                    //
                    // Implementation: rewrite the buffer in place by
                    // dropping the opener bytes from `buf`, leaving
                    // the pre-opener bytes still un-emitted at
                    // positions [emitted .. emitted+idx], and the
                    // post-opener bytes shifted down. This keeps
                    // `compute_safe_end` simple.
                    let opener_start = self.state.emitted + idx;
                    // Drain the opener bytes (`<think>`) from the
                    // buffer so they never appear in the emit slice.
                    self.state.buf.drain(opener_start..opener_start + OPEN.len());
                    // Mark the think state. `emitted` does not move:
                    // the pre-opener bytes are still pending.
                    self.state.in_think = true;
                    continue;
                } else {
                    // No complete opener. Stop scanning; the
                    // holdback / safe-emit pass below will trim any
                    // partial trailing `<think>` prefix.
                    return;
                }
            }
        }
    }

    /// Compute the largest emit-end offset `>= state.emitted` such
    /// that the slice `[emitted..end]` is safe to emit. "Safe" means:
    ///
    /// - Ends on a UTF-8 codepoint boundary.
    /// - Does NOT have a tail that is a non-empty prefix of any
    ///   `holdback_prefix` or, if `strip_think` is on, of `<think>`
    ///   (which would otherwise leak the start of an opener).
    /// - Does NOT have a tail that is a non-empty prefix of any
    ///   `stop_at` sequence (else we'd leak the head of a stop marker
    ///   we would have caught next iteration).
    fn compute_safe_end(&self) -> usize {
        let buf = &self.state.buf;
        let lo = self.state.emitted;
        let hi = buf.len();
        if lo >= hi {
            return lo;
        }

        // Start by trimming back to a UTF-8 boundary.
        let mut end = utf8_safe_end(&buf[lo..hi]) + lo;

        // Trim back further if the trailing bytes match a non-empty
        // prefix of any holdback / stop-at / think-opener pattern.
        let mut watch_prefixes: Vec<&[u8]> = Vec::new();
        for p in &self.config.holdback_prefixes {
            if !p.is_empty() {
                watch_prefixes.push(p.as_slice());
            }
        }
        for s in &self.config.stop_at {
            if !s.is_empty() {
                watch_prefixes.push(s.as_slice());
            }
        }
        if self.config.strip_think {
            watch_prefixes.push(b"<think>");
        }

        if !watch_prefixes.is_empty() {
            // Find the longest non-empty prefix `p` such that the
            // tail of buf[lo..end] equals `p[..k]` for some 1 <= k <
            // p.len(). If such a tail exists, pull `end` back past
            // that tail.
            let mut max_trim = 0usize;
            for p in &watch_prefixes {
                let max_k = p.len().saturating_sub(1).min(end - lo);
                for k in (1..=max_k).rev() {
                    if k <= max_trim {
                        break;
                    }
                    if buf[end - k..end] == p[..k] {
                        max_trim = k;
                        break;
                    }
                }
            }
            end -= max_trim;
        }

        end
    }
}

// --- helpers -------------------------------------------------------

/// Return the largest `k <= bytes.len()` such that `bytes[..k]` ends
/// on a UTF-8 codepoint boundary. Mirrors the inline
/// `match str::from_utf8 { Ok(_) => bytes.len(), Err(e) =>
/// e.valid_up_to() }` snippet that appears across `daemon.rs` ahead of
/// each `writeln!(stdout, ...)` token write.
fn utf8_safe_end(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(e) => e.valid_up_to(),
    }
}

/// Naive substring search — pulled into a helper so we can drop in a
/// faster scanner later without changing the call sites.
fn memmem(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    let n = needle.len();
    for i in 0..=haystack.len() - n {
        if haystack[i..i + n] == *needle {
            return Some(i);
        }
    }
    None
}

/// Return the smallest `k` such that `bytes[k..]` is a non-empty
/// prefix of `needle` (i.e. the start of a possible occurrence
/// straddling the end of `bytes`). If no such tail exists, returns
/// `bytes.len()` — meaning the whole input can be safely consumed.
fn trailing_prefix_start(bytes: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() || bytes.is_empty() {
        return bytes.len();
    }
    let max_overlap = bytes.len().min(needle.len() - 1);
    for k in (1..=max_overlap).rev() {
        let start = bytes.len() - k;
        match bytes[start..].cmp(&needle[..k]) {
            Ordering::Equal => return start,
            _ => continue,
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
            stop_at: vec![b"<|im_end|>".to_vec()],
            holdback_prefixes: Vec::new(),
        }
    }

    fn cfg_strip_think() -> EosFilterConfig {
        EosFilterConfig {
            strip_think: true,
            stop_at: Vec::new(),
            holdback_prefixes: Vec::new(),
        }
    }

    fn cfg_gemma4_eot() -> EosFilterConfig {
        // Mirrors the Gemma 4 daemon path: literal '<end_of_turn>' is
        // a stop marker, and any prefix of it must be held back so
        // false-prefix bytes (e.g. '<en' followed by something else)
        // can flush correctly.
        EosFilterConfig {
            strip_think: false,
            stop_at: vec![b"<end_of_turn>".to_vec()],
            holdback_prefixes: vec![b"<end_of_turn>".to_vec()],
        }
    }

    #[test]
    fn empty_input_with_empty_state_holds() {
        let mut f = EosFilter::new(cfg_default());
        // The pre-extraction daemon behavior on zero-byte tokens was to
        // skip the JSON `{"type":"token",...}` frame entirely. Match it
        // with Hold so the caller never emits an empty payload.
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
        // Three-byte codepoint U+1F600 is four bytes in UTF-8 (😀).
        // Feed in two halves; the first must Hold, the second must
        // Emit the full codepoint.
        let mut f = EosFilter::new(cfg_default());
        let smile = "😀".as_bytes();
        assert_eq!(smile.len(), 4);
        // First two bytes — incomplete codepoint — Hold.
        assert_eq!(f.observe(&smile[..2]), FilterAction::Hold);
        // Remaining two bytes — Emit the full 4-byte codepoint.
        assert_eq!(
            f.observe(&smile[2..]),
            FilterAction::Emit(smile.to_vec())
        );
    }

    #[test]
    fn think_open_holds_until_close() {
        let mut f = EosFilter::new(cfg_strip_think());
        // Pre-think prose flushes immediately.
        assert_eq!(
            f.observe(b"hello "),
            FilterAction::Emit(b"hello ".to_vec())
        );
        // Opening tag + reasoning content — held.
        assert_eq!(f.observe(b"<think>reasoning"), FilterAction::Hold);
        assert_eq!(f.observe(b" more"), FilterAction::Hold);
        // Closing tag + post-answer — only post-answer flushes.
        match f.observe(b"</think>answer") {
            FilterAction::Emit(bytes) => assert_eq!(bytes, b"answer"),
            other => panic!("expected Emit(\"answer\"), got {:?}", other),
        }
    }

    #[test]
    fn close_think_alone_resumes_emit() {
        let mut f = EosFilter::new(cfg_strip_think());
        assert_eq!(f.observe(b"<think>x"), FilterAction::Hold);
        // Closer in its own observe call.
        assert_eq!(f.observe(b"</think>"), FilterAction::Hold);
        // Subsequent prose must flow normally.
        assert_eq!(
            f.observe(b" world"),
            FilterAction::Emit(b" world".to_vec())
        );
    }

    #[test]
    fn stop_at_full_match_returns_stop() {
        let mut f = EosFilter::new(cfg_im_end());
        assert_eq!(
            f.observe(b"hi"),
            FilterAction::Emit(b"hi".to_vec())
        );
        assert_eq!(f.observe(b"<|im_end|>"), FilterAction::Stop);
    }

    #[test]
    fn partial_holdback_prefix_holds_then_flushes_on_false_match() {
        // Gemma 4 false-prefix case from commit 7f37b99: bytes that
        // *look* like the start of '<end_of_turn>' must be held until
        // the next token confirms or denies the match.
        let mut f = EosFilter::new(cfg_gemma4_eot());
        // Feed a partial prefix '<en' — must hold.
        assert_eq!(f.observe(b"<en"), FilterAction::Hold);
        // Next token is something else: 'glish'. The held '<en' is
        // now disambiguated as not-a-stop-marker, so the combined
        // 'english' must flush in this observe.
        match f.observe(b"glish") {
            FilterAction::Emit(bytes) => assert_eq!(bytes, b"<english"),
            other => panic!("expected Emit('<english'), got {:?}", other),
        }
    }

    #[test]
    fn partial_holdback_prefix_then_full_match_stops() {
        let mut f = EosFilter::new(cfg_gemma4_eot());
        assert_eq!(f.observe(b"<en"), FilterAction::Hold);
        // The continuation completes the marker — Stop.
        assert_eq!(f.observe(b"d_of_turn>"), FilterAction::Stop);
    }

    #[test]
    fn reset_clears_state() {
        let mut f = EosFilter::new(cfg_strip_think());
        assert_eq!(f.observe(b"<think>"), FilterAction::Hold);
        // Without reset the next bytes would still be held.
        f.reset();
        // After reset, behaves as freshly constructed.
        assert_eq!(
            f.observe(b"clean"),
            FilterAction::Emit(b"clean".to_vec())
        );
    }

    #[test]
    fn flush_pending_drains_held_utf8_bytes() {
        // When the caller breaks on a token-level stop signal but the
        // filter still has half a UTF-8 codepoint buffered, the caller
        // can call flush_pending to drain. (In practice the held
        // bytes are a half-codepoint and would render as REPLACEMENT
        // CHARACTER; flush_pending exposes them so the caller can
        // decide what to do.)
        let mut f = EosFilter::new(cfg_default());
        let smile = "😀".as_bytes();
        assert_eq!(f.observe(&smile[..2]), FilterAction::Hold);
        let drained = f.flush_pending();
        assert_eq!(drained, &smile[..2]);
        // After flush, has_pending must be false.
        assert!(!f.has_pending());
    }

    #[test]
    fn stop_at_spanning_two_tokens_stops() {
        // The marker straddles two observe calls. Must still trip.
        let mut f = EosFilter::new(cfg_im_end());
        // Half of the marker. The trailing bytes here are a prefix of
        // a stop_at sequence, so they must be held back, not emitted
        // as plain text.
        assert_eq!(f.observe(b"<|im_"), FilterAction::Hold);
        assert_eq!(f.observe(b"end|>"), FilterAction::Stop);
    }
}
