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
//! - `FilterAction::Stop { emit }` — generation should stop. The stop
//!   marker is discarded and `emit` contains safe visible bytes before it.
//!
//! Construction normalizes the configuration and allocates the bounded
//! stop-marker state. The filter is `Send` and stateless across requests
//! after `reset()`.

use std::cmp::Ordering;

use crate::stop_quarantine::{QuarantineOutcome, StopQuarantine};

/// Output action emitted by `EosFilter::observe`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterAction {
    /// Emit these bytes to the consumer.
    Emit(Vec<u8>),
    /// Hold these bytes; the filter is buffering until the stream
    /// disambiguates (e.g. partial marker prefix that may or may not
    /// be a stop token, or bytes inside an active `<think>` block).
    Hold,
    /// Generation should stop. The stop marker and following bytes are
    /// discarded; `emit` contains safe visible bytes before it.
    Stop { emit: Vec<u8> },
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
    /// Use for non-stop downstream patterns. Prefixes of `stop_at` markers
    /// are already quarantined upstream and are removed from this list.
    /// On a false match, the buffered bytes are flushed (Emit).
    pub holdback_prefixes: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Default)]
struct EosFilterState {
    /// Bytes accumulated by the downstream stage since the last emitted
    /// prefix. Includes UTF-8 boundary safety and in-flight `<think>`
    /// content. Cleared by `reset()`.
    buf: Vec<u8>,
    /// True while we are inside a `<think>...</think>` block and
    /// `strip_think` is on. Set when the opener is seen, cleared on
    /// the closer.
    in_think: bool,
    /// Number of bytes already returned to the caller (in Emit
    /// actions). Used to compute the "new emit" delta on each call.
    emitted: usize,
    /// Lifecycle state. Stopped retains safe visible bytes for the final
    /// flush; Finished is inert until reset.
    terminal: EosFilterTerminal,
    /// Offset at which the current stripped think block starts.
    think_start: Option<usize>,
    /// Scan cursor for the current stripped think block. Unlike
    /// `emitted`, this advances through hidden reasoning content.
    think_scan: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EosFilterTerminal {
    Open,
    Stopped,
    Finished,
}

impl Default for EosFilterTerminal {
    fn default() -> Self {
        Self::Open
    }
}

/// Per-request output-stream filter. Construct from a
/// `EosFilterConfig` once per generation; feed each token's freshly
/// decoded bytes to `observe`. Reset between conversations / requests.
pub struct EosFilter {
    config: EosFilterConfig,
    quarantine: StopQuarantine,
    state: EosFilterState,
}

impl EosFilter {
    /// Construct from a config. The empty default (`strip_think=false`,
    /// no `stop_at`, no `holdback_prefixes`) is the master daemon's
    /// pre-extraction behavior: a UTF-8-boundary-safe pass-through.
    pub fn new(config: EosFilterConfig) -> Self {
        // Sort holdback_prefixes longest-first so prefix-match scans
        // pick the longest matching prefix, not the first one.
        let mut config = config;
        config
            .holdback_prefixes
            .sort_by(|a, b| b.len().cmp(&a.len()));
        config.holdback_prefixes.retain(|prefix| {
            !config
                .stop_at
                .iter()
                .any(|marker| !prefix.is_empty() && marker.starts_with(prefix))
        });
        Self {
            quarantine: StopQuarantine::new(config.stop_at.clone()),
            config,
            state: EosFilterState::default(),
        }
    }

    /// Reset between turns / requests. After this, the filter behaves
    /// as if freshly constructed from the same config.
    pub fn reset(&mut self) {
        self.quarantine.reset();
        self.state = EosFilterState::default();
    }

    /// Whether the filter currently has buffered bytes that have not
    /// been emitted. Useful for decisions like "did we drop content?"
    /// at end-of-stream. Unresolved raw scanner bytes count as pending.
    pub fn has_pending(&self) -> bool {
        self.quarantine.has_pending() || self.state.emitted < self.state.buf.len()
    }

    /// Finalize the stream and drain safe visible bytes held back due to
    /// UTF-8 or marker-prefix buffering. EOF disambiguates ordinary
    /// `holdback_prefixes`; incomplete UTF-8 and open `<think>` content are
    /// discarded. A false stop-marker prefix is recovered. Intended for use
    /// at end-of-stream; repeated calls are idempotent and return no bytes
    /// after the first call.
    pub fn finish(&mut self) -> Vec<u8> {
        if self.state.terminal != EosFilterTerminal::Open {
            return Vec::new();
        }

        let mut pending = Vec::new();
        let raw_pending = self.quarantine.finish();
        if !raw_pending.is_empty() {
            if let FilterAction::Emit(bytes) = self.observe_downstream(&raw_pending) {
                pending.extend(bytes);
            }
        }
        pending.extend(self.finish_visible_pending());
        self.state.terminal = EosFilterTerminal::Finished;
        pending
    }

    /// Feed newly-decoded bytes from a single token. Returns the next
    /// action.
    ///
    /// State machine, per call:
    /// 1. Prepend the bounded unresolved raw suffix from the previous call.
    /// 2. Scan that raw candidate for a complete stop marker. If found,
    ///    forward only the raw prefix before it downstream and return Stop.
    /// 3. Otherwise retain the longest unresolved stop-marker prefix and
    ///    forward the resolved raw prefix downstream.
    /// 4. The downstream stage computes the maximal "safe" emit prefix:
    ///    - It must end on a UTF-8 codepoint boundary.
    ///    - Its tail must not match any `holdback_prefix`.
    ///    Anything after that point stays buffered.
    /// 5. Return `Emit(prefix)` if non-empty, else `Hold`.
    pub fn observe(&mut self, raw_bytes: &[u8]) -> FilterAction {
        if self.state.terminal != EosFilterTerminal::Open {
            return FilterAction::Hold;
        }
        if raw_bytes.is_empty() && !self.quarantine.has_pending() && self.state.buf.is_empty() {
            // Nothing in flight. Pre-existing daemon behavior on
            // zero-byte tokens (e.g. "decode_bytes returned empty")
            // was to emit nothing — match it with Hold so the caller
            // does not write a JSON token frame for an empty payload.
            return FilterAction::Hold;
        }

        match self.quarantine.push(raw_bytes) {
            QuarantineOutcome::Stop { bytes } => {
                let mut emit = match self.observe_downstream(&bytes) {
                    FilterAction::Emit(bytes) => bytes,
                    FilterAction::Hold => Vec::new(),
                    FilterAction::Stop { .. } => unreachable!("stop scanner owns stop markers"),
                };
                emit.extend(self.finish_visible_pending());
                self.state.terminal = EosFilterTerminal::Stopped;
                FilterAction::Stop { emit }
                }
            QuarantineOutcome::Continue { bytes } => self.observe_downstream(&bytes),
                }
            }

    fn observe_downstream(&mut self, raw_bytes: &[u8]) -> FilterAction {
        if raw_bytes.is_empty() {
            return FilterAction::Hold;
        }
        self.state.buf.extend_from_slice(raw_bytes);

        if self.config.strip_think {
            self.advance_think_state();
        }

        if self.state.emitted > self.state.buf.len() {
            self.state.emitted = self.state.buf.len();
        }
        if self.state.in_think {
            return FilterAction::Hold;
        }

        let safe_end = self.compute_safe_end();
        if safe_end > self.state.emitted {
            let out = self.state.buf[self.state.emitted..safe_end].to_vec();
            self.state.emitted = safe_end;
            FilterAction::Emit(out)
        } else {
            FilterAction::Hold
        }
    }

    fn finish_visible_pending(&mut self) -> Vec<u8> {
        let visible_end = if self.state.in_think {
            self.state
                .think_start
                .unwrap_or(self.state.emitted)
                .min(self.state.buf.len())
        } else {
            self.state.buf.len()
        };
        let safe_end = self.compute_finish_end_for(visible_end);
        let pending = if safe_end > self.state.emitted {
            self.state.buf[self.state.emitted..safe_end].to_vec()
        } else {
            Vec::new()
        };
        self.state.emitted = self.state.buf.len();
        self.state.in_think = false;
        self.state.think_start = None;
        self.state.think_scan = None;
        pending
    }

    fn compute_finish_end_for(&self, hi: usize) -> usize {
        let buf = &self.state.buf;
        let lo = self.state.emitted;
        let hi = hi.min(buf.len());
        if lo >= hi {
            return lo;
        }

        let mut end = utf8_safe_end(&buf[lo..hi]) + lo;
        if self.config.strip_think {
            const OPEN: &[u8] = b"<think>";
            let max_overlap = (end - lo).min(OPEN.len() - 1);
            for len in (1..=max_overlap).rev() {
                if buf[end - len..end] == OPEN[..len] {
                    end -= len;
                    break;
                }
            }
        }
        end
    }

    /// Internal: advance the think scan cursor and toggle
    /// `state.in_think` based on the buffer contents. Called only when
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
                let scan = self.state.think_scan.unwrap_or(self.state.buf.len());
                if let Some(idx) = memmem(&self.state.buf[scan..], CLOSE) {
                    // Remove hidden reasoning and the closer while
                    // keeping any visible prefix before the opener.
                    let close_start = scan + idx;
                    let think_start = self.state.think_start.unwrap_or(close_start);
                    self.state.buf.drain(think_start..close_start + CLOSE.len());
                    self.state.in_think = false;
                    self.state.think_start = None;
                    self.state.think_scan = None;
                    continue;
                } else {
                    // No complete closer yet. Advance only the hidden
                    // scan cursor, never the output cursor: visible
                    // bytes before the opener may still need flushing.
                    let cut = trailing_prefix_start(&self.state.buf[scan..], CLOSE);
                    self.state.think_scan = Some(scan + cut);
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
                    self.state
                        .buf
                        .drain(opener_start..opener_start + OPEN.len());
                    // Mark the think state. `emitted` does not move:
                    // the pre-opener bytes are still pending.
                    self.state.in_think = true;
                    self.state.think_start = Some(opener_start);
                    self.state.think_scan = Some(opener_start);
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
    ///   non-stop `holdback_prefix` or, if `strip_think` is on, of
    ///   `<think>` (which would otherwise leak the start of an opener).
    fn compute_safe_end(&self) -> usize {
        self.compute_safe_end_for(self.state.buf.len())
    }

    fn compute_safe_end_for(&self, hi: usize) -> usize {
        let buf = &self.state.buf;
        let lo = self.state.emitted;
        let hi = hi.min(buf.len());
        if lo >= hi {
            return lo;
        }

        // Start by trimming back to a UTF-8 boundary.
        let mut end = utf8_safe_end(&buf[lo..hi]) + lo;

        // Trim back further if the trailing bytes match a non-empty
        // prefix of any downstream holdback / think-opener pattern.
        let mut watch_prefixes: Vec<&[u8]> = Vec::new();
        for p in &self.config.holdback_prefixes {
            if !p.is_empty() {
                watch_prefixes.push(p.as_slice());
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
        // a stop marker. Keep the legacy duplicate prefix here to verify
        // EosFilter removes stop-owned holdback entries at construction.
        EosFilterConfig {
            strip_think: false,
            stop_at: vec![b"<end_of_turn>".to_vec()],
            holdback_prefixes: vec![b"<end_of_turn>".to_vec()],
        }
    }

    #[test]
    fn stop_marker_prefixes_are_not_reheld_downstream() {
        let filter = EosFilter::new(EosFilterConfig {
            strip_think: false,
            stop_at: vec![b"<stop>".to_vec()],
            holdback_prefixes: vec![b"<st".to_vec(), b"<partial>".to_vec()],
        });
        assert_eq!(filter.config.holdback_prefixes, vec![b"<partial>".to_vec()]);
    }

    fn drain_chunks(config: EosFilterConfig, input: &[u8], cuts: u32) -> (Vec<u8>, bool) {
        let mut filter = EosFilter::new(config);
        let mut emitted = Vec::new();
        let mut offset = 0;
        for end in 1..=input.len() {
            if cuts & (1 << (end - 1)) != 0 {
                match filter.observe(&input[offset..end]) {
                    FilterAction::Emit(bytes) => emitted.extend(bytes),
                    FilterAction::Hold => {}
                    FilterAction::Stop { emit } => {
                        emitted.extend(emit);
                        return (emitted, true);
                    }
                }
                offset = end;
            }
        }
        if offset < input.len() {
            match filter.observe(&input[offset..]) {
                FilterAction::Emit(bytes) => emitted.extend(bytes),
                FilterAction::Hold => {}
                FilterAction::Stop { emit } => {
                    emitted.extend(emit);
                    return (emitted, true);
                }
            }
        }
        emitted.extend(filter.finish());
        (emitted, false)
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
        assert_eq!(f.observe(&smile[2..]), FilterAction::Emit(smile.to_vec()));
    }

    #[test]
    fn think_open_holds_until_close() {
        let mut f = EosFilter::new(cfg_strip_think());
        // Pre-think prose flushes immediately.
        assert_eq!(f.observe(b"hello "), FilterAction::Emit(b"hello ".to_vec()));
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
        assert_eq!(f.observe(b" world"), FilterAction::Emit(b" world".to_vec()));
    }

    #[test]
    fn stop_at_full_match_returns_stop() {
        let mut f = EosFilter::new(cfg_im_end());
        assert_eq!(f.observe(b"hi"), FilterAction::Emit(b"hi".to_vec()));
        assert_eq!(
            f.observe(b"<|im_end|>"),
            FilterAction::Stop { emit: Vec::new() }
        );
    }

    #[test]
    fn stop_discards_marker_but_preserves_safe_visible_prefix_once() {
        let mut f = EosFilter::new(cfg_im_end());
        assert_eq!(
            f.observe(b"safe<|im_end|>"),
            FilterAction::Stop {
                emit: b"safe".to_vec()
            }
        );
        assert_eq!(f.finish(), Vec::<u8>::new());
        assert_eq!(f.finish(), Vec::<u8>::new());
        assert_eq!(f.observe(b"after-stop"), FilterAction::Hold);
    }

    #[test]
    fn stop_inside_think_discards_hidden_bytes() {
        let mut f = EosFilter::new(EosFilterConfig {
            strip_think: true,
            stop_at: vec![b"<|im_end|>".to_vec()],
            holdback_prefixes: Vec::new(),
        });
        assert_eq!(
            f.observe(b"safe<think>hidden<|im_end|></think>"),
            FilterAction::Stop {
                emit: b"safe".to_vec()
            }
        );
        assert_eq!(f.finish(), Vec::<u8>::new());
    }

    #[test]
    fn literal_marker_inside_reasoning_stops_at_raw_position() {
        let mut f = EosFilter::new(EosFilterConfig {
            strip_think: true,
            stop_at: vec![b"<stop>".to_vec()],
            holdback_prefixes: Vec::new(),
        });
        assert_eq!(
            f.observe(b"visible<think>hidden<stop>after"),
            FilterAction::Stop {
                emit: b"visible".to_vec()
            }
        );
    }

    #[test]
    fn marker_after_prior_stripped_reasoning_stops_normally() {
        let mut f = EosFilter::new(EosFilterConfig {
            strip_think: true,
            stop_at: vec![b"<stop>".to_vec()],
            holdback_prefixes: Vec::new(),
        });
        assert_eq!(
            f.observe(b"before<think>hidden</think>after"),
            FilterAction::Emit(b"beforeafter".to_vec())
        );
        assert_eq!(
            f.observe(b"<stop>suffix"),
            FilterAction::Stop { emit: Vec::new() }
        );
        assert_eq!(f.finish(), Vec::<u8>::new());
    }

    #[test]
    fn reasoning_stripping_does_not_synthesize_stop_marker() {
        let mut f = EosFilter::new(EosFilterConfig {
            strip_think: true,
            stop_at: vec![b"ab".to_vec()],
            holdback_prefixes: Vec::new(),
        });
        // The bytes forming `ab` are separated by stripped reasoning in the
        // literal input, so they must not become a stop marker after stripping.
        assert_eq!(
            f.observe(b"a<think>hidden</think>b"),
            FilterAction::Emit(b"ab".to_vec())
        );
        assert_eq!(f.observe(b"c"), FilterAction::Emit(b"c".to_vec()));
        assert!(f.finish().is_empty());
    }

    #[test]
    fn identical_start_stop_markers_choose_longest() {
        let mut f = EosFilter::new(EosFilterConfig {
            strip_think: false,
            stop_at: vec![b"ab".to_vec(), b"abc".to_vec()],
            holdback_prefixes: Vec::new(),
        });
        assert_eq!(
            f.observe(b"safeabc-tail"),
            FilterAction::Stop {
                emit: b"safe".to_vec()
            }
        );
    }

    #[test]
    fn earliest_stop_marker_wins_across_multiple_patterns() {
        let mut f = EosFilter::new(EosFilterConfig {
            strip_think: false,
            stop_at: vec![b"!".to_vec(), b"<long>".to_vec()],
            holdback_prefixes: Vec::new(),
        });
        assert_eq!(
            f.observe(b"safe!leak<long>"),
            FilterAction::Stop {
                emit: b"safe".to_vec()
            }
        );
        // The later, longer marker must not cause `!leak` to be retained or
        // leak either marker during finalization.
        assert_eq!(f.finish(), Vec::<u8>::new());
    }

    #[test]
    fn observe_is_silent_while_stopped_before_finish() {
        let mut f = EosFilter::new(cfg_im_end());
        assert_eq!(
            f.observe(b"safe<|im_end|>"),
            FilterAction::Stop {
                emit: b"safe".to_vec()
            }
        );
        assert_eq!(f.observe(b"after-stop"), FilterAction::Hold);
        assert_eq!(f.finish(), Vec::<u8>::new());
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
        assert_eq!(
            f.observe(b"d_of_turn>"),
            FilterAction::Stop { emit: Vec::new() }
        );
    }

    #[test]
    fn finish_flushes_ordinary_holdback_prefix() {
        let mut f = EosFilter::new(EosFilterConfig {
            strip_think: false,
            stop_at: Vec::new(),
            holdback_prefixes: vec![b"<partial>".to_vec()],
        });
        assert_eq!(
            f.observe(b"visible<part"),
            FilterAction::Emit(b"visible".to_vec())
        );
        assert_eq!(f.finish(), b"<part");
    }

    #[test]
    fn exhaustive_splits_stop_on_overlapping_markers() {
        let input = b"safeabab-tail";
        let config = EosFilterConfig {
            strip_think: false,
            stop_at: vec![b"aba".to_vec(), b"bab".to_vec()],
            holdback_prefixes: Vec::new(),
        };
        let split_count = input.len() - 1;
        for cuts in 0..(1u32 << split_count) {
            assert_eq!(
                drain_chunks(config.clone(), input, cuts),
                (b"safe".to_vec(), true),
                "cuts={cuts:#b}"
            );
        }
    }

    #[test]
    fn exhaustive_splits_finish_false_stop_prefix_when_valid_utf8() {
        let input = b"visible<en";
        let config = cfg_gemma4_eot();
        let split_count = input.len() - 1;
        for cuts in 0..(1u32 << split_count) {
            assert_eq!(
                drain_chunks(config.clone(), input, cuts),
                (input.to_vec(), false),
                "cuts={cuts:#b}"
            );
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut f = EosFilter::new(cfg_strip_think());
        assert_eq!(f.observe(b"<think>"), FilterAction::Hold);
        // Without reset the next bytes would still be held.
        f.reset();
        // After reset, behaves as freshly constructed.
        assert_eq!(f.observe(b"clean"), FilterAction::Emit(b"clean".to_vec()));
    }

    #[test]
    fn finish_discards_incomplete_utf8_bytes() {
        let mut f = EosFilter::new(cfg_default());
        let smile = "😀".as_bytes();
        assert_eq!(f.observe(&smile[..2]), FilterAction::Hold);
        let drained = f.finish();
        assert!(drained.is_empty());
        // After finish, has_pending must be false.
        assert!(!f.has_pending());
    }

    #[test]
    fn finish_emits_valid_unresolved_marker_suffix() {
        let mut f = EosFilter::new(cfg_gemma4_eot());
        assert_eq!(
            f.observe(b"visible<en"),
            FilterAction::Emit(b"visible".to_vec())
        );
        assert_eq!(f.finish(), b"<en");
        assert_eq!(f.finish(), Vec::<u8>::new());
    }

    #[test]
    fn finish_discards_open_think_content_but_keeps_visible_prefix() {
        let mut f = EosFilter::new(cfg_strip_think());
        assert_eq!(f.observe(b"visible<think>secret"), FilterAction::Hold);
        assert_eq!(f.finish(), b"visible");
        assert_eq!(f.finish(), Vec::<u8>::new());
    }

    #[test]
    fn reset_reopens_a_stopped_or_finished_filter() {
        let mut f = EosFilter::new(cfg_im_end());
        assert_eq!(
            f.observe(b"<|im_end|>"),
            FilterAction::Stop { emit: Vec::new() }
        );
        assert!(f.finish().is_empty());
        f.reset();
        assert_eq!(f.observe(b"open"), FilterAction::Emit(b"open".to_vec()));

        let mut f = EosFilter::new(cfg_default());
        assert_eq!(f.observe(b"held<"), FilterAction::Emit(b"held<".to_vec()));
        assert!(f.finish().is_empty());
        f.reset();
        assert_eq!(
            f.observe(b"open-again"),
            FilterAction::Emit(b"open-again".to_vec())
        );
    }

    #[test]
    fn stop_at_spanning_two_tokens_stops() {
        // The marker straddles two observe calls. Must still trip.
        let mut f = EosFilter::new(cfg_im_end());
        // Half of the marker. The trailing bytes here are a prefix of
        // a stop_at sequence, so they must be held back, not emitted
        // as plain text.
        assert_eq!(f.observe(b"<|im_"), FilterAction::Hold);
        assert_eq!(f.observe(b"end|>"), FilterAction::Stop { emit: Vec::new() });
    }

    #[test]
    fn finish_flushes_unresolved_scanner_suffix() {
        let mut f = EosFilter::new(cfg_im_end());
        assert_eq!(
            f.observe(b"safe<|im_"),
            FilterAction::Emit(b"safe".to_vec())
        );
        assert_eq!(f.finish(), b"<|im_");
        assert_eq!(f.finish(), Vec::<u8>::new());
    }
}
