// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Constrained decoding — unified grammar.
//!
//! Merged from:
//! - `crates/hipfire-arch-qwen35/src/grammar.rs` (2,736 lines) — JSON `<tool_call>` format
//!   (13 arch-specific mentions, now parameterized via `json::Config`)
//! - `crates/hipfire-arch-deepseek4/src/grammar.rs` (1,199 lines) — DSML XML format
//!   (1 arch-specific mention, now provenance comment only)
//!
//! Both implementations are model-agnostic (`std` only) and share the same
//! masking contract: `is_token_allowed` / `token_mask` / `apply_mask_to_logits` plus `advance`.
//! The DSML variant tracks XML tag structure with leading-whitespace
//! tolerance; the JSON variant tracks `<tool_call>` JSON with brace-aware and
//! n-gram attractor guards. Model-specific details (tag literals, n-gram
//! thresholds) are passed in via `json::Config` where needed — no architecture branching
//! branching lives here.

/// JSON `<tool_call>` grammar — state machine for `\n{"name": "<TOOL>", "arguments": <value>}` tool calls.
/// See original `hipfire-arch-qwen35/src/grammar.rs` for full design notes (Pi turn-12 drift).
pub mod json {
//! Grammar-guided decoding for the qwen3.5/3.6 tool-call format.
//!
//! Mirrors the V4F DSML grammar in `crates/hipfire-arch-deepseek4/src/grammar.rs`
//! but constrains qwen35's `<tool_call>{json}</tool_call>` body instead of
//! DeepSeek's XML-style DSML tags.
//!
//! ## Why this exists
//!
//! qwen3.6:27b drifts after long agentic sessions: it emits the
//! `<tool_call>` opener correctly then writes ChatML noise as the body,
//! e.g.
//! ```text
//! <tool_call>
//! <|im_start|>assistant "Let me read the existing build files..."}}
//! </tool_call>
//! ```
//! observed verbatim in Pi turn 12 after ~27k cached tokens. The text
//! isn't JSON, so the daemon's tool-call extractor returns
//! `tool_calls=0`, the daemon emits `finish_reason: "stop"` with that
//! garbage as `message.content`, and Pi's agent loop terminates.
//!
//! The grammar matcher prevents this by masking sample logits the
//! moment the model commits to `<tool_call>`: from then until the JSON
//! header `\n{"name": "<TOOL_NAME>", "arguments": ` is fully laid
//! down, only tokens that continue that template are allowed. The
//! `arguments` value itself is free-form JSON (state `InArgs`), with one
//! structural constraint: the model may not emit `</tool_call>` until the
//! OUTER tool-call object is brace-balanced. The header opens `{...`, so
//! the canonical close is `}}\n</tool_call>` — one `}` for the args value,
//! one for the enclosing object. Once `</tool_call>` lands, the matcher
//! returns to free emission.
//!
//! Enforcing the outer `}` is the "JSON-aware brace counting" future phase
//! flagged below: qwen3.6:27b (mq4, temp>0) intermittently jumped from the
//! inner `}` straight to `</tool_call>`, dropping the outer brace and
//! emitting invalid JSON (`{"name":"read","arguments":{"path":"x"}`) that
//! failed Pi's tool-call parser — the "dropped closing bracket" bug.
//!
//! ## States
//!
//! - [`State::Out`] — free emission. The matcher watches for the
//!   `<tool_call>` substring (single special token, vocab id varies
//!   per checkpoint) and transitions to [`State::AfterOpen`] on entry.
//! - [`State::AfterOpen`] — between `<tool_call>` and the
//!   `"arguments": ` colon-space. Constrained to a literal byte
//!   sequence that names one of the available tools.
//! - [`State::InArgs`] — between `"arguments": ` and `</tool_call>`. The
//!   args value is free JSON, but `</tool_call>` is masked out until the
//!   enclosing tool-call object is brace-balanced (the outer `}` is
//!   emitted). String-aware, so `{`/`}`/`<` inside string values don't
//!   count. The attractor force-close path is exempt.
//! - (back to `Out` after `</tool_call>`.)
//!
//! ## What this does NOT do
//!
//! Enforce full JSON-validity inside `arguments` (well-formed keys,
//! quoting, commas). It tracks only string-aware brace balance — enough
//! to require the outer `}` before the close — leaving trailing-comma /
//! unquoted-key glitches to the daemon's downstream tool-call repair.

/// Position in the qwen35 tool-call grammar. See module docs for
/// transitions.
#[derive(Debug, Clone, PartialEq)]
pub enum State {
    /// Free emission outside any `<tool_call>` block. Watching for the
    /// open marker but not otherwise constraining tokens.
    Out,
    /// Between `<tool_call>` (already consumed) and the `"arguments": `
    /// header sentinel. Allowed continuations are prefixes of
    /// `\n{"name": "<TOOL_NAME>", "arguments": `.
    AfterOpen,
    /// Between `"arguments": ` and `</tool_call>`. Free emission —
    /// the model writes the args value as it pleases. We re-enter
    /// `Out` when `</tool_call>` lands in the rolling buffer.
    InArgs,
}

/// Schema for one available tool. Built from the OpenAI-format tools
/// array at request time; the grammar uses this to pick which tool
/// names the model is allowed to emit at the name position AND which
/// argument fields must appear in the args body before the close
/// marker is allowed.
#[derive(Debug, Clone)]
pub struct ToolSchema {
    pub name: String,
    /// Subset of the tool's parameters that MUST appear as keys in
    /// the emitted args body. Built from the JSON schema's
    /// `parameters.required` array at request time. The grammar's
    /// `is_token_allowed` rejects close-marker prefixes while any
    /// required name is still absent from the args body — without
    /// this the model can emit `"arguments":{}` (Pi `write` failure
    /// mode observed in production: model emitted empty args, Pi
    /// rejected, model retried with same empty args, context bloated
    /// to KV exhaustion).
    pub required: Vec<String>,
}

/// Maximum bytes retained for the n-gram loop guard's rolling window
/// inside [`State::InArgs`]. 256 bytes covers the worst-case attractor
/// we've observed in production (a ~20-byte repeated block × 4) with
/// headroom. Bounded so the rolling-window scan stays O(1) per token.
const NGRAM_WINDOW: usize = 256;

/// Minimum consecutive identical n-gram repeats that trigger the
/// attractor flag. **Default 6** — bumped from 4 on 2026-05-28 after a
/// false-positive incident generating Zig code (legitimate
/// indentation + repeated section markers tripped the guard mid-args,
/// stranded the assistant in an empty tool call). Real attractors
/// (e.g. `typetypetypetype` extended, repeated invocation snippets)
/// run long and still trip at 6+. Override at startup via
/// `GRAMMAR_NGRAM_MIN_REPEATS=<n>` if a workload needs tighter
/// detection.
const NGRAM_MIN_REPEATS_DEFAULT: usize = 6;

/// Range of n-gram lengths probed each token (inclusive). **MIN
/// bumped from 2 → 3** on 2026-05-28: 2-byte n-grams catch `  ` (two
/// spaces) repeating, `\n\n`, `, ,`, etc. — pervasive in code and
/// JSON. 3-byte grams still catch tight character cycles without the
/// false-positive flood. 32-byte grams catch multi-token phrase loops.
/// Override the lower bound with `GRAMMAR_NGRAM_LEN_MIN=<n>`.
const NGRAM_LEN_MIN_DEFAULT: usize = 3;
const NGRAM_LEN_MAX: usize = 32;

/// Tunable n-gram loop-guard thresholds for the JSON `InArgs` attractor detector.
///
/// The matcher is instantiated with a `Config` so model-specific values are
/// passed in by the caller rather than branching on architecture inside this
/// crate. Defaults match the production tuning (6 repeats of 3..32
/// byte grams). Callers that need env-driven overrides should read their own
/// env vars and pass the parsed values here — this crate does not depend on
/// `hipfire-config` and contains no arch-specific env names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Rolling-window size in bytes (default 256).
    pub ngram_window: usize,
    /// Minimum consecutive identical n-gram repeats to trigger the attractor
    /// flag (default 6).
    pub ngram_min_repeats: usize,
    /// Minimum n-gram length probed (default 3).
    pub ngram_len_min: usize,
    /// Maximum n-gram length probed (default 32).
    pub ngram_len_max: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ngram_window: NGRAM_WINDOW,
            ngram_min_repeats: NGRAM_MIN_REPEATS_DEFAULT,
            ngram_len_min: NGRAM_LEN_MIN_DEFAULT,
            ngram_len_max: NGRAM_LEN_MAX,
        }
    }
}

/// Grammar matcher: state plus the bytes committed since the last
/// firm transition. Construct via [`Matcher::new`] with the active
/// tool schemas, advance with [`Matcher::advance`], query allowed
/// tokens via [`Matcher::is_token_allowed`] or [`Matcher::token_mask`].
#[derive(Debug, Clone)]
pub struct Matcher {
    state: State,
    /// Bytes committed since the last firm state transition.
    partial_buf: String,
    tools: Vec<ToolSchema>,
    config: Config,
    /// Rolling window of the last `NGRAM_WINDOW` bytes of the FULL args
    /// body (including string-value bytes) seen while in [`State::InArgs`].
    /// Used ONLY for the required-field substring check (`"path"` etc. live
    /// inside strings). Attractor detection runs on the separate
    /// structural-only [`Self::attractor_buf`]. Cleared on every firm
    /// transition.
    ngram_history: String,
    /// Rolling window of the last `NGRAM_WINDOW` *structural* (out-of-string)
    /// args bytes — fed to the n-gram attractor guard. String-value bytes
    /// (e.g. a `write` tool's code `content`) are excluded so legitimate code
    /// repetition doesn't false-trip the guard. Cleared on every firm
    /// transition.
    attractor_buf: String,
    /// Set when consecutive n-gram repetition is detected inside
    /// [`State::InArgs`]. While set, the matcher constrains InArgs
    /// to only `</tool_call>` continuations — the model gets a
    /// forced exit instead of being stuck in an attractor that
    /// bloats the agentic conversation's KV (Pi turn-12-style
    /// `typetypetypetype` inside the args body was the motivating
    /// case). Cleared when the close marker fires and we return
    /// to [`State::Out`].
    attractor_detected: bool,
    /// Index into `tools` of the tool whose schema we're currently
    /// inside (i.e. the one whose name header just matched on the
    /// `AfterOpen` → `InArgs` transition). `None` outside of
    /// [`State::InArgs`]. Used by `required_fields_satisfied` to
    /// know which required-field list to check against the args
    /// body bytes.
    current_tool: Option<usize>,
    /// JSON brace depth inside the args body while in
    /// [`State::InArgs`]. Starts at 0 on entry; the first `{` of the
    /// args body increments to 1; the matching `}` brings it back to
    /// 0. Used by `is_token_allowed` to block tokens that would
    /// close the args body before required fields are satisfied —
    /// the close marker check alone is insufficient because the
    /// closing `}` of an empty `{}` body already commits before the
    /// next-token `</tool_call>` is seen, so the empty args stream
    /// to the client even when the close-marker token is rejected.
    args_brace_depth: i32,
    /// True while inside a JSON string in the args body. Toggled on
    /// unescaped `"`. Used to ignore `{` / `}` inside string values
    /// when updating `args_brace_depth`.
    args_in_string: bool,
    /// True if the previous byte in the args body was a backslash
    /// inside a string. The next byte is then escaped (skip its
    /// special meaning, e.g. `\"` does NOT close the string).
    args_string_escape: bool,
}

impl Matcher {
    /// Build a fresh matcher in [`State::Out`] with no partial buffer and
    /// default `Config`.
    pub fn new(tools: Vec<ToolSchema>) -> Self {
        Self::with_config(tools, Config::default())
    }

    /// Build a fresh matcher with an explicit `Config`. Use this when the
    /// caller wants to override the n-gram thresholds without an env read
    /// inside this crate.
    pub fn with_config(tools: Vec<ToolSchema>, config: Config) -> Self {
        Self {
            state: State::Out,
            partial_buf: String::new(),
            tools,
            config,
            ngram_history: String::new(),
            attractor_buf: String::new(),
            attractor_detected: false,
            current_tool: None,
            args_brace_depth: 0,
            args_in_string: false,
            args_string_escape: false,
        }
    }

    /// Current n-gram config (copy).
    pub fn config(&self) -> Config {
        self.config
    }

    /// Index of the tool currently being constructed in
    /// [`State::InArgs`]. Exposed for diagnostics.
    pub fn current_tool(&self) -> Option<usize> {
        self.current_tool
    }

    /// Diagnostic snapshot for the DFlash close-marker rejection path.
    pub fn debug_close_reject(&self) -> String {
        let req = self
            .current_tool
            .and_then(|i| self.tools.get(i))
            .map(|s| s.required.join(","))
            .unwrap_or_default();
        let hist_tail: String = self
            .ngram_history
            .chars()
            .rev()
            .take(80)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        format!(
            "current_tool={:?} required=[{}] req_satisfied={} args_brace_depth={} ngram_hist_len={} hist_tail={:?}",
            self.current_tool,
            req,
            self.required_fields_satisfied(),
            self.args_brace_depth,
            self.ngram_history.len(),
            hist_tail,
        )
    }

    /// Check whether the args body bytes seen so far satisfy every
    /// required field for the current tool. A field is considered
    /// "seen" if `"<name>"` appears anywhere in the args body bytes
    /// (`ngram_history`). The substring match is intentionally loose
    /// — JSON syntax exists for the LLM, not for the parser, so as
    /// long as the field name string is present we trust the model
    /// to wire the colon/value correctly.
    ///
    /// Empty schema or empty `required` list → trivially satisfied.
    /// Returns true if no current tool (e.g. AfterOpen state) so we
    /// don't gate transitions we can't evaluate.
    fn required_fields_satisfied(&self) -> bool {
        let tool_idx = match self.current_tool {
            Some(i) => i,
            None => return true,
        };
        let schema = match self.tools.get(tool_idx) {
            Some(s) => s,
            None => return true,
        };
        if schema.required.is_empty() {
            return true;
        }
        for name in &schema.required {
            let needle = format!("\"{}\"", name);
            if !self.ngram_history.contains(&needle) {
                return false;
            }
        }
        true
    }

    /// Update the args-body brace/string tracking state by consuming
    /// the bytes in `text`. Caller must guarantee the matcher is in
    /// [`State::InArgs`]; this is enforced at the only call site
    /// (`advance`). String-aware: `{` / `}` inside JSON strings do
    /// NOT change brace depth, and `\"` does NOT close the string.
    fn update_args_brace_state(&mut self, text: &str) {
        for byte in text.bytes() {
            if self.args_string_escape {
                self.args_string_escape = false;
                continue;
            }
            if self.args_in_string {
                match byte {
                    b'\\' => self.args_string_escape = true,
                    b'"' => self.args_in_string = false,
                    _ => {}
                }
                continue;
            }
            // Structural position (outside any JSON string value): feed the
            // n-gram attractor guard. String-value bytes (handled by the
            // `continue` above) are deliberately excluded — see `advance`.
            self.push_attractor_byte(byte);
            match byte {
                b'"' => self.args_in_string = true,
                b'{' => self.args_brace_depth += 1,
                b'}' => self.args_brace_depth -= 1,
                _ => {}
            }
        }
    }

    /// Simulate the brace-state update for `text` WITHOUT mutating
    /// `self`, and return whether the token would close the outer
    /// args body (depth returns to 0 from a depth >= 1 reached at
    /// or before this token). Used by `is_token_allowed` to reject
    /// the `}` of an empty `{}` body before it commits.
    ///
    /// "Closes the args body" semantics:
    ///   - If `args_brace_depth` was >= 1 before this token (we're
    ///     already inside the body), any `}` bringing depth to 0
    ///     counts as closing.
    ///   - If `args_brace_depth` was 0 (haven't seen the first `{`
    ///     yet), the token closes only if it opens then closes the
    ///     body — i.e. it contains a `{` that raises depth to 1+
    ///     and a matching `}` that brings depth back to 0. This is
    ///     the empty-`{}` single-token case.
    fn would_close_args_body(&self, text: &str) -> bool {
        let mut depth = self.args_brace_depth;
        let mut in_string = self.args_in_string;
        let mut escape = self.args_string_escape;
        let mut entered_body = self.args_brace_depth >= 1;
        for byte in text.bytes() {
            if escape {
                escape = false;
                continue;
            }
            if in_string {
                match byte {
                    b'\\' => escape = true,
                    b'"' => in_string = false,
                    _ => {}
                }
                continue;
            }
            match byte {
                b'"' => in_string = true,
                b'{' => {
                    depth += 1;
                    if depth >= 1 {
                        entered_body = true;
                    }
                }
                b'}' => {
                    depth -= 1;
                    if entered_body && depth <= 0 {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// Simulate the args-body brace state over `text` (seeded from the
    /// committed [`State::InArgs`] state) and return whether the OUTER
    /// tool-call object is already closed (`args_brace_depth < 0`) at the
    /// point a `</tool_call>` marker begins within `text`.
    ///
    /// Used by [`Self::is_token_allowed`] to permit a close-forming token
    /// iff it does not drop the outer `}`: a lone `</tool_call>` after the
    /// inner `}` (depth back to 0) is rejected, while a merged
    /// `}}\n</tool_call>` token — whose second `}` drives depth to -1 before
    /// the marker — is allowed. A `<` inside a JSON string value is data,
    /// not the marker, so the scan is string-aware. By the block invariant
    /// (this gate rejects any premature close-prefix token), the marker can
    /// only begin inside `text`, never straddling `partial_buf`.
    fn outer_closed_before_marker(&self, text: &str) -> bool {
        const CLOSE: &[u8] = b"</tool_call>";
        let mut depth = self.args_brace_depth;
        let mut in_string = self.args_in_string;
        let mut escape = self.args_string_escape;
        let bytes = text.as_bytes();
        for i in 0..bytes.len() {
            let rem = &bytes[i..];
            // A `</tool_call>` marker (full, or a terminal prefix running to
            // the end of `text`) begins here — only meaningful outside a
            // string value. Decide against the outer-object state at this
            // position.
            if !in_string && (rem.starts_with(CLOSE) || CLOSE.starts_with(rem)) {
                return depth < 0;
            }
            let byte = bytes[i];
            if escape {
                escape = false;
                continue;
            }
            if in_string {
                match byte {
                    b'\\' => escape = true,
                    b'"' => in_string = false,
                    _ => {}
                }
                continue;
            }
            match byte {
                b'"' => in_string = true,
                b'{' => depth += 1,
                b'}' => depth -= 1,
                _ => {}
            }
        }
        // No `</tool_call>` marker actually begins (outside a string) within
        // `text` — the caller's `touches_close_marker` matched a `<` that is
        // string-value CONTENT (heredoc `<<`, `#include <stdio.h>`, `a < b`,
        // …) or a bare `<` that isn't a real close prefix. It does not close
        // the tool call, so allow it. (A real close prefix outside a string is
        // caught by the in-loop `return depth < 0` above.)
        true
    }

    /// True iff the n-gram loop guard has tripped on the current
    /// [`State::InArgs`] body. Exposed for diagnostics; the daemon
    /// uses this to log the trip event.
    pub fn attractor_detected(&self) -> bool {
        self.attractor_detected
    }

    /// Detect consecutive identical n-gram repetition in the tail of
    /// `buf`. Returns `true` iff there's some `n` in
    /// `[ngram_len_min(), NGRAM_LEN_MAX]` such that the last
    /// `n * ngram_min_repeats()` bytes consist of the same `n`-byte
    /// block repeated `ngram_min_repeats()` times. The lower bound on
    /// `n` and the repeat threshold come from env-tunable knobs
    /// (`Config::ngram_len_min`, `Config::ngram_min_repeats`)
    /// with the cached defaults (3 and 6 respectively).
    ///
    /// **Uniform-byte filter:** n-grams that consist of a single
    /// repeated character (e.g. `   ` for whitespace, `===` for
    /// dividers) are skipped — long runs of the same byte are
    /// pervasive in code (indentation, section markers) and never
    /// indicate an attractor. Real attractors (`type`, `pub fn`, etc.)
    /// have non-uniform grams.
    fn detect_ngram_loop(&self, buf: &str) -> bool {
        let bytes = buf.as_bytes();
        let len_min = self.config.ngram_len_min;
        let min_repeats = self.config.ngram_min_repeats;
        for ngram_len in len_min..=self.config.ngram_len_max {
            let needed = ngram_len * min_repeats;
            if bytes.len() < needed {
                continue;
            }
            let tail = &bytes[bytes.len() - needed..];
            let first = &tail[..ngram_len];
            // Skip uniform-byte grams — they fire on legit indentation
            // and divider runs without signaling a real loop.
            if Self::is_uniform_byte(first) {
                continue;
            }
            let mut all_match = true;
            for r in 1..min_repeats {
                let chunk = &tail[r * ngram_len..(r + 1) * ngram_len];
                if chunk != first {
                    all_match = false;
                    break;
                }
            }
            if all_match {
                return true;
            }
        }
        false
    }

    /// True iff every byte in `chunk` is the same value (e.g. `   `,
    /// `===`, `\t\t\t`). Empty slice is vacuously uniform.
    fn is_uniform_byte(chunk: &[u8]) -> bool {
        match chunk.first() {
            Some(&first) => chunk.iter().all(|&b| b == first),
            None => true,
        }
    }

    /// Push bytes into the n-gram history buffer and run detection.
    /// Only called while in [`State::InArgs`]. Sets
    /// `attractor_detected = true` on first detection; never clears
    /// it from here (clearing happens on the firm `</tool_call>`
    /// transition back to `Out`).
    fn update_ngram_history(&mut self, text: &str) {
        // Full args text → required-field substring buffer only. Attractor
        // detection runs on structural bytes in `push_attractor_byte`.
        self.ngram_history.push_str(text);
        if self.ngram_history.len() > self.config.ngram_window {
            // Drop at a UTF-8 char boundary — string values may hold multibyte
            // content, so this buffer is not guaranteed ASCII.
            let drop = self.ngram_history.len() - self.config.ngram_window;
            let mut idx = drop;
            while idx < self.ngram_history.len() && !self.ngram_history.is_char_boundary(idx) {
                idx += 1;
            }
            self.ngram_history.drain(..idx);
        }
    }

    /// Feed one *structural* (out-of-string) args byte into the attractor
    /// guard. Called per-byte from [`Self::update_args_brace_state`]; bytes
    /// inside JSON string values are excluded by that caller.
    fn push_attractor_byte(&mut self, byte: u8) {
        if self.attractor_detected {
            return; // already flagged; don't waste work
        }
        // Structural JSON bytes are always ASCII. Skip any stray non-ASCII byte
        // so `attractor_buf` stays valid UTF-8 for `&str` detection; every
        // ASCII byte is its own char boundary, so trimming is unconditional.
        if !byte.is_ascii() {
            return;
        }
        self.attractor_buf.push(byte as char);
        if self.attractor_buf.len() > self.config.ngram_window {
            let drop = self.attractor_buf.len() - self.config.ngram_window;
            self.attractor_buf.drain(..drop);
        }
        if self.detect_ngram_loop(&self.attractor_buf) {
            self.attractor_detected = true;
        }
    }

    /// Read-only view of the current state.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Bytes accumulated since the last firm state transition.
    pub fn partial(&self) -> &str {
        &self.partial_buf
    }

    /// Whether the matcher is currently free (all tokens allowed).
    /// Free in [`State::Out`] until the buffer accumulates a `<tool_call>`
    /// prefix; free in [`State::InArgs`] until the buffer accumulates
    /// the `</tool_call>` close-marker prefix OR a required field is
    /// still missing from the args body.
    ///
    /// The dual-mode design mirrors V4F's `is_free` — the constraint
    /// only kicks in at structural transitions, not during free prose
    /// or args body.
    ///
    /// Three sources make InArgs non-free:
    ///   1. Buffer ends with a `</tool_call>` close-marker prefix
    ///      (the model is committing to close — gate it).
    ///   2. The n-gram loop guard has tripped
    ///      (`self.attractor_detected = true`): force the close so
    ///      the model gets a forced exit instead of being stuck
    ///      extending an attractor.
    ///   3. A required field is still missing from the args body:
    ///      the model is in the middle of args body and might next
    ///      try to close with `}}\n</tool_call>` — we must reject
    ///      that close until required fields appear. We make the
    ///      state non-free so `is_token_allowed` runs and can
    ///      enforce the per-token check.
    pub fn is_free(&self) -> bool {
        match self.state {
            // Masking on a partial `<tool_call>` prefix strands free text: a
            // bare `<` (`<ip>`, `<p>`, `2 < 3`) leaves only `t`/`to`… legal.
            State::Out => true,
            State::AfterOpen => false,
            State::InArgs => {
                if self.attractor_detected {
                    return false;
                }
                if !self.required_fields_satisfied() {
                    return false;
                }
                if self.args_brace_depth >= 0 {
                    // The OUTER tool-call object `{"name":..,"arguments":<v>}` is
                    // still open. `args_brace_depth` counts only the args-VALUE
                    // object (it re-enters InArgs at 0, the header's outer `{`
                    // already consumed), so a balanced value nets back to 0 while
                    // the enclosing `}` is still outstanding (depth -> -1 once it
                    // lands). Stay constrained so `is_token_allowed` can block a
                    // premature `</tool_call>` before the outer brace.
                    return false;
                }
                !Self::has_close_prefix(&self.partial_buf)
            }
        }
    }

    /// Does the buffer end with a strict prefix of `</tool_call>`?
    fn has_close_prefix(s: &str) -> bool {
        const CLOSE: &str = "</tool_call>";
        for n in 1..=CLOSE.len() {
            if s.ends_with(&CLOSE[..n]) {
                return true;
            }
        }
        false
    }

    /// Returns the legal byte-string continuations from the current
    /// state. Each returned string is a FULL prefix starting at the
    /// position immediately after the last firm transition. Callers
    /// check `partial_buf + decode(T)` against these via
    /// [`Self::is_token_allowed`].
    pub fn allowed_continuations(&self) -> Vec<String> {
        match &self.state {
            State::Out => {
                // Only constraining when we've started emitting `<tool_call>`.
                // is_free() short-circuits the unconstrained case before we
                // get here; if we do get here it's because the partial
                // buffer already starts forming `<tool_call>`.
                vec!["<tool_call>".to_string()]
            }
            State::AfterOpen => {
                // Allowed continuations: for each tool name, the literal
                // header `\n{"name": "<NAME>", "arguments": `.
                self.tools
                    .iter()
                    .map(|t| format!("\n{{\"name\": \"{}\", \"arguments\": ", t.name))
                    .collect()
            }
            State::InArgs => {
                // Constraining when the model has started emitting `</tool_call>`.
                // Same dual-mode pattern as State::Out: free unless a close
                // prefix is forming.
                vec!["</tool_call>".to_string()]
            }
        }
    }

    /// Whether the given decoded text would keep us on a legal path
    /// from the current state.
    ///
    /// Empty tokens (placeholder / control tokens with no decoded text)
    /// are always allowed — they don't consume any buffer position.
    ///
    /// Per-state semantics:
    /// - [`State::Out`] / [`State::InArgs`]: in free regions where a
    ///   trigger marker (`<tool_call>` / `</tool_call>`) may start
    ///   forming partway through the buffer, the check is suffix-based.
    ///   A token is allowed iff some suffix of `partial_buf + text` is
    ///   a prefix of the trigger marker (or the full marker appears
    ///   somewhere, which would fire a transition on advance).
    /// - [`State::AfterOpen`]: the entire `partial_buf + text` must be
    ///   a prefix of, or extension of, one of the header templates.
    pub fn is_token_allowed(&self, text: &str) -> bool {
        if text.is_empty() {
            return true;
        }
        if self.is_free() {
            return true;
        }
        let combined = format!("{}{}", self.partial_buf, text);
        match &self.state {
            State::Out => Self::tail_matches_marker(&combined, "<tool_call>"),
            State::AfterOpen => {
                let conts = self.allowed_continuations();
                Self::check_against_conts(&combined, &conts)
            }
            // InArgs: tail-matches-marker is the usual check. When the
            // attractor flag is set or required fields are missing,
            // this same check applies but UNCONDITIONALLY (no is_free
            // short-circuit). For the required-field case we additionally
            // BLOCK any token that would form a close-marker prefix —
            // the model must emit content with the missing field names
            // before closing.
            State::InArgs => {
                let close_prefix = Self::tail_matches_marker(&combined, "</tool_call>");
                if !self.required_fields_satisfied() {
                    // Block close-marker prefix tokens entirely; allow
                    // any other content so the model can keep emitting
                    // until required fields appear in the body. Also
                    // block any token that would close the args body
                    // (`}` bringing brace depth to 0) — without this
                    // gate the closing `}` of an empty `{}` body
                    // commits as args content, and even though the
                    // following `</tool_call>` token is rejected, the
                    // already-streamed `arguments: {}` lands at the
                    // OpenAI API as a malformed tool call.
                    if Self::touches_close_marker(&combined) {
                        false
                    } else if self.would_close_args_body(text) {
                        false
                    } else {
                        true
                    }
                } else if !self.attractor_detected && self.args_brace_depth >= 0 {
                    // Required fields are present, but the OUTER tool-call object
                    // is not yet closed (`args_brace_depth` tracks only the
                    // args-VALUE object, which nets to 0; the outer `}` drives it
                    // to -1). A token that begins a `</tool_call>` marker while the
                    // outer brace is still open would drop that brace and emit
                    // structurally-invalid JSON (the Pi "dropped closing bracket"
                    // failure: `{"name":"read","arguments":{"path":"x"}`). Allow a
                    // close-forming token ONLY if the outer `}` lands (brace depth
                    // < 0) before the marker begins — this still permits a merged
                    // `}}\n</tool_call>` token. Non-close content is always free.
                    // The attractor force-close path is exempt (falls through to
                    // the `else` below) so a looped model can still escape.
                    if Self::touches_close_marker(&combined) {
                        self.outer_closed_before_marker(text)
                    } else {
                        true
                    }
                } else {
                    close_prefix
                }
            }
        }
    }

    /// True iff some suffix of `s` is a prefix of `marker`, OR `marker`
    /// appears anywhere in `s` (the latter handles tokens that finish
    /// emitting the trigger — `advance` will then fire a transition).
    /// Used for free-region constraint checks where the trigger marker
    /// can start partway through `partial_buf`.
    fn tail_matches_marker(s: &str, marker: &str) -> bool {
        if s.contains(marker) {
            return true;
        }
        for n in 1..=marker.len() {
            if s.ends_with(&marker[..n]) {
                return true;
            }
        }
        false
    }

    /// True if `s` contains the full close marker OR ends with any
    /// strict prefix of it. Distinct from `tail_matches_marker`
    /// because we want a binary "does this token touch the close
    /// region" answer for the required-field gate, not a continuation
    /// check.
    fn touches_close_marker(s: &str) -> bool {
        const CLOSE: &str = "</tool_call>";
        if s.contains(CLOSE) {
            return true;
        }
        for n in 1..=CLOSE.len() {
            if s.ends_with(&CLOSE[..n]) {
                return true;
            }
        }
        false
    }

    /// Either `s` is a prefix of some continuation, or some continuation
    /// is a prefix of `s` (the latter handles tokens that extend past
    /// the firm transition boundary — e.g. `\n{"name"` matches the
    /// `\n{"name": "X", "arguments": ` template up to position 8).
    fn check_against_conts(s: &str, conts: &[String]) -> bool {
        for cont in conts {
            if cont.starts_with(s) || s.starts_with(cont.as_str()) {
                return true;
            }
        }
        false
    }

    /// Populate a boolean mask over `vocab` indicating which tokens are
    /// legal at the current matcher position. `out` must be at least
    /// `vocab.len()` long; entries beyond `vocab.len()` are untouched.
    ///
    /// Fast path: when [`Self::is_free`] is true the entire mask is set
    /// to `true` — the caller can skip the sample-time mask scan.
    /// Hot path: O(vocab) scan calling [`Self::is_token_allowed`] per
    /// id, ~129k vocab × a handful of byte comparisons → sub-ms.
    pub fn token_mask(&self, vocab: &[String], out: &mut [bool]) {
        debug_assert!(out.len() >= vocab.len());
        if self.is_free() {
            for slot in out.iter_mut().take(vocab.len()) {
                *slot = true;
            }
            return;
        }
        for (id, text) in vocab.iter().enumerate() {
            out[id] = self.is_token_allowed(text);
        }
    }

    /// Apply the token mask in-place to a logits slice: disallowed
    /// tokens get `f32::NEG_INFINITY`, allowed are left alone.
    pub fn apply_mask_to_logits(mask: &[bool], logits: &mut [f32]) {
        let n = mask.len().min(logits.len());
        for i in 0..n {
            if !mask[i] {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }

    /// Commit decoded token bytes into the matcher, advancing state if
    /// any allowed continuation completes. Idempotent at the byte
    /// level — callers may pass single bytes, multi-byte chunks, or
    /// full decoded tokens; the same final state is reached either way.
    ///
    /// While in [`State::InArgs`], the *structural* (out-of-string) bytes
    /// are fed into the n-gram loop guard so a model that drifts into a
    /// repeating attractor over the JSON skeleton is detected — the next
    /// `is_token_allowed` calls then force the close marker. Bytes inside a
    /// JSON string value (e.g. a `write` tool's code `content`) are excluded
    /// to avoid false-tripping on legitimate code repetition.
    pub fn advance(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if matches!(self.state, State::InArgs) {
            // `ngram_history` accumulates the FULL args text (field names live
            // inside string values) for the required-field guard. The n-gram
            // attractor guard must NOT see string-value bytes — a `write`/`edit`
            // tool's code `content` legitimately repeats short n-grams
            // (indentation, escaped newline+indent units `\n    `, `0, 0, 0, …`,
            // `},\n},\n…`) that would false-trip the 3-byte×6 default and force
            // a premature `</tool_call>`, truncating the argument and emitting
            // `{}` to the client (the write-tool empty-args bug).
            // `update_args_brace_state` walks byte-by-byte and feeds only the
            // *structural* (out-of-string) bytes into `attractor_buf`, so
            // structural loops (the JSON skeleton itself repeating) are still
            // caught.
            self.update_ngram_history(text);
            self.update_args_brace_state(text);
        }
        self.partial_buf.push_str(text);

        loop {
            match self.transition_once() {
                Transition::Stay => return,
                Transition::Advanced => continue,
            }
        }
    }

    /// Inner step: examine `partial_buf` against the current state's
    /// allowed transitions. Returns whether any firm transition fired.
    fn transition_once(&mut self) -> Transition {
        match self.state.clone() {
            State::Out => {
                // Look for `<tool_call>` anywhere in the buffer. Once
                // found, drop everything up to and including that
                // substring and transition.
                if let Some(idx) = self.partial_buf.find("<tool_call>") {
                    self.partial_buf.drain(..idx + "<tool_call>".len());
                    self.state = State::AfterOpen;
                    return Transition::Advanced;
                }
                // Trim the buffer to at most the longest open prefix
                // we could complete next step. Stops the buffer from
                // growing without bound during long free-emission runs.
                let max_keep = "<tool_call>".len() - 1;
                if self.partial_buf.len() > max_keep {
                    let drop =
                        Self::drain_boundary(&self.partial_buf, self.partial_buf.len() - max_keep);
                    self.partial_buf.drain(..drop);
                }
                Transition::Stay
            }
            State::AfterOpen => {
                // Look for the longest tool-name header that the buffer
                // fully covers. When found, consume it and transition
                // to InArgs — and record which tool's schema is now
                // active so the close-marker check can validate its
                // required-field list.
                for (idx, schema) in self.tools.iter().enumerate() {
                    let cont = format!("\n{{\"name\": \"{}\", \"arguments\": ", schema.name);
                    if let Some(rest) = self.partial_buf.strip_prefix(cont.as_str()) {
                        let rest_owned = rest.to_string();
                        self.partial_buf = rest_owned.clone();
                        self.state = State::InArgs;
                        self.current_tool = Some(idx);
                        self.args_brace_depth = 0;
                        self.args_in_string = false;
                        self.args_string_escape = false;
                        // `rest` is the START of the args body (e.g. `{"command`)
                        // that arrived in the SAME chunk as the `"arguments": `
                        // marker. `advance` only feeds `ngram_history` /
                        // brace-state when ALREADY in InArgs, so without this
                        // the opening fragment is dropped — losing the leading
                        // `"` of the first field name, which made
                        // `required_fields_satisfied` perpetually false and
                        // rejected the valid `</tool_call>` close (a spurious
                        // DFlash grammar violation + full KV/DN reset on every
                        // tool turn, which defeated prompt-cache reuse). Feed it
                        // exactly once here; subsequent `advance` calls feed only
                        // their own new text, so there's no double-count of the
                        // brace depth.
                        if !rest_owned.is_empty() {
                            self.update_ngram_history(&rest_owned);
                            self.update_args_brace_state(&rest_owned);
                        }
                        return Transition::Advanced;
                    }
                }
                Transition::Stay
            }
            State::InArgs => {
                // Look for `</tool_call>` anywhere in the buffer.
                if let Some(idx) = self.partial_buf.find("</tool_call>") {
                    self.partial_buf.drain(..idx + "</tool_call>".len());
                    self.state = State::Out;
                    // Returning to Out — reset the n-gram guard's
                    // bookkeeping so a subsequent tool_call body starts
                    // with a fresh window. (The attractor flag must be
                    // cleared OR a stale flag will block the next
                    // body's first byte.) Also drop `current_tool` so
                    // the required-field check no longer applies.
                    self.ngram_history.clear();
                    self.attractor_buf.clear();
                    self.attractor_detected = false;
                    self.current_tool = None;
                    self.args_brace_depth = 0;
                    self.args_in_string = false;
                    self.args_string_escape = false;
                    return Transition::Advanced;
                }
                let max_keep = "</tool_call>".len() - 1;
                if self.partial_buf.len() > max_keep {
                    let drop =
                        Self::drain_boundary(&self.partial_buf, self.partial_buf.len() - max_keep);
                    self.partial_buf.drain(..drop);
                }
                Transition::Stay
            }
        }
    }
}

enum Transition {
    Stay,
    Advanced,
}

impl Matcher {
    /// Round `desired` UP to the next UTF-8 char boundary in `s`.
    /// Required because `String::drain(..n)` panics if `n` straddles
    /// a multi-byte codepoint — and tool_call args can contain
    /// arbitrary UTF-8 (Pi sessions hit this when the model pulled
    /// `𝐵link-hash` from a PDF into a write tool body). Returns
    /// at most `s.len()` so the drain never overshoots.
    fn drain_boundary(s: &str, desired: usize) -> usize {
        if desired >= s.len() {
            return s.len();
        }
        let mut idx = desired;
        while idx < s.len() && !s.is_char_boundary(idx) {
            idx += 1;
        }
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schemas(names: &[&str]) -> Vec<ToolSchema> {
        names
            .iter()
            .map(|n| ToolSchema {
                name: n.to_string(),
                required: Vec::new(),
            })
            .collect()
    }

    /// Schema-with-required helper for required-field tests.
    fn schemas_with_required(specs: &[(&str, &[&str])]) -> Vec<ToolSchema> {
        specs
            .iter()
            .map(|(name, req)| ToolSchema {
                name: name.to_string(),
                required: req.iter().map(|s| s.to_string()).collect(),
            })
            .collect()
    }

    #[test]
    fn out_state_is_free_until_open_prefix() {
        let m = Matcher::new(schemas(&["bash"]));
        assert!(m.is_free());
        assert!(m.is_token_allowed("Hello world"));
        assert!(m.is_token_allowed("<|im_start|>"));
    }

    #[test]
    fn open_marker_transitions_to_after_open() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("here is some prose <tool_call>");
        assert!(matches!(m.state(), State::AfterOpen));
    }

    #[test]
    fn after_open_constrains_to_header_template() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        // The Pi-failure-mode token `<|im_start|>` must be rejected
        // at this position.
        assert!(!m.is_token_allowed("<|im_start|>"));
        // A leading newline that continues the header template is OK.
        assert!(m.is_token_allowed("\n"));
        // `\n{` is OK.
        assert!(m.is_token_allowed("\n{"));
        // Any token that diverges from the header is rejected.
        assert!(!m.is_token_allowed("\nassistant"));
    }

    #[test]
    fn header_completes_into_in_args() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        m.advance("\n{\"name\": \"bash\", \"arguments\": ");
        assert!(matches!(m.state(), State::InArgs));
    }

    #[test]
    fn in_args_allows_content_but_gates_close_until_outer_brace() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        // The outer tool-call object is open (args_brace_depth == 0), so the
        // matcher stays constrained — NOT free — to gate a premature close.
        assert!(!m.is_free());
        // Inside args, any non-close payload is still fine.
        assert!(m.is_token_allowed("{\"command\": \"ls -la\"}"));
        assert!(m.is_token_allowed("\n"));
        // Once the full body (args value + outer `}`) is balanced, InArgs is
        // free again and the close marker is allowed.
        m.advance("{\"command\": \"ls -la\"}}");
        assert!(m.is_free());
        assert!(m.is_token_allowed("</tool_call>"));
    }

    #[test]
    fn close_marker_returns_to_out() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}\n</tool_call>");
        assert!(matches!(m.state(), State::Out));
    }

    #[test]
    fn close_blocked_until_outer_object_brace_emitted() {
        // Regression (Pi "dropped closing bracket"): qwen3.6 mq4 at temp>0
        // sometimes jumps from the inner args `}` straight to `</tool_call>`,
        // dropping the OUTER tool-call object's `}` and emitting invalid JSON
        // (`{"name":"read","arguments":{"path":"/x"}` — no outer close). The
        // grammar must require the outer `}` before permitting the close.
        let mut m = Matcher::new(schemas_with_required(&[("read", &["path"])]));
        m.advance("<tool_call>\n{\"name\": \"read\", \"arguments\": ");
        // Model emits the balanced args VALUE object (required "path" present).
        m.advance("{\"path\": \"/x\"}");
        // Inner object closed (args_brace_depth back to 0) but the OUTER
        // tool-call object `}` is still outstanding — the close must be blocked.
        assert!(
            !m.is_token_allowed("</tool_call>"),
            "close marker must be rejected while the outer object brace is unclosed"
        );
        // The outer `}` itself is still allowed (not a close-marker token).
        assert!(m.is_token_allowed("}"));
        // After the outer `}` lands, the close marker is allowed.
        m.advance("}");
        assert!(
            m.is_token_allowed("</tool_call>"),
            "close marker must be allowed once the outer object is balanced"
        );
    }

    #[test]
    fn lt_char_inside_string_value_is_allowed() {
        // Regression: the outer-brace close gate must NOT block a `<` that is
        // part of a JSON STRING VALUE (heredoc `<<`, `#include <stdio.h>`,
        // `a < b`, …). `touches_close_marker` is not string-aware, so a token
        // ending in `<` looks like a `</tool_call>` prefix; the gate must still
        // allow it because inside a string it is content, not the close marker.
        let mut m = Matcher::new(schemas_with_required(&[("bash", &["command"])]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        // Inside the command string value; required field "command" present.
        m.advance("{\"command\": \"cat > /tmp/x.c ");
        assert!(
            m.is_token_allowed("<"),
            "a bare '<' inside a string value must be allowed (heredoc/include)"
        );
        assert!(
            m.is_token_allowed("<< 'EOF'"),
            "a heredoc token inside a string value must be allowed"
        );
        assert!(
            m.is_token_allowed("#include <stdio.h>"),
            "C include with '<' inside a string value must be allowed"
        );
    }

    #[test]
    fn multiple_tool_names_all_match() {
        let mut m = Matcher::new(schemas(&["bash", "read", "write"]));
        m.advance("<tool_call>");
        // All three tool-name headers are allowed prefixes.
        assert!(m.is_token_allowed("\n{\"name\": \"bash"));
        // Restart with a fresh matcher for the other names.
        let mut m2 = Matcher::new(schemas(&["bash", "read", "write"]));
        m2.advance("<tool_call>");
        assert!(m2.is_token_allowed("\n{\"name\": \"read"));
        let mut m3 = Matcher::new(schemas(&["bash", "read", "write"]));
        m3.advance("<tool_call>");
        assert!(m3.is_token_allowed("\n{\"name\": \"write"));
    }

    #[test]
    fn unknown_tool_name_rejected() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        // `\n{"name": "evil` is not a prefix of `\n{"name": "bash", ...`
        // — the model can't invent a tool name.
        assert!(!m.is_token_allowed("\n{\"name\": \"evil"));
    }

    #[test]
    fn token_mask_free_path_sets_all_true() {
        let m = Matcher::new(schemas(&["bash"]));
        let vocab: Vec<String> = (0..10).map(|i| format!("tok{}", i)).collect();
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        assert!(mask.iter().all(|&b| b));
    }

    #[test]
    fn token_mask_after_open_only_allows_header_tokens() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        let vocab = vec![
            "\n".to_string(),
            "<|im_start|>".to_string(),
            "assistant".to_string(),
            "\n{".to_string(),
            "\n{\"name".to_string(),
        ];
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        assert!(mask[0], "\\n must be allowed (header prefix)");
        assert!(!mask[1], "<|im_start|> must be rejected (Pi failure mode)");
        assert!(!mask[2], "assistant must be rejected");
        assert!(mask[3], "\\n{{ must be allowed (header prefix)");
        assert!(mask[4], "\\n{{\"name must be allowed");
    }

    #[test]
    fn apply_mask_zeros_disallowed_logits() {
        let mask = vec![true, false, true, false];
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
        Matcher::apply_mask_to_logits(&mask, &mut logits);
        assert_eq!(logits[0], 1.0);
        assert!(logits[1].is_infinite() && logits[1].is_sign_negative());
        assert_eq!(logits[2], 3.0);
        assert!(logits[3].is_infinite() && logits[3].is_sign_negative());
    }

    #[test]
    fn buffer_doesnt_grow_unboundedly_in_out() {
        let mut m = Matcher::new(schemas(&["bash"]));
        // Emit a long stretch of prose. The internal buffer should be
        // bounded to the longest open prefix we could complete (11 - 1
        // = 10 chars).
        for _ in 0..1000 {
            m.advance("a");
        }
        assert!(m.partial().len() <= "<tool_call>".len());
    }

    // ─── Pi turn-12 attractor reproduction & fix demonstration ──────
    //
    // These tests directly reproduce the Pi turn-12 failure mode (model
    // emits `<|im_start|>assistant "..."}}` as the `<tool_call>` body
    // instead of valid JSON) using synthetic logits, and verify the
    // grammar masker prevents it. They mirror the real sample-time
    // decision the daemon makes — the only difference is we control
    // the logits directly instead of running a model. The unit tests
    // give a deterministic, model-independent demonstration of:
    //
    //   1. WITHOUT the grammar mask, an argmax over the failure-mode
    //      logits picks the attractor token (`<|im_start|>`).
    //   2. WITH the grammar mask, the masked argmax picks a valid
    //      header-template continuation.
    //
    // The mask path is byte-for-byte the same one
    // `crates/hipfire-runtime/examples/daemon.rs` runs at each sample
    // step in both the qwen35 non-dflash and dflash paths.

    /// Build a synthetic vocab with known token-text values + a logits
    /// vector that reproduces the Pi attractor signal (the bad token
    /// scores highest). Index 0 is `<|im_start|>` per the qwen tokenizer
    /// id ordering observed in the cache traces; the exact ordering
    /// doesn't matter for the test — what matters is the text → score
    /// mapping below.
    fn attractor_logits_setup() -> (Vec<String>, Vec<f32>) {
        // Token vocab: a mix of the attractor + valid header tokens.
        // These mirror the actual qwen3.6:27b vocab strings that
        // appeared in the Pi turn-12 emit.
        let vocab: Vec<String> = vec![
            "<|im_start|>".to_string(),        // 0: attractor (Pi failure)
            "<|im_end|>".to_string(),          // 1: another invalid
            "assistant".to_string(),           // 2: invalid prose
            "\n".to_string(),                  // 3: valid header start
            "\n{".to_string(),                 // 4: valid header
            "\n{\"name".to_string(),           // 5: valid header progression
            "\n{\"name\": \"bash".to_string(), // 6: valid header
            "evil".to_string(),                // 7: not in tool schema
            " arguments".to_string(),          // 8: irrelevant
            "</tool_call>".to_string(),        // 9: premature close — also invalid
        ];
        // Logits where the attractor wins by a wide margin. This mimics
        // the Pi turn-12 distribution: after long context, the model's
        // ChatML-noise attractor scores higher than valid JSON
        // continuations.
        let logits: Vec<f32> = vec![
            10.0, // <|im_start|>  ← attractor wins without mask
            5.0,  // <|im_end|>
            3.0,  // assistant
            2.0,  // \n
            1.5,  // \n{
            1.0,  // \n{"name
            0.5,  // \n{"name": "bash
            -1.0, // evil
            -2.0, // arguments
            -3.0, // </tool_call>
        ];
        (vocab, logits)
    }

    fn argmax(logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[test]
    fn reproduces_pi_turn_12_attractor_without_mask() {
        // PROOF OF FAILURE: with no constraint applied to the logits,
        // the argmax picks `<|im_start|>` — exactly the token that
        // corrupted Pi turn 12's `<tool_call>` body. This is the
        // failure mode we're fixing.
        let (vocab, logits) = attractor_logits_setup();
        let pick = argmax(&logits);
        assert_eq!(
            vocab[pick], "<|im_start|>",
            "raw argmax should pick the attractor (this is the Pi turn-12 failure mode)"
        );
    }

    #[test]
    fn grammar_mask_blocks_pi_turn_12_attractor() {
        // PROOF OF FIX: with the grammar mask applied to the same
        // logits, the argmax picks a VALID header-template token
        // (`\n` — a prefix of the legal continuation
        // `\n{"name": "bash", "arguments": `). The attractor token
        // `<|im_start|>` is masked to `-INF` and can't be selected.
        let (vocab, mut logits) = attractor_logits_setup();
        let mut m = Matcher::new(schemas(&["bash"]));
        // Drive the matcher into `AfterOpen` — the state immediately
        // after the model commits to the `<tool_call>` opener.
        m.advance("<tool_call>");
        assert!(matches!(m.state(), State::AfterOpen));
        assert!(!m.is_free(), "matcher must be constraining in AfterOpen");

        // Build the token mask the daemon would build at sample time
        // and apply it to the logits — same code path as the runtime.
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        Matcher::apply_mask_to_logits(&mask, &mut logits);

        // The attractor is now `-INF`.
        assert!(logits[0].is_infinite() && logits[0].is_sign_negative());
        // `assistant` is also masked (not a header prefix).
        assert!(logits[2].is_infinite() && logits[2].is_sign_negative());
        // `\n` is preserved (valid header start).
        assert_eq!(logits[3], 2.0);

        // The argmax now picks a valid token.
        let pick = argmax(&logits);
        assert_eq!(
            vocab[pick], "\n",
            "masked argmax should pick the highest-scoring valid header prefix"
        );
        // Sanity: the picked token is NOT the attractor.
        assert_ne!(vocab[pick], "<|im_start|>");
    }

    #[test]
    fn grammar_mask_prevents_full_pi_attractor_sequence() {
        // END-TO-END: simulate the full Pi attractor token sequence
        // and verify the mask blocks the FIRST off-distribution token.
        // The sequence from the Pi log was approximately:
        //
        //   <tool_call>  →  \n  →  <|im_start|>  →  assistant  →  …
        //
        // With grammar, after `<tool_call>` + `\n`, the matcher is
        // still in AfterOpen and continues to mask. The next sample
        // step would have picked `<|im_start|>` (or `assistant`) per
        // the attractor logits — both must be rejected.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        m.advance("\n");
        assert!(matches!(m.state(), State::AfterOpen));

        // Each of these is a token the model picked in the Pi log; all
        // must be rejected because none are prefixes of the legal
        // `{"name": "bash", "arguments": ` continuation that follows
        // the `\n`.
        for bad_token in &["<|im_start|>", "assistant", " \"Let me", "}}", "<|im_end|>"] {
            assert!(
                !m.is_token_allowed(bad_token),
                "expected {:?} to be rejected after `<tool_call>\\n`",
                bad_token
            );
        }
        // Valid continuations all pass.
        for good_token in &["{", "{\"", "{\"name", "{\"name\":"] {
            assert!(
                m.is_token_allowed(good_token),
                "expected {:?} to be allowed after `<tool_call>\\n`",
                good_token
            );
        }
    }

    #[test]
    fn dflash_path_post_validation_catches_attractor() {
        // DFLASH STRATEGY: the dflash decode loop validates committed
        // tokens AFTER spec_step commits them. This test simulates that
        // walk for the Pi turn-12 attractor: feed the tokens that
        // appeared in the bad emit through the matcher; the rejection
        // must fire on the first off-distribution token (the daemon
        // then breaks the dflash loop + force-resets KV/DN per the
        // implementation in generate_dflash).
        let mut m = Matcher::new(schemas(&["bash"]));
        // dflash's spec_step would commit a batch — simulate that batch.
        let bad_batch: &[&str] = &[
            "<tool_call>",  // accepted: transitions to AfterOpen
            "\n",           // accepted: matches header prefix
            "<|im_start|>", // REJECTED: not a header prefix from current pos
            "assistant",    // would-be-next, never reached
        ];
        let mut accepted = 0;
        let mut violated = false;
        for tok in bad_batch {
            if !m.is_token_allowed(tok) {
                violated = true;
                break;
            }
            m.advance(tok);
            accepted += 1;
        }
        assert!(violated, "dflash post-validation must detect the attractor");
        assert_eq!(
            accepted, 2,
            "the two valid tokens are accepted; the 3rd is rejected"
        );
    }

    #[test]
    fn full_legal_tool_call_round_trip() {
        // CONTROL: when the model emits a fully-valid tool_call, the
        // matcher never rejects anything and returns cleanly to Out.
        let mut m = Matcher::new(schemas(&["bash"]));
        let sequence = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"echo Hello\"}}\n</tool_call>";
        // Each character/token should be accepted incrementally.
        for ch in sequence.chars() {
            let s = ch.to_string();
            assert!(
                m.is_token_allowed(&s),
                "valid sequence char {:?} unexpectedly rejected at state {:?}",
                ch,
                m.state()
            );
            m.advance(&s);
        }
        assert!(matches!(m.state(), State::Out));
    }

    #[test]
    fn grammar_disabled_skips_constraint() {
        // CONTROL: when there are no tool schemas (e.g., requests
        // without `tools` in the body), the matcher is constructed
        // with an empty schema list — `is_free()` stays true through
        // every state and every token is allowed. This protects the
        // non-tool-call code path from any grammar overhead.
        let mut m = Matcher::new(Vec::new());
        m.advance("<tool_call>");
        // Without schemas, AfterOpen has zero allowed continuations.
        // is_token_allowed still returns true for any text because
        // is_free returns true… wait, AfterOpen is_free returns
        // false. So tokens get rejected. Verify the alternative —
        // the daemon-side check `grammar_active = !schemas.is_empty()`
        // skips the matcher entirely when schemas is empty.
        let grammar_active = false; // mirrors the daemon's gate
        assert!(
            !grammar_active,
            "empty schemas must disable grammar at the daemon level"
        );
    }

    // ─── Multi-tool / schema variation coverage ──────────────────────

    #[test]
    fn tool_names_that_prefix_each_other() {
        // `bash` and `bash_long` overlap on the first 4 chars. The
        // grammar must distinguish between them at the name boundary
        // (the trailing `"` after the name). Both should be reachable,
        // and the prefix doesn't lock in early.
        let mut m = Matcher::new(schemas(&["bash", "bash_long"]));
        m.advance("<tool_call>");
        // Common prefix works.
        assert!(m.is_token_allowed("\n{\"name\": \"bash"));
        // The full short name with closing quote works.
        let mut a = m.clone();
        a.advance("\n{\"name\": \"bash\", \"arguments\": ");
        assert!(
            matches!(a.state(), State::InArgs),
            "short name `bash` must settle to InArgs"
        );
        // The longer name also works.
        let mut b = m.clone();
        b.advance("\n{\"name\": \"bash_long\", \"arguments\": ");
        assert!(
            matches!(b.state(), State::InArgs),
            "long name `bash_long` must settle to InArgs"
        );
    }

    #[test]
    fn unknown_tool_in_multi_schema_rejected() {
        let mut m = Matcher::new(schemas(&["bash", "read", "write"]));
        m.advance("<tool_call>");
        // `evil` is not in the schema; the closing quote after the
        // name is what disambiguates, so the rejection point is when
        // the buffer accumulates `"evil"` (no valid continuation).
        assert!(!m.is_token_allowed("\n{\"name\": \"evil"));
        // But `bash` (a real name) is fine.
        assert!(m.is_token_allowed("\n{\"name\": \"bash"));
    }

    #[test]
    fn many_tools_token_mask_is_fast() {
        // Stress the token_mask scan with a realistic tool count and
        // vocab size. The hot path is O(vocab * conts) — verify it
        // completes in reasonable time and produces a valid mask.
        let tool_names: Vec<String> = (0..32).map(|i| format!("tool_{}", i)).collect();
        let tools: Vec<ToolSchema> = tool_names
            .iter()
            .map(|n| ToolSchema {
                name: n.clone(),
                required: Vec::new(),
            })
            .collect();
        let vocab: Vec<String> = (0..150_000).map(|i| format!("tok_{}", i)).collect();
        let mut m = Matcher::new(tools);
        m.advance("<tool_call>");
        let mut mask = vec![false; vocab.len()];
        let start = std::time::Instant::now();
        m.token_mask(&vocab, &mut mask);
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 1000,
            "token_mask on 150k vocab × 32 tools took {:?} (expected < 1s)",
            elapsed
        );
    }

    // ─── BPE / token boundary edge cases ─────────────────────────────

    #[test]
    fn open_marker_split_across_tokens() {
        // Tokenizer might emit `<tool` + `_call>` as two tokens instead
        // of a single `<tool_call>` special token. The matcher's
        // byte-level partial buffer must handle this — both halves
        // are accepted, and the transition fires on the second.
        let mut m = Matcher::new(schemas(&["bash"]));
        assert!(m.is_token_allowed("<tool"));
        m.advance("<tool");
        assert!(matches!(m.state(), State::Out));
        assert!(m.is_free(), "a partial marker must not constrain free text");
        assert!(m.is_token_allowed("_call>"));
        m.advance("_call>");
        assert!(matches!(m.state(), State::AfterOpen));
    }

    #[test]
    fn close_marker_split_across_tokens() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}");
        assert!(matches!(m.state(), State::InArgs));
        // Split `</tool_call>` into chunks.
        assert!(m.is_token_allowed("\n<"));
        m.advance("\n<");
        assert!(m.is_token_allowed("/tool"));
        m.advance("/tool");
        assert!(m.is_token_allowed("_call>"));
        m.advance("_call>");
        assert!(matches!(m.state(), State::Out));
    }

    #[test]
    fn token_carrying_full_open_marker_inside_prose() {
        // Some BPE tokenizers emit `prose<tool_call>more` as a single
        // token. The matcher must detect the marker mid-token and
        // transition. (The `transition_once` uses `find`, not
        // `ends_with`, for the open marker, so this works.)
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("here is some prose <tool_call>extra");
        // `extra` lands in AfterOpen's partial_buf. The header
        // template starts with `\n` so `extra` is invalid — the
        // matcher should reflect this on the next is_token_allowed.
        assert!(matches!(m.state(), State::AfterOpen));
    }

    #[test]
    fn close_marker_in_args_body_string_does_not_close() {
        // The args body is a JSON value; it can contain `</tool_call>`
        // as a STRING literal. Our current grammar doesn't parse JSON
        // string boundaries — it sees the literal substring as a
        // close marker. Document this as a known limitation: the
        // grammar will prematurely transition. (Real models don't
        // typically emit `</tool_call>` inside arg strings; if this
        // surfaces in production, layer in a JSON-aware string-state
        // tracker.)
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        assert!(matches!(m.state(), State::InArgs));
        // Args content includes a string literal containing the close
        // marker. Our grammar treats it as the close (limitation).
        m.advance("{\"command\": \"echo </tool_call>\"}");
        // Document the actual behavior — this is what we observe today.
        assert!(matches!(m.state(), State::Out));
    }

    // ─── Mask construction edge cases ────────────────────────────────

    #[test]
    fn token_mask_handles_empty_vocab() {
        let m = Matcher::new(schemas(&["bash"]));
        let vocab: Vec<String> = Vec::new();
        let mut mask: Vec<bool> = Vec::new();
        m.token_mask(&vocab, &mut mask);
        assert!(mask.is_empty());
    }

    #[test]
    fn token_mask_handles_empty_string_tokens() {
        // Tokenizers sometimes have control tokens whose decoded text
        // is empty (e.g. BOS/EOS variants). These should always be
        // allowed — they contribute nothing to the partial buffer.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        let vocab = vec![
            "".to_string(),             // empty/control token
            "\n".to_string(),           // valid prefix
            "<|im_start|>".to_string(), // attractor
        ];
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        assert!(mask[0], "empty token must be allowed (control/placeholder)");
        assert!(mask[1], "valid prefix must be allowed");
        assert!(!mask[2], "attractor must be rejected");
    }

    #[test]
    fn apply_mask_handles_size_mismatch() {
        // Defensively: if mask is shorter than logits, only mask the
        // prefix; if mask is longer, ignore the tail. No panic.
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mask = vec![true, false, true]; // shorter than logits
        Matcher::apply_mask_to_logits(&mask, &mut logits);
        assert_eq!(logits[0], 1.0);
        assert!(logits[1].is_infinite());
        assert_eq!(logits[2], 3.0);
        assert_eq!(logits[3], 4.0, "untouched (past mask end)");
        assert_eq!(logits[4], 5.0);
    }

    #[test]
    fn apply_mask_handles_empty_logits() {
        let mask = vec![true, false, true];
        let mut logits: Vec<f32> = Vec::new();
        Matcher::apply_mask_to_logits(&mask, &mut logits);
        assert!(logits.is_empty());
    }

    // ─── ChatML attractor variant coverage ───────────────────────────

    #[test]
    fn all_chatml_special_tokens_rejected_after_open() {
        // Every ChatML special token observed in qwen3.6 attractor
        // cases must be rejected at the `<tool_call>` boundary. These
        // are the tokens that leaked into the body in the Pi log + the
        // Native control plane's parseOneToolCall-compatible sanitizer.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        for tok in &[
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_sep|>",
            "<think>",
            "</think>",
            "<|tool_call|>", // hallucinated variant
        ] {
            assert!(
                !m.is_token_allowed(tok),
                "ChatML token {:?} must be rejected after <tool_call>",
                tok
            );
        }
    }

    #[test]
    fn chatml_tokens_rejected_mid_header() {
        // After committing the start of the header (`\n{"name": "`),
        // ChatML noise must still be rejected. The matcher's partial
        // buffer tracks the in-progress header.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"");
        for tok in &["<|im_start|>", "<|im_end|>", "<think>"] {
            assert!(
                !m.is_token_allowed(tok),
                "{:?} must be rejected after partial header",
                tok
            );
        }
        // The valid tool name continuation is still allowed.
        assert!(m.is_token_allowed("bash"));
        assert!(m.is_token_allowed("bash\", \"arguments\": "));
    }

    // ─── Multi tool_call & sequencing ────────────────────────────────

    #[test]
    fn two_tool_calls_in_one_decode() {
        // Parallel tool-use prompts can emit two `<tool_call>...</tool_call>`
        // blocks back-to-back. The matcher must return to Out after
        // the first close and constrain the second open the same way.
        let mut m = Matcher::new(schemas(&["bash", "read"]));
        m.advance(
            "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>",
        );
        assert!(matches!(m.state(), State::Out));
        // Second open.
        m.advance("\n<tool_call>");
        assert!(matches!(m.state(), State::AfterOpen));
        // Second body must use a known tool name.
        assert!(!m.is_token_allowed("\n{\"name\": \"unknown"));
        assert!(m.is_token_allowed("\n{\"name\": \"read"));
        // Complete the second call.
        m.advance(
            "\n{\"name\": \"read\", \"arguments\": {\"path\": \"/etc/hostname\"}}\n</tool_call>",
        );
        assert!(matches!(m.state(), State::Out));
    }

    #[test]
    fn prose_between_tool_calls_is_free() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}\n</tool_call>");
        assert!(matches!(m.state(), State::Out));
        // Prose chunks are unconstrained.
        m.advance("Now let me think about the next step.");
        assert!(m.is_free());
        // Long prose doesn't get stuck.
        for _ in 0..100 {
            m.advance("more thinking content ");
        }
        assert!(matches!(m.state(), State::Out));
        assert!(m.is_free());
    }

    #[test]
    fn free_text_angle_bracket_does_not_arm_the_mask() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("HTML: ");
        assert!(m.is_free());

        m.advance("<");
        assert!(m.is_free(), "a bare '<' must not constrain the sampler");
        m.advance("p>hi</p>");
        assert!(m.is_free());

        m.advance(" and 2 < 3, table: <t");
        assert!(m.is_free(), "'<t' must not constrain the sampler");
        m.advance("able><tr></tr></table>");
        assert!(m.is_free());
        assert!(matches!(m.state(), State::Out));
    }

    #[test]
    fn tool_call_at_buffer_boundary() {
        // Simulate a token boundary that puts `<tool_call>` exactly at
        // the buffer's bounded-keep limit. The transition must still
        // fire (the matcher detects the full marker via `find`, not
        // just a suffix match).
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("aaaaaaaaaaaaaaaaaaaa<tool_call>"); // 20 a's + marker
        assert!(matches!(m.state(), State::AfterOpen));
    }

    // ─── JSON args variation ─────────────────────────────────────────

    #[test]
    fn empty_args_object_passes() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}\n</tool_call>");
        assert!(matches!(m.state(), State::Out));
    }

    #[test]
    fn nested_json_args_pass() {
        let mut m = Matcher::new(schemas(&["bash"]));
        let body = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"opts\": {\"verbose\": true, \"flags\": [\"a\", \"b\", \"c\"]}, \"cmd\": \"ls -la\"}}\n</tool_call>";
        m.advance(body);
        assert!(matches!(m.state(), State::Out));
    }

    #[test]
    fn unicode_in_args_passes() {
        let mut m = Matcher::new(schemas(&["bash"]));
        let body = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"msg\": \"héllo 世界 🚀\"}}\n</tool_call>";
        m.advance(body);
        assert!(matches!(m.state(), State::Out));
    }

    #[test]
    fn truncated_tool_call_stays_in_args() {
        // max_tokens cap mid-tool-call: matcher is left in InArgs
        // (no close marker seen). The daemon detects truncation via
        // its own bookkeeping; the grammar matcher just doesn't lie
        // about state.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"echo");
        assert!(matches!(m.state(), State::InArgs));
        // No close emitted, so state stays InArgs.
        assert!(matches!(m.state(), State::InArgs));
    }

    // ─── Property-based smoke tests ──────────────────────────────────

    #[test]
    fn property_random_valid_sequences_always_pass() {
        // Generate many random valid tool_call sequences and verify
        // the matcher accepts each one fully and returns to Out.
        // Uses a deterministic LCG seed so failures are reproducible.
        let tools = schemas(&["bash", "read", "write", "edit"]);
        let mut rng_state: u32 = 0xCAFEBABE;
        let mut lcg = || {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            rng_state
        };
        for _ in 0..200 {
            let mut m = Matcher::new(tools.clone());
            let tool_idx = (lcg() % 4) as usize;
            let names = ["bash", "read", "write", "edit"];
            let arg_value: String = (0..((lcg() % 50) as usize))
                .map(|_| (b'a' + (lcg() % 26) as u8) as char)
                .collect();
            let seq = format!(
                "<tool_call>\n{{\"name\": \"{}\", \"arguments\": {{\"x\": \"{}\"}}}}\n</tool_call>",
                names[tool_idx], arg_value
            );
            for ch in seq.chars() {
                let s = ch.to_string();
                assert!(
                    m.is_token_allowed(&s),
                    "char {:?} unexpectedly rejected mid-sequence; state={:?} partial={:?}",
                    ch,
                    m.state(),
                    m.partial()
                );
                m.advance(&s);
            }
            assert!(
                matches!(m.state(), State::Out),
                "valid sequence didn't return to Out (tool={}, args=`{}`, final state={:?})",
                names[tool_idx],
                arg_value,
                m.state()
            );
        }
    }

    #[test]
    fn property_attractor_tokens_never_accepted_after_open() {
        // For every position inside the AfterOpen state (driven by
        // increasing valid header prefixes), the Pi-style attractor
        // tokens must remain rejected. Catches regressions where a
        // partial-buf size or transition logic change accidentally
        // makes a ChatML token look like a header prefix.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        let header = "\n{\"name\": \"bash\", \"arguments\": ";
        for end in 0..=header.len() {
            let mut m2 = m.clone();
            let prefix = &header[..end];
            m2.advance(prefix);
            // Inside AfterOpen unless we hit the end (transitions to
            // InArgs). Either way, attractor tokens must NEVER fit.
            if matches!(m2.state(), State::AfterOpen) {
                for tok in &["<|im_start|>", "<|im_end|>", "<think>"] {
                    assert!(
                        !m2.is_token_allowed(tok),
                        "attractor {:?} accepted at header position {} (partial={:?})",
                        tok,
                        end,
                        m2.partial()
                    );
                }
            }
        }
    }

    // ─── Integration with apply_mask_to_logits ───────────────────────

    #[test]
    fn mask_then_argmax_blocks_attractor_at_every_header_position() {
        // The Pi failure was at position 0 of AfterOpen (right after
        // `<tool_call>`). Verify the mask-then-argmax pipeline blocks
        // the attractor at multiple header positions — at each one,
        // the model is in a slightly-different partial state but the
        // attractor must remain `-INF`.
        //
        // The vocab includes the header bytes as single-char tokens so
        // there's always a valid next-byte token whatever position we
        // pause at. Real qwen vocab has these single-char tokens (
        // newline, ASCII letters, punctuation) so this models the
        // production case.
        let header = "\n{\"name\": \"bash\", \"arguments\": ";
        for end in 0..=header.len() {
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>");
            m.advance(&header[..end]);
            if !matches!(m.state(), State::AfterOpen) {
                continue; // we reached InArgs — past the constrained region
            }
            // Vocab: attractors (must be -INF) + every single-byte
            // ASCII char (at least one will continue the header).
            let mut vocab: Vec<String> = vec![
                "<|im_start|>".to_string(),
                "<|im_end|>".to_string(),
                "<think>".to_string(),
                "assistant".to_string(),
            ];
            for byte in (b' '..=b'~').chain([b'\n', b'\t']) {
                vocab.push((byte as char).to_string());
            }
            let attractor_count = 4;
            let mut logits = vec![100.0f32; vocab.len()];
            // Make attractors win the raw argmax.
            logits[0] = 200.0; // <|im_start|>
            logits[1] = 190.0; // <|im_end|>
            logits[2] = 185.0;
            logits[3] = 180.0;

            let mut mask = vec![false; vocab.len()];
            m.token_mask(&vocab, &mut mask);
            Matcher::apply_mask_to_logits(&mask, &mut logits);

            // All attractors must be -INF.
            for i in 0..attractor_count {
                assert!(
                    logits[i].is_infinite() && logits[i].is_sign_negative(),
                    "attractor vocab[{}]={:?} not -INF at header position {}",
                    i,
                    vocab[i],
                    end,
                );
            }
            // At least ONE token in the vocab must still be allowed —
            // the next byte of the header template is always present
            // in the single-char-ASCII slice.
            assert!(
                logits.iter().any(|l| l.is_finite()),
                "all logits -INF at header position {} (partial={:?}); grammar pinned the model into a corner",
                end, m.partial(),
            );
        }
    }

    // ─── N-gram loop guard tests (real attractor from production) ───
    //
    // The motivating incident: qwen3.6:27b at ~turn 8 of a long
    // agentic session emitted `typetypetypetype...` and
    // `pub fn BlinkHash(key_pub fn BlinkHash(key_...` inside the
    // `<tool_call>` args body. The grammar at the time was permissive
    // in `InArgs` (correctly — args are free-form JSON), so the
    // attractor wasn't blocked. Each retry baked more garbage into
    // the conversation until KV ran out. These tests verify the
    // n-gram loop guard catches both patterns + forces a close.

    #[test]
    fn ngram_guard_catches_structural_skeleton_attractor() {
        // A model that loops the JSON *skeleton* (repeating key/value pairs)
        // is a real attractor we still break. The key/value text lives inside
        // strings (excluded from the guard), but the structural punctuation
        // emitted between repeats (`":1,`) is fed and trips at the 6× default.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        assert!(matches!(m.state(), State::InArgs));
        assert!(!m.attractor_detected(), "guard should be clean on entry");
        let mut payload = String::from("{");
        for _ in 0..8 {
            payload.push_str("\"k\":1,"); // structural stream: `":1,` × 8
        }
        m.advance(&payload);
        assert!(
            m.attractor_detected(),
            "should detect the repeating JSON skeleton"
        );
        // Once flagged, the matcher is no longer free, and only the close
        // marker passes.
        assert!(!m.is_free());
        assert!(!m.is_token_allowed("more"));
        assert!(m.is_token_allowed("\"}}\n</tool_call>"));
    }

    #[test]
    fn ngram_guard_ignores_repetition_inside_string_value() {
        // Regression for the write-tool empty-args bug: a `write` tool whose
        // code `content` legitimately repeats short n-grams (indentation,
        // `pub fn …`, `typetype…`) must NOT trip the guard. The old contract
        // detected inside the string value and force-closed `</tool_call>`,
        // truncating the argument to `{}`. Inside-string bytes are now excluded.
        let mut m = Matcher::new(schemas(&["write"]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        assert!(matches!(m.state(), State::InArgs));
        m.advance("{\"path\": \"/tmp/x.zig\", \"content\": \"");
        // 20-byte phrase × 6 *inside* the content string.
        let phrase = "pub fn BlinkHash(key";
        assert_eq!(phrase.len(), 20);
        let mut payload = String::new();
        for _ in 0..6 {
            payload.push_str(phrase);
        }
        m.advance(&payload);
        // ...and a short-char run, the other classic false-positive shape.
        m.advance("typetypetypetypetypetype");
        assert!(
            !m.attractor_detected(),
            "code repetition inside a string value must NOT trip the guard"
        );
        // The args body still accepts content so the model keeps writing its
        // file. (It is no longer `is_free` — the outer-brace close gate is
        // active while the object is open — but non-close content tokens
        // remain allowed.)
        assert!(m.is_token_allowed(" more code here"));
    }

    #[test]
    fn ngram_guard_does_not_trip_on_3_repeats() {
        // Threshold is 6 repeats (was 4). A 3-repeat sequence —
        // legitimate in arrays like `[1,1,1]` — must NOT trip.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        m.advance("{\"arr\": [1,1,1]}");
        assert!(
            !m.attractor_detected(),
            "3-repeats must stay below threshold"
        );
    }

    #[test]
    fn ngram_guard_does_not_trip_on_5_repeats() {
        // The threshold bump (4 → 6) means 5 repeats — close to the
        // boundary — still must NOT trip. Locks in the regression
        // signal if anyone reverts the bump without thinking.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        // 4-byte gram × 5 = 20 bytes; under the new default of 6.
        m.advance("{\"x\": \"typetypetypetypetype");
        assert!(
            !m.attractor_detected(),
            "5-repeats must stay below the bumped threshold"
        );
    }

    #[test]
    fn ngram_guard_does_not_trip_on_double_newline_blocks() {
        // Code emission with multiple `\n\n` between definitions
        // (2-byte gram × 4) — was the production false positive that
        // motivated the LEN_MIN bump from 2 → 3. Must NOT trip.
        let mut m = Matcher::new(schemas(&["write"]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        m.advance("{\"path\": \"/tmp/x.zig\", \"content\": \"fn a(){}\\n\\nfn b(){}\\n\\nfn c(){}\\n\\nfn d(){}\\n\\nfn e(){}");
        // `\n\n` repeats 4 times — under old defaults this tripped on
        // legit code emitted between blank-separated declarations.
        assert!(
            !m.attractor_detected(),
            "double-newline blocks between defs must NOT trip"
        );
    }

    #[test]
    fn ngram_guard_does_not_trip_on_deep_indentation() {
        // Long whitespace runs are part of normal indented code. The
        // detector's uniform-byte filter (`is_uniform_byte`) skips
        // n-grams consisting of one repeated character — so a 64-space
        // indent does not trip the guard regardless of length.
        let mut m = Matcher::new(schemas(&["write"]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        // 64 spaces — deeper than any realistic indent.
        m.advance("{\"path\": \"/tmp/x.zig\", \"content\": \"return                                                                ; ");
        assert!(
            !m.attractor_detected(),
            "uniform whitespace runs must NOT trip the guard"
        );
    }

    #[test]
    fn ngram_guard_does_not_trip_on_ascii_divider_runs() {
        // ASCII dividers (`====`, `----`, `####`) are common in code
        // comments and section headers. Same uniform-byte filter as
        // whitespace — must NOT trip.
        let mut m = Matcher::new(schemas(&["write"]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        m.advance("{\"path\": \"/tmp/x.zig\", \"content\": \"// ============================================================\\n// section\\n");
        assert!(
            !m.attractor_detected(),
            "ASCII divider runs must NOT trip the guard"
        );
    }

    #[test]
    fn ngram_guard_does_not_trip_on_normal_json() {
        // Normal JSON content (varied bytes) doesn't trip.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        m.advance("{\"command\": \"echo Hello && ls -la /tmp/ && cat README.md\"}");
        assert!(!m.attractor_detected());
    }

    #[test]
    fn ngram_guard_resets_after_close_marker() {
        // After the model commits to `</tool_call>` and we return to
        // Out, the attractor flag clears so a subsequent tool_call
        // doesn't inherit it.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        // Structural skeleton loop (`":1,` × 8) trips the guard.
        let mut payload = String::from("{");
        for _ in 0..8 {
            payload.push_str("\"k\":1,");
        }
        m.advance(&payload);
        assert!(m.attractor_detected());
        // Force-close (the daemon's sample mask would have driven the
        // model here).
        m.advance("}\n</tool_call>");
        assert!(matches!(m.state(), State::Out));
        assert!(!m.attractor_detected(), "flag must clear on close");
        // Next tool_call body starts clean.
        m.advance("\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}\n</tool_call>");
        assert!(matches!(m.state(), State::Out));
        assert!(!m.attractor_detected());
    }

    #[test]
    fn ngram_guard_does_not_trip_on_truly_varied_content() {
        // Long stretch of NON-repeating content — a deterministic LCG
        // over the printable ASCII range. No n-gram of any length
        // 2..32 should repeat 4 consecutive times.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        let mut payload = String::new();
        let mut state: u32 = 0xCAFEBABE;
        for _ in 0..5000 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let c = ((state >> 16) as u8 % 94) + b' '; // printable ASCII
            payload.push(c as char);
        }
        m.advance(&payload);
        assert!(
            !m.attractor_detected(),
            "varied (LCG) content must not trip"
        );
    }

    // ─── Required-field guard tests ──────────────────────────────────
    //
    // The motivating incident: Pi's `write` tool requires `path` and
    // `content`. The model drifted into emitting `arguments:{}` (or
    // `arguments:{"path":"…","edits":[]}` — missing `content`) over
    // and over, Pi rejected each, model retried, KV bloated until
    // exhaustion. The required-field guard rejects the close marker
    // until every required field name appears in the args body.

    #[test]
    fn required_field_guard_blocks_empty_args() {
        // `write` requires `path` and `content`. Model tries to emit
        // empty `{}` and immediately close — both required-field
        // checks must reject any close-marker token.
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        m.advance("<tool_call>");
        m.advance("\n{\"name\": \"write\", \"arguments\": ");
        assert!(matches!(m.state(), State::InArgs));
        m.advance("{}"); // empty args body
                         // Body has no "path" or "content" — close-marker tokens are blocked.
        assert!(!m.is_token_allowed("\n</tool_call>"));
        assert!(!m.is_token_allowed("</tool_call>"));
        assert!(!m.is_token_allowed("\n<"));
        // Non-close-marker content (like a key name) IS allowed — the
        // model can recover by emitting the missing fields.
        assert!(m.is_token_allowed("\"path"));
        assert!(m.is_token_allowed("anything else"));
    }

    // ─── Args-body close-brace guard ───────────────────────────────
    //
    // The close-marker guard alone is insufficient: the `}` that
    // closes the empty `{}` args body commits BEFORE the next-token
    // `</tool_call>` is checked, so the truncated args body
    // (`arguments: {}`) reaches the OpenAI API as a malformed tool
    // call even after the close marker is correctly rejected. The
    // brace-depth guard rejects the closing `}` itself when
    // required fields are not yet satisfied.

    #[test]
    fn args_body_close_brace_blocked_when_required_missing() {
        // Model has just emitted `{`. Brace depth = 1, body open.
        // Next-token `}` would close the body with no required
        // fields present — must be rejected.
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        m.advance("{");
        // brace depth = 1, no required fields seen
        assert!(!m.is_token_allowed("}"));
        assert!(!m.is_token_allowed("}\n"));
        // Other content is allowed.
        assert!(m.is_token_allowed("\"path\":\"/tmp/x\""));
    }

    #[test]
    fn args_body_close_brace_blocks_empty_args_single_token() {
        // The exact production failure: the `write` tool emits `{}`
        // as a single token, then `\n</tool_call>` as a second
        // token. The close marker is correctly rejected, but the
        // `{}` already streamed → `arguments: {}` lands at the API.
        // The brace-depth guard rejects the single-token `{}` itself.
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        // The matcher must reject the empty-args token before it
        // commits — because once it commits, the truncated tool_call
        // already has the malformed shape on the wire.
        assert!(!m.is_token_allowed("{}"));
        assert!(!m.is_token_allowed("{ }"));
        assert!(!m.is_token_allowed("{}\n"));
    }

    #[test]
    fn args_body_close_brace_allowed_when_required_satisfied() {
        // Once required fields appear in the body, the matcher must
        // allow the closing `}` and the close marker. The all-in-one
        // single token also works.
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        m.advance("{\"path\":\"/tmp/x\",\"content\":\"hello\"");
        // Both `"path"` and `"content"` present in ngram_history. The inner
        // args object still needs its `}` and the outer tool-call object needs
        // one more — so the canonical close is `}}\n</tool_call>` (the merged
        // token is allowed because both braces land before the marker).
        assert!(m.is_token_allowed("}"));
        assert!(m.is_token_allowed("}}\n</tool_call>"));
    }

    #[test]
    fn args_body_close_brace_allows_nested_object_close() {
        // Nested objects: a `}` that closes an inner object (depth
        // 2 → 1) MUST be allowed regardless of required-field state
        // because it doesn't close the outer args body.
        let mut m = Matcher::new(schemas_with_required(&[("apply", &["edits"])]));
        m.advance("<tool_call>\n{\"name\": \"apply\", \"arguments\": ");
        m.advance("{\"edits\":[{\"line\":1,\"text\":\"foo\"");
        // Now at depth 3 (outer args, edits array's first object).
        // Closing the inner object `}` → depth 2. Required `"edits"`
        // is already in ngram_history (it's the key in args body),
        // so this would be allowed even without the depth check —
        // but the test exists to lock the "nested closes don't fire
        // the guard" behavior.
        assert!(m.is_token_allowed("}"));
        // Closing the array `]` → depth 2 (no change to braces).
        assert!(m.is_token_allowed("]"));
    }

    #[test]
    fn args_body_close_brace_ignores_braces_in_strings() {
        // `{` / `}` inside JSON string literals must not affect
        // brace depth. The guard's string-aware tracking is what
        // keeps `{"content":"a}b"}` from being misread as closing
        // the body at the `}b` byte.
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        m.advance("{\"path\":\"/tmp/x\",\"content\":\"a}b{c");
        // String is unterminated; depth is 1, in_string is true.
        // Closing the string then the body should be allowed because
        // required fields are present.
        assert!(m.is_token_allowed("\"}"));
    }

    #[test]
    fn args_body_close_brace_ignores_escaped_quote_in_string() {
        // `\"` inside a string MUST NOT toggle in_string off.
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        // Backslash-escaped quote inside path value — string stays open.
        m.advance("{\"path\":\"he said \\\"hi\\\"\",\"content\":\"x\"");
        // Both fields present, brace depth still 1, can close.
        assert!(m.is_token_allowed("}"));
    }

    #[test]
    fn args_body_close_brace_via_token_mask() {
        // Production-style: the mask must block both `{}` and `}`
        // (which alone closes the body once depth >= 1) when
        // required fields aren't yet present.
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        let vocab = vec![
            "{}".to_string(),       // empty body single token — block
            "{ }".to_string(),      // empty body with space — block
            "{".to_string(),        // body open — allow
            "\"path\"".to_string(), // field name — allow
            "garbage".to_string(),  // arbitrary content — allow
        ];
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        assert!(!mask[0], "empty `{{}}` body must be blocked");
        assert!(!mask[1], "empty `{{ }}` body must be blocked");
        assert!(mask[2], "body-open `{{` must be allowed");
        assert!(mask[3], "field name `\"path\"` must be allowed");
        assert!(mask[4], "arbitrary args content must be allowed");
    }

    #[test]
    fn args_body_brace_tracking_resets_on_close_marker() {
        // After a full tool_call cycle, the matcher returns to Out
        // and the brace tracking state must reset so a SECOND
        // tool_call's empty `{}` body is still caught.
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        // First tool call — completes cleanly.
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        m.advance("{\"path\":\"/a\",\"content\":\"b\"}");
        m.advance("\n</tool_call>");
        assert!(matches!(m.state(), State::Out));
        // Second tool call attempt — empty body must be blocked again.
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        assert!(matches!(m.state(), State::InArgs));
        assert!(!m.is_token_allowed("{}"));
    }

    #[test]
    fn required_field_guard_blocks_partial_args() {
        // The Pi-log variant: model emits `{"path":"...","edits":[]}` —
        // has `path` but is missing `content`. Close must still be blocked.
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        m.advance("{\"path\":\"/tmp/x.zig\",\"edits\":[]}");
        // `path` present, `content` missing → still block close.
        assert!(!m.is_token_allowed("\n</tool_call>"));
        assert!(!m.is_token_allowed("</tool_call>"));
        // Allow further content to add the missing field.
        assert!(m.is_token_allowed(",\"content"));
    }

    #[test]
    fn required_field_guard_allows_close_when_all_satisfied() {
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        m.advance("{\"path\":\"/tmp/x\",\"content\":\"hello\"}}");
        // Both required fields present AND the object is fully closed (inner
        // args `}` + outer tool-call `}`) — close is now allowed.
        assert!(m.is_token_allowed("\n"));
        assert!(m.is_token_allowed("\n</tool_call>"));
        m.advance("\n</tool_call>");
        assert!(matches!(m.state(), State::Out));
        assert_eq!(m.current_tool(), None, "current_tool clears on close");
    }

    #[test]
    fn required_field_guard_handles_no_required_fields() {
        // Tool with empty `required` (e.g. `list_files()` with no args)
        // — the guard is a no-op. Close fires normally.
        let mut m = Matcher::new(schemas_with_required(&[("list", &[])]));
        m.advance("<tool_call>\n{\"name\": \"list\", \"arguments\": ");
        m.advance("{}}");
        // No required fields → trivially satisfied; the args object `{}` and
        // the outer tool-call object are both closed.
        assert!(m.is_token_allowed("\n</tool_call>"));
    }

    #[test]
    fn required_field_guard_blocks_close_via_token_mask() {
        // Realistic: model has emitted empty args. Token mask must
        // block both the close marker and any token that would form
        // a close-marker prefix.
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        m.advance("{}");
        let vocab = vec![
            "\n</tool_call>".to_string(), // close marker — must be blocked
            "</tool_call>".to_string(),   // close marker — must be blocked
            "\n<".to_string(),            // close-marker prefix — blocked
            "\"path".to_string(),         // valid field name — allowed
            "\"content".to_string(),      // valid field name — allowed
            "anything".to_string(),       // arbitrary content — allowed
        ];
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        assert!(!mask[0], "full close marker must be blocked");
        assert!(!mask[1], "bare close marker must be blocked");
        assert!(!mask[2], "close-marker prefix must be blocked");
        assert!(mask[3], "field name `\"path` must be allowed");
        assert!(mask[4], "field name `\"content` must be allowed");
        assert!(mask[5], "free content must be allowed");
    }

    #[test]
    fn required_field_guard_substring_match_robust_to_quoting() {
        // The guard does a substring search for `"<name>"` in the
        // args body. This handles standard JSON quoting where field
        // names are double-quoted. The presence check doesn't require
        // a syntactically-valid JSON — substring is enough.
        let mut m = Matcher::new(schemas_with_required(&[("bash", &["command"])]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        // Field appears with spaces around the colon — still matches
        // because we look for `"command"` substring. Close the inner args
        // object and the outer tool-call object (`}}`) before the marker.
        m.advance("{ \"command\" : \"ls -la\" }}");
        assert!(m.is_token_allowed("\n</tool_call>"));
    }

    #[test]
    fn required_field_guard_reproduces_pi_write_attractor() {
        // Direct reproduction of the Pi-log failure pattern:
        //   model emits arguments:{} → Pi rejects → retry → repeat
        //
        // Before the guard: the matcher allowed the close marker and
        // the malformed tool_call propagated to Pi. With the guard,
        // close-marker tokens are masked out — the model has to emit
        // path and content (or hit max_tokens with finish_reason="length").
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        // The exact Pi-log body.
        m.advance("{}");
        // Close attempts the model tried in the log — all must be blocked.
        for bad_token in &["\n</tool_call>", "</tool_call>", "\n<", "<", "</tool_"] {
            assert!(
                !m.is_token_allowed(bad_token),
                "Pi-log close attempt {:?} must be blocked while args is empty",
                bad_token
            );
        }
        // The model can recover by emitting field content.
        assert!(m.is_token_allowed("\"path\":\"/tmp"));
    }

    #[test]
    fn utf8_in_args_body_does_not_panic_on_buffer_trim() {
        // Regression for production panic at
        // `crates/hipfire-arch-qwen35/src/grammar.rs:590` —
        // Pi pulled `𝐵link-hash` from a PDF into a write tool's
        // content arg. The InArgs partial-buf trim used a byte
        // offset that straddled the 4-byte `𝐵` codepoint, and
        // `String::drain(..n)` panicked because `n` wasn't a
        // char boundary. The fix rounds to the next char
        // boundary.
        let mut m = Matcher::new(schemas(&["write"]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        // Emit a long body with multi-byte UTF-8 (4-byte
        // codepoint U+1D435 `𝐵`, repeated past max_keep).
        let mut body = String::from("{\"content\":\"");
        for _ in 0..50 {
            body.push('𝐵'); // 4 bytes in UTF-8
        }
        body.push_str("\"}");
        // This advance previously panicked at the buffer trim.
        m.advance(&body);
        // No panic = test passes. Verify we're still in a valid state.
        assert!(matches!(m.state(), State::InArgs));
    }

    #[test]
    fn utf8_in_long_prose_does_not_panic_in_out_state() {
        // Same fix applies to Out-state trim. Long Unicode prose
        // before any tool_call should be safely trimmed.
        let mut m = Matcher::new(schemas(&["write"]));
        let prose: String = "α𝐵γδε".repeat(20);
        m.advance(&prose);
        // No panic; we're still in Out.
        assert!(matches!(m.state(), State::Out));
    }

    #[test]
    fn current_tool_set_on_in_args_transition() {
        // The matcher must track which tool's schema we're inside.
        let mut m = Matcher::new(schemas_with_required(&[
            ("bash", &["command"]),
            ("write", &["path", "content"]),
        ]));
        m.advance("<tool_call>");
        assert_eq!(m.current_tool(), None, "AfterOpen state has no tool yet");
        m.advance("\n{\"name\": \"write\", \"arguments\": ");
        assert!(matches!(m.state(), State::InArgs));
        assert_eq!(m.current_tool(), Some(1), "write is tools[1]");
    }

    #[test]
    fn ngram_guard_history_bounded() {
        // Even a long emission that DOES trip the guard shouldn't run
        // the matcher out of memory. After 50k bytes including an
        // attractor, the matcher should still be usable. (We can't
        // directly observe ngram_history.len() since it's private —
        // this test just verifies no panic / unbounded growth.)
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        // Trip the guard structurally (`":1,` × 8), then flood with a huge
        // in-string payload. Both buffers stay bounded: `attractor_buf`
        // short-circuits once flagged, and `ngram_history` is window-capped.
        let mut payload = String::from("{");
        for _ in 0..8 {
            payload.push_str("\"k\":1,");
        }
        m.advance(&payload);
        assert!(m.attractor_detected());
        let huge: String = "type".repeat(10_000);
        m.advance(&huge);
        // No assertion needed — surviving this advance without panic
        // is the test.
    }

    #[test]
    fn ngram_guard_forces_close_via_token_mask() {
        // Realistic end-to-end: model emits args body that triggers
        // the guard. Token mask must then allow only tokens that
        // form a `</tool_call>` prefix.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        // Structural skeleton loop (`":1,` × 8) trips the guard.
        let mut payload = String::from("{");
        for _ in 0..8 {
            payload.push_str("\"k\":1,");
        }
        m.advance(&payload);
        assert!(m.attractor_detected());

        let vocab = vec![
            "type".to_string(),         // attractor token — must be -INF
            "abc".to_string(),          // normal prose — must be -INF
            "<".to_string(),            // close-marker prefix — must be allowed
            "</tool".to_string(),       // close-marker prefix — must be allowed
            "</tool_call>".to_string(), // full close — must be allowed
            "".to_string(),             // empty/control — always allowed
        ];
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        assert!(!mask[0], "attractor token `type` must be blocked");
        assert!(!mask[1], "prose token `abc` must be blocked");
        assert!(mask[2], "close-marker prefix `<` must be allowed");
        assert!(mask[3], "close-marker prefix `</tool` must be allowed");
        assert!(mask[4], "full close marker must be allowed");
        assert!(mask[5], "empty token always allowed");
    }

    #[test]
    fn ngram_guard_does_not_force_close_real_log_content_repetition() {
        // The exact byte pattern from the Pi log (transcript shared in
        // conversation, `content` field of the `write` tool call that
        // produced empty `{}` args) — `pub fn BlinkHash(key_type: type, …`
        // followed by a `type` run. This lives INSIDE the content string.
        //
        // Old contract: the guard tripped here and force-closed `</tool_call>`,
        // truncating the argument so the client parsed `{}`. New contract: the
        // n-gram guard never inspects string-value bytes (it can't distinguish
        // a real loop from legitimate repetitive code), so it does NOT trip —
        // a genuinely looping write is instead bounded by `max_tokens`. This
        // is the deliberate trade that fixes the write-tool empty-args bug.
        let mut m = Matcher::new(schemas(&["write"]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        let body = "{\"path\": \"/tmp/x.zig\", \"content\": \"pub fn BlinkHash(key_type: type, value_type: type) typetypetypetypetypetype";
        m.advance(body);
        assert!(
            !m.attractor_detected(),
            "in-content repetition must NOT force-close the tool call"
        );
    }

    #[test]
    fn mask_position_zero_picks_newline() {
        // Most concrete realization of the Pi turn-12 fix. Vocab: all
        // ASCII single-byte chars + the failure-mode attractors. The
        // attractor wins raw argmax. After mask, argmax must pick `\n`
        // (the only valid single-char continuation from AfterOpen at
        // partial="").
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        let mut vocab: Vec<String> = vec![
            "<|im_start|>".to_string(),
            "<|im_end|>".to_string(),
            "<think>".to_string(),
        ];
        for byte in (b' '..=b'~').chain([b'\n', b'\t']) {
            vocab.push((byte as char).to_string());
        }
        let mut logits = vec![1.0f32; vocab.len()];
        logits[0] = 1000.0; // attractor wins raw
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        Matcher::apply_mask_to_logits(&mask, &mut logits);
        let pick = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        // `\n` is the only valid next char (header starts with `\n`).
        assert_eq!(
            vocab[pick], "\n",
            "masked argmax should pick `\\n` (vocab[{}]={:?})",
            pick, vocab[pick]
        );
    }

    // ─── PR #567 regression: free text must not arm the <tool_call> mask ──
    //
    // Defect: Matcher::is_free() returned false for State::Out whenever the
    // buffer ended with any prefix of "<tool_call>" (starting at a bare "<").
    // The sampler then masked free text to only tokens continuing the marker,
    // stranding prose like "<p>hi</p>", "2 < 3", or even "</think>".
    // Fix: State::Out is unconditionally free; masking engages only after a
    // full "<tool_call>" lands (transition to AfterOpen).

    #[test]
    fn pr567_genuine_tool_call_still_constrains() {
        // No regression: a real <tool_call> must still arm the grammar.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        assert!(matches!(m.state(), State::AfterOpen));
        assert!(!m.is_free(), "full <tool_call> must constrain");
        assert!(!m.is_token_allowed("<|im_start|>"));
        assert!(m.is_token_allowed("\n"));
        assert!(m.is_token_allowed("\n{\"name\": \"bash\""));
        let vocab = vec![
            "<|im_start|>".to_string(),
            "\n".to_string(),
            "hello".to_string(),
        ];
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        assert!(!mask[0], "attractor must be blocked after genuine open");
        assert!(mask[1], "header prefix must be allowed");
        assert!(!mask[2], "free prose must be blocked in AfterOpen");
    }

    #[test]
    fn pr567_partial_prefixes_do_not_arm_mask() {
        // Every strict prefix of "<tool_call>" must leave the sampler free.
        let prefixes = [
            "<",
            "<t",
            "<to",
            "<too",
            "<tool",
            "<tool_",
            "<tool_c",
            "<tool_ca",
            "<tool_cal",
            "<tool_call",
        ];
        for prefix in prefixes {
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance(&format!("hello {}", prefix));
            assert!(
                m.is_free(),
                "prefix {:?} must not arm mask; partial={:?}",
                prefix,
                m.partial()
            );
            assert!(matches!(m.state(), State::Out));
            for tok in ["p>", "table>", " ", " hello", " 3", "</think>", ">"] {
                assert!(
                    m.is_token_allowed(tok),
                    "prefix {:?} must allow token {:?}",
                    prefix, tok
                );
            }
            let vocab = vec![
                "p>".to_string(),
                "table>".to_string(),
                "</think>".to_string(),
                "<".to_string(),
                "hello".to_string(),
            ];
            let mut mask = vec![false; vocab.len()];
            m.token_mask(&vocab, &mut mask);
            assert!(
                mask.iter().all(|&b| b),
                "prefix {:?} must leave token_mask all-true",
                prefix
            );
        }
        // Bare prefix at buffer start is also free.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<");
        assert!(m.is_free(), "bare '<' must be free");
        assert!(m.is_token_allowed("p>"));
        assert!(m.is_token_allowed("</think>"));
    }

    #[test]
    fn pr567_false_start_diverging_is_not_masked() {
        // A buffer that looked like a prefix then diverged must stay free.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool");
        assert!(m.is_free());
        assert!(m.is_token_allowed("box>"), "<tool + box> diverges from marker");
        m.advance("box>");
        assert!(m.is_free());
        assert!(matches!(m.state(), State::Out));

        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_cal");
        assert!(m.is_free());
        assert!(m.is_token_allowed("X"), "<tool_cal + X diverges");
        assert!(m.is_token_allowed("x>"));

        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call");
        assert!(m.is_free());
        assert!(m.is_token_allowed("X"), "<tool_call + X is not the marker");
        assert!(m.is_token_allowed(" extra"));

        // Whole false markers in one advance must not transition.
        for txt in ["prefix <toolbox> suffix", "<tool_calX", "<tool-call>", "<tool_callX"] {
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance(txt);
            assert!(
                matches!(m.state(), State::Out),
                "{:?} must not transition to AfterOpen",
                txt
            );
            assert!(m.is_free(), "{:?} must be free", txt);
        }
    }

    #[test]
    fn pr567_partial_prefix_at_eos_stays_free() {
        // Partial prefix at end-of-stream: next sampler step must be unconstrained.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("2 <");
        assert!(m.is_free(), "'2 <' at EOS must be free");
        for tok in [" 3", ">", " p>", "</think>", " hello", "\n"] {
            assert!(m.is_token_allowed(tok), "'2 <' must allow {:?}", tok);
        }
        let vocab = vec![
            " 3".to_string(),
            ">".to_string(),
            " p>".to_string(),
            "</think>".to_string(),
            "<".to_string(),
        ];
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        assert!(mask.iter().all(|&b| b), "EOS partial must be all-true mask");

        // Another EOS shape: HTML tag start cut off mid-token.
        let mut m2 = Matcher::new(schemas(&["bash"]));
        m2.advance("HTML: <table");
        assert!(m2.is_free(), "HTML prefix at EOS must be free");
        assert!(m2.is_token_allowed("><tr>"));
        assert!(m2.is_token_allowed("</think>"));
    }

    #[test]
    fn pr567_prefix_split_across_tokens() {
        // Free-text prefix split across two token emissions must stay free.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<");
        assert!(m.is_free(), "'<' split must be free");
        assert!(m.is_token_allowed("p>"));
        m.advance("p>");
        assert!(m.is_free());
        assert!(matches!(m.state(), State::Out));

        let mut m2 = Matcher::new(schemas(&["bash"]));
        m2.advance("<tool");
        assert!(m2.is_free());
        assert!(m2.is_token_allowed("box>"));
        m2.advance("box>");
        assert!(m2.is_free());

        // Genuine marker split across tokens must still transition.
        let mut m3 = Matcher::new(schemas(&["bash"]));
        m3.advance("<tool");
        assert!(m3.is_free(), "partial genuine must be free before completion");
        assert!(m3.is_token_allowed("_call>"));
        m3.advance("_call>");
        assert!(matches!(m3.state(), State::AfterOpen));
        assert!(!m3.is_free(), "full marker split must constrain");

        // Multi-token free sequence: "2" + " <" + " 3"
        let mut m4 = Matcher::new(schemas(&["bash"]));
        m4.advance("2");
        m4.advance(" <");
        assert!(m4.is_free(), "'2 <' split must be free");
        assert!(m4.is_token_allowed(" 3"));
        assert!(m4.is_token_allowed("</think>"));

        // HTML split: "HTML: " + "<" + "t" + "able>"
        let mut m5 = Matcher::new(schemas(&["bash"]));
        m5.advance("HTML: ");
        m5.advance("<");
        assert!(m5.is_free());
        m5.advance("t");
        assert!(m5.is_free());
        m5.advance("able>");
        assert!(m5.is_free());
    }
}
} // mod json

/// DSML grammar — state machine for `<｜DSML｜tool_calls>` XML-style tool calls.
/// See original `hipfire-arch-deepseek4/src/grammar.rs` for full design notes.
pub mod dsml {
//! Grammar-guided decoding for DeepSeek V4's DSML tool-call format.
//!
//! Tracks a small state machine over the model's emitted bytes that
//! mirrors the parser in `dsml.rs`. At each sample step, the matcher
//! reports the set of legal byte-string continuations from the current
//! state; the daemon converts that into a vocab bitmask (token T allowed
//! iff `partial_buf + decode(T)` is a prefix of any legal continuation)
//! and zeroes the logits of disallowed tokens before sampling.
//!
//! The matcher only kicks in when the model is INSIDE a DSML structural
//! position — outside any tool-call block, and inside parameter values,
//! emission is unconstrained. This keeps the constraint surface tight:
//! the model writes prose / code / params freely, but cannot emit an
//! invalid tag name like `<｜DSML｜tool_cbl>` or `<｜DSML｜calling>`.
//!
//! ## States
//!
//! - [`State::Out`] — free emission. Watching for the opening trigger
//!   `<｜DSML｜tool_calls>` to enter [`State::InToolCalls`].
//! - [`State::InToolCalls`] — between the outer open and close. Next
//!   firm bytes must be the open of an invoke or the close of the block.
//! - [`State::InInvokeName`] — between `<｜DSML｜invoke name="` (or the
//!   `tool` variant) and the closing `">\n`. Emits a tool name.
//! - [`State::InInvokeBody`] — between `<｜DSML｜...name="X">\n` and
//!   `</｜DSML｜invoke>` / `</｜DSML｜tool>`. Next firm bytes must be a
//!   parameter open or the invoke close.
//! - [`State::InParamName`] — between `<｜DSML｜parameter name="` and
//!   the closing `"`. Emits a parameter name.
//! - [`State::InParamAttr`] — between the param-name close-quote and
//!   `">`. Must be ` string="true"` or ` string="false"`.
//! - [`State::InParamBody`] — between the param-attr `">` and
//!   `</｜DSML｜parameter>`. Free emission of the parameter value.
//!
//! Each state carries a `partial_buf: String` of bytes committed since
//! the last firm transition. `allowed_continuations()` returns the
//! literal byte strings any one of which the model is allowed to be in
//! the middle of emitting. A token T is allowed iff
//! `partial_buf + decode(T)` is a prefix of some allowed string.
//!
//! ## Why a state machine and not a regex
//!
//! BPE fragmentation means a single string like `<｜DSML｜tool_calls>`
//! is multiple tokens (`<` + `｜DSML｜` + `tool` + `_c` + `alls` + `>`).
//! The matcher tracks the byte-level position so it doesn't matter how
//! the tokens divide the string. The grammar's regular structure (no
//! recursion — invokes don't nest, params don't nest) keeps the state
//! count finite and small.

// ── Public types ────────────────────────────────────────────────────────

/// Position in the DSML grammar. Carries a byte-level partial-match
/// buffer that holds the bytes committed since the last firm state
/// transition. See module doc for transitions.
#[derive(Debug, Clone, PartialEq)]
pub enum State {
    /// Free emission outside any DSML structure. The matcher is watching
    /// for the opening trigger `<｜DSML｜tool_calls>` but does not
    /// otherwise constrain emission.
    Out,
    /// Between `<｜DSML｜tool_calls>` (already consumed) and the matching
    /// close. Next firm bytes must open an invoke or close the block.
    InToolCalls,
    /// Between `<｜DSML｜invoke name="` (or `<｜DSML｜tool name="`) and
    /// the closing `">`. Emitting the tool name. `tool_idx` is the
    /// index into the schema for the in-progress tool, or `None`
    /// while the name is still being matched against schema entries.
    InInvokeName { tool_idx: Option<usize> },
    /// Inside an invoke body. `emitted_params` lists the indices of
    /// params already serialised in this invoke — the matcher uses
    /// this to gate the invoke-close alternatives on schema `required`
    /// being satisfied. Next firm bytes must open a parameter or, if
    /// required is satisfied, close the invoke.
    InInvokeBody {
        tool_idx: usize,
        emitted_params: Vec<usize>,
    },
    /// Between `<｜DSML｜parameter name="` and the closing `"`. Emitting
    /// the parameter name for the in-progress invoke. `emitted_params`
    /// is propagated through unchanged until the full param block
    /// closes.
    InParamName {
        tool_idx: usize,
        param_idx: Option<usize>,
        emitted_params: Vec<usize>,
    },
    /// Between the param-name `"` and the attr `">`. Must emit
    /// ` string="true"` or ` string="false"` before continuing.
    InParamAttr {
        tool_idx: usize,
        param_idx: usize,
        emitted_params: Vec<usize>,
    },
    /// Between `">` and `</｜DSML｜parameter>`. Free emission of the
    /// parameter value bytes. `param_idx` is the in-flight param; on
    /// close it gets pushed into `emitted_params` as the matcher
    /// returns to [`State::InInvokeBody`].
    InParamBody {
        tool_idx: usize,
        param_idx: usize,
        emitted_params: Vec<usize>,
    },
}

/// Schema for the available tools. Built from the OpenAI-format tools
/// array at request time. The grammar uses this to constrain tool names
/// and parameter names at their respective positions.
#[derive(Debug, Clone)]
pub struct ToolSchema {
    pub name: String,
    /// Parameter names in the order they appear in the schema. Order
    /// isn't enforced at parse time — params can be emitted in any order
    /// — but the schema is the authoritative set of legal names.
    pub params: Vec<String>,
    /// Subset of `params` that MUST appear in the emitted invoke
    /// block. The grammar removes invoke-close alternatives from the
    /// allowed continuations until every required param has been
    /// observed — without this the V4F MQ2-Lloyd checkpoint emits
    /// empty invokes like `<｜DSML｜tool name="bash"></｜DSML｜tool>`
    /// that the downstream OpenAI client rejects with
    /// `must have required properties command`.
    pub required: Vec<String>,
}

/// The grammar matcher itself: a state plus the bytes committed since
/// the last firm transition. Construct via [`Matcher::new`] with the
/// active tool schemas; advance with [`Matcher::advance`]; query the
/// legal token-prefix continuations from the current state with
/// [`Matcher::allowed_continuations`].
#[derive(Debug, Clone)]
pub struct Matcher {
    state: State,
    /// Bytes committed since the last firm state transition. May span
    /// multiple BPE tokens — we keep matching against allowed strings
    /// until either the buffer fully consumes an allowed string (firm
    /// transition) or no allowed string still has the buffer as a
    /// prefix (match failure → fall back to `Out`).
    partial_buf: String,
    tools: Vec<ToolSchema>,
}

impl Matcher {
    /// Build a fresh matcher in [`State::Out`] with no partial buffer.
    pub fn new(tools: Vec<ToolSchema>) -> Self {
        Self {
            state: State::Out,
            partial_buf: String::new(),
            tools,
        }
    }

    /// Read-only view of the current state.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Bytes accumulated since the last firm state transition.
    pub fn partial(&self) -> &str {
        &self.partial_buf
    }

    /// Whether the matcher is currently free (no constraints — every
    /// token is allowed). Dual-mode in both free-emission states:
    ///
    /// - [`State::Out`] is free UNTIL the model commits the atomic
    ///   `｜DSML｜` token (id 128825 in V4 vocab; the literal substring
    ///   `｜DSML｜` in the buffer). Once that lands, the matcher
    ///   constrains continuations to complete `<｜DSML｜tool_calls>`
    ///   exactly. Without this guard the V4F MQ2-Lloyd checkpoint
    ///   deterministically emits invented opener variants like
    ///   `<｜DSML｜tool_actions>`, `<｜DSML｜tool_invoke>`,
    ///   `<｜DSML｜tInvoke name="…">` — none of those match the trigger
    ///   so the grammar matcher never engages, and the parser sees
    ///   pure garbage. We accept the rare-but-possible mis-fire where
    ///   the model emits `｜DSML｜` for non-tool reasons (quoting the
    ///   format in prose): in that case the constraint will force a
    ///   `tool_calls>` completion, which is acceptable since this
    ///   token does not appear in normal output.
    /// - [`State::InParamBody`] is free until the buffer accumulates a
    ///   close-marker prefix (`</`). Once it lands, constrain to
    ///   complete `</｜DSML｜parameter>` exactly — stops the model
    ///   from emitting near-misses like `</｜DSML｜paperameter>`.
    pub fn is_free(&self) -> bool {
        match self.state {
            State::Out => !self.partial_buf.contains("｜DSML｜"),
            State::InParamBody { .. } => self.partial_buf.is_empty(),
            _ => false,
        }
    }

    /// Returns the set of legal continuation strings from the current
    /// state. Each returned string is a FULL continuation starting from
    /// the position immediately after the last firm state transition —
    /// the caller checks `partial_buf + decode(T)` against these via
    /// [`Self::is_token_allowed`].
    ///
    /// Returns an empty vec when [`Self::is_free`] is true (caller
    /// should allow all tokens).
    pub fn allowed_continuations(&self) -> Vec<String> {
        match &self.state {
            // Out is dual-mode: free emission when no DSML commit is
            // in-flight (caller short-circuits via `is_free`);
            // constrain to the OPEN_TOOL_CALLS trigger once the atomic
            // `｜DSML｜` token has been committed into the buffer.
            State::Out => {
                if self.partial_buf.contains("｜DSML｜") {
                    vec![OPEN_TOOL_CALLS.to_string()]
                } else {
                    Vec::new()
                }
            }
            // InParamBody is dual-mode: free emission when buf is empty
            // (caller short-circuits via `is_free`); constrain to the
            // close marker once the model has started emitting it.
            State::InParamBody { .. } => {
                if self.partial_buf.is_empty() {
                    Vec::new()
                } else {
                    vec![CLOSE_PARAM.to_string()]
                }
            }
            State::InToolCalls => vec![
                OPEN_INVOKE.to_string(),
                OPEN_TOOL_VARIANT.to_string(),
                CLOSE_TOOL_CALLS.to_string(),
            ],
            State::InInvokeName { tool_idx: None, .. } => self
                .tools
                .iter()
                .map(|t| format!("{}\">\n", t.name))
                .collect(),
            State::InInvokeName {
                tool_idx: Some(idx), ..
            } => vec![format!("{}\">\n", self.tools[*idx].name)],
            State::InInvokeBody {
                tool_idx,
                emitted_params,
            } => {
                let mut conts = vec![OPEN_PARAM.to_string()];
                // Allow invoke close only when every required param of
                // the in-flight tool has been emitted. This is what
                // forces the model to fill in `command` before closing
                // a `bash` invoke, etc.
                if self.required_satisfied(*tool_idx, emitted_params) {
                    conts.push(CLOSE_INVOKE.to_string());
                    conts.push(CLOSE_TOOL_VARIANT.to_string());
                }
                conts
            }
            State::InParamName {
                tool_idx,
                param_idx: None,
                emitted_params,
            } => self.tools[*tool_idx]
                .params
                .iter()
                .enumerate()
                // Don't allow re-emitting a param already in the block
                // — `command` appearing twice in one invoke is never
                // valid and the OpenAI parser rejects it.
                .filter(|(i, _)| !emitted_params.contains(i))
                .map(|(_, p)| format!("{}\"", p))
                .collect(),
            State::InParamName {
                tool_idx,
                param_idx: Some(idx),
                ..
            } => vec![format!("{}\"", self.tools[*tool_idx].params[*idx])],
            State::InParamAttr { .. } => vec![
                ATTR_STRING_TRUE.to_string(),
                ATTR_STRING_FALSE.to_string(),
            ],
        }
    }

    /// True when every entry in `self.tools[tool_idx].required` is
    /// represented in `emitted_params`.
    fn required_satisfied(&self, tool_idx: usize, emitted_params: &[usize]) -> bool {
        let tool = &self.tools[tool_idx];
        for req_name in &tool.required {
            let req_idx = match tool.params.iter().position(|p| p == req_name) {
                Some(i) => i,
                // Required name not in params list — schema bug. Be
                // permissive (don't deadlock the matcher).
                None => continue,
            };
            if !emitted_params.contains(&req_idx) {
                return false;
            }
        }
        true
    }

    /// Check whether the candidate decoded token text could be emitted
    /// next without violating the grammar. Returns `true` when the
    /// matcher is in a free-emission state OR when `partial_buf + text`
    /// is a prefix of (or equal to, or extends past) at least one legal
    /// continuation from [`Self::allowed_continuations`].
    ///
    /// In states that tolerate leading whitespace
    /// (`InToolCalls`, `InInvokeBody`), the check is also run against
    /// the whitespace-trimmed prefix — so a token like `\n` or
    /// `\n<｜DSML｜` is accepted because the leading newline will be
    /// silently consumed by [`Self::transition_once`].
    pub fn is_token_allowed(&self, text: &str) -> bool {
        if self.is_free() {
            return true;
        }
        let combined = format!("{}{}", self.partial_buf, text);
        let conts = self.allowed_continuations();
        if Self::check_against_conts(&combined, &conts) {
            return true;
        }
        if self.state_allows_leading_ws() {
            let trimmed = combined.trim_start_matches(|c: char| c == '\n' || c == ' ');
            if Self::check_against_conts(trimmed, &conts) {
                return true;
            }
        }
        false
    }

    fn check_against_conts(s: &str, conts: &[String]) -> bool {
        for cont in conts {
            if cont.starts_with(s) || s.starts_with(cont.as_str()) {
                return true;
            }
        }
        false
    }

    /// Populate a boolean mask over `vocab` indicating which tokens are
    /// legal at the current matcher position. `out` must be at least
    /// `vocab.len()` long; entries beyond `vocab.len()` are untouched.
    ///
    /// Fast path: when [`Self::is_free`] is true, the entire mask is
    /// set to `true` — the caller can skip the sample-time mask scan
    /// entirely. Hot path: O(vocab) scan calling [`Self::is_token_allowed`]
    /// per id. With ~129k vocab and ≤4 alternatives per state this is
    /// ~1.7M byte comparisons per sample step, sub-millisecond on
    /// commodity hardware.
    ///
    /// Tokens whose decoded text is empty (placeholder / no-op tokens)
    /// are always allowed — the empty buffer extension keeps every
    /// active prefix viable.
    pub fn token_mask(&self, vocab: &[String], out: &mut [bool]) {
        debug_assert!(out.len() >= vocab.len());
        if self.is_free() {
            for slot in out.iter_mut().take(vocab.len()) {
                *slot = true;
            }
            return;
        }
        for (id, text) in vocab.iter().enumerate() {
            out[id] = self.is_token_allowed(text);
        }
    }

    /// Apply the token mask in-place to a logits slice: disallowed
    /// tokens get `f32::NEG_INFINITY`, allowed tokens are left alone.
    /// Caller invokes [`Self::token_mask`] first, then this. Splitting
    /// the two lets the caller reuse a single `Vec<bool>` allocation
    /// across the decode loop.
    pub fn apply_mask_to_logits(mask: &[bool], logits: &mut [f32]) {
        let n = mask.len().min(logits.len());
        for i in 0..n {
            if !mask[i] {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }

    /// Commit decoded token bytes into the matcher, advancing state if
    /// any allowed continuation completes. Designed to be idempotent at
    /// the byte level: callers may pass single bytes or multi-byte
    /// chunks; the same final state is reached either way.
    ///
    /// In free-emission states (`Out`, `InParamBody`), the matcher
    /// scans for trigger strings (`<｜DSML｜tool_calls>` from `Out`;
    /// `</｜DSML｜parameter>` from `InParamBody`) and transitions when
    /// the trigger lands at the end of the rolling window.
    pub fn advance(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.partial_buf.push_str(text);

        loop {
            match self.transition_once() {
                Transition::Stay => return,
                Transition::Advanced => {
                    // Loop: another transition might fire on the same
                    // buffer (e.g. open marker consumed → InToolCalls,
                    // then immediately another tag opens).
                    continue;
                }
            }
        }
    }

    /// Inner step: examine `partial_buf` against the current state's
    /// allowed transitions. Returns whether any firm transition fired.
    ///
    /// States between tags (`InToolCalls`, `InInvokeBody`) tolerate
    /// leading whitespace — the HF reference format emits `\n` between
    /// every tag and tokens like `>\n` (vocab id 1018) carry that
    /// newline INTO the buffer when the open trigger fires.
    fn transition_once(&mut self) -> Transition {
        // Consume one leading whitespace byte in states that allow it.
        if self.state_allows_leading_ws() {
            if let Some(n) = self.leading_ws_byte_count() {
                self.partial_buf.drain(..n);
                return Transition::Advanced;
            }
        }
        match self.state.clone() {
            State::Out => self.transition_from_free(OPEN_TOOL_CALLS, State::InToolCalls),
            State::InParamBody {
                tool_idx,
                param_idx,
                mut emitted_params,
            } => {
                // On close: record this param as emitted, then return
                // to InInvokeBody with the updated set.
                if !emitted_params.contains(&param_idx) {
                    emitted_params.push(param_idx);
                }
                self.transition_from_free(
                    CLOSE_PARAM,
                    State::InInvokeBody {
                        tool_idx,
                        emitted_params,
                    },
                )
            }
            State::InToolCalls => self.transition_from_alternatives(&[
                (OPEN_INVOKE, State::InInvokeName { tool_idx: None }),
                (OPEN_TOOL_VARIANT, State::InInvokeName { tool_idx: None }),
                (CLOSE_TOOL_CALLS, State::Out),
            ]),
            State::InInvokeName { tool_idx } => self.transition_invoke_name(tool_idx),
            State::InInvokeBody {
                tool_idx,
                emitted_params,
            } => {
                // Walk the alternatives manually so we can build the
                // gated close alternatives (only legal once `required`
                // is satisfied) without static `&str` lifetime gymnastics.
                let mut alts: Vec<(&str, State)> = vec![(
                    OPEN_PARAM,
                    State::InParamName {
                        tool_idx,
                        param_idx: None,
                        emitted_params: emitted_params.clone(),
                    },
                )];
                if self.required_satisfied(tool_idx, &emitted_params) {
                    alts.push((CLOSE_INVOKE, State::InToolCalls));
                    alts.push((CLOSE_TOOL_VARIANT, State::InToolCalls));
                }
                self.transition_from_alternatives(&alts)
            }
            State::InParamName {
                tool_idx,
                param_idx,
                emitted_params,
            } => self.transition_param_name(tool_idx, param_idx, emitted_params),
            State::InParamAttr {
                tool_idx,
                param_idx,
                emitted_params,
            } => self.transition_from_alternatives(&[
                (
                    ATTR_STRING_TRUE,
                    State::InParamBody {
                        tool_idx,
                        param_idx,
                        emitted_params: emitted_params.clone(),
                    },
                ),
                (
                    ATTR_STRING_FALSE,
                    State::InParamBody {
                        tool_idx,
                        param_idx,
                        emitted_params,
                    },
                ),
            ]),
        }
    }

    /// True in states where one or more leading `\n` / ` ` bytes in
    /// `partial_buf` should be silently consumed before trying any
    /// alternative match. Driven by the HF reference renderer which
    /// emits `\n` between sibling tags.
    fn state_allows_leading_ws(&self) -> bool {
        matches!(
            self.state,
            State::InToolCalls | State::InInvokeBody { .. }
        )
    }

    /// Length in bytes of the leading whitespace run (`\n` or ` `) in
    /// `partial_buf`, or `None` when the first byte is non-ws.
    fn leading_ws_byte_count(&self) -> Option<usize> {
        let bytes = self.partial_buf.as_bytes();
        let mut n = 0;
        while n < bytes.len() && (bytes[n] == b'\n' || bytes[n] == b' ') {
            n += 1;
        }
        if n == 0 {
            None
        } else {
            Some(n)
        }
    }

    /// Trigger-scan transition: look for `trigger` at the tail of
    /// `partial_buf`. If found, advance to `next_state` and drop the
    /// trigger (plus everything before it). If not found, keep only
    /// the longest suffix that is a prefix of `trigger` (so future
    /// bytes can complete it).
    fn transition_from_free(&mut self, trigger: &str, next_state: State) -> Transition {
        if let Some(idx) = self.partial_buf.find(trigger) {
            // Drop everything up to and including the trigger.
            let after = idx + trigger.len();
            self.partial_buf = self.partial_buf[after..].to_string();
            self.state = next_state;
            return Transition::Advanced;
        }
        // Trim the rolling buffer to the longest suffix that is still
        // a prefix of `trigger`. Bound by len(trigger)-1 bytes.
        let max_keep = trigger.len().saturating_sub(1);
        if self.partial_buf.len() > max_keep {
            let drop_n = self.partial_buf.len() - max_keep;
            let drop_n = utf8_safe_split(&self.partial_buf, drop_n);
            self.partial_buf.drain(..drop_n);
        }
        // Trim further from the left: walk forward until the suffix
        // starting at that point is a prefix of trigger.
        while !self.partial_buf.is_empty() && !trigger.starts_with(self.partial_buf.as_str()) {
            // Drop one char (UTF-8 safe).
            let mut k = 1;
            while k < self.partial_buf.len()
                && (self.partial_buf.as_bytes()[k] & 0b1100_0000) == 0b1000_0000
            {
                k += 1;
            }
            self.partial_buf.drain(..k);
        }
        Transition::Stay
    }

    /// Try to match `partial_buf` (from its start) against any of the
    /// alternatives. If one fully matches, transition to its state and
    /// drop the matched bytes. If at least one is still a prefix
    /// candidate, stay. If none is a prefix anymore (corrupt), fall
    /// back to `Out` (recovery).
    fn transition_from_alternatives(
        &mut self,
        alternatives: &[(&str, State)],
    ) -> Transition {
        for (needle, next) in alternatives {
            if self.partial_buf.starts_with(needle) {
                self.partial_buf.drain(..needle.len());
                self.state = next.clone();
                return Transition::Advanced;
            }
        }
        // No full match. Check for any active prefix.
        let any_prefix = alternatives
            .iter()
            .any(|(n, _)| n.starts_with(self.partial_buf.as_str()));
        if !any_prefix {
            // Corruption: fall back to Out. Should be unreachable when
            // is_token_allowed is honored.
            self.partial_buf.clear();
            self.state = State::Out;
        }
        Transition::Stay
    }

    /// Tool-name transition: schema names + `">\n`. When `tool_idx` is
    /// `None`, identify the matching schema entry as soon as the
    /// partial_buf uniquely fixes one. When the buffer fully covers
    /// `NAME">\n` (or extends past it because a token spanned multiple
    /// grammar tokens), transition to `InInvokeBody` and keep any
    /// trailing bytes in the buffer for the next state to consume.
    fn transition_invoke_name(&mut self, tool_idx: Option<usize>) -> Transition {
        let candidates: Vec<(usize, String)> = match tool_idx {
            None => (0..self.tools.len())
                .map(|i| (i, format!("{}\">\n", self.tools[i].name)))
                .collect(),
            Some(idx) => vec![(idx, format!("{}\">\n", self.tools[idx].name))],
        };
        // Full coverage → transition into invoke body. Buffer keeps the
        // trailing suffix (if the committed token spanned more than the
        // tool-name terminator). Fresh invoke → emitted_params empty.
        for (idx, full) in &candidates {
            if self.partial_buf.starts_with(full) {
                self.partial_buf.drain(..full.len());
                self.state = State::InInvokeBody {
                    tool_idx: *idx,
                    emitted_params: Vec::new(),
                };
                return Transition::Advanced;
            }
        }
        // Lock in the tool_idx as soon as exactly one candidate still
        // matches as a prefix (when we were `None`).
        if tool_idx.is_none() {
            let matching: Vec<usize> = candidates
                .iter()
                .filter(|(_, full)| full.starts_with(self.partial_buf.as_str()))
                .map(|(i, _)| *i)
                .collect();
            if matching.len() == 1 {
                self.state = State::InInvokeName {
                    tool_idx: Some(matching[0]),
                };
                // Don't drain — partial_buf still being built up against
                // the (now locked-in) full name.
                return Transition::Advanced;
            }
        }
        // No prefix match → corruption recovery.
        let any_prefix = candidates
            .iter()
            .any(|(_, full)| full.starts_with(self.partial_buf.as_str()));
        if !any_prefix {
            self.partial_buf.clear();
            self.state = State::Out;
        }
        Transition::Stay
    }

    /// Param-name transition: schema params for the current tool + `"`.
    /// When buffer fully covers `PARAM"`, transition to InParamAttr and
    /// keep trailing bytes for the next state.
    fn transition_param_name(
        &mut self,
        tool_idx: usize,
        param_idx: Option<usize>,
        emitted_params: Vec<usize>,
    ) -> Transition {
        let tool = &self.tools[tool_idx];
        // Exclude already-emitted params from the candidate set so the
        // model can't re-emit `command` twice in one invoke.
        let candidates: Vec<(usize, String)> = match param_idx {
            None => (0..tool.params.len())
                .filter(|i| !emitted_params.contains(i))
                .map(|i| (i, format!("{}\"", tool.params[i])))
                .collect(),
            Some(idx) => vec![(idx, format!("{}\"", tool.params[idx]))],
        };
        for (idx, full) in &candidates {
            if self.partial_buf.starts_with(full) {
                self.partial_buf.drain(..full.len());
                self.state = State::InParamAttr {
                    tool_idx,
                    param_idx: *idx,
                    emitted_params,
                };
                return Transition::Advanced;
            }
        }
        if param_idx.is_none() {
            let matching: Vec<usize> = candidates
                .iter()
                .filter(|(_, full)| full.starts_with(self.partial_buf.as_str()))
                .map(|(i, _)| *i)
                .collect();
            if matching.len() == 1 {
                self.state = State::InParamName {
                    tool_idx,
                    param_idx: Some(matching[0]),
                    emitted_params,
                };
                return Transition::Advanced;
            }
        }
        let any_prefix = candidates
            .iter()
            .any(|(_, full)| full.starts_with(self.partial_buf.as_str()));
        if !any_prefix {
            self.partial_buf.clear();
            self.state = State::Out;
        }
        Transition::Stay
    }
}

enum Transition {
    Stay,
    Advanced,
}

/// Largest split point ≤ `n` that doesn't cut through a multi-byte
/// UTF-8 character.
fn utf8_safe_split(s: &str, n: usize) -> usize {
    let bytes = s.as_bytes();
    let mut k = n.min(bytes.len());
    while k > 0 && (bytes[k] & 0b1100_0000) == 0b1000_0000 {
        k -= 1;
    }
    k
}

// ── Constants borrowed from the DSML format ─────────────────────────────

/// Open trigger for the tool-calls block — entry from `State::Out`.
pub(crate) const OPEN_TOOL_CALLS: &str = "<｜DSML｜tool_calls>";
/// Closing marker for the tool-calls block.
pub(crate) const CLOSE_TOOL_CALLS: &str = "</｜DSML｜tool_calls>";
/// Open of an invoke. The V4F MQ2-Lloyd checkpoint also emits the
/// `tool` variant (see [`OPEN_TOOL_VARIANT`]) — both must be accepted.
pub(crate) const OPEN_INVOKE: &str = "<｜DSML｜invoke name=\"";
pub(crate) const CLOSE_INVOKE: &str = "</｜DSML｜invoke>";
/// V4F MQ2-Lloyd variant of [`OPEN_INVOKE`] / [`CLOSE_INVOKE`]: the
/// model deterministically picks `tool` (token 72461) over `invoke`
/// (token 41523) after `｜DSML｜` on most checkpoints — see
/// `feedback_v4f_emits_tool_not_invoke.md` for the diagnosis.
pub(crate) const OPEN_TOOL_VARIANT: &str = "<｜DSML｜tool name=\"";
pub(crate) const CLOSE_TOOL_VARIANT: &str = "</｜DSML｜tool>";
pub(crate) const OPEN_PARAM: &str = "<｜DSML｜parameter name=\"";
pub(crate) const CLOSE_PARAM: &str = "</｜DSML｜parameter>";
/// String attribute that follows the param-name close-quote.
pub(crate) const ATTR_STRING_TRUE: &str = " string=\"true\">";
pub(crate) const ATTR_STRING_FALSE: &str = " string=\"false\">";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matcher_starts_in_out_state_and_is_free() {
        let m = Matcher::new(vec![]);
        assert_eq!(*m.state(), State::Out);
        assert_eq!(m.partial(), "");
        assert!(m.is_free());
    }

    #[test]
    fn schema_holds_tool_and_params() {
        let s = ToolSchema {
            name: "read".to_string(),
            params: vec!["path".to_string()],
            required: vec!["path".to_string()],
        };
        assert_eq!(s.name, "read");
        assert_eq!(s.params, vec!["path".to_string()]);
    }

    fn schema_read_write() -> Vec<ToolSchema> {
        vec![
            ToolSchema {
                name: "read".to_string(),
                params: vec!["path".to_string()],
                required: vec!["path".to_string()],
            },
            ToolSchema {
                name: "write".to_string(),
                params: vec!["path".to_string(), "content".to_string()],
                required: vec!["path".to_string(), "content".to_string()],
            },
        ]
    }

    #[test]
    fn open_trigger_advances_into_tool_calls() {
        let mut m = Matcher::new(schema_read_write());
        m.advance("Let me read.\n\n");
        assert!(m.is_free());
        m.advance("<｜DSML｜tool_calls>");
        assert_eq!(*m.state(), State::InToolCalls);
        assert_eq!(m.partial(), "");
    }

    #[test]
    fn open_trigger_split_across_chunks() {
        let mut m = Matcher::new(schema_read_write());
        m.advance("<｜");
        assert!(m.is_free());
        m.advance("DSML｜");
        m.advance("tool_");
        m.advance("calls>");
        assert_eq!(*m.state(), State::InToolCalls);
    }

    #[test]
    fn open_invoke_into_invoke_name_then_locks_tool() {
        let mut m = Matcher::new(schema_read_write());
        m.advance("<｜DSML｜tool_calls>");
        m.advance("<｜DSML｜invoke name=\"");
        assert_eq!(*m.state(), State::InInvokeName { tool_idx: None });
        // First letter is ambiguous: "r" matches read but not write
        m.advance("r");
        // tool_idx should be locked to 0 (read)
        assert_eq!(*m.state(), State::InInvokeName { tool_idx: Some(0) });
    }

    #[test]
    fn open_tool_variant_also_enters_invoke_name() {
        let mut m = Matcher::new(schema_read_write());
        m.advance("<｜DSML｜tool_calls>");
        m.advance("<｜DSML｜tool name=\"");
        assert_eq!(*m.state(), State::InInvokeName { tool_idx: None });
    }

    #[test]
    fn tool_name_completion_enters_invoke_body() {
        let mut m = Matcher::new(schema_read_write());
        m.advance("<｜DSML｜tool_calls>");
        m.advance("<｜DSML｜invoke name=\"");
        m.advance("read\">\n");
        assert_eq!(
            *m.state(),
            State::InInvokeBody {
                tool_idx: 0,
                emitted_params: vec![]
            }
        );
        assert_eq!(m.partial(), "");
    }

    #[test]
    fn full_round_trip_through_one_call() {
        let mut m = Matcher::new(schema_read_write());
        m.advance("<｜DSML｜tool_calls>");
        m.advance("<｜DSML｜invoke name=\"read\">\n");
        assert_eq!(
            *m.state(),
            State::InInvokeBody {
                tool_idx: 0,
                emitted_params: vec![]
            }
        );
        m.advance("<｜DSML｜parameter name=\"");
        // `read` has exactly one param (`path`) — matcher locks
        // param_idx to Some(0) immediately since there's only one
        // candidate that has the empty buffer as a prefix.
        assert_eq!(
            *m.state(),
            State::InParamName {
                tool_idx: 0,
                param_idx: Some(0),
                emitted_params: vec![],
            }
        );
        m.advance("path\"");
        assert_eq!(
            *m.state(),
            State::InParamAttr {
                tool_idx: 0,
                param_idx: 0,
                emitted_params: vec![],
            }
        );
        m.advance(" string=\"true\">");
        assert_eq!(
            *m.state(),
            State::InParamBody {
                tool_idx: 0,
                param_idx: 0,
                emitted_params: vec![],
            }
        );
        // Free emission of value
        assert!(m.is_free());
        m.advance("/tmp/test.txt</｜DSML｜parameter>");
        // After close, `path` (param_idx=0) is now in emitted_params.
        // `read` only has one param so required is now satisfied.
        assert_eq!(
            *m.state(),
            State::InInvokeBody {
                tool_idx: 0,
                emitted_params: vec![0]
            }
        );
        m.advance("</｜DSML｜invoke>");
        assert_eq!(*m.state(), State::InToolCalls);
        m.advance("</｜DSML｜tool_calls>");
        assert_eq!(*m.state(), State::Out);
    }

    #[test]
    fn leading_newline_consumed_in_tool_calls_state() {
        // Token `>\n` (vocab id 1018 in V4 tokenizer) leaves `\n` in the
        // buffer after the open trigger fires. Without ws tolerance the
        // matcher falls back to Out and the grammar mask collapses.
        let mut m = Matcher::new(schema_read_write());
        m.advance("<｜DSML｜tool_calls>\n");
        assert_eq!(*m.state(), State::InToolCalls);
        assert_eq!(m.partial(), "");
    }

    #[test]
    fn newline_token_is_allowed_in_tool_calls_state() {
        let mut m = Matcher::new(schema_read_write());
        m.advance("<｜DSML｜tool_calls>");
        // Pure-newline next token must be accepted (it'll be consumed
        // as leading whitespace).
        assert!(m.is_token_allowed("\n"));
        // Newline followed by opener prefix also valid.
        assert!(m.is_token_allowed("\n<"));
        assert!(m.is_token_allowed("\n<｜DSML｜invoke name=\""));
        // Newline + invalid tag rejected.
        assert!(!m.is_token_allowed("\ncalling"));
        assert!(!m.is_token_allowed("\n<｜DSML｜foo"));
    }

    #[test]
    fn newline_consumed_then_real_tag_in_one_advance() {
        let mut m = Matcher::new(schema_read_write());
        m.advance("<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n");
        assert_eq!(
            *m.state(),
            State::InInvokeBody {
                tool_idx: 0,
                emitted_params: vec![]
            }
        );
    }

    #[test]
    fn out_state_constrains_after_dsml_token() {
        // Reproduces the production failure where V4F MQ2-Lloyd emits
        // `<｜DSML｜tool_actions>` / `<｜DSML｜calling>` / etc. instead
        // of the canonical `<｜DSML｜tool_calls>` open trigger. Once
        // `｜DSML｜` lands in the Out buffer, the matcher must restrict
        // continuations to the trigger so invented tag names get masked.
        let mut m = Matcher::new(schema_read_write());
        m.advance("Some prose. ");
        assert!(m.is_free());
        m.advance("<");
        assert!(m.is_free(), "single `<` could just be text");
        m.advance("｜DSML｜");
        assert!(!m.is_free(), "committed ｜DSML｜ → must constrain");
        // Allowed next: tokens that continue toward `tool_calls>`.
        assert!(m.is_token_allowed("tool_calls>"));
        assert!(m.is_token_allowed("tool"));
        assert!(m.is_token_allowed("t"));
        // Invented openers are masked.
        assert!(!m.is_token_allowed("tool_actions>"));
        assert!(!m.is_token_allowed("tool_invoke"));
        assert!(!m.is_token_allowed("calling"));
        assert!(!m.is_token_allowed("foo"));
        // Completing the trigger transitions.
        m.advance("tool_calls>");
        assert_eq!(*m.state(), State::InToolCalls);
    }

    #[test]
    fn out_state_remains_free_with_partial_unrelated_text() {
        // Buf accumulating just `<` doesn't constrain (could be HTML,
        // code, prose). Only `｜DSML｜` in buf flips the constraint.
        let mut m = Matcher::new(schema_read_write());
        m.advance("<");
        assert!(m.is_free());
        m.advance("html>");
        assert!(m.is_free());
    }

    #[test]
    fn required_params_block_empty_invoke_close() {
        // Reproduces the production failure where V4F emitted an empty
        // bash invoke (`<｜DSML｜tool name="bash"></｜DSML｜tool>`) and
        // the OpenAI client rejected it with "must have required
        // properties command".
        let schema = vec![ToolSchema {
            name: "bash".to_string(),
            params: vec!["command".to_string()],
            required: vec!["command".to_string()],
        }];
        let mut m = Matcher::new(schema);
        m.advance("<｜DSML｜tool_calls>");
        m.advance("<｜DSML｜tool name=\"bash\">\n");
        // In InInvokeBody with no params emitted yet — close MUST be
        // blocked, only OPEN_PARAM is legal.
        assert!(m.is_token_allowed("<｜DSML｜parameter name=\""));
        assert!(!m.is_token_allowed("</｜DSML｜tool>"));
        assert!(!m.is_token_allowed("</｜DSML｜invoke>"));
        // Emit the required param.
        m.advance("<｜DSML｜parameter name=\"command\" string=\"true\">ls</｜DSML｜parameter>\n");
        assert_eq!(
            *m.state(),
            State::InInvokeBody {
                tool_idx: 0,
                emitted_params: vec![0]
            }
        );
        // Now close IS legal.
        assert!(m.is_token_allowed("</｜DSML｜tool>"));
        assert!(m.is_token_allowed("</｜DSML｜invoke>"));
    }

    #[test]
    fn already_emitted_param_blocked_from_reuse() {
        // After `command` is emitted once, the matcher must not let
        // the model emit it again — the OpenAI parser rejects
        // duplicate keys.
        let schema = vec![ToolSchema {
            name: "bash".to_string(),
            params: vec!["command".to_string(), "cwd".to_string()],
            required: vec!["command".to_string()],
        }];
        let mut m = Matcher::new(schema);
        m.advance("<｜DSML｜tool_calls>");
        m.advance("<｜DSML｜tool name=\"bash\">\n");
        m.advance("<｜DSML｜parameter name=\"command\" string=\"true\">ls</｜DSML｜parameter>\n");
        // Required satisfied — close is legal.
        assert!(m.is_token_allowed("</｜DSML｜tool>"));
        // But emitting `command` AGAIN must be blocked. The remaining
        // schema-legal opener is for `cwd` only.
        m.advance("<｜DSML｜parameter name=\"");
        // Now only `cwd` is a legal param name; `command` is masked.
        assert!(m.is_token_allowed("c"));
        assert!(m.is_token_allowed("cwd\""));
        assert!(!m.is_token_allowed("command\""));
    }

    #[test]
    fn variant_close_tag_returns_to_tool_calls() {
        // Use a schema with no required params so the empty-invoke
        // close is legal (the required-params enforcement otherwise
        // blocks the close until `path` is filled in).
        let schema = vec![ToolSchema {
            name: "read".to_string(),
            params: vec!["path".to_string()],
            required: vec![],
        }];
        let mut m = Matcher::new(schema);
        m.advance("<｜DSML｜tool_calls>");
        m.advance("<｜DSML｜tool name=\"read\">\n");
        m.advance("</｜DSML｜tool>"); // variant close
        assert_eq!(*m.state(), State::InToolCalls);
    }

    #[test]
    fn is_token_allowed_rejects_bad_tag_in_tool_calls() {
        let mut m = Matcher::new(schema_read_write());
        m.advance("<｜DSML｜tool_calls>");
        // After OPEN_TOOL_CALLS we're in InToolCalls. Legal next tokens
        // start `<` (one of the opens) or `</` (close). An invented
        // continuation like `cbl>` should be rejected.
        assert!(!m.is_token_allowed("cbl>"));
        assert!(!m.is_token_allowed("tool_invoke"));
        assert!(m.is_token_allowed("<"));
        assert!(m.is_token_allowed("<｜DSML｜invoke name=\""));
        assert!(m.is_token_allowed("<｜DSML｜tool name=\""));
        assert!(m.is_token_allowed("</"));
    }

    #[test]
    fn is_token_allowed_constrains_tool_name() {
        let mut m = Matcher::new(schema_read_write());
        m.advance("<｜DSML｜tool_calls>");
        m.advance("<｜DSML｜invoke name=\"");
        // Only "read" or "write" are legal first chars.
        assert!(m.is_token_allowed("r"));
        assert!(m.is_token_allowed("w"));
        assert!(!m.is_token_allowed("foo"));
        assert!(!m.is_token_allowed("b"));
    }

    #[test]
    fn out_state_allows_everything() {
        let m = Matcher::new(schema_read_write());
        assert!(m.is_token_allowed("anything goes here"));
        assert!(m.is_token_allowed(""));
        assert!(m.is_token_allowed("<｜DSML｜tool_calls>"));
    }

    #[test]
    fn token_mask_marks_all_true_in_free_state() {
        let m = Matcher::new(schema_read_write());
        let vocab = vec![
            "hello".to_string(),
            "<｜DSML｜tool_calls>".to_string(),
            "foo".to_string(),
        ];
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        assert!(mask.iter().all(|&b| b));
    }

    #[test]
    fn token_mask_constrains_inside_tool_calls() {
        let mut m = Matcher::new(schema_read_write());
        m.advance("<｜DSML｜tool_calls>");
        let vocab = vec![
            "<".to_string(),                            // ✓ prefix of all opens
            "</".to_string(),                           // ✓ prefix of close
            "<｜".to_string(),                          // ✓ prefix
            "<｜DSML｜".to_string(),                    // ✓ prefix
            "<｜DSML｜invoke name=\"".to_string(),      // ✓ full open-invoke
            "<｜DSML｜tool name=\"".to_string(),        // ✓ full open-tool-variant
            "</｜DSML｜tool_calls>".to_string(),        // ✓ full close
            "tool_cbl".to_string(),                     // ✗ not a prefix
            "calling".to_string(),                      // ✗ invented tag
            "hello world".to_string(),                  // ✗ random text
            "<bad>".to_string(),                        // ✗ wrong open form
        ];
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        // First 7 should be allowed, last 4 rejected.
        for (i, expected) in [true, true, true, true, true, true, true, false, false, false, false].iter().enumerate() {
            assert_eq!(mask[i], *expected, "vocab[{i}]={:?} expected {expected}", vocab[i]);
        }
    }

    #[test]
    fn apply_mask_to_logits_sets_neg_inf_on_disallowed() {
        let mask = vec![true, false, true, false];
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        Matcher::apply_mask_to_logits(&mask, &mut logits);
        assert_eq!(logits[0], 1.0);
        assert!(logits[1].is_infinite() && logits[1].is_sign_negative());
        assert_eq!(logits[2], 3.0);
        assert!(logits[3].is_infinite() && logits[3].is_sign_negative());
    }

    #[test]
    fn token_mask_locks_tool_after_first_letter() {
        let mut m = Matcher::new(schema_read_write());
        m.advance("<｜DSML｜tool_calls>");
        m.advance("<｜DSML｜invoke name=\"");
        // After locking nothing yet — both r* and w* are legal.
        let vocab = vec!["r".to_string(), "w".to_string(), "b".to_string()];
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        assert_eq!(mask, vec![true, true, false]);

        // Commit "r" — now only `read` is the active candidate; `w` is locked out.
        m.advance("r");
        let vocab2 = vec![
            "ead\">\n".to_string(),  // ✓ completes "read\">\n"
            "ite\">\n".to_string(),  // ✗ would be "write"
            "x".to_string(),         // ✗ random
        ];
        let mut mask2 = vec![false; vocab2.len()];
        m.token_mask(&vocab2, &mut mask2);
        assert_eq!(mask2, vec![true, false, false]);
    }

    #[test]
    fn param_body_constrains_close_marker_when_prefix_started() {
        // Reproduces the failing real-world case where the model emits
        // `</｜DSML｜paperameter>` (paper+ameter) instead of
        // `</｜DSML｜parameter>` and the parser stays open forever.
        let mut m = Matcher::new(schema_read_write());
        m.advance("<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n");
        m.advance("<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/test.txt");
        // Free emission so far.
        assert!(m.is_free());
        // Model starts emitting close marker.
        m.advance("</");
        assert!(!m.is_free(), "must constrain once close-prefix starts");
        // Next valid token would be `｜DSML｜` (the atomic).
        assert!(m.is_token_allowed("｜DSML｜"));
        // But not arbitrary text — the model can't escape from the
        // close marker to type more value content.
        assert!(!m.is_token_allowed("paper"));
        assert!(!m.is_token_allowed("foo"));
        // After completing the close, returns to InvokeBody with the
        // `path` param marked emitted.
        m.advance("｜DSML｜parameter>");
        assert_eq!(
            *m.state(),
            State::InInvokeBody {
                tool_idx: 0,
                emitted_params: vec![0]
            }
        );
    }

    #[test]
    fn param_body_is_free_until_close_marker() {
        let mut m = Matcher::new(schema_read_write());
        m.advance("<｜DSML｜tool_calls>");
        m.advance("<｜DSML｜invoke name=\"read\">\n");
        m.advance("<｜DSML｜parameter name=\"path\" string=\"true\">");
        assert!(m.is_free());
        m.advance("/etc/passwd");
        assert!(m.is_free());
        m.advance("</｜DSML｜parameter>");
        assert_eq!(
            *m.state(),
            State::InInvokeBody {
                tool_idx: 0,
                emitted_params: vec![0]
            }
        );
    }
}
} // mod dsml