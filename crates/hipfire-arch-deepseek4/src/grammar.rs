// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

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
