// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek native tool-call format for the V3/V4 family.
//!
//! The model's tokenizer ships nine atomic special tokens for tool calling
//! (vocab IDs 128806–128814 in DeepSeek-V4-Flash):
//!
//! | id     | token                       |
//! |--------|-----------------------------|
//! | 128806 | `<｜tool▁calls▁begin｜>`     |
//! | 128807 | `<｜tool▁calls▁end｜>`       |
//! | 128808 | `<｜tool▁call▁begin｜>`      |
//! | 128809 | `<｜tool▁call▁end｜>`        |
//! | 128810 | `<｜tool▁outputs▁begin｜>`   |
//! | 128811 | `<｜tool▁outputs▁end｜>`     |
//! | 128812 | `<｜tool▁output▁begin｜>`    |
//! | 128813 | `<｜tool▁output▁end｜>`      |
//! | 128814 | `<｜tool▁sep｜>`             |
//!
//! Note the `▁` (U+2581) word-piece separators — these are real characters
//! in the token string, not formatting. The fullwidth pipes `｜` (U+FF5C)
//! are also part of the literal token.
//!
//! Wire format for a tool-call block (what the model emits):
//!
//! ```text
//! <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>FUNCTION_NAME
//! ```json
//! { "argument_name": "value", ... }
//! ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
//! ```
//!
//! Multiple calls go inside a single outer `<｜tool▁calls▁begin｜>` wrapper,
//! each in its own `<｜tool▁call▁begin｜>…<｜tool▁call▁end｜>` envelope.
//!
//! Tool outputs (results fed back to the model in the next user turn) use:
//!
//! ```text
//! <｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>RESULT_PAYLOAD<｜tool▁output▁end｜><｜tool▁outputs▁end｜>
//! ```
//!
//! This module provides:
//!   - [`tools_prompt_block`] — schema preamble appended to the system
//!     message when `tools` is non-empty.
//!   - [`render_assistant_tool_calls`] — serialise prior assistant
//!     tool_call messages for multi-turn history replay.
//!   - [`render_tool_result`] — wrap a tool response for the next user
//!     turn.
//!   - [`StreamParser`] — incremental decoder for streamed model output;
//!     surfaces [`StreamEvent::ToolCalls`] when a complete block arrives,
//!     [`StreamEvent::Reasoning`] for `<think>…</think>` content, and
//!     [`StreamEvent::Token`] for plain text.
//!
//! Malformed blocks fall through as raw [`StreamEvent::Token`] content so
//! the caller still sees the bytes rather than silently losing them.
//!
//! ## Module history
//!
//! Earlier revisions of this module shipped an invented format that
//! wrapped invocations in `<｜DSML｜…>` strings — those token strings do
//! not exist in the V4 vocabulary, so the model BPE-fragmented them on
//! output and tool calls never parsed. Replaced 2026-05-23 with the real
//! token strings extracted from the bundled tokenizer's `added_tokens`.

use serde_json::{json, Value};

// ── Native tool-call tokens — vocab IDs 128806–128814 ───────────────────

pub const TOOL_CALLS_OPEN: &str = "<｜tool▁calls▁begin｜>";
pub const TOOL_CALLS_CLOSE: &str = "<｜tool▁calls▁end｜>";
pub const TOOL_CALL_OPEN: &str = "<｜tool▁call▁begin｜>";
pub const TOOL_CALL_CLOSE: &str = "<｜tool▁call▁end｜>";
pub const TOOL_SEP: &str = "<｜tool▁sep｜>";
pub const TOOL_OUTPUTS_OPEN: &str = "<｜tool▁outputs▁begin｜>";
pub const TOOL_OUTPUTS_CLOSE: &str = "<｜tool▁outputs▁end｜>";
pub const TOOL_OUTPUT_OPEN: &str = "<｜tool▁output▁begin｜>";
pub const TOOL_OUTPUT_CLOSE: &str = "<｜tool▁output▁end｜>";

pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";

// ── Structured tool-call type ───────────────────────────────────────────

/// A single tool invocation produced by the model. `arguments` is parsed
/// from the call's fenced ```json``` body.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    /// Serialise this call into its native single-call envelope, including
    /// the `<｜tool▁call▁begin｜>…<｜tool▁call▁end｜>` markers but NOT the
    /// outer `<｜tool▁calls▁begin｜>` wrapper. Use
    /// [`render_assistant_tool_calls`] to render one or more calls with
    /// the wrapper.
    pub fn to_native(&self) -> String {
        let args_json = serde_json::to_string(&self.arguments)
            .unwrap_or_else(|_| "{}".to_string());
        format!(
            "{open}function{sep}{name}\n```json\n{args}\n```{close}",
            open = TOOL_CALL_OPEN,
            sep = TOOL_SEP,
            name = self.name,
            args = args_json,
            close = TOOL_CALL_CLOSE,
        )
    }
}

/// Render a slice of tool calls into a `<｜tool▁calls▁begin｜>`-wrapped
/// block suitable for embedding into a historical assistant turn.
pub fn render_assistant_tool_calls(calls: &[ToolCall]) -> String {
    let mut out = String::with_capacity(64 + calls.len() * 96);
    out.push_str(TOOL_CALLS_OPEN);
    for c in calls {
        out.push_str(&c.to_native());
    }
    out.push_str(TOOL_CALLS_CLOSE);
    out
}

// ── Prompt-side: render the tools preamble ──────────────────────────────

/// "## Tools" preamble that documents the available function schemas and
/// the native call format. Appended to the system message (or used as the
/// system message if none was supplied) when `tools` is non-empty.
///
/// `tools` is the OpenAI-format tools array, each entry shaped like:
/// ```json
/// { "type": "function", "function": { "name": "...", "description": "...", "parameters": {...} } }
/// ```
pub fn tools_prompt_block(tools: &[Value]) -> String {
    let schema = serde_json::to_string_pretty(&json!(tools))
        .unwrap_or_else(|_| "[]".to_string());
    format!(
"## Tools

You can call any of the following functions to help answer the user's request. Available function schemas:

{schema}

To invoke a function, emit a tool-call block in this exact form (multiple calls can sit inside one outer wrapper):

{open}{call_open}function{sep}FUNCTION_NAME
```json
{{\"argument_name\": \"value\"}}
```{call_close}{close}

Arguments are a single JSON object whose keys match the function's `parameters.properties`. After every tool-call block, wait for a tool-output message before continuing.",
        open = TOOL_CALLS_OPEN,
        close = TOOL_CALLS_CLOSE,
        call_open = TOOL_CALL_OPEN,
        call_close = TOOL_CALL_CLOSE,
        sep = TOOL_SEP,
        schema = schema,
    )
}

/// Wrap a single tool-result payload in the native outputs envelope.
/// `result_body` is the raw payload string (JSON-encoded or plain text —
/// caller's choice; the model treats it as opaque content).
pub fn render_tool_result(result_body: &str) -> String {
    format!(
        "{outputs_open}{output_open}{body}{output_close}{outputs_close}",
        outputs_open = TOOL_OUTPUTS_OPEN,
        output_open = TOOL_OUTPUT_OPEN,
        body = result_body,
        output_close = TOOL_OUTPUT_CLOSE,
        outputs_close = TOOL_OUTPUTS_CLOSE,
    )
}

// ── Output-side: streaming parser ───────────────────────────────────────

/// Event emitted by [`StreamParser::feed`] / [`StreamParser::finish`].
#[derive(Debug, PartialEq)]
pub enum StreamEvent {
    /// Plain content the agent should surface to the user.
    Token(String),
    /// Content from inside a `<think>…</think>` block. Caller decides
    /// whether to surface it (e.g. mapped to OpenAI `reasoning_content`)
    /// or drop it.
    Reasoning(String),
    /// A completed `<｜tool▁calls▁begin｜>…<｜tool▁calls▁end｜>` block parsed
    /// into structured invocations.
    ToolCalls(Vec<ToolCall>),
}

#[derive(Debug)]
enum State {
    /// Plain content. Watching for `<think>` or `<｜tool▁calls▁begin｜>`.
    Normal,
    /// Inside a `<think>…</think>` block.
    InThink,
    /// Inside a `<｜tool▁calls▁begin｜>…<｜tool▁calls▁end｜>` block.
    InToolCalls,
}

/// Incremental parser for the model's streamed token output. Feed each
/// token's decoded text via [`feed`](Self::feed); flush trailing buffered
/// content via [`finish`](Self::finish) at end-of-generation.
///
/// Handles markers that arrive split across token boundaries — holds back
/// a small lookahead buffer (the length of the longest unresolved prefix)
/// and emits it as soon as the marker is disambiguated.
pub struct StreamParser {
    state: State,
    buf: String,
    normal_holdback: usize,
}

impl Default for StreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamParser {
    pub fn new() -> Self {
        let holdback = TOOL_CALLS_OPEN.len().max(THINK_OPEN.len());
        Self {
            state: State::Normal,
            buf: String::new(),
            normal_holdback: holdback,
        }
    }

    /// Feed a chunk of decoded text. May emit zero, one, or many events.
    pub fn feed(&mut self, chunk: &str) -> Vec<StreamEvent> {
        self.buf.push_str(chunk);
        let mut events = Vec::new();
        loop {
            let progressed = self.step(&mut events);
            if !progressed {
                break;
            }
        }
        events
    }

    /// End-of-stream: flush whatever's buffered. Unclosed `<think>` or
    /// tool-calls blocks are surfaced as best-effort content so the user
    /// sees the partial.
    pub fn finish(mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        match self.state {
            State::Normal => {
                if !self.buf.is_empty() {
                    events.push(StreamEvent::Token(std::mem::take(&mut self.buf)));
                }
            }
            State::InThink => {
                if !self.buf.is_empty() {
                    events.push(StreamEvent::Reasoning(std::mem::take(&mut self.buf)));
                }
            }
            State::InToolCalls => {
                let mut raw = String::from(TOOL_CALLS_OPEN);
                raw.push_str(&self.buf);
                events.push(StreamEvent::Token(raw));
            }
        }
        events
    }

    fn step(&mut self, events: &mut Vec<StreamEvent>) -> bool {
        match self.state {
            State::Normal => self.step_normal(events),
            State::InThink => self.step_in_think(events),
            State::InToolCalls => self.step_in_tool_calls(events),
        }
    }

    fn step_normal(&mut self, events: &mut Vec<StreamEvent>) -> bool {
        let first_think = self.buf.find(THINK_OPEN);
        let first_tools = self.buf.find(TOOL_CALLS_OPEN);
        let (cut, marker_len, new_state) = match (first_think, first_tools) {
            (None, None) => {
                if self.buf.len() > self.normal_holdback {
                    let emit_len = self.buf.len() - self.normal_holdback;
                    let emit_len = utf8_safe_split(&self.buf, emit_len);
                    if emit_len > 0 {
                        let emitted: String = self.buf.drain(..emit_len).collect();
                        events.push(StreamEvent::Token(emitted));
                        return true;
                    }
                }
                return false;
            }
            (Some(t), None) => (t, THINK_OPEN.len(), State::InThink),
            (None, Some(t)) => (t, TOOL_CALLS_OPEN.len(), State::InToolCalls),
            (Some(a), Some(b)) => {
                if a <= b {
                    (a, THINK_OPEN.len(), State::InThink)
                } else {
                    (b, TOOL_CALLS_OPEN.len(), State::InToolCalls)
                }
            }
        };
        if cut > 0 {
            let head: String = self.buf.drain(..cut).collect();
            events.push(StreamEvent::Token(head));
        }
        let _: String = self.buf.drain(..marker_len).collect();
        self.state = new_state;
        true
    }

    fn step_in_think(&mut self, events: &mut Vec<StreamEvent>) -> bool {
        if let Some(idx) = self.buf.find(THINK_CLOSE) {
            let content: String = self.buf.drain(..idx).collect();
            if !content.is_empty() {
                events.push(StreamEvent::Reasoning(content));
            }
            let _: String = self.buf.drain(..THINK_CLOSE.len()).collect();
            self.state = State::Normal;
            return true;
        }
        let holdback = THINK_CLOSE.len();
        if self.buf.len() > holdback {
            let emit_len = utf8_safe_split(&self.buf, self.buf.len() - holdback);
            if emit_len > 0 {
                let emitted: String = self.buf.drain(..emit_len).collect();
                events.push(StreamEvent::Reasoning(emitted));
                return true;
            }
        }
        false
    }

    fn step_in_tool_calls(&mut self, events: &mut Vec<StreamEvent>) -> bool {
        if let Some(idx) = self.buf.find(TOOL_CALLS_CLOSE) {
            let body: String = self.buf.drain(..idx).collect();
            let _: String = self.buf.drain(..TOOL_CALLS_CLOSE.len()).collect();
            self.state = State::Normal;
            let calls = parse_tool_calls_body(&body);
            events.push(StreamEvent::ToolCalls(calls));
            return true;
        }
        false
    }
}

// ── Tool-call body parsing ──────────────────────────────────────────────

/// Parse the body between `<｜tool▁calls▁begin｜>` and `<｜tool▁calls▁end｜>`.
/// Each inner call is wrapped in `<｜tool▁call▁begin｜>…<｜tool▁call▁end｜>`
/// with the shape `function<｜tool▁sep｜>NAME\n```json\n{...}\n```\n?`.
///
/// Best-effort: malformed invocations are skipped; well-formed ones are
/// returned in document order.
pub fn parse_tool_calls_body(body: &str) -> Vec<ToolCall> {
    let mut out = Vec::new();
    let mut cursor = 0;
    while let Some(rel) = body[cursor..].find(TOOL_CALL_OPEN) {
        let inner_start = cursor + rel + TOOL_CALL_OPEN.len();
        let close = match body[inner_start..].find(TOOL_CALL_CLOSE) {
            Some(i) => inner_start + i,
            None => break,
        };
        if let Some(call) = parse_one_call(&body[inner_start..close]) {
            out.push(call);
        }
        cursor = close + TOOL_CALL_CLOSE.len();
    }
    out
}

/// Parse a single call's interior bytes (between the begin/end markers).
fn parse_one_call(inner: &str) -> Option<ToolCall> {
    // The prefix before <｜tool▁sep｜> is the call type tag (always
    // "function" in V3/V4); we don't enforce it — accept any prefix and
    // skip straight to the separator.
    let sep_at = inner.find(TOOL_SEP)?;
    let after_sep = sep_at + TOOL_SEP.len();
    let name_end = inner[after_sep..]
        .find('\n')
        .map(|i| after_sep + i)
        .unwrap_or(inner.len());
    let name = inner[after_sep..name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    // Locate the ```json ... ``` fenced block.
    const FENCE_OPEN: &str = "```json";
    const FENCE_CLOSE: &str = "```";
    let fence_open_rel = inner[name_end..].find(FENCE_OPEN)?;
    let payload_start = name_end + fence_open_rel + FENCE_OPEN.len();
    let payload_end_rel = inner[payload_start..].find(FENCE_CLOSE)?;
    let raw = inner[payload_start..payload_start + payload_end_rel].trim();
    let arguments: Value = serde_json::from_str(raw)
        .unwrap_or_else(|_| Value::String(raw.to_string()));
    Some(ToolCall { name, arguments })
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Largest split point ≤ `n` that doesn't cut through a multi-byte UTF-8
/// character.
fn utf8_safe_split(s: &str, n: usize) -> usize {
    let bytes = s.as_bytes();
    let mut k = n.min(bytes.len());
    while k > 0 && (bytes[k] & 0b1100_0000) == 0b1000_0000 {
        k -= 1;
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(p: &mut StreamParser, s: &str) -> Vec<StreamEvent> {
        p.feed(s)
    }

    #[test]
    fn plain_text_pass_through() {
        let mut p = StreamParser::new();
        let mut events = drain(&mut p, "hello world");
        events.extend(p.finish());
        let joined: String = events
            .iter()
            .map(|e| match e {
                StreamEvent::Token(t) => t.as_str(),
                _ => "",
            })
            .collect();
        assert_eq!(joined, "hello world");
    }

    #[test]
    fn think_block_isolated() {
        let mut p = StreamParser::new();
        let mut events = drain(&mut p, "before<think>reasoning</think>after");
        events.extend(p.finish());
        let mut tokens = String::new();
        let mut reasoning = String::new();
        for e in &events {
            match e {
                StreamEvent::Token(t) => tokens.push_str(t),
                StreamEvent::Reasoning(t) => reasoning.push_str(t),
                _ => {}
            }
        }
        assert_eq!(tokens, "beforeafter");
        assert_eq!(reasoning, "reasoning");
    }

    #[test]
    fn think_split_across_chunks() {
        let mut p = StreamParser::new();
        let mut events = drain(&mut p, "x<th");
        events.extend(drain(&mut p, "ink>r1"));
        events.extend(drain(&mut p, "r2</th"));
        events.extend(drain(&mut p, "ink>y"));
        events.extend(p.finish());
        let mut reasoning = String::new();
        let mut tokens = String::new();
        for e in &events {
            match e {
                StreamEvent::Token(t) => tokens.push_str(t),
                StreamEvent::Reasoning(t) => reasoning.push_str(t),
                _ => {}
            }
        }
        assert_eq!(reasoning, "r1r2");
        assert_eq!(tokens, "xy");
    }

    #[test]
    fn single_tool_call_round_trip() {
        let payload = format!(
            "{open}{co}function{sep}fn1\n```json\n{{\"arg1\":\"value one\",\"arg2\":42}}\n```{cc}{close}",
            open = TOOL_CALLS_OPEN,
            close = TOOL_CALLS_CLOSE,
            co = TOOL_CALL_OPEN,
            cc = TOOL_CALL_CLOSE,
            sep = TOOL_SEP,
        );
        let mut p = StreamParser::new();
        let mut events = p.feed(&payload);
        events.extend(p.finish());
        let calls: Vec<&Vec<ToolCall>> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCalls(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].len(), 1);
        assert_eq!(calls[0][0].name, "fn1");
        assert_eq!(calls[0][0].arguments["arg1"], json!("value one"));
        assert_eq!(calls[0][0].arguments["arg2"], json!(42));
    }

    #[test]
    fn two_calls_one_wrapper() {
        let payload = format!(
            "{open}{co}function{sep}a\n```json\n{{\"x\":1}}\n```{cc}{co}function{sep}b\n```json\n{{\"y\":2}}\n```{cc}{close}",
            open = TOOL_CALLS_OPEN,
            close = TOOL_CALLS_CLOSE,
            co = TOOL_CALL_OPEN,
            cc = TOOL_CALL_CLOSE,
            sep = TOOL_SEP,
        );
        let mut p = StreamParser::new();
        let mut events = p.feed(&payload);
        events.extend(p.finish());
        let calls: Vec<&Vec<ToolCall>> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCalls(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].len(), 2);
        assert_eq!(calls[0][0].name, "a");
        assert_eq!(calls[0][0].arguments["x"], json!(1));
        assert_eq!(calls[0][1].name, "b");
        assert_eq!(calls[0][1].arguments["y"], json!(2));
    }

    #[test]
    fn renderer_parser_round_trip() {
        let calls = vec![ToolCall {
            name: "read".into(),
            arguments: json!({"path": "/tmp/x", "limit": 100}),
        }];
        let rendered = render_assistant_tool_calls(&calls);
        let mut p = StreamParser::new();
        let mut events = p.feed(&rendered);
        events.extend(p.finish());
        let parsed: Vec<&Vec<ToolCall>> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCalls(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].len(), 1);
        assert_eq!(parsed[0][0].name, "read");
        assert_eq!(parsed[0][0].arguments["path"], json!("/tmp/x"));
        assert_eq!(parsed[0][0].arguments["limit"], json!(100));
    }

    #[test]
    fn malformed_tool_call_passes_through_at_finish() {
        let mut p = StreamParser::new();
        let _ = p.feed(TOOL_CALLS_OPEN);
        let _ = p.feed("function|fake|fn not closed");
        let events = p.finish();
        let raw: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Token(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(raw.contains(TOOL_CALLS_OPEN));
        assert!(raw.contains("function"));
    }

    #[test]
    fn tool_result_envelope_shape() {
        let s = render_tool_result(r#"{"result":"ok"}"#);
        assert!(s.starts_with(TOOL_OUTPUTS_OPEN));
        assert!(s.ends_with(TOOL_OUTPUTS_CLOSE));
        assert!(s.contains(TOOL_OUTPUT_OPEN));
        assert!(s.contains(TOOL_OUTPUT_CLOSE));
        assert!(s.contains(r#"{"result":"ok"}"#));
    }

    #[test]
    fn tool_call_split_across_chunks() {
        // Stress: the open marker arrives in pieces across feeds.
        let full = format!(
            "{open}{co}function{sep}f\n```json\n{{\"a\":1}}\n```{cc}{close}",
            open = TOOL_CALLS_OPEN,
            close = TOOL_CALLS_CLOSE,
            co = TOOL_CALL_OPEN,
            cc = TOOL_CALL_CLOSE,
            sep = TOOL_SEP,
        );
        let mut p = StreamParser::new();
        let mut events = Vec::new();
        // 7-byte chunks force splits inside multibyte sequences and inside
        // markers (utf8_safe_split protects multibyte boundaries).
        let bytes = full.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            // Find a safe split point.
            let mut end = (i + 7).min(bytes.len());
            while end < bytes.len() && (bytes[end] & 0b1100_0000) == 0b1000_0000 {
                end += 1;
            }
            let chunk = std::str::from_utf8(&bytes[i..end]).unwrap();
            events.extend(p.feed(chunk));
            i = end;
        }
        events.extend(p.finish());
        let calls: Vec<&Vec<ToolCall>> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCalls(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0][0].name, "f");
        assert_eq!(calls[0][0].arguments["a"], json!(1));
    }
}
