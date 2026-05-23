// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek Markup Language (DSML) — tool-calling format for V4 family.
//!
//! Spec source: `huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/encoding/README.md`.
//!
//! DSML wraps tool calls in an XML-style block with special tokens
//! (`<｜DSML｜tool_calls>`, `<｜DSML｜invoke name="…">`,
//! `<｜DSML｜parameter name="…" string="true|false">`, etc.). Parameter
//! value encoding depends on the `string="true|false"` attribute:
//!   - `string="true"`  → raw text payload (no JSON quoting).
//!   - `string="false"` → JSON-encoded value (numbers, booleans, arrays,
//!     objects, or quoted strings).
//!
//! This module provides:
//!   - [`tools_prompt_block`] — render the "## Tools" preamble that gets
//!     prepended to the system message when `tools` is non-empty.
//!   - [`render_assistant_tool_calls`] — serialise prior assistant
//!     tool_call messages back into DSML for multi-turn history.
//!   - [`StreamParser`] — incremental decoder for the streamed token
//!     output: emits [`StreamEvent::Token`] for plain content,
//!     [`StreamEvent::Reasoning`] for `<think>…</think>` content,
//!     [`StreamEvent::ToolCalls`] when a `<｜DSML｜tool_calls>` block
//!     closes. Markers split across token boundaries are buffered
//!     until they resolve.
//!
//! The parser is conservative: any malformed DSML is surfaced as a raw
//! token stream rather than swallowed — the model occasionally emits a
//! near-miss like `<DSML|invoke>` and we'd rather forward the bytes than
//! eat them silently.

use serde_json::{json, Value};

// ── DSML constants — exact strings from the HF docs ─────────────────────

pub const TOOL_CALLS_OPEN: &str = "<｜DSML｜tool_calls>";
pub const TOOL_CALLS_CLOSE: &str = "</｜DSML｜tool_calls>";
pub const INVOKE_OPEN_PREFIX: &str = "<｜DSML｜invoke name=\"";
pub const INVOKE_CLOSE: &str = "</｜DSML｜invoke>";
pub const PARAMETER_OPEN_PREFIX: &str = "<｜DSML｜parameter name=\"";
pub const PARAMETER_CLOSE: &str = "</｜DSML｜parameter>";
pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";
pub const TOOL_RESULT_OPEN: &str = "<tool_result>";
pub const TOOL_RESULT_CLOSE: &str = "</tool_result>";

// ── Structured tool-call type ───────────────────────────────────────────

/// A single tool invocation produced by the model. `arguments` is the
/// reconstructed JSON object whose keys are parameter names and values
/// are decoded per the `string` attribute on each `<｜DSML｜parameter>`.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    /// Serialise this call back into DSML so it can be embedded into a
    /// prior-turn assistant message during multi-turn prompt rendering.
    pub fn to_dsml(&self) -> String {
        let mut out = String::new();
        out.push_str(INVOKE_OPEN_PREFIX);
        out.push_str(&xml_attr_escape(&self.name));
        out.push_str("\">\n");
        if let Value::Object(map) = &self.arguments {
            for (k, v) in map {
                let (string_attr, payload) = match v {
                    Value::String(s) => ("true", s.clone()),
                    other => ("false", other.to_string()),
                };
                out.push_str(PARAMETER_OPEN_PREFIX);
                out.push_str(&xml_attr_escape(k));
                out.push_str("\" string=\"");
                out.push_str(string_attr);
                out.push_str("\">");
                out.push_str(&payload);
                out.push_str(PARAMETER_CLOSE);
                out.push('\n');
            }
        }
        out.push_str(INVOKE_CLOSE);
        out
    }
}

/// Render a slice of tool calls into a `<｜DSML｜tool_calls>`-wrapped block
/// suitable for prepending to (or replacing the body of) a historical
/// assistant turn.
pub fn render_assistant_tool_calls(calls: &[ToolCall]) -> String {
    let mut out = String::new();
    out.push_str(TOOL_CALLS_OPEN);
    out.push('\n');
    for c in calls {
        out.push_str(&c.to_dsml());
        out.push('\n');
    }
    out.push_str(TOOL_CALLS_CLOSE);
    out
}

// ── Prompt-side: render the tools preamble ──────────────────────────────

/// The "## Tools" preamble per HF `encoding/README.md`. Prepended to the
/// system message (or used as the system message if none was supplied)
/// when the request specifies `tools`.
///
/// `tools` is the OpenAI-format tools array, each entry shaped like:
/// ```json
/// { "type": "function", "function": { "name": "...", "description": "...", "parameters": {...} } }
/// ```
/// We dump the JSON verbatim as the schema; the model treats the block
/// as a free-form schema reference.
pub fn tools_prompt_block(tools: &[Value]) -> String {
    let schema = serde_json::to_string_pretty(&json!(tools)).unwrap_or_else(|_| "[]".to_string());
    format!(
"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"{open}\" block like the following:

{open}
{invoke_open}$TOOL_NAME\">
{param_open}$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE{param_close}
...
{invoke_close}
{close}

### Available Tool Schemas

{schema}
",
        open = TOOL_CALLS_OPEN,
        close = TOOL_CALLS_CLOSE,
        invoke_open = INVOKE_OPEN_PREFIX,
        invoke_close = INVOKE_CLOSE,
        param_open = PARAMETER_OPEN_PREFIX,
        param_close = PARAMETER_CLOSE,
        schema = schema,
    )
}

/// Render a tool-result payload to embed inside the user-turn of a
/// follow-up message. `result_json` should already be a JSON-encoded
/// string of whatever the tool returned.
pub fn render_tool_result(result_json: &str) -> String {
    format!("{TOOL_RESULT_OPEN}{result_json}{TOOL_RESULT_CLOSE}")
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
    /// A completed `<｜DSML｜tool_calls>` block parsed into structured
    /// invocations.
    ToolCalls(Vec<ToolCall>),
}

/// Parser state at any moment.
#[derive(Debug)]
enum State {
    /// Plain content. Watching for `<think>` or `<｜DSML｜tool_calls>`.
    Normal,
    /// Inside a `<think>…</think>` block. Watching for `</think>`.
    InThink,
    /// Inside a `<｜DSML｜tool_calls>…</｜DSML｜tool_calls>` block. Watching
    /// for `</｜DSML｜tool_calls>`.
    InToolCalls,
}

/// Incremental parser for the model's streamed token output. Feed each
/// token's decoded text via [`feed`](Self::feed); flush trailing buffered
/// content via [`finish`](Self::finish) at end-of-generation.
///
/// The parser handles markers that arrive split across token boundaries
/// (e.g. one token ends with `<th` and the next starts with `ink>`). It
/// holds back a small lookahead buffer (the length of the longest marker
/// prefix it might still be matching) and emits it as soon as the marker
/// is disambiguated.
pub struct StreamParser {
    state: State,
    /// Bytes seen but not yet emitted in the current state. In Normal,
    /// holds the unsettled tail (potential marker prefix). In InThink
    /// and InToolCalls, accumulates the block content.
    buf: String,
    /// Cached longest marker length (in bytes) we might still be inside,
    /// used as the "hold-back" window in Normal state. Computed once.
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
    /// `<｜DSML｜tool_calls>` blocks are surfaced as best-effort content
    /// so the user sees the partial.
    pub fn finish(mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        match self.state {
            State::Normal => {
                if !self.buf.is_empty() {
                    events.push(StreamEvent::Token(std::mem::take(&mut self.buf)));
                }
            }
            State::InThink => {
                // Unclosed think block — emit what we have as reasoning.
                if !self.buf.is_empty() {
                    events.push(StreamEvent::Reasoning(std::mem::take(&mut self.buf)));
                }
            }
            State::InToolCalls => {
                // Malformed: never saw the close. Surface as raw content.
                let mut raw = String::from(TOOL_CALLS_OPEN);
                raw.push_str(&self.buf);
                events.push(StreamEvent::Token(raw));
            }
        }
        events
    }

    /// One inner step. Returns true if state advanced (more work might
    /// be possible), false if blocked waiting for more input.
    fn step(&mut self, events: &mut Vec<StreamEvent>) -> bool {
        match self.state {
            State::Normal => self.step_normal(events),
            State::InThink => self.step_in_think(events),
            State::InToolCalls => self.step_in_tool_calls(events),
        }
    }

    fn step_normal(&mut self, events: &mut Vec<StreamEvent>) -> bool {
        // Look for the FIRST occurrence of either opener.
        let first_think = self.buf.find(THINK_OPEN);
        let first_tools = self.buf.find(TOOL_CALLS_OPEN);
        let (cut, marker_len, new_state) = match (first_think, first_tools) {
            (None, None) => {
                // Emit everything except a potential marker prefix at
                // the tail. Hold back up to `normal_holdback` bytes.
                if self.buf.len() > self.normal_holdback {
                    let emit_len = self.buf.len() - self.normal_holdback;
                    // Don't split a multi-byte UTF-8 char.
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
        // Emit pre-marker bytes as Token.
        if cut > 0 {
            let head: String = self.buf.drain(..cut).collect();
            events.push(StreamEvent::Token(head));
        }
        // Drop the marker itself.
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
        // No close yet — emit everything up to the last `<` (might be
        // start of `</think>`) so the agent sees thinking progress.
        // Hold back the last `THINK_CLOSE.len()` bytes.
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
        // No close yet: don't emit anything (tool calls are atomic; we
        // surface them only when the block is complete).
        false
    }
}

// ── Tool-call body parsing ──────────────────────────────────────────────

/// Parse the body of a `<｜DSML｜tool_calls>…</｜DSML｜tool_calls>` block
/// (without the wrapper tags themselves) into a vector of [`ToolCall`].
///
/// Best-effort: malformed invocations are skipped; well-formed ones are
/// returned in document order.
pub fn parse_tool_calls_body(body: &str) -> Vec<ToolCall> {
    let mut out = Vec::new();
    let mut cursor = 0;
    while let Some(invoke_start) = body[cursor..].find(INVOKE_OPEN_PREFIX) {
        let abs = cursor + invoke_start + INVOKE_OPEN_PREFIX.len();
        // name attribute runs until the closing `">`.
        let close_attr = match body[abs..].find("\">") {
            Some(i) => abs + i,
            None => break,
        };
        let name = body[abs..close_attr].to_string();
        // Body of invoke runs from after `">` to the matching </｜DSML｜invoke>.
        let body_start = close_attr + 2;
        let invoke_close = match body[body_start..].find(INVOKE_CLOSE) {
            Some(i) => body_start + i,
            None => break,
        };
        let invoke_body = &body[body_start..invoke_close];
        let args = parse_parameters(invoke_body);
        out.push(ToolCall { name, arguments: args });
        cursor = invoke_close + INVOKE_CLOSE.len();
    }
    out
}

/// Parse the parameters inside a single invoke block. Returns a JSON
/// object whose keys are parameter names.
fn parse_parameters(body: &str) -> Value {
    let mut map = serde_json::Map::new();
    let mut cursor = 0;
    while let Some(param_start) = body[cursor..].find(PARAMETER_OPEN_PREFIX) {
        let abs = cursor + param_start + PARAMETER_OPEN_PREFIX.len();
        // name="..." string="true|false">
        let name_end = match body[abs..].find('"') {
            Some(i) => abs + i,
            None => break,
        };
        let name = body[abs..name_end].to_string();
        // Find ` string="...">`
        let after_name = name_end + 1;
        let string_attr_idx = match body[after_name..].find("string=\"") {
            Some(i) => after_name + i + "string=\"".len(),
            None => break,
        };
        let string_attr_end = match body[string_attr_idx..].find('"') {
            Some(i) => string_attr_idx + i,
            None => break,
        };
        let is_string = &body[string_attr_idx..string_attr_end] == "true";
        // After `">` find content + </｜DSML｜parameter>.
        let content_start = match body[string_attr_end..].find("\">") {
            Some(i) => string_attr_end + i + 2,
            None => break,
        };
        let content_end = match body[content_start..].find(PARAMETER_CLOSE) {
            Some(i) => content_start + i,
            None => break,
        };
        let raw_value = &body[content_start..content_end];
        let value: Value = if is_string {
            Value::String(raw_value.to_string())
        } else {
            serde_json::from_str(raw_value.trim()).unwrap_or_else(|_| Value::String(raw_value.to_string()))
        };
        map.insert(name, value);
        cursor = content_end + PARAMETER_CLOSE.len();
    }
    Value::Object(map)
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// XML-attribute escape: replace the four characters that would break
/// attribute parsing. The DSML format is forgiving (no quoting required
/// for parameter values) but tool/parameter names need to be valid.
fn xml_attr_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("&quot;"),
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Find the largest split point ≤ `n` that doesn't cut through a
/// multi-byte UTF-8 character.
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
    fn tool_call_round_trip() {
        let dsml = format!(
            "{open}\n{io}fn1\">\n{po}arg1\" string=\"true\">value one{pc}\n{po}arg2\" string=\"false\">42{pc}\n{ic}\n{close}",
            open = TOOL_CALLS_OPEN,
            close = TOOL_CALLS_CLOSE,
            io = INVOKE_OPEN_PREFIX,
            ic = INVOKE_CLOSE,
            po = PARAMETER_OPEN_PREFIX,
            pc = PARAMETER_CLOSE,
        );
        let mut p = StreamParser::new();
        let mut events = p.feed(&dsml);
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
    fn malformed_tool_call_passes_through_at_finish() {
        // Opens but never closes — finish() flushes as raw token.
        let mut p = StreamParser::new();
        let _ = p.feed(TOOL_CALLS_OPEN);
        let _ = p.feed("\n<｜DSML｜invoke name=\"f\">");
        let events = p.finish();
        let raw: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Token(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(raw.contains(TOOL_CALLS_OPEN));
        assert!(raw.contains("invoke"));
    }
}
