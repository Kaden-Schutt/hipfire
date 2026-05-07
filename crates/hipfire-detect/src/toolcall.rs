//! Tool-call shape detector — port of `scripts/agentic-gate.sh:487-509`.
//!
//! Runs four sub-checks against every `<tool_call>...</tool_call>`
//! block in the visible-text stream:
//!
//! - Stacked openers: `<tool_call>\s*<tool_call>` → HARD
//! - JSON parse fail (body is not valid JSON) → HARD
//! - Schema fail (parsed JSON missing `name` or `arguments`) → HARD
//! - Tool call inside `<think>...</think>` → SOFT
//!
//! Auto-engagement is decided by the probe binary. When this detector
//! is in the bank, it fires on any visible tool_call. When the bank
//! does not include it, no checks happen.

use crate::{Detector, Event, Verdict};
use regex::Regex;

pub struct ToolcallShape {
    buf: String,
    /// First fired sub-check, in order: stacked, parse, schema. Sticky.
    hard: Option<String>,
    soft: Vec<String>,
    saw_any: bool,
}

impl ToolcallShape {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            hard: None,
            soft: Vec::new(),
            saw_any: false,
        }
    }
}

impl Default for ToolcallShape {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector for ToolcallShape {
    fn name(&self) -> &'static str {
        "toolcall_shape"
    }

    fn observe(&mut self, ev: &Event<'_>) -> Option<Verdict> {
        if let Event::Token { text, .. } = ev {
            self.buf.push_str(text);
        }
        None
    }

    fn finalize(&mut self) -> Verdict {
        let stacked = Regex::new(r"<tool_call>\s*<tool_call>").unwrap();
        if stacked.is_match(&self.buf) {
            self.hard = Some("stacked openers (<tool_call><tool_call>)".to_string());
        }

        // Iterate every <tool_call>...</tool_call> block.
        let block = Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>").unwrap();
        for caps in block.captures_iter(&self.buf) {
            self.saw_any = true;
            let body = caps.get(1).map(|m| m.as_str()).unwrap_or("");

            // Strip a nested opener if present (a stacked-openers block
            // would otherwise fail JSON parse with that as a header).
            let body_clean = body.replacen("<tool_call>", "", 1);
            let body_clean = body_clean.trim();

            // JSON parse.
            let parsed: serde_json::Result<serde_json::Value> =
                serde_json::from_str(body_clean);
            match parsed {
                Err(e) => {
                    if self.hard.is_none() {
                        self.hard = Some(format!("tool_call body fails JSON parse: {}", e));
                    }
                }
                Ok(v) => {
                    let obj = v.as_object();
                    let has_name = obj.map(|o| o.contains_key("name")).unwrap_or(false);
                    let has_args = obj.map(|o| o.contains_key("arguments")).unwrap_or(false);
                    if !(has_name && has_args) {
                        if self.hard.is_none() {
                            self.hard = Some(format!(
                                "tool_call body missing required field(s): name={}, arguments={}",
                                has_name, has_args
                            ));
                        }
                    }
                }
            }
        }

        // Soft: tool_call appearing inside a <think>...</think> block.
        let in_think = Regex::new(r"(?s)<think>.*?<tool_call>.*?</think>").unwrap();
        if in_think.is_match(&self.buf) {
            self.soft.push("tool_call emitted inside <think>".to_string());
        }

        if let Some(reason) = self.hard.take() {
            return Verdict::fail(reason);
        }
        if !self.soft.is_empty() {
            return Verdict::warn(self.soft.join("; "));
        }
        Verdict::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(t: &str) -> Event<'_> {
        Event::Token {
            text: t,
            t_ms: 0,
            synthetic: false,
        }
    }

    fn run(payload: &str) -> Verdict {
        let mut d = ToolcallShape::new();
        d.observe(&tok(payload));
        d.finalize()
    }

    #[test]
    fn clean_tool_call_passes() {
        let v = run(
            r#"thinking is done.
<tool_call>
{"name": "read", "arguments": {"path": "/tmp/x"}}
</tool_call><|im_end|>"#,
        );
        assert!(matches!(v, Verdict::Ok), "got {:?}", v);
    }

    #[test]
    fn stacked_openers_hard_fail() {
        let v = run(
            r#"<tool_call>
<tool_call>
{"name": "x", "arguments": {}}
</tool_call>"#,
        );
        assert!(v.is_fail(), "got {:?}", v);
    }

    #[test]
    fn malformed_json_hard_fail() {
        let v = run(r#"<tool_call>{"name": "read", "arguments": {oops}}</tool_call>"#);
        assert!(v.is_fail(), "got {:?}", v);
    }

    #[test]
    fn missing_arguments_field_hard_fail() {
        let v = run(r#"<tool_call>{"name": "read"}</tool_call>"#);
        assert!(v.is_fail(), "got {:?}", v);
    }

    #[test]
    fn missing_name_field_hard_fail() {
        let v = run(r#"<tool_call>{"arguments": {"x": 1}}</tool_call>"#);
        assert!(v.is_fail(), "got {:?}", v);
    }

    #[test]
    fn tool_call_inside_think_soft_warn() {
        let v = run(
            r#"<think>let me <tool_call>{"name":"a","arguments":{}}</tool_call> first</think>
<tool_call>{"name": "real", "arguments": {}}</tool_call>"#,
        );
        assert!(v.is_warn(), "got {:?}", v);
    }
}
