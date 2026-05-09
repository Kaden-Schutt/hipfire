//! Per-arch tool-call output parsers.
//!
//! Stage 3 of the Jinja transition: move tool-call parsing out of
//! `cli/index.ts:parseToolCalls` (TypeScript, single-format JSON) into
//! the Rust runtime so the daemon can parse per-arch and emit structured
//! `tool_calls` events on the SSE stream. Once Stage 5 lands, the CLI
//! becomes a passthrough and `cli/parse_tool_calls.test.ts` is archived.
//!
//! Three implementations cover the formats observed in production:
//!
//! - [`Qwen35XmlParser`] — Qwen3.5/3.6 (dense + MoE/A3B + VL) emit
//!   `<tool_call><function=NAME><parameter=ARG>\nVALUE\n</parameter>...</function></tool_call>`
//!   per their upstream chat_template's tools-block instructions.
//!   This is the format scenario C of `jinja_smoke` validated all
//!   four target models emit cleanly.
//!
//! - [`Gemma4NativeParser`] — Gemma 4 emits its own
//!   `<|tool_call|>{...json...}<|/tool_call|>` shape using literal
//!   special tokens. (Wired in via Stage 3; arch-gemma4 crate pulled
//!   in alongside Stage 7 — Gemma 4 itself is not present on master
//!   yet.)
//!
//! - [`HermesJsonParser`] — bare `<tool_call>{...JSON...}</tool_call>`
//!   shape. Port of `cli/index.ts:parseToolCalls` + `parseOneToolCall`.
//!   Includes the MQ4 #111 stopgap repairs (stacked-opener strip,
//!   flat-object coercion, XML-tag head fallback). This is the
//!   default `Architecture::tool_call_parser` because most current
//!   model paths still go through the daemon's hand-rolled Hermes
//!   prompt — Stage 5 flips that.

use serde::{Deserialize, Serialize};

/// One assistant-emitted tool call recovered from the model's output
/// stream. `arguments` is a free-form JSON value (typically an object).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
    /// `true` when the parser had to coerce off-spec input (flat object
    /// with sibling args, XML-tag `<plain></param>` shape, stacked
    /// `<tool_call>` openers from the MQ4 #111 attractor, etc.). The
    /// daemon emits a stderr trace when this is set so operators can
    /// see when the legacy repair path saved a request.
    pub repaired: bool,
}

/// Result of parsing an assistant-emitted output buffer for tool calls.
///
/// `prose` is everything before the first `<tool_call>` opener, trimmed
/// — the visible "natural language" content the user sees. `tool_calls`
/// is the list of recovered structured calls.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolCallParseResult {
    pub prose: Option<String>,
    pub tool_calls: Vec<ParsedToolCall>,
}

/// Parser surface. Each arch's preferred format gets its own
/// implementation; `Architecture::tool_call_parser` selects the
/// right one at load time.
pub trait ToolCallParser: Send + Sync {
    /// Parse the full assistant output (after EOS / generation stop).
    /// Streaming partial-output parsing is a follow-up; for now the
    /// daemon buffers complete responses and parses once.
    fn parse(&self, text: &str) -> ToolCallParseResult;

    /// Diagnostic name for logs.
    fn name(&self) -> &'static str;
}

// ── Hermes JSON parser ──────────────────────────────────────────────

/// Hermes-style: `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.
/// Port of `cli/index.ts:parseToolCalls` + `parseOneToolCall` with the
/// MQ4 #111 stopgap repairs (stacked-opener strip, flat-object
/// coercion, XML-tag fallback).
pub struct HermesJsonParser;

impl HermesJsonParser {
    pub fn new() -> Self { Self }
}

impl Default for HermesJsonParser {
    fn default() -> Self { Self::new() }
}

impl ToolCallParser for HermesJsonParser {
    fn name(&self) -> &'static str { "hermes_json" }

    fn parse(&self, text: &str) -> ToolCallParseResult {
        if !text.contains("<tool_call>") {
            return ToolCallParseResult {
                prose: trim_to_option(text),
                tool_calls: Vec::new(),
            };
        }
        let mut tool_calls = Vec::new();
        let mut cursor = 0usize;
        while let Some(open_off) = text[cursor..].find("<tool_call>") {
            let open_abs = cursor + open_off;
            let body_start = open_abs + "<tool_call>".len();
            let close_off = text[body_start..].find("</tool_call>");
            let body_end = close_off
                .map(|o| body_start + o)
                .unwrap_or(text.len());
            let raw = text[body_start..body_end].trim();
            cursor = match close_off {
                Some(_) => body_end + "</tool_call>".len(),
                None => text.len(),
            };
            // Strip MQ4-#111 stacked openers — sometimes the model
            // emits 1-2 nested `<tool_call>` before the JSON body.
            let mut working = raw.to_string();
            let mut stripped = 0;
            while working.starts_with("<tool_call>") {
                working = working["<tool_call>".len()..].trim_start().to_string();
                stripped += 1;
            }
            if working.is_empty() { continue; }
            if let Some(parsed) = parse_one_hermes(&working, stripped > 0) {
                tool_calls.push(parsed);
            }
            if close_off.is_none() { break; }
        }
        // Prose is the text before the FIRST `<tool_call>` opener.
        let prose = match text.find("<tool_call>") {
            Some(idx) => trim_to_option(&text[..idx]),
            None => trim_to_option(text),
        };
        ToolCallParseResult { prose, tool_calls }
    }
}

fn parse_one_hermes(raw: &str, already_repaired: bool) -> Option<ParsedToolCall> {
    // Form 1: spec-compliant `{"name": ..., "arguments": {...}}`.
    if let Ok(tc) = serde_json::from_str::<serde_json::Value>(raw) {
        if let serde_json::Value::Object(map) = &tc {
            if let Some(name) = map.get("name").and_then(|v| v.as_str()) {
                if let Some(args) = map.get("arguments") {
                    return Some(ParsedToolCall {
                        name: name.to_string(),
                        arguments: args.clone(),
                        repaired: already_repaired,
                    });
                }
                // Form 2: flat object — sibling args without an
                // `arguments` wrapper. Promote everything except a few
                // known metadata keys.
                let drop: &[&str] = &["name", "type", "id", "function"];
                let mut args = serde_json::Map::new();
                let mut coerced = false;
                for (k, v) in map.iter() {
                    if drop.contains(&k.as_str()) { continue; }
                    args.insert(k.clone(), v.clone());
                    coerced = true;
                }
                if coerced {
                    return Some(ParsedToolCall {
                        name: name.to_string(),
                        arguments: serde_json::Value::Object(args),
                        repaired: true,
                    });
                }
                // Bare `{"name": "X"}` is legal for zero-arg tools.
                return Some(ParsedToolCall {
                    name: name.to_string(),
                    arguments: serde_json::json!({}),
                    repaired: already_repaired,
                });
            }
        }
    }

    // Form 3: XML-tag head + balanced JSON. Patterns observed in MQ4
    // attractor output:
    //   <plain>NAME</param> {...}
    //   <function=NAME> {...}
    //   <tool name="NAME"> {...}
    let candidates: &[(&str, &str)] = &[
        ("<plain>", "</param>"),
        ("<function=", ">"),
        ("<tool name=\"", "\">"),
    ];
    for (open, close) in candidates {
        if let Some(rest) = raw.strip_prefix(open) {
            if let Some(close_idx) = rest.find(close) {
                let name = rest[..close_idx].trim().trim_matches('"');
                if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.') {
                    let after = rest[close_idx + close.len()..].trim();
                    let args = extract_first_json_object(after).unwrap_or(serde_json::json!({}));
                    return Some(ParsedToolCall {
                        name: name.to_string(),
                        arguments: args,
                        repaired: true,
                    });
                }
            }
        }
    }
    None
}

/// Extract the first balanced top-level JSON object from `s`. Returns
/// the parsed `Value` or `None` if no balanced object found.
fn extract_first_json_object(s: &str) -> Option<serde_json::Value> {
    let start = s.find('{')?;
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escape = false;
    for i in start..bytes.len() {
        let ch = bytes[i] as char;
        if in_str {
            if escape { escape = false; continue; }
            if ch == '\\' { escape = true; continue; }
            if ch == '"' { in_str = false; }
            continue;
        }
        match ch {
            '"' => in_str = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let candidate = &s[start..=i];
                    return serde_json::from_str(candidate).ok();
                }
            }
            _ => {}
        }
    }
    None
}

fn trim_to_option(s: &str) -> Option<String> {
    let trimmed = s.trim();
    if trimmed.is_empty() { None } else { Some(trimmed.to_string()) }
}

// ── Qwen3.5/3.6 XML parser ──────────────────────────────────────────

/// Qwen3.5/3.6 native XML format:
///
/// ```text
/// <tool_call>
/// <function=NAME>
/// <parameter=ARG1>
/// VALUE1
/// </parameter>
/// <parameter=ARG2>
/// VALUE2
/// </parameter>
/// </function>
/// </tool_call>
/// ```
///
/// VALUE may contain literal `<` characters inside string-like values;
/// the scanner uses `</parameter>` as the close marker (not just `<`)
/// and is escape-aware on parameter contents.
pub struct Qwen35XmlParser;

impl Qwen35XmlParser {
    pub fn new() -> Self { Self }
}

impl Default for Qwen35XmlParser {
    fn default() -> Self { Self::new() }
}

impl ToolCallParser for Qwen35XmlParser {
    fn name(&self) -> &'static str { "qwen35_xml" }

    fn parse(&self, text: &str) -> ToolCallParseResult {
        if !text.contains("<tool_call>") {
            return ToolCallParseResult {
                prose: trim_to_option(text),
                tool_calls: Vec::new(),
            };
        }
        let mut tool_calls = Vec::new();
        let mut cursor = 0usize;
        while let Some(open_off) = text[cursor..].find("<tool_call>") {
            let open_abs = cursor + open_off;
            let body_start = open_abs + "<tool_call>".len();
            let close_off = text[body_start..].find("</tool_call>");
            let body_end = close_off.map(|o| body_start + o).unwrap_or(text.len());
            let body = text[body_start..body_end].trim();
            cursor = match close_off {
                Some(_) => body_end + "</tool_call>".len(),
                None => text.len(),
            };
            if let Some(parsed) = parse_one_qwen35_xml(body) {
                tool_calls.push(parsed);
            } else {
                // XML parse failed — try Hermes JSON fallback for this
                // block (MQ4 attractor sometimes flips a Qwen XML model
                // into JSON shape). Lets `Qwen35XmlParser` recover
                // without a separate format hint.
                if let Some(parsed) = parse_one_hermes(body, false) {
                    tool_calls.push(ParsedToolCall { repaired: true, ..parsed });
                }
            }
            if close_off.is_none() { break; }
        }
        let prose = match text.find("<tool_call>") {
            Some(idx) => trim_to_option(&text[..idx]),
            None => trim_to_option(text),
        };
        ToolCallParseResult { prose, tool_calls }
    }
}

fn parse_one_qwen35_xml(body: &str) -> Option<ParsedToolCall> {
    // Find <function=NAME> opener.
    let fn_open = "<function=";
    let fn_idx = body.find(fn_open)?;
    let after_fn_open = &body[fn_idx + fn_open.len()..];
    let name_end = after_fn_open.find('>')?;
    let name = after_fn_open[..name_end].trim();
    if name.is_empty() { return None; }
    // Walk parameters between <function=NAME> and </function>.
    let after_name_open = &after_fn_open[name_end + 1..];
    let fn_close = after_name_open.find("</function>").unwrap_or(after_name_open.len());
    let params_region = &after_name_open[..fn_close];
    let mut args = serde_json::Map::new();
    let mut walker = params_region;
    while let Some(p_open) = walker.find("<parameter=") {
        let after_p_open = &walker[p_open + "<parameter=".len()..];
        let arg_end = match after_p_open.find('>') {
            Some(i) => i,
            None => break,
        };
        let arg_name = after_p_open[..arg_end].trim();
        let after_arg_open = &after_p_open[arg_end + 1..];
        let p_close = match after_arg_open.find("</parameter>") {
            Some(i) => i,
            None => after_arg_open.len(),
        };
        // Value may have a leading + trailing newline (template
        // emits `<parameter=ARG>\nVALUE\n</parameter>`); strip them.
        let value_str = after_arg_open[..p_close].trim_matches('\n').trim().to_string();
        // If the value parses as a JSON value (e.g., model emitted
        // `{"key": "..."}` because the parameter is structured), use
        // that. Otherwise treat as a raw string.
        let value_json: serde_json::Value = serde_json::from_str(&value_str)
            .unwrap_or_else(|_| serde_json::Value::String(value_str));
        if !arg_name.is_empty() {
            args.insert(arg_name.to_string(), value_json);
        }
        walker = match after_arg_open[p_close..].find("</parameter>") {
            Some(i) => &after_arg_open[p_close + i + "</parameter>".len()..],
            None => "",
        };
    }
    Some(ParsedToolCall {
        name: name.to_string(),
        arguments: serde_json::Value::Object(args),
        repaired: false,
    })
}

// ── Gemma 4 native parser ───────────────────────────────────────────

/// Gemma 4's native shape: `<|tool_call|>{...JSON...}<|/tool_call|>`.
/// Same JSON body shape as Hermes but wrapped in literal special-token
/// markers. Accepts `name`+`args` (Gemma's spec) or `name`+`arguments`
/// (Hermes-spec compatibility).
pub struct Gemma4NativeParser;

impl Gemma4NativeParser {
    pub fn new() -> Self { Self }
}

impl Default for Gemma4NativeParser {
    fn default() -> Self { Self::new() }
}

impl ToolCallParser for Gemma4NativeParser {
    fn name(&self) -> &'static str { "gemma4_native" }

    fn parse(&self, text: &str) -> ToolCallParseResult {
        const OPEN: &str = "<|tool_call|>";
        const CLOSE: &str = "<|/tool_call|>";
        if !text.contains(OPEN) {
            return ToolCallParseResult {
                prose: trim_to_option(text),
                tool_calls: Vec::new(),
            };
        }
        let mut tool_calls = Vec::new();
        let mut cursor = 0usize;
        while let Some(open_off) = text[cursor..].find(OPEN) {
            let open_abs = cursor + open_off;
            let body_start = open_abs + OPEN.len();
            let close_off = text[body_start..].find(CLOSE);
            let body_end = close_off.map(|o| body_start + o).unwrap_or(text.len());
            let body = text[body_start..body_end].trim();
            cursor = match close_off {
                Some(_) => body_end + CLOSE.len(),
                None => text.len(),
            };
            if let Ok(tc) = serde_json::from_str::<serde_json::Value>(body) {
                if let serde_json::Value::Object(map) = &tc {
                    if let Some(name) = map.get("name").and_then(|v| v.as_str()) {
                        // Gemma uses `args`; accept `arguments` as a
                        // Hermes-shape alias.
                        let args = map.get("args")
                            .or_else(|| map.get("arguments"))
                            .cloned()
                            .unwrap_or(serde_json::json!({}));
                        tool_calls.push(ParsedToolCall {
                            name: name.to_string(),
                            arguments: args,
                            repaired: false,
                        });
                    }
                }
            }
            if close_off.is_none() { break; }
        }
        let prose = match text.find(OPEN) {
            Some(idx) => trim_to_option(&text[..idx]),
            None => trim_to_option(text),
        };
        ToolCallParseResult { prose, tool_calls }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_eq(a: &serde_json::Value, b: serde_json::Value) -> bool { *a == b }

    // ── Hermes JSON parser ────────────────────────────────────────

    #[test]
    fn hermes_clean() {
        let p = HermesJsonParser::new();
        let text = "Sure, calling read.\n<tool_call>\n{\"name\":\"read\",\"arguments\":{\"path\":\"/etc/hosts\"}}\n</tool_call><|im_end|>";
        let r = p.parse(text);
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].name, "read");
        assert!(args_eq(&r.tool_calls[0].arguments, serde_json::json!({"path":"/etc/hosts"})));
        assert!(!r.tool_calls[0].repaired);
        assert_eq!(r.prose.as_deref(), Some("Sure, calling read."));
    }

    #[test]
    fn hermes_no_tool_call_returns_prose() {
        let p = HermesJsonParser::new();
        let r = p.parse("Hi there.");
        assert!(r.tool_calls.is_empty());
        assert_eq!(r.prose.as_deref(), Some("Hi there."));
    }

    #[test]
    fn hermes_flat_form_coerced() {
        let p = HermesJsonParser::new();
        let text = "<tool_call>{\"name\":\"write\",\"path\":\"/tmp/x\",\"contents\":\"hi\"}</tool_call>";
        let r = p.parse(text);
        assert_eq!(r.tool_calls.len(), 1);
        assert!(r.tool_calls[0].repaired);
        assert!(args_eq(&r.tool_calls[0].arguments, serde_json::json!({"path":"/tmp/x","contents":"hi"})));
    }

    #[test]
    fn hermes_stacked_opener_repair() {
        let p = HermesJsonParser::new();
        let text = "<tool_call><tool_call>{\"name\":\"read\",\"arguments\":{\"p\":1}}</tool_call>";
        let r = p.parse(text);
        assert_eq!(r.tool_calls.len(), 1);
        assert!(r.tool_calls[0].repaired);
    }

    #[test]
    fn hermes_xml_function_head() {
        let p = HermesJsonParser::new();
        let text = "<tool_call><function=cat>{\"path\":\"/x\"}</tool_call>";
        let r = p.parse(text);
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].name, "cat");
        assert!(r.tool_calls[0].repaired);
    }

    #[test]
    fn hermes_zero_arg_call() {
        let p = HermesJsonParser::new();
        let text = "<tool_call>{\"name\":\"ping\"}</tool_call>";
        let r = p.parse(text);
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].name, "ping");
        assert!(args_eq(&r.tool_calls[0].arguments, serde_json::json!({})));
    }

    #[test]
    fn hermes_multiple_calls() {
        let p = HermesJsonParser::new();
        let text = "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\n<tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>";
        let r = p.parse(text);
        assert_eq!(r.tool_calls.len(), 2);
        assert_eq!(r.tool_calls[0].name, "a");
        assert_eq!(r.tool_calls[1].name, "b");
    }

    // ── Qwen3.5/3.6 XML parser ────────────────────────────────────

    #[test]
    fn qwen35_xml_clean() {
        let p = Qwen35XmlParser::new();
        let text = "<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hosts\n</parameter>\n</function>\n</tool_call><|im_end|>";
        let r = p.parse(text);
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].name, "read_file");
        assert!(args_eq(&r.tool_calls[0].arguments, serde_json::json!({"path":"/etc/hosts"})));
        assert!(!r.tool_calls[0].repaired);
    }

    #[test]
    fn qwen35_xml_multi_param() {
        let p = Qwen35XmlParser::new();
        let text = "<tool_call><function=write><parameter=path>\n/tmp/x\n</parameter><parameter=contents>\nhello world\n</parameter></function></tool_call>";
        let r = p.parse(text);
        assert_eq!(r.tool_calls.len(), 1);
        assert!(args_eq(&r.tool_calls[0].arguments, serde_json::json!({
            "path":"/tmp/x", "contents":"hello world"
        })));
    }

    #[test]
    fn qwen35_xml_value_with_lt_char() {
        let p = Qwen35XmlParser::new();
        // VALUE contains < but parameter close is full `</parameter>`,
        // so the scanner should not be fooled.
        let text = "<tool_call><function=set><parameter=expr>\nx<5\n</parameter></function></tool_call>";
        let r = p.parse(text);
        assert_eq!(r.tool_calls.len(), 1);
        assert!(args_eq(&r.tool_calls[0].arguments, serde_json::json!({"expr":"x<5"})));
    }

    #[test]
    fn qwen35_xml_falls_back_to_hermes_on_json_body() {
        // MQ4 attractor: model trained on XML emits Hermes JSON.
        // Qwen35XmlParser recovers via internal Hermes fallback.
        let p = Qwen35XmlParser::new();
        let text = "<tool_call>{\"name\":\"read\",\"arguments\":{\"p\":\"/x\"}}</tool_call>";
        let r = p.parse(text);
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].name, "read");
        assert!(r.tool_calls[0].repaired);
    }

    // ── Gemma 4 native parser ─────────────────────────────────────

    #[test]
    fn gemma4_native_clean() {
        let p = Gemma4NativeParser::new();
        let text = "<|tool_call|>{\"name\":\"read\",\"args\":{\"path\":\"/x\"}}<|/tool_call|>";
        let r = p.parse(text);
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].name, "read");
        assert!(args_eq(&r.tool_calls[0].arguments, serde_json::json!({"path":"/x"})));
    }

    #[test]
    fn gemma4_native_accepts_arguments_alias() {
        let p = Gemma4NativeParser::new();
        let text = "<|tool_call|>{\"name\":\"read\",\"arguments\":{\"path\":\"/x\"}}<|/tool_call|>";
        let r = p.parse(text);
        assert_eq!(r.tool_calls.len(), 1);
        assert!(args_eq(&r.tool_calls[0].arguments, serde_json::json!({"path":"/x"})));
    }

    #[test]
    fn gemma4_native_no_match() {
        let p = Gemma4NativeParser::new();
        let r = p.parse("<tool_call>{\"name\":\"read\"}</tool_call>");
        assert!(r.tool_calls.is_empty());
        assert!(r.prose.is_some());
    }
}
