// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cohere2-MoE / North-Mini-Code per-token spec-decode emission (`SpecEmit`).
//!
//! Ports the agentic-marker state machine from the daemon's bespoke
//! `generate_cohere2moe` AR loop into the arch-generic `SpecEmit` seam so the
//! model-free n-gram spec loop can drive North without leaking markers. North
//! emits six structural markers (`<|START/END_THINKING|>`, `<|START/END_TEXT|>`,
//! `<|START/END_ACTION|>`): the markers themselves are never surfaced; thinking
//! content routes to the reasoning channel, text content to the visible answer,
//! and an `<|START_ACTION|>…<|END_ACTION|>` body parses into `tool_calls`.
//!
//! Two generation-side recoveries that a pure emitter cannot express on its own
//! ride the [`SpecEmit::take_forced`] hook (the loop force-injects the returned
//! tokens, advancing the target over them):
//! - **empty-turn guard**: when the model ends a turn (EOS) with no visible
//!   output, suppress the EOS and force `<|START_TEXT|>` (closing thinking first
//!   if still open) so it answers instead of returning reasoning-only.
//! - **think-budget force-close**: when thinking out-runs its token reserve,
//!   force `<|END_THINKING|><|START_TEXT|>` so the answer fits before the cap.
//! Both are gated by `HIPFIRE_C2M_EMPTY_TURN_GUARD` (default on) and bounded by
//! `MAX_EOS_SUPPRESS` / a one-shot `think_force_closed`, so forcing terminates.

use hipfire_runtime::prompt_frame::ToolCall;
use hipfire_runtime::spec::{
    ClientEvent, EmitOutcome, FinishSummary, SpecEmit, SpecEmitCtx, StopReason,
};
use hipfire_runtime::tokenizer::Tokenizer;

const MAX_EOS_SUPPRESS: usize = 3;

#[derive(PartialEq, Clone, Copy)]
enum Sec {
    Pre,
    Think,
    Text,
    Action,
}

pub struct Cohere2MoeEmit<'a> {
    tokenizer: &'a Tokenizer,
    eos: u32,
    // Structural marker ids (resolved from the tokenizer, with the North fixed
    // ids as fallback — mirrors the AR loop's `mark(name, fb)`).
    mk_think0: u32,
    mk_think1: u32,
    mk_text0: u32,
    mk_text1: u32,
    mk_act0: u32,
    mk_act1: u32,
    sec: Sec,
    action_buf: String,
    vis_buf: String,
    emitted_count: usize,
    emitted_visible: bool,
    eos_suppressions: usize,
    empty_turn_guard: bool,
    think_count: usize,
    think_budget: usize,
    think_force_closed: bool,
    known_tools: Vec<String>,
    tool_params: Vec<(String, Vec<String>)>,
    tool_calls_buf: Vec<ToolCall>,
    tool_calls_emitted: bool,
    /// Tokens the loop must force after the current step (drained by `take_forced`).
    forced: Vec<u32>,
}

impl<'a> Cohere2MoeEmit<'a> {
    pub fn from_ctx(ctx: SpecEmitCtx<'a>) -> Box<dyn SpecEmit + 'a> {
        let tk = ctx.tokenizer;
        let mark = |s: &str, fb: u32| -> u32 { tk.special_token_id(s).unwrap_or(fb) };
        let empty_turn_guard = hipfire_config::developer_var("HIPFIRE_C2M_EMPTY_TURN_GUARD")
            .ok()
            .as_deref()
            != Some("0");
        // Think-budget: reserve room for an answer; honor an explicit max_think.
        // Byte-mirrors the AR loop's `think_reserve` / `think_budget`.
        let max_tokens = ctx.max_tokens;
        let think_reserve = (max_tokens / 4).clamp(64, 512).min(max_tokens / 2);
        let think_budget = if ctx.max_think > 1 {
            ctx.max_think.min(max_tokens.saturating_sub(think_reserve))
        } else {
            max_tokens.saturating_sub(think_reserve)
        };
        let (known_tools, tool_params) = extract_tools(ctx.tools);
        Box::new(Self {
            tokenizer: tk,
            eos: ctx.eos,
            mk_think0: mark("<|START_THINKING|>", 255010),
            mk_think1: mark("<|END_THINKING|>", 255011),
            mk_text0: mark("<|START_TEXT|>", 255012),
            mk_text1: mark("<|END_TEXT|>", 255013),
            mk_act0: mark("<|START_ACTION|>", 255014),
            mk_act1: mark("<|END_ACTION|>", 255015),
            sec: Sec::Pre,
            action_buf: String::new(),
            vis_buf: String::new(),
            emitted_count: 0,
            emitted_visible: false,
            eos_suppressions: 0,
            empty_turn_guard,
            think_count: 0,
            think_budget,
            think_force_closed: false,
            known_tools,
            tool_params,
            tool_calls_buf: Vec::new(),
            tool_calls_emitted: false,
            forced: Vec::new(),
        })
    }

    /// Append a `Committed` event for `token` (token-id tracking; gated to the
    /// wire by `HIPFIRE_EMIT_TOKEN_IDS` at render). Every surfaced token gets one
    /// so the loop's `generated` count matches the AR loop (which counts every
    /// sampled token); the only token that does NOT is a masked EOS.
    fn committed(&mut self, token: u32, events: &mut Vec<ClientEvent>) {
        events.push(ClientEvent::Committed {
            id: token,
            idx: self.emitted_count,
        });
        self.emitted_count += 1;
    }

    /// The shared begin/observe body: run one committed token through the marker
    /// state machine, returning the events + any stop/forced request.
    fn process(&mut self, token: u32) -> EmitOutcome {
        // ── EOS, with the empty-turn guard ──
        if token == self.eos {
            if self.empty_turn_guard
                && !self.emitted_visible
                && self.eos_suppressions < MAX_EOS_SUPPRESS
            {
                // Reasoning-only turn ending with nothing visible: suppress the
                // EOS and force a `<|START_TEXT|>` continuation (closing thinking
                // first if still open). The masked EOS is NOT surfaced/counted.
                self.eos_suppressions += 1;
                if self.sec == Sec::Think {
                    self.forced.push(self.mk_think1);
                }
                self.forced.push(self.mk_text0);
                return EmitOutcome::held();
            }
            return EmitOutcome {
                events: Vec::new(),
                stop: Some(StopReason::Eos),
            };
        }

        let mut events = Vec::new();
        // ── Marker-id transitions (markers are never surfaced) ──
        if token == self.mk_think0 {
            self.sec = Sec::Think;
            self.committed(token, &mut events);
        } else if token == self.mk_text0 {
            self.sec = Sec::Text;
            self.committed(token, &mut events);
        } else if token == self.mk_act0 {
            self.sec = Sec::Action;
            self.action_buf.clear();
            self.committed(token, &mut events);
        } else if token == self.mk_think1 || token == self.mk_text1 {
            self.sec = Sec::Pre;
            self.committed(token, &mut events);
        } else if token == self.mk_act1 {
            // End of an action block → parse + snap into tool_calls.
            let mut calls = parse_cohere_action(&self.action_buf);
            snap_call_names(&mut calls, &self.known_tools, &self.tool_params);
            let converted = to_tool_calls(&calls);
            if !converted.is_empty() {
                self.tool_calls_buf.extend(converted.iter().cloned());
                events.push(ClientEvent::ToolCalls(converted));
                self.emitted_visible = true;
                self.tool_calls_emitted = true;
            }
            self.sec = Sec::Pre;
            self.committed(token, &mut events);
        } else {
            let frag = self.tokenizer.decode(&[token]);
            // Defense-in-depth: never surface a Cohere structural marker the id
            // state machine missed (START_OF_TURN_TOKEN, CHATBOT_TOKEN, …). The
            // token is still committed (target advanced over it); only its emit
            // is dropped, so a state-machine miss can never leak a marker.
            let is_marker = frag.len() > 4
                && frag.starts_with("<|")
                && frag.ends_with("|>")
                && frag[2..frag.len() - 2]
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c == '_');
            if !is_marker {
                match self.sec {
                    Sec::Action => self.action_buf.push_str(&frag),
                    Sec::Think => {
                        events.push(ClientEvent::Reasoning(frag));
                        self.think_count += 1;
                    }
                    Sec::Text | Sec::Pre => {
                        self.vis_buf.push_str(&frag);
                        events.push(ClientEvent::Token(frag));
                        self.emitted_visible = true;
                    }
                }
            }
            self.committed(token, &mut events);
        }

        // ── Think-budget force-close (mechanism #2) ──
        if self.empty_turn_guard
            && !self.think_force_closed
            && !self.emitted_visible
            && self.sec == Sec::Think
            && self.think_count >= self.think_budget
        {
            self.forced.push(self.mk_think1);
            self.forced.push(self.mk_text0);
            self.think_force_closed = true;
        }

        EmitOutcome { events, stop: None }
    }
}

impl<'a> SpecEmit for Cohere2MoeEmit<'a> {
    fn begin(&mut self, first_token: u32) -> EmitOutcome {
        self.process(first_token)
    }

    fn observe(&mut self, token: u32) -> EmitOutcome {
        self.process(token)
    }

    fn take_forced(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.forced)
    }

    fn finish(mut self: Box<Self>) -> FinishSummary {
        let mut events = Vec::new();
        // Tool-call-as-text recovery: a non-Cohere harness can prime North to
        // write a tool-call JSON array as TEXT instead of via <|START_ACTION|>.
        if !self.tool_calls_emitted {
            let mut recovered = parse_cohere_action(&self.vis_buf);
            snap_call_names(&mut recovered, &self.known_tools, &self.tool_params);
            let converted = to_tool_calls(&recovered);
            if !converted.is_empty() {
                self.tool_calls_buf.extend(converted.iter().cloned());
                events.push(ClientEvent::ToolCalls(converted));
            }
        }
        let tool_calls = self.tool_calls_buf.len();
        let finish_reason = if tool_calls > 0 { "tool_calls" } else { "stop" };
        FinishSummary {
            events,
            finish_reason,
            tool_calls,
            visible_text: String::new(),
            decoded_eot: false,
            open_think: false,
        }
    }
}

/// Convert `parse_cohere_action`'s `{"name","arguments"}` JSON values to `ToolCall`.
fn to_tool_calls(calls: &[serde_json::Value]) -> Vec<ToolCall> {
    calls
        .iter()
        .filter_map(|c| {
            let name = c.get("name").and_then(|v| v.as_str())?.to_string();
            let arguments = c
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            Some(ToolCall { name, arguments })
        })
        .collect()
}

/// Extract `(known_tool_names, [(tool_name, param_names)])` from the request's
/// raw tool JSON (OpenAI `{function:{name,parameters}}` or flat `{name}`).
fn extract_tools(tools: Option<&[serde_json::Value]>) -> (Vec<String>, Vec<(String, Vec<String>)>) {
    let known: Vec<String> = tools
        .map(|ts| {
            ts.iter()
                .filter_map(|t| {
                    t.get("function")
                        .and_then(|f| f.get("name"))
                        .or_else(|| t.get("name"))
                        .and_then(|n| n.as_str())
                        .map(String::from)
                })
                .collect()
        })
        .unwrap_or_default();
    let params: Vec<(String, Vec<String>)> = tools
        .map(|ts| {
            ts.iter()
                .filter_map(|t| {
                    let f = t.get("function").unwrap_or(t);
                    let name = f.get("name").and_then(|n| n.as_str())?.to_string();
                    let p = f
                        .get("parameters")
                        .and_then(|p| p.get("properties"))
                        .and_then(|p| p.as_object())
                        .map(|o| o.keys().cloned().collect())
                        .unwrap_or_default();
                    Some((name, p))
                })
                .collect()
        })
        .unwrap_or_default();
    (known, params)
}

// ─── Tool-call parsing/snapping helpers (shared with the daemon AR path) ─────

/// Parse a Cohere `<|START_ACTION|>` … `<|END_ACTION|>` body — a JSON array of
/// `{tool_name, parameters}` — into `[{name, arguments}]`. The exact JSON the
/// model emits as the action body (or as TEXT when a non-Cohere harness primes
/// it with a generic tool-call format).
pub fn parse_cohere_action(buf: &str) -> Vec<serde_json::Value> {
    let t = buf.trim();
    let slice = match (t.find('['), t.rfind(']')) {
        (Some(s), Some(e)) if e > s => &t[s..=e],
        _ => return Vec::new(),
    };
    let parsed: serde_json::Value = match serde_json::from_str(slice) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let arr = match parsed.as_array() {
        Some(a) => a,
        None => return Vec::new(),
    };
    arr.iter()
        .filter_map(|tc| {
            let name = tc.get("tool_name").and_then(|v| v.as_str())?;
            let args = tc
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            Some(serde_json::json!({"name": name, "arguments": args}))
        })
        .collect()
}

/// Snap a hallucinated/verbose tool name back to a real tool from the request
/// (e.g. `bash immediate return command` → `bash`): exact, then leading-token,
/// then any-token, preferring the longest known match; pass through unchanged.
pub fn snap_tool_name(name: &str, known: &[String]) -> String {
    if known.is_empty() || known.iter().any(|k| k == name) {
        return name.to_string();
    }
    let toks: Vec<&str> = name.split_whitespace().collect();
    let mut best: Option<&str> = None;
    for k in known {
        let hit = toks.first() == Some(&k.as_str()) || toks.iter().any(|t| *t == k.as_str());
        if hit && best.map_or(true, |b| k.len() > b.len()) {
            best = Some(k.as_str());
        }
    }
    best.map(String::from).unwrap_or_else(|| name.to_string())
}

/// Snap a glitched argument key (e.g. `path_`, `path_l`) to a real parameter:
/// exact, then prefix either way, preferring the longest valid parameter.
pub fn snap_param_name(key: &str, valid: &[String]) -> String {
    if valid.is_empty() || valid.iter().any(|v| v == key) {
        return key.to_string();
    }
    let mut best: Option<&str> = None;
    for v in valid {
        if (key.starts_with(v.as_str()) || v.starts_with(key))
            && best.map_or(true, |b| v.len() > b.len())
        {
            best = Some(v.as_str());
        }
    }
    best.map(String::from).unwrap_or_else(|| key.to_string())
}

/// Normalize each parsed tool_call against the request's tool schemas: snap the
/// `name` to a known tool, then snap each argument key to a real parameter.
pub fn snap_call_names(
    calls: &mut [serde_json::Value],
    known: &[String],
    tool_params: &[(String, Vec<String>)],
) {
    for c in calls.iter_mut() {
        let name = match c.get("name").and_then(|v| v.as_str()) {
            Some(n) => snap_tool_name(n, known),
            None => continue,
        };
        c["name"] = serde_json::Value::String(name.clone());
        if let Some((_, valid)) = tool_params.iter().find(|(tn, _)| *tn == name) {
            if !valid.is_empty() {
                if let Some(args) = c.get_mut("arguments").and_then(|a| a.as_object_mut()) {
                    for k in args.keys().cloned().collect::<Vec<_>>() {
                        let sk = snap_param_name(&k, valid);
                        if sk != k && !args.contains_key(&sk) {
                            if let Some(v) = args.remove(&k) {
                                args.insert(sk, v);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_cohere_action, snap_call_names, snap_param_name, snap_tool_name};

    #[test]
    fn snaps_glitched_param_key() {
        let valid = ["path".to_string()];
        assert_eq!(snap_param_name("path_", &valid), "path");
        assert_eq!(snap_param_name("path_l", &valid), "path");
        assert_eq!(snap_param_name("path", &valid), "path");
        assert_eq!(snap_param_name("command", &valid), "command");
    }

    #[test]
    fn snaps_verbose_hallucinated_name() {
        let known = ["bash".to_string(), "read".to_string(), "write".to_string()];
        assert_eq!(
            snap_tool_name("bash immediate return command", &known),
            "bash"
        );
        assert_eq!(snap_tool_name("read", &known), "read");
        assert_eq!(snap_tool_name("please write the file", &known), "write");
        assert_eq!(snap_tool_name("frobnicate", &known), "frobnicate");
        let k2 = ["bash".to_string(), "bash_script".to_string()];
        assert_eq!(snap_tool_name("bash_script now", &k2), "bash_script");
    }

    #[test]
    fn parses_action_body() {
        let body = r#"[{"tool_name": "bash", "parameters": {"command": "ls"}}]"#;
        let calls = parse_cohere_action(body);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["name"], "bash");
        assert_eq!(calls[0]["arguments"]["command"], "ls");
        assert!(parse_cohere_action("just prose, no array").is_empty());
    }

    #[test]
    fn recovers_tool_call_written_as_text() {
        let text = r#"[
    {"tool_call_id": "12", "tool_name": "read", "parameters": {"path": "/home/nick/CLionProjects/tb/.idea/.gitignore"}}
]"#;
        let mut calls = parse_cohere_action(text);
        assert_eq!(calls.len(), 1);
        snap_call_names(
            &mut calls,
            &["read".to_string()],
            &[("read".to_string(), vec!["path".to_string()])],
        );
        assert_eq!(calls[0]["name"], "read");
        assert_eq!(
            calls[0]["arguments"]["path"],
            "/home/nick/CLionProjects/tb/.idea/.gitignore"
        );
        assert!(parse_cohere_action("I'll read the file now.").is_empty());
        assert!(parse_cohere_action("The result is [1, 2, 3].").is_empty());
    }

    #[test]
    fn snaps_name_inside_recovered_call() {
        let text =
            r#"[{"tool_name": "bash immediate return command", "parameters": {"command": "ls"}}]"#;
        let mut calls = parse_cohere_action(text);
        snap_call_names(
            &mut calls,
            &["bash".to_string()],
            &[("bash".to_string(), vec!["command".to_string()])],
        );
        assert_eq!(calls[0]["name"], "bash");
    }
}
