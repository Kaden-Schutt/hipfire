// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Qwen3.5 per-token spec-decode emission (`SpecEmit`).
//!
//! Relocated out of the daemon example: the emitter names qwen35-local
//! `grammar::{Matcher, ToolSchema}`, so it is a model-family definition and
//! belongs in this crate. The daemon obtains it arch-erased via
//! `Carrier::make_spec_emitter` → `Qwen35Emit::from_ctx`, which builds the
//! grammar `ToolSchema` list from the request's raw tool JSON internally (that
//! JSON→schema extraction also used to live in the daemon).
//!
//! Client-visible text and structured tool calls are owned by the runtime
//! [`ToolOutputRouter`]: EosFilter-emitted UTF-8 is routed incrementally;
//! protocol markers never enter `ClientEvent::Token`; complete calls are held
//! on [`FinishSummary`] until the daemon wrapper classifies a tool-safe terminal.

use crate::grammar;
use hipfire_runtime::emit_text::{
    currently_in_think, ThinkOutputRouter, ThinkRouteEvent, ToolOutputRouter, ToolRouteEvent,
};
use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction};
use hipfire_runtime::prompt_frame::AssistantPrefix;
use hipfire_runtime::spec::{
    ClientEvent, EmitOutcome, FinishSummary, SpecEmit, SpecEmitCtx, StopReason,
};
use hipfire_runtime::tokenizer::Tokenizer;

pub struct Qwen35Emit<'a> {
    tokenizer: &'a Tokenizer,
    filter: EosFilter,
    /// Index into the freshly-decoded byte stream past which bytes have not yet
    /// been fed to the filter (the daemon's old `bytes_fed_to_filter`).
    bytes_fed_to_filter: usize,
    /// Every committed token in order, for byte decoding + the cache-store the
    /// daemon does after the loop (exposed via [`Self::streamed_tokens`]).
    streamed_tokens: Vec<u32>,
    /// Incremental tool-protocol authority. Fed only EosFilter-emitted UTF-8.
    router: ToolOutputRouter,
    /// Incremental reasoning/content authority, upstream of tool routing.
    think_router: ThinkOutputRouter,
    /// Classifier-authorized visible prose accumulated this turn (cache fp).
    visible_acc: String,
    /// Latched when the router fails closed mid-stream or at finish.
    router_malformed: bool,
    /// Filter saw a decoded stop marker (`stop_at`) this turn.
    filter_stopped: bool,
    grammar_active: bool,
    grammar_matcher: grammar::Matcher,
    /// Set when a committed token violated the grammar; the daemon reads it via
    /// [`Self::grammar_violated`] to force the post-turn KV/recurrent reset.
    grammar_violated: bool,
    eos_token: u32,
    im_end_token: Option<u32>,
    stop: Vec<String>,
    max_think_tokens: usize,
    open_think_prefix: bool,
    think_count: usize,
    prev_in_think: bool,
    /// Think-budget force-close continuation drained by `take_forced` so the
    /// target advances over `</think>` and continues into visible output.
    forced: Vec<u32>,
    /// Prevent an endlessly re-opened reasoning span from being force-closed
    /// repeatedly. A second cap hit hard-stops through `ThinkCap`.
    think_force_closed: bool,
    /// `generated` counter at the point of the most recent `observe` — only used
    /// for the attractor-detect log message (byte-for-byte stderr parity).
    generated_hint: usize,
}

fn think_continuation_text() -> String {
    hipfire_config::developer_var("HIPFIRE_THINK_CONTINUATION")
        .unwrap_or_else(|_| "</think>\n\n".to_string())
}

/// EosFilter config for the Qwen DFlash/spec semantic-v2 producer. EosFilter
/// owns UTF-8/EOT filtering; ThinkOutputRouter owns think-channel routing.
fn qwen_dflash_eos_filter_config() -> EosFilterConfig {
    EosFilterConfig {
        strip_think: false,
        started_in_think: false,
        stop_at: vec![b"<|im_end|>".to_vec(), b"<|endoftext|>".to_vec()],
        holdback_prefixes: Vec::new(),
    }
}

impl<'a> Qwen35Emit<'a> {
    /// Build the qwen35 emitter from the model-independent [`SpecEmitCtx`],
    /// extracting the grammar `ToolSchema` list from the request's raw tool JSON
    /// (`ctx.tools`). Returns the arch-erased `Box<dyn SpecEmit>` the daemon
    /// drives. The JSON→schema extraction mirrors the daemon's old
    /// `tool_schemas_dflash` builder.
    pub fn from_ctx(ctx: SpecEmitCtx<'a>) -> Box<dyn SpecEmit + 'a> {
        let tool_protocol_enabled = ctx.tools.is_some();
        let tool_schemas: Vec<grammar::ToolSchema> = ctx
            .tools
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| {
                        let func = t.get("function").unwrap_or(t);
                        let name = func
                            .get("name")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())?
                            .to_string();
                        // Required-field list from JSON schema's
                        // `parameters.required`. Empty if the tool
                        // declares no required args.
                        let required: Vec<String> = func
                            .get("parameters")
                            .and_then(|p| p.get("required"))
                            .and_then(|r| r.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        Some(grammar::ToolSchema { name, required })
                    })
                    .collect()
            })
            .unwrap_or_default();
        let grammar_active = !tool_schemas.is_empty();
        let open_think_prefix = matches!(ctx.assistant_prefix, AssistantPrefix::OpenThink);
        Box::new(Self {
            tokenizer: ctx.tokenizer,
            filter: EosFilter::new(qwen_dflash_eos_filter_config()),
            bytes_fed_to_filter: 0,
            streamed_tokens: Vec::new(),
            router: if tool_protocol_enabled {
                ToolOutputRouter::new()
            } else {
                ToolOutputRouter::disabled()
            },
            think_router: ThinkOutputRouter::new(open_think_prefix),
            visible_acc: String::new(),
            router_malformed: false,
            filter_stopped: false,
            grammar_active,
            grammar_matcher: grammar::Matcher::with_config(
                tool_schemas,
                crate::grammar_config::resolve_qwen35_grammar_config(),
            ),
            grammar_violated: false,
            eos_token: ctx.eos,
            im_end_token: ctx.im_end,
            stop: ctx.stop,
            max_think_tokens: ctx.max_think,
            open_think_prefix,
            think_count: 0,
            prev_in_think: false,
            forced: Vec::new(),
            think_force_closed: false,
            generated_hint: 0,
        })
    }

    /// Route one content chunk. Visible prose becomes
    /// `ClientEvent::Token`; complete tool calls stay buffered in the router
    /// until [`SpecEmit::finish`]. Malformed protocol latches fail-closed.
    fn route_content_text(&mut self, text: &str, events: &mut Vec<ClientEvent>) {
        if text.is_empty() || self.router_malformed {
            return;
        }
        match self.router.push(text) {
            Ok(evs) => {
                for ev in evs {
                    match ev {
                        ToolRouteEvent::VisibleText(vt) => {
                            self.visible_acc.push_str(vt.as_str());
                            events.push(ClientEvent::Token(vt.into_string()));
                        }
                        ToolRouteEvent::ToolCall(_) => {
                            // Buffered in router; released only on tool-safe finish.
                        }
                    }
                }
            }
            Err(_) => {
                self.router_malformed = true;
            }
        }
    }

    /// Route classified reasoning/content events. Content continues through
    /// the tool protocol router; reasoning bypasses it and is never cached as
    /// visible prose.
    fn route_think_events(
        &mut self,
        channel_events: Vec<ThinkRouteEvent>,
        events: &mut Vec<ClientEvent>,
    ) {
        for channel_event in channel_events {
            if self.router_malformed {
                continue;
            }
            match channel_event {
                ThinkRouteEvent::Reasoning(text) => {
                    events.push(ClientEvent::Reasoning(text));
                }
                ThinkRouteEvent::Content(text) => self.route_content_text(&text, events),
            }
        }
    }

    /// Route one EosFilter-emitted UTF-8 chunk through reasoning classification.
    fn route_filter_text(&mut self, text: &str, events: &mut Vec<ClientEvent>) {
        if text.is_empty() {
            return;
        }
        let mut channel_events = Vec::new();
        self.think_router.push_into(text, &mut channel_events);
        self.route_think_events(channel_events, events);
    }

    /// Apply one filter observe step into the semantic router.
    fn apply_filter_action(&mut self, action: FilterAction, events: &mut Vec<ClientEvent>) {
        match action {
            FilterAction::Emit(text_bytes) => {
                let text = std::str::from_utf8(&text_bytes).unwrap_or("");
                self.route_filter_text(text, events);
            }
            FilterAction::EmitAndStop(text_bytes) => {
                let text = std::str::from_utf8(&text_bytes).unwrap_or("");
                self.route_filter_text(text, events);
                self.filter_stopped = true;
            }
            FilterAction::Stop => {
                self.filter_stopped = true;
            }
            FilterAction::Hold => {}
        }
    }

    /// Push a committed token, run it through the byte filter + router, and
    /// return the `Committed` + (optional) visible `Token` events — the shared
    /// tail of `begin`'s first-token emit and `observe`'s per-token emit.
    fn push_and_filter(&mut self, token: u32) -> Vec<ClientEvent> {
        let mut events = Vec::new();
        self.streamed_tokens.push(token);
        events.push(ClientEvent::Committed {
            id: token,
            idx: self.streamed_tokens.len() - 1,
        });
        let all_bytes = self.tokenizer.decode_bytes(&self.streamed_tokens);
        let new_bytes = &all_bytes[self.bytes_fed_to_filter..];
        self.bytes_fed_to_filter = all_bytes.len();
        let action = self.filter.observe(new_bytes);
        self.apply_filter_action(action, &mut events);
        events
    }

    /// Drain pending EOT-prefix prose, then finalize partial think markers.
    fn drain_pending_into_router(&mut self, events: &mut Vec<ClientEvent>) {
        let pending = self.filter.flush_pending();
        if !pending.is_empty() {
            let text = std::str::from_utf8(&pending).unwrap_or("");
            self.route_filter_text(text, events);
        }

        let mut channel_events = Vec::new();
        self.think_router.finish_into(&mut channel_events);
        self.route_think_events(channel_events, events);
    }

    /// Whether a streamed token id is a decoded EOT / terminator.
    fn token_is_eot(&self, token: u32) -> bool {
        token == self.eos_token
            || self.im_end_token == Some(token)
            || self.tokenizer.is_terminator(token)
    }

    /// User stop-sequence match against the decoded streamed suffix.
    /// Shared by `begin` (first token) and `observe` (later tokens).
    fn matches_stop_sequence(&self) -> bool {
        if self.stop.is_empty() {
            return false;
        }
        let decoded_suffix = self.tokenizer.decode(&self.streamed_tokens);
        self.stop
            .iter()
            .any(|s| decoded_suffix.ends_with(s.as_str()))
    }

    /// True when this turn ended on a decoded EOT (token id or filter stop_at).
    pub fn decoded_eot(&self) -> bool {
        self.filter_stopped
            || self
                .streamed_tokens
                .last()
                .copied()
                .map(|t| self.token_is_eot(t))
                .unwrap_or(false)
    }

    /// Producer-authorized visible text for asst-turn cache fingerprinting.
    pub fn visible_text(&self) -> &str {
        &self.visible_acc
    }
}

impl<'a> SpecEmit for Qwen35Emit<'a> {
    fn begin(&mut self, first_token: u32) -> EmitOutcome {
        // First-token emit (committed + filtered token), then seed the grammar
        // matcher with the first token's text. Mirrors generate_dflash 4452-4489.
        let events = self.push_and_filter(first_token);
        if self.grammar_active {
            let text = self.tokenizer.decode(&[first_token]);
            self.grammar_matcher.advance(&text);
        }
        // First-token EOS guard: if the prefill's first token is itself a
        // terminator we must NOT enter the spec loop (the inline `while
        // !first_token_is_eos`). Surface as a stop so the caller skips the loop.
        // Same precedence as `observe`: EOT token id, then filter stop_at, then
        // user stop-sequence — so generate_spec never enters spec.step.
        if self.token_is_eot(first_token) || self.filter_stopped {
            return EmitOutcome {
                events,
                generation_advanced: false,
                stop: Some(StopReason::Eos),
            };
        }
        if self.matches_stop_sequence() {
            return EmitOutcome {
                events,
                generation_advanced: false,
                stop: Some(StopReason::StopSequence),
            };
        }
        EmitOutcome {
            events,
            generation_advanced: true,
            stop: None,
        }
    }

    fn observe(&mut self, token: u32) -> EmitOutcome {
        // Grammar pre-check (POST-acceptance, before emit). A rejected token is
        // NOT emitted; treat as a grammar violation → stop. Mirrors 4565-4584.
        if self.grammar_active {
            let text = self.tokenizer.decode(&[token]);
            if !self.grammar_matcher.is_token_allowed(&text) {
                eprintln!(
                    "[grammar-dflash] rejected token id={} text={:?} (matcher.state={:?}) — forcing EOS | {}",
                    token, text, self.grammar_matcher.state(), self.grammar_matcher.debug_close_reject(),
                );
                self.grammar_violated = true;
                return EmitOutcome {
                    events: Vec::new(),
                    generation_advanced: false,
                    stop: Some(StopReason::GrammarViolation),
                };
            }
            let was_detected = self.grammar_matcher.attractor_detected();
            self.grammar_matcher.advance(&text);
            if !was_detected && self.grammar_matcher.attractor_detected() {
                eprintln!(
                    "[grammar-dflash-ngram] attractor detected in tool_call args at gen={} — forcing close",
                    self.generated_hint,
                );
            }
        }

        // Emit the token (committed + filtered + routed).
        let mut events = self.push_and_filter(token);

        // EOS check (token id). Mirrors 4608-4612.
        if self.token_is_eot(token) {
            return EmitOutcome {
                events,
                generation_advanced: false,
                stop: Some(StopReason::Eos),
            };
        }
        // Filter-decoded stop marker (byte form of im_end / endoftext).
        if self.filter_stopped {
            return EmitOutcome {
                events,
                generation_advanced: false,
                stop: Some(StopReason::Eos),
            };
        }

        // User stop-sequence match against the decoded suffix. Mirrors 4622-4628.
        if self.matches_stop_sequence() {
            return EmitOutcome {
                events,
                generation_advanced: false,
                stop: Some(StopReason::StopSequence),
            };
        }

        // max_think_tokens enforcement. Mirrors 4632-4664.
        if self.max_think_tokens > 0 {
            let raw_so_far = self.tokenizer.decode_bytes(&self.streamed_tokens);
            let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
            let in_think = currently_in_think(raw_str, self.open_think_prefix);
            if in_think && !self.prev_in_think {
                self.think_count = 0;
            }
            if in_think {
                self.think_count += 1;
            }
            self.prev_in_think = in_think;

            if in_think && self.think_count >= self.max_think_tokens {
                if !self.think_force_closed {
                    self.forced = self.tokenizer.encode(&think_continuation_text());
                    self.think_force_closed = true;
                    return EmitOutcome {
                        events,
                        generation_advanced: true,
                        stop: None,
                    };
                }
                // A pathological re-open after the injected close hard-stops.
                // Do not push raw "</think>\n" as a Token; surface stop only.
                return EmitOutcome {
                    events,
                    generation_advanced: true,
                    stop: Some(StopReason::ThinkCap),
                };
            }
        }

        EmitOutcome {
            events,
            generation_advanced: true,
            stop: None,
        }
    }

    fn finish(mut self: Box<Self>) -> FinishSummary {
        // Drain filter pending, then finalize the incremental router. Structured
        // ToolCalls are held on FinishSummary for the wrapper — generate_spec
        // must NOT render them before length/malformed/open-think is known.
        let mut events = Vec::new();
        self.drain_pending_into_router(&mut events);
        let open_think = self.think_router.in_think();
        let decoded_eot = self.decoded_eot();

        if open_think {
            // Unsafe terminal: drop pending (already flushed/dropped), no calls,
            // no safe stop. Daemon emits error-only (no done/cache).
            let _ = self.router.finish();
            return FinishSummary {
                events: Vec::new(),
                finish_reason: "open_think",
                tool_calls: 0,
                finalized: None,
                visible_text: String::new(),
                decoded_eot: false,
                open_think: true,
            };
        }

        if self.router_malformed || self.router.is_malformed() {
            let _ = self.router.finish();
            return FinishSummary {
                events,
                finish_reason: "malformed_protocol",
                tool_calls: 0,
                finalized: None,
                visible_text: std::mem::take(&mut self.visible_acc),
                decoded_eot,
                open_think: false,
            };
        }

        let buffered_before = self.router.tool_calls().to_vec();
        match self.router.finish() {
            Err(_) => FinishSummary {
                events,
                finish_reason: "malformed_protocol",
                tool_calls: 0,
                finalized: None,
                visible_text: std::mem::take(&mut self.visible_acc),
                decoded_eot,
                open_think: false,
            },
            Ok(evs) => {
                let mut finished_calls = buffered_before;
                for ev in evs {
                    match ev {
                        ToolRouteEvent::VisibleText(vt) => {
                            self.visible_acc.push_str(vt.as_str());
                            events.push(ClientEvent::Token(vt.into_string()));
                        }
                        ToolRouteEvent::ToolCall(tc) => {
                            finished_calls.push(tc);
                        }
                    }
                }
                let visible_text = std::mem::take(&mut self.visible_acc);
                if !finished_calls.is_empty() {
                    let n = finished_calls.len();
                    events.push(ClientEvent::ToolCalls(finished_calls));
                    FinishSummary {
                        events,
                        finish_reason: "tool_calls",
                        tool_calls: n,
                        finalized: None,
                        visible_text,
                        decoded_eot,
                        open_think: false,
                    }
                } else {
                    FinishSummary {
                        events,
                        finish_reason: "stop",
                        tool_calls: 0,
                        finalized: None,
                        visible_text,
                        decoded_eot,
                        open_think: false,
                    }
                }
            }
        }
    }

    /// The full committed-token stream (incl. the first token), for the daemon's
    /// post-loop asst-turn cache store.
    fn streamed_tokens(&self) -> &[u32] {
        &self.streamed_tokens
    }

    /// Whether a committed token tripped the grammar matcher (daemon forces a
    /// full KV/recurrent reset for the next turn when true).
    fn grammar_violated(&self) -> bool {
        self.grammar_violated
    }

    fn decoded_eot(&self) -> bool {
        // Prefer the inherent helper (token id + filter stop_at).
        Qwen35Emit::decoded_eot(self)
    }

    /// Hint the emitter of the daemon's current `generated` count so the
    /// attractor-detect log message reports the same number it did inline.
    fn set_generated_hint(&mut self, generated: usize) {
        self.generated_hint = generated;
    }

    fn take_forced(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.forced)
    }
}

// ── Deterministic non-GPU producer tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::prompt_frame::ToolCall;
    use hipfire_runtime::spec::SpecEmit;
    use hipfire_runtime::tokenizer::Tokenizer;

    fn json_escape(s: &str) -> String {
        let mut out = String::new();
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out
    }

    /// Mirror of production GPT-2 byte_to_unicode (prompt_frame test helper).
    fn byte_to_gpt2_char_test(b: u8) -> char {
        let mut bs: Vec<u32> = Vec::new();
        bs.extend((b'!' as u32)..=(b'~' as u32));
        bs.extend((0xA1u32)..=(0xACu32));
        bs.extend((0xAEu32)..=(0xFFu32));
        let mut cs: Vec<u32> = bs.clone();
        let mut n: u32 = 0;
        for byte in 0u32..=255u32 {
            if !bs.contains(&byte) {
                bs.push(byte);
                cs.push(256 + n);
                n += 1;
            }
        }
        for (bb, cc) in bs.into_iter().zip(cs.into_iter()) {
            if bb == b as u32 {
                return char::from_u32(cc).unwrap();
            }
        }
        char::from_u32(b as u32).unwrap()
    }

    /// Minimal BPE tokenizer covering ASCII + ChatML / think specials.
    fn test_tokenizer() -> Tokenizer {
        let mut entries: Vec<String> = Vec::new();
        entries.push(r#""<|im_start|>": 0"#.to_string());
        entries.push(r#""<|im_end|>": 1"#.to_string());
        entries.push(r#""<think>": 2"#.to_string());
        entries.push(r#""</think>": 3"#.to_string());
        entries.push(r#""system": 4"#.to_string());
        entries.push(r#""user": 5"#.to_string());
        entries.push(r#""assistant": 6"#.to_string());
        entries.push(r#""\n": 7"#.to_string());
        entries.push(r#""Ġ": 8"#.to_string());
        entries.push(r#""<|endoftext|>": 9"#.to_string());
        for b in 0u32..=255u32 {
            let ch = byte_to_gpt2_char_test(b as u8);
            let escaped = json_escape(&ch.to_string());
            entries.push(format!(r#""{}": {}"#, escaped, 100 + b));
        }
        let vocab_block = entries.join(", ");
        let json = format!(
            r#"{{
                "model": {{"type": "BPE", "vocab": {{ {vocab} }}, "merges": []}},
                "added_tokens": [
                    {{"id": 0, "content": "<|im_start|>", "special": true}},
                    {{"id": 1, "content": "<|im_end|>", "special": true}},
                    {{"id": 2, "content": "<think>", "special": true}},
                    {{"id": 3, "content": "</think>", "special": true}},
                    {{"id": 9, "content": "<|endoftext|>", "special": true}}
                ]
            }}"#,
            vocab = vocab_block,
        );
        Tokenizer::from_hf_json(&json).expect("test tokenizer")
    }

    fn make_emit<'a>(tok: &'a Tokenizer) -> Box<dyn SpecEmit + 'a> {
        Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: tok,
            eos: 9,
            im_end: Some(1),
            tools: Some(&[]),
            stop: Vec::new(),
            max_think: 0,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        })
    }

    /// Drive the production emitter with whole-string encodes (one observe per
    /// encoded token). Returns (stream events, finish, streamed_tokens).
    fn drive_text(text: &str) -> (Vec<ClientEvent>, FinishSummary, Vec<u32>) {
        let tok = test_tokenizer();
        let ids = tok.encode(text);
        assert!(!ids.is_empty(), "encode produced no tokens for {text:?}");
        let mut emit = make_emit(&tok);
        let mut stream = Vec::new();
        let mut first = true;
        for id in &ids {
            let outcome = if first {
                first = false;
                emit.begin(*id)
            } else {
                emit.observe(*id)
            };
            stream.extend(outcome.events);
            if outcome.stop.is_some() {
                break;
            }
        }
        let streamed = emit.streamed_tokens().to_vec();
        let finish = emit.finish();
        (stream, finish, streamed)
    }

    fn tokens_text(events: &[ClientEvent]) -> String {
        let mut s = String::new();
        for ev in events {
            if let ClientEvent::Token(t) = ev {
                s.push_str(t);
            }
        }
        s
    }

    fn reasoning_text(events: &[ClientEvent]) -> String {
        let mut s = String::new();
        for ev in events {
            if let ClientEvent::Reasoning(t) = ev {
                s.push_str(t);
            }
        }
        s
    }

    fn held_calls(finish: &FinishSummary) -> Vec<ToolCall> {
        finish
            .events
            .iter()
            .find_map(|ev| match ev {
                ClientEvent::ToolCalls(c) => Some(c.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    fn hermes_call(name: &str, args: &str) -> String {
        format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>")
    }

    #[test]
    fn prose_only_is_visible_stop() {
        let (stream, finish, raw) = drive_text("Hello world");
        assert!(!raw.is_empty());
        assert_eq!(tokens_text(&stream), "Hello world");
        assert_eq!(finish.finish_reason, "stop");
        assert_eq!(finish.tool_calls, 0);
        assert!(held_calls(&finish).is_empty());
        assert!(!tokens_text(&stream).contains("<tool_call>"));
    }

    #[test]
    fn valid_hermes_call_held_on_finish() {
        let body = format!("Sure.{}", hermes_call("get_weather", r#"{"city":"SF"}"#));
        let (stream, finish, raw) = drive_text(&body);
        assert!(!raw.is_empty());
        assert_eq!(tokens_text(&stream), "Sure.");
        assert!(!tokens_text(&stream).contains("<tool_call>"));
        assert!(!tokens_text(&stream).contains("get_weather"));
        assert_eq!(finish.finish_reason, "tool_calls");
        assert_eq!(finish.tool_calls, 1);
        let calls = held_calls(&finish);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn multiple_calls_and_surrounding_prose() {
        let body = format!(
            "A{}B{}",
            hermes_call("alpha", r#"{}"#),
            hermes_call("beta", r#"{"x":1}"#)
        );
        let (stream, finish, _) = drive_text(&body);
        assert_eq!(tokens_text(&stream), "AB");
        assert_eq!(finish.finish_reason, "tool_calls");
        assert_eq!(finish.tool_calls, 2);
        let names: Vec<_> = held_calls(&finish).into_iter().map(|c| c.name).collect();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn marker_split_across_tokens_never_leaks() {
        let body = hermes_call("t", r#"{}"#);
        let (stream, finish, _) = drive_text(&body);
        let visible = tokens_text(&stream);
        assert!(!visible.contains("<tool_call>"));
        assert!(!visible.contains("</tool_call>"));
        assert!(!visible.contains("\"name\""));
        assert_eq!(finish.finish_reason, "tool_calls");
        assert_eq!(finish.tool_calls, 1);
    }

    #[test]
    fn malformed_unclosed_span_is_fail_closed() {
        let (stream, finish, _) = drive_text("pre<tool_call>{\"name\":\"x\"");
        assert!(!tokens_text(&stream).contains("<tool_call>"));
        assert_eq!(finish.finish_reason, "malformed_protocol");
        assert_eq!(finish.tool_calls, 0);
        assert!(held_calls(&finish).is_empty());
    }

    #[test]
    fn malformed_empty_body_is_fail_closed() {
        let (stream, finish, _) = drive_text("x<tool_call></tool_call>");
        assert!(!tokens_text(&stream).contains("<tool_call>"));
        assert_eq!(finish.finish_reason, "malformed_protocol");
        assert_eq!(finish.tool_calls, 0);
    }

    #[test]
    fn think_bytes_suppressed_from_token_channel() {
        let (stream, finish, raw) = drive_text("<think>secret</think>answer");
        assert!(!raw.is_empty());
        let visible = tokens_text(&stream);
        assert!(!visible.contains("<think>"));
        assert!(!visible.contains("secret"));
        assert!(visible.contains("answer"));
        assert_eq!(reasoning_text(&stream), "secret");
        assert_eq!(finish.finish_reason, "stop");
    }

    #[test]
    fn im_end_marker_suppressed_and_decoded_eot() {
        let tok = test_tokenizer();
        let mut ids = tok.encode("hi");
        ids.push(1); // <|im_end|>
        let mut emit = make_emit(&tok);
        let mut stream = Vec::new();
        let mut first = true;
        let mut stopped = false;
        for id in &ids {
            let outcome = if first {
                first = false;
                emit.begin(*id)
            } else {
                emit.observe(*id)
            };
            stream.extend(outcome.events);
            if outcome.stop == Some(StopReason::Eos) {
                stopped = true;
                break;
            }
        }
        assert!(stopped, "im_end must stop via Eos");
        let visible = tokens_text(&stream);
        assert!(!visible.contains("<|im_end|>"));
        assert!(visible.contains("hi"));
        let finish = emit.finish();
        assert_eq!(finish.finish_reason, "stop");
        assert_eq!(finish.tool_calls, 0);
    }

    #[test]
    fn raw_committed_once_per_token_including_markers() {
        let body = format!("x{}", hermes_call("t", r#"{}"#));
        let (stream, _, raw) = drive_text(&body);
        let committed: Vec<_> = stream
            .iter()
            .filter_map(|ev| match ev {
                ClientEvent::Committed { id, idx } => Some((*id, *idx)),
                _ => None,
            })
            .collect();
        assert_eq!(committed.len(), raw.len());
        for (i, (id, idx)) in committed.iter().enumerate() {
            assert_eq!(*idx, i);
            assert_eq!(*id, raw[i]);
        }
    }

    #[test]
    fn finish_does_not_use_whole_output_extraction_authority() {
        let open = "<tool_call>{\"name\": \"recovered\", \"arguments\": {}}";
        let (_stream, finish, _) = drive_text(open);
        assert_eq!(finish.finish_reason, "malformed_protocol");
        assert_eq!(finish.tool_calls, 0);
        assert!(held_calls(&finish).is_empty());
    }

    #[test]
    fn begin_first_token_stop_sequence_terminates_before_observe() {
        // A stop string completed by the first generated token must surface
        // StopSequence from begin so generate_spec never enters spec.step.
        let tok = test_tokenizer();
        let ids = tok.encode("STOP");
        assert!(!ids.is_empty());
        let first = ids[0];
        let first_text = tok.decode(&[first]);
        let mut emit = Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tok,
            eos: 9,
            im_end: Some(1),
            tools: None,
            stop: vec![first_text.clone()],
            max_think: 0,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });
        let outcome = emit.begin(first);
        assert_eq!(
            outcome.stop,
            Some(StopReason::StopSequence),
            "first token completing stop {first_text:?} must stop in begin"
        );
        assert!(
            outcome
                .events
                .iter()
                .any(|e| matches!(e, ClientEvent::Committed { id, .. } if *id == first)),
            "stop still commits the raw first token"
        );
        // Later-token path unchanged: multi-token stop still matches on observe.
        let multi = tok.encode("abSTOP");
        assert!(multi.len() > 1, "need multi-token path for observe parity");
        let mut emit2 = Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tok,
            eos: 9,
            im_end: Some(1),
            tools: None,
            stop: vec!["STOP".to_string()],
            max_think: 0,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });
        let mut saw_stop = false;
        let mut first = true;
        let mut n = 0usize;
        for id in &multi {
            let outcome = if first {
                first = false;
                emit2.begin(*id)
            } else {
                emit2.observe(*id)
            };
            n += 1;
            if outcome.stop == Some(StopReason::StopSequence) {
                saw_stop = true;
                break;
            }
            assert!(
                outcome.stop.is_none(),
                "unexpected early stop: {:?}",
                outcome.stop
            );
        }
        assert!(saw_stop, "multi-token stop must still fire via observe");
        assert!(n > 1, "multi-token stop must not only hit begin");
    }

    #[test]
    fn filter_config_matches_ar_semantic_v2() {
        let cfg = qwen_dflash_eos_filter_config();
        assert!(!cfg.strip_think);
        assert!(!cfg.started_in_think);
        assert!(cfg.stop_at.iter().any(|s| s == b"<|im_end|>"));
        assert!(cfg.stop_at.iter().any(|s| s == b"<|endoftext|>"));
    }
}
