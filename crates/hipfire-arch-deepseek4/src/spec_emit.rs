// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! DeepSeek V4 per-token spec-decode emission (`SpecEmit`).
//!
//! Relocated out of the daemon example: the emitter names ds4-local
//! `dsml::StreamParser` and `mtp_speculator::Deepseek4SpecGrammar`, so it is a
//! model-family definition and belongs in this crate. The daemon obtains it
//! arch-erased via `Carrier::make_spec_emitter` → [`Deepseek4Emit::from_ctx`],
//! which builds the in-step tool-call grammar from the request's raw tool JSON
//! plus the pre-decoded vocab the daemon supplies on the neutral `SpecEmitCtx`.

use crate::dsml::{self, StreamEvent};
use crate::grammar::ToolSchema;
use crate::mtp_speculator::Deepseek4SpecGrammar;
use hipfire_runtime::prompt_frame::{ThinkMode, ToolCall};
use hipfire_runtime::spec::{
    ClientEvent, EmitOutcome, FinishSummary, SpecEmit, SpecEmitCtx, SpecGrammar, StopReason,
};
use hipfire_runtime::tokenizer::Tokenizer;

pub struct Deepseek4Emit<'a> {
    tokenizer: &'a Tokenizer,
    parser: dsml::StreamParser,
    /// Parsed tool calls accumulated across the turn (the old `emit_tool_calls_buf`).
    tool_calls_buf: Vec<ToolCall>,
    eos_token: u32,
    /// Count of tokens actually emitted (committed) so far, kept in lockstep with
    /// the daemon's `generated_count` so each `Committed { idx }` equals the
    /// inline `emit_committed_event`'s `pos` argument.
    emitted_count: usize,
    /// In-step tool-call grammar, threaded into the fused spec step via
    /// `grammar()`. `None` ⇒ no tools (or the bespoke loop, which owns its own
    /// matcher and never calls `grammar()`). The matcher advances inside the spec
    /// step ONLY — `observe` must NOT touch it (single-advance invariant).
    grammar: Option<Deepseek4SpecGrammar>,
}

/// Build the in-step tool-call grammar from the request's raw tool JSON and the
/// daemon-supplied pre-decoded vocab. Returns `None` when no usable tool schema
/// is present (or no vocab was supplied). Mirrors the daemon's old
/// `build_deepseek4_spec_grammar`, minus the lazy `m.decoded_vocab` cache (the
/// daemon now populates that Arc before building the neutral `SpecEmitCtx`).
fn build_grammar(
    tools: Option<&[serde_json::Value]>,
    decoded_vocab: Option<std::sync::Arc<Vec<String>>>,
) -> Option<Deepseek4SpecGrammar> {
    let tool_schemas: Vec<ToolSchema> = tools
        .map(|arr| {
            arr.iter()
                .map(|t| {
                    let func = t.get("function").unwrap_or(t);
                    let name = func
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let parameters = func.get("parameters");
                    let params: Vec<String> = parameters
                        .and_then(|p| p.get("properties"))
                        .and_then(|p| p.as_object())
                        .map(|m| m.keys().cloned().collect())
                        .unwrap_or_default();
                    let required: Vec<String> = parameters
                        .and_then(|p| p.get("required"))
                        .and_then(|r| r.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    ToolSchema {
                        name,
                        params,
                        required,
                    }
                })
                .filter(|s: &ToolSchema| !s.name.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if tool_schemas.is_empty() {
        return None;
    }
    let decoded_vocab = decoded_vocab?;
    Some(Deepseek4SpecGrammar::new(tool_schemas, decoded_vocab))
}

impl<'a> Deepseek4Emit<'a> {
    /// Build the ds4 emitter from the model-independent [`SpecEmitCtx`]. The
    /// think-mode picks the DSML parser's initial state; the in-step grammar is
    /// built from `ctx.tools` + `ctx.decoded_vocab`.
    pub fn from_ctx(ctx: SpecEmitCtx<'a>) -> Box<dyn SpecEmit + 'a> {
        let parser = match ctx.think_mode {
            ThinkMode::High | ThinkMode::Max => dsml::StreamParser::new_in_think(),
            ThinkMode::NonThink => dsml::StreamParser::new(),
        };
        let grammar = build_grammar(ctx.tools, ctx.decoded_vocab);
        Box::new(Self {
            tokenizer: ctx.tokenizer,
            parser,
            tool_calls_buf: Vec::new(),
            eos_token: ctx.eos,
            emitted_count: 0,
            grammar,
        })
    }

    /// Feed one committed token's decoded text through the DSML parser, mapping
    /// each `StreamEvent` to its `ClientEvent` (and absorbing tool calls), then
    /// append the `Committed` event LAST — matching the inline `parser.feed` →
    /// `emit_stream_event` → `emit_committed_event` ordering. Shared by `begin`'s
    /// first-token emit and `observe`'s per-token emit.
    fn feed_and_emit(&mut self, token: u32) -> Vec<ClientEvent> {
        let mut events = Vec::new();
        let frag = self.tokenizer.decode(&[token]);
        for ev in self.parser.feed(&frag) {
            match ev {
                StreamEvent::Token(text) => events.push(ClientEvent::Token(text)),
                StreamEvent::Reasoning(text) => events.push(ClientEvent::Reasoning(text)),
                StreamEvent::ToolCalls(calls) => {
                    let converted: Vec<ToolCall> = calls
                        .iter()
                        .map(|c| ToolCall {
                            name: c.name.clone(),
                            arguments: c.arguments.clone(),
                        })
                        .collect();
                    self.tool_calls_buf.extend(converted.iter().cloned());
                    events.push(ClientEvent::ToolCalls(converted));
                }
            }
        }
        events.push(ClientEvent::Committed {
            id: token,
            idx: self.emitted_count,
        });
        self.emitted_count += 1;
        events
    }
}

impl<'a> SpecEmit for Deepseek4Emit<'a> {
    /// In-step grammar: hand the fused spec step the erased ds4 grammar handle so
    /// it masks draft+verify logits and advances the matcher. `None` ⇒ no tools.
    /// Because the matcher advances HERE (in-step), `observe` must NOT re-advance
    /// it — and ds4's `observe` only feeds the DSML parser, so the invariant holds.
    fn grammar(&mut self) -> Option<&mut dyn SpecGrammar> {
        self.grammar.as_mut().map(|g| g as &mut dyn SpecGrammar)
    }

    fn begin(&mut self, first_token: u32) -> EmitOutcome {
        // First generated token (the prefill argmax). Mirrors generate_deepseek4
        // 9537-9553: EOS-first yields an empty turn — the inline `if
        // spec_last_token != eos_tok` guard dropped it (no feed, no committed).
        if first_token == self.eos_token {
            return EmitOutcome {
                events: Vec::new(),
                stop: Some(StopReason::Eos),
            };
        }
        EmitOutcome {
            events: self.feed_and_emit(first_token),
            stop: None,
        }
    }

    fn observe(&mut self, token: u32) -> EmitOutcome {
        // Per-accepted-token. Mirrors generate_deepseek4 9597-9622: an accepted
        // token equal to `eos_tok` breaks the loop BEFORE emit — no feed, no
        // committed event. The `generated_count >= max_tokens` guard stays in the
        // decode loop (loop state, not emit policy).
        if token == self.eos_token {
            return EmitOutcome {
                events: Vec::new(),
                stop: Some(StopReason::Eos),
            };
        }
        EmitOutcome {
            events: self.feed_and_emit(token),
            stop: None,
        }
    }

    fn finish(mut self: Box<Self>) -> FinishSummary {
        // Post-loop flush. Mirrors generate_deepseek4 9632-9638: `parser.finish()`
        // → absorb + emit, then `tool_calls_parsed_count = emit_tool_calls_buf.len()`.
        // The ds4 `done` envelope drives `finish_reason` off `tool_calls_parsed_count`
        // (length-cap override is the caller's call).
        let mut events = Vec::new();
        // `finish` consumes the parser by value; move it out of `self`.
        let parser = std::mem::replace(&mut self.parser, dsml::StreamParser::new());
        for ev in parser.finish() {
            match ev {
                StreamEvent::Token(text) => events.push(ClientEvent::Token(text)),
                StreamEvent::Reasoning(text) => events.push(ClientEvent::Reasoning(text)),
                StreamEvent::ToolCalls(calls) => {
                    let converted: Vec<ToolCall> = calls
                        .iter()
                        .map(|c| ToolCall {
                            name: c.name.clone(),
                            arguments: c.arguments.clone(),
                        })
                        .collect();
                    self.tool_calls_buf.extend(converted.iter().cloned());
                    events.push(ClientEvent::ToolCalls(converted));
                }
            }
        }
        let tool_calls = self.tool_calls_buf.len();
        let finish_reason = if tool_calls > 0 { "tool_calls" } else { "stop" };
        FinishSummary {
            events,
            finish_reason,
            tool_calls,
        }
    }
}
