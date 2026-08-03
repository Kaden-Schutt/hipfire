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

use crate::dsml::{self, DsmlDeferredCalls, DsmlDeferredOutcome, StreamEvent};
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
    /// Production turn-wide call buffer. Structured calls stay here until the
    /// daemon wrapper classifies the terminal cause (length vs stop vs
    /// malformed) — never released from [`SpecEmit::finish`] alone.
    deferred: DsmlDeferredCalls,
    eos_token: u32,
    /// Every committed (non-EOS) token in order for DSML asst-turn cache replay.
    /// Exposed via [`SpecEmit::streamed_tokens`]; must match the raw sequence
    /// `build_deepseek4_dsml_prompt` replays on a cache hit.
    streamed_tokens: Vec<u32>,
    /// Visible Token-channel prose accumulated this turn (think/Reasoning
    /// excluded). Fingerprint text for the asst-turn cache store.
    visible_acc: String,
    /// In-step tool-call grammar, threaded into the fused spec step via
    /// `grammar()`. `None` ⇒ no tools (or the bespoke loop, which owns its own
    /// matcher and never calls `grammar()`). The matcher advances inside the
    /// spec step ONLY — `observe` must NOT touch it (single-advance invariant).
    grammar: Option<Deepseek4SpecGrammar>,
}

/// Map a visible DSML channel event into a client event. ToolCalls/Malformed
/// never reach here — they are absorbed by [`DsmlDeferredCalls`].
fn visible_client_event(ev: StreamEvent) -> Option<ClientEvent> {
    match ev {
        StreamEvent::Token(text) => Some(ClientEvent::Token(text)),
        StreamEvent::Reasoning(text) => Some(ClientEvent::Reasoning(text)),
        StreamEvent::ToolCalls(_) | StreamEvent::Malformed { .. } => None,
    }
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
            deferred: DsmlDeferredCalls::new(),
            eos_token: ctx.eos,
            streamed_tokens: Vec::new(),
            visible_acc: String::new(),
            grammar,
        })
    }

    /// Feed one committed token's decoded text through the DSML parser, mapping
    /// visible channels to `ClientEvent` and absorbing tool calls turn-wide via
    /// the production [`DsmlDeferredCalls`] component. Structured calls are
    /// **never** released here or from [`SpecEmit::finish`] — only after the
    /// daemon wrapper classifies a tool-safe terminal.
    fn feed_and_emit(&mut self, token: u32) -> Vec<ClientEvent> {
        let mut events = Vec::new();
        self.streamed_tokens.push(token);
        let frag = self.tokenizer.decode(&[token]);
        for ev in self.deferred.absorb_all(self.parser.feed(&frag)) {
            if let Some(ce) = visible_client_event(ev) {
                if let ClientEvent::Token(ref t) = ce {
                    self.visible_acc.push_str(t);
                }
                events.push(ce);
            }
        }
        events.push(ClientEvent::Committed {
            id: token,
            idx: self.streamed_tokens.len() - 1,
        });
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

    fn streamed_tokens(&self) -> &[u32] {
        &self.streamed_tokens
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
        // Post-loop flush into the production deferred buffer. Visible
        // Token/Reasoning may flush here. Structured ToolCalls are attached as
        // held finish events for the wrapper — the generic generate_spec core
        // must NOT render them before length/malformed is known.
        let mut events = Vec::new();
        let parser = std::mem::replace(&mut self.parser, dsml::StreamParser::new());
        for ev in self.deferred.absorb_all(parser.finish()) {
            if let Some(ce) = visible_client_event(ev) {
                if let ClientEvent::Token(ref t) = ce {
                    self.visible_acc.push_str(t);
                }
                events.push(ce);
            }
        }
        let visible_text = std::mem::take(&mut self.visible_acc);
        let deferred = std::mem::take(&mut self.deferred);
        if deferred.is_malformed() {
            // Fail closed: discard every buffered call; magic finish_reason for
            // the ds4 wrapper (typed FinishSummary field still deferred Minor).
            let _ = deferred.finalize(false);
            return FinishSummary {
                events,
                finish_reason: "malformed_protocol",
                tool_calls: 0,
                visible_text: String::new(),
                decoded_eot: false,
                open_think: false,
            };
        }
        // Provisional finalize without length: length is applied by the wrapper
        // after generate_spec returns (generated >= max_tokens). Held ToolCalls
        // stay on FinishSummary.events for a tool-safe release only.
        let buffered = deferred.buffered_len();
        match deferred.finalize(false) {
            DsmlDeferredOutcome::ToolCalls(calls) => {
                let held: Vec<ToolCall> = calls
                    .into_iter()
                    .map(|c| ToolCall {
                        name: c.name,
                        arguments: c.arguments,
                    })
                    .collect();
                events.push(ClientEvent::ToolCalls(held));
                FinishSummary {
                    events,
                    finish_reason: "tool_calls",
                    tool_calls: buffered,
                    visible_text,
                    decoded_eot: false,
                    open_think: false,
                }
            }
            DsmlDeferredOutcome::Stop | DsmlDeferredOutcome::Length => FinishSummary {
                events,
                finish_reason: "stop",
                tool_calls: 0,
                visible_text,
                decoded_eot: false,
                open_think: false,
            },
            DsmlDeferredOutcome::Malformed { .. } => FinishSummary {
                events,
                finish_reason: "malformed_protocol",
                tool_calls: 0,
                visible_text: String::new(),
                decoded_eot: false,
                open_think: false,
            },
        }
    }
}
