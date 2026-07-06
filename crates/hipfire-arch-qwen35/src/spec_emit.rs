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

use crate::grammar;
use hipfire_runtime::emit_text::{currently_in_think, extract_tool_calls_from_text};
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
    /// `generated` counter at the point of the most recent `observe` — only used
    /// for the attractor-detect log message (byte-for-byte stderr parity).
    generated_hint: usize,
}

impl<'a> Qwen35Emit<'a> {
    /// Build the qwen35 emitter from the model-independent [`SpecEmitCtx`],
    /// extracting the grammar `ToolSchema` list from the request's raw tool JSON
    /// (`ctx.tools`). Returns the arch-erased `Box<dyn SpecEmit>` the daemon
    /// drives. The JSON→schema extraction mirrors the daemon's old
    /// `tool_schemas_dflash` builder.
    pub fn from_ctx(ctx: SpecEmitCtx<'a>) -> Box<dyn SpecEmit + 'a> {
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
        Box::new(Self {
            tokenizer: ctx.tokenizer,
            filter: EosFilter::new(EosFilterConfig::default()),
            bytes_fed_to_filter: 0,
            streamed_tokens: Vec::new(),
            grammar_active,
            grammar_matcher: grammar::Matcher::new(tool_schemas),
            grammar_violated: false,
            eos_token: ctx.eos,
            im_end_token: ctx.im_end,
            stop: ctx.stop,
            max_think_tokens: ctx.max_think,
            open_think_prefix: matches!(ctx.assistant_prefix, AssistantPrefix::OpenThink),
            think_count: 0,
            prev_in_think: false,
            generated_hint: 0,
        })
    }

    /// Push a committed token, run it through the byte filter, and return the
    /// `Committed` + (optional) `Token` events — the shared tail of `begin`'s
    /// first-token emit and `observe`'s per-token emit.
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
        if let FilterAction::Emit(text_bytes) = self.filter.observe(new_bytes) {
            let text = std::str::from_utf8(&text_bytes).unwrap();
            events.push(ClientEvent::Token(text.to_string()));
        }
        events
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
        let first_token_is_eos = first_token == self.eos_token
            || self.im_end_token == Some(first_token)
            || self.tokenizer.is_terminator(first_token);
        EmitOutcome {
            events,
            stop: if first_token_is_eos {
                Some(StopReason::Eos)
            } else {
                None
            },
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

        // Emit the token (committed + filtered).
        let mut events = self.push_and_filter(token);

        // EOS check (token id). Mirrors 4608-4612.
        if token == self.eos_token
            || self.im_end_token == Some(token)
            || self.tokenizer.is_terminator(token)
        {
            return EmitOutcome {
                events,
                stop: Some(StopReason::Eos),
            };
        }

        // User stop-sequence match against the decoded suffix. Mirrors 4622-4628.
        if !self.stop.is_empty() {
            let decoded_suffix = self.tokenizer.decode(&self.streamed_tokens);
            if self
                .stop
                .iter()
                .any(|s| decoded_suffix.ends_with(s.as_str()))
            {
                return EmitOutcome {
                    events,
                    stop: Some(StopReason::StopSequence),
                };
            }
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
                // Force-close: stream `</think>\n` and stop. The inline path
                // wrote this literal token frame directly (not through the
                // filter, not into streamed_tokens) — preserve that exactly.
                events.push(ClientEvent::Token("</think>\n".to_string()));
                return EmitOutcome {
                    events,
                    stop: Some(StopReason::ThinkCap),
                };
            }
        }

        EmitOutcome { events, stop: None }
    }

    fn finish(self: Box<Self>) -> FinishSummary {
        // Tool-call extraction + finish-flush. The conversation-token bake, KV
        // resets, eviction, cache-store, and `done` envelope stay in
        // generate_dflash. Mirrors 4733-4752 + the finish_reason branch 4827-4833
        // (length cap is the caller's call; this returns stop/tool_calls).
        let decoded_full = self.tokenizer.decode(&self.streamed_tokens);
        let emit_tool_calls = extract_tool_calls_from_text(&decoded_full);
        let mut events = Vec::new();
        let tool_calls = emit_tool_calls.len();
        let finish_reason = if !emit_tool_calls.is_empty() {
            "tool_calls"
        } else {
            "stop"
        };
        if !emit_tool_calls.is_empty() {
            events.push(ClientEvent::ToolCalls(emit_tool_calls));
        }
        FinishSummary {
            events,
            finish_reason,
            tool_calls,
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

    /// Hint the emitter of the daemon's current `generated` count so the
    /// attractor-detect log message reports the same number it did inline.
    fn set_generated_hint(&mut self, generated: usize) {
        self.generated_hint = generated;
    }
}
