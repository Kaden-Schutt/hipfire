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
use hipfire_runtime::emit_text::currently_in_think;
use hipfire_runtime::prompt_frame::AssistantPrefix;
use hipfire_runtime::spec::{
    ClientEvent, EmitOutcome, FinishSummary, SpecEmit, SpecEmitCtx, StopReason,
};
use hipfire_runtime::spec_transcript::OpenAssistantTurn;
use hipfire_runtime::tokenizer::Tokenizer;

pub struct Qwen35Emit<'a> {
    tokenizer: &'a Tokenizer,
    /// The sole owner of raw bytes, stop quarantine, visible projection, and
    /// the sealed turn result. No parallel token or transcript heuristic is
    /// allowed here.
    turn: OpenAssistantTurn,
    /// Raw bytes used only for the existing think-budget guard. This is not a
    /// second output transcript: the sealed turn remains authoritative for all
    /// client-visible text and tools.
    think_scan: Vec<u8>,
    grammar_active: bool,
    grammar_matcher: grammar::Matcher,
    /// Set when a committed token violated the grammar; the daemon reads it via
    /// [`Self::grammar_violated`] to force the post-turn KV/recurrent reset.
    grammar_violated: bool,
    eos_token: u32,
    im_end_token: Option<u32>,
    max_think_tokens: usize,
    open_think_prefix: bool,
    think_count: usize,
    prev_in_think: bool,
    /// Think-budget force-close continuation drained by `take_forced` so the
    /// target advances over `</think>` and continues into visible output.
    forced: Vec<u32>,
    /// Number of queued continuation tokens still being fed back through
    /// `observe`. They must update grammar/filter state but must not re-trigger
    /// the think budget before the close has drained.
    forced_remaining: usize,
    /// Prevent an endlessly re-opened reasoning span from being force-closed
    /// repeatedly. A second cap hit hard-stops through `ThinkCap`.
    think_force_closed: bool,
    /// `generated` counter at the point of the most recent `observe` — only used
    /// for the attractor-detect log message (byte-for-byte stderr parity).
    generated_hint: usize,
}

fn think_continuation_text() -> String {
    std::env::var("HIPFIRE_THINK_CONTINUATION").unwrap_or_else(|_| "</think>\n\n".to_string())
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
        // StopQuarantine is for user-provided stop sequences only. Protocol
        // terminators are resolved by `semantic_stop` before decoding; adding
        // their decoded bytes here would make an ordinary user payload look
        // like a stop and would conflate the two termination paths.
        let stop_markers: Vec<Vec<u8>> = ctx
            .stop
            .iter()
            .filter(|stop| !stop.is_empty())
            .map(|stop| stop.as_bytes().to_vec())
            .collect();
        let turn = OpenAssistantTurn::new_with_reasoning_open(
            stop_markers.iter().map(Vec::as_slice),
            matches!(ctx.assistant_prefix, AssistantPrefix::OpenThink),
        );
        Box::new(Self {
            tokenizer: ctx.tokenizer,
            turn,
            think_scan: Vec::new(),
            grammar_active,
            grammar_matcher: grammar::Matcher::new(tool_schemas),
            grammar_violated: false,
            eos_token: ctx.eos,
            im_end_token: ctx.im_end,
            max_think_tokens: ctx.max_think,
            open_think_prefix: matches!(ctx.assistant_prefix, AssistantPrefix::OpenThink),
            think_count: 0,
            prev_in_think: false,
            forced: Vec::new(),
            forced_remaining: 0,
            think_force_closed: false,
            generated_hint: 0,
        })
    }

    /// Admit one target token to the sealed-turn owner. Terminal tokens are
    /// owned stops and never become wire diagnostics.
    fn semantic_stop(&self, token: u32) -> bool {
        token == self.eos_token
            || self.im_end_token == Some(token)
            || self.tokenizer.is_terminator(token)
    }

    fn admit(&mut self, token: u32) -> (Vec<ClientEvent>, bool, bool) {
        let mut events = Vec::new();
        if self.turn.stopped() {
            return (events, true, false);
        }
        // Semantic EOS is an owner stop.  Do not decode it through the
        // transcript: otherwise a tokenizer whose EOS text is ordinary bytes
        // can leak those bytes when the request has no user stop markers.
        if self.semantic_stop(token) {
            self.turn.stop();
            return (events, true, false);
        }
        let raw = self.tokenizer.decode_bytes(&[token]);
        let delta = self.turn.observe(token, &raw);
        self.think_scan.extend_from_slice(&raw);
        if !delta.bytes.is_empty() {
            let text = std::str::from_utf8(&delta.bytes)
                .expect("OpenAssistantTurn emits valid UTF-8")
                .to_string();
            events.push(ClientEvent::Token(text));
        }
        (events, delta.stopped, !delta.stopped)
    }
}

impl<'a> SpecEmit for Qwen35Emit<'a> {
    fn begin(&mut self, first_token: u32) -> EmitOutcome {
        // First-token emit (committed + filtered token), then seed the grammar
        // matcher with the first token's text. Mirrors generate_dflash 4452-4489.
        let (events, stopped, generation_advanced) = self.admit(first_token);
        if self.grammar_active && !self.semantic_stop(first_token) {
            let text = self.tokenizer.decode(&[first_token]);
            self.grammar_matcher.advance(&text);
        }
        // First-token EOS guard: if the prefill's first token is itself a
        // terminator we must NOT enter the spec loop (the inline `while
        // !first_token_is_eos`). Surface as a stop so the caller skips the loop.
        let first_token_is_eos = self.semantic_stop(first_token);
        EmitOutcome {
            events,
            generation_advanced,
            stop: if first_token_is_eos {
                Some(StopReason::Eos)
            } else if stopped {
                Some(StopReason::StopSequence)
            } else {
                None
            },
        }
    }

    fn observe(&mut self, token: u32) -> EmitOutcome {
        // The owner is terminal after a stop. Preserve the explicit generation
        // advancement event, but do not run grammar or any other side effect.
        if self.turn.stopped() {
            return EmitOutcome {
                events: Vec::new(),
                generation_advanced: false,
                stop: Some(StopReason::StopSequence),
            };
        }

        // Semantic EOS is resolved before grammar or byte projection.  The
        // owner, not the marker filter, decides when the turn ends.
        if self.semantic_stop(token) {
            let (events, _, generation_advanced) = self.admit(token);
            return EmitOutcome {
                events,
                generation_advanced,
                stop: Some(StopReason::Eos),
            };
        }

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

        // Admit the token (committed + sealed-turn projection).
        let (events, stopped, generation_advanced) = self.admit(token);
        let draining_forced = self.forced_remaining > 0;
        if draining_forced {
            self.forced_remaining -= 1;
        }

        // EOS check (token id). Mirrors 4608-4612.
        if token == self.eos_token
            || self.im_end_token == Some(token)
            || self.tokenizer.is_terminator(token)
        {
            return EmitOutcome {
                events,
                generation_advanced,
                stop: Some(StopReason::Eos),
            };
        }
        if stopped {
                return EmitOutcome {
                    events,
                generation_advanced,
                    stop: Some(StopReason::StopSequence),
                };
            }

        // max_think_tokens enforcement. Mirrors 4632-4664.
        if self.max_think_tokens > 0 {
            let raw_str = std::str::from_utf8(&self.think_scan).unwrap_or("");
            let in_think = currently_in_think(raw_str, self.open_think_prefix);
            if in_think && !self.prev_in_think {
                if self.think_force_closed && !draining_forced {
                    return EmitOutcome {
                        events,
                        generation_advanced,
                        stop: Some(StopReason::ThinkCap),
                    };
                }
                self.think_count = 0;
            }
            if in_think {
                self.think_count += 1;
            }
            self.prev_in_think = in_think;

            if in_think && self.think_count >= self.max_think_tokens {
                if draining_forced {
                    return EmitOutcome {
                        events,
                        generation_advanced,
                        stop: None,
                    };
                }
                if !self.think_force_closed {
                    self.forced = self.tokenizer.encode(&think_continuation_text());
                    self.forced_remaining = self.forced.len();
                    self.think_force_closed = true;
                    return EmitOutcome {
                        events,
                        generation_advanced,
                        stop: None,
                    };
                }
                // A pathological re-open after the injected close hard-stops.
                return EmitOutcome {
                    events,
                    generation_advanced,
                    stop: Some(StopReason::ThinkCap),
                };
            }
        }

        EmitOutcome {
            events,
            generation_advanced,
            stop: None,
        }
    }

    fn finish(self: Box<Self>) -> FinishSummary {
        // The consuming seal is the only source for terminal bytes and tools.
        let mut events = Vec::new();
        let finalized = self.turn.seal();
        // Token-ID diagnostics are published only after the consuming seal;
        // no held marker prefix or post-stop token can reach the wire.
        events.extend(
            finalized
                .diagnostic_tokens()
                .iter()
                .enumerate()
                .map(|(idx, &id)| ClientEvent::Committed { id, idx }),
        );
        if !finalized.terminal_delta().bytes.is_empty() {
            let text = std::str::from_utf8(&finalized.terminal_delta().bytes)
                .expect("OpenAssistantTurn emits valid UTF-8")
                .to_string();
            events.push(ClientEvent::Token(text));
        }
        let emit_tool_calls = finalized.tool_calls().to_vec();
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
            finalized: Some(finalized),
        }
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

    fn take_forced(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.forced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_hf_json(
            r#"{
                "model": {
                    "vocab": {
                        "<think>": 0,
                        "a": 1,
                        "b": 2,
                        "<": 3,
                        "/": 4,
                        "t": 5,
                        "h": 6,
                        "i": 7,
                        "n": 8,
                        "k": 9,
                        ">": 10,
                        "\\n": 11,
                        "<|endoftext|>": 12,
                        "<stop>": 13,
                        "<|im_end|>": 14,
                        "<tool_call>{\"name\":\"bash\",\"arguments\":{\"command\":\"ls\"}}</tool_call>": 15,
                        "<think>reason</think>answer": 16,
                        "safe<st": 17,
                        "op>tail": 18,
                        "<|im_end|>ordinary": 19
                    },
                    "merges": []
                },
                "added_tokens": [
                    {"id": 12, "content": "<|endoftext|>", "special": true},
                    {"id": 14, "content": "<|im_end|>", "special": true}
                ]
            }"#,
        )
        .expect("test tokenizer")
    }

    #[test]
    fn first_think_cap_drains_close_then_reopened_think_stops() {
        let tokenizer = tokenizer();
        let ctx = SpecEmitCtx {
            tokenizer: &tokenizer,
            eos: 12,
            im_end: None,
            tools: None,
            stop: Vec::new(),
            max_think: 2,
            max_tokens: 64,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        };
        let mut emit = Qwen35Emit::from_ctx(ctx);

        emit.begin(0);
        assert!(emit.observe(1).stop.is_none());
        assert!(emit.observe(2).stop.is_none());
        let forced = emit.take_forced();
        assert_eq!(forced, tokenizer.encode(&think_continuation_text()));
        assert!(!forced.is_empty());

        for &token in &forced {
            let outcome = emit.observe(token);
            assert!(outcome.stop.is_none());
            assert!(outcome.generation_advanced);
        }
        assert_eq!(emit.observe(0).stop, Some(StopReason::ThinkCap));
    }

    #[test]
    fn from_ctx_preserves_think_visibility_and_wires_stop_markers() {
        let tokenizer = tokenizer();
        let make_emit = || {
            Qwen35Emit::from_ctx(SpecEmitCtx {
                tokenizer: &tokenizer,
                eos: 12,
                im_end: Some(14),
                tools: None,
                stop: vec!["<stop>".into()],
                max_think: 0,
                max_tokens: 64,
                assistant_prefix: AssistantPrefix::Plain,
                think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
                decoded_vocab: None,
            })
        };
        let mut emit = make_emit();

        let visible: Vec<String> = emit
            .begin(16)
            .events
            .into_iter()
            .filter_map(|event| match event {
                ClientEvent::Token(text) => Some(text),
                _ => None,
            })
            .collect();
        assert_eq!(visible.concat(), "answer");

        let stop = make_emit().begin(13);
        assert_eq!(stop.stop, Some(StopReason::StopSequence));
        assert!(!stop
            .events
            .iter()
            .any(|event| { matches!(event, ClientEvent::Token(text) if text.contains("<stop>")) }));

        let im_end = make_emit().begin(14);
        assert_eq!(im_end.stop, Some(StopReason::Eos));
        assert!(!im_end.events.iter().any(|event| {
            matches!(event, ClientEvent::Token(text) if text.contains("<|im_end|>"))
        }));

        // A decoded protocol-looking byte sequence is ordinary user content
        // unless its token takes the semantic EOS path. It must not be fed to
        // the user-stop quarantine as a static marker.
        let protocol_prefix = make_emit().begin(19);
        assert_eq!(protocol_prefix.stop, None);
        assert!(protocol_prefix.events.iter().any(|event| {
            matches!(event, ClientEvent::Token(text) if text == "<|im_end|>ordinary")
        }));
    }

    #[test]
    fn stop_tool_call_suppresses_visible_output_and_tool_calls() {
        let tokenizer = tokenizer();
        let mut emit = Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tokenizer,
            eos: 12,
            im_end: None,
            tools: None,
            stop: vec!["<tool_call>".into()],
            max_think: 0,
            max_tokens: 64,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });

        let outcome = emit.begin(15);
        assert_eq!(outcome.stop, Some(StopReason::StopSequence));
        assert!(!outcome
            .events
            .iter()
            .any(|event| { matches!(event, ClientEvent::Token(_) | ClientEvent::ToolCalls(_)) }));
        let post_stop = emit.observe(1);
        assert_eq!(post_stop.stop, Some(StopReason::StopSequence));
        assert!(post_stop.events.is_empty());
        assert!(!post_stop.generation_advanced);
        let summary = emit.finish();
        assert_eq!(summary.tool_calls, 0);
        assert!(!summary
            .events
            .iter()
            .any(|event| matches!(event, ClientEvent::ToolCalls(_))));
    }

    #[test]
    fn tool_calls_before_stop_are_extracted_from_safe_transcript() {
        let tokenizer = tokenizer();
        let mut emit = Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tokenizer,
            eos: 12,
            im_end: None,
            tools: None,
            stop: vec!["<stop>".into()],
            max_think: 0,
            max_tokens: 64,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });

        assert!(emit.begin(15).stop.is_none());
        assert_eq!(emit.observe(13).stop, Some(StopReason::StopSequence));
        let summary = emit.finish();
        assert_eq!(summary.tool_calls, 1);
        assert!(summary.events.iter().any(|event| {
            matches!(event, ClientEvent::ToolCalls(calls) if calls.len() == 1 && calls[0].name == "bash")
        }));
    }

    #[test]
    fn split_stop_marker_is_quarantined_by_the_turn_owner() {
        let tokenizer = tokenizer();
        let mut emit = Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tokenizer,
            eos: 12,
            im_end: None,
            tools: None,
            stop: vec!["<stop>".into()],
            max_think: 0,
            max_tokens: 64,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });

        let first = emit.begin(17);
        assert!(first.stop.is_none());
        assert!(first
            .events
            .iter()
            .any(|event| { matches!(event, ClientEvent::Token(text) if text == "safe") }));
        let second = emit.observe(18);
        assert_eq!(second.stop, Some(StopReason::StopSequence));
        assert!(!second
            .events
            .iter()
            .any(|event| { matches!(event, ClientEvent::Token(text) if text.contains("<stop>")) }));
        let summary = emit.finish();
        assert!(!summary
            .events
            .iter()
            .any(|event| { matches!(event, ClientEvent::Token(text) if text.contains("<stop>")) }));
    }

    #[test]
    fn semantic_eos_precedes_filter_stop_and_still_suppresses_marker() {
        let tokenizer = tokenizer();
        let mut emit = Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tokenizer,
            eos: 12,
            im_end: None,
            tools: None,
            stop: Vec::new(),
            max_think: 0,
            max_tokens: 64,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });

        assert!(emit.begin(1).stop.is_none());
        let outcome = emit.observe(12);
        assert_eq!(outcome.stop, Some(StopReason::Eos));
        assert!(!outcome.events.iter().any(|event| {
            matches!(event, ClientEvent::Token(text) if text.contains("<|endoftext|>"))
        }));
    }

    #[test]
    fn semantic_eos_seals_trailing_partial_user_stop_prefix() {
        let tokenizer = tokenizer();
        let mut emit = Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tokenizer,
            eos: 12,
            im_end: None,
            tools: None,
            stop: vec!["<stop>".into()],
            max_think: 0,
            max_tokens: 64,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });

        assert!(emit.begin(17).stop.is_none());
        assert_eq!(emit.observe(12).stop, Some(StopReason::Eos));

        let summary = emit.finish();
        let finalized = summary.finalized.expect("Qwen seal");
        assert_eq!(finalized.text(), "safe<st");
        assert_eq!(finalized.terminal_delta().bytes, b"<st");
        assert_eq!(finalized.replay_tokens(), Some([17].as_slice()));
    }

    #[test]
    fn sealed_owner_exposes_exact_replay_and_only_sealed_diagnostics() {
        let mut aligned = OpenAssistantTurn::new([b"<stop>".as_slice()]);
        aligned.observe(21, b"hello ");
        aligned.observe(22, b"world");
        let finalized = aligned.seal();
        assert_eq!(finalized.replay_tokens(), Some([21, 22].as_slice()));
        assert_eq!(finalized.diagnostic_tokens(), [21, 22].as_slice());

        let mut cut = OpenAssistantTurn::new([b"<stop>".as_slice()]);
        cut.observe(31, b"hello<stop>");
        cut.observe(32, b"post-stop-tool-payload");
        let finalized = cut.seal();
        assert_eq!(finalized.replay_tokens(), None);
        assert!(finalized.diagnostic_tokens().is_empty());

        let mut aligned = OpenAssistantTurn::new([b"<stop>".as_slice()]);
        aligned.observe(21, b"hello");
        aligned.observe(22, b"<stop>");
        let finalized = aligned.seal();
        assert_eq!(finalized.replay_tokens(), Some([21].as_slice()));
        assert_eq!(finalized.diagnostic_tokens(), [21].as_slice());
    }

    #[test]
    fn post_stop_tool_payload_is_absent_from_wire_tools_and_replay() {
        let tokenizer = tokenizer();
        let mut emit = Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tokenizer,
            eos: 12,
            im_end: None,
            tools: None,
            stop: vec!["<stop>".into()],
            max_think: 0,
            max_tokens: 64,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });

        assert!(emit.begin(17).generation_advanced);
        let stopped = emit.observe(18);
        assert_eq!(stopped.stop, Some(StopReason::StopSequence));
        assert!(!stopped.generation_advanced);
        assert!(emit.observe(15).events.is_empty());
        let summary = emit.finish();
        let finalized = summary.finalized.expect("Qwen seal");
        assert_eq!(finalized.text(), "safe");
        assert!(finalized.tool_calls().is_empty());
        assert!(finalized.replay_tokens().is_none());
        assert!(finalized.diagnostic_tokens().is_empty());
        assert!(!summary
            .events
            .iter()
            .any(|event| { matches!(event, ClientEvent::Committed { id: 15, .. }) }));
    }
}
