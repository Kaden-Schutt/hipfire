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
//! The sealed [`OpenAssistantTurn`] is the single byte/stop/visible authority:
//! user stop sequences are quarantined byte-level (split across tokens),
//! `<think>` spans are stripped from the visible projection, and the consuming
//! seal owns text, tool calls, replay tokens, and diagnostics. On top of that
//! core the emitter tracks the beta producer surface the daemon's terminal
//! lowering reads: the full committed token stream (`streamed_tokens`), the
//! decoded-EOT verdict (token-id EOT or a byte-fragmented `<|im_end|>` /
//! `<|endoftext|>` marker completing inside the quarantine), and the unclosed
//! think verdict at finish.

use crate::grammar;
use hipfire_runtime::emit_text::currently_in_think;
use hipfire_runtime::prompt_frame::AssistantPrefix;
use hipfire_runtime::spec::{
    ClientEvent, EmitOutcome, FinishSummary, SpecEmit, SpecEmitCtx, StopReason,
};
use hipfire_runtime::spec_transcript::OpenAssistantTurn;
use hipfire_runtime::tokenizer::Tokenizer;

/// Decoded byte forms of the protocol EOT markers. They join the user stop
/// sequences in the turn's quarantine so a marker split across token
/// boundaries is detected byte-level (suppressed from visible, latches
/// `decoded_eot`) — mirroring beta's EosFilter `stop_at` list.
const PROTOCOL_EOT_MARKERS: [&[u8]; 2] = [b"<|im_end|>", b"<|endoftext|>"];

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
    /// Every committed token in order, for byte decoding + the cache-store the
    /// daemon does after the loop (exposed via [`SpecEmit::streamed_tokens`]).
    streamed_tokens: Vec<u32>,
    /// Latched when a decoded EOT was observed: a semantic EOS token id, or a
    /// byte-level `<|im_end|>` / `<|endoftext|>` marker completing in the turn
    /// quarantine. Carried onto [`FinishSummary::decoded_eot`] so the daemon's
    /// terminal lowering need not re-decode the token stream.
    decoded_eot: bool,
    /// User stop sequences from the request (also fed to the turn quarantine).
    /// Kept for distinguishing a user-stop byte-stop from a protocol EOT
    /// byte-stop when the quarantine fires.
    stop: Vec<String>,
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
        // The turn's byte quarantine holds user stop sequences AND the decoded
        // protocol EOT markers. User stops are quarantined byte-level so a
        // marker split across tokens is held/suppressed exactly like beta's
        // EosFilter does for its `stop_at` list; protocol EOT bytes complete to
        // a decoded-EOT verdict instead of a plain StopSequence.
        let mut stop_markers: Vec<Vec<u8>> = ctx
            .stop
            .iter()
            .filter(|stop| !stop.is_empty())
            .map(|stop| stop.as_bytes().to_vec())
            .collect();
        for marker in PROTOCOL_EOT_MARKERS {
            stop_markers.push(marker.to_vec());
        }
        let turn = OpenAssistantTurn::new_with_reasoning_open(
            stop_markers.iter().map(Vec::as_slice),
            matches!(ctx.assistant_prefix, AssistantPrefix::OpenThink),
        );
        Box::new(Self {
            tokenizer: ctx.tokenizer,
            turn,
            think_scan: Vec::new(),
            streamed_tokens: Vec::new(),
            decoded_eot: false,
            stop: ctx.stop,
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

    /// Whether the most recent byte-stop (turn quarantine `Stop`) was a
    /// protocol EOT marker rather than a user stop sequence. The turn's
    /// quarantine stops at the EARLIEST complete marker in stream order, so
    /// comparing the earliest EOT-marker position against the earliest user
    /// stop position identifies which one fired.
    fn stopped_on_eot(&self) -> bool {
        let raw = std::str::from_utf8(&self.think_scan).unwrap_or("");
        let eot_pos = PROTOCOL_EOT_MARKERS
            .iter()
            .filter_map(|m| raw.find(std::str::from_utf8(m).expect("EOT markers are UTF-8")))
            .min();
        let user_pos = self
            .stop
            .iter()
            .filter(|s| !s.is_empty())
            .filter_map(|s| raw.find(s.as_str()))
            .min();
        match (eot_pos, user_pos) {
            (Some(e), Some(u)) => e < u,
            (Some(_), None) => true,
            _ => false,
        }
    }

    /// Admit one committed token to the sealed-turn owner and the committed
    /// stream. Returns `(events, stopped, generation_advanced)`.
    ///
    /// * Post-stop tokens return empty events and `stopped` (the sealed turn
    ///   is terminal; nothing further is committed).
    /// * A semantic EOS token id is committed to the stream (the daemon trims
    ///   the trailing `im_end` from the cache body), stops the turn, and
    ///   latches `decoded_eot`.
    /// * A byte-stop (user stop or protocol EOT marker completing inside the
    ///   quarantine) is NOT committed: the marker-completing token carries no
    ///   Committed event and no visible bytes.
    /// * Otherwise the token is committed: `Committed` + think-stripped,
    ///   marker-free visible `Token` events.
    fn admit(&mut self, token: u32) -> (Vec<ClientEvent>, bool, bool) {
        let mut events = Vec::new();
        if self.turn.stopped() {
            return (events, true, false);
        }
        // Semantic EOS is an owner stop.  Do not decode it through the
        // transcript: otherwise a tokenizer whose EOS text is ordinary bytes
        // can leak those bytes when the request has no user stop markers.
        // The EOS token id is still committed so the daemon's cache body
        // trims it exactly like beta's push_and_filter stream.
        if self.semantic_stop(token) {
            self.streamed_tokens.push(token);
            events.push(ClientEvent::Committed {
                id: token,
                idx: self.streamed_tokens.len() - 1,
            });
            self.decoded_eot = true;
            self.turn.stop();
            return (events, true, false);
        }
        let raw = self.tokenizer.decode_bytes(&[token]);
        let delta = self.turn.observe(token, &raw);
        self.think_scan.extend_from_slice(&raw);
        if delta.stopped {
            // Byte-stop: the marker-completing token is not committed (its
            // bytes are the marker + discarded post-marker payload). The stop
            // reason is resolved by the caller via `stopped_on_eot`.
            return (events, true, false);
        }
        self.streamed_tokens.push(token);
        events.push(ClientEvent::Committed {
            id: token,
            idx: self.streamed_tokens.len() - 1,
        });
        if !delta.bytes.is_empty() {
            let text = std::str::from_utf8(&delta.bytes)
                .expect("OpenAssistantTurn emits valid UTF-8")
                .to_string();
            events.push(ClientEvent::Token(text));
        }
        (events, false, true)
    }
}

impl<'a> SpecEmit for Qwen35Emit<'a> {
    fn begin(&mut self, first_token: u32) -> EmitOutcome {
        // First-token emit (committed + filtered token), then seed the grammar
        // matcher with the first token's text. Mirrors generate_dflash 4452-4489.
        // The first token is ALWAYS committed (daemon: "stop still commits the
        // raw first token") so `spec_outcome_seed_committable` stays true even
        // when it immediately completes a stop marker.
        self.streamed_tokens.push(first_token);
        let mut events = vec![ClientEvent::Committed {
            id: first_token,
            idx: self.streamed_tokens.len() - 1,
        }];
        if self.grammar_active && !self.semantic_stop(first_token) {
            let text = self.tokenizer.decode(&[first_token]);
            self.grammar_matcher.advance(&text);
        }
        // First-token EOS guard: if the prefill's first token is itself a
        // terminator we must NOT enter the spec loop (the inline `while
        // !first_token_is_eos`). Surface as a stop so the caller skips the loop.
        let first_token_is_eos = self.semantic_stop(first_token);
        if first_token_is_eos {
            self.decoded_eot = true;
            self.turn.stop();
            return EmitOutcome {
                events,
                generation_advanced: false,
                stop: Some(StopReason::Eos),
            };
        }
        let raw = self.tokenizer.decode_bytes(&[first_token]);
        let delta = self.turn.observe(first_token, &raw);
        self.think_scan.extend_from_slice(&raw);
        if delta.stopped {
            let eot = self.stopped_on_eot();
            if eot {
                self.decoded_eot = true;
            }
            return EmitOutcome {
                events,
                generation_advanced: false,
                stop: Some(if eot {
                    StopReason::Eos
                } else {
                    StopReason::StopSequence
                }),
            };
        }
        if !delta.bytes.is_empty() {
            let text = std::str::from_utf8(&delta.bytes)
                .expect("OpenAssistantTurn emits valid UTF-8")
                .to_string();
            events.push(ClientEvent::Token(text));
        }
        EmitOutcome {
            events,
            generation_advanced: true,
            stop: None,
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

        if stopped {
            // Byte-stop (semantic EOS was handled above): a protocol EOT
            // marker completing in the quarantine is Eos, a user stop
            // sequence is StopSequence.
            let eot = self.stopped_on_eot();
            if eot {
                self.decoded_eot = true;
            }
            return EmitOutcome {
                events,
                generation_advanced,
                stop: Some(if eot {
                    StopReason::Eos
                } else {
                    StopReason::StopSequence
                }),
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
        // Unclosed think at finish is a nonretryable unsafe terminal (beta's
        // `filter.in_think()` verdict, computed from the same raw stream).
        let raw_str = std::str::from_utf8(&self.think_scan).unwrap_or("");
        let open_think = currently_in_think(raw_str, self.open_think_prefix);
        if open_think {
            return FinishSummary {
                events: Vec::new(),
                finish_reason: "open_think",
                tool_calls: 0,
                finalized: Some(finalized),
                visible_text: String::new(),
                decoded_eot: false,
                open_think: true,
            };
        }
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
        let visible_text = finalized.text().to_string();
        let decoded_eot = self.decoded_eot
            || PROTOCOL_EOT_MARKERS.iter().any(|m| {
                raw_str.contains(std::str::from_utf8(m).expect("EOT markers are UTF-8"))
            });
        FinishSummary {
            events,
            finish_reason,
            tool_calls,
            finalized: Some(finalized),
            visible_text,
            decoded_eot,
            open_think: false,
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

    /// Emitter-authoritative decoded EOT (token id and/or byte marker).
    fn decoded_eot(&self) -> bool {
        self.decoded_eot
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
    use hipfire_runtime::prompt_frame::ToolCall;
    use hipfire_runtime::spec::SpecEmit;
    use hipfire_runtime::tokenizer::Tokenizer;

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

        // A decoded protocol EOT marker is a byte-level stop even inside an
        // ordinary-looking token: the marker completes in the turn quarantine
        // and latches decoded EOT (beta's EosFilter `stop_at` semantics).
        let protocol_prefix = make_emit().begin(19);
        assert_eq!(protocol_prefix.stop, Some(StopReason::Eos));
        assert!(protocol_prefix.events.iter().all(|event| {
            !matches!(event, ClientEvent::Token(text) if text.contains("<|im_end|>"))
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

    // ── Beta producer-surface tests (streamed tokens / decoded EOT) ───────

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
            tools: None,
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
    fn raw_committed_once_per_token_including_markers() {
        // Tools are not grammar-active here (tools: None), so the raw
        // tool-call marker tokens are ordinary committed bytes — each appears
        // exactly once as a Committed event with a sequential idx.
        let body = "x<tool_call>{}</tool_call>";
        let (stream, _, raw) = drive_text(body);
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
    fn think_bytes_suppressed_from_token_channel() {
        let (stream, finish, raw) = drive_text("<think>secret</think>answer");
        assert!(!raw.is_empty());
        let visible = tokens_text(&stream);
        assert!(!visible.contains("<think>"));
        assert!(!visible.contains("secret"));
        assert!(visible.contains("answer"));
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
        assert!(finish.decoded_eot, "token-id EOT must latch decoded_eot");
    }

    #[test]
    fn split_decoded_eot_across_tokens_latches_and_suppresses_marker() {
        // Byte-fragment the <|im_end|> marker across tokens (100+b map).
        let marker = b"<|im_end|>";
        let mut ids: Vec<u32> = Vec::new();
        ids.push(100 + b'h' as u32);
        ids.push(100 + b'i' as u32);
        let mid = marker.len() / 2;
        for &b in &marker[..mid] {
            ids.push(100 + b as u32);
        }
        for &b in &marker[mid..] {
            ids.push(100 + b as u32);
        }
        let tok = test_tokenizer();
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
        let visible = tokens_text(&stream);
        assert!(!visible.contains("<|im_end|>"), "marker bytes suppressed");
        assert!(visible.contains("hi"));
        let finish = emit.finish();
        assert!(finish.decoded_eot, "split EOT must latch decoded_eot");
        assert_eq!(finish.finish_reason, "stop");
        assert_eq!(finish.tool_calls, 0);
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
}
