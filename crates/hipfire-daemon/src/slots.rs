// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Experimental multi-slot daemon backend — alternate model owner, NOT a
//! continuous-batching mode.
//!
//! This module is a **daemon-owned** backend that owns exactly one `SlotEngine`
//! (one weight copy) when `experimental_multi_slot=true`. It is an *alternate*
//! to the ordinary `LoadedModel` path, not a switch that enables batching on
//! that model.
//!
//! Continuous-batching integration is **deferred** — this backend deliberately
//! does not integrate with `ContinuousBatchScheduler`, `qwen35/batch.rs`, or
//! `hipfire-generate` batch code. The `continuous_batch_capable` advertisement
//! remains `false` while this backend is active, and any future batch
//! integration must be an explicit, separate cutover that preserves the
//! invariant: ordinary/default daemon load/generate remains byte-for-byte on the
//! existing daemon path when `experimental_multi_slot` is absent.
//!
//! Load-time CPU preflight opens the HFQ, requires arch_id 5|6, rejects vision
//! tensors/model, parses Qwen config/tokenizer/chat template, then spawns
//! `SlotEngine`. Generate validates text-only supported wire fields and builds
//! prompt/convo/continuation with the CLI `slots.rs` semantics.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use hipfire_arch_qwen35::spec_emit::Qwen35Emit;
use hipfire_engine::emit::{
    emit_gen_start, emit_qwen_ar_cancelled, emit_reasoning_token, emit_visible_token,
    QWEN_AR_SEMANTIC_CONTRACT_VERSION,
};
use hipfire_engine::terminal::{
    batch_bind_active, batch_check_abort, batch_clear_terminal, batch_mark_ready_with_pending,
    batch_transition_to_queued, batch_wait_decision, BatchAttemptScope, LaneTicket,
    CLIENT_TERMINAL_COMMIT_TIMEOUT,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::prompt_frame::{
    AssistantPrefix, ChatFrame, JinjaChatFrame, Message, Role, ThinkMode,
};
use hipfire_runtime::serve::{Continuation, DoneReason, Event, SubmitRequest};
use hipfire_runtime::spec::{ClientEvent, SpecEmitCtx};
use hipfire_runtime::tokenizer::Tokenizer;

/// Daemon-owned slot backend. Owns one weight copy via SlotEngine.
pub struct SlotBackend {
    chat_template: Option<String>,
    engine: hipfire_arch_qwen35::serve_engine::SlotEngine,
    tokenizer: Tokenizer,
    arch_str: String,
    dim: usize,
    layers: usize,
    vocab: usize,
    active: AtomicUsize,
}

impl SlotBackend {
    /// CPU preflight then GPU load. Called only when experimental_multi_slot load is requested.
    pub fn load(
        model_path: &str,
        n_slots: usize,
        cap_tokens: usize,
        prefill_chunk: usize,
    ) -> Result<Self, String> {
        // CPU preflight: open HFQ, arch, VL, config, tokenizer.
        let preflight = cpu_preflight(model_path)?;
        let arch_str = preflight.arch_str.clone();
        let dim = preflight.dim;
        let layers = preflight.layers;
        let chat_template = preflight.chat_template;
        let vocab = preflight.vocab;
        let tokenizer = preflight.tokenizer;

        let n_slots = n_slots.max(1);
        let cap_tokens = cap_tokens.max(1);
        let prefill_chunk = prefill_chunk.max(1).min(cap_tokens);

        let engine = hipfire_arch_qwen35::serve_engine::SlotEngine::spawn(
            hipfire_arch_qwen35::serve_engine::EngineConfig {
                model_path: PathBuf::from(model_path),
                n_slots,
                cap_tokens,
                prefill_chunk,
                host_budget_bytes: 16 * 1024 * 1024 * 1024,
                swap_dir: std::env::temp_dir().join("hipfire-serve-swap"),
            },
        )
        .map_err(|e| format!("SlotEngine spawn: {e}"))?;

        Ok(Self {
            engine,
            tokenizer,
            arch_str,
            dim,
            chat_template,
            layers,
            vocab,
            active: AtomicUsize::new(0),
        })
    }

    pub fn arch_str(&self) -> &str {
        &self.arch_str
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn layers(&self) -> usize {
        self.layers
    }
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn active_count(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }

    fn acquire_guard(&self) -> Option<SlotGuard<'_>> {
        let prev = self.active.fetch_add(1, Ordering::SeqCst);
        if prev >= 32 {
            self.active.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(SlotGuard { backend: self })
    }
    /// Reset idle sessions. May reject if requests are in flight. Checked teardown.
    pub fn reset(&self) -> Result<(), String> {
        if self.active_count() > 0 {
            return Err("reset refused: slot requests active".to_string());
        }
        let res = self.engine.reset();
        if res.is_ok() {
            // Clear keyed registry on successful teardown is done by daemon, but also clear here for safety.
            hipfire_engine::terminal::batch_clear_all_terminals();
        }
        res
    }

    /// Checked consuming shutdown for unload / model swap. The caller must
    /// first own the sole Arc and verify no request guards are active.
    pub fn shutdown(self) -> Result<(), String> {
        if self.active_count() > 0 {
            return Err("unload refused: slot requests active".to_string());
        }
        self.engine.shutdown()
    }

    /// Generate handler for experimental slot mode.
    ///
    /// Requires `experimental_multi_slot=true` on the wire, validates
    /// text-only fields, builds prompt/convo/continuation, submits with
    /// sampling, streams token/reasoning, handles abort via keyed registry +
    /// close, and performs two-phase terminal commit with byte-identical done.
    pub fn handle_generate<W: std::io::Write>(
        &self,
        msg: &serde_json::Value,
        stdout: &mut W,
        id: &str,
        attempt_id: u64,
    ) -> Result<(), String> {
        // Guard active count and ensure keyed entry cleared on every exit.
        let Some(_guard) = self.acquire_guard() else {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                "too many concurrent slot requests (bounded worker limit hit)",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            batch_clear_terminal(id, attempt_id);
            return Ok(());
        };
        // Ensure keyed registry entry cleared on exit (including error paths).
        struct ClearOnExit<'a> {
            id: String,
            attempt_id: u64,
            _marker: std::marker::PhantomData<&'a ()>,
        }
        impl Drop for ClearOnExit<'_> {
            fn drop(&mut self) {
                batch_clear_terminal(&self.id, self.attempt_id);
            }
        }
        let _clear = ClearOnExit {
            id: id.to_string(),
            attempt_id,
            _marker: std::marker::PhantomData,
        };
        hipfire_engine::terminal::set_active_attempt_id(attempt_id);
        let _scope = BatchAttemptScope::enter(attempt_id);

        // Require experimental marker.
        if !is_experimental_generate(msg) {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                "experimental_multi_slot=true required for slot backend",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }

        // Defensive validation of unsupported wire fields.
        if let Some(err) = validate_generate_caps(msg) {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                &err,
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }

        let temperature = msg
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let top_p = msg.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        let top_k = msg.get("top_k").and_then(|v| v.as_i64()).unwrap_or(0);
        if !(0..=i32::MAX as i64).contains(&top_k) {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                "top_k must fit a non-negative 32-bit integer",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }
        let client_seed = match hipfire_engine::wire_seed::parse_wire_seed(msg.get("seed")) {
            Ok(seed) => seed,
            Err(reason) => {
                hipfire_engine::emit::emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!("seed: {reason}"),
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return Ok(());
            }
        };
        let seed = hipfire_engine::scheduler::request_seed_for(
            &hipfire_engine::terminal::AttemptKey::new(id, attempt_id),
            client_seed,
        );

        let max_tokens = msg
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize;
        let max_think_tokens = msg
            .get("max_think_tokens")
            .and_then(|v| v.as_u64())
            .or_else(|| {
                msg.get("params")
                    .and_then(|p| p.get("max_think_tokens"))
                    .and_then(|v| v.as_u64())
            });

        if max_think_tokens.is_some_and(|n| n >= 2) {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                "finite reasoning caps not supported in experimental multi-slot (max_think_tokens >=2)",
                "unsupported",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }

        // --- Projected reasoning authority ---
        let has_think = self.tokenizer.special_token_id("<think>").is_some();
        let thinking_enabled_opt = msg.get("thinking_enabled").and_then(|v| v.as_bool());
        let assistant_prefix_opt = msg.get("assistant_prefix").and_then(|v| v.as_str());
        if max_think_tokens == Some(1)
            && (thinking_enabled_opt == Some(true) || assistant_prefix_opt == Some("open_think"))
        {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                "max_think_tokens=1 contradicts enabled/open-think framing",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }
        // Validate assistant_prefix values
        let assistant_prefix_valid = matches!(
            assistant_prefix_opt,
            None | Some("open_think") | Some("closed_think") | Some("plain")
        );
        if !assistant_prefix_valid {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                "assistant_prefix must be open_think|closed_think|plain",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }
        // Derive enable_thinking and expected prefix according to authority.
        // thinking_enabled and assistant_prefix are authority; max_think_tokens is fallback only.
        let (enable_thinking, expected_prefix): (bool, AssistantPrefix) =
            if let Some(enabled) = thinking_enabled_opt {
                let exp = if enabled {
                    if !has_think {
                        // No think token but thinking requested -> contradiction fail closed
                        hipfire_engine::emit::emit_active_attempt_error(
                            stdout,
                            Some(id),
                            "thinking_enabled true but model has no <think> token",
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        return Ok(());
                    }
                    AssistantPrefix::OpenThink
                } else if has_think {
                    AssistantPrefix::ClosedThink
                } else {
                    AssistantPrefix::Plain
                };
                // Validate assistant_prefix if present
                if let Some(ap_str) = assistant_prefix_opt {
                    let actual = match ap_str {
                        "open_think" => AssistantPrefix::OpenThink,
                        "closed_think" => AssistantPrefix::ClosedThink,
                        _ => AssistantPrefix::Plain,
                    };
                    if actual != exp {
                        hipfire_engine::emit::emit_active_attempt_error(
                            stdout,
                            Some(id),
                            &format!(
                                "thinking_enabled {enabled} contradicts assistant_prefix {ap_str}"
                            ),
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        return Ok(());
                    }
                }
                (enabled, exp)
            } else if let Some(ap_str) = assistant_prefix_opt {
                let actual = match ap_str {
                    "open_think" => AssistantPrefix::OpenThink,
                    "closed_think" => AssistantPrefix::ClosedThink,
                    _ => AssistantPrefix::Plain,
                };
                // assistant_prefix is authority when thinking_enabled absent
                let enabled = match actual {
                    AssistantPrefix::OpenThink => {
                        if !has_think {
                            hipfire_engine::emit::emit_active_attempt_error(
                                stdout,
                                Some(id),
                                "assistant_prefix open_think but model has no <think> token",
                                "validation",
                                false,
                                false,
                            );
                            let _ = stdout.flush();
                            return Ok(());
                        }
                        true
                    }
                    AssistantPrefix::ClosedThink => false,
                    AssistantPrefix::Plain => false,
                };
                // Also validate against max_think fallback if present and contradictory?
                // max_think is fallback only, so when assistant_prefix present we ignore it, but if max_think==1 vs open_think contradiction we still prefer authority and ignore fallback.
                (enabled, actual)
            } else {
                // Fallback to max_think_tokens compatibility
                let enabled = has_think && max_think_tokens != Some(1);
                let exp = if enabled {
                    AssistantPrefix::OpenThink
                } else if has_think {
                    AssistantPrefix::ClosedThink
                } else {
                    AssistantPrefix::Plain
                };
                (enabled, exp)
            };
        // Additional contradiction: if max_think==1 but thinking_enabled true? Already handled as authority wins, but spec says reject contradictions. If fallback path derived enabled true but max_think==1 would have been false, but since thinking_enabled absent we use max_think, so not contradictory.
        // If has_think false, ensure plain regardless.

        // Build prompt/convo/continuation using CLI slots semantics with system+name identity.
        let (system, turns, last_user) = match project_messages(msg) {
            Ok(v) => v,
            Err(e) => {
                hipfire_engine::emit::emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &e,
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return Ok(());
            }
        };
        // Build convo with system hash + name-aware user hashes
        let convo = if let Some(arr) = msg.get("messages").and_then(|v| v.as_array()) {
            let mut msgs: Vec<Message> = Vec::new();
            for m in arr {
                let role = match m.get("role").and_then(|v| v.as_str()) {
                    Some("system") => Role::System,
                    Some("assistant") => Role::Assistant,
                    Some("tool") => Role::User,
                    _ => Role::User,
                };
                let content = m
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned();
                let name = m.get("name").and_then(|v| v.as_str()).map(|s| s.to_owned());
                msgs.push(Message {
                    role,
                    content,
                    reasoning_content: None,
                    name: name.clone(),
                    rendered_name: None,
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    tool_plan: String::new(),
                });
            }
            build_convo_from_messages(&msgs)
        } else {
            // Fallback prompt-only path: single user is last_user, no history turns
            let mut c = Vec::new();
            c.push(system_hash(system.as_deref()));
            c.push(turn_hash_named(None, &last_user));
            c
        };

        let (prompt_tokens, started_in_think) =
            if let Some(template) = self.chat_template.as_deref() {
                let messages = match project_jinja_messages(msg) {
                    Ok(messages) => messages,
                    Err(reason) => {
                        hipfire_engine::emit::emit_active_attempt_error(
                            stdout,
                            Some(id),
                            &reason,
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        return Ok(());
                    }
                };
                let frame = JinjaChatFrame {
                    tokenizer: &self.tokenizer,
                    template,
                    system: None,
                    user: "",
                    enable_thinking,
                    bos_token: None,
                    reasoning_strength: None,
                    reasoning_effort: None,
                };
                let rendered = match frame.render_messages(&messages, None, None) {
                    Ok(rendered) => rendered,
                    Err(reason) => {
                        hipfire_engine::emit::emit_active_attempt_error(
                            stdout,
                            Some(id),
                            &format!("multi_slot Jinja render failed: {reason}"),
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        return Ok(());
                    }
                };
                let started = rendered.trim_end().ends_with("<think>");
                (self.tokenizer.encode(&rendered), started)
            } else {
                let history: Vec<(Role, &str)> = turns
                    .iter()
                    .map(|(role, text)| (*role, text.as_str()))
                    .collect();
                let frame = ChatFrame {
                    tokenizer: &self.tokenizer,
                    system: system.as_deref(),
                    user: &last_user,
                    assistant_prefix: expected_prefix,
                    raw: false,
                };
                (
                    frame.build_multi_turn(&history),
                    matches!(expected_prefix, AssistantPrefix::OpenThink),
                )
            };
        // Now prefix for continuation is expected_prefix but also must agree with started_in_think
        let prefix = if started_in_think {
            AssistantPrefix::OpenThink
        } else if has_think {
            AssistantPrefix::ClosedThink
        } else {
            AssistantPrefix::Plain
        };
        // Enforce that prefix matches expected_prefix and enable_thinking
        if prefix != expected_prefix {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                &format!(
                    "reasoning prefix mismatch: expected {:?} got {:?}",
                    expected_prefix, prefix
                ),
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }
        let prompt_len = prompt_tokens.len();
        let continuation = if convo.len() >= 3 {
            Continuation::UserTurn(hipfire_runtime::prompt_frame::continuation_suffix(
                &self.tokenizer,
                &last_user,
                prefix,
            ))
        } else {
            Continuation::Cold
        };

        // Emit gen_start before blocking on engine. Must agree with enable_thinking.
        emit_gen_start(
            stdout,
            id,
            started_in_think,
            Some(QWEN_AR_SEMANTIC_CONTRACT_VERSION),
        );
        if batch_check_abort(id, attempt_id) {
            batch_clear_terminal(id, attempt_id);
            emit_qwen_ar_cancelled(stdout, id, 0);
            return Ok(());
        }
        // Transition queued (stdin reader announced). Bind will happen after Accepted.
        let _ = batch_transition_to_queued(id, attempt_id);

        let (tx, rx) = mpsc::channel::<Event>();
        let req = SubmitRequest {
            prompt_tokens,
            convo,
            continuation,
            max_tokens,
            temperature,
            top_p,
            top_k: top_k as i32,
            seed,
            reply: tx,
        };
        if let Err(e) = self.engine.submit(req) {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("multi_slot submit: {e}"),
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }

        // Stream loop with keyed terminal binding
        let mut accepted_session: Option<u64> = None;
        let mut accepted_ticket: Option<LaneTicket> = None;
        let mut produced: usize = 0;
        let mut cached_tokens: usize = 0;
        let mut prefill_tokens: usize = prompt_len;
        let t_start = Instant::now();
        let think_mode = if started_in_think {
            ThinkMode::Low
        } else {
            ThinkMode::NonThink
        };
        // Validate think_mode agrees with enable_thinking
        let expected_think_mode = if enable_thinking {
            ThinkMode::Low
        } else {
            ThinkMode::NonThink
        };
        if think_mode != expected_think_mode {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                "think_mode mismatch with enable_thinking",
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            if let Some(sess) = accepted_session.take() {
                let _ = self.engine.close(sess);
            }
            return Ok(());
        }
        let mut emitter = Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &self.tokenizer,
            eos: self.tokenizer.eos_id,
            im_end: self.tokenizer.special_token_id("<|im_end|>"),
            tools: None,
            enable_grammar: false,
            stop: Vec::new(),
            max_think: 0,
            max_tokens,
            assistant_prefix: prefix,
            think_mode,
            decoded_vocab: None,
        });
        let render_events = |events: Vec<ClientEvent>, stdout: &mut W| -> Result<(), String> {
            for event in events {
                match event {
                    ClientEvent::Token(text) => emit_visible_token(stdout, id, &text),
                    ClientEvent::Reasoning(text) => emit_reasoning_token(stdout, id, &text),
                    ClientEvent::Committed { .. } => {}
                    ClientEvent::ToolCalls(_) => {
                        return Err(
                            "model emitted tool calls on a tool-disabled multi-slot request"
                                .to_string(),
                        )
                    }
                }
            }
            Ok(())
        };
        let mut first_token = true;

        let mut done_reason: Option<(DoneReason, usize)> = None;
        let mut rejected: Option<String> = None;

        loop {
            if batch_check_abort(id, attempt_id) {
                drop(rx);
                if let Some(sess) = accepted_session.take() {
                    let _ = self.engine.close(sess);
                }
                emit_qwen_ar_cancelled(stdout, id, produced);
                return Ok(());
            }
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(ev) => match ev {
                    Event::Accepted {
                        session,
                        reused,
                        prefill,
                    } => {
                        accepted_session = Some(session);
                        cached_tokens = reused;
                        prefill_tokens = prefill;
                        let ticket = LaneTicket {
                            lane: (session % 1024) as usize,
                            generation: attempt_id,
                        };
                        accepted_ticket = Some(ticket);
                        let _ = batch_bind_active(id, attempt_id, ticket);
                    }
                    Event::Token { id: tok_id } => {
                        let outcome = if first_token {
                            first_token = false;
                            emitter.begin(tok_id)
                        } else {
                            emitter.observe(tok_id)
                        };
                        if let Err(reason) = render_events(outcome.events, stdout) {
                            rejected = Some(reason);
                            break;
                        }
                        produced += 1;
                    }
                    Event::Done { reason, generated } => {
                        done_reason = Some((reason, generated));
                        break;
                    }
                    Event::Rejected { reason } => {
                        rejected = Some(reason);
                        break;
                    }
                },
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }

        if let Some(reason) = rejected {
            drop(rx);
            if let Some(sess) = accepted_session.take() {
                let _ = self.engine.close(sess);
            }
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("multi_slot rejected: {reason}"),
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }
        let summary = emitter.finish();
        if let Err(reason) = render_events(summary.events, stdout) {
            if let Some(sess) = accepted_session.take() {
                let _ = self.engine.close(sess);
            }
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                &reason,
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }

        let (reason, generated) = done_reason.unwrap_or((DoneReason::Eos, produced));
        if matches!(reason, DoneReason::ClientGone) {
            if let Some(sess) = accepted_session.take() {
                let _ = self.engine.close(sess);
            }
            emit_qwen_ar_cancelled(stdout, id, produced);
            return Ok(());
        }

        // Stage normal done payload and await commit via keyed registry.
        let finish_reason = match reason {
            DoneReason::MaxTokens => "length",
            _ => "stop",
        };
        let total_s = t_start.elapsed().as_secs_f64();
        let tok_s = if total_s > 0.0 {
            produced as f64 / total_s
        } else {
            0.0
        };
        let pending_done = serde_json::json!({
            "type": "done",
            "id": id,
            "attempt_id": attempt_id,
            "finish_reason": finish_reason,
            "tokens": generated,
            "prompt_tokens": cached_tokens + prefill_tokens,
            "cached_tokens": cached_tokens,
            "prefill_tokens": prefill_tokens,
            "tok_s": (tok_s * 10.0).round() / 10.0,
        });

        // Mark ready with exact pending done, publish commit_ready, poll keyed Commit/Abort with normal timeout
        let ticket = accepted_ticket.unwrap_or(LaneTicket {
            lane: 0,
            generation: attempt_id,
        });
        // If no session accepted, we still need to handle terminal: directly emit cancelled? But we have pending_done
        if accepted_session.is_some() {
            let _ = batch_mark_ready_with_pending(id, attempt_id, ticket, pending_done.clone());
        } else {
            // No session: treat as ready with dummy ticket to allow commit wait? Just emit done directly
            let _ = batch_mark_ready_with_pending(id, attempt_id, ticket, pending_done.clone());
        }
        let mut commit_ready = pending_done.clone();
        if let Some(map) = commit_ready.as_object_mut() {
            map.insert(
                "type".to_string(),
                serde_json::Value::String("commit_ready".to_string()),
            );
        }
        let _ = writeln!(stdout, "{}", commit_ready);
        let _ = stdout.flush();

        let decision = batch_wait_decision(id, attempt_id, CLIENT_TERMINAL_COMMIT_TIMEOUT);
        match decision {
            hipfire_engine::terminal::ClientTerminalDecision::Commit => {
                // Emit byte-identical done
                let _ = writeln!(stdout, "{}", pending_done);
                let _ = stdout.flush();
            }
            hipfire_engine::terminal::ClientTerminalDecision::Abort => {
                if let Some(sess) = accepted_session.take() {
                    let _ = self.engine.close(sess);
                }
                emit_qwen_ar_cancelled(stdout, id, produced);
            }
        }
        Ok(())
    }
}

struct SlotGuard<'a> {
    backend: &'a SlotBackend,
}
impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        self.backend.active.fetch_sub(1, Ordering::SeqCst);
    }
}

// ── CPU preflight ────────────────────────────────────────────────────────────

struct Preflight {
    arch_str: String,
    dim: usize,
    layers: usize,
    vocab: usize,
    tokenizer: Tokenizer,
    chat_template: Option<String>,
}

fn cpu_preflight(model_path: &str) -> Result<Preflight, String> {
    let hfq =
        HfqFile::open(std::path::Path::new(model_path)).map_err(|e| format!("open model: {e}"))?;
    validate_arch_id(hfq.arch_id)?;
    if is_vision_hfq(&hfq) {
        return Err("vision model not supported in experimental multi-slot".to_string());
    }
    let config = hipfire_arch_qwen35::qwen35::config_from_hfq(&hfq)
        .map_err(|e| format!("qwen config: {e}"))?;
    let tokenizer =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| format!("tokenizer: {e}"))?;
    let chat_template = hfq.chat_template();
    if chat_template
        .as_deref()
        .is_some_and(|template| template.trim().is_empty())
    {
        return Err("empty chat_template".to_string());
    }
    let arch_str = match hfq.arch_id {
        5 => "qwen3_5".to_string(),
        6 => "qwen3_5_moe".to_string(),
        _ => unreachable!(),
    };
    Ok(Preflight {
        arch_str,
        dim: config.dim,
        layers: config.n_layers,
        vocab: config.vocab_size,
        tokenizer,
        chat_template,
    })
}

// ── Validation helpers (pure) ────────────────────────────────────────────────

pub fn validate_arch_id(arch_id: u32) -> Result<(), String> {
    if matches!(arch_id, 5 | 6) {
        Ok(())
    } else {
        Err(format!(
            "experimental multi-slot only supports arch_id 5|6 (qwen3_5), got {arch_id}"
        ))
    }
}

pub fn is_vision_hfq(hfq: &HfqFile) -> bool {
    hfq.tensor_data("model.visual.patch_embed.proj.weight")
        .is_some()
}

pub fn validate_load_caps(msg: &serde_json::Value) -> Option<String> {
    // continuous_batch_size
    if let Some(v) = msg
        .get("params")
        .and_then(|p| p.get("continuous_batch_size"))
        .and_then(|v| v.as_u64())
    {
        if v > 1 {
            return Some(
                "experimental_multi_slot and continuous_batch_size>1 are mutually exclusive"
                    .to_string(),
            );
        }
    }
    if let Some(v) = msg.get("continuous_batch_size").and_then(|v| v.as_u64()) {
        if v > 1 {
            return Some(
                "experimental_multi_slot and continuous_batch_size>1 are mutually exclusive"
                    .to_string(),
            );
        }
    }
    let tp = msg
        .get("params")
        .and_then(|p| p.get("tp"))
        .and_then(|v| v.as_u64())
        .unwrap_or(1);
    let pp = msg
        .get("params")
        .and_then(|p| p.get("pp"))
        .and_then(|v| v.as_u64())
        .unwrap_or(1);
    if tp != 1 || pp != 1 {
        return Some("experimental multi-slot requires pp=tp=1".to_string());
    }
    // The slot kernels currently own a fixed Q8 KV/state path and no
    // speculative or eviction sidecars. Refuse instead of silently ignoring
    // an ordinary serve configuration that the alternate backend cannot honor.
    let params = msg.get("params");
    if params
        .and_then(|p| p.get("draft"))
        .and_then(|v| v.as_str())
        .is_some_and(|s| !s.is_empty())
    {
        return Some("draft not supported in experimental multi-slot".to_string());
    }
    if params
        .and_then(|p| p.get("drafter"))
        .and_then(|v| v.as_str())
        .is_some_and(|s| !s.is_empty())
    {
        return Some("spec/drafter not supported in experimental multi-slot".to_string());
    }
    for (key, label) in [
        ("dflash_mode", "DFlash"),
        ("mtp_mode", "MTP"),
        ("prefill_compression", "PFlash"),
    ] {
        if params
            .and_then(|p| p.get(key))
            .and_then(|v| v.as_str())
            .is_some_and(|v| !v.is_empty() && v != "off")
        {
            return Some(format!("{label} not supported in experimental multi-slot"));
        }
    }
    if params
        .and_then(|p| p.get("ngram_draft"))
        .and_then(|v| v.as_bool())
        == Some(true)
    {
        return Some("n-gram speculation not supported in experimental multi-slot".to_string());
    }
    if params.and_then(|p| p.get("cask")).and_then(|v| v.as_bool()) == Some(true)
        || params
            .and_then(|p| p.get("cask_sidecar"))
            .and_then(|v| v.as_str())
            .is_some_and(|v| !v.is_empty())
    {
        return Some("CASK not supported in experimental multi-slot".to_string());
    }
    if params
        .and_then(|p| p.get("kv_adaptive"))
        .and_then(|v| v.as_str())
        .is_some_and(|v| !v.is_empty() && v != "off")
    {
        return Some("adaptive KV not supported in experimental multi-slot".to_string());
    }
    if params
        .and_then(|p| p.get("kv_mode"))
        .and_then(|v| v.as_str())
        .is_some_and(|v| !v.is_empty() && v != "q8")
    {
        return Some("experimental multi-slot currently requires kv_mode=q8".to_string());
    }
    if params
        .and_then(|p| p.get("kv_backend"))
        .and_then(|v| v.as_str())
        .is_some_and(|v| !v.is_empty() && v != "contiguous")
    {
        return Some(
            "experimental multi-slot uses fixed slot arenas and requires kv_backend=contiguous"
                .to_string(),
        );
    }
    None
}

pub fn is_experimental_generate(msg: &serde_json::Value) -> bool {
    if msg.get("experimental_multi_slot").and_then(|v| v.as_bool()) == Some(true) {
        return true;
    }
    if msg
        .get("params")
        .and_then(|p| p.get("experimental_multi_slot"))
        .and_then(|v| v.as_bool())
        == Some(true)
    {
        return true;
    }
    false
}

pub fn validate_generate_caps(msg: &serde_json::Value) -> Option<String> {
    if msg.get("image").is_some_and(|v| !v.is_null())
        || msg.get("image_base64").is_some_and(|v| !v.is_null())
        || msg
            .get("messages")
            .and_then(|v| v.as_array())
            .is_some_and(|messages| {
                messages.iter().any(|message| {
                    message.get("role").and_then(|v| v.as_str()) == Some("tool")
                        || message
                            .get("tool_calls")
                            .and_then(|v| v.as_array())
                            .is_some_and(|calls| !calls.is_empty())
                        || message
                            .get("content")
                            .and_then(|v| v.as_array())
                            .is_some_and(|parts| {
                                parts.iter().any(|part| {
                                    part.get("type").and_then(|v| v.as_str()) == Some("image_url")
                                })
                            })
                })
            })
    {
        return Some(
            "images and tool-result roles are not supported in experimental multi-slot".to_string(),
        );
    }
    if msg
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|tools| !tools.is_empty())
    {
        return Some("tools not supported in experimental multi-slot".to_string());
    }
    if msg.get("stop").is_some_and(|v| !v.is_null()) {
        return Some("custom stop not supported in experimental multi-slot".to_string());
    }
    if msg.get("logprobs").and_then(|v| v.as_bool()) == Some(true)
        || msg.get("top_logprobs").is_some_and(|v| !v.is_null())
    {
        return Some("logprobs not supported in experimental multi-slot".to_string());
    }
    for (key, neutral) in [
        ("repeat_penalty", 1.0),
        ("presence_penalty", 0.0),
        ("frequency_penalty", 0.0),
        ("min_p", 0.0),
    ] {
        if msg
            .get(key)
            .and_then(|v| v.as_f64())
            .is_some_and(|v| v != neutral)
        {
            return Some(format!(
                "non-neutral {key} not supported in experimental multi-slot"
            ));
        }
    }
    if msg
        .get("reasoning_effort")
        .and_then(|v| v.as_str())
        .is_some_and(|v| !v.eq_ignore_ascii_case("none"))
    {
        return Some("reasoning_effort is not supported in experimental multi-slot".to_string());
    }
    if msg
        .get("max_think_tokens")
        .and_then(|v| v.as_u64())
        .is_some_and(|n| n >= 2)
    {
        return Some(
            "finite reasoning caps not supported in experimental multi-slot (max_think_tokens >=2)"
                .to_string(),
        );
    }
    None
}

// ── Message projection (pure) ────────────────────────────────────────────────

/// FNV-1a 64 hash of user turn text.
pub fn turn_hash(s: &str) -> u64 {
    turn_hash_named(None, s)
}

/// Hash with optional name for conversation identity.
pub fn turn_hash_named(name: Option<&str>, content: &str) -> u64 {
    let mut h = 0xcbf29ce484222325_u64;
    if let Some(n) = name {
        for b in n.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0xFF;
        h = h.wrapping_mul(0x100000001b3);
    }
    for b in content.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Tagged hash of all system messages. Prepended to convo to make system differences cold-miss.
pub fn system_hash(system: Option<&str>) -> u64 {
    let mut h = 0xcbf29ce484222325_u64;
    for b in b"system:" {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    if let Some(s) = system {
        for b in s.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Project messages/prompt into (system, turns, last_user) per CLI slots semantics.
pub fn project_messages(
    msg: &serde_json::Value,
) -> Result<(Option<String>, Vec<(Role, String)>, String), String> {
    if let Some(arr) = msg.get("messages").and_then(|v| v.as_array()) {
        let mut system: Option<String> = None;
        let mut turns: Vec<(Role, String)> = Vec::new();
        for m in arr {
            let text = m
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();
            match m.get("role").and_then(|v| v.as_str()) {
                Some("system") => match system.as_mut() {
                    Some(existing) => {
                        existing.push('\n');
                        existing.push_str(&text);
                    }
                    None => system = Some(text),
                },
                Some("assistant") => turns.push((Role::Assistant, text)),
                Some("tool") => turns.push((Role::User, text)),
                _ => turns.push((Role::User, text)),
            }
        }
        let last_user = match turns.iter().rposition(|(r, _)| *r == Role::User) {
            Some(i) => turns.remove(i).1,
            None => return Err("messages must contain at least one user message".to_string()),
        };
        Ok((system, turns, last_user))
    } else {
        // Fallback to prompt + optional system
        let prompt = msg
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("Hello")
            .to_owned();
        let system = msg
            .get("system")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned());
        Ok((system, Vec::new(), prompt))
    }
}
fn project_jinja_messages(msg: &serde_json::Value) -> Result<Vec<Message>, String> {
    let owned_fallback;
    let messages = match msg.get("messages").and_then(|value| value.as_array()) {
        Some(messages) => messages.as_slice(),
        None => {
            owned_fallback = vec![serde_json::json!({
                "role": "user",
                "content": msg
                    .get("prompt")
                    .and_then(|value| value.as_str())
                    .unwrap_or("Hello"),
            })];
            owned_fallback.as_slice()
        }
    };
    let mut projected = Vec::with_capacity(messages.len());
    for message in messages {
        let role = match message.get("role").and_then(|value| value.as_str()) {
            Some("system") => Role::System,
            Some("assistant") => Role::Assistant,
            Some("user") | None => Role::User,
            Some(other) => {
                return Err(format!(
                    "message role {other:?} is not supported in experimental multi-slot"
                ))
            }
        };
        projected.push(Message {
            role,
            content: message
                .get("content")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .to_owned(),
            reasoning_content: message
                .get("reasoning_content")
                .and_then(|value| value.as_str())
                .map(str::to_owned),
            name: message
                .get("name")
                .and_then(|value| value.as_str())
                .map(str::to_owned),
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        });
    }
    if !projected.iter().any(|message| message.role == Role::User) {
        return Err("messages must contain at least one user message".to_string());
    }
    Ok(projected)
}

/// Build convo hashes from turns + last_user (user turns only) with system and name identity.
/// Prepends tagged system hash; each user hash includes name.
pub fn build_convo(turns: &[(Role, String)], last_user: &str) -> Vec<u64> {
    build_convo_with_system(None, turns, last_user, None)
}

pub fn build_convo_with_system(
    system: Option<&str>,
    turns: &[(Role, String)],
    last_user: &str,
    last_user_name: Option<&str>,
) -> Vec<u64> {
    let mut convo: Vec<u64> =
        Vec::with_capacity(turns.iter().filter(|(r, _)| *r == Role::User).count() + 2);
    convo.push(system_hash(system));
    for (r, t) in turns {
        if *r == Role::User {
            convo.push(turn_hash_named(None, t));
        }
    }
    convo.push(turn_hash_named(last_user_name, last_user));
    convo
}

/// Build convo directly from Message slice (includes names and system).
pub fn build_convo_from_messages(messages: &[Message]) -> Vec<u64> {
    let system = messages
        .iter()
        .find(|m| m.role == Role::System)
        .map(|m| m.content.as_str());
    let mut convo = Vec::new();
    convo.push(system_hash(system));
    for m in messages {
        if m.role == Role::User {
            convo.push(turn_hash_named(m.name.as_deref(), &m.content));
        }
    }
    convo
}

// ── Tests (pure, no GPU) ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::prompt_frame::Role;
    use serde_json::json;

    #[test]
    fn arch_validation_accepts_5_and_6() {
        assert!(validate_arch_id(5).is_ok());
        assert!(validate_arch_id(6).is_ok());
        assert!(validate_arch_id(7).is_err());
        assert!(validate_arch_id(9).is_err());
    }

    #[test]
    fn generate_caps_rejects_images() {
        let m = json!({"image": "/tmp/a.png", "experimental_multi_slot": true});
        assert!(validate_generate_caps(&m).is_some());
        let m2 = json!({"image_base64": "abcd", "experimental_multi_slot": true});
        assert!(validate_generate_caps(&m2).is_some());
    }

    #[test]
    fn generate_caps_rejects_tools_and_stop_and_logprobs() {
        let m = json!({"tools": [{"type":"function"}], "experimental_multi_slot": true});
        assert!(validate_generate_caps(&m).is_some());
        let m2 = json!({"stop": ["END"], "experimental_multi_slot": true});
        assert!(validate_generate_caps(&m2).is_some());
        let m3 = json!({"logprobs": true, "experimental_multi_slot": true});
        assert!(validate_generate_caps(&m3).is_some());
        let m4 = json!({"max_think_tokens": 5, "experimental_multi_slot": true});
        assert!(validate_generate_caps(&m4).is_some());
        let m5 = json!({"max_think_tokens": 1, "experimental_multi_slot": true});
        assert!(validate_generate_caps(&m5).is_none());
    }

    #[test]
    fn load_caps_rejects_continuous_and_tp_pp() {
        let m = json!({"params": {"continuous_batch_size": 2}});
        assert!(validate_load_caps(&m).is_some());
        let m2 = json!({"params": {"tp": 2}});
        assert!(validate_load_caps(&m2).is_some());
        let m3 = json!({"params": {"pp": 2}});
        assert!(validate_load_caps(&m3).is_some());
        let m4 = json!({"params": {"draft": "some.hfq"}});
        assert!(validate_load_caps(&m4).is_some());
        let m5 = json!({"params": {"prefill_compression": "on"}});
        assert!(validate_load_caps(&m5).is_some());
    }

    #[test]
    fn turn_hash_is_fnv1a() {
        // Known FNV-1a for "a" is 0xaf63dc4c8601ec8c etc. Just test determinism.
        assert_ne!(turn_hash("hello"), turn_hash("hello "));
        assert_eq!(turn_hash("test"), turn_hash("test"));
    }

    #[test]
    fn turn_hash_name_changes_identity() {
        assert_ne!(
            turn_hash_named(Some("alice"), "hello"),
            turn_hash_named(None, "hello")
        );
        assert_ne!(
            turn_hash_named(Some("alice"), "hello"),
            turn_hash_named(Some("bob"), "hello")
        );
    }

    #[test]
    fn system_hash_differs() {
        assert_ne!(system_hash(Some("sys1")), system_hash(Some("sys2")));
        assert_ne!(system_hash(None), system_hash(Some("sys")));
    }

    #[test]
    fn project_messages_builds_convo() {
        let msg = json!({
            "messages": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "follow up"}
            ]
        });
        let (system, turns, last) = project_messages(&msg).unwrap();
        assert_eq!(system.as_deref(), Some("you are helpful"));
        assert_eq!(turns.len(), 2); // hi -> assistant hello, then follow up removed as last
        assert_eq!(turns[0].0, Role::User);
        assert_eq!(turns[0].1, "hi");
        assert_eq!(turns[1].0, Role::Assistant);
        assert_eq!(last, "follow up");
        let convo = build_convo_with_system(system.as_deref(), &turns, &last, None);
        // system hash + 2 user hashes
        assert_eq!(convo.len(), 3);
        assert_eq!(convo[0], system_hash(Some("you are helpful")));
        assert_eq!(convo[1], turn_hash("hi"));
        assert_eq!(convo[2], turn_hash("follow up"));
    }

    #[test]
    fn project_messages_requires_user() {
        let msg = json!({"messages": [{"role": "assistant", "content": "hi"}]});
        assert!(project_messages(&msg).is_err());
    }

    #[test]
    fn project_messages_fallback_prompt() {
        let msg = json!({"prompt": "Hello", "system": "sys"});
        let (system, turns, last) = project_messages(&msg).unwrap();
        assert_eq!(system.as_deref(), Some("sys"));
        assert!(turns.is_empty());
        assert_eq!(last, "Hello");
        let convo = build_convo_with_system(system.as_deref(), &turns, &last, None);
        assert_eq!(convo[0], system_hash(Some("sys")));
    }
    #[test]
    fn jinja_projection_preserves_supported_message_roles() {
        let messages = project_jinja_messages(&json!({
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "reason"
                }
            ]
        }))
        .unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[2].role, Role::Assistant);
        assert_eq!(messages[2].reasoning_content.as_deref(), Some("reason"));
        assert!(project_jinja_messages(&json!({
            "messages": [{"role": "tool", "content": "result"}]
        }))
        .is_err());
    }

    #[test]
    fn experimental_load_requires_its_fixed_q8_non_speculative_contract() {
        let supported = json!({"params": {
            "continuous_batch_size": 1,
            "tp": 1,
            "pp": 1,
            "kv_mode": "q8",
            "kv_backend": "contiguous",
            "kv_adaptive": "off",
            "dflash_mode": "off",
            "mtp_mode": "off",
            "prefill_compression": "off",
            "ngram_draft": false,
            "cask": false
        }});
        assert_eq!(validate_load_caps(&supported), None);
        for params in [
            json!({"kv_mode": "asym3"}),
            json!({"kv_backend": "vmm"}),
            json!({"dflash_mode": "auto"}),
            json!({"mtp_mode": "on"}),
            json!({"ngram_draft": true}),
            json!({"cask": true}),
        ] {
            assert!(
                validate_load_caps(&json!({"params": params})).is_some(),
                "unsupported params passed: {params}"
            );
        }
    }

    #[test]
    fn generate_caps_rejects_silently_ignored_controls() {
        assert!(validate_generate_caps(&json!({"temperature": 0.7, "top_p": 0.9})).is_none());
        for request in [
            json!({"repeat_penalty": 1.05}),
            json!({"presence_penalty": 0.1}),
            json!({"min_p": 0.05}),
            json!({"reasoning_effort": "high"}),
            json!({"messages": [{"role": "tool", "content": "result"}]}),
            json!({"messages": [{
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_0"}]
            }]}),
        ] {
            assert!(
                validate_generate_caps(&request).is_some(),
                "unsupported request passed: {request}"
            );
        }
    }

    #[test]
    fn build_convo_from_messages_includes_names_and_system() {
        let msgs = vec![
            Message {
                role: Role::System,
                content: "sys".to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::User,
                content: "hi".to_string(),
                reasoning_content: None,
                name: Some("alice".to_string()),
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
        ];
        let c1 = build_convo_from_messages(&msgs);
        let c2 = build_convo_from_messages(&[
            Message {
                role: Role::System,
                content: "sys".to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::User,
                content: "hi".to_string(),
                reasoning_content: None,
                name: Some("bob".to_string()),
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
        ]);
        assert_ne!(c1, c2);
    }
}
