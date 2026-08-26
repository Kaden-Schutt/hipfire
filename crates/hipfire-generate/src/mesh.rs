// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Device-mesh dense carrier serving (the `MeshCarrier` generation route).
//!
//! Admitted dense TP (`LoadedModel.tp_model`) and dense PP
//! (`LoadedModel.pp_model`) loads keep weights + KV inside the carrier —
//! `LoadedModel.state` is `None` for them, so no in-process dense route can
//! serve them. This module drives whichever carrier is present through a
//! minimal AR loop: render → prefill → host-sampled decode → emit.
//!
//! Ported verbatim from the pre-merge daemon's `generate_mesh_carrier`
//! (backup `crates/hipfire-runtime/examples/daemon.rs`), which replaced the
//! feature branch's `dense_serve_via_ar_generate` driver; the tp/pp parity
//! examples exercise the same carrier API.

use hipfire_engine::terminal::*;
use hipfire_loader::LoadedModel;
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message, Role};
use hipfire_runtime::sampler::{self, SamplerConfig};
use std::io::Write;
use std::time::Instant;

/// Unified serve surface for the device-mesh dense carriers (Tp axis:
/// `TpModel`, dense Pp axis: `PpModel`). Weights + KV live inside the carrier
/// (`LoadedModel.state` is None for these loads), so the in-process dense
/// routes cannot serve them. Inherent methods are called via fully-qualified
/// paths so the trait forwarders don't self-recurse.
trait MeshServed {
    fn forward_token(&mut self, token: u32, pos: usize) -> Result<(), String>;
    fn logits(&mut self) -> Result<Vec<f32>, String>;
    fn prefill(&mut self, tokens: &[u32]) -> Result<(), String>;
    fn eos_token(&self) -> u32;
}

impl MeshServed for hipfire_runtime::tp_serve::TpModel {
    fn forward_token(&mut self, token: u32, pos: usize) -> Result<(), String> {
        hipfire_runtime::tp_serve::TpModel::forward_token(self, token, pos)
    }
    fn logits(&mut self) -> Result<Vec<f32>, String> {
        hipfire_runtime::tp_serve::TpModel::logits(self)
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<(), String> {
        hipfire_runtime::tp_serve::TpModel::prefill(self, tokens)
    }
    fn eos_token(&self) -> u32 {
        hipfire_runtime::tp_serve::TpModel::eos_token(self)
    }
}

impl MeshServed for hipfire_runtime::pp_serve::PpModel {
    fn forward_token(&mut self, token: u32, pos: usize) -> Result<(), String> {
        hipfire_runtime::pp_serve::PpModel::forward_token(self, token, pos)
    }
    fn logits(&mut self) -> Result<Vec<f32>, String> {
        hipfire_runtime::pp_serve::PpModel::logits(self)
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<(), String> {
        hipfire_runtime::pp_serve::PpModel::prefill(self, tokens)
    }
    fn eos_token(&self) -> u32 {
        hipfire_runtime::pp_serve::PpModel::eos_token(self)
    }
}

/// Drive a device-mesh dense carrier (Tp axis `tp_model` / dense Pp axis
/// `pp_model`) through a minimal AR loop: render → prefill → host-sampled
/// decode → emit. Per-request full-context replay (no LCP reuse yet) —
/// correct, just slower on multi-turn; mirrors `generate_qwen2`'s contract
/// with `sampler::sample_cpu`.
#[allow(clippy::too_many_arguments)]
pub fn generate_mesh_carrier(
    m: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: Option<f32>,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[Message]>,
    stop: &[String],
) {
    let tokenizer = match m.tokenizer.as_ref() {
        Some(t) => t,
        None => {
            crate::dense::emit_active_attempt_error(
                stdout,
                Some(id),
                "tokenizer not loaded",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    };
    let carrier: &mut dyn MeshServed = if let Some(tp) = m.tp_model.as_mut() {
        tp
    } else if let Some(pp) = m.pp_model.as_mut() {
        pp
    } else {
        crate::dense::emit_active_attempt_error(
            stdout,
            Some(id),
            "mesh carrier missing on MeshCarrier route",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    };
    let capacity = m.physical_cap;

    // Canonical multi-turn render via the arch's chat template; falls back
    // to raw encode when no template is loaded. Mirrors generate_ep.
    let prompt_ids: Vec<u32> = if let Some(template) = m.chat_template.as_ref() {
        let frame = JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking: true,
            bos_token: None,
            reasoning_strength: None,
            reasoning_effort: None,
        };
        let render_result = if tools.is_some() || messages_history.is_some() {
            let synthesized: Vec<Message>;
            let messages_slice: &[Message] = match messages_history {
                Some(h) => h,
                None => {
                    let mut v = Vec::new();
                    if let Some(sys) = system_prompt {
                        v.push(Message {
                            role: Role::System,
                            content: sys.to_string(),
                            reasoning_content: None,
                            name: None,
                            rendered_name: None,
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                            tool_plan: String::new(),
                        });
                    }
                    v.push(Message {
                        role: Role::User,
                        content: prompt.to_string(),
                        reasoning_content: None,
                        name: None,
                        rendered_name: None,
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                        tool_plan: String::new(),
                    });
                    synthesized = v;
                    &synthesized
                }
            };
            frame.render_messages(messages_slice, tools, None)
        } else {
            frame.render()
        };
        match render_result {
            Ok(rendered) => tokenizer.encode(&rendered),
            Err(e) => {
                crate::dense::emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!("mesh render: {}", format!("{e}").replace('"', "'")),
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return;
            }
        }
    } else {
        tokenizer.encode(prompt)
    };
    if prompt_ids.is_empty() {
        crate::dense::emit_active_attempt_error(
            stdout,
            Some(id),
            "mesh: empty prompt after render",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    // Capacity guard: the carrier's KV is sized for max_seq at load; replay
    // the full prompt from position 0 each request (no LCP reuse yet), so
    // the absolute span is prompt + max_tokens.
    if prompt_ids.len().saturating_add(max_tokens) > capacity {
        crate::dense::emit_active_attempt_error(
            stdout,
            Some(id),
            &format!(
                "prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq",
                prompt_ids.len(),
                max_tokens,
                capacity
            ),
            "context_length",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    let t0 = Instant::now();
    // Full-context replay: clear the prior turn's conversation state.
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    if let Err(e) = carrier.prefill(&prompt_ids) {
        let message = format!("mesh prefill failed: {e}");
        crate::ar::write_error(stdout, id, &message);
        let _ = stdout.flush();
        return;
    }
    m.conversation_tokens.extend_from_slice(&prompt_ids);
    let prefill_ms = t0.elapsed().as_millis();

    let eos = carrier.eos_token();
    let mut generated: usize = 0;
    let mut text_acc = String::new();
    let decode_t0 = Instant::now();
    let mut pos = prompt_ids.len();
    let sampler_cfg = SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty,
        repeat_window,
        presence_penalty,
        frequency_penalty,
        blocked_tokens: Vec::new(),
        top_k,
        min_p,
    };
    let mut next_tok = match carrier.logits() {
        Ok(mut logits) => sampler::sample_cpu(&mut logits, &[], &sampler_cfg),
        Err(e) => {
            let message = format!("mesh logits (post-prefill) failed: {e}");
            crate::ar::write_error(stdout, id, &message);
            let _ = stdout.flush();
            return;
        }
    };

    loop {
        if generated >= max_tokens || next_tok == eos {
            break;
        }
        let frag = tokenizer.decode(&[next_tok]);
        text_acc.push_str(&frag);
        let _ = writeln!(
            stdout,
            r#"{{"type":"token","id":"{}","text":{},"attempt_id":{}}}"#,
            id,
            serde_json::to_string(&frag).unwrap_or_else(|_| "\"\"".to_string()),
            active_attempt_id()
        );
        let _ = stdout.flush();
        m.conversation_tokens.push(next_tok);
        generated += 1;
        if stop.iter().any(|s| !s.is_empty() && text_acc.ends_with(s)) {
            break;
        }
        if let Err(e) = carrier.forward_token(next_tok, pos) {
            let message = format!("mesh forward failed: {e}");
            crate::ar::write_error(stdout, id, &message);
            let _ = stdout.flush();
            return;
        }
        pos += 1;
        let scope = &m.conversation_tokens[prompt_ids.len()..];
        next_tok = match carrier.logits() {
            Ok(mut logits) => sampler::sample_cpu(&mut logits, scope, &sampler_cfg),
            Err(e) => {
                let message = format!("mesh logits failed: {e}");
                crate::ar::write_error(stdout, id, &message);
                let _ = stdout.flush();
                return;
            }
        };
    }

    m.seq_pos = pos;

    let decode_ms = decode_t0.elapsed().as_millis().max(1);
    let total_ms = t0.elapsed().as_millis().max(1);
    let tok_s = if generated > 0 && decode_ms > 0 {
        (generated as f64 * 1000.0) / decode_ms as f64
    } else {
        0.0
    };
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated,
        "tok_s": (tok_s * 100.0).round() / 100.0,
        "prefill_ms": prefill_ms,
        "total_ms": total_ms,
        "attempt_id": active_attempt_id(),
    });
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => emit_staged_terminal_done(stdout, &pending_done),
        ClientTerminalDecision::Abort => {}
    }
}
