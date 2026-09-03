// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5/3.6 family generation — extracted from daemon.rs (D3, wave 5).
//! Verbatim move of generate_multi, generate_ep, generate_spec, generate_dflash
//! plus their exclusive EP/cache helpers. No logic changes.

use base64::Engine;
use hipfire_arch_cohere2moe as cohere2moe;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_dots_ocr::dots_ocr;
use hipfire_arch_gemma4 as gemma4;
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState;
use hipfire_arch_lfm2moe::forward_batch::{
    forward_decode_batch_lfm, forward_decode_batch_prepared_lfm, prepare_decode_batch_inputs_lfm,
};
use hipfire_arch_minimax as minimax;
use hipfire_arch_muse_glimmer as glimmer;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative;
use hipfire_arch_qwen35_vl::image;
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_runtime::emit_text::{
    currently_in_think, extract_tool_calls_from_text, ThinkOutputRouter, ThinkRouteEvent,
    ToolOutputRouter, ToolRouteError, ToolRouteEvent,
};
use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction};
use hipfire_runtime::llama;
use hipfire_runtime::prompt_frame::ThinkMode;
use hipfire_runtime::sampler::{self, SamplerConfig};
use std::any::Any;
use std::io::{BufRead, Write};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use hipfire_engine::emit::*;
use hipfire_engine::prompt::*;
use hipfire_engine::redline::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_loader::{AsstTurnCache, EpArch, EpState, Eviction, LoadedModel};
use hipfire_runtime::spec::{
    ClientEvent, EmitOutcome, EvictRetain, FinishSummary, PrefillOutcome, SpecAdvance, SpecEmit,
    SpecRequestConfig, SpecTarget, Speculator, StopReason,
};

use crate::common::*;
use hipfire_pflash;
use hipfire_runtime::prompt_frame;
use rdna_compute;

/// Expert-parallel streaming generate (task #26, ds4 first). Greedy AR via
/// `forward_ep` across the EP ranks; logits gathered on rank 0 and sampled on
/// the host. v1: greedy + basic token streaming (no grammar / tool-calls /
/// think-budget — absent on the EP path). The DeepSeek chat template
/// (`<｜User｜>…<｜Assistant｜>`) is applied here; the daemon's full prompt-frame
/// (multi-turn, messages_history) is a follow-up. See docs/plans/daemon-ep-wiring.md.
#[allow(clippy::too_many_arguments)]
/// Resolved sampling config for the EP (multi-GPU) decode loops. Carries the
/// single-GPU handler's request>rec_*>arch-default resolution (computed at the
/// `generate` call site) into `ep_serve_ds4` / `ep_serve_minimax`, which apply
/// it host-side over the downloaded f32 logits via `llama::sample_full_dist`.
#[derive(Clone, Copy)]
pub struct EpSampling {
    pub temp: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
}

pub fn generate_ep(
    m: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
    max_think_tokens: usize,
    think_mode: ThinkMode,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
    stop: &[String],
    sampling: EpSampling,
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
) {
    // ── Canonical multi-turn render via the arch's trained chat_template
    // (ds4/minimax). Mirrors generate_minimax: `messages_history` (the full
    // conversation, live user last) → render_messages with `tools` threaded;
    // falls back to a synthesized [system?, user] turn when no history is
    // supplied. The trim_blocks/lstrip_blocks env (prompt_frame) keeps the
    // structural prefix history-invariant so the EP LCP cache below can hit.
    // `primed_think` records whether the render ended on the MiniMax `<think>`
    // generation primer (re-emitted display-only in ep_serve_minimax). ──
    let mut primed_think = false;
    let prompt_ids: Vec<u32> = match hipfire_loader::ep_prompt_route(m.arch_id) {
        hipfire_loader::EpPromptRoute::Dsml => {
            primed_think = false;
            let tokenizer = m.tokenizer.as_ref().unwrap();
            let eos_tok = m.deepseek4_eos_tok;
            build_deepseek4_dsml_prompt(
                tokenizer,
                system_prompt,
                tools,
                messages_history,
                prompt,
                think_mode,
                eos_tok,
                &mut m.asst_turn_cache,
            )
        }
        hipfire_loader::EpPromptRoute::Jinja => {
            let tokenizer = m.tokenizer.as_ref().unwrap();
            if let Some(template) = m.chat_template.as_ref() {
                let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
                    tokenizer,
                    template,
                    system: system_prompt,
                    user: prompt,
                    enable_thinking,
                    bos_token: None,
                    reasoning_strength: None,
                    reasoning_effort,
                };
                let render_result = if tools.is_some() || messages_history.is_some() {
                    let synthesized: Vec<hipfire_runtime::prompt_frame::Message>;
                    let messages_slice: &[hipfire_runtime::prompt_frame::Message] =
                        match messages_history {
                            Some(h) => h,
                            None => {
                                let mut v = Vec::new();
                                if let Some(sys) = system_prompt {
                                    v.push(hipfire_runtime::prompt_frame::Message {
                                        role: hipfire_runtime::prompt_frame::Role::System,
                                        content: sys.to_string(),
                                        reasoning_content: None,
                                        name: None,
                                        rendered_name: None,
                                        tool_calls: Vec::new(),
                                        tool_call_id: None,
                                        tool_plan: String::new(),
                                    });
                                }
                                v.push(hipfire_runtime::prompt_frame::Message {
                                    role: hipfire_runtime::prompt_frame::Role::User,
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
                    Ok(rendered) => {
                        primed_think = rendered.trim_end().ends_with("<think>");
                        tokenizer.encode(&rendered)
                    }
                    Err(e) => {
                        emit_active_attempt_error(
                            stdout,
                            Some(id),
                            &format!("EP jinja render: {}", format!("{e}").replace('"', "'")),
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        return;
                    }
                }
            } else {
                // No embedded template — minimal ds4-style fallback (single-turn).
                let mut ids = Vec::new();
                if let Some(b) = tokenizer.special_token_id("<｜begin▁of▁sentence｜>") {
                    ids.push(b);
                }
                ids.extend(tokenizer.encode(&format!("<｜User｜>{prompt}<｜Assistant｜>")));
                ids
            }
        }
    };
    if std::env::var("HIPFIRE_DEEPSEEK4_DUMP_PROMPT")
        .ok()
        .as_deref()
        == Some("1")
    {
        let tk = m.tokenizer.as_ref().unwrap();
        eprintln!(
            "[ep prompt dump] arch={} {} tokens, decoded:\n>>>\n{}\n<<<",
            m.arch_id,
            prompt_ids.len(),
            tk.decode(&prompt_ids)
        );
    }
    if prompt_ids.is_empty() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "EP: empty prompt after render",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    let eos_tok = match hipfire_loader::ep_eos_route(m.arch_id) {
        hipfire_loader::EpEosRoute::Minimax => {
            // MiniMax EP state lives in `m.ep`, not `m.state`, so `minimax()` is
            // None here — read the EP eos carried on LoadedModel (set at load).
            m.minimax_eos_tok
        }
        hipfire_loader::EpEosRoute::Qwen35 => m.qwen35_eos_tok,
        hipfire_loader::EpEosRoute::Deepseek4 => m.deepseek4_eos_tok,
    };
    match m.arch_id {
        10 => ep_serve_minimax(
            m,
            stdout,
            id,
            &prompt_ids,
            eos_tok,
            max_tokens,
            stop,
            primed_think,
            sampling,
        ),
        5 | 6 => ep_serve_qwen35_dense_tp(
            m,
            stdout,
            id,
            &prompt_ids,
            eos_tok,
            max_tokens,
            max_think_tokens,
            stop,
            primed_think,
            sampling,
        ),
        _ => ep_serve_ds4(
            m,
            stdout,
            id,
            &prompt_ids,
            eos_tok,
            max_tokens,
            think_mode,
            tools,
            stop,
            sampling,
        ),
    }
}

/// Stream a token JSON event; returns true if a stop sequence is now satisfied.
pub fn ep_emit_token(
    stdout: &mut std::io::Stdout,
    id: &str,
    piece: &str,
    text_acc: &mut String,
    stop: &[String],
) -> bool {
    text_acc.push_str(piece);
    let _ = writeln!(
        stdout,
        r#"{{"type":"token","id":"{}","text":{},"attempt_id":{}}}"#,
        id,
        serde_json::to_string(piece).unwrap_or_else(|_| "\"\"".to_string()),
        active_attempt_id()
    );
    let _ = stdout.flush();
    stop.iter().any(|s| !s.is_empty() && text_acc.ends_with(s))
}

/// Dense Qwen TP2..TP5 serving loop with batched tensor-parallel prefill and
/// Qwen contract-v2 semantic streaming.
#[allow(clippy::too_many_arguments)]
pub fn ep_serve_qwen35_dense_tp(
    m: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt_ids: &[u32],
    eos_tok: u32,
    max_tokens: usize,
    max_think_tokens: usize,
    stop: &[String],
    primed_think: bool,
    sampling: EpSampling,
) {
    let prompt_n = prompt_ids.len();
    if prompt_n.saturating_add(max_tokens) > m.physical_cap {
        emit_active_attempt_error(
            stdout,
            Some(id),
            &format!(
                "prompt exceeds context capacity: prompt={prompt_n} + max_tokens={max_tokens} > capacity={}",
                m.physical_cap
            ),
            "context_length",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    // This route replays the complete rendered conversation each request.
    if let Some(EpState { gpus, inner }) = m.ep.as_mut() {
        if let EpArch::Qwen35DenseTp { dn_states, .. } = inner {
            for (rank, state) in dn_states.iter_mut().enumerate() {
                let gpu = &mut gpus.devices[rank];
                if let Err(e) = gpu.bind_thread() {
                    emit_active_attempt_error(
                        stdout,
                        Some(id),
                        &format!("dense TP bind_thread rank {rank}: {e:?}"),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    return;
                }
                if let Err(e) = state.reset(gpu) {
                    emit_active_attempt_error(
                        stdout,
                        Some(id),
                        &format!("dense TP state reset rank {rank}: {e:?}"),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    return;
                }
                gpu.invalidate_graph_state();
            }
        }
    }
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    // `primed_think` preserves Jinja enable_thinking semantics (render ended on
    // an open `<think>` primer). Tool requests fail closed before this route.
    emit_gen_start(
        stdout,
        id,
        primed_think,
        gen_start_contract_version_for_arch(m.arch_id),
    );

    let t_prefill = Instant::now();
    for (chunk_index, chunk) in prompt_ids.chunks(32).enumerate() {
        if check_abort(id) {
            ep_emit_abort(stdout, id, m, 0);
            return;
        }
        let result = {
            let Some(EpState { gpus, inner }) = m.ep.as_mut() else {
                return;
            };
            let EpArch::Qwen35DenseTp {
                shard,
                configs,
                weights,
                kv_caches,
                dn_states,
                scratches,
            } = inner
            else {
                emit_active_attempt_error(
                    stdout,
                    Some(id),
                    "EP arch mismatch (expected dense Qwen TP)",
                    "validation",
                    false,
                    false,
                );
                return;
            };
            qwen35::forward_prefill_dense_tp(
                gpus,
                shard,
                weights,
                configs,
                chunk,
                chunk_index * 32,
                kv_caches,
                dn_states,
                scratches,
            )
        };
        if let Err(e) = result {
            emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("dense TP prefill: {e:?}"),
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    }
    // Bookkeeping matches the fully replayed prompt so decode commits extend
    // the same conversation/stream positions the single-GPU path would.
    m.conversation_tokens.extend_from_slice(prompt_ids);
    m.seq_pos = prompt_n;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    let mut logits = {
        let Some(EpState { gpus, inner }) = m.ep.as_mut() else {
            return;
        };
        let EpArch::Qwen35DenseTp { scratches, .. } = inner else {
            return;
        };
        if let Err(e) = gpus.devices[0].bind_thread() {
            emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("dense TP first-logits bind_thread: {e:?}"),
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
        match gpus.devices[0].download_f32(&scratches[0].logits) {
            Ok(v) => v,
            Err(e) => {
                emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!("dense TP first-logits download: {e:?}"),
                    "validation",
                    false,
                    false,
                );
                return;
            }
        }
    };

    let t_decode = Instant::now();
    let mut semantic =
        crate::ar::QwenArSemanticProducer::new_with_tool_protocol(id, primed_think, false);
    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut bytes_fed_to_filter = 0usize;
    let mut generated = 0usize;
    let mut think_count = 0usize;
    let mut hit_custom_stop = false;
    while generated < max_tokens {
        if check_abort(id) {
            ep_emit_abort(stdout, id, m, generated);
            return;
        }
        let next = llama::sample_full_dist(
            &logits,
            sampling.temp,
            sampling.top_p,
            sampling.top_k,
            sampling.min_p,
        );

        // KV write before any client-visible classify/emit (same contract as AR).
        let write_pos = m.seq_pos;
        let forward = {
            let Some(EpState { gpus, inner }) = m.ep.as_mut() else {
                return;
            };
            let EpArch::Qwen35DenseTp {
                shard,
                configs,
                weights,
                kv_caches,
                dn_states,
                scratches,
            } = inner
            else {
                return;
            };
            qwen35::forward_scratch_dense_tp(
                gpus, shard, weights, configs, next, write_pos, kv_caches, dn_states, scratches,
            )
        };
        if let Err(e) = forward {
            emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("dense TP decode: {e:?}"),
                "validation",
                false,
                false,
            );
            return;
        }

        let prev_fed = bytes_fed_to_filter;
        let elapsed_ms = t_decode.elapsed().as_millis() as u64;
        let filter_stop = match semantic.commit_and_classify(
            stdout,
            next,
            || {
                let pos = crate::ar::qwen_ar_raw_commit_token(
                    &mut m.conversation_tokens,
                    &mut streamed_tokens,
                    &mut m.seq_pos,
                    next,
                    crate::ar::QwenArRawCommitDisposition::ClassifiedVisible,
                );
                let all_bytes = m.tokenizer.as_ref().unwrap().decode_bytes(&streamed_tokens);
                let new_bytes = all_bytes[prev_fed.min(all_bytes.len())..].to_vec();
                bytes_fed_to_filter = all_bytes.len();
                (pos, new_bytes)
            },
            |pos, out| {
                emit_committed_event(out, id, next, pos, elapsed_ms);
            },
        ) {
            Ok(stop) => stop,
            Err(err) => {
                emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!("dense TP semantic classify: {err}"),
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return;
            }
        };
        generated += 1;

        // Custom stops match visible answer text (post EosFilter/think route),
        // never raw protocol bytes.
        if stop
            .iter()
            .any(|s| !s.is_empty() && semantic.visible().ends_with(s.as_str()))
        {
            hit_custom_stop = true;
            break;
        }

        // Conservative think-budget: count tokens while the router is inside a
        // think span. Exceeding a nonzero cap fails closed — no force-close
        // splice and no partial semantic done.
        if max_think_tokens > 0 {
            if semantic.think_router.in_think() {
                think_count = think_count.saturating_add(1);
                if think_count >= max_think_tokens {
                    let ep = ep_reset_after_abort(m);
                    emit_fail_closed_error(
                        stdout,
                        Some(id),
                        "think token budget exceeded (validation)",
                        "validation",
                        false,
                        &ep,
                    );
                    return;
                }
            } else {
                think_count = 0;
            }
        }

        if filter_stop || next == eos_tok || generated >= max_tokens {
            break;
        }

        logits = {
            let Some(EpState { gpus, inner }) = m.ep.as_mut() else {
                return;
            };
            let EpArch::Qwen35DenseTp { scratches, .. } = inner else {
                return;
            };
            if let Err(e) = gpus.devices[0].bind_thread() {
                emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!("dense TP decode logits bind_thread: {e:?}"),
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return;
            }
            match gpus.devices[0].download_f32(&scratches[0].logits) {
                Ok(v) => v,
                Err(e) => {
                    emit_active_attempt_error(
                        stdout,
                        Some(id),
                        &format!("dense TP decode logits download: {e:?}"),
                        "validation",
                        false,
                        false,
                    );
                    return;
                }
            }
        };
    }

    // Custom stop is a natural/filter-class terminal, not length.
    let hit_length_cap = generated >= max_tokens && !hit_custom_stop;
    let (finish, _visible) = match semantic.finish(stdout, hit_length_cap) {
        Ok(pair) => pair,
        Err(err) => {
            emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("dense TP semantic finish: {err}"),
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    };
    if matches!(finish.cause, crate::ar::QwenArTerminalCause::OpenThink) {
        let ep = ep_reset_after_abort(m);
        crate::ar::emit_qwen_ar_open_think_terminal(stdout, id, generated, &ep);
        return;
    }
    let finish_reason = match finish.finish_reason {
        "length" => "length",
        "error" => "error",
        _ => "stop",
    };
    ep_emit_done(
        stdout,
        id,
        m,
        generated,
        prompt_n,
        prefill_ms,
        t_decode.elapsed().as_secs_f64() * 1000.0,
        finish_reason,
    );
}

pub fn ep_emit_done(
    stdout: &mut std::io::Stdout,
    id: &str,
    m: &mut LoadedModel,
    generated: usize,
    prompt_n: usize,
    prefill_ms: f64,
    decode_ms: f64,
    finish_reason: &str,
) {
    let decode_tok_s = if decode_ms > 0.0 {
        generated as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };
    let prefill_tok_s = if prefill_ms > 0.0 {
        prompt_n as f64 / (prefill_ms / 1000.0)
    } else {
        0.0
    };
    tracing::info!(
        request_id = id,
        generated_tokens = generated,
        prompt_tokens = prompt_n,
        prefill_ms,
        decode_ms,
        decode_tok_s,
        finish_reason,
        "expert-parallel generation completed"
    );
    eprintln!("[daemon] EP generate done: {generated} tok, {decode_tok_s:.1} tok/s");
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated,
        "tok_s": (decode_tok_s * 10.0).round() / 10.0,
        "prefill_tokens": prompt_n,
        "prefill_ms": (prefill_ms * 10.0).round() / 10.0,
        "prefill_tok_s": (prefill_tok_s * 10.0).round() / 10.0,
        "decode_tok_s": (decode_tok_s * 10.0).round() / 10.0,
        "ttft_ms": (prefill_ms * 10.0).round() / 10.0,
        "finish_reason": finish_reason,
        "attempt_id": active_attempt_id(),
    });
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => emit_staged_terminal_done(stdout, &pending_done),
        ClientTerminalDecision::Abort => ep_emit_abort(stdout, id, m, generated),
    }
}

/// Full EP route-complete abort reset: per-rank bind + cursor reset + decode
/// cache zero + graph invalidate, then device_synchronize on every rank.
/// `rolled_back` is true only when every bind/reset and synchronize succeeds.
pub fn ep_reset_after_abort(m: &mut LoadedModel) -> RollbackEpilogue {
    let mut first_err: Option<String> = None;
    if let Some(ep) = m.ep.as_mut() {
        let EpState { gpus, inner } = ep;
        match inner {
            EpArch::Ds4 { state, .. } => {
                for (rank, s) in state.iter_mut().enumerate() {
                    let g = &mut gpus.devices[rank];
                    if let Err(e) = g.bind_thread() {
                        push_reset_err(&mut first_err, &format!("ep rank{rank} bind_thread"), e);
                    }
                    s.reset();
                    s.zero_decode_caches(g);
                    g.invalidate_graph_state();
                }
            }
            EpArch::Minimax { state, .. } => {
                for (rank, s) in state.iter_mut().enumerate() {
                    let g = &mut gpus.devices[rank];
                    if let Err(e) = g.bind_thread() {
                        push_reset_err(&mut first_err, &format!("ep rank{rank} bind_thread"), e);
                    }
                    s.reset();
                    g.invalidate_graph_state();
                }
            }
            EpArch::Qwen35 { batch, .. } => {
                if let Some(batch) = batch.as_mut() {
                    if let Err(e) = batch.reset_all(gpus) {
                        push_reset_err(&mut first_err, "qwen35 ep batch reset_all", e);
                    }
                }
                for dev in &mut gpus.devices {
                    dev.invalidate_graph_state();
                }
            }
            EpArch::Qwen35DenseTp { dn_states, .. } => {
                for (rank, state) in dn_states.iter_mut().enumerate() {
                    let gpu = &mut gpus.devices[rank];
                    if let Err(e) = gpu.bind_thread() {
                        push_reset_err(
                            &mut first_err,
                            &format!("dense qwen TP rank{rank} bind_thread"),
                            e,
                        );
                    }
                    if let Err(e) = state.reset(gpu) {
                        push_reset_err(
                            &mut first_err,
                            &format!("dense qwen TP rank{rank} state reset"),
                            e,
                        );
                    }
                    gpu.invalidate_graph_state();
                }
            }
        }
        for (rank, dev) in gpus.devices.iter_mut().enumerate() {
            if let Err(e) = dev.bind_thread() {
                push_reset_err(
                    &mut first_err,
                    &format!("ep rank{rank} sync bind_thread"),
                    e,
                );
            }
            if let Err(e) = dev.hip.device_synchronize() {
                push_reset_err(
                    &mut first_err,
                    &format!("ep rank{rank} device_synchronize"),
                    e,
                );
            }
        }
    }
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    if first_err.is_none() {
        RollbackEpilogue {
            rolled_back: true,
            context: None,
        }
    } else {
        RollbackEpilogue {
            rolled_back: false,
            context: first_err.or_else(|| Some("EP abort reset could not be attested".into())),
        }
    }
}

/// EP cancel terminal: reset/sync first, then emit attempt-correlated
/// `aborted`+`done(aborted)` only when rollback is attested. Unattested →
/// one fail-closed error, no done.
pub fn ep_emit_abort(
    stdout: &mut std::io::Stdout,
    id: &str,
    m: &mut LoadedModel,
    completion_tokens: usize,
) {
    let epilogue = ep_reset_after_abort(m);
    if !epilogue.rolled_back {
        emit_fail_closed_error(
            stdout,
            Some(id),
            "client cancelled; fail-closed EP rollback could not be attested",
            "validation",
            false,
            &epilogue,
        );
        return;
    }
    let attempt_id = active_attempt_id();
    let (aborted, done) = ds4_ep_abort_wire_events(id, completion_tokens, attempt_id);
    let _ = writeln!(stdout, "{}", aborted);
    let _ = writeln!(stdout, "{}", done);
    let _ = stdout.flush();
}

/// ds4 EP prefill + greedy decode.
pub fn ep_serve_ds4(
    m: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt_ids: &[u32],
    eos_tok: u32,
    max_tokens: usize,
    think_mode: ThinkMode,
    tools: Option<&[serde_json::Value]>,
    stop: &[String],
    sampling: EpSampling,
) {
    use hipfire_arch_deepseek4::dsml::StreamEvent;
    use std::time::Instant;

    let prompt_n = prompt_ids.len();

    // O2b-2 capacity guard (ds4 EP): this path replays the full prompt from
    // position 0 every turn (no LCP reuse), so the absolute KV span is
    // prompt_n + max_tokens. Without eviction the EP state KV was allocated
    // for `m.physical_cap` (== max_seq at load). Overrunning it drives
    // forward_ep past the KV buffer → corruption/panic (serve-wide crash).
    // Emit a clean error and return BEFORE prefill — mirror the qwen35 guard.
    // saturating_add: an adversarially huge max_tokens must not wrap usize and
    // slip under the cap.
    if prompt_n.saturating_add(max_tokens) > m.physical_cap {
        emit_active_attempt_error(
            stdout,
            Some(id),
            &format!("prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq", prompt_n, max_tokens, m.physical_cap),
            "context_length",
            false,
            false
        );
        let _ = stdout.flush();
        return;
    }

    // ── Cross-conversation reset (FIX: ds4 EP turn-to-turn contamination) ──
    // ds4 EP replays the full prompt from position 0 every turn (no LCP reuse —
    // see the capacity-guard comment above), so a CONTINUING conversation
    // re-prefills its whole history straight from the prompt; nothing in the EP
    // state is reusable across turns, so we reset unconditionally here. This is
    // the EP analogue of the single-GPU cache-miss reset in generate_deepseek4:
    // `reset()` alone only rewinds n_tokens — the position-indexed decode caches
    // (SWA ring, compressed/full KV, indexer scratch) retain the PRIOR turn's
    // residue and bleed into the next conversation (observed: turn 2 echoing
    // turn 1's answer) unless explicitly zeroed. Do it per rank on its own
    // device. Without this, ep_serve_ds4 only ever reset on abort (and even that
    // path, ep_reset_after_abort, omitted zero_decode_caches).
    if let Some(ep) = m.ep.as_mut() {
        let EpState { gpus, inner } = ep;
        if let EpArch::Ds4 { state, .. } = inner {
            for (rank, s) in state.iter_mut().enumerate() {
                let g = &mut gpus.devices[rank];
                let _ = g.bind_thread();
                s.reset();
                s.zero_decode_caches(g);
                g.invalidate_graph_state();
            }
        }
    }
    m.seq_pos = 0;
    m.conversation_tokens.clear();

    let mut parser = match think_mode {
        ThinkMode::Low | ThinkMode::High | ThinkMode::Max => {
            deepseek4::dsml::StreamParser::new_in_think()
        }
        ThinkMode::NonThink => deepseek4::dsml::StreamParser::new(),
    };
    let tool_schemas: Vec<deepseek4::grammar::ToolSchema> = tools
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
                    deepseek4::grammar::ToolSchema {
                        name,
                        params,
                        required,
                    }
                })
                .filter(|s: &deepseek4::grammar::ToolSchema| !s.name.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let grammar_active = !tool_schemas.is_empty();
    let mut matcher = deepseek4::grammar::Matcher::new(tool_schemas);
    let decoded_vocab_arc: Option<std::sync::Arc<Vec<String>>> = if grammar_active {
        if m.decoded_vocab.is_none() {
            let tokenizer = m.tokenizer.as_ref().unwrap();
            let n = tokenizer.vocab_size();
            let v: Vec<String> = (0..n).map(|id| tokenizer.decode(&[id as u32])).collect();
            m.decoded_vocab = Some(std::sync::Arc::new(v));
        }
        m.decoded_vocab.clone()
    } else {
        None
    };
    let empty_vocab: Vec<String> = Vec::new();
    let decoded_vocab: &[String] = decoded_vocab_arc
        .as_deref()
        .map(|v| v.as_slice())
        .unwrap_or(&empty_vocab);
    let mut grammar_mask: Vec<bool> = vec![true; decoded_vocab.len()];
    let mut emit_text_buf = String::new();
    let mut emit_tool_calls_buf: Vec<hipfire_runtime::prompt_frame::ToolCall> = Vec::new();
    let mut dsml_malformed: Option<String> = None;
    // ToolCalls stay buffered turn-wide; only Token/Reasoning go live on the wire.
    let mut absorb_event = |ev: &StreamEvent| {
        ds4_absorb_stream_event(
            ev,
            &mut emit_text_buf,
            &mut emit_tool_calls_buf,
            &mut dsml_malformed,
        );
    };

    // The HTTP stream contract rejects any token before `gen_start`.  EP has
    // a bespoke decode loop, so unlike the single-device AR/spec paths it does
    // not inherit their emitter-side latch.  Open it after all early request
    // validation but before prefill/decode can produce a client event.
    emit_ds4_ep_gen_start(stdout, id, think_mode);

    let t_prefill = Instant::now();
    // FIX #1 (ep-prefill-abort): set when check_abort fires inside the prefill
    // loop. Declared outside the borrow scope so the post-loop abort guard can
    // read it after the `gpus`/`state` borrow is dropped.
    let mut aborted_in_prefill = false;
    {
        let EpState { gpus, inner } = m.ep.as_mut().unwrap();
        let EpArch::Ds4 {
            config,
            weights,
            state,
            partials,
            prefill,
        } = inner
        else {
            emit_active_attempt_error(
                stdout,
                Some(id),
                "EP arch mismatch (expected ds4)",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        };
        if !prefill.is_empty() {
            if check_abort(id) {
                aborted_in_prefill = true;
            } else if let Err(e) = deepseek4::forward::forward_ep_prefill_batch_chunked(
                gpus,
                weights,
                config,
                state,
                prefill,
                &prompt_ids,
                0,
            ) {
                emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!(
                        "forward_ep batched prefill: {}",
                        format!("{e}").replace('"', "'")
                    ),
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return;
            }
        } else {
            for (pos, &t) in prompt_ids.iter().enumerate() {
                // FIX #1 (ep-prefill-abort): check the cancel signal at the TOP of
                // every prefill iteration, not just after the loop. A long prompt
                // (thousands of tokens) means the post-loop check below would still
                // run the entire multi-GPU prefill before honoring a cancel. Mirror
                // the decode loop: on abort, emit aborted+done, reset KV cursors,
                // and stop. We must drop the `gpus`/`state` borrow before calling
                // `ep_emit_abort` (which re-borrows `m.ep`), so break out and let
                // the post-loop guard fire — but set the abort flag is consumed by
                // check_abort, so call it here and short-circuit via a flag.
                if check_abort(id) {
                    // Drop the EpState borrow by breaking; the post-loop guard
                    // re-checks via a sentinel. Simpler: emit + return is blocked
                    // by the borrow, so we set `aborted` and break.
                    aborted_in_prefill = true;
                    break;
                }
                if let Err(e) = deepseek4::forward::forward_ep(
                    gpus, weights, config, state, partials, t, pos as u32,
                ) {
                    emit_active_attempt_error(
                        stdout,
                        Some(id),
                        &format!("forward_ep prefill: {}", format!("{e}").replace('"', "'")),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    return;
                }
            }
        }
    }
    // FIX #1 / FIX #3 (ep-no-abort): a client cancel during the (potentially
    // long) prefill should stop here instead of running the whole decode loop.
    // `aborted_in_prefill` is set when check_abort fired mid-loop (it already
    // consumed the signal, so we don't re-call check_abort); the post-loop
    // check_abort catches a cancel that arrived after the final iteration.
    // Mirror the single-GPU paths: emit aborted+done and reset every rank's KV
    // cursor.
    if aborted_in_prefill || check_abort(id) {
        ep_emit_abort(stdout, id, m, 0);
        return;
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    let mut logits = {
        let EpState { gpus, inner } = m.ep.as_mut().unwrap();
        let EpArch::Ds4 { state, .. } = inner else {
            emit_active_attempt_error(
                stdout,
                Some(id),
                "EP arch mismatch (expected ds4)",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        };
        let _ = gpus.devices[0].bind_thread();
        match state[0].logits.as_ref() {
            // FIX #4 (ep-download-swallow): on a download failure, emit a JSON
            // error and STOP — never `unwrap_or_default()` into an all-zero
            // logits vec (argmax → token 0, an undetectable corruption).
            Some(l) => match gpus.devices[0].download_f32(l) {
                Ok(v) => v,
                Err(e) => {
                    emit_active_attempt_error(
                        stdout,
                        Some(id),
                        &format!(
                            "EP first-logits download failed: {}",
                            format!("{e:?}").replace('"', "'")
                        ),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    return;
                }
            },
            None => {
                emit_active_attempt_error(
                    stdout,
                    Some(id),
                    "EP logits unset after prefill",
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return;
            }
        }
    };

    let t_decode = Instant::now();
    let mut generated = 0usize;
    let mut pos = prompt_n;
    let mut text_acc = String::new();
    let mut local_emitted_ids: Vec<u32> = Vec::new();
    while generated < max_tokens {
        // FIX #3 (ep-no-abort): client cancel mid-decode → emit aborted+done,
        // reset EP cursors, stop. Without this a Pi/CLI cancel leaves the EP
        // decode loop running for the full max_tokens of wasted multi-GPU work.
        if check_abort(id) {
            ep_emit_abort(stdout, id, m, generated);
            return;
        }
        if grammar_active && !matcher.is_free() {
            matcher.token_mask(decoded_vocab, &mut grammar_mask);
            deepseek4::grammar::Matcher::apply_mask_to_logits(&grammar_mask, &mut logits);
        }
        // Host-side sampler over the downloaded f32 logits (temp → top_k →
        // top_p → min_p → seeded draw, temp<=1e-6 = argmax). RNG seeded once
        // per request via reset_cpu_sampler_rng(request_seed) in generate().
        let next = hipfire_runtime::llama::sample_full_dist(
            &logits,
            sampling.temp,
            sampling.top_p,
            sampling.top_k,
            sampling.min_p,
        );
        if next == eos_tok {
            break;
        }
        let piece = m.tokenizer.as_ref().unwrap().decode(&[next]);
        for ev in parser.feed(&piece) {
            absorb_event(&ev);
            emit_stream_event(stdout, id, ev);
        }
        emit_committed_event(
            stdout,
            id,
            next,
            generated,
            t_decode.elapsed().as_millis() as u64,
        );
        let _ = stdout.flush();
        if grammar_active {
            matcher.advance(&piece);
        }
        local_emitted_ids.push(next);
        text_acc.push_str(&piece);
        generated += 1;
        if stop.iter().any(|s| !s.is_empty() && text_acc.ends_with(s)) {
            break;
        }
        let EpState { gpus, inner } = m.ep.as_mut().unwrap();
        let EpArch::Ds4 {
            config,
            weights,
            state,
            partials,
            ..
        } = inner
        else {
            break;
        };
        if let Err(e) =
            deepseek4::forward::forward_ep(gpus, weights, config, state, partials, next, pos as u32)
        {
            emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("forward_ep decode: {}", format!("{e}").replace('"', "'")),
                "validation",
                false,
                false,
            );
            return;
        }
        pos += 1;
        let _ = gpus.devices[0].bind_thread();
        // FIX #4 (ep-download-swallow): explicit error handling on the per-token
        // logits download — emit a JSON error and stop, never feed a zeroed
        // (token-0) logits vec.
        logits = match state[0].logits.as_ref() {
            Some(l) => match gpus.devices[0].download_f32(l) {
                Ok(v) => v,
                Err(e) => {
                    emit_active_attempt_error(
                        stdout,
                        Some(id),
                        &format!(
                            "EP decode logits download failed: {}",
                            format!("{e:?}").replace('"', "'")
                        ),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    return;
                }
            },
            None => break,
        };
    }
    for ev in parser.finish() {
        absorb_event(&ev);
        emit_stream_event(stdout, id, ev);
    }
    let _ = stdout.flush();
    drop(absorb_event);

    // Pure turn-wide terminal: malformed discards every buffered call.
    let terminal =
        ds4_ar_ep_finish_route(dsml_malformed, emit_tool_calls_buf, generated >= max_tokens);
    let (finish_reason, wire_tool_calls, store_cache) = match &terminal {
        Ds4ArEpRouteTerminal::Malformed(action) => {
            emit_ds4_malformed_action(stdout, id, action);
            return;
        }
        Ds4ArEpRouteTerminal::Safe {
            finish_reason,
            wire_tool_calls,
            store_cache,
        } => (*finish_reason, wire_tool_calls.clone(), *store_cache),
    };
    // Timing fields are fixed before handshake so commit_ready carries the
    // exact eventual done payload.
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    let decode_tok_s = if decode_ms > 0.0 {
        generated as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };
    let prefill_tok_s = if prefill_ms > 0.0 {
        prompt_n as f64 / (prefill_ms / 1000.0)
    } else {
        0.0
    };
    eprintln!("[daemon] EP generate done: {generated} tok, {decode_tok_s:.1} tok/s");
    let mut pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated,
        "tok_s": decode_tok_s,
        "prefill_tokens": prompt_n,
        "prefill_ms": prefill_ms,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "ttft_ms": prefill_ms,
        "finish_reason": finish_reason,
        "attempt_id": active_attempt_id(),
    });
    // Stage canonical calls before handshake so pure-tool HTTP terminals are
    // complete at commit_ready. Final done is payload-identical.
    stage_terminal_tool_calls(&mut pending_done, finish_reason, &wire_tool_calls);
    // Two-phase commit before cache / done. No post-commit tool_calls event.
    let decision = await_client_terminal_commit(stdout, id, &pending_done);
    let effects = ds4_client_commit_effects(decision, finish_reason == "tool_calls", store_cache);
    if !effects.emit_done {
        ep_emit_abort(stdout, id, m, generated);
        return;
    }
    let mut action = ds4_ar_ep_cache_action(&terminal, &emit_text_buf);
    if !effects.store_cache {
        action.store = false;
    }
    // Mirror prior gate: require generated > 0 so empty turns never store.
    if action.store && generated > 0 {
        if std::env::var("HIPFIRE_DEEPSEEK4_CACHE_TRACE")
            .ok()
            .as_deref()
            == Some("1")
        {
            eprintln!(
                "[asst-cache store] content.len={} tool_calls={} tokens={}",
                emit_text_buf.len(),
                wire_tool_calls.len(),
                local_emitted_ids.len(),
            );
        }
        let _ = ds4_apply_cache_action(
            |fp, seq| {
                if std::env::var("HIPFIRE_DEEPSEEK4_CACHE_TRACE")
                    .ok()
                    .as_deref()
                    == Some("1")
                {
                    eprintln!("[asst-cache store] fp={:#018x} tokens={}", fp, seq.len());
                }
                m.asst_turn_cache.insert(
                    fp,
                    hipfire_runtime::prompt_frame::CachedAssistantTurn {
                        reasoning: None,
                        tools: Vec::new(),
                        content: Some(hipfire_runtime::prompt_frame::CachedAssistantBody {
                            token_ids: seq,
                            text: String::new(),
                        }),
                    },
                );
            },
            &action,
            local_emitted_ids,
        );
    }

    emit_staged_terminal_done(stdout, &pending_done);
    let _ = stdout.flush();
}

/// MiniMax-M2 EP prefill + greedy decode (mirror of ep_serve_ds4, MiniMax types).
/// Carries the single-GPU prefix cache to EP: an LCP over the shared
/// `conversation_tokens` rewinds every rank's KV cursor to the common prefix
/// and re-prefills only the divergent suffix (interleaved-thinking partial
/// reuse — see generate_minimax for the full rationale). `primed_think`
/// re-emits the MiniMax `<think>\n` opener display-only for a well-formed turn.
#[allow(clippy::too_many_arguments)]
pub fn ep_serve_minimax(
    m: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt_ids: &[u32],
    eos_tok: u32,
    max_tokens: usize,
    stop: &[String],
    primed_think: bool,
    sampling: EpSampling,
) {
    use std::time::Instant;
    let prompt_n = prompt_ids.len();

    // O2b-2 capacity guard (minimax EP): even with LCP reuse the KV ends up
    // holding [0, prompt_n) after prefill, then decode appends max_tokens, so
    // the absolute span is prompt_n + max_tokens. The EP state KV was allocated
    // for `m.physical_cap` (== max_seq at load); overrunning it writes past the
    // per-rank KV buffer → corruption/panic (serve-wide crash). Emit a clean
    // error and return BEFORE any state mutation — mirror the qwen35 guard.
    // saturating_add: an adversarially huge max_tokens must not wrap usize and
    // slip under the cap.
    if prompt_n.saturating_add(max_tokens) > m.physical_cap {
        emit_active_attempt_error(
            stdout,
            Some(id),
            &format!("prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq", prompt_n, max_tokens, m.physical_cap),
            "context_length",
            false,
            false
        );
        let _ = stdout.flush();
        return;
    }

    // ── LCP partial reuse. The per-rank KV holds [0, prior_total) from last
    // turn; `conversation_tokens` mirrors it. Rewind n_tokens to the common
    // prefix and re-prefill the (reasoning-free, shorter) suffix; MiniMax is
    // standard attention so KV ≥ lcp is overwritten and the stale tail is
    // never attended. lcp == 0 ⇒ cold prefill. ──
    let prefill_from: usize = {
        let prior_len = m.conversation_tokens.len();
        let max_match = prior_len.min(prompt_n);
        let mut lcp = 0usize;
        while lcp < max_match && m.conversation_tokens[lcp] == prompt_ids[lcp] {
            lcp += 1;
        }
        let cache_hit = lcp > 0 && lcp < prompt_n;
        if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "[minimax-ep-cache] prior_len={} rendered_len={} lcp={} hit={} partial={}",
                prior_len,
                prompt_n,
                lcp,
                cache_hit,
                cache_hit && lcp < prior_len,
            );
        }
        if cache_hit {
            m.conversation_tokens.truncate(lcp);
            lcp
        } else {
            m.conversation_tokens.clear();
            0
        }
    };
    // Rewind every rank's KV cursor to the reuse point.
    {
        let EpState { inner, .. } = m.ep.as_mut().unwrap();
        if let EpArch::Minimax { state, .. } = inner {
            for s in state.iter_mut() {
                s.n_tokens = prefill_from;
            }
        }
    }

    // ── Prefill the suffix [prefill_from, prompt_n) across ranks. ──
    let t_prefill = Instant::now();
    // FIX #1 (ep-prefill-abort): set when check_abort fires inside the prefill
    // loop. Declared outside the borrow scope so the abort guard below can read
    // it after the `gpus`/`state` borrow is dropped.
    let mut aborted_in_prefill = false;
    // Track how many suffix tokens were actually prefilled so that, on a
    // mid-prefill abort, we don't mirror un-prefilled tokens into
    // `conversation_tokens` (which would desync the cache from KV state).
    let mut prefilled_n = 0usize;
    {
        let EpState { gpus, inner } = m.ep.as_mut().unwrap();
        let EpArch::Minimax {
            config,
            weights,
            state,
            partials,
        } = inner
        else {
            emit_active_attempt_error(
                stdout,
                Some(id),
                "EP arch mismatch (expected minimax)",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        };
        for (i, &t) in prompt_ids[prefill_from..].iter().enumerate() {
            // FIX #1 (ep-prefill-abort): honor a client cancel at the TOP of
            // every prefill iteration, not just after the loop. Without this a
            // long prompt runs the full multi-GPU prefill before the post-loop
            // check fires. check_abort consumes the signal, so record it in a
            // flag and break; the borrow on `m.ep` is dropped before we call
            // ep_emit_abort below.
            if check_abort(id) {
                aborted_in_prefill = true;
                break;
            }
            let pos = (prefill_from + i) as u32;
            if let Err(e) =
                minimax::forward::forward_ep(gpus, weights, config, state, partials, t, pos)
            {
                emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!("forward_ep prefill: {}", format!("{e}").replace('"', "'")),
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return;
            }
            prefilled_n = i + 1;
        }
    }
    // Mirror only the actually-prefilled suffix into conversation_tokens (the
    // prefix is kept). On a mid-prefill abort, ep_emit_abort resets every
    // rank's KV cursor, so leaving conversation_tokens at the prefix keeps the
    // cache consistent with KV state.
    for &t in &prompt_ids[prefill_from..prefill_from + prefilled_n] {
        m.conversation_tokens.push(t);
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    // FIX #1 / FIX #3 (ep-no-abort): client cancel during prefill → stop
    // cleanly. `aborted_in_prefill` already consumed the signal mid-loop; the
    // post-loop check_abort catches a cancel that arrived after the last token.
    if aborted_in_prefill || check_abort(id) {
        ep_emit_abort(stdout, id, m, 0);
        return;
    }

    // MiniMax primes the assistant with `<think>\n`; re-emit display-only so the
    // assistant message is a well-formed think block (parity with single-GPU).
    if primed_think {
        let _ = writeln!(
            stdout,
            "{}",
            serde_json::json!({"type":"token","id":id,"text":"<think>\n", "attempt_id": active_attempt_id()})
        );
        let _ = stdout.flush();
    }

    let mut logits = {
        let EpState { gpus, inner } = m.ep.as_mut().unwrap();
        let EpArch::Minimax { state, .. } = inner else {
            return;
        };
        let _ = gpus.devices[0].bind_thread();
        // FIX #4 (ep-download-swallow): explicit error handling — never feed a
        // zeroed (token-0) logits vec on a download failure.
        match gpus.devices[0].download_f32(&state[0].logits) {
            Ok(v) => v,
            Err(e) => {
                emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!(
                        "EP first-logits download failed: {}",
                        format!("{e:?}").replace('"', "'")
                    ),
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return;
            }
        }
    };
    let t_decode = Instant::now();
    let mut generated = 0usize;
    let mut pos = prompt_n;
    let mut text_acc = String::new();
    let mut natural_stop = false;
    while generated < max_tokens {
        // FIX #3 (ep-no-abort): client cancel mid-decode → emit aborted+done,
        // reset EP cursors, stop.
        if check_abort(id) {
            ep_emit_abort(stdout, id, m, generated);
            return;
        }
        // Host-side sampler over downloaded f32 logits (temp → top_k → top_p →
        // min_p → seeded draw, temp<=1e-6 = argmax). MiniMax's card carries
        // top_k=40, threaded here via sampling.top_k. RNG seeded per request in
        // generate() (reset_cpu_sampler_rng).
        let next = hipfire_runtime::llama::sample_full_dist(
            &logits,
            sampling.temp,
            sampling.top_p,
            sampling.top_k,
            sampling.min_p,
        );
        if next == eos_tok {
            natural_stop = true;
            break;
        }
        let piece = m.tokenizer.as_ref().unwrap().decode(&[next]);
        generated += 1;
        m.conversation_tokens.push(next);
        if ep_emit_token(stdout, id, &piece, &mut text_acc, stop) {
            natural_stop = true;
            break;
        }
        let EpState { gpus, inner } = m.ep.as_mut().unwrap();
        let EpArch::Minimax {
            config,
            weights,
            state,
            partials,
        } = inner
        else {
            break;
        };
        if let Err(e) =
            minimax::forward::forward_ep(gpus, weights, config, state, partials, next, pos as u32)
        {
            emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("forward_ep decode: {}", format!("{e}").replace('"', "'")),
                "validation",
                false,
                false,
            );
            return;
        }
        pos += 1;
        let _ = gpus.devices[0].bind_thread();
        // FIX #4 (ep-download-swallow): explicit error handling on the per-token
        // download — emit a JSON error and stop, never feed a zeroed (token-0)
        // logits vec.
        logits = match gpus.devices[0].download_f32(&state[0].logits) {
            Ok(v) => v,
            Err(e) => {
                emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!(
                        "EP decode logits download failed: {}",
                        format!("{e:?}").replace('"', "'")
                    ),
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return;
            }
        };
    }
    let finish_reason = if !natural_stop && generated >= max_tokens {
        "length"
    } else {
        "stop"
    };
    ep_emit_done(
        stdout,
        id,
        m,
        generated,
        prompt_n,
        prefill_ms,
        t_decode.elapsed().as_secs_f64() * 1000.0,
        finish_reason,
    );
}

/// Format for re-rendering a historical assistant `tool_call` on a cache
/// MISS in the non-jinja ChatScaffold path (`HIPFIRE_JINJA_CHAT=0`).
/// Mirrors the CLI's per-model grammar gating: grammar OFF (the default
/// for native-XML Qwen3.5/3.6) => native XML; grammar ON (Hermes-format
/// models like carnice, or a user override) => Hermes JSON. Both
/// `build_cached_history` call sites (LCP planning in `plan_prompt_cache`
/// and the actual prompt render) read this so their token streams stay
/// byte-consistent — a mismatch would break the LCP forward-extension.
pub fn qwen_history_tool_render(model_path: &str) -> hipfire_runtime::prompt_frame::ToolCallRender {
    hipfire_runtime::prompt_frame::qwen35_history_render(
        std::env::var("HIPFIRE_QWEN35_GRAMMAR").ok().as_deref(),
        model_path,
    )
}

/// Pure LCP prompt-cache decision shared in spirit with the AR `generate`
/// path's inline block — but side-effect-free (touches no GPU/seq_pos state),
/// so the DFlash path can use it too. Renders the canonical conversation via
/// `build_cached_history` (verbatim assistant-turn replay through
/// `asst_turn_cache`, which is what makes the LCP byte-exact across turns), then
/// compares against `m.conversation_tokens`. Reports a HIT only on a strict
/// forward extension (`lcp == prior_len && lcp < rendered.len()`), which keeps
/// the recurrent DeltaNet state valid by construction (the prior turn left it at
/// exactly `prior_len`, so prefilling the suffix advances it correctly with no
/// rewind). The exact-match edge (`lcp == rendered.len()`) degrades to a miss to
/// avoid a 1-token DeltaNet over-advance. Caller must be in the
/// `messages_history.is_some()` case.
#[allow(clippy::too_many_arguments)]
pub fn plan_prompt_cache(
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    asst_turn_cache: &mut AsstTurnCache,
    conversation_tokens: &[u32],
    eviction_is_none: bool,
    system_prompt: Option<&str>,
    prompt: &str,
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    tool_render: hipfire_runtime::prompt_frame::ToolCallRender,
    messages_history: &[hipfire_runtime::prompt_frame::Message],
    cache_disabled: bool,
    // Ascending DeltaNet checkpoint positions (from `m.dflash_checkpoints`) and
    // whether resume-from-checkpoint is enabled. On a divergence the plan picks
    // the latest checkpoint `<= lcp && < rendered.len()` to resume from.
    dflash_ckpt_positions: &[usize],
    resume_enabled: bool,
) -> PromptCachePlan {
    let q_tokens = tokenizer.encode(prompt);
    let rendered = hipfire_runtime::prompt_frame::build_cached_history(
        tokenizer,
        system_prompt,
        messages_history,
        &q_tokens,
        assistant_prefix,
        tool_render,
        |msg| {
            let stripped = strip_think_for_fingerprint(&msg.content);
            let normalized =
                hipfire_runtime::tokenizer::maybe_normalize_prompt(&stripped).into_owned();
            let fp = asst_turn_fingerprint(&normalized, &msg.tool_calls);
            asst_turn_cache
                .get(&fp)
                .and_then(|t| t.content.as_ref().map(|c| c.token_ids.clone()))
        },
    );
    let cache_eligible = !cache_disabled && eviction_is_none && !conversation_tokens.is_empty();
    plan_from_rendered(
        conversation_tokens,
        rendered,
        cache_eligible,
        dflash_ckpt_positions,
        resume_enabled,
        "dflash",
    )
}

/// LCP / hit / resume planner shared by the Plain canonical-render path
/// (`plan_prompt_cache`) and the jinja path (`generate_dflash` under
/// `try_jinja`, which renders through `build_cached_history_jinja` —
/// verbatim assistant-turn splice through the model's trained template,
/// mirroring generate()'s item-#37 cache). `rendered` is the full
/// canonical conversation stream for this turn; eligibility is the
/// caller's precomputed decision.
pub fn plan_from_rendered(
    conversation_tokens: &[u32],
    rendered: Vec<u32>,
    cache_eligible: bool,
    dflash_ckpt_positions: &[usize],
    resume_enabled: bool,
    trace_tag: &str,
) -> PromptCachePlan {
    if cache_eligible {
        let prior_len = conversation_tokens.len();
        let max_match = prior_len.min(rendered.len());
        let mut lcp = 0usize;
        while lcp < max_match && conversation_tokens[lcp] == rendered[lcp] {
            lcp += 1;
        }
        if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "[qwen-cache lcp {trace_tag}] prior_len={} rendered_len={} lcp={}",
                prior_len,
                rendered.len(),
                lcp
            );
        }
        if lcp == prior_len && lcp < rendered.len() && lcp > 0 {
            return PromptCachePlan {
                new_tokens: rendered[lcp..].to_vec(),
                start_pos: lcp,
                cached_tokens: lcp,
                cache_hit: true,
                resume_from: None,
                rendered,
            };
        }
        // Divergent render (lcp < prior_len, or the exact-match edge): not a
        // pure extension, so the recurrent state at the end is stale. If resume
        // is enabled, rewind to the latest checkpoint at-or-before lcp that
        // still leaves ≥1 token to re-prefill, and resume from there instead of
        // cold-prefilling the whole conversation.
        if resume_enabled {
            if let Some(&ckpt) = dflash_ckpt_positions
                .iter()
                .filter(|&&p| p <= lcp && p < rendered.len())
                .max()
            {
                eprintln!(
                    "[qwen-cache resume {trace_tag}] checkpoint pos={} (lcp={}, prior_len={}, rendered_len={}) — replaying {} tokens vs cold-prefilling {}",
                    ckpt, lcp, prior_len, rendered.len(), rendered.len() - ckpt, rendered.len(),
                );
                return PromptCachePlan {
                    new_tokens: rendered[ckpt..].to_vec(),
                    start_pos: ckpt,
                    cached_tokens: ckpt,
                    cache_hit: true,
                    resume_from: Some(ckpt),
                    rendered,
                };
            }
        }
    }
    PromptCachePlan {
        new_tokens: rendered.clone(),
        start_pos: 0,
        cached_tokens: 0,
        cache_hit: false,
        resume_from: None,
        rendered,
    }
}

/// DFlash-powered greedy decode. Mirrors `generate`'s ChatML shape and
/// token-streaming output but replaces the AR sample loop with
/// `spec_step_dflash` cycles — each cycle drafts B tokens via the diffusion
/// model and verifies them in one target forward, committing accept_len+1
/// at a time.
///
/// Prompt cache: for `messages_history`-bearing chat requests this path now
/// reuses the target KV + DeltaNet prefix on a pure conversation extension
/// (via [`plan_prompt_cache`] + `seed_target_hidden_suffix_abortable`), and the
/// draft's cumulative `target_hidden` is extended by scattering only the suffix
/// rows — so DFlash keeps its decode speedup AND skips re-prefilling the cached
/// prefix. A divergent / first / raw-prompt turn full-resets and prefills the
/// whole conversation as before.
///
/// The arch-dispatched borrow of the spec-decode target as `&mut dyn SpecTarget`
/// now lives behind the loader's `spec_target_guard()` + the runtime
/// `SpecTargetGuard` trait — this fn only ever sees `&mut dyn SpecTarget` and
/// never learns which arch (qwen35 moved-bundle vs llama borrow-in-place) it drives.
#[allow(clippy::too_many_arguments)]
/// Render the [`ClientEvent`]s a [`SpecEmit`] step produced to the daemon's
/// JSONL wire format, byte-identical to `generate_dflash`'s old inline writes.
/// `t_ms` is the per-step timestamp the inline path attached to committed +
/// token frames (`t0.elapsed()`); tool_calls frames carry no timing.
///
/// `hold_tool_calls`: when true (terminal `finish` flush), skip rendering
/// `ClientEvent::ToolCalls` so the arch wrapper can classify length/malformed
/// before any executable release. Mid-loop begin/observe pass false.
pub fn render_client_events(
    stdout: &mut impl std::io::Write,
    id: &str,
    events: &[ClientEvent],
    t_ms: u64,
    hold_tool_calls: bool,
) {
    for ev in events {
        match ev {
            ClientEvent::Committed { id: tok_id, idx } => {
                emit_committed_event(stdout, id, *tok_id, *idx, t_ms);
            }
            ClientEvent::Token(text) => {
                let envelope = qwen_dflash_token_event_value(id, text, active_attempt_id());
                let _ = writeln!(stdout, "{}", envelope);
                let _ = stdout.flush();
            }
            ClientEvent::Reasoning(text) => {
                let envelope = qwen_dflash_reasoning_event_value(id, text, active_attempt_id());
                let _ = writeln!(stdout, "{}", envelope);
                let _ = stdout.flush();
            }
            ClientEvent::ToolCalls(calls) => {
                if hold_tool_calls {
                    continue;
                }
                emit_tool_calls_event(stdout, id, calls);
            }
        }
    }
}

/// Release held terminal `ClientEvent::ToolCalls` after a tool-safe verdict.
pub fn release_held_finish_tool_calls(
    stdout: &mut impl std::io::Write,
    id: &str,
    finish: &FinishSummary,
) {
    for ev in &finish.events {
        if let ClientEvent::ToolCalls(calls) = ev {
            emit_tool_calls_event(stdout, id, calls);
        }
    }
}

pub fn generate_dflash(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
    max_think_tokens: usize,
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    pflash_bypass_reason: Option<&str>,
    pflash_alpha: Option<f32>,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
    stop: &[String],
    // Request-resolved sampling temperature. 0.0 → greedy/argmax-accept (the
    // historical DFlash posture). >0 → distribution-preserving spec decode.
    // MTP (name=="mtp") honors temp>0 including user-explicit sampling and min_p
    // via SpecRequestConfig; DFlash still ignores min_p (those requests stay AR).
    temp: f32,
    // Nucleus (top_p) cutoff for the chain rejection-sampling path, applied
    // IDENTICALLY to both draft + target softmaxes (lossless == AR at this top_p).
    // 1.0 (>= 0.999) disables it. Ignored by the ddtree SWOR arm.
    top_p: f32,
    // Top-k cutoff (request/card recipe, e.g. qwen3.6 top_k=20) for the chain
    // sampled path, applied to both draft + target softmax rows. 0 = disabled.
    // Ignored by the ddtree SWOR arm.
    top_k: usize,
    // Min-p floor. 0.0 disables. Installed on SpecRequestConfig for MTP;
    // DFlash route selection still sends min_p requests to AR.
    min_p: f32,
    // Cactus-style acceptance bump. 0.0 → lossless (distribution-preserving).
    // >0 → deliberately lossy (KL-bounded τ-for-correctness tradeoff). The
    // daemon hardcodes 0.0; the param exists only so a future opt-in request
    // field can reach it without re-touching this signature.
    cactus_delta: f32,
    // Per-request sampler seed for the drafter's sampled-draw RNG (see
    // Speculator::set_request_seed). Derived by hipfire-engine's
    // request_seed_for: explicit wire `seed` wins, otherwise attempt-key +
    // counter entropy. Greedy requests never draw it.
    request_seed: u64,
    reasoning_effort: Option<&str>,
    enable_thinking: bool,
    // Returns false in exactly one case: the request does not fit the loaded
    // speculator's reported ctx capacity (draft-side structures are capped at
    // load — see DEFAULT_DFLASH_CTX_CAP) — and NO output/events were emitted,
    // so the caller must fall through to the arch's AR path. Every other exit
    // (success, abort, error) has already written its envelope → true.
) -> bool {
    // Zero-budget reject before Jinja/render, configure_request, gen_start, or any
    // GPU/host mutation. Handled (true) so the caller does not fall through to AR.
    // Inner generate_spec keeps the same guard as defense-in-depth.
    if max_tokens == 0 {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "max_tokens must be > 0",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return true;
    }

    let spec_name = m.speculator.as_ref().map(|s| s.name()).unwrap_or("");

    // Adaptive KV has no maybe_downshift on the generic spec path. Fail closed
    // rather than run DSpark/DFlash/ngram past floor-reserved capacity. The
    // error envelope is written here, so this counts as handled (true) — the
    // caller must NOT also emit an AR response.
    if m.kv_adaptive.is_some() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "kv_adaptive cannot use generic speculative decode (DFlash/DSpark/MTP/n-gram); use AR",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return true;
    }

    // The spec-step dispatch, ModelSlot assembly, checkpoint ring, and
    // SpecStats that this function used to drive inline now live behind the
    // arch-generic `Speculator` trait (the loader's `DflashSpeculator`) and the
    // `Qwen35SlotGuard` RAII target borrow — see the prefill/step/on_evict/
    // reset/rewind_to calls below.

    // Prompt build: same two-path branch as the AR-path generate() — when
    // Jinja is enabled (default; opt out with HIPFIRE_JINJA_CHAT=0) AND the
    // model carries a chat_template, render via `JinjaChatFrame` so structured
    // `tools` / `messages` can reach the upstream template's `{% if tools %}` /
    // multi-turn branches. Plain is allowed only for explicit opt-out or when
    // no embedded template resolves. A configured-template render failure is
    // fail-closed (non-retryable validation, handled=true) — never silently
    // changes prompt semantics via Plain fallback.
    //
    // DFlash is single-turn by construction — `seq_pos` is reset to 0
    // below before seed_target_hidden_from_prompt runs — so we never
    // need to guard on `seq_pos == 0` here.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    // LFM2.5 (arch_id 11) REQUIRES its embedded Jinja chat_template — the
    // hand-rolled Plain ChatML path omits LFM2's `<|startoftext|>` BOS and
    // produces garbage. Force jinja on for arch 11 (Plain only if the .hfq
    // carries no template, e.g. an older A1B convert).
    // Jinja default-ON (flipped 2026-06-09): render through the model's chat
    // template for ALL arches; opt out with HIPFIRE_JINJA_CHAT=0 (hand-rolled
    // ChatML/Plain). No template ⇒ Plain. Template present + render Err ⇒
    // fail closed (see match below).
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
    let try_jinja = jinja_enabled && m.chat_template.is_some();
    let mut started_in_think = matches!(
        assistant_prefix,
        hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
    );
    let prompt_tokens: Vec<u32> = if try_jinja {
        let template = m.chat_template.as_ref().unwrap();
        let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking,
            bos_token: None,
            reasoning_strength: None,
            reasoning_effort,
        };
        let render_result = if tools.is_some() || messages_history.is_some() {
            let synthesized: Vec<hipfire_runtime::prompt_frame::Message>;
            let messages_slice: &[hipfire_runtime::prompt_frame::Message] = match messages_history {
                Some(m) => m,
                None => {
                    let mut v = Vec::new();
                    if let Some(sys) = system_prompt {
                        v.push(hipfire_runtime::prompt_frame::Message {
                            role: hipfire_runtime::prompt_frame::Role::System,
                            content: sys.to_string(),
                            reasoning_content: None,
                            name: None,
                            rendered_name: None,
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                            tool_plan: String::new(),
                        });
                    }
                    v.push(hipfire_runtime::prompt_frame::Message {
                        role: hipfire_runtime::prompt_frame::Role::User,
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
            Ok(rendered) => {
                started_in_think = render_tail_opens_think(&rendered);
                tokenizer.encode(&rendered)
            }
            Err(e) => {
                if reasoning_effort.is_some() {
                    emit_active_attempt_error(
                        stdout,
                        Some(id),
                        &format!("DFlash jinja render: {e}"),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    return true;
                }
                eprintln!("[daemon] jinja render failed ({e}) — falling back to Plain");
                hipfire_runtime::prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: prompt,
                    assistant_prefix,
                    raw: false,
                }
                .build()
            }
        }
    } else {
        hipfire_runtime::prompt_frame::ChatFrame {
            tokenizer,
            system: system_prompt,
            user: prompt,
            assistant_prefix,
            raw: false,
        }
        .build()
    };

    // `im_end_token` is still needed downstream for the EOS check.
    let im_end = tokenizer.encode("<|im_end|>");
    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };

    // DFlash ctx-capacity downgrade. The draft's context-indexed structures
    // (target_hidden, per-layer K/V caches, hidden ring) are sized at load to
    // min(max_seq, HIPFIRE_DFLASH_CTX_CAP|8192) to bound draft-side VRAM on
    // large-max_seq serve loads. A request that outgrows the cap cannot run
    // spec — but DFlash is verify-gated, so AR produces the same tokens, just
    // slower. Emit a correlated info event and hand the request back to the
    // caller's AR fallthrough (handled=false: no gen_start/done yet). Mirrors
    // the hard guard in generate_spec (belt-and-suspenders last line).
    let spec_ctx_capacity = m
        .speculator
        .as_ref()
        .map(|s| s.ctx_capacity())
        .unwrap_or(usize::MAX);
    if prompt_tokens.len().saturating_add(max_tokens) > spec_ctx_capacity {
        emit_qwen_ar_info(
            stdout,
            id,
            &format!(
                "prompt={} + max_tokens={} exceeds {} draft ctx capacity {} — falling back to AR (identical output, slower; raise HIPFIRE_DFLASH_CTX_CAP to re-enable spec)",
                prompt_tokens.len(),
                max_tokens,
                if spec_name == "mtp" { "MTP" } else { "DFlash" },
                spec_ctx_capacity
            ),
        );
        return false;
    }

    // Prompt-cache plan (native DFlash reuse). Decide whether this turn is a
    // pure extension of the cached conversation. On a HIT we reuse target KV
    // + DeltaNet[0..start_pos] and the draft's cumulative target_hidden,
    // prefilling only the suffix; on a MISS we full-reset and prefill the
    // whole conversation (legacy behaviour).
    //
    // Jinja mode renders through `build_cached_history_jinja` — the same
    // item-#37 machinery generate() uses: verbatim assistant-turn token
    // splice (fingerprint replay, no decode→encode drift) through the model's
    // trained template, with the assistant-opener primer prepended so the
    // spliced stream byte-matches the end-of-turn bake. Divergence (edited
    // history, roundtrip-unstable text) lands on the checkpoint-resume path —
    // worst case equals today's cold prefill, never wrong tokens.
    let cache_disabled = std::env::var("HIPFIRE_QWEN_PROMPT_CACHE").ok().as_deref() == Some("0");
    // DFlash divergent-render resume (default ON; opt out with
    // HIPFIRE_DFLASH_CKPT_RESUME=0). Requires no eviction (resume rewinds the
    // resident KV prefix). When on, the recurrent state is checkpointed during
    // the prompt seed and a divergent render resumes from the latest checkpoint
    // ≤ lcp — byte-identical to a cold prefill of the same render (verified),
    // so worst case equals the legacy cold-reset path. Off ⇒ no checkpoints
    // (zero overhead) + legacy cold-reset-on-divergence.
    let dflash_resume_enabled = std::env::var("HIPFIRE_DFLASH_CKPT_RESUME").ok().as_deref()
        != Some("0")
        && m.eviction.is_none();
    let dflash_ckpt_positions: Vec<usize> = m
        .speculator
        .as_ref()
        .map(|s| s.checkpoint_positions())
        .unwrap_or_default();
    let cache_eligible = !cache_disabled
        && messages_history.is_some()
        && m.eviction.is_none()
        && pflash_bypass_reason.is_none()
        && !m.conversation_tokens.is_empty();
    let cache_plan: Option<PromptCachePlan> = if try_jinja {
        if let Some(hist) = messages_history {
            // Assistant-opener primer from THIS turn's cold jinja render
            // (everything after the last `<|im_start|>assistant\n`). The
            // replay prepends it only when the template does not already
            // re-emit it on history assistant turns (Qwen3.5 does not,
            // Qwen3.8 does) — mirrors generate()'s item-#37 primer.
            let tok = m.tokenizer.as_ref().unwrap();
            let im_start = tok.special_token_id("<|im_start|>");
            let opener_len = tok.encode("<|im_start|>assistant\n").len();
            let primer: Vec<u32> =
                match im_start.and_then(|id| prompt_tokens.iter().rposition(|&t| t == id)) {
                    Some(q) if q + opener_len <= prompt_tokens.len() => {
                        prompt_tokens[q + opener_len..].to_vec()
                    }
                    _ => Vec::new(),
                };
            let template = m.chat_template.as_ref().unwrap();
            let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
                tokenizer: tok,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking,
                bos_token: None,
                reasoning_strength: None,
                reasoning_effort,
            };
            let primer: Vec<u32> =
                if hipfire_runtime::prompt_frame::template_emits_history_primer(&frame, &primer) {
                    Vec::new()
                } else {
                    primer
                };
            let cache_ref = &mut m.asst_turn_cache;
            let trace_cache =
                std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1");
            let rendered = match hipfire_runtime::prompt_frame::build_cached_history_jinja(
                &frame,
                hist,
                tools,
                |msg| {
                    let normalized = normalize_asst_turn_for_fingerprint(&msg.content);
                    let fp = asst_turn_fingerprint(&normalized, &msg.tool_calls);
                    // The qwen family has no Harmony reasoning/tool channels: its whole
                    // assistant turn is one content slot. `text` must be the message's own
                    // content so the splice's `content.text == m.content` guard
                    // (prompt_frame.rs) passes trivially and behaviour is byte-identical to
                    // the pre-per-channel implementation.
                    let hit = cache_ref.get(&fp).and_then(|turn| {
                        turn.content.as_ref().map(|c| {
                            let mut v = primer.clone();
                            v.extend_from_slice(&c.token_ids);
                            hipfire_runtime::prompt_frame::CachedAssistantTurn {
                                reasoning: None,
                                tools: Vec::new(),
                                content: Some(hipfire_runtime::prompt_frame::CachedAssistantBody {
                                    token_ids: v,
                                    text: msg.content.clone(),
                                }),
                            }
                        })
                    });
                    if trace_cache {
                        eprintln!(
                            "[qwen-cache jinja lookup dflash] fp={:#018x} role={:?} primer={} hit={}",
                            fp,
                            msg.role,
                            primer.len(),
                            hit.is_some()
                        );
                    }
                    hit
                },
            ) {
                Ok(v) => v,
                Err(e) => {
                    if reasoning_effort.is_some() {
                        emit_active_attempt_error(
                            stdout,
                            Some(id),
                            &format!("DFlash qwen-cache jinja build: {e}"),
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        return true;
                    }
                    eprintln!(
                        "[qwen-cache] DFlash jinja cached-history build failed ({e}) — cold render"
                    );
                    prompt_tokens.clone()
                }
            };
            Some(plan_from_rendered(
                &m.conversation_tokens,
                rendered,
                cache_eligible,
                &dflash_ckpt_positions,
                dflash_resume_enabled,
                if spec_name == "mtp" {
                    "mtp-jinja"
                } else {
                    "dflash-jinja"
                },
            ))
        } else {
            None
        }
    } else {
        messages_history.map(|hist| {
            let tok = m.tokenizer.as_ref().unwrap();
            plan_prompt_cache(
                tok,
                &mut m.asst_turn_cache,
                &m.conversation_tokens,
                m.eviction.is_none(),
                system_prompt,
                prompt,
                assistant_prefix,
                qwen_history_tool_render(&m.model_path),
                hist,
                cache_disabled,
                &dflash_ckpt_positions,
                dflash_resume_enabled,
            )
        })
    };
    let resume_from: Option<usize> = cache_plan.as_ref().and_then(|p| p.resume_from);
    // `prompt_tokens` becomes the full canonical conversation when the cache
    // plan rendered it (keeps the end-of-turn `conversation_tokens` bake and the
    // next turn's LCP byte-consistent). Otherwise keep the jinja/ChatFrame build.
    let prompt_tokens: Vec<u32> = match &cache_plan {
        Some(p) => p.rendered.clone(),
        None => prompt_tokens,
    };
    let (prefill_tokens, prefill_start, cache_hit, cached_tokens_dflash): (
        Vec<u32>,
        usize,
        bool,
        usize,
    ) = match &cache_plan {
        Some(p) => (
            p.new_tokens.clone(),
            p.start_pos,
            p.cache_hit,
            p.cached_tokens,
        ),
        None => (prompt_tokens.clone(), 0, false, 0),
    };

    // ── Grammar-guided decoding setup (dflash path) ─────────────
    //
    // qwen35 enforces tool-call grammar POST-acceptance inside the emitter
    // (`Qwen35Emit::observe`); the emitter now extracts its own `ToolSchema`
    // list from the raw tool JSON inside `make_spec_emitter`. This wrapper only
    // honors the `HIPFIRE_QWEN35_GRAMMAR=0` kill-switch by withholding `tools`
    // (⇒ empty schema ⇒ grammar inactive).
    let grammar_enabled = hipfire_runtime::prompt_frame::qwen35_grammar_on(
        std::env::var("HIPFIRE_QWEN35_GRAMMAR").ok().as_deref(),
        &m.model_path,
    );
    let emit_tools: Option<Vec<serde_json::Value>> = if grammar_enabled {
        tools.map(|t| t.to_vec())
    } else {
        None
    };

    // The decode core (slot guard, prefill, accept-window loop, bake, finish) is
    // the arch-generic `generate_spec`. This wrapper owns the qwen35/llama-specific
    // prologue (jinja/Plain render + LCP cache plan), the emitter recipe
    // (`EmitSpec::Qwen35`), and the epilogue (asst-turn cache store + `done`
    // envelope). A future ds4 wrapper (Phase 4 T4c-2) builds its DSML render +
    // ds4 cache plan + `EmitSpec::Deepseek4` and writes its own ds4 `done`.
    //
    // Thread the request's sampling into the speculator BEFORE the step loop.
    // SpecRequestConfig is installed once; greedy (temp 0) is unchanged.
    // ngram-mod is greedy MTP only: env opt-in, thinking off (`max_think_tokens==1`).
    if let Some(spec) = m.speculator.as_mut() {
        spec.configure_request(SpecRequestConfig {
            temp,
            top_p,
            top_k,
            min_p,
            cactus_delta,
            rng_seed: request_seed,
            allow_ngram_modifier: spec_name == "mtp"
                && std::env::var("HIPFIRE_MTP_NGRAM").ok().as_deref() == Some("1")
                && temp <= 1e-6
                && max_think_tokens == 1,
        });
    }
    let prefill_tokens_full = prefill_tokens.len();
    // The Jinja prompt can open `<think>` without emitting that token during
    // generation. Prime both the client channel router and the speculative
    // think-budget tracker from the rendered tail, exactly as the AR path does.
    // Advertise semantic-v2 only when this turn's arch has a correlated
    // router-backed DFlash producer (qwen35 / qwen35-vl). Other arches still
    // use whole-output tool extraction and stay on legacy contract.
    emit_gen_start(
        stdout,
        id,
        started_in_think,
        gen_start_contract_version_for_arch(m.arch_id),
    );
    let run = match generate_spec(
        m,
        gpu,
        stdout,
        id,
        prompt_tokens,
        prefill_tokens,
        prefill_start,
        cache_hit,
        resume_from,
        max_tokens,
        SpecEmitRequest {
            im_end: im_end_token,
            tools: emit_tools,
            stop: stop.to_vec(),
            max_think: max_think_tokens,
            assistant_prefix: spec_assistant_prefix(started_in_think),
            think_mode: ThinkMode::NonThink,
            decoded_vocab: None,
        },
        temp,
    ) {
        Some(r) => r,
        // Abort / error early-exit already wrote its own done/error envelope.
        None => return true,
    };
    debug_assert_eq!(run.prefill_tokens_len, prefill_tokens_full);

    // ── parse tool_calls + populate asst_turn_cache ──────────────
    //
    // Qwen DFlash/spec: terminal ToolCalls are held by generate_spec; the
    // producer-authorized visible channel + router finish drive release,
    // finish_reason, and cache store. Length without decoded EOT never
    // releases calls or stores cache. Other arches keep whole-output extract.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let qwen_semantic_v2 = hipfire_loader::carrier_for(m.arch_id)
        .map(|c| c.caps().semantic_contract_version == Some(2))
        .unwrap_or(false);

    // Trim the trailing `<|im_end|>` + newline trailer from streamed_tokens so
    // the cached body slots cleanly between the assistant_prefix and the
    // im_end+nl trailer that `build_cached_history` re-adds on replay
    // (mirrors qwen35 cache writer).
    //
    // Only strip it when it IS the trailer. A turn cut short by max_tokens never
    // emitted `im_end`, so if its last token happens to be a newline that
    // newline is BODY: popping it made the replayed body one token shorter than
    // what KV holds, the re-added `im_end` then landed on the newline's slot, and
    // the LCP stopped one token before `prior_len` — turning a clean hit into a
    // checkpoint resume (measured: 29 s of re-prefill on a 16.5K turn).
    let nl_token = tokenizer.encode("\n");
    let nl_set: std::collections::HashSet<u32> = nl_token.iter().copied().collect();
    let cached_seq: Vec<u32> = qwen_dflash_cache_seq(&run.streamed_tokens, im_end_token, &nl_set);

    let tok_s = if run.total_s > 0.0 {
        run.generated as f64 / run.total_s
    } else {
        0.0
    };
    let decode_tok_s = if run.decode_s > 0.0 {
        run.generated as f64 / run.decode_s
    } else {
        0.0
    };
    // New-token count (not full rendered length) so the prefill rate reflects
    // actual work on a cache HIT/resume — matches every other path's numerator.
    let prefill_tok_s = if run.prefill_s > 0.0 {
        run.prefill_tokens_len as f64 / run.prefill_s
    } else {
        0.0
    };
    let tau = if run.spec_cycles > 0 {
        run.spec_accepted as f64 / run.spec_cycles as f64
    } else {
        0.0
    };
    // Per PRD §3.1, when PFlash bypassed (e.g. dflash_decode_active for
    // this branch) the `done` object must surface the bypass reason and
    // alpha alongside the dflash perf metrics.
    let pflash_done_field = match (pflash_bypass_reason, pflash_alpha) {
        (Some(r), Some(a)) => format!(
            r#","pflash":{{"bypass_reason":"{}","alpha":{:.6}}}"#,
            r.replace('"', "'"),
            a,
        ),
        _ => String::new(),
    };

    if qwen_semantic_v2 {
        let eos = m
            .tokenizer
            .as_ref()
            .and_then(|t| t.special_token_id("<|endoftext|>"))
            .unwrap_or(0);
        // Prefer emitter-authoritative decoded EOT (split byte markers included);
        // fall back to token-id scan only when the producer left the flag unset.
        // SpecRun.semantic_stop (StopSequence/EOS/ThinkCap) also beats length at
        // the budget boundary — independent of decoded_eot (user stop may not
        // set that flag).
        let decoded_eot = run.finish.decoded_eot
            || qwen_dflash_decoded_eot_from_tokens(
                tokenizer,
                &run.streamed_tokens,
                eos,
                im_end_token,
            );
        let semantic_stop = run.semantic_stop.is_some();
        let hit_length_cap =
            qwen_dflash_hit_length_cap(run.generated, max_tokens, decoded_eot, semantic_stop);
        // Prefer producer-visible channel; fall back to finish Token events.
        let visible = if !run.finish.visible_text.is_empty() {
            run.finish.visible_text.clone()
        } else {
            qwen_dflash_visible_from_finish(&run.finish)
        };
        let rolled_back = run
            .fail_closed_rollback
            .as_ref()
            .map(|ep| ep.rolled_back)
            .unwrap_or(false);
        let terminal = qwen_dflash_wire_terminal(
            &run.finish,
            hit_length_cap,
            run.grammar_violated,
            &visible,
            rolled_back,
        );
        match &terminal {
            QwenDflashWireTerminal::Malformed {
                message,
                class,
                retryable,
                rolled_back: _,
            } => {
                // Prefer the live epilogue from generate_spec (truthful rolled_back
                // + optional sync-failure context). Fall back to a no-GPU attest.
                let fallback = RollbackEpilogue {
                    rolled_back: false,
                    context: None,
                };
                let ep = run.fail_closed_rollback.as_ref().unwrap_or(&fallback);
                // emit_fail_closed_error appends ep.context when !rolled_back.
                emit_qwen_dflash_malformed_terminal(stdout, id, message, class, *retryable, ep);
                // Fail-closed: no done, no cache, no tool release.
                let _ = qwen_dflash_apply_cache_action(
                    |_fp, _seq| {},
                    &qwen_dflash_cache_action(&terminal),
                    cached_seq,
                );
                return true;
            }
            QwenDflashWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
                fingerprint_text: _,
                wire_tool_calls,
            } => {
                // Successful classify → stage exact pending done, then client
                // terminal handshake before any tool release / cache / done.
                let pflash = match (pflash_bypass_reason, pflash_alpha) {
                    (Some(r), Some(a)) => Some((r, a)),
                    _ => None,
                };
                let mut pending_done = qwen_dflash_done_value(
                    id,
                    run.generated,
                    (tok_s * 10.0).round() / 10.0,
                    run.prefill_tokens_len,
                    (run.prefill_s * 1000.0 * 10.0).round() / 10.0,
                    (prefill_tok_s * 10.0).round() / 10.0,
                    (decode_tok_s * 10.0).round() / 10.0,
                    (run.prefill_s * 1000.0 * 10.0).round() / 10.0,
                    (tau * 100.0).round() / 100.0,
                    run.spec_cycles,
                    cached_tokens_dflash,
                    finish_reason,
                    active_attempt_id(),
                );
                if spec_name == "mtp" {
                    if let Some(obj) = pending_done.as_object_mut() {
                        obj.remove("dflash");
                        obj.insert("mtp".to_string(), serde_json::json!(true));
                        if let Some(stats) = m.speculator.as_ref().map(|s| s.request_stats()) {
                            if stats.mtp_ngram {
                                obj.insert("mtp_ngram".into(), serde_json::json!(true));
                                obj.insert(
                                    "ngram_mod_windows".into(),
                                    serde_json::json!(stats.ngram_mod_windows),
                                );
                                obj.insert(
                                    "ngram_mod_drafts".into(),
                                    serde_json::json!(stats.ngram_mod_drafts),
                                );
                                obj.insert(
                                    "ngram_mod_accepted".into(),
                                    serde_json::json!(stats.ngram_mod_accepted),
                                );
                                obj.insert(
                                    "ngram_mod_accept_rate".into(),
                                    serde_json::json!(stats.ngram_mod_accept_rate),
                                );
                                obj.insert(
                                    "mtp_windows".into(),
                                    serde_json::json!(stats.mtp_windows),
                                );
                                obj.insert(
                                    "ar_windows".into(),
                                    serde_json::json!(stats.ar_windows),
                                );
                                obj.insert(
                                    "mtp_retired".into(),
                                    serde_json::json!(stats.mtp_retired),
                                );
                            }
                        }
                    }
                }
                if let Some((reason, alpha)) = pflash {
                    pending_done["pflash"] = serde_json::json!({
                        "bypass_reason": reason,
                        "alpha": alpha,
                    });
                }
                // Stage canonical calls before handshake so pure-tool HTTP
                // terminals are complete at commit_ready. Final done is identical.
                stage_terminal_tool_calls(&mut pending_done, finish_reason, wire_tool_calls);
                let decision = await_client_terminal_commit(stdout, id, &pending_done);
                let effects = qwen_client_commit_effects(
                    decision,
                    *release_tool_calls && !wire_tool_calls.is_empty(),
                    *store_cache,
                );
                if !effects.emit_done {
                    let ep = production_fail_closed_rollback(m, gpu, None, None);
                    emit_spec_cancel_after_rollback(stdout, id, run.generated, &ep);
                    return true;
                }
                // No post-commit tool_calls event; calls live on staged terminal.
                let mut action = qwen_dflash_cache_action(&terminal);
                action.store = effects.store_cache && action.store;
                if action.store {
                    if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
                        eprintln!(
                            "[qwen-cache store dflash] fp_text.len={} tool_calls={} preview={:?}",
                            action.fingerprint_text.len(),
                            action.tool_calls.len(),
                            action.fingerprint_text.chars().take(60).collect::<String>(),
                        );
                    }
                    let _ = qwen_dflash_apply_cache_action(
                        |fp, seq| {
                            if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref()
                                == Some("1")
                            {
                                eprintln!(
                                    "[qwen-cache store dflash] fp={:#018x} cached_seq={}",
                                    fp,
                                    seq.len()
                                );
                            }
                            m.asst_turn_cache.insert(
                                fp,
                                hipfire_runtime::prompt_frame::CachedAssistantTurn {
                                    reasoning: None,
                                    tools: Vec::new(),
                                    content: Some(
                                        hipfire_runtime::prompt_frame::CachedAssistantBody {
                                            token_ids: seq,
                                            text: String::new(),
                                        },
                                    ),
                                },
                            );
                        },
                        &action,
                        cached_seq,
                    );
                }
                emit_staged_terminal_done(stdout, &pending_done);
            }
        }
    } else {
        // Legacy non-qwen DFlash: grammar / open-think / malformed is
        // error-only — never whole-output extract, release held calls,
        // store cache, or emit done. Prefer generate_spec's production
        // epilogue so rolled_back + sync-failure context stay truthful.
        if run.fail_closed_rollback.is_some() || run.grammar_violated {
            let fallback = RollbackEpilogue {
                rolled_back: false,
                context: None,
            };
            let ep = run.fail_closed_rollback.as_ref().unwrap_or(&fallback);
            let message = if run.grammar_violated {
                "grammar violation during speculative decode"
            } else if run.finish.open_think || run.finish.finish_reason == "open_think" {
                "open think span at end of generation (validation)"
            } else if run.finish.finish_reason == "malformed_protocol" {
                "malformed tool protocol"
            } else {
                "fail-closed speculative decode"
            };
            // emit_fail_closed_error appends ep.context when !rolled_back.
            emit_fail_closed_error(stdout, Some(id), message, "validation", false, ep);
            return true;
        }
        // Legacy non-qwen DFlash: whole-output extract + cache only on a safe
        // completed terminal. Length still emits finish_reason=length but must
        // not release held tool calls or store asst_turn_cache (partial/truncated
        // turns are not safe to prime). Fail-closed already returned above.
        let decoded_full = tokenizer.decode(&run.streamed_tokens);
        let emit_tool_calls = extract_tool_calls_from_text(&decoded_full);
        // Semantic stop / decoded_eot at the budget boundary is stop/tool_calls,
        // not length — same rule as the qwen_semantic_v2 path.
        let hit_length_cap = qwen_dflash_hit_length_cap(
            run.generated,
            max_tokens,
            run.finish.decoded_eot,
            run.semantic_stop.is_some(),
        );
        let finish_reason = if hit_length_cap {
            "length"
        } else if !emit_tool_calls.is_empty() {
            "tool_calls"
        } else {
            "stop"
        };
        // Prefer held finish ToolCalls when present; whole-output extract is fallback.
        let wire_calls = {
            let held = finish_summary_held_tool_calls(&run.finish);
            if !held.is_empty() {
                held
            } else {
                emit_tool_calls
            }
        };
        let mut pending_done = serde_json::json!({
            "type": "done",
            "id": id,
            "tokens": run.generated,
            "tok_s": (tok_s * 10.0).round() / 10.0,
            "prefill_tokens": run.prefill_tokens_len,
            "prefill_ms": ((run.prefill_s * 1000.0) * 10.0).round() / 10.0,
            "prefill_tok_s": (prefill_tok_s * 10.0).round() / 10.0,
            "decode_tok_s": (decode_tok_s * 10.0).round() / 10.0,
            "ttft_ms": ((run.prefill_s * 1000.0) * 10.0).round() / 10.0,
            "dflash": true,
            "tau": (tau * 100.0).round() / 100.0,
            "cycles": run.spec_cycles,
            "cached_tokens": cached_tokens_dflash,
            "finish_reason": finish_reason,
            "attempt_id": active_attempt_id(),
        });
        if spec_name == "mtp" {
            if let Some(obj) = pending_done.as_object_mut() {
                obj.remove("dflash");
                obj.insert("mtp".to_string(), serde_json::json!(true));
                if let Some(stats) = m.speculator.as_ref().map(|s| s.request_stats()) {
                    if stats.mtp_ngram {
                        obj.insert("mtp_ngram".into(), serde_json::json!(true));
                        obj.insert(
                            "ngram_mod_windows".into(),
                            serde_json::json!(stats.ngram_mod_windows),
                        );
                        obj.insert(
                            "ngram_mod_drafts".into(),
                            serde_json::json!(stats.ngram_mod_drafts),
                        );
                        obj.insert(
                            "ngram_mod_accepted".into(),
                            serde_json::json!(stats.ngram_mod_accepted),
                        );
                        obj.insert(
                            "ngram_mod_accept_rate".into(),
                            serde_json::json!(stats.ngram_mod_accept_rate),
                        );
                        obj.insert("mtp_windows".into(), serde_json::json!(stats.mtp_windows));
                        obj.insert("ar_windows".into(), serde_json::json!(stats.ar_windows));
                        obj.insert("mtp_retired".into(), serde_json::json!(stats.mtp_retired));
                    }
                }
            }
        }
        if !pflash_done_field.is_empty() {
            let padded = format!("{{{}}}", pflash_done_field.trim_start_matches(','));
            if let Ok(serde_json::Value::Object(map)) =
                serde_json::from_str::<serde_json::Value>(&padded)
            {
                for (k, v) in map {
                    pending_done[k] = v;
                }
            }
        }
        stage_terminal_tool_calls(&mut pending_done, finish_reason, &wire_calls);
        let decision = await_client_terminal_commit(stdout, id, &pending_done);
        let intended_release = finish_reason == "tool_calls" && !wire_calls.is_empty();
        let intended_store = !hit_length_cap && !cached_seq.is_empty();
        let effects = qwen_client_commit_effects(decision, intended_release, intended_store);
        if !effects.emit_done {
            let ep = production_fail_closed_rollback(m, gpu, None, None);
            emit_spec_cancel_after_rollback(stdout, id, run.generated, &ep);
            return true;
        }
        // Safe stop/tool_calls only — length never stores; Abort suppressed above.
        if effects.store_cache {
            let stripped = strip_think_for_fingerprint(&decoded_full);
            let emit_text =
                hipfire_runtime::tokenizer::maybe_normalize_prompt(&stripped).into_owned();
            let fp = asst_turn_fingerprint(&emit_text, &wire_calls);
            if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
                eprintln!(
                    "[qwen-cache store dflash] fp={:#018x} cached_seq={} emit_text.len={} tool_calls={} preview={:?}",
                    fp, cached_seq.len(), emit_text.len(), wire_calls.len(),
                    emit_text.chars().take(60).collect::<String>(),
                );
            }
            m.asst_turn_cache.insert(
                fp,
                hipfire_runtime::prompt_frame::CachedAssistantTurn {
                    reasoning: None,
                    tools: Vec::new(),
                    content: Some(hipfire_runtime::prompt_frame::CachedAssistantBody {
                        token_ids: cached_seq,
                        text: String::new(),
                    }),
                },
            );
        }
        emit_staged_terminal_done(stdout, &pending_done);
    }
    let _ = stdout.flush();
    // Per-request debug summary (stderr → serve.log): active drafter, τ, tok/s.
    let drafter = m.speculator.as_ref().map(|s| s.name()).unwrap_or("none");
    eprintln!(
        "[req {id}] drafter={drafter} tau={tau:.2} tok/s={decode_tok_s:.1} decode ({} tok, {} windows)",
        run.generated, run.spec_cycles
    );
    true
}

/// Arch-generic spec-decode core extracted from `generate_dflash` (Phase 4 T4a).
/// Drives any `Speculator` (`m.speculator`) + `SpecTarget` (via `spec_target_guard`)
/// + `SpecEmit` through one prefill → accept-window loop → bake → done. The caller
/// (`generate_dflash` for qwen35/llama today) prepares the arch-specific inputs:
/// the already-rendered `prompt_tokens`, the LCP cache decision
/// (`prefill_tokens`/`prefill_start`/`cache_hit`/`resume_from`), and the emitter
/// recipe (`EmitSpec`). It returns a [`SpecRun`] summary from which the wrapper
/// writes its arch-specific `done` envelope + cache store; `None` on the
/// abort/error early-exits (which already wrote their own done/error).
/// T4c-2 adds the deepseek4 wrapper + `EmitSpec::Deepseek4` variant.
#[allow(clippy::too_many_arguments)]
pub fn generate_spec(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt_tokens: Vec<u32>,
    prefill_tokens: Vec<u32>,
    prefill_start: usize,
    cache_hit: bool,
    resume_from: Option<usize>,
    max_tokens: usize,
    emit_req: SpecEmitRequest,
    // Request sampling temperature. >0 only reaches here for speculators that
    // report `supports_temp_verify()` (qwen35 DFlash ddtree → SWOR); greedy
    // drafters ignore it. The daemon's routing gate enforces that invariant.
    temp: f32,
) -> Option<SpecRun> {
    // Zero-budget reject: no first token, no prefill/GPU/state/client mutation.
    // Correlated validation error only — wrapper sees None and skips done/cache.
    if max_tokens == 0 {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "max_tokens must be > 0",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return None;
    }

    // Adaptive KV has no maybe_downshift on the generic spec path. Fail closed
    // rather than run unsupported speculation past floor-reserved capacity.
    if m.kv_adaptive.is_some() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "kv_adaptive cannot use generic speculative decode (DFlash/DSpark/MTP/n-gram); use AR",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return None;
    }

    let tokenizer = m.tokenizer.as_ref().unwrap();

    // Acquire the target via the RAII slot guard — it restores the bundle into
    // m.state on EVERY exit path (return, `?`, panic), which structurally
    // eliminates the eight hand-written reconstruction sites that were the
    // #462 cross-request state-bleed class. `m.speculator`, `m.state`,
    // `m.seq_pos`, `m.conversation_tokens` and `m.eviction` are disjoint fields,
    // so the guard, the speculator borrow, and the bookkeeping below coexist.
    let (block_size, ctx_capacity) = match m.speculator.as_ref() {
        Some(s) => (s.block_size(), s.ctx_capacity()),
        None => {
            emit_active_attempt_error(
                stdout,
                Some(id),
                "dflash path entered without a loaded speculator",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return None;
        }
    };
    // Resolve the arch's carrier once — the single dispatch the spec path routes
    // through for BOTH the target borrow and the emitter (the daemon never
    // arch-matches for spec-decode). `&'static dyn Carrier` borrows nothing from
    // `m`, so it coexists with the `tokenizer`/`&mut m.state` borrows below.
    let arch_id = m.arch_id;
    let carrier = match hipfire_loader::carrier_for(arch_id) {
        Some(c) => c,
        None => {
            emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("no carrier for arch_id {}", arch_id),
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return None;
        }
    };
    // Arch-dispatched target borrow via `Carrier::spec_target_guard()`
    // (`m.model_path` is a disjoint field → no borrow conflict with the
    // `&mut m.state` the guard takes). qwen35 moves the bundle out + reopens its
    // HfqFile (restored on Drop); the pure-attention arms borrow in place. The
    // boxed `SpecTargetGuard` yields `&mut dyn SpecTarget` either way.
    let mut guard = match carrier.spec_target_guard(&mut m.state, &m.model_path) {
        Ok(g) => g,
        Err(e) => {
            emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("{}", e),
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return None;
        }
    };
    let spec = m.speculator.as_mut().unwrap();

    // Divergent-render RESUME: restore the drafter-local + target recurrent
    // state to the latest checkpoint ≤ ckpt and drop the now-stale tail of the
    // checkpoint ring (`rewind_to` does both), then rewind the daemon's seq_pos
    // / conversation_tokens. The turn then proceeds exactly like a HIT with
    // start_pos == ckpt (the cache plan already set cache_hit=true).
    if let Some(ckpt) = resume_from {
        let slot = match guard.slot() {
            Ok(s) => s,
            Err(e) => {
                emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!("{}", e),
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return None;
            }
        };
        if let Err(e) = spec.rewind_to(gpu, slot, ckpt) {
            // Partial GPU restore must not continue with host checkpoint
            // coordinates. Fail-closed: live rollback + one correlated error.
            let msg = format!("rewind_to: {e}");
            let ep = production_fail_closed_rollback_live(
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                gpu,
                slot,
                spec.as_mut(),
            );
            emit_fail_closed_error(stdout, Some(id), &msg, "validation", false, &ep);
            return None;
        }
        m.seq_pos = ckpt;
        m.conversation_tokens.truncate(ckpt);
    }

    if !cache_hit {
        // Fresh target state — full prefill from position 0. The DeltaNet
        // recurrent state is zeroed by the prefill seed itself
        // (`seed_target_hidden_from_prompt_abortable` calls `target.reset_state`,
        // which also zeroes s_ef_residual — more complete than the memset loop
        // the old inline path ran here), so only the daemon-side position
        // bookkeeping remains.
        m.seq_pos = 0;
        m.conversation_tokens.clear();
    } else if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
        eprintln!(
            "[qwen-cache HIT dflash] reuse prefix={} suffix={} (no reset)",
            prefill_start,
            prefill_tokens.len()
        );
    }

    let t0 = Instant::now();
    // Capacity checks. With eviction enabled the advertised context window is
    // effectively unbounded (eviction fires between spec cycles), but the
    // *prompt* must still fit in one physical_cap span because the prompt seed
    // writes it per-token without chunking. Error returns just `return` — the
    // slot guard restores the bundle into m.state on the way out.
    let eff_prompt_cap = if m.eviction.is_some() {
        m.physical_cap
    } else {
        ctx_capacity
    };
    if prompt_tokens.len().saturating_add(block_size) > eff_prompt_cap {
        emit_active_attempt_error(
            stdout,
            Some(id),
            &format!(
                "prompt+block_size exceeds {} {} (eviction {})",
                if m.eviction.is_some() {
                    "physical_cap"
                } else {
                    "ctx_capacity"
                },
                eff_prompt_cap,
                if m.eviction.is_some() { "on" } else { "off" },
            ),
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return None;
    }
    if m.eviction.is_none()
        && prompt_tokens
            .len()
            .saturating_add(max_tokens)
            .saturating_add(block_size)
            > ctx_capacity
    {
        emit_active_attempt_error(
            stdout,
            Some(id),
            &format!(
                "prompt+max_tokens exceeds ctx_capacity {} (enable cask_sidecar for long decode)",
                ctx_capacity,
            ),
            "context_length",
            false,
            false,
        );
        let _ = stdout.flush();
        return None;
    }

    // Prefill: the speculator seeds the target's hidden state (advancing its KV
    // + recurrent state), primes the drafter's cached target-hidden, snapshots
    // the divergent-render checkpoint ring, and returns the target's first
    // token. On a cache hit only the suffix is seeded; on a miss the seed
    // self-resets target state and the full prompt is seeded. Client cancel is
    // surfaced as `PrefillOutcome::Aborted`.
    let id_for_abort = id.to_string();
    let slot = match guard.slot() {
        Ok(s) => s,
        Err(e) => {
            emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("{}", e),
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return None;
        }
    };
    let prefill_outcome = spec.prefill(
        gpu,
        slot,
        &prompt_tokens,
        &prefill_tokens,
        prefill_start,
        cache_hit,
        resume_from,
        &|| check_abort(&id_for_abort),
    );
    let first_token = match prefill_outcome {
        Ok(PrefillOutcome::Ready { first_token }) => first_token,
        Ok(PrefillOutcome::Aborted) => {
            // Fail-closed cancel: reset+sync first; terminal follows attestation.
            let ep = production_fail_closed_rollback_live(
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                gpu,
                slot,
                spec.as_mut(),
            );
            emit_spec_cancel_after_rollback(stdout, id, 0, &ep);
            return None;
        }
        Err(e) => {
            let msg = format!("prefill: {}", e);
            let ep = production_fail_closed_rollback_live(
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                gpu,
                slot,
                spec.as_mut(),
            );
            emit_fail_closed_error(stdout, Some(id), &msg, "validation", false, &ep);
            return None;
        }
    };

    // serve-fault-inject: fire after prefill mutated GPU/KV/recurrent and
    // before first_token begin / any visible event. Host fields only —
    // guard holds m.state; spec holds m.speculator.
    #[cfg(feature = "serve-fault-inject")]
    if maybe_inject_fault_after_prefill_dflash(
        arch_id,
        &mut m.seq_pos,
        &mut m.conversation_tokens,
        &mut m.prefill_checkpoints,
        &mut m.dflash_checkpoints,
        &mut m.asst_turn_cache,
        gpu,
        stdout,
        id,
        slot,
        spec.as_mut(),
    ) {
        drop(guard);
        return None;
    }

    let t_prefill = Instant::now();

    // Per-token emission (EosFilter byte stream + grammar + think force-close +
    // stop-sequence match) lives behind the arch-generic `SpecEmit` seam. The
    // emitter is built HERE (not by the caller) because it needs `slot.eos_token()`
    // (only available after the slot guard) AND borrows the tokenizer (derived
    // from `m`). The wrapper supplies the model-independent recipe as an owned
    // `SpecEmitRequest`; the arch's carrier turns it into the concrete
    // `Box<dyn SpecEmit>` (extracting its own grammar schema from `tools`).
    let emit_ctx = hipfire_runtime::spec::SpecEmitCtx {
        tokenizer,
        eos: slot.eos_token(),
        im_end: emit_req.im_end,
        tools: emit_req.tools.as_deref(),
        stop: emit_req.stop,
        max_think: emit_req.max_think,
        max_tokens,
        assistant_prefix: emit_req.assistant_prefix,
        think_mode: emit_req.think_mode,
        decoded_vocab: emit_req.decoded_vocab,
    };
    let mut emit: Box<dyn SpecEmit> = match carrier.make_spec_emitter(emit_ctx) {
        Ok(e) => e,
        Err(e) => {
            emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("{}", e),
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return None;
        }
    };

    // Decode loop — spec.step returns one acceptance window (SpecStep) per cycle.
    // `emitted` is the speculator's repeat / n-gram context (NOT emission state);
    // it stays in the loop and excludes any grammar-rejected token.
    // `raw_decode` is every position-advancing token after the prefill seed
    // (includes empty-event EOS observes). Used for mid-window prefix realign
    // so GPU state is replayed from the exact raw consumed sequence.
    // State-committable decode tokens only (excludes DS4 empty-event EOS).
    // Prefill first_token is pushed after `begin` if its outcome is event-bearing.
    let mut emitted: Vec<u32> = Vec::new();
    let mut raw_decode: Vec<u32> = Vec::new();
    let mut position = prompt_tokens.len();
    let mut seed_token = first_token;
    // τ accounting, inlined from the unified `SpecStep` (the old `SpecStats`
    // type took the arch-specific `SpecStepResult`, which the daemon no longer
    // sees): τ = accepted drafts / cycle.
    let mut spec_cycles = 0usize;
    let mut spec_accepted = 0usize;
    let mut generated = 0usize;

    // Post-prefill compaction (FlashCASK pattern from dflash_spec_demo).
    // If the prompt already filled past budget+beta, compact once before
    // entering the spec loop so the first spec step writes at physical slot
    // `budget`. compact_offset is maintained on slot.kv_cache; subsequent
    // forwards inside the speculator read it for RoPE phase automatically. The
    // drafter-local hidden cache is compacted to match via `on_evict`.
    if let Some(ref ev) = m.eviction {
        // Eviction is only ever configured for KvCache-backed arches
        // (qwen35/llama); a `Qwen2State`-backed target never reaches here.
        // Missing optional hook is an ordinary fail-closed error (no panic).
        // Keep the kv borrow inside maybe_evict so rollback can reborrow `slot`.
        let evict_outcome = match slot.kv_cache_mut() {
            Some(kv) => ev.maybe_evict(gpu, kv, position),
            None => {
                let _ = classify_evict_failure_wire();
                let ep = production_fail_closed_rollback_live(
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                    &mut m.dflash_checkpoints,
                    &mut m.asst_turn_cache,
                    gpu,
                    slot,
                    spec.as_mut(),
                );
                emit_fail_closed_error(
                    stdout,
                    Some(id),
                    "kv_cache_mut missing (post-prefill)",
                    "validation",
                    false,
                    &ep,
                );
                drop(guard);
                return None;
            }
        };
        match evict_outcome {
            Ok(Some(res)) => {
                let pre_phys = position;
                let compact_offset = slot.kv_cache_mut().map(|kv| kv.compact_offset).unwrap_or(0);
                eprintln!(
                    "[dflash] post-prefill evict: {} -> {} (compact_offset={})",
                    pre_phys, res.new_physical, compact_offset,
                );
                position = res.new_physical;
                if !res.retain_mask.is_empty() {
                    if let Err(e) = spec.on_evict(
                        gpu,
                        &EvictRetain {
                            retain_mask: res.retain_mask,
                            pre_phys,
                        },
                    ) {
                        let _ = classify_evict_failure_wire();
                        let ep = production_fail_closed_rollback_live(
                            &mut m.seq_pos,
                            &mut m.conversation_tokens,
                            &mut m.prefill_checkpoints,
                            &mut m.dflash_checkpoints,
                            &mut m.asst_turn_cache,
                            gpu,
                            slot,
                            spec.as_mut(),
                        );
                        emit_fail_closed_error(
                            stdout,
                            Some(id),
                            &format!("on_evict (post-prefill): {e}"),
                            "validation",
                            false,
                            &ep,
                        );
                        drop(guard);
                        return None;
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                let _ = classify_evict_failure_wire();
                let ep = production_fail_closed_rollback_live(
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                    &mut m.dflash_checkpoints,
                    &mut m.asst_turn_cache,
                    gpu,
                    slot,
                    spec.as_mut(),
                );
                emit_fail_closed_error(
                    stdout,
                    Some(id),
                    &format!("maybe_evict (post-prefill): {e}"),
                    "validation",
                    false,
                    &ep,
                );
                drop(guard);
                return None;
            }
        }
    }

    // Emit the first token immediately so TTFT is the prefill time. `begin`
    // pushes + filters it, seeds the grammar matcher, and reports whether the
    // first token is itself a terminator (→ skip the spec loop entirely, so
    // spec_step_dflash never drafts a whole block seeded on a terminal token).
    let first_begin = emit.begin(first_token);
    render_client_events(
        stdout,
        id,
        &first_begin.events,
        t0.elapsed().as_millis() as u64,
        false,
    );

    // Count/bake the first token only when the emitter emitted it (same guard
    // as the accept loop). qwen35 always emits a `Committed`; the ds4 emitter
    // returns no events for an EOS-first prefill argmax — leave history at the
    // prompt and mark the pending seed non-committable so terminal flush skips.
    let mut pending_seed_committable = spec_outcome_seed_committable(&first_begin);
    if pending_seed_committable {
        emitted.push(first_token);
        generated += 1;
    }
    let mut first_token_is_eos = first_begin.stop.is_some();
    // Sticky semantic terminal from begin (StopSequence/EOS/ThinkCap). Carried
    // on SpecRun so wrappers classify stop/tool_calls over pure length when
    // generated == max_tokens (e.g. max_tokens=1 first-token user stop).
    let mut semantic_stop: Option<StopReason> = if spec_stop_is_semantic(first_begin.stop) {
        first_begin.stop
    } else {
        None
    };

    // Begin-triggered forced continuation (e.g. think-budget force-close on the
    // prefill first token). Must run before any speculative `spec.step` so
    // max_tokens=1 cannot spend its remaining budget on an extra model step.
    // Shares [`apply_spec_forced_pending_seed`] with the mid-window path.
    // Runs after first-seed emitted/pending_seed_committable init (DS4 empty-EOS).
    // Stopped also skips every subsequent speculative step.
    {
        let forced_begin = emit.take_forced();
        if !forced_begin.is_empty() {
            match apply_spec_forced_pending_seed(
                &forced_begin,
                &mut generated,
                max_tokens,
                &mut seed_token,
                &mut position,
                &mut raw_decode,
                &mut emitted,
                &mut pending_seed_committable,
                emit.as_mut(),
                gpu,
                slot,
                spec.as_mut(),
                stdout,
                id,
                t0,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                m.physical_cap,
                m.eviction.as_ref(),
            ) {
                SpecForcedApplyResult::Terminal => {
                    drop(guard);
                    return None;
                }
                SpecForcedApplyResult::Stopped(reason) => {
                    // Forced stop terminates the turn: no later force, no spec.step.
                    if semantic_stop.is_none() && spec_stop_is_semantic(Some(reason)) {
                        semantic_stop = Some(reason);
                    }
                    first_token_is_eos = true;
                }
                SpecForcedApplyResult::Applied | SpecForcedApplyResult::Skipped => {}
            }
        }
    }

    // (The DFlash RNG cell and the chain-vs-tree resolution that used to live
    // here are now resolved once at build time and
    // owned inside the speculator — see `build_dflash_speculator`.)

    // Fast path exit conditions (mirrors the dflash_spec_demo outer loop).
    // `!first_token_is_eos` short-circuits the entire spec loop when the prefill's
    // first sampled token was already a terminator (see the guard above).
    while !first_token_is_eos && generated < max_tokens {
        // Decode-side abort (dflash path). See the matching block in
        // `generate()` for rationale. Without this, a Pi cancel
        // mid-decode leaves the spec-decode loop running for max_tokens
        // worth of wasted work.
        if check_abort(id) {
            // Mid-decode cancel: reset+sync first; terminal follows attestation.
            let ep = production_fail_closed_rollback_live(
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                gpu,
                slot,
                spec.as_mut(),
            );
            emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
            return None;
        }
        if position.saturating_add(block_size) >= ctx_capacity {
            break;
        }

        // One acceptance window. The speculator owns chain-vs-tree dispatch
        // internally; the daemon just hands it the borrowed target and
        // the prior committed tokens (drafter repeat / n-gram context). The
        // in-step grammar mask comes from the emitter: qwen35 returns `None`
        // (post-hoc grammar in `observe`); a ds4 emitter returns its erased
        // matcher so the fused step constrains drafts in-place. `emit.grammar()`'s
        // borrow ends when `step` returns, before the per-token `emit.observe`.
        let max_emit = max_tokens.saturating_sub(generated);
        let step = match spec.step(
            gpu,
            slot,
            position,
            seed_token,
            &emitted,
            emit.grammar(),
            temp,
            max_emit,
        ) {
            Ok(s) => s,
            Err(e) => {
                // Authoritative failure: reset+sync first, then one correlated
                // error; no finish/calls/cache/done (return None before epilogue).
                let ep = production_fail_closed_rollback_live(
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                    &mut m.dflash_checkpoints,
                    &mut m.asst_turn_cache,
                    gpu,
                    slot,
                    spec.as_mut(),
                );
                emit_spec_failure_terminal(stdout, id, "spec_step", &e.to_string(), &ep);
                drop(guard);
                return None;
            }
        };
        spec_cycles += 1;
        spec_accepted += step.accepted;
        // `emit` is already the committed tail with the seed re-echo stripped.
        let committed_tail: Vec<u32> = step.emit.to_vec();
        // Absolute host cursor before this window's commit. Spec.step left
        // target/drafter advanced over the FULL emit; semantic observe may
        // stop on a strict prefix (EOT/stop/forced/budget) and must not keep
        // the unobserved tail in host or GPU state.
        let position_before = position;

        let mut hit_eos = false;
        let mut think_cap_hit = false;
        let mut forced_after: Vec<u32> = Vec::new();
        // Tokens of `committed_tail` the emitter actually observed (not
        // grammar-rejected, not past max_tokens). May be a strict prefix.
        let mut consumed = 0usize;
        for &tok in &committed_tail {
            if generated >= max_tokens {
                break;
            }
            // `emit.observe` runs the per-token emission policy: grammar
            // pre-check (reject → grammar violation, NO emit, post-loop forces a
            // full KV/DN reset to clear the polluted slots spec_step wrote), then
            // committed/token frames, EOS, user stop-sequence, and think
            // force-close. Loop state (emitted/generated) stays here; position
            // advances by `consumed` only after the loop (never by unobserved tail).
            emit.set_generated_hint(generated);
            let outcome = emit.observe(tok);
            if outcome.stop == Some(StopReason::GrammarViolation) {
                // Rejected before emit — not consumed, not added to the repeat
                // context, not streamed, not counted. Treat as EOS for this turn.
                render_client_events(
                    stdout,
                    id,
                    &outcome.events,
                    t0.elapsed().as_millis() as u64,
                    false,
                );
                hit_eos = true;
                break;
            }
            // Count/bake/render a committed token only when the emitter actually
            // emitted it (non-empty events). qwen35 always emits a `Committed`
            // event (even on EOS / held bytes), so this is a no-op there; the
            // deepseek4 emitter returns NO events for an accepted EOS, so that
            // terminator is neither counted into `generated` nor baked into
            // `conversation_tokens` — byte-matching the bespoke ds4 loop (which
            // broke on accepted-EOS before push/increment). Track whether the
            // observed token is state-committable: it becomes the window's
            // pending seed (bonus / last consumed), so empty-event EOS must not
            // terminal-flush or history-bake.
            pending_seed_committable = spec_outcome_seed_committable(&outcome);
            if pending_seed_committable {
                emitted.push(tok);
                render_client_events(
                    stdout,
                    id,
                    &outcome.events,
                    t0.elapsed().as_millis() as u64,
                    false,
                );
                generated += 1;
            }
            // Observed (including empty-event EOS): counts toward the consumed
            // prefix and the position-advancing raw decode log.
            consumed += 1;
            raw_decode.push(tok);
            // Generation-intervention hook: the emitter may request tokens to
            // FORCE after this one, suppressing the step's terminator (cohere2moe
            // empty-turn guard / think-budget force-close). Default empty for
            // every other emitter ⇒ this branch never taken, loop byte-identical.
            // Forced advance runs only after host/GPU are aligned to the consumed
            // prefix (never from an ahead-of-history full-window cursor).
            let forced = emit.take_forced();
            if !forced.is_empty() {
                forced_after = forced;
                break;
            }
            match outcome.stop {
                Some(StopReason::Eos) | Some(StopReason::StopSequence) => {
                    if semantic_stop.is_none() {
                        semantic_stop = outcome.stop;
                    }
                    hit_eos = true;
                    break;
                }
                Some(StopReason::ThinkCap) => {
                    if semantic_stop.is_none() {
                        semantic_stop = outcome.stop;
                    }
                    think_cap_hit = true;
                    break;
                }
                Some(StopReason::GrammarViolation) => unreachable!("handled above"),
                None => {}
            }
        }
        // Host cursor: advance by the observed prefix only (C8 pure helper).
        // Inline emitted/generated already reflect event-nonempty tokens; the
        // helper is authoritative for position + next seed from `consumed`.
        // When consumed==0 the helper falls back to step.next_seed — override
        // with the pre-window seed so we never install an unobserved bonus.
        let host = spec_host_advance_after_step(
            position_before,
            0,
            Vec::new(),
            &committed_tail,
            step.next_seed,
            consumed,
        );
        position = host.position;
        seed_token = if consumed == 0 {
            // No new decode tokens observed (budget/grammar before first tok).
            raw_decode.last().copied().unwrap_or(first_token)
        } else {
            host.seed_token
        };

        // Strict-prefix semantic stop: drop unobserved speculative tail from
        // target + drafter via conservative reset + production prefill of the
        // exact KV-resident prefix (`spec_prefix_realign_plan`). Full-window
        // observe keeps the step's already-committed GPU state.
        //
        // Capacity-aware: admit BEFORE reset/prefill. Realign is full-history
        // replay after reset (compact_offset cleared) — never overrun
        // physical_cap/ctx_capacity and never silently reconstruct compacted
        // state from an invalid oversize history. Abort/prefill errors share
        // the single fail-closed terminal (no second done/error).
        let keep = consumed.min(committed_tail.len());
        if keep < committed_tail.len() {
            let plan = spec_prefix_realign_plan(&prompt_tokens, first_token, &raw_decode);
            let compact_offset = slot.kv_cache_mut().map(|kv| kv.compact_offset).unwrap_or(0);
            if let Err(msg) = spec_prefix_realign_admit(
                &plan,
                m.physical_cap,
                ctx_capacity,
                compact_offset,
                m.eviction.is_some(),
            ) {
                let ep = production_fail_closed_rollback_live(
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                    &mut m.dflash_checkpoints,
                    &mut m.asst_turn_cache,
                    gpu,
                    slot,
                    spec.as_mut(),
                );
                emit_fail_closed_error(stdout, Some(id), &msg, "validation", false, &ep);
                drop(guard);
                return None;
            }
            let reset_error = slot
                .reset_recurrent(gpu)
                .err()
                .map(|e| format!("reset_recurrent: {e}"))
                .or_else(|| {
                    spec.reset_for_realign(gpu)
                        .err()
                        .map(|e| format!("spec.reset_for_realign: {e}"))
                });
            if let Some(msg) = reset_error {
                let ep = production_fail_closed_rollback_live(
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                    &mut m.dflash_checkpoints,
                    &mut m.asst_turn_cache,
                    gpu,
                    slot,
                    spec.as_mut(),
                );
                emit_fail_closed_error(
                    stdout,
                    Some(id),
                    &format!("prefix realign reset failed: {msg}"),
                    "gpu",
                    true,
                    &ep,
                );
                drop(guard);
                return None;
            }
            let id_for_realign = id.to_string();
            let realign = spec.prefill(
                gpu,
                slot,
                &plan.replay,
                &plan.replay,
                0,
                false,
                None,
                &|| check_abort(&id_for_realign),
            );
            match realign {
                Ok(PrefillOutcome::Ready { first_token: _ }) => {
                    // Prefill argmax is NOT history — host seed/position follow
                    // the pure plan (processed prefix + unwritten pending seed).
                    debug_assert_eq!(plan.replay.len(), plan.position);
                    position = plan.position;
                    seed_token = plan.seed_token;
                }
                Ok(PrefillOutcome::Aborted) => {
                    // One cancel terminal only — classified by rollback attestation.
                    let ep = production_fail_closed_rollback_live(
                        &mut m.seq_pos,
                        &mut m.conversation_tokens,
                        &mut m.prefill_checkpoints,
                        &mut m.dflash_checkpoints,
                        &mut m.asst_turn_cache,
                        gpu,
                        slot,
                        spec.as_mut(),
                    );
                    emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
                    return None;
                }
                Err(e) => {
                    let msg = format!("spec_prefix_realign: {e}");
                    let ep = production_fail_closed_rollback_live(
                        &mut m.seq_pos,
                        &mut m.conversation_tokens,
                        &mut m.prefill_checkpoints,
                        &mut m.dflash_checkpoints,
                        &mut m.asst_turn_cache,
                        gpu,
                        slot,
                        spec.as_mut(),
                    );
                    emit_fail_closed_error(stdout, Some(id), &msg, "validation", false, &ep);
                    drop(guard);
                    return None;
                }
            }
        }

        // Forced-token injection (cohere2moe generation guards; no-op for every
        // other emitter — `take_forced` defaulted empty). Host/GPU are aligned
        // to the observed prefix (full window or realigned strict prefix).
        //
        // Pending-seed transaction via shared [`apply_spec_forced_pending_seed`]
        // (same path as begin-triggered force): hard budget clip, stop-prefix
        // stage/observe before GPU, trigger retained when committable, last
        // forced pending once, abort latch + exclusive terminal,
        // pending_seed_committable from each forced observe. Stopped returns
        // the terminal cause so we break without further force or spec.step.
        // Bounded by the emitter's re-entry guard (e.g. MAX_EOS_SUPPRESS).
        if !forced_after.is_empty() {
            let forced_all = std::mem::take(&mut forced_after);
            match apply_spec_forced_pending_seed(
                &forced_all,
                &mut generated,
                max_tokens,
                &mut seed_token,
                &mut position,
                &mut raw_decode,
                &mut emitted,
                &mut pending_seed_committable,
                emit.as_mut(),
                gpu,
                slot,
                spec.as_mut(),
                stdout,
                id,
                t0,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                m.physical_cap,
                m.eviction.as_ref(),
            ) {
                SpecForcedApplyResult::Terminal => {
                    drop(guard);
                    return None;
                }
                SpecForcedApplyResult::Stopped(reason) => {
                    if semantic_stop.is_none() && spec_stop_is_semantic(Some(reason)) {
                        semantic_stop = Some(reason);
                    }
                    match reason {
                        StopReason::ThinkCap => think_cap_hit = true,
                        StopReason::Eos
                        | StopReason::StopSequence
                        | StopReason::GrammarViolation => hit_eos = true,
                    }
                }
                SpecForcedApplyResult::Applied | SpecForcedApplyResult::Skipped => {}
            }
        }
        // Per-cycle eviction (FlashCASK). Fires whenever current physical
        // has grown to budget+β since the last compaction. No-op when
        // physical < budget+β, so non-firing cycles pay only the check cost.
        if let Some(ref ev) = m.eviction {
            // Eviction ⇒ KvCache-backed target (qwen35/llama); never qwen2.
            // Missing optional hook is an ordinary fail-closed error (no panic).
            // Keep the kv borrow inside maybe_evict so rollback can reborrow `slot`.
            let evict_outcome = match slot.kv_cache_mut() {
                Some(kv) => ev.maybe_evict(gpu, kv, position),
                None => {
                    let _ = classify_evict_failure_wire();
                    let ep = production_fail_closed_rollback_live(
                        &mut m.seq_pos,
                        &mut m.conversation_tokens,
                        &mut m.prefill_checkpoints,
                        &mut m.dflash_checkpoints,
                        &mut m.asst_turn_cache,
                        gpu,
                        slot,
                        spec.as_mut(),
                    );
                    emit_fail_closed_error(
                        stdout,
                        Some(id),
                        "kv_cache_mut missing (per-cycle)",
                        "validation",
                        false,
                        &ep,
                    );
                    drop(guard);
                    return None;
                }
            };
            match evict_outcome {
                Ok(Some(res)) => {
                    let pre_phys = position;
                    position = res.new_physical;
                    if !res.retain_mask.is_empty() {
                        if let Err(e) = spec.on_evict(
                            gpu,
                            &EvictRetain {
                                retain_mask: res.retain_mask,
                                pre_phys,
                            },
                        ) {
                            let _ = classify_evict_failure_wire();
                            let ep = production_fail_closed_rollback_live(
                                &mut m.seq_pos,
                                &mut m.conversation_tokens,
                                &mut m.prefill_checkpoints,
                                &mut m.dflash_checkpoints,
                                &mut m.asst_turn_cache,
                                gpu,
                                slot,
                                spec.as_mut(),
                            );
                            emit_fail_closed_error(
                                stdout,
                                Some(id),
                                &format!("on_evict (per-cycle): {e}"),
                                "validation",
                                false,
                                &ep,
                            );
                            drop(guard);
                            return None;
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    let _ = classify_evict_failure_wire();
                    let ep = production_fail_closed_rollback_live(
                        &mut m.seq_pos,
                        &mut m.conversation_tokens,
                        &mut m.prefill_checkpoints,
                        &mut m.dflash_checkpoints,
                        &mut m.asst_turn_cache,
                        gpu,
                        slot,
                        spec.as_mut(),
                    );
                    emit_fail_closed_error(
                        stdout,
                        Some(id),
                        &format!("maybe_evict (per-cycle): {e}"),
                        "validation",
                        false,
                        &ep,
                    );
                    drop(guard);
                    return None;
                }
            }
        }
        if hit_eos || think_cap_hit {
            break;
        }
    }

    // Snapshot the emitter's state needed by the post-loop bookkeeping before
    // the terminal `finish` consumes it: the full streamed-token stream (for the
    // asst-turn cache) and whether a committed token tripped the grammar matcher.
    let streamed_tokens = emit.streamed_tokens().to_vec();
    let grammar_violated = emit.grammar_violated();

    // Safe-terminal pending-seed flush: every normal exit leaves one unprocessed
    // seed (see SpecPendingSeedTx). Forward it exactly once so KV/DeltaNet/
    // drafter-hidden end on the same logical token as conversation history.
    // Skip on grammar fail-closed (rollback wipes state) and when the pending
    // seed is intentionally non-committable (DS4 empty-event EOS — position
    // still reflects already-processed prior seeds/drafts; no GPU forward /
    // history bake of the terminator). Cancel/error already returned above.
    if spec_should_flush_pending_seed(grammar_violated, pending_seed_committable) {
        let tx = spec_terminal_pending_seed_tx(seed_token);
        if check_abort(id) {
            let _ = classify_forced_gpu_advance(true);
            let ep = production_fail_closed_rollback_live(
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                gpu,
                slot,
                spec.as_mut(),
            );
            emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
            drop(guard);
            return None;
        }
        let abort_latched = std::cell::Cell::new(false);
        let abort_cb = || {
            if check_abort(id) {
                abort_latched.set(true);
                true
            } else {
                false
            }
        };
        let flush_res = match spec.on_forced_advance(gpu, slot, &tx.commit, position, &abort_cb) {
            Ok(true) => Ok(()),
            Ok(false) => match slot.spec_advance(gpu, &tx.commit, position, false, &abort_cb, None)
            {
                Ok(SpecAdvance::Ready { .. }) => Ok(()),
                Ok(SpecAdvance::Aborted) => {
                    abort_latched.set(true);
                    Ok(())
                }
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        };
        if let Err(e) = flush_res {
            let ep = production_fail_closed_rollback_live(
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                gpu,
                slot,
                spec.as_mut(),
            );
            emit_spec_failure_terminal(stdout, id, "forced", &e.to_string(), &ep);
            drop(guard);
            return None;
        }
        let abort_observed = abort_latched.get() || check_abort(id);
        if matches!(
            classify_forced_gpu_advance(abort_observed),
            ForcedGpuAdvanceKind::Cancelled
        ) {
            let ep = production_fail_closed_rollback_live(
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                gpu,
                slot,
                spec.as_mut(),
            );
            emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
            drop(guard);
            return None;
        }
        position = position.saturating_add(tx.position_delta);
        // seed_token remains the flushed token; no further unprocessed seed.
    }

    m.seq_pos = position;
    // Bake the FULL conversation (prefill + decode) into conversation_tokens
    // so subsequent turns can compute LCP against it. Previously this stored
    // only the decoded portion (`emitted`), making the next non-dflash turn
    // full-reset because no system/user prefix was present.
    // Host raw/conversation stay exact even when client events were held.
    m.conversation_tokens = {
        let mut v = Vec::with_capacity(prompt_tokens.len() + emitted.len());
        v.extend_from_slice(&prompt_tokens);
        v.extend_from_slice(&emitted);
        v
    };

    // Grammar-violation cleanup: the speculator wrote KV + DN state for the
    // rejected token(s) before the post-acceptance grammar check saw them.
    // Those slots are now poisoned — force a full reset via the production
    // epilogue so the wrapper can attest truthful `rolled_back`.
    let mut fail_closed_rollback = None;
    if grammar_violated {
        eprintln!("[grammar-dflash] grammar violation — forcing full KV/DN reset for next turn");
        fail_closed_rollback = Some(production_fail_closed_rollback_live(
            &mut m.seq_pos,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            &mut m.dflash_checkpoints,
            &mut m.asst_turn_cache,
            gpu,
            slot,
            spec.as_mut(),
        ));
    }

    // Restore the target bundle into m.state via the slot guard's Drop, before
    // the wrapper's end-of-turn cache bookkeeping (which no longer needs the slot).
    drop(guard);

    // Terminal `finish` flush — parses tool calls from the decoded text and
    // renders the `tool_calls` ClientEvent. The arch-specific epilogue (the
    // asst-turn cache store + the `done` envelope) is the WRAPPER's job: it
    // differs per arch (qwen35: `dflash`/`tau`/`cycles` + ChatML token-replay
    // cache; ds4: `spec_k`/`spec_windows`/`spec_accept_pct`), so this core
    // returns a `SpecRun` summary instead of writing them itself.
    let finish = emit.finish();
    // Open-think / malformed finish reasons also need a truthful rollback when
    // grammar did not already reset (state may still be baked).
    if fail_closed_rollback.is_none()
        && (finish.open_think
            || finish.finish_reason == "open_think"
            || finish.finish_reason == "malformed_protocol")
    {
        // Guard already dropped — reset via host-held bundle/speculator.
        fail_closed_rollback = Some(production_fail_closed_rollback(m, gpu, None, None));
    }
    // Safe Done paths must not release held tool_calls events on fail-closed.
    let release_events = fail_closed_rollback.is_none();
    if release_events {
        render_client_events(stdout, id, &finish.events, 0, true);
    }

    let t_end = Instant::now();
    Some(SpecRun {
        generated,
        spec_cycles,
        spec_accepted,
        streamed_tokens,
        prefill_tokens_len: prefill_tokens.len(),
        finish,
        grammar_violated,
        semantic_stop,
        fail_closed_rollback,
        prefill_s: t_prefill.duration_since(t0).as_secs_f64(),
        total_s: t_end.duration_since(t0).as_secs_f64(),
        decode_s: t_end.duration_since(t_prefill).as_secs_f64(),
    })
}

/// Wire `kind` for one successful MTP decode window. Classified from the route
/// actually taken *before* the step (and before ngram acceptance can retire MTP).
pub fn mtp_window_timing_kind(
    used_ngram: bool,
    mtp_ngram: bool,
    mtp_retired: bool,
) -> &'static str {
    if used_ngram {
        "ngram"
    } else if mtp_ngram && mtp_retired {
        "ar"
    } else {
        "mtp"
    }
}

/// Build one `mtp_window_timings[]` record from already-measured microsecond deltas.
/// Pure: no clocks or launch counters. Field names match the wire schema exactly.
pub fn mtp_window_timing_record(
    kind: &str,
    wall_us: u64,
    draft_lookup_us: u64,
    launch_us: u64,
    h2d_us: u64,
    d2h_us: u64,
    d2d_us: u64,
    memset_us: u64,
    stream_sync_us: u64,
    event_sync_us: u64,
    device_sync_us: u64,
    graph_launch_us: u64,
) -> serde_json::Value {
    serde_json::json!({
        "kind": kind,
        "wall_us": wall_us,
        "draft_lookup_us": draft_lookup_us,
        "launch_us": launch_us,
        "h2d_us": h2d_us,
        "d2h_us": d2h_us,
        "d2d_us": d2d_us,
        "memset_us": memset_us,
        "stream_sync_us": stream_sync_us,
        "event_sync_us": event_sync_us,
        "device_sync_us": device_sync_us,
        "graph_launch_us": graph_launch_us,
    })
}

/// Attach `mtp_window_timings` to the staged `done` object only when host timing
/// is enabled. Disabled path leaves the field absent (not null).
pub fn attach_mtp_window_timings(
    pending_done: &mut serde_json::Value,
    host_timing: bool,
    timings: Vec<serde_json::Value>,
) {
    if host_timing {
        pending_done["mtp_window_timings"] = serde_json::Value::Array(timings);
    }
}

/// Multi-GPU pipeline-parallel AR decode (Stage 7 of #58). Mirrors the pp=1
/// `generate` Qwen3.5 branch feature-for-feature: ChatFrame ChatML wrap,
/// EosFilter UTF-8 streaming + strip-think + stop_at, LoopGuard n-gram
/// detection, repeat penalty, attractor block on unclosed tool/think
/// openers, max_think_tokens force-close, budget-alert nudge, ChatML \n
/// trailer. Forward calls fan out to per-device tensors via
/// `gpus.devices[dev]` and `scratch_set.per_device[dev]`; the final
/// sample lives on `gpus.output_device`. DFlash, CASK, PFlash, VL and
/// arch_id < 5 are refused upstream at load.
#[allow(clippy::too_many_arguments)]
pub fn generate_multi(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    pflash_state: Option<&mut hipfire_pflash::pflash::PflashState>,
    pflash_cfg: Option<&hipfire_pflash::pflash::PflashConfig>,
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
    _repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    budget_alert_at_tok: usize,
    budget_alert_text: &str,
    max_think_tokens: usize,
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
    stop: &[String],
    reasoning_effort: Option<&str>,
    enable_thinking: bool,
    // Per-request sampler seed (see hipfire-engine::request_seed_for). Replaces
    // the historical fixed 0x13579BDF that made PP>1 same-prompt requests
    // byte-identical at temp>0.
    request_seed: u64,
) {
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let prompt_est = tokenizer.encode(prompt).len() + 20;
    if m.seq_pos
        .saturating_add(prompt_est)
        .saturating_add(max_tokens)
        > m.max_seq
    {
        eprintln!(
            "[daemon] context full ({}/{}) — resetting conversation",
            m.seq_pos, m.max_seq
        );
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        free_checkpoints(&mut m.prefill_checkpoints, gpu);
        free_checkpoints(&mut m.dflash_checkpoints, gpu);
        // qwen35 recurrent state lives in the bundle (ModelState::Qwen35), not
        // the always-None m.dn_state/m.kv_cache. Inlined (disjoint field access)
        // because a `&tokenizer` borrow of `m` is live here; covers both the
        // pp>1 per-LA-device path and the single-GPU path.
        if m.pp > 1 {
            if let (Some(b), Some(gpus), Some(la)) = (
                m.state.as_mut().and_then(|s| {
                    (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
                }),
                m.pp_gpus.as_mut(),
                m.pp_dn_la_to_device.as_ref(),
            ) {
                let dn = &b.dn_state;
                for (i, s) in dn.s_matrices.iter().enumerate() {
                    let g = &mut gpus.devices[la[i] as usize];
                    let _ = g.bind_thread();
                    let _ = g.hip.memset(&s.buf, 0, s.buf.size());
                }
                for (i, s) in dn.s_scales.iter().enumerate() {
                    let g = &mut gpus.devices[la[i] as usize];
                    let _ = g.bind_thread();
                    let _ = g.hip.memset(&s.buf, 0, s.buf.size());
                }
                for (i, s) in dn.conv_states.iter().enumerate() {
                    let g = &mut gpus.devices[la[i] as usize];
                    let _ = g.bind_thread();
                    let _ = g.hip.memset(&s.buf, 0, s.buf.size());
                }
                // multi-GPU currently leaves s_ef_residual empty; loop is a no-op then,
                // but keeps single-GPU parity if EF is ever wired per-device.
                for (i, s) in dn.s_ef_residual.iter().enumerate() {
                    let g = &mut gpus.devices[la[i] as usize];
                    let _ = g.bind_thread();
                    let _ = g.hip.memset(&s.buf, 0, s.buf.size());
                }
            }
        } else if let Some(b) = m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) {
            let dn = &b.dn_state;
            for s in &dn.s_matrices {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
            for s in &dn.s_scales {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
            for s in &dn.conv_states {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
            for s in &dn.s_ef_residual {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
        }
        if let Some(b) = m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) {
            b.kv_cache.compact_offset = 0;
        }
        if let Some(ad) = m.kv_adaptive.as_mut() {
            if let Some(b) = m.state.as_mut().and_then(|s| {
                (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
            }) {
                ad.reset_with_cache(gpu, &mut b.kv_cache);
            } else {
                ad.reset();
            }
        }
    }

    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let raw_q_tokens = tokenizer.encode(prompt);

    // PFlash compression on first turn (seq_pos == 0). Drafter runs on the
    // daemon's single-GPU `gpu` handle, which binds to the same physical
    // device as `pp_gpus.devices[0]` (HIP enumerates within ROCR_VISIBLE).
    // VRAM is shared between the two Gpu handles via the HIP heap, so
    // drafter weights coexist with the target's dev 0 portion. Output is
    // a Vec<u32> of kept token IDs which feeds forward_prefill_batch_multi
    // unchanged. Mode=Off / drafter unloaded falls through to raw tokens.
    let request_kind = match tokenizer.special_token_id("<tool_call>") {
        Some(tid) => {
            let in_user = raw_q_tokens.iter().any(|&t| t == tid);
            let in_system = system_prompt
                .map(|s| tokenizer.encode(s).iter().any(|&t| t == tid))
                .unwrap_or(false);
            if in_user || in_system {
                hipfire_pflash::pflash::RequestKind::ToolCall
            } else {
                hipfire_pflash::pflash::RequestKind::Text
            }
        }
        None => hipfire_pflash::pflash::RequestKind::Text,
    };
    let q_tokens = if let (Some(state), Some(cfg)) = (pflash_state, pflash_cfg) {
        if m.seq_pos == 0 {
            match hipfire_pflash::pflash::maybe_compress_prompt(
                gpu,
                state,
                cfg,
                &raw_q_tokens,
                request_kind,
                &[],
            ) {
                Ok(hipfire_pflash::pflash::PflashDecision::Compressed(cp)) => {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"pflash_compressed","id":"{}","source_tokens":{},"kept_tokens":{},"keep_ratio":{:.6},"source_md5":"{}","compressed_md5":"{}","score_ms":{},"total_ms":{}}}"#,
                        id,
                        cp.source_tokens,
                        cp.kept_tokens,
                        cp.kept_tokens as f32 / cp.source_tokens.max(1) as f32,
                        cp.source_md5,
                        cp.compressed_md5,
                        cp.timings.score_ms,
                        cp.timings.total_ms,
                    );
                    let _ = stdout.flush();
                    cp.token_ids
                }
                Ok(hipfire_pflash::pflash::PflashDecision::Bypass { reason }) => {
                    if !matches!(reason, hipfire_pflash::pflash::BypassReason::ModeOff) {
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"pflash_bypass","id":"{}","reason":"{}"}}"#,
                            id,
                            reason.as_str().replace('"', "'"),
                        );
                        let _ = stdout.flush();
                    }
                    raw_q_tokens
                }
                Err(e) => {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"pflash_error","id":"{}","reason":"{}"}}"#,
                        id,
                        e.to_string().replace('"', "'"),
                    );
                    let _ = stdout.flush();
                    raw_q_tokens
                }
            }
        } else {
            raw_q_tokens
        }
    } else {
        raw_q_tokens
    };

    // ChatML framing — two paths, same shape as the single-GPU AR
    // generate() (line 3147+):
    //
    //   1) HIPFIRE_JINJA_CHAT=1 + model has chat_template + seq_pos==0
    //      → render via JinjaChatFrame so structured tools/messages
    //      reach the upstream template. PFlash compression is bypassed
    //      under Jinja (q_tokens is unused; the rendered prompt string
    //      is re-tokenized straight through).
    //
    //   2) Default: hand-rolled ChatFrame::Plain scaffold, byte-
    //      identical to the pp=1 default path so multi-turn behavior
    //      matches between pp=1 and pp>1 when both run the same prompt.
    // LFM2.5 (arch_id 11) REQUIRES its embedded Jinja chat_template — the
    // hand-rolled Plain ChatML path omits LFM2's `<|startoftext|>` BOS and
    // produces garbage. Force jinja on for arch 11 (falls back to Plain only if
    // the .hfq carries no template, e.g. an older A1B convert).
    // Jinja default-ON (flipped 2026-06-09): render through the model's chat
    // template for ALL arches; opt out with HIPFIRE_JINJA_CHAT=0 (hand-rolled
    // ChatML/Plain). Falls back to Plain automatically when no template resolves.
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
    // hunt3 H-A: drop the `seq_pos == 0` gate (PR #389 removed it from generate()).
    // With the gate, turn 2+ fell through to the Plain scaffold, dropping the
    // system prompt and the full history replay that render_messages provides.
    // Now Jinja renders the full conversation every turn; the cold-reset block
    // below (guarded on seq_pos > 0) re-zeros recurrent state so the full render
    // writes from position 0 instead of appending to the prior turn's KV/DeltaNet.
    let try_jinja = jinja_enabled && m.chat_template.is_some();
    // Request flag seeds Plain fallback only. Successful Jinja owns the
    // generation suffix — derive initial think from the rendered tail.
    let mut started_in_think = matches!(
        assistant_prefix,
        hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
    );
    let new_tokens = if try_jinja {
        let template = m.chat_template.as_ref().unwrap();
        let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking,
            bos_token: None,
            reasoning_strength: None,
            reasoning_effort,
        };
        let render_result = if tools.is_some() || messages_history.is_some() {
            let synthesized: Vec<hipfire_runtime::prompt_frame::Message>;
            let messages_slice: &[hipfire_runtime::prompt_frame::Message] = match messages_history {
                Some(m) => m,
                None => {
                    let mut v = Vec::new();
                    if let Some(sys) = system_prompt {
                        v.push(hipfire_runtime::prompt_frame::Message {
                            role: hipfire_runtime::prompt_frame::Role::System,
                            content: sys.to_string(),
                            reasoning_content: None,
                            name: None,
                            rendered_name: None,
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                            tool_plan: String::new(),
                        });
                    }
                    v.push(hipfire_runtime::prompt_frame::Message {
                        role: hipfire_runtime::prompt_frame::Role::User,
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
            Ok(rendered) => {
                started_in_think = render_tail_opens_think(&rendered);
                tokenizer.encode(&rendered)
            }
            Err(e) => {
                if reasoning_effort.is_some() {
                    emit_active_attempt_error(
                        stdout,
                        Some(id),
                        &format!("jinja render: {e}"),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    return;
                }
                eprintln!("[daemon] jinja render failed in pp path ({e}) — falling back to Plain");
                hipfire_runtime::prompt_frame::ChatFrame {
                    tokenizer,
                    system: if m.seq_pos == 0 { system_prompt } else { None },
                    user: "",
                    assistant_prefix,
                    raw: false,
                }
                .build_with_user_tokens(&q_tokens)
            }
        }
    } else {
        hipfire_runtime::prompt_frame::ChatFrame {
            tokenizer,
            system: if m.seq_pos == 0 { system_prompt } else { None },
            user: "",
            assistant_prefix,
            raw: false,
        }
        .build_with_user_tokens(&q_tokens)
    };

    // hunt3 H-A: under Jinja the full conversation (system + history) is
    // re-rendered every turn, so turn 2+ must cold-reset BEFORE the budget guard
    // + prefill — otherwise the full render appends to the prior turn's dirty
    // KV / DeltaNet / checkpoint state (stale recurrent state → drift; the
    // system prompt was also being silently dropped on turn 2+). Mirrors the
    // `reset_pp_uncommitted_state!` semantics, written inline because that macro
    // is defined later (after kv/dn/gpus are borrowed). Same shape as the
    // context-full reset at the top of this fn and generate()'s `jinja_active &&
    // seq_pos > 0` block.
    if try_jinja && m.seq_pos > 0 {
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        free_checkpoints(&mut m.prefill_checkpoints, gpu);
        free_checkpoints(&mut m.dflash_checkpoints, gpu);
        // qwen35 recurrent state lives in the bundle (ModelState::Qwen35), not
        // the always-None m.dn_state/m.kv_cache. Covers pp>1 + single-GPU.
        if m.pp > 1 {
            if let (Some(b), Some(gpus), Some(la)) = (
                m.state.as_mut().and_then(|s| {
                    (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
                }),
                m.pp_gpus.as_mut(),
                m.pp_dn_la_to_device.as_ref(),
            ) {
                let dn = &b.dn_state;
                for (i, s) in dn.s_matrices.iter().enumerate() {
                    let g = &mut gpus.devices[la[i] as usize];
                    let _ = g.bind_thread();
                    let _ = g.hip.memset(&s.buf, 0, s.buf.size());
                }
                for (i, s) in dn.s_scales.iter().enumerate() {
                    let g = &mut gpus.devices[la[i] as usize];
                    let _ = g.bind_thread();
                    let _ = g.hip.memset(&s.buf, 0, s.buf.size());
                }
                for (i, s) in dn.conv_states.iter().enumerate() {
                    let g = &mut gpus.devices[la[i] as usize];
                    let _ = g.bind_thread();
                    let _ = g.hip.memset(&s.buf, 0, s.buf.size());
                }
                // multi-GPU currently leaves s_ef_residual empty; loop is a no-op then,
                // but keeps single-GPU parity if EF is ever wired per-device.
                for (i, s) in dn.s_ef_residual.iter().enumerate() {
                    let g = &mut gpus.devices[la[i] as usize];
                    let _ = g.bind_thread();
                    let _ = g.hip.memset(&s.buf, 0, s.buf.size());
                }
            }
        } else if let Some(b) = m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) {
            let dn = &b.dn_state;
            for s in &dn.s_matrices {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
            for s in &dn.s_scales {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
            for s in &dn.conv_states {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
            for s in &dn.s_ef_residual {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
        }
        if let Some(b) = m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) {
            b.kv_cache.compact_offset = 0;
        }
        if let Some(b) = m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_llama::LlamaBundle>()
        }) {
            b.kv.compact_offset = 0;
        }
    }

    let trailer = nl.len();
    if m.seq_pos
        .saturating_add(new_tokens.len())
        .saturating_add(max_tokens)
        .saturating_add(trailer)
        > m.physical_cap
    {
        emit_active_attempt_error(
            stdout,
            Some(id),
            &format!("request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > physical_cap={} — reload model with a larger max_seq", m.seq_pos, new_tokens.len(), max_tokens, trailer, m.physical_cap),
            "context_length",
            false,
            false
        );
        let _ = stdout.flush();
        return;
    }

    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };
    let tool_call_pair = match (
        tokenizer.special_token_id("<tool_call>"),
        tokenizer.special_token_id("</tool_call>"),
    ) {
        (Some(o), Some(c)) => Some((o, c)),
        _ => None,
    };
    let think_pair = match (
        tokenizer.special_token_id("<think>"),
        tokenizer.special_token_id("</think>"),
    ) {
        (Some(o), Some(c)) => Some((o, c)),
        _ => None,
    };

    let prefill_tokens = new_tokens.len();
    let t0 = Instant::now();

    let Some(b) = m.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
    }) else {
        unreachable!()
    };
    let config = &b.config;
    let weights = &b.weights;
    let scratch_set = b.pp_scratch_set.as_ref().unwrap();
    let kv = &mut b.kv_cache;
    let dn = &mut b.dn_state;
    let gpus = m.pp_gpus.as_mut().unwrap();
    let dn_la_to_device = m.pp_dn_la_to_device.as_ref().unwrap();

    macro_rules! reset_pp_uncommitted_state {
        () => {{
            m.seq_pos = 0;
            m.conversation_tokens.clear();
            free_checkpoints(&mut m.prefill_checkpoints, gpu);
            free_checkpoints(&mut m.dflash_checkpoints, gpu);
            for (i, s) in dn.s_matrices.iter().enumerate() {
                let g = &mut gpus.devices[dn_la_to_device[i] as usize];
                let _ = g.bind_thread();
                let _ = g.hip.memset(&s.buf, 0, s.buf.size());
            }
            for (i, s) in dn.s_scales.iter().enumerate() {
                let g = &mut gpus.devices[dn_la_to_device[i] as usize];
                let _ = g.bind_thread();
                let _ = g.hip.memset(&s.buf, 0, s.buf.size());
            }
            for (i, s) in dn.conv_states.iter().enumerate() {
                let g = &mut gpus.devices[dn_la_to_device[i] as usize];
                let _ = g.bind_thread();
                let _ = g.hip.memset(&s.buf, 0, s.buf.size());
            }
            // multi-GPU currently leaves s_ef_residual empty; loop is a no-op then,
            // but keeps single-GPU parity if EF is ever wired per-device.
            for (i, s) in dn.s_ef_residual.iter().enumerate() {
                let g = &mut gpus.devices[dn_la_to_device[i] as usize];
                let _ = g.bind_thread();
                let _ = g.hip.memset(&s.buf, 0, s.buf.size());
            }
            kv.compact_offset = 0;
            if let Some(b) = m.state.as_mut().and_then(|s| {
                (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_llama::LlamaBundle>()
            }) {
                b.kv.compact_offset = 0;
            }
        }};
    }

    let dev_last = gpus.output_device;
    let vocab_size = config.vocab_size;
    // Effective penalty window = request `_repeat_window` (default 128),
    // bounded by repeat_buf capacity (2048). Default stays 128; the wide buffer
    // only enables a larger window when a request explicitly sets one.
    let repeat_buf_cap =
        (scratch_set.per_device[dev_last].repeat_buf.buf.size() / 4).min(_repeat_window.max(1));

    // hunt3 M-C: grammar-guided decoding for pp>1 (mirrors generate() ~8168).
    // Without this, a pp>1 + tools request samples unconstrained once the model
    // commits to <tool_call>, reproducing the ChatML-noise-in-tool_call-body
    // attractor the single-GPU path masks via the qwen35 Matcher. The decoded
    // vocab is built into a request-local Vec rather than cached on `m`
    // (m.decoded_vocab) because `m` is already mutably borrowed here (kv/dn/gpus)
    // — pp>1 + tools is uncommon, so the per-request decode is acceptable.
    let grammar_enabled = hipfire_runtime::prompt_frame::qwen35_grammar_on(
        std::env::var("HIPFIRE_QWEN35_GRAMMAR").ok().as_deref(),
        &m.model_path,
    );
    let tool_schemas_qwen: Vec<hipfire_arch_qwen35::grammar::ToolSchema> = if grammar_enabled {
        tools
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| {
                        let func = t.get("function").unwrap_or(t);
                        let name = func
                            .get("name")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())?
                            .to_string();
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
                        Some(hipfire_arch_qwen35::grammar::ToolSchema { name, required })
                    })
                    .collect()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let grammar_active = !tool_schemas_qwen.is_empty();
    let mut grammar_matcher = hipfire_arch_qwen35::grammar::Matcher::with_config(
        tool_schemas_qwen,
        hipfire_arch_qwen35::grammar_config::resolve_qwen35_grammar_config(),
    );
    let grammar_vocab: Vec<String> = if grammar_active {
        let n = tokenizer.vocab_size();
        (0..n).map(|id| tokenizer.decode(&[id as u32])).collect()
    } else {
        Vec::new()
    };
    let mut grammar_mask: Vec<bool> = vec![true; grammar_vocab.len()];

    if let Err(e) = qwen35::forward_prefill_batch_multi(
        gpus,
        weights,
        config,
        &new_tokens,
        m.seq_pos,
        kv,
        dn,
        scratch_set,
    ) {
        // hunt3 M-A: a partial-band prefill failure leaves DeltaNet partially
        // advanced; without resetting, the next cold turn prefills over dirty
        // recurrent state (drift). Mirror both abort paths, which already reset.
        reset_pp_uncommitted_state!();
        emit_active_attempt_error(
            stdout,
            Some(id),
            &format!("forward_prefill_batch_multi: {}", e),
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    m.seq_pos += new_tokens.len();
    m.conversation_tokens.extend_from_slice(&new_tokens);

    if check_abort(id) {
        reset_pp_uncommitted_state!();
        let ep = production_fail_closed_rollback(m, gpu, None, None);
        emit_spec_cancel_after_rollback(stdout, id, 0, &ep);
        return;
    }

    // ngram scope: generated tokens only (matches pp=1).
    let ngram_scope_start = m.conversation_tokens.len();

    let mut rng_state: u32 = request_seed as u32;

    let attractor_pairs: Vec<(u32, u32)> = tool_call_pair
        .into_iter()
        .chain(think_pair.into_iter())
        .collect();

    // First sample on the output device.
    let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
    let mut blocked0: Vec<u32> = Vec::new();
    sampler::collect_unclosed_attractor_blocks(ngram_scope, &attractor_pairs, 20, 2, &mut blocked0);
    let cfg0 = SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty,
        repeat_window: repeat_buf_cap,
        presence_penalty,
        frequency_penalty,
        blocked_tokens: blocked0,
        top_k,
        min_p,
    };
    // hunt3 M-C: grammar-gated first sample (GPU fast path when matcher free;
    // CPU mask-then-sample when constraining). Matches generate()'s tok0 site.
    let tok0 = {
        let s_last = &scratch_set.per_device[dev_last];
        let g_last = &mut gpus.devices[dev_last];
        if grammar_active && !grammar_matcher.is_free() {
            let _ = g_last.bind_thread();
            let mut logits = g_last
                .download_f32(&s_last.logits)
                .unwrap_or_else(|_| vec![0.0f32; vocab_size]);
            grammar_matcher.token_mask(&grammar_vocab, &mut grammar_mask);
            hipfire_arch_qwen35::grammar::Matcher::apply_mask_to_logits(&grammar_mask, &mut logits);
            sampler::sample_cpu(&mut logits, ngram_scope, &cfg0)
        } else {
            sampler::sample(
                g_last,
                &s_last.logits,
                &s_last.sample_buf,
                &s_last.repeat_buf,
                vocab_size,
                ngram_scope,
                &cfg0,
                &mut rng_state,
            )
        }
    };
    if grammar_active {
        grammar_matcher.advance(&tokenizer.decode(&[tok0]));
    }
    let t_prefill = Instant::now();
    let mut next_token = tok0;

    let mut generated = 0usize;
    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut bytes_fed_to_filter = 0usize;
    let mut filter = EosFilter::new(EosFilterConfig::default());
    let mut alert_fired = false;
    let mut think_count: usize = 0;
    let mut prev_in_think: bool = false;
    let mut force_answer_latched = false;
    let think_open_tok = tokenizer.special_token_id("<think>");
    let max_total_think = hipfire_runtime::config::get().max_total_think_tokens;
    let mut total_think_tokens: usize = 0;
    // Post-latch answer bound. Once the think-cap latches we force-close <think>
    // and ask the model to answer; but `total_think_tokens` only advances
    // in-think, so a model that rambles a NON-think answer (or re-opens <think>
    // in a tight loop the force-close keeps re-closing) never trips the +256 EOS
    // and runs to max_tokens. Mark the latch position and hard-EOS once
    // generation runs this many tokens past it — generous for a real final
    // answer, bounded against runaway.
    let post_latch_answer_budget: usize = std::env::var("HIPFIRE_POST_LATCH_ANSWER_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(768);
    let mut latch_gen_mark: Option<usize> = None;
    let loop_guard =
        hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get());

    while generated < max_tokens {
        if check_abort(id) {
            reset_pp_uncommitted_state!();
            let ep = production_fail_closed_rollback(m, gpu, None, None);
            emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
            return;
        }
        generated += 1;
        m.conversation_tokens.push(next_token);
        streamed_tokens.push(next_token);
        emit_committed_event(
            stdout,
            id,
            next_token,
            streamed_tokens.len() - 1,
            t0.elapsed().as_millis() as u64,
        );
        let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
        let new_bytes = &all_bytes[bytes_fed_to_filter..];
        bytes_fed_to_filter = all_bytes.len();
        if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
            let text = std::str::from_utf8(&text_bytes).unwrap();
            let _ = writeln!(
                stdout,
                r#"{{"type":"token","id":"{}","text":{},"attempt_id":{}}}"#,
                id,
                serde_json::to_string(&text).unwrap_or_default(),
                active_attempt_id()
            );
            let _ = stdout.flush();
        }

        if let Err(e) = qwen35::forward_scratch_multi(
            gpus,
            weights,
            config,
            next_token,
            m.seq_pos,
            kv,
            dn,
            scratch_set,
        ) {
            // hunt3 M-A: a decode-step failure leaves DeltaNet advanced past the
            // (un-baked) conversation_tokens; reset so the next cold turn starts
            // clean. Mirrors both abort paths.
            reset_pp_uncommitted_state!();
            emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("forward_scratch_multi decode: {}", e),
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
        m.seq_pos += 1;

        if next_token == config.eos_token {
            break;
        }
        if im_end_token == Some(next_token) {
            break;
        }
        if tokenizer.is_terminator(next_token) {
            break;
        }

        // hunt3 M-F: user stop-sequence match against the decoded output suffix
        // (pp>1 multi-GPU path). Mirrors the AR generate() loop; matches the
        // full decoded text so a stop string spanning a token boundary is
        // caught. A plain break exits the `while generated < max_tokens` loop
        // (this path's `done` event carries no finish_reason field, so there is
        // no reason to resolve — terminating generation is the contract). Gated
        // behind `!stop.is_empty()` so the common path pays nothing.
        if !stop.is_empty() {
            let decoded_suffix = tokenizer.decode(&streamed_tokens);
            if stop.iter().any(|s| decoded_suffix.ends_with(s.as_str())) {
                break;
            }
        }

        // max_think_tokens / force-answer enforcement: same decoded-text scan
        // as pp=1, but all recurrent-state writes route through *_multi.
        let force_answer_now = check_force_answer(id);
        if force_answer_now {
            force_answer_latched = true;
        }
        if max_think_tokens > 0 || force_answer_now || force_answer_latched || max_total_think > 0 {
            let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
            let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
            let in_think = currently_in_think(raw_str, started_in_think);
            if in_think {
                total_think_tokens += 1;
            }
            if max_total_think > 0 && total_think_tokens >= max_total_think {
                force_answer_latched = true;
            }
            if force_answer_latched && latch_gen_mark.is_none() {
                latch_gen_mark = Some(generated);
            }
            if max_total_think > 0 && in_think && total_think_tokens >= max_total_think + 256 {
                eprintln!("[think-cap] id={} — total think {} exceeded cap {}+256 while still thinking; forcing EOS", id, total_think_tokens, max_total_think);
                break;
            }
            if let Some(mark) = latch_gen_mark {
                if generated.saturating_sub(mark) >= post_latch_answer_budget {
                    eprintln!("[think-cap] id={} — {} tokens since think-cap latch without finishing; forcing EOS", id, generated.saturating_sub(mark));
                    break;
                }
            }
            if max_think_tokens > 0 {
                if in_think {
                    if !prev_in_think {
                        think_count = 1;
                    } else {
                        think_count += 1;
                    }
                } else {
                    think_count = 0;
                }
                prev_in_think = in_think;
            }
            let budget_hit = max_think_tokens > 0 && think_count >= max_think_tokens;
            let request_cap_latched_now = latch_request_think_cap(
                budget_hit,
                generated,
                &mut force_answer_latched,
                &mut latch_gen_mark,
            );

            if in_think && (budget_hit || force_answer_now || force_answer_latched) {
                if request_cap_latched_now {
                    eprintln!(
                        "[think-cap] id={} — per-request think cap {} reached; closing <think>",
                        id, max_think_tokens
                    );
                } else if force_answer_now {
                    eprintln!(
                        "[force-answer] id={} — closing <think> mid-turn to commit to the answer",
                        id
                    );
                }
                let close_tokens = tokenizer.encode(&think_continuation());
                let budget_left = max_tokens.saturating_sub(generated);
                let take = close_tokens.len().min(budget_left);
                for &t in &close_tokens[..take] {
                    if let Err(e) = qwen35::forward_scratch_multi(
                        gpus,
                        weights,
                        config,
                        t,
                        m.seq_pos,
                        kv,
                        dn,
                        scratch_set,
                    ) {
                        eprintln!("[daemon] max_think close forward_scratch_multi: {}", e);
                        break;
                    }
                    m.seq_pos += 1;
                    m.conversation_tokens.push(t);
                    // hunt3 M-C: keep the grammar matcher in sync over force-closed
                    // </think> tokens, exactly as generate() does (~8591). Without
                    // this a tools request that force-closes <think> leaves the
                    // matcher stale → malformed tool calls after the forced close.
                    if grammar_active {
                        grammar_matcher.advance(&tokenizer.decode(&[t]));
                    }
                    streamed_tokens.push(t);
                    emit_committed_event(
                        stdout,
                        id,
                        t,
                        streamed_tokens.len() - 1,
                        t0.elapsed().as_millis() as u64,
                    );
                    let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
                    let new_bytes = &all_bytes[bytes_fed_to_filter..];
                    bytes_fed_to_filter = all_bytes.len();
                    if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
                        let text = std::str::from_utf8(&text_bytes).unwrap();
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"token","id":"{}","text":{},"attempt_id":{}}}"#,
                            id,
                            serde_json::to_string(&text).unwrap_or_default(),
                            active_attempt_id()
                        );
                        let _ = stdout.flush();
                    }
                    generated += 1;
                }
                think_count = 0;
                prev_in_think = false;
                if generated >= max_tokens {
                    break;
                }
            }
        }

        // N-gram loop detector (token-side, no GPU work).
        if let Some(hipfire_runtime::loop_guard::StopReason::NgramRepeat { count, .. }) =
            loop_guard.check(&streamed_tokens)
        {
            let window_len = loop_guard.window_len(streamed_tokens.len());
            let _ = writeln!(
                stdout,
                r#"{{"type":"info","id":"{}","message":"ngram loop detected (4gram repeated {}× in last {} tokens) — forcing EOS"}}"#,
                id, count, window_len
            );
            let _ = stdout.flush();
            break;
        }

        // Budget-alert injection: gated to inside an open <think> block.
        if !alert_fired
            && budget_alert_at_tok > 0
            && generated >= budget_alert_at_tok
            && !budget_alert_text.is_empty()
        {
            alert_fired = true;
            let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
            let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
            let in_think = currently_in_think(raw_str, started_in_think);
            if !in_think {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not inside an open <think> block"}}"#,
                    id
                );
                let _ = stdout.flush();
                let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
                let mut blocked: Vec<u32> = Vec::new();
                sampler::collect_unclosed_attractor_blocks(
                    ngram_scope,
                    &attractor_pairs,
                    20,
                    2,
                    &mut blocked,
                );
                if force_answer_latched {
                    if let Some(t) = think_open_tok {
                        blocked.push(t);
                    }
                }
                let cfg = SamplerConfig {
                    temperature: temp,
                    top_p,
                    repeat_penalty,
                    repeat_window: repeat_buf_cap,
                    presence_penalty,
                    frequency_penalty,
                    blocked_tokens: blocked,
                    top_k,
                    min_p,
                };
                // hunt3 M-C: grammar-gated budget-alert resample.
                next_token = {
                    let s_last = &scratch_set.per_device[dev_last];
                    let g_last = &mut gpus.devices[dev_last];
                    if grammar_active && !grammar_matcher.is_free() {
                        let _ = g_last.bind_thread();
                        let mut logits = g_last
                            .download_f32(&s_last.logits)
                            .unwrap_or_else(|_| vec![0.0f32; vocab_size]);
                        grammar_matcher.token_mask(&grammar_vocab, &mut grammar_mask);
                        hipfire_arch_qwen35::grammar::Matcher::apply_mask_to_logits(
                            &grammar_mask,
                            &mut logits,
                        );
                        sampler::sample_cpu(&mut logits, ngram_scope, &cfg)
                    } else {
                        sampler::sample(
                            g_last,
                            &s_last.logits,
                            &s_last.sample_buf,
                            &s_last.repeat_buf,
                            vocab_size,
                            ngram_scope,
                            &cfg,
                            &mut rng_state,
                        )
                    }
                };
                if grammar_active {
                    grammar_matcher.advance(&tokenizer.decode(&[next_token]));
                }
                continue;
            }
            let nudge_tokens = tokenizer.encode(budget_alert_text);
            let budget_left = max_tokens.saturating_sub(generated);
            let nudge_len = nudge_tokens.len().min(budget_left);
            let need_kv = m
                .seq_pos
                .saturating_add(nudge_len)
                .saturating_add(
                    max_tokens
                        .saturating_sub(generated)
                        .saturating_sub(nudge_len),
                )
                .saturating_add(nl.len());
            if nudge_len > 0 && need_kv <= m.physical_cap {
                for &tok in &nudge_tokens[..nudge_len] {
                    m.conversation_tokens.push(tok);
                    streamed_tokens.push(tok);
                    emit_committed_event(
                        stdout,
                        id,
                        tok,
                        streamed_tokens.len() - 1,
                        t0.elapsed().as_millis() as u64,
                    );
                    let all_bytes2 = tokenizer.decode_bytes(&streamed_tokens);
                    let new_bytes2 = &all_bytes2[bytes_fed_to_filter..];
                    bytes_fed_to_filter = all_bytes2.len();
                    if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes2) {
                        let t = std::str::from_utf8(&text_bytes).unwrap();
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"token","id":"{}","text":{},"attempt_id":{}}}"#,
                            id,
                            serde_json::to_string(&t).unwrap_or_default(),
                            active_attempt_id()
                        );
                        let _ = stdout.flush();
                    }
                    if let Err(e) = qwen35::forward_scratch_multi(
                        gpus,
                        weights,
                        config,
                        tok,
                        m.seq_pos,
                        kv,
                        dn,
                        scratch_set,
                    ) {
                        eprintln!("[daemon] budget_alert forward_scratch_multi: {}", e);
                        break;
                    }
                    m.seq_pos += 1;
                    generated += 1;
                }
            } else if nudge_len < nudge_tokens.len() {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"info","id":"{}","message":"budget_alert clipped or skipped: nudge_len={} budget_left={}"}}"#,
                    id, nudge_len, budget_left
                );
                let _ = stdout.flush();
            } else {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not enough KV headroom"}}"#,
                    id
                );
                let _ = stdout.flush();
            }
            if generated >= max_tokens {
                break;
            }
        }

        // Steady-state sample.
        let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
        let mut blocked: Vec<u32> = Vec::new();
        sampler::collect_unclosed_attractor_blocks(
            ngram_scope,
            &attractor_pairs,
            20,
            2,
            &mut blocked,
        );
        if force_answer_latched {
            if let Some(t) = think_open_tok {
                blocked.push(t);
            }
        }
        let cfg = SamplerConfig {
            temperature: temp,
            top_p,
            repeat_penalty,
            repeat_window: repeat_buf_cap,
            presence_penalty,
            frequency_penalty,
            blocked_tokens: blocked,
            top_k,
            min_p,
        };
        // hunt3 M-C: grammar-gated steady-state sample.
        next_token = {
            let s_last = &scratch_set.per_device[dev_last];
            let g_last = &mut gpus.devices[dev_last];
            if grammar_active && !grammar_matcher.is_free() {
                let _ = g_last.bind_thread();
                let mut logits = g_last
                    .download_f32(&s_last.logits)
                    .unwrap_or_else(|_| vec![0.0f32; vocab_size]);
                grammar_matcher.token_mask(&grammar_vocab, &mut grammar_mask);
                hipfire_arch_qwen35::grammar::Matcher::apply_mask_to_logits(
                    &grammar_mask,
                    &mut logits,
                );
                sampler::sample_cpu(&mut logits, ngram_scope, &cfg)
            } else {
                sampler::sample(
                    g_last,
                    &s_last.logits,
                    &s_last.sample_buf,
                    &s_last.repeat_buf,
                    vocab_size,
                    ngram_scope,
                    &cfg,
                    &mut rng_state,
                )
            }
        };
        if grammar_active {
            let was_detected = grammar_matcher.attractor_detected();
            grammar_matcher.advance(&tokenizer.decode(&[next_token]));
            if !was_detected && grammar_matcher.attractor_detected() {
                eprintln!(
                    "[grammar-ngram pp] attractor detected in tool_call args at gen={} — forcing close",
                    generated,
                );
            }
        }
    }

    // ChatML \n trailer so the next turn opens cleanly.
    if im_end_token == Some(*m.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
        for &t in &nl {
            if let Err(e) = qwen35::forward_scratch_multi(
                gpus,
                weights,
                config,
                t,
                m.seq_pos,
                kv,
                dn,
                scratch_set,
            ) {
                eprintln!("[daemon] trailer forward_scratch_multi: {}", e);
                break;
            }
            m.seq_pos += 1;
            m.conversation_tokens.push(t);
        }
    }

    // Timing + pending done fixed before handshake so commit_ready carries
    // the exact eventual done payload. Abort rolls back + emits one
    // cancellation lifecycle (or fail-closed error if unattested).
    let t_end = Instant::now();
    let total_s = t_end.duration_since(t0).as_secs_f64();
    let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
    let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
    let tok_s = if total_s > 0.0 {
        generated as f64 / total_s
    } else {
        0.0
    };
    let prefill_tok_s = if prefill_s > 0.0 {
        prefill_tokens as f64 / prefill_s
    } else {
        0.0
    };
    let decode_tok_s = if decode_s > 0.0 {
        generated as f64 / decode_s
    } else {
        0.0
    };
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated,
        "tok_s": (tok_s * 10.0).round() / 10.0,
        "prefill_tokens": prefill_tokens,
        "prefill_ms": (prefill_s * 1000.0 * 10.0).round() / 10.0,
        "prefill_tok_s": (prefill_tok_s * 10.0).round() / 10.0,
        "decode_tok_s": (decode_tok_s * 10.0).round() / 10.0,
        "ttft_ms": (prefill_s * 1000.0 * 10.0).round() / 10.0,
        "attempt_id": active_attempt_id(),
    });
    let decision = await_client_terminal_commit(stdout, id, &pending_done);
    if decision != ClientTerminalDecision::Commit {
        reset_pp_uncommitted_state!();
        let ep = production_fail_closed_rollback(m, gpu, None, None);
        emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
        return;
    }
    emit_staged_terminal_done(stdout, &pending_done);
}

// --- Auto-appended shared helpers (shared-temp, dedup at merge) ---

/// Walk a [`serde_json::Value`] and produce a canonical-key
/// representation: objects emit keys in lexical order (recursively),
/// arrays preserve order. Used by [`asst_turn_fingerprint`] so two
/// messages with the same logical tool args hash identically
/// regardless of source-side insertion order.

pub fn qwen_client_commit_effects(
    decision: ClientTerminalDecision,
    intended_release: bool,
    intended_store: bool,
) -> QwenClientCommitEffects {
    match decision {
        ClientTerminalDecision::Commit => QwenClientCommitEffects {
            release_tool_calls: intended_release,
            store_cache: intended_store,
            emit_done: true,
        },
        ClientTerminalDecision::Abort => QwenClientCommitEffects {
            release_tool_calls: false,
            store_cache: false,
            emit_done: false,
        },
    }
}

/// Open the DS4 EP wire contract before prefill can eventually emit tokens.
///
/// EP owns its generation loop instead of routing through the single-device
/// AR/spec emitters, so it must establish the same contract latch explicitly.
pub fn emit_ds4_ep_gen_start(stdout: &mut impl std::io::Write, id: &str, think_mode: ThinkMode) {
    emit_gen_start(
        stdout,
        id,
        !matches!(think_mode, ThinkMode::NonThink),
        ds4_gen_start_contract_version(),
    );
}

/// Speculative wire terminal after `Qwen35Emit::finish` + length/EOT known.
/// Length without decoded EOT never releases calls or stores cache; malformed
/// is error XOR done; decoded EOT on the final budget token beats length.
#[derive(Debug, Clone)]
pub enum QwenDflashWireTerminal {
    Malformed {
        message: String,
        class: &'static str,
        retryable: bool,
        rolled_back: bool,
    },
    Done {
        finish_reason: &'static str,
        release_tool_calls: bool,
        store_cache: bool,
        /// Producer-authorized fingerprint text (visible only).
        fingerprint_text: String,
        wire_tool_calls: Vec<hipfire_runtime::prompt_frame::ToolCall>,
    },
}

/// Emitter stop reasons that are semantic terminals (not grammar fail-closed).
/// Survives into SpecRun so wrappers classify stop/tool_calls over length when
/// `generated == max_tokens`.
pub fn spec_stop_is_semantic(stop: Option<StopReason>) -> bool {
    matches!(
        stop,
        Some(StopReason::Eos) | Some(StopReason::StopSequence) | Some(StopReason::ThinkCap)
    )
}

/// Production terminal + cache decision for Qwen DFlash/spec epilogue.
/// Shared by `generate_dflash` and deterministic non-GPU tests.
pub fn qwen_dflash_wire_terminal(
    finish: &FinishSummary,
    hit_length_cap: bool,
    grammar_violated: bool,
    visible_for_cache: &str,
    // Truthful rollback attestation from the production epilogue.
    // Unit tests that do not drive GPU reset pass `false`.
    rolled_back: bool,
) -> QwenDflashWireTerminal {
    if finish.finish_reason == "malformed_protocol" {
        return QwenDflashWireTerminal::Malformed {
            message: "malformed tool protocol".to_string(),
            class: "validation",
            retryable: false,
            rolled_back,
        };
    }
    // Open think is a nonretryable unsafe terminal (error XOR done).
    if finish.open_think || finish.finish_reason == "open_think" {
        return QwenDflashWireTerminal::Malformed {
            message: "open think span at end of generation (validation)".to_string(),
            class: "validation",
            retryable: false,
            rolled_back,
        };
    }
    // Grammar failure is error-only: no done/calls/cache.
    if grammar_violated {
        return QwenDflashWireTerminal::Malformed {
            message: "grammar violation during speculative decode".to_string(),
            class: "validation",
            retryable: false,
            rolled_back,
        };
    }
    if hit_length_cap {
        return QwenDflashWireTerminal::Done {
            finish_reason: "length",
            release_tool_calls: false,
            store_cache: false,
            fingerprint_text: String::new(),
            wire_tool_calls: Vec::new(),
        };
    }
    let held = finish_summary_held_tool_calls(finish);
    let fingerprint_text = normalize_asst_turn_for_fingerprint(visible_for_cache);
    if !held.is_empty() || finish.tool_calls > 0 {
        QwenDflashWireTerminal::Done {
            finish_reason: "tool_calls",
            release_tool_calls: true,
            store_cache: true,
            fingerprint_text,
            wire_tool_calls: held,
        }
    } else {
        QwenDflashWireTerminal::Done {
            finish_reason: "stop",
            release_tool_calls: false,
            store_cache: true,
            fingerprint_text,
            wire_tool_calls: Vec::new(),
        }
    }
}

pub fn qwen_dflash_cache_action(terminal: &QwenDflashWireTerminal) -> QwenDflashCacheAction {
    match terminal {
        QwenDflashWireTerminal::Malformed { .. } => QwenDflashCacheAction {
            store: false,
            fingerprint_text: String::new(),
            tool_calls: Vec::new(),
        },
        QwenDflashWireTerminal::Done {
            store_cache,
            fingerprint_text,
            wire_tool_calls,
            ..
        } => QwenDflashCacheAction {
            store: *store_cache,
            fingerprint_text: fingerprint_text.clone(),
            tool_calls: wire_tool_calls.clone(),
        },
    }
}

pub fn qwen_dflash_apply_cache_action<F>(
    mut insert: F,
    action: &QwenDflashCacheAction,
    cached_seq: Vec<u32>,
) -> Option<u64>
where
    F: FnMut(u64, Vec<u32>),
{
    if !action.store || cached_seq.is_empty() {
        return None;
    }
    let fp = asst_turn_fingerprint(&action.fingerprint_text, &action.tool_calls);
    insert(fp, cached_seq);
    Some(fp)
}

/// Decode whether the last streamed token is a terminator for EOT-vs-length.
pub fn qwen_dflash_decoded_eot_from_tokens(
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    streamed: &[u32],
    eos: u32,
    im_end: Option<u32>,
) -> bool {
    let Some(&last) = streamed.last() else {
        return false;
    };
    last == eos || im_end == Some(last) || tokenizer.is_terminator(last)
}

/// Visible fingerprint text from held finish events (Token channel only).
pub fn qwen_dflash_visible_from_finish(finish: &FinishSummary) -> String {
    let mut s = String::new();
    for ev in &finish.events {
        if let ClientEvent::Token(t) = ev {
            s.push_str(t);
        }
    }
    s
}

/// Production serde value for a correlated DFlash/spec token event.
pub fn qwen_dflash_token_event_value(id: &str, text: &str, attempt_id: u64) -> serde_json::Value {
    serde_json::json!({
        "type": "token",
        "id": id,
        "text": text,
        "attempt_id": attempt_id,
    })
}

/// Production serde value for a correlated DFlash/spec reasoning event.
pub fn qwen_dflash_reasoning_event_value(
    id: &str,
    text: &str,
    attempt_id: u64,
) -> serde_json::Value {
    serde_json::json!({
        "type": "reasoning",
        "id": id,
        "text": text,
        "attempt_id": attempt_id,
    })
}

/// Production done envelope core for Qwen DFlash epilogue + tests.
/// Optional pflash fields are merged by the caller after construction.
pub fn qwen_dflash_done_value(
    id: &str,
    generated: usize,
    tok_s: f64,
    prefill_tokens: usize,
    prefill_ms: f64,
    prefill_tok_s: f64,
    decode_tok_s: f64,
    ttft_ms: f64,
    tau: f64,
    cycles: usize,
    cached_tokens: usize,
    finish_reason: &str,
    attempt_id: u64,
) -> serde_json::Value {
    serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated,
        "tok_s": tok_s,
        "prefill_tokens": prefill_tokens,
        "prefill_ms": prefill_ms,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "ttft_ms": ttft_ms,
        "dflash": true,
        "tau": tau,
        "cycles": cycles,
        "cached_tokens": cached_tokens,
        "finish_reason": finish_reason,
        "attempt_id": attempt_id,
    })
}

/// Write one Qwen DFlash Malformed terminal via the production fail-closed
/// error writer (same envelope as step/forced/grammar failures).
pub fn emit_qwen_dflash_malformed_terminal(
    stdout: &mut impl std::io::Write,
    id: &str,
    message: &str,
    class: &str,
    retryable: bool,
    epilogue: &RollbackEpilogue,
) {
    emit_fail_closed_error(stdout, Some(id), message, class, retryable, epilogue);
}

/// Spec-step / forced-advance failure terminal (production + tests).
/// Call only after [`production_fail_closed_rollback`] / `_live` (or with a
/// GPU-less attested epilogue in unit tests).
pub fn emit_spec_failure_terminal(
    stdout: &mut impl std::io::Write,
    id: &str,
    what: &str,
    err: &str,
    epilogue: &RollbackEpilogue,
) {
    let msg = spec_failure_message(what, err);
    emit_fail_closed_error(stdout, Some(id), &msg, "validation", false, epilogue);
}

/// Pure trailer trim for asst-turn cache sequence (production + tests).
///
/// Strips a trailing im_end + newline trailer only when im_end is present as
/// the true trailer. A length-capped body that ends on a newline is restored
/// verbatim so cached body length stays parity-equal with streamed tokens.
pub fn qwen_dflash_cache_seq(
    streamed: &[u32],
    im_end: Option<u32>,
    nl: &std::collections::HashSet<u32>,
) -> Vec<u32> {
    let mut cached_seq: Vec<u32> = streamed.to_vec();
    let body_len_before_trim = cached_seq.len();
    while let Some(&last) = cached_seq.last() {
        if nl.contains(&last) {
            cached_seq.pop();
        } else {
            break;
        }
    }
    match cached_seq.last() {
        Some(&last) if im_end == Some(last) => {
            cached_seq.pop();
        }
        // No im_end behind the newlines ⇒ they were never a trailer.
        _ => cached_seq = streamed[..body_len_before_trim].to_vec(),
    }
    cached_seq
}

pub fn spec_host_advance_after_step(
    mut position: usize,
    mut generated: usize,
    mut emitted: Vec<u32>,
    step_emit: &[u32],
    step_next_seed: u32,
    consumed: usize,
) -> SpecHostAdvance {
    let keep = consumed.min(step_emit.len());
    let prefix = &step_emit[..keep];
    emitted.extend_from_slice(prefix);
    generated = generated.saturating_add(prefix.len());
    position = position.saturating_add(prefix.len());
    let seed_token = prefix.last().copied().unwrap_or(step_next_seed);
    SpecHostAdvance {
        position,
        generated,
        emitted,
        seed_token,
    }
}

/// Terminal flush: forward the final pending seed exactly once.
///
/// After this commit, `position += 1` and model state ends on the same
/// logical token as conversation history (no lagging unwritten seed).
pub fn spec_terminal_pending_seed_tx(pending_seed: u32) -> SpecPendingSeedTx {
    SpecPendingSeedTx {
        commit: vec![pending_seed],
        pending_seed,
        position_delta: 1,
    }
}

/// Whether a [`SpecEmit`] outcome's pending seed is state-committable.
///
/// Event-bearing outcomes (including hidden/raw protocol `Committed` bytes)
/// stay committable so history/GPU flush keep them. DS4 empty-event EOS is a
/// model terminator only — not baked, not terminal-flushed.
pub fn spec_outcome_seed_committable(outcome: &EmitOutcome) -> bool {
    !outcome.events.is_empty()
}

/// Terminal pending-seed GPU flush gate.
///
/// Skip on grammar fail-closed (rollback wipes state) and when the current
/// pending seed is intentionally non-committable (DS4 empty-event EOS).
pub fn spec_should_flush_pending_seed(
    grammar_violated: bool,
    pending_seed_committable: bool,
) -> bool {
    !grammar_violated && pending_seed_committable
}

pub fn spec_prefix_realign_plan(
    prompt: &[u32],
    first_token: u32,
    raw_decode: &[u32],
) -> SpecPrefixRealignPlan {
    if raw_decode.is_empty() {
        return SpecPrefixRealignPlan {
            replay: prompt.to_vec(),
            position: prompt.len(),
            seed_token: first_token,
        };
    }
    let mut replay = Vec::with_capacity(prompt.len() + raw_decode.len());
    replay.extend_from_slice(prompt);
    replay.push(first_token);
    replay.extend_from_slice(&raw_decode[..raw_decode.len() - 1]);
    let position = prompt.len() + raw_decode.len();
    debug_assert_eq!(
        replay.len(),
        position,
        "processed prefix length must equal next write slot"
    );
    SpecPrefixRealignPlan {
        position,
        seed_token: *raw_decode.last().unwrap(),
        replay,
    }
}

/// Prove a strict-prefix realign replay fits target physical + speculator caps
/// and is compatible with current eviction/compact-offset state.
///
/// Realign is reset + full-history prefill (`compact_offset` cleared). It must
/// never overrun `physical_cap` / `ctx_capacity`, and must not silently rebuild
/// a compacted/evicted cache from an oversize full-history replay. On `Err`,
/// callers fail closed with one rollback+error terminal (no reset/prefill).
///
/// `compact_offset` / `eviction_enabled` are part of the admit contract: when
/// either is live the physical buffer is a retained window, but realign still
/// full-replays logical history after reset. Retain-mask partial repair is
/// **not** used here (no source contract proves DeltaNet/drafter-hidden
/// equivalence). Capacity fit is the only proof; oversized history fails closed.
pub fn spec_prefix_realign_admit(
    plan: &SpecPrefixRealignPlan,
    physical_cap: usize,
    ctx_capacity: usize,
    compact_offset: usize,
    eviction_enabled: bool,
) -> Result<(), String> {
    // Processed-prefix / pending-seed split: replay holds only processed tokens;
    // `position` is the next write slot for the unprocessed pending seed.
    if plan.replay.len() != plan.position {
        return Err(format!(
            "spec_prefix_realign: invariant broken: replay.len()={} != position={} (pending seed must stay unprocessed)",
            plan.replay.len(),
            plan.position
        ));
    }
    // Invariant holds ⇒ `position == processed`. Reset+prefill writes
    // `processed` rows at physical [0, processed). The pending seed is
    // unwritten but needs a legal next write slot at `position`.
    let processed = plan.replay.len();
    if plan.position >= physical_cap {
        return Err(format!(
            "spec_prefix_realign: replay len {processed} (pending seed slot {}) exceeds physical_cap {physical_cap}",
            plan.position
        ));
    }
    if plan.position >= ctx_capacity {
        return Err(format!(
            "spec_prefix_realign: replay len {processed} (pending seed slot {}) exceeds ctx_capacity {ctx_capacity}",
            plan.position
        ));
    }
    // Eviction/compact_offset: realign always reset+full-replays (offset cleared).
    // Retain-mask partial repair is intentionally unused (no source contract for
    // DeltaNet/drafter-hidden equivalence). Capacity fit above is the only admit
    // proof — oversize full-history under compaction fails closed, never silent
    // reconstruct. Params stay in the signature so callers cannot ignore the seam.
    let _ = (compact_offset, eviction_enabled);
    Ok(())
}

/// Outcome of a forced GPU advance after abort is observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForcedGpuAdvanceKind {
    /// Advance completed; token may be baked/emitted.
    Committed,
    /// Abort fired around the advance — rollback + cancelled terminal; do not bake.
    Cancelled,
}

/// Classify forced GPU advance after pre/mid/post abort observability.
pub fn classify_forced_gpu_advance(abort_observed: bool) -> ForcedGpuAdvanceKind {
    if abort_observed {
        ForcedGpuAdvanceKind::Cancelled
    } else {
        ForcedGpuAdvanceKind::Committed
    }
}

/// Outcome of [`apply_spec_forced_pending_seed`] (begin + mid-window share this).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecForcedApplyResult {
    /// Empty input or hard-budget clip left nothing to apply.
    Skipped,
    /// GPU/host advanced; last forced token is the new unprocessed pending seed.
    Applied,
    /// Forced suffix halted on a non-`None` stop after successful commit/render.
    /// Caller must skip remaining forcing and all subsequent `spec.step` work.
    Stopped(StopReason),
    /// Exclusive cancel or error terminal already written — caller returns `None`.
    /// Staged emitter outcomes were discarded (no client events).
    Terminal,
}

/// Apply a forced continuation with the single pending-seed transaction.
///
/// Shared by `generate_spec` immediately after `emit.begin(first_token)` and by
/// the mid-window observe path after prefix realign. Same hard budget, commit
/// ordering (trigger retained only when committable; last forced pending once),
/// physical-cap admission, optional post-commit eviction, abort latch, rollback
/// attestation, client observe/render, raw history, and pending-seed
/// committability rules — begin/mid-window cannot drift.
///
/// Ordering (commit-before-client-visible):
/// 1. Stage `emit.observe` over the budget-clipped forced list and stop at the
///    first non-`None` stop — later forced tokens are never observed.
/// 2. Build the exact pending+forced processed prefix; reject when the commit
///    slice cannot fit `physical_cap` (no-eviction keeps a pending-seed slot;
///    eviction may end at cap only with step 4).
/// 3. GPU-commit that exact prefix (`on_forced_advance` / `spec_advance`).
/// 4. When eviction is active: `maybe_evict` + `on_evict` on the post-commit
///    physical position before any host seed/raw/render/generated update.
/// 5. Only after successful commit (+ eviction): set host position, push raw
///    history, render events, update `generated` / `pending_seed_committable`.
///
/// `forced_all` is the raw `take_forced()` payload (may be empty). On
/// [`SpecForcedApplyResult::Terminal`] the exclusive cancel/error envelope is
/// already on the wire; caller must `drop(guard); return None`. On
/// [`SpecForcedApplyResult::Stopped`] the terminal stop is returned so callers
/// skip remaining force and every subsequent speculative step.
///
/// Each forced `observe` updates `pending_seed_committable` via
/// [`spec_outcome_seed_committable`] so the final pending seed matches the last
/// kept forced token's event-bearing status (DS4 empty-event EOS stays unflushed).
#[allow(clippy::too_many_arguments)]
pub fn apply_spec_forced_pending_seed(
    forced_all: &[u32],
    generated: &mut usize,
    max_tokens: usize,
    seed_token: &mut u32,
    position: &mut usize,
    raw_decode: &mut Vec<u32>,
    emitted: &mut Vec<u32>,
    pending_seed_committable: &mut bool,
    emit: &mut dyn SpecEmit,
    gpu: &mut rdna_compute::Gpu,
    slot: &mut dyn SpecTarget,
    spec: &mut dyn Speculator,
    stdout: &mut impl std::io::Write,
    id: &str,
    t0: Instant,
    seq_pos: &mut usize,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    dflash_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    asst_turn_cache: &mut hipfire_loader::AsstTurnCache,
    physical_cap: usize,
    eviction: Option<&Eviction>,
) -> SpecForcedApplyResult {
    if forced_all.is_empty() {
        return SpecForcedApplyResult::Skipped;
    }
    let forced = spec_forced_tokens_within_budget(*generated, max_tokens, forced_all);
    if forced.is_empty() {
        return SpecForcedApplyResult::Skipped;
    }
    // Incoming seed committability gates whether the pre-forced pending seed is
    // prepended to the GPU commit (ForcedCommittableTx). Capture before staging
    // observes overwrite `pending_seed_committable` for the new seed.
    let incoming_seed_committable = *pending_seed_committable;

    // Stage observe outcomes before any GPU commit. Halt the forced suffix at
    // the first non-None stop so later forced tokens stay untouched.
    let mut staged: Vec<(u32, EmitOutcome)> = Vec::with_capacity(forced.len());
    let mut stop_reason: Option<StopReason> = None;
    let mut hint_generated = *generated;
    for &ft in forced {
        emit.set_generated_hint(hint_generated);
        let fo = emit.observe(ft);
        if !fo.events.is_empty() {
            hint_generated = hint_generated.saturating_add(1);
        }
        let stop = fo.stop;
        staged.push((ft, fo));
        if let Some(reason) = stop {
            stop_reason = Some(reason);
            break;
        }
    }
    debug_assert!(!staged.is_empty(), "budget-clipped forced is non-empty");
    let kept: Vec<u32> = staged.iter().map(|(t, _)| *t).collect();
    let tx = spec_forced_pending_seed_tx(*seed_token, &kept, incoming_seed_committable);

    // Physical-cap admission on the exact commit slice (actual tx.commit.len()).
    // Reject before any GPU write. Capacity failure is fail-closed ErrorOnly.
    if !spec_forced_commit_admits(*position, tx.commit.len(), physical_cap, eviction.is_some()) {
        let ep = production_fail_closed_rollback_live(
            seq_pos,
            conversation_tokens,
            prefill_checkpoints,
            dflash_checkpoints,
            asst_turn_cache,
            gpu,
            slot,
            spec,
        );
        emit_fail_closed_error(
            stdout,
            Some(id),
            &format!(
                "forced commit exceeds physical_cap: position={} commit_len={} physical_cap={} eviction={}",
                *position,
                tx.commit.len(),
                physical_cap,
                eviction.is_some()
            ),
            "validation",
            false,
            &ep,
        );
        return SpecForcedApplyResult::Terminal;
    }

    // Abort observability around forced GPU advance: check before and after,
    // and latch any mid-advance callback fire. Cancel/error never bakes/emits
    // the staged forced tokens (fail-closed discards the whole turn).
    if check_abort(id) {
        let _ = classify_forced_gpu_advance(true);
        let ep = production_fail_closed_rollback_live(
            seq_pos,
            conversation_tokens,
            prefill_checkpoints,
            dflash_checkpoints,
            asst_turn_cache,
            gpu,
            slot,
            spec,
        );
        emit_spec_cancel_after_rollback(stdout, id, *generated, &ep);
        return SpecForcedApplyResult::Terminal;
    }
    let abort_latched = std::cell::Cell::new(false);
    let abort_cb = || {
        if check_abort(id) {
            abort_latched.set(true);
            true
        } else {
            false
        }
    };
    // Speculator first refusal: drafters with per-position cached target-hidden
    // (DFlash) must advance themselves with hidden extraction. `false` ⇒ plain
    // KV+recurrent advance. Empty commit (non-committable seed + single forced)
    // skips GPU entirely — the forced token only replaces the pending seed.
    //
    // Note: DFlash/DSpark `on_forced_advance` currently returns Ok(true) even
    // when the underlying advance aborted (state is torn down by the caller).
    // Rely on the abort latch + post-check rather than SpecAdvance::Aborted
    // alone when handled.
    let forced_res = if tx.commit.is_empty() {
        Ok(())
    } else {
        match spec.on_forced_advance(gpu, slot, &tx.commit, *position, &abort_cb) {
            Ok(true) => Ok(()),
            Ok(false) => {
                match slot.spec_advance(gpu, &tx.commit, *position, false, &abort_cb, None) {
                    Ok(SpecAdvance::Ready { .. }) => Ok(()),
                    Ok(SpecAdvance::Aborted) => {
                        abort_latched.set(true);
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            Err(e) => Err(e),
        }
    };
    if let Err(e) = forced_res {
        // Authoritative failure: reset+sync first, then one correlated error;
        // no finish/calls/cache/done. Staged outcomes are discarded.
        let ep = production_fail_closed_rollback_live(
            seq_pos,
            conversation_tokens,
            prefill_checkpoints,
            dflash_checkpoints,
            asst_turn_cache,
            gpu,
            slot,
            spec,
        );
        emit_spec_failure_terminal(stdout, id, "forced", &e.to_string(), &ep);
        return SpecForcedApplyResult::Terminal;
    }
    let abort_observed = abort_latched.get() || check_abort(id);
    if matches!(
        classify_forced_gpu_advance(abort_observed),
        ForcedGpuAdvanceKind::Cancelled
    ) {
        let ep = production_fail_closed_rollback_live(
            seq_pos,
            conversation_tokens,
            prefill_checkpoints,
            dflash_checkpoints,
            asst_turn_cache,
            gpu,
            slot,
            spec,
        );
        emit_spec_cancel_after_rollback(stdout, id, *generated, &ep);
        return SpecForcedApplyResult::Terminal;
    }
    // GPU committed kept prefix (trigger only if committable + forced[..n-1]);
    // last kept forced is pending. position_delta tracks actual commit length.
    // Post-commit eviction (when active) MUST complete before host seed/raw/render
    // so client-visible state never observes over-cap physical occupancy.
    let mut post_position = position.saturating_add(tx.position_delta);
    if let Some(ev) = eviction {
        let evict_outcome = match slot.kv_cache_mut() {
            Some(kv) => ev.maybe_evict(gpu, kv, post_position),
            None => {
                let _ = classify_evict_failure_wire();
                let ep = production_fail_closed_rollback_live(
                    seq_pos,
                    conversation_tokens,
                    prefill_checkpoints,
                    dflash_checkpoints,
                    asst_turn_cache,
                    gpu,
                    slot,
                    spec,
                );
                emit_fail_closed_error(
                    stdout,
                    Some(id),
                    "kv_cache_mut missing (forced)",
                    "validation",
                    false,
                    &ep,
                );
                return SpecForcedApplyResult::Terminal;
            }
        };
        match evict_outcome {
            Ok(Some(res)) => {
                let pre_phys = post_position;
                if !res.retain_mask.is_empty() {
                    if let Err(e) = spec.on_evict(
                        gpu,
                        &EvictRetain {
                            retain_mask: res.retain_mask,
                            pre_phys,
                        },
                    ) {
                        let _ = classify_evict_failure_wire();
                        let ep = production_fail_closed_rollback_live(
                            seq_pos,
                            conversation_tokens,
                            prefill_checkpoints,
                            dflash_checkpoints,
                            asst_turn_cache,
                            gpu,
                            slot,
                            spec,
                        );
                        emit_fail_closed_error(
                            stdout,
                            Some(id),
                            &format!("on_evict (forced): {e}"),
                            "validation",
                            false,
                            &ep,
                        );
                        return SpecForcedApplyResult::Terminal;
                    }
                }
                post_position = res.new_physical;
            }
            Ok(None) => {}
            Err(e) => {
                let _ = classify_evict_failure_wire();
                let ep = production_fail_closed_rollback_live(
                    seq_pos,
                    conversation_tokens,
                    prefill_checkpoints,
                    dflash_checkpoints,
                    asst_turn_cache,
                    gpu,
                    slot,
                    spec,
                );
                emit_fail_closed_error(
                    stdout,
                    Some(id),
                    &format!("maybe_evict (forced): {e}"),
                    "validation",
                    false,
                    &ep,
                );
                return SpecForcedApplyResult::Terminal;
            }
        }
        // Pending seed requires a legal next write slot after eviction.
        if post_position >= physical_cap {
            let _ = classify_evict_failure_wire();
            let ep = production_fail_closed_rollback_live(
                seq_pos,
                conversation_tokens,
                prefill_checkpoints,
                dflash_checkpoints,
                asst_turn_cache,
                gpu,
                slot,
                spec,
            );
            emit_fail_closed_error(
                stdout,
                Some(id),
                &format!(
                    "forced commit leaves no pending-seed slot: position={post_position} >= physical_cap={physical_cap}"
                ),
                "validation",
                false,
                &ep,
            );
            return SpecForcedApplyResult::Terminal;
        }
    }

    *position = post_position;
    *seed_token = tx.pending_seed;
    // Post-commit host path only: raw history + client render for staged prefix.
    // Last kept observe owns pending_seed_committable for the new seed.
    for (ft, fo) in staged {
        raw_decode.push(ft);
        *pending_seed_committable = spec_outcome_seed_committable(&fo);
        if !fo.events.is_empty() {
            emitted.push(ft);
            render_client_events(
                stdout,
                id,
                &fo.events,
                t0.elapsed().as_millis() as u64,
                false,
            );
            *generated += 1;
        }
    }
    match stop_reason {
        Some(reason) => SpecForcedApplyResult::Stopped(reason),
        None => SpecForcedApplyResult::Applied,
    }
}

/// Eviction / on_evict failure always uses the exclusive error terminal.
pub fn classify_evict_failure_wire() -> SpecFailClosedWire {
    SpecFailClosedWire::ErrorOnly
}

// --- iter appended ---

/// Pure Commit/Abort side-effect gate for Qwen AR + DFlash successful terminals.
/// Commit preserves the intended release/store/done flags; Abort suppresses all
/// three so producers never store or release after a client cancel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenClientCommitEffects {
    pub release_tool_calls: bool,
    pub store_cache: bool,
    pub emit_done: bool,
}

/// Cache-store action for Qwen DFlash — production and tests share this seam.
#[derive(Debug, Clone)]
pub struct QwenDflashCacheAction {
    pub store: bool,
    pub fingerprint_text: String,
    pub tool_calls: Vec<hipfire_runtime::prompt_frame::ToolCall>,
}

/// Host-side bookkeeping after one SpecStep is consumed by the semantic loop.
///
/// `consumed` is the number of step.emit tokens the emitter actually observed
/// (may be a strict prefix on EOT/stop/forced). Position and conversation must
/// advance by that prefix only — never by an unconsumed speculative tail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecHostAdvance {
    pub position: usize,
    pub generated: usize,
    pub emitted: Vec<u32>,
    pub seed_token: u32,
}

/// GPU-side pending-seed transaction for `generate_spec`.
///
/// # Pending-seed invariant (single source of truth)
///
/// After every safe boundary in `generate_spec`:
/// - `position` is the next target **write** slot.
/// - `seed_token` is the **unprocessed** pending seed (not yet in KV /
///   DeltaNet / drafter-hidden). Never infer GPU progress from client-event
///   counts.
/// - `Speculator::step(position, seed)` leaves target/drafter advanced over
///   the prior pending seed plus accepted drafts; `step.emit` is those
///   accepted drafts plus the bonus; `step.next_seed` is emitted but
///   **unprocessed**.
/// - A forced suffix commits the current pending seed only when it is
///   committable, then every forced token except the last; the last forced
///   token becomes the new pending seed. Non-committable seeds (DS4 empty-
///   event EOS) are omitted so a forced token can occupy that same slot
///   (`position_delta == commit.len()`; last stays unwritten).
/// - Safe terminal completion flushes the final pending seed **exactly once**
///   before host cache / conversation bake.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecPendingSeedTx {
    /// Tokens to GPU-commit at `position`, in order. Length == `position_delta`.
    pub commit: Vec<u32>,
    /// Unprocessed pending seed after the commit (not present in `commit`).
    pub pending_seed: u32,
    /// How far to advance `position` (== `commit.len()`).
    pub position_delta: usize,
}

/// Build the forced-continuation GPU transaction.
///
/// `pending_seed` is the current unwritten seed (trigger / last consumed).
/// `forced` is the non-empty emitter continuation (already budget-clipped).
/// `pending_seed_committable` gates whether the current seed is prepended:
/// suppressed empty-event EOS must not be GPU-written; forced tokens then
/// occupy that same slot.
///
/// - if committable: `commit = [pending_seed] ++ forced[..n-1]`
/// - if not: `commit = forced[..n-1]` (may be empty when `forced.len() == 1`)
/// - new pending seed = `forced[n-1]`
/// - `position_delta = commit.len()`
///
/// The last forced token is never double-forwarded: it remains the next
/// window's unprocessed seed (or the terminal flush input).
pub fn spec_forced_pending_seed_tx(
    pending_seed: u32,
    forced: &[u32],
    pending_seed_committable: bool,
) -> SpecPendingSeedTx {
    debug_assert!(
        !forced.is_empty(),
        "forced continuation tx requires at least one forced token"
    );
    let mut commit = Vec::with_capacity(forced.len());
    if pending_seed_committable {
        commit.push(pending_seed);
    }
    if forced.len() > 1 {
        commit.extend_from_slice(&forced[..forced.len() - 1]);
    }
    SpecPendingSeedTx {
        pending_seed: *forced.last().expect("forced non-empty"),
        position_delta: commit.len(),
        commit,
    }
}

/// Hard `max_tokens` ceiling for forced tokens: no GPU commit for a token
/// that cannot fit. `generated` already includes the trigger when it produced
/// client events. Returns a (possibly empty) prefix of `forced`.
pub fn spec_forced_tokens_within_budget<'a>(
    generated: usize,
    max_tokens: usize,
    forced: &'a [u32],
) -> &'a [u32] {
    let room = max_tokens.saturating_sub(generated);
    let n = forced.len().min(room);
    &forced[..n]
}

/// Pure plan for mid-window GPU/drafter realign after a strict-prefix consume.
///
/// Prefill leaves `position = filled.len()` and the next `step` WRITES `seed` at
/// that slot. Therefore the last consumed token must remain the unwritten seed
/// and must NOT be included in the prefill replay.
///
/// `raw_decode` is every position-advancing token after the prefill `first_token`
/// (includes empty-event EOS observes; excludes the prefill seed itself).
/// - raw nonempty: replay = prompt + [first_token] + raw[..len-1],
///   position = prompt.len() + raw.len(), seed = raw.last()
/// - raw empty: replay = prompt, position = prompt.len(), seed = first_token
///
/// Invariant: `replay.len() == position` (processed prefix only). The pending
/// seed is host-side and unprocessed until the next commit. Callers MUST run
/// [`spec_prefix_realign_admit`] before any reset/prefill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecPrefixRealignPlan {
    /// Tokens that will be GPU-prefilled (processed positions only).
    pub replay: Vec<u32>,
    /// Next target write slot (= `replay.len()`); pending seed is unwritten.
    pub position: usize,
    /// Pending unprocessed seed (not in `replay`).
    pub seed_token: u32,
}

/// Message text used by live spec-step / forced-advance failure branches.
pub fn spec_failure_message(what: &str, err: &str) -> String {
    match what {
        "forced" | "forced-token advance" => format!("forced-token advance: {err}"),
        _ => format!("spec_step: {err}"),
    }
}

/// Pure physical-cap admission for a forced pending-seed GPU commit slice.
///
/// `commit_len` is the exact transaction commit length (`tx.commit.len()`), not
/// the forced token count (non-committable seed may omit the trigger).
/// - No eviction: admits iff `position + commit_len < physical_cap` so one legal
///   write slot remains for the unprocessed pending seed.
/// - Eviction enabled: admits iff `position + commit_len <= physical_cap`. Exact-cap
///   is allowed only because the caller MUST run post-commit `maybe_evict` /
///   `on_evict` before host seed/raw/render updates and still leave a legal
///   pending-seed slot (`post_evict < physical_cap`).
pub fn spec_forced_commit_admits(
    position: usize,
    commit_len: usize,
    physical_cap: usize,
    eviction_enabled: bool,
) -> bool {
    let post_position = position.saturating_add(commit_len);
    if eviction_enabled {
        post_position <= physical_cap
    } else {
        post_position < physical_cap
    }
}

/// Wire shape for fail-closed terminals on the Task 4 spec path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecFailClosedWire {
    /// Correlated error only — never done / calls / cache.
    ErrorOnly,
    /// Client cancel — aborted + done(finish_reason=aborted).
    Cancelled,
}

/// Outcome of the LCP prompt-cache decision (see [`plan_prompt_cache`]).
pub struct PromptCachePlan {
    /// Full canonical conversation tokens (system + history + live user +
    /// assistant prefix). Stored as `conversation_tokens` after generation so
    /// the next turn can LCP against it.
    pub rendered: Vec<u32>,
    /// Tokens to actually prefill: the suffix `rendered[start_pos..]` on a hit,
    /// the whole `rendered` on a miss.
    pub new_tokens: Vec<u32>,
    /// Absolute position the prefill starts at (the reused-prefix length on a
    /// hit, 0 on a miss).
    pub start_pos: usize,
    /// `cached_tokens` for OpenAI usage reporting (== start_pos).
    pub cached_tokens: usize,
    /// True ⇒ reuse existing KV/DeltaNet[0..start_pos]; prefill only the suffix.
    /// False ⇒ caller must full-reset and prefill the whole conversation.
    pub cache_hit: bool,
    /// `Some(ckpt)` ⇒ this is a divergent-render RESUME (not a pure extension):
    /// the caller must restore the DeltaNet recurrent state from the checkpoint
    /// at `ckpt`, rewind seq_pos/conversation_tokens to `ckpt`, then treat the
    /// turn like a HIT with `start_pos == ckpt` (re-prefill only the tail) and
    /// drop `draft_ctx_cached_rows` to `ckpt`. `None` on a normal hit/miss.
    pub resume_from: Option<usize>,
}

/// Opt-in per-window HIP host/API snapshot. Constructed only when
/// `HIPFIRE_HOST_TIMING=1`; the disabled path never reads clocks or counters.
pub struct MtpWindowTimingSnap {
    pub wall_start: Instant,
    pub l_start: u64,
    pub htod_start: u64,
    pub dtoh_start: u64,
    pub dtod_start: u64,
    pub memset_start: u64,
    pub ssync_start: u64,
    pub esync_start: u64,
    pub dsync_start: u64,
    pub glaunch_start: u64,
}

impl MtpWindowTimingSnap {
    pub fn take() -> Self {
        use hip_bridge::launch_counters as lc;
        Self {
            wall_start: Instant::now(),
            l_start: lc::launch_kernel::time_ns(),
            htod_start: lc::memcpy_htod::time_ns(),
            dtoh_start: lc::memcpy_dtoh::time_ns(),
            dtod_start: lc::memcpy_dtod::time_ns(),
            memset_start: lc::memset::time_ns(),
            ssync_start: lc::stream_sync::time_ns(),
            esync_start: lc::event_sync::time_ns(),
            dsync_start: lc::device_sync::time_ns(),
            glaunch_start: lc::graph_launch::time_ns(),
        }
    }

    /// Consume the snapshot after a successful step into one ordered wire record.
    pub fn into_record(self, kind: &str, draft_lookup_us: u64) -> serde_json::Value {
        use hip_bridge::launch_counters as lc;
        mtp_window_timing_record(
            kind,
            self.wall_start.elapsed().as_micros() as u64,
            draft_lookup_us,
            (lc::launch_kernel::time_ns() - self.l_start) / 1000,
            (lc::memcpy_htod::time_ns() - self.htod_start) / 1000,
            (lc::memcpy_dtoh::time_ns() - self.dtoh_start) / 1000,
            (lc::memcpy_dtod::time_ns() - self.dtod_start) / 1000,
            (lc::memset::time_ns() - self.memset_start) / 1000,
            (lc::stream_sync::time_ns() - self.ssync_start) / 1000,
            (lc::event_sync::time_ns() - self.esync_start) / 1000,
            (lc::device_sync::time_ns() - self.dsync_start) / 1000,
            (lc::graph_launch::time_ns() - self.glaunch_start) / 1000,
        )
    }
}

#[cfg(test)]
mod deepseek4_reasoning_prefix_tests {
    use super::{
        deepseek4_reasoning_prefix, ThinkMode, DEEPSEEK4_REASONING_HIGH_PREFIX,
        DEEPSEEK4_REASONING_MAX_PREFIX,
    };

    #[test]
    fn parent_effort_prefixes_are_distinct_and_low_is_empty() {
        assert_eq!(deepseek4_reasoning_prefix(ThinkMode::NonThink), "");
        assert_eq!(deepseek4_reasoning_prefix(ThinkMode::Low), "");
        assert_eq!(
            deepseek4_reasoning_prefix(ThinkMode::High),
            DEEPSEEK4_REASONING_HIGH_PREFIX
        );
        assert_eq!(
            deepseek4_reasoning_prefix(ThinkMode::Max),
            DEEPSEEK4_REASONING_MAX_PREFIX
        );
        assert_ne!(
            DEEPSEEK4_REASONING_HIGH_PREFIX,
            DEEPSEEK4_REASONING_MAX_PREFIX
        );
        assert!(DEEPSEEK4_REASONING_HIGH_PREFIX.ends_with("\n\n"));
        assert!(DEEPSEEK4_REASONING_MAX_PREFIX.ends_with("\n\n"));
    }
}

// --- iter appended ---
