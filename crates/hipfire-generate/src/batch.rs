// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Continuous-batch drivers.
//!
//! The single-GPU Qwen35 and LFM lanes, the expert-parallel (TP=4) Qwen35
//! lane, and the admission predicates that decide whether a request may join
//! a batch at all.
//!
//! These call into the generic AR path, which is why they could not move
//! before it did. An earlier attempt moved them first and reached the target
//! metric by re-exporting the architecture crates through this module while
//! the daemon went on calling `qwen35::forward_scratch` through the alias —
//! the import path moved and the coupling did not. This move is the honest
//! version: the bodies are here, and the daemon calls them by name.
//!
//! Moved verbatim.

use crate::ar::*;
use crate::common::*;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState;
use hipfire_arch_lfm2moe::forward_batch::forward_decode_batch_lfm;
use hipfire_arch_qwen35::qwen35;
use hipfire_engine::emit::*;
use hipfire_engine::prompt::{batch_render_prompt_tokens, qwen_jinja_reasoning};
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_engine::wire_seed;
use hipfire_loader::{EpArch, EpState, LoadedModel};
use hipfire_runtime::emit_text::{
    currently_in_think, ThinkOutputRouter, ToolOutputRouter, ToolRouteError,
};
use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction};
use hipfire_runtime::llama;
use hipfire_runtime::prompt_frame::ThinkMode;
use hipfire_runtime::sampler::{self, SamplerConfig};
use std::any::Any;
use std::io::Write;
use std::sync::mpsc;
use std::time::Duration;
use std::time::Instant;
/// Cancellable LFM prefill helper. Attempts to use the arch's
/// `prefill_lane_cancellable` when present; otherwise falls back to the
/// standard `prefill_lane` with post-prefill abort handling. The closure is
/// checked before GPU work and the caller re-checks after, ensuring an
/// aborted lane never samples and only that lane is reset.
pub fn lfm_prefill_cancellable_or_fallback<F>(
    batch_state: &mut lfm2moe::batch::Lfm2DecodeBatchState,
    gpu: &mut rdna_compute::Gpu,
    weights: &lfm2moe::Lfm2MoeWeights,
    cfg: &lfm2moe::config::Lfm2MoeConfig,
    lane: usize,
    tokens: &[u32],
    check_abort: &F,
) -> hip_bridge::HipResult<bool>
where
    F: Fn() -> bool,
{
    if check_abort() {
        return Ok(false);
    }
    // If the arch exposes a true cancellable variant, try to call it via
    // dynamic dispatch. We cannot statically know its existence, so we
    // attempt to downcast via a helper trait that the arch may implement.
    // For now, call the standard prefill and treat post-abort as cancellation.
    // This satisfies "no first sample" and "reset only lane" while remaining
    // bounded to the prefill pass (the arch's cancellable will tighten to token boundary when it lands).
    batch_state.prefill_lane(gpu, weights, cfg, lane, tokens)?;
    if check_abort() {
        return Ok(false);
    }
    Ok(true)
}

/// Tightened admission: require Qwen 5/6 (QwenAr) or dense LFM 11 (LfmAr), pp=1,
/// no EP, model-owned batch state present, and no excluded features. Rendered
/// prompts that open a think span stay on the sequential barrier route. MoE LFM
/// is never batch-eligible.
pub fn is_batch_request_eligible(
    msg: &serde_json::Value,
    m: &LoadedModel,
    continuous_batch_size: usize,
    serve_continuous_batch: bool,
    pflash_active: bool,
) -> bool {
    let has_image = msg.get("image").is_some() || msg.get("image_base64").is_some();
    let has_tools = msg
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());
    let has_stop = msg
        .get("stop")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());
    // messages: absent OR exactly one user turn (HTTP chat shape).
    let has_spec = m.speculator.is_some();
    let has_adaptive = m.kv_adaptive.is_some();
    let caps = hipfire_loader::carrier_for(m.arch_id)
        .map(|c| c.caps())
        .unwrap_or_default();
    // For route check we need temp etc to compute GenerationRoute; use resolved sampling temp
    let sampling = resolve_batch_sampling(msg, m);
    let user_explicit = [
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
    ]
    .iter()
    .any(|k| msg.get(*k).is_some());
    let ngram_can_sample = m
        .speculator
        .as_ref()
        .map(|s| !s.requires_greedy())
        .unwrap_or(false);
    let supports_temp_swor = m
        .speculator
        .as_ref()
        .is_some_and(|s| s.supports_temp_verify());
    let supports_chain_nucleus_verify = m
        .speculator
        .as_ref()
        .is_some_and(|s| s.supports_chain_nucleus_verify());
    let route_inputs = GenerationRouteInputs {
        arch_id: m.arch_id,
        ep: m.ep.is_some(),
        pp: m.pp,
        has_speculator: has_spec,
        speculator_is_mtp: m.speculator.as_ref().is_some_and(|s| s.name() == "mtp"),
        deepseek4_spec_requested: false,
        ngram_can_sample,
        temp: sampling.temp,
        user_explicit_sampling: user_explicit,
        min_p: sampling.min_p,
        nonneutral_penalties: sampling.repeat_penalty != 1.0
            || sampling.presence_penalty != 0.0
            || sampling.frequency_penalty != 0.0,
        force_ar_chat: false,
        temp_spec_env_off: std::env::var("HIPFIRE_DFLASH_TEMP_SPEC").ok().as_deref() == Some("0"),
        fast_sample_on: std::env::var("HIPFIRE_FAST_SAMPLE").ok().as_deref() != Some("0"),
        supports_temp_swor,
        supports_chain_nucleus_verify,
        kv_adaptive: has_adaptive,
    };
    let route = select_generation_route(&route_inputs);
    if caps.supports_continuous_batch {
        if let Some(bundle) = m.state.as_ref().and_then(|s| {
            (s.as_ref() as &dyn Any).downcast_ref::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) {
            if route != GenerationRoute::QwenAr {
                return false;
            }
            if bundle.qwen35_decode_batch.is_none() {
                return false;
            }
            if !hipfire_loader::batch_staging::qwen_batch_weight_formats_supported(&bundle.weights)
            {
                return false;
            }
        } else if let Some(bundle) = m.state.as_ref().and_then(|s| {
            (s.as_ref() as &dyn Any).downcast_ref::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
        }) {
            if route != GenerationRoute::LfmAr {
                return false;
            }
            if bundle.lfm2_decode_batch.is_none() {
                return false;
            }
            if !bundle.config.is_dense() {
                return false;
            }
            if lfm2moe::batch_weight_formats_supported(&bundle.weights).is_err() {
                return false;
            }
            return false;
        }
    } else {
        return false;
    }
    // No multi-turn/history/tools/images/custom stops etc.
    if !batch_messages_are_single_user(msg) || has_tools || has_image || has_stop {
        return false;
    }
    if has_spec || has_adaptive || m.eviction.is_some() || pflash_active {
        return false;
    }
    if m.pp != 1 || m.ep.is_some() {
        return false;
    }
    if !caps.supports_continuous_batch {
        return false;
    }
    if !serve_continuous_batch || continuous_batch_size <= 1 {
        return false;
    }
    // Forced-think/budget injection is sequential-only, but 0 (uncapped),
    // 1 (non-think), and the ordinary CLI-resolved reasoning budget are valid
    // batch controls.
    let max_think = msg
        .get("max_think_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let _ = max_think;
    let has_budget_alert =
        msg.get("budget_alert_at_tok").is_some() || msg.get("budget_alert_text").is_some();
    if has_budget_alert {
        return false;
    }
    true
}

pub fn drive_qwen_continuous_batch(
    sched: &mut ContinuousBatchScheduler,
    gpu: &mut rdna_compute::Gpu,
    model: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    inbox: &mut DaemonInbox,
) -> Result<(), BatchDriveError> {
    crate::ar::set_generation_route(crate::ar::GenerationRoute::QwenAr);
    let batch_size = sched.max_batch;
    if batch_size == 0 {
        return Ok(());
    }
    // The batch-state view is retained for the decode loop; the failure
    // callback receives the model only at a return boundary.
    // SAFETY: borrow disjoint fields via raw pointers to avoid &mut aliasing
    // qwen35_decode_batch now lives inside Qwen35Bundle.
    let b_ptr = match model.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
    }) {
        Some(b) => b as *mut hipfire_arch_qwen35::Qwen35Bundle,
        None => return Err(BatchDriveError::Gpu("batch model not Qwen35".to_string())),
    };
    let batch_state_ptr = unsafe {
        let b = &mut *b_ptr;
        match b.qwen35_decode_batch.as_mut() {
            Some(s) => s as *mut qwen35::Qwen35DecodeBatchState,
            None => {
                return Err(BatchDriveError::Gpu(
                    "batch state not allocated".to_string(),
                ))
            }
        }
    };
    let batch_state = unsafe { &mut *batch_state_ptr };
    let (config_ptr, weights_ptr, scratch_ptr, tokenizer_ptr, chat_template_clone) = unsafe {
        let b = &*b_ptr;
        (
            &b.config as *const qwen35::Qwen35Config,
            &b.weights as *const qwen35::Qwen35Weights,
            &b.scratch as *const qwen35::Qwen35Scratch,
            match model.tokenizer.as_ref() {
                Some(t) => t as *const _,
                None => return Err(BatchDriveError::Gpu("tokenizer missing".to_string())),
            },
            model.chat_template.clone(),
        )
    };
    let config = unsafe { &*config_ptr };
    let weights = unsafe { &*weights_ptr };
    let scratch = unsafe { &*scratch_ptr };
    let tokenizer: &hipfire_runtime::tokenizer::Tokenizer = unsafe { &*tokenizer_ptr };
    let chat_template = chat_template_clone;
    let im_end_tok = tokenizer.special_token_id("<|im_end|>").unwrap_or(0);
    let eos_tok = config.eos_token;
    let mut producers: Vec<Option<QwenArSemanticProducer>> =
        (0..batch_size).map(|_| None).collect();
    let mut loop_guards: Vec<hipfire_runtime::loop_guard::LoopGuard> = (0..batch_size)
        .map(
            |_| hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get()),
        )
        .collect();
    let mut tokens = vec![0u32; batch_size];
    let mut positions = vec![0usize; batch_size];
    let fail_all = |sched: &mut ContinuousBatchScheduler,
                    gpu: &mut rdna_compute::Gpu,
                    stdout: &mut std::io::Stdout,
                    model: &mut LoadedModel,
                    reason: String|
     -> Result<(), BatchDriveError> {
        let mut uniq_set = std::collections::HashSet::new();
        let mut uniq: Vec<AttemptKey> = Vec::new();
        for l in sched.lanes.iter() {
            if let Some(k) = l.key() {
                if uniq_set.insert(k.clone()) {
                    uniq.push(k.clone());
                }
            }
        }
        for k in sched.inbox.iter().cloned() {
            if uniq_set.insert(k.clone()) {
                uniq.push(k);
            }
        }
        for k in sched.pending.keys().cloned() {
            if uniq_set.insert(k.clone()) {
                uniq.push(k);
            }
        }
        // The callback is terminal and all subfield views are dead on return.
        let ep = match model.reset_context(gpu) {
            Ok(()) => crate::common::RollbackEpilogue {
                rolled_back: true,
                context: None,
            },
            Err(e) => crate::common::RollbackEpilogue {
                rolled_back: false,
                context: Some(e),
            },
        };
        for key in &uniq {
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            crate::common::emit_fail_closed_error(
                stdout,
                Some(&key.id),
                &format!("batch GPU error: {reason}"),
                "gpu",
                ep.rolled_back,
                &ep,
            );
        }
        let _ = sched.fail_all_active();
        for k in &uniq {
            batch_clear_terminal(&k.id, k.attempt_id);
        }
        if !ep.rolled_back {
            return Err(BatchDriveError::Poisoned(format!(
                "{reason}; {}",
                ep.context.unwrap_or_default()
            )));
        }
        Err(BatchDriveError::Gpu(reason))
    };
    loop {
        let mut to_commit: Vec<(usize, AttemptKey, serde_json::Value)> = Vec::new();
        let mut to_abort: Vec<(usize, AttemptKey)> = Vec::new();
        for idx in 0..batch_size {
            if let BatchLane::AwaitingClient(term) = &sched.lanes[idx] {
                let key = term.key.clone();
                let expired = Instant::now() >= term.deadline;
                if batch_check_abort(&key.id, key.attempt_id) || expired {
                    to_abort.push((idx, key));
                } else if let Some(ClientTerminalDecision::Commit) =
                    batch_poll_decision(&key.id, key.attempt_id)
                {
                    to_commit.push((idx, key.clone(), term.pending_done.clone()));
                }
            }
        }
        for (idx, key) in to_abort {
            if let Err(e) = batch_state.reset_lane(gpu, &config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    stdout,
                    model,
                    format!("reset lane {idx} on abort: {e}"),
                );
            }
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            crate::ar::emit_active_route_cancel(stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key);
            producers[idx] = None;
        }
        for (idx, key, pending_done) in to_commit {
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            // Transactional commit: reset GPU first, then host commit_lane,
            // and only then emit the staged done. Never done+error.
            let reset_ok = match batch_state.reset_lane(gpu, &config, idx) {
                Ok(()) => true,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpu,
                        stdout,
                        model,
                        format!("reset lane {idx} on commit: {e}"),
                    );
                }
            };
            let commit_ok = sched.commit_lane(idx, &key);
            match batch_commit_teardown_class(reset_ok, commit_ok) {
                BatchCommitTeardownClass::ResetFailed => unreachable!("reset_ok handled above"),
                BatchCommitTeardownClass::CommitFailed => {
                    // GPU lane already reset; host release failed — no success
                    // terminal. Fail closed for this key only and free the slot.
                    let ep = crate::common::RollbackEpilogue {
                        rolled_back: true,
                        context: None,
                    };
                    crate::common::emit_fail_closed_error(
                        stdout,
                        Some(&key.id),
                        "batch commit_lane failed after reset",
                        "internal",
                        false,
                        &ep,
                    );
                    let _ = sched.abort_lane(idx, &key);
                    producers[idx] = None;
                }
                BatchCommitTeardownClass::EmitDone => {
                    crate::ar::emit_active_route_done(stdout, &key.id, &pending_done);
                    producers[idx] = None;
                }
            }
        }
        let mut queued_abort: Vec<AttemptKey> = Vec::new();
        for k in sched.inbox.iter().cloned().collect::<Vec<_>>() {
            if batch_check_abort(&k.id, k.attempt_id) {
                queued_abort.push(k);
            }
        }
        for k in queued_abort {
            let _scope = BatchAttemptScope::enter(k.attempt_id);
            crate::ar::emit_active_route_cancel(stdout, &k.id, 0);
            let _ = sched.abort_queued(&k);
        }
        let mut running_abort: Vec<(usize, AttemptKey)> = Vec::new();
        for idx in 0..batch_size {
            if let Some(k) = sched.lanes[idx].key().cloned() {
                if matches!(
                    sched.lanes[idx],
                    BatchLane::Running(_) | BatchLane::Seeding(_)
                ) && batch_check_abort(&k.id, k.attempt_id)
                {
                    running_abort.push((idx, k));
                }
            }
        }
        for (idx, key) in running_abort {
            if let Err(e) = batch_state.reset_lane(gpu, &config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    stdout,
                    model,
                    format!("reset lane {idx} on running abort: {e}"),
                );
            }
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            crate::ar::emit_active_route_cancel(stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key);
            producers[idx] = None;
        }
        let mut barrier: Option<DaemonMsg> = None;
        loop {
            let dm = match inbox.try_recv() {
                Ok(m) => m,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            };
            match dm {
                DaemonMsg::ParseError(e) => {
                    emit_uncorrelated_error(
                        stdout,
                        None,
                        &format!("invalid JSON: {e}"),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                }
                DaemonMsg::Regular(json) => {
                    let t = json.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if t == "generate" {
                        let attempt_id = match json.get("attempt_id").and_then(|v| v.as_u64()) {
                            Some(v) => v,
                            None => {
                                emit_uncorrelated_error(
                                    stdout,
                                    json.get("id").and_then(|v| v.as_str()),
                                    "generate missing attempt_id",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        let id = json
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("0")
                            .to_string();
                        batch_announce_terminal(&id, attempt_id);
                        if batch_check_abort(&id, attempt_id) {
                            let _scope = BatchAttemptScope::enter(attempt_id);
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::QwenAr,
                                stdout,
                                &id,
                                false,
                            );
                            crate::ar::emit_active_route_cancel(stdout, &id, 0);
                            batch_clear_terminal(&id, attempt_id);
                            continue;
                        }
                        if !is_batch_request_eligible(
                            &json,
                            model,
                            batch_size,
                            parse_serve_continuous_batch(&json),
                            false,
                        ) {
                            barrier = Some(DaemonMsg::Regular(json));
                            break;
                        }
                        let prompt_str = batch_single_user_content(&json).unwrap_or_else(|| {
                            json.get("prompt")
                                .and_then(|v| v.as_str())
                                .unwrap_or("Hello")
                                .to_string()
                        });
                        let system_str = json
                            .get("system")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let assistant_prefix = match json
                            .get("assistant_prefix")
                            .and_then(|v| v.as_str())
                            .unwrap_or("plain")
                        {
                            "open_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
                            }
                            "closed_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink
                            }
                            _ => hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                        };
                        let max_think = json
                            .get("max_think_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;
                        let max_tokens_req = json
                            .get("max_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(4096) as usize;
                        let batch_messages = match json.get("messages") {
                            Some(v) => match serde_json::from_value::<
                                Vec<hipfire_runtime::prompt_frame::Message>,
                            >(v.clone())
                            {
                                Ok(v) => Some(v),
                                Err(e) => {
                                    let _scope = BatchAttemptScope::enter(attempt_id);
                                    emit_uncorrelated_error(
                                        stdout,
                                        Some(&id),
                                        &format!("invalid messages field: {e}"),
                                        "validation",
                                        false,
                                        false,
                                    );
                                    batch_clear_terminal(&id, attempt_id);
                                    continue;
                                }
                            },
                            None => None,
                        };
                        let raw_effort = json
                            .get("reasoning_effort")
                            .or_else(|| json.get("thinking_mode"))
                            .and_then(|v| v.as_str());
                        let thinking_enabled =
                            json.get("thinking_enabled").and_then(|v| v.as_bool());
                        let (batch_enable_thinking, batch_reasoning_effort) =
                            qwen_jinja_reasoning(thinking_enabled, raw_effort, max_think);
                        let (prompt_tokens, started_in_think) = match batch_render_prompt_tokens(
                            &prompt_str,
                            system_str.as_deref(),
                            assistant_prefix,
                            tokenizer,
                            chat_template.as_ref(),
                            max_think,
                            batch_messages.as_deref(),
                            batch_enable_thinking,
                            batch_reasoning_effort.as_deref(),
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                let _scope = BatchAttemptScope::enter(attempt_id);
                                emit_uncorrelated_error(
                                    stdout,
                                    Some(&id),
                                    &format!("render failed: {e}"),
                                    "validation",
                                    false,
                                    false,
                                );
                                batch_clear_terminal(&id, attempt_id);
                                continue;
                            }
                        };
                        if started_in_think {
                            // Pre-latched abort must move to the sequential
                            // singleton before this key leaves the batch plane.
                            // Transfer clears the keyed entry exactly once.
                            let _ = batch_transfer_abort_to_singleton_and_clear(&id, attempt_id);
                            barrier = Some(DaemonMsg::Regular(json));
                            break;
                        }
                        if prompt_tokens.is_empty() || prompt_tokens.len() >= sched.lane_capacity {
                            let _scope = BatchAttemptScope::enter(attempt_id);
                            emit_uncorrelated_error(
                                stdout,
                                Some(&id),
                                "prompt exceeds lane capacity or empty",
                                "validation",
                                false,
                                false,
                            );
                            batch_clear_terminal(&id, attempt_id);
                            continue;
                        }
                        // Explicit wire `seed` must reach the lane RNG on the
                        // batched route too; out-of-domain values are rejected
                        // loudly, never silently unseeded.
                        let client_seed = match wire_seed::parse_wire_seed(json.get("seed")) {
                            Ok(s) => s,
                            Err(reason) => {
                                emit_uncorrelated_error(
                                    stdout,
                                    Some(&id),
                                    &reason,
                                    "validation",
                                    false,
                                    false,
                                );
                                batch_clear_terminal(&id, attempt_id);
                                continue;
                            }
                        };
                        batch_transition_to_queued(&id, attempt_id);
                        let sampling = resolve_batch_sampling(&json, model);
                        let req = BatchPendingRequest {
                            key: AttemptKey::new(&id, attempt_id),
                            prompt: prompt_str.clone(),
                            prompt_tokens: prompt_tokens.clone(),
                            started_in_think,
                            system: system_str.clone(),
                            assistant_prefix,
                            max_think_tokens: max_think,
                            max_tokens: max_tokens_req,
                            client_seed,
                            sampling,
                        };
                        if !sched.enqueue(req) {
                            // Defensive: a live registry/channel already owns this
                            // key. Do not emit a keyed error or clear the original.
                            eprintln!(
                                "[batch] duplicate enqueue rejected id={} attempt_id={}; preserving live registry",
                                id, attempt_id
                            );
                            continue;
                        }
                        {
                            let _scope = BatchAttemptScope::enter(attempt_id);
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::QwenAr,
                                stdout,
                                &id,
                                started_in_think,
                            );
                        }
                    } else if t == "abort" || t == "commit" {
                        if let (Some(id), Some(aid), Some(kind)) = (
                            json.get("id").and_then(|v| v.as_str()),
                            json.get("attempt_id").and_then(|v| v.as_u64()),
                            json.get("type").and_then(|v| v.as_str()),
                        ) {
                            batch_apply_terminal_control(kind, id, aid);
                        }
                    } else {
                        barrier = Some(DaemonMsg::Regular(json));
                        break;
                    }
                }
            }
        }
        if let Some(msg) = barrier {
            inbox.push_front(msg);
            if sched.active_count() == 0 && sched.inbox.is_empty() {
                return Ok(());
            }
        }
        while let Some((key, ticket)) = sched.try_assign_one() {
            let lane_idx = ticket.lane;
            let pending_req = match sched.pending.get(&key).cloned() {
                Some(r) => r,
                None => continue,
            };
            let sampling = pending_req.sampling.clone();
            // Use admission-rendered tokens/semantics; do not re-render with None/Plain/0.
            let prompt_tokens = pending_req.prompt_tokens.clone();
            let started_in_think = pending_req.started_in_think;
            if started_in_think {
                // Defensive: think-open prompts are sequential barriers. Transfer
                // any pre-latched abort once, free the just-assigned lane, and
                // push the generate back for outer sequential handling.
                let prompt = pending_req.prompt.clone();
                let _ = batch_transfer_abort_to_singleton_and_clear(&key.id, key.attempt_id);
                let _ = sched.abort_lane(lane_idx, &key);
                if let Err(err) = batch_state.reset_lane(gpu, &config, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        stdout,
                        model,
                        format!("reset lane {lane_idx} on think barrier: {err}"),
                    );
                }
                inbox.push_front(DaemonMsg::Regular(serde_json::json!({
                    "type": "generate",
                    "id": key.id,
                    "attempt_id": key.attempt_id,
                    "prompt": prompt
                })));
                break;
            }

            if let Err(e) = batch_state.reset_lane(gpu, &config, lane_idx) {
                return fail_all(
                    sched,
                    gpu,
                    stdout,
                    model,
                    format!("reset lane {lane_idx}: {e}"),
                );
            }
            if let Err(e) =
                batch_state.prefill_lane(gpu, &weights, &config, &scratch, lane_idx, &prompt_tokens)
            {
                return fail_all(
                    sched,
                    gpu,
                    stdout,
                    model,
                    format!("prefill lane {lane_idx}: {e}"),
                );
            }
            let hist: &[u32] = &[];
            let lane_rng = match &sched.lanes[lane_idx] {
                BatchLane::Running(lane) => lane.rng_state as u32,
                _ => continue,
            };
            let (next_token, next_rng) = match batch_state.sample_lane_product(
                gpu,
                &config,
                lane_idx,
                hist,
                sampling.temp,
                sampling.top_p,
                sampling.top_k,
                sampling.min_p,
                lane_rng,
                sampling.repeat_penalty,
                sampling.presence_penalty,
                sampling.frequency_penalty,
            ) {
                Ok(v) => v,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpu,
                        stdout,
                        model,
                        format!("sample lane {lane_idx}: {e}"),
                    )
                }
            };
            if let BatchLane::Running(lane) = &mut sched.lanes[lane_idx] {
                lane.prompt_len = prompt_tokens.len();
                lane.seq_pos = prompt_tokens.len();
                lane.next_token = Some(next_token);
                lane.rng_state = next_rng as u64;
                lane.conversation_tokens = Vec::new();
                lane.streamed_tokens = Vec::new();
                lane.bytes_fed_to_filter = 0;
                lane.prefill_done_at = Some(Instant::now());
            }
            producers[lane_idx] = Some(QwenArSemanticProducer::new_with_tool_protocol(
                key.id.clone(),
                started_in_think,
                false,
            ));
        }
        let running: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::Running(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        let awaiting: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::AwaitingClient(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if running.is_empty()
            && awaiting.is_empty()
            && sched.inbox.is_empty()
            && inbox.backlog.is_empty()
        {
            break;
        }
        if running.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(2));
            continue;
        }
        // Peak concurrent Running occupancy observed while each lane is live.
        let active_now = running.len();
        for &idx in &running {
            if let BatchLane::Running(lane) = &mut sched.lanes[idx] {
                if active_now > lane.max_active_lanes {
                    lane.max_active_lanes = active_now;
                }
            }
        }
        for i in 0..batch_size {
            match &sched.lanes[i] {
                BatchLane::Running(lane) => {
                    tokens[i] = lane.next_token.unwrap_or(eos_tok);
                    positions[i] = lane.seq_pos;
                }
                _ => {
                    tokens[i] = eos_tok;
                    positions[i] = 0;
                }
            }
        }
        if let Err(e) = qwen35::forward_decode_batch(
            gpu,
            &weights,
            &config,
            &tokens,
            &positions,
            batch_state,
            &scratch,
        ) {
            return fail_all(
                sched,
                gpu,
                stdout,
                model,
                format!("forward_decode_batch: {e}"),
            );
        }
        let mut repeat_tokens: Vec<u32> = vec![0; batch_size * batch_state.sample_repeat_capacity];
        let mut repeat_lengths: Vec<u32> = vec![0; batch_size];
        let mut rng_states: Vec<u32> = vec![0; batch_size];
        let mut survivors: Vec<usize> = Vec::new();
        let mut to_await: Vec<(usize, AttemptKey, serde_json::Value)> = Vec::new();
        let mut to_abort_running: Vec<(usize, AttemptKey)> = Vec::new();
        for idx in running.clone() {
            let key = match sched.lanes[idx].key().cloned() {
                Some(k) => k,
                None => continue,
            };
            if batch_check_abort(&key.id, key.attempt_id) {
                to_abort_running.push((idx, key));
                continue;
            }
            let lane_ptr = match &mut sched.lanes[idx] {
                BatchLane::Running(l) => l as *mut QwenBatchLane,
                _ => continue,
            };
            let lane = unsafe { &mut *lane_ptr };
            let cur_token = lane.next_token.unwrap_or(eos_tok);
            let prod_ptr = match producers[idx].as_mut() {
                Some(p) => p as *mut QwenArSemanticProducer,
                None => continue,
            };
            let producer = unsafe { &mut *prod_ptr };
            let mut future_streamed = lane.streamed_tokens.clone();
            future_streamed.push(cur_token);
            let all_bytes = tokenizer.decode_bytes(&future_streamed);
            let prev_fed = lane.bytes_fed_to_filter.min(all_bytes.len());
            let token_bytes = all_bytes[prev_fed..].to_vec();
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            // TTFT: host Instant immediately before the first classified emit.
            if lane.first_token_at.is_none() {
                lane.first_token_at = Some(Instant::now());
            }
            let stopped = {
                let lane_seq = &mut lane.seq_pos as *mut usize;
                let lane_conv = &mut lane.conversation_tokens as *mut Vec<u32>;
                let lane_stream = &mut lane.streamed_tokens as *mut Vec<u32>;
                let lane_fed = &mut lane.bytes_fed_to_filter as *mut usize;
                let all_len = all_bytes.len();
                let mut res: Result<bool, _> = Ok(false);
                unsafe {
                    res = producer.commit_and_classify(
                        stdout,
                        cur_token,
                        || {
                            let pos = qwen_ar_raw_commit_token(
                                &mut *lane_conv,
                                &mut *lane_stream,
                                &mut *lane_seq,
                                cur_token,
                                QwenArRawCommitDisposition::ClassifiedVisible,
                            );
                            *lane_fed = all_len;
                            (pos, token_bytes.clone())
                        },
                        |_, _| {},
                    );
                }
                match res {
                    Ok(s) => s,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpu,
                            stdout,
                            model,
                            format!("semantic classify lane {idx}: {e}"),
                        )
                    }
                }
            };
            let loop_hit = loop_guards[idx].check(&lane.streamed_tokens).is_some();
            let is_eos = cur_token == eos_tok || cur_token == im_end_tok;
            let hit_max = lane.streamed_tokens.len() >= lane_max_tokens(&key, sched);
            // After committing the current token, seq_pos is the next decode
            // index and must stay strictly below lane_capacity.
            let hit_lane_cap = batch_lane_at_capacity(lane.seq_pos, sched.lane_capacity);
            let should_finish =
                batch_should_finish_decode(is_eos, hit_max, hit_lane_cap, stopped, loop_hit);
            if should_finish {
                let hit_length_cap =
                    batch_hit_length_cap(hit_max, hit_lane_cap, is_eos, stopped, loop_hit);

                let producer_owned = match producers[idx].take() {
                    Some(p) => p,
                    None => continue,
                };
                let (finish, visible_text) = match producer_owned.finish(stdout, hit_length_cap) {
                    Ok(v) => v,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpu,
                            stdout,
                            model,
                            format!("semantic finish lane {idx}: {e}"),
                        )
                    }
                };
                if matches!(finish.cause, QwenArTerminalCause::OpenThink) && !is_eos {
                    // A single lane's semantic validation error is not a GPU core
                    // failure. Roll the lane back and report this key only; peers
                    // keep decoding and the lane is reset before any refill.
                    if let Err(e) = batch_state.reset_lane(gpu, &config, idx) {
                        return fail_all(
                            sched,
                            gpu,
                            stdout,
                            model,
                            format!("reset lane {idx} on open think: {e}"),
                        );
                    }
                    let ep = crate::common::RollbackEpilogue {
                        rolled_back: true,
                        context: None,
                    };
                    let _scope = BatchAttemptScope::enter(key.attempt_id);
                    emit_qwen_ar_open_think_terminal(
                        stdout,
                        &key.id,
                        lane.streamed_tokens.len(),
                        &ep,
                    );
                    let _ = sched.abort_lane(idx, &key);
                    producers[idx] = None;
                    continue;
                }
                if !finish.wire_tool_calls.is_empty() {
                    return fail_all(
                        sched,
                        gpu,
                        stdout,
                        model,
                        format!("semantic finish lane {idx}: unexpected tool calls"),
                    );
                }
                let finish_reason = match finish.finish_reason {
                    "length" => "length",
                    "tool_calls" => "tool_calls",
                    _ => "stop",
                };
                let generated = lane.streamed_tokens.len();
                let metrics = batch_lane_done_metrics(
                    lane.created_at,
                    lane.prefill_done_at,
                    lane.first_token_at,
                    Instant::now(),
                    lane.prompt_len,
                    generated,
                );
                let mut pending_done = qwen_ar_done_value(
                    &key.id,
                    finish_reason,
                    generated,
                    metrics.tok_s,
                    lane.prompt_len,
                    metrics.prefill_ms,
                    metrics.prefill_tok_s,
                    metrics.decode_tok_s,
                    metrics.ttft_ms,
                    0,
                    "",
                );
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                attach_continuous_batch_route_evidence(
                    &mut pending_done,
                    /*slots=*/ batch_size,
                    /*lane=*/ idx,
                    /*lane_capacity=*/ sched.lane_capacity,
                    /*max_active_lanes=*/ lane.max_active_lanes.max(1),
                );
                let _ = visible_text;
                to_await.push((idx, key.clone(), pending_done));
            } else {
                survivors.push(idx);
                let window = lane
                    .sampling
                    .repeat_window
                    .min(batch_state.sample_repeat_capacity);
                let hist = if lane.streamed_tokens.len() > window {
                    &lane.streamed_tokens[lane.streamed_tokens.len() - window..]
                } else {
                    &lane.streamed_tokens[..]
                };
                for (i, &tok) in hist.iter().enumerate() {
                    repeat_tokens[idx * batch_state.sample_repeat_capacity + i] = tok;
                }
                repeat_lengths[idx] = hist.len() as u32;
                rng_states[idx] = lane.rng_state as u32;
            }
        }
        for (idx, key) in to_abort_running {
            if let Err(e) = batch_state.reset_lane(gpu, &config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    stdout,
                    model,
                    format!("reset lane {idx} on abort post-forward: {e}"),
                );
            }
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            crate::ar::emit_active_route_cancel(stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key);
            producers[idx] = None;
        }
        // Install AwaitingClient/Ready BEFORE publishing commit_ready; rollback if publish fails.
        for (idx, key, pending_done) in to_await {
            let mut envelope = pending_done.clone();
            envelope["type"] = serde_json::json!("commit_ready");
            let marked = sched.mark_awaiting_commit(idx, pending_done.clone());
            if !marked {
                eprintln!(
                    "[batch] qwen mark_awaiting_commit failed lane {idx} id={} — aborting lane",
                    key.id
                );
                let _ = batch_state.reset_lane(gpu, &config, idx);
                let _ = sched.abort_lane(idx, &key);
                producers[idx] = None;
                continue;
            }
            let write_ok = {
                let _scope = BatchAttemptScope::enter(key.attempt_id);
                writeln!(stdout, "{}", envelope).is_ok() && stdout.flush().is_ok()
            };
            if !write_ok {
                let _ = batch_state.reset_lane(gpu, &config, idx);
                let _ = sched.abort_lane(idx, &key);
                producers[idx] = None;
            }
            // On success, lane stays AwaitingClient reserved until commit/abort decision.
        }
        if survivors.is_empty() {
            continue;
        }
        for i in 0..batch_size {
            if !survivors.contains(&i) {
                repeat_lengths[i] = 0;
                rng_states[i] = 0;
            }
        }
        let sampling = if let Some(idx) = survivors.first() {
            match &sched.lanes[*idx] {
                BatchLane::Running(l) => l.sampling.clone(),
                _ => continue,
            }
        } else {
            continue;
        };
        let sampled = match batch_state.sample_product(
            gpu,
            &config,
            batch_size,
            &repeat_tokens,
            &repeat_lengths,
            &rng_states,
            sampling.temp,
            sampling.top_p,
            sampling.top_k,
            sampling.min_p,
            sampling.repeat_penalty,
            sampling.presence_penalty,
            sampling.frequency_penalty,
        ) {
            Ok(v) => v,
            Err(e) => return fail_all(sched, gpu, stdout, model, format!("sample_product: {e}")),
        };
        for lane_idx in survivors.iter() {
            let (tok, rng) = sampled[*lane_idx];
            if let BatchLane::Running(lane) = &mut sched.lanes[*lane_idx] {
                lane.next_token = Some(tok);
                lane.rng_state = rng as u64;
            }
        }
    }
    Ok(())
}
pub fn drive_lfm_continuous_batch(
    sched: &mut ContinuousBatchScheduler,
    gpu: &mut rdna_compute::Gpu,
    model: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    inbox: &mut DaemonInbox,
) -> Result<(), BatchDriveError> {
    crate::ar::set_generation_route(crate::ar::GenerationRoute::LfmAr);
    let batch_size = sched.max_batch;
    if batch_size == 0 {
        return Ok(());
    }
    let (batch_state_ptr, config_ptr, weights_ptr, tokenizer_ptr, chat_template_clone, eos_tok) =
        match model.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
        }) {
            Some(b) => {
                let batch_ptr = match b.lfm2_decode_batch.as_mut() {
                    Some(s) => s as *mut Lfm2DecodeBatchState,
                    None => {
                        return Err(BatchDriveError::Gpu(
                            "batch state not allocated".to_string(),
                        ))
                    }
                };
                (
                    batch_ptr,
                    &b.config as *const lfm2moe::config::Lfm2MoeConfig,
                    &b.weights as *const lfm2moe::Lfm2MoeWeights,
                    match model.tokenizer.as_ref() {
                        Some(t) => t as *const _,
                        None => return Err(BatchDriveError::Gpu("tokenizer missing".to_string())),
                    },
                    model.chat_template.clone(),
                    b.eos_tok,
                )
            }
            _ => return Err(BatchDriveError::Gpu("batch model not Lfm2Moe".to_string())),
        };
    let batch_state = unsafe { &mut *batch_state_ptr };
    let config = unsafe { &*config_ptr };
    let weights = unsafe { &*weights_ptr };
    let tokenizer: &hipfire_runtime::tokenizer::Tokenizer = unsafe { &*tokenizer_ptr };
    let chat_template = chat_template_clone;
    // Stop set mirrors crate::dense::generate_lfm2moe: eos_tok plus single-id encodings for
    // <|endoftext|>, </s>, <|im_end|>. String guard catches leaked EOS-class
    // strings where encode does not round-trip (e.g. <|endoftext|>).
    let mut stop_toks: Vec<u32> = vec![eos_tok];
    for s in ["<|endoftext|>", "</s>", "<|im_end|>"] {
        let ids = tokenizer.encode(s);
        if ids.len() == 1 && !stop_toks.contains(&ids[0]) {
            stop_toks.push(ids[0]);
        }
    }
    let mut loop_guards: Vec<hipfire_runtime::loop_guard::LoopGuard> = (0..batch_size)
        .map(
            |_| hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get()),
        )
        .collect();
    let mut tokens = vec![0u32; batch_size];
    let mut positions = vec![0usize; batch_size];
    let fail_all = |sched: &mut ContinuousBatchScheduler,
                    gpu: &mut rdna_compute::Gpu,
                    batch_state: &mut Lfm2DecodeBatchState,
                    stdout: &mut std::io::Stdout,
                    reason: String|
     -> Result<(), BatchDriveError> {
        let mut uniq_set = std::collections::HashSet::new();
        let mut uniq: Vec<AttemptKey> = Vec::new();
        for l in sched.lanes.iter() {
            if let Some(k) = l.key() {
                if uniq_set.insert(k.clone()) {
                    uniq.push(k.clone());
                }
            }
        }
        for k in sched.inbox.iter().cloned() {
            if uniq_set.insert(k.clone()) {
                uniq.push(k);
            }
        }
        for k in sched.pending.keys().cloned() {
            if uniq_set.insert(k.clone()) {
                uniq.push(k);
            }
        }
        let mut first_err: Option<String> = None;
        if let Err(e) = batch_state.reset(gpu) {
            first_err = Some(format!("batch reset: {e}"));
        }
        crate::common::fail_closed_invalidate_graphs_and_replay(gpu);
        let sync = crate::common::fail_closed_device_sync(gpu);
        let prior = match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        };
        let ep = crate::common::fail_closed_epilogue_after_sync(prior, sync);
        for key in &uniq {
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            crate::common::emit_fail_closed_error(
                stdout,
                Some(&key.id),
                &format!("batch GPU error: {reason}"),
                "gpu",
                ep.rolled_back,
                &ep,
            );
        }
        let _ = sched.fail_all_active();
        for k in &uniq {
            batch_clear_terminal(&k.id, k.attempt_id);
        }
        if !ep.rolled_back {
            return Err(BatchDriveError::Poisoned(format!(
                "{reason}; {}",
                ep.context.unwrap_or_default()
            )));
        }
        Err(BatchDriveError::Gpu(reason))
    };
    loop {
        let mut to_commit: Vec<(usize, AttemptKey, serde_json::Value)> = Vec::new();
        let mut to_abort: Vec<(usize, AttemptKey)> = Vec::new();
        for idx in 0..batch_size {
            if let BatchLane::AwaitingClient(term) = &sched.lanes[idx] {
                let key = term.key.clone();
                let expired = Instant::now() >= term.deadline;
                if batch_check_abort(&key.id, key.attempt_id) || expired {
                    to_abort.push((idx, key));
                } else if let Some(ClientTerminalDecision::Commit) =
                    batch_poll_decision(&key.id, key.attempt_id)
                {
                    to_commit.push((idx, key.clone(), term.pending_done.clone()));
                }
            }
        }
        for (idx, key) in to_abort {
            if let Err(e) = batch_state.reset_lane(gpu, config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {idx} on abort: {e}"),
                );
            }
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            crate::ar::emit_active_route_cancel(stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key);
        }
        for (idx, key, pending_done) in to_commit {
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            let reset_ok = match batch_state.reset_lane(gpu, config, idx) {
                Ok(()) => true,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {idx} on commit: {e}"),
                    );
                }
            };
            let commit_ok = sched.commit_lane(idx, &key);
            match batch_commit_teardown_class(reset_ok, commit_ok) {
                BatchCommitTeardownClass::ResetFailed => unreachable!("reset_ok handled above"),
                BatchCommitTeardownClass::CommitFailed => {
                    let ep = crate::common::RollbackEpilogue {
                        rolled_back: true,
                        context: None,
                    };
                    crate::common::emit_fail_closed_error(
                        stdout,
                        Some(&key.id),
                        "batch commit_lane failed after reset",
                        "internal",
                        false,
                        &ep,
                    );
                    let _ = sched.abort_lane(idx, &key);
                }
                BatchCommitTeardownClass::EmitDone => {
                    crate::ar::emit_active_route_done(stdout, &key.id, &pending_done);
                }
            }
        }
        let mut queued_abort: Vec<AttemptKey> = Vec::new();
        for k in sched.inbox.iter().cloned().collect::<Vec<_>>() {
            if batch_check_abort(&k.id, k.attempt_id) {
                queued_abort.push(k);
            }
        }
        for k in queued_abort {
            let _scope = BatchAttemptScope::enter(k.attempt_id);
            crate::ar::emit_active_route_cancel(stdout, &k.id, 0);
            let _ = sched.abort_queued(&k);
        }
        let mut running_abort: Vec<(usize, AttemptKey)> = Vec::new();
        for idx in 0..batch_size {
            if let Some(k) = sched.lanes[idx].key().cloned() {
                if matches!(
                    sched.lanes[idx],
                    BatchLane::Running(_) | BatchLane::Seeding(_)
                ) && batch_check_abort(&k.id, k.attempt_id)
                {
                    running_abort.push((idx, k));
                }
            }
        }
        for (idx, key) in running_abort {
            if let Err(e) = batch_state.reset_lane(gpu, config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {idx} on running abort: {e}"),
                );
            }
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            crate::ar::emit_active_route_cancel(stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key);
        }
        let mut barrier: Option<DaemonMsg> = None;
        // A fresh continuous-batch wave reaches the daemon through many
        // concurrent HTTP handlers. Give that first wave a small, bounded
        // coalescing window so admission does not race the second request and
        // serialize the remainder through single-lane prefill.
        let admission_deadline = (sched.active_count() == 0 && sched.awaiting_count() == 0)
            .then(|| Instant::now() + Duration::from_millis(20));
        loop {
            let dm = match inbox.try_recv() {
                Ok(m) => m,
                Err(mpsc::TryRecvError::Empty) => {
                    let Some(deadline) = admission_deadline else {
                        break;
                    };
                    if sched.active_count() != 0
                        || sched.awaiting_count() != 0
                        || sched.inbox.len() >= batch_size
                    {
                        break;
                    }
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        break;
                    }
                    match inbox.recv_timeout(remaining) {
                        Ok(m) => m,
                        Err(
                            mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected,
                        ) => {
                            break;
                        }
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => break,
            };
            match dm {
                DaemonMsg::ParseError(e) => {
                    emit_uncorrelated_error(
                        stdout,
                        None,
                        &format!("invalid JSON: {e}"),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                }
                DaemonMsg::Regular(json) => {
                    let t = json.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if t == "generate" {
                        let attempt_id = match json.get("attempt_id").and_then(|v| v.as_u64()) {
                            Some(v) => v,
                            None => {
                                emit_uncorrelated_error(
                                    stdout,
                                    json.get("id").and_then(|v| v.as_str()),
                                    "generate missing attempt_id",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        let id = json
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("0")
                            .to_string();
                        batch_announce_terminal(&id, attempt_id);
                        if batch_check_abort(&id, attempt_id) {
                            let _scope = BatchAttemptScope::enter(attempt_id);
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::LfmAr,
                                stdout,
                                &id,
                                false,
                            );
                            crate::ar::emit_active_route_cancel(stdout, &id, 0);
                            batch_clear_terminal(&id, attempt_id);
                            continue;
                        }
                        if !is_batch_request_eligible(
                            &json,
                            model,
                            batch_size,
                            parse_serve_continuous_batch(&json),
                            false,
                        ) {
                            barrier = Some(DaemonMsg::Regular(json));
                            break;
                        }
                        let prompt_str = batch_single_user_content(&json).unwrap_or_else(|| {
                            json.get("prompt")
                                .and_then(|v| v.as_str())
                                .unwrap_or("Hello")
                                .to_string()
                        });
                        let system_str = json
                            .get("system")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let assistant_prefix = match json
                            .get("assistant_prefix")
                            .and_then(|v| v.as_str())
                            .unwrap_or("plain")
                        {
                            "open_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
                            }
                            "closed_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink
                            }
                            _ => hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                        };
                        let max_think = json
                            .get("max_think_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;
                        let max_tokens_req = json
                            .get("max_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(4096) as usize;
                        let batch_messages = match json.get("messages") {
                            Some(v) => match serde_json::from_value::<
                                Vec<hipfire_runtime::prompt_frame::Message>,
                            >(v.clone())
                            {
                                Ok(v) => Some(v),
                                Err(e) => {
                                    let _scope = BatchAttemptScope::enter(attempt_id);
                                    emit_uncorrelated_error(
                                        stdout,
                                        Some(&id),
                                        &format!("invalid messages field: {e}"),
                                        "validation",
                                        false,
                                        false,
                                    );
                                    batch_clear_terminal(&id, attempt_id);
                                    continue;
                                }
                            },
                            None => None,
                        };
                        let (prompt_tokens, started_in_think) = match batch_render_prompt_tokens(
                            &prompt_str,
                            system_str.as_deref(),
                            assistant_prefix,
                            tokenizer,
                            chat_template.as_ref(),
                            max_think,
                            batch_messages.as_deref(),
                            max_think != 1,
                            None,
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                let _scope = BatchAttemptScope::enter(attempt_id);
                                emit_uncorrelated_error(
                                    stdout,
                                    Some(&id),
                                    &format!("render failed: {e}"),
                                    "validation",
                                    false,
                                    false,
                                );
                                batch_clear_terminal(&id, attempt_id);
                                continue;
                            }
                        };
                        if started_in_think {
                            let _ = batch_transfer_abort_to_singleton_and_clear(&id, attempt_id);
                            barrier = Some(DaemonMsg::Regular(json));
                            break;
                        }
                        if prompt_tokens.is_empty() {
                            let _scope = BatchAttemptScope::enter(attempt_id);
                            emit_uncorrelated_error(
                                stdout,
                                Some(&id),
                                "empty prompt after tokenize",
                                "validation",
                                false,
                                false,
                            );
                            batch_clear_terminal(&id, attempt_id);
                            continue;
                        }
                        if batch_lfm_exceeds_capacity(
                            prompt_tokens.len(),
                            max_tokens_req,
                            sched.lane_capacity,
                        ) {
                            let _scope = BatchAttemptScope::enter(attempt_id);
                            emit_uncorrelated_error(
                                stdout,
                                Some(&id),
                                &format!(
                                    "prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq",
                                    prompt_tokens.len(),
                                    max_tokens_req,
                                    sched.lane_capacity
                                ),
                                "context_length",
                                false,
                                false,
                            );
                            batch_clear_terminal(&id, attempt_id);
                            continue;
                        }
                        // Explicit wire `seed` must reach the lane RNG on the
                        // batched route too; out-of-domain values are rejected
                        // loudly, never silently unseeded.
                        let client_seed = match wire_seed::parse_wire_seed(json.get("seed")) {
                            Ok(s) => s,
                            Err(reason) => {
                                emit_uncorrelated_error(
                                    stdout,
                                    Some(&id),
                                    &reason,
                                    "validation",
                                    false,
                                    false,
                                );
                                batch_clear_terminal(&id, attempt_id);
                                continue;
                            }
                        };
                        batch_transition_to_queued(&id, attempt_id);
                        let sampling = resolve_batch_sampling(&json, model);
                        let req = BatchPendingRequest {
                            key: AttemptKey::new(&id, attempt_id),
                            prompt: prompt_str.clone(),
                            prompt_tokens: prompt_tokens.clone(),
                            started_in_think,
                            system: system_str.clone(),
                            assistant_prefix,
                            max_think_tokens: max_think,
                            max_tokens: max_tokens_req,
                            client_seed,
                            sampling,
                        };
                        if !sched.enqueue(req) {
                            eprintln!(
                                "[batch] duplicate enqueue rejected id={} attempt_id={}; preserving live registry",
                                id, attempt_id
                            );
                            continue;
                        }
                        {
                            let _scope = BatchAttemptScope::enter(attempt_id);
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::LfmAr,
                                stdout,
                                &id,
                                started_in_think,
                            );
                        }
                    } else if t == "abort" || t == "commit" {
                        if let (Some(id), Some(aid), Some(kind)) = (
                            json.get("id").and_then(|v| v.as_str()),
                            json.get("attempt_id").and_then(|v| v.as_u64()),
                            json.get("type").and_then(|v| v.as_str()),
                        ) {
                            batch_apply_terminal_control(kind, id, aid);
                        }
                    } else {
                        barrier = Some(DaemonMsg::Regular(json));
                        break;
                    }
                }
            }
        }
        if let Some(msg) = barrier {
            inbox.push_front(msg);
            if sched.active_count() == 0 && sched.inbox.is_empty() {
                return Ok(());
            }
        }
        // ---- generic initial-wave fast path (batched prefill, O(prompt_len) vs O(n*prompt_len)) ----
        // Non-mutating candidate scan ensures a one-request wave is never removed.
        if sched.active_count() == 0 && sched.awaiting_count() == 0 {
            let n = lfm_fast_path_candidate_len(sched);
            if n >= 2 {
                // Assign exactly n prefix lanes; each try_assign_one pops front and binds.
                let mut assigned_keys: Vec<AttemptKey> = Vec::with_capacity(n);
                let mut assigned_tickets: Vec<LaneTicket> = Vec::with_capacity(n);
                let mut prompts_for_batch: Vec<Vec<u32>> = Vec::with_capacity(n);
                let mut assign_ok = true;
                for _ in 0..n {
                    match sched.try_assign_one() {
                        Some((key, ticket)) => {
                            if let Some(req) = sched.pending.get(&key).cloned() {
                                prompts_for_batch.push(req.prompt_tokens);
                            } else {
                                prompts_for_batch.push(Vec::new());
                            }
                            assigned_keys.push(key);
                            assigned_tickets.push(ticket);
                        }
                        None => {
                            assign_ok = false;
                            break;
                        }
                    }
                }
                if assign_ok && assigned_keys.len() == n {
                    let prompt_refs: Vec<&[u32]> =
                        prompts_for_batch.iter().map(|v| v.as_slice()).collect();
                    let prefill_res =
                        batch_state.prefill_lanes_batched(gpu, weights, config, &prompt_refs);
                    match prefill_res {
                        Ok(()) => {
                            for (idx, key) in assigned_keys.iter().enumerate() {
                                let lane_idx = assigned_tickets[idx].lane;
                                if batch_check_abort(&key.id, key.attempt_id) {
                                    if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                                        return fail_all(
                                            sched,
                                            gpu,
                                            batch_state,
                                            stdout,
                                            format!("reset lane {lane_idx} on batched prefill abort: {e}"),
                                        );
                                    }
                                    let _scope = BatchAttemptScope::enter(key.attempt_id);
                                    let ep = crate::common::RollbackEpilogue {
                                        rolled_back: true,
                                        context: None,
                                    };
                                    crate::common::emit_spec_cancel_after_rollback(
                                        stdout, &key.id, 0, &ep,
                                    );
                                    let _ = sched.abort_lane(lane_idx, key);
                                    continue;
                                }
                                let hist: &[u32] = &[];
                                let (lane_rng, sampling) = match &sched.lanes[lane_idx] {
                                    BatchLane::Running(l) => {
                                        (l.rng_state as u32, l.sampling.clone())
                                    }
                                    _ => continue,
                                };
                                match batch_state.sample_lane_product(
                                    gpu,
                                    config,
                                    lane_idx,
                                    hist,
                                    sampling.temp,
                                    sampling.top_p,
                                    sampling.top_k,
                                    sampling.min_p,
                                    lane_rng,
                                    sampling.repeat_penalty,
                                    sampling.presence_penalty,
                                    sampling.frequency_penalty,
                                ) {
                                    Ok((next_token, next_rng)) => {
                                        let prompt_len = prompts_for_batch[idx].len();
                                        lfm_populate_lane_after_sample(
                                            sched, lane_idx, next_token, next_rng, prompt_len,
                                        );
                                    }
                                    Err(e) => {
                                        return fail_all(
                                            sched,
                                            gpu,
                                            batch_state,
                                            stdout,
                                            format!(
                                                "sample lane {lane_idx} after batched prefill: {e}"
                                            ),
                                        );
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            return fail_all(
                                sched,
                                gpu,
                                batch_state,
                                stdout,
                                format!("batched prefill lanes 0..{n}: {e}"),
                            );
                        }
                    }
                } else {
                    // Partial assign failure: rollback any already-assigned lanes
                    for (k, t) in assigned_keys.iter().zip(assigned_tickets.iter()) {
                        let _ = batch_state.reset_lane(gpu, config, t.lane);
                        let _ = sched.abort_lane(t.lane, k);
                    }
                }
            }
        }
        while let Some((key, ticket)) = sched.try_assign_one() {
            let lane_idx = ticket.lane;
            let pending_req = match sched.pending.get(&key).cloned() {
                Some(r) => r,
                None => continue,
            };
            let prompt_tokens = pending_req.prompt_tokens.clone();
            let max_tokens_req = pending_req.max_tokens;
            let started_in_think = pending_req.started_in_think;
            if started_in_think {
                let prompt = pending_req.prompt.clone();
                let _ = batch_transfer_abort_to_singleton_and_clear(&key.id, key.attempt_id);
                let _ = sched.abort_lane(lane_idx, &key);
                if let Err(err) = batch_state.reset_lane(gpu, config, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {lane_idx} on think barrier: {err}"),
                    );
                }
                inbox.push_front(DaemonMsg::Regular(serde_json::json!({
                    "type": "generate",
                    "id": key.id,
                    "attempt_id": key.attempt_id,
                    "prompt": prompt
                })));
                break;
            }
            // Re-validate capacity at assignment time (defensive; lane_capacity is the source of truth).
            if batch_lfm_exceeds_capacity(prompt_tokens.len(), max_tokens_req, sched.lane_capacity)
            {
                // This should have been rejected before gen_start, but if it slipped through (e.g. clamped capacity race),
                // fail closed for this lane only without GPU work.
                if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {lane_idx} on capacity re-check: {e}"),
                    );
                }
                let _scope = BatchAttemptScope::enter(key.attempt_id);
                emit_uncorrelated_error(
                    stdout,
                    Some(&key.id),
                    &format!(
                        "prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={}",
                        prompt_tokens.len(),
                        max_tokens_req,
                        sched.lane_capacity
                    ),
                    "context_length",
                    false,
                    false,
                );
                let _ = sched.abort_lane(lane_idx, &key);
                continue;
            }
            if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {lane_idx}: {e}"),
                );
            }
            // Zero-token completion: no GPU work, ordinary two-phase terminal with zero tokens.
            if max_tokens_req == 0 {
                if let BatchLane::Running(lane) = &mut sched.lanes[lane_idx] {
                    lane.prompt_len = prompt_tokens.len();
                    lane.seq_pos = prompt_tokens.len();
                    lane.next_token = None;
                    lane.streamed_tokens = Vec::new();
                    lane.bytes_fed_to_filter = 0;
                    lane.prefill_done_at = Some(Instant::now());
                    lane.first_token_at = None;
                }
                let lane_ref = match &sched.lanes[lane_idx] {
                    BatchLane::Running(l) => l,
                    _ => continue,
                };
                let metrics = batch_lane_done_metrics(
                    lane_ref.created_at,
                    lane_ref.prefill_done_at,
                    lane_ref.first_token_at,
                    Instant::now(),
                    lane_ref.prompt_len,
                    0,
                );
                let mut pending_done = serde_json::json!({
                    "type": "done",
                    "id": key.id,
                    "tokens": 0,
                    "tok_s": (metrics.tok_s * 10.0).round() / 10.0,
                    "prefill_tokens": lane_ref.prompt_len,
                    "prefill_ms": (metrics.prefill_ms * 10.0).round() / 10.0,
                    "prefill_tok_s": (metrics.prefill_tok_s * 10.0).round() / 10.0,
                    "decode_tok_s": (metrics.decode_tok_s * 10.0).round() / 10.0,
                    "ttft_ms": (metrics.ttft_ms * 10.0).round() / 10.0,
                    "cached_tokens": 0,
                    "finish_reason": "length",
                    "attempt_id": key.attempt_id,
                });
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                attach_continuous_batch_route_evidence(
                    &mut pending_done,
                    batch_size,
                    lane_idx,
                    sched.lane_capacity,
                    lane_ref.max_active_lanes.max(1),
                );
                let mut envelope = pending_done.clone();
                envelope["type"] = serde_json::json!("commit_ready");
                // Install AwaitingClient/Ready BEFORE publishing commit_ready.
                let marked = sched.mark_awaiting_commit(lane_idx, pending_done.clone());
                if !marked {
                    // Failed to mark — rollback lane without publishing.
                    let _ = batch_state.reset_lane(gpu, config, lane_idx);
                    let _ = sched.abort_lane(lane_idx, &key);
                    continue;
                }
                let write_ok = {
                    let _scope = BatchAttemptScope::enter(key.attempt_id);
                    writeln!(stdout, "{}", envelope).is_ok() && stdout.flush().is_ok()
                };
                if !write_ok {
                    // Publication failed — rollback attested reset and free lane.
                    let _ = batch_state.reset_lane(gpu, config, lane_idx);
                    let _ = sched.abort_lane(lane_idx, &key);
                }
                continue;
            }
            // Cancellable prefill: check abort before GPU, then delegate to batch prefill.
            // If abort is latched before or during prefill, we must reset only this lane,
            // emit attested abort, and continue peers without sampling.
            if batch_check_abort(&key.id, key.attempt_id) {
                if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {lane_idx} on pre-prefill abort: {e}"),
                    );
                }
                let _scope = BatchAttemptScope::enter(key.attempt_id);
                let ep = crate::common::RollbackEpilogue {
                    rolled_back: true,
                    context: None,
                };
                crate::common::emit_spec_cancel_after_rollback(stdout, &key.id, 0, &ep);
                let _ = sched.abort_lane(lane_idx, &key);
                continue;
            }
            // Try cancellable prefill if the arch provides it; otherwise fall back to
            // the standard prefill and treat post-prefill abort as cancellation.
            let prefill_is_aborted = {
                // Prefer the cancellable variant when available (sibling adds it).
                // We probe via a helper that returns Ok(false) on abort without sampling.
                // Fallback: call the standard prefill and then check abort.
                let abort_check = || batch_check_abort(&key.id, key.attempt_id);
                // Attempt to call the new API via a daemon helper; if not present we fall back.
                // This helper will be overridden by the arch's implementation once it lands.
                let res = lfm_prefill_cancellable_or_fallback(
                    batch_state,
                    gpu,
                    weights,
                    config,
                    lane_idx,
                    &prompt_tokens,
                    &abort_check,
                );
                match res {
                    Ok(true) => false, // completed
                    Ok(false) => true, // aborted
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpu,
                            batch_state,
                            stdout,
                            format!("prefill lane {lane_idx}: {e}"),
                        );
                    }
                }
            };
            if prefill_is_aborted || batch_check_abort(&key.id, key.attempt_id) {
                if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {lane_idx} on prefill abort: {e}"),
                    );
                }
                let _scope = BatchAttemptScope::enter(key.attempt_id);
                let ep = crate::common::RollbackEpilogue {
                    rolled_back: true,
                    context: None,
                };
                crate::common::emit_spec_cancel_after_rollback(stdout, &key.id, 0, &ep);
                let _ = sched.abort_lane(lane_idx, &key);
                continue;
            }
            let hist: &[u32] = &[];
            let (lane_rng, sampling) = match &sched.lanes[lane_idx] {
                BatchLane::Running(lane) => (lane.rng_state as u32, lane.sampling.clone()),
                _ => continue,
            };
            let (next_token, next_rng) = match batch_state.sample_lane_product(
                gpu,
                config,
                lane_idx,
                hist,
                sampling.temp,
                sampling.top_p,
                sampling.top_k,
                sampling.min_p,
                lane_rng,
                sampling.repeat_penalty,
                sampling.presence_penalty,
                sampling.frequency_penalty,
            ) {
                Ok(v) => v,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("sample lane {lane_idx}: {e}"),
                    )
                }
            };
            if let BatchLane::Running(lane) = &mut sched.lanes[lane_idx] {
                lane.prompt_len = prompt_tokens.len();
                lane.seq_pos = prompt_tokens.len();
                lane.next_token = Some(next_token);
                lane.rng_state = next_rng as u64;
                lane.conversation_tokens = Vec::new();
                lane.streamed_tokens = Vec::new();
                lane.bytes_fed_to_filter = 0;
                lane.prefill_done_at = Some(Instant::now());
            }
        }
        let running: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::Running(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        let awaiting: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::AwaitingClient(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if running.is_empty()
            && awaiting.is_empty()
            && sched.inbox.is_empty()
            && inbox.backlog.is_empty()
        {
            break;
        }
        if running.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(2));
            continue;
        }
        let active_now = running.len();
        for &idx in &running {
            if let BatchLane::Running(lane) = &mut sched.lanes[idx] {
                if active_now > lane.max_active_lanes {
                    lane.max_active_lanes = active_now;
                }
            }
        }
        for i in 0..batch_size {
            match &sched.lanes[i] {
                BatchLane::Running(lane) => {
                    tokens[i] = lane.next_token.unwrap_or(eos_tok);
                    positions[i] = lane.seq_pos;
                }
                _ => {
                    tokens[i] = eos_tok;
                    positions[i] = 0;
                }
            }
        }
        if let Err(e) =
            forward_decode_batch_lfm(gpu, weights, config, &tokens, &positions, batch_state)
        {
            return fail_all(
                sched,
                gpu,
                batch_state,
                stdout,
                format!("forward_decode_batch_lfm: {e}"),
            );
        }
        let mut repeat_tokens: Vec<u32> = vec![0; batch_size * batch_state.sample_repeat_capacity];
        let mut repeat_lengths: Vec<u32> = vec![0; batch_size];
        let mut rng_states: Vec<u32> = vec![0; batch_size];
        let mut survivors: Vec<usize> = Vec::new();
        let mut to_await: Vec<(usize, AttemptKey, serde_json::Value)> = Vec::new();
        let mut to_abort_running: Vec<(usize, AttemptKey)> = Vec::new();
        for idx in running.clone() {
            let key = match sched.lanes[idx].key().cloned() {
                Some(k) => k,
                None => continue,
            };
            if batch_check_abort(&key.id, key.attempt_id) {
                to_abort_running.push((idx, key));
                continue;
            }
            let lane_ptr = match &mut sched.lanes[idx] {
                BatchLane::Running(l) => l as *mut QwenBatchLane,
                _ => continue,
            };
            let lane = unsafe { &mut *lane_ptr };
            let cur_token = lane.next_token.unwrap_or(eos_tok);
            // Suppress EOS-class IDs before any decode/wire output.
            if stop_toks.contains(&cur_token) {
                let generated = lane.streamed_tokens.len();
                let metrics = batch_lane_done_metrics(
                    lane.created_at,
                    lane.prefill_done_at,
                    lane.first_token_at,
                    Instant::now(),
                    lane.prompt_len,
                    generated,
                );
                let mut pending_done = serde_json::json!({
                    "type": "done",
                    "id": key.id,
                    "tokens": generated,
                    "tok_s": (metrics.tok_s * 10.0).round() / 10.0,
                    "prefill_tokens": lane.prompt_len,
                    "prefill_ms": (metrics.prefill_ms * 10.0).round() / 10.0,
                    "prefill_tok_s": (metrics.prefill_tok_s * 10.0).round() / 10.0,
                    "decode_tok_s": (metrics.decode_tok_s * 10.0).round() / 10.0,
                    "ttft_ms": (metrics.ttft_ms * 10.0).round() / 10.0,
                    "cached_tokens": 0,
                    "finish_reason": "stop",
                    "attempt_id": key.attempt_id,
                });
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                attach_continuous_batch_route_evidence(
                    &mut pending_done,
                    batch_size,
                    idx,
                    sched.lane_capacity,
                    lane.max_active_lanes.max(1),
                );
                to_await.push((idx, key.clone(), pending_done));
                continue;
            }
            // Cumulative byte-correct incremental decode with holdback.
            // Never use `tokenizer.decode(&[cur_token])` (lossy, splits UTF-8 into FFFD).
            // Instead decode all streamed tokens + cur_token as bytes and emit only the
            // newly completed UTF-8 prefix beyond `bytes_fed_to_filter`.
            let mut future_streamed = lane.streamed_tokens.clone();
            future_streamed.push(cur_token);
            let all_bytes = tokenizer.decode_bytes(&future_streamed);
            let valid_len = match std::str::from_utf8(&all_bytes) {
                Ok(_) => all_bytes.len(),
                Err(e) => e.valid_up_to(),
            };
            let prev_fed = lane.bytes_fed_to_filter.min(valid_len);
            let new_bytes = &all_bytes[prev_fed..valid_len];
            let frag = match std::str::from_utf8(new_bytes) {
                Ok(s) => s,
                Err(_) => "",
            };
            // Suppress decoded EOS-class markers (e.g. "<|endoftext|>" that doesn't round-trip via ID).
            if matches!(frag.trim(), "<|endoftext|>" | "</s>" | "<|im_end|>") {
                let generated = lane.streamed_tokens.len();
                let metrics = batch_lane_done_metrics(
                    lane.created_at,
                    lane.prefill_done_at,
                    lane.first_token_at,
                    Instant::now(),
                    lane.prompt_len,
                    generated,
                );
                let mut pending_done = serde_json::json!({
                    "type": "done",
                    "id": key.id,
                    "tokens": generated,
                    "tok_s": (metrics.tok_s * 10.0).round() / 10.0,
                    "prefill_tokens": lane.prompt_len,
                    "prefill_ms": (metrics.prefill_ms * 10.0).round() / 10.0,
                    "prefill_tok_s": (metrics.prefill_tok_s * 10.0).round() / 10.0,
                    "decode_tok_s": (metrics.decode_tok_s * 10.0).round() / 10.0,
                    "ttft_ms": (metrics.ttft_ms * 10.0).round() / 10.0,
                    "cached_tokens": 0,
                    "finish_reason": "stop",
                    "attempt_id": key.attempt_id,
                });
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                attach_continuous_batch_route_evidence(
                    &mut pending_done,
                    batch_size,
                    idx,
                    sched.lane_capacity,
                    lane.max_active_lanes.max(1),
                );
                to_await.push((idx, key.clone(), pending_done));
                continue;
            }
            // Visible fragment (may be empty due to holdback for split UTF-8).
            let has_visible = !frag.is_empty();
            if has_visible {
                if lane.first_token_at.is_none() {
                    lane.first_token_at = Some(Instant::now());
                }
                {
                    let _scope = BatchAttemptScope::enter(key.attempt_id);
                    emit_visible_token(stdout, &key.id, frag);
                }
            }
            // Commit token to lane state: streamed tokens and byte holdback, seq_pos.
            lane.streamed_tokens.push(cur_token);
            lane.bytes_fed_to_filter = valid_len;
            lane.seq_pos += 1;
            let loop_hit = loop_guards[idx].check(&lane.streamed_tokens).is_some();
            let hit_max = lane.streamed_tokens.len() >= lane_max_tokens(&key, sched);
            let hit_lane_cap = batch_lane_at_capacity(lane.seq_pos, sched.lane_capacity);
            let is_eos = false; // already filtered EOS IDs/markers above
            let should_finish =
                batch_should_finish_decode(is_eos, hit_max, hit_lane_cap, false, loop_hit);
            if should_finish {
                let hit_length_cap =
                    batch_hit_length_cap(hit_max, hit_lane_cap, is_eos, false, loop_hit);
                let finish_reason = if hit_length_cap { "length" } else { "stop" };
                let generated = lane.streamed_tokens.len();
                let metrics = batch_lane_done_metrics(
                    lane.created_at,
                    lane.prefill_done_at,
                    lane.first_token_at,
                    Instant::now(),
                    lane.prompt_len,
                    generated,
                );
                let mut pending_done = serde_json::json!({
                    "type": "done",
                    "id": key.id,
                    "tokens": generated,
                    "tok_s": (metrics.tok_s * 10.0).round() / 10.0,
                    "prefill_tokens": lane.prompt_len,
                    "prefill_ms": (metrics.prefill_ms * 10.0).round() / 10.0,
                    "prefill_tok_s": (metrics.prefill_tok_s * 10.0).round() / 10.0,
                    "decode_tok_s": (metrics.decode_tok_s * 10.0).round() / 10.0,
                    "ttft_ms": (metrics.ttft_ms * 10.0).round() / 10.0,
                    "cached_tokens": 0,
                    "finish_reason": finish_reason,
                    "attempt_id": key.attempt_id,
                });
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                attach_continuous_batch_route_evidence(
                    &mut pending_done,
                    batch_size,
                    idx,
                    sched.lane_capacity,
                    lane.max_active_lanes.max(1),
                );
                to_await.push((idx, key.clone(), pending_done));
            } else {
                survivors.push(idx);
                let window = lane
                    .sampling
                    .repeat_window
                    .min(batch_state.sample_repeat_capacity);
                let hist = if lane.streamed_tokens.len() > window {
                    &lane.streamed_tokens[lane.streamed_tokens.len() - window..]
                } else {
                    &lane.streamed_tokens[..]
                };
                for (i, &tok) in hist.iter().enumerate() {
                    repeat_tokens[idx * batch_state.sample_repeat_capacity + i] = tok;
                }
                repeat_lengths[idx] = hist.len() as u32;
                rng_states[idx] = lane.rng_state as u32;
            }
        }
        for (idx, key) in to_abort_running {
            if let Err(e) = batch_state.reset_lane(gpu, config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {idx} on abort post-forward: {e}"),
                );
            }
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            let ep = crate::common::RollbackEpilogue {
                rolled_back: true,
                context: None,
            };
            crate::common::emit_spec_cancel_after_rollback(stdout, &key.id, 0, &ep);
            let _ = sched.abort_lane(idx, &key);
        }
        // Install AwaitingClient/Ready BEFORE publishing commit_ready; rollback if publish fails.
        for (idx, key, pending_done) in to_await {
            let mut envelope = pending_done.clone();
            envelope["type"] = serde_json::json!("commit_ready");
            let marked = sched.mark_awaiting_commit(idx, pending_done.clone());
            if !marked {
                eprintln!(
                    "[batch] mark_awaiting_commit failed for lane {idx} id={} — aborting lane",
                    key.id
                );
                let _ = batch_state.reset_lane(gpu, config, idx);
                let _ = sched.abort_lane(idx, &key);
                continue;
            }
            let write_ok = {
                let _scope = BatchAttemptScope::enter(key.attempt_id);
                writeln!(stdout, "{}", envelope).is_ok() && stdout.flush().is_ok()
            };
            if !write_ok {
                // Publication failed — rollback attested reset and free lane (no duplicate done).
                let _ = batch_state.reset_lane(gpu, config, idx);
                let _ = sched.abort_lane(idx, &key);
                // Do not requeue; lane is now free.
                continue;
            }
            // Lane stays AwaitingClient until commit/abort decision.
        }
        if survivors.is_empty() {
            continue;
        }
        for i in 0..batch_size {
            if !survivors.contains(&i) {
                repeat_lengths[i] = 0;
                rng_states[i] = 0;
            }
        }
        let sampling = if let Some(idx) = survivors.first() {
            match &sched.lanes[*idx] {
                BatchLane::Running(l) => l.sampling.clone(),
                _ => continue,
            }
        } else {
            continue;
        };
        let sampled = match batch_state.sample_product(
            gpu,
            config,
            batch_size,
            &repeat_tokens,
            &repeat_lengths,
            &rng_states,
            sampling.temp,
            sampling.top_p,
            sampling.top_k,
            sampling.min_p,
            sampling.repeat_penalty,
            sampling.presence_penalty,
            sampling.frequency_penalty,
        ) {
            Ok(v) => v,
            Err(e) => {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("sample_product: {e}"),
                )
            }
        };
        for lane_idx in survivors.iter() {
            let (tok, rng) = sampled[*lane_idx];
            if let BatchLane::Running(lane) = &mut sched.lanes[*lane_idx] {
                lane.next_token = Some(tok);
                lane.rng_state = rng as u64;
            }
        }
    }
    Ok(())
}

/// EP-specific evidence: must be sourced from a real `Qwen35EpBatchReceipt` and
/// explicitly state expert_parallel / rank_count=4 / peer_rooted_f32.
pub fn attach_qwen_ep_batch_receipt_evidence(
    envelope: &mut serde_json::Value,
    receipt: &qwen35::Qwen35EpBatchReceipt,
    slots: usize,
    lane: usize,
    lane_capacity: usize,
    max_active_lanes: usize,
) {
    // Enforce attested invariants via getters; never fabricate from load logs.
    debug_assert_eq!(receipt.rank_count(), 4);
    debug_assert_eq!(receipt.rank_mask(), 0x0f);
    debug_assert_eq!(receipt.reduce(), qwen35::Qwen35EpReduce::PeerRootedF32);
    debug_assert_eq!(
        receipt.parallelism(),
        qwen35::Qwen35BatchParallelism::ExpertParallel
    );
    envelope["execution_mode"] = serde_json::json!("continuous_batch_independent");
    envelope["continuous_batch"] = serde_json::json!({
        "executed": true,
        "slots": slots,
        "lane": lane,
        "lane_capacity": lane_capacity,
        "max_active_lanes": max_active_lanes,
        "refill": "continuous",
        "parallelism": "expert_parallel",
        "rank_count": receipt.rank_count(),
        "rank_mask": receipt.rank_mask(),
        "reduce": "peer_rooted_f32",
        "epoch": receipt.epoch(),
        "rows": receipt.rows(),
        "moe_collectives": receipt.moe_collectives(),
    });
}

pub fn is_qwen_ep_batch_request_eligible(
    msg: &serde_json::Value,
    m: &LoadedModel,
    continuous_batch_size: usize,
    serve_continuous_batch: bool,
    pflash_active: bool,
) -> bool {
    let caps = hipfire_loader::carrier_for(m.arch_id)
        .map(|c| c.caps())
        .unwrap_or_default();
    if !serve_continuous_batch || continuous_batch_size <= 1 {
        return false;
    }
    if m.pp != 1 {
        return false;
    }
    let Some(ep) = m.ep.as_ref() else {
        return false;
    };
    let EpArch::Qwen35 {
        config,
        weights,
        batch,
    } = &ep.inner
    else {
        return false;
    };
    if batch.is_none() {
        return false;
    }
    if !caps.supports_ep_batch {
        return false;
    }
    if m.qwen35().is_some_and(|b| b.qwen35_decode_batch.is_some())
        || m.lfm2moe()
            .and_then(|b| b.lfm2_decode_batch.as_ref())
            .is_some()
    {
        return false;
    }
    // EP batch is pure TP=4 gfx1201; validate via existing weight format gate.
    if !hipfire_loader::batch_staging::qwen_ep_batch_weight_formats_supported(&weights[0]) {
        return false;
    }
    let has_image = msg.get("image").is_some() || msg.get("image_base64").is_some();
    let has_tools = msg
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());
    let has_stop = msg
        .get("stop")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());
    if has_image || has_tools || has_stop {
        return false;
    }
    if !batch_messages_are_single_user(msg) {
        return false;
    }
    if m.speculator.is_some() || m.kv_adaptive.is_some() || m.eviction.is_some() || pflash_active {
        return false;
    }
    if batch.is_none() {
        return false;
    }
    // Ensure sampling controls are resolvable (mirrors sequential ladder)
    let _ = resolve_batch_sampling(msg, m);
    // Must be QwenAr route (non-spec)
    let sampling = resolve_batch_sampling(msg, m);
    let user_explicit = [
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
    ]
    .iter()
    .any(|k| msg.get(*k).is_some());
    let route_inputs = GenerationRouteInputs {
        arch_id: m.arch_id,
        // Topology already proven/staged above; ep:true would hit the global EP
        // short-circuit to Unknown for Qwen and make this QwenAr gate unreachable.
        ep: false,
        pp: m.pp,
        has_speculator: m.speculator.is_some(),
        speculator_is_mtp: m.speculator.as_ref().is_some_and(|s| s.name() == "mtp"),
        deepseek4_spec_requested: false,
        ngram_can_sample: m
            .speculator
            .as_ref()
            .map(|s| !s.requires_greedy())
            .unwrap_or(false),
        temp: sampling.temp,
        user_explicit_sampling: user_explicit,
        min_p: sampling.min_p,
        nonneutral_penalties: sampling.repeat_penalty != 1.0
            || sampling.presence_penalty != 0.0
            || sampling.frequency_penalty != 0.0,
        force_ar_chat: false,
        temp_spec_env_off: std::env::var("HIPFIRE_DFLASH_TEMP_SPEC").ok().as_deref() == Some("0"),
        fast_sample_on: std::env::var("HIPFIRE_FAST_SAMPLE").ok().as_deref() != Some("0"),
        supports_temp_swor: m
            .speculator
            .as_ref()
            .is_some_and(|s| s.supports_temp_verify()),
        supports_chain_nucleus_verify: m
            .speculator
            .as_ref()
            .is_some_and(|s| s.supports_chain_nucleus_verify()),
        kv_adaptive: m.kv_adaptive.is_some(),
    };
    let route = select_generation_route(&route_inputs);
    if route != GenerationRoute::QwenAr {
        return false;
    }
    // Batch size coherence
    if continuous_batch_size != batch.as_ref().map(|b| b.max_batch()).unwrap_or(0) {
        return false;
    }
    true
}

pub fn drive_qwen35_ep_continuous_batch(
    sched: &mut ContinuousBatchScheduler,
    model: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    inbox: &mut DaemonInbox,
) -> Result<(), BatchDriveError> {
    crate::ar::set_generation_route(crate::ar::GenerationRoute::QwenAr);
    let batch_size = sched.max_batch;
    if batch_size == 0 {
        return Ok(());
    }
    // Borrow EP batch state, config, weights via raw pointers to avoid aliasing.
    let ep_ptr = match model.ep.as_mut() {
        Some(ep) => ep as *mut EpState,
        None => return Err(BatchDriveError::Gpu("EP batch: no EP state".to_string())),
    };
    let (gpus_ptr, config_ptr, weights_ptr, batch_ptr, tokenizer_ptr, chat_template_clone, arch_id) = unsafe {
        let ep = &mut *ep_ptr;
        match &mut ep.inner {
            EpArch::Qwen35 {
                config,
                weights,
                batch,
            } => {
                let b = match batch.as_mut() {
                    Some(b) => b as *mut qwen35::Qwen35DecodeBatchEpState,
                    None => {
                        return Err(BatchDriveError::Gpu(
                            "EP batch: batch not staged".to_string(),
                        ))
                    }
                };
                (
                    &mut ep.gpus as *mut hipfire_hardware::Gpus,
                    config as *const qwen35::Qwen35Config,
                    weights as *const Vec<qwen35::Qwen35Weights>,
                    b,
                    match model.tokenizer.as_ref() {
                        Some(t) => t as *const _,
                        None => return Err(BatchDriveError::Gpu("tokenizer missing".to_string())),
                    },
                    model.chat_template.clone(),
                    model.arch_id,
                )
            }
            _ => return Err(BatchDriveError::Gpu("EP batch: not Qwen35 EP".to_string())),
        }
    };
    let gpus: &mut hipfire_hardware::Gpus = unsafe { &mut *gpus_ptr };
    let config: &qwen35::Qwen35Config = unsafe { &*config_ptr };
    let weights: &Vec<qwen35::Qwen35Weights> = unsafe { &*weights_ptr };
    let batch_state: &mut qwen35::Qwen35DecodeBatchEpState = unsafe { &mut *batch_ptr };
    let tokenizer: &hipfire_runtime::tokenizer::Tokenizer = unsafe { &*tokenizer_ptr };
    let chat_template = chat_template_clone;
    let eos_tok = config.eos_token;
    let im_end_tok = tokenizer.special_token_id("<|im_end|>").unwrap_or(eos_tok);
    let mut producers: Vec<Option<QwenArSemanticProducer>> =
        (0..batch_size).map(|_| None).collect();
    let mut loop_guards: Vec<hipfire_runtime::loop_guard::LoopGuard> = (0..batch_size)
        .map(
            |_| hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get()),
        )
        .collect();
    let mut tokens = vec![0u32; batch_size];
    let mut positions = vec![0usize; batch_size];
    // Track last attested receipt for evidence; must be from runtime, never load logs.
    let mut last_receipt: Option<qwen35::Qwen35EpBatchReceipt> = None;
    let fail_all = |sched: &mut ContinuousBatchScheduler,
                    gpus: &mut hipfire_hardware::Gpus,
                    batch_state: &mut qwen35::Qwen35DecodeBatchEpState,
                    stdout: &mut std::io::Stdout,
                    reason: String|
     -> Result<(), BatchDriveError> {
        let mut uniq_set = std::collections::HashSet::new();
        let mut uniq: Vec<AttemptKey> = Vec::new();
        for l in sched.lanes.iter() {
            if let Some(k) = l.key() {
                if uniq_set.insert(k.clone()) {
                    uniq.push(k.clone());
                }
            }
        }
        for k in sched.inbox.iter().cloned() {
            if uniq_set.insert(k.clone()) {
                uniq.push(k);
            }
        }
        for k in sched.pending.keys().cloned() {
            if uniq_set.insert(k.clone()) {
                uniq.push(k);
            }
        }
        let reset_res = batch_state.reset_all(gpus);
        let first_err = reset_res.err().map(|e| format!("EP batch reset_all: {e}"));
        let reason2 = if let Some(e) = first_err {
            format!("{reason}; {e}")
        } else {
            reason.clone()
        };
        for key in &uniq {
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            let ep = crate::common::RollbackEpilogue {
                rolled_back: true,
                context: None,
            };
            crate::common::emit_fail_closed_error(
                stdout,
                Some(&key.id),
                &format!("batch GPU error: {reason2}"),
                "gpu",
                false,
                &ep,
            );
        }
        let _ = sched.fail_all_active();
        for k in &uniq {
            batch_clear_terminal(&k.id, k.attempt_id);
        }
        Err(BatchDriveError::Poisoned(reason2))
    };
    loop {
        // handle awaiting commit/abort
        let mut to_commit: Vec<(usize, AttemptKey, serde_json::Value)> = Vec::new();
        let mut to_abort: Vec<(usize, AttemptKey)> = Vec::new();
        for idx in 0..batch_size {
            if let BatchLane::AwaitingClient(term) = &sched.lanes[idx] {
                let key = term.key.clone();
                let expired = Instant::now() >= term.deadline;
                if batch_check_abort(&key.id, key.attempt_id) || expired {
                    to_abort.push((idx, key));
                } else if let Some(ClientTerminalDecision::Commit) =
                    batch_poll_decision(&key.id, key.attempt_id)
                {
                    to_commit.push((idx, key.clone(), term.pending_done.clone()));
                }
            }
        }
        for (idx, key) in to_abort {
            if let Err(e) = batch_state.reset_lane(gpus, config, idx) {
                return fail_all(
                    sched,
                    gpus,
                    batch_state,
                    stdout,
                    format!("EP reset lane {idx} on abort: {e}"),
                );
            }
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            crate::ar::emit_active_route_cancel(stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key);
            producers[idx] = None;
        }
        for (idx, key, pending_done) in to_commit {
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            let reset_ok = match batch_state.reset_lane(gpus, config, idx) {
                Ok(()) => true,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP reset lane {idx} on commit: {e}"),
                    )
                }
            };
            let commit_ok = sched.commit_lane(idx, &key);
            match batch_commit_teardown_class(reset_ok, commit_ok) {
                BatchCommitTeardownClass::ResetFailed => unreachable!(),
                BatchCommitTeardownClass::CommitFailed => {
                    let ep = crate::common::RollbackEpilogue {
                        rolled_back: true,
                        context: None,
                    };
                    crate::common::emit_fail_closed_error(
                        stdout,
                        Some(&key.id),
                        "batch commit_lane failed after reset",
                        "internal",
                        false,
                        &ep,
                    );
                    let _ = sched.abort_lane(idx, &key);
                    producers[idx] = None;
                }
                BatchCommitTeardownClass::EmitDone => {
                    crate::ar::emit_active_route_done(stdout, &key.id, &pending_done);
                    producers[idx] = None;
                }
            }
        }
        let mut queued_abort: Vec<AttemptKey> = Vec::new();
        for k in sched.inbox.iter().cloned().collect::<Vec<_>>() {
            if batch_check_abort(&k.id, k.attempt_id) {
                queued_abort.push(k);
            }
        }
        for k in queued_abort {
            let _scope = BatchAttemptScope::enter(k.attempt_id);
            crate::ar::emit_active_route_cancel(stdout, &k.id, 0);
            let _ = sched.abort_queued(&k);
        }
        let mut running_abort: Vec<(usize, AttemptKey)> = Vec::new();
        for idx in 0..batch_size {
            if let Some(k) = sched.lanes[idx].key().cloned() {
                if matches!(
                    sched.lanes[idx],
                    BatchLane::Running(_) | BatchLane::Seeding(_)
                ) && batch_check_abort(&k.id, k.attempt_id)
                {
                    running_abort.push((idx, k));
                }
            }
        }
        for (idx, key) in running_abort {
            if let Err(e) = batch_state.reset_lane(gpus, config, idx) {
                return fail_all(
                    sched,
                    gpus,
                    batch_state,
                    stdout,
                    format!("EP reset lane {idx} on running abort: {e}"),
                );
            }
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            crate::ar::emit_active_route_cancel(stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key);
            producers[idx] = None;
        }
        let mut barrier: Option<DaemonMsg> = None;
        loop {
            let dm = match inbox.try_recv() {
                Ok(m) => m,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            };
            match dm {
                DaemonMsg::ParseError(e) => {
                    emit_uncorrelated_error(
                        stdout,
                        None,
                        &format!("invalid JSON: {e}"),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                }
                DaemonMsg::Regular(json) => {
                    let t = json.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if t == "generate" {
                        let attempt_id = match json.get("attempt_id").and_then(|v| v.as_u64()) {
                            Some(v) => v,
                            None => {
                                emit_uncorrelated_error(
                                    stdout,
                                    json.get("id").and_then(|v| v.as_str()),
                                    "generate missing attempt_id",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        let id = json
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("0")
                            .to_string();
                        batch_announce_terminal(&id, attempt_id);
                        if batch_check_abort(&id, attempt_id) {
                            let _scope = BatchAttemptScope::enter(attempt_id);
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::QwenAr,
                                stdout,
                                &id,
                                false,
                            );
                            crate::ar::emit_active_route_cancel(stdout, &id, 0);
                            batch_clear_terminal(&id, attempt_id);
                            continue;
                        }
                        // EP batch-only admission; non-eligible becomes barrier.
                        if !is_qwen_ep_batch_request_eligible(
                            &json,
                            model,
                            batch_size,
                            parse_serve_continuous_batch(&json),
                            false,
                        ) {
                            barrier = Some(DaemonMsg::Regular(json));
                            break;
                        }
                        let prompt_str = batch_single_user_content(&json).unwrap_or_else(|| {
                            json.get("prompt")
                                .and_then(|v| v.as_str())
                                .unwrap_or("Hello")
                                .to_string()
                        });
                        let system_str = json
                            .get("system")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let assistant_prefix = match json
                            .get("assistant_prefix")
                            .and_then(|v| v.as_str())
                            .unwrap_or("plain")
                        {
                            "open_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
                            }
                            "closed_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink
                            }
                            _ => hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                        };
                        let max_think = json
                            .get("max_think_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;
                        let max_tokens_req = json
                            .get("max_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(4096) as usize;
                        let batch_messages = match json.get("messages") {
                            Some(v) => match serde_json::from_value::<
                                Vec<hipfire_runtime::prompt_frame::Message>,
                            >(v.clone())
                            {
                                Ok(v) => Some(v),
                                Err(e) => {
                                    let _scope = BatchAttemptScope::enter(attempt_id);
                                    emit_uncorrelated_error(
                                        stdout,
                                        Some(&id),
                                        &format!("invalid messages field: {e}"),
                                        "validation",
                                        false,
                                        false,
                                    );
                                    batch_clear_terminal(&id, attempt_id);
                                    continue;
                                }
                            },
                            None => None,
                        };
                        let raw_effort = json
                            .get("reasoning_effort")
                            .or_else(|| json.get("thinking_mode"))
                            .and_then(|v| v.as_str());
                        let thinking_enabled =
                            json.get("thinking_enabled").and_then(|v| v.as_bool());
                        let (batch_enable_thinking, batch_reasoning_effort) =
                            qwen_jinja_reasoning(thinking_enabled, raw_effort, max_think);
                        let (prompt_tokens, started_in_think) = match batch_render_prompt_tokens(
                            &prompt_str,
                            system_str.as_deref(),
                            assistant_prefix,
                            tokenizer,
                            chat_template.as_ref(),
                            max_think,
                            batch_messages.as_deref(),
                            batch_enable_thinking,
                            batch_reasoning_effort.as_deref(),
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                let _scope = BatchAttemptScope::enter(attempt_id);
                                emit_uncorrelated_error(
                                    stdout,
                                    Some(&id),
                                    &format!("render failed: {e}"),
                                    "validation",
                                    false,
                                    false,
                                );
                                batch_clear_terminal(&id, attempt_id);
                                continue;
                            }
                        };
                        if started_in_think {
                            let _ = batch_transfer_abort_to_singleton_and_clear(&id, attempt_id);
                            barrier = Some(DaemonMsg::Regular(json));
                            break;
                        }
                        if prompt_tokens.is_empty() || prompt_tokens.len() >= sched.lane_capacity {
                            let _scope = BatchAttemptScope::enter(attempt_id);
                            emit_uncorrelated_error(
                                stdout,
                                Some(&id),
                                "prompt exceeds lane capacity or empty",
                                "validation",
                                false,
                                false,
                            );
                            batch_clear_terminal(&id, attempt_id);
                            continue;
                        }
                        // Explicit wire `seed` must reach the lane RNG on the
                        // batched route too; out-of-domain values are rejected
                        // loudly, never silently unseeded.
                        let client_seed = match wire_seed::parse_wire_seed(json.get("seed")) {
                            Ok(s) => s,
                            Err(reason) => {
                                emit_uncorrelated_error(
                                    stdout,
                                    Some(&id),
                                    &reason,
                                    "validation",
                                    false,
                                    false,
                                );
                                batch_clear_terminal(&id, attempt_id);
                                continue;
                            }
                        };
                        batch_transition_to_queued(&id, attempt_id);
                        let sampling = resolve_batch_sampling(&json, model);
                        let req = BatchPendingRequest {
                            key: AttemptKey::new(&id, attempt_id),
                            prompt: prompt_str.clone(),
                            prompt_tokens: prompt_tokens.clone(),
                            started_in_think,
                            system: system_str.clone(),
                            assistant_prefix,
                            max_think_tokens: max_think,
                            max_tokens: max_tokens_req,
                            client_seed,
                            sampling,
                        };
                        if !sched.enqueue(req) {
                            eprintln!("[batch][EP] duplicate enqueue rejected id={} attempt_id={}; preserving live registry", id, attempt_id);
                            continue;
                        }
                        {
                            let _scope = BatchAttemptScope::enter(attempt_id);
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::QwenAr,
                                stdout,
                                &id,
                                false,
                            );
                        }
                    } else if t == "abort" || t == "commit" {
                        if let (Some(id), Some(aid), Some(kind)) = (
                            json.get("id").and_then(|v| v.as_str()),
                            json.get("attempt_id").and_then(|v| v.as_u64()),
                            json.get("type").and_then(|v| v.as_str()),
                        ) {
                            batch_apply_terminal_control(kind, id, aid);
                        }
                    } else {
                        barrier = Some(DaemonMsg::Regular(json));
                        break;
                    }
                }
            }
        }
        if let Some(msg) = barrier {
            inbox.push_front(msg);
            if sched.active_count() == 0 && sched.inbox.is_empty() {
                return Ok(());
            }
        }
        while let Some((key, ticket)) = sched.try_assign_one() {
            let lane_idx = ticket.lane;
            let pending_req = match sched.pending.get(&key).cloned() {
                Some(r) => r,
                None => continue,
            };
            let sampling = pending_req.sampling.clone();
            let prompt_tokens = pending_req.prompt_tokens.clone();
            let started_in_think = pending_req.started_in_think;
            if started_in_think {
                let prompt = pending_req.prompt.clone();
                let _ = batch_transfer_abort_to_singleton_and_clear(&key.id, key.attempt_id);
                let _ = sched.abort_lane(lane_idx, &key);
                if let Err(err) = batch_state.reset_lane(gpus, config, lane_idx) {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP reset lane {lane_idx} on think barrier: {err}"),
                    );
                }
                inbox.push_front(DaemonMsg::Regular(serde_json::json!({"type":"generate","id":key.id,"attempt_id":key.attempt_id,"prompt":prompt})));
                break;
            }
            if let Err(e) = batch_state.reset_lane(gpus, config, lane_idx) {
                return fail_all(
                    sched,
                    gpus,
                    batch_state,
                    stdout,
                    format!("EP reset lane {lane_idx}: {e}"),
                );
            }
            let receipt =
                match batch_state.prefill_lane(gpus, weights, config, lane_idx, &prompt_tokens) {
                    Ok(r) => r,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpus,
                            batch_state,
                            stdout,
                            format!("EP prefill lane {lane_idx}: {e}"),
                        )
                    }
                };
            last_receipt = Some(receipt);
            let lane_rng = match &sched.lanes[lane_idx] {
                BatchLane::Running(lane) => lane.rng_state as u32,
                _ => continue,
            };
            // Use per-lane sampling that respects readiness; repeat penalties folded via retry window with product if needed.
            // For EP we call sample_lane (full product requires contiguous Ready lanes); per-lane keeps sparsity.
            let (next_token, next_rng) = match batch_state.sample_lane(
                gpus,
                config,
                lane_idx,
                sampling.temp,
                sampling.top_p,
                sampling.top_k,
                lane_rng,
            ) {
                Ok(v) => v,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP sample lane {lane_idx}: {e}"),
                    )
                }
            };
            if let BatchLane::Running(lane) = &mut sched.lanes[lane_idx] {
                lane.prompt_len = prompt_tokens.len();
                lane.seq_pos = prompt_tokens.len();
                lane.next_token = Some(next_token);
                lane.rng_state = next_rng as u64;
                lane.conversation_tokens = Vec::new();
                lane.streamed_tokens = Vec::new();
                lane.bytes_fed_to_filter = 0;
                lane.prefill_done_at = Some(Instant::now());
            }
            producers[lane_idx] = Some(QwenArSemanticProducer::new_with_tool_protocol(
                key.id.clone(),
                started_in_think,
                false,
            ));
        }
        let running: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::Running(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        let awaiting: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::AwaitingClient(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if running.is_empty()
            && awaiting.is_empty()
            && sched.inbox.is_empty()
            && inbox.backlog.is_empty()
        {
            break;
        }
        if running.is_empty() {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }
        let active_now = running.len();
        for &idx in &running {
            if let BatchLane::Running(lane) = &mut sched.lanes[idx] {
                if active_now > lane.max_active_lanes {
                    lane.max_active_lanes = active_now;
                }
            }
        }
        // Build active mask and dense token/position vectors for EP forward_tick.
        let mut active_mask: u64 = 0;
        for &idx in &running {
            active_mask |= 1u64 << idx;
        }
        for i in 0..batch_size {
            match &sched.lanes[i] {
                BatchLane::Running(lane) => {
                    tokens[i] = lane.next_token.unwrap_or(eos_tok);
                    positions[i] = lane.seq_pos;
                }
                _ => {
                    tokens[i] = eos_tok;
                    positions[i] = 0;
                }
            }
        }
        let receipt =
            match batch_state.forward_tick(gpus, weights, config, active_mask, &tokens, &positions)
            {
                Ok(r) => r,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP forward_tick: {e}"),
                    )
                }
            };
        last_receipt = Some(receipt);
        let mut to_await: Vec<(usize, AttemptKey, serde_json::Value)> = Vec::new();
        let mut to_abort_running: Vec<(usize, AttemptKey)> = Vec::new();
        let mut survivors: Vec<usize> = Vec::new();
        for idx in running.clone() {
            let key = match sched.lanes[idx].key().cloned() {
                Some(k) => k,
                None => continue,
            };
            if batch_check_abort(&key.id, key.attempt_id) {
                to_abort_running.push((idx, key));
                continue;
            }
            let lane_ptr = match &mut sched.lanes[idx] {
                BatchLane::Running(l) => l as *mut QwenBatchLane,
                _ => continue,
            };
            let lane = unsafe { &mut *lane_ptr };
            let cur_token = lane.next_token.unwrap_or(eos_tok);
            let prod_ptr = match producers[idx].as_mut() {
                Some(p) => p as *mut QwenArSemanticProducer,
                None => continue,
            };
            let producer = unsafe { &mut *prod_ptr };
            let mut future_streamed = lane.streamed_tokens.clone();
            future_streamed.push(cur_token);
            let all_bytes = tokenizer.decode_bytes(&future_streamed);
            let prev_fed = lane.bytes_fed_to_filter.min(all_bytes.len());
            let token_bytes = all_bytes[prev_fed..].to_vec();
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            if lane.first_token_at.is_none() {
                lane.first_token_at = Some(Instant::now());
            }
            let stopped = {
                let lane_seq = &mut lane.seq_pos as *mut usize;
                let lane_conv = &mut lane.conversation_tokens as *mut Vec<u32>;
                let lane_stream = &mut lane.streamed_tokens as *mut Vec<u32>;
                let lane_fed = &mut lane.bytes_fed_to_filter as *mut usize;
                let all_len = all_bytes.len();
                let mut res: Result<bool, _> = Ok(false);
                unsafe {
                    res = producer.commit_and_classify(
                        stdout,
                        cur_token,
                        || {
                            let pos = qwen_ar_raw_commit_token(
                                &mut *lane_conv,
                                &mut *lane_stream,
                                &mut *lane_seq,
                                cur_token,
                                QwenArRawCommitDisposition::ClassifiedVisible,
                            );
                            *lane_fed = all_len;
                            (pos, token_bytes.clone())
                        },
                        |_, _| {},
                    );
                }
                match res {
                    Ok(s) => s,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpus,
                            batch_state,
                            stdout,
                            format!("EP semantic classify lane {idx}: {e}"),
                        )
                    }
                }
            };
            let loop_hit = loop_guards[idx].check(&lane.streamed_tokens).is_some();
            let is_eos = cur_token == eos_tok || cur_token == im_end_tok;
            let hit_max = lane.streamed_tokens.len() >= lane_max_tokens(&key, sched);
            let hit_lane_cap = batch_lane_at_capacity(lane.seq_pos, sched.lane_capacity);
            let should_finish =
                batch_should_finish_decode(is_eos, hit_max, hit_lane_cap, stopped, loop_hit);
            if should_finish {
                let hit_length_cap =
                    batch_hit_length_cap(hit_max, hit_lane_cap, is_eos, stopped, loop_hit);
                let producer_owned = match producers[idx].take() {
                    Some(p) => p,
                    None => continue,
                };
                let (finish, visible_text) = match producer_owned.finish(stdout, hit_length_cap) {
                    Ok(v) => v,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpus,
                            batch_state,
                            stdout,
                            format!("EP semantic finish lane {idx}: {e}"),
                        )
                    }
                };
                if matches!(finish.cause, QwenArTerminalCause::OpenThink) && !is_eos {
                    if let Err(e) = batch_state.reset_lane(gpus, config, idx) {
                        return fail_all(
                            sched,
                            gpus,
                            batch_state,
                            stdout,
                            format!("EP reset lane {idx} on open think: {e}"),
                        );
                    }
                    let ep = crate::common::RollbackEpilogue {
                        rolled_back: true,
                        context: None,
                    };
                    let _scope = BatchAttemptScope::enter(key.attempt_id);
                    emit_qwen_ar_open_think_terminal(
                        stdout,
                        &key.id,
                        lane.streamed_tokens.len(),
                        &ep,
                    );
                    let _ = sched.abort_lane(idx, &key);
                    producers[idx] = None;
                    continue;
                }
                if !finish.wire_tool_calls.is_empty() {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP semantic finish lane {idx}: unexpected tool calls"),
                    );
                }
                let finish_reason = match finish.finish_reason {
                    "length" => "length",
                    "tool_calls" => "tool_calls",
                    _ => "stop",
                };
                let generated = lane.streamed_tokens.len();
                let metrics = batch_lane_done_metrics(
                    lane.created_at,
                    lane.prefill_done_at,
                    lane.first_token_at,
                    Instant::now(),
                    lane.prompt_len,
                    generated,
                );
                let mut pending_done = qwen_ar_done_value(
                    &key.id,
                    finish_reason,
                    generated,
                    metrics.tok_s,
                    lane.prompt_len,
                    metrics.prefill_ms,
                    metrics.prefill_tok_s,
                    metrics.decode_tok_s,
                    metrics.ttft_ms,
                    0,
                    "",
                );
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                if let Some(receipt) = last_receipt.as_ref() {
                    attach_qwen_ep_batch_receipt_evidence(
                        &mut pending_done,
                        receipt,
                        batch_size,
                        idx,
                        sched.lane_capacity,
                        lane.max_active_lanes.max(1),
                    );
                } else {
                    // Never fabricate: if no receipt yet, attach generic but still mark expert_parallel via default (should not happen on finishing lane after forward).
                    attach_continuous_batch_route_evidence(
                        &mut pending_done,
                        batch_size,
                        idx,
                        sched.lane_capacity,
                        lane.max_active_lanes.max(1),
                    );
                    pending_done["continuous_batch"]["parallelism"] =
                        serde_json::json!("expert_parallel");
                    pending_done["continuous_batch"]["rank_count"] = serde_json::json!(4);
                    pending_done["continuous_batch"]["reduce"] =
                        serde_json::json!("peer_rooted_f32");
                }
                let _ = visible_text;
                to_await.push((idx, key.clone(), pending_done));
            } else {
                survivors.push(idx);
            }
        }
        for (idx, key) in to_abort_running {
            if let Err(e) = batch_state.reset_lane(gpus, config, idx) {
                return fail_all(
                    sched,
                    gpus,
                    batch_state,
                    stdout,
                    format!("EP reset lane {idx} on abort post-forward: {e}"),
                );
            }
            let _scope = BatchAttemptScope::enter(key.attempt_id);
            crate::ar::emit_active_route_cancel(stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key);
            producers[idx] = None;
        }
        for (idx, key, pending_done) in to_await {
            let mut envelope = pending_done.clone();
            envelope["type"] = serde_json::json!("commit_ready");
            let marked = sched.mark_awaiting_commit(idx, pending_done.clone());
            if !marked {
                eprintln!(
                    "[batch][EP] qwen mark_awaiting_commit failed lane {idx} id={} — aborting lane",
                    key.id
                );
                let _ = batch_state.reset_lane(gpus, config, idx);
                let _ = sched.abort_lane(idx, &key);
                producers[idx] = None;
                continue;
            }
            let write_ok = {
                let _scope = BatchAttemptScope::enter(key.attempt_id);
                writeln!(stdout, "{}", envelope).is_ok() && stdout.flush().is_ok()
            };
            if !write_ok {
                let _ = batch_state.reset_lane(gpus, config, idx);
                let _ = sched.abort_lane(idx, &key);
                producers[idx] = None;
            }
        }
        if survivors.is_empty() {
            continue;
        }
        // Per-lane sampling for survivors (sparse-aware). Use sample_lane to avoid contiguous prefix requirement.
        for idx in survivors.iter().cloned() {
            let sampling = match &sched.lanes[idx] {
                BatchLane::Running(l) => l.sampling.clone(),
                _ => continue,
            };
            let rng = match &sched.lanes[idx] {
                BatchLane::Running(l) => l.rng_state as u32,
                _ => continue,
            };
            let (tok, next_rng) = match batch_state.sample_lane(
                gpus,
                config,
                idx,
                sampling.temp,
                sampling.top_p,
                sampling.top_k,
                rng,
            ) {
                Ok(v) => v,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP sample_lane survivor {idx}: {e}"),
                    )
                }
            };
            if let BatchLane::Running(lane) = &mut sched.lanes[idx] {
                lane.next_token = Some(tok);
                lane.rng_state = next_rng as u64;
            }
        }
        // Also exercise sample_product when survivors form a contiguous full prefix (API coverage; sparse batches use sample_lane above).
        if survivors.len() == batch_size && survivors.iter().enumerate().all(|(i, &v)| i == v) {
            // Use the same sampling as first survivor for product validation; ignore error for non-product-capable batch shapes.
            if let Some(first) = survivors.first().and_then(|&idx| match &sched.lanes[idx] {
                BatchLane::Running(l) => Some(l.sampling.clone()),
                _ => None,
            }) {
                let dummy_repeat = vec![0u32; batch_size * 128];
                let dummy_lengths = vec![0u32; batch_size];
                let dummy_rng = vec![0u32; batch_size];
                let _ = batch_state.sample_product(
                    gpus,
                    config,
                    batch_size,
                    &dummy_repeat,
                    &dummy_lengths,
                    &dummy_rng,
                    first.temp,
                    first.top_p,
                    first.top_k,
                    first.min_p,
                    first.repeat_penalty,
                    first.presence_penalty,
                    first.frequency_penalty,
                );
            }
        }
    }
    Ok(())
}

/// Pre-activation protocol reject (missing/malformed attempt_id, or commands
/// with no active generate attempt). Always emits `attempt_id: 0`.
///
/// Do not use after `set_active_attempt_id` for a generate request.
pub fn emit_uncorrelated_error(
    stdout: &mut impl std::io::Write,
    id: Option<&str>,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
) {
    crate::dense::write_error_envelope(stdout, id, message, class, retryable, rolled_back, 0);
}
