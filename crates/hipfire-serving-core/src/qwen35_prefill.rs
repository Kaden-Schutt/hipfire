// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 prefill: turning a (possibly multi-session) batch of prompts into
//! resident KV/DeltaNet state.
//!
//! Covers the fused-dense and grouped-MoE batched suffix-prefill kernels, the
//! serial reference path, batch-prompt materialization, prefix-hash candidate
//! computation + preflight (for prefix-cache reuse), semantic-boundary
//! checkpoints, and the per-session prefill-checkpoint emit helpers. Extracted
//! verbatim from the former `main.rs` monolith (no behavior change); items
//! called from `main.rs` are `pub`.

use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use hipfire_arch_qwen35::qwen35;
use hipfire_generate::sampler::SamplerConfig;
use hipfire_generate::{
    build_qwen35_fused_dense_prefill_batch_contract, compute_qwen35_prefix_hash,
    plan_generate_batch_prefill_qwen35, prefix_hash_preflight_done_json,
    qwen35_fused_prefill_boundary_cuts, qwen35_generate_batch_prefill_done_json,
    qwen35_generate_batch_prefill_session_done_json, qwen35_prefill_checkpoint_boundary_kind,
    qwen35_prefill_checkpoint_session_id, qwen35_prefill_scratch_target_batch,
    select_qwen35_prefill_batch_backend, validate_qwen35_fused_grouped_moe_prefill_batch_preflight,
    GenerateBatchPrefillEnvelope, GenerateBatchPrefillPlan, GenerateBatchPrefillSession,
    PrefixHashPreflightCandidate, PrefixHashPreflightEnvelope, Qwen35FusedDensePrefillInputKind,
    Qwen35PrefillBatchBackend, Qwen35PrefillBatchResult, Qwen35PrefillCheckpointHook,
    Qwen35PrefillCheckpointKind, Qwen35PrefillSessionResult, Qwen35PreparedPrefillSession,
    Qwen35SemanticBoundaryCheckpoint,
};
use hipfire_model::is_qwen35_family_arch_id;
use hipfire_prompt as prompt_frame;
use hipfire_runtime::sampler;
use hipfire_state::{
    generate_state_kinds_include_required, model_worker_runtime_view_json,
    SequenceStateArenaBackend, SequenceStateCheckpointRequest, SequenceStateForkRequest,
    SequenceStatePageKind,
};

use crate::model::{effective_raw, LoadedModel};
use crate::output_filter::normalize_daemon_prompt;
use crate::session::{
    loaded_model_state_arena_backend, loaded_model_worker_runtime_view, qwen35_activate_session,
    qwen35_active_logical_position, qwen35_allocate_session_state, qwen35_save_active_session,
    sequence_state_arena_activate_session, sequence_state_arena_active_logical_position,
    sequence_state_arena_checkpoint_session_state, sequence_state_arena_fork_session_state,
    sequence_state_arena_is_session_resident, sequence_state_arena_reset_active_session,
    sequence_state_arena_resident_session_count,
    validate_qwen35_fused_grouped_moe_prefill_model_capability, Qwen35RequestSessionState,
};

/// Emit a `generate_batch_prefill_session_done` checkpoint event for one
/// session as the batch prefill advances past a semantic boundary (the hook the
/// prefill kernels call so clients can resume from a cached prefix).
pub fn emit_qwen35_prefill_checkpoint(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    arena_backend: SequenceStateArenaBackend,
    hook: Qwen35PrefillCheckpointHook<'_>,
) -> Result<String, String> {
    if qwen35_prefill_checkpoint_boundary_kind(hook).is_empty() {
        return Err("qwen35 prefill checkpoint boundary kind is empty".to_string());
    }
    let checkpoint_id = qwen35_prefill_checkpoint_session_id(hook);
    sequence_state_arena_checkpoint_session_state(
        arena_backend,
        m,
        gpu,
        SequenceStateCheckpointRequest {
            source_session_id: hook.source_state_handle,
            dest_session_id: &checkpoint_id,
            expected_logical_position: hook.logical_position,
            requested_prefix_hash: None,
            checkpoint_prefix_hash: Some(hook.prefix_hash),
        },
    )?;
    Ok(checkpoint_id)
}

/// Prefill the active session's prompt suffix into resident KV/DeltaNet state,
/// selecting the batched (fused-dense / grouped-MoE) or serial backend and
/// emitting per-boundary checkpoints. The single-session prefill entry point.
pub fn qwen35_prefill_active_session(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    tokens: &[u32],
    replay_as_generated_suffix: bool,
) -> Result<usize, String> {
    if tokens.is_empty() {
        return Ok(0);
    }
    if m.seq_pos + tokens.len() > m.physical_cap {
        return Err(format!(
            "generate_batch_prefill exceeds loaded KV budget: seq_pos={} + prefill={} > physical_cap={}",
            m.seq_pos,
            tokens.len(),
            m.physical_cap
        ));
    }
    let config = m
        .q35_config
        .as_ref()
        .ok_or_else(|| "qwen35 config missing".to_string())?;
    let weights = m
        .q35_weights
        .as_ref()
        .ok_or_else(|| "qwen35 weights missing".to_string())?;
    let scratch = m
        .q35_scratch
        .as_ref()
        .ok_or_else(|| "qwen35 scratch missing; PP batch-prefill is not supported".to_string())?;
    let kv = m
        .kv_cache
        .as_mut()
        .ok_or_else(|| "qwen35 active session missing KV cache".to_string())?;
    let dn = m
        .dn_state
        .as_mut()
        .ok_or_else(|| "qwen35 active session missing DeltaNet state".to_string())?;
    if replay_as_generated_suffix {
        for &token in tokens {
            m.conversation_tokens.push(token);
            qwen35::forward_scratch(gpu, weights, config, token, m.seq_pos, kv, dn, scratch)
                .map_err(|e| format!("qwen35 forward_scratch suffix replay failed: {e:?}"))?;
            m.seq_pos += 1;
        }
    } else {
        let pos = m.seq_pos;
        qwen35::forward_prefill_batch(
            gpu, weights, config, tokens, pos, kv, dn, scratch, None, None, None, None,
        )
        .map_err(|e| format!("qwen35 forward_prefill_batch failed: {e:?}"))?;
        m.seq_pos += tokens.len();
        m.conversation_tokens.extend_from_slice(tokens);
    }
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("qwen35 batch-prefill session sync failed: {e:?}"))?;
    m.q35_active_prefilled_generated_suffix_len = if replay_as_generated_suffix {
        tokens.len()
    } else {
        0
    };
    Ok(tokens.len())
}

/// Serial-path prefill of one owned session's token segment (the per-session
/// unit the serial batch driver loops over).
pub fn qwen35_prefill_owned_session_serial_segment(
    gpu: &mut rdna_compute::Gpu,
    weights: &qwen35::Qwen35Weights,
    config: &qwen35::Qwen35Config,
    scratch: &qwen35::Qwen35Scratch,
    state: &mut Qwen35RequestSessionState,
    tokens: &[u32],
) -> Result<usize, String> {
    for &token in tokens {
        qwen35::forward_scratch(
            gpu,
            weights,
            config,
            token,
            state.seq_pos,
            &mut state.kv_cache,
            &mut state.dn_state,
            scratch,
        )
        .map_err(|e| format!("qwen35 serial boundary prefill segment failed: {e:?}"))?;
        state.seq_pos += 1;
        state.conversation_tokens.push(token);
    }
    Ok(tokens.len())
}

/// Turn a batch-prefill session request (prompt text or pre-tokenized suffix +
/// system prompt) into the concrete token sequence to prefill, applying the
/// chat frame and prompt normalization.
pub fn qwen35_materialize_batch_prefill_prompt(
    m: &LoadedModel,
    session: &GenerateBatchPrefillSession,
) -> Result<Vec<u32>, String> {
    let tokenizer = m
        .tokenizer
        .as_ref()
        .ok_or_else(|| "tokenizer not loaded".to_string())?;
    let prompt = session.prompt.as_deref().unwrap_or("");
    let prompt_norm = normalize_daemon_prompt(prompt);
    let prompt = prompt_norm.as_ref();
    let raw_q_tokens = tokenizer.encode(prompt);
    // Prompt-hash/preload sessions that declare a zero logical position need
    // to materialize the full prompt from position zero even if another active
    // resident session has advanced m.seq_pos. Attached prompt sessions also
    // render from zero so the daemon can slice off the cached prefix that was
    // fingerprinted by prefix_hash_preflight.
    let seq_pos_for_prompt = if session.state_handle.runtime_state_handle.is_some()
        || (session.state_handle.logical_position == 0
            && session.state_handle.cached_prefix_tokens == 0)
    {
        0
    } else {
        m.seq_pos
    };
    let assistant_prefix =
        prompt_frame::AssistantPrefix::from_label(Some(&session.assistant_prefix));
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
    let try_jinja = jinja_enabled && seq_pos_for_prompt == 0 && m.chat_template.is_some();
    let system_prompt = session.system_prompt.as_deref();
    let tools = session.tools.as_deref();
    let messages_history = session.messages_history.as_deref();

    if try_jinja {
        let template = m.chat_template.as_ref().unwrap();
        let frame = prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking: session.max_think_tokens != 1,
            bos_token: None,
        };
        let render_result = if tools.is_some() || messages_history.is_some() {
            let synthesized: Vec<prompt_frame::Message>;
            let messages_slice: &[prompt_frame::Message] = match messages_history {
                Some(m) => m,
                None => {
                    let mut v = Vec::new();
                    if let Some(sys) = system_prompt {
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::System,
                            content: sys.to_string(),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                        });
                    }
                    v.push(prompt_frame::Message {
                        role: prompt_frame::Role::User,
                        content: prompt.to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
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
            Ok(rendered) => Ok(tokenizer.encode(&rendered)),
            Err(e) => {
                eprintln!(
                    "[daemon] batch-prefill jinja render failed ({e}) -- falling back to Plain"
                );
                Ok(prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: "",
                    assistant_prefix,
                    raw: effective_raw(m),
                }
                .build_with_user_tokens(&raw_q_tokens))
            }
        }
    } else {
        Ok(prompt_frame::ChatFrame {
            tokenizer,
            system: if seq_pos_for_prompt == 0 {
                system_prompt
            } else {
                None
            },
            user: "",
            assistant_prefix,
            raw: effective_raw(m),
        }
        .build_with_user_tokens(&raw_q_tokens))
    }
}

/// Compute the prefix-hash candidates for a session's prompt (one per semantic
/// boundary), used to match against cached prefixes for KV reuse.
pub fn qwen35_prefix_hash_candidates(
    m: &LoadedModel,
    session: &GenerateBatchPrefillSession,
) -> Result<Vec<PrefixHashPreflightCandidate>, String> {
    let full_tokens = qwen35_materialize_batch_prefill_prompt(m, session)?;
    qwen35_prefix_hash_candidates_for_tokens(m, session, &full_tokens)
}

/// [`qwen35_prefix_hash_candidates`] over an explicit token slice (the
/// tokenizer-free core, shared by the request and preflight paths).
pub fn qwen35_prefix_hash_candidates_for_tokens(
    m: &LoadedModel,
    session: &GenerateBatchPrefillSession,
    full_tokens: &[u32],
) -> Result<Vec<PrefixHashPreflightCandidate>, String> {
    let tokenizer = m
        .tokenizer
        .as_ref()
        .ok_or_else(|| "tokenizer not loaded".to_string())?;
    let full_hash = compute_qwen35_prefix_hash(
        m.arch_id,
        m.q35_kv_mode.as_deref(),
        &session.state_handle.state_kinds,
        &session.assistant_prefix,
        session.max_think_tokens,
        full_tokens,
    );
    let mut candidates = Vec::new();
    let boundary_tokens: Vec<(&str, Vec<u32>)> = [
        ("message_end", "<|im_end|>"),
        ("vision_end", "<|vision_end|>"),
        ("tool_end", "<|tool_call_end|>"),
        ("tool_response_end", "<|tool_response_end|>"),
    ]
    .into_iter()
    .filter_map(|(boundary, marker)| {
        let marker_tokens = tokenizer
            .special_token_id(marker)
            .map(|id| vec![id])
            .unwrap_or_else(|| tokenizer.encode(marker));
        if marker_tokens.is_empty() {
            None
        } else {
            Some((boundary, marker_tokens))
        }
    })
    .collect();
    let mut boundary_index = 0usize;
    let mut push_boundary_candidate = |candidates: &mut Vec<PrefixHashPreflightCandidate>,
                                       prefix_len: usize,
                                       boundary: &str| {
        if prefix_len == 0 || prefix_len >= full_tokens.len() {
            return;
        }
        let hash = compute_qwen35_prefix_hash(
            m.arch_id,
            m.q35_kv_mode.as_deref(),
            &session.state_handle.state_kinds,
            &session.assistant_prefix,
            session.max_think_tokens,
            &full_tokens[..prefix_len],
        );
        if !candidates
            .iter()
            .any(|candidate: &PrefixHashPreflightCandidate| {
                candidate.hash.prefix_len == hash.prefix_len && candidate.hash.value == hash.value
            })
        {
            candidates.push(PrefixHashPreflightCandidate {
                hash,
                boundary: boundary.to_string(),
                boundary_index,
                checkpoint_id: None,
            });
            boundary_index += 1;
        }
    };
    for (idx, _) in full_tokens.iter().enumerate() {
        let prefix_len = idx + 1;
        if prefix_len >= full_tokens.len() {
            continue;
        }
        let Some((boundary, _)) = boundary_tokens.iter().find(|(_, marker_tokens)| {
            prefix_len >= marker_tokens.len()
                && full_tokens[prefix_len - marker_tokens.len()..prefix_len] == marker_tokens[..]
        }) else {
            continue;
        };
        push_boundary_candidate(&mut candidates, prefix_len, boundary);
    }

    let assistant_start: Vec<u32> = [
        tokenizer.encode("<|im_start|>"),
        tokenizer.encode("assistant"),
        tokenizer.encode("\n"),
    ]
    .into_iter()
    .flatten()
    .collect();
    if !assistant_start.is_empty() && full_tokens.len() > assistant_start.len() {
        for idx in 0..=full_tokens.len() - assistant_start.len() {
            if full_tokens[idx..idx + assistant_start.len()] == assistant_start[..] {
                push_boundary_candidate(&mut candidates, idx, "assistant_turn_start");
            }
        }
    }
    candidates.push(PrefixHashPreflightCandidate {
        hash: full_hash,
        boundary: "full".to_string(),
        boundary_index: candidates.len(),
        checkpoint_id: None,
    });
    candidates.sort_by_key(|candidate| candidate.hash.prefix_len);
    Ok(candidates)
}

/// Compute the semantic-boundary token offsets (turn / message boundaries) at
/// which prefill emits checkpoints and prefix hashes are anchored.
pub fn qwen35_semantic_boundary_checkpoints(
    m: &LoadedModel,
    session: &GenerateBatchPrefillSession,
    full_tokens: &[u32],
) -> Result<Vec<Qwen35SemanticBoundaryCheckpoint>, String> {
    if !session.semantic_boundary_checkpoints {
        return Ok(Vec::new());
    }
    if matches!(
        std::env::var("HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS")
            .ok()
            .as_deref(),
        Some("0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO")
    ) {
        return Ok(Vec::new());
    }
    let candidates = qwen35_prefix_hash_candidates_for_tokens(m, session, full_tokens)?;
    if std::env::var_os("HIPFIRE_DEBUG_PREFIX_BOUNDARIES").is_some() {
        eprintln!(
            "[daemon] prefix boundary candidates session={} tokens={} candidates={}",
            session.id,
            full_tokens.len(),
            candidates.len()
        );
        for candidate in &candidates {
            eprintln!(
                "[daemon] prefix boundary candidate session={} boundary={} index={} len={} hash={}",
                session.id,
                candidate.boundary,
                candidate.boundary_index,
                candidate.hash.prefix_len,
                candidate.hash.value
            );
        }
    }
    Ok(candidates
        .into_iter()
        .filter(|candidate| candidate.boundary != "full")
        .filter(|candidate| candidate.hash.prefix_len > 0)
        .filter(|candidate| candidate.hash.prefix_len < full_tokens.len())
        .map(|candidate| Qwen35SemanticBoundaryCheckpoint {
            checkpoint_id: None,
            prefix_len: candidate.hash.prefix_len,
            hash: candidate.hash,
            boundary: candidate.boundary,
            boundary_index: candidate.boundary_index,
        })
        .collect())
}

/// Handle a `prefix_hash_preflight` request: compute the candidate hashes for a
/// prompt and report which prefix lengths the client can reuse, before any GPU
/// prefill work.
pub fn run_prefix_hash_preflight_qwen35(
    m: &LoadedModel,
    stdout: &mut std::io::Stdout,
    envelope: &PrefixHashPreflightEnvelope,
) -> Result<(), String> {
    if !is_qwen35_family_arch_id(m.arch_id) {
        return Err(format!(
            "prefix_hash_preflight currently supports qwen35/qwen35-moe only (arch_id={})",
            m.arch_id
        ));
    }
    if envelope.boundary_policy != "semantic_chat_template" {
        return Err(
            "prefix_hash_preflight.boundary_policy must be semantic_chat_template".to_string(),
        );
    }
    let candidates = qwen35_prefix_hash_candidates(m, &envelope.session)?;
    let line =
        prefix_hash_preflight_done_json(&envelope.id, &envelope.boundary_policy, &candidates)?;
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
    Ok(())
}

/// Batched suffix prefill dispatcher: pick the fused-dense or grouped-MoE kernel
/// for the loaded arch (falling back to the serial reference) and run it over
/// the prompt suffix.
pub fn qwen35_prefill_suffix_batch(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    batch_id: &str,
    prepared: &[Qwen35PreparedPrefillSession],
    plan: GenerateBatchPrefillPlan,
    backend: Qwen35PrefillBatchBackend,
) -> Result<Qwen35PrefillBatchResult, String> {
    let (attach_only, non_empty): (Vec<_>, Vec<_>) = prepared
        .iter()
        .partition(|session| session.tokens.is_empty());
    if !attach_only.is_empty() {
        let effective_backend = if non_empty.len() < 2 {
            Qwen35PrefillBatchBackend::SerialReference
        } else {
            backend
        };
        let mut sessions_by_id: HashMap<String, Qwen35PrefillSessionResult> = HashMap::new();
        let mut total_prefill_tokens = 0usize;
        let mut mode = match effective_backend {
            Qwen35PrefillBatchBackend::SerialReference => "serial_prefill",
            Qwen35PrefillBatchBackend::FusedDense => "qwen35_fused_dense_prefill",
            Qwen35PrefillBatchBackend::FusedGroupedMoe => "qwen35_fused_grouped_moe_prefill",
        };
        if !non_empty.is_empty() {
            let non_empty_prepared: Vec<Qwen35PreparedPrefillSession> =
                non_empty.into_iter().cloned().collect();
            let result = qwen35_prefill_suffix_batch(
                m,
                gpu,
                batch_id,
                &non_empty_prepared,
                plan,
                effective_backend,
            )?;
            total_prefill_tokens += result.total_prefill_tokens;
            mode = result.mode;
            for session in result.sessions {
                sessions_by_id.insert(session.id.clone(), session);
            }
        }
        for session in attach_only {
            qwen35_activate_session(m, gpu, &session.id)?;
            qwen35_save_active_session(m, gpu)?;
            let saved = m.q35_sessions.get(&session.id).ok_or_else(|| {
                format!(
                    "qwen35 attach-only session {} missing after activation",
                    session.id
                )
            })?;
            let logical_position = saved.seq_pos + saved.kv_cache.compact_offset;
            let prefix_hash = compute_qwen35_prefix_hash(
                m.arch_id,
                m.q35_kv_mode.as_deref(),
                &session.state_kinds,
                &session.assistant_prefix,
                session.max_think_tokens,
                &saved.conversation_tokens,
            );
            if let Some(saved) = m.q35_sessions.get_mut(&session.id) {
                saved.prefix_hash = Some(prefix_hash.clone());
            }
            sessions_by_id.insert(
                session.id.clone(),
                Qwen35PrefillSessionResult {
                    id: session.id.clone(),
                    prefill_tokens: 0,
                    logical_position,
                    cached_prefix_tokens: session.cached_prefix_tokens,
                    prefix_hash,
                    debug_sample_token: None,
                    boundary_checkpoints: Vec::new(),
                },
            );
        }
        let sessions = prepared
            .iter()
            .map(|session| {
                sessions_by_id
                    .remove(&session.id)
                    .ok_or_else(|| format!("qwen35 prefill result missing session {}", session.id))
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Qwen35PrefillBatchResult {
            mode,
            plan,
            backend: effective_backend,
            total_prefill_tokens,
            sessions,
        });
    }

    if matches!(
        backend,
        Qwen35PrefillBatchBackend::FusedDense | Qwen35PrefillBatchBackend::FusedGroupedMoe
    ) {
        if let Err(err) = qwen35_fused_prefill_boundary_cuts(prepared) {
            if std::env::var_os("HIPFIRE_DEBUG_PREFIX_BOUNDARIES").is_some() {
                eprintln!("[daemon] fused prefill boundary checkpoint fallback: {err}");
            }
            return qwen35_prefill_suffix_batch_serial_reference(
                m,
                gpu,
                batch_id,
                prepared,
                plan,
                Qwen35PrefillBatchBackend::SerialReference,
            );
        }
    }

    match backend {
        Qwen35PrefillBatchBackend::SerialReference => {
            qwen35_prefill_suffix_batch_serial_reference(m, gpu, batch_id, prepared, plan, backend)
        }
        Qwen35PrefillBatchBackend::FusedDense => {
            qwen35_prefill_suffix_batch_fused_dense(m, gpu, batch_id, prepared, plan, backend)
        }
        Qwen35PrefillBatchBackend::FusedGroupedMoe => {
            qwen35_prefill_suffix_batch_fused_grouped_moe(m, gpu, batch_id, prepared, plan, backend)
        }
    }
}

/// Checkpoint-emit variant for the owned-session serial segment path (mirrors
/// [`emit_qwen35_prefill_checkpoint`] for serially-prefilled owned sessions).
pub fn emit_qwen35_owned_prefill_checkpoint(
    sessions: &mut HashMap<String, Qwen35RequestSessionState>,
    gpu: &mut rdna_compute::Gpu,
    hook: Qwen35PrefillCheckpointHook<'_>,
    source: &mut Qwen35RequestSessionState,
) -> Result<String, String> {
    if qwen35_prefill_checkpoint_boundary_kind(hook).is_empty() {
        return Err("qwen35 prefill checkpoint boundary kind is empty".to_string());
    }
    let logical_position = source.seq_pos + source.kv_cache.compact_offset;
    if logical_position != hook.logical_position {
        return Err(format!(
            "qwen35 owned prefill checkpoint source {} logical_position mismatch: expected={} resident={}",
            hook.source_state_handle, hook.logical_position, logical_position
        ));
    }
    source.prefix_hash = Some(hook.prefix_hash.clone());
    let checkpoint_id = qwen35_prefill_checkpoint_session_id(hook);
    let checkpoint = Qwen35RequestSessionState::fork_from(gpu, source)?;
    sessions.insert(checkpoint_id.clone(), checkpoint);
    Ok(checkpoint_id)
}

/// Grouped-MoE batched suffix prefill: the fused kernel path for Qwen3.5 MoE
/// (arch_id 6), prefilling the whole suffix in batched layer passes with the
/// expert-grouped FFN.
pub fn qwen35_prefill_suffix_batch_fused_grouped_moe(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    batch_id: &str,
    prepared: &[Qwen35PreparedPrefillSession],
    plan: GenerateBatchPrefillPlan,
    backend: Qwen35PrefillBatchBackend,
) -> Result<Qwen35PrefillBatchResult, String> {
    if plan != GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate {
        return Err(format!(
            "qwen35 grouped-MoE fused prefill-session batch worker requires plan={}, got {}",
            GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate.as_str(),
            plan.as_str()
        ));
    }
    validate_qwen35_fused_grouped_moe_prefill_batch_preflight(prepared, plan)?;
    qwen35_save_active_session(m, gpu)?;
    let config = m
        .q35_config
        .as_ref()
        .ok_or_else(|| "qwen35 config missing".to_string())?;
    if config.num_experts == 0 || !config.has_shared_expert {
        return Err(
            "qwen35 grouped-MoE fused prefill-session batch requires MoE/A3B weights".to_string(),
        );
    }
    let weights = m
        .q35_weights
        .as_ref()
        .ok_or_else(|| "qwen35 weights missing".to_string())?;
    let boundary_cuts = qwen35_fused_prefill_boundary_cuts(prepared)?;
    let mut owned_sessions: Vec<(String, Qwen35RequestSessionState)> =
        Vec::with_capacity(prepared.len());
    for spec in prepared {
        let state = match m.q35_sessions.remove(&spec.id) {
            Some(state) => state,
            None => match qwen35_allocate_session_state(m, gpu) {
                Ok(state) => state,
                Err(e) => {
                    for (restore_id, restore_state) in owned_sessions {
                        m.q35_sessions.insert(restore_id, restore_state);
                    }
                    return Err(e);
                }
            },
        };
        if state.seq_pos + spec.tokens.len() > m.physical_cap {
            let id = spec.id.to_string();
            let seq_pos = state.seq_pos;
            m.q35_sessions.insert(id.clone(), state);
            for (restore_id, restore_state) in owned_sessions {
                m.q35_sessions.insert(restore_id, restore_state);
            }
            return Err(format!(
                "generate_batch_prefill exceeds loaded KV budget for session {}: seq_pos={} + prefill={} > physical_cap={}",
                id,
                seq_pos,
                spec.tokens.len(),
                m.physical_cap
            ));
        }
        owned_sessions.push((spec.id.to_string(), state));
    }

    if let Some(boundary_cuts) = boundary_cuts {
        let total_tokens = prepared.iter().map(|spec| spec.tokens.len()).sum::<usize>();
        let mut progress = vec![0usize; prepared.len()];
        let mut boundary_checkpoints_by_session = vec![Vec::new(); prepared.len()];
        let mut shape_total_tokens = 0usize;
        for &cut in &boundary_cuts {
            let active_indices: Vec<usize> = prepared
                .iter()
                .enumerate()
                .filter_map(|(idx, spec)| {
                    let end = spec.tokens.len().min(cut);
                    (progress[idx] < end).then_some(idx)
                })
                .collect();
            if active_indices.len() < 2 {
                let scratch = m.q35_scratch.as_ref().ok_or_else(|| {
                    "qwen35 scratch missing; grouped-MoE serial boundary segment is pp=1 only"
                        .to_string()
                })?;
                for &idx in &active_indices {
                    let start = progress[idx];
                    let end = prepared[idx].tokens.len().min(cut);
                    let state = &mut owned_sessions[idx].1;
                    let segment_tokens = match qwen35_prefill_owned_session_serial_segment(
                        gpu,
                        weights,
                        config,
                        scratch,
                        state,
                        &prepared[idx].tokens[start..end],
                    ) {
                        Ok(tokens) => tokens,
                        Err(err) => {
                            for (id, state) in owned_sessions {
                                m.q35_sessions.insert(id, state);
                            }
                            return Err(err);
                        }
                    };
                    shape_total_tokens += segment_tokens;
                    progress[idx] = end;
                    for mut boundary in prepared[idx]
                        .boundary_checkpoints
                        .iter()
                        .filter(|boundary| boundary.prefix_len == end)
                        .cloned()
                    {
                        let hook = Qwen35PrefillCheckpointHook {
                            batch_id,
                            session_id: &prepared[idx].id,
                            source_state_handle: &prepared[idx].id,
                            logical_position: end,
                            kind: Qwen35PrefillCheckpointKind::SemanticBoundary {
                                boundary: &boundary.boundary,
                                boundary_index: boundary.boundary_index,
                            },
                            prefix_hash: &boundary.hash,
                        };
                        let checkpoint_id_for_error = qwen35_prefill_checkpoint_session_id(hook);
                        let checkpoint_id = emit_qwen35_owned_prefill_checkpoint(
                            &mut m.q35_sessions,
                            gpu,
                            hook,
                            state,
                        )
                        .map_err(|e| {
                            format!(
                                "qwen35 session {} failed to create fused semantic boundary checkpoint {}: {}",
                                prepared[idx].id, checkpoint_id_for_error, e
                            )
                        })?;
                        boundary.checkpoint_id = Some(checkpoint_id);
                        boundary_checkpoints_by_session[idx].push(boundary);
                    }
                }
                continue;
            }
            let worker_result = {
                let scratch = m.q35_scratch.as_mut().ok_or_else(|| {
                    "qwen35 scratch missing; grouped-MoE fused prefill is pp=1 only".to_string()
                })?;
                let scratch_target_batch = qwen35_prefill_scratch_target_batch(
                    config.paged_experts,
                    total_tokens,
                    std::env::var("HIPFIRE_PREFILL_MAX_BATCH").ok().as_deref(),
                    qwen35::PREFILL_MAX_BATCH,
                );
                let needs_scratch = scratch
                    .prefill_batch
                    .as_ref()
                    .map(|pbs| pbs.max_batch < scratch_target_batch)
                    .unwrap_or(true);
                if needs_scratch {
                    if let Some(existing) = scratch.prefill_batch.take() {
                        existing.free_gpu(gpu);
                    }
                    scratch.prefill_batch = Some(
                        qwen35::PrefillBatchScratch::new(gpu, config, scratch_target_batch)
                            .map_err(|e| {
                                format!("allocate qwen35 grouped-MoE fused prefill scratch: {e:?}")
                            })?,
                    );
                }
                let pbs_max_batch = scratch.prefill_batch.as_ref().unwrap().max_batch;
                if pbs_max_batch < total_tokens {
                    return Err(format!(
                        "qwen35 grouped-MoE fused prefill scratch max_batch={pbs_max_batch} is smaller than required fused rows {}; increase HIPFIRE_PREFILL_MAX_BATCH or restart the daemon",
                        total_tokens,
                    ));
                }
                let pbs = scratch.prefill_batch.as_ref().unwrap();
                let mut rows: Vec<qwen35::DensePrefillSessionBatchRow<'_>> = owned_sessions
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(idx, (_, state))| {
                        let end = prepared[idx].tokens.len().min(cut);
                        (progress[idx] < end).then(|| qwen35::DensePrefillSessionBatchRow {
                            tokens: &prepared[idx].tokens[progress[idx]..end],
                            start_pos: state.seq_pos,
                            kv_cache: &mut state.kv_cache,
                            dn_state: &mut state.dn_state,
                            logits: &state.logits,
                        })
                    })
                    .collect();
                qwen35::forward_prefill_grouped_moe_session_batch(
                    gpu, weights, config, &mut rows, scratch, pbs,
                )
            };
            let shape = match worker_result {
                Ok(shape) => shape,
                Err(e) => {
                    for (id, state) in owned_sessions {
                        m.q35_sessions.insert(id, state);
                    }
                    return Err(format!(
                        "qwen35 grouped-MoE fused boundary prefill-session batch backend failed: {e:?}; \
                         use HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=auto or serial"
                    ));
                }
            };
            shape_total_tokens += shape.total_tokens;
            for idx in active_indices {
                let start = progress[idx];
                let end = prepared[idx].tokens.len().min(cut);
                let state = &mut owned_sessions[idx].1;
                state.seq_pos += end - start;
                state
                    .conversation_tokens
                    .extend_from_slice(&prepared[idx].tokens[start..end]);
                progress[idx] = end;
                for mut boundary in prepared[idx]
                    .boundary_checkpoints
                    .iter()
                    .filter(|boundary| boundary.prefix_len == end)
                    .cloned()
                {
                    let hook = Qwen35PrefillCheckpointHook {
                        batch_id,
                        session_id: &prepared[idx].id,
                        source_state_handle: &prepared[idx].id,
                        logical_position: end,
                        kind: Qwen35PrefillCheckpointKind::SemanticBoundary {
                            boundary: &boundary.boundary,
                            boundary_index: boundary.boundary_index,
                        },
                        prefix_hash: &boundary.hash,
                    };
                    let checkpoint_id_for_error = qwen35_prefill_checkpoint_session_id(hook);
                    let checkpoint_id =
                        emit_qwen35_owned_prefill_checkpoint(&mut m.q35_sessions, gpu, hook, state).map_err(|e| {
                            format!(
                                "qwen35 session {} failed to create fused semantic boundary checkpoint {}: {}",
                                prepared[idx].id, checkpoint_id_for_error, e
                            )
                        })?;
                    boundary.checkpoint_id = Some(checkpoint_id);
                    boundary_checkpoints_by_session[idx].push(boundary);
                }
            }
        }
        let mut sessions = Vec::with_capacity(owned_sessions.len());
        for (idx, (id, mut state)) in owned_sessions.into_iter().enumerate() {
            state.prefilled_generated_suffix_len = 0;
            let logical_position = state.seq_pos + state.kv_cache.compact_offset;
            let prefix_hash = compute_qwen35_prefix_hash(
                m.arch_id,
                m.q35_kv_mode.as_deref(),
                &prepared[idx].state_kinds,
                &prepared[idx].assistant_prefix,
                prepared[idx].max_think_tokens,
                &state.conversation_tokens,
            );
            state.prefix_hash = Some(prefix_hash.clone());
            sessions.push(Qwen35PrefillSessionResult {
                id: id.clone(),
                prefill_tokens: prepared[idx].tokens.len(),
                logical_position,
                cached_prefix_tokens: prepared[idx].cached_prefix_tokens,
                prefix_hash,
                debug_sample_token: None,
                boundary_checkpoints: std::mem::take(&mut boundary_checkpoints_by_session[idx]),
            });
            m.q35_sessions.insert(id, state);
        }
        return Ok(Qwen35PrefillBatchResult {
            mode: "qwen35_fused_grouped_moe_prefill_boundary_chunked",
            plan,
            backend,
            total_prefill_tokens: shape_total_tokens,
            sessions,
        });
    }

    let worker_result = {
        let scratch = m.q35_scratch.as_mut().ok_or_else(|| {
            "qwen35 scratch missing; grouped-MoE fused prefill is pp=1 only".to_string()
        })?;
        let total_tokens = prepared.iter().map(|spec| spec.tokens.len()).sum::<usize>();
        let scratch_target_batch = qwen35_prefill_scratch_target_batch(
            config.paged_experts,
            total_tokens,
            std::env::var("HIPFIRE_PREFILL_MAX_BATCH").ok().as_deref(),
            qwen35::PREFILL_MAX_BATCH,
        );
        let needs_scratch = scratch
            .prefill_batch
            .as_ref()
            .map(|pbs| pbs.max_batch < scratch_target_batch)
            .unwrap_or(true);
        if needs_scratch {
            if let Some(existing) = scratch.prefill_batch.take() {
                existing.free_gpu(gpu);
            }
            scratch.prefill_batch = Some(
                qwen35::PrefillBatchScratch::new(gpu, config, scratch_target_batch).map_err(
                    |e| format!("allocate qwen35 grouped-MoE fused prefill scratch: {e:?}"),
                )?,
            );
        }
        let pbs_max_batch = scratch.prefill_batch.as_ref().unwrap().max_batch;
        if pbs_max_batch < total_tokens {
            return Err(format!(
                "qwen35 grouped-MoE fused prefill scratch max_batch={pbs_max_batch} is smaller than required fused rows {}; increase HIPFIRE_PREFILL_MAX_BATCH or restart the daemon",
                total_tokens,
            ));
        }
        let pbs = scratch.prefill_batch.as_ref().unwrap();
        let mut rows: Vec<qwen35::DensePrefillSessionBatchRow<'_>> = owned_sessions
            .iter_mut()
            .zip(prepared.iter())
            .map(|((_, state), spec)| qwen35::DensePrefillSessionBatchRow {
                tokens: &spec.tokens,
                start_pos: state.seq_pos,
                kv_cache: &mut state.kv_cache,
                dn_state: &mut state.dn_state,
                logits: &state.logits,
            })
            .collect();
        qwen35::forward_prefill_grouped_moe_session_batch(
            gpu, weights, config, &mut rows, scratch, pbs,
        )
    };

    let shape = match worker_result {
        Ok(shape) => shape,
        Err(e) => {
            for (id, state) in owned_sessions {
                m.q35_sessions.insert(id, state);
            }
            return Err(format!(
                "qwen35 grouped-MoE fused prefill-session batch backend failed: {e:?}; \
                 use HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=auto or serial"
            ));
        }
    };

    let mut sessions = Vec::with_capacity(owned_sessions.len());
    for ((id, mut state), spec) in owned_sessions.into_iter().zip(prepared.iter()) {
        state.seq_pos += spec.tokens.len();
        state.conversation_tokens.extend_from_slice(&spec.tokens);
        state.prefilled_generated_suffix_len = if spec.replay_as_generated_suffix {
            spec.tokens.len()
        } else {
            0
        };
        let logical_position = state.seq_pos + state.kv_cache.compact_offset;
        let debug_sample_token = if spec.replay_as_generated_suffix
            && std::env::var_os("HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE").is_some()
        {
            let scratch = m.q35_scratch.as_ref().ok_or_else(|| {
                "qwen35 scratch missing; fused grouped-MoE debug sampling unavailable".to_string()
            })?;
            let mut rng_state = 0x13579BDFu32;
            let cfg = SamplerConfig {
                temperature: 0.0,
                top_p: 1.0,
                repeat_window: 0,
                repeat_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                blocked_tokens: Vec::new(),
            };
            Some(sampler::sample(
                gpu,
                &state.logits,
                &scratch.sample_buf,
                &scratch.repeat_buf,
                config.vocab_size,
                &spec.tokens,
                &cfg,
                &mut rng_state,
            ))
        } else {
            None
        };
        let prefix_hash = compute_qwen35_prefix_hash(
            m.arch_id,
            m.q35_kv_mode.as_deref(),
            &spec.state_kinds,
            &spec.assistant_prefix,
            spec.max_think_tokens,
            &state.conversation_tokens,
        );
        state.prefix_hash = Some(prefix_hash.clone());
        sessions.push(Qwen35PrefillSessionResult {
            id: id.clone(),
            prefill_tokens: spec.tokens.len(),
            logical_position,
            cached_prefix_tokens: spec.cached_prefix_tokens,
            prefix_hash,
            debug_sample_token,
            boundary_checkpoints: Vec::new(),
        });
        m.q35_sessions.insert(id, state);
    }

    Ok(Qwen35PrefillBatchResult {
        mode: if prepared[0].replay_as_generated_suffix {
            "qwen35_fused_grouped_moe_suffix_replay"
        } else {
            "qwen35_fused_grouped_moe_prefill"
        },
        plan,
        backend,
        total_prefill_tokens: shape.total_tokens,
        sessions,
    })
}

/// Fused-dense batched suffix prefill: the fused kernel path for dense Qwen3.5
/// (arch_id 5), prefilling the suffix in batched layer passes.
pub fn qwen35_prefill_suffix_batch_fused_dense(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    batch_id: &str,
    prepared: &[Qwen35PreparedPrefillSession],
    plan: GenerateBatchPrefillPlan,
    backend: Qwen35PrefillBatchBackend,
) -> Result<Qwen35PrefillBatchResult, String> {
    let contract = build_qwen35_fused_dense_prefill_batch_contract(prepared, plan)?;

    // Worker API seam for the real dense implementation:
    //
    //   prefill_suffix_batch(&mut [&mut RequestSession])
    //
    // The serial-reference worker below owns the correctness oracle: every
    // session has isolated KV, DeltaNet recurrent state, conversation tokens,
    // and logits. The fused worker must preserve the same ownership contract.
    //
    // Do not call qwen35::forward_prefill_batch over concatenated session
    // tokens here. That function batches rows inside ONE causal sequence and
    // ONE DeltaNetState; using it across independent request sessions would
    // leak KV/DN state and produce numerically plausible but wrong continuations.
    //
    // The next implementation step is an arch-level dense-Qwen35 session batch
    // worker that accepts per-session KV/DN/logits handles and writes one
    // independent result row per session.
    qwen35_save_active_session(m, gpu)?;
    let config = m
        .q35_config
        .as_ref()
        .ok_or_else(|| "qwen35 config missing".to_string())?;
    let weights = m
        .q35_weights
        .as_ref()
        .ok_or_else(|| "qwen35 weights missing".to_string())?;
    let boundary_cuts = qwen35_fused_prefill_boundary_cuts(prepared)?;
    let mut owned_sessions: Vec<(String, Qwen35RequestSessionState)> =
        Vec::with_capacity(contract.sessions.len());
    for spec in &contract.sessions {
        let state = match m.q35_sessions.remove(spec.id) {
            Some(state) => state,
            None => match qwen35_allocate_session_state(m, gpu) {
                Ok(state) => state,
                Err(e) => {
                    for (restore_id, restore_state) in owned_sessions {
                        m.q35_sessions.insert(restore_id, restore_state);
                    }
                    return Err(e);
                }
            },
        };
        if state.seq_pos + spec.tokens.len() > m.physical_cap {
            let id = spec.id.to_string();
            let seq_pos = state.seq_pos;
            m.q35_sessions.insert(id.clone(), state);
            for (restore_id, restore_state) in owned_sessions {
                m.q35_sessions.insert(restore_id, restore_state);
            }
            return Err(format!(
                "generate_batch_prefill exceeds loaded KV budget for session {}: seq_pos={} + prefill={} > physical_cap={}",
                id,
                seq_pos,
                spec.tokens.len(),
                m.physical_cap
            ));
        }
        owned_sessions.push((spec.id.to_string(), state));
    }

    if let Some(boundary_cuts) = boundary_cuts {
        let mut progress = vec![0usize; contract.sessions.len()];
        let mut boundary_checkpoints_by_session = vec![Vec::new(); contract.sessions.len()];
        let mut shape_total_tokens = 0usize;
        for &cut in &boundary_cuts {
            let active_indices: Vec<usize> = contract
                .sessions
                .iter()
                .enumerate()
                .filter_map(|(idx, spec)| {
                    let end = spec.tokens.len().min(cut);
                    (progress[idx] < end).then_some(idx)
                })
                .collect();
            if active_indices.len() < 2 {
                let scratch = m.q35_scratch.as_ref().ok_or_else(|| {
                    "qwen35 scratch missing; fused dense serial boundary segment is pp=1 only"
                        .to_string()
                })?;
                for &idx in &active_indices {
                    let start = progress[idx];
                    let end = contract.sessions[idx].tokens.len().min(cut);
                    let state = &mut owned_sessions[idx].1;
                    let segment_tokens = match qwen35_prefill_owned_session_serial_segment(
                        gpu,
                        weights,
                        config,
                        scratch,
                        state,
                        &contract.sessions[idx].tokens[start..end],
                    ) {
                        Ok(tokens) => tokens,
                        Err(err) => {
                            for (id, state) in owned_sessions {
                                m.q35_sessions.insert(id, state);
                            }
                            return Err(err);
                        }
                    };
                    shape_total_tokens += segment_tokens;
                    progress[idx] = end;
                    for mut boundary in prepared[idx]
                        .boundary_checkpoints
                        .iter()
                        .filter(|boundary| boundary.prefix_len == end)
                        .cloned()
                    {
                        let hook = Qwen35PrefillCheckpointHook {
                            batch_id,
                            session_id: contract.sessions[idx].id,
                            source_state_handle: contract.sessions[idx].id,
                            logical_position: end,
                            kind: Qwen35PrefillCheckpointKind::SemanticBoundary {
                                boundary: &boundary.boundary,
                                boundary_index: boundary.boundary_index,
                            },
                            prefix_hash: &boundary.hash,
                        };
                        let checkpoint_id_for_error = qwen35_prefill_checkpoint_session_id(hook);
                        let checkpoint_id = emit_qwen35_owned_prefill_checkpoint(
                            &mut m.q35_sessions,
                            gpu,
                            hook,
                            state,
                        )
                        .map_err(|e| {
                            format!(
                                "qwen35 session {} failed to create fused semantic boundary checkpoint {}: {}",
                                contract.sessions[idx].id, checkpoint_id_for_error, e
                            )
                        })?;
                        boundary.checkpoint_id = Some(checkpoint_id);
                        boundary_checkpoints_by_session[idx].push(boundary);
                    }
                }
                continue;
            }
            let worker_result = {
                let scratch = m.q35_scratch.as_mut().ok_or_else(|| {
                    "qwen35 scratch missing; fused dense prefill is pp=1 only".to_string()
                })?;
                let needs_scratch = scratch
                    .prefill_batch
                    .as_ref()
                    .map(|pbs| pbs.max_batch < contract.total_tokens)
                    .unwrap_or(true);
                if needs_scratch {
                    if let Some(existing) = scratch.prefill_batch.take() {
                        existing.free_gpu(gpu);
                    }
                    let max_batch = std::env::var("HIPFIRE_PREFILL_MAX_BATCH")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .filter(|&v| v >= 2)
                        .unwrap_or(qwen35::PREFILL_MAX_BATCH)
                        .max(contract.total_tokens);
                    scratch.prefill_batch = Some(
                        qwen35::PrefillBatchScratch::new(gpu, config, max_batch).map_err(|e| {
                            format!("allocate qwen35 fused dense prefill scratch: {e:?}")
                        })?,
                    );
                }
                let pbs_max_batch = scratch.prefill_batch.as_ref().unwrap().max_batch;
                if pbs_max_batch < contract.total_tokens {
                    return Err(format!(
                        "qwen35 fused dense prefill scratch max_batch={pbs_max_batch} is smaller than required fused rows {}; increase HIPFIRE_PREFILL_MAX_BATCH or restart the daemon",
                        contract.total_tokens,
                    ));
                }
                let pbs = scratch.prefill_batch.as_ref().unwrap();
                let mut rows: Vec<qwen35::DensePrefillSessionBatchRow<'_>> = owned_sessions
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(idx, (_, state))| {
                        let end = contract.sessions[idx].tokens.len().min(cut);
                        (progress[idx] < end).then(|| qwen35::DensePrefillSessionBatchRow {
                            tokens: &contract.sessions[idx].tokens[progress[idx]..end],
                            start_pos: state.seq_pos,
                            kv_cache: &mut state.kv_cache,
                            dn_state: &mut state.dn_state,
                            logits: &state.logits,
                        })
                    })
                    .collect();
                qwen35::forward_prefill_dense_session_batch(
                    gpu, weights, config, &mut rows, scratch, pbs,
                )
            };
            let shape = match worker_result {
                Ok(shape) => shape,
                Err(e) => {
                    for (id, state) in owned_sessions {
                        m.q35_sessions.insert(id, state);
                    }
                    return Err(format!(
                        "qwen35 fused dense boundary prefill-session batch backend failed: {e:?}; \
                         use HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=auto or serial"
                    ));
                }
            };
            shape_total_tokens += shape.total_tokens;
            for idx in active_indices {
                let start = progress[idx];
                let end = contract.sessions[idx].tokens.len().min(cut);
                let state = &mut owned_sessions[idx].1;
                state.seq_pos += end - start;
                state
                    .conversation_tokens
                    .extend_from_slice(&contract.sessions[idx].tokens[start..end]);
                progress[idx] = end;
                for mut boundary in prepared[idx]
                    .boundary_checkpoints
                    .iter()
                    .filter(|boundary| boundary.prefix_len == end)
                    .cloned()
                {
                    let hook = Qwen35PrefillCheckpointHook {
                        batch_id,
                        session_id: contract.sessions[idx].id,
                        source_state_handle: contract.sessions[idx].id,
                        logical_position: end,
                        kind: Qwen35PrefillCheckpointKind::SemanticBoundary {
                            boundary: &boundary.boundary,
                            boundary_index: boundary.boundary_index,
                        },
                        prefix_hash: &boundary.hash,
                    };
                    let checkpoint_id_for_error = qwen35_prefill_checkpoint_session_id(hook);
                    let checkpoint_id =
                        emit_qwen35_owned_prefill_checkpoint(&mut m.q35_sessions, gpu, hook, state).map_err(|e| {
                            format!(
                                "qwen35 session {} failed to create fused semantic boundary checkpoint {}: {}",
                                contract.sessions[idx].id, checkpoint_id_for_error, e
                            )
                        })?;
                    boundary.checkpoint_id = Some(checkpoint_id);
                    boundary_checkpoints_by_session[idx].push(boundary);
                }
            }
        }
        let mut sessions = Vec::with_capacity(owned_sessions.len());
        for (idx, (id, mut state)) in owned_sessions.into_iter().enumerate() {
            state.prefilled_generated_suffix_len = 0;
            let logical_position = state.seq_pos + state.kv_cache.compact_offset;
            let prefix_hash = compute_qwen35_prefix_hash(
                m.arch_id,
                m.q35_kv_mode.as_deref(),
                contract.sessions[idx].state_kinds,
                contract.sessions[idx].assistant_prefix,
                contract.sessions[idx].max_think_tokens,
                &state.conversation_tokens,
            );
            state.prefix_hash = Some(prefix_hash.clone());
            sessions.push(Qwen35PrefillSessionResult {
                id: id.clone(),
                prefill_tokens: contract.sessions[idx].tokens.len(),
                logical_position,
                cached_prefix_tokens: contract.sessions[idx].cached_prefix_tokens,
                prefix_hash,
                debug_sample_token: None,
                boundary_checkpoints: std::mem::take(&mut boundary_checkpoints_by_session[idx]),
            });
            m.q35_sessions.insert(id, state);
        }
        return Ok(Qwen35PrefillBatchResult {
            mode: "qwen35_fused_dense_prefill_boundary_chunked",
            plan,
            backend,
            total_prefill_tokens: shape_total_tokens,
            sessions,
        });
    }

    let worker_result = {
        let scratch = m.q35_scratch.as_mut().ok_or_else(|| {
            "qwen35 scratch missing; fused dense prefill is pp=1 only".to_string()
        })?;
        let needs_scratch = scratch
            .prefill_batch
            .as_ref()
            .map(|pbs| pbs.max_batch < contract.total_tokens)
            .unwrap_or(true);
        if needs_scratch {
            if let Some(existing) = scratch.prefill_batch.take() {
                existing.free_gpu(gpu);
            }
            let max_batch = std::env::var("HIPFIRE_PREFILL_MAX_BATCH")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| v >= 2)
                .unwrap_or(qwen35::PREFILL_MAX_BATCH)
                .max(contract.total_tokens);
            scratch.prefill_batch = Some(
                qwen35::PrefillBatchScratch::new(gpu, config, max_batch)
                    .map_err(|e| format!("allocate qwen35 fused dense prefill scratch: {e:?}"))?,
            );
        }
        let pbs_max_batch = scratch.prefill_batch.as_ref().unwrap().max_batch;
        if pbs_max_batch < contract.total_tokens {
            return Err(format!(
                "qwen35 fused dense prefill scratch max_batch={pbs_max_batch} is smaller than required fused rows {}; increase HIPFIRE_PREFILL_MAX_BATCH or restart the daemon",
                contract.total_tokens,
            ));
        }
        let pbs = scratch.prefill_batch.as_ref().unwrap();
        let mut rows: Vec<qwen35::DensePrefillSessionBatchRow<'_>> = owned_sessions
            .iter_mut()
            .zip(contract.sessions.iter())
            .map(|((_, state), spec)| qwen35::DensePrefillSessionBatchRow {
                tokens: spec.tokens,
                start_pos: state.seq_pos,
                kv_cache: &mut state.kv_cache,
                dn_state: &mut state.dn_state,
                logits: &state.logits,
            })
            .collect();
        qwen35::forward_prefill_dense_session_batch(gpu, weights, config, &mut rows, scratch, pbs)
    };

    let shape = match worker_result {
        Ok(shape) => shape,
        Err(e) => {
            for (id, state) in owned_sessions {
                m.q35_sessions.insert(id, state);
            }
            return Err(format!(
                "qwen35 fused dense prefill-session batch backend failed: {e:?}; \
                 use HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=auto or serial"
            ));
        }
    };

    let mut sessions = Vec::with_capacity(owned_sessions.len());
    for ((id, mut state), spec) in owned_sessions.into_iter().zip(contract.sessions.iter()) {
        state.seq_pos += spec.tokens.len();
        state.conversation_tokens.extend_from_slice(spec.tokens);
        state.prefilled_generated_suffix_len = if spec.replay_as_generated_suffix {
            spec.tokens.len()
        } else {
            0
        };
        let logical_position = state.seq_pos + state.kv_cache.compact_offset;
        let debug_sample_token = if spec.replay_as_generated_suffix
            && std::env::var_os("HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE").is_some()
        {
            let scratch = m.q35_scratch.as_ref().ok_or_else(|| {
                "qwen35 scratch missing; fused dense debug sampling unavailable".to_string()
            })?;
            let mut rng_state = 0x13579BDFu32;
            let cfg = SamplerConfig {
                temperature: 0.0,
                top_p: 1.0,
                repeat_window: 0,
                repeat_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                blocked_tokens: Vec::new(),
            };
            Some(sampler::sample(
                gpu,
                &state.logits,
                &scratch.sample_buf,
                &scratch.repeat_buf,
                config.vocab_size,
                spec.tokens,
                &cfg,
                &mut rng_state,
            ))
        } else {
            None
        };
        let prefix_hash = compute_qwen35_prefix_hash(
            m.arch_id,
            m.q35_kv_mode.as_deref(),
            spec.state_kinds,
            spec.assistant_prefix,
            spec.max_think_tokens,
            &state.conversation_tokens,
        );
        state.prefix_hash = Some(prefix_hash.clone());
        sessions.push(Qwen35PrefillSessionResult {
            id: id.clone(),
            prefill_tokens: spec.tokens.len(),
            logical_position,
            cached_prefix_tokens: spec.cached_prefix_tokens,
            prefix_hash,
            debug_sample_token,
            boundary_checkpoints: Vec::new(),
        });
        m.q35_sessions.insert(id, state);
    }

    Ok(Qwen35PrefillBatchResult {
        mode: match contract.input_kind {
            Qwen35FusedDensePrefillInputKind::FullPrompt => "qwen35_fused_dense_prefill",
            Qwen35FusedDensePrefillInputKind::GeneratedSuffixReplay => {
                "qwen35_fused_dense_suffix_replay"
            }
        },
        plan,
        backend,
        total_prefill_tokens: shape.total_tokens,
        sessions,
    })
}

/// Serial reference suffix prefill: one token at a time via the per-token
/// forward — the correctness baseline the batched kernels are checked against
/// and the fallback when no fused kernel applies.
pub fn qwen35_prefill_suffix_batch_serial_reference(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    batch_id: &str,
    prepared: &[Qwen35PreparedPrefillSession],
    plan: GenerateBatchPrefillPlan,
    backend: Qwen35PrefillBatchBackend,
) -> Result<Qwen35PrefillBatchResult, String> {
    // Reference implementation: exact serial activation/prefill over isolated
    // per-session KV + DeltaNet state. This is the correctness oracle for the
    // future fused dense-Qwen35 path. Do not replace this with concatenating
    // sessions into one `forward_prefill_batch` call: that would share one
    // DeltaNet recurrent state and one causal sequence across independent
    // requests.
    let mut total_prefill_tokens = 0usize;
    let mut results = Vec::with_capacity(prepared.len());
    for session in prepared {
        qwen35_activate_session(m, gpu, &session.id)?;
        let mut boundary_checkpoints = Vec::new();
        let prefilled = if session.boundary_checkpoints.is_empty()
            || session.replay_as_generated_suffix
        {
            qwen35_prefill_active_session(
                m,
                gpu,
                &session.tokens,
                session.replay_as_generated_suffix,
            )?
        } else {
            let mut prefilled = 0usize;
            let mut boundaries = session.boundary_checkpoints.clone();
            boundaries.sort_by_key(|boundary| boundary.prefix_len);
            for mut boundary in boundaries {
                if boundary.prefix_len <= prefilled || boundary.prefix_len > session.tokens.len() {
                    continue;
                }
                let segment = &session.tokens[prefilled..boundary.prefix_len];
                prefilled += qwen35_prefill_active_session(m, gpu, segment, false)?;
                let logical_position = qwen35_active_logical_position(m)?;
                if logical_position != boundary.prefix_len {
                    return Err(format!(
                        "qwen35 semantic boundary checkpoint position mismatch for session {}: boundary_len={} logical_position={}",
                        session.id, boundary.prefix_len, logical_position
                    ));
                }
                let hook = Qwen35PrefillCheckpointHook {
                    batch_id,
                    session_id: &session.id,
                    source_state_handle: &session.id,
                    logical_position,
                    kind: Qwen35PrefillCheckpointKind::SemanticBoundary {
                        boundary: &boundary.boundary,
                        boundary_index: boundary.boundary_index,
                    },
                    prefix_hash: &boundary.hash,
                };
                let checkpoint_id_for_error = qwen35_prefill_checkpoint_session_id(hook);
                let checkpoint_id = emit_qwen35_prefill_checkpoint(
                    m,
                    gpu,
                    loaded_model_state_arena_backend(m),
                    hook,
                )
                .map_err(|e| {
                    format!(
                        "qwen35 session {} failed to create semantic boundary checkpoint {}: {}",
                        session.id, checkpoint_id_for_error, e
                    )
                })?;
                qwen35_activate_session(m, gpu, &session.id)?;
                boundary.checkpoint_id = Some(checkpoint_id);
                boundary_checkpoints.push(boundary);
            }
            if prefilled < session.tokens.len() {
                prefilled +=
                    qwen35_prefill_active_session(m, gpu, &session.tokens[prefilled..], false)?;
            }
            prefilled
        };
        let logical_position = qwen35_active_logical_position(m)?;
        let debug_sample_token = if session.replay_as_generated_suffix
            && std::env::var_os("HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE").is_some()
        {
            let config = m
                .q35_config
                .as_ref()
                .ok_or_else(|| "qwen35 config missing".to_string())?;
            let scratch = m.q35_scratch.as_ref().ok_or_else(|| {
                "qwen35 scratch missing; PP batch-prefill is not supported".to_string()
            })?;
            let mut rng_state = 0x13579BDFu32;
            let cfg = SamplerConfig {
                temperature: 0.0,
                top_p: 1.0,
                repeat_window: 0,
                repeat_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                blocked_tokens: Vec::new(),
            };
            Some(sampler::sample(
                gpu,
                &scratch.logits,
                &scratch.sample_buf,
                &scratch.repeat_buf,
                config.vocab_size,
                &session.tokens,
                &cfg,
                &mut rng_state,
            ))
        } else {
            None
        };
        qwen35_save_active_session(m, gpu)?;
        let prefix_hash = {
            let saved = m.q35_sessions.get(&session.id).ok_or_else(|| {
                format!("qwen35 session {} missing after prefill save", session.id)
            })?;
            compute_qwen35_prefix_hash(
                m.arch_id,
                m.q35_kv_mode.as_deref(),
                &session.state_kinds,
                &session.assistant_prefix,
                session.max_think_tokens,
                &saved.conversation_tokens,
            )
        };
        if let Some(saved) = m.q35_sessions.get_mut(&session.id) {
            saved.prefix_hash = Some(prefix_hash.clone());
        }
        total_prefill_tokens += prefilled;
        results.push(Qwen35PrefillSessionResult {
            id: session.id.clone(),
            prefill_tokens: prefilled,
            logical_position,
            cached_prefix_tokens: session.cached_prefix_tokens,
            prefix_hash,
            debug_sample_token,
            boundary_checkpoints,
        });
    }

    Ok(Qwen35PrefillBatchResult {
        mode: "serial_prefill",
        plan,
        backend,
        total_prefill_tokens,
        sessions: results,
    })
}

/// Drive a multi-session `generate_batch_prefill` request serially: materialize
/// each session's prompt, prefill it, save its state, and emit per-session +
/// batch done events. The non-fused multi-session prefill entry point.
pub fn run_generate_batch_prefill_serial_qwen35(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    envelope: &GenerateBatchPrefillEnvelope,
    pflash_active: bool,
) -> Result<(), String> {
    if !is_qwen35_family_arch_id(m.arch_id) {
        return Err(format!(
            "generate_batch_prefill currently supports qwen35/qwen35-moe only (arch_id={})",
            m.arch_id
        ));
    }
    if m.pp > 1 {
        return Err(
            "generate_batch_prefill does not support pipeline-parallel models yet".to_string(),
        );
    }
    if m.dflash.is_some() {
        return Err("generate_batch_prefill does not support DFlash-loaded models yet".to_string());
    }
    if m.eviction.is_some() {
        return Err(
            "generate_batch_prefill does not support CASK/TriAttention eviction yet".to_string(),
        );
    }
    if pflash_active {
        return Err("generate_batch_prefill does not support PFlash compression yet".to_string());
    }
    let arena_backend = loaded_model_state_arena_backend(m);

    let plan = plan_generate_batch_prefill_qwen35(m.arch_id, envelope.session_count);
    let requested_backend = std::env::var("HIPFIRE_QWEN35_PREFILL_SESSION_BATCH").ok();
    let fused_grouped_moe_supported =
        validate_qwen35_fused_grouped_moe_prefill_model_capability(m, envelope.session_count);
    let backend = select_qwen35_prefill_batch_backend(
        plan,
        requested_backend.as_deref(),
        fused_grouped_moe_supported,
    )?;
    let started = serde_json::json!({
        "type": "generate_batch_prefill_started",
        "id": envelope.id,
        "batch_id": envelope.batch_id,
        "sessions": envelope.session_count,
        "mode": "serial_prefill",
        "plan": plan.as_str(),
        "backend": backend.as_str(),
    });
    let _ = writeln!(stdout, "{started}");
    let _ = stdout.flush();

    let t0 = Instant::now();
    let mut prepared = Vec::with_capacity(envelope.sessions.len());
    for session in &envelope.sessions {
        if !generate_state_kinds_include_required(
            &session.state_handle.state_kinds,
            SequenceStatePageKind::Kv,
        ) {
            return Err(format!(
                "generate_batch_prefill session {} missing attention_kv state kind",
                session.id
            ));
        }
        if !generate_state_kinds_include_required(
            &session.state_handle.state_kinds,
            SequenceStatePageKind::DeltaNet,
        ) {
            return Err(format!(
                "generate_batch_prefill session {} missing deltanet_recurrent state kind",
                session.id
            ));
        }

        if let Some(runtime_state_handle) = session.state_handle.runtime_state_handle.as_deref() {
            sequence_state_arena_fork_session_state(
                arena_backend,
                m,
                gpu,
                SequenceStateForkRequest {
                    source_session_id: runtime_state_handle,
                    dest_session_id: &session.id,
                    requested_prefix_hash: session.state_handle.prefix_hash.as_ref(),
                },
            )
            .map_err(|e| {
                format!(
                    "generate_batch_prefill session {} failed to attach checkpoint {}: {}",
                    session.id, runtime_state_handle, e
                )
            })?;
        }

        let resident = sequence_state_arena_is_session_resident(arena_backend, m, &session.id);
        if !resident
            && (session.state_handle.logical_position > 0
                || session.state_handle.cached_prefix_tokens > 0)
        {
            return Err(format!(
                "generate_batch_prefill session {} references cached state at logical_position={} cached_prefix_tokens={} but no resident session exists",
                session.id,
                session.state_handle.logical_position,
                session.state_handle.cached_prefix_tokens
            ));
        }

        let created = sequence_state_arena_activate_session(arena_backend, m, gpu, &session.id)?;
        let mut boundary_checkpoints = Vec::new();
        let tokens: Vec<u32> = if session.prompt.is_some() {
            let full_tokens = qwen35_materialize_batch_prefill_prompt(m, session)?;
            if session.state_handle.runtime_state_handle.is_some() {
                let prefix_len = session
                    .state_handle
                    .prefix_hash
                    .as_ref()
                    .map(|hash| hash.prefix_len)
                    .unwrap_or(session.state_handle.cached_prefix_tokens);
                if prefix_len > full_tokens.len() {
                    return Err(format!(
                        "generate_batch_prefill prompt session {} cached prefix length {} exceeds rendered token length {}",
                        session.id,
                        prefix_len,
                        full_tokens.len()
                    ));
                }
                full_tokens[prefix_len..].to_vec()
            } else if session.state_handle.logical_position != 0
                || session.state_handle.cached_prefix_tokens != 0
            {
                return Err(format!(
                    "generate_batch_prefill prompt session {} must start at logical_position=0 cached_prefix_tokens=0 in the first slice",
                    session.id
                ));
            } else {
                let _ = created;
                sequence_state_arena_reset_active_session(arena_backend, m, gpu)?;
                boundary_checkpoints =
                    qwen35_semantic_boundary_checkpoints(m, session, &full_tokens)?;
                full_tokens
            }
        } else {
            let current_position = sequence_state_arena_active_logical_position(arena_backend, m)?;
            if created && session.state_handle.logical_position != 0 {
                return Err(format!(
                    "generate_batch_prefill suffix session {} is new but logical_position={} (expected 0)",
                    session.id, session.state_handle.logical_position
                ));
            }
            if !created && current_position != session.state_handle.logical_position {
                return Err(format!(
                    "generate_batch_prefill session {} logical_position mismatch: request={} resident={}",
                    session.id, session.state_handle.logical_position, current_position
                ));
            }
            session.suffix_tokens.clone().unwrap_or_default()
        };

        prepared.push(Qwen35PreparedPrefillSession {
            id: session.id.clone(),
            tokens,
            cached_prefix_tokens: session.state_handle.cached_prefix_tokens,
            replay_as_generated_suffix: session.suffix_tokens.is_some(),
            state_kinds: session.state_handle.state_kinds.clone(),
            assistant_prefix: session.assistant_prefix.clone(),
            max_think_tokens: session.max_think_tokens,
            boundary_checkpoints,
        });
    }

    let result = qwen35_prefill_suffix_batch(m, gpu, &envelope.batch_id, &prepared, plan, backend)?;
    for session in &result.sessions {
        let hook = Qwen35PrefillCheckpointHook {
            batch_id: &envelope.batch_id,
            session_id: &session.id,
            source_state_handle: &session.id,
            logical_position: session.logical_position,
            kind: Qwen35PrefillCheckpointKind::Final,
            prefix_hash: &session.prefix_hash,
        };
        let checkpoint_id_for_error = qwen35_prefill_checkpoint_session_id(hook);
        let checkpoint_id =
            emit_qwen35_prefill_checkpoint(m, gpu, arena_backend, hook).map_err(|e| {
                format!(
                    "generate_batch_prefill session {} failed to create checkpoint {}: {}",
                    session.id, checkpoint_id_for_error, e
                )
            })?;
        let line = qwen35_generate_batch_prefill_session_done_json(
            envelope,
            session,
            &checkpoint_id,
            &result,
        );
        let _ = writeln!(stdout, "{line}");
        let _ = stdout.flush();
    }

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let worker = loaded_model_worker_runtime_view(m);
    let done = qwen35_generate_batch_prefill_done_json(
        envelope,
        &result,
        elapsed_ms,
        sequence_state_arena_resident_session_count(arena_backend, m),
        model_worker_runtime_view_json(&worker),
    );
    let _ = writeln!(stdout, "{done}");
    let _ = stdout.flush();
    Ok(())
}
