// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM2 generate_batch_prefill support.
//!
//! This is the correctness-first resident-session path: each request session
//! owns an isolated `Lfm2MoeState`, and each prompt/suffix is run through the
//! arch-local `prefill_batch` for that one session. A future fused worker can
//! batch across sessions behind the same protocol surface.

use std::io::Write;
use std::time::Instant;

use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_generate::{
    compute_qwen35_prefix_hash, generate_prefix_hash_json, prefix_hash_preflight_done_json,
    GenerateBatchPrefillEnvelope, GenerateBatchPrefillSession, PrefixHashPreflightCandidate,
    PrefixHashPreflightEnvelope, Qwen35SemanticBoundaryCheckpoint,
};
use hipfire_model::ARCH_ID_LFM2_MOE;
use hipfire_state::{
    generate_state_kinds_include_required, model_worker_runtime_view_json,
    SequenceStateCheckpointRequest, SequenceStateForkRequest, SequenceStatePageKind,
};

use crate::model::LoadedModel;
use crate::qwen35_prefill::qwen35_materialize_batch_prefill_prompt;
use crate::session::{
    lfm2_activate_session, lfm2_active_logical_position, lfm2_checkpoint_session_state,
    lfm2_fork_session_state, lfm2_release_sessions, lfm2_request_session_count,
    lfm2_reset_active_session, lfm2_save_active_session, loaded_model_worker_runtime_view,
    LFM2_LEGACY_SESSION_ID,
};

const LFM2_PREFIX_HASH_KV_MODE: &str = "lfm2_q8_kv";

#[derive(Clone, Debug)]
struct Lfm2PrefillSessionResult {
    id: String,
    prefill_tokens: usize,
    logical_position: usize,
    cached_prefix_tokens: usize,
    prefix_hash: hipfire_state::SequenceStatePrefixHash,
    allocation_epoch: u64,
    boundary_checkpoints: Vec<Qwen35SemanticBoundaryCheckpoint>,
}

fn lfm2_prefill_session_done_json(
    envelope: &GenerateBatchPrefillEnvelope,
    session: &Lfm2PrefillSessionResult,
) -> serde_json::Value {
    let mut line = serde_json::json!({
        "type": "generate_batch_prefill_session_done",
        "id": envelope.id,
        "batch_id": envelope.batch_id,
        "session_id": session.id,
        "prefill_tokens": session.prefill_tokens,
        "logical_position": session.logical_position,
        "cached_prefix_tokens": session.cached_prefix_tokens,
        "state_handle": {
            "id": session.id,
            "kind": "lfm2_session",
            "runtime_state": "resident",
            "runtime_state_handle": session.id,
            "session_id": session.id,
            "checkpoint_runtime_state": "resident_only",
            "logical_position": session.logical_position,
            "cached_prefix_tokens": session.cached_prefix_tokens,
            "prefix_hash": generate_prefix_hash_json(&session.prefix_hash),
            "prefix_len": session.prefix_hash.prefix_len,
            "allocation_epoch": session.allocation_epoch,
        },
        "mode": "lfm2_serial_prefill_batch",
        "plan": "serial_exact",
        "backend": "lfm2_arch_prefill_batch",
    });
    let prefix_checkpoints = session
        .boundary_checkpoints
        .iter()
        .filter_map(|checkpoint| {
            checkpoint.checkpoint_id.as_ref().map(|checkpoint_id| {
                serde_json::json!({
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_runtime_state": "attachable",
                    "runtime_state": "resident",
                    "runtime_state_handle": checkpoint_id,
                    "logical_position": checkpoint.prefix_len,
                    "cached_prefix_tokens": checkpoint.prefix_len,
                    "prefix_hash": generate_prefix_hash_json(&checkpoint.hash),
                    "prefix_len": checkpoint.hash.prefix_len,
                    "boundary": checkpoint.boundary,
                    "boundary_index": checkpoint.boundary_index,
                })
            })
        })
        .collect::<Vec<_>>();
    if !prefix_checkpoints.is_empty() {
        line["state_handle"]["prefix_checkpoints"] = serde_json::json!(prefix_checkpoints);
    }
    line
}

pub fn emit_lfm2_generate_batch_prefill_ready(
    stdout: &mut std::io::Stdout,
    envelope: &GenerateBatchPrefillEnvelope,
) {
    let line = serde_json::json!({
        "type": "generate_batch_prefill_ready",
        "id": envelope.id,
        "batch_id": envelope.batch_id,
        "sessions": envelope.session_count,
        "supported": true,
        "mode": "lfm2_serial_prefill_batch",
        "reason": "lfm2_arch_prefill_batch_available",
    });
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
}

fn compute_lfm2_prefix_hash(
    m: &LoadedModel,
    session: &GenerateBatchPrefillSession,
    tokens: &[u32],
) -> hipfire_state::SequenceStatePrefixHash {
    compute_qwen35_prefix_hash(
        m.arch_id,
        Some(LFM2_PREFIX_HASH_KV_MODE),
        &session.state_handle.state_kinds,
        &session.assistant_prefix,
        session.max_think_tokens,
        tokens,
    )
}

pub fn lfm2_prefix_hash_candidates_for_tokens(
    m: &LoadedModel,
    session: &GenerateBatchPrefillSession,
    full_tokens: &[u32],
) -> Result<Vec<PrefixHashPreflightCandidate>, String> {
    let tokenizer = m
        .tokenizer
        .as_ref()
        .ok_or_else(|| "tokenizer not loaded".to_string())?;
    let full_hash = compute_lfm2_prefix_hash(m, session, full_tokens);
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
        let hash = compute_lfm2_prefix_hash(m, session, &full_tokens[..prefix_len]);
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

pub fn lfm2_prefix_hash_candidates(
    m: &LoadedModel,
    session: &GenerateBatchPrefillSession,
) -> Result<Vec<PrefixHashPreflightCandidate>, String> {
    let full_tokens = qwen35_materialize_batch_prefill_prompt(m, session)?;
    lfm2_prefix_hash_candidates_for_tokens(m, session, &full_tokens)
}

pub fn lfm2_semantic_boundary_checkpoints(
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
    let candidates = lfm2_prefix_hash_candidates_for_tokens(m, session, full_tokens)?;
    if std::env::var_os("HIPFIRE_DEBUG_PREFIX_BOUNDARIES").is_some() {
        eprintln!(
            "[daemon] lfm2 prefix boundary candidates session={} tokens={} candidates={}",
            session.id,
            full_tokens.len(),
            candidates.len()
        );
        for candidate in &candidates {
            eprintln!(
                "[daemon] lfm2 prefix boundary candidate session={} boundary={} index={} len={} hash={}",
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

pub fn run_prefix_hash_preflight_lfm2(
    m: &LoadedModel,
    stdout: &mut std::io::Stdout,
    envelope: &PrefixHashPreflightEnvelope,
) -> Result<(), String> {
    if m.arch_id != ARCH_ID_LFM2_MOE {
        return Err(format!(
            "lfm2 prefix_hash_preflight requires arch_id=11, got {}",
            m.arch_id
        ));
    }
    if m.pp > 1 {
        return Err(
            "lfm2 prefix_hash_preflight does not support pipeline-parallel models yet".to_string(),
        );
    }
    if envelope.boundary_policy != "semantic_chat_template" {
        return Err(
            "prefix_hash_preflight.boundary_policy must be semantic_chat_template".to_string(),
        );
    }
    let candidates = lfm2_prefix_hash_candidates(m, &envelope.session)?;
    let line =
        prefix_hash_preflight_done_json(&envelope.id, &envelope.boundary_policy, &candidates)?;
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
    Ok(())
}

fn lfm2_boundary_checkpoint_session_id(
    batch_id: &str,
    session_id: &str,
    logical_position: usize,
    boundary_index: usize,
) -> String {
    format!("lfm2-checkpoint:{batch_id}:{session_id}:boundary:{boundary_index}:{logical_position}")
}

fn lfm2_materialize_prefill_tokens(
    m: &LoadedModel,
    session: &GenerateBatchPrefillSession,
    created: bool,
    current_position: usize,
) -> Result<(Vec<u32>, Vec<Qwen35SemanticBoundaryCheckpoint>), String> {
    if let Some(prompt) = session.prompt.as_ref() {
        let _ = prompt;
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
                    "lfm2 generate_batch_prefill prompt session {} cached prefix length {} exceeds rendered token length {}",
                    session.id,
                    prefix_len,
                    full_tokens.len()
                ));
            }
            Ok((full_tokens[prefix_len..].to_vec(), Vec::new()))
        } else if session.state_handle.logical_position != 0
            || session.state_handle.cached_prefix_tokens != 0
        {
            return Err(format!(
                "lfm2 generate_batch_prefill prompt session {} must start at logical_position=0 cached_prefix_tokens=0",
                session.id
            ));
        } else {
            let boundary_checkpoints =
                lfm2_semantic_boundary_checkpoints(m, session, &full_tokens)?;
            Ok((full_tokens, boundary_checkpoints))
        }
    } else {
        if created && session.state_handle.logical_position != 0 {
            return Err(format!(
                "lfm2 generate_batch_prefill suffix session {} is new but logical_position={} (expected 0)",
                session.id, session.state_handle.logical_position
            ));
        }
        if !created && current_position != session.state_handle.logical_position {
            return Err(format!(
                "lfm2 generate_batch_prefill session {} logical_position mismatch: request={} resident={}",
                session.id, session.state_handle.logical_position, current_position
            ));
        }
        Ok((
            session.suffix_tokens.clone().unwrap_or_default(),
            Vec::new(),
        ))
    }
}

fn lfm2_prefill_active_session_tokens(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    tokens: &[u32],
) -> Result<usize, String> {
    if tokens.is_empty() {
        return Ok(0);
    }
    let (current_position, capacity, capacity_label) = {
        let state = m
            .lfm2moe_state
            .as_ref()
            .ok_or_else(|| "lfm2 active session missing state".to_string())?;
        if m.eviction.is_some() {
            (
                state.n_tokens + state.kv.compact_offset,
                state.max_seq,
                "max_seq",
            )
        } else {
            (state.n_tokens, m.physical_cap, "physical_cap")
        }
    };
    if current_position + tokens.len() > capacity {
        return Err(format!(
            "lfm2 generate_batch_prefill exceeds loaded KV budget: logical_position={} + prefill={} > {}={}",
            current_position,
            tokens.len(),
            capacity_label,
            capacity
        ));
    }
    if let Some(df) = m.lfm2_dflash.as_ref() {
        let dflash_capacity = df.ctx_capacity;
        if current_position + tokens.len() > dflash_capacity {
            return Err(format!(
                "lfm2 generate_batch_prefill exceeds DFlash context: logical_position={} + prefill={} > ctx_capacity={}",
                current_position,
                tokens.len(),
                dflash_capacity
            ));
        }
    }
    let cfg = m
        .lfm2moe_config
        .as_ref()
        .ok_or_else(|| "lfm2 config missing".to_string())?;
    let weights = m
        .lfm2moe_weights
        .as_ref()
        .ok_or_else(|| "lfm2 weights missing".to_string())?;
    let eviction = m.eviction.as_ref();
    let state = m
        .lfm2moe_state
        .as_mut()
        .ok_or_else(|| "lfm2 active session missing state".to_string())?;
    if let Some(eviction) = eviction {
        for &token in tokens {
            let position = state.n_tokens as u32;
            lfm2moe::forward::decode_step(cfg, weights, state, gpu, token, position).map_err(
                |e| format!("lfm2 generate_batch_prefill eviction prefill failed: {e:?}"),
            )?;
            match eviction.maybe_evict(gpu, &mut state.kv, state.n_tokens) {
                Ok(Some(hipfire_runtime::triattn::EvictionResult { new_physical, .. })) => {
                    state.n_tokens = new_physical;
                }
                Ok(None) => {}
                Err(e) => {
                    return Err(format!(
                        "lfm2 generate_batch_prefill eviction failed: {e:?}"
                    ));
                }
            }
        }
        m.seq_pos = state.n_tokens + state.kv.compact_offset;
    } else if let Some(df) = m.lfm2_dflash.as_mut() {
        let mut capture = lfm2moe::forward::Lfm2HiddenCapture::new(
            cfg.num_hidden_layers,
            cfg.hidden_size,
            df.draft_config.target_layer_ids.clone(),
        )?;
        lfm2moe::forward::prefill_batch_with_hidden_logits(
            cfg,
            weights,
            state,
            gpu,
            tokens,
            &mut capture,
        )
        .map_err(|e| format!("lfm2 generate_batch_prefill DFlash prefill failed: {e:?}"))?;
        df.target_hidden_host
            .extend_from_slice(&capture.take_rows());
        m.seq_pos = state.n_tokens;
    } else {
        lfm2moe::forward::prefill_batch(cfg, weights, state, gpu, tokens)
            .map_err(|e| format!("lfm2 generate_batch_prefill failed: {e:?}"))?;
        m.seq_pos = state.n_tokens;
    }
    m.conversation_tokens.extend_from_slice(tokens);
    Ok(tokens.len())
}

fn lfm2_prefill_with_boundary_checkpoints(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    batch_id: &str,
    session: &GenerateBatchPrefillSession,
    tokens: &[u32],
    boundary_checkpoints: &mut [Qwen35SemanticBoundaryCheckpoint],
) -> Result<usize, String> {
    if boundary_checkpoints.is_empty() {
        return lfm2_prefill_active_session_tokens(m, gpu, tokens);
    }
    boundary_checkpoints.sort_by_key(|checkpoint| checkpoint.prefix_len);
    let mut prefilled = 0usize;
    for checkpoint in boundary_checkpoints.iter_mut() {
        let prefix_len = checkpoint.prefix_len;
        if prefix_len <= prefilled || prefix_len > tokens.len() {
            continue;
        }
        lfm2_prefill_active_session_tokens(m, gpu, &tokens[prefilled..prefix_len])?;
        prefilled = prefix_len;
        let logical_position = lfm2_active_logical_position(m)?;
        if logical_position != checkpoint.prefix_len {
            return Err(format!(
                "lfm2 generate_batch_prefill session {} boundary logical_position mismatch: expected={} resident={}",
                session.id,
                checkpoint.prefix_len,
                logical_position
            ));
        }
        let checkpoint_id = lfm2_boundary_checkpoint_session_id(
            batch_id,
            &session.id,
            logical_position,
            checkpoint.boundary_index,
        );
        lfm2_checkpoint_session_state(
            m,
            gpu,
            SequenceStateCheckpointRequest {
                source_session_id: &session.id,
                dest_session_id: &checkpoint_id,
                expected_logical_position: logical_position,
                requested_prefix_hash: None,
                checkpoint_prefix_hash: Some(&checkpoint.hash),
            },
        )
        .map_err(|e| {
            format!(
                "lfm2 generate_batch_prefill session {} failed to create boundary checkpoint {}: {}",
                session.id, checkpoint_id, e
            )
        })?;
        lfm2_activate_session(m, gpu, &session.id)?;
        checkpoint.checkpoint_id = Some(checkpoint_id);
    }
    if prefilled < tokens.len() {
        lfm2_prefill_active_session_tokens(m, gpu, &tokens[prefilled..])?;
    }
    Ok(tokens.len())
}

pub fn run_generate_batch_prefill_serial_lfm2(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    envelope: &GenerateBatchPrefillEnvelope,
) -> Result<(), String> {
    if m.arch_id != ARCH_ID_LFM2_MOE {
        return Err(format!(
            "lfm2 generate_batch_prefill requires arch_id=11, got {}",
            m.arch_id
        ));
    }
    if m.pp > 1 {
        return Err(
            "lfm2 generate_batch_prefill does not support pipeline-parallel models yet".to_string(),
        );
    }
    if m.lfm2_dflash.is_some() && m.eviction.is_some() {
        return Err(
            "lfm2 generate_batch_prefill does not support DFlash with CASK/TriAttention eviction yet"
                .to_string(),
        );
    }
    for session in &envelope.sessions {
        if !generate_state_kinds_include_required(
            &session.state_handle.state_kinds,
            SequenceStatePageKind::Kv,
        ) {
            return Err(format!(
                "lfm2 generate_batch_prefill session {} missing attention_kv state kind",
                session.id
            ));
        }
    }

    let started = serde_json::json!({
        "type": "generate_batch_prefill_started",
        "id": envelope.id,
        "batch_id": envelope.batch_id,
        "sessions": envelope.session_count,
        "mode": "lfm2_serial_prefill_batch",
        "plan": "serial_exact",
        "backend": "lfm2_arch_prefill_batch",
    });
    let _ = writeln!(stdout, "{started}");
    let _ = stdout.flush();

    let t0 = Instant::now();
    let mut results = Vec::with_capacity(envelope.sessions.len());
    for session in &envelope.sessions {
        if let Some(runtime_state_handle) = session.state_handle.runtime_state_handle.as_deref() {
            lfm2_fork_session_state(
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
                    "lfm2 generate_batch_prefill session {} failed to attach checkpoint {}: {}",
                    session.id, runtime_state_handle, e
                )
            })?;
        }
        let existed = crate::session::lfm2_session_resident(m, &session.id);
        let created = lfm2_activate_session(m, gpu, &session.id)?;
        let current_position = lfm2_active_logical_position(m)?;
        let (tokens, mut boundary_checkpoints) =
            lfm2_materialize_prefill_tokens(m, session, created, current_position)?;
        if session.prompt.is_some() && session.state_handle.runtime_state_handle.is_none() {
            lfm2_reset_active_session(m, gpu)?;
        }
        lfm2_prefill_with_boundary_checkpoints(
            m,
            gpu,
            &envelope.batch_id,
            session,
            &tokens,
            &mut boundary_checkpoints,
        )?;
        let logical_position = lfm2_active_logical_position(m)?;
        let prefix_hash = compute_lfm2_prefix_hash(m, session, &m.conversation_tokens);
        let result = Lfm2PrefillSessionResult {
            id: session.id.clone(),
            prefill_tokens: tokens.len(),
            logical_position,
            cached_prefix_tokens: session
                .state_handle
                .prefix_hash
                .as_ref()
                .map(|hash| hash.prefix_len)
                .unwrap_or(session.state_handle.cached_prefix_tokens),
            prefix_hash: prefix_hash.clone(),
            allocation_epoch: m.lfm2_registry.allocation_epoch,
            boundary_checkpoints,
        };
        lfm2_save_active_session(m)?;
        if let Some(saved) = m.lfm2_registry.sessions.get_mut(&session.id) {
            saved.prefix_hash = Some(prefix_hash);
        } else if existed {
            return Err(format!(
                "lfm2 generate_batch_prefill session {} disappeared after save",
                session.id
            ));
        }
        results.push(result);
    }

    let created_legacy = lfm2_activate_session(m, gpu, LFM2_LEGACY_SESSION_ID)?;
    if created_legacy {
        lfm2_reset_active_session(m, gpu)?;
    }

    for session in &results {
        let line = lfm2_prefill_session_done_json(envelope, session);
        let _ = writeln!(stdout, "{line}");
        let _ = stdout.flush();
    }

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let worker = loaded_model_worker_runtime_view(m);
    let done = serde_json::json!({
        "type": "generate_batch_prefill_done",
        "id": envelope.id,
        "batch_id": envelope.batch_id,
        "sessions": envelope.session_count,
        "prefill_tokens": results.iter().map(|s| s.prefill_tokens).sum::<usize>(),
        "elapsed_ms": elapsed_ms,
        "mode": "lfm2_serial_prefill_batch",
        "plan": "serial_exact",
        "backend": "lfm2_arch_prefill_batch",
        "resident_sessions": lfm2_request_session_count(m),
        "model_worker": model_worker_runtime_view_json(&worker),
    });
    let _ = writeln!(stdout, "{done}");
    let _ = stdout.flush();
    Ok(())
}

#[allow(dead_code)]
pub fn release_lfm2_prefill_sessions(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    session_ids: &[String],
) -> Result<usize, String> {
    lfm2_release_sessions(m, gpu, session_ids)
}
